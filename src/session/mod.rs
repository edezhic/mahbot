//! Session persistence — Turso-backed store + native history decoding.

pub mod manager;
pub use manager::Session;

pub mod summarization;

use crate::turso::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall as ProviderToolCall};
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};

crate::define_store! {
    /// Global session store.
    pub static SESSIONS: SessionStore,
    db_name = "sessions",
    schema = SCHEMA,
    expect = "SESSIONS not initialized",
}

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_key_id ON sessions(session_key, id);

CREATE TABLE IF NOT EXISTS session_metadata (
    session_key   TEXT PRIMARY KEY,
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

// Session list with metadata (4-column SELECT: sm.session_key, sm.created_at,
// sm.last_activity, COUNT(s.id))
crate::columns! {
    SESSION_LIST_COLUMNS [SL] {
        SESSION_KEY    => "sm.session_key",
        CREATED_AT     => "sm.created_at",
        LAST_ACTIVITY  => "sm.last_activity",
        MESSAGE_COUNT  => "COUNT(s.id)",
    }
}

/// Session key prefixes for transient (background-only, non-user-facing) sessions.
///
/// These sessions are created automatically by agents (analysts, engineers, maintainer,
/// discovery, etc.) and are cleaned up periodically by
/// [`cleanup_old_transient_sessions`].
///
/// User-facing sessions — those the user can directly converse with — persist
/// indefinitely and are intentionally excluded:
/// - Direct chat: `{channel}_{user_name}_{role}_{ws_name}`
/// - Manager: `manager_{ws_name}` — the Manager session carries both chat conversation
///   and notification context and must never be added here.
///
/// If a new agent role is added that can talk to users directly, its session key
/// must also be excluded from this list.
pub(crate) const TRANSIENT_SESSION_PREFIXES: &[&str] =
    &["ticket_", "ask_", "maintainer_", "discovery_"];

#[derive(Debug, Clone)]
pub(crate) struct SessionMetadata {
    pub key: String,
    #[expect(dead_code)]
    pub created_at: DateTime<Utc>,
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

fn session_metadata_from_row(
    key: &str,
    created_str: &str,
    activity_str: &str,
    count: i64,
) -> SessionMetadata {
    SessionMetadata {
        key: key.to_string(),
        created_at: parse_ts_or_now(created_str, "created_at"),
        last_activity: parse_ts_or_now(activity_str, "last_activity"),
        message_count: usize::try_from(count).unwrap_or(0),
    }
}

/// Insert messages into `sessions` and upsert `session_metadata` within an existing transaction.
/// Shared helper used by [`SessionStore::append_messages`].
async fn insert_messages_in_transaction(
    tx: &TxGuard<'_>,
    session_key: &str,
    messages: &[ChatMessage],
) -> Result<()> {
    let now = turso::now();
    for msg in messages {
        tx.execute(
            "INSERT INTO sessions (session_key, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![
                session_key,
                msg.role.to_string(),
                msg.content.clone(),
                now.clone()
            ],
        )
        .await?;
    }
    tx.execute(
        "INSERT INTO session_metadata (session_key, created_at, last_activity) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(session_key) DO UPDATE SET \
         last_activity = excluded.last_activity",
        params![session_key, now.clone(), now],
    )
    .await?;
    Ok(())
}

/// Execute a `query_map`, logging warnings on failure and skipping unparseable rows.
/// Returns an empty [`Vec`] on query error.
///
/// `session_key` is passed as a structured tracing field; when `None`, tracing
/// automatically suppresses it from the output.
async fn query_map_collect<T, E>(
    conn: &turso::Connection,
    sql: &str,
    params: impl IntoParams + Send + 'static,
    row_parser: impl FnMut(&Row) -> std::result::Result<T, E> + Send + 'static,
    warn_context: &str,
    session_key: Option<&str>,
) -> Vec<T>
where
    T: Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let rows = match conn.query_map(sql, params, row_parser).await {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!(error = %e, session_key, "{warn_context}: query failed, returning empty");
            return Vec::new();
        }
    };
    rows.into_iter()
        .filter_map(|r| match r {
            Ok(val) => Some(val),
            Err(e) => {
                tracing::warn!(error = %e, session_key, "{warn_context}: row decode failed, skipping");
                None
            }
        })
        .collect()
}

// ── Methods — callable on the static ──────────────────────────

impl SessionStore {
    pub(crate) async fn load(&self, session_key: &str) -> Vec<ChatMessage> {
        query_map_collect(
            &self.conn,
            &format!("SELECT {SESSION_MESSAGE_COLUMNS} FROM sessions WHERE session_key = ?1 ORDER BY id ASC"),
            params![session_key],
            |row| {
                Ok::<_, anyhow::Error>(ChatMessage {
                    role: row.get::<String>(COL_SM_ROLE)?.parse().map_err(|e: String| anyhow!(e))?,
                    content: row.get(COL_SM_CONTENT)?,
                })
            },
            "load session",
            Some(session_key),
        )
        .await
    }

    pub(crate) async fn append(&self, session_key: &str, message: &ChatMessage) -> Result<()> {
        self.batch_append(session_key, std::slice::from_ref(message))
            .await
    }

    async fn append_messages(
        &self,
        session_key: &str,
        messages: &[ChatMessage],
        replace: bool,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        if replace {
            tx.execute(
                "DELETE FROM sessions WHERE session_key = ?1",
                params![session_key],
            )
            .await?;
        }
        insert_messages_in_transaction(&tx, session_key, messages).await?;
        tx.commit().await?;
        Ok(())
    }

    pub(crate) async fn batch_append(
        &self,
        session_key: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        self.append_messages(session_key, messages, false).await
    }

    pub(crate) async fn replace_messages(
        &self,
        session_key: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        self.append_messages(session_key, messages, true).await
    }

    pub(crate) async fn delete(&self, session_key: &str) -> Result<bool> {
        let tx = self.conn.begin_tx().await?;
        let deleted = tx
            .execute(
                "DELETE FROM sessions WHERE session_key = ?1",
                params![session_key],
            )
            .await?;
        tx.execute(
            "DELETE FROM session_metadata WHERE session_key = ?1",
            params![session_key],
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
                 LEFT JOIN sessions s ON s.session_key = sm.session_key \
                 GROUP BY sm.session_key \
                 ORDER BY sm.last_activity DESC",
            ),
            (),
            |row| {
                Ok::<_, anyhow::Error>(session_metadata_from_row(
                    &row.get::<String>(COL_SL_SESSION_KEY)?,
                    &row.get::<String>(COL_SL_CREATED_AT)?,
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

/// Delete all transient (background-only) sessions whose `last_activity` is older than
/// the given RFC 3339 `cutoff`. Returns the number of deleted session metadata rows.
///
/// Transient session keys start with the prefixes listed in
/// `TRANSIENT_SESSION_PREFIXES`.
///
/// Both `sessions` and `session_metadata` tables are cleaned up in a single transaction.
pub async fn cleanup_old_transient_sessions(cutoff: &str) -> Result<u64> {
    let session_store = store();
    let tx = session_store.conn.begin_tx().await?;

    let likes = TRANSIENT_SESSION_PREFIXES
        .iter()
        .map(|_| "session_key LIKE ?")
        .collect::<Vec<_>>()
        .join(" OR ");
    let prefix_patterns = format!("({likes})");

    let build_params = {
        let mut p = vec![Value::Text(cutoff.to_string())];
        p.extend(
            TRANSIENT_SESSION_PREFIXES
                .iter()
                .map(|prefix| Value::Text(format!("{prefix}%"))),
        );
        p
    };

    // Delete session messages for matching transient sessions
    tx.execute(
        &format!(
            "DELETE FROM sessions WHERE session_key IN ( \
             SELECT session_key FROM session_metadata \
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

/// Construct a session key for direct (non-ticket) user ↔ agent conversations.
///
/// Format: `{channel}_{user_name}_{role}_{ws_name}`
#[must_use]
pub fn direct_session_key(channel: &str, user_name: &str, role: &str, ws_name: &str) -> String {
    format!("{channel}_{user_name}_{role}_{ws_name}")
}

/// Construct a base session key for ticket-driven agent work.
///
/// The base key format is `ticket_{ticket_id}_{role}`.
///
/// ## Usage
///
/// * **Singular dispatch** (e.g., Engineer at `dispatch_engineer`): the base
///   key is used directly — no suffix is appended.
///
/// * **Parallel agents** (analysts, reviewers, QA via
///   `run_parallel_agents`): the caller appends `_{index}_{suffix}`
///   for disambiguation, producing keys like
///   `ticket_{ticket_id}_{role}_0_nano`.
#[must_use]
pub(crate) fn ticket_session_key(ticket_id: &str, role: &str) -> String {
    format!("ticket_{ticket_id}_{role}")
}

/// Construct a session key for Manager agents (workspace-scoped).
///
/// Format: `manager_{ws_name}`
#[must_use]
pub fn manager_session_key(ws_name: &str) -> String {
    format!("manager_{ws_name}")
}

/// Construct a session key for Maintainer agents (workspace-scoped, unique per run).
///
/// Format: `maintainer_{ws_name}_{suffix}`
/// Each run gets a fresh key (via random suffix) — maintainer runs should not
/// accumulate conversation history across maintenance cycles.
#[must_use]
pub(crate) fn maintainer_session_key(ws_name: &str) -> String {
    format!("maintainer_{}_{}", ws_name, crate::generate_suffix())
}

/// Construct a session key for sub-agent asks (Engineer/Maintainer → sub-agent).
///
/// Format: `ask_{ws_name}_{role}_{suffix}`
#[must_use]
pub(crate) fn ask_session_key(ws_name: &str, role: &str) -> String {
    format!("ask_{}_{}_{}", ws_name, role, crate::generate_suffix())
}

/// Construct a session key for workspace role discovery.
///
/// Format: `discovery_{ws_name}_{role}_{suffix}`
#[must_use]
pub(crate) fn discovery_session_key(ws_name: &str, role: &str) -> String {
    format!(
        "discovery_{}_{}_{}",
        ws_name,
        role,
        crate::generate_suffix()
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

// ── TRANSIENT SESSION PREFIX GUARDS ───────────────────────────
//
// [`TRANSIENT_SESSION_PREFIXES`] controls which sessions are cleaned up by
// [`cleanup_old_transient_sessions`] (SQL `LIKE '{prefix}%'`, equivalent to
// `key.starts_with(prefix)`).
//
// Two invariants:
// 1. **Forward (no collision)**: User-facing session keys must never start with
//    a transient prefix or the periodic cleanup would silently delete user history.
// 2. **Reverse (inclusion)**: Transient session key builders must produce keys
//    starting with a prefix registered in [`TRANSIENT_SESSION_PREFIXES`];
//    an unregistered prefix means transient sessions never get cleaned up (leak).
//
// Limitations: `forward_no_collision_with_user_facing_sessions` covers
// `direct_session_key()` and `manager_session_key()` patterns.
// `reverse_transient_builders_use_registered_prefixes` covers all transient
// builders (ticket, ask, maintainer, discovery). If a new transient role
// adds a session key builder, add it to the reverse test.
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

    /// Known channel identifiers in the system. Must never produce keys
    /// matching a transient prefix.
    const SAFE_CHANNELS: &[&str] = &["telegram", "gui"];

    #[test]
    fn forward_no_collision_with_user_facing_sessions() {
        // For every transient prefix, verify that none of the user-facing
        // session key patterns start with it. Direct keys have the format
        // {channel}_{user}_{role}_{ws}, and `starts_with` only checks the
        // first segment (channel name). Since safe channels ("telegram",
        // "gui") don't match any transient prefix, the role segment (third)
        // has no effect on the assertion outcome — a single role suffices.
        for prefix in TRANSIENT_SESSION_PREFIXES {
            // Manager uses a separate key format (manager_{ws_name}).
            let manager_key = manager_session_key("test-ws");
            assert!(
                !manager_key.starts_with(prefix),
                "MANAGER SESSION KEY COLLISION: \
                 prefix='{prefix}' matches key='{manager_key}'. \
                 Fix: remove '{prefix}' from TRANSIENT_SESSION_PREFIXES \
                 or change the manager_session_key pattern.",
            );

            // Direct chat keys across all safe channels.
            for channel in SAFE_CHANNELS {
                let key = direct_session_key(channel, "testuser", "analyst", "test-ws");
                assert!(
                    !key.starts_with(prefix),
                    "DIRECT SESSION KEY COLLISION: prefix='{prefix}' \
                     matches key='{key}' (channel='{channel}'). \
                     Fix: remove '{prefix}' from TRANSIENT_SESSION_PREFIXES \
                     or change the session key pattern.",
                );
            }
        }
    }

    fn assert_transient_key(key: &str, expected_prefix: &str, builder_expr: &str) {
        assert!(
            key.starts_with(expected_prefix),
            "{builder_expr} = '{key}' does not start with '{expected_prefix}'.\n\
             Fix: update {builder_expr} to produce keys starting with '{expected_prefix}'.",
        );
        assert!(
            TRANSIENT_SESSION_PREFIXES.contains(&expected_prefix),
            "TRANSIENT_SESSION_PREFIXES is missing '{expected_prefix}' — \
             {builder_expr} sessions will never be cleaned up.\n\
             Fix: add \"{expected_prefix}\" to TRANSIENT_SESSION_PREFIXES.",
        );
    }

    #[test]
    fn reverse_transient_builders_use_registered_prefixes() {
        // Each transient key builder must produce keys starting with a
        // prefix that is actually registered in TRANSIENT_SESSION_PREFIXES.
        assert_transient_key(
            &ticket_session_key("abc123", "analyst"),
            "ticket_",
            "ticket_session_key('abc123', 'analyst')",
        );
        assert_transient_key(
            &ask_session_key("ws", "coder"),
            "ask_",
            "ask_session_key('ws', 'coder')",
        );
        assert_transient_key(
            &maintainer_session_key("ws"),
            "maintainer_",
            "maintainer_session_key('ws')",
        );
        assert_transient_key(
            &discovery_session_key("ws", "analyst"),
            "discovery_",
            "discovery_session_key('ws', 'analyst')",
        );
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
        tool_calls: Option<Vec<ProviderToolCall>>,
        reasoning: Option<Reasoning>,
    },
    ToolResult {
        tool_call_id: Option<String>,
        content: String,
    },
}

/// Shared fields extracted from a [`DecodedNativeHistoryMessage`] that providers
/// use to build their local native message types.
/// Tool calls are returned as `Vec<ProviderToolCall>` so each provider can convert them
/// to its own tool-call type.
#[derive(Debug)]
pub(crate) struct NativeMessageParts {
    pub role: String,
    pub content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<ProviderToolCall>>,
    pub reasoning: Option<Reasoning>,
}

impl DecodedNativeHistoryMessage {
    pub(crate) fn into_parts(self) -> NativeMessageParts {
        match self {
            DecodedNativeHistoryMessage::Assistant {
                content,
                tool_calls,
                reasoning,
            } => NativeMessageParts {
                role: ChatRole::Assistant.to_string(),
                content,
                tool_call_id: None,
                tool_calls,
                reasoning,
            },
            DecodedNativeHistoryMessage::ToolResult {
                tool_call_id,
                content,
            } => NativeMessageParts {
                role: ChatRole::Tool.to_string(),
                content: Some(content),
                tool_call_id,
                tool_calls: None,
                reasoning: None,
            },
        }
    }
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

        // Extract reasoning fields once, before the tool_calls branch.
        let (r, rc, rd) =
            crate::providers::reasoning_roundtrip::json_lossless_assistant_reasoning_fields(value);
        let reasoning = Reasoning::from_optional_parts(r, rc, rd);

        if let Some(tool_calls_value) = value.get("tool_calls")
            && let Ok(mut parsed_calls) =
                serde_json::from_value::<Vec<ProviderToolCall>>(tool_calls_value.clone())
        {
            for call in &mut parsed_calls {
                if let Some(s) = call.arguments.as_str()
                    && let Ok(v) = serde_json::from_str::<serde_json::Value>(s)
                {
                    call.arguments = v;
                }
            }

            return Some(DecodedNativeHistoryMessage::Assistant {
                content,
                tool_calls: Some(parsed_calls),
                reasoning,
            });
        }

        return Some(DecodedNativeHistoryMessage::Assistant {
            content,
            tool_calls: None,
            reasoning,
        });
    }

    if message.role == ChatRole::Tool
        && let Some(value) = parsed.as_ref()
    {
        return Some(DecodedNativeHistoryMessage::ToolResult {
            tool_call_id: value
                .get("tool_call_id")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string),
            content: value
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map_or_else(|| message.content.clone(), ToString::to_string),
        });
    }

    None
}
