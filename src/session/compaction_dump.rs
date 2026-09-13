//! Compaction spill-dump: durable recovery artifact of the conversation
//! messages a summarization compaction deletes.
//!
//! When [`crate::session::Session::apply_summary`] rebuilds history it keeps
//! only a fresh system prompt, the summary, and the latest retention window
//! (see [`crate::session::select_retention_window`]) — everything else is
//! dropped from the session permanently. This module writes the dropped
//! messages to a readable `.md` dump and returns its path so the persisted
//! summary message can point the agent at it: earlier context stays
//! recoverable with the read tool for the session's whole lifetime.
//!
//! Lifecycle: the dump is NOT registered for owner-delete-at-run-end cleanup
//! (unlike the `spill_*` shell spill files) and is never GC'd — the daemon
//! builds no startup purge, so crash leftovers are the OS temp sweep's job.
//! A compaction whose persist fails orphans its dump; that is accepted —
//! orphan dumps are dead weight, not a correctness problem, and deleting a
//! dump that a concurrently-retried persist might reference would be worse.
//!
//! Cross-references: media-marker handling in [`crate::util`] (data-URI
//! stripping) and the read tool's `MAX_FILE_SIZE_BYTES` cap (the reason data
//! URIs are stripped).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::{ChatMessage, ChatRole, Role};

/// Directory name (inside the role-appropriate dump root) holding compaction
/// dumps. For the Assistant this is a hidden dir in the user's personal
/// workspace; for every other role a plain dir under the pinned agent temp
/// root. Hidden-dir skips (personal-files walk, ripgrep defaults) already
/// keep it out of listings.
const DUMP_DIR_NAME: &str = ".compaction";

/// Dump the messages this compaction deletes to a durable `.md` file.
///
/// This is a session-lifetime recovery artifact: the messages it contains are
/// removed from the session by the caller's persist, and this dump is the
/// only place they survive. NOT registered for owner-delete-at-run-end
/// cleanup (unlike the shell `spill_*` files) and never GC'd — orphan dumps
/// from failed compaction persists are accepted.
///
/// Fail-open: any filesystem failure logs a warning and returns `None` — a
/// dump failure never blocks compaction.
///
/// Returns `Some(path)` on success; `None` when there is nothing to dump
/// (deletion set empty), the dump root cannot be resolved/created, or the
/// write fails.
pub(crate) async fn dump_deleted_messages(
    agent_id: &str,
    old_history: &[ChatMessage],
    retained_window: &[ChatMessage],
    role: &Role,
    user_name: &str,
) -> Option<PathBuf> {
    let deleted = deleted_conversation_messages(old_history, retained_window);
    if deleted.is_empty() {
        return None;
    }
    let body = render_dump(&deleted);
    let dir = dump_dir(role, user_name).await?;
    let path = next_dump_path(&dir, agent_id).await?;
    if let Err(e) = tokio::fs::write(&path, &body).await {
        tracing::warn!(
            agent_id = %agent_id,
            path = %path.display(),
            error = %e,
            "Failed to write compaction dump — continuing without it"
        );
        return None;
    }
    Some(path)
}

/// Compute the messages `old_history` contributes to the dump: everything
/// except the retained window and System-role messages.
///
/// Multiset matching, not set matching: `old_history` is walked in order and
/// each non-System message consumes the FIRST not-yet-consumed equal message
/// in `retained_window` (equality on role + content). Two identical user
/// messages are both deleted only if the window really contains two —
/// positional or set semantics would silently drop one of them. System-role
/// messages are skipped entirely: the regenerated context prefix and the
/// superseded previous summary are not conversation content, and the fresh
/// system prompt is never part of the retained window anyway.
///
/// The returned items borrow `old_history` so the render can stay
/// allocation-free per message.
fn deleted_conversation_messages<'a>(
    old_history: &'a [ChatMessage],
    retained_window: &[ChatMessage],
) -> Vec<&'a ChatMessage> {
    let mut consumed = vec![false; retained_window.len()];
    let mut deleted = Vec::new();
    for msg in old_history {
        if msg.role == ChatRole::System {
            // Regenerated context prefix / superseded summary — not
            // conversation content, never dumped.
            continue;
        }
        if let Some(slot) = retained_window
            .iter()
            .zip(&consumed)
            .position(|(retained, &used)| !used && retained == msg)
        {
            consumed[slot] = true;
        } else {
            deleted.push(msg);
        }
    }
    deleted
}

/// Render the deleted messages as a plain readable transcript. The whole
/// output runs through [`crate::util::scrub_credentials`] — the same
/// scrubbing tool output gets — because dumps outlive the session and may be
/// read by any agent with a read tool.
fn render_dump(messages: &[&ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let _ = writeln!(out, "=== {} ===", msg.role);
        match super::decode_native_history_message(msg) {
            Some(super::DecodedNativeHistoryMessage::Assistant {
                content,
                tool_calls,
                ..
            }) => {
                // Reasoning is deliberately skipped: scratch space, not
                // conversation content.
                if let Some(content) = content.filter(|c| !c.is_empty()) {
                    let _ = writeln!(out, "{}", crate::util::strip_data_uris(&content));
                }
                if let Some(calls) = tool_calls {
                    for call in calls {
                        let _ = writeln!(
                            out,
                            "[tool_call {}] {}: {}",
                            call.id,
                            call.name,
                            crate::util::strip_data_uris(&call.arguments.to_string())
                        );
                    }
                }
            }
            Some(super::DecodedNativeHistoryMessage::ToolResult {
                tool_call_id,
                content,
            }) => {
                let _ = writeln!(out, "[result for tool_call {tool_call_id}]");
                let _ = writeln!(out, "{}", crate::util::strip_data_uris(&content));
            }
            // Plain text (never JSON-wrapped): render as-is.
            None => {
                let _ = writeln!(out, "{}", crate::util::strip_data_uris(&msg.content));
            }
        }
        out.push('\n');
    }
    crate::util::scrub_credentials(out.trim_end())
}

/// Resolve the directory holding compaction dumps for this role.
///
/// - [`Role::Assistant`]: pinned to the user's personal workspace — the only
///   place its strict (workspace-only) read tool can open — so the dump goes
///   to `<personal_workspace>/{[`DUMP_DIR_NAME`]}`. `user_name` is validated
///   via [`crate::users::is_valid_personal_user_name`]; `None` when invalid.
/// - Every other role has the general read tool, and the whole pinned temp
///   root is already inside its allowlist, so the dump goes to
///   `<agent_temp_dir>/compaction`. It deliberately does NOT use the
///   `spill_*` filename shape and is NOT registered in `SPILL_OWNERS` —
///   dumps outlive the run.
///
/// Fail-open: `None` when validation fails or the directory cannot be
/// created.
async fn dump_dir(role: &Role, user_name: &str) -> Option<PathBuf> {
    let dir = if matches!(role, Role::Assistant) {
        if !crate::users::is_valid_personal_user_name(user_name) {
            return None;
        }
        crate::users::personal_workspace_path(user_name).join(DUMP_DIR_NAME)
    } else {
        crate::tools::shell::agent_temp_dir()?.join("compaction")
    };
    tokio::fs::create_dir_all(&dir).await.ok()?;
    Some(dir)
}

/// Pick the next dump path: `compaction-{sanitized_agent_id}-{seq:03}-{unix_millis}.md`.
///
/// `seq` is per-session monotonic, derived by counting existing files whose
/// name starts with `compaction-{sanitized}-` (start at 1). The unix-millis
/// suffix keeps names unique even if the OS temp sweep removed earlier dumps
/// (a plain counter would then reuse a name whose file is gone — harmless,
/// but a fresh name avoids any ambiguity). Fail-open: `None` on `read_dir`
/// failure.
async fn next_dump_path(dir: &Path, agent_id: &str) -> Option<PathBuf> {
    let sanitized: String = agent_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let prefix = format!("compaction-{sanitized}-");
    let mut existing = tokio::fs::read_dir(dir).await.ok()?;
    let mut count = 0usize;
    while let Ok(Some(entry)) = existing.next_entry().await {
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            count += 1;
        }
    }
    let millis = chrono::Utc::now().timestamp_millis();
    Some(dir.join(format!("{prefix}{count:03}-{millis}.md")))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatRole;

    #[test]
    fn deleted_set_keeps_window_and_drops_older_turns() {
        let system = ChatMessage::system("fresh system prompt");
        let history = vec![
            system,
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("u2"),
            ChatMessage::assistant("a2"),
            ChatMessage::user("u3"),
            ChatMessage::assistant("a3"),
            ChatMessage::user("u4"),
            ChatMessage::assistant("a4"),
        ];
        // Mirrors select_retention_window output: last 3 users + assistants.
        let retained: Vec<ChatMessage> = history[3..].to_vec();

        let deleted = deleted_conversation_messages(&history, &retained);
        let contents: Vec<&str> = deleted.iter().map(|m| m.content.as_str()).collect();
        // Exactly the older user/assistant turns; system never dumped.
        assert_eq!(contents, vec!["u1", "a1"]);
    }

    #[test]
    fn deleted_set_handles_duplicate_messages_as_multiset() {
        let system = ChatMessage::system("sys");
        let history = vec![
            system,
            ChatMessage::user("same text"),
            ChatMessage::user("same text"),
            ChatMessage::assistant("answer"),
        ];
        // Window holds only ONE of the two identical users: the second user
        // and the unretained assistant are deleted. With set semantics the
        // second user would wrongly be considered retained.
        let retained = vec![ChatMessage::user("same text")];
        let deleted = deleted_conversation_messages(&history, &retained);
        assert_eq!(deleted.len(), 2);
        assert_eq!(deleted[0].content, "same text");
        assert_eq!(deleted[1].content, "answer");

        // Window holds both users: only the unretained assistant is deleted.
        let retained = vec![
            ChatMessage::user("same text"),
            ChatMessage::user("same text"),
        ];
        let deleted = deleted_conversation_messages(&history, &retained);
        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0].content, "answer");
    }

    #[test]
    fn render_dump_renders_native_frames_and_scrubs_credentials() {
        let frame = ChatMessage {
            role: ChatRole::Assistant,
            content: serde_json::json!({
                "content": "checking api_key: supersecretvalue123",
                "tool_calls": [
                    {"id": "call_1", "name": "read", "arguments": {"path": "x.md"}}
                ]
            })
            .to_string(),
        };
        let result = ChatMessage::tool_result("call_1", "file contents here");

        let out = render_dump(&[&frame, &result]);
        assert!(out.contains("=== assistant ==="));
        assert!(out.contains("checking api_key: supe*[REDACTED]"));
        assert!(!out.contains("supersecretvalue123"));
        assert!(out.contains("[tool_call call_1] read: {\"path\":\"x.md\"}"));
        assert!(out.contains("=== tool ==="));
        assert!(out.contains("[result for tool_call call_1]"));
        assert!(out.contains("file contents here"));
    }
}
