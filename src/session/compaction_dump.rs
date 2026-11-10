//! Compaction spill-dump: the recovery artifact of the conversation messages a
//! summarization compaction deletes.
//!
//! [`crate::session::Session::apply_summary`] rebuilds history keeping only a fresh system
//! prompt, the summary and the latest retention window
//! ([`crate::session::select_retention_window`]); everything else is dropped permanently. This
//! module writes the dropped messages to a readable `.md` dump and returns its path so the
//! persisted summary can point the agent at it.
//!
//! Lifecycle: the dump is NOT registered for owner-delete-at-run-end cleanup (unlike the
//! `spill_*` shell files) and no startup purge exists. For every role but the Assistant it
//! lives under the pinned temp root, so the periodic temp cleaner sweeps it once it is a day
//! old ([`crate::temp`]); the Assistant's lives in its personal workspace, outside the
//! cleaner's scope. A compaction whose persist fails orphans its dump — accepted, since
//! deleting a dump a concurrently-retried persist might reference would be worse.
//!
//! Media-marker handling in [`crate::util`] (data-URI stripping) exists for the read tool's
//! `MAX_FILE_SIZE_BYTES` cap.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::{ChatMessage, ChatRole, Role};

/// Directory name (inside the role-appropriate root) holding compaction dumps. For the
/// Assistant it is a hidden dir in the personal workspace, elsewhere a plain dir under the agent
/// temp root. `personal_changes::is_skipped` prunes it at any depth, so neither the
/// session-start picture nor the change report shows it, and ripgrep keeps it out of search.
const DUMP_DIR_NAME: &str = ".compaction";

/// Dump the messages this compaction deletes to a `.md` file. The caller's persist removes them
/// from the session, so this dump is their only copy; lifecycle is in the module header.
///
/// Fail-open: any filesystem failure logs a warning and returns `None` — a dump failure never
/// blocks compaction. `None` also when there is nothing to dump, the root cannot be
/// resolved/created, or the write fails.
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

/// Compute the messages `old_history` contributes to the dump: everything except the retained
/// window and System-role messages.
///
/// Multiset matching, not set matching: each non-System message consumes the FIRST
/// not-yet-consumed equal message (role + content) in `retained_window`, so two identical user
/// messages are both deleted only if the window really holds two. System-role messages are
/// skipped: the regenerated context prefix and the superseded summary are not conversation
/// content. The items borrow `old_history`, so the render stays allocation-free per message.
fn deleted_conversation_messages<'a>(
    old_history: &'a [ChatMessage],
    retained_window: &[ChatMessage],
) -> Vec<&'a ChatMessage> {
    let mut consumed = vec![false; retained_window.len()];
    let mut deleted = Vec::new();
    for msg in old_history {
        if msg.role == ChatRole::System {
            // Regenerated context prefix / superseded summary — never dumped.
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

/// Render the deleted messages as a plain readable transcript, the whole output through
/// [`crate::util::scrub_credentials`] — the same scrubbing tool output gets, since any agent with
/// a read tool can open a dump.
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
                // Reasoning is scratch space, not conversation content.
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

/// - [`Role::Assistant`]: `<personal_workspace>/{[`DUMP_DIR_NAME`]}` — the one location both
///   audiences' read tools can open (the admin's reaches the machine, a guest's only its
///   workspace, and the general read's temp allowlist is not the guest's). `user_name` is
///   validated via [`crate::users::is_valid_personal_user_name`].
/// - Every other role has the general read tool and the pinned temp root is inside its
///   allowlist, so the dump goes to `<agent_temp_dir>/compaction` — deliberately NOT the
///   `spill_*` shape and NOT in `SPILL_OWNERS`, since a dump outlives its run.
///
/// Fail-open: `None` when validation fails or the directory cannot be created.
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
/// `seq` is per-session monotonic, derived by counting existing files with the same prefix (from
/// 1). The unix-millis suffix keeps names unique even after an earlier dump was reclaimed. `None`
/// on `read_dir` failure.
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

    /// The dump carries the complement of the retention window as a MULTISET: older turns only,
    /// never the system prompt, and one of two identical user messages still counts as deleted
    /// while the window holds only the other.
    #[test]
    fn deleted_set_is_the_window_complement_counted_as_a_multiset() {
        let history = vec![
            ChatMessage::system("fresh system prompt"),
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

        let duplicated = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("same text"),
            ChatMessage::user("same text"),
            ChatMessage::assistant("answer"),
        ];
        let cases: [(Vec<ChatMessage>, &[&str]); 2] = [
            (
                vec![ChatMessage::user("same text")],
                &["same text", "answer"],
            ),
            (
                vec![
                    ChatMessage::user("same text"),
                    ChatMessage::user("same text"),
                ],
                &["answer"],
            ),
        ];
        for (retained, expected) in &cases {
            let deleted = deleted_conversation_messages(&duplicated, retained);
            let contents: Vec<&str> = deleted.iter().map(|m| m.content.as_str()).collect();
            assert_eq!(contents, *expected, "one message per retained occurrence");
        }
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
