//! Shared agent-session rendering layer — one view-model ("ledger") plus shared
//! components for the Sessions page (full transcript) and the Running Agents page (a
//! compact projection of the same ledger, never a second data structure).

use iced::widget::markdown;

use crate::ChatMessage;
use crate::agent::registry::RunningTool;
use crate::providers::plaintext_for_display;
use crate::session::{DecodedNativeHistoryMessage, decode_native_history_message};
use crate::util::none_if_empty;

use super::media_markers;

pub(crate) mod preview;
mod render;

// Shared components re-exported for both pages.
pub(crate) use render::{MAX_TOOL_TOOLTIP_WIDTH, ToolBlockView, tool_block, truncate_at_boundary};

/// One entry of the flat session ledger: a regular message (optionally carrying a thinking
/// block) or one assistant tool-call round with its `tool_call_id`-matched results.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SessionEntry {
    Message {
        role: crate::ChatRole,
        content: Option<String>,
        /// The decoded Reasoning, else a literal [thinking] block from the content.
        thinking: Option<String>,
    },
    ToolRound {
        narration: Option<String>,
        reasoning: Option<String>,
        calls: Vec<ToolCallEntry>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCallEntry {
    pub tool_call_id: String,
    pub tool: RunningTool,
    pub result: Option<String>,
}

/// Split a decoded assistant message into (thinking, visible body): the decoded Reasoning
/// wins, else a literal `[thinking]…[/thinking]` pair is extracted panic-safely — the closer
/// is searched only after the opener (historical slice-panic regression — see tests).
fn split_thinking(
    content: Option<String>,
    reasoning: Option<String>,
) -> (Option<String>, Option<String>) {
    if let Some(reasoning) = reasoning
        && !reasoning.is_empty()
    {
        return (Some(reasoning), content.and_then(|c| none_if_empty(&c)));
    }
    let Some(content) = content else {
        return (None, None);
    };
    if let Some(thinking_start) = content.find("[thinking]") {
        let body_start = thinking_start + "[thinking]".len();
        if let Some(rel_end) = content[body_start..].find("[/thinking]") {
            let end = body_start + rel_end;
            let thinking = content[body_start..end].trim().to_string();
            let after = content[end + "[/thinking]".len()..].trim().to_string();
            return (Some(thinking), none_if_empty(&after));
        }
    }
    (None, none_if_empty(&content))
}

/// [`build_ledger_with_sources`]'s entries alone, for callers that never need the row mapping.
pub(crate) fn build_ledger(messages: &[ChatMessage]) -> Vec<SessionEntry> {
    build_ledger_with_sources(messages).0
}

/// Decode the native history messages into the flat ledger plus a parallel `sources` vec
/// mapping each entry back to the message row that CREATED it: the assistant tool-call frame
/// for a [`SessionEntry::ToolRound`], the message row for a [`SessionEntry::Message`].
///
/// Pure and deterministic: each assistant message becomes a regular message (with an
/// optional thinking block) or a tool round, and each tool result attaches to the first
/// unmatched call of the nearest preceding tool round; stray results render like a regular
/// message — so `sources` is NOT the identity.
///
/// Contract: `sources` is strictly increasing, and cutting stored rows from `sources[i]` on
/// removes exactly the entries `i..` — the Sessions page's truncate relies on both.
pub(crate) fn build_ledger_with_sources(
    messages: &[ChatMessage],
) -> (Vec<SessionEntry>, Vec<usize>) {
    let mut entries: Vec<SessionEntry> = Vec::new();
    let mut sources: Vec<usize> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let before = entries.len();
        match decode_native_history_message(message) {
            Some(DecodedNativeHistoryMessage::Assistant {
                content,
                tool_calls,
                reasoning,
            }) => {
                if let Some(calls) = tool_calls.filter(|calls| !calls.is_empty()) {
                    entries.push(SessionEntry::ToolRound {
                        narration: content.filter(|c| !c.is_empty()),
                        reasoning: plaintext_for_display(reasoning.as_ref()),
                        calls: calls
                            .iter()
                            .map(|call| ToolCallEntry {
                                tool_call_id: call.id.clone(),
                                tool: RunningTool::from_tool_call(call),
                                result: None,
                            })
                            .collect(),
                    });
                } else {
                    // No (or an empty) tool-call list: a regular message.
                    let (thinking, body) =
                        split_thinking(content, plaintext_for_display(reasoning.as_ref()));
                    entries.push(SessionEntry::Message {
                        role: message.role,
                        content: body,
                        thinking,
                    });
                }
            }
            Some(DecodedNativeHistoryMessage::ToolResult {
                tool_call_id,
                content,
            }) => {
                let mut remaining = Some(content);
                let mut attached = false;
                for entry in entries.iter_mut().rev() {
                    if let SessionEntry::ToolRound { calls, .. } = entry
                        && let Some(call) = calls
                            .iter_mut()
                            .find(|c| c.tool_call_id == tool_call_id && c.result.is_none())
                    {
                        call.result = remaining.take();
                        attached = true;
                        break;
                    }
                }
                if !attached {
                    entries.push(SessionEntry::Message {
                        role: message.role,
                        content: remaining,
                        thinking: None,
                    });
                }
            }
            _ => {
                entries.push(SessionEntry::Message {
                    role: message.role,
                    content: Some(message.content.clone()),
                    thinking: None,
                });
            }
        }
        // source row.
        if entries.len() != before {
            sources.push(index);
        }
    }
    (entries, sources)
}

/// A transcript body parsed for display: its markdown items and the attachments in body
/// order (drawn as one block above the text, a file row's click revealing it).
pub(crate) struct ParsedBody {
    pub items: Vec<markdown::Item>,
    pub attachments: Vec<media_markers::MediaAttachment>,
}

/// Per-entry markdown parse for bubble bodies: Message → its content, ToolRound → its
/// narration; `None` = nothing to render. `workspace` is the root relative marker targets
/// resolve against — every body follows the same rule whatever role it arrived under.
pub(crate) fn parse_entry_bodies(
    entries: &[SessionEntry],
    workspace: Option<&std::path::Path>,
) -> Vec<Option<ParsedBody>> {
    entries
        .iter()
        .map(|entry| {
            let display = match entry {
                SessionEntry::Message { content, .. } => content.as_deref()?,
                SessionEntry::ToolRound { narration, .. } => narration.as_deref()?,
            };
            let (items, attachments) = media_markers::parse_body(display, workspace);
            Some(ParsedBody { items, attachments })
        })
        .collect()
}

/// The reasoning promoted into a tool round's narration slot, if any: the assistant text
/// content is empty (whitespace-only counts) while a decoded Reasoning block is present.
/// Render-only — the ledger is untouched, so empty content stays empty in the history and
/// in the model replay.
pub(crate) fn promoted_reasoning<'a>(
    narration: Option<&'a str>,
    reasoning: Option<&'a str>,
) -> Option<&'a str> {
    match reasoning {
        Some(reasoning) if narration.is_none_or(|n| n.trim().is_empty()) => Some(reasoning),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatRole;

    type SplitCase<'a> = (
        &'a str,
        &'a str,
        Option<&'a str>,
        Option<&'a str>,
        Option<&'a str>,
    );

    #[test]
    fn split_thinking_cases() {
        let cases: &[SplitCase] = &[
            (
                "first pair only",
                "intro [thinking] inner [/thinking] mid [thinking] later [/thinking] tail",
                None,
                Some("inner"),
                Some("mid [thinking] later [/thinking] tail"),
            ),
            (
                "decoded reasoning wins",
                "[thinking] literal [/thinking]",
                Some("decoded reasoning"),
                Some("decoded reasoning"),
                Some("[thinking] literal [/thinking]"),
            ),
        ];
        for &(name, content, reasoning, expected_thinking, expected_body) in cases {
            let (thinking, body) =
                split_thinking(Some(content.to_string()), reasoning.map(str::to_string));
            assert_eq!(
                thinking.as_deref(),
                expected_thinking,
                "case {name}: thinking"
            );
            assert_eq!(body.as_deref(), expected_body, "case {name}: body");
        }
    }

    #[test]
    fn empty_tool_call_list_never_forms_a_tool_round() {
        let messages = vec![ChatMessage {
            role: ChatRole::Assistant,
            content: r#"{"content":"hello","tool_calls":[]}"#.to_string(),
        }];
        let entries = build_ledger(&messages);
        match &entries[..] {
            [SessionEntry::Message { content, .. }] => {
                assert_eq!(content.as_deref(), Some("hello"));
            }
            other => panic!("expected a single Message entry, got {other:?}"),
        }
    }

    #[test]
    fn ledger_sources_track_the_source_row_of_each_entry() {
        // The tool result attaches to its round, so the entry→message map is not the identity.
        let messages = vec![
            ChatMessage::user("go"),
            assistant_tool_call(Some("checking"), None),
            ChatMessage::tool_result("call_1", "result"),
            ChatMessage::user("next"),
        ];
        let (_entries, sources) = build_ledger_with_sources(&messages);
        assert_eq!(sources, vec![0, 1, 3]);
    }

    fn assistant_tool_call(content: Option<&str>, reasoning: Option<&str>) -> ChatMessage {
        let calls = vec![serde_json::json!({
            "id": "call_1",
            "name": "read",
            "arguments": serde_json::to_string(&serde_json::json!({"path": "a.rs"})).unwrap(),
        })];
        let mut body = serde_json::json!({ "content": content, "tool_calls": calls });
        if let Some(reasoning) = reasoning {
            body["reasoning"] = serde_json::json!(reasoning);
        }
        ChatMessage::assistant(body.to_string())
    }

    #[derive(Default)]
    struct PromotedCase {
        name: &'static str,
        content: Option<&'static str>,
        reasoning: Option<&'static str>,
        promoted: Option<&'static str>,
    }

    /// Reasoning is promoted only when the round has no visible narration and there is
    /// reasoning to promote.
    #[test]
    fn promoted_reasoning_cases() {
        let cases: &[PromotedCase] = &[
            PromotedCase {
                name: "textless round with reasoning",
                reasoning: Some("Checking the file first"),
                promoted: Some("Checking the file first"),
                ..PromotedCase::default()
            },
            PromotedCase {
                name: "visible content does not promote",
                content: Some("Doing it"),
                reasoning: Some("Reasoning here"),
                ..PromotedCase::default()
            },
            PromotedCase {
                name: "whitespace content counts as empty",
                content: Some("   "),
                reasoning: Some("Reasoning here"),
                promoted: Some("Reasoning here"),
            },
            PromotedCase {
                name: "without reasoning does not promote",
                ..PromotedCase::default()
            },
        ];
        for case in cases {
            let name = case.name;
            let messages = vec![assistant_tool_call(case.content, case.reasoning)];
            let entries = build_ledger(&messages);
            match &entries[..] {
                [
                    SessionEntry::ToolRound {
                        narration,
                        reasoning,
                        ..
                    },
                ] => {
                    assert_eq!(narration.as_deref(), case.content, "case {name}: narration");
                    assert_eq!(
                        reasoning.as_deref(),
                        case.reasoning,
                        "case {name}: reasoning"
                    );
                    assert_eq!(
                        promoted_reasoning(narration.as_deref(), reasoning.as_deref()),
                        case.promoted,
                        "case {name}: promoted"
                    );
                }
                other => panic!("case {name}: expected a single ToolRound entry, got {other:?}"),
            }
        }
    }
}
