//! Shared agent-session rendering layer — one view-model ("ledger") plus
//! shared components used by both the Sessions page (full historical
//! transcript) and the Running Agents page (live compact projection). The
//! compact view is a projection of the same ledger, never a second data
//! structure.

use std::borrow::Cow;
use std::ops::Range;

use iced::widget::markdown;

use crate::ChatMessage;
use crate::agent::registry::RunningTool;
use crate::providers::plaintext_for_display;
use crate::session::{DecodedNativeHistoryMessage, decode_native_history_message};
use crate::util::none_if_empty;

use super::markdown_breaks;
use super::media_markers;

pub(crate) mod preview;
mod render;

// Shared components re-exported for the Sessions page (full view) and the
// Running Agents page (compact projection).
pub(crate) use render::{MAX_TOOL_TOOLTIP_WIDTH, tool_block, truncate_at_boundary};

/// One entry of the flat session ledger: a regular message (system/user/
/// assistant, optionally carrying a thinking block) or one assistant
/// tool-call round (one assistant message's calls plus their
/// tool_call_id-matched results).
#[derive(Debug, Clone)]
pub(crate) enum SessionEntry {
    Message {
        role: crate::ChatRole,
        /// Visible body text (thinking extracted). None = empty.
        content: Option<String>,
        /// Thinking text: the decoded Reasoning when present, else a literal
        /// [thinking] block extracted from the content.
        thinking: Option<String>,
    },
    ToolRound {
        /// Assistant text content that appeared alongside the calls.
        narration: Option<String>,
        /// The assistant message's decoded long Reasoning block.
        reasoning: Option<String>,
        calls: Vec<ToolCallEntry>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct ToolCallEntry {
    pub tool_call_id: String,
    /// Full unscrubbed args pairs (RunningTool flattening).
    pub tool: RunningTool,
    /// tool_call_id-matched result content.
    pub result: Option<String>,
}

/// Split a decoded assistant message into (thinking, visible body).
/// The decoded Reasoning wins; otherwise a literal `[thinking]...[/thinking]`
/// pair is extracted from the content, panic-safely: the closer is searched
/// only after the opener (historical slice-panic regression — see tests).
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

/// Decode the native history messages into the flat session ledger. Pure and
/// deterministic: each assistant message becomes a regular message (with an
/// optional thinking block) or a tool round, and each tool result attaches to
/// the first unmatched call of the nearest preceding tool round; stray results
/// render like a regular message.
pub(crate) fn build_ledger(messages: &[ChatMessage]) -> Vec<SessionEntry> {
    let mut entries: Vec<SessionEntry> = Vec::new();
    for message in messages {
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
                    // No (or an empty) tool-call list: a regular assistant
                    // message, possibly carrying a thinking block.
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
            _ => entries.push(SessionEntry::Message {
                role: message.role,
                content: Some(message.content.clone()),
                thinking: None,
            }),
        }
    }
    entries
}

/// Per-entry markdown parse for bubble bodies: Message → its content,
/// ToolRound → its narration. None = nothing to render.
pub(crate) fn parse_entry_bodies(entries: &[SessionEntry]) -> Vec<Option<Vec<markdown::Item>>> {
    entries
        .iter()
        .map(|entry| {
            let display = match entry {
                SessionEntry::Message { content, .. } => content.as_deref()?,
                SessionEntry::ToolRound { narration, .. } => narration.as_deref()?,
            };
            let processed = media_markers::preprocess(display);
            let escaped = escape_html_blocks(&processed);
            let with_breaks = markdown_breaks::hard_breaks(&escaped);
            Some(markdown::parse(&with_breaks).collect())
        })
        .collect()
}

/// Escape `<` → `&lt;` only inside the regions the markdown parser classifies
/// as HTML (block-level `Event::Html` plus inline `Event::InlineHtml`), so
/// angle-bracket tag blocks (`<system-notification>`, `<analyze-tool-result>`,
/// `<timestamp>` …) render as visible literal text instead of being dropped
/// wholesale.
///
/// This is deliberately a `<`-only escape: `util::escape_html` is not reused
/// (it also escapes `>` — breaking blockquotes — and `&`, which would
/// double-encode existing entities).
fn escape_html_blocks(text: &str) -> Cow<'_, str> {
    if !text.contains('<') {
        return Cow::Borrowed(text);
    }
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (event, range) in
        pulldown_cmark::Parser::new_ext(text, markdown_breaks::iced_markdown_options())
            .into_offset_iter()
    {
        match event {
            pulldown_cmark::Event::Html(_) | pulldown_cmark::Event::InlineHtml(_) => {
                ranges.push(range);
            }
            _ => {}
        }
    }
    if ranges.is_empty() {
        return Cow::Borrowed(text);
    }
    // Events arrive in document order with disjoint ranges; merge overlapping
    // or adjacent ones so multi-line HTML blocks collapse into single
    // replacement regions.
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = merged.last_mut() {
            if range.start <= last.end {
                last.end = last.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    let mut out = String::with_capacity(text.len() + merged.len() * 4);
    let mut pos = 0;
    for range in &merged {
        out.push_str(&text[pos..range.start]);
        out.push_str(&text[range.clone()].replace('<', "&lt;"));
        pos = range.end;
    }
    out.push_str(&text[pos..]);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatRole;

    #[test]
    fn split_thinking_reversed_markers_never_panics() {
        // Regression: a literal "[/thinking]" before "[thinking]" previously
        // sliced `content[body_start..end]` with `end < body_start`
        // ("byte range starts at X but ends at Y"), aborting the GUI. The
        // closer is searched only after the opener, so this input resolves to
        // plain body text instead of crashing.
        let (thinking, body) =
            split_thinking(Some("[/thinking] before [thinking]".to_string()), None);
        assert_eq!(thinking, None);
        assert_eq!(body.as_deref(), Some("[/thinking] before [thinking]"));
    }

    #[test]
    fn split_thinking_extracts_first_pair_only() {
        let (thinking, body) = split_thinking(
            Some(
                "intro [thinking] inner [/thinking] mid [thinking] later [/thinking] tail"
                    .to_string(),
            ),
            None,
        );
        assert_eq!(thinking.as_deref(), Some("inner"));
        // Only the first pair is consumed; a later literal block stays visible
        // body text.
        assert_eq!(
            body.as_deref(),
            Some("mid [thinking] later [/thinking] tail")
        );
    }

    #[test]
    fn split_thinking_decoded_reasoning_wins() {
        let (thinking, body) = split_thinking(
            Some("[thinking] literal [/thinking]".to_string()),
            Some("decoded reasoning".to_string()),
        );
        assert_eq!(thinking.as_deref(), Some("decoded reasoning"));
        assert_eq!(body.as_deref(), Some("[thinking] literal [/thinking]"));
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
}
