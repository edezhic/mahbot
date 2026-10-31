//! Reasoning / chain-of-thought plaintext extraction and OpenAI-compatible replay helpers.
//!
//! These functions process model-provided reasoning data — OpenRouter-style
//! `reasoning_details` JSON, inline `<think>...</think>` tags, and merged
//! `reasoning_content` / `reasoning` strings — into human-readable plaintext
//! for display in the UI or for API roundtrip synthesis.
//!
//! Roundtrip helpers preserve thinking/reasoning across OpenAI-compatible APIs:
//! `OpenRouter` uses `reasoning` and `reasoning_details`; DeepSeek-native uses `reasoning_content`.

use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;

use crate::{Reasoning, ToolCall};

// ── OpenRouter reasoning_details extraction ──────────────────────────────

fn reasoning_detail_type(item: &Value) -> Option<&str> {
    item.get("type").and_then(Value::as_str)
}

fn append_reasoning_fragment(out: &mut String, fragment: &str) {
    let t = fragment.trim();
    if t.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(t);
}

fn append_plaintext_from_detail_item(out: &mut String, item: &Value) {
    let Some(ty) = reasoning_detail_type(item) else {
        return;
    };
    if ty.contains("encrypted") {
        return;
    }
    if ty.contains("summary") {
        if let Some(s) = item.get("summary").and_then(Value::as_str) {
            append_reasoning_fragment(out, s);
        }
        return;
    }
    if ty.contains("text")
        && let Some(s) = item.get("text").and_then(Value::as_str)
    {
        append_reasoning_fragment(out, s);
    }
}

/// Extract human-readable chain-of-thought from OpenRouter-style `reasoning_details` JSON.
///
/// Handles `reasoning.text`, `reasoning.summary`, and similar `type` strings; skips encrypted blobs.
#[must_use]
fn plaintext_from_reasoning_details(details: &Value) -> String {
    let mut out = String::new();
    match details {
        Value::Array(items) => {
            for item in items {
                append_plaintext_from_detail_item(&mut out, item);
            }
        }
        Value::Object(_) => append_plaintext_from_detail_item(&mut out, details),
        _ => {}
    }
    out
}

// ── Inline think-tag stripping ──────────────────────────────────────────

/// Strip `<think>...</think>` reasoning blocks from model output.
///
/// Some models (e.g. `MiniMax`) embed their reasoning inline in `content` using
/// `<think>...</think>` tags instead of (or in addition to) the standard
/// `reasoning_content` API field.
///
/// Returns `None` when stripping leaves an empty string (the model only emitted
/// reasoning wrapped in think tags with no visible output).
#[must_use]
pub(crate) fn strip_think_tags(s: &str) -> Option<String> {
    static THINK_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)<think>.*?</think>|<think>.*$").expect("think tag regex must compile")
    });
    let stripped = THINK_RE.replace_all(s, "").trim().to_string();
    if stripped.is_empty() {
        None
    } else {
        Some(stripped)
    }
}

// ── Merged reasoning string ─────────────────────────────────────────────

/// Prefer `reasoning_content`, then `reasoning` (`OpenRouter`). **Display / effective text only**
/// — never use for API replay fields.
fn merged_reasoning_string(
    reasoning_content: Option<String>,
    reasoning: Option<String>,
) -> Option<String> {
    reasoning_content
        .filter(|s| !s.trim().is_empty())
        .or_else(|| reasoning.filter(|s| !s.trim().is_empty()))
}

// ── Display-ready plaintext ─────────────────────────────────────────────

/// Human-readable thinking line for UI (merges plaintext fields; extracts from details when needed).
#[must_use]
pub(crate) fn plaintext_for_display(reasoning: Option<&Reasoning>) -> Option<String> {
    let r = reasoning?;
    merged_reasoning_string(r.reasoning_content.clone(), r.reasoning.clone()).or_else(|| {
        r.reasoning_details.as_ref().and_then(|d| {
            let s = plaintext_from_reasoning_details(d);
            (!s.trim().is_empty()).then_some(s)
        })
    })
}

/// True when `reasoning_details` carries at least one block worth replaying (non-empty array, etc.).
fn details_has_preservable_blocks(details: &Value) -> bool {
    match details {
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
        _ => false,
    }
}

/// Plain `reasoning_content` string to store on assistant messages and send back to the API.
///
/// When the model only streams structured `reasoning_details` (no `reasoning` / `reasoning_content`
/// string), `DeepSeek` behind `OpenRouter` still expects a `reasoning_content` field on **tool-call**
/// turns — derive text from details or send an empty string when details exist but are not textual,
/// or when the API returns an **empty** `reasoning` string (schema placeholder with no `CoT` yet).
fn reasoning_plaintext_for_roundtrip(
    explicit: Option<&str>,
    details: Option<&Value>,
    has_tool_calls: bool,
) -> Option<String> {
    if let Some(s) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        return Some(s.to_string());
    }
    if let Some(d) = details {
        let extracted = plaintext_from_reasoning_details(d);
        if !extracted.is_empty() {
            return Some(extracted);
        }
        if has_tool_calls && details_has_preservable_blocks(d) {
            return Some(String::new());
        }
    }
    if has_tool_calls && explicit.is_some() {
        return Some(String::new());
    }
    None
}

/// Augment reasoning fields for replay, filling `reasoning_content` when missing.
///
/// `DeepSeek` (and some OpenRouter-routed thinking models) reject follow-up requests unless prior
/// assistant turns echo chain-of-thought in `reasoning_content`. We derive it from `reasoning`
/// and/or `reasoning_details` using [`reasoning_plaintext_for_roundtrip`].
pub(crate) fn native_reasoning_triple_for_replay(
    reasoning: Option<&Reasoning>,
    has_tool_calls: bool,
) -> (Option<String>, Option<String>, Option<Value>) {
    let r_reasoning = reasoning.and_then(|r| r.reasoning.clone());
    let r_content = reasoning.and_then(|r| r.reasoning_content.clone());
    let r_details = reasoning.and_then(|r| r.reasoning_details.clone());

    if r_content.is_some() {
        return (r_reasoning, r_content, r_details);
    }
    let synthesized = reasoning_plaintext_for_roundtrip(
        r_reasoning.as_deref(),
        r_details.as_ref(),
        has_tool_calls,
    );
    (r_reasoning, synthesized, r_details)
}

/// `reasoning_details` when present (opaque JSON).
fn json_reasoning_details(value: &Value) -> Option<Value> {
    value
        .get("reasoning_details")
        .cloned()
        .filter(|v| !v.is_null())
}

/// Lossless string field: key must be present with a JSON string value (including `""`).
fn json_string_field_if_present(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(std::string::ToString::to_string)
}

/// Read `reasoning`, `reasoning_content`, and `reasoning_details` exactly as stored on assistant JSON.
pub(crate) fn json_lossless_assistant_reasoning_fields(
    value: &Value,
) -> (Option<String>, Option<String>, Option<Value>) {
    (
        json_string_field_if_present(value, "reasoning"),
        json_string_field_if_present(value, "reasoning_content"),
        json_reasoning_details(value),
    )
}

fn apply_reasoning_to_payload(payload: &mut Value, reasoning: &Reasoning) {
    if let Some(s) = &reasoning.reasoning {
        payload["reasoning"] = json!(s);
    }
    if let Some(s) = &reasoning.reasoning_content {
        payload["reasoning_content"] = json!(s);
    }
    if let Some(d) = &reasoning.reasoning_details {
        payload["reasoning_details"] = d.clone();
    }
}

fn augment_replay_payload_reasoning_content(payload: &mut Value, has_tool_calls: bool) {
    if payload
        .get("reasoning_content")
        .is_some_and(serde_json::Value::is_string)
    {
        return;
    }

    let explicit = payload.get("reasoning").and_then(serde_json::Value::as_str);
    let details = json_reasoning_details(payload);
    let reasoning_content =
        reasoning_plaintext_for_roundtrip(explicit, details.as_ref(), has_tool_calls);

    let content = match reasoning_content {
        Some(s) => s,
        // reasoning was entirely absent (no `reasoning` or `reasoning_details` fields in the
        // payload) but tool calls exist → DeepSeek still expects `reasoning_content: ""` on the wire.
        None if has_tool_calls && explicit.is_none() && details.is_none() => String::new(),
        None => return,
    };

    if let Some(value) = payload.as_object_mut() {
        value.insert(
            "reasoning_content".to_string(),
            serde_json::Value::String(content),
        );
    }
}

/// Build assistant replay payload with reasoning fields preserved for roundtrip.
#[must_use]
pub(crate) fn assistant_replay_payload(
    text: Option<&str>,
    tool_calls: &[ToolCall],
    reasoning: Option<&Reasoning>,
) -> Value {
    let mut payload = if tool_calls.is_empty() {
        json!({ "content": text.unwrap_or_default() })
    } else {
        let calls_json: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string()),
                })
            })
            .collect();
        let content_value = text
            .filter(|s| !s.is_empty())
            .map_or(Value::Null, |s| Value::String(s.to_string()));
        json!({
            "content": content_value,
            "tool_calls": calls_json,
        })
    };

    if let Some(r) = reasoning {
        apply_reasoning_to_payload(&mut payload, r);
    }
    augment_replay_payload_reasoning_content(&mut payload, !tool_calls.is_empty());

    payload
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod strip_think_tag_tests {
    use super::strip_think_tags;

    #[test]
    fn table() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: Option<&'static str>,
        }

        let cases = [
            Case {
                name: "inline think block",
                input: "visible<think>hidden",
                expected: Some("visible"),
            },
            Case {
                name: "multiple think blocks",
                input: "Answer A <think>hidden 1</think> and B <think>hidden 2</think> done",
                expected: Some("Answer A  and B  done"),
            },
            Case {
                name: "unclosed think tag",
                input: "Visible<think>hidden tail",
                expected: Some("Visible"),
            },
            Case {
                name: "multiline think block",
                input: "Hello<think>\nmulti\nline\n</think>world",
                expected: Some("Helloworld"),
            },
            Case {
                name: "multiple multiline blocks",
                input: "<think>\nblock 1\n</think>A<think>\nblock 2\n</think>B",
                expected: Some("AB"),
            },
            Case {
                name: "empty think block",
                input: "before<think></think>after",
                expected: Some("beforeafter"),
            },
            Case {
                name: "only think tags returns none",
                input: "<think>hidden</think>",
                expected: None,
            },
            Case {
                name: "whitespace only returns none",
                input: "  <think>content</think>  ",
                expected: None,
            },
        ];

        for case in &cases {
            assert_eq!(
                strip_think_tags(case.input).as_deref(),
                case.expected,
                "case: {}",
                case.name
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// What a replay carries as reasoning: the explicit `reasoning` field when
    /// the upstream sent one, else the text of the `reasoning_details` blocks
    /// (encrypted ones say nothing), and on a tool turn an empty string even
    /// when there is nothing to say — OpenRouter sends `reasoning: ""` there and
    /// a replayed tool call without the field is a shape its API rejects.
    #[expect(clippy::type_complexity)] // (label, explicit, details, tool turn, expected)
    #[test]
    fn reasoning_plaintext_is_the_sent_text_and_a_tool_turn_always_replays() {
        let blocks = json!([
            {"type": "reasoning.summary", "summary": "Plan: step A", "format": "x", "index": 0},
            {"type": "reasoning.text", "text": "Details here.", "format": "x", "index": 1}
        ]);
        let plaintext = plaintext_from_reasoning_details(&blocks);
        assert!(plaintext.contains("Plan: step A"));
        assert!(plaintext.contains("Details here."));

        let text_only = json!([{"type": "reasoning.text", "text": "from details", "format": "x"}]);
        let encrypted = json!([{"type": "reasoning.encrypted", "data": "x", "format": "x"}]);
        let nothing = json!([]);
        assert_eq!(
            plaintext_from_reasoning_details(&encrypted),
            "",
            "an encrypted block is not plaintext"
        );

        // (name, explicit, details, is a tool turn, expected plaintext)
        let cases: &[(&str, Option<&str>, Option<&Value>, bool, Option<&str>)] = &[
            (
                "prefers explicit over details",
                Some("explicit"),
                Some(&text_only),
                true,
                Some("explicit"),
            ),
            (
                "derives from details when no explicit",
                None,
                Some(&text_only),
                true,
                Some("from details"),
            ),
            (
                "tool call uses empty string when details are non-textual",
                None,
                Some(&encrypted),
                true,
                Some(""),
            ),
            (
                "no tool does not force empty for encrypted-only",
                None,
                Some(&encrypted),
                false,
                None,
            ),
            (
                "empty explicit on a tool turn still replays",
                Some(""),
                None,
                true,
                Some(""),
            ),
            (
                "whitespace explicit on a tool turn still replays",
                Some("  \t"),
                None,
                true,
                Some(""),
            ),
            (
                "empty explicit on a plain turn replays none",
                Some(""),
                None,
                false,
                None,
            ),
            (
                "empty explicit and empty details on a tool turn still replay",
                Some(""),
                Some(&nothing),
                true,
                Some(""),
            ),
        ];
        for (name, explicit, details, tool, expected) in cases {
            assert_eq!(
                reasoning_plaintext_for_roundtrip(*explicit, *details, *tool).as_deref(),
                *expected,
                "case: {name}"
            );
        }
    }

    /// A replayed assistant turn with tool calls always carries
    /// `reasoning_content`: synthesized from the details when there are any,
    /// empty when there is nothing to say, and absent only when the reasoning
    /// slot itself holds nothing textual — the triple handed to the wire is the
    /// plaintext, the details, and the same plaintext as `reasoning_content`.
    #[test]
    fn assistant_replay_carries_reasoning_content_for_tool_turns() {
        let tc = ToolCall {
            id: "t1".into(),
            name: "x".into(),
            arguments: json!({}),
        };
        let replayed = |reasoning: Option<&Reasoning>| {
            assistant_replay_payload(Some("a"), std::slice::from_ref(&tc), reasoning)
                .get("reasoning_content")
                .and_then(Value::as_str)
                .map(str::to_string)
        };

        assert_eq!(
            replayed(None).as_deref(),
            Some(""),
            "a tool turn with no parsed reasoning still replays the field"
        );

        let details =
            json!([{"type": "reasoning.text", "text": "from details", "format": "x", "index": 0}]);
        assert_eq!(
            replayed(Some(
                &Reasoning::from_optional_parts(None, None, Some(details.clone()))
                    .expect("non-empty reasoning slot")
            ))
            .as_deref(),
            Some("from details"),
            "the details are the text when the model sent no reasoning field"
        );

        assert_eq!(
            replayed(Some(
                &Reasoning::from_optional_parts(Some(String::new()), None, None)
                    .expect("non-empty reasoning slot")
            ))
            .as_deref(),
            Some(""),
            "an empty reasoning string on a tool turn is still the field"
        );

        assert!(
            replayed(Some(
                &Reasoning::from_optional_parts(None, None, Some(json!([])))
                    .expect("reasoning_details")
            ))
            .is_none(),
            "details with no text cannot synthesize content, and the field is not invented"
        );

        // The triple a native replay frame is built from.
        let reasoning = Reasoning::from_optional_parts(Some("openrouter".into()), None, None)
            .expect("reasoning");
        let (r, rc, rd) = native_reasoning_triple_for_replay(Some(&reasoning), true);
        assert_eq!(r.as_deref(), Some("openrouter"));
        assert_eq!(rc.as_deref(), Some("openrouter"));
        assert!(rd.is_none());

        let reasoning =
            Reasoning::from_optional_parts(None, None, Some(details.clone())).expect("reasoning");
        let (r, rc, rd) = native_reasoning_triple_for_replay(Some(&reasoning), true);
        assert!(r.is_none());
        assert_eq!(rc.as_deref(), Some("from details"));
        assert_eq!(rd.as_ref(), Some(&details));
    }
}
