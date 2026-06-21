//! Shared helpers for preserving thinking/reasoning across OpenAI-compatible APIs.
//! `OpenRouter` uses `reasoning` and `reasoning_details`; DeepSeek-native uses `reasoning_content`.

use crate::{Reasoning, ToolCall};
use serde_json::{Value, json};

/// True when `reasoning_details` carries at least one block worth replaying (non-empty array, etc.).
pub(crate) fn details_has_preservable_blocks(details: &Value) -> bool {
    match details {
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
        _ => false,
    }
}

fn flattened_detail_items(patches: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for patch in patches {
        match patch {
            Value::Null => {}
            Value::Array(items) => out.extend(items.clone()),
            other => out.push(other.clone()),
        }
    }
    out
}

/// Try to merge `item` into the last element of `items` when both share the
/// same type and carry the named `field`. Returns `true` when the item was
/// consumed (either merged in or pushed).
fn try_merge_detail_field(
    items: &mut Vec<Value>,
    item: Value,
    ty: &str,
    field: &str,
    join: &str,
) -> bool {
    let Some(value) = item.get(field).and_then(Value::as_str) else {
        items.push(item);
        return true;
    };
    if let Some(last) = items.last_mut()
        && last.get("type").and_then(Value::as_str) == Some(ty)
        && let Some(existing) = last.get(field).and_then(Value::as_str)
    {
        let mut merged = existing.to_string();
        if !join.is_empty() && !merged.is_empty() && !value.is_empty() {
            merged.push_str(join);
        }
        merged.push_str(value);
        *last.get_mut(field).expect("field exists") = Value::String(merged);
    } else {
        items.push(item);
    }
    true
}

fn merge_adjacent_textual_detail_items(items: Vec<Value>) -> Vec<Value> {
    // Fields eligible for consecutive merging, with their join separator.
    // A newline between consecutive summaries improves readability.
    const MERGE_SPECS: &[(&str, &str)] = &[("text", ""), ("summary", "\n")];

    let mut out: Vec<Value> = Vec::new();

    'outer: for item in items {
        let Some(ty) = item.get("type").and_then(Value::as_str).map(str::to_string) else {
            out.push(item);
            continue;
        };

        for &(field, join) in MERGE_SPECS {
            if ty.contains(field) {
                try_merge_detail_field(&mut out, item, &ty, field, join);
                continue 'outer;
            }
        }

        out.push(item);
    }

    out
}

/// Plain `reasoning_content` string to store on assistant messages and send back to the API.
///
/// When the model only streams structured `reasoning_details` (no `reasoning` / `reasoning_content`
/// string), `DeepSeek` behind `OpenRouter` still expects a `reasoning_content` field on **tool-call**
/// turns — derive text from details or send an empty string when details exist but are not textual,
/// or when the API returns an **empty** `reasoning` string (schema placeholder with no `CoT` yet).
pub fn reasoning_plaintext_for_roundtrip(
    explicit: Option<&str>,
    details: Option<&Value>,
    has_tool_calls: bool,
) -> Option<String> {
    if let Some(s) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        return Some(s.to_string());
    }
    if let Some(d) = details {
        let extracted = crate::util::plaintext_from_reasoning_details(d);
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

/// After extracting reasoning fields, fill `reasoning_content` when missing.
///
/// `DeepSeek` (and some OpenRouter-routed thinking models) reject follow-up requests unless prior
/// assistant turns echo chain-of-thought in `reasoning_content`. We derive it from `reasoning`
/// and/or `reasoning_details` using [`reasoning_plaintext_for_roundtrip`].
pub(crate) fn augment_native_reasoning_triple_for_replay(
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    reasoning_details: Option<Value>,
    has_tool_calls: bool,
) -> (Option<String>, Option<String>, Option<Value>) {
    if reasoning_content.is_some() {
        return (reasoning, reasoning_content, reasoning_details);
    }
    let synthesized = reasoning_plaintext_for_roundtrip(
        reasoning.as_deref(),
        reasoning_details.as_ref(),
        has_tool_calls,
    );
    (reasoning, synthesized, reasoning_details)
}

pub(crate) fn native_reasoning_triple_for_replay(
    reasoning: Option<&Reasoning>,
    has_tool_calls: bool,
) -> (Option<String>, Option<String>, Option<Value>) {
    augment_native_reasoning_triple_for_replay(
        reasoning.and_then(|r| r.reasoning.clone()),
        reasoning.and_then(|r| r.reasoning_content.clone()),
        reasoning.and_then(|r| r.reasoning_details.clone()),
        has_tool_calls,
    )
}

/// Coalesce streamed `reasoning_details` patches into one value suitable for storage/replay.
///
/// Preserves append-only ordering; only merges *consecutive* textual blocks of the same type
/// to avoid token-level patch explosion (same logical sequence per `OpenRouter` guidance).
#[must_use]
pub fn coalesce_streamed_reasoning_details(patches: &[Value]) -> Option<Value> {
    let items = merge_adjacent_textual_detail_items(flattened_detail_items(patches));
    (!items.is_empty()).then_some(Value::Array(items))
}

/// Append streaming `reasoning_details` deltas (arrays or single objects) in order.
pub fn merge_reasoning_details_delta(into: &mut Vec<serde_json::Value>, patch: serde_json::Value) {
    match patch {
        serde_json::Value::Array(items) => into.extend(items),
        serde_json::Value::Null => {}
        other => into.push(other),
    }
}

/// `reasoning_details` when present (opaque JSON).
pub(crate) fn json_reasoning_details(value: &Value) -> Option<Value> {
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
    let Some(reasoning_content) =
        reasoning_plaintext_for_roundtrip(explicit, details.as_ref(), has_tool_calls)
    else {
        return;
    };

    if let Some(value) = payload.as_object_mut() {
        value.insert(
            "reasoning_content".to_string(),
            serde_json::Value::String(reasoning_content),
        );
    }
}

/// Build assistant replay payload with reasoning fields preserved for roundtrip.
pub fn assistant_replay_payload(
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
        augment_replay_payload_reasoning_content(&mut payload, !tool_calls.is_empty());
    } else if !tool_calls.is_empty() {
        // Provider may send `reasoning: null` on tool turns; nothing parses into [`Reasoning`], but
        // DeepSeek still expects `reasoning_content` on the wire for replay.
        if !payload
            .get("reasoning_content")
            .is_some_and(Value::is_string)
            && let Some(obj) = payload.as_object_mut()
        {
            obj.insert(
                "reasoning_content".to_string(),
                Value::String(String::new()),
            );
        }
    }

    payload
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plaintext_from_reasoning_details_behavior() {
        let d = json!([
            {"type": "reasoning.summary", "summary": "Plan: step A", "format": "x", "index": 0},
            {"type": "reasoning.text", "text": "Details here.", "format": "x", "index": 1}
        ]);
        let s = crate::util::plaintext_from_reasoning_details(&d);
        assert!(s.contains("Plan: step A"));
        assert!(s.contains("Details here."));
        // Encrypted blocks are skipped
        let d = json!([{"type": "reasoning.encrypted", "data": "abc", "format": "x", "index": 0}]);
        assert_eq!(crate::util::plaintext_from_reasoning_details(&d), "");
    }

    #[test]
    fn reasoning_plaintext_for_roundtrip_scenarios() {
        // prefers explicit over details
        let d = json!([{"type": "reasoning.text", "text": "from details", "format": "x"}]);
        assert_eq!(
            reasoning_plaintext_for_roundtrip(Some("explicit"), Some(&d), true).as_deref(),
            Some("explicit")
        );
        // derives from details when no explicit
        assert_eq!(
            reasoning_plaintext_for_roundtrip(None, Some(&d), true).as_deref(),
            Some("from details")
        );
        // tool_call uses empty string when details are non-textual
        let d = json!([{"type": "reasoning.encrypted", "data": "x", "format": "x"}]);
        assert_eq!(
            reasoning_plaintext_for_roundtrip(None, Some(&d), true).as_deref(),
            Some("")
        );
        // no tool does not force empty for encrypted-only
        assert!(reasoning_plaintext_for_roundtrip(None, Some(&d), false).is_none());
        // OpenRouter may send reasoning: "" on tool turns — still synthesize reasoning_content for replay
        assert_eq!(
            reasoning_plaintext_for_roundtrip(Some(""), None, true).as_deref(),
            Some("")
        );
        assert_eq!(
            reasoning_plaintext_for_roundtrip(Some("  \t"), None, true).as_deref(),
            Some("")
        );
        assert!(reasoning_plaintext_for_roundtrip(Some(""), None, false).is_none());
        // empty explicit + empty details array + tool: still replay placeholder
        assert_eq!(
            reasoning_plaintext_for_roundtrip(Some(""), Some(&json!([])), true).as_deref(),
            Some("")
        );
    }

    #[test]
    fn coalesce_streamed_reasoning_details_merges_textual_blocks() {
        let patches = vec![
            json!({"type": "reasoning.text", "text": "Now", "format": "x", "index": 0}),
            json!({"type": "reasoning.text", "text": " let", "format": "x", "index": 1}),
            json!({"type": "reasoning.text", "text": " me", "format": "x", "index": 2}),
        ];
        let details = coalesce_streamed_reasoning_details(&patches).expect("reasoning_details");

        let arr = details.as_array().cloned().unwrap_or_default();
        assert_eq!(arr.len(), 1);
        assert_eq!(
            arr[0].get("text").and_then(Value::as_str),
            Some("Now let me")
        );
    }

    #[test]
    fn assistant_replay_payload_status() {
        // OpenRouter+deepseek + tool calls + no parsed Reasoning → synthesize empty reasoning_content
        let tc = ToolCall {
            id: "t1".into(),
            name: "x".into(),
            arguments: json!({}),
        };
        let payload = assistant_replay_payload(Some("a"), std::slice::from_ref(&tc), None);
        assert_eq!(
            payload.get("reasoning_content").and_then(Value::as_str),
            Some("")
        );
        // Synthesized from details
        let details =
            json!([{"type": "reasoning.text", "text": "from details", "format": "x", "index": 0}]);
        let payload = assistant_replay_payload(
            Some("a"),
            std::slice::from_ref(&tc),
            Some(&Reasoning::from_optional_parts(None, None, Some(details)).unwrap()),
        );
        assert_eq!(
            payload.get("reasoning_content").and_then(Value::as_str),
            Some("from details")
        );
        // empty reasoning string on tool turn → synthesized reasoning_content for DeepSeek/OpenRouter replay
        let payload = assistant_replay_payload(
            Some("a"),
            std::slice::from_ref(&tc),
            Some(
                &Reasoning::from_optional_parts(Some(String::new()), None, None)
                    .expect("non-empty reasoning slot"),
            ),
        );
        assert_eq!(
            payload.get("reasoning_content").and_then(Value::as_str),
            Some("")
        );
        // Reasoning value present but only empty reasoning_details — augment cannot synthesize
        // content, and the `reasoning: None` fallback is not reached, so no reasoning_content is set.
        let payload = assistant_replay_payload(
            Some("a"),
            std::slice::from_ref(&tc),
            Some(
                &Reasoning::from_optional_parts(None, None, Some(json!([])))
                    .expect("reasoning_details"),
            ),
        );
        assert!(
            payload
                .get("reasoning_content")
                .and_then(Value::as_str)
                .is_none()
        );
    }

    #[test]
    fn json_lossless_reads_empty_reasoning_content() {
        let msg = json!({"content": "x", "reasoning_content": ""});
        let (r, rc, rd) = json_lossless_assistant_reasoning_fields(&msg);
        assert!(r.is_none());
        assert_eq!(rc.as_deref(), Some(""));
        assert!(rd.is_none());
    }

    #[test]
    fn augment_fills_reasoning_content_from_reasoning_or_details() {
        let d = json!([{"type": "reasoning.text", "text": "x", "format": "f", "index": 0}]);
        let (r, rc, rd) =
            augment_native_reasoning_triple_for_replay(Some("openrouter".into()), None, None, true);
        assert_eq!(r.as_deref(), Some("openrouter"));
        assert_eq!(rc.as_deref(), Some("openrouter"));
        assert!(rd.is_none());

        let (r, rc, rd) =
            augment_native_reasoning_triple_for_replay(None, None, Some(d.clone()), true);
        assert!(r.is_none());
        assert_eq!(rc.as_deref(), Some("x"));
        assert_eq!(rd.as_ref(), Some(&d));
    }
}
