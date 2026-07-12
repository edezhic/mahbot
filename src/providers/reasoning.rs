//! Reasoning / chain-of-thought plaintext extraction helpers.
//!
//! These functions process model-provided reasoning data — OpenRouter-style
//! `reasoning_details` JSON, inline `<think>...</think>` tags, and merged
//! `reasoning_content` / `reasoning` strings — into human-readable plaintext
//! for display in the UI or for API roundtrip synthesis.

use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

use crate::Reasoning;

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
pub(crate) fn plaintext_from_reasoning_details(details: &Value) -> String {
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
pub(crate) fn merged_reasoning_string(
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
pub fn plaintext_for_display(reasoning: Option<&Reasoning>) -> Option<String> {
    let r = reasoning?;
    merged_reasoning_string(r.reasoning_content.clone(), r.reasoning.clone()).or_else(|| {
        r.reasoning_details.as_ref().and_then(|d| {
            let s = plaintext_from_reasoning_details(d);
            (!s.trim().is_empty()).then_some(s)
        })
    })
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
