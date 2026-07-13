//! JSON helper functions for extracting typed values from `serde_json::Value`
//! and parsing JSON from LLM output (including markdown-fenced blocks with repair).
//!
//! Value extraction functions operate on `&serde_json::Value` and a string key,
//! providing convenient access to commonly-needed extraction patterns used throughout
//! the codebase — particularly in tool argument parsing.
//!
//! Parse/repair functions handle the common case of LLMs emitting JSON inside
//! fenced code blocks with minor formatting issues (trailing commas, unquoted keys,
//! single quotes, etc.).

use serde::de::DeserializeOwned;
use serde_json::Value;

/// Extract a required string field from JSON args, returning an error if missing.
pub(crate) fn get_str<'a>(val: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    val.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("Missing required field: {key}"))
}

/// Extract an optional string field from JSON args.
pub(crate) fn get_opt_str<'a>(val: &'a Value, key: &str) -> Option<&'a str> {
    val.get(key).and_then(Value::as_str)
}

/// Extract a boolean field with default value.
pub(crate) fn get_bool(val: &Value, key: &str, default: bool) -> bool {
    val.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Extract an optional i64 field.
pub(crate) fn get_opt_i64(val: &Value, key: &str) -> Option<i64> {
    val.get(key).and_then(Value::as_i64)
}

/// Extract an optional u64 field.
pub(crate) fn get_opt_u64(val: &Value, key: &str) -> Option<u64> {
    val.get(key).and_then(Value::as_u64)
}

/// Extract a usize field with default value.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn get_usize(val: &Value, key: &str, default: usize) -> usize {
    val.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |v| v as usize)
}

/// Extract a string array field as `Vec<String>`.
pub(crate) fn get_str_array(val: &Value, key: &str) -> Vec<String> {
    val.get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract an optional bool field.
pub(crate) fn get_opt_bool(val: &Value, key: &str) -> Option<bool> {
    val.get(key).and_then(Value::as_bool)
}

// ── JSON parsing from LLM output ─────────────────────────────────────────

/// Attempt to repair malformed JSON using [`jsonrepair_rs::jsonrepair`] then re-parse.
///
/// Returns `None` if either the repair or the re-parse fails.
#[must_use]
pub(crate) fn try_repair_json<T: DeserializeOwned>(s: &str) -> Option<T> {
    jsonrepair_rs::jsonrepair(s)
        .ok()
        .and_then(|repaired| serde_json::from_str(&repaired).ok())
}

/// Parse a JSON value from text that may be markdown-fenced.
///
/// Supports ` ```json ... ``` ` blocks, generic ` ``` ... ``` ` blocks,
/// and bare JSON objects. Generic over `T: DeserializeOwned` so callers
/// can deserialize directly into their target type.
///
/// On parse failure, attempts [`try_repair_json`] to heal
/// common LLM JSON formatting issues (single quotes, trailing commas,
/// unquoted keys, Python keywords, etc.) before retrying.
pub(crate) fn parse_fenced_json<T: DeserializeOwned>(text: &str) -> anyhow::Result<T> {
    let trimmed = text.trim();

    // Try markdown-fenced json block first — search anywhere in the text.
    // json-tagged fence checked before bare fence to prefer language-tagged blocks.
    let json_str = if let Some(start) = trimmed.find("```json") {
        extract_fenced_content(&trimmed[start + 7..])
    } else if let Some(start) = trimmed.find("```") {
        extract_fenced_content(&trimmed[start + 3..])
    } else {
        trimmed
    };

    serde_json::from_str::<T>(json_str).or_else(|parse_err| {
        // Attempt JSON repair before giving up
        if let Some(value) = try_repair_json::<T>(json_str) {
            tracing::warn!(
                original_error = %parse_err,
                "Repaired malformed JSON in fenced extraction"
            );
            return Ok(value);
        }
        Err(anyhow::anyhow!("Failed to parse JSON: {parse_err}"))
    })
}

/// Extract content between an opening fence and a closing ` ``` `.
///
/// `text` should be the portion of input immediately after the opening fence marker.
/// Returns the trimmed text up to (but not including) the closing fence.
fn extract_fenced_content(text: &str) -> &str {
    let end = text.find("```").unwrap_or(text.len());
    text.get(..end).unwrap_or(text).trim()
}

#[cfg(test)]
mod tests {
    use super::parse_fenced_json;
    use crate::Verdict;

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct TestVerdict {
        score: u8,
        #[serde(default)]
        critique: String,
        #[serde(default)]
        issues: Vec<String>,
    }

    // ── parse_fenced_json tests ──────────────────────────────────────────

    #[test]
    fn parse_fenced_json_valid_inputs() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected_score: u8,
            expected_critique: &'static str,
        }

        let cases = [
            Case {
                name: "json-tagged fence",
                input: "Based on the analysis, here's my verdict:\n\n```json\n{\"score\": 8, \"critique\": \"Looks good\"}\n```",
                expected_score: 8,
                expected_critique: "Looks good",
            },
            Case {
                name: "bare fence",
                input: "```\n{\"score\": 7, \"critique\": \"Some issues\"}\n```",
                expected_score: 7,
                expected_critique: "Some issues",
            },
            Case {
                name: "unfenced",
                input: r#"{"score": 10, "critique": "Perfect", "issues": []}"#,
                expected_score: 10,
                expected_critique: "Perfect",
            },
            Case {
                name: "commentary before fence",
                input: "I have reviewed the code.\n\n```json\n{\"score\": 6, \"critique\": \"Needs improvement\"}\n```\n\nOverall, acceptable.",
                expected_score: 6,
                expected_critique: "Needs improvement",
            },
            Case {
                name: "multiple fences uses first json",
                input: "```json\n{\"score\": 9}\n```\n\nSome text\n\n```\n{\"score\": 5}\n```",
                expected_score: 9,
                expected_critique: "",
            },
        ];

        for case in &cases {
            let result: TestVerdict = parse_fenced_json(case.input).unwrap();
            assert_eq!(result.score, case.expected_score, "case: {}", case.name);
            assert_eq!(
                result.critique, case.expected_critique,
                "case: {}",
                case.name
            );
        }
    }

    #[test]
    fn parse_fenced_json_with_issues() {
        let text = r#"```json
{"score": 5, "critique": "Problems found", "issues": ["Bug in edge case", "Missing error handling"]}
```"#;
        let result: Verdict = parse_fenced_json(text).unwrap();
        assert_eq!(result.score, 5);
        assert_eq!(result.critique.as_deref(), Some("Problems found"));
        assert_eq!(result.issues_detected.len(), 2);
        assert!(
            result
                .issues_detected
                .contains(&"Bug in edge case".to_string())
        );
    }

    #[test]
    fn parse_fenced_json_invalid_json_returns_err() {
        let text = "```json\n{invalid: true}\n```";
        let result = parse_fenced_json::<Verdict>(text);
        assert!(result.is_err());
    }

    #[test]
    fn parse_fenced_json_no_json_at_all() {
        let text = "This is just plain text with no JSON whatsoever.";
        let result = parse_fenced_json::<Verdict>(text);
        assert!(result.is_err());
    }
}
