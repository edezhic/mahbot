//! Typed value extraction and LLM-output JSON parsing helpers.
//!
//! # Tool argument extraction
//!
//! The `get_*` functions below are the shared argument-parsing layer for tools
//! (see [`crate::tools`]). They follow the workspace tool-error convention:
//! errors carry a leading lowercase machine-recognizable code token, and for
//! wrong tool arguments that token is `usage`.
//!
//! Semantics:
//! - Absent (or JSON `null`) is *not* an error — helpers with a default or an
//!   `Option` return it, and [`get_str`] reports a distinct "missing" message.
//! - A present value of the wrong type *is* a usage error; helpers never
//!   silently default or silently drop values anymore.
//!
//! The one deliberate exception is [`get_opt_str`], which stays silent
//! (`Option<&str>`) because its callers treat any non-string as absent.
//!
//! # JSON parsing from LLM output
//!
//! Parse/repair functions handle the common case of LLMs emitting JSON inside
//! fenced code blocks with minor formatting issues (trailing commas, unquoted keys,
//! single quotes, etc.).

use serde::de::DeserializeOwned;
use serde_json::Value;

/// Human-readable description of a `serde_json::Value`'s JSON type.
#[must_use]
fn found_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Build a standardized usage error for a wrong-typed tool argument.
///
/// Shared by the `get_*` helpers and by tools that do bespoke optional-field
/// type checks (the deliberately silent [`get_opt_str`] has no error path of
/// its own).
pub(crate) fn wrong_type(key: &str, expected: &str, found_value: &Value) -> anyhow::Error {
    let hint = match expected {
        "a string" => "wrap the value in double quotes".to_string(),
        "a boolean" => {
            "pass unquoted true or false, or omit the argument to use the default".to_string()
        }
        "a non-negative integer" => format!("pass a JSON number, e.g. {key}: 5"),
        "an integer" => "pass a JSON number".to_string(),
        "an array of strings" => {
            format!("pass a JSON array of strings, e.g. {key}: [\"a\", \"b\"]")
        }
        _ => "pass a JSON value of the expected type".to_string(),
    };
    anyhow::anyhow!(
        "usage: argument \"{key}\" must be {expected}, got {} — hint: {hint}",
        found_type(found_value)
    )
}

/// Extract a required string field from JSON args.
///
/// Absent or null is a distinct "missing" usage error; a present non-string is
/// a wrong-type usage error.
pub(crate) fn get_str<'a>(val: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    match val.get(key) {
        None | Some(Value::Null) => Err(anyhow::anyhow!(
            "usage: missing required argument \"{key}\" (expected a string) — \
             hint: pass it as a JSON string, e.g. \"{key}\": \"value\""
        )),
        Some(v) => v.as_str().ok_or_else(|| wrong_type(key, "a string", v)),
    }
}

/// Extract an optional string field from JSON args.
///
/// Deliberately silent: any absent/non-string value maps to [`Option::None`].
pub(crate) fn get_opt_str<'a>(val: &'a Value, key: &str) -> Option<&'a str> {
    val.get(key).and_then(Value::as_str)
}

/// Extract a boolean field with default value.
///
/// Absent or null yields `default`; a present non-boolean is a usage error.
pub(crate) fn get_bool(val: &Value, key: &str, default: bool) -> anyhow::Result<bool> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => Err(wrong_type(key, "a boolean", v)),
    }
}

/// Extract an optional i64 field.
///
/// Absent or null yields [`Option::None`]; a present value outside the i64
/// range (or non-integer) is a usage error.
pub(crate) fn get_opt_i64(val: &Value, key: &str) -> anyhow::Result<Option<i64>> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_i64()
            .map(Some)
            .ok_or_else(|| wrong_type(key, "an integer", v)),
    }
}

/// Extract an optional u64 field.
///
/// Absent or null yields [`Option::None`]; a present negative or non-integer
/// value is a usage error.
pub(crate) fn get_opt_u64(val: &Value, key: &str) -> anyhow::Result<Option<u64>> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| wrong_type(key, "a non-negative integer", v)),
    }
}

/// Extract a usize field with default value.
///
/// Absent or null yields `default`; a present value that is not a u64
/// representable as `usize` is a usage error.
pub(crate) fn get_usize(val: &Value, key: &str, default: usize) -> anyhow::Result<usize> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| wrong_type(key, "a non-negative integer", v))?;
            // Only reachable on 32-bit targets; still reported as a usage error.
            usize::try_from(n).map_err(|_| wrong_type(key, "a non-negative integer", v))
        }
    }
}

/// Extract a string array field as `Vec<String>`.
///
/// Absent or null yields an empty vector; a non-array value or an array with
/// any non-string element is a usage error.
pub(crate) fn get_str_array(val: &Value, key: &str) -> anyhow::Result<Vec<String>> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                let Some(s) = v.as_str() else {
                    return Err(wrong_type(key, "an array of strings", v));
                };
                out.push(s.to_string());
            }
            Ok(out)
        }
        Some(v) => Err(wrong_type(key, "an array of strings", v)),
    }
}

/// Extract an object field, defaulting to an empty object.
///
/// Absent or null yields an empty object (the [`get_str_array`] shape, so
/// callers need no default of their own); a present non-object is a usage error.
pub(crate) fn get_object(val: &Value, key: &str) -> anyhow::Result<serde_json::Map<String, Value>> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(serde_json::Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(v) => Err(wrong_type(key, "an object", v)),
    }
}

/// Extract an optional bool field.
///
/// Absent or null yields [`Option::None`]; a present non-boolean is a usage
/// error.
pub(crate) fn get_opt_bool(val: &Value, key: &str) -> anyhow::Result<Option<bool>> {
    match val.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(wrong_type(key, "a boolean", v)),
    }
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

#[cfg(test)]
mod arg_extraction_tests {
    use super::{
        get_bool, get_opt_bool, get_opt_i64, get_opt_str, get_opt_u64, get_str, get_str_array,
        get_usize,
    };
    use serde_json::{Value, json};

    /// Absent and null both count as "not provided" for defaulted helpers.
    #[test]
    fn absent_and_null_yield_defaults() {
        let absent = json!({});
        let null = json!({ "k": null });

        for (val, label) in [(&absent, "absent"), (&null, "null")] {
            assert!(get_bool(val, "k", true).unwrap(), "{label}");
            assert_eq!(get_usize(val, "k", 7).unwrap(), 7, "{label}");
            assert_eq!(get_opt_i64(val, "k").unwrap(), None, "{label}");
            assert_eq!(get_opt_u64(val, "k").unwrap(), None, "{label}");
            assert_eq!(get_opt_bool(val, "k").unwrap(), None, "{label}");
            assert!(get_str_array(val, "k").unwrap().is_empty(), "{label}");
        }

        // get_opt_str stays silent for absent values.
        assert_eq!(get_opt_str(&absent, "k"), None);
    }

    /// get_str must distinguish "missing" from "wrong type" in its message.
    #[test]
    fn get_str_distinguishes_absent_from_wrong_type() {
        let absent = json!({});
        let err = get_str(&absent, "path").unwrap_err().to_string();
        assert!(err.contains("usage:"), "got: {err}");
        assert!(
            err.contains("missing required argument \"path\""),
            "got: {err}"
        );

        let wrong = json!({ "path": 42 });
        let err = get_str(&wrong, "path").unwrap_err().to_string();
        assert!(err.contains("usage:"), "got: {err}");
        assert!(
            err.contains("argument \"path\" must be a string"),
            "got: {err}"
        );
        assert!(err.contains("got a number"), "got: {err}");

        // null is "missing", not "wrong type".
        let null = json!({ "path": null });
        let err = get_str(&null, "path").unwrap_err().to_string();
        assert!(err.contains("missing required argument"), "got: {err}");

        // Happy path.
        let ok = json!({ "path": "a" });
        assert_eq!(get_str(&ok, "path").unwrap(), "a");
    }

    /// Wrong-typed present values are usage errors naming the field and type.
    #[test]
    fn wrong_type_errors_are_usage_errors() {
        let cases: [(&Value, &str, &str, &str); 5] = [
            (
                &json!({ "k": "yes" }),
                "bool",
                "must be a boolean",
                "got a string",
            ),
            (
                &json!({ "k": "5" }),
                "usize",
                "must be a non-negative integer",
                "got a string",
            ),
            (
                &json!({ "k": "x" }),
                "opt_i64",
                "must be an integer",
                "got a string",
            ),
            (
                &json!({ "k": -1 }),
                "opt_u64",
                "must be a non-negative integer",
                "got a number",
            ),
            (
                &json!({ "k": 1.5 }),
                "opt_bool",
                "must be a boolean",
                "got a number",
            ),
        ];
        for (val, helper, expected_phrase, found_phrase) in cases {
            let err = match helper {
                "bool" => get_bool(val, "k", false).unwrap_err().to_string(),
                "usize" => get_usize(val, "k", 0).unwrap_err().to_string(),
                "opt_i64" => get_opt_i64(val, "k").unwrap_err().to_string(),
                "opt_u64" => get_opt_u64(val, "k").unwrap_err().to_string(),
                "opt_bool" => get_opt_bool(val, "k").unwrap_err().to_string(),
                _ => unreachable!(),
            };
            assert!(err.contains("usage:"), "helper {helper}: got: {err}");
            assert!(err.contains("\"k\""), "helper {helper}: got: {err}");
            assert!(err.contains(expected_phrase), "helper {helper}: got: {err}");
            assert!(err.contains(found_phrase), "helper {helper}: got: {err}");
        }
    }

    /// get_str_array rejects both non-array values and non-string elements.
    #[test]
    fn get_str_array_rejects_non_string_elements() {
        let non_array = json!({ "k": "a, b" });
        let err = get_str_array(&non_array, "k").unwrap_err().to_string();
        assert!(err.contains("usage:"), "got: {err}");
        assert!(
            err.contains("\"k\" must be an array of strings"),
            "got: {err}"
        );

        let mixed = json!({ "k": ["a", 2, "b"] });
        let err = get_str_array(&mixed, "k").unwrap_err().to_string();
        assert!(
            err.contains("\"k\" must be an array of strings"),
            "got: {err}"
        );
        assert!(err.contains("got a number"), "got: {err}");

        let ok = json!({ "k": ["a", "b"] });
        assert_eq!(
            get_str_array(&ok, "k").unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    /// Happy paths: valid values round-trip, including a bool overriding its default.
    #[test]
    fn valid_values_round_trip() {
        let val = json!({
            "flag": false,
            "count": 5,
            "signed": -3,
            "text": "hello",
            "list": ["a"],
        });
        assert!(!get_bool(&val, "flag", true).unwrap());
        assert_eq!(get_usize(&val, "count", 0).unwrap(), 5);
        assert_eq!(get_opt_i64(&val, "signed").unwrap(), Some(-3));
        assert_eq!(get_opt_u64(&val, "count").unwrap(), Some(5));
        assert_eq!(get_opt_bool(&val, "flag").unwrap(), Some(false));
        assert_eq!(get_str_array(&val, "list").unwrap(), vec!["a".to_string()]);
    }
}
