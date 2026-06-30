//! Utility modules for shared helper functions.

pub mod error;
pub mod html;
pub mod http;
pub mod json;
pub mod macros;
#[cfg(test)]
pub mod test;
pub mod tree_sitter;

use directories::UserDirs;
use regex::Regex;
use regex::RegexBuilder;
use serde::de::DeserializeOwned;
use std::path::PathBuf;
use std::sync::LazyLock;

use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::Reasoning;
use serde_json::Value;

/// Extension trait to unwrap poisoned lock results, replacing
/// `.unwrap_or_else(std::sync::PoisonError::into_inner)` with `.unwrap_poison()`.
pub trait UnwrapPoison {
    type Inner;
    /// Unwrap the lock result, recovering the inner value even if the lock is poisoned.
    #[must_use]
    fn unwrap_poison(self) -> Self::Inner;
}

impl<T> UnwrapPoison for Result<T, std::sync::PoisonError<T>> {
    type Inner = T;
    fn unwrap_poison(self) -> T {
        self.unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The regex pattern for `[KIND:path]` media markers.
///
/// This is the single source of truth for the marker pattern. Both the case-sensitive
/// [`MEDIA_MARKER_RE`] and the case-insensitive [`TELEGRAM_MEDIA_MARKER_RE`] are built
/// from this constant, so adding a new marker kind here automatically keeps both in sync.
const MEDIA_MARKER_PATTERN: &str = r"\[(?P<kind>IMAGE|AUDIO|VIDEO):(?P<path>[^\]]+)\]";

/// Matches `[IMAGE:path]`, `[AUDIO:path]`, or `[VIDEO:path]` markers in message content.
/// Path must be non-empty (uses `+` quantifier).
///
/// **Invariant — multimodal stripping:** When enriching messages in multimodal
/// mode, IMAGE markers are preserved (they're needed for vision API integration
/// via `to_message_content()`), while all non-IMAGE markers (AUDIO, VIDEO, and
/// any future marker kinds) are stripped from the content. This is enforced by
/// the marker-stripping logic at the end of `enrich_message` which mirrors the
/// `parse_image_markers()` pattern. Adding a new marker kind to this regex will
/// cause it to be automatically stripped in multimodal mode unless the closure
/// is explicitly updated to preserve it.
pub(crate) static MEDIA_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(MEDIA_MARKER_PATTERN).expect("MEDIA_MARKER_RE must compile"));

/// Case-insensitive variant of [`MEDIA_MARKER_RE`] used by `telegram.rs` to
/// accept `[image:...]`, `[Image:...]`, etc. Built from the same
/// [`MEDIA_MARKER_PATTERN`] constant to stay in sync.
pub(crate) static TELEGRAM_MEDIA_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(MEDIA_MARKER_PATTERN)
        .case_insensitive(true)
        .build()
        .expect("TELEGRAM_MEDIA_MARKER_RE must compile")
});

/// Truncate a string to `max_chars` Unicode characters, appending "…" if truncated.
#[must_use]
pub fn truncate(input: &str, max_chars: usize) -> String {
    match input.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", input[..idx].trim_end()),
        None => input.to_string(),
    }
}

/// Current Unix timestamp in milliseconds since the epoch.
///
/// Returns `0` if the system clock is set before the Unix epoch (January 1, 1970).
///
/// Returns `u64` — sufficient for timestamps up to ~500 million years from now.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Produce a short human-readable summary of tool arguments.
#[must_use]
pub fn summarize_args(args: &serde_json::Value) -> String {
    match args {
        serde_json::Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    let val = match v {
                        serde_json::Value::String(s) => truncate(s, 80),
                        other => truncate(&other.to_string(), 80),
                    };
                    format!("{k}: {val}")
                })
                .collect();
            parts.join(", ")
        }
        other => truncate(&other.to_string(), 120),
    }
}

/// Extract a human-readable message from a panic payload returned by
/// [`catch_unwind`](futures_util::FutureExt::catch_unwind).
#[must_use]
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Attempt to repair malformed JSON using [`jsonrepair_rs::jsonrepair`] then re-parse.
///
/// Heals common LLM JSON formatting issues (single quotes, trailing commas,
/// unquoted keys, Python keywords, etc.).
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

/// Truncate a string to at most `max_bytes` bytes using a head/tail
/// "sandwich" strategy: keeps the first ~2/3 and last ~1/3, inserting an
/// omission marker between them. Returns the input unchanged if it fits
/// within the limit.
///
/// The marker format is `"... (N bytes omitted at {label} truncation)\n"`,
/// where `label` provides context for the truncation (e.g., `"shell output"`,
/// `"tool output"`, `"stderr"`).
///
/// Slicing respects UTF-8 character boundaries via `floor_char_boundary`.
/// An overlap guard is included as defense-in-depth; it only triggers if
/// the head and tail ranges would intersect (impossible under the 2/3 + 1/3
/// split, but guards against future ratio changes).
#[must_use]
pub fn truncate_sandwich(s: &str, max_bytes: usize, label: &str) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head_bytes = max_bytes * 2 / 3;
    let tail_bytes = max_bytes / 3;
    let head_end = s.floor_char_boundary(head_bytes);
    let tail_start = s.floor_char_boundary(s.len().saturating_sub(tail_bytes));
    if head_end < tail_start {
        let omitted = s[head_end..tail_start].len();
        format!(
            "{}... ({} bytes omitted at {label} truncation)\n{}",
            &s[..head_end],
            omitted,
            &s[tail_start..]
        )
    } else {
        // Head and tail would overlap — simple truncation fallback
        let boundary = s.floor_char_boundary(max_bytes);
        let mut out = s[..boundary].to_string();
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("\n... [{label} truncated at {max_bytes} bytes]"),
        );
        out
    }
}

/// Truncate tool output for LLM consumption (delegates to [`truncate_sandwich`]
/// with a 5 000-byte limit). Returns input unchanged if within limit.
#[must_use]
pub fn format_tool_output(output: &str) -> String {
    truncate_sandwich(output, 5_000, "tool output")
}

/// Read a local image file and return a base64 data URI suitable for native
/// multimodal model input (e.g., `data:image/png;base64,...`).
pub(crate) async fn local_image_to_data_uri(path: &std::path::Path) -> anyhow::Result<String> {
    let bytes = tokio::fs::read(path).await?;
    let mime = mime_for_extension(path);
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(&bytes)))
}

/// Load a reference image from disk, validate it does not exceed `max_bytes`,
/// and return a base64 data URI suitable for multimodal model input.
#[allow(clippy::cast_precision_loss)]
pub(crate) async fn load_reference_image(
    path: &std::path::Path,
    max_bytes: u64,
) -> anyhow::Result<String> {
    if !path.exists() {
        anyhow::bail!("Reference image not found: {}", path.display());
    }
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read reference image {}: {e}", path.display()))?;
    if metadata.len() > max_bytes {
        let mb = max_bytes as f64 / (1024.0 * 1024.0);
        anyhow::bail!(
            "Reference image {} is {} bytes, exceeds {:.1} MB limit. \
             Use a smaller or compressed image.",
            path.display(),
            metadata.len(),
            mb,
        );
    }
    local_image_to_data_uri(path).await
}

/// Map a file path's extension to a MIME type string.
pub(crate) fn mime_for_extension(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    }
}

// ── Reasoning plaintext extraction ──

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

/// Redact sensitive values for safe logging. Shows first 4 characters + "***" suffix.
/// Uses char-boundary-safe indexing to avoid panics on multi-byte UTF-8 strings.
static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(token|api[_-]?key|password|secret|user[_-]?key|bearer|credential)["']?\s*[:=]\s*(?:"([^"]{8,})"|'([^']{8,})'|([a-zA-Z0-9_\-\./+=]{8,}))"#).expect("hardcoded regex is valid")
});

/// Scrub credentials from tool output to prevent accidental exfiltration.
/// Replaces known credential patterns with a redacted placeholder while preserving
/// a small prefix for context.
#[must_use]
pub fn scrub_credentials(input: &str) -> String {
    SENSITIVE_KV_REGEX
        .replace_all(input, |caps: &regex::Captures| {
            let full_match = &caps[0];
            let key = &caps[1];
            let val = caps
                .get(2)
                .or(caps.get(3))
                .or(caps.get(4))
                .map_or("", |m| m.as_str());

            // Preserve first 4 chars for context, then redact.
            debug_assert!(val.len() >= 8, "regex guarantees values >= 8 chars");
            let prefix = val
                .char_indices()
                .nth(4)
                .map_or(val, |(byte_idx, _)| &val[..byte_idx]);

            // Determine quote style from which capture group matched the value.
            // Group 2 = double-quoted, Group 3 = single-quoted, Group 4 (else) = unquoted.
            // Using capture-group identity avoids false positives from quotes/apostrophes
            // appearing elsewhere in the match (e.g., a double-quoted key name with a
            // single-quoted value, or an apostrophe in a key like `don't_share`).
            let quote = if caps.get(2).is_some() {
                Some('"')
            } else if caps.get(3).is_some() {
                Some('\'')
            } else {
                None
            };

            let redacted = format!("{prefix}*[REDACTED]");

            if full_match.contains(':') {
                match quote {
                    Some('"') => format!("\"{key}\": \"{redacted}\""),
                    Some('\'') => format!("{key}: '{redacted}'"),
                    _ => format!("{key}: {redacted}"),
                }
            } else {
                match quote {
                    Some('"') => format!("{key}=\"{redacted}\""),
                    Some('\'') => format!("{key}='{redacted}'"),
                    _ => format!("{key}={redacted}"),
                }
            }
        })
        .to_string()
}

/// Resolve the cargo bin directory.
///
/// Resolution order:
/// 1. `$CARGO_HOME/bin` if `CARGO_HOME` environment variable is set and non-empty.
/// 2. `~/.cargo/bin` using `directories::UserDirs`.
/// 3. `None` — no home directory available.
#[must_use]
pub(crate) fn cargo_bin_dir() -> Option<PathBuf> {
    if let Ok(cargo_home) = std::env::var("CARGO_HOME")
        && !cargo_home.is_empty()
    {
        return Some(PathBuf::from(cargo_home).join("bin"));
    }

    let dirs = UserDirs::new()?;
    Some(dirs.home_dir().join(".cargo").join("bin"))
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

#[cfg(test)]
mod truncate_tests {
    use super::*;

    // ── truncate_sandwich: passthrough ────────────────────────────────────

    #[test]
    fn passthrough_under_limit() {
        let input = "hello world";
        let result = truncate_sandwich(input, 5_000, "test");
        assert_eq!(
            result, input,
            "should pass through unchanged when under limit"
        );
    }

    #[test]
    fn passthrough_at_exact_limit() {
        let input = "a".repeat(5_000);
        assert_eq!(input.len(), 5_000);
        let result = truncate_sandwich(&input, 5_000, "test");
        assert_eq!(result, input, "exact limit should pass through unchanged");
    }

    // ── truncate_sandwich: head/tail sandwich ─────────────────────────────

    #[test]
    fn sandwich_just_over_limit() {
        // Input is exactly limit+1 bytes — sandwich marker may add overhead
        // making output longer than input, which is expected for tiny overshoot.
        let input = "x".repeat(5_001);
        let result = truncate_sandwich(&input, 5_000, "test");
        assert!(
            result.starts_with("xxx"),
            "head portion should be preserved"
        );
        assert!(
            result.contains("bytes omitted at test truncation"),
            "should contain the omission marker"
        );
        assert!(result.ends_with('x'), "tail should contain input suffix");
    }

    #[test]
    fn sandwich_large_input() {
        // Input well over the limit — classic head/tail sandwich with label
        let line = "hello world\n".repeat(200_000);
        assert!(line.len() > 1_048_576, "input should exceed 1MB");
        let result = truncate_sandwich(&line, 1_048_576, "output");
        assert!(result.len() < line.len(), "should truncate");
        assert!(
            result.contains("bytes omitted at output truncation"),
            "should contain label in omission marker"
        );
        // Head portion appears
        assert!(
            result.starts_with("hello world"),
            "head should be preserved"
        );
        // Tail portion appears
        let last_line = result.lines().last().unwrap_or("");
        assert_eq!(last_line, "hello world", "tail should be preserved");
    }

    #[test]
    fn sandwich_preserves_utf8_boundaries() {
        // Place a multibyte char ('🐱', 4 bytes) right at the head/tail
        // boundary so it straddles the cut point. floor_char_boundary must
        // back up to the character boundary. Build: 3329 'x's, then 🐱
        // (bytes 3329-3332), then 'y's. head_bytes ≈ 3333, so 🐱 is the
        // last complete char in head. Verify it appears intact.
        let mut input = String::new();
        input.push_str(&"x".repeat(3_329));
        input.push('🐱'); // bytes 3329..=3332
        input.push_str(&"y".repeat(20_000));
        let result = truncate_sandwich(&input, 5_000, "test");
        assert!(
            result.contains('🐱'),
            "multibyte char at boundary should survive intact"
        );
    }

    #[test]
    fn sandwich_line_boundaries_intact() {
        // Lines should not be concatenated across truncation boundaries
        let line = "hello world!\n".repeat(100_000);
        let result = truncate_sandwich(&line, 500_000, "test");
        assert!(result.len() < line.len(), "should truncate");
        for l in result.lines().filter(|l| !l.starts_with("...")) {
            assert!(
                !l.contains("hello world!hello"),
                "lines should not be concatenated"
            );
        }
    }

    // ── truncate_sandwich: custom label ───────────────────────────────────

    #[test]
    fn custom_label_appears_in_marker() {
        let input = "x".repeat(10_000);
        let result = truncate_sandwich(&input, 5_000, "my custom label");
        assert!(
            result.contains("bytes omitted at my custom label truncation"),
            "custom label should appear verbatim in marker"
        );
    }

    #[test]
    fn empty_label() {
        let input = "x".repeat(10_000);
        let result = truncate_sandwich(&input, 5_000, "");
        assert!(
            result.contains("bytes omitted at  truncation"),
            "empty label should still produce coherent marker"
        );
    }

    // ── format_tool_output compatibility ──────────────────────────────────

    #[test]
    fn format_tool_output_delegates_correctly() {
        let input = "abc".repeat(2_000); // 6_000 bytes > 5_000 limit
        let result = format_tool_output(&input);
        assert!(result.len() < input.len(), "should truncate");
        assert!(
            result.contains("bytes omitted at tool output truncation"),
            "should use 'tool output' label"
        );
        assert!(result.starts_with("abcabc"), "head should be preserved");
    }

    #[test]
    fn format_tool_output_passthrough() {
        let input = "short";
        let result = format_tool_output(input);
        assert_eq!(result, input, "short input passes through unchanged");
    }
}

// ── scrub_credentials tests ────────────────────────────────────────────

#[cfg(test)]
mod scrub_tests {
    use super::scrub_credentials;

    #[test]
    fn scrub_redacts_credentials() {
        /// Cases verifying `[REDACTED]` appears, with optional negative and
        /// prefix checks. Fields: (name, input, must_not_contain, must_start_with).
        /// Empty string for must_not_contain/must_start_with = skip check.
        const CASES: &[(&str, &str, &str, &str)] = &[
            (
                "alphanumeric unquoted value",
                "API_KEY=sk-1234567890abcdef",
                "1234567890abcdef",
                "API_KEY=sk-1",
            ),
            // Standard Base64-encoded secret containing +, /, =
            (
                "Base64 unquoted value with plus and slash",
                "api_key=u2FsdGVkX1+h/wZ/L3Y+Q==",
                "u2FsdGVkX1+h/wZ/L3Y+Q==",
                "api_key=u2Fs",
            ),
            (
                "double-quoted value with colon separator",
                r#"token: "abcdefgh1234567890""#,
                "1234567890",
                "",
            ),
            (
                "bearer colon-separated value",
                "bearer: eyJhbGciOiJIUzI1NiJ9",
                "eyJhbG",
                "",
            ),
            // Hyphen-key variant: regex `user[_-]?key` also matches `user-key`.
            (
                "hyphen-key variant",
                "user-key=abcdefgh12345678",
                "12345678",
                "user-key=abcd",
            ),
        ];

        for &(name, input, not_contains, prefix) in CASES {
            let out = scrub_credentials(input);
            assert!(out.contains("[REDACTED]"), "{name}: should redact: {out}");
            if !not_contains.is_empty() {
                assert!(
                    !out.contains(not_contains),
                    "{name}: should not leak value: {out}"
                );
            }
            if !prefix.is_empty() {
                assert!(out.starts_with(prefix), "{name}: should keep prefix: {out}");
            }
        }
    }

    #[test]
    fn scrub_exact_output() {
        /// Cases verifying exact output strings. Exact match is the strictest
        /// assertion — it subsumes containment and non-leakage checks.
        /// Fields: (name, input, expected_output).
        const CASES: &[(&str, &str, &str)] = &[
            // Single quotes must be preserved (the bug this test guards against).
            (
                "single-quoted value with colon separator",
                "password: 's3cr3t_p@ssw0rd!!'",
                "password: 's3cr*[REDACTED]'",
            ),
            (
                "single-quoted value with equals separator",
                "password='mysecretvalue123'",
                "password='myse*[REDACTED]'",
            ),
            // Edge case: the key-level optional quote in the regex can produce
            // full_match containing a double-quote from the key suffix, e.g.
            // "password": 'secretvalue1234'. The capture-group approach correctly
            // identifies this as a single-quoted value despite the double-quote
            // appearing in the full match string.
            // Note: the key-suffix " is consumed by the regex match and not
            // reconstructed — this is a pre-existing cosmetic issue also present
            // in the double-quote path, and out of scope for this fix.
            (
                "double-quoted key with single-quoted value",
                r#""password": 'secretvalue123'"#,
                "\"password: 'secr*[REDACTED]'",
            ),
        ];

        for &(name, input, expected) in CASES {
            assert_eq!(scrub_credentials(input), expected, "{name}");
        }
    }

    #[test]
    fn scrub_passthrough() {
        /// Cases where the input is not a credential pattern and must pass
        /// through unchanged. Fields: (name, input).
        const CASES: &[(&str, &str)] = &[
            ("short unquoted values (under 8 chars)", "key=short"),
            (
                "non-secret lines with = and /",
                "normal line with = equals and / slash",
            ),
        ];

        for &(name, input) in CASES {
            assert_eq!(scrub_credentials(input), input, "{name}");
        }
    }

    #[test]
    fn unix_millis_is_reasonable() {
        // Any reasonable system: timestamp should be > year 2020 (~1_577_836_800_000)
        // and < year 2100 (~4_102_444_800_000) to catch drastic clock skew.
        let ts = super::unix_millis();
        assert!(
            ts > 1_577_836_800_000,
            "unix_millis() seems too small: {ts}"
        );
        assert!(
            ts < 4_102_444_800_000,
            "unix_millis() seems too large: {ts}"
        );
    }
}
