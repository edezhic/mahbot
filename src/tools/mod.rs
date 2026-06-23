//! Tool implementations for agent-callable capabilities.

use anyhow::Context;

pub mod ask;
pub mod browser;
pub mod edit;
pub mod image_gen;
pub mod path;
pub mod read;
pub mod search;
pub mod search_archived_tickets;
pub mod shell;
pub mod ticket;
pub mod video_gen;
pub mod web_search;

/// Maximum file size allowed for read, edit, search tool operations, and the dashboard editor (10 MB).
/// Guards against OOM when agents or the GUI attempt to read very large files.
/// Used directly or via `check_file_size()` by ReadTool, EditTool, SearchTool, and the Iced Editor.
pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum size for a single reference image in bytes.
/// OpenRouter enforces a ~2 MB request body limit; base64 adds ~33% overhead so
/// we cap raw image data at 1.5 MB (1_500_000 bytes) to stay well under.
pub(crate) const MAX_REFERENCE_IMAGE_BYTES: u64 = 1_500_000;

/// Canonical list of argument aliases for file path parameters.
///
/// Tools accept `"file"` and `"filename"` as aliases for the primary `"path"`
/// argument. This constant is the single source of truth for those aliases,
/// used by [`find_path_arg`].
///
/// If a new alias needs to be added, update this list — all path resolution
/// goes through [`find_path_arg`] which picks it up automatically.
pub(crate) const PATH_ALIAS_KEYS: &[&str] = &["file", "filename"];

/// Check that a file's size is within the allowed limit.
/// Returns `Ok(())` or bails with a descriptive error.
pub(crate) fn check_file_size(meta: &std::fs::Metadata) -> anyhow::Result<()> {
    if meta.len() > MAX_FILE_SIZE_BYTES {
        anyhow::bail!(
            "File too large: {} bytes (limit: {} bytes)",
            meta.len(),
            MAX_FILE_SIZE_BYTES
        );
    }
    Ok(())
}

// ── Re-exports ─────────────────────────────────────────────────────────

pub use ask::AskTool;
pub use browser::BrowserTool;
pub use edit::EditTool;
pub use image_gen::ImageGenTool;
pub use read::ReadTool;
pub use search::SearchTool;
pub use search_archived_tickets::SearchArchivedTicketsTool;
pub use shell::{ShellMode, ShellTool};
pub use ticket::{
    AddCommentTool, CreateTicketTool, GetTicketTool, ListTicketsTool, UpdateTicketTool,
};
pub use video_gen::VideoGenTool;
pub use web_search::WebSearchTool;

use crate::{Tool, Workspace};
use std::path::PathBuf;

// ── JSON helpers ─────────────────────────────────────────────────────────

pub(crate) use crate::util::json::{
    get_bool, get_opt_bool, get_opt_i64, get_opt_str, get_opt_u64, get_str, get_str_array,
    get_usize,
};

/// Find the path argument value from tool call arguments, respecting aliases.
///
/// Iterates over the primary `"path"` key first, then [`PATH_ALIAS_KEYS`],
/// returning the value of the first matching key as a string slice. Returns
/// `None` if no matching key is present or the value is not a string.
///
/// This is the borrowed counterpart of [`require_path_arg`] — it returns
/// `Option<&str>` (borrowed, optional) while [`require_path_arg`] returns
/// `Result<String>` (owned, required).
#[must_use]
pub(crate) fn find_path_arg(args: &serde_json::Value) -> Option<&str> {
    std::iter::once("path")
        .chain(PATH_ALIAS_KEYS.iter().copied())
        .find_map(|k| args.get(k))
        .and_then(|v| v.as_str())
}

/// Extract a required `"path"` argument from tool call arguments, respecting aliases.
///
/// Returns the path as an owned `String`, or a descriptive error if no path
/// argument is present (even after checking [`PATH_ALIAS_KEYS`] aliases).
///
/// This is the owned counterpart of [`find_path_arg`] — it returns
/// `Result<String>` (owned, required) while [`find_path_arg`] returns
/// `Option<&str>` (borrowed, optional).
pub(crate) fn require_path_arg(args: &serde_json::Value) -> anyhow::Result<String> {
    find_path_arg(args).map(ToString::to_string).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing required field: 'path'. \
                 Example: {{\"path\": \"src/main.rs\"}}"
        )
    })
}

/// Returns true when the path looks like a glob rather than a literal file path.
#[must_use]
pub(crate) fn path_contains_wildcard(path: &str) -> bool {
    path.contains(['*', '?', '[', ']'])
}

/// Build a JSON schema for tool parameters.
///
/// Wraps `properties` in the standard `{"type": "object", "properties": {...}}`
/// envelope and conditionally adds `"required"` only when the slice is non-empty.
///
/// This eliminates repetitive boilerplate across tool implementations.
/// Tools with non-standard top-level keys (e.g., `oneOf` in WebSearchTool,
/// or BrowserTool's own `action_schema` helper) should not use this.
#[must_use]
pub(crate) fn tool_params_schema(
    properties: &serde_json::Value,
    required: &[&str],
) -> serde_json::Value {
    let mut schema = serde_json::json!({
        "type": "object",
        "properties": properties,
    });
    if !required.is_empty() {
        schema["required"] = serde_json::json!(required);
    }
    schema
}

// ── sanitize ───────────────────────────────────────────────────────────

use regex::Regex;
use std::sync::LazyLock;

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

/// Scrub successful tool output; delegates the scrubbing policy to the tool.
#[must_use]
pub fn sanitize_success_tool_output(
    tool: &dyn Tool,
    call_arguments: &serde_json::Value,
    output: &str,
) -> String {
    if tool.should_scrub_output(call_arguments) {
        scrub_credentials(output)
    } else {
        output.to_string()
    }
}

#[must_use]
pub fn format_tool_failure_feedback(
    tool_name: &str,
    tool_args: &serde_json::Value,
    reason: &str,
) -> String {
    // The `reason` parameter is pre-scrubbed by the caller
    // ([`failure_outcome`](crate::agent::Agent::failure_outcome)) and passed
    // through as-is to avoid double-scrubbing. The `tool_args` are scrubbed
    // here since they're formatted for display.
    let args_preview = scrub_credentials(&crate::util::truncate(&tool_args.to_string(), 1000));
    format!(
        "Tool call failed.\n\
         tool: {tool_name}\n\
         arguments: {args_preview}\n\
         reason:\n{reason}"
    )
}

/// Outcome for a tool execution.
#[derive(Debug, Clone)]
pub struct ToolExecutionOutcome {
    pub output: String,
    pub success: bool,
}

/// Normalize a tool call name and arguments, repairing common agent mistakes.
///
/// Returns `(normalized_name, normalized_args)`. Stats and dispatch should use
/// the normalized values so recovered calls are attributed to the real tool.
#[must_use]
pub fn normalize_tool_call(name: &str, args: serde_json::Value) -> (String, serde_json::Value) {
    let (normalized_name, mut args) = normalize_tool_name(name, args);
    normalize_tool_arguments(&normalized_name, &mut args);
    (normalized_name, args)
}

/// Map known tool-name aliases to their canonical names.
///
/// This is the single source of truth for tool-name normalization, shared by
/// [`normalize_tool_name`] (full call normalization) and [`find_tool`] (direct
/// lookup).  Adding a new alias here immediately affects both paths.
///
/// The `"glob"` alias is included because it resolves to `"search"` regardless
/// of arguments; the parallel `mode:"files"` injection is handled separately
/// in [`normalize_tool_name`] when args are available.
fn normalize_tool_name_str(name: &str) -> &str {
    match name {
        "bash" | "run_terminal_cmd" => "shell",
        "grep" | "rg" | "grep_search" | "glob" => "search",
        "read_file" => "read",
        "str_replace" => "edit",
        _ => name,
    }
}

fn normalize_tool_name(name: &str, mut args: serde_json::Value) -> (String, serde_json::Value) {
    if name == "glob"
        && let Some(obj) = args.as_object_mut()
        && !obj.contains_key("mode")
    {
        obj.insert("mode".to_string(), serde_json::json!("files"));
    }
    let normalized = normalize_tool_name_str(name);
    (normalized.to_string(), args)
}

fn normalize_tool_arguments(name: &str, args: &mut serde_json::Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    match name {
        "shell" => {
            remap_arg_key(obj, "cmd", "command");
            remap_arg_key(obj, "script", "command");
        }
        "get_ticket" | "update_ticket" | "add_comment" => {
            remap_arg_key(obj, "id", "ticket_id");
            remap_arg_key(obj, "ticket", "ticket_id");
        }
        "edit" => {
            remap_arg_key(obj, "old_str", "old_string");
            remap_arg_key(obj, "new_str", "new_string");
        }
        _ => {}
    }
}

/// Move `from` → `to` only when the canonical key is absent.
fn remap_arg_key(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if !obj.contains_key(to)
        && let Some(v) = obj.remove(from)
    {
        obj.insert(to.to_string(), v);
    }
}

/// Look up a tool by name in a slice of boxed `dyn Tool` values.
///
/// Tool-name aliases are resolved via `normalize_tool_name_str` so that all
/// callers benefit from the same alias mapping.  Prefer [`normalize_tool_call`]
/// before dispatch when full argument normalization is also desired.
#[must_use]
pub fn find_tool<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> Option<&'a dyn Tool> {
    let normalized = normalize_tool_name_str(name);
    tools
        .iter()
        .find(|t| t.name() == normalized)
        .map(Box::as_ref)
}

/// User-facing reason when no static tool matches `call_name`.
#[must_use]
pub fn unknown_tool_message(call_name: &str) -> String {
    format!("Unknown tool: {call_name}")
}

/// Save generated media bytes to `workspace/generated/{prefix}_{timestamp}.{ext}`.
///
/// Creates the `generated/` directory if needed, generates a millisecond-precision
/// timestamp, writes the file, and returns the full `PathBuf`.
///
/// # Security note
/// This function deliberately bypasses path security (no `resolve_write_target`
/// check) because `generated/` is an ephemeral tool-owned directory within the
/// workspace. Do not use this function for user-uploaded or arbitrary content.
pub(crate) async fn save_generated_file(
    ws: &Workspace,
    bytes: &[u8],
    prefix: &str,
    ext: &str,
) -> anyhow::Result<PathBuf> {
    let generated_dir = ws.as_path().join("generated");
    tokio::fs::create_dir_all(&generated_dir)
        .await
        .with_context(|| {
            format!(
                "Failed to create generated directory at {}",
                generated_dir.display()
            )
        })?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let output_path = generated_dir.join(format!("{prefix}_{timestamp}.{ext}"));

    tokio::fs::write(&output_path, bytes)
        .await
        .with_context(|| {
            format!(
                "Failed to write generated file to {}",
                output_path.display()
            )
        })?;

    Ok(output_path)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;
    use crate::ToolSpec;
    use tempfile::TempDir;

    // ── ToolSpec serde ───────────────────────────────────────────

    #[test]
    fn tool_spec_serde_roundtrip() {
        let spec = ToolSpec {
            name: "test".into(),
            description: "A test tool".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let parsed: ToolSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(parsed.name, "test");
    }

    // ── find_tool aliases ──────────────────────────────────────────

    #[test]
    fn find_tool_aliases() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(SearchTool),
            Box::new(ShellTool::new(ShellMode::Full)),
            Box::new(ReadTool),
            Box::new(EditTool),
        ];

        // Each case: (alias, expected_tool_name or None).
        let cases: &[(&str, Option<&str>)] = &[
            // Canonical names
            ("search", Some("search")),
            ("shell", Some("shell")),
            ("read", Some("read")),
            ("edit", Some("edit")),
            // Shell aliases
            ("bash", Some("shell")),
            ("run_terminal_cmd", Some("shell")),
            // Search aliases
            ("grep", Some("search")),
            ("rg", Some("search")),
            ("grep_search", Some("search")),
            ("glob", Some("search")),
            // Read aliases
            ("read_file", Some("read")),
            // Edit aliases
            ("str_replace", Some("edit")),
            // Unknown tool
            ("unknown", None),
        ];

        for &(input, expected) in cases {
            let found = find_tool(&tools, input);
            assert_eq!(found.map(Tool::name), expected, "find_tool({input:?})");
        }
    }

    #[test]
    fn path_contains_wildcard_detects_globs() {
        assert!(path_contains_wildcard("src/*.rs"));
        assert!(path_contains_wildcard("lib?.rs"));
        assert!(!path_contains_wildcard("src/main.rs"));
    }

    #[test]
    fn normalize_tool_call_repairs_names_and_args() {
        let (name, args) = normalize_tool_call("bash", serde_json::json!({"cmd": "echo hi"}));
        assert_eq!(name, "shell");
        assert_eq!(args["command"], "echo hi");

        let (name, args) = normalize_tool_call("glob", serde_json::json!({"query": "main.rs"}));
        assert_eq!(name, "search");
        assert_eq!(args["mode"], "files");

        let (name, args) = normalize_tool_call("get_ticket", serde_json::json!({"id": "mahbot-1"}));
        assert_eq!(name, "get_ticket");
        assert_eq!(args["ticket_id"], "mahbot-1");
    }

    #[test]
    // ── media_marker coverage ────────────────────────────────────────
    fn all_media_tools_implement_media_marker() {
        /// Assert that a tool returns Some from media_marker().
        macro_rules! assert_media_marker {
            ($tool:expr) => {
                assert!(
                    $tool.media_marker().is_some(),
                    "{} should return Some from media_marker()",
                    $tool.name(),
                );
            };
        }

        assert_media_marker!(ImageGenTool);
        assert_media_marker!(VideoGenTool);
    }

    // ── PATH_ALIAS_KEYS regression tests ──────────────────────────────

    /// Direct tests for [`require_path_arg`].
    /// Covers alias resolution at the API boundary (owned `Result<String>`
    /// with descriptive error), building on the [`find_path_arg`] tests
    /// which cover the borrowed `Option<&str>` path already.
    #[test]
    fn require_path_arg_resolves_aliases() {
        // "path" key works directly
        assert_eq!(
            require_path_arg(&serde_json::json!({"path": "src/main.rs"})).unwrap(),
            "src/main.rs"
        );
        // Falls back to "file" alias
        assert_eq!(
            require_path_arg(&serde_json::json!({"file": "lib.rs"})).unwrap(),
            "lib.rs"
        );
        // Falls back to "filename" alias
        assert_eq!(
            require_path_arg(&serde_json::json!({"filename": "src/lib.rs"})).unwrap(),
            "src/lib.rs"
        );
        // "path" takes priority over "file"
        assert_eq!(
            require_path_arg(&serde_json::json!({"path": "main.rs", "file": "other.rs"})).unwrap(),
            "main.rs"
        );
        // "file" takes priority over "filename"
        assert_eq!(
            require_path_arg(&serde_json::json!({"file": "a.rs", "filename": "b.rs"})).unwrap(),
            "a.rs"
        );
        // Missing path → descriptive error mentioning 'path'
        let err = require_path_arg(&serde_json::json!({"other": "value"})).unwrap_err();
        assert!(err.to_string().contains("path"));
        // Non-string "path" exists as a key — no fallthrough to aliases, returns error
        let err = require_path_arg(&serde_json::json!({"path": ["invalid"], "file": "real.rs"}))
            .unwrap_err();
        assert!(err.to_string().contains("path"));
    }

    // ── scrub_credentials tests ────────────────────────────────────────────

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

    // ── save_generated_file tests ──────────────────────────────────────────

    #[tokio::test]
    async fn save_generated_file_creates_file() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = Workspace {
            name: "test".into(),
            path: tmp.path().to_string_lossy().to_string(),
            status: "ready".into(),
            created_at: String::new(),
            updated_at: String::new(),
            maintenance: false,
            paused: false,
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            diagnostics_updated_at: None,
        };

        let data = b"hello world";
        let path = save_generated_file(&ws, data, "img", "png")
            .await
            .expect("save_generated_file should succeed");

        assert!(path.exists(), "file should exist: {}", path.display());
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "hello world");

        // Verify filename format: {prefix}_{timestamp}.{ext}
        let file_name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            file_name.starts_with("img_"),
            "filename should start with 'img_': {file_name}",
        );
        assert!(
            std::path::Path::new(file_name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png")),
            "filename should end with '.png': {file_name}",
        );

        let _ = tokio::fs::remove_dir_all(tmp.path()).await;
    }

    #[tokio::test]
    async fn save_generated_file_creates_directory_if_missing() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = Workspace {
            name: "test".into(),
            path: tmp.path().join("nested").to_string_lossy().to_string(),
            status: "ready".into(),
            created_at: String::new(),
            updated_at: String::new(),
            maintenance: false,
            paused: false,
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            diagnostics_updated_at: None,
        };

        let data = b"test content";
        let path = save_generated_file(&ws, data, "vid", "mp4")
            .await
            .expect("save_generated_file should create dirs");

        assert!(path.exists(), "file should exist: {}", path.display());
        assert!(
            path.starts_with(tmp.path().join("nested")),
            "file should be inside workspace"
        );

        let _ = tokio::fs::remove_dir_all(tmp.path()).await;
    }
}
