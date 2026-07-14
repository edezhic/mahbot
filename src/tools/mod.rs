//! Tool implementations for agent-callable capabilities.

use anyhow::Context;
use tracing::debug;

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
/// Used by ReadTool and EditTool via `check_file_size()`;
/// SearchTool and the Iced Editor use `MAX_FILE_SIZE_BYTES` directly.
pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum size for a single reference image in bytes.
/// OpenRouter enforces a ~2 MB request body limit; base64 adds ~33% overhead so
/// we cap raw image data at 1.5 MB (1_500_000 bytes) to stay well under.
const MAX_REFERENCE_IMAGE_BYTES: u64 = 1_500_000;

/// Canonical list of argument aliases for file path parameters.
///
/// Tools accept `"file"` and `"filename"` as aliases for the primary `"path"`
/// argument. This constant is the single source of truth for those aliases,
/// used by [`normalize_tool_arguments`] to remap to the canonical `"path"` key.
///
/// If a new alias needs to be added, update this list — the remapping is
/// applied unconditionally to all tools, so no per-tool match arm is needed.
const PATH_ALIAS_KEYS: &[&str] = &["file", "filename"];

/// Check that a file's size is within the allowed limit.
/// Returns `Ok(())` or bails with a descriptive error.
fn check_file_size(meta: &std::fs::Metadata) -> anyhow::Result<()> {
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

pub use ask::{AskTool, DispatchMode};
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
pub use web_search::{WebSearchBackend, WebSearchTool};

use crate::{Tool, Workspace};
use std::path::PathBuf;

// ── JSON helpers ─────────────────────────────────────────────────────────

use crate::util::json::{
    get_bool, get_opt_bool, get_opt_i64, get_opt_str, get_opt_u64, get_str, get_str_array,
    get_usize,
};

/// Find the path argument value from tool call arguments.
///
/// Returns the value of the `"path"` key as a string slice, or `None` if
/// the key is missing or the value is not a string.
///
/// Note: argument aliases (`file`, `filename`) are expected to be remapped
/// to `path` before reaching this function — callers should normalize via
/// [`normalize_tool_call`] first.
///
/// This is the borrowed counterpart of [`require_path_arg`] — it returns
/// `Option<&str>` (borrowed, optional) while [`require_path_arg`] returns
/// `Result<String>` (owned, required).
#[must_use]
fn find_path_arg(args: &serde_json::Value) -> Option<&str> {
    args.get("path")?.as_str()
}

/// Extract a required `"path"` argument from tool call arguments.
///
/// Returns the path as an owned `String`, or a descriptive error if no `"path"`
/// argument is present.
///
/// Note: argument aliases (`file`, `filename`) must be remapped to `path`
/// before calling this function — see [`normalize_tool_call`].
///
/// This is the owned counterpart of [`find_path_arg`] — it returns
/// `Result<String>` (owned, required) while [`find_path_arg`] returns
/// `Option<&str>` (borrowed, optional).
fn require_path_arg(args: &serde_json::Value) -> anyhow::Result<String> {
    find_path_arg(args).map(ToString::to_string).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing required field: 'path'. \
                     Example: {{\"path\": \"src/main.rs\"}}"
        )
    })
}

/// Returns true when the path looks like a glob rather than a literal file path.
#[must_use]
fn path_contains_wildcard(path: &str) -> bool {
    path.contains(['*', '?', '[', ']'])
}

/// Build a JSON schema for tool parameters.
///
/// Wraps `properties` in the standard `{"type": "object", "properties": {...}}`
/// envelope and conditionally adds `"required"` only when the slice is non-empty.
///
/// This eliminates repetitive boilerplate across tool implementations.
/// Tools with non-standard top-level keys in their top-level schema
/// (e.g., `oneOf` in WebSearchTool) should not use this directly;
/// they may still use it internally as a building block (e.g.,
/// BrowserTool's `action_schema` calls it for each inner entry).
#[must_use]
fn tool_params_schema(properties: &serde_json::Value, required: &[&str]) -> serde_json::Value {
    let mut schema = serde_json::json!({
        "type": "object",
        "properties": properties,
    });
    if !required.is_empty() {
        schema["required"] = serde_json::json!(required);
    }
    schema
}

// ── scrub ────────────────────────────────────────────────────────────

use crate::util::scrub_credentials;

/// Scrub credentials from tool output; delegates the scrubbing policy to the tool.
///
/// Call flow:
///
/// ```text
/// agent::execute_tool                           (agent.rs:455)
///   └─ scrub_tool_output(tool, args, output)     (tools/mod.rs)
///        ├─ tool.should_scrub_output(args)        (lib.rs, per-tool override)
///        │    ├─ returns true  → scrub_credentials(output)  (util/mod.rs)
///        │    └─ returns false → output as-is
/// ```
///
/// Integration with shell tool's internal scrubbing:
/// The shell tool's `apply_profile_pipeline` (shell/mod.rs) scrubs stdout and stderr
/// at pipeline entry, so `ShellTool::should_scrub_output` returns `false` to prevent
/// this function from double-scrubbing. See the `ShellTool::should_scrub_output` doc
/// comment for the rationale. Any tool that performs its own credential scrubbing
/// internally must follow the same pattern — return `false` from `should_scrub_output`.
#[must_use]
pub(crate) fn scrub_tool_output(
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
pub(crate) fn format_tool_failure_feedback(
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
pub(crate) struct ToolExecutionOutcome {
    pub output: String,
    pub success: bool,
}

/// Normalize a tool call name and arguments, repairing common agent mistakes.
///
/// Returns `(normalized_name, normalized_args)`. Stats and dispatch should use
/// the normalized values so recovered calls are attributed to the real tool.
#[must_use]
pub(crate) fn normalize_tool_call(
    name: &str,
    mut args: serde_json::Value,
) -> (String, serde_json::Value) {
    if name == "glob"
        && let Some(obj) = args.as_object_mut()
        && !obj.contains_key("mode")
    {
        obj.insert("mode".to_string(), serde_json::json!("files"));
    }
    let normalized_name = normalize_tool_name(name).to_string();
    normalize_tool_arguments(&normalized_name, &mut args);
    (normalized_name, args)
}

/// Map known tool-name aliases to their canonical names.
///
/// This is the single source of truth for tool-name normalization, shared by
/// [`normalize_tool_call`] (full call normalization) and [`find_tool`] (direct
/// lookup).  Adding a new alias here immediately affects both paths.
///
/// The `"glob"` alias is included because it resolves to `"search"` regardless
/// of arguments; the parallel `mode:"files"` injection is handled separately
/// in [`normalize_tool_call`] when args are available.
fn normalize_tool_name(name: &str) -> &str {
    match name {
        "bash" | "run_terminal_cmd" => "shell",
        "grep" | "rg" | "grep_search" | "glob" => "search",
        "read_file" => "read",
        "str_replace" => "edit",
        _ => name,
    }
}

fn normalize_tool_arguments(name: &str, args: &mut serde_json::Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    // Path aliases are applied to ALL tools unconditionally.
    for &alias in PATH_ALIAS_KEYS {
        remap_arg_key(obj, alias, "path");
    }

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
///
/// When both the alias (`from`) and canonical (`to`) keys are present the
/// canonical value is used and a diagnostic is logged so the behaviour is
/// observable during agent debugging.
fn remap_arg_key(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if !obj.contains_key(to)
        && let Some(v) = obj.remove(from)
    {
        obj.insert(to.to_string(), v);
    } else if from != to && obj.contains_key(to) && obj.contains_key(from) {
        // Both keys present — log a diagnostic so behavior is observable.
        debug!("remap_arg_key: both '{from}' (alias) and '{to}' (canonical) present, using '{to}'");
    }
}

/// Look up a tool by name in a slice of boxed `dyn Tool` values.
///
/// Tool-name aliases are resolved via `normalize_tool_name` so that all
/// callers benefit from the same alias mapping.  Prefer [`normalize_tool_call`]
/// before dispatch when full argument normalization is also desired.
#[must_use]
pub(crate) fn find_tool<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> Option<&'a dyn Tool> {
    let normalized = normalize_tool_name(name);
    tools
        .iter()
        .find(|t| t.name() == normalized)
        .map(Box::as_ref)
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
async fn save_generated_file(
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

    let timestamp = crate::util::unix_millis();
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
    use crate::workspace::test_ws_named;
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

        // "read" tool: file/filename → path
        let (name, args) = normalize_tool_call("read", serde_json::json!({"file": "src/main.rs"}));
        assert_eq!(name, "read");
        assert_eq!(args["path"], "src/main.rs");

        let (name, args) =
            normalize_tool_call("read_file", serde_json::json!({"filename": "lib.rs"}));
        assert_eq!(name, "read");
        assert_eq!(args["path"], "lib.rs");

        // "edit" tool: file/filename → path, old_str/new_str → old_string/new_string
        let (name, args) = normalize_tool_call(
            "edit",
            serde_json::json!({"file": "main.rs", "old_str": "foo", "new_str": "bar"}),
        );
        assert_eq!(name, "edit");
        assert_eq!(args["path"], "main.rs");
        assert_eq!(args["old_string"], "foo");
        assert_eq!(args["new_string"], "bar");

        // Canonical "path" key is preserved when both "path" and alias are present.
        // remap_arg_key only moves when canonical key is absent, so "file" remains.
        let (name, args) = normalize_tool_call(
            "read",
            serde_json::json!({"path": "canonical.rs", "file": "alias.rs"}),
        );
        assert_eq!(name, "read");
        assert_eq!(args["path"], "canonical.rs");
        assert!(args.as_object().unwrap().contains_key("file"));
    }

    #[test]
    // ── media_marker coverage ────────────────────────────────────────
    fn all_media_tools_implement_media_marker() {
        // Each media-generation tool must return Some from media_marker()
        let tools: [(&str, Box<dyn Tool>); 2] = [
            ("ImageGenTool", Box::new(ImageGenTool)),
            ("VideoGenTool", Box::new(VideoGenTool)),
        ];
        for (name, tool) in &tools {
            let marker = tool.media_marker();
            assert!(
                marker.is_some(),
                "{name} should return Some from media_marker()"
            );
            let marker = marker.unwrap();
            // Validate format: `[KIND:` where KIND is uppercase letters
            assert!(
                marker.starts_with('['),
                "{name} marker {marker:?} should start with '['"
            );
            assert!(
                marker.ends_with(':'),
                "{name} marker {marker:?} should end with ':'"
            );
            let kind = &marker[1..marker.len() - 1]; // strip [ and :
            assert!(
                !kind.is_empty() && kind.chars().all(char::is_uppercase),
                "{name} marker kind {kind:?} should be non-empty uppercase letters"
            );
            // Validate against the canonical MEDIA_MARKER_RE pattern
            let full_marker = format!("{marker}/some/path]");
            assert!(
                crate::util::MEDIA_MARKER_RE.is_match(&full_marker),
                "{name} marker + path should match MEDIA_MARKER_RE, got: {full_marker:?}"
            );
        }
    }

    // ── require_path_arg tests ─────────────────────────────────────────

    /// Basic tests for [`require_path_arg`].
    /// After the simplification of [`find_path_arg`] (which now only checks
    /// the `"path"` key), alias resolution is verified exclusively by the
    /// [`normalize_tool_call_remaps_all_path_aliases`] test below.
    #[test]
    fn require_path_arg_path_lookup() {
        // "path" key works directly
        assert_eq!(
            require_path_arg(&serde_json::json!({"path": "src/main.rs"})).unwrap(),
            "src/main.rs"
        );
        // "path" ignored when non-string
        let err = require_path_arg(&serde_json::json!({"path": ["invalid"]})).unwrap_err();
        assert!(err.to_string().contains("path"));
        // Missing path → descriptive error mentioning 'path'
        let err = require_path_arg(&serde_json::json!({"other": "value"})).unwrap_err();
        assert!(err.to_string().contains("path"));
    }

    /// Verify that [`normalize_tool_call`] remaps every alias in
    /// [`PATH_ALIAS_KEYS`] to `"path"` unconditionally — the remapping is
    /// applied to all tools, not just those that accept file paths.
    ///
    /// This test explicitly iterates the constant so the loop-based approach in
    /// [`normalize_tool_arguments`] is verified against all current aliases.
    /// If an alias is added to [`PATH_ALIAS_KEYS`], this test immediately
    /// exercises it — preventing any gap between the lookup path and the
    /// normalization path.
    #[test]
    fn normalize_tool_call_remaps_all_path_aliases() {
        for &alias in PATH_ALIAS_KEYS {
            for (tool_name, extra) in &[
                ("read", serde_json::json!({})),
                ("edit", serde_json::json!({"old_str": "a", "new_str": "b"})),
                ("search", serde_json::json!({})),
                ("shell", serde_json::json!({"cmd": "echo hi"})),
            ] {
                let mut input = serde_json::json!({});
                input[alias] = serde_json::json!("src/main.rs");
                if let Some(obj) = extra.as_object() {
                    for (k, v) in obj {
                        input[k] = v.clone();
                    }
                }
                let (name, args) = normalize_tool_call(tool_name, input);
                assert_eq!(
                    name, *tool_name,
                    "tool name should not change for {tool_name} with alias {alias}"
                );
                assert_eq!(
                    args["path"], "src/main.rs",
                    "alias {alias} should be remapped to 'path' for tool {tool_name}"
                );
                // The alias key itself should have been removed since "path" was absent.
                assert!(
                    !args.as_object().unwrap().contains_key(alias),
                    "alias key {alias} should be removed after normalization for {tool_name}"
                );
                // Edit-specific remaps must be unaffected.
                if *tool_name == "edit" {
                    assert_eq!(args["old_string"], "a");
                    assert_eq!(args["new_string"], "b");
                }
            }
        }
    }

    // ── save_generated_file tests ──────────────────────────────────────────

    #[tokio::test]
    async fn save_generated_file_creates_file() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = test_ws_named(&tmp.path().to_string_lossy(), "test");

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
        let ws = test_ws_named(&tmp.path().join("nested").to_string_lossy(), "test");

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
