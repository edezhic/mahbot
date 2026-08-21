//! Tool implementations for agent-callable capabilities.

use anyhow::Context;
pub(crate) mod active_models;
pub(crate) mod analyze;
pub mod browser;
pub mod browser_daemon;
pub(crate) mod catalog_cache;
pub(crate) mod edit;
pub mod image_catalog;
pub(crate) mod image_gen;
pub(crate) mod implement;
pub(crate) mod path;
pub(crate) mod read;
pub(crate) mod research;
pub(crate) mod search;
pub(crate) mod search_archived_tickets;
pub(crate) mod shell;
pub(crate) mod ticket;
pub(crate) mod video_catalog;
pub(crate) mod video_edit;
pub(crate) mod video_gen;
pub(crate) mod web_search;

/// Maximum file size allowed for read, edit, search tool operations, and the dashboard editor (10 MB).
/// Guards against OOM when agents or the GUI attempt to read very large files.
/// Used by ReadTool and EditTool via `check_file_size()`;
/// SearchTool and the Iced Editor use `MAX_FILE_SIZE_BYTES` directly.
pub(crate) const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum size for a single reference image in bytes.
/// OpenRouter enforces a ~2 MB request body limit; base64 adds ~33% overhead so
/// we cap raw image data at 1.5 MB (1_500_000 bytes) to stay well under.
/// Over-cap references are compressed by the loader (see `load_reference_image`).
/// Note: the aggregate body budget (`MAX_REQUEST_BODY_BYTES`) re-encodes even
/// under-cap references when the serialized request exceeds it, so literal
/// pass-through holds up to ~1.499 MB per single reference (base64 of 1.5 MB
/// already reaches the 2 MB budget once the data-URI prefix and JSON envelope
/// are added).
const MAX_REFERENCE_IMAGE_BYTES: u64 = 1_500_000;

/// Aggregate budget for the serialized generation request body. The per-image
/// cap does not bound the total body when multiple references are present, so
/// the final serialized body is checked against this budget and references are
/// compressed further until it fits. The 2 MB figure is a client-side sanity
/// budget mirroring the pre-existing ~2 MB provider body-limit premise (not a
/// documented OpenRouter number); the dev-time /videos acceptance test verified
/// the common case against the live provider.
const MAX_REQUEST_BODY_BYTES: usize = 2_000_000;

/// Conservative cap on reference images per generation request, applied only
/// when the model catalog is unavailable (fail-open). Per-model caps come from
/// the catalog; provider acceptance of `input_references` varies, so this is a
/// memory-bounding sanity limit — the aggregate body budget is the real
/// backstop for what reaches the wire.
pub(crate) const MAX_REFERENCE_IMAGES_PER_REQUEST: usize = 16;

/// Canonical list of argument aliases for file path parameters.
///
/// The aliases `"file"` and `"filename"` are remapped to `"path"` for every
/// tool call by [`normalize_tool_arguments`]. Tools that do not accept a
/// `"path"` parameter are unaffected — unknown keys are discarded downstream.
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

// ── Generation request reference images ─────────────────────────────────

/// Build the OpenRouter `input_references` array (image_url entries) for
/// generation requests. Single source of truth for the reference request shape.
fn reference_json(references: &[crate::util::ReferenceImage]) -> serde_json::Value {
    serde_json::json!(
        references
            .iter()
            .map(|r| serde_json::json!({
                "type": "image_url",
                "image_url": { "url": r.data_uri() }
            }))
            .collect::<Vec<_>>()
    )
}

/// Key of the reference-image array in the generation request body.
const INPUT_REFERENCES_KEY: &str = "input_references";

/// Ensure the final serialized request body fits `max_body_bytes`, compressing
/// references (largest first, falling through to the next-largest when a
/// reference's ladder is exhausted) one step at a time until it does. The
/// per-image cap does not bound the total body (N refs × cap), so this is the
/// real guard against oversized generation requests.
///
/// The reference array is always (re)synced from the current reference state
/// when references are present, so a caller cannot silently drop them; the
/// loop rewrites it after each compression step.
pub(crate) fn fit_request_body_budget(
    body: &mut serde_json::Value,
    references: &mut [crate::util::ReferenceImage],
    max_body_bytes: usize,
) -> anyhow::Result<()> {
    if !references.is_empty() {
        body[INPUT_REFERENCES_KEY] = reference_json(references);
    }
    loop {
        if serde_json::to_vec(body)?.len() <= max_body_bytes {
            // The request body is final — drop the retained source bytes so
            // they don't sit in memory through the generation fetch/poll.
            for r in references {
                r.release_source_bytes();
            }
            return Ok(());
        }
        let Some(idx) = references
            .iter()
            .enumerate()
            .filter(|(_, r)| r.has_compression_left())
            .max_by_key(|(_, r)| r.data_uri().len())
            .map(|(i, _)| i)
        else {
            break;
        };
        references[idx].compress_more()?;
        body[INPUT_REFERENCES_KEY] = reference_json(references);
    }
    let body_len = serde_json::to_vec(body)?.len();
    if references.is_empty() {
        anyhow::bail!(
            "Generation request body is too large ({body_len} bytes; limit {max_body_bytes} bytes). \
             Shorten the prompt.",
        );
    }
    let advice = if references.len() > 1 {
        "Reduce the reference image size, the number of references, or the prompt length."
    } else {
        "Reduce the reference image size or shorten the prompt."
    };
    anyhow::bail!(
        "Generation request body is too large ({body_len} bytes after compression; \
         limit {max_body_bytes} bytes). {advice}",
    );
}

// ── Re-exports ─────────────────────────────────────────────────────────

pub(crate) use analyze::{AnalyzeTool, DispatchMode};
pub(crate) use browser::BrowserTool;
pub(crate) use edit::EditTool;
pub(crate) use image_gen::ImageGenTool;
pub(crate) use implement::ImplementTool;
pub(crate) use read::ReadTool;
pub(crate) use research::ResearchTool;
pub(crate) use search::SearchTool;
pub(crate) use search_archived_tickets::SearchArchivedTicketsTool;
pub(crate) use shell::{ShellMode, ShellTool};
pub(crate) use ticket::{
    AddCommentTool, CreateTicketTool, GetTicketTool, ListTicketsTool, UpdateTicketTool,
};
pub(crate) use video_edit::VideoEditTool;
pub(crate) use video_gen::VideoGenTool;
pub(crate) use web_search::{WebSearchBackend, WebSearchTool};

use crate::{Tool, Workspace};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ── JSON helpers ─────────────────────────────────────────────────────────

use crate::util::json::{
    get_bool, get_opt_bool, get_opt_i64, get_opt_str, get_opt_u64, get_str, get_str_array,
    get_usize,
};

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
/// agent::execute_tool
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

/// Prefix of a failed tool result — shared so the research wrap-up's
/// success detection checks the same marker [`format_tool_failure_feedback`]
/// produces (a successful result never starts with it).
pub(crate) const TOOL_FAILURE_MARKER: &str = "Tool call failed.";

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
        "{TOOL_FAILURE_MARKER}\n\
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
/// in [`normalize_tool_call`] when args are available. `pub(crate)` so the
/// research sanitizer can gate on the canonical name before paying for an
/// argument clone.
pub(crate) fn normalize_tool_name(name: &str) -> &str {
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

    // Apply path-aliases (file/filename) universally for every tool call.
    // Tools that do not accept a "path" parameter silently ignore the extra key.
    for &alias in PATH_ALIAS_KEYS {
        remap_arg_key(obj, alias, "path");
    }

    match name {
        "edit" => {
            // Per-tool argument remaps specific to "edit".
            remap_arg_key(obj, "old_str", "old_string");
            remap_arg_key(obj, "new_str", "new_string");
        }
        "shell" => {
            remap_arg_key(obj, "cmd", "command");
            remap_arg_key(obj, "script", "command");
        }
        "get_ticket" | "update_ticket" | "add_comment" => {
            remap_arg_key(obj, "id", "ticket_id");
            remap_arg_key(obj, "ticket", "ticket_id");
        }
        _ => {}
    }
}

/// Move `from` → `to` only when the canonical key is absent.
///
/// When both the alias (`from`) and canonical (`to`) keys are present the
/// canonical value is used.
fn remap_arg_key(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if !obj.contains_key(to)
        && let Some(v) = obj.remove(from)
    {
        obj.insert(to.to_string(), v);
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

/// Format a video tool result: the `[VIDEO:path]` marker first (so the reply
/// path and GUI keep working), then a "Video content:" description line when
/// the shared video transcription succeeds. Fail-open: marker-only on any
/// failure. The transcription's live tracking resolves via the tool-execution
/// task-local (the owning agent's activity indicator); `None` is passed for
/// workspace because the in-agent path never needs a non-agent call row.
pub(crate) async fn format_video_result(marker: String, output_path: &std::path::Path) -> String {
    match crate::providers::transcribe_video_file(output_path, None).await {
        Some(text) => format!("{marker}\n\nVideo content: {text}"),
        None => marker,
    }
}

// ── Async video jobs (video_gen / video_edit) ───────────────────────────

/// Wall-clock deadline for an async-video job (1 hour) — shared by the
/// video_gen and video_edit tools.
const VIDEO_JOB_TIMEOUT: Duration = Duration::from_hours(1);

/// Polling interval between video job status checks (30 s).
const VIDEO_POLL_INTERVAL: Duration = Duration::from_secs(30);

/// Kind-dependent labels for an OpenRouter async-video job.
///
/// Fields stay separate so every user-facing string is byte-identical with the
/// pre-refactor tools (e.g. gen's download context is bare "Video download",
/// not "Video generation download").
struct VideoJobLabels {
    /// "Video generation" / "Video edit" — submission/poll HTTP contexts,
    /// tracing, "failed" and timeout bails.
    label: &'static str,
    /// "generation" / "editing" — the 402 credits bail.
    gerund: &'static str,
    /// "Video download" / "Video edit download" — download context + MP4 bail.
    download: &'static str,
}

impl VideoJobLabels {
    const GENERATION: Self = Self {
        label: "Video generation",
        gerund: "generation",
        download: "Video download",
    };
    const EDIT: Self = Self {
        label: "Video edit",
        gerund: "editing",
        download: "Video edit download",
    };
}

/// Submit an OpenRouter async-video job (exactly one POST — no retry; each
/// submission is a billable job and the endpoint has no idempotency key),
/// poll for completion (1-hour wall-clock deadline), download the result, and
/// validate it is a real MP4. Returns the video bytes; callers save the file
/// and format the media marker.
#[expect(clippy::too_many_lines)]
async fn fetch_async_video(
    api_base: &str,
    body: &serde_json::Value,
    labels: VideoJobLabels,
) -> anyhow::Result<Vec<u8>> {
    // ── Step 1: Submit video job (exactly one POST — no retry) ─────────
    // Each submission is a billable job; the endpoint has no idempotency key.
    let submit_url = format!("{api_base}/videos");
    let submit_body: serde_json::Value = match crate::util::http::post_json_to_provider(
        &submit_url,
        body,
        &format!("{} submission", labels.label),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            if e.downcast_ref::<crate::util::error::HttpError>()
                .map(|e| e.status)
                == Some(402)
            {
                anyhow::bail!(
                    "Insufficient OpenRouter credits for video {} (HTTP 402). \
                     Please add credits to your OpenRouter account and try again.",
                    labels.gerund,
                );
            }
            return Err(e);
        }
    };

    // OpenRouter returns: { id, polling_url, status, ... }
    let job_id = match submit_body.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            anyhow::bail!("No job ID in submission response: {submit_body}");
        }
    };

    let polling_url = match submit_body.get("polling_url").and_then(|v| v.as_str()) {
        Some(url) => url.to_string(),
        None => format!("{api_base}/videos/{job_id}"),
    };

    tracing::info!(%job_id, "{} job submitted", labels.label);

    // ── Step 2: Poll for completion (1-hour wall-clock deadline) ────────
    let deadline = Instant::now() + VIDEO_JOB_TIMEOUT;
    let mut result_url: Option<String> = None;
    let mut attempt: u32 = 0;

    while Instant::now() < deadline {
        attempt += 1;

        // Sleep the full interval, but never past the deadline.
        let remaining = deadline.saturating_duration_since(Instant::now());
        tokio::time::sleep(VIDEO_POLL_INTERVAL.min(remaining)).await;
        if Instant::now() >= deadline {
            break;
        }

        // Race the poll against the remaining window so a slow poll request
        // cannot push the total wait past the hour.
        let remaining = deadline.saturating_duration_since(Instant::now());
        let poll_body = match tokio::time::timeout(
            remaining,
            crate::util::http::get_json_from_provider(
                &polling_url,
                &format!("{} poll", labels.label),
            ),
        )
        .await
        {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                tracing::debug!(%job_id, attempt, error = %e, "Poll failed");
                continue;
            }
            // Deadline elapsed while the poll request was in flight.
            Err(_) => break,
        };

        let status = poll_body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        tracing::debug!(%job_id, %status, attempt, "{} poll", labels.label);

        if status == "completed" {
            // Download URL: OpenRouter provides unsigned_urls array or
            // a content endpoint at /api/v1/videos/{jobId}/content?index=0
            result_url = poll_body
                .get("unsigned_urls")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| {
                    // Fallback: use content endpoint
                    Some(format!("{api_base}/videos/{job_id}/content?index=0"))
                });
            break;
        }

        if status == "failed" || status == "cancelled" || status == "expired" {
            let err_msg = poll_body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            anyhow::bail!("{} failed: {err_msg}", labels.label);
        }
    }

    let Some(download_url) = result_url else {
        anyhow::bail!(
            "{} did not complete within the 1-hour timeout period",
            labels.label
        );
    };

    // ── Step 3: Download the video ──────────────────────────────────────
    // The result URL requires the bearer key despite the "unsigned" name.
    // The per-request timeout is the remaining job window, so large files on
    // slow connections are not cut short by the shared client's 2-minute cap.
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        anyhow::bail!(
            "{} did not complete within the 1-hour timeout period",
            labels.label
        );
    }
    let video_bytes =
        crate::util::http::get_bytes_from_provider(&download_url, labels.download, remaining)
            .await?;

    // Validate the payload is a real MP4 (no Content-Length on the response;
    // an error page would slip through otherwise).
    if video_bytes.len() <= 100_000 || &video_bytes[4..8] != b"ftyp" {
        anyhow::bail!(
            "{} returned an invalid file ({} bytes, no ftyp header)",
            labels.download,
            video_bytes.len(),
        );
    }

    Ok(video_bytes)
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
    fn contains_glob_detects_wildcards() {
        assert!(crate::tools::path::contains_glob("src/*.rs", true));
        assert!(crate::tools::path::contains_glob("lib?.rs", true));
        assert!(!crate::tools::path::contains_glob("src/main.rs", true));
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
        let tools: [(&str, Box<dyn Tool>); 3] = [
            ("ImageGenTool", Box::new(ImageGenTool)),
            ("VideoGenTool", Box::new(VideoGenTool)),
            ("VideoEditTool", Box::new(VideoEditTool)),
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

    // ── normalize_tool_call tests ─────────────────────────────────────

    /// Verify that [`normalize_tool_call`] universally remaps every alias in
    /// [`PATH_ALIAS_KEYS`] to `"path"` — regardless of whether the tool
    /// explicitly accepts a `"path"` parameter.
    ///
    /// This test verifies path-aliasing for both a path-accepting tool (`read`)
    /// and a non-path tool (`shell`). The remap happens unconditionally;
    /// tools that do not use `"path"` simply ignore the extra key.
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
                ("shell", serde_json::json!({"cmd": "ls"})),
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
                // The alias is remapped to "path" unconditionally — even for tools
                // that don't use the "path" parameter (unknown keys are discarded).
                assert_eq!(
                    args["path"], "src/main.rs",
                    "alias {alias} should be remapped to 'path' for tool {tool_name}"
                );
                // The alias key itself should have been removed since "path" was absent.
                assert!(
                    !args.as_object().unwrap().contains_key(alias),
                    "alias key {alias} should be removed after normalization for {tool_name}"
                );
                // Shell-specific remaps must be unaffected.
                if *tool_name == "shell" {
                    assert_eq!(args["command"], "ls");
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

    // ── Reference-image body budget ─────────────────────────────────

    #[tokio::test]
    async fn reference_body_budget_compresses_further() {
        const BUDGET: usize = 150_000;
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ref.png");
        std::fs::write(&path, crate::util::test::noisy_png(512, 512)).unwrap();
        let mut refs = vec![
            crate::util::load_reference_image(&path, MAX_REFERENCE_IMAGE_BYTES)
                .await
                .unwrap(),
        ];
        let original = refs[0].data_uri().to_string();
        let mut body = serde_json::json!({
            "model": "test",
            "prompt": "test",
            "input_references": reference_json(&refs),
        });
        // The per-image cap passes a 512×512 ref through unchanged (~1 MB
        // base64 body); a small budget forces the aggregate ladder.
        assert!(serde_json::to_vec(&body).unwrap().len() > BUDGET);
        fit_request_body_budget(&mut body, &mut refs, BUDGET).unwrap();
        assert!(serde_json::to_vec(&body).unwrap().len() <= BUDGET);
        assert_ne!(
            refs[0].data_uri(),
            original,
            "reference should be compressed"
        );
    }

    #[tokio::test]
    async fn body_budget_falls_through_to_smaller_reference() {
        // When the largest reference's ladder is exhausted, the loop must
        // fall through to the next-largest instead of bailing.
        let tmp = TempDir::new().unwrap();
        let big = tmp.path().join("big.png");
        let small = tmp.path().join("small.png");
        std::fs::write(&big, crate::util::test::noisy_png(640, 640)).unwrap();
        std::fs::write(&small, crate::util::test::noisy_png(128, 128)).unwrap();
        let mut refs = vec![
            crate::util::load_reference_image(&big, MAX_REFERENCE_IMAGE_BYTES)
                .await
                .unwrap(),
            crate::util::load_reference_image(&small, MAX_REFERENCE_IMAGE_BYTES)
                .await
                .unwrap(),
        ];
        let small_original = refs[1].data_uri().to_string();
        let mut body = serde_json::json!({
            "model": "test",
            "prompt": "test",
            "input_references": reference_json(&refs),
        });
        // Exhaust the big reference's ladder, then measure the raw body so the
        // budget can be pinned to a window that only the small reference can
        // close — deterministically forcing the fall-through path.
        while refs[0].has_compression_left() {
            refs[0].compress_more().unwrap();
        }
        body[INPUT_REFERENCES_KEY] = reference_json(&refs);
        let raw_body = serde_json::to_vec(&body).unwrap().len();
        fit_request_body_budget(&mut body, &mut refs, raw_body - 1).unwrap();
        assert!(serde_json::to_vec(&body).unwrap().len() < raw_body);
        assert_ne!(
            refs[1].data_uri(),
            small_original,
            "smaller reference should have been compressed"
        );
    }
}
