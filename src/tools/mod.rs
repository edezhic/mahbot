//! Tool implementations for agent-callable capabilities.

use anyhow::Context;

pub mod ask;
pub mod browser;
pub mod edit;
pub mod image_gen;
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
use std::path::{Path, PathBuf};

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

/// File paths whose read output should be scrubbed for credentials (`.env`, certs, keys).
/// Other extensions (e.g. `.rs`, `.md`) are left intact so the model sees source accurately.
#[must_use]
pub fn should_scrub_file_path(path: &str) -> bool {
    let Some(file_name) = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
    else {
        return true;
    };
    let lower = file_name.to_ascii_lowercase();

    // Single rsplit_once handles both dotfiles (.env, .env.local) and regular extensions (.pem, .key).
    // The `name == ".env"` arm catches `.env.local`-style dotfile prefixes.
    match lower.rsplit_once('.') {
        Some((name, ext)) => {
            name == ".env"
                || ext == "env"
                || matches!(ext, "pem" | "key" | "p12" | "pfx" | "crt" | "cer")
        }
        None => false,
    }
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
    // Callers must pre-scrub `reason` before passing it; this function does
    // not scrub to avoid double-scrubbing — scrubbing is the responsibility
    // of [`failure_outcome`](crate::agent::Agent::failure_outcome).
    let args_preview = scrub_credentials(&crate::util::truncate(&tool_args.to_string(), 1000));
    format!(
        "Tool call failed.\n\
         tool: {tool_name}\n\
         arguments: {args_preview}\n\
         reason:\n{reason}"
    )
}

/// Shared result formatting for file-based tools (write, edit).
/// Handles the "After" phase with truncation and either a code fence or
/// expandable blockquote depending on content size.
#[must_use]
pub fn format_file_tool_result(
    action: &str,
    content: &str,
    args: &serde_json::Value,
    outcome: &ToolExecutionOutcome,
) -> String {
    let path = crate::tools::find_path_arg(args).unwrap_or("?");
    if !outcome.success {
        return format!("❌ {action} attempted on {path}");
    }

    let block = crate::util::truncate_sandwich(content, 2000, "debug");
    format!("✏️ {path}\n{block}")
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
/// Tool-name aliases are resolved via [`normalize_tool_name_str`] so that all
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

/// Fallback directory resolution when full-path [`canonicalize`] fails with
/// `NotFound` but the lexical path still exists as a directory.
///
/// Uses parent canonicalization + final component (same strategy as write-mode
/// path resolution) so existing directories are listable even when agents omit
/// a trailing `/` or when full-path canonicalization fails on edge-case paths.
pub(crate) async fn resolve_directory_read_fallback(full_path: &Path) -> Option<PathBuf> {
    let meta = tokio::fs::symlink_metadata(full_path).await.ok()?;
    if !meta.is_dir() {
        return None;
    }

    let parent = full_path.parent()?;
    let name = full_path.file_name()?;
    let canon_parent = tokio::fs::canonicalize(parent).await.ok()?;
    let resolved = canon_parent.join(name);
    if tokio::fs::symlink_metadata(&resolved)
        .await
        .is_ok_and(|m| m.is_dir())
    {
        return Some(resolved);
    }

    Some(full_path.to_path_buf())
}

/// Resolve and validate a file target for write/edit operations.
///
/// Path security is enforced via [`is_path_safe_for_workspace`] (pre- and
/// post-canonicalization). Extra read paths (spill files, dependency caches)
/// are **not** allowed for writes.
///
/// Additional security:
/// 1. Canonicalize the **parent** directory only — the file itself may not exist yet.
/// 2. Symlink check: if the target exists and is a symlink, refuse (unlike reading,
///    where `canonicalize` resolves through symlinks safely).
/// 3. If `ensure_parent` is `true`, creates parent directories before canonicalizing.
///
/// See [`resolve_read_target`] for the read-side counterpart.
///
/// Returns `Ok(path)` on success, or an error message to propagate to the agent.
pub async fn resolve_write_target(
    workspace_root: &Path,
    path: &str,
    ensure_parent: bool,
) -> anyhow::Result<PathBuf> {
    let full_path = resolve_tool_path_with_base(path, workspace_root);

    // Pre-canonicalization check — strict, no extra allowed paths
    if !is_path_safe_for_workspace(path, workspace_root) {
        anyhow::bail!("Path not allowed by security policy: {path}");
    }

    let Some(parent) = full_path.parent() else {
        anyhow::bail!("Invalid path: missing parent directory");
    };

    if ensure_parent {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create parent directories")?;
    }

    // Canonicalize parent only — the file itself may not exist yet
    let resolved_parent = tokio::fs::canonicalize(parent)
        .await
        .context("Failed to resolve file path")?;

    if !is_path_safe_for_workspace(&resolved_parent.to_string_lossy(), workspace_root) {
        anyhow::bail!(
            "Path not allowed by security policy: {}",
            resolved_parent.display()
        );
    }

    let Some(file_name) = full_path.file_name() else {
        anyhow::bail!("Invalid path: missing file name");
    };

    let resolved_target = resolved_parent.join(file_name);

    // Explicit symlink refusal (read resolves symlinks via canonicalize instead)
    if let Ok(meta) = tokio::fs::symlink_metadata(&resolved_target).await
        && meta.file_type().is_symlink()
    {
        anyhow::bail!(
            "Refusing to write through symlink: {}",
            resolved_target.display()
        );
    }

    Ok(resolved_target)
}

/// Resolve and validate a file path for read operations.
///
/// Path security is enforced by `check_path_read_allowed` (pre- and
/// post-canonicalization), permitting `EXTRA_READ_ALLOWED` paths (temp files,
/// dependency source directories) in addition to workspace-scoped paths.
///
/// Key differences from [`resolve_write_target`]:
/// - Canonicalizes the **full path**, not just the parent (file must exist).
/// - No `ensure_parent` parameter — parent creation is a write-only concept.
/// - No explicit symlink refusal — `tokio::fs::canonicalize` resolves symlinks,
///   so the post-canonicalization check catches escapes via the resolved path.
/// - Also allows `EXTRA_READ_ALLOWED` paths (e.g. /tmp files, dependency caches).
///
/// Returns `Ok(path)` on success, or an error message to propagate to the agent.
pub async fn resolve_read_target(workspace_root: &Path, path: &str) -> anyhow::Result<PathBuf> {
    let full_path = resolve_tool_path_with_base(path, workspace_root);

    // Pre-canonicalization check — allows EXTRA_READ_ALLOWED paths
    // (temp files, dependency source directories) outside the workspace
    check_path_read_allowed(path, workspace_root)?;

    // Canonicalize full path (file must exist). Resolves symlinks,
    // so the post-canonicalization check catches escapes.
    let resolved_path = match tokio::fs::canonicalize(&full_path).await {
        Ok(resolved) => resolved,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            resolve_directory_read_fallback(&full_path)
                .await
                .ok_or_else(|| anyhow::anyhow!("File not found: {}", full_path.display()))?
        }
        Err(e) => {
            return Err(match e.kind() {
                std::io::ErrorKind::PermissionDenied => {
                    anyhow::anyhow!("Permission denied: {}", full_path.display())
                }
                _ => anyhow::anyhow!("Failed to resolve file path: {}: {e}", full_path.display()),
            });
        }
    };

    check_path_read_allowed(&resolved_path.to_string_lossy(), workspace_root)?;

    Ok(resolved_path)
}

/// If the path starts with `~`, expand it to the user's home directory.
/// Otherwise returns the original path as-is.
///
/// This is used by path-check helpers that compare user-provided (pre-canonicalization)
/// paths against init-time expanded allowlist entries.
fn expand_tilde_for_path_check(path: &Path) -> std::borrow::Cow<'_, Path> {
    if path.to_str().is_some_and(|s| s.starts_with('~')) {
        std::borrow::Cow::Owned(crate::config::expand_tilde(&path.to_string_lossy()))
    } else {
        std::borrow::Cow::Borrowed(path)
    }
}

/// Check whether `path` is under an [`EXTRA_READ_ALLOWED`] directory.
///
/// The path may contain a leading `~` (user-provided input before
/// canonicalization). In that case the `~` is expanded to the user's
/// home directory before comparing against the (already-expanded)
/// allowlist entries.
fn is_path_in_extra_allowed(path: &Path) -> bool {
    let check_path = expand_tilde_for_path_check(path);

    EXTRA_READ_ALLOWED
        .iter()
        .any(|allowed| check_path.starts_with(allowed))
}

/// Check that a path is allowed by the read-path security policy.
///
/// The path must be either within the workspace (via [`is_path_safe_for_workspace`])
/// or under one of the [`EXTRA_READ_ALLOWED`] directories (temp files,
/// dependency caches, SDK headers, etc.).
fn check_path_read_allowed(path: &str, workspace_root: &Path) -> anyhow::Result<()> {
    if !is_path_safe_for_workspace(path, workspace_root)
        && !is_path_in_extra_allowed(Path::new(path))
    {
        anyhow::bail!("Path not allowed by security policy: {path}");
    }
    Ok(())
}

/// Helper for [`EXTRA_READ_ALLOWED`] initialization: canonicalizes `raw` and
/// pushes both the canonical and raw paths (if they differ) into `dirs`,
/// ensuring no duplicates. On macOS `/tmp` → `/private/tmp` symlink, this
/// ensures both `/tmp` and `/private/tmp` are in the allowed set so that
/// both the raw and resolved forms match during [`resolve_read_target`]'s
/// pre- and post-canonicalization checks.
fn add_path_with_canonical(dirs: &mut Vec<PathBuf>, raw: PathBuf) {
    if dirs.contains(&raw) {
        return;
    }
    match std::fs::canonicalize(&raw) {
        Ok(canonical) => {
            if !dirs.contains(&canonical) {
                dirs.push(canonical.clone());
            }
            if canonical != raw {
                dirs.push(raw);
            }
        }
        Err(_) => {
            dirs.push(raw);
        }
    }
}

/// Map of XDG subdirectory (under `~`) to the corresponding environment variable.
/// Used to generate alternative paths when e.g. `$XDG_CACHE_HOME` is set to a
/// non-default location.
const XDG_SUBDIR_TO_ENV: &[(&str, &str)] = &[
    (".cache/", "XDG_CACHE_HOME"),
    (".config/", "XDG_CONFIG_HOME"),
    (".local/share/", "XDG_DATA_HOME"),
    (".local/state/", "XDG_STATE_HOME"),
];

/// For a `~`-prefixed path that starts with an XDG subdirectory
/// (e.g. `~/.cache/pypoetry/`), generate the alternative path using the
/// corresponding XDG environment variable if it's set and different from
/// the default.
///
/// Returns `None` if the path doesn't start with a known XDG subdirectory,
/// or if the corresponding env var is unset.
fn xdg_variant_path(tilde_path: &str) -> Option<String> {
    for (xdg_subdir, env_var) in XDG_SUBDIR_TO_ENV {
        if let Some(suffix) = tilde_path
            .strip_prefix("~/")
            .and_then(|p| p.strip_prefix(xdg_subdir))
            && let Ok(xdg_dir) = std::env::var(env_var)
        {
            let xdg_dir = xdg_dir.trim_end_matches('/');
            return Some(format!("{xdg_dir}/{suffix}"));
        }
    }
    None
}

/// All paths from the ticket that should be allowed for reading.
/// Paths starting with `~` are expanded at init time. Paths that don't
/// exist on the current platform are harmless (they fail canonicalization
/// and just get added as-is, never matching any read request).
const EXTRA_ALLOWED_RAW_PATHS: &[&str] = &[
    // ── Rust (Cargo) ────────────────────────────────────────
    "~/.cargo/registry/src/",
    "~/.cargo/git/checkouts/",
    // ── Python ──────────────────────────────────────────────
    "~/.local/lib/",
    "~/Library/Python/",
    "~/AppData/Roaming/Python/",
    "~/AppData/Local/Programs/Python/",
    "/usr/local/lib/",
    "/usr/lib/",
    "/Library/Frameworks/Python.framework/Versions/",
    "/opt/homebrew/lib/",
    "~/anaconda3/",
    "~/miniconda3/",
    "/opt/anaconda3/",
    "/opt/miniconda3/",
    "~/AppData/Local/conda/",
    "~/.cache/pypoetry/",
    "~/Library/Caches/pypoetry/",
    "~/AppData/Local/pypoetry/",
    "~/.local/share/virtualenvs/",
    "~/.cache/pipenv/",
    "~/Library/Caches/pipenv/",
    "~/AppData/Local/pipenv/",
    "~/.cache/uv/",
    "~/.local/share/uv/",
    "~/AppData/Local/uv/",
    "~/.rye/",
    // ── JavaScript / TypeScript ─────────────────────────────
    "~/.bun/install/cache/",
    "~/.local/share/pnpm/",
    "~/Library/pnpm/",
    "~/AppData/Local/pnpm/",
    "~/AppData/Roaming/npm/",
    // ── Go ──────────────────────────────────────────────────
    "~/go/pkg/mod/",
    // ── Ruby ────────────────────────────────────────────────
    "~/.gem/",
    "~/.local/share/gem/",
    "~/.bundle/",
    // ── PHP (Composer) ──────────────────────────────────────
    "~/.composer/",
    "~/.cache/composer/",
    "~/Library/Caches/composer/",
    "~/AppData/Local/composer/",
    // ── C/C++ ───────────────────────────────────────────────
    "~/.conan/",
    "~/.conan2/",
    "/usr/local/Cellar/",
    "/opt/homebrew/Cellar/",
    "/usr/local/Homebrew/Library/Taps/",
    r"C:\ProgramData\chocolatey\lib\",
    r"C:\msys64\mingw64\include\",
    r"C:\msys64\ucrt64\include\",
    r"C:\msys64\clang64\include\",
    r"C:\msys64\usr\include\",
    // ── Windows SDK / MSVC ──────────────────────────────────
    r"C:\Program Files (x86)\Windows Kits\",
    r"C:\Program Files\Microsoft Visual Studio\",
    // ── Swift ───────────────────────────────────────────────
    "~/Library/Caches/org.swift.swiftpm/",
    "~/Library/org.swift.swiftpm/",
    "~/Library/Developer/Xcode/DerivedData/",
    // ── Dart / Flutter ──────────────────────────────────────
    "~/.pub-cache/",
    "~/AppData/Local/Pub/Cache/",
    // ── Elixir / Erlang ─────────────────────────────────────
    "~/.hex/",
    "~/.mix/",
    // ── Haskell ─────────────────────────────────────────────
    "~/.cache/cabal/",
    "~/.local/state/cabal/",
    "~/.cabal/",
    "~/AppData/Local/cabal/",
    "~/AppData/Roaming/cabal/",
    "~/.stack/",
    "~/.local/share/stack/",
    "~/AppData/Local/stack/",
    "~/AppData/Roaming/stack/",
    // ── Lua (LuaRocks) ──────────────────────────────────────
    "~/.luarocks/",
    "~/.cache/luarocks/",
    "~/Library/Caches/luarocks/",
    "~/AppData/Local/luarocks/",
    // ── R ───────────────────────────────────────────────────
    "~/Library/R/",
    "~/R/",
    "~/Documents/R/",
    // ── OCaml (opam) ────────────────────────────────────────
    "~/.opam/",
    // ── Julia ───────────────────────────────────────────────
    "~/.julia/",
    // ── Nix ─────────────────────────────────────────────────
    "/nix/store/",
    // ── System package managers ─────────────────────────────
    "/opt/local/",
    "~/.local/pipx/",
];

/// Approved filesystem roots for scratch/temp files (single source of truth).
///
/// Used by read-path allowlists and read-only shell redirect / scratch-write policy.
static APPROVED_TEMP_ROOTS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs = Vec::new();
    add_path_with_canonical(&mut dirs, std::env::temp_dir());
    add_path_with_canonical(&mut dirs, PathBuf::from("/tmp"));
    add_path_with_canonical(&mut dirs, PathBuf::from("/private/tmp"));
    add_path_with_canonical(&mut dirs, PathBuf::from("/var/tmp"));
    // Explicit spill directory (usually under `temp_dir()`; documents intent).
    add_path_with_canonical(&mut dirs, std::env::temp_dir().join(".agent"));
    dirs
});

/// Check whether `path` is under an approved temp/scratch root.
#[must_use]
pub fn is_path_under_allowed_temp(path: &Path) -> bool {
    let check_path = expand_tilde_for_path_check(path);

    APPROVED_TEMP_ROOTS
        .iter()
        .any(|root| check_path.starts_with(root))
}

/// Returns true when the path looks like a glob rather than a literal file path.
#[must_use]
pub(crate) fn path_contains_wildcard(path: &str) -> bool {
    path.contains(['*', '?', '[', ']'])
}

/// Paths under any of these directories are allowed for reading (temp dir,
/// dependency caches, SDK headers, etc.). Paths are canonicalized at init
/// to handle symlinks (e.g. macOS `/tmp` → `/private/tmp`) so that
/// `is_path_in_extra_allowed()` matches paths resolved by `resolve_read_target`.
///
/// Both the canonicalized and raw paths are included because `resolve_read_target`
/// validates the path twice — once before canonicalization (raw user-provided path)
/// and once after — and both validations may bypass via `EXTRA_READ_ALLOWED`.
///
/// `~`-prefixed entries are expanded at init time using
/// [`crate::config::expand_tilde`]. If `$HOME` (and `$USERPROFILE` on Windows)
/// is unset, `~`-prefixed entries are skipped. Entries that follow XDG Base
/// Directory conventions (`~/.cache/`, `~/.local/share/`, `~/.local/state/`,
/// `~/.config/`) also generate variants using the corresponding `$XDG_*`
/// environment variable when set.
static EXTRA_READ_ALLOWED: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs = APPROVED_TEMP_ROOTS.clone();

    // Dependency source directories (cross-platform)
    for raw_path in EXTRA_ALLOWED_RAW_PATHS {
        if raw_path.starts_with('~') {
            let expanded = crate::config::expand_tilde(raw_path);
            // Skip if expansion didn't work (HOME unset → literal ~ kept)
            if expanded.to_string_lossy().starts_with('~') {
                continue;
            }
            add_path_with_canonical(&mut dirs, expanded);

            // XDG variant (e.g. ~/.cache/pypoetry → $XDG_CACHE_HOME/pypoetry)
            if let Some(xdg_path) = xdg_variant_path(raw_path) {
                add_path_with_canonical(&mut dirs, PathBuf::from(xdg_path));
            }
        } else {
            add_path_with_canonical(&mut dirs, PathBuf::from(raw_path));
        }
    }

    dirs
});

/// User-facing reason when no static tool matches `call_name`.
#[must_use]
pub fn unknown_tool_message(call_name: &str) -> String {
    format!("Unknown tool: {call_name}")
}

// ── Path validation ──────────────────────────

/// Check whether `path` is safe to access within the given `workspace_root`.
///
/// This is the central security gate for all file-path operations. It performs
/// the following checks in order:
///
/// 1. **Whitespace trimming** — Leading/trailing whitespace is stripped before
///    any other check, preventing trivial bypass attempts (e.g. `"  ../etc"`).
/// 2. **Empty / whitespace-only paths** — After trimming, an empty path is
///    treated as relative and allowed (safe).
/// 3. **Null byte rejection** — Paths containing `\0` are immediately denied.
/// 4. **Parent directory traversal** — Paths with `..` components (e.g.
///    `"../etc/passwd"`, `"foo/../../bar"`) are denied.
/// 5. **URL-encoded traversal** — Patterns `..%2f` and `%2f..` (case-insensitive)
///    are denied, covering percent-encoded bypass attempts.
/// 6. **Tilde validation** — Bare `~` is shorthand for the workspace root
///    (see [`resolve_tool_path_with_base`]), `~/…` expands to the user's home
///    directory and must be inside the workspace; everything else starting with
///    `~` (e.g. `~root`, `~nobody`) is denied to prevent access to other users'
///    home directories.
/// 7. **Tilde expansion** — The leading `~` (if present) is expanded to the
///    current user's home directory.
/// 8. **Absolute path prefix check** — Absolute paths (including those produced
///    by tilde expansion) must start with `workspace_root`. This is a lexical
///    check only — no I/O is performed.
/// 9. **Relative path allowance** — Relative paths that pass all previous checks
///    (no traversal, no null bytes, valid tilde) are unconditionally allowed.
///
/// # Note
///
/// The prefix check for absolute paths is purely lexical and does not
/// canonicalize the input. The caller is responsible for any post-canonicalization
/// checks (see `resolve_write_target` / `resolve_read_target`), which catch
/// symlink-based escapes that could bypass this pre-check.
#[must_use]
pub fn is_path_safe_for_workspace(path: &str, workspace_root: &Path) -> bool {
    let path = path.trim();
    if path.is_empty() {
        return true; // empty after trim → relative, safe
    }
    // Bare tilde is shorthand for workspace root (see resolve_tool_path_with_base)
    if path == "~" {
        return true;
    }
    if path.contains('\0') {
        return false;
    }
    if Path::new(path)
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return false;
    }
    let lower = path.to_lowercase();
    if lower.contains("..%2f") || lower.contains("%2f..") {
        return false;
    }
    if path.starts_with('~') && path != "~" && !path.starts_with("~/") {
        return false;
    }
    let expanded_path = crate::config::expand_tilde(path);
    if expanded_path.is_absolute() {
        // Lexical prefix check — no sync I/O.
        //
        // Safety: workspace_root is pre-canonicalized at workspace registration
        // (see canonicalize_workspace_path in workspace.rs). The lexical check
        // may reject absolute paths whose prefix is a symlink into the workspace
        // (e.g. /tmp/… when the real workspace is /private/tmp/… on macOS), but
        // this is harmless: agents use relative paths, and the post-canonicalization
        // checks in resolve_read_target / resolve_write_target catch any symlink
        // escapes that would bypass this pre-check.
        expanded_path.starts_with(workspace_root)
    } else {
        // Relative path without parent-dir components — always safe
        true
    }
}

/// Resolve a user path segment against `workspace_root`.
#[must_use]
pub fn resolve_tool_path_with_base(path: &str, workspace_root: &Path) -> PathBuf {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "~" {
        return workspace_root.to_path_buf();
    }
    let expanded = crate::config::expand_tilde(trimmed);
    if expanded.is_absolute() {
        return expanded;
    }
    workspace_root.join(expanded)
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

        // Canonical names
        assert!(
            find_tool(&tools, "search").is_some(),
            "search tool should be found by name"
        );
        assert!(
            find_tool(&tools, "shell").is_some(),
            "shell tool should be found by name"
        );
        assert!(
            find_tool(&tools, "read").is_some(),
            "read tool should be found by name"
        );
        assert!(
            find_tool(&tools, "edit").is_some(),
            "edit tool should be found by name"
        );

        // Shell aliases
        assert!(
            find_tool(&tools, "bash").is_some(),
            "bash alias should resolve to shell tool"
        );
        assert!(
            find_tool(&tools, "run_terminal_cmd").is_some(),
            "run_terminal_cmd alias should resolve to shell tool"
        );

        // Search aliases
        assert!(
            find_tool(&tools, "grep").is_some(),
            "grep alias should resolve to search tool"
        );
        assert!(
            find_tool(&tools, "rg").is_some(),
            "rg alias should resolve to search tool"
        );
        assert!(
            find_tool(&tools, "grep_search").is_some(),
            "grep_search alias should resolve to search tool"
        );
        assert!(
            find_tool(&tools, "glob").is_some(),
            "glob alias should resolve to search tool"
        );

        // Read aliases
        assert!(
            find_tool(&tools, "read_file").is_some(),
            "read_file alias should resolve to read tool"
        );

        // Edit aliases
        assert!(
            find_tool(&tools, "str_replace").is_some(),
            "str_replace alias should resolve to edit tool"
        );

        // Unknown tool
        assert!(
            find_tool(&tools, "unknown").is_none(),
            "unknown tool should return None"
        );
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
    fn is_path_under_allowed_temp_covers_common_roots() {
        let temp = std::env::temp_dir();
        let spill = temp.join(".agent/spill_test.txt");
        assert!(is_path_under_allowed_temp(&temp.join("scratch.txt")));
        assert!(is_path_under_allowed_temp(&spill));
        assert!(is_path_under_allowed_temp(Path::new("/tmp/out.txt")));
        assert!(is_path_under_allowed_temp(Path::new("/var/tmp/out.txt")));
        assert!(!is_path_under_allowed_temp(Path::new("relative.txt")));
        assert!(!is_path_under_allowed_temp(Path::new("/etc/passwd")));
    }

    #[test]
    fn check_path_read_allowed_var_tmp() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(
            check_path_read_allowed("/var/tmp/mahbot-test.txt", &workspace).is_ok(),
            "/var/tmp should be readable via EXTRA_READ_ALLOWED"
        );
    }

    // ── media_marker coverage ────────────────────────────────────────

    #[test]
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

    // ── sanitize tests ─────────────────────────────────────────────────

    #[test]
    fn should_scrub_file_path_env_and_certs() {
        assert!(should_scrub_file_path(".env"));
        assert!(should_scrub_file_path("proj/.env"));
        assert!(should_scrub_file_path(".env.local"));
        assert!(should_scrub_file_path("/abs/path/.env.production"));
        assert!(should_scrub_file_path("secrets/local.env"));
        assert!(should_scrub_file_path("tls/cert.pem"));
        assert!(should_scrub_file_path("C:\\keys\\id_rsa.key"));

        assert!(!should_scrub_file_path("src/main.rs"));
        assert!(!should_scrub_file_path("crates/foo/lib.rs"));
        assert!(!should_scrub_file_path("README.md"));
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

    // ── truncate tests ─────────────────────────────────────────────────

    #[test]
    fn floor_char_boundary_handles_multibyte_offsets() {
        let text = "aé你好";
        assert_eq!(text.floor_char_boundary(5), 3);
        assert_eq!(text.floor_char_boundary(usize::MAX), text.len());
    }

    // ── path validation tests ──────────────────────────────────────────

    #[test]
    fn path_traversal_edge_cases() {
        let base = Path::new(".");
        assert!(!is_path_safe_for_workspace("../etc/passwd", base));
        assert!(!is_path_safe_for_workspace("foo/../etc/passwd", base));
        assert!(is_path_safe_for_workspace("my..file.txt", base));
        assert!(!is_path_safe_for_workspace(
            "foo/..%2f..%2fetc/passwd",
            base
        ));
    }

    #[test]
    fn path_blocked_system_and_sensitive() {
        let base = Path::new(".");
        assert!(!is_path_safe_for_workspace("file\0.txt", base));
        assert!(!is_path_safe_for_workspace(
            "/proc/self/root/etc/passwd",
            base
        ));
        assert!(!is_path_safe_for_workspace("/var/run/docker.sock", base));
        assert!(!is_path_safe_for_workspace("~/.ssh/id_rsa", base));
        assert!(!is_path_safe_for_workspace("~/.gnupg/secring.gpg", base));
        assert!(!is_path_safe_for_workspace("~root/.ssh/id_rsa", base));
        assert!(!is_path_safe_for_workspace("~nobody", base));
    }

    #[test]
    fn checklist_path_blocking() {
        let base = Path::new(".");
        assert!(!is_path_safe_for_workspace("/", base));
        assert!(!is_path_safe_for_workspace("/anything", base));
        assert!(!is_path_safe_for_workspace("/tmp", base));
        assert!(!is_path_safe_for_workspace("/var/log", base));
        // Leading whitespace bypasses (all three variants)
        assert!(!is_path_safe_for_workspace("  /etc/passwd", base));
        assert!(!is_path_safe_for_workspace("\t/etc/passwd", base));
        assert!(!is_path_safe_for_workspace("  ~root/.ssh/id_rsa", base));
        assert!(!is_path_safe_for_workspace("  ../foo", base));
        // Whitespace-only paths are treated as empty (relative, safe)
        assert!(is_path_safe_for_workspace("  ", base));

        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(is_path_safe_for_workspace(
            workspace.join("test.txt").to_str().unwrap(),
            &workspace
        ));
        assert!(is_path_safe_for_workspace("relative.txt", &workspace));
    }

    #[test]
    fn bare_tilde_is_allowed_as_workspace_root_shorthand() {
        // Bare ~ is shorthand for the workspace root — is_path_safe_for_workspace
        // must accept it, matching resolve_tool_path_with_base's behaviour.
        // Write operations are still protected by the post-canonicalization
        // parent check in resolve_write_target (the parent of workspace root
        // is outside the workspace).
        let base = Path::new(".");
        // Pathological: bare tilde with no workspace context
        assert!(is_path_safe_for_workspace("~", base));
        // Bare tilde with leading/trailing whitespace (trimmed before check)
        assert!(is_path_safe_for_workspace("  ~", base));
        assert!(is_path_safe_for_workspace("~  ", base));
        assert!(is_path_safe_for_workspace("  ~  ", base));

        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(is_path_safe_for_workspace("~", &workspace));

        // ~/… still correctly resolved relative to home and blocked if outside
        // workspace (sensitive files like .ssh should never pass)
        assert!(!is_path_safe_for_workspace("~/.ssh/id_rsa", &workspace));
        assert!(!is_path_safe_for_workspace(
            "~/.gnupg/secring.gpg",
            &workspace
        ));
    }

    #[test]
    fn path_allows_relative_and_blocks_absolute() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(is_path_safe_for_workspace("src/main.rs", &workspace));
        assert!(is_path_safe_for_workspace(
            "deep/nested/dir/file.txt",
            &workspace
        ));
        assert!(is_path_safe_for_workspace(".gitignore", &workspace));
        assert!(is_path_safe_for_workspace(".env", &workspace));
        assert!(is_path_safe_for_workspace("", &workspace));
        assert!(!is_path_safe_for_workspace("../etc/passwd", &workspace));
        assert!(!is_path_safe_for_workspace(
            "../../root/.ssh/id_rsa",
            &workspace
        ));
        assert!(!is_path_safe_for_workspace(
            "foo/../../../etc/shadow",
            &workspace
        ));
        assert!(!is_path_safe_for_workspace("..", &workspace));
    }

    // ── EXTRA_READ_ALLOWED tests ──────────────────────────────────────

    /// Init doesn't panic even with many non-existent paths (e.g. Windows
    /// paths on macOS, or missing optional toolchains).
    #[test]
    fn extra_allowed_init_does_not_panic() {
        // Force initialization of the LazyLock.
        let dirs = &*EXTRA_READ_ALLOWED;
        // The temp dirs + all dependency paths should be present.
        // On a typical dev machine, at least a few entries should exist.
        assert!(!dirs.is_empty(), "EXTRA_READ_ALLOWED should not be empty");
    }

    /// No literal `~` path is ever stored in the allowlist — all `~`-prefixed
    /// entries are expanded at init time (or skipped if `$HOME` is unset).
    #[test]
    fn extra_allowed_no_literal_tilde() {
        let dirs = &*EXTRA_READ_ALLOWED;
        for dir in dirs {
            let s = dir.to_string_lossy();
            assert!(
                !s.starts_with('~'),
                "Literal tilde path should never be stored: {s}"
            );
        }
    }

    /// `is_path_in_extra_allowed` matches `~`-prefixed user input against
    /// the expanded allowlist entries. This simulates the pre-canonicalization
    /// check in `resolve_read_target`.
    #[test]
    fn extra_allowed_tilde_input_matches() {
        // Use a known-expanded path from the allowlist to construct a ~ variant
        if let Ok(home) = std::env::var("HOME") {
            // ~/.cargo/registry/src/ should be in the allowlist (expanded to $HOME/.cargo/registry/src/)
            let tilde_input = "~/.cargo/registry/src/some-crate/src/lib.rs";
            assert!(
                is_path_in_extra_allowed(Path::new(tilde_input)),
                "~-prefixed path should match expanded allowlist entry"
            );

            // Verify the expanded path also matches (post-canonicalization)
            let expanded = PathBuf::from(&home).join(".cargo/registry/src/some-crate/src/lib.rs");
            assert!(
                is_path_in_extra_allowed(&expanded),
                "Expanded path should match allowlist entry"
            );
        }
    }

    /// Prefix matching works for version-variant subdirectories.
    /// `~/.cargo/registry/src/crate-0.1.0/src/lib.rs` should match because
    /// the allowlist entry is `~/.cargo/registry/src/`.
    #[test]
    fn extra_allowed_prefix_matching() {
        if let Ok(home) = std::env::var("HOME") {
            // Allowlist has ~/.cargo/registry/src/ — any child path should match.
            let cases: &[&str] = &[
                "~/.cargo/registry/src/crate-0.1.0/src/lib.rs",
                "~/.cargo/registry/src/crate-0.1.0/",
                "~/.cargo/registry/src/serde-1.0.0/src/lib.rs",
                // A path where a component starts with the same prefix but is
                // NOT a path-boundary match (Path::starts_with does component
                // matching, so ~/.cargo/registry/src-other should NOT match
                // the ~/.cargo/registry/src entry).
            ];
            for case in cases {
                assert!(
                    is_path_in_extra_allowed(Path::new(case)),
                    "Should match: {case}"
                );
            }

            // Path that is NOT under any allowlist entry — verify component-boundary matching
            // ~/.cargo/registry/src-other is NOT under ~/.cargo/registry/src/
            // because Path::starts_with uses component-level comparison.
            if std::fs::metadata(format!("{home}/.cargo")).is_ok()
                && std::fs::metadata(format!("{home}/.cargo/registry/src-other")).is_err()
            {
                // Only test if the allowlist entry exists but src-other doesn't
                assert!(
                    !is_path_in_extra_allowed(Path::new("~/.cargo/registry/src-other/something")),
                    "src-other should NOT match src/ prefix"
                );
            }
        }
    }

    /// `check_path_read_allowed` permits paths under dependency source
    /// directories even when they're outside the workspace.
    #[test]
    fn check_path_read_allowed_extra_dependency_paths() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        // Temp dir spill files — should be allowed for read
        let temp_file = std::env::temp_dir().join("test-spill.txt");
        let temp_str = temp_file.to_string_lossy().to_string();
        assert!(
            check_path_read_allowed(&temp_str, &workspace).is_ok(),
            "Temp file should be allowed for read"
        );

        // Dependency path (if $HOME is set)
        if let Ok(home) = std::env::var("HOME") {
            let dep_path = format!("{home}/.cargo/registry/src/crate-0.1.0/src/lib.rs");
            assert!(
                check_path_read_allowed(&dep_path, &workspace).is_ok(),
                "Dependency path should be allowed for read"
            );

            // Tilde variant should also pass pre-canonicalization check
            let tilde_input = "~/.cargo/registry/src/crate-0.1.0/src/lib.rs";
            assert!(
                check_path_read_allowed(tilde_input, &workspace).is_ok(),
                "~-prefixed dependency path should be allowed for read"
            );
        }
    }

    /// `is_path_safe_for_workspace` blocks paths outside the workspace even
    /// when they're in [`EXTRA_READ_ALLOWED`] — the extra-read bypass is
    /// read-only and must not affect write-path checks.
    #[test]
    fn is_path_safe_for_workspace_blocks_extra_dependency_paths() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        // Temp dir spill files — should be blocked by is_path_safe_for_workspace
        let temp_file = std::env::temp_dir().join("test-spill.txt");
        let temp_str = temp_file.to_string_lossy().to_string();
        assert!(
            !is_path_safe_for_workspace(&temp_str, &workspace),
            "Temp file should be blocked by base check"
        );

        // Dependency path (if $HOME is set)
        if let Ok(home) = std::env::var("HOME") {
            let dep_path = format!("{home}/.cargo/registry/src/crate-0.1.0/src/lib.rs");
            assert!(
                !is_path_safe_for_workspace(&dep_path, &workspace),
                "Dependency path should be blocked by base check"
            );
        }
    }

    // ── scrub_credentials tests ────────────────────────────────────────────

    #[test]
    fn scrub_redacts_alphanumeric_unquoted_value() {
        let input = "API_KEY=sk-1234567890abcdef";
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact: {out}");
        assert!(
            !out.contains("1234567890abcdef"),
            "should not leak full value: {out}"
        );
        assert!(out.starts_with("API_KEY=sk-1"), "should keep prefix: {out}");
    }

    #[test]
    fn scrub_redacts_base64_unquoted_value_with_plus_and_slash() {
        // Standard Base64-encoded secret containing +, /, =
        let input = "api_key=u2FsdGVkX1+h/wZ/L3Y+Q==";
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact Base64: {out}");
        assert!(
            !out.contains("u2FsdGVkX1+h/wZ/L3Y+Q=="),
            "should not leak value: {out}"
        );
        assert!(out.starts_with("api_key=u2Fs"), "prefix: {out}");
    }

    #[test]
    fn scrub_redacts_double_quoted_value() {
        let input = r#"token: "abcdefgh1234567890""#;
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact: {out}");
        assert!(!out.contains("1234567890"), "should not leak value: {out}");
    }

    #[test]
    fn scrub_redacts_single_quoted_value() {
        let input = "password: 's3cr3t_p@ssw0rd!!'";
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact: {out}");
        assert!(!out.contains("s3cr3t"), "should not leak full: {out}");
        // Single quotes must be preserved (the bug this test guards against).
        assert_eq!(
            out, "password: 's3cr*[REDACTED]'",
            "single quotes should be preserved"
        );
    }

    #[test]
    fn scrub_single_quoted_value_equals_separator() {
        let input = "password='mysecretvalue123'";
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact: {out}");
        assert_eq!(
            out, "password='myse*[REDACTED]'",
            "single quotes preserved with equals"
        );
    }

    #[test]
    fn scrub_double_quoted_key_single_quoted_value() {
        // Edge case: the key-level optional quote in the regex can produce
        // full_match containing a double-quote from the key suffix, e.g.
        // "password": 'secretvalue1234'. The capture-group approach correctly
        // identifies this as a single-quoted value despite the double-quote
        // appearing in the full match string.
        // Note: the key-suffix " is consumed by the regex match and not
        // reconstructed — this is a pre-existing cosmetic issue also present
        // in the double-quote path, and out of scope for this fix.
        let input = r#""password": 'secretvalue123'"#;
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact: {out}");
        assert!(!out.contains("secretvalue"), "should not leak value: {out}");
        assert_eq!(
            out, "\"password: 'secr*[REDACTED]'",
            "single quotes preserved when key is double-quoted"
        );
    }

    #[test]
    fn scrub_does_not_redact_short_unquoted_values() {
        let input = "key=short";
        let out = scrub_credentials(input);
        assert_eq!(out, input, "short values should not redact");
    }

    #[test]
    fn scrub_handles_colon_separator() {
        let input = "bearer: eyJhbGciOiJIUzI1NiJ9";
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "should redact: {out}");
        assert!(!out.contains("eyJhbG"), "should not leak: {out}");
    }

    #[test]
    fn scrub_redacts_hyphen_key_variants() {
        let input = "user-key=abcdefgh12345678";
        let out = scrub_credentials(input);
        assert!(out.contains("[REDACTED]"), "user-key: {out}");
        assert!(
            !out.contains("12345678"),
            "should not leak full value: {out}"
        );
        assert!(
            out.starts_with("user-key=abcd"),
            "should keep prefix: {out}"
        );
    }

    #[test]
    fn scrub_keeps_non_secret_lines_unchanged() {
        let input = "normal line with = equals and / slash";
        let out = scrub_credentials(input);
        assert_eq!(out, input, "non-secret line must be unchanged");
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

    // ── Path resolution tests ──────────────────────────────────────────

    /// Create a temporary workspace and return `(TempDir, canonical_ws_path)`.
    /// The `TempDir` guard must be held alive for the test duration.
    async fn test_workspace() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().expect("tempdir");
        let ws_raw = tmp.path().join("ws");
        tokio::fs::create_dir(&ws_raw).await.unwrap();
        let ws = tokio::fs::canonicalize(&ws_raw).await.unwrap();
        (tmp, ws)
    }

    #[tokio::test]
    async fn resolve_read_target_file_exists() {
        let (_tmp, ws) = test_workspace().await;
        let file_path = ws.join("existing.txt");
        tokio::fs::write(&file_path, "hello").await.unwrap();

        let result = resolve_read_target(&ws, "existing.txt").await;
        assert!(
            result.is_ok(),
            "Should resolve existing file: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        let canonical = tokio::fs::canonicalize(&file_path).await.unwrap();
        assert_eq!(resolved, canonical, "should resolve to the canonical path");
    }

    #[tokio::test]
    async fn resolve_read_target_existing_subdirectory_without_trailing_slash() {
        let (_tmp, ws) = test_workspace().await;
        let sub = ws.join("nested");
        tokio::fs::create_dir_all(&sub).await.unwrap();
        tokio::fs::write(sub.join("leaf.txt"), "hello")
            .await
            .unwrap();

        let result = resolve_read_target(&ws, "nested").await;
        assert!(
            result.is_ok(),
            "Should resolve existing directory without trailing slash: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        let canonical = tokio::fs::canonicalize(&sub).await.unwrap();
        assert_eq!(resolved, canonical);
    }

    #[tokio::test]
    async fn resolve_read_target_file_not_found() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_read_target(&ws, "nonexistent.txt").await;
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("File not found"),
            "Should report File not found: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_read_target_permission_denied() {
        use std::os::unix::fs::PermissionsExt;

        let (_tmp, ws) = test_workspace().await;
        let restricted_dir = ws.join("secret");
        tokio::fs::create_dir(&restricted_dir).await.unwrap();
        let file_path = restricted_dir.join("file.txt");
        tokio::fs::write(&file_path, "secret").await.unwrap();

        // Remove search permission from directory so canonicalize can't enter it
        std::fs::set_permissions(&restricted_dir, std::fs::Permissions::from_mode(0o000)).unwrap();

        let result = resolve_read_target(&ws, "secret/file.txt").await;

        // Restore permissions so TempDir can clean up
        let _ = std::fs::set_permissions(&restricted_dir, std::fs::Permissions::from_mode(0o755));

        assert!(result.is_err(), "Should fail with Permission denied");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Permission denied"),
            "Should mention Permission denied: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_read_target_symlink_resolution() {
        let (_tmp, ws) = test_workspace().await;
        let real_file = ws.join("real.txt");
        tokio::fs::write(&real_file, "content").await.unwrap();
        let link = ws.join("link.txt");
        std::os::unix::fs::symlink(&real_file, &link).unwrap();

        let result = resolve_read_target(&ws, "link.txt").await;
        assert!(result.is_ok(), "Should resolve symlink: {:?}", result.err());
        let resolved = result.unwrap();
        let canonical_real = tokio::fs::canonicalize(&real_file).await.unwrap();
        assert_eq!(
            resolved, canonical_real,
            "symlink should resolve to the real file"
        );
    }

    #[tokio::test]
    async fn resolve_read_target_extra_allowed_path() {
        let (_tmp, ws) = test_workspace().await;
        let spill_file = std::env::temp_dir().join(format!(
            "mahbot_test_resolve_read_{}.txt",
            std::process::id()
        ));
        tokio::fs::write(&spill_file, "spill content")
            .await
            .unwrap();
        let spill_str = spill_file.to_string_lossy().to_string();

        let result = resolve_read_target(&ws, &spill_str).await;
        let _ = tokio::fs::remove_file(&spill_file).await;

        assert!(
            result.is_ok(),
            "Should allow extra read paths (e.g. /tmp): {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn resolve_read_target_non_canonicalized_workspace_root() {
        // When workspace_root is not canonicalized (e.g. /tmp vs /private/tmp on macOS),
        // reading via relative path should still succeed because resolve_tool_path_with_base
        // joins the (non-canonical) root with the relative path, and canonicalize resolves
        // the full path before the post-check.
        let tmp = TempDir::new().expect("tempdir");
        let ws_dir = tmp.path().join("ws");
        tokio::fs::create_dir(&ws_dir).await.unwrap();
        let file_path = ws_dir.join("hello.txt");
        tokio::fs::write(&file_path, "world").await.unwrap();

        // Use the non-canonicalized ws_dir as workspace_root.
        let result = resolve_read_target(&ws_dir, "hello.txt").await;
        assert!(
            result.is_ok(),
            "Read should succeed even with non-canonicalized root: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        // The resolved path is canonical; verify the content is correct.
        let content = tokio::fs::read_to_string(&resolved).await.unwrap();
        assert_eq!(content, "world", "should read the correct file content");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_in_existing_dir() {
        let (_tmp, ws) = test_workspace().await;
        let subdir = ws.join("subdir");
        tokio::fs::create_dir(&subdir).await.unwrap();

        let result = resolve_write_target(&ws, "subdir/new_file.rs", false).await;
        assert!(
            result.is_ok(),
            "Should resolve new file in existing dir: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert!(resolved.starts_with(&ws), "Path should be within workspace");
        assert_eq!(resolved.file_name().unwrap(), "new_file.rs");
        // The file should NOT exist yet
        assert!(!resolved.exists(), "File should not exist yet");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_new_dir_with_ensure_parent() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_write_target(&ws, "a/b/c/new_file.rs", true).await;
        assert!(
            result.is_ok(),
            "Should create parent directories: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert!(resolved.starts_with(&ws), "Path should be within workspace");
        assert_eq!(resolved.file_name().unwrap(), "new_file.rs");
        // Parent chain should exist
        assert!(
            ws.join("a/b/c").exists(),
            "Parent directories should be created"
        );
        // File should NOT exist yet
        assert!(!resolved.exists(), "File should not exist yet");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_new_dir_no_ensure_parent() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_write_target(&ws, "nonexistent_dir/new_file.rs", false).await;
        assert!(result.is_err(), "Should fail when parent doesn't exist");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to resolve file path")
                || err.to_string().contains("No such file or directory"),
            "Should mention resolution failure: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_write_target_symlink_refusal() {
        let (_tmp, ws) = test_workspace().await;
        // Create a symlink at the file target location
        let link = ws.join("malicious_link.txt");
        std::os::unix::fs::symlink("/etc/passwd", &link).unwrap();

        let result = resolve_write_target(&ws, "malicious_link.txt", false).await;
        assert!(result.is_err(), "Should refuse to write through symlink");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("symlink"),
                "Error should mention symlink: {e}"
            );
        }
    }

    #[tokio::test]
    async fn resolve_write_target_outside_workspace_rejected() {
        let (_tmp, ws) = test_workspace().await;
        let outside = std::env::temp_dir().join(format!(
            "mahbot_test_write_outside_{}.txt",
            std::process::id()
        ));

        let result = resolve_write_target(&ws, &outside.to_string_lossy(), false).await;
        assert!(result.is_err(), "Should reject write outside workspace");
    }
}
