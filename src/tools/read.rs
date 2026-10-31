//! The `read` tool: line-numbered content reads, tree-sitter `symbols` and
//! `zoom`, directory listings, credential scrubbing for sensitive paths, and
//! the one-metadata-sniff decision that hands a document container to
//! `tools::read_document`.
//!
//! Not the document converter itself — the readers live in `crate::document`
//! and the tool-facing wrapper in `tools::read_document`; this tool only
//! decides that a content read is a document and delegates.

use std::path::{Path, PathBuf};

use super::listing::{ListingEntry, format_listing, human_readable_size};
use super::path::PathAccess;
use crate::tools::search::SearchTool;
use crate::tools::shell::try_spill_to_file;
use crate::util::TOOL_OUTPUT_BUDGET_BYTES;
use crate::util::tree_sitter::supported_extensions;
use crate::{Tool, Workspace};
use async_trait::async_trait;
use serde_json::json;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator, Tree};

/// The `read` tool. `access` is the caller's [`PathAccess`], fixed when the
/// role's toolset is built: a guest Assistant's read is confined to its
/// workspace, a pipeline role's reaches the read allowlist too, and the admin
/// Assistant's reaches any path the shell could name.
pub struct ReadTool {
    access: PathAccess,
}

impl ReadTool {
    #[must_use]
    pub(crate) const fn new(access: PathAccess) -> Self {
        Self { access }
    }

    /// General read: workspace plus dependency-source / temp-file paths.
    #[must_use]
    pub fn general() -> Self {
        Self::new(PathAccess::Allowlisted)
    }
}

/// Recognized sensitive file extensions whose read output should be scrubbed for credentials.
const SENSITIVE_EXTENSIONS: &[&str] = &["cer", "crt", "env", "key", "p12", "pem", "pfx"];

/// Exact file names (lowercased) whose read output is scrubbed for credentials.
const SENSITIVE_FILE_NAMES: &[&str] = &[
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "settings.xml",
    "settings-security.xml",
    "gradle.properties",
    "credentials.toml",
];

/// File-name prefixes whose read output is scrubbed (e.g. `init.gradle`, `init.gradle.kts`).
const SENSITIVE_FILE_NAME_PREFIXES: &[&str] = &["init.gradle"];

/// File paths whose read output should be scrubbed for credentials (`.env`, certs, keys,
/// credential config files). Other extensions (e.g. `.rs`, `.md`) are left intact so the
/// model sees source accurately.
#[must_use]
fn is_sensitive_file_path(path: &str) -> bool {
    let Some(file_name) = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
    else {
        return true;
    };
    let lower = file_name.to_ascii_lowercase();

    // Exact-name / prefix matches (dotfiles like `.netrc` with no extension, and
    // credential config files like `settings.xml` / `gradle.properties`) are
    // checked first because the rsplit_once('.') extension match below cannot
    // classify them.
    if SENSITIVE_FILE_NAMES.contains(&lower.as_str()) {
        return true;
    }
    if SENSITIVE_FILE_NAME_PREFIXES
        .iter()
        .any(|p| lower.starts_with(*p))
    {
        return true;
    }

    // Single rsplit_once handles both dotfiles (.env, .env.local) and regular extensions (.pem, .key).
    // The `name == ".env"` arm catches `.env.local`-style dotfile prefixes.
    match lower.rsplit_once('.') {
        Some((name, ext)) => name == ".env" || SENSITIVE_EXTENSIONS.contains(&ext),
        None => false,
    }
}

/// Classify a file's bytes for the image-aware Read behaviour.
enum SniffedImage {
    /// A native raster (PNG/JPEG/WebP) that can be attached as a native image.
    /// The base annotation is claim-neutral (no "attached" sentence) and
    /// dims-less — the authoritative post-EXIF dims come from the payload the
    /// agent loop actually injects.
    Native { label: String },
    /// A recognised-but-unsupported image format (GIF/BMP/...). Reported
    /// gracefully rather than decoded to garbage.
    Unsupported { label: String },
}

/// Cheap magic-sniff `bytes` for a raster image. Returns `None` for non-images
/// (SVG, text, arbitrary binary) so those keep the text/lossy path. Only
/// PNG/JPEG/WebP are treatable as native; any other recognised format (GIF,
/// BMP, ...) is reported as unsupported. No decode is performed here — the
/// decode is deferred to [`ReadTool::image_payload`], the only decoder of a
/// native image file.
#[must_use]
fn sniff_read_image(bytes: &[u8]) -> Option<SniffedImage> {
    let format = image::guess_format(bytes).ok()?;
    match crate::util::image_format_native_label(format) {
        Some(label) => Some(SniffedImage::Native {
            label: label.to_string(),
        }),
        None => Some(SniffedImage::Unsupported {
            label: format!("{format:?}").to_ascii_uppercase(),
        }),
    }
}

/// Regular-file-only file-magic gate: true only when `path` (a regular file)
/// begins with PNG/JPEG/WebP magic. Reuses [`sniff_read_image`] so the
/// native-format decision has a single source — no duplicated format-match
/// arms, no drift risk. Returns `false` for text, unsupported formats, and any
/// special file, without a full read or decode. This is the robust gate
/// `image_payload` uses instead of the annotation wording.
async fn is_native_image_file(path: &Path) -> bool {
    use tokio::io::AsyncReadExt;
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return false;
    };
    // A special file is never reopened — a stream cannot be sampled without
    // blocking.
    if !meta.is_file() {
        return false;
    }
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return false;
    };
    let mut buf = [0u8; 64];
    let Ok(n) = file.read(&mut buf).await else {
        return false;
    };
    matches!(
        sniff_read_image(&buf[..n]),
        Some(SniffedImage::Native { .. })
    )
}

/// Produce the annotation text for a content-mode read that discovered a raster
/// image. Shared between the UTF-8 and binary fallback paths so the annotation
/// is byte-identical either way.
#[must_use]
fn image_read_annotation(path: &Path, kind: SniffedImage) -> String {
    match kind {
        SniffedImage::Native { label } => {
            // Claim-neutral, dims-less base: the agent loop appends the
            // 'attached'/'already attached' qualifier (with the authoritative
            // post-EXIF dims) from the payload it actually injects, so a read
            // that cannot be injected (over-cap, decode failure) never falsely
            // claims an attachment.
            format!("Read image file {} ({label}).", path.display())
        }
        SniffedImage::Unsupported { label } => {
            format!(
                "Read image file {} — unsupported image format \
                 ({label}); cannot attach as a native image (only PNG, JPEG, \
                 WebP supported).",
                path.display()
            )
        }
    }
}

/// Shared content-mode gate: a raster read (detected by magic bytes, never
/// the extension) yields the image annotation instead of text.
fn sniff_guard(resolved_path: &Path, bytes: &[u8]) -> Option<String> {
    sniff_read_image(bytes).map(|kind| image_read_annotation(resolved_path, kind))
}

/// Candidate paths for a literal `path` that `resolve_read_target` could not
/// resolve (a typo/missing path), queried by filename. Its only caller is
/// [`resolve_content_read`].
async fn find_recovery_candidates(ws: &Workspace, path: &str) -> Vec<String> {
    let hint = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(path);
    SearchTool::find_file_paths(ws, hint, 8)
        .await
        .unwrap_or_default()
}

/// The `[Recovered path: ...]` note shown when a literal `path` was recovered to
/// a single high-confidence filename match. Shared so the annotation path and
/// the injection path word it identically.
#[must_use]
fn recovery_note(path: &str, recovered: &str) -> String {
    format!("[Recovered path: requested '{path}', using '{recovered}']")
}

/// Outcome of resolving a content-read `path`: the canonical path to read, plus
/// the recovery note when the literal path was recovered to a unique match.
pub(super) struct ResolvedRead {
    pub(super) path: PathBuf,
    pub(super) recovery_note: Option<String>,
}

/// Resolve a content-read `path` to a file the read should open — the one
/// resolution every content read goes through, whatever it ends up doing with
/// the file. When the literal path does not exist (and only then) a single
/// high-confidence filename match is recovered, so a typo'd raster is still
/// attached by `image_payload` and a typo'd document is still converted by
/// [`super::read_document::read_document`]; both receive the original `path`,
/// hence the note travelling with the result.
///
/// The error is the one `execute` shows: the resolution failure, with the
/// `Did you mean:` candidates appended when the search found several.
pub(super) async fn resolve_content_read(
    ws: &Workspace,
    path: &str,
    access: PathAccess,
) -> anyhow::Result<ResolvedRead> {
    match super::path::resolve_read_target(ws.as_path(), path, access).await {
        Ok(resolved) => Ok(ResolvedRead {
            path: resolved,
            recovery_note: None,
        }),
        // Recovery is for a typo's sake, and only where the workspace is the
        // frame: the admin reaches the whole machine, so a path that is not
        // there is answered as missing rather than answered with a different
        // file of the same name from the workspace.
        Err(e)
            if access == PathAccess::Unrestricted || !e.to_string().contains("File not found") =>
        {
            Err(e)
        }
        Err(e) => {
            let matches = find_recovery_candidates(ws, path).await;
            match matches.len() {
                1 => {
                    let recovered = &matches[0];
                    let resolved =
                        super::path::resolve_read_target(ws.as_path(), recovered, access).await?;
                    Ok(ResolvedRead {
                        path: resolved,
                        recovery_note: Some(recovery_note(path, recovered)),
                    })
                }
                0 => Err(e),
                _ => Err(anyhow::anyhow!(
                    "{e}\nDid you mean:\n  {}",
                    matches.join("\n  ")
                )),
            }
        }
    }
}

/// The reader a read's `mode` argument selects — the one classification the
/// content gates, [`read_resolved`]'s dispatch and the advertised enum all refer
/// to, so none of them can disagree about the same call.
#[derive(PartialEq, Eq)]
enum ReadMode {
    Content,
    Symbols,
    Zoom,
}

/// Classify `args`'s `mode`: the default, and anything that is not one of the two
/// modes with a reader of their own, is a content read.
#[must_use]
fn read_mode(args: &serde_json::Value) -> ReadMode {
    match super::get_opt_str(args, "mode") {
        Some("symbols") => ReadMode::Symbols,
        Some("zoom") => ReadMode::Zoom,
        _ => ReadMode::Content,
    }
}

/// Build the read tool's parameter schema. `path_desc` states this read's own
/// path boundary, so each [`PathAccess`] advertises the one it enforces.
fn read_parameters_schema(path_desc: &str) -> serde_json::Value {
    super::tool_params_schema(
        &json!({
            "path": {
                "type": "string",
                "description": path_desc
            },
            "mode": {
                "type": "string",
                "enum": ["content", "symbols", "zoom"],
                "description": "Read mode. 'content' (default): line-numbered file read — large outputs are truncated to a ~5 KB budget — or, for a raster image (PNG, JPEG, WebP), attaches it to the conversation as a native image instead; a PDF or Office document (Word, Excel, PowerPoint) is converted to text instead of its raw bytes (the tool description says what a converted document yields). 'symbols': list all top-level AST symbols with line ranges. 'zoom': extract a single symbol's source by name. 'symbols'/'zoom' work for supported code formats only.",
                "default": "content"
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name for zoom mode. Required when mode is 'zoom'.",
                "minLength": 1
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (1-based, default: 1). Not used for a converted document.",
                "default": 1,
                "minimum": 1
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: all). Not used for a converted document.",
                "minimum": 1
            }
        }),
        &["path"],
    )
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> String {
        let ppt = crate::reader_output::ppt_description_substitutions();
        // The chart block prints the kind it read where this spells `<kind>`, a
        // series its own name where this spells `<name>` — or its number — and a
        // bubble's size where this spells `<size>`.
        let chart_header = crate::reader_output::chart_header(Some("<kind>"));
        let chart_none = crate::reader_output::chart_header(None);
        let chart_series = crate::reader_output::chart_series(Some("<name>"), 0);
        let chart_series_numbered = crate::reader_output::chart_series(None, 0);
        let chart_size = crate::reader_output::chart_size("<size>");
        // The kind words a PDF form field is listed with, and the words its line
        // prints, spelled the way the reading prints them.
        let pdf_field_kinds = crate::docgen::pdf_field_kinds().spelled();
        let form_words = crate::docgen::pdf_form_words();
        // The path boundary, in the wording of this read's own access level —
        // the skeleton is shared, so the three statements cannot drift from
        // each other's structure.
        let path_policy = crate::prompt::load_prompt(match self.access {
            PathAccess::Workspace => "tool/read_paths_workspace.md",
            PathAccess::Allowlisted => "tool/read_paths_allowlisted.md",
            PathAccess::Unrestricted => "tool/read_paths_admin.md",
        });
        let unshown_report = crate::reader_output::unshown_report_examples();
        let mut pairs: Vec<(&str, &str)> = vec![
            ("{{path_policy}}", &path_policy),
            (
                "{{doc_headers_footers}}",
                crate::reader_output::DOC_HEADERS_FOOTERS,
            ),
            ("{{doc_footnotes}}", crate::reader_output::DOC_FOOTNOTES),
            ("{{doc_endnotes}}", crate::reader_output::DOC_ENDNOTES),
            ("{{doc_comments}}", crate::reader_output::DOC_COMMENTS),
            ("{{text_box_mark}}", crate::reader_output::TEXT_BOX),
            ("{{chart_sheet_mark}}", crate::reader_output::CHART_SHEET),
            ("{{chart_header}}", &chart_header),
            ("{{chart_none}}", &chart_none),
            ("{{chart_series_numbered}}", &chart_series_numbered),
            ("{{chart_title}}", crate::reader_output::CHART_TITLE),
            ("{{chart_text}}", crate::reader_output::CHART_TEXT),
            ("{{chart_label}}", crate::reader_output::CHART_LABEL),
            ("{{chart_series}}", &chart_series),
            ("{{chart_size}}", &chart_size),
            ("{{pdf_field_kinds}}", &pdf_field_kinds),
            ("{{pdf_form_fields}}", &form_words.fields),
            ("{{pdf_no_value}}", &form_words.no_value),
            ("{{pdf_not_fillable}}", &form_words.not_fillable),
            ("{{mark_stored}}", crate::reader_output::MARK_STORED.trim()),
            ("{{chart_no_values}}", crate::reader_output::CHART_NO_VALUES),
            (
                "{{chart_attached}}",
                crate::reader_output::CHART_VALUES_ATTACHED,
            ),
            (
                "{{chart_unreadable}}",
                crate::reader_output::CHART_UNREADABLE,
            ),
            ("{{chart_extended}}", crate::reader_output::CHART_EXTENDED),
            ("{{unshown_report}}", &unshown_report),
        ];
        pairs.extend(ppt.iter().map(|(key, value)| (*key, value.as_str())));
        crate::prompt::substitute(&crate::prompt::load_prompt("tool/read.md"), &pairs)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        read_parameters_schema(match self.access {
            PathAccess::Workspace => "Path to the file or directory within the personal workspace.",
            PathAccess::Allowlisted | PathAccess::Unrestricted => "Path to the file or directory.",
        })
    }

    /// The raw-text read. Documents are converted by
    /// [`Tool::execute_with_payloads`] (their pages need images), so a direct
    /// `execute` call gets the lossy path — the agent loop always uses the hook.
    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        execute_read(ws, args, self.access).await
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        // The read tool scrubs internally in `read_resolved` based on the
        // resolved path (see `scrub_if_sensitive`) — an args-only decision here
        // cannot see symlink targets or typo-recovered files. Returning `false`
        // prevents the central agent-level pass from double-scrubbing.
        false
    }

    fn side_effects(&self) -> bool {
        false // read-only file inspection
    }

    fn format_output(&self, output: &str) -> String {
        if output.len() <= TOOL_OUTPUT_BUDGET_BYTES {
            return output.to_string();
        }

        // The output has a header line like "[N lines total]" or
        // "[Lines X-Y of Z]" followed by "\n" and numbered lines.
        // Find that separator and keep the header intact.
        if let Some(nl) = output.find('\n') {
            let header = &output[..nl];
            let expected = parse_header_line_count(header);
            if expected > 0 {
                let body = &output[nl + 1..];

                // Worst-case marker length — `omitted ≤ expected` guarantees the actual marker never exceeds this
                let marker_budget = format!("\n... ({expected} lines omitted)").len();
                let body_budget =
                    TOOL_OUTPUT_BUDGET_BYTES.saturating_sub(header.len() + marker_budget + 1);

                // Truncate at last complete line boundary within budget
                let cut = body.floor_char_boundary(body_budget.min(body.len()));
                let last_nl = body[..cut].rfind('\n').unwrap_or(cut);
                let kept_body = &body[..last_nl];

                let kept = if kept_body.is_empty() {
                    0
                } else {
                    kept_body.bytes().filter(|&b| b == b'\n').count() + 1
                };
                let omitted = expected.saturating_sub(kept);
                let marker = format!("\n... ({omitted} lines omitted)");

                return format!("{header}\n{kept_body}{marker}");
            }
        }

        // Fallback (lossy binary output, etc.): standard head+tail truncation
        crate::util::truncate_tool_output(output)
    }

    /// A document conversion is a by-product of this execution, so the read tool
    /// overrides the combined hook: one resolution feeds both the document
    /// decision and the ordinary read, and a document is converted — images
    /// encoded included — in that single pass.
    async fn execute_with_payloads(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<crate::tools::ToolOutput> {
        // Only a content read of a literal path can be a document: symbols/zoom
        // run tree-sitter on source files, and a wildcard path is a listing.
        // Everything else takes the ordinary read below.
        if read_mode(&args) == ReadMode::Content
            && let Ok(path) = super::get_str(&args, "path")
            && !super::path::is_wildcard_read(ws.as_path(), path, self.access).await
        {
            let res = resolve_content_read(ws, path, self.access).await?;
            // A read that is not a document has paid one extra metadata + 16-byte
            // head read for the sniff; sharing the converter's own detection
            // (rather than an extension pre-filter that could drift from it) is
            // what that buys. The raster `image_payload` below still resolves the
            // path for itself, as it always did.
            if let Some(out) = super::read_document::read_document(ws, &res, self.access).await? {
                return Ok(out);
            }
            let text = read_resolved(&res, &args).await?;
            return Ok(super::with_image_payload(self, text, ws, &args).await);
        }
        // The ordinary read, and the payload the default hook pairs with it —
        // one shared site ([`crate::tools::with_image_payload`]) so an override
        // that produced its text here cannot drift from the default flow.
        let text = Tool::execute(self, ws, args.clone()).await?;
        Ok(super::with_image_payload(self, text, ws, &args).await)
    }

    async fn image_payload(
        &self,
        ws: &Workspace,
        args: &serde_json::Value,
    ) -> Option<crate::tools::ImagePayload> {
        read_image_payload(ws, args, self.access).await
    }
}

/// Produce the image payload for a read, resolving the path under the caller's
/// [`PathAccess`] — the same boundary the read itself used.
async fn read_image_payload(
    ws: &Workspace,
    args: &serde_json::Value,
    access: PathAccess,
) -> Option<crate::tools::ImagePayload> {
    // Robust file-magic gate (NOT the annotation wording): only a PNG/JPEG/
    // WebP raster opens the decode+encode below. This runs AFTER path
    // resolution, so text/unsupported reads still pay resolve + metadata +
    // a 64-byte sniff — but never a full read/decode of the file.
    let path = super::get_str(args, "path").ok()?.to_string();
    // Image behaviour is content-mode only (symbols/zoom run tree-sitter).
    if read_mode(args) != ReadMode::Content {
        return None;
    }
    if super::path::is_wildcard_read(ws.as_path(), &path, access).await {
        return None;
    }
    // Resolve the literal path, or — for a typo'd path that `execute`
    // already recovered to a unique match — the recovered path, so a
    // recovered raster is attached rather than only annotated.
    let res = resolve_content_read(ws, &path, access).await.ok()?;
    if !is_native_image_file(&res.path).await {
        return None;
    }
    // The compressed encode applies EXIF orientation and reports the
    // post-EXIF/post-resize dims + source-format label; its metadata-first
    // `is_file()` guard keeps a special file from being reopened.
    let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&res.path)
        .await
        .ok()?;
    Some(crate::tools::ImagePayload::from_compressed_meta(
        &res.path,
        meta,
        res.recovery_note,
        crate::tools::ImagePayloadSource::Read,
    ))
}

/// Scrub a file-read's output when the *resolved* file is credential-bearing.
/// (A converted document scrubs unconditionally instead: this path-based rule
/// sees only the container's path, never the text extracted from it.)
///
/// Deciding here rather than from `should_scrub_output` is what makes the
/// tier correct: `resolved_path` is canonical, so it already accounts for
/// symlink targets and for typo recovery to a different file — neither is
/// visible in the raw `path` argument. Directory listings are not scrubbed
/// (file names are not secret values). Returning `false` from
/// `should_scrub_output` prevents the central agent-level pass from
/// double-scrubbing (same integration pattern as `ShellTool`).
fn scrub_if_sensitive(resolved_path: &Path, body: String) -> String {
    if is_sensitive_file_path(&resolved_path.to_string_lossy()) {
        crate::util::scrub_credentials(&body)
    } else {
        body
    }
}

/// The tail a read's body goes through: scrub it when the resolved file is
/// credential-bearing, then prefix the typo-recovery note, if any.
fn finish_read_body(resolved_path: &Path, body: String, recovery_note: Option<&str>) -> String {
    let body = scrub_if_sensitive(resolved_path, body);
    super::with_recovery_note(recovery_note, body)
}

/// Read a resolved file path (content, symbols, or zoom mode) — the shared tail
/// of a non-document read: [`execute_read`] and the command line's document
/// reader both return through it for a file the converter did not take, so a
/// step added to one reaches the other. Empty `args` is the read tool's own
/// default mode, which is what the command line's reader asks for.
pub(super) async fn read_resolved(
    res: &ResolvedRead,
    args: &serde_json::Value,
) -> anyhow::Result<String> {
    let resolved_path = res.path.as_path();
    let recovery_note = res.recovery_note.as_deref();
    let meta = tokio::fs::metadata(resolved_path)
        .await
        .map_err(|e| path_io_error(resolved_path, &e, "Failed to read file metadata"))?;
    if meta.is_dir() {
        return list_directory(resolved_path).await;
    }
    super::check_size_within(&meta, super::MAX_FILE_SIZE_BYTES, "File too large")?;

    let body = match read_mode(args) {
        ReadMode::Symbols => execute_symbols(resolved_path).await?,
        ReadMode::Zoom => execute_zoom(resolved_path, args).await?,
        ReadMode::Content => execute_content(resolved_path, args).await?,
    };

    Ok(finish_read_body(resolved_path, body, recovery_note))
}

/// Wildcard path: return matching workspace files instead of failing open.
async fn recover_wildcard_path(ws: &Workspace, path: &str) -> anyhow::Result<String> {
    if !crate::workspace_index::registry_initialized() {
        anyhow::bail!(
            "Wildcard path '{path}' requires the workspace search index, which is unavailable."
        );
    }
    let matches = SearchTool::find_file_paths(ws, path, 20).await?;
    if matches.is_empty() {
        anyhow::bail!(
            "No files matching wildcard path '{path}' found in workspace.\n\
                 Use the search tool with mode='files' to browse paths."
        );
    }
    let mut output = format!("Wildcard path '{path}' matched:\n");
    for m in &matches {
        output.push_str("  ");
        output.push_str(m);
        output.push('\n');
    }
    Ok(output)
}

/// Execute the standard content read mode.
async fn execute_content(resolved_path: &Path, args: &serde_json::Value) -> anyhow::Result<String> {
    match tokio::fs::read_to_string(resolved_path).await {
        Ok(contents) => {
            // A raster can be valid UTF-8 (e.g. a minimal GIF patch / a
            // NUL-padded header) — sniff magic bytes before treating it as
            // text so it gets the image annotation rather than rendering as
            // garbage. SVG, source, and other text are not image magic and
            // stay readable.
            if let Some(annotation) = sniff_guard(resolved_path, contents.as_bytes()) {
                return Ok(annotation);
            }
            Ok(format_content(&contents, args)?)
        }
        Err(e) => {
            // Not valid UTF-8 — read raw bytes and try to extract text
            let bytes = tokio::fs::read(resolved_path).await.map_err(|ee| {
                anyhow::anyhow!(
                    "io: cannot read {}: failed to read file: {ee} \
                     (initial error: {e}) — hint: verify the file exists and is readable",
                    resolved_path.display()
                )
            })?;

            // Content-sniff (magic bytes, never the extension) so SVG and
            // other text files stay readable as text and only real raster
            // images take the image path.
            if let Some(annotation) = sniff_guard(resolved_path, &bytes) {
                return Ok(annotation);
            }

            // Lossy UTF-8 for every file but a background session's own output,
            // which is a program's output and takes the shell's reading of one.
            let text = crate::tools::shell::decode_session_output(resolved_path, &bytes)
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
            Ok(text)
        }
    }
}

/// List all top-level AST symbols with line ranges.
async fn execute_symbols(resolved_path: &Path) -> anyhow::Result<String> {
    let ctx = prepare_symbol_query(resolved_path, "symbol extraction").await?;

    let symbols = collect_symbols(&ctx.ps, &ctx.query);
    let mut lines: Vec<String> = symbols
        .iter()
        .map(|s| {
            let kind_label = symbol_kind_label(&s.kind);
            format!(
                "  {kind_label} `{}` ({}-{})",
                s.name, s.start_line, s.end_line
            )
        })
        .collect();
    lines.sort();
    lines.dedup();

    let filename = display_filename(resolved_path);
    let output = if lines.is_empty() {
        format!("[No symbols found in {filename}]")
    } else {
        format!("[Symbols in {filename}]\n{}", lines.join("\n"))
    };

    Ok(output)
}

/// Extract a single named symbol's complete source.
async fn execute_zoom(resolved_path: &Path, args: &serde_json::Value) -> anyhow::Result<String> {
    let symbol_name = match super::get_opt_str(args, "symbol") {
        Some(s) if !s.is_empty() => s,
        _ => {
            anyhow::bail!("Missing 'symbol' parameter — required for zoom mode");
        }
    };

    let ctx = prepare_symbol_query(resolved_path, "zoom").await?;

    // Find the named symbol via query-based matching (restricts to declarations only)
    let root_node = ctx.ps.tree.root_node();
    let mut qcursor = QueryCursor::new();
    let mut qmatches = qcursor.matches(&ctx.query, root_node, ctx.ps.source.as_bytes());
    let mut found_node = None;
    qmatches.advance();
    while let Some(m) = qmatches.get() {
        for c in m.captures {
            if let Ok(name) = c.node.utf8_text(ctx.ps.source.as_bytes())
                && name == symbol_name
            {
                // Found the matching declaration — grab parent node for zoom
                found_node = c.node.parent();
                break;
            }
        }
        if found_node.is_some() {
            break;
        }
        qmatches.advance();
    }

    let Some(node) = found_node else {
        let suggestions = symbol_suggestions(&ctx.ps, &ctx.query, symbol_name);
        if suggestions.is_empty() {
            anyhow::bail!(
                "Symbol '{symbol_name}' not found in {}",
                display_filename(resolved_path),
            );
        }
        anyhow::bail!(
            "Symbol '{symbol_name}' not found in {}. Did you mean: {}",
            display_filename(resolved_path),
            suggestions.join(", ")
        );
    };

    let start = node.start_position().row + 1;
    let end = node.end_position().row + 1;
    let byte_range = node.byte_range();
    let extracted = &ctx.ps.source[byte_range.start..byte_range.end];
    let kind_label = symbol_kind_label(node.kind());

    Ok(format!(
        "[Symbol: {kind_label} `{symbol_name}` (lines {start}-{end})]\n{extracted}",
    ))
}

/// Suggest symbol names when zoom lookup fails.
fn symbol_suggestions(ps: &ParsedSource, query: &Query, wanted: &str) -> Vec<String> {
    // Use collect_symbols for cursor iteration, then filter out any "?"
    // placeholders that were substituted for non-UTF-8 bytes. This preserves
    // the original behavior where unrepresentable identifiers were silently
    // skipped (old code used `if let Ok(name) = utf8_text(...)`).
    let symbols = collect_symbols(ps, query);
    let mut names: Vec<String> = symbols
        .into_iter()
        .map(|s| s.name)
        .filter(|n| n != "?")
        .collect();
    names.sort();
    names.dedup();

    let wanted_lc = wanted.to_ascii_lowercase();
    names.sort_by_cached_key(|name| {
        let name_lc = name.to_ascii_lowercase();
        let tier = if name_lc == wanted_lc {
            0 // exact match
        } else if name_lc.starts_with(&wanted_lc) || wanted_lc.starts_with(&name_lc) {
            1 // prefix-related
        } else {
            2 // everything else
        };
        (tier, name_lc)
    });
    names.truncate(8);
    names
}

/// The read tool's raw-text read, reached through `Tool::execute`: a wildcard
/// listing, or a file the document hook did not take. `access` is the caller's
/// [`PathAccess`], the boundary [`super::path::resolve_read_target`] applies.
async fn execute_read(
    ws: &Workspace,
    args: serde_json::Value,
    access: PathAccess,
) -> anyhow::Result<String> {
    let path = super::get_str(&args, "path")?.to_string();

    if super::path::is_wildcard_read(ws.as_path(), &path, access).await {
        return recover_wildcard_path(ws, &path).await;
    }

    let res = resolve_content_read(ws, &path, access).await?;
    read_resolved(&res, &args).await
}

/// Format file contents for content-mode output (line numbering + offset/limit).
fn format_content(contents: &str, args: &serde_json::Value) -> anyhow::Result<String> {
    let lines: Vec<&str> = contents.lines().collect();
    let total = lines.len();

    if total == 0 {
        return Ok(String::new());
    }

    let offset = super::get_opt_u64(args, "offset")?.map_or(0, |v| {
        usize::try_from(v).unwrap_or(usize::MAX).saturating_sub(1)
    });
    let start = offset.min(total);

    let end = match super::get_opt_u64(args, "limit")? {
        Some(l) => {
            let limit = usize::try_from(l).unwrap_or(usize::MAX);
            (start.saturating_add(limit)).min(total)
        }
        None => total,
    };

    if start >= end {
        return Ok(format!("[No lines in range, file has {total} lines]"));
    }

    let numbered: String = lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start + i + 1, line))
        .collect::<Vec<_>>()
        .join("\n");

    let partial = start > 0 || end < total;
    let summary = if partial {
        format!("[Lines {}-{} of {total}]", start + 1, end)
    } else {
        format!("[{total} lines total]")
    };

    Ok(format!("{summary}\n{numbered}"))
}

// ── Tree-sitter infrastructure ────────────────────────────────────────

/// Extract a human-readable filename from a path for use in display messages.
///
/// Returns `"?"` if the path has no filename component or if the filename
/// is not valid UTF-8.
fn display_filename(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
}

#[derive(Debug)]
struct ParsedSource {
    source: String,
    language: Language,
    symbol_query: &'static str,
    tree: Tree,
}

async fn read_and_parse(resolved_path: &Path, mode_label: &str) -> anyhow::Result<ParsedSource> {
    let source = match tokio::fs::read_to_string(resolved_path).await {
        Ok(s) => s,
        Err(e) => anyhow::bail!("Could not read file for {mode_label}: {e}"),
    };

    let ext = resolved_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_owned();
    let Some(ls) = language_support(&ext) else {
        anyhow::bail!(
            "Unsupported file extension '.{ext}' for {mode_label}. \
             Supported extensions: .{}",
            supported_extensions().collect::<Vec<_>>().join(", .")
        );
    };
    let language = ls.language;
    let symbol_query = ls.symbol_query;

    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| anyhow::anyhow!("Failed to set tree-sitter language: {e}"))?;

    let Some(tree) = parser.parse(&source, None) else {
        anyhow::bail!("Could not parse file for {mode_label}");
    };

    Ok(ParsedSource {
        source,
        language,
        symbol_query,
        tree,
    })
}

struct LanguageSupport {
    language: Language,
    symbol_query: &'static str,
}

/// Single source of truth mapping extensions to tree-sitter language and symbol query.
#[expect(clippy::too_many_lines)]
fn language_support(ext: &str) -> Option<LanguageSupport> {
    const TS_SYMBOL_QUERY: &str = r"(
            [
                (function_declaration name: (identifier) @name)
                (class_declaration name: (type_identifier) @name)
                (method_definition name: (property_identifier) @name)
                (arrow_function name: (identifier) @name)
                (variable_declarator name: (identifier) @name)
                (interface_declaration name: (type_identifier) @name)
                (enum_declaration name: (identifier) @name)
                (type_alias_declaration name: (type_identifier) @name)
                (export_statement (function_declaration name: (identifier) @name))
                (export_statement (class_declaration name: (type_identifier) @name))
                (export_statement (interface_declaration name: (type_identifier) @name))
                (export_statement (enum_declaration name: (identifier) @name))
                (export_statement (type_alias_declaration name: (type_identifier) @name))
            ]
        )";

    let language = crate::util::tree_sitter::tree_sitter_language_for_extension(ext)?;

    let symbol_query = match ext {
        "rs" => {
            r"(
            [
                (function_item name: (identifier) @name)
                (struct_item name: (type_identifier) @name)
                (enum_item name: (type_identifier) @name)
                (trait_item name: (type_identifier) @name)
                (impl_item type: (_) @name)
                (const_item name: (identifier) @name)
                (static_item name: (identifier) @name)
                (type_item name: (type_identifier) @name)
                (macro_definition name: (identifier) @name)
                (mod_item name: (identifier) @name)
            ]
        )"
        }
        "js" | "jsx" | "mjs" | "cjs" => {
            r"(
            [
                (function_declaration name: (identifier) @name)
                (class_declaration name: (type_identifier) @name)
                (method_definition name: (property_identifier) @name)
                (arrow_function name: (identifier) @name)
                (variable_declarator name: (identifier) @name)
                (export_statement (function_declaration name: (identifier) @name))
                (export_statement (class_declaration name: (type_identifier) @name))
            ]
        )"
        }
        "ts" | "tsx" => TS_SYMBOL_QUERY,
        "py" | "pyi" | "pyx" => {
            r"(
            [
                (function_definition name: (identifier) @name)
                (class_definition name: (identifier) @name)
            ]
        )"
        }
        "sh" | "bash" | "zsh" => {
            r"(
            [
                (function_definition name: (word) @name)
            ]
        )"
        }
        "go" => {
            r"(
            [
                (function_declaration name: (identifier) @name)
                (method_declaration name: (field_identifier) @name)
                (type_declaration (type_spec name: (type_identifier) @name))
                (const_declaration (const_spec name: (identifier) @name))
                (var_declaration (var_spec name: (identifier) @name))
            ]
        )"
        }
        "rb" => {
            r"(
            [
                (method name: (identifier) @name)
                (singleton_method name: (identifier) @name)
                (class name: (constant) @name)
                (module name: (constant) @name)
            ]
        )"
        }
        "c" | "h" => {
            r"(
            [
                (function_definition declarator: (function_declarator declarator: (identifier) @name))
                (struct_specifier name: (type_identifier) @name)
                (enum_specifier name: (type_identifier) @name)
                (union_specifier name: (type_identifier) @name)
                (type_definition declarator: (type_identifier) @name)
            ]
        )"
        }
        "sql" => {
            r"(
            [
                (create_table (object_reference name: (identifier) @name))
                (create_view (object_reference name: (identifier) @name))
                (create_index (object_reference name: (identifier) @name))
                (create_trigger (object_reference name: (identifier) @name))
            ]
        )"
        }
        _ => "",
    };

    Some(LanguageSupport {
        language,
        symbol_query,
    })
}

/// Compile the tree-sitter symbol query for the source file's language.
fn build_symbol_query(ps: &ParsedSource) -> anyhow::Result<Query> {
    Query::new(&ps.language, ps.symbol_query)
        .map_err(|e| anyhow::anyhow!("Failed to build symbol query: {e}"))
}

/// A single symbol extracted from a tree-sitter query match.
#[derive(Debug)]
struct SymbolMatch {
    name: String,
    start_line: usize,
    end_line: usize,
    kind: String,
}

/// Bundles a parsed source file with its compiled symbol query,
/// avoiding redundant `build_symbol_query` calls.
#[derive(Debug)]
struct SymbolQueryContext {
    ps: ParsedSource,
    query: Query,
}

/// Parse and build a symbol query for the given file path.
///
/// Returns an error if the file cannot be read, has an unsupported extension,
/// or the symbol query fails to compile.
async fn prepare_symbol_query(
    resolved_path: &Path,
    mode: &str,
) -> anyhow::Result<SymbolQueryContext> {
    let ps = read_and_parse(resolved_path, mode).await?;
    let query = build_symbol_query(&ps)?;
    Ok(SymbolQueryContext { ps, query })
}

/// Collect all symbol matches from a parsed source using the given query.
///
/// Returns unsorted results — callers are responsible for sorting and dedup
/// as needed. This function is infallible once a valid [`ParsedSource`] and
/// [`Query`] have been obtained.
fn collect_symbols(ps: &ParsedSource, query: &Query) -> Vec<SymbolMatch> {
    let root_node = ps.tree.root_node();
    let mut cursor = QueryCursor::new();
    let mut matches_iter = cursor.matches(query, root_node, ps.source.as_bytes());
    let mut symbols = Vec::new();
    matches_iter.advance();
    while let Some(m) = matches_iter.get() {
        for capture in m.captures {
            let node = capture.node;
            let name = node
                .utf8_text(ps.source.as_bytes())
                .unwrap_or("?")
                .to_string();
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;
            let kind = node.parent().map_or("?", |p| p.kind()).to_string();
            symbols.push(SymbolMatch {
                name,
                start_line,
                end_line,
                kind,
            });
        }
        matches_iter.advance();
    }
    symbols
}

/// Map tree-sitter node kind to a short human-readable label.
fn symbol_kind_label(kind: &str) -> &'static str {
    match kind {
        "function_item" | "function_declaration" | "function_definition" => "fn",
        "struct_item" | "struct_declaration" | "struct_specifier" => "struct",
        "enum_item" | "enum_declaration" | "enum_specifier" => "enum",
        "trait_item" | "trait_declaration" => "trait",
        "impl_item" | "impl_declaration" => "impl",
        "type_item"
        | "type_declaration"
        | "type_alias_declaration"
        | "type_definition"
        | "type_spec" => "type",
        "const_item" | "const_declaration" | "static_item" | "static_declaration"
        | "const_spec" => "const",
        "macro_definition" | "macro_declaration" => "macro",
        "mod_item" | "mod_declaration" => "mod",
        "class_declaration" | "class_definition" | "class" => "class",
        "method_definition" | "method_declaration" | "method" | "singleton_method" => "method",
        "arrow_function" | "variable_declarator" | "var_spec" => "let",
        "identifier" | "type_identifier" | "field_identifier" | "constant" | "word" => "name",
        "interface_declaration" => "interface",
        "union_specifier" => "union",
        "module" => "module",
        "create_table" => "table",
        "create_view" => "view",
        "create_index" => "index",
        "create_trigger" => "trigger",
        _ => "decl",
    }
}

/// Parse the expected line count from a header like "[42 lines total]"
/// or "[Lines 10-20 of 100]". Returns 0 if unparseable.
fn parse_header_line_count(header: &str) -> usize {
    // "[N lines total]"
    if let Some(rest) = header.strip_prefix('[') {
        if let Some(n_str) = rest.strip_suffix(" lines total]") {
            return n_str.parse().unwrap_or(0);
        }
        // "[Lines X-Y of Z]"
        if let Some(inner) = rest.strip_suffix(']')
            && let Some(range) = inner.strip_prefix("Lines ")
            && let Some((start, _end)) = range.split_once(" of ")
            && let Some((lo, hi)) = start.split_once('-')
        {
            let lo: usize = lo.parse().unwrap_or(0);
            let hi: usize = hi.parse().unwrap_or(0);
            return hi.saturating_sub(lo) + 1;
        }
    }
    0
}

/// List the directory at `dir` in-process, on every platform: the format is
/// [`format_listing`]'s, which the shell's own `ls` profile renders too, and no
/// Unix listing program is involved (Windows has none on the shells' path).
///
/// Classification comes from non-following metadata, so a link is a file entry
/// carrying the link's own size and never a directory entry:
///
/// * Unix — `DirEntry::file_type` reads `d_type` (falling back to `lstat`) and
///   `DirEntry::metadata` is an `lstat`, so a symlink is `S_IFLNK`, i.e. not a
///   directory, and its size is the length of its target path.
/// * Windows — both read the values the directory scan already returned
///   (`FindNextFile`), so neither call can fail: a link's size is the reparse
///   point's own recorded size (0 B for a symlink), and `FileType::is_dir` is
///   false for a symlink there too. std counts a junction as a symlink as well
///   (both reparse tags carry the name-surrogate bit), so it is a file entry
///   too; a non-name-surrogate reparse point (a cloud placeholder, say) is an
///   ordinary entry.
///
/// A failed call is a Unix-only race that leaves an entry uninspectable: it is
/// still listed, as a file with an unknown size, and only a directory the
/// process cannot read at all is a failure.
async fn list_directory(dir: &Path) -> anyhow::Result<String> {
    let mut reader = tokio::fs::read_dir(dir)
        .await
        .map_err(|e| path_io_error(dir, &e, "Failed to list directory"))?;
    let mut entries: Vec<ListingEntry> = Vec::new();
    while let Some(entry) = reader
        .next_entry()
        .await
        .map_err(|e| path_io_error(dir, &e, "Failed to list directory"))?
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry
            .file_type()
            .await
            .is_ok_and(|file_type| file_type.is_dir());
        // A directory's size is never printed, so only a file is measured. A
        // failed measurement keeps the entry, with its size unknown.
        let size = if is_dir {
            None
        } else {
            entry.metadata().await.ok().map(|meta| meta.len())
        };
        entries.push(ListingEntry {
            name,
            is_dir,
            size: size.map(human_readable_size),
        });
    }
    // Byte order — what `ls` printed under a `C` collation — never the
    // filesystem's own enumeration order, so the listing is deterministic.
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(try_spill_to_file(
        format_listing(&entries),
        TOOL_OUTPUT_BUDGET_BYTES,
    ))
}

/// The error to report for an I/O failure on `path`: the wording the read tool
/// uses for a missing path and for a denied one, and `context` with the path
/// and the raw cause otherwise.
fn path_io_error(path: &Path, err: &std::io::Error, context: &str) -> anyhow::Error {
    match err.kind() {
        std::io::ErrorKind::NotFound => anyhow::anyhow!("File not found: {}", path.display()),
        std::io::ErrorKind::PermissionDenied => {
            anyhow::anyhow!("Permission denied: {}", path.display())
        }
        _ => anyhow::anyhow!("{context} {}: {err}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the `#[cfg(unix)]` FIFO test drives the write side of a refused read.
    #[cfg(unix)]
    use crate::tools::EditTool;
    use crate::workspace::test_ws;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// The general read tool for the common test scenarios.
    fn tool() -> ReadTool {
        ReadTool::general()
    }

    /// Create a temporary workspace directory for read tests.
    /// Writes initial `files` (relative_path, content) if any.
    /// Returns `(TempDir, PathBuf)` — hold the `TempDir` to keep the dir alive.
    /// The directory is auto-cleaned on drop (panic-safe).
    fn temp_workspace(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        for (rel_path, content) in files {
            let full_path = dir.path().join(rel_path);
            std::fs::create_dir_all(full_path.parent().unwrap()).unwrap();
            std::fs::write(full_path, content).unwrap();
        }
        let path = dir.path().to_path_buf();
        (dir, path)
    }

    /// Every extension in the canonical tree-sitter mapping (`supported_extensions`)
    /// must have a `language_support` entry, and must be listed only once.
    ///
    /// This is a regression check: `language_support` calls
    /// `tree_sitter_language_for_extension` first (which returns `None` for
    /// unsupported extensions), then matches on the extension for a symbol
    /// query with a `_ => ""` catch-all. As long as those two conditions hold,
    /// every recognized extension will have a `Some` result — but if someone
    /// accidentally restructures `language_support` to return `None` for a
    /// previously supported extension, this test catches the regression.
    #[test]
    fn all_supported_extensions_have_language() {
        let mut seen = std::collections::HashSet::new();
        for ext in supported_extensions() {
            assert!(
                seen.insert(ext),
                "extension '{ext}' is listed twice in the tree-sitter grammar mapping"
            );
            assert!(
                language_support(ext).is_some(),
                "expected language support for .{ext}"
            );
        }
    }

    /// `format_output`'s three shapes: short output passes through unchanged, a
    /// structured long output keeps its header and cuts at the last complete line,
    /// and unstructured long output falls back to the head+tail truncation.
    #[test]
    fn format_output_cases() {
        // Short output: byte-identical passthrough.
        let short = "[3 lines total]\n1: a\n2: b\n3: c".to_string();
        assert_eq!(
            tool().format_output(&short),
            short,
            "a short output passes through"
        );

        // Structured long output: the header survives and the body is cut at a
        // complete line, with the omitted count in the marker.
        let header = "[100 lines total]";
        let complete_line = "x".repeat(100);
        let long_body = vec![complete_line.as_str(); 100].join("\n");
        let marker = "\n... (51 lines omitted)";
        let result = tool().format_output(&format!("{header}\n{long_body}"));
        assert!(
            result.starts_with(header),
            "the header is preserved: {result}"
        );
        assert!(
            result.ends_with(marker),
            "the omitted count is {marker:?}: {result}"
        );
        let body = result
            .strip_prefix(header)
            .and_then(|rest| rest.strip_suffix(marker))
            .and_then(|rest| rest.strip_prefix('\n'))
            .expect("the header and marker frame the body");
        assert!(
            body.split('\n').all(|line| line == complete_line.as_str()),
            "every kept line is complete: {result}"
        );
        assert!(
            !result.contains("more bytes"),
            "a structured output uses the line marker: {result}"
        );

        // Unstructured long output: the byte-count fallback marker.
        let result = tool().format_output(&"a".repeat(6000));
        assert!(
            result.contains("bytes omitted at tool output truncation"),
            "the fallback marker is present: {result}"
        );
    }

    #[tokio::test]
    async fn file_read_basic_scenarios() {
        let (_dir, ws_path) = temp_workspace(&[("test.txt", "hello world")]);

        // existing file
        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "test.txt"}))
            .await;
        assert!(result.is_ok(), "read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("1: hello world"));
        assert!(result.contains("[1 lines total]"));
        // nonexistent file
        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "nope.txt"}))
            .await;
        assert!(
            result.is_err(),
            "read should fail for nonexistent file: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("File not found"));
        // empty file
        tokio::fs::write(ws_path.join("empty.txt"), "")
            .await
            .unwrap();
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "empty.txt"}),
            )
            .await;
        assert!(result.is_ok(), "empty file read should succeed: {result:?}");
        let result = result.unwrap();
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn file_read_blocks_unsafe_paths() {
        // path traversal
        let (dir1, ws_path1) = temp_workspace(&[]);

        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path1),
                json!({"path": "../../../etc/passwd"}),
            )
            .await;
        assert!(result.is_err(), "traversal should be blocked: {result:?}");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("outside the allowed read envelope"));
        // absolute path
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path1),
                json!({"path": "/etc/passwd"}),
            )
            .await;
        assert!(
            result.is_err(),
            "absolute path should be blocked: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("outside the allowed read envelope"));
        // null byte in path — separate workspace
        drop(dir1);
        let (_dir2, ws_path2) = temp_workspace(&[]);

        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path2),
                json!({"path": "test\0evil.txt"}),
            )
            .await;
        assert!(
            result.is_err(),
            "null byte path should be blocked: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("outside the allowed read envelope"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_read_blocks_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        // Symlink to /etc/passwd — a real file outside workspace and temp_dir
        symlink("/etc/passwd", workspace.join("escape.txt")).unwrap();

        let result = tool()
            .execute(
                &Workspace::from_path(&workspace),
                json!({"path": "escape.txt"}),
            )
            .await;

        assert!(
            result.is_err(),
            "symlink escape should be blocked: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("outside the allowed read envelope"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn read_reaches_private_key_places() {
        use std::os::unix::fs::symlink;

        // A file literally named `id_rsa`, inside a folder named `.ssh`: nothing
        // about the name or the place is a refusal any more, for any variant.
        let (_dir, ws_path) = temp_workspace(&[(".ssh/id_rsa", "PRIVATE KEY BODY")]);
        let ws = Workspace::from_path(&ws_path);
        for tool in [
            &ReadTool::general() as &dyn Tool,
            &ReadTool::new(PathAccess::Workspace) as &dyn Tool,
        ] {
            let result = tool.execute(&ws, json!({"path": ".ssh/id_rsa"})).await;
            let text = result.expect("a private key's name is not a refusal");
            assert!(
                text.contains("PRIVATE KEY BODY"),
                "the key body must come back whole: {text}"
            );
        }

        // The same through a symlinked name and a symlinked directory.
        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let keys = root.path().join(".ssh");
        std::fs::create_dir_all(&keys).unwrap();
        std::fs::write(keys.join("id_rsa"), "PRIVATE KEY BODY").unwrap();
        symlink(keys.join("id_rsa"), workspace.join("data.txt")).unwrap();
        symlink(&keys, workspace.join("dirlink")).unwrap();

        let ws = Workspace::from_path(&workspace);
        for path in ["data.txt", "dirlink/id_rsa"] {
            let result = tool().execute(&ws, json!({"path": path})).await;
            let text = result.unwrap_or_else(|e| panic!("{path} should read: {e}"));
            assert!(text.contains("PRIVATE KEY BODY"), "{path}: {text}");
        }
    }

    /// The admin's read reaches the machine, and a path that is not there is
    /// answered as missing: no same-named file from the workspace is substituted
    /// for it (which is what the other access levels still do for a typo).
    #[tokio::test]
    async fn unrestricted_read_neither_frames_nor_substitutes() {
        // The fuzzy recovery needs the global search-engine registry.
        crate::util::test::init_test_stores().await;
        let (dir, ws_path) = temp_workspace(&[("notes.md", "workspace notes")]);
        let ws = Workspace::from_path(&ws_path);

        // A typo inside the workspace is still recovered for the general read.
        let recovered = tool()
            .execute(&ws, json!({"path": "notez.md"}))
            .await
            .expect("a typo'd workspace path is recovered");
        assert!(recovered.contains("[Recovered path:"), "{recovered}");

        // An absent path outside every allowed root — nothing is created for it,
        // and it is named after this process so it cannot be someone's file.
        let absent = PathBuf::from(format!(
            "/mahbot-read-outside-{}/notes.md",
            std::process::id()
        ));

        // The admin's read of a missing path is a missing path — never the
        // workspace's own file of the same name.
        let err = ReadTool::new(PathAccess::Unrestricted)
            .execute(&ws, json!({"path": absent.to_string_lossy()}))
            .await
            .expect_err("a missing path must be reported as missing");
        assert!(err.to_string().contains("File not found"), "{err}");
        assert!(!err.to_string().contains("Recovered"), "{err}");

        // And a file outside the workspace is read as given.
        let outside = TempDir::new().unwrap();
        let outside_notes = outside.path().join("notes.md");
        std::fs::write(&outside_notes, "outside notes").unwrap();
        let text = ReadTool::new(PathAccess::Unrestricted)
            .execute(&ws, json!({"path": outside_notes.to_string_lossy()}))
            .await
            .expect("an outside file is readable for the admin");
        assert!(text.contains("outside notes"), "{text}");

        // The general read still stops at its own envelope on the same path.
        let err = tool()
            .execute(&ws, json!({"path": absent.to_string_lossy()}))
            .await
            .expect_err("the general read stays inside its envelope");
        assert!(
            err.to_string()
                .contains("outside the allowed read envelope"),
            "{err}"
        );

        drop((dir, outside));
    }

    /// A path whose own name carries glob metacharacters is read as the file it
    /// names when that file exists — the shell's own reading, and what the edit
    /// tool does — while a pattern that names nothing still lists the workspace.
    #[tokio::test]
    async fn a_metacharacter_name_is_read_as_the_file_it_names() {
        let (dir, ws_path) = temp_workspace(&[("report[1].txt", "the file's own text")]);
        let ws = Workspace::from_path(&ws_path);

        let text = tool()
            .execute(&ws, json!({"path": "report[1].txt"}))
            .await
            .expect("a metacharacter name that exists is read as a file");
        assert!(text.contains("the file's own text"), "{text}");

        // The admin reads the same kind of name outside the workspace.
        let outside = TempDir::new().unwrap();
        let outside_file = outside.path().join("report[2].txt");
        std::fs::write(&outside_file, "outside the workspace").unwrap();
        let text = ReadTool::new(PathAccess::Unrestricted)
            .execute(&ws, json!({"path": outside_file.to_string_lossy()}))
            .await
            .expect("the admin reads a metacharacter name outside the workspace");
        assert!(text.contains("outside the workspace"), "{text}");

        // A pattern that names no file is still a wildcard.
        let err = tool()
            .execute(&ws, json!({"path": "*.missing"}))
            .await
            .expect_err("a pattern that names nothing falls through to the listing");
        assert!(err.to_string().to_lowercase().contains("wildcard"), "{err}");

        drop((dir, outside));
    }

    #[tokio::test]
    async fn read_sensitive_config_is_not_denied() {
        // A credential-bearing config file is readable (Ok) — hardening is the
        // scrub tier (output scrubbing in `read_resolved`), not a hard deny.
        let (_dir, ws_path) = temp_workspace(&[("settings.xml", "<settings/>")]);
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "settings.xml"}),
            )
            .await;
        assert!(result.is_ok(), "settings.xml should read: {result:?}");
    }

    #[tokio::test]
    async fn read_scrubs_credential_bearing_file() {
        // End-to-end scrub tier: the resolved path `.env` is credential-bearing,
        // so the output is scrubbed before the LLM sees it.
        let (_dir, ws_path) = temp_workspace(&[(".env", "API_KEY=sk-1234567890")]);
        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": ".env"}))
            .await;
        assert!(result.is_ok(), "read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(
            result.contains("[REDACTED]"),
            "credential should be redacted: {result}"
        );
        assert!(
            !result.contains("sk-1234567890"),
            "raw key must not leak: {result}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn read_scrubs_symlinked_credential_file() {
        // A benign basename whose symlink resolves to a credential file is
        // scrubbed based on the resolved path (the case the old
        // `is_sensitive_read_path` test covered).
        use std::os::unix::fs::symlink;

        let (_dir, ws_path) = temp_workspace(&[(".env", "API_KEY=sk-1234567890")]);
        symlink(".env", ws_path.join("innocent.txt")).unwrap();
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "innocent.txt"}),
            )
            .await;
        assert!(result.is_ok(), "symlinked read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(
            result.contains("[REDACTED]"),
            "symlink target should be redacted: {result}"
        );
        assert!(
            !result.contains("sk-1234567890"),
            "raw key must not leak via symlink: {result}"
        );
    }

    #[tokio::test]
    async fn read_does_not_scrub_plain_file() {
        // A non-sensitive filename with the same content is returned verbatim.
        let (_dir, ws_path) = temp_workspace(&[("notes.txt", "API_KEY=sk-1234567890")]);
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "notes.txt"}),
            )
            .await;
        assert!(result.is_ok(), "read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(
            !result.contains("[REDACTED]"),
            "plain file should not be redacted: {result}"
        );
        assert!(
            result.contains("sk-1234567890"),
            "plain file content should be verbatim: {result}"
        );
    }

    #[tokio::test]
    async fn file_read_offset_handling() {
        let (_dir, ws_path) = temp_workspace(&[("lines.txt", "aaa\nbbb\nccc\nddd\neee")]);

        // Read lines 2-3
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "lines.txt", "offset": 2, "limit": 2}),
            )
            .await;
        assert!(result.is_ok(), "offset read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("2: bbb") && result.contains("3: ccc"));
        assert!(!result.contains("1: aaa") && !result.contains("4: ddd"));
        // Offset to end
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "lines.txt", "offset": 4}),
            )
            .await;
        assert!(result.is_ok(), "offset to end should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("4: ddd") && result.contains("5: eee"));
        // Limit only (first 2 lines)
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "lines.txt", "limit": 2}),
            )
            .await;
        assert!(result.is_ok(), "limit read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(!result.contains("3: ccc"));
        // Offset beyond end
        tokio::fs::write(ws_path.join("short.txt"), "one\ntwo")
            .await
            .unwrap();
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "short.txt", "offset": 100}),
            )
            .await;
        assert!(
            result.is_ok(),
            "offset beyond end should succeed: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("[No lines in range, file has 2 lines]"));
    }

    #[tokio::test]
    async fn file_read_rejects_oversized_file() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();

        // Create a file just over 10 MB
        let big = vec![b'x'; 10 * 1024 * 1024 + 1];
        tokio::fs::write(ws_path.join("huge.bin"), &big)
            .await
            .unwrap();

        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "huge.bin"}))
            .await;
        assert!(
            result.is_err(),
            "oversized file should be rejected: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("File too large"));
    }

    /// Non-UTF-8 binary files should be read with lossy conversion.
    #[tokio::test]
    async fn file_read_lossy_reads_binary_file() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();

        // Write bytes that are not valid UTF-8
        let binary_data: Vec<u8> = vec![0x00, 0x80, 0xFF, 0xFE, b'h', b'i', 0x80];
        tokio::fs::write(ws_path.join("data.bin"), &binary_data)
            .await
            .unwrap();

        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "data.bin"}))
            .await;

        assert!(
            result.is_ok(),
            "lossy read must succeed, error: {:?}",
            result.as_ref().unwrap_err()
        );
        let result = result.unwrap();
        assert!(
            result.contains('\u{FFFD}'),
            "lossy output must contain replacement character, got: {result:?}",
        );
        assert!(
            result.contains("hi"),
            "lossy output must preserve valid ASCII, got: {result:?}",
        );
    }

    /// Encode a tiny solid-red PNG (test helper).
    fn tiny_png_bytes(width: u32, height: u32) -> Vec<u8> {
        use std::io::Cursor;
        let img = ::image::RgbaImage::from_pixel(width, height, ::image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ::image::ImageFormat::Png)
            .expect("test PNG must encode");
        buf
    }

    /// A raster file the read is asked for and what the two halves of the image
    /// path must answer: the annotation `execute` prints and the payload (the
    /// only decoder) that the agent loop injects.
    struct ImageCase {
        name: &'static str,
        file: &'static str,
        bytes: Vec<u8>,
        /// Substrings the `execute` annotation must carry.
        present: &'static [&'static str],
        /// Substrings the `execute` annotation must not carry.
        absent: &'static [&'static str],
        /// The `(width, height, format)` of the payload, when one is produced.
        payload: Option<(u32, u32, &'static str)>,
        /// Whether the payload names the resolved path of the file.
        payload_path: bool,
    }

    #[tokio::test]
    async fn image_payload_cases() {
        let png = tiny_png_bytes(4, 4);
        let mut unsupported: Vec<u8> = b"GIF89a".to_vec();
        unsupported.extend_from_slice(&[0u8; 20]);
        let cases = vec![
            ImageCase {
                name: "unsupported_raster_is_reported",
                file: "bad.gif",
                bytes: unsupported,
                present: &["unsupported image format", "GIF"],
                absent: &[],
                payload: None,
                payload_path: false,
            },
            ImageCase {
                name: "corrupt_native_image_claims_no_attachment",
                // Truncated to the PNG signature + IHDR: valid magic and header
                // dimensions, but no IDAT/IEND. The cheap sniff still reports a
                // PNG, while the decode — deferred to `image_payload` — fails, so
                // the annotation stays honest about what the read demonstrated.
                file: "corrupt.png",
                bytes: png[..33].to_vec(),
                present: &["Read image file", "PNG"],
                absent: &["attached to the conversation"],
                payload: None,
                payload_path: false,
            },
            ImageCase {
                name: "native_image_is_annotated_then_attached",
                file: "tiny.png",
                bytes: png,
                present: &["Read image file", "PNG"],
                absent: &["attached to the conversation"],
                payload: Some((4, 4, "PNG")),
                payload_path: true,
            },
            ImageCase {
                name: "a_text_file_has_no_payload",
                file: "hello.txt",
                bytes: b"hello world".to_vec(),
                present: &[],
                absent: &[],
                payload: None,
                payload_path: false,
            },
        ];
        for case in cases {
            let dir = TempDir::new().unwrap();
            let ws_path = dir.path().to_path_buf();
            tokio::fs::write(ws_path.join(case.file), &case.bytes)
                .await
                .unwrap();
            let ws = Workspace::from_path(&ws_path);

            let text = tool()
                .execute(&ws, json!({"path": case.file}))
                .await
                .unwrap_or_else(|e| panic!("{}: the read must succeed: {e}", case.name));
            for expected in case.present {
                assert!(
                    text.contains(expected),
                    "{}: the annotation does not carry {expected:?}: {text}",
                    case.name
                );
            }
            for unexpected in case.absent {
                assert!(
                    !text.contains(unexpected),
                    "{}: the annotation claims {unexpected:?}: {text}",
                    case.name
                );
            }

            let payload = tool().image_payload(&ws, &json!({"path": case.file})).await;
            let Some((width, height, format)) = case.payload else {
                assert!(
                    payload.is_none(),
                    "{}: no payload may be produced",
                    case.name
                );
                continue;
            };
            let payload =
                payload.unwrap_or_else(|| panic!("{}: the payload must be produced", case.name));
            assert!(
                payload.data_uri.starts_with("data:image/jpeg;base64,"),
                "{}: unexpected data-uri: {}",
                case.name,
                payload.data_uri
            );
            assert_eq!(payload.width, width, "{}: payload width", case.name);
            assert_eq!(payload.height, height, "{}: payload height", case.name);
            assert_eq!(payload.format, format, "{}: payload format", case.name);
            if case.payload_path {
                // `payload.path` is the resolved path: canonicalized (which on
                // macOS resolves the /tmp → /private/tmp symlink) in the spelling
                // the tool hands on.
                let expected_path =
                    crate::util::test::canonical_without_verbatim_prefix(&ws_path.join(case.file));
                assert_eq!(
                    payload.path,
                    expected_path.display().to_string(),
                    "{}: unexpected payload path",
                    case.name
                );
            }
        }
    }

    /// Regression: a typo'd image path that `execute` recovers to a unique
    /// match is actually attached by `image_payload` (not just annotated), and
    /// the payload carries the `[Recovered path: ...]` note. Pins the fact that
    /// both go through the one resolution
    /// ([`resolve_content_read`]) so the recovery logic cannot drift back into
    /// the 'annotated but not attached' gap.
    #[tokio::test]
    async fn recovered_image_path_is_attached() {
        // The fuzzy search needs the global search-engine registry.
        crate::util::test::init_test_stores().await;

        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let png = tiny_png_bytes(4, 4);
        tokio::fs::write(ws_path.join("tiny_image.png"), &png)
            .await
            .unwrap();
        let ws = Workspace::from_path(&ws_path);

        // execute with a typo'd path that the fuzzy matcher recovers to the file.
        let result = tool()
            .execute(&ws, json!({"path": "tiny_imag.png"}))
            .await
            .expect("recovered image read must succeed");
        assert!(result.contains("[Recovered path:"), "got: {result}");
        assert!(result.contains("Read image file"), "got: {result}");

        // image_payload must attach the recovered image (not just annotate it),
        // and surface the recovery note for the tool-result annotation.
        let payload = tool()
            .image_payload(&ws, &json!({"path": "tiny_imag.png"}))
            .await
            .expect("recovered image payload must be produced");
        assert_eq!(payload.width, 4, "unexpected payload width");
        assert_eq!(payload.height, 4, "unexpected payload height");
        assert_eq!(payload.format, "PNG", "unexpected payload format");
        let note = payload
            .recovery_note
            .expect("recovered read must carry a recovery note");
        assert!(note.contains("tiny_imag.png"), "note: {note}");
    }

    /// One symbol-mode request — the answer's own labels when it succeeds, the
    /// refusal's words when it does not.
    struct SymbolModeCase {
        name: &'static str,
        file: &'static str,
        code: &'static str,
        mode: &'static str,
        symbol: Option<&'static str>,
        /// Substrings the answer must carry.
        present: &'static [&'static str],
        /// Substrings the refusal must carry — empty when the request succeeds.
        refusal: &'static [&'static str],
    }

    const SYMBOL_MODE_CASES: &[SymbolModeCase] = &[
        SymbolModeCase {
            name: "listing_every_kind",
            file: "lib.rs",
            code: r"
fn hello() {}
struct Point { x: i32, y: i32 }
enum Color { Red, Blue }
trait Draw { fn draw(&self); }
impl Point { fn new() -> Self { Point { x: 0, y: 0 } } }
const MAX: usize = 100;
type MyInt = i32;
macro_rules! my_macro { () => {} }
mod utils;
",
            mode: "symbols",
            symbol: None,
            present: &[
                "[Symbols in lib.rs]",
                "fn `hello`",
                "struct `Point`",
                "enum `Color`",
                "trait `Draw`",
                "impl `Point`",
                "const `MAX`",
                "type `MyInt`",
                "mod `utils`",
            ],
            refusal: &[],
        },
        SymbolModeCase {
            name: "listing_an_unsupported_extension",
            file: "data.yaml",
            code: "{}",
            mode: "symbols",
            symbol: None,
            present: &[],
            refusal: &["Unsupported"],
        },
        SymbolModeCase {
            name: "zoom_extracts_the_named_function",
            file: "main.rs",
            code: "fn greet(name: &str) -> String {\n    format!(\"Hi, {name}!\")\n}\n\nfn main() {}",
            mode: "zoom",
            symbol: Some("greet"),
            present: &["fn `greet`", "format!(\"Hi, {name}!\")"],
            refusal: &[],
        },
        SymbolModeCase {
            name: "zoom_of_an_absent_symbol_suggests_what_is_there",
            file: "lib.rs",
            code: "fn existing() {}",
            mode: "zoom",
            symbol: Some("nope"),
            present: &[],
            refusal: &["'nope'", "Did you mean", "existing"],
        },
        SymbolModeCase {
            name: "zoom_without_a_symbol",
            file: "lib.rs",
            code: "fn f() {}",
            mode: "zoom",
            symbol: None,
            present: &[],
            refusal: &["Missing 'symbol' parameter"],
        },
    ];

    #[tokio::test]
    async fn symbol_mode_cases() {
        for case in SYMBOL_MODE_CASES {
            let (_dir, ws_path) = temp_workspace(&[(case.file, case.code)]);
            let mut args = json!({"path": case.file, "mode": case.mode});
            if let Some(symbol) = case.symbol {
                args["symbol"] = json!(symbol);
            }
            let result = tool().execute(&test_ws(&ws_path), args).await;
            if case.refusal.is_empty() {
                let text = result
                    .unwrap_or_else(|e| panic!("{}: the request must succeed: {e}", case.name));
                for expected in case.present {
                    assert!(
                        text.contains(expected),
                        "{}: the answer does not name {expected:?}: {text}",
                        case.name
                    );
                }
            } else {
                let err = format!(
                    "{}",
                    result
                        .err()
                        .unwrap_or_else(|| panic!("{}: the request must be refused", case.name))
                );
                for expected in case.refusal {
                    assert!(
                        err.contains(expected),
                        "{}: the refusal does not carry {expected:?}: {err}",
                        case.name
                    );
                }
            }
        }
    }

    /// Directory listing returns file names instead of erroring.
    #[tokio::test]
    async fn directory_listing_returns_contents() {
        let (_dir, ws_path) =
            temp_workspace(&[("a.txt", "alpha"), ("b.rs", "beta"), (".hidden", "h")]);
        tokio::fs::create_dir(ws_path.join("sub")).await.unwrap();

        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "."}))
            .await;
        assert!(result.is_ok(), "dir listing should succeed: {result:?}");
        // Directories first with a mark, then files in byte order with their
        // size — hidden entries included — and the extension summary.
        assert_eq!(
            result.unwrap(),
            "sub/\n.hidden  1B\na.txt  5B\nb.rs  4B\nSummary: 3 files, 1 dirs (1 .hidden, 1 .rs, 1 .txt)\n"
        );
    }

    /// An empty directory is reported as empty, never as a failed listing.
    #[tokio::test]
    async fn directory_listing_empty() {
        let (_dir, ws_path) = temp_workspace(&[]);

        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "."}))
            .await;
        assert!(
            result.is_ok(),
            "empty dir listing should succeed: {result:?}"
        );
        assert_eq!(result.unwrap(), "(empty)\n");
    }

    /// A directory the process cannot read is a failed listing, not an empty one.
    /// (Skipped for a supervisor user, who can read a `0o000` directory.)
    #[cfg(unix)]
    #[tokio::test]
    async fn directory_listing_unreadable_is_a_failure() {
        use std::os::unix::fs::PermissionsExt;

        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        let (_dir, ws_path) = temp_workspace(&[("sub/locked.txt", "x")]);
        let locked = ws_path.join("sub");
        tokio::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .await
            .unwrap();

        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "sub"}))
            .await;

        // Restore the mode before asserting, so the temp dir can still be cleaned.
        tokio::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))
            .await
            .unwrap();
        let err = format!(
            "{}",
            result.expect_err("a denied directory must not read as an empty listing")
        );
        assert!(err.contains("Permission denied"), "unexpected error: {err}");
    }

    /// An entry the platform cannot inspect is still listed — in the files
    /// group, with an unknown size — instead of silently disappearing.
    /// (Skipped for a supervisor user, who can inspect a `0o444` directory.)
    #[cfg(unix)]
    #[tokio::test]
    async fn directory_listing_uninspectable_entry_is_listed() {
        use std::os::unix::fs::PermissionsExt;

        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        let (_dir, ws_path) = temp_workspace(&[("sub/inside.txt", "x")]);
        let unsearchable = ws_path.join("sub");
        // Readable (the names can still be listed) but not searchable, so
        // nothing inside it can be measured.
        tokio::fs::set_permissions(&unsearchable, std::fs::Permissions::from_mode(0o444))
            .await
            .unwrap();

        let result = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "sub"}))
            .await;

        // Restore the mode before asserting, so the temp dir can still be cleaned.
        tokio::fs::set_permissions(&unsearchable, std::fs::Permissions::from_mode(0o755))
            .await
            .unwrap();
        assert_eq!(
            result.expect("unmeasured entries must not fail the listing"),
            "inside.txt  ?\nSummary: 1 files, 0 dirs (1 .txt)\n"
        );
    }

    /// A listing too large for the tool output budget is spilled to a file with
    /// the usual hint, so the full listing stays recoverable.
    #[tokio::test]
    async fn directory_listing_large_spills_to_file() {
        let names: Vec<String> = (0..600).map(|i| format!("file_{i:03}.txt")).collect();
        let refs: Vec<(&str, &str)> = names.iter().map(|name| (name.as_str(), "x")).collect();
        let (_dir, ws_path) = temp_workspace(&refs);

        let output = tool()
            .execute(&Workspace::from_path(&ws_path), json!({"path": "."}))
            .await
            .unwrap();
        assert!(
            output.contains("[Output saved to"),
            "expected a spill: {output}"
        );
        assert!(
            output.contains("[view with: read "),
            "expected a read hint: {output}"
        );
    }

    /// Directory listing resolves symlinks to directories.
    #[cfg(unix)]
    #[tokio::test]
    async fn directory_listing_symlink() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let real_dir = ws_path.join("real");
        tokio::fs::create_dir_all(&real_dir).await.unwrap();
        tokio::fs::write(real_dir.join("nested.txt"), "data")
            .await
            .unwrap();
        let link = ws_path.join("link_to_real");
        symlink(&real_dir, &link).unwrap();

        // Reading the symlink directly (it resolves to the directory)
        let result = tool()
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "link_to_real"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "symlinked dir listing should succeed: {result:?}"
        );
        let output = result.unwrap();
        assert!(
            output.contains("nested.txt"),
            "should list nested file: {output}"
        );
    }

    #[test]
    fn is_sensitive_file_path_env_and_certs() {
        // extension + dotfile matches
        assert!(is_sensitive_file_path(".env"));
        assert!(is_sensitive_file_path("proj/.env"));
        assert!(is_sensitive_file_path(".env.local"));
        assert!(is_sensitive_file_path("/abs/path/.env.production"));
        assert!(is_sensitive_file_path("secrets/local.env"));
        assert!(is_sensitive_file_path("tls/cert.pem"));
        assert!(is_sensitive_file_path("C:\\keys\\id_rsa.key"));
        assert!(is_sensitive_file_path("a.pem"));
        assert!(is_sensitive_file_path("a.key"));
        // exact credential config names
        assert!(is_sensitive_file_path("settings.xml"));
        assert!(is_sensitive_file_path("settings-security.xml"));
        assert!(is_sensitive_file_path("gradle.properties"));
        assert!(is_sensitive_file_path("credentials.toml"));
        assert!(is_sensitive_file_path(".netrc"));
        assert!(is_sensitive_file_path(".npmrc"));
        assert!(is_sensitive_file_path(".pypirc"));
        assert!(is_sensitive_file_path(".git-credentials"));
        // path-qualified exact names match on basename
        assert!(is_sensitive_file_path("~/.m2/settings.xml"));
        // init.gradle prefix (matches init.gradle and init.gradle.kts)
        assert!(is_sensitive_file_path("init.gradle"));
        assert!(is_sensitive_file_path("init.gradle.kts"));
        assert!(is_sensitive_file_path("init.gradle.sh"));

        // non-credential files stay untouched
        assert!(!is_sensitive_file_path("src/main.rs"));
        assert!(!is_sensitive_file_path("crates/foo/lib.rs"));
        assert!(!is_sensitive_file_path("README.md"));
        assert!(!is_sensitive_file_path("id_rsa.pub")); // public key, not a private key
        assert!(!is_sensitive_file_path("notes.txt"));
        assert!(!is_sensitive_file_path("pom.xml"));
        assert!(!is_sensitive_file_path("build.gradle")); // project build file, not credentials
        assert!(!is_sensitive_file_path(
            "gradle/wrapper/gradle-wrapper.properties"
        )); // basename-only
    }
    // ── prepare_symbol_query / collect_symbols ──────────────────────────────

    /// A file handed to the symbol query, and the declarations it holds — every
    /// `(name, start line)` the query collects, in the order it finds them.
    struct SymbolQueryCase {
        name: &'static str,
        file: &'static str,
        code: &'static str,
        symbols: &'static [(&'static str, usize)],
        /// The refusal a file the query cannot be built for must give.
        refusal: &'static str,
    }

    const SYMBOL_QUERY_CASES: &[SymbolQueryCase] = &[
        SymbolQueryCase {
            name: "a_supported_file_builds_its_query",
            file: "lib.rs",
            code: "fn hello() {}\nstruct World;\n",
            symbols: &[("hello", 1), ("World", 2)],
            refusal: "",
        },
        SymbolQueryCase {
            name: "an_unsupported_extension_has_no_query",
            file: "data.txt",
            code: "hello world",
            symbols: &[],
            refusal: "Unsupported",
        },
        SymbolQueryCase {
            name: "an_empty_file_holds_no_symbol",
            file: "empty.rs",
            code: "",
            symbols: &[],
            refusal: "",
        },
        SymbolQueryCase {
            name: "every_kind_of_declaration_is_collected",
            file: "main.rs",
            code: r"
fn foo() {}
fn bar() {}
struct Baz;
enum Qux {}
impl Baz {}
",
            // foo, bar, Baz, Qux, Baz (impl)
            symbols: &[("foo", 2), ("bar", 3), ("Baz", 4), ("Qux", 5), ("Baz", 6)],
            refusal: "",
        },
        SymbolQueryCase {
            name: "the_line_numbers_are_the_declarations_own",
            file: "lib.rs",
            code: "fn hello() {}\n\n\nfn world() {}\n",
            symbols: &[("hello", 1), ("world", 4)],
            refusal: "",
        },
    ];

    #[tokio::test]
    async fn symbol_query_cases() {
        for case in SYMBOL_QUERY_CASES {
            let (_dir, ws_path) = temp_workspace(&[(case.file, case.code)]);
            let file_path = ws_path.join(case.file);
            let result = prepare_symbol_query(&file_path, "test").await;
            if !case.refusal.is_empty() {
                let err = format!(
                    "{}",
                    result
                        .err()
                        .unwrap_or_else(|| panic!("{}: the query must be refused", case.name))
                );
                assert!(
                    err.contains(case.refusal),
                    "{}: the refusal does not name {}: {err}",
                    case.name,
                    case.refusal
                );
                continue;
            }
            let ctx =
                result.unwrap_or_else(|e| panic!("{}: the query must be built: {e}", case.name));
            assert!(
                !ctx.ps.symbol_query.is_empty(),
                "{}: expected a non-empty symbol_query",
                case.name
            );
            let symbols = collect_symbols(&ctx.ps, &ctx.query);
            let found: Vec<(&str, usize)> = symbols
                .iter()
                .map(|symbol| (symbol.name.as_str(), symbol.start_line))
                .collect();
            assert_eq!(found, case.symbols, "{}: the collected symbols", case.name);
        }
    }

    /// A channel (FIFO) is neither a regular file nor a directory: the read
    /// refuses it for every role rather than blocking on a writer that may
    /// never appear, and the same refusal meets the read an edit does.
    #[cfg(unix)]
    #[tokio::test]
    async fn non_regular_file_is_refused() {
        let dir = TempDir::new().unwrap();
        let fifo_path = dir.path().join("nofifo.pipe");
        let c_path = std::ffi::CString::new(fifo_path.as_os_str().as_encoded_bytes()).unwrap();
        let ret = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(ret, 0, "mkfifo should succeed");

        let ws = test_ws(dir.path());

        let err = tool()
            .execute(
                &ws,
                json!({"path": fifo_path.to_string_lossy().into_owned()}),
            )
            .await
            .expect_err("a channel must be refused");
        assert!(
            err.to_string()
                .contains("neither a regular file nor a directory"),
            "the refusal must say what the path is: {err}"
        );

        // The read an edit performs meets the same refusal — a relative path, so
        // it is the file's own kind that decides, not the workspace frame.
        let editor = EditTool::confined();
        let edited = editor.execute(
            &ws,
            json!({ "path": "nofifo.pipe", "old_string": "a", "new_string": "b" }),
        );
        let err = edited.await.expect_err("an edit must refuse a channel");
        assert!(
            err.to_string().contains("not a regular file"),
            "the write side must refuse it too: {err}"
        );
    }
}
