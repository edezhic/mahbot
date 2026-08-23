use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::time::Duration;

use super::path::shell_quote;
use crate::tools::{ShellMode, ShellTool, search::SearchTool};
use crate::util::TOOL_OUTPUT_BUDGET_BYTES;
use crate::util::tree_sitter::ALL_TREE_SITTER_EXTENSIONS;
use crate::{Tool, Workspace};
use async_trait::async_trait;
use serde_json::json;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator, Tree};

pub struct ReadTool;

/// Recognized sensitive file extensions whose read output should be scrubbed for credentials.
const SENSITIVE_EXTENSIONS: &[&str] = &["cer", "crt", "env", "key", "p12", "pem", "pfx"];

/// File paths whose read output should be scrubbed for credentials (`.env`, certs, keys).
/// Other extensions (e.g. `.rs`, `.md`) are left intact so the model sees source accurately.
#[must_use]
fn is_sensitive_file_path(path: &str) -> bool {
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
/// decode is deferred to [`ReadTool::image_payload`], the single decoder.
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

/// FIFO-safe file-magic gate: true only when `path` (a regular file) begins
/// with PNG/JPEG/WebP magic. Reuses [`sniff_read_image`] so the native-format
/// decision has a single source — no duplicated format-match arms, no drift
/// risk. Returns `false` for text, unsupported formats, and FIFO/special files
/// without a full read or decode. This is the robust gate `image_payload` uses
/// instead of the annotation wording.
async fn is_native_image_file(path: &Path) -> bool {
    use tokio::io::AsyncReadExt;
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return false;
    };
    // FIFO/special files are never reopened — a stream cannot be sampled
    // without blocking.
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

/// Candidate paths for a literal `path` that `resolve_read_target` could not
/// resolve (a typo/missing path). Queried by filename, as
/// [`recover_missing_path`](ReadTool::recover_missing_path) does. This is the
/// single recovery-search source shared by the read-annotation path and
/// `image_payload`'s resolve path, so the two can never drift apart.
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
struct ResolvedRead {
    path: PathBuf,
    recovery_note: Option<String>,
}

/// Resolve a content-read `path` to a readable file path, mirroring the
/// recovery that `execute` applies via `recover_missing_path`: if the literal
/// path does not exist (and only then), a single high-confidence filename match
/// is used so a recovered raster can still be attached by `image_payload`
/// (which receives the original, possibly typo'd, `path`). Returns `None` when
/// the path is missing and no unique match exists, and when it does not resolve
/// at all.
async fn resolve_for_read(ws: &Workspace, path: &str) -> Option<ResolvedRead> {
    match super::path::resolve_read_target(ws.as_path(), path).await {
        Ok(resolved) => {
            return Some(ResolvedRead {
                path: resolved,
                recovery_note: None,
            });
        }
        Err(e) if !e.to_string().contains("File not found") => return None,
        Err(_) => {}
    }
    let matches = find_recovery_candidates(ws, path).await;
    if matches.len() != 1 {
        return None;
    }
    let recovered = &matches[0];
    let resolved = super::path::resolve_read_target(ws.as_path(), recovered)
        .await
        .ok()?;
    Some(ResolvedRead {
        path: resolved,
        recovery_note: Some(recovery_note(path, recovered)),
    })
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "path": {
                    "type": "string",
                    "description": "Path to the file. Relative paths resolve from workspace; outside paths require policy allowlist."
                },
                "mode": {
                    "type": "string",
                    "enum": ["content", "symbols", "zoom"],
                    "description": "Read mode. 'content' (default): line-numbered file read, or — for a raster image (PNG, JPEG, WebP) — attaches it to the conversation as a native image rather than reading text. 'symbols': list all top-level AST symbols with line ranges. 'zoom': extract a single symbol's source by name.",
                    "default": "content"
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol name for zoom mode. Required when mode is 'zoom'.",
                    "minLength": 1
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based, default: 1)",
                    "default": 1,
                    "minimum": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: all)",
                    "minimum": 1
                }
            }),
            &["path"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let path = super::get_str(&args, "path")?.to_string();

        if super::path::contains_glob(&path, true) {
            return self.recover_wildcard_path(ws, &path).await;
        }

        let resolved_path = match super::path::resolve_read_target(ws.as_path(), &path).await {
            Ok(p) => p,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("File not found") {
                    return self.recover_missing_path(ws, &path, &args, &msg).await;
                }
                return Err(e);
            }
        };

        self.read_resolved(ws, &resolved_path, None, &args).await
    }

    fn should_scrub_output(&self, args: &serde_json::Value) -> bool {
        match super::get_opt_str(args, "path") {
            Some(path) => is_sensitive_file_path(path),
            None => true, // No path? Be safe and scrub.
        }
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

    async fn image_payload(
        &self,
        ws: &Workspace,
        args: &serde_json::Value,
    ) -> Option<crate::tools::ImagePayload> {
        // Robust file-magic gate (NOT the annotation wording): only a PNG/JPEG/
        // WebP raster opens the decode+encode below. This runs AFTER path
        // resolution, so text/unsupported reads still pay resolve + metadata +
        // a 64-byte sniff — but never a full read/decode of the file.
        let path = super::get_str(args, "path").ok()?.to_string();
        // Image behaviour is content-mode only (symbols/zoom run tree-sitter).
        if super::get_opt_str(args, "mode").unwrap_or("content") != "content" {
            return None;
        }
        if super::path::contains_glob(&path, true) {
            return None;
        }
        // Resolve the literal path, or — for a typo'd path that `execute`
        // already recovered to a unique match — the recovered path, so a
        // recovered raster is attached rather than only annotated.
        let res = resolve_for_read(ws, &path).await?;
        if !is_native_image_file(&res.path).await {
            return None;
        }
        // The compressed encode applies EXIF orientation and reports the
        // post-EXIF/post-resize dims + source-format label; its metadata-first
        // `is_file()` guard keeps a FIFO/special file from being reopened.
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
}

impl ReadTool {
    /// Read a resolved file path (content, symbols, or zoom mode).
    async fn read_resolved(
        &self,
        ws: &Workspace,
        resolved_path: &Path,
        recovery_note: Option<&str>,
        args: &serde_json::Value,
    ) -> anyhow::Result<String> {
        match tokio::fs::metadata(resolved_path).await {
            Ok(meta) => {
                if meta.is_dir() {
                    return list_directory(resolved_path, ws).await;
                }
                super::check_file_size(&meta)?;
                // FIFOs are streams, not seekable files — read them with a
                // bounded non-blocking wait so a missing writer cannot hang
                // the tool (see read_fifo). Other modes (symbols/zoom) make no
                // sense on a pipe, so content is always used.
                #[cfg(unix)]
                if std::os::unix::fs::FileTypeExt::is_fifo(&meta.file_type()) {
                    let bytes = read_fifo(resolved_path, fifo_read_timeout()).await?;
                    let contents = String::from_utf8_lossy(&bytes);
                    let body = format_content(&contents, args);
                    return Ok(match recovery_note {
                        Some(note) => format!("{note}\n{body}"),
                        None => body,
                    });
                }
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => {
                    anyhow::bail!("File not found: {}", resolved_path.display());
                }
                std::io::ErrorKind::PermissionDenied => {
                    anyhow::bail!("Permission denied: {}", resolved_path.display());
                }
                _ => {
                    anyhow::bail!("Failed to read file metadata: {e}");
                }
            },
        }

        let mode = super::get_opt_str(args, "mode").unwrap_or("content");

        let body = match mode {
            "symbols" => self.execute_symbols(resolved_path).await?,
            "zoom" => self.execute_zoom(resolved_path, args).await?,
            _ => self.execute_content(resolved_path, args).await?,
        };

        Ok(match recovery_note {
            Some(note) => format!("{note}\n{body}"),
            None => body,
        })
    }

    /// Wildcard path: return matching workspace files instead of failing open.
    async fn recover_wildcard_path(&self, ws: &Workspace, path: &str) -> anyhow::Result<String> {
        if !crate::search_engine::registry_initialized() {
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

    /// Missing literal path: suggest matches or auto-read a single high-confidence hit.
    async fn recover_missing_path(
        &self,
        ws: &Workspace,
        path: &str,
        args: &serde_json::Value,
        original_err: &str,
    ) -> anyhow::Result<String> {
        let matches = find_recovery_candidates(ws, path).await;
        if matches.is_empty() {
            anyhow::bail!("{original_err}");
        }

        if matches.len() == 1 {
            let recovered = &matches[0];
            let resolved = super::path::resolve_read_target(ws.as_path(), recovered).await?;
            let note = recovery_note(path, recovered);
            return self.read_resolved(ws, &resolved, Some(&note), args).await;
        }

        anyhow::bail!("{original_err}\nDid you mean:\n  {}", matches.join("\n  "))
    }

    /// Execute the standard content read mode.
    async fn execute_content(
        &self,
        resolved_path: &Path,
        args: &serde_json::Value,
    ) -> anyhow::Result<String> {
        match tokio::fs::read_to_string(resolved_path).await {
            Ok(contents) => {
                // A raster can be valid UTF-8 (e.g. a minimal GIF patch / a
                // NUL-padded header) — sniff magic bytes before treating it as
                // text so it gets the image annotation rather than rendering as
                // garbage. SVG, source, and other text are not image magic and
                // stay readable.
                if let Some(kind) = sniff_read_image(contents.as_bytes()) {
                    return Ok(image_read_annotation(resolved_path, kind));
                }
                Ok(format_content(&contents, args))
            }
            Err(e) => {
                // Not valid UTF-8 — read raw bytes and try to extract text
                let bytes = tokio::fs::read(resolved_path).await.map_err(|ee| {
                    anyhow::anyhow!(
                        "Initial error: {e}\n\
                         Failed to read file: {ee}"
                    )
                })?;

                // Content-sniff (magic bytes, never the extension) so SVG and
                // other text files stay readable as text and only real raster
                // images take the image path.
                if let Some(kind) = sniff_read_image(&bytes) {
                    return Ok(image_read_annotation(resolved_path, kind));
                }

                // Lossy fallback — replaces invalid bytes with U+FFFD
                let lossy = String::from_utf8_lossy(&bytes).into_owned();
                Ok(lossy)
            }
        }
    }

    /// List all top-level AST symbols with line ranges.
    async fn execute_symbols(&self, resolved_path: &Path) -> anyhow::Result<String> {
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
    async fn execute_zoom(
        &self,
        resolved_path: &Path,
        args: &serde_json::Value,
    ) -> anyhow::Result<String> {
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
            let suggestions = Self::symbol_suggestions(&ctx.ps, &ctx.query, symbol_name);
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
}

/// Format file contents for content-mode output (line numbering + offset/limit).
#[must_use]
fn format_content(contents: &str, args: &serde_json::Value) -> String {
    let lines: Vec<&str> = contents.lines().collect();
    let total = lines.len();

    if total == 0 {
        return String::new();
    }

    let offset = super::get_opt_u64(args, "offset").map_or(0, |v| {
        usize::try_from(v.max(1))
            .unwrap_or(usize::MAX)
            .saturating_sub(1)
    });
    let start = offset.min(total);

    let end = match super::get_opt_u64(args, "limit") {
        Some(l) => {
            let limit = usize::try_from(l).unwrap_or(usize::MAX);
            (start.saturating_add(limit)).min(total)
        }
        None => total,
    };

    if start >= end {
        return format!("[No lines in range, file has {total} lines]");
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

    format!("{summary}\n{numbered}")
}

/// Default bound on FIFO (named pipe) reads: how long to wait for a writer
/// before erroring. A FIFO with no writer must never hang the tool.
#[cfg(unix)]
const DEFAULT_FIFO_READ_TIMEOUT_SECS: u64 = 10;

/// FIFO read bound for [`read_fifo`]. Overridable via env for tuning; tests
/// pass explicit durations.
#[cfg(unix)]
fn fifo_read_timeout() -> Duration {
    crate::util::env_duration_secs(
        "MAHBOT_FIFO_READ_TIMEOUT_SECS",
        DEFAULT_FIFO_READ_TIMEOUT_SECS,
    )
}

/// Read all bytes from a FIFO with a bounded wait, so a missing writer errors
/// instead of hanging forever.
///
/// The FIFO is opened `O_NONBLOCK` — the open itself never blocks on a
/// missing writer (unlike a blocking open), and every read returns
/// immediately (data, EOF, or `WouldBlock`). The loop polls with a sliding
/// deadline: while no writer is open the reads return EOF (0 bytes), which is
/// treated as "no data yet" and polled again — a live writer that opens after
/// the reader still delivers its data (FIFOs are not sticky-EOF). A
/// successful read resets the deadline, so a writer that keeps producing data
/// keeps the read alive (capped by the size limit) — deliberate: only the
/// no-data / no-EOF case must be bounded. A writer that wrote data then went
/// idle past the bound still gets its buffered bytes delivered (fail-open);
/// only a writer that produced nothing errors. No blocking thread is ever
/// leaked — a bare `timeout` around a blocking read would pin a tokio
/// blocking-pool thread for every hang.
#[cfg(unix)]
async fn read_fifo(path: &Path, timeout: Duration) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open {}: {e}", path.display()))?;

    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut last_activity = std::time::Instant::now();
    loop {
        let remaining = timeout.saturating_sub(last_activity.elapsed());
        if remaining.is_zero() {
            if !buf.is_empty() {
                // Data arrived before the bound — deliver it rather than
                // discarding bytes the agent already received.
                return Ok(buf);
            }
            // One final non-blocking read: a writer may have delivered bytes
            // during the last inactivity sleep — deliver them instead of a
            // spurious error (bytes stay in the FIFO buffer either way).
            let timed_out = || {
                anyhow::anyhow!(
                    "Timed out after {:.0}s waiting for FIFO data on {} \
                 (no writer appeared or a writer left the pipe open)",
                    timeout.as_secs_f64(),
                    path.display()
                )
            };
            match file.read(&mut chunk) {
                Ok(0) => return Err(timed_out()),
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    return Ok(buf);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Err(timed_out()),
                Err(e) => {
                    anyhow::bail!("Failed reading FIFO {}: {e}", path.display());
                }
            }
        }
        match file.read(&mut chunk) {
            Ok(0) => {
                // EOF with an empty buffer: no writer is open right now. A
                // writer may still appear, so keep polling until the bound.
                if buf.is_empty() {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
                return Ok(buf); // writer finished — normal EOF
            }
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() as u64 > super::MAX_FILE_SIZE_BYTES {
                    anyhow::bail!(
                        "FIFO {} output exceeded {} bytes",
                        path.display(),
                        super::MAX_FILE_SIZE_BYTES
                    );
                }
                last_activity = std::time::Instant::now();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Writer present but idle — poll again shortly.
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(e) => {
                anyhow::bail!("Failed reading FIFO {}: {e}", path.display());
            }
        }
    }
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
            ALL_TREE_SITTER_EXTENSIONS.join(", .")
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
            && let Some((start, end)) = range.split_once(" of ")
        {
            if let Some((lo, hi)) = start.split_once('-') {
                let lo: usize = lo.parse().unwrap_or(0);
                let hi: usize = hi.parse().unwrap_or(0);
                return hi.saturating_sub(lo) + 1;
            }
            // edge: "[Lines X of Z]" shouldn't happen but handle gracefully
            if let Ok(n) = start.parse::<usize>() {
                let end_n: usize = end.parse().unwrap_or(0);
                return end_n.saturating_sub(n) + 1;
            }
        }
    }
    0
}

/// Delegate directory listing to [`ShellTool`] when [`ReadTool`] receives a
/// directory path.
///
/// Constructs a `ls -lA -- <quoted_path>` command and executes it in read-only
/// mode. The result goes through `process_shell_output` which applies
/// compact_ls formatting (directory/file separation, sizes, extension
/// summaries), timing, and spill-to-file for large listings.
///
/// The `--` separator prevents directory names starting with `-` from being
/// misinterpreted as flags. The path is shell-quoted via [`shell_quote`] to
/// handle special characters.
async fn list_directory(resolved_path: &std::path::Path, ws: &Workspace) -> anyhow::Result<String> {
    let quoted = shell_quote(&resolved_path.to_string_lossy());
    let command = format!("ls -lA -- {quoted}");
    let shell_tool = ShellTool::new(ShellMode::ReadOnly);
    shell_tool.execute(ws, json!({"command": command})).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use std::path::PathBuf;
    use tempfile::TempDir;

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

    /// Every extension in the canonical `ALL_TREE_SITTER_EXTENSIONS` constant
    /// must have a `language_support` entry. Derived automatically from the
    /// canonical source (`crate::util::tree_sitter::ALL_TREE_SITTER_EXTENSIONS`)
    /// so that adding an extension to the canonical function automatically
    /// updates this test.
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
        for ext in ALL_TREE_SITTER_EXTENSIONS {
            assert!(
                language_support(ext).is_some(),
                "expected language support for .{ext}"
            );
        }
    }

    /// Spot-check that common non-code extensions return no language support.
    /// Helps catch accidental regressions in `language_support` match arms.
    /// Not exhaustive — adding a new supported extension without updating the
    /// error message still passes silently.
    #[test]
    fn unsupported_extensions_return_none() {
        let unsupported: &[&str] = &[
            "txt", "yml", "yaml", "xml", "svg", "config", "ini", "cfg", "log", "csv", "tsv", "pdf",
            "png", "jpg", "gif", "woff", "ttf",
        ];
        for ext in unsupported {
            assert!(
                language_support(ext).is_none(),
                "expected no language support for .{ext}"
            );
        }
    }

    #[tokio::test]
    async fn file_read_basic_scenarios() {
        let (_dir, ws_path) = temp_workspace(&[("test.txt", "hello world")]);

        // existing file
        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "test.txt"}))
            .await;
        assert!(result.is_ok(), "read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("1: hello world"));
        assert!(result.contains("[1 lines total]"));
        // nonexistent file
        let result = ReadTool
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
        let result = ReadTool
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
    async fn read_wildcard_without_search_index_returns_helpful_error() {
        // When the search engine is already initialized by other tests (full
        // suite), the "without index" condition cannot be reproduced. Skip
        // silently so CI isn't broken; run this test individually to verify
        // the error path: cargo test -- read_wildcard_without_search_index
        if crate::search_engine::registry_initialized() {
            return;
        }

        let (_dir, ws_path) = temp_workspace(&[("alpha.rs", "fn alpha() {}")]);

        let result = ReadTool
            .execute(&test_ws(&ws_path), json!({"path": "*.rs"}))
            .await;
        assert!(
            result.is_err(),
            "wildcard without index should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("search index") || err.contains("Wildcard"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn file_read_blocks_unsafe_paths() {
        // path traversal
        let (dir1, ws_path1) = temp_workspace(&[]);

        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path1),
                json!({"path": "../../../etc/passwd"}),
            )
            .await;
        assert!(result.is_err(), "traversal should be blocked: {result:?}");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("not allowed"));
        // absolute path
        let result = ReadTool
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
        assert!(err.contains("not allowed"));
        // null byte in path — separate workspace
        drop(dir1);
        let (_dir2, ws_path2) = temp_workspace(&[]);

        let result = ReadTool
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
        assert!(err.contains("not allowed"));
    }

    #[tokio::test]
    async fn file_read_nested_path() {
        let (_dir, ws_path) = temp_workspace(&[("sub/dir/deep.txt", "deep content")]);

        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "sub/dir/deep.txt"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "nested path read should succeed: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("1: deep content"));
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

        let result = ReadTool
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
        assert!(err.contains("security policy"));
    }

    #[tokio::test]
    async fn file_read_offset_handling() {
        let (_dir, ws_path) = temp_workspace(&[("lines.txt", "aaa\nbbb\nccc\nddd\neee")]);

        // Read lines 2-3
        let result = ReadTool
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
        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "lines.txt", "offset": 4}),
            )
            .await;
        assert!(result.is_ok(), "offset to end should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("4: ddd") && result.contains("5: eee"));
        // Limit only (first 2 lines)
        let result = ReadTool
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
        let result = ReadTool
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

        let result = ReadTool
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

        // Write bytes that are not valid UTF-8 and not a PDF
        let binary_data: Vec<u8> = vec![0x00, 0x80, 0xFF, 0xFE, b'h', b'i', 0x80];
        tokio::fs::write(ws_path.join("data.bin"), &binary_data)
            .await
            .unwrap();

        let result = ReadTool
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

    /// Content-mode read of a native raster returns the image annotation and
    /// attaches the image to the conversation instead of lossy garbage.
    #[tokio::test]
    async fn file_read_native_image_returns_annotation() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let png = tiny_png_bytes(4, 4);
        tokio::fs::write(ws_path.join("tiny.png"), &png)
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "tiny.png"}))
            .await
            .expect("native image read must succeed");
        // The ReadTool's execute output is claim-neutral — the agent loop
        // appends the 'attached'/'already attached' qualifier from the payload
        // it actually injects, so a read that cannot be injected never falsely
        // claims an attachment.
        assert!(result.contains("Read image file"), "got: {result}");
        assert!(result.contains("PNG"), "got: {result}");
        assert!(
            !result.contains("attached to the conversation"),
            "read output must stay claim-neutral, got: {result}"
        );
    }

    /// A recognised-but-unsupported raster (GIF) is reported gracefully, not
    /// decoded into garbage or treated as an error.
    #[tokio::test]
    async fn file_read_unsupported_image_reports_unsupported() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let mut gif: Vec<u8> = b"GIF89a".to_vec();
        gif.extend_from_slice(&[0u8; 20]);
        tokio::fs::write(ws_path.join("bad.gif"), &gif)
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "bad.gif"}))
            .await
            .expect("unsupported image read must succeed");
        assert!(result.contains("unsupported image format"), "got: {result}");
        assert!(result.contains("GIF"), "got: {result}");
    }

    /// A native raster with valid magic + header dimensions but a corrupt /
    /// truncated body (no pixel data) is still reported as a PNG (the cheap
    /// magic sniff only looks at the leading magic bytes) but never claims an
    /// attachment — the decode is deferred to `image_payload`, so the base
    /// annotation stays honest about what the read itself demonstrated.
    #[tokio::test]
    async fn file_read_corrupt_image_does_not_claim_attachment() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let png = tiny_png_bytes(4, 4);
        // Truncate to the PNG signature + IHDR (valid magic + header dimensions,
        // but no IDAT/IEND) — must be reported, not claimed as attached.
        let truncated = &png[..33];
        assert!(image::guess_format(truncated).is_ok());
        tokio::fs::write(ws_path.join("corrupt.png"), truncated)
            .await
            .unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "corrupt.png"}),
            )
            .await
            .expect("corrupt image read must succeed");
        assert!(result.contains("Read image file"), "got: {result}");
        assert!(result.contains("PNG"), "got: {result}");
        assert!(
            !result.contains("attached to the conversation"),
            "corrupt image must not claim attachment, got: {result}"
        );
    }

    /// A native image read produces a compressed JPEG data-URI payload that the
    /// agent loop injects as a synthetic user message.
    #[tokio::test]
    async fn file_read_image_payload_produces_data_uri() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let png = tiny_png_bytes(4, 4);
        tokio::fs::write(ws_path.join("tiny.png"), &png)
            .await
            .unwrap();
        let ws = Workspace::from_path(&ws_path);

        let payload = ReadTool
            .image_payload(&ws, &json!({"path": "tiny.png"}))
            .await;
        let payload = payload.expect("image payload must be produced");
        assert!(
            payload.data_uri.starts_with("data:image/jpeg;base64,"),
            "unexpected data-uri: {}",
            payload.data_uri
        );
        assert_eq!(payload.width, 4, "unexpected payload width");
        assert_eq!(payload.height, 4, "unexpected payload height");
        assert_eq!(payload.format, "PNG", "unexpected payload format");
        // `payload.path` is the canonicalized resolved path (resolve_read_target
        // canonicalizes, which on macOS resolves the /tmp → /private/tmp symlink).
        let expected_path = tokio::fs::canonicalize(ws_path.join("tiny.png"))
            .await
            .unwrap();
        assert_eq!(
            payload.path,
            expected_path.display().to_string(),
            "unexpected payload path"
        );
    }

    /// Non-image files (text) produce no image payload.
    #[tokio::test]
    async fn image_payload_non_image_returns_none() {
        let (_dir, ws_path) = temp_workspace(&[("hello.txt", "hello world")]);
        let ws = Workspace::from_path(&ws_path);
        let payload = ReadTool
            .image_payload(&ws, &json!({"path": "hello.txt"}))
            .await;
        assert!(payload.is_none());
    }

    /// End-to-end pipeline regression: `execute` produces a claim-neutral
    /// base, then `image_payload` (the single decoder) produces the payload
    /// that the agent loop would inject — validated without passing the
    /// annotation wording to the gate.
    #[tokio::test]
    async fn execute_then_payload_attaches_native_image() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let png = tiny_png_bytes(4, 4);
        tokio::fs::write(ws_path.join("tiny.png"), &png)
            .await
            .unwrap();
        let ws = Workspace::from_path(&ws_path);

        let result = ReadTool
            .execute(&ws, json!({"path": "tiny.png"}))
            .await
            .expect("native image read must succeed");
        assert!(result.contains("Read image file"), "got: {result}");
        assert!(result.contains("PNG"), "got: {result}");
        assert!(
            !result.contains("attached to the conversation"),
            "execute output must stay claim-neutral, got: {result}"
        );

        let payload = ReadTool
            .image_payload(&ws, &json!({"path": "tiny.png"}))
            .await
            .expect("pipeline image payload must be produced");
        assert_eq!(payload.width, 4, "unexpected payload width");
        assert_eq!(payload.height, 4, "unexpected payload height");
        assert_eq!(payload.format, "PNG", "unexpected payload format");
    }

    /// Regression: a typo'd image path that `execute` recovers to a unique
    /// match is actually attached by `image_payload` (not just annotated), and
    /// the payload carries the `[Recovered path: ...]` note. Pins the symmetry
    /// between `recover_missing_path` and `resolve_for_read` so the recovery
    /// logic cannot drift back into the 'annotated but not attached' gap.
    #[tokio::test]
    async fn recovered_image_path_is_attached() {
        // The fuzzy search needs the global search-engine registry plus a
        // configured storage root (for the persistent query tracker).
        crate::util::test::init_test_stores().await;

        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().to_path_buf();
        let png = tiny_png_bytes(4, 4);
        tokio::fs::write(ws_path.join("tiny_image.png"), &png)
            .await
            .unwrap();
        let ws = Workspace::from_path(&ws_path);

        // execute with a typo'd path that the fuzzy matcher recovers to the file.
        let result = ReadTool
            .execute(&ws, json!({"path": "tiny_imag.png"}))
            .await
            .expect("recovered image read must succeed");
        assert!(result.contains("[Recovered path:"), "got: {result}");
        assert!(result.contains("Read image file"), "got: {result}");

        // image_payload must attach the recovered image (not just annotate it),
        // and surface the recovery note for the tool-result annotation.
        let payload = ReadTool
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

    /// A payload's recovered-path note is prepended to the tool-result
    /// annotation, so recovered image reads keep the same `[Recovered path: ...]`
    /// context that recovered text reads already show.
    #[test]
    fn image_payload_recovery_note_prepends_annotation() {
        let payload = crate::tools::ImagePayload {
            path: "/tmp/y.png".into(),
            data_uri: "data:image/jpeg;base64,aaa".into(),
            width: 4,
            height: 4,
            format: "PNG".into(),
            recovery_note: Some("[Recovered path: requested 'x.png', using 'y.png']".into()),
            source: crate::tools::ImagePayloadSource::Read,
        };
        let fresh = payload.attached_annotation();
        assert!(
            fresh.starts_with("[Recovered path: requested 'x.png', using 'y.png']\n"),
            "fresh must keep the recovery note: {fresh}"
        );
        assert!(
            fresh.contains("Image content attached to the conversation as a native image."),
            "fresh: {fresh}"
        );
        let dup = payload.already_attached_annotation();
        assert!(
            dup.starts_with("[Recovered path: requested 'x.png', using 'y.png']\n"),
            "dup must keep the recovery note: {dup}"
        );
        assert!(dup.contains("already attached"), "dup: {dup}");
    }

    /// Short output should pass through unchanged.
    #[test]
    fn format_output_short_passthrough() {
        let input = "[3 lines total]\n1: a\n2: b\n3: c";
        let result = ReadTool.format_output(input);
        assert_eq!(result, input);
    }

    /// Long output keeps the header + as many complete lines as fit + omitted count.
    #[test]
    fn format_output_truncates_at_line_boundary() {
        // Build a header line + many long body lines
        let header = "[500 lines total]";
        let body_lines: String = (1..=500)
            .map(|i| format!("{}: {}", i, "x".repeat(200)))
            .collect::<Vec<_>>()
            .join("\n");
        let input = format!("{header}\n{body_lines}");

        let result = ReadTool.format_output(&input);

        // Header must be at the top, preserved
        assert!(result.starts_with(header), "header must be first");
        // Must end with "N lines omitted" marker
        assert!(
            result.contains("lines omitted)"),
            "must contain omitted count, got: {result}"
        );
        // No "more bytes" marker (that's the default head+tail behavior we're avoiding)
        assert!(
            !result.contains("more bytes"),
            "must not contain head+tail marker"
        );
        // Kept lines count + omitted should equal expected
        let omitted: usize = result
            .lines()
            .last()
            .and_then(|l| l.strip_prefix("... ("))
            .and_then(|l| l.strip_suffix(" lines omitted)"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let kept = result.lines().count() - 2; // minus header and marker
        assert_eq!(kept + omitted, 500, "kept + omitted must equal 500");
    }

    /// Lossy/binary output without a structured header falls back to default truncation.
    #[test]
    fn format_output_fallback_for_unstructured_output() {
        let input = "a".repeat(6000);
        let result = ReadTool.format_output(&input);
        assert!(result.contains("bytes omitted at tool output truncation"));
    }

    /// Symbols mode lists top-level declarations for Rust files.
    #[tokio::test]
    async fn symbols_mode_lists_rust_symbols() {
        let code = r"
fn hello() {}
struct Point { x: i32, y: i32 }
enum Color { Red, Blue }
trait Draw { fn draw(&self); }
impl Point { fn new() -> Self { Point { x: 0, y: 0 } } }
const MAX: usize = 100;
type MyInt = i32;
macro_rules! my_macro { () => {} }
mod utils;
";
        let (_dir, ws_path) = temp_workspace(&[("lib.rs", code)]);

        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "lib.rs", "mode": "symbols"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "symbols failed: {:?}",
            result.as_ref().unwrap_err()
        );
        let result = result.unwrap();
        assert!(result.contains("[Symbols in lib.rs]"), "missing header");
        assert!(result.contains("fn `hello`"), "missing fn hello");
        assert!(result.contains("struct `Point`"), "missing struct Point");
        assert!(result.contains("enum `Color`"), "missing enum Color");
        assert!(result.contains("trait `Draw`"), "missing trait Draw");
        assert!(result.contains("impl `Point`"), "missing impl Point");
        assert!(result.contains("const `MAX`"), "missing const MAX");
        assert!(result.contains("type `MyInt`"), "missing type MyInt");
        assert!(result.contains("mod `utils`"), "missing mod utils");
    }

    /// Symbols mode returns clear error for unsupported extensions.
    #[tokio::test]
    async fn symbols_mode_unsupported_extension() {
        let (_dir, ws_path) = temp_workspace(&[("data.yaml", "{}")]);

        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "data.yaml", "mode": "symbols"}),
            )
            .await;
        assert!(
            result.is_err(),
            "unsupported extension should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Unsupported"));
    }

    /// Zoom mode extracts a specific symbol's source.
    /// Also verifies correct disambiguation: parameter names and local variables
    /// with the same name as another function should not match.
    #[tokio::test]
    async fn zoom_mode_extracts_rust_function() {
        let code =
            "fn greet(name: &str) -> String {\n    format!(\"Hi, {name}!\")\n}\n\nfn main() {}";
        let (_dir, ws_path) = temp_workspace(&[("main.rs", code)]);

        let result = ReadTool
            .execute(
                &test_ws(&ws_path),
                json!({"path": "main.rs", "mode": "zoom", "symbol": "greet"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "zoom failed: {:?}",
            result.as_ref().unwrap_err()
        );
        let result = result.unwrap();
        assert!(result.contains("fn `greet`"), "missing fn greet label");
        assert!(
            result.contains("format!(\"Hi, {name}!\")"),
            "missing function body"
        );
    }

    /// Zoom mode returns helpful error for nonexistent symbol.
    #[tokio::test]
    async fn zoom_mode_symbol_not_found() {
        let (_dir, ws_path) = temp_workspace(&[("lib.rs", "fn existing() {}")]);

        let result = ReadTool
            .execute(
                &test_ws(&ws_path),
                json!({"path": "lib.rs", "mode": "zoom", "symbol": "nope"}),
            )
            .await;
        assert!(result.is_err(), "missing symbol should fail: {result:?}");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("'nope'"), "missing symbol name in error");
        assert!(
            err.contains("Did you mean"),
            "should suggest available symbols: {err}"
        );
        assert!(
            err.contains("existing"),
            "should list existing symbol: {err}"
        );
    }

    /// Zoom mode requires symbol parameter.
    #[tokio::test]
    async fn zoom_mode_missing_symbol_param() {
        let (_dir, ws_path) = temp_workspace(&[("lib.rs", "fn f() {}")]);

        let result = ReadTool
            .execute(
                &Workspace::from_path(&ws_path),
                json!({"path": "lib.rs", "mode": "zoom"}),
            )
            .await;
        assert!(
            result.is_err(),
            "missing symbol param should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Missing 'symbol' parameter"));
    }

    /// Directory listing returns file names instead of erroring.
    #[tokio::test]
    async fn directory_listing_returns_contents() {
        let (_dir, ws_path) = temp_workspace(&[("a.txt", "alpha"), ("b.rs", "beta")]);
        tokio::fs::create_dir(ws_path.join("sub")).await.unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "."}))
            .await;
        assert!(result.is_ok(), "dir listing should succeed: {result:?}");
        let output = result.unwrap();
        // Should contain file names
        assert!(output.contains("a.txt"), "should list a.txt: {output}");
        assert!(output.contains("b.rs"), "should list b.rs: {output}");
        // Should contain subdirectory name with trailing slash
        assert!(output.contains("sub/"), "should list sub/: {output}");
        // Should NOT be the old error message
        assert!(!output.contains("Path is a directory"), "should not error");
    }

    /// Subdirectories without a trailing slash should list contents, not error.
    #[tokio::test]
    async fn directory_listing_subdir_without_trailing_slash() {
        let (_dir, ws_path) = temp_workspace(&[("sub/inside.txt", "nested")]);

        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "sub"}))
            .await;
        assert!(
            result.is_ok(),
            "subdir without trailing slash should list: {result:?}"
        );
        let output = result.unwrap();
        assert!(
            output.contains("inside.txt"),
            "should list inside.txt: {output}"
        );
        assert!(
            !output.contains("File not found"),
            "should not report missing file: {output}"
        );
    }

    /// Directory listing shows "(empty)" for empty directories.
    #[tokio::test]
    async fn directory_listing_empty() {
        let (_dir, ws_path) = temp_workspace(&[]);

        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "."}))
            .await;
        assert!(
            result.is_ok(),
            "empty dir listing should succeed: {result:?}"
        );
        let output = result.unwrap();
        // compact_ls preserves "total 0" for empty directories with no entries
        assert!(
            output.contains("total 0") || output.contains("(empty)"),
            "empty dir should indicate emptiness: {output}"
        );
    }

    /// Directory listing handles paths with spaces and special characters.
    #[tokio::test]
    async fn directory_listing_spaces_in_path() {
        let dir = TempDir::new().unwrap();
        let ws_path = dir.path().join("my workspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();
        tokio::fs::write(ws_path.join("my file.txt"), "content")
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&ws_path), json!({"path": "."}))
            .await;
        assert!(result.is_ok(), "dir with spaces should succeed: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("my file.txt"), "should list file: {output}");
    }

    /// Directory listing resolves symlinks to directories.
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
        let result = ReadTool
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

    /// The shell_quote function handles various edge cases.
    #[test]
    fn shell_quoting_edge_cases() {
        // Simple path
        assert_eq!(shell_quote("/tmp/dir"), "'/tmp/dir'");
        // Path with spaces
        assert_eq!(shell_quote("/my dir/file"), "'/my dir/file'");
        // Path with single quote
        assert_eq!(shell_quote("/it's dir"), "'/it'\\''s dir'");
        // Path with dollar sign
        assert_eq!(shell_quote("/$dir"), "'/$dir'");
        // Path with backtick
        assert_eq!(shell_quote("/`dir`"), "'/`dir`'");
        // Path with backslash
        assert_eq!(shell_quote("/dir\\name"), "'/dir\\name'");
        // Empty string
        assert_eq!(shell_quote(""), "''");
        // Already quoted — just wraps
        assert_eq!(shell_quote("normal"), "'normal'");
    }

    #[test]
    fn is_sensitive_file_path_env_and_certs() {
        assert!(is_sensitive_file_path(".env"));
        assert!(is_sensitive_file_path("proj/.env"));
        assert!(is_sensitive_file_path(".env.local"));
        assert!(is_sensitive_file_path("/abs/path/.env.production"));
        assert!(is_sensitive_file_path("secrets/local.env"));
        assert!(is_sensitive_file_path("tls/cert.pem"));
        assert!(is_sensitive_file_path("C:\\keys\\id_rsa.key"));

        assert!(!is_sensitive_file_path("src/main.rs"));
        assert!(!is_sensitive_file_path("crates/foo/lib.rs"));
        assert!(!is_sensitive_file_path("README.md"));
    }

    // ── prepare_symbol_query / collect_symbols ──────────────────────────────

    #[tokio::test]
    async fn prepare_symbol_query_valid_file() {
        let (_dir, ws_path) = temp_workspace(&[("lib.rs", "fn hello() {}\nstruct World;\n")]);
        let file_path = ws_path.join("lib.rs");

        let result = prepare_symbol_query(&file_path, "test").await;
        assert!(
            result.is_ok(),
            "prepare_symbol_query should succeed for .rs: {result:?}"
        );

        let ctx = result.unwrap();
        // The query was built successfully
        assert!(
            !ctx.ps.symbol_query.is_empty(),
            "expected non-empty symbol_query for .rs"
        );
        // collect_symbols should find our symbols
        let symbols = collect_symbols(&ctx.ps, &ctx.query);
        assert_eq!(symbols.len(), 2, "expected 2 symbols, got {symbols:?}");
        // fn hello
        assert!(symbols.iter().any(|s| s.name == "hello"));
        // struct World
        assert!(symbols.iter().any(|s| s.name == "World"));
    }

    #[tokio::test]
    async fn prepare_symbol_query_unsupported_extension() {
        let (_dir, ws_path) = temp_workspace(&[("data.txt", "hello world")]);
        let file_path = ws_path.join("data.txt");

        let result = prepare_symbol_query(&file_path, "test").await;
        assert!(result.is_err(), "expected error for unsupported extension");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("Unsupported"),
            "error should mention unsupported: {err}"
        );
    }

    #[tokio::test]
    async fn collect_symbols_empty_file() {
        let (_dir, ws_path) = temp_workspace(&[("empty.rs", "")]);
        let file_path = ws_path.join("empty.rs");

        let ctx = prepare_symbol_query(&file_path, "test").await.unwrap();
        let symbols = collect_symbols(&ctx.ps, &ctx.query);
        assert!(
            symbols.is_empty(),
            "expected no symbols in empty file, got {symbols:?}"
        );
    }

    #[tokio::test]
    async fn collect_symbols_multiple_captures() {
        // A Rust file with various symbol types
        let code = r"
fn foo() {}
fn bar() {}
struct Baz;
enum Qux {}
impl Baz {}
";
        let (_dir, ws_path) = temp_workspace(&[("main.rs", code)]);
        let file_path = ws_path.join("main.rs");

        let ctx = prepare_symbol_query(&file_path, "test").await.unwrap();
        let symbols = collect_symbols(&ctx.ps, &ctx.query);
        // We expect: foo, bar, Baz, Qux, Baz (impl)
        assert_eq!(symbols.len(), 5, "expected 5 symbols, got {symbols:?}");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"bar"));
        assert!(names.contains(&"Baz"));
        assert!(names.contains(&"Qux"));
    }

    #[tokio::test]
    async fn execute_symbols_integration() {
        let (_dir, ws_path) = temp_workspace(&[("app.rs", "fn greet() {}\nstruct Person;\n")]);
        let result = ReadTool
            .execute(
                &crate::workspace::test_ws(&ws_path),
                json!({"path": "app.rs", "mode": "symbols"}),
            )
            .await;
        assert!(result.is_ok(), "execute_symbols should succeed: {result:?}");
        let output = result.unwrap();
        assert!(
            output.contains("`greet`"),
            "output should contain greet: {output}"
        );
        assert!(
            output.contains("`Person`"),
            "output should contain Person: {output}"
        );
        assert!(
            output.contains("fn"),
            "output should have 'fn' kind label: {output}"
        );
        assert!(
            output.contains("struct"),
            "output should have 'struct' kind label: {output}"
        );
    }

    #[tokio::test]
    async fn collect_symbols_preserves_line_numbers() {
        let code = "fn hello() {}\n\n\nfn world() {}\n";
        let (_dir, ws_path) = temp_workspace(&[("lib.rs", code)]);
        let file_path = ws_path.join("lib.rs");

        let ctx = prepare_symbol_query(&file_path, "test").await.unwrap();
        let symbols = collect_symbols(&ctx.ps, &ctx.query);
        let hello = symbols.iter().find(|s| s.name == "hello").unwrap();
        let world = symbols.iter().find(|s| s.name == "world").unwrap();
        assert_eq!(hello.start_line, 1, "hello starts at line 1");
        assert_eq!(world.start_line, 4, "world starts at line 4");
    }

    /// A FIFO with no writer must error within the bound, never hang. The
    /// no-writer case is detected at the bound (empty EOF keeps polling).
    #[cfg(unix)]
    #[tokio::test]
    async fn fifo_without_writer_errors_bounded() {
        let dir = TempDir::new().unwrap();
        let fifo_path = dir.path().join("nofifo.pipe");
        let c_path = std::ffi::CString::new(fifo_path.as_os_str().as_encoded_bytes()).unwrap();
        let ret = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(ret, 0, "mkfifo should succeed");

        let result = read_fifo(&fifo_path, Duration::from_millis(300)).await;
        let err = result.expect_err("no-writer FIFO must error");
        assert!(
            err.to_string().contains("no writer"),
            "error should name the no-writer cause: {err}"
        );

        // The tool-level path also errors instead of hanging (bounded via env).
        let _guard = crate::util::test::set_env_var("MAHBOT_FIFO_READ_TIMEOUT_SECS", Some("1"));
        let ws = test_ws(dir.path());
        let result = ReadTool
            .execute(
                &ws,
                json!({"path": fifo_path.to_string_lossy().into_owned()}),
            )
            .await;
        let err = result.expect_err("tool read of no-writer FIFO must error");
        assert!(
            err.to_string().contains("no writer"),
            "tool error should name the no-writer cause: {err}"
        );
    }

    /// A FIFO with a live writer keeps working — the bounded read returns the
    /// written data instead of erroring. The writer opens after the reader
    /// (read-fifo opens O_NONBLOCK and polls), proving late writers work.
    #[cfg(unix)]
    #[tokio::test]
    async fn fifo_with_writer_reads_data() {
        let dir = TempDir::new().unwrap();
        let fifo_path = dir.path().join("wfifo.pipe");
        let c_path = std::ffi::CString::new(fifo_path.as_os_str().as_encoded_bytes()).unwrap();
        let ret = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(ret, 0, "mkfifo should succeed");

        // Writer on the blocking pool: the write-open blocks until a reader
        // appears, which must not pin the async executor.
        let writer_path = fifo_path.clone();
        let writer = tokio::task::spawn_blocking(move || {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&writer_path)
                .expect("open FIFO for writing");
            f.write_all(b"fifo data\n").expect("write to FIFO");
        });

        let result = read_fifo(&fifo_path, Duration::from_secs(5)).await;
        writer.await.unwrap();
        let bytes = result.expect("FIFO with writer should read data");
        assert_eq!(bytes, b"fifo data\n");
    }

    /// A writer that wrote data then holds the pipe open idle past the bound:
    /// the already-received bytes are delivered (fail-open), never discarded.
    ///
    /// This test is `#[ignore]` by default because the writer holds the pipe open with a hardcoded 2 s sleep. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test fifo_writer_idle_after_data_delivers_buffered_bytes -- --ignored --nocapture
    /// ```
    #[ignore = "hardcoded 2 s writer sleep to verify fail-open buffered delivery; runs only when explicitly invoked"]
    #[cfg(unix)]
    #[tokio::test]
    async fn fifo_writer_idle_after_data_delivers_buffered_bytes() {
        let dir = TempDir::new().unwrap();
        let fifo_path = dir.path().join("idle.pipe");
        let c_path = std::ffi::CString::new(fifo_path.as_os_str().as_encoded_bytes()).unwrap();
        let ret = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(ret, 0, "mkfifo should succeed");

        let writer_path = fifo_path.clone();
        let writer = tokio::task::spawn_blocking(move || {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&writer_path)
                .expect("open FIFO for writing");
            f.write_all(b"buffered data\n").expect("write to FIFO");
            // Hold the pipe open idle well past the reader's bound.
            std::thread::sleep(std::time::Duration::from_secs(2));
        });

        let result = read_fifo(&fifo_path, Duration::from_millis(300)).await;
        writer.await.unwrap();
        let bytes = result.expect("bytes written before the bound must be delivered");
        assert_eq!(bytes, b"buffered data\n");
    }
}
