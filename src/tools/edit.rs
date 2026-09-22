use crate::{Tool, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

/// Edit a file by replacing a string match with new content.
///
/// Two modes:
/// - **Write mode**: when `old_string` is omitted or empty, creates a new file with
///   `new_string` as the content (including parent directories). Refuses to
///   overwrite an existing file — use edit mode for changes.
/// - **Edit mode**: when `old_string` is provided, performs precise replacement
///   within an existing file. Line endings are always tolerated, for every file
///   type: `\r\n` and `\n` are interchangeable when matching, and the written
///   text takes the ending of the file around the match rather than the one it
///   was spelled with, so an edit introduces no ending the file did not have.
///   Matching is additionally semi-insensitive to whitespace for code files
///   (.rs, .js, .ts, .c, .cpp, .go, etc.): extra/missing spaces outside string
///   literals are tolerated there. By default the `old_string` must appear
///   exactly once (zero matches = not found, multiple = ambiguous).
///   `new_string` may be empty to delete the matched text.
pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "path": {
                    "type": "string",
                    "description": "Path to the file (required on every call). Relative paths resolve from workspace; absolute paths are validated against the workspace boundary."
                },
                "old_string": {
                    "type": "string",
                    "description": "If omitted or empty: creates a new file with `new_string` (refuses if file exists). If provided and non-empty: this text is replaced by `new_string`. Its line endings never have to match the file's (LF and CRLF are interchangeable, for every file type), and it is semi-insensitive to whitespace in code files; it must appear exactly once unless multiple is true."
                },
                "new_string": {
                    "type": "string",
                    "description": "When old_string is omitted or empty: the content to write to the new file. When old_string is provided and non-empty: the replacement text (may be empty to delete the matched text); its line endings are rewritten to the ending the file uses around it. Must differ from old_string — identical old and new strings are rejected as a no-op."
                },
                "multiple": {
                    "type": "boolean",
                    "description": "Only used when old_string is provided. Allow replacing multiple occurrences of old_string (default: false). When true, replaces all occurrences instead of requiring exactly one.",
                    "default": false
                }
            }),
            &["path", "new_string"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let path = super::get_str(&args, "path")?.to_string();
        let new_string = super::get_str(&args, "new_string")?;
        let old_string = super::get_opt_str(&args, "old_string");

        match old_string {
            None | Some("") => self.execute_write(ws, &path, new_string).await,
            Some(old) => {
                let multiple = super::get_bool(&args, "multiple", false)?;
                self.execute_edit(ws, &path, old, new_string, multiple)
                    .await
            }
        }
    }
}

/// Build the canonical `not-found` tool error for a failed `old_string` match.
///
/// `detail` carries the optional variant clause, e.g.
/// `" (whitespace differs; try without multiple=true)"`.
fn not_found_error(path: &str, detail: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "not-found: cannot edit {path}: old_string not found in file{detail} \
         — hint: re-read the file and copy old_string exactly from its current contents"
    )
}

impl EditTool {
    /// Write a new file with the given content.
    async fn execute_write(&self, ws: &Workspace, path: &str, new_string: &str) -> Result<String> {
        let resolved_target = super::path::resolve_write_target(ws.as_path(), path, true).await?;

        if tokio::fs::try_exists(&resolved_target)
            .await
            .map_err(|e| anyhow::anyhow!("Cannot verify whether {path} exists: {e}"))?
        {
            anyhow::bail!(
                "File already exists: {path}. Use `old_string` to edit it instead of overwriting."
            );
        }

        tokio::fs::write(&resolved_target, new_string)
            .await
            .map_err(|e| anyhow::anyhow!("io: cannot write {path}: failed to write file: {e} — hint: check that the parent directory exists and is writable"))?;
        update_search_index_after_write(ws, &resolved_target);
        Ok(format!("Written {} bytes to {path}", new_string.len()))
    }

    /// Edit an existing file by replacing `old_string` with `new_string`.
    async fn execute_edit(
        &self,
        ws: &Workspace,
        path: &str,
        old_string: &str,
        new_string: &str,
        multiple: bool,
    ) -> Result<String> {
        // ── No-op guard: reject edits where old and new are identical ──
        // Raw string comparison (no line-ending or whitespace normalization) so
        // we fail fast before touching the file.  Accepts the trade-off that a
        // whitespace-normalization edit (where old == new but the file bytes
        // actually differ in spacing) will be incorrectly rejected
        // — this edge case is rare, and the alternative (allowing literal
        // no-ops to pass through as "replaced 1 occurrence") is worse.  An edit
        // that only turns out to be a no-op once a match is found — because the
        // text written ends up byte-identical to the text it replaced — is
        // caught below, after the splice.
        if old_string == new_string {
            anyhow::bail!("old_string equals new_string — no change needed");
        }

        // ── 1. Path pre-validation ───────────────────────────────
        let resolved_target = super::path::resolve_write_target(ws.as_path(), path, false).await?;

        let use_ws_matching = is_ws_insensitive_extension(path);

        // ── 2. Size guard: reject oversized files before read_to_string ──
        match tokio::fs::metadata(&resolved_target).await {
            Ok(meta) => {
                super::check_size_within(&meta, super::MAX_FILE_SIZE_BYTES, "File too large")?;
            }
            Err(e) => anyhow::bail!("Cannot access file {path}: {e}"),
        }

        // ── 3. Read → match → splice → write ─────────────────────
        let content = match tokio::fs::read_to_string(&resolved_target).await {
            Ok(c) => c,
            Err(e) => {
                anyhow::bail!(
                    "io: cannot edit {path}: failed to read file: {e} — hint: verify the file exists and contains valid UTF-8 text"
                );
            }
        };

        let plan = plan_edits(
            path,
            &content,
            old_string,
            new_string,
            multiple,
            use_ws_matching,
        )?;
        let edits = &plan.edits;

        // Splice the planned replacements in, in file order — every byte
        // outside a planned span is copied through untouched.
        let mut new_content = String::with_capacity(content.len());
        let mut cursor = 0;
        for edit in edits {
            new_content.push_str(&content[cursor..edit.span.start]);
            new_content.push_str(&edit.replacement);
            cursor = edit.span.end;
        }
        new_content.push_str(&content[cursor..]);

        // An edit that only respelled its line endings is a no-op: the written
        // text took the file's endings and landed byte for byte on the text it
        // replaced, so the file would not change at all.
        if new_content == content && !plan.whitespace_fallback {
            anyhow::bail!(
                "old_string equals new_string once line endings are accounted for — no change needed"
            );
        }

        tokio::fs::write(&resolved_target, &new_content)
            .await
            .map_err(|e| anyhow::anyhow!("io: cannot edit {path}: failed to write file: {e} — hint: check disk space and file permissions"))?;
        update_search_index_after_write(ws, &resolved_target);
        Ok(format!(
            "Edited {path}: replaced {} occurrence{} ({} bytes)",
            edits.len(),
            if edits.len() == 1 { "" } else { "s" },
            new_content.len()
        ))
    }
}

// ── Match planning ──────────────────────────────────────────────────

/// One planned replacement: the byte span of a match in the file's raw content,
/// and the text to write there with its line endings already adapted to the
/// file.
struct Edit {
    span: std::ops::Range<usize>,
    replacement: String,
}

impl Edit {
    /// Plan a replacement of `span` by `new_string`. The written text takes the
    /// ending of the line the span starts on, else the file's prevailing ending,
    /// else — over a file with no ending at all — the ending it was spelled with.
    fn new(
        content: &str,
        span: std::ops::Range<usize>,
        new_string: &str,
        prevailing: Option<LineEnding>,
    ) -> Self {
        let replacement = match local_ending(content, span.start).or(prevailing) {
            Some(ending) => {
                let adapted = ending.apply(new_string);
                // A lone `\r` — content, not an ending — can sit right before the
                // span, the real endings being kept whole by the widening in
                // `resolve`. It already supplies the carriage return of the break
                // the written text opens with, so that break must not bring its
                // own and leave the file with two in a row.
                if adapted.starts_with("\r\n") && content[..span.start].ends_with('\r') {
                    adapted[1..].to_string()
                } else {
                    adapted
                }
            }
            None => new_string.to_string(),
        };
        Self { span, replacement }
    }
}

/// The planned replacements for an edit, and how they were matched.
struct Plan {
    edits: Vec<Edit>,
    /// Set when the match came from the whitespace-tolerant fallback. Its span
    /// can over-extend past the caller's text and already hold the text written
    /// into it, so an identical splice there is not a line-ending no-op and is
    /// left to the write exactly as it was before line endings were handled.
    whitespace_fallback: bool,
}

/// Plan the replacements for an edit against `content`.
///
/// Matching order:
/// 1. an exact match;
/// 2. a match that differs in line endings only (`\r\n` and `\n` are
///    interchangeable) — for every file type;
/// 3. the whitespace-tolerant fallback, which stays restricted to the code file
///    types that have it and to single mode.
///
/// In `multiple` mode every match found is replaced; otherwise a match count
/// other than one is an error — not found, or ambiguous when several places
/// match.
fn plan_edits(
    path: &str,
    content: &str,
    old_string: &str,
    new_string: &str,
    multiple: bool,
    use_ws_matching: bool,
) -> Result<Plan> {
    let exact = exact_spans(content, old_string);
    if !exact.is_empty() {
        return Ok(Plan {
            edits: resolve(exact, content, new_string, multiple)?,
            whitespace_fallback: false,
        });
    }

    let by_ending = line_ending_spans(content, old_string);
    if !by_ending.is_empty() {
        return Ok(Plan {
            edits: resolve(by_ending, content, new_string, multiple)?,
            whitespace_fallback: false,
        });
    }

    if !multiple && use_ws_matching {
        return match find_ws_insensitive(content, old_string)? {
            Some(span) => Ok(Plan {
                edits: resolve(vec![span], content, new_string, multiple)?,
                whitespace_fallback: true,
            }),
            None => Err(not_found_error(
                path,
                " (whitespace-insensitive matching tried)",
            )),
        };
    }

    if multiple {
        // Whitespace — as opposed to line endings — is the one difference
        // `multiple` cannot tolerate; say so when that is what is in the way.
        if use_ws_matching && find_ws_insensitive(content, old_string).is_ok_and(|s| s.is_some()) {
            return Err(not_found_error(
                path,
                " (whitespace differs; try without multiple=true)",
            ));
        }
        return Err(not_found_error(path, " (multiple=true mode)"));
    }
    Err(not_found_error(
        path,
        " (exact match required apart from line endings)",
    ))
}

/// Turn the matched spans into a plan: all of them in `multiple` mode, exactly
/// one otherwise.
fn resolve(
    spans: Vec<std::ops::Range<usize>>,
    content: &str,
    new_string: &str,
    multiple: bool,
) -> Result<Vec<Edit>> {
    let spans = if multiple {
        spans
    } else {
        match spans.as_slice() {
            [span] => vec![span.clone()],
            spans => anyhow::bail!(
                "old_string matches {} times; must match exactly once (or pass multiple=true to replace all)",
                spans.len()
            ),
        }
    };

    // A span that starts on the `\n` of a `\r\n` pair covers the line break the
    // caller's text spells, so the pair's `\r` belongs to it too — widening the
    // start over that `\r` is what keeps the pair from being split into a stray
    // carriage return. The previous span's end bounds the widening: a `\r` an
    // earlier span already replaced is not there to widen over.
    let mut previous_end = 0;
    let spans: Vec<_> = spans
        .into_iter()
        .map(|span| {
            let widened = span.start > previous_end
                && content[..span.start].ends_with('\r')
                && content[span.start..].starts_with('\n');
            previous_end = span.end;
            let start = if widened { span.start - 1 } else { span.start };
            start..span.end
        })
        .collect();

    // Scanned once for the whole plan rather than per match.
    let prevailing = prevailing_ending(content);
    Ok(spans
        .into_iter()
        .map(|span| Edit::new(content, span, new_string, prevailing))
        .collect())
}

/// Byte spans of every non-overlapping occurrence of `old_string`.
fn exact_spans(content: &str, old_string: &str) -> Vec<std::ops::Range<usize>> {
    content
        .match_indices(old_string)
        .map(|(start, matched)| start..start + matched.len())
        .collect()
}

/// Byte spans of `old_string` in `content` when line endings are their only
/// difference: both sides are reduced to `\n` endings before matching, so an
/// `old_string` that omits the carriage returns — as an ordinary reading of a
/// CRLF file shows it — matches the file's `\r\n` bytes, and one that is
/// spelled with carriage returns matches content that has none. A lone `\r` is
/// content on both sides and still has to match. Non-overlapping matches only,
/// exactly like an exact search.
fn line_ending_spans(content: &str, old_string: &str) -> Vec<std::ops::Range<usize>> {
    // Without a `\r\n` on either side reducing the endings changes nothing,
    // and the exact search has already failed on both strings verbatim.
    if !content.contains("\r\n") && !old_string.contains("\r\n") {
        return Vec::new();
    }
    let view = LfView::new(content);
    let needle = LfView::new(old_string).text;
    view.text
        .match_indices(&needle)
        .map(|(start, matched)| view.raw_span(start, start + matched.len()))
        .collect()
}

// ── Line endings ────────────────────────────────────────────────────

/// The line endings this tool recognises. A lone `\r` is content, not an
/// ending, so it is never recognised here and never rewritten.
#[derive(Clone, Copy, Debug)]
enum LineEnding {
    Lf,
    CrLf,
}

impl LineEnding {
    /// Rewrite every ending in `text` to this ending, leaving lone `\r` bytes
    /// (content) untouched.
    fn apply(self, text: &str) -> String {
        let lf_only = text.replace("\r\n", "\n");
        match self {
            Self::Lf => lf_only,
            Self::CrLf => lf_only.replace('\n', "\r\n"),
        }
    }
}

/// Whether the `\n` at `newline` closes a `\r\n` pair.
fn ending_of(content: &str, newline: usize) -> LineEnding {
    if newline > 0 && content.as_bytes()[newline - 1] == b'\r' {
        LineEnding::CrLf
    } else {
        LineEnding::Lf
    }
}

/// The ending of the line that `at` lies on: that of the file's first `\n` at or
/// after `at`. `None` over a final line the file left unterminated.
fn local_ending(content: &str, at: usize) -> Option<LineEnding> {
    content[at..]
        .find('\n')
        .map(|offset| ending_of(content, at + offset))
}

/// The ending most of the file's lines use; a tie goes to the first one seen.
fn prevailing_ending(content: &str) -> Option<LineEnding> {
    let bytes = content.as_bytes();
    let (mut crlf, mut lf) = (0usize, 0usize);
    let mut first = None;
    for (i, byte) in bytes.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }
        let ending = ending_of(content, i);
        first.get_or_insert(ending);
        match ending {
            LineEnding::CrLf => crlf += 1,
            LineEnding::Lf => lf += 1,
        }
    }
    match crlf.cmp(&lf) {
        std::cmp::Ordering::Greater => Some(LineEnding::CrLf),
        std::cmp::Ordering::Less => Some(LineEnding::Lf),
        std::cmp::Ordering::Equal => first,
    }
}

/// The file's content with `\r\n` endings reduced to `\n` (a lone `\r` is
/// content and survives), plus the offsets in that reduced view where the
/// carriage returns sat — enough to map a match back onto the raw content.
struct LfView {
    text: String,
    removed_cr: Vec<usize>,
}

impl LfView {
    fn new(content: &str) -> Self {
        let mut text = String::with_capacity(content.len());
        let mut removed_cr = Vec::new();
        let mut chars = content.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\r' && chars.peek() == Some(&'\n') {
                removed_cr.push(text.len());
                continue;
            }
            text.push(ch);
        }
        Self { text, removed_cr }
    }

    /// The raw span of a match found at `[start, end)` in [`LfView::text`]. A
    /// match that starts at the `\n` of a `\r\n`, or covers that `\n`, maps
    /// onto the whole ending, so a replacement made through this view never
    /// splits a `\r\n` pair.
    fn raw_span(&self, start: usize, end: usize) -> std::ops::Range<usize> {
        let removed_before = |at: usize| self.removed_cr.partition_point(|&cr| cr < at);
        start + removed_before(start)..end + removed_before(end)
    }
}

// ── Search index maintenance ────────────────────────────────────────

/// Synchronously update the search engine's file index after a write.
///
/// This mirrors what the background filesystem watcher does, but without
/// the latency — fsevents/inotify may take hundreds of milliseconds to
/// process the event. Without this update, an agent that immediately
/// searches after an edit would get stale results.
///
/// If the search engine hasn't been initialized for this workspace (no
/// searches have occurred), this is a no-op — the tool shouldn't fail
/// just because the search engine isn't ready.
///
/// If `handle_create_or_modify` returns `None` (index capacity
/// exhausted), we log a warning but don't fail — the background watcher
/// will eventually trigger a full rescan.
fn update_search_index_after_write(ws: &Workspace, file_path: &std::path::Path) {
    let Some(entry) = crate::search_engine::get_engine_by_name(&ws.name) else {
        return;
    };

    // parking_lot RwLock is non-poisoning — write() cannot fail. Held for
    // microseconds; the synchronous I/O inside handle_create_or_modify
    // (stat, binary detection read) is acceptable.
    if let Ok(mut guard) = entry.picker.write()
        && let Some(ref mut picker) = *guard
        && picker.handle_create_or_modify(file_path).is_none()
    {
        tracing::warn!(
            workspace = ws.name,
            path = %file_path.display(),
            "Search index capacity exhausted after file write — \
             background rescan needed"
        );
    }
}

// ── Whitespace-insensitive matching ─────────────────────────────────

/// File extensions for languages where whitespace between tokens has no
/// semantic meaning, making whitespace-insensitive editing safe.
const WS_INSENSITIVE_EXTENSIONS: &[&str] = &[
    "rs", "js", "jsx", "ts", "tsx", "c", "h", "cpp", "hpp", "cc", "cxx", "java", "kt", "kts", "go",
    "swift", "dart", "cs", "zig", "scala",
];

/// Check whether a file path has a recognized whitespace-insensitive extension.
fn is_ws_insensitive_extension(path: &str) -> bool {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    WS_INSENSITIVE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
}

/// A segment of the normalized string, tracking its byte range in both the
/// normalized and original versions.
#[derive(Debug, Clone)]
struct Segment {
    norm_range: std::ops::Range<usize>,
    orig_range: std::ops::Range<usize>,
}

/// Normalize a string by collapsing consecutive ASCII whitespace outside
/// string literals into single spaces. Returns the normalized string and a
/// list of segments that map normalized byte positions back to original ones.
fn normalize_ws(s: &str) -> (String, Vec<Segment>) {
    let mut normalized = String::new();
    let mut segments = Vec::new();
    let mut chars = s.char_indices().peekable();

    while let Some((i, ch)) = chars.next() {
        let norm_start = normalized.len();
        let orig_start = i;
        let mut orig_end = i;

        match ch {
            // String literals — copy verbatim (handles escape sequences)
            '"' | '\'' | '`' => {
                normalized.push(ch);
                while let Some((j, next_ch)) = chars.next() {
                    normalized.push(next_ch);
                    orig_end = j.saturating_add(next_ch.len_utf8());
                    if next_ch == '\\' {
                        if let Some((k, esc_ch)) = chars.next() {
                            normalized.push(esc_ch);
                            orig_end = k.saturating_add(esc_ch.len_utf8());
                        }
                    } else if next_ch == ch {
                        break;
                    }
                }
            }
            // Whitespace outside strings — collapse to single space
            _ if ch.is_ascii_whitespace() => {
                orig_end = i.saturating_add(ch.len_utf8());
                normalized.push(' ');
                while let Some(&(j, next_ch)) = chars.peek() {
                    if next_ch.is_ascii_whitespace() {
                        chars.next();
                        orig_end = j.saturating_add(next_ch.len_utf8());
                    } else {
                        break;
                    }
                }
            }
            // Regular content — copy as-is, stop before strings/whitespace
            _ => {
                normalized.push(ch);
                orig_end = i.saturating_add(ch.len_utf8());
                while let Some(&(j, next_ch)) = chars.peek() {
                    if next_ch.is_ascii_whitespace()
                        || next_ch == '"'
                        || next_ch == '\''
                        || next_ch == '`'
                    {
                        break;
                    }
                    normalized.push(next_ch);
                    chars.next();
                    orig_end = j.saturating_add(next_ch.len_utf8());
                }
            }
        }

        let norm_end = normalized.len();
        if norm_end > norm_start {
            segments.push(Segment {
                norm_range: norm_start..norm_end,
                orig_range: orig_start..orig_end,
            });
        }
    }

    (normalized, segments)
}

/// Find the segment containing a normalized byte position.
/// Returns an error if `pos` is out of range (which indicates a
/// normalization bug producing non-contiguous segments).
fn segment_at(pos: usize, segments: &[Segment]) -> Result<&Segment> {
    segments
        .iter()
        .find(|seg| pos < seg.norm_range.end && pos >= seg.norm_range.start)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "segment_at: position {pos} not found in {len} segments",
                len = segments.len()
            )
        })
}

/// Map a span of normalized byte positions back to original byte positions.
///
/// # Approximation in collapsed-whitespace segments
///
/// Whitespace normalization collapses runs of consecutive original whitespace
/// characters into a single space (e.g., 15 spaces → 1 space). This mapping is
/// **lossy** — given a normalized position inside such a segment, we cannot
/// determine exactly which original whitespace character it corresponds to.
///
/// When `norm_start` or `norm_end` falls inside a collapsed-whitespace segment,
/// this function maps the normalized position to the **boundary** of the
/// original whitespace run: the start of the run for `norm_start`, and the end
/// of the run for `norm_end`. This means the resulting original span may cover
/// more characters than strictly necessary.
///
/// This is an intentional choice: **replacing a bit too much is safer than
/// replacing too little**. Tools that consume the mapped span should expect
/// that they may receive a superset of the intended region when whitespace
/// normalization was involved.
fn map_norm_span(
    norm_start: usize,
    norm_end: usize,
    segments: &[Segment],
) -> Result<std::ops::Range<usize>> {
    let seg = segment_at(norm_start, segments)?;
    let orig_start = if seg.orig_range.len() == seg.norm_range.len() {
        seg.orig_range
            .start
            .saturating_add(norm_start.saturating_sub(seg.norm_range.start))
    } else {
        seg.orig_range.start
    };

    let end_seg = segment_at(norm_end.saturating_sub(1), segments)?;
    let orig_end = if end_seg.orig_range.len() == end_seg.norm_range.len() {
        end_seg
            .orig_range
            .start
            .saturating_add(norm_end.saturating_sub(end_seg.norm_range.start))
    } else {
        end_seg.orig_range.end
    };

    Ok(orig_start..orig_end)
}

/// Find `old_string` in `content` using whitespace-insensitive matching.
///
/// Consecutive whitespace outside string literals is collapsed to single
/// spaces before matching. Normalizes both strings once, then checks for
/// ambiguity (multiple normalized occurrences of `old_string`). Returns:
///
/// - `Ok(Some(range))` — a single unambiguous match found.
/// - `Ok(None)` — pattern not found after normalization.
/// - `Err(...)` — ambiguous (pattern matches multiple times after
///   normalization) or a normalization bug in `map_norm_span`.
fn find_ws_insensitive(content: &str, old_string: &str) -> Result<Option<std::ops::Range<usize>>> {
    if old_string.is_empty() || content.is_empty() {
        return Ok(None);
    }

    let (norm_content, segments) = normalize_ws(content);
    let (norm_old, _) = normalize_ws(old_string);

    let Some(norm_pos) = norm_content.find(&norm_old) else {
        return Ok(None);
    };
    let norm_end = norm_pos + norm_old.len();

    // Check for ambiguity: a second occurrence after the first char of norm_old.
    // Using char boundary instead of raw +1 avoids panicking on multi-byte chars.
    let first_char_len = norm_old.chars().next().unwrap().len_utf8();
    let search_start = norm_pos + first_char_len;
    if norm_content[search_start..].find(&norm_old).is_some() {
        anyhow::bail!(
            "old_string matches multiple times after whitespace normalization; \
             provide more surrounding context to disambiguate"
        );
    }

    Ok(Some(map_norm_span(norm_pos, norm_end, &segments)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use tempfile::TempDir;

    /// Helper: creates a temp workspace directory for an edit test, writes
    /// initial files, runs the test closure, and cleans up afterwards.
    /// The temp directory is backed by [`TempDir`] and is automatically
    /// cleaned up on drop — even if the closure panics.
    async fn with_temp_workspace<F, Fut>(files: &[(&str, &str)], test: F)
    where
        F: FnOnce(PathBuf) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let dir = TempDir::new().unwrap();
        for (filename, content) in files {
            let full_path = dir.path().join(filename);
            tokio::fs::create_dir_all(
                full_path
                    .parent()
                    .expect("TempDir-joined path always has a parent"),
            )
            .await
            .unwrap();
            tokio::fs::write(&full_path, content).await.unwrap();
        }
        let path = dir.path().to_path_buf();
        test(path).await;
        // TempDir::drop auto-cleans — panic-safe
    }

    // ── Extension check tests ────────────────────────────────────────

    #[test]
    fn ws_insensitive_extensions() {
        let cases: &[(&str, bool)] = &[
            ("main.rs", true),
            ("src/lib.rs", true),
            ("app.js", true),
            ("component.jsx", true),
            ("app.ts", true),
            ("component.tsx", true),
            ("main.c", true),
            ("main.h", true),
            ("main.cpp", true),
            ("main.hpp", true),
            ("main.cc", true),
            ("main.cxx", true),
            ("Main.java", true),
            ("Main.kt", true),
            ("Main.kts", true),
            ("main.go", true),
            ("main.swift", true),
            ("main.dart", true),
            ("Program.cs", true),
            ("main.zig", true),
            ("Main.scala", true),
            ("Main.rs", true),
            ("Main.RS", true),
            ("App.JS", true),
            ("Main.Rs", true),
            ("config.toml", false),
            ("config.json", false),
            ("config.yaml", false),
            ("config.yml", false),
            ("readme.md", false),
            ("Dockerfile", false),
            ("Makefile", false),
            ("main.py", false),
            ("main.rb", false),
            ("main.php", false),
            ("style.css", false),
            ("script.sh", false),
            ("docker-compose", false),
        ];
        for &(path, expected_match) in cases {
            assert_eq!(
                is_ws_insensitive_extension(path),
                expected_match,
                "case: {path}"
            );
        }
    }

    // ── Whitespace-insensitive matching tests ────────────────────────

    /// (content, old_string, expected_span) that should produce a match.
    const MATCH_CASES: &[(&str, &str, &str)] = &[
        ("let x = 5;", "let x = 5;", "let x = 5;"),
        ("let  x  =  5;", "let x = 5;", "let  x  =  5;"),
        ("let\tx\t=\t5;", "let x = 5;", "let\tx\t=\t5;"),
        ("let\nx\n=\n5;", "let x = 5;", "let\nx\n=\n5;"),
        ("\nlet x = 5;", "let x = 5;", "let x = 5;"),
        ("   \n\t   ", " ", "   \n\t   "),
        ("   \n\t   ", "  \n  ", "   \n\t   "),
        ("let x = \"\";", "let x = \"\";", "let x = \"\";"),
        (
            "let  msg  =  \"hello  world\";  let  y  =  5;",
            "let msg = \"hello  world\";",
            "let  msg  =  \"hello  world\";",
        ),
        (
            "function  hello()  {\n  return  42;\n}",
            "hello() {",
            "hello()  {",
        ),
        (
            "pub  fn  foo<T>(x:  T)  ->  T  where  T:  Debug  {  x  }",
            "pub fn foo<T>(x: T) -> T where T: Debug { x }",
            "pub  fn  foo<T>(x:  T)  ->  T  where  T:  Debug  {  x  }",
        ),
        (
            "fn  main()  {\n    let  x  =  5;\n    let  y  =  10;\n    x  +  y\n}",
            "fn main() {\n    let x = 5;\n    let y = 10;\n    x + y\n}",
            "fn  main()  {\n    let  x  =  5;\n    let  y  =  10;\n    x  +  y\n}",
        ),
        (
            "let  c:  char  =  'x';",
            "let c: char = 'x';",
            "let  c:  char  =  'x';",
        ),
        (
            "const  fn  =  (x)  =>  {  return  x  *  2;  };",
            "const fn = (x) => { return x * 2; };",
            "const  fn  =  (x)  =>  {  return  x  *  2;  };",
        ),
        (
            "let\t x\t= 5;\n\tlet\ty = 10;",
            "let x = 5;\nlet y = 10;",
            "let\t x\t= 5;\n\tlet\ty = 10;",
        ),
        (
            "  let x = helper(  arg1,  arg2  );",
            "helper( arg1, arg2 )",
            "helper(  arg1,  arg2  )",
        ),
        ("a\nb", "a b", "a\nb"),
        ("x y ", "x y ", "x y "),
        ("abc123", "abc123", "abc123"),
        (
            "let  name  =  \"café  créme\";",
            "let name = \"café  créme\";",
            "let  name  =  \"café  créme\";",
        ),
        (
            "let  x  =  \"a\"  +  \"b\";",
            "let x = \"a\" + \"b\";",
            "let  x  =  \"a\"  +  \"b\";",
        ),
        (
            "fn  foo() {\n\tlet  x  =  1;\n}",
            "fn foo() {\n\tlet x = 1;\n}",
            "fn  foo() {\n\tlet  x  =  1;\n}",
        ),
        (
            "const  x  =  \"hello  world\";",
            "const x = \"hello  world\";",
            "const  x  =  \"hello  world\";",
        ),
        ("fn  foo()  {}", "fn foo()", "fn  foo()"),
        ("fn  foo()  {}", "foo() {}", "foo()  {}"),
        ("x   +   y", "x + y", "x   +   y"),
        ("  a  +  b", "  a + b", "  a  +  b"),
        ("a  +  b  ", "a + b  ", "a  +  b  "),
        (
            "fn  main()  {}\nfn  other()  {}",
            "fn  main()  {}",
            "fn  main()  {}",
        ),
        (
            "fn  main()  {}\nfn  other()  {}",
            "fn other() {}",
            "fn  other()  {}",
        ),
        (
            "let x = \"hello \\\"world\\\"  foo\";",
            "let x = \"hello \\\"world\\\"  foo\";",
            "let x = \"hello \\\"world\\\"  foo\";",
        ),
        (
            "let s1 = 'simple', s2 = \"double\", s3 = `template`;",
            "let s1 = 'simple', s2 = \"double\", s3 = `template`;",
            "let s1 = 'simple', s2 = \"double\", s3 = `template`;",
        ),
        ("hello        world", "hello world", "hello        world"),
        ("let x = 5;\n", "let x = 5;", "let x = 5;"),
    ];

    /// (content, old_string) that should NOT match.
    const NOMATCH_CASES: &[(&str, &str)] = &[
        ("let x = \"hello    world\";", "hello  world"),
        ("let  msg  =  \"a    b\";", "let msg = \"a b\""),
        (
            "let x = \"hello \\\"world\\\"  foo\";",
            "let x = \"hello \\\"world\\\" foo\";",
        ),
        ("let x = 5;", "let y = 5;"),
        ("hello", ""),
        ("", "hello"),
        ("let msg = \"hello  world\";", "let msg = \"hello world\";"),
        ("let msg = \"hello\nworld\";", "let msg = \"hello world\";"),
        ("let x = 'hello  world';", "let x = 'hello world';"),
        ("let x = `hello  world`;", "let x = `hello world`;"),
    ];

    #[test]
    fn ws_insensitive_should_match() {
        for (content, old, expected) in MATCH_CASES {
            let m = find_ws_insensitive(content, old)
                .unwrap()
                .unwrap_or_else(|| panic!("Expected match: content={content:?} old={old:?}"));
            assert_eq!(&content[m], *expected, "content={content:?} old={old:?}");
        }
    }

    #[test]
    fn ws_insensitive_should_not_match() {
        for (content, old) in NOMATCH_CASES {
            assert!(
                find_ws_insensitive(content, old).unwrap().is_none(),
                "Expected no match: content={content:?} old={old:?}"
            );
        }
    }

    /// Cases for whitespace-insensitive matching ambiguity decisions.
    /// `expected_ambiguous = true` means the normalized old_string appears
    /// more than once (produces an error); `false` means it does not (ok).
    #[test]
    fn ws_match_is_ambiguous() {
        let cases: &[(&str, &str, &str, bool)] = &[
            // Repeated pattern — "a b" appears 3x in normalized "a b a b a b"
            ("repeated_pattern", "a  b  a  b  a  b", "a b", true),
            // Two lines that normalize to the same thing
            (
                "two_lines_normalize_same",
                "let  x  =  1;\nlet  x  =  1;",
                "let x = 1;",
                true,
            ),
            // Overlapping match in repeated tokens
            ("overlapping_repeated_tokens", "a a a", "a a", true),
            // Multi-byte first character (2-byte Latin ñ)
            ("multibyte_2byte_latin", "ñ b ñ b", "ñ b", true),
            // Multi-byte first character (3-byte CJK)
            ("multibyte_3byte_cjk", "字 符 字 符", "字 符", true),
            // Multi-byte first character (4-byte emoji)
            ("multibyte_4byte_emoji", "🚀 b 🚀 b", "🚀 b", true),
            // Single match
            ("single_match", "fn  foo()  {}", "fn foo() {}", false),
            // Two different functions — only one matches the pattern
            (
                "two_functions_one_matches",
                "fn  foo()  {}\nfn  bar()  {}",
                "fn bar() {}",
                false,
            ),
            // Old string appears only once after normalization
            (
                "appears_once_after_normalization",
                "let  x  =  5;\nlet  y  =  10;\n    x  +  y",
                "let x = 5;",
                false,
            ),
            // Empty old_string should not be ambiguous
            ("empty_old_string", "anything", "", false),
            // Not found at all
            ("not_found", "fn foo() {}", "fn bar() {}", false),
            // Multi-byte single match (2-byte Latin ñ)
            (
                "multibyte_2byte_latin_single",
                "fn  ñ  foo()  {}",
                "fn ñ foo()",
                false,
            ),
            // Multi-byte single match (3-byte CJK)
            (
                "multibyte_3byte_cjk_single",
                "let  字  =  1;",
                "let 字 = 1;",
                false,
            ),
            // Multi-byte single match (4-byte emoji)
            (
                "multibyte_4byte_emoji_single",
                "let  🚀  =  1;",
                "let 🚀 = 1;",
                false,
            ),
        ];
        for &(name, content, old, expected_ambiguous) in cases {
            let result = find_ws_insensitive(content, old);
            if expected_ambiguous {
                assert!(
                    result.is_err(),
                    "case: {name} — expected ambiguous error, content={content:?} old={old:?} got {result:?}"
                );
                let err = format!("{}", result.unwrap_err());
                assert!(
                    err.contains("multiple times after whitespace normalization"),
                    "case: {name} — error should mention ambiguity, got: {err}"
                );
            } else {
                assert!(
                    result.is_ok(),
                    "case: {name} — expected no ambiguity error, content={content:?} old={old:?} got {result:?}"
                );
            }
        }
    }

    #[test]
    fn segment_at_rejects_malformed_segments() {
        // Normal segments always cover the full span contiguously, but a
        // future normalize_ws bug could leave gaps. Verify that segment_at
        // returns an error for gap positions, not just positions beyond all segments.
        let segments = vec![
            Segment {
                norm_range: 0..5,
                orig_range: 0..5,
            },
            // Gap: positions 5-6 not covered (first segment covers 0..4,
            // second starts at 7). Positions 5-6 are in the gap.
            Segment {
                norm_range: 7..10,
                orig_range: 10..13,
            },
        ];
        // Position beyond all segments should produce an error
        assert!(segment_at(15, &segments).is_err());
        // Empty segments should produce an error
        assert!(segment_at(0, &[]).is_err());
        // Positions in the gap between segments should produce an error
        assert!(
            segment_at(6, &segments).is_err(),
            "position 6 is in the gap (0..5, 7..10)"
        );
        // Valid position in first segment should still work
        assert!(segment_at(3, &segments).is_ok());
        // Valid position at segment boundary is ok
        assert!(segment_at(7, &segments).is_ok());
    }

    #[tokio::test]
    async fn file_edit_multiple_replacements() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("test.txt"), "a b a c a d")
            .await
            .unwrap();

        // single match without multiple flag still works
        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "test.txt", "old_string": "b", "new_string": "x"}),
            )
            .await;
        assert!(result.is_ok(), "edit should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("test.txt"))
                .await
                .unwrap(),
            "a x a c a d"
        );

        // multiple=true replaces all occurrences
        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "test.txt", "old_string": "a", "new_string": "y", "multiple": true}),
            )
            .await;
        assert!(result.is_ok(), "multiple edit should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("replaced 3 occurrences"));
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("test.txt"))
                .await
                .unwrap(),
            "y x y c y d"
        );

        // multiple=true with no matches still fails
        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "test.txt", "old_string": "z", "new_string": "w", "multiple": true}),
            )
            .await;
        assert!(
            result.is_err(),
            "edit with no matches should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("not-found: cannot edit test.txt: old_string not found in file"),
            "canonical not-found shape: {err}"
        );

        // multiple=true with single match works too
        tokio::fs::write(dir.path().join("test.txt"), "only one")
            .await
            .unwrap();
        let result = EditTool
            .execute(&Workspace::from_path(dir.path()), json!({"path": "test.txt", "old_string": "one", "new_string": "two", "multiple": true}))
            .await;
        assert!(
            result.is_ok(),
            "single match with multiple flag: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("test.txt"))
                .await
                .unwrap(),
            "only two"
        );
    }

    #[tokio::test]
    async fn file_edit_match_operations() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("test.txt"), "hello world")
            .await
            .unwrap();

        // replace single match
        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "test.txt", "old_string": "hello", "new_string": "goodbye"}),
            )
            .await;
        assert!(result.is_ok(), "edit should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("test.txt"))
                .await
                .unwrap(),
            "goodbye world"
        );
        // not found
        let result = EditTool.execute(&Workspace::from_path(dir.path()), json!({"path": "test.txt", "old_string": "nonexistent", "new_string": "replacement"})).await;
        assert!(
            result.is_err(),
            "edit with nonexistent string should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("not-found: cannot edit test.txt: old_string not found in file"),
            "canonical not-found shape: {err}"
        );
        // multiple matches rejected
        tokio::fs::write(dir.path().join("test.txt"), "aaa bbb aaa")
            .await
            .unwrap();
        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "test.txt", "old_string": "aaa", "new_string": "ccc"}),
            )
            .await;
        assert!(result.is_err(), "multiple matches should fail: {result:?}");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("matches 2 times"));
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("test.txt"))
                .await
                .unwrap(),
            "aaa bbb aaa"
        );
    }

    #[tokio::test]
    async fn file_edit_delete_via_empty_new_string() {
        with_temp_workspace(&[("test.txt", "keep remove keep")], |dir| async move {
            let result = EditTool
                .execute(
                    &test_ws(&dir),
                    json!({"path": "test.txt", "old_string": " remove", "new_string": ""}),
                )
                .await;
            assert!(
                result.is_ok(),
                "delete edit should succeed: {:?}",
                result.as_ref().unwrap_err()
            );
            let content = tokio::fs::read_to_string(dir.join("test.txt"))
                .await
                .unwrap();
            assert_eq!(content, "keep keep");
        })
        .await;
    }

    #[tokio::test]
    async fn edit_write_mode_treats_omitted_and_empty_old_string_alike() {
        struct Case {
            name: &'static str,
            old_string: Option<&'static str>,
        }
        for case in [
            Case {
                name: "omitted old_string",
                old_string: None,
            },
            Case {
                name: "empty old_string",
                old_string: Some(""),
            },
        ] {
            let dir = TempDir::new().unwrap();

            let mut args = json!({"path": "out.txt", "new_string": "written!"});
            if let Some(old) = case.old_string {
                args["old_string"] = json!(old);
            }

            let result = EditTool.execute(&test_ws(dir.path()), args).await;
            assert!(result.is_ok(), "{} should succeed: {result:?}", case.name);
            assert!(
                result.unwrap().contains("8 bytes"),
                "{} should report written size",
                case.name
            );

            let content = tokio::fs::read_to_string(dir.path().join("out.txt"))
                .await
                .unwrap();
            assert_eq!(content, "written!", "{}", case.name);
        }
    }

    #[tokio::test]
    async fn edit_write_mode_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "a/b/c/deep.txt", "new_string": "deep"}),
            )
            .await;
        assert!(result.is_ok(), "write with parent dirs: {result:?}");
        let content = tokio::fs::read_to_string(dir.path().join("a/b/c/deep.txt"))
            .await
            .unwrap();
        assert_eq!(content, "deep");
    }

    #[tokio::test]
    async fn file_edit_blocks_dangerous_paths() {
        with_temp_workspace(&[], |dir| async move {
            let result = EditTool
                .execute(
                    &test_ws(&dir),
                    json!({"path": "../../etc/passwd", "old_string": "root", "new_string": "x"}),
                )
                .await;
            assert!(result.is_err(), "traversal should be blocked: {result:?}");
            let err = format!("{}", result.unwrap_err());
            assert!(
                err.contains("forbidden: cannot write to ../../etc/passwd"),
                "canonical forbidden shape: {err}"
            );
            let result = EditTool
                .execute(
                    &test_ws(&dir),
                    json!({"path": "/etc/passwd", "old_string": "root", "new_string": "x"}),
                )
                .await;
            assert!(
                result.is_err(),
                "absolute path should be blocked: {result:?}"
            );
            let err = format!("{}", result.unwrap_err());
            assert!(
                err.contains("forbidden: cannot write to /etc/passwd"),
                "canonical forbidden shape: {err}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn file_edit_normalizes_relative_path() {
        with_temp_workspace(
            &[("workspace/nested/target.txt", "hello world")],
            |root| async move {
                let workspace = root.join("workspace");
                let result = EditTool
                    .execute(
                        &test_ws(&workspace), json!({"path": "nested/target.txt", "old_string": "world", "new_string": "mahbot"}),
                    )
                    .await;

                assert!(result.is_ok(), "relative path edit: {result:?}");
                let content = tokio::fs::read_to_string(workspace.join("nested/target.txt")).await.unwrap();
                assert_eq!(content, "hello mahbot");
            },
        )
        .await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_edit_blocks_symlink_target_file() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        let outside = root.path().join("outside");

        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&outside).await.unwrap();

        tokio::fs::write(outside.join("target.txt"), "original")
            .await
            .unwrap();
        symlink(outside.join("target.txt"), workspace.join("linked.txt")).unwrap();

        let result = EditTool
            .execute(
                &test_ws(&workspace),
                json!({
                    "path": "linked.txt",
                    "old_string": "original",
                    "new_string": "hacked"
                }),
            )
            .await;

        assert!(
            result.is_err(),
            "editing through symlink must be blocked: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("symlink"),
            "error should mention symlink, got: {err}"
        );

        let content = tokio::fs::read_to_string(outside.join("target.txt"))
            .await
            .unwrap();
        assert_eq!(content, "original", "original file must not be modified");
    }

    #[tokio::test]
    async fn file_edit_nonexistent_file() {
        with_temp_workspace(&[], |dir| async move {
            let result = EditTool
                .execute(
                    &test_ws(&dir),
                    json!({"path": "missing.txt", "old_string": "a", "new_string": "b"}),
                )
                .await;
            assert!(result.is_err(), "edit of nonexistent file: {result:?}");
            let err = format!("{}", result.unwrap_err());
            assert!(err.contains("Cannot access file"));
        })
        .await;
    }

    #[tokio::test]
    async fn file_edit_absolute_path_in_workspace() {
        with_temp_workspace(
            &[("target.txt", "old content")],
            |dir| async move {
                // Canonicalize so the workspace dir matches resolved paths on macOS (/private/var/…)
                let dir = tokio::fs::canonicalize(&dir).await.unwrap();
                let abs_path = dir.join("target.txt");
                let result = EditTool
                    .execute(
                        &test_ws(&dir), json!({"path": abs_path.to_string_lossy().to_string(), "old_string": "old content", "new_string": "new content"}),
                    )
                    .await;
                assert!(result.is_ok(), "editing via absolute workspace path should succeed, error: {:?}", result.as_ref().unwrap_err());
                let content = tokio::fs::read_to_string(dir.join("target.txt")).await.unwrap();
                assert_eq!(content, "new content");
            },
        )
        .await;
    }

    // ── WS-insensitive ambiguity tests ────────────────────────────

    #[tokio::test]
    async fn ws_ambiguous_rejects_multiple_matches() {
        // Two assignments that normalize to the same string — the WS fallback
        // must detect ambiguity and refuse to pick one arbitrarily.
        with_temp_workspace(
            &[("lib.rs", "let  x  =  1;\nlet  x  =  1;\nlet  y  =  2;\n")],
            |dir| async move {
                let result = EditTool
                    .execute(
                        &test_ws(&dir),
                        json!({
                            "path": "lib.rs",
                            "old_string": "let x = 1;",  // exact match not found, WS fallback
                            "new_string": "let x = 42;"
                        }),
                    )
                    .await;
                assert!(result.is_err(), "WS-ambiguous edit should fail: {result:?}");
                let err = format!("{}", result.unwrap_err());
                assert!(
                    err.contains("multiple times after whitespace normalization"),
                    "Error should mention whitespace normalization ambiguity, got: {err}"
                );
                assert!(
                    err.contains("surrounding context"),
                    "Error should suggest adding surrounding context, got: {err}"
                );
                // File must not be modified
                let content = tokio::fs::read_to_string(dir.join("lib.rs")).await.unwrap();
                assert_eq!(content, "let  x  =  1;\nlet  x  =  1;\nlet  y  =  2;\n");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn ws_unambiguous_single_match_still_works() {
        // Regression: a .rs file with a single WS-only match should still succeed.
        with_temp_workspace(
            &[("lib.rs", "let  x  =  1;\nlet  y  =  2;\n")],
            |dir| async move {
                let result = EditTool
                    .execute(
                        &test_ws(&dir),
                        json!({
                            "path": "lib.rs",
                            "old_string": "let x = 1;",  // exact match not found, WS fallback finds unique match
                            "new_string": "let x = 42;"
                        }),
                    )
                    .await;
                assert!(result.is_ok(), "Single WS match should succeed: {result:?}");
                let content = tokio::fs::read_to_string(dir.join("lib.rs")).await.unwrap();
                assert_eq!(content, "let x = 42;\nlet  y  =  2;\n");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn ws_ambiguous_not_triggered_for_non_code_files() {
        // Non-code files use exact matching only, so the WS ambiguity check
        // is never reached. Ensure a .txt file with multiple normalized matches
        // gets the standard exact-match error, not the WS-ambiguity error.
        with_temp_workspace(&[("readme.txt", "a  b  a  b")], |dir| async move {
            let result = EditTool
                .execute(
                    &test_ws(&dir),
                    json!({
                        "path": "readme.txt",
                        "old_string": "a b",
                        "new_string": "x"
                    }),
                )
                .await;
            assert!(
                result.is_err(),
                "Exact match for .txt should fail (no matches): {result:?}"
            );
            let err = format!("{}", result.unwrap_err());
            assert!(
                err.contains(
                    "not-found: cannot edit readme.txt: old_string not found in file \
                     (exact match required apart from line endings)"
                ),
                ".txt should use exact matching only, got: {err}"
            );
        })
        .await;
    }

    // ── Line-ending handling ──────────────────────────────────────

    /// The raw bytes of `name` in `dir`. Every reading surface hides a CRLF's
    /// carriage return, so a line-ending claim can only be checked on bytes.
    async fn file_bytes(dir: &Path, name: &str) -> Vec<u8> {
        tokio::fs::read(dir.join(name))
            .await
            .expect("read file bytes")
    }

    /// The text an agent sees for `path`: the `read` tool's content mode with
    /// its line numbers and summary line stripped — carriage returns hidden.
    async fn read_view(ws: &Workspace, path: &str) -> String {
        let out = crate::tools::ReadTool::general()
            .execute(ws, json!({"path": path}))
            .await
            .expect("read should succeed");
        out.lines()
            .skip(1) // the "[N lines total]" summary
            .map(|line| line.split_once(": ").expect("numbered line").1)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Writing the initial file and reading it back hides the carriage return of
    /// every CRLF, so an `old_string` built from a reading is spelled with LF.
    /// Applying such an edit must land for any file type, and must leave the
    /// file on the ending it had.
    #[tokio::test]
    async fn read_then_edit_round_trip_for_every_file_type_and_ending() {
        for name in [
            "notes.md",
            "notes.txt",
            "script.py",
            "config.json",
            "lib.rs",
        ] {
            for ending in ["\n", "\r\n"] {
                let dir = TempDir::new().unwrap();
                let initial = format!("alpha{ending}beta{ending}gamma{ending}");
                tokio::fs::write(dir.path().join(name), &initial)
                    .await
                    .unwrap();
                let ws = test_ws(dir.path());

                assert_eq!(
                    read_view(&ws, name).await,
                    "alpha\nbeta\ngamma",
                    "{name} ({ending:?}) is read with LF"
                );

                let result = EditTool
                    .execute(
                        &ws,
                        json!({
                            "path": name,
                            "old_string": "beta\ngamma", // built from the reading
                            "new_string": "beta\nGAMMA\nadded",
                        }),
                    )
                    .await;
                assert!(
                    result.is_ok(),
                    "{name} ({ending:?}): edit built from a read must land: {result:?}"
                );
                assert_eq!(
                    file_bytes(dir.path(), name).await,
                    format!("alpha{ending}beta{ending}GAMMA{ending}added{ending}").as_bytes(),
                    "{name} ({ending:?}): the change landed, the untouched line kept \
                     its bytes and the file kept its endings"
                );
            }
        }
    }

    /// `multiple` has no whitespace-tolerant fallback, but line endings are not
    /// whitespace: an `old_string` read from a CRLF file matches every
    /// occurrence, and each written copy takes the file's ending.
    #[tokio::test]
    async fn crlf_multiple_mode_replaces_every_match() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(
            dir.path().join("data.txt"),
            b"item = 1\r\nkeep = 0\r\nitem = 1\r\nkeep = 0\r\n",
        )
        .await
        .unwrap();
        let ws = test_ws(dir.path());
        assert_eq!(
            read_view(&ws, "data.txt").await,
            "item = 1\nkeep = 0\nitem = 1\nkeep = 0"
        );

        let result = EditTool
            .execute(
                &ws,
                json!({
                    "path": "data.txt",
                    "old_string": "item = 1\nkeep = 0", // built from the reading
                    "new_string": "item = 2\nkeep = 1",
                    "multiple": true,
                }),
            )
            .await;
        assert!(
            result.is_ok(),
            "multiple mode must work on a CRLF file: {result:?}"
        );
        assert!(result.unwrap().contains("replaced 2 occurrences"));
        assert_eq!(
            file_bytes(dir.path(), "data.txt").await,
            b"item = 2\r\nkeep = 1\r\nitem = 2\r\nkeep = 1\r\n".as_slice()
        );
    }

    /// Surfaces that keep the carriage returns (a zoomed symbol, a redirected
    /// command) hand the agent a CR-spelled `old_string`. It keeps matching,
    /// and the text the edit writes still takes the file's ending.
    #[tokio::test]
    async fn old_string_spelled_with_carriage_returns_still_matches() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("zoom.txt"), b"one\r\ntwo\r\nthree\r\n")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({
                    "path": "zoom.txt",
                    "old_string": "one\r\ntwo",     // spelled with CRLF
                    "new_string": "one\ntwo\nhalf", // spelled with LF
                }),
            )
            .await;
        assert!(
            result.is_ok(),
            "CR-spelled old_string must land: {result:?}"
        );
        assert_eq!(
            file_bytes(dir.path(), "zoom.txt").await,
            b"one\r\ntwo\r\nhalf\r\nthree\r\n".as_slice(),
            "the written line took the file's CRLF, not the LF it was spelled with"
        );
    }

    /// An LF file behaves exactly as it did before line endings were handled at
    /// all, and a replacement spelled with CRLF puts no carriage return into it.
    #[tokio::test]
    async fn unix_endings_are_preserved_and_no_foreign_ending_is_introduced() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("unix.txt"), b"a\nb\nc\n")
            .await
            .unwrap();
        let ws = test_ws(dir.path());

        let result = EditTool
            .execute(
                &ws,
                json!({"path": "unix.txt", "old_string": "b\nc", "new_string": "b\nC"}),
            )
            .await;
        assert!(result.is_ok(), "multi-line edit: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "unix.txt").await,
            b"a\nb\nC\n".as_slice()
        );

        let result = EditTool
            .execute(
                &ws,
                json!({"path": "unix.txt", "old_string": "a\nb", "new_string": "a\r\nB"}),
            )
            .await;
        assert!(result.is_ok(), "CRLF-spelled replacement: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "unix.txt").await,
            b"a\nB\nC\n".as_slice(),
            "an LF file stays LF"
        );
    }

    /// A file whose endings are already mixed keeps that mixture everywhere the
    /// edit did not reach; the text written into a line takes that line's ending.
    #[tokio::test]
    async fn mixed_endings_survive_outside_the_edit() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("mixed.txt"), b"head\nmid\r\ntail\n")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "mixed.txt", "old_string": "mid", "new_string": "mid\nmiddle"}),
            )
            .await;
        assert!(result.is_ok(), "edit in a mixed file: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "mixed.txt").await,
            b"head\nmid\r\nmiddle\r\ntail\n".as_slice(),
            "the untouched LF line and CRLF pair kept their bytes; the written line \
             took the CRLF of the line it was written into"
        );
    }

    /// A `\r` that is not followed by `\n` is content, not an ending: it is
    /// never converted, and it is not matched across.
    #[tokio::test]
    async fn lone_carriage_return_is_content() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("classic.txt"), b"a\rb\r\nc\r\n")
            .await
            .unwrap();
        let ws = test_ws(dir.path());

        let result = EditTool
            .execute(
                &ws,
                json!({"path": "classic.txt", "old_string": "a\rb", "new_string": "a\rB"}),
            )
            .await;
        assert!(result.is_ok(), "lone CR in old_string: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "classic.txt").await,
            b"a\rB\r\nc\r\n".as_slice()
        );

        // A lone `\r` inside the replacement survives; only real endings are
        // rewritten to the file's ending.
        let result = EditTool
            .execute(
                &ws,
                json!({"path": "classic.txt", "old_string": "c", "new_string": "c\rd\ne"}),
            )
            .await;
        assert!(result.is_ok(), "lone CR in new_string: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "classic.txt").await,
            b"a\rB\r\nc\rd\r\ne\r\n".as_slice()
        );

        // An LF-spelled pattern does not match across the lone CR.
        let result = EditTool
            .execute(
                &ws,
                json!({"path": "classic.txt", "old_string": "a\nb", "new_string": "x"}),
            )
            .await;
        assert!(
            result.is_err(),
            "a lone CR is content, so it cannot be matched as an ending: {result:?}"
        );
    }

    /// An edit that is a no-op once the endings are accounted for is rejected
    /// as a no-op and leaves the file untouched — in either spelling.
    #[tokio::test]
    async fn edit_that_only_respells_endings_is_a_no_op() {
        // The file's own CRLF, asked for with a read-spelled old_string and with
        // a CR-spelled new_string alike: both edits are already satisfied.
        for (name, old, new) in [
            ("lf_spelling.txt", "a\r\nb", "a\nb"),
            ("crlf_spelling.txt", "a\nb", "a\r\nb"),
        ] {
            let dir = TempDir::new().unwrap();
            let before = b"a\r\nb\r\n".as_slice();
            tokio::fs::write(dir.path().join(name), before)
                .await
                .unwrap();

            let result = EditTool
                .execute(
                    &test_ws(dir.path()),
                    json!({"path": name, "old_string": old, "new_string": new}),
                )
                .await;
            let err = format!(
                "{}",
                result.expect_err("respelling the file's endings is a no-op")
            );
            assert!(err.contains("no change needed"), "{name}: {err}");
            assert_eq!(file_bytes(dir.path(), name).await, before, "{name}");
        }
    }

    /// Several places matching after the endings are ignored is an error, for
    /// code and non-code files alike — the first match is never picked. The
    /// pattern is multi-line and read-spelled, so it has no exact match at all.
    #[tokio::test]
    async fn line_ending_match_reports_ambiguity() {
        for name in ["dup.txt", "dup.rs"] {
            let dir = TempDir::new().unwrap();
            tokio::fs::write(dir.path().join(name), b"x = 1\r\nkeep\r\nx = 1\r\nkeep\r\n")
                .await
                .unwrap();

            let result = EditTool
                .execute(
                    &test_ws(dir.path()),
                    json!({
                        "path": name,
                        "old_string": "x = 1\nkeep", // built from the reading
                        "new_string": "x = 2\nkeep",
                    }),
                )
                .await;
            let err = format!("{}", result.expect_err("two matches are ambiguous"));
            assert!(err.contains("matches 2 times"), "{name}: {err}");
            assert!(err.contains("must match exactly once"), "{name}: {err}");
            assert_eq!(
                file_bytes(dir.path(), name).await,
                b"x = 1\r\nkeep\r\nx = 1\r\nkeep\r\n".as_slice()
            );
        }
    }

    /// The line-ending tolerance is symmetric: a CR-spelled `old_string` also
    /// matches content that has no carriage returns, and the file keeps its LF.
    #[tokio::test]
    async fn cr_spelled_old_string_matches_content_without_carriage_returns() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("unix.txt"), b"a\nb\n")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "unix.txt", "old_string": "a\r\nb", "new_string": "a\nB"}),
            )
            .await;
        assert!(result.is_ok(), "CR-spelled old on an LF file: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "unix.txt").await,
            b"a\nB\n".as_slice()
        );
    }

    /// The written text takes the ending of the line it is written into; over a
    /// final line the file left unterminated it takes the file's prevailing
    /// ending instead.
    #[tokio::test]
    async fn written_text_takes_the_files_ending() {
        // A whole line of a CRLF file.
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("mid.txt"), b"one two\r\nthree\r\n")
            .await
            .unwrap();
        let ws = test_ws(dir.path());

        let result = EditTool
            .execute(
                &ws,
                json!({"path": "mid.txt", "old_string": "ne", "new_string": "NE\nline"}),
            )
            .await;
        assert!(result.is_ok(), "mid-word edit: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "mid.txt").await,
            b"oNE\r\nline two\r\nthree\r\n".as_slice()
        );

        // A last line with no ending of its own.
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("tail.txt"), b"a\r\nb")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "tail.txt", "old_string": "b", "new_string": "b\nx"}),
            )
            .await;
        assert!(result.is_ok(), "unterminated last line: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "tail.txt").await,
            b"a\r\nb\r\nx".as_slice()
        );
    }

    /// Over a file with no ending at all the written text is left as it is
    /// spelled.
    #[tokio::test]
    async fn a_file_with_no_ending_gets_the_text_as_written() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("oneline"), b"abc")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "oneline", "old_string": "b", "new_string": "x\ny\r\nz"}),
            )
            .await;
        assert!(result.is_ok(), "single-line file: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "oneline").await,
            b"ax\ny\r\nzc".as_slice()
        );
    }

    /// A pattern that starts on the `\n` of a `\r\n` pair leaves that pair's
    /// `\r` outside the match. The written text must not then bring its own
    /// carriage return — which would strand the pair's `\r` as content — while
    /// its other line breaks still take the file's ending.
    #[tokio::test]
    async fn pattern_starting_on_the_lf_of_a_crlf_pair_stays_on_crlf() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("pair.txt"), b"alpha\r\nbeta\r\ngamma\r\n")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({
                    "path": "pair.txt",
                    "old_string": "\nbeta",
                    "new_string": "\nBETA\ninserted",
                }),
            )
            .await;
        assert!(result.is_ok(), "pattern starting on a bare LF: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "pair.txt").await,
            b"alpha\r\nBETA\r\ninserted\r\ngamma\r\n".as_slice()
        );
    }

    /// A pattern that is nothing but a line break removes the whole ending —
    /// rather than leaving the carriage return of a `\r\n` pair behind as
    /// content — so a CRLF file and its LF twin give the same visible text.
    #[tokio::test]
    async fn deleting_a_line_break_removes_the_whole_ending() {
        for (name, ending) in [("unix.txt", "\n"), ("windows.txt", "\r\n")] {
            let dir = TempDir::new().unwrap();
            tokio::fs::write(dir.path().join(name), format!("a{ending}b"))
                .await
                .unwrap();

            let result = EditTool
                .execute(
                    &test_ws(dir.path()),
                    json!({"path": name, "old_string": "\n", "new_string": " "}),
                )
                .await;
            assert!(result.is_ok(), "{name}: deleting a line break: {result:?}");
            assert_eq!(
                file_bytes(dir.path(), name).await,
                b"a b".as_slice(),
                "{name}: the ending went with the line break it spelled"
            );
        }
    }

    /// A whitespace-driven no-op is still written as a no-op: the guard for an
    /// edit that only respelled its line endings must not catch it.
    #[tokio::test]
    async fn whitespace_driven_no_op_still_succeeds() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("lib.rs"), b"a  b\n")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "lib.rs", "old_string": "a b", "new_string": "a  b"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "the whitespace fallback may replace a span with its own text: {result:?}"
        );
        assert_eq!(file_bytes(dir.path(), "lib.rs").await, b"a  b\n".as_slice());
    }

    /// The line-ending tolerance does not hand whitespace tolerance to file
    /// types that never had it.
    #[tokio::test]
    async fn line_ending_tolerance_is_not_whitespace_tolerance() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("readme.txt"), b"a  b\r\nc\r\n")
            .await
            .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({"path": "readme.txt", "old_string": "a b", "new_string": "a+b"}),
            )
            .await;
        assert!(
            result.is_err(),
            "a .txt file still gets no whitespace tolerance: {result:?}"
        );
    }

    /// In a code file the whitespace fallback still fires for a spacing
    /// difference, and the text it writes takes the file's ending.
    #[tokio::test]
    async fn whitespace_fallback_in_a_crlf_code_file_keeps_the_ending() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(
            dir.path().join("lib.rs"),
            b"let  x  =  1;\r\nlet  y  =  2;\r\n",
        )
        .await
        .unwrap();

        let result = EditTool
            .execute(
                &test_ws(dir.path()),
                json!({
                    "path": "lib.rs",
                    "old_string": "let x = 1;\nlet y = 2;", // spacing and endings differ
                    "new_string": "let x = 42;\nlet y = 2;",
                }),
            )
            .await;
        assert!(result.is_ok(), "whitespace fallback: {result:?}");
        assert_eq!(
            file_bytes(dir.path(), "lib.rs").await,
            b"let x = 42;\r\nlet y = 2;\r\n".as_slice()
        );
    }
}
