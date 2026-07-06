use crate::{Tool, ToolOutputPhase, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

/// Edit a file by replacing an exact string match with new content.
///
/// Two modes:
/// - **Write mode**: when `old_string` is omitted or empty, creates a new file with
///   `new_string` as the content (including parent directories). Refuses to
///   overwrite an existing file — use edit mode for changes.
/// - **Edit mode**: when `old_string` is provided, performs precise replacement
///   within an existing file. Matching is semi-insensitive to whitespace for
///   code files (.rs, .js, .ts, .c, .cpp, .go, etc.): extra/missing spaces
///   outside string literals are tolerated. By default the `old_string` must
///   appear exactly once (zero matches = not found, multiple = ambiguous).
///   `new_string` may be empty to delete the matched text.
pub struct EditTool;

#[allow(clippy::too_many_lines)]
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
                    "description": "Path to the file. Relative paths resolve from workspace; absolute paths are validated against the workspace boundary."
                },
                "old_string": {
                    "type": "string",
                    "description": "If omitted or empty: creates a new file with `new_string` (refuses if file exists). If provided and non-empty: this exact text is replaced by `new_string` (semi-insensitive to whitespace in code files, must appear exactly once unless multiple is true)."
                },
                "new_string": {
                    "type": "string",
                    "description": "When old_string is omitted or empty: the content to write to the new file. When old_string is provided and non-empty: the replacement text (may be empty to delete the matched text). Must differ from old_string — identical old and new strings are rejected as a no-op."
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
        // ── 1. Extract parameters ──────────────────────────────────
        let path = super::require_path_arg(&args)?;

        let old_string = super::get_opt_str(&args, "old_string");

        let new_string = super::get_str(&args, "new_string")?;

        // ── Write/Edit mode ──────────────────────────────────────
        // empty old_string deliberately triggers write mode (no way to match/replace empty string in a file)
        match old_string {
            None | Some("") => {
                let resolved_target =
                    super::path::resolve_write_target(ws.as_path(), &path, true).await?;

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
                    .map_err(|e| anyhow::anyhow!("Failed to write file: {e}"))?;
                update_search_index_after_write(ws, &resolved_target);
                Ok(format!("Written {} bytes to {path}", new_string.len()))
            }
            Some(old_string) => {
                // ── Edit mode ─────────────────────────────────────
                let multiple = super::get_bool(&args, "multiple", false);

                // ── No-op guard: reject edits where old and new are identical ──
                // Raw string comparison (no whitespace normalization) so we fail
                // fast before touching the file.  Accepts the trade-off that a
                // whitespace-normalization edit (where old == new but the file
                // bytes actually differ in spacing) will be incorrectly rejected
                // — this edge case is rare, and the alternative (allowing literal
                // no-ops to pass through as "replaced 1 occurrence") is worse.
                if old_string == new_string {
                    anyhow::bail!("old_string equals new_string — no change needed");
                }

                // ── 2. Path pre-validation ───────────────────────────────
                let resolved_target =
                    super::path::resolve_write_target(ws.as_path(), &path, false).await?;

                let use_ws_matching = is_ws_insensitive_extension(&path);

                // ── 3. Size guard: reject oversized files before read_to_string ──
                match tokio::fs::metadata(&resolved_target).await {
                    Ok(meta) => {
                        super::check_file_size(&meta)?;
                    }
                    Err(e) => anyhow::bail!("Cannot access file {path}: {e}"),
                }

                // ── 4. Read → match → replace → write ───────────────────
                let content = match tokio::fs::read_to_string(&resolved_target).await {
                    Ok(c) => c,
                    Err(e) => {
                        anyhow::bail!("Failed to read file: {e}");
                    }
                };

                let new_content;
                let replaced_count;

                let exact_count = content.matches(old_string).count();

                if multiple {
                    // Multiple mode: use exact match (whitespace-insensitive multi-replace is a future concern)

                    if exact_count == 0 {
                        // Try a whitespace-insensitive match for a better error message
                        if use_ws_matching
                            && find_ws_insensitive(&content, old_string).is_ok_and(|r| r.is_some())
                        {
                            anyhow::bail!(
                                "old_string not found exactly (whitespace differs); try without multiple=true"
                            );
                        }
                        anyhow::bail!("old_string not found in file (multiple=true mode)");
                    }

                    new_content = content.replace(old_string, new_string);
                    replaced_count = exact_count;
                } else {
                    // Single mode: try exact match first, fall back to whitespace-insensitive

                    match exact_count {
                        1 => {
                            new_content = content.replacen(old_string, new_string, 1);
                            replaced_count = 1;
                        }
                        0 if use_ws_matching => {
                            // No exact match — try whitespace-insensitive matching
                            match find_ws_insensitive(&content, old_string) {
                                Ok(Some(ws_match)) => {
                                    new_content = format!(
                                        "{}{}{}",
                                        &content[..ws_match.start],
                                        new_string,
                                        &content[ws_match.end..]
                                    );
                                    replaced_count = 1;
                                }
                                Ok(None) => {
                                    anyhow::bail!(
                                        "old_string not found in file (whitespace-insensitive matching tried)"
                                    );
                                }
                                Err(e) => {
                                    return Err(e);
                                }
                            }
                        }
                        0 => {
                            anyhow::bail!("old_string not found in file (exact match required)");
                        }
                        _ => {
                            anyhow::bail!(
                                "old_string matches {exact_count} times; must match exactly once (or pass multiple=true to replace all)"
                            );
                        }
                    }
                }

                tokio::fs::write(&resolved_target, &new_content)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to write file: {e}"))?;
                update_search_index_after_write(ws, &resolved_target);
                Ok(format!(
                    "Edited {path}: replaced {replaced_count} occurrence{} ({} bytes)",
                    if replaced_count == 1 { "" } else { "s" },
                    new_content.len()
                ))
            }
        }
    }

    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<(&str, bool)>,
    ) -> Option<String> {
        match phase {
            ToolOutputPhase::Before => None,
            ToolOutputPhase::After => {
                let (_output, success) = outcome?;
                let old_string = super::get_opt_str(args, "old_string").filter(|s| !s.is_empty());
                let new_string = super::get_opt_str(args, "new_string").unwrap_or("?");
                if let Some(old) = old_string {
                    let combined = format!("{old}\n-----------\n{new_string}");
                    Some(format_file_tool_result("Edit", &combined, args, success))
                } else {
                    Some(format_file_tool_result("Write", new_string, args, success))
                }
            }
        }
    }
}

/// Result formatting for the edit tool's "After" phase.
/// Handles truncation and either a code fence or expandable blockquote
/// depending on content size.
#[must_use]
fn format_file_tool_result(
    action: &str,
    content: &str,
    args: &serde_json::Value,
    success: bool,
) -> String {
    let path = super::find_path_arg(args).unwrap_or("?");
    if !success {
        return format!("❌ {action} attempted on {path}");
    }

    let block = crate::util::truncate_sandwich(content, 2000, "debug");
    format!("✏️ {path}\n{block}")
}

// ── Whitespace-insensitive matching ───────────────────────────────

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
    let Some(entry) = crate::search_engine::get_engine_if_exists(ws) else {
        return;
    };

    // This is a parking_lot RwLock (non-poisoning) held for microseconds.
    // The synchronous I/O inside handle_create_or_modify (stat, binary
    // detection read) is acceptable — the hold time is negligible.
    match entry.picker.write() {
        Ok(mut guard) => {
            if let Some(ref mut picker) = *guard
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
        Err(e) => {
            tracing::warn!(
                workspace = ws.name,
                path = %file_path.display(),
                error = %e,
                "Failed to acquire search index write lock after file write"
            );
        }
    }
}

// ── Whitespace-insensitive matching details ─────────────────────────

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

/// Result of a whitespace-insensitive match.
#[derive(Debug)]
struct WsMatch {
    /// Byte offset in the original content where the match starts.
    start: usize,
    /// Byte offset in the original content where the match ends (exclusive).
    end: usize,
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
fn map_norm_span(
    norm_start: usize,
    norm_end: usize,
    segments: &[Segment],
) -> Result<(usize, usize)> {
    let seg = segment_at(norm_start, segments)?;
    // Verbatim segments (strings/regular content) map 1:1; whitespace segments map to the whole run
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

    Ok((orig_start, orig_end))
}

/// Find `old_string` in `content` using whitespace-insensitive matching.
///
/// Consecutive whitespace outside string literals is collapsed to single
/// spaces before matching. Normalizes both strings once, then checks for
/// ambiguity (multiple normalized occurrences of `old_string`). Returns:
///
/// - `Ok(Some(WsMatch))` — a single unambiguous match found.
/// - `Ok(None)` — pattern not found after normalization.
/// - `Err(...)` — ambiguous (pattern matches multiple times after
///   normalization) or a normalization bug in `map_norm_span`.
fn find_ws_insensitive(content: &str, old_string: &str) -> Result<Option<WsMatch>> {
    if old_string.is_empty() || content.is_empty() {
        return Ok(None);
    }

    let (norm_content, segments) = normalize_ws(content);
    let (norm_old, _) = normalize_ws(old_string);

    if norm_old.is_empty() {
        return Ok(None);
    }

    let Some(norm_pos) = norm_content.find(&norm_old) else {
        return Ok(None);
    };
    let norm_end = norm_pos + norm_old.len();

    // Check for ambiguity: a second occurrence after the first char of norm_old.
    // Using char boundary instead of raw +1 avoids panicking on multi-byte chars.
    let first_char_len = norm_old.chars().next().map_or(1, char::len_utf8);
    let search_start = norm_pos + first_char_len;
    if norm_content[search_start..].find(&norm_old).is_some() {
        anyhow::bail!(
            "old_string matches multiple times after whitespace normalization; \
             provide more surrounding context to disambiguate"
        );
    }

    let (start, end) = map_norm_span(norm_pos, norm_end, &segments)?;

    Ok(Some(WsMatch { start, end }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;

    /// Helper: creates a temp workspace directory for an edit test, writes
    /// initial files, runs the test closure, and cleans up afterwards.
    async fn with_temp_workspace<F, Fut>(test_name: &str, files: &[(&str, &str)], test: F)
    where
        F: FnOnce(PathBuf) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let dir = std::env::temp_dir().join(test_name);
        let _ = tokio::fs::remove_dir_all(&dir).await;
        if files.is_empty() {
            tokio::fs::create_dir_all(&dir).await.unwrap();
        }
        for (filename, content) in files {
            let path = dir.join(filename);
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.unwrap();
            }
            tokio::fs::write(&path, content).await.unwrap();
        }
        test(dir.clone()).await;
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    // ── Extension check tests ────────────────────────────────────────

    #[test]
    fn ws_insensitive_included_extensions() {
        for path in [
            "main.rs",
            "src/lib.rs",
            "app.js",
            "component.jsx",
            "app.ts",
            "component.tsx",
            "main.c",
            "main.h",
            "main.cpp",
            "main.hpp",
            "main.cc",
            "main.cxx",
            "Main.java",
            "Main.kt",
            "Main.kts",
            "main.go",
            "main.swift",
            "main.dart",
            "Program.cs",
            "main.zig",
            "Main.scala",
            "Main.rs",
            "Main.RS",
            "App.JS",
            "Main.Rs",
        ] {
            assert!(
                is_ws_insensitive_extension(path),
                "Expected match for {path}"
            );
        }
    }

    #[test]
    fn ws_insensitive_excluded_extensions() {
        for path in [
            "config.toml",
            "config.json",
            "config.yaml",
            "config.yml",
            "readme.md",
            "Dockerfile",
            "Makefile",
            "main.py",
            "main.rb",
            "main.php",
            "style.css",
            "script.sh",
            "docker-compose",
        ] {
            assert!(
                !is_ws_insensitive_extension(path),
                "Expected no match for {path}"
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
            assert_eq!(
                &content[m.start..m.end],
                *expected,
                "content={content:?} old={old:?}"
            );
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

    /// Cases where whitespace-insensitive matching is ambiguous
    /// (the normalized old_string appears more than once).
    #[test]
    fn ws_match_is_ambiguous_true_cases() {
        let ambiguous_cases: &[(&str, &str)] = &[
            // Repeated pattern — "a b" appears 3x in normalized "a b a b a b"
            ("a  b  a  b  a  b", "a b"),
            // Two lines that normalize to the same thing
            ("let  x  =  1;\nlet  x  =  1;", "let x = 1;"),
            // Overlapping match in repeated tokens
            ("a a a", "a a"),
            // Multi-byte first character (2-byte Latin ñ)
            ("ñ b ñ b", "ñ b"),
            // Multi-byte first character (3-byte CJK)
            ("字 符 字 符", "字 符"),
            // Multi-byte first character (4-byte emoji)
            ("🚀 b 🚀 b", "🚀 b"),
        ];
        for (content, old) in ambiguous_cases {
            let result = find_ws_insensitive(content, old);
            assert!(
                result.is_err(),
                "Expected ambiguous error: content={content:?} old={old:?} got {result:?}"
            );
            let err = format!("{}", result.unwrap_err());
            assert!(
                err.contains("multiple times after whitespace normalization"),
                "Error should mention ambiguity, got: {err}"
            );
        }
    }

    #[test]
    fn ws_match_is_ambiguous_false_cases() {
        let unambiguous_cases: &[(&str, &str)] = &[
            // Single match
            ("fn  foo()  {}", "fn foo() {}"),
            // Two different functions — only one matches the pattern
            ("fn  foo()  {}\nfn  bar()  {}", "fn bar() {}"),
            // Old string appears only once after normalization
            ("let  x  =  5;\nlet  y  =  10;\n    x  +  y", "let x = 5;"),
            // Empty old_string should not be ambiguous
            ("anything", ""),
            // Not found at all
            ("fn foo() {}", "fn bar() {}"),
            // Multi-byte single match (2-byte Latin ñ)
            ("fn  ñ  foo()  {}", "fn ñ foo()"),
            // Multi-byte single match (3-byte CJK)
            ("let  字  =  1;", "let 字 = 1;"),
            // Multi-byte single match (4-byte emoji)
            ("let  🚀  =  1;", "let 🚀 = 1;"),
        ];
        for (content, old) in unambiguous_cases {
            let result = find_ws_insensitive(content, old);
            assert!(
                result.is_ok(),
                "Expected no ambiguity error: content={content:?} old={old:?} got {result:?}"
            );
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
        let dir = std::env::temp_dir().join("mahbot_test_file_edit_multiple");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("test.txt"), "a b a c a d")
            .await
            .unwrap();

        // single match without multiple flag still works
        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "test.txt", "old_string": "b", "new_string": "x"}),
            )
            .await;
        assert!(result.is_ok(), "edit should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio::fs::read_to_string(dir.join("test.txt"))
                .await
                .unwrap(),
            "a x a c a d"
        );

        // multiple=true replaces all occurrences
        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "test.txt", "old_string": "a", "new_string": "y", "multiple": true}),
            )
            .await;
        assert!(result.is_ok(), "multiple edit should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("replaced 3 occurrences"));
        assert_eq!(
            tokio::fs::read_to_string(dir.join("test.txt"))
                .await
                .unwrap(),
            "y x y c y d"
        );

        // multiple=true with no matches still fails
        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "test.txt", "old_string": "z", "new_string": "w", "multiple": true}),
            )
            .await;
        assert!(
            result.is_err(),
            "edit with no matches should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("not found"));

        // multiple=true with single match works too
        tokio::fs::write(dir.join("test.txt"), "only one")
            .await
            .unwrap();
        let result = EditTool
            .execute(&Workspace::from_path(&dir), json!({"path": "test.txt", "old_string": "one", "new_string": "two", "multiple": true}))
            .await;
        assert!(
            result.is_ok(),
            "single match with multiple flag: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio::fs::read_to_string(dir.join("test.txt"))
                .await
                .unwrap(),
            "only two"
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn file_edit_match_operations() {
        let dir = std::env::temp_dir().join("mahbot_test_file_edit_match");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("test.txt"), "hello world")
            .await
            .unwrap();

        // replace single match
        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "test.txt", "old_string": "hello", "new_string": "goodbye"}),
            )
            .await;
        assert!(result.is_ok(), "edit should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio::fs::read_to_string(dir.join("test.txt"))
                .await
                .unwrap(),
            "goodbye world"
        );
        // not found
        let result = EditTool.execute(&Workspace::from_path(&dir), json!({"path": "test.txt", "old_string": "nonexistent", "new_string": "replacement"})).await;
        assert!(
            result.is_err(),
            "edit with nonexistent string should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("not found"));
        // multiple matches rejected
        tokio::fs::write(dir.join("test.txt"), "aaa bbb aaa")
            .await
            .unwrap();
        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "test.txt", "old_string": "aaa", "new_string": "ccc"}),
            )
            .await;
        assert!(result.is_err(), "multiple matches should fail: {result:?}");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("matches 2 times"));
        assert_eq!(
            tokio::fs::read_to_string(dir.join("test.txt"))
                .await
                .unwrap(),
            "aaa bbb aaa"
        );
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn file_edit_delete_via_empty_new_string() {
        with_temp_workspace(
            "mahbot_test_file_edit_delete",
            &[("test.txt", "keep remove keep")],
            |dir| async move {
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
            },
        )
        .await;
    }

    #[tokio::test]
    async fn edit_write_mode_creates_file() {
        let dir = std::env::temp_dir().join("mahbot_test_edit_write_mode");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let result = EditTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "out.txt", "new_string": "written!"}),
            )
            .await;
        assert!(result.is_ok(), "write mode should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("8 bytes"));

        let content = tokio::fs::read_to_string(dir.join("out.txt"))
            .await
            .unwrap();
        assert_eq!(content, "written!");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn edit_write_mode_with_empty_old_string() {
        let dir = std::env::temp_dir().join("mahbot_test_edit_write_mode_empty");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "out.txt", "old_string": "", "new_string": "content"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "write mode with empty old_string: {result:?}"
        );

        let content = tokio::fs::read_to_string(dir.join("out.txt"))
            .await
            .unwrap();
        assert_eq!(content, "content");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn edit_write_mode_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("mahbot_test_edit_write_mode_nested");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let result = EditTool
            .execute(
                &test_ws(&dir),
                json!({"path": "a/b/c/deep.txt", "new_string": "deep"}),
            )
            .await;
        assert!(result.is_ok(), "write with parent dirs: {result:?}");
        let content = tokio::fs::read_to_string(dir.join("a/b/c/deep.txt"))
            .await
            .unwrap();
        assert_eq!(content, "deep");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn file_edit_blocks_dangerous_paths() {
        with_temp_workspace("mahbot_test_file_edit_traversal", &[], |dir| async move {
            let result = EditTool
                .execute(
                    &test_ws(&dir),
                    json!({"path": "../../etc/passwd", "old_string": "root", "new_string": "x"}),
                )
                .await;
            assert!(result.is_err(), "traversal should be blocked: {result:?}");
            let err = format!("{}", result.unwrap_err());
            assert!(err.contains("not allowed"));
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
            assert!(err.contains("not allowed"));
        })
        .await;
    }

    #[tokio::test]
    async fn file_edit_normalizes_relative_path() {
        with_temp_workspace(
            "mahbot_test_file_edit_relative",
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

        let root = std::env::temp_dir().join("mahbot_test_file_edit_symlink_target");
        let workspace = root.join("workspace");
        let outside = root.join("outside");

        let _ = tokio::fs::remove_dir_all(&root).await;
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

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn file_edit_nonexistent_file() {
        with_temp_workspace("mahbot_test_file_edit_nofile", &[], |dir| async move {
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
            "mahbot_test_file_edit_abs_path",
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
            "mahbot_test_ws_ambiguous",
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
            "mahbot_test_ws_unambiguous",
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
        with_temp_workspace(
            "mahbot_test_ws_nocode",
            &[("readme.txt", "a  b  a  b")],
            |dir| async move {
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
                    err.contains("not found"),
                    ".txt should use exact matching only, got: {err}"
                );
            },
        )
        .await;
    }
}
