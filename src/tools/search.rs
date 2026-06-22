//! Backed by `fff-search` — an indexed search engine with fuzzy file
//! name matching, content grep, frecency ranking, constraint filtering,
//! and pagination.
//!
//! All agents searching the same workspace share a single engine instance
//! managed by [`crate::search_engine`]. Background scanning starts eagerly.

use crate::search_engine;
use crate::{Tool, ToolOutputPhase};
use async_trait::async_trait;
use fff_search::file_picker::FuzzySearchOptions;
use fff_search::grep::{GrepMode, GrepSearchOptions};
use fff_search::parse_grep_query;
use fff_search::{Constraint, GitStatusFilter};
use serde_json::json;
use std::fmt::Write;

const DEFAULT_MAX_RESULTS: usize = 50;
const MAX_RESULTS_LIMIT: usize = 500;

/// Valid parameter keys (schema + normalization aliases).
/// Used for unknown-parameter detection in the execute path.
const KNOWN_KEYS: &[&str] = &[
    // schema keys
    "mode",
    "query",
    "grep_mode",
    "case_sensitive",
    "max_results",
    "offset",
    "context_lines",
    // normalization aliases (consumed but still may appear in args)
    "pattern",
    "search",
    "search_term",
    "file_pattern",
    "grep_search",
    "path",
    "ext",
];

/// Repair common agent mistakes in search tool arguments before execution.
fn normalize_search_args(args: &mut serde_json::Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    // Query aliases — only when canonical key is absent.
    if !obj.contains_key("query") {
        for alias in ["pattern", "search", "search_term", "grep_search"] {
            if let Some(v) = obj.remove(alias) {
                obj.insert("query".to_string(), v);
                break;
            }
        }
    }

    // file_pattern → files-mode query when obvious.
    if !obj.contains_key("query")
        && let Some(fp) = obj.remove("file_pattern")
    {
        obj.insert("query".to_string(), fp);
        if !obj.contains_key("mode") {
            obj.insert("mode".to_string(), json!("files"));
        }
    }

    // Top-level mode conflated with grep_mode.
    if let Some(mode) = obj.get("mode").and_then(|v| v.as_str()).map(str::to_string) {
        match mode.as_str() {
            "plain_text" => {
                obj.insert("mode".to_string(), json!("grep"));
                if !obj.contains_key("grep_mode") {
                    obj.insert("grep_mode".to_string(), json!("plain_text"));
                }
            }
            "regex" => {
                obj.insert("mode".to_string(), json!("grep"));
                if !obj.contains_key("grep_mode") {
                    obj.insert("grep_mode".to_string(), json!("regex"));
                }
            }
            "content" => {
                obj.insert("mode".to_string(), json!("grep"));
            }
            "code" => {
                obj.insert("mode".to_string(), json!("grep"));
                if !obj.contains_key("grep_mode") {
                    obj.insert("grep_mode".to_string(), json!("fuzzy"));
                }
            }
            _ => {}
        }
    }

    // Invalid grep_mode values observed in production telemetry.
    if let Some(gm) = obj
        .get("grep_mode")
        .and_then(|v| v.as_str())
        .map(str::to_string)
    {
        match gm.as_str() {
            "exact" => {
                obj.insert("grep_mode".to_string(), json!("plain_text"));
            }
            "files" => {
                obj.remove("grep_mode");
                obj.insert("mode".to_string(), json!("files"));
            }
            _ => {}
        }
    }
}

/// Normalize search arguments by resolving the effective query string.
///
/// Handles common agent input mistakes:
/// - `"pattern"` as an alias for `"query"`
/// - Standalone `"path"` key → trailing-slash path segment constraint
/// - Standalone `"ext"` key → glob extension constraint
///
/// Returns `None` when no query components are present.
fn resolve_query(args: &serde_json::Value) -> Option<String> {
    let raw_query = super::get_opt_str(args, "query")
        .or_else(|| super::get_opt_str(args, "pattern"))
        .or_else(|| super::get_opt_str(args, "search"))
        .or_else(|| super::get_opt_str(args, "search_term"));

    let path_constraint = super::get_opt_str(args, "path")
        .filter(|p| !p.is_empty() && *p != "/")
        .map(|p| {
            let trimmed = p.trim_end_matches('/');
            format!("{trimmed}/")
        });

    let ext_constraint = super::get_opt_str(args, "ext")
        .filter(|e| !e.is_empty())
        .map(|e| {
            // Strip leading dots and asterisks (agents may copy-paste from docs)
            let trimmed = e.trim_start_matches(['.', '*']);
            format!("*.{trimmed}")
        });

    let mut query = String::new();
    if let Some(q) = raw_query.filter(|q| !q.is_empty()) {
        query.push_str(q);
    }
    if let Some(ref p) = path_constraint {
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(p);
    }
    if let Some(ref e) = ext_constraint {
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(e);
    }

    if query.is_empty() { None } else { Some(query) }
}

/// Unified workspace search tool.
///
/// - `mode = "files"` — fuzzy file/path name search (replaces `glob`)
/// - `mode = "grep"`  — file content search (replaces `rg`)
#[derive(Default)]
pub struct SearchTool;

impl SearchTool {
    /// Resolve the shared engine for a workspace and ensure the background
    /// scan has finished.
    async fn resolve_engine(ws: &crate::Workspace) -> Result<search_engine::EngineHandle, String> {
        let entry = search_engine::get_or_init_engine(ws)?;
        search_engine::ensure_scanned(&entry).await?;
        Ok(search_engine::EngineHandle::new(entry))
    }

    fn search_files(
        handle: &search_engine::EngineHandle,
        query: &str,
        max_results: usize,
        offset: usize,
        constraints: &[Constraint<'_>],
    ) -> anyhow::Result<String> {
        let paths = Self::find_file_path_list(handle, query, max_results, offset)?;
        if paths.is_empty() {
            return Self::format_files_zero_result(handle, query, offset, max_results, constraints);
        }

        let mut output = paths.join("\n");
        let total = paths.len();
        let _ = write!(output, "\n\nTotal: {total} files");
        Ok(output)
    }

    fn format_files_zero_result(
        handle: &search_engine::EngineHandle,
        query: &str,
        max_results: usize,
        offset: usize,
        constraints: &[Constraint<'_>],
    ) -> anyhow::Result<String> {
        let fff_query = parse_grep_query(query);
        let search_opts = FuzzySearchOptions {
            max_threads: 4,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 10,
            min_combo_count: 2,
            pagination: fff_search::PaginationArgs {
                offset,
                limit: max_results,
            },
        };
        let guard = handle.picker.read().unwrap();
        let Some(picker) = guard.as_ref() else {
            anyhow::bail!("Search engine not yet initialized.")
        };
        let qt_guard = handle.query_tracker.read().unwrap();
        let qt_ref = qt_guard.as_ref();
        let result = picker.fuzzy_search(&fff_query, qt_ref, search_opts);

        let mut diag = format!("No files matching '{query}' found in workspace.\nTotal: 0 files");
        let pagination_past_end = result.total_matched > 0 && result.total_matched <= offset;
        diag.push_str("\n── diagnostics ──\n");
        let _ = writeln!(
            diag,
            "  total_files: {} (files in index)",
            result.total_files
        );
        let _ = writeln!(
            diag,
            "  total_matched: {} (matches before pagination)",
            result.total_matched
        );
        let _ = writeln!(diag, "  offset: {offset}  limit: {max_results}");
        if !constraints.is_empty() {
            let _ = writeln!(diag, "  constraints: {}", format_constraints(constraints));
        }
        if pagination_past_end {
            let _ = writeln!(
                diag,
                "  ⚠ offset={offset} exceeds total prefiltered files ({matched}) — no files to search. Try offset=0.",
                matched = result.total_matched
            );
        } else if result.total_files == 0 {
            diag.push_str("  ⚠ index has 0 files — workspace may not be scanned yet\n");
        } else if result.total_matched == 0 {
            diag.push_str(
                "  ℹ 0 files matched before pagination — try broader query or remove constraints\n",
            );
        }
        Ok(diag)
    }

    /// Return relative file paths matching a fuzzy/glob query (for read recovery).
    pub(crate) fn find_file_path_list(
        handle: &search_engine::EngineHandle,
        query: &str,
        max_results: usize,
        offset: usize,
    ) -> anyhow::Result<Vec<String>> {
        let fff_query = parse_grep_query(query);
        let search_opts = FuzzySearchOptions {
            max_threads: 4,
            current_file: None,
            project_path: None,
            combo_boost_score_multiplier: 10,
            min_combo_count: 2,
            pagination: fff_search::PaginationArgs {
                offset,
                limit: max_results,
            },
        };
        let guard = handle.picker.read().unwrap();
        let Some(picker) = guard.as_ref() else {
            anyhow::bail!("Search engine not yet initialized.")
        };
        let qt_guard = handle.query_tracker.read().unwrap();
        let qt_ref = qt_guard.as_ref();
        let result = picker.fuzzy_search(&fff_query, qt_ref, search_opts);
        Ok(result
            .items
            .iter()
            .map(|file| file.relative_path(picker))
            .collect())
    }

    /// Workspace file search helper for other tools (e.g. read path recovery).
    pub(crate) async fn find_file_paths(
        ws: &crate::Workspace,
        query: &str,
        max_results: usize,
    ) -> anyhow::Result<Vec<String>> {
        if !crate::search_engine::registry_initialized() {
            return Ok(vec![]);
        }
        let handle = Self::resolve_engine(ws)
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        Self::find_file_path_list(&handle, query, max_results, 0)
    }

    fn search_grep(
        handle: &search_engine::EngineHandle,
        query: &str,
        max_results: usize,
        offset: usize,
        args: &serde_json::Value,
        constraints: &[Constraint<'_>],
    ) -> anyhow::Result<String> {
        let grep_mode_str = super::get_opt_str(args, "grep_mode").unwrap_or("fuzzy");

        let grep_mode = match grep_mode_str {
            "plain_text" => GrepMode::PlainText,
            "regex" => GrepMode::Regex,
            "fuzzy" => GrepMode::Fuzzy,
            _ => {
                anyhow::bail!(
                    "Invalid grep_mode '{grep_mode_str}'. Allowed values: fuzzy, plain_text, regex."
                );
            }
        };

        let case_sensitive = super::get_bool(args, "case_sensitive", true);

        let context_lines = super::get_usize(args, "context_lines", 0);

        // Parse the query (extracts constraints + pattern)
        let fff_query = parse_grep_query(query);

        // Build grep options
        let grep_opts = GrepSearchOptions {
            mode: grep_mode,
            smart_case: !case_sensitive,
            max_file_size: super::MAX_FILE_SIZE_BYTES,
            max_matches_per_file: 100,
            file_offset: offset,
            page_limit: max_results,
            time_budget_ms: 10_000, // 10 seconds
            before_context: context_lines,
            after_context: context_lines,
            classify_definitions: false,
            trim_whitespace: true,
            abort_signal: None,
        };

        // Lock the picker and perform grep
        let guard = handle.picker.read().unwrap();
        let Some(picker) = guard.as_ref() else {
            anyhow::bail!("Search engine not yet initialized.")
        };

        let result = picker.grep(&fff_query, &grep_opts);

        let regex_note = match &result.regex_fallback_error {
            Some(err) => format!(
                "\n\nNote: Regex pattern had an error: {err}. Fell back to plain-text matching."
            ),
            None => String::new(),
        };

        if result.matches.is_empty() {
            return Ok(build_grep_zero_diag(
                query,
                &result,
                max_results,
                offset,
                constraints,
                &regex_note,
            ));
        }

        // Format output (mirrors rg-style: file:line:content with context)
        let mut output = String::new();

        // Group matches by file_index so we can group them per file
        let mut last_file_index: Option<usize> = None;
        for grep_match in &result.matches {
            let rel_path = result.files[grep_match.file_index].relative_path(picker);

            // Separate match groups from different files with a blank line
            if let Some(last) = last_file_index
                && last != grep_match.file_index
            {
                output.push('\n');
            }
            last_file_index = Some(grep_match.file_index);

            // Context before
            let ctx_before_count = grep_match.context_before.len();
            for (i, ctx_line) in grep_match.context_before.iter().enumerate() {
                let offset_back = (ctx_before_count - i) as u64;
                let line_num = grep_match.line_number.saturating_sub(offset_back);
                let _ = writeln!(output, "{rel_path}-{line_num}-{ctx_line}");
            }

            // The match line itself
            let _ = writeln!(
                output,
                "{rel_path}:{}:{}",
                grep_match.line_number, grep_match.line_content
            );

            // Context after
            for (i, ctx_line) in grep_match.context_after.iter().enumerate() {
                let line_num = grep_match.line_number + (i + 1) as u64;
                let _ = writeln!(output, "{rel_path}-{line_num}-{ctx_line}");
            }
        }

        let _ = write!(
            output,
            "Total: {} matches in {} files{regex_note}",
            result.matches.len(),
            result.files_with_matches
        );

        // Pagination hint: surface next_file_offset for correct file-based pagination
        if result.next_file_offset > 0 {
            let _ = write!(
                output,
                "\n── pagination ──\n  next_file_offset: {} (pass as 'offset' for the next page)",
                result.next_file_offset
            );
        }

        Ok(output)
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &'static str {
        "search"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "mode": {
                    "type": "string",
                    "enum": ["files", "grep"],
                    "description": "'files' = fuzzy file/path name search (ranked, replaces glob). 'grep' = file content search (replaces rg). Note: 'mode' and 'grep_mode' are DIFFERENT parameters — 'mode' selects files vs grep; 'grep_mode' selects the matching strategy within grep."
                },
                "query": {
                    "type": "string",
                    "description": "The search query. Use inline constraint syntax like '*.rs' (extension) or 'src/' (path segment) for filtering — or use the standalone 'ext'/'path' parameters. For 'files': fuzzy text like 'lib.rs' or 'types/query'. For 'grep': pattern to find in file contents (supports regex, plain_text, or fuzzy via grep_mode)."
                },
                "grep_mode": {
                    "type": "string",
                    "enum": ["fuzzy", "plain_text", "regex"],
                    "description": "Grep matching mode (only used when mode='grep', NOT the same as the top-level 'mode' parameter). 'fuzzy' (default) — approximate/similar matching, the recommended starting mode for most searches. 'plain_text' — literal substring search, use for exact identifiers, variable names, or error strings. 'regex' — regex pattern matching, use for complex patterns (patterns with special chars like ., *, +, ?, [], (), {}, ^, $, |, ). Note: lookahead/lookbehind assertions ((?=...), (?!...), (?<=...), (?<!...)) are not supported and will return an error. Ignored when mode='files'."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Grep mode only, default: true. If true, matches case-sensitively. If false, uses smart-case (case-insensitive when pattern is all-lowercase, case-sensitive otherwise). Ignored in 'files' mode."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default: 50, max: 500). Use with offset for pagination."
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0, 0-based). In 'files' mode: skips this many items (item-based). In 'grep' mode: skips this many *files*, not matches (file-based) — use 'next_file_offset' from the previous response for correct pagination."
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Context lines before/after each match (grep mode only, default: 0). Ignored in 'files' mode."
                },
                "path": {
                    "type": "string",
                    "description": "Optional path segment filter. Appends a trailing-slash constraint (e.g. 'src' becomes 'src/') to restrict results to a directory or path prefix. Use this parameter or inline 'src/' syntax in the query — they are additive."
                },
                "ext": {
                    "type": "string",
                    "description": "Optional file extension filter. Appends a glob constraint (e.g. 'rs' becomes '*.rs') to restrict results by extension. Use this parameter or inline '*.rs' syntax in the query — they are additive."
                }
            }),
            &["query"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false // source code, not secrets
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false // indexed search, no mutations
    }

    #[allow(clippy::too_many_lines)]
    async fn execute(
        &self,
        ws: &crate::Workspace,
        mut args: serde_json::Value,
    ) -> anyhow::Result<String> {
        normalize_search_args(&mut args);

        let mode = super::get_opt_str(&args, "mode")
            .unwrap_or("grep")
            .to_string();

        // --- Normalize: handle common agent input mistakes ---
        let query = resolve_query(&args).ok_or_else(|| {
            anyhow::anyhow!(
                "Missing required field: 'query'. \
                 Note: the parameter is called 'query' (not 'pattern' or 'search'). \
                 Use the 'ext'/'path' parameters or inline constraints like *.rs, src/, \
                 e.g. {{\"query\": \"my_search\", \"path\": \"src\", \"ext\": \"rs\"}}."
            )
        })?;

        // ── Lookaround detection for regex mode ──────────────────────────
        // The Rust `regex` crate does not support lookahead/lookbehind.
        // Detecting these before dispatch prevents silent plain-text fallback
        // that would return wrong results without the agent knowing.
        //
        // Note: substring scanning may false-positive on patterns like [(?=)]
        // where lookaround syntax appears inside a character class, but such
        // patterns are vanishingly rare in agent-generated regex.
        if mode == "grep" {
            let grep_mode = super::get_opt_str(&args, "grep_mode").unwrap_or("fuzzy");
            if grep_mode == "regex"
                && (query.contains("(?=")
                    || query.contains("(?!")
                    || query.contains("(?<=")
                    || query.contains("(?<!"))
            {
                anyhow::bail!(
                    "This regex pattern uses lookahead/lookbehind assertions which are not supported by the search tool's regex engine. Use plain_text or fuzzy mode instead, or reformulate the pattern without lookaround."
                );
            }
        }

        // ── Unknown parameter detection ────────────────────────────
        // Schema keys + normalization aliases that are valid in args.
        // Any other top-level key is an agent mistake worth warning about.
        let unknown_keys: Vec<&str> = args
            .as_object()
            .map(|obj| {
                obj.keys()
                    .filter_map(|k| {
                        if KNOWN_KEYS.contains(&k.as_str()) {
                            None
                        } else {
                            Some(k.as_str())
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let unknown_param_warning = if unknown_keys.is_empty() {
            None
        } else {
            let keys = unknown_keys.join(", ");
            Some(format!(
                "\n── note ──\n  Unrecognized parameter(s): {keys}. \
                 These were ignored. Valid parameters: {}.",
                KNOWN_KEYS.join(", ")
            ))
        };

        // ── Parse query constraints for diagnostics ─────────────────
        let fff_query = parse_grep_query(&query);
        let query_constraints: Vec<Constraint<'_>> = fff_query.constraints.clone();

        let max_results =
            super::get_usize(&args, "max_results", DEFAULT_MAX_RESULTS).min(MAX_RESULTS_LIMIT);

        if max_results == 0 {
            anyhow::bail!(
                "max_results must be at least 1. Got 0. \
                 Use the default (50) or a positive value."
            );
        }

        let offset = super::get_usize(&args, "offset", 0);

        let handle = Self::resolve_engine(ws)
            .await
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut output = match mode.as_str() {
            "files" => Self::search_files(&handle, &query, max_results, offset, &query_constraints),
            "grep" => Self::search_grep(
                &handle,
                &query,
                max_results,
                offset,
                &args,
                &query_constraints,
            ),
            _ => anyhow::bail!("Invalid mode '{mode}'. Allowed values: 'files', 'grep'."),
        }?;

        // Append unknown parameter warning to the result (after the main output,
        // so it doesn't disrupt valid results)
        if let Some(warning) = unknown_param_warning {
            output.push_str(&warning);
        }

        Ok(output)
    }

    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<&crate::tools::ToolExecutionOutcome>,
    ) -> Option<String> {
        match phase {
            ToolOutputPhase::Before => None,
            ToolOutputPhase::After => {
                let mode = super::get_opt_str(args, "mode");

                // Resolve the effective query using the same normalization as
                // execute() — 'pattern' alias, path/ext constraint composition.
                let query = resolve_query(args).unwrap_or_else(|| "?".to_string());

                let prefix = match mode {
                    Some(m) => format!("{m}: "),
                    None => String::new(),
                };

                let outcome = outcome?;
                if outcome.success {
                    // Detect zero results by checking for the `── diagnostics ──` marker.
                    // Both zero-result output paths (`search_files()` and
                    // `build_grep_zero_diag()`) append this structured block.
                    // Using a structural delimiter avoids false-positives from file
                    // paths or grep content that happen to start with "No ".
                    let is_empty = outcome.output.contains("\n── diagnostics ──\n");
                    if is_empty {
                        // Enrich zero-result log entries with call arguments for
                        // debugging — mode, path/ext constraints, offset, case_sensitivity.
                        let mut details = Vec::new();

                        if let Some(p) = super::get_opt_str(args, "path").filter(|p| !p.is_empty())
                        {
                            details.push(format!("path={p}"));
                        }
                        if let Some(e) = super::get_opt_str(args, "ext").filter(|e| !e.is_empty()) {
                            details.push(format!("ext={e}"));
                        }
                        let offset_val = super::get_usize(args, "offset", 0);
                        if offset_val > 0 {
                            details.push(format!("offset={offset_val}"));
                        }
                        // Show case_sensitive only when non-default (explicitly false)
                        if let Some(cs) = super::get_opt_bool(args, "case_sensitive")
                            && !cs
                        {
                            details.push("case_sensitive=false".to_string());
                        }

                        let suffix = if details.is_empty() {
                            String::new()
                        } else {
                            format!(" [{}]", details.join(", "))
                        };

                        Some(format!("🔍 {prefix}{query} — 0 results{suffix}"))
                    } else {
                        Some(format!("🔍 {prefix}{query}"))
                    }
                } else {
                    let output = outcome.output.trim();
                    let err = if output.is_empty() {
                        "unknown error"
                    } else {
                        output
                    };
                    Some(format!("🔍 {prefix}{query} — ❌ {err}"))
                }
            }
        }
    }
}

// ── Constraint formatting helpers ─────────────────────────────────────

/// Format a single constraint to human-readable text.
///
/// The `Constraint` enum only implements `Debug` (producing `Extension("rs")`,
/// `FilePath("src/")`, etc.), which is machine-oriented. This function
/// produces agent-friendly labels using the inline query syntax: `*.rs`,
/// `src/`, `!*.md`, etc.
fn format_constraint(constraint: &Constraint<'_>) -> String {
    match constraint {
        Constraint::Extension(s) => format!("*.{s}"),
        Constraint::Glob(s) => format!("glob:{s}"),
        Constraint::Parts(parts) => parts.join(" "),
        Constraint::Text(s) => (*s).to_string(),
        Constraint::Exclude(parts) => format!("!{}", parts.join(" ")),
        Constraint::PathSegment(s) | Constraint::FilePath(s) => format!("{s}/"),
        Constraint::FileType(s) => format!("type:{s}"),
        Constraint::GitStatus(status) => {
            let label = match status {
                GitStatusFilter::Modified => "modified",
                GitStatusFilter::Untracked => "untracked",
                GitStatusFilter::Staged => "staged",
                GitStatusFilter::Unmodified => "unmodified",
            };
            format!("status:{label}")
        }
        Constraint::Not(inner) => format!("!{}", format_constraint(inner)),
    }
}

/// Format a list of constraints as a comma-separated string.
fn format_constraints(constraints: &[Constraint<'_>]) -> String {
    constraints
        .iter()
        .map(|c| format_constraint(c))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build diagnostic output for a zero-result grep search.
///
/// Extracted from `search_grep` to keep the main function under the
/// clippy `too_many_lines` threshold while providing rich diagnostics
/// for agents to self-correct.
fn build_grep_zero_diag(
    query: &str,
    result: &fff_search::grep::GrepResult<'_>,
    max_results: usize,
    offset: usize,
    constraints: &[Constraint<'_>],
    regex_note: &str,
) -> String {
    let mut diag =
        format!("No matches found for '{query}'.\nTotal: 0 matches in 0 files{regex_note}");

    let pagination_past_end =
        result.next_file_offset == 0 && result.total_files_searched == 0 && offset > 0;

    diag.push_str("\n── diagnostics ──\n");
    let _ = writeln!(
        diag,
        "  total_files: {} (files in index)",
        result.total_files
    );
    let _ = writeln!(
        diag,
        "  total_files_searched: {} (files searched this call)",
        result.total_files_searched
    );
    let _ = writeln!(
        diag,
        "  filtered_file_count: {} (searchable files after filtering: \
         excludes binary, too-large, etc.)",
        result.filtered_file_count
    );
    let _ = writeln!(diag, "  files_with_matches: {}", result.files_with_matches);
    let _ = writeln!(diag, "  offset: {offset}  limit: {max_results}");
    let _ = writeln!(
        diag,
        "  next_file_offset: {} (0 = no more files to search)",
        result.next_file_offset
    );

    if !constraints.is_empty() {
        let _ = writeln!(diag, "  constraints: {}", format_constraints(constraints));
    }

    if pagination_past_end {
        let _ = writeln!(
            diag,
            "  ⚠ offset={offset} exceeds total prefiltered files ({total}) — no files to search. Try offset=0.",
            total = result.filtered_file_count
        );
    } else if result.next_file_offset > 0 {
        // There are more files to search but the current offset yielded no matches.
        // This can happen when the offset lands in a gap between matching files.
        let _ = writeln!(
            diag,
            "  ℹ offset={} covered the searched range but found no matches — \
             try offset={} to continue searching further files",
            offset, result.next_file_offset
        );
    } else if result.total_files == 0 {
        diag.push_str("  ⚠ index has 0 files — workspace may not be scanned yet\n");
    } else if result.filtered_file_count == 0
        && result.total_files_searched == 0
        && result.total_files > 0
    {
        diag.push_str(
            "  ℹ all files filtered out by constraints — try broader query or fewer constraints\n",
        );
    } else if offset > 0 && result.total_files_searched > 0 && result.next_file_offset == 0 {
        diag.push_str(
            "  ⚠ offset may be stale from a previous search — try offset=0 for a fresh search\n",
        );
    } else if result.files_with_matches == 0 {
        diag.push_str(
            "  ℹ 0 files matched — try broader query, different grep_mode, or remove constraints\n",
        );
    }

    diag
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── normalize_search_args ──────────────────────────────────────────

    #[test]
    fn normalize_search_args_query_aliases() {
        // Each alias (pattern, search, search_term, grep_search) is mapped
        // to the canonical "query" key and removed.
        let mut args = json!({"pattern": "foo"});
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "foo");
        assert!(args.get("pattern").is_none());

        let mut args = json!({"search": "bar"});
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "bar");
        assert!(args.get("search").is_none());

        let mut args = json!({"search_term": "baz"});
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "baz");
        assert!(args.get("search_term").is_none());

        // grep_search is the lowest-priority alias, only recognized by
        // normalize_search_args (not by resolve_query).
        let mut args = json!({"grep_search": "qux"});
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "qux");
        assert!(args.get("grep_search").is_none());
    }

    #[test]
    fn normalize_search_args_query_alias_priority() {
        // When multiple aliases are present, first wins: pattern > search > search_term > grep_search
        let mut args = json!({
            "pattern": "from_pattern",
            "search": "from_search",
            "search_term": "from_search_term",
            "grep_search": "from_grep_search"
        });
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "from_pattern");

        // pattern absent, search present
        let mut args = json!({
            "search": "from_search",
            "search_term": "from_search_term"
        });
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "from_search");
    }

    #[test]
    fn normalize_search_args_query_alias_does_not_overwrite_existing_query() {
        // When `query` already exists, aliases are ignored
        let mut args = json!({"query": "existing", "pattern": "alias"});
        normalize_search_args(&mut args);
        // query unchanged, pattern remains as orphan key
        assert_eq!(args["query"], "existing");
        assert_eq!(args["pattern"], "alias");
    }

    #[test]
    fn normalize_search_args_file_pattern_implies_files_mode() {
        // file_pattern with no mode → files mode
        let mut args = json!({"file_pattern": "*.rs"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "files");
        assert_eq!(args["query"], "*.rs");
        assert!(args.get("file_pattern").is_none());

        // file_pattern with existing mode → mode preserved, query set
        let mut args = json!({"file_pattern": "main.rs", "mode": "grep"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        assert_eq!(args["query"], "main.rs");

        // file_pattern is NOT consumed when query already exists (orphan key behavior)
        let mut args = json!({"query": "already", "file_pattern": "other.rs"});
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "already");
        // file_pattern remains as orphan key (not removed when query exists)
        assert_eq!(args["file_pattern"], "other.rs");
    }

    #[test]
    fn normalize_search_args_file_pattern_blocked_by_early_alias() {
        // If pattern was already consumed as query alias, file_pattern is the
        // *second* fallback. Since query now exists, file_pattern is not processed.
        let mut args = json!({"pattern": "my_query", "file_pattern": "*.rs"});
        normalize_search_args(&mut args);
        assert_eq!(args["query"], "my_query");
        // file_pattern remains as orphan key because query is already set by alias
        assert_eq!(args["file_pattern"], "*.rs");
    }

    #[test]
    fn normalize_search_args_mode_plain_text() {
        let mut args = json!({"mode": "plain_text"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        assert_eq!(args["grep_mode"], "plain_text");

        // mode "plain_text" with existing grep_mode → mode remapped, grep_mode preserved
        let mut args = json!({"mode": "plain_text", "grep_mode": "fuzzy"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        // grep_mode was already set, so it's NOT overwritten
        assert_eq!(args["grep_mode"], "fuzzy");
    }

    #[test]
    fn normalize_search_args_mode_regex() {
        let mut args = json!({"mode": "regex"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        assert_eq!(args["grep_mode"], "regex");
    }

    #[test]
    fn normalize_search_args_mode_content() {
        let mut args = json!({"mode": "content"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        // grep_mode should NOT be set for "content"
        assert!(
            args.get("grep_mode").is_none(),
            "content should not set grep_mode"
        );
    }

    #[test]
    fn normalize_search_args_mode_code() {
        let mut args = json!({"mode": "code"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        assert_eq!(args["grep_mode"], "fuzzy");

        // code with existing grep_mode → grep_mode preserved
        let mut args = json!({"mode": "code", "grep_mode": "regex"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        assert_eq!(args["grep_mode"], "regex");
    }

    #[test]
    fn normalize_search_args_mode_files_unchanged() {
        // "files" is already a valid mode — no remapping
        let mut args = json!({"mode": "files", "query": "lib.rs"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "files");
    }

    #[test]
    fn normalize_search_args_unknown_mode_unchanged() {
        // Unknown mode values pass through unchanged
        let mut args = json!({"mode": "unknown_value"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "unknown_value");
    }

    #[test]
    fn normalize_search_args_grep_mode_rewrites() {
        // "exact" → "plain_text"
        let mut args = json!({"grep_mode": "exact"});
        normalize_search_args(&mut args);
        assert_eq!(args["grep_mode"], "plain_text");

        // "files" → mode="files", grep_mode removed
        let mut args = json!({"grep_mode": "files"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "files");
        assert!(
            args.get("grep_mode").is_none(),
            "grep_mode should be removed when value is 'files'"
        );

        // "files" when mode already "files" → mode stays "files", grep_mode removed
        let mut args = json!({"mode": "files", "grep_mode": "files"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "files");
        assert!(args.get("grep_mode").is_none());

        // "files" overrides a non-"files" mode to "files"
        let mut args = json!({"mode": "grep", "grep_mode": "files"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "files");
    }

    #[test]
    fn normalize_search_args_non_object_early_return() {
        // Non-object values should return early (defensive guard)
        let mut args = json!("string_value");
        normalize_search_args(&mut args);
        assert_eq!(args, json!("string_value"));

        let mut args = json!(42);
        normalize_search_args(&mut args);
        assert_eq!(args, json!(42));

        let mut args = json!(null);
        normalize_search_args(&mut args);
        assert_eq!(args, json!(null));

        let mut args = json!([1, 2, 3]);
        normalize_search_args(&mut args);
        assert_eq!(args, json!([1, 2, 3]));
    }

    #[test]
    fn normalize_search_args_empty_object_noop() {
        let mut args = json!({});
        normalize_search_args(&mut args);
        assert_eq!(args, json!({}));
    }

    #[test]
    fn normalize_search_args_mode_plain_text_and_grep_mode_exact_double_mapping() {
        // Mode remapping runs before grep_mode remapping. "plain_text" mode first
        // sets grep_mode to "plain_text" (because grep_mode is absent), then the
        // grep_mode remapping sees the already-valid "plain_text" and leaves it.
        let mut args = json!({"mode": "plain_text", "grep_mode": "exact"});
        normalize_search_args(&mut args);
        assert_eq!(args["mode"], "grep");
        assert_eq!(args["grep_mode"], "plain_text");
    }

    // ── resolve_query ──────────────────────────────────────────────────

    #[test]
    fn resolve_query_query_aliases() {
        // Each non-canonical key maps to the effective query.
        let args = json!({"pattern": "struct Foo"});
        assert_eq!(resolve_query(&args).as_deref(), Some("struct Foo"));

        let args = json!({"search": "bar"});
        assert_eq!(resolve_query(&args).as_deref(), Some("bar"));

        let args = json!({"search_term": "baz"});
        assert_eq!(resolve_query(&args).as_deref(), Some("baz"));

        // grep_search is NOT recognized by resolve_query (only by normalize_search_args).
        let args = json!({"grep_search": "qux"});
        assert_eq!(resolve_query(&args), None);
    }

    #[test]
    fn resolve_query_query_overrides_aliases() {
        // Canonical "query" key takes priority over aliases
        let args = json!({
            "query": "primary",
            "pattern": "secondary",
            "search": "tertiary",
            "search_term": "quaternary"
        });
        assert_eq!(resolve_query(&args).as_deref(), Some("primary"));
    }

    #[test]
    fn resolve_query_pattern_overrides_search() {
        let args = json!({
            "pattern": "from_pattern",
            "search": "from_search",
            "search_term": "from_term"
        });
        assert_eq!(resolve_query(&args).as_deref(), Some("from_pattern"));
    }

    #[test]
    fn resolve_query_search_overrides_search_term() {
        let args = json!({
            "search": "from_search",
            "search_term": "from_term"
        });
        assert_eq!(resolve_query(&args).as_deref(), Some("from_search"));
    }

    #[test]
    fn resolve_query_path_constraint() {
        let args = json!({"path": "src"});
        assert_eq!(resolve_query(&args).as_deref(), Some("src/"));

        // Path with trailing slash should have only one trailing slash
        let args = json!({"path": "src/"});
        assert_eq!(resolve_query(&args).as_deref(), Some("src/"));
    }

    #[test]
    fn resolve_query_path_ignores_empty_or_root() {
        let args = json!({"path": ""});
        assert_eq!(resolve_query(&args), None);

        let args = json!({"path": "/"});
        assert_eq!(resolve_query(&args), None);
    }

    #[test]
    fn resolve_query_ext_constraint() {
        let args = json!({"ext": "rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("*.rs"));
    }

    #[test]
    fn resolve_query_ext_strips_leading_dots_and_asterisks() {
        let args = json!({"ext": ".rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("*.rs"));

        let args = json!({"ext": "*.rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("*.rs"));

        let args = json!({"ext": "**.rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("*.rs"));
    }

    #[test]
    fn resolve_query_ext_ignores_empty() {
        let args = json!({"ext": ""});
        assert_eq!(resolve_query(&args), None);
    }

    #[test]
    fn resolve_query_constraint_combinations() {
        let args = json!({"query": "foo", "path": "src"});
        assert_eq!(resolve_query(&args).as_deref(), Some("foo src/"));

        let args = json!({"query": "foo", "ext": "rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("foo *.rs"));

        // Three-part: query, path, ext
        let args = json!({"query": "foo", "path": "src", "ext": "rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("foo src/ *.rs"));

        // No explicit query, path and ext only
        let args = json!({"path": "src", "ext": "rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("src/ *.rs"));
    }

    #[test]
    fn resolve_query_empty_query_behavior() {
        // No query components at all → None
        let args = json!({});
        assert_eq!(resolve_query(&args), None);

        // query key present but empty string → None
        let args = json!({"query": ""});
        assert_eq!(resolve_query(&args), None);

        // Empty query with path → path constraint only
        let args = json!({"query": "", "path": "src"});
        assert_eq!(resolve_query(&args).as_deref(), Some("src/"));

        // Empty query with ext → ext constraint only
        let args = json!({"query": "", "ext": "rs"});
        assert_eq!(resolve_query(&args).as_deref(), Some("*.rs"));
    }

    #[test]
    fn resolve_query_non_string_values_ignored() {
        // Non-string values are filtered out by get_opt_str
        let args = json!({"query": 42});
        assert_eq!(resolve_query(&args), None);

        // Non-string path is silently skipped, query still resolves
        let args = json!({"query": "foo", "path": 42});
        assert_eq!(resolve_query(&args).as_deref(), Some("foo"));

        // Non-string ext is silently skipped, query still resolves
        let args = json!({"query": "foo", "ext": true});
        assert_eq!(resolve_query(&args).as_deref(), Some("foo"));
    }
}
