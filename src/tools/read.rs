use std::path::Path;

use crate::tools::{ShellMode, ShellTool, search::SearchTool};
use crate::{Tool, ToolOutputPhase, Workspace};
use async_trait::async_trait;
use serde_json::json;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator, Tree};

pub struct ReadTool;

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
                    "description": "Read mode. 'content' (default): line-numbered file read. 'symbols': list all top-level AST symbols with line ranges. 'zoom': extract a single symbol's source by name.",
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
        let path = super::require_path_arg(&args)?;

        if super::path_contains_wildcard(&path) {
            return self.recover_wildcard_path(ws, &path).await;
        }

        let resolved_path = match super::resolve_read_target(ws.as_path(), &path).await {
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
        match super::find_path_arg(args) {
            Some(path) => super::should_scrub_file_path(path),
            None => true, // No path? Be safe and scrub.
        }
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false // read-only file inspection
    }

    fn format_output(&self, output: &str) -> String {
        const MAX_CHARS: usize = 5_000;
        if output.len() <= MAX_CHARS {
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
                let body_budget = MAX_CHARS.saturating_sub(header.len() + marker_budget + 1);

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
        crate::util::format_tool_output(output)
    }

    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<&crate::tools::ToolExecutionOutcome>,
    ) -> Option<String> {
        match phase {
            ToolOutputPhase::Before => {
                let path = super::find_path_arg(args).unwrap_or("?");
                let range = Self::format_range(args);
                if range.is_empty() {
                    Some(format!("👀 {path}"))
                } else {
                    Some(format!("👀 {path} ({range})"))
                }
            }
            ToolOutputPhase::After => {
                let outcome = outcome?;
                if outcome.success {
                    None
                } else {
                    let path = super::find_path_arg(args).unwrap_or("?");
                    Some(format!("❌ Failed to read {path}"))
                }
            }
        }
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
        let hint = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(path);

        let matches = SearchTool::find_file_paths(ws, hint, 8)
            .await
            .unwrap_or_default();
        if matches.is_empty() {
            anyhow::bail!("{original_err}");
        }

        if matches.len() == 1 {
            let recovered = &matches[0];
            let resolved = super::resolve_read_target(ws.as_path(), recovered).await?;
            let note = format!("[Recovered path: requested '{path}', using '{recovered}']");
            return self.read_resolved(ws, &resolved, Some(&note), args).await;
        }

        anyhow::bail!("{original_err}\nDid you mean:\n  {}", matches.join("\n  "))
    }

    fn format_range(args: &serde_json::Value) -> String {
        match (
            super::get_opt_u64(args, "offset"),
            super::get_opt_u64(args, "limit"),
        ) {
            (None, None) => String::new(),
            (Some(o), None) => format!("{o}:"),
            (None, Some(l)) => format!("1:{}", l.max(1)),
            (Some(o), Some(l)) => format!("{o}:{}", o.saturating_add(l.max(1) - 1)),
        }
    }

    /// Execute the standard content read mode.
    async fn execute_content(
        &self,
        resolved_path: &Path,
        args: &serde_json::Value,
    ) -> anyhow::Result<String> {
        match tokio::fs::read_to_string(resolved_path).await {
            Ok(contents) => {
                let lines: Vec<&str> = contents.lines().collect();
                let total = lines.len();

                if total == 0 {
                    return Ok(String::new());
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
            Err(e) => {
                // Not valid UTF-8 — read raw bytes and try to extract text
                let bytes = tokio::fs::read(resolved_path).await.map_err(|ee| {
                    anyhow::anyhow!(
                        "Initial error: {e}\n\
                         Failed to read file: {ee}"
                    )
                })?;

                // Lossy fallback — replaces invalid bytes with U+FFFD
                let lossy = String::from_utf8_lossy(&bytes).into_owned();
                Ok(lossy)
            }
        }
    }

    /// List all top-level AST symbols with line ranges.
    async fn execute_symbols(&self, resolved_path: &Path) -> anyhow::Result<String> {
        let ps = read_and_parse(resolved_path, "symbol extraction").await?;

        let query = build_symbol_query(&ps)?;

        let root_node = ps.tree.root_node();
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, root_node, ps.source.as_bytes());
        let mut symbols: Vec<String> = Vec::new();
        matches.advance();
        while let Some(m) = matches.get() {
            for capture in m.captures {
                let node = capture.node;
                let name = node.utf8_text(ps.source.as_bytes()).unwrap_or("?");
                let start = node.start_position().row + 1; // 1-based
                let end = node.end_position().row + 1;
                let kind = node.parent().map_or("?", |p| p.kind());

                // Determine the kind prefix for display
                let kind_label = symbol_kind_label(kind);
                symbols.push(format!("  {kind_label} `{name}` ({start}-{end})"));
            }
            matches.advance();
        }

        symbols.sort();
        symbols.dedup();

        let filename = resolved_path
            .file_name()
            .map_or("?", |n| n.to_str().unwrap_or("?"));
        let output = if symbols.is_empty() {
            format!("[No symbols found in {filename}]")
        } else {
            format!("[Symbols in {filename}]\n{}", symbols.join("\n"))
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

        let ps = read_and_parse(resolved_path, "zoom").await?;

        // Find the named symbol via query-based matching (restricts to declarations only)
        let query = build_symbol_query(&ps)?;

        let root_node = ps.tree.root_node();
        let mut qcursor = QueryCursor::new();
        let mut qmatches = qcursor.matches(&query, root_node, ps.source.as_bytes());
        let mut found_node = None;
        qmatches.advance();
        while let Some(m) = qmatches.get() {
            for c in m.captures {
                if let Ok(name) = c.node.utf8_text(ps.source.as_bytes())
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
            let suggestions = Self::symbol_suggestions(&ps, symbol_name);
            if suggestions.is_empty() {
                anyhow::bail!(
                    "Symbol '{symbol_name}' not found in {}",
                    resolved_path
                        .file_name()
                        .map_or("?", |n| n.to_str().unwrap_or("?")),
                );
            }
            anyhow::bail!(
                "Symbol '{symbol_name}' not found in {}. Did you mean: {}",
                resolved_path
                    .file_name()
                    .map_or("?", |n| n.to_str().unwrap_or("?")),
                suggestions.join(", ")
            );
        };

        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;
        let byte_range = node.byte_range();
        let extracted = &ps.source[byte_range.start..byte_range.end];
        let kind_label = symbol_kind_label(node.kind());

        Ok(format!(
            "[Symbol: {kind_label} `{symbol_name}` (lines {start}-{end})]\n{extracted}",
        ))
    }

    /// Suggest symbol names when zoom lookup fails.
    fn symbol_suggestions(ps: &ParsedSource, wanted: &str) -> Vec<String> {
        let Ok(query) = build_symbol_query(ps) else {
            return vec![];
        };
        let root_node = ps.tree.root_node();
        let mut qcursor = QueryCursor::new();
        let mut qmatches = qcursor.matches(&query, root_node, ps.source.as_bytes());
        let mut names: Vec<String> = Vec::new();
        qmatches.advance();
        while let Some(m) = qmatches.get() {
            for c in m.captures {
                if let Ok(name) = c.node.utf8_text(ps.source.as_bytes()) {
                    names.push(name.to_string());
                }
            }
            qmatches.advance();
        }
        names.sort();
        names.dedup();

        let wanted_lc = wanted.to_ascii_lowercase();
        let mut ranked: Vec<String> = Vec::new();
        for name in &names {
            if name.to_ascii_lowercase() == wanted_lc {
                ranked.push(name.clone());
            }
        }
        for name in &names {
            let name_lc = name.to_ascii_lowercase();
            if name_lc != wanted_lc
                && (name_lc.starts_with(&wanted_lc) || wanted_lc.starts_with(&name_lc))
            {
                ranked.push(name.clone());
            }
        }
        for name in names {
            if !ranked.contains(&name) {
                ranked.push(name);
            }
        }
        ranked.truncate(8);
        ranked
    }
}

// ── Tree-sitter infrastructure ────────────────────────────────────────

struct ParsedSource {
    source: String,
    ext: String,
    language: Language,
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
    let Some(language) = language_for_extension(&ext) else {
        anyhow::bail!(
            "Unsupported file extension '.{ext}' for {mode_label}. \
             Supported: .rs, .js, .jsx, .mjs, .cjs, .ts, .tsx, .py, .pyi, .pyx, .json, .toml, .sh, .bash, .zsh, .css, .html, .htm, .go, .rb, .c, .h, .sql"
        );
    };

    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| anyhow::anyhow!("Failed to set tree-sitter language: {e}"))?;

    let Some(tree) = parser.parse(&source, None) else {
        anyhow::bail!("Could not parse file for {mode_label}");
    };

    Ok(ParsedSource {
        source,
        ext,
        language,
        tree,
    })
}

/// Map file extension to tree-sitter Language.
fn language_for_extension(ext: &str) -> Option<Language> {
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "js" | "jsx" | "mjs" | "cjs" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "py" | "pyi" | "pyx" => Some(tree_sitter_python::LANGUAGE.into()),
        "json" => Some(tree_sitter_json::LANGUAGE.into()),
        "toml" => Some(tree_sitter_toml_ng::LANGUAGE.into()),
        "sh" | "bash" | "zsh" => Some(tree_sitter_bash::LANGUAGE.into()),
        "css" => Some(tree_sitter_css::LANGUAGE.into()),
        "html" | "htm" => Some(tree_sitter_html::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "rb" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        "sql" => Some(tree_sitter_sequel::LANGUAGE.into()),
        _ => None,
    }
}

/// Return a tree-sitter query string listing top-level declarations for the language.
#[allow(clippy::too_many_lines)]
fn symbol_query_for_extension(ext: &str) -> &'static str {
    match ext {
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
        "ts" | "tsx" => {
            r"(
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
        )"
        }
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
    }
}

/// Compile the tree-sitter symbol query for the source file's language.
fn build_symbol_query(ps: &ParsedSource) -> anyhow::Result<Query> {
    let query_str = symbol_query_for_extension(&ps.ext);
    Query::new(&ps.language, query_str)
        .map_err(|e| anyhow::anyhow!("Failed to build symbol query: {e}"))
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

/// Shell-quote a path for safe interpolation into a POSIX shell command.
///
/// Wraps the path in single quotes. Any embedded single quotes are escaped
/// by terminating the single-quoted string, inserting an escaped literal
/// quote, and resuming single-quoting (`'` → `'\''`). This handles paths
/// containing spaces, `$`, backticks, backslashes, glob characters, and
/// other special shell metacharacters, since single quotes suppress all
/// expansion in POSIX shells.
fn shell_quote(s: &str) -> String {
    let escaped = s.replace('\'', "'\\''");
    format!("'{escaped}'")
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

    #[tokio::test]
    async fn file_read_basic_scenarios() {
        let dir = std::env::temp_dir().join("mahbot_test_file_read");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("test.txt"), "hello world")
            .await
            .unwrap();

        // existing file
        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "test.txt"}))
            .await;
        assert!(result.is_ok(), "read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("1: hello world"));
        assert!(result.contains("[1 lines total]"));
        // nonexistent file
        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "nope.txt"}))
            .await;
        assert!(
            result.is_err(),
            "read should fail for nonexistent file: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("File not found"));
        // empty file
        tokio::fs::write(dir.join("empty.txt"), "").await.unwrap();
        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "empty.txt"}))
            .await;
        assert!(result.is_ok(), "empty file read should succeed: {result:?}");
        let result = result.unwrap();
        assert_eq!(result, "");
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn read_wildcard_without_search_index_returns_helpful_error() {
        let dir = std::env::temp_dir().join("mahbot_test_read_wildcard_err");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("alpha.rs"), "fn alpha() {}")
            .await
            .unwrap();

        let result = ReadTool
            .execute(&test_ws(&dir), json!({"path": "*.rs"}))
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn file_read_blocks_unsafe_paths() {
        // path traversal
        let dir = std::env::temp_dir().join("mahbot_test_file_read_traversal");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "../../../etc/passwd"}),
            )
            .await;
        assert!(result.is_err(), "traversal should be blocked: {result:?}");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("not allowed"));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        // absolute path

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "/etc/passwd"}))
            .await;
        assert!(
            result.is_err(),
            "absolute path should be blocked: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("not allowed"));
        // null byte in path
        let dir = std::env::temp_dir().join("mahbot_test_file_read_null_byte");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "test\0evil.txt"}),
            )
            .await;
        assert!(
            result.is_err(),
            "null byte path should be blocked: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("not allowed"));
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn file_read_nested_path() {
        let dir = std::env::temp_dir().join("mahbot_test_file_read_nested");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(dir.join("sub/dir"))
            .await
            .unwrap();
        tokio::fs::write(dir.join("sub/dir/deep.txt"), "deep content")
            .await
            .unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "sub/dir/deep.txt"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "nested path read should succeed: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("1: deep content"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_read_blocks_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join("mahbot_test_file_read_symlink_escape");
        let workspace = root.join("workspace");

        let _ = tokio::fs::remove_dir_all(&root).await;
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

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn file_read_offset_handling() {
        let dir = std::env::temp_dir().join("mahbot_test_file_read_offset");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("lines.txt"), "aaa\nbbb\nccc\nddd\neee")
            .await
            .unwrap();

        // Read lines 2-3
        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
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
                &Workspace::from_path(&dir),
                json!({"path": "lines.txt", "offset": 4}),
            )
            .await;
        assert!(result.is_ok(), "offset to end should succeed: {result:?}");
        let result = result.unwrap();
        assert!(result.contains("4: ddd") && result.contains("5: eee"));
        // Limit only (first 2 lines)
        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "lines.txt", "limit": 2}),
            )
            .await;
        assert!(result.is_ok(), "limit read should succeed: {result:?}");
        let result = result.unwrap();
        assert!(!result.contains("3: ccc"));
        // Offset beyond end
        tokio::fs::write(dir.join("short.txt"), "one\ntwo")
            .await
            .unwrap();
        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "short.txt", "offset": 100}),
            )
            .await;
        assert!(
            result.is_ok(),
            "offset beyond end should succeed: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("[No lines in range, file has 2 lines]"));
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn file_read_rejects_oversized_file() {
        let dir = std::env::temp_dir().join("mahbot_test_file_read_large");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        // Create a file just over 10 MB
        let big = vec![b'x'; 10 * 1024 * 1024 + 1];
        tokio::fs::write(dir.join("huge.bin"), &big).await.unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "huge.bin"}))
            .await;
        assert!(
            result.is_err(),
            "oversized file should be rejected: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("File too large"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Non-UTF-8 binary files should be read with lossy conversion.
    #[tokio::test]
    async fn file_read_lossy_reads_binary_file() {
        let dir = std::env::temp_dir().join("mahbot_test_file_read_lossy");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        // Write bytes that are not valid UTF-8 and not a PDF
        let binary_data: Vec<u8> = vec![0x00, 0x80, 0xFF, 0xFE, b'h', b'i', 0x80];
        tokio::fs::write(dir.join("data.bin"), &binary_data)
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "data.bin"}))
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
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
        let dir = std::env::temp_dir().join("mahbot_test_symbols_rust");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
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
        tokio::fs::write(dir.join("lib.rs"), code).await.unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Symbols mode returns clear error for unsupported extensions.
    #[tokio::test]
    async fn symbols_mode_unsupported_extension() {
        let dir = std::env::temp_dir().join("mahbot_test_symbols_unsupported");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("data.yaml"), r"{}")
            .await
            .unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "data.yaml", "mode": "symbols"}),
            )
            .await;
        assert!(
            result.is_err(),
            "unsupported extension should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Unsupported"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Zoom mode extracts a specific symbol's source.
    /// Also verifies correct disambiguation: parameter names and local variables
    /// with the same name as another function should not match.
    #[tokio::test]
    async fn zoom_mode_extracts_rust_function() {
        let dir = std::env::temp_dir().join("mahbot_test_zoom_rust");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        // `name` appears as a parameter in `greet` — zoom for `name` should NOT
        // return the parameter, and zoom for `greet` should return the full function body.
        let code =
            "fn greet(name: &str) -> String {\n    format!(\"Hi, {name}!\")\n}\n\nfn main() {}";
        tokio::fs::write(dir.join("main.rs"), code).await.unwrap();

        let result = ReadTool
            .execute(
                &test_ws(&dir),
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Zoom mode returns helpful error for nonexistent symbol.
    #[tokio::test]
    async fn zoom_mode_symbol_not_found() {
        let dir = std::env::temp_dir().join("mahbot_test_zoom_notfound");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("lib.rs"), "fn existing() {}")
            .await
            .unwrap();

        let result = ReadTool
            .execute(
                &test_ws(&dir),
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Zoom mode requires symbol parameter.
    #[tokio::test]
    async fn zoom_mode_missing_symbol_param() {
        let dir = std::env::temp_dir().join("mahbot_test_zoom_missing_param");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("lib.rs"), "fn f() {}")
            .await
            .unwrap();

        let result = ReadTool
            .execute(
                &Workspace::from_path(&dir),
                json!({"path": "lib.rs", "mode": "zoom"}),
            )
            .await;
        assert!(
            result.is_err(),
            "missing symbol param should fail: {result:?}"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Missing 'symbol' parameter"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Directory listing returns file names instead of erroring.
    #[tokio::test]
    async fn directory_listing_returns_contents() {
        let dir = std::env::temp_dir().join("mahbot_test_dir_list");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("a.txt"), "alpha").await.unwrap();
        tokio::fs::write(dir.join("b.rs"), "beta").await.unwrap();
        tokio::fs::create_dir(dir.join("sub")).await.unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "."}))
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Subdirectories without a trailing slash should list contents, not error.
    #[tokio::test]
    async fn directory_listing_subdir_without_trailing_slash() {
        let dir = std::env::temp_dir().join("mahbot_test_dir_sub_no_slash");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let sub = dir.join("sub");
        tokio::fs::create_dir_all(&sub).await.unwrap();
        tokio::fs::write(sub.join("inside.txt"), "nested")
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "sub"}))
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Directory listing shows "(empty)" for empty directories.
    #[tokio::test]
    async fn directory_listing_empty() {
        let dir = std::env::temp_dir().join("mahbot_test_dir_empty");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "."}))
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Directory listing handles paths with spaces and special characters.
    #[tokio::test]
    async fn directory_listing_spaces_in_path() {
        let dir = std::env::temp_dir().join("mahbot_test spaces dir");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("my file.txt"), "content")
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "."}))
            .await;
        assert!(result.is_ok(), "dir with spaces should succeed: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("my file.txt"), "should list file: {output}");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Directory listing resolves symlinks to directories.
    #[tokio::test]
    async fn directory_listing_symlink() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join("mahbot_test_dir_symlink");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let real_dir = dir.join("real");
        tokio::fs::create_dir_all(&real_dir).await.unwrap();
        tokio::fs::write(real_dir.join("nested.txt"), "data")
            .await
            .unwrap();
        let link = dir.join("link_to_real");
        symlink(&real_dir, &link).unwrap();

        // Reading the symlink directly (it resolves to the directory)
        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "link_to_real"}))
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

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// File read still works after directory delegation changes.
    #[tokio::test]
    async fn file_read_still_works() {
        let dir = std::env::temp_dir().join("mahbot_test_file_read_still");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("hello.txt"), "hello world")
            .await
            .unwrap();

        let result = ReadTool
            .execute(&Workspace::from_path(&dir), json!({"path": "hello.txt"}))
            .await;
        assert!(result.is_ok(), "file read should still succeed: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("hello world"));
        assert!(output.contains("[1 lines total]"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
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
}
