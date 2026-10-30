//! Tree-sitter syntax highlighting for diff code lines and the code editor.
//!
//! Provides per-language highlight queries that classify token spans into
//! categories (keyword, string, comment, type, function, number, operator).
//!
//! Diff view: files are parsed in their entirety — the caller reads the old
//! version (via `git show HEAD:<path>`) and new version (from disk), then
//! passes the complete source to [`parse_file_highlights`] which distributes
//! tree-sitter captures across lines. This preserves multi-line context
//! for block comments, docstrings, and multi-line strings.
//!
//! Editor: [`parse_highlights`] classifies a single in-memory buffer the
//! same way for the live editor widget.

use iced::Color;
use std::collections::HashMap;
use std::sync::OnceLock;
use strum::EnumCount;
use strum::VariantArray;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};
use tree_sitter_md::{INLINE_LANGUAGE as MD_INLINE_LANG, MarkdownParser, MarkdownTree};
// Built-in highlight queries for new languages.
use tree_sitter_bash::HIGHLIGHT_QUERY as BASH_HIGHLIGHT_QUERY;
use tree_sitter_c::HIGHLIGHT_QUERY as C_HIGHLIGHT_QUERY;
use tree_sitter_css::HIGHLIGHTS_QUERY as CSS_HIGHLIGHTS_QUERY;
use tree_sitter_go::HIGHLIGHTS_QUERY as GO_HIGHLIGHTS_QUERY;
use tree_sitter_html::HIGHLIGHTS_QUERY as HTML_HIGHLIGHTS_QUERY;
use tree_sitter_json::HIGHLIGHTS_QUERY as JSON_HIGHLIGHTS_QUERY;
use tree_sitter_md::HIGHLIGHT_QUERY_BLOCK as MD_HIGHLIGHT_QUERY_BLOCK;
use tree_sitter_md::HIGHLIGHT_QUERY_INLINE as MD_HIGHLIGHT_QUERY_INLINE;
use tree_sitter_ruby::HIGHLIGHTS_QUERY as RUBY_HIGHLIGHTS_QUERY;
use tree_sitter_sequel::HIGHLIGHTS_QUERY as SQL_HIGHLIGHTS_QUERY;
use tree_sitter_toml_ng::HIGHLIGHTS_QUERY as TOML_HIGHLIGHTS_QUERY;

use super::text_rendering::MAX_HIGHLIGHT_SIZE;
use super::theme;

// ── Compiled query cache — avoids re-parsing query patterns per line ───

// Note: MARKDOWN_INLINE_QUERY is not part of this array because the inline
// grammar is a separate tree-sitter language (MD_INLINE_LANG) not tied to
// the HighlightLanguage::Markdown variant (which uses the block grammar).
static MARKDOWN_INLINE_QUERY: OnceLock<Option<Query>> = OnceLock::new();

/// Per-language highlight query cache, indexed by `#[repr(usize)]`
/// discriminant of [`HighlightLanguage`]. Initialised lazily on first
/// access via [`cached_query`]. Array size is automatically determined
/// from the enum variant count via `strum::EnumCount`.
static QUERIES: [OnceLock<Option<Query>>; HighlightLanguage::COUNT] =
    [const { OnceLock::new() }; HighlightLanguage::COUNT];

/// Get (or compile) the highlight query for a language.
/// Returns None if the query is invalid (should not happen with baked-in queries).
fn cached_query(lang: HighlightLanguage) -> Option<&'static Query> {
    let cell = &QUERIES[lang as usize];
    cell.get_or_init(|| {
        let (ts_lang, query_str) = lang.language_and_query();
        Query::new(&ts_lang, query_str).ok()
    })
    .as_ref()
}

/// A color-coded span within a single line of code.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlightSpan {
    /// Byte offset within the line (not the file).
    pub start: usize,
    /// Byte offset within the line (exclusive).
    pub end: usize,
    /// The CSS-like class name for styling.
    pub highlight_class: HighlightClass,
}

/// Per-line highlight spans for a complete parsed file.
///
/// `spans[n]` gives the highlight spans for line `n` (0-based).
/// Byte offsets in each span are relative to the start of that line.
#[derive(Debug, Clone)]
pub struct FileHighlights {
    pub spans: Vec<Vec<HighlightSpan>>,
}

impl FileHighlights {
    /// Empty highlights — every line gets an empty vec (caller renders as plain text).
    #[must_use]
    pub fn empty(line_count: usize) -> Self {
        Self {
            spans: vec![Vec::new(); line_count],
        }
    }
}

/// Categories of highlighted tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HighlightClass {
    Keyword,
    /// String literals
    String,
    /// Type names (struct, enum, class, trait names both in definition and use)
    Type,
    /// Function/method names
    Function,
    /// Comments (single-line and block)
    Comment,
    /// Numeric literals
    Number,
    /// Operators and punctuation
    Operator,
    /// Default — no special highlighting
    Text,
}

impl HighlightClass {
    /// Map to a theme color.
    ///
    /// Uses a neutral/muted palette inspired by the ayu dark theme (Zed editor).
    /// All values are opaque — no alpha blending is applied here.
    #[must_use]
    pub const fn color(self) -> Color {
        match self {
            // Ayu dark keyword: #D580FF (purple)
            HighlightClass::Keyword => Color::from_rgb(0.835, 0.502, 1.0),
            // Ayu dark string: #C2D94C (olive green)
            HighlightClass::String => Color::from_rgb(0.761, 0.851, 0.298),
            // Ayu dark type: #59C2FF (sky blue)
            HighlightClass::Type => Color::from_rgb(0.349, 0.761, 1.0),
            // Ayu dark function: #FFB454 (warm orange)
            HighlightClass::Function => Color::from_rgb(1.0, 0.706, 0.329),
            // Ayu dark comment: #5A6673 (muted blue-gray)
            HighlightClass::Comment => Color::from_rgb(0.353, 0.400, 0.451),
            // Ayu dark number: #5CCFFF (light cyan)
            HighlightClass::Number => Color::from_rgb(0.361, 0.812, 1.0),
            // Ayu dark operator: #F29668 (peach)
            HighlightClass::Operator => Color::from_rgb(0.949, 0.588, 0.408),
            // Default text — matches the dashboard's primary text color
            HighlightClass::Text => theme::TEXT_PRIMARY,
        }
    }
}

/// Parse file highlights with a fresh tree-sitter parser, unless the
/// content exceeds [`super::text_rendering::MAX_HIGHLIGHT_SIZE`]
/// (returns `None` so callers fall back to plain text rendering).
/// Shared entry point for the editor and diff widgets.
#[must_use]
pub(crate) fn parse_highlights(content: &str, lang: HighlightLanguage) -> Option<FileHighlights> {
    if content.len() > MAX_HIGHLIGHT_SIZE {
        return None;
    }
    let mut parser = Parser::new();
    Some(parse_file_highlights(&mut parser, content, lang))
}

/// Parse an entire source file and return per-line highlight spans.
///
/// Parses the complete file with tree-sitter and distributes capture spans
/// across lines. The returned `FileHighlights` maps line index (0-based)
/// to spans with byte offsets relative to each line's start.
///
/// Lines with no captures get an empty `Vec` — the caller should render
/// those lines as plain text.
///
/// For Markdown, delegates to `parse_markdown_highlights` which uses
/// the dual-grammar (block + inline) parsing approach.
#[must_use]
pub(crate) fn parse_file_highlights(
    parser: &mut Parser,
    source: &str,
    lang: HighlightLanguage,
) -> FileHighlights {
    if lang == HighlightLanguage::Markdown {
        return parse_markdown_highlights(source);
    }

    let ts_lang = lang.language_and_query().0;
    let _ = parser.set_language(&ts_lang);

    let tree = parser.parse(source, None);
    let Some(tree) = tree else {
        return FileHighlights::empty(source.lines().count());
    };

    let Some(query_obj) = cached_query(lang) else {
        return FileHighlights::empty(source.lines().count());
    };

    build_highlights_from_tree(&tree, source, query_obj)
}

/// Parse a Markdown source file using the dual-grammar approach.
///
/// 1. Parse with block grammar → collect block captures.
/// 2. Walk block tree for inline content nodes → re-parse each range
///    with inline grammar → collect inline captures.
/// 3. Both capture sets are sorted together and distributed to lines.
///    When block and inline spans overlap, the distribution loop
///    pushes both; the inline span typically comes later in sorted
///    order and wins visually when colors differ.
#[must_use]
fn parse_markdown_highlights(source: &str) -> FileHighlights {
    let mut markdown_parser = MarkdownParser::default();
    let Some(markdown_tree) = markdown_parser.parse(source.as_bytes(), None) else {
        return FileHighlights::empty(source.lines().count());
    };
    build_markdown_highlights_from_tree(&markdown_tree, source)
}

/// Build per-line highlight spans from an already-parsed [`MarkdownTree`].
///
/// Runs the block highlight query, then the inline highlight query on each
/// inline content tree. Both capture sets are collected into a single
/// sorted byte-span list and distributed to lines via
/// [`distribute_byte_spans`]. Used by the editor highlighter which reuses
/// a persistent [`MarkdownParser`] across edits.
#[must_use]
fn build_markdown_highlights_from_tree(
    markdown_tree: &MarkdownTree,
    source: &str,
) -> FileHighlights {
    // ── 1. Collect block-level highlights ──────────────────────────
    let mut byte_spans: Vec<(usize, usize, HighlightClass)> = Vec::new();
    if let Some(query) = cached_query(HighlightLanguage::Markdown) {
        let block_tree = markdown_tree.block_tree();
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(query, block_tree.root_node(), source.as_bytes());
        matches.advance();
        while let Some(m) = matches.get() {
            for capture in m.captures {
                let name = capture_index_to_name(query, capture.index);
                // Skip `@none` captures (code fence content — injection is out of scope).
                if name == "none" {
                    continue;
                }
                byte_spans.push((
                    capture.node.start_byte(),
                    capture.node.end_byte(),
                    capture_class(name),
                ));
            }
            matches.advance();
        }
    }

    // ── 2. Collect inline-level highlights ─────────────────────────
    let inline_query = MARKDOWN_INLINE_QUERY.get_or_init(|| {
        let inline_lang: Language = MD_INLINE_LANG.into();
        Query::new(&inline_lang, MD_HIGHLIGHT_QUERY_INLINE).ok()
    });
    if let Some(query) = inline_query.as_ref() {
        for inline_tree in markdown_tree.inline_trees() {
            let root = inline_tree.root_node();
            let offset = root.start_byte();
            let mut cursor = QueryCursor::new();
            let inline_source = &source.as_bytes()[offset..root.end_byte()];
            let mut matches = cursor.matches(query, root, inline_source);
            matches.advance();
            while let Some(m) = matches.get() {
                for capture in m.captures {
                    byte_spans.push((
                        capture.node.start_byte(),
                        capture.node.end_byte(),
                        capture_class(capture_index_to_name(query, capture.index)),
                    ));
                }
                matches.advance();
            }
        }
    }

    byte_spans.sort_by_key(|(s, e, _)| (*s, *e));

    // ── 3. Distribute byte spans to lines ──────────────────────────
    distribute_byte_spans(source, &byte_spans)
}

/// Distribute byte-offset spans across lines, with gap-filling and
/// converting to line-relative offsets.
///
/// # Overlap handling
///
/// When multiple spans overlap at the same byte range (e.g. a delimiter
/// span and its parent formatting span), the span with higher paint
/// priority wins for that byte range. Delimiter spans (`Operator`) beat
/// parent emphasis/strong spans (`Function` / `Keyword`), so both opening
/// and closing `*` / `**` delimiters keep delimiter color.
#[must_use]
fn distribute_byte_spans(
    source: &str,
    byte_spans: &[(usize, usize, HighlightClass)],
) -> FileHighlights {
    // Compute byte offsets of each line start (and the past-end sentinel).
    let mut line_starts: Vec<usize> = Vec::new();
    let mut pos = 0;
    line_starts.push(0);
    for ch in source.bytes() {
        pos += 1;
        if ch == b'\n' {
            line_starts.push(pos);
        }
    }

    let mut lines: Vec<Vec<HighlightSpan>> = Vec::with_capacity(line_starts.len());

    for line_idx in 0..line_starts.len() {
        let line_start = line_starts[line_idx];
        let line_end = line_starts
            .get(line_idx + 1)
            .map_or(source.len(), |e| if *e > 0 { e - 1 } else { 0 });

        if line_start >= line_end {
            lines.push(Vec::new());
            continue;
        }

        let mut clipped: Vec<(usize, usize, HighlightClass)> = Vec::new();
        let mut boundaries: Vec<usize> = vec![line_start, line_end];

        for &(span_start, span_end, class) in byte_spans {
            if span_end <= line_start || span_start >= line_end {
                continue;
            }
            let s = span_start.max(line_start);
            let e = span_end.min(line_end);
            if s < e {
                boundaries.push(s);
                boundaries.push(e);
                clipped.push((s, e, class));
            }
        }

        boundaries.sort_unstable();
        boundaries.dedup();
        let mut line_spans: Vec<HighlightSpan> = Vec::new();

        for window in boundaries.windows(2) {
            let seg_start = window[0];
            let seg_end = window[1];
            if seg_start >= seg_end {
                continue;
            }

            let mut best_class = HighlightClass::Text;
            let mut best_pri = span_paint_priority(HighlightClass::Text);
            for &(s, e, class) in &clipped {
                if s <= seg_start && e >= seg_end {
                    let pri = span_paint_priority(class);
                    if pri < best_pri {
                        best_pri = pri;
                        best_class = class;
                    }
                }
            }

            line_spans.push(HighlightSpan {
                start: seg_start - line_start,
                end: seg_end - line_start,
                highlight_class: best_class,
            });
        }

        // Merge adjacent spans with the same class.
        let mut merged: Vec<HighlightSpan> = Vec::with_capacity(line_spans.len());
        for span in line_spans {
            if let Some(last) = merged.last_mut() {
                if (last.highlight_class == span.highlight_class) && (last.end == span.start) {
                    last.end = span.end;
                    continue;
                }
            }
            merged.push(span);
        }

        lines.push(merged);
    }

    FileHighlights { spans: lines }
}

/// Lower values win when spans overlap at the same byte offset.
const fn span_paint_priority(class: HighlightClass) -> u8 {
    match class {
        HighlightClass::Operator => 0,
        HighlightClass::Type => 1,
        HighlightClass::String => 2,
        HighlightClass::Function => 3,
        HighlightClass::Keyword => 4,
        HighlightClass::Number => 5,
        HighlightClass::Comment => 6,
        HighlightClass::Text => 255,
    }
}

/// Build per-line highlight spans from an already-parsed tree-sitter tree.
///
/// Collects capture spans from the tree and delegates to
/// [`distribute_byte_spans`] for line distribution.
#[must_use]
fn build_highlights_from_tree(
    tree: &tree_sitter::Tree,
    source: &str,
    query_obj: &Query,
) -> FileHighlights {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query_obj, tree.root_node(), source.as_bytes());

    // Collect all highlight spans as (byte_start, byte_end, class) tuples.
    let mut byte_spans: Vec<(usize, usize, HighlightClass)> = Vec::new();
    matches.advance();
    while let Some(m) = matches.get() {
        for capture in m.captures {
            byte_spans.push((
                capture.node.start_byte(),
                capture.node.end_byte(),
                capture_class(capture_index_to_name(query_obj, capture.index)),
            ));
        }
        matches.advance();
    }
    byte_spans.sort_by_key(|(s, e, _)| (*s, *e));

    distribute_byte_spans(source, &byte_spans)
}

fn capture_index_to_name(query: &Query, index: u32) -> &str {
    query
        .capture_names()
        .get(index as usize)
        .map_or("text", |s| &s[..])
}

#[expect(
    clippy::match_same_arms,
    reason = "text.reference and string.escape intentionally return Text, matching the wildcard default"
)]
fn capture_class(capture_name: &str) -> HighlightClass {
    match capture_name {
        // Keywords and keyword-like constructs
        "keyword" | "keyword.operator" | "constant" | "constant.builtin" | "boolean"
        | "conditional" | "storageclass" | "charset" | "import" | "keyframes" | "media"
        | "namespace" | "supports" | "label" => HighlightClass::Keyword,
        // Types
        "type" | "type.builtin" | "type.qualifier" | "tag" | "tag.error" | "constructor" => {
            HighlightClass::Type
        }
        // Functions, methods, properties, attributes
        "function"
        | "function.builtin"
        | "function.call"
        | "function.special"
        | "function.method"
        | "function.method.builtin"
        | "method"
        | "attribute"
        | "property"
        | "field" => HighlightClass::Function,
        // Strings (including escapes and special string variants)
        "string"
        | "string.special"
        | "string.special.key"
        | "string.special.regex"
        | "string.special.symbol"
        | "escape"
        | "embedded" => HighlightClass::String,
        "comment" => HighlightClass::Comment,
        "number" | "float" => HighlightClass::Number,
        // Operators, punctuation, and delimiters
        "operator"
        | "delimiter"
        | "punctuation.bracket"
        | "punctuation.delimiter"
        | "punctuation.special" => HighlightClass::Operator,
        // Variables, parameters, spell-check — no special highlighting
        "variable" | "variable.builtin" | "variable.parameter" | "parameter" | "spell" => {
            HighlightClass::Text
        }
        // ── Markdown block captures ──────────────────────────────────────
        // Headings — use Type for distinctive color
        "text.title" => HighlightClass::Type,
        // Code blocks / literal text — use String style
        "text.literal" => HighlightClass::String,
        // URIs — use Function style (like clickable links)
        "text.uri" => HighlightClass::Function,
        // Link labels / references — plain text
        "text.reference" => HighlightClass::Text,
        // ── Markdown inline captures ─────────────────────────────────────
        //
        // text.emphasis and text.strong only come from the inline query —
        // not the block query. The inline query is applied per-inline-tree
        // (paragraph, heading content, list item content, etc.).
        "text.emphasis" => HighlightClass::Function, // pink (italic)
        "text.strong" => HighlightClass::Keyword,    // orange (bold)
        // string.escape covers both backslash_escape and hard_line_break from
        // the inline grammar. We intentionally leave it as Text (plain) so
        // that hard_line_break rendering is not affected.
        "string.escape" => HighlightClass::Text,
        _ => HighlightClass::Text,
    }
}

// ── Language definitions ──────────────────────────────────────────────

/// Supported languages for syntax highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCount, strum::VariantArray)]
#[allow(clippy::upper_case_acronyms)]
#[repr(usize)]
pub enum HighlightLanguage {
    Rust,
    JavaScript,
    TypeScript,
    TSX,
    Python,
    Json,
    Toml,
    Bash,
    Css,
    Html,
    Go,
    Ruby,
    C,
    Sql,
    Markdown,
}

/// Lazy reverse-lookup map: tree-sitter [`Language`] → [`HighlightLanguage`].
///
/// Each grammar `Language` value is a process-lifetime singleton (pointer
/// identity), so the map is stable for the lifetime of the process.
static LANGUAGE_TO_HIGHLIGHT: OnceLock<HashMap<Language, HighlightLanguage>> = OnceLock::new();

impl HighlightLanguage {
    /// Determine language from a file extension.
    ///
    /// Delegates to the canonical mapping in
    /// [`crate::util::tree_sitter::tree_sitter_language_for_extension`],
    /// then looks up the returned tree-sitter [`Language`] in the
    /// reverse-lookup map to obtain the corresponding [`HighlightLanguage`]
    /// variant.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        let lang = crate::util::tree_sitter::tree_sitter_language_for_extension(ext)?;
        LANGUAGE_TO_HIGHLIGHT
            .get_or_init(Self::build_language_map)
            .get(&lang)
            .copied()
    }

    /// Determine language from a file path.
    #[must_use]
    pub fn from_path(path: &str) -> Option<Self> {
        std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(HighlightLanguage::from_extension)
    }

    /// Populate [`LANGUAGE_TO_HIGHLIGHT`] with the inverse of
    /// [`language_and_query`].
    ///
    /// Because this iterates over all [`HighlightLanguage`] variants and calls
    /// [`language_and_query`] for each, the map is automatically maintained
    /// when a new language is added — no manual insert calls needed.
    #[must_use]
    fn build_language_map() -> HashMap<Language, HighlightLanguage> {
        let mut map = HashMap::new();
        for variant in HighlightLanguage::VARIANTS {
            let (lang, _) = variant.language_and_query();
            map.insert(lang, *variant);
        }
        map
    }

    /// Return the tree-sitter Language and highlight query string for this language.
    ///
    /// For Markdown, returns the **block** grammar — inline Markdown uses
    /// a separate grammar (see [`MD_INLINE_LANG`] and
    /// [`MD_HIGHLIGHT_QUERY_INLINE`]).
    fn language_and_query(self) -> (Language, &'static str) {
        match self {
            HighlightLanguage::Rust => (tree_sitter_rust::LANGUAGE.into(), RUST_HIGHLIGHT_QUERY),
            HighlightLanguage::JavaScript => (
                tree_sitter_javascript::LANGUAGE.into(),
                JS_LIKE_HIGHLIGHT_QUERY,
            ),
            HighlightLanguage::TypeScript => (
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
                JS_LIKE_HIGHLIGHT_QUERY,
            ),
            HighlightLanguage::TSX => (
                tree_sitter_typescript::LANGUAGE_TSX.into(),
                JS_LIKE_HIGHLIGHT_QUERY,
            ),
            HighlightLanguage::Python => {
                (tree_sitter_python::LANGUAGE.into(), PYTHON_HIGHLIGHT_QUERY)
            }
            HighlightLanguage::Json => (tree_sitter_json::LANGUAGE.into(), JSON_HIGHLIGHTS_QUERY),
            HighlightLanguage::Toml => {
                (tree_sitter_toml_ng::LANGUAGE.into(), TOML_HIGHLIGHTS_QUERY)
            }
            HighlightLanguage::Bash => (tree_sitter_bash::LANGUAGE.into(), BASH_HIGHLIGHT_QUERY),
            HighlightLanguage::Css => (tree_sitter_css::LANGUAGE.into(), CSS_HIGHLIGHTS_QUERY),
            HighlightLanguage::Html => (tree_sitter_html::LANGUAGE.into(), HTML_HIGHLIGHTS_QUERY),
            HighlightLanguage::Go => (tree_sitter_go::LANGUAGE.into(), GO_HIGHLIGHTS_QUERY),
            HighlightLanguage::Ruby => (tree_sitter_ruby::LANGUAGE.into(), RUBY_HIGHLIGHTS_QUERY),
            HighlightLanguage::C => (tree_sitter_c::LANGUAGE.into(), C_HIGHLIGHT_QUERY),
            HighlightLanguage::Sql => (tree_sitter_sequel::LANGUAGE.into(), SQL_HIGHLIGHTS_QUERY),
            HighlightLanguage::Markdown => {
                (tree_sitter_md::LANGUAGE.into(), MD_HIGHLIGHT_QUERY_BLOCK)
            }
        }
    }
}

// ── Highlight queries ─────────────────────────────────────────────────
// These queries use tree-sitter's pattern syntax to capture tokens.
// Each capture name maps to a HighlightClass via capture_class().

const RUST_HIGHLIGHT_QUERY: &str = r#"
;; Keywords
[
  "as" "async" "await" "break" "const" "continue" "dyn"
  "else" "enum" "extern" "false" "fn" "for" "if" "impl" "in"
  "let" "loop" "match" "mod" "move" "pub" "ref" "return"
  "static" "struct" "trait" "true" "type" "unsafe"
  "use" "where" "while" "yield"
] @keyword

;; mutable_specifier for "mut"
(mutable_specifier) @keyword

;; Types
(type_identifier) @type
(primitive_type) @type
(scoped_type_identifier path: (identifier) @type)
(generic_function type_arguments: (type_arguments (type_identifier) @type))

;; Function definitions and calls
(function_item name: (identifier) @function)
(function_signature_item name: (identifier) @function)
(call_expression function: (identifier) @function.call)
(call_expression function: (field_expression field: (field_identifier) @function.call))
(macro_invocation macro: (identifier) @function.call)

;; String literals
(string_literal) @string
(raw_string_literal) @string
(char_literal) @string

;; Comments
(line_comment) @comment
(block_comment) @comment

;; Numbers
(integer_literal) @number
(float_literal) @number

;; Operators and punctuation
[
  "+" "-" "*" "/" "%" "=" "==" "!=" "<" ">" "<=" ">=" "&&" "||" "!"
  "&" "|" "^" "<<" ">>" "+=" "-=" "*=" "/=" "%=" "&=" "|=" "^=" "<<=" ">>="
  "->" "=>" "::" "." ".." "..=" ";" "," ":" "@"
] @operator
"#;

const JS_LIKE_HIGHLIGHT_QUERY: &str = r"
;; Functions
(function_declaration name: (identifier) @function)
(method_definition name: (property_identifier) @function)
(arrow_function) @function
(generator_function_declaration name: (identifier) @function)

;; Strings
(string) @string
(template_string) @string

;; Comments
(comment) @comment

;; Numbers
(number) @number
";

const PYTHON_HIGHLIGHT_QUERY: &str = r#"
;; Types (class names)
(class_definition name: (identifier) @type)

;; Functions
(function_definition name: (identifier) @function)
(call function: (identifier) @function.call)
(call function: (attribute attribute: (identifier) @function.call))

;; Strings
(string) @string
(string_start) @string
(string_content) @string
(string_end) @string

;; Comments
(comment) @comment

;; Numbers
(integer) @number
(float) @number

;; Operators
[
  "+" "-" "*" "/" "//" "%" "**" "=" "+=" "-=" "*=" "/=" "//=" "%="
  "**=" "==" "!=" "<" ">" "<=" ">=" "and" "or" "not" "is" "in"
  "&" "|" "^" "~" "<<" ">>" "@" ":=" "." ";" "," ":" "->"
] @operator
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_queries_compile() {
        // Test all standard HighlightLanguage variants via VARIANTS.
        for variant in HighlightLanguage::VARIANTS {
            let name = format!("{variant:?}");
            // Direct query compilation (doesn't go through the cache layer).
            let (lang, query) = variant.language_and_query();
            let q = tree_sitter::Query::new(&lang, query);
            assert!(q.is_ok(), "{name} query failed: {:?}", q.err());

            // Also verify it compiles through the cache layer (array-indexed lookup).
            let cached = cached_query(*variant);
            assert!(cached.is_some(), "{name} cached_query returned None");
        }

        // Inline Markdown uses a separate grammar that has no extension mapping,
        // so construct its Language directly.
        let md_inline = MD_INLINE_LANG.into();
        let q = tree_sitter::Query::new(&md_inline, MD_HIGHLIGHT_QUERY_INLINE);
        assert!(q.is_ok(), "MD inline query failed: {:?}", q.err());
    }

    /// Every canonical extension maps to a variant and every variant has at least one extension.
    #[test]
    fn test_variant_extension_roundtrip() {
        use crate::util::tree_sitter::supported_extensions;

        for ext in supported_extensions() {
            assert!(
                HighlightLanguage::from_extension(ext).is_some(),
                "extension '{ext}' is in the tree-sitter grammar table but \
                 from_extension returned None. \
                 Add a matching language_and_query arm in gui::highlight.",
            );
        }

        for variant in HighlightLanguage::VARIANTS {
            let has_extension = supported_extensions()
                .any(|ext| HighlightLanguage::from_extension(ext) == Some(*variant));
            assert!(
                has_extension,
                "HighlightLanguage variant {variant:?} has no extension mapped to it. \
                 Add at least one extension for this variant's tree-sitter Language \
                 to the grammar table in util::tree_sitter.",
            );
        }

        let cases: &[(&str, &str, &str, Option<HighlightLanguage>)] = &[
            (
                "markdown extension",
                "md",
                "README.md",
                Some(HighlightLanguage::Markdown),
            ),
            (
                "markdown long extension",
                "markdown",
                "docs/guide.markdown",
                Some(HighlightLanguage::Markdown),
            ),
            ("unsupported extension", "cpp", "main.cpp", None),
        ];
        for (name, ext, path, expected) in cases {
            assert_eq!(
                HighlightLanguage::from_extension(ext),
                *expected,
                "case {name}: from_extension({ext:?})"
            );
            assert_eq!(
                HighlightLanguage::from_path(path),
                *expected,
                "case {name}: from_path({path:?})"
            );
        }
    }

    enum Lines {
        Exact(usize),
        AtLeast(usize),
    }

    struct ParseCase {
        name: &'static str,
        code: &'static str,
        language: HighlightLanguage,
        lines: Lines,
        classes: &'static [(usize, HighlightClass)],
        line0_spans: Option<&'static [(usize, usize, HighlightClass)]>,
        line0_end_any: Option<&'static [usize]>,
    }

    impl Default for ParseCase {
        fn default() -> Self {
            Self {
                name: "",
                code: "",
                language: HighlightLanguage::Rust,
                lines: Lines::Exact(1),
                classes: &[],
                line0_spans: None,
                line0_end_any: None,
            }
        }
    }

    #[test]
    fn test_parse_file_highlights() {
        let cases: &[ParseCase] = &[
            ParseCase {
                name: "single_line",
                code: "fn main() {}",
                classes: &[(0, HighlightClass::Keyword)],
                ..ParseCase::default()
            },
            ParseCase {
                name: "rust",
                code: "fn main() {\n    let x = 42;\n}\n",
                lines: Lines::AtLeast(3),
                classes: &[(1, HighlightClass::Keyword), (1, HighlightClass::Number)],
                ..ParseCase::default()
            },
            ParseCase {
                name: "empty",
                code: "",
                lines: Lines::Exact(1),
                line0_spans: Some(&[]),
                ..ParseCase::default()
            },
            ParseCase {
                name: "python_multiline",
                code: "def foo():\n    \"\"\"A docstring.\"\"\"\n    return 1\n",
                language: HighlightLanguage::Python,
                lines: Lines::Exact(4),
                classes: &[(0, HighlightClass::Function), (1, HighlightClass::String)],
                ..ParseCase::default()
            },
            // Regression: a line with exactly one capture ("42") must not be filtered.
            ParseCase {
                name: "single_token_line_not_filtered",
                code: "42\n",
                lines: Lines::Exact(2),
                classes: &[(0, HighlightClass::Number)],
                line0_end_any: Some(&[2, 0]),
                ..ParseCase::default()
            },
        ];

        for case in cases {
            let name = case.name;
            let mut parser = Parser::new();
            let fh = parse_file_highlights(&mut parser, case.code, case.language);
            match case.lines {
                Lines::Exact(lines) => {
                    assert_eq!(fh.spans.len(), lines, "case {name}: line count");
                }
                Lines::AtLeast(lines) => {
                    assert!(fh.spans.len() >= lines, "case {name}: line count");
                }
            }
            for &(line, class) in case.classes {
                let spans = fh
                    .spans
                    .get(line)
                    .unwrap_or_else(|| panic!("case {name}: expected line {line} to exist"));
                assert!(
                    spans.iter().any(|s| s.highlight_class == class),
                    "case {name}: expected {class:?} on line {line}; spans: {:?}",
                    spans
                        .iter()
                        .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                        .collect::<Vec<_>>()
                );
            }
            if let Some(expected) = case.line0_spans {
                assert_eq!(
                    fh.spans[0]
                        .iter()
                        .map(|s| (s.start, s.end, s.highlight_class))
                        .collect::<Vec<(usize, usize, HighlightClass)>>(),
                    expected,
                    "case {name}: line 0 spans"
                );
            }
            if let Some(ends) = case.line0_end_any {
                assert_eq!(fh.spans[0].len(), 1, "case {name}: one span on line 0");
                let end = fh.spans[0][0].end;
                assert!(
                    ends.contains(&end),
                    "case {name}: single span should cover full line or be empty; end {end}"
                );
            }
        }
    }

    #[expect(clippy::too_many_lines)] // one table over every markdown flavour
    #[test]
    fn test_markdown_highlight_spans() {
        #[derive(Default)]
        struct Case {
            name: &'static str,
            input: &'static str,
            lines: usize,
            present: &'static [(usize, HighlightClass, usize, usize, &'static str)],
            absent: &'static [(usize, HighlightClass, usize, usize, &'static str)],
            line0_spans: Option<&'static [(usize, usize, HighlightClass)]>,
        }

        let cases: &[Case] = &[
            Case {
                name: "heading",
                input: "# Hello World\n\nSome text.\n",
                lines: 4,
                present: &[(0, HighlightClass::Type, 0, 13, "text.title")],
                ..Case::default()
            },
            Case {
                name: "link",
                input: "Visit [example](https://example.com) for info.\n",
                lines: 2,
                present: &[(0, HighlightClass::Function, 15, 36, "link URI")],
                ..Case::default()
            },
            Case {
                name: "empty",
                input: "",
                lines: 1,
                line0_spans: Some(&[]),
                ..Case::default()
            },
            Case {
                name: "bold_single_char",
                input: "**X**",
                lines: 1,
                present: &[
                    (0, HighlightClass::Operator, 0, 2, "opening **"),
                    (0, HighlightClass::Keyword, 2, 3, "X content"),
                    (0, HighlightClass::Operator, 3, 5, "closing **"),
                ],
                ..Case::default()
            },
            Case {
                name: "bold_star",
                input: "**bold**",
                lines: 1,
                present: &[
                    (0, HighlightClass::Operator, 0, 2, "opening **"),
                    (0, HighlightClass::Keyword, 2, 6, "bold content"),
                    (0, HighlightClass::Operator, 6, 8, "closing **"),
                ],
                ..Case::default()
            },
            Case {
                name: "bold_underscore",
                input: "__bold__",
                lines: 1,
                present: &[
                    (0, HighlightClass::Operator, 0, 2, "opening __"),
                    (0, HighlightClass::Keyword, 2, 6, "bold content"),
                    (0, HighlightClass::Operator, 6, 8, "closing __"),
                ],
                ..Case::default()
            },
            Case {
                name: "italic_star",
                input: "*italic*",
                lines: 1,
                present: &[
                    (0, HighlightClass::Operator, 0, 1, "opening *"),
                    (0, HighlightClass::Function, 1, 7, "italic content"),
                    (0, HighlightClass::Operator, 7, 8, "closing *"),
                ],
                ..Case::default()
            },
            Case {
                name: "italic_underscore",
                input: "_italic_",
                lines: 1,
                present: &[
                    (0, HighlightClass::Operator, 0, 1, "opening _"),
                    (0, HighlightClass::Function, 1, 7, "italic content"),
                    (0, HighlightClass::Operator, 7, 8, "closing _"),
                ],
                ..Case::default()
            },
            Case {
                name: "inline_code",
                input: "`code`",
                lines: 1,
                present: &[
                    (0, HighlightClass::Operator, 0, 1, "opening `"),
                    (0, HighlightClass::String, 1, 5, "code content"),
                    (0, HighlightClass::Operator, 5, 6, "closing `"),
                ],
                ..Case::default()
            },
            Case {
                name: "backslash_escape",
                input: r"\*literal",
                lines: 1,
                absent: &[
                    (0, HighlightClass::Function, 0, 2, "escaped \\*"),
                    (0, HighlightClass::Keyword, 0, 2, "escaped \\*"),
                ],
                ..Case::default()
            },
            Case {
                name: "heading_with_emphasis",
                input: "# *hello*\n",
                lines: 2,
                present: &[(0, HighlightClass::Type, 0, 17, "text.title")],
                ..Case::default()
            },
            Case {
                name: "inline_after_blank_line",
                input: "First paragraph.\n\n*emphasis after blank*\n",
                lines: 4,
                present: &[
                    (2, HighlightClass::Function, 1, 22, "emphasis content"),
                    (2, HighlightClass::Operator, 0, 1, "opening *"),
                ],
                ..Case::default()
            },
            Case {
                name: "inline_in_list_items",
                input: "- **bold item**\n- *italic item*\n",
                lines: 3,
                present: &[
                    (0, HighlightClass::Keyword, 4, 13, "bold list content"),
                    (1, HighlightClass::Function, 4, 15, "italic list content"),
                ],
                ..Case::default()
            },
            Case {
                name: "inline_after_second_paragraph",
                input: "Paragraph one.\n\nParagraph two with `code`.\n",
                lines: 4,
                present: &[(2, HighlightClass::String, 20, 24, "inline code")],
                ..Case::default()
            },
        ];

        for case in cases {
            let name = case.name;
            let fh = parse_markdown_highlights(case.input);
            assert_eq!(fh.spans.len(), case.lines, "case {name}: line count");
            for &(line, class, lo, hi, label) in case.present {
                let spans = fh
                    .spans
                    .get(line)
                    .unwrap_or_else(|| panic!("case {name}: expected line {line} to exist"));
                assert!(
                    spans
                        .iter()
                        .any(|s| s.highlight_class == class && s.start < hi && s.end > lo),
                    "case {name}: expected {label} ({class:?}) in [{lo},{hi}) on line {line}; \
                     spans: {:?}",
                    spans
                        .iter()
                        .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                        .collect::<Vec<_>>()
                );
            }
            for &(line, class, lo, hi, label) in case.absent {
                let spans = fh
                    .spans
                    .get(line)
                    .unwrap_or_else(|| panic!("case {name}: expected line {line} to exist"));
                assert!(
                    !spans
                        .iter()
                        .any(|s| s.highlight_class == class && s.start < hi && s.end > lo),
                    "case {name}: {label} must not carry {class:?} in [{lo},{hi}) on line {line}; \
                     spans: {:?}",
                    spans
                        .iter()
                        .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                        .collect::<Vec<_>>()
                );
            }
            if let Some(expected) = case.line0_spans {
                assert_eq!(
                    fh.spans[0]
                        .iter()
                        .map(|s| (s.start, s.end, s.highlight_class))
                        .collect::<Vec<(usize, usize, HighlightClass)>>(),
                    expected,
                    "case {name}: line 0 spans"
                );
            }
        }
    }

    #[test]
    fn test_distribute_byte_spans() {
        fn span(start: usize, end: usize, class: HighlightClass) -> HighlightSpan {
            HighlightSpan {
                start,
                end,
                highlight_class: class,
            }
        }

        #[derive(Default)]
        struct Case {
            name: &'static str,
            source: &'static str,
            spans: Vec<(usize, usize, HighlightClass)>,
            expected_lines: Vec<Vec<HighlightSpan>>,
        }

        let cases = vec![
            Case {
                name: "no_overlap",
                source: "abcde",
                spans: vec![
                    (0, 1, HighlightClass::Keyword),
                    (2, 5, HighlightClass::Number),
                ],
                expected_lines: vec![vec![
                    span(0, 1, HighlightClass::Keyword),
                    span(1, 2, HighlightClass::Text),
                    span(2, 5, HighlightClass::Number),
                ]],
            },
            Case {
                name: "tail_emission",
                source: "**bold**",
                spans: vec![
                    (0, 2, HighlightClass::Operator),
                    (0, 8, HighlightClass::Keyword),
                    (6, 8, HighlightClass::Operator),
                ],
                expected_lines: vec![vec![
                    span(0, 2, HighlightClass::Operator),
                    span(2, 6, HighlightClass::Keyword),
                    span(6, 8, HighlightClass::Operator),
                ]],
            },
            Case {
                name: "partial_overlap",
                source: "abcdef",
                spans: vec![
                    (1, 3, HighlightClass::String),
                    (2, 5, HighlightClass::Keyword),
                ],
                expected_lines: vec![vec![
                    span(0, 1, HighlightClass::Text),
                    span(1, 3, HighlightClass::String),
                    span(3, 5, HighlightClass::Keyword),
                    span(5, 6, HighlightClass::Text),
                ]],
            },
            Case {
                name: "multi_line_overlap",
                source: "hello\n*world*",
                spans: vec![
                    (6, 7, HighlightClass::Operator),
                    (6, 13, HighlightClass::Function),
                    (12, 13, HighlightClass::Operator),
                ],
                expected_lines: vec![
                    vec![span(0, 5, HighlightClass::Text)],
                    vec![
                        span(0, 1, HighlightClass::Operator),
                        span(1, 6, HighlightClass::Function),
                        span(6, 7, HighlightClass::Operator),
                    ],
                ],
            },
        ];

        for case in &cases {
            let name = case.name;
            let fh = distribute_byte_spans(case.source, &case.spans);
            assert_eq!(
                fh.spans.len(),
                case.expected_lines.len(),
                "case {name}: line count"
            );
            for (line_idx, expected) in case.expected_lines.iter().enumerate() {
                assert_eq!(
                    fh.spans[line_idx], *expected,
                    "case {name}: line {line_idx}"
                );
            }
        }
    }
}
