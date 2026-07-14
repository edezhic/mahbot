//! Tree-sitter syntax highlighting for diff code lines.
//!
//! Provides per-language highlight queries that classify token spans into
//! categories (keyword, string, comment, type, function, number, operator).
//!
//! Files are parsed in their entirety — the caller reads the old version
//! (via `git show HEAD:<path>`) and new version (from disk), then passes
//! the complete source to [`parse_file_highlights`] which distributes
//! tree-sitter captures across lines. This preserves multi-line context
//! for block comments, docstrings, and multi-line strings.

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
    /// Search match highlight (find/replace)
    Search,
    /// Currently focused search match
    SearchCurrent,
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
            // Search/find highlights — kept as amber (editor navigation, not syntax)
            HighlightClass::Search => Color::from_rgb(1.0, 0.667, 0.0),
            HighlightClass::SearchCurrent => Color::from_rgb(1.0, 0.8, 0.2),
        }
    }
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
        HighlightClass::Search | HighlightClass::SearchCurrent => 7,
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
            let name = format!("{:?}", variant);
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

    /// Verify that every extension in [`ALL_TREE_SITTER_EXTENSIONS`] maps to a
    /// [`HighlightLanguage`] variant and that every variant has at least one
    /// extension mapped to it.
    ///
    /// This guards against:
    /// - Adding an extension to [`tree_sitter_language_for_extension`] and
    ///   [`ALL_TREE_SITTER_EXTENSIONS`] without a matching `language_and_query`
    ///   arm (extension unmapped — `from_extension` returns `None`).
    /// - Adding a variant without any extension in
    ///   [`tree_sitter_language_for_extension`] (variant unreachable from
    ///   extension).
    ///
    /// Unlike the previous test that used a separate `VARIANT_EXTENSIONS`
    /// constant, this test derives all checks from the canonical
    /// [`ALL_TREE_SITTER_EXTENSIONS`] list, eliminating the maintenance burden
    /// of keeping a test-side copy in sync.
    ///
    /// [`ALL_TREE_SITTER_EXTENSIONS`]: crate::util::tree_sitter::ALL_TREE_SITTER_EXTENSIONS
    /// [`tree_sitter_language_for_extension`]: crate::util::tree_sitter::tree_sitter_language_for_extension
    #[test]
    fn test_variant_extension_roundtrip() {
        use crate::util::tree_sitter::ALL_TREE_SITTER_EXTENSIONS;

        // Every extension in ALL_TREE_SITTER_EXTENSIONS must map to *some* variant.
        for ext in ALL_TREE_SITTER_EXTENSIONS {
            assert!(
                HighlightLanguage::from_extension(ext).is_some(),
                "extension '{ext}' is listed in ALL_TREE_SITTER_EXTENSIONS but \
                 from_extension returned None. \
                 Either add a match arm in tree_sitter_language_for_extension, or \
                 remove '{ext}' from ALL_TREE_SITTER_EXTENSIONS.",
            );
        }

        // Every variant must have at least one extension that maps to it.
        // This catches orphaned variants that have no extension mapping.
        for variant in HighlightLanguage::VARIANTS {
            let has_extension = ALL_TREE_SITTER_EXTENSIONS
                .iter()
                .any(|ext| HighlightLanguage::from_extension(ext) == Some(*variant));
            assert!(
                has_extension,
                "HighlightLanguage variant {variant:?} has no extension mapped to it. \
                 Add at least one extension to tree_sitter_language_for_extension and \
                 ALL_TREE_SITTER_EXTENSIONS that maps to this variant's tree-sitter Language.",
            );
        }
    }

    #[test]
    fn test_unsupported_extension() {
        assert!(HighlightLanguage::from_extension("cpp").is_none());
    }

    #[test]
    fn test_line_starts_simple() {
        // Directly verify the line-splitting logic.
        let source = "fn main() {\n    let x = 42;\n}\n";
        let mut line_starts: Vec<usize> = Vec::new();
        let mut pos = 0;
        line_starts.push(0);
        for ch in source.bytes() {
            pos += 1;
            if ch == b'\n' {
                line_starts.push(pos);
            }
        }
        // source has 3 \n chars, so line_starts should have 4 entries.
        assert_eq!(
            line_starts.len(),
            4,
            "line_starts: {line_starts:?}, source len: {}",
            source.len()
        );
        // Line 0: byte range [0, 11]
        let line_end = line_starts
            .get(1)
            .map_or(source.len(), |e| e.saturating_sub(1));
        assert_eq!(line_end, 11);
        let line0 = &source[0..line_end];
        assert_eq!(line0, "fn main() {");
    }

    #[test]
    fn test_parse_file_highlights_single_line() {
        let code = "fn main() {}";
        let mut parser = Parser::new();
        let fh = parse_file_highlights(&mut parser, code, HighlightLanguage::Rust);
        assert_eq!(fh.spans.len(), 1);
        // Should find keyword "fn"
        let has_keyword = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Keyword);
        assert!(
            has_keyword,
            "spans: {:?}",
            fh.spans[0]
                .iter()
                .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_parse_file_highlights_rust() {
        let code = "fn main() {\n    let x = 42;\n}\n";
        let mut parser = Parser::new();
        let fh = parse_file_highlights(&mut parser, code, HighlightLanguage::Rust);
        assert!(fh.spans.len() >= 3, "got {} spans", fh.spans.len());
        // Line 1 ("    let x = 42;") should have keyword "let"
        let has_keyword = fh.spans[1]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Keyword);
        assert!(has_keyword, "Expected keyword 'let' on line 1");
        // Line 1 should also have a number ("42")
        let has_number = fh.spans[1]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Number);
        assert!(has_number, "Expected number '42' on line 1");
    }

    #[test]
    fn test_parse_file_highlights_empty() {
        let mut parser = Parser::new();
        let fh = parse_file_highlights(&mut parser, "", HighlightLanguage::Rust);
        // Empty source produces 1 line (the parser treats "" as a single empty line).
        assert!(fh.spans.len() <= 1);
        // Empty source should have empty spans (no captures).
        if !fh.spans.is_empty() {
            assert!(
                fh.spans[0].is_empty(),
                "empty source should have no captures"
            );
        }
    }

    #[test]
    fn test_parse_file_highlights_python_multiline() {
        let code = "def foo():\n    \"\"\"A docstring.\"\"\"\n    return 1\n";
        let mut parser = Parser::new();
        let fh = parse_file_highlights(&mut parser, code, HighlightLanguage::Python);
        // "def foo():" / "    \"\"\"A docstring.\"\"\"" / "    return 1" / empty trailing
        assert_eq!(fh.spans.len(), 4);
    }

    #[test]
    fn test_file_highlights_empty_constructor() {
        let fh = FileHighlights::empty(3);
        assert_eq!(fh.spans.len(), 3);
        // Each line should have an empty vec.
        for span_vec in &fh.spans {
            assert!(
                span_vec.is_empty(),
                "empty() should produce empty vecs, got {span_vec:?}"
            );
        }
    }

    #[test]
    fn test_single_token_line_not_filtered() {
        // Regression: line with exactly one capture ("42") must not be filtered.
        let code = "42\n";
        let mut parser = Parser::new();
        let fh = parse_file_highlights(&mut parser, code, HighlightLanguage::Rust);
        assert_eq!(
            fh.spans.len(),
            2,
            "should have 2 lines (42 + trailing empty)"
        );
        // Line 0: "42" — should have a Number span.
        let has_number = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Number);
        assert!(
            has_number,
            "single token line '42' should have Number highlighting, got: {:?}",
            fh.spans[0]
                .iter()
                .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                .collect::<Vec<_>>()
        );
        // If we have exactly 1 span, it should cover the full line.
        if fh.spans[0].len() == 1 {
            let s = &fh.spans[0][0];
            assert!(
                s.end == 2 || s.end == 0,
                "single span should cover full line or be empty"
            );
        }
    }

    #[test]
    fn test_markdown_extension() {
        assert_eq!(
            HighlightLanguage::from_extension("md"),
            Some(HighlightLanguage::Markdown)
        );
        assert_eq!(
            HighlightLanguage::from_extension("markdown"),
            Some(HighlightLanguage::Markdown)
        );
    }

    #[test]
    fn test_markdown_path() {
        assert_eq!(
            HighlightLanguage::from_path("README.md"),
            Some(HighlightLanguage::Markdown)
        );
        assert_eq!(
            HighlightLanguage::from_path("docs/guide.markdown"),
            Some(HighlightLanguage::Markdown)
        );
    }

    #[test]
    fn test_parse_markdown_highlights_heading() {
        let code = "# Hello World\n\nSome text.\n";
        let fh = parse_markdown_highlights(code);
        assert!(
            fh.spans.len() >= 3,
            "expected at least 3 lines, got {}",
            fh.spans.len()
        );
        // First line should have a text.title highlight (mapped to Type)
        let has_title = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Type);
        assert!(
            has_title,
            "expected text.title (Type) on heading line, got: {:?}",
            fh.spans[0]
                .iter()
                .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_parse_markdown_highlights_link() {
        let code = "Visit [example](https://example.com) for info.\n";
        let fh = parse_markdown_highlights(code);
        let has_uri = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Function);
        assert!(
            has_uri,
            "expected URI (Function) highlight in link, got: {:?}",
            fh.spans[0]
                .iter()
                .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_parse_markdown_highlights_empty() {
        let code = "";
        let fh = parse_markdown_highlights(code);
        // Empty source produces 1 line (consistent with other language parsers).
        assert_eq!(fh.spans.len(), 1);
        assert!(
            fh.spans[0].is_empty(),
            "empty source should have no captures"
        );
    }

    // ── Markdown inline formatting tests ────────────────────────────

    /// Assert that `fh.spans[0]` contains at least one span of `class` whose
    /// byte range **overlaps** `[lo, hi)` (i.e. `s.start < hi && s.end > lo`).
    /// This is an overlap check, not a strict-coverage check — the span may
    /// extend beyond `[lo, hi)`. All test callers pass ranges that exactly
    /// match the expected span positions, so overlap is equivalent to coverage
    /// in practice.
    fn line0_has_class_in_range(
        fh: &FileHighlights,
        class: HighlightClass,
        lo: usize,
        hi: usize,
        label: &str,
    ) {
        let found = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == class && s.start < hi && s.end > lo);
        assert!(
            found,
            "expected {label} ({class:?}) in [{lo},{hi}) on line 0; spans: {:?}",
            fh.spans[0]
                .iter()
                .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                .collect::<Vec<_>>()
        );
    }

    fn line_has_class_in_range(
        fh: &FileHighlights,
        line: usize,
        class: HighlightClass,
        lo: usize,
        hi: usize,
        label: &str,
    ) {
        let spans = fh
            .spans
            .get(line)
            .unwrap_or_else(|| panic!("expected line {line} to exist"));
        let found = spans
            .iter()
            .any(|s| s.highlight_class == class && s.start < hi && s.end > lo);
        assert!(
            found,
            "expected {label} ({class:?}) in [{lo},{hi}) on line {line}; spans: {:?}",
            spans
                .iter()
                .map(|s| format!("({},{},{:?})", s.start, s.end, s.highlight_class))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_markdown_formatting() {
        // Table-driven test consolidating 6 individual inline formatting tests.
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: Vec<(HighlightClass, usize, usize, &'static str)>,
        }

        let cases = vec![
            Case {
                name: "test_markdown_bold_single_char",
                input: "**X**",
                expected: vec![
                    (HighlightClass::Operator, 0, 2, "opening **"),
                    (HighlightClass::Keyword, 2, 3, "X content"),
                    (HighlightClass::Operator, 3, 5, "closing **"),
                ],
            },
            Case {
                name: "test_markdown_bold_star",
                input: "**bold**",
                expected: vec![
                    (HighlightClass::Operator, 0, 2, "opening **"),
                    (HighlightClass::Keyword, 2, 6, "bold content"),
                    (HighlightClass::Operator, 6, 8, "closing **"),
                ],
            },
            Case {
                name: "test_markdown_bold_underscore",
                input: "__bold__",
                expected: vec![
                    (HighlightClass::Operator, 0, 2, "opening __"),
                    (HighlightClass::Keyword, 2, 6, "bold content"),
                    (HighlightClass::Operator, 6, 8, "closing __"),
                ],
            },
            Case {
                name: "test_markdown_italic_star",
                input: "*italic*",
                expected: vec![
                    (HighlightClass::Operator, 0, 1, "opening *"),
                    (HighlightClass::Function, 1, 7, "italic content"),
                    (HighlightClass::Operator, 7, 8, "closing *"),
                ],
            },
            Case {
                name: "test_markdown_italic_underscore",
                input: "_italic_",
                expected: vec![
                    (HighlightClass::Operator, 0, 1, "opening _"),
                    (HighlightClass::Function, 1, 7, "italic content"),
                    (HighlightClass::Operator, 7, 8, "closing _"),
                ],
            },
            Case {
                name: "test_markdown_inline_code",
                input: "`code`",
                expected: vec![
                    (HighlightClass::Operator, 0, 1, "opening `"),
                    (HighlightClass::String, 1, 5, "code content"),
                    (HighlightClass::Operator, 5, 6, "closing `"),
                ],
            },
        ];

        for case in &cases {
            let fh = parse_markdown_highlights(case.input);
            assert_eq!(
                fh.spans.len(),
                1,
                "case: {} — expected single line",
                case.name
            );
            for &(class, lo, hi, label) in &case.expected {
                line0_has_class_in_range(&fh, class, lo, hi, label);
            }
        }
    }

    #[test]
    fn test_markdown_backslash_escape() {
        let code = r"\*literal";
        let fh = parse_markdown_highlights(code);
        assert_eq!(fh.spans.len(), 1);
        // backslash_escape → string.escape → Text (unchanged).
        // The escaped asterisk should not have emphasis colour — just plain Text.
        // Assert that no Function or Keyword span overlaps the escape range [0, 2).
        let escape_has_bad_class = fh.spans[0].iter().any(|s| {
            s.start < 2
                && s.end > 0
                && (s.highlight_class == HighlightClass::Function
                    || s.highlight_class == HighlightClass::Keyword)
        });
        assert!(
            !escape_has_bad_class,
            "escaped `\\*` should not have emphasis/bold colour; spans: {:?}",
            fh.spans[0]
        );
    }

    #[test]
    fn test_markdown_heading_with_emphasis() {
        let code = "# *hello*\n";
        let fh = parse_markdown_highlights(code);
        assert!(!fh.spans.is_empty(), "at least heading line");
        // Block-level text.title (Type) subsumes emphasis due to stable sort
        // order (block captures come before inline in the flat span list).
        let has_title = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Type);
        assert!(has_title, "heading should have Type (text.title) highlight");
        // Emphasis inside headings is subsumed by block-level text.title
        // (Type) due to stable sort order — this is an inherent limitation
        // of the flat span model, not a regression from this fix.
    }

    #[test]
    fn test_markdown_combined_heading_bold() {
        let code = "## **important**\n";
        let fh = parse_markdown_highlights(code);
        assert!(!fh.spans.is_empty());
        // Block heading has Type.
        let has_type = fh.spans[0]
            .iter()
            .any(|s| s.highlight_class == HighlightClass::Type);
        assert!(has_type, "heading should have Type");
        // Since text.title subsumes text.strong, we won't see Keyword independently
        // on the heading line — but the important thing is that nothing regresses.
    }

    #[test]
    fn test_markdown_inline_after_blank_line() {
        let code = "First paragraph.\n\n*emphasis after blank*\n";
        let fh = parse_markdown_highlights(code);
        assert!(
            fh.spans.len() >= 3,
            "expected paragraph, blank, and emphasis lines; got {}",
            fh.spans.len()
        );
        // Line 2: "*emphasis after blank*"
        line_has_class_in_range(&fh, 2, HighlightClass::Function, 1, 22, "emphasis content");
        line_has_class_in_range(&fh, 2, HighlightClass::Operator, 0, 1, "opening *");
    }

    #[test]
    fn test_markdown_inline_in_list_items() {
        let code = "- **bold item**\n- *italic item*\n";
        let fh = parse_markdown_highlights(code);
        assert!(fh.spans.len() >= 2, "expected two list lines");
        line_has_class_in_range(&fh, 0, HighlightClass::Keyword, 4, 13, "bold list content");
        line_has_class_in_range(
            &fh,
            1,
            HighlightClass::Function,
            4,
            15,
            "italic list content",
        );
    }

    #[test]
    fn test_markdown_inline_after_second_paragraph() {
        let code = "Paragraph one.\n\nParagraph two with `code`.\n";
        let fh = parse_markdown_highlights(code);
        assert!(fh.spans.len() >= 3);
        // Line 2 contains inline code in the second paragraph.
        line_has_class_in_range(&fh, 2, HighlightClass::String, 20, 24, "inline code");
    }

    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_distribute_byte_spans() {
        // Table-driven test consolidating 4 distribute_byte_spans tests.
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
                    HighlightSpan {
                        start: 0,
                        end: 1,
                        highlight_class: HighlightClass::Keyword,
                    },
                    HighlightSpan {
                        start: 1,
                        end: 2,
                        highlight_class: HighlightClass::Text,
                    },
                    HighlightSpan {
                        start: 2,
                        end: 5,
                        highlight_class: HighlightClass::Number,
                    },
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
                    HighlightSpan {
                        start: 0,
                        end: 2,
                        highlight_class: HighlightClass::Operator,
                    },
                    HighlightSpan {
                        start: 2,
                        end: 6,
                        highlight_class: HighlightClass::Keyword,
                    },
                    HighlightSpan {
                        start: 6,
                        end: 8,
                        highlight_class: HighlightClass::Operator,
                    },
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
                    HighlightSpan {
                        start: 0,
                        end: 1,
                        highlight_class: HighlightClass::Text,
                    },
                    HighlightSpan {
                        start: 1,
                        end: 3,
                        highlight_class: HighlightClass::String,
                    },
                    HighlightSpan {
                        start: 3,
                        end: 5,
                        highlight_class: HighlightClass::Keyword,
                    },
                    HighlightSpan {
                        start: 5,
                        end: 6,
                        highlight_class: HighlightClass::Text,
                    },
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
                    vec![HighlightSpan {
                        start: 0,
                        end: 5,
                        highlight_class: HighlightClass::Text,
                    }],
                    vec![
                        HighlightSpan {
                            start: 0,
                            end: 1,
                            highlight_class: HighlightClass::Operator,
                        },
                        HighlightSpan {
                            start: 1,
                            end: 6,
                            highlight_class: HighlightClass::Function,
                        },
                        HighlightSpan {
                            start: 6,
                            end: 7,
                            highlight_class: HighlightClass::Operator,
                        },
                    ],
                ],
            },
        ];

        for case in &cases {
            let fh = distribute_byte_spans(case.source, &case.spans);
            assert_eq!(
                fh.spans.len(),
                case.expected_lines.len(),
                "case: {} — line count mismatch",
                case.name
            );
            for (line_idx, expected) in case.expected_lines.iter().enumerate() {
                assert_eq!(
                    fh.spans[line_idx], *expected,
                    "case: {} — line {} mismatch",
                    case.name, line_idx
                );
            }
        }
    }
}
