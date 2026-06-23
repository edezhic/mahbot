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
use std::sync::OnceLock;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};
use tree_sitter_md::{INLINE_LANGUAGE as MD_INLINE_LANG, MarkdownParser, MarkdownTree};
// Built-in highlight queries for new languages (ticket mahbot-1244).
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

static RUST_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static JAVASCRIPT_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static PYTHON_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static TYPESCRIPT_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static TSX_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static JSON_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static TOML_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static BASH_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static CSS_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static HTML_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static GO_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static RUBY_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static C_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static SQL_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static MARKDOWN_BLOCK_QUERY: OnceLock<Option<Query>> = OnceLock::new();
static MARKDOWN_INLINE_QUERY: OnceLock<Option<Query>> = OnceLock::new();

/// Get (or compile) the highlight query for a language.
/// Returns None if the query is invalid (should not happen with baked-in queries).
pub(crate) fn cached_query(lang: HighlightLanguage) -> Option<&'static Query> {
    let cell = match lang {
        HighlightLanguage::Rust => &RUST_QUERY,
        HighlightLanguage::JavaScript => &JAVASCRIPT_QUERY,
        HighlightLanguage::TypeScript => &TYPESCRIPT_QUERY,
        HighlightLanguage::TSX => &TSX_QUERY,
        HighlightLanguage::Python => &PYTHON_QUERY,
        HighlightLanguage::Json => &JSON_QUERY,
        HighlightLanguage::Toml => &TOML_QUERY,
        HighlightLanguage::Bash => &BASH_QUERY,
        HighlightLanguage::Css => &CSS_QUERY,
        HighlightLanguage::Html => &HTML_QUERY,
        HighlightLanguage::Go => &GO_QUERY,
        HighlightLanguage::Ruby => &RUBY_QUERY,
        HighlightLanguage::C => &C_QUERY,
        HighlightLanguage::Sql => &SQL_QUERY,
        HighlightLanguage::Markdown => &MARKDOWN_BLOCK_QUERY,
    };
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
    #[must_use]
    pub const fn color(self) -> Color {
        match self {
            HighlightClass::Keyword => Color::from_rgb(0.941, 0.439, 0.110),
            HighlightClass::String => Color::from_rgb(0.298, 0.722, 0.114),
            HighlightClass::Type => Color::from_rgb(0.231, 0.510, 0.965),
            HighlightClass::Function => Color::from_rgb(0.925, 0.282, 0.600),
            HighlightClass::Comment => Color::from_rgb(0.420, 0.420, 0.420),
            HighlightClass::Number => Color::from_rgb(0.957, 0.247, 0.369),
            HighlightClass::Operator => Color::from_rgb(0.357, 0.749, 0.710),
            HighlightClass::Text => theme::TEXT_PRIMARY,
            HighlightClass::Search => Color::from_rgb(1.0, 0.667, 0.0), // amber
            HighlightClass::SearchCurrent => Color::from_rgb(1.0, 0.8, 0.2), // brighter amber
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
pub fn parse_file_highlights(
    parser: &mut Parser,
    source: &str,
    lang: HighlightLanguage,
) -> FileHighlights {
    if lang == HighlightLanguage::Markdown {
        return parse_markdown_highlights(source);
    }

    let ts_lang = lang.tree_sitter_language();
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
pub(crate) fn build_markdown_highlights_from_tree(
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
/// converting to line-relative offsets. Uses the same algorithm as
/// [`build_highlights_from_tree`].
///
/// # Overlap handling
///
/// When multiple spans overlap at the same byte range (e.g. a delimiter
/// span and its parent formatting span), only the first-encountered span
/// is emitted for the overlapping region. The subsequent overlapping span
/// emits its *non-overlapping tail* (from `cursor_pos` onward) rather than
/// being skipped entirely.
///
/// This matters most for markdown inline formatting, where delimiters
/// (`` ` `` / `*` / `**` / `_` / `__`) are captured both individually
/// (as `punctuation.delimiter` → Operator) and as part of the parent
/// formatting span (`text.emphasis` / `text.strong` / `text.literal`):
///
/// ```text
///    delimiter span:  [0, 1)  Operator    ← emitted first (smaller end)
///    parent span:     [0, 7)  Function    ← emits tail [1, 7)  (fix)
/// ```
///
/// ## Known limitations
///
/// - **Closing delimiters lose Operator color**: The closing delimiter
///   span (e.g. trailing `*` in `*italic*`) is subsumed into its parent
///   formatting span's color because `cursor_pos` already covers it after
///   the tail emission. Preserving the closing delimiter color would
///   require span-splitting, which the current flat span model does not
///   support.
/// - **Emphasis inside headings** (`# *italic*`): The block-level
///   `text.title` capture (Type) sorts before the inline `text.emphasis`
///   capture (Function) at the same byte position due to stable sort order.
///   The block capture wins visually, so emphasis highlighting is invisible
///   within heading text. This is not a regression — it's inherent to the
///   flat span model where block captures are collected before inline ones.
#[must_use]
pub(crate) fn distribute_byte_spans(
    source: &str,
    byte_spans: &[(usize, usize, HighlightClass)],
) -> FileHighlights {
    // Compute byte offsets of each line start (and the past-end sentinel).
    let mut line_starts: Vec<usize> = Vec::with_capacity(source.lines().count() + 1);
    let mut pos = 0;
    line_starts.push(0);
    for ch in source.bytes() {
        pos += 1;
        if ch == b'\n' {
            line_starts.push(pos);
        }
    }

    let mut lines: Vec<Vec<HighlightSpan>> = Vec::with_capacity(line_starts.len());
    let mut span_iter = byte_spans.iter().copied().peekable();

    for line_idx in 0..line_starts.len() {
        let line_start = line_starts[line_idx];
        let line_end = line_starts
            .get(line_idx + 1)
            .map_or(source.len(), |e| if *e > 0 { e - 1 } else { 0 });

        let mut line_spans: Vec<HighlightSpan> = Vec::new();
        let mut cursor_pos = line_start;

        // Advance through spans that overlap this line.
        while let Some(&(span_start, span_end, class)) = span_iter.peek() {
            if span_end <= line_start {
                // Span is entirely before this line — skip it.
                span_iter.next();
                continue;
            }
            if span_start >= line_end {
                // Span starts after this line — done with this line.
                break;
            }

            let s = span_start.max(line_start);
            let e = span_end.min(line_end);

            if s < cursor_pos {
                // Span overlaps a previously emitted span — skip the
                // already-covered prefix but emit the non-overlapping tail
                // (from cursor_pos onward). This typically happens when an
                // inline delimiter span (emphasis_delimiter, code_span_delimiter,
                // captured as Operator) overlaps with its parent formatting span
                // (emphasis, strong_emphasis, code_span).
                //
                // See the function-level doc for known limitations of this
                // approach (closing delimiter subsumption, emphasis in headings).
                if e > cursor_pos {
                    line_spans.push(HighlightSpan {
                        start: cursor_pos - line_start,
                        end: e - line_start,
                        highlight_class: class,
                    });
                    cursor_pos = e;
                }
                span_iter.next();
                continue;
            }

            if s > cursor_pos {
                // Gap between cursor and next span — fill with Text.
                line_spans.push(HighlightSpan {
                    start: cursor_pos - line_start,
                    end: s - line_start,
                    highlight_class: HighlightClass::Text,
                });
            }

            if e > s {
                line_spans.push(HighlightSpan {
                    start: s - line_start,
                    end: e - line_start,
                    highlight_class: class,
                });
                cursor_pos = e;
            }

            // If span extends past this line, keep it for the next line.
            if span_end <= line_end {
                span_iter.next();
            } else {
                break;
            }
        }

        if cursor_pos < line_end {
            line_spans.push(HighlightSpan {
                start: cursor_pos - line_start,
                end: line_end - line_start,
                highlight_class: HighlightClass::Text,
            });
        }

        lines.push(line_spans);
    }

    FileHighlights { spans: lines }
}

/// Build per-line highlight spans from an already-parsed tree-sitter tree.
///
/// Collects capture spans from the tree and delegates to
/// [`distribute_byte_spans`] for line distribution.
#[must_use]
pub(crate) fn build_highlights_from_tree(
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl HighlightLanguage {
    /// Determine language from a file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(HighlightLanguage::Rust),
            "js" | "jsx" | "mjs" | "cjs" => Some(HighlightLanguage::JavaScript),
            "ts" => Some(HighlightLanguage::TypeScript),
            "tsx" => Some(HighlightLanguage::TSX),
            "py" | "pyi" | "pyx" => Some(HighlightLanguage::Python),
            "json" => Some(HighlightLanguage::Json),
            "toml" => Some(HighlightLanguage::Toml),
            "sh" | "bash" | "zsh" => Some(HighlightLanguage::Bash),
            "css" => Some(HighlightLanguage::Css),
            "html" | "htm" => Some(HighlightLanguage::Html),
            "go" => Some(HighlightLanguage::Go),
            "rb" => Some(HighlightLanguage::Ruby),
            "c" | "h" => Some(HighlightLanguage::C),
            "sql" => Some(HighlightLanguage::Sql),
            "md" | "markdown" => Some(HighlightLanguage::Markdown),
            _ => None,
        }
    }

    /// Determine language from a file path.
    #[must_use]
    pub fn from_path(path: &str) -> Option<Self> {
        std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(HighlightLanguage::from_extension)
    }

    /// Return the tree-sitter Language and highlight query string for this language.
    ///
    /// The Language is obtained from the shared extension-to-language mapping
    /// ([`crate::util::tree_sitter::tree_sitter_language_for_extension`]).
    ///
    /// For Markdown, returns the **block** grammar — inline Markdown uses
    /// a separate grammar (see [`MD_INLINE_LANG`] and
    /// [`MD_HIGHLIGHT_QUERY_INLINE`]).
    pub(crate) fn language_and_query(self) -> (Language, &'static str) {
        let lang = crate::util::tree_sitter::tree_sitter_language_for_extension(self.extension())
            .expect("HighlightLanguage variant should have a valid extension mapping");
        let query = match self {
            HighlightLanguage::Rust => RUST_HIGHLIGHT_QUERY,
            HighlightLanguage::JavaScript => JAVASCRIPT_HIGHLIGHT_QUERY,
            HighlightLanguage::TypeScript => TYPESCRIPT_HIGHLIGHT_QUERY,
            HighlightLanguage::TSX => TSX_HIGHLIGHT_QUERY,
            HighlightLanguage::Python => PYTHON_HIGHLIGHT_QUERY,
            HighlightLanguage::Json => JSON_HIGHLIGHTS_QUERY,
            HighlightLanguage::Toml => TOML_HIGHLIGHTS_QUERY,
            HighlightLanguage::Bash => BASH_HIGHLIGHT_QUERY,
            HighlightLanguage::Css => CSS_HIGHLIGHTS_QUERY,
            HighlightLanguage::Html => HTML_HIGHLIGHTS_QUERY,
            HighlightLanguage::Go => GO_HIGHLIGHTS_QUERY,
            HighlightLanguage::Ruby => RUBY_HIGHLIGHTS_QUERY,
            HighlightLanguage::C => C_HIGHLIGHT_QUERY,
            HighlightLanguage::Sql => SQL_HIGHLIGHTS_QUERY,
            HighlightLanguage::Markdown => MD_HIGHLIGHT_QUERY_BLOCK,
        };
        (lang, query)
    }

    /// Return the tree-sitter Language for this language.
    ///
    /// A thin wrapper over [`Self::language_and_query`] for callers that only need
    /// the language (e.g., creating a parser).
    pub(crate) fn tree_sitter_language(self) -> Language {
        self.language_and_query().0
    }

    /// Return the canonical file extension for this language variant.
    ///
    /// Used to look up the shared tree-sitter [`Language`] via
    /// [`crate::util::tree_sitter::tree_sitter_language_for_extension`].
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            HighlightLanguage::Rust => "rs",
            HighlightLanguage::JavaScript => "js",
            HighlightLanguage::TypeScript => "ts",
            HighlightLanguage::TSX => "tsx",
            HighlightLanguage::Python => "py",
            HighlightLanguage::Json => "json",
            HighlightLanguage::Toml => "toml",
            HighlightLanguage::Bash => "sh",
            HighlightLanguage::Css => "css",
            HighlightLanguage::Html => "html",
            HighlightLanguage::Go => "go",
            HighlightLanguage::Ruby => "rb",
            HighlightLanguage::C => "c",
            HighlightLanguage::Sql => "sql",
            HighlightLanguage::Markdown => "md",
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

const JAVASCRIPT_HIGHLIGHT_QUERY: &str = r"
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

const TYPESCRIPT_HIGHLIGHT_QUERY: &str = r"
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

const TSX_HIGHLIGHT_QUERY: &str = r"
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
        // Test all standard HighlightLanguage variants via the shared extension mapping.
        for (name, variant) in [
            ("Rust", HighlightLanguage::Rust),
            ("JS", HighlightLanguage::JavaScript),
            ("TS", HighlightLanguage::TypeScript),
            ("TSX", HighlightLanguage::TSX),
            ("Python", HighlightLanguage::Python),
            ("JSON", HighlightLanguage::Json),
            ("TOML", HighlightLanguage::Toml),
            ("Bash", HighlightLanguage::Bash),
            ("CSS", HighlightLanguage::Css),
            ("HTML", HighlightLanguage::Html),
            ("Go", HighlightLanguage::Go),
            ("Ruby", HighlightLanguage::Ruby),
            ("C", HighlightLanguage::C),
            ("SQL", HighlightLanguage::Sql),
            ("MD block", HighlightLanguage::Markdown),
        ] {
            let (lang, query) = variant.language_and_query();
            let q = tree_sitter::Query::new(&lang, query);
            assert!(q.is_ok(), "{name} query failed: {:?}", q.err());
        }

        // Inline Markdown uses a separate grammar that has no extension mapping,
        // so construct its Language directly.
        let md_inline = MD_INLINE_LANG.into();
        let q = tree_sitter::Query::new(&md_inline, MD_HIGHLIGHT_QUERY_INLINE);
        assert!(q.is_ok(), "MD inline query failed: {:?}", q.err());
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

    #[test]
    fn test_markdown_italic_star() {
        let code = "*italic*";
        let fh = parse_markdown_highlights(code);
        assert_eq!(fh.spans.len(), 1, "single line expected");
        // Opening delimiter retains Operator colour.
        line0_has_class_in_range(&fh, HighlightClass::Operator, 0, 1, "opening *");
        // Content between delimiters gets text.emphasis → Function.
        line0_has_class_in_range(&fh, HighlightClass::Function, 1, 7, "italic content");
        // Closing delimiter is subsumed into emphasis span (known limitation).
    }

    #[test]
    fn test_markdown_bold_star() {
        let code = "**bold**";
        let fh = parse_markdown_highlights(code);
        assert_eq!(fh.spans.len(), 1);
        // Opening delimiter retains Operator colour.
        line0_has_class_in_range(&fh, HighlightClass::Operator, 0, 2, "opening **");
        // Content between delimiters gets text.strong → Keyword.
        line0_has_class_in_range(&fh, HighlightClass::Keyword, 2, 6, "bold content");
    }

    #[test]
    fn test_markdown_italic_underscore() {
        let code = "_italic_";
        let fh = parse_markdown_highlights(code);
        assert_eq!(fh.spans.len(), 1);
        line0_has_class_in_range(&fh, HighlightClass::Operator, 0, 1, "opening _");
        line0_has_class_in_range(&fh, HighlightClass::Function, 1, 7, "italic content");
    }

    #[test]
    fn test_markdown_bold_underscore() {
        let code = "__bold__";
        let fh = parse_markdown_highlights(code);
        assert_eq!(fh.spans.len(), 1);
        line0_has_class_in_range(&fh, HighlightClass::Operator, 0, 2, "opening __");
        line0_has_class_in_range(&fh, HighlightClass::Keyword, 2, 6, "bold content");
    }

    #[test]
    fn test_markdown_inline_code() {
        let code = "`code`";
        let fh = parse_markdown_highlights(code);
        assert_eq!(fh.spans.len(), 1);
        // Opening backtick is Operator (code_span_delimiter → punctuation.delimiter).
        line0_has_class_in_range(&fh, HighlightClass::Operator, 0, 1, "opening `");
        // Code content gets text.literal → String.
        line0_has_class_in_range(&fh, HighlightClass::String, 1, 5, "code content");
        // Closing backtick is subsumed (known limitation).
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

    // ── Direct distribute_byte_spans tests ──────────────────────────

    #[test]
    fn test_distribute_byte_spans_no_overlap() {
        // Non-overlapping spans should be emitted as-is.
        let source = "abcde";
        let spans = vec![
            (0, 1, HighlightClass::Keyword),
            (2, 5, HighlightClass::Number),
        ];
        let fh = distribute_byte_spans(source, &spans);
        assert_eq!(fh.spans.len(), 1);
        let expected = vec![
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
        ];
        assert_eq!(fh.spans[0], expected, "non-overlapping spans");
    }

    #[test]
    fn test_distribute_byte_spans_overlap_tail_emission() {
        // Overlapping spans: delimiter (Operator) followed by parent formatting
        // (Keyword). The parent span should emit its tail after the delimiter.
        let source = "**bold**";
        let spans = vec![
            (0, 2, HighlightClass::Operator), // opening ** delimiter
            (0, 8, HighlightClass::Keyword),  // **bold** strong_emphasis
            (6, 8, HighlightClass::Operator), // closing ** delimiter
        ];
        let fh = distribute_byte_spans(source, &spans);
        assert_eq!(fh.spans.len(), 1, "single line");
        // Expected: [0,2) Operator, [2,8) Keyword (closing delimiter subsumed).
        let expected = vec![
            HighlightSpan {
                start: 0,
                end: 2,
                highlight_class: HighlightClass::Operator,
            },
            HighlightSpan {
                start: 2,
                end: 8,
                highlight_class: HighlightClass::Keyword,
            },
        ];
        assert_eq!(fh.spans[0], expected, "overlap tail emission");
    }

    #[test]
    fn test_distribute_byte_spans_partial_overlap() {
        // Partial overlap: the second span only partially overlaps.
        let source = "abcdef";
        let spans = vec![
            (1, 3, HighlightClass::String),  // "bc" -> String
            (2, 5, HighlightClass::Keyword), // "cde" -> Keyword, overlaps at [2,3)
        ];
        let fh = distribute_byte_spans(source, &spans);
        assert_eq!(fh.spans.len(), 1);
        // Expected: [0,1) Text, [1,3) String, [3,5) Keyword, [5,6) Text.
        let expected = vec![
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
        ];
        assert_eq!(fh.spans[0], expected, "partial overlap");
    }

    #[test]
    fn test_distribute_byte_spans_multi_line_overlap() {
        // Multi-line input where spans cross line boundaries.
        // "hello\n*world*\n" (14 bytes, \n at positions 5 and 12)
        // line 0: "hello" (bytes 0-5), line 1: "*world*" (bytes 6-13)
        let source = "hello\n*world*";
        let spans = vec![
            // emphasis_delimiter (opening) on line 1
            (6, 7, HighlightClass::Operator),
            // emphasis covering line 1 content
            (6, 13, HighlightClass::Function),
            // emphasis_delimiter (closing) on line 1
            (12, 13, HighlightClass::Operator),
        ];
        let fh = distribute_byte_spans(source, &spans);
        assert_eq!(fh.spans.len(), 2, "two lines");
        // Line 0: "hello" — no capture spans, so a single Text fill.
        assert_eq!(
            fh.spans[0],
            vec![HighlightSpan {
                start: 0,
                end: 5,
                highlight_class: HighlightClass::Text
            }],
            "line 0 plain text fill"
        );
        // Line 1: [0,1) Operator, [1,7) Function (closing subsumed, line_relative).
        // (byte offsets are line-relative: line 1 starts at byte 6)
        let expected_line1 = vec![
            HighlightSpan {
                start: 0,
                end: 1,
                highlight_class: HighlightClass::Operator,
            },
            HighlightSpan {
                start: 1,
                end: 7,
                highlight_class: HighlightClass::Function,
            },
        ];
        assert_eq!(fh.spans[1], expected_line1, "line 1 overlap");
    }
}
