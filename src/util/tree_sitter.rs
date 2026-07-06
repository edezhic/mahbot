//! Canonical mapping from file extensions to tree-sitter [`Language`] objects.
//!
//! This is the **single source of truth** for which file extension maps to which
//! tree-sitter grammar. Both the `read` tool's symbol extraction
//! ([`crate::tools::read`]) and the GUI editor's syntax highlighting
//! ([`crate::gui::highlight::HighlightLanguage::from_extension`]) delegate here.
//!
//! When adding support for a new language:
//! 1. Add the extension(s) to [`tree_sitter_language_for_extension`] (this file).
//! 2. Add a variant to [`HighlightLanguage`] and a `language_and_query` arm in
//!    [`crate::gui::highlight`] — the reverse-lookup map is derived automatically.

use tree_sitter::Language;

/// Map a file extension to its corresponding tree-sitter [`Language`].
///
/// This is the **canonical** extension-to-language mapping used by both the
/// `read` tool ([`crate::tools::read`]) and the GUI editor's syntax
/// highlighting ([`crate::gui::highlight::HighlightLanguage::from_extension`]).
///
/// Supported extensions and their languages:
///
/// | Extension(s) | Language |
/// |---|---|
/// | `rs` | Rust |
/// | `js`, `jsx`, `mjs`, `cjs` | JavaScript |
/// | `ts` | TypeScript |
/// | `tsx` | TSX |
/// | `py`, `pyi`, `pyx` | Python |
/// | `json` | JSON |
/// | `toml` | TOML |
/// | `sh`, `bash`, `zsh` | Bash |
/// | `css` | CSS |
/// | `html`, `htm` | HTML |
/// | `go` | Go |
/// | `rb` | Ruby |
/// | `c`, `h` | C |
/// | `sql` | SQL |
/// | `md`, `markdown` | Markdown |
#[must_use]
pub fn tree_sitter_language_for_extension(ext: &str) -> Option<Language> {
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
        "md" | "markdown" => Some(tree_sitter_md::LANGUAGE.into()),
        _ => None,
    }
}
