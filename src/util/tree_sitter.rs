//! Canonical mapping from file extensions to tree-sitter [`Language`] objects.
//!
//! This is the **single source of truth** for which file extension maps to which
//! tree-sitter grammar. Both the `read` tool's symbol extraction
//! ([`crate::tools::read`]) and the GUI editor's syntax highlighting
//! ([`crate::gui::highlight::HighlightLanguage::from_extension`]) delegate here.
//!
//! ## Adding support for a new language
//!
//! 1. Add the extension(s) to [`tree_sitter_language_for_extension`] (this file)
//!    AND to [`ALL_TREE_SITTER_EXTENSIONS`] just below.
//! 2. Add a variant to [`HighlightLanguage`] and a `language_and_query` arm in
//!    [`crate::gui::highlight`] — the reverse-lookup map is derived automatically.
//! 3. Add a `line_comment_prefix` arm in [`crate::gui::editor_widget`] (if
//!    applicable for the language).
//! 4. Add a `language_support` arm in [`crate::tools::read`] if the language
//!    should have symbol extraction (the `_ => ""` fallback gives empty symbols).
//!
//! [`HighlightLanguage`]: crate::gui::highlight::HighlightLanguage
//! [`ALL_TREE_SITTER_EXTENSIONS`]: self::ALL_TREE_SITTER_EXTENSIONS

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
/// Every extension that [`tree_sitter_language_for_extension`] can recognize.
///
/// **Must be kept in sync with the match arms in the function below.** When a
/// new extension is added to a match arm, add it here too. This constant
/// is used by the `read` tool's error message and test to avoid hardcoded
/// copies elsewhere.
pub(crate) const ALL_TREE_SITTER_EXTENSIONS: &[&str] = &[
    "rs", "js", "jsx", "mjs", "cjs", "ts", "tsx", "py", "pyi", "pyx", "json", "toml", "sh", "bash",
    "zsh", "css", "html", "htm", "go", "rb", "c", "h", "sql", "md", "markdown",
];

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every extension listed in [`ALL_TREE_SITTER_EXTENSIONS`] must be
    /// recognized by [`tree_sitter_language_for_extension`].
    ///
    /// This catches the case where an extension is removed from the function
    /// without being removed from the constant (false-positive constant entry).
    ///
    /// **Note:** The reverse check (every extension in the function appears in
    /// the constant) cannot be automated because the function's match arms
    /// aren't inspectable at compile time. When adding a new extension to a
    /// match arm, you must also add it to [`ALL_TREE_SITTER_EXTENSIONS`].
    #[test]
    fn all_constant_extensions_are_supported() {
        for ext in ALL_TREE_SITTER_EXTENSIONS {
            assert!(
                tree_sitter_language_for_extension(ext).is_some(),
                "expected tree_sitter_language_for_extension(\"{ext}\") to return Some, \
                 but the function returned None. \
                 If you removed this extension from the function, remove it from \
                 ALL_TREE_SITTER_EXTENSIONS too."
            );
        }
    }
}
