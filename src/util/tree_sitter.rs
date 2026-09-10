//! Canonical mapping from file extensions to tree-sitter [`Language`] objects.
//!
//! This is the **single source of truth** for which file extension maps to which
//! tree-sitter grammar. Both the `read` tool's symbol extraction
//! ([`crate::tools::read`]) and the GUI editor's syntax highlighting
//! ([`crate::gui::highlight::HighlightLanguage::from_extension`]) delegate here.
//!
//! ## Adding support for a new language
//!
//! 1. Add the extension(s) and grammar to `GRAMMARS` (this file).
//! 2. Add a variant to [`HighlightLanguage`] and a `language_and_query` arm in
//!    [`crate::gui::highlight`] — the reverse-lookup map is derived automatically.
//! 3. Add a `line_comment_prefix` arm in [`crate::gui::editor_widget`] (if
//!    applicable for the language).
//! 4. Add a `language_support` arm in [`crate::tools::read`] if the language
//!    should have symbol extraction (the `_ => ""` fallback gives empty symbols).
//!
//! [`HighlightLanguage`]: crate::gui::highlight::HighlightLanguage

use tree_sitter::Language;

/// One `GRAMMARS` row: a file extension and a constructor for its grammar.
type Grammar = (&'static str, fn() -> Language);

/// The canonical extension-to-grammar mapping, one row per extension.
/// Declaration order is the order `supported_extensions` — and therefore the
/// `read` tool's unsupported-extension error message — lists extensions in.
const GRAMMARS: &[Grammar] = &[
    ("rs", || tree_sitter_rust::LANGUAGE.into()),
    ("js", || tree_sitter_javascript::LANGUAGE.into()),
    ("jsx", || tree_sitter_javascript::LANGUAGE.into()),
    ("mjs", || tree_sitter_javascript::LANGUAGE.into()),
    ("cjs", || tree_sitter_javascript::LANGUAGE.into()),
    ("ts", || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
    ("tsx", || tree_sitter_typescript::LANGUAGE_TSX.into()),
    ("py", || tree_sitter_python::LANGUAGE.into()),
    ("pyi", || tree_sitter_python::LANGUAGE.into()),
    ("pyx", || tree_sitter_python::LANGUAGE.into()),
    ("json", || tree_sitter_json::LANGUAGE.into()),
    ("toml", || tree_sitter_toml_ng::LANGUAGE.into()),
    ("sh", || tree_sitter_bash::LANGUAGE.into()),
    ("bash", || tree_sitter_bash::LANGUAGE.into()),
    ("zsh", || tree_sitter_bash::LANGUAGE.into()),
    ("css", || tree_sitter_css::LANGUAGE.into()),
    ("html", || tree_sitter_html::LANGUAGE.into()),
    ("htm", || tree_sitter_html::LANGUAGE.into()),
    ("go", || tree_sitter_go::LANGUAGE.into()),
    ("rb", || tree_sitter_ruby::LANGUAGE.into()),
    ("c", || tree_sitter_c::LANGUAGE.into()),
    ("h", || tree_sitter_c::LANGUAGE.into()),
    ("sql", || tree_sitter_sequel::LANGUAGE.into()),
    ("md", || tree_sitter_md::LANGUAGE.into()),
    ("markdown", || tree_sitter_md::LANGUAGE.into()),
];

/// Every extension recognized by [`tree_sitter_language_for_extension`], in
/// declaration order. Used by the `read` tool's unsupported-extension error.
pub(crate) fn supported_extensions() -> impl Iterator<Item = &'static str> {
    GRAMMARS.iter().map(|(ext, _)| *ext)
}

/// Map a file extension to its corresponding tree-sitter [`Language`].
///
/// The mapping itself lives in `GRAMMARS`; this is only the lookup.
#[must_use]
pub fn tree_sitter_language_for_extension(ext: &str) -> Option<Language> {
    GRAMMARS
        .iter()
        .find(|(candidate, _)| *candidate == ext)
        .map(|(_, language)| language())
}
