//! Shared mapping from file extensions to tree-sitter [`Language`] objects.
//!
//! This is the single source of truth for which `tree_sitter::Language` corresponds
//! to each file extension. Both the `read` tool (symbol extraction) and the GUI
//! editor (syntax highlighting) use this function instead of maintaining separate
//! extension-to-language mappings.

use tree_sitter::Language;

/// Map a file extension to its corresponding tree-sitter [`Language`].
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
