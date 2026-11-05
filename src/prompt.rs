//! The embedded prompt assets and the one-pass machinery that fills them in: `rust_embed`
//! over `src/prompt/`, the `\n---\n` section splitter, the `{{key}}` substituter, and the
//! workspace and ticket context blocks installed into a call.
//!
//! Not the prompts themselves — those are the `.md` files, loaded by asset key — and not a
//! template engine: substitution is one pass and never cascades.

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use regex::Regex;
use std::sync::LazyLock;

#[derive(rust_embed::RustEmbed)]
#[folder = "src/prompt"]
struct PromptAssets;

/// Regex for single-pass template substitution. Only word characters (`[a-zA-Z0-9_]`)
/// match, so a future key must not contain hyphens, dots or spaces; `(?-u)` enforces
/// ASCII-only `\w`, without which `{{résumé}}` would be accepted.
pub(crate) static TEMPLATE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?-u)\{\{(\w+)\}\}").expect("TEMPLATE_RE must compile"));

/// The text an embedded asset delivers: its bytes with the checkout's line endings
/// normalised to LF.
///
/// `rust_embed` embeds an asset as the working copy holds it, so a checkout that rewrites
/// line endings (`core.autocrlf`) would hand the build CRLF — changing the text a model is
/// sent and, by hiding the `\n---\n` separator, collapsing a variant asset into one section.
/// The repository pins LF for these files, so this is a no-op on a correct checkout.
fn asset_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n")
}

/// Load a prompt template from embedded assets. Panics if the asset is missing — prompt
/// files are always present, so that is a stale key or a partial checkout.
#[must_use]
pub(crate) fn load_prompt(asset_key: &str) -> String {
    let file = PromptAssets::get(asset_key).unwrap_or_else(|| {
        panic!(
            "Embedded prompt '{asset_key}' not found. \
             Create the file at src/prompt/{asset_key} and rebuild."
        )
    });
    asset_text(file.data.as_ref())
}

/// The one statement of the render-delivery rule — a reply that names some of the turn's video
/// renders sends those, one that names none sends every render — substituted as
/// `{{media_delivery}}` into every prompt an agent that can render writes a reply from: both
/// Assistant role prompts and a custom role's context block.
#[must_use]
pub(crate) fn media_delivery() -> String {
    load_prompt("context/media_delivery.md").trim().to_owned()
}

/// Load a prompt asset and split it into `---`-delimited sections (trimmed, empty dropped).
#[must_use]
pub(crate) fn load_prompt_sections(asset_key: &str) -> Vec<String> {
    split_sections(&load_prompt(asset_key))
}

/// Deal a section list out strided across `count` slots, rendered as slot `k`'s own share.
/// No section is left unhanded: the slots share the list in rotation and the surplus stacks
/// onto the leading slots when there are fewer slots than sections. Every section reaches
/// exactly one slot.
#[must_use]
pub(crate) fn deal_sections(sections: &[String], count: usize, k: usize) -> String {
    sections
        .iter()
        .skip(k)
        .step_by(count)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn split_sections(text: &str) -> Vec<String> {
    text.split("\n---\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Append a named file section to the context string, with a markdown header and truncated
/// content. Skips empty content.
fn append_file_section(out: &mut String, name: &str, content: &str) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    let _ = writeln!(out, "--- {name} ---\n");
    push_truncated(out, trimmed);
}

/// The account's personal memory — a continuity record for its own Assistant only, never
/// context for another role or another account.
pub(crate) const MEMORY_FILE: &str = "MEMORY.md";

/// How much of one workspace file a context render keeps, in characters (Unicode scalar
/// values), not bytes.
pub(crate) const MAX_WORKSPACE_FILE_CHARS: usize = 10_000;

const WORKSPACE_FILES: &[&str] = &[
    "README.md",
    "BOOTSTRAP.md",
    MEMORY_FILE,
    "CLAUDE.md",
    "AGENTS.md",
    "AGENTS.local.md",
    "CLAUDE.local.md",
    ".cursorrules",
    "copilot-instructions.md",
    ".github/copilot-instructions.md",
];

/// Build workspace context content from workspace files. `personal_memory` admits the
/// [`MEMORY_FILE`] section: only the Assistant of the account whose personal folder this is
/// passes `true`, so this builder never renders a continuity record for anyone else.
pub(crate) async fn build_workspace_context(workspace: &Path, personal_memory: bool) -> String {
    let mut out = String::new();
    for &filename in WORKSPACE_FILES {
        if filename == MEMORY_FILE && !personal_memory {
            continue;
        }
        let path = workspace.join(filename);
        if let Ok(raw) = tokio::fs::read_to_string(&path).await {
            append_file_section(&mut out, filename, &raw);
        }
    }
    for (rel_path, content) in discover_claude_rules(workspace).await {
        append_file_section(&mut out, &rel_path, &content);
    }
    out
}

/// Wrap workspace context content in a stable `<workspace-context>` block; empty when
/// `content` is blank.
pub(crate) fn wrap_workspace_context(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    format!("\n<workspace-context>\n{trimmed}\n</workspace-context>\n")
}

/// The general (non-role) workspace context block for non-agent LLM calls. Uses the stored
/// general context row (role = NULL) when discovery produced one, else the file-derived
/// builder minus the personal memory (a non-agent call is nobody's Assistant session). Stable
/// per workspace; empty when no source has content.
async fn build_general_workspace_context(ws: &crate::Workspace) -> String {
    // Degrade to the file-derived fallback when the store is uninitialized (early tests) —
    // never panic on a missing global.
    let stored = match crate::workspace::WORKSPACES.get() {
        Some(store) => store.get_general_context(&ws.name).await.ok().flatten(),
        None => None,
    };
    let content = match stored {
        Some(ctx) => ctx,
        None => build_workspace_context(ws.as_path(), false).await,
    };
    wrap_workspace_context(&content)
}

/// Prepend the general workspace context to `messages` as the leading system message when a
/// stored or file-derived context exists. No-op otherwise.
pub(crate) async fn prepend_general_context(
    messages: &mut Vec<crate::ChatMessage>,
    ws: &crate::Workspace,
) {
    let context = build_general_workspace_context(ws).await;
    if !context.is_empty() {
        messages.insert(0, crate::ChatMessage::system(&context));
    }
}

/// Format a ticket as a `<current-ticket>` block for injection as a separate system message
/// (after the workspace context, before the user message).
pub(crate) fn format_ticket_block(ticket: &crate::pipeline::board::Ticket) -> String {
    let mut comments = String::new();
    if !ticket.comments.is_empty() {
        let _ = writeln!(comments);
        let _ = writeln!(comments, "### Comments ({})", ticket.comments.len());
        let _ = writeln!(comments);

        for comment in &ticket.comments {
            let ts = format_local_timestamp(&comment.created_at);
            let _ = writeln!(comments, "**{}** ({}):", comment.role, ts);
            let _ = writeln!(comments, "{}", comment.content);
            let _ = writeln!(comments);
            let _ = writeln!(comments, "---");
            let _ = writeln!(comments);
        }
    }

    substitute(
        &load_prompt("context/ticket.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_reporter}}", &ticket.reporter),
            ("{{ticket_priority}}", &format!("P{}", ticket.priority)),
            ("{{ticket_description}}", &ticket.description),
            ("{{ticket_comments}}", &comments),
        ],
    )
}

fn format_local_timestamp(iso_str: &str) -> String {
    crate::db::parse_utc_timestamp(iso_str).map_or_else(
        |e| {
            tracing::warn!(iso_str = %iso_str, error = %e, "Failed to parse timestamp, falling back to raw string");
            iso_str.to_string()
        },
        |dt| {
            dt.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        },
    )
}

fn truncation_point(text: &str) -> Option<usize> {
    text.char_indices()
        .nth(MAX_WORKSPACE_FILE_CHARS)
        .map(|(idx, _)| idx)
}

pub(crate) fn exceeds_workspace_file_limit(text: &str) -> bool {
    truncation_point(text).is_some()
}

fn push_truncated(out: &mut String, text: &str) {
    if let Some(idx) = truncation_point(text) {
        out.push_str(&text[..idx]);
        let _ = writeln!(
            out,
            "\n\n{}\n",
            substitute(
                &load_prompt("context/truncation_notice.md"),
                &[("{{max_chars}}", &MAX_WORKSPACE_FILE_CHARS.to_string())],
            ),
        );
    } else {
        out.push_str(text);
        out.push_str("\n\n");
    }
}

/// Discover `.claude/rules/*.md` files and return their relative paths with their content, in
/// deterministic byte-sorted name order (not `read_dir` order), so identical rules render
/// byte-identically and the workspace-context block stays stable across renders. Names within
/// one directory are unique, so the sort cannot tie.
async fn discover_claude_rules(workspace: &Path) -> Vec<(String, String)> {
    let rules_dir = workspace.join(".claude").join("rules");

    let Ok(mut entries) = tokio::fs::read_dir(&rules_dir).await else {
        return Vec::new();
    };

    let mut files = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "md") {
            files.push(path);
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut rules = Vec::new();
    for path in files {
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            let rel_path = path
                .strip_prefix(workspace)
                .unwrap_or(&path)
                .display()
                .to_string();
            rules.push((rel_path, content));
        }
    }
    rules
}

/// Single-pass template substitution: all `{{key}}` patterns are replaced at once from
/// `replacements`, so values are never re-substituted — a value containing a later key appears
/// literally.
///
/// Keys must be word characters (`[a-zA-Z0-9_]`); a key with hyphens or dots stays in the
/// output unexpanded (`all_template_variables_are_word_chars` enforces this across the embedded
/// assets). A key with no replacement is left literal and warns at runtime, so a typo shows up
/// in the log instead of silently corrupting the prompt. Callers pass keys with the full
/// wrapper (`"{{ticket_id}}"`).
pub(crate) fn substitute(template: &str, replacements: &[(&str, &str)]) -> String {
    let map: HashMap<&str, &str> = replacements.iter().copied().collect();
    TEMPLATE_RE
        .replace_all(template, |caps: &regex::Captures| {
            let whole = caps
                .get(0)
                .expect("capture group 0 always matches")
                .as_str();
            if let Some(val) = map.get(whole) {
                (*val).to_owned()
            } else {
                tracing::warn!(
                    template_var = %whole,
                    "prompt substitution: no replacement provided for '{whole}' — \
                     literal text will appear in the prompt output. \
                     Check that the variable name in the template matches a \
                     replacement key at the call site."
                );
                whole.to_owned()
            }
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pass over the template with the replacements at hand: a key with no replacement is
    /// left as the literal `{{key}}`, and a template with no placeholders comes back unchanged.
    #[test]
    fn substitute_cases() {
        type Pair = (&'static str, &'static str);
        /// Case name, template, replacements, expected output.
        type Case = (&'static str, &'static str, &'static [Pair], &'static str);

        let cases: &[Case] = &[
            (
                "both keys replaced",
                "Hello {{name}}, your {{item}} is ready.",
                &[("{{name}}", "Alice"), ("{{item}}", "order")],
                "Hello Alice, your order is ready.",
            ),
            (
                "unknown key preserved",
                "Hello {{name}}, here is {{missing}} key.",
                &[("{{name}}", "Alice")],
                "Hello Alice, here is {{missing}} key.",
            ),
            ("empty template", "", &[("{{key}}", "value")], ""),
            ("no replacements", "Hello {{name}}!", &[], "Hello {{name}}!"),
        ];
        for (case, template, replacements, expected) in cases {
            assert_eq!(
                substitute(template, replacements),
                *expected,
                "case '{case}'"
            );
        }
    }

    #[test]
    fn substitute_no_cascade() {
        // A replacement value containing a later key pattern must NOT be re-substituted —
        // the cascade bug from sequential str::replace().
        let result = substitute(
            "First: {{a}}, Second: {{b}}",
            &[("{{a}}", "value-{{b}}"), ("{{b}}", "actual-b")],
        );
        assert_eq!(result, "First: value-{{b}}, Second: actual-b");
    }

    /// Only the `.md` rules are discovered, and a rule the loader cannot read (a directory
    /// named `*.md`) is skipped rather than failing the scan.
    #[tokio::test]
    async fn discover_claude_rules_finds_only_readable_md_files() {
        let dir = tempfile::tempdir().unwrap();
        let rules_dir = dir.path().join(".claude").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        std::fs::write(rules_dir.join("testing.md"), "Test content").unwrap();
        std::fs::write(rules_dir.join("style.md"), "Style rules").unwrap();
        std::fs::write(rules_dir.join("notes.txt"), "irrelevant").unwrap();
        std::fs::create_dir(rules_dir.join("broken.md")).unwrap();

        let rules = discover_claude_rules(dir.path()).await;
        let paths: Vec<&str> = rules.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths,
            [".claude/rules/style.md", ".claude/rules/testing.md"],
            "only the readable .md rules are discovered"
        );
    }

    #[tokio::test]
    async fn discover_claude_rules_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let rules = discover_claude_rules(dir.path()).await;
        assert!(rules.is_empty());
    }

    #[tokio::test]
    async fn discover_claude_rules_returns_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        let rules_dir = dir.path().join(".claude").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        // Scrambled creation order — `read_dir` order is not guaranteed, so the loader must
        // return rules byte-sorted for a stable render.
        for name in ["zebra", "alpha", "middle"] {
            std::fs::write(
                rules_dir.join(format!("{name}.md")),
                format!("{name} rules"),
            )
            .unwrap();
        }

        let rules = discover_claude_rules(dir.path()).await;
        let paths: Vec<&str> = rules.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths,
            [
                ".claude/rules/alpha.md",
                ".claude/rules/middle.md",
                ".claude/rules/zebra.md",
            ]
        );
    }

    #[test]
    fn all_template_variables_are_word_chars() {
        // Every {{...}} placeholder in the embedded assets must use \w+ keys (ASCII
        // alphanumeric and underscore), or TEMPLATE_RE never matches and the placeholder
        // silently reaches the prompt.
        let broad_re =
            regex::Regex::new(r"\{\{([^}]+)\}\}").expect("broad placeholder regex must compile");
        for asset_key in PromptAssets::iter() {
            let asset = PromptAssets::get(&asset_key)
                .unwrap_or_else(|| panic!("asset {asset_key} disappeared between iter and get"));
            let content = String::from_utf8_lossy(asset.data.as_ref());
            for cap in broad_re.captures_iter(&content) {
                let var_name = cap.get(1).unwrap().as_str();
                assert!(
                    var_name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_'),
                    "Template variable '{{{{{var_name}}}}}' in '{asset_key}' contains non-\\w \
                     characters (only ASCII alphanumeric and underscore are allowed).\n\
                     Template keys must match \\w+ so that TEMPLATE_RE can substitute them. \
                     Use underscores instead of hyphens, dots, or spaces.",
                );
            }
        }
    }

    #[test]
    fn all_prompt_assets_load_non_empty() {
        for asset_key in PromptAssets::iter() {
            let content = load_prompt(&asset_key);
            assert!(
                !content.trim().is_empty(),
                "Prompt asset '{asset_key}' is empty or whitespace-only.\n\
                 Each embedded prompt file must contain meaningful content.",
            );
        }
    }

    /// The assets the pipeline rotates through parallel agents, with the number of variants
    /// each must yield: a collapse to one section silently hands every analyst the same
    /// variant instead of a decorrelated set.
    const VARIANT_ASSETS: &[(&str, usize)] = &[
        ("analyze/angles.md", 3),
        ("analyze/ticket_angles.md", 3),
        ("review_angles.md", 3),
        ("qa_angles.md", 3),
        ("synthesis/repair_framing.md", 6),
    ];

    #[test]
    fn embedded_assets_are_checked_out_with_lf() {
        // `rust_embed` embeds the checkout's bytes, so CR bytes here mean the build would
        // carry something other than the repository's text — the Windows build did exactly
        // that before `.gitattributes` pinned LF.
        let mut scanned = 0;
        // The instruction texts are the `.md` assets (a stray `.DS_Store` is not one).
        for asset_key in PromptAssets::iter().filter(|key| key.ends_with(".md")) {
            let asset =
                PromptAssets::get(&asset_key).expect("asset disappeared between iter and get");
            assert!(
                !asset.data.contains(&b'\r'),
                "Embedded prompt '{asset_key}' holds CR bytes.\n\
                 src/prompt must be checked out with LF — see .gitattributes, \
                 and re-checkout rather than keeping an existing working copy.",
            );
            scanned += 1;
        }
        assert!(scanned > 0, "no embedded prompt asset was inspected");
    }

    /// The context cut is also the nightly inventory's trigger: 10,000 characters render
    /// whole, 10,001 do not, and the count is characters — a multi-byte file measures the same.
    #[test]
    fn the_file_limit_counts_characters() {
        let fits = "ё".repeat(MAX_WORKSPACE_FILE_CHARS);
        assert_eq!(
            fits.len(),
            MAX_WORKSPACE_FILE_CHARS * 2,
            "two bytes per char"
        );
        assert!(!exceeds_workspace_file_limit(&fits));
        assert!(exceeds_workspace_file_limit(&format!("{fits}ё")));
    }

    #[test]
    fn variant_assets_split_alike_under_either_checkout() {
        for &(asset_key, variants) in VARIANT_ASSETS {
            let file =
                PromptAssets::get(asset_key).expect("asset disappeared between iter and get");
            let text = asset_text(file.data.as_ref());
            assert_eq!(
                split_sections(&text).len(),
                variants,
                "'{asset_key}' must split into its {variants} variants — \
                 the `---` separators are a contract, not formatting",
            );

            // A working copy checked out under the Windows convention hands the build the
            // same file with CRLF endings.
            let crlf = text.replace('\n', "\r\n");
            assert_ne!(crlf, text, "'{asset_key}' has no line endings to convert");
            assert_eq!(
                asset_text(crlf.as_bytes()),
                text,
                "'{asset_key}' must deliver the same text under either \
                 checkout convention",
            );
        }
    }
}
