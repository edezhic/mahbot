use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use regex::Regex;
use std::sync::LazyLock;

// ── Prompt Asset Loading ─────────────────────────────────────────────

#[derive(rust_embed::RustEmbed)]
#[folder = "src/prompt"]
struct PromptAssets;

/// Regex for single-pass template substitution.
///
/// Only matches keys consisting of word characters (`\w` = `[a-zA-Z0-9_]`).
/// Future template keys must not contain hyphens, dots, or spaces.
static TEMPLATE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{(\w+)\}\}").expect("TEMPLATE_RE must compile"));

/// Load a prompt template from embedded assets.
///
/// # Panics
/// Panics if the prompt file is not found
/// (the asset was not embedded or the asset key is misspelled).
#[must_use]
pub(crate) fn load_prompt(asset_key: &str) -> String {
    let file = PromptAssets::get(asset_key).unwrap_or_else(|| {
        panic!(
            "Embedded prompt '{asset_key}' not found. \
             Create the file at src/prompt/{asset_key} and rebuild."
        )
    });
    String::from_utf8_lossy(file.data.as_ref()).into_owned()
}

// Utils
// ──────────────────────────────────────────────────────────────────────────────

/// Append a named file section to the context string, with a markdown header
/// and truncated content. Skips empty content.
fn append_file_section(out: &mut String, name: &str, content: &str) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    let _ = writeln!(out, "--- {name} ---\n");
    push_truncated(out, trimmed);
}

/// Build workspace context content from workspace files.
pub(crate) async fn build_workspace_context(workspace: &Path) -> String {
    const WORKSPACE_FILES: &[&str] = &[
        "README.md",
        "BOOTSTRAP.md",
        "MEMORY.md",
        "CLAUDE.md",
        "AGENTS.md",
        "AGENTS.local.md",
        "CLAUDE.local.md",
        ".cursorrules",
        "copilot-instructions.md",
        ".github/copilot-instructions.md",
    ];

    let mut out = String::new();
    for &filename in WORKSPACE_FILES {
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

/// Format a ticket as a `<current-ticket>` block for injection as a
/// separate system message (after memory context, before user message).
pub(crate) fn format_ticket_block(ticket: &crate::board::Ticket) -> String {
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
        &load_prompt("ticket.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_reporter}}", &ticket.reporter),
            ("{{ticket_description}}", &ticket.description),
            ("{{ticket_comments}}", &comments),
        ],
    )
}

/// Parse an ISO 8601 timestamp and format it as local date+time.
fn format_local_timestamp(iso_str: &str) -> String {
    crate::turso::parse_utc_timestamp(iso_str).map_or_else(
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

/// Push `text` to `out`, truncating at `MAX_WORKSPACE_FILE_CHARS` if needed.
fn push_truncated(out: &mut String, text: &str) {
    const MAX_WORKSPACE_FILE_CHARS: usize = 10_000;

    if let Some((idx, _)) = text.char_indices().nth(MAX_WORKSPACE_FILE_CHARS) {
        out.push_str(&text[..idx]);
        let _ = writeln!(
            out,
            "\n\n[... truncated at {MAX_WORKSPACE_FILE_CHARS} chars — use `read` for full file]\n"
        );
    } else {
        out.push_str(text);
        out.push_str("\n\n");
    }
}

/// Discover `.claude/rules/*.md` files and return their relative paths + content.
async fn discover_claude_rules(workspace: &Path) -> Vec<(String, String)> {
    let rules_dir = workspace.join(".claude").join("rules");

    let Ok(mut entries) = tokio::fs::read_dir(&rules_dir).await else {
        return Vec::new();
    };

    let mut rules = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "md")
            && let Ok(content) = tokio::fs::read_to_string(&path).await
        {
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

/// Single-pass template substitution.
///
/// All `{{key}}` patterns in the template are replaced with their corresponding
/// values from `replacements`. The replacement uses a regex to match all keys
/// at once, so values can never be re-substituted — a value containing a later
/// key will appear literally, not as a replacement.
///
/// Placeholder keys (the text between `{{` and `}}`) must consist entirely
/// of word characters (`[a-zA-Z0-9_]`). Keys with hyphens (`{{my-key}}`),
/// dots (`{{config.key}}`), or other non‑word characters will remain in the
/// output unexpanded.
///
/// Callers must pass replacement map keys with the full `{{key}}` wrapper
/// (e.g. `"{{ticket_id}}"`), not just the inner key name.
pub(crate) fn substitute(template: &str, replacements: &[(&str, &str)]) -> String {
    let map: HashMap<&str, &str> = replacements.iter().copied().collect();
    TEMPLATE_RE
        .replace_all(template, |caps: &regex::Captures| {
            let whole = caps
                .get(0)
                .expect("capture group 0 always matches")
                .as_str();
            map.get(whole).copied().unwrap_or(whole).to_owned()
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_basic_replacement() {
        let result = substitute(
            "Hello {{name}}, your {{item}} is ready.",
            &[("{{name}}", "Alice"), ("{{item}}", "order")],
        );
        assert_eq!(result, "Hello Alice, your order is ready.");
    }

    #[test]
    fn substitute_preserves_unknown_keys() {
        let result = substitute(
            "Hello {{name}}, here is {{missing}} key.",
            &[("{{name}}", "Alice")],
        );
        assert_eq!(result, "Hello Alice, here is {{missing}} key.");
    }

    #[test]
    fn substitute_no_cascade() {
        // If a replacement value contains a later key pattern, it must NOT
        // be re-substituted — the cascade bug from sequential str::replace().
        let result = substitute(
            "First: {{a}}, Second: {{b}}",
            &[("{{a}}", "value-{{b}}"), ("{{b}}", "actual-b")],
        );
        assert_eq!(result, "First: value-{{b}}, Second: actual-b");
    }

    #[test]
    fn substitute_empty_template() {
        let result = substitute("", &[("{{key}}", "value")]);
        assert_eq!(result, "");
    }

    #[test]
    fn substitute_no_replacements() {
        let result = substitute("Hello {{name}}!", &[]);
        assert_eq!(result, "Hello {{name}}!");
    }

    #[tokio::test]
    async fn discover_claude_rules_finds_md_files() {
        let dir = tempfile::tempdir().unwrap();
        let rules_dir = dir.path().join(".claude").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        std::fs::write(rules_dir.join("testing.md"), "Test content").unwrap();
        std::fs::write(rules_dir.join("style.md"), "Style rules").unwrap();
        // Non-markdown file should be ignored
        std::fs::write(rules_dir.join("notes.txt"), "irrelevant").unwrap();

        let rules = discover_claude_rules(dir.path()).await;
        assert_eq!(rules.len(), 2);
        let paths: Vec<&str> = rules.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&".claude/rules/testing.md"));
        assert!(paths.contains(&".claude/rules/style.md"));
    }

    #[tokio::test]
    async fn discover_claude_rules_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let rules = discover_claude_rules(dir.path()).await;
        assert!(rules.is_empty());
    }

    #[tokio::test]
    async fn discover_claude_rules_skips_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        let rules_dir = dir.path().join(".claude").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        // Create a file that looks like a dir (unreadable as file)
        let bad = rules_dir.join("broken.md");
        std::fs::write(&bad, "fine").unwrap();
        // Create a valid one too
        std::fs::write(rules_dir.join("good.md"), "good content").unwrap();
        // Both should be found since both are readable
        let rules = discover_claude_rules(dir.path()).await;
        assert_eq!(rules.len(), 2);
    }

    #[tokio::test]
    async fn discover_claude_rules_returns_full_content() {
        let dir = tempfile::tempdir().unwrap();
        let rules_dir = dir.path().join(".claude").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        // Content over MAX_WORKSPACE_FILE_CHARS — truncation is handled by push_truncated later
        let long = "x".repeat(100_000);
        std::fs::write(rules_dir.join("long.md"), &long).unwrap();
        let rules = discover_claude_rules(dir.path()).await;
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].1.len(), 100_000);
    }

    #[test]
    fn all_tool_descriptions_exist_in_embedded_assets() {
        /// Extract the tool name from an embedded asset key like `"tool/shell.md"`.
        /// Returns `None` if the key is not a `tool/*.md` asset.
        fn tool_name_from_asset_key(key: &str) -> Option<&str> {
            const PREFIX: &str = "tool/";
            const SUFFIX: &str = ".md";
            if key.starts_with(PREFIX) && key.ends_with(SUFFIX) {
                Some(&key[PREFIX.len()..key.len() - SUFFIX.len()])
            } else {
                None
            }
        }

        // ── Build the set of expected tool names ────────────────────────
        //
        // Iterate over every role's tool list so that adding a new tool to
        // Role::tools() automatically checks it here.  web_search is
        // conditionally added when a Firecrawl API key is configured, so it may
        // not appear in tests — we add it explicitly.
        use crate::Role;
        use std::collections::HashSet;

        let mut expected: HashSet<String> = HashSet::new();
        for role in <Role as strum::IntoEnumIterator>::iter() {
            for tool in role.tools() {
                expected.insert(tool.name().to_string());
            }
        }
        // web_search is gated on CONFIG.firecrawl_key() which is None in tests.
        expected.insert("web_search".to_string());

        // ── Collect available tool description files from embedded assets ─
        let mut available: HashSet<String> = HashSet::new();
        for asset_key in PromptAssets::iter() {
            // asset_key is Cow<'static, str> (from rust-embed).
            // Filter to only tool/*.md files and extract the tool name.
            if let Some(tool_name) = tool_name_from_asset_key(&asset_key) {
                available.insert(tool_name.to_string());
            }
        }

        // ── Forward check: every expected tool must have a description file ─
        let mut missing: Vec<&str> = expected
            .difference(&available)
            .map(String::as_str)
            .collect();
        missing.sort_unstable();
        assert!(
            missing.is_empty(),
            "Missing embedded tool description file(s):\n\
             {}\n\
             Each tool returned by Role::tools() must have a corresponding\n\
             src/prompt/tool/<name>.md file. Create one for each missing tool.",
            missing
                .iter()
                .map(|n| format!("  tool/{n}.md"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        // ── Reverse check: every description file must map to a known tool ─
        let mut orphaned: Vec<&str> = available
            .difference(&expected)
            .map(String::as_str)
            .collect();
        orphaned.sort_unstable();
        assert!(
            orphaned.is_empty(),
            "Orphaned tool description file(s) — no tool uses them:\n\
             {}\n\
             Remove or archive the extraneous src/prompt/tool/<name>.md file(s).",
            orphaned
                .iter()
                .map(|n| format!("  tool/{n}.md"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        // ── Content check: every description file must be non-empty ─────
        for name in &available {
            let key = format!("tool/{name}.md");
            let asset = PromptAssets::get(&key)
                .unwrap_or_else(|| panic!("asset {key} disappeared between iter and get"));
            let content = String::from_utf8_lossy(asset.data.as_ref());
            assert!(
                !content.trim().is_empty(),
                "Tool description file '{key}' is empty or whitespace-only.\n\
                 Add a meaningful description for the tool '{name}'.",
            );
        }
    }
}
