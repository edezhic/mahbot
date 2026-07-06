//! Skills subsystem — loads workspace skills and injects them into prompts.
//!
//! Skills live in any of these locations (prioritised in order):
//! - `<workspace>/skills/<name>/SKILL.md`
//! - `<workspace>/.claude/skills/<name>/SKILL.md`
//! - `<workspace>/.agents/skills/<name>/SKILL.md`
//!
//! Each SKILL.md is markdown with optional YAML frontmatter for `name` and
//! `description`. Loaded skills are rendered into the system prompt as name +
//! description, with a `<location>` path the model can `read` for full content.

use crate::Skill;
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fmt::Write;
use std::path::Path;

// ── Frontmatter parsing ─────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct SkillMarkdownMeta {
    name: Option<String>,
    description: Option<String>,
}

/// Parse a minimal YAML frontmatter block (`---` delimited) from markdown.
/// Only `name:` and `description:` keys are extracted; everything else is ignored.
fn parse_frontmatter(content: &str) -> SkillMarkdownMeta {
    /// Strip leading/trailing whitespace and surrounding quotes.
    fn strip_quotes(s: &str) -> String {
        s.trim().trim_matches('"').trim_matches('\'').to_string()
    }

    let content = content.trim();
    if !content.starts_with("---") {
        return SkillMarkdownMeta::default();
    }

    let rest = &content[3..];
    let end = rest.find("\n---").map_or(0, |i| i + 3);
    if end < 3 {
        return SkillMarkdownMeta::default();
    }

    let frontmatter = rest[..end - 3].trim();
    let mut meta = SkillMarkdownMeta::default();

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("name:") {
            meta.name = Some(strip_quotes(value));
        } else if let Some(value) = line.strip_prefix("description:") {
            meta.description = Some(strip_quotes(value));
        }
    }

    meta
}

// ── Loading ─────────────────────────────────────────────────────────────

/// Load all skills from the workspace skills directories.
///
/// Scans three locations (in priority order):
/// 1. `<workspace>/skills/`
/// 2. `<workspace>/.claude/skills/`
/// 3. `<workspace>/.agents/skills/`
///
/// If multiple directories contain a skill with the same name, the first one wins.
#[must_use]
pub async fn load_skills(ws: &crate::Workspace) -> Vec<Skill> {
    let workspace = ws.as_path();
    let dirs = [
        workspace.join("skills"),
        workspace.join(".claude").join("skills"),
        workspace.join(".agents").join("skills"),
    ];

    let mut seen = HashSet::new();
    let mut skills = Vec::new();

    for dir in dirs {
        for skill in scan_skills_dir(&dir).await {
            if seen.insert(skill.name.clone()) {
                skills.push(skill);
            }
        }
    }

    skills
}

/// Scan a single directory for skill subdirectories (each containing `SKILL.md`).
async fn scan_skills_dir(dir: &Path) -> Vec<Skill> {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return Vec::new();
    };

    let mut skills = Vec::new();

    while let Ok(Some(entry)) = entries.next_entry().await {
        let Ok(file_type) = entry.file_type().await else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }

        let path = entry.path();
        let md_path = path.join("SKILL.md");
        if !tokio::fs::try_exists(&md_path).await.unwrap_or(false) {
            continue;
        }

        if let Ok(skill) = load_skill(&md_path, &path).await {
            skills.push(skill);
        }
    }

    skills
}

async fn load_skill(path: &Path, skill_dir: &Path) -> Result<Skill> {
    let content = tokio::fs::read_to_string(path)
        .await
        .context("failed to read SKILL.md")?;
    let meta = parse_frontmatter(&content);

    Ok(Skill {
        name: meta.name.unwrap_or_else(|| {
            skill_dir
                .file_name()
                .map_or("unnamed".to_string(), |s| s.to_string_lossy().into_owned())
        }),
        description: meta
            .description
            .unwrap_or("No description provided.".to_string()),
        location: path.to_path_buf(),
    })
}

// ── Prompt rendering ────────────────────────────────────────────────────

/// Render skills into the system prompt.
///
/// Only name and description are inlined. The LLM must use `read` with
/// the `<location>` path to get the full content.
#[must_use]
pub fn skills_to_prompt(skills: &[Skill], ws: &crate::Workspace) -> String {
    let mut skills_xml = String::new();
    for skill in skills {
        let _ = writeln!(skills_xml, "  <skill>");
        write_xml_text_element(&mut skills_xml, 4, "name", &skill.name);
        write_xml_text_element(&mut skills_xml, 4, "description", &skill.description);

        let location = render_skill_location(skill, ws.as_path());
        write_xml_text_element(&mut skills_xml, 4, "location", &location);

        let _ = writeln!(skills_xml, "  </skill>");
    }
    crate::prompt::substitute(
        &crate::prompt::load_prompt("skills.md"),
        &[("{{skills}}", skills_xml.trim())],
    )
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn write_xml_text_element(w: &mut String, indent: usize, name: &str, value: &str) {
    let padding = " ".repeat(indent);
    let escaped = crate::util::html::escape_html(value);
    let _ = writeln!(w, "{padding}<{name}>{escaped}</{name}>");
}

fn render_skill_location(skill: &Skill, workspace: &Path) -> String {
    if let Ok(relative) = skill.location.strip_prefix(workspace) {
        return relative.display().to_string();
    }
    skill.location.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use std::path::PathBuf;

    #[tokio::test]
    async fn load_md_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = dir.path().join("skills");
        let sd = skills_dir.join("my-skill");
        std::fs::create_dir_all(&sd).unwrap();
        std::fs::write(
            sd.join("SKILL.md"),
            "---\nname: my-skill\ndescription: A markdown skill\n---\n\n# Instructions\nDo something.",
        )
        .unwrap();
        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "my-skill");
        assert_eq!(skills[0].description, "A markdown skill");
    }

    #[tokio::test]
    async fn load_md_skill_without_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = dir.path().join("skills");
        let sd = skills_dir.join("bare-skill");
        std::fs::create_dir_all(&sd).unwrap();
        std::fs::write(sd.join("SKILL.md"), "# Instructions\nDo something.").unwrap();
        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "bare-skill");
        assert_eq!(skills[0].description, "No description provided.");
    }

    #[tokio::test]
    async fn empty_skills_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let skills = load_skills(&test_ws(dir.path())).await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn nonexistent_workspace_returns_empty() {
        let skills = load_skills(&test_ws(Path::new("/nonexistent"))).await;
        assert!(skills.is_empty());
    }

    #[test]
    fn parse_frontmatter_works() {
        let content = "---\nname: my-skill\ndescription: Does stuff\n---\n\nContent here";
        let meta = parse_frontmatter(content);
        assert_eq!(meta.name.as_deref(), Some("my-skill"));
        assert_eq!(meta.description.as_deref(), Some("Does stuff"));
    }

    #[test]
    fn parse_no_frontmatter() {
        let content = "# Just a heading\n\nSome content";
        let meta = parse_frontmatter(content);
        assert!(meta.name.is_none());
    }

    #[test]
    fn skills_to_prompt_shows_name_and_description() {
        let skill = Skill {
            name: "my-skill".into(),
            description: "Does stuff".into(),
            location: PathBuf::from("skills/my-skill/SKILL.md"),
        };
        let prompt = skills_to_prompt(&[skill], &test_ws(Path::new("")));
        assert!(prompt.contains("my-skill"));
        assert!(prompt.contains("Does stuff"));
        assert!(prompt.contains("skills/my-skill/SKILL.md"));
    }

    #[test]
    fn skills_to_prompt_no_instructions_inlined() {
        let skill = Skill {
            name: "quiet-skill".into(),
            description: "Does stuff".into(),
            location: PathBuf::from("skills/quiet-skill/SKILL.md"),
        };
        let prompt = skills_to_prompt(&[skill], &test_ws(Path::new("")));
        assert!(prompt.contains("quiet-skill"));
        assert!(prompt.contains("Does stuff"));
    }

    #[tokio::test]
    async fn load_skills_from_claude_skills_dir() {
        let dir = tempfile::tempdir().unwrap();
        let sd = dir.path().join(".claude").join("skills").join("my-skill");
        std::fs::create_dir_all(&sd).unwrap();
        std::fs::write(
            sd.join("SKILL.md"),
            "---\nname: claude-skill\ndescription: From .claude/skills\n---\n\nContent",
        )
        .unwrap();
        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "claude-skill");
    }

    #[tokio::test]
    async fn load_skills_from_agents_skills_dir() {
        let dir = tempfile::tempdir().unwrap();
        let sd = dir.path().join(".agents").join("skills").join("my-skill");
        std::fs::create_dir_all(&sd).unwrap();
        std::fs::write(
            sd.join("SKILL.md"),
            "---\nname: agents-skill\ndescription: From .agents/skills\n---\n\nContent",
        )
        .unwrap();
        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "agents-skill");
    }

    #[tokio::test]
    async fn load_skills_dedup_workspace_priority() {
        let dir = tempfile::tempdir().unwrap();

        // Same skill name in all three directories
        let sd1 = dir.path().join("skills").join("common");
        std::fs::create_dir_all(&sd1).unwrap();
        std::fs::write(
            sd1.join("SKILL.md"),
            "---\nname: common\ndescription: From workspace skills/\n---\n\nContent",
        )
        .unwrap();

        let sd2 = dir.path().join(".claude").join("skills").join("common");
        std::fs::create_dir_all(&sd2).unwrap();
        std::fs::write(
            sd2.join("SKILL.md"),
            "---\nname: common\ndescription: From .claude/skills\n---\n\nContent",
        )
        .unwrap();

        let sd3 = dir.path().join(".agents").join("skills").join("common");
        std::fs::create_dir_all(&sd3).unwrap();
        std::fs::write(
            sd3.join("SKILL.md"),
            "---\nname: common\ndescription: From .agents/skills\n---\n\nContent",
        )
        .unwrap();

        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "From workspace skills/");
    }

    #[tokio::test]
    async fn load_skills_dedup_claude_over_agents() {
        let dir = tempfile::tempdir().unwrap();

        // Same skill name in claude and agents (no workspace/skills)
        let sd2 = dir.path().join(".claude").join("skills").join("common");
        std::fs::create_dir_all(&sd2).unwrap();
        std::fs::write(
            sd2.join("SKILL.md"),
            "---\nname: common\ndescription: From .claude/skills\n---\n\nContent",
        )
        .unwrap();

        let sd3 = dir.path().join(".agents").join("skills").join("common");
        std::fs::create_dir_all(&sd3).unwrap();
        std::fs::write(
            sd3.join("SKILL.md"),
            "---\nname: common\ndescription: From .agents/skills\n---\n\nContent",
        )
        .unwrap();

        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "From .claude/skills");
    }

    #[tokio::test]
    async fn load_skills_unique_names_from_multiple_dirs() {
        let dir = tempfile::tempdir().unwrap();

        let sd1 = dir.path().join("skills").join("skill-a");
        std::fs::create_dir_all(&sd1).unwrap();
        std::fs::write(
            sd1.join("SKILL.md"),
            "---\nname: skill-a\ndescription: From workspace\n---\n\nContent",
        )
        .unwrap();

        let sd2 = dir.path().join(".claude").join("skills").join("skill-b");
        std::fs::create_dir_all(&sd2).unwrap();
        std::fs::write(
            sd2.join("SKILL.md"),
            "---\nname: skill-b\ndescription: From .claude\n---\n\nContent",
        )
        .unwrap();

        let sd3 = dir.path().join(".agents").join("skills").join("skill-c");
        std::fs::create_dir_all(&sd3).unwrap();
        std::fs::write(
            sd3.join("SKILL.md"),
            "---\nname: skill-c\ndescription: From .agents\n---\n\nContent",
        )
        .unwrap();

        let skills = load_skills(&test_ws(dir.path())).await;
        assert_eq!(skills.len(), 3);
    }

    #[test]
    fn write_xml_text_element_escapes_special_chars() {
        // Verify that `<`, `>`, `&`, `"`, and `'` are all properly escaped
        // using the shared `escape_html` from `util::html`.
        let mut out = String::new();
        write_xml_text_element(&mut out, 0, "test", "<hello> & \"world\" 'test'");
        assert_eq!(
            out,
            "<test>&lt;hello&gt; &amp; &quot;world&quot; &#39;test&#39;</test>\n"
        );
    }
}
