//! Role metadata consolidation — single source of truth for all static [`Role`] properties.
//!
//! This module is the canonical home for [`Role`]'s static methods, trait impls,
//! and metadata lookups — including role descriptions, discovery prompts,
//! tool assignments, and [`RoleInfo`]. Used by [`crate::agent`] and other modules
//! that need role data.

use std::sync::LazyLock;

use strum::IntoEnumIterator;

use crate::Role;

/// Role string for diagnostics comments — used both when posting diagnostics
/// comments and in the circuit breaker filter. Must stay in sync between
/// both sites to prevent silent miscounting on re-dispatch.
pub(crate) const DIAGNOSTICS_ROLE: &str = "diagnostics";

/// Role string for system comments — used when posting system comments on
/// tickets (notifications, circuit breaker trip comments, agent summaries)
/// and when filtering comments in circuit breaker [`CircuitBreakerKind::trip_count`]
/// implementations. Must stay in sync between all sites to prevent silent miscounting.
pub(crate) const SYSTEM_ROLE: &str = "system";

// ── RoleInfo ──────────────────────────────────────────────────────────────

/// All static metadata for a [`Role`] variant.
///
/// Every accessor goes through a single match in [`role_info()`], replacing
/// the match statements that were previously scattered across the codebase
/// for role metadata lookups. Icon widgets live in `theme::role_icon()`.
///
/// Adding a new role requires updating the [`Role`] enum in `lib.rs`,
/// this match, the [`Role::tools()`] method, and the `theme::role_icon()` match.
/// The compiler will catch missing arms in exhaustive matches, but it
/// cannot catch an arm that returns an empty tool set or silently uses
/// struct update defaults — the tests in this module guard against those:
///
/// * `badge_fg` black sentinel (struct update syntax)
/// * `display_label` empty string sentinel (struct update syntax)
/// * `default_model` and `default_reasoning_effort` non-empty (struct update)
/// * [`Role::tools()`] non-empty for every variant
pub struct RoleInfo {
    /// Whether this role has a discovery prompt for workspace exploration.
    pub has_discovery: bool,
    /// Model temperature for LLM calls for this role.
    pub temperature: f32,
    /// Whether this role requires a vision-capable (multimodal) model.
    ///
    /// Controls two downstream behaviors:
    /// 1. **Message enrichment** (in `main.rs`): local image files are uploaded as
    ///    data URIs instead of being transcribed to text.
    /// 2. **Provider payload format** (in `agent.rs`): user-provided image markers
    ///    are embedded as `image_url` parts in the chat request.
    ///
    /// Only [`Role::Artist`] currently sets this to `true`.
    /// Note: image/video *generation* tools ([`ImageGenTool`], [`VideoGenTool`])
    /// use their own generation model configuration and do **not** depend on this flag.
    pub requires_multimodal: bool,
    /// Badge foreground color as an RGB tuple.
    ///
    /// Converted to an [`iced::Color`] badge in `gui/theme.rs`. The badge
    /// background is always this color at 0.1 alpha.
    pub badge_fg: (f32, f32, f32),
    /// Default model ID for this role, used when no per-role override is configured.
    pub default_model: &'static str,
    /// Default reasoning effort for this role, used when no per-role override is configured.
    pub default_reasoning_effort: &'static str,
    /// Human-readable display label (e.g. `"QA"` for [`Role::Qa`]).
    pub display_label: &'static str,
}

// ── Single source of truth ────────────────────────────────────────────────

/// Default values shared by most [`Role`] variants in [`role_info()`].
///
/// Used via struct update syntax (`..BASE_ROLE_INFO`) to keep each arm
/// concise and make future field additions cheap.
const BASE_ROLE_INFO: RoleInfo = RoleInfo {
    has_discovery: true,
    temperature: 0.1,
    requires_multimodal: false,
    badge_fg: (0.0, 0.0, 0.0),
    default_model: "deepseek/deepseek-v4-flash",
    default_reasoning_effort: "xhigh",
    display_label: "",
};

/// Look up static metadata for a role.
///
/// # Panics
/// Never — this is a complete match over all [`Role`] variants.
#[must_use]
pub const fn role_info(role: &Role) -> &'static RoleInfo {
    match role {
        Role::Manager => &RoleInfo {
            temperature: 0.01,
            badge_fg: (0.816, 0.635, 0.082),
            default_model: "deepseek/deepseek-v4-pro",
            display_label: "Manager",
            ..BASE_ROLE_INFO
        },
        Role::Engineer => &RoleInfo {
            badge_fg: (0.855, 0.439, 0.173),
            display_label: "Engineer",
            ..BASE_ROLE_INFO
        },
        Role::Analyst => &RoleInfo {
            temperature: 0.3,
            badge_fg: (0.263, 0.522, 0.745),
            display_label: "Analyst",
            ..BASE_ROLE_INFO
        },
        Role::Coder => &RoleInfo {
            temperature: 0.01,
            badge_fg: (0.353, 0.604, 0.416),
            display_label: "Coder",
            ..BASE_ROLE_INFO
        },
        Role::Qa => &RoleInfo {
            temperature: 0.4,
            badge_fg: (0.545, 0.494, 0.784),
            display_label: "QA",
            ..BASE_ROLE_INFO
        },
        Role::Reviewer => &RoleInfo {
            temperature: 0.2,
            badge_fg: (0.431, 0.494, 0.784),
            display_label: "Reviewer",
            ..BASE_ROLE_INFO
        },
        Role::Discovery => &RoleInfo {
            has_discovery: false,
            badge_fg: (0.227, 0.663, 0.624),
            display_label: "Discovery",
            ..BASE_ROLE_INFO
        },
        Role::Artist => &RoleInfo {
            has_discovery: false,
            requires_multimodal: true,
            badge_fg: (0.808, 0.365, 0.592),
            default_model: "qwen/qwen3.6-plus",
            default_reasoning_effort: "medium",
            display_label: "Artist",
            ..BASE_ROLE_INFO
        },
        Role::Maintainer => &RoleInfo {
            temperature: 0.5,
            badge_fg: (0.753, 0.376, 0.502),
            display_label: "Maintainer",
            ..BASE_ROLE_INFO
        },
        Role::Sanitation => &RoleInfo {
            badge_fg: (0.482, 0.482, 0.482),
            display_label: "Sanitation",
            ..BASE_ROLE_INFO
        },
    }
}

// ── Trait impls ─────────────────────────────────────────────────────────

/// Valid role names, pre-computed once to avoid re-iteration in error paths.
static ALL_ROLE_NAMES: LazyLock<String> = LazyLock::new(|| {
    Role::iter()
        .map(|r| r.as_str())
        .collect::<Vec<_>>()
        .join(", ")
});

impl std::str::FromStr for Role {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        Role::iter().find(|r| r.as_str() == lower).ok_or_else(|| {
            anyhow::anyhow!("Unknown role '{s}', expected one of: {}", *ALL_ROLE_NAMES)
        })
    }
}

// ── Role metadata methods ──────────────────────────────────────────────

impl Role {
    /// Canonical role name as a `&'static str` (lowercase).
    ///
    /// Delegates to the [`strum::IntoStaticStr`] derive, which produces
    /// string literals with a `'static` lifetime. This is the canonical
    /// method for obtaining the role's string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        self.into()
    }

    /// Whether this role requires a vision-capable (multimodal) model.
    #[must_use]
    pub const fn requires_multimodal(&self) -> bool {
        role_info(self).requires_multimodal
    }

    /// Collects all roles into a [`Vec<Role>`].
    ///
    /// Uses [`Role::iter()`] internally and collects into a `Vec`.
    /// Prefer using [`Role::iter()`] directly in most cases to avoid allocation.
    #[must_use]
    pub fn all_roles() -> Vec<Role> {
        Role::iter().collect()
    }

    /// Role description loaded from embedded prompt files.
    #[must_use]
    pub fn role_description(&self) -> String {
        crate::prompt::load_prompt(&format!("role/{}.md", self.as_str()))
    }

    /// Discovery prompt for this role, loaded from embedded prompt files.
    ///
    /// # Panics
    /// Panics if the role does not have a discovery prompt (see
    /// [`RoleInfo::has_discovery`]) — callers must check `has_discovery`
    /// before calling this method or use a role that is known to have one.
    #[must_use]
    pub fn discovery_prompt(&self) -> String {
        let info = role_info(self);
        if info.has_discovery {
            crate::prompt::load_prompt(&format!("discovery/{}.md", self.as_str()))
        } else {
            panic!("Discovery prompt for role '{self}' does not exist")
        }
    }

    /// Conversation compaction prompt for this role, composed from the shared
    /// template (`summarize/template.md`) and the shared OMIT section
    /// (`summarize/omit.md`).
    ///
    /// Falls back to a `[PROMPT MISSING: …]` placeholder in the relevant section
    /// if either prompt asset is missing (see `load_prompt`).
    #[must_use]
    pub fn summary_prompt(&self) -> String {
        let template = crate::prompt::load_prompt("summarize/template.md");
        let omit = crate::prompt::load_prompt("summarize/omit.md");
        let omit = omit.trim_end_matches('\n').to_string();
        let (role_name, role_section, extra_omit) = self.summarize_prompt_data();
        crate::prompt::substitute(
            &template,
            &[
                ("{{role_name}}", role_name),
                ("{{role_section}}", role_section),
                ("{{extra_omit}}", extra_omit),
                ("{{omit_section}}", &omit),
            ],
        )
    }

    #[allow(clippy::too_many_lines)]
    /// Role-specific data for the shared [`summary_prompt`] template.
    ///
    /// Returns `(role_name, role_section, extra_omit)` where:
    /// - `role_name` is the uppercase role name (e.g. `"ANALYST"`)
    /// - `role_section` is the role-specific PRESERVE bullet list (no trailing newline)
    /// - `extra_omit` is an optional extra OMIT line with trailing newline,
    ///   or the empty string for roles that have no extra omit line
    ///
    /// See `src/prompt/summarize/template.md` for how these placeholders are used.
    #[must_use]
    pub(crate) fn summarize_prompt_data(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Role::Analyst => (
                "ANALYST",
                "- The question or investigation goal\n\
                 - Evidence gathered: files, symbols, commands, URLs, observations, and test results\n\
                 - Facts vs inferences — label uncertainty explicitly\n\
                 - Rejected hypotheses and why they were ruled out\n\
                 - External docs, APIs, and version-specific findings\n\
                 - Trade-offs, risks, assumptions, and unresolved questions\n\
                 - Recommended next research steps",
                "",
            ),
            Role::Artist => (
                "ARTIST",
                "- User's visual request and realism/style constraints\n\
                 - Original uploads and reference images (paths, markers)\n\
                 - Generation attempts: prompts used, tool calls, and output paths (`[IMAGE:path]`, `[VIDEO:path]`)\n\
                 - User feedback on each iteration\n\
                 - Adjustment options offered and what the user chose next\n\
                 - Rules followed: single attempt before user review, minimal-edit framing, no unrequested additions",
                "",
            ),
            Role::Coder => (
                "CODER",
                "- The exact implementation specification and acceptance criteria\n\
                 - Target files, symbols, and local conventions discovered\n\
                 - Concrete code changes made (edits, new helpers, tests)\n\
                 - Compile, lint, and test errors encountered and how they were addressed\n\
                 - Patterns and utilities reused from the workspace\n\
                 - What remains to edit or verify",
                "- Broad research narrative not needed for the next edit\n",
            ),
            Role::Discovery => (
                "DISCOVERY",
                "- Durable workspace facts gathered for downstream agent roles\n\
                 - Project purpose, architecture, conventions, tooling, and dependencies\n\
                 - Source locations checked: docs, configs, tests, entrypoints, observability surfaces\n\
                 - Official docs and version-specific findings consulted\n\
                 - Project-specific vs generic advice — keep only workspace-specific facts\n\
                 - Unresolved gaps or areas not yet explored",
                "- Conversational filler, intros, or meta-commentary about the exploration process\n",
            ),
            Role::Engineer => (
                "ENGINEER",
                "- User requests and the implementation objective\n\
                 - Affected files, modules, symbols, and integration points\n\
                 - Edits made (file operations, diffs, refactors)\n\
                 - Commands and local checks run, with key results and errors\n\
                 - Sub-agent (`ask`) findings from analysts and coders\n\
                 - Design constraints, conventions followed, and invariants respected\n\
                 - Blockers, unresolved technical questions, and remaining implementation steps",
                "",
            ),
            Role::Maintainer => (
                "MAINTAINER",
                "- Cleanup and refactoring opportunities discovered\n\
                 - Evidence supporting each opportunity (duplication, dead code, drift, complexity)\n\
                 - Affected files, modules, and safe refactor boundaries\n\
                 - Safety and value rationale for each suggestion\n\
                 - Rejected ideas and why they were skipped\n\
                 - Constraints: no macros, no module-directory splits, LoC impact considered\n\
                 - Tickets created (IDs, titles) and areas not yet investigated\n\
                 - Sub-agent (`ask`) findings from deeper investigations",
                "",
            ),
            Role::Manager => (
                "MANAGER",
                "- User requests, approved outcomes, and product/scope decisions\n\
                 - Ticket IDs, titles, statuses, phases, and reporter (especially Maintainer tickets)\n\
                 - Board actions taken: created, updated, canceled, superseded, advanced, or blocked tickets\n\
                 - Analyst results and technical context that informed decisions\n\
                 - Pending user decisions and your recommendations\n\
                 - Why work was advanced, canceled, superseded, or left waiting\n\
                 - Prerequisites and dependencies between tickets",
                "- Low-level implementation details unless they affect a product decision\n",
            ),
            Role::Qa => (
                "QA",
                "- Acceptance criteria and expected user-facing behavior\n\
                 - Prior diagnostics, test, and reviewer evidence already considered\n\
                 - Verification steps attempted and their outcomes\n\
                 - Code paths and runtime flows inspected\n\
                 - Confirmed failures, gaps, and user-impacting issues\n\
                 - Residual risks and unverified assumptions\n\
                 - Score rationale if a score was assigned or discussed",
                "",
            ),
            Role::Reviewer => (
                "REVIEWER",
                "- Changed-code context: files, symbols, and architectural boundaries touched\n\
                 - Review findings by severity and why each matters\n\
                 - Architectural or invariant concerns\n\
                 - Missing tests or verification gaps\n\
                 - Confirmed issues vs speculative concerns\n\
                 - Final review posture (approved, changes requested, blocking)",
                "",
            ),
            Role::Sanitation => (
                "SANITATION",
                "- List of untracked/new files inspected\n\
                 - File contents and patterns examined\n\
                 - Garbage indicators detected (compiled binaries, temp files, build artifacts, etc.)\n\
                 - Legitimate file indicators (referenced in manifests, build configs, etc.)\n\
                 - Verdict rationale",
                "",
            ),
        }
    }
}

// ── Tool set factory ──────────────────────────────────────────────────────

use crate::Tool;
use crate::config::CONFIG;
use crate::tools::{
    AddCommentTool, AskTool, BrowserTool, CreateTicketTool, DispatchMode, EditTool, GetTicketTool,
    ImageGenTool, ListTicketsTool, ReadTool, SearchArchivedTicketsTool, SearchTool, ShellMode,
    ShellTool, UpdateTicketTool, VideoGenTool, WebSearchBackend, WebSearchTool,
};

impl Role {
    /// Core read/search/read-only-shell tools for inspector-style roles
    /// (Analyst, QA, Reviewer, Discovery, Sanitation, Maintainer).
    fn readonly_core_tools() -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ReadTool),
            Box::new(SearchTool),
            Box::new(ShellTool::new(ShellMode::ReadOnly)),
        ]
    }

    /// Core full-shell/read/edit/search tools for full-access roles
    /// (Engineer, Coder).
    fn full_core_tools() -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ShellTool::new(ShellMode::Full)),
            Box::new(ReadTool),
            Box::new(EditTool),
            Box::new(SearchTool),
        ]
    }

    /// Build the tool set for this role.
    #[must_use]
    pub fn tools(&self) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = match self {
            Role::Engineer => {
                let mut t = Self::full_core_tools();
                t.push(Box::new(AskTool::new(
                    vec![Role::Analyst, Role::Coder],
                    DispatchMode::Sync,
                )));
                t
            }
            Role::Manager => {
                vec![
                    Box::new(CreateTicketTool::new("manager")),
                    Box::new(UpdateTicketTool),
                    Box::new(ListTicketsTool),
                    Box::new(GetTicketTool),
                    Box::new(AddCommentTool),
                    Box::new(SearchArchivedTicketsTool),
                    Box::new(AskTool::new(vec![Role::Analyst], DispatchMode::Async)),
                ]
            }
            Role::Analyst => {
                let mut t = Self::readonly_core_tools();
                t.push(Box::new(BrowserTool::default()));
                t
            }
            Role::Coder => Self::full_core_tools(),
            Role::Qa | Role::Reviewer | Role::Discovery | Role::Sanitation => {
                Self::readonly_core_tools()
            }
            Role::Artist => {
                vec![
                    Box::new(BrowserTool::default()),
                    Box::new(SearchTool),
                    Box::new(ImageGenTool),
                    Box::new(VideoGenTool),
                ]
            }
            Role::Maintainer => {
                let mut t = Self::readonly_core_tools();
                t.push(Box::new(AskTool::new(
                    vec![Role::Analyst],
                    DispatchMode::Sync,
                )));
                t.push(Box::new(CreateTicketTool::new("maintainer")));
                t
            }
        };

        // Manager does not need the web search tool as he is expected to
        // use ask with analysts for that.
        if !matches!(self, Role::Manager) {
            Self::add_web_search_tool(&mut tools);
        }

        tools
    }

    /// Appends a web search tool based on the current configuration.
    ///
    /// At most one web search tool is registered — if an explicit provider
    /// is configured but its API key is missing, no tool is added.
    /// Auto-selection: Firecrawl wins ties (both keys set, no preference).
    /// The caller is responsible for skipping this for Manager (who is
    /// expected to delegate web searches to analysts via [`AskTool`]).
    fn add_web_search_tool(tools: &mut Vec<Box<dyn Tool>>) {
        let provider = CONFIG.web_search_provider();
        let firecrawl_key = CONFIG.firecrawl_key();
        let exa_key = CONFIG.exa_key();

        let backend: Option<WebSearchBackend> = match provider.as_deref() {
            Some(p) if p.eq_ignore_ascii_case("firecrawl") => {
                firecrawl_key.map(|key| WebSearchBackend::Firecrawl { key })
            }
            Some(p) if p.eq_ignore_ascii_case("exa") => {
                exa_key.map(|key| WebSearchBackend::Exa { key })
            }
            Some(other) => {
                tracing::warn!("Unknown web_search_provider: {other}");
                None
            }
            None => firecrawl_key
                .map(|key| WebSearchBackend::Firecrawl { key })
                .or_else(|| exa_key.map(|key| WebSearchBackend::Exa { key })),
        };

        if let Some(backend) = backend {
            tools.push(Box::new(WebSearchTool::new(backend)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_roundtrip() {
        // FromStr for every variant by lowercase name
        for role in Role::iter() {
            let parsed: crate::Role = role.as_str().parse().unwrap();
            assert_eq!(parsed, role, "roundtrip failed for '{}'", role.as_str());
            // Display (strum-generated) must match the canonical as_str()
            assert_eq!(role.to_string(), role.as_str());
            // as_str() returns a &'static str — verify it's non-empty
            assert!(
                !role.as_str().is_empty(),
                "as_str() empty for '{}'",
                role.as_str()
            );
        }

        // Error case
        assert!("unknown_role".parse::<crate::Role>().is_err());
    }

    #[test]
    fn requires_multimodal_only_artist() {
        // Only Artist should require multimodal; every other role should not.
        for role in Role::iter() {
            let info = super::role_info(&role);
            let expected = matches!(role, crate::Role::Artist);
            assert_eq!(
                info.requires_multimodal,
                expected,
                "{}: expected requires_multimodal={expected}, got {}",
                role.as_str(),
                info.requires_multimodal
            );
        }
    }

    #[test]
    fn badge_colors_set() {
        // Guards against the BASE_ROLE_INFO default of (0,0,0) — a new role
        // added with struct update syntax must set badge_fg explicitly.
        for role in Role::iter() {
            let info = super::role_info(&role);
            let (r, g, b) = info.badge_fg;
            let is_black = r == 0.0 && g == 0.0 && b == 0.0;
            assert!(
                !is_black,
                "{}: badge_fg must not be (0,0,0) — set a visible color",
                role.as_str()
            );
        }
    }

    #[test]
    fn defaults_set() {
        // Guards against empty default_model or default_reasoning_effort — a new
        // role added with struct update syntax must set them if they differ from
        // the BASE_ROLE_INFO defaults, and even the base must be non-empty.
        for role in Role::iter() {
            let info = super::role_info(&role);
            assert!(
                !info.default_model.is_empty(),
                "{}: default_model must not be empty",
                role.as_str()
            );
            assert!(
                !info.default_reasoning_effort.is_empty(),
                "{}: default_reasoning_effort must not be empty",
                role.as_str()
            );
        }
    }

    #[test]
    fn display_labels_set() {
        // Guards against the BASE_ROLE_INFO sentinel of "" — every role must
        // set a display_label explicitly.
        for role in Role::iter() {
            let info = super::role_info(&role);
            assert!(
                !info.display_label.is_empty(),
                "{}: display_label must not be empty — set a display_label in role_info()",
                role.as_str()
            );
        }
    }

    #[test]
    fn all_roles_have_tools() {
        // Guards against an empty Vec in Role::tools() — the compiler catches
        // missing arms in the match, but cannot catch an arm that returns
        // vec![]. Every role needs at least one tool to function.
        for role in Role::iter() {
            let tools = role.tools();
            assert!(
                !tools.is_empty(),
                "{}: Role::tools() must not be empty — every role needs at least one tool",
                role.as_str()
            );
        }
    }

    #[test]
    fn qa_display_label() {
        // QA has a special display label (not "Qa").
        let info = super::role_info(&crate::Role::Qa);
        assert_eq!(info.display_label, "QA");
    }

    #[test]
    fn all_roles_have_summary_prompt() {
        for role in Role::iter() {
            let prompt = role.summary_prompt();

            // Basic sanity checks.
            assert!(
                !prompt.trim().is_empty(),
                "{}: summary_prompt() must not be empty",
                role.as_str()
            );
            assert!(
                prompt.contains("DO NOT USE ANY TOOLS"),
                "{}: summary prompt must instruct no tool use",
                role.as_str()
            );

            // Verify no unsubstituted template keys survive substitution.
            assert!(
                !crate::prompt::TEMPLATE_RE.is_match(&prompt),
                "{}: summary prompt must not contain unsubstituted template keys",
                role.as_str()
            );

            // Verify byte-identical output against golden expected file.
            // Golden files live in tests/fixtures/summarize/<role>.md.
            // To regenerate after template or data changes:
            //   1. Update template.md, omit.md, or summarize_prompt_data()
            //   2. Delete tests/fixtures/summarize/<role>.md for the roles
            //      whose output changed
            //   3. Run the test — it will fail with the new correct output.
            //      Copy that output into the corresponding golden file.
            //   (Or run: cargo test all_roles_have_summary_prompt)
            let (expected, file) = match role {
                Role::Analyst => (
                    include_str!("../tests/fixtures/summarize/analyst.md"),
                    "tests/fixtures/summarize/analyst.md",
                ),
                Role::Artist => (
                    include_str!("../tests/fixtures/summarize/artist.md"),
                    "tests/fixtures/summarize/artist.md",
                ),
                Role::Coder => (
                    include_str!("../tests/fixtures/summarize/coder.md"),
                    "tests/fixtures/summarize/coder.md",
                ),
                Role::Discovery => (
                    include_str!("../tests/fixtures/summarize/discovery.md"),
                    "tests/fixtures/summarize/discovery.md",
                ),
                Role::Engineer => (
                    include_str!("../tests/fixtures/summarize/engineer.md"),
                    "tests/fixtures/summarize/engineer.md",
                ),
                Role::Maintainer => (
                    include_str!("../tests/fixtures/summarize/maintainer.md"),
                    "tests/fixtures/summarize/maintainer.md",
                ),
                Role::Manager => (
                    include_str!("../tests/fixtures/summarize/manager.md"),
                    "tests/fixtures/summarize/manager.md",
                ),
                Role::Qa => (
                    include_str!("../tests/fixtures/summarize/qa.md"),
                    "tests/fixtures/summarize/qa.md",
                ),
                Role::Reviewer => (
                    include_str!("../tests/fixtures/summarize/reviewer.md"),
                    "tests/fixtures/summarize/reviewer.md",
                ),
                Role::Sanitation => (
                    include_str!("../tests/fixtures/summarize/sanitation.md"),
                    "tests/fixtures/summarize/sanitation.md",
                ),
            };
            assert_eq!(
                prompt,
                expected,
                "{}: summary_prompt() output must match golden file at {}\n\
                 To regenerate:\n\
                 1. Update template.md, omit.md, or summarize_prompt_data()\n\
                 2. Delete the stale golden file at {}\n\
                 3. Run this test — the new correct output will appear\n\
                    in the failure diff. Copy it back into the golden file.",
                role.as_str(),
                file,
                file,
            );
        }
    }

    #[test]
    fn all_roles_have_discovery_prompt() {
        for role in Role::iter() {
            if !super::role_info(&role).has_discovery {
                continue;
            }
            let prompt = role.discovery_prompt();
            assert!(
                !prompt.trim().is_empty(),
                "{}: discovery_prompt() must not be empty",
                role.as_str()
            );
            assert!(
                !crate::prompt::TEMPLATE_RE.is_match(&prompt),
                "{}: discovery prompt must not contain unsubstituted template keys",
                role.as_str()
            );
        }
    }
}
