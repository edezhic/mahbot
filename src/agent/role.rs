//! Role metadata consolidation — single source of truth for all static [`Role`] properties.
//!
//! This module is the canonical home for [`Role`]'s static methods, trait impls,
//! and metadata lookups — including role descriptions, discovery prompts,
//! tool assignments, and [`RoleInfo`]. Used by [`crate::agent`] and other modules
//! that need role data.

use std::sync::LazyLock;

use strum::IntoEnumIterator;

use crate::Role;

/// Role string for diagnostics comments — `"diagnostics"`.
///
/// Used when posting diagnostics success/failure comments (`diagnostics.rs`),
/// as the diagnostics run's shell spill-owner key (`tools/shell/mod.rs`, via
/// `cleanup_agent_spills`), as the sentinel that collapses diagnostics comments
/// to a summary in the GUI board (`gui/board.rs`), and as the
/// diagnostics-discovery agent id / stale-write log label (`workspace.rs`).
///
/// The literal is a key: it's compared against `comment.role` (`gui/board.rs`)
/// and against agent ids (the shell spill collision guard in `tools/shell/mod.rs`),
/// so a change must be coordinated across those sites.
pub(crate) const DIAGNOSTICS_ROLE: &str = "diagnostics";

/// Role string for system comments — `"system"`.
///
/// Used only to post non-agent ticket comments: the engineer hard-failure
/// notice (`development.rs`), the skip-review notice (`review.rs`), and the
/// phase-reset / bounce-breaker trip notices (`pipeline/mod.rs`). It is
/// posting-only — no site reads or compares a comment's role against this
/// value, and the GUI badge for `"system"` is an unrelated generic fallback
/// in `gui/theme.rs`.
pub(crate) const SYSTEM_ROLE: &str = "system";

// ── RoleInfo ──────────────────────────────────────────────────────────────

/// All static metadata for a [`Role`] variant.
///
/// Every accessor goes through a single match in [`role_info()`], replacing
/// the match statements that were previously scattered across the codebase
/// for role metadata lookups. Icon widgets live in `theme::role_icon()`.
///
/// Adding a new role requires updating the [`Role`] enum in `lib.rs`,
/// creating prompt files at `src/prompt/role/{name}.md` and
/// `src/prompt/summarize/{name}.md` (and optionally
/// `src/prompt/discovery/{name}.md` if `has_discovery` is true),
/// adding an arm in this match, the [`Role::tools()`] method,
/// and the `theme::role_icon()` match.
/// The compiler will catch missing arms in exhaustive matches, but it
/// cannot catch an arm that returns an empty tool set or silently uses
/// struct update defaults — the tests in this module guard against those:
///
/// * `badge_fg` black sentinel (struct update syntax)
/// * `display_label` empty string sentinel (struct update syntax)
/// * `default_reasoning_effort` non-empty (struct update)
/// * [`Role::tools()`] non-empty for every variant
/// * `role_description()` contains real content (no placeholder)
/// * `summary_prompt()` contains real content (no placeholder)
/// * `discovery_prompt()` contains real content (no placeholder, for roles where `has_discovery` is true)
pub struct RoleInfo {
    /// Whether this role has a discovery prompt for workspace exploration.
    pub has_discovery: bool,
    /// Badge foreground color as an RGB tuple.
    ///
    /// Converted to an [`iced::Color`] badge in `gui/theme.rs`. The badge
    /// background is always this color at 0.1 alpha.
    pub badge_fg: (f32, f32, f32),
    /// Default reasoning effort for this role.
    ///
    /// Authoritative: reasoning effort is baked into the binary and not
    /// user-tunable, so this default is what request time always uses.
    pub default_reasoning_effort: &'static str,
    /// Human-readable display label (e.g. `"QA"` for [`Role::Qa`]).
    pub display_label: &'static str,
}

// ── Single source of truth ────────────────────────────────────────────────

/// Default values shared by most [`Role`] variants in [`role_info()`].
///
/// Used via struct update syntax (`..BASE_ROLE_INFO`) to keep each arm
/// concise and make future field additions cheap. Arms that override every
/// field (Discovery, Artist, Assistant) spell them out — clippy's
/// `needless_update` fires when the base contributes nothing, so the update
/// syntax is only valid when at least one field comes from the base.
const BASE_ROLE_INFO: RoleInfo = RoleInfo {
    has_discovery: true,
    badge_fg: (0.0, 0.0, 0.0),
    default_reasoning_effort: "high",
    display_label: "",
};

/// Look up static metadata for a role.
///
/// # Panics
/// Never — this is a complete match over all [`Role`] variants.
#[must_use]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub const fn role_info(role: &Role) -> &'static RoleInfo {
    match role {
        Role::Manager => &RoleInfo {
            badge_fg: (0.816, 0.635, 0.082),
            default_reasoning_effort: "xhigh",
            display_label: "Manager",
            ..BASE_ROLE_INFO
        },
        Role::Engineer => &RoleInfo {
            badge_fg: (0.855, 0.439, 0.173),
            default_reasoning_effort: "xhigh",
            display_label: "Engineer",
            ..BASE_ROLE_INFO
        },
        Role::Analyst => &RoleInfo {
            badge_fg: (0.263, 0.522, 0.745),
            display_label: "Analyst",
            ..BASE_ROLE_INFO
        },
        Role::Coder => &RoleInfo {
            badge_fg: (0.353, 0.604, 0.416),
            display_label: "Coder",
            ..BASE_ROLE_INFO
        },
        Role::Qa => &RoleInfo {
            badge_fg: (0.545, 0.494, 0.784),
            display_label: "QA",
            ..BASE_ROLE_INFO
        },
        Role::Reviewer => &RoleInfo {
            badge_fg: (0.431, 0.494, 0.784),
            display_label: "Reviewer",
            ..BASE_ROLE_INFO
        },
        Role::Discovery => &RoleInfo {
            has_discovery: false,
            default_reasoning_effort: "xhigh",
            badge_fg: (0.227, 0.663, 0.624),
            display_label: "Discovery",
        },
        Role::Artist => &RoleInfo {
            has_discovery: false,
            badge_fg: (0.808, 0.365, 0.592),
            default_reasoning_effort: "high",
            display_label: "Artist",
        },
        Role::Maintainer => &RoleInfo {
            badge_fg: (0.753, 0.376, 0.502),
            default_reasoning_effort: "xhigh",
            display_label: "Maintainer",
            ..BASE_ROLE_INFO
        },
        Role::Sanitation => &RoleInfo {
            badge_fg: (0.482, 0.482, 0.482),
            display_label: "Sanitation",
            ..BASE_ROLE_INFO
        },
        Role::Assistant => &RoleInfo {
            has_discovery: false,
            badge_fg: (0.153, 0.820, 0.757),
            default_reasoning_effort: "xhigh",
            display_label: "Assistant",
        },
        Role::Support => &RoleInfo {
            has_discovery: false,
            badge_fg: (0.463, 0.647, 0.843),
            default_reasoning_effort: "high",
            display_label: "Support",
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

    /// Human-readable display label (e.g. `"QA"` for [`Role::Qa`]).
    #[must_use]
    pub const fn display_label(&self) -> &'static str {
        role_info(self).display_label
    }

    /// Role description loaded from embedded prompt files.
    #[must_use]
    pub fn role_description(&self) -> String {
        crate::prompt::load_prompt(&format!("role/{}.md", self.as_str()))
    }

    /// Role description for this role, optionally widened for a full-access
    /// (admin) Assistant.
    ///
    /// Only the Assistant has a separate description when the triggering user
    /// has `permissions='full'`; every other role returns its canonical
    /// [`Role::role_description`] regardless of `full_access`.
    #[must_use]
    pub fn role_description_for(&self, full_access: bool) -> String {
        if *self == Role::Assistant && full_access {
            crate::prompt::load_prompt("role/assistant_full.md")
        } else {
            self.role_description()
        }
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

    /// Conversation compaction prompt for this role, loaded from
    /// `src/prompt/summarize/{role}.md`.
    #[must_use]
    pub fn summary_prompt(&self) -> String {
        crate::prompt::load_prompt(&format!("summarize/{}.md", self.as_str()))
    }
}

// ── Tool set factory ──────────────────────────────────────────────────────

use crate::Tool;
use crate::Workspace;
use crate::config::CONFIG;
use crate::tools::{
    AddAlarmTool, AddCommentTool, AddUserTool, AddWorkspaceTool, AnalyzeTool, BindTelegramTool,
    BrowserTool, CreateTicketTool, DispatchMode, EditTool, FinalizeTool, GetTicketTool,
    ImageGenTool, ImplementTool, InstallChromeUseTool, ListAlarmsTool, ListTicketsTool,
    MahbotDebugTool, ReadTool, RemoveAlarmTool, ResearchTool, SearchArchivedTicketsTool,
    SearchTool, SetupTelegramBotTool, SetupWebSearchTool, ShellMode, ShellTool, StrictReadTool,
    UpdateTicketTool, VideoEditTool, VideoGenTool, WebSearchBackend, WebSearchTool,
};

impl Role {
    /// Core read/search/read-only-shell tools for inspector-style roles
    /// (Analyst, QA, Reviewer, Discovery, Maintainer).
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
    ///
    /// Ticket tools are bound to `ws` at construction time — all their
    /// operations are confined to that workspace.
    ///
    /// `full_access` is the triggering user's `permissions='full'` (admin)
    /// flag. It only widens the Assistant's toolset (adding `shell`,
    /// `implement`, and `research`); every other role's toolset is
    /// byte-identical regardless of its value.
    #[must_use]
    pub(crate) fn tools(self, ws: &Workspace, full_access: bool) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = match self {
            Role::Engineer => {
                let mut t = Self::full_core_tools();
                t.push(Box::new(AnalyzeTool::new(
                    DispatchMode::Sync,
                    Role::Engineer,
                )));
                t.push(Box::new(ImplementTool::new(
                    DispatchMode::Sync,
                    Role::Engineer,
                )));
                t
            }
            Role::Manager => {
                let reporter = self.as_str();
                vec![
                    Box::new(CreateTicketTool::new(reporter, ws)),
                    Box::new(UpdateTicketTool::new(ws)),
                    Box::new(ListTicketsTool::new(ws)),
                    Box::new(GetTicketTool::new(reporter, ws)),
                    Box::new(AddCommentTool::new(ws)),
                    Box::new(SearchArchivedTicketsTool::new(ws)),
                    Box::new(AnalyzeTool::new(DispatchMode::Async, Role::Manager)),
                    Box::new(ResearchTool::new(Role::Manager)),
                ]
            }
            Role::Analyst => {
                let mut t = Self::readonly_core_tools();
                t.push(Box::new(MahbotDebugTool));
                t.push(Box::new(BrowserTool::default()));
                t
            }
            Role::Coder => Self::full_core_tools(),
            Role::Qa | Role::Reviewer | Role::Discovery => Self::readonly_core_tools(),
            Role::Sanitation => {
                // Sanitation deliberately has NO search tools (local `search`
                // or `web_search`): the role inspects and cleans specific
                // filesystem artifacts and never needs to index the temp tree
                // or hit the web — for the periodic temp-dir cleaner a
                // `search` over the whole temp folder would index junk and
                // start filesystem watchers. Read + read-only shell cover
                // inspection and temp-root mutation.
                vec![
                    Box::new(ReadTool),
                    Box::new(ShellTool::new(ShellMode::ReadOnly)),
                ]
            }
            Role::Artist => {
                vec![
                    Box::new(SearchTool),
                    Box::new(ImageGenTool),
                    Box::new(VideoGenTool),
                    Box::new(VideoEditTool),
                ]
            }
            Role::Maintainer => {
                let mut t = Self::readonly_core_tools();
                t.push(Box::new(AnalyzeTool::new(
                    DispatchMode::Sync,
                    Role::Maintainer,
                )));
                t.push(Box::new(CreateTicketTool::new("maintainer", ws)));
                t
            }
            Role::Assistant => {
                let mut t: Vec<Box<dyn Tool>> = vec![
                    Box::new(AnalyzeTool::new(DispatchMode::Async, Role::Assistant)),
                    Box::new(AddAlarmTool),
                    Box::new(ListAlarmsTool),
                    Box::new(RemoveAlarmTool),
                    Box::new(EditTool),
                    Box::new(SearchTool),
                ];
                // Base Assistant is workspace-bounded (strict read only);
                // full-access retains the general ReadTool so it can also read
                // dependency sources / temp files.
                if full_access {
                    t.push(Box::new(ReadTool));
                } else {
                    t.push(Box::new(StrictReadTool));
                }
                if full_access {
                    t.push(Box::new(ShellTool::new(ShellMode::Full)));
                    t.push(Box::new(ImplementTool::new(
                        DispatchMode::Async,
                        Role::Assistant,
                    )));
                    t.push(Box::new(ResearchTool::new(Role::Assistant)));
                }
                t
            }
            Role::Support => {
                vec![
                    Box::new(MahbotDebugTool),
                    Box::new(SetupTelegramBotTool),
                    Box::new(BindTelegramTool),
                    Box::new(AddWorkspaceTool),
                    Box::new(AddUserTool),
                    Box::new(SetupWebSearchTool),
                    Box::new(InstallChromeUseTool),
                    Box::new(FinalizeTool),
                ]
            }
        };

        if !matches!(self, Role::Manager | Role::Sanitation | Role::Support) {
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
    /// expected to delegate web searches to analysts via [`AnalyzeTool`])
    /// and Sanitation (whose toolset is deliberately search-free — see
    /// [`Role::tools`]).
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
        // Guards against an empty default_reasoning_effort — a new role added
        // with struct update syntax must set it if it differs from the
        // BASE_ROLE_INFO default, and even the base must be non-empty.
        for role in Role::iter() {
            let info = super::role_info(&role);
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
            let tools = role.tools(&crate::workspace::test_ws("test"), false);
            assert!(
                !tools.is_empty(),
                "{}: Role::tools() must not be empty — every role needs at least one tool",
                role.as_str()
            );
        }
    }

    #[test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    fn sanitation_toolset_is_read_and_shell_only() {
        // Acceptance pin: the Sanitation role advertises NO
        // search tools — even with web search configured — exactly read +
        // read-only shell.
        let snapshot = crate::config::CONFIG.snapshot();
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);
        let _ = crate::config::CONFIG.set_string_field("web_search_provider", "exa");
        let _ = crate::config::CONFIG.set_string_field("exa_key", "test-key");
        let names: Vec<&str> = crate::Role::Sanitation
            .tools(&crate::workspace::test_ws("test"), false)
            .iter()
            .map(|t| t.name())
            .collect();
        crate::config::CONFIG.swap(snapshot);
        assert_eq!(
            names,
            ["read", "shell"],
            "Sanitation toolset must be exactly read + read-only shell, got: {names:?}"
        );
    }

    #[test]
    fn analyst_and_support_expose_mahbot_debug_and_other_roles_do_not() {
        // Acceptance pin for the in-process read-only SQL tool: only Analyst and
        // Support advertise `mahbot_debug` — it must not appear for the
        // Assistant (either base or full-access) or any other role.
        let ws = crate::workspace::test_ws("test");
        for role in [crate::Role::Analyst, crate::Role::Support] {
            let tools = role.tools(&ws, false);
            assert!(
                tools.iter().any(|t| t.name() == "mahbot_debug"),
                "{} toolset must contain `mahbot_debug`",
                role.as_str()
            );
        }
        for (role, full_access) in [
            (crate::Role::Manager, false),
            (crate::Role::Engineer, false),
            (crate::Role::Coder, false),
            (crate::Role::Qa, false),
            (crate::Role::Reviewer, false),
            (crate::Role::Discovery, false),
            (crate::Role::Artist, false),
            (crate::Role::Maintainer, false),
            (crate::Role::Sanitation, false),
            (crate::Role::Assistant, false),
            (crate::Role::Assistant, true),
        ] {
            let tools = role.tools(&ws, full_access);
            assert!(
                !tools.iter().any(|t| t.name() == "mahbot_debug"),
                "{} must not advertise `mahbot_debug`",
                role.as_str()
            );
        }
    }

    #[test]
    fn assistant_toolset_gates_full_access_tools() {
        // The Assistant toolset must differ by the triggering user's
        // full-access (admin) flag: base mode has no shell/implement/research,
        // full mode does. Base gets the workspace-only StrictReadTool; full
        // keeps the general ReadTool (which also permits dependency sources).
        let ws = crate::workspace::test_ws("test");
        let base = crate::Role::Assistant.tools(&ws, false);
        let full = crate::Role::Assistant.tools(&ws, true);

        for name in ["shell", "implement", "research"] {
            assert!(
                !base.iter().any(|t| t.name() == name),
                "base Assistant toolset must not contain '{name}'"
            );
            assert!(
                full.iter().any(|t| t.name() == name),
                "full Assistant toolset must contain '{name}'"
            );
        }
        // The read boundary is what the model is told: base (restricted) only
        // advertises the personal workspace; full keeps the general allowlist.
        let base_read = base.iter().find(|t| t.name() == "read");
        let base_path_desc = base_read
            .map(|t| t.parameters_schema()["properties"]["path"]["description"].to_string())
            .unwrap_or_default();
        assert!(
            base_path_desc.contains("Only the workspace is accessible"),
            "base Assistant read must advertise the workspace-only boundary, got: {base_path_desc}"
        );
        let full_read = full.iter().find(|t| t.name() == "read");
        let full_path_desc = full_read
            .map(|t| t.parameters_schema()["properties"]["path"]["description"].to_string())
            .unwrap_or_default();
        assert!(
            full_path_desc.contains("policy allowlist"),
            "full Assistant read must advertise the general allowlist boundary, got: {full_path_desc}"
        );
    }

    #[test]
    fn assistant_role_descriptions_are_populated() {
        // Both Assistant descriptions (base + full) must be non-empty, free of
        // unsubstituted template keys, and carry the alarm-notification marker.
        for full_access in [false, true] {
            let desc = crate::Role::Assistant.role_description_for(full_access);
            assert!(
                !desc.trim().is_empty(),
                "Assistant role description must not be empty"
            );
            assert!(
                !crate::prompt::TEMPLATE_RE.is_match(&desc),
                "Assistant role description must not contain unsubstituted template keys"
            );
            assert!(
                desc.contains("<alarm-notification>"),
                "Assistant role description must explain the <alarm-notification> marker"
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
    fn all_roles_have_role_description() {
        // Guards against an empty or unsubstituted role description file.
        for role in Role::iter() {
            let desc = role.role_description();
            assert!(
                !desc.trim().is_empty(),
                "{}: role_description() must not be empty",
                role.as_str()
            );
            assert!(
                !crate::prompt::TEMPLATE_RE.is_match(&desc),
                "{}: role description must not contain unsubstituted template keys",
                role.as_str()
            );
        }
    }

    #[test]
    fn all_roles_have_summary_prompt() {
        // Guards against an empty or unsubstituted summary prompt file.
        for role in Role::iter() {
            let prompt = role.summary_prompt();
            assert!(
                !prompt.trim().is_empty(),
                "{}: summary_prompt() must not be empty",
                role.as_str()
            );
            assert!(
                !crate::prompt::TEMPLATE_RE.is_match(&prompt),
                "{}: summary prompt must not contain unsubstituted template keys",
                role.as_str()
            );
        }
    }

    #[test]
    fn all_roles_have_discovery_prompt() {
        // Guards against an empty or unsubstituted discovery prompt file.
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
