//! Role metadata consolidation — single source of truth for all static [`Role`] properties.
//!
//! This module is the canonical home for [`Role`]'s static methods, trait impls,
//! and metadata lookups — including role descriptions, discovery prompts,
//! tool assignments, and [`RoleInfo`]. Used by [`crate::agent`] and other modules
//! that need role data.

use crate::Role;

// ── RoleInfo ──────────────────────────────────────────────────────────────

/// All static metadata for a [`Role`] variant.
///
/// Every accessor goes through a single match in [`role_info()`], replacing
/// the match statements that were previously scattered across the codebase
/// for role metadata lookups. Icon widgets live in `theme::role_icon()`.
///
/// **Important:** [`crate::Agent::new()`] may inject additional tools
/// after [`Role::tools()`] returns — for example, the Manager role receives
/// an async `AskTool` there because it needs the session key for async
/// dispatch (which [`Role::tools()`] doesn't have access to). If adding a
/// role that needs agent-identity data for its tools, check there too.
///
/// Adding a new role requires updating the [`Role`] enum in `lib.rs`,
/// this match, the [`Role::tools()`] method,
/// [`crate::Agent::new()`] (for roles that need session-key-dependent
/// tools), and the `theme::role_icon()` match.
/// The compiler will catch missing arms in exhaustive matches, but it
/// cannot catch an arm that returns an empty tool set or silently uses
/// struct update defaults — the tests in this module guard against those:
///
/// * `badge_fg` black sentinel (struct update syntax)
/// * `display_label` empty string sentinel (struct update syntax)
/// * `default_model` and `default_reasoning_effort` non-empty (struct update)
/// * [`Role::tools()`] non-empty for every variant
#[allow(clippy::struct_excessive_bools)]
pub struct RoleInfo {
    /// Whether this role has a discovery prompt for workspace exploration.
    pub has_discovery: bool,
    /// Whether users can explicitly select this role (via `/agent` command).
    pub selectable: bool,
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
    selectable: false,
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
#[allow(clippy::too_many_lines)]
pub fn role_info(role: &Role) -> &'static RoleInfo {
    match role {
        Role::Manager => &RoleInfo {
            selectable: true,
            temperature: 0.01,
            badge_fg: (0.961, 0.620, 0.043),
            default_model: "deepseek/deepseek-v4-pro",
            display_label: "Manager",
            ..BASE_ROLE_INFO
        },
        Role::Engineer => &RoleInfo {
            selectable: true,
            badge_fg: (0.976, 0.451, 0.086),
            default_model: "deepseek/deepseek-v4-pro",
            display_label: "Engineer",
            ..BASE_ROLE_INFO
        },
        Role::Analyst => &RoleInfo {
            selectable: true,
            temperature: 0.3,
            badge_fg: (0.231, 0.510, 0.965),
            display_label: "Analyst",
            ..BASE_ROLE_INFO
        },
        Role::Coder => &RoleInfo {
            temperature: 0.01,
            badge_fg: (0.063, 0.725, 0.506),
            display_label: "Coder",
            ..BASE_ROLE_INFO
        },
        Role::Qa => &RoleInfo {
            temperature: 0.4,
            badge_fg: (0.545, 0.361, 0.965),
            display_label: "QA",
            ..BASE_ROLE_INFO
        },
        Role::Reviewer => &RoleInfo {
            temperature: 0.2,
            badge_fg: (0.388, 0.400, 0.945),
            display_label: "Reviewer",
            ..BASE_ROLE_INFO
        },
        Role::Discovery => &RoleInfo {
            has_discovery: false,
            badge_fg: (0.024, 0.714, 0.831),
            display_label: "Discovery",
            ..BASE_ROLE_INFO
        },
        Role::Artist => &RoleInfo {
            has_discovery: false,
            selectable: true,
            requires_multimodal: true,
            badge_fg: (0.957, 0.247, 0.369),
            default_model: "qwen/qwen3.6-plus",
            default_reasoning_effort: "medium",
            display_label: "Artist",
            ..BASE_ROLE_INFO
        },
        Role::Maintainer => &RoleInfo {
            temperature: 0.5,
            badge_fg: (0.925, 0.282, 0.600),
            default_model: "deepseek/deepseek-v4-pro",
            display_label: "Maintainer",
            ..BASE_ROLE_INFO
        },
    }
}

// ── Trait impls ─────────────────────────────────────────────────────────

impl std::str::FromStr for Role {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        <Role as strum::IntoEnumIterator>::iter()
            .find(|r| r.as_str() == lower)
            .ok_or_else(|| {
                let names: Vec<&str> = <Role as strum::IntoEnumIterator>::iter()
                    .map(|r| r.as_str())
                    .collect();
                anyhow::anyhow!("Unknown role '{s}', expected one of: {}", names.join(", "))
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

    /// Whether users can explicitly select this role.
    #[must_use]
    pub fn is_selectable(&self) -> bool {
        role_info(self).selectable
    }

    /// Whether this role requires a vision-capable (multimodal) model.
    #[must_use]
    pub fn requires_multimodal(&self) -> bool {
        role_info(self).requires_multimodal
    }

    /// Roles that users can switch between via `/agent` command.
    #[must_use]
    pub fn selectable_roles() -> Vec<&'static str> {
        <Role as strum::IntoEnumIterator>::iter()
            .filter(Role::is_selectable)
            .map(|r| r.as_str())
            .collect()
    }

    /// All roles as an iterator.
    #[must_use]
    pub fn all_roles() -> Vec<Role> {
        <Role as strum::IntoEnumIterator>::iter().collect()
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
}

// ── Tool set factory ──────────────────────────────────────────────────────

use crate::Tool;
use crate::tools::{
    AddCommentTool, AskTool, BrowserTool, CreateTicketTool, EditTool, GetTicketTool, ImageGenTool,
    ListTicketsTool, ReadTool, SearchArchivedTicketsTool, SearchTool, ShellMode, ShellTool,
    UpdateTicketTool, VideoGenTool, WebSearchTool,
};

impl Role {
    /// Build the tool set for this role.
    #[must_use]
    pub fn tools(&self) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = match self {
            Role::Engineer => {
                vec![
                    Box::new(ShellTool::new(ShellMode::Full)),
                    Box::new(ReadTool),
                    Box::new(EditTool),
                    Box::new(SearchTool),
                    Box::new(AskTool::new(vec![Role::Analyst, Role::Coder], None)),
                ]
            }
            Role::Manager => {
                vec![
                    Box::new(CreateTicketTool::new("manager")),
                    Box::new(UpdateTicketTool),
                    Box::new(ListTicketsTool),
                    Box::new(GetTicketTool),
                    Box::new(AddCommentTool),
                    Box::new(SearchArchivedTicketsTool),
                ]
            }
            Role::Analyst => {
                vec![
                    Box::new(ReadTool),
                    Box::new(SearchTool),
                    Box::new(BrowserTool::default()),
                    Box::new(ShellTool::new(ShellMode::ReadOnly)),
                ]
            }
            Role::Coder => {
                vec![
                    Box::new(ShellTool::new(ShellMode::Full)),
                    Box::new(ReadTool),
                    Box::new(EditTool),
                    Box::new(SearchTool),
                ]
            }
            Role::Qa | Role::Reviewer | Role::Discovery => {
                vec![
                    Box::new(ReadTool),
                    Box::new(SearchTool),
                    Box::new(ShellTool::new(ShellMode::ReadOnly)),
                ]
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
                vec![
                    Box::new(ReadTool),
                    Box::new(SearchTool),
                    Box::new(AskTool::new(vec![Role::Analyst], None)),
                    Box::new(CreateTicketTool::new("maintainer")),
                    Box::new(ShellTool::new(ShellMode::ReadOnly)),
                ]
            }
        };

        if let Some(key) = crate::config::CONFIG.exa_key() {
            tools.push(Box::new(WebSearchTool::new(key)));
        }

        tools
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn role_roundtrip() {
        // FromStr for every variant by lowercase name
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
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
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
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
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
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
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
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
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
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
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
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
}
