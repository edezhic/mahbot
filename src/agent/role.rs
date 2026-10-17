//! Role metadata consolidation — single source of truth for all static [`Role`] properties.
//!
//! This module is the canonical home for [`Role`]'s static methods, trait impls,
//! and metadata lookups — including role descriptions, discovery prompts,
//! tool assignments, and [`RoleInfo`]. Used by [`crate::agent`] and other modules
//! that need role data.

use std::sync::LazyLock;

use strum::IntoEnumIterator;

use crate::Role;

/// Role string for the workspace's project commands — `"diagnostics"`.
///
/// The value outlives the retired commands-only phase as a shared key: the
/// shell spill-owner key of a run with no agent identity (`tools/shell/mod.rs`,
/// via `cleanup_agent_spills`), the sentinel that collapses already-stored
/// diagnostics comments to a summary in the GUI board (`gui/board.rs`), and the
/// diagnostics-discovery agent id / stale-write log label (`workspace.rs`).
///
/// The literal is a key: it's compared against `comment.role` (`gui/board.rs`)
/// and against agent ids (the shell spill collision guard in `tools/shell/mod.rs`),
/// so a change must be coordinated across those sites.
pub(crate) const DIAGNOSTICS_ROLE: &str = "diagnostics";

/// Role string for system comments — `"system"`.
///
/// Used only to post non-agent ticket comments: the engineer hard-failure
/// notice (`development.rs`) and the phase-reset / bounce-breaker trip notices
/// (`pipeline/mod.rs`). It is posting-only — no site reads or compares a
/// comment's role against this value, and the GUI badge for `"system"` is an
/// unrelated generic fallback in `gui/theme.rs`.
pub(crate) const SYSTEM_ROLE: &str = "system";

// ── RoleInfo ──────────────────────────────────────────────────────────────

/// All static metadata for a [`Role`] variant.
///
/// Static role metadata lives here and is looked up through a single match in
/// [`role_info()`], replacing the match statements that were previously
/// scattered across the codebase for role metadata lookups. Exception:
/// prompt-loading accessors (`role_description`, `role_description_for`,
/// `discovery_prompt`, `summary_prompt`) read the embedded `.md` assets
/// directly. Icon widgets live in `theme::role_icon()`.
///
/// Adding a new role requires updating the [`Role`] enum in `lib.rs`,
/// creating prompt files at `src/prompt/role/{name}.md` and
/// `src/prompt/summarize/{name}.md` (and optionally
/// `src/prompt/discovery/{name}.md` if `has_discovery` is true),
/// adding an arm in this match, the [`Role::tools()`] method,
/// and the `theme::role_icon()` match.
/// [`Role::Custom`] is the one exception: it has no compiled
/// `role/custom.md` asset (its system prompt is assembled at launch from the
/// administrator's own record) and no static tool set
/// ([`Role::custom_role_tools`] derives it per record), so the tool-set and
/// role-description guards below exempt it while every other guard applies.
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
/// concise and make future field additions cheap. Arms that spell out every
/// field (Discovery, Assistant) do so because clippy's
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
        Role::Custom => &RoleInfo {
            has_discovery: false,
            badge_fg: (0.62, 0.47, 0.86),
            default_reasoning_effort: "high",
            display_label: "Custom",
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

    /// Role description for this role, widened for the admin's Assistant.
    ///
    /// Only the Assistant has a separate description when the triggering
    /// account is the admin; every other role returns its canonical
    /// [`Role::role_description`] regardless of `is_admin`.
    #[must_use]
    pub fn role_description_for(&self, is_admin: bool) -> String {
        if *self == Role::Assistant && is_admin {
            crate::prompt::load_prompt("role/assistant_admin.md")
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

    /// The roles that hold a long-lived, owner-facing session: the admin's and a
    /// guest's Assistant, each project's Manager, and a custom role. Unlike the
    /// pipeline roles, whose session ends with its phase, these are meant to be
    /// continued — the roles granted `sleep`, and the ones whose over-limit
    /// sessions compact in the pause after a reply rather than at the next
    /// message.
    #[must_use]
    pub(crate) const fn is_long_lived_chat(self) -> bool {
        matches!(self, Role::Assistant | Role::Manager | Role::Custom)
    }
}

// ── Tool set factory ──────────────────────────────────────────────────────

use crate::Tool;
use crate::Workspace;
use crate::config::CONFIG;
#[cfg(not(target_os = "macos"))]
use crate::tools::ComputerTool;
use crate::tools::path::PathAccess;
use crate::tools::{
    AddAlarmTool, AddCommentTool, Addressee, AnalyzeDepth, AnalyzeTool, ChromeTool,
    CreateTicketTool, CustomTool, DispatchMode, DocumentTool, EditTool, GetTicketTool,
    ImageGenTool, ListAlarmsTool, ListTicketsTool, MahbotConfigTool, MahbotDebugTool, ReadTool,
    RemoveAlarmTool, SearchArchivedTicketsTool, SearchTool, SendTool, ShellMode, ShellTool,
    SleepTool, UpdateTicketTool, VectorRasterizeTool, VideoEditTool, VideoGenTool,
    WebSearchBackend, WebSearchTool, WorkspaceTool,
};

impl Role {
    /// Core read/search tools — plus the shell in `shell` mode when one is
    /// requested — for the inspector roles (Analyst, QA, Reviewer, Discovery,
    /// Maintainer). None of them may mutate the workspace, so none gets `edit`;
    /// they differ only in the shell mode, and an Analyst in a guest's personal
    /// workspace loses the shell entirely (see [`Self::analyst_shell_allowed`]).
    fn inspector_core_tools(shell: Option<ShellMode>) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> =
            vec![Box::new(ReadTool::general()), Box::new(SearchTool)];
        if let Some(mode) = shell {
            tools.push(Box::new(ShellTool::new(mode)));
        }
        tools
    }

    /// Whether an Analyst running in `ws` keeps its read-only shell.
    ///
    /// Analysts are built with no identity of their own (an empty user name and
    /// `is_admin = false`), so the workspace they inherit from the summoning
    /// session is the only carrier of who called — and it is re-read from
    /// durable state on every construction, so the rule holds across restarts.
    /// A guest's Assistant is pinned to `personal:<user>`, so its analysts
    /// must lose the shell: that shell has no reading restriction and would be
    /// a direct route into the admin's `shared` folder the per-user grants
    /// exist to gate. Anything else — a project workspace (the pipeline's own
    /// analysts), or the admin's own personal workspace — keeps it; an
    /// unrecognised `personal:` name falls to the stricter side.
    fn analyst_shell_allowed(ws: &Workspace) -> bool {
        match crate::users::personal_user_name(&ws.name) {
            Some(user) => crate::users::is_admin_name(user),
            None => true,
        }
    }

    /// Core full-shell/read/edit/search tools for the build roles
    /// (Engineer, Coder).
    fn full_core_tools() -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ShellTool::new(ShellMode::Full)),
            Box::new(ReadTool::general()),
            Box::new(EditTool::confined()),
            Box::new(SearchTool),
        ]
    }

    /// The Assistant's static core tool set — everything the admin's Assistant
    /// gets except `web_search`, which [`Role::tools`] appends afterwards.
    ///
    /// `is_admin` widens the set exactly as it does for the Assistant: the
    /// admin-only shell / send / computer / config
    /// / pipeline tools, whose access level and `analyze` flavour also follow it.
    fn assistant_core_tools(is_admin: bool) -> Vec<Box<dyn Tool>> {
        // The optional `place` argument belongs to the admin's Assistant: a
        // guest's round always runs in its own workspace, and a custom role's
        // dispatch tools are rebuilt from its own grants
        // ([`Role::custom_role_tools`]), never taken from this set.
        let analyze = if is_admin {
            AnalyzeTool::admin(Role::Assistant)
        } else {
            AnalyzeTool::new(DispatchMode::Async, Role::Assistant)
        };
        // The admin's Assistant reaches the machine; a guest's is
        // confined to its own workspace (see `PathAccess`).
        let file_access = if is_admin {
            PathAccess::Unrestricted
        } else {
            PathAccess::Workspace
        };
        let mut t: Vec<Box<dyn Tool>> = vec![
            Box::new(analyze),
            Box::new(AddAlarmTool),
            Box::new(ListAlarmsTool),
            Box::new(RemoveAlarmTool),
            Box::new(EditTool::new(file_access)),
            Box::new(SearchTool),
            Box::new(SleepTool),
            // Media tools serve every Assistant alike: the media
            // models they use resolve per-user, so no access-level
            // gate applies.
            Box::new(ImageGenTool),
            Box::new(VideoGenTool),
            Box::new(VideoEditTool),
            // Likewise, the vector-file action serves every account: it reads the
            // drawing — and every resource the drawing refers to — under the
            // caller's own access, and writes into that caller's workspace.
            Box::new(VectorRasterizeTool::new(file_access)),
            // Likewise, the document tool serves every account: an agent's
            // own call keeps its outputs inside that agent's workspace, and
            // its input paths follow the caller's access.
            Box::new(DocumentTool::new(file_access)),
        ];
        // The grant-gated forwarding tool: every Assistant can call
        // admin-authored scripts; the grant (or admin rights) decides
        // what each call may reach.
        t.push(Box::new(CustomTool));
        // A guest's Assistant read access is workspace-bounded; the
        // admin's reaches any path the shell could name. (An agent's own
        // media call writes only into the workspace's generated/ tree in
        // both cases.)
        t.push(Box::new(ReadTool::new(file_access)));
        if is_admin {
            t.push(Box::new(ShellTool::new(ShellMode::Full)));
            // The coder and the branch surface, and the Manager/custom-role
            // addressees — one tool, chosen per call.
            t.push(Box::new(SendTool::admin(Role::Assistant)));
            // The admin's session is a trusted local session, so the
            // Assistant can also observe/act on the local GUI directly
            // (gated per-run by the accessibility grant at agent
            // construction). No structural authorization gate beyond
            // `is_admin`, matching the full shell widening. Not on
            // macOS, where computer use is withdrawn entirely
            // (see `tools/computer/macos.rs`).
            #[cfg(not(target_os = "macos"))]
            t.push(Box::new(ComputerTool));
            // The admin config surface: the merged `mahbot_config`
            // setup tool and the read-only DB query tool. The admin's
            // Assistant is now the single user-facing role, so these
            // migrated over from the removed Support role.
            t.push(Box::new(MahbotConfigTool));
            t.push(Box::new(MahbotDebugTool));
            // The admin's own project view: list / add / board / ticket /
            // comment / pause / resume over the registered workspaces.
            t.push(Box::new(WorkspaceTool));
        }
        t
    }

    /// Build the tool set for this role.
    ///
    /// Ticket tools are bound to `ws` at construction time — all their
    /// operations are confined to that workspace, except that the archive
    /// search spans the installation and `get_ticket` opens a prefixed ID
    /// from any workspace.
    ///
    /// `is_admin` says whether the triggering account is the admin. It only
    /// widens the Assistant's toolset: `shell`, `send`,
    /// `computer`, `mahbot_config`, `mahbot_debug` and
    /// `workspace` are added.
    /// Every other role's toolset is byte-identical regardless of its value.
    /// `computer` exists only where the capability does: on macOS it is
    /// withdrawn, so the admin's toolset there has no such tool (see
    /// `tools/computer/macos.rs`).
    ///
    /// `chrome_sessions` is the run-scoped tracker shared with the agent's
    /// `ChromeTool`: every session the run opens is handed to the run-end
    /// release queue (see `crate::agent::RunEndCleanup`).
    #[must_use]
    pub(crate) fn tools(
        self,
        ws: &Workspace,
        is_admin: bool,
        chrome_sessions: std::sync::Arc<crate::tools::chrome::ChromeRunSessions>,
    ) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = match self {
            Role::Engineer => {
                let mut t = Self::full_core_tools();
                t.push(Box::new(AnalyzeTool::new(
                    DispatchMode::Sync,
                    Role::Engineer,
                )));
                t.push(Box::new(SendTool::engineer()));
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
                    Box::new(SearchArchivedTicketsTool),
                    Box::new(AnalyzeTool::new(DispatchMode::Async, Role::Manager)),
                    // The Manager often has nothing to say until a ticket
                    // advances or an async analysis lands — sleep
                    // ends the turn instead of a premature/no-op reply.
                    Box::new(SleepTool),
                ]
            }
            Role::Analyst => {
                let shell = Self::analyst_shell_allowed(ws).then_some(ShellMode::ReadOnly);
                let mut t = Self::inspector_core_tools(shell);
                t.push(Box::new(ChromeTool::new(chrome_sessions)));
                t
            }
            Role::Coder => Self::full_core_tools(),
            // QA runs the build roles' unrestricted shell so it can start the
            // product or its service and observe it while it runs. The mode is
            // the only thing it shares with them: no `edit`, and the
            // no-mutation rule lives in `prompt/role/qa.md`.
            Role::Qa => Self::inspector_core_tools(Some(ShellMode::Full)),
            Role::Reviewer | Role::Discovery => {
                Self::inspector_core_tools(Some(ShellMode::ReadOnly))
            }
            Role::Sanitation => {
                // Sanitation deliberately has NO search tools (local `search`
                // or `web_search`): the role inspects and cleans specific
                // filesystem artifacts and never needs to index the temp tree
                // or hit the web — for the periodic temp cleaner a
                // `search` over the whole temp folder would index junk and
                // start filesystem watchers. Read + read-only shell cover
                // inspection and temp-root mutation.
                vec![
                    Box::new(ReadTool::general()),
                    Box::new(ShellTool::new(ShellMode::ReadOnly)),
                ]
            }
            Role::Maintainer => {
                let mut t = Self::inspector_core_tools(Some(ShellMode::ReadOnly));
                t.push(Box::new(AnalyzeTool::new(
                    DispatchMode::Sync,
                    Role::Maintainer,
                )));
                t.push(Box::new(CreateTicketTool::new("maintainer", ws)));
                t
            }
            Role::Assistant => Self::assistant_core_tools(is_admin),
            // A custom role's tools come from its own record — the
            // administrator's chosen subset of this static Assistant set
            // ([`Role::custom_role_tools`], catalogued by
            // [`Role::custom_role_tool_catalog`]) — so the static factory has
            // none for it. The agent is built from the record at launch, never
            // from here.
            Role::Custom => Vec::new(),
        };

        // `Role::Custom` is excluded alongside the two above: it has no static
        // tool set at all, and `web_search` is appended per record by
        // [`Self::custom_role_tools`], never here.
        if !matches!(self, Role::Manager | Role::Sanitation | Role::Custom) {
            Self::add_web_search_tool(&mut tools);
        }

        tools
    }

    /// The tool set of a custom role: the subset of the admin Assistant's core
    /// tools (at the admin's own strength) named by the role's record, plus the
    /// configured `web_search` and the granted dispatch tools with their kinds.
    ///
    /// Each stored entry is `"tool"` (its narrowest kind) or `"tool:kind1,kind2"`
    /// (see [`crate::tools::grants`]). The two dispatch tools (`analyze`, `send`)
    /// are never taken from the static admin set — they are rebuilt from the
    /// grants so a role gets exactly its granted depths/addressees.
    ///
    /// The read is lenient ([`crate::tools::grants::parse_lossy`]): a stored
    /// entry the grammar refuses drops itself alone, so it cannot cost the role
    /// the rest of its tool set.
    ///
    /// A chosen tool the installation cannot serve right now (a platform-gated
    /// or configuration-dependent tool) is simply absent from the built list —
    /// the record keeps the choice, and the launch carries on with what it can
    /// serve. The caller applies `is_advertised()` to the result (as
    /// [`crate::agent::role_tools_and_specs`] does for every role).
    #[must_use]
    pub(crate) fn custom_role_tools(entries: &[String]) -> Vec<Box<dyn Tool>> {
        // One unparsable stored entry drops itself alone, never the record's
        // whole tool set.
        let grants = crate::tools::grants::parse_lossy(entries);
        let granted: Vec<&str> = grants.iter().map(|grant| grant.tool.as_str()).collect();
        let mut tools: Vec<Box<dyn Tool>> = Self::assistant_core_tools(true)
            .into_iter()
            // The dispatch tools are rebuilt from their grants below, never taken
            // from the static admin set (the grants carry the kinds).
            .filter(|tool| crate::tools::grants::kinds_of(tool.name()).is_none())
            .filter(|tool| granted.contains(&tool.name()))
            .collect();
        // `web_search` is not part of the static admin set, and its presence
        // depends on the configured provider.
        if granted.contains(&"web_search") {
            Self::add_web_search_tool(&mut tools);
        }
        for grant in &grants {
            match grant.tool.as_str() {
                "analyze" => {
                    tools.push(Box::new(AnalyzeTool::granted(
                        &grant
                            .kinds
                            .iter()
                            .filter_map(|kind| AnalyzeDepth::parse(kind))
                            .collect::<Vec<_>>(),
                    )));
                }
                "send" => {
                    tools.push(Box::new(SendTool::granted(
                        &grant
                            .kinds
                            .iter()
                            .filter_map(|kind| Addressee::parse(kind))
                            .collect::<Vec<_>>(),
                    )));
                }
                _ => {}
            }
        }
        tools
    }

    /// The catalog a custom role's tool subset is chosen from: the admin
    /// Assistant's static core tool names plus `"web_search"` — offered even
    /// before a provider is configured, so a role keeps its choice across a
    /// provider change. This is the list the settings picker shows and
    /// [`crate::custom_role::validate_tools_against`] accepts.
    #[must_use]
    pub(crate) fn custom_role_tool_catalog() -> Vec<&'static str> {
        let mut names: Vec<&'static str> = Self::assistant_core_tools(true)
            .iter()
            .map(|tool| tool.name())
            .collect();
        if !names.contains(&"web_search") {
            names.push("web_search");
        }
        names
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

    /// Run-scoped chrome session tracker for toolset tests — the caller-visible
    /// per-run tracker is never used here (these tests never open a browser).
    fn test_sessions() -> std::sync::Arc<crate::tools::chrome::ChromeRunSessions> {
        std::sync::Arc::new(crate::tools::chrome::ChromeRunSessions::default())
    }

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

    /// A custom role's tools are exactly the chosen subset of the admin
    /// Assistant's own set: a servable name builds, a name the installation
    /// cannot serve is simply absent (the record keeps it), and `sleep` is part
    /// of what a role may choose.
    #[test]
    fn custom_role_tools_are_the_chosen_subset_of_the_admin_set() {
        let catalog = Role::custom_role_tool_catalog();
        assert!(
            catalog.contains(&"sleep"),
            "sleep must be selectable — it is how a role ends a quiet turn"
        );
        let all_names: Vec<String> = catalog.iter().map(ToString::to_string).collect();
        let built: Vec<&str> = Role::custom_role_tools(&all_names)
            .iter()
            .map(|tool| tool.name())
            .collect();
        assert!(
            built.iter().all(|name| catalog.contains(name)),
            "nothing outside the catalog may be built: {built:?}"
        );
        for name in &catalog {
            // `web_search` is offered before a provider is configured and is
            // built only once one is — every other name always builds.
            if *name == "web_search" {
                continue;
            }
            assert!(built.contains(name), "'{name}' must build");
        }

        let chosen = Role::custom_role_tools(&[
            "read".to_string(),
            "sleep".to_string(),
            "not-a-tool".to_string(),
        ]);
        let mut chosen_names: Vec<&str> = chosen.iter().map(|tool| tool.name()).collect();
        chosen_names.sort_unstable();
        assert_eq!(chosen_names, vec!["read", "sleep"]);
        assert!(Role::custom_role_tools(&[]).is_empty());
    }

    /// A granted dispatch tool is built with exactly the kinds its entry names,
    /// declared and required: a kind the holder does not have is absent from its
    /// interface, and a one-kind holder still states its choice.
    #[test]
    fn granted_dispatch_tools_offer_exactly_the_granted_kinds() {
        let granted = |entries: &[&str]| {
            let entries: Vec<String> = entries.iter().map(|s| (*s).to_string()).collect();
            Role::custom_role_tools(&entries)
        };
        let props =
            |tools: &[Box<dyn Tool>], name: &str| -> serde_json::Map<String, serde_json::Value> {
                let tool = tools
                    .iter()
                    .find(|t| t.name() == name)
                    .unwrap_or_else(|| panic!("'{name}' must build"));
                tool.parameters_schema()["properties"]
                    .as_object()
                    .cloned()
                    .unwrap_or_default()
            };
        let deep = granted(&["analyze:deep"]);
        let analyze = props(&deep, "analyze");
        assert!(
            analyze.get("depth").is_none(),
            "a one-depth analyze declares no choice: {analyze:?}"
        );
        assert!(
            analyze.get("place").is_none(),
            "the deep depth takes no folder: {analyze:?}"
        );
        let both = granted(&["analyze:quick,deep"]);
        assert_eq!(
            props(&both, "analyze")["depth"]["enum"],
            serde_json::json!(["quick", "deep"])
        );
        let coders = granted(&["send:coder,coder_branch"]);
        let send = props(&coders, "send");
        assert_eq!(
            send["addressee"]["enum"],
            serde_json::json!(["coder", "coder_branch"]),
            "only the granted addressees are offered: {send:?}"
        );
        assert!(
            !send.contains_key("name"),
            "no `role` kind, no `name`: {send:?}"
        );
        assert!(
            send.contains_key("branch"),
            "`coder_branch` takes a branch name: {send:?}"
        );
        let branch_only = granted(&["send:coder_branch"]);
        let send = props(&branch_only, "send");
        assert_eq!(
            send["addressee"]["enum"],
            serde_json::json!(["coder_branch"]),
            "a coder_branch-only holder declares the choice it holds: {send:?}"
        );
        assert!(
            send.contains_key("branch"),
            "a coder_branch holder requires its branch — the continuing kind does \
             not imply the single run: {send:?}"
        );
        let manager = granted(&["send:manager"]);
        let send = props(&manager, "send");
        assert_eq!(
            send["addressee"]["enum"],
            serde_json::json!(["manager"]),
            "a manager-only holder declares the choice it holds: {send:?}"
        );
        assert!(
            send.get("place").is_some(),
            "a manager-only send requires its project: {send:?}"
        );
        let bare = granted(&["send"]);
        assert_eq!(
            props(&bare, "send")["addressee"]["enum"],
            serde_json::json!(["coder"]),
            "a bare entry is the narrowest kind, declared and required like any other"
        );
    }

    #[test]
    fn all_roles_have_tools() {
        // Guards against an empty Vec in Role::tools() — the compiler catches
        // missing arms in the match, but cannot catch an arm that returns
        // vec![]. Every role needs at least one tool to function, and every
        // advertised tool must build its spec: `spec()` loads
        // `src/prompt/tool/{name}.md`, so a misnamed description asset panics
        // here instead of at the first agent run.
        for role in Role::iter() {
            // A custom role's tool set is per-record, never static
            // (`Role::custom_role_tools`); the static factory has none for it.
            if role == Role::Custom {
                continue;
            }
            for is_admin in [false, true] {
                let tools = role.tools(
                    &crate::workspace::test_ws("test"),
                    is_admin,
                    test_sessions(),
                );
                assert!(
                    !tools.is_empty(),
                    "{}: Role::tools() must not be empty — every role needs at least one tool",
                    role.as_str()
                );
                for tool in &tools {
                    let _ = tool.spec();
                }
            }
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
            .tools(&crate::workspace::test_ws("test"), false, test_sessions())
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
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    fn qa_toolset_is_the_build_shell_without_edit() {
        // Acceptance pin: QA verifies runtime behaviour by starting the product
        // or its service, so it runs the engineer/coder shell (Full mode,
        // background included) — but it must not change the workspace, so it
        // gets no `edit`. The prohibition itself is prompt text
        // (`prompt/role/qa.md`), not a tool restriction.
        let snapshot = crate::config::CONFIG.snapshot();
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);
        let ws = crate::workspace::test_ws("test");
        let qa = crate::Role::Qa.tools(&ws, false, test_sessions());
        let engineer = crate::Role::Engineer.tools(&ws, false, test_sessions());
        crate::config::CONFIG.swap(snapshot);

        let names: Vec<&str> = qa.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            ["read", "search", "shell"],
            "QA toolset must be exactly read + search + the unrestricted shell, got: {names:?}"
        );
        let qa_shell = qa.iter().find(|t| t.name() == "shell").expect("QA shell");
        let engineer_shell = engineer
            .iter()
            .find(|t| t.name() == "shell")
            .expect("engineer shell");
        assert_eq!(
            qa_shell.description(),
            engineer_shell.description(),
            "QA must run the same shell the engineer gets"
        );
        assert!(
            qa_shell.parameters_schema()["properties"]
                .as_object()
                .is_some_and(|props| props.contains_key("background")),
            "QA's shell must advertise the background argument"
        );
    }

    #[test]
    fn manager_has_quick_analyze_and_no_send() {
        // Acceptance pin: the Manager's single delegation path is a `quick`
        // `analyze` — a deep analysis and every `send` addressee belong to the
        // admin's Assistant. The Manager's analyze declares no `depth` choice
        // (it holds only `quick`) and no `place`.
        let tools =
            crate::Role::Manager.tools(&crate::workspace::test_ws("test"), false, test_sessions());
        let analyze = tools
            .iter()
            .find(|t| t.name() == "analyze")
            .expect("the Manager holds `analyze`");
        let props = analyze.parameters_schema()["properties"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        assert!(
            !props.contains_key("depth"),
            "a single-depth analyze must not declare `depth`: {props:?}"
        );
        assert!(!props.contains_key("place"), "the Manager has no `place`");
        assert!(
            !tools.iter().any(|t| t.name() == "send"),
            "Manager toolset must not contain 'send'"
        );
    }

    #[test]
    fn assistant_toolset_gates_admin_tools() {
        // The Assistant toolset must differ by whether the triggering account
        // is the admin: a guest's Assistant has no shell/send/computer, the
        // admin's does. A guest gets the workspace-only
        // read access; the admin's reaches the machine.
        let ws = crate::workspace::test_ws("test");
        let guest = crate::Role::Assistant.tools(&ws, false, test_sessions());
        let admin = crate::Role::Assistant.tools(&ws, true, test_sessions());

        // `computer` is admin-only where the capability exists at all; on macOS
        // it is withdrawn entirely (see `tools/computer/macos.rs`).
        let admin_only: &[&str] = if cfg!(target_os = "macos") {
            &["shell", "send"]
        } else {
            &["shell", "send", "computer"]
        };
        for &name in admin_only {
            assert!(
                !guest.iter().any(|t| t.name() == name),
                "a guest's Assistant toolset must not contain '{name}'"
            );
            assert!(
                admin.iter().any(|t| t.name() == name),
                "the admin's Assistant toolset must contain '{name}'"
            );
        }
        if cfg!(target_os = "macos") {
            assert!(
                !admin.iter().any(|t| t.name() == "computer"),
                "the admin's Assistant must not advertise `computer` on macOS"
            );
        }
        // The read boundary is what the model is told — the tool's own
        // description, not a restatement in its parameter schema: a guest
        // (restricted) only reaches the personal workspace; the admin's names
        // the machine and what stays closed.
        let guest_read = guest.iter().find(|t| t.name() == "read");
        let guest_read_desc = guest_read.map(|t| t.description()).unwrap_or_default();
        assert!(
            guest_read_desc.contains("limited to your own personal workspace"),
            "a guest's Assistant read must advertise the workspace-only boundary, got: \
             {guest_read_desc}"
        );
        let admin_read_desc = admin
            .iter()
            .find(|t| t.name() == "read")
            .map(|t| t.description())
            .unwrap_or_default();
        assert!(
            admin_read_desc.contains("your home directory")
                && admin_read_desc.contains("own live databases"),
            "the admin's Assistant read must advertise the machine-wide boundary and what it \
             refuses, got: {admin_read_desc}"
        );

        // The write boundary likewise: the admin's edit names the two refusals,
        // a guest's edit states the workspace boundary.
        let admin_edit_desc = admin
            .iter()
            .find(|t| t.name() == "edit")
            .map(|t| t.description())
            .unwrap_or_default();
        assert!(
            admin_edit_desc.contains("registered project workspaces")
                && admin_edit_desc.contains("home directory"),
            "the admin's Assistant edit must name what stays closed, got: {admin_edit_desc}"
        );
        let guest_edit_desc = guest
            .iter()
            .find(|t| t.name() == "edit")
            .map(|t| t.description())
            .unwrap_or_default();
        assert!(
            guest_edit_desc.contains("confined to your workspace"),
            "a guest's Assistant edit must keep the workspace boundary, got: {guest_edit_desc}"
        );
        for (who, desc) in [
            ("the admin's read", &admin_read_desc),
            ("the admin's edit", &admin_edit_desc),
            ("a guest's read", &guest_read_desc),
            ("a guest's edit", &guest_edit_desc),
        ] {
            assert!(
                !desc.contains("{{"),
                "{who} description left a placeholder unrendered: {desc}"
            );
        }

        // The optional `place` delegation argument is the admin's alone: the
        // admin's `analyze` and `send` advertise it, the guest's `analyze`
        // (the guest has no `send`) does not.
        let advertises_place = |tools: &[Box<dyn Tool>], name: &str| {
            tools.iter().find(|t| t.name() == name).is_some_and(|t| {
                t.parameters_schema()["properties"]
                    .as_object()
                    .is_some_and(|props| props.contains_key("place"))
            })
        };
        assert!(
            advertises_place(&admin, "analyze"),
            "the admin's Assistant analyze must advertise `place`"
        );
        assert!(
            advertises_place(&admin, "send"),
            "the admin's Assistant send must advertise `place`"
        );
        assert!(
            !advertises_place(&guest, "analyze"),
            "a guest's Assistant analyze must not advertise `place`"
        );
    }

    #[test]
    fn admin_only_tools_only_in_admin_assistant() {
        // Acceptance pin: the admin's administrative tools — the merged
        // `mahbot_config` setup tool, the read-only `mahbot_debug` query tool and
        // `workspace` (the assistant's own project view: list / add / board /
        // ticket / comment / pause / resume) — live in the admin's Assistant's own
        // static tool set and in no other role's static set. A guest's Assistant
        // and every other role must never advertise any of them here. (A custom
        // role may additionally choose them from the catalogue derived from that
        // admin set, `assistant_core_tools(true)`, which this static-set pin does
        // not cover.)
        let ws = crate::workspace::test_ws("test");
        for role in Role::iter() {
            for is_admin in [false, true] {
                let names: Vec<&str> = role
                    .tools(&ws, is_admin, test_sessions())
                    .iter()
                    .map(|t| t.name())
                    .collect();
                for name in ["mahbot_config", "mahbot_debug", "workspace"] {
                    let has = names.contains(&name);
                    if role == crate::Role::Assistant && is_admin {
                        assert!(has, "the admin's Assistant must advertise `{name}`");
                    } else {
                        assert!(
                            !has,
                            "{}{} must not advertise `{name}`",
                            role.as_str(),
                            if is_admin { " (admin)" } else { "" }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn custom_tool_is_advertised_by_every_assistant_only() {
        // Acceptance pin: the single forwarding tool goes to every Assistant —
        // the admin's and a guest's alike, since the grants (not the toolset)
        // decide what each call may reach — and to no other role.
        let ws = crate::workspace::test_ws("test");
        for role in Role::iter() {
            for is_admin in [false, true] {
                let has = role
                    .tools(&ws, is_admin, test_sessions())
                    .iter()
                    .any(|t| t.name() == "custom");
                assert_eq!(
                    has,
                    role == crate::Role::Assistant,
                    "{}{} custom-tool availability",
                    role.as_str(),
                    if is_admin { " (admin)" } else { "" }
                );
            }
        }
    }

    #[test]
    fn document_tool_is_advertised_by_every_assistant_only() {
        // Acceptance pin: document work is built into EVERY Assistant — the
        // owner's and a guest's alike, with no grant or permission in between —
        // and into no other role. (The managed runtime's absence hides it at
        // advertisement time, which is a separate per-agent decision; this pins
        // the toolset itself.)
        let ws = crate::workspace::test_ws("test");
        for role in Role::iter() {
            for is_admin in [false, true] {
                let has = role
                    .tools(&ws, is_admin, test_sessions())
                    .iter()
                    .any(|t| t.name() == "document");
                assert_eq!(
                    has,
                    role == crate::Role::Assistant,
                    "{}{} document-tool availability",
                    role.as_str(),
                    if is_admin { " (admin)" } else { "" }
                );
            }
        }
    }

    #[test]
    fn analyst_shell_follows_the_summoning_session() {
        // The pipeline's own analysts (project workspaces) and the owner
        // admin's Assistant keep the read-only shell; every other user's
        // Assistant summons shell-less analysts, including an unresolvable
        // personal name (fail closed).
        for name in ["proj", "personal:admin"] {
            let ws = crate::workspace::test_ws_named("/tmp/proj", name);
            let has = Role::Analyst
                .tools(&ws, false, test_sessions())
                .iter()
                .any(|t| t.name() == "shell");
            assert!(has, "analysts in {name} must keep the shell");
        }
        for name in ["personal:lyuba", "personal:"] {
            let ws = crate::workspace::test_ws_named("/tmp/personal", name);
            let names: Vec<&str> = Role::Analyst
                .tools(&ws, false, test_sessions())
                .iter()
                .map(|t| t.name())
                .collect();
            assert!(
                !names.contains(&"shell"),
                "analysts in {name} must not get a shell, got {names:?}"
            );
            // Everything else about them is unchanged.
            for expected in ["read", "search", "chrome"] {
                assert!(names.contains(&expected), "{name} keeps {expected}");
            }
        }
    }

    #[test]
    fn computer_tool_only_in_admin_assistant() {
        // Acceptance pin: `computer` lives in the admin's Assistant's own
        // static tool set and in no other role's. Every other role (and a
        // guest's Assistant) relies on delegation (chrome/analyze/shell) rather
        // than direct local GUI access, so the tool must never leak into their
        // static toolset. (A custom role may choose `computer` from the
        // catalogue derived from that admin set.) On macOS the capability is
        // withdrawn entirely (see `tools/computer/macos.rs`), so no role
        // advertises it there at all.
        let ws = crate::workspace::test_ws("test");
        for role in Role::iter() {
            for is_admin in [false, true] {
                let has = role
                    .tools(&ws, is_admin, test_sessions())
                    .iter()
                    .any(|t| t.name() == "computer");
                if cfg!(target_os = "macos") {
                    assert!(
                        !has,
                        "{}{} must not advertise `computer` on macOS",
                        role.as_str(),
                        if is_admin { " (admin)" } else { "" }
                    );
                } else if role == crate::Role::Assistant && is_admin {
                    assert!(has, "the admin's Assistant must advertise `computer`");
                } else {
                    assert!(
                        !has,
                        "{}{} must not advertise `computer`",
                        role.as_str(),
                        if is_admin { " (admin)" } else { "" }
                    );
                }
            }
        }
    }

    #[test]
    fn manager_addressee_only_in_the_admin_assistants_send() {
        // Acceptance pin: the Manager and custom-role addressees belong to the
        // admin's Assistant's `send` (and any custom role it grants them to).
        // The choice declaration names exactly the addressees the holder has, so
        // manager+role being offered pins this surface.
        let ws = crate::workspace::test_ws("test");
        for role in Role::iter() {
            for is_admin in [false, true] {
                let Some(send) = role
                    .tools(&ws, is_admin, test_sessions())
                    .into_iter()
                    .find(|t| t.name() == "send")
                else {
                    continue;
                };
                let props = send.parameters_schema()["properties"]
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                let holder_is_admin_assistant = role == crate::Role::Assistant && is_admin;
                let kinds = props
                    .get("addressee")
                    .and_then(|choice| choice.get("enum"))
                    .and_then(|kinds| kinds.as_array())
                    .cloned()
                    .unwrap_or_default();
                let offers_manager_or_role = ["manager", "role"]
                    .iter()
                    .all(|kind| kinds.iter().any(|entry| entry == kind));
                assert_eq!(
                    offers_manager_or_role,
                    holder_is_admin_assistant,
                    "{}{} send addressee surface",
                    role.as_str(),
                    if is_admin { " (admin)" } else { "" }
                );
            }
        }
    }

    #[test]
    fn sleep_tool_only_in_assistant_and_manager_toolsets() {
        // Acceptance pin on the STATIC per-role sets: `sleep` is granted only to
        // the Assistant (the admin's and a guest's) and the Manager — every
        // other built-in role always has pending work and must not end its turn.
        // A custom role is the deliberate exception: its set is the
        // administrator's choice (taken from the Assistant's own, which carries
        // `sleep`), so it has no static arm to pin — `Role::tools()` returns an
        // empty vec for it by design.
        let ws = crate::workspace::test_ws("test");
        for role in Role::iter() {
            for is_admin in [false, true] {
                let has = role
                    .tools(&ws, is_admin, test_sessions())
                    .iter()
                    .any(|t| t.name() == "sleep");
                if matches!(role, crate::Role::Assistant | crate::Role::Manager) {
                    assert!(
                        has,
                        "{} (is_admin={is_admin}) must advertise `sleep`",
                        role.as_str()
                    );
                } else {
                    assert!(
                        !has,
                        "{}{} must not advertise `sleep`",
                        role.as_str(),
                        if is_admin { " (admin)" } else { "" }
                    );
                }
            }
        }
    }

    #[test]
    fn assistant_role_descriptions_are_populated() {
        // Both Assistant descriptions (a guest's + the admin's) must be
        // non-empty, free of unsubstituted template keys, and carry the
        // alarm-notification marker.
        for is_admin in [false, true] {
            let desc = crate::Role::Assistant.role_description_for(is_admin);
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
            // A custom role has no compiled `role/custom.md`: its system prompt
            // is assembled at launch from the administrator's record, so only
            // the built-in roles carry a description asset.
            if role == Role::Custom {
                continue;
            }
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
        // Guards against an empty or unsubstituted summary prompt file, and
        // against a role losing the no-tools / summary-only instruction the
        // summarisation request depends on (its tool set is sent, so the
        // model must be told not to use it).
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
            assert!(
                prompt.contains("DO NOT USE ANY TOOLS. ONLY RESPOND WITH THE SUMMARY."),
                "{}: summary prompt must forbid tools and allow only the summary",
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
