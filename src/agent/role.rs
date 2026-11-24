//! Role metadata consolidation — the single source of truth for all static [`Role`] properties:
//! role descriptions, discovery prompts, tool assignments and [`RoleInfo`]. Used by
//! [`crate::agent`] and anything else that needs role data.

use std::sync::LazyLock;

use strum::IntoEnumIterator;

use crate::Role;

/// Role string for the workspace's project commands — `"diagnostics"`. The value outlives the
/// commands-only phase as a shared key: the shell spill-owner key of a run with no agent identity,
/// the sentinel that collapses already-stored diagnostics comments to a summary in the GUI board,
/// and the diagnostics-discovery agent id / stale-write log label. The
/// literal is compared against `comment.role` and against agent ids, so a change must be
/// coordinated across those sites.
pub(crate) const DIAGNOSTICS_ROLE: &str = "diagnostics";

/// Role string for system comments — `"system"`, used where a ticket has no agent author: the
/// circuit-breaker trip comment `bounce_to_development` posts, and the actor `attributed_actor`
/// substitutes in the chronicle when `last_transition_actor` is absent or blank — so this value
/// reaches a stored `ticket_chronicle.actor`, not only comments. `gui/theme.rs` gives `"system"`
/// the generic unknown-role badge colour on its own; it does not read this constant.
pub(crate) const SYSTEM_ROLE: &str = "system";

// All static metadata for a [`Role`] variant, looked up through a single match in [`role_info()`]
// rather than the scattered matches this replaced. Prompt-loading accessors (`role_description`,
// `role_description_for`, `discovery_prompt`, `summary_prompt`) read the embedded `.md` assets
// directly; icon widgets live in `theme::role_icon()`.
//
// Adding a role: the [`Role`] enum in `lib.rs`, prompts at `src/prompt/role/{name}.md` and
// `src/prompt/summarize/{name}.md` (plus `src/prompt/discovery/{name}.md` when `has_discovery`), an
// arm here, the [`Role::tools()`] arm and the `theme::role_icon()` arm. [`Role::Custom`] is the one
// exception: it has no compiled `role/custom.md` (its prompt is assembled at launch) and no static
// tool set ([`Role::custom_role_tools`] derives it per record), so the guards below exempt it. The
// compiler catches missing arms but not one returning an empty tool set or silently using
// struct-update defaults — the tests here guard those: the [`RoleInfo`] sentinels (a black
// `badge_fg`, an empty `display_label`), non-empty [`Role::tools()`], and real content in the role,
// summary and discovery prompts.

pub struct RoleInfo {
    pub has_discovery: bool,
    /// Badge foreground colour as an RGB tuple, converted to an [`iced::Color`] badge in
    /// `gui/theme.rs`. The badge background is always this colour at 0.1 alpha.
    pub badge_fg: (f32, f32, f32),
    /// Default reasoning effort: authoritative, since effort is baked into the binary and not
    /// user-tunable, so request time always uses this value.
    pub default_reasoning_effort: &'static str,
    pub display_label: &'static str,
}

// Default values shared by most [`Role`] variants, used via `..BASE_ROLE_INFO` to keep each arm
// concise and make future field additions cheap. Arms that spell out every field (Discovery,
// Assistant) do so because clippy's `needless_update` fires when the base contributes nothing.
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
        Role::Worker => &RoleInfo {
            badge_fg: (0.353, 0.604, 0.416),
            display_label: "Worker",
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

/// The marker a transcript carries where the recogniser heard speech and read nothing, as
/// the role prompts spell it: one substitution from the engine's own constant, so the
/// wording the model is taught and the wording a transcript carries cannot drift.
const UNTRANSCRIBED: (&str, &str) = ("{{untranscribed}}", crate::speech::UNTRANSCRIBED);

impl Role {
    /// Canonical role name as a `&'static str` (lowercase), via the [`strum::IntoStaticStr`]
    /// derive.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        self.into()
    }

    #[must_use]
    pub const fn display_label(&self) -> &'static str {
        role_info(self).display_label
    }

    /// Role description, loaded from the embedded prompt files.
    #[must_use]
    pub fn role_description(&self) -> String {
        let media_delivery = crate::prompt::media_delivery();
        crate::prompt::substitute(
            &crate::prompt::load_prompt(&format!("role/{}.md", self.as_str())),
            &[UNTRANSCRIBED, ("{{media_delivery}}", &media_delivery)],
        )
    }

    /// Role description widened for the admin's Assistant: only the Assistant has a separate one
    /// when the triggering account is the admin; every other role ignores `is_admin`.
    #[must_use]
    pub fn role_description_for(&self, is_admin: bool) -> String {
        if *self == Role::Assistant && is_admin {
            let say_note = crate::prompt::load_prompt("role/assistant_admin_say.md");
            let media_delivery = crate::prompt::media_delivery();
            crate::prompt::substitute(
                &crate::prompt::load_prompt("role/assistant_admin.md"),
                &[
                    ("{{say_tool}}", say_note.trim()),
                    UNTRANSCRIBED,
                    ("{{media_delivery}}", &media_delivery),
                ],
            )
        } else {
            self.role_description()
        }
    }

    /// Discovery prompt, loaded from the embedded prompt files.
    ///
    /// # Panics
    ///
    /// Panics if the role has no discovery prompt (see [`RoleInfo::has_discovery`]) — check it
    /// first.
    #[must_use]
    pub fn discovery_prompt(&self) -> String {
        let info = role_info(self);
        if info.has_discovery {
            crate::prompt::load_prompt(&format!("discovery/{}.md", self.as_str()))
        } else {
            panic!("Discovery prompt for role '{self}' does not exist")
        }
    }

    /// Conversation compaction prompt for this role, from `src/prompt/summarize/{role}.md`.
    #[must_use]
    pub fn summary_prompt(&self) -> String {
        crate::prompt::load_prompt(&format!("summarize/{}.md", self.as_str()))
    }

    /// The roles that hold a long-lived, owner-facing session: the admin's and a guest's Assistant,
    /// each project's Manager, and a custom role. Unlike the pipeline roles, whose session ends
    /// with its phase, these are meant to be continued — the roles granted `sleep`, and the ones
    /// whose over-limit sessions compact in the pause after a reply rather than at the next
    /// message.
    #[must_use]
    pub(crate) const fn is_long_lived_chat(self) -> bool {
        matches!(self, Role::Assistant | Role::Manager | Role::Custom)
    }
}

use crate::Tool;
use crate::Workspace;
use crate::config::CONFIG;
use crate::tools::SayTool;
use crate::tools::path::PathAccess;
use crate::tools::{
    AddAutomationTool, AddCommentTool, Addressee, AnalyzeDepth, AnalyzeTool, ChromeTool,
    ComputerTool, CreateTicketTool, CustomTool, DispatchMode, DocumentTool, EditTool, FilesTool,
    GetTicketTool, HyperframesTool, ImageGenTool, ListAutomationsTool, ListTicketsTool,
    MahbotConfigTool, MahbotDebugTool, PhotocraftTool, RasterizeTool, ReadTool, ReferenceTool,
    RemoveAutomationTool, ReportGapTool, SearchArchivedTicketsTool, SearchTool, SendTool,
    ShellMode, ShellTool, ShowTabTool, SleepTool, UpdateTicketTool, VideoGenTool, WebSearchTool,
    WorkspaceTool,
};

/// Tools in the admin Assistant's set that a custom role is never offered or built: the `computer`
/// surface, the `say` voice and the `show_tab` raise are the administrator's Assistant's alone. A
/// stored entry naming one stays on the record and builds nothing.
const CUSTOM_ROLE_WITHHELD_TOOLS: &[&str] = &["computer", "say", "show_tab"];

impl Role {
    /// Core read/search tools — plus the shell in `shell` mode when one is requested — for the
    /// inspector roles (Analyst, QA, Reviewer, Discovery, Maintainer). None of them may mutate the
    /// workspace, so none gets `edit`; they differ only in the shell mode, and a guest's Analyst
    /// loses the shell entirely.
    fn inspector_core_tools(shell: Option<ShellMode>) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> =
            vec![Box::new(ReadTool::general()), Box::new(SearchTool)];
        if let Some(mode) = shell {
            tools.push(Box::new(ShellTool::new(mode)));
        }
        tools
    }

    /// Whether an Analyst running in `ws` keeps its read-only shell. Analysts are built with no
    /// identity of their own (empty user name, `is_admin = false`), so the inherited workspace is
    /// the only carrier of who called — and it is re-read from durable state on every construction,
    /// so the rule holds across restarts. A guest's Assistant is pinned to `personal:<user>`, so
    /// its analysts must lose the shell: that shell has no reading restriction and would be a
    /// direct route into the admin's `shared` folder. Anything else (a managed workspace, the
    /// admin's own personal workspace) keeps it; an unrecognised `personal:` name falls to the
    /// stricter side.
    fn analyst_shell_allowed(ws: &Workspace) -> bool {
        match crate::users::personal_user_name(&ws.name) {
            Some(user) => crate::users::is_admin_name(user),
            None => true,
        }
    }

    fn full_core_tools() -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ShellTool::new(ShellMode::Full)),
            Box::new(ReadTool::general()),
            Box::new(EditTool::confined()),
            Box::new(SearchTool),
        ]
    }

    /// The Assistant's static core tool set: the admin's own minus `web_search` ([`Role::tools`]
    /// appends that afterwards), plus the guest-only `files` and `report_gap`. `is_admin` widens it
    /// with the admin-only shell, send, computer, browser-raise, config, pipeline and voice tools,
    /// and narrows it in two places — `files`, which exists only because a guest has no shell, and
    /// `report_gap`, the guest's only route to the owner. A custom role's derivation starts from
    /// this set minus [`CUSTOM_ROLE_WITHHELD_TOOLS`].
    fn assistant_core_tools(is_admin: bool) -> Vec<Box<dyn Tool>> {
        // The optional `place` argument belongs to the admin's Assistant: a guest's round always
        // runs in its own workspace, and a custom role's dispatch tools come from its own grants.
        let analyze = if is_admin {
            AnalyzeTool::admin(Role::Assistant)
        } else {
            AnalyzeTool::new(DispatchMode::Async, Role::Assistant)
        };
        // The admin's Assistant reaches the machine; a guest's is confined to its own workspace.
        let file_access = if is_admin {
            PathAccess::Unrestricted
        } else {
            PathAccess::Workspace
        };
        let mut t: Vec<Box<dyn Tool>> = vec![
            Box::new(analyze),
            Box::new(AddAutomationTool),
            Box::new(ListAutomationsTool),
            Box::new(RemoveAutomationTool),
            Box::new(EditTool::new(file_access)),
            Box::new(SearchTool),
            Box::new(SleepTool),
            // Media tools serve every Assistant alike: the models they use resolve per-user.
            Box::new(ImageGenTool),
            Box::new(VideoGenTool),
            // The rasterizer serves every account: it reads the file and its resources under the
            // caller's own access and writes into that caller's workspace.
            Box::new(RasterizeTool::new(file_access)),
            // Likewise the document tool: outputs stay in the agent's workspace, input paths follow
            // the access.
            Box::new(DocumentTool::new(file_access)),
            Box::new(HyperframesTool::new(file_access)),
            // PhotoCraft takes no access: it holds no boundary on a path, for the admin and a
            // guest alike, so it is built without one.
            Box::new(PhotocraftTool),
        ];
        // A guest's Assistant has no shell, so arranging its own files needs a tool of its own.
        if !is_admin {
            t.push(Box::new(FilesTool));
            // Nor has it any mail, calendar or shell to offer a user who asks for one, so that
            // gap needs a route to the owner. The report grants nothing — it is a notice.
            t.push(Box::new(ReportGapTool));
        }
        // The grant-gated forwarding tool: the grant (or admin rights) decides what each call may
        // reach.
        t.push(Box::new(CustomTool::for_account()));
        // A guest's read access is workspace-bounded; the admin's reaches any path the shell could
        // name.
        t.push(Box::new(ReadTool::new(file_access)));
        if is_admin {
            t.push(Box::new(ShellTool::new(ShellMode::Full)));
            // The worker and branch surface, the Manager/custom-role addressees, and the one-way
            // handover to a guest's own assistant — one tool, chosen per call.
            t.push(Box::new(SendTool::admin(Role::Assistant)));
            // The admin's session is a trusted local one, so the Assistant may also observe/act on
            // the local GUI. No authorization gate beyond `is_admin`, matching the full shell
            // widening; on macOS the privacy grants belong to the terminal the daemon runs from and
            // are requested per call. A custom role is never offered it.
            t.push(Box::new(ComputerTool));
            // Raising a browser tab takes the owner's focus, and no CLI verb could carry it: a
            // command-line verb would be open to anything that can run a command. So it stays with
            // the admin's Assistant like `computer`, and the chrome tool never holds it because an
            // analyst's tool reaches guest workspaces and has no identity to refuse on.
            t.push(Box::new(ShowTabTool));
            // The admin's settings surface: `mahbot_config` (every setting, the setup, accounts,
            // grants and custom roles) and the read-only DB query tool.
            t.push(Box::new(MahbotConfigTool));
            t.push(Box::new(MahbotDebugTool));
            // The admin's own project view and each workspace's settings: the reading actions, the
            // pipeline controls, and the project's own settings — every action the tool's
            // `reference` lists.
            t.push(Box::new(WorkspaceTool));
            // The voice: one short phrase spoken aloud, for the admin's own Assistant only — a
            // custom role is never offered it.
            t.push(Box::new(SayTool));
        }
        t
    }

    /// The admin Assistant's own set as a custom role sees it: the same list minus
    /// [`CUSTOM_ROLE_WITHHELD_TOOLS`], so neither the catalogue nor the built set can offer a
    /// withheld tool. The `custom` dispatcher is not a grant at all — a role holds it exactly when
    /// its record names at least one script, and [`Self::custom_role_tools`] builds it from those
    /// names — so the base set has none.
    fn custom_role_base_tools() -> Vec<Box<dyn Tool>> {
        Self::assistant_core_tools(true)
            .into_iter()
            .filter(|tool| !CUSTOM_ROLE_WITHHELD_TOOLS.contains(&tool.name()))
            .filter(|tool| tool.name() != CustomTool::NAME)
            .collect()
    }

    /// Build the tool set for this role. Ticket tools are bound to `ws` at construction — every
    /// operation is confined to it, except that the archive search spans the installation and
    /// `get_ticket` opens a prefixed ID from any workspace. `is_admin` widens the Assistant's set
    /// with `shell`, `send`, `computer`, `show_tab`, `mahbot_config`, `mahbot_debug`, `workspace`
    /// and `say`, and removes the guest-only `files` and `report_gap`; every other role's set is
    /// byte-identical whatever its value. [`Tool::is_advertised`] then drops a tool this machine
    /// cannot serve: `computer` without a supported platform backend, `document` and `hyperframes`
    /// without the bun runtime, `chrome` and `show_tab` while the browser daemon is down.
    /// `chrome_sessions` is the run-scoped tracker shared with the agent's `ChromeTool`, so every
    /// session the run opens is handed to the run-end release queue.
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
                    // The Manager often has nothing to say until a ticket advances or an async
                    // analysis lands.
                    Box::new(SleepTool),
                ]
            }
            Role::Analyst => {
                let shell = Self::analyst_shell_allowed(ws).then_some(ShellMode::ReadOnly);
                let mut t = Self::inspector_core_tools(shell);
                t.push(Box::new(ChromeTool::new(chrome_sessions)));
                t
            }
            Role::Worker => Self::full_core_tools(),
            // QA runs the build roles' unrestricted shell so it can start the product and observe
            // it. The mode is the only thing it shares with them: no `edit`, and the no-mutation
            // rule lives in `role/qa.md`.
            Role::Qa => Self::inspector_core_tools(Some(ShellMode::Full)),
            Role::Reviewer | Role::Discovery => {
                Self::inspector_core_tools(Some(ShellMode::ReadOnly))
            }
            Role::Sanitation => {
                // Sanitation deliberately has NO search tools: it inspects and cleans specific
                // filesystem artifacts, and a `search` over the temp tree would index junk and
                // start watchers.
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
            // A custom role's tools come from its own record — the chosen subset of the static
            // Assistant set minus [`CUSTOM_ROLE_WITHHELD_TOOLS`] — so the static factory has none
            // for it.
            Role::Custom => Vec::new(),
        };

        // `Role::Custom` is excluded alongside the two above: it has no static set at all, and
        // `web_search` is appended per record by [`Self::custom_role_tools`].
        if !matches!(self, Role::Manager | Role::Sanitation | Role::Custom) {
            Self::add_web_search_tool(&mut tools);
        }
        // Every role's standing descriptions stay short and point at this one entry point. It is
        // handed the tools this machine actually serves — the same `is_advertised` filter the
        // caller applies — so a role is never told about a capability it cannot call; a custom
        // role's set is per record, so [`Self::custom_role_tools`] adds its own.
        if self != Role::Custom {
            // Only the admin Assistant's toolset holds the owner's `send`, so only its reference
            // carries the detail; a guest's holds no `send` at all. The rule itself lives on
            // [`Role::is_owner_assistant`], so the argument and its detail cannot diverge.
            let shaped = is_admin && self.is_owner_assistant();
            tools.push(Box::new(ReferenceTool::new(Self::holders(&tools), shaped)));
        }

        tools
    }

    /// The tool set of a custom role: the subset of the admin Assistant's core tools (at the
    /// admin's own strength, minus [`CUSTOM_ROLE_WITHHELD_TOOLS`]) named by the record, plus the
    /// configured `web_search`, the granted dispatch tools with their kinds, the `custom`
    /// dispatcher holding the scripts the record names, and the shared `reference` entry point
    /// every set carries. Each entry is `"tool"` (its narrowest kind), `"tool:kind1,kind2"`, or
    /// `"custom:<script>"`. `analyze` and `send` are never taken from the static set — they are
    /// rebuilt from the grants so a role gets exactly its granted depths and addressees. The read
    /// is lenient: an entry the grammar refuses drops itself alone.
    ///
    /// The dispatcher follows the record, not the folder: a role that names a script whose file has
    /// since gone still holds `custom`, and the call is refused when it is made.
    ///
    /// A tool the installation cannot serve right now is simply absent (the record keeps the
    /// choice), as is an entry naming a withheld tool. The caller applies `is_advertised()` to the
    /// result.
    #[must_use]
    pub(crate) fn custom_role_tools(entries: &[String]) -> Vec<Box<dyn Tool>> {
        // One unparsable stored entry drops itself alone, never the record's whole tool set.
        let grants = crate::tools::grants::parse_lossy(entries);
        let granted: Vec<&str> = grants
            .iter()
            .filter_map(crate::tools::grants::Grant::tool)
            .collect();
        let mut tools: Vec<Box<dyn Tool>> = Self::custom_role_base_tools()
            .into_iter()
            // The dispatch tools are rebuilt from their grants below, never taken from the static
            // admin set.
            .filter(|tool| crate::tools::grants::kinds_of(tool.name()).is_none())
            .filter(|tool| granted.contains(&tool.name()))
            .collect();
        // `web_search` is not part of the static set, and its presence depends on the configured
        // provider.
        if granted.contains(&"web_search") {
            Self::add_web_search_tool(&mut tools);
        }
        // The dispatcher is not a grant either: it exists exactly when the record names a script.
        let scripts: Vec<String> = grants
            .iter()
            .filter_map(|grant| grant.script().map(str::to_string))
            .collect();
        if !scripts.is_empty() {
            tools.push(Box::new(CustomTool::held(scripts)));
        }
        for grant in &grants {
            let crate::tools::grants::Grant::Tool { tool, kinds } = grant else {
                continue;
            };
            match tool.as_str() {
                "analyze" => tools.push(Box::new(AnalyzeTool::granted(
                    &kinds
                        .iter()
                        .filter_map(|kind| AnalyzeDepth::parse(kind))
                        .collect::<Vec<_>>(),
                ))),
                "send" => tools.push(Box::new(SendTool::granted(
                    &kinds
                        .iter()
                        .filter_map(|kind| Addressee::parse(kind))
                        .collect::<Vec<_>>(),
                ))),
                _ => {}
            }
        }
        // The shared reference entry point is not a grant: every set has it, so a custom role's
        // short descriptions can point at the same route the built-in roles use. It is added once
        // the whole set is assembled, so it serves exactly the tools this record granted, and no
        // granted `send` takes `extract` — only the owner's own does.
        tools.push(Box::new(ReferenceTool::new(Self::holders(&tools), false)));
        tools
    }

    /// The names the shared reference serves: the tools this machine advertises, since a tool
    /// [`Tool::is_advertised`] withholds is not in the schema the model is sent.
    fn holders(tools: &[Box<dyn Tool>]) -> Vec<&'static str> {
        tools
            .iter()
            .filter(|tool| tool.is_advertised())
            .map(|tool| tool.name())
            .collect()
    }

    /// The catalog a custom role's subset is chosen from: the admin Assistant's static core tool
    /// names minus [`CUSTOM_ROLE_WITHHELD_TOOLS`] (and the `custom` dispatcher, which is not a
    /// grant), plus `"web_search"` — offered even before a provider is configured, so a role keeps
    /// its choice across a provider change. This is what the settings picker shows and
    /// [`crate::custom_role::validate_tools_against`] accepts as a tool; a role's scripts are
    /// separate entries validated against the shared folder.
    #[must_use]
    pub(crate) fn custom_role_tool_catalog() -> Vec<&'static str> {
        let mut names: Vec<&'static str> = Self::custom_role_base_tools()
            .iter()
            .map(|tool| tool.name())
            .collect();
        if !names.contains(&"web_search") {
            names.push("web_search");
        }
        names
    }

    /// Append a web search tool for the configured providers; at most one is registered. An
    /// explicit provider choice is honoured even when the other key is configured; with no choice,
    /// text goes to DeepSeek and pictures to Exa when that key exists. A key missing for the chosen
    /// capability adds no tool. The caller skips this for Manager (which delegates searches to
    /// analysts) and Sanitation (search-free by design).
    fn add_web_search_tool(tools: &mut Vec<Box<dyn Tool>>) {
        let provider = CONFIG.web_search_provider();
        let tool = WebSearchTool::from_config(
            provider.as_deref(),
            CONFIG.deepseek_key(),
            CONFIG.exa_key(),
        );
        if let Some(tool) = tool {
            tools.push(Box::new(tool));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run-scoped chrome session tracker for toolset tests — these tests never open a browser.
    fn test_sessions() -> std::sync::Arc<crate::tools::chrome::ChromeRunSessions> {
        std::sync::Arc::new(crate::tools::chrome::ChromeRunSessions::default())
    }

    /// Every [`RoleInfo`] field a struct update can silently leave at its sentinel is set: a
    /// visible `badge_fg` (never the base `(0,0,0)`) and a non-empty `display_label`, pinned for
    /// QA's spelling.
    #[test]
    fn role_info_sentinels_are_all_set() {
        for role in Role::iter() {
            let info = super::role_info(&role);
            let (r, g, b) = info.badge_fg;
            assert!(
                !(r == 0.0 && g == 0.0 && b == 0.0),
                "{}: badge_fg must not be (0,0,0) — set a visible color",
                role.as_str()
            );
            assert!(
                !info.display_label.is_empty(),
                "{}: display_label must not be empty — set a display_label in role_info()",
                role.as_str()
            );
        }
        assert_eq!(
            super::role_info(&crate::Role::Qa).display_label,
            "QA",
            "QA's display label is not its snake_case name"
        );
    }

    #[test]
    fn custom_role_tools_are_the_chosen_subset_of_the_admin_set() {
        let catalog = Role::custom_role_tool_catalog();
        assert!(
            catalog.contains(&"sleep"),
            "sleep must be selectable — it is how a role ends a quiet turn"
        );
        let mut expected: Vec<&str> = Role::assistant_core_tools(true)
            .iter()
            .map(|tool| tool.name())
            .filter(|name| !CUSTOM_ROLE_WITHHELD_TOOLS.contains(name))
            // The dispatcher is not a grant: a role holds it exactly when it names a script.
            .filter(|name| *name != CustomTool::NAME)
            .collect();
        expected.push("web_search");
        assert_eq!(
            catalog, expected,
            "the catalogue is the admin set minus {CUSTOM_ROLE_WITHHELD_TOOLS:?} and the \
             dispatcher, plus web_search"
        );
        let all_names: Vec<String> = catalog.iter().map(ToString::to_string).collect();
        let built: Vec<&str> = Role::custom_role_tools(&all_names)
            .iter()
            .map(|tool| tool.name())
            .collect();
        // The shared reference is not a grant: every set carries it, so it is the one built name
        // the catalogue does not offer.
        assert!(
            built
                .iter()
                .all(|name| *name == "reference" || catalog.contains(name)),
            "nothing outside the catalog may be built: {built:?}"
        );
        for name in &catalog {
            // `web_search` is offered before a provider is configured and builds only once one is.
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
        // The shared reference is not a grant and every set carries it, so the assertions below
        // read the granted set alone.
        assert_eq!(chosen_names, vec!["read", "reference", "sleep"]);
        let empty: Vec<&str> = Role::custom_role_tools(&[])
            .iter()
            .map(|tool| tool.name())
            .collect();
        assert_eq!(
            empty,
            ["reference"],
            "a record granting nothing builds nothing but the shared reference"
        );

        // Computer use stays with the administrator's Assistant: the record keeps the grant,
        // nothing builds.
        assert!(
            !catalog.contains(&"computer"),
            "computer use must not be offered to a custom role"
        );
        let granted = |entries: &[&str]| -> Vec<String> {
            let entries: Vec<String> = entries.iter().map(|s| (*s).to_string()).collect();
            Role::custom_role_tools(&entries)
                .iter()
                .map(|tool| tool.name().to_string())
                .filter(|name| name != "reference")
                .collect()
        };
        assert!(
            granted(&["computer"]).is_empty(),
            "a stored `computer` grant must build nothing"
        );
        assert_eq!(
            granted(&["read", "computer"]),
            vec!["read"],
            "the withheld entry costs the record nothing else"
        );

        // A script is held individually: the dispatcher follows the record's entries, never the
        // folder, and a bare `custom` entry means nothing for a role.
        assert!(
            !catalog.contains(&CustomTool::NAME),
            "the dispatcher is not a grantable tool"
        );
        assert!(
            !granted(&["custom"]).contains(&CustomTool::NAME.to_string()),
            "a bare `custom` entry must build no dispatcher"
        );
        assert!(
            granted(&["custom:weather"]).contains(&CustomTool::NAME.to_string()),
            "one named script must build the dispatcher"
        );
        assert_eq!(
            granted(&["custom:weather", "read"]),
            vec!["read", "custom"],
            "the dispatcher sits beside the other granted tools"
        );
    }

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
        let workers = granted(&["send:worker,worker_branch"]);
        let send = props(&workers, "send");
        assert_eq!(
            send["addressee"]["enum"],
            serde_json::json!(["worker", "worker_branch"]),
            "only the granted addressees are offered: {send:?}"
        );
        assert!(
            !send.contains_key("name"),
            "no `role` kind, no `name`: {send:?}"
        );
        assert!(
            send.contains_key("branch"),
            "`worker_branch` takes a branch name: {send:?}"
        );
        let branch_only = granted(&["send:worker_branch"]);
        let send = props(&branch_only, "send");
        assert_eq!(
            send["addressee"]["enum"],
            serde_json::json!(["worker_branch"]),
            "a worker_branch-only holder declares the choice it holds: {send:?}"
        );
        assert!(
            send.contains_key("branch"),
            "a worker_branch holder requires its branch — the continuing kind does \
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
            serde_json::json!(["worker"]),
            "a bare entry is the narrowest kind, declared and required like any other"
        );
    }

    #[test]
    fn all_roles_have_tools() {
        // Guards against an empty Vec in `Role::tools()` — the compiler catches missing arms but
        // not one returning `vec![]`. Every role needs a tool.
        for role in Role::iter() {
            // A custom role's tool set is per-record, never static; the static factory has none for
            // it.
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
                // Every tool loads its own description asset, so a misnamed or missing
                // `tool/{name}.md` panics here rather than on the first run. (The reference
                // assets are checked where the full tool list, `web_search` included, is built.)
                for tool in &tools {
                    let _ = tool.spec();
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    fn sanitation_toolset_is_read_shell_and_reference() {
        // Acceptance pin: the Sanitation role advertises NO search tools — even with web search
        // configured.
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
            ["read", "shell", "reference"],
            "Sanitation toolset must be exactly read + read-only shell + the shared reference, \
             got: {names:?}"
        );
    }

    #[test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    fn qa_toolset_is_the_build_shell_without_edit() {
        // Acceptance pin: QA verifies runtime behaviour by starting the product, so it runs the
        // worker shell (Full mode, background included) — but it must not change the workspace, so
        // it gets no `edit`. The prohibition itself is prompt text (`prompt/role/qa.md`), not a
        // tool restriction.
        let snapshot = crate::config::CONFIG.snapshot();
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);
        let ws = crate::workspace::test_ws("test");
        let qa = crate::Role::Qa.tools(&ws, false, test_sessions());
        crate::config::CONFIG.swap(snapshot);

        let names: Vec<&str> = qa.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            ["read", "search", "shell", "reference"],
            "QA toolset must be exactly read + search + the unrestricted shell + the shared \
             reference, got: {names:?}"
        );
        let qa_shell = qa.iter().find(|t| t.name() == "shell").expect("QA shell");
        assert!(
            qa_shell.parameters_schema()["properties"]
                .as_object()
                .is_some_and(|props| props.contains_key("background")),
            "QA's shell must advertise the background argument"
        );
    }

    #[test]
    fn manager_has_quick_analyze_and_no_send() {
        // Acceptance pin: the Manager's single delegation path is a `quick` `analyze` — a deep
        // analysis and every `send` addressee belong to the admin's Assistant. Its analyze declares
        // no `depth` choice and no `place`.
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
        // The Assistant toolset must differ by whether the triggering account is the admin: a
        // guest's has no shell/send/computer, the admin's does.
        let ws = crate::workspace::test_ws("test");
        let guest = crate::Role::Assistant.tools(&ws, false, test_sessions());
        let admin = crate::Role::Assistant.tools(&ws, true, test_sessions());

        // `computer` is admin-only wherever the platform can serve it.
        for &name in &["shell", "send", "computer"] {
            assert!(
                !guest.iter().any(|t| t.name() == name),
                "a guest's Assistant toolset must not contain '{name}'"
            );
            assert!(
                admin.iter().any(|t| t.name() == name),
                "the admin's Assistant toolset must contain '{name}'"
            );
        }
        // `files` is the guest's alone: the admin's Assistant moves files with its shell.
        assert!(
            guest.iter().any(|t| t.name() == "files"),
            "a guest's Assistant needs `files`: it has no shell to arrange its own files"
        );
        assert!(
            !admin.iter().any(|t| t.name() == "files"),
            "the admin's Assistant must not advertise `files` — its shell already moves files"
        );
        // The optional `place` delegation argument is the admin's alone: the admin's `analyze` and
        // `send` advertise it, the guest's `analyze` (a guest has no `send`) does not.
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

    /// One row per specially-restricted tool: it reaches exactly the roles the row's predicate
    /// names and no other, the admin's Assistant included. The predicate is the whole rule — the
    /// custom-role catalogue is a separate derivation with its own pin.
    #[test]
    fn restricted_tools_reach_only_the_roles_that_may_hold_them() {
        fn admin_assistant(role: crate::Role, is_admin: bool) -> bool {
            role == crate::Role::Assistant && is_admin
        }
        fn any_assistant(role: crate::Role, _is_admin: bool) -> bool {
            role == crate::Role::Assistant
        }
        fn guest_assistant(role: crate::Role, is_admin: bool) -> bool {
            role == crate::Role::Assistant && !is_admin
        }
        fn assistant_or_manager(role: crate::Role, _is_admin: bool) -> bool {
            matches!(role, crate::Role::Assistant | crate::Role::Manager)
        }

        /// A tool name and the roles that may hold it.
        type Rule = (&'static str, fn(crate::Role, bool) -> bool);

        let ws = crate::workspace::test_ws("test");
        let rules: Vec<Rule> = vec![
            ("mahbot_config", admin_assistant),
            ("mahbot_debug", admin_assistant),
            ("workspace", admin_assistant),
            ("computer", admin_assistant),
            ("custom", any_assistant),
            ("document", any_assistant),
            // PhotoCraft holds no path boundary, so its access rule is the whole gate: every
            // Assistant — the admin's and a guest's — and no pipeline role, whose work is the ticket
            // rather than the owner's own pictures.
            ("photocraft", any_assistant),
            ("files", guest_assistant),
            ("report_gap", guest_assistant),
            ("sleep", assistant_or_manager),
            ("say", admin_assistant),
            ("show_tab", admin_assistant),
        ];
        for (name, may_hold) in rules {
            for role in Role::iter() {
                for is_admin in [false, true] {
                    let has = role
                        .tools(&ws, is_admin, test_sessions())
                        .iter()
                        .any(|t| t.name() == name);
                    assert_eq!(
                        has,
                        may_hold(role, is_admin),
                        "{}{} `{name}` availability",
                        role.as_str(),
                        if is_admin { " (admin)" } else { "" }
                    );
                }
            }
        }
        assert!(
            !Role::custom_role_tool_catalog().contains(&"say"),
            "a custom role must not be offered the voice"
        );
        assert!(
            !Role::custom_role_tool_catalog().contains(&"show_tab"),
            "a custom role must not be offered the browser raise"
        );
        assert!(
            Role::custom_role_tool_catalog().contains(&"photocraft"),
            "a custom role's base must hold `photocraft`"
        );
    }

    #[test]
    fn analyst_shell_follows_the_summoning_session() {
        // The pipeline's own analysts (managed workspaces) and the owner admin's Assistant keep the
        // read-only shell; every other user's Assistant summons shell-less analysts, an
        // unresolvable name included.
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
            for expected in ["read", "search", "chrome"] {
                assert!(names.contains(&expected), "{name} keeps {expected}");
            }
        }
    }

    #[test]
    fn manager_addressee_only_in_the_admin_assistants_send() {
        // Acceptance pin: the Manager and custom-role addressees belong to the admin's Assistant's
        // `send` (and to any custom role it grants them to), so the choice declaration names
        // exactly the holder's addressees.
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
}
