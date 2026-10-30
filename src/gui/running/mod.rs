//! "Running Agents" dashboard page — a live view of every currently-running
//! agent and in-flight non-agent LLM work, grouped by WORKSPACE, then by
//! DIRECT PARENT INVOCATION (ticket / analyze round / research run / branch
//! launch / workspace singleton / unattributable orchestrator calls).
//!
//! The view is read-only: at render time it reads the in-memory registries
//! (`AGENT_REGISTRY` and `NON_AGENT_CALLS`) plus the live transcript snapshots
//! (`TRANSCRIPT_REGISTRY`), so it reflects the live in-memory conversation
//! including the unpersisted tail. No database reads, no schema changes, no
//! history retained between ticks. Re-renders are driven by coalesced runtime
//! change events (agent/non-agent registries, live transcript content, voice
//! status) — the GUI refreshes as activity happens, not on a fixed cadence.
//!
//! Truthfulness rules:
//! - The card's first row is the role icon followed inline by the token/elapsed
//!   metrics (collapsed and expanded alike), with no fill spacers; the rest of
//!   the card content (trace-group lines, tool blocks, activity line) starts on
//!   the second row at the card's left edge. Every trace group renders one line
//!   about what the agent is doing, derived from that group's newest step by
//!   [`trace_line`] — see [`visible_lines`] for how a group with no text of its
//!   own holds the line already on the card, and when the neutral "clanking"
//!   placeholder appears instead. The live activity phase label (a
//!   non-tool LLM call) renders as a separate line at the very bottom of the
//!   card in both the collapsed and expanded state.
//!   In-flight tool state is derived from the live transcript history: a tool
//!   call whose result has not been committed yet renders with a live
//!   in-flight marker. Only non-tool LLM phases remain invisible in the
//!   transcript and are still shown solely by the registry activity label.
//! - Trace groups (below the LIVE line) are a projection of the shared
//!   session ledger (the live transcript snapshot decoded via
//!   `session_view::build_ledger`): tool-call assistant rounds grouped by
//!   shared narration, bounded by real user turns. Content-only final answers
//!   are omitted.
//! - The trace groups shown on a card are scoped to the CURRENT user turn:
//!   groups from before the last real user message are never shown.
//! - Parallel tool execution is a history-derived over-approximation: a call
//!   shows as in-flight while its result is uncommitted (its tool-call round
//!   is already in the ledger) — finished-but-uncommitted calls in a parallel
//!   batch, unknown/pre-flight-cancelled calls before their round commits,
//!   and drain-suspended calls all appear in-flight. This is accepted for an
//!   observability view; tools that never produce a committed round simply
//!   never appear.
//! - The instrumentation is purely observational — it never affects
//!   shutdown/drain logic and gains no cancellation semantics.
//! - The live view shows tool arguments EXACTLY as the agent passed them —
//!   including any secrets. This is a deliberate, documented divergence from
//!   the durable logs (tool-call stats, failure feedback), which remain
//!   credential-scrubbed: the live view is transient and exists for full
//!   visibility of what the agent is doing.
//!
//! The page is DB-free by design: everything it shows is derived from the
//! live in-memory registries and the live transcript snapshots, plus the
//! dashboard's managed-workspace map (only for the "(external)" marker on
//! sections for workspaces outside the managed set — personal and ephemeral
//! ones). All header labels (ticket titles, analyze/research questions) are
//! captured observationally at spawn — never read from the DB.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::ChatRole;
use crate::agent::registry::{AgentHandle, NonAgentCallHandle, ParentKey};
use crate::gui::dialog;
use crate::gui::session_view::{
    MAX_TOOL_TOOLTIP_WIDTH, SessionEntry, ToolBlockView, ToolCallEntry, build_ledger, tool_block,
    truncate_at_boundary,
};
use crate::gui::theme;
use crate::gui::widgets;
use chrono::{DateTime, Utc};

use iced::widget::{Column, Row, Space, button, column, container, mouse_area, row, text, tooltip};
use iced::{Alignment, Element, Length};

use iced_fonts::lucide;

use super::Message;

use crate::Workspace;

mod trace_line;

/// How many collapsed previous trace groups the expanded card shows, besides
/// the current one.
const MAX_PREVIOUS_GROUPS: usize = 5;

/// Maximum display length (Unicode chars) of an analyze/research group header
/// label (the question/task text). Truncated at a word/path-delimiter boundary
/// with "…" (see [`truncate_at_boundary`]).
const MAX_GROUP_LABEL_CHARS: usize = 104;

/// Maximum width (px) of a metric hover tooltip body. Much narrower than
/// [`MAX_TOOL_TOOLTIP_WIDTH`] (tuned for multi-arg tool calls) since a metric
/// tooltip holds one short line of explanatory text.
const MAX_METRIC_TOOLTIP_WIDTH: f32 = 360.0;

/// What a Running-Agents stop action stops: a research run or one branch
/// launch. Carried by the messages and the confirmation state; never rendered
/// as text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopTarget {
    Research(String),
    Launch(String),
}

/// Messages emitted by the Running Agents page.
///
/// The page itself stays stateless — the pending stop confirmation lives on the
/// [`Dashboard`](super::Dashboard) so the dialog requires no per-run page state
/// beyond the pending confirmation itself.
#[derive(Debug, Clone)]
pub(crate) enum RunningMessage {
    /// Stop button pressed on a research-run or branch group header.
    /// Carries the target whose key (run key / launch agent id) is NEVER
    /// rendered as text; the button action alone carries it.
    StopRequest(StopTarget),
    /// The confirmation dialog's danger button — start the async stop.
    StopConfirmed,
    /// Keep/Dismiss (or Escape/backdrop) — close the dialog with no effect.
    StopDismissed,
    /// The async stop finished; carries the toast outcome the task produced.
    StopFinished(Result<String, String>),
    /// Toggle the expanded/collapsed state of a running-agent card. Keyed by
    /// (agent_id, generation) so a recycled agent_id never inherits a stale
    /// expansion.
    ToggleAgentExpanded { agent_id: String, generation: u64 },
}

/// Render the live running-agents page.
///
/// The page is observational — it reads the live in-memory registries and
/// renders directly into the dashboard's [`Message`] type; its only
/// self-emitted messages are the manual-stop flow for research runs and
/// branch launches ([`RunningMessage`]) and the card expand/collapse
/// toggle. `workspaces` is the dashboard's managed-workspace map (name →
/// info) — used only to mark sections for workspaces outside the managed set
/// — personal and ephemeral ones — with the "(external)" suffix. Everything
/// else comes from the live in-memory registries.
///
/// `expanded` is the dashboard's set of expanded agent cards, keyed by
/// (agent_id, generation). Stale keys are pruned by the dashboard's
/// `RuntimeChanged` handler against the freshly-listed agents; rendering here
/// only reads the set.
pub(crate) fn view(
    workspaces: &HashMap<String, Workspace>,
    pending_stop: Option<&StopTarget>,
    expanded: &HashSet<(String, u64)>,
) -> Element<'static, Message> {
    let agents = crate::agent::registry::AGENT_REGISTRY.list();
    let calls = crate::agent::registry::NON_AGENT_CALLS.list();

    // Workspace-first sections: each section holds its groups, groups
    // sorted by kind (tickets → analyze rounds → research runs → branch
    // launches → singletons → unattributed), sections alphabetically by name.
    let sections = build_sections(build_groups(&agents, &calls), workspaces);

    let body: Element<'_, RunningMessage> = if sections.is_empty() {
        // Shared empty-state pattern: large radar glyph (the page's nav
        // icon) + label, centered.
        widgets::empty_state_placeholder(
            lucide::radar::<iced::Theme, iced::Renderer>(),
            "Nothing is currently running.",
            theme::TEXT_MUTED,
        )
    } else {
        let mut content = Column::new().spacing(theme::SPACE_20);
        for section in &sections {
            content = content.push(render_section(section, expanded));
        }
        widgets::vscroll(content)
    };

    // Confirm-dialog overlay: a type-stable placeholder while closed, so the
    // page keeps its widget state when the dialog opens and closes.
    let confirm_layer: Element<'_, RunningMessage> = if let Some(target) = pending_stop {
        stop_confirm_dialog(target)
    } else {
        widgets::no_dialog()
    };
    widgets::page_with_dialog(body, confirm_layer).map(Message::RunningAgents)
}

// ── Display model ─────────────────────────────────────────────────────────

/// One running work item: either an agent card or a non-agent LLM call row.
/// The agent card is boxed — `AgentHandle` is large and the variant would
/// otherwise dominate the enum's size (clippy::large_enum_variant).
#[derive(Debug, Clone)]
enum DisplayItem {
    Agent(Box<AgentCard>),
    Call(CallRow),
}

#[derive(Debug, Clone)]
struct AgentCard {
    handle: AgentHandle,
}

#[derive(Debug, Clone)]
struct CallRow {
    handle: NonAgentCallHandle,
}

/// Ordered group kinds — the page's display order.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GroupKind {
    Ticket,
    AnalyzeRound,
    Research,
    Launch,
    Singleton,
    Unattributed,
}

#[derive(Debug, Clone)]
struct DisplayGroup {
    kind: GroupKind,
    /// Group key (ticket id / round key / run key / launch agent id /
    /// workspace name).
    key: String,
    /// Workspace name for the group header.
    workspace: String,
    /// Human-readable header label for the group (ticket title / analyze
    /// question / research question), captured observationally from any
    /// member's `parent_label`. `None` for singletons/unattributed and for
    /// parent-keyed groups whose members carried no label (defensive fallback
    /// — the header then degrades to a generic label, never a raw key).
    label: Option<String>,
    items: Vec<DisplayItem>,
    /// True when this research group carries a run-lifetime orchestrator
    /// marker (rendered as a run-lifetime indicator, not a call row).
    run_lifetime: bool,
}

/// One workspace section on the page: the groups running in one workspace,
/// keyed by the raw workspace name and ordered by its resolved label.
#[derive(Debug, Clone)]
struct WorkspaceSection {
    /// Raw workspace name — the grouping key.
    workspace: String,
    /// Resolved display label: the bare name for managed workspaces, the
    /// "(external)"-suffixed form for workspaces outside the managed set
    /// (personal and ephemeral ones), and "workspace" for an empty name.
    label: String,
    groups: Vec<DisplayGroup>,
}

impl DisplayGroup {
    /// (order, key) — groups sort by kind order first, then by key within
    /// the kind (stable, deterministic).
    fn sort_key(&self) -> (u8, String) {
        let order = match self.kind {
            GroupKind::Ticket => 0,
            GroupKind::AnalyzeRound => 1,
            GroupKind::Research => 2,
            GroupKind::Launch => 3,
            GroupKind::Singleton => 4,
            GroupKind::Unattributed => 5,
        };
        (order, self.key.clone())
    }
}

/// Map a parent key to its display group kind — the DIRECT PARENT INVOCATION
/// the work item belongs to.
fn parent_group_kind(parent: &ParentKey) -> GroupKind {
    match parent {
        ParentKey::Ticket(_) => GroupKind::Ticket,
        ParentKey::AnalyzeRound(_) => GroupKind::AnalyzeRound,
        ParentKey::Research(_) => GroupKind::Research,
        ParentKey::Launch(_) => GroupKind::Launch,
    }
}

/// The group key payload for a parent key (ticket id / round key / run key /
/// launch agent id).
fn parent_group_key(parent: &ParentKey) -> &str {
    match parent {
        ParentKey::Ticket(id) => id,
        ParentKey::AnalyzeRound(key) | ParentKey::Research(key) | ParentKey::Launch(key) => key,
    }
}

/// Group the running agents and non-agent calls by DIRECT PARENT INVOCATION.
///
/// - Tickets: agents with a [`ParentKey::Ticket`] plus calls with the same
///   ticket parent.
/// - Analyze rounds: agents + calls sharing an [`ParentKey::AnalyzeRound`] key.
/// - Research runs: agents + calls sharing an [`ParentKey::Research`] key.
/// - Branch launches: agents + calls sharing a [`ParentKey::Launch`]
///   key (the launch's agent id).
/// - Workspace singletons: agents with no parent key (manager / maintainer /
///   discovery / direct chat).
/// - Unattributed: non-agent calls with no parent key (workspace-scoped).
///
/// Group identity is (kind, key) plus — for tickets/analyze rounds/singletons —
/// the workspace: ticket ids already embed the workspace name and analyze-round
/// keys are scoped to one workspace, so the extra component closes any
/// cross-workspace key collision (e.g. two concurrent sync analyze rounds in
/// different workspaces drawing the same short NanoID suffix). Research and
/// branch-launch groups are matched on (kind, key) alone because a run/launch
/// moves between workspaces:
/// - a research run's worker round executes in an EPHEMERAL per-run workspace
///   whose name IS the run key (job_id) while its analysts and orchestrator
///   calls run in the real workspace — a workspace component would split the
///   group;
/// - a branch launch's folder is a property of the LAUNCH, not of the branch
///   identity, so a branch (a custom role's or a worker's) pointed at a
///   different folder keeps one group: a launch group is keyed by the branch
///   address alone, so a member whose recorded workspace differs must join the
///   existing group instead of splitting into a second section.
///
/// The run key / launch agent id is a durable unique id, so (kind, key) stays
/// unique across workspaces; the group's displayed workspace is resolved to the
/// REAL workspace whenever a non-ephemeral member is seen.
fn build_groups(agents: &[AgentHandle], calls: &[NonAgentCallHandle]) -> Vec<DisplayGroup> {
    let mut groups: Vec<DisplayGroup> = Vec::new();

    let find_group =
        |kind: GroupKind, key: &str, workspace: &str, groups: &mut Vec<DisplayGroup>| -> usize {
            let research = kind == GroupKind::Research;
            let workspace_agnostic = research || kind == GroupKind::Launch;
            if let Some(idx) = groups.iter().position(|g| {
                g.kind == kind && g.key == key && (workspace_agnostic || g.workspace == workspace)
            }) {
                if research && groups[idx].workspace == key && workspace != key {
                    // The group currently shows the ephemeral per-run
                    // workspace; adopt the joining member's real workspace.
                    groups[idx].workspace = workspace.to_string();
                }
                return idx;
            }
            groups.push(DisplayGroup {
                kind,
                key: key.to_string(),
                workspace: workspace.to_string(),
                label: None,
                items: Vec::new(),
                run_lifetime: false,
            });
            groups.len() - 1
        };

    // Adopt a member's parent label into the group (first non-None wins —
    // all members of one invocation carry the same label; the fallback keeps
    // a group whose only remaining member is label-less from degrading).
    let adopt_label = |label: &Option<String>, group: &mut DisplayGroup| {
        if group.label.is_none() && label.is_some() {
            group.label.clone_from(label);
        }
    };

    for agent in agents {
        let workspace = agent.workspace_name.clone();
        if let Some(parent) = &agent.parent_key {
            let idx = find_group(
                parent_group_kind(parent),
                parent_group_key(parent),
                &workspace,
                &mut groups,
            );
            adopt_label(&agent.parent_label, &mut groups[idx]);
            groups[idx]
                .items
                .push(DisplayItem::Agent(Box::new(AgentCard {
                    handle: agent.clone(),
                })));
        } else {
            // Workspace singleton. An empty workspace name groups under the
            // empty key; the section label resolver renders it as "workspace".
            let idx = find_group(GroupKind::Singleton, &workspace, &workspace, &mut groups);
            groups[idx]
                .items
                .push(DisplayItem::Agent(Box::new(AgentCard {
                    handle: agent.clone(),
                })));
        }
    }

    for call in calls {
        let workspace = call.workspace.clone();
        if let Some(parent) = &call.parent_key {
            let idx = find_group(
                parent_group_kind(parent),
                parent_group_key(parent),
                &workspace,
                &mut groups,
            );
            adopt_label(&call.parent_label, &mut groups[idx]);
            // Run-lifetime orchestrator markers render as a run-lifetime
            // indicator on the group, not as a transient call card.
            if call.run_lifetime {
                groups[idx].run_lifetime = true;
            } else {
                groups[idx].items.push(DisplayItem::Call(CallRow {
                    handle: call.clone(),
                }));
            }
        } else {
            let idx = find_group(GroupKind::Unattributed, &workspace, &workspace, &mut groups);
            groups[idx].items.push(DisplayItem::Call(CallRow {
                handle: call.clone(),
            }));
        }
    }

    groups
}

/// Group the display groups into WORKSPACE SECTIONS — the page is organized
/// by workspace, whose resolved label is used only for deterministic ordering
/// (the label no longer renders as a per-section heading).
///
/// Sections are ordered alphabetically by their resolved label (deterministic
/// — activity-based ordering would flicker as agents come and go); within a
/// section, groups keep the canonical kind order (tickets → analyze rounds →
/// research runs → branch launches → singletons → unattributed) via
/// [`DisplayGroup::sort_key`].
fn build_sections(
    groups: Vec<DisplayGroup>,
    workspaces: &std::collections::HashMap<String, Workspace>,
) -> Vec<WorkspaceSection> {
    let mut sections: Vec<WorkspaceSection> = Vec::new();
    for group in groups {
        let label = workspace_label_for(&group.workspace, workspaces);
        if let Some(section) = sections.iter_mut().find(|s| s.workspace == group.workspace) {
            section.groups.push(group);
        } else {
            sections.push(WorkspaceSection {
                workspace: group.workspace.clone(),
                label,
                groups: vec![group],
            });
        }
    }
    sections.sort_by(|a, b| a.label.cmp(&b.label));
    for section in &mut sections {
        section.groups.sort_by_key(DisplayGroup::sort_key);
    }
    sections
}

// ── Rendering ─────────────────────────────────────────────────────────────

/// Render one workspace section: just the section's groups. The workspace
/// name no longer gets a visual heading — `section.label` exists only for the
/// deterministic section ordering in [`build_sections`].
///
/// The returned element owns all rendered content (text widgets take owned
/// Strings), so its lifetime is independent of the `section` borrow.
fn render_section(
    section: &WorkspaceSection,
    expanded: &HashSet<(String, u64)>,
) -> Element<'static, RunningMessage> {
    let mut groups = Column::new().spacing(theme::SPACE_10);
    for group in &section.groups {
        groups = groups.push(render_group(group, expanded));
    }
    groups.into()
}

/// Resolve a group's header parts: the primary label plus an optional ticket
/// id. `render_group` composes a single flat header — for ticket groups
/// `[{id}] {title}`, otherwise just the label.
///
/// - Ticket groups: the ticket NAME plus the ticket ID. When no title was
///   captured (defensive — every live ticket group carries one via its agents
///   or the synthesis call), the key becomes the sole label.
/// - Analyze/research groups: the truncated question/task text; a generic
///   fallback when no label was captured. The raw NanoID key is NEVER shown.
/// - Branch launches: the display label built at spawn (a custom role's name
///   plus its branch identifier when named, or a worker branch's own label); a
///   generic fallback when absent.
/// - Singleton/unattributed groups: generic labels — their workspace name is
///   already the section grouping, so repeating it would be redundant.
fn group_title(group: &DisplayGroup) -> (String, Option<String>) {
    match &group.kind {
        GroupKind::Ticket => match &group.label {
            Some(title) => (title.clone(), Some(group.key.clone())),
            None => (group.key.clone(), None),
        },
        GroupKind::AnalyzeRound => (
            group.label.as_deref().map_or_else(
                || "Analyze round".to_string(),
                |l| truncate_at_boundary(l, MAX_GROUP_LABEL_CHARS),
            ),
            None,
        ),
        GroupKind::Research => (
            group.label.as_deref().map_or_else(
                || "Research run".to_string(),
                |l| truncate_at_boundary(l, MAX_GROUP_LABEL_CHARS),
            ),
            None,
        ),
        GroupKind::Launch => (
            group
                .label
                .as_deref()
                .map_or_else(|| "Launch".to_string(), str::to_string),
            None,
        ),
        GroupKind::Singleton => ("Standalone".to_string(), None),
        GroupKind::Unattributed => ("Other LLM work".to_string(), None),
    }
}

/// Render one group: header (flat label + run-lifetime marker) then its
/// cards/rows directly on the page background — each agent card carries its
/// own card style; call rows are unadorned by design.
fn render_group(
    group: &DisplayGroup,
    expanded: &HashSet<(String, u64)>,
) -> Element<'static, RunningMessage> {
    let (title, id) = group_title(group);
    let mut header_parts: Vec<Element<'_, RunningMessage>> = Vec::new();
    // Ticket groups render a single flat `[{id}] {title}` header — the id is
    // composed into the text, never styled as a separate ACCENT element. When
    // no title was captured (defensive), the key is the sole label. All other
    // groups render their label as the same flat line.
    let header = if group.kind == GroupKind::Ticket {
        match id {
            Some(id) => format!("[{id}] {title}"),
            None => title.clone(),
        }
    } else {
        title.clone()
    };
    header_parts.push(
        text(header)
            .size(theme::TEXT_12)
            .color(theme::TEXT_SECONDARY)
            .into(),
    );
    if group.run_lifetime {
        header_parts.push(
            text("run active")
                .size(theme::TEXT_11)
                .color(theme::TEXT_SECONDARY)
                .into(),
        );
    }
    // Manual stop: danger-styled button on RESEARCH-run group headers ("Cancel
    // run" — the run is removed permanently) and on BRANCH launch headers
    // ("Stop launch" — the branch is marked stopped, not erased). The group key
    // (run key / launch agent id) is carried by the message, never rendered as
    // text.
    let stop = match group.kind {
        GroupKind::Research => Some(("Cancel run", StopTarget::Research(group.key.clone()))),
        GroupKind::Launch => Some(("Stop launch", StopTarget::Launch(group.key.clone()))),
        _ => None,
    };
    if let Some((label, target)) = stop {
        header_parts.push(Space::new().width(Length::Fill).into());
        header_parts.push(
            button(text(label).size(theme::TEXT_11))
                .style(theme::button_danger)
                .on_press(RunningMessage::StopRequest(target))
                .into(),
        );
    }

    let mut items = Column::new().spacing(theme::SPACE_6);
    for item in &group.items {
        match item {
            DisplayItem::Agent(card) => {
                let is_expanded =
                    expanded.contains(&(card.handle.agent_id.clone(), card.handle.generation));
                items = items.push(render_agent_card(card, is_expanded));
            }
            DisplayItem::Call(call) => {
                items = items.push(render_call_row(call));
            }
        }
    }

    column![
        Row::with_children(header_parts)
            .spacing(theme::SPACE_8)
            .align_y(Alignment::Center),
        items,
    ]
    .spacing(theme::SPACE_6)
    .into()
}

/// Build the manual-stop confirmation dialog for a research run or a
/// branch launch, mirroring the board's ticket-cancel pattern:
/// consequences listed, danger confirm button, Keep/Dismiss (and Escape/
/// backdrop) to close with no effect.
///
/// The target's key (run job id / launch agent id) is NEVER displayed as text —
/// the button action alone carries it (see the enum docs).
fn stop_confirm_dialog(target: &StopTarget) -> Element<'static, RunningMessage> {
    let (title, body, keep, confirm): (&str, &[&str], &str, &str) = match target {
        StopTarget::Research(_) => (
            "Cancel this research run?",
            &[
                "Confirming will:",
                "• stop all agents of this run — an in-flight tool or LLM call \
                 may finish, but no further work happens;\n\
                 • stop the orchestrator — no more rounds, no report, and \
                 nothing is delivered to the Manager;\n\
                 • remove the run's temporary traces and its run folder — \
                 dispatching the cleanup agent while the orchestrator is still \
                 running, a direct sweep otherwise;\n\
                 • remove the run permanently — it can never resume.",
            ],
            "Keep run",
            "Cancel run",
        ),
        StopTarget::Launch(_) => (
            "Stop this launch?",
            &[
                "Confirming will:",
                "• stop the launch — an in-flight tool or LLM call may finish, \
                 but no further work happens;\n\
                 • mark the branch as deliberately stopped — automatic recovery \
                 never restarts it;\n\
                 • drop the launch's pending messages — they are not replayed \
                 when the service restarts.",
            ],
            "Keep launch",
            "Stop launch",
        ),
    };
    widgets::modal_backdrop(
        dialog::confirm_dialog(
            dialog::dialog_title(title),
            dialog::dialog_body(body.iter().copied()),
            [
                dialog::DialogAction::secondary(keep, RunningMessage::StopDismissed),
                dialog::DialogAction::danger(confirm, RunningMessage::StopConfirmed),
            ],
        ),
        RunningMessage::StopDismissed,
        0.5,
    )
}

/// Fallback label for workspaces outside the dashboard's managed set
/// (personal user spaces, ephemeral run-scoped workspaces).
fn fallback_workspace_label(workspace: &str) -> String {
    if workspace.is_empty() {
        "workspace".to_string()
    } else {
        format!("{workspace} (external)")
    }
}

/// Resolve a workspace's display label, used for deterministic section
/// ordering: the bare name for managed workspaces, the fallback label
/// (with the "(external)" marker) for workspaces outside the managed set
/// (personal and ephemeral ones), and "workspace" for an empty name.
fn workspace_label_for(
    name: &str,
    workspaces: &std::collections::HashMap<String, Workspace>,
) -> String {
    if name.is_empty() {
        "workspace".to_string()
    } else if workspaces.contains_key(name) {
        name.to_string()
    } else {
        fallback_workspace_label(name)
    }
}

/// Render one running-agent card: the role icon and the token/elapsed metrics
/// inline in the first row, then the current user turn's trace groups. The
/// whole card is clickable to expand/collapse the trace groups.
///
/// Each trace group renders one line about what the agent is doing, following
/// that group's newest step (see [`trace_line`]); no line a newer group already
/// shows is repeated, and the neutral "clanking" placeholder appears only while
/// the card has no line at all (see [`visible_lines`]). The live activity phase
/// label (an in-flight non-tool LLM call, e.g.
/// extracting/summarizing/transcribing) renders as a separate accent line at the
/// very bottom of the card in both the collapsed and expanded state. Trace groups
/// come from the live transcript snapshot (the unpersisted tail included): the
/// current (latest) group, plus up to [`MAX_PREVIOUS_GROUPS`] previous groups
/// when expanded.
fn render_agent_card(card: &AgentCard, expanded: bool) -> Element<'static, RunningMessage> {
    let h = &card.handle;
    // Color resolution via the canonical string helper (handles derivative
    // names and falls back to muted grey for unknown roles); the icon still
    // needs a typed Role, so parse with the same Engineer fallback as before.
    let (fg, _bg) = theme::role_badge_color(&h.role);
    let role: crate::Role = h.role.parse().unwrap_or(crate::Role::Engineer);
    let icon = theme::role_icon(&role).size(theme::TEXT_24).color(fg);
    let elapsed = format_elapsed(h.started_at);

    // The live transcript snapshot feeds the trace groups below (and their
    // lines), scoped to the current user turn, so fetch it once up front. The
    // groups borrow the ledger, so it outlives them within this render.
    let snapshot = crate::session::TRANSCRIPT_REGISTRY.snapshot(&h.agent_id);
    let ledger = snapshot
        .as_ref()
        .map(|snapshot| build_ledger(&snapshot.history))
        .unwrap_or_default();
    let groups = derive_trace_groups(current_turn_entries(&ledger));

    let token_count = snapshot.as_ref().and_then(|s| s.token_count);

    let mut content = Column::new()
        .spacing(theme::SPACE_6)
        .align_x(Alignment::Start)
        .width(Length::Fill);
    // The card's first row is the role icon followed inline by the token /
    // elapsed metrics, sitting at the left edge of the card (no fill spacers).
    let first_row = Row::new()
        .spacing(theme::SPACE_12)
        .align_y(Alignment::Center)
        .push(icon)
        .push(render_metrics(token_count, &elapsed));
    content = content.push(first_row);
    // The line each visible group renders, and — while no group has one at all
    // (a brand-new agent, a new user turn, or a turn whose groups carry no text
    // yet) — the card-level "clanking" placeholder standing in for it.
    let lines = visible_lines(&groups, expanded);
    if !lines.iter().any(|line| matches!(line, GroupLine::Line(_))) {
        content = content.push(clanking_placeholder());
    }
    // Trace groups from the live snapshot, scoped to the current user turn.
    // Content-only assistant messages (final answers) are deliberately
    // omitted — only tool-call assistant messages form groups.
    for group in render_visible_groups(&groups, lines) {
        content = content.push(group);
    }
    // The live activity phase label (an in-flight non-tool LLM call) renders
    // as a separate accent line at the very bottom of the card, below the
    // trace groups, in both the collapsed and expanded view.
    if let Some(activity) = &h.activity {
        content = content.push(
            text(activity.to_owned())
                .size(theme::TEXT_13)
                .color(theme::ACCENT),
        );
    }

    let on_press = RunningMessage::ToggleAgentExpanded {
        agent_id: h.agent_id.clone(),
        generation: h.generation,
    };
    // The whole card is a click target (expand/collapse), but it stays a
    // `container` under a transparent `mouse_area` so the inner tool blocks
    // keep their hover tooltips (a `button` wrapper would swallow them).
    mouse_area(
        container(content)
            .width(Length::Fill)
            .padding(theme::PAD_8)
            .style(theme::surface_card_style),
    )
    .on_press(on_press)
    .into()
}

/// The typography of a trace group's line: italic at
/// [`theme::NARRATION_TEXT_SIZE`], wrapping within the row. The text is already
/// one line of whole sentences within the card's character budget (see
/// [`trace_line::line_for_step`]); only a sentence longer than the budget is cut.
fn trace_line_text(line: &str) -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    text(line.to_owned())
        .size(theme::NARRATION_TEXT_SIZE)
        .font(theme::FONT_ITALIC)
        .color(theme::TEXT_SECONDARY)
        .wrapping(iced::widget::text::Wrapping::WordOrGlyph)
}

/// The neutral "clanking" placeholder: the card has no line to show yet — a
/// brand-new agent, a new user turn, or a turn whose groups are all textless.
fn clanking_placeholder() -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    text("clanking")
        .size(theme::TEXT_13)
        .color(theme::TEXT_SECONDARY)
}

/// Render the card's metrics: the live session-token count (when known) and
/// the elapsed run time, each as a compact labeled tooltip. Renders inline in
/// the card's first row, right after the role icon (see
/// [`render_agent_card`]).
fn render_metrics(token_count: Option<u64>, elapsed: &str) -> Element<'static, RunningMessage> {
    let mut metrics = Row::new()
        .spacing(theme::SPACE_12)
        .align_y(Alignment::Center);
    // The live token count comes from the published transcript snapshot (the
    // registry no longer carries a session_tokens mirror). It renders as a
    // compact count; hovering reveals the label and the exact unrounded value.
    if let Some(token_count) = token_count {
        metrics = metrics.push(
            tooltip(
                text(theme::format_compact_tokens(token_count))
                    .size(theme::TEXT_11)
                    .color(theme::TEXT_SECONDARY),
                render_metric_tooltip(format!("Session tokens: {token_count}")),
                tooltip::Position::Top,
            )
            .gap(theme::SPACE_4)
            .style(theme::tooltip_style),
        );
    }
    // Elapsed time since the agent started; hovering explains the meaning.
    metrics = metrics.push(
        tooltip(
            text(elapsed.to_owned())
                .size(theme::TEXT_11)
                .color(theme::TEXT_SECONDARY),
            render_metric_tooltip("Elapsed run time since the agent started".to_string()),
            tooltip::Position::Top,
        )
        .gap(theme::SPACE_4)
        .style(theme::tooltip_style),
    );
    metrics.into()
}

/// What a trace group's line row renders.
enum GroupLine<'a> {
    /// The line the group shows.
    Line(trace_line::Line<'a>),
    /// Nothing to show here: the group carries no text and no earlier line to
    /// hold, or a newer group already shows the very same line.
    Blank,
}

/// Render the visible trace groups for a card: the current (latest) group
/// always (last, chronologically, newest step expanded), plus up to
/// [`MAX_PREVIOUS_GROUPS`] previous groups (collapsed) when the card is
/// expanded. `lines` are the lines [`visible_lines`] derived for that same tail.
fn render_visible_groups<'a>(
    groups: &'a [TraceGroup<'a>],
    lines: Vec<GroupLine<'a>>,
) -> Vec<Element<'static, RunningMessage>> {
    let start = groups.len() - lines.len();
    let current = groups.len().saturating_sub(1);
    groups[start..]
        .iter()
        .zip(lines)
        .enumerate()
        .map(|(offset, (group, line))| render_trace_group(group, start + offset == current, line))
        .collect()
}

/// The line every visible group renders, oldest first. A group shows its own
/// line (its newest step's text); the current group, while it carries no text,
/// holds the line already on the card — the newest line of an earlier group — so
/// a running agent's card never goes blank mid-run. One phrase appears at most
/// once: a line a newer group already shows is dropped from the older group, the
/// card's line always being the newest copy. A group with nothing to show renders
/// no line row at all, and the caller renders the neutral placeholder once while
/// no visible group has a line (a fresh agent, a new user turn, or a turn whose
/// groups carry no text yet).
fn visible_lines<'a>(groups: &'a [TraceGroup<'a>], expanded: bool) -> Vec<GroupLine<'a>> {
    let depth = if expanded { MAX_PREVIOUS_GROUPS + 1 } else { 1 };
    let start = groups.len().saturating_sub(depth);
    let mut lines: Vec<GroupLine<'a>> = groups[start..]
        .iter()
        .map(|group| group_line(group).map_or(GroupLine::Blank, GroupLine::Line))
        .collect();
    if let Some(GroupLine::Blank) = lines.last() {
        // The line the card already shows: the newest one already computed for a
        // visible group, else the newest from the groups scrolled out of the
        // list (the collapsed card has none of the former).
        let held = lines[..lines.len() - 1]
            .iter()
            .rev()
            .find_map(|line| match line {
                GroupLine::Line(line) => Some((line.text.clone(), line.hover)),
                GroupLine::Blank => None,
            })
            .or_else(|| {
                groups[..start]
                    .iter()
                    .rev()
                    .find_map(group_line)
                    .map(|line| (line.text, line.hover))
            });
        if let Some((text, hover)) = held {
            *lines.last_mut().expect("non-empty tail") =
                GroupLine::Line(trace_line::Line { text, hover });
        }
    }
    let mut shown: HashSet<String> = HashSet::new();
    for line in lines.iter_mut().rev() {
        let GroupLine::Line(group_line) = line else {
            continue;
        };
        if !shown.insert(group_line.text.clone()) {
            *line = GroupLine::Blank;
        }
    }
    lines
}

/// The line a trace group shows: the newest step's own line (its narration, else
/// its reasoning — see [`trace_line::line_for_step`]), or an earlier step's line
/// while the newest one carries no usable text, so a group that already showed
/// a line keeps showing one. `None` when no step of the group has any text.
fn group_line<'a>(group: &TraceGroup<'a>) -> Option<trace_line::Line<'a>> {
    group
        .steps
        .iter()
        .rev()
        .find_map(|step| trace_line::line_for_step(step.narration, step.reasoning))
}

/// Render one trace group. `expand_current` renders the newest step's calls
/// expanded (current group); otherwise every call collapses to a name-only line
/// (previous groups). `line` is the text line the group shows (see
/// [`visible_lines`]); hovering it reveals the reasoning of the step it came
/// from (see [`reasoning_tooltip`]).
fn render_trace_group<'a>(
    group: &TraceGroup<'a>,
    expand_current: bool,
    line: GroupLine<'a>,
) -> Element<'static, RunningMessage> {
    let mut column = Column::new()
        .spacing(theme::SPACE_4)
        .align_x(Alignment::Start);
    match line {
        GroupLine::Line(line) => {
            column = column.push(reasoning_tooltip(
                line.hover,
                trace_line_text(&line.text).into(),
            ));
        }
        // Nothing to show here: the card-level placeholder covers the case where
        // no group has a line at all, and a second copy of another group's line
        // is never rendered.
        GroupLine::Blank => {}
    }

    if expand_current {
        let (newest, earlier) = group
            .steps
            .split_last()
            .expect("a trace group always carries at least one step");
        if !earlier.is_empty() {
            column = column.push(
                text(collapsed_calls_line(earlier))
                    .size(theme::TEXT_10)
                    .color(theme::TEXT_SECONDARY),
            );
        }
        for entry in newest.calls {
            column = column.push(tool_row(entry));
        }
    } else {
        column = column.push(
            text(collapsed_calls_line(&group.steps))
                .size(theme::TEXT_10)
                .color(theme::TEXT_SECONDARY),
        );
    }
    column.into()
}

/// A single expanded tool-call row in the current group. Settled calls render
/// exactly as [`tool_block`]; an in-flight call (result not yet committed)
/// gains the pulsing activity dot after the block.
fn tool_row(entry: &ToolCallEntry) -> Element<'static, RunningMessage> {
    let block = tool_block(&entry.tool, ToolBlockView::Compact);
    if entry.result.is_none() {
        Row::new()
            .spacing(theme::SPACE_6)
            .align_y(Alignment::Center)
            .push(block)
            .push(super::activity_dot())
            .into()
    } else {
        block
    }
}

/// Wrap a group's line in a hover tooltip revealing the reasoning of the step
/// the line came from (see [`TraceStep::reasoning`]), already cut to the prefix
/// that was read: the whole block reaches tens of kilobytes and is copied per
/// render. When that step has no reasoning the line renders bare.
fn reasoning_tooltip<'a>(
    reasoning: Option<&str>,
    line: Element<'a, RunningMessage>,
) -> Element<'a, RunningMessage> {
    let Some(reasoning) = reasoning else {
        return line;
    };
    let tooltip_content: Element<'static, RunningMessage> = container(
        text(reasoning.to_string())
            .size(theme::TEXT_11)
            .color(theme::TEXT_SECONDARY)
            .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
    )
    .max_width(MAX_TOOL_TOOLTIP_WIDTH)
    .into();
    tooltip(line, tooltip_content, tooltip::Position::Top)
        .gap(theme::SPACE_4)
        .style(theme::tooltip_style)
        .into()
}

/// Compose the collapsed name-only line for a set of trace steps: tool names
/// with underscores replaced by spaces, in first-appearance order, each unique
/// name suffixed with `xN` when it appears more than once within the collapsed
/// set (e.g. `read file x2, list files`). An in-flight call (result not yet
/// committed) gets its display name suffixed with a running marker (`…`) before
/// the xN counting, so it never looks settled.
fn collapsed_calls_line(steps: &[TraceStep<'_>]) -> String {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for step in steps {
        for call in step.calls {
            let base = call.tool.name.replace('_', " ");
            let display = if call.result.is_none() {
                format!("{base} …")
            } else {
                base
            };
            if !counts.contains_key(&display) {
                order.push(display.clone());
            }
            *counts.entry(display).or_default() += 1;
        }
    }
    order
        .iter()
        .map(|name| {
            let c = counts[name];
            if c > 1 {
                format!("{name} x{c}")
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Prune stale expanded-state keys: an agent that finished (or was replaced by
/// a new generation of the same agent_id) must not keep a stale expansion.
pub(crate) fn prune_expanded(expanded: &mut HashSet<(String, u64)>, agents: &[AgentHandle]) {
    let live: HashSet<(String, u64)> = agents
        .iter()
        .map(|h| (h.agent_id.clone(), h.generation))
        .collect();
    expanded.retain(|key| live.contains(key));
}

/// One step of an agent's work: one tool-call assistant round with the text
/// that round itself carried, borrowed from the ledger entry it was projected
/// from so a re-render copies neither reasoning nor tool payload.
struct TraceStep<'a> {
    /// The assistant's own intermediate text, the short visible narration it
    /// wrote for the reader (the decoded assistant `content`, NEVER the long
    /// `Reasoning` / `[thinking]` block). Empty when the round narrated
    /// nothing.
    narration: &'a str,
    /// The round's decoded long Reasoning. The step's line falls back to it
    /// when the narration carries no line, and its bounded prefix is the line's
    /// hover text otherwise.
    reasoning: Option<&'a str>,
    /// The round's tool calls (name/args pairs plus the tool_call_id-matched
    /// result), so the display can distinguish an in-flight call (result not
    /// yet committed) from a settled one.
    calls: &'a [ToolCallEntry],
}

/// One trace group in a running agent's live transcript: a maximal run of
/// consecutive tool-call assistant rounds sharing a single (possibly empty)
/// visible narration, bounded by a real user turn or a new narration. Every
/// round's own text is kept (see [`TraceStep`]) because the group's line
/// follows the newest step. A group always carries at least one step.
struct TraceGroup<'a> {
    /// The group's steps in chronological order; the LAST one is the newest
    /// (expanded in the current group, and the one its line is derived from).
    steps: Vec<TraceStep<'a>>,
}

/// A real user turn boundary in the ledger: a User message that is neither
/// an injected tool-result-image message (content starting with
/// [`crate::util::INJECTED_IMAGE_TAG`]) nor an empty resume signal (no/empty
/// content). Only such a message starts a new user turn.
fn is_user_turn_boundary(entry: &SessionEntry) -> bool {
    matches!(entry, SessionEntry::Message { role: ChatRole::User, content, .. }
    if content.as_deref().is_some_and(|c| {
        !c.trim().is_empty() && !c.starts_with(crate::util::INJECTED_IMAGE_TAG)
    }))
}

/// The slice of the ledger belonging to the CURRENT user turn: everything
/// after the last real user turn boundary ([`is_user_turn_boundary`]). The
/// running card shows only the current turn, so groups from previous turns
/// never leak onto it. When no boundary exists the whole ledger is current.
fn current_turn_entries(entries: &[SessionEntry]) -> &[SessionEntry] {
    let start = entries
        .iter()
        .rposition(is_user_turn_boundary)
        .map_or(0, |i| i + 1);
    &entries[start..]
}

/// Project the shared session ledger into a running agent's trace groups, in
/// chronological order.
///
/// Boundary rules:
/// - A new (non-blank) narration in a tool-call assistant round starts a new
///   group; that round becomes the group's first step.
/// - Consecutive tool-call rounds with no narration accumulate into the current
///   group as further steps, each keeping its own narration and reasoning.
/// - A real user turn closes the current group; the next tool-call round then
///   starts a fresh one even with no narration, and the card has no line to
///   hold across that turn. An empty (whitespace) User message or a synthetic
///   tool-injected one (content starting with the
///   [`crate::util::INJECTED_IMAGE_TAG`] marker) is NOT a boundary.
fn derive_trace_groups(entries: &[SessionEntry]) -> Vec<TraceGroup<'_>> {
    let mut groups: Vec<TraceGroup> = Vec::new();
    let mut current: Option<TraceGroup> = None;
    for entry in entries {
        match entry {
            SessionEntry::ToolRound {
                narration,
                reasoning,
                calls,
            } => {
                let narration = narration.as_deref().unwrap_or_default();
                if !narration.trim().is_empty()
                    && let Some(group) = current.take()
                {
                    groups.push(group);
                }
                let group = current.get_or_insert_with(|| TraceGroup { steps: Vec::new() });
                group.steps.push(TraceStep {
                    narration,
                    reasoning: reasoning.as_deref(),
                    calls,
                });
            }
            SessionEntry::Message { .. } => {
                if is_user_turn_boundary(entry)
                    && let Some(group) = current.take()
                {
                    groups.push(group);
                }
            }
        }
    }
    if let Some(group) = current.take() {
        groups.push(group);
    }
    groups
}

/// Render a one-line metric tooltip body in a narrow, styled container: the
/// explanatory text (metric label plus, where useful, the precise value). Kept
/// narrower than [`MAX_TOOL_TOOLTIP_WIDTH`] because a metric tooltip holds a
/// single short line, not multi-arg tool calls.
fn render_metric_tooltip(label: String) -> Element<'static, RunningMessage> {
    container(
        text(label)
            .size(theme::TEXT_11)
            .color(theme::TEXT_PRIMARY)
            .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
    )
    .max_width(MAX_METRIC_TOOLTIP_WIDTH)
    .into()
}

/// Render a compact non-agent LLM call row: zap marker + purpose + elapsed.
/// No workspace name (the workspace grouping already names it); the elapsed
/// time uses the same brighter tone as agent cards. The purpose is the static
/// human-readable label — raw kind names never leak.
fn render_call_row(call: &CallRow) -> Element<'static, RunningMessage> {
    let h = &call.handle;
    let elapsed = format_elapsed(h.started_at);
    let purpose = crate::agent::registry::call_kind_label(h.kind);

    row![
        lucide::zap::<iced::Theme, iced::Renderer>()
            .size(theme::TEXT_16)
            .color(theme::ACCENT),
        text(purpose)
            .size(theme::TEXT_12)
            .color(theme::TEXT_SECONDARY),
        Space::new().width(Length::Fill),
        text(elapsed)
            .size(theme::TEXT_11)
            .color(theme::TEXT_SECONDARY),
    ]
    .spacing(theme::SPACE_6)
    .align_y(Alignment::Center)
    .into()
}

/// Format the elapsed time since `started_at` at render time.
fn format_elapsed(started_at: DateTime<Utc>) -> String {
    let now = Utc::now();
    let secs = now.signed_duration_since(started_at).num_seconds().max(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatMessage;
    use crate::agent::registry::{AgentHandle, RunningTool};
    use crate::gui::session_view::build_ledger;

    /// A parent key from its spec: `""` for none, otherwise `t:<id>`, `a:<key>`,
    /// `r:<key>` or `l:<id>`.
    fn parent(spec: &str) -> Option<ParentKey> {
        let (kind, key) = spec.split_once(':')?;
        let key = key.to_string();
        Some(match kind {
            "t" => ParentKey::Ticket(key),
            "a" => ParentKey::AnalyzeRound(key),
            "r" => ParentKey::Research(key),
            _ => ParentKey::Launch(key),
        })
    }

    fn agent(id: &str, role: &str, workspace: &str, parent_spec: &str) -> AgentHandle {
        AgentHandle {
            agent_id: id.to_string(),
            role: role.to_string(),
            ticket_id: None,
            workspace_path: format!("/ws/{workspace}"),
            workspace_name: workspace.to_string(),
            parent_key: parent(parent_spec),
            parent_label: None,
            started_at: Utc::now(),
            generation: 0,
            activity: None,
            paused: false,
        }
    }

    fn call(
        kind: &'static str,
        workspace: &str,
        parent_spec: &str,
        run_lifetime: bool,
    ) -> NonAgentCallHandle {
        NonAgentCallHandle {
            kind,
            workspace: workspace.to_string(),
            started_at: Utc::now(),
            parent_key: parent(parent_spec),
            parent_label: None,
            run_lifetime,
        }
    }

    /// An agent row: id, role, workspace, parent spec.
    type AgentRow = (&'static str, &'static str, &'static str, &'static str);
    /// A call row: kind, workspace, parent spec, run lifetime.
    type CallRow = (&'static str, &'static str, &'static str, bool);
    /// An expected group: kind, key, member count, workspace when claimed, run lifetime.
    type GroupRow = (GroupKind, &'static str, usize, Option<&'static str>, bool);

    struct GroupCase {
        name: &'static str,
        agents: &'static [AgentRow],
        calls: &'static [CallRow],
        total: usize,
        groups: &'static [GroupRow],
    }

    fn case(
        name: &'static str,
        agents: &'static [AgentRow],
        calls: &'static [CallRow],
        total: usize,
        groups: &'static [GroupRow],
    ) -> GroupCase {
        GroupCase {
            name,
            agents,
            calls,
            total,
            groups,
        }
    }

    #[test]
    fn build_groups_keeps_runs_and_rounds_apart() {
        let cases = [
            case(
                "ticket agents and synthesis share the ticket group",
                &[
                    ("a1", "analyst", "ws1", "t:T1"),
                    ("mgr", "manager", "ws1", ""),
                ],
                &[("synthesis", "ws1", "t:T1", false)],
                2,
                &[(GroupKind::Ticket, "T1", 2, None, false)],
            ),
            case(
                "two analyze rounds never mix members",
                &[
                    ("a1", "analyst", "ws1", "a:roundA"),
                    ("a2", "analyst", "ws1", "a:roundA"),
                    ("b1", "analyst", "ws1", "a:roundB"),
                ],
                &[("consolidate", "ws1", "a:roundA", false)],
                2,
                &[
                    (GroupKind::AnalyzeRound, "roundA", 3, None, false),
                    (GroupKind::AnalyzeRound, "roundB", 1, None, false),
                ],
            ),
            case(
                "research run members share one key across phases",
                &[
                    ("x1", "analyst", "ws1", "r:run1"),
                    ("x2", "analyst", "ws1", "r:run1"),
                    ("y1", "analyst", "ws1", "r:run2"),
                ],
                &[
                    ("synthesize", "ws1", "r:run1", false),
                    ("research_orchestrator", "ws1", "r:run1", true),
                ],
                2,
                &[
                    (GroupKind::Research, "run1", 3, None, true),
                    (GroupKind::Research, "run2", 1, None, false),
                ],
            ),
            case(
                "a run resolves to the real workspace across ephemeral workers",
                &[
                    ("w1", "worker", "run1", "r:run1"),
                    ("r1", "analyst", "ws1", "r:run1"),
                    ("r2", "analyst", "ws1", "r:run2"),
                    ("w2", "worker", "run2", "r:run2"),
                ],
                &[],
                2,
                &[
                    (GroupKind::Research, "run1", 2, Some("ws1"), false),
                    (GroupKind::Research, "run2", 2, Some("ws1"), false),
                ],
            ),
            case(
                "cross-workspace round keys never merge groups",
                &[
                    ("a1", "analyst", "ws1", "a:abc123"),
                    ("b1", "analyst", "ws2", "a:abc123"),
                ],
                &[],
                2,
                &[
                    (GroupKind::AnalyzeRound, "abc123", 1, Some("ws1"), false),
                    (GroupKind::AnalyzeRound, "abc123", 1, Some("ws2"), false),
                ],
            ),
        ];

        for case in cases {
            let agents: Vec<_> = case
                .agents
                .iter()
                .map(|&(id, role, ws, p)| agent(id, role, ws, p))
                .collect();
            let calls: Vec<_> = case
                .calls
                .iter()
                .map(|&(k, ws, p, run)| call(k, ws, p, run))
                .collect();
            let groups = build_groups(&agents, &calls);
            assert_case(&case, &groups);
        }
    }

    fn assert_case(case: &GroupCase, groups: &[DisplayGroup]) {
        let name = case.name;
        assert_eq!(groups.len(), case.total, "case {name}: group count");
        for row in case.groups {
            let (kind, key, items) = (&row.0, row.1, row.2);
            let (workspace, run_lifetime) = (row.3, row.4);
            let found = groups.iter().find(|group| {
                &group.kind == kind
                    && group.key == key
                    && workspace.is_none_or(|ws| group.workspace == ws)
            });
            let group = found.unwrap_or_else(|| panic!("case {name}: no {kind:?} group {key:?}"));
            assert_eq!(group.items.len(), items, "case {name}: {key} members");
            if let Some(ws) = workspace {
                assert_eq!(group.workspace, ws, "case {name}: {key} workspace");
            }
            assert_eq!(
                group.run_lifetime, run_lifetime,
                "case {name}: {key} run lifetime"
            );
        }
    }

    #[test]
    fn singleton_and_unattributed_groups_stay_generic_per_workspace() {
        // (case, agents, calls, singleton groups)
        let cases: &[(&str, &[AgentRow], &[CallRow], usize)] = &[
            (
                "one singleton group per workspace",
                &[
                    ("mgr_ws1", "manager", "ws1", ""),
                    ("mnt_ws2", "maintainer", "ws2", ""),
                ],
                &[("some_orchestrator_call", "ws1", "", false)],
                2,
            ),
            (
                "group headers never repeat the workspace",
                &[("mgr", "manager", "ws1", "")],
                &[("some_orchestrator_call", "ws1", "", false)],
                1,
            ),
        ];

        for &(name, agent_rows, call_rows, singletons) in cases {
            let agents: Vec<_> = agent_rows
                .iter()
                .map(|&(id, role, ws, p)| agent(id, role, ws, p))
                .collect();
            let calls: Vec<_> = call_rows
                .iter()
                .map(|&(k, ws, p, run)| call(k, ws, p, run))
                .collect();
            let groups = build_groups(&agents, &calls);
            let count = |kind: &GroupKind| groups.iter().filter(|g| &g.kind == kind).count();
            assert_eq!(
                count(&GroupKind::Singleton),
                singletons,
                "case {name}: singletons"
            );
            assert_eq!(
                count(&GroupKind::Unattributed),
                1,
                "case {name}: unattributed"
            );
            let unattributed = groups.iter().find(|g| g.kind == GroupKind::Unattributed);
            assert_eq!(
                unattributed.unwrap().workspace,
                "ws1",
                "case {name}: workspace"
            );
            // Inside a workspace section the group headers must not repeat the
            // workspace name — the section's order already names it.
            for group in &groups {
                let (title, _) = group_title(group);
                assert!(
                    !title.contains("ws1"),
                    "case {name}: no workspace in {title}"
                );
            }
        }
    }

    #[test]
    fn sort_key_orders_groups_by_kind() {
        let group = |kind: GroupKind| DisplayGroup {
            kind,
            key: "ws".to_string(),
            workspace: "ws".to_string(),
            label: None,
            items: Vec::new(),
            run_lifetime: false,
        };
        let mut groups = [
            group(GroupKind::Unattributed),
            group(GroupKind::Ticket),
            group(GroupKind::Research),
        ];
        groups.sort_by_key(DisplayGroup::sort_key);
        let kinds: Vec<_> = groups.iter().map(|group| group.kind.clone()).collect();
        let expected = [
            GroupKind::Ticket,
            GroupKind::Research,
            GroupKind::Unattributed,
        ];
        assert_eq!(kinds, expected);
    }

    #[test]
    fn ticket_group_label_comes_from_members_not_the_key() {
        // The title is captured observationally at spawn (agents via their
        // ticket, the synthesis call via the joint verdict), so a group carries
        // it even when only the synthesis call remains.
        let mut analyst = agent("a1", "analyst", "ws1", "t:T1");
        analyst.parent_label = Some("Fix the login flow".to_string());
        let mut synthesis = call("synthesis", "ws1", "t:T1", false);
        synthesis.parent_label = Some("Fix the login flow".to_string());
        let groups = build_groups(&[analyst], &[synthesis]);
        let ticket = groups.iter().find(|g| g.kind == GroupKind::Ticket);
        let ticket = ticket.expect("ticket group exists");
        assert_eq!(
            ticket.label.as_deref(),
            Some("Fix the login flow"),
            "label from a member"
        );
        // group_title preserves (title, id); render_group composes `[id] title`.
        let (title, secondary) = group_title(ticket);
        assert_eq!(title, "Fix the login flow");
        assert_eq!(secondary.as_deref(), Some("T1"));
    }

    #[test]
    fn analyze_and_research_groups_render_question_not_raw_key() {
        // Analyze/research headers show the question text; the raw NanoID key is
        // never rendered, and a missing label degrades to a generic one.
        let mut analyst = agent("a1", "analyst", "ws1", "a:job_abc");
        analyst.parent_label = Some("Why is CI flaky?".to_string());
        let groups = build_groups(&[analyst], &[]);
        let analyze = groups.iter().find(|g| g.kind == GroupKind::AnalyzeRound);
        let analyze = analyze.expect("analyze group exists");
        let (title, secondary) = group_title(analyze);
        assert_eq!(title, "Why is CI flaky?");
        assert!(secondary.is_none(), "no id secondary for analyze groups");
        assert!(!title.contains("job_abc"), "raw key never leaks");

        let mut bare = analyze.clone();
        bare.label = None;
        let (fallback, _) = group_title(&bare);
        assert_eq!(fallback, "Analyze round");
        assert!(!fallback.contains("job_abc"), "generic fallback, no key");
    }

    #[test]
    fn sections_group_by_workspace_and_keep_kind_order() {
        // Workspace-first layout: sections sort alphabetically, groups keep the
        // canonical kind order inside one, and a singleton lands in its own
        // workspace's section.
        let mut workspaces = std::collections::HashMap::new();
        workspaces.insert(
            "ws2".to_string(),
            Workspace {
                path: "/ws/ws2".to_string(),
                ..Default::default()
            },
        );
        let agents = vec![
            agent("t_agent", "engineer", "ws1", "t:T1"),
            agent("mgr_ws2", "manager", "ws2", ""),
            agent("mgr_ws1", "manager", "ws1", ""),
        ];
        let sections = build_sections(build_groups(&agents, &[]), &workspaces);
        assert_eq!(sections.len(), 2, "one section per workspace");
        assert_eq!(sections[0].workspace, "ws1");
        assert_eq!(sections[1].workspace, "ws2");
        assert_eq!(sections[0].groups[0].kind, GroupKind::Ticket);
        assert_eq!(sections[0].groups[1].kind, GroupKind::Singleton);
        // ws2 is managed → bare name; ws1 is outside the managed set → "(external)".
        assert_eq!(sections[1].label, "ws2");
        assert_eq!(sections[0].label, "ws1 (external)");
    }

    // ── Trace-group derivation ──────────────────────────────────────

    fn assistant_tool_call(narration: &str, calls: &[(&str, serde_json::Value)]) -> ChatMessage {
        assistant_tool_call_reasoning(narration, calls, None)
    }

    /// A tool-call round carrying `reasoning`, so the ledger's
    /// `ToolRound.reasoning` is populated.
    fn assistant_tool_call_reasoning(
        narration: &str,
        calls: &[(&str, serde_json::Value)],
        reasoning: Option<&str>,
    ) -> ChatMessage {
        let calls_json: Vec<serde_json::Value> = calls
            .iter()
            .map(|(name, args)| {
                serde_json::json!({
                    "id": "call_1",
                    "name": name,
                    "arguments": serde_json::to_string(args).unwrap_or_default(),
                })
            })
            .collect();
        let content = if narration.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(narration.to_string())
        };
        let mut body = serde_json::json!({ "content": content, "tool_calls": calls_json });
        if let Some(reasoning) = reasoning {
            body["reasoning"] = serde_json::json!(reasoning);
        }
        ChatMessage::assistant(body.to_string())
    }

    /// One tool-call round with a single `read`, as the agent's turns look.
    fn round(narration: &str) -> ChatMessage {
        assistant_tool_call(narration, &[("read", serde_json::json!({"path": "a.rs"}))])
    }

    /// A tool-call round whose step carries `reasoning` (the line's hover text).
    fn thinking(narration: &str, reasoning: &str) -> ChatMessage {
        assistant_tool_call_reasoning(
            narration,
            &[("read", serde_json::json!({"path": "a.rs"}))],
            Some(reasoning),
        )
    }

    fn user_turn(text: &str) -> ChatMessage {
        ChatMessage::user(format!(
            "{text}\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>"
        ))
    }

    fn injected_image() -> ChatMessage {
        let marker = "data:image/png;base64,xxx".to_string();
        ChatMessage::user(crate::util::injected_image_user_message(&[marker]))
    }

    fn assert_trace_groups(name: &str, groups: &[TraceGroup<'_>], expected: &[&[&str]]) {
        assert_eq!(groups.len(), expected.len(), "case {name}: group count");
        for (group, steps) in groups.iter().zip(expected) {
            assert_eq!(
                group.steps.len(),
                steps.len(),
                "case {name}: steps in group"
            );
            for (step, wanted) in group.steps.iter().zip(*steps) {
                assert_eq!(step.narration, *wanted, "case {name}: step narration");
            }
        }
    }

    type TurnCase<'a> = (&'a str, Vec<ChatMessage>, Vec<Vec<&'a str>>);

    #[test]
    fn narration_and_turn_boundaries_derive_groups() {
        // (case, history, the narration of every step of every group it derives)
        let cases: &[TurnCase] = &[
            ("empty history produces no groups", Vec::new(), Vec::new()),
            (
                "a content-only assistant message is not a group",
                vec![
                    user_turn("hello"),
                    ChatMessage::assistant("Final answer here."),
                ],
                Vec::new(),
            ),
            (
                "a new narration starts a new group",
                vec![round("Reading the file"), round("Editing it")],
                vec![vec!["Reading the file"], vec!["Editing it"]],
            ),
            (
                "consecutive no-narration rounds accumulate",
                vec![round(""), round("")],
                vec![vec!["", ""]],
            ),
            (
                "a real user turn resets the group",
                vec![round("Narration"), user_turn("ok now fix it"), round("")],
                vec![vec!["Narration"], vec![""]],
            ),
            (
                "an injected image user message does not reset the group",
                vec![round(""), injected_image(), round("")],
                vec![vec!["", ""]],
            ),
            (
                "an empty user message is not a turn boundary",
                vec![round(""), ChatMessage::user(""), round("")],
                vec![vec!["", ""]],
            ),
        ];

        for (name, history, expected) in cases {
            let ledger = build_ledger(history);
            let groups = derive_trace_groups(&ledger);
            let expected: Vec<&[&str]> = expected.iter().map(Vec::as_slice).collect();
            assert_trace_groups(name, &groups, &expected);
        }
    }

    #[test]
    fn every_step_keeps_its_own_reasoning() {
        // Each round keeps its own reasoning, so the label can follow the newest
        // step instead of latching onto the group's first one.
        let history = vec![
            assistant_tool_call_reasoning(
                "",
                &[("read", serde_json::json!({"path": "a.rs"}))],
                None,
            ),
            thinking("", "first"),
            thinking("", "second"),
        ];
        let ledger = build_ledger(&history);
        let groups = derive_trace_groups(&ledger);
        assert_eq!(groups.len(), 1, "no narration → same group");
        let reasoning: Vec<_> = groups[0].steps.iter().map(|step| step.reasoning).collect();
        assert_eq!(reasoning, vec![None, Some("first"), Some("second")]);
    }

    /// A ledger tool-call entry for the collapsed/in-flight projection tests;
    /// `result: None` marks the call in-flight.
    fn entry(name: &str, args: serde_json::Value, result: Option<&str>) -> ToolCallEntry {
        ToolCallEntry {
            tool_call_id: String::new(),
            tool: RunningTool::from_tool_call(&crate::ToolCall {
                id: String::new(),
                name: name.to_string(),
                arguments: args,
            }),
            result: result.map(ToString::to_string),
        }
    }

    fn step(calls: &[ToolCallEntry]) -> TraceStep<'_> {
        TraceStep {
            narration: "",
            reasoning: None,
            calls,
        }
    }

    #[test]
    fn collapsed_calls_line_counts_and_orders() {
        let read_calls = [
            entry("read_file", serde_json::json!({"path": "a.rs"}), Some("ok")),
            entry("list_files", serde_json::json!({"path": "."}), Some("ok")),
        ];
        let edit_calls = [
            entry("read_file", serde_json::json!({"path": "b.rs"}), Some("ok")),
            entry("read_file", serde_json::json!({"path": "c.rs"}), None),
        ];
        let steps = vec![step(&read_calls), step(&edit_calls)];
        // Underscores become spaces; settled names stay exact, an in-flight call
        // gets a running suffix; `xN` counts collapsed repetitions; first-seen order.
        assert_eq!(
            collapsed_calls_line(&steps),
            "read file x2, list files, read file …"
        );
    }

    #[test]
    fn the_card_shows_only_the_current_turn() {
        // (case, history, the narration of every step of the current turn's groups)
        let cases: &[TurnCase] = &[
            (
                "the pre-turn group is gone",
                vec![round("First"), user_turn("ok now do it"), round("Second")],
                vec![vec!["Second"]],
            ),
            (
                "no user turn shows the whole history",
                vec![round("A"), round("B")],
                vec![vec!["A"], vec!["B"]],
            ),
            (
                "a turn without tool rounds is empty",
                vec![round("A"), user_turn("stop now")],
                Vec::new(),
            ),
            (
                "injected and empty user messages do not start a turn",
                vec![
                    round(""),
                    injected_image(),
                    ChatMessage::user(""),
                    round(""),
                ],
                vec![vec!["", ""]],
            ),
        ];

        for (name, history, expected) in cases {
            let ledger = build_ledger(history);
            let groups = derive_trace_groups(current_turn_entries(&ledger));
            let expected: Vec<&[&str]> = expected.iter().map(Vec::as_slice).collect();
            assert_trace_groups(name, &groups, &expected);
        }
    }

    #[test]
    fn rounds_carry_results_for_in_flight_distinction() {
        // One round with two calls, only one of which got a result: after the
        // ledger one call is settled and the other in-flight, both preserved.
        let calls_json = serde_json::json!([
            {"id": "call_a", "name": "read", "arguments": "{\"path\": \"a.rs\"}"},
            {"id": "call_b", "name": "edit", "arguments": "{\"path\": \"b.rs\"}"},
        ]);
        let body = serde_json::json!({ "content": null, "tool_calls": calls_json });
        let history = vec![
            ChatMessage::assistant(body.to_string()),
            ChatMessage::tool_result("call_a", "contents of a.rs"),
        ];
        let ledger = build_ledger(&history);
        let groups = derive_trace_groups(&ledger);
        assert_eq!(groups.len(), 1);
        let round = &groups[0].steps[0].calls;
        assert_eq!(round.len(), 2, "both calls preserved in the round");
        let settled = round
            .iter()
            .find(|c| c.tool_call_id == "call_a")
            .expect("call_a");
        assert_eq!(
            settled.result.as_deref(),
            Some("contents of a.rs"),
            "matched tool result attaches to call_a"
        );
        let in_flight = round
            .iter()
            .find(|c| c.tool_call_id == "call_b")
            .expect("call_b");
        assert_eq!(
            in_flight.result, None,
            "call_b has no result yet → in-flight"
        );
    }

    /// What each visible line row shows: the text itself, or `<blank>` where the
    /// group renders no line (no text to show, or a newer group shows it).
    fn shown_lines(groups: &[TraceGroup<'_>], expanded: bool) -> Vec<String> {
        visible_lines(groups, expanded)
            .into_iter()
            .map(|line| match line {
                GroupLine::Line(line) => line.text,
                GroupLine::Blank => "<blank>".to_string(),
            })
            .collect()
    }

    #[test]
    fn the_line_follows_the_newest_step_and_holds_the_previous_one() {
        const DRIFT: &str = "The parser struct is missing the offset field, so the columns drift.";
        const CHECKED: &str = "Checking the whole file first.";
        // (case, history, the line, its hover text when the case claims it)
        let cases: &[(&str, Vec<ChatMessage>, &str, Option<&str>)] = &[
            (
                "the line follows the newest step",
                vec![
                    thinking("Now editing the parser", "Reading the whole file first."),
                    thinking("", DRIFT),
                ],
                DRIFT,
                Some(DRIFT),
            ),
            (
                "a silent step keeps the previous line",
                vec![thinking("", CHECKED), round("")],
                CHECKED,
                None,
            ),
            (
                "a whitespace-only narration keeps it too",
                vec![thinking("", CHECKED), round(" ")],
                CHECKED,
                None,
            ),
        ];

        for (name, history, text, hover) in cases {
            let ledger = build_ledger(history);
            let groups = derive_trace_groups(&ledger);
            assert_eq!(groups.len(), 1, "case {name}: one group");
            let line = group_line(&groups[0])
                .unwrap_or_else(|| panic!("case {name}: the group carries text"));
            assert_eq!(line.text, *text, "case {name}: line text");
            if let Some(hover) = hover {
                assert_eq!(line.hover, Some(*hover), "case {name}: hover");
            }
        }
    }

    type TextlessCase<'a> = (
        &'a str,
        Vec<ChatMessage>,
        usize,
        bool,
        Option<&'a [&'a str]>,
        Option<&'a [&'a str]>,
    );

    #[test]
    fn textless_groups_hold_or_drop_the_line() {
        const CHECKED: &str = "Checking the whole file first.";
        // (case, history, groups, newest group is textless, collapsed card, expanded card)
        let cases: &[TextlessCase] = &[
            (
                "a textless group holds the previous group's line",
                vec![thinking("", CHECKED), round("---")],
                2,
                true,
                Some(&[CHECKED]),
                None,
            ),
            (
                "a turn's first group falls back to the placeholder",
                vec![round("---")],
                1,
                true,
                Some(&["<blank>"]),
                None,
            ),
            (
                "a held line is not shown twice in the previous group's list",
                vec![thinking("", CHECKED), round("---")],
                2,
                false,
                Some(&[CHECKED]),
                Some(&["<blank>", CHECKED]),
            ),
            (
                "a repeated line is dropped even when older groups sit between",
                vec![
                    thinking("", "The same line here."),
                    user_turn("go on"),
                    round("Something else."),
                    user_turn("now the rest"),
                    thinking("", "The same line here."),
                ],
                3,
                false,
                None,
                Some(&["<blank>", "Something else.", "The same line here."]),
            ),
            (
                "a textless previous group renders no line rather than a placeholder",
                vec![round("---"), round("Now editing.")],
                2,
                false,
                None,
                Some(&["<blank>", "Now editing."]),
            ),
        ];

        for (name, history, count, textless, collapsed, expanded) in cases {
            let ledger = build_ledger(history);
            let groups = derive_trace_groups(&ledger);
            assert_eq!(groups.len(), *count, "case {name}: group count");
            if *textless {
                assert!(
                    group_line(&groups[groups.len() - 1]).is_none(),
                    "case {name}: the new group has no text"
                );
            }
            if let Some(collapsed) = collapsed {
                assert_eq!(
                    shown_lines(&groups, false),
                    *collapsed,
                    "case {name}: collapsed"
                );
            }
            if let Some(expanded) = expanded {
                assert_eq!(
                    shown_lines(&groups, true),
                    *expanded,
                    "case {name}: expanded"
                );
            }
        }
    }

    #[test]
    fn prune_expanded_drops_stale_keys() {
        let mut expanded: HashSet<(String, u64)> =
            HashSet::from([("a".to_string(), 0), ("b".to_string(), 2)]);
        let agents = vec![agent("a", "analyst", "ws1", "")];
        prune_expanded(&mut expanded, &agents);
        assert_eq!(expanded.len(), 1);
        assert!(
            expanded.contains(&("a".to_string(), 0)),
            "live agent keeps its expanded key"
        );
    }
}
