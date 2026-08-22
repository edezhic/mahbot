//! "Running Agents" dashboard page — a live view of every currently-running
//! agent and in-flight non-agent LLM work, grouped by WORKSPACE, then by
//! DIRECT PARENT INVOCATION (ticket / analyze round / research run / workspace
//! singleton / unattributable orchestrator calls).
//!
//! The view is a running-only window: it reads the in-memory registries
//! ([`crate::registry::AGENT_REGISTRY`] and
//! [`crate::registry::NON_AGENT_CALLS`]) at render time — no database
//! reads, no schema changes, no new subscriptions, no history retained
//! between ticks. The existing 1-second UI tick re-renders the page, so the
//! view refreshes at that cadence for free.
//!
//! Truthfulness rules:
//! - An agent not executing a tool (in an LLM call, between rounds) shows its
//!   last COMPLETED tool instead of nothing — a fast tool no longer flashes
//!   and vanishes (deliberate override of the strict absence-is-honest rule
//!   for tools; the last-tool block is a historical marker, always labeled by
//!   its tool name, so it cannot be mistaken for live activity).
//! - Parallel tool execution is represented honestly: every tool that
//!   actually started executing appears as its own tool block; tools that never
//!   execute (unknown tool, pre-flight cancellation) never show.
//! - The instrumentation is purely observational — it never affects
//!   shutdown/drain logic and gains no cancellation semantics.
//! - The live view shows tool arguments EXACTLY as the agent passed them —
//!   including any secrets. This is a deliberate, documented divergence from
//!   the durable logs (tool-call stats, failure feedback), which remain
//!   credential-scrubbed: the live view is transient and exists for full
//!   visibility of what the agent is doing.
//!
//! The page is DB-free by design: everything it shows is derived from the
//! live in-memory registries, plus the dashboard's registered-workspace map
//! (only for the "(external)" marker on unregistered/ephemeral workspace
//! sections). All header labels (ticket titles, analyze/research questions)
//! are captured observationally at spawn — never read from the DB.

use crate::gui::theme;
use crate::gui::widgets;
use crate::registry::{AgentHandle, NonAgentCallHandle, ParentKey};
use chrono::{DateTime, Utc};

use iced::widget::{
    Column, Row, Space, button, column, container, row, scrollable, stack, text, tooltip,
};
use iced::{Alignment, Element, Length};

use iced_fonts::lucide;

use super::widget_helpers;
use super::{Message, WorkspaceInfo};

/// Maximum display length (Unicode chars) of an analyze/research group header
/// label (the question/task text). Truncated at a char boundary with "…".
const MAX_GROUP_LABEL_CHARS: usize = 80;

/// Maximum display length (Unicode chars) of a single argument VALUE in the
/// row's comma-separated key-value pairs line. The hover tooltip always shows
/// the full untruncated value.
const MAX_TOOL_VALUE_CHARS: usize = 20;

/// Maximum display length (Unicode chars) of the whole key-value pairs line
/// in the row (values already truncated per [`MAX_TOOL_VALUE_CHARS`]). Cut at
/// a pair boundary with a trailing "…" so the line can never overflow into
/// the right-aligned metrics (iced rows do not wrap).
const MAX_TOOL_PAIRS_LINE_CHARS: usize = 80;

/// Maximum width (px) of the hover tooltip content; long values wrap within it.
const MAX_TOOL_TOOLTIP_WIDTH: f32 = 560.0;

/// Messages emitted by the Running Agents page.
///
/// The page itself stays stateless — the pending research-cancel
/// confirmation lives on the [`Dashboard`](super::Dashboard) so the dialog
/// requires no per-run page state beyond the pending confirmation itself.
#[derive(Debug, Clone)]
#[expect(clippy::enum_variant_names)] // the Cancel* names are deliberate — one cancel flow, four stages
pub(crate) enum RunningMessage {
    /// Cancel button pressed on a research-run group header. Carries the
    /// run's durable job id — NEVER rendered as text; the button action
    /// alone carries it.
    CancelRequest(String),
    /// The confirmation dialog's danger button — start the async cancel.
    CancelConfirmed,
    /// Keep/Dismiss (or Escape/backdrop) — close the dialog with no effect.
    CancelDismissed,
    /// The async cancel finished: `Ok(())` = run stopped and removed
    /// permanently; `Err` = the durable sweep failed (surfaced as a toast).
    CancelFinished(Result<(), String>),
}

/// Render the live running-agents page.
///
/// The page is observational — it reads the live in-memory registries and
/// renders directly into the dashboard's [`Message`] type; its only
/// self-emitted messages are the research-run manual-cancel flow
/// ([`RunningMessage`]). `workspaces` is the dashboard's
/// registered-workspace map (name → info) — used only to mark
/// unregistered/ephemeral workspace sections with the "(external)" suffix.
/// Everything else comes from the live in-memory registries.
pub(crate) fn view(
    workspaces: &std::collections::HashMap<String, WorkspaceInfo>,
    pending_cancel: Option<&str>,
) -> Element<'static, Message> {
    let agents = crate::registry::AGENT_REGISTRY.list();
    let calls = crate::registry::NON_AGENT_CALLS.list();

    // Workspace-first sections: each section holds its groups, groups
    // sorted by kind (tickets → analyze rounds → research runs →
    // singletons → unattributed), sections alphabetically by name.
    let sections = build_sections(build_groups(&agents, &calls), workspaces);

    let body: Element<'_, RunningMessage> = if sections.is_empty() {
        // Shared empty-state pattern: large radar glyph (the page's nav
        // icon) + label, centered.
        widgets::empty_state_placeholder(
            lucide::radar::<iced::Theme, iced::Renderer>(),
            "Nothing is currently running.",
        )
    } else {
        let mut content = Column::new().spacing(20);
        for section in &sections {
            content = content.push(render_section(section));
        }
        scrollable(container(content).width(Length::Fill).padding([0, 4]))
            .height(Length::Fill)
            .direction(theme::vertical_scrollbar())
            .style(theme::scrollbar_style)
            .into()
    };

    // Uniform page chrome with the rest of the dashboard: base Flexoki
    // fill + 24px padding, matching the other pages.
    let page: Element<'_, RunningMessage> = container(body)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(24)
        .style(theme::base_container_style)
        .into();
    // Confirm-dialog overlay: the page content is stack child 0, the dialog
    // (or a type-stable placeholder) child 1 — the widget shapes never
    // change, so no page state is lost when the dialog opens/closes.
    let confirm_layer: Element<'_, RunningMessage> = if let Some(run_key) = pending_cancel {
        cancel_confirm_dialog(run_key)
    } else {
        stack([widget_helpers::empty_stack_placeholder()]).into()
    };
    let stacked: Element<'_, RunningMessage> = stack([page, confirm_layer]).into();
    stacked.map(Message::RunningAgents)
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
    Singleton,
    Unattributed,
}

#[derive(Debug, Clone)]
struct DisplayGroup {
    kind: GroupKind,
    /// Group key (ticket id / round key / run key / workspace name).
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

/// One workspace section on the page: a section header (the workspace name)
/// plus the groups running in that workspace.
#[derive(Debug, Clone)]
struct WorkspaceSection {
    /// Raw workspace name — the grouping key.
    workspace: String,
    /// Resolved display label: the bare name for registered workspaces, the
    /// "(external)"-suffixed form for unregistered/ephemeral ones, and
    /// "workspace" for an empty name.
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
            GroupKind::Singleton => 3,
            GroupKind::Unattributed => 4,
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
    }
}

/// The group key payload for a parent key (ticket id / round key / run key).
fn parent_group_key(parent: &ParentKey) -> &str {
    match parent {
        ParentKey::Ticket(id) => id,
        ParentKey::AnalyzeRound(key) | ParentKey::Research(key) => key,
    }
}

/// Group the running agents and non-agent calls by DIRECT PARENT INVOCATION.
///
/// - Tickets: agents with a [`ParentKey::Ticket`] plus calls with the same
///   ticket parent.
/// - Analyze rounds: agents + calls sharing an [`ParentKey::AnalyzeRound`] key.
/// - Research runs: agents + calls sharing a [`ParentKey::Research`] key.
/// - Workspace singletons: agents with no parent key (manager / maintainer /
///   discovery / direct chat).
/// - Unattributed: non-agent calls with no parent key (workspace-scoped).
///
/// Group identity is (kind, key) plus — for everything except research — the
/// workspace: ticket ids already embed the workspace name and analyze-round keys
/// are scoped to one workspace, so the extra component closes any
/// cross-workspace key collision (e.g. two concurrent sync analyze rounds in
/// different workspaces drawing the same short NanoID suffix). Research groups
/// are matched on (kind, key) alone because a run's coder round executes in an
/// EPHEMERAL per-run workspace whose name IS the run key (job_id) while its
/// analysts and orchestrator calls run in the real workspace — a workspace
/// component would split the group. The run key is a durable 10-char NanoID
/// (~60 bits entropy), so (kind, key) stays unique across workspaces; the
/// group's displayed workspace is resolved to the REAL workspace whenever a
/// non-ephemeral member is seen.
fn build_groups(agents: &[AgentHandle], calls: &[NonAgentCallHandle]) -> Vec<DisplayGroup> {
    let mut groups: Vec<DisplayGroup> = Vec::new();

    let find_group =
        |kind: GroupKind, key: &str, workspace: &str, groups: &mut Vec<DisplayGroup>| -> usize {
            let research = kind == GroupKind::Research;
            if let Some(idx) = groups.iter().position(|g| {
                g.kind == kind && g.key == key && (research || g.workspace == workspace)
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
/// by workspace, each section headed by the workspace name.
///
/// Sections are ordered alphabetically by their resolved label (deterministic
/// — activity-based ordering would flicker as agents come and go); within a
/// section, groups keep the canonical kind order (tickets → analyze rounds →
/// research runs → singletons → unattributed) via [`DisplayGroup::sort_key`].
fn build_sections(
    groups: Vec<DisplayGroup>,
    workspaces: &std::collections::HashMap<String, WorkspaceInfo>,
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

/// Render one workspace section: the workspace name as the page-level header
/// (styled like a page title), then the section's groups.
///
/// The returned element owns all rendered content (text widgets take owned
/// Strings), so its lifetime is independent of the `section` borrow.
fn render_section(section: &WorkspaceSection) -> Element<'static, RunningMessage> {
    let mut groups = Column::new().spacing(10);
    for group in &section.groups {
        groups = groups.push(render_group(group));
    }
    column![
        text(section.label.clone())
            .size(18)
            .color(theme::TEXT_PRIMARY),
        groups,
    ]
    .spacing(10)
    .into()
}

/// Resolve a group's header: the primary title plus an optional small
/// secondary element (the ticket id).
///
/// - Ticket groups: the ticket NAME prominently, the ticket ID as a small
///   secondary element. When no title was captured (defensive — every live
///   ticket group carries one via its agents or the synthesis call), the ID
///   becomes the title.
/// - Analyze/research groups: the truncated question/task text; a generic
///   fallback when no label was captured. The raw NanoID key is NEVER shown.
/// - Singleton/unattributed groups: generic labels — their workspace name is
///   already the section header, so repeating it would be redundant.
fn group_title(group: &DisplayGroup) -> (String, Option<String>) {
    match &group.kind {
        GroupKind::Ticket => match &group.label {
            Some(title) => (title.clone(), Some(group.key.clone())),
            None => (group.key.clone(), None),
        },
        GroupKind::AnalyzeRound => (
            group.label.as_deref().map_or_else(
                || "Analyze round".to_string(),
                |l| crate::util::truncate(l, MAX_GROUP_LABEL_CHARS),
            ),
            None,
        ),
        GroupKind::Research => (
            group.label.as_deref().map_or_else(
                || "Research run".to_string(),
                |l| crate::util::truncate(l, MAX_GROUP_LABEL_CHARS),
            ),
            None,
        ),
        GroupKind::Singleton => ("Standalone".to_string(), None),
        GroupKind::Unattributed => ("Other LLM work".to_string(), None),
    }
}

/// Render one group: header (title + secondary + run-lifetime marker) then
/// the group panel holding its cards/rows. Cards are visually separated from
/// the panel via the established card style.
fn render_group(group: &DisplayGroup) -> Element<'static, RunningMessage> {
    let (title, secondary) = group_title(group);
    let mut header_parts: Vec<Element<'_, RunningMessage>> =
        vec![text(title).size(15).color(theme::ACCENT).into()];
    if let Some(secondary) = secondary {
        header_parts.push(text(secondary).size(11).color(theme::TEXT_MUTED).into());
    }
    if group.run_lifetime {
        header_parts.push(text("run active").size(11).color(theme::ACCENT).into());
    }
    // Manual cancel: danger-styled button on RESEARCH-run group headers only.
    // The run key is carried by the message, never rendered as text.
    if group.kind == GroupKind::Research {
        header_parts.push(Space::new().width(Length::Fill).into());
        header_parts.push(
            button(text("Cancel run").size(11))
                .style(theme::button_danger)
                .on_press(RunningMessage::CancelRequest(group.key.clone()))
                .into(),
        );
    }

    let mut items = Column::new().spacing(6);
    for item in &group.items {
        match item {
            DisplayItem::Agent(card) => {
                items = items.push(render_agent_card(card));
            }
            DisplayItem::Call(call) => {
                items = items.push(render_call_row(call));
            }
        }
    }

    column![
        Row::with_children(header_parts)
            .spacing(8)
            .align_y(Alignment::Center),
        container(items)
            .width(Length::Fill)
            .padding(10)
            .style(theme::elevated_card_style),
    ]
    .spacing(6)
    .into()
}

/// Build the research-run manual-cancel confirmation dialog, mirroring the
/// board's ticket-cancel pattern: consequences listed, danger confirm
/// button, Keep/Dismiss (and Escape/backdrop) to close with no effect.
/// `run_key` is the run's durable job id — never displayed as text.
fn cancel_confirm_dialog(run_key: &str) -> Element<'static, RunningMessage> {
    // Deliberately unused: the run key must never appear in the dialog's
    // rendered text — the button action alone carries it (see the enum docs).
    let _ = run_key;
    let dialog = container(
        column![
            text("Cancel this research run?")
                .size(16)
                .color(theme::TEXT_PRIMARY)
                .font(theme::FONT_BOLD),
            Space::new().height(12),
            text("Confirming will:")
                .size(13)
                .color(theme::TEXT_SECONDARY),
            Space::new().height(8),
            text(
                "• stop all agents of this run — an in-flight tool or LLM call \
                 may finish, but no further work happens;\n\
                 • stop the orchestrator — no more rounds, no report, no \
                 cleanup agent;\n\
                 • delete the run's temporary folder and archived result;\n\
                 • remove the run permanently — nothing is delivered to the \
                 Manager, and it can never resume.",
            )
            .size(13)
            .color(theme::TEXT_SECONDARY),
            Space::new().height(16),
            row![
                Space::new().width(Length::Fill),
                button(text("Keep run").size(13))
                    .style(theme::button_secondary)
                    .on_press(RunningMessage::CancelDismissed),
                Space::new().width(8),
                button(text("Cancel run").size(13))
                    .style(theme::button_danger)
                    .on_press(RunningMessage::CancelConfirmed),
            ]
            .align_y(Alignment::Center),
        ]
        .width(Length::Fill),
    )
    .width(Length::Fixed(480.0))
    .padding(24)
    .style(theme::dialog_container_style);

    widget_helpers::modal_backdrop(dialog, RunningMessage::CancelDismissed, 0.5)
}

/// Fallback label for workspaces outside the dashboard's registered set
/// (personal user spaces, ephemeral run-scoped workspaces).
fn fallback_workspace_label(workspace: &str) -> String {
    if workspace.is_empty() {
        "workspace".to_string()
    } else {
        format!("{workspace} (external)")
    }
}

/// Resolve a workspace's display label for a section header: the bare name
/// for registered workspaces, the fallback label (with the "(external)"
/// marker) for unregistered/ephemeral ones, and "workspace" for an empty
/// name.
fn workspace_label_for(
    name: &str,
    workspaces: &std::collections::HashMap<String, WorkspaceInfo>,
) -> String {
    if name.is_empty() {
        "workspace".to_string()
    } else if workspaces.contains_key(name) {
        name.to_string()
    } else {
        fallback_workspace_label(name)
    }
}

/// Render one compact running-agent card: role icon, elapsed time, and last
/// tool. The role text label and the workspace name are gone — the role icon
/// carries the role and the workspace section header carries the workspace.
/// Cards use the established surface-card style so they read as distinct from
/// the group panel.
fn render_agent_card(card: &AgentCard) -> Element<'static, RunningMessage> {
    let h = &card.handle;
    // Color resolution via the canonical string helper (handles derivative
    // names and falls back to muted grey for unknown roles); the icon still
    // needs a typed Role, so parse with the same Engineer fallback as before.
    let (fg, _bg) = theme::role_badge_color(&h.role);
    let role: crate::Role = h.role.parse().unwrap_or(crate::Role::Engineer);
    let icon = theme::role_icon(&role).size(20).color(fg);

    // Elapsed time in the brighter readable tone (~4:1 on the card
    // background, versus the old ~2:1 muted tone).
    let elapsed = format_elapsed(h.started_at);

    // Fill slot between the icon and the right-aligned metrics: running tools
    // if any, else the live activity phase, else the LAST COMPLETED tool — a
    // fast tool no longer flashes and vanishes. Each tool renders as its own
    // block (bold white name + truncated key-value pairs) wrapped in a hover
    // tooltip showing the full untruncated args; the activity label is plain
    // accent text with no tooltip.
    let mut fill = Column::new().spacing(4).align_x(Alignment::Start);
    for tool in &h.current_tools {
        fill = fill.push(render_tool_block(tool));
    }
    if h.current_tools.is_empty() {
        // Activity indicator — a non-tool LLM phase (verdict/summary
        // extraction, media transcription) running inside this agent. The
        // agent card is the single tracker for these calls (no separate call
        // rows are ever created), so the label is what keeps an
        // extracting/transcribing agent from looking idle.
        if let Some(activity) = &h.activity {
            fill = fill.push(text(activity.clone()).size(11).color(theme::ACCENT));
        } else if let Some(last) = &h.last_tool {
            fill = fill.push(render_tool_block(last));
        }
    }

    // Single row: role icon on the left; the tool/activity/last-tool display
    // fills the middle; the session length (when known) and elapsed time sit
    // right-aligned.
    let mut first_row = row![icon, fill.width(Length::Fill)]
        .spacing(8)
        .align_y(Alignment::Center);
    if let Some(len) = h.session_tokens {
        first_row = first_row.push(
            text(theme::format_compact_tokens(len))
                .size(11)
                .color(theme::TEXT_SECONDARY),
        );
    }
    first_row = first_row.push(text(elapsed).size(11).color(theme::TEXT_SECONDARY));

    container(first_row)
        .width(Length::Fill)
        .padding(8)
        .style(theme::surface_card_style)
        .into()
}

/// Render one tool block: bold white tool name on top, the comma-separated
/// key-value pairs below it in regular weight. The whole block is the hover
/// target of a tooltip showing the FULL untruncated argument values.
fn render_tool_block(tool: &crate::registry::RunningTool) -> Element<'static, RunningMessage> {
    let mut block = Column::new().spacing(2).align_x(Alignment::Start).push(
        text(tool.name.clone())
            .size(11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY),
    );
    if !tool.args.is_empty() {
        block = block.push(
            text(render_tool_pairs_line(&tool.args))
                .size(11)
                .color(theme::TEXT_SECONDARY),
        );
    }
    tooltip(block, render_tool_tooltip(tool), tooltip::Position::Top)
        .gap(4)
        .style(theme::tooltip_style)
        .into()
}

/// Render the row's key-value pairs line: `name: value` pairs, comma-
/// separated, each value collapsed to a single line and truncated to
/// [`MAX_TOOL_VALUE_CHARS`] chars. The whole line is capped at
/// [`MAX_TOOL_PAIRS_LINE_CHARS`] chars, cut at a pair boundary with "…" —
/// it can never overflow into the right-aligned metrics.
fn render_tool_pairs_line(pairs: &[(String, String)]) -> String {
    let rendered: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{k}: {}", value_display(v)))
        .collect();
    let joined = rendered.join(", ");
    if joined.chars().count() <= MAX_TOOL_PAIRS_LINE_CHARS {
        return joined;
    }
    // Over budget: keep whole pairs while they fit (leaving room for the
    // trailing "…"), then mark the cut.
    let mut out = String::new();
    for pair in &rendered {
        let sep = if out.is_empty() { "" } else { ", " };
        let candidate_len = out.chars().count() + sep.chars().count() + pair.chars().count();
        if candidate_len + 1 > MAX_TOOL_PAIRS_LINE_CHARS {
            break;
        }
        out.push_str(sep);
        out.push_str(pair);
    }
    if out.is_empty() {
        // Even the first pair alone does not fit — hard-truncate it.
        crate::util::truncate(&rendered[0], MAX_TOOL_PAIRS_LINE_CHARS)
    } else {
        format!("{out}…")
    }
}

/// Single-line display form of a value: control characters (newlines, tabs)
/// collapsed to spaces, then truncated to [`MAX_TOOL_VALUE_CHARS`] chars with
/// "…" when cut.
fn value_display(value: &str) -> String {
    let single_line: String = value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    crate::util::truncate(&single_line, MAX_TOOL_VALUE_CHARS)
}

/// Render the hover tooltip: the tool name header (bold white) followed by
/// every argument pair on its own line, sorted by FULL value length ascending
/// (stable, so equal-length pairs keep registration order) — the shortest
/// pairs sit at the top and stay visible even when the longest values extend
/// beyond the viewport. Values are full and untruncated, including secrets.
fn render_tool_tooltip(tool: &crate::registry::RunningTool) -> Element<'static, RunningMessage> {
    let mut pairs = tool.args.clone();
    pairs.sort_by_key(|(_, v)| v.chars().count());

    let mut content = Column::new().spacing(2).push(
        text(tool.name.clone())
            .size(11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY),
    );
    for (k, v) in &pairs {
        content = content.push(
            text(format!("{k}: {v}"))
                .size(11)
                .font(super::JETBRAINS_MONO)
                .color(theme::TEXT_SECONDARY)
                .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
        );
    }
    container(content).max_width(MAX_TOOL_TOOLTIP_WIDTH).into()
}

/// Render a compact non-agent LLM call row: zap marker + purpose + elapsed.
/// No workspace name (the section header carries it); the elapsed time uses
/// the same brighter tone as agent cards. The purpose is the static
/// human-readable label — raw kind names never leak.
fn render_call_row(call: &CallRow) -> Element<'static, RunningMessage> {
    let h = &call.handle;
    let elapsed = format_elapsed(h.started_at);
    let purpose = crate::registry::call_kind_label(h.kind);

    row![
        lucide::zap::<iced::Theme, iced::Renderer>()
            .size(16)
            .color(theme::ACCENT),
        text(purpose).size(12).color(theme::TEXT_SECONDARY),
        Space::new().width(Length::Fill),
        text(elapsed).size(11).color(theme::TEXT_SECONDARY),
    ]
    .spacing(6)
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
    use crate::registry::AgentHandle;

    fn agent_handle(
        id: &str,
        role: &str,
        ticket_id: Option<String>,
        workspace: &str,
        parent: Option<ParentKey>,
    ) -> AgentHandle {
        AgentHandle {
            agent_id: id.to_string(),
            role: role.to_string(),
            ticket_id,
            workspace_path: format!("/ws/{workspace}"),
            workspace_name: workspace.to_string(),
            parent_key: parent,
            parent_label: None,
            started_at: Utc::now(),
            label: role.to_string(),
            current_tools: Vec::new(),
            last_tool: None,
            activity: None,
            session_tokens: None,
        }
    }

    fn call_handle(
        kind: &'static str,
        workspace: &str,
        parent: Option<ParentKey>,
        run_lifetime: bool,
    ) -> NonAgentCallHandle {
        NonAgentCallHandle {
            kind,
            workspace: workspace.to_string(),
            started_at: Utc::now(),
            parent_key: parent,
            parent_label: None,
            run_lifetime,
        }
    }

    #[test]
    fn groups_ticket_agents_and_synthesis_together() {
        let agents = vec![
            agent_handle(
                "a1",
                "analyst",
                Some("T1".to_string()),
                "ws1",
                Some(ParentKey::Ticket("T1".to_string())),
            ),
            agent_handle("mgr", "manager", None, "ws1", None),
        ];
        let calls = vec![call_handle(
            "synthesis",
            "ws1",
            Some(ParentKey::Ticket("T1".to_string())),
            false,
        )];
        let groups = build_groups(&agents, &calls);
        assert_eq!(groups.len(), 2, "ticket group + singleton group");
        let ticket = &groups[0];
        assert_eq!(ticket.kind, GroupKind::Ticket);
        assert_eq!(ticket.key, "T1");
        assert_eq!(ticket.items.len(), 2, "analyst agent + synthesis call");
    }

    #[test]
    fn two_analyze_rounds_never_mix_members() {
        let agents = vec![
            agent_handle(
                "analyze_ws_AAA_0_analyst",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::AnalyzeRound("roundA".to_string())),
            ),
            agent_handle(
                "analyze_ws_AAA_1_analyst",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::AnalyzeRound("roundA".to_string())),
            ),
            agent_handle(
                "analyze_ws_BBB_0_analyst",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::AnalyzeRound("roundB".to_string())),
            ),
        ];
        let calls = vec![call_handle(
            "consolidate",
            "ws1",
            Some(ParentKey::AnalyzeRound("roundA".to_string())),
            false,
        )];
        let groups = build_groups(&agents, &calls);
        let analyze_groups: Vec<_> = groups
            .iter()
            .filter(|g| g.kind == GroupKind::AnalyzeRound)
            .collect();
        assert_eq!(analyze_groups.len(), 2, "two distinct analyze round groups");
        let round_a = analyze_groups
            .iter()
            .find(|g| g.key == "roundA")
            .expect("round A exists");
        assert_eq!(round_a.items.len(), 3, "2 analysts + consolidation call");
        let round_b = analyze_groups
            .iter()
            .find(|g| g.key == "roundB")
            .expect("round B exists");
        assert_eq!(round_b.items.len(), 1, "only its own analyst");
    }

    #[test]
    fn research_run_members_share_one_key_across_phases() {
        // Two concurrent research runs in one workspace must never mix
        // members even though every spawn carries a unique agent-id suffix.
        let agents = vec![
            agent_handle(
                "research_ws_x1_decompose_0",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::Research("run1".to_string())),
            ),
            agent_handle(
                "research_ws_x2_r1_0",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::Research("run1".to_string())),
            ),
            agent_handle(
                "research_ws_y1_decompose_0",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::Research("run2".to_string())),
            ),
        ];
        let calls = vec![
            call_handle(
                "synthesize",
                "ws1",
                Some(ParentKey::Research("run1".to_string())),
                false,
            ),
            call_handle(
                "research_orchestrator",
                "ws1",
                Some(ParentKey::Research("run1".to_string())),
                true,
            ),
        ];
        let groups = build_groups(&agents, &calls);
        let research_groups: Vec<_> = groups
            .iter()
            .filter(|g| g.kind == GroupKind::Research)
            .collect();
        assert_eq!(research_groups.len(), 2);
        let run1 = research_groups
            .iter()
            .find(|g| g.key == "run1")
            .expect("run 1 exists");
        assert_eq!(run1.items.len(), 3, "2 analysts + synthesize call");
        assert!(
            run1.run_lifetime,
            "run-lifetime orchestrator marker attached"
        );
        let run2 = research_groups
            .iter()
            .find(|g| g.key == "run2")
            .expect("run 2 exists");
        assert_eq!(run2.items.len(), 1, "only its own member");
        assert!(!run2.run_lifetime);
    }

    #[test]
    fn singletons_and_unattributed_calls_group_by_workspace() {
        let agents = vec![
            agent_handle("manager_ws1", "manager", None, "ws1", None),
            agent_handle("maintainer_ws2_abc", "maintainer", None, "ws2", None),
        ];
        let calls = vec![call_handle("some_orchestrator_call", "ws1", None, false)];
        let groups = build_groups(&agents, &calls);
        let singleton: Vec<_> = groups
            .iter()
            .filter(|g| g.kind == GroupKind::Singleton)
            .collect();
        assert_eq!(singleton.len(), 2, "one per workspace");
        let unattributed: Vec<_> = groups
            .iter()
            .filter(|g| g.kind == GroupKind::Unattributed)
            .collect();
        assert_eq!(unattributed.len(), 1);
        assert_eq!(unattributed[0].workspace, "ws1");
    }

    #[test]
    fn sort_key_orders_groups_by_kind() {
        let mut groups = [
            DisplayGroup {
                kind: GroupKind::Unattributed,
                key: "ws".to_string(),
                workspace: "ws".to_string(),
                label: None,
                items: Vec::new(),
                run_lifetime: false,
            },
            DisplayGroup {
                kind: GroupKind::Ticket,
                key: "T1".to_string(),
                workspace: "ws".to_string(),
                label: None,
                items: Vec::new(),
                run_lifetime: false,
            },
            DisplayGroup {
                kind: GroupKind::Research,
                key: "r".to_string(),
                workspace: "ws".to_string(),
                label: None,
                items: Vec::new(),
                run_lifetime: false,
            },
        ];
        groups.sort_by_key(DisplayGroup::sort_key);
        assert_eq!(groups[0].kind, GroupKind::Ticket);
        assert_eq!(groups[1].kind, GroupKind::Research);
        assert_eq!(groups[2].kind, GroupKind::Unattributed);
    }

    #[test]
    fn research_groups_resolve_workspace_across_ephemeral_coder_members() {
        // A run's coder round executes in an ephemeral per-run workspace whose
        // name IS the run key (job_id), while its analysts and orchestrator
        // calls run in the real workspace. The group must resolve to the REAL
        // workspace whether the coder registers before or after the run's
        // real-workspace members, and two concurrent runs must never mix.
        let agents = vec![
            // run1: ephemeral coder registers first, then a real-workspace
            // analyst.
            agent_handle(
                "research_ws1_run1_coder0",
                "coder",
                None,
                "run1",
                Some(ParentKey::Research("run1".to_string())),
            ),
            agent_handle(
                "research_ws1_run1_r1_0",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::Research("run1".to_string())),
            ),
            // run2: real-workspace analyst registers first, then the
            // ephemeral coder.
            agent_handle(
                "research_ws1_run2_r1_0",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::Research("run2".to_string())),
            ),
            agent_handle(
                "research_ws1_run2_coder0",
                "coder",
                None,
                "run2",
                Some(ParentKey::Research("run2".to_string())),
            ),
        ];
        let groups = build_groups(&agents, &[]);
        let runs: Vec<_> = groups
            .iter()
            .filter(|g| g.kind == GroupKind::Research)
            .collect();
        assert_eq!(runs.len(), 2, "two concurrent runs never mix members");
        for run in runs {
            assert_eq!(
                run.workspace, "ws1",
                "the real workspace wins over the ephemeral per-run workspace"
            );
            assert_eq!(run.items.len(), 2, "coder + analyst in one group");
        }
    }

    #[test]
    fn cross_workspace_round_keys_never_merge_groups() {
        // Two concurrent sync analyze rounds in DIFFERENT workspaces could draw
        // the same short NanoID suffix — the workspace component of the group
        // identity must keep their groups separate.
        let agents = vec![
            agent_handle(
                "analyze_ws_AAA_0_analyst",
                "analyst",
                None,
                "ws1",
                Some(ParentKey::AnalyzeRound("abc123".to_string())),
            ),
            agent_handle(
                "analyze_ws_BBB_0_analyst",
                "analyst",
                None,
                "ws2",
                Some(ParentKey::AnalyzeRound("abc123".to_string())),
            ),
        ];
        let groups = build_groups(&agents, &[]);
        let analyze_groups: Vec<_> = groups
            .iter()
            .filter(|g| g.kind == GroupKind::AnalyzeRound)
            .collect();
        assert_eq!(
            analyze_groups.len(),
            2,
            "same round key in two workspaces stays separate"
        );
    }

    /// Build a handle with a parent label (the display-level label captured
    /// at spawn — ticket title / question text).
    fn agent_handle_labeled(
        id: &str,
        role: &str,
        workspace: &str,
        parent: Option<ParentKey>,
        parent_label: Option<&str>,
    ) -> AgentHandle {
        let mut h = agent_handle(id, role, None, workspace, parent);
        h.parent_label = parent_label.map(ToString::to_string);
        h
    }

    fn call_handle_labeled(
        kind: &'static str,
        workspace: &str,
        parent: Option<ParentKey>,
        parent_label: Option<&str>,
    ) -> NonAgentCallHandle {
        let mut h = call_handle(kind, workspace, parent, false);
        h.parent_label = parent_label.map(ToString::to_string);
        h
    }

    #[test]
    fn ticket_group_label_comes_from_members_not_the_key() {
        // Ticket title is captured observationally at spawn (agents via
        // Agent::new's ticket, the synthesis call via the joint-verdict
        // threading). A group must carry the title even when only the
        // synthesis call remains — never degrade to an ID-only header.
        let agents = vec![agent_handle_labeled(
            "a1",
            "analyst",
            "ws1",
            Some(ParentKey::Ticket("T1".to_string())),
            Some("Fix the login flow"),
        )];
        let calls = vec![call_handle_labeled(
            "synthesis",
            "ws1",
            Some(ParentKey::Ticket("T1".to_string())),
            Some("Fix the login flow"),
        )];
        let groups = build_groups(&agents, &calls);
        let ticket = groups
            .iter()
            .find(|g| g.kind == GroupKind::Ticket)
            .expect("ticket group exists");
        assert_eq!(
            ticket.label.as_deref(),
            Some("Fix the login flow"),
            "group label adopted from a member's parent label"
        );
        // The header title is the ticket name; the id is only the secondary.
        let (title, secondary) = group_title(ticket);
        assert_eq!(title, "Fix the login flow");
        assert_eq!(secondary.as_deref(), Some("T1"));
    }

    #[test]
    fn analyze_and_research_groups_render_question_not_raw_key() {
        // Analyze/research headers show the question text; the raw NanoID key
        // is never rendered. A missing label degrades to a generic label,
        // never the key.
        let agents = vec![agent_handle_labeled(
            "a1",
            "analyst",
            "ws1",
            Some(ParentKey::AnalyzeRound("job_abc".to_string())),
            Some("Why is CI flaky?"),
        )];
        let groups = build_groups(&agents, &[]);
        let analyze = groups
            .iter()
            .find(|g| g.kind == GroupKind::AnalyzeRound)
            .expect("analyze group exists");
        let (title, secondary) = group_title(analyze);
        assert_eq!(title, "Why is CI flaky?");
        assert!(secondary.is_none(), "no id secondary for analyze groups");
        assert!(!title.contains("job_abc"), "raw key never leaks");

        let mut no_label = analyze.clone();
        no_label.label = None;
        let (fallback, _) = group_title(&no_label);
        assert_eq!(fallback, "Analyze round");
        assert!(!fallback.contains("job_abc"), "generic fallback, no key");
    }

    #[test]
    fn sections_group_by_workspace_and_keep_kind_order() {
        // Workspace-first layout. Sections are alphabetical; within a section,
        // groups keep the canonical kind order. Singleton groups are keyed by
        // workspace, so they land in their own workspace's section.
        let mut ws_map = std::collections::HashMap::new();
        ws_map.insert(
            "ws2".to_string(),
            WorkspaceInfo::test_new("/ws/ws2".to_string(), false, false),
        );
        let agents = vec![
            agent_handle(
                "t_agent",
                "engineer",
                Some("T1".to_string()),
                "ws1",
                Some(ParentKey::Ticket("T1".to_string())),
            ),
            agent_handle("mgr_ws2", "manager", None, "ws2", None),
            agent_handle("mgr_ws1", "manager", None, "ws1", None),
        ];
        let groups = build_groups(&agents, &[]);
        let sections = build_sections(groups, &ws_map);
        assert_eq!(sections.len(), 2, "one section per workspace");
        assert_eq!(sections[0].workspace, "ws1");
        assert_eq!(sections[1].workspace, "ws2");
        // ws1: ticket group before singleton (kind order).
        assert_eq!(sections[0].groups[0].kind, GroupKind::Ticket);
        assert_eq!(sections[0].groups[1].kind, GroupKind::Singleton);
        // ws2 is registered → bare name; ws1 is unregistered → "(external)".
        assert_eq!(sections[1].label, "ws2");
        assert_eq!(sections[0].label, "ws1 (external)");
    }

    #[test]
    fn singleton_and_unattributed_headers_are_generic_not_workspace_titled() {
        // Inside a workspace section the singleton/unattributed group headers
        // must NOT repeat the workspace name (the section header carries it).
        let agents = vec![agent_handle("mgr", "manager", None, "ws1", None)];
        let calls = vec![call_handle("some_orchestrator_call", "ws1", None, false)];
        let groups = build_groups(&agents, &calls);
        for group in &groups {
            let (title, _) = group_title(group);
            assert_ne!(title, "ws1", "group header must not duplicate the section");
            assert!(
                !title.contains("ws1"),
                "no workspace name in group headers: {title}"
            );
        }
    }
}
