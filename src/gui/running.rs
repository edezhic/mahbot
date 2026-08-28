//! "Running Agents" dashboard page — a live view of every currently-running
//! agent and in-flight non-agent LLM work, grouped by WORKSPACE, then by
//! DIRECT PARENT INVOCATION (ticket / analyze round / research run / workspace
//! singleton / unattributable orchestrator calls).
//!
//! The view is read-only: at render time it reads the in-memory registries
//! (`AGENT_REGISTRY` and `NON_AGENT_CALLS`) plus the live transcript snapshots
//! (`TRANSCRIPT_REGISTRY`), so it reflects the live in-memory conversation
//! including the unpersisted tail. No database reads, no schema changes, no
//! new subscriptions, no history retained between ticks. The existing 1-second
//! UI tick re-renders the page, so the view refreshes at that cadence for free.
//!
//! Truthfulness rules:
//! - The LIVE state line (top of each card) shows what is running right now:
//!   the currently-executing tool(s) (expanded with their args) if any, else
//!   the live activity phase label (a non-tool LLM call), else the current
//!   trace group's visible narration (the assistant's short reasoning text)
//!   when available, else a static "thinking…" indicator. The transcript
//!   cannot see an in-flight tool or a non-tool LLM phase (those are only
//!   committed after execution / produce no message respectively), so this is
//!   the only record of "what's running now".
//! - Trace groups (below the LIVE line) are derived from the live transcript
//!   snapshot: tool-call assistant rounds grouped by shared narration, bounded
//!   by real user turns. Content-only final answers are omitted.
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
//! live in-memory registries and the live transcript snapshots, plus the
//! dashboard's registered-workspace map (only for the "(external)" marker on
//! unregistered/ephemeral workspace sections). All header labels (ticket
//! titles, analyze/research questions) are captured observationally at spawn —
//! never read from the DB.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::agent::registry::{AgentHandle, NonAgentCallHandle, ParentKey, RunningTool};
use crate::gui::theme;
use crate::gui::widgets;
use crate::session::{DecodedNativeHistoryMessage, decode_native_history_message};
use crate::{ChatMessage, ChatRole};
use chrono::{DateTime, Utc};

use iced::widget::{
    Column, Row, Space, button, column, container, mouse_area, row, scrollable, stack, text,
    tooltip,
};
use iced::{Alignment, Element, Length};

use iced_fonts::lucide;

use super::{Message, WorkspaceInfo};

/// Maximum display length (Unicode chars) of an analyze/research group header
/// label (the question/task text). Truncated at a word/path-delimiter boundary
/// with "…" (see [`truncate_at_boundary`]).
const MAX_GROUP_LABEL_CHARS: usize = 104;

/// Maximum display length (Unicode chars) of a single argument VALUE in the
/// row's comma-separated key-value pairs line. The hover tooltip always shows
/// the full untruncated value.
const MAX_TOOL_VALUE_CHARS: usize = 26;

/// Maximum display length (Unicode chars) of the whole key-value pairs line
/// in the row (values already truncated per [`MAX_TOOL_VALUE_CHARS`]). Cut at
/// a pair boundary with a trailing "…" so the line can never overflow into
/// the right-aligned metrics (iced rows do not wrap).
const MAX_TOOL_PAIRS_LINE_CHARS: usize = 104;

/// Maximum display length (Unicode chars) of the visible narration (the
/// assistant's short reasoning text) on the LIVE state line and its
/// trace-group label. Truncated at a word/path-delimiter boundary with "…" so
/// a long narration cannot crowd the row or push the metrics off.
const MAX_NARRATION_CHARS: usize = 104;

/// Maximum width (px) of the hover tooltip content; long values wrap within it.
const MAX_TOOL_TOOLTIP_WIDTH: f32 = 560.0;

/// Maximum width (px) of a metric hover tooltip body. Much narrower than
/// [`MAX_TOOL_TOOLTIP_WIDTH`] (tuned for multi-arg tool calls) since a metric
/// tooltip holds one short line of explanatory text.
const MAX_METRIC_TOOLTIP_WIDTH: f32 = 360.0;

/// A word/path-delimiter boundary at which wrapped/truncated page text may be
/// cut: whitespace (word wrap) and `/`, `_`, `-` (paths, URLs, identifiers).
fn is_delim(c: char) -> bool {
    c.is_whitespace() || matches!(c, '/' | '_' | '-')
}

/// Truncate `s` to at most `max_chars` Unicode chars, choosing a cut at a word
/// or path/URL delimiter boundary (whitespace, `/`, `_`, `-`) rather than
/// mid-token, and appending "…" when truncated. When no delimiter falls within
/// the cap the cut falls back to a hard char boundary so the limit is always
/// honoured. Used by the page-local truncation constants; leaves
/// [`crate::util::truncate`](crate::util::truncate) untouched for its ~20
/// non-UI callers.
fn truncate_at_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let hard_end = s
        .char_indices()
        .nth(max_chars)
        .map(|(byte_idx, _)| byte_idx)
        .expect("char count exceeds max_chars, so a char exists at index max_chars");
    // If the cut lands right before a delimiter, the first `max_chars` chars
    // already end at a complete word/path segment — cut exactly there.
    if s[hard_end..].chars().next().is_some_and(is_delim) {
        return format!("{}…", s[..hard_end].trim_end());
    }
    // Otherwise back up to the last delimiter strictly inside the cap so the
    // cut lands at a wrap boundary (the delimiter is the wrapped segment's
    // final char) instead of mid-word.
    let mut cut = None;
    for (char_pos, (byte_idx, c)) in s.char_indices().enumerate() {
        if char_pos >= max_chars {
            break;
        }
        if is_delim(c) {
            cut = Some(byte_idx + c.len_utf8());
        }
    }
    let end = cut.unwrap_or(hard_end);
    format!("{}…", s[..end].trim_end())
}

/// Messages emitted by the Running Agents page.
///
/// The page itself stays stateless — the pending research-cancel
/// confirmation lives on the [`Dashboard`](super::Dashboard) so the dialog
/// requires no per-run page state beyond the pending confirmation itself.
#[derive(Debug, Clone)]
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
    /// Toggle the expanded/collapsed state of a running-agent card. Keyed by
    /// (agent_id, generation) so a recycled agent_id never inherits a stale
    /// expansion.
    ToggleAgentExpanded { agent_id: String, generation: u64 },
}

/// Render the live running-agents page.
///
/// The page is observational — it reads the live in-memory registries and
/// renders directly into the dashboard's [`Message`] type; its only
/// self-emitted messages are the research-run manual-cancel flow
/// ([`RunningMessage`]) and the card expand/collapse toggle. `workspaces` is
/// the dashboard's registered-workspace map (name → info) — used only to mark
/// unregistered/ephemeral workspace sections with the "(external)" suffix.
/// Everything else comes from the live in-memory registries.
///
/// `expanded` is the dashboard's set of expanded agent cards, keyed by
/// (agent_id, generation). Stale keys are pruned by the dashboard's
/// [`process_tick`](super::Dashboard::process_tick) against the freshly-listed
/// agents; rendering here only reads the set.
pub(crate) fn view(
    workspaces: &HashMap<String, WorkspaceInfo>,
    pending_cancel: Option<&str>,
    expanded: &HashSet<(String, u64)>,
) -> Element<'static, Message> {
    let agents = crate::agent::registry::AGENT_REGISTRY.list();
    let calls = crate::agent::registry::NON_AGENT_CALLS.list();

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
            content = content.push(render_section(section, expanded));
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
        stack([widgets::empty_stack_placeholder()]).into()
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
fn render_section(
    section: &WorkspaceSection,
    expanded: &HashSet<(String, u64)>,
) -> Element<'static, RunningMessage> {
    let mut groups = Column::new().spacing(10);
    for group in &section.groups {
        groups = groups.push(render_group(group, expanded));
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

/// Resolve a group's header parts: the primary label plus an optional
/// secondary element (the ticket id). `render_group` applies the visual
/// hierarchy — for ticket groups the ID is rendered large/prominent and the
/// title is the small, brighter subordinate line; single-label groups render
/// their label as the small, brighter subordinate line.
///
/// - Ticket groups: the ticket NAME plus the ticket ID as the secondary. When
///   no title was captured (defensive — every live ticket group carries one
///   via its agents or the synthesis call), the ID becomes the sole label.
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
        GroupKind::Singleton => ("Standalone".to_string(), None),
        GroupKind::Unattributed => ("Other LLM work".to_string(), None),
    }
}

/// Render one group: header (ID/label + run-lifetime marker) then the group
/// panel holding its cards/rows. Cards are visually separated from the panel
/// via the established card style.
fn render_group(
    group: &DisplayGroup,
    expanded: &HashSet<(String, u64)>,
) -> Element<'static, RunningMessage> {
    let (title, secondary) = group_title(group);
    let mut header_parts: Vec<Element<'_, RunningMessage>> = Vec::new();
    if group.kind == GroupKind::Ticket {
        // Ticket group: the ID is the prominent ACCENT element rendered first;
        // the title is the small, brighter subordinate line rendered second.
        // When no title was captured (defensive), the ID is the sole large
        // element.
        if let Some(id) = secondary {
            header_parts.push(text(id).size(15).color(theme::ACCENT).into());
            header_parts.push(text(title).size(12).color(theme::TEXT_SECONDARY).into());
        } else {
            header_parts.push(text(title).size(15).color(theme::ACCENT).into());
        }
    } else {
        // Single-label groups (analyze/research/singleton/unattributed): the
        // label is the small, brighter subordinate line (no large element).
        header_parts.push(text(title).size(12).color(theme::TEXT_SECONDARY).into());
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

    widgets::modal_backdrop(dialog, RunningMessage::CancelDismissed, 0.5)
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

/// Render one running-agent card: role icon, LIVE state line, trace groups,
/// and right-aligned metrics (live token count + elapsed time). The whole card
/// is clickable to expand/collapse the trace groups.
///
/// Live state: the currently-executing tool(s) (expanded with their args) if
/// any, else the live activity phase label, else the current trace group's
/// visible narration (the assistant's short reasoning text) when available,
/// else a static "thinking…" placeholder. Trace groups are derived from the
/// live transcript snapshot (the unpersisted tail included) — the current
/// (latest) group in the collapsed card, plus up to 5 previous groups when
/// expanded.
fn render_agent_card(card: &AgentCard, expanded: bool) -> Element<'static, RunningMessage> {
    let h = &card.handle;
    // Color resolution via the canonical string helper (handles derivative
    // names and falls back to muted grey for unknown roles); the icon still
    // needs a typed Role, so parse with the same Engineer fallback as before.
    let (fg, _bg) = theme::role_badge_color(&h.role);
    let role: crate::Role = h.role.parse().unwrap_or(crate::Role::Engineer);
    let icon = theme::role_icon(&role).size(20).color(fg);
    let elapsed = format_elapsed(h.started_at);

    // The live transcript snapshot feeds both the LIVE line narration and the
    // trace groups below, so fetch it once up front.
    let snapshot = crate::session::TRANSCRIPT_REGISTRY.snapshot(&h.agent_id);
    let groups = snapshot
        .as_ref()
        .map(|s| derive_trace_groups(&s.history))
        .unwrap_or_default();

    // Live state column: running tools (expanded), else activity phase, else
    // the current trace group's narration, else "thinking…". The transcript
    // cannot see an in-flight tool or a non-tool LLM phase, so this is the only
    // record of "what's running now".
    let status = live_status(h, &groups);
    // The current trace group's narration label is suppressed whenever the LIVE
    // line already carries the status text (a narration or "thinking…").
    let current_label_suppressed =
        matches!(status, LiveStatus::Narration(_) | LiveStatus::Thinking);
    let mut live = Column::new().spacing(4).align_x(Alignment::Start);
    match status {
        LiveStatus::Tools => {
            for tool in &h.current_tools {
                live = live.push(render_tool_block(tool));
            }
        }
        LiveStatus::Activity(activity) => {
            live = live.push(text(activity.to_owned()).size(11).color(theme::ACCENT));
        }
        LiveStatus::Narration(narration) => {
            live = live.push(
                text(truncate_at_boundary(narration, MAX_NARRATION_CHARS))
                    .size(11)
                    .font(theme::FONT_BOLD)
                    .color(theme::ACCENT),
            );
        }
        LiveStatus::Thinking => {
            live = live.push(render_thinking_indicator());
        }
    }

    let mut first_row = row![icon, live.width(Length::Fill)]
        .spacing(8)
        .align_y(Alignment::Center);
    // The live token count comes from the published transcript snapshot (the
    // registry no longer carries a session_tokens mirror). It renders as a
    // compact count; hovering reveals the label and the exact unrounded value.
    if let Some(token_count) = snapshot.as_ref().and_then(|s| s.token_count) {
        first_row = first_row.push(
            tooltip(
                text(theme::format_compact_tokens(token_count))
                    .size(11)
                    .color(theme::TEXT_SECONDARY),
                render_metric_tooltip(format!("Session tokens: {token_count}")),
                tooltip::Position::Top,
            )
            .gap(4)
            .style(theme::tooltip_style),
        );
    }
    // Elapsed time since the agent started; hovering explains the meaning.
    first_row = first_row.push(
        tooltip(
            text(elapsed).size(11).color(theme::TEXT_SECONDARY),
            render_metric_tooltip("Elapsed run time since the agent started".to_string()),
            tooltip::Position::Top,
        )
        .gap(4)
        .style(theme::tooltip_style),
    );

    let mut content = Column::new().spacing(6).align_x(Alignment::Start);
    content = content.push(first_row);
    // Trace groups from the live snapshot (current unless the agent has no
    // transcript yet). Content-only assistant messages (final answers) are
    // deliberately omitted — only tool-call assistant messages form groups.
    if !groups.is_empty() {
        for group in render_visible_groups(&groups, expanded, current_label_suppressed) {
            content = content.push(group);
        }
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
            .padding(8)
            .style(theme::surface_card_style),
    )
    .on_press(on_press)
    .into()
}

/// Render the visible trace groups for a card: the current (latest) group
/// always (last, chronologically), plus up to 5 previous groups when the card
/// is expanded. Returns owned elements. The current group's narration label is
/// omitted when the LIVE line already carries the status text
/// (`current_label_suppressed`).
fn render_visible_groups(
    groups: &[TraceGroup],
    expanded: bool,
    current_label_suppressed: bool,
) -> Vec<Element<'static, RunningMessage>> {
    let mut rendered = Vec::new();
    let (current, previous) = groups
        .split_last()
        .expect("caller guarantees at least one group");
    if expanded {
        // The 5 most-recent previous groups, oldest first, then current.
        for group in previous.iter().rev().take(5).rev() {
            rendered.push(render_trace_group(group, false, true));
        }
    }
    // Current group: newest round expanded, earlier rounds collapsed.
    rendered.push(render_trace_group(current, true, !current_label_suppressed));
    rendered
}

/// Render one trace group. `expand_current` renders the newest round's calls
/// expanded (current group); otherwise every call collapses to a name-only
/// line (previous groups). `show_label` renders the narration label (the short
/// reasoning text, or the "thinking…" placeholder when empty); the current
/// group omits it when the card's LIVE line already carries the status text
/// (a narration, or the "thinking…" placeholder).
fn render_trace_group(
    group: &TraceGroup,
    expand_current: bool,
    show_label: bool,
) -> Element<'static, RunningMessage> {
    let mut column = Column::new().spacing(4).align_x(Alignment::Start);
    if show_label {
        if group.narration.is_empty() {
            column = column.push(text("thinking…").size(11).color(theme::TEXT_SECONDARY));
        } else {
            column = column.push(
                text(truncate_at_boundary(&group.narration, MAX_NARRATION_CHARS))
                    .size(11)
                    .font(theme::FONT_BOLD)
                    .color(theme::TEXT_SECONDARY),
            );
        }
    }

    if expand_current {
        let (newest, earlier) = group
            .rounds
            .split_last()
            .expect("a trace group always carries at least one round");
        if !earlier.is_empty() {
            column = column.push(
                text(collapsed_calls_line(earlier))
                    .size(10)
                    .color(theme::TEXT_SECONDARY),
            );
        }
        for call in newest {
            column = column.push(render_tool_block(call));
        }
    } else {
        column = column.push(
            text(collapsed_calls_line(&group.rounds))
                .size(10)
                .color(theme::TEXT_SECONDARY),
        );
    }
    column.into()
}

/// Compose the collapsed name-only line for a set of tool-call rounds: tool
/// names with underscores replaced by spaces, in first-appearance order, each
/// unique name suffixed with `xN` when it appears more than once within the
/// collapsed set (e.g. `read file x2, list files`).
fn collapsed_calls_line(rounds: &[Vec<RunningTool>]) -> String {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for round in rounds {
        for call in round {
            let name = call.name.replace('_', " ");
            if !counts.contains_key(&name) {
                order.push(name.clone());
            }
            *counts.entry(name).or_default() += 1;
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

/// One trace group in a running agent's live transcript: a maximal run of
/// consecutive tool-call assistant rounds sharing a single (possibly empty)
/// visible narration, bounded by a real user turn or a new narration.
///
/// Derived only from the decoded assistant `content` (the short visible
/// narration) and `tool_calls` — never from the long `reasoning`
/// (`Reasoning` struct) block. A group always carries at least one round.
struct TraceGroup {
    /// The assistant's short visible narration (empty → rendered "thinking…").
    /// This is the decoded assistant `content`, NEVER the long `Reasoning`
    /// / `[thinking]` block.
    narration: String,
    /// Tool-call rounds in first-appearance order; the LAST round is the newest
    /// (expanded in the current group), earlier rounds collapse.
    rounds: Vec<Vec<RunningTool>>,
}

/// Derive the trace groups for a live transcript in chronological order.
///
/// Boundary rules (see the ticket):
/// - A new (non-empty) narration in a tool-call assistant message starts a new
///   group; that narration becomes the group's `narration`.
/// - Consecutive tool-call rounds with no narration accumulate into the current
///   group.
/// - A REAL user turn (content NOT starting with the injected-image tag) closes
///   the current group; the next tool-call round starts a fresh one even with
///   no narration (its `narration` slots to "thinking…").
/// - Synthetic tool-injected `[IMAGE:*]` user messages are part of the ongoing
///   tool sequence and never reset the group.
fn derive_trace_groups(history: &[ChatMessage]) -> Vec<TraceGroup> {
    let mut groups: Vec<TraceGroup> = Vec::new();
    let mut current: Option<TraceGroup> = None;
    for msg in history {
        match msg.role {
            ChatRole::User => {
                if !msg.content.starts_with(crate::util::INJECTED_IMAGE_TAG) {
                    if let Some(group) = current.take() {
                        groups.push(group);
                    }
                }
            }
            ChatRole::Assistant => {
                let Some(DecodedNativeHistoryMessage::Assistant {
                    content,
                    tool_calls,
                    ..
                }) = decode_native_history_message(msg)
                else {
                    continue;
                };
                let Some(calls) = tool_calls else {
                    continue;
                };
                if calls.is_empty() {
                    continue;
                }
                let round: Vec<RunningTool> =
                    calls.iter().map(RunningTool::from_tool_call).collect();
                let narration = content.unwrap_or_default();
                let new_group = !narration.is_empty() || current.is_none();
                if new_group {
                    if let Some(group) = current.take() {
                        groups.push(group);
                    }
                }
                let group = current.get_or_insert_with(|| TraceGroup {
                    narration,
                    rounds: Vec::new(),
                });
                group.rounds.push(round);
            }
            _ => {}
        }
    }
    if let Some(group) = current.take() {
        groups.push(group);
    }
    groups
}

/// What the LIVE state line shows for a running-agent card. Drives both the
/// rendered status and whether the current trace group's narration label is
/// suppressed (suppressed for [`LiveStatus::Narration`] and
/// [`LiveStatus::Thinking`], where the LIVE line already carries the status
/// text).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveStatus<'a> {
    /// One or more tools are executing (rendered as tool blocks).
    Tools,
    /// A non-tool LLM phase is running (activity label).
    Activity(&'a str),
    /// The current trace group's narration (the assistant's short reasoning).
    Narration(&'a str),
    /// No tool, no activity, no narration → static "thinking…".
    Thinking,
}

/// Determine the LIVE state line's status for a card: the currently-executing
/// tool(s) if any, else the live activity phase label, else the current trace
/// group's narration when available, else the "thinking…" placeholder. The
/// narration is the assistant's short decoded `content` — never the long
/// `Reasoning` block.
fn live_status<'a>(h: &'a AgentHandle, groups: &'a [TraceGroup]) -> LiveStatus<'a> {
    if !h.current_tools.is_empty() {
        LiveStatus::Tools
    } else if let Some(activity) = &h.activity {
        LiveStatus::Activity(activity.as_str())
    } else if let Some(narration) = groups
        .last()
        .map(|g| g.narration.as_str())
        .filter(|n| !n.is_empty())
    {
        LiveStatus::Narration(narration)
    } else {
        LiveStatus::Thinking
    }
}

/// Render the "thinking…" placeholder shown when an agent is alive and
/// working but has no currently-executing tool and no live activity phase
/// (the pre-first-tool LLM reasoning window). Static icon + text, styled to
/// match the surrounding activity label (small accent).
fn render_thinking_indicator() -> Element<'static, RunningMessage> {
    row![
        lucide::loader_circle::<iced::Theme, iced::Renderer>()
            .size(12)
            .color(theme::ACCENT),
        text("thinking…").size(11).color(theme::ACCENT),
    ]
    .spacing(4)
    .align_y(Alignment::Center)
    .into()
}

/// Render one tool block: the tool name (bold primary) and its comma-separated
/// key-value pairs (regular secondary) on a single line — `tool args`. The
/// pairs portion is length-capped ([`MAX_TOOL_PAIRS_LINE_CHARS`]) so the short
/// tool name plus the args fit on one row without crowding the right-aligned
/// metrics. The whole block is the hover target of a tooltip showing the FULL
/// untruncated argument values.
fn render_tool_block(
    tool: &crate::agent::registry::RunningTool,
) -> Element<'static, RunningMessage> {
    let mut line = Row::new().spacing(4).align_y(Alignment::Center).push(
        text(tool.name.clone())
            .size(11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY),
    );
    if !tool.args.is_empty() {
        line = line.push(
            text(render_tool_pairs_line(&tool.args))
                .size(11)
                .color(theme::TEXT_SECONDARY),
        );
    }
    tooltip(line, render_tool_tooltip(tool), tooltip::Position::Top)
        .gap(4)
        .style(theme::tooltip_style)
        .into()
}

/// Render the row's key-value pairs line: `name: value` pairs, comma-
/// separated, each value collapsed to a single line and truncated at a
/// word/path-delimiter boundary to [`MAX_TOOL_VALUE_CHARS`] chars. The whole
/// line is capped at [`MAX_TOOL_PAIRS_LINE_CHARS`] chars, cut at a pair
/// boundary with "…".
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
        // Even the first pair alone does not fit — truncate at a delimiter
        // boundary so the cut lands at a sensible wrap point.
        truncate_at_boundary(&rendered[0], MAX_TOOL_PAIRS_LINE_CHARS)
    } else {
        format!("{out}…")
    }
}

/// Single-line display form of a value: control characters (newlines, tabs)
/// collapsed to spaces, then truncated to [`MAX_TOOL_VALUE_CHARS`] chars at a
/// word/path-delimiter boundary with "…" when cut.
fn value_display(value: &str) -> String {
    let single_line: String = value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    truncate_at_boundary(&single_line, MAX_TOOL_VALUE_CHARS)
}

/// Render the hover tooltip: the tool name header (bold white) followed by
/// every argument pair on its own line, sorted by FULL value length ascending
/// (stable, so equal-length pairs keep registration order) — the shortest
/// pairs sit at the top and stay visible even when the longest values extend
/// beyond the viewport. Values are full and untruncated, including secrets.
fn render_tool_tooltip(
    tool: &crate::agent::registry::RunningTool,
) -> Element<'static, RunningMessage> {
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

/// Render a one-line metric tooltip body in a narrow, styled container: the
/// explanatory text (metric label plus, where useful, the precise value). Kept
/// narrower than [`MAX_TOOL_TOOLTIP_WIDTH`] because a metric tooltip holds a
/// single short line, not multi-arg tool calls.
fn render_metric_tooltip(label: String) -> Element<'static, RunningMessage> {
    container(
        text(label)
            .size(11)
            .color(theme::TEXT_PRIMARY)
            .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
    )
    .max_width(MAX_METRIC_TOOLTIP_WIDTH)
    .into()
}

/// Render a compact non-agent LLM call row: zap marker + purpose + elapsed.
/// No workspace name (the section header carries it); the elapsed time uses
/// the same brighter tone as agent cards. The purpose is the static
/// human-readable label — raw kind names never leak.
fn render_call_row(call: &CallRow) -> Element<'static, RunningMessage> {
    let h = &call.handle;
    let elapsed = format_elapsed(h.started_at);
    let purpose = crate::agent::registry::call_kind_label(h.kind);

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
    use crate::agent::registry::AgentHandle;

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
            generation: 0,
            current_tools: Vec::new(),
            activity: None,
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
        // group_title preserves (name, id); render_group swaps which is emphasized.
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

    // ── Trace-group derivation ──────────────────────────────────────

    fn assistant_tool_call(narration: &str, calls: &[(&str, serde_json::Value)]) -> ChatMessage {
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
        ChatMessage::assistant(
            serde_json::json!({ "content": content, "tool_calls": calls_json }).to_string(),
        )
    }

    #[test]
    fn empty_and_non_tool_history_produce_no_groups() {
        assert!(derive_trace_groups(&[]).is_empty());
        // A content-only assistant message (final answer) is NOT a group.
        let history = vec![
            ChatMessage::user("hello\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>"),
            ChatMessage::assistant("Final answer here."),
        ];
        assert!(derive_trace_groups(&history).is_empty());
    }

    #[test]
    fn narration_starts_and_boundary_groups() {
        let history = vec![
            assistant_tool_call(
                "Reading the file",
                &[("read", serde_json::json!({"path": "a.rs"}))],
            ),
            assistant_tool_call(
                "Editing it",
                &[("edit", serde_json::json!({"path": "b.rs"}))],
            ),
        ];
        let groups = derive_trace_groups(&history);
        assert_eq!(groups.len(), 2, "new narration starts a new group");
        assert_eq!(groups[0].narration, "Reading the file");
        assert_eq!(groups[1].narration, "Editing it");
    }

    #[test]
    fn consecutive_no_narration_rounds_accumulate() {
        let history = vec![
            assistant_tool_call("", &[("read", serde_json::json!({"path": "a.rs"}))]),
            assistant_tool_call("", &[("list", serde_json::json!({"path": "."}))]),
        ];
        let groups = derive_trace_groups(&history);
        assert_eq!(groups.len(), 1, "no narration → same group");
        assert_eq!(groups[0].rounds.len(), 2);
        assert!(groups[0].narration.is_empty());
    }

    #[test]
    fn real_user_turn_resets_group() {
        let history = vec![
            assistant_tool_call(
                "Narration",
                &[("read", serde_json::json!({"path": "a.rs"}))],
            ),
            ChatMessage::user("ok now fix it\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>"),
            assistant_tool_call("", &[("edit", serde_json::json!({"path": "b.rs"}))]),
        ];
        let groups = derive_trace_groups(&history);
        assert_eq!(groups.len(), 2, "user turn closes the group");
        assert_eq!(groups[0].narration, "Narration");
        // The post-reset empty-narration round starts a fresh group.
        assert_eq!(groups[1].rounds.len(), 1);
    }

    #[test]
    fn injected_image_user_message_does_not_reset_group() {
        let history = vec![
            assistant_tool_call("", &[("read", serde_json::json!({"path": "a.rs"}))]),
            ChatMessage::user(crate::util::injected_image_user_message(
                "data:image/png;base64,xxx",
            )),
            assistant_tool_call("", &[("read", serde_json::json!({"path": "a.rs"}))]),
        ];
        let groups = derive_trace_groups(&history);
        assert_eq!(groups.len(), 1, "synthetic image stays in the same group");
        assert_eq!(groups[0].rounds.len(), 2);
    }

    #[test]
    fn collapsed_calls_line_counts_and_orders() {
        let rounds = vec![
            vec![
                RunningTool::from_tool_call(&crate::ToolCall {
                    id: "1".into(),
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path": "a.rs"}),
                }),
                RunningTool::from_tool_call(&crate::ToolCall {
                    id: "2".into(),
                    name: "list_files".into(),
                    arguments: serde_json::json!({"path": "."}),
                }),
            ],
            vec![RunningTool::from_tool_call(&crate::ToolCall {
                id: "3".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "b.rs"}),
            })],
        ];
        assert_eq!(
            collapsed_calls_line(&rounds),
            "read file x2, list files",
            "underscores become spaces; `xN` counts collapsed repetitions; first-appearance order"
        );
    }

    #[test]
    fn live_status_selects_tools_activity_narration_or_thinking() {
        // A running tool takes precedence over activity/narration.
        let mut tools = agent_handle("a", "engineer", None, "ws1", None);
        tools.current_tools = vec![RunningTool::from_tool_call(&crate::ToolCall {
            id: "1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        })];
        assert_eq!(live_status(&tools, &[]), LiveStatus::Tools);

        // No tool, an activity phase label when present.
        let mut activity = agent_handle("a", "engineer", None, "ws1", None);
        activity.activity = Some("analyzing".to_string());
        assert_eq!(
            live_status(&activity, &[]),
            LiveStatus::Activity("analyzing")
        );

        // Idle + a non-empty narration → the current/latest group's narration.
        let idle = agent_handle("a", "engineer", None, "ws1", None);
        let groups = derive_trace_groups(&[assistant_tool_call(
            "Reading the file",
            &[("read", serde_json::json!({"path": "a.rs"}))],
        )]);
        assert_eq!(
            live_status(&idle, &groups),
            LiveStatus::Narration("Reading the file")
        );

        // Multi-group transcript: the LAST group's narration is shown.
        let groups = derive_trace_groups(&[
            assistant_tool_call(
                "Editing it",
                &[("edit", serde_json::json!({"path": "b.rs"}))],
            ),
            assistant_tool_call(
                "Reading again",
                &[("read", serde_json::json!({"path": "c.rs"}))],
            ),
        ]);
        assert_eq!(
            live_status(&idle, &groups),
            LiveStatus::Narration("Reading again")
        );

        // Idle + no narration → the "thinking…" placeholder.
        assert_eq!(live_status(&idle, &[]), LiveStatus::Thinking);
    }

    #[test]
    fn prune_expanded_drops_stale_keys() {
        let mut expanded: HashSet<(String, u64)> =
            HashSet::from([("a".to_string(), 0), ("b".to_string(), 2)]);
        let agents = vec![agent_handle("a", "analyst", None, "ws1", None)];
        prune_expanded(&mut expanded, &agents);
        assert_eq!(expanded.len(), 1);
        assert!(
            expanded.contains(&("a".to_string(), 0)),
            "live agent keeps its expanded key"
        );
    }

    // ── Page truncation (boundary-aware) ─────────────────────────────

    #[test]
    fn truncation_constants_are_30_percent_larger() {
        assert_eq!(MAX_GROUP_LABEL_CHARS, 104);
        assert_eq!(MAX_TOOL_VALUE_CHARS, 26);
        assert_eq!(MAX_TOOL_PAIRS_LINE_CHARS, 104);
        assert_eq!(MAX_NARRATION_CHARS, 104);
    }

    #[test]
    fn truncate_at_boundary_cuts_at_word_and_path_delimiters() {
        // Cut at a word boundary rather than mid-word.
        assert_eq!(truncate_at_boundary("abc def ghi", 5), "abc…");
        // When the cap lands exactly before a delimiter, keep the full words.
        assert_eq!(truncate_at_boundary("abc def ghi", 7), "abc def…");
        // Paths break at a `/` segment boundary.
        assert_eq!(
            truncate_at_boundary("/Users/egordezic/Desktop/foo.rs", 26),
            "/Users/egordezic/Desktop/…"
        );
        // An unbroken token falls back to a hard char cut at the cap.
        assert_eq!(truncate_at_boundary("aaaaaaaa", 4), "aaaa…");
        // Under the cap: unchanged.
        assert_eq!(truncate_at_boundary("short", 10), "short");
    }

    #[test]
    fn value_display_truncates_at_delimiter_boundary() {
        // Path value cut at a `/`, content within the cap, trailing "…".
        assert_eq!(
            value_display("/Users/egordezic/Desktop/foo.rs"),
            "/Users/egordezic/Desktop/…"
        );
        // Under the cap: unchanged.
        assert_eq!(value_display("a.rs"), "a.rs");
        // Control chars collapse to spaces.
        assert_eq!(value_display("a\nb"), "a b");
        // Unbroken long token hard-cuts at [`MAX_TOOL_VALUE_CHARS`].
        assert_eq!(
            value_display(&"a".repeat(40)),
            format!("{}…", "a".repeat(MAX_TOOL_VALUE_CHARS))
        );
    }

    #[test]
    fn tool_pairs_line_truncates_at_pair_boundary() {
        // Under the line cap: returned verbatim.
        let short = vec![("path".to_string(), "a.rs".to_string())];
        assert_eq!(render_tool_pairs_line(&short), "path: a.rs");

        // Over the line cap: whole pairs kept while they fit, then a trailing
        // "…" marks the cut; the excluded final pair never appears.
        let pairs = vec![
            (
                "path".to_string(),
                "/Users/egordezic/Desktop/project/foo.rs".to_string(),
            ),
            (
                "offset".to_string(),
                "/Users/egordezic/Desktop/project/bar.rs".to_string(),
            ),
            (
                "limit".to_string(),
                "/Users/egordezic/Desktop/project/baz.rs".to_string(),
            ),
            (
                "query".to_string(),
                "/Users/egordezic/Desktop/project/qux.rs".to_string(),
            ),
        ];
        let line = render_tool_pairs_line(&pairs);
        assert_eq!(line.chars().count(), MAX_TOOL_PAIRS_LINE_CHARS);
        assert!(line.ends_with('…'));
        assert!(line.contains("limit"));
        assert!(!line.contains("query"), "excluded pair must not appear");
    }

    #[test]
    fn tool_pairs_line_fallback_truncates_first_pair_at_boundary() {
        // A single pair longer than the line cap: when the first pair contains
        // no delimiter it is hard-cut exactly at the cap (with "…"), never left
        // overflowing the line.
        let pairs = vec![("k".repeat(120), "v".to_string())];
        let line = render_tool_pairs_line(&pairs);
        assert_eq!(line, format!("{}…", "k".repeat(MAX_TOOL_PAIRS_LINE_CHARS)));
    }
}
