//! "Running Agents" dashboard page — a live view of every currently-running
//! agent and in-flight non-agent LLM work, grouped by DIRECT PARENT
//! INVOCATION (ticket / analyze round / research run / workspace singleton /
//! unattributable orchestrator calls).
//!
//! The view is a running-only window: it reads the in-memory registries
//! ([`crate::registry::AGENT_REGISTRY`] and
//! [`crate::call_registry::NON_AGENT_CALLS`]) at render time — no database
//! reads, no schema changes, no new subscriptions, no history retained
//! between ticks. The existing 1-second UI tick re-renders the page, so the
//! view refreshes at that cadence for free.
//!
//! Truthfulness rules:
//! - An agent not executing a tool (in an LLM call, between rounds) shows no
//!   tool badge — absence is honest.
//! - Parallel tool execution is represented honestly: every tool that
//!   actually started executing appears as its own badge; tools that never
//!   execute (unknown tool, pre-flight cancellation) never show.
//! - The instrumentation is purely observational — it never affects
//!   shutdown/drain logic and gains no cancellation semantics.

use crate::call_registry::NonAgentCallHandle;
use crate::gui::theme;
use crate::gui::widgets;
use crate::registry::{AgentHandle, ParentKey};
use chrono::{DateTime, Utc};

use iced::widget::{Column, Row, Space, column, container, pick_list, row, scrollable, text};
use iced::{Alignment, Element, Length, Padding};

use iced_fonts::lucide;

use super::WorkspaceInfo;

/// Workspace filter selection; `None` = "All workspaces". Persists while the
/// page is open (the running data itself is not retained between ticks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunningMessage {
    FilterWorkspace(Option<String>),
}

/// Page state — workspace filter selection plus the filter options cache.
///
/// The filter options are recomputed on navigation and on each 1-second tick
/// (in-memory reads only) so the pick_list can borrow them; the running data
/// itself is always read live at render time and never retained between ticks.
#[derive(Debug, Default)]
pub struct RunningState {
    filter: Option<String>,
    workspace_options: Vec<widgets::PickOption>,
}

impl RunningState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, msg: RunningMessage) -> iced::Task<RunningMessage> {
        match msg {
            RunningMessage::FilterWorkspace(sel) => self.filter = sel,
        }
        iced::Task::none()
    }

    /// Recompute the workspace filter options from the live registries and the
    /// dashboard's registered workspace set. Called on navigation to the page
    /// and on each 1-second tick while the page is active (bounded freshness —
    /// the same cadence that refreshes paused state).
    pub fn refresh(&mut self, workspaces: &std::collections::HashMap<String, WorkspaceInfo>) {
        let agents = crate::registry::AGENT_REGISTRY.list();
        let calls = crate::call_registry::NON_AGENT_CALLS.list();
        let mut ws_names: Vec<String> = Vec::new();
        let mut push_ws = |name: &str| {
            if name.is_empty() || ws_names.iter().any(|n| n == name) {
                return;
            }
            ws_names.push(name.to_string());
        };
        for a in &agents {
            push_ws(&a.workspace_name);
        }
        for c in &calls {
            push_ws(&c.workspace);
        }
        for name in workspaces.keys() {
            push_ws(name);
        }
        ws_names.sort();
        let mut options: Vec<widgets::PickOption> = Vec::with_capacity(ws_names.len() + 1);
        options.push(widgets::PickOption {
            value: String::new(),
            label: "All workspaces".to_string(),
        });
        for name in &ws_names {
            options.push(widgets::PickOption {
                value: name.clone(),
                label: name.clone(),
            });
        }
        self.workspace_options = options;
        // Keep the selection if it still exists; otherwise fall back to All.
        if let Some(f) = &self.filter
            && !self.workspace_options.iter().any(|o| o.value == *f)
        {
            self.filter = None;
        }
    }

    /// Render the live running-agents page.
    ///
    /// `workspaces` is the dashboard's registered-workspace map (name →
    /// info, incl. paused state) — used for paused pills and to distinguish
    /// registered workspaces from personal/ephemeral ones.
    pub fn view<'a>(
        &'a self,
        workspaces: &'a std::collections::HashMap<String, WorkspaceInfo>,
    ) -> Element<'a, RunningMessage> {
        let agents = crate::registry::AGENT_REGISTRY.list();
        let calls = crate::call_registry::NON_AGENT_CALLS.list();

        // Build the ordered display groups.
        let groups = build_groups(&agents, &calls);

        // Filter by workspace selection.
        let mut filtered: Vec<DisplayGroup> = groups
            .into_iter()
            .filter(|g| match &self.filter {
                None => true,
                Some(ws) => g.workspace == *ws,
            })
            .collect();

        // Stable sort: tickets, analyze rounds, research runs, singletons,
        // unattributed (the DisplayGroup::sort_key already encodes this).
        filtered.sort_by_key(DisplayGroup::sort_key);

        // The pick_list must SHOW the "All workspaces" label when no filter is
        // active — with `selected = None` iced renders its blank placeholder,
        // so select the empty-value option (label "All workspaces") instead.
        let selected = match &self.filter {
            Some(f) => self
                .workspace_options
                .iter()
                .find(|o| o.value == *f)
                .cloned(),
            None => self
                .workspace_options
                .iter()
                .find(|o| o.value.is_empty())
                .cloned(),
        };

        let header = row![
            text("Running Agents").size(18).color(theme::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            pick_list(self.workspace_options.as_slice(), selected, |opt| {
                RunningMessage::FilterWorkspace(
                    (!opt.value.is_empty()).then_some(opt.value.clone()),
                )
            },)
            .style(widgets::pick_list_style)
            .padding([4, 8])
            .width(Length::Fixed(220.0)),
        ]
        .spacing(12)
        .align_y(Alignment::Center);

        let mut body = Column::new().spacing(12);
        if filtered.is_empty() {
            body = body.push(
                container(
                    text("Nothing is currently running.")
                        .size(14)
                        .color(theme::TEXT_MUTED),
                )
                .width(Length::Fill)
                .center_x(Length::Fill)
                .padding(24),
            );
        } else {
            for group in &mut filtered {
                body = body.push(render_group(group, workspaces));
            }
        }

        let content = column![
            header,
            scrollable(container(body).width(Length::Fill).padding([0, 4]))
                .height(Length::Fill)
                .style(theme::scrollbar_style),
        ]
        .spacing(16)
        .padding(Padding::from([16, 24]))
        .width(Length::Fill)
        .height(Length::Fill);

        content.into()
    }
}

// ── Display model ─────────────────────────────────────────────────────────

/// One running work item: either an agent card or a non-agent LLM call row.
#[derive(Debug, Clone)]
enum DisplayItem {
    Agent(AgentCard),
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

/// Ordered group kinds — the ticket's grouping order.
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
    items: Vec<DisplayItem>,
    /// True when this research group carries a run-lifetime orchestrator
    /// marker (rendered as a run-lifetime indicator, not a call row).
    run_lifetime: bool,
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
                items: Vec::new(),
                run_lifetime: false,
            });
            groups.len() - 1
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
            groups[idx].items.push(DisplayItem::Agent(AgentCard {
                handle: agent.clone(),
            }));
        } else {
            // Workspace singleton. Empty workspace name falls back to a
            // generic label.
            let ws_label = if workspace.is_empty() {
                "workspace"
            } else {
                &workspace
            };
            let idx = find_group(GroupKind::Singleton, ws_label, ws_label, &mut groups);
            groups[idx].items.push(DisplayItem::Agent(AgentCard {
                handle: agent.clone(),
            }));
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
            let ws_label = if workspace.is_empty() {
                "workspace"
            } else {
                &workspace
            };
            let idx = find_group(GroupKind::Unattributed, ws_label, ws_label, &mut groups);
            groups[idx].items.push(DisplayItem::Call(CallRow {
                handle: call.clone(),
            }));
        }
    }

    groups
}

// ── Rendering ─────────────────────────────────────────────────────────────

/// Render one group: header (title + workspace + paused pill) then cards.
///
/// The returned element owns all rendered content (text widgets take owned
/// Strings), so its lifetime is independent of the `group` borrow — only the
/// `workspaces` lookup borrows from the caller.
fn render_group<'a>(
    group: &DisplayGroup,
    workspaces: &'a std::collections::HashMap<String, WorkspaceInfo>,
) -> Element<'a, RunningMessage> {
    // Paused pill: only for registered workspaces; personal/ephemeral
    // workspaces render with a fallback label and no pill.
    let (workspace_label, paused) = match workspaces.get(&group.workspace) {
        Some(info) if !group.workspace.is_empty() => (group.workspace.clone(), info.paused),
        _ => (fallback_workspace_label(&group.workspace), false),
    };

    // Singleton / Unattributed groups are keyed BY the workspace name, so the
    // resolved label (with the "(external)" marker when unregistered) IS the
    // title — never push a second workspace label beside it. Parent-keyed
    // groups (ticket / analyze round / research) show the workspace label next to
    // the title, with the round/run key included so concurrent rounds in one
    // workspace are visually distinguishable at the header level.
    let show_workspace_label =
        !matches!(group.kind, GroupKind::Singleton | GroupKind::Unattributed);
    let title = match &group.kind {
        GroupKind::Ticket => format!("Ticket {}", group.key),
        GroupKind::AnalyzeRound => format!("Analyze round {}", group.key),
        GroupKind::Research => format!("Research run {}", group.key),
        GroupKind::Singleton => workspace_label.clone(),
        GroupKind::Unattributed => format!("Other LLM work — {workspace_label}"),
    };

    let mut header_parts: Vec<Element<'_, RunningMessage>> =
        vec![text(title).size(15).color(theme::ACCENT).into()];
    if show_workspace_label && !group.workspace.is_empty() {
        header_parts.push(
            text(workspace_label)
                .size(12)
                .color(theme::TEXT_MUTED)
                .into(),
        );
    }
    if paused {
        header_parts.push(widgets::badge_pill(
            "Paused".to_string(),
            (theme::STATUS_WARNING, theme::TEXT_PRIMARY),
            11,
            [2, 8],
        ));
    }
    if group.run_lifetime {
        header_parts.push(
            row![
                lucide::loader_circle::<iced::Theme, iced::Renderer>()
                    .size(13)
                    .color(theme::ACCENT),
                text("run active").size(11).color(theme::ACCENT),
            ]
            .spacing(4)
            .align_y(Alignment::Center)
            .into(),
        );
    }

    let mut items = Column::new().spacing(6);
    for item in &group.items {
        match item {
            DisplayItem::Agent(card) => {
                items = items.push(render_agent_card(card, workspaces));
            }
            DisplayItem::Call(call) => {
                items = items.push(render_call_row(call, workspaces));
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

/// Fallback label for workspaces outside the dashboard's registered set
/// (personal user spaces, ephemeral run-scoped workspaces).
fn fallback_workspace_label(workspace: &str) -> String {
    if workspace.is_empty() {
        "workspace".to_string()
    } else {
        format!("{workspace} (external)")
    }
}

/// Resolve a workspace's display label for a card/row: the bare name for
/// registered workspaces, the fallback label (with the "(external)" marker)
/// for unregistered/ephemeral ones, and "workspace" for an empty name.
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

/// Render one running-agent card: role icon + label + workspace + current-tool
/// badges + elapsed since the agent started (derived at render time, honestly
/// shown as since-registration).
fn render_agent_card<'a>(
    card: &AgentCard,
    workspaces: &'a std::collections::HashMap<String, WorkspaceInfo>,
) -> Element<'a, RunningMessage> {
    let h = &card.handle;
    // Color resolution via the canonical string helper (handles derivative
    // names and falls back to muted grey for unknown roles); the icon still
    // needs a typed Role, so parse with the same Engineer fallback as before.
    let (fg, _bg) = theme::role_badge_color(&h.role);
    let role: crate::Role = h.role.parse().unwrap_or(crate::Role::Engineer);
    let icon = theme::role_icon(&role).size(20).color(fg);

    let workspace_label = workspace_label_for(&h.workspace_name, workspaces);

    let elapsed = format_elapsed(h.started_at);

    let mut first_row = Row::new().spacing(8).align_y(Alignment::Center);
    first_row = first_row.push(icon);
    first_row = first_row.push(
        text(h.label.clone())
            .size(13)
            .color(theme::TEXT_PRIMARY)
            .width(Length::FillPortion(3)),
    );
    first_row = first_row.push(
        text(workspace_label)
            .size(11)
            .color(theme::TEXT_MUTED)
            .width(Length::FillPortion(2)),
    );
    first_row = first_row.push(text(elapsed).size(11).color(theme::TEXT_MUTED));

    // Tool badges — only tools that actually started executing. Absence is
    // honest (agent in an LLM call / between rounds).
    let mut badge_row = Row::new().spacing(6);
    for tool in &h.current_tools {
        let tool_label = if tool.args.is_empty() {
            tool.name.clone()
        } else {
            format!("{} {}", tool.name, tool.args)
        };
        badge_row = badge_row.push(
            container(text(tool_label).size(11).color(theme::ACCENT))
                .padding([1, 6])
                .style(theme::pill_style(theme::ACCENT_DIM)),
        );
    }
    // Activity indicator — a non-tool LLM phase (verdict/summary extraction,
    // media transcription) running inside this agent. The agent card is the
    // single tracker for these calls (no separate call rows are ever created),
    // so the badge is what keeps an extracting/transcribing agent from looking
    // idle. Loader + label distinguishes it from tool badges.
    if let Some(activity) = &h.activity {
        badge_row = badge_row.push(
            container(
                row![
                    lucide::loader_circle::<iced::Theme, iced::Renderer>()
                        .size(11)
                        .color(theme::ACCENT),
                    text(activity.clone()).size(11).color(theme::ACCENT),
                ]
                .spacing(4)
                .align_y(Alignment::Center),
            )
            .padding([1, 6])
            .style(theme::pill_style(theme::ACCENT_DIM)),
        );
    }

    let mut col = Column::new().spacing(4);
    col = col.push(first_row);
    if !h.current_tools.is_empty() || h.activity.is_some() {
        col = col.push(badge_row);
    }

    container(col)
        .width(Length::Fill)
        .padding(8)
        .style(theme::container_bar)
        .into()
}

/// Render a compact non-agent LLM call row: zap marker + purpose + elapsed.
fn render_call_row<'a>(
    call: &CallRow,
    workspaces: &'a std::collections::HashMap<String, WorkspaceInfo>,
) -> Element<'a, RunningMessage> {
    let h = &call.handle;
    let workspace_label = workspace_label_for(&h.workspace, workspaces);
    let elapsed = format_elapsed(h.started_at);
    let purpose = call_purpose(h.kind);

    row![
        lucide::zap::<iced::Theme, iced::Renderer>()
            .size(16)
            .color(theme::ACCENT),
        text(purpose).size(12).color(theme::TEXT_SECONDARY),
        Space::new().width(Length::Fill),
        text(workspace_label).size(11).color(theme::TEXT_MUTED),
        text(elapsed).size(11).color(theme::TEXT_MUTED),
    ]
    .spacing(6)
    .align_y(Alignment::Center)
    .into()
}

/// Human-readable purpose label for a call kind (the registry stores the
/// same purpose string the call's `ChatRequestMeta` uses).
///
/// Note: `"research_orchestrator"` is deliberately absent — the orchestrator
/// always registers with `run_lifetime = true`, so it renders as the group's
/// run-lifetime indicator and never appears as a call row.
fn call_purpose(kind: &str) -> String {
    match kind {
        "consolidate" => "Analyze consolidation".to_string(),
        "synthesis" => "Ticket synthesis".to_string(),
        "synthesize" => "Research synthesis".to_string(),
        "decompose_merge" => "Research plan merge".to_string(),
        "gap_extract" => "Research gap extraction".to_string(),
        "abstain_check" => "Research answerability check".to_string(),
        "claim_annotate" => "Research claim annotation".to_string(),
        "confirm_links" => "Research link confirmation".to_string(),
        "research_wrap_up" => "Research wrap-up".to_string(),
        "media_transcription" => "Media transcription".to_string(),
        other => other.to_string(),
    }
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
            started_at: Utc::now(),
            label: role.to_string(),
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
        let mut groups = vec![
            DisplayGroup {
                kind: GroupKind::Unattributed,
                key: "ws".to_string(),
                workspace: "ws".to_string(),
                items: Vec::new(),
                run_lifetime: false,
            },
            DisplayGroup {
                kind: GroupKind::Ticket,
                key: "T1".to_string(),
                workspace: "ws".to_string(),
                items: Vec::new(),
                run_lifetime: false,
            },
            DisplayGroup {
                kind: GroupKind::Research,
                key: "r".to_string(),
                workspace: "ws".to_string(),
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
}
