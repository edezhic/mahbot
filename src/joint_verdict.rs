//! Joint verdict comments for pipeline stages (analysis, review, QA).
//!
//! Replaces the per-agent verdict comments with ONE comment per round —
//! written even on fully clean rounds so the audit trail is uniform. The
//! merge backbone is the shared LLM grouping core ([`crate::consensus`]):
//! a progress-preserving repair synthesis pass groups the agents' exact issue
//! statements — accepted groups freeze, repair rounds only touch the
//! remainder — and every group renders with a code-computed `[n/N]` bracket
//! derived from distinct cited agent ids. The DISPUTED marker appears only
//! when a group carries a genuine contradiction (contradiction:true — never
//! for solo findings). The LLM never produces counts — brackets are pure
//! arithmetic here.
//!
//! Items are referenced by stable numeric ids (global flat numbering across
//! all agents); validation is strictly structural (id range, duplicate
//! placement, completeness, contradiction ≥2 agents) and termination
//! deterministically places every remaining item in the ungrouped section,
//! eventually falling back to a deterministic raw member dump with an
//! explicit marker when nothing ever freezes.

use std::fmt::Write as _;

use crate::{ChatMessage, ChatRequest, ChatRequestMeta, Role, Workspace};

// ── Hardcoded review-count calibration defaults (no config surface) ──────

pub(crate) const DEFAULT_REVIEW_COUNT_LOW_CHURN: u64 = 50;
pub(crate) const DEFAULT_REVIEW_COUNT_HIGH_CHURN: u64 = 400;

/// Maximum tolerated bounces before the ticket fails (the 6th bounce fails).
pub(crate) const MAX_BOUNCES: usize = 5;

// ── Round data ─────────────────────────────────────────────────────────

/// One valid (parsed) verdict from a parallel round, in agent order.
pub(crate) struct JointVerdict<'a> {
    pub agent_index: usize,
    pub verdict: &'a crate::Verdict,
}

/// One failed agent (no response / parse failure) with its rendered dump.
pub(crate) struct JointFailure {
    pub agent_index: usize,
    pub dump: String,
}

/// Everything the joint-comment renderer needs about a round.
pub(crate) struct JointRound<'a> {
    /// Stage name: "Analysis", "Review" or "QA" (comment role = stage name).
    pub stage: &'a str,
    /// Number of dispatched agents (N_gate — fail-closed gate denominator).
    pub dispatched: usize,
    /// Valid verdicts, one per responding agent.
    pub verdicts: Vec<JointVerdict<'a>>,
    /// Failed agents (no response / parse failure).
    pub failures: Vec<JointFailure>,
    /// Code-computed agreement summary line (counts/classification — never
    /// the LLM's numbers).
    pub header: String,
    /// Pass threshold: issues from verdicts below this render as blockers.
    pub threshold: u8,
}

impl JointRound<'_> {
    /// Number of valid verdicts (N — bracket denominator).
    #[must_use]
    pub fn n_valid(&self) -> usize {
        self.verdicts.len()
    }
}

// ── Per-agent issue lists (id universe) ─────────────────────────────────

/// Key the per-agent issue lists by the ORIGINAL dispatch index. A failed
/// agent's slot stays empty. Per-agent duplicates are NOT deduped: two
/// identical issues from one agent are two distinct item ids in the global
/// flat numbering, and the model places each exactly once.
#[must_use]
pub(crate) fn issues_by_agent(round: &JointRound<'_>) -> Vec<Vec<String>> {
    let mut by_agent: Vec<Vec<String>> = vec![Vec::new(); round.dispatched];
    for v in &round.verdicts {
        by_agent[v.agent_index].clone_from(&v.verdict.issues_detected);
    }
    by_agent
}

// ── Synthesis request ──────────────────────────────────────────────────

/// Max output tokens for the pipeline grouping pass: raised above the old
/// budget (8K) — probes truncated at 4K output on 36-issue rounds; 16K is the
/// floor for claim-length rounds.
const PIPELINE_GROUPING_MAX_TOKENS: u32 = 16_000;

/// Build the synthesis chat request for a stage role (the stage role's own
/// model, reasoning effort, and provider routing — no separate grouping
/// model). The shared contradiction package is appended to the system prompt;
/// the general workspace context is prepended by the consensus core. The
/// system prompt is byte-stable across rounds — repair-round schema selection
/// lives in the appended user-section instructions.
fn synthesis_request(round: &JointRound<'_>, role: Role, ws: &Workspace) -> ChatRequest {
    let system = format!(
        "{}\n\n{}",
        crate::prompt::load_prompt("synthesis.md"),
        crate::prompt::load_prompt("grouping_contradictions.md"),
    );
    let mut material = String::new();
    // Global flat item ids across ALL agents: ids 0..N are assigned in
    // (agent, issue) order and match the schema's `id` field — a faithful LLM
    // can copy an id straight into the JSON. Scores are deliberately NOT
    // included: they are not needed for grouping, and a faithful LLM echoing
    // one would be rejected (the renderer adds scores/severity from code).
    let mut id = 0usize;
    for (agent_idx, issues) in issues_by_agent(round).into_iter().enumerate() {
        if issues.is_empty() {
            continue;
        }
        let _ = std::fmt::write(&mut material, format_args!("Agent {agent_idx}:\n"));
        for issue in issues {
            let _ = writeln!(material, "- {id}: {issue}");
            id += 1;
        }
    }
    let user = format!(
        "{}\n\nStage: {}\nAgent issues (id-numbered):\n{}",
        crate::prompt::load_prompt("synthesis_input.md"),
        round.stage,
        material,
    );
    let model = crate::config::CONFIG.role_model(role);
    let routing = crate::config::CONFIG.model_routing(&model);
    ChatRequest {
        messages: vec![ChatMessage::system(&system), ChatMessage::user(&user)],
        tools: None,
        model,
        allow_image_parts: false,
        max_tokens: Some(PIPELINE_GROUPING_MAX_TOKENS),
        reasoning_effort: Some(crate::config::CONFIG.role_reasoning_effort(role)),
        provider_order: routing.provider_order,
        provider_allow_fallbacks: routing.allow_fallbacks,
        response_format_json_object: true,
        meta: Some(ChatRequestMeta {
            purpose: "synthesis",
            agent_id: format!("joint_verdict_{}", crate::generate_suffix()),
            role: role.as_str().to_string(),
            workspace: ws.name.clone(),
            ticket_id: None,
        }),
    }
}

/// Convenience: run the synthesis pass and render the joint comment.
pub(crate) async fn build_joint_comment(
    round: &JointRound<'_>,
    role: Role,
    ws: &Workspace,
) -> String {
    let items = issues_by_agent(round);
    let outcome = run_synthesis(round, role, ws).await;
    render_joint_comment(round, &outcome, &crate::consensus::ItemTable::new(&items))
}

/// Run the repair-mode synthesis pass through the shared consensus core
/// (1 full call + up to N-1 repair rounds; frozen groups; per-group
/// acceptance; deterministic remainder placement; narrowed fail-open).
#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn run_synthesis(
    round: &JointRound<'_>,
    role: Role,
    ws: &Workspace,
) -> crate::consensus::RepairOutcome {
    let request = synthesis_request(round, role, ws);
    let items = issues_by_agent(round);
    crate::consensus::run_grouping_repair(ws, "synthesis", request, &items).await
}

// ── Joint comment rendering ─────────────────────────────────────────────

/// Render the joint comment for a round given the repair-mode synthesis
/// outcome.
///
/// Structure: header (code-computed counts), groups with code-computed
/// brackets from distinct cited agent ids (frozen by the repair protocol when
/// synthesis succeeded, raw member dump otherwise), the code-computed
/// ungrouped remainder in a deterministic trailing section (DISPUTED
/// cross-references for items that flag a contradiction against a frozen
/// group), per-agent critiques, the first-accepted LLM summary prose (or an
/// explicit marker), and a raw-dump appendix for failed agents.
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn render_joint_comment(
    round: &JointRound<'_>,
    outcome: &crate::consensus::RepairOutcome,
    table: &crate::consensus::ItemTable<'_>,
) -> String {
    let mut out = String::new();
    let n_valid = round.n_valid();
    let _ = std::fmt::write(
        &mut out,
        format_args!(
            "## {} round — {n_valid}/{} valid verdicts",
            round.stage, round.dispatched
        ),
    );

    // Code-computed agreement counts (never the LLM's numbers).
    if !round.header.is_empty() {
        let _ = write!(out, "\n{}", round.header);
    }

    // Issues with brackets — grouped by the LLM when available.
    let has_issues = table.len() > 0;
    if has_issues {
        match outcome {
            crate::consensus::RepairOutcome::Repaired { output, references } => {
                for group in &output.groups {
                    let n = crate::consensus::distinct_agents(group, table).len();
                    let bracket = crate::consensus::bracket_label(n, n_valid, group.contradiction);
                    let _ = write!(out, "\n\n**{}** {bracket}", group.heading);
                    for member in &group.members {
                        let _ = write!(out, "\n- {}", render_member_line(round, member, table));
                    }
                }
                // Code-computed ungrouped remainder — deterministic trailing
                // section (every remaining item lands here exactly once).
                if !output.ungrouped.is_empty() {
                    out.push_str("\n\n**Ungrouped**");
                    for member in &output.ungrouped {
                        let mut line = render_member_line(round, member, table);
                        for reference in references.iter().filter(|r| r.member.id == member.id) {
                            if let Some(group) = output.groups.get(reference.group) {
                                let _ = write!(
                                    line,
                                    " [DISPUTED — contradicts group {} \"{}\"]",
                                    reference.group, group.heading
                                );
                            }
                        }
                        let _ = write!(out, "\n- {line}");
                    }
                }
            }
            crate::consensus::RepairOutcome::Fallback => {
                // Deterministic fail-open: raw per-agent issue dump + marker
                // (global flat id order = (agent, item) order).
                out.push_str("\n\n**Issues**");
                for id in 0..table.len() {
                    if let Some((agent, text)) = table.resolve(id) {
                        let _ = write!(out, "\n- Agent {agent}: {text}");
                    }
                }
            }
        }
    }

    // Critiques (per-agent, score from code).
    let has_critiques = round.verdicts.iter().any(|v| {
        v.verdict
            .critique
            .as_deref()
            .is_some_and(|c| !c.trim().is_empty())
    });
    if has_critiques {
        out.push_str("\n\n### Critiques");
        for v in &round.verdicts {
            if let Some(c) = v
                .verdict
                .critique
                .as_deref()
                .filter(|c| !c.trim().is_empty())
            {
                let _ = write!(
                    out,
                    "\n- Agent {} ({}/10): {}",
                    v.agent_index + 1,
                    v.verdict.score,
                    c
                );
            }
        }
    }

    // First-accepted LLM summary or explicit marker.
    match outcome {
        crate::consensus::RepairOutcome::Repaired { output, .. } => {
            out.push_str("\n\n### Summary");
            let summary = output.summary.trim();
            if summary.is_empty() {
                out.push_str("\nLLM summary unavailable — deterministic member render.");
            } else {
                let _ = write!(out, "\n{summary}");
            }
        }
        crate::consensus::RepairOutcome::Fallback => {
            if has_issues {
                out.push_str(
                    "\n\n### Summary\nLLM grouping unavailable — deterministic member dump only.",
                );
            } else {
                // No issues existed to merge — the synthesis pass was
                // deliberately skipped, so the summary must not imply it
                // failed. "Passed clean" additionally requires every valid
                // verdict to clear the round threshold: a sub-threshold
                // verdict with an empty issues list still bounces the round.
                let clean = round.failures.is_empty()
                    && round
                        .verdicts
                        .iter()
                        .all(|v| v.verdict.score >= round.threshold);
                let summary = if clean {
                    format!(
                        "\n\n### Summary\nNo issues found — all {} agents passed clean.",
                        round.n_valid()
                    )
                } else if round.n_valid() > 0 {
                    "\n\n### Summary\nNo issues found by the responding agents.".to_string()
                } else {
                    "\n\n### Summary\nNo issues to merge — no agents produced a verdict."
                        .to_string()
                };
                out.push_str(&summary);
            }
        }
    }

    // Raw-dump appendix for failed agents.
    if !round.failures.is_empty() {
        out.push_str("\n\n### Agent failures");
        for f in &round.failures {
            let _ = write!(out, "\n- Agent {}: {}", f.agent_index + 1, f.dump);
        }
    }

    crate::util::truncate_sandwich(
        &crate::util::scrub_credentials(&out),
        crate::util::FAILURE_DETAIL_CAP,
        "joint verdict comment",
    )
}

/// Render one member line: `Agent N: text` with a `[blocker]` prefix when the
/// cited agent's verdict scored below the round threshold (code-computed
/// severity — the LLM never produces it). The cited agent is resolved via the
/// item table — no text equality.
fn render_member_line(
    round: &JointRound<'_>,
    member: &crate::consensus::GroupingMember,
    table: &crate::consensus::ItemTable<'_>,
) -> String {
    let Some((agent, text)) = table.resolve(member.id) else {
        return format!("Agent ?: <unknown item id {}>", member.id);
    };
    let blocker = round
        .verdicts
        .iter()
        .any(|v| v.agent_index == agent && v.verdict.score < round.threshold);
    let severity = if blocker { "[blocker] " } else { "" };
    format!("{severity}Agent {agent}: {text}")
}

// ── Calibrated dynamic agent counts ─────────────────────────────────────

/// Compute the reviewer-count base from the working-tree churn signals:
/// 2 reviewers for low churn (with zero added files), 4 for high churn or
/// any added file, 3 otherwise. Never 1.
#[must_use]
pub(crate) fn review_base_from_signals(
    total_churn: i64,
    max_per_file_churn: i64,
    added_files: usize,
    low_churn: i64,
    high_churn: i64,
) -> usize {
    if total_churn < low_churn && max_per_file_churn < low_churn && added_files == 0 {
        2
    } else if total_churn >= high_churn || max_per_file_churn >= high_churn || added_files > 0 {
        4
    } else {
        3
    }
}

/// Compute the reviewer count for a review round from a base (computed from
/// the ORIGINAL dispatch signals and frozen, or re-derived). Bounced tickets
/// get a flat +1 (capped at 4) — never re-computed upward after bounces,
/// preventing escalation loops. Priority-0 tickets never get 2 (floor 3).
#[must_use]
pub(crate) fn review_agent_count(base: usize, priority: i64, bounced_before: bool) -> usize {
    let count = if bounced_before { base + 1 } else { base };
    let count = count.min(4);
    if priority == 0 { count.max(3) } else { count }
}

/// Whether a second batch of analysts is needed: the base dispatch (the
/// actually-dispatched count, passed by the caller) all produced verdicts AND
/// every one flagged blockers (score below the analyst pass threshold —
/// unanimous blocker escalation).
#[must_use]
pub(crate) fn analysis_escalation_needed(
    results: &[crate::management::ParallelVerdict],
    dispatched: usize,
) -> bool {
    results.len() == dispatched
        && results.iter().all(|r| {
            matches!(
                r,
                crate::management::ParallelVerdict::Verdict(v)
                    if v.score < crate::management::ANALYST_PASS_THRESHOLD
            )
        })
}

#[cfg(test)]
#[path = "joint_verdict_tests.rs"]
mod tests;
