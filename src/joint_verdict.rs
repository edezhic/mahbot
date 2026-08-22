//! Joint verdict comments for pipeline stages (analysis, review, QA).
//!
//! Replaces the per-agent verdict comments with ONE comment per round —
//! written even on fully clean rounds so the audit trail is uniform. The
//! merge backbone is the shared LLM grouping core ([`crate::consensus`]):
//! a progress-preserving repair synthesis pass groups the agents' exact issue
//! statements — accepted groups freeze, repair rounds only touch the
//! remainder — contradiction groups carry a `— DISPUTED` marker on their
//! heading. The DISPUTED cross-reference appears in the ungrouped section
//! only when a member contradicts a frozen group (contradiction:true — never
//! for solo findings). Per-agent attribution (brackets, "Agent N:" /
//! "[blocker]" prefixes) and free-form critiques are stripped from comments —
//! scores + issues are persisted in the verdict store instead. The Analysis
//! stage shares this renderer, so analyst critiques are dropped too (the
//! shared `Verdict` schema carries score + issues only — critiques were never
//! persisted anywhere).
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

pub(crate) const DEFAULT_REVIEW_COUNT_LOW_CHURN: i64 = 500;
pub(crate) const DEFAULT_REVIEW_COUNT_HIGH_CHURN: i64 = 2000;

/// Maximum tolerated bounces before the ticket fails (the 11th bounce fails).
pub(crate) const MAX_BOUNCES: usize = 10;

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
    /// Pass threshold: the clean-round summary only claims "passed clean" when
    /// every valid verdict clears it (a sub-threshold verdict bounces the round
    /// even with an empty issues list).
    pub threshold: u8,
}

impl JointRound<'_> {
    /// Number of valid verdicts (responding agents with parseable verdicts).
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
        crate::prompt::load_prompt("synthesis/synthesis.md"),
        crate::prompt::load_prompt("synthesis/grouping_contradictions.md"),
    );
    // Scores are deliberately NOT included: grouping needs issue text only —
    // scores are persisted in the verdict store, never in the comment.
    let material = crate::consensus::numbered_items_material(&issues_by_agent(round));
    let user = format!(
        "{}\n\nStage: {}\nAgent issues (id-numbered):\n{}",
        crate::prompt::load_prompt("synthesis/synthesis_input.md"),
        round.stage,
        material,
    );
    let model = crate::config::CONFIG.role_model(role);
    let routing = crate::config::CONFIG.model_routing(&model);
    ChatRequest {
        messages: vec![ChatMessage::system(&system), ChatMessage::user(&user)],
        tools: None,
        model,
        max_tokens: Some(PIPELINE_GROUPING_MAX_TOKENS),
        reasoning_effort: Some(
            crate::role::role_info(&role)
                .default_reasoning_effort
                .to_string(),
        ),
        provider_order: routing.provider_order,
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
///
/// `ticket_id` attaches the synthesis LLM call to the ticket's group in the
/// Running Agents view (the ticket's own work — joint-verdict synthesis of a
/// review/QA/analysis round); `ticket_title` is the group header label for
/// that ticket (purely presentational — the header keeps the ticket name even
/// when the round's agents have already deregistered and only this synthesis
/// call remains in the group).
pub(crate) async fn build_joint_comment(
    round: &JointRound<'_>,
    role: Role,
    ws: &Workspace,
    ticket_id: &str,
    ticket_title: &str,
) -> String {
    let items = issues_by_agent(round);
    let outcome = run_synthesis(round, role, ws, ticket_id, ticket_title).await;
    render_joint_comment(round, &outcome, &crate::consensus::ItemTable::new(&items))
}

/// Run the repair-mode synthesis pass through the shared consensus core
/// (1 full call + up to N-1 repair rounds; frozen groups; per-group
/// acceptance; deterministic remainder placement; narrowed fail-open).
pub(crate) async fn run_synthesis(
    round: &JointRound<'_>,
    role: Role,
    ws: &Workspace,
    ticket_id: &str,
    ticket_title: &str,
) -> crate::consensus::RepairOutcome {
    let request = synthesis_request(round, role, ws);
    let items = issues_by_agent(round);
    crate::consensus::run_grouping_repair(
        ws,
        "synthesis",
        request,
        &items,
        Some(crate::registry::ParentKey::Ticket(ticket_id.to_string())),
        Some(ticket_title.to_string()),
    )
    .await
}

// ── Joint comment rendering ─────────────────────────────────────────────

/// Render the joint comment for a round given the repair-mode synthesis
/// outcome.
///
/// Structure: optional stage header (analysis-only — verifier rounds pass an
/// empty header), groups of issue statements (frozen by the repair protocol
/// when synthesis succeeded, raw member dump otherwise; contradiction groups
/// carry a `— DISPUTED` marker), the code-computed
/// ungrouped remainder in a deterministic trailing section (DISPUTED
/// cross-references for items that flag a contradiction against a frozen
/// group), the first-accepted LLM summary prose (or an explicit marker), and a
/// raw-dump appendix for failed agents. Per-agent attribution (brackets,
/// "Agent N:" / "[blocker]" prefixes) and free-form critiques are noise and
/// are not rendered — scores + issues are already persisted in the verdict
/// store.
#[must_use]
pub(crate) fn render_joint_comment(
    round: &JointRound<'_>,
    outcome: &crate::consensus::RepairOutcome,
    table: &crate::consensus::ItemTable<'_>,
) -> String {
    let mut out = String::new();

    // Code-computed stage summary (analysis only — verifier rounds pass an
    // empty header).
    if !round.header.is_empty() {
        out.push_str(&round.header);
    }

    // Issues — grouped by the LLM when available.
    let has_issues = table.len() > 0;
    if has_issues {
        match outcome {
            crate::consensus::RepairOutcome::Repaired { output, references } => {
                for group in &output.groups {
                    let _ = write!(out, "\n\n**{}**", group.heading);
                    if group.contradiction {
                        out.push_str(" — DISPUTED");
                    }
                    for member in &group.members {
                        let _ = write!(out, "\n- {}", member_text(table, member));
                    }
                }
                out.push_str(&crate::consensus::render_ungrouped_section(
                    output,
                    references,
                    |member, disputed| member_text(table, member) + disputed,
                ));
            }
            crate::consensus::RepairOutcome::Fallback => {
                // Deterministic fail-open: raw per-agent issue dump + marker
                // (global flat id order = (agent, item) order).
                out.push_str("\n\n**Issues**");
                for id in 0..table.len() {
                    if let Some((_, text)) = table.resolve(id) {
                        let _ = write!(out, "\n- {text}");
                    }
                }
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

    // A removed stage header (verifier rounds) would leave the first section's
    // separator as leading blank lines — strip them.
    crate::util::truncate_sandwich(
        &crate::util::scrub_credentials(out.trim_start_matches('\n')),
        crate::util::FAILURE_DETAIL_CAP,
        "joint verdict comment",
    )
}

/// Resolve a grouped member's issue text via the item table (no text
/// equality). Unknown ids (defensive) render with an explicit marker.
fn member_text(
    table: &crate::consensus::ItemTable<'_>,
    member: &crate::consensus::GroupingMember,
) -> String {
    table.resolve(member.id).map_or_else(
        || format!("<unknown item id {}>", member.id),
        |(_, text)| text.to_string(),
    )
}

// ── Calibrated dynamic agent counts ─────────────────────────────────────

/// Compute the reviewer-count base from total working-tree churn (added +
/// deleted lines, including lines of new files): 2 for churn ≤ low, 4 for
/// churn > high (strict — exactly `high` stays 3), 3 otherwise. Never 1.
#[must_use]
pub(crate) fn review_base_from_signals(total_churn: i64, low_churn: i64, high_churn: i64) -> usize {
    if total_churn <= low_churn {
        2
    } else if total_churn > high_churn {
        4
    } else {
        3
    }
}

/// Apply the P0 floor: priority-0 tickets never get 2 reviewers (floor 3).
/// Bounces do not change the count.
#[must_use]
pub(crate) fn review_agent_count(base: usize, priority: i64) -> usize {
    if priority == 0 { base.max(3) } else { base }
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
