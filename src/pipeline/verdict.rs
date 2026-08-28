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
use std::sync::Arc;

use crate::jobs::RowStatus;
use crate::pipeline::board::Ticket;
use crate::retry::RetryExhausted;
use crate::util::{panic_message, scrub_credentials};
use crate::{
    BlockerVerificationVerdict, ChatMessage, ChatRequest, ChatRequestMeta, Role, Verdict, Workspace,
};

use super::{
    FinalizeOutcome, TicketPhase, TransitionCtx, bounce_to_development, comment_and_transition,
    info, raw_response_dump_section, reset_phase_attempt,
};

// ── Hardcoded review-count calibration defaults (no config surface) ──────

pub(crate) const DEFAULT_REVIEW_COUNT_TINY_CHURN: i64 = 100;
pub(crate) const DEFAULT_REVIEW_COUNT_LOW_CHURN: i64 = 500;
pub(crate) const DEFAULT_REVIEW_COUNT_HIGH_CHURN: i64 = 2000;

// ── Round data ─────────────────────────────────────────────────────────

/// One valid (parsed) verdict from a parallel round, in agent order.
pub(crate) struct JointVerdict<'a> {
    pub agent_index: usize,
    pub verdict: &'a crate::Verdict,
}

/// One failed agent (no response / parse failure) with its rendered dump.
pub(crate) struct JointFailure {
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
    let derived = crate::agent::role_chat_params(role);
    ChatRequest {
        messages: vec![ChatMessage::system(&system), ChatMessage::user(&user)],
        tools: None,
        model: derived.model,
        // Override the default 32K budget with the 16K aggregation budget.
        max_tokens: Some(PIPELINE_GROUPING_MAX_TOKENS),
        reasoning_effort: derived.reasoning_effort,
        provider_order: derived.provider_order,
        meta: Some(ChatRequestMeta {
            purpose: "synthesis",
            agent_id: format!("verdict_{}", crate::generate_suffix()),
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
        Some(crate::agent::registry::ParentKey::Ticket(
            ticket_id.to_string(),
        )),
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
            let summary = output.summary.trim();
            if !summary.is_empty() {
                out.push_str("\n\n### Summary");
                let _ = write!(out, "\n{summary}");
            }
        }
        crate::consensus::RepairOutcome::Fallback => {
            if has_issues {
                // the grouping pass either failed or was
                // deliberately skipped (single-verdict verifier round) — both
                // reduce to the deterministic per-agent dump.
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
                let summary = if clean || round.n_valid() > 0 {
                    "\n\n### Summary\nNo issues found.".to_string()
                } else {
                    "\n\n### Summary\nNo issues to merge — no agent produced a verdict.".to_string()
                };
                out.push_str(&summary);
            }
        }
    }

    // Raw-dump appendix for failed agents.
    if !round.failures.is_empty() {
        out.push_str("\n\n### Plain verifier responses");
        for f in &round.failures {
            let _ = write!(out, "\n- {}\n", f.dump);
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
/// deleted lines, including lines of new files): 1 for churn < tiny, 2 for
/// churn ≤ low, 4 for churn > high (strict — exactly `high` stays 3), 3
/// otherwise.
#[must_use]
pub(crate) fn review_base_from_signals(
    total_churn: i64,
    tiny_churn: i64,
    low_churn: i64,
    high_churn: i64,
) -> usize {
    if total_churn < tiny_churn {
        1
    } else if total_churn <= low_churn {
        2
    } else if total_churn > high_churn {
        4
    } else {
        3
    }
}

/// Apply the P0 floor: priority-0 tickets never drop below 2 reviewers
/// (floor 2). Bounces do not change the count.
#[must_use]
pub(crate) fn review_agent_count(base: usize, priority: i64) -> usize {
    if priority == 0 { base.max(2) } else { base }
}

/// Human-readable stage name for a parallel-verdict role (used in the joint
/// comment title and the comment role).
#[must_use]
pub(crate) fn stage_name(role: Role) -> &'static str {
    match role {
        Role::Analyst => "Analysis",
        Role::Reviewer => "Review",
        Role::Qa => "QA",
        // Only the three parallel-verdict roles reach this function (all call
        // sites pass Analyst/Reviewer/Qa).
        _ => unreachable!("stage_name called with a non-verdict role"),
    }
}

/// Inverse of [`stage_name`]: resolve a stage-name comment role back to the
/// verdict role (used by the GUI to color joint-comment badges).
#[must_use]
pub(crate) fn stage_role(name: &str) -> Option<Role> {
    match name {
        "Analysis" => Some(Role::Analyst),
        "Review" => Some(Role::Reviewer),
        "QA" => Some(Role::Qa),
        _ => None,
    }
}

// ── Parallel-round data shapes & joint-comment builder ──────────────────
//
// These are the verdict-round types the renderer above consumes, so they
// live with it rather than in the orchestrator.

/// Result from a single parallel verifier agent.
#[derive(Clone)]
pub(crate) enum ParallelVerdict {
    /// Agent failed to produce any response (crashed, timed out, empty output).
    NoResponse(String),
    /// Agent produced a response but structured verdict extraction failed.
    ParseFailed(RetryExhausted),
    /// Agent produced a successfully-parsed verdict.
    Verdict(Verdict),
    /// Agent produced a blocker-verification verdict (analysis escalation).
    BlockerVerification(BlockerVerificationVerdict),
}

/// Structured-extraction behavior for a parallel round member.
#[derive(Clone)]
pub(crate) enum ExtractionMode {
    /// Standard score+issues verdict (analysis base, review, QA).
    ScoreVerdict,
    /// Blocker verification (analysis escalation).
    BlockerVerification { blockers: Arc<[String]> },
}

/// One roster slot of a ticket phase round.
#[derive(Clone)]
pub(crate) struct AgentSlot {
    /// Dispatch slot index (0-based; escalation continues at 3, 4).
    pub idx: i64,
    pub agent_id: String,
    /// FINAL per-agent rendered prompt (angle appended).
    pub task: String,
    pub status: RowStatus,
    /// Stored agents.outcome (tagged JSON) — set on replay of a done slot.
    pub outcome: Option<String>,
}

/// Map a panicked round member's [`tokio::task::JoinError`] to a contained
/// [`ParallelVerdict::NoResponse`] (round continues fail-open).
pub(crate) fn round_member_failed(e: tokio::task::JoinError) -> ParallelVerdict {
    let reason = scrub_credentials(&panic_message(&*e.into_panic()));
    tracing::warn!(%reason, "round member task failed");
    ParallelVerdict::NoResponse(reason)
}

/// Validate a verdict score is within [0, 10].
pub(crate) fn validate_verdict_score(v: &Verdict) -> Result<(), String> {
    if v.score <= 10 {
        Ok(())
    } else {
        Err(format!("verdict score {} out of range [0,10]", v.score))
    }
}

/// Validate a blocker-verification verdict.
pub(crate) fn validate_blocker_verification(
    v: &BlockerVerificationVerdict,
    blockers: &[String],
) -> Result<(), String> {
    if v.verdicts.is_empty() {
        return Err("blocker verification returned no verdicts".to_string());
    }
    if v.verdicts.len() != blockers.len() {
        return Err(format!(
            "blocker verification returned {} verdicts for {} blockers",
            v.verdicts.len(),
            blockers.len()
        ));
    }
    let mut seen = vec![false; blockers.len()];
    for item in &v.verdicts {
        if item.index >= blockers.len() {
            return Err(format!("blocker index {} out of range", item.index));
        }
        if seen[item.index] {
            return Err(format!("duplicate blocker index {}", item.index));
        }
        seen[item.index] = true;
        if item.reasoning.trim().is_empty() {
            return Err(format!("blocker {} missing reasoning", item.index));
        }
        if item.impact.trim().is_empty() {
            return Err(format!("blocker {} missing impact", item.index));
        }
    }
    Ok(())
}

/// Serialize a [`ParallelVerdict`] into the agents.outcome column.
#[must_use]
pub(crate) fn serialize_verdict_outcome(result: &ParallelVerdict) -> String {
    match result {
        ParallelVerdict::Verdict(v) => serde_json::json!({ "verdict": v }).to_string(),
        ParallelVerdict::NoResponse(reason) => {
            serde_json::json!({ "no_response": reason }).to_string()
        }
        ParallelVerdict::ParseFailed(f) => {
            serde_json::json!({ "parse_failed": raw_response_dump_section(f) }).to_string()
        }
        ParallelVerdict::BlockerVerification(v) => {
            serde_json::json!({ "blocker_verification": v }).to_string()
        }
    }
}

/// Reconstruct a [`ParallelVerdict`] from the agents.outcome column.
#[must_use]
pub(crate) fn deserialize_verdict_outcome(outcome: &str) -> ParallelVerdict {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(outcome) else {
        return ParallelVerdict::NoResponse("unreadable stored outcome".to_string());
    };
    if let Some(verdict) = v.get("verdict") {
        match serde_json::from_value(verdict.clone()) {
            Ok(v) => ParallelVerdict::Verdict(v),
            Err(_) => ParallelVerdict::NoResponse("unreadable stored verdict".to_string()),
        }
    } else if let Some(r) = v.get("no_response").and_then(serde_json::Value::as_str) {
        ParallelVerdict::NoResponse(r.to_string())
    } else if let Some(p) = v.get("parse_failed").and_then(serde_json::Value::as_str) {
        ParallelVerdict::NoResponse(p.to_string())
    } else if let Some(v) = v.get("blocker_verification") {
        match serde_json::from_value(v.clone()) {
            Ok(bv) => ParallelVerdict::BlockerVerification(bv),
            Err(_) => {
                ParallelVerdict::NoResponse("unreadable stored blocker verification".to_string())
            }
        }
    } else {
        ParallelVerdict::NoResponse("unrecognized stored outcome".to_string())
    }
}

/// Build the joint comment for a round: deterministic merge + a single LLM
/// synthesis pass.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn build_round_joint_comment(
    stage: &str,
    results: &[ParallelVerdict],
    threshold: u8,
    role: Role,
    header: &str,
    ws: &Workspace,
    ticket_id: &str,
    ticket_title: &str,
) -> String {
    let mut verdicts: Vec<JointVerdict<'_>> = Vec::new();
    let mut failures: Vec<JointFailure> = Vec::new();
    for (i, r) in results.iter().enumerate() {
        match r {
            ParallelVerdict::Verdict(v) => {
                verdicts.push(JointVerdict {
                    agent_index: i,
                    verdict: v,
                });
            }
            ParallelVerdict::NoResponse(reason) => {
                failures.push(JointFailure {
                    dump: reason.clone(),
                });
            }
            ParallelVerdict::ParseFailed(f) => {
                failures.push(JointFailure {
                    dump: scrub_credentials(&raw_response_dump_section(f)),
                });
            }
            ParallelVerdict::BlockerVerification(_) => {}
        }
    }
    let round = JointRound {
        stage,
        dispatched: results.len(),
        verdicts,
        failures,
        header: header.to_string(),
        threshold,
    };
    let has_no_issues = round
        .verdicts
        .iter()
        .all(|v| v.verdict.issues_detected.is_empty());
    let single_verifier_verdict = matches!(role, Role::Reviewer | Role::Qa) && round.n_valid() == 1;
    if has_no_issues || single_verifier_verdict {
        render_joint_comment(
            &round,
            &crate::consensus::RepairOutcome::Fallback,
            &crate::consensus::ItemTable::new(&issues_by_agent(&round)),
        )
    } else {
        build_joint_comment(&round, role, ws, ticket_id, ticket_title).await
    }
}

// ── Verifier round processing (review / QA) ─────────────────────────────

/// Minimum acceptable verification score (0-10) for review and QA phases.
const REVIEW_QA_THRESHOLD: u8 = 9;

/// Check whether a review or QA verdict passes (score at or above threshold).
#[must_use]
fn verdict_passes(verdict: &crate::Verdict) -> bool {
    verdict.score >= REVIEW_QA_THRESHOLD
}

/// Static metadata driving a verifier round (reviewer or QA).
#[derive(Copy, Clone)]
pub(crate) struct VerifierInfo {
    pub(crate) role: Role,
    /// Human-readable label used in logs and bounce-breaker messages.
    pub(crate) log_label: &'static str,
    /// The phase the ticket advances to when every verifier agent passes.
    pub(crate) success_phase: TicketPhase,
    /// The phase the verifier is actively working in.
    pub(crate) active_phase: TicketPhase,
    pub(crate) prompt_template: &'static str,
    pub(crate) extraction_prompt_path: &'static str,
}

pub(crate) const REVIEWER_VI: VerifierInfo = VerifierInfo {
    role: Role::Reviewer,
    log_label: "Reviewers",
    success_phase: TicketPhase::InQa,
    active_phase: TicketPhase::InReview,
    prompt_template: "review.md",
    extraction_prompt_path: "extraction/reviewer.md",
};

pub(crate) const QA_VI: VerifierInfo = VerifierInfo {
    role: Role::Qa,
    log_label: "QA",
    success_phase: TicketPhase::InSanitation,
    active_phase: TicketPhase::InQa,
    prompt_template: "qa.md",
    extraction_prompt_path: "extraction/qa.md",
};

/// Process parallel verifier results: add the joint comment, determine
/// pass/fail, and update ticket phase accordingly.
pub(crate) async fn process_verifier_verdicts(
    ws: &Workspace,
    ticket: &Ticket,
    results: &[ParallelVerdict],
    verifier: VerifierInfo,
    job_id: &str,
) -> bool {
    // Distinguish the two failure classes: a verifier that did NOT complete
    // (NoResponse/ParseFailed) is a HARD TECHNICAL failure — reset the attempt
    // (comment + delete job + pause; no bounce budget). A verifier that DID
    // complete but found issues (a Verdict below threshold) is a rework verdict
    // — bounce to development, consuming bounce budget.
    let technical_failure = results.iter().any(|r| {
        matches!(
            r,
            ParallelVerdict::NoResponse(_) | ParallelVerdict::ParseFailed(_)
        )
    });
    let rework_failure = !technical_failure
        && results.iter().any(|r| match r {
            ParallelVerdict::Verdict(v) => !verdict_passes(v),
            _ => false,
        });

    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            stage = %verifier.log_label,
            "Verifier round cut short by drain — job stays launched for boot resume",
        );
        return false;
    }

    if technical_failure {
        // Hard technical failure: a verifier did not complete. Reset the
        // attempt (the round is destroyed; the puller creates a fresh one).
        let comment = format!(
            "{} could not complete the round (a verifier did not respond).",
            verifier.log_label,
        );
        reset_phase_attempt(
            ticket,
            verifier.active_phase,
            job_id,
            verifier.log_label,
            &comment,
        )
        .await;
        return false;
    }

    // Build the joint comment only for the success / rework paths — the reset
    // path above uses its own short failure comment.
    let joint_comment = build_round_joint_comment(
        stage_name(verifier.role),
        results,
        REVIEW_QA_THRESHOLD,
        verifier.role,
        "",
        ws,
        &ticket.id,
        &ticket.title,
    )
    .await;

    if !rework_failure {
        return apply_clean_verifier_round(ticket, verifier, &joint_comment, job_id).await;
    }

    let outcome = bounce_to_development(
        ticket,
        verifier.active_phase,
        verifier.log_label,
        /* drains_siblings */ true,
        stage_name(verifier.role),
        &joint_comment,
        job_id,
        ws,
    )
    .await;
    matches!(outcome, FinalizeOutcome::Applied)
}

/// Apply the clean-pass outcome of a verifier round: write the joint comment,
/// transition the ticket to its next phase, and delete the phase job. Returns
/// `false` if the transition was not applied (phase moved concurrently).
async fn apply_clean_verifier_round(
    ticket: &Ticket,
    verifier: VerifierInfo,
    joint_comment: &str,
    job_id: &str,
) -> bool {
    if !matches!(
        comment_and_transition(
            TransitionCtx::buffered(
                ticket,
                verifier.active_phase,
                verifier.success_phase,
                verifier.log_label,
            ),
            stage_name(verifier.role),
            joint_comment,
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        return false;
    }
    info!(
        ticket = %ticket.id,
        "{log_label}: all passed (≥ {threshold}/10)",
        log_label = verifier.log_label,
        threshold = REVIEW_QA_THRESHOLD,
    );
    // Delete the phase job; the puller creates the next phase job.
    let _ = crate::jobs::complete_ticket_job(&crate::session::store().conn, job_id).await;
    true
}
