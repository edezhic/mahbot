//! Joint verdict comments for pipeline stages (analysis, review, QA).
//!
//! Replaces the per-agent verdict comments with ONE comment per non-all-pass
//! round. The merge backbone is deterministic: issues are merged by
//! exact-normalized text match, every issue renders with a code-computed
//! [n/N] bracket, and a single LLM synthesis pass groups related issues into
//! prose. The LLM never produces numbers — brackets, counts, agreement and
//! blocker/minor classification are all computed here from the actual
//! verdicts.
//!
//! Anti-fabrication: every synthesis group member must carry the verbatim
//! (exact-normalized) issue text of a specific agent; any violation rejects
//! the output and retries the synthesis, eventually falling back to a
//! deterministic comment.

use serde::Deserialize;
use std::fmt::Write as _;
use std::time::Instant;

use crate::retry::{FailureClass, RetryFailureRecord, RetryLoop, RetryPolicy};
use crate::{ChatMessage, ChatRequest, ChatRequestMeta, Role};

// ── Config defaults (string + typed, lockstep pair) ─────────────────────

pub(crate) const DEFAULT_REVIEW_COUNT_LOW_CHURN_STR: &str = "50";
pub(crate) const DEFAULT_REVIEW_COUNT_HIGH_CHURN_STR: &str = "400";
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

// ── Deterministic merge backbone ────────────────────────────────────────

/// Normalize an issue text for exact-match merging and validation:
/// trim and collapse runs of whitespace to a single space. Case is
/// preserved — merging must never silently fold distinct claims.
#[must_use]
pub(crate) fn normalize_issue_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A distinct issue raised by one or more agents, merged by exact-normalized
/// text match.
pub(crate) struct MergedIssue {
    /// Original (unnormalized) text from the first agent that raised it.
    pub text: String,
    /// Agent indices whose verdicts raised this issue.
    pub agents: Vec<usize>,
    /// Number of valid verdicts in the round (N).
    pub n_valid: usize,
    /// How many of the raising agents' verdicts scored below the threshold.
    /// `> 0` ⇒ the issue renders as a blocker.
    pub sources_below_threshold: usize,
}

/// Merge the issues of all valid verdicts by exact-normalized text.
///
/// An agent listing the same issue twice contributes one vote. Order is
/// first-seen across agents. This is the deterministic backbone — it can
/// never lie about consensus because it only merges byte-identical texts.
#[must_use]
pub(crate) fn merge_issues(round: &JointRound<'_>) -> Vec<MergedIssue> {
    let n_valid = round.n_valid();
    let mut issues: Vec<MergedIssue> = Vec::new();
    for v in &round.verdicts {
        let mut seen_here: Vec<String> = Vec::new();
        for issue in &v.verdict.issues_detected {
            let norm = normalize_issue_text(issue);
            if seen_here.contains(&norm) {
                continue;
            }
            seen_here.push(norm.clone());
            match issues
                .iter_mut()
                .find(|m| normalize_issue_text(&m.text) == norm)
            {
                Some(m) => {
                    if !m.agents.contains(&v.agent_index) {
                        m.agents.push(v.agent_index);
                        if v.verdict.score < round.threshold {
                            m.sources_below_threshold += 1;
                        }
                    }
                }
                None => issues.push(MergedIssue {
                    text: issue.clone(),
                    agents: vec![v.agent_index],
                    n_valid,
                    sources_below_threshold: usize::from(v.verdict.score < round.threshold),
                }),
            }
        }
    }
    issues
}

/// Render the bracket for a merged issue.
///
/// Semantics (per spec):
/// - `[N/N]` when all valid agents agree (N > 1);
/// - `DISPUTED` when exactly one valid verdict raised the issue (n == 1,
///   N > 1) or `disputed` is set (synthesis surfaced a genuine contradiction);
/// - `[1/1]` with an explicit "not cross-checked" note when N == 1;
/// - `[n/N]` otherwise.
#[must_use]
pub(crate) fn bracket_label(issue: &MergedIssue, disputed: bool) -> String {
    let n = issue.agents.len();
    let n_valid = issue.n_valid;
    if n_valid == 1 {
        "[1/1] single-agent finding, not cross-checked".to_string()
    } else if disputed || n == 1 {
        "DISPUTED".to_string()
    } else {
        format!("[{n}/{n_valid}]")
    }
}

/// Classify whether two issue texts differ only in numeric details
/// (line numbers, counts, ranges — locators/evidence, not property
/// contradictions). Used to veto the LLM's "contradiction" flag: a group
/// whose members differ only numerically is never a genuine disagreement.
#[must_use]
pub(crate) fn issues_differ_only_in_numeric_details(a: &str, b: &str) -> bool {
    let stripped = |s: &str| -> String {
        normalize_issue_text(
            &s.chars()
                .filter(|c| !(c.is_ascii_digit() || matches!(c, '-' | '–' | ',' | '.' | ':')))
                .collect::<String>(),
        )
    };
    stripped(a) == stripped(b)
}

// ── Synthesis output (strict schema — the LLM never produces numbers) ──

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SynthesisMember {
    /// Agent index whose verbatim issue this member carries (subsumption label).
    pub agent: usize,
    /// Verbatim issue text from that agent's verdict list.
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SynthesisGroup {
    pub heading: String,
    /// Set by the LLM only for genuine contradictions between agents.
    #[serde(default)]
    pub contradiction: bool,
    pub members: Vec<SynthesisMember>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SynthesisOutput {
    /// Human-readable overall summary prose.
    pub summary: String,
    pub groups: Vec<SynthesisGroup>,
}

/// Outcome of the synthesis pass: grouped by the LLM, or the deterministic
/// fallback when the synthesis exhausted its retry budget.
pub(crate) enum SynthesisOutcome {
    Grouped(SynthesisOutput),
    Fallback,
}

/// Validate the synthesis output against the actual verdicts (membership
/// evidence — the anti-fabrication invariant).
///
/// Rejects:
/// - a member whose text does not exist verbatim-normalized in that agent's
///   verdict issue list (fabricated consensus);
/// - the same issue (by normalized text) appearing more than once across all
///   groups — double-counting is rejected regardless of which agent's copy
///   carries it;
/// - an invalid agent index;
/// - any number in LLM-authored prose (summary and group headings). Member
///   text is exempt: it is a verbatim copy and the membership check is
///   authoritative for it.
pub(crate) fn validate_synthesis_output(
    output: &SynthesisOutput,
    issues_by_agent: &[Vec<String>],
) -> Result<(), String> {
    if output.summary.trim().is_empty() {
        return Err("synthesis summary is empty".to_string());
    }
    // Numbers are banned in LLM-authored prose (summary + headings) — checked
    // even when no groups are present.
    if contains_number(&output.summary) {
        return Err(format!(
            "synthesis summary contains a number: {:?}",
            output.summary
        ));
    }
    // Each normalized issue text may appear at most once across all groups
    // (a merged issue carries one member, whichever agent's copy the LLM
    // chose; the bracket shows how many agents raised it).
    let mut seen: Vec<String> = Vec::new();
    for (group_idx, group) in output.groups.iter().enumerate() {
        if group.heading.trim().is_empty() {
            return Err(format!("group {group_idx} has an empty heading"));
        }
        if contains_number(&group.heading) {
            return Err(format!(
                "group {group_idx} heading contains a number: {:?}",
                group.heading
            ));
        }
        for member in &group.members {
            let norm = normalize_issue_text(&member.text);
            let agent = member.agent;
            let Some(issues) = issues_by_agent.get(agent) else {
                return Err(format!(
                    "group {group_idx} references unknown agent index {agent}"
                ));
            };
            if !issues.iter().any(|i| i == &norm) {
                return Err(format!(
                    "group {group_idx} member text not found verbatim in agent {agent}'s verdict: {norm:?}"
                ));
            }
            if seen.contains(&norm) {
                return Err(format!(
                    "group {group_idx} double-counts the issue: {norm:?}"
                ));
            }
            seen.push(norm);
        }
    }
    Ok(())
}

/// Reject any number in LLM-authored prose (summary and group headings): the
/// synthesis prompt forbids ALL numbers there — brackets, counts, scores,
/// percentages, and standalone digits. Member text is exempt: it is a verbatim
/// copy of an agent issue and the membership check is authoritative for it.
fn contains_number(text: &str) -> bool {
    static NUMBER: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"\d").expect("number regex is valid"));
    NUMBER.is_match(text)
}

// ── Synthesis retry loop ────────────────────────────────────────────────

/// Build the synthesis chat request for a stage role (the stage role's own
/// model, reasoning effort, and provider routing — no separate grouping
/// model).
fn synthesis_request(round: &JointRound<'_>, role: Role) -> ChatRequest {
    let system = crate::prompt::load_prompt("synthesis.md");
    let mut material = String::new();
    for v in &round.verdicts {
        // Agent labels are ZERO-BASED and match the schema's `agent` field —
        // a faithful LLM can copy the label directly into the JSON. Scores are
        // deliberately NOT included: they are not needed for grouping, and a
        // faithful LLM echoing one would be rejected (the LLM never produces
        // numbers — the renderer adds scores from code).
        let _ = std::fmt::write(&mut material, format_args!("Agent {}:\n", v.agent_index));
        for issue in &v.verdict.issues_detected {
            let _ = writeln!(material, "- {issue}");
        }
    }
    let user = format!(
        "{}\n\nStage: {}\nAgent issues (verbatim):\n{}",
        crate::prompt::load_prompt("synthesis_input.md"),
        round.stage,
        material,
    );
    let model = crate::config::CONFIG.role_model(role);
    let routing = crate::config::CONFIG.model_routing(&model);
    ChatRequest {
        messages: vec![ChatMessage::system(system), ChatMessage::user(user)],
        tools: None,
        model,
        allow_image_parts: false,
        max_tokens: Some(8_000),
        reasoning_effort: Some(crate::config::CONFIG.role_reasoning_effort(role)),
        provider_order: routing.provider_order,
        provider_allow_fallbacks: routing.allow_fallbacks,
        response_format_json_object: true,
        meta: Some(ChatRequestMeta {
            purpose: "synthesis",
            agent_id: format!("joint_verdict_{}", crate::generate_suffix()),
            role: role.as_str().to_string(),
            workspace: String::new(),
            ticket_id: None,
        }),
    }
}

/// Run the single LLM synthesis pass with the dedicated retry policy
/// (3 attempts, 30–45 s backoff). Every failure — transport, unparseable
/// output, or membership-validation rejection — retries; a validation
/// rejection is fed back into the next attempt so the LLM can self-correct.
/// Exhaustion yields the deterministic fallback (never a fabricated comment).
#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn run_synthesis(
    round: &JointRound<'_>,
    role: Role,
    ws_name: &str,
) -> SynthesisOutcome {
    let _call = crate::call_registry::NON_AGENT_CALLS.register("synthesis", ws_name);
    let policy = RetryPolicy::synthesis_from_config();
    let mut loop_state = RetryLoop::new(&policy);
    let mut request = synthesis_request(round, role);
    // Key the per-agent issue lists by the ORIGINAL dispatch index (the label
    // the LLM sees in the input) — never a compacted 0..n_valid-1 space. A
    // failed agent's slot stays empty, so a member referencing it is rejected.
    let mut issues_by_agent: Vec<Vec<String>> = vec![Vec::new(); round.dispatched];
    for v in &round.verdicts {
        issues_by_agent[v.agent_index] = v
            .verdict
            .issues_detected
            .iter()
            .map(|i| normalize_issue_text(i))
            .collect();
    }

    for attempt in 1..=policy.max_attempts {
        if loop_state.expired() {
            break;
        }
        let attempt_started = Instant::now();
        match crate::providers::chat_scoped(
            request.clone(),
            policy.idle_timeout,
            loop_state.deadline(),
        )
        .await
        {
            Ok(resp) => {
                let raw = resp.text_or_empty();
                match crate::util::json::parse_fenced_json::<SynthesisOutput>(raw) {
                    Ok(output) => match validate_synthesis_output(&output, &issues_by_agent) {
                        Ok(()) => {
                            tracing::info!(
                                stage = round.stage,
                                agents = round.dispatched,
                                groups = output.groups.len(),
                                "Joint verdict synthesis succeeded",
                            );
                            return SynthesisOutcome::Grouped(output);
                        }
                        Err(msg) => {
                            // Feed the rejection back so the next attempt can
                            // self-correct instead of re-failing identically.
                            if let Some(last) = request.messages.last_mut() {
                                let _ = std::fmt::write(
                                    &mut last.content,
                                    format_args!(
                                        "\n\nYour previous response was rejected: {msg}.\n\
                                         Fix the violation and try again."
                                    ),
                                );
                            }
                            let err = anyhow::anyhow!("synthesis validation failed: {msg}");
                            let rec = RetryFailureRecord::new_simple(
                                attempt,
                                FailureClass::Parse,
                                &err,
                                attempt_started.elapsed().as_millis() as u64,
                                None,
                            );
                            loop_state.record(attempt, rec).await;
                        }
                    },
                    Err(e) => {
                        let rec = RetryFailureRecord::new_simple(
                            attempt,
                            FailureClass::Parse,
                            &e,
                            attempt_started.elapsed().as_millis() as u64,
                            None,
                        );
                        loop_state.record(attempt, rec).await;
                    }
                }
            }
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(attempt, err.record).await;
                if non_retryable {
                    break;
                }
            }
        }
        if let Err(FailureClass::Shutdown) = loop_state.sleep_between(attempt).await {
            break;
        }
    }

    tracing::warn!(
        stage = round.stage,
        agents = round.dispatched,
        attempts = policy.max_attempts,
        "Joint verdict synthesis exhausted — writing deterministic fallback comment",
    );
    SynthesisOutcome::Fallback
}

/// Convenience: run the synthesis pass and render the joint comment.
pub(crate) async fn build_joint_comment(
    round: &JointRound<'_>,
    role: Role,
    ws_name: &str,
) -> String {
    let outcome = run_synthesis(round, role, ws_name).await;
    render_joint_comment(round, &outcome)
}

// ── Joint comment rendering ─────────────────────────────────────────────

/// Render the joint comment for a round given the synthesis outcome.
///
/// Structure: header (code-computed counts), issues with code-computed
/// brackets (grouped by the LLM when synthesis succeeded, deterministic
/// otherwise), per-agent critiques, the LLM summary prose (or an explicit
/// "LLM grouping unavailable" marker), and a raw-dump appendix for failed
/// agents.
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn render_joint_comment(round: &JointRound<'_>, outcome: &SynthesisOutcome) -> String {
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

    // Issues with brackets — grouped by the LLM when available. Only rendered
    // when the round actually produced issues.
    let issues = merge_issues(round);
    if !issues.is_empty() {
        // Normalized texts already placed in a group (issue-level: a merged
        // issue is placed once, whichever agent's copy the LLM chose).
        let mut used: Vec<String> = Vec::new();
        if let SynthesisOutcome::Grouped(output) = outcome {
            for group in &output.groups {
                let mut lines: Vec<String> = Vec::new();
                for member in &group.members {
                    let norm = normalize_issue_text(&member.text);
                    // Validation guarantees this member maps to a merged issue
                    // (both derive from the same verdicts) — direct lookup.
                    let issue = issues
                        .iter()
                        .find(|m| {
                            normalize_issue_text(&m.text) == norm
                                && m.agents.contains(&member.agent)
                        })
                        .expect("validated synthesis member maps to a merged issue");
                    let other_texts: Vec<&str> = group
                        .members
                        .iter()
                        .filter(|m| m.agent != member.agent)
                        .map(|m| m.text.as_str())
                        .collect();
                    let numeric_only = !other_texts.is_empty()
                        && other_texts
                            .iter()
                            .all(|t| issues_differ_only_in_numeric_details(&member.text, t));
                    let disputed = group.contradiction && !numeric_only;
                    lines.push(render_issue_line(issue, disputed));
                    if !used.contains(&norm) {
                        used.push(norm);
                    }
                }
                if lines.is_empty() {
                    continue;
                }
                let _ = write!(out, "\n\n**{}**", group.heading);
                for line in &lines {
                    let _ = write!(out, "\n- {line}");
                }
            }
            // Deterministic catch-all for issues the LLM did not group.
            let mut remaining: Vec<String> = Vec::new();
            for issue in &issues {
                let norm = normalize_issue_text(&issue.text);
                if !used.contains(&norm) {
                    remaining.push(render_issue_line(issue, false));
                }
            }
            if !remaining.is_empty() {
                out.push_str("\n\n**Remaining issues**");
                for line in remaining {
                    let _ = write!(out, "\n- {line}");
                }
            }
        } else {
            out.push_str("\n\n**Issues**");
            for issue in &issues {
                let _ = write!(out, "\n- {}", render_issue_line(issue, false));
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

    // LLM summary or explicit fallback marker.
    match outcome {
        SynthesisOutcome::Grouped(output) => {
            out.push_str("\n\n### Summary");
            let _ = write!(out, "\n{}", output.summary.trim());
        }
        SynthesisOutcome::Fallback => {
            out.push_str("\n\n### Summary\nLLM grouping unavailable — deterministic merge only.");
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

fn render_issue_line(issue: &MergedIssue, disputed: bool) -> String {
    let bracket = bracket_label(issue, disputed);
    let severity = if issue.sources_below_threshold > 0 {
        "[blocker] "
    } else {
        ""
    };
    format!("{bracket} {severity}{}", issue.text)
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
