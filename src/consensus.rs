//! Shared LLM grouping/synthesis core for pipeline joint-verdict and ask
//! consolidation.
//!
//! One LLM pass groups indexed per-member items (pipeline issues; ask claims)
//! into thematic/semantic groups. The LLM never produces counts: brackets
//! `[n/N]` and the DISPUTED marker are computed here from distinct cited
//! member ids — pure arithmetic. Anti-fabrication comes from verbatim
//! membership validation, index-arithmetic completeness, and a contradiction
//! guardrail (contradiction:true requires ≥2 distinct cited agents).

use serde::Deserialize;
use std::time::Instant;

use crate::retry::{FailureClass, RetryFailureRecord, RetryLoop, RetryPolicy};
use crate::{ChatRequest, ChatRequestMeta, Workspace};

// ── Schema (strict — unknown fields rejected) ──────────────────────────

/// One member of a group: a verbatim item from one source (agent).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingMember {
    /// Zero-based source (agent) index — matches the input list labels.
    pub agent: usize,
    /// Verbatim item text from that source's list.
    pub text: String,
}

/// A group of related items plus the LLM's contradiction judgment.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingGroup {
    pub heading: String,
    /// Set by the LLM only for genuine contradictions between agents.
    #[serde(default)]
    pub contradiction: bool,
    pub members: Vec<GroupingMember>,
}

/// The full grouping output: summary prose + groups + an explicit ungrouped
/// list (a valid, visible destination — silent dropout is rejected).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupingOutput {
    pub summary: String,
    pub groups: Vec<GroupingGroup>,
    #[serde(default)]
    pub ungrouped: Vec<GroupingMember>,
}

/// Outcome of the grouping pass: grouped by the LLM, or the deterministic
/// fail-open (raw member dump + explicit marker) when retries exhausted.
pub(crate) enum GroupingOutcome {
    Grouped(GroupingOutput),
    Fallback,
}

// ── Normalization ──────────────────────────────────────────────────────

/// Normalize an item for verbatim membership validation: trim and collapse
/// whitespace runs to a single space. Case is preserved — validation must
/// never silently fold distinct items.
#[must_use]
pub(crate) fn normalize_item(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── Validation ─────────────────────────────────────────────────────────

/// Validate a grouping output against the actual per-agent item lists.
///
/// Rejects:
/// - a member whose text does not exist verbatim-normalized in the cited
///   agent's list (fabricated consensus);
/// - the same (agent, normalized text) appearing more than once across groups
///   and the ungrouped list (double-counting);
/// - an out-of-range agent index;
/// - a group flagged contradiction:true without ≥2 distinct cited agents;
/// - incomplete coverage — every input item must appear exactly once across
///   groups + ungrouped (index-arithmetic completeness; silent dropout is
///   rejected and retried, never silently absorbed).
pub(crate) fn validate_grouping(
    output: &GroupingOutput,
    items_by_agent: &[Vec<String>],
) -> Result<(), String> {
    if output.summary.trim().is_empty() {
        return Err("grouping summary is empty".to_string());
    }
    let mut seen: Vec<(usize, String)> = Vec::new();
    let mut check_members = |members: &[GroupingMember], where_: &str| -> Result<(), String> {
        for member in members {
            let norm = normalize_item(&member.text);
            let agent = member.agent;
            let Some(items) = items_by_agent.get(agent) else {
                return Err(format!("{where_}: unknown agent index {agent}"));
            };
            if !items.iter().any(|i| i == &norm) {
                return Err(format!(
                    "{where_}: member text not found verbatim in agent {agent}'s list: {norm:?}"
                ));
            }
            if seen.contains(&(agent, norm.clone())) {
                return Err(format!(
                    "{where_}: double-counts agent {agent}'s item: {norm:?}"
                ));
            }
            seen.push((agent, norm));
        }
        Ok(())
    };
    for (group_idx, group) in output.groups.iter().enumerate() {
        if group.heading.trim().is_empty() {
            return Err(format!("group {group_idx} has an empty heading"));
        }
        if group.contradiction {
            let mut agents: Vec<usize> = group.members.iter().map(|m| m.agent).collect();
            agents.sort_unstable();
            agents.dedup();
            if agents.len() < 2 {
                return Err(format!(
                    "group {group_idx} flagged contradiction without ≥2 distinct cited agents"
                ));
            }
        }
        check_members(&group.members, &format!("group {group_idx}"))?;
    }
    check_members(&output.ungrouped, "ungrouped list")?;
    // Index-arithmetic completeness: every input item appears exactly once.
    for (agent, items) in items_by_agent.iter().enumerate() {
        for item in items {
            if !seen.contains(&(agent, item.clone())) {
                return Err(format!(
                    "agent {agent} item missing from every group and the ungrouped list: {item:?}"
                ));
            }
        }
    }
    Ok(())
}

// ── Bracket arithmetic (pure code — never LLM-written counts) ───────────

/// Distinct cited agent ids of a group, sorted.
#[must_use]
pub(crate) fn distinct_agents(group: &GroupingGroup) -> Vec<usize> {
    let mut agents: Vec<usize> = group.members.iter().map(|m| m.agent).collect();
    agents.sort_unstable();
    agents.dedup();
    agents
}

/// Render the bracket for a group: `[n/N]`, with ` · DISPUTED` appended only
/// when the group carries a genuine contradiction (contradiction:true — which
/// validation guarantees implies ≥2 distinct cited agents). Solo `[1/N]`
/// renders without DISPUTED.
#[must_use]
pub(crate) fn bracket_label(n: usize, n_valid: usize, disputed: bool) -> String {
    if disputed {
        format!("[{n}/{n_valid} · DISPUTED]")
    } else {
        format!("[{n}/{n_valid}]")
    }
}

// ── Rendering helpers (shared) ─────────────────────────────────────────

/// Render one member line attributed to its source: `Agent N: text`.
#[must_use]
pub(crate) fn render_member_line(member: &GroupingMember) -> String {
    format!("Agent {}: {}", member.agent, member.text)
}

// ── Retry loop ─────────────────────────────────────────────────────────

/// Run the grouping pass with the dedicated synthesis retry policy (3
/// attempts, 30–45 s backoff). Every failure — transport, unparseable output,
/// or validation rejection — retries; a validation rejection is fed back into
/// the next attempt so the LLM can self-correct. Exhaustion yields the
/// deterministic fail-open (never a fabricated grouping).
///
/// The caller supplies the full request (system + user messages with the
/// consumer's prompt, model, and max_tokens); the leading general workspace
/// context message is prepended here so every consumer consumes it.
#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn run_grouping(
    ws: &Workspace,
    purpose: &'static str,
    mut request: ChatRequest,
    items_by_agent: &[Vec<String>],
) -> GroupingOutcome {
    let _call = crate::call_registry::NON_AGENT_CALLS.register(purpose, &ws.name);
    crate::prompt::prepend_general_context(&mut request.messages, ws).await;
    let policy = RetryPolicy::synthesis_from_config();
    let mut loop_state = RetryLoop::new(&policy);
    let operation_started = Instant::now();

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
                match crate::util::json::parse_fenced_json::<GroupingOutput>(raw) {
                    Ok(output) => match validate_grouping(&output, items_by_agent) {
                        Ok(()) => {
                            tracing::info!(
                                purpose,
                                workspace = %ws.name,
                                groups = output.groups.len(),
                                ungrouped = output.ungrouped.len(),
                                "Grouping synthesis succeeded",
                            );
                            crate::stats::record_llm_success(
                                &request,
                                operation_started,
                                attempt,
                                &resp,
                            )
                            .await;
                            return GroupingOutcome::Grouped(output);
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
                            let err = anyhow::anyhow!("grouping validation failed: {msg}");
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
        purpose,
        workspace = %ws.name,
        attempts = policy.max_attempts,
        "Grouping synthesis exhausted — writing deterministic fail-open output",
    );
    let final_class = loop_state.final_class();
    let exhausted =
        crate::retry::RetryExhausted::with_last_raw(loop_state.into_failures(), final_class, None);
    crate::stats::record_llm_failure(&request, operation_started, &exhausted).await;
    GroupingOutcome::Fallback
}

/// Build a grouping request with the given system prompt, user material, and
/// consumer-supplied model/effort/routing/max_tokens. The `meta` purpose and
/// agent id are filled in here so every consumer shares the same telemetry
/// shape (the registry entry uses the same purpose string).
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouping_request(
    ws: &Workspace,
    purpose: &'static str,
    system: &str,
    user: &str,
    model: String,
    reasoning_effort: Option<String>,
    provider_order: Option<String>,
    provider_allow_fallbacks: Option<bool>,
    max_tokens: Option<u32>,
) -> ChatRequest {
    ChatRequest {
        messages: vec![
            crate::ChatMessage::system(system),
            crate::ChatMessage::user(user),
        ],
        tools: None,
        model,
        allow_image_parts: false,
        max_tokens,
        reasoning_effort,
        provider_order,
        provider_allow_fallbacks,
        response_format_json_object: true,
        meta: Some(ChatRequestMeta {
            purpose,
            agent_id: format!("grouping_{}_{}", ws.name, crate::generate_suffix()),
            role: "grouping".to_string(),
            workspace: ws.name.clone(),
            ticket_id: None,
        }),
    }
}
