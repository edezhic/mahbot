//! AskTool — spawns a sub-agent to ask a question.
//!
//! Available to the Engineer and Maintainer agents (sync mode), and to the
//! Manager and Assistant agents (async mode). In sync mode the caller blocks
//! until the sub-agent completes. In async mode the sub-agent is dispatched
//! in a background task and the result is injected back to the caller's
//! agent channel via [`crate::message_router::route`].
//!
//! Analyst batches run three decorrelated analysts (distinct research angles)
//! that report structured claim-level findings; consolidation runs the shared
//! LLM grouping pass ([`crate::consensus`]) — semantic grouping + contradiction
//! judgment with code-computed agreement brackets. The verification pass for
//! disputed claims lives in the deep research tool only. Fail-open is
//! preserved throughout: findings are never silently lost.

use crate::agent::run_agent;
use crate::config::CONFIG;
use crate::message_router::{self, AgentJob, JobKind};
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::session::{ask_agent_id, resolve_agent_id};
use crate::tools::Tool;
use crate::{Agent, DEFAULT_MAX_TOKENS, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::Write as _;

/// Controls sub-agent dispatch behaviour.
///
/// [`Sync`](DispatchMode::Sync) blocks the caller until the sub-agent completes.
/// [`Async`](DispatchMode::Async) dispatches the sub-agent in a background task
/// and injects the result via the caller's agent queue.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DispatchMode {
    Sync,
    Async,
}

impl DispatchMode {
    /// Returns `true` when this dispatch mode is [`Async`](DispatchMode::Async).
    #[must_use]
    pub const fn is_async(self) -> bool {
        matches!(self, Self::Async)
    }
}

pub struct AskTool {
    pub allowed_roles: Vec<Role>,
    /// Controls how the sub-agent is dispatched.
    /// - [`DispatchMode::Sync`] — blocks the caller until the sub-agent completes.
    /// - [`DispatchMode::Async`] — dispatches in a background task, result
    ///   delivered via the caller's agent queue.
    dispatch_mode: DispatchMode,
    /// The role of the calling agent. Used to route async results to the
    /// correct agent channel (Manager → manager_{ws}, Assistant → direct_{...}).
    pub caller_role: Role,
}

impl AskTool {
    #[must_use]
    pub const fn new(
        allowed_roles: Vec<Role>,
        dispatch_mode: DispatchMode,
        caller_role: Role,
    ) -> Self {
        Self {
            allowed_roles,
            dispatch_mode,
            caller_role,
        }
    }

    fn formatted_allowed_roles(&self) -> String {
        self.allowed_roles
            .iter()
            .map(|r| format!("'{}'", r.as_str()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[async_trait]
impl Tool for AskTool {
    fn name(&self) -> &'static str {
        "ask"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let role_str = self.formatted_allowed_roles();
        super::tool_params_schema(
            &json!({
                "role": {
                    "type": "string",
                    "description": format!("Role name of the agent: {role_str}")
                },
                "ask": {
                    "type": "string",
                    "description": "The ask to delegate to the agent"
                }
            }),
            &["role", "ask"],
        )
    }

    /// Async sub-agents could theoretically produce side effects through their
    /// own tool sets, but in practice the sub-agents dispatched by AskTool
    /// (Analysts) have no mutation tools (no edit, no full shell — only
    /// read-only shell, which reports [`Self::side_effects`] = false). This
    /// classification is coupled to `Role::tools()`; if Analyst ever gains
    /// side-effecting tools, this must be reconsidered.
    fn side_effects(&self) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let role_str = super::get_str(&args, "role")?;
        let ask = super::get_str(&args, "ask")?;

        let allowed_str = self.formatted_allowed_roles();
        let role: Role = match role_str.parse() {
            Ok(r) if self.allowed_roles.contains(&r) => r,
            Ok(_) => {
                anyhow::bail!("Cannot delegate to '{role_str}'. Only {allowed_str} are supported.");
            }
            Err(_) => {
                anyhow::bail!("Unknown role '{role_str}'. Use {allowed_str}.");
            }
        };

        // Async dispatch path — delegate to sub-agent in background.
        // Read user context from task-locals (set by Agent work loop
        // before each tool.execute() call) so the queued result carries
        // the correct user identity for per-user delivery.
        if self.dispatch_mode.is_async() {
            let ws = ws.clone();
            let ask = ask.to_string();
            let caller_role = self.caller_role;
            let user_name = crate::agent::CURRENT_TOOL_USER_NAME
                .try_with(String::clone)
                .unwrap_or_default();
            let channel = crate::agent::CURRENT_TOOL_CHANNEL
                .try_with(String::clone)
                .unwrap_or_default();

            tokio::spawn(async move {
                let message = build_async_ask_message(run_sub_agent(&ws, role, &ask).await);

                // Route result to the caller's agent channel.
                let target_agent_id =
                    resolve_agent_id(&channel, &user_name, caller_role.as_str(), &ws.name);
                message_router::route(
                    &target_agent_id,
                    AgentJob {
                        content: message,
                        workspace_name: ws.name.clone(),
                        user_name,
                        channel,
                        kind: JobKind::AskToolResult,
                        role: caller_role,
                        reply_target: None,
                    },
                );
            });

            return Ok("Sub-agent dispatched. Results will follow shortly.".to_string());
        }

        // Sync path — blocks caller until sub-agent completes.
        run_sub_agent(ws, role, ask).await
    }
}

/// Build the `<ask-tool-result>` envelope message for an async ask dispatch.
///
/// Shared by the async dispatch path (the `tokio::spawn` body in
/// [`AskTool::execute`]) and tests — the envelope shape that reaches the
/// caller's agent channel is production code, not a test re-wrap.
fn build_async_ask_message(result: anyhow::Result<String>) -> String {
    build_async_result_envelope(result, "ask-tool-result")
}

/// Wrap a sub-agent/tool result in the async `<tag>` envelope delivered to
/// the caller's agent channel. Failures carry an explicit marker — findings
/// are never silently dropped. Shared with the deep research tool.
pub(crate) fn build_async_result_envelope(result: anyhow::Result<String>, tag: &str) -> String {
    match result {
        Ok(text) => format!("<{tag}>\n\n{text}</{tag}>"),
        Err(e) => {
            tracing::debug!(error = %e, %tag, "async tool result failed");
            format!("<{tag}>\n\nAn error occurred: {e}</{tag}>")
        }
    }
}

/// Shared sub-agent runner — delegates to [`run_agent`] for the given role
/// and ask. Used by both sync and async paths of [`AskTool::execute`].
///
/// For [`Role::Analyst`], spawns 3 decorrelated parallel analysts and
/// consolidates their claim-level findings. For all other roles, dispatches a
/// single agent.
async fn run_sub_agent(ws: &Workspace, role: Role, ask: &str) -> Result<String> {
    if role == Role::Analyst {
        return run_parallel_analysts_and_consolidate(ws, ask).await;
    }

    // Single-agent path for non-Analyst roles — delegate lifecycle to run_agent.
    let agent_id = ask_agent_id(&ws.name, role.as_str());
    let (agent, response) = run_agent(
        agent_id,
        role,
        ws,
        None,
        ask,
        String::new(),
        String::new(),
        None,
    )
    .await;

    if let Some(response) = response {
        Ok(response)
    } else if agent.is_cancelled() || crate::shutdown::shutdown_token().is_cancelled() {
        anyhow::bail!("Sub-agent cancelled");
    } else {
        anyhow::bail!(
            "Sub-agent failed: {}",
            agent.failure.as_deref().unwrap_or("unknown error")
        );
    }
}

// ── Structured claim-level findings ──────────────────────────────────────

/// A single claim with source and confidence, extracted from an analyst's
/// response. Shared with the deep research tool (`research.rs`).
#[allow(clippy::struct_field_names)] // field name matches the JSON schema key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Claim {
    pub claim: String,
    pub source: String,
    pub confidence: String,
    pub contradictions: Vec<String>,
}

/// Structured claim-level findings extracted from an analyst's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnalystFindings {
    pub claims: Vec<Claim>,
    pub coverage: Vec<String>,
    pub unanswered: Vec<String>,
}

/// Per-analyst outcome after findings extraction. Three mutually-exclusive
/// states mirroring `management::ParallelVerdict` — the type system guarantees
/// "no response" and "parse failure" cannot be confused.
enum AnalystOutcome {
    /// Agent failed to produce a response (crashed, cancelled, empty output).
    /// Carries the collapsed failure reason (scrubbed).
    NoResponse(String),
    /// Agent responded and structured extraction succeeded.
    Findings {
        raw: String,
        findings: AnalystFindings,
    },
    /// Agent responded but structured extraction failed after retries; the
    /// raw response is preserved for the fail-open dump.
    ParseFailed { raw: String, failure: String },
}

/// Verdict of a single targeted claim verification. Shared with the deep
/// research tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VerificationVerdict {
    pub claim: String,
    pub verdict: String,
    pub evidence: String,
    pub confidence: String,
}

/// Verification outcome merged into the final report. The telemetry fields
/// (the verifier analyst's tool calls / searches / queries) feed the deep
/// research run summary and its query ledger; the ask path ignores them.
#[derive(Debug, Clone)]
pub(crate) struct VerificationResult {
    pub claim: String,
    pub verdict: String,
    pub evidence: String,
    pub tool_calls: usize,
    pub searches: usize,
    pub queries: Vec<String>,
}

/// Reject verification verdicts outside the accepted vocabulary — fail-closed
/// inside the extraction retry loop.
pub(crate) fn validate_verification_verdict(v: &VerificationVerdict) -> Result<(), String> {
    if matches!(
        v.verdict.as_str(),
        "supported" | "contradicted" | "unresolved"
    ) {
        Ok(())
    } else {
        Err(format!(
            "verification verdict '{}' not in [supported, contradicted, unresolved]",
            v.verdict
        ))
    }
}

/// Cap on verification analysts spawned by the deep research tool's
/// verification gate (exactly one pass, bounded).
pub(crate) const VERIFY_MAX_ANALYSTS: usize = 4;

// ── Parallel analyst batch ───────────────────────────────────────────────

/// Spawn 3 decorrelated parallel analysts, then consolidate their claim-level
/// findings into a single comprehensive answer.
async fn run_parallel_analysts_and_consolidate(ws: &Workspace, ask: &str) -> Result<String> {
    const PARALLEL_ANALYST_COUNT: usize = 3;

    let runs = run_parallel_analysts(ws, ask, PARALLEL_ANALYST_COUNT).await;
    consolidate_analyst_runs(ws, ask, runs).await
}

/// Load the decorrelation angles asset — one distinct research angle per
/// parallel analyst. Malformed or missing sections degrade to an empty angle
/// (the plain ask is used, preserving the original single-question behavior).
pub(crate) fn load_analyst_angles() -> Vec<String> {
    load_prompt_sections("ask/angles.md")
}

/// Run `count` parallel analyst agents with decorrelated research angles,
/// returning the (still-alive) agent with its raw response so structured
/// extraction can reuse the agent's KV-cache parameters.
async fn run_parallel_analysts(
    ws: &Workspace,
    ask: &str,
    count: usize,
) -> Vec<(Agent, Option<String>)> {
    let suffix = crate::generate_suffix();
    let angles = load_analyst_angles();
    let futures: Vec<_> = (0..count)
        .map(|i| {
            let ws = ws.clone();
            let suffix = suffix.clone();
            let angle = angles.get(i).cloned().unwrap_or_default();
            let agent_id = format!("ask_{}_{}_{}_analyst", ws.name, suffix, i);
            // KV-cache discipline: vary ONLY the user message (the research
            // angle) — never per-analyst model/effort/tools.
            let analyst_ask = if angle.is_empty() {
                ask.to_string()
            } else {
                format!("{ask}\n\nResearch angle:\n{angle}")
            };
            async move {
                run_agent(
                    agent_id,
                    Role::Analyst,
                    &ws,
                    None,
                    &analyst_ask,
                    String::new(),
                    String::new(),
                    None,
                )
                .await
            }
        })
        .collect();
    join_all(futures).await
}

/// Consolidate analyst runs: 0 valid → error, 1 valid → raw passthrough,
/// ≥2 valid → extract findings and run the shared grouping pass (≥2 with
/// parseable claims group; a single parseable source skips grouping).
async fn consolidate_analyst_runs(
    ws: &Workspace,
    ask: &str,
    runs: Vec<(Agent, Option<String>)>,
) -> Result<String> {
    let valid_count = runs
        .iter()
        .filter(|(_, r)| r.as_deref().is_some_and(|t| !t.trim().is_empty()))
        .count();
    match valid_count {
        0 => {
            anyhow::bail!("All parallel analysts failed to produce a response");
        }
        1 => Ok(single_raw_response(&runs).expect("exactly one valid response")),
        _ => {
            let outcomes = extract_findings(runs).await;
            consolidate_findings(ws, ask, outcomes).await
        }
    }
}

fn single_raw_response(runs: &[(Agent, Option<String>)]) -> Option<String> {
    runs.iter()
        .find_map(|(_, r)| r.clone().filter(|t| !t.trim().is_empty()))
}

/// Extract structured claim-level findings from each valid analyst response
/// while the agent is still alive (see [`Agent::extract_verdict`] for the
/// KV-cache rationale). Fail-open: parse failures keep the raw response and
/// are never silently dropped.
async fn extract_findings(runs: Vec<(Agent, Option<String>)>) -> Vec<AnalystOutcome> {
    let extraction_prompt = load_prompt("extraction/findings.md");
    let futures = runs.into_iter().map(|(agent, response)| {
        let extraction_prompt = extraction_prompt.clone();
        async move {
            let Some(raw) = response else {
                let reason = if crate::shutdown::shutdown_token().is_cancelled() {
                    "service shutting down".to_string()
                } else if agent.is_cancelled() {
                    "agent cancelled by user".to_string()
                } else {
                    agent
                        .failure
                        .clone()
                        .unwrap_or_else(|| "analyst produced no response".to_string())
                };
                return AnalystOutcome::NoResponse(crate::util::scrub_credentials(&reason));
            };
            if raw.trim().is_empty() {
                return AnalystOutcome::NoResponse("analyst produced no response".to_string());
            }
            match agent
                .extract_verdict::<AnalystFindings>(&extraction_prompt, None)
                .await
            {
                Ok(findings) => AnalystOutcome::Findings { raw, findings },
                Err(e) => AnalystOutcome::ParseFailed {
                    raw,
                    failure: e.to_string(),
                },
            }
        }
    });
    join_all(futures).await
}

/// Consolidate extracted outcomes: build the per-agent claim lists and run the
/// shared repair-mode grouping pass (semantic grouping + contradiction
/// judgment, frozen groups + deterministic remainder). ≥2 valid responses go
/// through grouping; the output contract is summary + groups + optional
/// ungrouped list, with brackets computed from distinct cited agent ids.
/// Partial success (≥1 frozen group) is delivered as-is; only an exhaustion
/// with zero groups ever frozen keeps the fail-open flat claim list plus raw
/// analyst dumps with an explicit marker — never a fabricated consensus.
async fn consolidate_findings(
    ws: &Workspace,
    ask: &str,
    outcomes: Vec<AnalystOutcome>,
) -> Result<String> {
    // Per-agent claim texts (agent index = outcome index; failed agents get an
    // empty list so the id space matches the input material). Per-agent
    // duplicates are NOT deduped: two identical claims are two distinct ids,
    // and the model places each exactly once.
    let mut items_by_agent: Vec<Vec<String>> = Vec::with_capacity(outcomes.len());
    for o in &outcomes {
        let claims = match o {
            AnalystOutcome::Findings { findings, .. } => {
                findings.claims.iter().map(|c| c.claim.clone()).collect()
            }
            _ => Vec::new(),
        };
        items_by_agent.push(claims);
    }
    let n_valid = items_by_agent.iter().filter(|l| !l.is_empty()).count();
    if n_valid == 0 {
        // No valid response produced parseable claims — fail open with raw
        // reports and the per-analyst reasons (nothing silently lost). The
        // marker distinguishes extraction failures from zero-claim reports.
        let has_parse_failures = outcomes
            .iter()
            .any(|o| matches!(o, AnalystOutcome::ParseFailed { .. }));
        let raw_dump = render_raw_analyst_dump(&outcomes);
        let failures = render_extraction_failures(&outcomes);
        let marker = if has_parse_failures {
            "unconsolidated — findings extraction failed"
        } else {
            "unconsolidated — no extractable claims from any analyst"
        };
        return Ok(format!(
            "## Analyst Reports\n\n({marker})\n\n{failures}\n\n{raw_dump}"
        ));
    }
    if n_valid == 1 {
        // Only one analyst produced parseable claims — a grouping pass over a
        // single source cannot produce agreement brackets or contradictions,
        // so skip the provider call and deliver the flat claim list + raw
        // dumps with an explicit marker.
        tracing::info!("Single-analyst consolidation — skipping grouping pass");
        let flat = render_flat_claim_list(&items_by_agent);
        let raw_dump = render_raw_analyst_dump(&outcomes);
        return Ok(format!(
            "## Analyst Reports\n\n(only one analyst produced parseable claims — grouping skipped)\n\n{flat}\n\n{raw_dump}"
        ));
    }
    // User material: global flat ids across ALL agents (each claim exactly one
    // id, in (agent, claim) order) — the schema's `id` field matches.
    let mut material = String::new();
    let mut id = 0usize;
    for (agent_idx, claims) in items_by_agent.iter().enumerate() {
        if claims.is_empty() {
            continue;
        }
        let _ = writeln!(material, "Agent {agent_idx}:");
        for c in claims {
            let _ = writeln!(material, "- {id}: {c}");
            id += 1;
        }
    }
    let user =
        format!("# Original Question\n\n{ask}\n\n# Agent Claims (id-numbered)\n\n{material}");

    let model = CONFIG.role_model(Role::Analyst);
    let routing = CONFIG.model_routing(&model);
    let system = format!(
        "{}\n\n{}",
        load_prompt("consolidate/analyst.md"),
        load_prompt("grouping_contradictions.md"),
    );
    let request = crate::consensus::grouping_request(
        ws,
        "consolidate",
        &system,
        &user,
        model,
        Some(CONFIG.role_reasoning_effort(Role::Analyst)),
        routing.provider_order,
        routing.allow_fallbacks,
        Some(DEFAULT_MAX_TOKENS),
    );

    let table = crate::consensus::ItemTable::new(&items_by_agent);
    match crate::consensus::run_grouping_repair(ws, "consolidate", request, &items_by_agent).await {
        crate::consensus::RepairOutcome::Repaired { output, references } => Ok(render_ask_groups(
            ask,
            &output,
            &references,
            &table,
            n_valid,
            &outcomes,
        )),
        crate::consensus::RepairOutcome::Fallback => {
            // Fail-open deliverable: flat numbered claim list (the grouping
            // input) + raw analyst dumps + explicit marker. The marker is
            // head-placed so the sync path's 5 KB sandwich truncation keeps it.
            tracing::warn!("Analyst consolidation failed — delivering raw claim list");
            let flat = render_flat_claim_list(&items_by_agent);
            let raw_dump = render_raw_analyst_dump(&outcomes);
            Ok(format!(
                "## Analyst Reports\n\n(unconsolidated — consolidation failed)\n\n{flat}\n\n{raw_dump}"
            ))
        }
    }
}

/// Render the ask output contract: summary + groups (heading, contradiction
/// flag, members citing item ids) + ungrouped remainder + DISPUTED
/// cross-references from remainder items to frozen groups. Brackets [n/N]
/// come from distinct cited agent ids; DISPUTED appears only when the group's
/// contradiction flag is true. Self-reported caveats of a cited claim render
/// as member metadata, never as contradictions.
#[must_use]
fn render_ask_groups(
    ask: &str,
    output: &crate::consensus::GroupingOutput,
    references: &[crate::consensus::GroupingReference],
    table: &crate::consensus::ItemTable<'_>,
    n_valid: usize,
    outcomes: &[AnalystOutcome],
) -> String {
    let mut out = String::new();
    let summary = output.summary.trim();
    if summary.is_empty() {
        let _ = writeln!(
            out,
            "LLM summary unavailable — deterministic member render."
        );
    } else {
        let _ = writeln!(out, "{summary}");
    }
    for group in &output.groups {
        let n = crate::consensus::distinct_agents(group, table).len();
        let bracket = crate::consensus::bracket_label(n, n_valid, group.contradiction);
        let _ = write!(out, "\n\n**{}** {bracket}", group.heading);
        for member in &group.members {
            let caveat = member_caveat(member, table, outcomes);
            let _ = write!(
                out,
                "\n- {}",
                crate::consensus::render_member_line(member, table)
            );
            if let Some(c) = caveat {
                let _ = write!(out, " — caveat: {c}");
            }
        }
    }
    if !output.ungrouped.is_empty() {
        out.push_str("\n\n**Ungrouped**");
        for member in &output.ungrouped {
            let mut line = crate::consensus::render_member_line(member, table);
            for reference in references.iter().filter(|r| r.member.id == member.id) {
                if let Some(group) = output.groups.get(reference.group) {
                    let _ = write!(
                        line,
                        " [DISPUTED — contradicts group {} \"{}\"]",
                        reference.group, group.heading
                    );
                }
            }
            let caveat = member_caveat(member, table, outcomes);
            let _ = write!(out, "\n- {line}");
            if let Some(c) = caveat {
                let _ = write!(out, " — caveat: {c}");
            }
        }
    }
    // Original question for context (answers are delivered out of band).
    let _ = write!(out, "\n\n_Original question: {ask}_");
    out
}

/// Look up a cited member's self-reported caveats (the claim's own
/// `contradictions` field — analyst self-report, NOT a group contradiction).
/// The claim is resolved via the item table's (agent, per-agent index) — no
/// text equality.
fn member_caveat(
    member: &crate::consensus::GroupingMember,
    table: &crate::consensus::ItemTable<'_>,
    outcomes: &[AnalystOutcome],
) -> Option<String> {
    let (agent, item) = table.resolve_index(member.id)?;
    let AnalystOutcome::Findings { findings, .. } = outcomes.get(agent)? else {
        return None;
    };
    let claim = findings.claims.get(item)?;
    if claim.contradictions.is_empty() {
        None
    } else {
        Some(claim.contradictions.join("; "))
    }
}

/// Render the flat numbered claim list (the grouping-pass input) for the
/// fail-open deliverable.
#[must_use]
fn render_flat_claim_list(items_by_agent: &[Vec<String>]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "### Claims (grouping input)");
    for (agent_idx, claims) in items_by_agent.iter().enumerate() {
        for (i, c) in claims.iter().enumerate() {
            let _ = writeln!(out, "{}. [Agent {agent_idx}] {c}", i + 1);
        }
    }
    out
}

/// Normalize a claim for deterministic comparisons (QueryLedger, unanswered
/// dedup). Shared with the deep research tool.
pub(crate) fn normalize_claim(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Max of two confidence tiers ("low" < "medium" < "high"). Shared with the
/// deep research tool's cross-round claim merge (`research.rs`).
pub(crate) fn max_confidence(a: &str, b: &str) -> String {
    let rank = |c: &str| match c {
        "high" => 2,
        "medium" => 1,
        _ => 0,
    };
    if rank(b) > rank(a) {
        b.to_string()
    } else {
        a.to_string()
    }
}

/// One claim targeted for verification by the deep research tool's
/// verification gate, plus the material fed into the fresh analyst's task.
pub(crate) struct VerificationTarget {
    pub(crate) claim: String,
    pub(crate) sources: String,
    pub(crate) contradictions: String,
}

impl VerificationTarget {
    pub(crate) fn new(claim: &str, sources: &str, contradictions: &str) -> Self {
        Self {
            claim: claim.to_string(),
            sources: sources.to_string(),
            contradictions: contradictions.to_string(),
        }
    }
}

/// Extract search/web_search query strings + tool-call counts from an
/// agent's session history (for the query ledger and the run summary).
/// Shared with the deep research tool (`research.rs`).
pub(crate) fn extract_query_telemetry(agent: &Agent) -> (usize, usize, Vec<String>) {
    let mut tool_calls = 0usize;
    let mut searches = 0usize;
    let mut queries = Vec::new();
    for msg in agent.session.history() {
        let Some(decoded) = crate::session::decode_native_history_message(msg) else {
            continue;
        };
        let crate::session::DecodedNativeHistoryMessage::Assistant {
            tool_calls: Some(calls),
            ..
        } = decoded
        else {
            continue;
        };
        tool_calls += calls.len();
        for call in calls {
            if matches!(call.name.as_str(), "web_search" | "search") {
                searches += 1;
                if let Some(q) = call.arguments.get("query").and_then(|v| v.as_str()) {
                    queries.push(q.to_string());
                }
            }
        }
    }
    (tool_calls, searches, queries)
}

/// Dispatch one fresh Analyst per verification target (bounded by
/// [`VERIFY_MAX_ANALYSTS`]) in parallel. `id_prefix` seeds the per-target
/// agent IDs; `task_extra` is appended to every task (e.g. the query
/// ledger). Shared with the deep research tool's verification gate.
pub(crate) async fn dispatch_verifiers(
    ws: &Workspace,
    id_prefix: &str,
    targets: &[VerificationTarget],
    task_extra: &str,
) -> Vec<VerificationResult> {
    let task_template = load_prompt("ask/verify.md");
    let extraction_prompt = load_prompt("extraction/verify.md");
    let suffix = crate::generate_suffix();
    let futures: Vec<_> = targets
        .iter()
        .take(VERIFY_MAX_ANALYSTS)
        .enumerate()
        .map(|(i, t)| {
            let ws = ws.clone();
            let agent_id = format!("{id_prefix}_{suffix}_{i}");
            let mut task = substitute(
                &task_template,
                &[
                    ("{{claim}}", &t.claim),
                    ("{{sources}}", &t.sources),
                    ("{{contradictions}}", &t.contradictions),
                ],
            );
            task.push_str(task_extra);
            let extraction_prompt = extraction_prompt.clone();
            let claim_text = t.claim.clone();
            async move {
                run_claim_verifier(&ws, &agent_id, &claim_text, &task, &extraction_prompt).await
            }
        })
        .collect();
    join_all(futures).await
}

/// Run one claim verifier: a fresh Analyst researches the claim and returns
/// its structured verdict. Any failure yields "unresolved" — fail-open.
async fn run_claim_verifier(
    ws: &Workspace,
    agent_id: &str,
    target_claim: &str,
    task: &str,
    extraction_prompt: &str,
) -> VerificationResult {
    let (agent, response) = run_agent(
        agent_id.to_string(),
        Role::Analyst,
        ws,
        None,
        task,
        String::new(),
        String::new(),
        None,
    )
    .await;
    let (tool_calls, searches, queries) = extract_query_telemetry(&agent);
    let Some(raw) = response else {
        return VerificationResult {
            claim: target_claim.to_string(),
            verdict: "unresolved".to_string(),
            evidence: "verifier failed to produce a response".to_string(),
            tool_calls,
            searches,
            queries,
        };
    };
    if raw.trim().is_empty() {
        return VerificationResult {
            claim: target_claim.to_string(),
            verdict: "unresolved".to_string(),
            evidence: "verifier produced an empty response".to_string(),
            tool_calls,
            searches,
            queries,
        };
    }
    match agent
        .extract_verdict::<VerificationVerdict>(
            extraction_prompt,
            Some(&validate_verification_verdict),
        )
        .await
    {
        Ok(v) => VerificationResult {
            // Key by the target claim (not the verifier's possibly-rephrased
            // restatement) so report reconciliation matches exactly.
            claim: target_claim.to_string(),
            verdict: v.verdict,
            evidence: v.evidence,
            tool_calls,
            searches,
            queries,
        },
        Err(e) => VerificationResult {
            claim: target_claim.to_string(),
            verdict: "unresolved".to_string(),
            evidence: format!("verification extraction failed: {e}"),
            tool_calls,
            searches,
            queries,
        },
    }
}

/// Escape triple-backtick code fences to prevent markdown-structure
/// corruption in the consolidated output.
pub(crate) fn escape_fences(s: &str) -> String {
    s.replace("```", "\\`\\`\\`")
}

/// Render the fail-open raw dump: the raw response of every valid analyst
/// (extraction success or failure) in `### Report from Analyst N` sections.
/// No-response analysts render with their reason so nothing is silently lost.
fn render_raw_analyst_dump(outcomes: &[AnalystOutcome]) -> String {
    let mut sections = Vec::new();
    for (i, o) in outcomes.iter().enumerate() {
        match o {
            AnalystOutcome::Findings { raw, .. } | AnalystOutcome::ParseFailed { raw, .. } => {
                sections.push(format!(
                    "### Report from Analyst {}\n{}",
                    i + 1,
                    escape_fences(raw)
                ));
            }
            AnalystOutcome::NoResponse(reason) => {
                sections.push(format!(
                    "### Report from Analyst {}\nno response: {reason}",
                    i + 1
                ));
            }
        }
    }
    sections.join("\n\n")
}

/// Render extraction-failure reasons for the fail-open marker (which analysts
/// produced no parseable findings and why).
#[must_use]
fn render_extraction_failures(outcomes: &[AnalystOutcome]) -> String {
    let mut out = String::new();
    for (i, o) in outcomes.iter().enumerate() {
        match o {
            AnalystOutcome::ParseFailed { failure, .. } => {
                let _ = writeln!(
                    out,
                    "- Analyst {}: findings extraction failed ({failure})",
                    i + 1
                );
            }
            AnalystOutcome::NoResponse(reason) => {
                let _ = writeln!(out, "- Analyst {}: no response ({reason})", i + 1);
            }
            AnalystOutcome::Findings { .. } => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{FakeProvider, install_fake_provider, retry_tests_lock};
    use crate::workspace::test_ws;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_ask_missing_args() {
        let tool = AskTool::new(
            vec![Role::Analyst, Role::Coder, Role::Qa],
            DispatchMode::Sync,
            Role::Engineer,
        );
        let ws = test_ws("/tmp/test_ws");

        // Missing role
        let result = tool.execute(&ws, json!({"ask": "do something"})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: role"),
            "Should mention missing role"
        );

        // Missing ask
        let result = tool.execute(&ws, json!({"role": "analyst"})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: ask"),
            "Should mention missing ask"
        );
    }

    #[tokio::test]
    async fn test_ask_unsupported_role() {
        let tool = AskTool::new(vec![Role::Analyst], DispatchMode::Sync, Role::Engineer);
        let ws = test_ws("/tmp/test_ws");
        // "manager" is a valid Role but not one that AskTool can delegate to
        let args = json!({"role": "manager", "ask": "do something"});
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Invalid role should fail");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("Cannot delegate"),
            "Should mention cannot delegate: {err}"
        );
    }

    #[tokio::test]
    async fn test_ask_unknown_role() {
        let tool = AskTool::new(vec![], DispatchMode::Sync, Role::Engineer);
        let ws = test_ws("/tmp/test_ws");
        // Truly unknown role string — returns bail!
        let args = json!({"role": "nonexistent", "ask": "do something"});
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Unknown role should fail");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("Unknown role") || err.contains("nonexistent"),
            "Should mention 'Unknown role': {err}"
        );
        assert!(
            !err.contains("sage") && !err.contains("discovery") && !err.contains("maintainer"),
            "Error message should not leak internal role names: {err}"
        );
    }

    /// Tests that ask dispatches correctly to each supported role.
    /// Requires an LLM provider to be configured.
    #[tokio::test]
    #[ignore = "requires LLM provider"]
    async fn test_ask_all_roles() {
        struct Case {
            role: &'static str,
            ask: &'static str,
        }

        let cases = [
            Case {
                role: "analyst",
                ask: "Say 'hello analyst' and nothing else.",
            },
            Case {
                role: "coder",
                ask: "Say 'hello coder' and nothing else.",
            },
            Case {
                role: "qa",
                ask: "Say 'hello qa' and nothing else.",
            },
        ];

        for c in &cases {
            let tool = AskTool::new(
                vec![Role::Analyst, Role::Coder, Role::Qa],
                DispatchMode::Sync,
                Role::Engineer,
            );
            let ws = test_ws("/tmp/test_ws");
            let args = json!({"role": c.role, "ask": c.ask});
            let result = tool.execute(&ws, args).await.expect("execute");
            assert!(
                result.contains("hello"),
                "{} output should contain hello",
                c.role
            );
        }
    }

    // ── Consolidation edge-case tests (no LLM provider needed) ──

    /// Build a bare analyst run for pipeline tests: a real `Agent` (no stores
    /// touched) plus an optional raw response.
    fn test_analyst_run(ws: &Workspace, response: Option<&str>) -> (Agent, Option<String>) {
        let agent = Agent::new(
            format!("ask_test_{}", crate::generate_suffix()),
            Role::Analyst,
            ws,
            None,
            String::new(),
            String::new(),
        );
        (agent, response.map(ToString::to_string))
    }

    fn findings(claims: Vec<(&str, &str, &str)>) -> AnalystFindings {
        AnalystFindings {
            claims: claims
                .into_iter()
                .map(|(claim, source, confidence)| Claim {
                    claim: claim.to_string(),
                    source: source.to_string(),
                    confidence: confidence.to_string(),
                    contradictions: Vec::new(),
                })
                .collect(),
            coverage: Vec::new(),
            unanswered: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_consolidate_zero_responses_returns_error() {
        let ws = test_ws("/tmp/test_ws");
        let runs = vec![
            test_analyst_run(&ws, None),
            test_analyst_run(&ws, None),
            test_analyst_run(&ws, None),
        ];
        let result = consolidate_analyst_runs(&ws, "test question", runs).await;
        assert!(result.is_err(), "0 responses should error");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("All parallel analysts failed"),
            "Error should mention analyst failure"
        );
    }

    #[tokio::test]
    async fn test_consolidate_one_response_returned_directly() {
        let ws = test_ws("/tmp/test_ws");
        let runs = vec![
            test_analyst_run(&ws, Some("only answer")),
            test_analyst_run(&ws, None),
            test_analyst_run(&ws, None),
        ];
        let result = consolidate_analyst_runs(&ws, "test question", runs).await;
        assert!(result.is_ok(), "1 response should succeed");
        assert_eq!(
            result.unwrap(),
            "only answer",
            "Should return the single response directly without consolidation"
        );
    }

    #[tokio::test]
    async fn test_consolidate_empty_responses_filtered() {
        // Empty/whitespace-only strings should be filtered the same as None
        let ws = test_ws("/tmp/test_ws");
        let runs = vec![
            test_analyst_run(&ws, Some("valid")),
            test_analyst_run(&ws, Some("")),
            test_analyst_run(&ws, Some("   ")),
        ];
        let result = consolidate_analyst_runs(&ws, "test question", runs).await;
        assert!(
            result.is_ok(),
            "1 valid after filtering empty should succeed"
        );
        assert_eq!(
            result.unwrap(),
            "valid",
            "Should return the only non-empty response"
        );
    }

    // ── Ask grouping renderer ────────────────────────────────────────────

    #[test]
    fn render_ask_groups_computes_brackets_from_distinct_agents() {
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![
                    ("alpha is true", "url1", "high"),
                    ("beta is true", "url2", "medium"),
                ]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![
                    ("alpha is true", "url1", "high"),
                    ("gamma is true", "url3", "low"),
                ]),
            },
            AnalystOutcome::NoResponse("analyst produced no response".into()),
        ];
        // Global flat ids: 0 = a0 "alpha is true", 1 = a0 "beta is true",
        // 2 = a1 "alpha is true", 3 = a1 "gamma is true".
        let items: Vec<Vec<String>> = outcomes
            .iter()
            .map(|o| match o {
                AnalystOutcome::Findings { findings, .. } => {
                    findings.claims.iter().map(|c| c.claim.clone()).collect()
                }
                _ => Vec::new(),
            })
            .collect();
        let table = crate::consensus::ItemTable::new(&items);
        let output = crate::consensus::GroupingOutput {
            summary: "Two facts, one solo finding.".into(),
            groups: vec![
                crate::consensus::GroupingGroup {
                    heading: "Alpha".into(),
                    contradiction: false,
                    members: vec![
                        crate::consensus::GroupingMember { id: 0 },
                        crate::consensus::GroupingMember { id: 2 },
                    ],
                },
                crate::consensus::GroupingGroup {
                    heading: "Beta".into(),
                    contradiction: false,
                    members: vec![crate::consensus::GroupingMember { id: 1 }],
                },
            ],
            ungrouped: vec![crate::consensus::GroupingMember { id: 3 }],
        };
        let text = render_ask_groups("q", &output, &[], &table, 2, &outcomes);
        assert!(
            text.contains("**Alpha** [2/2]"),
            "consensus group renders [2/2] from distinct cited agents: {text}"
        );
        assert!(
            text.contains("**Beta** [1/2]"),
            "solo group renders [1/2] without DISPUTED: {text}"
        );
        assert!(
            text.contains("**Ungrouped**"),
            "ungrouped list renders: {text}"
        );
        assert!(
            text.contains("- Agent 1: gamma is true"),
            "ungrouped member attributes its source: {text}"
        );
        assert!(
            !text.contains("DISPUTED"),
            "no contradiction flag means no DISPUTED anywhere: {text}"
        );
    }

    #[test]
    fn render_ask_groups_disputed_only_on_contradiction() {
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![("alpha is false", "url2", "high")]),
            },
        ];
        let items: Vec<Vec<String>> = outcomes
            .iter()
            .map(|o| match o {
                AnalystOutcome::Findings { findings, .. } => {
                    findings.claims.iter().map(|c| c.claim.clone()).collect()
                }
                _ => Vec::new(),
            })
            .collect();
        let table = crate::consensus::ItemTable::new(&items);
        let output = crate::consensus::GroupingOutput {
            summary: "Agents disagree on alpha.".into(),
            groups: vec![crate::consensus::GroupingGroup {
                heading: "Alpha".into(),
                contradiction: true,
                members: vec![
                    crate::consensus::GroupingMember { id: 0 },
                    crate::consensus::GroupingMember { id: 1 },
                ],
            }],
            ungrouped: vec![],
        };
        let text = render_ask_groups("q", &output, &[], &table, 2, &outcomes);
        assert!(
            text.contains("**Alpha** [2/2 · DISPUTED]"),
            "contradiction group renders [2/2 · DISPUTED]: {text}"
        );
    }

    #[test]
    fn render_ask_groups_surfaces_member_caveats_as_metadata() {
        let mut claims = findings(vec![("alpha is true", "url1", "high")]);
        claims.claims[0].contradictions = vec!["source B says alpha may be false".into()];
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: claims,
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
        ];
        let items: Vec<Vec<String>> = outcomes
            .iter()
            .map(|o| match o {
                AnalystOutcome::Findings { findings, .. } => {
                    findings.claims.iter().map(|c| c.claim.clone()).collect()
                }
                _ => Vec::new(),
            })
            .collect();
        let table = crate::consensus::ItemTable::new(&items);
        let output = crate::consensus::GroupingOutput {
            summary: "alpha is agreed.".into(),
            groups: vec![crate::consensus::GroupingGroup {
                heading: "Alpha".into(),
                contradiction: false,
                members: vec![
                    crate::consensus::GroupingMember { id: 0 },
                    crate::consensus::GroupingMember { id: 1 },
                ],
            }],
            ungrouped: vec![],
        };
        let text = render_ask_groups("q", &output, &[], &table, 2, &outcomes);
        assert!(
            text.contains("caveat: source B says alpha may be false"),
            "self-reported caveat renders as member metadata: {text}"
        );
        assert!(
            !text.contains("DISPUTED"),
            "a self-reported caveat is NOT a contradiction: {text}"
        );
    }

    // ── consolidation fail-open ────────────────────────────────────────

    /// Helper: run consolidation over the given extracted outcomes with the
    /// given fake provider outcomes scripted.
    async fn consolidate_with_script(
        outcomes: Vec<AnalystOutcome>,
        fake: FakeProvider,
    ) -> anyhow::Result<String> {
        let provider: Arc<dyn crate::Provider> = Arc::new(fake);
        let _guard = install_fake_provider(provider);
        let ws = test_ws("/tmp/test_ws");
        consolidate_findings(&ws, "test question", outcomes).await
    }

    /// Two analysts agreeing on one claim (2/3, no contradictions).
    fn agreed_outcomes(raw_a: &str, raw_b: &str) -> Vec<AnalystOutcome> {
        vec![
            AnalystOutcome::Findings {
                raw: raw_a.to_string(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::Findings {
                raw: raw_b.to_string(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::NoResponse("analyst produced no response".into()),
        ]
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_fail_open_delivers_raw_reports() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Synthesis retries (transport failures) exhaust → fail open.
        let fake = FakeProvider::new()
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            );
        let result =
            consolidate_with_script(agreed_outcomes("report one", "report two"), fake).await;
        let text = result.expect("fail-open must succeed with raw reports");
        assert!(
            text.contains("unconsolidated — consolidation failed"),
            "must carry the unconsolidated marker: {text}"
        );
        assert!(text.contains("## Analyst Reports"), "{text}");
        assert!(text.contains("### Report from Analyst 1"), "{text}");
        assert!(text.contains("report one"), "{text}");
        assert!(text.contains("### Report from Analyst 2"), "{text}");
        assert!(text.contains("report two"), "{text}");
        assert!(
            text.contains("no response: analyst produced no response"),
            "no-response analyst renders its reason — nothing silently lost: {text}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_non_retryable_fails_open() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Even an immediate non-retryable error fails open.
        let fake = FakeProvider::new().err(
            crate::retry::FailureClass::NonRetryable,
            "insufficient balance",
        );
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "alpha".into(),
                findings: findings(vec![("a", "u1", "low")]),
            },
            AnalystOutcome::Findings {
                raw: "beta".into(),
                findings: findings(vec![("a", "u1", "low")]),
            },
            AnalystOutcome::Findings {
                raw: "gamma".into(),
                findings: findings(vec![("a", "u1", "low")]),
            },
        ];
        let result = consolidate_with_script(outcomes, fake).await;
        let text = result.expect("non-retryable consolidation failure fails open");
        assert!(
            text.contains("unconsolidated — consolidation failed"),
            "{text}"
        );
        assert!(text.contains("alpha"), "{text}");
        assert!(text.contains("beta"), "{text}");
        assert!(text.contains("gamma"), "{text}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_success_synthesizes() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // The grouping pass parses a strict id-based GroupingOutput; a
        // faithful model groups the two agreed claims and leaves nothing
        // ungrouped.
        let fake = FakeProvider::new().ok(
            r#"{"summary":"alpha is true.","groups":[{"heading":"Alpha","contradiction":false,"members":[{"id":0},{"id":1}]}],"ungrouped":[]}"#,
        );
        let result = consolidate_with_script(agreed_outcomes("report a", "report b"), fake).await;
        let text = result.expect("success");
        assert!(text.contains("alpha is true"), "{text}");
        assert!(
            !text.contains("unconsolidated"),
            "successful consolidation must not hit the fail-open marker: {text}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_single_parseable_source_skips_grouping() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // ≥2 valid responses but only one produced parseable claims: a
        // grouping pass over a single source can't yield agreement brackets
        // or contradictions — the flat claim list + raw dumps are delivered
        // without calling the provider.
        let fake = FakeProvider::new(); // no script — any provider call would fail
        let provider: Arc<dyn crate::Provider> = Arc::new(fake);
        let _provider_guard = install_fake_provider(provider);
        let ws = test_ws("/tmp/test_ws");
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "sole report".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::ParseFailed {
                raw: "unparseable report".into(),
                failure: "parse failed".into(),
            },
        ];
        let result = consolidate_findings(&ws, "test question", outcomes).await;
        let text = result.expect("single parseable source succeeds");
        assert!(
            text.contains("only one analyst produced parseable claims — grouping skipped"),
            "skip marker must be present: {text}"
        );
        assert!(text.contains("alpha is true"), "{text}");
        assert!(text.contains("### Report from Analyst 2"), "{text}");
        assert!(text.contains("unparseable report"), "{text}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_fail_open_mixed_failure_classes() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Mixed transport + empty-response failures exercise both failure
        // arms of the grouping retry loop; exhaustion still fails open.
        // Script: transport error, then two empty responses.
        let fake = FakeProvider::new()
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .ok("")
            .ok("");
        let result = consolidate_with_script(agreed_outcomes("only usable", "second"), fake).await;
        let text = result.expect("mixed failures still fail open");
        assert!(
            text.contains("unconsolidated — consolidation failed"),
            "{text}"
        );
        assert!(text.contains("only usable"), "{text}");
        assert!(text.contains("second"), "{text}");
    }

    /// Extraction fails for every valid analyst → the raw reports are
    /// delivered with the extraction-failure marker (fail-open at the
    /// extraction step, not the synthesis step).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_extraction_fail_open_delivers_raw_reports() {
        let _lock = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = FakeProvider::new()
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            );
        let provider: Arc<dyn crate::Provider> = Arc::new(fake);
        let _provider_guard = install_fake_provider(provider);
        let ws = test_ws("/tmp/test_ws");
        let runs = vec![
            test_analyst_run(&ws, Some("report one")),
            test_analyst_run(&ws, Some("report two")),
            test_analyst_run(&ws, None),
        ];
        let result = consolidate_analyst_runs(&ws, "test question", runs).await;
        let text = result.expect("extraction fail-open must deliver raw reports");
        assert!(
            text.contains("unconsolidated — findings extraction failed"),
            "{text}"
        );
        assert!(text.contains("### Report from Analyst 1"), "{text}");
        assert!(text.contains("report one"), "{text}");
        assert!(text.contains("### Report from Analyst 2"), "{text}");
        assert!(text.contains("report two"), "{text}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_async_envelope_carries_marker() {
        // The async dispatch path (tokio::spawn in AskTool::execute) builds
        // its envelope via build_async_ask_message — this test drives the REAL
        // fail-open consolidation result through that production builder
        // (not a manual re-wrap), asserting the exact envelope + marker shape
        // that reaches the caller's agent channel.
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = FakeProvider::new()
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down");
        let result =
            consolidate_with_script(agreed_outcomes("raw report", "raw report 2"), fake).await;
        let envelope = build_async_ask_message(result);
        assert!(envelope.contains("<ask-tool-result>"), "{envelope}");
        assert!(
            envelope.contains("unconsolidated — consolidation failed"),
            "{envelope}"
        );
        assert!(envelope.contains("raw report"), "{envelope}");
        assert!(
            envelope.ends_with("</ask-tool-result>"),
            "envelope must close: {envelope}"
        );
    }

    #[tokio::test]
    async fn async_envelope_wraps_sub_agent_errors() {
        // The async dispatch path's error branch: a failed sub-agent is
        // wrapped in the same envelope with the error text.
        let envelope = build_async_ask_message(Err(anyhow::anyhow!("sub-agent exploded")));
        assert!(envelope.contains("<ask-tool-result>"), "{envelope}");
        assert!(
            envelope.contains("An error occurred: sub-agent exploded"),
            "{envelope}"
        );
        assert!(
            envelope.ends_with("</ask-tool-result>"),
            "envelope must close: {envelope}"
        );
    }

    /// The sync ask path truncates tool output via the 5 KB sandwich (head
    /// 2/3 + tail 1/3) — the fail-open marker is head-placed so it survives
    /// even when the consolidated output overflows the budget.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn fail_open_marker_survives_sync_truncation() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = FakeProvider::new()
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down");
        // A long raw report guarantees the output overflows the 5 KB budget.
        let long_report = "lorem ipsum ".repeat(2_000);
        let result = consolidate_with_script(agreed_outcomes(&long_report, "second"), fake).await;
        let text = result.expect("fail-open must succeed");
        // Simulate the sync path's tool-output truncation (Tool::format_output).
        let truncated = crate::util::truncate_tool_output(&text);
        assert!(
            truncated.contains("unconsolidated — consolidation failed"),
            "head-placed marker must survive 5 KB sandwich truncation: {truncated}"
        );
    }

    #[test]
    fn render_ask_groups_renders_disputed_cross_references() {
        // A remainder item that contradicts a frozen group renders with
        // DISPUTED + a code-computed cross-reference (id-resolved — no text
        // equality).
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![("safe", "url1", "high")]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![("actually unsafe", "url2", "high")]),
            },
        ];
        let items: Vec<Vec<String>> = outcomes
            .iter()
            .map(|o| match o {
                AnalystOutcome::Findings { findings, .. } => {
                    findings.claims.iter().map(|c| c.claim.clone()).collect()
                }
                _ => Vec::new(),
            })
            .collect();
        let table = crate::consensus::ItemTable::new(&items);
        let output = crate::consensus::GroupingOutput {
            summary: "One consensus, one dispute.".into(),
            groups: vec![crate::consensus::GroupingGroup {
                heading: "Safety".into(),
                contradiction: false,
                members: vec![crate::consensus::GroupingMember { id: 0 }],
            }],
            ungrouped: vec![crate::consensus::GroupingMember { id: 1 }],
        };
        let references = vec![crate::consensus::GroupingReference {
            group: 0,
            member: crate::consensus::GroupingMember { id: 1 },
        }];
        let text = render_ask_groups("q", &output, &references, &table, 2, &outcomes);
        assert!(
            text.contains("Agent 1: actually unsafe [DISPUTED — contradicts group 0 \"Safety\"]"),
            "reference must render with DISPUTED + cross-ref: {text}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_partial_success_with_remainder() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Round 1 freezes one group and leaves one item ungrouped; round 2's
        // delta leaves that item ungrouped → zero-progress stop. The frozen
        // group + deterministic remainder are delivered (partial success —
        // never the raw-dump fallback), and the round-1 summary is final.
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![
                    ("alpha is true", "url1", "high"),
                    ("beta is true", "url2", "medium"),
                ]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::NoResponse("analyst produced no response".into()),
        ];
        // ids: 0 = a0 "alpha is true", 1 = a0 "beta is true", 2 = a1 "alpha is true".
        let fake = FakeProvider::new()
            .ok(
                r#"{"summary":"alpha agreed, beta solo.","groups":[{"heading":"Alpha","contradiction":false,"members":[{"id":0},{"id":2}]}],"ungrouped":[{"id":1}]}"#,
            )
            .ok(r#"{"groups":[],"ungrouped":[{"id":1}]}"#);
        let result = consolidate_with_script(outcomes, fake).await;
        let text = result.expect("partial success must be delivered");
        assert!(
            text.contains("**Alpha** [2/2]"),
            "frozen group with code-computed bracket renders: {text}"
        );
        assert!(
            text.contains("**Ungrouped**") && text.contains("- Agent 0: beta is true"),
            "deterministic remainder renders in the ungrouped section: {text}"
        );
        assert!(
            text.contains("alpha agreed, beta solo."),
            "round-1 summary is final: {text}"
        );
        assert!(
            !text.contains("unconsolidated"),
            "partial success replaces the raw-dump fallback: {text}"
        );
    }
}
