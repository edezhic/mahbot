//! AskTool — spawns a sub-agent to ask a question.
//!
//! Available to the Engineer and Maintainer agents (sync mode), and to the
//! Manager and Assistant agents (async mode). In sync mode the caller blocks
//! until the sub-agent completes. In async mode the sub-agent is dispatched
//! in a background task and the result is injected back to the caller's
//! agent channel via [`crate::message_router::route`].
//!
//! Analyst batches run three decorrelated analysts (distinct research angles)
//! that report structured claim-level findings; consolidation merges at claim
//! level, grades per-claim agreement, and runs exactly one targeted
//! verification pass for disputed claims or surfaced contradictions.
//! Fail-open is preserved throughout: findings are never silently lost.

use crate::agent::run_agent;
use crate::config::CONFIG;
use crate::message_router::{self, AgentJob, JobKind};
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::session::{ask_agent_id, resolve_agent_id};
use crate::tools::Tool;
use crate::{
    Agent, ChatMessage, ChatRequest, ChatRequestMeta, DEFAULT_MAX_TOKENS, Role, Workspace,
};
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

/// A claim merged across analysts with per-claim agreement grading.
#[derive(Debug, Clone)]
struct MergedClaim {
    claim: String,
    /// Indices of the analysts (0-based) that stated this claim.
    analysts: Vec<usize>,
    sources: Vec<String>,
    contradictions: Vec<String>,
    confidence: String,
}

impl MergedClaim {
    fn agreement(&self) -> usize {
        self.analysts.len()
    }

    /// Disputed claims (1/3 agreement) or claims with surfaced contradictions
    /// trigger the targeted verification pass.
    fn is_disputed(&self) -> bool {
        self.agreement() == 1 || !self.contradictions.is_empty()
    }
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

/// Cap on verification analysts spawned per ask (exactly one pass, bounded).
/// Also used by the deep research tool's verification gate.
pub(crate) const VERIFY_MAX_ANALYSTS: usize = 4;

/// Embedding similarity for claim-level merge (top of the 0.85–0.90 band —
/// conservative so only clearly-equivalent claims merge into one graded claim).
pub(crate) const CLAIM_MERGE_SIMILARITY_THRESHOLD: f32 = 0.90;

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
/// ≥2 valid → extract findings, merge at claim level, verify, synthesize.
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

/// Merge findings at claim level and grade per-claim agreement (3/3, 2/3,
/// 1/3). Claims are matched by [`ClaimKey`] — exact-normalized text first,
/// then embedding similarity (numeric literals must not disagree), so
/// similarly phrased claims from decorrelated analysts grade as agreement
/// instead of fragmenting into artificial disputes. Contradictions are
/// surfaced, never resolved by fiat.
fn merge_and_grade(outcomes: &[AnalystOutcome]) -> (Vec<MergedClaim>, Vec<String>, Vec<String>) {
    let mut coverage: Vec<String> = Vec::new();
    let mut unanswered: Vec<String> = Vec::new();
    let mut merged: Vec<(MergedClaim, ClaimKey)> = Vec::new();
    for (i, outcome) in outcomes.iter().enumerate() {
        let AnalystOutcome::Findings { findings, .. } = outcome else {
            continue;
        };
        coverage.extend(findings.coverage.iter().cloned());
        unanswered.extend(findings.unanswered.iter().cloned());
        for c in &findings.claims {
            let key = ClaimKey::new(&c.claim);
            if let Some((g, _)) = merged
                .iter_mut()
                .find(|(_, k)| key.equivalent_to(k, CLAIM_MERGE_SIMILARITY_THRESHOLD))
            {
                if !g.analysts.contains(&i) {
                    g.analysts.push(i);
                }
                if !c.source.is_empty() && !g.sources.contains(&c.source) {
                    g.sources.push(c.source.clone());
                }
                for x in &c.contradictions {
                    if !g.contradictions.contains(x) {
                        g.contradictions.push(x.clone());
                    }
                }
                g.confidence = max_confidence(&g.confidence, &c.confidence);
            } else {
                merged.push((
                    MergedClaim {
                        claim: c.claim.trim().to_string(),
                        analysts: vec![i],
                        sources: if c.source.is_empty() {
                            Vec::new()
                        } else {
                            vec![c.source.clone()]
                        },
                        contradictions: c.contradictions.clone(),
                        confidence: c.confidence.clone(),
                    },
                    key,
                ));
            }
        }
    }
    merged.sort_by(|(a, _), (b, _)| {
        b.analysts
            .len()
            .cmp(&a.analysts.len())
            .then_with(|| a.claim.to_lowercase().cmp(&b.claim.to_lowercase()))
    });
    let merged: Vec<MergedClaim> = merged.into_iter().map(|(g, _)| g).collect();
    (
        merged,
        dedup_preserve_order(&coverage),
        dedup_preserve_order(&unanswered),
    )
}

pub(crate) fn normalize_claim(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Claim match key: normalized text, lazily-computed embedding, and numeric
/// literals. Exact-normalized equality wins (no embedding needed); otherwise
/// embedding similarity must clear the caller's threshold AND the numeric
/// literal sets must not disagree — "X costs $100 in 2024" and "X costs $200
/// in 2024" never merge into a single claim, so contradictions surface
/// instead of being averaged away. [`ClaimKey::equivalent_to`] short-circuits
/// both those cases before computing embeddings, so the common all-agree case
/// never touches the embedder. Shared with the deep research tool's novelty
/// check (`research.rs`).
#[derive(Debug)]
pub(crate) struct ClaimKey {
    norm: String,
    embedding: std::sync::OnceLock<Option<Vec<f32>>>,
    numeric: std::collections::HashSet<String>,
}

impl ClaimKey {
    pub(crate) fn new(s: &str) -> Self {
        Self {
            norm: normalize_claim(s),
            // Computed on first need — `equivalent_to` short-circuits exact
            // matches and numeric conflicts before calling this, so the
            // common all-agree case never touches the embedder. Test builds
            // skip the embedder entirely (it races tests: panic on unset
            // CONFIG, or spawn the download loop inside a tokio test
            // runtime) and use the exact-match fallback.
            embedding: std::sync::OnceLock::new(),
            numeric: s
                .split(|c: char| !c.is_ascii_digit())
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect(),
        }
    }

    fn embedding(&self) -> Option<&[f32]> {
        self.embedding
            .get_or_init(|| {
                #[cfg(test)]
                {
                    None
                }
                #[cfg(not(test))]
                {
                    crate::embedder::embed_query(&self.norm)
                }
            })
            .as_deref()
    }

    pub(crate) fn equivalent_to(&self, other: &ClaimKey, threshold: f32) -> bool {
        // Exact matches and numeric conflicts decide without embeddings —
        // short-circuit before touching the embedder (the common all-agree
        // case never initializes it).
        if let Some(decided) =
            equivalence_short_circuit(&self.norm, &self.numeric, &other.norm, &other.numeric)
        {
            return decided;
        }
        match (self.embedding(), other.embedding()) {
            (Some(a), Some(b)) => crate::vector::cosine_similarity(a, b) >= threshold,
            _ => false,
        }
    }
}

/// Pure short-circuit shared by [`ClaimKey::equivalent_to`] and
/// [`claims_equivalent`]: identical normalized text is equivalent; claims
/// that both carry numeric literals but different ones never merge (a shared
/// token like a year must not defeat the guard). `None` means the decision
/// needs embedding similarity.
fn equivalence_short_circuit(
    norm_a: &str,
    numbers_a: &std::collections::HashSet<String>,
    norm_b: &str,
    numbers_b: &std::collections::HashSet<String>,
) -> Option<bool> {
    if norm_a == norm_b {
        Some(true)
    } else if !numbers_a.is_empty() && !numbers_b.is_empty() && numbers_a != numbers_b {
        Some(false)
    } else {
        None
    }
}

/// Pure claim-equivalence decision, the tested oracle for
/// [`ClaimKey::equivalent_to`] (production goes through the method, which
/// short-circuits before computing embeddings). Exact-normalized text wins;
/// otherwise claims that both carry numeric literals but different ones
/// never merge (contradictions surface instead of being averaged away) and
/// embedding similarity must clear the threshold. Without embeddings the
/// check degrades to exact-match-only.
#[cfg(test)]
pub(crate) fn claims_equivalent(
    norm_a: &str,
    numbers_a: &std::collections::HashSet<String>,
    emb_a: Option<&[f32]>,
    norm_b: &str,
    numbers_b: &std::collections::HashSet<String>,
    emb_b: Option<&[f32]>,
    threshold: f32,
) -> bool {
    if let Some(decided) = equivalence_short_circuit(norm_a, numbers_a, norm_b, numbers_b) {
        return decided;
    }
    match (emb_a, emb_b) {
        (Some(a), Some(b)) => crate::vector::cosine_similarity(a, b) >= threshold,
        _ => false,
    }
}

fn dedup_preserve_order(items: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    items
        .iter()
        .filter(|&s| seen.insert(s.clone()))
        .cloned()
        .collect()
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

/// Consolidate extracted outcomes: merge at claim level, run exactly one
/// verification pass for disputed claims, then synthesize the final report.
/// Fail-open at every step — findings are never silently lost.
async fn consolidate_findings(
    ws: &Workspace,
    ask: &str,
    outcomes: Vec<AnalystOutcome>,
) -> Result<String> {
    let (merged, coverage, unanswered) = merge_and_grade(&outcomes);
    if merged.is_empty() {
        // All valid responses failed extraction — fail open with raw reports.
        let raw_dump = render_raw_analyst_dump(&outcomes);
        return Ok(format!(
            "## Analyst Reports\n\n(unconsolidated — findings extraction failed)\n\n{raw_dump}"
        ));
    }
    let verification = run_verification_pass(ws, &merged).await;
    synthesize_claim_report(
        ws,
        ask,
        &merged,
        &coverage,
        &unanswered,
        &verification,
        &outcomes,
    )
    .await
}

/// Run exactly one targeted verification pass: a fresh analyst per disputed
/// claim (1/3 agreement or surfaced contradictions), bounded by
/// [`VERIFY_MAX_ANALYSTS`]. No recursion — anything still disputed after the
/// pass is marked unresolved in the final report.
async fn run_verification_pass(ws: &Workspace, merged: &[MergedClaim]) -> Vec<VerificationResult> {
    let disputed: Vec<VerificationTarget> = merged
        .iter()
        .filter(|c| c.is_disputed())
        .map(|c| {
            VerificationTarget::new(
                &c.claim,
                &c.sources.join("; "),
                &c.contradictions.join("; "),
            )
        })
        .collect();
    if disputed.is_empty() {
        return Vec::new();
    }
    dispatch_verifiers(ws, &format!("ask_{}_verify", ws.name), &disputed, "").await
}

/// One claim targeted for verification plus the material fed into the fresh
/// analyst's task.
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

/// One-shot LLM synthesis over the deterministic claim-level report.
///
/// The synthesis input is the claim report (per-claim agreement grades,
/// surfaced contradictions, verification results) — the LLM writes the final
/// prose answer but cannot silently average away graded disagreements.
/// Fail-open: synthesis failure delivers the claim report plus raw analyst
/// reports with an explicit marker — findings are never lost.
#[allow(clippy::too_many_arguments)]
async fn synthesize_claim_report(
    ws: &Workspace,
    ask: &str,
    merged: &[MergedClaim],
    coverage: &[String],
    unanswered: &[String],
    verification: &[VerificationResult],
    outcomes: &[AnalystOutcome],
) -> Result<String> {
    let model = CONFIG.role_model(Role::Analyst);
    let routing = CONFIG.model_routing(&model);
    let prompt_template = load_prompt("consolidate/analyst.md");

    // System messages: general workspace context first, then the filled
    // template (instructions + original_ask only).
    let mut messages = Vec::with_capacity(3);
    crate::prompt::prepend_general_context(&mut messages, ws).await;
    messages.push(ChatMessage::system(substitute(
        &prompt_template,
        &[("{{original_ask}}", ask)],
    )));

    // User message = deterministic claim-level report only.
    let claim_report =
        render_claim_report(ask, merged, coverage, unanswered, verification, outcomes);
    let user_content = format!("## Claim-level Findings\n\n{claim_report}");
    messages.push(ChatMessage::user(&user_content));

    let request = ChatRequest {
        messages,
        tools: None,
        model,
        allow_image_parts: false,
        max_tokens: Some(DEFAULT_MAX_TOKENS),
        reasoning_effort: Some(CONFIG.role_reasoning_effort(Role::Analyst)),
        provider_order: routing.provider_order,
        provider_allow_fallbacks: routing.allow_fallbacks,
        response_format_json_object: false,
        meta: Some(ChatRequestMeta {
            purpose: "consolidate",
            agent_id: format!("ask_{}_consolidation", ws.name),
            role: Role::Analyst.as_str().to_string(),
            workspace: ws.name.clone(),
            ticket_id: None,
        }),
    };

    let policy = crate::retry::RetryPolicy::current();
    match crate::retry::retry_chat(request, &policy).await {
        // retry_chat only returns Ok with non-empty (trimmed) text —
        // empty responses are classified as NoResponse and retried.
        Ok(response) => Ok(response.text_or_empty().to_string()),
        Err(exhausted) => {
            // Fail-open: deliver the deterministic claim report plus the raw
            // valid analyst reports with an explicit marker instead of
            // discarding them. Findings are never lost.
            tracing::warn!(
                error = %exhausted,
                "Analyst consolidation failed — delivering raw claim report"
            );
            let raw_dump = render_raw_analyst_dump(outcomes);
            Ok(format!(
                "## Analyst Reports\n\n(unconsolidated — consolidation failed: {exhausted})\n\n{claim_report}\n\n{raw_dump}"
            ))
        }
    }
}

/// Wrap escaped analyst reports in the canonical markdown shape used by the
/// fail-open raw dump: `### Report from Analyst N` sections joined by blank
/// lines. Triple-backtick escaping is applied by the caller before this
/// function.
fn format_analyst_reports_markdown(escaped: &[String]) -> String {
    let mut parts = Vec::new();
    for (i, report) in escaped.iter().enumerate() {
        parts.push(format!("### Report from Analyst {}", i + 1));
        parts.push(report.clone());
    }
    parts.join("\n\n")
}

/// Escape triple-backtick code fences to prevent markdown-structure
/// corruption in the consolidated output.
pub(crate) fn escape_fences(s: &str) -> String {
    s.replace("```", "\\`\\`\\`")
}

/// Render the fail-open raw dump: the raw response of every valid analyst
/// (extraction success or failure) in `### Report from Analyst N` sections.
fn render_raw_analyst_dump(outcomes: &[AnalystOutcome]) -> String {
    let escaped: Vec<String> = outcomes
        .iter()
        .filter_map(|o| match o {
            AnalystOutcome::Findings { raw, .. } | AnalystOutcome::ParseFailed { raw, .. } => {
                Some(escape_fences(raw))
            }
            AnalystOutcome::NoResponse(_) => None,
        })
        .collect();
    format_analyst_reports_markdown(&escaped)
}

/// Render the deterministic claim-level report: agreement-graded claims,
/// surfaced contradictions, verification results, and explicitly marked
/// failed/unanswered items. This is both the synthesis input and the
/// fail-open deliverable — load-bearing content (summary, grades, unresolved
/// markers) sits at the top so it survives the sync-path 5 KB truncation.
#[allow(clippy::too_many_lines)]
fn render_claim_report(
    ask: &str,
    merged: &[MergedClaim],
    coverage: &[String],
    unanswered: &[String],
    verification: &[VerificationResult],
    outcomes: &[AnalystOutcome],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Research Findings");
    let _ = writeln!(out);
    let _ = writeln!(out, "**Question**: {ask}");
    let _ = writeln!(out);

    // Grade against the actual number of valid analyst responses (n=2 with a
    // unanimous claim renders as [2/2], never an unreachable [3/3] bracket).
    // A single valid response (total == 1) is never "unanimous" — every claim
    // there is disputed (agreement == 1) and still goes through verification.
    let total = outcomes
        .iter()
        .filter(|o| matches!(o, AnalystOutcome::Findings { .. }))
        .count();
    let full = merged
        .iter()
        .filter(|c| c.agreement() == total && total > 1 && !c.is_disputed())
        .count();
    let majority = merged
        .iter()
        .filter(|c| c.agreement() > 1 && c.agreement() < total && !c.is_disputed())
        .count();
    // Disputed = claims that triggered the verification pass: 1/{total}
    // agreement or surfaced contradictions (a majority-agreement claim with
    // contradictions is verified, so it is not labeled majority).
    let disputed = merged.iter().filter(|c| c.is_disputed()).count();
    let unresolved = verification
        .iter()
        .filter(|v| v.verdict == "unresolved")
        .count()
        + merged
            .iter()
            .filter(|c| {
                c.is_disputed()
                    && !verification
                        .iter()
                        .any(|v| normalize_claim(&v.claim) == normalize_claim(&c.claim))
            })
            .count();
    let _ = writeln!(
        out,
        "**Summary**: {} claims — {full} unanimous, {majority} majority, {disputed} disputed; {unresolved} unresolved after verification.",
        merged.len()
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "### Claims");
    for c in merged {
        // Disputed claims (1/n agreement, or any agreement level carrying
        // surfaced contradictions) render with the DISPUTED bracket — the
        // verification pass fires for them, so the grade can never say
        // "unanimous" or bare majority.
        let grade = if c.is_disputed() {
            format!("[{}/{} · DISPUTED]", c.agreement(), total)
        } else if c.agreement() == total && total > 1 {
            format!("[{total}/{total}]")
        } else {
            format!("[{}/{}]", c.agreement(), total)
        };
        let sources = if c.sources.is_empty() {
            "no source".to_string()
        } else {
            c.sources.join("; ")
        };
        let _ = writeln!(
            out,
            "- {grade} ({}) {} — source: {sources}",
            c.confidence,
            escape_fences(&c.claim),
        );
        if !c.contradictions.is_empty() {
            let _ = writeln!(out, "  - contradictions: {}", c.contradictions.join("; "));
        }
    }

    if !verification.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "### Verification");
        for v in verification {
            let _ = writeln!(
                out,
                "- {} → **{}** — {}",
                escape_fences(&v.claim),
                v.verdict,
                escape_fences(&v.evidence),
            );
        }
    }

    let has_failed = outcomes.iter().any(|o| {
        matches!(
            o,
            AnalystOutcome::ParseFailed { .. } | AnalystOutcome::NoResponse(_)
        )
    });
    if has_failed {
        let _ = writeln!(out);
        let _ = writeln!(out, "### Failed Analysts");
        for (i, o) in outcomes.iter().enumerate() {
            match o {
                AnalystOutcome::NoResponse(reason) => {
                    let _ = writeln!(out, "- Analyst {}: no response ({reason})", i + 1);
                }
                AnalystOutcome::ParseFailed { failure, .. } => {
                    let _ = writeln!(
                        out,
                        "- Analyst {}: findings extraction failed ({failure})",
                        i + 1
                    );
                }
                AnalystOutcome::Findings { .. } => {}
            }
        }
    }

    if !unanswered.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "### Unanswered Questions");
        for u in unanswered {
            let _ = writeln!(out, "- {}", escape_fences(u));
        }
    }

    if !coverage.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "### Coverage");
        for c in coverage {
            let _ = writeln!(out, "- {}", escape_fences(c));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{
        FakeProvider, init_test_stores, install_fake_provider, retry_tests_lock,
    };
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

    // ── Claim-level merge ────────────────────────────────────────────────

    #[test]
    fn test_claim_merge_grades_agreement() {
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
            AnalystOutcome::Findings {
                raw: "r3".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
        ];
        let (merged, _, _) = merge_and_grade(&outcomes);
        assert_eq!(merged.len(), 3);
        let alpha = merged
            .iter()
            .find(|c| c.claim == "alpha is true")
            .expect("alpha");
        assert_eq!(alpha.agreement(), 3);
        assert_eq!(alpha.analysts, vec![0, 1, 2]);
        assert!(
            !alpha.is_disputed(),
            "3/3 with no contradictions is not disputed"
        );
        let beta = merged
            .iter()
            .find(|c| c.claim == "beta is true")
            .expect("beta");
        assert_eq!(beta.agreement(), 1);
        assert!(beta.is_disputed(), "1/3 claim must be disputed");
        let gamma = merged
            .iter()
            .find(|c| c.claim == "gamma is true")
            .expect("gamma");
        assert_eq!(gamma.agreement(), 1);
    }

    #[test]
    fn test_claim_merge_surfaces_contradictions() {
        let mut findings_a = findings(vec![("alpha is true", "url1", "high")]);
        findings_a.claims[0].contradictions = vec!["source B says alpha is false".to_string()];
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings_a,
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::Findings {
                raw: "r3".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
        ];
        let (merged, _, _) = merge_and_grade(&outcomes);
        let alpha = merged
            .iter()
            .find(|c| c.claim == "alpha is true")
            .expect("alpha");
        // 3/3 agreement still disputes the claim because a contradiction surfaced.
        assert_eq!(alpha.agreement(), 3);
        assert!(alpha.is_disputed());
        assert_eq!(alpha.contradictions, vec!["source B says alpha is false"]);
    }

    #[test]
    fn test_claims_equivalent_rules() {
        let nums = |tokens: &[&str]| {
            tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<std::collections::HashSet<_>>()
        };
        // Exact-normalized text wins without embeddings.
        assert!(claims_equivalent(
            "x is true",
            &nums(&[]),
            None,
            "x is true",
            &nums(&[]),
            None,
            0.9
        ));
        // Semantically similar claims (different phrasing, similar vectors)
        // merge — the path that makes 2/3 and 3/3 agreement reachable.
        let similar = |a: &[f32], b: &[f32]| {
            claims_equivalent(
                "one phrasing",
                &nums(&[]),
                Some(a),
                "other phrasing",
                &nums(&[]),
                Some(b),
                0.9,
            )
        };
        assert!(similar(&[1.0, 0.0], &[0.95, 0.05]), "similar vectors merge");
        assert!(
            !similar(&[1.0, 0.0], &[0.0, 1.0]),
            "dissimilar vectors stay separate"
        );
        // Claims disagreeing on numbers never merge — a shared token (the
        // year) must not defeat the guard.
        let priced = |a: &[&str], b: &[&str]| {
            claims_equivalent(
                "alpha costs 100 in 2024",
                &nums(a),
                Some(&[1.0, 0.0][..]),
                "alpha costs 200 in 2024",
                &nums(b),
                Some(&[1.0, 0.0][..]),
                0.9,
            )
        };
        assert!(
            !priced(&["100", "2024"], &["200", "2024"]),
            "price differs → never merge"
        );
        assert!(
            priced(&["100", "2024"], &["100", "2024"]),
            "identical numeric content merges"
        );
        // One-sided numbers defer to embedding — the strings match the sets
        // (no price contradiction is being merged).
        let one_sided = |a: &[&str], b: &[&str]| {
            claims_equivalent(
                "alpha costs 100 in 2024",
                &nums(a),
                Some(&[1.0, 0.0][..]),
                "alpha is expensive",
                &nums(b),
                Some(&[1.0, 0.0][..]),
                0.9,
            )
        };
        assert!(
            one_sided(&["100", "2024"], &[]),
            "one-sided numbers defer to embedding"
        );
        // Missing embeddings degrade to exact-match-only.
        assert!(!claims_equivalent(
            "one",
            &nums(&[]),
            None,
            "two",
            &nums(&[]),
            None,
            0.9
        ));
        // `ClaimKey::equivalent_to` wiring: same short-circuits, exercised
        // through the method (embeddings are always None in test builds).
        let k1 = ClaimKey::new("x is true");
        assert!(k1.equivalent_to(&ClaimKey::new("x is true"), 0.9));
        let k2 = ClaimKey::new("alpha costs 100 in 2024");
        assert!(!k2.equivalent_to(&ClaimKey::new("alpha costs 200 in 2024"), 0.9));
        assert!(!k1.equivalent_to(&ClaimKey::new("y is true"), 0.9));
    }

    #[test]
    fn test_claim_merge_keeps_numeric_contradictions_separate() {
        // Price-differing claims sharing a year must never merge into one
        // claim — the contradiction would be silently averaged away.
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![("alpha costs $100 in 2024", "url1", "medium")]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![("alpha costs $200 in 2024", "url2", "medium")]),
            },
            AnalystOutcome::Findings {
                raw: "r3".into(),
                findings: findings(vec![("alpha costs $100 in 2024", "url1", "medium")]),
            },
        ];
        let (merged, _, _) = merge_and_grade(&outcomes);
        assert_eq!(merged.len(), 2, "price-differing claims stay separate");
        let hundred = merged
            .iter()
            .find(|c| c.claim.contains("$100"))
            .expect("the $100 claim");
        assert_eq!(hundred.agreement(), 2);
        let two_hundred = merged
            .iter()
            .find(|c| c.claim.contains("$200"))
            .expect("the $200 claim");
        assert_eq!(two_hundred.agreement(), 1);
        assert!(
            two_hundred.is_disputed(),
            "minority price is disputed → verification fires"
        );
    }

    #[test]
    fn test_render_claim_report_disputed_bracket_on_contradictions() {
        // A claim graded 3/3 (or majority 2/3) that carries surfaced
        // contradictions must render with the DISPUTED bracket — the summary
        // counts it as disputed and verification fires for it, so the grade
        // can never say unanimous / bare majority.
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![
                    ("alpha is true", "url1", "high"),
                    ("beta is true", "url2", "medium"),
                ]),
            },
            AnalystOutcome::Findings {
                raw: "r3".into(),
                findings: findings(vec![
                    ("alpha is true", "url1", "high"),
                    ("beta is true", "url2", "medium"),
                ]),
            },
        ];
        let merged = vec![
            // 3/3 agreement but a contradiction surfaced — disputed, not
            // unanimous.
            MergedClaim {
                claim: "alpha is true".into(),
                analysts: vec![0, 1, 2],
                sources: vec!["url1".into()],
                contradictions: vec!["urlB says alpha is false".into()],
                confidence: "high".into(),
            },
            // 2/3 majority with a contradiction — also disputed.
            MergedClaim {
                claim: "beta is true".into(),
                analysts: vec![1, 2],
                sources: vec!["url2".into()],
                contradictions: vec!["urlC says beta is false".into()],
                confidence: "medium".into(),
            },
            // 2/3 majority without contradictions — bare grade stays.
            MergedClaim {
                claim: "gamma is true".into(),
                analysts: vec![0, 1],
                sources: vec!["url3".into()],
                contradictions: vec![],
                confidence: "medium".into(),
            },
        ];
        let report = render_claim_report("q", &merged, &[], &[], &[], &outcomes);
        assert!(
            report.contains("[3/3 · DISPUTED]"),
            "contradicted unanimous claim must render disputed: {report}"
        );
        assert!(
            report.contains("[2/3 · DISPUTED]"),
            "contradicted majority claim must render disputed: {report}"
        );
        assert!(
            report.contains("[2/3] (medium) gamma is true"),
            "clean majority keeps its bare grade: {report}"
        );
    }

    // ── consolidation fail-open ────────────────────────────────────────

    /// Helper: run consolidation over the given extracted outcomes with the
    /// given fake provider outcomes scripted. Disputed claims spawn real
    /// verifier agents — call `init_test_stores()` first in that case (see
    /// `test_verification_pass_marks_disputed_claims`).
    async fn consolidate_with_script(
        outcomes: Vec<AnalystOutcome>,
        fake: FakeProvider,
    ) -> anyhow::Result<String> {
        let provider: Arc<dyn crate::Provider> = Arc::new(fake);
        let _guard = install_fake_provider(provider);
        let ws = test_ws("/tmp/test_ws");
        consolidate_findings(&ws, "test question", outcomes).await
    }

    /// Two analysts agreeing on one claim (2/3, no contradictions) — never
    /// triggers the verification pass.
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
        assert!(!text.contains("### Report from Analyst 3"), "{text}");
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
        let fake = FakeProvider::new().ok("synthesized answer");
        let result = consolidate_with_script(agreed_outcomes("report a", "report b"), fake).await;
        assert_eq!(
            result.expect("success"),
            "synthesized answer",
            "successful consolidation output is the synthesis text"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_fail_open_mixed_failure_classes() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Mixed transport + NoResponse (empty text) failures exercise both
        // retry_chat branches in one script; exhaustion still fails open.
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

    /// Disputed claims spawn exactly one verification pass with fresh
    /// analysts; the verdict lands in the final report (observed via the
    /// fail-open dump so the intermediate claim report is visible).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_verification_pass_marks_disputed_claims() {
        let _lock = retry_tests_lock();
        init_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // "beta is true" is stated by one analyst (1/3 → disputed); the
        // verification pass spawns a fresh analyst for it.
        let fake = FakeProvider::new()
            .ok("verifier report") // verifier agent run
            .ok(
                r#"{"claim": "beta is true", "verdict": "contradicted", "evidence": "primary source says otherwise", "confidence": "high"}"#,
            )
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down");
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: findings(vec![
                    ("alpha is true", "url1", "high"),
                    ("beta is true", "url2", "medium"),
                ]),
            },
            AnalystOutcome::Findings {
                raw: "r3".into(),
                findings: findings(vec![("alpha is true", "url1", "high")]),
            },
        ];
        let result = consolidate_with_script(outcomes, fake).await;
        let text = result.expect("fail-open dump carries the claim report");
        assert!(text.contains("### Verification"), "{text}");
        assert!(text.contains("beta is true → **contradicted**"), "{text}");
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
}
