//! ResearchTool — Manager-only deep multi-round research orchestrator.
//!
//! Unlike [`AskTool`](super::ask::AskTool) (one round of parallel analysts
//! for quick clarification), `research` decomposes the question into 4–6
//! sub-questions via three independent plans, runs one analyst per
//! sub-question, then runs conditional gap rounds with fresh analysts
//! targeting only the named gaps. Stopping is artifact-based — coverage
//! completion, negative-evidence saturation, marginal-gap saturation,
//! answerability abstention, a verification gate, and a hard agent-spawn cap
//! (never agent self-assessment). Exactly one envelope is delivered
//! asynchronously to the Manager; intermediate rounds never reach the user.
//!
//! Budgeting is by analysts spawned (decomposers, round-1 researchers,
//! gap-round researchers, and verification analysts all count); orchestrator
//! coordination LLM calls do not. The cap is enforced at reservation time and
//! never refunded. No per-agent tool-call caps and no wall-clock limit — the
//! existing global iteration backstop and retry machinery remain untouched.

use crate::agent::run_agent;
use crate::config::CONFIG;
use crate::message_router::{self, AgentJob, JobKind};
use crate::prompt::{load_prompt, substitute};
use crate::session::resolve_agent_id;
use crate::tools::Tool;
use crate::tools::ask::{
    AnalystFindings, Claim, VerificationResult, VerificationTarget, build_async_result_envelope,
    dispatch_verifiers, escape_fences, extract_query_telemetry, load_analyst_angles,
    max_confidence, normalize_claim,
};
use crate::util::UnwrapPoison;
use crate::{ChatMessage, ChatRequest, ChatRequestMeta, DEFAULT_MAX_TOKENS, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};

// ── Constants (module-local defaults) ────────────────────────────────────

/// Hard cap on research analysts spawned per run — decomposers, round-1
/// researchers, gap-round researchers, and verification analysts all count;
/// orchestrator coordination LLM calls do not. Enforced at reservation time,
/// never refunded.
const RESEARCH_MAX_ANALYSTS: usize = 20;
/// Round-0 decomposition fan-out (three independent plans).
const DECOMPOSE_FAN_OUT: usize = 3;
/// Gap-round dispatch widths — rounds shrink as they progress.
const GAP_ROUND_WIDTHS: &[usize] = &[4, 3, 2];
/// Consecutive quiet rounds (no new URLs / novel claims) that trigger
/// negative-evidence saturation.
const QUIET_ROUNDS_TO_STOP: usize = 2;
/// Marginal-gap saturation: >50% near-duplicate gaps = no-progress stop.
const MARGINAL_GAP_SATURATION_PERCENT: usize = 50;
/// Explicit marker when the orchestrator cannot determine the remaining gaps.
const GAP_EXTRACTION_FAILED: &str = "gap extraction failed — remaining gaps unknown";
/// Explicit marker when the plan merge fails — the run falls back to the
/// first valid decomposition plan verbatim.
const PLAN_MERGE_FAILED: &str = "plan merge failed — using first valid decomposition plan verbatim";
/// Explicit marker when the claim annotation pass exhausts its retries — all
/// pending claims are treated as novel.
const CLAIM_ANNOTATION_FAILED: &str = "claim annotation failed — all new claims treated as novel";

// ── Shared orchestration types ───────────────────────────────────────────

/// A sub-question from the round-0 decomposition plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubQuestion {
    question: String,
    evidence_needed: String,
    /// "low" | "medium" | "high" — how hard solid evidence is to find.
    risk: String,
}

/// Round-0 decomposition plan (one per decomposer).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecompositionPlan {
    sub_questions: Vec<SubQuestion>,
}

/// A merged sub-question carrying provenance: the input plan it was copied
/// from verbatim, plus every other plan containing the identical tuple.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MergedSubQuestion {
    question: String,
    evidence_needed: String,
    risk: String,
    /// Index of the input plan this sub-question is a verbatim copy of.
    from_plan: usize,
    /// Other input plans containing the identical verbatim tuple.
    also_in_plans: Vec<usize>,
}

/// A round-0 plan item the merge explicitly dropped (never silently omitted).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DroppedSubQuestion {
    question: String,
    evidence_needed: String,
    risk: String,
    reason: String,
}

/// The merged round-0 plan with full coverage provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MergedPlan {
    sub_questions: Vec<MergedSubQuestion>,
    dropped: Vec<DroppedSubQuestion>,
}

/// One item in the interim gap list.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Gap {
    /// "unanswered" | "partially_answered" | "contradictory" | "low_evidence".
    #[serde(rename = "type")]
    kind: String,
    /// The specific missing claim a fresh analyst could hunt for.
    item: String,
    /// 0-based index into the merged plan's sub_questions.
    traces_to: usize,
    evidence_seen: String,
}

/// The interim gap list.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct GapList {
    gaps: Vec<Gap>,
}

/// Answerability verdict after two stagnant rounds.
#[derive(Debug, Clone, Deserialize)]
struct AnswerabilityCheck {
    answerable: bool,
    reason: String,
}

// ── Budget (analysts spawned) ────────────────────────────────────────────

/// Agent-spawn budget: 1 unit = 1 spawned analyst, counted at reservation
/// time, never refunded.
#[derive(Debug)]
struct ResearchBudget {
    spent: usize,
    cap: usize,
}

impl ResearchBudget {
    fn new(cap: usize) -> Self {
        Self { spent: 0, cap }
    }

    /// Reserve `n` analyst slots — `Ok(())` only when the whole batch fits.
    fn try_reserve(&mut self, n: usize) -> Result<(), String> {
        if self.spent + n > self.cap {
            return Err(format!(
                "research analyst budget exhausted ({}/{})",
                self.spent + n,
                self.cap
            ));
        }
        self.spent += n;
        Ok(())
    }

    fn is_exhausted(&self) -> bool {
        self.spent >= self.cap
    }
}

// ── Per-workspace in-flight guard ────────────────────────────────────────

/// One active deep run per workspace (atomic check-and-set). The guard is
/// moved into the spawned orchestrator task and released on completion.
static ACTIVE_RUNS: LazyLock<RwLock<HashSet<String>>> =
    LazyLock::new(|| RwLock::new(HashSet::new()));

struct ResearchRunGuard {
    ws_name: String,
}

impl ResearchRunGuard {
    fn try_start(ws_name: &str) -> Option<Self> {
        let mut guard = ACTIVE_RUNS.write().unwrap_poison();
        if guard.contains(ws_name) {
            return None;
        }
        guard.insert(ws_name.to_string());
        Some(Self {
            ws_name: ws_name.to_string(),
        })
    }
}

impl Drop for ResearchRunGuard {
    fn drop(&mut self) {
        ACTIVE_RUNS.write().unwrap_poison().remove(&self.ws_name);
    }
}

// ── Tool ─────────────────────────────────────────────────────────────────

pub struct ResearchTool {
    /// The role of the calling agent (Manager).
    pub caller_role: Role,
}

impl ResearchTool {
    #[must_use]
    pub const fn new(caller_role: Role) -> Self {
        Self { caller_role }
    }
}

#[async_trait]
impl Tool for ResearchTool {
    fn name(&self) -> &'static str {
        "research"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "question": {
                    "type": "string",
                    "description": "The deep research question to investigate"
                }
            }),
            &["question"],
        )
    }

    /// The orchestrator only reads evidence and spawns analysts — it never
    /// mutates the workspace, so it may run inside a parallel tool group.
    fn side_effects(&self) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        // Manager-only by construction: role.rs adds this tool to the
        // Manager's set exclusively.
        let question = super::get_str(&args, "question")?;

        // Per-workspace in-flight guard (atomic check-and-set): a second deep
        // run in the same workspace is rejected with a clear error.
        let Some(run_guard) = ResearchRunGuard::try_start(&ws.name) else {
            anyhow::bail!(
                "A deep research run is already active for workspace '{}'. \
                 Wait for the current run to finish before starting another.",
                ws.name
            );
        };

        // Read user context from task-locals BEFORE tokio::spawn so the
        // single result envelope carries the correct user identity.
        let ws = ws.clone();
        let question = question.to_string();
        let caller_role = self.caller_role;
        let user_name = crate::agent::CURRENT_TOOL_USER_NAME
            .try_with(String::clone)
            .unwrap_or_default();
        let channel = crate::agent::CURRENT_TOOL_CHANNEL
            .try_with(String::clone)
            .unwrap_or_default();

        tokio::spawn(async move {
            // The guard lives for the whole orchestrator run.
            let _run_guard = run_guard;
            let message = build_async_research_message(run_deep_research(&ws, &question).await);

            // Route exactly one final envelope to the Manager's agent channel.
            let target_agent_id =
                resolve_agent_id(&channel, &user_name, caller_role.as_str(), &ws.name);
            message_router::route(
                &target_agent_id,
                AgentJob {
                    content: message,
                    workspace_name: ws.name.clone(),
                    user_name,
                    channel,
                    kind: JobKind::ResearchResult,
                    role: caller_role,
                    reply_target: None,
                },
            );
        });

        Ok(
            "Deep research dispatched. One report will be delivered when the run completes."
                .to_string(),
        )
    }
}

/// Build the `<research-result>` envelope message for the async research
/// dispatch. Follows the ask convention: failures are wrapped with an
/// explicit marker, never silently dropped.
fn build_async_research_message(result: anyhow::Result<String>) -> String {
    build_async_result_envelope(result, "research-result")
}

// ── Evidence accumulation ────────────────────────────────────────────────

/// Evidence collected in one research round (post-ledger-dedup).
#[derive(Debug, Default)]
struct EvidenceRound {
    /// Unique source URLs seen this round.
    urls: Vec<String>,
    /// All claims reported this round.
    claims: Vec<Claim>,
    /// Analysts' self-reported uncovered aspects (from `AnalystFindings`).
    unanswered: Vec<String>,
    /// Total queries issued by the round's analysts; `repeat_queries` of
    /// them were repeats of earlier rounds' queries (no-progress signal).
    queries: usize,
    repeat_queries: usize,
    /// Raw responses of analysts whose structured extraction failed — never
    /// silently lost.
    raw_reports: Vec<String>,
}

/// All evidence accumulated across rounds.
#[derive(Debug, Default)]
struct AccumulatedEvidence {
    urls: HashSet<String>,
    /// Claims accumulated across rounds; a claim's stable id is its index in
    /// this vec (claims are only appended or merged in place, never removed).
    claims: Vec<Claim>,
    /// Deduplicated analysts' self-reported unanswered aspects.
    unanswered: Vec<String>,
    unanswered_keys: HashSet<String>,
    /// Raw responses of analysts whose structured extraction failed — never
    /// silently lost.
    raw_reports: Vec<String>,
}

/// Per-claim verdict of the per-round annotation pass: is the new claim
/// novel, a duplicate of an existing claim, or a direct contradiction?
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimAnnotation {
    /// 0-based index of the NEW claim within this round's pending claims.
    new_id: usize,
    /// "novel" | "duplicate" | "contradicts".
    verdict: String,
    /// Index into the EXISTING acc.claims for duplicate/contradicts.
    existing_id: Option<usize>,
    /// Verbatim text of the existing claim (proof).
    proof: Option<String>,
    /// For "contradicts": the contradiction note.
    contradiction: Option<String>,
}

/// The full annotation pass over one round's pending claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnnotationPass {
    annotations: Vec<ClaimAnnotation>,
}

impl AccumulatedEvidence {
    /// Absorb a round's evidence, returning `(novel_urls, novel_unanswered,
    /// pending_claims)`. URLs and unanswered aspects dedup exactly as before;
    /// claims are collected into a pending list — novelty is decided by the
    /// per-round LLM annotation pass, not embedding similarity.
    fn absorb(&mut self, round: &EvidenceRound) -> (usize, usize, Vec<Claim>) {
        let novel_urls = round
            .urls
            .iter()
            .filter(|u| self.urls.insert((*u).clone()))
            .count();
        let mut novel_unanswered = 0usize;
        for u in &round.unanswered {
            let key = normalize_claim(u);
            if !key.is_empty() && self.unanswered_keys.insert(key) {
                self.unanswered.push(u.clone());
                novel_unanswered += 1;
            }
        }
        for r in &round.raw_reports {
            if !self.raw_reports.contains(r) {
                self.raw_reports.push(r.clone());
            }
        }
        (novel_urls, novel_unanswered, round.claims.clone())
    }

    /// Apply an annotation pass over a round's pending claims (the validator
    /// guarantees id completeness and in-range existing ids): novel claims
    /// are appended, duplicates merge into the existing claim (sources joined
    /// deduplicated, confidence upgraded, contradictions appended — never
    /// dropped), and contradicting claims are appended AND linked to the
    /// existing claim so the verification gate targets both sides. Returns
    /// the number of novel claims (the saturation signal).
    fn apply_annotations(&mut self, pass: &AnnotationPass, pending: &[Claim]) -> usize {
        let mut novel = 0usize;
        for a in &pass.annotations {
            let pending_claim = &pending[a.new_id];
            match a.verdict.as_str() {
                "novel" => {
                    self.claims.push(pending_claim.clone());
                    novel += 1;
                }
                "duplicate" => {
                    // Validator guarantees existing_id is Some and in range.
                    let existing_id = a.existing_id.expect("duplicate cites an existing claim");
                    let existing = &mut self.claims[existing_id];
                    existing.confidence =
                        max_confidence(&existing.confidence, &pending_claim.confidence);
                    for c in &pending_claim.contradictions {
                        if !existing.contradictions.contains(c) {
                            existing.contradictions.push(c.clone());
                        }
                    }
                    let mut merged: Vec<String> = existing
                        .source
                        .split("; ")
                        .filter(|s| !s.trim().is_empty())
                        .map(|s| s.trim().to_string())
                        .collect();
                    for s in pending_claim.source.split("; ") {
                        let s = s.trim();
                        if !s.is_empty() && !merged.iter().any(|m| m == s) {
                            merged.push(s.to_string());
                        }
                    }
                    existing.source = merged.join("; ");
                }
                "contradicts" => {
                    let existing_id = a.existing_id.expect("contradicts cites an existing claim");
                    let note = a.contradiction.as_deref().unwrap_or_default();
                    let existing = &mut self.claims[existing_id];
                    if !existing.contradictions.iter().any(|c| c == note) {
                        existing.contradictions.push(note.to_string());
                    }
                    // The new claim is kept and links back to the existing one.
                    let mut new_claim = pending_claim.clone();
                    if !new_claim
                        .contradictions
                        .iter()
                        .any(|c| c == &existing.claim)
                    {
                        new_claim.contradictions.push(existing.claim.clone());
                    }
                    self.claims.push(new_claim);
                    novel += 1;
                }
                _ => unreachable!("validator guarantees the verdict vocabulary"),
            }
        }
        novel
    }
}

/// Cross-agent query ledger: later rounds are given the ledger in their task
/// prompts ("do not repeat these verbatim") — concurrent round-1 analysts
/// cannot see each other's queries, so pre-dispatch suppression is only
/// feasible across rounds. Repeats are tallied per round
/// ([`EvidenceRound::repeat_queries`]) and count as no-progress toward
/// saturation in `gap_rounds` (telemetry in the run summary).
#[derive(Debug, Default)]
struct QueryLedger {
    queries: HashSet<String>,
}

impl QueryLedger {
    /// Register a normalized query. Returns `true` when it was novel.
    fn register(&mut self, query: &str) -> bool {
        let norm = normalize_claim(query);
        !norm.is_empty() && self.queries.insert(norm)
    }

    fn render(&self) -> String {
        if self.queries.is_empty() {
            return "none yet".to_string();
        }
        let mut v: Vec<String> = self.queries.iter().cloned().collect();
        v.sort();
        v.join("\n")
    }
}

/// Running per-run telemetry for the summary.
#[derive(Debug, Default)]
struct RunStats {
    tool_calls: usize,
    searches: usize,
    /// Exact/normalized repeats of an earlier round's queries — summary
    /// telemetry only (the saturation signal lives in
    /// `EvidenceRound::repeat_queries`, wired in `gap_rounds`).
    repeat_queries: usize,
    /// Analysts that failed (no response, empty output, or extraction
    /// failure) — reported explicitly so failures are never silent.
    failed_analysts: usize,
}

/// Outcome of the conditional gap-round phase.
struct GapRoundsOutcome {
    abstention: Option<String>,
    unresolved: Vec<String>,
    rounds_dispatched: usize,
    /// Set when the orchestrator could not determine the remaining gaps —
    /// the report must carry an explicit marker instead of looking like
    /// coverage completion.
    incomplete: Option<String>,
}

// ── Agent runners ────────────────────────────────────────────────────────

/// One analyst run. Mirrors ask's three-state fail-open typing: a
/// parse-failed analyst's raw response is preserved, never dropped.
enum AnalystRun<T> {
    /// Agent produced no response (crashed, cancelled, empty output).
    NoResponse,
    /// Agent responded and structured extraction succeeded.
    Findings(AnalystRunOutcome<T>),
    /// Agent responded but structured extraction failed; the raw response is
    /// preserved for the fail-open report, plus the run's telemetry (queries
    /// must still reach the ledger so later rounds do not re-ask them).
    ParseFailed {
        raw: String,
        tool_calls: usize,
        searches: usize,
        queries: Vec<String>,
    },
}

/// The successful result of one analyst run.
struct AnalystRunOutcome<T> {
    value: T,
    tool_calls: usize,
    searches: usize,
    queries: Vec<String>,
}

/// Run a single analyst agent on `task` and extract structured output `T`
/// while the agent is alive (KV-cache reuse). Returns the extraction plus
/// telemetry `(tool_calls, searches, queries)` from the session history.
async fn run_structured_analyst<T: serde::de::DeserializeOwned>(
    ws: &Workspace,
    agent_id: &str,
    task: &str,
    extraction_prompt: &str,
    validate: Option<&crate::ExtractionValidator<T>>,
) -> AnalystRun<T> {
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
    let Some(raw) = response else {
        return AnalystRun::NoResponse;
    };
    if raw.trim().is_empty() {
        return AnalystRun::NoResponse;
    }
    let (tool_calls, searches, queries) = extract_query_telemetry(&agent);
    match agent
        .extract_verdict::<T>(extraction_prompt, validate)
        .await
    {
        Ok(value) => AnalystRun::Findings(AnalystRunOutcome {
            value,
            tool_calls,
            searches,
            queries,
        }),
        Err(_) => AnalystRun::ParseFailed {
            raw,
            tool_calls,
            searches,
            queries,
        },
    }
}

/// Collect a round's evidence from its analyst runs: register queries in the
/// ledger, collect claims + source URLs, accumulate telemetry, and preserve
/// parse-failed analysts' raw responses (fail-open — never dropped).
fn collect_evidence(
    runs: &[AnalystRun<AnalystFindings>],
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
) -> EvidenceRound {
    let mut round = EvidenceRound::default();
    for run in runs {
        let run = match run {
            AnalystRun::NoResponse => {
                run_stats.failed_analysts += 1;
                continue;
            }
            AnalystRun::ParseFailed {
                raw,
                tool_calls,
                searches,
                queries,
            } => {
                run_stats.failed_analysts += 1;
                run_stats.tool_calls += tool_calls;
                run_stats.searches += searches;
                // Failed analysts' queries still reach the ledger (later
                // rounds must not re-ask them) and the summary telemetry;
                // they don't drive the per-round saturation signal
                // (`round.queries` stays from successful analysts only).
                for q in queries {
                    if !ledger.register(q) {
                        run_stats.repeat_queries += 1;
                    }
                }
                round.raw_reports.push(raw.clone());
                continue;
            }
            AnalystRun::Findings(run) => run,
        };
        run_stats.tool_calls += run.tool_calls;
        run_stats.searches += run.searches;
        for q in &run.queries {
            round.queries += 1;
            if !ledger.register(q) {
                round.repeat_queries += 1;
                run_stats.repeat_queries += 1;
            }
        }
        for claim in &run.value.claims {
            round.claims.push(claim.clone());
            if !claim.source.is_empty() {
                round.urls.push(claim.source.clone());
            }
        }
        round
            .unanswered
            .extend(run.value.unanswered.iter().cloned());
    }
    round
}

// ── Orchestrator LLM helpers (not budgeted) ──────────────────────────────

/// Build the orchestrator's chat params: cheap Analyst model (per-role
/// overrides respected), constant model/effort/tools across all coordination
/// calls of a run (KV-cache friendly — the leading general workspace context
/// system message is constant too; only the user message varies).
fn orchestrator_params(ws: &Workspace, purpose: &'static str) -> ChatRequest {
    let model = CONFIG.role_model(Role::Analyst);
    let routing = CONFIG.model_routing(&model);
    ChatRequest {
        messages: Vec::new(),
        tools: None,
        model,
        allow_image_parts: false,
        max_tokens: Some(DEFAULT_MAX_TOKENS),
        reasoning_effort: Some(CONFIG.role_reasoning_effort(Role::Analyst)),
        provider_order: routing.provider_order,
        provider_allow_fallbacks: routing.allow_fallbacks,
        response_format_json_object: false,
        meta: Some(ChatRequestMeta {
            purpose,
            agent_id: format!("research_{}_orchestrator", ws.name),
            role: Role::Analyst.as_str().to_string(),
            workspace: ws.name.clone(),
            ticket_id: None,
        }),
    }
}

/// Free-form orchestrator LLM call via the hardened outer retry loop.
/// The general workspace context is prepended as the leading system message.
async fn orchestrator_chat(ws: &Workspace, purpose: &'static str, user: &str) -> Result<String> {
    let _call = crate::call_registry::NON_AGENT_CALLS.register(purpose, &ws.name);
    let mut params = orchestrator_params(ws, purpose);
    let mut messages = Vec::with_capacity(2);
    crate::prompt::prepend_general_context(&mut messages, ws).await;
    messages.push(ChatMessage::user(user));
    params.messages = messages;
    let policy = crate::retry::RetryPolicy::current();
    let response = crate::retry::retry_chat(params, &policy)
        .await
        .map_err(|e| anyhow::anyhow!("orchestrator call '{purpose}' failed: {e}"))?;
    Ok(response.text_or_empty().to_string())
}

/// Structured orchestrator extraction via the hardened scoped retry loop.
/// `prompt` embeds the JSON schema request. The general workspace context is
/// prepended as the leading system message.
async fn orchestrator_extract<T: serde::de::DeserializeOwned>(
    ws: &Workspace,
    purpose: &'static str,
    prompt: &str,
    validate: Option<&crate::ExtractionValidator<T>>,
) -> Result<T> {
    let _call = crate::call_registry::NON_AGENT_CALLS.register(purpose, &ws.name);
    let params = orchestrator_params(ws, purpose);
    let mut messages = Vec::with_capacity(2);
    crate::prompt::prepend_general_context(&mut messages, ws).await;
    messages.push(ChatMessage::user(prompt));
    crate::extraction::retry_extract_structured_scoped::<T>(&messages, "", &params, validate)
        .await
        .map_err(|e| anyhow::anyhow!("orchestrator extraction '{purpose}' failed: {e}"))
}

// ── Round 0: decomposition ───────────────────────────────────────────────

/// Round 0: three independent decomposition plans merged into one
/// consolidated plan with provenance via a single orchestrator LLM call.
/// Budget: 3 decomposers. The x3 redundancy lives here, at the steering
/// decision. On merge failure the run falls back to the first valid plan
/// verbatim with an explicit marker; only when ALL decomposers failed does
/// the run error (no plan = no research).
///
/// Asymmetry note: a parse-failed decomposer's raw response is intentionally
/// NOT preserved (unlike `collect_evidence`, which keeps failed analysts'
/// raws for the fail-open report). Decomposer plans are steering, not
/// evidence — the failure still counts in `run_stats.failed_analysts`, and
/// the run aborts with an explicit error if no plan parses.
async fn round0_decompose(
    ws: &Workspace,
    question: &str,
    budget: &mut ResearchBudget,
    run_stats: &mut RunStats,
) -> Result<(MergedPlan, Option<String>)> {
    budget
        .try_reserve(DECOMPOSE_FAN_OUT)
        .map_err(anyhow::Error::msg)?;
    let task_template = load_prompt("research/decompose.md");
    let extraction_prompt = load_prompt("extraction/decompose.md");
    let futures: Vec<_> = (0..DECOMPOSE_FAN_OUT)
        .map(|i| {
            let ws = ws.clone();
            let question = question.to_string();
            let task = substitute(&task_template, &[("{{question}}", &question)]);
            let agent_id = crate::session::research_agent_id(&ws.name, &format!("decompose_{i}"));
            let extraction_prompt = extraction_prompt.clone();
            async move {
                run_structured_analyst::<DecompositionPlan>(
                    &ws,
                    &agent_id,
                    &task,
                    &extraction_prompt,
                    None,
                )
                .await
            }
        })
        .collect();
    let plans: Vec<AnalystRun<DecompositionPlan>> = join_all(futures).await;
    let mut valid = Vec::new();
    for run in plans {
        match run {
            AnalystRun::Findings(o) => valid.push(o.value),
            AnalystRun::NoResponse | AnalystRun::ParseFailed { .. } => {
                run_stats.failed_analysts += 1;
            }
        }
    }
    if valid.is_empty() {
        anyhow::bail!("all decomposition analysts failed — no research plan produced");
    }
    if let Ok(plan) = merge_decomposition_plans(ws, question, &valid).await {
        Ok((plan, None))
    } else {
        // Fail-open: a failed merge must never lose the run — fall back to
        // the first valid plan verbatim with an explicit marker.
        let first = &valid[0];
        let plan = MergedPlan {
            sub_questions: first
                .sub_questions
                .iter()
                .map(|sq| MergedSubQuestion {
                    question: sq.question.clone(),
                    evidence_needed: sq.evidence_needed.clone(),
                    risk: sq.risk.clone(),
                    from_plan: 0,
                    also_in_plans: Vec::new(),
                })
                .collect(),
            dropped: Vec::new(),
        };
        Ok((plan, Some(PLAN_MERGE_FAILED.to_string())))
    }
}

/// Merge 1–3 independent plans into one consolidated plan with provenance
/// (orchestrator extraction call — not budgeted).
async fn merge_decomposition_plans(
    ws: &Workspace,
    question: &str,
    plans: &[DecompositionPlan],
) -> Result<MergedPlan> {
    let plans_json = serde_json::to_string_pretty(plans)?;
    let prompt = substitute(
        &load_prompt("research/decompose_merge.md"),
        &[("{{question}}", question), ("{{plans}}", &plans_json)],
    );
    // The validator captures an owned copy of the input plans (the validator
    // type is `'static`).
    let plans_owned = plans.to_vec();
    orchestrator_extract::<MergedPlan>(
        ws,
        "decompose_merge",
        &prompt,
        Some(&move |p| validate_merged_plan(p, &plans_owned)),
    )
    .await
}

/// Validate a merged plan: 4–6 sub-questions, verbatim provenance against the
/// cited input plans, and full coverage — every input plan item appears
/// exactly once across merged sub-questions (as `from_plan` or
/// `also_in_plans`) plus dropped. Silent dropout is rejected (fail-closed
/// inside the extraction retry loop).
fn validate_merged_plan(
    plan: &MergedPlan,
    input_plans: &[DecompositionPlan],
) -> Result<(), String> {
    let n = plan.sub_questions.len();
    if !(4..=6).contains(&n) {
        return Err(format!("merged plan has {n} sub-questions, expected 4-6"));
    }
    let plan_has = |p: &DecompositionPlan, q: &str, e: &str, r: &str| {
        p.sub_questions
            .iter()
            .position(|s| s.question == q && s.evidence_needed == e && s.risk == r)
    };
    // Coverage tally: every input item must be referenced exactly once.
    let mut covered: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut mark = |p_idx: usize, s_idx: usize| {
        *covered.entry((p_idx, s_idx)).or_insert(0) += 1;
    };
    for sq in &plan.sub_questions {
        if sq.from_plan >= input_plans.len() {
            return Err(format!(
                "merged sub-question '{}' cites from_plan {} but only {} plans exist",
                sq.question,
                sq.from_plan,
                input_plans.len()
            ));
        }
        for &i in &sq.also_in_plans {
            if i >= input_plans.len() {
                return Err(format!(
                    "merged sub-question '{}' cites also_in_plans {i} but only {} plans exist",
                    sq.question,
                    input_plans.len()
                ));
            }
        }
        let (q, e, r) = (&sq.question, &sq.evidence_needed, &sq.risk);
        let Some(s_idx) = plan_has(&input_plans[sq.from_plan], q, e, r) else {
            return Err(format!(
                "merged sub-question tuple not found verbatim in plan {}: ({q}, {e}, {r})",
                sq.from_plan
            ));
        };
        mark(sq.from_plan, s_idx);
        for &i in &sq.also_in_plans {
            let Some(s_idx) = plan_has(&input_plans[i], q, e, r) else {
                return Err(format!(
                    "merged sub-question tuple not found verbatim in plan {i}: ({q}, {e}, {r})"
                ));
            };
            mark(i, s_idx);
        }
    }
    for dropped in &plan.dropped {
        for (p_idx, p) in input_plans.iter().enumerate() {
            if let Some(s_idx) = plan_has(
                p,
                &dropped.question,
                &dropped.evidence_needed,
                &dropped.risk,
            ) {
                mark(p_idx, s_idx);
            }
        }
    }
    for (p_idx, p) in input_plans.iter().enumerate() {
        for s_idx in 0..p.sub_questions.len() {
            match covered.get(&(p_idx, s_idx)).copied().unwrap_or(0) {
                1 => {}
                0 => {
                    return Err(format!(
                        "silent dropout: input plan {p_idx} sub-question {s_idx} is never covered by the merged plan or dropped list"
                    ));
                }
                k => {
                    return Err(format!(
                        "input plan {p_idx} sub-question {s_idx} is covered {k} times — every plan item must appear exactly once"
                    ));
                }
            }
        }
    }
    Ok(())
}

// ── Round 1: one analyst per sub-question ────────────────────────────────

/// Round 1: one analyst per sub-question; two for high-risk items (the
/// second gets a decorrelated research angle). Returns the round's evidence,
/// or `None` when the analyst budget is exhausted before dispatch.
async fn round1_research(
    ws: &Workspace,
    question: &str,
    plan: &MergedPlan,
    budget: &mut ResearchBudget,
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
) -> Option<EvidenceRound> {
    let spawn_count = plan.sub_questions.len()
        + plan
            .sub_questions
            .iter()
            .filter(|s| s.risk == "high")
            .count();
    if budget.try_reserve(spawn_count).is_err() {
        tracing::warn!(
            spent = %budget.spent,
            cap = %budget.cap,
            "research budget exhausted before round 1"
        );
        return None;
    }
    let task_template = load_prompt("research/round1.md");
    let extraction_prompt = load_prompt("extraction/findings.md");
    let angles = load_analyst_angles();
    let ledger_snapshot = ledger.render();
    let mut futures = Vec::new();
    let mut idx = 0usize;
    for sq in &plan.sub_questions {
        for k in 0..=usize::from(sq.risk == "high") {
            let ws = ws.clone();
            let question = question.to_string();
            let mut task = substitute(
                &task_template,
                &[
                    ("{{question}}", &question),
                    ("{{sub_question}}", &sq.question),
                    ("{{evidence_needed}}", &sq.evidence_needed),
                    ("{{query_ledger}}", &ledger_snapshot),
                ],
            );
            // KV-cache discipline: vary ONLY the user message (the angle).
            // Second analysts for high-risk items get a distinct angle each
            // (cycled, not always the first one).
            if k == 1 && !angles.is_empty() {
                task.push_str("\n\nResearch angle:\n");
                task.push_str(&angles[idx % angles.len()]);
            }
            let agent_id = crate::session::research_agent_id(&ws.name, &format!("r1_{idx}"));
            idx += 1;
            let extraction_prompt = extraction_prompt.clone();
            futures.push(async move {
                run_structured_analyst::<AnalystFindings>(
                    &ws,
                    &agent_id,
                    &task,
                    &extraction_prompt,
                    None,
                )
                .await
            });
        }
    }
    let runs = join_all(futures).await;
    Some(collect_evidence(&runs, ledger, run_stats))
}

// ── Interim consolidation + conditional gap rounds ───────────────────────

/// Interim consolidation: extract the structured gap list from the
/// accumulated evidence (orchestrator call, not budgeted). `None` marks an
/// extraction failure — the caller must surface it explicitly, never collapse
/// it into "coverage completion".
async fn extract_gap_list(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    plan: &MergedPlan,
) -> Option<GapList> {
    let evidence = render_accumulated_evidence(acc);
    let plan_json = serde_json::to_string(plan).unwrap_or_default();
    let prompt = substitute(
        &load_prompt("research/gap_extract.md"),
        &[
            ("{{question}}", question),
            ("{{plan}}", &plan_json),
            ("{{evidence}}", &evidence),
        ],
    );
    // The validator captures an owned copy of the plan (the validator type is
    // `'static`).
    let plan_owned = plan.clone();
    orchestrator_extract::<GapList>(
        ws,
        "gap_extract",
        &prompt,
        Some(&move |g| validate_gap_list(g, &plan_owned)),
    )
    .await
    .ok()
}

/// Validate the gap list: every gap's `traces_to` must be a 0-based index
/// into the merged plan's sub-questions (fail-closed inside the extraction
/// retry loop — index-range validation guarantees traceability).
fn validate_gap_list(gaps: &GapList, plan: &MergedPlan) -> Result<(), String> {
    for g in &gaps.gaps {
        if g.traces_to >= plan.sub_questions.len() {
            return Err(format!(
                "gap '{}' traces to plan sub-question {} but the merged plan has only {} sub-questions",
                g.item,
                g.traces_to,
                plan.sub_questions.len()
            ));
        }
    }
    Ok(())
}

/// The gap items as plain strings (for the report's unresolved list).
fn gap_items(gaps: &[Gap]) -> Vec<String> {
    gaps.iter().map(|g| g.item.clone()).collect()
}

/// Marginal-gap saturation: >50% of the traceable gaps are near-duplicates
/// of each other → no-progress stop (deterministic normalized comparison).
fn is_marginal_gap_saturated(gaps: &[Gap]) -> bool {
    if gaps.len() < 2 {
        return false;
    }
    let mut seen = HashSet::new();
    let mut dups = 0usize;
    for g in gaps {
        if !seen.insert(normalize_claim(&g.item)) {
            dups += 1;
        }
    }
    dups * 100 > gaps.len() * MARGINAL_GAP_SATURATION_PERCENT
}

/// Run one gap round: fresh analysts, one per targeted gap (width-shrinking
/// 4→3→2). Returns the round's analyst runs.
async fn run_gap_round(
    ws: &Workspace,
    question: &str,
    gaps: &[&Gap],
    ledger: &QueryLedger,
) -> Vec<AnalystRun<AnalystFindings>> {
    let task_template = load_prompt("research/gap.md");
    let extraction_prompt = load_prompt("extraction/findings.md");
    let ledger_snapshot = ledger.render();
    let futures: Vec<_> = gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| {
            let ws = ws.clone();
            let question = question.to_string();
            let task = substitute(
                &task_template,
                &[
                    ("{{question}}", &question),
                    (
                        "{{gaps}}",
                        &format!(
                            "- [{}] {} (traces to: plan sub-question {})",
                            gap.kind, gap.item, gap.traces_to
                        ),
                    ),
                    ("{{query_ledger}}", &ledger_snapshot),
                ],
            );
            let agent_id = crate::session::research_agent_id(&ws.name, &format!("gap_{i}"));
            let extraction_prompt = extraction_prompt.clone();
            async move {
                run_structured_analyst::<AnalystFindings>(
                    &ws,
                    &agent_id,
                    &task,
                    &extraction_prompt,
                    None,
                )
                .await
            }
        })
        .collect();
    join_all(futures).await
}

/// Conditional gap rounds. Stopping is artifact-based, never agent
/// self-assessment: coverage completion, negative-evidence saturation (two
/// consecutive quiet rounds), marginal-gap saturation, answerability
/// abstention, budget exhaustion, and shutdown.
#[allow(clippy::too_many_arguments)]
async fn gap_rounds(
    ws: &Workspace,
    question: &str,
    plan: &MergedPlan,
    budget: &mut ResearchBudget,
    ledger: &mut QueryLedger,
    acc: &mut AccumulatedEvidence,
    run_stats: &mut RunStats,
    markers: &mut Vec<String>,
) -> GapRoundsOutcome {
    let mut outcome = GapRoundsOutcome {
        abstention: None,
        unresolved: Vec::new(),
        rounds_dispatched: 0,
        incomplete: None,
    };
    let Some(mut gap_list) = extract_gap_list(ws, question, acc, plan).await else {
        // Explicit marker — never collapse into coverage completion.
        outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
        return outcome;
    };
    let mut quiet_rounds = 0usize;
    let mut round_index = 0usize;
    loop {
        if crate::shutdown::shutdown_token().is_cancelled() {
            outcome.unresolved = gap_items(&gap_list.gaps);
            return outcome;
        }
        if budget.is_exhausted() {
            outcome.unresolved = gap_items(&gap_list.gaps);
            return outcome;
        }
        // The gap list is validated (traces_to in range) — every gap is
        // traceable, so it is used directly.
        let gaps = &gap_list.gaps;
        if gaps.is_empty() {
            // Coverage completion.
            return outcome;
        }
        if is_marginal_gap_saturated(gaps) {
            outcome.unresolved = gap_items(gaps);
            return outcome;
        }
        let width = GAP_ROUND_WIDTHS[round_index.min(GAP_ROUND_WIDTHS.len() - 1)];
        round_index += 1;
        let targeted: Vec<&Gap> = gaps.iter().take(width).collect();
        if budget.try_reserve(targeted.len()).is_err() {
            outcome.unresolved = gap_items(gaps);
            return outcome;
        }
        let runs = run_gap_round(ws, question, &targeted, ledger).await;
        let round = collect_evidence(&runs, ledger, run_stats);
        let (new_urls, _, pending) = acc.absorb(&round);
        let novel_claims = annotate_round(ws, acc, &pending, markers).await;
        outcome.rounds_dispatched += 1;
        // A round whose analysts only re-asked already-asked queries counts
        // as no-progress toward saturation (repeat queries are never
        // pre-dispatch-droppable — concurrent analysts generate them live).
        let all_repeat_queries = round.queries > 0 && round.queries == round.repeat_queries;
        if (new_urls == 0 && novel_claims == 0) || all_repeat_queries {
            quiet_rounds += 1;
            if quiet_rounds >= QUIET_ROUNDS_TO_STOP {
                // Negative-evidence saturation: answerability abstain check.
                outcome.abstention = check_answerability(ws, question, acc).await;
                outcome.unresolved = gap_items(gaps);
                return outcome;
            }
        } else {
            quiet_rounds = 0;
        }
        // Refresh the gap list from the accumulated evidence only when the
        // round produced progress; a quiet round reuses the current list.
        if new_urls != 0 || novel_claims != 0 {
            let Some(next_gap_list) = extract_gap_list(ws, question, acc, plan).await else {
                outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
                outcome.unresolved = gap_items(gaps);
                return outcome;
            };
            gap_list = next_gap_list;
        }
    }
}

/// Per-round claim annotation pass: classify each pending claim against the
/// existing accumulated claims via a single orchestrator extraction call.
/// The validator guarantees id completeness and verbatim proofs (fail-closed
/// inside the extraction retry loop).
async fn annotate_claims(
    ws: &Workspace,
    existing: &[Claim],
    pending: &[Claim],
) -> Result<AnnotationPass> {
    let mut existing_claims = String::new();
    for (i, c) in existing.iter().enumerate() {
        let _ = writeln!(existing_claims, "{i}: {}", c.claim);
    }
    let mut pending_claims = String::new();
    for (i, c) in pending.iter().enumerate() {
        let _ = writeln!(pending_claims, "{i}: {}", c.claim);
    }
    let mut user = substitute(
        &load_prompt("research/annotate.md"),
        &[
            ("{{existing_claims}}", &existing_claims),
            ("{{pending_claims}}", &pending_claims),
        ],
    );
    user.push_str("\n\n");
    user.push_str(&load_prompt("extraction/annotate.md"));
    // The validator captures owned copies (the validator type is `'static`).
    let existing_owned = existing.to_vec();
    let pending_owned = pending.to_vec();
    orchestrator_extract::<AnnotationPass>(
        ws,
        "claim_annotate",
        &user,
        Some(&move |a| validate_annotations(a, &existing_owned, &pending_owned)),
    )
    .await
}

/// Validate an annotation pass: every pending claim annotated exactly once;
/// `duplicate`/`contradicts` cite an in-range existing claim with a verbatim
/// proof; `novel` cites nothing; the contradiction note is present exactly
/// when the verdict is "contradicts".
fn validate_annotations(
    pass: &AnnotationPass,
    existing: &[Claim],
    pending: &[Claim],
) -> Result<(), String> {
    let mut ids: Vec<usize> = pass.annotations.iter().map(|a| a.new_id).collect();
    ids.sort_unstable();
    let expected: Vec<usize> = (0..pending.len()).collect();
    if ids != expected {
        return Err(format!(
            "annotation pass must annotate every new claim exactly once: ids {ids:?} != 0..{}",
            pending.len()
        ));
    }
    for a in &pass.annotations {
        let verdict = a.verdict.as_str();
        if !matches!(verdict, "novel" | "duplicate" | "contradicts") {
            return Err(format!(
                "verdict '{verdict}' not in [novel, duplicate, contradicts]"
            ));
        }
        let has_note = a
            .contradiction
            .as_deref()
            .is_some_and(|c| !c.trim().is_empty());
        if (verdict == "contradicts") != has_note {
            return Err(format!(
                "verdict '{verdict}' for new claim {} must carry the contradiction note exactly when it contradicts",
                a.new_id
            ));
        }
        if verdict == "novel" {
            if a.existing_id.is_some() {
                return Err(format!(
                    "novel annotation for new claim {} must not cite an existing claim",
                    a.new_id
                ));
            }
            continue;
        }
        let Some(existing_id) = a.existing_id else {
            return Err(format!(
                "{verdict} annotation for new claim {} must cite an existing claim",
                a.new_id
            ));
        };
        if existing_id >= existing.len() {
            return Err(format!(
                "existing_id {existing_id} out of range ({} existing claims)",
                existing.len()
            ));
        }
        let Some(proof) = a.proof.as_deref().filter(|p| !p.trim().is_empty()) else {
            return Err(format!(
                "{verdict} annotation for new claim {} must carry a verbatim proof",
                a.new_id
            ));
        };
        // Verbatim proof check uses the shared core's case-preserving
        // normalization (consistency with the grouping validator's membership
        // invariant — a proof that folds case would be weaker than the
        // consensus path it feeds).
        if crate::consensus::normalize_item(proof)
            != crate::consensus::normalize_item(&existing[existing_id].claim)
        {
            return Err(format!(
                "proof for new claim {} does not match existing claim {existing_id} verbatim",
                a.new_id
            ));
        }
    }
    Ok(())
}

/// Run the annotation pass over a round's pending claims and apply the
/// results. Returns the number of novel claims (the saturation signal). On
/// exhaustion every pending claim is treated as novel with an explicit
/// marker — claims are never dropped.
async fn annotate_round(
    ws: &Workspace,
    acc: &mut AccumulatedEvidence,
    pending: &[Claim],
    markers: &mut Vec<String>,
) -> usize {
    if pending.is_empty() {
        return 0;
    }
    if acc.claims.is_empty() {
        // First round: nothing to compare against — every pending claim is
        // novel by construction. Skip the wasted orchestrator call (and with
        // it the spurious failure marker an out-of-range existing_id would
        // otherwise trigger).
        acc.claims.extend(pending.iter().cloned());
        return pending.len();
    }
    if let Ok(pass) = annotate_claims(ws, &acc.claims, pending).await {
        acc.apply_annotations(&pass, pending)
    } else {
        acc.claims.extend(pending.iter().cloned());
        if !markers.iter().any(|m| m == CLAIM_ANNOTATION_FAILED) {
            markers.push(CLAIM_ANNOTATION_FAILED.to_string());
        }
        pending.len()
    }
}

/// After two stagnant rounds: is the question genuinely unanswerable with the
/// evidence gathered? `Some(abstention)` when it is (orchestrator call, not
/// budgeted).
async fn check_answerability(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
) -> Option<String> {
    let evidence = render_accumulated_evidence(acc);
    let prompt = substitute(
        &load_prompt("research/abstain.md"),
        &[("{{question}}", question), ("{{evidence}}", &evidence)],
    );
    let verdict = orchestrator_extract::<AnswerabilityCheck>(ws, "abstain_check", &prompt, None)
        .await
        .ok()?;
    (!verdict.answerable).then_some(verdict.reason)
}

// ── Final synthesis + verification gate ──────────────────────────────────

/// One-shot final synthesis from the accumulated evidence. The report is
/// NEVER regenerated — this is the only synthesis pass.
async fn synthesize(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    abstention: Option<&str>,
) -> Result<String> {
    let evidence = render_accumulated_evidence(acc);
    let mut user = substitute(
        &load_prompt("research/synthesize.md"),
        &[("{{question}}", question), ("{{evidence}}", &evidence)],
    );
    if let Some(abstain) = abstention {
        let _ = writeln!(
            user,
            "\n\n# Answerability Note\n\nThe research team determined the question is not \
             answerable with the available evidence: {abstain}\n\
             State this clearly and explain what evidence would be needed."
        );
    }
    orchestrator_chat(ws, "synthesize", &user).await
}

/// Verification gate: fresh analysts verify the disputed / low-confidence
/// accumulated claims (budgeted, bounded). Anything still disputed stays
/// marked unresolved in the final report. Verifier tool calls / searches /
/// queries count toward the run summary and register in the query ledger
/// (later rounds must not re-ask them) — repeats here inflate the summary's
/// repeat-queries line but never the per-round saturation signal
/// (`EvidenceRound::repeat_queries` is untouched).
async fn research_verification_pass(
    ws: &Workspace,
    acc: &AccumulatedEvidence,
    budget: &mut ResearchBudget,
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
) -> Vec<VerificationResult> {
    let targets: Vec<VerificationTarget> = acc
        .claims
        .iter()
        .filter(|c| !c.contradictions.is_empty() || c.confidence == "low")
        .map(|c| VerificationTarget::new(&c.claim, &c.source, &c.contradictions.join("; ")))
        .collect();
    if targets.is_empty() {
        return Vec::new();
    }
    let cap = targets.len().min(crate::tools::ask::VERIFY_MAX_ANALYSTS);
    // Reserve what fits — a near-exhausted budget still verifies the
    // highest-priority targets instead of nothing.
    let n = cap.min(budget.cap.saturating_sub(budget.spent));
    let mut results = Vec::new();
    if n > 0 && budget.try_reserve(n).is_ok() {
        let ledger_snapshot = ledger.render();
        let task_extra = format!(
            "\n# Queries Already Asked (do not repeat these verbatim)\n\n{ledger_snapshot}"
        );
        let prefix = format!("research_{}_verify", ws.name);
        results = dispatch_verifiers(ws, &prefix, &targets[..n], &task_extra).await;
    }
    for v in &results {
        run_stats.tool_calls += v.tool_calls;
        run_stats.searches += v.searches;
        for q in &v.queries {
            if !ledger.register(q) {
                run_stats.repeat_queries += 1;
            }
        }
    }
    // Targets beyond the verified set are explicitly marked unresolved —
    // never silently skipped.
    for t in targets.iter().skip(results.len()) {
        results.push(VerificationResult {
            claim: t.claim.clone(),
            verdict: "unresolved".to_string(),
            evidence: "verification skipped — analyst budget exhausted".to_string(),
            tool_calls: 0,
            searches: 0,
            queries: Vec::new(),
        });
    }
    results
}

// ── Rendering ────────────────────────────────────────────────────────────

/// Render the accumulated evidence as a compact numbered list for the
/// orchestrator prompts, plus analysts' self-reported unanswered aspects.
/// Claim ids are their stable 0-based indices in `acc.claims` — the
/// annotation pass and final synthesis reference them by these ids.
fn render_accumulated_evidence(acc: &AccumulatedEvidence) -> String {
    let mut out = String::new();
    for (i, c) in acc.claims.iter().enumerate() {
        let source = if c.source.is_empty() {
            "n/a"
        } else {
            &c.source
        };
        let _ = writeln!(
            out,
            "{i}. [{}] {} — source: {source}",
            c.confidence, c.claim,
        );
        if !c.contradictions.is_empty() {
            let _ = writeln!(out, "   contradictions: {}", c.contradictions.join("; "));
        }
    }
    if !acc.unanswered.is_empty() {
        let _ = writeln!(out, "\nAnalysts reported these as still unanswered:");
        for u in &acc.unanswered {
            let _ = writeln!(out, "- {u}");
        }
    }
    if !acc.raw_reports.is_empty() {
        let _ = writeln!(
            out,
            "\nRaw notes from analysts whose structured extraction failed (preserve any \
             usable content):"
        );
        for raw in &acc.raw_reports {
            let _ = writeln!(out, "- {raw}");
        }
    }
    out
}

/// Render the run summary: rounds, agents vs budget, tool calls, searches,
/// wall time, unresolved gaps, and any abstention or incomplete marker.
#[allow(clippy::too_many_arguments)]
fn render_run_summary(
    run_stats: &RunStats,
    budget: &ResearchBudget,
    rounds_used: usize,
    acc: &AccumulatedEvidence,
    abstention: Option<&str>,
    unresolved: &[String],
    incomplete: Option<&str>,
    markers: &[String],
    wall: Duration,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Run Summary");
    let _ = writeln!(out, "- rounds used: {rounds_used}");
    if let Some(reason) = incomplete {
        let _ = writeln!(out, "- gap rounds incomplete: {reason}");
    }
    if !markers.is_empty() {
        let _ = writeln!(out, "- markers:");
        for m in markers {
            let _ = writeln!(out, "  - {m}");
        }
    }
    let _ = writeln!(out, "- agents spawned: {} / {}", budget.spent, budget.cap);
    let _ = writeln!(out, "- tool calls: {}", run_stats.tool_calls);
    let _ = writeln!(out, "- searches: {}", run_stats.searches);
    let _ = writeln!(
        out,
        "- repeat queries (no-progress): {}",
        run_stats.repeat_queries
    );
    if run_stats.failed_analysts > 0 {
        let _ = writeln!(
            out,
            "- analysts failed (no response / extraction failure): {}",
            run_stats.failed_analysts
        );
    }
    let _ = writeln!(out, "- wall time: {:.0}s", wall.as_secs_f64());
    let _ = writeln!(
        out,
        "- evidence: {} claims, {} unique sources",
        acc.claims.len(),
        acc.urls.len()
    );
    if let Some(a) = abstention {
        let _ = writeln!(out, "- answerability: QUESTION ABSTAINED — {a}");
    }
    if !unresolved.is_empty() {
        let _ = writeln!(out, "- unresolved gaps:");
        for u in unresolved {
            let _ = writeln!(out, "  - {}", escape_fences(u));
        }
    }
    out
}

/// Partial envelope on shutdown mid-run: whatever evidence was gathered is
/// delivered with an explicit incomplete marker — findings are never lost.
fn partial_report(question: &str, acc: &AccumulatedEvidence, reason: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Research Report (incomplete — {reason})");
    let _ = writeln!(out);
    let _ = writeln!(out, "**Question**: {question}");
    let _ = writeln!(out);
    let _ = writeln!(out, "### Evidence Gathered So Far");
    if acc.claims.is_empty() {
        let _ = writeln!(out, "- none");
    } else {
        for c in &acc.claims {
            let source = if c.source.is_empty() {
                "n/a"
            } else {
                &c.source
            };
            let _ = writeln!(
                out,
                "- [{}] {} — source: {source}",
                c.confidence,
                escape_fences(&c.claim),
            );
        }
    }
    if !acc.unanswered.is_empty() {
        let _ = writeln!(out, "\nAnalysts reported these as still unanswered:");
        for u in &acc.unanswered {
            let _ = writeln!(out, "- {}", escape_fences(u));
        }
    }
    if !acc.raw_reports.is_empty() {
        let _ = writeln!(out, "\n### Failed Analyst Reports");
        for (i, raw) in acc.raw_reports.iter().enumerate() {
            let _ = writeln!(out, "### Report from Analyst {}", i + 1);
            let _ = writeln!(out, "{}", escape_fences(raw));
        }
    }
    out
}

// ── Orchestrator ─────────────────────────────────────────────────────────

/// Run the full deep-research orchestrator: round 0 (decomposition), round 1
/// (per-sub-question researchers), conditional gap rounds, one-shot final
/// synthesis, verification gate, and the run summary.
#[allow(clippy::too_many_lines)]
async fn run_deep_research(ws: &Workspace, question: &str) -> Result<String> {
    let start = Instant::now();
    let mut budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
    let mut ledger = QueryLedger::default();
    let mut acc = AccumulatedEvidence::default();
    let mut run_stats = RunStats::default();
    let mut markers: Vec<String> = Vec::new();

    // Round 0 — three independent decomposition plans merged into one.
    let plan = match round0_decompose(ws, question, &mut budget, &mut run_stats).await {
        Ok((plan, marker)) => {
            if let Some(m) = marker {
                markers.push(m);
            }
            plan
        }
        Err(_) if crate::shutdown::shutdown_token().is_cancelled() => {
            return Ok(partial_report(
                question,
                &acc,
                "shutdown during decomposition",
            ));
        }
        Err(e) => return Err(e),
    };

    // Check shutdown between rounds — never spawn round-1 analysts during
    // shutdown.
    if crate::shutdown::shutdown_token().is_cancelled() {
        return Ok(partial_report(
            question,
            &acc,
            "shutdown after decomposition",
        ));
    }

    // Round 1 — one analyst per sub-question (two for high-risk).
    let Some(r1) = round1_research(
        ws,
        question,
        &plan,
        &mut budget,
        &mut ledger,
        &mut run_stats,
    )
    .await
    else {
        return Ok(partial_report(
            question,
            &acc,
            "round 1 skipped — analyst budget exhausted",
        ));
    };
    let (_, _, pending) = acc.absorb(&r1);
    annotate_round(ws, &mut acc, &pending, &mut markers).await;

    // Interim consolidation + conditional gap rounds.
    let gap_outcome = gap_rounds(
        ws,
        question,
        &plan,
        &mut budget,
        &mut ledger,
        &mut acc,
        &mut run_stats,
        &mut markers,
    )
    .await;
    let rounds_used = 2 + gap_outcome.rounds_dispatched;

    // The gap loop may have exited early on shutdown — never run a full
    // synthesis or spawn verification analysts during shutdown.
    if crate::shutdown::shutdown_token().is_cancelled() {
        return Ok(partial_report(question, &acc, "shutdown during gap rounds"));
    }

    // One-shot final synthesis (never regenerated). Fail-open: a synthesis
    // failure delivers the accumulated evidence with an explicit marker
    // instead of an error-only envelope — findings are never lost.
    let synthesis = match synthesize(ws, question, &acc, gap_outcome.abstention.as_deref()).await {
        Ok(s) => s,
        Err(_) if crate::shutdown::shutdown_token().is_cancelled() => {
            return Ok(partial_report(
                question,
                &acc,
                "shutdown before final synthesis",
            ));
        }
        Err(e) => {
            return Ok(partial_report(
                question,
                &acc,
                &format!("synthesis failed: {e}"),
            ));
        }
    };

    // Verification gate (budgeted). Never spawn verifiers during shutdown —
    // deliver the synthesized report as-is (partial is acceptable).
    let verification = if crate::shutdown::shutdown_token().is_cancelled() {
        Vec::new()
    } else {
        research_verification_pass(ws, &acc, &mut budget, &mut ledger, &mut run_stats).await
    };

    // Fail-open markers survive delivery: head-placed so they survive the
    // manager's sandwich truncation of long reports.
    let mut report = String::new();
    if !markers.is_empty() {
        let _ = writeln!(report, "## Run markers");
        for m in &markers {
            let _ = writeln!(report, "- {m}");
        }
        let _ = writeln!(report);
    }
    report.push_str(&synthesis);
    if !verification.is_empty() {
        let _ = writeln!(report);
        let _ = writeln!(report, "## Verification");
        for v in &verification {
            let _ = writeln!(
                report,
                "- {} → **{}** — {}",
                escape_fences(&v.claim),
                v.verdict,
                escape_fences(&v.evidence),
            );
        }
    }
    if !acc.raw_reports.is_empty() {
        let _ = writeln!(report);
        let _ = writeln!(report, "## Failed Analyst Reports");
        for (i, raw) in acc.raw_reports.iter().enumerate() {
            let _ = writeln!(report, "### Report from Analyst {}", i + 1);
            let _ = writeln!(report, "{}", escape_fences(raw));
        }
    }
    let _ = writeln!(report);
    report.push_str(&render_run_summary(
        &run_stats,
        &budget,
        rounds_used,
        &acc,
        gap_outcome.abstention.as_deref(),
        &gap_outcome.unresolved,
        gap_outcome.incomplete.as_deref(),
        &markers,
        start.elapsed(),
    ));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_cap_enforced() {
        let mut budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
        assert!(budget.try_reserve(RESEARCH_MAX_ANALYSTS - 1).is_ok());
        assert!(budget.try_reserve(1).is_ok());
        assert!(
            budget.try_reserve(1).is_err(),
            "spawn cap is unconditional and never refunded"
        );
        assert_eq!(budget.spent, RESEARCH_MAX_ANALYSTS);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_run_guard_one_active_run_per_workspace() {
        let g1 = ResearchRunGuard::try_start("ws");
        assert!(g1.is_some());
        assert!(
            ResearchRunGuard::try_start("ws").is_none(),
            "second run in the same workspace must be rejected"
        );
        assert!(
            ResearchRunGuard::try_start("other").is_some(),
            "a different workspace is unaffected"
        );
        drop(g1);
        assert!(
            ResearchRunGuard::try_start("ws").is_some(),
            "released after the run completes"
        );
    }

    #[test]
    fn test_research_fail_open_envelope() {
        // All-decomposers-failed follows the ask convention: an error envelope
        // with an explicit marker, never a silent drop.
        let envelope =
            build_async_research_message(Err(anyhow::anyhow!("all decomposition analysts failed")));
        assert!(envelope.contains("<research-result>"), "{envelope}");
        assert!(
            envelope.contains("An error occurred: all decomposition analysts failed"),
            "{envelope}"
        );
        assert!(
            envelope.ends_with("</research-result>"),
            "envelope must close: {envelope}"
        );
    }

    #[test]
    fn test_marginal_gap_saturation() {
        let gaps = vec![
            Gap {
                kind: "unanswered".into(),
                item: "price of X".into(),
                traces_to: 0,
                evidence_seen: String::new(),
            },
            Gap {
                kind: "unanswered".into(),
                item: "price  of x".into(),
                traces_to: 0,
                evidence_seen: String::new(),
            },
            Gap {
                kind: "unanswered".into(),
                item: "price of X".into(),
                traces_to: 1,
                evidence_seen: String::new(),
            },
        ];
        assert!(
            is_marginal_gap_saturated(&gaps),
            "2/3 near-duplicate gaps exceed the 50% no-progress ratio"
        );
    }

    #[test]
    fn test_validate_gap_list_traces_to_plan() {
        let plan = MergedPlan {
            sub_questions: vec![
                MergedSubQuestion {
                    question: "What is the price of X?".into(),
                    evidence_needed: "pricing page".into(),
                    risk: "low".into(),
                    from_plan: 0,
                    also_in_plans: vec![],
                },
                MergedSubQuestion {
                    question: "Who maintains X?".into(),
                    evidence_needed: "repo metadata".into(),
                    risk: "medium".into(),
                    from_plan: 0,
                    also_in_plans: vec![],
                },
            ],
            dropped: vec![],
        };
        let in_range = GapList {
            gaps: vec![Gap {
                kind: "unanswered".into(),
                item: "exact price".into(),
                traces_to: 0,
                evidence_seen: String::new(),
            }],
        };
        assert!(validate_gap_list(&in_range, &plan).is_ok());
        let out_of_range = GapList {
            gaps: vec![Gap {
                kind: "unanswered".into(),
                item: "unrelated".into(),
                traces_to: 5,
                evidence_seen: String::new(),
            }],
        };
        assert!(
            validate_gap_list(&out_of_range, &plan).is_err(),
            "out-of-range traces_to is rejected — index-range validation guarantees traceability"
        );
    }

    #[test]
    fn test_validate_merged_plan_coverage() {
        let sq = |q: &str| SubQuestion {
            question: q.into(),
            evidence_needed: "e".into(),
            risk: "low".into(),
        };
        let plans = vec![
            DecompositionPlan {
                sub_questions: vec![sq("q1"), sq("q2")],
            },
            DecompositionPlan {
                sub_questions: vec![sq("q1"), sq("q3")],
            },
            DecompositionPlan {
                sub_questions: vec![sq("q4"), sq("q5")],
            },
        ];
        let base = |also: bool, dropped: bool| MergedPlan {
            sub_questions: vec![
                MergedSubQuestion {
                    question: "q1".into(),
                    evidence_needed: "e".into(),
                    risk: "low".into(),
                    from_plan: 0,
                    also_in_plans: if also { vec![1] } else { vec![] },
                },
                MergedSubQuestion {
                    question: "q2".into(),
                    evidence_needed: "e".into(),
                    risk: "low".into(),
                    from_plan: 0,
                    also_in_plans: vec![],
                },
                MergedSubQuestion {
                    question: "q3".into(),
                    evidence_needed: "e".into(),
                    risk: "low".into(),
                    from_plan: 1,
                    also_in_plans: vec![],
                },
                MergedSubQuestion {
                    question: "q4".into(),
                    evidence_needed: "e".into(),
                    risk: "low".into(),
                    from_plan: 2,
                    also_in_plans: vec![],
                },
            ],
            dropped: if dropped {
                vec![DroppedSubQuestion {
                    question: "q5".into(),
                    evidence_needed: "e".into(),
                    risk: "low".into(),
                    reason: "redundant".into(),
                }]
            } else {
                vec![]
            },
        };
        assert!(
            validate_merged_plan(&base(true, true), &plans).is_ok(),
            "full coverage via from_plan + also_in_plans + dropped"
        );
        assert!(
            validate_merged_plan(&base(false, true), &plans).is_err(),
            "silent dropout: plan 1's q1 is never covered"
        );
        assert!(
            validate_merged_plan(&base(true, false), &plans).is_err(),
            "silent dropout: q5 is never covered"
        );
        let mut bad = base(true, true);
        bad.sub_questions[0].from_plan = 9;
        assert!(
            validate_merged_plan(&bad, &plans).is_err(),
            "out-of-range from_plan is rejected"
        );
        let mut bad = base(true, true);
        bad.sub_questions[3].question = "q9".into();
        assert!(
            validate_merged_plan(&bad, &plans).is_err(),
            "a merged tuple not found verbatim in its cited plan is rejected"
        );
    }

    // ── Orchestrator helpers (no provider needed) ───────────────────────

    #[test]
    fn test_evidence_absorb_counts_novelty() {
        // URLs and unanswered aspects dedup inside absorb exactly as before;
        // claims are returned as a pending list — novelty is decided by the
        // annotation pass, never dropped here.
        let mut acc = AccumulatedEvidence::default();
        let round1 = EvidenceRound {
            urls: vec!["u1".into(), "u2".into()],
            claims: vec![
                Claim {
                    claim: "alpha is true".into(),
                    source: "u1".into(),
                    confidence: "high".into(),
                    contradictions: vec![],
                },
                Claim {
                    claim: "beta is true".into(),
                    source: "u2".into(),
                    confidence: "medium".into(),
                    contradictions: vec![],
                },
            ],
            unanswered: vec!["how beta relates to alpha".into()],
            ..Default::default()
        };
        let (urls, unanswered, pending) = acc.absorb(&round1);
        assert_eq!((urls, unanswered, pending.len()), (2, 1, 2));
        let round2 = EvidenceRound {
            urls: vec!["u1".into(), "u3".into()],
            claims: vec![
                Claim {
                    claim: "alpha is true".into(),
                    source: "u1".into(),
                    confidence: "high".into(),
                    contradictions: vec![],
                },
                Claim {
                    claim: "gamma is true".into(),
                    source: "u3".into(),
                    confidence: "low".into(),
                    contradictions: vec![],
                },
            ],
            unanswered: vec!["how beta relates to alpha".into(), "delta timeline".into()],
            ..Default::default()
        };
        let (urls, unanswered, pending) = acc.absorb(&round2);
        assert_eq!(
            (urls, unanswered, pending.len()),
            (1, 1, 2),
            "only new URL (u3) and unanswered aspect count; every claim stays pending for annotation"
        );
        assert_eq!(
            acc.unanswered,
            vec!["how beta relates to alpha", "delta timeline"],
            "unanswered aspects accumulate deduplicated across rounds"
        );
    }

    #[test]
    fn test_apply_annotations_contradicts_appends_and_links() {
        // A contradicting claim is never deduped away: it is appended AND
        // linked to the existing claim so the verification gate targets both
        // sides of the dispute.
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(Claim {
            claim: "alpha costs $100 in 2024".into(),
            source: "u1".into(),
            confidence: "medium".into(),
            contradictions: vec![],
        });
        let pending = vec![Claim {
            claim: "alpha costs $200 in 2024".into(),
            source: "u2".into(),
            confidence: "high".into(),
            contradictions: vec![],
        }];
        let pass = AnnotationPass {
            annotations: vec![ClaimAnnotation {
                new_id: 0,
                verdict: "contradicts".into(),
                existing_id: Some(0),
                proof: Some("alpha costs $100 in 2024".into()),
                contradiction: Some("price differs: $200 vs $100".into()),
            }],
        };
        let novel = acc.apply_annotations(&pass, &pending);
        assert_eq!(novel, 1, "a contradicting claim is new evidence");
        assert_eq!(
            acc.claims.len(),
            2,
            "the contradiction is preserved, never merged away"
        );
        assert_eq!(
            acc.claims[0].contradictions,
            vec!["price differs: $200 vs $100"],
            "the existing claim carries the contradiction note — the verification gate fires"
        );
        assert!(
            acc.claims[1]
                .contradictions
                .contains(&"alpha costs $100 in 2024".to_string()),
            "the new claim links back to the existing one"
        );
    }

    #[test]
    fn test_apply_annotations_merges_sources_and_upgrades_confidence() {
        // A duplicate merges into the existing claim: confidence upgraded,
        // sources joined deduplicated — including multi-source '; ' joins on
        // both sides (duplicate entries must never appear in the merged list).
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(Claim {
            claim: "alpha is true".into(),
            source: "u1; u2".into(),
            confidence: "low".into(),
            contradictions: vec![],
        });
        let pending = vec![Claim {
            claim: "alpha is true".into(),
            source: "u2; u3".into(),
            confidence: "high".into(),
            contradictions: vec![],
        }];
        let pass = AnnotationPass {
            annotations: vec![ClaimAnnotation {
                new_id: 0,
                verdict: "duplicate".into(),
                existing_id: Some(0),
                proof: Some("alpha is true".into()),
                contradiction: None,
            }],
        };
        let novel = acc.apply_annotations(&pass, &pending);
        assert_eq!(novel, 0, "a duplicate is never counted as novel");
        assert_eq!(
            acc.claims.len(),
            1,
            "a duplicate is never dropped — it merges into the existing claim"
        );
        let c = &acc.claims[0];
        assert_eq!(
            c.confidence, "high",
            "a higher-confidence re-statement upgrades the merged claim"
        );
        assert_eq!(
            c.source, "u1; u2; u3",
            "sources merge across rounds without duplicates"
        );
    }
}
