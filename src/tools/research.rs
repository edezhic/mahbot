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
    AnalystFindings, Claim, ClaimKey, VerificationResult, VerificationTarget,
    build_async_result_envelope, dispatch_verifiers, escape_fences, extract_query_telemetry,
    load_analyst_angles, max_confidence, normalize_claim,
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
/// Embedding similarity threshold for claim novelty (0.85–0.90 band).
const NOVELTY_SIMILARITY_THRESHOLD: f32 = 0.85;
/// Marginal-gap saturation: >50% near-duplicate gaps = no-progress stop.
const MARGINAL_GAP_SATURATION_PERCENT: usize = 50;
/// Explicit marker when the orchestrator cannot determine the remaining gaps.
const GAP_EXTRACTION_FAILED: &str = "gap extraction failed — remaining gaps unknown";

// ── Shared orchestration types ───────────────────────────────────────────

/// A sub-question from the round-0 decomposition plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubQuestion {
    question: String,
    evidence_needed: String,
    /// "low" | "medium" | "high" — how hard solid evidence is to find.
    risk: String,
}

/// Round-0 decomposition plan (one per decomposer, then one merged plan).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecompositionPlan {
    sub_questions: Vec<SubQuestion>,
}

/// One item in the interim gap list.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Gap {
    /// "unanswered" | "partially_answered" | "contradictory" | "low_evidence".
    #[serde(rename = "type")]
    kind: String,
    /// The specific missing claim a fresh analyst could hunt for.
    item: String,
    /// The round-0 sub-question this gap traces to.
    traces_to: String,
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
    /// Claim match keys (normalized text + optional embedding) for
    /// near-duplicate detection across rounds.
    claim_keys: Vec<ClaimKey>,
    claims: Vec<Claim>,
    /// Deduplicated analysts' self-reported unanswered aspects.
    unanswered: Vec<String>,
    unanswered_keys: HashSet<String>,
    /// Raw responses of analysts whose structured extraction failed — never
    /// silently lost.
    raw_reports: Vec<String>,
}

impl AccumulatedEvidence {
    /// Absorb a round's evidence, returning `(novel_urls, novel_claims)`
    /// counts for the negative-evidence-saturation check. Claim novelty uses
    /// embedding similarity (~0.85) when the local embedder is available and
    /// degrades to deterministic exact-match comparison otherwise; claims
    /// disagreeing on numeric literals are always novel (never deduped).
    fn absorb(&mut self, round: &EvidenceRound) -> (usize, usize) {
        let novel_urls = round
            .urls
            .iter()
            .filter(|u| self.urls.insert((*u).clone()))
            .count();
        let mut novel_claims = 0usize;
        for claim in &round.claims {
            let key = ClaimKey::new(&claim.claim);
            if let Some(idx) = self
                .claim_keys
                .iter()
                .position(|k| k.equivalent_to(&key, NOVELTY_SIMILARITY_THRESHOLD))
            {
                // Near-duplicate re-statement: preserve any new contradicting
                // evidence and distinct source instead of dropping them —
                // ask's merge_and_grade accumulates the same way. A later
                // round re-stating the claim with higher confidence upgrades
                // it; sources merge deduplicated across rounds (both sides
                // may already be multi-source '; ' joins).
                let existing = &mut self.claims[idx];
                for c in &claim.contradictions {
                    if !existing.contradictions.contains(c) {
                        existing.contradictions.push(c.clone());
                    }
                }
                existing.confidence = max_confidence(&existing.confidence, &claim.confidence);
                if !claim.source.is_empty() {
                    let mut merged: Vec<String> = existing
                        .source
                        .split("; ")
                        .filter(|s| !s.trim().is_empty())
                        .map(|s| s.trim().to_string())
                        .collect();
                    for s in claim.source.split("; ") {
                        let s = s.trim();
                        if !s.is_empty() && !merged.iter().any(|m| m == s) {
                            merged.push(s.to_string());
                        }
                    }
                    existing.source = merged.join("; ");
                }
                continue;
            }
            self.claim_keys.push(key);
            self.claims.push(claim.clone());
            novel_claims += 1;
        }
        for u in &round.unanswered {
            let key = normalize_claim(u);
            if !key.is_empty() && self.unanswered_keys.insert(key) {
                self.unanswered.push(u.clone());
            }
        }
        for r in &round.raw_reports {
            if !self.raw_reports.contains(r) {
                self.raw_reports.push(r.clone());
            }
        }
        (novel_urls, novel_claims)
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
/// consolidated plan via a single orchestrator LLM call. Budget: 3
/// decomposers. The x3 redundancy lives here, at the steering decision.
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
) -> Result<DecompositionPlan> {
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
    merge_decomposition_plans(ws, question, &valid).await
}

/// Merge 1–3 independent plans into one consolidated plan (orchestrator
/// extraction call — not budgeted).
async fn merge_decomposition_plans(
    ws: &Workspace,
    question: &str,
    plans: &[DecompositionPlan],
) -> Result<DecompositionPlan> {
    let plans_json = serde_json::to_string_pretty(plans)?;
    let prompt = substitute(
        &load_prompt("research/decompose_merge.md"),
        &[("{{question}}", question), ("{{plans}}", &plans_json)],
    );
    orchestrator_extract::<DecompositionPlan>(
        ws,
        "decompose_merge",
        &prompt,
        Some(&validate_decomposition),
    )
    .await
}

/// Validate a merged decomposition plan: 4–6 sub-questions (fail-closed
/// inside the extraction retry loop).
fn validate_decomposition(plan: &DecompositionPlan) -> Result<(), String> {
    let n = plan.sub_questions.len();
    if (4..=6).contains(&n) {
        Ok(())
    } else {
        Err(format!(
            "decomposition plan has {n} sub-questions, expected 4-6"
        ))
    }
}

// ── Round 1: one analyst per sub-question ────────────────────────────────

/// Round 1: one analyst per sub-question; two for high-risk items (the
/// second gets a decorrelated research angle). Returns the round's evidence,
/// or `None` when the analyst budget is exhausted before dispatch.
async fn round1_research(
    ws: &Workspace,
    question: &str,
    plan: &DecompositionPlan,
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
    plan: &DecompositionPlan,
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
    orchestrator_extract::<GapList>(ws, "gap_extract", &prompt, None)
        .await
        .ok()
}

/// The gap items as plain strings (for the report's unresolved list).
fn gap_items(gaps: &[Gap]) -> Vec<String> {
    gaps.iter().map(|g| g.item.clone()).collect()
}

/// Keep only gaps traceable to a round-0 sub-question (lenient bidirectional
/// substring match on the sub-question text).
fn triage_gaps(gap_list: &GapList, plan: &DecompositionPlan) -> Vec<Gap> {
    gap_list
        .gaps
        .iter()
        .filter(|g| {
            plan.sub_questions.iter().any(|sq| {
                let g_t = g.traces_to.trim().to_lowercase();
                let sq_q = sq.question.trim().to_lowercase();
                g_t.contains(&sq_q) || sq_q.contains(&g_t)
            })
        })
        .cloned()
        .collect()
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
                            "- [{}] {} (traces to: {})",
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
async fn gap_rounds(
    ws: &Workspace,
    question: &str,
    plan: &DecompositionPlan,
    budget: &mut ResearchBudget,
    ledger: &mut QueryLedger,
    acc: &mut AccumulatedEvidence,
    run_stats: &mut RunStats,
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
        let traceable = triage_gaps(&gap_list, plan);
        if traceable.is_empty() {
            // Coverage completion.
            return outcome;
        }
        if is_marginal_gap_saturated(&traceable) {
            outcome.unresolved = gap_items(&traceable);
            return outcome;
        }
        let width = GAP_ROUND_WIDTHS[round_index.min(GAP_ROUND_WIDTHS.len() - 1)];
        round_index += 1;
        let targeted: Vec<&Gap> = traceable.iter().take(width).collect();
        if budget.try_reserve(targeted.len()).is_err() {
            outcome.unresolved = gap_items(&traceable);
            return outcome;
        }
        let runs = run_gap_round(ws, question, &targeted, ledger).await;
        let round = collect_evidence(&runs, ledger, run_stats);
        let (new_urls, new_claims) = acc.absorb(&round);
        outcome.rounds_dispatched += 1;
        // A round whose analysts only re-asked already-asked queries counts
        // as no-progress toward saturation (repeat queries are never
        // pre-dispatch-droppable — concurrent analysts generate them live).
        let all_repeat_queries = round.queries > 0 && round.queries == round.repeat_queries;
        if (new_urls == 0 && new_claims == 0) || all_repeat_queries {
            quiet_rounds += 1;
            if quiet_rounds >= QUIET_ROUNDS_TO_STOP {
                // Negative-evidence saturation: answerability abstain check.
                outcome.abstention = check_answerability(ws, question, acc).await;
                outcome.unresolved = gap_items(&traceable);
                return outcome;
            }
        } else {
            quiet_rounds = 0;
        }
        // Refresh the gap list from the accumulated evidence (gaps shrink).
        let Some(next_gap_list) = extract_gap_list(ws, question, acc, plan).await else {
            outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
            outcome.unresolved = gap_items(&traceable);
            return outcome;
        };
        gap_list = next_gap_list;
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
            "{}. [{}] {} — source: {source}",
            i + 1,
            c.confidence,
            c.claim,
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
    wall: Duration,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Run Summary");
    let _ = writeln!(out, "- rounds used: {rounds_used}");
    if let Some(reason) = incomplete {
        let _ = writeln!(out, "- gap rounds incomplete: {reason}");
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

    // Round 0 — three independent decomposition plans merged into one.
    let plan = match round0_decompose(ws, question, &mut budget, &mut run_stats).await {
        Ok(plan) => plan,
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
    acc.absorb(&r1);

    // Interim consolidation + conditional gap rounds.
    let gap_outcome = gap_rounds(
        ws,
        question,
        &plan,
        &mut budget,
        &mut ledger,
        &mut acc,
        &mut run_stats,
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

    let mut report = synthesis;
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
                traces_to: "sq1".into(),
                evidence_seen: String::new(),
            },
            Gap {
                kind: "unanswered".into(),
                item: "price  of x".into(),
                traces_to: "sq1".into(),
                evidence_seen: String::new(),
            },
            Gap {
                kind: "unanswered".into(),
                item: "price of X".into(),
                traces_to: "sq2".into(),
                evidence_seen: String::new(),
            },
        ];
        assert!(
            is_marginal_gap_saturated(&gaps),
            "2/3 near-duplicate gaps exceed the 50% no-progress ratio"
        );
    }

    #[test]
    fn test_triage_gaps_traces_to_plan() {
        let plan = DecompositionPlan {
            sub_questions: vec![
                SubQuestion {
                    question: "What is the price of X?".into(),
                    evidence_needed: "pricing page".into(),
                    risk: "low".into(),
                },
                SubQuestion {
                    question: "Who maintains X?".into(),
                    evidence_needed: "repo metadata".into(),
                    risk: "medium".into(),
                },
            ],
        };
        let list = GapList {
            gaps: vec![
                Gap {
                    kind: "unanswered".into(),
                    item: "exact price".into(),
                    traces_to: "price of X".into(),
                    evidence_seen: String::new(),
                },
                Gap {
                    kind: "unanswered".into(),
                    item: "totally unrelated".into(),
                    traces_to: "quantum physics".into(),
                    evidence_seen: String::new(),
                },
            ],
        };
        let triaged = triage_gaps(&list, &plan);
        assert_eq!(triaged.len(), 1);
        assert_eq!(triaged[0].item, "exact price");
    }

    // ── Orchestrator helpers (no provider needed) ───────────────────────

    #[test]
    fn test_evidence_absorb_counts_novelty() {
        // ClaimKey skips embedding in test builds (cfg!(test)), so absorb
        // deterministically exercises the exact-match novelty fallback with
        // no process-global CONFIG/embedder mutation.
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
        let (urls, claims) = acc.absorb(&round1);
        assert_eq!((urls, claims), (2, 2));
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
        let (urls, claims) = acc.absorb(&round2);
        assert_eq!(
            (urls, claims),
            (1, 1),
            "only new URL (u3) and novel claim (gamma) count"
        );
        assert_eq!(
            acc.unanswered,
            vec!["how beta relates to alpha", "delta timeline"],
            "unanswered aspects accumulate deduplicated across rounds"
        );
    }

    #[test]
    fn test_absorb_keeps_numeric_contradictions() {
        // Price-differing claims sharing a year are never near-duplicates —
        // deduping them would silently drop the contradicted claim.
        let mut acc = AccumulatedEvidence::default();
        let round = EvidenceRound {
            claims: vec![
                Claim {
                    claim: "alpha costs $100 in 2024".into(),
                    source: "u1".into(),
                    confidence: "medium".into(),
                    contradictions: vec![],
                },
                Claim {
                    claim: "alpha costs $200 in 2024".into(),
                    source: "u2".into(),
                    confidence: "medium".into(),
                    contradictions: vec![],
                },
            ],
            ..Default::default()
        };
        let (_, novel) = acc.absorb(&round);
        assert_eq!(
            novel, 2,
            "both price claims are novel — the contradiction is preserved"
        );
        assert_eq!(acc.claims.len(), 2);
    }

    #[test]
    fn test_absorb_merges_sources_and_upgrades_confidence() {
        // A later round re-stating a claim upgrades its confidence and merges
        // sources deduplicated — including multi-source '; ' joins on both
        // sides (duplicate entries must never appear in the merged list).
        // `urls` mirrors collect_evidence, which pushes each claim source as
        // a URL entry.
        let mut acc = AccumulatedEvidence::default();
        let round1 = EvidenceRound {
            urls: vec!["u1; u2".into()],
            claims: vec![Claim {
                claim: "alpha is true".into(),
                source: "u1; u2".into(),
                confidence: "low".into(),
                contradictions: vec![],
            }],
            ..Default::default()
        };
        acc.absorb(&round1);
        let round2 = EvidenceRound {
            urls: vec!["u2; u3".into()],
            claims: vec![Claim {
                claim: "alpha is true".into(),
                source: "u2; u3".into(),
                confidence: "high".into(),
                contradictions: vec![],
            }],
            ..Default::default()
        };
        let (urls, claims) = acc.absorb(&round2);
        assert_eq!(
            (urls, claims),
            (1, 0),
            "the round-2 source string is novel; the claim itself is a re-statement"
        );
        assert_eq!(acc.claims.len(), 1);
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
