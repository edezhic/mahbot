//! ResearchTool — Manager-only deep multi-round research orchestrator.
//!
//! Unlike [`AskTool`](super::ask::AskTool) (one round of parallel analysts
//! for quick clarification), `research` decomposes the question into
//! sub-questions via three independent plans (merged by id-based coverage),
//! runs one analyst per sub-question, then runs conditional gap rounds with
//! fresh analysts targeting only the named gaps. Stopping is artifact-based —
//! coverage completion, answerability abstention (checked on every structural
//! quiet round), a verification gate, and a hard agent-spawn cap (never agent
//! self-assessment). Exactly one envelope is delivered asynchronously to the
//! Manager; intermediate rounds never reach the user.
//!
//! Budgeting is by analysts spawned (decomposers, round-1 researchers,
//! gap-round researchers, and verification analysts all count); orchestrator
//! coordination LLM calls do not. The cap is enforced at reservation time and
//! never refunded. No per-agent tool-call caps and no wall-clock limit — the
//! existing global iteration backstop and retry machinery remain untouched.

use crate::agent::{chat_request, role_tools_and_specs, run_agent};
use crate::message_router::{self, AgentJob, JobKind};
use crate::prompt::{load_prompt, substitute};
use crate::retry::FailureClass;
use crate::tools::Tool;
use crate::tools::ask::{
    AnalystFindings, Claim, RoundMember, VerificationResult, VerificationTarget,
    await_round_members, build_async_result_envelope, complete_durable_job_and_route,
    dispatch_claim_verifiers, escape_fences, extract_query_telemetry,
    extract_query_telemetry_from_history, load_analyst_angles, max_confidence, normalize_claim,
    round_timeout,
};
use crate::{ChatMessage, ChatRequest, ChatRequestMeta, Role, ToolSpec, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

// ── Constants (module-local defaults) ────────────────────────────────────

/// Hard cap on research analysts spawned per run — decomposers, round-1
/// researchers, gap-round researchers, and verification analysts all count;
/// orchestrator coordination LLM calls do not. Enforced at reservation time,
/// never refunded.
const RESEARCH_MAX_ANALYSTS: usize = 30;
/// Round-0 decomposition fan-out (three independent plans).
const DECOMPOSE_FAN_OUT: usize = 3;
/// Gap-round dispatch widths — rounds shrink as they progress.
const GAP_ROUND_WIDTHS: &[usize] = &[4, 3, 2];
/// Explicit marker when the orchestrator cannot determine the remaining gaps.
const GAP_EXTRACTION_FAILED: &str = "gap extraction failed — remaining gaps unknown";
/// Explicit marker when the plan merge fails — the run falls back to the
/// first valid decomposition plan verbatim.
const PLAN_MERGE_FAILED: &str = "plan merge failed — using first valid decomposition plan verbatim";
/// Explicit marker when the claim annotation pass exhausts its retries — all
/// pending claims are treated as novel.
const CLAIM_ANNOTATION_FAILED: &str = "claim annotation failed — all new claims treated as novel";
/// Explicit marker when the confirm pass over a round's mutating annotation
/// links fails entirely — every mutating verdict is treated as weak.
const CONFIRM_FAILED: &str =
    "annotation link confirmation failed — mutating links treated as weak/unconfirmed";
/// Minimum remaining round time for a coder round to start (a coder run is
/// not interrupted — starting one with less left would eat the subsequent
/// gap rounds).
const CODER_MIN_REMAINING: Duration = Duration::from_mins(30);
/// Cap on the accumulated outside-zone commands in the checkpointed state
/// blob. The cleaner ticket truncates at 32 KiB anyway — 2× gives the ticket
/// its full body while bounding long runs' checkpoints (newest commands win).
const MAX_STATE_COMMANDS_BYTES: usize = 64 * 1024;
/// Default bound on the wrap-up stage: concurrent extraction of
/// deadline-aborted analysts' accumulated findings (env-overridable via
/// `MAHBOT_WRAP_UP_TIMEOUT_SECS`, distinct from the round deadline
/// `MAHBOT_ROUND_TIMEOUT_SECS`). Counted from the stage's own start.
const DEFAULT_WRAP_UP_TIMEOUT_SECS: u64 = 5 * 60;

/// Resume pointer for a durable research run: the 5-value stage
/// enum (plus implicit Done).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResearchStage {
    Decompose,
    Round1,
    GapRounds,
    Verification,
    Synthesis,
}

/// The research_jobs.state blob: ONE JSON column carrying everything needed
/// to resume a run at the checkpointed stage. RunStats is NOT stored (see
/// RunStats pin) — the resumed report's "Run Summary" undercounts the
/// pre-crash segment and the report carries a one-line best-effort note.
#[derive(Debug, Serialize, Deserialize)]
struct ResearchState {
    stage: ResearchStage,
    plan: Option<MergedPlan>,
    gap_list: Option<GapList>,
    acc: AccumulatedEvidence,
    ledger: QueryLedger,
    markers: Vec<String>,
    gap_outcome: GapRoundsOutcome,
    budget_spent: usize,
    /// Gap-loop resume pointer: the next gap round to dispatch (the gap list
    /// is stored; empty unresolved + gap_outcome.rounds_dispatched = k means
    /// "continue at round k+1").
    round_index: usize,
    verification: Vec<VerificationResult>,
    /// Accumulated outside-zone shell commands (write-intent + zone filter
    /// applied at collection) from completed rounds' sessions — collected
    /// incrementally at dispatch so long runs don't lose early sessions to the
    /// transient TTL. Capped at [`MAX_STATE_COMMANDS_BYTES`] so a long run's
    /// checkpoint blob stays bounded (the ticket body is capped at 32 KiB).
    #[serde(default)]
    commands: Vec<String>,
    /// Gap-loop round keys after which a coder round already ran (0 = the
    /// pre-loop coder, k = after gap round k). Boot resume never re-runs a
    /// completed coder round (no duplicate prototypes / LLM spend).
    #[serde(default)]
    coder_rounds_done: Vec<usize>,
}

impl Default for ResearchState {
    fn default() -> Self {
        Self {
            stage: ResearchStage::Decompose,
            plan: None,
            gap_list: None,
            acc: AccumulatedEvidence::default(),
            ledger: QueryLedger::default(),
            markers: Vec::new(),
            gap_outcome: GapRoundsOutcome::default(),
            budget_spent: 0,
            round_index: 0,
            verification: Vec::new(),
            commands: Vec::new(),
            coder_rounds_done: Vec::new(),
        }
    }
}

impl ResearchState {
    /// Load the persisted state for a job (or a fresh default).
    async fn load(job_id: &str) -> Self {
        let row = crate::session::store()
            .conn
            .query_optional(
                "SELECT state FROM research_jobs WHERE id = ?1",
                crate::turso::params![job_id],
                |r| r.get::<String>(0),
            )
            .await
            .ok()
            .flatten();
        let Some(json) = row else {
            return Self::default();
        };
        // state='{}' is the spawn-time seed — never a valid ResearchState, so
        // treat it as a silent fresh run rather than corruption.
        if json.trim().is_empty() || json == "{}" {
            return Self::default();
        }
        match serde_json::from_str::<ResearchState>(&json) {
            Ok(mut s) => {
                s.acc.rebuild_keys();
                s
            }
            Err(e) => {
                tracing::warn!(job = %job_id, error = %e, "Research state unreadable — fresh run");
                Self::default()
            }
        }
    }

    /// Checkpoint the state blob + jobs updated_at touch in ONE transaction
    /// (both rows live in sessions.db — the documented single transaction
    /// domain), so a crash can't advance research_jobs.state without the
    /// matching jobs touch. retry_count is deliberately untouched: the boot
    /// scan's MAX_BOOT_REDISPATCH bump is the only writer and must survive
    /// checkpoints. A failed checkpoint logs a structured warning and the run
    /// continues.
    async fn save(&self, job_id: &str) {
        let json = serde_json::to_string(self).unwrap_or_default();
        let now = crate::turso::now();
        let conn = &crate::session::store().conn;
        let tx = match conn.begin_tx().await {
            Ok(tx) => tx,
            Err(e) => {
                tracing::warn!(job = %job_id, error = %e, "Research checkpoint: failed to begin transaction");
                return;
            }
        };
        let outcome: Result<()> = async {
            tx.execute(
                "UPDATE research_jobs SET state = ?1, updated_at = ?2 WHERE id = ?3",
                crate::turso::params![json, now.clone(), job_id],
            )
            .await?;
            tx.execute(
                "UPDATE jobs SET status = ?1, updated_at = ?2 WHERE id = ?3",
                crate::turso::params![crate::jobs::RowStatus::Launched.as_str(), now, job_id],
            )
            .await?;
            Ok(())
        }
        .await;
        match outcome {
            Ok(()) => {
                if let Err(e) = tx.commit().await {
                    tracing::warn!(job = %job_id, error = %e, "Research checkpoint: failed to commit");
                }
            }
            Err(e) => {
                tracing::warn!(job = %job_id, error = %e, "Research checkpoint failed — state not persisted");
                let _ = tx.rollback().await;
            }
        }
    }

    /// Collect the given dispatched agents' outside-zone shell commands from
    /// their persisted sessions (right after each completed round — early
    /// sessions of >8h runs are TTL'd, so collection must be incremental). The
    /// zone filter applies here, at capture time, so in-zone commands never
    /// enter the state blob. The accumulated blob is capped — the ticket body
    /// itself is capped at 32 KiB, so older commands are dropped (newest win).
    async fn capture_round(&mut self, agent_ids: &[String], run_root: &Path) {
        let fresh =
            crate::research_cleanup::collect_agent_shell_commands(agent_ids, run_root).await;
        for cmd in fresh {
            if !self.commands.contains(&cmd) {
                self.commands.push(cmd);
            }
        }
        cap_commands(&mut self.commands, MAX_STATE_COMMANDS_BYTES);
    }
}

/// Drop commands until the accumulated byte length fits `cap` (newest evidence
/// wins; the ticket body truncates at 32 KiB regardless). A single command
/// larger than the cap is dropped ON ITS OWN — it could never fit the ticket
/// body, and dropping the older fitting commands for it would lose the only
/// evidence that ever could be reported.
fn cap_commands(commands: &mut Vec<String>, cap: usize) {
    while commands.last().is_some_and(|c| c.len() > cap) {
        commands.pop();
    }
    // Single pass to the drop boundary, then one drain (remove(0) in a loop
    // would be O(n²)).
    let mut total: usize = commands.iter().map(String::len).sum();
    let mut drop = 0usize;
    while drop < commands.len() && total > cap {
        total -= commands[drop].len();
        drop += 1;
    }
    if drop > 0 {
        commands.drain(..drop);
    }
}

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

/// A merged sub-question carrying provenance by global flat item id: the
/// input-plan item it is a verbatim copy of, plus every other plan's item
/// containing the identical tuple. question/evidence_needed/risk are resolved
/// by the system from the cited item after validation — never trusted from
/// the model's copy.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MergedSubQuestion {
    /// Global flat id of the input-plan item this sub-question copies
    /// (ids are assigned in (plan, item) order across all plans).
    from_id: usize,
    /// Global flat ids of identical items in other plans.
    #[serde(default)]
    also_ids: Vec<usize>,
    #[serde(default)]
    question: String,
    #[serde(default)]
    evidence_needed: String,
    #[serde(default)]
    risk: String,
}

/// A round-0 plan item the merge explicitly dropped (never silently omitted),
/// cited by its global flat id.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DroppedSubQuestion {
    /// Global flat id of the dropped input-plan item.
    id: usize,
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
}

/// The interim gap list.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct GapList {
    gaps: Vec<Gap>,
}

/// Answerability verdict after a structural quiet round.
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
            // Catch panics so the Manager ALWAYS receives the single result
            // envelope — a panic in the dispatch task would otherwise leave
            // the caller waiting forever on a result that can never arrive.
            let run = std::panic::AssertUnwindSafe(async {
                dispatch_durable_research(
                    &ws,
                    &question,
                    caller_role,
                    user_name.clone(),
                    channel.clone(),
                )
                .await
            })
            .catch_unwind()
            .await;
            let envelope = match run {
                Ok(Some(envelope)) => envelope,
                Ok(None) => {
                    // Shutdown/drain abort: the run stays alive for the next
                    // boot's resume — nothing is routed now (the result and
                    // artifacts arrive at the real terminalization).
                    tracing::info!("Research run aborted by shutdown/drain — resumes at boot");
                    return;
                }
                Err(panic) => {
                    let panic = crate::util::panic_message(&*panic);
                    tracing::error!(panic = %panic, "research dispatch panicked");
                    AgentJob {
                        content: build_async_research_message(&Err(anyhow::anyhow!(
                            "research dispatch panicked: {panic}"
                        ))),
                        workspace_name: ws.name.clone(),
                        user_name,
                        channel,
                        kind: JobKind::ResearchResult,
                        role: caller_role,
                        reply_target: None,
                        pending_job_id: None,
                    }
                }
            };

            // Route exactly one final envelope to the caller's agent channel
            // (the completion helper's own copy — persisted and routed copies
            // can never drift). A drain in progress may drop the routed copy
            // (the consumer has stopped pulling), but the pending row survives
            // for boot replay — at-least-once.
            message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
        });

        Ok(
            "Deep research dispatched. One report will be delivered when the run completes."
                .to_string(),
        )
    }
}

/// Durable deep-research dispatch: SPAWN the job (one tx) → run the resumable
/// orchestrator (checkpoints at stage boundaries) → COMPLETE (one tx — INSERT
/// pending_jobs envelope + DELETE jobs row). Returns `Some(envelope)` on real
/// terminalizations — the envelope is the completion helper's own copy
/// (pending_job_id set by the tx), routed as-is so persisted and routed copies
/// cannot drift. Returns `None` on a shutdown/drain abort: the run STAYS ALIVE
/// for the next boot's resume (the jobs row is left 'launched'), nothing is
/// terminalized and nothing is routed (design pin).
async fn dispatch_durable_research(
    ws: &Workspace,
    question: &str,
    caller_role: Role,
    user_name: String,
    channel: String,
) -> Option<AgentJob> {
    let job_id = crate::generate_id();
    let spawn = async {
        // SPAWN: one tx — jobs + research_jobs child row (the shared in-tx
        // child pattern; a crash mid-spawn leaves either all or none).
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            &job_id,
            question,
            &ws.name,
            &user_name,
            &channel,
            caller_role,
            &[],
            &crate::jobs::SpawnChild::Research {
                question: question.to_string(),
            },
        )
        .await
    };
    // Spawn failures still route an error envelope — never a silent drop.
    let spawn_out = spawn.await;
    let spawned = spawn_out.is_ok();
    let exit = match spawn_out {
        Ok(()) => run_deep_research(ws, question, &job_id, false).await,
        Err(e) => ResearchExit::Terminal(Err(e)),
    };
    let result = match exit {
        // Abort: the run stays alive for the next boot's resume — the jobs row
        // survives as 'launched' and boot-recovery re-enters the checkpointed
        // run, so the result (and the artifacts) arrive at the real
        // terminalization. Nothing is written or routed here (design pin:
        // "Shutdown/drain abort НЕ терминализация — ран жив, resume позже").
        ResearchExit::Aborted => {
            tracing::info!(
                job = %job_id,
                "Research run aborted by shutdown/drain — resumes at next boot"
            );
            return None;
        }
        ResearchExit::Terminal(result) => result,
    };
    // Terminalization artifacts BEFORE the exactly-once boundary, only for
    // runs that actually started (a spawn failure has no state to archive).
    // The aborted flag (not the global shutdown state) already gated aborts
    // above: shutdown firing in the window after a fully successful run
    // returned must not skip the artifacts.
    if spawned {
        let state = ResearchState::load(&job_id).await;
        let delivered = build_async_research_message(&result);
        write_terminalization_artifacts(&job_id, ws, question, &delivered, &state).await;
    }
    let envelope = crate::jobs::complete_durable_job(
        &job_id,
        build_async_research_message(&result),
        JobKind::ResearchResult,
        caller_role,
        &user_name,
        &channel,
        &ws.name,
    )
    .await;
    Some(envelope)
}

/// Real-terminalization artifacts (results.md + the cleaner ticket), written
/// BEFORE the exactly-once terminalizing boundary at exactly three points:
/// fresh dispatch, boot resume, boot-cap partial report. Shutdown/drain
/// aborts and panics write nothing (the run stays alive for the next boot —
/// both the fresh-dispatch and resume paths now leave the job row 'launched'
/// on abort, so boot-recovery re-enters and terminalizes for real). A
/// cleaner-ticket DB failure is logged — never silent (results.md logs
/// internally).
async fn write_terminalization_artifacts(
    job_id: &str,
    ws: &Workspace,
    question: &str,
    delivered: &str,
    state: &ResearchState,
) {
    crate::research_cleanup::write_results_md(job_id, question, delivered).await;
    if let Err(e) =
        crate::research_cleanup::maybe_create_cleaner_ticket(job_id, ws, question, &state.commands)
            .await
    {
        tracing::warn!(
            job = %job_id,
            error = %e,
            "Cleaner ticket creation failed — outside-zone writes unreported"
        );
    }
}

/// Boot resume of a research run: re-enter the orchestrator at the
/// checkpointed stage (retry_count capped by the boot scan), then terminalize
/// into the durable envelope exactly like a fresh dispatch. Aborts quietly on
/// shutdown/drain — no routing, no terminalization (the checkpointed state is
/// reused by the next boot; routing a partial result here would race the
/// exit).
pub(crate) async fn resume_research_run(job_id: &str, ws: &Workspace) {
    let Some((caller, caller_role)) = crate::jobs::resume_job_preamble(
        &crate::session::store().conn,
        job_id,
        "Research resume",
        "Research resume",
    )
    .await
    else {
        return;
    };
    let result = match run_deep_research(ws, &caller.task, job_id, true).await {
        ResearchExit::Aborted => {
            tracing::info!(
                job = %job_id,
                "Research resume aborted after run — job stays for next boot",
            );
            return;
        }
        ResearchExit::Terminal(result) => result,
    };
    // Terminalization artifacts BEFORE the exactly-once boundary (same points
    // as a fresh dispatch — boot resume is a real terminalization).
    let state = ResearchState::load(job_id).await;
    let delivered = build_async_research_message(&result);
    write_terminalization_artifacts(job_id, ws, &caller.task, &delivered, &state).await;
    complete_durable_job_and_route(
        job_id,
        delivered,
        JobKind::ResearchResult,
        caller_role,
        &caller,
        &ws.name,
    )
    .await;
}

/// Boot-scan over-cap handling: the job exceeded MAX_BOOT_REDISPATCH — deliver
/// a PARTIAL REPORT from the checkpointed state (the research envelope is the
/// Manager's only result path; marking failed with no envelope would strand
/// the caller forever).
pub(crate) async fn research_capped_partial_report(job_id: &str, ws: &Workspace) {
    let Some((caller, caller_role)) = crate::jobs::resume_job_preamble(
        &crate::session::store().conn,
        job_id,
        "Research capped report",
        "Research cap",
    )
    .await
    else {
        return;
    };
    let state = ResearchState::load(job_id).await;
    // Boot path: no recovered findings exist (they are never checkpointed).
    let result: anyhow::Result<String> = Ok(partial_report(
        &caller.task,
        &state.acc,
        "boot re-dispatch cap exceeded — partial report from last checkpoint",
        &[],
    ));
    // Boot-cap is a real terminalization — archive + cleaner ticket (the cap
    // path never enters run_deep_research, so both artifacts are produced here).
    let delivered = build_async_research_message(&result);
    write_terminalization_artifacts(job_id, ws, &caller.task, &delivered, &state).await;
    complete_durable_job_and_route(
        job_id,
        delivered,
        JobKind::ResearchResult,
        caller_role,
        &caller,
        &ws.name,
    )
    .await;
}

/// Build the `<research-result>` envelope message for the async research
/// dispatch. Follows the ask convention: failures are wrapped with an
/// explicit marker, never silently dropped.
fn build_async_research_message(result: &anyhow::Result<String>) -> String {
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
#[derive(Debug, Default, Serialize, Deserialize)]
struct AccumulatedEvidence {
    urls: HashSet<String>,
    /// Claims accumulated across rounds; a claim's stable id is its index in
    /// this vec (claims are only appended or merged in place, never removed).
    claims: Vec<Claim>,
    /// Deduplicated analysts' self-reported unanswered aspects.
    unanswered: Vec<String>,
    /// Rebuilt from `unanswered` after deserialize (normalize_claim keys) —
    /// NOT stored in the state JSON.
    #[serde(skip)]
    unanswered_keys: HashSet<String>,
    /// Raw responses of analysts whose structured extraction failed — never
    /// silently lost.
    raw_reports: Vec<String>,
    /// Research-local weak/unconfirmed annotation links (never consensus).
    weak: WeakLinks,
}

impl AccumulatedEvidence {
    /// Rebuild the derived `unanswered_keys` set after deserialization.
    fn rebuild_keys(&mut self) {
        self.unanswered_keys = self
            .unanswered
            .iter()
            .filter_map(|u| {
                let key = crate::tools::ask::normalize_claim(u);
                (!key.is_empty()).then_some(key)
            })
            .collect();
    }
}

/// Research-local weak/unconfirmed annotation links — the "keep weak, clarify
/// later" side structure. Claim ids are stable indices into
/// [`AccumulatedEvidence::claims`] (claims are only appended or merged in
/// place, never removed), so hints never go stale. Weakness lives ONLY here:
/// it never leaks into claim notes, consensus markers, or verifier prompts.
/// Resolution is the verification gate — verifier verdicts are the feedback
/// loop; this run-local record is intentionally never written back, so the
/// weak count reflects the run's history, not post-verification status.
#[derive(Debug, Default, Serialize, Deserialize)]
struct WeakLinks {
    /// Weak duplicate hints: standalone claim id → the suspected duplicate
    /// target's id. Recorded even when the confirm model rejected the link —
    /// fail-open: a possible relation is never silently dropped.
    duplicates: Vec<(usize, usize)>,
    /// Weak contradiction hints: new claim id → existing claim id. Both sides
    /// still carry their contradiction notes (verification qualifies them);
    /// the unconfirmed relation is recorded here.
    contradictions: Vec<(usize, usize)>,
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
    /// For "contradicts": the contradiction note.
    contradiction: Option<String>,
}

/// The full annotation pass over one round's pending claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnnotationPass {
    annotations: Vec<ClaimAnnotation>,
}

/// Per-pair re-judgment of one mutating annotation link (duplicate /
/// contradicts) from the optional confirm pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmLink {
    /// 0-based index of the NEW claim within this round's pending claims.
    /// Pair identity is pinned by the annotation pass — the model never
    /// re-transcribes the existing id (the transcription-error class this
    /// pass exists to catch).
    new_id: usize,
    /// "confirm" | "reject" — uncertainty maps to reject (weak/unconfirmed).
    verdict: String,
}

/// The confirm pass over one round's mutating annotation links.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmPass {
    links: Vec<ConfirmLink>,
}

/// Outcome of the optional confirm pass: per-pair verdicts, or the whole
/// call failed (every mutating verdict is weak). Typed so the two states can
/// never be conflated.
enum ConfirmOutcome {
    Passed(ConfirmPass),
    Failed,
}

impl AccumulatedEvidence {
    /// Absorb a round's evidence, returning `(novel_urls, pending_claims)`.
    /// URLs and unanswered aspects dedup exactly as before; claims are
    /// collected into a pending list — novelty is decided by the per-round
    /// LLM annotation pass, not embedding similarity.
    fn absorb(&mut self, round: &EvidenceRound) -> (usize, Vec<Claim>) {
        let novel_urls = round
            .urls
            .iter()
            .filter(|u| self.urls.insert((*u).clone()))
            .count();
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
        (novel_urls, round.claims.clone())
    }

    /// Apply an annotation pass over a round's pending claims (the validator
    /// guarantees id completeness and in-range existing ids): novel claims
    /// are appended, confirmed duplicates merge into the existing claim
    /// (sources joined deduplicated, confidence upgraded, contradictions
    /// appended — never dropped), and contradicting claims are appended AND
    /// linked to the existing claim so the verification gate targets both
    /// sides. Weak (unconfirmed) mutating verdicts never merge: a weak
    /// duplicate stays standalone with a side hint and does NOT count as
    /// novel; a weak contradiction keeps the bidirectional notes but records
    /// the unconfirmed relation in the side structure. Returns the number of
    /// novel claims (the saturation signal).
    fn apply_annotations(
        &mut self,
        pass: &AnnotationPass,
        pending: &[Claim],
        confirm: &ConfirmOutcome,
    ) -> usize {
        let confirmed: HashSet<usize> = match confirm {
            ConfirmOutcome::Passed(p) => p
                .links
                .iter()
                .filter(|l| l.verdict == "confirm")
                .map(|l| l.new_id)
                .collect(),
            ConfirmOutcome::Failed => HashSet::new(),
        };
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
                    if confirmed.contains(&a.new_id) {
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
                    } else {
                        // Weak duplicate: never merged, never novel — the
                        // pending claim stays standalone with a hint in the
                        // side structure ("keep weak, clarify later").
                        let id = self.claims.len();
                        self.claims.push(pending_claim.clone());
                        self.weak.duplicates.push((id, existing_id));
                    }
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
                    let id = self.claims.len();
                    self.claims.push(new_claim);
                    novel += 1;
                    if !confirmed.contains(&a.new_id) {
                        // Weak contradiction: same bidirectional notes (both
                        // sides qualify for verification), the unconfirmed
                        // relation marked in the side structure — never in
                        // the note text.
                        self.weak.contradictions.push((id, existing_id));
                    }
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
#[derive(Debug, Default, Serialize, Deserialize)]
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
#[derive(Debug, Default, Serialize, Deserialize)]
struct GapRoundsOutcome {
    abstention: Option<String>,
    unresolved: Vec<String>,
    rounds_dispatched: usize,
    /// Set when the orchestrator could not determine the remaining gaps —
    /// the report must carry an explicit marker instead of looking like
    /// coverage completion.
    incomplete: Option<String>,
}

// ── Wrap-up: recover deadline-aborted analysts' findings ──────────────────

/// One deadline-aborted analyst's dispatch-time snapshot: the agent_id plus
/// the frozen chat params (model, tools, reasoning_effort, routing,
/// max_tokens) captured BEFORE spawn so the wrap-up call replays the same
/// KV-cache prefix as the analyst's own calls. Known limitation: config or
/// daemon-state drift (e.g. browser tool advertisement flipping between the
/// snapshot capture and the spawned agent's own derivation) is not reflected
/// in the snapshot (fail-open — a miss only costs the tail re-encode).
#[derive(Clone)]
struct WrapUpEntry {
    agent_id: String,
    params: ChatRequest,
}

/// A timed-out analyst's loaded session, ready for the wrap-up LLM call.
struct WrapUpPrepared {
    params: ChatRequest,
    history: Vec<ChatMessage>,
}

/// Wrap-up stage bound for [`wrap_up_timed_out`]. Overridable via env.
fn wrap_up_timeout() -> Duration {
    crate::util::env_duration_secs("MAHBOT_WRAP_UP_TIMEOUT_SECS", DEFAULT_WRAP_UP_TIMEOUT_SECS)
}

/// Build Analyst chat params carrying `purpose`/`agent_id` metadata and
/// optional tool specs; byte-relevant fields come from the shared
/// [`chat_request`] helper (same source as
/// [`crate::agent::Agent::build_chat_request`] — model, reasoning_effort,
/// routing, max_tokens).
fn research_params(
    ws: &Workspace,
    purpose: &'static str,
    agent_id: String,
    tool_specs: Option<Vec<ToolSpec>>,
) -> ChatRequest {
    ChatRequest {
        meta: Some(ChatRequestMeta {
            purpose,
            agent_id,
            role: Role::Analyst.as_str().to_string(),
            workspace: ws.name.clone(),
            ticket_id: None,
        }),
        ..chat_request(Role::Analyst, tool_specs, Vec::new(), false)
    }
}

/// Wrap-up params: the advertised tool specs frozen at dispatch time. The
/// dedicated purpose tag separates wrap-up calls in the request journal
/// (post-rollout cached_input_tokens check).
fn wrap_up_params(ws: &Workspace, agent_id: &str, tool_specs: Vec<ToolSpec>) -> ChatRequest {
    research_params(
        ws,
        "research_wrap_up",
        agent_id.to_string(),
        Some(tool_specs),
    )
}

/// Wrap-up stage after a round deadline: for every analyst aborted by the
/// deadline, load its persisted session, register its search queries in the
/// ledger (BEFORE any LLM call — independent of the extraction outcome),
/// and — only when the session shows at least one successful tool result —
/// extract the accumulated findings via a fresh LLM call replaying the
/// dispatch-time params. Fail-open: a failed extraction keeps the analyst
/// failed. Skipped entirely when shutdown/drain is already in progress;
/// aborts promptly if the drain starts mid-stage (the orchestrator guard
/// blocks the drain-watch token, so the wrap-up self-aborts on the drain
/// flag — the partial report is not delayed past the drain).
async fn wrap_up_timed_out(
    timed_out: Vec<WrapUpEntry>,
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
) -> Vec<AnalystFindings> {
    if timed_out.is_empty() || crate::shutdown::aborting() {
        return Vec::new();
    }
    let stage_deadline = std::time::Instant::now() + wrap_up_timeout();
    // Prepare sequentially (cheap DB reads): load the session, register its
    // queries (mirroring the parse-failed-analyst pattern — ledger + summary
    // telemetry only, never the round's saturation counters; the shared
    // ledger makes a later gap-round re-ask count as a repeat), and gate the
    // LLM call on the presence of at least one successful tool result. Prep
    // always completes so query registration is guaranteed for every timed-
    // out analyst independent of the stage deadline — the deadline bounds
    // only the LLM batch below; a drain mid-prep still aborts the rest.
    let mut prepared = Vec::new();
    for entry in timed_out {
        if crate::shutdown::aborting() {
            break;
        }
        let history = crate::session::store().load(&entry.agent_id).await;
        let (tool_calls, searches, queries) = extract_query_telemetry_from_history(&history);
        run_stats.tool_calls += tool_calls;
        run_stats.searches += searches;
        for q in &queries {
            if !ledger.register(q) {
                run_stats.repeat_queries += 1;
            }
        }
        let has_success = session_has_successful_tool_result(&history);
        if has_success {
            prepared.push(WrapUpPrepared {
                params: entry.params,
                history,
            });
        }
    }
    // Skip the batch when nothing is prepared or a drain started during the
    // last prep iteration. An expired stage deadline needs no guard — the
    // batch's deadline arm cancels the tasks immediately (fail-open).
    if prepared.is_empty() || crate::shutdown::aborting() {
        return Vec::new();
    }
    let wrap_up_prompt = load_prompt("research/wrap_up.md");
    let handles: Vec<_> = prepared
        .into_iter()
        .map(|p| {
            let wrap_up_prompt = wrap_up_prompt.clone();
            tokio::spawn(async move {
                let mut messages = p.history;
                messages.push(ChatMessage::user(&wrap_up_prompt));
                // Ticket-mandated ~3 attempts / ~90s. The policy's
                // operation_timeout is a whole-operation wall-clock deadline
                // (retry.rs), NOT per-attempt — on the worst-case cache-miss
                // tail (~170K tokens of prefill) all attempts share the 90s;
                // the stage deadline caps the batch but cannot extend a
                // task's 90s. Post-rollout telemetry distinguishes cache-hit
                // from policy starvation (the policy's own backoff/attempts
                // apply; the wrap-up adds nothing beyond them).
                crate::extraction::retry_extract_structured_scoped::<AnalystFindings>(
                    &messages,
                    "",
                    &p.params,
                    None,
                    Some(&crate::retry::RetryPolicy::comment()),
                )
                .await
                .ok()
            })
        })
        .collect();
    await_wrap_up_batch(handles, stage_deadline)
        .await
        .into_iter()
        .flatten()
        // Empty extractions (no claims AND no unanswered) are dropped so the
        // recovered section never renders an orphan header with no bullets.
        .filter(|f| !f.claims.is_empty() || !f.unanswered.is_empty())
        .collect()
}

/// True when the session history shows at least one successful tool result —
/// a persisted tool result whose content does not start with the
/// tool-failure marker (all-failure or no-result sessions skip the wrap-up
/// LLM call; their queries are still registered). A successful result
/// coincidentally beginning with the marker is misclassified as a failure —
/// accepted, negligible probability. Non-native (non-JSON-wrapped) results
/// decode to None and count as no-success — conservative fail-open, but
/// unreachable for research analysts (their results persist native).
fn session_has_successful_tool_result(history: &[ChatMessage]) -> bool {
    history.iter().any(|m| {
        matches!(
            crate::session::decode_native_history_message(m),
            Some(crate::session::DecodedNativeHistoryMessage::ToolResult { content, .. })
                if !content.starts_with(crate::tools::TOOL_FAILURE_MARKER)
        )
    })
}

/// Await the concurrent wrap-up extraction tasks, bounded by the stage
/// deadline and interrupted by the drain flag (each task's inner provider
/// call is dropped via the batch cancel token). Returns one result per task
/// slot. Force-cancel (drain cap / second signal) needs no select arm here —
/// the inner retry loop is shutdown-abortable and resolves on its own.
/// Not shared with [`await_round_members`] deliberately: round waits must NOT
/// abort on drain (in-flight analysts complete their turn), while the wrap-up
/// MUST (the ticket's partial-report-not-delayed requirement).
async fn await_wrap_up_batch(
    handles: Vec<tokio::task::JoinHandle<Option<AnalystFindings>>>,
    deadline: std::time::Instant,
) -> Vec<Option<AnalystFindings>> {
    use futures_util::StreamExt;
    use futures_util::stream::FuturesUnordered;

    let cancel = tokio_util::sync::CancellationToken::new();
    let drain = crate::shutdown::drain_wait();
    tokio::pin!(drain);
    let mut pending: FuturesUnordered<_> = handles
        .into_iter()
        .enumerate()
        .map(|(i, mut handle)| {
            let cancel = cancel.clone();
            async move {
                tokio::select! {
                    biased;
                    r = &mut handle => (i, match r {
                        Ok(v) => v,
                        Err(e) if e.is_panic() => {
                            let panic = crate::util::panic_message(&*e.into_panic());
                            tracing::warn!(member = i, %panic, "wrap-up task panicked");
                            None
                        }
                        Err(_) => {
                            tracing::warn!(member = i, "wrap-up task cancelled externally");
                            None
                        }
                    }),
                    () = cancel.cancelled() => {
                        handle.abort();
                        (i, None)
                    }
                }
            }
        })
        .collect();
    let mut out: Vec<Option<AnalystFindings>> = (0..pending.len()).map(|_| None).collect();
    // The drain/deadline arms fire once each — after that the cancel token
    // aborts every remaining task so the batch drains promptly (a
    // permanently-ready select arm would otherwise starve `pending.next()`).
    let mut drain_fired = false;
    let mut deadline_fired = false;
    while !pending.is_empty() {
        tokio::select! {
            biased;
            () = drain.as_mut(), if !drain_fired => {
                drain_fired = true;
                cancel.cancel();
            }
            Some((i, result)) = pending.next() => out[i] = result,
            () = tokio::time::sleep_until(deadline.into()), if !deadline_fired => {
                deadline_fired = true;
                cancel.cancel();
            }
        }
    }
    out
}

/// Collapse awaited round members into analyst runs, splitting out the
/// deadline-aborted members with their dispatch-time snapshots (the wrap-up
/// stage replays them). Panicked/cancelled members stay plain NoResponse —
/// the TimedOut distinction survives until the wrap-up stage.
fn resolve_round_members_with_timeouts<T>(
    members: Vec<RoundMember<AnalystRun<T>>>,
    snapshots: &[WrapUpEntry],
) -> (Vec<AnalystRun<T>>, Vec<WrapUpEntry>) {
    // Dispatch pushes one snapshot per member in the same order; the empty
    // slice (decomposition path) yields no wrap-up entries.
    debug_assert!(snapshots.is_empty() || snapshots.len() == members.len());
    let mut runs = Vec::with_capacity(members.len());
    let mut timed_out = Vec::new();
    for (i, m) in members.into_iter().enumerate() {
        match m {
            RoundMember::Done(run) => runs.push(run),
            RoundMember::TimedOut => {
                runs.push(AnalystRun::NoResponse);
                if let Some(entry) = snapshots.get(i) {
                    timed_out.push(entry.clone());
                } else if !snapshots.is_empty() {
                    // Only the research path expects a snapshot per member —
                    // the decomposition path (empty slice) has none by design.
                    tracing::warn!(
                        member = i,
                        "timed-out member has no wrap-up snapshot — findings unrecoverable"
                    );
                }
            }
            RoundMember::Panicked | RoundMember::Cancelled => runs.push(AnalystRun::NoResponse),
        }
    }
    (runs, timed_out)
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

/// Collapse awaited round members into analyst runs — the decomposition-path
/// convenience (no wrap-up snapshots): timed-out, panicked, and cancelled
/// members all map to [`AnalystRun::NoResponse`]. Research rounds use
/// [`resolve_round_members_with_timeouts`] to keep the TimedOut set alive.
fn resolve_round_members<T>(members: Vec<RoundMember<AnalystRun<T>>>) -> Vec<AnalystRun<T>> {
    resolve_round_members_with_timeouts(members, &[]).0
}

/// Run a single analyst agent on `task` and extract structured output `T`
/// while the agent is alive (KV-cache reuse). Returns the extraction plus
/// telemetry `(tool_calls, searches, queries)` from the session history.
async fn run_structured_analyst<T: serde::de::DeserializeOwned>(
    ws: &Workspace,
    agent_id: &str,
    task: &str,
    extraction_prompt: &str,
    round: crate::agent::RoundOpts,
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
        false,
        Some(round),
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
        .extract_verdict::<T>(extraction_prompt, None, None)
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

/// Build a round member closure: run one analyst on `task` and extract
/// structured output `T`. Takes owned values so the returned closure (and its
/// boxed future) are `'static` + `Send`, as `spawn_staggered_round` requires.
fn make_round_member<T: serde::de::DeserializeOwned + Send>(
    ws: Workspace,
    agent_id: String,
    task: String,
    extraction_prompt: String,
) -> impl FnOnce(crate::agent::RoundOpts) -> futures_util::future::BoxFuture<'static, AnalystRun<T>> + Send
{
    move |round| {
        Box::pin(async move {
            run_structured_analyst::<T>(&ws, &agent_id, &task, &extraction_prompt, round).await
        })
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
/// system message is constant too; only the user message varies). Byte-relevant
/// fields come from the shared [`chat_request`] helper.
fn orchestrator_params(ws: &Workspace, purpose: &'static str) -> ChatRequest {
    research_params(
        ws,
        purpose,
        format!("research_{}_orchestrator", ws.name),
        None,
    )
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
    crate::extraction::retry_extract_structured_scoped::<T>(&messages, "", &params, validate, None)
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
#[allow(clippy::too_many_arguments)]
async fn round0_decompose(
    ws: &Workspace,
    question: &str,
    budget: &mut ResearchBudget,
    run_stats: &mut RunStats,
    deadline: std::time::Instant,
    resume: bool,
    run_root: &str,
    captured: &mut Vec<String>,
) -> Result<(MergedPlan, Option<String>)> {
    budget
        .try_reserve(DECOMPOSE_FAN_OUT)
        .map_err(anyhow::Error::msg)?;
    let task_template = load_prompt("research/decompose.md");
    let extraction_prompt = load_prompt("extraction/decompose.md");
    let members: Vec<_> = (0..DECOMPOSE_FAN_OUT)
        .map(|i| {
            let ws = ws.clone();
            let question = question.to_string();
            let task = substitute(
                &task_template,
                &[("{{question}}", &question), ("{{run_root}}", run_root)],
            );
            let agent_id = crate::session::research_agent_id(&ws.name, &format!("decompose_{i}"));
            captured.push(agent_id.clone());
            make_round_member::<DecompositionPlan>(ws, agent_id, task, extraction_prompt.clone())
        })
        .collect();
    let handles = crate::agent::spawn_staggered_round(members, resume).await;
    let plans: Vec<AnalystRun<DecompositionPlan>> =
        resolve_round_members(await_round_members(handles, deadline).await);
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
    if let Ok(mut plan) = merge_decomposition_plans(ws, question, &valid).await {
        resolve_merged_plan_ids(&mut plan, &valid);
        Ok((plan, None))
    } else {
        // Fail-open: a failed merge must never lose the run — fall back to
        // the first valid plan verbatim with an explicit marker. Plan 0's
        // items are the leading ids of the global flat numbering.
        let first = &valid[0];
        let plan = MergedPlan {
            sub_questions: first
                .sub_questions
                .iter()
                .enumerate()
                .map(|(i, sq)| MergedSubQuestion {
                    question: sq.question.clone(),
                    evidence_needed: sq.evidence_needed.clone(),
                    risk: sq.risk.clone(),
                    from_id: i,
                    also_ids: Vec::new(),
                })
                .collect(),
            dropped: Vec::new(),
        };
        Ok((plan, Some(PLAN_MERGE_FAILED.to_string())))
    }
}

/// Render the input plans with their global flat item ids for the merge
/// prompt: `Plan N:` header per plan, one `- {id}: question [evidence, risk]`
/// line per item, ids numbered flat across all plans in (plan, item) order.
fn render_plans_with_ids(plans: &[DecompositionPlan]) -> String {
    let mut out = String::new();
    let mut id = 0usize;
    for (p, plan) in plans.iter().enumerate() {
        let _ = writeln!(out, "Plan {p}:");
        for sq in &plan.sub_questions {
            let _ = writeln!(
                out,
                "- {id}: {} [evidence: {}, risk: {}]",
                sq.question, sq.evidence_needed, sq.risk
            );
            id += 1;
        }
    }
    out
}

/// Merge 1–3 independent plans into one consolidated plan with provenance
/// (orchestrator extraction call — not budgeted).
async fn merge_decomposition_plans(
    ws: &Workspace,
    question: &str,
    plans: &[DecompositionPlan],
) -> Result<MergedPlan> {
    let prompt = substitute(
        &load_prompt("research/decompose_merge.md"),
        &[
            ("{{question}}", question),
            ("{{plans}}", &render_plans_with_ids(plans)),
        ],
    );
    // The validator captures owned copies (the validator type is `'static`).
    let plans_owned = plans.to_vec();
    orchestrator_extract::<MergedPlan>(
        ws,
        "decompose_merge",
        &prompt,
        Some(&move |p| validate_merged_plan(p, &plans_owned)),
    )
    .await
}

/// The input-plan items as the global flat id universe — id = (plan, item)
/// order, matching `render_plans_with_ids` and the merge prompt's numbering.
fn plan_item_table(plans: &[DecompositionPlan]) -> Vec<Vec<String>> {
    plans
        .iter()
        .map(|p| p.sub_questions.iter().map(|s| s.question.clone()).collect())
        .collect()
}

/// Fill in merged entries' tuple text from the cited input-plan items (the
/// system resolves id → tuple via the global flat numbering — the model's
/// copied text is never used).
fn resolve_merged_plan_ids(plan: &mut MergedPlan, plans: &[DecompositionPlan]) {
    let items = plan_item_table(plans);
    let table = crate::consensus::ItemTable::new(&items);
    for sq in &mut plan.sub_questions {
        if let Some((p, i)) = table.resolve_index(sq.from_id) {
            let src = &plans[p].sub_questions[i];
            sq.question.clone_from(&src.question);
            sq.evidence_needed.clone_from(&src.evidence_needed);
            sq.risk.clone_from(&src.risk);
        }
    }
}

/// Validate a merged plan by id-based coverage: every cited id in range, no
/// duplicate placement, and full coverage — every input plan item id appears
/// exactly once across merged sub-questions (as `from_id` or `also_ids`) plus
/// dropped. Silent dropout is rejected (fail-closed inside the extraction
/// retry loop). Structural only — tuple text is resolved by the system, never
/// machine-checked against the plans.
fn validate_merged_plan(plan: &MergedPlan, plans: &[DecompositionPlan]) -> Result<(), String> {
    let items = plan_item_table(plans);
    let table = crate::consensus::ItemTable::new(&items);
    let mut covered = HashSet::new();
    let mut mark = |id: usize, where_: &str| -> Result<(), String> {
        if id >= table.len() {
            return Err(format!("{where_}: out-of-range item id {id}"));
        }
        if !covered.insert(id) {
            return Err(format!("{where_}: item {id} covered more than once"));
        }
        Ok(())
    };
    for (i, sq) in plan.sub_questions.iter().enumerate() {
        mark(sq.from_id, &format!("merged sub-question {i}"))?;
        for &id in &sq.also_ids {
            mark(id, &format!("merged sub-question {i} also_ids"))?;
        }
    }
    for (i, d) in plan.dropped.iter().enumerate() {
        mark(d.id, &format!("dropped entry {i}"))?;
    }
    for id in 0..table.len() {
        if !covered.contains(&id) {
            return Err(format!(
                "silent dropout: input plan item {id} is never covered by the merged plan or dropped list"
            ));
        }
    }
    Ok(())
}

// ── Round 1: one analyst per sub-question ────────────────────────────────

/// Round 1: one analyst per sub-question; two for high-risk items (the
/// second gets a decorrelated research angle). Returns the round's evidence
/// plus the dispatch-time snapshots of deadline-aborted analysts (the caller
/// checkpoints FIRST, then runs the wrap-up stage — a crash mid-wrap-up must
/// not lose the round-1 evidence), or `None` when the analyst budget is
/// exhausted before dispatch.
#[allow(clippy::too_many_arguments)]
async fn round1_research(
    ws: &Workspace,
    question: &str,
    plan: &MergedPlan,
    budget: &mut ResearchBudget,
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
    deadline: std::time::Instant,
    resume: bool,
    run_root: &str,
    captured: &mut Vec<String>,
) -> Option<(EvidenceRound, Vec<WrapUpEntry>)> {
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
    let mut members = Vec::new();
    let mut snapshots: Vec<WrapUpEntry> = Vec::new();
    // Dispatch-time wrap-up snapshots: frozen params + advertised tool
    // schemas, captured before spawn (aborted tasks lose their values). The
    // specs are constant across the round's members (same role+workspace).
    let wrap_up_specs = role_tools_and_specs(Role::Analyst, ws).1;
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
                    ("{{run_root}}", run_root),
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
            captured.push(agent_id.clone());
            snapshots.push(WrapUpEntry {
                agent_id: agent_id.clone(),
                params: wrap_up_params(&ws, &agent_id, wrap_up_specs.clone()),
            });
            idx += 1;
            members.push(make_round_member::<AnalystFindings>(
                ws,
                agent_id,
                task,
                extraction_prompt.clone(),
            ));
        }
    }
    let handles = crate::agent::spawn_staggered_round(members, resume).await;
    let members_out = await_round_members(handles, deadline).await;
    let (runs, timed_out) = resolve_round_members_with_timeouts(members_out, &snapshots);
    // collect_evidence runs first so the wrap-up's ledger registrations can
    // never skew this round's saturation counters (round.queries /
    // repeat_queries stay from successful analysts only).
    let round = collect_evidence(&runs, ledger, run_stats);
    Some((round, timed_out))
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

/// Run one gap round: fresh analysts, one per targeted gap (width-shrinking
/// 4→3→2). Returns the round's analyst runs plus the dispatch-time snapshots
/// of analysts aborted by the round deadline (the caller runs the wrap-up
/// stage AFTER collecting the round's evidence so ledger registrations from
/// recovered analysts cannot skew the round's saturation counters).
#[allow(clippy::too_many_arguments)]
async fn run_gap_round(
    ws: &Workspace,
    question: &str,
    gaps: &[&Gap],
    ledger: &QueryLedger,
    deadline: std::time::Instant,
    resume: bool,
    run_root: &str,
    captured: &mut Vec<String>,
) -> (Vec<AnalystRun<AnalystFindings>>, Vec<WrapUpEntry>) {
    let task_template = load_prompt("research/gap.md");
    let extraction_prompt = load_prompt("extraction/findings.md");
    let ledger_snapshot = ledger.render();
    let mut members: Vec<_> = Vec::new();
    let mut snapshots: Vec<WrapUpEntry> = Vec::new();
    // Dispatch-time wrap-up snapshots (see round1_research) — same specs for
    // every member of the round.
    let wrap_up_specs = role_tools_and_specs(Role::Analyst, ws).1;
    for (i, gap) in gaps.iter().enumerate() {
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
                ("{{run_root}}", run_root),
            ],
        );
        let agent_id = crate::session::research_agent_id(&ws.name, &format!("gap_{i}"));
        captured.push(agent_id.clone());
        snapshots.push(WrapUpEntry {
            agent_id: agent_id.clone(),
            params: wrap_up_params(&ws, &agent_id, wrap_up_specs.clone()),
        });
        members.push(make_round_member::<AnalystFindings>(
            ws,
            agent_id,
            task,
            extraction_prompt.clone(),
        ));
    }
    let handles = crate::agent::spawn_staggered_round(members, resume).await;
    let members_out = await_round_members(handles, deadline).await;
    resolve_round_members_with_timeouts(members_out, &snapshots)
}

/// Record a coder-round marker — a GATE-SKIP (`"skipped — {reason}"`; the
/// round is NOT claimed) or an OUTCOME (dispatched but failed/cancelled).
/// One marker per key: a fresh marker clears the prior one (a stale
/// skip/outcome marker is superseded — the report shows one truth per key).
/// Re-attempt semantics: only the PRE-LOOP key-0 round is re-examined on
/// boot-resume (`!coder_rounds_done.contains(&0)`); a post-progress round
/// (key ≥ 1) is dispatched only as a side-effect of a progress event inside
/// the gap loop, so a gate-skip after a long gap round is FINAL — the loop's
/// already-advanced round_index never revisits the key. That is fail-open per
/// design ("Тихих пропусков нет" — the marker IS the report note; the run
/// continues without the prototype).
fn set_coder_marker(state: &mut ResearchState, round_key: usize, suffix: &str) {
    let marker = format!("coder round {round_key} {suffix}");
    clear_coder_markers(state, round_key);
    state.markers.push(marker);
}

/// Clear stale coder-round markers for a key — one truth per key.
fn clear_coder_markers(state: &mut ResearchState, round_key: usize) {
    state
        .markers
        .retain(|m| !m.starts_with(&format!("coder round {round_key} ")));
}

/// Claim a coder round at DISPATCH. The claim is IN-MEMORY here — it reaches
/// the persisted checkpoint only at the next per-round save (after the round
/// completes), so a crash mid-coder loses it and boot-resume re-dispatches
/// (the accepted crash-duplicate pin). Stale skip/outcome markers for this
/// key are cleared — the report shows one truth (the final outcome), not
/// a dead skip plus completed prototypes.
fn claim_coder_round(state: &mut ResearchState, round_key: usize) {
    if !state.coder_rounds_done.contains(&round_key) {
        state.coder_rounds_done.push(round_key);
    }
    clear_coder_markers(state, round_key);
}

/// Un-claim a round that was dispatched but never completed (failure or
/// cancellation). Only the pre-loop key-0 round is re-attempted by
/// boot-resume; a post-progress failure is final (marked, fail-open).
fn unclaim_coder_round(state: &mut ResearchState, round_key: usize) {
    state.coder_rounds_done.retain(|k| *k != round_key);
}

/// One coder round: a single Coder sub-agent builds prototypes in the per-run
/// folder targeting the current gap list. Blocks on the coder (never hard
/// timed out — a long coder may overrun the round deadline; the loop-top
/// checks then skip the following gap rounds, fail-open). The coder's response
/// text is NOT inserted into the report (prototypes list only); failures and
/// skips are marked in the report, never silent.
#[allow(clippy::too_many_arguments)]
async fn run_coder_round(
    job_id: &str,
    run_root: &str,
    ws: &Workspace,
    question: &str,
    gap_list: &GapList,
    deadline: std::time::Instant,
    state: &mut ResearchState,
    round_key: usize,
) {
    if crate::shutdown::aborting() {
        set_coder_marker(state, round_key, "skipped — shutdown/drain");
        return;
    }
    if std::time::Instant::now() + CODER_MIN_REMAINING >= deadline {
        set_coder_marker(
            state,
            round_key,
            "skipped — less than 30 minutes remaining until the round deadline",
        );
        return;
    }
    // Claimed at dispatch (in-memory — persisted only at the next per-round
    // checkpoint, which happens AFTER the round completes, so a crash
    // mid-coder loses the claim. Only the PRE-LOOP key-0 round is
    // re-dispatched by boot-resume (the accepted crash-duplicate pin);
    // a crashed post-progress round is final, fail-open.
    claim_coder_round(state, round_key);
    let evidence = render_accumulated_evidence(&state.acc);
    let gaps = gap_items(&gap_list.gaps).join("\n");
    let task = substitute(
        &load_prompt("synthesis/coder_brief.md"),
        &[
            ("{{question}}", question),
            ("{{evidence}}", &evidence),
            ("{{gaps}}", &gaps),
            ("{{run_root}}", run_root),
        ],
    );
    let coder_ws = Workspace::ephemeral_run(job_id, Path::new(run_root));
    let agent_id = crate::session::research_agent_id(&ws.name, "coder");
    // Command collection happens after the run completes — a crash mid-coder
    // loses the session from the sanitizer (accepted: only the PRE-LOOP key-0
    // round is re-dispatched by boot-resume — with a fresh agent id, since
    // `research_agent_id` embeds a fresh NanoID suffix per call — and the
    // design's crash-duplicate pin covers the re-run's prototype duplication;
    // a crashed post-progress round is final, fail-open).
    let (agent, response) = run_agent(
        agent_id.clone(),
        Role::Coder,
        &coder_ws,
        None,
        &task,
        String::new(),
        String::new(),
        None,
        false,
        None,
    )
    .await;
    state.capture_round(&[agent_id], Path::new(run_root)).await;
    if response.is_some() {
        tracing::info!(job = %job_id, coder_round = round_key, "Coder round completed");
    } else {
        let cancelled = agent.is_cancelled() || crate::shutdown::aborting();
        let outcome = if cancelled { "cancelled" } else { "failed" };
        // Never completed → not done (only the pre-loop key-0 round is
        // re-attempted by boot-resume; post-progress failures are final —
        // fail-open per design).
        unclaim_coder_round(state, round_key);
        set_coder_marker(state, round_key, outcome);
    }
}

/// Gate a coder round with the gap-loop top checks (budget/deadline — the
/// same checks that would skip the following gap round). A gate-skip is
/// marked in the report, never silent. Only the PRE-LOOP key-0 round is
/// re-attempted on boot-resume (`!coder_rounds_done.contains(&0)` in
/// `gap_rounds`); post-progress rounds (key ≥ 1) are dispatched only as a
/// side-effect of a progress event inside the loop, so a skip after a long
/// gap round is final (fail-open — the run continues without the prototype).
/// No speculative inline retry: `run_agent` already exhausts its internal
/// retry bounds before returning Failed, so a second full coder session would
/// only double the LLM spend of a confirmed failure (design: "Сбой кодера =
/// fail-open").
#[allow(clippy::too_many_arguments)]
async fn run_coder_gated(
    job_id: &str,
    run_root: &str,
    ws: &Workspace,
    question: &str,
    budget: &ResearchBudget,
    gap_list: &GapList,
    deadline: std::time::Instant,
    state: &mut ResearchState,
    round_key: usize,
) {
    if budget.is_exhausted() {
        set_coder_marker(state, round_key, "skipped — analyst budget exhausted");
        return;
    }
    if std::time::Instant::now() >= deadline {
        set_coder_marker(state, round_key, "skipped — round deadline expired");
        return;
    }
    run_coder_round(
        job_id, run_root, ws, question, gap_list, deadline, state, round_key,
    )
    .await;
}

/// Conditional gap rounds. Stopping is artifact-based, never agent
/// self-assessment: coverage completion, answerability abstention (checked on
/// every structural quiet round — a non-abstain verdict continues, with the
/// analyst budget as the hard bound), budget exhaustion, and shutdown.
///
/// Checkpoints AFTER EACH gap round: the gap-loop locals (round_index,
/// rounds_dispatched, the current gap list, budget) are not derivable from
/// the accumulated evidence — a crash mid-loop would otherwise revert to the
/// post-round-1 checkpoint and re-run the ENTIRE gap stage from round 0 with
/// fresh analyst sessions (whole-stage re-run, duplicated LLM spend).
///
/// Coder-in-loop: one prototype pass (key 0) before the loop over the initial
/// gap list, then one after every progress round that refreshes a non-empty
/// gap list (key = the round's index). Each pass is gated by the same
/// budget/deadline/shutdown checks as the loop top and persisted with the
/// per-round checkpoint. Boot-resume re-attempts ONLY the pre-loop key-0
/// round (the `!coder_rounds_done.contains(&0)` check below); a skipped or
/// failed post-progress round (key ≥ 1) is FINAL — the loop resumes at the
/// advanced round_index and never revisits the key (fail-open per design:
/// "Сбой кодера = fail-open", skip marker in the report).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn gap_rounds(
    ws: &Workspace,
    question: &str,
    plan: &MergedPlan,
    budget: &mut ResearchBudget,
    state: &mut ResearchState,
    run_stats: &mut RunStats,
    deadline: std::time::Instant,
    job_id: &str,
    run_root: &str,
    resume: bool,
    recovered: &mut Vec<AnalystFindings>,
) -> GapRoundsOutcome {
    let mut outcome = GapRoundsOutcome {
        abstention: None,
        unresolved: Vec::new(),
        rounds_dispatched: 0,
        incomplete: None,
    };
    let initial_list = match state.gap_list.take() {
        Some(list) => Some(list),
        None => extract_gap_list(ws, question, &state.acc, plan).await,
    };
    let Some(mut gap_list) = initial_list else {
        // Explicit marker — never collapse into coverage completion.
        outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
        return outcome;
    };
    // Pre-loop coder round (key 0): one prototype pass over the initial gap
    // list before the first gap round dispatches.
    if !gap_list.gaps.is_empty() && !state.coder_rounds_done.contains(&0) {
        run_coder_gated(
            job_id, run_root, ws, question, budget, &gap_list, deadline, state, 0,
        )
        .await;
    }
    let mut round_index = state.round_index;
    loop {
        if crate::shutdown::aborting()
            || budget.is_exhausted()
            // Round-wide deadline expired: further rounds would be spawned
            // and instantly aborted — stop instead of burning budget on
            // no-progress work.
            || std::time::Instant::now() >= deadline
        {
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
        let width = GAP_ROUND_WIDTHS[round_index.min(GAP_ROUND_WIDTHS.len() - 1)];
        round_index += 1;
        let targeted: Vec<&Gap> = gaps.iter().take(width).collect();
        if budget.try_reserve(targeted.len()).is_err() {
            outcome.unresolved = gap_items(gaps);
            return outcome;
        }
        let mut round_agents: Vec<String> = Vec::new();
        let (runs, timed_out) = run_gap_round(
            ws,
            question,
            &targeted,
            &state.ledger,
            deadline,
            resume,
            run_root,
            &mut round_agents,
        )
        .await;
        state
            .capture_round(&round_agents, Path::new(run_root))
            .await;
        let round = collect_evidence(&runs, &mut state.ledger, run_stats);
        // Wrap-up stays BEFORE the per-round checkpoint here (unlike round 1):
        // the loop's early returns (abstention / gap-extract failure) follow
        // this position, and the final round's wrap-up must survive them; the
        // pre-checkpoint window already contains the annotate + gap-extract
        // orchestrator calls, so the wrap-up only extends an existing window.
        recovered.extend(wrap_up_timed_out(timed_out, &mut state.ledger, run_stats).await);
        let (new_urls, pending) = state.acc.absorb(&round);
        let novel_claims = annotate_round(ws, &mut state.acc, &pending, &mut state.markers).await;
        outcome.rounds_dispatched += 1;
        // A round whose analysts only re-asked already-asked queries counts
        // as no-progress (repeat queries are never pre-dispatch-droppable —
        // concurrent analysts generate them live).
        let all_repeat_queries = round.queries > 0 && round.queries == round.repeat_queries;
        if (new_urls == 0 && novel_claims == 0) || all_repeat_queries {
            // Structural quiet-round signal: no new claims and no new sources.
            // A weak-duplicate-only round also lands here (weak duplicates
            // append standalone claims but never count as novel) — the
            // answerability check fires and the gap list is not refreshed,
            // which is the intended premature-saturation protection: gap-round
            // criteria, not reclassification, decide saturation.
            if let Some(reason) = check_answerability(ws, question, &state.acc).await {
                outcome.abstention = Some(reason);
                outcome.unresolved = gap_items(gaps);
                return outcome;
            }
        }
        // Refresh the gap list from the accumulated evidence only when the
        // round produced progress; a quiet round reuses the current list.
        if new_urls != 0 || novel_claims != 0 {
            let Some(next_gap_list) = extract_gap_list(ws, question, &state.acc, plan).await else {
                outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
                outcome.unresolved = gap_items(gaps);
                return outcome;
            };
            gap_list = next_gap_list;
            // Post-progress coder round (key = this gap round's index): fresh
            // prototypes targeting the refreshed gap list.
            if !gap_list.gaps.is_empty() && !state.coder_rounds_done.contains(&round_index) {
                run_coder_gated(
                    job_id,
                    run_root,
                    ws,
                    question,
                    budget,
                    &gap_list,
                    deadline,
                    state,
                    round_index,
                )
                .await;
            }
        }
        // Checkpoint after EACH gap round (see the function doc — the locals
        // must survive a crash mid-loop or the whole stage re-runs).
        state.round_index = round_index;
        state.gap_outcome.rounds_dispatched = outcome.rounds_dispatched;
        state.budget_spent = budget.spent;
        state.gap_list = Some(gap_list.clone());
        state.save(job_id).await;
    }
}

/// Per-round claim annotation pass: classify each pending claim against the
/// existing accumulated claims via a single orchestrator extraction call.
/// The validator guarantees id completeness and in-range existing ids
/// (fail-closed inside the extraction retry loop). Weak hints render in the
/// existing-claims listing so later rounds see them (never silent).
async fn annotate_claims(
    ws: &Workspace,
    existing: &[Claim],
    weak: &WeakLinks,
    pending: &[Claim],
) -> Result<AnnotationPass> {
    let mut existing_claims = String::new();
    for (i, c) in existing.iter().enumerate() {
        let _ = writeln!(existing_claims, "{i}: {}", c.claim);
        existing_claims.push_str(&render_weak_hints(weak, i));
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
/// `duplicate`/`contradicts` cite an in-range existing claim; `novel` cites
/// nothing; the contradiction note is present exactly when the verdict is
/// "contradicts". Structural only — the verbatim-proof requirement is gone
/// (id references + the LLM's semantic judgment are the only gates).
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
    }
    Ok(())
}

/// Confirm pass over a round's mutating annotation links (duplicate /
/// contradicts pairs only): ONE lightweight orchestrator extraction
/// re-judging each pair, between the annotation pass and applying its
/// results. Fail-closed completeness — every mutating pair judged exactly
/// once with the {confirm, reject} vocabulary; the caller maps a failed call
/// to all-weak + marker (never all-novel fallback, never dropped claims).
async fn confirm_links(
    ws: &Workspace,
    existing: &[Claim],
    pending: &[Claim],
    pass: &AnnotationPass,
) -> Result<ConfirmPass> {
    let mut links = String::new();
    for a in pass.annotations.iter().filter(|a| a.verdict != "novel") {
        let existing_id = a
            .existing_id
            .expect("mutating verdict cites an existing claim");
        let p = &pending[a.new_id];
        let e = &existing[existing_id];
        let _ = writeln!(
            links,
            "- new claim {}: \"{}\" [{}] ↔ existing claim {}: \"{}\" (annotation: {})",
            a.new_id, p.claim, p.confidence, existing_id, e.claim, a.verdict
        );
    }
    let mut user = substitute(
        &load_prompt("research/confirm.md"),
        &[("{{links}}", &links)],
    );
    user.push_str("\n\n");
    user.push_str(&load_prompt("extraction/confirm.md"));
    // The validator captures the mutating new_ids (the validator type is
    // `'static`) — pair identity is pinned by the annotation pass, the model
    // never re-transcribes existing ids.
    let mutating: Vec<usize> = pass
        .annotations
        .iter()
        .filter(|a| a.verdict != "novel")
        .map(|a| a.new_id)
        .collect();
    orchestrator_extract::<ConfirmPass>(
        ws,
        "confirm_links",
        &user,
        Some(&move |c| validate_confirm(c, &mutating)),
    )
    .await
}

/// Validate a confirm pass: exactly the mutating links judged, each exactly
/// once (set equality on new_ids), verdicts in [confirm, reject]. Structural
/// only, like the annotation validator.
fn validate_confirm(pass: &ConfirmPass, mutating: &[usize]) -> Result<(), String> {
    let mut ids: Vec<usize> = pass.links.iter().map(|l| l.new_id).collect();
    ids.sort_unstable();
    let mut expected = mutating.to_vec();
    expected.sort_unstable();
    if ids != expected {
        return Err(format!(
            "confirm pass must judge exactly the mutating links (new_ids {expected:?}), got {ids:?}"
        ));
    }
    for l in &pass.links {
        if !matches!(l.verdict.as_str(), "confirm" | "reject") {
            return Err(format!("verdict '{}' not in [confirm, reject]", l.verdict));
        }
    }
    Ok(())
}

/// Run the annotation pass over a round's pending claims, then the optional
/// confirm pass over its mutating links, and apply the results. Returns the
/// number of novel claims (the saturation signal). On annotation exhaustion
/// every pending claim is treated as novel with an explicit marker — claims
/// are never dropped. A failed confirm call degrades every mutating verdict
/// to weak/unconfirmed with an explicit marker — never all-novel fallback.
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
    let Ok(pass) = annotate_claims(ws, &acc.claims, &acc.weak, pending).await else {
        acc.claims.extend(pending.iter().cloned());
        if !markers.iter().any(|m| m == CLAIM_ANNOTATION_FAILED) {
            markers.push(CLAIM_ANNOTATION_FAILED.to_string());
        }
        return pending.len();
    };
    // Confirm pass over the mutating links only (one call per round). No
    // mutating links → empty outcome, every verdict applies as-is.
    let confirm = if pass.annotations.iter().any(|a| a.verdict != "novel") {
        if let Ok(c) = confirm_links(ws, &acc.claims, pending, &pass).await {
            ConfirmOutcome::Passed(c)
        } else {
            // Fail-open: every mutating verdict becomes weak/unconfirmed
            // (never all-novel fallback, never dropped claims).
            if !markers.iter().any(|m| m == CONFIRM_FAILED) {
                markers.push(CONFIRM_FAILED.to_string());
            }
            ConfirmOutcome::Failed
        }
    } else {
        ConfirmOutcome::Passed(ConfirmPass::default())
    };
    acc.apply_annotations(&pass, pending, &confirm)
}

/// After a structural quiet round (no new claims, no new sources): is the
/// question genuinely unanswerable with the evidence gathered? `Some(abstention)`
/// when it is (orchestrator call, not budgeted).
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

/// Marker surfaced in the head-placed `## Run markers` section when the final
/// synthesis was delivered provider-truncated — never silent success.
const SYNTHESIS_TRUNCATED_MARKER: &str =
    "final synthesis truncated by the provider — last produced output delivered";

/// Delivered synthesis: the report text plus an optional truncation marker.
#[derive(Debug)]
struct SynthesisOutput {
    text: String,
    marker: Option<String>,
}

/// Final synthesis from the accumulated evidence with the shared synthesis
/// policy (≤3 informed attempts, transport-only backoff, hard time cap).
/// A truncated, empty, or failed attempt counts toward the budget; the next
/// attempt carries feedback to shorten/compress. The last produced output
/// wins — a provider-truncated report is delivered with an explicit marker,
/// never silent success. An exhaustion with no usable output at all (transport
/// failures / empty responses) errors — the caller's partial-report fail-open
/// path applies.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
async fn synthesize(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    abstention: Option<&str>,
) -> Result<SynthesisOutput> {
    let _call = crate::call_registry::NON_AGENT_CALLS.register("synthesize", &ws.name);
    let evidence = render_accumulated_evidence(acc);
    let mut base_user = substitute(
        &load_prompt("research/synthesize.md"),
        &[("{{question}}", question), ("{{evidence}}", &evidence)],
    );
    if let Some(abstain) = abstention {
        let _ = writeln!(
            base_user,
            "\n\n# Answerability Note\n\nThe research team determined the question is not \
             answerable with the available evidence: {abstain}\n\
             State this clearly and explain what evidence would be needed."
        );
    }
    let policy = crate::retry::RetryPolicy::synthesis();
    let mut params = orchestrator_params(ws, "synthesize");
    let mut loop_state = crate::retry::RetryLoop::new(&policy, params.meta.as_ref());
    let operation_started = Instant::now();
    let mut prefix = Vec::with_capacity(2);
    crate::prompt::prepend_general_context(&mut prefix, ws).await;
    let mut last: Option<String> = None;
    let mut last_truncated = false;
    let mut any_truncated = false;
    let mut feedback = String::new();
    let mut transport_failures = 0u32;

    for attempt in 1..=policy.max_attempts {
        if loop_state.expired() {
            break;
        }
        let mut user = base_user.clone();
        if !feedback.is_empty() {
            let _ = writeln!(user, "\n\n# Previous Attempt Feedback\n\n{feedback}");
        }
        let mut messages = prefix.clone();
        messages.push(ChatMessage::user(&user));
        params.messages = messages;
        let attempt_started = Instant::now();
        match crate::providers::chat_scoped(
            params.clone(),
            policy.idle_timeout,
            loop_state.deadline(),
        )
        .await
        {
            Ok(resp) => {
                let text = resp.text_or_empty().to_string();
                let truncated = resp.finish_reason.as_deref() == Some("length");
                if truncated {
                    any_truncated = true;
                    let err = anyhow::anyhow!(
                        "synthesis truncated by the provider (finish_reason=length)"
                    );
                    let rec = crate::retry::RetryFailureRecord::new_simple(
                        attempt,
                        FailureClass::TruncatedOutput,
                        &err,
                        attempt_started.elapsed().as_millis() as u64,
                        None,
                    );
                    loop_state.record(attempt, rec).await;
                    feedback = "Your previous report was truncated by the output limit — \
                                produce a SHORTER, more compressed version. Keep every \
                                load-bearing claim and its source, but tighten the prose so \
                                the whole report fits within the limit."
                        .to_string();
                    if !text.trim().is_empty() {
                        last = Some(text);
                        last_truncated = true;
                    }
                    continue;
                }
                if text.trim().is_empty() {
                    let err = anyhow::anyhow!("synthesis attempt returned empty text");
                    let rec = crate::retry::RetryFailureRecord::new_simple(
                        attempt,
                        FailureClass::NoResponse,
                        &err,
                        attempt_started.elapsed().as_millis() as u64,
                        None,
                    );
                    loop_state.record(attempt, rec).await;
                    feedback = "Your previous attempt returned an empty response — \
                                produce the report now."
                        .to_string();
                    continue;
                }
                // Clean completion wins.
                crate::stats::record_llm_success(&params, operation_started, attempt, &resp).await;
                return Ok(SynthesisOutput { text, marker: None });
            }
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(attempt, err.record).await;
                if non_retryable {
                    break;
                }
                transport_failures += 1;
                if attempt < policy.max_attempts
                    && let Err(FailureClass::Shutdown) =
                        loop_state.sleep_between(transport_failures).await
                {
                    break;
                }
            }
        }
    }
    // Exhausted: the last produced output wins (marked when truncated); no
    // usable output at all errors into the caller's partial-report fail-open.
    let final_class = loop_state.final_class();
    let exhausted = crate::retry::RetryExhausted::with_last_raw(
        loop_state.into_failures(),
        final_class,
        last.clone(),
    );
    crate::stats::record_llm_failure(&params, operation_started, &exhausted).await;
    if let Some(text) = last {
        let marker = last_truncated.then(|| SYNTHESIS_TRUNCATED_MARKER.to_string());
        Ok(SynthesisOutput { text, marker })
    } else if any_truncated {
        Err(anyhow::anyhow!(
            "final synthesis truncated with no usable output: {SYNTHESIS_TRUNCATED_MARKER}"
        ))
    } else {
        Err(anyhow::anyhow!(
            "final synthesis produced no usable output after {} attempts",
            policy.max_attempts
        ))
    }
}

/// Build the verification target list: primary targets (contradiction notes
/// or low confidence) first, then weak-duplicate claims not already primary —
/// filling only empty slots, never displacing higher-priority targets, never
/// double-dispatching a claim that qualifies both ways (weak-contradiction
/// claims are never appended here: they already carry notes and qualify via
/// the primary filter). Returns `(targets, primary_count)` so the caller can
/// bound the unresolved filler to primary targets only.
fn verification_targets(acc: &AccumulatedEvidence) -> (Vec<VerificationTarget>, usize) {
    let mut seen = HashSet::new();
    let mut targets = Vec::new();
    for (i, c) in acc.claims.iter().enumerate() {
        if !c.contradictions.is_empty() || c.confidence == "low" {
            seen.insert(i);
            targets.push(VerificationTarget::new(
                &c.claim,
                &c.source,
                &c.contradictions.join("; "),
            ));
        }
    }
    let primary_count = targets.len();
    // Weak duplicates fill only empty slots — appended AFTER primaries with a
    // hint-free target (weakness never leaks toward verifiers). The verify
    // schema has no "duplicate" verdict, so the verifier judges the standalone
    // claim's truth and the duplicate-vs-novel ambiguity resolves indirectly:
    // the claim stays visible as its own evidence entry, and the fresh
    // analyst's web tools can surface the relation.
    for &(claim_id, _) in &acc.weak.duplicates {
        if seen.insert(claim_id) {
            let c = &acc.claims[claim_id];
            targets.push(VerificationTarget::new(&c.claim, &c.source, ""));
        }
    }
    (targets, primary_count)
}

/// Verification gate: fresh analysts verify the disputed / low-confidence
/// accumulated claims (budgeted, bounded), plus any weak-duplicate claims
/// filling empty verifier slots. Anything still disputed stays marked
/// unresolved in the final report. Verifier tool calls / searches / queries
/// count toward the run summary and register in the query ledger (later
/// rounds must not re-ask them) — repeats here inflate the summary's
/// repeat-queries line but never the per-round saturation signal
/// (`EvidenceRound::repeat_queries` is untouched).
#[allow(clippy::too_many_arguments)]
async fn research_verification_pass(
    ws: &Workspace,
    acc: &AccumulatedEvidence,
    budget: &mut ResearchBudget,
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
    deadline: std::time::Instant,
    resume: bool,
    run_root: &str,
    captured: &mut Vec<String>,
) -> Vec<VerificationResult> {
    let (targets, primary_count) = verification_targets(acc);
    if targets.is_empty() {
        return Vec::new();
    }
    let cap = targets.len().min(crate::tools::ask::VERIFY_MAX_ANALYSTS);
    // Reserve what fits — a near-exhausted budget still verifies the
    // highest-priority targets instead of nothing.
    let n = cap.min(budget.cap.saturating_sub(budget.spent));
    let mut results = Vec::new();
    // Never spawn verifiers once the round-wide deadline expired — they
    // would be aborted instantly; the primary targets are marked unresolved
    // below instead (same shape as the budget-exhaustion path).
    if n > 0 && std::time::Instant::now() < deadline && budget.try_reserve(n).is_ok() {
        let ledger_snapshot = ledger.render();
        let task_extra = format!(
            "\n# Queries Already Asked (do not repeat these verbatim)\n\n{ledger_snapshot}\n\n\
             # Scratch Workspace\n\nTemporary per-run folder (wiped after the run):\n\n{run_root}"
        );
        let prefix = format!("research_{}_verify", ws.name);
        let (verify_results, verify_ids) =
            dispatch_claim_verifiers(ws, &prefix, &targets[..n], &task_extra, deadline, resume)
                .await;
        captured.extend(verify_ids);
        results = verify_results;
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
    // Primary targets beyond the verified set are explicitly marked
    // unresolved — never silently skipped. Appended weak-duplicate targets
    // that did not fit are not (they fill only empty slots; their weak state
    // stays visible in the evidence view and run summary).
    for t in targets
        .iter()
        .skip(results.len())
        .take(primary_count.saturating_sub(results.len()))
    {
        results.push(VerificationResult {
            claim: t.claim.clone(),
            verdict: "unresolved".to_string(),
            evidence: "verification skipped — budget exhausted or round deadline expired"
                .to_string(),
            tool_calls: 0,
            searches: 0,
            queries: Vec::new(),
        });
    }
    results
}

// ── Rendering ────────────────────────────────────────────────────────────

/// Render the weak/unconfirmed hints involving claim `id` (one indented line
/// per hint, empty when none). Shared by the accumulated-evidence view, the
/// annotate listing, and the partial report — weakness is per-claim visible
/// and never silent.
fn render_weak_hints(weak: &WeakLinks, id: usize) -> String {
    let mut out = String::new();
    for &(claim, target) in &weak.duplicates {
        if claim == id {
            let _ = writeln!(
                out,
                "   weak: possibly duplicate of #{target} (unconfirmed)"
            );
        }
    }
    for &(claim, target) in &weak.contradictions {
        if claim == id {
            let _ = writeln!(out, "   weak: possibly contradicts #{target} (unconfirmed)");
        }
        if target == id {
            let _ = writeln!(out, "   weak: possibly contradicts #{claim} (unconfirmed)");
        }
    }
    out
}

/// "n/a" fallback for empty claim sources.
fn source_or_na(source: &str) -> &str {
    if source.is_empty() { "n/a" } else { source }
}

/// Unanswered section; `escape` fences entries for the manager-visible
/// partial report (the orchestrator-prompt view stays raw).
fn render_unanswered(out: &mut String, unanswered: &[String], escape: bool) {
    if unanswered.is_empty() {
        return;
    }
    let _ = writeln!(out, "\nAnalysts reported these as still unanswered:");
    for u in unanswered {
        let _ = if escape {
            writeln!(out, "- {}", escape_fences(u))
        } else {
            writeln!(out, "- {u}")
        };
    }
}

/// Per-analyst raw-report section under `heading` (### partial / ## final).
fn render_raw_reports(out: &mut String, raw_reports: &[String], heading: &str) {
    if raw_reports.is_empty() {
        return;
    }
    let _ = writeln!(out, "\n{heading} Failed Analyst Reports");
    for (i, raw) in raw_reports.iter().enumerate() {
        let _ = writeln!(out, "### Report from Analyst {}", i + 1);
        let _ = writeln!(out, "{}", escape_fences(raw));
    }
}

/// Render the accumulated evidence as a compact numbered list for the
/// orchestrator prompts, plus analysts' self-reported unanswered aspects.
/// Claim ids are their stable 0-based indices in `acc.claims` — the
/// annotation pass and final synthesis reference them by these ids.
fn render_accumulated_evidence(acc: &AccumulatedEvidence) -> String {
    let mut out = String::new();
    for (i, c) in acc.claims.iter().enumerate() {
        let source = source_or_na(&c.source);
        let _ = writeln!(
            out,
            "{i}. [{}] {} — source: {source}",
            c.confidence, c.claim,
        );
        if !c.contradictions.is_empty() {
            let _ = writeln!(out, "   contradictions: {}", c.contradictions.join("; "));
        }
        out.push_str(&render_weak_hints(&acc.weak, i));
    }
    render_unanswered(&mut out, &acc.unanswered, false);
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
    let weak_links = acc.weak.duplicates.len() + acc.weak.contradictions.len();
    if weak_links > 0 {
        let _ = writeln!(out, "- weak/unconfirmed links: {weak_links}");
    }
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

/// Render the recovered-from-timed-out-analysts section. Separate from the
/// main evidence by design: recovered findings never entered verification or
/// synthesis. Empty when there is nothing recovered (also the boot path).
/// Head-placed by callers so the section survives report truncation.
fn render_recovered_findings(recovered: &[AnalystFindings]) -> String {
    if recovered.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let _ = writeln!(out, "\n## Recovered from timed-out analysts");
    let _ = writeln!(
        out,
        "Findings recovered from analysts whose work was cut short by the round \
         deadline (deadline exceeded, unverified — not subject to verification; each \
         analyst's final in-flight turn could not be recovered):"
    );
    for r in recovered {
        for c in &r.claims {
            let source = source_or_na(&c.source);
            let _ = writeln!(
                out,
                "- [{}] {} — source: {}",
                c.confidence,
                escape_fences(&c.claim),
                escape_fences(source),
            );
            if !c.contradictions.is_empty() {
                let joined = c.contradictions.join("; ");
                let _ = writeln!(out, "  contradictions: {}", escape_fences(&joined));
            }
        }
        for u in &r.unanswered {
            let _ = writeln!(out, "- unanswered: {}", escape_fences(u));
        }
    }
    out
}

/// Partial envelope on shutdown mid-run: whatever evidence was gathered is
/// delivered with an explicit incomplete marker — findings are never lost.
fn partial_report(
    question: &str,
    acc: &AccumulatedEvidence,
    reason: &str,
    recovered: &[AnalystFindings],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Research Report (incomplete — {reason})");
    let _ = writeln!(out);
    let _ = writeln!(out, "**Question**: {question}");
    let _ = writeln!(out);
    out.push_str(&render_recovered_findings(recovered));
    let _ = writeln!(out, "### Evidence Gathered So Far");
    if acc.claims.is_empty() {
        let _ = writeln!(out, "- none");
    } else {
        for (i, c) in acc.claims.iter().enumerate() {
            let source = source_or_na(&c.source);
            let _ = writeln!(
                out,
                "- {i}. [{}] {} — source: {source}",
                c.confidence,
                escape_fences(&c.claim),
            );
            out.push_str(&render_weak_hints(&acc.weak, i));
        }
    }
    render_unanswered(&mut out, &acc.unanswered, true);
    render_raw_reports(&mut out, &acc.raw_reports, "###");
    out
}

// ── Orchestrator ─────────────────────────────────────────────────────────

/// Exit of one `run_deep_research` invocation. Self-documenting where the old
/// `(Result<String>, bool)` was subtle: `aborted` distinguished a real
/// terminalization (artifacts + job completion in the caller) from a
/// shutdown/drain abort (run stays alive, nothing written).
enum ResearchExit {
    /// Real terminalization (success, partial, or error) — the caller writes
    /// results.md + cleaner ticket and completes the durable job.
    Terminal(anyhow::Result<String>),
    /// Shutdown/drain abort — the run stays alive for the next boot; nothing
    /// is written, terminalized, or routed (design pin).
    Aborted,
}

/// Run the resumable deep-research orchestrator: round 0 (decomposition),
/// round 1 (per-sub-question researchers), conditional gap rounds, one-shot
/// final synthesis, verification gate, and the run summary. `job_id` names
/// the durable research job whose `research_jobs.state` checkpoint is loaded
/// on entry and saved at every stage boundary.
#[allow(clippy::too_many_lines)]
async fn run_deep_research(
    ws: &Workspace,
    question: &str,
    job_id: &str,
    resume: bool,
) -> ResearchExit {
    // Hold a non-agent-call guard for the WHOLE run: the drain-watch fires the
    // token only when both registries are empty, and this guard bridges the
    // inter-phase windows (analyst deregistration → the next orchestrator LLM
    // call) so the token is never fired into a just-about-to-start call. The
    // orchestrator checks the drain at every round boundary and exits via the
    // partial-report path, releasing the guard promptly.
    let _orchestrator_guard =
        crate::call_registry::NON_AGENT_CALLS.register("research_orchestrator", &ws.name);
    let start = Instant::now();
    // One round-wide bound shared by every phase's member waits
    // (decomposition, research rounds, verification): a stuck analyst is
    // aborted at it, so no phase can hang the round. Sequential provider
    // calls after the deadline finish within their own retry bounds.
    let deadline = std::time::Instant::now() + round_timeout();
    let mut state = ResearchState::load(job_id).await;
    let mut budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
    budget.spent = state.budget_spent;
    let mut run_stats = RunStats::default();
    // Per-run scratch folder — created idempotently BEFORE round 0 (decompose
    // receives it first). The absolute temp-dir path is what the readonly
    // shell allows; `$TMPDIR` in the agent shell differs on macOS.
    let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
    let run_root_str = run_root.to_string_lossy().to_string();
    // Recovered-from-timed-out-analysts findings: round locals only (never
    // checkpointed) — a crash between rounds loses them, and a resumed run
    // starts with an empty set (analysts re-run from their unchanged sessions).
    let mut recovered: Vec<AnalystFindings> = Vec::new();

    // ── Round 0 — decomposition + merge (resumable) ───────────────────
    let mut round_agents: Vec<String> = Vec::new();
    if state.stage == ResearchStage::Decompose {
        let round0 = round0_decompose(
            ws,
            question,
            &mut budget,
            &mut run_stats,
            deadline,
            resume,
            &run_root_str,
            &mut round_agents,
        )
        .await;
        // Capture on EVERY outcome — a hard decompose failure still leaves
        // the dispatched agents' sessions behind (their OS-temp scratch must
        // reach the cleaner ticket).
        state.capture_round(&round_agents, &run_root).await;
        round_agents.clear();
        let plan = match round0 {
            Ok((plan, marker)) => {
                if let Some(m) = marker {
                    state.markers.push(m);
                }
                plan
            }
            Err(_) if crate::shutdown::aborting() => {
                // Persist captured commands: the run stays alive for boot
                // resume, which re-dispatches round 0 with FRESH agent ids —
                // the aborted agents' OS-temp scratch must reach the cleaner
                // ticket (the in-memory capture dies with this invocation).
                state.save(job_id).await;
                return ResearchExit::Aborted;
            }
            Err(e) => {
                // Persist the captured commands — the dispatch-time cleaner
                // ticket reloads state from the store and a hard decompose
                // failure must not lose the agents' sessions from the report.
                state.save(job_id).await;
                return ResearchExit::Terminal(Err(e));
            }
        };
        state.plan = Some(plan);
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::Round1;
        state.save(job_id).await;
    }

    // Check shutdown/drain between rounds — never spawn round-1 analysts
    // during shutdown or the graceful drain.
    if crate::shutdown::aborting() {
        return ResearchExit::Aborted;
    }

    // ── Round 1 — one analyst per sub-question (resumable) ────────────
    if state.stage == ResearchStage::Round1 {
        let Some(plan) = state.plan.as_ref() else {
            return ResearchExit::Terminal(Err(anyhow::anyhow!(
                "research state missing plan at round 1"
            )));
        };
        let Some((r1, r1_timed_out)) = round1_research(
            ws,
            question,
            plan,
            &mut budget,
            &mut state.ledger,
            &mut run_stats,
            deadline,
            resume,
            &run_root_str,
            &mut round_agents,
        )
        .await
        else {
            return ResearchExit::Terminal(Ok(partial_report(
                question,
                &state.acc,
                "round 1 skipped — analyst budget exhausted",
                &recovered,
            )));
        };
        state.capture_round(&round_agents, &run_root).await;
        round_agents.clear();
        let (_, pending) = state.acc.absorb(&r1);
        annotate_round(ws, &mut state.acc, &pending, &mut state.markers).await;
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::GapRounds;
        state.save(job_id).await;
        // Wrap-up AFTER the checkpoint: a crash mid-wrap-up loses only the
        // recovered findings (ticket-accepted), never the round-1 evidence.
        // Its ledger registrations are not in the checkpoint (fail-open — a
        // crash during the wrap-up resumes with a ledger missing the dead
        // analysts' queries, so gap rounds may re-ask them, bounded).
        recovered.extend(wrap_up_timed_out(r1_timed_out, &mut state.ledger, &mut run_stats).await);
    }

    // ── Interim consolidation + conditional gap rounds (resumable) ────
    if state.stage == ResearchStage::GapRounds {
        // Clone the plan so gap_rounds can take &mut state (per-round
        // checkpoints) while still referencing the merged decomposition.
        let Some(plan) = state.plan.clone() else {
            return ResearchExit::Terminal(Err(anyhow::anyhow!(
                "research state missing plan at gap rounds"
            )));
        };
        let gap_outcome = gap_rounds(
            ws,
            question,
            &plan,
            &mut budget,
            &mut state,
            &mut run_stats,
            deadline,
            job_id,
            &run_root_str,
            resume,
            &mut recovered,
        )
        .await;
        // Aborting mid-gap-loop (drain/shutdown): leave the stage at
        // GapRounds so the next boot's resume CONTINUES the loop at the
        // accumulated round_index (the per-round checkpoints inside
        // gap_rounds already persisted it + the current gap list) instead of
        // skipping the remaining gap rounds and synthesizing from truncated
        // evidence (design pin: "empty unresolved + rounds_dispatched = k
        // means continue at round k+1").
        if crate::shutdown::aborting() {
            state.gap_outcome = gap_outcome;
            state.budget_spent = budget.spent;
            state.save(job_id).await;
            return ResearchExit::Aborted;
        }
        // Round-trip the gap-loop locals on NORMAL exit (coverage complete,
        // abstention, or budget/deadline exhaustion): round_index is already
        // accumulated by the per-round checkpoints inside gap_rounds (it
        // starts from the stored value and increments per round — do NOT
        // clobber it with the per-invocation count).
        state.gap_outcome = gap_outcome;
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::Verification;
        state.save(job_id).await;
    }
    let rounds_used = 2 + state.gap_outcome.rounds_dispatched;

    // The gap loop may have exited early on shutdown/drain — never run a full
    // synthesis or spawn verification analysts during shutdown or the drain.
    if crate::shutdown::aborting() {
        return ResearchExit::Aborted;
    }

    // ── Final synthesis (resumable) ───────────────────────────────────
    // Runs for every stage ≥ GapRounds: a first synthesis advances the stage
    // and persists; a resume after Synthesis re-synthesizes (bounded — the
    // accumulated evidence is unchanged, so the synthesis is deterministic
    // modulo LLM nondeterminism). The marker-dedup guard makes re-runs
    // idempotent (a resume after a failed stage save cannot duplicate
    // markers).
    let synthesis = match synthesize(
        ws,
        question,
        &state.acc,
        state.gap_outcome.abstention.as_deref(),
    )
    .await
    {
        Ok(s) => {
            if let Some(marker) = s.marker
                && !state.markers.contains(&marker)
            {
                state.markers.push(marker);
            }
            if state.stage != ResearchStage::Synthesis {
                state.stage = ResearchStage::Synthesis;
                state.save(job_id).await;
            }
            s.text
        }
        Err(_) if crate::shutdown::aborting() => {
            return ResearchExit::Aborted;
        }
        Err(e) => {
            return ResearchExit::Terminal(Ok(partial_report(
                question,
                &state.acc,
                &format!("synthesis failed: {e}"),
                &recovered,
            )));
        }
    };

    // ── Verification gate (budgeted, resumable) ───────────────────────
    // Never spawn verifiers during shutdown or the drain — deliver the
    // synthesized report as-is (partial is acceptable).
    let verification = if crate::shutdown::aborting() {
        Vec::new()
    } else if !state.verification.is_empty() {
        std::mem::take(&mut state.verification)
    } else {
        let v = research_verification_pass(
            ws,
            &state.acc,
            &mut budget,
            &mut state.ledger,
            &mut run_stats,
            deadline,
            resume,
            &run_root_str,
            &mut round_agents,
        )
        .await;
        state.capture_round(&round_agents, &run_root).await;
        round_agents.clear();
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::Synthesis;
        state.save(job_id).await;
        v
    };

    // Fail-open markers survive delivery: head-placed so they survive the
    // manager's sandwich truncation of long reports. The recovered-findings
    // section is head-placed right after them for the same reason.
    let mut report = String::new();
    if !state.markers.is_empty() {
        let _ = writeln!(report, "## Run markers");
        for m in &state.markers {
            let _ = writeln!(report, "- {m}");
        }
        let _ = writeln!(report);
    }
    report.push_str(&render_recovered_findings(&recovered));
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
    render_raw_reports(&mut report, &state.acc.raw_reports, "##");
    // Prototypes section: files the coder-in-loop left in the per-run folder.
    // The folder is temporary — swept after the delivery grace. A resume after
    // an OS temp cleanup recreates the folder with no prototypes (fail-open).
    if !state.coder_rounds_done.is_empty() {
        let files = crate::research_cleanup::run_root_files(job_id).await;
        let _ = writeln!(report);
        let _ = writeln!(report, "## Prototypes");
        if files.is_empty() {
            // Distinguish a coder round that wrote nothing from prototypes
            // lost to an OS temp cleanup between boots: a resumed run with
            // completed coder rounds and an empty folder means the OS cleaned
            // temp while the daemon was down (ensure_run_root recreates the
            // folder empty; the completed rounds never re-run). A fresh run's
            // empty folder is just "wrote nothing".
            if resume {
                let _ = writeln!(
                    report,
                    "None found in the per-run folder (prototypes may have been lost \
                     if the OS cleaned the temp dir between boots)."
                );
            } else {
                let _ = writeln!(report, "No prototypes were written by the coder rounds.");
            }
        } else {
            let _ = writeln!(
                report,
                "Per-run folder: `{run_root_str}` — TEMPORARY, swept after the delivery \
                 grace. Files produced by the coder rounds:"
            );
            for f in &files {
                let _ = writeln!(report, "- {f}");
            }
        }
    }
    // RunStats pin: a resumed run's counts undercount the pre-crash segment —
    // the summary carries a one-line best-effort note instead of pretending.
    // Keyed off the job's boot-resume retry count (the real resume signal) —
    // gap rounds dispatched within THIS process are complete and need no
    // caveat (and the abort path can skip a run's own accumulation).
    if crate::jobs::job_retry_count(&crate::session::store().conn, job_id).await > 0 {
        let _ = writeln!(
            report,
            "\n> Run telemetry is best-effort: this run resumed from a checkpoint, so tool-call \
             and query counts reflect only the post-resume segment."
        );
    }
    let _ = writeln!(report);
    report.push_str(&render_run_summary(
        &run_stats,
        &budget,
        rounds_used,
        &state.acc,
        state.gap_outcome.abstention.as_deref(),
        &state.gap_outcome.unresolved,
        state.gap_outcome.incomplete.as_deref(),
        &state.markers,
        start.elapsed(),
    ));
    ResearchExit::Terminal(Ok(report))
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
    fn test_research_fail_open_envelope() {
        // All-decomposers-failed follows the ask convention: an error envelope
        // with an explicit marker, never a silent drop.
        let envelope = build_async_research_message(&Err(anyhow::anyhow!(
            "all decomposition analysts failed"
        )));
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

    /// The boot-scan over-cap path must deliver a PARTIAL REPORT to the
    /// stored caller — the research envelope is the caller's only result
    /// path, so a failed row with no envelope would strand the Manager
    /// forever (the exact stranding class this path exists to prevent).
    ///
    /// Serialized with the drain-flag writers: `research_capped_partial_report`
    /// consults the process-global drain flag and aborts early while it is
    /// set (project convention: retry_tests_lock).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn research_capped_delivers_partial_report_to_caller() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_capped");
        let job_id = "research_job_capped_1";
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
             retry_count, created_at, updated_at) \
             VALUES (?1, 'research', 'launched', ?2, ?3, 'caller-user', 'telegram', 'assistant', \
             3, ?4, ?4)",
            crate::turso::params![job_id, "question?", ws.name.clone(), now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, question, state, created_at, updated_at) \
             VALUES (?1, 'question?', '{}', ?2, ?2)",
            crate::turso::params![job_id, now],
        )
        .await
        .unwrap();

        research_capped_partial_report(job_id, &ws).await;

        let job_rows = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(job_rows.len(), 0, "capped job must be terminalized");
        let pending = conn
            .query(
                "SELECT role, user_name, envelope FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(pending.len(), 1, "partial-report envelope persisted");
        assert_eq!(
            pending[0].get::<String>(0).unwrap(),
            "assistant",
            "delivered to the original caller role, not Manager"
        );
        assert_eq!(pending[0].get::<String>(1).unwrap(), "caller-user");
        let envelope: String = pending[0].get(2).unwrap();
        assert!(
            envelope.contains("boot re-dispatch cap exceeded"),
            "the partial report must surface the cap reason: {envelope}"
        );
    }

    /// Direct resume test for `resume_research_run`: a job
    /// checkpointed at stage=Verification with pre-populated verification
    /// resumes — it re-enters the orchestrator, synthesizes from the
    /// accumulated evidence (ONE provider call), skips the verification pass
    /// (stored results reused — no analysts spawned), terminalizes into the
    /// durable envelope, and delivers to the ORIGINAL caller
    /// (role/user/channel persisted on the job row, never the Manager).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn resume_research_run_continues_at_verification_stage() {
        crate::util::test::init_management_test_stores().await;
        let _lock = crate::util::test::retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // One synthesis call (the only LLM work left at stage=Verification).
        let fake = crate::util::test::FakeProvider::new()
            .ok("final synthesized report for the resumed run");
        let _provider_guard = crate::util::test::install_fake_provider(std::sync::Arc::new(fake));

        let ws = crate::workspace::test_ws("/tmp/test_ws_research_resume");
        let job_id = "research_job_resume_1";
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
             retry_count, created_at, updated_at) \
             VALUES (?1, 'research', 'launched', ?2, ?3, 'caller-user', 'telegram', 'assistant', \
             0, ?4, ?4)",
            crate::turso::params![job_id, "question?", ws.name.clone(), now.clone()],
        )
        .await
        .unwrap();

        // Checkpointed state: stage=Verification, one accumulated claim, one
        // stored verification result (the post-crash resume must reuse it —
        // never re-run the verification pass).
        let mut state = ResearchState {
            stage: ResearchStage::Verification,
            plan: None,
            gap_list: None,
            acc: AccumulatedEvidence {
                urls: std::collections::HashSet::new(),
                claims: vec![crate::tools::ask::Claim {
                    claim: "alpha is a real project".into(),
                    source: "s1".into(),
                    confidence: "high".into(),
                    contradictions: vec![],
                }],
                unanswered: vec![],
                unanswered_keys: std::collections::HashSet::new(),
                raw_reports: vec![],
                weak: WeakLinks::default(),
            },
            ledger: QueryLedger::default(),
            markers: vec![],
            gap_outcome: GapRoundsOutcome::default(),
            budget_spent: 0,
            round_index: 0,
            verification: vec![crate::tools::ask::VerificationResult {
                claim: "alpha is a real project".into(),
                verdict: "confirmed".into(),
                evidence: "primary source".into(),
                tool_calls: 0,
                searches: 0,
                queries: vec![],
            }],
            commands: vec![],
            coder_rounds_done: vec![],
        };
        state.acc.rebuild_keys();
        let state_json = serde_json::to_string(&state).unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, question, state, created_at, updated_at) \
             VALUES (?1, 'question?', ?2, ?3, ?3)",
            crate::turso::params![job_id, state_json, now.clone()],
        )
        .await
        .unwrap();

        resume_research_run(job_id, &ws).await;

        // Terminalized into the durable envelope addressed to the stored
        // caller (the consumer skips it — the workspace is not registered, so
        // the pending row survives for the assertion).
        let job_rows = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(job_rows.len(), 0, "resumed job must be terminalized");
        let pending = conn
            .query(
                "SELECT role, user_name, envelope FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(pending.len(), 1, "resume envelope persisted");
        assert_eq!(
            pending[0].get::<String>(0).unwrap(),
            "assistant",
            "delivered to the original caller role, not Manager"
        );
        assert_eq!(pending[0].get::<String>(1).unwrap(), "caller-user");
        let envelope: String = pending[0].get(2).unwrap();
        assert!(
            envelope.contains("final synthesized report"),
            "the resume envelope must carry the synthesized report: {envelope}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn test_synthesis_truncated_output_is_marked_and_transport_fails_open() {
        let _lock = crate::util::test::retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = crate::workspace::test_ws("/tmp/test_ws");
        let acc = AccumulatedEvidence::default();

        // Every attempt provider-truncated (finish_reason=length): the last
        // produced output wins, marked — never silent success.
        let fake = crate::util::test::FakeProvider::new()
            .ok_with_finish("report part one", Some("length"))
            .ok_with_finish("report part two", Some("length"))
            .ok_with_finish("compressed final", Some("length"));
        let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
        let _provider_guard = crate::util::test::install_fake_provider(provider);
        let out = synthesize(&ws, "q", &acc, None)
            .await
            .expect("last produced output must be delivered");
        assert_eq!(out.text, "compressed final");
        assert!(
            out.marker
                .as_deref()
                .is_some_and(|m| m.contains("truncated")),
            "truncated delivery must carry the explicit marker: {:?}",
            out.marker
        );

        // A clean (non-truncated) attempt wins immediately without a marker.
        let fake = crate::util::test::FakeProvider::new()
            .ok_with_finish("part one", Some("length"))
            .ok("complete report");
        let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
        let _provider_guard = crate::util::test::install_fake_provider(provider);
        let out = synthesize(&ws, "q", &acc, None)
            .await
            .expect("clean completion wins");
        assert_eq!(out.text, "complete report");
        assert!(out.marker.is_none(), "clean completion is unmarked");

        // All transport failures → no usable output → Err (the caller's
        // partial-report fail-open path).
        let fake = crate::util::test::FakeProvider::new()
            .err(crate::retry::FailureClass::Transport, "outage")
            .err(crate::retry::FailureClass::Transport, "outage")
            .err(crate::retry::FailureClass::Transport, "outage");
        let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
        let _provider_guard = crate::util::test::install_fake_provider(provider);
        let err = synthesize(&ws, "q", &acc, None)
            .await
            .expect_err("transport exhaustion must error into the partial-report path");
        assert!(err.to_string().contains("no usable output"), "{err}");
    }

    #[test]
    fn test_validate_gap_list_traces_to_plan() {
        let plan = MergedPlan {
            sub_questions: vec![
                MergedSubQuestion {
                    question: "What is the price of X?".into(),
                    evidence_needed: "pricing page".into(),
                    risk: "low".into(),
                    from_id: 0,
                    also_ids: vec![],
                },
                MergedSubQuestion {
                    question: "Who maintains X?".into(),
                    evidence_needed: "repo metadata".into(),
                    risk: "medium".into(),
                    from_id: 1,
                    also_ids: vec![],
                },
            ],
            dropped: vec![],
        };
        let in_range = GapList {
            gaps: vec![Gap {
                kind: "unanswered".into(),
                item: "exact price".into(),
                traces_to: 0,
            }],
        };
        assert!(validate_gap_list(&in_range, &plan).is_ok());
        let out_of_range = GapList {
            gaps: vec![Gap {
                kind: "unanswered".into(),
                item: "unrelated".into(),
                traces_to: 5,
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
        // Global flat ids: plan 0: q1=0 q2=1; plan 1: q1=2 q3=3; plan 2: q4=4 q5=5.
        let base = |also: bool, dropped: bool| MergedPlan {
            sub_questions: vec![
                MergedSubQuestion {
                    question: String::new(),
                    evidence_needed: String::new(),
                    risk: String::new(),
                    from_id: 0,
                    also_ids: if also { vec![2] } else { vec![] },
                },
                MergedSubQuestion {
                    question: String::new(),
                    evidence_needed: String::new(),
                    risk: String::new(),
                    from_id: 1,
                    also_ids: vec![],
                },
                MergedSubQuestion {
                    question: String::new(),
                    evidence_needed: String::new(),
                    risk: String::new(),
                    from_id: 3,
                    also_ids: vec![],
                },
                MergedSubQuestion {
                    question: String::new(),
                    evidence_needed: String::new(),
                    risk: String::new(),
                    from_id: 4,
                    also_ids: vec![],
                },
            ],
            dropped: if dropped {
                vec![DroppedSubQuestion { id: 5 }]
            } else {
                vec![]
            },
        };
        assert!(
            validate_merged_plan(&base(true, true), &plans).is_ok(),
            "full coverage via from_id + also_ids + dropped"
        );
        assert!(
            validate_merged_plan(&base(false, true), &plans).is_err(),
            "silent dropout: plan 1's q1 (id 2) is never covered"
        );
        assert!(
            validate_merged_plan(&base(true, false), &plans).is_err(),
            "silent dropout: q5 (id 5) is never covered"
        );
        let mut bad = base(true, true);
        bad.sub_questions[0].from_id = 9;
        assert!(
            validate_merged_plan(&bad, &plans).is_err(),
            "out-of-range from_id is rejected"
        );
        let mut bad = base(true, true);
        bad.sub_questions[3].from_id = 0;
        assert!(
            validate_merged_plan(&bad, &plans).is_err(),
            "duplicate placement (id 0 twice) is rejected"
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
        let (urls, pending) = acc.absorb(&round1);
        assert_eq!((urls, pending.len()), (2, 2));
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
        let (urls, pending) = acc.absorb(&round2);
        assert_eq!(
            (urls, pending.len()),
            (1, 2),
            "only new URL (u3); every claim stays pending for annotation"
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
                contradiction: Some("price differs: $200 vs $100".into()),
            }],
        };
        let confirm = ConfirmOutcome::Passed(ConfirmPass {
            links: vec![ConfirmLink {
                new_id: 0,
                verdict: "confirm".into(),
            }],
        });
        let novel = acc.apply_annotations(&pass, &pending, &confirm);
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
                contradiction: None,
            }],
        };
        let confirm = ConfirmOutcome::Passed(ConfirmPass {
            links: vec![ConfirmLink {
                new_id: 0,
                verdict: "confirm".into(),
            }],
        });
        let novel = acc.apply_annotations(&pass, &pending, &confirm);
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

    #[test]
    fn test_apply_annotations_weak_duplicate_stays_standalone() {
        // A weak duplicate (confirm pass rejected / unclear / failed) is never
        // merged and never counts as novel: it stays standalone with a hint in
        // the side structure — "keep weak, clarify later".
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(Claim {
            claim: "alpha is true".into(),
            source: "u1".into(),
            confidence: "medium".into(),
            contradictions: vec![],
        });
        let pending = vec![Claim {
            claim: "alpha is true (restated)".into(),
            source: "u2".into(),
            confidence: "high".into(),
            contradictions: vec![],
        }];
        let pass = AnnotationPass {
            annotations: vec![ClaimAnnotation {
                new_id: 0,
                verdict: "duplicate".into(),
                existing_id: Some(0),
                contradiction: None,
            }],
        };
        let confirm = ConfirmOutcome::Passed(ConfirmPass {
            links: vec![ConfirmLink {
                new_id: 0,
                verdict: "reject".into(),
            }],
        });
        let novel = acc.apply_annotations(&pass, &pending, &confirm);
        assert_eq!(novel, 0, "a weak duplicate is never counted as novel");
        assert_eq!(
            acc.claims.len(),
            2,
            "the weak duplicate stays standalone — never merged"
        );
        assert_eq!(
            acc.weak.duplicates,
            vec![(1, 0)],
            "the suspected relation is recorded in the side structure"
        );
        assert!(
            acc.weak.contradictions.is_empty(),
            "only duplicate hints apply here"
        );
    }

    #[test]
    fn test_apply_annotations_weak_contradiction_keeps_notes_marks_unconfirmed() {
        // A weak contradiction keeps the bidirectional notes (both sides
        // qualify for the verification gate) but records the unconfirmed
        // relation in the side structure — never in the note text.
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
                contradiction: Some("price differs: $200 vs $100".into()),
            }],
        };
        let confirm = ConfirmOutcome::Passed(ConfirmPass {
            links: vec![ConfirmLink {
                new_id: 0,
                verdict: "reject".into(),
            }],
        });
        let novel = acc.apply_annotations(&pass, &pending, &confirm);
        assert_eq!(novel, 1, "a contradiction is new evidence even when weak");
        assert_eq!(
            acc.claims[0].contradictions,
            vec!["price differs: $200 vs $100"],
            "the existing claim keeps its contradiction note"
        );
        assert!(
            acc.claims[1]
                .contradictions
                .contains(&"alpha costs $100 in 2024".to_string()),
            "the new claim links back to the existing one"
        );
        assert_eq!(
            acc.weak.contradictions,
            vec![(1, 0)],
            "the unconfirmed relation lives in the side structure"
        );
        assert!(
            acc.claims
                .iter()
                .all(|c| c.contradictions.iter().all(|n| !n.contains("unconfirmed"))),
            "weakness never leaks into note text"
        );
    }

    #[test]
    fn test_verification_targets_primaries_first_weak_dups_fill_empty_slots() {
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(Claim {
            claim: "a".into(),
            source: "u1".into(),
            confidence: "low".into(),
            contradictions: vec![],
        });
        acc.claims.push(Claim {
            claim: "b".into(),
            source: "u2".into(),
            confidence: "high".into(),
            contradictions: vec!["b vs c".into()],
        });
        // Weak duplicate of claim 0 — appended after primaries.
        acc.claims.push(Claim {
            claim: "a restated".into(),
            source: "u3".into(),
            confidence: "high".into(),
            contradictions: vec![],
        });
        // Weak duplicate of claim 1 that is ALSO low confidence — already
        // primary, must not be double-appended.
        acc.claims.push(Claim {
            claim: "b restated".into(),
            source: "u4".into(),
            confidence: "low".into(),
            contradictions: vec![],
        });
        acc.weak.duplicates.push((2, 0));
        acc.weak.duplicates.push((3, 1));
        // A weak contradiction must NEVER be appended from the side structure
        // — it already carries notes and qualifies via the primary filter.
        acc.weak.contradictions.push((1, 2));
        let (targets, primary_count) = verification_targets(&acc);
        assert_eq!(
            primary_count, 3,
            "claims 0, 1 and 3 qualify as primary — the weak contradiction is not appended"
        );
        assert_eq!(targets.len(), 4, "claim 2 fills the only empty slot");
        assert_eq!(targets[0].claim, "a");
        assert_eq!(targets[1].claim, "b");
        assert_eq!(targets[2].claim, "b restated", "primary targets come first");
        assert_eq!(
            targets[3].claim, "a restated",
            "weak duplicate appended last"
        );
        assert_eq!(
            targets[3].contradictions, "",
            "weakness never leaks toward verifiers"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn test_annotate_round_confirm_failure_fail_open() {
        // End-to-end fail-open: the annotation pass succeeds, the confirm pass
        // fails entirely (transport) — every mutating verdict becomes weak
        // with the CONFIRM_FAILED marker; claims are never dropped, never
        // all-novel fallback.
        let _lock = crate::util::test::retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = crate::workspace::test_ws("/tmp/test_ws");
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(Claim {
            claim: "alpha is true".into(),
            source: "u1".into(),
            confidence: "medium".into(),
            contradictions: vec![],
        });
        let pending = vec![
            Claim {
                claim: "alpha is true (restated)".into(),
                source: "u2".into(),
                confidence: "high".into(),
                contradictions: vec![],
            },
            Claim {
                claim: "beta contradicts alpha".into(),
                source: "u3".into(),
                confidence: "high".into(),
                contradictions: vec![],
            },
        ];
        // Script: annotation pass OK (pending 0 = duplicate, pending 1 =
        // contradicts), then the confirm pass hits only transport failures.
        let annotation_json = r#"{"annotations": [{"new_id": 0, "verdict": "duplicate", "existing_id": 0}, {"new_id": 1, "verdict": "contradicts", "existing_id": 0, "contradiction": "alpha vs beta differ"}]}"#;
        let fake = crate::util::test::FakeProvider::new()
            .ok(annotation_json)
            .err(crate::retry::FailureClass::Transport, "confirm outage")
            .err(crate::retry::FailureClass::Transport, "confirm outage")
            .err(crate::retry::FailureClass::Transport, "confirm outage");
        let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
        let _provider_guard = crate::util::test::install_fake_provider(provider);
        let mut markers = Vec::new();
        let novel = annotate_round(&ws, &mut acc, &pending, &mut markers).await;
        assert_eq!(
            novel, 1,
            "only the weak contradiction counts as novel — the weak duplicate never does"
        );
        assert_eq!(
            acc.claims.len(),
            3,
            "both mutating claims stay standalone — never dropped"
        );
        assert_eq!(acc.weak.duplicates, vec![(1, 0)]);
        assert_eq!(acc.weak.contradictions, vec![(2, 0)]);
        assert!(
            acc.claims[0]
                .contradictions
                .contains(&"alpha vs beta differ".to_string()),
            "the weak contradiction keeps its note — verification still qualifies it"
        );
        assert!(
            markers.iter().any(|m| m.contains("confirmation failed")),
            "the confirm failure is never silent: {markers:?}"
        );
    }

    #[test]
    fn test_resolve_round_members_with_timeouts_preserves_timed_out() {
        // The TimedOut distinction survives until the wrap-up stage: only
        // deadline-aborted members are paired with their snapshots; panicked
        // and cancelled members collapse to NoResponse like before.
        let snapshots: Vec<WrapUpEntry> = (0..4)
            .map(|i| WrapUpEntry {
                agent_id: format!("a{i}"),
                params: wrap_up_params(&crate::workspace::test_ws("/tmp/test_ws"), "a", vec![]),
            })
            .collect();
        let members: Vec<RoundMember<AnalystRun<AnalystFindings>>> = vec![
            RoundMember::Done(AnalystRun::NoResponse),
            RoundMember::TimedOut,
            RoundMember::Panicked,
            RoundMember::Cancelled,
        ];
        let (runs, timed_out) = resolve_round_members_with_timeouts(members, &snapshots);
        assert_eq!(runs.len(), 4);
        assert!(runs.iter().all(|r| matches!(r, AnalystRun::NoResponse)));
        assert_eq!(timed_out.len(), 1, "only the TimedOut member is recovered");
        assert_eq!(timed_out[0].agent_id, "a1", "snapshot is index-parallel");
    }

    #[test]
    fn test_render_recovered_findings_section_is_separate_and_marked() {
        // The empty set renders nothing (boot path), and recovered findings
        // render in their own English-marked section — never merged into the
        // evidence list.
        assert_eq!(render_recovered_findings(&[]), "");
        let recovered = vec![AnalystFindings {
            claims: vec![Claim {
                claim: "found claim".into(),
                source: "u1".into(),
                confidence: "medium".into(),
                contradictions: vec!["counter".into()],
            }],
            unanswered: vec!["still open".into()],
        }];
        let out = render_recovered_findings(&recovered);
        assert!(
            out.contains("## Recovered from timed-out analysts"),
            "{out}"
        );
        assert!(out.contains("deadline exceeded, unverified"), "{out}");
        assert!(out.contains("found claim"), "{out}");
        assert!(out.contains("contradictions: counter"), "{out}");
        assert!(out.contains("still open"), "{out}");
    }

    #[test]
    fn test_session_has_successful_tool_result_gates_wrap_up_llm_call() {
        // The wrap-up LLM call is gated on at least one successful tool
        // result; all-failure and empty sessions skip it (their queries are
        // still registered by the caller).
        assert!(!session_has_successful_tool_result(&[]));
        assert!(!session_has_successful_tool_result(&[ChatMessage::user(
            "task"
        )]));
        let failed =
            crate::tools::format_tool_failure_feedback("search", &json!({"query": "q"}), "boom");
        assert!(!session_has_successful_tool_result(&[
            ChatMessage::tool_result("t1", &failed)
        ]));
        assert!(session_has_successful_tool_result(&[
            ChatMessage::tool_result("t1", "search results")
        ]));
        let mixed = vec![
            ChatMessage::tool_result("t1", &failed),
            ChatMessage::tool_result("t2", "results ok"),
        ];
        assert!(session_has_successful_tool_result(&mixed));
    }

    #[test]
    fn coder_round_lifecycle_skip_vs_claim_vs_unclaim() {
        let mut state = ResearchState::default();
        // Gate-skip: marked in the report, NOT claimed — the pre-loop key-0
        // round is re-attempted by boot-resume (post-progress keys are final:
        // the loop's round_index has advanced past them — fail-open per design).
        set_coder_marker(&mut state, 0, "skipped — analyst budget exhausted");
        assert!(
            state
                .markers
                .iter()
                .any(|m| m == "coder round 0 skipped — analyst budget exhausted")
        );
        assert!(!state.coder_rounds_done.contains(&0));
        // Dispatch: claimed; a stale skip marker (a previous gate-skip that
        // was later re-attempted) is cleared — the report shows one truth.
        claim_coder_round(&mut state, 0);
        assert!(state.coder_rounds_done.contains(&0));
        assert!(!state.markers.iter().any(|m| m.contains("coder round 0 ")));
        // Failure: un-claimed (only key-0 is re-attempted by resume); outcome
        // marked — and the outcome marker CLEARS a prior skip marker for the
        // same key (one truth: a failed-then-gate-skipped round must not
        // render both).
        unclaim_coder_round(&mut state, 0);
        set_coder_marker(&mut state, 0, "failed");
        assert!(!state.coder_rounds_done.contains(&0));
        assert!(state.markers.iter().any(|m| m == "coder round 0 failed"));
        assert!(
            !state
                .markers
                .iter()
                .any(|m| m.contains("coder round 0 skipped")),
            "outcome marker supersedes the stale skip marker"
        );
        // A later successful dispatch clears the stale outcome marker too.
        claim_coder_round(&mut state, 0);
        assert!(!state.markers.iter().any(|m| m.contains("coder round 0 ")));
        // Other rounds' markers are untouched.
        set_coder_marker(&mut state, 1, "skipped — round deadline expired");
        claim_coder_round(&mut state, 0);
        assert!(
            state
                .markers
                .iter()
                .any(|m| m == "coder round 1 skipped — round deadline expired")
        );
        assert!(!state.markers.iter().any(|m| m.contains("coder round 0 ")));
        // A gate-skip after a failure of the SAME key supersedes the failure
        // marker (one truth per key in the final report).
        set_coder_marker(&mut state, 1, "skipped — round deadline expired");
        assert_eq!(
            state
                .markers
                .iter()
                .filter(|m| m.starts_with("coder round 1 "))
                .count(),
            1,
            "skip re-push dedupes per key"
        );
    }

    #[test]
    fn captured_commands_are_capped_newest_win() {
        let mut commands = vec![
            "cat > /tmp/old_1.txt".to_string(),
            "cat > /tmp/old_2.txt".to_string(),
        ];
        cap_commands(&mut commands, 50);
        assert_eq!(commands.len(), 2, "under the cap — untouched");
        // Adding a third pushes the total over: the OLDEST command is dropped
        // (newest evidence wins) until the blob fits.
        commands.push("cat > /tmp/new.txt".to_string());
        cap_commands(&mut commands, 50);
        assert_eq!(
            commands,
            vec![
                "cat > /tmp/old_2.txt".to_string(),
                "cat > /tmp/new.txt".to_string()
            ]
        );
        // A single command larger than the cap is dropped ON ITS OWN (it could
        // not fit in the 32 KiB ticket body either) — the older fitting
        // commands survive (dropping them for it would lose all evidence).
        commands.push("cat > /tmp/huge.txt << 'EOF'\n".repeat(100));
        cap_commands(&mut commands, 50);
        assert_eq!(
            commands,
            vec![
                "cat > /tmp/old_2.txt".to_string(),
                "cat > /tmp/new.txt".to_string()
            ],
            "oversized newest dropped alone, older fitting commands kept"
        );
    }
}
