//! The deep-analysis arm of [`AnalyzeTool`](super::analyze::AnalyzeTool) (its `deep`
//! depth): the multi-round research orchestrator.
//!
//! Round 0 decomposes the question (two independent plans merged by id-based coverage), round 1
//! runs an analyst per sub-question (high-risk items get a decorrelated second angle), then
//! conditional gap rounds of shrinking width. A worker-in-loop prototype pass runs before the
//! first gap round and after every progress round that refreshes a non-empty gap list, gated by
//! budget, deadline, shutdown/cancel and `WORKER_MIN_REMAINING`; skips are fail-open markers.
//!
//! The plan is sized to what the run can carry, not to what the decomposers proposed: the merge is
//! told the plan's slot allowance, a plan that still comes out too wide is fitted by merging
//! neighbouring directions (never by dropping one), and round 1 dispatches only the prefix the
//! plan's allowance carries — never the reserve the rest of the run needs (one gap ladder plus the
//! verification pass). A plan wider than the allowance is therefore still worked instead of
//! refusing the round.
//!
//! Stopping is artifact-based (coverage completion, answerability abstention, a hard
//! analyst-spawn cap — never self-assessment) plus budget exhaustion, deadline expiry, shutdown
//! and manual cancel. One wall-clock deadline (`round_timeout`, default 3h) bounds every phase's
//! waits; the budget counts analysts spawned, not coordination calls, and is never refunded.
//! Exactly one envelope reaches the caller, and every run ends with an answer — a total
//! decomposition failure included: the synthesized one when a synthesis attempt came in clean and
//! within its size ceiling, otherwise one built from what the run did establish, naming the
//! questions it could not settle (the plan directions it never examined, at that granularity). The
//! answer is compact (10–20 KB) and produced at that size, never cut down afterwards; the product
//! appends the compact open-questions section and the reference to the detail file, so both are
//! part of the enforced ceiling. The full findings, sources, verification, worker-round output, any
//! undelivered synthesis and the run bookkeeping are delivered next to it in that detail file. The
//! wrap-up stage recovers deadline-aborted analysts by replaying their dispatch-time params (the
//! KV-cache prefix).

use crate::agent::message_router::{self, AgentJob, MessageKind};
use crate::agent::{chat_request, role_tool_specs, run_default_agent};
use crate::prompt::{load_prompt, substitute};
use crate::retry::FailureClass;
use crate::tools::analyze::{
    AnalystFindings, Claim, RoundMember, VerificationResult, VerificationTarget,
    await_round_members, dispatch_claim_verifiers, escape_fences, extract_query_telemetry,
    extract_query_telemetry_from_history, load_analyst_angles, max_confidence, normalize_claim,
    round_timeout,
};
use crate::tools::build_async_result_envelope;
use crate::{ChatMessage, ChatRequest, ChatRequestMeta, Role, ToolSpec, Workspace};
use anyhow::Result;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

/// Hard cap on research analysts spawned per run — decomposers, round-1
/// researchers, gap-round researchers and verification analysts all count;
/// coordination calls do not. Enforced at reservation time, never refunded.
const RESEARCH_MAX_ANALYSTS: usize = 30;
/// Round-0 decomposition fan-out (two independent plans).
const DECOMPOSE_FAN_OUT: usize = 2;
/// Gap-round dispatch widths — the ladder a plan must leave room for.
const GAP_ROUND_WIDTHS: &[usize] = &[4, 3, 2];
/// The answer's size target and ceiling. The answer is produced at this size, never cut down
/// afterwards: everything the run collected lives in the detail file next to it. The target is a
/// prompt instruction plus one re-ask when an answer comes in below it — a run with thin evidence
/// legitimately answers briefly, and padding it would contradict "built from what it established".
/// The ceiling is structural and covers the whole delivered answer: the text, the compact
/// open-questions section the product appends and the closing reference to the detail file are all
/// reserved out of it, so a long storage-root path can never push the answer past it. (The delivery
/// envelope wraps the answer in its own fixed framing — the tag and the job id — which is protocol,
/// not answer.)
const ANSWER_MIN_BYTES: usize = 10 * 1024;
const ANSWER_MAX_BYTES: usize = 20 * 1024;
/// How many claims / open questions the product-built answer names before deferring to the detail
/// file — the fallback answer is compact for the same reason the synthesized one is.
const ANSWER_LIST_CAP: usize = 25;
/// How many open questions the answer names before deferring the rest to the detail file. Tighter
/// than [`ANSWER_LIST_CAP`]: this section is appended to every answer, so its worst case is part of
/// the answer's fixed overhead.
const ANSWER_OPEN_CAP: usize = 10;
/// Byte cap on one product-built answer line: a single over-long finding must not crowd the rest
/// of the list out (and must never leave a section showing only a count).
const ANSWER_LINE_MAX_BYTES: usize = 400;
/// Explicit marker when the orchestrator cannot determine the remaining gaps.
const GAP_EXTRACTION_FAILED: &str = "gap extraction failed — remaining gaps unknown";
/// Explicit marker when the plan merge fails — the run falls back to all valid plans concatenated,
/// fitted to the allowance like any other plan (no decomposer's directions are lost).
const PLAN_MERGE_FAILED: &str =
    "plan merge failed — using all valid decomposition plans, fitted to the run's allowance";
/// Explicit marker when the annotation pass exhausts its retries — claims are novel.
const CLAIM_ANNOTATION_FAILED: &str = "claim annotation failed — all new claims treated as novel";
/// Explicit marker when the confirm pass fails — every mutating verdict is weak.
const CONFIRM_FAILED: &str =
    "annotation link confirmation failed — mutating links treated as weak/unconfirmed";
/// Explicit marker when a checkpoint lost the plan: nothing can be dispatched, so the run reports
/// what it established and names the question itself among what stayed open.
const PLAN_LOST_MARKER: &str = "the plan was lost from the checkpoint — the answer is built from the \
                                 evidence gathered so far";
/// Minimum remaining round time for a worker round to start (a worker run is
/// not interrupted — starting one with less left would eat the subsequent
/// gap rounds).
const WORKER_MIN_REMAINING: Duration = Duration::from_mins(30);
/// Default bound on the wrap-up stage (env `MAHBOT_WRAP_UP_TIMEOUT_SECS`, distinct from
/// the round deadline `MAHBOT_ROUND_TIMEOUT_SECS`), counted from its own start.
const DEFAULT_WRAP_UP_TIMEOUT_SECS: u64 = 5 * 60;

/// Resume pointer for a durable research run: the stage enum, plus implicit Done.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum ResearchStage {
    #[default]
    Decompose,
    Round1,
    GapRounds,
    /// Legacy checkpoint value from before the Verification collapse —
    /// synthesis runs before the verification pass, so old "verification"
    /// blobs resume at synthesis (a load error would silently restart).
    #[serde(alias = "verification")]
    Synthesis,
}

/// The research_jobs.state blob: ONE JSON column with what is needed to resume a run at
/// the checkpointed stage. RunStats is NOT stored (see RunStats pin) — the resumed
/// detail file's "Run Summary" undercounts the pre-crash segment and says so.
#[derive(Debug, Default, Serialize, Deserialize)]
struct ResearchState {
    stage: ResearchStage,
    plan: Option<MergedPlan>,
    gap_list: Option<GapList>,
    acc: AccumulatedEvidence,
    ledger: QueryLedger,
    markers: Vec<String>,
    gap_outcome: GapRoundsOutcome,
    budget_spent: usize,
    /// Gap-loop resume pointer / cumulative completed-round count: incremented pre-dispatch,
    /// persisted at every per-round checkpoint, restored on resume as the continuation key.
    /// Distinct from the per-invocation [`GapRoundsOutcome::rounds_dispatched`].
    round_index: usize,
    verification: Vec<VerificationResult>,
    /// Raw shell commands (UNFILTERED — no zone classification) from completed rounds'
    /// sessions, collected incrementally at dispatch so long runs don't lose early sessions
    /// to the transient TTL. NOT persisted (a ~10 MiB blob rewritten every round) — the
    /// run-folder dump is the durable artifact.
    #[serde(skip)]
    commands: Vec<String>,
    /// Seen-command set for O(1) dedup during collection (a `Vec::contains` scan per command
    /// would be O(n²)). Rebuilt from the dump file on boot resume; not persisted.
    #[serde(skip)]
    seen_commands: std::collections::HashSet<String>,
    /// Gap-loop round keys after which a worker round already ran (0 = the
    /// pre-loop worker, k = after gap round k). Boot resume never re-runs a
    /// completed worker round (no duplicate prototypes / LLM spend).
    #[serde(default)]
    worker_rounds_done: Vec<usize>,
    /// Indices of the plan directions round 1 examined — dispatched with an analyst response. The
    /// answer's open questions are derived from the plan against this list (a direction the run
    /// never examined is named, at that granularity). A gap round's narrower follow-up does not add
    /// a direction: it answers part of a direction, so the direction stays open. A checkpoint
    /// written before this list existed has it empty: every direction is then named open, because
    /// the run cannot prove one was examined and hiding an unexamined direction is the failure the
    /// answer must never make.
    #[serde(default)]
    examined: Vec<usize>,
    /// Worker-round reports ("what was tested in practice") — detail-file material, never part of
    /// the answer. Not in the blob (like `commands`): a worker response can be large and the blob is
    /// rewritten at every round checkpoint, so the run folder holds them and `load` re-seeds from
    /// there.
    #[serde(skip)]
    worker_reports: Vec<String>,
}

impl ResearchState {
    /// Load the persisted state for a job (or a fresh default).
    async fn load(job_id: &str) -> Self {
        let row = crate::session::store()
            .conn
            .query_optional(
                "SELECT state FROM research_jobs WHERE id = ?1",
                crate::db::params![job_id],
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
                let run_root = crate::research_cleanup::run_root_path(job_id);
                s.commands = crate::research_cleanup::read_command_dump(&run_root).await;
                s.seen_commands = s.commands.iter().cloned().collect();
                s.worker_reports = crate::research_cleanup::read_worker_reports(&run_root).await;
                s
            }
            Err(e) => {
                tracing::warn!(job = %job_id, error = %e, "Research state unreadable — fresh run");
                Self::default()
            }
        }
    }

    /// Checkpoint the state blob + jobs updated_at in ONE transaction (both rows live in the
    /// consolidated domain database), so a crash can't advance research_jobs.state without the jobs
    /// touch. retry_count is deliberately untouched — the boot scan's bump must survive
    /// checkpoints.
    async fn save(&self, job_id: &str) {
        let json = serde_json::to_string(self).unwrap_or_default();
        let now = crate::db::now();
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
                "UPDATE research_jobs SET state = ?1 WHERE id = ?2",
                crate::db::params![json, job_id],
            )
            .await?;
            tx.execute(
                "UPDATE jobs SET status = ?1, updated_at = ?2 WHERE id = ?3",
                crate::db::params![crate::jobs::RowStatus::Launched.as_str(), now, job_id],
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

    /// Collect the given agents' raw shell commands from their persisted sessions right after
    /// each round (a run longer than the session TTL loses early sessions, so collection must
    /// be incremental). UNFILTERED: the full history goes to the run-folder dump and the
    /// Sanitation cleanup agent does the attribution.
    async fn capture_round(&mut self, job_id: &str, agent_ids: &[String], run_root: &Path) {
        let fresh = crate::research_cleanup::collect_agent_shell_commands(agent_ids).await;
        for cmd in fresh {
            if self.seen_commands.insert(cmd.clone()) {
                self.commands.push(cmd);
            }
        }
        crate::research_cleanup::cap_command_dump(
            &mut self.commands,
            crate::research_cleanup::COMMAND_DUMP_CAP_BYTES,
        );
        // Rebuild the seen-set from the capped list: commands evicted by the
        // cap must be re-admittable when a later round re-issues them, and the
        // set stays bounded by the cap instead of growing without limit.
        self.seen_commands = self.commands.iter().cloned().collect();
        crate::research_cleanup::write_command_dump(run_root, job_id, &self.commands).await;
    }
}

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

/// A merged sub-question carrying provenance by global flat item id: the input-plan item it is
/// primarily built from, plus every other item folded into it — identical duplicates and closely
/// related directions alike. question/evidence_needed/risk are resolved by the system from the
/// cited items (text deduplicated, highest risk) — never trusted from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MergedSubQuestion {
    /// Global flat id of the input-plan item this sub-question is primarily built from
    /// (ids are assigned in (plan, item) order across all plans).
    from_id: usize,
    /// Global flat ids of the other items folded into this direction (duplicates and related ones).
    #[serde(default)]
    also_ids: Vec<usize>,
    #[serde(default)]
    question: String,
    #[serde(default)]
    evidence_needed: String,
    #[serde(default)]
    risk: String,
}

/// A round-0 plan item the merge explicitly dropped (never silently), by flat id — only ever a
/// genuine duplicate of a kept direction, never distinct scope dropped to fit.
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

/// A plan direction's analyst cost: two for a high-risk direction (it gets a decorrelated second
/// angle), one otherwise.
fn sub_question_slots(sq: &MergedSubQuestion) -> usize {
    1 + usize::from(sq.risk == "high")
}

/// The plan's total analyst cost.
fn plan_slot_cost(plan: &MergedPlan) -> usize {
    plan.sub_questions.iter().map(sub_question_slots).sum()
}

/// Slots the plan may never touch: the verification pass plus one full gap ladder. Sizing the plan
/// to what is left is what stops round 1 from spending the whole allowance at once and leaving the
/// run with no gaps worked and no verification.
fn plan_slot_reserve() -> usize {
    crate::tools::analyze::VERIFY_MAX_ANALYSTS + GAP_ROUND_WIDTHS.iter().sum::<usize>()
}

/// Slots the plan may spend: the cap minus what is already spent minus the reserve.
fn plan_slot_budget(budget: &ResearchBudget) -> usize {
    budget
        .cap
        .saturating_sub(budget.spent)
        .saturating_sub(plan_slot_reserve())
}

/// Fit an over-wide plan into `max_slots` by merging neighbouring directions — the operation the
/// merge model performs, applied deterministically: two adjacent directions become one wider
/// direction carrying both evidence needs and the higher risk label, so the split stays complete
/// and nothing is dropped for size. Each merge strictly lowers the plan's slot cost, so this
/// terminates; a plan of one direction cannot be narrowed further (round 1 then works what fits
/// and names the rest). The fallback path runs it too, so a failed merge cannot bypass sizing.
fn fit_plan_to_slots(plan: &mut MergedPlan, max_slots: usize) {
    while plan_slot_cost(plan) > max_slots && plan.sub_questions.len() > 1 {
        let mut fitted = Vec::with_capacity(plan.sub_questions.len().div_ceil(2));
        let mut items = std::mem::take(&mut plan.sub_questions).into_iter();
        while let Some(first) = items.next() {
            match items.next() {
                Some(second) => fitted.push(merge_directions(first, second)),
                None => fitted.push(first),
            }
        }
        plan.sub_questions = fitted;
    }
}

/// Append another direction's text to `question`/`evidence` (joined with `; `) and keep the higher
/// risk label — the ONE fold rule, shared by the merge's `also_ids` resolution and neighbour
/// fitting. Callers decide whether the text is already named; folding itself never loses a
/// direction.
fn fold_text(
    question: &mut String,
    evidence: &mut String,
    risk: &mut String,
    other_question: &str,
    other_evidence: &str,
    other_risk: &str,
) {
    question.push_str("; ");
    question.push_str(other_question.trim_end_matches('.'));
    evidence.push_str("; ");
    evidence.push_str(other_evidence.trim_end_matches('.'));
    *risk = max_confidence(risk, other_risk);
}

/// One direction from two: the first keeps its `from_id`, the second's ids join `also_ids`, so both
/// directions stay accounted for by the merged plan's provenance.
fn merge_directions(mut first: MergedSubQuestion, second: MergedSubQuestion) -> MergedSubQuestion {
    fold_text(
        &mut first.question,
        &mut first.evidence_needed,
        &mut first.risk,
        &second.question,
        &second.evidence_needed,
        &second.risk,
    );
    first.also_ids.push(second.from_id);
    first.also_ids.extend(second.also_ids);
    first
}

/// The plan directions round 1 dispatches and what they cost: the prefix `slots` of analyst
/// allowance carries. Round 1 dispatches exactly that prefix, so it never touches the reserve the
/// rest of the run needs. On a fresh run the prefix is the whole plan — the merge validator and
/// [`fit_plan_to_slots`] guarantee the plan fits; the prefix only cuts a checkpoint written by an
/// older build or a single direction too costly to narrow. Directions past it stay unexamined and
/// are named among the answer's open questions.
fn plan_prefix(costs: &[usize], slots: usize) -> (usize, usize) {
    let mut spent = 0usize;
    for (i, &cost) in costs.iter().enumerate() {
        if spent + cost > slots {
            return (i, spent);
        }
        spent += cost;
    }
    (costs.len(), spent)
}

/// Dispatch a deep analysis run — the `deep` depth of [`AnalyzeTool`]: decompose the
/// question, run analyst rounds against the remaining gaps, and deliver one compact answer with
/// the full detail next to it. Spawns the resumable background run and returns the dispatch ack;
/// exactly one answer envelope is delivered to the caller when the run completes.
pub(crate) fn dispatch_deep_analysis(ws: &Workspace, question: &str, caller_role: Role) -> String {
    // Read user context from task-locals BEFORE tokio::spawn so the single
    // result envelope carries the correct user identity. The job id is generated
    // here (before the spawn) so the immediate ack placeholder and the delivered
    // envelope reference one identical id.
    let ws = ws.clone();
    let question = question.to_string();
    let user_name = crate::agent::tool_user_name();
    let channel = crate::agent::tool_channel();
    // Read the caller's launch synchronously before the spawn (task-local contract): the
    // round's envelope and job row must carry the custom branch for a custom role.
    let custom = crate::agent::tool_custom_launch();
    let job_id = crate::generate_id();
    let ack = format!(
        "Deep analysis dispatched (job {job_id}). A compact answer will arrive when the run \
         completes, with the full detail delivered alongside it."
    );

    tokio::spawn(async move {
        // Catch panics so the caller ALWAYS receives the single result envelope
        // — a panic in the dispatch task would otherwise leave the caller waiting
        // forever on a result that can never arrive.
        let run = std::panic::AssertUnwindSafe(async {
            dispatch_durable_research(
                &ws,
                &question,
                caller_role,
                user_name.clone(),
                channel.clone(),
                &job_id,
                custom.clone(),
            )
            .await
        })
        .catch_unwind()
        .await;
        let envelope = match run {
            Ok(Some(envelope)) => envelope,
            Ok(None) => {
                // Shutdown/drain abort OR manual cancel: no envelope is routed now. An abort keeps
                // the run alive for the next boot's resume (result and artifacts arrive at the real
                // terminalization); a manual cancel stops it permanently (the cancelled-run handoff
                // dispatches the cleanup agent — nothing is delivered).
                tracing::info!(
                    "Deep analysis run ended without delivery (aborted or manually cancelled)"
                );
                return;
            }
            Err(panic) => {
                let panic = crate::util::panic_message(&*panic);
                tracing::error!(panic = %panic, "deep analysis dispatch panicked");
                AgentJob {
                    content: build_async_research_message(
                        &job_id,
                        &Err(anyhow::anyhow!("deep analysis dispatch panicked: {panic}")),
                    ),
                    workspace_name: ws.name.clone(),
                    user_name,
                    channel,
                    kind: MessageKind::ResearchResult,
                    role: caller_role.for_launch(custom.as_ref()),
                    reply_target: None,
                    pending_job_id: Some(job_id),
                    originating_workspace: None,
                    custom_role: custom,
                    reply_to_custom: None,
                    urgent: false,
                }
            }
        };

        // Route exactly one final envelope — the completion helper's own copy, so persisted and
        // routed copies can never drift. A drain may drop the routed copy; the pending row
        // survives for boot replay (at-least-once).
        message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
    });

    ack
}

/// Durable dispatch: SPAWN the job (one tx) → run the resumable orchestrator → COMPLETE (one tx
/// — INSERT pending envelope + DELETE jobs row). `Some(envelope)` on real terminalizations (the
/// completion helper's own copy, routed as-is), `None` on a shutdown/drain abort (run alive for
/// boot resume) or a manual cancel. `job_id` is caller-generated BEFORE the spawn.
async fn dispatch_durable_research(
    ws: &Workspace,
    question: &str,
    caller_role: Role,
    user_name: String,
    channel: String,
    job_id: &str,
    custom: Option<crate::Launch>,
) -> Option<AgentJob> {
    // The run's cancel signal lives for the whole invocation — fresh dispatch and its
    // terminalization tail. A manual cancel fires it; the orchestrator's boundary gates stop
    // permanently (ResearchExit::Cancelled).
    let _cancel_guard = crate::round_cancel::register(job_id, None);
    // A round dispatched by a custom role owns a custom branch address — its jobs row and
    // envelope must say `Role::Custom` plus the launch, not the construction role.
    let caller_role = caller_role.for_launch(custom.as_ref());
    let spawn = async {
        // SPAWN: one tx — jobs + research_jobs child row (the shared in-tx
        // child pattern; a crash mid-spawn leaves either all or none).
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            job_id,
            question,
            &ws.name,
            &user_name,
            &channel,
            caller_role,
            &[],
            &crate::jobs::SpawnChild::Research,
            None,
            None,
            custom.as_ref(),
        )
        .await
    };
    // Spawn failures still route an error envelope — never a silent drop.
    let spawn_out = spawn.await;
    let spawned = spawn_out.is_ok();
    let exit = match spawn_out {
        Ok(()) => run_deep_research(ws, question, job_id, false).await,
        Err(e) => ResearchExit::Terminal(Err(e)),
    };
    let result = match exit {
        // Abort: the run stays alive for the next boot's resume — the jobs row survives as
        // 'launched' and boot-recovery re-enters the checkpointed run, so result and artifacts
        // arrive at the real terminalization (design pin: a shutdown/drain abort is not
        // terminalization). Nothing is written or routed here.
        ResearchExit::Aborted => {
            tracing::info!(
                job = %job_id,
                "Research run aborted by shutdown/drain — resumes at next boot"
            );
            return None;
        }
        // Manual cancel: a PERMANENT stop — the cancelled-run handoff
        // dispatches the cleanup agent with the run's dump intent; the folder
        // is released only by the cleanup tail. Nothing is delivered.
        ResearchExit::Cancelled => {
            tracing::info!(
                job = %job_id,
                "Research run manually cancelled — permanent stop, nothing delivered"
            );
            hand_off_cancelled_or_warn(job_id, ws, question).await;
            return None;
        }
        ResearchExit::Terminal(result) => result,
    };
    terminalize_research(
        job_id,
        ws,
        &result,
        caller_role,
        question,
        &user_name,
        &channel,
        spawned,
        custom,
    )
    .await
}

/// The two artifacts one terminalization delivers: the compact answer (10–20 KB) and the full
/// detail written next to it. Nothing the run produced is dropped — it moves to `detail`.
struct ResearchOutput {
    answer: String,
    detail: String,
}

/// Write the run's artifacts: the detail file, the answer and the command dump. Returns the answer
/// as archived and delivered — with its closing reference to the detail file only when that file was
/// actually written, so the answer never points at a file that does not exist.
async fn write_terminalization_artifacts(
    job_id: &str,
    answer: &str,
    detail: &str,
    state: &ResearchState,
) -> String {
    let answer = match crate::research_cleanup::write_detail_md(job_id, detail).await {
        Some(detail_path) => format!("{answer}{}", detail_reference(&detail_path)),
        None => answer.to_string(),
    };
    crate::research_cleanup::write_results_md(job_id, &answer).await;
    let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
    crate::research_cleanup::write_command_dump(&run_root, job_id, &state.commands).await;
    answer
}

/// Real-terminalization tail shared by fresh dispatch and boot resume: artifacts written BEFORE
/// the exactly-once boundary, then durable completion, then cleanup dispatch (the cleanup row
/// reuses id == run_id, so the DELETE must free it first). The Sanitation cleanup is NOT dispatched
/// from [`write_terminalization_artifacts`] for exactly that reason. `Some` on a real
/// terminalization, `None` on a manual cancel (the cancelled-run handoff dispatches the cleanup
/// agent).
#[expect(clippy::too_many_arguments)]
async fn terminalize_research(
    job_id: &str,
    ws: &Workspace,
    result: &anyhow::Result<ResearchOutput>,
    caller_role: Role,
    question: &str,
    user_name: &str,
    channel: &str,
    spawned: bool,
    custom: Option<crate::Launch>,
) -> Option<AgentJob> {
    // Manual-cancel gate BEFORE any artifact is written: a cancel landing after the
    // orchestrator's last boundary check must not write artifacts, complete the job, dispatch
    // cleanup, or route anything. The cancelled-run handoff dispatches cleanup with the
    // already-final dump (no racing folder deletion).
    if crate::round_cancel::is_cancelled(job_id) {
        tracing::info!(job = %job_id, "Research terminalization suppressed by manual cancel");
        hand_off_cancelled_or_warn(job_id, ws, question).await;
        return None;
    }
    let delivered = match result {
        Ok(out) => {
            // Terminalization artifacts BEFORE the exactly-once boundary, only for runs that
            // started. The aborted flag (not the global shutdown state) already gated aborts
            // above: a shutdown after a fully successful run must not skip the artifacts.
            let answer = if spawned {
                let state = ResearchState::load(job_id).await;
                write_terminalization_artifacts(job_id, &out.answer, &out.detail, &state).await
            } else {
                out.answer.clone()
            };
            build_async_research_message(job_id, &Ok(answer))
        }
        // No run output (a failed spawn): the error envelope is the archive's content.
        Err(e) => {
            let delivered = build_async_research_message(job_id, &Err(anyhow::anyhow!("{e:#}")));
            if spawned {
                let state = ResearchState::load(job_id).await;
                write_terminalization_artifacts(job_id, &delivered, "", &state).await;
            }
            delivered
        }
    };
    // Complete BEFORE the cleanup dispatch (the cleanup row reuses id == run_id; the DELETE
    // must free it first) and BEFORE the route: a crash in between leaves the envelope
    // pending, and boot replay recreates the cleanup row while the folder still exists.
    // UNCONDITIONAL — a failed spawn relies on this completion for its error envelope.
    let envelope = crate::jobs::complete_durable_job(
        job_id,
        delivered,
        MessageKind::ResearchResult,
        caller_role,
        user_name,
        channel,
        &ws.name,
        custom,
    )
    .await;
    // Defensive post-completion gate: the in-tx cancel gate rolled the completion back when
    // the cancel fired mid-tx — never route the answer or dispatch the cleanup.
    if crate::round_cancel::is_cancelled(job_id) {
        tracing::info!(job = %job_id, "Research completion suppressed by manual cancel — not routed");
        hand_off_cancelled_or_warn(job_id, ws, question).await;
        return None;
    }
    // Cleanup dispatch AFTER the exactly-once boundary: the cleanup row reuses id == run_id as
    // the marker that holds the folder until it completes (a wrongly-ordered row would be
    // deleted by the completion DELETE).
    if spawned
        && let Err(e) =
            crate::research_cleanup::dispatch_research_cleanup(job_id, question, ws).await
    {
        tracing::warn!(
            job = %job_id,
            error = %e,
            "Research cleanup dispatch failed — run folder left for the periodic temp cleaner"
        );
    }
    Some(envelope)
}

/// Dispatches the cancelled-run handoff, warning on failure — rows stay for
/// boot resume (the documented failure-mode contract). Callers keep their own
/// site-specific `tracing::info!` and early exit.
async fn hand_off_cancelled_or_warn(job_id: &str, ws: &Workspace, question: &str) {
    if let Err(e) = crate::research_cancel::hand_off_cancelled_run(job_id, ws, question).await {
        tracing::warn!(job = %job_id, error = %e, "cancelled-run cleanup handoff failed — rows stay for boot resume");
    }
}

/// Boot resume: re-enter the orchestrator at the checkpointed stage (retry_count bumped by
/// the boot scan), then terminalize exactly like a fresh dispatch. Aborts quietly on
/// shutdown/drain — the checkpointed state is reused by the next boot; routing a partial
/// result here would race the exit.
pub(crate) async fn resume_research_run(job_id: &str, ws: &Workspace) {
    // Register the run's cancel signal for this invocation — the boot-resume
    // path can be cancelled exactly like a fresh dispatch.
    let _cancel_guard = crate::round_cancel::register(job_id, None);
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
    let custom = caller.custom_launch.as_deref().cloned();
    let result = match run_deep_research(ws, &caller.task, job_id, true).await {
        ResearchExit::Aborted => {
            tracing::info!(
                job = %job_id,
                "Research resume aborted after run — job stays for next boot",
            );
            return;
        }
        ResearchExit::Cancelled => {
            tracing::info!(
                job = %job_id,
                "Research resume manually cancelled — permanent stop, nothing delivered",
            );
            hand_off_cancelled_or_warn(job_id, ws, &caller.task).await;
            return;
        }
        ResearchExit::Terminal(result) => result,
    };
    // Terminalize exactly like a fresh dispatch, then route here — the
    // resume path has no execute() wrapper to route for it.
    let Some(envelope) = terminalize_research(
        job_id,
        ws,
        &result,
        caller_role,
        &caller.task,
        &caller.user_name,
        &caller.channel,
        true,
        custom,
    )
    .await
    else {
        return;
    };
    crate::agent::message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
}

/// Build the `<analyze-tool-result>` envelope — the tag both analysis depths use. The
/// caller-generated `job_id` is embedded so the delivered answer correlates with the
/// immediate ack; failures are wrapped with an explicit marker, never silently dropped.
fn build_async_research_message(job_id: &str, answer: &anyhow::Result<String>) -> String {
    build_async_result_envelope(job_id, answer, "analyze-tool-result", None)
}

/// Evidence collected in one research round (post-ledger-dedup).
#[derive(Debug, Default)]
struct EvidenceRound {
    /// Unique source URLs seen this round.
    urls: Vec<String>,
    claims: Vec<Claim>,
    /// Analysts' self-reported uncovered aspects (from `AnalystFindings`).
    unanswered: Vec<String>,
    /// Total queries issued by the round's analysts; `repeat_queries` of
    /// them were repeats of earlier rounds' queries (no-progress signal).
    queries: usize,
    repeat_queries: usize,
    /// Raw responses of analysts whose extraction failed — never silently lost.
    raw_reports: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AccumulatedEvidence {
    urls: HashSet<String>,
    /// Claims accumulated across rounds; a claim's stable id is its index in
    /// this vec (claims are only appended or merged in place, never removed).
    claims: Vec<Claim>,
    /// Deduplicated analysts' self-reported unanswered aspects.
    unanswered: Vec<String>,
    /// Rebuilt from `unanswered` after deserialize (normalize_claim keys) — not stored.
    #[serde(skip)]
    unanswered_keys: HashSet<String>,
    /// Raw responses of analysts whose extraction failed — never silently lost.
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
                let key = crate::tools::analyze::normalize_claim(u);
                (!key.is_empty()).then_some(key)
            })
            .collect();
    }
}

/// Research-local weak/unconfirmed annotation links — the "keep weak, clarify later" structure.
/// Claim ids are stable indices into [`AccumulatedEvidence::claims`], so hints never go stale.
/// Weakness lives ONLY here and is never written back: resolution is the verification gate.
#[derive(Debug, Default, Serialize, Deserialize)]
struct WeakLinks {
    /// Weak duplicate hints: standalone claim id → suspected duplicate id, recorded even when
    /// the confirm model rejected the link — fail-open, a relation is never dropped.
    duplicates: Vec<(usize, usize)>,
    /// Weak contradiction hints: new claim id → existing claim id. Both sides still carry
    /// their contradiction notes; the unconfirmed relation is recorded here.
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
    /// 0-based index of the NEW claim within this round's pending claims. Pair identity is
    /// pinned by the annotation pass — the model never re-transcribes the existing id (the
    /// transcription-error class this pass exists to catch).
    new_id: usize,
    /// "confirm" | "reject" — uncertainty maps to reject (weak/unconfirmed).
    verdict: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmPass {
    links: Vec<ConfirmLink>,
}

/// Outcome of the optional confirm pass: per-pair verdicts, or the whole call failed
/// (every mutating verdict is weak). Typed so the two states can never be conflated.
enum ConfirmOutcome {
    Passed(ConfirmPass),
    Failed,
}

impl AccumulatedEvidence {
    /// Absorb a round's evidence, returning `(novel_urls, pending_claims)`. URLs and
    /// unanswered aspects dedup exactly as before; claims become a pending list — novelty is
    /// decided by the per-round LLM annotation pass, not embedding similarity.
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

    /// Apply an annotation pass over a round's pending claims: novel claims are appended, confirmed
    /// duplicates merge (sources joined, confidence upgraded, contradictions appended — never
    /// dropped), contradicting claims are appended AND linked so verification targets both sides.
    /// Weak mutating verdicts never merge and do not count as novel. Returns the number of novel
    /// claims.
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

/// Cross-agent query ledger: later rounds get it in their task prompts ("do not
/// repeat these"), because concurrent round-1 analysts cannot see each other's
/// queries — pre-dispatch suppression is only feasible across rounds. Repeats are
/// tallied per round ([`EvidenceRound::repeat_queries`]) as a no-progress signal.
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

#[derive(Debug, Default)]
struct RunStats {
    tool_calls: usize,
    searches: usize,
    /// Queries already issued earlier in the run (all phases dedupe against one ledger)
    /// — summary telemetry only; saturation lives in `EvidenceRound::repeat_queries`.
    repeat_queries: usize,
    /// Analysts that failed (no response, empty output, or extraction failure).
    failed_analysts: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct GapRoundsOutcome {
    abstention: Option<String>,
    unresolved: Vec<String>,
    rounds_dispatched: usize,
    /// Set when the remaining gaps could not be determined — the run's bookkeeping must carry an
    /// explicit marker instead of looking like coverage completion.
    incomplete: Option<String>,
}

/// One deadline-aborted analyst's dispatch-time snapshot: the agent_id plus the frozen
/// chat params (model, tools, reasoning_effort, routing, max_tokens) captured BEFORE
/// spawn so the wrap-up replays the analyst's KV-cache prefix. Config/daemon-state
/// drift after the snapshot is not reflected (fail-open: a miss re-encodes the tail).
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

/// Build Analyst chat params carrying `purpose`/`agent_id` metadata and optional tool
/// specs; byte-relevant fields come from the shared [`chat_request`] helper (model,
/// reasoning_effort, routing, max_tokens).
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
        ..chat_request(Role::Analyst, tool_specs, Vec::new())
    }
}

/// Wrap-up params: the tool specs frozen at dispatch. The dedicated purpose tag
/// separates wrap-up calls in the request journal.
fn wrap_up_params(ws: &Workspace, agent_id: &str, tool_specs: Vec<ToolSpec>) -> ChatRequest {
    research_params(
        ws,
        "research_wrap_up",
        agent_id.to_string(),
        Some(tool_specs),
    )
}

/// RAII guard for a research-run LLM call: registers with
/// [`crate::agent::registry::NON_AGENT_CALLS`] under the run's `ParentKey::Research`
/// group. Dropping it removes the entry, so the caller must keep the binding alive.
fn research_call_guard(
    kind: &'static str,
    ws_name: &str,
    run_key: &str,
    run_lifetime: bool,
    question: &str,
) -> crate::agent::registry::NonAgentCallGuard {
    crate::agent::registry::NON_AGENT_CALLS.register(
        kind,
        ws_name,
        Some(crate::agent::registry::ParentKey::Research(
            run_key.to_string(),
        )),
        run_lifetime,
        Some(question.to_string()),
    )
}

/// Wrap-up stage after a round deadline: for every deadline-aborted analyst, load its session,
/// register its queries in the ledger (BEFORE any LLM call), and — only when the session shows a
/// successful tool result — extract the accumulated findings by replaying the dispatch-time params.
/// Fail-open: a failed extraction keeps the analyst failed. Skipped when a drain is in progress
/// (the orchestrator guard blocks the drain-watch, so the stage self-aborts). One NON_AGENT_CALLS
/// row per stage.
async fn wrap_up_timed_out(
    ws: &Workspace,
    timed_out: Vec<WrapUpEntry>,
    ledger: &mut QueryLedger,
    run_stats: &mut RunStats,
    run_key: &str,
    question: &str,
) -> Vec<AnalystFindings> {
    if timed_out.is_empty() || crate::shutdown::aborting() {
        return Vec::new();
    }
    let stage_deadline = std::time::Instant::now() + wrap_up_timeout();
    // Prepare sequentially (cheap DB reads): load the session, register its queries (ledger +
    // summary telemetry only, never the saturation counters), and gate the LLM call on a successful
    // tool result. Prep always completes, so registration is guaranteed for every timed-out
    // analyst.
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
    let _wrap_up_call = research_call_guard("research_wrap_up", &ws.name, run_key, false, question);
    let wrap_up_prompt = load_prompt("research/wrap_up.md");
    let handles: Vec<_> = prepared
        .into_iter()
        .map(|p| {
            let wrap_up_prompt = wrap_up_prompt.clone();
            tokio::spawn(async move {
                let mut messages = p.history;
                messages.push(ChatMessage::user(&wrap_up_prompt));
                crate::agent::extraction::retry_extract_structured_scoped::<AnalystFindings>(
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

/// True when the session history holds a persisted tool result that does not start with the
/// tool-failure marker (all-failure or no-result sessions skip the wrap-up call). A success
/// starting with the marker is misclassified — accepted. Non-native results decode to None.
fn session_has_successful_tool_result(history: &[ChatMessage]) -> bool {
    history.iter().any(|m| {
        matches!(
            crate::session::decode_native_history_message(m),
            Some(crate::session::DecodedNativeHistoryMessage::ToolResult { content, .. })
                if !content.starts_with(crate::tools::TOOL_FAILURE_MARKER)
        )
    })
}

/// Await the concurrent wrap-up extraction tasks, bounded by the stage deadline and interrupted by
/// the drain flag. Deliberately not shared with [`await_round_members`]: round waits must NOT abort
/// on drain, while the wrap-up MUST.
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
    // The drain/deadline arms fire once each, then the cancel token aborts every
    // remaining task (a permanently-ready arm would starve `pending.next()`).
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

/// Collapse awaited round members into analyst runs, splitting out the deadline-aborted
/// members with their dispatch-time snapshots. Panicked/cancelled members stay plain
/// NoResponse — the TimedOut distinction survives until the wrap-up stage.
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

/// Dispatch a round of analyst members and resolve it: staggered spawn, deadline-bounded
/// await, timeout resolution against the dispatch-time wrap-up snapshots.
async fn dispatch_analyst_round<T, Fut, F>(
    members: Vec<F>,
    resume: bool,
    deadline: std::time::Instant,
    snapshots: &[WrapUpEntry],
) -> (Vec<AnalystRun<T>>, Vec<WrapUpEntry>)
where
    T: serde::de::DeserializeOwned + Send + 'static,
    Fut: std::future::Future<Output = AnalystRun<T>> + Send + 'static,
    F: FnOnce(crate::agent::RoundOpts) -> Fut + Send + 'static,
{
    let handles = crate::agent::spawn_staggered_round(members, resume).await;
    let members_out = await_round_members(handles, deadline).await;
    resolve_round_members_with_timeouts(members_out, snapshots)
}

/// One analyst run. Mirrors analyze's three-state fail-open typing: a parse-failed
/// analyst's raw response is preserved, never dropped.
enum AnalystRun<T> {
    /// Agent produced no response (crashed, cancelled, empty output).
    NoResponse,
    /// Agent responded and structured extraction succeeded.
    Findings(AnalystRunOutcome<T>),
    /// Agent responded but structured extraction failed; the raw response is preserved for
    /// the fail-open path, plus telemetry (queries must reach the ledger so later rounds
    /// do not re-ask them).
    ParseFailed {
        raw: String,
        tool_calls: usize,
        searches: usize,
        queries: Vec<String>,
    },
}

struct AnalystRunOutcome<T> {
    value: T,
    tool_calls: usize,
    searches: usize,
    queries: Vec<String>,
}

/// Run a single analyst agent on `task` and extract structured output `T` while the agent is alive
/// (KV-cache reuse). Telemetry from the session history is folded into the outcome. `run_key`
/// places the sub-agent in the Running Agents view; `question` is the group header label.
async fn run_structured_analyst<T: serde::de::DeserializeOwned>(
    ws: &Workspace,
    agent_id: &str,
    task: &str,
    extraction_prompt: &str,
    round: crate::agent::RoundOpts,
    run_key: &str,
    question: &str,
) -> AnalystRun<T> {
    let (agent, response) = run_default_agent(
        agent_id,
        Role::Analyst,
        ws,
        task,
        false,
        Some(round),
        Some(crate::agent::registry::ParentKey::Research(
            run_key.to_string(),
        )),
        Some(question.to_string()),
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

/// Build a round member closure: run one analyst on `task` and extract `T`. Takes owned
/// values so the returned closure (and its boxed future) are `'static` + `Send`.
fn make_round_member<T: serde::de::DeserializeOwned + Send>(
    ws: Workspace,
    agent_id: String,
    task: String,
    extraction_prompt: String,
    run_key: String,
    question: String,
) -> impl FnOnce(crate::agent::RoundOpts) -> futures_util::future::BoxFuture<'static, AnalystRun<T>> + Send
{
    move |round| {
        Box::pin(async move {
            run_structured_analyst::<T>(
                &ws,
                &agent_id,
                &task,
                &extraction_prompt,
                round,
                &run_key,
                &question,
            )
            .await
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
                // Failed analysts' queries still reach the ledger (later rounds must not re-ask
                // them) and the summary telemetry; they do not drive the per-round saturation
                // signal (`round.queries` stays from successful analysts only).
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

/// Orchestrator chat params: cheap Analyst model (per-role overrides respected),
/// constant model/effort/tools across a run's coordination calls (KV-cache friendly —
/// the leading workspace-context system message is constant too).
fn orchestrator_params(ws: &Workspace, purpose: &'static str) -> ChatRequest {
    research_params(
        ws,
        purpose,
        format!("research_{}_orchestrator", ws.name),
        None,
    )
}

/// Structured orchestrator extraction via the hardened scoped retry loop; `prompt`
/// embeds the JSON schema request and the workspace context is prepended as the leading
/// system message. `run_key`/`question` only place the call in the Running Agents view.
async fn orchestrator_extract<T: serde::de::DeserializeOwned>(
    ws: &Workspace,
    purpose: &'static str,
    prompt: &str,
    validate: Option<&crate::ExtractionValidator<T>>,
    run_key: &str,
    question: &str,
) -> Result<T> {
    let _call = research_call_guard(purpose, &ws.name, run_key, false, question);
    let params = orchestrator_params(ws, purpose);
    let mut messages = Vec::with_capacity(2);
    crate::prompt::prepend_general_context(&mut messages, ws).await;
    messages.push(ChatMessage::user(prompt));
    crate::agent::extraction::retry_extract_structured_scoped::<T>(
        &messages, "", &params, validate, None,
    )
    .await
    .map_err(|e| anyhow::anyhow!("orchestrator extraction '{purpose}' failed: {e}"))
}

/// Round 0: two independent decomposition plans merged into one consolidated plan with provenance
/// via one orchestrator call (budget: 2 decomposers; the x2 redundancy lives at the steering
/// decision). The plan is sized to the run's allowance — the decomposers and the merge are told it,
/// and a plan that still comes out too wide is fitted by merging directions. On merge failure the
/// run falls back to every valid plan concatenated, fitted the same way, with a marker; only when
/// ALL decomposers failed does this return an error, and the caller turns that into an answer
/// reporting that nothing was established.
///
/// A parse-failed decomposer's raw is intentionally NOT preserved (unlike `collect_evidence`) —
/// plans are steering, not evidence.
#[expect(clippy::too_many_arguments)]
async fn round0_decompose(
    ws: &Workspace,
    question: &str,
    budget: &mut ResearchBudget,
    run_stats: &mut RunStats,
    deadline: std::time::Instant,
    resume: bool,
    run_root: &str,
    captured: &mut Vec<String>,
    run_key: &str,
) -> Result<(MergedPlan, Option<String>)> {
    budget
        .try_reserve(DECOMPOSE_FAN_OUT)
        .map_err(anyhow::Error::msg)?;
    let max_slots = plan_slot_budget(budget);
    let max_slots_arg = max_slots.to_string();
    let task_template = load_prompt("research/decompose.md");
    let extraction_prompt = load_prompt("extraction/decompose.md");
    let members: Vec<_> = (0..DECOMPOSE_FAN_OUT)
        .map(|i| {
            let ws = ws.clone();
            let question = question.to_string();
            let task = substitute(
                &task_template,
                &[
                    ("{{question}}", &question),
                    ("{{run_root}}", run_root),
                    ("{{max_slots}}", &max_slots_arg),
                ],
            );
            let agent_id = crate::session::research_agent_id(&ws.name, &format!("decompose_{i}"));
            captured.push(agent_id.clone());
            make_round_member::<DecompositionPlan>(
                ws,
                agent_id,
                task,
                extraction_prompt.clone(),
                run_key.to_string(),
                question.clone(),
            )
        })
        .collect();
    let plans: Vec<_> = dispatch_analyst_round(members, resume, deadline, &[])
        .await
        .0;
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
    if let Ok(mut plan) = merge_decomposition_plans(ws, question, &valid, max_slots, run_key).await
    {
        resolve_merged_plan_ids(&mut plan, &valid);
        // The validator already rejected an over-wide plan; fitting is the deterministic backstop
        // (and the only sizing on the fallback path below).
        fit_plan_to_slots(&mut plan, max_slots);
        Ok((plan, None))
    } else {
        // Fail-open: a failed merge must never lose the run — every valid plan's directions are
        // concatenated (deduplicated) and fitted to the allowance exactly like a merged plan (the
        // fallback may not bypass sizing, and may not lose a decomposer's scope).
        let mut plan = concatenated_plan(&valid);
        // A plan with no directions leaves nothing to research or name — the same dead end as no
        // plan at all, so it errors instead of delivering a content-free answer.
        if plan.sub_questions.is_empty() {
            anyhow::bail!("all decomposition plans were empty — no research plan produced");
        }
        fit_plan_to_slots(&mut plan, max_slots);
        Ok((plan, Some(PLAN_MERGE_FAILED.to_string())))
    }
}

/// Every valid decomposition plan's directions in one plan, exact duplicates dropped. The
/// merge-failure fallback — it keeps every decomposer's scope instead of the first plan's only.
/// `from_id` stays the input item's global flat id, matching the merge's provenance numbering.
fn concatenated_plan(plans: &[DecompositionPlan]) -> MergedPlan {
    let mut seen: Vec<String> = Vec::new();
    let mut sub_questions = Vec::new();
    for (id, sq) in plans
        .iter()
        .flat_map(|p| p.sub_questions.iter())
        .enumerate()
    {
        let key = normalize_claim(sq.question.trim_end_matches('.'));
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        sub_questions.push(MergedSubQuestion {
            question: sq.question.clone(),
            evidence_needed: sq.evidence_needed.clone(),
            risk: sq.risk.clone(),
            from_id: id,
            also_ids: Vec::new(),
        });
    }
    MergedPlan {
        sub_questions,
        dropped: Vec::new(),
    }
}

/// Render the input plans with their global flat item ids for the merge prompt, ids
/// numbered flat across all plans in (plan, item) order.
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

/// Merge 1–2 independent plans into one consolidated plan with provenance (not budgeted), sized to
/// `max_slots` analyst slots — the merge decides the plan's width, so it is told the allowance
/// instead of being left to cover everything either plan named.
async fn merge_decomposition_plans(
    ws: &Workspace,
    question: &str,
    plans: &[DecompositionPlan],
    max_slots: usize,
    run_key: &str,
) -> Result<MergedPlan> {
    let max_slots_arg = max_slots.to_string();
    let prompt = substitute(
        &load_prompt("research/decompose_merge.md"),
        &[
            ("{{question}}", question),
            ("{{plans}}", &render_plans_with_ids(plans)),
            ("{{max_slots}}", &max_slots_arg),
        ],
    );
    // The validator captures owned copies (the validator type is `'static`).
    let plans_owned = plans.to_vec();
    orchestrator_extract::<MergedPlan>(
        ws,
        "decompose_merge",
        &prompt,
        Some(&move |p| validate_merged_plan(p, &plans_owned, max_slots)),
        run_key,
        question,
    )
    .await
}

/// The input-plan items as the global flat id universe — id = (plan, item) order,
/// matching `render_plans_with_ids` and the merge prompt's numbering.
fn plan_item_table(plans: &[DecompositionPlan]) -> Vec<Vec<String>> {
    plans
        .iter()
        .map(|p| p.sub_questions.iter().map(|s| s.question.clone()).collect())
        .collect()
}

/// Fill in merged entries' tuple text from the cited input-plan items — the model's copied text is
/// never used. An entry citing several non-identical items is a fold: its direction keeps every
/// cited item's question and evidence need (deduplicated) and the highest risk label, so folding
/// related directions loses none of them.
fn resolve_merged_plan_ids(plan: &mut MergedPlan, plans: &[DecompositionPlan]) {
    let items = plan_item_table(plans);
    let table = crate::consensus::ItemTable::new(&items);
    for sq in &mut plan.sub_questions {
        let Some((p, i)) = table.resolve_index(sq.from_id) else {
            continue;
        };
        let first = &plans[p].sub_questions[i];
        let mut question = first.question.trim_end_matches('.').to_string();
        let mut evidence = first.evidence_needed.trim_end_matches('.').to_string();
        let mut seen = vec![normalize_claim(&question)];
        let mut risk = first.risk.clone();
        for &id in &sq.also_ids {
            let Some((p, i)) = table.resolve_index(id) else {
                continue;
            };
            let other = &plans[p].sub_questions[i];
            let key = normalize_claim(other.question.trim_end_matches('.'));
            if seen.contains(&key) {
                // Already named — only its risk label can still raise the direction's cost.
                risk = max_confidence(&risk, &other.risk);
                continue;
            }
            seen.push(key);
            fold_text(
                &mut question,
                &mut evidence,
                &mut risk,
                &other.question,
                &other.evidence_needed,
                &other.risk,
            );
        }
        sq.question = question;
        sq.evidence_needed = evidence;
        sq.risk = risk;
    }
}

/// Validate a merged plan by id-based coverage: every cited id in range, no duplicate
/// placement, and full coverage — every input plan item appears exactly once across
/// merged sub-questions (as `from_id` or `also_ids`) plus dropped. Silent dropout is
/// rejected (fail-closed inside the extraction retry loop), and so is a drop that is not a
/// duplicate of a kept direction (dropping distinct scope for size would narrow the split). The
/// plan must also fit the run's allowance: a plan wider than `max_slots` analyst slots is rejected
/// so the merge model is told to fold related directions together rather than leave the run unable
/// to carry the plan.
fn validate_merged_plan(
    plan: &MergedPlan,
    plans: &[DecompositionPlan],
    max_slots: usize,
) -> Result<(), String> {
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
    if plan.sub_questions.is_empty() {
        return Err(
            "the merged plan has no directions — the question's scope must be split into \
                    researchable sub-questions"
                .to_string(),
        );
    }
    // Cost the plan as it will be dispatched: a fold (also_ids over non-identical items) carries
    // the highest cited risk label, so the model's own risk strings cannot understate the width.
    let mut resolved = plan.clone();
    resolve_merged_plan_ids(&mut resolved, plans);
    // A drop is only for a genuine duplicate. Dropping distinct scope to fit the allowance would
    // narrow the split silently, so a non-duplicate drop is rejected and the model must fold it
    // into the direction it belongs to instead.
    let kept: Vec<String> = resolved
        .sub_questions
        .iter()
        .map(|sq| sq.question.clone())
        .collect();
    for d in &plan.dropped {
        let Some((p, i)) = table.resolve_index(d.id) else {
            continue; // already rejected by the coverage pass above
        };
        let dropped = &plans[p].sub_questions[i].question;
        if !is_duplicate_drop(dropped, &kept) {
            return Err(format!(
                "dropped item {} ({dropped}) is not a duplicate of any kept direction — fold it \
                 into the direction it belongs to instead; the split stays complete, only genuine \
                 duplicates may be dropped",
                d.id
            ));
        }
    }
    let cost = plan_slot_cost(&resolved);
    if cost > max_slots {
        return Err(format!(
            "plan too wide: {} directions cost {cost} analyst slots (a high-risk direction costs \
             two), but the run can spend {max_slots} on the plan and must keep the rest for gap \
             rounds and verification — fold closely related directions into one",
            plan.sub_questions.len()
        ));
    }
    Ok(())
}

/// A drop is allowed only when a kept direction already covers the dropped one: every word of the
/// dropped direction appears in a kept direction (case and punctuation insensitive, so a re-phrased
/// duplicate is recognised and an unrelated direction is not). Dropping distinct scope to fit the
/// allowance is what a merge may never do; the model must fold it into the direction it belongs to
/// instead. A duplicate phrased with different words is rejected too — the merge retries, and the
/// complete fallback covers it.
fn is_duplicate_drop(dropped: &str, kept: &[String]) -> bool {
    fn words(text: &str) -> HashSet<String> {
        normalize_claim(text)
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }
    let dropped = words(dropped);
    !dropped.is_empty() && kept.iter().any(|k| dropped.is_subset(&words(k)))
}

/// Round 1's product: the round's evidence, the dispatch-time snapshots of deadline-aborted
/// analysts (the caller checkpoints FIRST, then wraps up — a crash mid-wrap-up must not lose the
/// evidence), and the plan directions the round examined (dispatched with an analyst response).
/// The directions it could not reach stay out of the list, which is how the answer names them.
struct Round1Output {
    round: EvidenceRound,
    timed_out: Vec<WrapUpEntry>,
    examined: Vec<usize>,
}

/// Round 1: one analyst per sub-question, two for high-risk items (the second gets a
/// decorrelated angle), dispatched as far as the plan's slot allowance carries. Round 1 never
/// spends the reserve the rest of the run needs, and a plan wider than the allowance is worked as
/// far as it fits instead of refusing the round — the directions left over stay unexamined and are
/// named among the answer's open questions.
#[expect(clippy::too_many_arguments)]
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
    run_key: &str,
) -> Round1Output {
    let costs: Vec<usize> = plan.sub_questions.iter().map(sub_question_slots).collect();
    let (dispatch, cost) = plan_prefix(&costs, plan_slot_budget(budget));
    let mut out = Round1Output {
        round: EvidenceRound::default(),
        timed_out: Vec::new(),
        examined: Vec::new(),
    };
    if dispatch == 0 {
        tracing::warn!(
            spent = %budget.spent,
            cap = %budget.cap,
            first_direction_cost = costs.first().copied().unwrap_or_default(),
            "research allowance left for the plan cannot carry the first direction — no plan \
             direction dispatched"
        );
        return out;
    }
    // Never dispatch after the round's deadline or a stop request — the directions it would have
    // carried stay unexamined and are named in the answer.
    if std::time::Instant::now() >= deadline
        || crate::shutdown::aborting()
        || crate::round_cancel::is_cancelled(run_key)
    {
        return out;
    }
    // `plan_prefix` sized this dispatch against `plan_slot_budget`, so the reservation fits by
    // construction — the run's reserve is untouched.
    budget.spent += cost;
    let task_template = load_prompt("research/round1.md");
    let extraction_prompt = load_prompt("extraction/findings.md");
    let angles = load_analyst_angles();
    // Dispatch-time wrap-up snapshots: frozen params + advertised tool schemas, captured before
    // spawn (aborted tasks lose their values). The specs are constant across the round's members
    // (same role+workspace).
    let wrap_up_specs = role_tool_specs(Role::Analyst, ws, false);
    let mut settled = vec![false; plan.sub_questions.len()];
    let mut idx = 0usize;
    let ledger_snapshot = ledger.render();
    let mut members = Vec::new();
    let mut member_dirs: Vec<usize> = Vec::new();
    let mut snapshots: Vec<WrapUpEntry> = Vec::new();
    for i in 0..dispatch {
        let sq = &plan.sub_questions[i];
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
            member_dirs.push(i);
            members.push(make_round_member::<AnalystFindings>(
                ws,
                agent_id,
                task,
                extraction_prompt.clone(),
                run_key.to_string(),
                question.clone(),
            ));
        }
    }
    let (runs, timed_out) = dispatch_analyst_round(members, resume, deadline, &snapshots).await;
    // A direction is examined when at least one of its analysts produced a response — findings
    // or a preserved raw report. A direction whose analysts all went silent stays unexamined.
    for (dir, run) in member_dirs.iter().zip(&runs) {
        if !matches!(run, AnalystRun::NoResponse) {
            settled[*dir] = true;
        }
    }
    // collect_evidence runs first so the wrap-up's ledger registrations can
    // never skew this round's saturation counters (round.queries /
    // repeat_queries stay from successful analysts only).
    out.round = collect_evidence(&runs, ledger, run_stats);
    out.timed_out = timed_out;
    out.examined = settled
        .iter()
        .enumerate()
        .filter(|(_, settled)| **settled)
        .map(|(i, _)| i)
        .collect();
    out
}

/// Interim consolidation: extract the structured gap list from the
/// accumulated evidence (orchestrator call, not budgeted). `None` marks an
/// extraction failure — the caller must surface it explicitly, never collapse
/// it into "coverage completion".
async fn extract_gap_list(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    plan: &MergedPlan,
    run_key: &str,
) -> Option<GapList> {
    let evidence = render_accumulated_evidence(acc, true);
    let plan_json = serde_json::to_string(plan).unwrap_or_default();
    let prompt = substitute(
        &load_prompt("research/gap_extract.md"),
        &[
            ("{{question}}", question),
            ("{{plan}}", &plan_json),
            ("{{evidence}}", &evidence),
        ],
    );
    // The validator captures an owned copy (the validator type is `'static`).
    let plan_owned = plan.clone();
    orchestrator_extract::<GapList>(
        ws,
        "gap_extract",
        &prompt,
        Some(&move |g| validate_gap_list(g, &plan_owned)),
        run_key,
        question,
    )
    .await
    .ok()
}

/// Validate the gap list: every gap's `traces_to` is a 0-based index into the merged
/// plan's sub-questions (fail-closed inside the extraction retry loop).
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

/// The gap items as plain strings (for the detail file's unresolved list).
fn gap_items(gaps: &[Gap]) -> Vec<String> {
    gaps.iter().map(|g| g.item.clone()).collect()
}

/// Run one gap round: fresh analysts, one per gap in the slice (the width-shrinking
/// 4→3→2 is the caller's). Returns the runs plus dispatch-time snapshots of analysts
/// aborted by the round deadline (the caller wraps up AFTER collecting the round's
/// evidence so recovered registrations cannot skew the saturation counters).
#[expect(clippy::too_many_arguments)]
async fn run_gap_round(
    ws: &Workspace,
    question: &str,
    gaps: &[&Gap],
    ledger: &QueryLedger,
    deadline: std::time::Instant,
    resume: bool,
    run_root: &str,
    captured: &mut Vec<String>,
    run_key: &str,
) -> (Vec<AnalystRun<AnalystFindings>>, Vec<WrapUpEntry>) {
    let task_template = load_prompt("research/gap.md");
    let extraction_prompt = load_prompt("extraction/findings.md");
    let ledger_snapshot = ledger.render();
    let mut members: Vec<_> = Vec::new();
    let mut snapshots: Vec<WrapUpEntry> = Vec::new();
    // Dispatch-time wrap-up snapshots (see round1_research) — same specs for every member.
    let wrap_up_specs = role_tool_specs(Role::Analyst, ws, false);
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
            run_key.to_string(),
            question.clone(),
        ));
    }
    dispatch_analyst_round(members, resume, deadline, &snapshots).await
}

/// Record a worker-round marker — a GATE-SKIP (`"skipped — {reason}"`; not claimed) or an
/// OUTCOME. One marker per key: a fresh one supersedes the prior. Only the PRE-LOOP key-0 round is
/// re-examined on boot-resume; a post-progress gate-skip is FINAL (fail-open — the marker IS the
/// detail-file note).
fn set_worker_marker(state: &mut ResearchState, round_key: usize, suffix: &str) {
    let marker = format!("worker round {round_key} {suffix}");
    clear_worker_markers(state, round_key);
    state.markers.push(marker);
}

/// Clear stale worker-round markers for a key — one truth per key.
fn clear_worker_markers(state: &mut ResearchState, round_key: usize) {
    state
        .markers
        .retain(|m| !m.starts_with(&format!("worker round {round_key} ")));
}

/// Claim a worker round at DISPATCH. The claim is IN-MEMORY: it reaches the checkpoint
/// only at the next per-round save, so a crash mid-worker loses it and boot-resume
/// re-dispatches (the accepted crash-duplicate pin). Stale skip/outcome markers for this
/// key are cleared — one truth per key.
fn claim_worker_round(state: &mut ResearchState, round_key: usize) {
    if !state.worker_rounds_done.contains(&round_key) {
        state.worker_rounds_done.push(round_key);
    }
    clear_worker_markers(state, round_key);
}

/// Un-claim a round that was dispatched but never completed (failure or cancellation).
/// Only the pre-loop key-0 round is re-attempted by boot-resume.
fn unclaim_worker_round(state: &mut ResearchState, round_key: usize) {
    state.worker_rounds_done.retain(|k| *k != round_key);
}

/// One worker round: a single Worker sub-agent builds prototypes in the per-run folder targeting
/// the current gap list. Blocks on the worker (never hard timed out — the loop-top checks then skip
/// the following gap rounds, fail-open). The response is "what was tested in practice" and is kept
/// verbatim in [`ResearchState::worker_reports`] for the detail file — never in the answer.
/// Gated by the loop top's checks plus [`WORKER_MIN_REMAINING`]; no speculative inline retry.
#[expect(clippy::too_many_arguments)]
async fn run_worker_round(
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
        set_worker_marker(state, round_key, "skipped — analyst budget exhausted");
        return;
    }
    if std::time::Instant::now() >= deadline {
        set_worker_marker(state, round_key, "skipped — round deadline expired");
        return;
    }
    if crate::shutdown::aborting() {
        set_worker_marker(state, round_key, "skipped — shutdown/drain");
        return;
    }
    if crate::round_cancel::is_cancelled(job_id) {
        set_worker_marker(state, round_key, "skipped — run cancelled");
        return;
    }
    if std::time::Instant::now() + WORKER_MIN_REMAINING >= deadline {
        set_worker_marker(
            state,
            round_key,
            "skipped — less than 30 minutes remaining until the round deadline",
        );
        return;
    }
    // Claimed at dispatch, in-memory until the next per-round checkpoint — a crash mid-worker
    // loses the claim; only the PRE-LOOP key-0 round is re-dispatched (crash-duplicate pin).
    claim_worker_round(state, round_key);
    let evidence = render_accumulated_evidence(&state.acc, true);
    let gaps = gap_items(&gap_list.gaps).join("\n");
    let task = substitute(
        &load_prompt("synthesis/worker_brief.md"),
        &[
            ("{{question}}", question),
            ("{{evidence}}", &evidence),
            ("{{gaps}}", &gaps),
            ("{{run_root}}", run_root),
        ],
    );
    let worker_ws = Workspace::ephemeral_run(job_id, Path::new(run_root));
    let agent_id = crate::session::research_agent_id(&ws.name, "worker");
    // Command collection happens after the run completes — a crash mid-worker loses the
    // session from the sanitizer (accepted: only key 0 is re-dispatched, with a fresh agent
    // id, since `research_agent_id` embeds a fresh NanoID suffix per call).
    let (agent, response) = run_default_agent(
        &agent_id,
        Role::Worker,
        &worker_ws,
        &task,
        false,
        None,
        Some(crate::agent::registry::ParentKey::Research(
            job_id.to_string(),
        )),
        Some(question.to_string()),
    )
    .await;
    state
        .capture_round(job_id, &[agent_id], Path::new(run_root))
        .await;
    if let Some(text) = response {
        tracing::info!(job = %job_id, worker_round = round_key, "Worker round completed");
        // "What was tested in practice" is detail-file material — kept verbatim in the run folder
        // (never in the answer), re-seeded by `ResearchState::load` after a crash.
        if !text.trim().is_empty() {
            state.worker_reports.push(text);
            crate::research_cleanup::write_worker_reports(
                Path::new(run_root),
                job_id,
                &state.worker_reports,
            )
            .await;
        }
    } else {
        let cancelled = agent.is_cancelled() || crate::shutdown::aborting();
        let outcome = if cancelled { "cancelled" } else { "failed" };
        // Never completed → not done (only the pre-loop key-0 round is
        // re-attempted by boot-resume; post-progress failures are final —
        // fail-open per design).
        unclaim_worker_round(state, round_key);
        set_worker_marker(state, round_key, outcome);
    }
}

/// Two counters: `round_index` is the persisted resume pointer, `rounds_dispatched` the
/// answer telemetry count; they diverge only on the abstention / gap-extract-failure exit,
/// where the round's checkpoint is skipped so `round_index` lags by one.
///
/// Checkpoint AFTER EACH gap round: the gap-loop locals are not derivable from the
/// accumulated evidence, so a crash mid-loop would re-run the ENTIRE gap stage from round 0
/// with fresh sessions.
///
/// Worker-in-loop: one pass (key 0) before the loop, then one after every progress round
/// that refreshes a non-empty gap list; each is gated like the loop top and persisted with
/// the per-round checkpoint. Boot-resume re-attempts ONLY key 0.
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
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
    // Cumulative across boot-resumes: seed the per-invocation dispatch count from the
    // persisted checkpoint so a resumed run's "rounds used" telemetry covers the whole run.
    // The increment happens after dispatch, not at the loop top.
    let mut outcome = GapRoundsOutcome {
        abstention: None,
        unresolved: Vec::new(),
        rounds_dispatched: state.gap_outcome.rounds_dispatched,
        incomplete: None,
    };
    // Manual-cancel gate before any further orchestrator call or round: stop without
    // extracting gaps, dispatching the worker, or spawning another gap round.
    if crate::round_cancel::is_cancelled(job_id) {
        return GapRoundsOutcome::default();
    }
    let initial_list = match state.gap_list.take() {
        Some(list) => Some(list),
        None => extract_gap_list(ws, question, &state.acc, plan, job_id).await,
    };
    let Some(mut gap_list) = initial_list else {
        // Explicit marker — never collapse into coverage completion.
        outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
        return outcome;
    };
    // Pre-loop worker round (key 0): one prototype pass over the initial gap list.
    if !gap_list.gaps.is_empty() && !state.worker_rounds_done.contains(&0) {
        run_worker_round(
            job_id, run_root, ws, question, budget, &gap_list, deadline, state, 0,
        )
        .await;
    }
    let mut round_index = state.round_index;
    loop {
        if crate::shutdown::aborting()
            || crate::round_cancel::is_cancelled(job_id)
            || budget.is_exhausted()
            // Round-wide deadline expired: further rounds would be spawned and instantly aborted —
            // stop instead of burning budget on no-progress work.
            || std::time::Instant::now() >= deadline
        {
            outcome.unresolved = gap_items(&gap_list.gaps);
            return outcome;
        }
        // The gap list is validated (traces_to in range) — every gap is used directly.
        let gaps = &gap_list.gaps;
        if gaps.is_empty() {
            // Coverage completion.
            return outcome;
        }
        let width = GAP_ROUND_WIDTHS[round_index.min(GAP_ROUND_WIDTHS.len() - 1)];
        let targeted: Vec<&Gap> = gaps.iter().take(width).collect();
        // The verification pass's slots are never spent on gap rounds: a round that would leave it
        // short is not started, and the remaining gaps stay open (named in the answer, listed in
        // the detail file).
        if budget.cap.saturating_sub(budget.spent)
            < targeted.len() + crate::tools::analyze::VERIFY_MAX_ANALYSTS
        {
            outcome.unresolved = gap_items(gaps);
            return outcome;
        }
        round_index += 1;
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
            job_id,
        )
        .await;
        state
            .capture_round(job_id, &round_agents, Path::new(run_root))
            .await;
        // A gap round does NOT settle the plan direction its gap traces to: it answers a narrower
        // follow-up, so the direction stays an open question unless round 1 examined it.
        let round = collect_evidence(&runs, &mut state.ledger, run_stats);
        // Wrap-up stays BEFORE the per-round checkpoint here (unlike round 1): the loop's early
        // returns (abstention / gap-extract failure) follow this position and the final round's
        // wrap-up must survive them.
        recovered.extend(
            wrap_up_timed_out(
                ws,
                timed_out,
                &mut state.ledger,
                run_stats,
                job_id,
                question,
            )
            .await,
        );
        let (new_urls, pending) = state.acc.absorb(&round);
        let novel_claims = annotate_round(
            ws,
            &mut state.acc,
            &pending,
            &mut state.markers,
            job_id,
            question,
        )
        .await;
        outcome.rounds_dispatched += 1;
        // A round whose analysts only re-asked already-asked queries counts as no-progress
        // (repeat queries are never pre-dispatch-droppable — concurrent analysts generate them).
        let all_repeat_queries = round.queries > 0 && round.queries == round.repeat_queries;
        if (new_urls == 0 && novel_claims == 0) || all_repeat_queries {
            // Structural quiet-round signal: no new claims and no new sources. A
            // weak-duplicate-only round also lands here (weak duplicates append standalone claims
            // but never count as novel) — the answerability check fires and the gap list is not
            // refreshed: intended premature-saturation protection.
            if let Some(reason) = check_answerability(ws, question, &state.acc, job_id).await {
                outcome.abstention = Some(reason);
                outcome.unresolved = gap_items(gaps);
                return outcome;
            }
        }
        // Refresh the gap list from the accumulated evidence only when the round produced progress.
        if new_urls != 0 || novel_claims != 0 {
            let Some(next_gap_list) =
                extract_gap_list(ws, question, &state.acc, plan, job_id).await
            else {
                outcome.incomplete = Some(GAP_EXTRACTION_FAILED.to_string());
                outcome.unresolved = gap_items(gaps);
                return outcome;
            };
            gap_list = next_gap_list;
            // Post-progress worker round (key = this gap round's index): fresh prototypes targeting
            // the refreshed gap list.
            if !gap_list.gaps.is_empty() && !state.worker_rounds_done.contains(&round_index) {
                run_worker_round(
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

/// Per-round claim annotation pass: classify each pending claim against the accumulated claims via
/// one orchestrator extraction call (the validator guarantees id completeness and in-range ids).
/// Weak hints render in the existing-claims listing so later rounds see them.
async fn annotate_claims(
    ws: &Workspace,
    existing: &[Claim],
    weak: &WeakLinks,
    pending: &[Claim],
    run_key: &str,
    question: &str,
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
        run_key,
        question,
    )
    .await
}

/// Validate an annotation pass: every pending claim annotated exactly once;
/// `duplicate`/`contradicts` cite an in-range existing claim; `novel` cites nothing; the
/// contradiction note is present exactly for "contradicts". Structural only.
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

/// Confirm pass over a round's mutating annotation links: one lightweight extraction re-judging
/// each pair between the annotation pass and applying its results. Fail-closed: every pair judged
/// exactly once with {confirm, reject}; a failed call maps to all-weak + marker.
async fn confirm_links(
    ws: &Workspace,
    existing: &[Claim],
    pending: &[Claim],
    pass: &AnnotationPass,
    run_key: &str,
    question: &str,
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
    // The validator captures the mutating new_ids (the validator type is `'static`) — pair
    // identity is pinned by the annotation pass.
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
        run_key,
        question,
    )
    .await
}

/// Validate a confirm pass: exactly the mutating links judged, each exactly once (set
/// equality on new_ids), verdicts in [confirm, reject]. Structural only.
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

/// Run the annotation pass over a round's pending claims, then the optional confirm pass over its
/// mutating links, and apply the results; returns the number of novel claims. On annotation
/// exhaustion every pending claim is novel with a marker — claims are never dropped.
async fn annotate_round(
    ws: &Workspace,
    acc: &mut AccumulatedEvidence,
    pending: &[Claim],
    markers: &mut Vec<String>,
    run_key: &str,
    question: &str,
) -> usize {
    if pending.is_empty() {
        return 0;
    }
    if acc.claims.is_empty() {
        // First round: nothing to compare against — every pending claim is novel by construction.
        // Skip the wasted orchestrator call (and the spurious failure marker an out-of-range
        // existing_id would otherwise trigger).
        acc.claims.extend(pending.iter().cloned());
        return pending.len();
    }
    let Ok(pass) = annotate_claims(ws, &acc.claims, &acc.weak, pending, run_key, question).await
    else {
        acc.claims.extend(pending.iter().cloned());
        if !markers.iter().any(|m| m == CLAIM_ANNOTATION_FAILED) {
            markers.push(CLAIM_ANNOTATION_FAILED.to_string());
        }
        return pending.len();
    };
    // Confirm pass over the mutating links only. No mutating links → every verdict applies as-is.
    let confirm = if pass.annotations.iter().any(|a| a.verdict != "novel") {
        if let Ok(c) = confirm_links(ws, &acc.claims, pending, &pass, run_key, question).await {
            ConfirmOutcome::Passed(c)
        } else {
            // Fail-open: every mutating verdict becomes weak/unconfirmed (never all-novel, no
            // drops).
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

/// After a structural quiet round (no new claims, no new sources): is the question
/// genuinely unanswerable with the evidence gathered? `Some(abstention)` when it is
/// (orchestrator call, not budgeted).
async fn check_answerability(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    run_key: &str,
) -> Option<String> {
    let evidence = render_accumulated_evidence(acc, true);
    let prompt = substitute(
        &load_prompt("research/abstain.md"),
        &[("{{question}}", question), ("{{evidence}}", &evidence)],
    );
    let verdict = orchestrator_extract::<AnswerabilityCheck>(
        ws,
        "abstain_check",
        &prompt,
        None,
        run_key,
        question,
    )
    .await
    .ok()?;
    (!verdict.answerable).then_some(verdict.reason)
}

/// What the synthesis loop produced. The answer text is only ever an attempt that completed cleanly
/// within the answer ceiling; an over-long or provider-truncated output is never delivered as the
/// answer but is kept whole for the detail file.
#[derive(Debug)]
enum SynthesisOutcome {
    /// The answer, plus any earlier attempt that could not be the answer and is kept for the detail
    /// file.
    Answer {
        text: String,
        undelivered: Option<String>,
    },
    /// No deliverable answer: `reason` says why (it becomes the product-built answer's header) and
    /// `output` is the last attempt's text, kept whole for the detail file.
    Undelivered {
        reason: String,
        output: Option<String>,
    },
}

/// Record one rejected synthesis attempt against the retry policy.
fn record_synthesis_failure(
    loop_state: &mut crate::retry::RetryLoop,
    class: FailureClass,
    resp: &crate::ChatResponse,
    reason: &str,
) {
    let err = anyhow::anyhow!("{reason}");
    loop_state.record(crate::retry::RetryFailureRecord::with_metadata(
        class,
        &err,
        resp.finish_reason.clone(),
        None,
    ));
}

/// Reject an answer that misses the size contract (`feedback_key` names the re-ask), recording it
/// against the retry policy, and return the feedback the next attempt is sent with.
fn reject_answer_size(
    loop_state: &mut crate::retry::RetryLoop,
    resp: &crate::ChatResponse,
    bytes: usize,
    floor: usize,
    ceiling: usize,
    feedback_key: &str,
) -> String {
    let reason = format!(
        "synthesis answer is {bytes} bytes (target {}–{} KB)",
        floor / 1024,
        ceiling / 1024
    );
    record_synthesis_failure(loop_state, FailureClass::ValidationOther, resp, &reason);
    substitute(
        &load_prompt(feedback_key),
        &[
            ("{{bytes}}", &bytes.to_string()),
            ("{{min}}", &(floor / 1024).to_string()),
            ("{{max}}", &(ceiling / 1024).to_string()),
        ],
    )
}

/// Final synthesis with the shared policy (≤3 informed attempts, transport-only backoff; the 600 s
/// idle timeout is the only per-attempt bound). A truncated, empty, over-long or failed attempt
/// counts toward the budget and is re-produced with feedback — never trimmed. The answer is
/// delivered only when an attempt comes in clean and within `ceiling`, so a reader never gets a
/// 2–3x over-ceiling or half-sentence answer; an over-long or truncated last output comes back as
/// undelivered and the detail file keeps it whole.
///
/// `ceiling` is the bytes the answer text may occupy: [`ANSWER_MAX_BYTES`] less the room the
/// product's own open-questions section and the detail-file reference need, so the delivered answer
/// (text + section + reference) never exceeds the ceiling however long the storage-root path is.
/// `recovered` (deadline-aborted analysts' findings) is evidence like any other: the answer must be
/// built from what the run established, including what the wrap-up recovered.
#[expect(clippy::too_many_lines, clippy::too_many_arguments)]
async fn synthesize(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    recovered: &[AnalystFindings],
    abstention: Option<&str>,
    open: &[String],
    ceiling: usize,
    run_key: &str,
) -> Result<SynthesisOutcome> {
    let _call = research_call_guard("synthesize", &ws.name, run_key, false, question);
    // The open-questions section below is the single place the analysts' unanswered aspects appear
    // in this prompt.
    let mut evidence = render_accumulated_evidence(acc, false);
    evidence.push_str(&render_recovered_findings(recovered, RenderTarget::Prompt));
    let mut open_list = String::new();
    for q in open {
        let _ = writeln!(open_list, "- {q}");
    }
    if open_list.is_empty() {
        open_list.push_str("- none recorded");
    }
    // The answer text's own floor: the target, never above the ceiling the product's section and
    // reference leave for it.
    let floor = ANSWER_MIN_BYTES.min(ceiling);
    let mut base_user = substitute(
        &load_prompt("research/synthesize.md"),
        &[
            ("{{question}}", question),
            ("{{evidence}}", &evidence),
            ("{{open_questions}}", &open_list),
            ("{{min}}", &(floor / 1024).to_string()),
            ("{{max}}", &(ceiling / 1024).to_string()),
        ],
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
    let mut loop_state = crate::retry::RetryLoop::new_scoped(&policy, &params);
    let operation_started = Instant::now();
    let mut prefix = Vec::with_capacity(2);
    crate::prompt::prepend_general_context(&mut prefix, ws).await;
    // The last undeliverable output (truncated or over-long) and why it is undeliverable.
    let mut last: Option<String> = None;
    let mut last_problem: Option<String> = None;
    // The first answer below the size target: complete and within the ceiling, so it stays
    // deliverable if no later attempt does better.
    let mut short: Option<String> = None;
    let mut asked_for_more = false;
    let mut any_truncated = false;
    let mut feedback = String::new();
    let mut transport_failures = 0u32;

    for attempt in 1..=policy.max_attempts {
        loop_state.begin_attempt(Instant::now());
        let mut user = base_user.clone();
        if !feedback.is_empty() {
            let _ = writeln!(user, "\n\n# Previous Attempt Feedback\n\n{feedback}");
        }
        let mut messages = prefix.clone();
        messages.push(ChatMessage::user(&user));
        params.messages = messages;
        match crate::providers::chat_scoped(params.clone()).await {
            Ok(resp) => {
                let text = resp.text_or_empty().to_string();
                let truncated = resp.finish_reason.as_deref() == Some("length");
                if truncated {
                    any_truncated = true;
                    record_synthesis_failure(
                        &mut loop_state,
                        FailureClass::TruncatedOutput,
                        &resp,
                        "synthesis truncated by the provider (finish_reason=length)",
                    );
                    feedback = load_prompt("research/synthesize_feedback_truncated.md");
                    if !text.trim().is_empty() {
                        last = Some(text);
                        last_problem = Some("the provider truncated the last attempt".to_string());
                    }
                    continue;
                }
                if text.trim().is_empty() {
                    record_synthesis_failure(
                        &mut loop_state,
                        FailureClass::NoResponse,
                        &resp,
                        "synthesis attempt returned empty text",
                    );
                    feedback = load_prompt("research/synthesize_feedback_empty.md");
                    continue;
                }
                // The answer has a size of its own. An over-long attempt is re-produced, not
                // trimmed: the detail is delivered in the companion file, so the answer only has
                // to carry what was established, what follows from it and what remains open.
                if text.len() > ceiling {
                    feedback = reject_answer_size(
                        &mut loop_state,
                        &resp,
                        text.len(),
                        floor,
                        ceiling,
                        "research/synthesize_feedback_over_length.md",
                    );
                    last = Some(text);
                    last_problem = Some(format!(
                        "the last attempt was over the {} KB answer ceiling",
                        ceiling / 1024
                    ));
                    continue;
                }
                // Below the target: one re-ask, then accept — a run with thin evidence answers
                // briefly, and padding it would contradict "built from what it established". Only
                // re-ask when the evidence could actually fill the target.
                if text.len() < floor && !asked_for_more && evidence.len() >= floor {
                    asked_for_more = true;
                    feedback = reject_answer_size(
                        &mut loop_state,
                        &resp,
                        text.len(),
                        floor,
                        ceiling,
                        "research/synthesize_feedback_below_target.md",
                    );
                    short = Some(text);
                    continue;
                }
                // Clean completion wins.
                crate::stats::record_llm_success(&params, operation_started, attempt, &resp);
                return Ok(SynthesisOutcome::Answer {
                    text,
                    undelivered: last,
                });
            }
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(err.record);
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
    // Exhausted: a below-target output is a complete answer within the ceiling; an over-long or
    // truncated one never is. No output at all errors into the caller's product-built answer.
    let final_class = loop_state.final_class();
    let exhausted = crate::retry::RetryExhausted::with_last_raw(
        loop_state.into_failures(),
        final_class,
        last.clone(),
    );
    crate::stats::record_llm_failure(&params, operation_started, &exhausted);
    // A below-target output is a complete answer within the ceiling; an over-long or truncated one
    // never is, so it is kept whole for the detail file instead.
    let undeliverable = last.zip(last_problem);
    match (short, undeliverable) {
        (Some(text), undelivered) => Ok(SynthesisOutcome::Answer {
            text,
            undelivered: undelivered.map(|(text, _)| text),
        }),
        (None, Some((text, reason))) => Ok(SynthesisOutcome::Undelivered {
            reason,
            output: Some(text),
        }),
        (None, None) if any_truncated => Err(anyhow::anyhow!(
            "final synthesis was truncated with no usable output"
        )),
        (None, None) => Err(anyhow::anyhow!(
            "final synthesis produced no usable output after {} attempts",
            policy.max_attempts
        )),
    }
}

/// Build the verification target list: primary targets (contradiction notes or low confidence)
/// first, then weak-duplicate claims not already primary, filling only empty slots and never
/// double-dispatching. Returns `(targets, primary_count)`.
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
/// unresolved in the detail file. Verifier tool calls / searches / queries
/// … register in the query ledger (later rounds must not re-ask them); repeats here
/// inflate the summary's repeat-queries line but never `EvidenceRound::repeat_queries`.
#[expect(clippy::too_many_arguments)]
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
    run_key: &str,
    question: &str,
) -> Vec<VerificationResult> {
    let (targets, primary_count) = verification_targets(acc);
    if targets.is_empty() {
        return Vec::new();
    }
    let cap = targets
        .len()
        .min(crate::tools::analyze::VERIFY_MAX_ANALYSTS);
    // Reserve what fits — a near-exhausted budget still verifies the
    // highest-priority targets instead of nothing.
    let n = cap.min(budget.cap.saturating_sub(budget.spent));
    let mut results = Vec::new();
    // Never spawn verifiers once the round-wide deadline expired — they would be aborted
    // instantly; the primary targets are marked unresolved below instead.
    if n > 0 && std::time::Instant::now() < deadline && budget.try_reserve(n).is_ok() {
        let ledger_snapshot = ledger.render();
        let task_extra = format!(
            "\n# Queries Already Asked (do not repeat these verbatim)\n\n{ledger_snapshot}\n\n\
             # Scratch Workspace\n\nTemporary per-run folder (wiped after the run):\n\n{run_root}"
        );
        let prefix = format!("research_{}_verify", ws.name);
        let (verify_results, verify_ids) = dispatch_claim_verifiers(
            ws,
            &prefix,
            &targets[..n],
            &task_extra,
            deadline,
            resume,
            Some(crate::agent::registry::ParentKey::Research(
                run_key.to_string(),
            )),
            question,
        )
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
    // Primary targets beyond the verified set are explicitly marked unresolved — never
    // silently skipped. Appended weak-duplicate targets that did not fit are not (their weak
    // state stays visible in the evidence view and run summary).
    for t in targets
        .iter()
        .skip(results.len())
        .take(primary_count.saturating_sub(results.len()))
    {
        results.push(VerificationResult::unresolved(
            &t.claim,
            "verification skipped — budget exhausted or round deadline expired",
            0,
            0,
            Vec::new(),
        ));
    }
    results
}

/// Render the weak/unconfirmed hints involving claim `id` (one indented line per hint,
/// empty when none). Shared by the evidence view, the annotate listing and the detail
/// file — weakness is per-claim visible and never silent.
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

/// The analysts' self-reported unanswered aspects, for the orchestrator prompts (raw — the prompt
/// views never escape, so the model sees the text as written).
fn render_unanswered(out: &mut String, unanswered: &[String]) {
    if unanswered.is_empty() {
        return;
    }
    let _ = writeln!(out, "\nAnalysts reported these as still unanswered:");
    for u in unanswered {
        let _ = writeln!(out, "- {u}");
    }
}

/// Per-analyst raw-report section (a `##` heading in the detail file).
fn render_raw_reports(out: &mut String, raw_reports: &[String]) {
    if raw_reports.is_empty() {
        return;
    }
    let _ = writeln!(out, "\n## Failed Analyst Reports");
    for (i, raw) in raw_reports.iter().enumerate() {
        let _ = writeln!(out, "### Report from Analyst {}", i + 1);
        let _ = writeln!(out, "{}", escape_fences(raw));
    }
}

/// Where a rendered line goes: into a Markdown document delivered to the reader (fence-escaped, so a
/// claim can never break the surrounding structure) or into an orchestrator prompt as evidence
/// (verbatim — the model gets what the analyst said).
#[derive(Clone, Copy)]
enum RenderTarget {
    Delivered,
    Prompt,
}

impl RenderTarget {
    fn text(self, s: &str) -> String {
        match self {
            RenderTarget::Delivered => escape_fences(s),
            RenderTarget::Prompt => s.to_string(),
        }
    }
}

/// One claim as a list line — numbered when `number` is given, bulleted otherwise — plus its
/// contradictions when `with_contradictions`. The ONE claim line, shared by the prompt evidence
/// view, the product-built answer and the detail file; callers add the weak-link hints they want.
fn render_claim_line(
    out: &mut String,
    c: &Claim,
    number: Option<usize>,
    with_contradictions: bool,
    target: RenderTarget,
) {
    let confidence = &c.confidence;
    let claim = target.text(&c.claim);
    let source = target.text(source_or_na(&c.source));
    match number {
        Some(n) => {
            let _ = writeln!(out, "{n}. [{confidence}] {claim} — source: {source}");
        }
        None => {
            let _ = writeln!(out, "- [{confidence}] {claim} — source: {source}");
        }
    }
    if with_contradictions && !c.contradictions.is_empty() {
        let _ = writeln!(
            out,
            "   contradictions: {}",
            target.text(&c.contradictions.join("; "))
        );
    }
}

/// The claim list with its weak/unconfirmed hints — numbered by the claim's stable id. Shared by the
/// prompt evidence view and the detail file.
fn render_claims(out: &mut String, acc: &AccumulatedEvidence, target: RenderTarget) {
    for (i, c) in acc.claims.iter().enumerate() {
        render_claim_line(out, c, Some(i), true, target);
        out.push_str(&render_weak_hints(&acc.weak, i));
    }
}

/// Render the accumulated evidence as a compact numbered list plus analysts' unanswered
/// aspects. Used by the orchestrator prompts and the worker brief (`run_worker_round`).
/// Claim ids are their stable 0-based indices in `acc.claims`. `with_unanswered` is false only for
/// the synthesis prompt, which has a dedicated open-questions section naming them.
fn render_accumulated_evidence(acc: &AccumulatedEvidence, with_unanswered: bool) -> String {
    let mut out = String::new();
    render_claims(&mut out, acc, RenderTarget::Prompt);
    if with_unanswered {
        render_unanswered(&mut out, &acc.unanswered);
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

#[expect(clippy::too_many_arguments)]
fn render_run_summary(
    run_stats: &RunStats,
    budget: &ResearchBudget,
    rounds_used: usize,
    acc: &AccumulatedEvidence,
    abstention: Option<&str>,
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
    out
}

/// Render the recovered-from-timed-out-analysts section. Separate from the main evidence by
/// design: recovered findings never entered verification. Fed to synthesis (the answer is built
/// from them) and rendered into the detail file; never into the answer verbatim. The analysts'
/// unanswered aspects are not repeated here — they are part of the run's open questions. Empty when
/// there is nothing recovered.
fn render_recovered_findings(recovered: &[AnalystFindings], target: RenderTarget) -> String {
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
            render_claim_line(&mut out, c, None, true, target);
        }
    }
    out
}

/// What the answer is built from: the synthesized text, or the reason there is none (the
/// product-built answer stands in).
enum AnswerSource {
    Synthesized(String),
    ProductBuilt(String),
}

/// Everything the two artifacts are rendered from — bundled so every exit path (a synthesized
/// answer or the product-built one) writes both from one value.
struct RunReport<'a> {
    state: &'a ResearchState,
    recovered: &'a [AnalystFindings],
    verification: &'a [VerificationResult],
    run_stats: &'a RunStats,
    budget: &'a ResearchBudget,
    /// The run's open questions — what the answer names compactly and the detail file lists in
    /// full — and their rendered compact section. Both are computed once by the caller, so the
    /// bytes measured into the answer ceiling are the very bytes the answer carries.
    open: &'a [String],
    open_section: &'a str,
    rounds_used: usize,
    wall: Duration,
}

impl RunReport<'_> {
    /// The answer (compact, at most [`ANSWER_MAX_BYTES`]) plus the product's compact open-questions
    /// section. A run never ends without an answer: [`AnswerSource::ProductBuilt`] carries the
    /// reason the product-built one stands in. The closing reference to the detail file is appended
    /// by [`write_terminalization_artifacts`], only once that file is actually written.
    fn output(&self, source: AnswerSource, job_id: &str) -> ResearchOutput {
        let mut answer = match source {
            AnswerSource::Synthesized(text) => text,
            AnswerSource::ProductBuilt(reason) => {
                self.answer(&reason, answer_text_ceiling(self.open_section, job_id))
            }
        };
        answer.push_str(self.open_section);
        ResearchOutput {
            answer,
            detail: self.detail(),
        }
    }

    /// The product-built answer for a run that could not synthesize one: what it did establish,
    /// compactly, within `ceiling`. Never empty and never an error envelope — a run that found
    /// anything reports something. Every line carrying run content is capped, so one over-long
    /// finding or failure reason can never crowd the answer out or leave a section showing only a
    /// count, and every line — the closing "and N more" included — is counted against the ceiling.
    fn answer(&self, reason: &str, ceiling: usize) -> String {
        let acc = &self.state.acc;
        let mut out = String::new();
        // The reason can be an entire error chain: cap it rather than let the header be refused.
        let header = format!("## Deep Analysis Answer (incomplete — {reason})");
        push_answer_line(
            &mut out,
            &cap_answer_line(&header, ANSWER_LINE_MAX_BYTES),
            ceiling,
        );
        push_answer_line(&mut out, "\n### Evidence Gathered So Far\n\n", ceiling);
        let claims: Vec<&Claim> = acc
            .claims
            .iter()
            .chain(self.recovered.iter().flat_map(|r| r.claims.iter()))
            .collect();
        if claims.is_empty() {
            // Raw reports and recovered findings are findings too — never claim nothing was found
            // while the detail file holds usable material.
            let line = if acc.raw_reports.is_empty() {
                "- nothing was established before the run stopped\n"
            } else {
                "- structured findings could not be extracted; the analysts' raw reports are in the \
                 detail file\n"
            };
            push_answer_line(&mut out, line, ceiling);
        } else {
            render_capped_answer_list(&mut out, &claims, ANSWER_LIST_CAP, Some(ceiling), |c| {
                let mut line = String::new();
                render_claim_line(&mut line, c, None, false, RenderTarget::Delivered);
                line
            });
        }
        out
    }

    /// The detail file delivered next to the answer: everything the run produced, untrimmed. The
    /// answer is compact because this file exists, not because anything was dropped.
    fn detail(&self) -> String {
        let state = self.state;
        let acc = &state.acc;
        let mut out = String::new();
        let _ = writeln!(out, "# Deep Analysis Detail\n");
        let _ = writeln!(out, "## Full Evidence\n");
        if acc.claims.is_empty() {
            let _ = writeln!(out, "- no structured claims were collected");
        }
        render_claims(&mut out, acc, RenderTarget::Delivered);
        out.push_str(&render_recovered_findings(
            self.recovered,
            RenderTarget::Delivered,
        ));
        render_raw_reports(&mut out, &acc.raw_reports);
        if !self.verification.is_empty() {
            let _ = writeln!(out, "\n## Verification");
            for v in self.verification {
                let _ = writeln!(
                    out,
                    "- {} → **{}** — {}",
                    escape_fences(&v.claim),
                    v.verdict,
                    escape_fences(&v.evidence),
                );
            }
        }
        if !state.worker_reports.is_empty() {
            let _ = writeln!(out, "\n## Tested in Practice");
            for report in &state.worker_reports {
                let _ = writeln!(out, "\n{}", escape_fences(report));
            }
        }
        let _ = writeln!(out, "\n## Open Questions");
        if self.open.is_empty() {
            let _ = writeln!(out, "- no open questions were recorded");
        }
        for q in self.open {
            let _ = writeln!(out, "- {}", escape_fences(q));
        }
        let _ = writeln!(out);
        out.push_str(&render_run_summary(
            self.run_stats,
            self.budget,
            self.rounds_used,
            acc,
            state.gap_outcome.abstention.as_deref(),
            state.gap_outcome.incomplete.as_deref(),
            &state.markers,
            self.wall,
        ));
        out
    }
}

/// Render at most `cap` items as capped answer lines, each counted against `bound` when one is
/// given, then name the rest in one line. `render` turns one item into its uncapped line. The ONE
/// capped list both the product-built answer and its open-questions section are built from.
fn render_capped_answer_list<T>(
    out: &mut String,
    items: &[T],
    cap: usize,
    bound: Option<usize>,
    render: impl Fn(&T) -> String,
) {
    let ceiling = bound.unwrap_or(usize::MAX);
    let mut shown = 0usize;
    for item in items.iter().take(cap) {
        if !push_answer_line(
            out,
            &cap_answer_line(&render(item), ANSWER_LINE_MAX_BYTES),
            ceiling,
        ) {
            break;
        }
        shown += 1;
    }
    if items.len() > shown {
        push_answer_line(
            out,
            &format!(
                "- …and {} more, listed in full in the detail file\n",
                items.len() - shown
            ),
            ceiling,
        );
    }
}

/// Append `line` to the answer when it still fits `ceiling` bytes, returning whether it was
/// appended. Every line of a product-built answer goes through here, so the ceiling holds for the
/// closing "and N more" line too.
fn push_answer_line(out: &mut String, line: &str, ceiling: usize) -> bool {
    if out.len() + line.len() > ceiling {
        return false;
    }
    out.push_str(line);
    true
}

/// Cap one answer line at `max` bytes *including* its newline, marking the cut — a single over-long
/// finding must not crowd out the rest of the compact answer.
fn cap_answer_line(line: &str, max: usize) -> String {
    let mut capped = crate::util::cut_bytes(line.trim_end_matches('\n'), max.saturating_sub(1));
    capped.push('\n');
    capped
}

/// The answer's closing reference to its companion detail file.
fn detail_reference(path: &Path) -> String {
    format!(
        "\n---\n\nFull detail — every finding with its sources, the failed analysts' reports, the \
         verification results, what was tested in practice and the run bookkeeping — is delivered \
         alongside this answer in:\n\n`{}`\n",
        path.display()
    )
}

/// Bytes [`detail_reference`] occupies — reserved out of the answer's ceiling, and zero when there
/// is no results directory to write the detail into (the answer then carries no reference).
fn detail_reference_len(job_id: &str) -> usize {
    crate::research_cleanup::detail_archive_path(job_id)
        .map_or(0, |path| detail_reference(&path).len())
}

/// The compact "Questions This Run Could Not Settle" section every answer ends with: the most
/// material few, the rest deferred to the detail file. The product writes it, not the model, so a
/// synthesized answer names what stayed open structurally. Bounded by [`ANSWER_OPEN_CAP`] ×
/// [`ANSWER_LINE_MAX_BYTES`]; its bytes are reserved out of the answer ceiling as a whole, so no
/// per-line budget is applied here.
fn render_open_section(open: &[String]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n### Questions This Run Could Not Settle\n");
    if open.is_empty() {
        let _ = writeln!(out, "- no open questions were recorded");
        return out;
    }
    render_capped_answer_list(&mut out, open, ANSWER_OPEN_CAP, None, |q| {
        format!("- {}", escape_fences(q))
    });
    out
}

/// Bytes the answer text may occupy: the whole-answer ceiling less the product's open-questions
/// section (rendered once by the caller) and the closing detail reference, so the delivered answer
/// never exceeds [`ANSWER_MAX_BYTES`].
fn answer_text_ceiling(open_section: &str, job_id: &str) -> usize {
    ANSWER_MAX_BYTES.saturating_sub(open_section.len() + detail_reference_len(job_id))
}

/// Every question the run did not settle: the plan directions round 1 never examined (derived from
/// the plan, so a checkpoint written before the examined list existed still names every direction),
/// the gaps the gap loop left open, and the aspects the analysts — including the deadline-aborted
/// ones the wrap-up recovered — reported as unanswered. Without a plan the question itself is what
/// stayed open. Named compactly in the answer, listed in full in the detail file.
fn open_questions(
    state: &ResearchState,
    recovered: &[AnalystFindings],
    question: &str,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    match &state.plan {
        Some(plan) => out.extend(
            plan.sub_questions
                .iter()
                .enumerate()
                .filter(|(i, _)| !state.examined.contains(i))
                .map(|(_, sq)| sq.question.clone()),
        ),
        None => out.push(question.to_string()),
    }
    let recovered_unanswered = recovered.iter().flat_map(|r| r.unanswered.iter());
    for q in state
        .gap_outcome
        .unresolved
        .iter()
        .chain(state.acc.unanswered.iter())
        .chain(recovered_unanswered)
    {
        if !out.contains(q) {
            out.push(q.clone());
        }
    }
    out
}

enum ResearchExit {
    /// Real terminalization (the answer plus its detail file, or a technical failure) — the caller
    /// writes the artifacts and completes the durable job.
    Terminal(anyhow::Result<ResearchOutput>),
    /// Shutdown/drain abort — the run stays alive for the next boot; nothing is written or routed.
    Aborted,
    /// Manual cancel (Running Agents page): a PERMANENT stop — the run must never resume,
    /// re-dispatch or deliver an answer, now or after any restart. The caller hands the record to
    /// the cancelled-run handoff, which decides what is held. Distinct from
    /// [`Aborted`](ResearchExit::Aborted).
    Cancelled,
}

/// Run the resumable deep-research orchestrator: round 0, round 1 (as far as the plan's allowance
/// carries), conditional gap rounds, one-shot synthesis, verification gate — then the two
/// artifacts every terminalization delivers: the compact answer and the full detail next to it.
/// `job_id` names the durable job whose `research_jobs.state` checkpoint is loaded on entry and
/// saved at every boundary.
#[expect(clippy::too_many_lines)]
async fn run_deep_research(
    ws: &Workspace,
    question: &str,
    job_id: &str,
    resume: bool,
) -> ResearchExit {
    // Hold a non-agent-call guard for the WHOLE run: the drain-watch fires the token only when both
    // registries are empty, and this guard bridges the inter-phase windows so the token never lands
    // in a just-starting call. It renders in the Running Agents view as a run-lifetime indicator.
    let _orchestrator_guard =
        research_call_guard("research_orchestrator", &ws.name, job_id, true, question);
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
    // Per-run scratch folder — created idempotently BEFORE round 0 (decompose receives it
    // first). The absolute temp-dir path is what the readonly shell allows; `$TMPDIR` in the
    // agent shell differs on macOS.
    let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
    let run_root_str = run_root.to_string_lossy().to_string();
    // Recovered-from-timed-out-analysts findings: round locals only (never checkpointed) — a
    // crash between rounds loses them, and a resumed run starts with an empty set.
    let mut recovered: Vec<AnalystFindings> = Vec::new();

    // Manual-cancel gate BEFORE round 0: a cancel that landed between the dispatch
    // registration and the first boundary must stop immediately.
    if crate::round_cancel::is_cancelled(job_id) {
        return ResearchExit::Cancelled;
    }

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
            job_id,
        )
        .await;
        // Capture on EVERY outcome — a hard decompose failure still leaves the dispatched agents'
        // sessions behind (their OS-temp scratch must reach the command dump).
        state.capture_round(job_id, &round_agents, &run_root).await;
        round_agents.clear();
        let plan = match round0 {
            Ok((plan, marker)) => {
                if let Some(m) = marker {
                    state.markers.push(m);
                }
                plan
            }
            Err(_) if crate::shutdown::aborting() => {
                // Checkpoint the state: boot resume re-dispatches round 0 with FRESH agent ids, and
                // the aborted agents' OS-temp scratch already reached the command dump
                // (capture_round wrote it), so the in-memory capture dying here loses nothing.
                state.save(job_id).await;
                return ResearchExit::Aborted;
            }
            Err(_) if crate::round_cancel::is_cancelled(job_id) => {
                return ResearchExit::Cancelled;
            }
            Err(e) => {
                // Checkpoint the state — the dump was already written by capture_round; a hard
                // decompose failure must not lose the agents' sessions from the run's detail.
                state.save(job_id).await;
                // A run that never got a plan still answers: it reports that nothing was
                // established and names the question itself as what stayed open — never an error
                // envelope.
                let open = open_questions(&state, &recovered, question);
                let open_section = render_open_section(&open);
                let report = RunReport {
                    state: &state,
                    recovered: &recovered,
                    verification: &[],
                    run_stats: &run_stats,
                    budget: &budget,
                    open: &open,
                    open_section: &open_section,
                    rounds_used: 0,
                    wall: start.elapsed(),
                };
                let output = report.output(
                    AnswerSource::ProductBuilt(format!(
                        "the question could not be decomposed: {e:#}"
                    )),
                    job_id,
                );
                return ResearchExit::Terminal(Ok(output));
            }
        };
        state.plan = Some(plan);
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::Round1;
        state.save(job_id).await;
    }

    // Check shutdown/drain between rounds — never spawn round-1 analysts during the drain.
    if crate::shutdown::aborting() {
        return ResearchExit::Aborted;
    }

    if crate::round_cancel::is_cancelled(job_id) {
        return ResearchExit::Cancelled;
    }

    if state.stage == ResearchStage::Round1 {
        if let Some(plan) = state.plan.as_ref() {
            let r1 = round1_research(
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
                job_id,
            )
            .await;
            state.capture_round(job_id, &round_agents, &run_root).await;
            round_agents.clear();
            let (_, pending) = state.acc.absorb(&r1.round);
            annotate_round(
                ws,
                &mut state.acc,
                &pending,
                &mut state.markers,
                job_id,
                question,
            )
            .await;
            // The directions the round examined are checkpointed with it, so a resumed run does not
            // re-report a direction it already worked as an open question.
            state.examined = r1.examined;
            state.budget_spent = budget.spent;
            state.stage = ResearchStage::GapRounds;
            state.save(job_id).await;
            // Wrap-up AFTER the checkpoint: a crash mid-wrap-up loses only the recovered findings
            // (ticket-accepted), never the round-1 evidence. Its ledger registrations are not in the
            // checkpoint (fail-open — a resumed ledger misses the dead analysts' queries).
            recovered.extend(
                wrap_up_timed_out(
                    ws,
                    r1.timed_out,
                    &mut state.ledger,
                    &mut run_stats,
                    job_id,
                    question,
                )
                .await,
            );
        } else {
            // A corrupt checkpoint can lose the plan. Nothing can be dispatched, so the run still
            // reports what it established: synthesis runs from the accumulated evidence and the
            // question itself is named among what stayed open.
            state.markers.push(PLAN_LOST_MARKER.to_string());
            state.stage = ResearchStage::Synthesis;
            state.save(job_id).await;
        }
    }

    if state.stage == ResearchStage::GapRounds {
        // Clone the plan so gap_rounds can take &mut state (per-round checkpoints) while still
        // referencing the merged decomposition. A corrupt checkpoint that lost the plan has nothing
        // to follow up on: the run reports what it established, with the question itself named among
        // what stayed open.
        if let Some(plan) = state.plan.clone() {
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
            // Aborting mid-gap-loop (drain/shutdown): leave the stage at GapRounds so the next
            // boot's resume CONTINUES the loop at the accumulated round_index (the per-round
            // checkpoints already persisted it plus the current gap list) instead of skipping the
            // remaining gap rounds and synthesizing from truncated evidence.
            if crate::shutdown::aborting() {
                state.gap_outcome = gap_outcome;
                state.budget_spent = budget.spent;
                state.save(job_id).await;
                return ResearchExit::Aborted;
            }
            if crate::round_cancel::is_cancelled(job_id) {
                // Manual cancel: permanent stop — the cancelled-run handoff owns the cleanup
                // dispatch; nothing further is checkpointed or delivered.
                return ResearchExit::Cancelled;
            }
            // Round-trip the gap-loop locals on NORMAL exit: round_index is already accumulated by
            // the per-round checkpoints inside gap_rounds — do NOT clobber it with the
            // per-invocation count.
            state.gap_outcome = gap_outcome;
            state.budget_spent = budget.spent;
            state.stage = ResearchStage::Synthesis;
            state.save(job_id).await;
        } else {
            state.markers.push(PLAN_LOST_MARKER.to_string());
            state.stage = ResearchStage::Synthesis;
            state.save(job_id).await;
        }
    }
    let rounds_used = 2 + state.gap_outcome.rounds_dispatched;

    // The gap loop may have exited early on shutdown/drain — never synthesize or verify then.
    if crate::shutdown::aborting() {
        return ResearchExit::Aborted;
    }

    if crate::round_cancel::is_cancelled(job_id) {
        return ResearchExit::Cancelled;
    }

    // The run's open questions — the plan directions never examined, the gaps left open and the
    // aspects the analysts reported as unanswered. Fed to synthesis so the answer can qualify its
    // conclusions, appended to the answer by the product, and listed in full in the detail file.
    // The compact section is rendered ONCE here: its bytes are measured into the answer ceiling and
    // the same text is appended, so what is measured and what is delivered cannot diverge.
    let open = open_questions(&state, &recovered, question);
    let open_section = render_open_section(&open);
    // Stage is always Synthesis here (set at the gap-loop exit or inherited on resume) — every
    // path re-synthesizes; the marker-dedup guard makes re-runs idempotent. A run that established
    // nothing has nothing to synthesize: its answer is built from the plan instead (never an empty
    // report, never an error envelope).
    let established =
        !state.acc.claims.is_empty() || !state.acc.raw_reports.is_empty() || !recovered.is_empty();
    // A synthesis output that could not be the answer (over-long or truncated) still reaches the
    // detail file whole — the answer is compact because the detail file exists, not by losing it.
    let mut undelivered_synthesis: Option<String> = None;
    let source = if established {
        match synthesize(
            ws,
            question,
            &state.acc,
            &recovered,
            state.gap_outcome.abstention.as_deref(),
            &open,
            answer_text_ceiling(&open_section, job_id),
            job_id,
        )
        .await
        {
            Ok(SynthesisOutcome::Answer { text, undelivered }) => {
                undelivered_synthesis = undelivered;
                AnswerSource::Synthesized(text)
            }
            Ok(SynthesisOutcome::Undelivered { reason, output }) => {
                undelivered_synthesis = output;
                // The answer names the real reason, and the run summary records it.
                let marker = format!("synthesis answer not delivered: {reason}");
                if !state.markers.contains(&marker) {
                    state.markers.push(marker);
                }
                AnswerSource::ProductBuilt(reason)
            }
            Err(_) if crate::shutdown::aborting() => {
                return ResearchExit::Aborted;
            }
            Err(_) if crate::round_cancel::is_cancelled(job_id) => {
                return ResearchExit::Cancelled;
            }
            Err(e) => AnswerSource::ProductBuilt(format!("synthesis failed: {e:#}")),
        }
    } else {
        AnswerSource::ProductBuilt(if budget.is_exhausted() {
            "analyst allowance exhausted before any evidence was gathered".to_string()
        } else {
            "no analyst produced usable findings".to_string()
        })
    };

    // Never spawn verifiers during shutdown or the drain — deliver the answer as it stands.
    let verification = if crate::shutdown::aborting() {
        Vec::new()
    } else if crate::round_cancel::is_cancelled(job_id) {
        return ResearchExit::Cancelled;
    } else if !state.verification.is_empty() {
        std::mem::take(&mut state.verification)
    } else {
        state.verification = research_verification_pass(
            ws,
            &state.acc,
            &mut budget,
            &mut state.ledger,
            &mut run_stats,
            deadline,
            resume,
            &run_root_str,
            &mut round_agents,
            job_id,
            question,
        )
        .await;
        state.capture_round(job_id, &round_agents, &run_root).await;
        round_agents.clear();
        state.budget_spent = budget.spent;
        state.save(job_id).await;
        std::mem::take(&mut state.verification)
    };

    let report = RunReport {
        state: &state,
        recovered: &recovered,
        verification: &verification,
        run_stats: &run_stats,
        budget: &budget,
        open: &open,
        open_section: &open_section,
        rounds_used,
        wall: start.elapsed(),
    };
    let mut output = report.output(source, job_id);
    if let Some(text) = undelivered_synthesis {
        let _ = writeln!(
            output.detail,
            "\n## Synthesis Output Not Delivered as the Answer\n\nThe answer is held to its size \
             and completeness target, so this output was kept here whole instead:\n\n{text}"
        );
    }
    if crate::jobs::job_retry_count(&crate::session::store().conn, job_id).await > 0 {
        let _ = writeln!(
            output.detail,
            "\n> Run telemetry is best-effort: this run resumed from a checkpoint, so tool-call \
             and query counts reflect only the post-resume segment."
        );
    }
    ResearchExit::Terminal(Ok(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Seed the durable job row + its `research_jobs` checkpoint for an end-to-end run.
    async fn seed_research_run(job_id: &str, ws: &Workspace, state: &ResearchState) {
        let conn = &crate::session::store().conn;
        crate::util::test::JobRowBuilder::new(conn, job_id, "research", "assistant", &ws.name)
            .task("question?")
            .user_name("caller-user")
            .channel("telegram")
            .timestamps(crate::db::now())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, state) VALUES (?1, ?2)",
            crate::db::params![job_id, serde_json::to_string(state).unwrap()],
        )
        .await
        .unwrap();
    }

    /// Drop the rows and the run folder [`seed_research_run`] left behind.
    async fn finish_research_run(job_id: &str) {
        let _ = tokio::fs::remove_dir_all(crate::research_cleanup::run_root_path(job_id)).await;
        crate::session::store()
            .conn
            .execute("DELETE FROM jobs WHERE id = ?1", crate::db::params![job_id])
            .await
            .unwrap();
    }

    fn claim(claim: &str, source: &str, confidence: &str) -> Claim {
        Claim {
            claim: claim.into(),
            source: source.into(),
            confidence: confidence.into(),
            contradictions: Vec::new(),
        }
    }

    /// The answer text of a delivered synthesis outcome (panics on an undelivered one).
    fn synthesis_answer(outcome: SynthesisOutcome) -> String {
        match outcome {
            SynthesisOutcome::Answer { text, .. } => text,
            SynthesisOutcome::Undelivered { reason, .. } => {
                panic!("expected a delivered answer, got: {reason}")
            }
        }
    }

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

        // The plan's allowance is what the cap leaves after the decomposers and the reserve (one
        // gap ladder plus the verification pass) — so a plan can never spend the run's ending.
        let mut budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
        budget.try_reserve(DECOMPOSE_FAN_OUT).unwrap();
        assert_eq!(
            plan_slot_budget(&budget),
            RESEARCH_MAX_ANALYSTS - DECOMPOSE_FAN_OUT - plan_slot_reserve()
        );
        assert!(
            plan_slot_budget(&budget) > 0,
            "a fresh run always has room for a plan"
        );
        // A cap too small to hold the reserve yields no plan allowance instead of stealing it.
        assert_eq!(
            plan_slot_budget(&ResearchBudget::new(plan_slot_reserve())),
            0
        );
    }

    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn resume_research_run_continues_at_synthesis_stage() {
        // Pre-collapse blobs stored stage="verification"; without this alias the state fails to
        // deserialize and the run silently restarts from Decompose.
        let legacy: ResearchStage =
            serde_json::from_str(r#""verification""#).expect("the old stage reads");
        assert_eq!(
            legacy,
            ResearchStage::Synthesis,
            "the collapsed stage is Synthesis"
        );
        assert_eq!(
            serde_json::to_string(&legacy).expect("the stage serializes"),
            r#""synthesis""#,
            "new checkpoints write the collapsed name"
        );

        crate::util::test::init_management_test_stores().await;
        // One synthesis call (the only LLM work left at stage=Synthesis). The answer is produced at
        // its target size, so the scripted report sits at the low end of it.
        let report = format!(
            "final synthesized report for the resumed run {}",
            "r".repeat(ANSWER_MIN_BYTES)
        );
        let fake = crate::util::test::FakeProvider::new().ok(&report);
        let _seam = crate::util::test::install_retry_seam(fake);

        let ws = crate::workspace::test_ws("/tmp/test_ws_research_resume");
        let job_id = "research_job_resume_1";
        let conn = &crate::session::store().conn;

        // Checkpointed state: stage=Synthesis, one accumulated claim, one stored verification
        // result (the post-crash resume must reuse it — never re-run the verification pass).
        let mut state = ResearchState {
            stage: ResearchStage::Synthesis,
            plan: None,
            gap_list: None,
            acc: AccumulatedEvidence {
                urls: std::collections::HashSet::new(),
                claims: vec![claim("alpha is a real project", "s1", "high")],
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
            verification: vec![crate::tools::analyze::VerificationResult {
                claim: "alpha is a real project".into(),
                verdict: "confirmed".into(),
                evidence: "primary source".into(),
                tool_calls: 0,
                searches: 0,
                queries: vec![],
            }],
            commands: vec![],
            seen_commands: std::collections::HashSet::new(),
            worker_rounds_done: vec![],
            examined: vec![],
            worker_reports: vec![],
        };
        state.acc.rebuild_keys();
        seed_research_run(job_id, &ws, &state).await;

        resume_research_run(job_id, &ws).await;

        // Terminalized into the durable envelope addressed to the stored caller (the consumer skips
        // it — the workspace is not managed, so the pending row survives for the assertion). The
        // surviving jobs row is the research_cleanup durability row (id == run_id == folder name).
        let job_rows = conn
            .query(
                "SELECT kind FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(
            job_rows.len(),
            1,
            "research job terminalized; the research_cleanup durability row remains"
        );
        assert_eq!(
            job_rows[0].get::<String>(0).unwrap(),
            "research_cleanup",
            "the surviving row is the cleanup durability row, not the research job"
        );
        let pending = conn
            .query(
                "SELECT envelope FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(pending.len(), 1, "resume envelope persisted");
        let envelope_json: String = pending[0].get(0).unwrap();
        let envelope: crate::agent::message_router::AgentJob =
            serde_json::from_str(&envelope_json).unwrap();
        assert_eq!(
            envelope.role,
            crate::Role::Assistant,
            "delivered to the original caller role, not Manager"
        );
        assert_eq!(envelope.user_name, "caller-user");
        assert!(
            envelope.content.contains("final synthesized report"),
            "the resume envelope must carry the synthesized report: {envelope_json}"
        );
    }

    /// A run whose synthesis never comes in under the answer ceiling still delivers an answer — the
    /// product-built one, naming the real reason and what stayed open — and the over-long output is
    /// kept whole in the detail file instead of being trimmed into the answer.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn an_over_long_synthesis_is_kept_in_the_detail_file_and_the_answer_still_ships() {
        crate::util::test::init_management_test_stores().await;
        // Every attempt is over the ceiling: the policy's three attempts exhaust.
        let over_long = format!("{} OVER-LONG-SYNTHESIS", "z".repeat(ANSWER_MAX_BYTES));
        let fake = crate::util::test::FakeProvider::new()
            .ok(&over_long)
            .ok(&over_long)
            .ok(&over_long);
        let _seam = crate::util::test::install_retry_seam(fake);

        let ws = crate::workspace::test_ws("/tmp/test_ws_research_overlong");
        let job_id = "research_job_overlong_1";
        let conn = &crate::session::store().conn;
        // Checkpointed at Synthesis with one claim and a stored verification result, so the only
        // LLM work left is the synthesis loop.
        let mut state = ResearchState {
            stage: ResearchStage::Synthesis,
            acc: AccumulatedEvidence {
                claims: vec![claim("alpha is a real project", "s1", "high")],
                ..AccumulatedEvidence::default()
            },
            verification: vec![crate::tools::analyze::VerificationResult {
                claim: "alpha is a real project".into(),
                verdict: "confirmed".into(),
                evidence: "primary source".into(),
                tool_calls: 0,
                searches: 0,
                queries: vec![],
            }],
            ..ResearchState::default()
        };
        state.acc.rebuild_keys();
        seed_research_run(job_id, &ws, &state).await;

        resume_research_run(job_id, &ws).await;

        let pending = conn
            .query(
                "SELECT envelope FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        let envelope_json: String = pending[0].get(0).unwrap();
        let envelope: crate::agent::message_router::AgentJob =
            serde_json::from_str(&envelope_json).unwrap();
        assert!(
            envelope.content.contains("over the")
                && envelope.content.contains("KB answer ceiling")
                && envelope
                    .content
                    .contains("### Questions This Run Could Not Settle"),
            "the answer names why synthesis was unavailable and what stayed open: {}",
            envelope.content
        );
        assert!(
            !envelope.content.contains("OVER-LONG-SYNTHESIS"),
            "an over-ceiling synthesis is never delivered as the answer"
        );
        // The answer part — not the envelope's own tag/job framing — stays within the ceiling.
        assert!(
            envelope.content.len() <= ANSWER_MAX_BYTES + 256,
            "the delivered answer stays within its ceiling, framing aside: {} bytes",
            envelope.content.len()
        );
        let detail_path = crate::research_cleanup::detail_archive_path(job_id)
            .expect("the detail file is written next to the answer");
        let detail = tokio::fs::read_to_string(&detail_path).await.unwrap();
        assert!(
            detail.contains("## Synthesis Output Not Delivered as the Answer")
                && detail.contains("OVER-LONG-SYNTHESIS"),
            "the undelivered synthesis is kept whole in the detail file"
        );
        assert!(
            detail.contains("alpha is a real project"),
            "the established finding is in the detail file"
        );

        let _ = tokio::fs::remove_file(&detail_path).await;
        finish_research_run(job_id).await;
    }

    /// Spawn-failure terminalization: when the spawn tx failed (no jobs row exists), the shared
    /// tail must STILL complete the durable job so the error envelope is delivered (never a
    /// silent drop), and must write no artifacts (no run state, no cleanup durability row). The
    /// pending row is NOT asserted on: the completion INSERT succeeds with no jobs row.
    #[tokio::test]
    async fn terminalization_spawn_failure_delivers_error_envelope_without_artifacts() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_spawn_fail");
        let envelope = terminalize_research(
            "research_job_spawn_fail_1",
            &ws,
            &Err(anyhow::anyhow!("spawn failed")),
            crate::Role::Assistant,
            "question?",
            "caller-user",
            "telegram",
            false,
            None,
        )
        .await
        .expect("spawn failure still delivers an error envelope");
        assert!(
            envelope.content.contains("An error occurred: spawn failed"),
            "{}",
            envelope.content
        );
        let conn = &crate::session::store().conn;
        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params!["research_job_spawn_fail_1"],
            )
            .await
            .unwrap();
        assert!(
            jobs.is_empty(),
            "no cleanup durability row without a spawned run"
        );
        let archive =
            crate::research_cleanup::results_archive_path("research_job_spawn_fail_1").unwrap();
        assert!(!archive.exists(), "spawn failure must not write results.md");
    }

    /// A fired manual-cancel signal must stop the orchestrator at its next stage boundary with
    /// `ResearchExit::Cancelled` (distinct from Aborted) — before synthesis, no LLM calls, no
    /// answer. The durable lifecycle is covered by research_cancel.rs tests.
    #[tokio::test]
    async fn cancelled_run_exits_cancelled_at_stage_boundary() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_cancel_boundary");
        let job_id = "research_job_cancel_boundary_1";
        let _guard = crate::round_cancel::register(job_id, None);
        crate::round_cancel::cancel(job_id);
        let exit = run_deep_research(&ws, "question?", job_id, true).await;
        assert!(
            matches!(exit, ResearchExit::Cancelled),
            "fired cancel signal must yield Cancelled at the boundary"
        );
    }

    /// Which research stage is driven into provider exhaustion.
    enum FailOpenStage {
        Synthesize,
        Annotate,
    }

    struct FailOpenCase {
        name: &'static str,
        stage: FailOpenStage,
    }

    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn provider_failure_fail_open_cases() {
        for case in [
            FailOpenCase {
                name: "synthesis never delivers a truncated or over-long answer",
                stage: FailOpenStage::Synthesize,
            },
            FailOpenCase {
                name: "a failed confirm pass keeps every mutating claim as weak",
                stage: FailOpenStage::Annotate,
            },
        ] {
            let _policy_guard =
                crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
            match case.stage {
                FailOpenStage::Synthesize => {
                    let ws =
                        crate::workspace::test_ws_named("/tmp/test_ws", "research_synth_truncated");
                    // Enough evidence to fill the answer target, so the below-target re-ask applies.
                    let acc = AccumulatedEvidence {
                        claims: (0..60)
                            .map(|i| claim(&format!("{i} {}", "e".repeat(200)), "s", "high"))
                            .collect(),
                        ..AccumulatedEvidence::default()
                    };
                    // A clean answer at the low end of the target — the size checks never touch it.
                    let compact = "y".repeat(ANSWER_MIN_BYTES);

                    // Every attempt provider-truncated: a half-sentence is never the answer; the
                    // last produced output is kept whole for the detail file.
                    let fake = crate::util::test::FakeProvider::new()
                        .ok_with_finish("report part one", Some("length"))
                        .ok_with_finish("report part two", Some("length"))
                        .ok_with_finish("compressed final", Some("length"));
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out =
                        synthesize(&ws, "q", &acc, &[], None, &[], ANSWER_MAX_BYTES, "test_run")
                            .await
                            .expect("truncated output is reported, not errored");
                    let SynthesisOutcome::Undelivered { reason, output } = out else {
                        panic!("[{}] a truncated output is not the answer", case.name);
                    };
                    assert_eq!(
                        output.as_deref(),
                        Some("compressed final"),
                        "[{}] the last produced output is kept for the detail file",
                        case.name
                    );
                    assert!(
                        reason.contains("truncated"),
                        "[{}] the product-built answer says why: {reason}",
                        case.name
                    );

                    // A clean (non-truncated) attempt wins immediately.
                    let fake = crate::util::test::FakeProvider::new()
                        .ok_with_finish("part one", Some("length"))
                        .ok(&compact);
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out =
                        synthesize(&ws, "q", &acc, &[], None, &[], ANSWER_MAX_BYTES, "test_run")
                            .await
                            .expect("clean completion wins");
                    let SynthesisOutcome::Answer { text, undelivered } = out else {
                        panic!("[{}] a clean attempt is the answer", case.name);
                    };
                    assert_eq!(text, compact, "[{}]", case.name);
                    assert_eq!(
                        undelivered.as_deref(),
                        Some("part one"),
                        "[{}] the truncated earlier attempt is kept for the detail file, not lost",
                        case.name
                    );

                    // All transport failures → no usable output → Err (the caller's product-built
                    // answer stands in).
                    let fake = crate::util::test::FakeProvider::new()
                        .err(crate::retry::FailureClass::Transport, "outage")
                        .err(crate::retry::FailureClass::Transport, "outage")
                        .err(crate::retry::FailureClass::Transport, "outage");
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let err =
                        synthesize(&ws, "q", &acc, &[], None, &[], ANSWER_MAX_BYTES, "test_run")
                            .await
                            .expect_err(
                                "transport exhaustion must error into the product-built answer",
                            );
                    assert!(
                        err.to_string().contains("no usable output"),
                        "[{}] {err}",
                        case.name
                    );

                    // The answer has a size of its own: an over-long attempt is re-produced with
                    // compression feedback (never trimmed), and the compliant retry wins.
                    let long = "x".repeat(ANSWER_MAX_BYTES + 1);
                    let fake = crate::util::test::FakeProvider::new()
                        .ok(&long)
                        .ok(&compact);
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out =
                        synthesize(&ws, "q", &acc, &[], None, &[], ANSWER_MAX_BYTES, "test_run")
                            .await
                            .expect("the retried answer is delivered");
                    assert_eq!(
                        synthesis_answer(out),
                        compact,
                        "[{}] the compliant retry is the answer",
                        case.name
                    );

                    // Every attempt over-long: the answer is never the over-long text — the
                    // product-built answer stands in and the over-long output is kept whole.
                    let fake = crate::util::test::FakeProvider::new()
                        .ok(&long)
                        .ok(&long)
                        .ok(&long);
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out =
                        synthesize(&ws, "q", &acc, &[], None, &[], ANSWER_MAX_BYTES, "test_run")
                            .await
                            .expect("the over-long output is reported, not errored");
                    let SynthesisOutcome::Undelivered { reason, output } = out else {
                        panic!("[{}] an over-long answer is never delivered", case.name);
                    };
                    assert_eq!(
                        output.as_deref(),
                        Some(long.as_str()),
                        "[{}] the over-long output is kept whole, never cut",
                        case.name
                    );
                    assert!(
                        reason.contains("ceiling"),
                        "[{}] the product-built answer says why: {reason}",
                        case.name
                    );

                    // Below the target: one re-ask (the answer may have left established findings
                    // out), and a still-short answer is delivered — a thin run answers briefly.
                    let fake = std::sync::Arc::new(
                        crate::util::test::FakeProvider::new()
                            .ok("brief answer")
                            .ok("still brief"),
                    );
                    let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out =
                        synthesize(&ws, "q", &acc, &[], None, &[], ANSWER_MAX_BYTES, "test_run")
                            .await
                            .expect("a short answer is still an answer");
                    assert_eq!(
                        synthesis_answer(out),
                        "still brief",
                        "[{}] a below-target answer is delivered, never padded",
                        case.name
                    );
                    assert!(
                        fake.request_messages.lock().unwrap()[1].contains("below the"),
                        "[{}] the re-ask names the target",
                        case.name
                    );

                    // Recovered findings are evidence: the answer is built from them.
                    let recovered = vec![AnalystFindings {
                        claims: vec![claim("recovered fact", "https://r", "medium")],
                        unanswered: Vec::new(),
                    }];
                    let fake =
                        std::sync::Arc::new(crate::util::test::FakeProvider::new().ok(&compact));
                    let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    synthesize(
                        &ws,
                        "q",
                        &acc,
                        &recovered,
                        None,
                        &[],
                        ANSWER_MAX_BYTES,
                        "test_run",
                    )
                    .await
                    .expect("clean completion wins");
                    assert!(
                        fake.request_messages.lock().unwrap()[0].contains("recovered fact"),
                        "[{}] the synthesis prompt carries the recovered findings",
                        case.name
                    );
                }
                FailOpenStage::Annotate => {
                    let ws = crate::workspace::test_ws_named(
                        "/tmp/test_ws",
                        "research_annotate_fail_open",
                    );
                    let mut acc = AccumulatedEvidence::default();
                    acc.claims.push(claim("alpha is true", "u1", "medium"));
                    let pending = vec![
                        claim("alpha is true (restated)", "u2", "high"),
                        claim("beta contradicts alpha", "u3", "high"),
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
                    let novel = annotate_round(
                        &ws,
                        &mut acc,
                        &pending,
                        &mut markers,
                        "test_run",
                        "question",
                    )
                    .await;
                    assert_eq!(
                        novel, 1,
                        "[{}] only the weak contradiction counts as novel — the weak duplicate never does",
                        case.name
                    );
                    assert_eq!(
                        acc.claims.len(),
                        3,
                        "[{}] both mutating claims stay standalone — never dropped",
                        case.name
                    );
                    assert_eq!(acc.weak.duplicates, vec![(1, 0)], "[{}]", case.name);
                    assert_eq!(acc.weak.contradictions, vec![(2, 0)], "[{}]", case.name);
                    assert!(
                        acc.claims[0]
                            .contradictions
                            .contains(&"alpha vs beta differ".to_string()),
                        "[{}] the weak contradiction keeps its note — verification still qualifies it",
                        case.name
                    );
                    assert!(
                        markers.iter().any(|m| m.contains("confirmation failed")),
                        "[{}] the confirm failure is never silent: {markers:?}",
                        case.name
                    );
                }
            }
        }
    }

    enum ValidationGate {
        GapTracesToPlan,
        MergedPlanCoverage,
    }

    struct ValidationGateCase {
        name: &'static str,
        gate: ValidationGate,
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn validation_gate_cases() {
        for case in [
            ValidationGateCase {
                name: "a gap's traces_to must index the merged plan",
                gate: ValidationGate::GapTracesToPlan,
            },
            ValidationGateCase {
                name: "every sub-question is placed exactly once",
                gate: ValidationGate::MergedPlanCoverage,
            },
        ] {
            match case.gate {
                ValidationGate::GapTracesToPlan => {
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
                    assert!(
                        validate_gap_list(&in_range, &plan).is_ok(),
                        "[{}]",
                        case.name
                    );
                    let out_of_range = GapList {
                        gaps: vec![Gap {
                            kind: "unanswered".into(),
                            item: "unrelated".into(),
                            traces_to: 5,
                        }],
                    };
                    assert!(
                        validate_gap_list(&out_of_range, &plan).is_err(),
                        "[{}] out-of-range traces_to is rejected — index-range validation guarantees traceability",
                        case.name
                    );
                }
                ValidationGate::MergedPlanCoverage => {
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
                    ];
                    // Global flat ids: plan 0: q1=0 q2=1; plan 1: q1=2 q3=3.
                    let entry = |from_id: usize, also_ids: Vec<usize>| MergedSubQuestion {
                        question: String::new(),
                        evidence_needed: String::new(),
                        risk: String::new(),
                        from_id,
                        also_ids,
                    };
                    let accepted = MergedPlan {
                        sub_questions: vec![entry(0, vec![2]), entry(1, vec![]), entry(3, vec![])],
                        dropped: vec![],
                    };
                    assert!(
                        validate_merged_plan(&accepted, &plans, 8).is_ok(),
                        "[{}] full coverage via from_id + also_ids",
                        case.name
                    );
                    // A drop of a genuine duplicate stays allowed.
                    let duplicate_drop = MergedPlan {
                        sub_questions: vec![entry(0, vec![]), entry(1, vec![]), entry(3, vec![])],
                        dropped: vec![DroppedSubQuestion { id: 2 }],
                    };
                    assert!(
                        validate_merged_plan(&duplicate_drop, &plans, 8).is_ok(),
                        "[{}] dropping a duplicate is the merge's normal job",
                        case.name
                    );
                    // Dropping distinct scope — the size-driven drop the plan may never make — is
                    // rejected: the direction must be folded in instead.
                    let distinct_drop = MergedPlan {
                        sub_questions: vec![entry(0, vec![2]), entry(1, vec![])],
                        dropped: vec![DroppedSubQuestion { id: 3 }],
                    };
                    let err = validate_merged_plan(&distinct_drop, &plans, 8)
                        .expect_err("a size-driven drop is rejected");
                    assert!(
                        err.contains("not a duplicate"),
                        "[{}] the rejection says why: {err}",
                        case.name
                    );
                    let dropout = MergedPlan {
                        sub_questions: vec![entry(0, vec![]), entry(1, vec![])],
                        dropped: vec![],
                    };
                    assert!(
                        validate_merged_plan(&dropout, &plans, 8).is_err(),
                        "[{}] silent dropout: plan 1's q1 (id 2) is never covered",
                        case.name
                    );
                    let mut bad = accepted.clone();
                    bad.sub_questions[0].from_id = 9;
                    assert!(
                        validate_merged_plan(&bad, &plans, 8).is_err(),
                        "[{}] out-of-range from_id is rejected",
                        case.name
                    );
                    let mut bad = accepted.clone();
                    bad.sub_questions[1].from_id = 0;
                    assert!(
                        validate_merged_plan(&bad, &plans, 8).is_err(),
                        "[{}] duplicate placement (id 0 twice) is rejected",
                        case.name
                    );
                    // The plan must fit the run's allowance: these three low-risk directions cost
                    // three analyst slots, so a two-slot allowance is rejected and the merge is told
                    // to fold them.
                    assert!(
                        validate_merged_plan(&accepted, &plans, 2).is_err(),
                        "[{}] an over-wide plan is rejected",
                        case.name
                    );
                    let too_wide =
                        validate_merged_plan(&accepted, &plans, 2).expect_err("over-wide plan");
                    assert!(
                        too_wide.contains("plan too wide"),
                        "[{}] the rejection says why: {too_wide}",
                        case.name
                    );
                }
            }
        }
    }

    #[test]
    fn test_evidence_absorb_counts_novelty() {
        // URLs and unanswered aspects dedup inside absorb exactly as before; claims come back as a
        // pending list — novelty is decided by the annotation pass, never dropped here.
        let mut acc = AccumulatedEvidence::default();
        let round1 = EvidenceRound {
            urls: vec!["u1".into(), "u2".into()],
            claims: vec![
                claim("alpha is true", "u1", "high"),
                claim("beta is true", "u2", "medium"),
            ],
            unanswered: vec!["how beta relates to alpha".into()],
            ..Default::default()
        };
        let (urls, pending) = acc.absorb(&round1);
        assert_eq!((urls, pending.len()), (2, 2));
        let round2 = EvidenceRound {
            urls: vec!["u1".into(), "u3".into()],
            claims: vec![
                claim("alpha is true", "u1", "high"),
                claim("gamma is true", "u3", "low"),
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

    /// Shared fixture for the verdict x confirm 2x2 matrix: one existing claim, one pending
    /// claim, a single-annotation pass linking new_id 0 to existing_id 0, and a Passed confirm
    /// pass with a single link for new_id 0.
    struct AnnotationsMatrix {
        acc: AccumulatedEvidence,
        pending: Vec<Claim>,
        pass: AnnotationPass,
        confirm: ConfirmOutcome,
    }

    fn annotations_matrix(
        existing: Claim,
        pending: Claim,
        verdict: &str,
        contradiction: Option<&str>,
        confirm_verdict: &str,
    ) -> AnnotationsMatrix {
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(existing);
        AnnotationsMatrix {
            acc,
            pending: vec![pending],
            pass: AnnotationPass {
                annotations: vec![ClaimAnnotation {
                    new_id: 0,
                    verdict: verdict.into(),
                    existing_id: Some(0),
                    contradiction: contradiction.map(Into::into),
                }],
            },
            confirm: ConfirmOutcome::Passed(ConfirmPass {
                links: vec![ConfirmLink {
                    new_id: 0,
                    verdict: confirm_verdict.into(),
                }],
            }),
        }
    }

    /// One verdict × confirm scenario: the two claims, the verdicts, and the applied result.
    struct AnnotationsCase {
        name: &'static str,
        existing: Claim,
        pending: Claim,
        verdict: &'static str,
        contradiction: Option<&'static str>,
        confirm: &'static str,
        novel: usize,
        claims_len: usize,
        existing_contradictions: &'static [&'static str],
        links_back_to: Option<&'static str>,
        weak_duplicates: &'static [(usize, usize)],
        weak_contradictions: &'static [(usize, usize)],
        merged_confidence: Option<&'static str>,
        merged_source: Option<&'static str>,
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn apply_annotations_cases() {
        for case in [
            AnnotationsCase {
                name: "a contradiction is appended and linked to the existing claim",
                existing: claim("alpha costs $100 in 2024", "u1", "medium"),
                pending: claim("alpha costs $200 in 2024", "u2", "high"),
                verdict: "contradicts",
                contradiction: Some("price differs: $200 vs $100"),
                confirm: "confirm",
                // A contradicting claim is new evidence, never deduped away.
                novel: 1,
                claims_len: 2,
                existing_contradictions: &["price differs: $200 vs $100"],
                links_back_to: Some("alpha costs $100 in 2024"),
                weak_duplicates: &[],
                weak_contradictions: &[],
                merged_confidence: None,
                merged_source: None,
            },
            AnnotationsCase {
                name: "a duplicate merges sources and upgrades confidence",
                existing: claim("alpha is true", "u1; u2", "low"),
                pending: claim("alpha is true", "u2; u3", "high"),
                verdict: "duplicate",
                contradiction: None,
                confirm: "confirm",
                novel: 0,
                claims_len: 1,
                existing_contradictions: &[],
                links_back_to: None,
                weak_duplicates: &[],
                weak_contradictions: &[],
                merged_confidence: Some("high"),
                merged_source: Some("u1; u2; u3"),
            },
            AnnotationsCase {
                name: "a weak duplicate stays standalone with a side-structure hint",
                existing: claim("alpha is true", "u1", "medium"),
                pending: claim("alpha is true (restated)", "u2", "high"),
                verdict: "duplicate",
                contradiction: None,
                confirm: "reject",
                novel: 0,
                claims_len: 2,
                existing_contradictions: &[],
                links_back_to: None,
                weak_duplicates: &[(1, 0)],
                weak_contradictions: &[],
                merged_confidence: None,
                merged_source: None,
            },
            AnnotationsCase {
                name: "a weak contradiction keeps both notes and is recorded unconfirmed",
                existing: claim("alpha costs $100 in 2024", "u1", "medium"),
                pending: claim("alpha costs $200 in 2024", "u2", "high"),
                verdict: "contradicts",
                contradiction: Some("price differs: $200 vs $100"),
                confirm: "reject",
                novel: 1,
                claims_len: 2,
                existing_contradictions: &["price differs: $200 vs $100"],
                links_back_to: Some("alpha costs $100 in 2024"),
                weak_duplicates: &[],
                weak_contradictions: &[(1, 0)],
                merged_confidence: None,
                merged_source: None,
            },
        ] {
            let mut m = annotations_matrix(
                case.existing,
                case.pending,
                case.verdict,
                case.contradiction,
                case.confirm,
            );
            let novel = m.acc.apply_annotations(&m.pass, &m.pending, &m.confirm);
            assert_eq!(novel, case.novel, "[{}] novel count", case.name);
            assert_eq!(
                m.acc.claims.len(),
                case.claims_len,
                "[{}] claim count",
                case.name
            );
            let expected_notes: Vec<String> = case
                .existing_contradictions
                .iter()
                .map(|note| (*note).to_string())
                .collect();
            assert_eq!(
                m.acc.claims[0].contradictions, expected_notes,
                "[{}] the existing claim's notes",
                case.name
            );
            if let Some(existing) = case.links_back_to {
                assert!(
                    m.acc.claims[1]
                        .contradictions
                        .contains(&existing.to_string()),
                    "[{}] the new claim must link back to the existing one",
                    case.name
                );
            }
            assert_eq!(
                m.acc.weak.duplicates, case.weak_duplicates,
                "[{}] weak duplicates",
                case.name
            );
            assert_eq!(
                m.acc.weak.contradictions, case.weak_contradictions,
                "[{}] weak contradictions",
                case.name
            );
            if let Some(confidence) = case.merged_confidence {
                assert_eq!(
                    m.acc.claims[0].confidence, confidence,
                    "[{}] merged confidence",
                    case.name
                );
            }
            if let Some(source) = case.merged_source {
                assert_eq!(
                    m.acc.claims[0].source, source,
                    "[{}] merged sources",
                    case.name
                );
            }
            assert!(
                m.acc
                    .claims
                    .iter()
                    .all(|c| c.contradictions.iter().all(|n| !n.contains("unconfirmed"))),
                "[{}] weakness never leaks into note text",
                case.name
            );
        }
    }

    #[test]
    fn test_verification_targets_primaries_first_weak_dups_fill_empty_slots() {
        let mut acc = AccumulatedEvidence::default();
        acc.claims.push(claim("a", "u1", "low"));
        acc.claims.push(Claim {
            claim: "b".into(),
            source: "u2".into(),
            confidence: "high".into(),
            contradictions: vec!["b vs c".into()],
        });
        // Weak duplicate of claim 0 — appended after primaries.
        acc.claims.push(claim("a restated", "u3", "high"));
        // Weak duplicate of claim 1 that is ALSO low confidence — already primary, must not be
        // double-appended.
        acc.claims.push(claim("b restated", "u4", "low"));
        acc.weak.duplicates.push((2, 0));
        acc.weak.duplicates.push((3, 1));
        // A weak contradiction must NEVER be appended from the side structure — it already carries
        // notes and qualifies via the primary filter.
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

    #[test]
    fn test_resolve_round_members_with_timeouts_preserves_timed_out() {
        // The TimedOut distinction survives until the wrap-up stage: only deadline-aborted members
        // are paired with their snapshots; panicked and cancelled members collapse to NoResponse.
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

        // An empty set renders nothing, and recovered findings get their own
        // section marked unverified — never merged into the evidence list.
        assert_eq!(
            render_recovered_findings(&[], RenderTarget::Delivered),
            "",
            "an empty set renders nothing"
        );
        let recovered = vec![AnalystFindings {
            claims: vec![Claim {
                claim: "found claim".into(),
                source: "u1".into(),
                confidence: "medium".into(),
                contradictions: vec!["counter".into()],
            }],
            unanswered: vec!["still open".into()],
        }];
        let out = render_recovered_findings(&recovered, RenderTarget::Delivered);
        assert!(
            out.starts_with("\n## Recovered from timed-out analysts"),
            "recovered findings get their own section: {out}"
        );
        assert!(
            !out.contains("### Evidence Gathered So Far"),
            "never merged into the evidence list: {out}"
        );
        assert!(
            out.contains("deadline exceeded, unverified"),
            "the section is marked unverified: {out}"
        );
        assert!(out.contains("found claim"), "the claim is carried: {out}");
        assert!(
            out.contains("contradictions: counter"),
            "the contradiction is carried: {out}"
        );
        // The analysts' unanswered aspects are the run's open questions, not a second copy here.
        // Without a plan the question itself is what stayed open.
        let state = ResearchState::default();
        let open = open_questions(&state, &recovered, "the original question");
        assert!(
            open.iter().any(|q| q == "still open")
                && open.iter().any(|q| q == "the original question"),
            "recovered unanswered aspects are named among the questions the run could not settle: {open:?}"
        );
        assert!(
            !out.contains("still open"),
            "the recovered section does not repeat them: {out}"
        );
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
    fn worker_round_lifecycle_skip_vs_claim_vs_unclaim() {
        let mut state = ResearchState::default();
        // Gate-skip: marked in the detail file, NOT claimed — the pre-loop key-0
        // round is re-attempted by boot-resume (post-progress keys are final:
        // the loop's round_index has advanced past them — fail-open per design).
        set_worker_marker(&mut state, 0, "skipped — analyst budget exhausted");
        assert!(
            state
                .markers
                .iter()
                .any(|m| m == "worker round 0 skipped — analyst budget exhausted")
        );
        assert!(!state.worker_rounds_done.contains(&0));
        // Dispatch: claimed; a stale skip marker (a previous gate-skip that
        // was later re-attempted) is cleared — the detail file shows one truth.
        claim_worker_round(&mut state, 0);
        assert!(state.worker_rounds_done.contains(&0));
        assert!(!state.markers.iter().any(|m| m.contains("worker round 0 ")));
        // Failure: un-claimed (only key-0 is re-attempted by resume); outcome
        // marked — and the outcome marker CLEARS a prior skip marker for the
        // same key (one truth: a failed-then-gate-skipped round must not
        // render both).
        unclaim_worker_round(&mut state, 0);
        set_worker_marker(&mut state, 0, "failed");
        assert!(!state.worker_rounds_done.contains(&0));
        assert!(state.markers.iter().any(|m| m == "worker round 0 failed"));
        assert!(
            !state
                .markers
                .iter()
                .any(|m| m.contains("worker round 0 skipped")),
            "outcome marker supersedes the stale skip marker"
        );
        // A later successful dispatch clears the stale outcome marker too.
        claim_worker_round(&mut state, 0);
        assert!(!state.markers.iter().any(|m| m.contains("worker round 0 ")));
        // Other rounds' markers are untouched.
        set_worker_marker(&mut state, 1, "skipped — round deadline expired");
        claim_worker_round(&mut state, 0);
        assert!(
            state
                .markers
                .iter()
                .any(|m| m == "worker round 1 skipped — round deadline expired")
        );
        assert!(!state.markers.iter().any(|m| m.contains("worker round 0 ")));
        // A gate-skip after a failure of the SAME key supersedes the failure
        // marker (one truth per key in the answer).
        set_worker_marker(&mut state, 1, "skipped — round deadline expired");
        assert_eq!(
            state
                .markers
                .iter()
                .filter(|m| m.starts_with("worker round 1 "))
                .count(),
            1,
            "skip re-push dedupes per key"
        );
    }

    #[tokio::test]
    async fn load_seeds_commands_from_dump_after_crash() {
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_dump_reload";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        // The durable research job row (research_jobs.id FKs jobs.id).
        crate::util::test::JobRowBuilder::new(conn, job_id, "research", "assistant", "ws")
            .task("question?")
            .user_name("caller-user")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();
        // A valid checkpointed state blob (no command history in it).
        let json = serde_json::to_string(&ResearchState::default()).unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, state) VALUES (?1, ?2)",
            crate::db::params![job_id, json],
        )
        .await
        .unwrap();
        // Pre-crash command history in the run folder's dump.
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        crate::research_cleanup::write_command_dump(
            &run_root,
            job_id,
            &["pre-crash cmd".to_string()],
        )
        .await;
        let loaded = ResearchState::load(job_id).await;
        assert_eq!(
            loaded.commands,
            vec!["pre-crash cmd".to_string()],
            "load() must seed commands from the dump"
        );
        assert!(
            loaded.seen_commands.contains("pre-crash cmd"),
            "seen-set rebuilt from the dump — post-resume dedup keeps the pre-crash capture"
        );
        // Clean up the run folder AND the DB rows the test created (research_jobs
        // FKs jobs(id) ON DELETE CASCADE — deleting the jobs row removes both).
        finish_research_run(job_id).await;
    }

    /// The plan's slot cost is its directions plus a second analyst for each high-risk one.
    fn plan_of(risks: &[&str]) -> MergedPlan {
        MergedPlan {
            sub_questions: risks
                .iter()
                .enumerate()
                .map(|(i, risk)| MergedSubQuestion {
                    question: format!("direction {i}"),
                    evidence_needed: format!("evidence {i}"),
                    risk: (*risk).to_string(),
                    from_id: i,
                    also_ids: Vec::new(),
                })
                .collect(),
            dropped: Vec::new(),
        }
    }

    #[test]
    fn test_plan_prefix_dispatches_what_the_allowance_carries() {
        // A plan that fits is dispatched whole, and its cost is its slot total.
        assert_eq!(plan_prefix(&[1, 1, 2, 1], 10), (4, 5));
        // The allowance covers the whole round, not each direction: the rest stays unexamined.
        assert_eq!(
            plan_prefix(&[1, 1, 1, 1, 1], 3),
            (3, 3),
            "round 1 spends at most the allowance in total"
        );
        assert_eq!(
            plan_prefix(&[2, 2, 2], 3),
            (1, 2),
            "a high-risk direction costs two slots"
        );
        // No room at all: nothing is dispatched, every direction stays unexamined.
        assert_eq!(plan_prefix(&[1, 1], 0), (0, 0));
        assert_eq!(plan_prefix(&[], 10), (0, 0));
    }

    #[test]
    fn test_fit_plan_to_slots_merges_instead_of_dropping() {
        // Ten low-risk directions cost ten slots; fitting to four pairs them into four directions.
        let mut plan = plan_of(&["low"; 10]);
        fit_plan_to_slots(&mut plan, 4);
        assert!(
            plan_slot_cost(&plan) <= 4,
            "the fitted plan fits the allowance: {}",
            plan_slot_cost(&plan)
        );
        assert_eq!(plan.sub_questions.len(), 3, "pairing halves until it fits");
        let provenance: Vec<usize> = plan
            .sub_questions
            .iter()
            .flat_map(|sq| std::iter::once(sq.from_id).chain(sq.also_ids.iter().copied()))
            .collect();
        let mut sorted = provenance.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted,
            (0..10).collect::<Vec<_>>(),
            "every direction stays accounted for — merged, never dropped"
        );
        assert_eq!(
            provenance.len(),
            10,
            "no direction is cited twice by the fitted plan"
        );
        assert!(
            plan.sub_questions[0].question.contains("direction 0")
                && plan.sub_questions[0].question.contains("direction 1"),
            "a merged direction carries both: {}",
            plan.sub_questions[0].question
        );
        assert_eq!(
            plan.sub_questions[0].risk, "low",
            "the merged risk is the higher of the two"
        );

        // A high-risk pair folds into one high-risk direction (two slots, not four).
        let mut plan = plan_of(&["high", "high"]);
        fit_plan_to_slots(&mut plan, 2);
        assert_eq!(plan_slot_cost(&plan), 2);
        assert_eq!(plan.sub_questions.len(), 1);
        assert_eq!(plan.sub_questions[0].risk, "high");

        // A plan that already fits is untouched.
        let mut plan = plan_of(&["low", "high"]);
        fit_plan_to_slots(&mut plan, 3);
        assert_eq!(plan.sub_questions.len(), 2);
    }

    /// A merge fold is expressed by citing several ids on one entry: the resolved direction keeps
    /// every distinct cited item's question and evidence need and the highest cited risk — folding
    /// loses nothing, and an identical duplicate is named once.
    #[test]
    fn resolve_merged_plan_ids_composes_a_fold_without_losing_a_direction() {
        let sq = |q: &str, e: &str, r: &str| SubQuestion {
            question: q.into(),
            evidence_needed: e.into(),
            risk: r.into(),
        };
        let plans = vec![
            DecompositionPlan {
                sub_questions: vec![
                    sq("how widgets are styled", "css docs", "low"),
                    sq("widget accessibility", "a11y guide", "medium"),
                ],
            },
            DecompositionPlan {
                sub_questions: vec![sq("how widgets are styled", "css docs", "low")],
            },
        ];
        // Global flat ids: 0 = styled, 1 = accessibility, 2 = styled (the other plan's duplicate).
        let mut plan = MergedPlan {
            sub_questions: vec![MergedSubQuestion {
                from_id: 0,
                also_ids: vec![1, 2],
                question: String::new(),
                evidence_needed: String::new(),
                risk: String::new(),
            }],
            dropped: Vec::new(),
        };
        resolve_merged_plan_ids(&mut plan, &plans);
        let resolved = &plan.sub_questions[0];
        assert_eq!(
            resolved.question, "how widgets are styled; widget accessibility",
            "the folded-in direction keeps its text"
        );
        assert_eq!(
            resolved.question.matches("how widgets are styled").count(),
            1,
            "an identical duplicate is named once"
        );
        assert_eq!(resolved.evidence_needed, "css docs; a11y guide");
        assert_eq!(
            resolved.risk, "medium",
            "a fold carries the highest cited risk — width is never understated"
        );
    }

    /// The merge-failure fallback keeps every decomposer's scope: all valid plans' directions in
    /// one plan (exact duplicates once), still fitted to the allowance.
    #[test]
    fn merge_failure_fallback_keeps_every_decomposer_scope() {
        let sq = |q: &str| SubQuestion {
            question: q.into(),
            evidence_needed: "e".into(),
            risk: "low".into(),
        };
        let plans = vec![
            DecompositionPlan {
                sub_questions: vec![sq("a"), sq("b")],
            },
            DecompositionPlan {
                sub_questions: vec![sq("b"), sq("c")],
            },
        ];
        let mut plan = concatenated_plan(&plans);
        let questions: Vec<&str> = plan
            .sub_questions
            .iter()
            .map(|s| s.question.as_str())
            .collect();
        assert_eq!(
            questions,
            vec!["a", "b", "c"],
            "both plans' directions are kept, the exact duplicate once"
        );
        assert_eq!(
            plan.sub_questions
                .iter()
                .map(|s| s.from_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 3],
            "provenance stays the input item's global flat id"
        );
        fit_plan_to_slots(&mut plan, 2);
        assert_eq!(plan_slot_cost(&plan), 2, "the fallback is fitted too");
    }

    /// The answer is never empty: a run whose allowance was spent before round 1 still names what
    /// it could not settle — the plan's unsettled directions — and refers to the detail file.
    #[tokio::test]
    async fn exhausted_run_still_delivers_an_answer_naming_the_unsettled_directions() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_exhausted");
        let job_id = "research_job_exhausted_1";
        // Stage Round1 with the whole allowance already spent and an empty gap list: no analyst is
        // dispatched and no orchestrator call is needed, so the run has to answer from the plan.
        let state = ResearchState {
            stage: ResearchStage::Round1,
            plan: Some(plan_of(&["low", "high"])),
            gap_list: Some(GapList::default()),
            budget_spent: RESEARCH_MAX_ANALYSTS,
            ..ResearchState::default()
        };
        seed_research_run(job_id, &ws, &state).await;

        let exit = run_deep_research(&ws, "question?", job_id, true).await;
        let ResearchExit::Terminal(Ok(output)) = exit else {
            panic!("an exhausted run still terminalizes with an answer");
        };
        assert!(
            output.answer.contains("analyst allowance exhausted"),
            "the answer names why it is incomplete: {}",
            output.answer
        );
        assert!(
            output.answer.contains("direction 0") && output.answer.contains("direction 1"),
            "every unsettled direction is named as an open question: {}",
            output.answer
        );
        assert!(
            !output.answer.contains("\n- none"),
            "an empty report is never a valid outcome: {}",
            output.answer
        );
        // The reference is appended where the detail file is written, so the answer can never point
        // at a file that does not exist.
        let state = ResearchState::load(job_id).await;
        let answer =
            write_terminalization_artifacts(job_id, &output.answer, &output.detail, &state).await;
        assert!(
            answer.contains(".detail.md"),
            "the delivered answer refers to the detail file: {answer}"
        );
        assert!(
            output.detail.contains("## Open Questions")
                && output.detail.contains("direction 0")
                && output.detail.contains("direction 1"),
            "the detail file carries the full open-question list: {}",
            output.detail
        );
        assert!(
            output.detail.contains("## Run Summary"),
            "run bookkeeping is detail, not answer: {}",
            output.detail
        );
        assert!(
            !output.answer.contains("## Run Summary"),
            "the answer stays compact: {}",
            output.answer
        );

        finish_research_run(job_id).await;
    }

    /// A plan wider than the run's allowance is still worked: round 1 dispatches only the prefix the
    /// allowance carries, the remaining direction stays open, and the run still reaches its answer.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn an_over_wide_plan_is_worked_as_far_as_the_allowance_carries_and_still_answers() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_allowance");
        let job_id = "research_job_allowance_1";
        let findings = r#"{"claims":[{"claim":"alpha is true","source":"https://a","confidence":"high","contradictions":[]}],"unanswered":[]}"#;
        let fake = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                .ok("alpha is true") // the dispatched analyst's turn
                .ok(findings) // its structured extraction
                .ok("the synthesized answer"), // synthesis
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _seam = crate::util::test::install_retry_seam_dyn(provider);

        // The run can spend ONE slot on the plan: of two low-risk directions only the first is
        // dispatched, the second never. The gap list is empty (coverage completion), so no gap
        // or worker round adds calls.
        let state = ResearchState {
            stage: ResearchStage::Round1,
            plan: Some(plan_of(&["low", "low"])),
            gap_list: Some(GapList::default()),
            budget_spent: RESEARCH_MAX_ANALYSTS - plan_slot_reserve() - 1,
            ..ResearchState::default()
        };
        seed_research_run(job_id, &ws, &state).await;

        let exit = run_deep_research(&ws, "question?", job_id, true).await;
        let ResearchExit::Terminal(Ok(output)) = exit else {
            panic!("the over-wide plan must still terminalize with an answer");
        };
        assert!(
            output.answer.contains("the synthesized answer"),
            "the run reached synthesis from the direction it could work: {}",
            output.answer
        );
        assert!(
            output.detail.contains("alpha is true"),
            "the dispatched direction's evidence reached the run: {}",
            output.detail
        );
        assert!(
            output.detail.contains("direction 1"),
            "the direction the allowance could not carry stays open: {}",
            output.detail
        );
        // One analyst = its turn + its extraction; a second dispatched direction would add two more.
        assert_eq!(
            fake.request_messages.lock().unwrap().len(),
            3,
            "only the direction the allowance covers was dispatched"
        );

        finish_research_run(job_id).await;
    }

    /// A failure reason can be an entire error chain: the product-built answer's header is capped
    /// like every other line, so the reason is named — never dropped for being too long — without
    /// crowding the answer out.
    #[test]
    fn a_long_failure_reason_is_named_in_a_capped_header() {
        let state = ResearchState {
            acc: AccumulatedEvidence {
                claims: vec![claim("alpha holds", "s", "high")],
                ..AccumulatedEvidence::default()
            },
            ..ResearchState::default()
        };
        let open: Vec<String> = Vec::new();
        let open_section = render_open_section(&open);
        let budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
        let report = RunReport {
            state: &state,
            recovered: &[],
            verification: &[],
            run_stats: &RunStats::default(),
            budget: &budget,
            open: &open,
            open_section: &open_section,
            rounds_used: 1,
            wall: Duration::ZERO,
        };
        let reason = format!("synthesis failed: {}", "x".repeat(ANSWER_MAX_BYTES));
        let out = report.output(AnswerSource::ProductBuilt(reason), "job");
        let header = out.answer.lines().next().expect("the header is present");
        assert!(
            header.starts_with("## Deep Analysis Answer (incomplete — synthesis failed: ")
                && header.len() <= ANSWER_LINE_MAX_BYTES,
            "the reason is named in a capped header: {} bytes",
            header.len()
        );
        assert!(
            out.answer.contains("alpha holds"),
            "the established finding is still reported"
        );
    }

    /// The answer stays inside its size ceiling — text, open-questions section and detail
    /// reference together — even when the evidence is huge, every line it shows is capped, and the
    /// detail file still holds everything the answer left out.
    #[test]
    fn product_built_answer_respects_the_size_ceiling() {
        let mut state = ResearchState {
            // 200 directions, none examined: every one is an open question, each over the line cap.
            plan: Some(MergedPlan {
                sub_questions: (0..200)
                    .map(|i| MergedSubQuestion {
                        question: format!("open question {i} {}", "y".repeat(500)),
                        evidence_needed: String::new(),
                        risk: "low".to_string(),
                        from_id: i,
                        also_ids: Vec::new(),
                    })
                    .collect(),
                dropped: Vec::new(),
            }),
            ..ResearchState::default()
        };
        state.acc.claims = (0..200)
            .map(|i| claim(&format!("{i} {}", "x".repeat(2000)), "s", "high"))
            .collect();
        let open = open_questions(&state, &[], "the question");
        let open_section = render_open_section(&open);
        let budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
        let report = RunReport {
            state: &state,
            recovered: &[],
            verification: &[],
            run_stats: &RunStats::default(),
            budget: &budget,
            open: &open,
            open_section: &open_section,
            rounds_used: 1,
            wall: Duration::ZERO,
        };
        let out = report.output(
            AnswerSource::ProductBuilt("no evidence was gathered".to_string()),
            "job",
        );
        // The delivered answer is the text plus the section plus the reference the terminalization
        // appends — the whole of it stays within the ceiling.
        let delivered_len = out.answer.len() + detail_reference_len("job");
        assert!(
            delivered_len <= ANSWER_MAX_BYTES,
            "the answer stays at its own size, reference included: {delivered_len} bytes"
        );
        assert!(
            out.answer
                .contains("more, listed in full in the detail file"),
            "what the answer leaves out is named: {}",
            out.answer
        );
        // The first item is shown despite being far over the per-line cap: a single huge finding
        // can never degrade a section to a bare count.
        assert!(
            out.answer.contains("0 xxxx")
                && !out.answer.contains(&"x".repeat(ANSWER_LINE_MAX_BYTES)),
            "every shown line is capped"
        );
        assert!(
            out.answer
                .contains("### Questions This Run Could Not Settle")
                && out.answer.contains("open question 0"),
            "the open questions are named compactly, the rest deferred"
        );
        assert!(
            out.detail.contains("open question 199") && out.detail.contains("199 xxxx"),
            "the detail file keeps everything"
        );
    }

    /// The detail file is the home of everything the answer is not: verification verdicts and the
    /// worker round's output ("what was tested in practice") included.
    #[test]
    fn detail_file_carries_verification_and_the_worker_output() {
        let mut state = ResearchState::default();
        state
            .worker_reports
            .push("prototype built and ran".to_string());
        let verification = vec![VerificationResult {
            claim: "alpha is true".to_string(),
            verdict: "confirmed".to_string(),
            evidence: "two sources agree".to_string(),
            tool_calls: 1,
            searches: 1,
            queries: Vec::new(),
        }];
        let open = open_questions(&state, &[], "the question");
        let open_section = render_open_section(&open);
        let budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
        let report = RunReport {
            state: &state,
            recovered: &[],
            verification: &verification,
            run_stats: &RunStats::default(),
            budget: &budget,
            open: &open,
            open_section: &open_section,
            rounds_used: 1,
            wall: Duration::ZERO,
        };
        let out = report.output(AnswerSource::Synthesized("the answer".to_string()), "job");
        assert_eq!(out.answer.lines().next(), Some("the answer"));
        assert!(
            !out.answer.contains("## Verification") && !out.answer.contains("Tested in Practice"),
            "detail material never enters the answer: {}",
            out.answer
        );
        assert!(
            out.detail.contains("## Verification")
                && out
                    .detail
                    .contains("alpha is true → **confirmed** — two sources agree"),
            "verification verdicts are detail: {}",
            out.detail
        );
        assert!(
            out.detail.contains("## Tested in Practice")
                && out.detail.contains("prototype built and ran"),
            "the worker round's output is detail: {}",
            out.detail
        );
    }

    /// The two artifacts land side by side: the answer at `{job}.md` and its detail at
    /// `{job}.detail.md` in the same directory, rewritten (never appended) on resume; an error
    /// terminalization writes no detail file.
    #[tokio::test]
    async fn terminalization_writes_the_answer_and_its_detail_side_by_side() {
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_job_artifacts_1";
        let state = ResearchState::default();
        let delivered =
            write_terminalization_artifacts(job_id, "the answer", "the detail", &state).await;
        let answer = crate::research_cleanup::results_archive_path(job_id).unwrap();
        let detail = crate::research_cleanup::detail_archive_path(job_id).unwrap();
        assert_eq!(
            answer.parent(),
            detail.parent(),
            "the detail sits in the same place the answer is kept"
        );
        assert!(
            delivered.contains(&detail.display().to_string()),
            "the answer refers to the detail file it was written with: {delivered}"
        );
        let answer_text = tokio::fs::read_to_string(&answer).await.unwrap();
        assert!(
            answer_text.starts_with("# Research research_job_artifacts_1")
                && answer_text.contains("the answer")
                && answer_text.contains(&detail.display().to_string()),
            "{answer_text}"
        );
        assert_eq!(
            tokio::fs::read_to_string(&detail).await.unwrap().trim_end(),
            "the detail"
        );

        // Resume rewrites rather than appends.
        write_terminalization_artifacts(job_id, "the answer", "the detail", &state).await;
        assert_eq!(
            tokio::fs::read_to_string(&detail)
                .await
                .unwrap()
                .matches("the detail")
                .count(),
            1,
            "the detail file is rewritten, not appended to"
        );

        // An error terminalization (no run output) writes no companion file.
        let other = "research_job_artifacts_2";
        write_terminalization_artifacts(other, "the answer", "", &state).await;
        assert!(
            crate::research_cleanup::results_archive_path(other)
                .unwrap()
                .exists()
        );
        assert!(
            !crate::research_cleanup::detail_archive_path(other)
                .unwrap()
                .exists(),
            "empty detail content writes nothing"
        );

        for id in [job_id, other] {
            for path in [
                crate::research_cleanup::results_archive_path(id),
                crate::research_cleanup::detail_archive_path(id),
            ] {
                let _ = tokio::fs::remove_file(path.unwrap()).await;
            }
            let _ = tokio::fs::remove_dir_all(crate::research_cleanup::run_root_path(id)).await;
        }
    }

    /// A merge drop is judged by word coverage, not by substring: a re-phrased duplicate is
    /// recognised and an unrelated direction that merely appears inside a kept one is not.
    #[test]
    fn duplicate_drop_is_word_coverage() {
        let kept = vec!["How widgets are styled in CSS".to_string()];
        assert!(is_duplicate_drop("how widgets are styled", &kept));
        assert!(
            is_duplicate_drop("Widgets, styled how?", &kept),
            "case, punctuation and word order do not matter"
        );
        assert!(
            !is_duplicate_drop("grid layout for widgets", &kept),
            "distinct scope is never a duplicate drop"
        );
        assert!(
            !is_duplicate_drop("   ", &kept),
            "an empty drop is not a duplicate"
        );
    }

    /// Open questions are derived from the plan against what round 1 examined (a gap round's
    /// narrower follow-up does not settle a direction), and a checkpoint written before the
    /// examined list existed names every direction instead of claiming nothing stayed open.
    #[test]
    fn open_questions_follow_the_plan_and_what_was_examined() {
        let state = ResearchState {
            plan: Some(plan_of(&["low", "high"])),
            ..ResearchState::default()
        };
        assert_eq!(
            open_questions(&state, &[], "q"),
            vec!["direction 0", "direction 1"],
            "an unexamined direction is always named"
        );
        let state = ResearchState {
            examined: vec![0],
            ..state
        };
        assert_eq!(
            open_questions(&state, &[], "q"),
            vec!["direction 1"],
            "an examined direction is no longer open"
        );
        let state = ResearchState {
            examined: vec![0, 1],
            ..state
        };
        assert!(open_questions(&state, &[], "q").is_empty());
        assert_eq!(
            open_questions(&ResearchState::default(), &[], "the question"),
            vec!["the question"],
            "without a plan the question itself is what stayed open"
        );
    }

    /// A synthesized answer names what stayed open structurally — the product appends the section,
    /// not the model — and the whole delivered answer (text + section + reference) fits the ceiling.
    #[test]
    fn a_synthesized_answer_carries_the_product_built_open_section() {
        let state = ResearchState {
            plan: Some(plan_of(&["low", "low"])),
            examined: vec![0],
            ..ResearchState::default()
        };
        let open = open_questions(&state, &[], "q");
        let open_section = render_open_section(&open);
        let budget = ResearchBudget::new(RESEARCH_MAX_ANALYSTS);
        let report = RunReport {
            state: &state,
            recovered: &[],
            verification: &[],
            run_stats: &RunStats::default(),
            budget: &budget,
            open: &open,
            open_section: &open_section,
            rounds_used: 1,
            wall: Duration::ZERO,
        };
        // A model answer at the exact text budget the synthesis loop is given.
        let text = "y".repeat(answer_text_ceiling(&open_section, "job"));
        let out = report.output(AnswerSource::Synthesized(text), "job");
        assert!(
            out.answer.starts_with("yyyy")
                && out
                    .answer
                    .contains("### Questions This Run Could Not Settle")
                && out.answer.contains("direction 1"),
            "the synthesized answer names what stayed open: {}",
            &out.answer[out.answer.len() - 500..]
        );
        assert!(
            !out.answer.contains("direction 0"),
            "an examined direction is not reported open"
        );
        assert!(
            out.answer.len() + detail_reference_len("job") <= ANSWER_MAX_BYTES,
            "text + section + reference fit the ceiling: {} bytes",
            out.answer.len() + detail_reference_len("job")
        );
    }

    /// The claim line is one renderer with two outputs: the model gets the analyst's text verbatim
    /// as evidence, the reader gets it fence-escaped; and the per-line cap counts the newline.
    #[test]
    fn claim_lines_are_verbatim_for_prompts_and_escaped_for_the_reader() {
        let acc = AccumulatedEvidence {
            claims: vec![claim("uses ``` fences", "s", "high")],
            ..AccumulatedEvidence::default()
        };
        let mut prompt = String::new();
        render_claims(&mut prompt, &acc, RenderTarget::Prompt);
        assert!(prompt.contains("uses ``` fences"), "{prompt}");
        let mut delivered = String::new();
        render_claims(&mut delivered, &acc, RenderTarget::Delivered);
        assert!(!delivered.contains("```"), "{delivered}");
        let capped = cap_answer_line("- 0123456789\n", 6);
        assert!(
            capped.len() <= 6 && capped.ends_with('\n'),
            "{capped:?} is {} bytes, newline included",
            capped.len()
        );
    }

    /// A checkpoint that lost its plan still ends with an answer: nothing can be dispatched, so the
    /// run reports what it established and names the question itself among what stayed open.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn a_checkpoint_without_a_plan_still_delivers_an_answer() {
        crate::util::test::init_management_test_stores().await;
        let fake =
            crate::util::test::FakeProvider::new().ok("answer from the accumulated evidence");
        let _seam = crate::util::test::install_retry_seam(fake);

        let ws = crate::workspace::test_ws("/tmp/test_ws_research_plan_lost");
        let job_id = "research_job_plan_lost_1";
        // A checkpoint at round 1 with no plan — a corrupt state blob.
        let mut state = ResearchState {
            stage: ResearchStage::Round1,
            acc: AccumulatedEvidence {
                claims: vec![claim("alpha is a real project", "s1", "high")],
                ..AccumulatedEvidence::default()
            },
            verification: vec![crate::tools::analyze::VerificationResult {
                claim: "alpha is a real project".into(),
                verdict: "confirmed".into(),
                evidence: "primary source".into(),
                tool_calls: 0,
                searches: 0,
                queries: vec![],
            }],
            ..ResearchState::default()
        };
        state.acc.rebuild_keys();
        seed_research_run(job_id, &ws, &state).await;

        let exit = run_deep_research(&ws, "question?", job_id, true).await;
        let ResearchExit::Terminal(Ok(output)) = exit else {
            panic!("a plan-less checkpoint still ends with an answer");
        };
        assert!(
            output
                .answer
                .contains("answer from the accumulated evidence"),
            "the answer is built from what the run established: {}",
            output.answer
        );
        assert!(
            output.answer.contains("question?") && output.answer.contains("Could Not Settle"),
            "the question itself is named as what stayed open: {}",
            output.answer
        );
        let state = ResearchState::load(job_id).await;
        assert!(
            state.markers.iter().any(|m| m == PLAN_LOST_MARKER),
            "the lost plan is recorded: {:?}",
            state.markers
        );

        finish_research_run(job_id).await;
    }

    /// The primary round-0 path end to end: two decomposers, a merge that is accepted on its first
    /// attempt (no fallback), round 1 working the merged plan, an empty gap list, a synthesis — the
    /// run carries the merge's plan and reaches its answer.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn an_accepted_merge_is_worked_through_to_an_answer() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_merge_ok");
        let job_id = "research_job_merge_ok_1";
        // Each decomposer proposes the same direction; the merge folds the two into one and is
        // accepted, so the fallback marker must never appear.
        let decomposed = r#"{"sub_questions":[{"question":"alpha direction","evidence_needed":"a docs","risk":"low"}]}"#;
        let merged = r#"{"sub_questions":[{"from_id":0,"also_ids":[1]}],"dropped":[]}"#;
        let findings = r#"{"claims":[{"claim":"alpha holds","source":"https://a","confidence":"high","contradictions":[]}],"unanswered":[]}"#;
        let fake = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                // Two decomposers (turn + extraction each, any interleaving).
                .ok(decomposed)
                .ok(decomposed)
                .ok(decomposed)
                .ok(decomposed)
                // The merge is accepted on its first attempt.
                .ok(merged)
                // Round 1: one low-risk direction, turn + extraction.
                .ok(findings)
                .ok(findings)
                // Gap extraction: coverage complete.
                .ok("{\"gaps\":[]}")
                .ok("the synthesized answer"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _seam = crate::util::test::install_retry_seam_dyn(provider);

        let state = ResearchState::default();
        seed_research_run(job_id, &ws, &state).await;

        let exit = run_deep_research(&ws, "question?", job_id, true).await;
        let ResearchExit::Terminal(Ok(output)) = exit else {
            panic!("an accepted merge must reach an answer");
        };
        assert!(
            output.answer.contains("the synthesized answer"),
            "the run worked the merged plan and reached synthesis: {}",
            output.answer
        );
        assert!(
            output.answer.contains("- no open questions were recorded"),
            "the merged direction was examined: {}",
            output.answer
        );
        assert!(
            output.detail.contains("alpha holds"),
            "the finding is in the detail file"
        );
        let state = ResearchState::load(job_id).await;
        assert!(
            !state.markers.iter().any(|m| m == PLAN_MERGE_FAILED),
            "the merge was accepted, not fallen back from: {:?}",
            state.markers
        );
        let plan = state.plan.expect("the plan is checkpointed");
        assert_eq!(
            plan.sub_questions.len(),
            1,
            "the merge's fold is the plan the run carried: {:?}",
            plan.sub_questions
        );
        assert_eq!(plan.sub_questions[0].from_id, 0);
        assert_eq!(
            plan.sub_questions[0].also_ids,
            vec![1],
            "both proposed directions are accounted for by the fold"
        );

        finish_research_run(job_id).await;
    }

    /// The round-0 path end to end: two decomposers, a merge that never yields a valid plan, and a
    /// fallback that keeps every direction and fits the plan to the allowance by folding neighbours
    /// — the run works the plan and reaches an answer instead of refusing the round.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn merge_failure_fits_the_plan_to_the_allowance_and_still_answers() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_round0");
        let job_id = "research_job_round0_1";
        // Four low-risk directions; the allowance below leaves room for two, so the fallback has to
        // fold neighbours rather than drop them.
        let plan = r#"{"sub_questions":[
            {"question":"alpha direction","evidence_needed":"a docs","risk":"low"},
            {"question":"beta direction","evidence_needed":"b docs","risk":"low"},
            {"question":"gamma direction","evidence_needed":"c docs","risk":"low"},
            {"question":"delta direction","evidence_needed":"d docs","risk":"low"}]}"#;
        let findings = r#"{"claims":[{"claim":"alpha holds","source":"https://a","confidence":"high","contradictions":[]}],"unanswered":[]}"#;
        let fake = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                // Two decomposers (turn + extraction each, any interleaving).
                .ok(plan)
                .ok(plan)
                .ok(plan)
                .ok(plan)
                // The merge never yields a valid plan: its retries exhaust into the fallback.
                .ok("not a plan")
                .ok("not a plan")
                .ok("not a plan")
                // Two analysts: turn + extraction each, any interleaving.
                .ok(findings)
                .ok(findings)
                .ok(findings)
                .ok(findings)
                // Gap extraction: coverage complete.
                .ok("{\"gaps\":[]}")
                .ok("the synthesized answer"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _seam = crate::util::test::install_retry_seam_dyn(provider);

        // Two decomposer slots plus two plan slots are all the run can carry: after round 0's
        // reserve, `plan_slot_budget` is exactly 2.
        let state = ResearchState {
            stage: ResearchStage::Decompose,
            budget_spent: RESEARCH_MAX_ANALYSTS - plan_slot_reserve() - DECOMPOSE_FAN_OUT - 2,
            ..ResearchState::default()
        };
        seed_research_run(job_id, &ws, &state).await;

        let exit = run_deep_research(&ws, "question?", job_id, true).await;
        let ResearchExit::Terminal(Ok(output)) = exit else {
            panic!("a failed merge must still end with an answer");
        };
        assert!(
            output.answer.contains("the synthesized answer"),
            "the run worked the fitted plan and reached synthesis: {}",
            output.answer
        );
        assert!(
            output.answer.contains("- no open questions were recorded"),
            "every fitted direction was examined: {}",
            output.answer
        );
        // The plan the run carried is the fallback's, fitted to the allowance by folding.
        let state = ResearchState::load(job_id).await;
        assert!(
            state.markers.iter().any(|m| m == PLAN_MERGE_FAILED),
            "the fallback is recorded: {:?}",
            state.markers
        );
        let plan = state.plan.expect("the plan is checkpointed");
        assert_eq!(
            plan.sub_questions.len(),
            2,
            "the fallback is fitted to the two plan slots: {:?}",
            plan.sub_questions
        );
        assert!(
            plan.sub_questions[0].question.contains("alpha direction")
                && plan.sub_questions[0].question.contains("beta direction")
                && plan.sub_questions[1].question.contains("gamma direction")
                && plan.sub_questions[1].question.contains("delta direction"),
            "no direction is dropped to fit: {:?}",
            plan.sub_questions
        );
        // Two decomposers (4 calls), the merge's exhausted retries (3), two analysts (4), the gap
        // extraction and the synthesis.
        assert_eq!(
            fake.request_messages.lock().unwrap().len(),
            13,
            "the run worked exactly the plan the allowance covers"
        );

        finish_research_run(job_id).await;
    }

    /// A run whose decomposition failed outright still delivers an answer — it reports that nothing
    /// was established and names the question itself as what stayed open, never an error envelope.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn a_total_decomposition_failure_still_delivers_an_answer() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_nodecompose");
        let job_id = "research_job_nodecompose_1";
        // Both decomposers' extractions exhaust their retries on unparseable output.
        let fake = crate::util::test::FakeProvider::new()
            .ok("not a plan")
            .ok("not a plan")
            .ok("not a plan")
            .ok("not a plan")
            .ok("not a plan")
            .ok("not a plan")
            .ok("not a plan")
            .ok("not a plan");
        let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
        let _seam = crate::util::test::install_retry_seam_dyn(provider);

        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        crate::util::test::JobRowBuilder::new(conn, job_id, "research", "assistant", &ws.name)
            .task("why is the sky blue?")
            .user_name("caller-user")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();

        let exit = run_deep_research(&ws, "why is the sky blue?", job_id, true).await;
        let ResearchExit::Terminal(Ok(output)) = exit else {
            panic!("a total decomposition failure still ends with an answer");
        };
        assert!(
            output.answer.contains("could not be decomposed")
                && output.answer.contains("nothing was established"),
            "the answer reports what it did establish: {}",
            output.answer
        );
        assert!(
            output.answer.contains("why is the sky blue?"),
            "the question itself is named as what stayed open: {}",
            output.answer
        );
        assert!(
            !output.answer.contains("- none"),
            "an empty report is never a valid outcome: {}",
            output.answer
        );
        assert!(
            output.detail.contains("## Open Questions"),
            "the detail file carries the open-question list: {}",
            output.detail
        );

        finish_research_run(job_id).await;
    }
}
