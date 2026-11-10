//! The deep-analysis arm of [`AnalyzeTool`](super::analyze::AnalyzeTool) (its `deep`
//! depth): the multi-round research orchestrator.
//!
//! Round 0 decomposes the question (two independent plans merged by id-based coverage), round 1
//! runs an analyst per sub-question (high-risk items get a decorrelated second angle), then
//! conditional gap rounds of shrinking width. A worker-in-loop prototype pass runs before the
//! first gap round and after every progress round that refreshes a non-empty gap list, gated by
//! budget, deadline, shutdown/cancel and `WORKER_MIN_REMAINING`; skips are fail-open markers.
//!
//! Stopping is artifact-based (coverage completion, answerability abstention, a hard
//! analyst-spawn cap — never self-assessment) plus budget exhaustion, deadline expiry, shutdown
//! and manual cancel. One wall-clock deadline (`round_timeout`, default 3h) bounds every phase's
//! waits; the budget counts analysts spawned, not coordination calls, and is never refunded.
//! Exactly one envelope reaches the caller, and exhaustion delivers a partial report. The
//! wrap-up stage recovers deadline-aborted analysts by replaying their dispatch-time params
//! (the KV-cache prefix).

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
/// Gap-round dispatch widths — rounds shrink as they progress.
const GAP_ROUND_WIDTHS: &[usize] = &[4, 3, 2];
/// Explicit marker when the orchestrator cannot determine the remaining gaps.
const GAP_EXTRACTION_FAILED: &str = "gap extraction failed — remaining gaps unknown";
/// Explicit marker when the plan merge fails — the run falls back to the first plan.
const PLAN_MERGE_FAILED: &str = "plan merge failed — using first valid decomposition plan verbatim";
/// Explicit marker when the annotation pass exhausts its retries — claims are novel.
const CLAIM_ANNOTATION_FAILED: &str = "claim annotation failed — all new claims treated as novel";
/// Explicit marker when the confirm pass fails — every mutating verdict is weak.
const CONFIRM_FAILED: &str =
    "annotation link confirmation failed — mutating links treated as weak/unconfirmed";
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
/// report's "Run Summary" undercounts the pre-crash segment and says so.
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

/// A merged sub-question carrying provenance by global flat item id: the input-plan item
/// it copies verbatim, plus every other plan's identical tuple. question/evidence_needed/
/// risk are resolved by the system from the cited item — never trusted from the model.
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

/// A round-0 plan item the merge explicitly dropped (never silently), by flat id.
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

/// Dispatch a deep analysis run — the `deep` depth of [`AnalyzeTool`]: decompose the
/// question, run analyst rounds against the remaining gaps, deliver one source-cited
/// report. Spawns the resumable background run and returns the dispatch ack; exactly one
/// report envelope is delivered to the caller when the run completes.
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
        "Deep analysis dispatched (job {job_id}). One report will arrive when the run completes."
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

/// The Sanitation cleanup is NOT dispatched here — [`terminalize_research`] does it after
/// `complete_durable_job` (the cleanup jobs row reuses id == run_id, so the completion
/// DELETE must have run first).
async fn write_terminalization_artifacts(job_id: &str, delivered: &str, state: &ResearchState) {
    crate::research_cleanup::write_results_md(job_id, delivered).await;
    let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
    crate::research_cleanup::write_command_dump(&run_root, job_id, &state.commands).await;
}

/// Real-terminalization tail shared by fresh dispatch and boot resume: artifacts written BEFORE
/// the exactly-once boundary, then durable completion, then cleanup dispatch (the cleanup row
/// reuses id == run_id, so the DELETE must free it first). `Some` on a real terminalization, `None`
/// on a manual cancel (the cancelled-run handoff dispatches the cleanup agent).
#[expect(clippy::too_many_arguments)]
async fn terminalize_research(
    job_id: &str,
    ws: &Workspace,
    result: &anyhow::Result<String>,
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
    let delivered = build_async_research_message(job_id, result);
    // Terminalization artifacts BEFORE the exactly-once boundary, only for runs that started.
    // The aborted flag (not the global shutdown state) already gated aborts above: a shutdown
    // after a fully successful run must not skip the artifacts.
    if spawned {
        let state = ResearchState::load(job_id).await;
        write_terminalization_artifacts(job_id, &delivered, &state).await;
    }
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
    // the cancel fired mid-tx — never route the report or dispatch the cleanup.
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
/// caller-generated `job_id` is embedded so the delivered report correlates with the
/// immediate ack; failures are wrapped with an explicit marker, never silently dropped.
fn build_async_research_message(job_id: &str, result: &anyhow::Result<String>) -> String {
    build_async_result_envelope(job_id, result, "analyze-tool-result", None)
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
    /// Set when the remaining gaps could not be determined — the report must carry an
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
    /// the fail-open report, plus telemetry (queries must reach the ledger so later rounds
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
/// decision). On merge failure the run falls back to the first valid plan verbatim with a marker;
/// only when ALL decomposers failed does it error.
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
    if let Ok(mut plan) = merge_decomposition_plans(ws, question, &valid, run_key).await {
        resolve_merged_plan_ids(&mut plan, &valid);
        Ok((plan, None))
    } else {
        // Fail-open: a failed merge must never lose the run — fall back to the first valid plan
        // verbatim with an explicit marker.
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

/// Merge 1–2 independent plans into one consolidated plan with provenance (not budgeted).
async fn merge_decomposition_plans(
    ws: &Workspace,
    question: &str,
    plans: &[DecompositionPlan],
    run_key: &str,
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

/// Fill in merged entries' tuple text from the cited input-plan items — the model's
/// copied text is never used.
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

/// Validate a merged plan by id-based coverage: every cited id in range, no duplicate
/// placement, and full coverage — every input plan item appears exactly once across
/// merged sub-questions (as `from_id` or `also_ids`) plus dropped. Silent dropout is
/// rejected (fail-closed inside the extraction retry loop).
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

/// Round 1: one analyst per sub-question, two for high-risk items (the second gets a
/// decorrelated angle). Returns the round's evidence plus dispatch-time snapshots of
/// deadline-aborted analysts (the caller checkpoints FIRST, then wraps up — a crash
/// mid-wrap-up must not lose the evidence), or `None` when the budget is exhausted.
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
    // Dispatch-time wrap-up snapshots: frozen params + advertised tool schemas, captured before
    // spawn (aborted tasks lose their values). The specs are constant across the round's members
    // (same role+workspace).
    let wrap_up_specs = role_tool_specs(Role::Analyst, ws, false);
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
                run_key.to_string(),
                question.clone(),
            ));
        }
    }
    let (runs, timed_out) = dispatch_analyst_round(members, resume, deadline, &snapshots).await;
    // collect_evidence runs first so the wrap-up's ledger registrations can
    // never skew this round's saturation counters (round.queries /
    // repeat_queries stay from successful analysts only).
    let round = collect_evidence(&runs, ledger, run_stats);
    Some((round, timed_out))
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

/// The gap items as plain strings (for the report's unresolved list).
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
/// report note).
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
/// the following gap rounds, fail-open). The response text is NOT inserted into the report.
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
    let evidence = render_accumulated_evidence(&state.acc);
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
    if response.is_some() {
        tracing::info!(job = %job_id, worker_round = round_key, "Worker round completed");
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
/// report telemetry count; they diverge only on the abstention / gap-extract-failure exit,
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
            job_id,
        )
        .await;
        state
            .capture_round(job_id, &round_agents, Path::new(run_root))
            .await;
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
    let evidence = render_accumulated_evidence(acc);
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

/// Marker surfaced in the head-placed `## Run markers` section when the final synthesis
/// was delivered provider-truncated — never silent success.
const SYNTHESIS_TRUNCATED_MARKER: &str =
    "final synthesis truncated by the provider — last produced output delivered";

/// Delivered synthesis: the report text plus an optional truncation marker.
#[derive(Debug)]
struct SynthesisOutput {
    text: String,
    marker: Option<String>,
}

/// Final synthesis with the shared policy (≤3 informed attempts, transport-only backoff; the 600 s
/// idle timeout is the only per-attempt bound). A truncated, empty or failed attempt counts toward
/// the budget; the last produced output wins and is delivered with a marker when truncated.
#[expect(clippy::too_many_lines)]
async fn synthesize(
    ws: &Workspace,
    question: &str,
    acc: &AccumulatedEvidence,
    abstention: Option<&str>,
    run_key: &str,
) -> Result<SynthesisOutput> {
    let _call = research_call_guard("synthesize", &ws.name, run_key, false, question);
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
    let mut loop_state = crate::retry::RetryLoop::new_scoped(&policy, &params);
    let operation_started = Instant::now();
    let mut prefix = Vec::with_capacity(2);
    crate::prompt::prepend_general_context(&mut prefix, ws).await;
    let mut last: Option<String> = None;
    let mut last_truncated = false;
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
                    let err = anyhow::anyhow!(
                        "synthesis truncated by the provider (finish_reason=length)"
                    );
                    let rec = crate::retry::RetryFailureRecord::with_metadata(
                        FailureClass::TruncatedOutput,
                        &err,
                        resp.finish_reason.clone(),
                        None,
                    );
                    loop_state.record(rec);
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
                    let rec = crate::retry::RetryFailureRecord::with_metadata(
                        FailureClass::NoResponse,
                        &err,
                        resp.finish_reason.clone(),
                        None,
                    );
                    loop_state.record(rec);
                    feedback = "Your previous attempt returned an empty response — \
                                produce the report now."
                        .to_string();
                    continue;
                }
                // Clean completion wins.
                crate::stats::record_llm_success(&params, operation_started, attempt, &resp);
                return Ok(SynthesisOutput { text, marker: None });
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
    // Exhausted: the last produced output wins (marked when truncated); no usable output at
    // all errors into the caller's partial-report fail-open.
    let final_class = loop_state.final_class();
    let exhausted = crate::retry::RetryExhausted::with_last_raw(
        loop_state.into_failures(),
        final_class,
        last.clone(),
    );
    crate::stats::record_llm_failure(&params, operation_started, &exhausted);
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
/// unresolved in the final report. Verifier tool calls / searches / queries
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
/// empty when none). Shared by the evidence view, the annotate listing and the partial
/// report — weakness is per-claim visible and never silent.
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

/// Unanswered section; `escape` fences entries for the partial report (prompt views stay raw).
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

/// Render the accumulated evidence as a compact numbered list plus analysts' unanswered
/// aspects. Used by the orchestrator prompts and the worker brief (`run_worker_round`).
/// Claim ids are their stable 0-based indices in `acc.claims`.
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

#[expect(clippy::too_many_arguments)]
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

/// Render the recovered-from-timed-out-analysts section. Separate from the main evidence by
/// design: recovered findings never entered verification or synthesis. Empty when there is
/// nothing recovered; head-placed by callers so it survives report truncation.
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

/// Partial envelope when a run cannot reach synthesis (round 1 skipped on budget
/// exhaustion, or a synthesis failure): whatever evidence was gathered is delivered with an
/// explicit incomplete marker — findings are never lost. Shutdown/abort paths never produce
/// this report.
fn partial_report(
    acc: &AccumulatedEvidence,
    reason: &str,
    recovered: &[AnalystFindings],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Research Report (incomplete — {reason})");
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

enum ResearchExit {
    /// Real terminalization (success, partial, or error) — the caller writes results.md + the
    /// command dump and completes the durable job.
    Terminal(anyhow::Result<String>),
    /// Shutdown/drain abort — the run stays alive for the next boot; nothing is written or routed.
    Aborted,
    /// Manual cancel (Running Agents page): a PERMANENT stop — the run must never resume,
    /// re-dispatch or deliver a report, now or after any restart. The caller hands the record to
    /// the cancelled-run handoff, which decides what is held. Distinct from
    /// [`Aborted`](ResearchExit::Aborted).
    Cancelled,
}

/// Run the resumable deep-research orchestrator: round 0, round 1, conditional gap rounds,
/// one-shot synthesis, verification gate and run summary. `job_id` names the durable job
/// whose `research_jobs.state` checkpoint is loaded on entry and saved at every boundary.
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
                // decompose failure must not lose the agents' sessions from the report.
                state.save(job_id).await;
                return ResearchExit::Terminal(Err(e));
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
        let Some(plan) = state.plan.as_ref() else {
            return ResearchExit::Terminal(Err(super::internal_fault(
                "research lost its plan state before round 1",
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
            job_id,
        )
        .await
        else {
            return ResearchExit::Terminal(Ok(partial_report(
                &state.acc,
                "round 1 skipped — analyst budget exhausted",
                &recovered,
            )));
        };
        state.capture_round(job_id, &round_agents, &run_root).await;
        round_agents.clear();
        let (_, pending) = state.acc.absorb(&r1);
        annotate_round(
            ws,
            &mut state.acc,
            &pending,
            &mut state.markers,
            job_id,
            question,
        )
        .await;
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::GapRounds;
        state.save(job_id).await;
        // Wrap-up AFTER the checkpoint: a crash mid-wrap-up loses only the recovered findings
        // (ticket-accepted), never the round-1 evidence. Its ledger registrations are not in the
        // checkpoint (fail-open — a resumed ledger misses the dead analysts' queries).
        recovered.extend(
            wrap_up_timed_out(
                ws,
                r1_timed_out,
                &mut state.ledger,
                &mut run_stats,
                job_id,
                question,
            )
            .await,
        );
    }

    if state.stage == ResearchStage::GapRounds {
        // Clone the plan so gap_rounds can take &mut state (per-round checkpoints) while still
        // referencing the merged decomposition.
        let Some(plan) = state.plan.clone() else {
            return ResearchExit::Terminal(Err(super::internal_fault(
                "research lost its plan state before gap rounds",
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
        // Aborting mid-gap-loop (drain/shutdown): leave the stage at GapRounds so the next boot's
        // resume CONTINUES the loop at the accumulated round_index (the per-round checkpoints
        // already persisted it plus the current gap list) instead of skipping the remaining gap
        // rounds and synthesizing from truncated evidence.
        if crate::shutdown::aborting() {
            state.gap_outcome = gap_outcome;
            state.budget_spent = budget.spent;
            state.save(job_id).await;
            return ResearchExit::Aborted;
        }
        if crate::round_cancel::is_cancelled(job_id) {
            // Manual cancel: permanent stop — the cancelled-run handoff owns the cleanup dispatch;
            // nothing further is checkpointed or delivered.
            return ResearchExit::Cancelled;
        }
        // Round-trip the gap-loop locals on NORMAL exit: round_index is already accumulated by the
        // per-round checkpoints inside gap_rounds — do NOT clobber it with the per-invocation
        // count.
        state.gap_outcome = gap_outcome;
        state.budget_spent = budget.spent;
        state.stage = ResearchStage::Synthesis;
        state.save(job_id).await;
    }
    let rounds_used = 2 + state.gap_outcome.rounds_dispatched;

    // The gap loop may have exited early on shutdown/drain — never synthesize or verify then.
    if crate::shutdown::aborting() {
        return ResearchExit::Aborted;
    }

    if crate::round_cancel::is_cancelled(job_id) {
        return ResearchExit::Cancelled;
    }

    // Stage is always Synthesis here (set at the gap-loop exit or inherited on resume) — every
    // path re-synthesizes; the marker-dedup guard makes re-runs idempotent.
    let synthesis = match synthesize(
        ws,
        question,
        &state.acc,
        state.gap_outcome.abstention.as_deref(),
        job_id,
    )
    .await
    {
        Ok(s) => {
            if let Some(marker) = s.marker
                && !state.markers.contains(&marker)
            {
                state.markers.push(marker);
            }
            s.text
        }
        Err(_) if crate::shutdown::aborting() => {
            return ResearchExit::Aborted;
        }
        Err(_) if crate::round_cancel::is_cancelled(job_id) => {
            return ResearchExit::Cancelled;
        }
        Err(e) => {
            return ResearchExit::Terminal(Ok(partial_report(
                &state.acc,
                &format!("synthesis failed: {e}"),
                &recovered,
            )));
        }
    };

    // Never spawn verifiers during shutdown or the drain — deliver the
    // synthesized report as-is (partial is acceptable).
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

    // Fail-open markers survive delivery: head-placed so they survive the
    // caller's sandwich truncation of long reports. The recovered-findings
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
    use serde_json::json;

    fn claim(claim: &str, source: &str, confidence: &str) -> Claim {
        Claim {
            claim: claim.into(),
            source: source.into(),
            confidence: confidence.into(),
            contradictions: Vec::new(),
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
        // One synthesis call (the only LLM work left at stage=Synthesis).
        let fake = crate::util::test::FakeProvider::new()
            .ok("final synthesized report for the resumed run");
        let _seam = crate::util::test::install_retry_seam(fake);

        let ws = crate::workspace::test_ws("/tmp/test_ws_research_resume");
        let job_id = "research_job_resume_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        crate::util::test::JobRowBuilder::new(conn, job_id, "research", "assistant", &ws.name)
            .task("question?")
            .user_name("caller-user")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();

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
        };
        state.acc.rebuild_keys();
        let state_json = serde_json::to_string(&state).unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, state) VALUES (?1, ?2)",
            crate::db::params![job_id, state_json],
        )
        .await
        .unwrap();

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
    /// partial report. The durable lifecycle is covered by research_cancel.rs tests.
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
                name: "synthesis delivers the last produced output and marks truncation",
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
                    let acc = AccumulatedEvidence::default();

                    // Every attempt provider-truncated: the last produced output wins, marked.
                    let fake = crate::util::test::FakeProvider::new()
                        .ok_with_finish("report part one", Some("length"))
                        .ok_with_finish("report part two", Some("length"))
                        .ok_with_finish("compressed final", Some("length"));
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out = synthesize(&ws, "q", &acc, None, "test_run")
                        .await
                        .expect("last produced output must be delivered");
                    assert_eq!(out.text, "compressed final", "[{}]", case.name);
                    assert!(
                        out.marker
                            .as_deref()
                            .is_some_and(|m| m.contains("truncated")),
                        "[{}] truncated delivery must carry the explicit marker: {:?}",
                        case.name,
                        out.marker
                    );

                    // A clean (non-truncated) attempt wins immediately without a marker.
                    let fake = crate::util::test::FakeProvider::new()
                        .ok_with_finish("part one", Some("length"))
                        .ok("complete report");
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let out = synthesize(&ws, "q", &acc, None, "test_run")
                        .await
                        .expect("clean completion wins");
                    assert_eq!(out.text, "complete report", "[{}]", case.name);
                    assert!(
                        out.marker.is_none(),
                        "[{}] clean completion is unmarked",
                        case.name
                    );

                    // All transport failures → no usable output → Err (the caller's partial-report
                    // fail-open).
                    let fake = crate::util::test::FakeProvider::new()
                        .err(crate::retry::FailureClass::Transport, "outage")
                        .err(crate::retry::FailureClass::Transport, "outage")
                        .err(crate::retry::FailureClass::Transport, "outage");
                    let provider: std::sync::Arc<dyn crate::Provider> = std::sync::Arc::new(fake);
                    let _provider_guard = crate::util::test::install_fake_provider(provider);
                    let err = synthesize(&ws, "q", &acc, None, "test_run")
                        .await
                        .expect_err("transport exhaustion must error into the partial-report path");
                    assert!(
                        err.to_string().contains("no usable output"),
                        "[{}] {err}",
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
                        ],
                        dropped: if dropped {
                            vec![DroppedSubQuestion { id: 3 }]
                        } else {
                            vec![]
                        },
                    };
                    assert!(
                        validate_merged_plan(&base(true, true), &plans).is_ok(),
                        "[{}] full coverage via from_id + also_ids + dropped",
                        case.name
                    );
                    assert!(
                        validate_merged_plan(&base(false, true), &plans).is_err(),
                        "[{}] silent dropout: plan 1's q1 (id 2) is never covered",
                        case.name
                    );
                    assert!(
                        validate_merged_plan(&base(true, false), &plans).is_err(),
                        "[{}] silent dropout: q3 (id 3) is never covered",
                        case.name
                    );
                    let mut bad = base(true, true);
                    bad.sub_questions[0].from_id = 9;
                    assert!(
                        validate_merged_plan(&bad, &plans).is_err(),
                        "[{}] out-of-range from_id is rejected",
                        case.name
                    );
                    let mut bad = base(true, true);
                    bad.sub_questions[1].from_id = 0;
                    assert!(
                        validate_merged_plan(&bad, &plans).is_err(),
                        "[{}] duplicate placement (id 0 twice) is rejected",
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
            render_recovered_findings(&[]),
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
        let out = render_recovered_findings(&recovered);
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
        assert!(
            out.contains("still open"),
            "the unanswered question is carried: {out}"
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
        // Gate-skip: marked in the report, NOT claimed — the pre-loop key-0
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
        // was later re-attempted) is cleared — the report shows one truth.
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
        // marker (one truth per key in the final report).
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
        let _ = tokio::fs::remove_dir_all(crate::research_cleanup::run_root_path(job_id)).await;
        conn.execute("DELETE FROM jobs WHERE id = ?1", crate::db::params![job_id])
            .await
            .unwrap();
    }
}
