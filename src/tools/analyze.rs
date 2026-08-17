//! AnalyzeTool — dispatches a batch of parallel analysts to research a question.
//!
//! Analyst-only: every call dispatches analysts (read-only research), never
//! mutation-capable agents. Available to the Engineer and Maintainer agents
//! (sync mode), and to the Manager and Assistant agents (async mode). In sync
//! mode the caller blocks until the sub-agents complete. In async mode the
//! sub-agents are dispatched in a background task and the result is injected
//! back to the caller's agent channel via [`crate::message_router::route`].
//!
//! Analyst batches run three decorrelated analysts (distinct research angles)
//! that report structured claim-level findings; consolidation runs the shared
//! LLM grouping pass ([`crate::consensus`]) — semantic grouping + contradiction
//! judgment with code-computed agreement brackets. The verification pass for
//! disputed claims lives in the deep research tool only. Fail-open is
//! preserved throughout: findings are never silently lost.

use crate::agent::{run_agent, run_default_agent};
use crate::config::CONFIG;
use crate::message_router::{self, AgentJob, JobKind};
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::tools::Tool;
use crate::{Agent, ChatMessage, DEFAULT_MAX_TOKENS, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::Write as _;
use std::time::Duration;

// ── Constants (module-local defaults) ────────────────────────────────────

/// Number of decorrelated parallel analysts spawned per analyze round (durable and
/// sync pipelines share the same batch width).
const PARALLEL_ANALYST_COUNT: usize = 3;

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

pub struct AnalyzeTool {
    /// Controls how the sub-agents are dispatched.
    /// - [`DispatchMode::Sync`] — blocks the caller until the sub-agents complete.
    /// - [`DispatchMode::Async`] — dispatches in a background task, result
    ///   delivered via the caller's agent queue.
    dispatch_mode: DispatchMode,
    /// The role of the calling agent. Used to route async results to the
    /// correct agent channel (Manager → manager_{ws}, Assistant → direct_{...}).
    pub caller_role: Role,
}

impl AnalyzeTool {
    #[must_use]
    pub const fn new(dispatch_mode: DispatchMode, caller_role: Role) -> Self {
        Self {
            dispatch_mode,
            caller_role,
        }
    }
}

#[async_trait]
impl Tool for AnalyzeTool {
    fn name(&self) -> &'static str {
        "analyze"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "analyze": {
                    "type": "string",
                    "description": "The question to delegate to the analysts"
                }
            }),
            &["analyze"],
        )
    }

    /// Sub-agents dispatched by AnalyzeTool are always Analysts, who have no
    /// mutation tools (no edit, no full shell — only read-only shell, which
    /// reports [`Self::side_effects`] = false). This classification is
    /// coupled to `Role::tools()`; if Analyst ever gains side-effecting
    /// tools, this must be reconsidered.
    fn side_effects(&self) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let analyze = super::get_str(&args, "analyze")?;

        // Async dispatch path — delegate to analysts in background.
        // Read user context from task-locals (set by Agent work loop
        // before each tool.execute() call) so the queued result carries
        // the correct user identity for per-user delivery.
        if self.dispatch_mode.is_async() {
            let ws = ws.clone();
            let analyze = analyze.to_string();
            let caller_role = self.caller_role;
            let user_name = crate::agent::CURRENT_TOOL_USER_NAME
                .try_with(String::clone)
                .unwrap_or_default();
            let channel = crate::agent::CURRENT_TOOL_CHANNEL
                .try_with(String::clone)
                .unwrap_or_default();

            tokio::spawn(async move {
                // Catch panics so the caller ALWAYS receives an envelope — a
                // panic in the dispatch task would otherwise leave the Manager
                // waiting forever on a result that can never arrive.
                let round = std::panic::AssertUnwindSafe(async {
                    dispatch_durable_analyze(
                        &ws,
                        &analyze,
                        caller_role,
                        user_name.clone(),
                        channel.clone(),
                    )
                    .await
                })
                .catch_unwind()
                .await;
                // The dispatch builds the envelope (with pending_job_id set by
                // the completion tx) — route it as-is so the persisted copy and
                // the routed copy can never drift. Only the panic path rebuilds
                // here.
                let envelope = match round {
                    Ok(Some(envelope)) => envelope,
                    Ok(None) => {
                        // Drain-cut: job stays status='launched' for boot resume —
                        // route NOTHING now (a spurious error envelope during
                        // the drain would discard the checkpointed outcomes;
                        // the result envelope is delivered after boot resume).
                        return;
                    }
                    Err(panic) => {
                        let panic = crate::util::panic_message(&*panic);
                        tracing::error!(panic = %panic, "analyze round dispatch panicked");
                        AgentJob {
                            content: build_async_analyze_message(&Err(anyhow::anyhow!(
                                "analyze round dispatch panicked: {panic}"
                            ))),
                            workspace_name: ws.name.clone(),
                            user_name,
                            channel,
                            kind: JobKind::AnalyzeToolResult,
                            role: caller_role,
                            reply_target: None,
                            pending_job_id: None,
                        }
                    }
                };

                // Drain/shutdown fired between the dispatch and the route: the
                // pending row (if any) survives for boot replay — skip routing
                // rather than deliver into a consumer that has stopped pulling.
                // (If the pending INSERT had failed, this also suppresses the
                // insert-failure policy's best-effort route — the surviving
                // launched job row is resumed at the next boot, so the result
                // is deferred, never lost.)
                if crate::shutdown::aborting() {
                    return;
                }
                message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
            });

            return Ok("Sub-agent dispatched. Results will follow shortly.".to_string());
        }

        // Sync path — blocks caller until the analysts complete.
        run_parallel_analysts_and_consolidate(ws, analyze).await
    }
}

/// One durable analyze-round roster slot (pre-generated agent id + final task).
struct AnalyzeSlot {
    agent_id: String,
    task: String,
}

/// Outcome of a durable analyze round.
enum AnalyzeRunOutcome {
    /// The round produced a result (Ok or Err — both terminalize with a
    /// durable envelope).
    Result(anyhow::Result<String>),
    /// Round cut by drain/shutdown — the job stays status='launched' for boot
    /// resume; nothing is routed or terminalized now.
    DrainCut,
}

/// Durable async analyze dispatch (SPAWN → run → CHECKPOINT → COMPLETE).
///
/// 1. SPAWN: one tx — INSERT jobs (kind=analyze, task=question) + INSERT agents
///    (all pre-generated analyst ids with final tasks). MUST commit before
///    the analysts' first session writes.
/// 2. RUN: parallel analysts.
/// 3. CHECKPOINT: per-agent raw-response outcomes.
/// 4. COMPLETE: one tx — INSERT pending_jobs (envelope id = job id) + DELETE
///    jobs row — the exactly-once persistence boundary.
///
/// Returns `None` only when the round was cut by drain/shutdown (job left
/// status='launched' for boot resume — nothing to route now). On error the
/// envelope still routes (errors wrapped in `<analyze-tool-result>`), and the
/// job is terminalized.
async fn dispatch_durable_analyze(
    ws: &Workspace,
    analyze: &str,
    caller_role: Role,
    user_name: String,
    channel: String,
) -> Option<AgentJob> {
    let job_id = crate::generate_id();
    let result = match run_analyze_with_job(
        ws,
        analyze,
        &job_id,
        caller_role,
        &user_name,
        &channel,
        false,
    )
    .await
    {
        Ok(AnalyzeRunOutcome::DrainCut) => {
            // Drain-cut: analyst outcomes are already
            // checkpointed — leave the job status='launched' for boot resume
            // (recoverable from the agent outcome checkpoints). No terminalization, no
            // error envelope: a spurious envelope here would discard the
            // checkpointed outcomes and contradict "jobs stay status='launched'
            // for boot resume".
            tracing::info!(
                job = %job_id,
                "Analyze round cut short by drain — job stays launched for boot resume",
            );
            return None;
        }
        Ok(AnalyzeRunOutcome::Result(result)) => result,
        Err(e) => Err(e),
    };
    let envelope = crate::jobs::complete_durable_job(
        &job_id,
        build_async_analyze_message(&result),
        JobKind::AnalyzeToolResult,
        caller_role,
        &user_name,
        &channel,
        &ws.name,
    )
    .await;
    Some(envelope)
}

/// Spawn the analyze job + roster (one tx), run the analysts, checkpoint per-agent
/// outcomes, consolidate. `pre_done` carries slots already completed (resume)
/// with their stored raw-response outcomes.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_analyze_with_job(
    ws: &Workspace,
    analyze: &str,
    job_id: &str,
    caller_role: Role,
    user_name: &str,
    channel: &str,
    resume: bool,
) -> anyhow::Result<AnalyzeRunOutcome> {
    let deadline = std::time::Instant::now() + round_timeout();

    // Fresh dispatch: build the roster and spawn the job BEFORE any analyst
    // session write (caller identity persisted on the job row so a later
    // resume delivers to the ORIGINAL caller). Resume: reuse the stored roster
    // (agent ids + final tasks) — never regenerate ids (the PK would conflict
    // AND the new ids would not match the stored roster rows).
    let (slots, pre_done) = if resume {
        let rows = crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
        let slots: Vec<AnalyzeSlot> = rows
            .iter()
            .map(|r| AnalyzeSlot {
                agent_id: r.agent_id.clone(),
                task: r.task.clone(),
            })
            .collect();
        let pre_done: Vec<(String, String)> = rows
            .into_iter()
            .filter(|r| r.status == crate::jobs::RowStatus::Done.as_str())
            .filter_map(|r| r.outcome.map(|o| (r.agent_id, o)))
            .collect();
        (slots, pre_done)
    } else {
        let suffix = crate::generate_suffix();
        let angles = load_analyst_angles();
        let mut slots: Vec<AnalyzeSlot> = Vec::with_capacity(PARALLEL_ANALYST_COUNT);
        let mut agents: Vec<crate::jobs::NewAgent> = Vec::with_capacity(PARALLEL_ANALYST_COUNT);
        for i in 0..PARALLEL_ANALYST_COUNT {
            let slot = analyst_slot(ws, &angles, &suffix, i, analyze);
            agents.push(crate::jobs::NewAgent {
                agent_id: slot.agent_id.clone(),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(i64::try_from(i).unwrap_or(i64::MAX)),
                task: slot.task.clone(),
            });
            slots.push(slot);
        }
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            job_id,
            analyze,
            &ws.name,
            user_name,
            channel,
            caller_role,
            &agents,
            &crate::jobs::SpawnChild::Analyze,
        )
        .await?;
        (slots, Vec::new())
    };

    // Run the launched slots; reconstruct done slots from stored outcomes.
    let fresh_slots: Vec<&AnalyzeSlot> = slots
        .iter()
        .filter(|s| !pre_done.iter().any(|(id, _)| id == &s.agent_id))
        .collect();
    // Round key = the durable job_id — stable across resume so a resumed
    // round's analysts and consolidation call group under the same key.
    let fresh_runs = run_analyze_slots(ws, &fresh_slots, deadline, resume, job_id, analyze).await;

    // Assemble runs in slot order.
    let mut runs: Vec<AnalyzeRun> = Vec::with_capacity(slots.len());
    let mut fresh_iter = fresh_runs.into_iter();
    for slot in &slots {
        if let Some((_, raw)) = pre_done.iter().find(|(id, _)| id == &slot.agent_id) {
            // Done slot — reconstruct the run from the stored response. The
            // fresh agent pre-loads the analyst's persisted session so
            // extract_findings keeps KV-cache compatibility.
            let mut agent = crate::Agent::new(
                slot.agent_id.clone(),
                crate::Role::Analyst,
                ws,
                None,
                String::new(),
                String::new(),
                Some(crate::registry::ParentKey::AnalyzeRound(job_id.to_string())),
                Some(analyze.to_string()),
            );
            let _ = agent
                .session
                .init(
                    &slot.agent_id,
                    "",
                    ws,
                    &crate::Role::Analyst,
                    None,
                    "",
                    "",
                    None,
                )
                .await;
            runs.push(AnalyzeRun::Completed {
                agent,
                response: Some(raw.clone()),
            });
        } else if let Some(run) = fresh_iter.next() {
            runs.push(run);
        } else {
            unreachable!("every fresh slot resolves to exactly one run (1:1 with fresh_slots)");
        }
    }

    // CHECKPOINT: per-agent outcomes (raw response or collapsed reason).
    let conn = &crate::session::store().conn;
    for (slot, run) in slots.iter().zip(&runs) {
        let (status, outcome) = match run {
            AnalyzeRun::Completed {
                response: Some(raw),
                ..
            } => (crate::jobs::RowStatus::Done, raw.clone()),
            AnalyzeRun::Completed {
                agent,
                response: None,
            } => (
                crate::jobs::RowStatus::Failed,
                agent.failure_reason("analyst produced no response"),
            ),
            AnalyzeRun::Failed { reason } => (crate::jobs::RowStatus::Failed, reason.clone()),
        };
        if let Err(e) =
            crate::jobs::write_agent_outcome(conn, job_id, &slot.agent_id, status, Some(&outcome))
                .await
        {
            tracing::warn!(job = %job_id, error = %e, "Failed to checkpoint analyze outcome");
        }
    }

    // Drain/shutdown cut the round mid-flight: skip the consolidate LLM call
    // (no new LLM work during the drain) — the checkpointed outcomes are
    // reused by the next boot's resume; the caller leaves the job launched.
    if crate::shutdown::aborting() {
        return Ok(AnalyzeRunOutcome::DrainCut);
    }

    let result = consolidate_analyst_runs(ws, analyze, runs, deadline, job_id).await;
    Ok(AnalyzeRunOutcome::Result(result))
}

/// Run the given analyst slots concurrently (deadline-bounded).
///
/// `round_key` is the analyze round's grouping key (the durable `job_id` on the
/// durable path, a fresh suffix on the sync path) — every analyst of the round
/// registers with `AnalyzeRound(round_key)` so the Running Agents view groups them
/// with the round's consolidation call.
async fn run_analyze_slots(
    ws: &Workspace,
    slots: &[&AnalyzeSlot],
    deadline: std::time::Instant,
    resume: bool,
    round_key: &str,
    question: &str,
) -> Vec<AnalyzeRun> {
    let members: Vec<_> = slots
        .iter()
        .map(|slot| {
            let ws = ws.clone();
            let agent_id = slot.agent_id.clone();
            let task = slot.task.clone();
            let round_key = round_key.to_string();
            let question = question.to_string();
            move |round| async move {
                // Session-non-emptiness discriminator: a resumed slot whose
                // session already contains the task continues with an empty
                // message (no duplicate task-prompt append); a missing/empty
                // session dispatches fresh with the stored task.
                let has_session = resume && crate::session::store().has_content(&agent_id).await;
                // Dynamic-resume path (resume + conditional message) — not a default dispatch.
                run_agent(
                    agent_id,
                    crate::Role::Analyst,
                    &ws,
                    None,
                    if has_session { "" } else { task.as_str() },
                    String::new(),
                    String::new(),
                    None,
                    resume,
                    Some(round),
                    Some(crate::registry::ParentKey::AnalyzeRound(round_key)),
                    Some(question),
                )
                .await
            }
        })
        .collect();
    let handles = crate::agent::spawn_staggered_round(members, resume).await;
    await_round_members(handles, deadline)
        .await
        .into_iter()
        .map(|m| match m {
            RoundMember::Done((agent, response)) => AnalyzeRun::Completed { agent, response },
            RoundMember::TimedOut => AnalyzeRun::Failed {
                reason: "analyst still running when the round deadline expired".to_string(),
            },
            RoundMember::Panicked => AnalyzeRun::Failed {
                reason: "analyst task panicked".to_string(),
            },
            RoundMember::Cancelled => AnalyzeRun::Failed {
                reason: "analyst task was cancelled".to_string(),
            },
        })
        .collect()
}

/// Boot resume of an analyze round: re-run not-done roster slots with
/// their stored tasks, reconstruct done slots from stored outcomes, then
/// re-consolidate. The consolidated result is terminalized into a pending
/// envelope exactly like a fresh dispatch — delivered to the ORIGINAL caller
/// (role/user/channel persisted on the job row at spawn), never the Manager.
///
/// Aborts quietly on shutdown/drain: no routing, no terminalization — the job
/// row stays for the next boot (checkpointed outcomes are reused, so the
/// already-completed LLM work is never lost or duplicated).
pub(crate) async fn resume_analyze_round(job_id: &str, ws: &Workspace) {
    let Some((caller, caller_role)) = crate::jobs::resume_job_preamble(
        &crate::session::store().conn,
        job_id,
        "Analyze resume",
        "Analyze resume",
    )
    .await
    else {
        return;
    };
    let result = match run_analyze_with_job(
        ws,
        &caller.task,
        job_id,
        caller_role,
        &caller.user_name,
        &caller.channel,
        true,
    )
    .await
    {
        Ok(AnalyzeRunOutcome::Result(result)) => result,
        // Round drain-cut mid-resume: abort quietly — the outcomes are
        // checkpointed, the next boot reuses them.
        Ok(AnalyzeRunOutcome::DrainCut) => {
            tracing::info!(
                job = %job_id,
                "Analyze resume aborted after analysts completed — job stays for next boot",
            );
            return;
        }
        Err(e) => Err(e),
    };
    // Drain/shutdown fired after the round returned: abort quietly WITHOUT
    // routing and WITHOUT deleting the row — the outcomes are checkpointed
    // (next boot reuses them) and routing a partial result here would race
    // the exit.
    if crate::shutdown::aborting() {
        tracing::info!(
            job = %job_id,
            "Analyze resume aborted after analysts completed — job stays for next boot",
        );
        return;
    }
    complete_durable_job_and_route(
        job_id,
        build_async_analyze_message(&result),
        JobKind::AnalyzeToolResult,
        caller_role,
        &caller,
        &ws.name,
    )
    .await;
}

/// Boot-scan over-cap handling for analyze jobs: the job exceeded
/// MAX_BOOT_REDISPATCH — deliver an ERROR envelope to the original caller
/// (the `<analyze-tool-result>` envelope is the async-analyze caller's ONLY result
/// path; marking failed with no envelope would strand the caller forever —
/// "failed = terminal … surface to user").
pub(crate) async fn analyze_capped_envelope(job_id: &str, ws: &Workspace) {
    let Some((caller, caller_role)) = crate::jobs::resume_job_preamble(
        &crate::session::store().conn,
        job_id,
        "Analyze capped report",
        "Analyze cap",
    )
    .await
    else {
        return;
    };
    let result: anyhow::Result<String> = Err(anyhow::anyhow!(format!(
        "analyze round aborted after {} boot re-dispatch attempts — the last crash lost the round; \
         please re-issue the analyze request",
        crate::jobs::MAX_BOOT_REDISPATCH,
    )));
    complete_durable_job_and_route(
        job_id,
        build_async_analyze_message(&result),
        JobKind::AnalyzeToolResult,
        caller_role,
        &caller,
        &ws.name,
    )
    .await;
}

/// Build the `<analyze-tool-result>` envelope message for an async analyze dispatch.
///
/// Shared by the async dispatch path (the `tokio::spawn` body in
/// [`AnalyzeTool::execute`]) and tests — the envelope shape that reaches the
/// caller's agent channel is production code, not a test re-wrap.
fn build_async_analyze_message(result: &anyhow::Result<String>) -> String {
    build_async_result_envelope(result, "analyze-tool-result")
}

/// Wrap a sub-agent/tool result in the async `<tag>` envelope delivered to
/// the caller's agent channel. Failures carry an explicit marker — findings
/// are never silently dropped. Shared with the deep research tool.
pub(crate) fn build_async_result_envelope(result: &anyhow::Result<String>, tag: &str) -> String {
    match result {
        Ok(text) => format!("<{tag}>\n\n{text}</{tag}>"),
        Err(e) => {
            tracing::debug!(error = %e, %tag, "async tool result failed");
            format!("<{tag}>\n\nAn error occurred: {e}</{tag}>")
        }
    }
}

/// Shared tail of the 4 resume/capped paths: terminalize a durable
/// analyze/research job and route its envelope to the stored caller. Takes
/// pre-built content so the caller's named wrapper keeps kind and tag
/// paired; the INSERT-failure best-effort route lives inside
/// [`crate::jobs::complete_durable_job`].
pub(crate) async fn complete_durable_job_and_route(
    job_id: &str,
    content: String,
    kind: JobKind,
    caller_role: Role,
    caller: &crate::jobs::JobCaller,
    workspace_name: &str,
) {
    let envelope = crate::jobs::complete_durable_job(
        job_id,
        content,
        kind,
        caller_role,
        &caller.user_name,
        &caller.channel,
        workspace_name,
    )
    .await;
    crate::message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
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

/// Outcome of one member of a parallel round awaited with a deadline.
#[derive(Debug)]
pub(crate) enum RoundMember<T> {
    Done(T),
    /// Member was still running when the round deadline expired — aborted.
    TimedOut,
    /// Member task panicked — surfaced, never silent.
    Panicked,
    /// Member task was cancelled externally (e.g. runtime shutdown), not by
    /// the round deadline — distinct from [`RoundMember::TimedOut`] so the
    /// surfaced reason stays accurate.
    Cancelled,
}

/// Await all round tasks with a single overall deadline. Members still
/// pending at the deadline are aborted and reported [`RoundMember::TimedOut`]
/// (warn-logged); panicked members are reported [`RoundMember::Panicked`] with
/// their panic message logged. Completed members keep their results, so a
/// stuck analyst never blocks the others' delivery.
///
/// Note: members are `tokio::spawn`ed (panic isolation), so if this future is
/// dropped mid-await the member tasks run detached to self-termination rather
/// than being cancelled — reachable only during process shutdown (tools run
/// to completion on agent cancel), where the runtime ends anyway; the
/// deadline covers the live-await path.
pub(crate) async fn await_round_members<T: Send + 'static>(
    handles: Vec<tokio::task::JoinHandle<T>>,
    deadline: std::time::Instant,
) -> Vec<RoundMember<T>> {
    use futures_util::StreamExt;
    use futures_util::stream::FuturesUnordered;

    let start = std::time::Instant::now();
    let cancel = tokio_util::sync::CancellationToken::new();
    let mut pending: FuturesUnordered<_> = handles
        .into_iter()
        .enumerate()
        .map(|(i, mut handle)| {
            let cancel = cancel.clone();
            async move {
                tokio::select! {
                    biased;
                    r = &mut handle => (i, match r {
                        Ok(v) => RoundMember::Done(v),
                        Err(e) if e.is_panic() => {
                            let panic = crate::util::panic_message(&*e.into_panic());
                            tracing::warn!(member = i, %panic, "round member task panicked");
                            RoundMember::Panicked
                        }
                        Err(_) => {
                            tracing::warn!(member = i, "round member task cancelled externally");
                            RoundMember::Cancelled
                        }
                    }),
                    () = cancel.cancelled() => {
                        handle.abort();
                        (i, RoundMember::TimedOut)
                    }
                }
            }
        })
        .collect();

    let mut out: Vec<Option<RoundMember<T>>> = (0..pending.len()).map(|_| None).collect();
    let deadline_expired = loop {
        if pending.is_empty() {
            break false;
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break true;
        }
        match tokio::time::timeout(remaining, pending.next()).await {
            Ok(Some((i, member))) => out[i] = Some(member),
            Ok(None) => break false,
            Err(_) => break true,
        }
    };
    if deadline_expired {
        cancel.cancel();
        while let Some((i, member)) = pending.next().await {
            out[i] = Some(member);
        }
        // After the drain every slot is filled; a `TimedOut` slot is exactly a
        // member that was still running at the deadline (aborted), while
        // members that finished just before it resolve to their real outcome
        // (biased select prefers the ready handle) — an accurate count.
        let aborted = out
            .iter()
            .filter(|m| matches!(m, Some(RoundMember::TimedOut)))
            .count();
        tracing::warn!(
            members_aborted = aborted,
            elapsed_secs = start.elapsed().as_secs_f64(),
            "round consolidation deadline expired — aborting stuck members"
        );
    }
    // Every loop-exit path (pending empty, deadline-drain, Ok(None)) fills
    // every slot exactly once — the invariant is enforced by construction.
    out.into_iter()
        .map(|m| m.expect("every round member resolves exactly once"))
        .collect()
}

/// Default bound on a parallel round's consolidation wait. Hours-scale: only
/// pathological stalls (a stuck analyst never completing) hit it; normal
/// analyst work finishes in minutes.
const DEFAULT_ROUND_TIMEOUT_SECS: u64 = 3 * 60 * 60;

/// Round consolidation bound for [`await_round_members`]. Overridable via env
/// for tuning.
pub(crate) fn round_timeout() -> Duration {
    crate::util::env_duration_secs("MAHBOT_ROUND_TIMEOUT_SECS", DEFAULT_ROUND_TIMEOUT_SECS)
}

/// One analyst's outcome in a parallel round. `Failed` covers members that
/// never completed (stuck past the round deadline, panicked, or cancelled) —
/// the reason is surfaced, never silent.
#[allow(clippy::large_enum_variant)] // Agent is inherently large; the enum is transient
enum AnalyzeRun {
    Completed {
        agent: Agent,
        response: Option<String>,
    },
    Failed {
        reason: String,
    },
}

impl AnalyzeRun {
    fn response(&self) -> Option<&str> {
        match self {
            Self::Completed { response, .. } => response.as_deref(),
            Self::Failed { .. } => None,
        }
    }
}

/// Verdict of a single targeted claim verification, dispatched via
/// `dispatch_claim_verifiers` on the deep research path only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VerificationVerdict {
    pub verdict: String,
    pub evidence: String,
}

/// Verification outcome merged into the deep research run summary. The
/// telemetry fields (the verifier analyst's tool calls / searches / queries)
/// feed its query ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
async fn run_parallel_analysts_and_consolidate(ws: &Workspace, analyze: &str) -> Result<String> {
    // One round-wide bound shared by the analyst and extraction member waits
    // — a stuck analyst is aborted at it, so no phase can hang the round.
    // The sequential consolidation call after the deadline finishes within
    // its own retry bounds.
    let deadline = std::time::Instant::now() + round_timeout();
    // Round key shared by the analysts and the consolidation call so the
    // Running Agents view groups them under one analyze round.
    let round_key = crate::generate_suffix();
    let runs =
        run_parallel_analysts(ws, analyze, PARALLEL_ANALYST_COUNT, deadline, &round_key).await;
    consolidate_analyst_runs(ws, analyze, runs, deadline, &round_key).await
}

/// Load the decorrelation angles asset — one distinct research angle per
/// parallel analyst. Malformed or missing sections degrade to an empty angle
/// (the plain analyze is used, preserving the original single-question behavior).
pub(crate) fn load_analyst_angles() -> Vec<String> {
    load_prompt_sections("analyze/angles.md")
}

/// Compose an analyst's roster slot (angle appended to the analyze).
/// KV-cache discipline: vary ONLY the user message (the research
/// angle) — never per-analyst model/effort/tools.
fn analyst_slot(
    ws: &Workspace,
    angles: &[String],
    suffix: &str,
    i: usize,
    analyze: &str,
) -> AnalyzeSlot {
    let angle = angles.get(i).cloned().unwrap_or_default();
    let task = if angle.is_empty() {
        analyze.to_string()
    } else {
        format!("{analyze}\n\nResearch angle:\n{angle}")
    };
    AnalyzeSlot {
        agent_id: format!("analyze_{}_{}_{}_analyst", ws.name, suffix, i),
        task,
    }
}

/// Run `count` parallel analyst agents with decorrelated research angles,
/// returning each member's outcome. The wait shares the round-wide `deadline`
/// (see [`run_parallel_analysts_and_consolidate`]): a member still running at
/// the deadline is aborted and reported as failed (with the reason) instead of
/// stalling the round, and the completed members' results are still delivered.
async fn run_parallel_analysts(
    ws: &Workspace,
    analyze: &str,
    count: usize,
    deadline: std::time::Instant,
    round_key: &str,
) -> Vec<AnalyzeRun> {
    let angles = load_analyst_angles();
    let slots: Vec<AnalyzeSlot> = (0..count)
        .map(|i| analyst_slot(ws, &angles, round_key, i, analyze))
        .collect();
    let slot_refs: Vec<&AnalyzeSlot> = slots.iter().collect();
    run_analyze_slots(ws, &slot_refs, deadline, false, round_key, analyze).await
}

/// Consolidate analyst runs: 0 valid → error, 1 valid → raw passthrough,
/// ≥2 valid → extract findings and run the shared grouping pass (≥2 with
/// parseable claims group; a single parseable source skips grouping). Failed
/// members count as no-response, with their reasons surfaced in the error.
async fn consolidate_analyst_runs(
    ws: &Workspace,
    analyze: &str,
    runs: Vec<AnalyzeRun>,
    deadline: std::time::Instant,
    round_key: &str,
) -> Result<String> {
    // Drain-watch coverage for this phase: extraction runs while the analyst
    // Agents are still registered (their registry entries keep the drain-watch
    // from firing); the grouping pass runs with zero live agents and is
    // covered by the guard INSIDE the shared consensus core
    // (`run_grouping_repair`) — the single tracker for the consolidation call.
    // `consolidate_findings` re-checks the drain flag before the grouping call,
    // closing the sync-only window between extraction end and the core guard's
    // registration (documented accepted-residual-window contract, jobs.rs). No
    // phase guard here: it would duplicate the core's row and register a
    // phantom row on 0/1-valid-response rounds where no grouping call runs.
    let valid_count = runs
        .iter()
        .filter(|r| r.response().is_some_and(|t| !t.trim().is_empty()))
        .count();
    match valid_count {
        0 => {
            let reasons: Vec<&str> = runs
                .iter()
                .filter_map(|r| match r {
                    AnalyzeRun::Failed { reason } => Some(reason.as_str()),
                    AnalyzeRun::Completed { .. } => None,
                })
                .collect();
            let suffix = if reasons.is_empty() {
                String::new()
            } else {
                format!(" ({})", reasons.join("; "))
            };
            anyhow::bail!("All parallel analysts failed to produce a response{suffix}");
        }
        1 => Ok(single_raw_response(&runs).expect("exactly one valid response")),
        _ => {
            let outcomes = extract_findings(runs, deadline).await;
            consolidate_findings(ws, analyze, outcomes, round_key).await
        }
    }
}

fn single_raw_response(runs: &[AnalyzeRun]) -> Option<String> {
    runs.iter().find_map(|r| match r {
        AnalyzeRun::Completed { response, .. } => response.clone().filter(|t| !t.trim().is_empty()),
        AnalyzeRun::Failed { .. } => None,
    })
}

/// Extract structured claim-level findings from each valid analyst response
/// while the agent is still alive (see [`Agent::extract_verdict`] for the
/// KV-cache rationale). Fail-open: parse failures keep the raw response and
/// are never silently dropped. The extraction wait shares the round-wide
/// `deadline` — a stuck extraction is reported as no-response, not awaited
/// forever.
async fn extract_findings(
    runs: Vec<AnalyzeRun>,
    deadline: std::time::Instant,
) -> Vec<AnalystOutcome> {
    let extraction_prompt = load_prompt("extraction/findings.md");
    let handles: Vec<_> = runs
        .into_iter()
        .map(|run| {
            let extraction_prompt = extraction_prompt.clone();
            tokio::spawn(async move {
                let (agent, response) = match run {
                    AnalyzeRun::Completed { agent, response } => (agent, response),
                    AnalyzeRun::Failed { reason } => {
                        return AnalystOutcome::NoResponse(crate::util::scrub_credentials(&reason));
                    }
                };
                let Some(raw) = response else {
                    let reason = agent.failure_reason("analyst produced no response");
                    return AnalystOutcome::NoResponse(crate::util::scrub_credentials(&reason));
                };
                if raw.trim().is_empty() {
                    return AnalystOutcome::NoResponse("analyst produced no response".to_string());
                }
                match agent
                    .extract_verdict::<AnalystFindings>(&extraction_prompt, None, None)
                    .await
                {
                    Ok(findings) => AnalystOutcome::Findings { raw, findings },
                    Err(e) => AnalystOutcome::ParseFailed {
                        raw,
                        failure: e.to_string(),
                    },
                }
            })
        })
        .collect();
    await_round_members(handles, deadline)
        .await
        .into_iter()
        .map(|m| match m {
            RoundMember::Done(outcome) => outcome,
            RoundMember::TimedOut => AnalystOutcome::NoResponse(
                "findings extraction still running when the round deadline expired".to_string(),
            ),
            RoundMember::Panicked => {
                AnalystOutcome::NoResponse("findings extraction task panicked".to_string())
            }
            RoundMember::Cancelled => {
                AnalystOutcome::NoResponse("findings extraction task was cancelled".to_string())
            }
        })
        .collect()
}

/// Per-agent claim texts keyed by the ORIGINAL outcome index. A failed agent's
/// slot stays empty so the id space matches the input material. Per-agent
/// duplicates are NOT deduped: two identical claims from one agent are two
/// distinct ids, and the model places each exactly once.
#[must_use]
fn claims_per_agent(outcomes: &[AnalystOutcome]) -> Vec<Vec<String>> {
    outcomes
        .iter()
        .map(|o| match o {
            AnalystOutcome::Findings { findings, .. } => {
                findings.claims.iter().map(|c| c.claim.clone()).collect()
            }
            _ => Vec::new(),
        })
        .collect()
}

/// Render the fail-open analyst deliverable: the flat numbered claim list
/// plus the raw analyst dumps, headed by an explicit `marker` naming why
/// consolidation produced no groups. Head-placed so the sync path's 5 KB
/// sandwich truncation keeps the marker. Shared by every no-groups path so
/// the fallback shape stays identical.
#[must_use]
fn render_unconsolidated_fallback(
    marker: &str,
    items_by_agent: &[Vec<String>],
    outcomes: &[AnalystOutcome],
) -> String {
    let flat = render_flat_claim_list(items_by_agent);
    let raw_dump = render_raw_analyst_dump(outcomes);
    format!("## Analyst Reports\n\n({marker})\n\n{flat}\n\n{raw_dump}")
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
    analyze: &str,
    outcomes: Vec<AnalystOutcome>,
    round_key: &str,
) -> Result<String> {
    let items_by_agent = claims_per_agent(&outcomes);
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
        return Ok(render_unconsolidated_fallback(
            "only one analyst produced parseable claims — grouping skipped",
            &items_by_agent,
            &outcomes,
        ));
    }
    // User material: global flat ids across ALL agents (each claim exactly one
    // id, in (agent, claim) order) — the schema's `id` field matches.
    let material = crate::consensus::numbered_items_material(&items_by_agent);
    let user =
        format!("# Original Question\n\n{analyze}\n\n# Agent Claims (id-numbered)\n\n{material}");

    let model = CONFIG.role_model(Role::Analyst);
    let routing = CONFIG.model_routing(&model);
    let system = format!(
        "{}\n\n{}",
        load_prompt("synthesis/analyst.md"),
        load_prompt("synthesis/grouping_contradictions.md"),
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
    let parent = Some(crate::registry::ParentKey::AnalyzeRound(
        round_key.to_string(),
    ));
    // Drain fired in the extraction-end → grouping-start gap (the analyst
    // agents have dropped, the consensus core's guard is not yet registered):
    // never start a new LLM call once the drain begins — deliver the fail-open
    // raw claim list instead (the same deliverable as a Fallback, without the
    // doomed call). The extraction itself is covered by the live analyst
    // cards; this check closes the sync-only residual window.
    if crate::shutdown::aborting() {
        tracing::warn!(
            "Analyst consolidation skipped — shutdown/drain in progress; delivering raw claim list"
        );
        return Ok(render_unconsolidated_fallback(
            "unconsolidated — shutdown during consolidation",
            &items_by_agent,
            &outcomes,
        ));
    }
    match crate::consensus::run_grouping_repair(
        ws,
        "consolidate",
        request,
        &items_by_agent,
        parent,
        Some(analyze.to_string()),
    )
    .await
    {
        crate::consensus::RepairOutcome::Repaired { output, references } => Ok(
            render_analyze_groups(analyze, &output, &references, &table, n_valid, &outcomes),
        ),
        crate::consensus::RepairOutcome::Fallback => {
            tracing::warn!("Analyst consolidation failed — delivering raw claim list");
            Ok(render_unconsolidated_fallback(
                "unconsolidated — consolidation failed",
                &items_by_agent,
                &outcomes,
            ))
        }
    }
}

/// Render the analyze output contract: summary + groups (heading, contradiction
/// flag, members citing item ids) + ungrouped remainder + DISPUTED
/// cross-references from remainder items to frozen groups. Brackets [n/N]
/// come from distinct cited agent ids; DISPUTED appears only when the group's
/// contradiction flag is true. Self-reported caveats of a cited claim render
/// as member metadata, never as contradictions.
#[must_use]
fn render_analyze_groups(
    analyze: &str,
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
    out.push_str(&crate::consensus::render_ungrouped_section(
        output,
        references,
        |member, disputed| {
            let mut line = crate::consensus::render_member_line(member, table);
            line.push_str(disputed);
            if let Some(c) = member_caveat(member, table, outcomes) {
                let _ = write!(line, " — caveat: {c}");
            }
            line
        },
    ));
    // Original question for context (answers are delivered out of band).
    let _ = write!(out, "\n\n_Original question: {analyze}_");
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

/// Extract search/web_search query strings + tool-call counts from session
/// history messages (for the query ledger and the run summary). Shared with
/// the deep research tool (`research.rs`), which also feeds it the persisted
/// history of deadline-aborted analysts (wrap-up stage).
pub(crate) fn extract_query_telemetry_from_history(
    history: &[ChatMessage],
) -> (usize, usize, Vec<String>) {
    let mut tool_calls = 0usize;
    let mut searches = 0usize;
    let mut queries = Vec::new();
    for msg in history {
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

/// Extract search/web_search query strings + tool-call counts from an
/// agent's session history (for the query ledger and the run summary).
/// Shared with the deep research tool (`research.rs`).
pub(crate) fn extract_query_telemetry(agent: &Agent) -> (usize, usize, Vec<String>) {
    extract_query_telemetry_from_history(agent.session.history())
}

/// Dispatch one fresh Analyst per verification target (bounded by
/// [`VERIFY_MAX_ANALYSTS`]) in parallel. `id_prefix` seeds the per-target
/// agent IDs; `task_extra` is appended to every task (e.g. the query
/// ledger). The wait shares the round-wide `deadline`. Shared with the deep
/// research tool's verification gate.
///
/// Returns the results PLUS the dispatched agent IDs (the deep research
/// sanitizer needs them at dispatch time — successful writers never appear in
/// wrap-up snapshots; the ids ride the return value so the shared helper
/// keeps no write-only out-parameter).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch_claim_verifiers(
    ws: &Workspace,
    id_prefix: &str,
    targets: &[VerificationTarget],
    task_extra: &str,
    deadline: std::time::Instant,
    resume: bool,
    run_key: &str,
    question: &str,
) -> (Vec<VerificationResult>, Vec<String>) {
    let task_template = load_prompt("analyze/verify.md");
    let extraction_prompt = load_prompt("extraction/verify.md");
    let suffix = crate::generate_suffix();
    let mut dispatched: Vec<String> = Vec::new();
    let members: Vec<_> = targets
        .iter()
        .take(VERIFY_MAX_ANALYSTS)
        .enumerate()
        .map(|(i, t)| {
            let ws = ws.clone();
            let agent_id = format!("{id_prefix}_{suffix}_{i}");
            dispatched.push(agent_id.clone());
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
            let run_key = run_key.to_string();
            let question = question.to_string();
            move |round| async move {
                run_claim_verifier(
                    &ws,
                    &agent_id,
                    &claim_text,
                    &task,
                    &extraction_prompt,
                    round,
                    &run_key,
                    &question,
                )
                .await
            }
        })
        .collect();
    let handles = crate::agent::spawn_staggered_round(members, resume).await;
    let results = await_round_members(handles, deadline)
        .await
        .into_iter()
        .enumerate()
        .map(|(i, m)| match m {
            RoundMember::Done(v) => v,
            RoundMember::TimedOut | RoundMember::Panicked | RoundMember::Cancelled => {
                VerificationResult {
                    claim: targets[i].claim.clone(),
                    verdict: "unresolved".to_string(),
                    evidence: "verifier never completed (round deadline or panic)".to_string(),
                    tool_calls: 0,
                    searches: 0,
                    queries: Vec::new(),
                }
            }
        })
        .collect();
    (results, dispatched)
}

/// Run one claim verifier: a fresh Analyst researches the claim and returns
/// its structured verdict. Any failure yields "unresolved" — fail-open.
///
/// `question` is the research run's question — threaded as the run group's
/// header label (purely presentational).
#[allow(clippy::too_many_arguments)]
async fn run_claim_verifier(
    ws: &Workspace,
    agent_id: &str,
    target_claim: &str,
    task: &str,
    extraction_prompt: &str,
    round: crate::agent::RoundOpts,
    run_key: &str,
    question: &str,
) -> VerificationResult {
    let (agent, response) = run_default_agent(
        agent_id,
        Role::Analyst,
        ws,
        task,
        Some(round),
        Some(crate::registry::ParentKey::Research(run_key.to_string())),
        Some(question.to_string()),
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
            None,
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
    async fn test_analyze_missing_args() {
        let tool = AnalyzeTool::new(DispatchMode::Sync, Role::Engineer);
        let ws = test_ws("/tmp/test_ws");

        // Missing analyze
        let result = tool.execute(&ws, json!({})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: analyze"),
            "Should mention missing analyze"
        );
    }

    /// Tests that analyze dispatches analysts and returns the consolidated result.
    /// Requires an LLM provider to be configured.
    #[tokio::test]
    #[ignore = "requires LLM provider"]
    async fn test_analyze_analyst() {
        let tool = AnalyzeTool::new(DispatchMode::Sync, Role::Engineer);
        let ws = test_ws("/tmp/test_ws");
        let args = json!({"analyze": "Say 'hello analyst' and nothing else."});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains("hello"),
            "output should contain hello: {result}"
        );
    }

    // ── Consolidation edge-case tests (no LLM provider needed) ──

    /// Build a bare analyst run for pipeline tests: a real `Agent` (no stores
    /// touched) plus an optional raw response.
    fn test_analyst_run(ws: &Workspace, response: Option<&str>) -> AnalyzeRun {
        let agent = Agent::new(
            format!("analyze_test_{}", crate::generate_suffix()),
            Role::Analyst,
            ws,
            None,
            String::new(),
            String::new(),
            None,
            None,
        );
        AnalyzeRun::Completed {
            agent,
            response: response.map(ToString::to_string),
        }
    }

    /// A failed round member (stuck past the deadline, panicked, cancelled).
    fn test_failed_run(reason: &str) -> AnalyzeRun {
        AnalyzeRun::Failed {
            reason: reason.to_string(),
        }
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
        let result = consolidate_analyst_runs(
            &ws,
            "test question",
            runs,
            std::time::Instant::now() + Duration::from_secs(60),
            "test_round",
        )
        .await;
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
    async fn test_consolidate_zero_responses_surfaces_failure_reasons() {
        let ws = test_ws("/tmp/test_ws");
        let runs = vec![
            test_failed_run("analyst 0 still running when the round deadline expired"),
            test_failed_run("analyst 1 task panicked"),
            test_analyst_run(&ws, None),
        ];
        let result = consolidate_analyst_runs(
            &ws,
            "test question",
            runs,
            std::time::Instant::now() + Duration::from_secs(60),
            "test_round",
        )
        .await;
        let err = result.expect_err("0 valid responses should error");
        assert!(
            err.to_string().contains("deadline expired"),
            "error should surface the stuck-analyst reason: {err}"
        );
        assert!(
            err.to_string().contains("panicked"),
            "error should surface the panicked-analyst reason: {err}"
        );
    }

    /// The round wait is bounded: a member that never completes is aborted at
    /// the deadline and reported [`RoundMember::TimedOut`], while completed
    /// members keep their results.
    #[tokio::test]
    async fn await_round_members_bounds_stuck_member() {
        struct AbortDetector(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for AbortDetector {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let aborted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = aborted.clone();
        let stuck = tokio::spawn(async move {
            let _detector = AbortDetector(flag);
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        });
        let done = tokio::spawn(async { 42u32 });
        let deadline = std::time::Instant::now() + Duration::from_millis(300);
        let members = await_round_members(vec![stuck, done], deadline).await;
        assert!(matches!(members[0], RoundMember::TimedOut), "{members:?}");
        assert!(matches!(members[1], RoundMember::Done(42)), "{members:?}");
        // Aborting the task drops its locals — the detector fires.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            aborted.load(std::sync::atomic::Ordering::SeqCst),
            "stuck member task should have been aborted at the deadline"
        );
    }

    /// A panicking member is surfaced as [`RoundMember::Panicked`], never
    /// silently dropped.
    #[tokio::test]
    async fn await_round_members_surfaces_panics() {
        let panicking = tokio::spawn(async { panic!("boom") });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let members = await_round_members(vec![panicking], deadline).await;
        assert!(matches!(members[0], RoundMember::Panicked), "{members:?}");
    }

    #[tokio::test]
    async fn test_consolidate_one_response_returned_directly() {
        let ws = test_ws("/tmp/test_ws");
        let runs = vec![
            test_analyst_run(&ws, Some("only answer")),
            test_analyst_run(&ws, None),
            test_analyst_run(&ws, None),
        ];
        let result = consolidate_analyst_runs(
            &ws,
            "test question",
            runs,
            std::time::Instant::now() + Duration::from_secs(60),
            "test_round",
        )
        .await;
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
        let result = consolidate_analyst_runs(
            &ws,
            "test question",
            runs,
            std::time::Instant::now() + Duration::from_secs(60),
            "test_round",
        )
        .await;
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

    // ── Analyze grouping renderer ────────────────────────────────────────────

    #[test]
    fn render_analyze_groups_computes_brackets_from_distinct_agents() {
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
        let items = claims_per_agent(&outcomes);
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
        let text = render_analyze_groups("q", &output, &[], &table, 2, &outcomes);
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
    fn render_analyze_groups_disputed_only_on_contradiction() {
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
        let items = claims_per_agent(&outcomes);
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
        let text = render_analyze_groups("q", &output, &[], &table, 2, &outcomes);
        assert!(
            text.contains("**Alpha** [2/2 · DISPUTED]"),
            "contradiction group renders [2/2 · DISPUTED]: {text}"
        );
    }

    #[test]
    fn render_analyze_groups_surfaces_member_caveats_as_metadata() {
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
        let items = claims_per_agent(&outcomes);
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
        let text = render_analyze_groups("q", &output, &[], &table, 2, &outcomes);
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
        consolidate_findings(&ws, "test question", outcomes, "test_round").await
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
        let result = consolidate_findings(&ws, "test question", outcomes, "test_round").await;
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
        let result = consolidate_analyst_runs(
            &ws,
            "test question",
            runs,
            std::time::Instant::now() + Duration::from_secs(60),
            "test_round",
        )
        .await;
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
        // The async dispatch path (tokio::spawn in AnalyzeTool::execute) builds
        // its envelope via build_async_analyze_message — this test drives the REAL
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
        let envelope = build_async_analyze_message(&result);
        assert!(envelope.contains("<analyze-tool-result>"), "{envelope}");
        assert!(
            envelope.contains("unconsolidated — consolidation failed"),
            "{envelope}"
        );
        assert!(envelope.contains("raw report"), "{envelope}");
        assert!(
            envelope.ends_with("</analyze-tool-result>"),
            "envelope must close: {envelope}"
        );
    }

    #[tokio::test]
    async fn async_envelope_wraps_sub_agent_errors() {
        // The async dispatch path's error branch: a failed sub-agent is
        // wrapped in the same envelope with the error text.
        let envelope = build_async_analyze_message(&Err(anyhow::anyhow!("sub-agent exploded")));
        assert!(envelope.contains("<analyze-tool-result>"), "{envelope}");
        assert!(
            envelope.contains("An error occurred: sub-agent exploded"),
            "{envelope}"
        );
        assert!(
            envelope.ends_with("</analyze-tool-result>"),
            "envelope must close: {envelope}"
        );
    }

    /// The sync analyze path truncates tool output via the 5 KB sandwich (head
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
    fn render_analyze_groups_renders_disputed_cross_references() {
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
        let items = claims_per_agent(&outcomes);
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
        let text = render_analyze_groups("q", &output, &references, &table, 2, &outcomes);
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

    /// Boot resume of a crashed async analyze must REUSE the existing job + roster
    /// (never re-spawn — the jobs.id PK would conflict AND freshly generated
    /// analyst ids would not match the stored roster rows, so outcomes would
    /// be lost). A single done slot reconstructs from its stored outcome
    /// (raw passthrough — no provider needed).
    ///
    /// Serialized with the drain-flag writers: `resume_analyze_round` consults the
    /// process-global drain flag and aborts early while it is set (project
    /// convention: retry_tests_lock).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn resume_analyze_round_reuses_existing_job() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_resume_analyze");
        let job_id = "analyze_job_resume_1";
        let agent_id = format!("analyze_{}_rs_0_analyst", ws.name);
        let conn = &crate::session::store().conn;
        // Pre-create the job row exactly as a crashed dispatch leaves it
        // (caller identity persisted on the job row; one done slot with a
        // stored outcome).
        crate::jobs::spawn_job(
            conn,
            job_id,
            "question?",
            &ws.name,
            "caller-user",
            "telegram",
            crate::Role::Assistant,
            &[crate::jobs::NewAgent {
                agent_id: agent_id.clone(),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(0),
                task: "question?".to_string(),
            }],
            &crate::jobs::SpawnChild::Analyze,
        )
        .await
        .unwrap();
        crate::jobs::write_agent_outcome(
            conn,
            job_id,
            &agent_id,
            crate::jobs::RowStatus::Done,
            Some("completed analyst response"),
        )
        .await
        .unwrap();

        // Resume: must NOT hit the jobs.id PK conflict, must reuse the stored
        // roster slot, then terminalize into a durable envelope delivered to
        // the ORIGINAL caller (Assistant, not Manager).
        resume_analyze_round(job_id, &ws).await;

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
                "SELECT envelope FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(pending.len(), 1, "envelope persisted with the job id");
        let envelope: crate::message_router::AgentJob =
            serde_json::from_str(&pending[0].get::<String>(0).unwrap()).unwrap();
        assert_eq!(
            envelope.role,
            crate::Role::Assistant,
            "resumed envelope routes to the original caller role, not Manager"
        );
        assert_eq!(envelope.user_name, "caller-user");
    }

    /// The boot-scan over-cap path must NOT strand the async-analyze caller: an
    /// error envelope (with the original caller identity) is delivered instead
    /// of a silent failed row.
    ///
    /// Serialized with the drain-flag writers: `analyze_capped_envelope` consults
    /// the process-global drain flag and aborts early while it is set (project
    /// convention: retry_tests_lock).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn analyze_capped_envelope_delivers_error_to_caller() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_analyze_capped");
        let job_id = "analyze_job_capped_1";
        let conn = &crate::session::store().conn;
        crate::jobs::spawn_job(
            conn,
            job_id,
            "question?",
            &ws.name,
            "caller-user",
            "telegram",
            crate::Role::Assistant,
            &[crate::jobs::NewAgent {
                agent_id: format!("analyze_{}_capped_0_analyst", ws.name),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(0),
                task: "question?".to_string(),
            }],
            &crate::jobs::SpawnChild::Analyze,
        )
        .await
        .unwrap();

        analyze_capped_envelope(job_id, &ws).await;

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
                "SELECT envelope FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(pending.len(), 1, "error envelope persisted with the job id");
        let envelope_json: String = pending[0].get(0).unwrap();
        let envelope: crate::message_router::AgentJob =
            serde_json::from_str(&envelope_json).unwrap();
        assert_eq!(
            envelope.role,
            crate::Role::Assistant,
            "capped envelope routes to the original caller role, not Manager"
        );
        assert_eq!(envelope.user_name, "caller-user");
        assert!(
            envelope.content.contains("analyze round aborted"),
            "the envelope must surface the cap failure to the caller"
        );
    }
}
