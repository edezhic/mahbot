//! AnalyzeTool — dispatches a batch of parallel analysts at `quick` (one round)
//! or `deep` (the multi-round investigation, [`super::research`]).
//!
//! Dispatch-only: analysts are read-only research agents, never mutation-capable.
//! The cancel action dispatches nothing — it deletes the named round's own rows.
//! A holder with both depths declares the `depth` choice; one with a fixed depth
//! does not. Holders: the Engineer, Manager and Maintainer run `quick` (sync for
//! the Engineer/Maintainer, async for the Manager); a guest Assistant runs `quick`
//! async; the admin Assistant runs both depths async and alone accepts an optional
//! `place` (a managed workspace name or an absolute path) for `quick`, moving the
//! round's execution root for that call only — `deep` ignores it. Argument
//! mechanics and refusals live in [`super::delegation`].
//!
//! Two decorrelated analysts per round report structured claim-level findings; the
//! shared [`crate::consensus`] pass groups them and computes agreement brackets in
//! code. Disputed groups get an optional annotation-only verification round,
//! appended as `## Verification`. Fail-open throughout — findings are never
//! silently lost — except a round the caller cancels: its partial findings never
//! become a result and it is never delivered (see [`crate::analyze_cancel`]).

use crate::agent::{run_agent, run_default_agent};
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::tools::Tool;
use crate::{Agent, ChatMessage, DEFAULT_MAX_TOKENS, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::Write as _;
use std::time::Duration;

/// Decorrelated parallel analysts per round (durable and sync share the width).
const PARALLEL_ANALYST_COUNT: usize = 2;

/// [`Sync`](DispatchMode::Sync) blocks the caller until the sub-agent completes;
/// [`Async`](DispatchMode::Async) injects its result via the caller's agent queue.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DispatchMode {
    Sync,
    Async,
}

impl DispatchMode {
    #[must_use]
    pub const fn is_async(self) -> bool {
        matches!(self, Self::Async)
    }
}

/// `quick` is one round of parallel analysts; `deep` the multi-round investigation
/// whose report arrives hours later. [`Self::ALL`] is in interface order.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AnalyzeDepth {
    Quick,
    Deep,
}

impl AnalyzeDepth {
    pub const ARG: &'static str = "depth";

    pub const ALL: [Self; 2] = [Self::Quick, Self::Deep];

    pub const STRS: &'static [&'static str] = &[Self::Quick.as_str(), Self::Deep.as_str()];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Deep => "deep",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|depth| depth.as_str() == value)
    }
}

/// The cancel action's argument spellings, declared only where an asynchronous
/// quick round can be dispatched ([`AnalyzeTool::cancels`]).
pub(crate) const ACTION_ARG: &str = "action";
/// The round the cancel action addresses — the id its acknowledgement carried.
pub(crate) const JOB_ID_ARG: &str = "job_id";
const ACTION_CANCEL: &str = "cancel";

pub struct AnalyzeTool {
    dispatch_mode: DispatchMode,
    /// Routes async results to the correct agent channel (Manager → manager_{ws},
    /// Assistant → direct_{...}).
    pub caller_role: Role,
    /// The depths this holder may call, in interface order. A single depth is never
    /// declared as a choice argument.
    kinds: Vec<AnalyzeDepth>,
    /// Whether the optional `place` argument is accepted — the admin Assistant and any
    /// custom role holding `quick`; every other holder keeps the argument-less interface.
    accepts_place: bool,
}

impl AnalyzeTool {
    /// The quick-only, no-`place` variant: Engineer, Manager, Maintainer, guest Assistant.
    #[must_use]
    pub fn new(dispatch_mode: DispatchMode, caller_role: Role) -> Self {
        Self {
            dispatch_mode,
            caller_role,
            kinds: vec![AnalyzeDepth::Quick],
            accepts_place: false,
        }
    }

    /// Both depths plus the optional `place` (honoured for `quick` only), always async:
    /// only a durable background round can honour a place, recording its directory for resume.
    #[must_use]
    pub fn admin(caller_role: Role) -> Self {
        Self {
            dispatch_mode: DispatchMode::Async,
            caller_role,
            kinds: vec![AnalyzeDepth::Quick, AnalyzeDepth::Deep],
            accepts_place: true,
        }
    }

    /// A custom role's exactly-granted depths, always async, with `place` iff `quick` is
    /// held. Never empty — a kindless grant takes the narrowest kind (`quick`).
    #[must_use]
    pub fn granted(kinds: &[AnalyzeDepth]) -> Self {
        let kinds = if kinds.is_empty() {
            vec![AnalyzeDepth::Quick]
        } else {
            kinds.to_vec()
        };
        let accepts_place = kinds.contains(&AnalyzeDepth::Quick);
        Self {
            dispatch_mode: DispatchMode::Async,
            caller_role: Role::Custom,
            kinds,
            accepts_place,
        }
    }

    /// Only a holder that can dispatch an asynchronous `quick` round is offered the
    /// cancel action: the Manager, the Assistant and a custom role granted `quick`. A
    /// sync or `deep`-only holder dispatches work this action deliberately does not stop.
    #[must_use]
    fn cancels(&self) -> bool {
        self.dispatch_mode.is_async() && self.kinds.contains(&AnalyzeDepth::Quick)
    }
}

#[async_trait]
impl Tool for AnalyzeTool {
    fn name(&self) -> &'static str {
        "analyze"
    }

    /// Compose from shared fragments: base text, the quick note, the depth note, the
    /// `place` note, the async note and — only where cancel is offered — the cancel
    /// note. The deep note always carries the long-running warning and the
    /// scope-confirmation requirement.
    fn description(&self) -> String {
        let mut parts = vec![load_prompt("tool/analyze.md")];
        if self.kinds.contains(&AnalyzeDepth::Quick) {
            parts.push(load_prompt("tool/analyze_quick.md"));
        }
        if self.kinds.contains(&AnalyzeDepth::Deep) {
            let note = if self.kinds.contains(&AnalyzeDepth::Quick) {
                "tool/analyze_depth.md"
            } else {
                "tool/analyze_deep.md"
            };
            parts.push(load_prompt(note));
        }
        if self.accepts_place {
            parts.push(load_prompt("tool/analyze_place.md"));
        }
        if self.dispatch_mode.is_async() {
            parts.push(load_prompt("tool/analyze_async.md"));
        }
        if self.cancels() {
            parts.push(load_prompt("tool/analyze_cancel.md"));
        }
        parts.join("\n\n")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let mut properties = json!({
            "message": {
                "type": "string",
                "description": "The question for the analysts"
            }
        });
        let mut dispatch = vec!["message"];
        if self.kinds.len() > 1 {
            let choices: Vec<&str> = self.kinds.iter().map(|depth| depth.as_str()).collect();
            properties[AnalyzeDepth::ARG] = json!({
                "type": "string",
                "enum": choices,
                "description": "How deep to go; the matching note says what each value does."
            });
            dispatch.push(AnalyzeDepth::ARG);
        }
        if self.accepts_place {
            // Declared per choice — `deep` never takes the folder — so only a
            // holder with the `depth` argument is told about it.
            let mut description = "Optional folder this round works in.".to_string();
            if self.kinds.contains(&AnalyzeDepth::Deep) {
                description.push_str(" `deep` ignores it.");
            }
            properties["place"] = json!({
                "type": "string",
                "description": description
            });
        }
        if !self.cancels() {
            return super::tool_params_schema(&properties, &dispatch);
        }
        // A second axis, not a second tool: declared beside the dispatch arguments
        // and validated on the call. A cancel carries no dispatch argument, so the
        // requirement is a disjunction the flat `required` envelope cannot express.
        properties[ACTION_ARG] = json!({
            "type": "string",
            "enum": [ACTION_CANCEL],
            "description": "Set to `cancel` to stop one of the rounds you dispatched, named by `job_id`, instead of dispatching a new one."
        });
        properties[JOB_ID_ARG] = json!({
            "type": "string",
            "description": "The round to cancel — the id from its dispatch acknowledgement."
        });
        let mut schema = super::tool_params_schema(&properties, &[]);
        schema["oneOf"] = json!([
            { "required": dispatch },
            { "required": [ACTION_ARG, JOB_ID_ARG] },
        ]);
        schema
    }

    /// Analysts are read-only (see `Role::tools()` for when even the read-only shell is
    /// withheld), and this classification is coupled to that: side-effecting Analyst tools would
    /// force a reconsideration. A cancel deletes rows, but only of a round acknowledged in an
    /// EARLIER tool round, so it never acts on a round dispatched beside it.
    fn side_effects(&self) -> bool {
        false
    }

    /// Sandwich-truncating it would silently drop findings and verification verdicts; it is
    /// bounded by the analyst round, never by the output budget.
    fn preserve_full_output(&self) -> bool {
        true
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        // Read first: a cancel takes no message, depth or place, so it is never
        // refused for a dispatch argument it does not carry.
        if self.cancels() {
            match super::get_opt_str_strict(&args, ACTION_ARG)? {
                None => {}
                Some(action) if action == ACTION_CANCEL => {
                    let job_id = super::get_str(&args, JOB_ID_ARG)?.trim();
                    anyhow::ensure!(
                        !job_id.is_empty(),
                        "usage: `{JOB_ID_ARG}` must name the round to cancel"
                    );
                    // Ownership is the calling SESSION, not the user: the round's
                    // delivery address is compared against it. A workspace-scoped
                    // Manager session is driven by pipeline notifications, whose
                    // turns carry no user (`tool_identity` would refuse them).
                    let caller = crate::agent::tool_agent_id()?;
                    let outcome = crate::analyze_cancel::cancel_round(job_id, &caller).await?;
                    return Ok(crate::analyze_cancel::answer(&outcome, job_id));
                }
                Some(action) => {
                    return Err(super::refuse_choice(ACTION_ARG, action, &[ACTION_CANCEL]));
                }
            }
        }
        let message = super::get_dispatch_message(&args)?;
        let depth = super::chosen_kind(
            &args,
            AnalyzeDepth::ARG,
            &self.kinds,
            AnalyzeDepth::as_str,
            AnalyzeDepth::parse,
        )?;

        match depth {
            AnalyzeDepth::Quick => {
                // Only a quick round takes `place`, and only a durable background
                // round can honour it. A refused place dispatches nothing and never
                // falls back to the caller's own workspace.
                let place = if self.accepts_place {
                    super::delegation::resolve(super::delegation::DelegationKind::Analyze, &args)
                        .await?
                } else {
                    None
                };
                if self.dispatch_mode.is_async() {
                    let job_id = crate::generate_id();
                    super::SyncDurableCore::Analyze.spawn_dispatch(
                        ws,
                        place.as_ref(),
                        message,
                        self.caller_role,
                        job_id.clone(),
                    );
                    return Ok(format!(
                        "Analysis dispatched (job {job_id}). The consolidated answer follows \
                         as an <analyze-tool-result> message unless you cancel the round first."
                    ));
                }
                // Sync path — a place is impossible here: `admin()`/`granted()`
                // are async-only.
                super::SyncDurableCore::Analyze
                    .run_sync_dispatch(ws, message, self.caller_role)
                    .await
            }
            AnalyzeDepth::Deep => Ok(super::research::dispatch_deep_analysis(
                ws,
                message,
                self.caller_role,
            )),
        }
    }
}

/// One durable analyze-round roster slot (pre-generated agent id + final task).
struct AnalyzeSlot {
    agent_id: String,
    task: String,
}

/// Spawn the analyze job + roster (one tx), run the analysts, checkpoint per-agent
/// outcomes, consolidate. `pre_done` are slots already completed by an earlier resume.
#[expect(clippy::too_many_lines)]
pub(crate) async fn run_analyze_with_job(
    ws: &Workspace,
    message: &str,
    args: crate::tools::CoreJobArgs<'_>,
) -> anyhow::Result<crate::tools::SyncCoreOutcome> {
    let crate::tools::CoreJobArgs {
        job_id,
        caller_role,
        user_name,
        channel,
        spawn,
        caller_agent_id,
        fail_on_checkpoint_error,
    } = args;
    let resume = spawn.is_none();
    let deadline = std::time::Instant::now() + round_timeout();
    // The signal may have fired before the round started: the caller was already
    // answered and the round's rows swept, so stop before writing anything.
    if crate::round_cancel::is_cancelled(job_id) {
        return Ok(crate::tools::SyncCoreOutcome::Cancelled);
    }

    let (slots, pre_done) = match &spawn {
        // Resume: reuse the stored roster (agent ids + final tasks) — never
        // regenerate ids, or the PK conflicts and the rows stop matching.
        None => {
            let rows =
                crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
            let slots: Vec<AnalyzeSlot> = rows
                .iter()
                .map(|r| AnalyzeSlot {
                    agent_id: r.agent_id.clone(),
                    task: r.task.clone(),
                })
                .collect();
            // Done slots from their stored outcomes; the rest re-run their tasks.
            let pre_done: Vec<(String, String)> = rows
                .iter()
                .filter(|r| r.status == crate::jobs::RowStatus::Done.as_str())
                .filter_map(|r| r.outcome.clone().map(|o| (r.agent_id.clone(), o)))
                .collect();
            (slots, pre_done)
        }
        // Fresh dispatch: roster and job BEFORE any analyst session write, with the
        // caller identity persisted so a later resume reaches the ORIGINAL caller.
        Some(spawn) => {
            let suffix = crate::generate_suffix();
            let angles = load_analyst_angles();
            let mut slots: Vec<AnalyzeSlot> = Vec::with_capacity(PARALLEL_ANALYST_COUNT);
            let mut agents: Vec<crate::jobs::NewAgent> = Vec::with_capacity(PARALLEL_ANALYST_COUNT);
            for i in 0..PARALLEL_ANALYST_COUNT {
                let slot = analyst_slot(ws, &angles, &suffix, i, message);
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
                message,
                spawn.delivery_name,
                user_name,
                channel,
                caller_role,
                &agents,
                &crate::jobs::SpawnChild::Analyze,
                caller_agent_id,
                // The round's execution directory, so a restart resumes there.
                spawn.exec_dir,
                // The custom role's launch, so a resume rebuilds its branch envelope.
                spawn.custom_launch,
            )
            .await?;
            // A cancel can land while the roster is built or before this INSERT
            // commits. `spawn_job`'s gate refuses the row then, so this only turns the
            // stop into the round's outcome — a cancelled round is never resumed.
            if crate::round_cancel::is_cancelled(job_id) {
                return Ok(crate::tools::SyncCoreOutcome::Cancelled);
            }
            (slots, Vec::new())
        }
    };

    // Run the launched slots; reconstruct done slots from stored outcomes.
    let fresh_slots: Vec<&AnalyzeSlot> = slots
        .iter()
        .filter(|s| !pre_done.iter().any(|(id, _)| id == &s.agent_id))
        .collect();
    // Round key = the durable job_id, stable across resume so a resumed round's
    // analysts and consolidation call group together.
    let fresh_runs = run_analyze_slots(ws, &fresh_slots, deadline, resume, job_id, message).await;

    // Cancelled as a group while the analysts ran, so their runs already carry the
    // stopped outcome: no checkpoint write, extraction or consolidation, and the
    // partial findings never become a result.
    if crate::round_cancel::is_cancelled(job_id) {
        return Ok(crate::tools::SyncCoreOutcome::Cancelled);
    }

    let mut runs: Vec<AnalyzeRun> = Vec::with_capacity(slots.len());
    let mut fresh_iter = fresh_runs.into_iter();
    for slot in &slots {
        if let Some((_, raw)) = pre_done.iter().find(|(id, _)| id == &slot.agent_id) {
            // Done slot from the stored response; the fresh agent pre-loads the
            // persisted session so extract_findings keeps KV-cache compatibility.
            let mut agent = crate::Agent::new(
                slot.agent_id.clone(),
                crate::Role::Analyst,
                ws,
                None,
                String::new(),
                String::new(),
                false,
                Some(crate::agent::registry::ParentKey::AnalyzeRound(
                    job_id.to_string(),
                )),
                Some(message.to_string()),
                None,
            );
            let _ = agent.session.init(&slot.agent_id).await;
            runs.push(AnalyzeRun::settle(agent, Some(raw.clone())));
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
            AnalyzeRun::Completed { response, .. } => (crate::jobs::RowStatus::Done, response),
            AnalyzeRun::Failed { reason } => (crate::jobs::RowStatus::Failed, reason),
        };
        if let Err(e) = crate::jobs::write_agent_outcome(
            conn,
            job_id,
            &slot.agent_id,
            status,
            Some(outcome.as_str()),
        )
        .await
        {
            // Sync fails the tool call on a checkpoint DB error so the model retries;
            // async/boot-resume warn and continue (outcomes are recomputable).
            if fail_on_checkpoint_error {
                return Err(e.context("failed to checkpoint analyze outcome"));
            }
            tracing::warn!(job = %job_id, error = %e, "Failed to checkpoint analyze outcome");
        }
    }

    // Drain cut the round mid-flight: skip the consolidate LLM call (no new LLM work
    // during the drain) — the checkpointed outcomes are reused by the next resume.
    if crate::shutdown::aborting() {
        return Ok(crate::tools::SyncCoreOutcome::DrainCut);
    }

    // The consolidation owns the round's last model calls, so a cancel landing
    // anywhere inside it stops the round instead of yielding a result.
    Ok(
        match consolidate_analyst_runs(ws, message, runs, deadline, job_id).await {
            Ok(Some(text)) => crate::tools::SyncCoreOutcome::Terminal(Ok(text)),
            Ok(None) => crate::tools::SyncCoreOutcome::Cancelled,
            Err(e) => crate::tools::SyncCoreOutcome::Terminal(Err(e)),
        },
    )
}

/// Run the slots concurrently, deadline-bounded. `round_key` (the durable `job_id`, or
/// a fresh suffix on the sync path) registers every analyst with `AnalyzeRound` so the
/// Running Agents view groups them with the round's consolidation call.
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
                // Session non-emptiness discriminator: a resumed slot whose session
                // already holds the task continues with an empty message (no duplicate
                // task prompt); a missing/empty session dispatches with the stored task.
                let has_session = resume && crate::session::store().has_content(&agent_id).await;
                // Dynamic resume (resume + conditional message), not a default dispatch.
                run_agent(
                    agent_id,
                    crate::Role::Analyst,
                    &ws,
                    None,
                    &[if has_session { "" } else { task.as_str() }],
                    String::new(),
                    String::new(),
                    false,
                    None,
                    resume,
                    Some(round),
                    Some(crate::agent::registry::ParentKey::AnalyzeRound(round_key)),
                    Some(question),
                    None,
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
            RoundMember::Done((agent, response)) => AnalyzeRun::settle(agent, response),
            RoundMember::TimedOut => AnalyzeRun::Failed {
                reason: "timeout: analyst still running when the round deadline expired"
                    .to_string(),
            },
            RoundMember::Panicked => AnalyzeRun::Failed {
                reason: "panic: analyst task panicked".to_string(),
            },
            RoundMember::Cancelled => AnalyzeRun::Failed {
                reason: "cancelled: analyst task was cancelled".to_string(),
            },
        })
        .collect()
}

/// Boot-resume a durable round via
/// [`crate::tools::SyncDurableCore::resume_durable_round`].
pub(crate) async fn resume_analyze_round(job_id: &str, ws: &Workspace, exec_dir: Option<&str>) {
    crate::tools::SyncDurableCore::Analyze
        .resume_durable_round(job_id, ws, exec_dir)
        .await;
}

/// A claim with source and confidence, shared with the deep investigation.
#[expect(clippy::struct_field_names)] // field name matches the JSON schema key
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

/// Per-analyst outcome after extraction — three mutually-exclusive states mirroring
/// `pipeline::ParallelVerdict`, so "no response" and "parse failure" cannot be confused.
enum AnalystOutcome {
    /// No response (crashed, cancelled, empty output); carries the scrubbed reason.
    NoResponse(String),
    /// Agent responded and structured extraction succeeded.
    Findings {
        raw: String,
        findings: AnalystFindings,
    },
    /// Responded, but extraction failed after retries; the raw response is kept for
    /// the fail-open dump.
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
    /// Cancelled externally (e.g. shutdown), not by the round deadline — kept distinct
    /// from [`RoundMember::TimedOut`] so the surfaced reason stays accurate.
    Cancelled,
}

/// Await all round tasks under one overall deadline. Pending members are aborted as
/// [`RoundMember::TimedOut`] (warn-logged), panicked ones as [`RoundMember::Panicked`]
/// with their message logged; completed members keep their results, so a stuck analyst
/// never blocks the others' delivery.
///
/// Members are `tokio::spawn`ed for panic isolation, so dropping this future mid-await
/// leaves them detached to self-termination. Reachable only during shutdown (tools run
/// to completion on agent cancel), where the runtime ends anyway.
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
        // member still running at the deadline. Biased select prefers a ready
        // handle, so members finishing just before it keep their real outcome.
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
    // Every exit path fills every slot exactly once — by construction.
    out.into_iter()
        .map(|m| m.expect("every round member resolves exactly once"))
        .collect()
}

/// Default bound on a round's consolidation wait. Hours-scale: normal analyst work
/// finishes in minutes, so only a pathological stall hits it.
const DEFAULT_ROUND_TIMEOUT_SECS: u64 = 3 * 60 * 60;

/// Round consolidation bound for [`await_round_members`]; env-overridable for tuning.
pub(crate) fn round_timeout() -> Duration {
    crate::util::env_duration_secs("MAHBOT_ROUND_TIMEOUT_SECS", DEFAULT_ROUND_TIMEOUT_SECS)
}

/// One analyst's outcome in a parallel round. `Completed` always carries text; every
/// way of producing none (deadline, panic, cancel, empty response) is [`Self::Failed`]
/// with its reason surfaced, never silent.
#[expect(clippy::large_enum_variant)] // Agent is inherently large; the enum is transient
enum AnalyzeRun {
    Completed { agent: Agent, response: String },
    Failed { reason: String },
}

impl AnalyzeRun {
    /// A member that finished without text is a [`Self::Failed`] carrying the agent's
    /// own recorded reason, so the real error chain reaches the caller.
    fn settle(agent: Agent, response: Option<String>) -> Self {
        match response.filter(|text| !text.trim().is_empty()) {
            Some(response) => Self::Completed { agent, response },
            None => Self::Failed {
                reason: agent.failure_reason("no-response: analyst produced no response"),
            },
        }
    }
}

/// Verdict of one targeted claim verification, via [`dispatch_claim_verifiers`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationVerdict {
    pub verdict: String,
    pub evidence: String,
}

/// One targeted claim's outcome: merged into the deep-research run summary (its
/// telemetry fields feed the query ledger) or rendered in the analyze verification section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VerificationResult {
    pub claim: String,
    pub verdict: String,
    pub evidence: String,
    pub tool_calls: usize,
    pub searches: usize,
    pub queries: Vec<String>,
}

impl VerificationResult {
    /// Fail-open: the claim could not be verified (deadline, panic, no/empty response,
    /// extraction failure, budget exhaustion). Canonical "unresolved" verdict string.
    pub(crate) fn unresolved(
        claim: &str,
        evidence: impl Into<String>,
        tool_calls: usize,
        searches: usize,
        queries: Vec<String>,
    ) -> Self {
        Self {
            claim: claim.to_string(),
            verdict: "unresolved".to_string(),
            evidence: evidence.into(),
            tool_calls,
            searches,
            queries,
        }
    }
}

/// Reject verdicts outside the accepted vocabulary — fail-closed inside the retry loop.
fn validate_verification_verdict(v: &VerificationVerdict) -> Result<(), String> {
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

/// Cap on verification analysts per gate (one bounded pass): the research tool caps
/// here, the analyze round needs at most 2 ([`verification_units`]). Bounds the shared
/// dispatch, never a per-run analyst budget.
pub(crate) const VERIFY_MAX_ANALYSTS: usize = 4;

/// The decorrelation angles asset, dealt out across the analyst cohort.
pub(crate) fn load_analyst_angles() -> Vec<String> {
    load_prompt_sections("analyze/angles.md")
}

/// An analyst's roster slot: the analyze plus its share of the decorrelation angles.
/// KV-cache discipline: vary ONLY the user message — never model/effort/tools.
fn analyst_slot(
    ws: &Workspace,
    angles: &[String],
    suffix: &str,
    i: usize,
    analyze: &str,
) -> AnalyzeSlot {
    let assigned = crate::prompt::deal_sections(angles, PARALLEL_ANALYST_COUNT, i);
    let task = if assigned.is_empty() {
        analyze.to_string()
    } else {
        format!("{analyze}\n\n{assigned}")
    };
    AnalyzeSlot {
        agent_id: format!("analyze_{}_{}_{}_analyst", ws.name, suffix, i),
        task,
    }
}

/// 0 valid → error, 1 valid → raw passthrough, ≥2 valid → extract findings and run the
/// shared grouping pass (a single parseable source skips grouping). A failed member is
/// the single spelling of "no response". `Ok(None)` is a round cancelled before this
/// phase's first model call: no text produced, nothing delivered.
async fn consolidate_analyst_runs(
    ws: &Workspace,
    analyze: &str,
    runs: Vec<AnalyzeRun>,
    deadline: std::time::Instant,
    round_key: &str,
) -> Result<Option<String>> {
    // Cancelled while the analysts ran: nothing new starts, so no extraction call and
    // the partial findings never become a result.
    if crate::round_cancel::is_cancelled(round_key) {
        return Ok(None);
    }
    // Drain-watch coverage: extraction runs while the analyst Agents are registered, which keeps
    // the drain-watch from firing; the grouping pass runs with zero live agents and is covered by
    // the guard inside the shared consensus core (`run_grouping_repair`). `consolidate_findings`
    // re-checks the drain flag before grouping, closing the sync-only window in between. No phase
    // guard here: it would duplicate the core's row and register a phantom one on 0/1-valid
    // rounds where no grouping call runs.
    let valid_count = runs
        .iter()
        .filter(|r| matches!(r, AnalyzeRun::Completed { .. }))
        .count();
    match valid_count {
        0 => {
            // Every run is a failure here (a completed run always carries text).
            let reasons: Vec<&str> = runs
                .iter()
                .filter_map(|r| match r {
                    AnalyzeRun::Failed { reason } => Some(reason.as_str()),
                    AnalyzeRun::Completed { .. } => None,
                })
                .collect();
            Err(super::internal_fault(&format!(
                "all parallel analysts failed to produce a response ({})",
                reasons.join("; ")
            )))
        }
        1 => Ok(Some(
            runs.into_iter()
                .find_map(|r| match r {
                    AnalyzeRun::Completed { response, .. } => Some(response),
                    AnalyzeRun::Failed { .. } => None,
                })
                .expect("exactly one valid response"),
        )),
        _ => {
            let outcomes = extract_findings(runs, deadline, round_key).await;
            consolidate_findings(ws, analyze, outcomes, round_key, deadline).await
        }
    }
}

/// Extract structured findings from each valid response while the agent is still alive
/// (see [`Agent::extract_verdict`] for the KV-cache rationale). Fail-open: a parse
/// failure keeps the raw response. The wait shares the round-wide `deadline` — a stuck
/// extraction is reported as no-response — and `round_key`'s cancel signal abandons
/// each extraction.
async fn extract_findings(
    runs: Vec<AnalyzeRun>,
    deadline: std::time::Instant,
    round_key: &str,
) -> Vec<AnalystOutcome> {
    let extraction_prompt = load_prompt("extraction/findings.md");
    let handles: Vec<_> = runs
        .into_iter()
        .map(|run| {
            let extraction_prompt = extraction_prompt.clone();
            let round_key = round_key.to_string();
            tokio::spawn(async move {
                let (agent, raw) = match run {
                    AnalyzeRun::Completed { agent, response } => (agent, response),
                    AnalyzeRun::Failed { reason } => {
                        return AnalystOutcome::NoResponse(crate::util::scrub_credentials(&reason));
                    }
                };
                // Nothing new starts for a cancelled round; the extraction loop re-asks
                // on a parse failure, so the whole chain is abandoned at the signal. The
                // outcome is discarded by the phase gate either way. `biased` polls the
                // signal first, so it wins over an extraction ready to return.
                let extracted = tokio::select! {
                    biased;
                    () = crate::round_cancel::cancelled(&round_key) => {
                        return AnalystOutcome::NoResponse(
                            "cancelled: findings extraction task was cancelled".to_string(),
                        );
                    }
                    result = agent
                        .extract_verdict::<AnalystFindings>(&extraction_prompt, None, None) => result,
                };
                match extracted {
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
                "timeout: findings extraction still running when the round deadline expired"
                    .to_string(),
            ),
            RoundMember::Panicked => {
                AnalystOutcome::NoResponse("panic: findings extraction task panicked".to_string())
            }
            RoundMember::Cancelled => AnalystOutcome::NoResponse(
                "cancelled: findings extraction task was cancelled".to_string(),
            ),
        })
        .collect()
}

/// Claim texts keyed by the ORIGINAL outcome index; a failed agent's slot stays empty
/// so the id space matches the input. Per-agent duplicates are NOT deduped: two
/// identical claims from one agent are two distinct ids.
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

/// The fail-open deliverable: the numbered claim list plus raw analyst dumps, headed by
/// a `marker` naming why consolidation produced no groups (head-placed so it stays
/// visible). The total-extraction-failure path (`n_valid == 0`) is the deliberate
/// exception, rendering failures in place of the claim list.
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

/// Build the per-agent claim lists and run the shared repair-mode grouping pass
/// (semantic grouping + contradiction judgment). ≥2 valid responses group; the contract
/// is summary + groups + optional ungrouped list, brackets computed from distinct cited
/// agent ids. Partial success (≥1 frozen group) is delivered as-is; only an exhaustion
/// with zero frozen groups keeps the flat claim list with an explicit marker — never a
/// fabricated consensus. `Ok(None)` is a round cancelled before this phase's first
/// model call or while it ran.
async fn consolidate_findings(
    ws: &Workspace,
    analyze: &str,
    outcomes: Vec<AnalystOutcome>,
    round_key: &str,
    deadline: std::time::Instant,
) -> Result<Option<String>> {
    // A round cancelled while its analysts ran produces no text at all — not even the
    // fail-open claim list, which would render a partial result for a caller already
    // told the round was stopped.
    if crate::round_cancel::is_cancelled(round_key) {
        return Ok(None);
    }
    let items_by_agent = claims_per_agent(&outcomes);
    let n_valid = items_by_agent.iter().filter(|l| !l.is_empty()).count();
    if n_valid == 0 {
        // No parseable claims: fail open with raw reports and the per-analyst reasons.
        // The marker distinguishes extraction failures from zero-claim reports.
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
        return Ok(Some(format!(
            "## Analyst Reports\n\n({marker})\n\n{failures}\n\n{raw_dump}"
        )));
    }
    if n_valid == 1 {
        // A single parseable source cannot produce agreement brackets or
        // contradictions, so skip the provider call and deliver the flat claim list.
        tracing::info!("Single-analyst consolidation — skipping grouping pass");
        return Ok(Some(render_unconsolidated_fallback(
            "only one analyst produced parseable claims — grouping skipped",
            &items_by_agent,
            &outcomes,
        )));
    }
    // Global flat ids across ALL agents, one per claim, in (agent, claim) order.
    let material = crate::consensus::numbered_items_material(&items_by_agent);
    let user =
        format!("# Original Question\n\n{analyze}\n\n# Agent Claims (id-numbered)\n\n{material}");

    let derived = crate::agent::role_chat_params(Role::Analyst);
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
        derived.model,
        derived.reasoning_effort,
        derived.provider_order,
        Some(DEFAULT_MAX_TOKENS),
    );

    let table = crate::consensus::ItemTable::new(&items_by_agent);
    let parent = Some(crate::agent::registry::ParentKey::AnalyzeRound(
        round_key.to_string(),
    ));
    // Drain fired in the extraction-end → grouping-start gap (analyst agents dropped,
    // consensus guard not yet registered): never start an LLM call once the drain begins
    // — deliver the fail-open claim list instead, without the doomed call.
    if crate::shutdown::aborting() {
        tracing::warn!(
            "Analyst consolidation skipped — shutdown/drain in progress; delivering raw claim list"
        );
        return Ok(Some(render_unconsolidated_fallback(
            "unconsolidated — shutdown during consolidation",
            &items_by_agent,
            &outcomes,
        )));
    }
    // The same stop at the phase's own model calls: a cancel landing after the entry
    // gate abandons the pass rather than starting another repair round. `biased` polls
    // the signal without the random branch choice, so a fired signal always wins.
    let outcome = tokio::select! {
        biased;
        () = crate::round_cancel::cancelled(round_key) => return Ok(None),
        outcome = crate::consensus::run_grouping_repair(
            ws,
            "consolidate",
            request,
            &items_by_agent,
            parent,
            Some(analyze.to_string()),
        ) => outcome,
    };
    match outcome {
        crate::consensus::RepairOutcome::Repaired { output, references } => {
            // Annotation-only verification of disputed groups: fresh analysts re-check
            // the contested findings. The grouped analysis is never re-run.
            let verification = verify_disputed_groups(
                ws, analyze, &output, &table, &outcomes, round_key, deadline,
            )
            .await;
            Ok(Some(render_analyze_groups(
                &output,
                &references,
                &table,
                n_valid,
                &outcomes,
                &verification,
            )))
        }
        crate::consensus::RepairOutcome::Fallback => {
            tracing::warn!("Analyst consolidation failed — delivering raw claim list");
            Ok(Some(render_unconsolidated_fallback(
                "unconsolidated — consolidation failed",
                &items_by_agent,
                &outcomes,
            )))
        }
    }
}

/// The output contract: summary + groups (heading, contradiction flag, members citing
/// item ids) + ungrouped remainder + DISPUTED cross-references from remainder items to
/// frozen groups. Brackets [n/N] come from distinct cited agent ids; DISPUTED appears
/// only when the contradiction flag is true. A cited claim's self-reported caveats are
/// member metadata, never contradictions. `verification` is appended after the grouped
/// analysis so that is never disturbed.
#[must_use]
fn render_analyze_groups(
    output: &crate::consensus::GroupingOutput,
    references: &[crate::consensus::GroupingReference],
    table: &crate::consensus::ItemTable<'_>,
    n_valid: usize,
    outcomes: &[AnalystOutcome],
    verification: &str,
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
    if !verification.is_empty() {
        let _ = write!(out, "\n\n{verification}");
    }
    out
}

/// A cited member's self-reported caveats (the claim's own `contradictions` field —
/// analyst self-report, NOT a group contradiction), resolved via the item table's
/// (agent, per-agent index), never text equality.
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

/// The flat numbered claim list — the grouping-pass input of the fail-open deliverable.
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

/// Normalize a claim for deterministic comparisons (QueryLedger, dedup).
pub(crate) fn normalize_claim(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Max of two confidence tiers ("low" < "medium" < "high").
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

/// One claim targeted for verification, plus the material fed to the fresh analyst.
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

/// Search/web_search query strings and tool-call counts from session-history messages,
/// for the query ledger and run summary. The deep investigation also feeds it the
/// persisted history of deadline-aborted analysts.
pub(crate) fn extract_query_telemetry_from_history(
    history: &[ChatMessage],
) -> (usize, usize, Vec<String>) {
    let mut tool_calls = 0usize;
    let mut searches = 0usize;
    let mut queries = Vec::new();
    for msg in history {
        let Some(calls) = crate::session::decoded_tool_calls(msg) else {
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

/// Query strings and tool-call counts from an agent's session history.
pub(crate) fn extract_query_telemetry(agent: &Agent) -> (usize, usize, Vec<String>) {
    extract_query_telemetry_from_history(agent.session.history())
}

/// Dispatch one fresh Analyst per target in parallel, bounded by
/// [`VERIFY_MAX_ANALYSTS`]. `id_prefix` seeds the agent IDs, `task_extra` is appended to
/// every task (e.g. the query ledger) and the wait shares the round-wide `deadline`.
///
/// `parent_key` names the parent invocation the verifiers group under — the
/// running-agents view and research-cancel sweep use it, so the caller must supply it,
/// never hardcode it.
///
/// Returns the results PLUS the dispatched agent IDs: the deep-research sanitizer needs
/// them at dispatch time, since successful writers never appear in wrap-up snapshots,
/// and riding the return value keeps this helper free of a write-only out-parameter.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn dispatch_claim_verifiers(
    ws: &Workspace,
    id_prefix: &str,
    targets: &[VerificationTarget],
    task_extra: &str,
    deadline: std::time::Instant,
    resume: bool,
    parent_key: Option<crate::agent::registry::ParentKey>,
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
            let parent_key = parent_key.clone();
            let question = question.to_string();
            move |round| async move {
                run_claim_verifier(
                    &ws,
                    &agent_id,
                    &claim_text,
                    &task,
                    &extraction_prompt,
                    round,
                    parent_key,
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
                VerificationResult::unresolved(
                    &targets[i].claim,
                    "verifier never completed (round deadline or panic)",
                    0,
                    0,
                    Vec::new(),
                )
            }
        })
        .collect();
    (results, dispatched)
}

/// One verifier: a fresh Analyst researches the claim and returns its structured
/// verdict. Any failure yields "unresolved" — fail-open. `question` is threaded as the
/// run group's header label (purely presentational); `parent_key` is supplied by the
/// dispatcher so the verifier groups under the correct parent invocation.
#[expect(clippy::too_many_arguments)]
async fn run_claim_verifier(
    ws: &Workspace,
    agent_id: &str,
    target_claim: &str,
    task: &str,
    extraction_prompt: &str,
    round: crate::agent::RoundOpts,
    parent_key: Option<crate::agent::registry::ParentKey>,
    question: &str,
) -> VerificationResult {
    let (agent, response) = run_default_agent(
        agent_id,
        Role::Analyst,
        ws,
        task,
        false,
        Some(round),
        parent_key,
        Some(question.to_string()),
    )
    .await;
    // The telemetry fields ride every VerificationResult: deep research consumes them
    // for its run summary and query ledger, while analyze renders only
    // claim/verdict/evidence — one shared shape for a cheap extra scan.
    let (tool_calls, searches, queries) = extract_query_telemetry(&agent);
    let Some(raw) = response else {
        return VerificationResult::unresolved(
            target_claim,
            "verifier failed to produce a response",
            tool_calls,
            searches,
            queries,
        );
    };
    if raw.trim().is_empty() {
        return VerificationResult::unresolved(
            target_claim,
            "verifier produced an empty response",
            tool_calls,
            searches,
            queries,
        );
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
            // Keyed by the target claim, not the verifier's rephrasing, so report
            // reconciliation matches exactly.
            claim: target_claim.to_string(),
            verdict: v.verdict,
            evidence: v.evidence,
            tool_calls,
            searches,
            queries,
        },
        Err(e) => VerificationResult::unresolved(
            target_claim,
            format!("verification extraction failed: {e}"),
            tool_calls,
            searches,
            queries,
        ),
    }
}

/// Escape triple-backtick code fences to prevent markdown-structure
/// corruption in the consolidated output.
pub(crate) fn escape_fences(s: &str) -> String {
    s.replace("```", "\\`\\`\\`")
}

/// Exactly one disputed group → one unit (1 verifier). More → two roughly-equal parts
/// by group count, the FIRST taking the extra group on an odd count (2 verifiers). Each
/// unit gets exactly one fresh analyst, so verification cost is bounded here — no
/// per-run analyst budget.
fn verification_units<'a>(
    disputed: &[&'a crate::consensus::GroupingGroup],
) -> Vec<Vec<&'a crate::consensus::GroupingGroup>> {
    if disputed.len() <= 1 {
        return vec![disputed.to_vec()];
    }
    // ceil(n / 2) — the first part takes the extra group on an odd count.
    let first_count = disputed.len().div_ceil(2);
    let (first, second) = disputed.split_at(first_count);
    vec![first.to_vec(), second.to_vec()]
}

/// ONE verifiable proposition for a unit: the group's heading plus its member claims,
/// with sources and self-reported contradictions aggregated into the
/// [`VerificationTarget`] fields so the verifier sees the assertion and the evidence both
/// ways. A group with no resolvable members degrades to its bare heading.
fn synthesize_verification_target(
    unit: &[&crate::consensus::GroupingGroup],
    table: &crate::consensus::ItemTable<'_>,
    outcomes: &[AnalystOutcome],
) -> VerificationTarget {
    let mut claim_parts = Vec::new();
    let mut sources = Vec::new();
    let mut contradictions = Vec::new();
    for group in unit {
        let mut group_claims = Vec::new();
        for member in &group.members {
            if let Some((_, text)) = table.resolve(member.id) {
                group_claims.push(text.to_string());
            }
            if let Some((agent_idx, item_idx)) = table.resolve_index(member.id)
                && let Some(AnalystOutcome::Findings { findings, .. }) = outcomes.get(agent_idx)
                && let Some(claim) = findings.claims.get(item_idx)
            {
                if !claim.source.is_empty() {
                    sources.push(claim.source.clone());
                }
                contradictions.extend(claim.contradictions.iter().cloned());
            }
        }
        let claim = if group_claims.is_empty() {
            group.heading.clone()
        } else {
            format!("{}: {}", group.heading, group_claims.join("; "))
        };
        claim_parts.push(claim);
    }
    VerificationTarget::new(
        &claim_parts.join("\n"),
        &sources.join("; "),
        &contradictions.join("; "),
    )
}

/// The `## Verification` section; empty when no verifier produced a verdict. Newlines in
/// composite claims AND multi-line evidence render as semicolons so each bullet stays on
/// one line.
fn render_verification_section(results: &[VerificationResult]) -> String {
    if results.is_empty() {
        return String::new();
    }
    let mut out = String::from("## Verification");
    for v in results {
        let claim_line = escape_fences(&v.claim).replace('\n', "; ");
        let evidence_line = escape_fences(&v.evidence).replace('\n', "; ");
        let _ = write!(
            out,
            "\n- {claim_line} → **{}** — {evidence_line}",
            v.verdict,
        );
    }
    out
}

/// Verification runs only while ≤ half the round window ([`round_timeout`], the bound the
/// analyst round shares) has elapsed; past that it is skipped entirely, not forced
/// all-unresolved.
fn verification_window_open(deadline: std::time::Instant) -> bool {
    let Some(anchor) = deadline.checked_sub(round_timeout() / 2) else {
        return false;
    };
    std::time::Instant::now() <= anchor
}

/// Dispatch fresh verifiers over groups flagged `contradiction:true` (rendered
/// `[n/N · DISPUTED]`), 1 for a single disputed group and 2 over split parts for more.
/// Results are appended as `## Verification`; the grouped analysis is never modified and
/// grouping never re-run. Best-effort, never checkpointed.
async fn verify_disputed_groups(
    ws: &Workspace,
    analyze: &str,
    output: &crate::consensus::GroupingOutput,
    table: &crate::consensus::ItemTable<'_>,
    outcomes: &[AnalystOutcome],
    round_key: &str,
    deadline: std::time::Instant,
) -> String {
    // This gate is the phase's first statement, so a cancelled round raises no complaint
    // about itself and neither skip warning can fire. Logged at info, like the
    // consolidation gate.
    if crate::round_cancel::is_cancelled(round_key) {
        tracing::info!(
            round = %round_key,
            "Analyze verification skipped — the round was cancelled"
        );
        return String::new();
    }
    let disputed: Vec<&crate::consensus::GroupingGroup> =
        output.groups.iter().filter(|g| g.contradiction).collect();
    if disputed.is_empty() {
        return String::new();
    }
    if !verification_window_open(deadline) {
        tracing::warn!(
            disputed_groups = disputed.len(),
            "Analyze verification skipped — more than half the round window elapsed"
        );
        return String::new();
    }
    if crate::shutdown::aborting() {
        tracing::warn!(
            disputed_groups = disputed.len(),
            "Analyze verification skipped — shutdown/drain in progress"
        );
        return String::new();
    }
    let units = verification_units(&disputed);
    let targets: Vec<VerificationTarget> = units
        .iter()
        .map(|unit| synthesize_verification_target(unit, table, outcomes))
        .collect();
    tracing::info!(
        disputed_groups = disputed.len(),
        verifiers = units.len(),
        "Analyze verification round dispatching fresh analysts"
    );
    let prefix = format!("analyze_{}_verify", ws.name);
    let (results, _dispatched) = dispatch_claim_verifiers(
        ws,
        &prefix,
        &targets,
        "",
        deadline,
        false,
        Some(crate::agent::registry::ParentKey::AnalyzeRound(
            round_key.to_string(),
        )),
        analyze,
    )
    .await;
    render_verification_section(&results)
}

/// The raw response of every valid analyst in `### Report from Analyst N` sections; a
/// no-response analyst renders with its reason.
fn render_raw_analyst_dump(outcomes: &[AnalystOutcome]) -> String {
    let sections: Vec<String> = outcomes
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let body = match o {
                AnalystOutcome::Findings { raw, .. } | AnalystOutcome::ParseFailed { raw, .. } => {
                    escape_fences(raw)
                }
                AnalystOutcome::NoResponse(reason) => format!("no response: {reason}"),
            };
            format!("### Report from Analyst {}\n{body}", i + 1)
        })
        .collect();
    sections.join("\n\n")
}

/// Extraction-failure reasons for the fail-open marker.
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
    use crate::util::UnwrapPoison as _;
    use crate::util::test::{FakeProvider, assert_contains_not_contains, install_fake_provider};
    use crate::workspace::test_ws;
    use serde_json::json;
    use std::sync::Arc;

    /// A bare analyst run: a real `Agent` (no stores touched) settled from a raw response.
    fn test_analyst_run(ws: &Workspace, response: Option<&str>) -> AnalyzeRun {
        let agent = Agent::new(
            format!("analyze_test_{}", crate::generate_suffix()),
            Role::Analyst,
            ws,
            None,
            String::new(),
            String::new(),
            false,
            None,
            None,
            None,
        );
        AnalyzeRun::settle(agent, response.map(ToString::to_string))
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

    /// One scripted analyst round for [`consolidate_analyst_runs`].
    enum RoundMemberSpec {
        Response(Option<&'static str>),
        Failed(&'static str),
    }

    /// A delivered round: the members, and the text or failure reasons the caller gets.
    struct ConsolidateRunsCase {
        name: &'static str,
        members: &'static [RoundMemberSpec],
        /// `Err` lists the substrings the surfaced failure must name.
        expected: Result<&'static str, &'static [&'static str]>,
    }

    #[tokio::test]
    async fn consolidate_analyst_runs_cases() {
        use RoundMemberSpec::{Failed, Response};
        let cases = [
            ConsolidateRunsCase {
                name: "zero valid responses surface every failure reason",
                members: &[
                    Failed("analyst 0 still running when the round deadline expired"),
                    Failed("analyst 1 task panicked"),
                    Response(None),
                ],
                expected: Err(&[
                    "internal: all parallel analysts failed",
                    "deadline expired",
                    "panicked",
                    "analyst produced no response",
                ]),
            },
            ConsolidateRunsCase {
                name: "a single response is returned directly",
                members: &[
                    Response(Some("only answer")),
                    Response(None),
                    Response(None),
                ],
                expected: Ok("only answer"),
            },
            ConsolidateRunsCase {
                name: "empty responses are filtered like missing ones",
                members: &[
                    Response(Some("valid")),
                    Response(Some("")),
                    Response(Some("   ")),
                ],
                expected: Ok("valid"),
            },
        ];

        let ws = test_ws("/tmp/test_ws");
        for case in &cases {
            let runs = case
                .members
                .iter()
                .map(|spec| match spec {
                    RoundMemberSpec::Response(response) => test_analyst_run(&ws, *response),
                    RoundMemberSpec::Failed(reason) => test_failed_run(reason),
                })
                .collect();
            let result = consolidate_analyst_runs(
                &ws,
                "test question",
                runs,
                std::time::Instant::now() + Duration::from_mins(1),
                "test_round",
            )
            .await;
            match (&case.expected, result) {
                (Ok(expected), Ok(actual)) => assert_eq!(
                    actual.as_deref(),
                    Some(*expected),
                    "[{}] delivered the wrong text",
                    case.name
                ),
                (Err(needles), Err(err)) => {
                    let text = err.to_string();
                    for needle in *needles {
                        assert!(
                            text.contains(needle),
                            "[{}] error must name {needle:?}: {text}",
                            case.name
                        );
                    }
                }
                (_, other) => panic!("[{}] unexpected outcome: {other:?}", case.name),
            }
        }
    }

    /// A member stuck past the deadline is aborted (its task dropped, cancelling the
    /// work), a completed member keeps its value, and a panicking one is surfaced as
    /// `Panicked` rather than taken down silently.
    #[tokio::test]
    async fn await_round_members_cases() {
        struct AbortDetector(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for AbortDetector {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }

        // A stuck member aborts at the deadline; the completed one keeps its result.
        let aborted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = aborted.clone();
        let stuck = tokio::spawn(async move {
            let _detector = AbortDetector(flag);
            loop {
                tokio::time::sleep(Duration::from_hours(1)).await;
            }
        });
        let done = tokio::spawn(async { 42u32 });
        let deadline = std::time::Instant::now() + Duration::from_millis(300);
        let members = await_round_members(vec![stuck, done], deadline).await;
        assert!(
            matches!(members[0], RoundMember::TimedOut),
            "a stuck member times out: {members:?}"
        );
        assert!(
            matches!(members[1], RoundMember::Done(42)),
            "a completed member keeps its result: {members:?}"
        );
        // Aborting drops the task's locals, so the detector fires.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            aborted.load(std::sync::atomic::Ordering::SeqCst),
            "the stuck member task is aborted at the deadline, not leaked"
        );

        // A panicking member is surfaced, never dropped.
        let panicking = tokio::spawn(async { panic!("boom") });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let members = await_round_members(vec![panicking], deadline).await;
        assert!(
            matches!(members[0], RoundMember::Panicked),
            "a panicking member is surfaced: {members:?}"
        );
    }

    /// One [`render_analyze_groups`] scenario: outcomes, grouping output, and the strings
    /// the section must (not) carry.
    struct RenderGroupsCase {
        name: &'static str,
        outcomes: Vec<AnalystOutcome>,
        output: crate::consensus::GroupingOutput,
        references: Vec<crate::consensus::GroupingReference>,
        contains: &'static [&'static str],
        not_contains: &'static [&'static str],
    }

    #[expect(clippy::too_many_lines)] // one grouping shape per rendering rule
    #[test]
    fn render_analyze_groups_cases() {
        use crate::consensus::{GroupingGroup, GroupingMember, GroupingOutput, GroupingReference};

        let member = |id: usize| GroupingMember {
            id,
            ..Default::default()
        };
        let mut caveated = findings(vec![("alpha is true", "url1", "high")]);
        caveated.claims[0].contradictions = vec!["source B says alpha may be false".into()];

        let cases = vec![
            RenderGroupsCase {
                name: "brackets count distinct citing agents",
                // Global flat ids: 0 = a0 "alpha is true", 1 = a0 "beta is true",
                // 2 = a1 "alpha is true", 3 = a1 "gamma is true".
                outcomes: vec![
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
                ],
                output: GroupingOutput {
                    summary: "Two facts, one solo finding.".into(),
                    groups: vec![
                        GroupingGroup {
                            heading: "Alpha".into(),
                            contradiction: false,
                            members: vec![member(0), member(2)],
                        },
                        GroupingGroup {
                            heading: "Beta".into(),
                            contradiction: false,
                            members: vec![member(1)],
                        },
                    ],
                    ungrouped: vec![member(3)],
                },
                references: Vec::new(),
                contains: &[
                    "**Alpha** [2/2]",
                    "**Beta** [1/2]",
                    "**Other**",
                    "- Agent 1: gamma is true",
                ],
                not_contains: &["DISPUTED"],
            },
            RenderGroupsCase {
                name: "DISPUTED only on a contradiction",
                outcomes: vec![
                    AnalystOutcome::Findings {
                        raw: "r1".into(),
                        findings: findings(vec![("alpha is true", "url1", "high")]),
                    },
                    AnalystOutcome::Findings {
                        raw: "r2".into(),
                        findings: findings(vec![("alpha is false", "url2", "high")]),
                    },
                ],
                output: GroupingOutput {
                    summary: "Agents disagree on alpha.".into(),
                    groups: vec![GroupingGroup {
                        heading: "Alpha".into(),
                        contradiction: true,
                        members: vec![member(0), member(1)],
                    }],
                    ungrouped: Vec::new(),
                },
                references: Vec::new(),
                contains: &["**Alpha** [2/2 · DISPUTED]"],
                not_contains: &[],
            },
            RenderGroupsCase {
                name: "a member caveat is metadata, not a contradiction",
                outcomes: vec![
                    AnalystOutcome::Findings {
                        raw: "r1".into(),
                        findings: caveated,
                    },
                    AnalystOutcome::Findings {
                        raw: "r2".into(),
                        findings: findings(vec![("alpha is true", "url1", "high")]),
                    },
                ],
                output: GroupingOutput {
                    summary: "alpha is agreed.".into(),
                    groups: vec![GroupingGroup {
                        heading: "Alpha".into(),
                        contradiction: false,
                        members: vec![member(0), member(1)],
                    }],
                    ungrouped: Vec::new(),
                },
                references: Vec::new(),
                contains: &["caveat: source B says alpha may be false"],
                not_contains: &["DISPUTED"],
            },
            RenderGroupsCase {
                name: "a disputed remainder renders a code-computed cross-reference",
                outcomes: vec![
                    AnalystOutcome::Findings {
                        raw: "r1".into(),
                        findings: findings(vec![("safe", "url1", "high")]),
                    },
                    AnalystOutcome::Findings {
                        raw: "r2".into(),
                        findings: findings(vec![("actually unsafe", "url2", "high")]),
                    },
                ],
                output: GroupingOutput {
                    summary: "One consensus, one dispute.".into(),
                    groups: vec![GroupingGroup {
                        heading: "Safety".into(),
                        contradiction: false,
                        members: vec![member(0)],
                    }],
                    ungrouped: vec![member(1)],
                },
                references: vec![GroupingReference {
                    group: 0,
                    member: member(1),
                }],
                contains: &["Agent 1: actually unsafe [DISPUTED — contradicts group 0 \"Safety\"]"],
                not_contains: &[],
            },
        ];

        for case in cases {
            let items = claims_per_agent(&case.outcomes);
            let table = crate::consensus::ItemTable::new(&items);
            let text = render_analyze_groups(
                &case.output,
                &case.references,
                &table,
                2,
                &case.outcomes,
                "",
            );
            assert_contains_not_contains(case.name, &text, case.contains, case.not_contains);
        }
    }

    /// Run consolidation over the given outcomes with the provider outcomes scripted.
    async fn consolidate_with_script(
        outcomes: Vec<AnalystOutcome>,
        fake: FakeProvider,
    ) -> anyhow::Result<String> {
        let provider: Arc<dyn crate::Provider> = Arc::new(fake);
        let _guard = install_fake_provider(provider);
        let ws = test_ws("/tmp/test_ws");
        consolidate_findings(
            &ws,
            "test question",
            outcomes,
            "test_round",
            std::time::Instant::now() + std::time::Duration::from_mins(60),
        )
        .await
        .map(|text| text.expect("a live round delivers its consolidated text"))
    }

    /// Two analysts agreeing on one claim, plus a no-response member.
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

    /// One consolidation scenario: the inputs, the scripted provider, and the strings the
    /// delivered text must (not) carry.
    struct ConsolidationCase {
        name: &'static str,
        input: ConsolidationInput,
        fake: FakeProvider,
        contains: &'static [&'static str],
        not_contains: &'static [&'static str],
    }

    enum ConsolidationInput {
        /// Runs that still need findings extraction (which may fail).
        Runs(Vec<AnalyzeRun>),
        /// Already-extracted outcomes.
        Outcomes(Vec<AnalystOutcome>),
    }

    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn consolidation_outcome_cases() {
        use crate::retry::FailureClass;
        use ConsolidationInput::{Outcomes, Runs};

        // Synthesis retries (transport failures) exhaust → fail open.
        let exhausted = FakeProvider::new()
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body");
        // Extraction fails for every analyst → extraction fail-open.
        let extraction_exhausted = FakeProvider::new()
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body")
            .err(FailureClass::Transport, "error reading response body");
        // The grouping pass parses a strict id-based GroupingOutput: a faithful model
        // groups the two agreed claims and leaves nothing ungrouped.
        let one_group = r#"{"summary":"alpha is true.","groups":[{"heading":"Alpha","contradiction":false,"members":[{"id":0},{"id":1}]}],"ungrouped":[]}"#;
        let runs_ws = test_ws("/tmp/test_ws");

        let cases = vec![
            ConsolidationCase {
                name: "transport exhaustion fails open with every raw report",
                input: Outcomes(agreed_outcomes("report one", "report two")),
                fake: exhausted,
                contains: &[
                    "unconsolidated — consolidation failed",
                    "## Analyst Reports",
                    "### Report from Analyst 1",
                    "report one",
                    "### Report from Analyst 2",
                    "report two",
                    "no response: analyst produced no response",
                ],
                not_contains: &[],
            },
            ConsolidationCase {
                name: "an immediate non-retryable error fails open",
                input: Outcomes(vec![
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
                ]),
                fake: FakeProvider::new()
                    .err(FailureClass::NonRetryable, "insufficient balance"),
                contains: &["unconsolidated — consolidation failed", "alpha", "beta", "gamma"],
                not_contains: &[],
            },
            ConsolidationCase {
                name: "a faithful grouping pass synthesizes without the fail-open marker",
                input: Outcomes(agreed_outcomes("report a", "report b")),
                fake: FakeProvider::new().ok(one_group),
                contains: &["alpha is true"],
                not_contains: &["unconsolidated"],
            },
            ConsolidationCase {
                name: "a single parseable source skips the grouping pass",
                input: Outcomes(vec![
                    AnalystOutcome::Findings {
                        raw: "sole report".into(),
                        findings: findings(vec![("alpha is true", "url1", "high")]),
                    },
                    AnalystOutcome::ParseFailed {
                        raw: "unparseable report".into(),
                        failure: "parse failed".into(),
                    },
                ]),
                // No script — any provider call would fail.
                fake: FakeProvider::new(),
                contains: &[
                    "only one analyst produced parseable claims — grouping skipped",
                    "alpha is true",
                    "### Report from Analyst 2",
                    "unparseable report",
                ],
                not_contains: &[],
            },
            ConsolidationCase {
                name: "mixed transport and empty-response failures fail open",
                input: Outcomes(agreed_outcomes("only usable", "second")),
                fake: FakeProvider::new()
                    .err(FailureClass::Transport, "error reading response body")
                    .ok("")
                    .ok(""),
                contains: &[
                    "unconsolidated — consolidation failed",
                    "only usable",
                    "second",
                ],
                not_contains: &[],
            },
            ConsolidationCase {
                name: "extraction failure fails open with the raw reports",
                input: Runs(vec![
                    test_analyst_run(&runs_ws, Some("report one")),
                    test_analyst_run(&runs_ws, Some("report two")),
                    test_analyst_run(&runs_ws, None),
                ]),
                fake: extraction_exhausted,
                contains: &[
                    "unconsolidated — findings extraction failed",
                    "### Report from Analyst 1",
                    "report one",
                    "### Report from Analyst 2",
                    "report two",
                ],
                not_contains: &[],
            },
            ConsolidationCase {
                name: "partial success delivers the frozen group and the deterministic remainder",
                input: Outcomes(vec![
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
                ]),
                // ids: 0 = a0 "alpha is true", 1 = a0 "beta is true", 2 = a1 "alpha is
                // true". Round 2 leaves the id-1 item ungrouped → zero-progress stop.
                fake: FakeProvider::new()
                    .ok(
                        r#"{"summary":"alpha agreed, beta solo.","groups":[{"heading":"Alpha","contradiction":false,"members":[{"id":0},{"id":2}]}],"ungrouped":[{"id":1}]}"#,
                    )
                    .ok(r#"{"groups":[],"ungrouped":[{"id":1}]}"#),
                contains: &[
                    "**Alpha** [2/2]",
                    "**Other**",
                    "- Agent 0: beta is true",
                    "alpha agreed, beta solo.",
                ],
                not_contains: &["unconsolidated"],
            },
            ConsolidationCase {
                // A report far over the shared tool-output budget still reaches
                // the caller whole when the analyze tool formats it.
                name: "a long report is delivered verbatim",
                input: Outcomes(agreed_outcomes(&"lorem ipsum ".repeat(2_000), "second")),
                fake: FakeProvider::new()
                    .err(FailureClass::Transport, "error reading response body")
                    .err(FailureClass::Transport, "error reading response body")
                    .err(FailureClass::Transport, "error reading response body"),
                contains: &["unconsolidated — consolidation failed"],
                not_contains: &[],
            },
        ];

        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = test_ws("/tmp/test_ws");
        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(60);
        for case in cases {
            let provider: Arc<dyn crate::Provider> = Arc::new(case.fake);
            let _provider_guard = install_fake_provider(provider);
            let result = match case.input {
                ConsolidationInput::Runs(runs) => {
                    consolidate_analyst_runs(&ws, "test question", runs, deadline, "test_round")
                        .await
                }
                ConsolidationInput::Outcomes(outcomes) => {
                    consolidate_findings(&ws, "test question", outcomes, "test_round", deadline)
                        .await
                }
            };
            let text = result
                .unwrap_or_else(|err| panic!("[{}] consolidation errored: {err}", case.name))
                .expect("a live round delivers its consolidated text");
            assert_contains_not_contains(case.name, &text, case.contains, case.not_contains);
            // The tool's format_output returns the output verbatim — no head/tail
            // sandwich — so a marker and all findings reach the caller.
            let tool = AnalyzeTool::new(DispatchMode::Sync, Role::Engineer);
            assert_eq!(
                tool.format_output(&text),
                text,
                "[{}] analyze output must be preserved in full",
                case.name
            );
        }
    }

    /// What the async path delivers to the caller's agent channel.
    enum AsyncEnvelopeInput {
        /// The real fail-open consolidation result (transport retries exhausted).
        FailOpenAfterTransportExhaustion,
        /// A failed sub-agent.
        SubAgentError,
    }

    /// One async analyze envelope: the delivered result, the place (if any) and
    /// the exact shape the caller must receive.
    struct AsyncEnvelopeCase {
        name: &'static str,
        input: AsyncEnvelopeInput,
        place: Option<&'static str>,
        contains: &'static [&'static str],
        not_contains: &'static [&'static str],
    }

    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn async_envelope_cases() {
        for case in [
            AsyncEnvelopeCase {
                name: "the fail-open marker and the place directory reach the caller",
                input: AsyncEnvelopeInput::FailOpenAfterTransportExhaustion,
                place: Some("/tmp/round-place"),
                contains: &[
                    "<analyze-tool-result>",
                    "unconsolidated — consolidation failed",
                    "raw report",
                    "\n\nJob: job123\n\nDirectory: /tmp/round-place\n\n",
                ],
                not_contains: &[],
            },
            AsyncEnvelopeCase {
                name: "a failed sub-agent is wrapped with its error and no directory",
                input: AsyncEnvelopeInput::SubAgentError,
                place: None,
                contains: &[
                    "<analyze-tool-result>",
                    "An error occurred: sub-agent exploded",
                ],
                not_contains: &["Directory:"],
            },
        ] {
            let _policy_guard =
                crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
            let result: anyhow::Result<String> = match case.input {
                AsyncEnvelopeInput::FailOpenAfterTransportExhaustion => {
                    let fake = FakeProvider::new()
                        .err(crate::retry::FailureClass::Transport, "down")
                        .err(crate::retry::FailureClass::Transport, "down")
                        .err(crate::retry::FailureClass::Transport, "down");
                    consolidate_with_script(agreed_outcomes("raw report", "raw report 2"), fake)
                        .await
                }
                AsyncEnvelopeInput::SubAgentError => Err(anyhow::anyhow!("sub-agent exploded")),
            };
            // Drives the production builder (not a manual re-wrap), so the
            // envelope + marker shape asserted here is the one that ships.
            let envelope = crate::tools::SyncDurableCore::Analyze.build_async_message(
                "job123",
                &result,
                case.place,
                &crate::workspace::test_ws("/tmp/round-delivery"),
            );
            let mut needles: Vec<&str> = case.contains.to_vec();
            needles.push("</analyze-tool-result>");
            assert_contains_not_contains(case.name, &envelope, &needles, case.not_contains);
            assert!(
                envelope.ends_with("</analyze-tool-result>"),
                "[{}] envelope must close: {envelope}",
                case.name
            );
        }
    }

    /// One boot resume of a crashed async analyze: the job row and roster the
    /// crash left behind, the place it was running in and what the delivered
    /// envelope must say.
    struct ResumeRoundCase {
        name: &'static str,
        job_id: &'static str,
        workspace: &'static str,
        /// The place directory the crashed round recorded, if it had one.
        place: Option<&'static str>,
        /// The analyst's stored outcome, when the crash left a completed slot.
        outcome: Option<&'static str>,
        /// Substrings the delivered envelope must carry.
        contains: &'static [&'static str],
    }

    /// Boot resume must REUSE the existing job + roster (never re-spawn: the PK would conflict
    /// and fresh analyst ids would not match the stored rows, losing outcomes): a done slot
    /// reconstructs from its stored outcome, and the round terminalizes into an envelope
    /// delivered to the ORIGINAL caller. A place round whose directory is gone must END with a
    /// plain failure — never left launched and silent forever.
    ///
    /// Serialized with the drain-flag writers: `resume_analyze_round` reads the process-global
    /// drain flag and aborts early while it is set.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn resume_round_cases() {
        crate::util::test::init_management_test_stores().await;
        for case in [
            // The job row as a crashed dispatch leaves it: caller identity persisted,
            // one done slot with a stored outcome.
            ResumeRoundCase {
                name: "a crashed round is resumed over its own roster",
                job_id: "analyze_job_resume_1",
                workspace: "/tmp/test_ws_resume_analyze",
                place: None,
                outcome: Some("completed analyst response"),
                contains: &["completed analyst response"],
            },
            ResumeRoundCase {
                name: "a place round whose directory vanished fails loudly",
                job_id: "analyze_job_resume_place",
                workspace: "/tmp/test_ws_resume_place",
                place: Some("/definitely/not/here/analyze-place"),
                outcome: None,
                contains: &[
                    "no longer available",
                    "Directory: /definitely/not/here/analyze-place",
                ],
            },
        ] {
            let ws = test_ws(case.workspace);
            let conn = &crate::session::store().conn;
            let agent_id = format!("{}_analyst", case.job_id);
            crate::jobs::spawn_job(
                conn,
                case.job_id,
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
                None,
                case.place,
                None,
            )
            .await
            .unwrap();
            if let Some(outcome) = case.outcome {
                crate::jobs::write_agent_outcome(
                    conn,
                    case.job_id,
                    &agent_id,
                    crate::jobs::RowStatus::Done,
                    Some(outcome),
                )
                .await
                .unwrap();
            }

            // Resume must not hit the PK conflict, must reuse the stored roster slot,
            // then terminalize into a durable envelope.
            resume_analyze_round(case.job_id, &ws, case.place).await;

            let job_rows = conn
                .query(
                    "SELECT id FROM jobs WHERE id = ?1",
                    crate::db::params![case.job_id],
                )
                .await
                .unwrap();
            assert_eq!(
                job_rows.len(),
                0,
                "[{}] the round must terminalize, not hang",
                case.name
            );
            let pending = conn
                .query(
                    "SELECT envelope FROM pending_jobs WHERE id = ?1",
                    crate::db::params![case.job_id],
                )
                .await
                .unwrap();
            assert_eq!(
                pending.len(),
                1,
                "[{}] the delivery is persisted with the job id",
                case.name
            );
            let envelope: crate::agent::message_router::AgentJob =
                serde_json::from_str(&pending[0].get::<String>(0).unwrap()).unwrap();
            // Delivered to the ORIGINAL caller (Assistant, not Manager), keyed to that
            // caller's workspace — never to the place.
            assert_eq!(
                envelope.role,
                crate::Role::Assistant,
                "[{}] the original caller role",
                case.name
            );
            assert_eq!(envelope.user_name, "caller-user", "[{}]", case.name);
            assert_eq!(
                envelope.workspace_name, ws.name,
                "[{}] the delivery key stays the caller's own workspace, never the place",
                case.name
            );
            for needle in case.contains {
                assert!(
                    envelope.content.contains(needle),
                    "[{}] the envelope must say {needle:?}: {}",
                    case.name,
                    envelope.content
                );
            }
        }
    }

    /// A provider parking every chat call until the test opens the gate, so durable state
    /// can be observed mid-round. The Assistant consumer the envelope routes into is parked
    /// indefinitely: it would otherwise reclaim (delete) the pending row.
    struct GatedProvider {
        calls: std::sync::Mutex<Vec<String>>,
        open: tokio::sync::watch::Receiver<bool>,
    }

    #[async_trait::async_trait]
    impl crate::Provider for GatedProvider {
        async fn chat_scoped(
            &self,
            request: crate::ChatRequest,
        ) -> Result<crate::ChatResponse, crate::providers::ScopedCallError> {
            self.calls.lock().unwrap_poison().push(
                request
                    .messages
                    .iter()
                    .map(|m| format!("{m:?}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            if request.meta.as_ref().is_some_and(|m| m.role == "assistant") {
                std::future::pending::<()>().await;
            }
            let mut open = self.open.clone();
            while !*open.borrow_and_update() {
                if open.changed().await.is_err() {
                    break;
                }
            }
            Ok(crate::ChatResponse {
                text: Some("ANALYST FINDING".to_string()),
                ..Default::default()
            })
        }
    }

    /// An ANALYST place round: analysts run in the place directory's synthetic workspace
    /// while the job and the delivered envelope stay keyed to the caller's workspace, and
    /// `jobs.exec_dir` records the directory really used so a restart resumes into it.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn place_round_records_exec_dir_and_delivers_to_the_caller() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
        let provider = Arc::new(GatedProvider {
            calls: std::sync::Mutex::new(Vec::new()),
            open: gate_rx,
        });
        let _provider_guard = install_fake_provider(provider.clone());

        let dir = tempfile::tempdir().unwrap();
        // The spelling a resolved place stores: canonical, without the platform's
        // verbatim prefix.
        let place_path =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();
        let caller = test_ws("/tmp/test_ws_analyze_place");

        // The admin Assistant's tool path: a place-bearing call with the caller identity
        // from its task-locals.
        let ack = crate::agent::CURRENT_TOOL_USER_NAME
            .scope("admin".to_string(), async {
                crate::agent::CURRENT_TOOL_CHANNEL
                    .scope("gui".to_string(), async {
                        AnalyzeTool::admin(Role::Assistant)
                            .execute(
                                &caller,
                                json!({ "message": "what is here", "place": place_path }),
                            )
                            .await
                    })
                    .await
            })
            .await
            .expect("dispatch acked");
        let job_id = ack
            .split("job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .expect("the ack names the job id")
            .to_string();

        // Parked on the gated provider: the job row records the place directory while the
        // delivery key stays the caller's.
        let conn = &crate::session::store().conn;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let (workspace_name, exec_dir) = loop {
            let rows = conn
                .query(
                    "SELECT workspace_name, exec_dir FROM jobs WHERE id = ?1",
                    crate::db::params![job_id.clone()],
                )
                .await
                .unwrap();
            if let Some(row) = rows.first() {
                break (
                    row.get::<String>(0).unwrap(),
                    row.get::<Option<String>>(1).unwrap(),
                );
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the parked round's job row never appeared"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        };
        assert_eq!(
            workspace_name, caller.name,
            "the durable delivery key never moves"
        );
        assert_eq!(
            exec_dir.as_deref(),
            Some(place_path.as_str()),
            "jobs.exec_dir records the directory the round really ran in"
        );

        gate_tx.send(true).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let envelope = loop {
            let rows = conn
                .query(
                    "SELECT envelope FROM pending_jobs WHERE id = ?1",
                    crate::db::params![job_id.clone()],
                )
                .await
                .unwrap();
            if let Some(row) = rows.first() {
                break serde_json::from_str::<crate::agent::message_router::AgentJob>(
                    &row.get::<String>(0).unwrap(),
                )
                .unwrap();
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the place round never delivered its envelope"
            );
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        };
        assert_eq!(envelope.workspace_name, caller.name);
        assert!(
            envelope.content.contains("ANALYST FINDING"),
            "{}",
            envelope.content
        );
        assert!(
            envelope
                .content
                .contains(&format!("Directory: {place_path}")),
            "{}",
            envelope.content
        );

        // The analysts really saw the place: its path is in their system prompt and
        // their sessions are keyed to its synthetic workspace, never the caller's.
        let calls = provider.calls.lock().unwrap_poison().join("\n");
        assert!(
            calls.contains(&place_path),
            "the analyst system prompt carries the execution workspace's path"
        );
        let place_name = crate::tools::delegation::workspace_at_dir(&place_path)
            .await
            .unwrap()
            .name;
        assert!(place_name.starts_with("dir-"), "got {place_name}");
        let place_sessions = conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![place_name],
            )
            .await
            .unwrap();
        assert!(
            !place_sessions.is_empty(),
            "the analysts' sessions are keyed to the place's synthetic workspace"
        );
        let caller_sessions = conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![caller.name],
            )
            .await
            .unwrap();
        assert!(
            caller_sessions.is_empty(),
            "nothing ran in the caller's own workspace"
        );
    }

    /// A `place` is inert for every holder but the admin's Assistant: the guest's tool
    /// keeps today's interface (no new error, even for a value the admin's tool refuses)
    /// and still runs in the caller's own workspace.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn a_place_is_inert_for_a_guest_assistant() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider_guard =
            install_fake_provider(Arc::new(FakeProvider::new().ok("ANALYST FINDING")));
        let caller = test_ws("/tmp/test_ws_guest_inert_place");
        let dir = tempfile::tempdir().unwrap();
        let place_path =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();

        let tool = AnalyzeTool::new(DispatchMode::Async, Role::Assistant);
        // A value the admin's tool refuses must not become a new error for the guest.
        for place in ["not_a_managed_workspace", place_path.as_str()] {
            let ack = tool
                .execute(&caller, json!({ "message": "q", "place": place }))
                .await
                .expect("an inert place must not surface a new error");
            assert!(ack.contains("dispatched"), "{ack}");
        }

        // Both rounds run in the caller's own workspace — never the place.
        let conn = &crate::session::store().conn;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let rows = conn
                .query(
                    "SELECT COUNT(*) FROM session_metadata WHERE workspace_name = ?1",
                    crate::db::params![caller.name.clone()],
                )
                .await
                .unwrap();
            if rows[0].get::<i64>(0).unwrap() > 0 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the guest's round never ran in the caller's own workspace"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let place_name = crate::tools::delegation::workspace_at_dir(&place_path)
            .await
            .unwrap()
            .name;
        let place_sessions = conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![place_name],
            )
            .await
            .unwrap();
        assert!(
            place_sessions.is_empty(),
            "the inert place directory never ran anything"
        );
    }

    /// Dispatch a tool as the agent loop does, with the caller identity it reads from
    /// its task-locals.
    async fn dispatch_analyze(
        tool: &AnalyzeTool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        crate::agent::CURRENT_TOOL_USER_NAME
            .scope("admin".to_string(), async {
                crate::agent::CURRENT_TOOL_CHANNEL
                    .scope("gui".to_string(), async { tool.execute(ws, args).await })
                    .await
            })
            .await
    }

    /// The `(kind, exec_dir)` a durable round's job row records, within a bounded wait —
    /// the spawned dispatch writes it asynchronously.
    async fn round_row(job_id: &str) -> (String, Option<String>) {
        let conn = &crate::session::store().conn;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let rows = conn
                .query(
                    "SELECT kind, exec_dir FROM jobs WHERE id = ?1",
                    crate::db::params![job_id],
                )
                .await
                .unwrap();
            if let Some(row) = rows.first() {
                return (
                    row.get::<String>(0).unwrap(),
                    row.get::<Option<String>>(1).unwrap(),
                );
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no job row for {job_id} — the round was never created"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    /// A folder name may never refuse a `deep` call: `place` is completely inert for
    /// `deep` — never a refusal, never a different execution root — and a deep-only holder
    /// is not offered `place` at all, so the round keeps the caller's workspace.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)] // serializes the process-global fake provider + channel registry + shutdown drain flag
    async fn a_place_never_touches_a_deep_round() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider_guard =
            install_fake_provider(Arc::new(FakeProvider::new().ok("ANALYST FINDING")));
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let home = tempfile::tempdir().unwrap();
        let caller = crate::workspace::test_ws_named(
            home.path().to_str().unwrap(),
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        );
        // Register the deep report's consumer exactly as the running service's bootstrap
        // leaves it.
        let assistant = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let _assistant_rx = crate::agent::message_router::register_agent(&assistant);

        // A `place` the `quick` depth refuses: not a directory at all.
        let bogus = "/definitely/not/a/directory/xyz";
        let admin = AnalyzeTool::admin(Role::Assistant);
        let err = dispatch_analyze(
            &admin,
            &caller,
            json!({ "message": "quick q", "depth": "quick", "place": bogus }),
        )
        .await
        .expect_err("a quick round refuses a place that is not a directory");
        assert!(err.to_string().contains("does not exist"), "{err}");

        // Completely inert for `deep`: the round is dispatched, never refused.
        let ack = dispatch_analyze(
            &admin,
            &caller,
            json!({ "message": "deep q", "depth": "deep", "place": bogus }),
        )
        .await
        .expect("a deep round is never refused for a place");
        assert!(ack.contains("Deep analysis dispatched"), "{ack}");
        let job_id = ack
            .split("job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .expect("the deep ack names its job")
            .to_string();
        let (kind, exec_dir) = round_row(&job_id).await;
        assert_eq!(kind, "research", "a deep round is a research job");
        assert_eq!(
            exec_dir, None,
            "a deep round ignores the place: no exec_dir is recorded"
        );
        // Completion is never required: the round is stopped the way the Running Agents
        // view stops one, so it never outlives the test.
        crate::research_cancel::cancel_research_run(&job_id).await;

        // A deep-only granted holder is not offered `place`; passing one still succeeds
        // and the round keeps the caller's workspace.
        let deep_only = AnalyzeTool::granted(&[AnalyzeDepth::Deep]);
        assert!(
            deep_only.parameters_schema()["properties"]
                .get("place")
                .is_none(),
            "a deep-only holder never declares a place"
        );
        let ack = dispatch_analyze(
            &deep_only,
            &caller,
            json!({ "message": "deep one", "place": bogus }),
        )
        .await
        .expect("a deep-only holder is not tripped by a place it never declares");
        assert!(ack.contains("Deep analysis dispatched"), "{ack}");
        let job_id = ack
            .split("job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .expect("the deep ack names its job")
            .to_string();
        let (kind, exec_dir) = round_row(&job_id).await;
        assert_eq!(kind, "research");
        assert_eq!(
            exec_dir, None,
            "the round runs in the caller's own workspace, not the named folder"
        );
        crate::research_cancel::cancel_research_run(&job_id).await;
    }

    // The cancel tests' caller: a guest Assistant's async quick variant under its own
    // user/workspace identity. The router and test store are process-global, so this is
    // an identity no other test routes to — a restart's replay cannot leak another test's
    // message into this receiver.
    const CANCEL_TEST_USER: &str = "cancel_test_user";
    const CANCEL_TEST_WS: &str = "cancel_test_ws";

    // The sibling-round identity: its delivered result stays in the shared store's
    // `pending_jobs`, so no other cancel test's replay assertions may observe it.
    const CANCEL_SIBLING_USER: &str = "cancel_sibling_user";
    const CANCEL_SIBLING_WS: &str = "cancel_sibling_ws";

    /// The agent address of [`CANCEL_SIBLING_USER`]'s Assistant session.
    fn cancel_sibling_caller_id() -> String {
        crate::session::resolve_agent_id(
            CANCEL_SIBLING_USER,
            Role::Assistant.as_str(),
            CANCEL_SIBLING_WS,
        )
    }

    /// [`CANCEL_TEST_USER`]'s Assistant agent address — registered as a consumer, and the
    /// one a round's `jobs` row derives to.
    fn cancel_test_caller_id() -> String {
        crate::session::resolve_agent_id(CANCEL_TEST_USER, Role::Assistant.as_str(), CANCEL_TEST_WS)
    }

    /// Whether the store holds a row with this `id` in `table` — the cancel
    /// tests' window into the durable state a round leaves behind.
    async fn cancel_row_exists(table: &str, id: &str) -> bool {
        !crate::session::store()
            .conn
            .query(
                &format!("SELECT id FROM {table} WHERE id = ?1"),
                crate::db::params![id.to_string()],
            )
            .await
            .unwrap()
            .is_empty()
    }

    /// Run the cancel tests' tool as the agent loop does, including the calling agent id
    /// the cancel action proves ownership with. The caller's workspace is the one the
    /// round's address derives from, so the dispatching session is the cancelling one.
    async fn run_as_caller(
        tool: &AnalyzeTool,
        ws: &Workspace,
        user: &str,
        caller_id: &str,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        crate::agent::CURRENT_TOOL_USER_NAME
            .scope(user.to_string(), async {
                crate::agent::CURRENT_TOOL_CHANNEL
                    .scope("gui".to_string(), async {
                        crate::agent::CURRENT_TOOL_AGENT_ID
                            .scope(Some(caller_id.to_string()), async {
                                tool.execute(ws, args).await
                            })
                            .await
                    })
                    .await
            })
            .await
    }

    /// [`run_as_caller`] for [`CANCEL_TEST_USER`]'s own session.
    async fn run_as_cancel_test_caller(
        tool: &AnalyzeTool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        run_as_caller(tool, ws, CANCEL_TEST_USER, &cancel_test_caller_id(), args).await
    }

    /// Records the level of every event emitted on the thread it is installed
    /// for, so a test can assert a path left no complaint behind.
    #[derive(Clone, Default)]
    struct LevelRecorder {
        levels: std::sync::Arc<std::sync::Mutex<Vec<tracing::Level>>>,
    }

    impl LevelRecorder {
        fn levels(&self) -> Vec<tracing::Level> {
            self.levels.lock().unwrap_poison().clone()
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for LevelRecorder {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.levels
                .lock()
                .unwrap_poison()
                .push(*event.metadata().level());
        }
    }

    /// A cancel stops a parked quick round: its `jobs` row and queued result go, the fired signal
    /// stays observable while the round unwinds, nothing is delivered, the analysts' session
    /// records survive, and a start does not re-arm the round.
    ///
    /// `current_thread` on purpose: the WARN/ERROR capture is thread-scoped.
    #[tokio::test(flavor = "current_thread")]
    #[serial_test::serial(provider, drain, recover_from_restart)]
    async fn cancel_stops_a_running_round_and_delivers_nothing() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
        let _provider_guard = install_fake_provider(Arc::new(GatedProvider {
            calls: std::sync::Mutex::new(Vec::new()),
            open: gate_rx,
        }));
        let caller_id = cancel_test_caller_id();
        let mut rx = crate::agent::message_router::register_agent(&caller_id);
        let caller = crate::workspace::test_ws_named("/tmp/test_ws_analyze_cancel", CANCEL_TEST_WS);
        let tool = AnalyzeTool::new(DispatchMode::Async, Role::Assistant);

        let ack = run_as_cancel_test_caller(&tool, &caller, json!({ "message": "what changed" }))
            .await
            .expect("dispatch acked");
        let job_id = ack
            .split("job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .expect("the ack names the job id")
            .to_string();
        assert!(ack.contains(&job_id), "{ack}");

        // Parked on the gated provider: the jobs row and analyst roster are written while
        // nothing has been delivered.
        let (kind, _exec_dir) = round_row(&job_id).await;
        assert_eq!(kind, "analyze", "a quick round is an analyze job");
        let conn = &crate::session::store().conn;
        let roster: Vec<String> = conn
            .query(
                "SELECT agent_id FROM agents WHERE job_id = ?1",
                crate::db::params![job_id.clone()],
            )
            .await
            .unwrap()
            .iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert!(!roster.is_empty(), "the round has an analyst roster");

        // A synthetic call card belonging to the round, as its consolidation call registers
        // one: the cancel must remove it so the round stops showing as running work.
        let _call_card = crate::agent::registry::NON_AGENT_CALLS.register(
            "consolidate",
            CANCEL_TEST_WS,
            Some(crate::agent::registry::ParentKey::AnalyzeRound(
                job_id.clone(),
            )),
            false,
            None,
        );

        let recorder = LevelRecorder::default();
        let answer = {
            use tracing_subscriber::layer::SubscriberExt as _;
            let _capture = tracing::subscriber::set_default(
                tracing_subscriber::registry().with(recorder.clone()),
            );
            run_as_cancel_test_caller(
                &tool,
                &caller,
                json!({ "action": "cancel", "job_id": job_id }),
            )
            .await
            .expect("the cancel is answered")
        };
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");
        assert!(answer.contains(&job_id), "{answer}");
        // The cancel leaves no complaint behind: nothing it emits is a WARN or ERROR (its
        // own record is the INFO line plus the tool call in the caller's session).
        // `tracing::Level` orders by VERBOSITY, so "above WARN" means quieter than a
        // warning. The capture must have seen the path's own INFO record — an empty capture
        // could only make the assertion pass vacuously.
        assert!(
            recorder.levels().contains(&tracing::Level::INFO),
            "the capture must observe the cancel's own record: {:?}",
            recorder.levels()
        );
        assert!(
            recorder
                .levels()
                .iter()
                .all(|level| *level > tracing::Level::WARN),
            "the cancel must leave no complaint behind: {:?}",
            recorder.levels()
        );

        // The stop is durable while the round is still parked: the rows are gone and the
        // fired signal is observable on the live entry.
        assert!(
            !cancel_row_exists("jobs", &job_id).await,
            "the cancelled round's jobs row must be gone"
        );
        assert!(
            !cancel_row_exists("pending_jobs", &job_id).await,
            "the cancelled round must leave no queued result"
        );
        assert!(
            crate::round_cancel::is_cancelled(&job_id),
            "the fired signal must be observable while the round unwinds"
        );
        assert!(
            !crate::agent::registry::NON_AGENT_CALLS
                .list()
                .iter()
                .any(|card| card.parent_key
                    == Some(crate::agent::registry::ParentKey::AnalyzeRound(
                        job_id.clone()
                    ))),
            "the cancelled round must stop being shown as running work"
        );

        // The round is still running (its entry is registered until its invocation exits),
        // so a second cancel of the same id is answered as stopped again.
        let answer = run_as_cancel_test_caller(
            &tool,
            &caller,
            json!({ "action": "cancel", "job_id": job_id }),
        )
        .await
        .expect("the repeat cancel is answered");
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");

        // Let the parked round unwind: nothing may resurrect its rows or reach the caller.
        gate_tx.send(true).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while crate::round_cancel::is_registered(&job_id) {
            assert!(
                std::time::Instant::now() < deadline,
                "the cancelled round never unwound after its gate opened"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            !cancel_row_exists("jobs", &job_id).await,
            "the unwound round must not resurrect its jobs row"
        );
        assert!(
            !cancel_row_exists("pending_jobs", &job_id).await,
            "the unwound round must not deliver a result"
        );
        assert!(
            rx.try_recv().is_err(),
            "a cancelled round must deliver nothing to the caller"
        );

        // The analysts' own records are not the round's to remove.
        for agent_id in &roster {
            assert_eq!(
                conn.query(
                    "SELECT agent_id FROM session_metadata WHERE agent_id = ?1",
                    crate::db::params![agent_id.clone()],
                )
                .await
                .unwrap()
                .len(),
                1,
                "the analyst {agent_id}'s session record must survive the cancel"
            );
        }

        // A start replays pending_jobs and re-arms resumable jobs: this round is neither.
        let resumable = crate::jobs::recover_from_restart().await.unwrap();
        assert!(
            !resumable.iter().any(|job| matches!(
                job,
                crate::jobs::ResumableJob::Analyze { job_id: id, .. } if id == &job_id
            )),
            "a cancelled round must not be re-armed by a start"
        );
        assert!(
            rx.try_recv().is_err(),
            "the pending replay must deliver nothing for the cancelled round"
        );
    }

    /// A cancel discards an already-produced result before delivery: its queued row goes,
    /// nothing reaches the caller, the discard survives a start's replay, and a second
    /// cancel answers exactly like an id that never existed.
    #[tokio::test]
    #[serial_test::serial(provider, drain, recover_from_restart)]
    async fn cancel_discards_a_produced_result_and_survives_a_restart() {
        crate::util::test::init_management_test_stores().await;
        let caller_id = cancel_test_caller_id();
        let mut rx = crate::agent::message_router::register_agent(&caller_id);
        let caller =
            crate::workspace::test_ws_named("/tmp/test_ws_analyze_cancel_discard", CANCEL_TEST_WS);
        let tool = AnalyzeTool::new(DispatchMode::Async, Role::Assistant);
        let job_id = "analyze_cancel_discard_job";

        // A produced result with no round row left: the delivery is all that remains.
        crate::jobs::complete_durable_job(
            job_id,
            format!("<analyze-tool-result>\n\nJob: {job_id}\n\nfindings"),
            crate::agent::message_router::MessageKind::AnalyzeToolResult,
            crate::Role::Assistant,
            CANCEL_TEST_USER,
            "gui",
            &caller.name,
            None,
        )
        .await;
        assert!(
            cancel_row_exists("pending_jobs", job_id).await,
            "the produced result waits in pending_jobs"
        );

        let answer = run_as_cancel_test_caller(
            &tool,
            &caller,
            json!({ "action": "cancel", "job_id": job_id }),
        )
        .await
        .expect("the cancel is answered");
        assert!(
            answer.contains("had already produced its result"),
            "{answer}"
        );
        assert!(answer.contains("discarded"), "{answer}");
        assert!(answer.contains(job_id), "{answer}");
        assert!(
            !cancel_row_exists("pending_jobs", job_id).await,
            "the discarded result's queued row is gone"
        );
        assert!(
            rx.try_recv().is_err(),
            "a discarded result must not be delivered"
        );

        // A start replays pending_jobs — the discarded row must not come back.
        crate::jobs::recover_from_restart().await.unwrap();
        assert!(
            !cancel_row_exists("pending_jobs", job_id).await,
            "the discard must survive a restart's pending replay"
        );
        assert!(
            rx.try_recv().is_err(),
            "the pending replay of a discarded result delivers nothing"
        );

        // Gone for good: a second cancel answers like an id that never existed.
        let answer = run_as_cancel_test_caller(
            &tool,
            &caller,
            json!({ "action": "cancel", "job_id": job_id }),
        )
        .await
        .expect("the second cancel is answered");
        assert!(
            answer.contains("No analysis round of this session to cancel"),
            "{answer}"
        );
    }

    /// A cancelled round leaves no resumable row: a dispatch that observes the cancel at its
    /// entry returns before writing anything, delivers nothing and releases the synthetic
    /// place's engine. The write inside the dispatch's own row transaction is the gate
    /// `jobs::spawn_job` covers, pinned in `spawn_job_refuses_a_cancelled_rounds_row`.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn a_cancelled_dispatch_leaves_no_resumable_row() {
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_analyze_cancel_dispatch");
        let job_id = "analyze_cancel_dispatch_job";

        // The cancelled dispatch must release the picker/watcher it created for the
        // synthetic `dir-…` place.
        let place_dir = std::env::temp_dir().join("mahbot_analyze_cancel_place");
        std::fs::create_dir_all(&place_dir).unwrap();
        let place =
            crate::workspace::test_ws_named(place_dir.to_str().unwrap(), "dir-cancel-place-test");
        crate::workspace_index::get_or_init_engine(&place.name, std::path::Path::new(&place.path))
            .unwrap();
        assert!(
            crate::workspace_index::get_engine_by_name(&place.name).is_some(),
            "the place's search engine must be registered before the dispatch"
        );

        let _cancel_guard = crate::round_cancel::register(job_id, None);
        crate::round_cancel::cancel(job_id);

        let delivered = crate::tools::SyncDurableCore::Analyze
            .dispatch_durable(
                &ws,
                Some(&place),
                "task",
                Role::Assistant,
                "user",
                "gui",
                job_id,
                None,
            )
            .await;
        assert!(delivered.is_none(), "a cancelled round delivers nothing");
        assert!(
            !cancel_row_exists("jobs", job_id).await,
            "the cancelled dispatch must write no row for a start to resume"
        );
        assert!(
            crate::workspace_index::get_engine_by_name(&place.name).is_none(),
            "a cancelled round must not leave its picker/watcher held"
        );
    }

    /// A cancel landing between the boot scan and the resume's cancel-signal registration
    /// still stops the round: the mark reaches the registration, so the resume runs nothing
    /// and a caller answered as stopped is never delivered to.
    #[tokio::test]
    #[serial_test::serial(drain, recover_from_restart)] // the resume consults the drain flag
    async fn a_cancel_before_a_resume_registers_still_stops_that_resume() {
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_analyze_cancel_before_resume");
        let job_id = "analyze_cancel_before_resume_job";
        let agent_id = format!("analyze_{}_rs_0_analyst", ws.name);
        let conn = &crate::session::store().conn;
        // A round a crashed dispatch left launched, with its stored outcome: without the
        // cancel, this resume runs and delivers it.
        crate::jobs::spawn_job(
            conn,
            job_id,
            "question?",
            &ws.name,
            "caller-user",
            "telegram",
            Role::Assistant,
            &[crate::jobs::NewAgent {
                agent_id: agent_id.clone(),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(0),
                task: "question?".to_string(),
            }],
            &crate::jobs::SpawnChild::Analyze,
            None,
            None,
            None,
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

        // The cancel fires while nothing is registered and its row removal has not landed:
        // the boot scan already listed this row, so the resume must observe the cancel.
        assert!(!crate::round_cancel::is_registered(job_id));
        crate::round_cancel::cancel(job_id);
        resume_analyze_round(job_id, &ws, None).await;

        // Asserted BEFORE the cancel action's own row removal below: an
        // uncancelled resume of this stored round writes exactly this row.
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(
            pending.len(),
            0,
            "a round cancelled before its resume must deliver nothing"
        );

        // The action then removes the round's own rows, so the next start cannot re-arm it.
        crate::jobs::delete_round_rows(conn, job_id).await.unwrap();
        assert!(
            !cancel_row_exists("jobs", job_id).await,
            "the cancelled round must not be re-armed for the next start"
        );
    }

    /// Seed one launched round row exactly as a dispatch writes it — the address shapes
    /// the ownership cases need.
    async fn seed_round_row(
        ws: &Workspace,
        job_id: &str,
        user: &str,
        role: Role,
        launch: Option<&crate::Launch>,
    ) {
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            job_id,
            "question?",
            &ws.name,
            user,
            "gui",
            role,
            &[],
            &crate::jobs::SpawnChild::Analyze,
            None,
            None,
            launch,
        )
        .await
        .unwrap();
    }

    /// Ownership is the calling SESSION, via the crate's ONE delivery-address derivation
    /// ([`crate::jobs::target_agent_id`]), covering both addresses that are NOT
    /// (user, role, workspace): the workspace's shared Manager session, cancelling in a pipeline
    /// turn with NO user, and a custom-role branch, whose address comes from its row's launch.
    #[tokio::test]
    #[serial_test::serial(drain, recover_from_restart)]
    async fn the_dispatchers_session_cancels_its_round_whatever_its_address_shape() {
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_analyze_cancel_addresses");

        // A Manager round cancelled by the workspace's manager session, in a turn with no
        // user (pipeline ticket notifications).
        let manager_job = "analyze_cancel_manager_job";
        seed_round_row(&ws, manager_job, "asking-user", Role::Manager, None).await;
        let manager_tool = AnalyzeTool::new(DispatchMode::Async, Role::Manager);
        let answer = run_as_caller(
            &manager_tool,
            &ws,
            "",
            &crate::session::manager_agent_id(&ws.name),
            json!({ "action": "cancel", "job_id": manager_job }),
        )
        .await
        .expect("the cancel is answered");
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");
        assert!(
            !cancel_row_exists("jobs", manager_job).await,
            "the manager's own round is stopped"
        );

        // A custom-role branch's round, cancelled by that branch.
        let branch_job = "analyze_cancel_custom_job";
        let branch = "role_audit:1";
        let launch = crate::Launch {
            role: Role::Custom,
            name: "audit".to_string(),
            identifier: Some("1".to_string()),
            prompt: String::new(),
            tools: Vec::new(),
            agent_id: branch.to_string(),
            reply_to: None,
        };
        seed_round_row(&ws, branch_job, "branch-user", Role::Custom, Some(&launch)).await;
        let answer = run_as_caller(
            &AnalyzeTool::new(DispatchMode::Async, Role::Assistant),
            &ws,
            "branch-user",
            branch,
            json!({ "action": "cancel", "job_id": branch_job }),
        )
        .await
        .expect("the cancel is answered");
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");
        assert!(
            !cancel_row_exists("jobs", branch_job).await,
            "the branch's own round is stopped"
        );
    }

    /// A placed round holds its synthetic `dir-…` search engine for a resume that reuses it.
    /// A cancel means no resume, so the cancel releases it from the directory its row
    /// records — even once that directory is gone, since the engine is NAMED by it.
    #[tokio::test]
    #[serial_test::serial(drain, recover_from_restart)]
    async fn a_cancel_releases_a_placed_rounds_workspace_index() {
        crate::util::test::init_management_test_stores().await;
        let caller = crate::workspace::test_ws_named(
            "/tmp/test_ws_analyze_cancel_place_release",
            CANCEL_TEST_WS,
        );
        let dir = tempfile::tempdir().unwrap();
        let place_path =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();
        // The engine a drain-cut round leaves behind.
        let place = crate::tools::delegation::workspace_at_dir(&place_path)
            .await
            .unwrap();
        crate::workspace_index::get_or_init_engine(&place.name, std::path::Path::new(&place.path))
            .unwrap();
        assert!(
            crate::workspace_index::get_engine_by_name(&place.name).is_some(),
            "the placed round's engine must be registered before the cancel"
        );
        let job_id = "analyze_cancel_place_release_job";
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            job_id,
            "analyze task",
            CANCEL_TEST_WS,
            CANCEL_TEST_USER,
            "gui",
            Role::Assistant,
            &[],
            &crate::jobs::SpawnChild::Analyze,
            None,
            Some(&place_path),
            None,
        )
        .await
        .unwrap();
        // The directory is gone by now — the state a drain-cut round leaves: its rows are
        // swept and the engine's name derives from the recorded path, which need not exist.
        dir.close().unwrap();
        assert!(
            !std::path::Path::new(&place_path).exists(),
            "the fixture's directory must be gone before the cancel"
        );

        let answer = run_as_cancel_test_caller(
            &AnalyzeTool::new(DispatchMode::Async, Role::Assistant),
            &caller,
            json!({ "action": "cancel", "job_id": job_id }),
        )
        .await
        .expect("the cancel is answered");
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");
        assert!(
            crate::workspace_index::get_engine_by_name(&place.name).is_none(),
            "a cancelled round must not leave its picker/watcher held for a resume that cannot happen"
        );
        assert!(!cancel_row_exists("jobs", job_id).await);
    }

    /// A cancel landing AFTER a phase's entry gate stops that phase's own model calls
    /// instead of letting the retry loop re-ask: both extraction and consolidation are
    /// abandoned at the fired signal.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn a_cancelled_round_asks_no_further_model_call() {
        // Parks the canceller at the phase's in-flight call instead of sleeping blindly:
        // bounded wait until the provider has recorded `expected` calls.
        async fn wait_for_calls(provider: &GatedProvider, expected: usize) {
            let mut waited = 0;
            while provider.calls.lock().unwrap_poison().len() < expected {
                assert!(waited < 400, "the provider never recorded {expected} calls");
                waited += 1;
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        }

        /// The phase is already over: a retry loop ignoring the signal would re-ask right
        /// away, so watching the count across a settle window catches it anywhere.
        async fn assert_no_further_calls(provider: &GatedProvider, expected: usize) {
            for _ in 0..20 {
                assert_eq!(
                    provider.calls.lock().unwrap_poison().len(),
                    expected,
                    "no further attempt may be asked after the cancel"
                );
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        }

        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Every call parks until the gate opens, then answers with unparseable text: were
        // the phase not abandoned at the cancel, the retry loop would re-ask.
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
        let provider = Arc::new(GatedProvider {
            calls: std::sync::Mutex::new(Vec::new()),
            open: gate_rx,
        });
        let _provider_guard = install_fake_provider(provider.clone() as Arc<dyn crate::Provider>);
        let ws = test_ws("/tmp/test_ws_analyze_cancel_no_further_call");
        let round = "analyze_cancel_no_further_call";
        let _cancel_guard = crate::round_cancel::register(round, None);

        // Part 1 — extraction: both calls are in flight when the cancel lands, so neither
        // may re-ask once the gate opens.
        let runs = vec![
            test_analyst_run(&ws, Some("report one")),
            test_analyst_run(&ws, Some("report two")),
        ];
        let (result, ()) = tokio::join!(
            consolidate_analyst_runs(
                &ws,
                "test question",
                runs,
                std::time::Instant::now() + Duration::from_mins(1),
                round,
            ),
            async {
                wait_for_calls(&provider, 2).await;
                crate::round_cancel::cancel(round);
                // Unblock the parked calls too: a retry loop that ignored the signal would
                // re-ask and the count below would grow.
                gate_tx.send(true).unwrap();
            },
        );
        assert!(
            result
                .expect("a cancelled round is not a failure")
                .is_none(),
            "a cancelled round produces no text"
        );
        assert_no_further_calls(&provider, 2).await;

        // Part 2 — consolidation: the grouping call is in flight when the cancel lands, so
        // no repair round may start. A fresh round key is registered (the entry gate would
        // catch the already-fired signal) and the gate re-armed so grouping parks.
        let round2 = "analyze_cancel_no_further_call_consolidate";
        let _cancel_guard2 = crate::round_cancel::register(round2, None);
        gate_tx.send(false).unwrap();
        let (result, ()) = tokio::join!(
            consolidate_findings(
                &ws,
                "test question",
                agreed_outcomes("report one", "report two"),
                round2,
                std::time::Instant::now() + Duration::from_mins(1),
            ),
            async {
                wait_for_calls(&provider, 3).await;
                crate::round_cancel::cancel(round2);
                // As in Part 1: the phase must stop at the signal, not after the
                // unblocked call returns.
                gate_tx.send(true).unwrap();
            },
        );
        assert!(
            result
                .expect("a cancelled round is not a failure")
                .is_none(),
            "a cancelled round starts no grouping call"
        );
        assert_no_further_calls(&provider, 3).await;
    }

    /// Cancelling one round leaves the caller's other in-flight rounds alone: a second round
    /// of the same session keeps its produced result, its signal is never fired, and a
    /// start's replay still delivers it.
    #[tokio::test]
    #[serial_test::serial(provider, drain, recover_from_restart)]
    async fn a_cancel_leaves_the_callers_other_rounds_delivering() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let caller_id = cancel_sibling_caller_id();
        let mut rx = crate::agent::message_router::register_agent(&caller_id);
        let caller = crate::workspace::test_ws_named(
            "/tmp/test_ws_analyze_cancel_siblings",
            CANCEL_SIBLING_WS,
        );
        let tool = AnalyzeTool::new(DispatchMode::Async, Role::Assistant);

        // The caller's other round, already produced and waiting.
        let sibling = "analyze_cancel_sibling_job";
        crate::jobs::complete_durable_job(
            sibling,
            format!("<analyze-tool-result>\n\nJob: {sibling}\n\nsibling findings"),
            crate::agent::message_router::MessageKind::AnalyzeToolResult,
            crate::Role::Assistant,
            CANCEL_SIBLING_USER,
            "gui",
            &caller.name,
            None,
        )
        .await;

        // A live round of the same session, cancelled by the action.
        crate::jobs::spawn_job(
            conn,
            "analyze_cancel_first_job",
            "first question",
            &caller.name,
            CANCEL_SIBLING_USER,
            "gui",
            crate::Role::Assistant,
            &[],
            &crate::jobs::SpawnChild::Analyze,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let answer = run_as_caller(
            &tool,
            &caller,
            CANCEL_SIBLING_USER,
            &caller_id,
            json!({ "action": "cancel", "job_id": "analyze_cancel_first_job" }),
        )
        .await
        .expect("the cancel is answered");
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");

        // The sibling round is untouched: its result still waits and its signal never fired.
        assert!(
            !crate::round_cancel::is_cancelled(sibling),
            "cancelling one round must not fire another round's signal"
        );
        assert!(
            cancel_row_exists("pending_jobs", sibling).await,
            "the sibling round's queued result must survive"
        );

        // …and it still reaches the caller on the start replay.
        crate::jobs::recover_from_restart().await.unwrap();
        let mut delivered = Vec::new();
        while let Ok(job) = rx.try_recv() {
            delivered.push(job.content);
        }
        assert!(
            delivered.iter().any(|content| content.contains(sibling)),
            "the sibling round's result is still delivered: {delivered:?}"
        );
        // Clear the delivered row so it cannot reach another test's assertions.
        crate::jobs::delete_pending_job(conn, sibling)
            .await
            .unwrap();
    }

    /// The action refuses everything that is not the caller's own live quick round — an unknown
    /// id, a `deep` investigation the caller owns, a deep investigation's queued report, a worker
    /// round's queued report, another session's round — and leaves each alone. The caller's own
    /// quick row is stopped and deleted, after which the same id answers as nothing of its own.
    #[tokio::test]
    #[serial_test::serial(provider, drain, recover_from_restart)]
    async fn a_cancel_refuses_everything_that_is_not_the_callers_live_quick_round() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let caller_id = cancel_sibling_caller_id();
        let _rx = crate::agent::message_router::register_agent(&caller_id);
        let caller = crate::workspace::test_ws_named(
            "/tmp/test_ws_analyze_cancel_refuse",
            CANCEL_SIBLING_WS,
        );
        let tool = AnalyzeTool::new(DispatchMode::Async, Role::Assistant);

        // An id that names nothing is answered without touching anything.
        let answer = run_as_caller(
            &tool,
            &caller,
            CANCEL_SIBLING_USER,
            &caller_id,
            json!({ "action": "cancel", "job_id": "no_such_cancel_round" }),
        )
        .await
        .expect("an unknown id is answered");
        assert!(
            answer.contains("No analysis round of this session to cancel"),
            "{answer}"
        );
        assert!(
            !crate::round_cancel::is_cancelled("no_such_cancel_round"),
            "a refused cancel must not fire a signal"
        );

        // A `deep` investigation is a research-kind row — out of scope even when the
        // caller owns it.
        crate::jobs::spawn_job(
            conn,
            "deep_cancel_round",
            "deep question",
            &caller.name,
            CANCEL_SIBLING_USER,
            "gui",
            crate::Role::Assistant,
            &[],
            &crate::jobs::SpawnChild::Research,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let answer = run_as_caller(
            &tool,
            &caller,
            CANCEL_SIBLING_USER,
            &caller_id,
            json!({ "action": "cancel", "job_id": "deep_cancel_round" }),
        )
        .await
        .expect("a deep round is answered");
        assert!(
            answer.contains("No analysis round of this session to cancel"),
            "{answer}"
        );
        assert!(
            !crate::round_cancel::is_cancelled("deep_cancel_round"),
            "a refused cancel must not fire a signal"
        );
        assert!(
            cancel_row_exists("jobs", "deep_cancel_round").await,
            "a refused deep round is left in place"
        );

        // Another session's analyze round is refused and untouched.
        crate::jobs::spawn_job(
            conn,
            "someone_elses_analyze",
            "their question",
            "mahbot",
            "someone-else",
            "gui",
            crate::Role::Engineer,
            &[],
            &crate::jobs::SpawnChild::Analyze,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let answer = run_as_caller(
            &tool,
            &caller,
            CANCEL_SIBLING_USER,
            &caller_id,
            json!({ "action": "cancel", "job_id": "someone_elses_analyze" }),
        )
        .await
        .expect("another session's round is answered");
        assert!(
            answer.contains("No analysis round of this session to cancel"),
            "{answer}"
        );
        assert!(
            !crate::round_cancel::is_cancelled("someone_elses_analyze"),
            "a refused cancel must not fire a signal"
        );
        assert!(
            cancel_row_exists("jobs", "someone_elses_analyze").await,
            "another session's round is left in place"
        );

        // Reports queued for THIS caller: only the round's kind tells them from a quick round's
        // result, and neither a deep report nor a worker report is this action's to destroy.
        for (job_id, kind) in [
            (
                "cancel_refuses_deep_report",
                crate::agent::message_router::MessageKind::ResearchResult,
            ),
            (
                "cancel_refuses_worker_report",
                crate::agent::message_router::MessageKind::ImplementResult,
            ),
        ] {
            crate::jobs::complete_durable_job(
                job_id,
                format!("<report>\n\nJob: {job_id}\n\nreport"),
                kind,
                crate::Role::Assistant,
                CANCEL_SIBLING_USER,
                "gui",
                &caller.name,
                None,
            )
            .await;
            let answer = run_as_caller(
                &tool,
                &caller,
                CANCEL_SIBLING_USER,
                &caller_id,
                json!({ "action": "cancel", "job_id": job_id }),
            )
            .await
            .expect("other queued work is answered");
            assert!(
                answer.contains("No analysis round of this session to cancel"),
                "{answer}"
            );
            assert!(
                !crate::round_cancel::is_cancelled(job_id),
                "a refused cancel must not fire a signal"
            );
            assert!(
                cancel_row_exists("pending_jobs", job_id).await,
                "{kind:?} queued for the caller must survive"
            );
        }

        // The caller's own live quick round: a live round holds its cancel-signal entry for
        // the duration of its invocation, so register one as a dispatch would. The first
        // cancel stops it, fires its signal and deletes the row; once the invocation exits
        // (the guard drops) the same id answers like one that never existed.
        crate::jobs::spawn_job(
            conn,
            "mine_cancel_round",
            "my question",
            &caller.name,
            CANCEL_SIBLING_USER,
            "gui",
            crate::Role::Assistant,
            &[],
            &crate::jobs::SpawnChild::Analyze,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let mine_round = crate::round_cancel::register("mine_cancel_round", Some(&caller_id));
        let answer = run_as_caller(
            &tool,
            &caller,
            CANCEL_SIBLING_USER,
            &caller_id,
            json!({ "action": "cancel", "job_id": "mine_cancel_round" }),
        )
        .await
        .expect("the caller's own round is answered");
        assert!(answer.contains("Cancelled the analysis round"), "{answer}");
        assert!(
            !cancel_row_exists("jobs", "mine_cancel_round").await,
            "the caller's own cancelled round is deleted"
        );
        assert!(
            crate::round_cancel::is_cancelled("mine_cancel_round"),
            "the caller's own live round has its signal fired"
        );
        drop(mine_round);

        let answer = run_as_caller(
            &tool,
            &caller,
            CANCEL_SIBLING_USER,
            &caller_id,
            json!({ "action": "cancel", "job_id": "mine_cancel_round" }),
        )
        .await
        .expect("the second cancel is answered");
        assert!(
            answer.contains("No analysis round of this session to cancel"),
            "{answer}"
        );

        // Leave nothing for a sibling test's `recover_from_restart` replay: every queued
        // report and launched row this test created goes.
        for report in ["cancel_refuses_deep_report", "cancel_refuses_worker_report"] {
            crate::jobs::delete_pending_job(conn, report).await.unwrap();
        }
        for launched in ["deep_cancel_round", "someone_elses_analyze"] {
            crate::jobs::terminalize_job(conn, launched).await.unwrap();
        }
    }

    /// Nothing new starts for a cancelled round: an analyst registering AFTER the signal
    /// fired is pre-cancelled at registration. Reachable because a round's verification
    /// analysts are dispatched after the ones it started with.
    #[tokio::test]
    async fn a_late_registering_analyst_of_a_cancelled_round_is_pre_cancelled() {
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_analyze_cancel_late_register");
        let job_id = "analyze_late_register_round";
        let round = crate::round_cancel::register(job_id, None);
        let analyst = |id: &str| {
            crate::Agent::new(
                id.to_string(),
                Role::Analyst,
                &ws,
                None,
                String::new(),
                String::new(),
                false,
                Some(crate::agent::registry::ParentKey::AnalyzeRound(
                    job_id.to_string(),
                )),
                Some("round question".to_string()),
                None,
            )
        };

        assert!(
            !analyst("analyze_late_register_first").is_cancelled(),
            "a round's analyst runs while its round is live"
        );
        crate::round_cancel::cancel(job_id);
        assert!(
            analyst("analyze_late_register_late").is_cancelled(),
            "an analyst the cancelled round dispatches afterwards must be pre-cancelled"
        );

        // The entry belongs to the round's own invocation: once it exits, a fresh round is
        // unaffected by the gone round's signal.
        drop(round);
        assert!(
            !analyst("analyze_late_register_fresh").is_cancelled(),
            "a fresh round's analyst is not pre-cancelled by a gone round's signal"
        );
    }

    /// A dangling cancel frame is re-executed, never bound to an unrelated job: the core
    /// resolution refuses an `analyze` call carrying an `action`, while the plain form
    /// still resolves to the quick round.
    #[test]
    fn a_cancel_frame_is_never_resolved_to_a_dispatch() {
        assert!(
            crate::tools::SyncDurableCore::from_call(
                "analyze",
                &json!({ "action": "cancel", "job_id": "abc" }),
            )
            .is_none(),
            "a dangling cancel frame must be re-executed, never resolved to a dispatch"
        );
        assert!(
            matches!(
                crate::tools::SyncDurableCore::from_call("analyze", &json!({ "message": "q" })),
                Some(crate::tools::SyncDurableCore::Analyze)
            ),
            "a plain analyze call still resolves to its quick round"
        );
    }

    /// Offered exactly where a quick async round can be dispatched: `action`/`job_id` and
    /// the cancel note appear on the admin Assistant, the async Manager and a quick-granted
    /// holder, and nowhere else.
    #[test]
    fn the_cancel_action_is_offered_only_where_a_quick_round_can_be_dispatched() {
        let note = load_prompt("tool/analyze_cancel.md");
        for tool in [
            AnalyzeTool::admin(Role::Assistant),
            AnalyzeTool::new(DispatchMode::Async, Role::Manager),
            AnalyzeTool::granted(&[AnalyzeDepth::Quick]),
        ] {
            let description = tool.description();
            let schema = tool.parameters_schema();
            assert!(
                schema["properties"]
                    .get(crate::tools::analyze::ACTION_ARG)
                    .is_some(),
                "a holder that can dispatch a quick round must offer the cancel action: \
                 {description}"
            );
            assert!(
                schema["properties"]
                    .get(crate::tools::analyze::JOB_ID_ARG)
                    .is_some(),
                "a cancelling holder must declare the round to cancel: {description}"
            );
            assert!(
                description.contains(&note),
                "the cancel note must be declared where the action is: {description}"
            );
            // A cancel carries no dispatch argument, so the requirement is per-branch:
            // `message` (+ `depth`) to dispatch, `action` + `job_id` to cancel — never a
            // flat `required` list demanding both.
            let branches = schema["oneOf"]
                .as_array()
                .expect("a cancelling holder's schema states the two call forms");
            assert!(
                branches[0]["required"]
                    .as_array()
                    .is_some_and(|required| required.iter().any(|field| field == "message")),
                "the dispatch branch requires `message`: {schema}"
            );
            assert_eq!(
                branches[1]["required"],
                json!([
                    crate::tools::analyze::ACTION_ARG,
                    crate::tools::analyze::JOB_ID_ARG
                ]),
                "the cancel branch requires the action and the round to cancel: {schema}"
            );
            assert!(
                schema.get("required").is_none(),
                "a cancelling holder must not also state a flat required list: {schema}"
            );
        }
        for tool in [
            AnalyzeTool::new(DispatchMode::Sync, Role::Engineer),
            AnalyzeTool::new(DispatchMode::Sync, Role::Maintainer),
            AnalyzeTool::granted(&[AnalyzeDepth::Deep]),
        ] {
            let description = tool.description();
            let schema = tool.parameters_schema();
            assert!(
                schema["properties"]
                    .get(crate::tools::analyze::ACTION_ARG)
                    .is_none(),
                "a holder that cannot cancel must not declare the action: {description}"
            );
            assert!(
                schema["properties"]
                    .get(crate::tools::analyze::JOB_ID_ARG)
                    .is_none(),
                "a holder that cannot cancel must not declare a job id: {description}"
            );
            assert!(
                !description.contains(&note),
                "the cancel note must not be declared where the action is not: {description}"
            );
            assert!(
                schema.get("oneOf").is_none() && schema["required"].is_array(),
                "a holder that cannot cancel keeps the plain required list: {schema}"
            );
        }
    }

    /// The depth choice's sweep: a holder is never told about a depth it does not have, the
    /// `depth` argument exists only where a choice is declared, and `place` only where it
    /// means something. Swept over every holder, so a third depth joins by existing.
    #[test]
    fn no_holder_reads_a_depth_it_does_not_have() {
        for mask in 1u32..(1 << AnalyzeDepth::ALL.len()) {
            let kinds: Vec<AnalyzeDepth> = AnalyzeDepth::ALL
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, depth)| *depth)
                .collect();
            let tool = AnalyzeTool::granted(&kinds);
            let declared = kinds.len() > 1;
            let schema = tool.parameters_schema();
            let text = format!("{}\n{schema}", tool.description());
            for depth in AnalyzeDepth::ALL {
                let labelled = format!("`{}`", depth.as_str());
                assert!(
                    !text.contains(&labelled) || kinds.contains(&depth),
                    "{kinds:?} is told about {labelled}, which it does not have: {text}"
                );
            }
            assert_eq!(
                schema["properties"].get(AnalyzeDepth::ARG).is_some(),
                declared,
                "{kinds:?}"
            );
            // The folder belongs to the quick depth alone.
            assert_eq!(
                schema["properties"].get("place").is_some(),
                kinds.contains(&AnalyzeDepth::Quick),
                "{kinds:?}"
            );
        }
    }

    /// Seed a caller-owned job with one roster slot. `outcome` `Some` marks it Done with that
    /// stored outcome (reconstructable); `None` leaves it launched.
    async fn seed_analyze_job(ws: &Workspace, job_id: &str, pin: &str, outcome: Option<&str>) {
        let conn = &crate::session::store().conn;
        let analyst_id = format!("{job_id}_analyst");
        crate::jobs::spawn_job(
            conn,
            job_id,
            "analyze task",
            &ws.name,
            "caller-user",
            "telegram",
            crate::Role::Engineer,
            &[crate::jobs::NewAgent {
                agent_id: analyst_id.clone(),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(0),
                task: "analyze task".to_string(),
            }],
            &crate::jobs::SpawnChild::Analyze,
            Some(pin),
            None,
            None,
        )
        .await
        .unwrap();
        if let Some(outcome) = outcome {
            crate::jobs::write_agent_outcome(
                conn,
                job_id,
                &analyst_id,
                crate::jobs::RowStatus::Done,
                Some(outcome),
            )
            .await
            .unwrap();
        }
    }

    /// `find_owned_launched_jobs` returns only the caller's launched analyze/implement jobs —
    /// a research-kind and a ticket-phase-kind job on the same pin are ignored — newest first.
    #[tokio::test]
    async fn find_owned_launched_jobs_ignores_other_kinds() {
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_find_owned");
        let pin = "sync_analyze_find_pin";
        let conn = &crate::session::store().conn;

        // A research-kind job owned by the pin.
        crate::jobs::spawn_job(
            conn,
            "research_job",
            "research task",
            &ws.name,
            "caller",
            "telegram",
            crate::Role::Analyst,
            &[crate::jobs::NewAgent {
                agent_id: "research_a".to_string(),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(0),
                task: "r".to_string(),
            }],
            &crate::jobs::SpawnChild::Research,
            Some(pin),
            None,
            None,
        )
        .await
        .unwrap();
        // A ticket-phase-kind job owned by the pin (requires a real ticket for
        // the FK).
        let ticket_id = crate::util::test::make_ticket(
            crate::pipeline::board::store(),
            &ws,
            "FindOwnedJob",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        crate::jobs::spawn_job(
            conn,
            "phase_job",
            "phase task",
            &ws.name,
            "caller",
            "telegram",
            crate::Role::Engineer,
            &[crate::jobs::NewAgent {
                agent_id: "phase_a".to_string(),
                kind: crate::jobs::AgentKind::Engineer,
                idx: Some(0),
                task: "p".to_string(),
            }],
            &crate::jobs::SpawnChild::Phase {
                phase: crate::pipeline::board::TicketPhase::Analysis,
                ticket_id: ticket_id.clone(),
            },
            Some(pin),
            None,
            None,
        )
        .await
        .unwrap();

        seed_analyze_job(&ws, "analyze_newer", pin, None).await;
        seed_analyze_job(&ws, "analyze_older", pin, None).await;
        // Backdated so the newest-first assertion is deterministic at `created_at` second
        // precision.
        conn.execute(
            "UPDATE jobs SET created_at = ?1 WHERE id = ?2",
            crate::db::params!["2023-01-02T00:00:00+00:00", "analyze_older"],
        )
        .await
        .unwrap();

        let owned = crate::jobs::find_owned_launched_jobs(conn, pin)
            .await
            .unwrap();
        let ids: Vec<&str> = owned.iter().map(|j| j.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["analyze_newer", "analyze_older"],
            "only analyze jobs returned, newest first"
        );
    }

    /// Build a disputed consensus group over the given flat item ids.
    fn disputed_group(heading: &str, members: &[usize]) -> crate::consensus::GroupingGroup {
        crate::consensus::GroupingGroup {
            heading: heading.to_string(),
            contradiction: true,
            members: members
                .iter()
                .map(|id| crate::consensus::GroupingMember {
                    id: *id,
                    ..Default::default()
                })
                .collect(),
        }
    }

    /// One [`verification_units`] partition scenario.
    struct VerificationUnitCase {
        name: &'static str,
        headings: &'static [&'static str],
        /// Group count of each unit, in order (the first takes the extra).
        expected_sizes: &'static [usize],
    }

    #[test]
    fn verification_units_cases() {
        for case in [
            VerificationUnitCase {
                name: "a single disputed group is one unit",
                headings: &["Alpha"],
                expected_sizes: &[1],
            },
            VerificationUnitCase {
                name: "three groups split 2/1 with the first taking the extra",
                headings: &["A", "B", "C"],
                expected_sizes: &[2, 1],
            },
            VerificationUnitCase {
                name: "four groups split 2/2",
                headings: &["A", "B", "C", "D"],
                expected_sizes: &[2, 2],
            },
        ] {
            let groups: Vec<_> = case
                .headings
                .iter()
                .map(|h| disputed_group(h, &[]))
                .collect();
            let refs = groups.iter().collect::<Vec<_>>();
            let units = verification_units(&refs);
            let sizes: Vec<usize> = units.iter().map(Vec::len).collect();
            assert_eq!(sizes, case.expected_sizes, "[{}] unit sizes", case.name);
            let flattened: Vec<&str> = units
                .iter()
                .flat_map(|unit| unit.iter().map(|group| group.heading.as_str()))
                .collect();
            assert_eq!(
                flattened, case.headings,
                "[{}] groups keep their order",
                case.name
            );
        }
    }

    #[test]
    fn synthesize_verification_target_merges_heading_claims_sources_contradictions() {
        // Two disputed groups whose member claims each carry a source and a self-reported
        // contradiction (member caveat).
        let mut f1 = findings(vec![("alpha is true", "url1", "high")]);
        f1.claims[0].contradictions = vec!["src says alpha may be false".into()];
        let f2 = findings(vec![("alpha is false", "url2", "high")]);
        let mut f3 = findings(vec![("beta is true", "url3", "medium")]);
        f3.claims[0].contradictions = vec!["src says beta unknown".into()];
        let f4 = findings(vec![("beta is false", "url4", "medium")]);
        let outcomes = vec![
            AnalystOutcome::Findings {
                raw: "r1".into(),
                findings: f1,
            },
            AnalystOutcome::Findings {
                raw: "r2".into(),
                findings: f2,
            },
            AnalystOutcome::Findings {
                raw: "r3".into(),
                findings: f3,
            },
            AnalystOutcome::Findings {
                raw: "r4".into(),
                findings: f4,
            },
        ];
        let items = claims_per_agent(&outcomes);
        let table = crate::consensus::ItemTable::new(&items);
        let group = |heading: &str, ids: &[usize]| disputed_group(heading, ids);
        // Unit = one disputed group (Alpha over ids 0 and 1).
        let groups = [group("Alpha", &[0, 1])];
        let unit = groups.iter().collect::<Vec<_>>();
        let target = synthesize_verification_target(&unit, &table, &outcomes);
        assert!(
            target
                .claim
                .contains("Alpha: alpha is true; alpha is false"),
            "claim synthesizes heading + member claims: {}",
            target.claim
        );
        assert!(
            target.sources.contains("url1") && target.sources.contains("url2"),
            "sources aggregate member sources: {}",
            target.sources
        );
        assert!(
            target
                .contradictions
                .contains("src says alpha may be false"),
            "member self-reported contradictions aggregate: {}",
            target.contradictions
        );

        // No results render nothing; every verdict is one bullet; multi-line claim/evidence
        // is collapsed to semicolons; fenced untrusted text is escaped.
        assert!(
            render_verification_section(&[]).is_empty(),
            "an empty set renders nothing"
        );
        let verdicts = vec![
            VerificationResult {
                claim: "Alpha: x\nsecond".into(),
                verdict: "supported".into(),
                evidence: "primary source confirms".into(),
                tool_calls: 1,
                searches: 2,
                queries: vec!["q".into()],
            },
            VerificationResult {
                claim: "Beta: y".into(),
                verdict: "unresolved".into(),
                evidence: "```no evidence```\nfound".into(),
                tool_calls: 0,
                searches: 1,
                queries: vec![],
            },
        ];
        assert_eq!(
            render_verification_section(&verdicts),
            "## Verification\n- Alpha: x; second → **supported** — primary source confirms\n- Beta: y → **unresolved** — \\`\\`\\`no evidence\\`\\`\\`; found",
            "one bullet per verdict, newlines collapsed and fences escaped"
        );
    }

    #[test]
    #[expect(clippy::unchecked_time_subtraction)] // fixture deliberately constructs a past deadline from monotonic arithmetic
    fn verification_window_open_guards_half_round_window() {
        // A fresh round (deadline = now + window) is open: verification runs.
        let open = verification_window_open(std::time::Instant::now() + round_timeout());
        assert!(
            open,
            "verification runs while ≤ half the window has elapsed"
        );
        // An elapsed round (deadline already in the past) is closed: skip.
        let closed = verification_window_open(std::time::Instant::now() - round_timeout() * 2);
        assert!(
            !closed,
            "verification skipped when more than half the window elapsed"
        );
    }

    /// (e) With drain active, a sync analyze dispatch surfaces [`CallSuspended`] and leaves
    /// the caller-owned job `launched` — never terminalized.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn sync_analyze_draincut_returns_call_suspended_error() {
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_sync_analyze_draincut");
        let pin = "sync_analyze_draincut_pin";
        let conn = &crate::session::store().conn;

        crate::shutdown::drain_begin();
        let tool = AnalyzeTool::new(DispatchMode::Sync, crate::Role::Engineer);
        let res = crate::agent::CURRENT_TOOL_AGENT_ID
            .scope(Some(pin.to_string()), async {
                tool.execute(&ws, json!({"message": "analyze task"})).await
            })
            .await;
        crate::shutdown::drain_clear();

        let err = res.expect_err("drain must cut the sync analyze dispatch");
        assert!(
            err.downcast_ref::<crate::tools::CallSuspended>().is_some(),
            "CallSuspended carrier expected: {err:#}"
        );

        // The durable job stays launched, caller-owned by the pin.
        let rows = conn
            .query(
                "SELECT status, caller_agent_id FROM jobs WHERE caller_agent_id = ?1 AND kind = 'analyze'",
                crate::db::params![pin],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "one launched analyze job");
        assert_eq!(rows[0].get::<String>(0).unwrap(), "launched");
        assert_eq!(
            rows[0].get::<String>(1).unwrap(),
            pin,
            "job is caller-owned by the session pin"
        );
    }

    /// (f) The resume-completion path resumes an all-Done roster: the Done slot is
    /// reconstructed (zero LLM calls), the result settles contiguously after the frame, and
    /// the job is terminalized.
    // Joins the `provider` serial group: the seam swaps the process-global provider.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn resume_hook_completion_resumes_durable_analyze() {
        let fake = std::sync::Arc::new(FakeProvider::new());
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_analyze_resume_hook");
        let pin = "sync_analyze_resume_pin";
        let conn = &crate::session::store().conn;
        let job_id = "sync_analyze_resume_job";

        // Caller session [user, assistant frame with an analyze tool call].
        crate::util::test::seed_session_row(conn, pin, "user", "analyze task").await;
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_analyze_f".to_string(),
                name: "analyze".to_string(),
                arguments: json!({"message": "analyze task"}),
            }],
            None,
        )
        .to_string();
        crate::util::test::seed_session_row(conn, pin, "assistant", &frame).await;
        // Owned analyze job with an all-Done roster (outcome RESULT_ONE).
        seed_analyze_job(&ws, job_id, pin, Some("RESULT_ONE")).await;

        // Load the caller session and run the resume+settle path.
        let mut session = crate::session::Session::default();
        session.init(pin).await.unwrap();
        let pending = session
            .pending_tool_frame()
            .expect("dangling analyze call")
            .calls;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "call_analyze_f");

        let outcome = crate::tools::SyncDurableCore::Analyze
            .resume_sync_core(&ws, job_id, true)
            .await
            .unwrap();
        let crate::jobs::SyncResumeOutcome::Terminal(_, _, result) = outcome else {
            panic!("expected a terminal resume outcome");
        };
        let text = result.expect("resumed analyze result");
        assert_eq!(text, "RESULT_ONE");
        crate::jobs::terminalize_job(conn, job_id).await.unwrap();

        session
            .settle_tool_results(pin, &[("call_analyze_f".to_string(), text.clone())], &[])
            .await
            .unwrap();

        // The Done slot was reconstructed — the seam saw zero LLM calls.
        assert!(
            fake.request_fingerprints.lock().unwrap().is_empty(),
            "resume of an all-Done roster must not call the model"
        );

        // Job terminalized.
        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "resumed job must be terminalized");

        // Tool result row contiguous after the frame; pending_jobs empty.
        let rows = conn
            .query(
                "SELECT id, role, content FROM sessions WHERE agent_id = ?1 ORDER BY id",
                crate::db::params![pin],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1].get::<String>(1).unwrap(), "assistant");
        assert_eq!(rows[2].get::<String>(1).unwrap(), "tool");
        let tool_content = rows[2].get::<String>(2).unwrap();
        let payload: crate::ToolResultPayload = serde_json::from_str(&tool_content).unwrap();
        assert_eq!(payload.tool_call_id, "call_analyze_f");
        assert_eq!(payload.content, "RESULT_ONE");
        let pending_jobs = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending_jobs.is_empty(), "no envelope pending after resume");
    }
}
