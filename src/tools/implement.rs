//! ImplementTool — spawns a single coder sub-agent to carry out a clearly-scoped
//! implementation task. The coder has full shell/read/edit/search access and
//! mutates the workspace, so this is a side-effecting tool that never runs
//! concurrently with other tools.
//!
//! Two dispatch modes:
//! - [`DispatchMode::Sync`] — the Engineer blocks until the coder completes and
//!   returns its response inline.
//! - [`DispatchMode::Async`] — the Assistant dispatches the coder in a durable
//!   background job and the result is injected back to the caller's agent
//!   channel via [`crate::agent::message_router::route`] as an
//!   [`MessageKind::ImplementResult`] envelope.

use crate::agent::message_router::{self, AgentJob, MessageKind};
use crate::agent::run_agent;
use crate::session::analyze_agent_id;
use crate::tools::Tool;
use crate::tools::analyze::DispatchMode;
use crate::{Role, Workspace};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::FutureExt;
use serde_json::json;

pub struct ImplementTool {
    /// Controls how the coder sub-agent is dispatched.
    /// - [`DispatchMode::Sync`] — blocks the caller until the coder completes.
    /// - [`DispatchMode::Async`] — dispatches in a durable background job, result
    ///   delivered via the caller's agent queue.
    dispatch_mode: DispatchMode,
    /// The role of the calling agent. Used to route async results to the
    /// correct agent channel.
    pub caller_role: Role,
}

impl ImplementTool {
    #[must_use]
    pub const fn new(dispatch_mode: DispatchMode, caller_role: Role) -> Self {
        Self {
            dispatch_mode,
            caller_role,
        }
    }
}

#[async_trait]
impl Tool for ImplementTool {
    fn name(&self) -> &'static str {
        "implement"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "task": {
                    "type": "string",
                    "description": "The implementation task to delegate to the coder sub-agent"
                }
            }),
            &["task"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let task = super::get_str(&args, "task")?;

        // Async dispatch path — delegate to a single durable coder in the
        // background. Read user context from task-locals (set once per tool
        // batch by the Agent work loop) so the queued result carries the
        // correct user identity for per-user delivery.
        if self.dispatch_mode.is_async() {
            let ws = ws.clone();
            let task = task.to_string();
            let caller_role = self.caller_role;
            let user_name = crate::agent::CURRENT_TOOL_USER_NAME
                .try_with(String::clone)
                .unwrap_or_default();
            let channel = crate::agent::CURRENT_TOOL_CHANNEL
                .try_with(String::clone)
                .unwrap_or_default();

            tokio::spawn(async move {
                // Catch panics so the caller ALWAYS receives an envelope — a
                // panic in the dispatch task would otherwise leave the caller
                // waiting forever on a result that can never arrive.
                let round = std::panic::AssertUnwindSafe(async {
                    dispatch_durable_implement(
                        &ws,
                        &task,
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
                        // route NOTHING now (a spurious error envelope during the
                        // drain would discard the checkpointed outcome; the result
                        // envelope is delivered after boot resume).
                        return;
                    }
                    Err(panic) => {
                        let panic = crate::util::panic_message(&*panic);
                        tracing::error!(panic = %panic, "implement round dispatch panicked");
                        AgentJob {
                            content: build_async_implement_message(&Err(anyhow!(
                                "implement round dispatch panicked: {panic}"
                            ))),
                            workspace_name: ws.name.clone(),
                            user_name,
                            channel,
                            kind: MessageKind::ImplementResult,
                            role: caller_role,
                            reply_target: None,
                            pending_job_id: None,
                        }
                    }
                };

                // Drain/shutdown fired between the dispatch and the route: the
                // pending row (if any) survives for boot replay — skip routing
                // rather than deliver into a consumer that has stopped pulling.
                if crate::shutdown::aborting() {
                    return;
                }
                message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
            });

            return Ok("Sub-agent dispatched. Results will follow shortly.".to_string());
        }

        // Sync path — spawn one coder and block until it completes, through
        // the durable implement core with a caller-owned jobs row keyed by the
        // caller's session pin: a graceful drain mid-call surfaces
        // [`crate::tools::CallSuspended`] — the call's result is left absent
        // (the job stays launched) and the session's universal
        // resume-completion step settles it before the next LLM call (this
        // dispatch itself never binds to prior jobs — it spawns fresh).
        // The coder inherits the calling agent's DIRECT PARENT INVOCATION group
        // (e.g. a ticket an engineer is working) via the tool task-local, so
        // the Running Agents view groups it under the same parent.
        run_sync_implement(ws, task, self.caller_role).await
    }
}

// ── Durable async dispatch (SPAWN → run → CHECKPOINT → COMPLETE) ─────────

/// Durable async implement dispatch (SPAWN → run → CHECKPOINT → COMPLETE).
///
/// 1. SPAWN: one tx — INSERT jobs (kind=implement, task) + INSERT agents (the
///    single pre-generated coder id). MUST commit before the coder's first
///    session write.
/// 2. RUN: the single coder.
/// 3. CHECKPOINT: the coder's terminal outcome.
/// 4. COMPLETE: one tx — INSERT pending_jobs (envelope id = job id) + DELETE
///    jobs row — the exactly-once persistence boundary.
///
/// Returns `None` only when the round was cut by drain/shutdown (job left
/// status='launched' for boot resume — nothing to route now). On error the
/// envelope still routes (errors wrapped in `<implement-tool-result>`), and the
/// job is terminalized.
async fn dispatch_durable_implement(
    ws: &Workspace,
    task: &str,
    caller_role: Role,
    user_name: String,
    channel: String,
) -> Option<AgentJob> {
    let job_id = crate::generate_id();
    let result = match run_implement_with_job(
        ws,
        task,
        crate::tools::CoreJobArgs {
            job_id: &job_id,
            caller_role,
            user_name: &user_name,
            channel: &channel,
            resume: false,
            caller_agent_id: None,
            fail_on_checkpoint_error: false,
        },
    )
    .await
    {
        Ok(crate::tools::SyncCoreOutcome::DrainCut) => {
            // Drain-cut: the coder's outcome is checkpointed — leave the job
            // status='launched' for boot resume (recoverable from the roster
            // outcome). No terminalization, no error envelope: a spurious
            // envelope here would discard the checkpointed outcome and
            // contradict "jobs stay status='launched' for boot resume".
            tracing::info!(
                job = %job_id,
                "Implement round cut short by drain — job stays launched for boot resume",
            );
            return None;
        }
        Ok(crate::tools::SyncCoreOutcome::Terminal(result)) => result,
        Err(e) => Err(e),
    };
    let envelope = crate::jobs::complete_durable_job(
        &job_id,
        build_async_implement_message(&result),
        MessageKind::ImplementResult,
        caller_role,
        &user_name,
        &channel,
        &ws.name,
    )
    .await;
    Some(envelope)
}

// ── Sync dispatch ────────────────────────────────────────────────────────

/// Sync implement dispatch through the durable core: the jobs row + coder
/// roster commit BEFORE any coder session write, keyed by the caller's
/// session pin (`CURRENT_TOOL_AGENT_ID`) so a graceful drain mid-call leaves
/// the job `launched` for deterministic resume — the emission-time frame is
/// already durably recorded, so the call's result is simply absent until the
/// universal resume-completion step settles it.
async fn run_sync_implement(ws: &Workspace, task: &str, caller_role: Role) -> Result<String> {
    crate::tools::SyncDurableCore::Implement
        .run_sync_dispatch(ws, task, caller_role)
        .await
}

/// Spawn the implement job + single-coder roster (one tx), then run the coder.
/// `resume` reuses the stored roster (never regenerate ids — the PK would
/// conflict AND the new id would not match the stored roster row).
#[expect(clippy::too_many_lines)]
pub(crate) async fn run_implement_with_job(
    ws: &Workspace,
    task: &str,
    args: crate::tools::CoreJobArgs<'_>,
) -> anyhow::Result<crate::tools::SyncCoreOutcome> {
    let crate::tools::CoreJobArgs {
        job_id,
        caller_role,
        user_name,
        channel,
        resume,
        caller_agent_id,
        fail_on_checkpoint_error,
    } = args;
    let (coder_agent_id, pre_done) = if resume {
        let rows = crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
        let row = rows
            .first()
            .ok_or_else(|| anyhow!("Implement resume: no coder roster row"))?;
        // Only a completed (done) coder's outcome is reconstructable — a
        // failed/launched coder is re-run with its stored task on resume.
        let pre_done = (row.status == crate::jobs::RowStatus::Done.as_str())
            .then(|| row.outcome.clone())
            .flatten();
        (row.agent_id.clone(), pre_done)
    } else {
        let suffix = crate::generate_suffix();
        let coder_agent_id = analyze_agent_id(&ws.name, Role::Coder.as_str()) + &suffix;
        // Single-coder roster: one tx puts the job + the coder row down before
        // any session write (caller identity + task persisted on the rows).
        let agents = vec![crate::jobs::NewAgent {
            agent_id: coder_agent_id.clone(),
            kind: crate::jobs::AgentKind::Coder,
            idx: Some(0),
            task: task.to_string(),
        }];
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            job_id,
            task,
            &ws.name,
            user_name,
            channel,
            caller_role,
            &agents,
            &crate::jobs::SpawnChild::Implement,
            caller_agent_id,
        )
        .await?;
        (coder_agent_id, None)
    };

    // A completed coder's stored outcome IS the final response — deliver it
    // without re-running (the LLM work is never lost or duplicated).
    if let Some(outcome) = pre_done {
        return Ok(crate::tools::SyncCoreOutcome::Terminal(Ok(outcome)));
    }

    // Fresh (or resume-without-outcome) run: run the single coder. On resume the
    // coder reuses its persisted session so an interrupted attempt continues
    // rather than redoing the whole task (an empty message when the session
    // already holds the task).
    let parent_key = crate::agent::CURRENT_TOOL_PARENT_KEY
        .try_with(std::clone::Clone::clone)
        .unwrap_or(None);
    let parent_label = crate::agent::CURRENT_TOOL_PARENT_LABEL
        .try_with(std::clone::Clone::clone)
        .unwrap_or(None);
    let has_session = resume && crate::session::store().has_content(&coder_agent_id).await;
    let (agent, response) = run_agent(
        coder_agent_id.clone(),
        Role::Coder,
        ws,
        None,
        if has_session { "" } else { task },
        user_name.to_string(),
        channel.to_string(),
        false,
        None,
        resume,
        None,
        parent_key,
        parent_label,
    )
    .await;

    // CHECKPOINT: persist the coder's terminal outcome so a drain-cut / crash
    // resume can reconstruct it without re-running a completed coder.
    let (status, outcome) = match &response {
        Some(r) => (crate::jobs::RowStatus::Done, r.clone()),
        None => (
            crate::jobs::RowStatus::Failed,
            agent.failure_reason("coder produced no response"),
        ),
    };
    if let Err(e) = crate::jobs::write_agent_outcome(
        &crate::session::store().conn,
        job_id,
        &coder_agent_id,
        status,
        Some(&outcome),
    )
    .await
    {
        // Sync calls fail the tool call on a checkpoint DB error so the model
        // retries; async/boot-resume warn-and-continue (the outcome is
        // recomputable on the next resume).
        if fail_on_checkpoint_error {
            return Err(e.context("failed to checkpoint implement outcome"));
        }
        tracing::warn!(job = %job_id, error = %e, "Failed to checkpoint implement outcome");
    }

    // Drain/shutdown cut the round: leave the job status='launched' for boot
    // resume (the checkpointed outcome is the resume boundary). No routing, no
    // terminalization.
    if crate::shutdown::aborting() {
        return Ok(crate::tools::SyncCoreOutcome::DrainCut);
    }

    Ok(crate::tools::SyncCoreOutcome::Terminal(match response {
        Some(r) => Ok(r),
        None => Err(anyhow!(
            "Sub-agent failed: {}",
            agent.failure_reason("unknown error")
        )),
    }))
}

/// Boot resume of an implement round: re-run (or reconstruct) the single coder
/// and terminalize the result into a pending envelope like a fresh dispatch —
/// delivered to the ORIGINAL caller (role/user/channel persisted on the job row
/// at spawn).
///
/// Aborts quietly on shutdown/drain: no routing, no terminalization — the job
/// row stays for the next boot (the checkpointed outcome is reused).
pub(crate) async fn resume_implement_round(job_id: &str, ws: &Workspace) {
    let (caller_role, caller, result) = match crate::tools::SyncDurableCore::Implement
        .resume_sync_core(ws, job_id, false)
        .await
    {
        Ok(crate::jobs::SyncResumeOutcome::Terminal(caller_role, caller, result)) => {
            (caller_role, caller, result)
        }
        // Drain-cut mid-resume / job row vanished: abort quietly — the core
        // logs the reason; a missing job was already warned + terminalized.
        Ok(
            crate::jobs::SyncResumeOutcome::DrainCut | crate::jobs::SyncResumeOutcome::JobMissing,
        ) => return,
        // Resume infra failure (roster load / checkpoint) — deliver an error
        // envelope to the ORIGINAL caller, exactly like a round-level failure.
        // The job row is still present (the core never terminalizes), so the
        // caller identity can be re-loaded for the route.
        Err(e) => {
            let Some((caller, caller_role)) = crate::jobs::resume_job_preamble(
                &crate::session::store().conn,
                job_id,
                "Implement resume",
                "Implement resume",
            )
            .await
            else {
                return;
            };
            let envelope = crate::jobs::complete_durable_job(
                job_id,
                build_async_implement_message(&Err(e)),
                MessageKind::ImplementResult,
                caller_role,
                &caller.user_name,
                &caller.channel,
                &ws.name,
            )
            .await;
            message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
            return;
        }
    };
    // Drain/shutdown fired after the round returned: abort quietly WITHOUT
    // routing and WITHOUT deleting the row — the outcome is checkpointed (next
    // boot reuses it) and routing a partial result here would race the exit.
    if crate::shutdown::aborting() {
        tracing::info!(job = %job_id, "Implement resume aborted — job stays for next boot");
        return;
    }
    let envelope = crate::jobs::complete_durable_job(
        job_id,
        build_async_implement_message(&result),
        MessageKind::ImplementResult,
        caller_role,
        &caller.user_name,
        &caller.channel,
        &ws.name,
    )
    .await;
    message_router::route(&crate::jobs::envelope_target(&envelope), envelope);
}

/// Build the `<implement-tool-result>` envelope message for an async implement
/// dispatch. Reuses the shared analyze builder — the envelope shape that
/// reaches the caller's agent channel is production code.
fn build_async_implement_message(result: &anyhow::Result<String>) -> String {
    crate::tools::analyze::build_async_result_envelope(result, "implement-tool-result")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use serde_json::json;

    #[tokio::test]
    async fn test_implement_missing_args() {
        let tool = ImplementTool::new(DispatchMode::Sync, Role::Coder);
        let ws = test_ws("/tmp/test_ws");

        let result = tool.execute(&ws, json!({})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: task"),
            "Should mention missing task"
        );
    }

    /// (h) Mirror the analyze durability lifecycle for implement: with drain
    /// active the sync dispatch surfaces [`CallSuspended`] and leaves the
    /// caller-owned job launched; a Done-coder resume then reconstructs
    /// "CODER_RESPONSE", settles it as the tool result contiguous after the
    /// frame, and terminalizes the job.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    #[expect(clippy::too_many_lines)] // deliberate: the drain-cut + resume lifecycle is one cohesive assertion bed
    async fn sync_implement_draincut_and_completion_resumes_durable_job() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let ws = test_ws("/tmp/test_ws_sync_implement");
        let pin = "sync_implement_pin";
        let conn = &crate::session::store().conn;
        crate::util::test::seed_session_row(conn, pin, "user", "implement this").await;

        // (e) drain → CallSuspended; the durable job stays launched, caller-owned.
        crate::shutdown::drain_begin();
        let tool = ImplementTool::new(DispatchMode::Sync, crate::Role::Engineer);
        let res = crate::agent::CURRENT_TOOL_AGENT_ID
            .scope(Some(pin.to_string()), async {
                tool.execute(&ws, json!({"task": "implement task"})).await
            })
            .await;
        crate::shutdown::drain_clear();
        let err = res.expect_err("drain must cut the sync implement dispatch");
        assert!(
            err.downcast_ref::<crate::tools::CallSuspended>().is_some(),
            "CallSuspended carrier expected: {err:#}"
        );

        let jobs = conn
            .query(
                "SELECT id, status, caller_agent_id FROM jobs WHERE caller_agent_id = ?1 AND kind = 'implement'",
                crate::db::params![pin],
            )
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1, "one launched implement job");
        assert_eq!(jobs[0].get::<String>(1).unwrap(), "launched");
        assert_eq!(
            jobs[0].get::<String>(2).unwrap(),
            pin,
            "job is caller-owned by the session pin"
        );
        let job_id = jobs[0].get::<String>(0).unwrap();

        // Simulate the coder checkpointing Done just before the drain cut: the
        // drain observed the round mid-flight, the durable outcome is intact.
        let roster = crate::jobs::list_agents_for_job(conn, &job_id)
            .await
            .unwrap();
        let coder_id = roster[0].agent_id.clone();
        crate::jobs::write_agent_outcome(
            conn,
            &job_id,
            &coder_id,
            crate::jobs::RowStatus::Done,
            Some("CODER_RESPONSE"),
        )
        .await
        .unwrap();

        // (f) completion: seed the caller frame and resume + settle.
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_implement_h".to_string(),
                name: "implement".to_string(),
                arguments: json!({"task": "implement task"}),
            }],
            None,
        )
        .to_string();
        crate::util::test::seed_session_row(conn, pin, "assistant", &frame).await;

        let mut session = crate::session::Session::default();
        session.init(pin).await.unwrap();
        let pending = session
            .pending_tool_calls()
            .expect("dangling implement call");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "call_implement_h");

        let outcome = crate::tools::SyncDurableCore::Implement
            .resume_sync_core(&ws, &job_id, true)
            .await
            .unwrap();
        let crate::jobs::SyncResumeOutcome::Terminal(_, _, result) = outcome else {
            panic!("expected a terminal resume outcome");
        };
        let text = result.expect("resumed implement result");
        assert_eq!(text, "CODER_RESPONSE");
        crate::jobs::terminalize_job(conn, &job_id).await.unwrap();
        session
            .settle_tool_results(pin, &[("call_implement_h".to_string(), text.clone())], &[])
            .await
            .unwrap();

        // Job terminalized; tool row CODER_RESPONSE contiguous after the frame.
        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "resumed job must be terminalized");
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
        let payload: crate::ToolResultPayload =
            serde_json::from_str(&rows[2].get::<String>(2).unwrap()).unwrap();
        assert_eq!(payload.tool_call_id, "call_implement_h");
        assert_eq!(payload.content, "CODER_RESPONSE");
    }
}
