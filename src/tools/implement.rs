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

use crate::agent::run_agent;
use crate::session::analyze_agent_id;
use crate::tools::Tool;
use crate::tools::analyze::DispatchMode;
use crate::{Role, Workspace};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
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

    /// Mode-keyed tool description (see [`DispatchMode`]).
    ///
    /// The sync variant returns the shared `tool/implement.md` asset verbatim.
    /// The async variant appends the `tool/implement_async.md` note so an agent
    /// reading the schema instantly understands that the coder's result arrives
    /// later as an injected follow-up result message, not in the tool's return
    /// value.
    fn description(&self) -> String {
        let base = crate::prompt::load_prompt(&format!("tool/{}.md", self.name()));
        if self.dispatch_mode.is_async() {
            let async_note = crate::prompt::load_prompt(&format!("tool/{}_async.md", self.name()));
            format!("{base}\n\n{async_note}")
        } else {
            base
        }
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
        // background. Spawn/identity/drain-cut/panic/route semantics live in
        // `SyncDurableCore::spawn_dispatch`.
        if self.dispatch_mode.is_async() {
            let job_id = crate::generate_id();
            super::SyncDurableCore::Implement.spawn_dispatch(
                ws,
                task,
                self.caller_role,
                job_id.clone(),
            );
            return Ok(format!(
                "Sub-agent dispatched (job {job_id}). Results will follow shortly."
            ));
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

/// Boot-resume a durable implement round — thin wrapper over
/// [`crate::tools::SyncDurableCore::resume_durable_round`].
pub(crate) async fn resume_implement_round(job_id: &str, ws: &Workspace) {
    crate::tools::SyncDurableCore::Implement
        .resume_durable_round(job_id, ws)
        .await;
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

    /// Mode-keyed description: the sync variant is the base asset; the async
    /// variant appends the async-note so an agent sees the result arrives later
    /// as an injected envelope, not in the return value.
    #[test]
    fn implement_description_mode_keyed() {
        let sync = ImplementTool::new(DispatchMode::Sync, Role::Coder);
        assert!(
            !sync.description().contains("dispatched asynchronously"),
            "sync description must not carry the async note"
        );

        let async_tool = ImplementTool::new(DispatchMode::Async, Role::Coder);
        let async_desc = async_tool.description();
        assert!(
            async_desc.contains("dispatched asynchronously"),
            "async description must carry the async note"
        );
        assert!(
            async_desc.contains("<implement-tool-result>"),
            "async note names the result envelope"
        );
    }

    /// (h) Mirror the analyze durability lifecycle for implement: with drain
    /// active the sync dispatch surfaces [`CallSuspended`] and leaves the
    /// caller-owned job launched; a Done-coder resume then reconstructs
    /// "CODER_RESPONSE", settles it as the tool result contiguous after the
    /// frame, and terminalizes the job.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn sync_implement_draincut_and_completion_resumes_durable_job() {
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
