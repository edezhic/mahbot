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
//!
//! The admin Assistant's instance additionally accepts an optional `target`:
//! for the coder the only usable form is an absolute path to a directory
//! outside the registered workspaces — a registered workspace, anything inside
//! one, the filesystem root and MahBot's own data directory (a personal
//! workspace directory is fine) are refused. Every other holder keeps its
//! interface with a `target` they pass inert; argument mechanics and refusals
//! live in [`super::delegation`].

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
    /// Whether the optional `target` argument is accepted — the admin
    /// Assistant's instance only. Every other holder keeps the argument-less
    /// interface; a `target` they pass is inert.
    accepts_target: bool,
}

impl ImplementTool {
    #[must_use]
    pub const fn new(dispatch_mode: DispatchMode, caller_role: Role) -> Self {
        Self {
            dispatch_mode,
            caller_role,
            accepts_target: false,
        }
    }

    /// The admin Assistant's variant: accepts the optional `target` argument.
    ///
    /// Always the async dispatch — a target can only be honoured by a durable
    /// background round, which records the directory it runs in for resume.
    #[must_use]
    pub const fn targeted(caller_role: Role) -> Self {
        Self {
            dispatch_mode: DispatchMode::Async,
            caller_role,
            accepts_target: true,
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
    /// value. The admin Assistant's targeted variant first appends the
    /// `tool/implement_target.md` note explaining the optional `target`
    /// argument.
    fn description(&self) -> String {
        let mut desc = crate::prompt::load_prompt(&format!("tool/{}.md", self.name()));
        if self.accepts_target {
            let target_note =
                crate::prompt::load_prompt(&format!("tool/{}_target.md", self.name()));
            desc = format!("{desc}\n\n{target_note}");
        }
        if self.dispatch_mode.is_async() {
            let async_note = crate::prompt::load_prompt(&format!("tool/{}_async.md", self.name()));
            desc = format!("{desc}\n\n{async_note}");
        }
        desc
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let mut properties = json!({
            "task": {
                "type": "string",
                "description": "The implementation task to delegate to the coder sub-agent"
            }
        });
        if self.accepts_target {
            properties["target"] = json!({
                "type": "string",
                "description": "Optional absolute path to the directory this coder round runs in (outside the registered workspaces). Omit to run in your own workspace."
            });
        }
        super::tool_params_schema(&properties, &["task"])
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let task = super::get_str(&args, "task")?;

        // Async dispatch path — delegate to a single durable coder in the
        // background. Spawn/identity/drain-cut/panic/route semantics live in
        // `SyncDurableCore::spawn_dispatch`.
        if self.dispatch_mode.is_async() {
            // The optional `target` is the admin's alone and only a durable
            // background round can honour it — resolved here, where it is used.
            // A refused target dispatches nothing and never falls back to the
            // caller's own workspace — resolution is exact and decided here.
            let target = if self.accepts_target {
                super::delegation::resolve(super::delegation::DelegationKind::Implement, &args)
                    .await?
            } else {
                None
            };
            let job_id = crate::generate_id();
            super::SyncDurableCore::Implement.spawn_dispatch(
                ws,
                target.as_ref(),
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
        // (a target is impossible here: `targeted()` is async-only)
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
/// A resume (`spawn: None`) reuses the stored roster (never regenerate ids —
/// the PK would conflict AND the new id would not match the stored roster row).
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
        spawn,
        caller_agent_id,
        fail_on_checkpoint_error,
    } = args;
    let resume = spawn.is_none();
    let (coder_agent_id, pre_done) = match &spawn {
        // Resume: reuse the stored roster (never regenerate ids — the PK would
        // conflict AND the new id would not match the stored roster row).
        None => {
            let rows =
                crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
            let row = rows.first().ok_or_else(|| {
                super::internal_fault(
                    "implement resume found no coder roster row for the stored job",
                )
            })?;
            // Only a completed (done) coder's outcome is reconstructable — a
            // failed/launched coder is re-run with its stored task on resume.
            let pre_done = (row.status == crate::jobs::RowStatus::Done.as_str())
                .then(|| row.outcome.clone())
                .flatten();
            (row.agent_id.clone(), pre_done)
        }
        // Fresh dispatch: spawn the job + single-coder roster (one tx).
        Some(spawn) => {
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
                spawn.delivery_name,
                user_name,
                channel,
                caller_role,
                &agents,
                &crate::jobs::SpawnChild::Implement,
                caller_agent_id,
                // The round's execution directory (the caller's target, when one was
                // asked for) so a restart resumes into the same directory.
                spawn.exec_dir,
            )
            .await?;
            (coder_agent_id, None)
        }
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
/// [`crate::tools::SyncDurableCore::resume_durable_round`]. `exec_dir` is the
/// recorded target directory the round ran in, when it had one.
pub(crate) async fn resume_implement_round(job_id: &str, ws: &Workspace, exec_dir: Option<&str>) {
    crate::tools::SyncDurableCore::Implement
        .resume_durable_round(job_id, ws, exec_dir)
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{FakeProvider, install_fake_provider};
    use crate::workspace::test_ws;
    use serde_json::json;

    /// The file the scripted coder writes into its execution workspace — a REAL
    /// write through the `edit` tool, so the round's root is observable on disk.
    const WRITTEN_FILE: &str = "coder-written.txt";

    /// Dispatch one targeted coder round and assert the shared contract: the
    /// coder writes into `target`'s directory, while the durable job and the
    /// delivered envelope stay keyed to the caller's own workspace and the
    /// envelope names the directory really used.
    async fn assert_targeted_round_write(delivery: &Workspace, target: &Workspace, job_id: &str) {
        assert!(
            target.name.starts_with("dir-"),
            "a plain directory's identity is the synthetic one, got {}",
            target.name
        );
        let envelope = crate::tools::SyncDurableCore::Implement
            .dispatch_durable(
                delivery,
                Some(target),
                "write the file",
                Role::Assistant,
                "admin",
                "gui",
                job_id,
            )
            .await
            .expect("the round completes");
        assert_eq!(
            envelope.workspace_name, delivery.name,
            "the delivery key stays the caller's own workspace"
        );
        assert!(
            envelope.content.contains("CODER DONE"),
            "{}",
            envelope.content
        );
        assert!(
            envelope
                .content
                .contains(&format!("Directory: {}", target.path)),
            "{}",
            envelope.content
        );
        let written = std::path::Path::new(&target.path).join(WRITTEN_FILE);
        assert_eq!(
            std::fs::read_to_string(&written).unwrap_or_default(),
            "hello from the coder",
            "the coder wrote into the target directory ({})",
            written.display()
        );
        let conn = &crate::session::store().conn;
        let target_sessions = conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![target.name.clone()],
            )
            .await
            .unwrap();
        assert_eq!(
            target_sessions.len(),
            1,
            "the coder's session belongs to the target directory"
        );
        assert!(
            target_sessions[0]
                .get::<String>(0)
                .unwrap()
                .contains(&target.name),
            "the coder's agent id carries the target's identity"
        );
        let caller_sessions = conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![delivery.name.clone()],
            )
            .await
            .unwrap();
        assert!(
            caller_sessions.is_empty(),
            "nothing ran in the caller's own workspace"
        );
    }

    /// The async targeting path end to end, in both shapes: a delivery of its
    /// own plus a separate target directory, and a target that IS the caller's
    /// own directory — still targeted, the reply names the directory and the
    /// coder's session is keyed to the directory's own identity, never silently
    /// downgraded to an untargeted call.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn targeted_rounds_run_in_the_target_and_cover_the_callers_own_directory() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Two rounds' scripts in order: each coder really writes the file
        // through `edit`, then answers (`ok` is text-only).
        let _provider_guard = install_fake_provider(std::sync::Arc::new(
            FakeProvider::new()
                .ok_text_and_tool_calls(
                    "writing the file",
                    &[(
                        "edit",
                        json!({ "path": WRITTEN_FILE, "new_string": "hello from the coder" }),
                    )],
                )
                .ok("CODER DONE")
                .ok_text_and_tool_calls(
                    "writing the file",
                    &[(
                        "edit",
                        json!({ "path": WRITTEN_FILE, "new_string": "hello from the coder" }),
                    )],
                )
                .ok("CODER DONE"),
        ));

        // Case 1: a delivery of its own plus a separate target tempdir.
        let delivery = test_ws("/tmp/test_ws_target_delivery");
        let dir = tempfile::tempdir().unwrap();
        let dir = std::fs::canonicalize(dir.path()).unwrap();
        let target = crate::tools::delegation::workspace_at_dir(dir.to_str().unwrap())
            .await
            .unwrap();
        assert_targeted_round_write(&delivery, &target, "implement_targeted_job").await;

        // Case 2: the target IS the caller's own directory.
        let dir = tempfile::tempdir().unwrap();
        let dir = std::fs::canonicalize(dir.path()).unwrap();
        let dir_path = dir.to_str().unwrap().to_string();
        let delivery = test_ws(&dir_path);
        let target = crate::tools::delegation::workspace_at_dir(&dir_path)
            .await
            .unwrap();
        assert_ne!(
            target.name, delivery.name,
            "the directory's own identity differs from the caller's name"
        );
        assert_targeted_round_write(&delivery, &target, "implement_own_dir_job").await;
    }

    /// The coder tool's own hop: `execute` resolves the target, refuses a
    /// forbidden one before dispatching anything, and hands the target to the
    /// async round (which then runs in that directory, never the caller's).
    /// Called without the `CURRENT_TOOL_*` task-locals: the dispatch still runs
    /// — only the routed envelope's user/channel are empty, which this test does
    /// not assert.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn the_targeted_implement_tool_dispatches_into_the_target() {
        crate::util::test::init_management_test_stores().await;
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider_guard =
            install_fake_provider(std::sync::Arc::new(FakeProvider::new().ok("CODER DONE")));
        let caller = test_ws("/tmp/test_ws_implement_tool_caller");
        let dir = tempfile::tempdir().unwrap();
        let target_path = crate::stored_workspace_path(dir.path())
            .to_str()
            .unwrap()
            .to_string();

        // A refused target dispatches nothing: the refusal comes back from the
        // tool itself, before any job id exists.
        let refused = ImplementTool::targeted(Role::Assistant)
            .execute(&caller, json!({ "task": "do it", "target": "/" }))
            .await
            .expect_err("the tool refuses the filesystem root");
        assert!(refused.to_string().starts_with("usage:"), "{refused}");

        let ack = ImplementTool::targeted(Role::Assistant)
            .execute(&caller, json!({ "task": "do it", "target": target_path }))
            .await
            .expect("a directory target dispatches");
        assert!(ack.contains("dispatched"), "{ack}");

        // The round runs in the target directory — never in the caller's.
        let target_name = crate::tools::delegation::workspace_at_dir(&target_path)
            .await
            .unwrap()
            .name;
        let conn = &crate::session::store().conn;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let rows = conn
                .query(
                    "SELECT COUNT(*) FROM session_metadata WHERE workspace_name = ?1",
                    crate::db::params![target_name.clone()],
                )
                .await
                .unwrap();
            if rows[0].get::<i64>(0).unwrap() > 0 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the round never ran in the target directory {target_name}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        let caller_sessions = conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![caller.name.clone()],
            )
            .await
            .unwrap();
        assert_eq!(
            caller_sessions[0].get::<i64>(0).unwrap(),
            0,
            "nothing ran in the caller's own workspace"
        );
    }

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
                .contains("usage: missing required argument \"task\""),
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
            .pending_tool_frame()
            .expect("dangling implement call")
            .calls;
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
