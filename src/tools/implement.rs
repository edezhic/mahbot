//! The durable single-coder round body (`run_implement_with_job`): the coder
//! sub-agent a `send` `coder` call dispatches. The coder has full
//! shell/read/edit/search access and mutates the workspace.
//!
//! The round itself is dispatched by [`crate::tools::SyncDurableCore::Coder`]
//! (sync for the Engineer, async for the admin Assistant and custom roles) —
//! this module owns the round body only; the tool half lives in
//! [`crate::tools::send`]. The jobs-row kind stays `implement` — the `Coder`
//! core's own kind ([`crate::jobs::SpawnChild::Implement`]).

use crate::agent::run_agent;
use crate::session::analyze_agent_id;
use crate::{Role, Workspace};
use anyhow::anyhow;

/// Write a fresh round's durable rows — the jobs row and the single-coder
/// roster, in ONE tx — and return the coder's agent id. The row fields come
/// from the call's own [`CoreJobArgs`](crate::tools::CoreJobArgs) (the same
/// bundle every durable sync core takes), whose spawn record must be present.
/// The fresh-spawn half of [`run_implement_with_job`], shared with the command
/// line's handover ([`hand_over_coder_round`]), which must have these rows down
/// BEFORE it answers its caller.
async fn spawn_implement_rows(
    exec: &Workspace,
    task: &str,
    args: &crate::tools::CoreJobArgs<'_>,
) -> anyhow::Result<String> {
    // The helper's own precondition, a fault rather than a caller-facing refusal:
    // rows are written for a FRESH dispatch only, and a fresh dispatch is exactly
    // the call whose spawn record is present (a resume inserts no row — see
    // [`CoreJobArgs::spawn`](crate::tools::CoreJobArgs::spawn)).
    let Some(record) = args.spawn.as_ref() else {
        return Err(super::internal_fault(
            "a fresh coder dispatch carries no spawn record",
        ));
    };
    let coder_agent_id =
        analyze_agent_id(&exec.name, Role::Coder.as_str()) + &crate::generate_suffix();
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
        args.job_id,
        task,
        record.delivery_name,
        args.user_name,
        args.channel,
        args.caller_role,
        &agents,
        &crate::jobs::SpawnChild::Implement,
        args.caller_agent_id,
        // The round's execution directory (the caller's place, when one was
        // asked for) so a restart resumes into the same directory.
        record.exec_dir,
        // The calling custom role's launch, when there is one, so a
        // resume rebuilds an envelope addressed to its branch.
        record.custom_launch,
    )
    .await?;
    Ok(coder_agent_id)
}

/// The COMMAND LINE's handover of a coder round.
///
/// [`crate::tools::SyncDurableCore::spawn_dispatch`] answers its caller as soon
/// as the round is started, which is right for an agent whose own turn is
/// durable: the jobs row lands moments later and a restart resumes the round
/// from it. The command line is answered and gone the moment it returns, so the
/// round's durable rows are written FIRST (awaited) and the round then runs as a
/// RESUME of its own rows — exactly the round a crash between the two would have
/// started at the next boot, so a restart neither loses the handover nor repeats
/// it.
///
/// The round's job row carries the command line's own user and channel. There is
/// no agent session and no custom-role launch behind a shell caller (the
/// caller-role branch cannot be reached), so neither is threaded through.
///
/// `Err` is a failed row write: nothing was dispatched and nothing is running,
/// so the caller may be told the handover did not happen.
pub(crate) async fn hand_over_coder_round(
    delivery: &Workspace,
    place: Option<&Workspace>,
    task: &str,
    caller_role: Role,
    job_id: &str,
) -> anyhow::Result<()> {
    // Read the caller's identity synchronously BEFORE the spawn (task-local
    // contract): the round's job row must carry the command line's own user and
    // channel.
    let user_name = crate::agent::tool_user_name();
    let channel = crate::agent::tool_channel();

    let exec = place.unwrap_or(delivery);
    let args = crate::tools::CoreJobArgs {
        job_id,
        caller_role,
        user_name: &user_name,
        channel: &channel,
        spawn: Some(crate::tools::SpawnRecord {
            delivery_name: &delivery.name,
            exec_dir: place.map(|ws| ws.path.as_str()),
            custom_launch: None,
        }),
        // The command line has no agent session behind it: no caller session pin,
        // and the row write is not a caller's retry (nothing is running yet).
        caller_agent_id: None,
        fail_on_checkpoint_error: false,
    };
    spawn_implement_rows(exec, task, &args).await?;

    // The rows are down: run the round as a RESUME of its own rows, so a restart
    // that lands between the two starts the very round this call promised. The
    // resume path reports its own outcome — a failure or a panic included — to
    // the conversation the report was promised to.
    let (delivery, job_id) = (delivery.clone(), job_id.to_string());
    let exec_dir = place.map(|ws| ws.path.clone());
    tokio::spawn(async move {
        crate::tools::SyncDurableCore::Coder
            .resume_durable_round(&job_id, &delivery, exec_dir.as_deref())
            .await;
    });
    Ok(())
}

/// Spawn the implement job + single-coder roster (one tx), then run the coder.
/// A resume (`spawn: None`) reuses the stored roster (never regenerate ids —
/// the PK would conflict AND the new id would not match the stored roster row).
pub(crate) async fn run_implement_with_job(
    ws: &Workspace,
    task: &str,
    args: crate::tools::CoreJobArgs<'_>,
) -> anyhow::Result<crate::tools::SyncCoreOutcome> {
    let resume = args.spawn.is_none();
    let (coder_agent_id, pre_done) = match &args.spawn {
        // Resume: reuse the stored roster (never regenerate ids — the PK would
        // conflict AND the new id would not match the stored roster row).
        None => {
            let rows = crate::jobs::list_agents_for_job(&crate::session::store().conn, args.job_id)
                .await?;
            let row = rows.first().ok_or_else(|| {
                super::internal_fault("coder round resume found no roster row for the stored job")
            })?;
            // Only a completed (done) coder's outcome is reconstructable — a
            // failed/launched coder is re-run with its stored task on resume.
            let pre_done = (row.status == crate::jobs::RowStatus::Done.as_str())
                .then(|| row.outcome.clone())
                .flatten();
            (row.agent_id.clone(), pre_done)
        }
        // Fresh dispatch: spawn the job + single-coder roster (one tx).
        Some(_) => (spawn_implement_rows(ws, task, &args).await?, None),
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
        args.user_name.to_string(),
        args.channel.to_string(),
        false,
        None,
        resume,
        None,
        parent_key,
        parent_label,
        None,
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
        args.job_id,
        &coder_agent_id,
        status,
        Some(&outcome),
    )
    .await
    {
        // Sync calls fail the tool call on a checkpoint DB error so the model
        // retries; async/boot-resume warn-and-continue (the outcome is
        // recomputable on the next resume).
        if args.fail_on_checkpoint_error {
            return Err(e.context("failed to checkpoint coder outcome"));
        }
        tracing::warn!(job = %args.job_id, error = %e, "Failed to checkpoint coder outcome");
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
    crate::tools::SyncDurableCore::Coder
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

    /// Dispatch one place coder round and assert the shared contract: the
    /// coder writes into `place`'s directory, while the durable job and the
    /// delivered envelope stay keyed to the caller's own workspace and the
    /// envelope names the directory really used.
    async fn assert_place_round_write(delivery: &Workspace, place: &Workspace, job_id: &str) {
        assert!(
            place.name.starts_with("dir-"),
            "a plain directory's identity is the synthetic one, got {}",
            place.name
        );
        let envelope = crate::tools::SyncDurableCore::Coder
            .dispatch_durable(
                delivery,
                Some(place),
                "write the file",
                Role::Assistant,
                "admin",
                "gui",
                job_id,
                None,
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
                .contains(&format!("folder=\"{}\"", place.path)),
            "the coder report names the folder it worked in: {}",
            envelope.content
        );
        let written = std::path::Path::new(&place.path).join(WRITTEN_FILE);
        assert_eq!(
            std::fs::read_to_string(&written).unwrap_or_default(),
            "hello from the coder",
            "the coder wrote into the place directory ({})",
            written.display()
        );
        let conn = &crate::session::store().conn;
        let place_sessions = conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE workspace_name = ?1",
                crate::db::params![place.name.clone()],
            )
            .await
            .unwrap();
        assert_eq!(
            place_sessions.len(),
            1,
            "the coder's session belongs to the place directory"
        );
        assert!(
            place_sessions[0]
                .get::<String>(0)
                .unwrap()
                .contains(&place.name),
            "the coder's agent id carries the place's identity"
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

    /// The async place path end to end, in both shapes: a delivery of its
    /// own plus a separate place directory, and a place that IS the caller's
    /// own directory — still a place, the reply names the directory and the
    /// coder's session is keyed to the directory's own identity, never silently
    /// downgraded to an unplaced call.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn place_rounds_run_in_the_place_and_cover_the_callers_own_directory() {
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

        // Case 1: a delivery of its own plus a separate place tempdir.
        let delivery = test_ws("/tmp/test_ws_place_delivery");
        let dir = tempfile::tempdir().unwrap();
        let dir = std::fs::canonicalize(dir.path()).unwrap();
        let place = crate::tools::delegation::workspace_at_dir(dir.to_str().unwrap())
            .await
            .unwrap();
        assert_place_round_write(&delivery, &place, "implement_place_job").await;

        // Case 2: the place IS the caller's own directory.
        let dir = tempfile::tempdir().unwrap();
        let dir = std::fs::canonicalize(dir.path()).unwrap();
        let dir_path = dir.to_str().unwrap().to_string();
        let delivery = test_ws(&dir_path);
        let place = crate::tools::delegation::workspace_at_dir(&dir_path)
            .await
            .unwrap();
        assert_ne!(
            place.name, delivery.name,
            "the directory's own identity differs from the caller's name"
        );
        assert_place_round_write(&delivery, &place, "implement_own_dir_job").await;
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
        let res = crate::agent::CURRENT_TOOL_AGENT_ID
            .scope(Some(pin.to_string()), async {
                crate::tools::SyncDurableCore::Coder
                    .run_sync_dispatch(&ws, "implement task", crate::Role::Engineer)
                    .await
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
                name: "send".to_string(),
                arguments: json!({"message": "implement task"}),
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

        let outcome = crate::tools::SyncDurableCore::Coder
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
