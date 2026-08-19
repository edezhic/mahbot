//! Periodic OS temp-dir cleaner: a Sanitation-role agent that removes
//! clearly-old, abandoned agent artifacts from the common OS temp folder —
//! including the pinned daemon temp root (`/tmp/mahbot`) — on the nightly
//! discovery pass cadence.
//!
//! ## Dispatch + cadence
//!
//! Dispatched fire-and-forget from the nightly-check loop's gated pass block
//! (`crate::workspace::run_nightly_check_loop`), so it inherits the rolling
//! 7-day discovery cadence (at most one cleaner per 7 days; the gate's
//! pass-start timestamp also dedups the two 30-min wakes in the nightly
//! window). The durable `temp_cleanup` jobs row isolates the cleaner from the
//! discovery pass: a crash leaves the row 'launched', and the boot scan
//! (`crate::jobs::recover_from_restart`) terminalizes leftover rows WITHOUT
//! resuming — the cleanup is best-effort and the next scheduled pass simply
//! runs again.
//!
//! ## Safety
//!
//! The cleaner's only safety mechanism is its task prompt
//! (`src/prompt/sanitation/temp_cleanup.md`) — deliberately over-specified
//! deletion criteria (age + artifact shape + daemon uid, with hard bans on
//! sockets/FIFOs, fresh files, and active research run folders). There are NO
//! programmatic exclusions and no shell-guard changes; the read-only shell's
//! existing TEMP_MUTATORS gate on temp roots is what permits deletion at all.

use crate::Workspace;
use anyhow::Result;
use std::path::Path;

/// Prompt asset for the periodic temp-dir cleaner task.
const TEMP_CLEANUP_PROMPT_KEY: &str = "sanitation/temp_cleanup.md";

/// Synthetic workspace name for the cleaner. Never registered in
/// workspaces.db — ephemeral, like research run roots. Explicit name: the
/// pinned temp root's last path component is "mahbot", which would collide
/// with the registered repo workspace via `Workspace::from_path`.
const TEMP_CLEANUP_WORKSPACE_NAME: &str = "tmp";

/// Does a `temp_cleanup` jobs row survive (status != 'done')? The dispatch
/// dedup marker: a surviving row means a previous cleaner never finished
/// (crash); the boot scan terminalizes leftover rows, so a row present at
/// dispatch time is an anomaly — skip rather than double-run.
pub(crate) async fn temp_cleanup_row_exists(conn: &crate::turso::Connection) -> Result<bool> {
    Ok(conn
        .query_optional(
            "SELECT 1 FROM jobs WHERE kind = 'temp_cleanup' AND status != 'done' LIMIT 1",
            (),
            |_| Ok::<(), anyhow::Error>(()),
        )
        .await?
        .is_some())
}

/// Dispatch the periodic temp-dir cleaner (fire-and-forget, Sanitation role).
///
/// Called from the nightly-check loop's gated pass block so the cleaner
/// inherits the discovery cadence (at most once per 7 days). A failure here
/// only logs — the discovery pass continues (and vice versa).
pub(crate) async fn dispatch_temp_cleanup() -> Result<()> {
    let conn = &crate::session::store().conn;
    if temp_cleanup_row_exists(conn).await? {
        tracing::info!("Temp-dir cleanup already in flight — skipping dispatch");
        return Ok(());
    }

    let job_id = crate::generate_id();
    // Ephemeral workspace over the common OS temp dir: the shell cwd lands
    // under the scan roots and workspace-relative reads resolve there.
    let ws = Workspace::ephemeral_run(TEMP_CLEANUP_WORKSPACE_NAME, Path::new("/tmp"));
    let prompt = crate::prompt::load_prompt(TEMP_CLEANUP_PROMPT_KEY);

    crate::jobs::spawn_job(
        conn,
        &job_id,
        &prompt,
        &ws.name,
        "",
        "",
        crate::Role::Sanitation,
        &[crate::jobs::NewAgent {
            agent_id: crate::research_cleanup::cleanup_agent_id(&job_id),
            kind: crate::jobs::AgentKind::Sanitation,
            idx: None,
            task: prompt.clone(),
        }],
        &crate::jobs::SpawnChild::TempCleanup,
    )
    .await
    .map_err(|e| {
        tracing::error!(job = %job_id, error = %e, "Failed to spawn temp-dir cleanup job");
        e
    })?;

    // Fire-and-forget: raw tokio::spawn (NOT spawn_cancellable) — same
    // rationale as the research-run cleanup. Durability is the jobs row; the
    // boot scan terminalizes leftovers instead of resuming them.
    let ws = ws.clone();
    let job_id_log = job_id.clone();
    let agent_id = crate::research_cleanup::cleanup_agent_id(&job_id);
    let agent_id_log = agent_id.clone();
    tokio::spawn(async move {
        run_temp_cleanup_and_finish(&job_id, &ws, &prompt).await;
    });

    tracing::info!(job = %job_id_log, agent = %agent_id_log, "Temp-dir cleanup dispatched");
    Ok(())
}

/// Run the cleaner agent and terminalize the job row on EVERY exit path.
///
/// No folder hold exists for this cleaner (unlike research cleanup) — the row
/// is simply deleted when the run finishes (success OR failure), so the next
/// scheduled pass re-runs cleanly.
async fn run_temp_cleanup_and_finish(job_id: &str, ws: &Workspace, prompt: &str) {
    let agent_id = crate::research_cleanup::cleanup_agent_id(job_id);
    let (agent, response) = crate::agent::run_default_agent(
        &agent_id,
        crate::Role::Sanitation,
        ws,
        prompt,
        None,
        None,
        None,
    )
    .await;
    let report = response.unwrap_or_else(|| {
        format!(
            "Temp-dir cleanup FAILED (job {job_id}): {}",
            agent
                .failure
                .clone()
                .unwrap_or_else(|| "no failure detail".to_string())
        )
    });
    tracing::info!(
        job = %job_id,
        agent = %agent_id,
        "Temp-dir cleanup finished: {}",
        crate::util::scrub_credentials(&report)
    );
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn init_stores() {
        crate::util::test::init_management_test_stores().await;
    }

    /// Dispatch must be deduped by a surviving `temp_cleanup` jobs row (what
    /// a crash-interrupted cleaner leaves behind) — no second row, no agent
    /// spawn. Mirrors the research-cleanup dedup test. Serialized with the
    /// jobs boot-arm test on `reset_inflight`: both create `temp_cleanup`
    /// rows in the same process-global session store, and this test counts
    /// ALL `temp_cleanup` rows (a wrongly-run dispatch would create a fresh
    /// random id, so the count cannot be id-scoped like the research dedup
    /// test) — a concurrent row from the boot-arm test would inflate it.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shares session-store rows with the jobs boot-arm test
    async fn dispatch_temp_cleanup_deduped_by_jobs_row() {
        init_stores().await;
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            "tmpclean_dedup",
            "task",
            "tmp",
            "",
            "",
            crate::Role::Sanitation,
            &[],
            &crate::jobs::SpawnChild::TempCleanup,
        )
        .await
        .unwrap();
        assert!(
            temp_cleanup_row_exists(&crate::session::store().conn)
                .await
                .unwrap(),
            "the pre-created row is the dedup marker"
        );
        dispatch_temp_cleanup().await.unwrap();
        let rows = crate::session::store()
            .conn
            .query("SELECT COUNT(*) FROM jobs WHERE kind = 'temp_cleanup'", ())
            .await
            .unwrap();
        assert_eq!(
            rows[0].get::<i64>(0).unwrap(),
            1,
            "single temp_cleanup job row"
        );
        let sessions = crate::session::store()
            .conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = 'cleanup_tmpclean_dedup'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            sessions[0].get::<i64>(0).unwrap(),
            0,
            "deduped dispatch must not spawn the temp cleaner"
        );
        crate::jobs::terminalize_job(&crate::session::store().conn, "tmpclean_dedup")
            .await
            .unwrap();
    }
}
