//! Per-run cancel signal registry for deep-research runs, plus the permanent
//! cancel sweep (durable row removal, folder release, archive deletion).
//!
//! A manual cancel from the Running Agents page is a DISTINCT, PERMANENT stop:
//! unlike the shutdown/drain abort (the run stays alive and resumes at boot),
//! a manual cancel removes every durable trace of the run so it can never
//! resume, re-dispatch, or deliver a report — now or after any restart.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use tokio_util::sync::CancellationToken;

use crate::registry::ParentKey;
use crate::util::UnwrapPoison;

/// Per-run cancel signals. An entry exists from the moment a run's
/// orchestrator starts (fresh dispatch or boot resume) until it exits (any
/// exit — terminal, abort, cancel, panic — via the RAII guard). [`cancel`]
/// FIRES the entry's token but deliberately leaves the entry in place so
/// [`is_cancelled`] keeps returning true for the orchestrator's gates until
/// the orchestrator actually stops and its guard removes the entry.
static RESEARCH_CANCELS: LazyLock<ResearchCancelRegistry> =
    LazyLock::new(ResearchCancelRegistry::default);

#[derive(Default)]
struct ResearchCancelRegistry {
    inner: Mutex<HashMap<String, CancellationToken>>,
}

impl ResearchCancelRegistry {
    fn register(&'static self, job_id: &str) -> ResearchCancelGuard {
        let token = CancellationToken::new();
        self.inner
            .lock()
            .unwrap_poison()
            .insert(job_id.to_string(), token.clone());
        ResearchCancelGuard {
            job_id: job_id.to_string(),
            registry: self,
        }
    }

    /// Fire a run's cancel signal. No-op when the run is not registered
    /// (already terminalized — the sweep handles the durable state alone).
    /// The entry is NOT removed here: [`is_cancelled`] must keep reporting
    /// true until the orchestrator exits, so mid-terminalize gates observe it.
    fn cancel(&self, job_id: &str) {
        let map = self.inner.lock().unwrap_poison();
        if let Some(token) = map.get(job_id) {
            token.cancel();
        }
    }

    /// Whether the run's cancel signal has been fired. False for runs never
    /// registered (terminalized) and for live runs whose token is unfired.
    fn is_cancelled(&self, job_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_poison()
            .get(job_id)
            .is_some_and(CancellationToken::is_cancelled)
    }

    fn unregister(&self, job_id: &str) {
        self.inner.lock().unwrap_poison().remove(job_id);
    }
}

/// RAII guard: removes the run's cancel-signal entry when the orchestrator
/// invocation exits — on every path, including panics (Drop runs during
/// unwind). The fields are private; only [`register`] constructs it and the
/// guard's drop is its only externally-visible behavior.
pub(crate) struct ResearchCancelGuard {
    job_id: String,
    registry: &'static ResearchCancelRegistry,
}

impl Drop for ResearchCancelGuard {
    fn drop(&mut self) {
        self.registry.unregister(&self.job_id);
    }
}

/// Register the run's cancel signal for the duration of one orchestrator
/// invocation (fresh dispatch or boot resume). Callers hold the returned
/// guard for the whole invocation.
pub(crate) fn register(job_id: &str) -> ResearchCancelGuard {
    RESEARCH_CANCELS.register(job_id)
}

/// Fire the run's cancel signal — the user cancelled the run from the
/// Running Agents page. Must be called BEFORE any row deletion so the
/// orchestrator observes it at its next boundary and stops.
pub(crate) fn cancel(job_id: &str) {
    RESEARCH_CANCELS.cancel(job_id);
}

/// Whether the run's cancel signal has been fired. Consulted by the
/// orchestrator's boundary gates and the completion/cleanup gates.
pub(crate) fn is_cancelled(job_id: &str) -> bool {
    RESEARCH_CANCELS.is_cancelled(job_id)
}

/// The manual cancel action, run from the GUI on confirm:
/// 1. Fire the run's cancel signal FIRST (before any row deletion) so the
///    orchestrator observes it at its next round/stage boundary and stops —
///    it must not spawn further rounds, synthesize, terminalize, or dispatch
///    a cleanup agent.
/// 2. Cancel every agent of the run (analysts, verifier, coder, and the
///    cleanup agent if one was already dispatched) as a group via the run's
///    parent key. Cancellation is cooperative: an in-flight tool or LLM call
///    may finish, but no further work happens and no new sub-agents spawn.
/// 3. Remove the run's observational call-registry rows so the group
///    disappears from Running Agents immediately (the guards' drops become
///    no-ops).
/// 4. Remove the run's durable state terminally (pending + jobs rows in one
///    tx, then the folder, then the results.md archive).
pub(crate) async fn cancel_research_run(job_id: &str) -> Result<(), String> {
    cancel(job_id);
    crate::registry::AGENT_REGISTRY.cancel_by_parent_key(&ParentKey::Research(job_id.to_string()));
    crate::registry::NON_AGENT_CALLS.remove_by_parent_key(&ParentKey::Research(job_id.to_string()));
    sweep_cancelled_run(job_id).await
}

/// Remove a cancelled run's durable state terminally and idempotently:
/// - pending_jobs row (a persisted report would be replayed at boot) and the
///   jobs row in ONE tx. The jobs DELETE cascades research_jobs + the roster;
///   it also removes the `research_cleanup` row when the run already
///   terminalized (the cleanup row reuses id == run_id, and at most one of
///   the research/cleanup job rows exists at any time — the research row is
///   deleted by the completion boundary before the cleanup row is created).
/// - the run folder (search/tracker state first, then the folder) — safe when
///   the folder is already missing or being released concurrently;
/// - the results.md archive (Behavior 5: no archive file).
///
/// Idempotent and safe to run concurrently from both the cancel action and
/// the orchestrator's cancelled-exit path — every statement is a no-op on
/// missing rows. The tx is the only fallible part (surfaced to the GUI
/// toast); folder/archive failures are logged, never fatal.
pub(crate) async fn sweep_cancelled_run(job_id: &str) -> Result<(), String> {
    let conn = &crate::session::store().conn;
    let tx = conn.begin_tx().await.map_err(|e| format!("{e:#}"))?;
    let outcome: anyhow::Result<()> = async {
        tx.execute(
            "DELETE FROM pending_jobs WHERE id = ?1",
            crate::turso::params![job_id],
        )
        .await?;
        tx.execute(
            "DELETE FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
        )
        .await?;
        Ok(())
    }
    .await;
    match outcome {
        Ok(()) => tx.commit().await.map_err(|e| format!("{e:#}"))?,
        Err(e) => return Err(format!("{e:#}")),
    }
    crate::research_cleanup::release_run_folder(job_id).await;
    delete_results_archive(job_id).await;
    Ok(())
}

/// Delete the run's results.md archive
/// (`<storage_root>/research/results/{run_id}.md`). Fail-open: a leftover
/// archive is a bounded race with a racing terminalize write (the accepted
/// envelope-race class — the file is never routed to any session); the
/// terminalize gates keep that window to a single racing write.
async fn delete_results_archive(job_id: &str) {
    let root = crate::config::CONFIG
        .try_storage_root()
        .or_else(|| crate::config::default_config_dir().ok());
    let Some(root) = root else {
        return;
    };
    let path = root
        .join("research")
        .join("results")
        .join(format!("{job_id}.md"));
    match tokio::fs::remove_file(&path).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(job = %job_id, error = %e, "results.md archive deletion failed — left on disk");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{JobRowBuilder, retry_tests_lock};

    #[tokio::test]
    async fn cancel_fires_signal_and_guard_removes_entry_on_drop() {
        let _lock = retry_tests_lock();
        // Unregistered run: cancel is a no-op, is_cancelled false.
        cancel("nope");
        assert!(!is_cancelled("nope"));
        let guard = register("run_signal_1");
        assert!(!is_cancelled("run_signal_1"));
        cancel("run_signal_1");
        assert!(is_cancelled("run_signal_1"), "fired signal stays visible");
        // A second cancel is idempotent.
        cancel("run_signal_1");
        assert!(is_cancelled("run_signal_1"));
        drop(guard);
        assert!(
            !is_cancelled("run_signal_1"),
            "guard drop removes the entry"
        );
    }

    /// Mid-run cancel: jobs + research_jobs rows + pending row + folder +
    /// archive all removed; nothing is resumable or deliverable.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn sweep_removes_rows_folder_and_archive_for_mid_run_cancel() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_midrun_1";
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        JobRowBuilder::new(conn, job_id, "research", "assistant", "ws")
            .task("q")
            .user_name("u")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, state) VALUES (?1, '{}')",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, envelope, target_agent_id, created_at) \
             VALUES (?1, ?2, 'manager_ws', ?3)",
            crate::turso::params![
                job_id,
                r#"{"content":"report","workspace_name":"ws","user_name":"u","channel":"telegram","kind":"ResearchResult","role":"assistant","reply_target":null,"pending_job_id":null}"#,
                now
            ],
        )
        .await
        .unwrap();
        // Run folder + archive.
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        tokio::fs::write(run_root.join("commands.dump"), "shell cmd")
            .await
            .unwrap();
        let root = crate::config::CONFIG
            .try_storage_root()
            .or_else(|| crate::config::default_config_dir().ok())
            .unwrap();
        let archive = root
            .join("research")
            .join("results")
            .join(format!("{job_id}.md"));
        tokio::fs::create_dir_all(archive.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&archive, "archive").await.unwrap();

        sweep_cancelled_run(job_id).await.unwrap();

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "jobs row removed");
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending.is_empty(), "pending row removed — no boot replay");
        assert!(!run_root.exists(), "run folder removed");
        assert!(!archive.exists(), "results.md archive removed");
    }

    /// Cancel after terminalization: the pending report row + the
    /// research_cleanup job row are removed (boot replay must have nothing to
    /// deliver, and the cleanup must not resume).
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn sweep_removes_pending_and_cleanup_rows_for_terminalized_run() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_terminal_1";
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        JobRowBuilder::new(conn, job_id, "research_cleanup", "sanitation", "ws")
            .task("cleanup prompt")
            .user_name("")
            .channel("")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, envelope, target_agent_id, created_at) \
             VALUES (?1, ?2, 'manager_ws', ?3)",
            crate::turso::params![
                job_id,
                r#"{"content":"report","workspace_name":"ws","user_name":"u","channel":"telegram","kind":"ResearchResult","role":"assistant","reply_target":null,"pending_job_id":null}"#,
                now
            ],
        )
        .await
        .unwrap();
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        let root = crate::config::CONFIG
            .try_storage_root()
            .or_else(|| crate::config::default_config_dir().ok())
            .unwrap();
        let archive = root
            .join("research")
            .join("results")
            .join(format!("{job_id}.md"));
        tokio::fs::create_dir_all(archive.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&archive, "archive").await.unwrap();

        sweep_cancelled_run(job_id).await.unwrap();

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "cleanup job row removed");
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending.is_empty(), "pending report row removed");
        assert!(!run_root.exists());
        assert!(!archive.exists());
    }

    /// The sweep is idempotent; a double release (racing cleanup tail) is safe.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn sweep_is_idempotent_and_double_release_safe() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_double_1";
        sweep_cancelled_run(job_id).await.unwrap();
        // Second pass with nothing present must not fail.
        sweep_cancelled_run(job_id).await.unwrap();
    }

    /// The in-tx cancel gate: a completion racing a cancel must roll back —
    /// no pending row may survive, and the jobs row stays for the sweep.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn complete_durable_job_rolls_back_when_run_cancelled() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_complete_1";
        let ws = crate::workspace::test_ws("/tmp/test_ws_research_cancel_complete");
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            job_id,
            "q",
            &ws.name,
            "caller-user",
            "telegram",
            crate::Role::Assistant,
            &[],
            &crate::jobs::SpawnChild::Research,
        )
        .await
        .unwrap();
        let _guard = register(job_id);
        cancel(job_id);
        let envelope = crate::jobs::complete_durable_job(
            job_id,
            "report".to_string(),
            crate::message_router::MessageKind::ResearchResult,
            crate::Role::Assistant,
            "caller-user",
            "telegram",
            &ws.name,
        )
        .await;
        // The in-tx gate rolled the completion back: no pending row exists
        // (no boot replay), and the jobs row survives for the cancel sweep.
        let conn = &crate::session::store().conn;
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert!(
            pending.is_empty(),
            "rolled-back completion leaves no pending row"
        );
        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::turso::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(
            jobs.len(),
            1,
            "job row survives the rollback — the sweep deletes it"
        );
        assert!(
            envelope.pending_job_id.is_some(),
            "the caller gate (is_cancelled) decides routing"
        );
    }
}
