//! Per-run cancel signal registry for deep-research runs, plus the cancelled-run
//! sweep (durable row removal, folder release, archive deletion).
//!
//! A manual cancel from the Running Agents page is a DISTINCT, PERMANENT stop:
//! unlike the shutdown/drain abort (the run stays alive and resumes at boot),
//! a manual cancel stops the run so it can never resume, re-dispatch, or
//! deliver a report. But the run folder is the cleanup intent: the
//! ORCHESTRATOR's cancelled-exit path owns dump finalization, cleanup
//! dispatch, and folder release; the GUI/abandon action only fires the signal
//! (plus sweeps rows itself only when no orchestrator is alive to observe it).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use tokio_util::sync::CancellationToken;

use crate::Workspace;
use crate::agent::registry::ParentKey;
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

    /// Whether a live orchestrator invocation is registered for the run. A
    /// registered entry means the orchestrator is alive and can observe the
    /// fired signal at its next boundary — the cancelled-exit handoff owns the
    /// durable sweep. An unregistered id means no orchestrator is alive, so the
    /// cancel action sweeps directly.
    fn is_registered(&self, job_id: &str) -> bool {
        self.inner.lock().unwrap_poison().contains_key(job_id)
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

/// The manual cancel action, run from the GUI on confirm and the workspace
/// abandon path:
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
/// 4. If no orchestrator is alive to observe the signal
///    ([`ResearchCancelRegistry::is_registered`]), sweep the durable rows
///    directly ([`sweep_cancelled_run`] — the folder release is dump-guarded)
///    and log it; if an orchestrator IS alive, return and let its
///    cancelled-exit handoff own the rest (cleanup dispatch + folder release).
///
/// Infallible by design: the orchestrator's cancelled-exit path (or the no-
/// orchestrator sweep) always makes progress, so there is nothing for the GUI
/// to surface as a cancel failure.
pub(crate) async fn cancel_research_run(job_id: &str) {
    cancel(job_id);
    crate::agent::registry::AGENT_REGISTRY
        .cancel_by_parent_key(&ParentKey::Research(job_id.to_string()));
    crate::agent::registry::NON_AGENT_CALLS
        .remove_by_parent_key(&ParentKey::Research(job_id.to_string()));
    if !RESEARCH_CANCELS.is_registered(job_id)
        && let Err(e) = sweep_cancelled_run(job_id).await
    {
        tracing::warn!(job = %job_id, error = %e, "cancel sweep failed — durable rows left for boot resume");
    }
}

/// Remove a cancelled run's durable state terminally and idempotently — the
/// no-orchestrator path (runs with NO cleanup intent, or a cancelled run whose
/// orchestrator is already gone so no live cancelled-exit handoff can sweep).
/// - pending_jobs row (a persisted report would be replayed at boot) and the
///   jobs row in ONE tx. The jobs DELETE cascades research_jobs + the roster;
///   it also removes the `research_cleanup` row when the run already
///   terminalized (the cleanup row reuses id == run_id, and at most one of
///   the research/cleanup job rows exists at any time — the research row is
///   deleted by the completion boundary before the cleanup row is created).
/// - the run folder ONLY when it has NO command dump: a folder holding a
///   `commands.dump` is the run's cleanup intent, which must survive for the
///   cleanup tail (or the OS temp sweep). See [`command_dump_exists`].
/// - the results.md archive (no archive file).
///
/// Idempotent and safe to run concurrently — every statement is a no-op on
/// missing rows. The tx is the only fallible part; folder/archive failures are
/// logged, never fatal.
pub(crate) async fn sweep_cancelled_run(job_id: &str) -> Result<(), String> {
    let conn = &crate::session::store().conn;
    let tx = conn.begin_tx().await.map_err(|e| format!("{e:#}"))?;
    tx.execute(
        "DELETE FROM pending_jobs WHERE id = ?1",
        crate::db::params![job_id],
    )
    .await
    .map_err(|e| format!("{e:#}"))?;
    tx.execute("DELETE FROM jobs WHERE id = ?1", crate::db::params![job_id])
        .await
        .map_err(|e| format!("{e:#}"))?;
    tx.commit().await.map_err(|e| format!("{e:#}"))?;
    if crate::research_cleanup::command_dump_exists(job_id).await {
        tracing::warn!(
            job = %job_id,
            "run folder has a command dump — cleanup intent present, folder NOT released (left for the cleanup tail / OS sweep)"
        );
    } else {
        crate::research_cleanup::release_run_folder(job_id).await;
    }
    delete_results_archive(job_id).await;
    Ok(())
}

/// The orchestrator's cancelled-exit cleanup, the owner of the run-folder
/// release for a cancelled run. Called from the orchestrator's Cancelled exit
/// (fresh dispatch and boot resume) and the terminalize cancel gates.
///
/// The command dump on disk is ALREADY FINAL at this point BY CONSTRUCTION:
/// [`ResearchState::capture_round`](crate::tools::research) writes are awaited
/// inline by the orchestrator, and the Cancelled exit is only reachable after
/// the current round's members were reaped — so the GUI/abandon action no
/// longer deletes the folder out from under the dump (the ENOENT race is dead).
pub(crate) async fn hand_off_cancelled_run(
    job_id: &str,
    ws: &Workspace,
    question: &str,
) -> Result<(), String> {
    if let Some(prompt) = hand_off_rows(job_id, ws).await? {
        crate::research_cleanup::spawn_cleanup_agent(job_id, question, ws, &prompt);
    }
    Ok(())
}

/// The row/fs part of [`hand_off_cancelled_run`], extracted for testability:
/// returns `Some(prompt)` when the run's research row was transitioned to a
/// `research_cleanup` row (the caller must spawn the cleanup agent with that
/// prompt), or `None` when the handoff was suppressed (a cleanup row already
/// exists, so the boot-resumable tail owns the folder) or swept (no dump —
/// nothing to clean).
async fn hand_off_rows(job_id: &str, ws: &Workspace) -> Result<Option<String>, String> {
    let conn = &crate::session::store().conn;
    if crate::research_cleanup::research_cleanup_row_exists(conn, job_id)
        .await
        .map_err(|e| format!("{e:#}"))?
    {
        // DEFENSIVE: a `research_cleanup` row already existing here is NOT
        // expected on the Cancelled exits — cleanup dispatch only happens after
        // the completion boundary that a Cancelled exit never reaches. The
        // branch exists to keep the racing-completion window safe: if the row
        // is there, the running/boot-resumable cleanup tail owns the folder,
        // so we suppress the pending envelope + archive and spawn nothing.
        if let Err(e) = conn
            .execute(
                "DELETE FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
        {
            tracing::warn!(job = %job_id, error = %e, "pending_jobs removal failed during cancel handoff");
        }
        delete_results_archive(job_id).await;
        return Ok(None);
    }
    if !crate::research_cleanup::command_dump_exists(job_id).await {
        // No agent ever ran — nothing to clean; delegate to the sweep (which
        // releases the dump-less folder).
        sweep_cancelled_run(job_id).await?;
        return Ok(None);
    }
    // Canonical run root — NEVER ensure_run_root: a manual cancel must not
    // resurrect a folder the OS temp sweep (or a prior release) already removed.
    let run_root = tokio::fs::canonicalize(crate::research_cleanup::run_root_path(job_id))
        .await
        .map_err(|e| format!("{e:#}"))?;
    let dump_path = run_root.join("commands.dump");
    let prompt = crate::research_cleanup::build_cleanup_prompt(job_id, &run_root, &dump_path, ws);
    crate::jobs::transition_research_to_cleanup(conn, job_id, &prompt, &ws.name)
        .await
        .map_err(|e| format!("{e:#}"))?;
    delete_results_archive(job_id).await;
    Ok(Some(prompt))
}

/// Delete the run's results.md archive
/// (`<storage_root>/research/results/{run_id}.md`). Fail-open: a leftover
/// archive is a bounded race with a racing terminalize write (the accepted
/// envelope-race class — the file is never routed to any session); the
/// terminalize gates keep that window to a single racing write.
async fn delete_results_archive(job_id: &str) {
    let Some(path) = crate::research_cleanup::results_archive_path(job_id) else {
        return;
    };
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

    /// Mid-run cancel with NO command dump (no cleanup intent): jobs +
    /// research_jobs rows + pending row + folder + archive all removed;
    /// nothing is resumable or deliverable.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn sweep_removes_rows_folder_and_archive_for_no_dump_run() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_midrun_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
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
            crate::db::params![job_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, envelope, target_agent_id, created_at) \
             VALUES (?1, ?2, 'manager_ws', ?3)",
            crate::db::params![
                job_id,
                r#"{"content":"report","workspace_name":"ws","user_name":"u","channel":"telegram","kind":"ResearchResult","role":"assistant","reply_target":null,"pending_job_id":null}"#,
                now
            ],
        )
        .await
        .unwrap();
        // Run folder (scratch only, NO command dump) + archive.
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        tokio::fs::write(run_root.join("scratch.txt"), "scratch")
            .await
            .unwrap();
        let archive = crate::research_cleanup::results_archive_path(job_id).unwrap();
        tokio::fs::create_dir_all(archive.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&archive, "archive").await.unwrap();

        sweep_cancelled_run(job_id).await.unwrap();

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "jobs row removed");
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending.is_empty(), "pending row removed — no boot replay");
        assert!(!run_root.exists(), "run folder removed");
        assert!(!archive.exists(), "results.md archive removed");
    }

    /// Cancel after terminalization (folder WITHOUT a command dump — no cleanup
    /// intent): the pending report row + the research_cleanup job row are
    /// removed (boot replay must have nothing to deliver, and the cleanup must
    /// not resume), and the dump-less folder + archive are released.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn sweep_removes_pending_and_cleanup_rows_for_terminalized_run() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_terminal_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
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
            crate::db::params![
                job_id,
                r#"{"content":"report","workspace_name":"ws","user_name":"u","channel":"telegram","kind":"ResearchResult","role":"assistant","reply_target":null,"pending_job_id":null}"#,
                now
            ],
        )
        .await
        .unwrap();
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        let archive = crate::research_cleanup::results_archive_path(job_id).unwrap();
        tokio::fs::create_dir_all(archive.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&archive, "archive").await.unwrap();

        sweep_cancelled_run(job_id).await.unwrap();

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "cleanup job row removed");
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending.is_empty(), "pending report row removed");
        assert!(!run_root.exists());
        assert!(!archive.exists());
    }

    /// A folder WITH a command dump is the run's cleanup intent: the sweep
    /// removes the jobs/pending rows and the archive, but the folder AND the
    /// dump survive (the cleanup tail / OS sweep owns their release). The warn
    /// log is not asserted.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn sweep_never_releases_folder_with_command_dump() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_dump_guard_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        JobRowBuilder::new(conn, job_id, "research", "assistant", "ws")
            .task("q")
            .user_name("u")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, envelope, target_agent_id, created_at) \
             VALUES (?1, ?2, 'manager_ws', ?3)",
            crate::db::params![
                job_id,
                r#"{"content":"report","workspace_name":"ws","user_name":"u","channel":"telegram","kind":"ResearchResult","role":"assistant","reply_target":null,"pending_job_id":null}"#,
                now
            ],
        )
        .await
        .unwrap();
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        tokio::fs::write(run_root.join("commands.dump"), "shell cmd")
            .await
            .unwrap();
        let archive = crate::research_cleanup::results_archive_path(job_id).unwrap();
        tokio::fs::create_dir_all(archive.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&archive, "archive").await.unwrap();

        sweep_cancelled_run(job_id).await.unwrap();

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "jobs row removed");
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending.is_empty(), "pending row removed");
        assert!(
            run_root.exists(),
            "run folder held — cleanup intent present"
        );
        assert!(
            run_root.join("commands.dump").exists(),
            "command dump survives"
        );
        assert!(!archive.exists(), "results.md archive removed");
    }

    /// The cancelled-run handoff with a live command dump: research + roster
    /// transition atomically to a `research_cleanup` row (+ cleanup roster),
    /// the pending envelope is dropped, and the archive removed — while the
    /// run folder AND its dump survive for the cleanup tail.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    #[expect(clippy::too_many_lines)] // the transition asserts every row/fs side effect
    async fn cancel_handoff_transitions_rows_and_holds_folder() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_handoff_transition_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        JobRowBuilder::new(conn, job_id, "research", "assistant", "ws")
            .task("question?")
            .user_name("u")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, state) VALUES (?1, '{}')",
            crate::db::params![job_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, envelope, target_agent_id, created_at) \
             VALUES (?1, ?2, 'manager_ws', ?3)",
            crate::db::params![
                job_id,
                r#"{"content":"report","workspace_name":"ws","user_name":"u","channel":"telegram","kind":"ResearchResult","role":"assistant","reply_target":null,"pending_job_id":null}"#,
                now
            ],
        )
        .await
        .unwrap();
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        tokio::fs::write(run_root.join("commands.dump"), "shell cmd")
            .await
            .unwrap();
        let archive = crate::research_cleanup::results_archive_path(job_id).unwrap();
        tokio::fs::create_dir_all(archive.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&archive, "archive").await.unwrap();

        let ws = crate::workspace::test_ws("/tmp/test_ws_handoff_transition");
        let prompt = hand_off_rows(job_id, &ws).await.unwrap();
        let prompt = prompt.expect("handoff transitions to cleanup and returns the prompt");

        let row = conn
            .query(
                "SELECT kind, role, task FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(row.len(), 1, "single jobs row after transition");
        assert_eq!(
            row[0].get::<String>(0).unwrap(),
            "research_cleanup",
            "kind transitioned to research_cleanup"
        );
        assert_eq!(
            row[0].get::<String>(1).unwrap(),
            "sanitation",
            "role is sanitation"
        );
        assert_eq!(
            row[0].get::<String>(2).unwrap(),
            prompt,
            "task is the returned cleanup prompt"
        );
        let roster = conn
            .query(
                "SELECT agent_id FROM agents WHERE job_id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(roster.len(), 1, "single cleanup roster row");
        assert_eq!(
            roster[0].get::<String>(0).unwrap(),
            format!("cleanup_{job_id}"),
            "cleanup agent roster row present"
        );
        let pending = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending.is_empty(), "pending envelope dropped");
        let child = conn
            .query(
                "SELECT id FROM research_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(
            child.is_empty(),
            "research_jobs child row removed by cascade"
        );
        assert!(run_root.exists(), "run folder held for the cleanup tail");
        assert!(
            run_root.join("commands.dump").exists(),
            "command dump survives — cleanup intent preserved"
        );
        assert!(!archive.exists(), "results.md archive removed");
    }

    /// A cancelled-run handoff with NO command dump has nothing to clean: the
    /// handoff delegates to the sweep and returns `None`; the research row and
    /// the dump-less folder are gone.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn cancel_handoff_without_dump_sweeps() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_handoff_no_dump_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        JobRowBuilder::new(conn, job_id, "research", "assistant", "ws")
            .task("question?")
            .user_name("u")
            .channel("telegram")
            .timestamps(now.clone())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO research_jobs (id, state) VALUES (?1, '{}')",
            crate::db::params![job_id],
        )
        .await
        .unwrap();
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        tokio::fs::write(run_root.join("scratch.txt"), "scratch")
            .await
            .unwrap();

        let ws = crate::workspace::test_ws("/tmp/test_ws_handoff_no_dump");
        let outcome = hand_off_rows(job_id, &ws).await.unwrap();
        assert!(outcome.is_none(), "no dump → no cleanup prompt");

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "research row swept");
        assert!(!run_root.exists(), "dump-less folder released");
    }

    /// The GUI/abandon path for a run with NO live orchestrator: firing the
    /// cancel signal sweeps the durable rows and the dump-less folder directly.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn cancel_research_run_sweeps_unregistered_run() {
        let _lock = retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_unregistered_1";
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
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
            crate::db::params![job_id],
        )
        .await
        .unwrap();
        let run_root = crate::research_cleanup::ensure_run_root(job_id).await;
        tokio::fs::write(run_root.join("scratch.txt"), "scratch")
            .await
            .unwrap();

        // No register() → no orchestrator is alive to observe the signal.
        crate::research_cancel::cancel_research_run(job_id).await;

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(
            jobs.is_empty(),
            "research row swept via the unregistered path"
        );
        assert!(!run_root.exists(), "dump-less folder released");
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
            None,
        )
        .await
        .unwrap();
        let _guard = register(job_id);
        cancel(job_id);
        let envelope = crate::jobs::complete_durable_job(
            job_id,
            "report".to_string(),
            crate::agent::message_router::MessageKind::ResearchResult,
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
                crate::db::params![job_id],
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
                crate::db::params![job_id],
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
