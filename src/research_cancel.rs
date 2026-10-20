//! The deep-research cancel action and the cancelled-run sweep (durable row
//! removal, folder release, archive deletion). The shared per-round cancel
//! signal registry itself lives in [`crate::round_cancel`].
//!
//! A manual cancel from the Running Agents page is a DISTINCT, PERMANENT stop:
//! unlike the shutdown/drain abort (the run stays alive and resumes at boot),
//! a manual cancel stops the run so it can never resume, re-dispatch, or
//! deliver a report. But the run folder is the cleanup intent: the
//! ORCHESTRATOR's cancelled-exit path owns dump finalization, cleanup
//! dispatch, and folder release; the GUI/abandon action only fires the signal
//! (plus sweeps rows itself only when no orchestrator is alive to observe it).

use crate::Workspace;
use crate::agent::registry::ParentKey;
use crate::research_cleanup::COMMAND_DUMP_FILE;

/// The manual cancel action, run from the GUI on confirm and the workspace
/// abandon path:
/// 1. Fire the run's cancel signal FIRST (before any row deletion) so the
///    orchestrator observes it at its next round/stage boundary and stops —
///    it must not spawn further rounds, synthesize, terminalize, or dispatch
///    a cleanup agent.
/// 2. Cancel every agent of the run (analysts, verifier, worker, and the
///    cleanup agent if one was already dispatched) as a group via the run's
///    parent key. Cancellation is cooperative: an in-flight tool or LLM call
///    may finish, but no further work happens and no new sub-agents spawn.
/// 3. Remove the run's observational call-registry rows so the group
///    disappears from Running Agents immediately (the guards' drops become
///    no-ops).
/// 4. If no orchestrator is alive to observe the signal
///    ([`crate::round_cancel::is_registered`]), sweep the durable rows
///    directly ([`sweep_cancelled_run`] — the folder release is dump-guarded)
///    and log it; if an orchestrator IS alive, return and let its
///    cancelled-exit handoff own the rest (cleanup dispatch + folder release).
///
/// Infallible by design: the orchestrator's cancelled-exit path (or the no-
/// orchestrator sweep) always makes progress, so there is nothing for the GUI
/// to surface as a cancel failure.
pub(crate) async fn cancel_research_run(job_id: &str) {
    crate::round_cancel::cancel(job_id);
    crate::agent::registry::AGENT_REGISTRY
        .cancel_by_parent_key(&ParentKey::Research(job_id.to_string()));
    crate::agent::registry::NON_AGENT_CALLS
        .remove_by_parent_key(&ParentKey::Research(job_id.to_string()));
    if !crate::round_cancel::is_registered(job_id)
        && let Err(e) = sweep_cancelled_run(job_id).await
    {
        tracing::warn!(job = %job_id, error = %e, "cancel sweep failed — durable rows left for boot resume");
    }
}

/// Remove a cancelled run's durable state terminally and idempotently — the
/// no-orchestrator path (runs with NO cleanup intent, or a cancelled run whose
/// orchestrator is already gone so no live cancelled-exit handoff can sweep).
/// - the durable rows — the pending_jobs row (a persisted report would be
///   replayed at boot) and the jobs row, in ONE tx
///   ([`crate::jobs::delete_round_rows`]), which also owns the ordering rule.
///   The jobs DELETE cascades research_jobs and the roster; it also removes the
///   `research_cleanup` row when the run already terminalized (the cleanup row
///   reuses id == run_id, and at most one of the research/cleanup job rows
///   exists at any time — the research row is deleted by the completion
///   boundary before the cleanup row is created).
/// - the run folder ONLY when it has NO command dump: a folder holding a
///   `commands.dump` is the run's cleanup intent, which must survive for the
///   cleanup tail (or the periodic temp cleaner). See [`command_dump_exists`].
/// - the results.md archive (no archive file).
///
/// Idempotent and safe to run concurrently — every statement is a no-op on
/// missing rows. The tx is the only fallible part; folder/archive failures are
/// logged, never fatal.
pub(crate) async fn sweep_cancelled_run(job_id: &str) -> Result<(), String> {
    let conn = &crate::session::store().conn;
    crate::jobs::delete_round_rows(conn, job_id)
        .await
        .map_err(|e| format!("{e:#}"))?;
    if crate::research_cleanup::command_dump_exists(job_id).await {
        tracing::warn!(
            job = %job_id,
            "run folder has a command dump — cleanup intent present, folder NOT released (left for the cleanup tail / periodic temp cleaner)"
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
    // resurrect a folder the periodic temp cleaner (or a prior release) already removed.
    let run_root = crate::research_cleanup::canonical_run_root(job_id)
        .await
        .map_err(|e| format!("{e:#}"))?;
    let dump_path = run_root.join(COMMAND_DUMP_FILE);
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
    use crate::jobs::SpawnChild;
    use crate::util::test::JobRowBuilder;
    use std::path::PathBuf;

    // ── Fixture ───────────────────────────────────────────────────────────
    // A cancelled run's durable state, stated once and never patched at the
    // call site: the jobs row shape (one constructor each), the folder's
    // contents, and whether the run left a persisted report. No `agents` roster
    // rows are seeded — every path here names the run by id, and the one roster
    // row the handoff test asserts is written by the handoff itself.

    /// The run folder's contents — the axis that decides whether the sweep may
    /// release the folder: a `commands.dump` is the cleanup intent.
    enum Folder {
        /// The folder exists and holds nothing.
        Empty,
        /// An unrelated scratch file and no dump.
        Scratch,
        /// The command dump.
        Dump,
    }

    /// The report a run left for boot replay: the completion boundary writes a
    /// `pending_jobs` envelope AND its results archive; a run cancelled before
    /// that boundary has neither.
    enum Report {
        /// No envelope, no archive.
        None,
        /// The pending envelope plus the `results.md` archive.
        Persisted,
    }

    /// A seeded run, handed back for the assertions.
    struct Seeded {
        id: &'static str,
        root: PathBuf,
        archive: PathBuf,
    }

    impl Seeded {
        /// A research run mid-flight: `kind='research'` plus its `research_jobs`
        /// child, written through the production dispatch — the sweep's jobs
        /// DELETE is what cascades the child.
        async fn research(id: &'static str, folder: Folder, report: Report) -> Self {
            crate::util::test::init_management_test_stores().await;
            crate::jobs::spawn_job(
                &crate::session::store().conn,
                id,
                "question?",
                "ws",
                "u",
                "telegram",
                crate::Role::Assistant,
                &[],
                &SpawnChild::Research,
                None,
                None,
                None,
            )
            .await
            .unwrap();
            Self::seed(id, folder, report).await
        }

        /// The same `kind='research'` row with NO `research_jobs` child:
        /// `spawn_job` always writes the child, so this shape is seeded directly.
        async fn research_without_child(id: &'static str, folder: Folder, report: Report) -> Self {
            crate::util::test::init_management_test_stores().await;
            JobRowBuilder::new(
                &crate::session::store().conn,
                id,
                "research",
                "assistant",
                "ws",
            )
            .task("question?")
            .user_name("u")
            .channel("telegram")
            .timestamps(crate::db::now())
            .insert()
            .await
            .unwrap();
            Self::seed(id, folder, report).await
        }

        /// A terminalized run: its `research_cleanup` row reuses the run id and
        /// carries no child row — the cleanup prompt lives in its `task`.
        async fn terminalized(id: &'static str, folder: Folder, report: Report) -> Self {
            crate::util::test::init_management_test_stores().await;
            crate::jobs::spawn_job(
                &crate::session::store().conn,
                id,
                "cleanup prompt",
                "ws",
                "",
                "",
                crate::Role::Sanitation,
                &[],
                &SpawnChild::ResearchCleanup,
                None,
                None,
                None,
            )
            .await
            .unwrap();
            Self::seed(id, folder, report).await
        }

        /// The rest of the run's state: the folder in the state under test, plus
        /// the persisted report — the `pending_jobs` envelope boot replay would
        /// deliver and its archive — when the run got that far.
        ///
        /// Every constructor reaches this only after the test stores are up:
        /// `results_archive_path` falls back to the real `~/.mahbot` root while
        /// `CONFIG` has none set, so resolving earlier would seed (and then
        /// delete) a file in the owner's storage.
        async fn seed(id: &'static str, folder: Folder, report: Report) -> Self {
            let root = crate::research_cleanup::ensure_run_root(id).await;
            match folder {
                Folder::Empty => {}
                Folder::Scratch => {
                    tokio::fs::write(root.join("scratch.txt"), "scratch")
                        .await
                        .unwrap();
                }
                Folder::Dump => {
                    tokio::fs::write(root.join(COMMAND_DUMP_FILE), "shell cmd")
                        .await
                        .unwrap();
                }
            }
            let archive = crate::research_cleanup::results_archive_path(id).unwrap();
            if matches!(report, Report::Persisted) {
                // The router's own persist: the envelope is the production
                // serialization, never a hand-written copy of it.
                crate::agent::message_router::persist_pending(
                    &crate::agent::message_router::AgentJob {
                        content: "report".to_string(),
                        workspace_name: "ws".to_string(),
                        user_name: "u".to_string(),
                        channel: "telegram".to_string(),
                        kind: crate::agent::message_router::MessageKind::ResearchResult,
                        role: crate::Role::Assistant,
                        reply_target: None,
                        pending_job_id: None,
                        originating_workspace: None,
                        custom_role: None,
                        reply_to_custom: None,
                    },
                    id.to_string(),
                )
                .await
                .unwrap();
                tokio::fs::create_dir_all(archive.parent().unwrap())
                    .await
                    .unwrap();
                tokio::fs::write(&archive, "archive").await.unwrap();
            }
            Self { id, root, archive }
        }

        /// How many rows `table` holds for this run: a count, not a presence
        /// flag, because a leftover duplicate is one of the failures these
        /// tests watch for. The three tables it serves are keyed by `id`; the
        /// roster is keyed by `job_id` and its one assertion reads the row's
        /// value, so it stays a bespoke query.
        async fn rows(&self, table: &str) -> usize {
            crate::session::store()
                .conn
                .query(
                    &format!("SELECT id FROM {table} WHERE id = ?1"),
                    crate::db::params![self.id],
                )
                .await
                .unwrap()
                .len()
        }
    }

    /// Mid-run cancel with NO command dump (no cleanup intent): jobs +
    /// research_jobs rows + pending row + folder + archive all removed;
    /// nothing is resumable or deliverable.
    #[tokio::test]
    async fn sweep_removes_rows_folder_and_archive_for_no_dump_run() {
        let run = Seeded::research(
            "research_cancel_midrun_1",
            Folder::Scratch,
            Report::Persisted,
        )
        .await;

        sweep_cancelled_run(run.id).await.unwrap();

        assert_eq!(run.rows("jobs").await, 0, "jobs row removed");
        assert_eq!(
            run.rows("research_jobs").await,
            0,
            "research_jobs child row removed by cascade"
        );
        assert_eq!(
            run.rows("pending_jobs").await,
            0,
            "pending row removed — no boot replay"
        );
        assert!(!run.root.exists(), "run folder removed");
        assert!(!run.archive.exists(), "results.md archive removed");
    }

    /// Cancel after terminalization (folder WITHOUT a command dump — no cleanup
    /// intent): the pending report row + the research_cleanup job row are
    /// removed (boot replay must have nothing to deliver, and the cleanup must
    /// not resume), and the dump-less folder + archive are released.
    #[tokio::test]
    async fn sweep_removes_pending_and_cleanup_rows_for_terminalized_run() {
        let run = Seeded::terminalized(
            "research_cancel_terminal_1",
            Folder::Empty,
            Report::Persisted,
        )
        .await;

        sweep_cancelled_run(run.id).await.unwrap();

        assert_eq!(run.rows("jobs").await, 0, "cleanup job row removed");
        assert_eq!(
            run.rows("pending_jobs").await,
            0,
            "pending report row removed"
        );
        assert!(!run.root.exists());
        assert!(!run.archive.exists());
    }

    /// A folder WITH a command dump is the run's cleanup intent: the sweep
    /// removes the jobs/pending rows and the archive, but the folder AND the
    /// dump survive (the cleanup tail / periodic temp cleaner owns their release). The warn
    /// log is not asserted.
    #[tokio::test]
    async fn sweep_never_releases_folder_with_command_dump() {
        // No `research_jobs` child row: the dump guard does not depend on it.
        let run = Seeded::research_without_child(
            "research_cancel_dump_guard_1",
            Folder::Dump,
            Report::Persisted,
        )
        .await;

        sweep_cancelled_run(run.id).await.unwrap();

        assert_eq!(run.rows("jobs").await, 0, "jobs row removed");
        assert_eq!(run.rows("pending_jobs").await, 0, "pending row removed");
        assert!(
            run.root.exists(),
            "run folder held — cleanup intent present"
        );
        assert!(
            run.root.join(COMMAND_DUMP_FILE).exists(),
            "command dump survives"
        );
        assert!(!run.archive.exists(), "results.md archive removed");
    }

    /// The cancelled-run handoff with a live command dump: research + roster
    /// transition atomically to a `research_cleanup` row (+ cleanup roster),
    /// the pending envelope is dropped, and the archive removed — while the
    /// run folder AND its dump survive for the cleanup tail.
    #[tokio::test]
    async fn cancel_handoff_transitions_rows_and_holds_folder() {
        let run = Seeded::research(
            "research_handoff_transition_1",
            Folder::Dump,
            Report::Persisted,
        )
        .await;
        let conn = &crate::session::store().conn;
        let ws = crate::workspace::test_ws("/tmp/test_ws_handoff_transition");

        let prompt = hand_off_rows(run.id, &ws).await.unwrap();
        let prompt = prompt.expect("handoff transitions to cleanup and returns the prompt");

        let row = conn
            .query(
                "SELECT kind, role, task FROM jobs WHERE id = ?1",
                crate::db::params![run.id],
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
                crate::db::params![run.id],
            )
            .await
            .unwrap();
        assert_eq!(roster.len(), 1, "single cleanup roster row");
        assert_eq!(
            roster[0].get::<String>(0).unwrap(),
            format!("cleanup_{}", run.id),
            "cleanup agent roster row present"
        );
        assert_eq!(
            run.rows("pending_jobs").await,
            0,
            "pending envelope dropped"
        );
        assert_eq!(
            run.rows("research_jobs").await,
            0,
            "research_jobs child row removed by cascade"
        );
        assert!(run.root.exists(), "run folder held for the cleanup tail");
        assert!(
            run.root.join(COMMAND_DUMP_FILE).exists(),
            "command dump survives — cleanup intent preserved"
        );
        assert!(!run.archive.exists(), "results.md archive removed");
    }

    /// A cancelled-run handoff with NO command dump has nothing to clean: the
    /// handoff delegates to the sweep and returns `None`; the research row and
    /// the dump-less folder are gone.
    #[tokio::test]
    async fn cancel_handoff_without_dump_sweeps() {
        let run =
            Seeded::research("research_handoff_no_dump_1", Folder::Scratch, Report::None).await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_handoff_no_dump");

        assert!(
            hand_off_rows(run.id, &ws).await.unwrap().is_none(),
            "no dump → no cleanup prompt"
        );
        assert_eq!(run.rows("jobs").await, 0, "research row swept");
        assert!(!run.root.exists(), "dump-less folder released");
    }

    /// The GUI/abandon path for a run with NO live orchestrator: firing the
    /// cancel signal sweeps the durable rows and the dump-less folder directly.
    #[tokio::test]
    async fn cancel_research_run_sweeps_unregistered_run() {
        let run = Seeded::research(
            "research_cancel_unregistered_1",
            Folder::Scratch,
            Report::None,
        )
        .await;

        // No register() → no orchestrator is alive to observe the signal.
        cancel_research_run(run.id).await;

        assert_eq!(
            run.rows("jobs").await,
            0,
            "research row swept via the unregistered path"
        );
        assert!(!run.root.exists(), "dump-less folder released");
    }

    /// The sweep is idempotent; a double release (racing cleanup tail) is safe.
    #[tokio::test]
    async fn sweep_is_idempotent_and_double_release_safe() {
        crate::util::test::init_management_test_stores().await;
        let job_id = "research_cancel_double_1";
        sweep_cancelled_run(job_id).await.unwrap();
        // Second pass with nothing present must not fail.
        sweep_cancelled_run(job_id).await.unwrap();
    }

    /// The in-tx cancel gate: a completion racing a cancel must roll back —
    /// no pending row may survive, and the jobs row stays for the sweep.
    #[tokio::test]
    async fn complete_durable_job_rolls_back_when_run_cancelled() {
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
            None,
            None,
        )
        .await
        .unwrap();
        let _guard = crate::round_cancel::register(job_id, None);
        crate::round_cancel::cancel(job_id);
        let envelope = crate::jobs::complete_durable_job(
            job_id,
            "report".to_string(),
            crate::agent::message_router::MessageKind::ResearchResult,
            crate::Role::Assistant,
            "caller-user",
            "telegram",
            &ws.name,
            None,
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
