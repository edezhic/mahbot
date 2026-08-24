//! Research-run cleanup: per-run temp folders, results archive, the raw
//! shell-command dump, and the Sanitation-agent cleanup dispatch.
//!
//! ## Run folders
//!
//! Every deep-research run gets a per-run folder under
//! `<temp_dir()>/mahbot-research/{job_id}` — inside the readonly-shell's
//! allowed temp roots, so analysts can write scratch files there. The folder
//! is created idempotently at dispatch AND boot resume (a resume after an OS
//! temp cleanup recreates it; lost prototypes are fail-open). The run's
//! ephemeral search tracker also lives inside the folder (see
//! `crate::search_engine`), so everything temporary dies with it.
//!
//! The folder is removed ONLY by the run's own completion flow: the cleanup
//! tail ([`run_cleanup_agent_and_finish`]) releases the search-engine state,
//! deletes the whole folder, then terminalizes the cleanup row. Folders of
//! crashed runs are left for the OS — the daemon builds no mechanisms for
//! crash leftovers, so there are no boot or periodic run-folder sweeps. The
//! only periodic temp sweep (the temp-dir cleaner, see `crate::temp`)
//! protects research run folders by prompt instruction — the daemon builds
//! no programmatic run-folder sweeps.
//!
//! ## Command dump + Sanitation cleanup
//!
//! At terminalization the run's accumulated raw shell commands (UNFILTERED —
//! no zone classification; attribution is the cleanup agent's job) are written
//! as a command dump file INSIDE the run's per-run folder. At the same time a
//! Sanitation-role agent is dispatched with a dedicated task prompt
//! (see `src/prompt/research/cleanup.md`) to clean artifacts attributable to
//! the run OUTSIDE the folder. Its run is recorded as a durable
//! `research_cleanup` jobs row (the row holds the folder until the cleanup
//! completes and survives a crash for boot resume — durability is a separate
//! mechanism from filesystem reclamation). The agent's final response is
//! logged for observability; there is no report archive.
//!
//! ## Artist media sweep
//!
//! `sweep_media` deletes generated/uploads files in userspaces that no Artist
//! session mentions (keep-detection is strictly session-based — solution 1).
//! It is orthogonal to research-run cleanup and stays in the periodic
//! cleanup loop.

use crate::Workspace;
use crate::config;
use crate::db::params;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ── Constants ─────────────────────────────────────────────────────────────

/// Command-dump cap (bytes, soft — newest-wins): the raw unfiltered shell
/// command history of a run, written at terminalization inside the run folder.
/// The dump is intent for the Sanitation cleanup agent, not a report body, so
/// it can be much larger than the old 32 KiB cleaner-ticket cap.
pub(crate) const COMMAND_DUMP_CAP_BYTES: usize = 10 * 1024 * 1024;
/// Per-tick artist-session scan budget (bytes of session content). Typical
/// bases (~3 MB) fit in one tick; pathological growth is cut across ticks.
const MEDIA_SCAN_BUDGET_BYTES: usize = 10 * 1024 * 1024;
/// Video extensions for case-insensitive keep-matching.
const MEDIA_VIDEO_EXTS: &[&str] = &[".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"];

// ── Run-root helpers ──────────────────────────────────────────────────────

/// Base directory holding all per-run folders. Under the daemon's private temp
/// root (after the TMPDIR pin, `temp_dir()` IS the root) — inside the
/// readonly-shell's allowed temp roots, so analysts can write scratch there.
#[must_use]
pub(crate) fn research_root_base() -> PathBuf {
    std::env::temp_dir().join("mahbot-research")
}

/// Absolute per-run folder path for a job.
#[must_use]
pub(crate) fn run_root_path(job_id: &str) -> PathBuf {
    research_root_base().join(job_id)
}

/// Create the per-run folder (idempotent — never delete+recreate: a boot
/// resume must keep whatever survived). Returns the CANONICAL absolute path
/// (the workspace path is canonicalized, and the Sanitation agent compares
/// paths against the same canonical form).
pub(crate) async fn ensure_run_root(job_id: &str) -> PathBuf {
    let path = run_root_path(job_id);
    if let Err(e) = tokio::fs::create_dir_all(&path).await {
        tracing::warn!(job = %job_id, error = %e, "Failed to create run root — analyst scratch writes may fail");
    }
    tokio::fs::canonicalize(&path).await.unwrap_or(path)
}

/// Does the run's per-run folder exist? The boot-replay crash-window
/// classifier: the cleanup tail removes the folder when the cleanup
/// completes, so folder absence on the replay path means the cleanup already
/// ran in a previous lifetime (see [`dispatch_cleanup_for_pending_envelope`]).
///
/// Fail-closed by design (`unwrap_or(false)`): a transient IO error on the
/// replay path skips row recreation, so a crash-window cleanup never runs and
/// its outside-folder scratch leaks to the OS temp sweep — the safe direction.
/// The alternative (fail-open on error) would re-dispatch a cleanup LLM round
/// for a run that ALREADY completed, per boot, until the envelope is delivered
/// or the row ages into the 8h purge (`crate::jobs::PURGE_CUTOFF_HOURS`) — a
/// bounded but avoidable cost for a transient read error that is far more
/// likely than a genuine crash-window hit.
async fn run_folder_exists(job_id: &str) -> bool {
    tokio::fs::try_exists(run_root_path(job_id))
        .await
        .unwrap_or(false)
}

/// Release a run's per-run folder and its search-engine state — the SOLE
/// run-folder deleter now that the sweeps are gone. Search-engine registry
/// entry first (drops the picker + LMDB tracker handles), then the whole
/// folder (the ephemeral search tracker lives inside it and dies with it).
///
/// Called by the cleanup tail ([`run_cleanup_agent_and_finish`], folder
/// before row terminalize). The cleanup jobs row is NOT touched here:
/// callers own row removal/aging.
///
/// Removal failure is swallowed (the folder is "left for the OS") and the row
/// is still terminalized by the caller — the crash-window classifier then
/// self-heals: if the completion envelope stayed undelivered, the next boot's
/// replay sees the surviving folder and re-dispatches the cleanup, which
/// retries the removal; if the envelope was delivered, the folder leaks to the
/// OS temp sweep only (the accepted crash-class edge — no retry machinery).
pub(crate) async fn release_run_folder(job_id: &str) {
    if crate::search_engine::registry_initialized() {
        crate::search_engine::remove_engine(job_id);
    }
    let path = run_root_path(job_id);
    match tokio::fs::remove_dir_all(&path).await {
        Ok(()) => {}
        // The tracker/folder only exists when the run actually created it.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(job = %job_id, error = %e, "Run-folder release: folder removal failed — left for the OS");
        }
    }
}

// ── Results archive (results.md) ──────────────────────────────────────────

/// Write the run's archived result to `<storage_root>/research/results/{run_id}.md`
/// (question + delivered result, including partial terminalizations).
/// Overwrites idempotently on resume; a failed write is fail-open (the
/// terminalization is never blocked on the archive). The storage root comes
/// from CONFIG when set (test stores point it at a temp dir — no test writes
/// into the real `~/.mahbot`), falling back to the default config dir.
pub(crate) async fn write_results_md(job_id: &str, question: &str, result: &str) {
    let root = config::CONFIG
        .try_storage_root()
        .or_else(|| config::default_config_dir().ok());
    let Some(root) = root else {
        tracing::warn!(job = %job_id, "results.md skipped — no config dir");
        return;
    };
    let dir = root.join("research").join("results");
    let path = dir.join(format!("{job_id}.md"));
    let content =
        format!("# Research {job_id}\n\n## Question\n\n{question}\n\n## Result\n\n{result}\n");
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        tracing::warn!(job = %job_id, error = %e, "results.md: failed to create archive dir");
    } else if let Err(e) = tokio::fs::write(&path, content).await {
        tracing::warn!(job = %job_id, error = %e, "Failed to write results.md — run result not archived");
    }
}

// ── Raw shell-command capture ────────────────────────────────────────────

/// Extract every shell tool-call command from a session history. The canonical
/// [`decode_native_history_message`](crate::session::decode_native_history_message)
/// already un-nests provider-wrapped JSON arguments — read them directly. Never
/// fails: unreadable sessions contribute nothing (fail-open).
fn commands_from_history(history: &[crate::ChatMessage]) -> Vec<String> {
    let mut out = Vec::new();
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
        for call in calls {
            // The persisted history holds the RAW model call — normalize the
            // same way execution does. Gate on the NAME first (no arg clone
            // for the common non-shell calls); shell calls then pay for the
            // arg-key remap (`cmd`/`script` → `command` — a name-only match
            // finds no `command` key in the raw args).
            if crate::tools::normalize_tool_name(&call.name) != "shell" {
                continue;
            }
            let (_, args) = crate::tools::normalize_tool_call(&call.name, call.arguments.clone());
            if let Some(cmd) = args.get("command").and_then(serde_json::Value::as_str) {
                out.push(cmd.to_string());
            }
        }
    }
    out
}

/// Collect ALL shell commands from the persisted sessions of the given agents
/// (incremental stage collection — early sessions of long runs are TTL'd, so
/// the research orchestrator collects right after each round). UNFILTERED:
/// no zone classification, no write-intent patterns — the raw history IS the
/// dump; attribution is the Sanitation cleanup agent's job (filesystem
/// enumeration as fact + this dump as intent).
///
/// Credentials are scrubbed AT COLLECTION (not only at dump-write): the
/// in-memory `commands`/`seen_commands` must match the scrubbed dump form, or
/// a credential-bearing command re-collected after a crash-resume would be
/// treated as distinct from its scrubbed dump form and both variants would
/// accumulate. Multi-line commands don't round-trip through the line-oriented
/// dump (a documented minor fidelity loss — the cap bounds the growth).
pub(crate) async fn collect_agent_shell_commands(agent_ids: &[String]) -> Vec<String> {
    let store = crate::session::store();
    let mut out = Vec::new();
    for id in agent_ids {
        let history = store.load(id).await;
        out.extend(
            commands_from_history(&history)
                .into_iter()
                .map(|c| crate::util::scrub_credentials(&c)),
        );
    }
    out
}

/// Deduplicate while preserving order, keeping the LAST occurrence of each
/// command (newest wins when a command repeats across rounds).
fn dedup_newest_wins(commands: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::with_capacity(commands.len());
    let mut out = Vec::with_capacity(commands.len());
    for cmd in commands.iter().rev() {
        if seen.insert(cmd.clone()) {
            out.push(cmd.clone());
        }
    }
    out.reverse();
    out
}

/// Cap a command list to [`COMMAND_DUMP_CAP_BYTES`], newest wins (drop the
/// oldest commands). A single command larger than the cap is dropped ON ITS
/// OWN — it could never fit the dump, and dropping the older fitting commands
/// for it would lose the only evidence that could be reported.
///
/// THE canonical cap helper: `ResearchState::capture_round` (in-memory bound)
/// and `write_command_dump` (file bound) both call it, so the drop order can
/// never diverge between the two surfaces.
pub(crate) fn cap_command_dump(commands: &mut Vec<String>, cap: usize) {
    // Drop oversized commands wherever they are (they can never fit), keeping
    // the rest in order.
    commands.retain(|c| c.len() <= cap);
    let mut total: usize = commands.iter().map(String::len).sum();
    let mut drop = 0usize;
    while drop < commands.len() && total > cap {
        total -= commands[drop].len();
        drop += 1;
    }
    if drop > 0 {
        commands.drain(..drop);
    }
}

/// Write the raw command dump file INSIDE the run's per-run folder. The
/// commands are already credential-scrubbed at collection (see
/// [`collect_agent_shell_commands`] — the in-memory capture and the dump must
/// match, or a crash-resume would accumulate both variants). Newest-wins,
/// capped at [`COMMAND_DUMP_CAP_BYTES`].
pub(crate) async fn write_command_dump(run_root: &Path, commands: &[String]) {
    let path = run_root.join("commands.dump");
    let mut deduped = dedup_newest_wins(commands);
    cap_command_dump(&mut deduped, COMMAND_DUMP_CAP_BYTES);
    let content = deduped.join("\n") + "\n";
    if let Err(e) = tokio::fs::write(&path, content).await {
        tracing::warn!(error = %e, "Failed to write command dump — cleanup intent degraded");
    }
}

/// Read the command dump file back from the run folder. Crash-resume hook:
/// `ResearchState.commands` is serde-skip so it loads empty after a crash;
/// re-seeding from this file means the first post-resume capture MERGES the
/// pre-crash history instead of overwriting it (early-round sessions are
/// already TTL'd — the dump is the only surviving capture). Missing/unreadable
/// dump → empty (fresh run or OS-swept folder — fail-open).
pub(crate) async fn read_command_dump(run_root: &Path) -> Vec<String> {
    let path = run_root.join("commands.dump");
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => content.lines().map(str::to_string).collect(),
        Err(_) => Vec::new(),
    }
}

// ── Sanitation cleanup dispatch ──────────────────────────────────────────

/// Dedicated task prompt for the research-cleanup Sanitation agent.
const CLEANUP_PROMPT_KEY: &str = "research/cleanup.md";

/// Build the cleanup task prompt: run id, per-run folder path, command dump
/// path, workspace. The boundaries (never touch other runs' folders, the
/// workspace repo, the live service, or other agents' active files; when in
/// doubt, leave it) live in the prompt asset.
pub(crate) fn build_cleanup_prompt(
    job_id: &str,
    run_root: &Path,
    dump_path: &Path,
    ws: &Workspace,
) -> String {
    crate::prompt::substitute(
        &crate::prompt::load_prompt(CLEANUP_PROMPT_KEY),
        &[
            ("{{run_id}}", job_id),
            ("{{run_folder}}", &run_root.to_string_lossy()),
            ("{{dump_path}}", &dump_path.to_string_lossy()),
            ("{{workspace}}", &ws.name),
        ],
    )
}

/// Dedup marker / folder-hold query: does a `research_cleanup` jobs row for
/// this run exist? Shared by the dispatch dedup, the boot-replay crash-window
/// hook, and tests (which pre-create the row to assert dedup without running
/// the agent).
pub(crate) async fn research_cleanup_row_exists(
    conn: &crate::db::Connection,
    job_id: &str,
) -> Result<bool> {
    Ok(conn
        .query_optional(
            "SELECT 1 FROM jobs WHERE id = ?1 AND kind = 'research_cleanup'",
            params![job_id],
            |_| Ok::<(), anyhow::Error>(()),
        )
        .await?
        .is_some())
}

/// Boot-replay crash-window hook: a pending research-completion envelope with
/// NO `research_cleanup` jobs row means the daemon crashed between
/// `complete_durable_job` and `dispatch_research_cleanup` — the cleanup agent
/// was never dispatched. Creates the durable row ONLY (the dedup marker +
/// folder-hold); the `research_cleanup` boot-scan arm — which runs AFTER the
/// envelope replay in `recover_from_restart` — is the SOLE dispatcher for the
/// actual agent (it sees the just-created row and resumes it, and
/// `resume_research_cleanup` runs + terminalizes exactly like a fresh
/// dispatch). Spawning the agent here would DOUBLE-run the same
/// `cleanup_{run_id}` agent on this boot: the scan arm would register the same
/// id via AgentRegistry, REPLACING and CANCELING this dispatch.
///
/// Replay discriminator = run-folder existence (a crash-window classifier on
/// this replay path only, NOT a dispatch condition — fresh dispatch stays
/// unconditional): the cleanup tail removes the run folder when the cleanup
/// completes, so a folder still present here means the cleanup never ran
/// (crash window) → recreate the row; a folder already gone means the cleanup
/// COMPLETED in a previous lifetime while the envelope stayed undelivered
/// (dead target session; envelopes are never purged) → skip, preventing a
/// per-boot duplicate cleanup LLM round for completed-but-undelivered runs.
///
/// Called from `recover_from_restart`'s pending-envelope replay, BEFORE the
/// envelope is routed.
pub(crate) async fn dispatch_cleanup_for_pending_envelope(
    job_id: &str,
    envelope: &crate::agent::message_router::AgentJob,
) {
    let Ok(Some(ws)) = crate::workspace::store()
        .get_by_name(&envelope.workspace_name)
        .await
    else {
        tracing::warn!(
            job = %job_id,
            workspace = %envelope.workspace_name,
            "Cleanup dispatch for pending envelope: workspace unresolvable"
        );
        return;
    };
    if !run_folder_exists(job_id).await {
        tracing::debug!(
            job = %job_id,
            "Run folder absent — cleanup completed in a previous lifetime; envelope replay only"
        );
        return;
    }
    if let Err(e) = create_cleanup_job_row(job_id, &ws).await {
        tracing::warn!(job = %job_id, error = %e, "Cleanup row creation for pending envelope failed");
    }
}

/// The cleanup Sanitation agent id for a run. Single builder — the
/// transient-prefix invariant test in session/mod.rs asserts against it, so a
/// format change is caught by tests instead of silently leaking sessions.
/// Shared by ALL transient `cleanup_` Sanitation agents: the research-run
/// cleanup AND the periodic temp-dir cleaner (`crate::temp`), so the
/// transient-prefix invariant test covers both.
#[must_use]
pub(crate) fn cleanup_agent_id(job_id: &str) -> String {
    format!("cleanup_{job_id}")
}

/// Create the durable `research_cleanup` jobs row for a run (dedup marker +
/// folder-hold). Shared by the fresh terminalization dispatch, the
/// boot-replay crash-window hook, and tests. Returns `Ok(Some(prompt))` when
/// the row was created (the caller spawns the agent with that prompt), or
/// `Ok(None)` when the row already exists (deduped — the boot-scan arm resumes
/// the surviving row). Idempotent.
pub(crate) async fn create_cleanup_job_row(job_id: &str, ws: &Workspace) -> Result<Option<String>> {
    // Manual-cancel gate: a cancelled run must never get a cleanup agent —
    // the cancel sweep owns the folder deletion. The research job row is
    // already gone at this point (cancelled runs are swept), so this is a
    // defensive guard for a racing terminalize.
    if crate::research_cancel::is_cancelled(job_id) {
        tracing::info!(job = %job_id, "Research cleanup suppressed by manual cancel");
        return Ok(None);
    }
    let conn = &crate::session::store().conn;
    // The row itself is the dedup marker: a surviving row means the cleanup is
    // already in flight (crash between dispatch and completion) — no second
    // agent is dispatched; the boot scan resumes the surviving row. The
    // invariant: cleanup row exists ⟺ cleanup not finished.
    if research_cleanup_row_exists(conn, job_id).await? {
        tracing::info!(job = %job_id, "Research cleanup already dispatched — skipping");
        return Ok(None);
    }

    let run_root = ensure_run_root(job_id).await;
    // `ensure_run_root` is idempotent (create_dir_all), so on the replay path
    // — where `dispatch_cleanup_for_pending_envelope` just verified the folder
    // exists — this is a no-op. It stays as a defensive recreate for the
    // fresh path: an OS temp sweep between research terminalization and the
    // cleanup dispatch would otherwise hand the agent a prompt pointing at a
    // missing folder.
    let dump_path = run_root.join("commands.dump");
    let prompt = build_cleanup_prompt(job_id, &run_root, &dump_path, ws);

    // Durable jobs row: id == run_id (folder name) → the row holds the folder
    // for the duration of the cleanup run. The row also makes the cleanup
    // boot-resumable (recover_from_restart arm for kind "research_cleanup")
    // and its failure visible (jobs table status).
    crate::jobs::spawn_job(
        conn,
        job_id,
        &prompt,
        &ws.name,
        "",
        "",
        crate::Role::Sanitation,
        &[crate::jobs::NewAgent {
            agent_id: cleanup_agent_id(job_id),
            kind: crate::jobs::AgentKind::Sanitation,
            idx: None,
            task: prompt.clone(),
        }],
        &crate::jobs::SpawnChild::ResearchCleanup,
    )
    .await
    .map_err(|e| {
        tracing::error!(job = %job_id, error = %e, "Failed to spawn research cleanup job");
        e
    })?;
    Ok(Some(prompt))
}

/// Dispatch the Sanitation-role cleanup agent for a terminalized research run.
///
/// Called at terminalization (all three terminalization points funnel through
/// the terminalize_research / write_terminalization_artifacts tail). The
/// cleanup runs FIRE-AND-FORGET (result delivery is never delayed); the run
/// folder is held by the durable `research_cleanup` jobs row (id == run_id —
/// the folder name) until the cleanup completes. Dispatch is UNCONDITIONAL:
/// research completed + command dump created ⇒ cleanup runs. The row-exists
/// dedup inside `create_cleanup_job_row` covers the crash window (a resumed
/// run terminalizes again while its cleanup row survives).
///
/// `question` is the research question/task text, threaded observationally to
/// the cleanup agent so the Running Agents view keeps the group's question
/// header during the cleanup window (all other run members have deregistered
/// by then). Purely presentational — never affects cleanup behavior.
///
/// Delete capability: Role::Sanitation's standard toolset (Read / Shell
/// ReadOnly) already permits rm/mv/cp under the allowed temp roots —
/// the readonly guard's TEMP_MUTATORS gate on the path, not the role. The
/// cleanup agent therefore needs NO custom toolset.
pub(crate) async fn dispatch_research_cleanup(
    job_id: &str,
    question: &str,
    ws: &Workspace,
) -> Result<()> {
    let Some(prompt) = create_cleanup_job_row(job_id, ws).await? else {
        return Ok(());
    };

    // Fire-and-forget: never block result delivery on the cleanup LLM round.
    // Deliberately a raw tokio::spawn, NOT spawn_cancellable: the cleanup is
    // not a background daemon task. NOTE: a raw spawn task does NOT survive
    // process shutdown (it is dropped with the runtime) — the durability is
    // the jobs row, which the boot scan resumes on the next start after a
    // HARD crash (the drain ignores unregistered agents; if the process dies
    // mid-run the row is the resume point and the folder stays held). A
    // graceful-drain abort is different: `run_cleanup_agent_and_finish`
    // terminalizes the row on EVERY exit path, so a drain-aborted cleanup is
    // terminalized with its outside-folder scratch never cleaned and never
    // retried — matching the pinned "terminalize on every exit path"
    // decision; only hard crashes get the boot-resume retry.
    let ws = ws.clone();
    let job_id_log = job_id.to_string();
    let agent_id = cleanup_agent_id(job_id);
    let agent_id_log = agent_id.clone();
    let job_id = job_id.to_string();
    let question = question.to_string();
    tokio::spawn(async move {
        run_cleanup_agent_and_finish(&job_id, Some(&question), &ws, &prompt).await;
    });

    tracing::info!(job = %job_id_log, agent = %agent_id_log, "Research cleanup dispatched");
    Ok(())
}

/// Run the cleanup agent and finish: log the outcome and release the run
/// folder — folder first, row last — on EVERY exit path. The shared tail of
/// the fresh dispatch and the boot-resume path — a divergence here would
/// silently change one path's folder-release/terminalize behavior.
///
/// `question` is the research question for the Running Agents group label
/// (fresh dispatch always has it; the boot-resume path only stores the
/// cleanup prompt, so it passes `None` and the header degrades to the generic
/// label — presentational only).
async fn run_cleanup_agent_and_finish(
    job_id: &str,
    question: Option<&str>,
    ws: &Workspace,
    prompt: &str,
) {
    let agent_id = cleanup_agent_id(job_id);
    let (agent, response) = crate::agent::run_default_agent(
        &agent_id,
        crate::Role::Sanitation,
        ws,
        prompt,
        None,
        Some(crate::agent::registry::ParentKey::Research(
            job_id.to_string(),
        )),
        question.map(str::to_string),
    )
    .await;
    // The cleaner's final response is logged for observability (no archive
    // append, no cleanup_report.md, no report consumer); on failure it is a
    // visible failure signal in the log. A manual-cancel kill of the run is
    // not a failure — the tail still releases the folder (double-release
    // safe) and terminalizes the (already deleted) row (no-op).
    let report = if agent.is_cancelled() {
        "cancelled (manual research-run cancel)".to_string()
    } else {
        response.unwrap_or_else(|| {
            format!(
                "Research cleanup FAILED (job {job_id}): {}",
                agent
                    .failure
                    .clone()
                    .unwrap_or_else(|| "no failure detail".to_string())
            )
        })
    };
    tracing::info!(
        job = %job_id,
        agent = %agent_id,
        "Research cleanup finished: {}",
        crate::util::scrub_credentials(&report)
    );
    // Release the folder — the SOLE run-folder deleter (no sweeps exist):
    // search-engine state, then the whole folder, then the row. Folder first,
    // row last: a crash between folder removal and row terminalize leaves
    // row + no folder, which boot resume converges from with one extra round.
    // Invariant: cleanup row exists ⟺ cleanup not finished.
    release_run_folder(job_id).await;
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

/// Boot-scan resume of a `research_cleanup` job: re-dispatch the Sanitation
/// agent with the stored task (the row survived a crash mid-cleanup). The
/// folder is still held by the jobs row (recreated empty if the crash was in
/// the folder-release window), so the cleanup completes and the folder is
/// released exactly like a fresh dispatch.
pub(crate) async fn resume_research_cleanup(job_id: &str, ws: &Workspace) {
    let Some((caller, _role)) = crate::jobs::resume_job_preamble(
        &crate::session::store().conn,
        job_id,
        "Research cleanup resume",
        "Research cleanup resume",
    )
    .await
    else {
        // The shared preamble returns None either on a drain abort (the row
        // STAYS — the folder stays held for the next boot) or on a missing /
        // unloadable row (the preamble terminalized it). Only the terminalize
        // path removes a research_cleanup row, so release the run folder there
        // to keep the invariant "row removal ⟹ folder released in the same
        // operation" (no sweep is a backstop anymore). The row-existence check
        // is the discriminator (more robust than re-reading the abort flag:
        // a drain that starts between the preamble's guard and here must not
        // suppress the release of an already-removed row). A row that still
        // exists after a failed terminalize keeps its folder for the boot scan.
        if !research_cleanup_row_exists(&crate::session::store().conn, job_id)
            .await
            .unwrap_or(true)
        {
            release_run_folder(job_id).await;
        }
        return;
    };
    let prompt = caller.task.clone();
    // The row stores the cleanup prompt, not the research question — the
    // group header falls back to the generic label on this path.
    run_cleanup_agent_and_finish(job_id, None, ws, &prompt).await;
}

// ── Sweep: artist generated/uploads keep-detection ────────────────────────

/// Per-user keep-scan cursor (in-memory; a restart resets it — safe, because
/// files only become deletion candidates after a full coverage pass within
/// one process lifetime). tokio Mutex: the sweep holds the guard across
/// awaits (DB + fs) and the cleanup-loop future must stay `Send`.
static MEDIA_CURSORS: tokio::sync::Mutex<Option<HashMap<String, MediaCursor>>> =
    tokio::sync::Mutex::const_new(None);

#[derive(Default)]
struct MediaCursor {
    /// Agent id → `session_metadata.last_activity` at scan time. Keyed by
    /// content VERSION, NOT index or bare id: a session scanned on an earlier
    /// tick whose content grew (new file mentions) must be re-scanned before
    /// the deletion pass — deleting against a stale keep-set is the unsafe
    /// direction. An unchanged activity skips the DB re-read entirely (the
    /// per-tick scan budget stays available for other users — no starvation).
    /// `last_activity` is written by [`crate::db::now`] (chrono AutoSi —
    /// fractional-second precision when non-zero), so the practical residual
    /// of an unchanged activity hiding a content append is sub-microsecond —
    /// far narrower than a whole-second window.
    scanned: HashMap<String, String>,
    /// Agent id → stripped content contribution. REPLACED on re-scan, never
    /// re-appended: `last_activity` bumps on every message append, so an
    /// active artist session would otherwise duplicate its whole history into
    /// the keep-set every tick — daemon memory growth plus a premature hit of
    /// the overflow cap that permanently disables the user's sweep.
    session_content: HashMap<String, String>,
    /// Entire session base scanned (files become candidates only then).
    /// Stays true after the deletion pass — the keep-set grows incrementally
    /// and the sweep never forces a full re-scan cycle.
    covered: bool,
    /// Rebuilt from `session_content` after each scan — the keep-set
    /// (paths/basenames are matched against it).
    content: String,
    /// Set when the accumulated content hit the hard cap — the keep-set is
    /// incomplete, so the deletion pass is SKIPPED (keep everything: the safe
    /// direction is never deleting a file whose mention was not scanned).
    overflowed: bool,
}

/// Reset the in-memory media-sweep cursors (tests only — production never
/// needs a reset: a restart clears the process-global anyway).
#[cfg(test)]
pub(crate) async fn reset_media_cursors() {
    if let Some(map) = MEDIA_CURSORS.lock().await.as_mut() {
        map.clear();
    }
}

/// Strip `data:` URI blobs from session content — a data URI is never a file
/// mention. Conservative: on regex trouble the blob stays (over-keep).
fn strip_data_uris(content: &str) -> String {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"data:[A-Za-z0-9+/=:;,._-]+").expect("data URI regex compiles")
    });
    RE.replace_all(content, "").into_owned()
}

/// Recursively enumerate REGULAR files under `dir` (missing dir → empty).
/// Entry-level symlinks are skipped ENTIRELY (no-follow): a symlinked
/// directory planted inside generated/uploads must not be traversed — the
/// deletion pass would otherwise collect (and later delete) real files
/// OUTSIDE the userspace root, violating "never delete a file whose mention
/// was not scanned" at the filesystem level. A top-level generated/uploads
/// dir that is itself a symlink is guarded by the CALLER (`symlink_metadata`
/// before the walk — `read_dir` would resolve it). Runs on the blocking pool
/// — the sweep runs on the async runtime.
async fn list_files(dir: &Path) -> Vec<PathBuf> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in rd.flatten() {
                // `DirEntry::file_type` does NOT follow symlinks — a symlink
                // reports as a symlink, not as its target's type, so both
                // symlinked dirs and files are skipped by the two arms below.
                let Ok(ft) = entry.file_type() else {
                    continue;
                };
                let path = entry.path();
                if ft.is_dir() {
                    walk(&path, out);
                } else if ft.is_file() {
                    out.push(path);
                }
            }
        }
        let mut out = Vec::new();
        walk(&dir, &mut out);
        out
    })
    .await
    .unwrap_or_default()
}

/// Artist sessions for a user: `session_metadata` rows with role=artist and
/// the user's name (agent id + last_activity, most-recent-first — the activity
/// doubles as the content-change signal for the scan cursor). A DB error is
/// returned (NOT swallowed as empty — an empty list would trivially satisfy
/// full coverage and trigger mass deletion against an empty keep-set).
async fn artist_session_ids(user_name: &str) -> anyhow::Result<Vec<(String, String)>> {
    let rows = crate::session::store()
        .conn
        .query(
            "SELECT agent_id, last_activity FROM session_metadata WHERE role = 'artist' \
             AND user_name = ?1 ORDER BY last_activity DESC",
            params![user_name],
        )
        .await?;
    // Fail CLOSED on a malformed row: silently dropping it would let full
    // coverage complete without scanning that session — its file mentions
    // would be missed and its files deleted (the same deletion-safety class
    // as the DB-error fail-open above).
    let mut out = Vec::with_capacity(rows.len());
    for r in &rows {
        let agent_id = r.get::<String>(0)?;
        let last_activity = r.get::<String>(1)?;
        out.push((agent_id, last_activity));
    }
    Ok(out)
}

/// One artist session's message content (fail-open only via the `Result` —
/// a DB read failure must not masquerade as an empty session: the sweep would
/// advance coverage and delete files the unread session mentions). Fail
/// CLOSED on a malformed row (same deletion-safety class as
/// [`artist_session_ids`]: a row failing to decode loses its mentions from
/// the keep-set while the session still counts as scanned).
async fn artist_session_content(agent_id: &str) -> anyhow::Result<String> {
    let rows = crate::session::store()
        .conn
        .query(
            "SELECT content FROM sessions WHERE agent_id = ?1 ORDER BY id ASC",
            params![agent_id],
        )
        .await?;
    let mut out = Vec::with_capacity(rows.len());
    for r in &rows {
        out.push(r.get::<String>(0)?);
    }
    Ok(out.join("\n"))
}

/// Sweep generated/uploads under the userspaces root — production entry.
///
/// Resolves where user workspaces live through `crate::users::userspaces_root`
/// so it always agrees with `personal_workspace_path` (production:
/// `~/.mahbot/userspaces`; tests: the shared test root).
pub async fn sweep_media() -> Result<u64> {
    sweep_media_at(&crate::users::userspaces_root()).await
}

/// Artist-media sweep over an explicit userspaces root (injectable for tests).
pub(crate) async fn sweep_media_at(userspaces_root: &Path) -> Result<u64> {
    sweep_media_at_budgeted(userspaces_root, MEDIA_SCAN_BUDGET_BYTES).await
}

/// Budgeted variant — tests inject a tiny per-tick budget to force the scan
/// across multiple ticks (full-coverage gating).
async fn sweep_media_at_budgeted(userspaces_root: &Path, budget_bytes: usize) -> Result<u64> {
    let mut deleted = 0u64;
    let mut budget = budget_bytes;
    let mut guard = MEDIA_CURSORS.lock().await;
    let cursor_map = guard.get_or_insert_with(HashMap::new);
    let Ok(mut user_dirs) = tokio::fs::read_dir(userspaces_root).await else {
        return Ok(0);
    };
    // Prune cursors of users whose userspace dir vanished — but only after a
    // COMPLETE pass: a budget-exhausted or entry-error tick saw only a prefix
    // of the dirs, so pruning against that partial set would drop live
    // cursors (a re-scan is harmless, but the cursor's coverage state is
    // real work thrown away).
    let mut seen_users: HashSet<String> = HashSet::new();
    let mut complete_pass = true;
    loop {
        let user_entry = match user_dirs.next_entry().await {
            Ok(Some(e)) => e,
            Ok(None) => break,
            // A single entry-read failure (OS temp race) must not abort the
            // whole multi-user tick — skip it, consistent with the per-entry
            // `continue` pattern used for file_type/metadata errors elsewhere.
            Err(e) => {
                complete_pass = false;
                tracing::warn!(error = %e, "Media sweep: read_dir entry failed — skipping");
                continue;
            }
        };
        let user_path = user_entry.path();
        let Ok(file_type) = user_entry.file_type().await else {
            // Transient stat error — this user was never scanned, so the pass
            // is NOT complete: pruning cursors now would drop a live one.
            complete_pass = false;
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Some(user_name) = user_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(std::string::ToString::to_string)
        else {
            // Unrepresentable dir name — never scanned, so the pass is not
            // complete (consistent with the entry-error and file_type-error
            // arms above).
            complete_pass = false;
            continue;
        };
        seen_users.insert(user_name.clone());
        // NOTE: no overflowed early-skip here — sweep_user_media must run so
        // the session-deletion reset (a /clear re-enables an overflowed user)
        // can fire; the overflowed check happens AFTER that reset, inside
        // sweep_user_media (returning before the scan/deletion phases).
        if budget == 0 {
            complete_pass = false;
            break; // budget exhausted — continue on the next tick
        }
        deleted += sweep_user_media(&user_name, &user_path, &mut budget, cursor_map).await;
    }
    if complete_pass {
        cursor_map.retain(|u, _| seen_users.contains(u));
    }
    Ok(deleted)
}

#[expect(clippy::too_many_lines)] // per-phase guards (fail-open, empty-base, overflow, budget, coverage) are sequential and inline
async fn sweep_user_media(
    user_name: &str,
    user_path: &Path,
    budget: &mut usize,
    cursors: &mut HashMap<String, MediaCursor>,
) -> u64 {
    let cursor = cursors.entry(user_name.to_string()).or_default();
    // Fail-open on DB trouble: a transient read failure must not be confused
    // with "no artist sessions" — the latter trivially satisfies coverage and
    // would delete EVERY file. Skip the user for this tick instead.
    let Ok(session_ids) = artist_session_ids(user_name).await else {
        tracing::warn!(user = %user_name, "Media sweep: artist-session query failed — user skipped for this tick");
        return 0;
    };
    // A DELETED session (e.g. /clear) invalidates the accumulated keep-set:
    // its mentions would otherwise keep files forever (accepted consequence
    // "/clear → файлы-кандидаты"). Reset the cursor so the remaining base is
    // re-scanned from scratch — reorders/additions never reset (incremental).
    // The overflowed flag is reset here too: a shrunk base can be re-scanned,
    // so /clear must re-enable an overflowed user's sweep (without this, the
    // flag would be permanently sticky for the process lifetime).
    let ids_now: HashSet<&str> = session_ids.iter().map(|(id, _)| id.as_str()).collect();
    if cursor
        .scanned
        .keys()
        .any(|id| !ids_now.contains(id.as_str()))
    {
        cursor.scanned.clear();
        cursor.session_content.clear();
        cursor.content.clear();
        cursor.covered = false;
        cursor.overflowed = false;
    }
    // No artist sessions → nothing was ever scanned → deleting everything
    // would violate the safe direction ("never delete a file whose mention
    // was not scanned"). Keep the user's files. Deliberate conflict with the
    // accepted "/clear → файлы-кандидаты" consequence: a FULL /clear (zero
    // sessions) is indistinguishable from a brand-new user, so its files are
    // kept too (safe direction wins — never delete on no evidence; the
    // rotation fires only when at least one session remains).
    if session_ids.is_empty() {
        tracing::debug!(user = %user_name, "Media sweep: no artist sessions — files kept");
        return 0;
    }
    // Overflowed keep-set (incomplete evidence): the safe direction is never
    // deleting — skip the scan and deletion phases for this tick. The flag is
    // only cleared by a session deletion above (a /clear re-enables the
    // sweep); a content-shrink without deletion stays disabled.
    if cursor.overflowed {
        return 0;
    }

    // Scan phase: advance through the session base within the tick budget.
    // Sessions whose activity is unchanged since the last scan are skipped
    // WITHOUT a DB re-read — their mentions are already in the keep-set and
    // the budget stays available for other users. `total` tracks the keep-set
    // size incrementally (the content rebuild is deferred below, so
    // `cursor.content.len()` would be stale across multiple changes in one
    // tick).
    let mut total = cursor.content.len();
    let mut changed = false;
    for (id, activity) in &session_ids {
        if cursor
            .scanned
            .get(id)
            .is_some_and(|saved| saved == activity)
        {
            continue;
        }
        if *budget == 0 {
            break;
        }
        let Ok(content) = artist_session_content(id).await else {
            tracing::warn!(user = %user_name, "Media sweep: session read failed — user skipped for this tick");
            return 0;
        };
        *budget = budget.saturating_sub(content.len());
        let stripped = strip_data_uris(&content);
        // REPLACE this session's prior contribution (a grown re-scan must not
        // re-append its whole history — last_activity bumps per append, so an
        // active session would duplicate its mentions every tick and blow the
        // cap without a real change to the keep-set).
        let prior = cursor.session_content.get(id).map_or(0, String::len);
        let new_total = total - prior + stripped.len();
        if new_total > MEDIA_SCAN_BUDGET_BYTES * 4 {
            // Hard cap on the accumulated keep-set. The set is now
            // incomplete — mark overflowed so this user's sweep is
            // disabled (dropping old content could delete mentioned
            // files — the unsafe direction). Record the overflowing
            // session in `scanned` too: a later /clear of it must
            // trigger the deletion-reset (which clears `overflowed`),
            // otherwise the flag stays permanently sticky.
            cursor.overflowed = true;
            cursor.scanned.insert(id.clone(), activity.clone());
            break;
        }
        total = new_total;
        cursor.session_content.insert(id.clone(), stripped);
        cursor.scanned.insert(id.clone(), activity.clone());
        changed = true;
    }
    // Rebuild the keep-set ONCE after the scan loop (a per-session rebuild is
    // O(total) per change — quadratic across many changed sessions; bounded by
    // the overflow cap but still a needless copy every tick).
    if changed {
        cursor.content = cursor
            .session_content
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
    }
    // Full coverage only when EVERY session's stored activity matches the
    // current one — a session whose content grew but was not yet re-scanned
    // (tick budget ran out) keeps `covered` false: the deletion pass must
    // never run against a stale keep-set.
    cursor.covered = session_ids
        .iter()
        .all(|(id, activity)| cursor.scanned.get(id) == Some(activity));

    // Deletion phase: only after FULL coverage AND a complete keep-set. The
    // cursor is NOT reset after the pass — the keep-set grows incrementally
    // (new/changed sessions extend it), so the full session base is never
    // re-scanned per cycle and later users are never starved of the budget.
    // Standing cost (design-accepted): the deletion pass itself is unbudgeted
    // — it re-lists every generated/uploads file and runs one `contains` per
    // file every 10-min tick for every covered user (~seconds of CPU for
    // heavy users near the cap; tracked at rollout).
    let mut deleted = 0u64;
    if cursor.covered && !cursor.overflowed {
        // Basename uniqueness across this user's generated+uploads UNION
        // (exact, case-sensitive) — the only path for basename matching.
        let mut base_counts: HashMap<String, usize> = HashMap::new();
        let mut files: Vec<PathBuf> = Vec::new();
        for dir in ["generated", "uploads"] {
            let d = user_path.join(dir);
            // `symlink_metadata` (no-follow): a top-level generated/uploads
            // that is ITSELF a symlink must not be traversed — `read_dir`
            // resolves the top-level path, so the per-entry no-follow inside
            // `list_files` cannot catch it (files outside the userspace root
            // could be collected and deleted).
            let Ok(md) = tokio::fs::symlink_metadata(&d).await else {
                continue;
            };
            if md.file_type().is_symlink() {
                continue;
            }
            for f in list_files(&d).await {
                if let Some(b) = f.file_name().and_then(|n| n.to_str()) {
                    *base_counts.entry(b.to_string()).or_default() += 1;
                }
                files.push(f);
            }
        }
        // Lowercase the keep-set ONCE — the video case-insensitive fallback
        // reuses it for every file (no O(files × content) re-allocation).
        let content_lower = cursor.content.to_ascii_lowercase();
        for f in files {
            if is_mentioned(&f, &cursor.content, &content_lower, &base_counts) {
                continue;
            }
            match tokio::fs::remove_file(&f).await {
                Ok(()) => deleted += 1,
                Err(e) => {
                    tracing::warn!(file = %f.display(), error = %e, "Media sweep: delete failed");
                }
            }
        }
    }
    deleted
}

/// Keep-detection for one file: mentioned by basename when that basename is
/// unique in the user's generated+uploads union. An AMBIGUOUS basename
/// (duplicate across the union) is KEPT — the design's safe direction
/// ("иначе файл сохраняется — пере-держать"): a bare-basename mention cannot
/// be attributed to one of the duplicates, so deleting either could destroy a
/// mentioned file. Video extensions match case-insensitively (`content_lower`
/// is the precomputed lowercase keep-set — shared by every file).
/// Absolute-path mentions need no separate check: any path mention contains
/// the file's basename (subsumed by the basename check), and ambiguous
/// basenames are always kept by the rule above.
fn is_mentioned(
    file: &Path,
    content: &str,
    content_lower: &str,
    base_counts: &HashMap<String, usize>,
) -> bool {
    let Some(base) = file.file_name().and_then(|n| n.to_str()) else {
        return true; // unrepresentable name — keep (safe)
    };
    if base_counts.get(base) != Some(&1) {
        return true; // ambiguous basename — keep (over-keep is the safe direction)
    }
    if content.contains(base) {
        return true;
    }
    // Case-insensitive fallback for video extensions (suffixes are already
    // dotted — no per-file format! allocation).
    let lower_base = base.to_ascii_lowercase();
    MEDIA_VIDEO_EXTS.iter().any(|e| lower_base.ends_with(e)) && content_lower.contains(&lower_base)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Open the sessions/board stores in a temp dir for sweep tests.
    async fn init_stores() {
        crate::util::test::init_management_test_stores().await;
    }

    /// Media-sweep fixture: fresh stores + a temp userspaces root with the
    /// user's generated/uploads dirs created. Returns the userspaces root.
    async fn media_fixture(user: &str) -> tempfile::TempDir {
        init_stores().await;
        let userspaces = tempfile::tempdir().unwrap();
        for sub in ["generated", "uploads"] {
            tokio::fs::create_dir_all(userspaces.path().join(user).join(sub))
                .await
                .unwrap();
        }
        userspaces
    }

    /// Write placeholder files under `userspaces/{user}/generated`.
    async fn write_gen_files(userspaces: &Path, user: &str, names: &[&str]) -> Vec<PathBuf> {
        let gdir = userspaces.join(user).join("generated");
        let mut out = Vec::with_capacity(names.len());
        for n in names {
            let p = gdir.join(n);
            tokio::fs::write(&p, "x").await.unwrap();
            out.push(p);
        }
        out
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_keeps_mentioned_and_removes_unmentioned_after_full_scan() {
        let userspaces = media_fixture("alice").await;
        let up = userspaces.path().join("alice").join("uploads");
        let files =
            write_gen_files(userspaces.path(), "alice", &["image_1.png", "image_2.png"]).await;
        let mentioned = &files[0];
        let orphan = &files[1];
        let upload_orphan = up.join("photo_1.jpg");
        tokio::fs::write(&upload_orphan, "x").await.unwrap();

        // One artist session mentioning the first file by absolute path.
        let session = format!("[IMAGE:{}]", mentioned.canonicalize().unwrap().display());
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO session_metadata (agent_id, last_activity, user_name, workspace_name, role) \
             VALUES ('artist_a', ?1, 'alice', 'personal:alice', 'artist')",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) VALUES ('artist_a', 'assistant', ?1, ?2)",
            params![session, now],
        )
        .await
        .unwrap();

        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 2, "both unmentioned files deleted");
        assert!(mentioned.exists(), "mentioned file kept");
        assert!(!orphan.exists());
        assert!(!upload_orphan.exists());

        // A second pass after a NEW artist-session mention: the freshly
        // mentioned orphan stays; the still-unmentioned image_3 is deleted.
        tokio::fs::write(&orphan, "x").await.unwrap();
        let image3 = files[0].parent().unwrap().join("image_3.png");
        tokio::fs::write(&image3, "x").await.unwrap();
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES ('artist_a', 'assistant', ?1, ?2)",
            params![
                format!("[IMAGE:{}]", orphan.canonicalize().unwrap().display()),
                now
            ],
        )
        .await
        .unwrap();
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "only the unmentioned image_3 deleted");
        assert!(orphan.exists(), "now-mentioned orphan kept");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_deleted_session_rotates_files() {
        let userspaces = media_fixture("clr").await;
        let files = write_gen_files(userspaces.path(), "clr", &["f_a.png", "f_b.png"]).await;
        insert_artist_session(
            "artist_clr1",
            "clr",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_clr2",
            "clr",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        // Tick 1: both sessions scanned — both files mentioned, nothing deleted.
        assert_eq!(sweep_media_at(userspaces.path()).await.unwrap(), 0);
        assert!(files[0].exists());
        assert!(files[1].exists());
        // /clear deletes artist_clr1's session: its stale mention must not
        // keep f_a forever — the cursor resets and f_a becomes a candidate.
        crate::session::store()
            .conn
            .execute(
                "DELETE FROM session_metadata WHERE agent_id = 'artist_clr1'",
                (),
            )
            .await
            .unwrap();
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "f_a's only mention was in the cleared session");
        assert!(!files[0].exists());
        assert!(
            files[1].exists(),
            "f_b still mentioned by the surviving session"
        );
    }

    #[test]
    fn command_dump_dedup_newest_wins_and_caps() {
        // Newest-wins dedup: a command repeated across rounds keeps its LAST
        // occurrence (the newest evidence).
        let cmds = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        assert_eq!(dedup_newest_wins(&cmds), vec!["b", "a"]);

        // Under the cap — untouched.
        let mut small = vec!["cat > /tmp/a".to_string(), "cat > /tmp/b".to_string()];
        cap_command_dump(&mut small, 50);
        assert_eq!(small.len(), 2, "under the cap — untouched");

        // Cap: newest commands survive; the oldest are dropped.
        let mut capped = vec!["x".repeat(100), "y".repeat(100), "z".repeat(100)];
        cap_command_dump(&mut capped, 250);
        assert_eq!(capped.len(), 2, "oldest dropped to fit the cap");
        assert_eq!(capped[0], "y".repeat(100));
        assert_eq!(capped[1], "z".repeat(100));

        // A single command larger than the cap can never fit — it is dropped
        // ON ITS OWN (keeping it would lose all older fitting evidence).
        let mut huge = vec!["small".to_string(), "x".repeat(500)];
        cap_command_dump(&mut huge, 400);
        assert_eq!(huge, vec!["small".to_string()]);
    }

    #[tokio::test]
    async fn command_dump_written_matches_scrubbed_collection() {
        let td = tempfile::tempdir().unwrap();
        let run_root = td.path();
        let secret =
            "curl -o /tmp/out.bin \"https://api.example.com/data?api_key=SECRET_API_KEY_123\"";
        // Credentials are scrubbed AT COLLECTION (collect_agent_shell_commands);
        // the dump is written verbatim from that already-scrubbed capture — the
        // two must match, or a crash-resume would accumulate both variants.
        let commands: Vec<String> = [secret, "cat > /tmp/x"]
            .into_iter()
            .map(crate::util::scrub_credentials)
            .collect();
        write_command_dump(run_root, &commands).await;
        let content = tokio::fs::read_to_string(run_root.join("commands.dump"))
            .await
            .unwrap();
        assert!(content.contains("/tmp/x"));
        assert!(
            !content.contains("SECRET_API_KEY_123"),
            "the scrubbed collection form is what lands in the dump"
        );
        assert_eq!(
            content,
            commands.join("\n") + "\n",
            "dump written verbatim from the scrubbed capture"
        );
    }

    #[test]
    fn commands_from_history_normalizes_aliased_calls() {
        // Native assistant payloads with RAW model calls: alias names AND
        // non-canonical arg keys must both be normalized — execution remaps
        // `bash`/`run_terminal_cmd` → `shell` and `cmd`/`script` → `command`,
        // while the persisted history holds the raw model form (this exact
        // bug class cost two review rounds).
        let native = |calls: &str| {
            crate::ChatMessage::assistant(format!(r#"{{"content":"","tool_calls":{calls}}}"#))
        };
        let history = vec![
            native(r#"[{"id":"1","name":"bash","arguments":{"cmd":"cat > /tmp/x"}}]"#),
            native(r#"[{"id":"2","name":"run_terminal_cmd","arguments":{"script":"tee /tmp/y"}}]"#),
            native(r#"[{"id":"3","name":"shell","arguments":{"command":"mkdir -p /tmp/z"}}]"#),
            // Non-shell calls are gated by name before any arg work.
            native(r#"[{"id":"4","name":"search","arguments":{"query":"foo"}}]"#),
            // Provider-wrapped nested-JSON arguments are un-nested too.
            native(r#"[{"id":"5","name":"shell","arguments":"{\"command\":\"touch /tmp/w\"}"}]"#),
            crate::ChatMessage::user("not a tool call"),
        ];
        assert_eq!(
            commands_from_history(&history),
            vec![
                "cat > /tmp/x",
                "tee /tmp/y",
                "mkdir -p /tmp/z",
                "touch /tmp/w",
            ],
            "alias names and arg keys normalized; non-shell calls skipped"
        );
    }

    #[tokio::test]
    async fn cleanup_prompt_builds_with_context() {
        let td = tempfile::tempdir().unwrap();
        let run_root = td.path();
        let dump_path = run_root.join("commands.dump");
        let ws = crate::workspace::test_ws_named(
            run_root.to_str().expect("temp path is utf8"),
            "cleanup_ws",
        );
        let prompt = build_cleanup_prompt("run_abc", run_root, &dump_path, &ws);
        assert!(prompt.contains("run_abc"), "run id substituted");
        assert!(prompt.contains(&run_root.to_string_lossy().to_string()));
        assert!(prompt.contains(&dump_path.to_string_lossy().to_string()));
        assert!(prompt.contains("cleanup_ws"));
        assert!(prompt.contains("Never touch another run's folder"));
    }

    #[tokio::test]
    async fn dispatch_research_cleanup_deduped_by_jobs_row() {
        init_stores().await;
        crate::util::test::create_test_workspace("/tmp/test_ws_cleanup", "test_ws").await;
        let ws = crate::workspace::test_ws_named("/tmp/test_ws_cleanup", "test_ws");
        // Pre-create the cleanup jobs row (what a first dispatch leaves
        // behind). Dispatch must see the row and return BEFORE spawning the
        // agent — this asserts the dedup without running a live LLM agent
        // (a fire-and-forget agent would race the assertions below).
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            "run_dedup",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Sanitation,
            &[],
            &crate::jobs::SpawnChild::ResearchCleanup,
        )
        .await
        .unwrap();
        assert!(
            research_cleanup_row_exists(&crate::session::store().conn, "run_dedup")
                .await
                .unwrap(),
            "the pre-created row is the dedup marker"
        );
        dispatch_research_cleanup("run_dedup", "test question", &ws)
            .await
            .unwrap();
        let rows = crate::session::store()
            .conn
            .query(
                "SELECT COUNT(*) FROM jobs WHERE id = 'run_dedup' AND kind = 'research_cleanup'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(rows[0].get::<i64>(0).unwrap(), 1, "single cleanup job row");
        // No agent was spawned (dispatch returned at the dedup check) — no
        // cleanup session may exist.
        let sessions = crate::session::store()
            .conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = 'cleanup_run_dedup'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            sessions[0].get::<i64>(0).unwrap(),
            0,
            "deduped dispatch must not spawn the cleanup agent"
        );
        // Clean up the spawned job row so the test store stays tidy.
        crate::jobs::terminalize_job(&crate::session::store().conn, "run_dedup")
            .await
            .unwrap();
    }

    /// Insert one artist session (metadata + one message) for a user.
    async fn insert_artist_session(agent_id: &str, user: &str, content: &str) {
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO session_metadata (agent_id, last_activity, user_name, workspace_name, role) \
             VALUES (?1, ?2, ?3, ?4, 'artist')",
            params![agent_id, now.clone(), user, format!("personal:{user}")],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES (?1, 'assistant', ?2, ?3)",
            params![agent_id, content, now],
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_gates_deletion_on_full_coverage() {
        let userspaces = media_fixture("gates").await;
        let files = write_gen_files(
            userspaces.path(),
            "gates",
            &["f_a.png", "f_b.png", "f_c.png"],
        )
        .await;
        // Each session mentions its own file; the per-tick budget (1 byte)
        // fits exactly one session per tick.
        insert_artist_session(
            "artist_g1",
            "gates",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_g2",
            "gates",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;

        // Tick 1: only artist_b (newest) is scanned — f_c is unmentioned but
        // must NOT be deleted until the whole session base is covered.
        let n = sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap();
        assert_eq!(n, 0, "no deletion before full coverage");
        assert!(files[2].exists());

        // Tick 2: the remaining session is scanned, coverage completes, and
        // only then does the deletion pass run.
        let n = sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap();
        assert_eq!(n, 1, "unmentioned f_c deleted after full coverage");
        assert!(files[0].exists());
        assert!(files[1].exists());
        assert!(!files[2].exists());
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_cursor_survives_list_reorder() {
        let userspaces = media_fixture("reorder").await;
        let files = write_gen_files(userspaces.path(), "reorder", &["f_a.png", "f_b.png"]).await;
        insert_artist_session(
            "artist_r1",
            "reorder",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_r2",
            "reorder",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;

        // Tick 1: scans artist_b (newest) only.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0
        );

        // artist_a gets newer activity — the ordered list flips. The cursor is
        // keyed by scanned agent id, so artist_a is still scanned on tick 2
        // (an index cursor would re-scan artist_b and declare coverage with
        // artist_a's mention missing from the keep-set — deleting f_a).
        let conn = &crate::session::store().conn;
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'artist_r1'",
            params![crate::db::now()],
        )
        .await
        .unwrap();
        let n = sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap();
        assert_eq!(n, 0, "both files mentioned — nothing to delete");
        assert!(
            files[0].exists(),
            "f_a mentioned in the re-ordered scan kept"
        );
        assert!(files[1].exists());
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_strips_data_uris() {
        let userspaces = media_fixture("duri").await;
        let files = write_gen_files(userspaces.path(), "duri", &["thumb.png"]).await;
        // The filename appears only INSIDE a data URI — not a file mention.
        insert_artist_session("artist_d1", "duri", "data:image/png;base64,AAAAthumb.png").await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1);
        assert!(
            !files[0].exists(),
            "data-URI-embedded name is not a mention"
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_keep_set_is_per_user() {
        let userspaces = media_fixture("pualice").await;
        let alice_files = write_gen_files(
            userspaces.path(),
            "pualice",
            &["alice_1.png", "alice_2.png"],
        )
        .await;
        let bgen = userspaces.path().join("pubob").join("generated");
        tokio::fs::create_dir_all(&bgen).await.unwrap();
        let bf = bgen.join("bob_pic.png");
        tokio::fs::write(&bf, "x").await.unwrap();
        // bob's sessions mention alice's a1 by absolute path — that must NOT
        // keep alice's files (keep-sets are per-user).
        insert_artist_session("artist_iso1", "pualice", "a log line with no file mentions").await;
        insert_artist_session(
            "artist_iso2",
            "pubob",
            &format!(
                "[IMAGE:{}]",
                alice_files[0].canonicalize().unwrap().display()
            ),
        )
        .await;
        insert_artist_session(
            "artist_iso3",
            "pubob",
            &format!("[IMAGE:{}]", bf.canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(
            n, 2,
            "alice's files deleted — bob's mention is out of zone; bob's own kept"
        );
        assert!(!alice_files[0].exists());
        assert!(!alice_files[1].exists());
        assert!(bf.exists(), "bob's mentioned file kept");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_empty_session_base_keeps_files() {
        let userspaces = media_fixture("guard").await;
        let files = write_gen_files(userspaces.path(), "guard", &["legacy.png"]).await;
        // Zero artist sessions: coverage would be vacuously true and delete
        // every file — but nothing was ever scanned, so nothing is deleted
        // (the safe direction). Brand-new users / legacy uploads are kept.
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 0, "no artist sessions → no deletion");
        assert!(files[0].exists(), "unscanned files are never deleted");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_rescans_grown_session_before_deleting() {
        let userspaces = media_fixture("grow").await;
        let files = write_gen_files(
            userspaces.path(),
            "grow",
            &["f_a.png", "f_b.png", "f_c.png"],
        )
        .await;
        insert_artist_session(
            "artist_grow1",
            "grow",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_grow2",
            "grow",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        // Deterministic scan order: grow2 is the newest session (scanned first).
        let conn = &crate::session::store().conn;
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'artist_grow1'",
            params![(chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339()],
        )
        .await
        .unwrap();
        reset_media_cursors().await;

        // Tick 1 (budget 1 byte): scans the newest session (grow2) only.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0,
            "no deletion before full coverage"
        );
        assert!(files[2].exists());

        // grow2's content GROWS after it was scanned: a new file f_c is
        // mentioned. Its last_activity bumps — the cursor must re-scan it
        // before the deletion pass, or f_c (mentioned only in the new tail)
        // would be deleted against the stale keep-set.
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES ('artist_grow2', 'assistant', ?1, ?2)",
            params![
                format!("[IMAGE:{}]", files[2].canonicalize().unwrap().display()),
                crate::db::now()
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'artist_grow2'",
            params![crate::db::now()],
        )
        .await
        .unwrap();

        // Tick 2 (budget 1 byte): grow2 (changed activity) is re-scanned —
        // its new mention reaches the keep-set; grow1 is not yet scanned so
        // coverage is incomplete and nothing is deleted.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0
        );
        assert!(
            files[2].exists(),
            "mention in the grown session not yet scanned"
        );
        // Re-scan REPLACES the session's prior contribution instead of
        // re-appending its whole history (last_activity bumps per append — an
        // active session would otherwise duplicate its mentions every tick and
        // blow the overflow cap): grow2's old mention appears exactly once.
        let cursors = MEDIA_CURSORS.lock().await;
        let c = cursors
            .as_ref()
            .expect("cursor map initialized")
            .get("grow")
            .expect("grow cursor");
        let fb_mention = format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display());
        assert_eq!(
            c.content.matches(&fb_mention).count(),
            1,
            "re-scan replaced, not re-appended"
        );
        drop(cursors);
        // Tick 3: grow1 is scanned, coverage completes, and f_c is kept.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0,
            "all three files mentioned — nothing to delete"
        );
        assert!(files[0].exists());
        assert!(files[1].exists());
        assert!(
            files[2].exists(),
            "file mentioned in the grown session kept"
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_no_rescan_cycle_does_not_starve_later_users() {
        let userspaces = media_fixture("starve1").await;
        let dirs = ["starve1", "starve2"];
        for u in dirs {
            if u != "starve1" {
                let gdir = userspaces.path().join(u).join("generated");
                tokio::fs::create_dir_all(&gdir).await.unwrap();
            }
            let orphan = format!("{u}_orphan.png");
            write_gen_files(userspaces.path(), u, &[&orphan]).await;
            insert_artist_session(
                &format!("artist_{u}"),
                u,
                "a log line with no file mentions",
            )
            .await;
        }
        reset_media_cursors().await;
        // Budget 1 byte fits exactly one session per tick. After a covered
        // user's deletion pass the cursor must NOT reset (a full re-scan
        // would consume the whole budget every tick and starve the second
        // user in read_dir order).
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            1
        );
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            1,
            "second user swept on the next tick — covered user consumed no budget"
        );
        for u in dirs {
            assert!(
                !userspaces
                    .path()
                    .join(u)
                    .join("generated")
                    .join(format!("{u}_orphan.png"))
                    .exists(),
                "{u}'s unmentioned file deleted"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_basename_unique_across_dirs() {
        let userspaces = media_fixture("base").await;
        let up = userspaces.path().join("base").join("uploads");
        tokio::fs::create_dir_all(&up).await.unwrap();
        let g = userspaces
            .path()
            .join("base")
            .join("generated")
            .join("pic.png");
        let u = up.join("pic.png");
        tokio::fs::write(&g, "x").await.unwrap();
        tokio::fs::write(&u, "x").await.unwrap();
        // A bare basename mention is ambiguous (duplicate across the
        // generated+uploads UNION) — the design's safe direction keeps BOTH
        // ("иначе файл сохраняется — пере-держать"): the mention cannot be
        // attributed to one duplicate, so deleting either could destroy a
        // mentioned file.
        insert_artist_session("artist_b1", "base", "here is pic.png").await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 0, "ambiguous basename keeps both files");
        assert!(g.exists());
        assert!(u.exists());

        // An absolute mention of one duplicate still keeps the OTHER: its
        // basename remains ambiguous across the union (the absolute path
        // matching is per-file, the ambiguity rule is per-union).
        insert_artist_session(
            "artist_b2",
            "base",
            &format!("[IMAGE:{}]", u.canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(
            n, 0,
            "ambiguous duplicate kept even when only one is mentioned"
        );
        assert!(g.exists());
        assert!(u.exists());
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_prunes_cursors_of_vanished_users() {
        let userspaces = media_fixture("gone").await;
        write_gen_files(userspaces.path(), "gone", &["f_a.png"]).await;
        insert_artist_session("artist_gone1", "gone", "no file mentions").await;
        reset_media_cursors().await;
        assert_eq!(sweep_media_at(userspaces.path()).await.unwrap(), 1);
        assert!(
            MEDIA_CURSORS
                .lock()
                .await
                .as_ref()
                .unwrap()
                .contains_key("gone")
        );
        // The userspace dir vanishes — the next complete pass prunes its
        // cursor (bounded per-user memory; a re-created dir starts fresh).
        tokio::fs::remove_dir_all(userspaces.path().join("gone"))
            .await
            .unwrap();
        assert_eq!(sweep_media_at(userspaces.path()).await.unwrap(), 0);
        assert!(
            !MEDIA_CURSORS
                .lock()
                .await
                .as_ref()
                .unwrap()
                .contains_key("gone")
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_video_case_insensitive() {
        let userspaces = media_fixture("video").await;
        let files = write_gen_files(userspaces.path(), "video", &["clip.mp4", "Photo.PNG"]).await;
        // Mentions in a different case: video matches case-insensitively,
        // non-video extensions do not.
        insert_artist_session("artist_v1", "video", "[VIDEO:CLIP.MP4] photo.png").await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1);
        assert!(
            files[0].exists(),
            "video kept via case-insensitive basename"
        );
        assert!(!files[1].exists(), "case-mismatched non-video not kept");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_overflowed_cleared_by_clear_rotation() {
        let userspaces = media_fixture("ovf").await;
        let files = write_gen_files(userspaces.path(), "ovf", &["f_a.png", "f_b.png"]).await;
        // A bloated session (>4× scan budget) overflows the keep-set cap:
        // nothing can be safely deleted while it is in the base.
        let huge = "x".repeat(MEDIA_SCAN_BUDGET_BYTES * 4 + 1);
        insert_artist_session(
            "artist_ovf1",
            "ovf",
            &format!(
                "[IMAGE:{}] {huge}",
                files[0].canonicalize().unwrap().display()
            ),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 0, "overflowed keep-set never deletes");
        assert!(files[0].exists());
        assert!(files[1].exists());
        // /clear rotation: the overflowed session is deleted; the cursor
        // resets (incl. the overflowed flag), so a fresh base re-enables the
        // sweep — f_a (mentioned only in the cleared session) becomes a
        // candidate, f_b survives via the surviving session.
        crate::session::store()
            .conn
            .execute(
                "DELETE FROM session_metadata WHERE agent_id = 'artist_ovf1'",
                (),
            )
            .await
            .unwrap();
        insert_artist_session(
            "artist_ovf2",
            "ovf",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "overflowed user re-enabled after /clear rotation");
        assert!(!files[0].exists());
        assert!(files[1].exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;
        let userspaces = media_fixture("sym").await;
        // A symlinked directory INSIDE generated/ pointing OUTSIDE the
        // userspace root: traversal must not follow it — the deletion pass
        // would otherwise collect (and delete) real files outside the root,
        // violating "never delete a file whose mention was not scanned" at
        // the filesystem level. The symlink itself is also never a deletion
        // candidate (skipped by the no-follow walk).
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("victim.png");
        tokio::fs::write(&victim, "x").await.unwrap();
        let link = userspaces
            .path()
            .join("sym")
            .join("generated")
            .join("linked");
        symlink(outside.path(), &link).unwrap();
        // An artist session mentioning ONE real file: the deletion pass must
        // actually run (the unmentioned file below is a candidate) — without
        // sessions the empty-base guard short-circuits and the no-follow
        // logic would never be exercised.
        let files = write_gen_files(userspaces.path(), "sym", &["kept.png", "orphan.png"]).await;
        insert_artist_session(
            "artist_sym",
            "sym",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "orphan deleted; symlink and outside victim untouched");
        assert!(files[0].exists(), "mentioned file kept");
        assert!(!files[1].exists(), "unmentioned file deleted");
        assert!(victim.exists(), "file outside the userspace root untouched");
        assert!(link.exists(), "symlink itself kept");
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_skips_top_level_symlinked_dir() {
        use std::os::unix::fs::symlink;
        let userspaces = media_fixture("topsym").await;
        // The generated/ dir ITSELF is a symlink to an outside tree: read_dir
        // on the top-level path would resolve it, so the entry-level no-follow
        // inside list_files cannot catch it — the symlink_metadata guard must.
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("victim.png");
        tokio::fs::write(&victim, "x").await.unwrap();
        let gen_dir = userspaces.path().join("topsym").join("generated");
        tokio::fs::remove_dir_all(&gen_dir).await.unwrap();
        symlink(outside.path(), &gen_dir).unwrap();
        // An artist session mentioning ONE uploads file proves the deletion
        // pass ran (the unmentioned orphan below is a candidate) while the
        // symlinked generated/ tree stays untouched.
        let up = userspaces.path().join("topsym").join("uploads");
        let kept = up.join("kept.png");
        let orphan = up.join("orphan.png");
        tokio::fs::write(&kept, "x").await.unwrap();
        tokio::fs::write(&orphan, "x").await.unwrap();
        insert_artist_session(
            "artist_topsym",
            "topsym",
            &format!("[IMAGE:{}]", kept.canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "orphan deleted; symlinked generated tree untouched");
        assert!(kept.exists(), "mentioned uploads file kept");
        assert!(!orphan.exists(), "unmentioned uploads file deleted");
        assert!(
            victim.exists(),
            "file behind the top-level symlink untouched"
        );
        assert!(gen_dir.exists(), "top-level symlink itself kept");
    }
}
