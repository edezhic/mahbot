//! Maintainer — autonomous periodic codebase investigation agent.
//!
//! The Maintainer scans workspaces for refactoring opportunities and creates
//! planning tickets on the board. It does NOT make direct code changes.

use chrono::Utc;
use std::time::Duration;
use tracing::{info, warn};

use futures_util::future::join_all;

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::agent::run_default_agent;
use crate::db;
use crate::pipeline::board::TicketPhase;

/// Maximum number of tickets allowed in Analysis + Planning + Queued
/// before the maintainer pauses ticket creation.
const MAX_PRE_DEV_TICKETS: i64 = 5;

/// Run the maintainer background loop.
///
/// Runs a Maintainer agent per workspace with the investigation prompt.
/// The agent id is deterministic (`maintainer_{ws}`) so runs are resumable:
/// a content-bearing session means an interrupted run, which is resumed with
/// an empty message and `resume = true` (bypassing the debounce gate — it is
/// not a new run). On success the created tickets are the durable outcome and
/// the session is deleted so the next cycle starts fresh; `maintainer_last_run_at`
/// is updated only on success. On failure or cancellation the session is kept
/// so the next cycle naturally resumes — unlike the pre-resume behavior, a
/// failed run is retried on the very next 1-minute cycle (resume bypasses the
/// debounce gate), which is accepted: repeated insta-deaths are a bug to fix
/// at root cause, not to paper over.
///
/// ## Concurrency
///
/// Workspaces are processed **concurrently** via `tokio::spawn` + `join_all`,
/// matching the pattern used by `poll_round`.
/// Each workspace's agent run is independent (disjoint agent IDs, no shared
/// mutable state between runs) so concurrency is safe.
///
/// ## Behavioural notes
///
/// * **Pipeline throttle**: `is_maintainer_pipeline_full` checks run concurrently
///   for all workspaces rather than sequentially, so the pre-dev ticket-count
///   throttle sees a slightly wider window of concurrent ticket counts —
///   acceptable since the maintainer is best-effort and the per-workspace
///   count check is still atomic.
/// * **Double-dispatch guard**: a defensive registry check per task — a live
///   run for the same deterministic id is never re-dispatched. The loop
///   itself is sequential (`join_all` blocks the next cycle), so the guard
///   only fires if a future dispatch path ever overlaps.
/// * **Shutdown**: all matching workspaces are spawned in a single batch even
///   if shutdown fires during dispatch; each task independently checks
///   cancellation before the LLM call. The original sequential loop's
///   immediate-break-on-cancellation is replaced by a cooperative per-task
///   check, which is consistent with the rest of the codebase (neither
///   `poll_round` nor
///   `process_single_workspace`
///   check cancellation mid-batch). Mid-execution cancellation within a
///   running agent is handled by the global `CancellationToken` inside
///   `Agent::work`.
pub async fn run_maintainer_loop() {
    let interval = Duration::from_mins(1);
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }

        // Fetch all workspaces
        let workspaces = match crate::workspace::get_workspaces().await {
            Ok(list) => list,
            Err(e) => {
                warn!(error = %e, "Maintainer: failed to list workspaces");
                continue;
            }
        };

        if workspaces.is_empty() {
            info!("Maintainer: no workspaces configured, skipping cycle");
            continue;
        }

        // Load prompt once before spawning (each spawned task gets its own clone).
        let prompt = crate::prompt::load_prompt("maintain.md");

        // Concurrent dispatch: sync pre-checks in `.filter()`, async DB check
        // and cancellation check inside each spawned task.
        let tasks: Vec<_> = workspaces
            .into_iter()
            .filter(|ws| {
                if !ws.maintenance_enabled {
                    return false;
                }
                if ws.status != WorkspaceStatus::Ready {
                    info!(workspace = %ws.name, status = %ws.status, "Maintainer: skipping — workspace not ready");
                    return false;
                }
                true
            })
            .map(|ws| {
                let prompt = prompt.clone();
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    // Deterministic agent ID — the session doubles as the resume state.
                    let agent_id = crate::session::maintainer_agent_id(&ws.name);

                    // Defensive double-dispatch guard: never re-dispatch a live
                    // run for the same deterministic ID.
                    if crate::agent::registry::AGENT_REGISTRY.contains(&agent_id) {
                        return;
                    }

                    // Resume decision: a session with content is an interrupted run to
                    // continue — resume bypasses the debounce gate (it is not a new run).
                    let resume = crate::session::store().has_content(&agent_id).await;
                    if !resume && should_skip_maintainer_debounce(&ws) {
                        return;
                    }

                    // Async pre-check (DB query) — cannot live in a sync .filter closure.
                    if is_maintainer_pipeline_full(&ws).await {
                        return;
                    }

                    // Early cancellation check before expensive LLM call.
                    if shutdown.is_cancelled() {
                        return;
                    }

                    info!(workspace = %ws.name, agent_id = %agent_id, resumed = resume, "Maintainer: starting maintenance run");

                    let message = if resume { "" } else { prompt.as_str() };
                    let (_, response) = run_default_agent(
                        &agent_id,
                        Role::Maintainer,
                        &ws,
                        message,
                        resume,
                        None,
                        None,
                        None,
                    )
                    .await;

                    if let Some(_response) = response {
                        info!(workspace = %ws.name, "Maintainer: run complete");

                        // ── Completion: clean state for the next cycle ─────────────────────────
                        // The created tickets are the durable outcome, not the transcript —
                        // delete the session so the next run starts fresh (does not rely on
                        // TTL cleanup), then record the last-run timestamp.
                        if let Err(e) = crate::session::store().delete(&agent_id).await {
                            warn!(workspace = %ws.name, error = %e, "Maintainer: failed to delete completed session");
                        }
                        if let Err(e) = crate::workspace::store()
                            .set_maintainer_last_run_at(&ws.name, &db::now())
                            .await
                        {
                            warn!(workspace = %ws.name, error = %e, "Maintainer: failed to update last-run timestamp");
                        }
                    } else {
                        // Error or cancellation — keep the session so the next cycle
                        // naturally resumes; do NOT update last_run_at.
                        info!(workspace = %ws.name, "Maintainer: run failed or cancelled — session kept for resume, last_run_at unchanged");
                    }

                    // Backlog tickets are discovered by the poll loop (BacklogAnalysis),
                    // not via explicit notification — no Manager notification needed here.
                })
            })
            .collect();

        let results = join_all(tasks).await;
        crate::util::log_join_failures(
            results,
            "Panic in maintainer task — maintainer loop continues",
            "Maintainer task was cancelled — maintainer loop continues",
        );
    }
}

/// Returns `true` if the maintainer should skip this workspace due to the
/// fixed debounce gate.
///
/// Checks whether enough time has passed since the last maintainer run by
/// parsing `maintainer_last_run_at`, computing elapsed time relative to the
/// [`Workspace::MAINTAINER_DEBOUNCE_MINS`] interval. On parse errors (stale data) or
/// when `last_run_at` is `None` (first run), returns `false` to allow the run
/// (fail-open).
fn should_skip_maintainer_debounce(ws: &Workspace) -> bool {
    let now = Utc::now();
    if let Some(ref last_str) = ws.maintainer_last_run_at {
        match db::parse_utc_timestamp(last_str) {
            Ok(last_time) => {
                let elapsed = now - last_time;
                let mins_elapsed = elapsed.num_minutes();
                if mins_elapsed < Workspace::MAINTAINER_DEBOUNCE_MINS {
                    return true;
                }
            }
            Err(e) => {
                warn!(
                    maintainer_last_run_at = %last_str,
                    error = %e,
                    "Failed to parse maintainer_last_run_at, letting through"
                );
            }
        }
    }
    false
}

/// Returns `true` if the maintainer should skip because the pre-dev pipeline
/// has reached `MAX_PRE_DEV_TICKETS` or more tickets (Analysis + Planning +
/// Queued).
///
/// If the board is unavailable, returns `false` to allow the run through.
async fn is_maintainer_pipeline_full(ws: &Workspace) -> bool {
    let Some(board) = crate::pipeline::board::BOARD.get() else {
        return false;
    };

    let count_phase = |phase: TicketPhase| async move {
        match board.count_by_phase(phase, Some(&ws.name)).await {
            Ok(c) => c,
            Err(e) => {
                warn!(workspace = %ws.name, %phase, error = %e, "Maintainer: failed to count tickets");
                0
            }
        }
    };

    let pre_dev_count = {
        let analysis = count_phase(TicketPhase::Analysis).await;
        let planning = count_phase(TicketPhase::Planning).await;
        let queued = count_phase(TicketPhase::Queued).await;
        analysis + planning + queued
    };

    if pre_dev_count >= MAX_PRE_DEV_TICKETS {
        info!(
            workspace = %ws.name,
            pre_dev = pre_dev_count,
            "Maintainer: skipping — pre-development pipeline has >= {} tickets",
            MAX_PRE_DEV_TICKETS,
        );
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal workspace with only the fields relevant to debounce tests.
    fn ws_with(last_run_at: Option<&str>) -> Workspace {
        Workspace {
            name: "test-ws".into(),
            path: "/tmp/test".into(),
            status: WorkspaceStatus::Ready,
            maintenance_enabled: true,
            paused: false,
            maintainer_last_run_at: last_run_at.map(String::from),
            diagnostics: None,
            notes: String::new(),
            last_analyzed_commit: None,
            ephemeral: false,
        }
    }

    /// Table-driven test for all `should_skip_maintainer_debounce` cases.
    ///
    /// Reasoning for the "just ran" case: `now_str` evaluates against the
    /// same instant, so any near-zero elapsed time produces
    /// `elapsed < debounce` → `true`.
    #[test]
    fn should_skip_maintainer_debounce_cases() {
        let now_str = Utc::now().to_rfc3339();
        let cases = [
            (
                ws_with(None),
                false,
                "no prior run → last_run_at is None → no debounce",
            ),
            (
                ws_with(Some("garbage-timestamp")),
                false,
                "unparseable timestamp → parse error → let through",
            ),
            (
                ws_with(Some(&now_str)),
                true,
                "just ran — elapsed < 5 min → skip",
            ),
            (
                ws_with(Some("2020-01-01T00:00:00Z")),
                false,
                "long ago — many years elapsed >= 5 → let through",
            ),
        ];
        for (ws, expected, reason) in &cases {
            assert_eq!(
                should_skip_maintainer_debounce(ws),
                *expected,
                "case: {reason}"
            );
        }
    }
}
