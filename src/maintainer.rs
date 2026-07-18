//! Maintainer — autonomous periodic codebase investigation agent.
//!
//! The Maintainer scans workspaces for refactoring opportunities and creates
//! planning tickets on the board. It does NOT make direct code changes.

use chrono::Utc;
use std::time::Duration;
use tracing::{error, info, warn};

use futures_util::future::join_all;

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::agent::run_agent;
use crate::board::TicketPhase;
use crate::turso;
use crate::util::panic_message;

/// Maximum number of tickets allowed in Analysis + Planning + ReadyForDevelopment
/// before the maintainer pauses ticket creation.
const MAX_PRE_DEV_TICKETS: i64 = 5;

/// Run the maintainer background loop.
///
/// Runs a Maintainer agent per workspace with the investigation prompt.
/// On success (agent produced a response), updates `maintainer_last_run_at`
/// and adjusts debounce: resets to 1 min if tickets were created, advances
/// otherwise (`advance_debounce`: clamps current to [`MAX_MAINTAINER_DEBOUNCE_MINS`],
/// doubles, caps at that value — producing the sequence 1 → 10 → 20 → … → `MAX_MAINTAINER_DEBOUNCE_MINS`).
/// On cancellation or error, debounce and last-run timestamp are left unchanged.
///
/// ## Concurrency
///
/// Workspaces are processed **concurrently** via `tokio::spawn` + `join_all`,
/// matching the pattern used by [`poll_round`](crate::management::poll_round).
/// Each workspace's agent run is independent (disjoint session keys, no shared
/// mutable state between runs) so concurrency is safe.
///
/// ## Behavioural notes
///
/// * **Pipeline throttle**: `is_maintainer_pipeline_full` checks run concurrently
///   for all workspaces rather than sequentially. The
///   [`MAX_PRE_DEV_TICKETS`] throttle therefore sees a slightly wider window
///   of concurrent ticket counts — acceptable since the maintainer is
///   best-effort and the per-workspace count check is still atomic.
/// * **Shutdown**: all matching workspaces are spawned in a single batch even
///   if shutdown fires during dispatch; each task independently checks
///   cancellation before the LLM call. The original sequential loop's
///   immediate-break-on-cancellation is replaced by a cooperative per-task
///   check, which is consistent with the rest of the codebase (neither
///   [`poll_round`](crate::management::poll_round) nor
///   [`process_single_workspace`](crate::management::process_single_workspace)
///   check cancellation mid-batch). Mid-execution cancellation within a
///   running agent is handled by the global `CancellationToken` inside
///   `Agent::work`.
pub async fn run_maintainer_loop() {
    let interval = Duration::from_mins(1);
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if !crate::shutdown::sleep_or_shutdown(interval).await {
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
                if should_skip_maintainer_debounce(ws) {
                    return false;
                }
                true
            })
            .map(|ws| {
                let prompt = prompt.clone();
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    // Async pre-check (DB query) — cannot live in a sync .filter closure.
                    if is_maintainer_pipeline_full(&ws).await {
                        return;
                    }

                    // Early cancellation check before expensive LLM call.
                    if shutdown.is_cancelled() {
                        return;
                    }

                    // Unique session key per run — don't accumulate history
                    let run_id = crate::session::maintainer_session_key(&ws.name);

                    info!(workspace = %ws.name, run = %run_id, "Maintainer: starting maintenance run");

                    let (agent, response) =
                        run_agent(run_id.clone(), Role::Maintainer, &ws, None, &prompt, String::new(), String::new()).await;

                    if let Some(_response) = response {
                        info!(workspace = %ws.name, "Maintainer: run complete");

                        // ── Debounce update after successful run ──────────────────
                        let now_str = turso::now();
                        let new_debounce = compute_debounce(
                            &agent.session_key,
                            ws.maintainer_debounce_mins,
                            ws.name.as_str(),
                        )
                        .await;

                        if let Err(e) = crate::workspace::store()
                            .set_maintenance_debounce(&ws.name, new_debounce, &now_str)
                            .await
                        {
                            warn!(workspace = %ws.name, error = %e, "Maintainer: failed to update debounce state");
                        }
                    } else {
                        // Error or cancellation — do NOT advance debounce, do NOT update last_run_at
                        info!(workspace = %ws.name, "Maintainer: run failed or cancelled — debounce unchanged");
                    }

                    // Backlog tickets are discovered by the poll loop (BacklogAnalysis),
                    // not via explicit notification — no Manager notification needed here.
                })
            })
            .collect();

        let results = join_all(tasks).await;
        for result in results {
            if let Err(e) = result {
                if e.is_panic() {
                    let payload = e.into_panic();
                    error!(
                        error = %panic_message(&*payload),
                        "Panic in maintainer task — maintainer loop continues",
                    );
                } else {
                    error!("Maintainer task was cancelled — maintainer loop continues");
                }
            }
        }
    }
}

/// Returns `true` if the maintainer should skip this workspace due to debounce.
///
/// Checks whether enough time has passed since the last maintainer run by
/// parsing `maintainer_last_run_at`, computing elapsed time relative to the
/// debounce interval. On parse errors (stale data) or when `last_run_at` is
/// `None` (first run), returns `false` to allow the run.
fn should_skip_maintainer_debounce(ws: &Workspace) -> bool {
    let now = Utc::now();
    let debounce = ws
        .maintainer_debounce_mins
        .clamp(0, Workspace::MAX_MAINTAINER_DEBOUNCE_MINS);
    if let Some(ref last_str) = ws.maintainer_last_run_at {
        match turso::parse_utc_timestamp(last_str) {
            Ok(last_time) => {
                let elapsed = now - last_time;
                let mins_elapsed = elapsed.num_minutes();
                if mins_elapsed < debounce {
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
/// ReadyForDevelopment).
///
/// If the board is unavailable, returns `false` to allow the run through.
async fn is_maintainer_pipeline_full(ws: &Workspace) -> bool {
    let Some(board) = crate::board::BOARD.get() else {
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
        let ready = count_phase(TicketPhase::ReadyForDevelopment).await;
        analysis + planning + ready
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

/// Compute the new debounce value based on whether the agent produced tickets.
///
/// - If `create_ticket` was called → reset to 1.
/// - If no `create_ticket` calls → double (clamped to `[5, Workspace::MAX_MAINTAINER_DEBOUNCE_MINS]`, capped at `Workspace::MAX_MAINTAINER_DEBOUNCE_MINS`).
async fn compute_debounce(agent_id: &str, current: i64, ws_name: &str) -> i64 {
    let store = crate::stats::store();

    match store.query_tool_usage(agent_id, "create_ticket").await {
        Ok(call_count) if call_count > 0 => {
            info!(workspace = %ws_name, "Maintainer: produced tickets — reset debounce to 1");
            1
        }
        Ok(_) => {
            let new_val = advance_debounce(current);
            if new_val >= Workspace::MAX_MAINTAINER_DEBOUNCE_MINS
                && current < Workspace::MAX_MAINTAINER_DEBOUNCE_MINS
            {
                info!(workspace = %ws_name, "Maintainer: no tickets created — debounce capped at {}", Workspace::MAX_MAINTAINER_DEBOUNCE_MINS);
            } else {
                info!(workspace = %ws_name, "Maintainer: no tickets created — debounce advanced to {new_val}");
            }
            new_val
        }
        Err(e) => {
            warn!(workspace = %ws_name, error = %e, "Maintainer: stats query failed, advancing debounce");
            advance_debounce(current)
        }
    }
}

/// Double the debounce value, clamped to `[5, Workspace::MAX_MAINTAINER_DEBOUNCE_MINS]`
/// with a hard cap at `Workspace::MAX_MAINTAINER_DEBOUNCE_MINS`.
fn advance_debounce(mins: i64) -> i64 {
    (mins.clamp(5, Workspace::MAX_MAINTAINER_DEBOUNCE_MINS) * 2)
        .min(Workspace::MAX_MAINTAINER_DEBOUNCE_MINS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal workspace with only the fields relevant to debounce tests.
    fn ws_with(last_run_at: Option<&str>, debounce_mins: i64) -> Workspace {
        Workspace {
            name: "test-ws".into(),
            path: "/tmp/test".into(),
            status: WorkspaceStatus::Ready,
            created_at: String::new(),
            updated_at: String::new(),
            maintenance_enabled: true,
            paused: false,
            maintainer_debounce_mins: debounce_mins,
            maintainer_last_run_at: last_run_at.map(String::from),
            diagnostics: None,
            diagnostics_updated_at: None,
            notes: String::new(),
            last_analyzed_commit: None,
        }
    }

    /// Table-driven test for all `should_skip_maintainer_debounce` cases.
    ///
    /// Reasoning for the "just ran" cases: both `now_str` cases evaluate
    /// against the same instant, so any near-zero elapsed time produces
    /// `elapsed < debounce` → `true`. The 500 value is clamped to 240
    /// internally, so the behaviour is identical to the 240 case.
    #[test]
    fn should_skip_maintainer_debounce_cases() {
        let now_str = Utc::now().to_rfc3339();
        let cases = [
            (
                ws_with(None, 5),
                false,
                "no prior run → last_run_at is None → no debounce",
            ),
            (
                ws_with(Some("garbage-timestamp"), 5),
                false,
                "unparseable timestamp → parse error → let through",
            ),
            (
                ws_with(Some(&now_str), Workspace::MAX_MAINTAINER_DEBOUNCE_MINS),
                true,
                "just ran — elapsed ~0s < 240 → skip",
            ),
            (
                ws_with(Some("2020-01-01T00:00:00Z"), 5),
                false,
                "long ago — many years elapsed >= 5 → let through",
            ),
            (
                ws_with(Some("2020-01-01T00:00:00Z"), -5),
                false,
                "debounce clamped from -5 to 0 → mins_elapsed < 0 never true",
            ),
            (
                ws_with(Some(&now_str), 500),
                true,
                "debounce clamped from 500 to 240 → elapsed ~0s < 240 → skip",
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

    #[test]
    fn advance_debounce_edges() {
        // Floor: values below 5 clamp to 5, then double to 10.
        assert_eq!(advance_debounce(0), 10);
        assert_eq!(advance_debounce(4), 10);
        assert_eq!(advance_debounce(5), 10);

        // Normal doubling.
        assert_eq!(advance_debounce(6), 12);
        assert_eq!(advance_debounce(60), 120);

        // Cap: 119 doubles to 238; 120 doubles to MAX; 121 clamps→MAX; MAX stays at MAX.
        assert_eq!(advance_debounce(119), 238);
        assert_eq!(
            advance_debounce(120),
            Workspace::MAX_MAINTAINER_DEBOUNCE_MINS
        );
        assert_eq!(
            advance_debounce(121),
            Workspace::MAX_MAINTAINER_DEBOUNCE_MINS
        );
        assert_eq!(
            advance_debounce(240),
            Workspace::MAX_MAINTAINER_DEBOUNCE_MINS
        );

        // Above MAX — clamp brings it down, then double hits cap.
        assert_eq!(
            advance_debounce(300),
            Workspace::MAX_MAINTAINER_DEBOUNCE_MINS
        );
    }
}
