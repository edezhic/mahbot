//! Maintainer — autonomous periodic codebase investigation agent.
//!
//! The Maintainer scans workspaces for refactoring opportunities and creates
//! planning tickets on the board. It does NOT make direct code changes.

use chrono::Utc;
use std::time::Duration;
use tracing::{info, warn};

use crate::Role;
use crate::agent::run_agent;
use crate::board::TicketPhase;
use crate::turso;
/// Run the maintainer background loop.
///
/// Runs a Maintainer agent per workspace with the investigation prompt.
/// On success (agent produced a response), updates `maintainer_last_run_at`
/// and adjusts debounce: resets to 1 min if tickets were created, doubles
/// (capped at 240 min) otherwise.  On cancellation or error, debounce and
/// last-run timestamp are left unchanged.
#[allow(clippy::too_many_lines)]
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

        for ws in &workspaces {
            if shutdown.is_cancelled() {
                break;
            }

            // Per-workspace pause — skip maintainer for this workspace.
            if ws.paused {
                continue;
            }

            // Skip workspace if maintainer is not explicitly enabled
            if !ws.maintenance {
                continue;
            }

            // Only maintain workspaces whose discovery has completed.
            if ws.status != "ready" {
                info!(workspace = %ws.name, status = %ws.status, "Maintainer: skipping — workspace not ready");
                continue;
            }

            // ── Debounce gate check ──────────────────────────────────────
            {
                let now = Utc::now();
                let debounce = ws.maintainer_debounce_mins.clamp(0, 240); // floor+cap guard
                if let Some(ref last_str) = ws.maintainer_last_run_at {
                    match crate::turso::parse_utc_timestamp(last_str) {
                        Ok(last_time) => {
                            let elapsed = now - last_time;
                            let mins_elapsed = elapsed.num_minutes();
                            if mins_elapsed < debounce {
                                continue;
                            }
                        }
                        Err(e) => {
                            warn!(maintainer_last_run_at = %last_str, error = %e, "Failed to parse maintainer_last_run_at, letting through");
                            // If parse fails, let it through — stale data shouldn't block
                        }
                    }
                }
                // If last_run_at is None, always run (first time).
            }

            // Skip if there are already enough planning or paused tickets — pipeline pressure valve
            // Skip if this workspace has active pipeline tickets (dev/review/QA)
            if let Some(board) = crate::board::BOARD.get() {
                let count_status = |status: TicketPhase| async move {
                    match board.count_by_status(status, Some(&ws.name)).await {
                        Ok(c) => c,
                        Err(e) => {
                            warn!(workspace = %ws.name, %status, error = %e, "Maintainer: failed to count tickets");
                            0
                        }
                    }
                };

                let c1 = count_status(TicketPhase::Planning).await;
                let c2 = count_status(TicketPhase::Paused).await;

                if c1 + c2 >= 5 {
                    info!(workspace = %ws.name, planning = c1, paused = c2, "Maintainer: skipping — tickets already in planning or paused");
                    continue;
                }

                match board.has_pipeline_blocker_for_workspace(&ws.name).await {
                    Ok(true) => {
                        info!(workspace = %ws.name, "Maintainer: skipping — workspace has active pipeline tickets");
                        continue;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        warn!(workspace = %ws.name, error = %e, "Maintainer: failed to check pipeline blocker");
                        continue;
                    }
                }
            }

            // Unique session key per run — don't accumulate history
            let run_id = crate::session::maintainer_session_key(&ws.name);

            info!(workspace = %ws.name, run = %run_id, "Maintainer: starting maintenance run");

            let prompt = crate::prompt::load_prompt("maintain.md");
            let (agent, response) =
                run_agent(run_id.clone(), Role::Maintainer, ws, None, &prompt).await;

            if let Some(_response) = response {
                info!(workspace = %ws.name, "Maintainer: run complete");

                // ── Debounce update after successful run ──────────────────
                let now_str = turso::now();
                let new_debounce =
                    compute_debounce(&agent.id, ws.maintainer_debounce_mins, ws.name.as_str())
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
        }
    }
}

/// Compute the new debounce value based on whether the agent produced tickets.
///
/// - If `create_ticket` was called → reset to 1.
/// - If no `create_ticket` calls → double (clamped to `[5, 240]`, capped at 240).
async fn compute_debounce(agent_id: &str, current: i64, ws_name: &str) -> i64 {
    let store = crate::stats::store();

    match store.query_tool_usage(agent_id, "create_ticket").await {
        Ok(Some(call_count)) if call_count > 0 => {
            info!(workspace = %ws_name, "Maintainer: produced tickets — reset debounce to 1");
            1
        }
        Ok(Some(_) | None) => {
            let new_val = advance_debounce(current);
            if new_val >= 240 && current < 240 {
                info!(workspace = %ws_name, "Maintainer: no tickets created — debounce capped at 240");
            } else {
                info!(workspace = %ws_name, "Maintainer: no tickets created — debounce doubled to {new_val}");
            }
            new_val
        }
        Err(e) => {
            warn!(workspace = %ws_name, error = %e, "Maintainer: stats query failed, advancing debounce");
            advance_debounce(current)
        }
    }
}

/// Double the debounce value, clamped to `[5, 240]` with a hard cap at 240.
fn advance_debounce(mins: i64) -> i64 {
    (mins.clamp(5, 240) * 2).min(240)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_debounce_edges() {
        // Floor: values below 5 clamp to 5, then double to 10.
        assert_eq!(advance_debounce(0), 10);
        assert_eq!(advance_debounce(4), 10);
        assert_eq!(advance_debounce(5), 10);

        // Normal doubling.
        assert_eq!(advance_debounce(6), 12);
        assert_eq!(advance_debounce(60), 120);

        // Cap: 120 doubles to 240, 121 clamps→240, 240 stays at 240.
        assert_eq!(advance_debounce(119), 238);
        assert_eq!(advance_debounce(120), 240);
        assert_eq!(advance_debounce(121), 240);
        assert_eq!(advance_debounce(240), 240);

        // Above 240 — clamp brings it down, then double hits cap.
        assert_eq!(advance_debounce(300), 240);
    }
}
