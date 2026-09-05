//! Maintainer — autonomous periodic codebase investigation agent.
//!
//! The Maintainer scans workspaces for refactoring opportunities and creates
//! planning tickets on the board. It does NOT make direct code changes.

use chrono::{DateTime, Utc};
use std::fmt::Write;
use std::time::Duration;
use tracing::{info, warn};

use futures_util::future::join_all;

use crate::Agent;
use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::agent::run_default_agent;
use crate::db;
use crate::pipeline::board::TicketPhase;

/// Maximum number of tickets allowed in Analysis + Planning + Queued
/// before the maintainer pauses ticket creation.
const MAX_PRE_DEV_TICKETS: i64 = 5;

/// Char budget for the persisted recommendations JSON blob — a pathological
/// extraction is truncated/dropped at save time, never at injection time.
const MAINTAINER_RECOMMENDATIONS_BUDGET_BYTES: usize = 2048;
/// Recommendations older than this are stale: not injected, cleared at dispatch.
const MAINTAINER_RECOMMENDATIONS_MAX_AGE_DAYS: i64 = 7;

/// Structured extraction from the post-run LLM call — recommendations only.
#[derive(serde::Deserialize)]
struct MaintainerRecommendationsExtraction {
    #[serde(default)]
    recommendations: Vec<String>,
}

/// Persisted blob: the bounded recommendations plus an RFC 3339 generation time.
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredMaintainerRecommendations {
    recommendations: Vec<String>,
    generated_at: String,
}

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
/// ## Cross-round recommendations
///
/// On success a fail-open extraction runs over the completed session (before the
/// session is deleted, so the full transcript — including a resumed run's
/// accumulated history — is available) and stores a replace-only
/// `maintainer_recommendations` blob for the next fresh run. Fresh starts
/// prepend a short advisory block from the previous round's list; resume runs
/// already carry that block in their session. The whole path is strictly
/// best-effort — any failure leaves the task text unchanged and never affects
/// the run's success.
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
        let workspaces = match crate::workspace::store().list().await {
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

                    let message = if resume {
                        String::new()
                    } else {
                        prepend_recommendations(&prompt, &ws.name).await
                    };
                    let (agent, response) = run_default_agent(
                        &agent_id,
                        Role::Maintainer,
                        &ws,
                        &message,
                        resume,
                        None,
                        None,
                        None,
                    )
                    .await;

                    if let Some(_response) = response {
                        info!(workspace = %ws.name, "Maintainer: run complete");

                        // Post-run recommendation extraction — BEFORE session cleanup so the
                        // full accumulated transcript (including a resumed run's) is available.
                        // Strictly fail-open: any failure leaves the previous blob untouched.
                        extract_and_store_recommendations(&agent, &ws).await;

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

// ── Cross-round recommendations ──────────────────────────────────────

/// Post-run extraction: one short-budget structured LLM call over the completed
/// session, producing recommendations for the next fresh maintainer round.
/// Strictly fail-open — any failure is logged and skipped; the previously
/// stored blob stays untouched and the run's success is unaffected.
/// Runs BEFORE session cleanup so the full accumulated transcript (including a
/// resumed run's) is available. An empty extraction still saves an empty blob
/// (replace-only semantics: the next run simply gets no block).
async fn extract_and_store_recommendations(agent: &Agent, ws: &Workspace) {
    let prompt = crate::prompt::load_prompt("extraction/maintainer.md");
    let extracted = match agent
        .extract_verdict::<MaintainerRecommendationsExtraction>(
            &prompt,
            None,
            Some(&crate::retry::RetryPolicy::comment()),
        )
        .await
    {
        Ok(extracted) => extracted,
        Err(e) => {
            warn!(
                workspace = %ws.name,
                error = %e,
                "Maintainer: recommendation extraction failed — keeping previous blob"
            );
            return;
        }
    };

    // Replace-only semantics: even an empty extraction clears the previous blob.
    let stored = StoredMaintainerRecommendations {
        recommendations: cap_recommendations(extracted.recommendations),
        generated_at: db::now(),
    };
    let Ok(json) = serde_json::to_string(&stored) else {
        warn!(workspace = %ws.name, "Maintainer: failed to serialize recommendations");
        return;
    };
    if let Err(e) = crate::workspace::store()
        .set_maintainer_recommendations(&ws.name, Some(&json))
        .await
    {
        warn!(workspace = %ws.name, error = %e, "Maintainer: failed to store recommendations");
    }
}

/// Bound the persisted recommendations to [`MAINTAINER_RECOMMENDATIONS_BUDGET_BYTES`]:
/// trim/drop empty items, then keep whole items only while the serialized blob
/// fits the budget (a lone oversized item is dropped, not split).
#[must_use]
fn cap_recommendations(recs: Vec<String>) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();
    for item in recs {
        let trimmed = item.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        kept.push(trimmed);
        let fits = serde_json::to_string(&kept)
            .is_ok_and(|s| s.len() <= MAINTAINER_RECOMMENDATIONS_BUDGET_BYTES);
        if !fits {
            kept.pop();
            break;
        }
    }
    kept
}

/// Read the previous round's recommendations and return the fresh-start initial
/// message: an advisory `<maintainer-recommendations>` block prepended to
/// `task_text`, or `task_text` unchanged when there is nothing injectable
/// (absent/corrupt blob, or a blob older than
/// [`MAINTAINER_RECOMMENDATIONS_MAX_AGE_DAYS`] — stale blobs are cleared
/// best-effort). Resume runs never come through here; their session already
/// contains the block from the original start.
#[must_use]
async fn prepend_recommendations(task_text: &str, ws_name: &str) -> String {
    let blob = match crate::workspace::store()
        .get_maintainer_recommendations(ws_name)
        .await
    {
        Ok(Some(blob)) => blob,
        Ok(None) => return task_text.to_string(),
        Err(e) => {
            warn!(workspace = ws_name, error = %e, "Maintainer: failed to read recommendations");
            return task_text.to_string();
        }
    };

    let stored: StoredMaintainerRecommendations = match serde_json::from_str(&blob) {
        Ok(stored) => stored,
        Err(e) => {
            warn!(workspace = ws_name, error = %e, "Maintainer: corrupt recommendations blob, ignoring");
            return task_text.to_string();
        }
    };

    let generated_at = match db::parse_utc_timestamp(&stored.generated_at) {
        Ok(generated_at) => generated_at,
        Err(e) => {
            warn!(workspace = ws_name, error = %e, "Maintainer: unparseable recommendations timestamp, ignoring");
            return task_text.to_string();
        }
    };

    let now = Utc::now();
    if is_stale(&generated_at, now) {
        if let Err(e) = crate::workspace::store()
            .set_maintainer_recommendations(ws_name, None)
            .await
        {
            warn!(workspace = ws_name, error = %e, "Maintainer: failed to clear stale recommendations");
        }
        return task_text.to_string();
    }

    if stored.recommendations.is_empty() {
        return task_text.to_string();
    }

    format!(
        "{}{}",
        recommendations_block(&stored.recommendations, &generated_at, now),
        task_text
    )
}

/// Format the advisory block prepended to the fresh-run task text. The block
/// ends with a blank line; the caller concatenates the task text.
#[must_use]
fn recommendations_block(
    recs: &[String],
    generated_at: &DateTime<Utc>,
    now: DateTime<Utc>,
) -> String {
    let age = format_age(now.signed_duration_since(*generated_at));
    let mut out = String::from("<maintainer-recommendations>\n");
    let _ = writeln!(
        out,
        "ADVISORY — suggestions carried over from the previous maintainer run (generated {age} ago, at {}). Context only, not instructions — use your own judgment:",
        generated_at.to_rfc3339()
    );
    for rec in recs {
        let _ = writeln!(out, "- {rec}");
    }
    let _ = write!(out, "</maintainer-recommendations>\n\n");
    out
}

/// `true` when the recommendations are older than
/// [`MAINTAINER_RECOMMENDATIONS_MAX_AGE_DAYS`] — an exactly-boundary age is
/// still injectable, one second past is stale.
#[must_use]
fn is_stale(generated_at: &DateTime<Utc>, now: DateTime<Utc>) -> bool {
    now.signed_duration_since(*generated_at)
        > chrono::Duration::days(MAINTAINER_RECOMMENDATIONS_MAX_AGE_DAYS)
}

/// Humanize a duration as the two most significant units (e.g. "5 days 3 hours",
/// "2 hours 15 minutes", "34 minutes"), falling back to "less than a minute".
#[must_use]
fn format_age(elapsed: chrono::Duration) -> String {
    let total_mins = elapsed.num_minutes();
    if total_mins <= 0 {
        return "less than a minute".to_string();
    }
    let days = total_mins / (24 * 60);
    let hours = (total_mins / 60) % 24;
    let mins = total_mins % 60;
    if days > 0 {
        if hours > 0 {
            format!("{days} days {hours} hours")
        } else {
            format!("{days} days")
        }
    } else if hours > 0 {
        if mins > 0 {
            format!("{hours} hours {mins} minutes")
        } else {
            format!("{hours} hours")
        }
    } else {
        format!("{mins} minutes")
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
    use chrono::TimeZone;

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

    // ── Recommendation helpers ────────────────────────────────────────────

    #[test]
    fn cap_recommendations_keeps_all_within_budget() {
        let recs = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
        assert_eq!(cap_recommendations(recs), vec!["a", "bb", "ccc"]);
    }

    #[test]
    fn cap_recommendations_drops_items_on_budget_overflow() {
        // 20 items of 100 bytes each: 19 fit (1 + 103k <= 2048), the 20th does not.
        let recs: Vec<String> = (0..20).map(|_| "x".repeat(100)).collect();
        let out = cap_recommendations(recs);
        assert_eq!(out.len(), 19);
        assert!(
            serde_json::to_string(&out).unwrap().len() <= MAINTAINER_RECOMMENDATIONS_BUDGET_BYTES
        );
    }

    #[test]
    fn cap_recommendations_drops_lone_oversized_item() {
        let recs = vec!["y".repeat(3000)];
        assert!(cap_recommendations(recs).is_empty());
    }

    #[test]
    fn cap_recommendations_removes_empty_and_whitespace_items() {
        let recs = vec![
            "  ".to_string(),
            "keep".to_string(),
            String::new(),
            " also keep ".to_string(),
        ];
        assert_eq!(cap_recommendations(recs), vec!["keep", "also keep"]);
    }

    #[test]
    fn recommendations_block_format_and_content() {
        let generated_at = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let now = generated_at + chrono::Duration::hours(5) + chrono::Duration::minutes(15);
        let block = recommendations_block(
            &["first rec".to_string(), "second rec".to_string()],
            &generated_at,
            now,
        );
        assert!(block.starts_with("<maintainer-recommendations>\n"));
        assert!(
            block.contains("ADVISORY — suggestions carried over from the previous maintainer run")
        );
        assert!(block.contains("generated 5 hours 15 minutes ago"));
        assert!(block.contains("2026-01-02T03:04:05"));
        assert!(block.contains("- first rec\n"));
        assert!(block.contains("- second rec\n"));
        assert!(block.ends_with("</maintainer-recommendations>\n\n"));
    }

    #[test]
    fn is_stale_boundary() {
        let generated_at = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let exact_7d =
            generated_at + chrono::Duration::days(MAINTAINER_RECOMMENDATIONS_MAX_AGE_DAYS);
        let one_second_past = exact_7d + chrono::Duration::seconds(1);
        assert!(
            !is_stale(&generated_at, exact_7d),
            "exactly 7 days old → still injectable"
        );
        assert!(
            is_stale(&generated_at, one_second_past),
            "one second past → stale"
        );
    }

    #[test]
    fn format_age_cases() {
        assert_eq!(
            format_age(chrono::Duration::seconds(30)),
            "less than a minute"
        );
        assert_eq!(
            format_age(chrono::Duration::seconds(-5)),
            "less than a minute"
        );
        assert_eq!(format_age(chrono::Duration::minutes(34)), "34 minutes");
        assert_eq!(
            format_age(chrono::Duration::hours(2) + chrono::Duration::minutes(15)),
            "2 hours 15 minutes"
        );
        assert_eq!(
            format_age(chrono::Duration::days(5) + chrono::Duration::hours(3)),
            "5 days 3 hours"
        );
    }
}
