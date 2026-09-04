//! Dead-session recovery poller.
//!
//! Periodically checks all direct user-agent sessions for signs of silent
//! failure and automatically re-triggers the agent by routing a recovery job
//! through the message router. Three states are treated as dead:
//!
//! * **Unanswered user message** — the user sent a message, no agent
//!   responded, and no agent is running.
//! * **Unfinished turn (terminal tool frame)** — the agent committed a tool
//!   result but was cut (drain/restart) before producing the final assistant
//!   reply, so the last persisted message is a tool result. The tool result
//!   is already in the session history; the recovery re-run continues from
//!   there and delivers the missing answer.
//! * **Dangling tool-call frame** — the assistant tool-call frame is persisted
//!   at emission time (before the results), so a run cut (crash/drain)
//!   mid-tool-round leaves the tail an assistant frame whose calls have no
//!   results. The recovery re-run's universal `complete_pending_tool_calls`
//!   step settles the dangling calls deterministically (resuming durable jobs
//!   / re-executing) before the next LLM call, so recovery continues the
//!   interrupted work.
//!
//! # Exclusion list
//!
//! The poller skips `manager_*` (Manager has its own lifecycle) plus every
//! prefix in [`crate::session::TRANSIENT_AGENT_ID_PREFIXES`] (transient or
//! background-only agents).
//!
//! Only direct user-agent sessions (format `{user}_{ws}_{role}`, or the
//! deduped `{user}_personal:{role}` for the user's own personal workspace)
//! are eligible for recovery.
//!
//! # Retry safety design
//!
//! Every recovery attempt (regardless of whether routing succeeds) counts
//! toward the per-session retry cap.  There is no "success feedback" from
//! the consumer loop — the agent executes asynchronously, so routing
//! success ≠ recovery success.  A persistently failing agent (rate-limited,
//! API errors) will hit `MAX_RETRIES` after approximately 5.7 hours
//! (accounting for adaptive backoff: first wait ~20 min, ~40 min per
//! subsequent attempt) and be permanently given up.
//! The session becoming healthy is detected naturally on the next poll
//! cycle when the last message is an assistant reply (condition 1 no longer
//! holds).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::agent::message_router::{AgentJob, MessageKind};
use crate::agent::registry::AGENT_REGISTRY;
use crate::session::SessionContext;
use crate::{ChatRole, Role};

// ── Constants ───────────────────────────────────────────────────────────────

/// Interval between poller cycles.
const POLL_INTERVAL: Duration = Duration::from_mins(5); // 5 minutes

/// Maximum consecutive recovery attempts before giving up on a session.
const MAX_RETRIES: u8 = 10;

/// Base backoff in minutes.  Doubles on each consecutive failure, capped at
/// `MAX_BACKOFF_MINUTES`.  This is the effective minimum delay between
/// recovery attempts (the `or_insert` default is immediately doubled).
const BASE_BACKOFF_MINUTES: i64 = 10;

/// Maximum backoff in minutes.
const MAX_BACKOFF_MINUTES: i64 = 40;

/// Grace period: only attempt recovery if the session's last candidate
/// message (user message, terminal tool frame, or dangling tool-call frame)
/// is at least this old.
/// Prevents races with agents that are still spawning or loading.
const STALE_GRACE_PERIOD: Duration = Duration::from_mins(5);

// ── Retry tracker (in-memory) ───────────────────────────────────────────────

struct RetryState {
    attempt_count: u8,
    last_attempt_at: DateTime<Utc>,
    backoff_minutes: i64,
}

/// Tracks recovery attempts per agent ID.
///
/// In-memory only — reset on daemon restart, which is consistent with the
/// "no startup grace period" reasoning in the ticket: after restart, all
/// sessions are treated as fresh, and the retry counter starts at zero.
///
/// ## Entry lifecycle
///
/// * An entry is created on the first recovery attempt (`record_attempt`).
/// * If the session self-heals (agent responds), the entry is removed by
///   `cleanup()` — the retry cap counts *consecutive* failures per episode.
/// * If the session exhausts `MAX_RETRIES`, the entry persists permanently,
///   causing `should_retry` to return `false` for the remainder of the
///   daemon's lifetime.
/// * On daemon restart all entries are lost, giving every session a fresh
///   retry budget.
pub(crate) struct DeadSessionTracker {
    inner: Mutex<HashMap<String, RetryState>>,
}

impl DeadSessionTracker {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Check whether a session is eligible for another recovery attempt.
    ///
    /// Returns `true` if the session has not exceeded the retry cap AND
    /// the adaptive backoff period has elapsed since the last attempt.
    fn should_retry(&self, agent_id: &str) -> bool {
        let map = self.inner.lock().expect("DeadSessionTracker lock poisoned");
        match map.get(agent_id) {
            Some(state) => {
                if state.attempt_count >= MAX_RETRIES {
                    return false;
                }
                let elapsed = (Utc::now() - state.last_attempt_at).num_minutes();
                elapsed >= state.backoff_minutes
            }
            None => true,
        }
    }

    /// Record a recovery attempt (increments counter, updates backoff).
    ///
    /// Called after **every** recovery attempt, regardless of whether the
    /// job was routed successfully.  This ensures the `MAX_RETRIES` cap
    /// and adaptive backoff work correctly for the primary failure mode
    /// (persistent agent failures like rate limits or API errors), since
    /// routing success ≠ recovery success.
    fn record_attempt(&self, agent_id: &str) {
        let mut map = self.inner.lock().expect("DeadSessionTracker lock poisoned");
        let state = map.entry(agent_id.to_string()).or_insert(RetryState {
            attempt_count: 0,
            last_attempt_at: Utc::now(),
            backoff_minutes: BASE_BACKOFF_MINUTES,
        });
        state.attempt_count += 1;
        state.last_attempt_at = Utc::now();
        state.backoff_minutes = (state.backoff_minutes * 2).min(MAX_BACKOFF_MINUTES);
    }

    /// Remove the tracking entry for a session that has self-healed.
    ///
    /// Called when the poller detects that a session is healthy (the last
    /// message is an assistant reply — the turn completed).  This ensures the
    /// retry cap counts *consecutive* failures per episode, matching the
    /// ticket spec — a session that fails, recovers, then fails again starts
    /// with a fresh retry budget.
    ///
    /// Safe to call for untracked sessions (no-op).
    fn cleanup(&self, agent_id: &str) {
        self.inner
            .lock()
            .expect("DeadSessionTracker lock poisoned")
            .remove(agent_id);
    }

    /// Check whether a session has exhausted all retries (for logging).
    #[cfg(test)]
    fn has_exhausted_retries(&self, agent_id: &str) -> bool {
        let map = self.inner.lock().expect("DeadSessionTracker lock poisoned");
        map.get(agent_id)
            .is_some_and(|s| s.attempt_count >= MAX_RETRIES)
    }
}

// ── Global instance ─────────────────────────────────────────────────────────

pub(crate) static DEAD_SESSION_TRACKER: std::sync::LazyLock<DeadSessionTracker> =
    std::sync::LazyLock::new(DeadSessionTracker::new);

// ── Poller loop ─────────────────────────────────────────────────────────────

/// Run the dead-session recovery poller loop.
///
/// Follows the established background-task pattern: cooperative shutdown via
/// [`crate::shutdown::sleep_or_shutdown_or_drain`], single-failure isolation
/// via `tracing::warn!`.
pub async fn run_dead_session_recovery_loop() {
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(POLL_INTERVAL).await {
            break;
        }
        if let Err(e) = recover_dead_sessions().await {
            tracing::warn!(error = %e, "Dead session recovery poller failed");
        }
    }
}

// ── Core detection + recovery logic ─────────────────────────────────────────

/// Classify a session by its last persisted message role (and, for an
/// Assistant tail, its raw content): is it a dead-session recovery candidate?
///
/// Candidates:
///
/// * `User` tail — the user sent a message and the agent never answered.
/// * `Tool` tail — the agent committed a tool result but was cut (drain or
///   restart) before producing the final assistant reply, so the turn is
///   unfinished. The tool result is already in the session history; the
///   recovery re-run continues from it and delivers the missing answer.
///   There is no user-facing cancel path in the codebase, so a tool-frame
///   tail cannot mean "cancelled" — resuming is safe.
/// * `Assistant` tail that is a dangling emission-time tool-call frame
///   (`last_content` = the tail row's raw content). A tail frame can have no
///   following result rows, so non-empty calls there means unresolved —
///   exactly what the consumer-side `complete_pending_tool_calls` settles.
///
/// Healthy (not candidates):
///
/// * `Assistant` tail with any other content — the turn completed; the agent
///   already responded.
/// * `System` tail — no in-flight turn.
///
/// `last_content` is only consulted for the `Assistant` arm.
fn is_recovery_candidate(last_role: ChatRole, last_content: &str) -> bool {
    match last_role {
        ChatRole::User | ChatRole::Tool => true,
        ChatRole::Assistant => super::is_dangling_tool_call_frame(last_role, last_content),
        ChatRole::System => false,
    }
}

/// Check all sessions for dead direct user-agent sessions and route recovery
/// jobs where needed.
async fn recover_dead_sessions() -> anyhow::Result<()> {
    let now = Utc::now();

    // Use SQL-side filtering to avoid loading excluded sessions (manager_ and
    // every prefix in TRANSIENT_AGENT_ID_PREFIXES) from the database.  The
    // per-session `get_last_message_tail` queries below are lightweight
    // (indexed `ORDER BY id DESC LIMIT 1`) — fetching both the role and the
    // content of the tail row in a single read — and only run for eligible
    // sessions.
    let sessions = crate::session::store()
        .list_sessions_with_metadata_excluding(
            &crate::session::reserved_agent_id_prefixes().collect::<Vec<&str>>(),
        )
        .await;

    for session in &sessions {
        let agent_id = &session.agent_id;

        // ── Condition 1: the tail is a recovery candidate ───────────────
        let Some((last_role, tail_content)) = crate::session::store()
            .get_last_message_tail(agent_id)
            .await
        else {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue; // empty session — clean up any stale tracker entry
        };
        if !is_recovery_candidate(last_role, &tail_content) {
            // Last message is an assistant reply that is not a dangling
            // frame, or a system message: the turn is complete — the agent
            // has already responded. Clean up the retry-tracking entry so a
            // future failure starts with a fresh retry budget
            // (consecutive-failure-per-episode, matching the ticket spec).
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue;
        }
        // An Assistant tail is a candidate only via the dangling-frame path —
        // remember that for the pause guard below.
        let needs_pause_guard = last_role == ChatRole::Assistant;

        // ── Condition 2: no live agent ─────────────────────────────────
        if AGENT_REGISTRY.contains(agent_id) {
            // Agent is still running (or was just spawned).
            continue;
        }

        // ── Condition 3: candidate message is stale ────────────────────
        let age = now - session.last_activity;
        let grace = chrono::Duration::from_std(STALE_GRACE_PERIOD)
            .expect("STALE_GRACE_PERIOD fits in chrono::Duration");
        if age < grace {
            // Too recent — give the agent time to start.
            continue;
        }

        // ── Condition 4: sleep-ended sessions ended their turns deliberately ──
        if crate::session::store().get_sleep_ended(agent_id).await {
            // The Assistant ended its turn via the `sleep` tool: the tool tail
            // is an intentional stop while waiting for new input, not a crash —
            // never recover it. Unmarked tool tails (real crash/drain cuts)
            // keep recovering exactly as before. The session has self-healed:
            // reset any stale retry budget from an earlier episode so a later
            // genuine crash on this session recovers with a fresh cap
            // (consecutive-failures-per-episode semantics).
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue;
        }

        // ── Safety guard: retry cap / adaptive backoff ─────────────────
        if !DEAD_SESSION_TRACKER.should_retry(agent_id) {
            // Either the session exhausted its retries (the entry persists
            // in the tracker permanently, blocking future attempts) or the
            // adaptive backoff hasn't elapsed yet (will be rechecked on the
            // next poll cycle).  Either way — silently skip.
            continue;
        }

        // ── Recover! ───────────────────────────────────────────────────
        //
        // Phase 1: retrieve session context and validate it.  Permanent
        // data issues (missing context, invalid role) are non-recoverable —
        // clean up the tracker entry and move on without consuming the
        // retry budget.
        let Some(ctx) = crate::session::store().get_session_context(agent_id).await else {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            tracing::warn!(
                agent_id = %agent_id,
                "Dead session recovery: no context found \
                 (corrupted data) — skipping permanently"
            );
            continue;
        };

        let Ok(role) = ctx.role.parse::<Role>() else {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            tracing::warn!(
                agent_id = %agent_id,
                role = %ctx.role,
                "Dead session recovery: invalid role in session context — \
                 skipping permanently"
            );
            continue;
        };

        // ── Pause guard (dangling-frame resumption only) ────────────────
        // A paused workspace is a strict freeze: auto-resuming interrupted work
        // the user explicitly paused must not happen (unpausing re-drives via
        // the puller / next user prompt). A frozen agent is not recoverable
        // while registered; this catches the tail it leaves after bailing.
        // User/Tool tails stay exempt: answering a pending user prompt matches
        // normal direct-chat behavior, which a workspace pause does not freeze.
        if needs_pause_guard {
            match crate::workspace::get_by_name(&ctx.workspace_name).await {
                Ok(Some(ws)) if ws.paused => {
                    tracing::debug!(
                        agent_id = %agent_id,
                        workspace = %ctx.workspace_name,
                        "Dead session recovery: workspace paused — skipping"
                    );
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(
                        agent_id = %agent_id,
                        error = %e,
                        "Dead session recovery: workspace lookup failed — skipping this cycle"
                    );
                    continue;
                }
            }
        }

        // Phase 2: route the recovery job.  This is fire-and-forget
        // (sends on an mpsc channel) — routing success ≠ recovery
        // success.  The agent executes asynchronously and may still
        // fail (rate limits, API errors, etc.), so we count every
        // routed attempt against the retry cap.
        //
        // Failures in Phase 1 (missing context, invalid role) are NOT
        // counted — they are permanent data issues that will never
        // resolve with retries.
        attempt_recovery(agent_id, &ctx, role);

        DEAD_SESSION_TRACKER.record_attempt(agent_id);
    }

    Ok(())
}

/// Returns `true` if the agent ID belongs to a session that should NOT be
/// recovered by the poller (`manager_` plus every
/// [`crate::session::TRANSIENT_AGENT_ID_PREFIXES`] prefix).
///
/// Note: the poller itself uses SQL-side filtering now — this function is
/// retained for test coverage and as documentation of the exclusion criteria.
#[cfg(test)]
fn is_excluded_agent_id(agent_id: &str) -> bool {
    crate::session::reserved_agent_id_prefixes().any(|p| {
        let bare = p.trim_end_matches('_');
        // Faithful to SQL `LIKE 'prefix_%'`: the `_` consumes exactly one char
        // after the bare reserved word, so a bare word alone is NOT excluded.
        agent_id.len() > bare.len() && crate::session::starts_with_ignore_ascii_case(agent_id, bare)
    })
}

/// Route a recovery job for a dead session with validated context.
///
/// # Preconditions
///
/// Caller must have already called [`get_session_context`](crate::session::SessionStore::get_session_context)
/// and validated that the role can be parsed.  The routing is fire-and-forget
/// (sends on an mpsc channel).
fn attempt_recovery(agent_id: &str, ctx: &SessionContext, role: Role) {
    // Recovery retry: uses `RecoveryRetry` kind with empty content so the
    // agent runs against the EXISTING session history (which already contains
    // the user's original message).  No boilerplate message is injected, and
    // no emoji error feedback is sent on retry failures — the emoji fires
    // only once on the original failure.
    let job = AgentJob {
        content: String::new(),
        workspace_name: ctx.workspace_name.clone(),
        user_name: ctx.user_name.clone(),
        channel: ctx.channel.clone(),
        kind: MessageKind::RecoveryRetry,
        role,
        // reply_target is not available from session metadata — the recovery
        // response will be persisted to chat_history and broadcast via the
        // GUI channel, but may not be deliverable via Telegram to unregistered
        // users.  This is an accepted limitation.
        reply_target: None,
        pending_job_id: None,
        reply_to_agent_id: None,
        reply_workspace_name: None,
    };

    tracing::info!(
        agent_id = %agent_id,
        role = %role.as_str(),
        workspace = %ctx.workspace_name,
        user = %ctx.user_name,
        channel = %ctx.channel,
        "Dead session recovery: routing retry job"
    );

    crate::agent::message_router::route(agent_id, job);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;

    #[test]
    fn test_is_excluded_agent_id_cases() {
        struct Case {
            name: &'static str,
            id: String,
            excluded: bool,
        }

        // Dynamic prefix drift-detection invariant: every registered reserved
        // prefix must be excluded.  Enumerating these as literal table rows
        // would silently lose coverage when TRANSIENT_AGENT_ID_PREFIXES grows
        // (or `manager_` changes), so the union is enumerated at runtime via
        // `reserved_agent_id_prefixes()` — the same source the poller uses.
        for prefix in crate::session::reserved_agent_id_prefixes() {
            let id = format!("{prefix}suffix");
            assert!(
                is_excluded_agent_id(&id),
                "case: transient prefix '{prefix}' — expected '{id}' to be excluded",
            );
        }

        let cases = vec![
            // Direct user-agent sessions should NOT be excluded by the union of
            // `TRANSIENT_AGENT_ID_PREFIXES` and `"manager_"`.
            Case {
                name: "direct_session",
                id: "alice_main_workspace_engineer".into(),
                excluded: false,
            },
            Case {
                name: "direct_session_ws_2",
                id: "bob_my_project_analyst".into(),
                excluded: false,
            },
            Case {
                name: "direct_session_ws_3",
                id: "charlie_personal_work_assistant".into(),
                excluded: false,
            },
            // Even if user/workspace names contain underscores, the start of the
            // agent_id is the user name ("some_user"), which doesn't match any
            // excluded prefix.
            Case {
                name: "underscore_in_names",
                id: "some_user_my_cool_workspace_reviewer".into(),
                excluded: false,
            },
            // A real user whose name collides with a reserved prefix is escaped by
            // `direct_agent_id` (a `user_` prefix), so their session is never
            // mistaken for a transient/background session and is never skipped by
            // the poller.  These are runtime-built (cannot be `&'static str`
            // rows) so the live link to the `user_`-escape coupling stays honest.
            Case {
                name: "colliding_user_manager",
                id: crate::session::direct_agent_id("manager", "engineer", "ws"),
                excluded: false,
            },
            Case {
                name: "colliding_user_ticket",
                id: crate::session::direct_agent_id("ticket_bob", "analyst", "ws"),
                excluded: false,
            },
            // The production SQL exclusion is `LIKE 'prefix_%'` (case-insensitive
            // for ASCII), so a case-variant reserved word is also excluded here.
            Case {
                name: "case_variant_ticket_upper",
                id: "Ticket_suffix".into(),
                excluded: true,
            },
            Case {
                name: "case_variant_ticket_mixed",
                id: "tIcKeT_suffix".into(),
                excluded: true,
            },
            Case {
                name: "case_variant_manager",
                id: "Manager_bob".into(),
                excluded: true,
            },
        ];

        for case in &cases {
            assert_eq!(
                is_excluded_agent_id(&case.id),
                case.excluded,
                "case: {name} — id='{id}'",
                name = case.name,
                id = case.id,
            );
        }
    }

    #[test]
    fn test_is_recovery_candidate_classification() {
        // User tail: user sent a message, agent never answered → candidate.
        assert!(is_recovery_candidate(ChatRole::User, ""));
        // Tool tail: committed tool frame without a following assistant reply
        // (turn cut by a drain/restart) → candidate — the recovery re-run
        // continues from the tool result already in the history.
        assert!(is_recovery_candidate(ChatRole::Tool, ""));
        // Assistant tail — only a dangling emission-time tool-call frame is a
        // candidate (interruption mid-tool-round); anything else is healthy.
        // Plain assistant reply: turn completed, agent responded → healthy.
        assert!(!is_recovery_candidate(
            ChatRole::Assistant,
            "plain assistant reply"
        ));
        // Hand-crafted non-frame JSON → healthy (not a tool-call frame).
        assert!(!is_recovery_candidate(
            ChatRole::Assistant,
            r#"{"answer": 42}"#
        ));
        // Emission-time tool-call frame with calls (the real dangling tail
        // encoding) → candidate.
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_frame".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.txt"}),
            }],
            None,
        )
        .to_string();
        assert!(is_recovery_candidate(ChatRole::Assistant, &frame));
        // Hand-crafted EMPTY-calls frame → healthy (must not false-positive;
        // `assistant_replay_payload` with an empty slice omits the key
        // entirely, hence the hand-crafted JSON).
        assert!(!is_recovery_candidate(
            ChatRole::Assistant,
            r#"{"content":"","tool_calls":[]}"#
        ));
        // Invalid JSON → healthy.
        assert!(!is_recovery_candidate(
            ChatRole::Assistant,
            "not json at all"
        ));
        // System tail: no in-flight turn → healthy.
        assert!(!is_recovery_candidate(ChatRole::System, ""));
    }

    #[test]
    fn test_dead_session_tracker_max_retries() {
        let tracker = DeadSessionTracker::new();
        let agent_id = "test_agent_engineer";

        // First attempt should be allowed for brand-new session
        assert!(tracker.should_retry(agent_id));
        tracker.record_attempt(agent_id);

        // Immediately after an attempt, backoff prevents retry
        assert!(!tracker.should_retry(agent_id));

        // Skip the backoff check for remaining MAX_RETRIES-1 attempts
        // by directly calling record_attempt (simulating that time has passed).
        for _ in 2..=MAX_RETRIES {
            tracker.record_attempt(agent_id);
        }

        // After max retries reached, should_retry returns false
        // regardless of backoff state.
        assert!(
            !tracker.should_retry(agent_id),
            "should be blocked after {MAX_RETRIES} attempts"
        );
    }

    #[test]
    fn test_dead_session_tracker_cleanup() {
        let tracker = DeadSessionTracker::new();
        let agent_id = "test_cleanup_engineer";

        // Record an attempt — entry created, backoff blocks retry
        tracker.record_attempt(agent_id);
        assert!(!tracker.should_retry(agent_id));

        // Cleanup removes the entry, allowing a fresh start
        tracker.cleanup(agent_id);
        assert!(tracker.should_retry(agent_id));
    }

    #[test]
    fn test_dead_session_tracker_exhaustion_is_permanent() {
        let tracker = DeadSessionTracker::new();
        let agent_id = "test_permanent_engineer";

        // Record MAX_RETRIES attempts
        for _ in 0..MAX_RETRIES {
            tracker.record_attempt(agent_id);
        }

        // should_retry returns false permanently — the entry stays in the
        // map, and no removal mechanism resets the counter (a previous
        // removal mechanism had a critical bug and was deliberately removed).
        assert!(!tracker.should_retry(agent_id));
        // Verify it's still blocked on a second check (not a transient
        // failure due to backoff timing)
        assert!(!tracker.should_retry(agent_id));
        // Verify has_exhausted_retries agrees
        assert!(tracker.has_exhausted_retries(agent_id));
    }

    #[test]
    fn test_dead_session_tracker_backoff_and_reset_on_restart() {
        let tracker = DeadSessionTracker::new();
        let agent_id = "test_backoff_engineer";

        // First attempt should be allowed (fresh session)
        assert!(tracker.should_retry(agent_id));
        tracker.record_attempt(agent_id);

        // Immediately after recording, backoff prevents retry
        assert!(!tracker.should_retry(agent_id));

        // Simulate daemon restart: create a new tracker (fresh state).
        let tracker2 = DeadSessionTracker::new();
        assert!(
            tracker2.should_retry(agent_id),
            "should be fresh after simulated restart"
        );
    }

    // ── End-to-end recovery tests ───────────────────────────────────────
    //
    // These three tests share the process-global test DB, the retry tracker,
    // the agent registry, and the drain flag, so each holds
    // `retry_tests_lock()` for its full duration (the e2e test acquires it via
    // its `install_retry_seam_dyn` guard).

    /// Seed a direct session whose tail is a dangling emission-time analyze
    /// tool-call frame plus the caller-owned launched durable job, checkpointed
    /// as if a prior run had completed the analyst round. Returns the `job_id`.
    ///
    /// This is the shared scaffolding for the e2e/live/paused recovery tests:
    /// it persists the frame + context, backdates `last_activity` past the
    /// grace period, drives the sync analyze dispatch under a drain (leaving
    /// the job `launched`, the frame's call result-less), then checkpoints the
    /// roster outcomes as the durable resume input.
    #[expect(clippy::too_many_lines)] // deliberate: seeding the durable-job state is one cohesive fixture
    async fn seed_dangling_durable_session(agent_id: &str, ws: &crate::Workspace) -> String {
        let conn = &crate::session::store().conn;

        // Emission-time frame: the caller's session durably records the analyze
        // call BEFORE the drain-cut execution leaves its result absent.
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_analyze_e2e".to_string(),
                name: "analyze".to_string(),
                arguments: serde_json::json!({"analyze": "analyze this"}),
            }],
            None,
        )
        .to_string();
        crate::session::store()
            .append_messages(
                agent_id,
                &[
                    crate::ChatMessage::user("run the analysis"),
                    crate::ChatMessage::assistant(frame),
                ],
                false,
                Some(("gui", "e2e_user", ws.name.as_str(), "engineer")),
            )
            .await
            .unwrap();

        // Backdate the session so the 5-minute grace period passes.
        let past = (chrono::Utc::now() - chrono::Duration::minutes(10)).to_rfc3339();
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = ?2",
            crate::db::params![past, agent_id],
        )
        .await
        .unwrap();

        // Drive the sync analyze dispatch under the caller's task-local pin with
        // an active drain: it leaves the durable job launched (DrainCut).
        crate::shutdown::drain_begin();
        let tool = crate::tools::analyze::AnalyzeTool::new(
            crate::tools::analyze::DispatchMode::Sync,
            crate::Role::Engineer,
        );
        let res = crate::agent::CURRENT_TOOL_AGENT_ID
            .scope(Some(agent_id.to_string()), async {
                tool.execute(ws, serde_json::json!({"analyze": "analyze this"}))
                    .await
            })
            .await;
        crate::shutdown::drain_clear();
        let err = res.expect_err("drain must cut the sync analyze dispatch");
        assert!(
            err.downcast_ref::<crate::tools::CallSuspended>().is_some(),
            "CallSuspended carrier expected: {err:#}"
        );

        // The frame row is persisted, NO tool-result row, job launched caller-owned.
        let jobs = conn
            .query(
                "SELECT id, status, caller_agent_id FROM jobs WHERE caller_agent_id = ?1 AND kind = 'analyze'",
                crate::db::params![agent_id],
            )
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1, "one launched analyze job");
        assert_eq!(jobs[0].get::<String>(1).unwrap(), "launched");
        assert_eq!(
            jobs[0].get::<String>(2).unwrap(),
            agent_id,
            "job is caller-owned by the session pin"
        );
        let job_id = jobs[0].get::<String>(0).unwrap();
        let session_rows = conn
            .query(
                "SELECT role FROM sessions WHERE agent_id = ?1 ORDER BY id",
                crate::db::params![agent_id],
            )
            .await
            .unwrap();
        let roles: Vec<String> = session_rows
            .iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            roles,
            vec!["user", "assistant"],
            "frame persisted with NO tool-result row: {roles:?}"
        );

        // Simulate the durable checkpoint: exactly one analyst slot reconstructed
        // as Done (its outcome) while the other slots produced no valid response.
        let roster = crate::jobs::list_agents_for_job(conn, &job_id)
            .await
            .unwrap();
        assert!(
            roster.len() >= 2,
            "analyze round spawns multiple analysts: {}",
            roster.len()
        );
        for (i, row) in roster.iter().enumerate() {
            let outcome = if i == 0 { "ANALYST_RAW" } else { "" };
            crate::jobs::write_agent_outcome(
                conn,
                &job_id,
                &row.agent_id,
                crate::jobs::RowStatus::Done,
                Some(outcome),
            )
            .await
            .unwrap();
        }

        job_id
    }

    /// Full poller e2e: a dangling emission-time analyze frame (drain-cut
    /// mid-tool-round) is classified as a recovery candidate, the recovery
    /// re-run routes a RecoveryRetry, the consumer's universal
    /// `complete_pending_tool_calls` resumes the durable job LLM-free, and only
    /// then does the recovery round call the model once — asserted to exactly
    /// one fingerprint.
    #[tokio::test]
    #[serial_test::serial(provider)]
    // Joins the `provider` group (order: provider → retry_tests_lock, as in the
    // pipeline/retry tests) so two per-test runtimes never drive the shared
    // global stores concurrently.
    async fn recovery_resumes_dangling_durable_job_end_to_end() {
        let fake = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                .ok("Recovery round: analysis resumed and complete."),
        );
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_management_test_stores().await;
        // The router's consumer-loop path (typing setup / unregistered delivery)
        // requires the channel registry. A default (empty) registry is enough:
        // there is no telegram channel and the recovery response is persisted
        // without a transport.
        let _ = crate::CHANNEL_REGISTRY.set(crate::ChannelRegistry::default());
        let ws = crate::util::test::create_test_workspace(
            "/tmp/dead_session_e2e_ws",
            "dead_session_e2e_ws",
        )
        .await;
        let agent_id = format!("e2e_user_{}_engineer", ws.name);
        let conn = &crate::session::store().conn;
        let job_id = seed_dangling_durable_session(&agent_id, &ws).await;

        // Recover: classify the dangling-frame session, route a RecoveryRetry.
        recover_dead_sessions().await.unwrap();

        // Poll until the resumed job is terminalized AND the session tail is
        // the recovery-round assistant reply.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            let jobs = conn
                .query(
                    "SELECT id FROM jobs WHERE id = ?1",
                    crate::db::params![job_id.clone()],
                )
                .await
                .unwrap();
            let tail = crate::session::store()
                .get_last_message_tail(&agent_id)
                .await;
            let tail_role = tail.as_ref().map(|(role, _)| *role);
            let tail_has_reply = tail
                .as_ref()
                .is_some_and(|(_, content)| content.contains("Recovery round"));
            if jobs.is_empty() && tail_role == Some(crate::ChatRole::Assistant) && tail_has_reply {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "recovery did not complete within 60s: jobs={jobs:?} \
                 tail_role={tail_role:?} tail_has_reply={tail_has_reply}",
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Final: tool result sits right before the reply, carrying the original
        // call id and the checkpointed outcome; EXACTLY one model call (the
        // durable resume sub-phase was LLM-free); no pending envelope remains.
        let last2 = conn
            .query(
                "SELECT role, content FROM sessions WHERE agent_id = ?1 ORDER BY id DESC LIMIT 2",
                crate::db::params![agent_id.as_str()],
            )
            .await
            .unwrap();
        assert_eq!(last2.len(), 2, "tail has tool-result + reply: {last2:?}");
        let reply_row = &last2[0];
        let tool_row = &last2[1];
        assert_eq!(tool_row.get::<String>(0).unwrap(), "tool");
        let payload: crate::ToolResultPayload =
            serde_json::from_str(&tool_row.get::<String>(1).unwrap()).unwrap();
        assert_eq!(payload.tool_call_id, "call_analyze_e2e");
        assert!(
            payload.content.contains("ANALYST_RAW"),
            "checkpointed outcome recorded: {}",
            payload.content
        );
        assert!(
            reply_row
                .get::<String>(1)
                .unwrap()
                .contains("Recovery round"),
            "recovery round reply persisted"
        );
        assert_eq!(
            fake.request_fingerprints.lock().unwrap().len(),
            1,
            "durable resume sub-phase was LLM-free; the recovery round called the model once"
        );
        let pending_jobs = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id.clone()],
            )
            .await
            .unwrap();
        assert!(pending_jobs.is_empty(), "no envelope pending after resume");
    }

    /// A live (registered) agent blocks recovery at the registry gate: the
    /// dangling durable job is NOT resumed and no attempt is recorded — the
    /// session is still eligible on a later cycle.
    #[tokio::test]
    #[serial_test::serial(provider)] // see recovery_resumes_dangling_durable_job_end_to_end
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global drain flag across the whole test
    async fn live_running_session_is_not_recovered() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            "/tmp/dead_session_live_ws",
            "dead_session_live_ws",
        )
        .await;
        let agent_id = format!("e2e_user_{}_engineer", ws.name);
        let conn = &crate::session::store().conn;
        let job_id = seed_dangling_durable_session(&agent_id, &ws).await;

        let generation = AGENT_REGISTRY.register(
            agent_id.clone(),
            "engineer".into(),
            None,
            &ws,
            tokio_util::sync::CancellationToken::new(),
            None,
            None,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        );

        recover_dead_sessions().await.unwrap();

        // The live agent blocked Condition 2 before Phase 2: no resume, no
        // attempt recorded.
        let jobs = conn
            .query(
                "SELECT status FROM jobs WHERE id = ?1",
                crate::db::params![job_id.clone()],
            )
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1, "job must not be terminalized");
        assert_eq!(jobs[0].get::<String>(0).unwrap(), "launched");
        assert!(
            DEAD_SESSION_TRACKER.should_retry(&agent_id),
            "no recovery attempt recorded for a live session"
        );

        AGENT_REGISTRY.deregister(&agent_id, generation);

        // Clean up the seeded dangling session + launched job so a LATER
        // `recover_dead_sessions` (e.g. the paused test, serialized after us by
        // `retry_tests_lock`) does not route a recovery for this leftover
        // candidate and consume the shared fake provider mid-flight.
        conn.execute(
            "DELETE FROM sessions WHERE agent_id = ?1",
            crate::db::params![agent_id.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "DELETE FROM session_metadata WHERE agent_id = ?1",
            crate::db::params![agent_id.as_str()],
        )
        .await
        .unwrap();
        for row in conn
            .query(
                "SELECT agent_id FROM agents WHERE job_id = ?1",
                crate::db::params![job_id.clone()],
            )
            .await
            .unwrap()
        {
            let a_id: String = row.get(0).unwrap();
            conn.execute(
                "DELETE FROM agents WHERE job_id = ?1 AND agent_id = ?2",
                crate::db::params![job_id.clone(), a_id],
            )
            .await
            .unwrap();
        }
        conn.execute(
            "DELETE FROM jobs WHERE id = ?1",
            crate::db::params![job_id.clone()],
        )
        .await
        .unwrap();
    }

    /// A paused workspace freezes dangling-frame resumption: the pause guard
    /// skips the session before routing (no budget consumed, no job resumed).
    #[tokio::test]
    #[serial_test::serial(provider)] // see recovery_resumes_dangling_durable_job_end_to_end
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global drain flag across the whole test
    async fn paused_workspace_dangling_frame_is_not_recovered() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            "/tmp/dead_session_paused_ws",
            "dead_session_paused_ws",
        )
        .await;
        let agent_id = format!("e2e_user_{}_engineer", ws.name);
        let conn = &crate::session::store().conn;
        let job_id = seed_dangling_durable_session(&agent_id, &ws).await;

        // Pause the workspace — the strict freeze that must not be auto-resumed.
        conn.execute(
            "UPDATE workspaces SET paused = 1 WHERE name = ?1",
            crate::db::params![ws.name],
        )
        .await
        .unwrap();

        recover_dead_sessions().await.unwrap();

        // The pause guard skipped before Phase 2: no resume, no attempt recorded.
        let jobs = conn
            .query(
                "SELECT status FROM jobs WHERE id = ?1",
                crate::db::params![job_id.clone()],
            )
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1, "job must not be terminalized");
        assert_eq!(jobs[0].get::<String>(0).unwrap(), "launched");
        assert!(
            DEAD_SESSION_TRACKER.should_retry(&agent_id),
            "no recovery attempt recorded for a paused workspace"
        );
    }

    /// A sleep-ended session is excluded from recovery: the tool tail is an
    /// intentional turn end (the Assistant waiting for new input), so the
    /// poller must not re-invoke it or consume retry budget. Unmarked tool
    /// tails (real crash/drain cuts) keep recovering as before.
    #[tokio::test]
    #[serial_test::serial(provider)] // see recovery_resumes_dangling_durable_job_end_to_end
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the shared store's recover_dead_sessions runs across tests
    async fn sleep_ended_session_is_not_recovered() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::util::test::init_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            "/tmp/dead_session_sleep_ws",
            "dead_session_sleep_ws",
        )
        .await;
        let agent_id = format!("e2e_user_{}_engineer", ws.name);
        let conn = &crate::session::store().conn;

        // Tool tail: a committed tool result is the last row — a recovery
        // candidate for unmarked sessions.
        crate::session::store()
            .append_messages(
                &agent_id,
                &[
                    crate::ChatMessage::user("hello"),
                    crate::ChatMessage::tool_result("call_sleep", "Zzz..."),
                ],
                false,
                Some(("gui", "e2e_user", ws.name.as_str(), "assistant")),
            )
            .await
            .unwrap();
        let past = (chrono::Utc::now() - chrono::Duration::minutes(10)).to_rfc3339();
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = ?2",
            crate::db::params![past, agent_id.as_str()],
        )
        .await
        .unwrap();
        crate::session::store()
            .set_sleep_ended(&agent_id, true)
            .await
            .unwrap();

        recover_dead_sessions().await.unwrap();

        // Condition 4 skipped the session BEFORE the retry-budget guard: no
        // attempt recorded, the flag survives the poll cycle.
        assert!(
            DEAD_SESSION_TRACKER.should_retry(&agent_id),
            "no recovery attempt recorded for a sleep-ended session"
        );
        assert!(
            crate::session::store().get_sleep_ended(&agent_id).await,
            "sleep-ended flag must survive the poll cycle"
        );

        // Clean up the seeded candidate so a LATER recover_dead_sessions
        // (serialized after us by retry_tests_lock) does not route a recovery
        // for this leftover session.
        conn.execute(
            "DELETE FROM sessions WHERE agent_id = ?1",
            crate::db::params![agent_id.as_str()],
        )
        .await
        .unwrap();
        conn.execute(
            "DELETE FROM session_metadata WHERE agent_id = ?1",
            crate::db::params![agent_id.as_str()],
        )
        .await
        .unwrap();
    }
}
