//! Dead-session recovery poller.
//!
//! Periodically checks all direct user-agent sessions for signs of silent
//! failure (user sent a message, no agent responded, and no agent is running)
//! and automatically re-triggers the agent by routing a recovery job through
//! the message router.
//!
//! # Exclusion list
//!
//! The poller skips agent IDs matching these prefixes:
//! - `manager_*` — Manager has its own lifecycle.
//! - `ticket_*`, `ask_*`, `maintainer_*`, `discovery_*` — transient or
//!   background-only agents (see [`TRANSIENT_AGENT_ID_PREFIXES`]).
//!
//! Only direct user-agent sessions (format `{channel}_{user}_{ws}_{role}`)
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
//! cycle when condition 1 (last message is user) no longer holds.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::message_router::{AgentJob, JobKind};
use crate::registry::AGENT_REGISTRY;
use crate::session::SessionContext;
use crate::session::TRANSIENT_AGENT_ID_PREFIXES;
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

/// Grace period: only attempt recovery if the user's last message is at least
/// this old.  Prevents races with agents that are still spawning or loading.
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
    /// message is from the agent, not the user).  This ensures the retry
    /// cap counts *consecutive* failures per episode, matching the ticket
    /// spec — a session that fails, recovers, then fails again starts with
    /// a fresh retry budget.
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
/// [`crate::shutdown::sleep_or_shutdown`], single-failure isolation via
/// `tracing::warn!`.
pub async fn run_dead_session_recovery_loop() {
    loop {
        if !crate::shutdown::sleep_or_shutdown(POLL_INTERVAL).await {
            break;
        }
        if let Err(e) = recover_dead_sessions().await {
            tracing::warn!(error = %e, "Dead session recovery poller failed");
        }
    }
}

// ── Core detection + recovery logic ─────────────────────────────────────────

/// Agent-ID prefixes excluded from dead-session recovery: `manager_` plus all
/// [`TRANSIENT_AGENT_ID_PREFIXES`]. `manager_` stays out of the slice itself
/// (Manager sessions must survive and keep their shared context) — this
/// builds the union from it without duplicating the prefix list.
fn excluded_agent_id_prefixes() -> impl Iterator<Item = &'static str> {
    std::iter::once("manager_").chain(TRANSIENT_AGENT_ID_PREFIXES.iter().copied())
}

/// Check all sessions for dead direct user-agent sessions and route recovery
/// jobs where needed.
async fn recover_dead_sessions() -> anyhow::Result<()> {
    let now = Utc::now();

    // Use SQL-side filtering to avoid loading excluded sessions (manager_,
    // ticket_, ask_, maintainer_, discovery_) from the database.  The
    // per-session `get_last_message_role` queries below are lightweight
    // (indexed `ORDER BY id DESC LIMIT 1`) and only run for eligible sessions.
    let sessions = crate::session::store()
        .list_sessions_with_metadata_excluding(&excluded_agent_id_prefixes().collect::<Vec<&str>>())
        .await;

    for session in &sessions {
        let agent_id = &session.agent_id;

        // ── Condition 1: last message is from the user ─────────────────
        let Some(last_role) = crate::session::store()
            .get_last_message_role(agent_id)
            .await
        else {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue; // empty session — clean up any stale tracker entry
        };
        if last_role != ChatRole::User {
            // Last message is from the agent (or is a tool/system message).
            // The session is healthy — the agent has already responded.
            // Clean up the retry-tracking entry so a future failure starts
            // with a fresh retry budget (consecutive-failure-per-episode,
            // matching the ticket spec).
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue;
        }

        // ── Condition 2: no live agent ─────────────────────────────────
        if AGENT_REGISTRY.contains(agent_id) {
            // Agent is still running (or was just spawned).
            continue;
        }

        // ── Condition 3: user message is stale ─────────────────────────
        let age = now - session.last_activity;
        let grace = chrono::Duration::from_std(STALE_GRACE_PERIOD)
            .expect("STALE_GRACE_PERIOD fits in chrono::Duration");
        if age < grace {
            // Too recent — give the agent time to start.
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
                 (pre-migration session or corrupted data) — \
                 skipping permanently"
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
/// recovered by the poller (Manager, ticket, ask, maintainer, discovery).
///
/// Note: the poller itself uses SQL-side filtering now — this function is
/// retained for test coverage and as documentation of the exclusion criteria.
#[cfg(test)]
fn is_excluded_agent_id(agent_id: &str) -> bool {
    excluded_agent_id_prefixes().any(|p| agent_id.starts_with(p))
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
        kind: JobKind::RecoveryRetry,
        role,
        // reply_target is not available from session metadata — the recovery
        // response will be persisted to chat_history and broadcast via the
        // GUI channel, but may not be deliverable via Telegram to unregistered
        // users.  This is an accepted limitation.
        reply_target: None,
    };

    tracing::info!(
        agent_id = %agent_id,
        role = %role.as_str(),
        workspace = %ctx.workspace_name,
        user = %ctx.user_name,
        channel = %ctx.channel,
        "Dead session recovery: routing retry job"
    );

    crate::message_router::route(agent_id, job);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_excluded_agent_id_transient_prefixes() {
        // Must match the union of TRANSIENT_AGENT_ID_PREFIXES + manager_.
        for prefix in excluded_agent_id_prefixes() {
            let id = format!("{prefix}suffix");
            assert!(
                is_excluded_agent_id(&id),
                "expected '{id}' (prefix '{prefix}') to be excluded"
            );
        }
    }

    #[test]
    fn test_is_excluded_agent_id_direct_session() {
        // Direct user-agent sessions should NOT be excluded by the union of
        // `TRANSIENT_AGENT_ID_PREFIXES` and `"manager_"`.
        assert!(!is_excluded_agent_id("gui_alice_main_workspace_engineer"));
        assert!(!is_excluded_agent_id("telegram_bob_my_project_analyst"));
        assert!(!is_excluded_agent_id(
            "voice_charlie_personal_work_assistant"
        ));
    }

    #[test]
    fn test_direct_session_underscore_in_names_not_mistaken_for_exclusion() {
        // Even if user/workspace names contain underscores,
        // the start of the agent_id is the channel name ("telegram"),
        // which doesn't match any excluded prefix.
        assert!(!is_excluded_agent_id(
            "telegram_some_user_my_cool_workspace_reviewer"
        ));
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
        // map, and no removal mechanism resets the counter (the critical
        // bug from try_remove_exhausted was fixed by removing it).
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
}
