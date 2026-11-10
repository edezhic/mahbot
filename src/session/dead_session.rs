//! Dead-session recovery poller: periodically check all direct user-agent sessions for silent
//! failure and re-trigger the agent by routing a recovery job through the message router.
//!
//! Three states are treated as dead: an **unanswered user message** (user sent one, no agent
//! responded, none running); an **unfinished turn** whose last persisted message is a tool result
//! (the run was cut before its final reply — recovery re-runs from the history it already has); and
//! a **dangling tool-call frame**, persisted at emission time, whose calls have no results (the
//! recovery re-run's `complete_pending_tool_calls` step settles them deterministically before the
//! next LLM call).
//!
//! Recovery is checked once a minute for `STARTUP_WINDOW` after start — when a restart matters —
//! and every `POLL_INTERVAL` afterwards, since an agent lost under a running process is rare. A
//! candidate is recovered only once its last message is at least `STALE_GRACE_PERIOD` old, so an
//! interrupted turn resumes ~two minutes after it was cut.
//!
//! The poller skips `manager_*` (Manager has its own lifecycle) plus every prefix in
//! [`super::TRANSIENT_AGENT_ID_PREFIXES`] — the union returned by
//! `super::non_user_facing_agent_id_prefixes`; a custom role's continuable `role_*` branch IS
//! recovered. Only direct user-agent sessions (`{user}_{ws}_{role}`, or the deduped
//! `{user}_personal:{role}`) and custom roles' continuable branches are eligible.
//!
//! Retry safety: every recovery attempt counts toward the per-session retry cap regardless of
//! whether routing succeeded, because the consumer loop gives no success feedback — the agent runs
//! asynchronously, so routing success is not recovery success. A persistently failing agent hits
//! `MAX_RETRIES` after roughly six hours of adaptive backoff (first wait ~20 min, ~40 min per
//! subsequent attempt) and is given up; a session becoming healthy is detected on the next poll
//! cycle, when the last message is an assistant reply.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use crate::agent::message_router::{AgentJob, MessageKind};
use crate::agent::registry::AGENT_REGISTRY;
use crate::session::SessionContext;
use crate::util::UnwrapPoison;
use crate::{ChatRole, Role};

/// Interval between poller cycles once the startup window has passed.
const POLL_INTERVAL: Duration = Duration::from_mins(10);

/// Interval between poller cycles during the startup window.
const STARTUP_POLL_INTERVAL: Duration = Duration::from_mins(1);

/// How long after start the poller keeps `STARTUP_POLL_INTERVAL`.
const STARTUP_WINDOW: Duration = Duration::from_mins(5);

/// Maximum consecutive recovery attempts before giving up on a session.
const MAX_RETRIES: u8 = 10;

/// Base backoff in minutes, doubled on each consecutive failure and capped at
/// `MAX_BACKOFF_MINUTES` — the effective minimum delay between attempts (the `or_insert` default
/// already starts doubled).
const BASE_BACKOFF_MINUTES: i64 = 10;

const MAX_BACKOFF_MINUTES: i64 = 40;

/// Grace period: only attempt recovery if the session's last candidate message (user message,
/// terminal tool frame, or dangling tool-call frame) is at least this old, so agents still spawning
/// or loading are not raced.
const STALE_GRACE_PERIOD: Duration = Duration::from_mins(2);

struct RetryState {
    attempt_count: u8,
    last_attempt_at: DateTime<Utc>,
    backoff_minutes: i64,
}

/// Tracks recovery attempts per agent ID, in memory only — a restart drops the tracker, which is
/// consistent with treating every session as fresh (the retry counter starts at zero).
///
/// An entry appears on the first attempt (`record_attempt`), is removed by `cleanup()` when the
/// session self-heals — the cap counts *consecutive* failures per episode — and persists once
/// `MAX_RETRIES` is exhausted, so `should_retry` stays `false` for the rest of the daemon's life.
struct DeadSessionTracker {
    inner: Mutex<HashMap<String, RetryState>>,
}

impl DeadSessionTracker {
    fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the session may be retried: under the retry cap AND the adaptive backoff has
    /// elapsed.
    fn should_retry(&self, agent_id: &str) -> bool {
        let map = self.inner.lock().unwrap_poison();
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

    /// Record a recovery attempt (increments the counter, updates the backoff).
    fn record_attempt(&self, agent_id: &str) {
        let mut map = self.inner.lock().unwrap_poison();
        let state = map.entry(agent_id.to_string()).or_insert(RetryState {
            attempt_count: 0,
            last_attempt_at: Utc::now(),
            backoff_minutes: BASE_BACKOFF_MINUTES,
        });
        state.attempt_count += 1;
        state.last_attempt_at = Utc::now();
        state.backoff_minutes = (state.backoff_minutes * 2).min(MAX_BACKOFF_MINUTES);
    }

    /// Remove the tracking entry for a session that no longer needs recovery — called on the
    /// self-heal paths (a completed assistant reply, or a turn ended by the `sleep` tool), on
    /// stale-entry cleanup and on permanent abandon (missing session context, invalid role). The
    /// cap counts *consecutive* failures per episode, so a session that fails, recovers or is
    /// abandoned, then fails again gets a fresh budget. Safe for untracked sessions (no-op).
    fn cleanup(&self, agent_id: &str) {
        self.inner.lock().unwrap_poison().remove(agent_id);
    }
}

static DEAD_SESSION_TRACKER: std::sync::LazyLock<DeadSessionTracker> =
    std::sync::LazyLock::new(DeadSessionTracker::new);

/// Run the dead-session recovery poller loop: once a minute for the first `STARTUP_WINDOW`, then
/// every `POLL_INTERVAL`. The first cycle is a minute in, never immediate —
/// `spawn_background_tasks` starts this task before `init_message_pipeline`, so an immediate scan
/// could route a recovery into a half-built pipeline. Cooperative shutdown via
/// [`crate::shutdown::sleep_or_shutdown_or_drain`], single-failure isolation via `tracing::warn!`.
pub async fn run_dead_session_recovery_loop() {
    let started = Instant::now();
    loop {
        let interval = if started.elapsed() < STARTUP_WINDOW {
            STARTUP_POLL_INTERVAL
        } else {
            POLL_INTERVAL
        };
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }
        if let Err(e) = recover_dead_sessions().await {
            tracing::warn!(error = %e, "Dead session recovery poller failed");
        }
    }
}

/// Classify a session by its last persisted message role (and, for an `Assistant` tail, its raw
/// content): is it a dead-session recovery candidate?
///
/// Candidates: a `User` tail (the user sent a message and the agent never answered); a `Tool` tail
/// (a tool result was committed but the run was cut before the final reply — resuming is safe
/// because the codebase has no user-facing cancel path, so this cannot mean "cancelled"); and an
/// `Assistant` tail that is a dangling emission-time tool-call frame (`last_content` = the tail
/// row's raw content), whose non-empty calls mean unresolved — exactly what
/// `complete_pending_tool_calls` settles.
///
/// Healthy: an `Assistant` tail with any other content (the turn completed) and a `System` tail (no
/// in-flight turn). `last_content` is only consulted for the `Assistant` arm.
fn is_recovery_candidate(last_role: ChatRole, last_content: &str) -> bool {
    match last_role {
        ChatRole::User | ChatRole::Tool => true,
        ChatRole::Assistant => super::is_dangling_tool_call_frame(last_role, last_content),
        ChatRole::System => false,
    }
}

/// Check all sessions for dead direct user-agent sessions and route recovery jobs where needed.
async fn recover_dead_sessions() -> anyhow::Result<()> {
    let now = Utc::now();

    // SQL-side filtering keeps excluded sessions (manager_ and every TRANSIENT_AGENT_ID_PREFIXES
    // prefix) from being loaded; the per-session `get_last_message_tail` reads below are
    // lightweight (indexed `ORDER BY id DESC LIMIT 1`) and run only for eligible sessions. A custom
    // role's continuable `role_` branch is deliberately not excluded.
    let sessions = crate::session::store()
        .list_sessions_with_metadata_excluding(
            &crate::session::non_user_facing_agent_id_prefixes().collect::<Vec<&str>>(),
        )
        .await;

    for session in &sessions {
        let agent_id = &session.agent_id;

        let Some((last_role, tail_content)) = crate::session::store()
            .get_last_message_tail(agent_id)
            .await
        else {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue; // empty session — clean up any stale tracker entry
        };
        if !is_recovery_candidate(last_role, &tail_content) {
            // An assistant reply that is not a dangling frame, or a system message, means the turn
            // is complete — the agent already responded. Clean up the retry-tracking entry so a
            // future failure starts with a fresh budget (consecutive-failure-per-episode).
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue;
        }
        // An Assistant tail is a candidate only via the dangling-frame path —
        // remember that for the pause guard below.
        let needs_pause_guard = last_role == ChatRole::Assistant;

        if AGENT_REGISTRY.contains(agent_id) {
            // Agent is still running (or was just spawned).
            continue;
        }

        let age = now - session.last_activity;
        let grace = chrono::Duration::from_std(STALE_GRACE_PERIOD)
            .expect("STALE_GRACE_PERIOD fits in chrono::Duration");
        if age < grace {
            // Too recent — give the agent time to start.
            continue;
        }

        if crate::session::store().get_sleep_ended(agent_id).await {
            // The agent ended its turn via the `sleep` tool: that tail is an intentional stop while
            // waiting for input, not a crash — never recover it. Unmarked tool tails (real
            // crash/drain cuts) keep recovering. The session has self-healed, so reset any stale
            // retry budget from an earlier episode.
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue;
        }

        // A branch launch stopped before it produced anything must never be auto-restarted. It does
        // NOT consume retry budget: clean up any stale tracker entry and skip, and a later fresh
        // launch re-enables recovery by clearing the mark (see `Agent::work`).
        if session.launch_stopped {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
            continue;
        }

        if !DEAD_SESSION_TRACKER.should_retry(agent_id) {
            // Either the session exhausted its retries (the entry persists
            // in the tracker permanently, blocking future attempts) or the
            // adaptive backoff hasn't elapsed yet (will be rechecked on the
            // next poll cycle).  Either way — silently skip.
            continue;
        }

        // Recover!
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

        // Pause guard (dangling-frame resumption only)
        // A paused workspace is a strict freeze: auto-resuming interrupted work
        // the user explicitly paused must not happen (unpausing re-drives via
        // the puller / next user prompt). A frozen agent is not recoverable
        // while registered; this catches the tail it leaves after bailing.
        // User/Tool tails stay exempt: answering a pending user prompt matches
        // normal direct-chat behavior, which a workspace pause does not freeze.
        // A custom role is exempt too: a workspace pause freezes only launches
        // tied to a ticket, never a continuable branch (a custom role's or a worker branch's).
        if needs_pause_guard && !is_continuable_branch(role, agent_id) {
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

        // Phase 2: route the recovery job (fire-and-forget mpsc send).
        // Failures in Phase 1 (missing context, invalid role) are NOT
        // counted — they are permanent data issues that will never
        // resolve with retries. A `false` return means the branch is
        // permanently un-recoverable (its custom-role record is gone): drop
        // the tracker entry rather than consuming retry budget on it.
        if attempt_recovery(agent_id, &ctx, role).await {
            DEAD_SESSION_TRACKER.record_attempt(agent_id);
        } else {
            DEAD_SESSION_TRACKER.cleanup(agent_id);
        }
    }

    Ok(())
}

/// Caller must have already called
/// [`get_session_context`](super::SessionStore::get_session_context).
async fn attempt_recovery(agent_id: &str, ctx: &SessionContext, role: Role) -> bool {
    // Execution-time invariant: pinned roles (Assistant) never
    // run outside the user's personal workspace, even on recovery. A pinned
    // role with an empty user has no personal identity — a permanent data
    // condition, like the other permanent-data refusals below: refuse the
    // retry for good.
    let Some(workspace_name) =
        crate::users::enforce_personal_pinning(role, &ctx.workspace_name, &ctx.user_name)
    else {
        tracing::error!(
            agent_id = %agent_id,
            role = %role.as_str(),
            workspace = %ctx.workspace_name,
            "Dead session recovery: pinned role with empty user — refusing to route unpinned"
        );
        return false;
    };

    // A custom-role branch is resumable only while its record still exists: the
    // record owns the authorial prompt and the tool subset, so a deleted role
    // leaves nothing to run. Abandon the branch permanently in that case. A worker
    // branch carries no record — its branch name is enough to rebuild the launch.
    let launch = if role == Role::Custom {
        let Some(name) = ctx.custom_role.as_deref() else {
            tracing::warn!(
                agent_id = %agent_id,
                "Dead session recovery: custom-role session without a role name — skipping permanently"
            );
            return false;
        };
        match crate::custom_role::get(name).await {
            Ok(Some(record)) => Some(crate::Launch {
                role: Role::Custom,
                name: name.to_string(),
                identifier: ctx.custom_identifier.clone(),
                prompt: record.prompt,
                tools: record.tools,
                agent_id: agent_id.to_string(),
                reply_to: None,
            }),
            Ok(None) => {
                tracing::warn!(
                    agent_id = %agent_id,
                    role = %name,
                    "Dead session recovery: custom role no longer exists — skipping permanently"
                );
                return false;
            }
            Err(e) => {
                // A transient read failure must not abandon a branch forever —
                // count the attempt and let the backoff retry later.
                tracing::warn!(
                    agent_id = %agent_id,
                    role = %name,
                    error = %e,
                    "Dead session recovery: custom role lookup failed — skipping this cycle"
                );
                return true;
            }
        }
    } else if role == Role::Worker && crate::session::is_worker_branch_agent_id(agent_id) {
        // The branch name is the recorded identifier, or — when that write
        // failed — the agent id's own suffix, which is the same name.
        let branch = ctx
            .custom_identifier
            .clone()
            .or_else(|| crate::session::worker_branch_name(agent_id).map(str::to_string));
        Some(crate::Launch {
            role: Role::Worker,
            name: String::new(),
            identifier: branch,
            prompt: String::new(),
            tools: Vec::new(),
            agent_id: agent_id.to_string(),
            // The conversation that asked for this branch's work, so its report
            // returns there instead of the admin's Assistant (an absent/NULL
            // column leaves it `None`, as for every existing session).
            reply_to: ctx.reply_to.clone().map(Box::new),
        })
    } else {
        None
    };

    // Recovery retry: uses `RecoveryRetry` kind with empty content so the
    // agent runs against the EXISTING session history (which already contains
    // the user's original message).  No boilerplate message is injected, and
    // no emoji error feedback is sent on retry failures — the emoji fires
    // only once on the original failure.
    tracing::info!(
        agent_id = %agent_id,
        role = %role.as_str(),
        workspace = %workspace_name,
        user = %ctx.user_name,
        channel = %ctx.channel,
        "Dead session recovery: routing retry job"
    );

    let job = AgentJob {
        content: String::new(),
        workspace_name,
        user_name: ctx.user_name.clone(),
        channel: ctx.channel.clone(),
        kind: MessageKind::RecoveryRetry,
        role,
        // reply_target is not available from session metadata — the recovery response is persisted
        // to chat_history and broadcast via the GUI channel, but may not reach unregistered
        // Telegram users. An accepted limitation.
        reply_target: None,
        pending_job_id: None,
        originating_workspace: None,
        custom_role: launch,
        // A recovered worker branch's asker travels on its launch above (`Launch::reply_to`), read
        // by the report path; this field's remaining user is the Manager leg.
        reply_to_custom: None,
        urgent: false,
    };

    crate::agent::message_router::route(agent_id, job);
    true
}

/// Whether a dead session is a continuable branch (a custom role's or a worker branch's). Such a
/// session is exempt from the workspace-pause guard: a pause freezes only launches tied to a
/// ticket.
fn is_continuable_branch(role: Role, agent_id: &str) -> bool {
    match role {
        Role::Custom => true,
        Role::Worker => crate::session::is_worker_branch_agent_id(agent_id),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::BatchMetadata;
    use super::*;
    use crate::Tool;

    /// A custom branch with no role record (or a session that never recorded one) has no prompt and
    /// no tool set: `attempt_recovery` abandons it rather than spending retry budget on a turn that
    /// could never run.
    #[tokio::test]
    async fn attempt_recovery_abandons_a_custom_branch_without_a_role_record() {
        crate::util::test::init_test_stores().await;
        let ctx = SessionContext {
            channel: "gui".to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            workspace_name: crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
            role: crate::Role::Custom.as_str().to_string(),
            custom_role: None,
            custom_identifier: None,
            reply_to: None,
        };
        assert!(
            !attempt_recovery("role_test_launchless", &ctx, Role::Custom).await,
            "a custom session without a recorded role is permanently un-recoverable"
        );
        let deleted = SessionContext {
            custom_role: Some("ghost_role".to_string()),
            ..ctx
        };
        assert!(
            !attempt_recovery("role_test_ghost", &deleted, Role::Custom).await,
            "a branch whose custom role was deleted is permanently un-recoverable"
        );
    }

    /// A worker branch survives a restart: the poller rebuilds its launch from the recorded branch
    /// name — no role record needed — and re-routes the unfinished turn with the conversation that
    /// asked restored, so the report returns there instead of the admin's Assistant.
    #[tokio::test]
    async fn attempt_recovery_rebuilds_a_worker_branch_with_its_asker() {
        crate::util::test::init_management_test_stores().await;
        let branch = crate::session::worker_branch_agent_id("night");
        let asker_folder = crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
        let ctx = SessionContext {
            channel: "gui".to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            workspace_name: asker_folder.clone(),
            role: Role::Worker.as_str().to_string(),
            custom_role: None,
            custom_identifier: Some("night".to_string()),
            reply_to: Some(crate::agent::message_router::Asker {
                launch: crate::Launch {
                    role: Role::Custom,
                    name: "briefer".to_string(),
                    identifier: Some("hourly".to_string()),
                    prompt: "you brief".to_string(),
                    tools: Vec::new(),
                    agent_id: crate::session::custom_role_agent_id("briefer", Some("hourly")),
                    reply_to: None,
                },
                workspace: asker_folder,
            }),
        };
        let mut rx = crate::agent::message_router::register_agent(&branch);
        assert!(
            attempt_recovery(&branch, &ctx, Role::Worker).await,
            "a worker branch is recoverable"
        );
        let job = rx.try_recv().expect("the retry job is routed");
        assert_eq!(job.kind, MessageKind::RecoveryRetry);
        assert_eq!(job.role, Role::Worker);
        let launch = job
            .custom_role
            .expect("a worker branch's job carries its launch");
        assert_eq!(launch.role, Role::Worker);
        assert_eq!(launch.identifier.as_deref(), Some("night"));
        assert_eq!(
            launch
                .reply_to
                .as_ref()
                .map(|reply| reply.launch.name.as_str()),
            Some("briefer"),
            "the asking conversation is restored, so the report returns there"
        );
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
        assert!(!is_recovery_candidate(
            ChatRole::Assistant,
            "plain assistant reply"
        ));
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
        assert!(!is_recovery_candidate(
            ChatRole::Assistant,
            "not json at all"
        ));
        assert!(!is_recovery_candidate(ChatRole::System, ""));
    }

    #[test]
    fn test_dead_session_tracker_exhaustion_is_permanent() {
        let tracker = DeadSessionTracker::new();
        let agent_id = "test_permanent_engineer";

        // First attempt is allowed for a brand-new session; immediately
        // after recording it, backoff blocks the retry.
        assert!(tracker.should_retry(agent_id));
        tracker.record_attempt(agent_id);
        assert!(!tracker.should_retry(agent_id));

        // Record the remaining attempts (simulating elapsed backoff).
        for _ in 2..=MAX_RETRIES {
            tracker.record_attempt(agent_id);
        }

        // Once the cap is reached, should_retry returns false permanently — the entry stays in the
        // map and no removal mechanism resets the counter (a previous one had a critical bug and
        // was deliberately removed).
        assert!(
            !tracker.should_retry(agent_id),
            "should be blocked after {MAX_RETRIES} attempts"
        );

        // Backdate the last attempt far past any possible backoff: the cap alone, not backoff
        // timing, must block further retries.
        if let Some(state) = tracker.inner.lock().unwrap_poison().get_mut(agent_id) {
            state.last_attempt_at = Utc::now() - chrono::Duration::hours(24);
        }
        assert!(
            !tracker.should_retry(agent_id),
            "cap must block even after the backoff has long elapsed"
        );

        // The entry persists with the cap reached — distinguishing a cap-blocked session from one
        // merely waiting out its backoff.
        assert_eq!(
            tracker
                .inner
                .lock()
                .unwrap_poison()
                .get(agent_id)
                .map(|s| s.attempt_count),
            Some(MAX_RETRIES)
        );
    }

    /// The tracker's per-session budget at every entry point: a first attempt is allowed, recording
    /// it backs the retry off, cleanup forgets the session (allowing a fresh start on the same
    /// tracker), and a daemon restart begins from an empty tracker.
    #[test]
    fn dead_session_tracker_backoff_cleanup_and_restart_reset() {
        let tracker = DeadSessionTracker::new();
        let agent_id = "test_backoff_engineer";

        assert!(tracker.should_retry(agent_id));
        tracker.record_attempt(agent_id);

        assert!(!tracker.should_retry(agent_id));

        tracker.cleanup(agent_id);
        assert!(tracker.should_retry(agent_id));

        tracker.record_attempt(agent_id);
        let tracker2 = DeadSessionTracker::new();
        assert!(
            tracker2.should_retry(agent_id),
            "should be fresh after simulated restart"
        );
    }

    // Four tests share the process-global test DB, retry tracker and agent registry, and all
    // but `sleep_ended_session_is_not_recovered` also depend on the drain flag, so each carries
    // the serial_test keys for the globals it touches (`provider` for the fake provider,
    // `drain` for the shutdown drain flag).

    /// Seed a direct session whose tail is a dangling emission-time analyze tool-call frame
    /// plus the caller-owned launched durable job, checkpointed as if a prior run completed the
    /// analyst round; returns the `job_id`. Shared scaffolding for the e2e/live/paused recovery
    /// tests: it persists the frame and context, backdates `last_activity` past the grace
    /// period, drives the sync analyze dispatch under a drain (leaving the job `launched` with
    /// the call result-less), then checkpoints the roster outcomes as the durable resume input.
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
                arguments: serde_json::json!({"message": "analyze this"}),
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
                BatchMetadata::context("gui", "e2e_user", ws.name.as_str(), "engineer"),
            )
            .await
            .unwrap();

        // Backdate the session so the recovery grace period passes.
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
                tool.execute(ws, serde_json::json!({"message": "analyze this"}))
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

    /// Full poller e2e: a dangling emission-time analyze frame (drain-cut mid-tool-round) is
    /// classified as a recovery candidate, the recovery re-run routes a RecoveryRetry, the
    /// consumer's universal `complete_pending_tool_calls` resumes the durable job LLM-free, and
    /// only then does the recovery round call the model once — asserted to exactly one
    /// fingerprint.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    // Joins the `provider` group (its serial key guards the global fake provider, `drain` the
    // shutdown drain flag) so two per-test runtimes never drive the shared global stores
    // concurrently.
    async fn recovery_resumes_dangling_durable_job_end_to_end() {
        let fake = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                .ok("Recovery round: analysis resumed and complete."),
        );
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_management_test_stores().await;
        // The router's consumer-loop path (typing setup / unregistered delivery) needs the
        // channel registry; a default (empty) one is enough — no telegram channel exists
        // and the recovery response is persisted without a transport.
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

    /// A live (registered) agent blocks recovery at the registry gate: the dangling durable job is
    /// NOT resumed and no attempt is recorded, so the session is still eligible on a later cycle.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn live_running_session_is_not_recovered() {
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
        // `recover_dead_sessions` (e.g. the paused test, serialized after us
        // by the serial key) does not route a recovery for this leftover
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
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn paused_workspace_dangling_frame_is_not_recovered() {
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
    async fn sleep_ended_session_is_not_recovered() {
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
                BatchMetadata::context("gui", "e2e_user", ws.name.as_str(), "assistant"),
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
        // (serialized after us by the serial key) does not route a recovery
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
