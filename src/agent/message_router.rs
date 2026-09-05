//! Agent-ID-based message router — replaces per-role queue with per-agent
//! channels for true instance-level parallelism.
//!
//! Every agent instance gets its own [`mpsc::UnboundedSender`] stored in a
//! global [`HashMap`] keyed by unique agent ID. Jobs are routed directly to
//! the correct consumer — no agent ever blocks another.
//!
//! # Producer paths
//!
//! Jobs reach [`route`] from several producer paths — user chat messages via
//! [`route_user_message`], ticket transitions, sub-agent results, boot-time
//! pending-job replay, and dead-session recovery — and the router stays
//! agnostic to the job's origin.
//!
//! # Agent ID formats
//!
//! Agent IDs are stable deterministic strings built by
//! `crate::session::resolve_agent_id` and friends. The [`Role`] is embedded
//! directly in [`AgentJob`] so the router never needs to parse the agent ID.
//!
//! # Response delivery
//!
//! - [`Role::Manager`] broadcasts to all workspace users.
//! - Other roles broadcast to all of the triggering user's channel bindings
//!   (Manager model generalized).

use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{OnceLock, RwLock};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::channels::{
    broadcast_and_persist_agent_response, spawn_scoped_typing_task, stop_typing,
};
use crate::db;
use crate::users::UserRecord;
use crate::util::UnwrapPoison;
use crate::{Channel, ChatEvent, Role, SendMessage};

// ── Job definition ─────────────────────────────────────────────────────────

/// Emoji sent to the user when an agent completes without producing a
/// response (LLM errors, retry exhaustion, context overflow, etc.).
/// Language-agnostic — pure emoji, no text.
///
/// Not sent when the agent was cancelled (deliberate stop) or during global
/// shutdown.
///
/// Known limitations:
/// - Voice channel: TTS speaks this as "robot warning retry" (acceptable for now).
/// - Emoji rendering varies across terminals and clients.
const AGENT_FAILURE_EMOJI: &str = "🤖⚠️🔄";

/// Per-role attribution emoji for Telegram deliveries — mirrors the GUI role
/// icons ([`crate::gui::theme::role_icon`]) per the product spec.
fn telegram_role_emoji(role: Role) -> &'static str {
    match role {
        Role::Manager => "🤖",
        Role::Engineer => "🔧",
        Role::Analyst => "🔍",
        Role::Coder => "💻",
        Role::Qa => "🔨",
        Role::Reviewer => "✅",
        Role::Discovery => "🔎",
        Role::Artist => "🎨",
        Role::Maintainer => "⚙️",
        Role::Sanitation => "🧼",
        Role::Assistant => "💬",
        Role::Support => "🛟",
    }
}

/// Telegram agent responses carry a first-line role attribution
/// (`"{emoji} {label}:\n"`) when the recipient can switch between multiple
/// roles. Other channels pass the response through unchanged (borrowed, no
/// allocation).
#[must_use]
fn telegram_delivery_content<'a>(
    channel: &str,
    role: Role,
    recipient_roles: &[String],
    response: &'a str,
) -> Cow<'a, str> {
    if channel != "telegram" || recipient_roles.len() < 2 {
        return Cow::Borrowed(response);
    }
    Cow::Owned(format!(
        "{} {}:\n{}",
        telegram_role_emoji(role),
        crate::agent::role::role_info(&role).display_label,
        response
    ))
}

/// Semantic category of a queue job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    /// User-typed message (chat or inline-button callback).
    /// For Manager: the ticket_chronicle timeline drains before the agent runs.
    /// For other roles: no ticket timeline drain.
    UserMessage,
    /// System notification from a ticket transition.
    /// Only enqueued for the Manager role.
    TicketNotify,
    /// Result from an async AnalyzeTool sub-agent, injected back into the caller's
    /// agent session.
    AnalyzeToolResult,
    /// Result from an async ImplementTool dispatch, injected back into the
    /// caller's agent session.
    ImplementResult,
    /// Result from an async deep research run (ResearchTool), injected back
    /// into the Manager's agent session. Exactly one envelope per run.
    ResearchResult,
    /// Comment added to a ticket while an agent is working on it.
    /// Delivered mid-work via the agent's inbox (not a consumer loop).
    TicketComment,
    /// Recovery retry for a dead session — routes without appending a new
    /// user message to the session history.  Emoji error feedback is
    /// suppressed for this kind (the emoji fires once on the original
    /// failure, not on silent retries).
    RecoveryRetry,
    /// Message sent by an assistant agent into a workspace Manager's session,
    /// wrapped in an `<assistant-message>` envelope. Manager-bound and durable
    /// (persisted to `pending_jobs` before routing, same class as Manager-bound
    /// `UserMessage`); the Manager's chronicle/CDC drain applies. Deliberately
    /// distinct from `UserMessage` so the failure-emoji gate (`== UserMessage`)
    /// structurally excludes it.
    AgentMessage,
    /// Durable, user-invisible copy of a Manager's response, addressed back
    /// into the originating Assistant agent's session (`<manager-reply>`
    /// envelope). Appended to the session like other result kinds; never
    /// broadcast/persisted to chat_history.
    ManagerReply,
}

/// A single unit of work for an agent consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentJob {
    /// The message content to process.
    pub content: String,
    /// Workspace name — resolved to a [`crate::Workspace`] inside the consumer.
    pub workspace_name: String,
    /// Sender identity — used for per-user response delivery.
    pub user_name: String,
    /// Channel origin (gui, telegram, voice).
    pub channel: String,
    /// The semantic job kind.
    pub kind: MessageKind,
    /// The agent's [`Role`] — embedded directly so the router never needs
    /// to parse agent ID strings.
    pub role: Role,
    /// Original reply target from the incoming message (e.g., Telegram chat_id).
    /// Used by [`deliver_unregistered_user_response`] when there is no
    /// [`UserRecord`](crate::users::UserRecord) in the users DB.
    /// `None` for non-user-facing jobs (ticket notifications, broadcast-only).
    pub reply_target: Option<String>,
    /// Durable pending_jobs row id (set when the job was persisted /
    /// replayed from a pending row). The consumer deletes the row only after
    /// `run_agent` returns — the at-least-once delivery boundary.
    #[serde(default)]
    pub pending_job_id: Option<String>,
    /// Agent session the response to this job must be addressed to. Set only on
    /// [`MessageKind::AgentMessage`] jobs (the originating Assistant's agent id,
    /// copied onto the resulting [`MessageKind::ManagerReply`] envelope job) and on
    /// `ManagerReply` jobs themselves (the addressed target). `None` on legacy
    /// persisted envelopes — `serde(default)` keeps them deserializing.
    #[serde(default)]
    pub reply_to_agent_id: Option<String>,
    /// Workspace of the session a response must be routed back into. Set on
    /// [`MessageKind::AgentMessage`] jobs (the originating Assistant's own
    /// workspace, consumed by `route_manager_reply`); `None` elsewhere.
    #[serde(default)]
    pub reply_workspace_name: Option<String>,
}

// ── Global router ─────────────────────────────────────────────────────────

/// Global router table: maps agent ID → unbounded sender for that agent's
/// consumer loop.
static ROUTER: OnceLock<RwLock<HashMap<String, mpsc::UnboundedSender<AgentJob>>>> = OnceLock::new();

/// Initialise the global router table.
///
/// Must be called after the Tokio runtime is active (i.e., during startup).
/// No consumer loops are spawned here — they are created lazily on first
/// [`route`] to each agent ID.
pub fn init_global() -> anyhow::Result<()> {
    ROUTER
        .set(RwLock::new(HashMap::new()))
        .map_err(|_| anyhow::anyhow!("ROUTER already initialised"))?;
    Ok(())
}

/// Route a job to the consumer for the given agent ID.
///
/// # Fast path
///
/// Locks the router for reading only, looks up the agent ID, clones the
/// sender, drops the lock, and sends. If the sender exists, we are done.
///
/// # Slow path
///
/// The agent ID is not yet registered: drop the read lock, acquire a write
/// lock, create a new channel, spawn a [`consumer_loop`], store the sender,
/// and forward the job.  A double-check pattern prevents races when two
/// tasks simultaneously encounter a missing entry.
pub fn route(agent_id: &str, job: AgentJob) {
    // ── Fast path: read-only lookup ───────────────────────────────────
    {
        let map = ROUTER
            .get()
            .expect("ROUTER not initialised — call init_global() first");
        let guard = map.read().unwrap_poison();
        if let Some(tx) = guard.get(agent_id) {
            if tx.send(job).is_err() {
                error!(agent_id = %agent_id, "Router: consumer dropped — failed to route job");
            }
            return;
        }
    }

    // ── Slow path: create new consumer ────────────────────────────────
    let map = ROUTER
        .get()
        .expect("ROUTER not initialised — call init_global() first");
    let mut guard = map.write().unwrap_poison();

    // Double-check: another task might have created the entry while we
    // waited for the write lock.
    if let Some(tx) = guard.get(agent_id) {
        if tx.send(job).is_err() {
            error!(agent_id = %agent_id, "Router: consumer dropped (double-check) — failed to route job");
        }
        return;
    }

    let (tx, rx) = mpsc::unbounded_channel::<AgentJob>();
    let agent_id_for_consumer = agent_id.to_string();
    let agent_id_for_cleanup = agent_id_for_consumer.clone();

    tokio::spawn(async move {
        // Wrap consumer_loop in catch_unwind so a panic doesn't leave a dead
        // sender in the router table, causing permanent message loss for this
        // agent ID.
        let result = std::panic::AssertUnwindSafe(consumer_loop(agent_id_for_consumer, rx))
            .catch_unwind()
            .await;

        // Always clean up the router entry — runs on both normal exit and panic.
        if let Some(map) = ROUTER.get() {
            let mut guard = map.write().unwrap_poison();
            guard.remove(&agent_id_for_cleanup);
        }

        if let Err(panic) = result {
            error!(
                agent_id = %agent_id_for_cleanup,
                "Consumer loop panicked — entry removed from router table",
            );
            error!(agent_id = %agent_id_for_cleanup, panic = %crate::util::panic_message(&*panic), "Consumer loop panic message");
        } else {
            debug!(
                agent_id = %agent_id_for_cleanup,
                "Consumer loop exited — removed from router table",
            );
        }
    });

    if tx.send(job).is_err() {
        error!(agent_id = %agent_id, "Router: brand-new consumer dropped immediately — failed to route job");
    }

    guard.insert(agent_id.to_string(), tx);
}

/// Route a user message to the agent for the given role in a workspace.
///
/// Computes the agent ID via `crate::session::resolve_agent_id` (Manager
/// role → `manager_{ws_name}`, others → channel-agnostic direct ID per
/// user+workspace+role) and enqueues
/// a [`MessageKind::UserMessage`] job. Surrounding per-site pipelines (broadcast,
/// persistence, enrichment) remain at the call sites.
///
/// # Durability (at-least-once)
///
/// Manager-bound messages are persisted to `pending_jobs` BEFORE routing —
/// a crash between persist and delivery replays the row at next boot (dedup
/// prevents duplicate append). Non-Manager messages are covered by the
/// dead-session poller (not durable). During the graceful drain the message
/// is persisted but NOT routed — the row is reclaimed by boot replay.
pub async fn route_user_message(
    content: String,
    workspace_name: String,
    user_name: String,
    channel: String,
    role: Role,
    reply_target: Option<String>,
) {
    // Invariant: the routed user identity is never an empty string. A real
    // admin user is always seeded. Normalize defensively so no path can create
    // a malformed identity (bare "_ws_role" key) or persist/reply under an
    // empty user. (direct_agent_id also normalizes the ID key; this layer
    // normalizes the delivery payload.)
    let user_name =
        crate::session::normalize_user_name(&user_name, "route_user_message").to_string();

    let agent_id = crate::session::resolve_agent_id(&user_name, role.as_str(), &workspace_name);
    let mut job = AgentJob {
        content,
        workspace_name,
        user_name,
        channel,
        kind: MessageKind::UserMessage,
        role,
        reply_target,
        pending_job_id: None,
        reply_to_agent_id: None,
        reply_workspace_name: None,
    };

    // Manager-bound UserMessage is durable: DURABLE kinds =
    // UserMessage (manager-bound only), AnalyzeToolResult, ResearchResult.
    if job.role == Role::Manager {
        let (persisted, id) = persist_manager_envelope(&job, "manager message").await;
        if let Some(id) = id {
            job.pending_job_id = Some(id);
        }
        // Persisted during the drain/shutdown → skip routing (boot replay
        // reclaims). NOT persisted → route best-effort; the realistic rescue
        // is the non-drain persist-failure case (the consumer is still
        // pulling). A drain-time best-effort route may land in a consumer that
        // has already stopped pulling, so the message can still be lost at
        // exit — the fix primarily closes the silent-drop on the normal-path
        // double fault, not the drain window.
        if crate::shutdown::aborting() && persisted {
            return;
        }
    }
    route(&agent_id, job);
}

/// Persist a durable Manager-bound envelope to `pending_jobs` before routing —
/// the at-least-once delivery boundary. `producer` names the producer in the
/// failure warn (the message text differs per producer, e.g. "manager message"
/// vs "agent message"). Returns `(persisted, id)`: on a successful write
/// `persisted` is `true` and `id` is the row id to stamp onto the job; on a
/// persistence failure `persisted` is `false` and `id` is `None`, and the
/// caller falls back to best-effort (at-most-once) routing.
async fn persist_manager_envelope(job: &AgentJob, producer: &str) -> (bool, Option<String>) {
    let id = crate::generate_id();
    match persist_pending(job, id.clone()).await {
        Ok(()) => (true, Some(id)),
        Err(e) => {
            warn!(
                error = %e,
                producer,
                "Failed to persist manager-bound envelope — routing best-effort (at-most-once)",
            );
            (false, None)
        }
    }
}

/// Build the `AgentMessage` envelope: the assistant's message normalized and
/// addressed to the workspace Manager, carrying the originating Assistant's
/// agent id + workspace for the addressed reply leg.
fn agent_message_job(
    content: String,
    workspace_name: String,
    origin_workspace_name: String,
    user_name: &str,
    reply_to_agent_id: String,
) -> AgentJob {
    let user_name =
        crate::session::normalize_user_name(user_name, "route_agent_message_to_manager")
            .to_string();
    AgentJob {
        content,
        workspace_name,
        user_name,
        channel: "gui".to_string(),
        kind: MessageKind::AgentMessage,
        role: Role::Manager,
        reply_target: None,
        pending_job_id: None,
        reply_to_agent_id: Some(reply_to_agent_id),
        reply_workspace_name: Some(origin_workspace_name),
    }
}

/// Route a message from an assistant agent into the workspace Manager's
/// session (`MessageKind::AgentMessage`), durably: persisted to `pending_jobs`
/// BEFORE routing (same reliability class as Manager-bound user messages).
/// `reply_to_agent_id` is the originating Assistant's agent id — the Manager's
/// response is additionally addressed back into that session.
pub async fn route_agent_message_to_manager(
    content: String,
    workspace_name: String,
    origin_workspace_name: String,
    user_name: String,
    reply_to_agent_id: String,
) {
    let mut job = agent_message_job(
        content,
        workspace_name,
        origin_workspace_name,
        &user_name,
        reply_to_agent_id,
    );
    // Same durable Manager-bound block as route_user_message.
    let (persisted, id) = persist_manager_envelope(&job, "agent message").await;
    if let Some(id) = id {
        job.pending_job_id = Some(id);
    }
    if crate::shutdown::aborting() && persisted {
        return;
    }
    route(&crate::jobs::envelope_target(&job), job);
}

/// Persist an envelope to `pending_jobs`. The target agent id is derived
/// from the envelope by [`crate::jobs::pending_job_params`].
/// Used by the durable producers: manager-bound messages routed here,
/// analyze/research results via [`crate::jobs::complete_job_with_envelope`],
/// and alarm notifications from [`crate::alarms::fire_alarm`].
pub(crate) async fn persist_pending(job: &AgentJob, id: String) -> anyhow::Result<()> {
    let now = db::now();
    crate::session::store()
        .conn
        .execute(
            crate::jobs::PENDING_JOB_INSERT_SQL,
            crate::jobs::pending_job_params(&id, job, &now)?,
        )
        .await?;
    Ok(())
}

/// Register an agent in the router table without spawning a consumer loop.
///
/// Returns a receiver that the caller (typically the agent's `llm_loop`)
/// drains manually via `try_recv()`. The sender is stored in the router
/// table so that [`try_route`] can deliver messages (e.g., ticket comments)
/// to this agent mid-work.
///
/// Call [`unregister_agent`] when the agent's work finishes to remove the
/// entry — for caller-registered paths (those passing a receiver into
/// [`crate::agent::run_agent`]), the exit guard inside `run_agent` does this
/// on every path including panic; the persistent consumer path cleans up in
/// [`route`]'s wrapper instead.
pub fn register_agent(agent_id: &str) -> mpsc::UnboundedReceiver<AgentJob> {
    let (tx, rx) = mpsc::unbounded_channel::<AgentJob>();
    let map = ROUTER
        .get()
        .expect("ROUTER not initialised — call init_global() first");
    let mut guard = map.write().unwrap_poison();
    guard.insert(agent_id.to_string(), tx);
    rx
}

/// Unregister an agent from the router table.
///
/// After this call, [`try_route`] returns `false` for this agent ID.
/// Must be called when the agent's work loop finishes to remove the
/// router entry. Safe to call even if the agent was never registered.
pub fn unregister_agent(agent_id: &str) {
    let Some(map) = ROUTER.get() else { return };
    let mut guard = map.write().unwrap_poison();
    guard.remove(agent_id);
}
/// Try to route a job to a previously registered agent.
///
/// Returns `true` if the agent was found and the job was delivered.
/// Returns `false` if the agent is not registered — the caller can
/// fall back to persisting the job in the DB for the next dispatch.
///
/// This does NOT spawn a consumer loop — it only sends to already-
/// registered agents. This is the fast path for mid-work message
/// delivery (e.g., ticket comments to running pipeline agents).
pub fn try_route(agent_id: &str, job: AgentJob) -> bool {
    let Some(map) = ROUTER.get() else {
        return false;
    };
    let guard = map.read().unwrap_poison();
    if let Some(tx) = guard.get(agent_id) {
        tx.send(job).is_ok()
    } else {
        false
    }
}

// ── Consumer loop ─────────────────────────────────────────────────────────

/// The consumer task that processes agent jobs for a single agent instance,
/// one job at a time.
///
/// Shutdown-aware loop: checks for global shutdown between jobs.
///
/// Cleanup (removing this consumer's entry from the router table) is handled
/// by the outer wrapper in [`route()`], which runs on both normal exit and
/// (via `catch_unwind`) panic exit.
#[expect(clippy::too_many_lines)]
async fn consumer_loop(agent_id: String, mut rx: mpsc::UnboundedReceiver<AgentJob>) {
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            info!(agent_id = %agent_id, "Message router: shutting down — queue drained");
            break;
        }
        // Shutdown/drain: stop pulling after the current job. The job already
        // pulled in the previous iteration completes its round.
        if crate::shutdown::aborting() {
            info!(agent_id = %agent_id, "Message router: draining — no new jobs pulled");
            break;
        }

        let mut job = tokio::select! {
            job = rx.recv() => {
                match job {
                    Some(job) => job,
                    None => break,
                }
            }
            () = shutdown.cancelled() => {
                info!(agent_id = %agent_id, "Message router: shutting down (global shutdown)");
                break;
            }
        };

        // Execution-time invariant: pinned roles (Assistant/Artist/Support)
        // never run outside the user's personal workspace, no matter what the
        // producer stored. Empty-user envelopes for pinned roles are refused
        // outright.
        let Some(pinned_ws) =
            crate::users::enforce_personal_pinning(job.role, &job.workspace_name, &job.user_name)
        else {
            error!(
                agent_id = %agent_id,
                role = %job.role.as_str(),
                workspace = %job.workspace_name,
                "Message router: pinned role with empty user — refusing to run unpinned, skipping job"
            );
            confirm_pending_delivery(&job).await;
            continue;
        };
        job.workspace_name = pinned_ws;

        debug!(
            agent_id = %agent_id,
            workspace = %job.workspace_name,
            user = %job.user_name,
            kind = ?job.kind,
            "Message router: processing job",
        );

        // ── Resolve workspace by name ─────────────────────────────────
        let ws = match crate::users::resolve_workspace(&job.workspace_name).await {
            Ok(Some(ws)) => ws,
            Ok(None) => {
                error!(
                    agent_id = %agent_id,
                    workspace = %job.workspace_name,
                    "Message router: workspace not found — skipping job",
                );
                continue;
            }
            Err(e) => {
                error!(
                    agent_id = %agent_id,
                    workspace = %job.workspace_name,
                    error = %e,
                    "Message router: failed to look up workspace — skipping job",
                );
                continue;
            }
        };

        // ── Role is embedded directly in the job ────────────────────────
        // No agent-ID string parsing needed — every caller knows the role.
        let role = job.role;

        // ── Resolve users for response delivery ───────────────────────
        // Manager: broadcast to all workspace users.
        // Other roles: deliver to the specific user.
        let users: Vec<UserRecord> = if role == Role::Manager {
            match crate::users::USER_STORE.get() {
                Some(store) => store
                    .find_by_workspace(&job.workspace_name)
                    .await
                    .unwrap_or_default(),
                None => Vec::new(),
            }
        } else {
            match resolve_single_user(&job.user_name).await {
                Some(user) => vec![user],
                None => Vec::new(),
            }
        };

        // ── Typing indicators ─────────────────────────────────────────
        let typing_tasks = setup_telegram_typing(&users).await;
        broadcast_typing(&users, &job.workspace_name, true);

        // ── Chronicle timeline drain (Manager only) ────────────────────────
        // TicketComment jobs should NEVER reach the consumer loop — they are
        // delivered directly via try_route() to agents that drain them in
        // llm_loop. If one arrives here, someone used route() instead of
        // try_route(), or the agent's receiver was dropped.
        let message = match (role, job.kind) {
            (Role::Manager, MessageKind::UserMessage | MessageKind::AgentMessage) => {
                // Flush the CDC drainer so a transition committed moments ago is
                // materialized into the chronicle before the manager reads it
                // (delayed, not lost, if skipped — the next drain delivers it).
                if let Err(e) =
                    crate::db::cdc::drain_once(&crate::pipeline::board::store().conn).await
                {
                    tracing::warn!(error = %e, "CDC flush before Manager user message failed");
                }
                let drained = crate::pipeline::chronicle::drain(&job.workspace_name).await;
                if drained.is_empty() {
                    job.content.clone()
                } else {
                    format!("{drained}\n{content}", content = job.content)
                }
            }
            (_, MessageKind::TicketComment) => {
                warn!(
                    agent_id = %agent_id,
                    "Consumer loop received TicketComment — was try_route() used instead of route()? Discarding",
                );
                continue;
            }
            _ => job.content.clone(),
        };

        // ── Run the agent ─────────────────────────────────────────────
        // Full-access (admin) users get the Assistant's full-permission toolset
        // and role prompt; everyone else runs the base Assistant.
        let full_access = users.first().is_some_and(UserRecord::is_admin);
        let (agent, response) = crate::agent::run_agent(
            agent_id.clone(),
            role,
            &ws,
            None,
            &message,
            job.user_name.clone(),
            job.channel.clone(),
            full_access,
            None,
            false,
            None,
            None,
            None,
        )
        .await;

        // ── Stop typing ───────────────────────────────────────────────
        for (cancel, handle) in typing_tasks {
            cancel.cancel();
            stop_typing(handle).await;
        }
        broadcast_typing(&users, &job.workspace_name, false);

        let Some(response) = response else {
            // Send emoji error only for UserMessage jobs where the agent truly
            // failed (not cancelled or shutdown). Internal job kinds
            // (TicketNotify, AnalyzeToolResult, ResearchResult) get no feedback.
            //
            // We check both the agent-specific token AND the
            // global shutdown token because during SIGTERM/SIGINT the global
            // token fires first — work() catches it and returns None, but the
            // agent-specific token may not have been cancelled yet.
            //
            // Drained agents must NOT emit the failure emoji: their round was
            // cut short by the graceful-drain window (or shutdown), not a
            // failure.
            if job.kind == MessageKind::UserMessage
                && !agent.is_cancelled()
                && !crate::shutdown::aborting()
            {
                deliver_unregistered_user_response(AGENT_FAILURE_EMOJI, &job, &role).await;
            }
            confirm_pending_delivery(&job).await;
            continue;
        };

        // Sleep-ended turn: the Assistant deliberately went silent waiting for
        // new input — deliver nothing (no empty bubble, no failure emoji), but
        // still confirm the at-least-once pending-jobs boundary.
        if agent.sleep_ended {
            confirm_pending_delivery(&job).await;
            continue;
        }

        // ── Response delivery ─────────────────────────────────────────
        // Manager: broadcast + persist to all workspace users.
        // Other roles: broadcast to all of the triggering user's channel
        // bindings (or use fallback for unregistered users).
        match role {
            Role::Manager => {
                deliver_manager_response(&response, &users, &job).await;
                if let Some(reply_to) = job.reply_to_agent_id.clone() {
                    route_manager_reply(&response, &job, reply_to).await;
                }
            }
            _ => {
                if users.is_empty() {
                    deliver_unregistered_user_response(&response, &job, &role).await;
                } else {
                    deliver_single_user_response(&response, &users[0], &job, &role).await;
                }
            }
        }
        confirm_pending_delivery(&job).await;
    }
}

/// Consumer-confirmed delivery: delete the durable pending_jobs row after the
/// agent ran (at-least-once boundary — the row is created before routing and
/// reclaimed only here or by boot replay). Also invoked on the
/// pin-refusal path (a pinned-role envelope with an empty user): the consumer
/// accept-and-discards the job, and the row is reclaimed here so boot replay
/// does not resurrect it. One sync retry on failure, then
/// log-and-continue: residual duplicate re-delivery at next boot is bounded
/// and accepted. Never called on workspace-not-found (the consumer skips the
/// job) — a pending row for a deleted workspace re-routes and consumer-skips
/// at every boot until the workspace returns; bounded and accepted (purge
/// reclaims only `jobs`, never `pending_jobs` — the at-least-once guarantee
/// keeps unconfirmed rows alive).
async fn confirm_pending_delivery(job: &AgentJob) {
    let Some(id) = job.pending_job_id.as_deref() else {
        return;
    };
    for attempt in 0..2 {
        match crate::jobs::delete_pending_job(&crate::session::store().conn, id).await {
            Ok(()) => return,
            Err(e) if attempt == 0 => {
                warn!(
                    pending_job = %id,
                    error = %e,
                    "Failed to confirm pending delivery — retrying once",
                );
            }
            Err(e) => {
                warn!(
                    pending_job = %id,
                    error = %e,
                    "Pending delivery confirm failed — duplicate re-delivery at next boot accepted",
                );
            }
        }
    }
}

// ── Typing helpers ────────────────────────────────────────────────────────

/// Set up Telegram typing indicators for the given users. Returns a list of
/// (CancellationToken, JoinHandle) pairs — one per unique Telegram chat.
async fn setup_telegram_typing(
    users: &[UserRecord],
) -> Vec<(CancellationToken, tokio::task::JoinHandle<()>)> {
    let telegram_channel = crate::channel_registry().get("telegram");
    let Some(ref tg_channel) = telegram_channel else {
        return Vec::new();
    };

    let mut typing_tasks = Vec::new();
    let mut seen_targets = HashSet::new();

    for user in users {
        let Some(telegram_binding) = user.channels.iter().find(|b| b.channel == "telegram") else {
            continue;
        };
        let Some(reply_target) = &telegram_binding.reply_target else {
            continue;
        };
        if !seen_targets.insert(reply_target.clone()) {
            continue;
        }

        let Some(recipient) = tg_channel.resolve_recipient(&user.name, reply_target) else {
            continue;
        };

        if let Err(e) = tg_channel.start_typing(&recipient).await {
            debug!("Message router: telegram start_typing failed: {e}");
        }

        let cancel = CancellationToken::new();
        let handle = spawn_scoped_typing_task(recipient, "telegram".to_string(), cancel.clone());
        typing_tasks.push((cancel, handle));
    }

    typing_tasks
}

/// Broadcast typing indicators to the GUI for the given users.
fn broadcast_typing(users: &[UserRecord], workspace: &str, is_typing: bool) {
    if let Some(tx) = crate::CHAT_BROADCAST.get() {
        for user in users {
            let _ = tx.send(ChatEvent::Typing {
                user_name: user.name.clone(),
                is_typing,
                workspace: workspace.to_string(),
            });
        }
    }
}

// ── Response delivery ─────────────────────────────────────────────────────

/// Outcome of attempting to deliver a response on a single channel.
enum DeliverOutcome {
    Sent,
    /// Recipient is not reachable on this channel.
    Unresolvable,
    /// Transport send failed.
    Failed(anyhow::Error),
}

/// Resolve the recipient and send `response` on `channel`.
///
/// Returns the outcome so each delivery function applies its own per-variant
/// logging — resolve-miss handling (silent skip vs warn+abort) and error
/// wording differ between manager / single-user / unregistered delivery.
async fn deliver_on_channel(
    channel: &dyn Channel,
    user_name: &str,
    reply_target: &str,
    response: &str,
) -> DeliverOutcome {
    let Some(recipient) = channel.resolve_recipient(user_name, reply_target) else {
        return DeliverOutcome::Unresolvable;
    };
    match channel
        .send(&SendMessage {
            content: response.to_string(),
            recipient,
            reply_markup: None,
        })
        .await
    {
        Ok(()) => DeliverOutcome::Sent,
        Err(e) => DeliverOutcome::Failed(e),
    }
}

/// Build the addressed `<manager-reply>` envelope (user-invisible, routed into
/// the originating Assistant's session). `None` when `response` is empty or
/// when the reply's pinned workspace cannot be resolved (no user identity).
fn manager_reply_job(response: &str, job: &AgentJob, reply_to: &str) -> Option<AgentJob> {
    if response.is_empty() {
        return None;
    }
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("manager_reply.md"),
        &[
            ("{{workspace}}", job.workspace_name.as_str()),
            ("{{message}}", response),
        ],
    );
    // Reply leg targets the originating Assistant's own pinned personal
    // workspace (carried via reply_workspace_name); legacy in-flight
    // envelopes predate the field. Re-apply the personal pin so a stale
    // project-workspace value can never route the reply outside the personal
    // session. No resolvable user -> refuse the reply leg.
    let reply_ws = job
        .reply_workspace_name
        .clone()
        .unwrap_or_else(|| crate::users::personal_workspace_name(&job.user_name));
    let Some(workspace_name) =
        crate::users::enforce_personal_pinning(crate::Role::Assistant, &reply_ws, &job.user_name)
    else {
        error!(
            reply_to_agent_id = %reply_to,
            workspace = %reply_ws,
            "Manager reply leg: pinned role with empty user — refusing to route unpinned, dropping reply"
        );
        return None;
    };
    Some(AgentJob {
        content,
        workspace_name,
        user_name: job.user_name.clone(),
        channel: "gui".to_string(),
        kind: MessageKind::ManagerReply,
        role: crate::Role::Assistant,
        reply_target: None,
        pending_job_id: None,
        reply_to_agent_id: Some(reply_to.into()),
        reply_workspace_name: None,
    })
}

/// Addressed reply leg for `AgentMessage` jobs: after the Manager's response
/// is broadcast (unchanged semantics), a durable user-invisible copy is routed
/// into the originating Assistant's session so it wakes even after a restart.
/// Persisted BEFORE routing (at-least-once, same class as fire_alarm); a
/// persist failure degrades to at-most-once with a warn.
async fn route_manager_reply(response: &str, job: &AgentJob, reply_to: String) {
    let Some(mut reply_job) = manager_reply_job(response, job, &reply_to) else {
        return;
    };
    let (persisted, id) = persist_manager_envelope(&reply_job, "manager reply envelope").await;
    if let Some(id) = id {
        reply_job.pending_job_id = Some(id);
    }
    if crate::shutdown::aborting() && persisted {
        return;
    }
    route(&crate::jobs::envelope_target(&reply_job), reply_job);
}

/// Deliver a response to all workspace users (Manager role).
async fn deliver_manager_response(response: &str, users: &[UserRecord], job: &AgentJob) {
    if users.is_empty() {
        warn!(
            workspace = %job.workspace_name,
            "Message router [manager]: no users with workspace — response delivered to nobody",
        );
    }
    deliver_agent_response_to_workspace(response, users, Role::Manager, &job.workspace_name).await;
}

/// Broadcast + persist one shared-broadcast_id copy per unique workspace user,
/// then transport-deliver to every channel binding. The shared broadcast id
/// lets the workspace chat stream dedupe the per-user copies exactly. Skips
/// silently when `users` is empty.
pub(crate) async fn deliver_agent_response_to_workspace(
    response: &str,
    users: &[UserRecord],
    role: Role,
    workspace: &str,
) {
    // One broadcast id shared by every per-user copy of this dispatch, so the
    // workspace stream can dedupe them exactly (distinct copies share it).
    let broadcast_id = Some(crate::generate_id());
    let agent_role = Some(role.as_str().to_string());

    // ── Broadcast + persist once per user ───────────────────────
    {
        let mut seen_names = HashSet::new();
        for user in users {
            if !seen_names.insert(&user.name) {
                continue;
            }
            let channel = user.channels.first().map_or("gui", |b| b.channel.as_str());
            broadcast_and_persist_agent_response(
                &user.name,
                channel,
                response,
                agent_role.clone(),
                workspace,
                broadcast_id.clone(),
            )
            .await;
        }
    }

    if !users.is_empty() {
        deliver_response_over_channels(response, users, role, workspace).await;
    }
}

/// Transport-deliver `response` to every channel binding of `users`
/// (broadcast + chat_history persistence happen at the call sites). Telegram
/// deliveries get the per-role attribution prefix when the recipient can
/// switch roles. Logs both diagnostics per reachable binding: an
/// unresolvable recipient (warn — the response is persisted but not
/// transport-delivered) and a transport failure (error). `workspace` names
/// the workspace in the no-channels diagnostic (which also names the affected
/// users).
async fn deliver_response_over_channels(
    response: &str,
    users: &[UserRecord],
    role: Role,
    workspace: &str,
) {
    let channels = crate::channel_registry().list();
    if channels.is_empty() {
        let user_names = users
            .iter()
            .map(|u| u.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        error!(
            role = %role,
            workspace = %workspace,
            users = %user_names,
            "Message router: no channels registered"
        );
        return;
    }

    for (channel_name, channel) in &channels {
        for user in users {
            let content = telegram_delivery_content(channel_name, role, &user.roles, response);
            for binding in &user.channels {
                if binding.channel != *channel_name {
                    continue;
                }
                let reply_target = binding.reply_target.as_deref().unwrap_or(&user.name);
                match deliver_on_channel(channel.as_ref(), &user.name, reply_target, &content).await
                {
                    DeliverOutcome::Failed(e) => error!(
                        channel = %channel_name,
                        user = %user.name,
                        "Message router [{role}]: failed to send response to {}: {e}",
                        user.name,
                    ),
                    DeliverOutcome::Unresolvable => warn!(
                        channel = %channel_name,
                        user = %user.name,
                        "Message router [{role}]: cannot resolve recipient on {} for {} — \
                         response was persisted but not delivered via transport",
                        channel_name,
                        user.name,
                    ),
                    DeliverOutcome::Sent => {}
                }
            }
        }
    }
}

/// Deliver a response to a single registered user, broadcasting it to ALL of
/// the user's channel bindings (the Manager model generalized to every role).
///
/// `user` is the resolved [`UserRecord`] for the job's sender — guaranteed
/// non-empty by the consumer loop before calling this function.
///
/// Broadcast+persist is performed exactly once per response, tagged with the
/// originating channel and a fresh `broadcast_id` (so the workspace stream
/// dedupes the agent-direction row exactly); transport delivery is delegated to
/// [`deliver_response_over_channels`], which iterates all registered channels
/// and every binding, using each binding's own reply_target. A user registered
/// on multiple channels sees the reply on all of them, regardless of the
/// channel the request came in on.
///
/// `job.reply_target` is not used to scope transport delivery for registered
/// users — each binding supplies its own reply_target on its own channel, the
/// broadcast model the Manager generalizes.
async fn deliver_single_user_response(
    response: &str,
    user: &UserRecord,
    job: &AgentJob,
    role: &Role,
) {
    // Broadcast + persist (tagged with the originating channel). A fresh
    // broadcast_id makes every NEW agent-direction row exactly dedupable;
    // legacy NULL rows keep the (agent_role, content) fallback.
    let channel = job.channel.as_str();
    broadcast_and_persist_agent_response(
        &user.name,
        channel,
        response,
        Some(role.as_str().to_string()),
        &job.workspace_name,
        Some(crate::generate_id()),
    )
    .await;

    deliver_response_over_channels(
        response,
        std::slice::from_ref(user),
        *role,
        &job.workspace_name,
    )
    .await;
}

/// Deliver a response to an unregistered user (no [`UserRecord`] in the users DB).
///
/// Falls back to the originating channel using `job.reply_target` (when
/// available) or `job.user_name` as the reply target. Works for any user
/// regardless of registration status and preserves the original message's
/// reply target (e.g. Telegram chat_id).
///
/// Also used directly by the binary for inline confirmations (e.g. Telegram
/// session-clear replies) via raw `reply_target` passthrough.
///
/// The response is always broadcast + persisted first (so it appears in the
/// GUI chat history even if the channel transport delivery fails).
pub async fn deliver_unregistered_user_response(response: &str, job: &AgentJob, role: &Role) {
    let ch = job.channel.as_str();

    // Invariant: never persist/reply under an empty user — seed 'admin'
    // (covers inline confirmations like the clear-session reply path).
    let user_name =
        crate::session::normalize_user_name(&job.user_name, "deliver_unregistered_user_response");

    // Broadcast + persist (works with just strings, no UserRecord needed). A
    // fresh broadcast_id makes the new agent row exactly dedupable; legacy
    // NULL rows keep the (agent_role, content) fallback.
    broadcast_and_persist_agent_response(
        user_name,
        ch,
        response,
        Some(role.as_str().to_string()),
        &job.workspace_name,
        Some(crate::generate_id()),
    )
    .await;

    // Try to send via the originating channel.
    let Some(chan) = crate::channel_registry().get(ch) else {
        return;
    };

    // Use reply_target from the original message when available (e.g. Telegram
    // chat_id), falling back to user_name.
    let reply_target = job.reply_target.as_deref().unwrap_or(user_name);
    match deliver_on_channel(chan.as_ref(), user_name, reply_target, response).await {
        DeliverOutcome::Unresolvable => warn!(
            workspace = %job.workspace_name,
            user = %user_name,
            channel = %ch,
            "Message router [{role}]: cannot resolve recipient for unregistered user — \
             response was persisted but not delivered via transport",
        ),
        DeliverOutcome::Failed(e) => error!(
            channel = %ch,
            user = %user_name,
            "Message router [{role}]: failed to send response to unregistered user: {e}",
        ),
        DeliverOutcome::Sent => {}
    }
}

/// Resolve a single user by name, returning their full record if available.
/// Uses a targeted database query instead of loading all users.
async fn resolve_single_user(user_name: &str) -> Option<UserRecord> {
    let store = crate::users::USER_STORE.get()?;
    match store.find_by_name(user_name).await {
        Ok(Some(user)) => Some(user),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;
    use crate::channels::GuiChannel;
    use std::sync::Arc;

    // ── Consumer loop lifecycle tests ─────────────────────────────────

    /// Shortcut to construct a minimal [`AgentJob`] for lifecycle tests.
    fn make_job(role: Role, workspace: &str, user: &str, channel: &str) -> AgentJob {
        AgentJob {
            content: String::new(),
            workspace_name: workspace.to_string(),
            user_name: user.to_string(),
            channel: channel.to_string(),
            kind: MessageKind::UserMessage,
            role,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        }
    }

    #[tokio::test]
    async fn test_route_creates_consumer_entry() {
        let _ = init_global();
        let id = "_test_ua_creates_entry";

        route(id, make_job(Role::Assistant, "", "", ""));

        let map = ROUTER.get().unwrap();
        let guard = map.read().unwrap_poison();
        assert!(guard.contains_key(id));
        drop(guard);

        // Cleanup
        let map = ROUTER.get().unwrap();
        let mut guard = map.write().unwrap_poison();
        guard.remove(id);
    }

    #[tokio::test]
    async fn test_route_reuses_existing_consumer() {
        let _ = init_global();
        let id = "_test_ua_reuses";

        // Slow path — first message creates consumer
        route(id, make_job(Role::Assistant, "ws", "alice", "gui"));

        // Fast path — second message uses same consumer
        route(id, make_job(Role::Assistant, "ws", "bob", "gui"));

        let map = ROUTER.get().unwrap();
        let guard = map.read().unwrap_poison();
        assert!(guard.contains_key(id));
        drop(guard);

        // Cleanup
        let map = ROUTER.get().unwrap();
        let mut guard = map.write().unwrap_poison();
        guard.remove(id);
    }

    #[tokio::test]
    async fn test_route_multiple_agents_get_separate_consumers() {
        let _ = init_global();
        let id_a = "_test_ua_mult_a";
        let id_b = "_test_ua_mult_b";

        route(id_a, make_job(Role::Assistant, "ws", "alice", "gui"));
        route(id_b, make_job(Role::Engineer, "ws", "bob", "gui"));

        let map = ROUTER.get().unwrap();
        let guard = map.read().unwrap_poison();
        assert!(guard.contains_key(id_a));
        assert!(guard.contains_key(id_b));
        drop(guard);

        // Cleanup
        let map = ROUTER.get().unwrap();
        let mut guard = map.write().unwrap_poison();
        guard.remove(id_a);
        guard.remove(id_b);
    }

    /// Verify that the consumer loop exits gracefully when the channel's
    /// last sender is dropped (causing `rx.recv()` to return `None`), AND
    /// that the consumer's cleanup wrapper removes the entry from the router
    /// table.
    ///
    /// We call [`route()`] which spawns the consumer and stores the sender.
    /// We then remove the sender and drop it to close the channel, but
    /// re-insert a dummy sender so the cleanup wrapper (which runs after the
    /// consumer exits) removes an actual entry from the map.
    #[tokio::test]
    async fn test_consumer_loop_exits_gracefully_on_sender_drop() {
        let _ = init_global();
        let id = "_test_ua_on_close";

        // Route a job — creates the consumer and stores the sender.
        route(id, make_job(Role::Assistant, "", "", ""));

        // Verify consumer is registered.
        assert!(
            ROUTER
                .get()
                .unwrap()
                .read()
                .unwrap_poison()
                .contains_key(id),
            "consumer should be registered after first route",
        );

        // Remove and drop the ONLY sender — this closes the channel.
        let dropped_tx = ROUTER
            .get()
            .unwrap()
            .write()
            .unwrap_poison()
            .remove(id)
            .expect("sender should exist");
        drop(dropped_tx);

        // Re-insert a dummy sender so the consumer's cleanup wrapper has an
        // entry to remove (proving it actually runs after the consumer exits).
        let (dummy_tx, _) = mpsc::unbounded_channel::<AgentJob>();
        ROUTER
            .get()
            .unwrap()
            .write()
            .unwrap_poison()
            .insert(id.to_string(), dummy_tx);

        // Poll until the consumer loop exits and its cleanup wrapper removes
        // the dummy entry.
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(2);
        let mut cleaned_up = false;
        while tokio::time::Instant::now() < deadline {
            if !ROUTER
                .get()
                .unwrap()
                .read()
                .unwrap_poison()
                .contains_key(id)
            {
                cleaned_up = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        assert!(
            cleaned_up,
            "consumer should have exited and its cleanup wrapper should have removed the entry",
        );
    }

    // ── Workspace resolution tests ─────────────────────────────────────

    /// Resolving a workspace that exists in the DB returns it.
    #[tokio::test]
    // Shares the process-global stores with the provider-group tests (uniform exclusion).
    #[serial_test::serial(provider)]
    async fn test_resolve_workspace_found() {
        crate::util::test::init_management_test_stores().await;

        crate::util::test::create_test_workspace("/tmp/test_resolve_ws", "test_resolve_ws").await;

        let result = crate::users::resolve_workspace("test_resolve_ws").await;
        let resolved = result.expect("resolve should succeed for DB workspace");
        assert!(resolved.is_some(), "DB workspace should be found");
        assert_eq!(resolved.unwrap().name, "test_resolve_ws");
    }

    /// Resolving a personal workspace constructs it on the fly when
    /// it is NOT in the DB.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_workspace_personal() {
        crate::util::test::init_management_test_stores().await;

        let result = crate::users::resolve_workspace("personal:liliana").await;
        let resolved = result.expect("resolve should succeed for personal workspace");
        let ws = resolved.expect("personal workspace should be constructed on the fly");

        assert_eq!(ws.name, "personal:liliana");
        assert_eq!(ws.status, crate::WorkspaceStatus::Ready);
        // Path should point to the userspace directory.
        let expected_path = crate::users::personal_workspace_path("liliana");
        assert_eq!(ws.path, expected_path);
    }

    /// Resolving a workspace that genuinely does not exist (and is not
    /// a personal workspace) returns `Ok(None)`.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_workspace_not_found() {
        crate::util::test::init_management_test_stores().await;

        let result = crate::users::resolve_workspace("nonexistent_workspace").await;
        let resolved = result.expect("resolve should succeed (no error) for missing workspace");
        assert!(
            resolved.is_none(),
            "nonexistent workspace should not be found",
        );
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// A channel name unique to the message_router delivery tests, so the
    /// spy registration never collides with a real "telegram" channel that a
    /// parallel test (e.g. telegram_tests) may have already registered.
    const TEST_SPY_CHANNEL: &str = "__test_spy_channel";

    /// Set up DB stores + channel registry for response-delivery tests.
    async fn setup_response_test_infra() {
        crate::util::test::init_management_test_stores().await;
        let _ = crate::CHANNEL_REGISTRY.set(crate::ChannelRegistry::default());
        let (gui_channel, _gui_tx) = GuiChannel::new();
        crate::channel_registry().register(Arc::new(gui_channel));
    }

    // ── Panic cleanup test ───────────────────────────────────────────────

    /// Verify that a panicking consumer does NOT leave a dead sender in the
    /// router table.  The outer wrapper in [`route()`] uses `catch_unwind`
    /// so that cleanup runs even on panic.
    ///
    /// We insert a sender, spawn a consumer that panics, and verify the
    /// entry is removed — this exercises the exact same cleanup pattern
    /// used in `route()`.
    #[tokio::test]
    async fn test_catch_unwind_cleanup_removes_entry_on_panic() {
        let _ = init_global();
        let id = "_test_panic_cleanup";

        // Insert a sender into the router table.
        let (tx, rx) = mpsc::unbounded_channel::<AgentJob>();
        {
            let mut guard = ROUTER.get().unwrap().write().unwrap_poison();
            guard.insert(id.to_string(), tx);
        }

        // Spawn the same catch_unwind + cleanup pattern used by route().
        let agent_id = id.to_string();
        let agent_id_for_cleanup = agent_id.clone();
        tokio::spawn(async move {
            let _result = std::panic::AssertUnwindSafe(async {
                // Drop the receiver to break out of any pending recv(),
                // then panic — simulating a consumer_loop failure.
                drop(rx);
                panic!("simulated consumer panic");
            })
            .catch_unwind()
            .await;

            // Cleanup — identical to route()'s wrapper.
            if let Some(map) = ROUTER.get() {
                let mut guard = map.write().unwrap_poison();
                guard.remove(&agent_id_for_cleanup);
            }
        });

        // Wait for the panic + cleanup to complete.
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(2);
        let mut cleaned_up = false;
        while tokio::time::Instant::now() < deadline {
            let found = {
                let guard = ROUTER.get().unwrap().read().unwrap_poison();
                guard.contains_key(id)
            };
            if !found {
                cleaned_up = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Verify entry was cleaned up despite the panic.
        assert!(
            cleaned_up,
            "entry should have been removed from router table after consumer panic",
        );
    }

    // ── Response delivery tests ──────────────────────────────────────────
    //
    // These are smoke tests: they verify that the delivery functions complete
    // without panic when called with realistic inputs.  They do NOT assert on
    // actual delivery outcomes (what was sent, via which channel, to which
    // recipient) because the channel transports (GuiChannel::send is a no-op
    // in this test context) and the broadcast/persist path are hard to mock
    // at this level of abstraction.
    //
    // Future engineers: if you add assertions here, you will likely need to
    // mock the channel registry or provide a test channel with observable
    // send().  Until then, the "no panic" guarantee at least proves that the
    // delivery functions handle edge cases (empty bindings, missing users,
    // etc.) without crashing.

    /// `resolve_single_user` returns a [`UserRecord`] when the user exists.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_single_user_found() {
        setup_response_test_infra().await;

        // The admin user is auto-created by ensure_admin_user.
        let user = resolve_single_user("admin").await;
        assert!(user.is_some(), "admin user should exist after store init");
        assert_eq!(user.as_ref().unwrap().name, "admin");
    }

    /// `resolve_single_user` returns `None` for a non-existent user.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_single_user_not_found() {
        setup_response_test_infra().await;

        let user = resolve_single_user("nonexistent_user").await;
        assert!(user.is_none(), "non-existent user should return None");
    }

    /// `deliver_unregistered_user_response` completes without error when
    /// the channel is registered, a user is NOT in the DB, and the fallback
    /// path to `reply_target` is exercised.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_unregistered_user_response() {
        setup_response_test_infra().await;

        let job = AgentJob {
            content: "hello from unregistered user".to_string(),
            workspace_name: "default".to_string(),
            user_name: "unregistered_alice".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::UserMessage,
            role: Role::Assistant,
            reply_target: Some("chat_123".to_string()),
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        };

        // Should complete without panic.
        deliver_unregistered_user_response("response to unregistered user", &job, &Role::Assistant)
            .await;
    }

    /// `deliver_single_user_response` completes without error and broadcasts
    /// to all of the user's channel bindings (the Manager model generalized to
    /// a single role). Here admin is bound to "gui" and the job originates on
    /// "gui" — broadcast+persist runs and transport delivery reaches the
    /// matching "gui" binding. No-panic is the assertion.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_single_user_response() {
        setup_response_test_infra().await;

        // Give the admin user a "gui" channel binding so the delivery
        // function can find it.
        let store = crate::users::USER_STORE.get().unwrap();
        store
            .bind_channel("admin", "gui", "admin")
            .await
            .expect("bind admin to gui channel");

        let user = resolve_single_user("admin").await.unwrap();

        let job = AgentJob {
            content: "hello from registered user".to_string(),
            workspace_name: "default".to_string(),
            user_name: "admin".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::UserMessage,
            role: Role::Assistant,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        };

        // Should complete without panic — broadcasts to the "gui" binding.
        deliver_single_user_response("response to registered user", &user, &job, &Role::Assistant)
            .await;
    }

    /// `deliver_single_user_response` handles the case where the user has NO
    /// channel bindings at all — only broadcast+persist runs, transport
    /// delivery is skipped. No-panic is the assertion.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_single_user_no_bindings() {
        setup_response_test_infra().await;

        // Admin user exists but has no "gui" channel binding.
        let user = resolve_single_user("admin").await.unwrap();

        let job = AgentJob {
            content: "hello".to_string(),
            workspace_name: "default".to_string(),
            user_name: "admin".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::UserMessage,
            role: Role::Assistant,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        };

        // Should complete without panic — broadcast+persist runs, transport
        // delivery is skipped because there are no matching bindings.
        deliver_single_user_response(
            "response to registered user without matching binding",
            &user,
            &job,
            &Role::Assistant,
        )
        .await;
    }

    /// `deliver_single_user_response` broadcasts the reply to ALL of the user's
    /// channel bindings. Admin is bound to both "gui" and a registered spy
    /// channel; the job originates on "gui" but the reply reaches every
    /// registered channel the user is bound to. The spy captures at least one
    /// send.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_single_user_broadcasts_to_all_bindings() {
        setup_response_test_infra().await;

        let store = crate::users::USER_STORE.get().unwrap();
        store
            .bind_channel("admin", "gui", "admin")
            .await
            .expect("bind admin to gui channel");

        // Register a spy channel under a unique name so it is always ours.
        let (spy, sent) = crate::util::test::SpyChannel::new(TEST_SPY_CHANNEL);
        crate::channel_registry().register(Arc::new(spy) as Arc<dyn crate::Channel>);
        store
            .bind_channel("admin", TEST_SPY_CHANNEL, "admin")
            .await
            .expect("bind admin to spy channel");

        let user = resolve_single_user("admin").await.unwrap();

        let job = AgentJob {
            content: "hello".to_string(),
            workspace_name: "default".to_string(),
            user_name: "admin".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::UserMessage,
            role: Role::Assistant,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        };

        deliver_single_user_response("broadcast to all bindings", &user, &job, &Role::Assistant)
            .await;

        let captured = sent.lock().unwrap_poison();
        assert!(
            !captured.is_empty(),
            "spy channel should have captured at least one broadcast send",
        );
    }

    /// `deliver_manager_response` broadcasts to all workspace users without
    /// panic when users have channel bindings.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_manager_response_with_users() {
        setup_response_test_infra().await;

        let store = crate::users::USER_STORE.get().unwrap();
        store
            .bind_channel("admin", "gui", "admin")
            .await
            .expect("bind admin to gui channel");

        let user = resolve_single_user("admin").await.unwrap();

        let job = AgentJob {
            content: "manager broadcast".to_string(),
            workspace_name: "default".to_string(),
            user_name: "admin".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::TicketNotify,
            role: Role::Manager,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        };

        deliver_manager_response("manager response", &[user], &job).await;
    }

    /// Regression: the Manager transport loop must skip bindings whose channel
    /// differs from the delivering transport (same filter the single-user path
    /// applies). The user is bound only to `SPY_MATCH`; a second registered spy
    /// channel must capture nothing — without the channel filter, the loop
    /// would replay the matching binding's delivery on every registered
    /// channel. Uses a dedicated user because the users store and channel
    /// registry are process-global and other tests deliver for `admin`
    /// concurrently.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_manager_response_skips_foreign_channel_bindings() {
        const SPY_MATCH: &str = "__test_mgr_spy_match";
        const SPY_OTHER: &str = "__test_mgr_spy_other";
        const USER: &str = "__test_mgr_skip_user";
        setup_response_test_infra().await;

        let store = crate::users::USER_STORE.get().unwrap();
        store
            .add_user(USER, None, Role::Assistant)
            .await
            .expect("create dedicated test user");
        store
            .bind_channel(USER, SPY_MATCH, "mgr-spy-target")
            .await
            .expect("bind user to matching spy channel");
        store
            .update_channel_contact(SPY_MATCH, "mgr-spy-target", "mgr-spy-target")
            .await
            .expect("set reply_target on the spy binding");

        let (match_spy, match_sent) = crate::util::test::SpyChannel::new(SPY_MATCH);
        let (other_spy, other_sent) = crate::util::test::SpyChannel::new(SPY_OTHER);
        crate::channel_registry().register(Arc::new(match_spy) as Arc<dyn crate::Channel>);
        crate::channel_registry().register(Arc::new(other_spy) as Arc<dyn crate::Channel>);

        let user = resolve_single_user(USER).await.unwrap();
        let job = AgentJob {
            content: "manager broadcast".to_string(),
            workspace_name: "default".to_string(),
            user_name: USER.to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::TicketNotify,
            role: Role::Manager,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: None,
            reply_workspace_name: None,
        };

        deliver_manager_response("manager response", &[user], &job).await;

        let captured = match_sent.lock().unwrap_poison();
        assert!(
            captured.len() == 1
                && captured[0].recipient == "mgr-spy-target"
                && captured[0].content == "manager response",
            "matching spy channel should deliver exactly once to the binding's \
             reply_target, got: {captured:?}",
        );
        assert!(
            other_sent.lock().unwrap_poison().is_empty(),
            "spy channel without a matching binding must not receive the manager response",
        );
    }

    // ── Job builder tests ───────────────────────────────────────────────

    /// `agent_message_job` threads the normalized identity + addressing fields.
    #[test]
    fn agent_message_job_threading() {
        let job = agent_message_job(
            "hello".to_string(),
            "proj_ws".to_string(),
            "origin_ws".to_string(),
            "alice",
            "agent_123".to_string(),
        );
        assert_eq!(job.user_name, "alice");
        assert_eq!(job.kind, MessageKind::AgentMessage);
        assert_eq!(job.role, Role::Manager);
        assert_eq!(job.workspace_name, "proj_ws");
        assert_eq!(job.reply_to_agent_id.as_deref(), Some("agent_123"));
        assert_eq!(job.reply_workspace_name.as_deref(), Some("origin_ws"));

        // Empty user_name normalizes to the seeded 'admin'.
        let job = agent_message_job(
            "hello".to_string(),
            "proj_ws".to_string(),
            "origin_ws".to_string(),
            "",
            "agent_124".to_string(),
        );
        assert_eq!(job.user_name, "admin");
    }

    /// `manager_reply_job` addresses the reply into the originating Assistant's
    /// session; the carried reply workspace is re-pinned to the Assistant's
    /// personal workspace (pinned roles never run in a non-personal workspace).
    #[test]
    fn manager_reply_job_threading() {
        let source = AgentJob {
            content: "assistant msg".to_string(),
            workspace_name: "team_ws".to_string(),
            user_name: "alice".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::AgentMessage,
            role: Role::Manager,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: Some("assistant_agent_1".to_string()),
            reply_workspace_name: Some("proj_ws".to_string()),
        };
        let reply = manager_reply_job("reply text", &source, "assistant_agent_1")
            .expect("non-empty response builds a reply job");
        // The reply routes into the originating Assistant's personal workspace
        // (the carried project workspace is re-pinned), while the envelope tags
        // the Manager's workspace (job.workspace_name).
        assert_eq!(reply.workspace_name, "personal:alice");
        assert_eq!(reply.kind, MessageKind::ManagerReply);
        assert_eq!(reply.role, Role::Assistant);
        assert_eq!(
            reply.reply_to_agent_id.as_deref(),
            Some("assistant_agent_1")
        );
        assert_eq!(reply.user_name, "alice");
        assert!(reply.content.contains("team_ws"));
        assert!(reply.content.contains("reply text"));
    }

    /// Legacy in-flight AgentMessage envelopes predate `reply_workspace_name`;
    /// the reply leg falls back to the personal-workspace pin.
    #[test]
    fn manager_reply_job_legacy_fallback() {
        let source = AgentJob {
            content: "assistant msg".to_string(),
            workspace_name: "team_ws".to_string(),
            user_name: "bob".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::AgentMessage,
            role: Role::Manager,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: Some("assistant_agent_2".to_string()),
            reply_workspace_name: None,
        };
        let reply = manager_reply_job("reply text", &source, "assistant_agent_2")
            .expect("non-empty response builds a reply job");
        assert_eq!(
            reply.workspace_name,
            crate::users::personal_workspace_name("bob")
        );
    }

    /// Persist-before-route round trip: a ManagerReply envelope written to
    /// pending_jobs reads back as the same job (kind + workspace), targeted at
    /// the addressed reply_to_agent_id. Cleans up its row so other tests'
    /// boot-replay paths never see it.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn manager_reply_persist_round_trip() {
        setup_response_test_infra().await;
        let source = AgentJob {
            content: "assistant msg".to_string(),
            workspace_name: "team_ws".to_string(),
            user_name: "alice".to_string(),
            channel: "gui".to_string(),
            kind: MessageKind::AgentMessage,
            role: Role::Manager,
            reply_target: None,
            pending_job_id: None,
            reply_to_agent_id: Some("assistant_agent_roundtrip".to_string()),
            reply_workspace_name: Some("proj_ws".to_string()),
        };
        let job = manager_reply_job("reply text", &source, "assistant_agent_roundtrip")
            .expect("non-empty response builds a reply job");
        let (persisted, id) = persist_manager_envelope(&job, "test").await;
        assert!(persisted, "persist should succeed");
        let id = id.expect("a persisted row id should be returned");

        let conn = &crate::session::store().conn;
        let pending = crate::jobs::list_pending_jobs(conn)
            .await
            .expect("list pending");
        let matching: Vec<_> = pending
            .iter()
            .filter(|r| r.target_agent_id == crate::jobs::envelope_target(&job))
            .collect();
        assert_eq!(matching.len(), 1, "exactly one addressed pending row");
        let row = matching[0];
        assert_eq!(row.id, id);
        let deserialized: AgentJob =
            serde_json::from_str(&row.envelope).expect("envelope deserializes to AgentJob");
        assert_eq!(deserialized.kind, MessageKind::ManagerReply);
        assert_eq!(deserialized.workspace_name, "personal:alice");

        crate::jobs::delete_pending_job(conn, &id)
            .await
            .expect("delete pending row");
    }

    // ── register_agent / unregister_agent / try_route tests ────────────

    /// `register_agent` creates a router entry that `try_route` can find.
    #[tokio::test]
    async fn test_register_agent_try_route_found() {
        let _ = init_global();
        let agent_id = "_test_register_agent_found";

        let _rx = register_agent(agent_id);
        let job = make_job(Role::Assistant, "hello", "user", "gui");
        assert!(
            try_route(agent_id, job),
            "try_route should return true for a registered agent",
        );

        unregister_agent(agent_id);
    }

    /// `try_route` returns `false` when the agent is not registered.
    #[tokio::test]
    async fn test_try_route_agent_not_found() {
        let _ = init_global();
        let agent_id = "_test_try_route_not_found";

        let job = make_job(Role::Assistant, "hello", "user", "gui");
        assert!(
            !try_route(agent_id, job),
            "try_route should return false for an unregistered agent",
        );
    }

    /// `try_route` returns `false` when the receiver has been dropped
    /// (sender channel is closed).
    #[tokio::test]
    async fn test_try_route_receiver_dropped() {
        let _ = init_global();
        let agent_id = "_test_try_route_dropped";

        // Register, create the receiver but immediately drop it.
        let rx = register_agent(agent_id);
        drop(rx);

        let job = make_job(Role::Assistant, "hello", "user", "gui");
        assert!(
            !try_route(agent_id, job),
            "try_route should return false when receiver is dropped",
        );

        unregister_agent(agent_id);
    }

    /// `unregister_agent` removes the entry so `try_route` returns `false`.
    #[tokio::test]
    async fn test_unregister_agent_removes_entry() {
        let _ = init_global();
        let agent_id = "_test_unregister_agent_removes";

        let _rx = register_agent(agent_id);
        unregister_agent(agent_id);

        let job = make_job(Role::Assistant, "hello", "user", "gui");
        assert!(
            !try_route(agent_id, job),
            "try_route should return false after unregister_agent",
        );
    }

    /// Multiple agents can be registered simultaneously.
    #[tokio::test]
    async fn test_register_agent_multiple_agents() {
        let _ = init_global();
        let id_a = "_test_multi_a";
        let id_b = "_test_multi_b";

        let _rx_a = register_agent(id_a);
        let _rx_b = register_agent(id_b);

        assert!(try_route(id_a, make_job(Role::Assistant, "a", "u", "g")));
        assert!(try_route(id_b, make_job(Role::Engineer, "b", "u", "g")));

        unregister_agent(id_a);
        unregister_agent(id_b);
    }

    /// `register_agent` replaces a stale entry without panicking.
    #[tokio::test]
    async fn test_register_agent_replaces_stale_entry() {
        let _ = init_global();
        let agent_id = "_test_replace_stale";

        // First registration — drop receiver so it's stale.
        let rx = register_agent(agent_id);
        drop(rx);

        // Second registration — should replace the stale sender.
        let _rx2 = register_agent(agent_id);

        let job = make_job(Role::Assistant, "hello", "user", "gui");
        assert!(
            try_route(agent_id, job),
            "try_route should succeed after replacing stale entry",
        );

        unregister_agent(agent_id);
    }

    // ── Agent failure emoji tests ──────────────────────────────────────
    //
    // These tests verify the AGENT_FAILURE_EMOJI constant and its delivery
    // path.  They follow the same smoke-test philosophy as the response
    // delivery tests above: no assertions on actual transport outcomes,
    // just a "no panic" guarantee.

    /// The emoji constant is defined and non-empty.
    #[test]
    fn test_agent_failure_emoji_constant() {
        assert!(!AGENT_FAILURE_EMOJI.is_empty(), "emoji should be non-empty");
        // Verify it contains actual emoji characters (not just whitespace).
        assert!(
            AGENT_FAILURE_EMOJI.chars().count() >= 3,
            "emoji should be at least 3 characters"
        );
    }

    // ── Telegram role attribution tests ──────────────────────────────────

    /// The attribution prefix fires only for Telegram + 2+ role pools, and
    /// pins the concrete emoji/label table from the spec.
    #[test]
    fn test_telegram_delivery_content() {
        let response = "plain answer";

        // 0-1 roles → response passed through unchanged (borrowed, no prefix).
        let content = telegram_delivery_content("telegram", Role::Manager, &[], response);
        assert_eq!(content, "plain answer");
        assert!(
            matches!(content, Cow::Borrowed(_)),
            "no-prefix deliveries must not allocate"
        );
        let content = telegram_delivery_content(
            "telegram",
            Role::Manager,
            &["manager".to_string()],
            response,
        );
        assert_eq!(content, "plain answer");
        assert!(
            matches!(content, Cow::Borrowed(_)),
            "no-prefix deliveries must not allocate"
        );

        // Non-Telegram channels never get a prefix, even with 2+ roles.
        let multi = ["manager".to_string(), "artist".to_string()];
        let content = telegram_delivery_content("gui", Role::Manager, &multi, response);
        assert_eq!(content, "plain answer");
        assert!(
            matches!(content, Cow::Borrowed(_)),
            "gui/voice deliveries must not allocate"
        );

        // 2+ roles on Telegram → first-line attribution pins the spec table.
        assert_eq!(
            telegram_delivery_content("telegram", Role::Manager, &multi, response),
            "🤖 Manager:\nplain answer"
        );
        assert_eq!(
            telegram_delivery_content(
                "telegram",
                Role::Qa,
                &["qa".to_string(), "coder".to_string()],
                response,
            ),
            "🔨 QA:\nplain answer"
        );
    }

    /// `RecoveryRetry` is intentionally NOT `UserMessage`.  The emoji gate in
    /// `consumer_loop` uses `job.kind == MessageKind::UserMessage` to decide whether
    /// to send the failure emoji — `RecoveryRetry` is automatically excluded
    /// because the equality check is specific to `UserMessage`.
    ///
    /// The same structural invariant holds for the Assistant↔Manager kinds:
    /// `AgentMessage` and `ManagerReply` are deliberately distinct from
    /// `UserMessage` (see the `MessageKind` docs) so the emoji gate excludes them
    /// without any `matches!` broadening.
    ///
    /// This test documents the structural invariant: distinct enum variants of a
    /// `PartialEq` enum never compare equal.  The compiler and derive macro
    /// guarantee this property; this test provides executable documentation of
    /// the design decision, not a regression guard against the derived equality.
    /// If the emoji gate condition is refactored in the future (e.g., to use
    /// `matches!()` with a broader set), this test will not catch that change.
    /// A full behavioral test of the emoji gate requires exercising the consumer
    /// loop with `RecoveryRetry` jobs — infrastructure disproportionate to the
    /// zero-regression-risk invariant.
    #[test]
    fn test_recovery_retry_kind_invariant() {
        assert_ne!(
            MessageKind::RecoveryRetry,
            MessageKind::UserMessage,
            "RecoveryRetry must be a distinct variant from UserMessage — \
             the emoji gate's `== MessageKind::UserMessage` check naturally excludes it",
        );
    }
}
