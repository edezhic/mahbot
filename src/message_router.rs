//! Agent-ID-based message router — replaces per-role queue with per-agent
//! channels for true instance-level parallelism.
//!
//! Every agent instance gets its own [`mpsc::UnboundedSender`] stored in a
//! global [`HashMap`] keyed by unique agent ID. Jobs are routed directly to
//! the correct consumer — no agent ever blocks another.
//!
//! # Producer paths
//!
//! Three sources feed into [`route`]:
//! - **User messages**: chat messages from users → enqueue
//! - **Ticket notifications**: ticket transitions → enqueue
//! - **AskTool results**: async sub-agent results → enqueue
//!
//! # Agent ID formats
//!
//! Agent IDs are stable deterministic strings built by
//! [`crate::session::resolve_agent_id`] and friends. The [`Role`] is embedded
//! directly in [`AgentJob`] so the router never needs to parse the agent ID.
//!
//! # Response delivery
//!
//! - [`Role::Manager`] broadcasts to all workspace users.
//! - Other roles deliver to the specific user who triggered the job, via the
//!   originating channel (scoped to `job.channel`).

use futures_util::FutureExt;
use std::collections::{HashMap, HashSet};
use std::sync::{OnceLock, RwLock};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::channels::{
    broadcast_and_persist_agent_response, spawn_scoped_typing_task, stop_typing,
};
use crate::users::UserRecord;
use crate::util::UnwrapPoison;
use crate::{ChatEvent, Role, SendMessage, Workspace};

// ── Job definition ─────────────────────────────────────────────────────────

/// Semantic category of a queue job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    /// User-typed message (chat or inline-button callback).
    /// For Manager: the ticket transition buffer drains before the agent runs.
    /// For other roles: no ticket buffer drain.
    UserMessage,
    /// System notification from a ticket transition.
    /// Only enqueued for the Manager role.
    TicketNotify,
    /// Result from an async AskTool sub-agent, injected back into the caller's
    /// agent session.
    AskToolResult,
}

/// A single unit of work for an agent consumer.
#[derive(Debug)]
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
    pub kind: JobKind,
    /// The agent's [`Role`] — embedded directly so the router never needs
    /// to parse agent ID strings.
    pub role: Role,
    /// Original reply target from the incoming message (e.g., Telegram chat_id).
    /// Used by [`deliver_unregistered_user_response`] when there is no
    /// [`UserRecord`](crate::users::UserRecord) in the users DB.
    /// `None` for non-user-facing jobs (ticket notifications, broadcast-only).
    pub reply_target: Option<String>,
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
            // Log the panic payload for debugging.
            if let Some(msg) = panic.downcast_ref::<&str>() {
                error!(agent_id = %agent_id_for_cleanup, panic = %msg, "Consumer loop panic message");
            } else if let Some(msg) = panic.downcast_ref::<String>() {
                error!(agent_id = %agent_id_for_cleanup, panic = %msg, "Consumer loop panic message");
            }
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

// ── Workspace resolution ───────────────────────────────────────────────────

/// Resolve a workspace by name, with personal workspace fallback.
///
/// Personal workspaces (names starting with `"personal:"`) are NOT stored
/// in `workspaces.db` — they live at `~/.mahbot/userspaces/<user>/` and are
/// constructed on the fly as ephemeral [`Workspace`] structs.
///
/// Returns `Ok(Some(ws))` when the workspace is found or constructed.
/// Returns `Ok(None)` when the workspace genuinely does not exist
/// (and is not a personal workspace).
/// Returns `Err(e)` on database errors.
async fn resolve_workspace(workspace_name: &str) -> anyhow::Result<Option<Workspace>> {
    match crate::workspace::get_by_name(workspace_name).await? {
        Some(ws) => Ok(Some(ws)),
        None if crate::users::is_personal_workspace(workspace_name) => {
            let user_name = workspace_name
                .strip_prefix("personal:")
                .expect("invariant: is_personal_workspace checked prefix");
            let path = crate::users::personal_workspace_path(user_name);
            Ok(Some(crate::users::personal_workspace_struct(
                user_name, &path,
            )))
        }
        None => Ok(None),
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
#[allow(clippy::too_many_lines)]
async fn consumer_loop(agent_id: String, mut rx: mpsc::UnboundedReceiver<AgentJob>) {
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            info!(agent_id = %agent_id, "Message router: shutting down — queue drained");
            break;
        }

        let job = tokio::select! {
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

        info!(
            agent_id = %agent_id,
            workspace = %job.workspace_name,
            user = %job.user_name,
            kind = ?job.kind,
            "Message router: processing job",
        );

        // ── Resolve workspace by name ─────────────────────────────────
        let ws = match resolve_workspace(&job.workspace_name).await {
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

        // ── Ticket buffer drain (Manager only) ────────────────────────
        let message = match (role, job.kind) {
            (Role::Manager, JobKind::UserMessage) => {
                let drained = crate::ticket_buffer::drain(&job.workspace_name);
                if drained.is_empty() {
                    job.content.clone()
                } else {
                    format!("{drained}\n{content}", content = job.content)
                }
            }
            _ => job.content.clone(),
        };

        // ── Run the agent ─────────────────────────────────────────────
        let (_agent, response) = crate::agent::run_agent(
            agent_id.clone(),
            role,
            &ws,
            None,
            &message,
            job.user_name.clone(),
            job.channel.clone(),
        )
        .await;

        // ── Stop typing ───────────────────────────────────────────────
        for (cancel, handle) in typing_tasks {
            cancel.cancel();
            stop_typing(handle).await;
        }
        broadcast_typing(&users, &job.workspace_name, false);

        let Some(response) = response else {
            continue;
        };

        // ── Response delivery ─────────────────────────────────────────
        // Manager: broadcast + persist to all workspace users.
        // Other roles: send reply to the specific user (or use fallback
        // for unregistered users).
        match role {
            Role::Manager => {
                deliver_manager_response(&response, &users, &job).await;
            }
            _ => {
                if users.is_empty() {
                    deliver_unregistered_user_response(&response, &job, &role).await;
                } else {
                    deliver_single_user_response(&response, &users[0], &job, &role).await;
                }
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

/// Deliver a response to all workspace users (Manager role).
async fn deliver_manager_response(response: &str, users: &[UserRecord], job: &AgentJob) {
    if users.is_empty() {
        warn!(
            workspace = %job.workspace_name,
            "Message router [manager]: no users with workspace — response delivered to nobody",
        );
    }

    let agent_role = Some("manager".to_string());
    let workspace = &job.workspace_name;

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
            )
            .await;
        }
    }

    let channels = crate::channel_registry().list();
    if channels.is_empty() {
        error!("Message router [manager]: no channels registered");
        return;
    }

    for (channel_name, channel) in &channels {
        for user in users {
            for binding in &user.channels {
                let reply_target = binding.reply_target.as_deref().unwrap_or(&user.name);
                let Some(recipient) = channel.resolve_recipient(&user.name, reply_target) else {
                    continue;
                };
                if let Err(e) = channel
                    .send(&SendMessage {
                        content: response.to_string(),
                        recipient,
                        reply_markup: None,
                    })
                    .await
                {
                    error!(
                        channel = %channel_name,
                        user = %user.name,
                        "Message router [manager]: failed to send response to {}: {e}",
                        user.name,
                    );
                }
            }
        }
    }
}

/// Deliver a response to a single user (Assistant and other per-user roles).
///
/// `user` is the resolved [`UserRecord`] for the job's sender — guaranteed
/// non-empty by the consumer loop before calling this function.
///
/// Delivery is scoped to the originating channel type (`job.channel`), then
/// further scoped to the specific `reply_target` when one is available on the
/// job (e.g. Telegram chat_id).  This makes registered-user delivery
/// consistent with the unregistered-user fallback and matches the pre-router
/// `send_channel_reply` behaviour which sent only to `msg.reply_target`.
///
/// Without `reply_target` (non-user-facing jobs like ticket notifications or
/// AskTool results), all matching channel bindings receive the response.
///
/// Broadcast+persist is always performed exactly once per response, and the
/// response is always sent via the originating channel type so it reaches
/// the user through the expected transport.
async fn deliver_single_user_response(
    response: &str,
    user: &UserRecord,
    job: &AgentJob,
    role: &Role,
) {
    // Broadcast + persist
    let channel = job.channel.as_str();
    broadcast_and_persist_agent_response(
        &user.name,
        channel,
        response,
        Some(role.as_str().to_string()),
        &job.workspace_name,
    )
    .await;

    // Send only via the originating channel — not all registered channels.
    let Some(chan) = crate::channel_registry().get(channel) else {
        return;
    };

    for binding in &user.channels {
        // Only send on the channel that matches the job's origin
        if binding.channel != channel {
            continue;
        }

        // When the original message had a specific reply target (e.g. Telegram
        // chat_id), scope delivery to only the binding whose reply_target
        // matches — this makes registered-user delivery consistent with the
        // unregistered-user fallback and matches the pre-router
        // `send_channel_reply` behaviour.
        if let Some(ref target) = job.reply_target
            && binding.reply_target.as_deref() != Some(target.as_str())
        {
            continue;
        }

        let target_addr = binding.reply_target.as_deref().unwrap_or(&user.name);
        let Some(recipient) = chan.resolve_recipient(&user.name, target_addr) else {
            continue;
        };
        if let Err(e) = chan
            .send(&SendMessage {
                content: response.to_string(),
                recipient,
                reply_markup: None,
            })
            .await
        {
            error!(
                channel = %channel,
                user = %user.name,
                "Message router [{role}]: failed to send response to {}: {e}",
                user.name,
            );
        }
    }
}

/// Deliver a response to an unregistered user (no [`UserRecord`] in the users DB).
///
/// Falls back to the originating channel using `job.reply_target` (when
/// available) or `job.user_name` as the reply target. This matches the
/// pre-router behaviour of [`send_channel_reply`](crate::channels::send_channel_reply),
/// which worked for any user regardless of registration status and preserved
/// the original message's reply target (e.g. Telegram chat_id).
///
/// The response is always broadcast + persisted first (so it appears in the
/// GUI chat history even if the channel transport delivery fails).
async fn deliver_unregistered_user_response(response: &str, job: &AgentJob, role: &Role) {
    let ch = job.channel.as_str();

    // Broadcast + persist (works with just strings, no UserRecord needed).
    broadcast_and_persist_agent_response(
        &job.user_name,
        ch,
        response,
        Some(role.as_str().to_string()),
        &job.workspace_name,
    )
    .await;

    // Try to send via the originating channel.
    let Some(chan) = crate::channel_registry().get(ch) else {
        return;
    };

    // Use reply_target from the original message when available (e.g. Telegram
    // chat_id), falling back to user_name.  This matches the pre-router
    // behaviour of `send_channel_reply` which preserved the original
    // `msg.reply_target`.
    let reply_target = job.reply_target.as_deref().unwrap_or(&job.user_name);
    let Some(recipient) = chan.resolve_recipient(&job.user_name, reply_target) else {
        warn!(
            workspace = %job.workspace_name,
            user = %job.user_name,
            channel = %ch,
            "Message router [{role}]: cannot resolve recipient for unregistered user — \
             response was persisted but not delivered via transport",
        );
        return;
    };

    if let Err(e) = chan
        .send(&SendMessage {
            content: response.to_string(),
            recipient,
            reply_markup: None,
        })
        .await
    {
        error!(
            channel = %ch,
            user = %job.user_name,
            "Message router [{role}]: failed to send response to unregistered user: {e}",
        );
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
    use crate::channels::gui::GuiChannel;
    use std::sync::Arc;

    // ── Consumer loop lifecycle tests ─────────────────────────────────

    /// Shortcut to construct a minimal [`AgentJob`] for lifecycle tests.
    fn make_job(role: Role, workspace: &str, user: &str, channel: &str) -> AgentJob {
        AgentJob {
            content: String::new(),
            workspace_name: workspace.to_string(),
            user_name: user.to_string(),
            channel: channel.to_string(),
            kind: JobKind::UserMessage,
            role,
            reply_target: None,
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
    async fn test_resolve_workspace_found() {
        crate::util::test::init_management_test_stores().await;

        let ws =
            crate::util::test::create_test_workspace("/tmp/test_resolve_ws", "test_resolve_ws")
                .await;

        let result = resolve_workspace("test_resolve_ws").await;
        let resolved = result.expect("resolve should succeed for DB workspace");
        assert!(resolved.is_some(), "DB workspace should be found");
        assert_eq!(resolved.unwrap().name, "test_resolve_ws");
    }

    /// Resolving a personal workspace constructs it on the fly when
    /// it is NOT in the DB.
    #[tokio::test]
    async fn test_resolve_workspace_personal() {
        crate::util::test::init_management_test_stores().await;

        let result = resolve_workspace("personal:liliana").await;
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
    async fn test_resolve_workspace_not_found() {
        crate::util::test::init_management_test_stores().await;

        let result = resolve_workspace("nonexistent_workspace").await;
        let resolved = result.expect("resolve should succeed (no error) for missing workspace");
        assert!(
            resolved.is_none(),
            "nonexistent workspace should not be found",
        );
    }

    // ── Helpers ──────────────────────────────────────────────────────────

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
            let guard = ROUTER.get().unwrap().read().unwrap_poison();
            if !guard.contains_key(id) {
                cleaned_up = true;
                break;
            }
            drop(guard);
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
    async fn test_resolve_single_user_found() {
        setup_response_test_infra().await;

        // The admin user is auto-created by ensure_admin_user.
        let user = resolve_single_user("admin").await;
        assert!(user.is_some(), "admin user should exist after store init");
        assert_eq!(user.as_ref().unwrap().name, "admin");
    }

    /// `resolve_single_user` returns `None` for a non-existent user.
    #[tokio::test]
    async fn test_resolve_single_user_not_found() {
        setup_response_test_infra().await;

        let user = resolve_single_user("nonexistent_user").await;
        assert!(user.is_none(), "non-existent user should return None");
    }

    /// `deliver_unregistered_user_response` completes without error when
    /// the channel is registered, a user is NOT in the DB, and the fallback
    /// path to `reply_target` is exercised.
    #[tokio::test]
    async fn test_deliver_unregistered_user_response() {
        setup_response_test_infra().await;

        let job = AgentJob {
            content: "hello from unregistered user".to_string(),
            workspace_name: "default".to_string(),
            user_name: "unregistered_alice".to_string(),
            channel: "gui".to_string(),
            kind: JobKind::UserMessage,
            role: Role::Assistant,
            reply_target: Some("chat_123".to_string()),
        };

        // Should complete without panic.
        deliver_unregistered_user_response("response to unregistered user", &job, &Role::Assistant)
            .await;
    }

    /// `deliver_single_user_response` completes without error when the
    /// user has a channel binding matching the job's origin channel.
    #[tokio::test]
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
            kind: JobKind::UserMessage,
            role: Role::Assistant,
            reply_target: None,
        };

        // Should complete without panic — sends response via "gui" channel.
        deliver_single_user_response("response to registered user", &user, &job, &Role::Assistant)
            .await;
    }

    /// `deliver_single_user_response` handles the case where the user has
    /// NO channel binding matching the job's origin — only broadcast+persist
    /// runs, transport delivery is skipped.
    #[tokio::test]
    async fn test_deliver_single_user_no_matching_binding() {
        setup_response_test_infra().await;

        // Admin user exists but has no "gui" channel binding.
        let user = resolve_single_user("admin").await.unwrap();

        let job = AgentJob {
            content: "hello".to_string(),
            workspace_name: "default".to_string(),
            user_name: "admin".to_string(),
            channel: "gui".to_string(),
            kind: JobKind::UserMessage,
            role: Role::Assistant,
            reply_target: None,
        };

        // Should complete without panic — broadcast+persist runs, transport
        // delivery is skipped because there's no matching "gui" binding.
        deliver_single_user_response(
            "response to registered user without matching binding",
            &user,
            &job,
            &Role::Assistant,
        )
        .await;
    }

    /// `deliver_manager_response` broadcasts to all workspace users without
    /// panic when users have channel bindings.
    #[tokio::test]
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
            kind: JobKind::TicketNotify,
            role: Role::Manager,
            reply_target: None,
        };

        deliver_manager_response("manager response", &[user], &job).await;
    }
}
