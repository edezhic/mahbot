//! Agent queue infrastructure — serializes agent requests through per-role
//! consumer tasks. Eliminates concurrent agent races by processing one job
//! at a time per role.
//!
//! Three producer paths feed into the queues:
//! - **User messages** (main.rs): chat messages from users → enqueue
//! - **Ticket notifications** (management.rs): ticket transitions → enqueue
//! - **AskTool results** (tools/ask.rs): async sub-agent results → enqueue
//!
//! Each role that needs async processing gets its own queue. Currently:
//! - **Manager**: receives user messages, ticket notifications, and ask-tool
//!   results. Broadcasts responses to all workspace users.
//! - **Assistant**: receives user messages and ask-tool results. Delivers
//!   responses to the specific user.

use crate::channels::{
    broadcast_and_persist_agent_response, spawn_scoped_typing_task, stop_typing,
};
use crate::session::manager_session_key;
use crate::users::UserRecord;
use crate::{ChatEvent, Role, SendMessage};
use std::collections::{HashMap, HashSet};
use std::sync::{OnceLock, RwLock};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

// ── Job definition ─────────────────────────────────────────────────

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
    /// Workspace name — resolved to a `Workspace` inside the consumer.
    pub workspace_name: String,
    /// Sender identity — used for per-user response delivery.
    pub user_name: String,
    /// Channel origin (gui, telegram, voice).
    pub channel: String,
    /// The semantic job kind.
    pub kind: JobKind,
}

// ── Queue ─────────────────────────────────────────────────────────

/// A serialized queue of agent jobs for a specific role.
#[derive(Clone)]
pub struct AgentQueue {
    tx: mpsc::UnboundedSender<AgentJob>,
}

impl AgentQueue {
    /// Enqueue a job for the agent consumer.
    /// Never blocks — unbounded channel.
    pub fn enqueue(&self, job: AgentJob) {
        if let Err(e) = self.tx.send(job) {
            error!("Failed to enqueue agent job: {e}");
        }
    }

    /// Convenience method to enqueue a user-message job.
    pub fn enqueue_user_message(
        &self,
        content: String,
        workspace_name: String,
        user_name: String,
        channel: String,
    ) {
        self.enqueue(AgentJob {
            content,
            workspace_name,
            user_name,
            channel,
            kind: JobKind::UserMessage,
        });
    }
}

// ── Registry ─────────────────────────────────────────────────────

/// Global registry of agent queues, keyed by role.
static AGENT_QUEUES: OnceLock<RwLock<HashMap<Role, AgentQueue>>> = OnceLock::new();

/// Initialize agent queues for all roles that need async processing.
/// Spawns consumer loops for each role.
/// Must be called after the Tokio runtime is active (i.e., during startup).
pub fn init_global() -> anyhow::Result<()> {
    let mut map = HashMap::new();
    for role in [Role::Manager, Role::Assistant] {
        let (tx, rx) = mpsc::unbounded_channel::<AgentJob>();
        tokio::spawn(consumer_loop(role, rx));
        map.insert(role, AgentQueue { tx });
    }
    AGENT_QUEUES
        .set(RwLock::new(map))
        .map_err(|_| anyhow::anyhow!("AGENT_QUEUES already initialized"))?;
    Ok(())
}

/// Get the agent queue for a specific role, if it has one.
pub fn queue_for(role: &Role) -> Option<AgentQueue> {
    AGENT_QUEUES.get()?.read().unwrap().get(role).cloned()
}

/// Get the Manager queue (backward-compatible accessor).
/// Panics if not initialized.
#[must_use]
pub fn manager_queue() -> AgentQueue {
    queue_for(&Role::Manager)
        .expect("Manager queue not initialized — call manager_queue::init_global()")
}

// ── Consumer loop ─────────────────────────────────────────────────

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
            debug!("Agent queue: telegram start_typing failed: {e}");
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

/// The consumer task that processes agent jobs for a specific role, one at a time.
///
/// Shutdown-aware loop: checks for global shutdown between jobs.
#[allow(clippy::too_many_lines)]
async fn consumer_loop(role: Role, mut rx: mpsc::UnboundedReceiver<AgentJob>) {
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            info!("Agent queue [{role}]: shutting down — queue drained");
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
                info!("Agent queue [{role}]: shutting down (global shutdown)");
                break;
            }
        };

        info!(
            workspace = %job.workspace_name,
            user = %job.user_name,
            kind = ?job.kind,
            "Agent queue [{role}]: processing job",
        );

        // Resolve workspace by name
        let ws = match crate::workspace::get_by_name(&job.workspace_name).await {
            Ok(Some(ws)) => ws,
            Ok(None) => {
                error!(
                    workspace = %job.workspace_name,
                    "Agent queue [{role}]: workspace not found — skipping job",
                );
                continue;
            }
            Err(e) => {
                error!(
                    workspace = %job.workspace_name,
                    error = %e,
                    "Agent queue [{role}]: failed to look up workspace — skipping job",
                );
                continue;
            }
        };

        // ── Role-specific session key ─────────────────────────────────
        // Manager: workspaces have one session shared by all users.
        // Assistant (and other per-user roles): include user_name so that
        // each user gets an isolated session — no cross-user context leak.
        let session_key = match role {
            Role::Manager => manager_session_key(&ws.name),
            _ => format!("queue_{}_{}_{}", role.as_str(), ws.name, job.user_name),
        };

        // ── Role-specific user resolution ─────────────────────────────
        // Manager: broadcast to all workspace users.
        // Assistant: deliver to the specific user.
        let users: Vec<UserRecord> = if role == Role::Manager {
            match crate::users::USER_STORE.get() {
                Some(store) => store
                    .find_by_workspace(&job.workspace_name)
                    .await
                    .unwrap_or_default(),
                None => Vec::new(),
            }
        } else {
            // For per-user roles, resolve the single user
            let user = resolve_single_user(&job.user_name).await;
            user.into_iter().collect()
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

        // ── Run the agent ────────────────────────────────────────────
        let (_agent, response) = crate::agent::run_agent(
            session_key,
            role,
            &ws,
            None,
            &message,
            job.user_name.clone(),
            job.channel.clone(),
        )
        .await;

        // ── Stop typing ──────────────────────────────────────────────
        for (cancel, handle) in typing_tasks {
            cancel.cancel();
            stop_typing(handle).await;
        }
        broadcast_typing(&users, &job.workspace_name, false);

        let Some(response) = response else {
            continue;
        };

        // ── Response delivery ────────────────────────────────────────
        // Manager: broadcast + persist to all workspace users.
        // Assistant: send reply to the specific user.
        match role {
            Role::Manager => {
                deliver_manager_response(&response, &users, &job).await;
            }
            _ => {
                deliver_single_user_response(&response, &users, &job, &role).await;
            }
        }
    }
}

/// Deliver a response to all workspace users (Manager role).
async fn deliver_manager_response(response: &str, users: &[UserRecord], job: &AgentJob) {
    if users.is_empty() {
        warn!(
            workspace = %job.workspace_name,
            "Agent queue [manager]: no users with workspace — response delivered to nobody",
        );
    }

    let agent_role = Some("manager".to_string());
    let workspace = &job.workspace_name;

    // ── Broadcast + persist once per user ─────────────────────────
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
        error!("Agent queue [manager]: no channels registered");
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
                        "Agent queue [manager]: failed to send response to {}: {e}",
                        user.name,
                    );
                }
            }
        }
    }
}

/// Deliver a response to a single user (Assistant and other per-user roles).
async fn deliver_single_user_response(
    response: &str,
    users: &[UserRecord],
    job: &AgentJob,
    role: &Role,
) {
    let Some(user) = users.first() else {
        warn!(
            workspace = %job.workspace_name,
            user = %job.user_name,
            "Agent queue [{role}]: user not found — response delivered to nobody",
        );
        return;
    };

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

    // Send via registered channels
    let channels = crate::channel_registry().list();
    for (channel_name, channel) in &channels {
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
                    "Agent queue [{role}]: failed to send response to {}: {e}",
                    user.name,
                );
            }
        }
    }
}

/// Resolve a single user by name, returning their full record if available.
/// Uses a targeted database query instead of loading all users.
async fn resolve_single_user(user_name: &str) -> Vec<UserRecord> {
    let Some(store) = crate::users::USER_STORE.get() else {
        return Vec::new();
    };
    match store.find_by_name(user_name).await {
        Ok(Some(user)) => vec![user],
        _ => Vec::new(),
    }
}
