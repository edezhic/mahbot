//! Manager queue — serializes all Manager agent requests through a single
//! consumer task. Eliminates concurrent Manager agent races by processing
//! one job at a time.
//!
//! Three producer paths feed into the queue:
//! - **User messages** (main.rs): chat messages from users → enqueue
//! - **Ticket notifications** (management.rs): ticket transitions → enqueue
//! - **AskTool results** (tools/ask.rs): async sub-agent results → enqueue
//!
//! The consumer resolves the workspace, runs the Manager agent, and delivers
//! the response via the global channel. No inline-button options are extracted
//! from the session — the button infrastructure and callback handling remain
//! available for future flows that produce their own reply markup.

use crate::channels::{
    broadcast_and_persist_agent_response, spawn_scoped_typing_task, stop_typing,
};
use crate::session::manager_session_key;
use crate::users::UserRecord;
use crate::{ChatEvent, Role, SendMessage};
use std::collections::HashSet;
use tokio::sync::OnceCell;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

// ── Job definition ─────────────────────────────────────────────────

/// Semantic category of a Manager queue job.
#[derive(Debug)]
pub enum JobKind {
    /// User-typed message (chat or inline-button callback).
    /// The ticket transition buffer drains before the agent runs so the
    /// Manager sees accumulated non-critical status changes.
    /// No inline-button options are extracted — user messages use
    /// the last stored reply markup.
    UserMessage,
    /// System notification from a ticket transition.
    /// The buffer drains inside `notify_ticket()` before enqueue,
    /// so the consumer skips buffer draining for this path.
    /// No inline-button options are extracted from the session.
    TicketNotify,
    /// Result from an async AskTool sub-agent, injected into the Manager
    /// session for the original asker to receive.
    /// No buffer drain (already done for the user message that
    /// triggered the ask) and no option extraction (sub-agent results
    /// don't produce UI buttons).
    AskToolResult,
}

/// A single unit of work for the Manager agent consumer.
pub struct ManagerJob {
    /// The message content to process. User messages and ticket-notification
    /// paths enrich (multimodal + link summarization) before enqueuing.
    /// Option callbacks produce synthetic text with no media or URLs and
    /// skip enrichment — functionally a no-op.
    pub content: String,
    /// Workspace name — resolved to a `Workspace` inside the consumer.
    pub workspace_name: String,
    /// The semantic job kind — determines whether the consumer drains the
    /// ticket buffer before running the agent.
    pub kind: JobKind,
}

// ── Queue ─────────────────────────────────────────────────────────

/// Global Manager queue — explicitly initialized during startup.
pub static MANAGER_QUEUE: OnceCell<ManagerQueue> = OnceCell::const_new();

/// Initialize the global Manager queue and spawn the consumer loop.
/// Must be called after the Tokio runtime is active (i.e., during startup in main()).
pub fn init_global() -> anyhow::Result<()> {
    let (tx, rx) = mpsc::unbounded_channel::<ManagerJob>();
    tokio::spawn(consumer_loop(rx));
    MANAGER_QUEUE
        .set(ManagerQueue { tx })
        .map_err(|_| anyhow::anyhow!("MANAGER_QUEUE already initialized"))?;
    Ok(())
}

/// Get a reference to the global Manager queue.
/// Panics if not initialized — call `init_global()` during startup.
pub fn manager_queue() -> &'static ManagerQueue {
    MANAGER_QUEUE
        .get()
        .expect("MANAGER_QUEUE not initialized — call manager_queue::init_global() in main()")
}

/// A serialized queue of Manager agent jobs.
pub struct ManagerQueue {
    tx: mpsc::UnboundedSender<ManagerJob>,
}

impl ManagerQueue {
    /// Enqueue a job for the Manager agent consumer.
    /// Never blocks — unbounded channel.
    pub fn enqueue(&self, job: ManagerJob) {
        if let Err(e) = self.tx.send(job) {
            error!("Failed to enqueue Manager job: {e}");
        }
    }
}

// ── Consumer loop ─────────────────────────────────────────────────

/// Set up Telegram typing indicators for all workspace users reachable
/// on the Telegram channel. Returns a list of (CancellationToken, JoinHandle)
/// pairs — one per unique Telegram chat (deduplicated by reply_target).
///
/// Users without a Telegram channel binding (never received an incoming
/// message) are silently skipped. If Telegram is not configured
/// (channel not in registry), returns an empty Vec.
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
        // Check if user has a Telegram channel binding
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

        // Immediate typing call — fire before the first periodic tick.
        if let Err(e) = tg_channel.start_typing(&recipient).await {
            debug!("Manager queue: telegram start_typing failed: {e}");
        }

        // Periodic 4-second refresh for the duration of agent execution.
        let cancel = CancellationToken::new();
        let handle = spawn_scoped_typing_task(recipient, "telegram".to_string(), cancel.clone());
        typing_tasks.push((cancel, handle));
    }

    typing_tasks
}

/// The single consumer task that processes Manager jobs one at a time.
///
/// Shutdown-aware loop: checks for global shutdown between jobs, races
/// [`mpsc::UnboundedReceiver::recv`] against the shutdown token while
/// idle, and never interrupts an in-flight job — the current job (agent
/// run + post-delivery) completes before the next shutdown check at the
/// top of the loop.
#[allow(clippy::too_many_lines)]
async fn consumer_loop(mut rx: mpsc::UnboundedReceiver<ManagerJob>) {
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        // Between jobs: check if global shutdown has been signalled.
        // Catches the case where shutdown fired during the previous job.
        if shutdown.is_cancelled() {
            info!("Manager queue: shutting down — agent queue drained");
            break;
        }

        // While waiting for the next job, race against shutdown so we
        // don't block indefinitely when the daemon is shutting down.
        let job = tokio::select! {
            job = rx.recv() => {
                match job {
                    Some(job) => job,
                    None => break, // channel closed
                }
            }
            () = shutdown.cancelled() => {
                info!("Manager queue: shutting down (global shutdown)");
                break;
            }
        };

        info!(
            workspace = %job.workspace_name,
            kind = ?job.kind,
            "Manager queue: processing job",
        );

        // Resolve workspace by name
        let ws = match crate::workspace::get_by_name(&job.workspace_name).await {
            Ok(Some(ws)) => ws,
            Ok(None) => {
                error!(
                    workspace = %job.workspace_name,
                    "Manager queue: workspace not found — skipping job",
                );
                continue;
            }
            Err(e) => {
                error!(
                    workspace = %job.workspace_name,
                    error = %e,
                    "Manager queue: failed to look up workspace — skipping job",
                );
                continue;
            }
        };

        let session_key = manager_session_key(&ws.name);

        // Look up all workspace users before running the agent. Cached here for
        // both typing-target discovery (Telegram) and response delivery below.
        // Reusing the list for delivery means users who switch workspaces during
        // agent execution won't receive this response — an accepted trade-off.
        let users: Vec<UserRecord> = match crate::users::USER_STORE.get() {
            Some(store) => store
                .find_by_workspace(&job.workspace_name)
                .await
                .unwrap_or_default(),
            None => Vec::new(),
        };

        // ── Telegram typing setup ───────────────────────────────────────
        let typing_tasks = setup_telegram_typing(&users).await;

        // Send typing start to GUI subscribers for all users in this workspace.
        let _ = crate::CHAT_BROADCAST.get().map(|tx| {
            for user in &users {
                let _ = tx.send(ChatEvent::Typing {
                    user_name: user.name.clone(),
                    is_typing: true,
                });
            }
        });

        // Drain buffered ticket transitions on user-initiated messages so
        // the Manager sees accumulated non-critical status changes.
        // Notification paths drain inside notify_ticket() instead.
        let message = match job.kind {
            JobKind::UserMessage => {
                let drained = crate::ticket_buffer::drain(&job.workspace_name);
                if drained.is_empty() {
                    job.content
                } else {
                    format!("{drained}\n{content}", content = job.content)
                }
            }
            JobKind::TicketNotify | JobKind::AskToolResult => job.content,
        };

        // Run the Manager agent (no reply context needed — routing comes from DB).
        let (_agent, response) =
            crate::agent::run_agent(session_key, Role::Manager, &ws, None, &message).await;

        // Cancel and await all Telegram typing refresh tasks.
        for (cancel, handle) in typing_tasks {
            cancel.cancel();
            stop_typing(handle).await;
        }

        // Typing stop for GUI.
        let _ = crate::CHAT_BROADCAST.get().map(|tx| {
            for user in &users {
                let _ = tx.send(ChatEvent::Typing {
                    user_name: user.name.clone(),
                    is_typing: false,
                });
            }
        });

        let Some(response) = response else {
            // Agent was cancelled or errored — nothing to deliver
            continue;
        };

        // No inline-button options are extracted from agent responses.
        // The callback parsing infrastructure (`extraction::is_callback`,
        // `extraction::decode_callback`, `handle_option_callback`)
        // remains available for future flows that produce their own reply markup.
        let reply_markup: Option<serde_json::Value> = None;

        if users.is_empty() {
            warn!(
                workspace = %job.workspace_name,
                "Manager queue: no users with workspace — response delivered to nobody",
            );
        }

        // Extract common SendMessage fields once; only `recipient` differs
        // between the broadcast+persist loop and the channel delivery loop.
        let content = &response;
        let agent_role = Some("manager".to_string());
        let workspace = &job.workspace_name;

        // ── Broadcast + persist once per user ─────────────────────────
        // Before the channel transport loop: broadcast the agent response
        // to the GUI dashboard and persist to chat_history once per unique
        // user. The channel for chat_history is taken from the first channel
        // binding (or "gui" if none).
        {
            let mut seen_names = HashSet::new();
            for user in &users {
                if !seen_names.insert(&user.name) {
                    continue;
                }
                let channel = user.channels.first().map_or("gui", |b| b.channel.as_str());
                broadcast_and_persist_agent_response(
                    &user.name,
                    channel,
                    content,
                    agent_role.clone(),
                    workspace,
                    reply_markup.clone(),
                )
                .await;
            }
        }

        let channels = crate::channel_registry().list();

        if channels.is_empty() {
            error!("Manager queue: no channels registered");
            continue;
        }

        for (channel_name, channel) in &channels {
            for user in &users {
                // Deliver on all channel bindings for this user.
                for binding in &user.channels {
                    let reply_target = binding.reply_target.as_deref().unwrap_or(&user.name);
                    let Some(recipient) = channel.resolve_recipient(&user.name, reply_target)
                    else {
                        continue;
                    };
                    if let Err(e) = channel
                        .send(&SendMessage {
                            content: content.clone(),
                            recipient,
                            reply_markup: reply_markup.clone(),
                            agent_role: agent_role.clone(),
                            workspace: workspace.clone(),
                        })
                        .await
                    {
                        error!(
                            channel = %channel_name,
                            user = %user.name,
                            "Manager queue: failed to send response to {}: {e}",
                            user.name,
                        );
                    }
                }
            }
        }
    }
}
