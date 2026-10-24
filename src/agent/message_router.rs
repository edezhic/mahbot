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
//! - [`Role::Manager`] responses relay to the admin's Assistant via
//!   the durable `<manager-message>` envelope ([`route_manager_notify`]) —
//!   there is no direct user delivery. When the Manager answers a custom-role
//!   caller, the reply instead returns into that branch
//!   ([`route_manager_reply_to_custom`]).
//! - [`Role::Custom`] responses relay to the admin's Assistant the same way,
//!   via the durable `<custom-role-message>` envelope
//!   ([`route_custom_role_notify`] / [`route_custom_role_failure`]) — a custom
//!   role delivers nothing to the owner's channels.
//! - [`Role::Worker`] responses on a branch job deliver the durable
//!   `<worker-report>` envelope into the conversation that asked
//!   ([`route_worker_branch_report`] / [`route_worker_branch_failure`]) — the
//!   asking custom role's branch when it carried one, else the admin's
//!   Assistant.
//! - A guest's Assistant is handed a one-way `<owner-assistant-message>` from the
//!   owner's Assistant ([`route_guest_message`]); it runs as that guest like any
//!   other incoming message, so its answer reaches the guest's own channels and
//!   nothing goes back to the sender.
//! - Other roles deliver to the triggering user's channel bindings (with an
//!   unregistered-user fallback).

use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
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
/// - Voice channel (macOS): TTS speaks this as "robot warning retry" (acceptable for now).
/// - Emoji rendering varies across terminals and clients.
const AGENT_FAILURE_EMOJI: &str = "🤖⚠️🔄";

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
    /// agent session. A round its caller cancels produces none.
    AnalyzeToolResult,
    /// Result from an async single-run worker dispatch (`send` `worker`), injected
    /// back into the caller's agent session. Also the kind a worker branch's
    /// report is delivered under (single run or branch).
    ImplementResult,
    /// Result from an async deep analysis run (the `deep` depth of `analyze`),
    /// injected back into the calling agent's session. Exactly one envelope per
    /// run.
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
    /// Durable auto-delivery copy of a Manager's response, routed into the
    /// single admin's Assistant session (`<manager-message>`
    /// envelope; the job's workspace is the admin's personal one, the source
    /// workspace rides in `originating_workspace` and in the envelope's
    /// `workspace` attribute, so its file references stay authorized).
    /// Appended to the session like other result kinds; never
    /// broadcast/persisted to chat_history. Deliberately distinct from
    /// `UserMessage` so the failure-emoji gate (`== UserMessage`) structurally
    /// excludes it. The serde alias keeps pre-unification `ManagerReply`
    /// pending rows deserializable at boot replay.
    #[serde(alias = "ManagerReply")]
    ManagerNotify,
    /// A custom-role branch's finished answer or its failure, routed into the
    /// asking conversation (`<custom-role-message>` envelope). Into the admin's
    /// Assistant session when the branch was the asker (the job's workspace is
    /// the admin's personal one, the folder the launch ran in travels in
    /// `originating_workspace`), or into the asking custom role's branch.
    /// Durable — produced through `stamp_and_route`. Deliberately distinct from
    /// `UserMessage` so the failure-emoji gate (`== UserMessage`) structurally
    /// excludes it. (A worker branch's report is the sibling shape under
    /// [`MessageKind::ImplementResult`], which carries no extra file-delivery
    /// root.)
    CustomRoleNotify,
    /// A one-way handover from the owner's Assistant into a guest's own Assistant
    /// conversation (`<owner-assistant-message>` envelope): the job runs AS that
    /// guest, in their personal workspace, and the answer it produces reaches the
    /// guest's own channels. Nothing returns to the sender — hence no reply
    /// target — and `originating_workspace` stays unset, so the guest's workspace
    /// is the only root its RELATIVE media markers resolve in (an absolute one is
    /// not confined by the media layer, the same as any other message). Durable —
    /// produced through [`hand_over`]. Deliberately distinct from `UserMessage` so
    /// the failure-emoji gate (`== UserMessage`) structurally excludes it: the
    /// guest never sees an unexplained failure mark from a message they did not
    /// send.
    GuestMessage,
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
    /// Channel origin (gui, telegram, voice, cli).
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
    /// For relayed jobs ([`MessageKind::ManagerNotify`]) this names the
    /// workspace the relayed content originated in, so a project-workspace
    /// media marker in that content (`[FILE:...]`, `[IMAGE:...]`, `[AUDIO:...]`,
    /// `[VIDEO:...]`) resolves and delivers from where it was produced, not just
    /// because the relaying Assistant sits in the admin's personal workspace.
    #[serde(default)]
    pub originating_workspace: Option<String>,
    /// The branch launch this job runs, when it runs one ([`Role::Custom`] or
    /// [`Role::Worker`]): the definition taken at launch plus the resolved branch
    /// address. `#[serde(default)]` keeps already-persisted `pending_jobs`
    /// envelopes (which predate the field) decodable at boot replay.
    #[serde(default)]
    pub custom_role: Option<crate::Launch>,
    /// The conversation a forwarded reply must return to, when the job's caller
    /// was a custom role: the Manager's answer goes back into that branch
    /// instead of the admin's Assistant (see [`route_manager_reply_to_custom`]).
    /// A worker branch carries its asker on the launch instead
    /// ([`crate::Launch::reply_to`]) — one carrier per fact. `None` for every
    /// other job.
    #[serde(default)]
    pub reply_to_custom: Option<Asker>,
}

/// The conversation that asked for a forwarded reply — the asking branch's
/// launch and the folder its turn ran in, so an answer returns there instead
/// of the admin's Assistant. Carried durably by the launch
/// (`crate::Launch::reply_to`) and on the routed job
/// (`AgentJob::reply_to_custom`) that must deliver into it; a Manager's
/// answer, a worker branch's report and its failure notice all travel this way.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Asker {
    /// The asking branch's launch — its authorial definition and its resolved
    /// branch address, so the reply re-enters that same conversation.
    pub launch: crate::Launch,
    /// The folder the asking turn runs in — the branch's own folder, so the
    /// answer re-enters the same conversation in the same place.
    pub workspace: String,
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
/// Locks the router for reading only, looks up the agent ID, and — when a
/// consumer exists — sends the job through the sender borrowed from the read
/// guard, holding the guard until the block ends. If the sender exists, we
/// are done.
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

/// The `MessageKind::UserMessage` envelope every producer of an owner-side
/// incoming message builds: a routed incoming message ([`route_user_message`]),
/// the command line's handover ([`deliver_command_line_message`]), and a notice
/// the product addresses to the owner's assistant
/// ([`assistant_notice_job`], shared by both of its delivery paths). The routing
/// fields such a message never carries are the shape's own defaults.
fn user_message_job(
    content: String,
    workspace_name: String,
    user_name: String,
    channel: String,
    role: Role,
    reply_target: Option<String>,
) -> AgentJob {
    AgentJob {
        content,
        workspace_name,
        user_name,
        channel,
        kind: MessageKind::UserMessage,
        role,
        reply_target,
        pending_job_id: None,
        originating_workspace: None,
        custom_role: None,
        reply_to_custom: None,
    }
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
    // admin account is always seeded. Normalize defensively so no path can create
    // a malformed identity (bare "_ws_role" key) or persist/reply under an
    // empty user. (direct_agent_id also normalizes the ID key; this layer
    // normalizes the delivery payload.)
    let user_name =
        crate::session::normalize_user_name(&user_name, "route_user_message").to_string();

    let agent_id = crate::session::resolve_agent_id(&user_name, role.as_str(), &workspace_name);
    let job = user_message_job(
        content,
        workspace_name,
        user_name,
        channel,
        role,
        reply_target,
    );

    // DURABLE kinds = UserMessage (manager-bound), AgentMessage, ManagerNotify,
    // CustomRoleNotify, GuestMessage, AnalyzeToolResult, ResearchResult,
    // ImplementResult.
    if job.role == Role::Manager {
        stamp_and_route(job, &agent_id, "manager message").await;
        return;
    }
    route(&agent_id, job);
}

/// Persist `job` to `pending_jobs` and stamp the row id onto it — the
/// at-least-once boundary every durable producer shares. An `Err` is a failed
/// write, not a dropped message: the caller still routes the job and logs the
/// cause against its own context.
async fn persist_and_stamp(job: &mut AgentJob) -> anyhow::Result<()> {
    let id = crate::generate_id();
    persist_pending(job, id.clone()).await?;
    job.pending_job_id = Some(id);
    Ok(())
}

/// Who answers for a durable handover: the one thing that decides what a failed
/// `pending_jobs` write means. The choice belongs to the PRODUCER, never to the
/// message — the same envelope can be a handover whose caller is answered and
/// then gone (the command line) or one whose producer's own turn is durable
/// already (an agent's tool call).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AnsweredFor {
    /// A handover whose failure nobody is told about: an agent, whose own turn is
    /// durable already, or a background sweep with no answerable caller at all
    /// (the alarm sweep). The failed write is logged and the envelope is routed
    /// best-effort instead (at-most-once): dropping the handover would be worse
    /// than delivering it unrecorded.
    Agent,
    /// The command line. The failure is returned, with NOTHING routed: its
    /// caller is answered and gone the moment it returns, so it must never be
    /// told a handover succeeded that a restart could lose.
    CommandLine,
}

/// Hand one durable envelope over, under `answered_for`'s contract — the ONE
/// handover body: persist the envelope to `pending_jobs`, then route it unless
/// a drain began after the write (the row is left to boot replay, which
/// reclaims it; the drain must not be raced with a live route). `Err` is the
/// failed write, and only a [`CommandLine`](AnsweredFor::CommandLine) handover
/// ever returns it — every other producer's failure is logged here and the
/// envelope is routed best-effort (at-most-once) instead.
///
/// A row persisted during the drain/shutdown is never routed: boot replay
/// reclaims it. A best-effort route — the failed write, or one that slipped in
/// as the drain began — may land in a consumer that has already stopped
/// pulling, so the message can still be lost at exit; that window is the price
/// of never dropping a handover whose write failed on the normal path.
pub(crate) async fn hand_over(
    mut job: AgentJob,
    target: &str,
    producer: &str,
    answered_for: AnsweredFor,
) -> anyhow::Result<()> {
    let persisted = match persist_and_stamp(&mut job).await {
        Ok(()) => true,
        Err(e) => {
            if answered_for == AnsweredFor::CommandLine {
                warn!(
                    error = %e,
                    producer,
                    "the handover could not be made durable — refused",
                );
                return Err(e.context("the handover could not be made durable"));
            }
            warn!(
                error = %e,
                producer,
                "Failed to persist the handover — routing best-effort (at-most-once)",
            );
            false
        }
    };
    if crate::shutdown::aborting() && persisted {
        return Ok(());
    }
    route(target, job);
    Ok(())
}

/// [`hand_over`] under [`AnsweredFor::Agent`]: the durable producer whose own
/// turn is already durable. A failed write is logged and the envelope is
/// routed best-effort rather than dropped — see [`hand_over`].
pub(crate) async fn stamp_and_route(job: AgentJob, target: &str, producer: &str) {
    let _ = hand_over(job, target, producer, AnsweredFor::Agent).await;
}

/// Build the `<assistant-message>` envelope job addressed to the workspace
/// Manager (`MessageKind::AgentMessage`, durable).
fn agent_message_job(
    content: String,
    workspace_name: String,
    user_name: &str,
    asker: Option<Asker>,
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
        originating_workspace: None,
        custom_role: None,
        reply_to_custom: asker,
    }
}

/// Route a message from an assistant agent into the workspace Manager's
/// session (`MessageKind::AgentMessage`). For a non-custom caller there is no
/// addressed reply leg — the Manager's response auto-delivers to the admin's
/// Assistant via [`route_manager_notify`]. A Manager-bound request from a custom
/// role DOES have one: `asker` carries the asking branch, and the answer
/// re-enters it via [`route_manager_reply_to_custom`].
///
/// `answered_for` carries the handover's contract ([`hand_over`]): what a failed
/// `pending_jobs` write means depends on who answers for the call.
pub(crate) async fn route_agent_message_to_manager(
    content: String,
    workspace_name: String,
    user_name: String,
    asker: Option<Asker>,
    answered_for: AnsweredFor,
) -> anyhow::Result<()> {
    let job = agent_message_job(content, workspace_name, &user_name, asker);
    let target = crate::jobs::envelope_target(&job);
    hand_over(job, &target, "agent message", answered_for).await
}

/// Hand one message from the owner's Assistant to a guest account's own Assistant
/// (`MessageKind::GuestMessage`, durable) — one-way: nothing is addressed back to
/// the sender.
///
/// The producer (`tools::send`) has already wrapped the text in
/// `<owner-assistant-message>`. The job runs AS the guest — their account is the
/// identity and their personal workspace the folder, so the consumer's pinning and
/// delivery give it the guest's own toolset, conversation and channels, never the
/// owner's. `answered_for` carries the handover's contract ([`hand_over`]).
pub(crate) async fn route_guest_message(
    content: String,
    guest_name: &str,
    answered_for: AnsweredFor,
) -> anyhow::Result<()> {
    let job = AgentJob {
        content,
        workspace_name: crate::users::personal_workspace_name(guest_name),
        user_name: guest_name.to_string(),
        // History-attribution tag only: the answer is broadcast to all of the
        // guest's channel bindings, so no single transport is claimed here.
        channel: "gui".to_string(),
        kind: MessageKind::GuestMessage,
        role: Role::Assistant,
        reply_target: None,
        pending_job_id: None,
        // No root of the sender's rides along: relative markers in the guest's
        // answer resolve in the guest's own workspace.
        originating_workspace: None,
        custom_role: None,
        reply_to_custom: None,
    };
    let target = crate::jobs::envelope_target(&job);
    hand_over(job, &target, "guest message", answered_for).await
}

/// Hand one message from the COMMAND LINE to the owner's own assistant.
///
/// The message is an ordinary incoming message from the owner and nothing in
/// its text marks it as scripted: it is broadcast to the GUI, persisted to
/// `chat_history`, and mirrored to his bound Telegram as one of his own —
/// exactly as a message typed into the dashboard is
/// ([`crate::channels::broadcast_and_persist_incoming_message`]) — and it reaches
/// the assistant's conversation through the same durable envelope every other
/// handover uses, so the assistant answers the way it normally does. It is
/// attributed to [`crate::cli::CHANNEL`] for the product's own record — a message
/// a shell sent can be told apart from one the owner typed (`gui`) or spoke
/// (`voice`), and the product's own logs record the handover as a command-line
/// operation — never in the message itself, which carries no trace of its
/// origin.
///
/// The durable handover comes FIRST and the history copy second — deliberately
/// the opposite of the other incoming-message producers, which broadcast first
/// so the transcript shows the message while the agent is still working. Here
/// the caller is answered only once the handover is durable, so the conversation
/// must not show a message the caller was told did not get through.
///
/// `Err` is a call carrying no owner, or a failed `pending_jobs` write — either
/// way the handover is REFUSED and nothing is routed: see
/// [`AnsweredFor::CommandLine`].
pub(crate) async fn deliver_command_line_message(content: String) -> anyhow::Result<()> {
    // The owner the CALL carries, never a second copy of the constant: the command
    // line scopes the identity it acts with
    // ([`crate::cli::instance`]), and every other handover this tool makes is
    // attributed the same way. Checked rather than defaulted, as the Manager arm
    // does: an empty identity would attribute the message to nobody and address
    // it to a `personal:` workspace that does not exist.
    let user_name = crate::agent::tool_user_name();
    anyhow::ensure!(
        !user_name.is_empty(),
        "No user identity available — a message the command line sends must name the owner it \
         comes from."
    );
    let workspace = crate::users::personal_workspace_name(&user_name);

    let job = user_message_job(
        content.clone(),
        workspace.clone(),
        user_name.clone(),
        crate::cli::CHANNEL.to_string(),
        Role::Assistant,
        None,
    );
    let target = crate::jobs::envelope_target(&job);
    hand_over(
        job,
        &target,
        "command-line message",
        AnsweredFor::CommandLine,
    )
    .await?;

    crate::channels::broadcast_and_persist_incoming_message(
        &crate::ChannelMessage {
            user_name,
            reply_target: String::new(),
            content: content.clone(),
            channel: crate::cli::CHANNEL.to_string(),
            workspace,
            ..Default::default()
        },
        &content,
        &content,
    )
    .await;

    Ok(())
}

/// Persist an envelope to `pending_jobs`. The target agent id is derived
/// from the envelope by [`crate::jobs::pending_job_params`].
/// Used by the durable producers: manager-bound messages routed here,
/// analyze/research results via [`crate::jobs::complete_job_with_envelope`],
/// and assistant notices ([`deliver_assistant_notice`]) from the alarm sweep
/// (`alarms::fire_alarm`) and the custom-tool grant path.
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

/// Deliver a durable notice into an Assistant session: build the
/// [`MessageKind::UserMessage`] envelope, persist it to `pending_jobs` BEFORE
/// routing (a crash after persisting replays it at boot), then route it to
/// `agent_id`.
///
/// The reminder-parity delivery path — a fired alarm and a custom-tool grant
/// change both arrive through it. `agent_id` is the resolved Assistant session
/// key; alarm delivery passes the id stored on the alarm rather than
/// re-deriving it (re-derivation would double-escape colliding user names). The
/// text is delivered as composed — notice text is not credential-scrubbed.
///
/// An `Err` is a failed persistence: the notice was still routed, best-effort
/// (at-most-once), and the caller logs the cause against its own context.
pub(crate) async fn deliver_assistant_notice(
    agent_id: &str,
    user_name: &str,
    content: String,
) -> anyhow::Result<()> {
    let mut job = assistant_notice_job(content, user_name);
    let durable = persist_and_stamp(&mut job).await;
    route(agent_id, job);
    durable
}

/// The envelope a notice addressed to an Assistant carries: the owner's personal
/// workspace, the Assistant role, and the history tag an owner-side message
/// carries. Both delivery paths build it here so the fields that must match an
/// incoming message are spelled once.
fn assistant_notice_job(content: String, user_name: &str) -> AgentJob {
    user_message_job(
        content,
        // The user's own personal workspace — the Assistant is pinned there.
        crate::users::personal_workspace_name(user_name),
        user_name.to_string(),
        // History-attribution tag only — the reply is broadcast to all of the
        // user's channel bindings, so no single transport is claimed here.
        "gui".to_string(),
        Role::Assistant,
        None,
    )
}

/// Post a notice addressed to an Assistant straight into its ordinary queue,
/// with no `pending_jobs` row.
///
/// This is the at-most-once sibling of [`deliver_assistant_notice`]: a durable
/// row is replayed at boot, so a producer that owns its own bookkeeping must
/// never also persist the envelope. It records nothing itself, so a caller that
/// wants a repeat rather than a loss settles its own record only after this
/// returns — the update notice does exactly that.
pub(crate) fn post_assistant_notice(agent_id: &str, user_name: &str, content: String) {
    route(agent_id, assistant_notice_job(content, user_name));
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

        // A launch and the role are one fact: derive the role from the launch
        // FIRST, so a job built without the derivation cannot smuggle a launch
        // past the pinning rule below or the custom-role guard that follows.
        job.role = job.role.for_launch(job.custom_role.as_ref());

        // Execution-time invariant: pinned roles (Assistant)
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

        // A custom-role job is meaningless without its launch — the record's
        // prompt, tool subset and branch address all travel in it — and a
        // branch whose record has since been deleted must never be resumed
        // with a definition that no longer exists (a round the role dispatched
        // may still complete after the delete). Both are refused here, in the
        // one funnel every producer goes through, and the durable pending row
        // is reclaimed so boot replay does not resurrect them. The cost is one
        // primary-key lookup per custom-role turn.
        if job.role == Role::Custom {
            let refusal = match job.custom_role.as_ref() {
                None => Some("the job carries no custom-role launch"),
                Some(launch) => match crate::custom_role::get(&launch.name).await {
                    Ok(Some(_)) => None,
                    Ok(None) => Some("the custom role no longer exists"),
                    Err(e) => {
                        // A transient read failure is not a verdict about the
                        // role: run the turn and let the error surface wherever
                        // the turn itself needs the store.
                        warn!(
                            agent_id = %agent_id,
                            role = %launch.name,
                            error = %e,
                            "Message router: custom role lookup failed — running the turn anyway",
                        );
                        None
                    }
                },
            };
            if let Some(refusal) = refusal {
                warn!(
                    agent_id = %agent_id,
                    reason = %refusal,
                    "Message router: refusing a custom-role job",
                );
                confirm_pending_delivery(&job).await;
                continue;
            }
        }

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
                report_unrunnable_branch(&job).await;
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

        // ── Roots anchoring outbound media markers ─────────────────────
        // The job's own workspace, plus the workspace the relayed content
        // originated in when it differs (a Manager→Assistant relay names it in
        // `originating_workspace`): a relative `[IMAGE:]`/`[AUDIO:]`/`[VIDEO:]`
        // target resolves against them, and `[FILE:]` is confined to them. A
        // failed lookup only narrows the roots.
        let mut file_roots = vec![std::path::PathBuf::from(&ws.path)];
        if let Some(origin) = job.originating_workspace.as_deref()
            && origin != job.workspace_name
        {
            match crate::users::resolve_workspace(origin).await {
                Ok(Some(origin_ws)) => file_roots.push(std::path::PathBuf::from(&origin_ws.path)),
                Ok(None) => debug!(
                    workspace = %origin,
                    "Message router: originating workspace not found — marker roots narrowed",
                ),
                Err(e) => debug!(
                    workspace = %origin,
                    error = %e,
                    "Message router: originating workspace lookup failed — marker roots narrowed",
                ),
            }
        }

        // ── Role is embedded directly in the job ────────────────────────
        // No agent-ID string parsing needed — every caller knows the role.
        let role = job.role;

        // ── Resolve users for response delivery ───────────────────────
        // Non-Manager roles deliver to the specific triggering user. The
        // Manager has no direct user delivery: its responses relay to the
        // admin's Assistant via the ManagerNotify envelope below. A custom role
        // delivers nothing to the owner's channels either — its answer goes to
        // the admin's Assistant via the CustomRoleNotify envelope. A worker branch
        // likewise: its report goes into the conversation that asked.
        let users: Vec<UserRecord> = if matches!(role, Role::Manager | Role::Custom | Role::Worker)
        {
            Vec::new()
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
        // The admin gets the Assistant's widened toolset and role prompt;
        // everyone else runs the guest Assistant.
        // Manager and custom-role jobs resolve no users, so `is_admin` is false
        // for both — and nothing reads it for either: a custom role's strength
        // comes from its own tool subset (`Role::custom_role_tools`), not from
        // this flag.
        let is_admin = users
            .first()
            .is_some_and(|user| crate::users::is_admin_name(&user.name));
        // The Running Agents grouping for this job: a branch launch is one
        // conversation keyed by its own agent id, with the display label naming
        // the role and (when named) its branch. Everything else stays
        // ungrouped, exactly as before.
        let (parent_key, parent_label) = match &job.custom_role {
            Some(launch) => (
                Some(crate::agent::registry::ParentKey::Launch(
                    launch.agent_id.clone(),
                )),
                Some(crate::launch::display_label(launch)),
            ),
            None => (None, None),
        };
        let (mut agent, response) = crate::agent::run_agent(
            agent_id.clone(),
            role,
            &ws,
            None,
            &message,
            job.user_name.clone(),
            job.channel.clone(),
            is_admin,
            None,
            false,
            None,
            parent_key,
            parent_label,
            job.custom_role.clone(),
        )
        .await;

        // ── Stop typing ───────────────────────────────────────────────
        for (cancel, handle) in typing_tasks {
            cancel.cancel();
            stop_typing(handle).await;
        }
        broadcast_typing(&users, &job.workspace_name, false);

        let Some(response) = response else {
            // A custom role's failure produces no answer: deliver the durable
            // failure notice to the admin's Assistant — never the emoji, never
            // anything to the owner. A worker branch's failure delivers the same
            // kind of notice into the asking conversation. Guarded exactly like
            // the emoji branch: a deliberately stopped or drained launch stays
            // silent. These arms are mutually exclusive with the emoji: neither a
            // custom role nor a worker branch may EVER emit the generic failure
            // emoji into the owner's channel.
            if role == Role::Custom {
                if !agent.is_cancelled() && !crate::shutdown::aborting() {
                    let name = job
                        .custom_role
                        .as_ref()
                        .map(|c| c.name.as_str())
                        .unwrap_or_default();
                    let identifier = job
                        .custom_role
                        .as_ref()
                        .and_then(|c| c.identifier.as_deref());
                    route_custom_role_failure(name, &job.workspace_name, identifier).await;
                }
            } else if let Some(launch) = worker_branch_launch(&job) {
                if !agent.is_cancelled() && !crate::shutdown::aborting() {
                    route_worker_branch_failure(launch, &ws.path).await;
                }
            }
            // Send emoji error only for UserMessage jobs where the agent truly
            // failed (not cancelled or shutdown). Internal job kinds
            // (TicketNotify, AnalyzeToolResult, ResearchResult, ManagerNotify,
            // CustomRoleNotify, GuestMessage) get no feedback.
            //
            // We check both the agent-specific token AND the
            // global shutdown token because during SIGTERM/SIGINT the global
            // token fires first — work() catches it and returns None, but the
            // agent-specific token may not have been cancelled yet.
            //
            // Drained agents must NOT emit the failure emoji: their round was
            // cut short by the graceful-drain window (or shutdown), not a
            // failure.
            else if job.kind == MessageKind::UserMessage
                && !agent.is_cancelled()
                && !crate::shutdown::aborting()
            {
                deliver_unregistered_user_response(AGENT_FAILURE_EMOJI, &job, &role, &file_roots)
                    .await;
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
        // Manager: the sole delivery path is the durable <manager-message>
        // envelope into the admin's personal Assistant — unless the Manager was
        // answering a custom-role caller, in which case the answer delivers
        // back into that branch.
        // A custom role: the durable <custom-role-message> envelope into the
        // admin's personal Assistant (the folder the launch ran in travels in
        // the envelope's `workspace` attribute).
        // A worker branch: the durable <worker-report> envelope into the asking
        // conversation — the asking custom role's branch, or the admin's
        // Assistant when the assistant asked.
        // Other roles: broadcast to all of the triggering user's channel
        // bindings (or use fallback for unregistered users).
        match role {
            Role::Manager => match job.reply_to_custom.clone() {
                Some(reply) => {
                    route_manager_reply_to_custom(&response, &job.workspace_name, &reply).await;
                }
                None => route_manager_notify(&response, &job.workspace_name).await,
            },
            Role::Custom => {
                let launch = job.custom_role.as_ref();
                let name = launch.map_or("", |c| c.name.as_str());
                let identifier = launch.and_then(|c| c.identifier.as_deref());
                route_custom_role_notify(&response, name, &job.workspace_name, identifier).await;
            }
            // A Worker job reaches the consumer loop only as a worker branch (a
            // plain worker sub-agent is run by the dispatching tool, never
            // routed), so the launch is always there — reporting the absence
            // rather than dropping the report silently.
            Role::Worker => {
                if let Some(launch) = worker_branch_launch(&job) {
                    route_worker_branch_report(&response, launch, &ws.path).await;
                } else {
                    error!(
                        agent_id = %agent_id,
                        "Message router: worker job carries no worker-branch launch — report dropped",
                    );
                }
            }
            _ => {
                if users.is_empty() {
                    deliver_unregistered_user_response(&response, &job, &role, &file_roots).await;
                } else {
                    deliver_single_user_response(&response, &users[0], &job, &role, &file_roots)
                        .await;
                }
            }
        }
        confirm_pending_delivery(&job).await;

        // A long-lived chat role's session over a summary trigger compacts here,
        // after this reply is delivered and its pending row confirmed (a crash
        // during the compaction can then not replay a delivered job), in the pause
        // before the user's next message.
        agent.maybe_summarize_after_delivery().await;
    }
}

/// The launch of a worker-branch job, when the job is one: `Some` only when the
/// job carries a launch whose role is [`Role::Worker`]. A plain worker sub-agent
/// never reaches the consumer loop, so this is the branch discriminator.
fn worker_branch_launch(job: &AgentJob) -> Option<&crate::Launch> {
    job.custom_role
        .as_ref()
        .filter(|launch| launch.role == Role::Worker)
}

/// Consumer-confirmed delivery: delete the durable pending_jobs row after the
/// agent ran (at-least-once boundary — the row is created before routing and
/// reclaimed only here or by boot replay). Also invoked on the
/// pin-refusal path (a pinned-role envelope with an empty user), the
/// custom-role refusal path, and an unrunnable branch launch
/// ([`report_unrunnable_branch`]): the consumer accept-and-discards the
/// job, and the row is reclaimed here so boot replay does not resurrect it. One
/// sync retry on failure, then log-and-continue: residual duplicate
/// re-delivery at next boot is bounded and accepted. Never called on
/// workspace-not-found for a built-in role (the consumer skips the job) — a
/// pending row for a deleted workspace re-routes and consumer-skips at every
/// boot until the workspace returns; bounded and accepted (purge reclaims only
/// `jobs`, never `pending_jobs` — the at-least-once guarantee keeps unconfirmed
/// rows alive).
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

/// Report a branch job that cannot run because the folder its launch names
/// is gone — the rule being that an unrunnable launch is reported to the admin's
/// Assistant (or the asking branch), naming the branch and the folder it ran in,
/// never to the owner directly and never to a guest. The durable pending row is
/// reclaimed with the notice, because the reported failure IS the delivery.
///
/// Only the positively-gone case reports: a transient store failure on the
/// lookup above is not a verdict about the folder, so it keeps the plain skip
/// (the pending row survives and re-routes). No-op for a built-in role, whose
/// pending row is deliberately kept so it re-routes when the workspace returns
/// (see [`confirm_pending_delivery`]).
async fn report_unrunnable_branch(job: &AgentJob) {
    let Some(launch) = job.custom_role.as_ref() else {
        return;
    };
    if launch.role == Role::Worker {
        route_worker_branch_failure(launch, &job.workspace_name).await;
    } else {
        route_custom_role_failure(
            &launch.name,
            &job.workspace_name,
            launch.identifier.as_deref(),
        )
        .await;
    }
    confirm_pending_delivery(job).await;
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
    file_roots: &[std::path::PathBuf],
) -> DeliverOutcome {
    let Some(recipient) = channel.resolve_recipient(user_name, reply_target) else {
        return DeliverOutcome::Unresolvable;
    };
    match channel
        .send(&SendMessage {
            content: response.to_string(),
            recipient,
            reply_markup: None,
            file_roots: file_roots.to_vec(),
        })
        .await
    {
        Ok(()) => DeliverOutcome::Sent,
        Err(e) => DeliverOutcome::Failed(e),
    }
}

/// The one spelling of the admin-pinning rule every relay-to-the-owner route
/// shares: a job addressed to the single admin's Assistant, in that admin's own
/// personal workspace, on the `gui` channel, with no reply target and no branch.
/// The admin is a constant identity rather than a lookup, so no account-store
/// hiccup can drop an envelope, and the consumer re-validates the pinning
/// invariant ([`crate::users::enforce_personal_pinning`]), so no site re-pins.
/// `originating_workspace` is the folder the relayed content was produced in,
/// when the kind grants its file references that root (see [`MessageKind`]).
fn admin_assistant_job(
    content: String,
    kind: MessageKind,
    originating_workspace: Option<&str>,
) -> AgentJob {
    AgentJob {
        content,
        workspace_name: crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        user_name: crate::users::ADMIN_USER_NAME.to_string(),
        channel: "gui".to_string(),
        kind,
        role: Role::Assistant,
        reply_target: None,
        pending_job_id: None,
        originating_workspace: originating_workspace.map(str::to_string),
        custom_role: None,
        reply_to_custom: None,
    }
}

/// The content of a `<manager-message>` envelope for `response` produced in
/// `source_workspace`, or `None` for an empty response (a sleep-ended or failed
/// turn delivers nothing).
fn manager_message_content(response: &str, source_workspace: &str) -> Option<String> {
    if response.is_empty() {
        return None;
    }
    Some(crate::prompt::substitute(
        &crate::prompt::load_prompt("manager_message.md"),
        &[
            (
                "{{workspace}}",
                crate::tools::envelope_attr(source_workspace).as_str(),
            ),
            ("{{message}}", response),
        ],
    ))
}

/// Auto-delivery hook: EVERY Manager response is additionally routed as a
/// durable `<manager-message>` envelope into the admin's Assistant session —
/// the only manager→assistant delivery path. `source_workspace` is the folder
/// the response was produced in; it rides in the job's `originating_workspace`
/// and in the envelope's `workspace` attribute, so a `[FILE:...]`/`[IMAGE:]`
/// marker in the response stays authorized. Empty responses deliver nothing
/// (sleep-ended/failed turns never reach the delivery branch at all).
async fn route_manager_notify(response: &str, source_workspace: &str) {
    let Some(content) = manager_message_content(response, source_workspace) else {
        return;
    };
    let job = admin_assistant_job(content, MessageKind::ManagerNotify, Some(source_workspace));
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "manager notify envelope").await;
}

/// A job delivered into the conversation that asked for it instead of the
/// admin's Assistant: the destination comes from `asker`, the envelope from
/// `kind`.
fn branch_job(content: String, asker: &Asker, kind: MessageKind) -> AgentJob {
    AgentJob {
        content,
        workspace_name: asker.workspace.clone(),
        user_name: crate::users::ADMIN_USER_NAME.to_string(),
        channel: "gui".to_string(),
        kind,
        role: Role::Custom,
        reply_target: None,
        pending_job_id: None,
        originating_workspace: None,
        custom_role: Some(asker.launch.clone()),
        reply_to_custom: None,
    }
}

/// Deliver a Manager's answer back into the custom-role branch that asked for
/// it — the mirror image of the analyze and worker result envelopes:
/// the job carries the branch address (via its launch) and the folder the
/// asking turn ran in, and is tagged `<manager-message>` because that is the
/// shape `send`'s `manager` addressee promises its caller.
async fn route_manager_reply_to_custom(response: &str, source_workspace: &str, asker: &Asker) {
    let Some(content) = manager_message_content(response, source_workspace) else {
        return;
    };
    let job = branch_job(content, asker, MessageKind::ManagerNotify);
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "manager reply envelope").await;
}

/// Deliver a finished custom role's answer into the admin Assistant's session
/// (the durable `<custom-role-message>` envelope, `MessageKind::CustomRoleNotify`).
/// `source_workspace` is the folder the launch ran in and travels in
/// `originating_workspace`, so a file/media reference in the answer is
/// validated against the folder it was produced in. `identifier` names the
/// branch when the launch ran on a named one. An empty answer delivers nothing.
async fn route_custom_role_notify(
    response: &str,
    role_name: &str,
    source_workspace: &str,
    identifier: Option<&str>,
) {
    if response.is_empty() {
        return;
    }
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("custom_role_message.md"),
        &[
            ("{{role}}", crate::tools::envelope_attr(role_name).as_str()),
            (
                "{{workspace}}",
                crate::tools::envelope_attr(source_workspace).as_str(),
            ),
            (
                "{{branch_attr}}",
                crate::tools::branch_attr(identifier).as_str(),
            ),
            ("{{message}}", response),
        ],
    );
    let job = admin_assistant_job(
        content,
        MessageKind::CustomRoleNotify,
        Some(source_workspace),
    );
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "custom role notify envelope").await;
}

/// Deliver the notice that a custom role's launch failed, naming the role, its
/// branch when it ran on a named one, and the folder it ran in. Nothing goes to
/// the owner directly.
async fn route_custom_role_failure(
    role_name: &str,
    source_workspace: &str,
    identifier: Option<&str>,
) {
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("custom_role_failure.md"),
        &[
            ("{{role}}", crate::tools::envelope_attr(role_name).as_str()),
            (
                "{{workspace}}",
                crate::tools::envelope_attr(source_workspace).as_str(),
            ),
            (
                "{{branch_attr}}",
                crate::tools::branch_attr(identifier).as_str(),
            ),
        ],
    );
    let job = admin_assistant_job(
        content,
        MessageKind::CustomRoleNotify,
        Some(source_workspace),
    );
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "custom role failure envelope").await;
}

/// Deliver a worker branch's finished turn into the conversation that asked for
/// it: the asking custom role's branch when the launch carries an asker
/// ([`crate::Launch::reply_to`]), else the admin Assistant's session. Both shapes
/// carry the `<worker-report>` envelope, naming the folder the turn ran in (its
/// resolved PATH, not the branch's durable address) and its branch. An empty
/// answer delivers nothing, and the report gains no file-delivery power the
/// worker's single-run report does not have.
async fn route_worker_branch_report(response: &str, launch: &crate::Launch, folder: &str) {
    if response.is_empty() {
        return;
    }
    let content = crate::tools::build_worker_envelope(
        "worker_report.md",
        folder,
        launch.identifier.as_deref(),
        response,
    );
    route_worker_branch_envelope(content, launch.reply_to.as_deref()).await;
}

/// Deliver the notice that a worker branch's turn produced no answer, into the
/// same destination as its report — the failure counterpart of
/// [`route_worker_branch_report`]. `folder` is the same resolved path the report
/// names; [`report_unrunnable_branch`] passes the branch's durable address
/// because no workspace was resolved to run in.
async fn route_worker_branch_failure(launch: &crate::Launch, folder: &str) {
    let content = crate::tools::build_worker_envelope(
        "worker_failure.md",
        folder,
        launch.identifier.as_deref(),
        "",
    );
    route_worker_branch_envelope(content, launch.reply_to.as_deref()).await;
}

/// Route a worker branch's envelope: into the asking custom role's branch when
/// the caller carried one (mirroring [`route_manager_reply_to_custom`]), else
/// into the admin's Assistant session (mirroring [`route_custom_role_notify`]).
/// Neither arm sets `originating_workspace`: a worker report must carry exactly
/// the file-delivery roots the single-run worker's report carries — its own
/// delivery workspace alone.
async fn route_worker_branch_envelope(content: String, asker: Option<&Asker>) {
    let job = match asker {
        Some(asker) => branch_job(content, asker, MessageKind::ImplementResult),
        None => admin_assistant_job(content, MessageKind::ImplementResult, None),
    };
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "worker branch report envelope").await;
}

/// Transport-deliver `response` to every channel binding of `users`
/// (broadcast + chat_history persistence happen at the call sites). Logs both
/// diagnostics per reachable binding: an unresolvable recipient (warn — the
/// response is persisted but not transport-delivered) and a transport failure
/// (error). `workspace` names the workspace in the no-channels diagnostic
/// (which also names the affected users).
async fn deliver_response_over_channels(
    response: &str,
    users: &[UserRecord],
    role: Role,
    workspace: &str,
    file_roots: &[std::path::PathBuf],
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
            for binding in &user.channels {
                if binding.channel != *channel_name {
                    continue;
                }
                let reply_target = binding.reply_target.as_deref().unwrap_or(&user.name);
                match deliver_on_channel(
                    channel.as_ref(),
                    &user.name,
                    reply_target,
                    response,
                    file_roots,
                )
                .await
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
/// the user's channel bindings.
///
/// `user` is the resolved [`UserRecord`] for the job's sender — guaranteed
/// non-empty by the consumer loop before calling this function.
///
/// Broadcast+persist is performed exactly once per response, tagged with the
/// originating channel and a fresh `broadcast_id` on the row; transport delivery
/// is delegated to [`deliver_response_over_channels`], which iterates all
/// registered channels and every binding, using each binding's own reply_target.
/// A user registered on multiple channels sees the reply on all of them,
/// regardless of the channel the request came in on.
///
/// `job.reply_target` is not used to scope transport delivery for registered
/// users — each binding supplies its own reply_target on its own channel.
async fn deliver_single_user_response(
    response: &str,
    user: &UserRecord,
    job: &AgentJob,
    role: &Role,
    file_roots: &[std::path::PathBuf],
) {
    // Broadcast + persist (tagged with the originating channel and a fresh
    // dispatch tag on the row).
    let channel = job.channel.as_str();
    broadcast_and_persist_agent_response(
        &user.name,
        channel,
        response,
        role.as_str().to_string(),
        &job.workspace_name,
        Some(crate::generate_id()),
    )
    .await;

    deliver_response_over_channels(
        response,
        std::slice::from_ref(user),
        *role,
        &job.workspace_name,
        file_roots,
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
///
/// `file_roots` anchors and confines outbound media markers exactly as in
/// [`deliver_single_user_response`]; an empty slice names no workspace, so a
/// relative marker cannot be delivered and no `[FILE:...]` is authorized —
/// which is what the inline-confirmation caller passes.
pub async fn deliver_unregistered_user_response(
    response: &str,
    job: &AgentJob,
    role: &Role,
    file_roots: &[std::path::PathBuf],
) {
    let ch = job.channel.as_str();

    // Invariant: never persist/reply under an empty user — seed 'admin'
    // (covers inline confirmations like the clear-session reply path).
    let user_name =
        crate::session::normalize_user_name(&job.user_name, "deliver_unregistered_user_response");

    // Broadcast + persist (works with just strings, no UserRecord needed),
    // tagged with the originating channel and a fresh dispatch tag on the row.
    broadcast_and_persist_agent_response(
        user_name,
        ch,
        response,
        role.as_str().to_string(),
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
    match deliver_on_channel(chan.as_ref(), user_name, reply_target, response, file_roots).await {
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
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
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

    /// A path that is a managed workspace's own folder resolves to that
    /// workspace — not to a separate path-named area over the same tree — so a
    /// launch addressed by path still runs with the workspace's notes, its pause
    /// state and its search engine.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_workspace_path_of_managed_workspace() {
        crate::util::test::init_management_test_stores().await;
        let dir = tempfile::tempdir().expect("create temp dir");
        let canonical = std::fs::canonicalize(dir.path()).expect("canonicalize temp dir");
        // The spelling `workspace::add` writes: canonical, trailing separator.
        let stored = format!(
            "{}{}",
            canonical.to_string_lossy(),
            std::path::MAIN_SEPARATOR
        );
        crate::util::test::create_test_workspace(&stored, "path_resolved_ws").await;

        let resolved = crate::users::resolve_workspace(&dir.path().to_string_lossy())
            .await
            .expect("resolve should succeed")
            .expect("a folder that is a managed workspace resolves");

        assert_eq!(resolved.name, "path_resolved_ws");
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

        // The admin is auto-created by ensure_admin_user.
        let user = resolve_single_user("admin").await;
        assert!(user.is_some(), "the admin should exist after store init");
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
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
        };

        // Should complete without panic.
        deliver_unregistered_user_response(
            "response to unregistered user",
            &job,
            &Role::Assistant,
            &[],
        )
        .await;
    }

    /// `deliver_single_user_response` completes without error and broadcasts
    /// to all of the user's channel bindings. Here admin is bound to "gui" and the job originates on
    /// "gui" — broadcast+persist runs and transport delivery reaches the
    /// matching "gui" binding. No-panic is the assertion.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_single_user_response() {
        setup_response_test_infra().await;

        // Give the admin a "gui" channel binding so the delivery
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
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
        };

        // Should complete without panic — broadcasts to the "gui" binding.
        deliver_single_user_response(
            "response to registered user",
            &user,
            &job,
            &Role::Assistant,
            &[],
        )
        .await;
    }

    /// `deliver_single_user_response` handles the case where the user has NO
    /// channel bindings at all — only broadcast+persist runs, transport
    /// delivery is skipped. No-panic is the assertion.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_single_user_no_bindings() {
        setup_response_test_infra().await;

        // The admin exists but has no "gui" channel binding.
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
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
        };

        // Should complete without panic — broadcast+persist runs, transport
        // delivery is skipped because there are no matching bindings.
        deliver_single_user_response(
            "response to registered user without matching binding",
            &user,
            &job,
            &Role::Assistant,
            &[],
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
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
        };

        deliver_single_user_response(
            "broadcast to all bindings",
            &user,
            &job,
            &Role::Assistant,
            &[],
        )
        .await;

        let captured = sent.lock().unwrap_poison();
        assert!(
            !captured.is_empty(),
            "spy channel should have captured at least one broadcast send",
        );
    }

    // ── Job builder tests ───────────────────────────────────────────────

    /// `agent_message_job` threads the normalized identity + durable kind.
    #[test]
    fn agent_message_job_threading() {
        let job = agent_message_job("hello".to_string(), "proj_ws".to_string(), "alice", None);
        assert_eq!(job.user_name, "alice");
        assert_eq!(job.kind, MessageKind::AgentMessage);
        assert_eq!(job.role, Role::Manager);
        assert_eq!(job.workspace_name, "proj_ws");
        assert!(job.reply_to_custom.is_none());

        // Empty user_name normalizes to the seeded 'admin'.
        let job = agent_message_job("hello".to_string(), "proj_ws".to_string(), "", None);
        assert_eq!(job.user_name, "admin");
    }

    /// Auto-delivery round trip: a Manager response persists one durable
    /// `<manager-message>` envelope for the admin, addressed to the admin's
    /// personal Assistant session, carrying the FULL untruncated response, with
    /// the source workspace in `originating_workspace` and in the envelope
    /// attribute. Empty responses persist nothing. Cleans up the row it created
    /// so other tests' boot-replay paths never see it.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn manager_notify_delivers_to_admin_assistant() {
        setup_response_test_infra().await;
        let target = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);

        // Snapshot existing rows so cleanup removes ONLY rows this test created.
        let before: HashSet<String> = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .expect("list pending")
            .into_iter()
            .map(|r| r.id)
            .collect();

        // Empty response: nothing persisted.
        route_manager_notify("", "team_ws").await;
        // Real response: one durable envelope with the full content.
        let response = "FULL RESPONSE BODY — must survive verbatim §1 ✓";
        route_manager_notify(response, "team_ws").await;

        let conn = &crate::session::store().conn;
        let pending = crate::jobs::list_pending_jobs(conn)
            .await
            .expect("list pending");
        // Scoped to the rows this test created: the shared global store is
        // written by parallel tests too (see `src/session/mod.rs`), so a bare
        // count for the target session would include rows that are not this
        // test's.
        let matching: Vec<_> = pending
            .iter()
            .filter(|r| r.target_agent_id == target && !before.contains(&r.id))
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "exactly one notify row for the admin's assistant"
        );
        let deserialized: AgentJob =
            serde_json::from_str(&matching[0].envelope).expect("envelope deserializes to AgentJob");
        assert_eq!(deserialized.kind, MessageKind::ManagerNotify);
        assert_eq!(
            deserialized.workspace_name,
            crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME)
        );
        assert_eq!(deserialized.user_name, crate::users::ADMIN_USER_NAME);
        assert_eq!(
            deserialized.originating_workspace.as_deref(),
            Some("team_ws"),
            "the relayed job must name the source workspace so its [FILE:...] \
             markers stay authorized"
        );
        assert!(
            deserialized
                .content
                .contains("<manager-message workspace=\"team_ws\">")
                && deserialized.content.contains(response),
            "envelope must carry the source workspace and the FULL response: {}",
            deserialized.content
        );

        // Pre-unification ManagerReply rows must stay replayable (serde alias).
        let legacy: AgentJob = serde_json::from_str(
            r#"{"content":"<manager-reply workspace=\"old_ws\">hi</manager-reply>","workspace_name":"personal:alice","user_name":"alice","channel":"gui","kind":"ManagerReply","role":"Assistant","reply_target":null}"#,
        )
        .expect("legacy ManagerReply row deserializes");
        assert_eq!(legacy.kind, MessageKind::ManagerNotify);
        assert_eq!(
            legacy.originating_workspace, None,
            "a legacy row without the field must default to no originating workspace"
        );

        for row in pending.iter().filter(|r| !before.contains(&r.id)) {
            crate::jobs::delete_pending_job(conn, &row.id)
                .await
                .expect("delete pending row");
        }
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

    // ── Structural message-kind invariants ───────────────────────────────

    /// `RecoveryRetry` is intentionally NOT `UserMessage`.  The emoji gate in
    /// `consumer_loop` uses `job.kind == MessageKind::UserMessage` to decide whether
    /// to send the failure emoji — `RecoveryRetry` is automatically excluded
    /// because the equality check is specific to `UserMessage`.
    ///
    /// The same structural invariant holds for the Assistant↔Manager kinds:
    /// `AgentMessage` and `ManagerNotify` are deliberately distinct from
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

    /// A custom role's answer is an internal envelope addressed to the admin's
    /// personal Assistant — never the owner's channels and never Telegram —
    /// persisted as one durable `<custom-role-message>` row naming the folder the
    /// launch ran in, so a `[FILE:]`/`[IMAGE:]` marker in the answer can be
    /// relayed out of it. An empty answer delivers nothing. Cleans up the row it
    /// created so other tests' boot-replay paths never see it.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn custom_role_notify_targets_the_admin_assistant() {
        setup_response_test_infra().await;
        let conn = &crate::session::store().conn;
        let target = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);

        let before: HashSet<String> = crate::jobs::list_pending_jobs(conn)
            .await
            .expect("list pending")
            .into_iter()
            .map(|r| r.id)
            .collect();

        // Empty answer: nothing persisted. Real answer: one durable envelope.
        route_custom_role_notify("", "reviewer", "/data/proj", Some("feature")).await;
        route_custom_role_notify("the answer", "reviewer", "/data/proj", Some("feature")).await;

        let pending = crate::jobs::list_pending_jobs(conn)
            .await
            .expect("list pending");
        let matching: Vec<_> = pending
            .iter()
            .filter(|r| r.target_agent_id == target && !before.contains(&r.id))
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "exactly one notify row for the admin's assistant"
        );
        let job: AgentJob =
            serde_json::from_str(&matching[0].envelope).expect("envelope deserializes to AgentJob");
        assert_eq!(job.kind, MessageKind::CustomRoleNotify);
        assert_eq!(job.role, Role::Assistant);
        assert_eq!(job.user_name, crate::users::ADMIN_USER_NAME);
        assert!(
            job.custom_role.is_none(),
            "the notify job carries no branch launch of its own"
        );
        assert_eq!(
            job.originating_workspace.as_deref(),
            Some("/data/proj"),
            "the folder the launch ran in is the answer's file-delivery root"
        );
        assert_eq!(crate::jobs::envelope_target(&job), target);
        assert!(
            job.content.starts_with(
                "<custom-role-message role=\"reviewer\" workspace=\"/data/proj\" branch=\"feature\">"
            ) && job.content.contains("the answer"),
            "the envelope names the role, the folder and the branch, and carries \
             the answer: {}",
            job.content
        );

        for row in pending.iter().filter(|r| !before.contains(&r.id)) {
            crate::jobs::delete_pending_job(conn, &row.id)
                .await
                .expect("delete pending row");
        }
    }

    /// A worker branch's report has the same shape as the single-run worker's: the
    /// `<worker-report>` envelope names the folder PATH the turn ran in (the
    /// resolved workspace path, never the branch's durable address) and its
    /// `branch` when the branch was named, and the delivered job carries NO
    /// `originating_workspace` — so it gains no `[FILE:...]` root the single run
    /// lacks. An empty answer delivers nothing.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn worker_branch_report_matches_the_single_run_shape() {
        setup_response_test_infra().await;

        let before: HashSet<String> = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .expect("list pending")
            .into_iter()
            .map(|r| r.id)
            .collect();

        let launch = crate::Launch {
            role: Role::Worker,
            name: String::new(),
            identifier: Some("feature".to_string()),
            prompt: String::new(),
            tools: Vec::new(),
            agent_id: crate::session::worker_branch_agent_id("feature"),
            reply_to: None,
        };
        // The branch's durable ADDRESS (what a folder-less branch records) — the
        // report must name the resolved folder path instead of it.
        let mut job = make_job(Role::Worker, "personal:admin", "admin", "gui");
        job.custom_role = Some(launch.clone());

        let assistant_agent = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut assistant_rx = register_agent(&assistant_agent);
        let mut branch_rx = register_agent(&launch.agent_id);

        // No asking custom role: the report lands in the admin's Assistant session.
        route_worker_branch_report("did the thing", &launch, "/data/folder").await;
        let delivered = assistant_rx
            .try_recv()
            .expect("the report is delivered to the admin's assistant");
        assert_eq!(delivered.kind, MessageKind::ImplementResult);
        assert!(
            delivered
                .content
                .contains("<worker-report folder=\"/data/folder\" branch=\"feature\">"),
            "the report names the folder PATH and the branch: {}",
            delivered.content
        );
        assert!(
            !delivered.content.contains("personal:admin"),
            "the branch's durable address must never stand in for the folder path: {}",
            delivered.content
        );
        assert!(
            delivered.originating_workspace.is_none(),
            "a worker report carries no extra file-delivery root: {:?}",
            delivered.originating_workspace
        );

        // A folder whose name carries a quote or an angle bracket cannot deform
        // the envelope: the three characters that could end the attribute are
        // escaped.
        route_worker_branch_report("did the thing", &launch, "/data/we\"ird<dir>").await;
        let hostile = assistant_rx
            .try_recv()
            .expect("the report is delivered to the admin's assistant");
        assert!(
            hostile.content.starts_with(
                "<worker-report folder=\"/data/we&quot;ird&lt;dir&gt;\" branch=\"feature\">"
            ),
            "a model-chosen folder must not be able to deform the envelope: {}",
            hostile.content
        );

        // An ordinary name that happens to carry an apostrophe or an ampersand
        // is NOT distorted: the model is told a folder it can hand back.
        route_worker_branch_report("did the thing", &launch, "/data/o'brien & co").await;
        let ordinary = assistant_rx
            .try_recv()
            .expect("the report is delivered to the admin's assistant");
        assert!(
            ordinary
                .content
                .starts_with("<worker-report folder=\"/data/o'brien & co\" branch=\"feature\">"),
            "a folder name must read back as the folder it names: {}",
            ordinary.content
        );

        // An asking custom role's branch: same shape, no extra root either. The
        // asker travels on the LAUNCH (its single carrier), which is what the
        // report reads.
        let asking = crate::Launch {
            reply_to: Some(Box::new(Asker {
                launch: launch.clone(),
                workspace: "role_ws".to_string(),
            })),
            ..launch.clone()
        };
        route_worker_branch_report("did the thing again", &asking, "/data/folder").await;
        let re_delivered = branch_rx
            .try_recv()
            .expect("the report re-enters the asking branch");
        assert_eq!(re_delivered.workspace_name, "role_ws");
        assert!(
            re_delivered.originating_workspace.is_none(),
            "the asking-branch arm carries no extra file-delivery root either: {:?}",
            re_delivered.originating_workspace
        );

        // An empty answer delivers nothing.
        route_worker_branch_report("", &launch, "/data/folder").await;
        assert!(
            branch_rx.try_recv().is_err(),
            "an empty answer delivers nothing"
        );

        unregister_agent(&assistant_agent);
        unregister_agent(&launch.agent_id);

        let pending = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .expect("list pending");
        for row in pending.iter().filter(|r| !before.contains(&r.id)) {
            crate::jobs::delete_pending_job(&crate::session::store().conn, &row.id)
                .await
                .expect("delete pending row");
        }
    }
}
