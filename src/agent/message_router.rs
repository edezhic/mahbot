//! Agent-ID-based message router: every agent instance gets its own
//! [`mpsc::UnboundedSender`] in a global [`HashMap`] keyed by unique agent ID, so
//! jobs reach their consumer directly and no agent ever blocks another.
//!
//! Producers are user chat, ticket transitions, sub-agent results, boot-time replay
//! and dead-session recovery; the router stays agnostic to the job's origin.
//! [`AgentJob`] carries the [`Role`] so the router never parses agent IDs.

use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};
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

/// Emoji sent when an agent finishes with no response (LLM error, retry exhaustion,
/// context overflow) — never when cancelled or shutting down. Language-agnostic.
const AGENT_FAILURE_EMOJI: &str = "🤖⚠️🔄";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    /// User-typed message (chat or inline-button callback); for Manager the
    /// ticket_chronicle timeline drains before the agent runs, for no other role.
    UserMessage,
    /// Ticket transition notification; only enqueued for the Manager role.
    TicketNotify,
    /// An async AnalyzeTool sub-agent's result, injected back into the caller's session;
    /// a round its caller cancels produces none.
    AnalyzeToolResult,
    /// An async single-run worker dispatch's result (`send` `worker`), injected back into
    /// the caller's session; also the kind a worker branch reports under.
    ImplementResult,
    /// An async deep analysis run's result (`analyze` at `deep` depth) — one envelope per run.
    ResearchResult,
    /// Ticket comment added mid-work: through the agent's inbox, never a consumer loop.
    TicketComment,
    /// Dead-session recovery retry: routes without appending a user message, and suppresses
    /// the emoji, which fires once on the original failure.
    RecoveryRetry,
    /// An assistant's message into a workspace Manager's session (`<assistant-message>`
    /// envelope), durable so the Manager's chronicle/CDC drain applies. Not
    /// `UserMessage`: the failure-emoji gate must structurally exclude it.
    AgentMessage,
    /// Durable auto-delivery of a Manager's answer into the admin's Assistant
    /// (`<manager-message>`): the job's workspace is the admin's personal one, while
    /// `originating_workspace` and the envelope's `workspace` attribute name the source,
    /// keeping its file references authorized. The serde alias keeps pre-unification
    /// `ManagerReply` rows replayable at boot.
    #[serde(alias = "ManagerReply")]
    ManagerNotify,
    /// A custom-role branch's answer or failure, routed into the asking conversation
    /// (`<custom-role-message>`): the admin's Assistant when that branch was the asker,
    /// else the asking custom role's branch. Durable through `stamp_and_route`; not
    /// `UserMessage`, so the failure-emoji gate excludes it.
    CustomRoleNotify,
    /// A one-way handover from the owner's Assistant into a guest's own Assistant
    /// (`<owner-assistant-message>`): the job runs AS that guest in their personal
    /// workspace and answers into the guest's channels, nothing returns to the sender.
    /// `originating_workspace` stays unset, so the guest's workspace is the only root its
    /// RELATIVE media markers resolve in.
    GuestMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentJob {
    pub content: String,
    pub workspace_name: String,
    /// Sender identity — used for per-user response delivery.
    pub user_name: String,
    pub channel: String,
    pub kind: MessageKind,
    /// The agent's [`Role`], embedded so the router never parses agent IDs.
    pub role: Role,
    /// Original reply target (e.g. Telegram chat_id), used by
    /// [`deliver_unregistered_user_response`] when the users DB has no record.
    pub reply_target: Option<String>,
    /// Durable pending_jobs row id. The consumer deletes the row only after `run_agent`
    /// returns — the at-least-once delivery boundary.
    #[serde(default)]
    pub pending_job_id: Option<String>,
    /// For relayed jobs, the workspace the relayed content originated in, so its media
    /// markers resolve and deliver from where they were produced.
    #[serde(default)]
    pub originating_workspace: Option<String>,
    /// The branch launch this job runs ([`Role::Custom`] or [`Role::Worker`]);
    /// `#[serde(default)]` keeps persisted envelopes decodable at boot replay.
    #[serde(default)]
    pub custom_role: Option<crate::Launch>,
    /// The conversation a forwarded reply must return to when the caller was a custom
    /// role (see [`route_manager_reply_to_custom`]); a worker branch carries its asker on
    /// the launch instead ([`crate::Launch::reply_to`]) — one carrier per fact.
    #[serde(default)]
    pub reply_to_custom: Option<Asker>,
    /// The user marked the message urgent: it must not wait behind the running turn.
    /// Only a surface a user types into sets it ([`route_user_message`]).
    #[serde(default)]
    pub urgent: bool,
}

/// The conversation that asked for a forwarded reply — the asking branch's launch and
/// the folder its turn ran in, so an answer returns there instead of the admin's
/// Assistant. Carried by `crate::Launch::reply_to` and `AgentJob::reply_to_custom`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Asker {
    /// The asking branch's launch, so the reply re-enters that conversation.
    pub launch: crate::Launch,
    /// The folder the asking turn runs in.
    pub workspace: String,
}

/// Global router table: maps agent ID → that agent's consumer.
static ROUTER: OnceLock<RwLock<HashMap<String, Consumer>>> = OnceLock::new();

/// How long an announced urgent message has to reach the queue before the turn cut for
/// it answers what it already has: generous, because its attachments are still being
/// understood while the turn holds; bounded, so a hand-over that never arrives cannot
/// stall the queue.
const URGENT_ARRIVAL_SECS: u64 = 600;

/// One agent's consumer: its job queue, plus urgent announcements on their way.
struct Consumer {
    jobs: mpsc::UnboundedSender<AgentJob>,
    urgent: Arc<UrgentArrivals>,
}

/// Urgent messages announced for one agent and not yet handed to its queue, with the
/// token that ends each wait: the consumer cuts its running turn and holds the next.
#[derive(Default)]
struct UrgentArrivals {
    announced: std::sync::Mutex<Vec<(tokio::time::Instant, CancellationToken)>>,
    changed: tokio::sync::Notify,
}

impl UrgentArrivals {
    /// An urgent message for this agent is on its way: cut the running turn and wait.
    fn announce(&self) -> UrgentAnnouncement {
        let token = CancellationToken::new();
        self.announced
            .lock()
            .unwrap_poison()
            .push((tokio::time::Instant::now(), token.clone()));
        // `notify_waiters`, not `notify_one`: only a consumer that is watching — one with
        // a turn running — is cut.
        self.changed.notify_waiters();
        UrgentAnnouncement(token)
    }

    /// The announcements outstanding now — what a turn cut at this instant waits for.
    /// Taking them means one is never waited for twice.
    fn take(&self) -> Vec<(tokio::time::Instant, CancellationToken)> {
        std::mem::take(&mut *self.announced.lock().unwrap_poison())
    }
}

/// An urgent message announced for an agent, owned by the task staging it: dropping
/// it ends the wait it started — after the hand-over, or because that task died.
#[must_use]
pub struct UrgentAnnouncement(CancellationToken);

impl Drop for UrgentAnnouncement {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Initialise the global router table once the Tokio runtime is active; consumers are
/// spawned lazily on first [`route`].
pub fn init_global() -> anyhow::Result<()> {
    ROUTER
        .set(RwLock::new(HashMap::new()))
        .map_err(|_| anyhow::anyhow!("ROUTER already initialised"))?;
    Ok(())
}

/// Cut the turn `agent_id` runs right now because a user's urgent message is on its way:
/// the cut lands before anything about that message is understood (see
/// [`route_user_message`]). `None` when the agent has no consumer at all.
#[must_use]
pub fn announce_urgent(agent_id: &str) -> Option<UrgentAnnouncement> {
    let table = ROUTER.get()?.read().unwrap_poison();
    table
        .get(agent_id)
        .map(|consumer| consumer.urgent.announce())
}

/// Route a job to the consumer for `agent_id`. Fast path: a read lock, a lookup and a
/// send. Slow path (agent not yet registered): write lock, new channel, spawn a
/// [`consumer_loop`], store the sender, forward the job — double-checked against races.
pub fn route(agent_id: &str, job: AgentJob) {
    {
        let map = ROUTER
            .get()
            .expect("ROUTER not initialised — call init_global() first");
        let guard = map.read().unwrap_poison();
        if let Some(consumer) = guard.get(agent_id) {
            if consumer.jobs.send(job).is_err() {
                error!(agent_id = %agent_id, "Router: consumer dropped — failed to route job");
            }
            return;
        }
    }

    let map = ROUTER
        .get()
        .expect("ROUTER not initialised — call init_global() first");
    let mut guard = map.write().unwrap_poison();

    // Double-check: another task may have registered the agent meanwhile.
    if let Some(consumer) = guard.get(agent_id) {
        if consumer.jobs.send(job).is_err() {
            error!(agent_id = %agent_id, "Router: consumer dropped (double-check) — failed to route job");
        }
        return;
    }

    let (tx, rx) = mpsc::unbounded_channel::<AgentJob>();
    let urgent = Arc::new(UrgentArrivals::default());
    let urgent_for_consumer = Arc::clone(&urgent);
    let agent_id_for_consumer = agent_id.to_string();
    let agent_id_for_cleanup = agent_id_for_consumer.clone();

    tokio::spawn(async move {
        // catch_unwind so a panic does not leave a dead sender in the router table and
        // lose this agent ID's messages for good.
        #[expect(clippy::large_futures)] // one consumer per agent, spawned once
        let result = std::panic::AssertUnwindSafe(consumer_loop(
            agent_id_for_consumer,
            rx,
            urgent_for_consumer,
        ))
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

    guard.insert(agent_id.to_string(), Consumer { jobs: tx, urgent });
}

/// The `MessageKind::UserMessage` envelope every owner-side producer builds; fields such a
/// message never carries stay at their defaults, `urgent` included.
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
        urgent: false,
    }
}

/// The agent ID [`route_user_message`] computes, for a caller that must reach the agent
/// before the message does ([`announce_urgent`]).
#[must_use]
pub fn user_message_agent_id(user_name: &str, role: Role, workspace_name: &str) -> String {
    // The same normalization `route_user_message` applies, so the two can never
    // name different agents for one message.
    let user_name = crate::session::normalize_user_name(user_name, "route_user_message");
    crate::session::resolve_agent_id(user_name, role.as_str(), workspace_name)
}

/// Route a user message to the agent for `role` in a workspace, resolving the agent ID
/// exactly as [`user_message_agent_id`] does; broadcast, persistence and enrichment stay
/// at the call sites. `urgent` (see `crate::channels::take_urgent`) cuts the running turn
/// and brings everything already waiting with it instead of queueing behind it.
///
/// At-least-once: Manager-bound messages are persisted to `pending_jobs` BEFORE routing,
/// so a crash between the two replays the row at boot (dedup prevents a double append);
/// non-Manager ones are covered only by the dead-session poller. During the graceful
/// drain the message is persisted but NOT routed — boot replay reclaims the row.
pub async fn route_user_message(
    content: String,
    workspace_name: String,
    user_name: String,
    channel: String,
    role: Role,
    reply_target: Option<String>,
    urgent: bool,
) {
    // Invariant: the routed user identity is never empty — an admin account is always
    // seeded. Normalized once here so the agent ID key and the delivery payload always
    // name the same person, never a bare "_ws_role" key.
    let user_name =
        crate::session::normalize_user_name(&user_name, "route_user_message").to_string();
    let agent_id = crate::session::resolve_agent_id(&user_name, role.as_str(), &workspace_name);

    let mut job = user_message_job(
        content,
        workspace_name,
        user_name,
        channel,
        role,
        reply_target,
    );
    job.urgent = urgent;

    if job.role == Role::Manager {
        stamp_and_route(job, &agent_id, "manager message").await;
        return;
    }
    route(&agent_id, job);
}

// Persist `job` to `pending_jobs` and stamp the row id onto it — the at-least-once
// boundary. An `Err` is a failed write, not a dropped message.
async fn persist_and_stamp(job: &mut AgentJob) -> anyhow::Result<()> {
    let id = crate::generate_id();
    persist_pending(job, id.clone()).await?;
    job.pending_job_id = Some(id);
    Ok(())
}

// Who answers for a durable handover: what a failed `pending_jobs` write means is
// the PRODUCER's choice, never the message's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AnsweredFor {
    // Nobody is told about this handover's failure — an agent, whose own turn is durable
    // already, or a background sweep: the failed write is logged and the envelope routed.
    Agent,
    /// The command line: the failure is returned with NOTHING routed, since its caller is
    /// answered and gone the moment it returns.
    CommandLine,
}

// Hand one durable envelope over under `answered_for`'s contract — the ONE handover body:
// persist to `pending_jobs`, then route unless a drain began after the write (the row is
// left to boot replay; the drain must not be raced with a live route). Only a
// [`CommandLine`](AnsweredFor::CommandLine) handover returns `Err`; any other failure is
// logged and routed best-effort — the price of never dropping a failed write.
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

/// [`hand_over`] under [`AnsweredFor::Agent`]: a failed write is logged and the envelope
/// routed best-effort rather than dropped.
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
        urgent: false,
    }
}

/// Route an assistant agent's message into the workspace Manager's session
/// ([`MessageKind::AgentMessage`]). A custom role's `asker` carries the asking branch,
/// answered through [`route_manager_reply_to_custom`]; a non-custom caller's answer
/// auto-delivers via [`route_manager_notify`].
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

/// Hand one message from the owner's Assistant to a guest's own Assistant
/// ([`MessageKind::GuestMessage`], durable) — one-way, nothing addressed back to the
/// sender. The job runs AS the guest, so pinning and delivery give it the guest's own
/// toolset, conversation and channels; `tools::send` wrapped the text already.
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
        urgent: false,
    };
    let target = crate::jobs::envelope_target(&job);
    hand_over(job, &target, "guest message", answered_for).await
}

/// Hand one message from the COMMAND LINE to the owner's own assistant as an ordinary
/// incoming message — broadcast to the GUI, persisted to `chat_history`, mirrored to his
/// bound Telegram — attributed to [`crate::cli::CHANNEL`], never in the message itself, so
/// a shell-sent message can be told from a typed (`gui`) or spoken (`voice`) one. The
/// durable handover comes FIRST and the history copy second, the opposite of the other
/// producers: the conversation must not show a message the caller was told did not get
/// through. `Err` (no owner, or a failed write) REFUSES the handover.
pub(crate) async fn deliver_command_line_message(content: String) -> anyhow::Result<()> {
    // The owner the CALL carries, never a second copy of the constant: the command line
    // scopes the identity it acts with ([`crate::cli::instance`]). Checked rather than
    // defaulted — an empty identity would address a `personal:` workspace that does not exist.
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

// Persist an envelope to `pending_jobs`; the target agent id is derived from it by
// [`crate::jobs::pending_job_params`]. Used by every durable producer.
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

/// Deliver a durable notice into an Assistant session: persist to `pending_jobs` BEFORE
/// routing (a crash after persisting replays it at boot), then route. `agent_id` is the
/// resolved session key — automation delivery passes the stored id, since re-deriving it would
/// double-escape colliding user names. An `Err` is a failed persistence.
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

// The envelope a notice addressed to an Assistant carries; both delivery paths
// build it here so the fields match an incoming message's.
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

/// Post a notice into an Assistant's ordinary queue with no `pending_jobs` row — the
/// at-most-once sibling of [`deliver_assistant_notice`], for a producer that owns its own
/// bookkeeping: settle the record after this returns if a repeat is affordable, else before.
pub(crate) fn post_assistant_notice(agent_id: &str, user_name: &str, content: String) {
    route(agent_id, assistant_notice_job(content, user_name));
}

/// Register an agent without a consumer loop and hand back the receiver (`llm_loop` drains
/// it with `try_recv()`); the stored sender lets [`try_route`] deliver mid-work messages.
/// Pair it with [`unregister_agent`] — `run_agent`'s exit guard calls it on every path.
pub fn register_agent(agent_id: &str) -> mpsc::UnboundedReceiver<AgentJob> {
    let (tx, rx) = mpsc::unbounded_channel::<AgentJob>();
    let map = ROUTER
        .get()
        .expect("ROUTER not initialised — call init_global() first");
    let mut guard = map.write().unwrap_poison();
    guard.insert(
        agent_id.to_string(),
        Consumer {
            jobs: tx,
            urgent: Arc::new(UrgentArrivals::default()),
        },
    );
    rx
}

/// Unregister an agent from the router table: [`try_route`] then returns `false` for
/// it. Safe when the agent was never registered.
pub fn unregister_agent(agent_id: &str) {
    let Some(map) = ROUTER.get() else { return };
    let mut guard = map.write().unwrap_poison();
    guard.remove(agent_id);
}
/// Try to route a job to a previously registered agent: `true` when delivered, `false`
/// when not (the caller may then persist the job for the next dispatch). Never spawns a
/// consumer — the fast path for mid-work delivery.
pub fn try_route(agent_id: &str, job: AgentJob) -> bool {
    let Some(map) = ROUTER.get() else {
        return false;
    };
    let guard = map.read().unwrap_poison();
    guard
        .get(agent_id)
        .is_some_and(|consumer| consumer.jobs.send(job).is_ok())
}

// The turn a queue forms: everything already waiting up to and including the first urgent
// message, answered together with the urgent one last — else the one job at the front.
// Everything behind that urgent message, a second urgent included, is the next turn's.
fn take_turn_batch(waiting: &mut std::collections::VecDeque<AgentJob>) -> Vec<AgentJob> {
    match waiting.iter().position(|job| job.urgent) {
        Some(at) => waiting.drain(..=at).collect(),
        None => waiting.drain(..1).collect(),
    }
}

// Hold the turn about to start for the urgent messages announced before it: their cut was
// taken at staging, so they are still on their way and must be answered last, with everything
// already waiting. The wait ends when the message is handed over or its staging task is gone,
// and is bounded from the announcement; a shutdown or drain ends it at once. `false` means the
// consumer must stop.
async fn hold_for_urgent(
    waiting: &mut std::collections::VecDeque<AgentJob>,
    urgent: &UrgentArrivals,
    rx: &mut mpsc::UnboundedReceiver<AgentJob>,
    shutdown: &CancellationToken,
    agent_id: &str,
    hold: std::time::Duration,
) -> bool {
    // Only what was announced before this turn started is waited for: a later
    // announcement either lands in this turn's queue or cuts the turn after the wait.
    let awaiting = urgent.take();
    let Some(deadline) = awaiting
        .iter()
        .map(|(announced, _)| *announced + hold)
        .min()
    else {
        return true;
    };
    loop {
        let settled = async {
            for (_, token) in &awaiting {
                token.cancelled().await;
            }
        };
        let pulled = tokio::select! {
            pulled = tokio::time::timeout_at(deadline, rx.recv()) => pulled,
            () = settled => break,
            () = shutdown.cancelled() => return false,
            () = crate::shutdown::drain_wait() => return false,
        };
        match pulled {
            Ok(Some(job)) => {
                waiting.push_back(job);
                // A queue that never empties must not postpone the turn past the
                // window: the deadline is checked here because `timeout_at` polls
                // the receive first and a steady stream would never reach it.
                if tokio::time::Instant::now() >= deadline {
                    break;
                }
            }
            // The queue closed: the caller's next receive sees it.
            Ok(None) => return true,
            Err(_) => {
                // A message that settled as the window ran out was handed over
                // after all — only one that is still awaited never arrived.
                if awaiting.iter().any(|(_, token)| !token.is_cancelled()) {
                    warn!(
                        agent_id = %agent_id,
                        "Message router: announced urgent message never arrived — answering what is queued",
                    );
                }
                break;
            }
        }
    }
    // The message that ended the wait was handed to the queue before its
    // announcement dropped, so take what is already here with it.
    while let Ok(job) = rx.try_recv() {
        waiting.push_back(job);
    }
    true
}

/// The consumer task for one agent: one turn at a time in queue order, an urgent message
/// cutting the running turn and answered by the next one (see [`AgentJob::urgent`]). The
/// wrapper in [`route`] removes the router entry on exit and on panic.
#[expect(clippy::too_many_lines)]
async fn consumer_loop(
    agent_id: String,
    mut rx: mpsc::UnboundedReceiver<AgentJob>,
    urgent: Arc<UrgentArrivals>,
) {
    let shutdown = crate::shutdown::shutdown_token();
    let urgent_hold = std::time::Duration::from_secs(URGENT_ARRIVAL_SECS);
    // Jobs that reached this queue while a turn was running: the next turn's
    // items, in queue order.
    let mut waiting: std::collections::VecDeque<AgentJob> = std::collections::VecDeque::new();

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

        // Nothing is waiting: block for the next job.
        if waiting.is_empty() {
            let job = tokio::select! {
                job = rx.recv() => match job {
                    Some(job) => job,
                    None => break,
                },
                () = shutdown.cancelled() => {
                    info!(agent_id = %agent_id, "Message router: shutting down (global shutdown)");
                    break;
                }
            };
            waiting.push_back(job);
        }
        // Everything else the queue already holds is gathered now, so an urgent message
        // among them starts ONE turn with them instead of interrupting a turn already begun.
        while let Ok(job) = rx.try_recv() {
            waiting.push_back(job);
        }
        // An announced urgent message is still being staged: wait for the message itself
        // and answer it last. A shutdown or drain ends this loop here, as the guards would.
        if !hold_for_urgent(
            &mut waiting,
            &urgent,
            &mut rx,
            &shutdown,
            agent_id.as_str(),
            urgent_hold,
        )
        .await
            || crate::shutdown::aborting()
        {
            info!(agent_id = %agent_id, "Message router: draining — no new jobs pulled");
            break;
        }
        // The batch ends with the first urgent message it holds — taken last, after
        // everything already waiting; what is behind it is a later arrival with its own turn.
        let batch = take_turn_batch(&mut waiting);

        // Execution-time normalization every job of the batch goes through (role, the
        // pinned-role rule, the custom-role guard); a refused job is confirmed and dropped,
        // so it cannot take the rest of the batch with it.
        let mut validated: Vec<AgentJob> = Vec::with_capacity(batch.len());
        for mut item in batch {
            // TicketComment jobs NEVER reach the consumer loop — `try_route()` delivers them
            // to agents draining in `llm_loop`. One arriving here means a misuse of `route()`.
            if item.kind == MessageKind::TicketComment {
                warn!(
                    agent_id = %agent_id,
                    "Consumer loop received TicketComment — was try_route() used instead of route()? Discarding",
                );
                confirm_pending_delivery(&item).await;
                continue;
            }

            // A launch and the role are one fact: derive the role from the launch FIRST, so
            // a job built without the derivation cannot smuggle a launch past the guards.
            item.role = item.role.for_launch(item.custom_role.as_ref());

            // Execution-time invariant: pinned roles (Assistant) never run outside the user's
            // personal workspace, whatever the producer stored; empty-user envelopes are refused.
            let Some(pinned_ws) = crate::users::enforce_personal_pinning(
                item.role,
                &item.workspace_name,
                &item.user_name,
            ) else {
                error!(
                    agent_id = %agent_id,
                    role = %item.role.as_str(),
                    workspace = %item.workspace_name,
                    "Message router: pinned role with empty user — refusing to run unpinned, skipping job"
                );
                confirm_pending_delivery(&item).await;
                continue;
            };
            item.workspace_name = pinned_ws;

            // A custom-role job is meaningless without its launch, and a branch whose record
            // was since deleted must never resume with a definition that no longer exists (a
            // round it dispatched may still complete after the delete). Both are refused in
            // this one funnel, the durable row reclaimed so boot replay cannot resurrect them.
            if item.role == Role::Custom {
                let refusal = match item.custom_role.as_ref() {
                    None => Some("the job carries no custom-role launch"),
                    Some(launch) => match crate::custom_role::get(&launch.name).await {
                        Ok(Some(_)) => None,
                        Ok(None) => Some("the custom role no longer exists"),
                        Err(e) => {
                            // A transient read failure is not a verdict about the role:
                            // run the turn and let the error surface where it needs the store.
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
                    confirm_pending_delivery(&item).await;
                    continue;
                }
            }
            validated.push(item);
        }
        // The turn's identity — surface, user, channel, reply target — is the last item's:
        // the urgent message is what the turn answers, the items before it its messages.
        let Some(job) = validated.pop() else {
            continue;
        };
        let queued = validated;

        debug!(
            agent_id = %agent_id,
            workspace = %job.workspace_name,
            user = %job.user_name,
            kind = ?job.kind,
            messages = queued.len() + 1,
            "Message router: processing job",
        );

        let ws = match crate::users::resolve_workspace(&job.workspace_name).await {
            Ok(Some(ws)) => ws,
            Ok(None) => {
                error!(
                    agent_id = %agent_id,
                    workspace = %job.workspace_name,
                    "Message router: workspace not found — skipping job",
                );
                for item in &queued {
                    report_unrunnable_branch(item).await;
                }
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

        // The job's own workspace, plus the one relayed content originated in when it differs:
        // relative `[IMAGE:]`/`[AUDIO:]`/`[VIDEO:]` targets resolve against them and `[FILE:]`
        // is confined to them. A failed lookup only narrows the roots.
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

        let role = job.role;

        // Non-Manager roles deliver to the triggering user. Manager, Custom and Worker branch
        // answers relay to the admin's Assistant (or the asking branch) through their envelopes.
        let users: Vec<UserRecord> = if matches!(role, Role::Manager | Role::Custom | Role::Worker)
        {
            Vec::new()
        } else {
            match resolve_single_user(&job.user_name).await {
                Some(user) => vec![user],
                None => Vec::new(),
            }
        };

        let typing_tasks = setup_telegram_typing(&users).await;
        broadcast_typing(&users, &job.workspace_name, true);

        // Each item the turn answers becomes its own message, in queue order; the
        // job that ended the batch (the urgent one) is last.
        let message = match (role, job.kind) {
            (Role::Manager, MessageKind::UserMessage | MessageKind::AgentMessage) => {
                // Flush the CDC drainer so a transition committed moments ago is materialized
                // before the manager reads it (delayed, not lost, if skipped).
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
            _ => job.content.clone(),
        };
        let mut messages: Vec<String> = queued.iter().map(|item| item.content.clone()).collect();
        messages.push(message);

        // The admin gets the Assistant's widened toolset and role prompt; everyone else runs
        // the guest Assistant. Manager and custom-role jobs resolve no users, so `is_admin` is
        // false for both and nothing reads it — a custom role's strength is its tool subset.
        let is_admin = users
            .first()
            .is_some_and(|user| crate::users::is_admin_name(&user.name));
        // The Running Agents grouping: a branch launch is one conversation keyed by its own
        // agent id, labelled with its role and branch; everything else stays ungrouped.
        let (parent_key, parent_label) = match &job.custom_role {
            Some(launch) => (
                Some(crate::agent::registry::ParentKey::Launch(
                    launch.agent_id.clone(),
                )),
                Some(crate::launch::display_label(launch)),
            ),
            None => (None, None),
        };
        // The turn answers every message of the batch. Two things cut it: an announcement
        // (taken at staging, so it never waits for that message's own attachments) and an
        // urgent job handed straight to the queue — neither while stopping or draining. What
        // reaches the queue meanwhile is the next turn's, so the urgent one is answered last.
        let message_refs: Vec<&str> = messages.iter().map(String::as_str).collect();
        let mut queue_open = true;
        let mut turn = std::pin::pin!(crate::agent::run_agent(
            agent_id.clone(),
            role,
            &ws,
            None,
            &message_refs,
            job.user_name.clone(),
            job.channel.clone(),
            is_admin,
            None,
            false,
            None,
            parent_key,
            parent_label,
            job.custom_role.clone(),
        ));
        // `interrupt` is a no-op when the run has already concluded or has not registered
        // itself yet; that message is then answered by the next turn as usual.
        let (mut agent, response) = loop {
            tokio::select! {
                done = &mut turn => break done,
                pulled = rx.recv(), if queue_open => match pulled {
                    Some(next) => {
                        let urgent_job = next.urgent;
                        waiting.push_back(next);
                        if urgent_job && !crate::shutdown::aborting() {
                            crate::agent::registry::AGENT_REGISTRY.interrupt(&agent_id);
                        }
                    }
                    None => queue_open = false,
                },
                () = urgent.changed.notified(), if !crate::shutdown::aborting() => {
                    crate::agent::registry::AGENT_REGISTRY.interrupt(&agent_id);
                },
            }
        };

        for (cancel, handle) in typing_tasks {
            cancel.cancel();
            stop_typing(handle).await;
        }
        broadcast_typing(&users, &job.workspace_name, false);

        let Some(response) = response else {
            // A custom role's failure produces no answer: deliver the durable failure notice to
            // the admin's Assistant — never the emoji, never anything to the owner. A worker
            // branch's failure goes into the asking conversation, guarded the same way.
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
            // The emoji only for a UserMessage job whose agent truly failed; internal kinds get
            // no feedback. Both the agent token AND the global shutdown token are checked, since
            // during SIGTERM/SIGINT the global one fires first. A drained agent must NOT emit it.
            else if job.kind == MessageKind::UserMessage
                && !agent.is_cancelled()
                && !crate::shutdown::aborting()
            {
                deliver_unregistered_user_response(AGENT_FAILURE_EMOJI, &job, &role, &file_roots)
                    .await;
            }
            confirm_pending_delivery(&job).await;
            confirm_turn_delivery(&queued).await;
            continue;
        };

        // A turn that delivers nothing: a sleep ended it deliberately, or an urgent message
        // cut it — the phrase is in the session, the chat and every bound channel get nothing
        // (no empty bubble, no emoji). The at-least-once pending-jobs boundary is still
        // confirmed.
        if agent.sleep_ended || agent.turn_state.interrupted {
            confirm_turn_delivery(&queued).await;
            confirm_pending_delivery(&job).await;
            continue;
        }

        // Manager: a durable `<manager-message>` into the admin's Assistant, or back into the
        // asking branch. Custom role: `<custom-role-message>`. Worker branch: `<worker-report>`
        // into the asking conversation. Other roles: broadcast to the user's channel bindings,
        // with the unregistered-user fallback.
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
            // A Worker job reaches the consumer loop only as a worker branch (a plain worker
            // sub-agent is run by the dispatching tool, never routed), so the launch is always
            // there — reporting its absence rather than dropping the report silently.
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
        confirm_turn_delivery(&queued).await;
        confirm_pending_delivery(&job).await;

        // A long-lived chat role's session over a summary trigger compacts here, after this
        // reply is delivered and its pending row confirmed (a crash during the compaction then
        // cannot replay a delivered job), in the pause before the user's next message.
        agent.maybe_summarize_after_delivery().await;
    }
}

/// The launch of a worker-branch job: `Some` only when the job carries a launch whose role
/// is [`Role::Worker`]. A plain worker sub-agent never reaches the consumer loop.
fn worker_branch_launch(job: &AgentJob) -> Option<&crate::Launch> {
    job.custom_role
        .as_ref()
        .filter(|launch| launch.role == Role::Worker)
}

// Consumer-confirmed delivery: delete the durable pending_jobs row after the agent ran
// (at-least-once — created before routing, reclaimed only here or by boot replay), also on the
// refusal paths. One sync retry, then log-and-continue: duplicate re-delivery at next boot is
// bounded and accepted. Never called on workspace-not-found for a built-in role, so such a row
// re-routes and skips at every boot until the workspace returns; accepted.
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

/// Confirm the turn's earlier items too: each carries its own durable pending row (the same
/// at-least-once boundary), so a turn answering several messages reclaims them all.
async fn confirm_turn_delivery(queued: &[AgentJob]) {
    for item in queued {
        confirm_pending_delivery(item).await;
    }
}

// Report a branch job whose launch names a folder that is gone: reported to the admin's
// Assistant (or the asking branch), naming branch and folder, never to the owner or a guest
// — and the pending row is reclaimed with the notice, because the failure IS the delivery.
// Only the positively-gone case reports: a transient store failure keeps the plain skip, and
// a built-in role's row is deliberately kept so it re-routes when the workspace returns.
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

/// Set up Telegram typing indicators for the given users — one per unique Telegram chat.
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

/// Outcome of attempting to deliver a response on a single channel.
enum DeliverOutcome {
    Sent,
    /// Recipient is not reachable on this channel.
    Unresolvable,
    /// Transport send failed.
    Failed(anyhow::Error),
}

/// Resolve the recipient and send `response` on `channel`, returning the outcome so each
/// delivery function applies its own per-variant logging: resolve-miss handling and error
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

// The one spelling of the admin-pinning rule every relay-to-the-owner route shares: a job
// for the single admin's Assistant, in that admin's personal workspace, on `gui`, with no
// reply target and no branch. The admin is a constant identity, so no account-store hiccup can
// drop an envelope; the consumer re-validates the invariant.
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
        urgent: false,
    }
}

/// The content of a `<manager-message>` envelope for `response` produced in
/// `source_workspace`, or `None` for an empty response.
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

/// Auto-delivery hook: EVERY Manager response is additionally routed as a durable
/// `<manager-message>` envelope into the admin's Assistant session. `source_workspace` rides in
/// the job's `originating_workspace` and in the envelope's `workspace` attribute, so a marker in
/// the response stays authorized. Empty responses deliver nothing.
async fn route_manager_notify(response: &str, source_workspace: &str) {
    let Some(content) = manager_message_content(response, source_workspace) else {
        return;
    };
    let job = admin_assistant_job(content, MessageKind::ManagerNotify, Some(source_workspace));
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "manager notify envelope").await;
}

/// A job delivered into the conversation that asked for it instead of the admin's
/// Assistant: the destination comes from `asker`, the envelope from `kind`.
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
        urgent: false,
    }
}

/// Deliver a Manager's answer back into the custom-role branch that asked for it — the
/// mirror image of the analyze and worker result envelopes: the job carries the branch
/// address and the folder the asking turn ran in, tagged `<manager-message>` because that is
/// the shape `send`'s `manager` addressee promises its caller.
async fn route_manager_reply_to_custom(response: &str, source_workspace: &str, asker: &Asker) {
    let Some(content) = manager_message_content(response, source_workspace) else {
        return;
    };
    let job = branch_job(content, asker, MessageKind::ManagerNotify);
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "manager reply envelope").await;
}

/// Deliver a finished custom role's answer into the admin Assistant's session (the durable
/// `<custom-role-message>` envelope). `source_workspace` travels in `originating_workspace`,
/// so a file/media reference in the answer is validated against the folder it was produced
/// in; `identifier` names the branch when the launch ran on one. An empty answer delivers nothing.
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

/// Deliver the notice that a custom role's launch failed, naming the role, its branch when
/// it ran on one, and the folder it ran in. Nothing goes to the owner directly.
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

/// Deliver a worker branch's finished turn into the conversation that asked for it: the asking
/// custom role's branch when the launch carries an asker, else the admin Assistant's session.
/// Both carry the `<worker-report>` envelope, naming the folder the turn ran in (its resolved
/// PATH, not the branch's durable address) and its branch; an empty answer delivers nothing.
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

/// Deliver the notice that a worker branch's turn produced no answer, into the same
/// destination as its report. `folder` is the same resolved path the report names;
/// [`report_unrunnable_branch`] passes the branch's durable address, since no workspace was
/// resolved to run in.
async fn route_worker_branch_failure(launch: &crate::Launch, folder: &str) {
    let content = crate::tools::build_worker_envelope(
        "worker_failure.md",
        folder,
        launch.identifier.as_deref(),
        "",
    );
    route_worker_branch_envelope(content, launch.reply_to.as_deref()).await;
}

/// Route a worker branch's envelope: into the asking custom role's branch when the caller
/// carried one (mirroring [`route_manager_reply_to_custom`]), else into the admin's Assistant
/// session. Neither arm sets `originating_workspace`: a worker report must carry exactly the
/// file-delivery roots the single-run worker's report carries — its own workspace alone.
async fn route_worker_branch_envelope(content: String, asker: Option<&Asker>) {
    let job = match asker {
        Some(asker) => branch_job(content, asker, MessageKind::ImplementResult),
        None => admin_assistant_job(content, MessageKind::ImplementResult, None),
    };
    let target = crate::jobs::envelope_target(&job);
    stamp_and_route(job, &target, "worker branch report envelope").await;
}

/// Transport-deliver `response` to every channel binding of `users` (broadcast and
/// chat_history persistence happen at the call sites). Logs an unresolvable recipient (warn —
/// persisted but not transport-delivered) and a transport failure (error) per binding.
/// `workspace` names the workspace in the no-channels diagnostic.
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

/// Deliver a response to a single registered user, broadcasting it to ALL of the user's
/// channel bindings through [`deliver_response_over_channels`], which walks every registered
/// channel with that binding's own reply_target, so a user bound on several channels sees the
/// reply on all of them. `job.reply_target` does not scope delivery for registered users.
async fn deliver_single_user_response(
    response: &str,
    user: &UserRecord,
    job: &AgentJob,
    role: &Role,
    file_roots: &[std::path::PathBuf],
) {
    // Broadcast + persist, tagged with the originating channel and a fresh dispatch tag.
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

/// Deliver a response to an unregistered user (no [`UserRecord`] in the users DB), falling
/// back to the originating channel with `job.reply_target` (or `job.user_name`) as the reply
/// target, so the original message's target (e.g. a Telegram chat_id) is preserved. Also used
/// by the binary for inline confirmations via raw `reply_target` passthrough.
///
/// Broadcast + persisted first, so it reaches the GUI chat history even if the transport
/// fails. `file_roots` anchors and confines media markers as in
/// [`deliver_single_user_response`]; an empty slice names no workspace, so no relative marker
/// and no `[FILE:...]` is authorized — what the inline-confirmation caller passes.
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

    // Broadcast + persist (strings only, no UserRecord needed), tagged with the originating
    // channel and a fresh dispatch tag on the row.
    broadcast_and_persist_agent_response(
        user_name,
        ch,
        response,
        role.as_str().to_string(),
        &job.workspace_name,
        Some(crate::generate_id()),
    )
    .await;

    let Some(chan) = crate::channel_registry().get(ch) else {
        return;
    };

    // reply_target from the original message when available (e.g. Telegram chat_id).
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

/// Resolve a single user by name with a targeted query instead of loading all users.
async fn resolve_single_user(user_name: &str) -> Option<UserRecord> {
    let store = crate::users::USER_STORE.get()?;
    match store.find_by_name(user_name).await {
        Ok(Some(user)) => Some(user),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;
    use crate::channels::GuiChannel;
    use std::sync::Arc;

    /// An urgent message never overtakes what already waited, nor pulls a later arrival.
    #[test]
    fn a_turn_ends_with_its_urgent_message_and_leaves_later_arrivals() {
        let job = |content: &str, urgent: bool| {
            let mut job = make_job(Role::Assistant, "ws", "alice", "gui");
            job.content = content.to_string();
            job.urgent = urgent;
            job
        };
        let contents = |batch: &[AgentJob]| -> Vec<String> {
            batch.iter().map(|job| job.content.clone()).collect()
        };

        let mut waiting: std::collections::VecDeque<_> =
            [job("one", false), job("two", false)].into();
        assert_eq!(contents(&take_turn_batch(&mut waiting)), ["one"]);
        assert_eq!(waiting.len(), 1, "the rest keeps its turn");

        let mut waiting: std::collections::VecDeque<_> = [
            job("one", false),
            job("two", false),
            job("urgent", true),
            job("later", false),
        ]
        .into();
        let batch = take_turn_batch(&mut waiting);
        assert_eq!(contents(&batch), ["one", "two", "urgent"]);
        assert_eq!(waiting.len(), 1, "a later arrival is not pulled forward");

        let mut waiting: std::collections::VecDeque<_> = [job("urgent", true)].into();
        assert_eq!(contents(&take_turn_batch(&mut waiting)), ["urgent"]);
        assert!(waiting.is_empty());

        let mut waiting: std::collections::VecDeque<_> = [
            job("urgent-a", true),
            job("urgent-b", true),
            job("later", false),
        ]
        .into();
        assert_eq!(contents(&take_turn_batch(&mut waiting)), ["urgent-a"]);
        assert_eq!(contents(&take_turn_batch(&mut waiting)), ["urgent-b"]);
        assert_eq!(contents(&take_turn_batch(&mut waiting)), ["later"]);
    }

    #[test]
    fn an_announcement_is_waited_for_once() {
        let arrivals = UrgentArrivals::default();
        assert!(arrivals.take().is_empty());
        let announcement = arrivals.announce();
        let second = arrivals.announce();
        assert_eq!(
            arrivals.take().len(),
            2,
            "both announcements are outstanding"
        );
        assert!(
            arrivals.take().is_empty(),
            "a taken announcement is not waited for again"
        );
        drop(announcement);
        drop(second);
    }

    #[tokio::test]
    #[serial_test::serial(drain)] // the hold observes the process-global drain flag
    async fn a_pending_announcement_holds_the_turn_for_its_message() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let urgent = UrgentArrivals::default();
        let announcement = urgent.announce();
        let sender = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let _ = tx.send(job_with_content("ordinary", false));
            let _ = tx.send(job_with_content("urgent", true));
            drop(announcement);
        });

        let mut waiting = std::collections::VecDeque::new();
        let kept = hold_for_urgent(
            &mut waiting,
            &urgent,
            &mut rx,
            &CancellationToken::new(),
            "test",
            std::time::Duration::from_secs(30),
        )
        .await;
        sender.await.expect("the sender must not panic");

        assert!(kept, "the hold ended on its own message, not on a stop");
        let contents: Vec<&str> = waiting.iter().map(|job| job.content.as_str()).collect();
        assert_eq!(contents, ["ordinary", "urgent"]);
    }

    /// A hand-over that never arrives is given up after its window; a staging task that is
    /// gone ends the wait at once.
    #[tokio::test]
    #[serial_test::serial(drain)] // the hold observes the process-global drain flag
    async fn an_announcement_that_never_arrives_is_given_up() {
        let (_tx, mut rx) = mpsc::unbounded_channel();
        let urgent = UrgentArrivals::default();
        let shutdown = CancellationToken::new();
        let hold = std::time::Duration::from_millis(50);
        let mut waiting = std::collections::VecDeque::new();

        let _lost = urgent.announce();
        let started = std::time::Instant::now();
        assert!(hold_for_urgent(&mut waiting, &urgent, &mut rx, &shutdown, "test", hold).await);
        assert!(
            started.elapsed() >= hold,
            "the announced message had its whole window"
        );
        assert!(waiting.is_empty());
        assert!(
            urgent.take().is_empty(),
            "the wait gave the announcement up"
        );

        // The staging task is gone before the hand-over: nothing is coming.
        let gone = urgent.announce();
        drop(gone);
        let started = std::time::Instant::now();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                hold_for_urgent(
                    &mut waiting,
                    &urgent,
                    &mut rx,
                    &shutdown,
                    "test",
                    std::time::Duration::from_secs(600)
                )
            )
            .await
            .expect("a hand-over that will never arrive must not hold the queue")
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "a hand-over that will never arrive must not hold the queue"
        );
        assert!(urgent.take().is_empty(), "and nothing is left outstanding");
    }

    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_drain_ends_the_urgent_hold() {
        let (_tx, mut rx) = mpsc::unbounded_channel();
        let urgent = UrgentArrivals::default();
        let _announcement = urgent.announce();
        let mut waiting = std::collections::VecDeque::new();

        crate::shutdown::drain_begin();
        let kept = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            hold_for_urgent(
                &mut waiting,
                &urgent,
                &mut rx,
                &CancellationToken::new(),
                "test",
                std::time::Duration::from_secs(600),
            ),
        )
        .await
        .expect("a drain ends the hold");
        crate::shutdown::drain_clear();

        assert!(!kept, "a drain stops the consumer, so nothing holds it");
    }

    /// An announcement lands on the addressed agent's consumer only.
    #[tokio::test]
    async fn announce_urgent_reaches_the_addressed_consumer() {
        let _ = init_global();
        let agent_id = format!("_test_announce_{}", crate::generate_suffix());
        let _rx = register_agent(&agent_id);
        let other = format!("_test_announce_other_{}", crate::generate_suffix());
        let _other_rx = register_agent(&other);

        let announcement = announce_urgent(&agent_id).expect("the consumer is running");

        let arrivals = |id: &str| -> Arc<UrgentArrivals> {
            Arc::clone(
                &ROUTER
                    .get()
                    .expect("router initialised")
                    .read()
                    .unwrap_poison()
                    .get(id)
                    .expect("consumer registered")
                    .urgent,
            )
        };
        assert_eq!(
            arrivals(&agent_id).take().len(),
            1,
            "the addressed consumer waits for the message"
        );
        assert!(
            arrivals(&other).take().is_empty(),
            "another agent's consumer is untouched"
        );

        // No consumer: the next turn started for that agent answers it; not an error.
        assert!(
            announce_urgent(&format!(
                "_test_announce_absent_{}",
                crate::generate_suffix()
            ))
            .is_none()
        );

        drop(announcement);
        unregister_agent(&agent_id);
        unregister_agent(&other);
    }

    fn job_with_content(content: &str, urgent: bool) -> AgentJob {
        AgentJob {
            content: content.to_string(),
            urgent,
            ..make_job(Role::Assistant, "ws", "alice", "gui")
        }
    }

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
            urgent: false,
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

        let map = ROUTER.get().unwrap();
        let mut guard = map.write().unwrap_poison();
        guard.remove(id);
    }

    /// The consumer exits when the channel's last sender is dropped, and its cleanup
    /// wrapper removes the router entry: a dummy sender is re-inserted after the drop so
    /// the removal proves the wrapper ran.
    #[tokio::test]
    async fn test_consumer_loop_exits_gracefully_on_sender_drop() {
        let _ = init_global();
        let id = "_test_ua_on_close";

        route(id, make_job(Role::Assistant, "", "", ""));

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

        // A dummy sender for the cleanup wrapper to remove once the consumer exits.
        let (dummy_tx, _) = mpsc::unbounded_channel::<AgentJob>();
        ROUTER.get().unwrap().write().unwrap_poison().insert(
            id.to_string(),
            Consumer {
                jobs: dummy_tx,
                urgent: Arc::new(UrgentArrivals::default()),
            },
        );

        // Poll until the consumer exits and the wrapper removes the dummy entry.
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

    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_workspace_personal() {
        crate::util::test::init_management_test_stores().await;

        let result = crate::users::resolve_workspace("personal:liliana").await;
        let resolved = result.expect("resolve should succeed for personal workspace");
        let ws = resolved.expect("personal workspace should be constructed on the fly");

        assert_eq!(ws.name, "personal:liliana");
        assert_eq!(ws.status, crate::WorkspaceStatus::Ready);
        let expected_path = crate::users::personal_workspace_path("liliana");
        assert_eq!(ws.path, expected_path);
    }

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

    /// A path that is a managed workspace's own folder resolves to that workspace, not to a
    /// path-named area over the same tree, so a launch by path keeps its notes, pause state
    /// and search engine.
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

    /// Unique to these delivery tests, so the spy never collides with a real "telegram"
    /// channel a parallel test (e.g. telegram_tests) may have registered.
    const TEST_SPY_CHANNEL: &str = "__test_spy_channel";

    /// Set up DB stores + channel registry for response-delivery tests.
    async fn setup_response_test_infra() {
        crate::util::test::init_management_test_stores().await;
        let _ = crate::CHANNEL_REGISTRY.set(crate::ChannelRegistry::default());
        let (gui_channel, _gui_tx) = GuiChannel::new();
        crate::channel_registry().register(Arc::new(gui_channel));
    }

    // The broadcast outcome is asserted through the spy channel below: the real transports
    // are no-ops at this level, so a path that only "completes without panic" proves nothing.

    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_single_user_found() {
        setup_response_test_infra().await;

        // The admin is auto-created by ensure_admin_user.
        let user = resolve_single_user("admin").await;
        assert!(user.is_some(), "the admin should exist after store init");
        assert_eq!(user.as_ref().unwrap().name, "admin");
    }

    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_resolve_single_user_not_found() {
        setup_response_test_infra().await;

        let user = resolve_single_user("nonexistent_user").await;
        assert!(user.is_none(), "non-existent user should return None");
    }

    /// The reply reaches every channel the user is bound to, whatever channel the job
    /// originated on.
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
            urgent: false,
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

    /// A Manager response persists one durable `<manager-message>` envelope for the admin's
    /// personal Assistant, carrying the FULL untruncated response with the source workspace
    /// in `originating_workspace` and in the envelope attribute. Empty responses persist
    /// nothing. Cleans up the row it created so boot replay never sees it.
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
        // Scoped to the rows this test created: the global store is written by parallel
        // tests too, so a bare count for the target session would include other tests' rows.
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

    #[tokio::test]
    async fn test_try_route_receiver_dropped() {
        let _ = init_global();
        let agent_id = "_test_try_route_dropped";

        let rx = register_agent(agent_id);
        drop(rx);

        let job = make_job(Role::Assistant, "hello", "user", "gui");
        assert!(
            !try_route(agent_id, job),
            "try_route should return false when receiver is dropped",
        );

        unregister_agent(agent_id);
    }

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

    /// A custom role's answer is an internal envelope for the admin's personal Assistant —
    /// never the owner's channels — persisted as one durable `<custom-role-message>` row
    /// naming the folder the launch ran in, so a marker in it can be relayed out. Cleans up
    /// the row it created.
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

    /// The `<worker-report>` envelope names the folder PATH the turn ran in (never the
    /// branch's durable address) and its `branch` when named, and the delivered job carries
    /// NO `originating_workspace` — no `[FILE:...]` root the single run lacks.
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

        // A folder name carrying a quote or angle bracket cannot deform the envelope: the
        // three characters that could end the attribute are escaped.
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

        // An asking custom role's branch: same shape, no extra root. The asker travels on
        // the LAUNCH, its single carrier, which is what the report reads.
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

    /// A user with no row still gets the reply: sent through the originating channel and
    /// persisted under the seeded user, so an unregistered sender is never silently dropped.
    #[tokio::test]
    #[serial_test::serial(provider)]
    async fn test_deliver_unregistered_user_response() {
        // Its own channel name: the registry is process-global, so sharing the
        // other delivery test's spy would route the reply to whichever spy
        // registered last.
        let spy_name = "__test_spy_unregistered_user";
        setup_response_test_infra().await;
        let (spy, sent) = crate::util::test::SpyChannel::new(spy_name);
        crate::channel_registry().register(Arc::new(spy) as Arc<dyn crate::Channel>);

        let job = AgentJob {
            content: "hello from unregistered user".to_string(),
            workspace_name: "default".to_string(),
            user_name: "unregistered_alice".to_string(),
            channel: spy_name.to_string(),
            kind: MessageKind::UserMessage,
            role: Role::Assistant,
            reply_target: Some("chat_123".to_string()),
            pending_job_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
        };

        deliver_unregistered_user_response(
            "response to unregistered user",
            &job,
            &Role::Assistant,
            &[],
        )
        .await;

        let captured = sent.lock().unwrap_poison();
        assert!(
            captured.iter().any(|msg| msg.recipient == "chat_123"
                && msg.content.contains("response to unregistered user")),
            "the reply must reach the originating channel's reply target, got: {captured:?}"
        );
    }
}
