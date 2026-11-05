//! One agent's turn: from a routed job to a finalized session through every tool
//! round in between, plus the role/toolset metadata a turn is built from
//! (`run_agent`, `run_default_agent`). The submodules own what a running agent
//! reads and publishes (`role`, `registry`, `message_router`, `skills`,
//! `extraction`, `maintainer`) — not the conversation store of `crate::session`.

use std::fmt::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use tracing::Instrument;

use crate::providers::reasoning::assistant_replay_payload;
use crate::session::Session;
use crate::tools::{
    ImagePayload, ToolExecutionOutcome, find_tool, format_tool_failure_feedback,
    normalize_tool_call, scrub_tool_output,
};
use crate::util::{
    MEDIA_MARKER_RE, MediaMarkerKind, UnwrapPoison, parse_media_marker, scrub_credentials,
};
use crate::{Agent, ChatMessage, ChatRequest, ChatResponse, Tool, ToolCall};

pub(crate) mod extraction;
pub mod maintainer;
pub mod message_router;
pub mod registry;
pub(crate) mod role;
pub(crate) mod skills;

// Set once per tool batch (per `execute_tool_group` round), not per individual
// tool `execute()`, and read synchronously — before any `tokio::spawn` boundary
// when the value must cross one. Background agents (management, maintainer,
// workspace) fall back to an empty string.
/// Synchronous read of [`CURRENT_TOOL_USER_NAME`]; the task-local contract above applies.
pub(crate) fn tool_user_name() -> String {
    CURRENT_TOOL_USER_NAME
        .try_with(String::clone)
        .unwrap_or_default()
}

/// The calling agent's session id alone — the identity a session-scoped action
/// needs when the user is not part of it: a Manager session driven by pipeline
/// ticket notifications carries no user name, so [`tool_identity`] refuses those
/// turns while this one answers them.
pub(crate) fn tool_agent_id() -> anyhow::Result<String> {
    current_agent_id()
        .context("No agent identity available — the tool must run in an agent session.")
}

/// Identity of the agent executing a tool call: `(agent_id, user_name)`. Errors
/// outside a user session — there is no addressable identity to attribute side
/// effects to.
pub(crate) fn tool_identity() -> anyhow::Result<(String, String)> {
    let user_name = tool_user_name();
    let agent_id = current_agent_id()
        .filter(|_| !user_name.is_empty())
        .context("No agent identity available — the tool must run in a user session.")?;
    Ok((agent_id, user_name))
}

/// The calling agent's session id as the task-local carries it: `None` outside an
/// agent run and for the empty id a turn can carry.
pub(crate) fn current_agent_id() -> Option<String> {
    CURRENT_TOOL_AGENT_ID
        .try_with(Option::clone)
        .ok()
        .flatten()
        .filter(|agent_id| !agent_id.is_empty())
}

/// The scope a tool-call row states; the ticket is empty when the run has none.
#[derive(Default)]
pub(crate) struct ToolRecordAttribution {
    pub agent_id: String,
    pub agent_role: String,
    pub ticket: String,
}

/// The attribution [`crate::logs::record_stray_event`] rows carry: empty outside an
/// agent run, ticket empty when the run has none. Read here, synchronously — no
/// task-local is in scope where the row is written.
pub(crate) fn tool_record_attribution() -> ToolRecordAttribution {
    let identity = CURRENT_TOOL_AGENT_TRACKING
        .try_with(|tracking| {
            tracking
                .as_ref()
                .map(|tracking| (tracking.role.clone(), tracking.ticket.clone()))
        })
        .ok()
        .flatten();
    let Some((agent_role, ticket)) = identity else {
        return ToolRecordAttribution::default();
    };
    ToolRecordAttribution {
        agent_id: CURRENT_TOOL_AGENT_ID
            .try_with(Option::clone)
            .unwrap_or_default()
            .unwrap_or_default(),
        agent_role,
        ticket,
    }
}

/// Synchronous read of [`CURRENT_TOOL_CHANNEL`]. Same contract as [`tool_user_name`].
pub(crate) fn tool_channel() -> String {
    CURRENT_TOOL_CHANNEL
        .try_with(String::clone)
        .unwrap_or_default()
}

/// Synchronous read of [`CURRENT_TOOL_COMMAND_LINE`]; false outside that call.
pub(crate) fn tool_command_line() -> bool {
    CURRENT_TOOL_COMMAND_LINE
        .try_with(|flag| *flag)
        .unwrap_or(false)
}

/// Synchronous read of [`CURRENT_TOOL_CUSTOM_LAUNCH`]: a tool that dispatches a
/// durable round must read it before its `tokio::spawn` boundary so the round
/// carries the caller's branch.
pub(crate) fn tool_custom_launch() -> Option<crate::Launch> {
    CURRENT_TOOL_CUSTOM_LAUNCH
        .try_with(std::clone::Clone::clone)
        .unwrap_or(None)
}

/// Synchronous read of [`CURRENT_TOOL_OUTPUT_DIR`]: where a command-line call's
/// produced files must land, `None` for every other caller.
pub(crate) fn tool_output_dir() -> Option<std::path::PathBuf> {
    CURRENT_TOOL_OUTPUT_DIR
        .try_with(std::clone::Clone::clone)
        .unwrap_or(None)
}

tokio::task_local! {
    pub(crate) static CURRENT_TOOL_USER_NAME: String;
    pub(crate) static CURRENT_TOOL_CHANNEL: String;
    /// Whether THIS call is the command line's own — `mahbot send …` (see
    /// [`crate::cli`]) in the instance with no agent session behind it. Set ONLY
    /// by [`crate::cli::instance`]; never inferred from values a call carries,
    /// since other executors carry the same ones (a message handed to an agent
    /// runs that agent's turn with the command line's channel like any other).
    pub(crate) static CURRENT_TOOL_COMMAND_LINE: bool;
    /// The caller's DIRECT PARENT INVOCATION key (ticket / analyze round /
    /// research run), so tools that spawn sub-agents (`send`) land in the caller's
    /// Running Agents group. `None` = workspace singleton caller.
    pub(crate) static CURRENT_TOOL_PARENT_KEY: Option<crate::agent::registry::ParentKey>;
    /// The caller's parent invocation label (ticket title / question), set
    /// alongside [`CURRENT_TOOL_PARENT_KEY`] for spawned sub-agents. Purely
    /// presentational; `None` for workspace singletons.
    pub(crate) static CURRENT_TOOL_PARENT_LABEL: Option<String>;
    /// The caller's background shell session registry, so the shell tool (Full
    /// roles only) registers/stops sessions that are force-killed when the agent
    /// drops. `None` outside an agent run — background mode is unavailable there.
    pub(crate) static CURRENT_TOOL_BACKGROUND_SESSIONS:
        Option<std::sync::Arc<crate::tools::shell::BackgroundSessions>>;
    /// The caller's id, so the shell tool attributes its spill files to the owning
    /// agent (deleted when the run ends). `None` outside an agent run.
    pub(crate) static CURRENT_TOOL_AGENT_ID: Option<String>;
    /// The caller's role and ticket, for the record a tool makes about itself — a
    /// stray event's log row. The agent id is `CURRENT_TOOL_AGENT_ID`'s. `None`
    /// outside an agent run.
    pub(crate) static CURRENT_TOOL_AGENT_TRACKING:
        Option<crate::agent::registry::AgentTracking>;
    /// The caller's custom-role launch, so a tool dispatching a durable round
    /// (`analyze`/`send`) carries its branch address into the envelope and job
    /// row. `None` for every built-in role.
    pub(crate) static CURRENT_TOOL_CUSTOM_LAUNCH: Option<crate::Launch>;
    /// Where a command-line call's produced files must land — set ONLY by
    /// [`crate::cli::instance`], because the tool's arguments never carry it.
    /// `None` (the whole agent path) means the tool's own workspace directory.
    pub(crate) static CURRENT_TOOL_OUTPUT_DIR: Option<std::path::PathBuf>;
}

/// Maximum number of completed tool rounds before the agent loop bails out.
///
/// A tool round is the LLM call(s) that returned tool calls plus execution and
/// commit of that group. The counter increments only after a committed group; the
/// final-answer call and the image-rejection strip `continue` do not, so an agent
/// can legitimately make more LLM calls than this cap. Do not "fix" these
/// semantics while touching this file.
///
/// **DO NOT REDUCE without benchmarking against real ticket tool-round
/// distributions.** 1000 is deliberate: legitimate work consumes hundreds of
/// rounds, a full run costs well under $1, and hitting the cap is EXTREMELY rare
/// — it is a safety net, not a common guard. Lowering it fails long-running
/// agents mid-task, wasting more on restarts than the rounds would have cost.
/// Value last intentionally reviewed: 2026-07-11
const MAX_TOOL_ROUNDS: usize = 1000;

/// Retry-exhaustion marker shared by `llm_call`/`summarize` and the
/// `engineer_failure_notice` match; drift is cosmetic (classification uses the
/// typed `RetryExhausted`).
pub(crate) const RETRY_EXHAUSTION_MARKER: &str = "exhausted retry budget";

/// Attempts for the summarization call when the model answers with a tool call
/// instead of a summary — re-sent verbatim, so attempts stay byte-identical
/// (prompt-prefix cache preserved) and the call is never executed. On exhaustion
/// the compaction is abandoned exactly as before: full history kept, owner not disturbed.
const SUMMARIZE_ATTEMPTS: u32 = 3;

/// The interrupted turn's whole entry in the session — the only place the interruption is
/// recorded. The chat and every bound channel get nothing: the turn simply ends, and the
/// urgent message is answered as the next turn.
const URGENT_INTERRUPT_ENTRY: &str = "interrupted by the user";

/// The result a call the urgent interrupt cut is settled with: an ordinary tool
/// result — not a failure — claiming no knowledge of the call's effect.
fn interrupted_outcome() -> ToolExecutionOutcome {
    ToolExecutionOutcome {
        output: crate::prompt::load_prompt("tool_interrupted.md")
            .trim()
            .to_string(),
        success: true,
        image_payloads: Vec::new(),
        text_is_content: false,
        preserve_text: false,
        suspended: false,
        ends_turn: false,
    }
}

/// Boxed future returned by [`Agent::complete_pending_tool_calls`]: naming the box
/// breaks the async opaque-type Send cycle a concrete `impl Future` would create
/// between `agent` and `tools::analyze` (resume cores re-enter sub-agent runs).
type CompletePendingToolCallsFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<bool>> + Send + 'a>>;

/// `(marker_prefix, path)` for every successful media tool outcome carrying its
/// [`Tool::media_marker`] (e.g. `[IMAGE:path]`); failed outcomes and non-media
/// tools are skipped. Limitation: a path containing `]` is truncated by the regex
/// — harmless, media tools write safe temporary names.
fn extract_media_from_outcomes(
    tools: &[Box<dyn Tool>],
    tool_calls: &[ToolCall],
    outcomes: &[ToolExecutionOutcome],
) -> Vec<(&'static str, String)> {
    let mut paths = Vec::new();
    for (call, outcome) in tool_calls.iter().zip(outcomes.iter()) {
        if !outcome.success {
            continue;
        }
        let Some(tool) = find_tool(tools, &call.name) else {
            continue;
        };
        let Some(marker_prefix) = tool.media_marker() else {
            continue;
        };
        // "[IMAGE:" -> "IMAGE", per `media_marker()`'s documented `[KIND:` shape.
        let marker = &marker_prefix[1..marker_prefix.len() - 1];
        let mut matched = false;
        for caps in MEDIA_MARKER_RE.captures_iter(&outcome.output) {
            let (captured_kind, path) = parse_media_marker(&caps);
            if captured_kind.token() == marker {
                matched = true;
                paths.push((marker_prefix, path.to_string()));
            }
        }
        if matched {
            continue;
        }
        // A marker-always tool ([`Tool::media_marker_always`]) that carried none is
        // a defect to see; a tool that also answers with text alone is normal here.
        let message = "a successful media call carried no media marker";
        if tool.media_marker_always() {
            tracing::warn!(
                media_tool = %call.name,
                marker = %marker_prefix,
                "{message}",
            );
        } else {
            tracing::debug!(
                media_tool = %call.name,
                marker = %marker_prefix,
                "{message}",
            );
        }
    }
    paths
}

/// Count DISTINCT data-URI image markers in the session's user messages — the
/// second summarization trigger ([`crate::session::SUMMARIZATION_IMAGE_COUNT`]).
/// The `data:image/` pre-gate keeps the per-turn scan cheap and never undercounts
/// (every vision-promoted marker carries that prefix); it is a deliberate
/// superset, so an invalid `data:image/*` payload may be counted too.
fn session_image_marker_count(history: &[ChatMessage]) -> usize {
    crate::session::image_cap::user_image_values(history)
        .filter(|value| value.starts_with("data:image/"))
        .collect::<std::collections::HashSet<_>>()
        .len()
}

/// An [`ImagePayload`] derived from a real on-disk `[IMAGE:path]` marker in a
/// successful tool output that supplied none, so a marker-delivered image reaches
/// the agent's own context. Fails open (`None`) for a non-absolute or
/// non-compressible path (e.g. an SVG); the marker still delivers the file.
///
/// Deliberately not routed through the shared classifier: as a *producer* it must
/// emit a bounded compressed payload, so the async full decode + resize + JPEG
/// (`local_image_to_compressed_data_uri_with_meta`) is its authoritative check;
/// the classifier would add a redundant decode and no encoding value.
async fn derive_image_payload_from_marker(output: &str) -> Option<ImagePayload> {
    // Cheap guard: skip the regex scan for outputs carrying no image marker.
    if !crate::util::has_media_marker(output, MediaMarkerKind::Image) {
        return None;
    }
    for caps in MEDIA_MARKER_RE.captures_iter(output) {
        let (kind, path) = parse_media_marker(&caps);
        if kind != MediaMarkerKind::Image {
            continue;
        }
        let p = Path::new(path);
        if !p.is_absolute() {
            continue;
        }
        let Ok(meta) = crate::util::local_image_to_compressed_data_uri_with_meta(p).await else {
            // Fail open for this marker (missing file, non-raster e.g. SVG,
            // over-cap input). The marker stays in the raw output for delivery.
            continue;
        };
        return Some(ImagePayload::from_compressed_meta(
            p,
            meta,
            None,
            crate::tools::ImagePayloadSource::Generated,
        ));
    }
    None
}

/// Format a tool's outcome for persistence and derive its fresh image payloads —
/// the shared "outcome → content" step of [`Agent::commit_tool_results`] and
/// resume-completion re-execution.
///
/// [`ToolExecutionOutcome::text_is_content`]: content keeps the annotations as an
/// appendix and its markers literal; a claim is replaced by them. `format_output`
/// (or the truncation fallback) runs first, then the outcome's own payloads take
/// precedence over one derived from a real on-disk `[IMAGE:path]` marker. The dedup
/// key `seen` (lazily the pictures `history` would still send) decides fresh/attached,
/// so a picture the request cap retired is attached again instead of being reported as
/// present. The returned vec is every FRESH attachment, injected as ONE synthetic user
/// message. Payloads are taken, not copied — they carry base64 data URIs.
async fn tool_result_content(
    tools: &[Box<dyn Tool>],
    call_name: &str,
    outcome: ToolExecutionOutcome,
    seen: &mut Option<std::collections::HashSet<String>>,
    history: &[ChatMessage],
) -> (String, Vec<ImagePayload>) {
    let formatted = if outcome.preserve_text {
        outcome.output.clone()
    } else {
        match find_tool(tools, call_name) {
            Some(t) => t.format_output(&outcome.output),
            None => crate::util::truncate_tool_output(&outcome.output),
        }
    };

    // The outcome's own payloads (read, image_gen) take precedence; otherwise derive one
    // from a real on-disk `[IMAGE:path]` marker, but only from a claim — in content (a
    // converted document's text) a marker is text. Fails open: a non-rasterizable path
    // yields no payload while the marker stays in the raw output.
    let mut payloads: Vec<ImagePayload> = outcome.image_payloads;
    if payloads.is_empty() && !outcome.text_is_content && outcome.success {
        payloads.extend(derive_image_payload_from_marker(&outcome.output).await);
    }
    if payloads.is_empty() {
        return (formatted, Vec::new());
    }

    let seen = seen.get_or_insert_with(|| crate::session::image_cap::sent_picture_values(history));
    let mut fresh = Vec::with_capacity(payloads.len());
    let mut annotations = Vec::with_capacity(payloads.len());
    for payload in payloads {
        if seen.insert(payload.data_uri.clone()) {
            annotations.push(payload.attached_annotation());
            fresh.push(payload);
        } else {
            annotations.push(payload.already_attached_annotation());
        }
    }
    let annotation_block = annotations.join("\n");
    let output = if outcome.text_is_content {
        format!("{formatted}\n{annotation_block}")
    } else {
        annotation_block
    };
    (output, fresh)
}

/// The single synthetic User-role message carrying a round's fresh tool images;
/// `None` when the round attached nothing.
///
/// A video's pictures are ordinary images: attached and stored like any other.
fn injected_images_message(
    agent_id: &str,
    role: crate::Role,
    payloads: Vec<ImagePayload>,
) -> Option<ChatMessage> {
    if payloads.is_empty() {
        return None;
    }
    let mut markers = Vec::with_capacity(payloads.len());
    for payload in payloads {
        tracing::debug!(
            agent_id,
            %role,
            path = %payload.path,
            "Injecting tool image as a synthetic user message"
        );
        markers.push(format!("[IMAGE:{}]", payload.data_uri));
    }
    Some(ChatMessage::user(crate::util::injected_image_user_message(
        &markers,
    )))
}

/// One-pass derivation of a role's advertised tools and their specs — the single
/// source for [`Agent::new`] and [`role_tool_specs`], so the frozen replay can never
/// drift from the live tools (KV-cache byte identity). The caller owns the run-scoped
/// chrome tracker, released at the run's end. `launch` is a branch role's launch (its
/// record's tool subset); `None` for built-ins.
#[must_use]
pub(crate) fn role_tools_and_specs(
    role: crate::Role,
    ws: &crate::Workspace,
    is_admin: bool,
    chrome_sessions: std::sync::Arc<crate::tools::chrome::ChromeRunSessions>,
    launch: Option<&crate::Launch>,
) -> (Vec<Box<dyn Tool>>, Vec<crate::ToolSpec>) {
    // A custom role's set is its record's chosen subset of the admin Assistant's
    // tools; every other role keeps the static per-role factory.
    let tools: Vec<Box<dyn Tool>> = if role == crate::Role::Custom {
        crate::Role::custom_role_tools(launch.map(|c| c.tools.as_slice()).unwrap_or_default())
    } else {
        role.tools(ws, is_admin, chrome_sessions)
    }
    .into_iter()
    .filter(|t| t.is_advertised())
    .collect();
    let tool_specs = tools.iter().map(|t| t.spec()).collect();
    (tools, tool_specs)
}

/// The advertised tool specs of one role without a run: the tools get a tracker
/// with no namespace (`ChromeRunSessions::default`), which never claims a run's.
#[must_use]
pub(crate) fn role_tool_specs(
    role: crate::Role,
    ws: &crate::Workspace,
    is_admin: bool,
) -> Vec<crate::ToolSpec> {
    role_tools_and_specs(
        role,
        ws,
        is_admin,
        std::sync::Arc::new(crate::tools::chrome::ChromeRunSessions::default()),
        None,
    )
    .1
}

/// A role's derived chat-request triplet (model → provider routing → reasoning
/// effort); named fields keep the order unambiguous for cross-module callers.
pub(crate) struct RoleChatParams {
    pub model: String,
    pub provider_order: Option<String>,
    pub reasoning_effort: Option<String>,
}

/// Derive a role's chat-request triplet — the single source shared by
/// [`chat_request`], [`crate::pipeline::verdict::synthesis_request`] and
/// [`crate::tools::analyze::consolidate_findings`]. Consumers keep their own
/// `max_tokens`, `tools` and `meta`; the call sites share no KV-cache prefix.
#[must_use]
pub(crate) fn role_chat_params(role: crate::Role) -> RoleChatParams {
    let model = crate::config::CONFIG.role_model(role);
    let routing = crate::config::CONFIG.model_routing(&model);
    RoleChatParams {
        model,
        provider_order: routing.provider_order,
        reasoning_effort: Some(
            crate::agent::role::role_info(&role)
                .default_reasoning_effort
                .to_string(),
        ),
    }
}

/// Build the byte-relevant chat params (model, tools, reasoning_effort, routing,
/// max_tokens) for a role; the triplet comes from [`role_chat_params`]. `meta` is
/// telemetry-only (never in the request body) and attached by call sites.
#[must_use]
pub(crate) fn chat_request(
    role: crate::Role,
    tool_specs: Option<Vec<crate::ToolSpec>>,
    messages: Vec<ChatMessage>,
) -> ChatRequest {
    let RoleChatParams {
        model,
        provider_order,
        reasoning_effort,
    } = role_chat_params(role);
    ChatRequest {
        messages,
        tools: tool_specs,
        model,
        max_tokens: Some(crate::DEFAULT_MAX_TOKENS),
        reasoning_effort,
        provider_order,
        meta: None,
    }
}

impl Agent {
    /// Create an agent (tools from [`crate::Role::tools`]) and register it with
    /// [`crate::agent::registry::AGENT_REGISTRY`]; deregistered on [`Drop`].
    ///
    /// `parent_key` is its DIRECT PARENT INVOCATION grouping key for the Running Agents
    /// view — `None` means a workspace singleton, ticket agents get [`ParentKey::Ticket`]
    /// implicitly; `parent_label` is that invocation's presentational label.
    ///
    /// `launch` is the launch this agent runs under; it is attached to the session so
    /// the system prompt is assembled from it.
    #[must_use]
    #[expect(clippy::too_many_arguments)] // one positional arg per construction field; callers use literals
    pub fn new(
        agent_id: String,
        role: crate::Role,
        ws: &crate::Workspace,
        ticket: Option<crate::pipeline::board::Ticket>,
        user_name: String,
        channel: String,
        is_admin: bool,
        parent_key: Option<crate::agent::registry::ParentKey>,
        parent_label: Option<String>,
        launch: Option<crate::Launch>,
    ) -> Self {
        let chrome_sessions = crate::tools::chrome::ChromeRunSessions::for_run(&agent_id);
        let (tools, tool_specs) = role_tools_and_specs(
            role,
            ws,
            is_admin,
            std::sync::Arc::clone(&chrome_sessions),
            launch.as_ref(),
        );

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let pause_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Ticket agents group by ticket; an explicit parent key takes precedence.
        let parent_key = parent_key.or_else(|| {
            ticket
                .as_ref()
                .map(|t| crate::agent::registry::ParentKey::Ticket(t.id.clone()))
        });
        // The group header label: an explicit one wins, a ticket-parented agent falls
        // back to the ticket title. Presentational only.
        let parent_label = parent_label.or_else(|| match parent_key {
            Some(crate::agent::registry::ParentKey::Ticket(_)) => {
                ticket.as_ref().map(|t| t.title.clone())
            }
            _ => None,
        });
        // Registration-time manual-cancel check: a sub-agent of a research run or
        // analyze round whose cancel signal already fired must never run a round —
        // even when it registers AFTER the registry's cancel sweep (the
        // late-register race). The token is pre-cancelled, so llm_loop bails at its
        // first round-top check.
        if let Some(
            crate::agent::registry::ParentKey::Research(run_id)
            | crate::agent::registry::ParentKey::AnalyzeRound(run_id),
        ) = &parent_key
            && crate::round_cancel::is_cancelled(run_id)
        {
            cancel_token.cancel();
        }
        let interrupt_token = tokio_util::sync::CancellationToken::new();
        let generation = crate::agent::registry::AGENT_REGISTRY.register(
            agent_id.clone(),
            role.to_string(),
            ticket.as_ref().map(|t| t.id.clone()),
            ws,
            cancel_token.clone(),
            interrupt_token.clone(),
            parent_key.clone(),
            parent_label.clone(),
            pause_stop.clone(),
        );

        // A fresh live-transcript snapshot holder for this run, so the Running
        // Agents GUI reads the in-memory conversation (including the unpersisted
        // tail) at render time; a replacement run gets a fresh one via the
        // registry's generation guard.
        let mut session = Session::default();
        session.attach_transcript(
            crate::session::TRANSCRIPT_REGISTRY.register(agent_id.clone(), generation),
        );
        // The launch travels with the session: the prompt is assembled from it at the
        // first turn, and the workspace block re-rendered from it later.
        session.launch = launch;

        Self {
            agent_id,
            role,
            session,
            workspace: Arc::new(ws.clone()),
            tools,
            tool_specs,
            cancel_token,
            pause_stop,
            paused_frozen: std::sync::atomic::AtomicBool::new(false),
            ticket,
            generation,
            tool_stats: std::sync::Mutex::new(Vec::new()),
            user_name,
            channel,
            is_admin,
            parent_key,
            parent_label,
            incoming_rx: None,
            round_ts: None,
            first_call_notify: None,
            failure: None,
            failure_class: None,
            run_failure: None,
            sleep_ended: false,
            turn_recorded: false,
            turn_state: crate::TurnState::default(),
            background_sessions: std::sync::Arc::new(
                crate::tools::shell::BackgroundSessions::default(),
            ),
            chrome_sessions,
            interrupt_token,
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if self.generation > 0 {
            crate::agent::registry::AGENT_REGISTRY.deregister(&self.agent_id, self.generation);
            crate::session::TRANSCRIPT_REGISTRY.deregister(&self.agent_id, self.generation);
        }
        // Teardown kill: background shell sessions must not outlive the agent —
        // force-kill only, no grace (SIGKILL per process group; one
        // TerminateJobObject per session on Windows).
        self.background_sessions.terminate_all();
    }
}

/// Render a board ticket comment as a user-visible message, shared by live
/// delivery and the resume catch-up so both render identically.
fn format_comment_message(user_name: &str, content: &str) -> String {
    format!("[Comment from {user_name} on ticket]: {content}")
}

impl Agent {
    /// Hand the tool stats to the telemetry queue, then persist the final assistant
    /// message. An empty unpersisted tail is logged: INFO after a drain cut (lossless),
    /// DEBUG after the `sleep` tool, INFO otherwise (genuine anomaly).
    pub async fn finalize_session(&mut self) -> anyhow::Result<()> {
        // The batch leaves on the telemetry queue ([`crate::telemetry`]).
        let stats = {
            let mut guard = self.tool_stats.lock().unwrap_poison();
            std::mem::take(&mut *guard)
        };
        if !stats.is_empty() {
            crate::telemetry::enqueue(crate::telemetry::TelemetryRecord::ToolCalls(
                crate::telemetry::ToolCallBatch {
                    source: crate::stats::CallSource::Agent,
                    agent_id: self.agent_id.clone(),
                    role: self.role.as_str().to_string(),
                    workspace: self.workspace.path.clone(),
                    recorded_at: crate::db::now(),
                    records: stats,
                },
            ));
        }

        // Cancelled (stop or shutdown): no assistant message is expected, so skip
        // finalization. Tool results are durable; a dropped tail loses nothing
        // (comments stay in the board DB).
        if self.cancel_token.is_cancelled() || crate::shutdown::shutdown_token().is_cancelled() {
            tracing::debug!(
                agent_id = %self.agent_id,
                role = %self.role,
                workspace = %self.workspace.name,
                ticket = self.ticket.as_ref().map(|t| t.id.as_str()),
                "Session finalize skipped (agent cancelled or shutdown)"
            );
            return Ok(());
        }

        match self.session.finalize(&self.agent_id).await? {
            crate::session::FinalizeOutcome::Flushed => {}
            crate::session::FinalizeOutcome::NoUnpersistedTail => {
                if crate::shutdown::is_draining() {
                    // Drain cut the turn right after the last commit: expected and
                    // lossless — committed frames are durable, ticket rounds resume at
                    // boot (job stays 'launched'), direct sessions on the next message.
                    tracing::info!(
                        agent_id = %self.agent_id,
                        role = %self.role,
                        workspace = %self.workspace.name,
                        ticket = self.ticket.as_ref().map(|t| t.id.as_str()),
                        "Session finalize no-op: turn cut by graceful drain — \
                         committed frames are durable; resumes at boot or on the next user message"
                    );
                } else if self.sleep_ended {
                    // Sleep tool: results are durable and the Assistant is
                    // intentionally silent — normal, not the anomaly below.
                    tracing::debug!(
                        agent_id = %self.agent_id,
                        role = %self.role,
                        workspace = %self.workspace.name,
                        ticket = self.ticket.as_ref().map(|t| t.id.as_str()),
                        "Session finalize no-op: turn ended via sleep — waiting for new input"
                    );
                } else {
                    // Genuine anomaly: no persisted output outside a drain. Keep the
                    // byte-identical message so external correlation keeps working.
                    tracing::info!(
                        agent_id = %self.agent_id,
                        role = %self.role,
                        workspace = %self.workspace.name,
                        ticket = self.ticket.as_ref().map(|t| t.id.as_str()),
                        "finalize called but no new assistant message in history"
                    );
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Whether a workspace-pause (strict freeze) stop was requested.
    #[must_use]
    fn is_cancelled_by_pause(&self) -> bool {
        self.pause_stop.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// IMMUTABLE bail-time snapshot of a cooperative workspace-pause freeze: set once at
    /// the LLM-round boundary and never cleared, so a finalizer can tell a frozen run from
    /// a technical failure after the workspace is resumed.
    #[must_use]
    pub(crate) fn is_paused_frozen(&self) -> bool {
        self.paused_frozen.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Failure reason, global-token-first like [`failure_classification`]: shutdown
    /// cancels every per-agent token, so it is checked before an internal cancel.
    #[must_use]
    pub(crate) fn failure_reason(&self, fallback: &str) -> String {
        if crate::shutdown::shutdown_token().is_cancelled() {
            "service shutting down".to_string()
        } else if self.is_cancelled() {
            "agent cancelled".to_string()
        } else {
            self.failure.clone().unwrap_or_else(|| fallback.to_string())
        }
    }

    /// Whether the last run failed technically — the gate for the destructive reset path.
    #[must_use]
    pub(crate) fn failed_technically(&self) -> bool {
        matches!(self.run_failure, Some(crate::RunFailure::Technical))
    }

    /// The shutdown/drain bail guard: the session is durable, so a cut round resumes
    /// when re-driven.
    fn bail_on_shutdown() -> anyhow::Result<()> {
        if crate::shutdown::aborting() {
            anyhow::bail!(
                "Agent round cut short by shutdown/drain — session is durable; resumes when re-driven"
            );
        }
        Ok(())
    }

    /// Log-only marker: drain began mid-completion, so the call stays dangling.
    fn log_drain_cut(tool: &str) {
        tracing::info!(tool = %tool, "Drain-cut during resume-completion — leaving calls dangling");
    }

    /// Show `label` on the agent's card while an observational LLM call runs — those
    /// register no non-agent row, so the card is the only tracker. Clears on drop.
    fn activity_guard(&self, label: &'static str) -> crate::agent::registry::ActivityGuard {
        crate::agent::registry::AGENT_REGISTRY.activity_started(
            &self.agent_id,
            self.generation,
            label,
        )
    }

    /// Run a complete agent turn: session init, the work loop under shutdown
    /// cancellation, finalize. `messages` arrives in arrival order (several when an
    /// urgent message cut the previous turn); each is its own user message, while the
    /// per-turn blocks ride the first one only. The turn's log scope is one span —
    /// agent, role, workspace name (never its path) and ticket.
    pub async fn work(&mut self, messages: &[&str], resume: bool) -> anyhow::Result<String> {
        let span = tracing::info_span!(
            "agent",
            agent_id = %self.agent_id,
            role = %self.role,
            workspace = %self.workspace.name,
            ticket = self.ticket.as_ref().map(|ticket| ticket.id.as_str())
        );
        self.run_turn(messages, resume).instrument(span).await
    }

    /// [`Self::work`]'s body, inside the turn's span.
    async fn run_turn(&mut self, messages: &[&str], resume: bool) -> anyhow::Result<String> {
        // Whether the turn's message lands decides whether it may advance a personal
        // folder's change baseline; cleared first, so no early return carries the old one.
        self.turn_recorded = false;
        // Load the session WITHOUT the turn message: completion below must settle
        // dangling calls BEFORE the new message lands (results insert right after their
        // frame, keeping the history provider-valid).
        self.session.init(&self.agent_id).await?;

        // Re-engaging (any job kind): clear a stale sleep-ended flag BEFORE this turn, so
        // a crash mid-round is still recovered by the dead-session poller. Only the
        // Assistant, the Manager and a custom role can write it, so other roles skip it.
        if self.role.is_long_lived_chat()
            && crate::session::store()
                .get_sleep_ended(&self.agent_id)
                .await
            && let Err(e) = crate::session::store()
                .set_sleep_ended(&self.agent_id, false)
                .await
        {
            tracing::warn!(
                agent_id = %self.agent_id,
                error = %e,
                "Failed to clear sleep-ended flag"
            );
        }

        // A branch launch also clears the persisted deliberate-stop mark, fail-soft
        // like the sleep mark: relaunching a stopped branch re-enables recovery.
        if self.session.launch.is_some()
            && crate::session::store()
                .get_launch_stopped(&self.agent_id)
                .await
            && let Err(e) = crate::session::store()
                .set_launch_stopped(&self.agent_id, false)
                .await
        {
            tracing::warn!(
                agent_id = %self.agent_id,
                error = %e,
                "Failed to clear launch-stopped flag"
            );
        }

        // Universal resume-completion: settle the last tool-call frame's result-less
        // calls BEFORE any LLM call, ungated — drain mid-call, hard crash and stage
        // re-drive all converge here. Recorded on the agent for the later gate.
        self.turn_state.settled_tool_results = if crate::shutdown::aborting() {
            false
        } else {
            self.complete_pending_tool_calls().await?
        };

        // The index the turn's messages start at: a compaction below must keep every one
        // of them.
        let turn_start = self.session.history().len();
        if messages.iter().any(|msg| !msg.is_empty()) {
            self.session
                .append_turn_messages(
                    &self.agent_id,
                    messages,
                    &self.workspace,
                    &self.role,
                    self.ticket.as_ref(),
                    &self.channel,
                    &self.user_name,
                    self.round_ts.as_deref(),
                    self.is_admin,
                )
                .await?;
            self.turn_recorded = true;
        }

        // Pre-maybe_summarize drain check: a dispatch starting during the drain must not
        // destructively compact an over-threshold session before llm_loop's own check.
        // The round is cut before any LLM work; boot resume continues it.
        Self::bail_on_shutdown()?;
        // Pre-maybe_summarize cancel check, same rule: an agent stopped before its first
        // model call must not spend a summarization call on itself.
        if self.cancel_token.is_cancelled() {
            anyhow::bail!("Agent cancelled");
        }

        // A resumed stage round with an empty message would otherwise never surface
        // board comments (the session-start ticket block is not re-rendered).
        self.inject_outstanding_comments(resume, messages).await;

        // Mandatory: destructive over-threshold compaction is SKIPPED on resumed turns
        // AND on turns that just settled dangling results — the pre-crash trail must
        // survive (do NOT infer resume from an empty message). A settlement is resume-like:
        // the retention window excludes tool frames/results, so compacting right after
        // settling would drop the settled result; the next plain turn summarizes.
        if !resume && !self.turn_state.settled_tool_results {
            self.maybe_summarize(turn_start).await;
        }

        let shutdown = crate::shutdown::shutdown_token();
        let response_result = tokio::select! {
            () = shutdown.cancelled() => {
                Err(anyhow::anyhow!("Shutting down"))
            }
            result = self.llm_loop() => result,
        };

        // Always finalize, even on global shutdown: persist progress.
        if let Err(e) = self.finalize_session().await {
            tracing::error!(error = %e, "Session finalize failed");
        }

        let response = response_result?;
        Ok(response)
    }

    /// Universal resume-completion: settle the last tool-call frame's result-less calls
    /// BEFORE any LLM call, on every role/path. Durable sync rounds (`analyze`, `send`)
    /// owned by this session pin are slot-resumed, everything else re-executed. Only
    /// rounds the frame itself dispatched are bindable — an older pool entry belongs to a
    /// settled frame and is abandoned, never bound. Settlement and terminalization commit
    /// in ONE transaction, so a crash rolls back both and the next cycle re-settles
    /// idempotently; results settle contiguously after their frame.
    ///
    /// Intentionally ungated: drain mid-call, hard crash and stage re-drive all converge
    /// here. Completed calls are never re-run (`pending_tool_frame` excludes them).
    /// Returns `true` when anything settled — the [`TurnState::settled_tool_results`] both
    /// summarization gates skip a resume-like turn from. Boxed to break the async-opaque
    /// Send cycle around the resume cores.
    #[expect(clippy::too_many_lines)] // deliberate: one cohesive resume+settle path
    fn complete_pending_tool_calls(&mut self) -> CompletePendingToolCallsFuture<'_> {
        Box::pin(async move {
            let Some(frame) = self.session.pending_tool_frame() else {
                return Ok(false);
            };
            let calls = frame.calls;
            tracing::info!(
                agent_id = %self.agent_id,
                role = %self.role,
                count = calls.len(),
                "Completing pending tool calls before the LLM call"
            );

            let conn = &crate::session::store().conn;
            // Launched jobs, fetched ONCE (newest-first) and CONSUMED per call, so two
            // same-kind calls bind DISTINCT jobs; jobs being newest-first while calls are
            // in dispatch order, each call takes the OLDEST remaining one.
            let pool = crate::jobs::find_owned_launched_jobs(conn, &self.agent_id).await?;
            // Bind only rounds THIS frame dispatched: the frame row precedes its calls
            // (emission-time durability), so its own rounds are strictly newer, while a
            // leftover from an earlier frame (an infra-failed dispatch never terminalizes
            // its row) is older and binding it would settle an old report as this call's
            // result. The split is frame OWNERSHIP, not freshness.
            let (pre_frame, mut owned) = match crate::session::store()
                .last_assistant_created_at(&self.agent_id)
                .await?
                .and_then(|frame| crate::db::parse_utc_timestamp(&frame).ok())
            {
                Some(frame) => pool.into_iter().partition(|j| {
                    crate::db::parse_utc_timestamp(&j.created_at).is_ok_and(|c| c < frame)
                }),
                // No frame row to anchor against (unreachable: it is persisted first):
                // NOTHING is bindable, so every call is re-run rather than risk settling
                // another round's report as its result.
                None => (pool, Vec::new()),
            };
            if !pre_frame.is_empty() {
                tracing::info!(
                    agent_id = %self.agent_id,
                    count = pre_frame.len(),
                    "Abandoning launched rounds this frame did not dispatch"
                );
            }

            let mut pairs: Vec<(String, String)> = Vec::with_capacity(calls.len());
            let mut follow_up: Vec<ChatMessage> = Vec::new();
            let mut terminalize: Vec<String> = Vec::new();
            // Fresh images of the resumed round, injected as ONE message after the loop.
            let mut fresh_images: Vec<ImagePayload> = Vec::new();
            // Shared dedup key for fresh-image injection, lazily computed.
            let mut seen_data_uris: Option<std::collections::HashSet<String>> = None;

            for call in calls {
                // Durable-resume attempt: the JOBS ROW is the source of truth. A terminal
                // job yields content (terminalized after settlement), a DrainCut stops the
                // completion, a gone job surfaces a short note. The round kind comes from
                // the call's tool and choice argument, never a stored kind.
                let mut resumed_content: Option<String> = None;
                if let Some(core) =
                    crate::tools::SyncDurableCore::from_call(&call.name, &call.arguments)
                    && let Some(pos) = owned.iter().rposition(|j| j.kind == core.kind_str())
                {
                    // `rposition`, not `position`: the FIRST call must bind the OLDEST
                    // matching job (`position` would cross the results).
                    let job = owned.swap_remove(pos);
                    let outcome = core.resume_sync_core(&self.workspace, &job.id, true).await;
                    match outcome {
                        Ok(crate::jobs::SyncResumeOutcome::Terminal(_, _, res)) => {
                            terminalize.push(job.id);
                            resumed_content = Some(match res {
                                Ok(text) => text,
                                Err(e) => e.to_string(),
                            });
                        }
                        Ok(crate::jobs::SyncResumeOutcome::DrainCut) => {
                            // Drain began mid-resume: leave everything dangling.
                            Self::log_drain_cut(&call.name);
                            return Ok(false);
                        }
                        Ok(crate::jobs::SyncResumeOutcome::Gone) => {
                            // Explicitly abandoned (session /clear, workspace deletion):
                            // the row is gone, so do NOT push onto `terminalize`.
                            tracing::info!(
                                job = %job.id,
                                "Job row gone (explicitly abandoned) — surfacing note instead of re-executing"
                            );
                            resumed_content = Some(
                                "The durable job behind this call was explicitly abandoned — \
                                 no result is available. Re-issue the call if the work is still needed."
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            // An INFRA failure (roster/checkpoint DB error) must fail
                            // the round rather than spawn a duplicate fresh job — the
                            // job row stays launched for the next drive.
                            tracing::error!(
                                job = %job.id,
                                error = %e,
                                "Resume-core infra failure — failing the round"
                            );
                            return Err(e).context("resume-core infra failure");
                        }
                    }
                }

                let formatted = if let Some(content) = resumed_content {
                    // Format the resumed raw content exactly like the commit path.
                    find_tool(&self.tools, &call.name)
                        .map(|t| t.format_output(&content))
                        .unwrap_or(content)
                } else if call.name == "sleep"
                    && let Some(reason) =
                        sleep_rejection_reason(frame.has_text, frame.call_count > 1)
                {
                    // Sleep-delivery guard on the resume path: re-executing the sleep
                    // would settle a "Zzz..." result over text the user never saw —
                    // settle the rejection so the model resends it as a plain round.
                    tracing::warn!(
                        agent_id = %self.agent_id,
                        tool = %call.name,
                        reason,
                        "Sleep call rejected during resume-completion"
                    );
                    self.push_tool_call_record(
                        call.name.clone(),
                        &call.arguments,
                        0,
                        false,
                        Some(reason.to_string()),
                    );
                    Self::failure_outcome(&call.name, &call.arguments, reason)
                        .0
                        .output
                } else {
                    // Re-execute under the tool task-local context.
                    let outcome = self
                        .in_tool_context(async {
                            self.execute_tool(&call.name, call.arguments.clone()).await
                        })
                        .await;
                    if outcome.suspended {
                        // Drain-cut mid-execution — leave the call dangling.
                        Self::log_drain_cut(&call.name);
                        return Ok(false);
                    }
                    // Format through the tool and collect fresh image payloads.
                    let (formatted, fresh_payloads) = tool_result_content(
                        &self.tools,
                        &call.name,
                        outcome,
                        &mut seen_data_uris,
                        self.session.history(),
                    )
                    .await;
                    fresh_images.extend(fresh_payloads);
                    formatted
                };
                pairs.push((call.id, formatted));
            }

            if let Some(message) = injected_images_message(&self.agent_id, self.role, fresh_images)
            {
                follow_up.push(message);
            }

            // Settle + terminalize + abandon in ONE transaction: a crash rolls back BOTH,
            // so the jobs stay launched and the next cycle re-settles idempotently (the
            // settle DELETEs the post-frame rows and reinserts, so re-settling is a
            // structural no-op). Every terminalize id also pushed a pair.
            if !pairs.is_empty() || !follow_up.is_empty() {
                let tx = conn
                    .begin_tx()
                    .await
                    .context("begin settle+terminalize tx")?;
                let settled = crate::session::store()
                    .settle_tool_results_tx(&tx, &self.agent_id, &pairs, &follow_up)
                    .await
                    .context("settle resumed results")?;
                for id in terminalize {
                    tx.execute(
                        "DELETE FROM jobs WHERE id = ?1",
                        crate::db::params![id.as_str()],
                    )
                    .await
                    .with_context(|| format!("terminalize job {id}"))?;
                }
                // Everything this frame cannot own is abandoned with the settle. Repeating
                // a cut-mid-flight call is the accepted cost; a crossed report is not.
                for job in pre_frame.iter().chain(owned.iter()) {
                    tx.execute(
                        "DELETE FROM jobs WHERE id = ?1",
                        crate::db::params![job.id.as_str()],
                    )
                    .await
                    .with_context(|| format!("abandon job {}", job.id))?;
                }
                tx.commit().await.context("commit settle+terminalize tx")?;
                // Mirror into the history only AFTER the durable commit.
                self.session
                    .mirror_settled(settled)
                    .context("mirror settled results")?;
            }
            Ok(!pairs.is_empty() || !follow_up.is_empty())
        })
    }

    /// The round loop: LLM call → tool execution → repeat until a final answer.
    /// Runs inside [`Self::work`]'s span, which carries the run's log scope.
    #[expect(clippy::too_many_lines)] // the round loop with its cut and drain gates
    async fn llm_loop(&mut self) -> anyhow::Result<String> {
        let mut tool_rounds = 0usize;
        let mut accumulated_media_paths: Vec<(&'static str, String)> = Vec::new();
        loop {
            if self.cancel_token.is_cancelled() {
                anyhow::bail!("Agent cancelled");
            }
            // Cooperative workspace-pause (strict freeze): the round ends here, no new LLM
            // call starts. The pause does NOT fire the cancel token, so an in-flight round
            // completes and commits first; the finalizer treats the agent as a no-op (the
            // job stays for the unpause re-drive).
            if self.is_cancelled_by_pause() {
                // Record the freeze IMMUTABLY, so a later unpause (which clears the
                // live pause_stop) cannot turn it into a misclassified hard failure.
                self.paused_frozen
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                anyhow::bail!("Agent frozen by workspace pause — resumes at unpause");
            }
            // Shutdown/drain: checked at round top, after the previous group's commit, so
            // the current group completes and no new LLM call starts. The round resumes at
            // boot from the job row.
            Self::bail_on_shutdown()?;
            // Urgent-message interrupt: not a cancellation — the turn ends here, every
            // result already committed, with the phrase as its only session entry. Checked
            // after shutdown/drain (their priority is unchanged) and before another model call.
            if self.interrupt_token.is_cancelled() {
                return Ok(self.end_interrupted_turn());
            }
            if tool_rounds >= MAX_TOOL_ROUNDS {
                anyhow::bail!(
                    "Agent exceeded maximum of {MAX_TOOL_ROUNDS} tool rounds \
                         — model may be stuck in a tool-calling loop"
                );
            }

            // Messages arrive as user messages with a descriptive prefix, so the
            // model understands their source (`format_comment_message`).
            self.drain_incoming_messages().await;

            // Leader-stagger signal: fired after the FIRST LLM call completes,
            // success or failure — fail-open; Notify's stored permit covers a
            // fire-before-wait race. Raced against the urgent interrupt: cutting a
            // request means dropping it (nothing recorded), which is what makes the
            // cut immediate. The signal is skipped on that path.
            let interrupt = self.interrupt_token.clone();
            let llm_result = tokio::select! {
                biased;
                () = interrupt.cancelled() => return Ok(self.end_interrupted_turn()),
                result = self.llm_call() => result,
            };
            if tool_rounds == 0
                && let Some(notify) = &self.first_call_notify
            {
                notify.notify_one();
            }

            // Provider input-image content-inspection rejection: strip the markers from
            // the most recent user message (durable-first) and continue — the next round
            // re-runs `llm_call` as part of the NORMAL loop. No retry guard: once the
            // message carries no image marker, a failure takes the normal path.
            //
            // The leader-stagger signal has already fired, so the stripped leader's next
            // request diverges from its followers' byte-stable prefix — fail-open, and
            // moot for direct chat (management rounds never send images). The retried
            // call re-runs the tool-round-0 `notify_one`, but the stagger's single waiter
            // has consumed it: the extra permit is dropped, no follower released twice.
            // The downcast is taken into a local first — `anyhow::Chain` is not `Send`
            // and must not be held across the await (a Send-bound spawned task).
            let image_rejection = llm_result.as_ref().err().and_then(|e| {
                e.chain()
                    .find_map(|cause| cause.downcast_ref::<crate::retry::RetryExhausted>())
            });
            if let Some(exhausted) = image_rejection
                && self.strip_rejected_input_image(exhausted).await
            {
                tracing::info!(
                    agent_id = %self.agent_id,
                    role = %self.role,
                    tool_rounds,
                    "Stripped provider-rejected input image from the most recent user \
                     message — continuing the normal loop"
                );
                continue;
            }

            let PreparedAssistantTurn {
                mut display_text,
                tool_calls,
                history_content,
                raw_text,
            } = prepare_assistant_turn(
                llm_result.with_context(|| llm_step_failure_context(&self.session, tool_rounds))?,
            );

            if tool_calls.is_empty() {
                self.session.push_assistant(history_content);
                // Append any pending media markers the model may have omitted
                for (marker_prefix, path) in &accumulated_media_paths {
                    let marker = format!("{marker_prefix}{path}]");
                    if !display_text.contains(&marker) {
                        let _ = write!(display_text, "\n{marker}");
                    }
                }
                return Ok(display_text);
            }

            // EMISSION-TIME DURABILITY: the assistant frame and its tool calls are
            // committed BEFORE the calls execute — a crash or drain mid-execution
            // leaves them recorded with results absent, and resume-completion (run
            // before any LLM call) re-runs or resumes them. The end-of-round commit
            // persists only the results.
            self.session
                .persist_messages(
                    &self.agent_id,
                    &[ChatMessage::assistant(history_content.clone())],
                )
                .await?; // Sleep-delivery guard: a sleep call in a round with
            // accompanying raw text or bundled with other tool calls is
            // rejected (see `execute_tool_round`) instead of silently
            // swallowing the text or ending the run.
            let has_sleep = tool_calls.iter().any(|c| c.name == "sleep");
            let sleep_rejection = has_sleep
                .then(|| sleep_rejection_reason(!raw_text.trim().is_empty(), tool_calls.len() > 1))
                .flatten();
            let all_outcomes = self.execute_tool_round(&tool_calls, sleep_rejection).await;

            // A call the interrupt cut (`None`) is settled with the fixed synthetic
            // result, in its own place, so every call ends with a result contiguous
            // with it and the exchange stays model-valid. A call already returned
            // keeps its real result.
            let cut = all_outcomes.iter().any(Option::is_none);
            let all_outcomes: Vec<ToolExecutionOutcome> = all_outcomes
                .into_iter()
                .map(|outcome| outcome.unwrap_or_else(interrupted_outcome))
                .collect();

            // Track media outcomes for the marker fallback.
            accumulated_media_paths.extend(extract_media_from_outcomes(
                &self.tools,
                &tool_calls,
                &all_outcomes,
            ));

            // Read before the commit, which consumes the outcomes' payloads.
            let sleep_ended = all_outcomes.iter().any(|o| o.success && o.ends_turn);

            self.commit_tool_results(&tool_calls, all_outcomes).await?;

            // Sleep stop: a successful `sleep` ends the run gracefully AFTER this
            // round — every result above is committed. It returns before the next
            // iteration, so the loop-top MAX_TOOL_ROUNDS guard never sees it.
            if sleep_ended {
                self.sleep_ended = true;
                if let Err(e) = crate::session::store()
                    .set_sleep_ended(&self.agent_id, true)
                    .await
                {
                    tracing::warn!(
                        agent_id = %self.agent_id,
                        error = %e,
                        "Failed to persist sleep-ended flag — a stale tool-tail session may be re-recovered (accepted)"
                    );
                }
                return Ok(String::new());
            }

            // The cut round's results are committed, so the interruption ends the turn here
            // instead of another round. A successful sleep returned above: that round
            // concluded inside its only call, so nothing was cut.
            if cut || self.interrupt_token.is_cancelled() {
                return Ok(self.end_interrupted_turn());
            }

            tool_rounds += 1;
        }
    }

    /// End this turn on an urgent-message interrupt: record the phrase as the session entry
    /// and hand back an empty reply. [`TurnState::interrupted`] keeps the router from
    /// delivering anything — no chat row, no channel message. The interrupted answer is
    /// abandoned: nothing of it is delivered later.
    fn end_interrupted_turn(&mut self) -> String {
        self.session
            .push_assistant(URGENT_INTERRUPT_ENTRY.to_string());
        self.turn_state.interrupted = true;
        String::new()
    }

    /// Execute a batch of tool calls respecting side-effect ordering: read-only
    /// runs of consecutive calls execute in parallel, a side-effecting call runs
    /// alone, and groups keep the original call order. Outcomes stay one-to-one with
    /// `tool_calls`; `None` is a call the urgent interrupt cut before it returned.
    async fn execute_tool_group(
        &self,
        tool_calls: &[ToolCall],
    ) -> Vec<Option<ToolExecutionOutcome>> {
        // Unknown tools are conservatively treated as side-effecting.
        let side_flags: Vec<bool> = tool_calls
            .iter()
            .map(|call| find_tool(&self.tools, &call.name).is_none_or(super::Tool::side_effects))
            .collect();

        let mut outcomes: Vec<Option<ToolExecutionOutcome>> = Vec::with_capacity(tool_calls.len());
        let mut i = 0usize;

        self.in_tool_context(async {
            while i < tool_calls.len() {
                if side_flags[i] {
                    // Side-effecting: a single-call group.
                    let outcome = self
                        .race_interrupt(
                            self.execute_tool(&tool_calls[i].name, tool_calls[i].arguments.clone()),
                        )
                        .await;
                    outcomes.push(outcome);
                    i += 1;
                } else {
                    // Read-only group: extend while the next call is read-only too.
                    let group_start = i;
                    while i < tool_calls.len() && !side_flags[i] {
                        i += 1;
                    }
                    let group_calls = &tool_calls[group_start..i];

                    // The whole group runs in parallel, each call raced on its own, so
                    // one already returned keeps its real result while cut siblings
                    // settle as interrupted.
                    let group_outcomes: Vec<_> =
                        futures_util::future::join_all(group_calls.iter().map(|call| {
                            self.race_interrupt(
                                self.execute_tool(&call.name, call.arguments.clone()),
                            )
                        }))
                        .await;

                    outcomes.extend(group_outcomes);
                }
            }
        })
        .await;

        outcomes
    }

    /// Run one tool call against the urgent interrupt: `None` when the cut came
    /// first, which drops the call and its whole process tree. A cut that already
    /// landed must not start the call at all — one poll is enough to spawn a child or
    /// write a file. Biased on the call, so one completing in the same instant keeps
    /// its real result.
    async fn race_interrupt<F>(&self, call: F) -> Option<ToolExecutionOutcome>
    where
        F: std::future::Future<Output = ToolExecutionOutcome>,
    {
        if self.interrupt_token.is_cancelled() {
            return None;
        }
        let interrupt = self.interrupt_token.clone();
        tokio::select! {
            biased;
            outcome = call => Some(outcome),
            () = interrupt.cancelled() => None,
        }
    }

    /// Execute one tool round under the sleep-delivery guard: with a rejection active,
    /// every `sleep` call is intercepted BEFORE execution and settled as a synthetic
    /// failure at its own position, siblings running normally. Outcomes stay one-to-one
    /// with `tool_calls` — `commit_tool_results` relies on that alignment.
    async fn execute_tool_round(
        &self,
        tool_calls: &[ToolCall],
        sleep_rejection: Option<&'static str>,
    ) -> Vec<Option<ToolExecutionOutcome>> {
        let Some(reason) = sleep_rejection else {
            return self.execute_tool_group(tool_calls).await;
        };

        let rejected = tool_calls.iter().filter(|c| c.name == "sleep").count();
        tracing::info!(
            agent_id = %self.agent_id,
            role = %self.role,
            rejected,
            reason,
            "Sleep call rejected — must run alone in a text-free round"
        );

        // Only non-sleep calls execute; sleep calls settle as synthetic failures at
        // their own positions, keeping the round aligned one-to-one.
        let sibling_calls: Vec<ToolCall> = tool_calls
            .iter()
            .filter(|c| c.name != "sleep")
            .cloned()
            .collect();
        let mut sibling_outcomes = self.execute_tool_group(&sibling_calls).await.into_iter();

        let mut outcomes: Vec<Option<ToolExecutionOutcome>> = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            if call.name == "sleep" {
                self.push_tool_call_record(
                    call.name.clone(),
                    &call.arguments,
                    0,
                    false,
                    Some(reason.to_string()),
                );
                outcomes.push(Some(
                    Self::failure_outcome(&call.name, &call.arguments, reason).0,
                ));
            } else {
                outcomes.push(
                    sibling_outcomes
                        .next()
                        .expect("non-sleep outcomes align one-to-one"),
                );
            }
        }
        outcomes
    }

    /// Run `fut` under this agent's `CURRENT_TOOL_*` task-locals, scoped once per tool
    /// batch so concurrent read-only tools never interleave their user context.
    async fn in_tool_context<T>(&self, fut: impl std::future::Future<Output = T>) -> T {
        let user_name = self.user_name.clone();
        let channel = self.channel.clone();
        let parent_key = self.parent_key.clone();
        let parent_label = self.parent_label.clone();
        let background_sessions = Some(self.background_sessions.clone());
        let agent_id = Some(self.agent_id.clone());
        let agent_tracking = Some(crate::agent::registry::AgentTracking {
            role: self.role.as_str().to_string(),
            ticket: self
                .ticket
                .as_ref()
                .map(|ticket| ticket.id.clone())
                .unwrap_or_default(),
        });
        let launch = self.session.launch.clone();
        CURRENT_TOOL_USER_NAME
            .scope(user_name, async {
                CURRENT_TOOL_CHANNEL
                    .scope(channel, async {
                        CURRENT_TOOL_PARENT_KEY
                            .scope(parent_key, async {
                                CURRENT_TOOL_PARENT_LABEL
                                    .scope(parent_label, async {
                                        CURRENT_TOOL_BACKGROUND_SESSIONS
                                            .scope(background_sessions, async {
                                                CURRENT_TOOL_AGENT_ID
                                                    .scope(agent_id, async {
                                                        CURRENT_TOOL_AGENT_TRACKING
                                                            .scope(agent_tracking, async {
                                                                CURRENT_TOOL_CUSTOM_LAUNCH
                                                                    .scope(launch, fut)
                                                                    .await
                                                            })
                                                            .await
                                                    })
                                                    .await
                                            })
                                            .await
                                    })
                                    .await
                            })
                            .await
                    })
                    .await
            })
            .await
    }

    /// Construct a failure outcome tuple for an error reason.
    ///
    /// Assumes `reason` may contain sensitive data; scrubs it before use in feedback
    /// text, tracing logs and stats.
    #[must_use]
    fn failure_outcome(
        call_name: &str,
        call_arguments: &serde_json::Value,
        reason: &str,
    ) -> (ToolExecutionOutcome, String) {
        let reason = scrub_credentials(reason);
        (
            ToolExecutionOutcome {
                output: format_tool_failure_feedback(call_name, call_arguments, &reason),
                success: false,
                // No payloads and no derivation on a failure: the flag is unread.
                image_payloads: Vec::new(),
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
            reason,
        )
    }

    /// Record one tool invocation into the in-memory stats (sent to telemetry at
    /// finalization); arguments go through [`crate::stats::recorded_arguments`].
    fn push_tool_call_record(
        &self,
        tool_name: String,
        arguments: &serde_json::Value,
        duration_ms: i64,
        success: bool,
        error_message: Option<String>,
    ) {
        let arguments = crate::stats::recorded_arguments(arguments, &[]);
        let mut guard = self.tool_stats.lock().unwrap_poison();
        guard.push(crate::ToolCallRecord {
            tool_name,
            arguments,
            duration_ms,
            success,
            error_message,
        });
    }

    #[expect(clippy::too_many_lines)]
    async fn execute_tool(
        &self,
        call_name: &str,
        call_arguments: serde_json::Value,
    ) -> ToolExecutionOutcome {
        // Pre-flight cancellation check: a call dispatched before llm_loop's round-top
        // check can still reach here, and a side-effecting tool (shell, file edits,
        // sub-agents) must not run after a Stop.
        if self.cancel_token.is_cancelled() {
            let reason = "Agent cancelled — tool execution skipped";
            tracing::debug!(
                tool = %call_name,
                "Agent cancelled — skipping tool execution"
            );
            return Self::failure_outcome(call_name, &call_arguments, reason).0;
        }

        let start = Instant::now();
        let (tool_name, tool_arguments) = normalize_tool_call(call_name, call_arguments);
        if tool_name != call_name {
            tracing::debug!(
                original = %call_name,
                normalized = %tool_name,
                "Repaired tool call name"
            );
        }

        // The arms log at distinct levels (info vs debug), so unknown-tool failures
        // are distinguishable from execution errors without string matching.
        let (outcome, error_reason) = match find_tool(&self.tools, &tool_name) {
            None => {
                let reason = format!("Unknown tool: {tool_name}");
                let duration = start.elapsed();
                tracing::debug!(
                    tool = %tool_name,
                    duration_ms = duration.as_millis(),
                    success = false,
                    "Unknown tool call"
                );
                Self::failure_outcome(&tool_name, &tool_arguments, &reason)
            }
            Some(tool) => {
                let exec_result = tool
                    .execute_with_payloads(&self.workspace, tool_arguments.clone())
                    .await;
                let duration = start.elapsed();
                match exec_result {
                    Ok(result) => {
                        let output_text = if result.text.is_empty() {
                            String::from("(no output)")
                        } else {
                            result.text
                        };
                        tracing::debug!(
                            tool = %tool_name,
                            duration_ms = duration.as_millis(),
                            "Tool execution completed"
                        );
                        (
                            ToolExecutionOutcome {
                                output: scrub_tool_output(tool, &tool_arguments, &output_text),
                                success: true,
                                image_payloads: result.image_payloads,
                                text_is_content: result.text_is_content,
                                preserve_text: result.preserve_text,
                                suspended: false,
                                ends_turn: tool.ends_turn_on_success(),
                            },
                            String::new(),
                        )
                    }
                    Err(e) => {
                        // Drain-cut suspension: the tool's durable work was deliberately
                        // left launched for deterministic resume. `suspended` makes the
                        // result commit SKIP it — the frame is durable and the result
                        // intentionally absent; resume-completion settles it later.
                        if e.downcast_ref::<crate::tools::CallSuspended>().is_some() {
                            tracing::debug!(
                                tool = %tool_name,
                                duration_ms = duration.as_millis(),
                                success = false,
                                suspended = true,
                                "Tool execution drain-cut — durable work left launched"
                            );
                            (
                                ToolExecutionOutcome {
                                    // Never read: suspended outcomes are skipped.
                                    output: String::new(),
                                    success: false,
                                    image_payloads: Vec::new(),
                                    text_is_content: false,
                                    preserve_text: false,
                                    suspended: true,
                                    ends_turn: false,
                                },
                                String::new(),
                            )
                        } else {
                            let (outcome, error_reason) = Self::failure_outcome(
                                &tool_name,
                                &tool_arguments,
                                &crate::tools::tool_failure_reason(&tool_name, &e.to_string()),
                            );
                            tracing::debug!(
                                tool = %tool_name,
                                duration_ms = duration.as_millis(),
                                success = false,
                                "Tool execution error: {error_reason}"
                            );
                            (outcome, error_reason)
                        }
                    }
                }
            }
        };

        // One stats record per invocation: arguments, duration, outcome, error.
        let elapsed_ms = start.elapsed().as_millis();
        let duration_ms = i64::try_from(elapsed_ms).unwrap_or(0);
        self.push_tool_call_record(
            tool_name,
            &tool_arguments,
            duration_ms,
            outcome.success,
            (!error_reason.is_empty()).then_some(error_reason),
        );
        outcome
    }

    async fn llm_call(&mut self) -> anyhow::Result<ChatResponse> {
        let messages = self.session.history().to_vec();
        let request = self.build_chat_request(messages.clone(), "agent");

        let policy = crate::retry::RetryPolicy::current();
        let response = crate::retry::agent_chat(request, &policy)
            .await
            .with_context(|| format!("LLM call {RETRY_EXHAUSTION_MARKER}"))?;

        // Reasoning-only stop: recovered here, before any promotion, persistence or
        // display, via the bounded continuation.
        let response = self
            .recover_if_reasoning_only_stop(messages, response, Self::AGENT_REASONING_RECOVERY)
            .await?;

        // Only a response actually returned to the loop updates the session length:
        // failures propagate above, and an exhausted reasoning-only turn returned
        // early having recorded nothing.
        self.record_session_usage(&response).await;
        Ok(response)
    }

    /// Detect a provider input-image content-inspection rejection and durably strip the
    /// image from the most recent user message. `true` when the strip applied, `false`
    /// for a conservative no-op (the original error path stands).
    ///
    /// Persist ordering: the store rewrite completes before the in-memory swap, so a
    /// persist failure leaves the session untouched and the LLM error propagates.
    /// Telemetry: the rejected attempt already recorded its `non_retryable` row; the
    /// corrected call records its own success row.
    async fn strip_rejected_input_image(
        &mut self,
        exhausted: &crate::retry::RetryExhausted,
    ) -> bool {
        let Some(idx) = crate::session::image_strip::detect_input_image_rejection(
            exhausted,
            self.session.history(),
        ) else {
            return false;
        };
        let reason = crate::session::image_strip::extract_provider_reason(exhausted);
        // Invariant guard: detection and the strip share one marker predicate, so a
        // detected rejection always changes the content. If they drift — a malformed
        // `[IMAGE:` passes detection but the regex leaves it verbatim — a no-change
        // rewrite would report `Rewritten`, the loop `continue` would skip
        // `tool_rounds += 1`, and retries unbounded. Treat it as a non-strip.
        let content = {
            let original = &self.session.history()[idx].content;
            let stripped =
                crate::session::image_strip::strip_image_markers(original, reason.as_deref());
            if stripped == *original {
                tracing::warn!(
                    agent_id = %self.agent_id,
                    role = %self.role,
                    "Input-image rejection detected but stripping produced no change — \
                     treating as a non-strip (normal failure path)"
                );
                return false;
            }
            stripped
        };
        match self
            .session
            .rewrite_last_user_message(&self.agent_id, content)
            .await
        {
            Ok(crate::session::RewriteOutcome::Rewritten) => true,
            Ok(crate::session::RewriteOutcome::UnpersistedTailNoop) => {
                tracing::info!(
                    agent_id = %self.agent_id,
                    role = %self.role,
                    "Input-image rejection detected but the most recent user message is in the \
                     unpersisted tail — conservative no-op, normal failure path applies"
                );
                false
            }
            Err(e) => {
                tracing::error!(
                    agent_id = %self.agent_id,
                    role = %self.role,
                    error = %e,
                    "Failed to persist stripped user message — keeping the original error path"
                );
                false
            }
        }
    }

    /// Bounded continuation recovery for the reasoning-only-stop class (empty content, no
    /// parsed tool calls): re-request with the thinking attached as the assistant's
    /// reasoning plus a continuation prompt. **Appended-only** — a transport failure NEVER
    /// grows the tail (the next attempt re-sends the PREVIOUS request verbatim,
    /// byte-identical even across a config hot-reload), so the prefix stays byte-stable
    /// (prompt-prefix cache preserved). The tail lives only in the re-request list, never
    /// in the session, so a failed attempt leaves no trace and the thinking cannot leak.
    ///
    /// Bounded by [`crate::retry::RetryPolicy::continuation`] (3 attempts, no wall-clock
    /// cap — the 600 s idle timeout is the only bound), checked against
    /// [`crate::shutdown::aborting`] and the cancel token between attempts; retryable
    /// errors re-send the same bytes, non-retryable ones break at once. Telemetry is
    /// operation-level: one success row, or one failure row via `fail_exhausted`, plus
    /// one `llm_failures` row per failed attempt.
    ///
    /// Exhaustion returns a [`crate::retry::RetryExhausted`] with `last_raw: None` and a
    /// class from the last failure ([`FailureClass::NoResponse`], or
    /// [`FailureClass::Shutdown`] when the global abort fired) — the thinking text is
    /// never embedded, so logs cannot leak it and it is not misclassified as provider
    /// retry exhaustion.
    async fn recover_reasoning_only_stop(
        &self,
        base: Vec<ChatMessage>,
        first: ChatResponse,
        purpose: &'static str,
    ) -> Result<ChatResponse, crate::retry::RetryExhausted> {
        let policy = crate::retry::RetryPolicy::continuation();
        let operation_started = Instant::now();
        let nudge = crate::prompt::load_prompt("resume_unfinished_turn.md")
            .trim()
            .to_string();
        let mut failures: Vec<crate::retry::RetryFailureRecord> = Vec::new();
        let mut last_request: Option<ChatRequest> = None;
        // Per-attempt `llm_failures` context — derived from the first built
        // request; `None` when the operation carries no metadata (same gating
        // as the `llm_requests` rows).
        let mut operation: Option<crate::stats::LlmOperationCtx> = None;

        // Seed the tail with the FIRST in-class response's reasoning (empty
        // content) + the continuation nudge. Each LATER in-class response
        // appends its own pair — the tail grows only when the model actually
        // produced a new response, so a transport failure never duplicates
        // the previous thinking.
        let mut tail: Vec<ChatMessage> = vec![
            // The payload always carries a string "content" key, so it decodes
            // to Some("") and serializes via the Some arm — never the
            // omitted-content shape (the serializer backstop covers any drift).
            ChatMessage::assistant(
                assistant_replay_payload(None, &[], first.reasoning.as_ref()).to_string(),
            ),
            ChatMessage::user(nudge.clone()),
        ];
        // True when a new pair was appended since the last request was built:
        // the request is rebuilt only then, otherwise the previous request
        // object is re-sent verbatim (a retryable transport error must not
        // pick up a concurrent config hot-reload).
        let mut tail_grew = true;

        for attempt in 1..=policy.max_attempts {
            let attempt_started = Instant::now();
            // The global abort dominates: it also cancels every per-agent
            // token, so classify it as shutdown, not cancellation.
            if crate::shutdown::aborting() {
                failures.push(crate::retry::RetryFailureRecord::new_simple(
                    crate::retry::FailureClass::Shutdown,
                    &anyhow::anyhow!("global shutdown or drain during continuation recovery"),
                    None,
                ));
                break;
            }
            if self.cancel_token.is_cancelled() {
                break;
            }

            let request = if tail_grew {
                tail_grew = false;
                let mut messages = base.clone();
                messages.extend(tail.iter().cloned());
                let built = self.build_chat_request(messages, purpose);
                if operation.is_none() {
                    operation = crate::stats::LlmOperationCtx::from_request(&built);
                }
                last_request = Some(built.clone());
                built
            } else {
                last_request
                    .clone()
                    .expect("the first iteration always builds a request")
            };

            match crate::providers::chat_scoped(request.clone()).await {
                // Still thinking with no answer — record the attempt (with its
                // finish_reason, so the terminal failure row keeps it), append
                // the NEW reasoning as the next tail pair, and continue.
                Ok(resp) if is_reasoning_only_stop(&resp) => {
                    let rec = crate::retry::RetryFailureRecord::with_metadata(
                        crate::retry::FailureClass::NoResponse,
                        &anyhow::anyhow!(
                            "model returned only reasoning with no answer \
                             (continuation attempt {attempt})"
                        ),
                        resp.finish_reason.clone(),
                        None,
                    );
                    record_continuation_failure(operation.as_ref(), attempt, &rec, attempt_started);
                    failures.push(rec);
                    tail.push(ChatMessage::assistant(
                        assistant_replay_payload(None, &[], resp.reasoning.as_ref()).to_string(),
                    ));
                    tail.push(ChatMessage::user(nudge.clone()));
                    tail_grew = true;
                }
                // Real answer or tool calls — the turn resolves normally.
                Ok(resp) => {
                    crate::stats::record_llm_success(&request, operation_started, attempt, &resp);
                    return Ok(resp);
                }
                // Provider failure — break on non-retryable (a payload-rejecting
                // 400 must not burn the budget); otherwise the next iteration
                // re-sends the byte-identical request (tail untouched).
                Err(err) => {
                    record_continuation_failure(
                        operation.as_ref(),
                        attempt,
                        &err.record,
                        attempt_started,
                    );
                    failures.push(err.record);
                    if !err.class.is_retryable() {
                        break;
                    }
                }
            }
        }

        let final_class = failures
            .last()
            .map_or(crate::retry::FailureClass::NoResponse, |r| r.class);
        let exhausted = crate::retry::RetryExhausted::with_last_raw(failures, final_class, None);
        match last_request {
            Some(request) => crate::retry::fail_exhausted(&request, operation_started, exhausted),
            None => Err(exhausted),
        }
    }

    /// Agent-loop policy: continuation exhaustion fails the turn.
    const AGENT_REASONING_RECOVERY: ReasoningOnlyStopRecovery = ReasoningOnlyStopRecovery {
        purpose: "agent-continuation",
        exhausted_ctx: "model returned only reasoning without an answer after continuation attempts",
    };

    /// Summarization policy: continuation exhaustion fails open with full history.
    const SUMMARIZE_REASONING_RECOVERY: ReasoningOnlyStopRecovery = ReasoningOnlyStopRecovery {
        purpose: "summarize-continuation",
        exhausted_ctx: "summarization continuation exhausted — failing open with full history",
    };

    /// Recover a reasoning-only stop via the bounded continuation, or pass the response
    /// through unchanged — the single classification point for that class, reached by both
    /// LLM paths before any promotion, persistence or display.
    ///
    /// On exhaustion the error carries `recovery.exhausted_ctx`: the agent loop fails the
    /// turn safely (nothing persisted or displayed), while summarize fails open with the
    /// full history. Leak safety: appended-only, never touching the transcript, and the
    /// exhaustion error never embeds the thinking text.
    async fn recover_if_reasoning_only_stop(
        &self,
        messages: Vec<ChatMessage>,
        response: ChatResponse,
        recovery: ReasoningOnlyStopRecovery,
    ) -> anyhow::Result<ChatResponse> {
        if !is_reasoning_only_stop(&response) {
            return Ok(response);
        }
        self.recover_reasoning_only_stop(messages, response, recovery.purpose)
            .await
            .map_err(|e| anyhow::Error::new(e).context(recovery.exhausted_ctx))
    }

    /// Record the provider-reported session length after a SUCCESSFUL agent-purpose call:
    /// `input_tokens + output_tokens`. Other builders are not session context and never
    /// pass through here; a compaction clears the value with the history it replaces.
    async fn record_session_usage(&mut self, response: &ChatResponse) {
        let Some(usage) = &response.usage else {
            return;
        };
        let (Some(input), Some(output)) = (usage.input_tokens, usage.output_tokens) else {
            return;
        };
        let token_length = input.saturating_add(output);

        self.session.set_token_length(Some(token_length));
        if let Err(e) = crate::session::store()
            .set_token_length(&self.agent_id, Some(token_length))
            .await
        {
            tracing::warn!(
                agent_id = %self.agent_id,
                error = %e,
                "Failed to persist session token length — in-memory value may drift from the store until the next successful call"
            );
        }
    }

    /// Persist tool results and push them into memory for the next round. The assistant
    /// frame is already durable from EMISSION time, so this writes ONLY the result rows.
    /// Suspended outcomes are SKIPPED, and an all-suspended round writes nothing. Takes
    /// `outcomes` by value: their payloads are MOVED into the injected image message.
    async fn commit_tool_results(
        &mut self,
        tool_calls: &[ToolCall],
        outcomes: Vec<ToolExecutionOutcome>,
    ) -> anyhow::Result<()> {
        let tools = &self.tools;

        // Dedup key for attached images, lazily computed.
        let mut seen_data_uris: Option<std::collections::HashSet<String>> = None;

        // Image messages go AFTER the tool-result block, preserving OpenAI-compatible
        // contiguity (assistant(tool_calls) → tool_result_* → user(IMAGE_*)).
        let mut fresh_image_payloads: Vec<ImagePayload> = Vec::new();

        let mut db_messages = Vec::with_capacity(outcomes.len());
        for (call, outcome) in tool_calls.iter().zip(outcomes) {
            // Suspended: nothing to commit; the work is left for resume-completion.
            if outcome.suspended {
                continue;
            }
            let (output, fresh_payloads) = tool_result_content(
                tools,
                &call.name,
                outcome,
                &mut seen_data_uris,
                self.session.history(),
            )
            .await;
            fresh_image_payloads.extend(fresh_payloads);

            db_messages.push(ChatMessage::tool_result(&call.id, &output));
        }

        // Inject the round's fresh images as ONE tagged synthetic user message.
        if let Some(message) =
            injected_images_message(&self.agent_id, self.role, fresh_image_payloads)
        {
            db_messages.push(message);
        }

        // Nothing to commit (all suspended): the frame is already durable.
        if db_messages.is_empty() {
            return Ok(());
        }

        // Batch-persist (after any unpersisted tail) and push into memory.
        self.session
            .persist_messages(&self.agent_id, &db_messages)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to persist tool results: {e}"))?;

        Ok(())
    }

    /// Drain routed messages (ticket comments) into the session history, at the start of
    /// each tool round before the LLM call. Persisted AND pushed to memory so they survive
    /// a crash. On a persist failure they are still visible this turn and stay in the
    /// unpersisted tail the next persist flushes; a cancel before finalize drops them.
    /// No-op without a receiver (chat agents use the consumer loop).
    async fn drain_incoming_messages(&mut self) {
        let Some(rx) = &mut self.incoming_rx else {
            return;
        };

        let mut messages = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(job) => {
                    let content = match job.kind {
                        crate::agent::message_router::MessageKind::TicketComment => {
                            format_comment_message(&job.user_name, &job.content)
                        }
                        _ => job.content,
                    };
                    messages.push(crate::session::user_msg_with_ts(&content, None));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    // Channel closed: drop the receiver so we never retry.
                    self.incoming_rx = None;
                    break;
                }
            }
        }

        if messages.is_empty() {
            return;
        }

        // Fail-open: the comment is already in the board DB, so log and continue.
        self.persist_messages_fail_open(&messages).await;
    }

    /// Persist fail-open: on failure they still reach the memory and the tail.
    async fn persist_messages_fail_open(&mut self, messages: &[ChatMessage]) {
        if let Err(e) = self
            .session
            .persist_messages(&self.agent_id, messages)
            .await
        {
            tracing::warn!(
                agent_id = %self.agent_id,
                error = %e,
                "Failed to persist messages to session DB — continuing without persistence",
            );
            self.session.push_messages_unpersisted(messages);
        }
    }

    /// Inject board comments a resumed stage round has not seen: it starts with an empty
    /// message, never re-renders `<current-ticket>`, and would miss comments added after
    /// the boot snapshot. Re-reads the board and injects those absent from the history
    /// (content-based dedup). Gated to `resume` + empty `msg`, so the bounce-feedback path
    /// is excluded. Runs once per turn, fail-open persist like live delivery.
    async fn inject_outstanding_comments(&mut self, resume: bool, messages: &[&str]) {
        if !resume || messages.iter().any(|msg| !msg.is_empty()) {
            return;
        }

        let Some(ticket_id) = self.ticket.as_ref().map(|t| t.id.clone()) else {
            return;
        };

        // Fresh snapshot — the boot get_ticket is stale. Snapshot the board FIRST, then
        // drain: a comment routed after registration is already in the channel and deduped
        // against the post-drain history. One routed after the drain can still arrive
        // twice — the accepted no-watermark limitation.
        let comments = match crate::pipeline::board::store()
            .get_comments(&ticket_id)
            .await
        {
            Ok(comments) => comments,
            Err(e) => {
                tracing::warn!(
                    agent_id = %self.agent_id,
                    ticket = %ticket_id,
                    error = %e,
                    "Failed to read board comments for resume catch-up — skipping this round",
                );
                return;
            }
        };
        if comments.is_empty() {
            return;
        }

        self.drain_incoming_messages().await;

        let new_messages: Vec<_> = {
            let history = self.session.history();
            comments
                .iter()
                .filter(|c| {
                    !c.content.trim().is_empty()
                        && !history.iter().any(|m| m.content.contains(&c.content))
                })
                .map(|c| {
                    crate::session::user_msg_with_ts(
                        &format_comment_message(&c.role, &c.content),
                        None,
                    )
                })
                .collect()
        };
        if new_messages.is_empty() {
            return;
        }

        self.persist_messages_fail_open(&new_messages).await;
        tracing::info!(
            agent_id = %self.agent_id,
            role = %self.role,
            ticket = %ticket_id,
            count = new_messages.len(),
            "Injected outstanding board comments into resumed stage-agent session",
        );
    }

    /// Build a [`ChatRequest`] from the messages with the agent's current model, tools,
    /// reasoning-effort and provider routing — resolved each call, so a hot-reload is
    /// reflected immediately (reasoning effort is baked into role metadata).
    ///
    /// Every call site for the same conversation *must* use exactly the same parameter
    /// sources — model, reasoning_effort, tools (critically tools!) and routing — or the
    /// provider recomputes the whole KV-cache prefix.
    ///
    /// The request never carries more pictures than [`crate::session::SUMMARIZATION_IMAGE_COUNT`]:
    /// the session's recorded count gates the walk, and the oldest pictures step aside in place
    /// ([`crate::session::image_cap`]), so the conversation itself is untouched.
    fn build_chat_request(
        &self,
        mut messages: Vec<ChatMessage>,
        purpose: &'static str,
    ) -> ChatRequest {
        crate::session::image_cap::apply(&mut messages, Some(self.session.image_count()));
        ChatRequest {
            meta: Some(crate::ChatRequestMeta {
                purpose,
                agent_id: self.agent_id.clone(),
                role: self.role.as_str().to_string(),
                workspace: self.workspace.name.clone(),
                ticket_id: self.ticket.as_ref().map(|t| t.id.clone()),
            }),
            ..chat_request(self.role, Some(self.tool_specs.clone()), messages)
        }
    }

    /// Extract a structured verdict with the hardened scoped retry loop. KV-cache:
    /// params come from [`Self::build_chat_request`], so attempts are byte-identical
    /// except the parse-failure re-prompt. `validate` runs fail-closed (e.g. score ∈
    /// [0,10]) and a rejected value is re-prompted as a parse failure — `None` for
    /// plain extraction. `policy_override` gives fail-open callers a shorter schedule.
    pub(crate) async fn extract_verdict<T: serde::de::DeserializeOwned>(
        &self,
        extraction_prompt: &str,
        validate: Option<&crate::ExtractionValidator<T>>,
        policy_override: Option<&crate::retry::RetryPolicy>,
    ) -> Result<T, crate::retry::RetryExhausted> {
        let _activity = self.activity_guard("extracting");
        let params = self.build_chat_request(vec![], "extraction");
        crate::agent::extraction::retry_extract_structured_scoped(
            self.session.history(),
            extraction_prompt,
            &params,
            validate,
            policy_override,
        )
        .await
    }

    /// Summarise the session history. KV-cache: see [`Self::build_chat_request`] — the
    /// request is built once and re-sent verbatim, so every attempt is byte-identical.
    ///
    /// A tool call is not a summary: it carries no text, so it is re-asked
    /// (`SUMMARIZE_ATTEMPTS` in total, each a successful `summarize` request of its
    /// own) and never executed; text alongside a tool call is the summary. Exhaustion
    /// raises the empty-response error, so [`Self::maybe_summarize`] fails open.
    pub(crate) async fn summarize(&self) -> anyhow::Result<String> {
        let _activity = self.activity_guard("summarizing");
        let mut history = self.session.history().to_vec();
        history.push(crate::ChatMessage::user(self.role.summary_prompt()));

        let request = self.build_chat_request(history.clone(), "summarize");
        let policy = crate::retry::RetryPolicy::current();

        for _ in 1..=SUMMARIZE_ATTEMPTS {
            let chat_resp = crate::retry::agent_chat(request.clone(), &policy)
                .await
                .with_context(|| format!("summarization LLM call {RETRY_EXHAUSTION_MARKER}"))?;

            // Reasoning-only stop: the same bounded continuation as the agent loop; on
            // exhaustion, fail open below with the full history.
            let chat_resp = self
                .recover_if_reasoning_only_stop(
                    history.clone(),
                    chat_resp,
                    Self::SUMMARIZE_REASONING_RECOVERY,
                )
                .await?;

            if let Some(ref u) = chat_resp.usage {
                tracing::debug!(
                    input_tokens = u.input_tokens,
                    cached_input_tokens = u.cached_input_tokens,
                    output_tokens = u.output_tokens,
                    "Summarization token usage",
                );
            }

            if let Some(summary_text) = chat_resp.text.filter(|t| !t.trim().is_empty()) {
                return Ok(crate::util::truncate(&summary_text, 32_000));
            }
            // No summary text — a tool-call answer. Re-ask (or leave the loop
            // with the error when the budget is spent).
        }

        anyhow::bail!("summarization produced empty response")
    }

    /// Check the summarization triggers and summarise if necessary: the
    /// provider-reported session token length over [`SUMMARIZATION_THRESHOLD`], or
    /// [`SUMMARIZATION_IMAGE_COUNT`] distinct data-URI images. The token length is the
    /// REAL value of the last successful agent call, loaded at session init; `None`
    /// (never recorded — approved no-backfill — or cleared by the last compaction)
    /// only disables the token trigger. A FAILED call never updates it.
    ///
    /// KV-cache preservation: [`Agent::summarize`] keeps all parameters identical, so
    /// the cached prefix stays reusable.
    async fn maybe_summarize(&mut self, turn_start: usize) {
        let token_length = self.session.token_length();
        let image_count = session_image_marker_count(self.session.history());
        tracing::debug!(
            agent_id = %self.agent_id,
            role = %self.role,
            token_length,
            image_count,
            "Session token length / image count",
        );
        let Some(trigger) = summarization_trigger(token_length, image_count) else {
            return;
        };
        tracing::info!(
            agent_id = %self.agent_id,
            role = %self.role,
            token_length,
            image_count,
            "Session exceeded summarization trigger ({trigger})",
        );
        match self.summarize().await {
            Ok(summary) => {
                self.session
                    .apply_summary(
                        &self.agent_id,
                        &summary,
                        &self.workspace,
                        &self.role,
                        self.ticket.as_ref(),
                        self.is_admin,
                        &self.user_name,
                        turn_start,
                    )
                    .await;
            }
            Err(e) => {
                tracing::warn!(
                    error_chain = %crate::util::failure_detail(&format!("{e:#}"), "summarization failure"),
                    "Summarization failed — continuing with full history"
                );
            }
        }
    }

    /// Compact a long-lived chat role's session (Assistant, Manager, custom) right
    /// after its reply was delivered, so the user's next message does not wait. Runs
    /// on the per-session consumer loop — the serialized path that already handles one
    /// turn at a time here.
    ///
    /// [`Self::maybe_summarize`] keeps the same limit, triggers, prompt, result and
    /// retention policy, and its pre-turn call in [`Self::work`] stays the safety net
    /// for everything this skips — settled tool results, cancellation, shutdown.
    /// `resume` needs no clause: this loop runs every job as a plain turn.
    ///
    /// A session over a trigger at delivery compacts whatever put it there, even when
    /// its own compaction failed to persist; a repeat costs at most one extra
    /// summarization call per turn, accepted over the wait it removes.
    pub(crate) async fn maybe_summarize_after_delivery(&mut self) {
        if !self.role.is_long_lived_chat()
            || self.turn_state.settled_tool_results
            || self.is_cancelled()
            || crate::shutdown::aborting()
        {
            return;
        }
        // The turn is over: nothing of it is still owed a full read.
        self.maybe_summarize(self.session.history().len()).await;
    }
}

/// Which trigger fires, if any: strict `>` for the token threshold but `>=` for the
/// image count — the asymmetry is deliberate.
#[must_use]
fn summarization_trigger(token_length: Option<u64>, image_count: usize) -> Option<&'static str> {
    if token_length.is_some_and(|t| t > crate::session::SUMMARIZATION_THRESHOLD) {
        Some("token threshold")
    } else if image_count >= crate::session::SUMMARIZATION_IMAGE_COUNT {
        Some("image count")
    } else {
        None
    }
}

/// Parameters for a reasoning-only-stop recovery: the telemetry purpose tag and the
/// context attached to the exhaustion error. Named fields, not two adjacent
/// `&'static str` arguments, so they cannot be silently swapped at a call site.
struct ReasoningOnlyStopRecovery {
    purpose: &'static str,
    exhausted_ctx: &'static str,
}

/// Result of preparing an assistant turn from the LLM response.
struct PreparedAssistantTurn {
    display_text: String,
    tool_calls: Vec<ToolCall>,
    history_content: String,
    /// The raw response text, the faithful carrier of accompanying text in tool rounds
    /// (where `history_content` is a JSON payload and `display_text` is empty).
    /// Reasoning never reaches this builder, so it excludes it by construction.
    raw_text: String,
}

/// The "reasoning-only stop" class: empty content (think tags stripped — they are
/// reasoning) with no parsed tool calls, REGARDLESS of finish reason. Tool-call turns
/// are excluded by the tool-call check. Degenerate members — fully empty responses and
/// think-tag-only content — are consciously in class: the continuation carries just the
/// nudge, and a silent `Ok("")` now resolves there or fails the turn safely (never CoT-as-text).
#[must_use]
fn is_reasoning_only_stop(response: &ChatResponse) -> bool {
    response.tool_calls.is_empty() && response.text.as_deref().is_none_or(|t| t.trim().is_empty())
}

/// Persist one continuation-attempt failure row when the operation is
/// metadata-tracked (fail-open); the duration is measured from `attempt_started`.
fn record_continuation_failure(
    operation: Option<&crate::stats::LlmOperationCtx>,
    attempt: u32,
    rec: &crate::retry::RetryFailureRecord,
    attempt_started: Instant,
) {
    if let Some(op) = operation {
        crate::stats::record_llm_attempt_failure(op, attempt, rec, Some(attempt_started));
    }
}

/// Context for a failed LLM step: the run-local round counter plus session depth — a
/// resumed long-lived session must not read as freshly started. The token segment is
/// omitted when unknown.
fn llm_step_failure_context(session: &Session, tool_rounds: usize) -> String {
    let tokens = session
        .token_length()
        .map_or_else(String::new, |t| format!(", {t} tokens"));
    format!(
        "LLM step failed at tool round {tool_rounds} (run-local counter; \
         session history: {} messages{tokens})",
        session.history().len(),
    )
}

fn prepare_assistant_turn(response: ChatResponse) -> PreparedAssistantTurn {
    let response_text = response.text_or_empty().to_string();
    let raw_text = response_text.clone();
    let tool_calls = response.tool_calls;
    let reasoning = response.reasoning.as_ref();

    // The payload captures the model's original response. Reasoning-only stops never
    // reach here — llm_call recovers them before any persistence or display.
    let json_payload =
        assistant_replay_payload(Some(&response_text), &tool_calls, reasoning).to_string();

    // Plain answer, reasoning+text, or tool calls. Reasoning-only responses are the
    // recovered class and never arrive here.
    let (display_text, history_content) = match (tool_calls.is_empty(), reasoning.is_some()) {
        // Plain final answer — both display and history use response text directly.
        (true, false) => (response_text.clone(), response_text),
        // Reasoning alongside visible text — show the text to the user,
        // persist structured JSON payload with content + reasoning fields.
        (true, true) => (response_text, json_payload),
        // Tool calls — nothing to display, persist structured JSON payload.
        (false, _) => (String::new(), json_payload),
    };

    PreparedAssistantTurn {
        display_text,
        tool_calls,
        history_content,
        raw_text,
    }
}

/// Rejection message for a `sleep` call made in a round with accompanying
/// raw text: the text is never delivered (tool rounds are never displayed)
/// and the successful sleep would end the run before anything is sent.
const SLEEP_WITH_TEXT_REJECTION: &str = "sleep must be called alone: your text was NOT delivered — text written in any round that contains a tool call never reaches the user. If that text was meant for the user, resend it as a plain-text round with no tool calls and no sleep call: that round ends your turn and delivers it. If it was not meant for the user, call sleep alone, with no text and no other tool call.";

/// Rejection message for a `sleep` call bundled with other tool calls.
const SLEEP_BUNDLED_REJECTION: &str = "sleep must be called alone: it was bundled with other tool calls. Call sleep in its own round with no other tool calls — and no text, since text written in a tool round is never delivered to the user.";

/// Sleep-delivery guard: a `sleep` call must run alone in a text-free round, or it
/// swallows the accompanying text or ends the run silently. `None` when legal.
#[must_use]
fn sleep_rejection_reason(has_text: bool, bundled: bool) -> Option<&'static str> {
    if has_text {
        Some(SLEEP_WITH_TEXT_REJECTION)
    } else if bundled {
        Some(SLEEP_BUNDLED_REJECTION)
    } else {
        None
    }
}

/// Round-scoped options for parallel-round members: `round_ts` pins the first user
/// message's timestamp (byte-identical task messages across members), and
/// `first_call_notify` is the leader-stagger signal, fired after the leader's first
/// LLM call — a leader exceeding the stagger bound misses the cached prefix.
#[derive(Clone)]
pub(crate) struct RoundOpts {
    pub(crate) round_ts: String,
    pub(crate) first_call_notify: Option<std::sync::Arc<tokio::sync::Notify>>,
}

/// Fail-open bound on the leader-stagger wait — a safety cap, not the expected wait
/// (first calls take ~4-6 s). Overridable via env for tuning, like
/// [`crate::tools::analyze::round_timeout`].
const DEFAULT_STAGGER_WAIT_SECS: u64 = 8;

fn leader_stagger_wait() -> std::time::Duration {
    crate::util::env_duration_secs("MAHBOT_STAGGER_WAIT_SECS", DEFAULT_STAGGER_WAIT_SECS)
}

/// Spawn a round's members with a leader-stagger: member 0 starts immediately, the
/// rest once the leader's first LLM call completes or the bounded wait elapses —
/// whichever first. The wait is fail-open (a failed or never-starting leader still
/// releases the followers) and interrupted by shutdown or drain.
///
/// Single-member rounds are a no-op stagger, and boot-resumed rounds (`resume = true`)
/// skip it: their sessions already diverged, so the wait would only add latency.
/// The round timestamp is rendered once and every member gets a ready-made
/// [`RoundOpts`] (only the leader carries the first-call signal, which has a single
/// waiter, so `notify_one()` releases all followers and its stored permit covers a
/// fire-before-wait race `notify_waiters()` would lose).
pub(crate) async fn spawn_staggered_round<T, Fut, F>(
    members: Vec<F>,
    resume: bool,
) -> Vec<tokio::task::JoinHandle<T>>
where
    T: Send + 'static,
    Fut: std::future::Future<Output = T> + Send + 'static,
    F: FnOnce(RoundOpts) -> Fut + Send + 'static,
{
    let mut members = members.into_iter();
    let Some(leader) = members.next() else {
        return Vec::new();
    };
    let followers: Vec<F> = members.collect();
    let round_ts = crate::session::render_timestamp();
    if resume || followers.is_empty() {
        // Resumed or single-member: no stagger, spawn everything immediately.
        let opts = RoundOpts {
            round_ts,
            first_call_notify: None,
        };
        let mut handles = Vec::with_capacity(followers.len() + 1);
        handles.push(tokio::spawn(leader(opts.clone())));
        handles.extend(followers.into_iter().map(|m| tokio::spawn(m(opts.clone()))));
        return handles;
    }
    let notify = std::sync::Arc::new(tokio::sync::Notify::new());
    let mut handles = vec![tokio::spawn(leader(RoundOpts {
        round_ts: round_ts.clone(),
        first_call_notify: Some(notify.clone()),
    }))];
    let follower_opts = RoundOpts {
        round_ts,
        first_call_notify: None,
    };
    let shutdown = crate::shutdown::shutdown_token();
    tokio::select! {
        () = notify.notified() => {}
        () = tokio::time::sleep(leader_stagger_wait()) => {}
        () = shutdown.cancelled() => {}
        () = crate::shutdown::drain_wait() => {}
    }
    // Drain/shutdown arms still spawn the followers: their Session::init append runs
    // before the aborting() bail, pre-seeding the session so boot resume's
    // has_session guard skips the re-append (as on the leader's path).
    handles.extend(
        followers
            .into_iter()
            .map(|m| tokio::spawn(m(follower_opts.clone()))),
    );
    handles
}

/// Everything a run releases for its own temporary use when it ends: its spill files,
/// its per-agent computer registry entries, and the chrome sessions it opened. A `Drop`
/// rather than trailing statements, because a run cut off mid-flight never reaches one;
/// the chrome part is only handed over (the `chrome-run-releases` task outlives the run).
///
/// Accepted limit: a run still working at process exit, or killed outright (`exit(0)`
/// bypasses destructors by design), queues nothing. The reclaim pass still closes its
/// tabs — it sweeps the browser's live `agent-tab-*` groups ([`crate::tools::chrome_tabs`]'s
/// honest limits) and closes any namespace no guard claims: not a live run's, not inside a
/// record's hold, not one a queued or parked record claims, not one durable state names for
/// a non-terminalized job ([`crate::tools::chrome_release::ProtectedNamespaces`]). It asks
/// the browser at a run end that queued names, at a revisit a leftover owes, and — when the
/// durable record file handed it something — at process start.
struct RunEndCleanup {
    agent_id: String,
    chrome: std::sync::Arc<crate::tools::chrome::ChromeRunSessions>,
    /// Whether [`RunEndCleanup::ended`] held the release back for a resumed segment.
    held: bool,
}

impl RunEndCleanup {
    fn new(
        agent_id: String,
        chrome: std::sync::Arc<crate::tools::chrome::ChromeRunSessions>,
    ) -> Self {
        Self {
            agent_id,
            chrome,
            held: false,
        }
    }

    /// Record how the run ended — which ends hold the release back is
    /// [`crate::tools::chrome_release::holds_run_end`]'s rule.
    fn ended(&mut self, classification: &str) {
        self.held = crate::tools::chrome_release::holds_run_end(classification);
    }
}

impl Drop for RunEndCleanup {
    fn drop(&mut self) {
        // Owner-deletes-at-end: remove this run's spill files (the run is over).
        crate::tools::shell::cleanup_agent_spills(&self.agent_id);
        // Drop what this turn's computer use owns; app/window ids last for the session.
        crate::tools::computer::cleanup_turn_state(&self.agent_id);
        // Hand the chrome sessions to the release queue under the run's own agent id,
        // so a leftover report names its run; the tracker's own `Drop` unregisters the
        // namespace once its last owner goes.
        crate::tools::chrome_release::queue_run_session_release(
            &self.chrome,
            self.held,
            &self.agent_id,
        );
    }
}

/// Core agent lifecycle: create the agent (auto-registered with its own
/// CancellationToken), run work, handle cancellation and errors. Returns the agent
/// even on failure, with the response only on success (an error is already logged).
///
/// **Cancellation safety**: a completed `Ok` result is never discarded on cancel
/// cause — downstream consumers (pipeline finalizers, the message router) decide how
/// a cancelled run's result is handled.
///
/// `user_name` and `channel` identify the message's origin, so tools route sub-agent
/// results to the right user. `launch` is the launch this run executes — `Some` for a
/// branch role, `None` for built-ins. `messages` arrives in arrival order (see
/// [`Agent::work`]).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn run_agent(
    agent_id: String,
    role: crate::Role,
    ws: &crate::Workspace,
    ticket: Option<&crate::pipeline::board::Ticket>,
    messages: &[&str],
    user_name: String,
    channel: String,
    is_admin: bool,
    incoming_rx: Option<
        tokio::sync::mpsc::UnboundedReceiver<crate::agent::message_router::AgentJob>,
    >,
    resume: bool,
    round: Option<RoundOpts>,
    parent_key: Option<crate::agent::registry::ParentKey>,
    parent_label: Option<String>,
    launch: Option<crate::Launch>,
) -> (Agent, Option<String>) {
    // Unregister from the message router on EVERY exit path, a mid-work panic
    // included — a leaked entry keeps routing comments to a dead agent. Only for
    // caller-registered paths: the consumer_loop path owns its entry via route()'s
    // catch_unwind cleanup and keeps it across jobs. Idempotent.
    struct UnregisterOnDrop(String);
    impl Drop for UnregisterOnDrop {
        fn drop(&mut self) {
            crate::agent::message_router::unregister_agent(&self.0);
        }
    }
    let _router_guard = incoming_rx
        .is_some()
        .then(|| UnregisterOnDrop(agent_id.clone()));
    let mut agent = Agent::new(
        agent_id,
        role,
        ws,
        ticket.cloned(),
        user_name,
        channel,
        is_admin,
        parent_key,
        parent_label,
        launch,
    );
    agent.incoming_rx = incoming_rx;
    if let Some(round) = round {
        agent.round_ts = Some(round.round_ts);
        agent.first_call_notify = round.first_call_notify;
    }
    // Everything this run owns is released by this guard, on every path.
    let mut run_end = RunEndCleanup::new(
        agent.agent_id.clone(),
        std::sync::Arc::clone(&agent.chrome_sessions),
    );
    let result = agent.work(messages, resume).await;

    // Absorb the product's own writes into a personal folder before the turn ends, or
    // a later change report blames the user for them. Every returning turn passes here;
    // the gate matches the report's.
    crate::session::personal_changes::refresh_after_turn(
        &agent.role,
        &agent.workspace,
        agent.turn_recorded,
    )
    .await;

    // A completed Ok is kept regardless of cancel cause — the token may have fired
    // just as work() finished.
    match result {
        Ok(response) => (agent, Some(response)),
        Err(e) => {
            // Capture the full chain, so dispatchers report the real cause instead of
            // a generic placeholder.
            agent.failure = Some(format!("{e:#}"));
            agent.failure_class = failure_class_from_error(&e);
            let classification = failure_classification(&agent, Some(&e));
            agent.run_failure = Some(if is_cooperative_stop(classification) {
                crate::RunFailure::CooperativeStop
            } else {
                crate::RunFailure::Technical
            });
            // The chrome release's hold (`holds_run_end`'s rule).
            run_end.ended(classification);
            let error_chain = crate::util::failure_detail(&format!("{e:#}"), "agent failure log");
            // Cooperative ends and a deliberate cancel get the same treatment; the
            // classification above decides which log fires.
            if is_cooperative_stop(classification) {
                tracing::debug!(
                    agent_id = %agent.agent_id,
                    workspace = %ws.name,
                    role = %role,
                    ticket = ticket.map(|t| t.id.as_str()),
                    classification,
                    error_chain,
                    "Agent stopped by a non-failure cancellation"
                );
            } else {
                tracing::error!(
                    agent_id = %agent.agent_id,
                    workspace = %ws.name,
                    role = %role,
                    ticket = ticket.map(|t| t.id.as_str()),
                    classification,
                    error_chain,
                    "Agent failed"
                );
            }
            (agent, None)
        }
    }
}

/// Default dispatch: no ticket, empty user/channel, no inbox. `resume` selects the
/// slot-resume primitive (empty committed-tail continuation, summarization skipped),
/// `false` for fresh dispatches. `parent_key`/`parent_label` are the Running Agents
/// grouping key and its presentational label; `None` for workspace singletons.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn run_default_agent(
    agent_id: &str,
    role: crate::Role,
    ws: &crate::Workspace,
    message: &str,
    resume: bool,
    round: Option<RoundOpts>,
    parent_key: Option<crate::agent::registry::ParentKey>,
    parent_label: Option<String>,
) -> (Agent, Option<String>) {
    run_agent(
        agent_id.to_string(),
        role,
        ws,
        None,
        &[message],
        String::new(),
        String::new(),
        false,
        None,
        resume,
        round,
        parent_key,
        parent_label,
        None,
    )
    .await
}

/// The typed provider failure class from an error chain, when it carries a
/// [`RetryExhausted`]. `None` means the error is not provider-classified (runtime
/// errors, panics, plain I/O) — callers treat that as a genuine non-provider failure.
pub(crate) fn failure_class_from_error(
    error: &anyhow::Error,
) -> Option<crate::retry::FailureClass> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<crate::retry::RetryExhausted>())
        .map(|exhausted| exhausted.final_class)
}

/// Labels meaning the run stopped cooperatively rather than failing technically.
const COOPERATIVE_STOP_CLASSES: [&str; 4] = ["drain", "shutdown", "pause", "internal_cancel"];

fn is_cooperative_stop(classification: &str) -> bool {
    COOPERATIVE_STOP_CLASSES.contains(&classification)
}

/// Stable failure classification for the agent-failure log.
///
/// Order matters: the global shutdown token fires on any stop request (SIGTERM/SIGINT,
/// the dashboard's quit, a console or session-end, a self-update) and on a completed
/// drain, so it is checked first — shutdown must never be mislabeled an internal
/// cancel. `error` is the raw source when present (`run_agent`'s `Err` arm), `None`
/// only in tests. A [`RetryExhausted`] carries the granular
/// [`crate::retry::FailureClass`]; everything else falls back to `runtime`. The
/// cooperative labels live in [`COOPERATIVE_STOP_CLASSES`], consumed by both the exit
/// log and [`Agent::failed_technically`].
fn failure_classification(agent: &Agent, error: Option<&anyhow::Error>) -> &'static str {
    if crate::shutdown::is_draining() {
        // The drain cut the round short — not a failure. Checked before the token, so
        // a drained agent is never mislabeled cancelled/shutdown (no failure emoji,
        // no exit rollback).
        "drain"
    } else if crate::shutdown::shutdown_token().is_cancelled() {
        "shutdown"
    } else if agent.is_paused_frozen() {
        // A strictly-frozen run — not a failure. The immutable bail-time snapshot is
        // authoritative, not the live pause_stop an unpause can clear.
        "pause"
    } else if agent.is_cancelled() {
        "internal_cancel"
    } else if let Some(class) = error.and_then(failure_class_from_error) {
        class.label()
    } else {
        "runtime"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;
    use crate::tools::chrome_release;
    use crate::util::test::{FakeProvider, install_fake_provider, install_test_retry_policy};
    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    struct TestTool {
        output: String,
        scrub: bool,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &'static str {
            if self.scrub {
                "always_scrub"
            } else {
                "never_scrub"
            }
        }

        fn description(&self) -> String {
            "test".into()
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }

        async fn execute(
            &self,
            _ws: &crate::Workspace,
            _args: serde_json::Value,
        ) -> anyhow::Result<String> {
            Ok(self.output.clone())
        }

        fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
            self.scrub
        }
    }

    /// Secret-like line that `scrub_credentials` always redacts.
    const SCRUBBABLE_LINE: &str = "API_KEY=sk-1234567890abcdef";

    fn make_agent(tools: Vec<Box<dyn Tool>>) -> Agent {
        make_agent_with_role(tools, crate::Role::Engineer)
    }

    fn make_agent_with_role(tools: Vec<Box<dyn Tool>>, role: crate::Role) -> Agent {
        let tool_specs = tools.iter().map(|t| t.spec()).collect();
        Agent {
            agent_id: "test-agent".into(),
            role,
            session: Session::default(),
            workspace: std::sync::Arc::new(crate::Workspace::default()),
            tools,
            tool_specs,
            cancel_token: CancellationToken::new(),
            pause_stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            paused_frozen: std::sync::atomic::AtomicBool::new(false),
            ticket: None,
            generation: 0,
            tool_stats: std::sync::Mutex::new(Vec::new()),
            user_name: String::new(),
            channel: String::new(),
            is_admin: false,
            parent_key: None,
            parent_label: None,
            incoming_rx: None,
            round_ts: None,
            first_call_notify: None,
            failure: None,
            failure_class: None,
            run_failure: None,
            sleep_ended: false,
            turn_recorded: false,
            turn_state: crate::TurnState::default(),
            background_sessions: std::sync::Arc::new(
                crate::tools::shell::BackgroundSessions::default(),
            ),
            chrome_sessions: std::sync::Arc::new(crate::tools::chrome::ChromeRunSessions::default()),
            interrupt_token: CancellationToken::new(),
        }
    }

    /// Construct an agent with an explicit workspace and id, for path-resolving tools.
    fn make_agent_on(tools: Vec<Box<dyn Tool>>, agent_id: &str, ws: crate::Workspace) -> Agent {
        let mut agent = make_agent_with_role(tools, crate::Role::Engineer);
        agent.agent_id = agent_id.to_string();
        agent.workspace = std::sync::Arc::new(ws);
        agent
    }

    #[test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    fn failure_classification_recovers_retry_exhaustion() {
        // Serialize against the drain-flag writers: failure_classification reads the
        // process-global flag first. Mirror llm_call's error construction — via
        // .context, not string flattening — so the granular FailureClass survives.
        let exhausted = crate::retry::RetryExhausted::with_last_raw(
            vec![],
            crate::retry::FailureClass::NonRetryable,
            None,
        );
        let err = anyhow::Error::new(exhausted).context("LLM call exhausted retry budget");
        // The {:#} chain must keep the marker engineer_failure_notice matches.
        assert!(format!("{err:#}").contains("exhausted retry budget"));

        let agent = make_agent(vec![]);
        assert_eq!(
            failure_classification(&agent, Some(&err)),
            "non_retryable",
            "RetryExhausted final_class must be recovered from the chain",
        );

        // Non-retry-loop errors (I/O, tool, panic) fall back to runtime.
        let runtime_err = anyhow::anyhow!("tool panicked");
        assert_eq!(
            failure_classification(&agent, Some(&runtime_err)),
            "runtime"
        );

        // An internal cancellation fires the generic token and must classify as an
        // internal cancel — every non-shutdown/non-pause stop is one.
        let internally_cancelled = make_agent(vec![]);
        internally_cancelled.cancel_token.cancel();
        assert_eq!(
            failure_classification(&internally_cancelled, Some(&runtime_err)),
            "internal_cancel"
        );
        assert_eq!(
            failure_classification(&internally_cancelled, None),
            "internal_cancel"
        );
    }

    /// The graceful-drain cut classifies as "drain", NOT cancelled/shutdown/failure: no
    /// failure emoji, and the dispatch tails skip the exit-time rollback. Serialized —
    /// the drain flag is process-global.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn failure_classification_recognizes_drain() {
        let agent = make_agent(vec![]);
        // Drain flag unset: normal classification.
        assert_eq!(failure_classification(&agent, None), "runtime");
        // Set the drain flag for the duration of the assertion.
        crate::shutdown::drain_begin();
        assert_eq!(
            failure_classification(&agent, None),
            "drain",
            "a drained agent must classify as drain, never failure",
        );
        // A per-agent cancel during drain still classifies as drain (checked first).
        agent.cancel_token.cancel();
        assert_eq!(failure_classification(&agent, None), "drain");
        // Restore isolation: the drain flag is process-global, and the token must never
        // be fired from a test — it would cancel every parallel retry loop.
        crate::shutdown::drain_clear();
    }

    /// Shared helper: an agent with a TestTool, executed and asserted for scrubbing.
    async fn assert_scrubbed(should_scrub: bool) {
        let tool: Box<dyn Tool> = Box::new(TestTool {
            output: SCRUBBABLE_LINE.into(),
            scrub: should_scrub,
        });
        let name = tool.name();
        let agent = make_agent(vec![tool]);
        let out = agent.execute_tool(name, serde_json::json!({})).await;

        assert!(out.success, "{name} should succeed");
        if should_scrub {
            assert!(out.output.contains("[REDACTED]"), "{name} should redact");
            assert!(
                !out.output.contains("abcdef"),
                "{name} should not leak original"
            );
        } else {
            assert!(
                !out.output.contains("[REDACTED]"),
                "{name} should not redact"
            );
            assert!(
                out.output.contains(SCRUBBABLE_LINE),
                "{name} should preserve output"
            );
        }
    }

    /// The scrub decision belongs to the tool: one that opts out returns its output
    /// verbatim, one that opts in has the secret-like line redacted.
    #[tokio::test]
    async fn tool_output_scrubbing_follows_the_tool_setting() {
        for should_scrub in [false, true] {
            assert_scrubbed(should_scrub).await;
        }
    }

    /// A media-generation test tool with a configurable [`Tool::media_marker`], for
    /// [`extract_media_from_outcomes`] without real generation.
    struct MediaTestTool {
        name: &'static str,
        marker: &'static str,
    }

    #[async_trait]
    impl Tool for MediaTestTool {
        fn name(&self) -> &'static str {
            self.name
        }

        fn description(&self) -> String {
            "media test tool".into()
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }

        async fn execute(
            &self,
            _ws: &crate::Workspace,
            _args: serde_json::Value,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn media_marker(&self) -> Option<&'static str> {
            Some(self.marker)
        }
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn extract_media_outcomes_consolidated() {
        enum ToolDef {
            /// [`NonMedia`] is a unit variant: only [`Tool::media_marker`] is consulted,
            /// never [`Tool::execute`], so the output value has no behavioral effect.
            Media {
                name: &'static str,
                marker: &'static str,
            },
            NonMedia,
        }

        struct OutcomeDef {
            output: &'static str,
            success: bool,
        }

        struct TestCase {
            name: &'static str,
            msg: &'static str,
            tools: Vec<ToolDef>,
            outcomes: Vec<OutcomeDef>,
            expected: Vec<(&'static str, &'static str)>,
        }

        let cases = vec![
            TestCase {
                name: "parses_valid_marker",
                msg: "valid marker with success=true should extract the path",
                tools: vec![ToolDef::Media {
                    name: "image_gen",
                    marker: "[IMAGE:",
                }],
                outcomes: vec![OutcomeDef {
                    output: "[IMAGE:/tmp/img.png]",
                    success: true,
                }],
                expected: vec![("[IMAGE:", "/tmp/img.png")],
            },
            // A bare marker prefix yields an empty path, which the filter rejects.
            TestCase {
                name: "skips_malformed_marker",
                msg: "malformed marker should be skipped",
                tools: vec![ToolDef::Media {
                    name: "image_gen",
                    marker: "[IMAGE:",
                }],
                outcomes: vec![OutcomeDef {
                    output: "description text [IMAGE:",
                    success: true,
                }],
                expected: vec![],
            },
            TestCase {
                name: "skips_empty_marker",
                msg: "empty marker '[IMAGE:]' should be skipped",
                tools: vec![ToolDef::Media {
                    name: "image_gen",
                    marker: "[IMAGE:",
                }],
                outcomes: vec![OutcomeDef {
                    output: "[IMAGE:]",
                    success: true,
                }],
                expected: vec![],
            },
            TestCase {
                name: "skips_no_closing_bracket_non_empty_path",
                msg: "output with '[IMAGE:bogus' (no closing bracket, non-empty path) should be skipped",
                tools: vec![ToolDef::Media {
                    name: "image_gen",
                    marker: "[IMAGE:",
                }],
                outcomes: vec![OutcomeDef {
                    output: "oops [IMAGE:bogus",
                    success: true,
                }],
                expected: vec![],
            },
            TestCase {
                name: "skips_non_media_tool",
                msg: "non-media tool should not be inspected for media markers",
                tools: vec![ToolDef::NonMedia],
                outcomes: vec![OutcomeDef {
                    output: "[IMAGE:path]",
                    success: true,
                }],
                expected: vec![],
            },
            TestCase {
                name: "skips_failed_outcome",
                msg: "failed outcomes should not produce media paths",
                tools: vec![ToolDef::Media {
                    name: "image_gen",
                    marker: "[IMAGE:",
                }],
                outcomes: vec![OutcomeDef {
                    output: "[IMAGE:/tmp/img.png]",
                    success: false,
                }],
                expected: vec![],
            },
            TestCase {
                name: "handles_mixed_tools",
                msg: "mixed tools with valid outcomes should extract only media paths",
                tools: vec![
                    ToolDef::Media {
                        name: "image_gen",
                        marker: "[IMAGE:",
                    },
                    ToolDef::NonMedia,
                    ToolDef::Media {
                        name: "video_gen",
                        marker: "[VIDEO:",
                    },
                ],
                outcomes: vec![
                    OutcomeDef {
                        output: "[IMAGE:/tmp/img.png]",
                        success: true,
                    },
                    OutcomeDef {
                        output: "non-media output",
                        success: true,
                    },
                    OutcomeDef {
                        output: "[VIDEO:/tmp/vid.mp4]",
                        success: true,
                    },
                ],
                expected: vec![("[IMAGE:", "/tmp/img.png"), ("[VIDEO:", "/tmp/vid.mp4")],
            },
        ];

        for case in cases {
            let tools: Vec<Box<dyn Tool>> = case
                .tools
                .iter()
                .map(|t| match t {
                    ToolDef::Media { name, marker } => {
                        Box::new(MediaTestTool { name, marker }) as Box<dyn Tool>
                    }
                    ToolDef::NonMedia => Box::new(TestTool {
                        // The output value is irrelevant: only media_marker() is consulted.
                        output: String::new(),
                        scrub: false,
                    }) as Box<dyn Tool>,
                })
                .collect();

            let calls: Vec<ToolCall> = case
                .tools
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let name = match t {
                        ToolDef::Media { name, .. } => *name,
                        ToolDef::NonMedia => "never_scrub",
                    };
                    ToolCall {
                        id: (i + 1).to_string(),
                        name: name.to_string(),
                        arguments: serde_json::json!({}),
                    }
                })
                .collect();

            let outcomes: Vec<ToolExecutionOutcome> = case
                .outcomes
                .iter()
                .map(|o| ToolExecutionOutcome {
                    output: o.output.to_string(),
                    success: o.success,
                    image_payloads: Vec::new(),
                    text_is_content: false,
                    preserve_text: false,
                    suspended: false,
                    ends_turn: false,
                })
                .collect();

            let expected: Vec<(&'static str, String)> = case
                .expected
                .iter()
                .map(|(n, p)| (*n, p.to_string()))
                .collect();

            let result = extract_media_from_outcomes(&tools, &calls, &outcomes);
            assert_eq!(result, expected, "case '{}': {}", case.name, case.msg);
        }
    }

    #[tokio::test]
    async fn finalize_session_skipped_when_cancelled() {
        let mut agent = make_agent(vec![]);
        agent.cancel_token.cancel();
        let result = agent.finalize_session().await;
        assert!(
            result.is_ok(),
            "finalize_session should return Ok when cancelled, \
             skipping the 'no assistant message' warning"
        );
    }

    #[tokio::test]
    async fn execute_tool_skips_when_cancelled() {
        let tool: Box<dyn Tool> = Box::new(TestTool {
            output: "should not run".into(),
            scrub: false,
        });
        let name = tool.name();
        let agent = make_agent(vec![tool]);
        agent.cancel_token.cancel();
        let out = agent.execute_tool(name, serde_json::json!({})).await;

        assert!(!out.success, "cancelled agent should not execute tool");
        assert!(
            out.output.contains("Agent cancelled"),
            "failure output should mention cancellation: {}",
            out.output
        );
    }

    #[tokio::test]
    async fn task_locals_propagate_to_parallel_tool_execution() {
        /// Reads CURRENT_TOOL_USER_NAME and CURRENT_TOOL_CHANNEL from the task-locals.
        struct ReadTaskLocalsTool;
        #[async_trait]
        impl Tool for ReadTaskLocalsTool {
            fn name(&self) -> &'static str {
                "read_task_locals"
            }
            fn description(&self) -> String {
                "test tool that reads task-local user context".into()
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }
            async fn execute(
                &self,
                _ws: &crate::Workspace,
                _args: serde_json::Value,
            ) -> anyhow::Result<String> {
                let user_name = tool_user_name();
                let channel = tool_channel();
                Ok(format!("user={user_name},channel={channel}"))
            }
        }

        let tool: Box<dyn Tool> = Box::new(ReadTaskLocalsTool);
        let name = tool.name();
        let mut agent = make_agent(vec![tool]);
        agent.user_name = "alice".into();
        agent.channel = "gui".into();

        // Through execute_tool_group: the task-locals are set once per batch, as in the
        // agent work loop.
        let call = ToolCall {
            id: "call_1".into(),
            name: name.to_string(),
            arguments: serde_json::json!({}),
        };
        let outcomes = agent.execute_tool_group(&[call]).await;

        assert_eq!(
            outcomes.len(),
            1,
            "one tool call should produce one outcome"
        );
        let outcome = outcomes[0]
            .as_ref()
            .expect("the call should produce an outcome");
        assert!(outcome.success, "ReadTaskLocalsTool should succeed");
        assert!(
            outcome.output.contains("user=alice"),
            "should propagate user_name: {}",
            outcome.output
        );
        assert!(
            outcome.output.contains("channel=gui"),
            "should propagate channel: {}",
            outcome.output
        );
    }

    /// A queued job becomes a user message: a ticket comment carries its author prefix,
    /// any other kind passes its content through unformatted.
    #[tokio::test]
    async fn test_drain_incoming_messages_injects_the_job_as_a_user_message() {
        crate::util::test::init_management_test_stores().await;
        let cases = [
            (
                "_test_drain_ticket_comment",
                crate::agent::message_router::MessageKind::TicketComment,
                crate::Role::Manager,
                "manager",
                "Please fix the formatting",
                Some("[Comment from manager on ticket]"),
            ),
            (
                "_test_drain_non_comment",
                crate::agent::message_router::MessageKind::UserMessage,
                crate::Role::Assistant,
                "user",
                "Hello agent",
                None,
            ),
        ];
        for (agent_id, kind, role, user_name, content, prefix) in cases {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let mut agent = make_agent(vec![]);
            agent.incoming_rx = Some(rx);
            agent.agent_id = agent_id.into();

            let job = crate::agent::message_router::AgentJob {
                content: content.to_string(),
                workspace_name: "test_ws".to_string(),
                user_name: user_name.to_string(),
                channel: String::new(),
                kind,
                role,
                reply_target: None,
                pending_job_id: None,
                originating_workspace: None,
                custom_role: None,
                reply_to_custom: None,
                urgent: false,
            };
            let _ = tx.send(job);

            agent.drain_incoming_messages().await;

            let history = agent.session.history();
            assert!(!history.is_empty(), "{agent_id}: at least one message");
            let last = history.last().unwrap();
            assert_eq!(
                last.role,
                crate::ChatRole::User,
                "{agent_id}: a drained job is a user message"
            );
            assert!(
                last.content.contains(content),
                "{agent_id}: content is carried: {}",
                last.content,
            );
            if let Some(prefix) = prefix {
                assert!(
                    last.content.contains(prefix),
                    "{agent_id}: comment prefix present: {}",
                    last.content,
                );
            }
        }
    }

    /// A closed receiver is handled: incoming_rx becomes None, no panic.
    #[tokio::test]
    async fn test_drain_incoming_messages_disconnected() {
        let (tx, rx) =
            tokio::sync::mpsc::unbounded_channel::<crate::agent::message_router::AgentJob>();
        let mut agent = make_agent(vec![]);
        agent.incoming_rx = Some(rx);
        agent.agent_id = "_test_drain_disconnected".into();

        drop(tx);

        agent.drain_incoming_messages().await;
        assert!(
            agent.incoming_rx.is_none(),
            "incoming_rx should be set to None after disconnect",
        );
    }

    /// An agent with the given role and ticket attached, its session seeded with one
    /// `sessions` row (when `seed_content` is `Some`) and initialized with an empty
    /// message — mirroring the empty-message resume dispatch.
    async fn make_inject_agent(
        agent_id: &str,
        role: crate::Role,
        ticket: crate::pipeline::board::Ticket,
        seed_content: Option<&str>,
    ) -> Agent {
        if let Some(content) = seed_content {
            let now = crate::db::now();
            crate::session::store()
                .conn
                .execute(
                    "INSERT INTO sessions (agent_id, role, content, created_at) \
                     VALUES (?1, 'user', ?2, ?3)",
                    crate::db::params![agent_id, content, now],
                )
                .await
                .unwrap();
        }
        let mut agent = make_agent_with_role(vec![], role);
        agent.agent_id = agent_id.to_string();
        agent.session.init(agent_id).await.unwrap();
        agent.ticket = Some(ticket);
        agent
    }

    /// A comment added during a no-agent window is injected into the session for
    /// an empty-message resume — for a stage role and a non-stage role alike,
    /// with no special case for a comment the agent's own role authored (only a
    /// content already present in the loaded history suppresses it).
    #[tokio::test]
    async fn test_inject_outstanding_comments_injects_for_every_role() {
        crate::util::test::init_management_test_stores().await;
        // (agent id, role, workspace name, author, comment, phase)
        let cases: &[(
            &str,
            crate::Role,
            &str,
            &str,
            &str,
            crate::pipeline::board::TicketPhase,
        )] = &[
            (
                "inject-new-agent",
                crate::Role::Engineer,
                "inject_new",
                "manager",
                "please fix the formatting",
                crate::pipeline::board::TicketPhase::InDevelopment,
            ),
            (
                "inject-sanitation-agent",
                crate::Role::Sanitation,
                "inject_sanitation",
                "manager",
                "please verify the cleanup",
                crate::pipeline::board::TicketPhase::InSanitation,
            ),
            (
                "inject-own-role-agent",
                crate::Role::Engineer,
                "inject_own_role",
                "engineer",
                "note from the engineer",
                crate::pipeline::board::TicketPhase::InDevelopment,
            ),
            (
                "inject-analyst-agent",
                crate::Role::Analyst,
                "inject_analyst",
                "manager",
                "please fix the formatting",
                crate::pipeline::board::TicketPhase::InDevelopment,
            ),
        ];
        for &(agent_id, role, ws_name, author, comment, phase) in cases {
            let ws = crate::workspace::test_ws_named("/tmp/test", ws_name);
            let ticket_id = crate::util::test::make_ticket(
                crate::pipeline::board::store(),
                &ws,
                "Inject Comment",
                phase,
            )
            .await;
            crate::pipeline::board::store()
                .add_comment(&ticket_id, author, comment)
                .await
                .unwrap();
            let ticket =
                crate::util::test::expect_ticket(crate::pipeline::board::store(), &ticket_id).await;

            let mut agent = make_inject_agent(agent_id, role, ticket, Some("round 1 task")).await;
            agent.inject_outstanding_comments(true, &[""]).await;

            let prefix = format!("[Comment from {author} on ticket]");
            let history = agent.session.history();
            assert!(
                history
                    .iter()
                    .any(|m| m.content.contains(&prefix) && m.content.contains(comment)),
                "{agent_id}: a new board comment should be injected into a resumed {role} session",
            );
        }
    }

    /// A comment whose content already appears in the loaded session history is
    /// NOT re-injected (content-based dedup).
    #[tokio::test]
    async fn test_inject_outstanding_comments_dedups_existing_content() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws_named("/tmp/test", "inject_dedup");
        let ticket_id = crate::util::test::make_ticket(
            crate::pipeline::board::store(),
            &ws,
            "Inject Dedup",
            crate::pipeline::board::TicketPhase::InDevelopment,
        )
        .await;
        let content = "the comment content already seen";
        crate::pipeline::board::store()
            .add_comment(&ticket_id, "manager", content)
            .await
            .unwrap();
        let ticket =
            crate::util::test::expect_ticket(crate::pipeline::board::store(), &ticket_id).await;

        // Seed a history row that already carries the comment content (as a
        // previously delivered `[Comment from ...]` message) — the dedup must
        // skip it.
        let seed = format!("[Comment from manager on ticket]: {content}");
        let mut agent = make_inject_agent(
            "inject-dedup-agent",
            crate::Role::Engineer,
            ticket,
            Some(&seed),
        )
        .await;
        agent.inject_outstanding_comments(true, &[""]).await;

        let history = agent.session.history();
        assert_eq!(
            history.len(),
            1,
            "an already-seen comment must not be re-injected",
        );
    }

    /// The bounce-feedback path (non-empty `msg`) and a non-resumed round
    /// (`resume=false`) are no-ops — no injection.
    #[tokio::test]
    async fn test_inject_outstanding_comments_noop_for_feedback_and_fresh() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws_named("/tmp/test", "inject_noop");
        let ticket_id = crate::util::test::make_ticket(
            crate::pipeline::board::store(),
            &ws,
            "Inject Noop",
            crate::pipeline::board::TicketPhase::InDevelopment,
        )
        .await;
        crate::pipeline::board::store()
            .add_comment(&ticket_id, "manager", "please fix the formatting")
            .await
            .unwrap();
        let ticket =
            crate::util::test::expect_ticket(crate::pipeline::board::store(), &ticket_id).await;

        let mut agent = make_inject_agent(
            "inject-noop-agent",
            crate::Role::Engineer,
            ticket,
            Some("round 1 task"),
        )
        .await;
        // Bounce-feedback path: non-empty msg.
        agent
            .inject_outstanding_comments(true, &["round 2 feedback"])
            .await;
        // Fresh (non-resumed) round.
        agent.inject_outstanding_comments(false, &[""]).await;

        let history = agent.session.history();
        assert!(
            !history
                .iter()
                .any(|m| m.content.contains("[Comment from manager on ticket]")),
            "feedback path and fresh round must not inject comments",
        );
    }

    /// Single-member rounds are a no-op stagger: the sole member starts
    /// immediately with no first-call signal.
    #[tokio::test]
    async fn spawn_staggered_round_single_member_is_noop() {
        let handles = crate::agent::spawn_staggered_round(
            vec![move |round: crate::agent::RoundOpts| async move {
                assert!(
                    round.first_call_notify.is_none(),
                    "sole member must not receive a signal"
                );
                1u8
            }],
            false,
        )
        .await;
        assert_eq!(handles.len(), 1);
        assert_eq!(handles.into_iter().next().unwrap().await.unwrap(), 1);
    }

    /// Boot-resumed rounds skip the stagger entirely: every member starts
    /// immediately with no first-call signal (their sessions already
    /// diverged — staggering would only add latency).
    #[tokio::test]
    async fn spawn_staggered_round_resume_skips_stagger() {
        let handles = crate::agent::spawn_staggered_round(
            (0..3)
                .map(|i| {
                    move |round: crate::agent::RoundOpts| async move {
                        assert!(
                            round.first_call_notify.is_none(),
                            "resume must not stagger (member {i})"
                        );
                        i
                    }
                })
                .collect(),
            true,
        )
        .await;
        assert_eq!(handles.len(), 3);
        let mut out = Vec::new();
        for h in handles {
            out.push(h.await.unwrap());
        }
        assert_eq!(out, vec![0, 1, 2]);
    }

    /// The leader's first-call signal releases the followers: once the notify
    /// fires (before or during the wait), the remaining members start and the
    /// round completes. All members share one round-fixed timestamp.
    #[tokio::test]
    async fn spawn_staggered_round_leader_notify_releases_followers() {
        let handles = crate::agent::spawn_staggered_round(
            (0..3)
                .map(|i| {
                    move |round: crate::agent::RoundOpts| async move {
                        let tag = if i == 0 {
                            round
                                .first_call_notify
                                .expect("leader receives the signal")
                                .notify_one();
                            "leader".to_string()
                        } else {
                            assert!(
                                round.first_call_notify.is_none(),
                                "followers must not receive the signal"
                            );
                            "follower".to_string()
                        };
                        (tag, round.round_ts)
                    }
                })
                .collect(),
            false,
        )
        .await;
        assert_eq!(handles.len(), 3);
        let mut out = Vec::new();
        for h in handles {
            out.push(h.await.unwrap());
        }
        assert!(out.iter().any(|(tag, _)| tag == "leader"));
        assert_eq!(out.iter().filter(|(tag, _)| tag == "follower").count(), 2);
        assert_eq!(
            out.iter()
                .map(|(_, ts)| ts)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1,
            "all members share one round-fixed timestamp"
        );
    }

    /// Fail-open: a leader that never fires its signal (stuck first call)
    /// still releases the followers after the bounded wait — the round never hangs.
    #[tokio::test]
    async fn spawn_staggered_round_leader_timeout_fail_open() {
        // 0 → immediate timeout (deliberate test seam, see env_duration_secs).
        let _guard = crate::util::test::set_env_var("MAHBOT_STAGGER_WAIT_SECS", Some("0"));
        let handles = crate::agent::spawn_staggered_round(
            (0..2)
                .map(|i| {
                    move |round: crate::agent::RoundOpts| async move {
                        if i == 0 {
                            let _ = round; // never fires — stuck leader
                            "stuck-leader".to_string()
                        } else {
                            assert!(
                                round.first_call_notify.is_none(),
                                "released follower gets no signal"
                            );
                            "released".to_string()
                        }
                    }
                })
                .collect(),
            false,
        )
        .await;
        assert_eq!(
            handles.len(),
            2,
            "followers must be released despite the stuck leader"
        );
        for h in handles {
            h.await.unwrap();
        }
    }

    /// Fail-open on leader FAILURE: a leader whose first LLM call errors (the
    /// retry budget exhausts) still fires the first-call signal — followers
    /// are released immediately instead of waiting out the bound.
    // Joins the `provider` serial group: the seam swaps the process-global
    // provider, so the test must not interleave with the other provider-group
    // tests (same exclusion the pipeline/retry tests rely on).
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn leader_first_call_failure_still_fires_signal() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let ws = crate::workspace::test_ws_named("/tmp/ws_leader_fail", "leader_fail");
        let notify = std::sync::Arc::new(tokio::sync::Notify::new());
        // 3 scripted failures = the tiny policy's full budget → exhausted.
        let provider = FakeProvider::new()
            .err(crate::retry::FailureClass::Transport, "boom")
            .err(crate::retry::FailureClass::Transport, "boom")
            .err(crate::retry::FailureClass::Transport, "boom");
        let _provider = install_fake_provider(std::sync::Arc::new(provider));
        let (_agent, response) = run_agent(
            "leader_fail_agent".to_string(),
            crate::Role::Analyst,
            &ws,
            None,
            &["task"],
            String::new(),
            String::new(),
            false,
            None,
            false,
            Some(RoundOpts {
                round_ts: crate::session::render_timestamp(),
                first_call_notify: Some(notify.clone()),
            }),
            None,
            None,
            None,
        )
        .await;
        assert!(
            response.is_none(),
            "leader round must fail with an exhausted budget"
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), notify.notified())
            .await
            .expect("first-call signal must fire even when the call fails");
    }

    /// A successful `sleep` call ends the run gracefully: the round's tool
    /// result commits as a normal tool result, the loop stops WITHOUT a
    /// follow-up LLM call (the empty response is the intended silent end,
    /// never the failure path), and the durable sleep-ended flag is set. The
    /// next turn on the session (any wake-up: user message, async result,
    /// automation) clears the flag and runs normally.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn sleep_call_ends_the_run_gracefully_and_flag_resets_on_reengage() {
        crate::util::test::init_test_stores().await;
        // Script exactly ONE sleep tool call; a follow-up LLM call would pop
        // the second scripted item and the first turn would end with a text
        // response instead of the silent empty one.
        let fake = std::sync::Arc::new(FakeProvider::new().ok_tool_call("sleep").ok("back online"));
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let ws = crate::workspace::test_ws_named("/tmp/ws_sleep_stop", "sleep_stop");
        let agent_id = "sleep_stop_agent".to_string();

        let (agent, response) = run_agent(
            agent_id.clone(),
            crate::Role::Assistant,
            &ws,
            None,
            &["hello"],
            "sleep_user".to_string(),
            "gui".to_string(),
            false,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .await;
        assert!(agent.sleep_ended, "sleep must mark the turn sleep-ended");
        assert_eq!(
            response.as_deref(),
            Some(""),
            "the turn must end silently — any text would mean the loop continued"
        );
        assert_eq!(
            fake.request_messages.lock().unwrap().len(),
            1,
            "no LLM call may follow the committed sleep round"
        );
        // The round committed: the session tail is the sleep tool result.
        let (tail_role, tail_content) = crate::session::store()
            .get_last_message_tail(&agent_id)
            .await
            .expect("session must have history");
        assert_eq!(tail_role, crate::ChatRole::Tool);
        // The tool tail is the raw tool-result payload (JSON with the call id
        // and content) — the committed "Zzz..." result.
        assert!(
            tail_content.contains("\"tool_call_id\":\"call_test\"")
                && tail_content.contains("\"content\":\"Zzz...\""),
            "sleep tool result must be committed as a normal tool result: {tail_content}"
        );
        assert!(
            crate::session::store().get_sleep_ended(&agent_id).await,
            "the durable sleep-ended flag must be set"
        );

        let (agent2, response2) = run_agent(
            agent_id.clone(),
            crate::Role::Assistant,
            &ws,
            None,
            &["wake up"],
            "sleep_user".to_string(),
            "gui".to_string(),
            false,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .await;
        assert_eq!(response2.as_deref(), Some("back online"));
        assert!(!agent2.sleep_ended);
        assert!(
            !crate::session::store().get_sleep_ended(&agent_id).await,
            "the flag must be cleared when the session re-engages"
        );
    }

    /// A `sleep` call written in a round that also carries text is rejected at
    /// runtime (the text is never delivered in a tool round and the successful
    /// sleep ends the run silently) — the loop continues and the model resends
    /// the text as a plain round.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn sleep_with_text_is_rejected_and_resend_is_delivered() {
        crate::util::test::init_test_stores().await;
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_text_and_tool_calls("here is the answer", &[("sleep", serde_json::json!({}))])
                .ok("here is the answer"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let ws = crate::workspace::test_ws_named("/tmp/ws_sleep_text", "sleep_text");
        let agent_id = "sleep_text_agent".to_string();

        let (agent, response) = run_agent(
            agent_id.clone(),
            crate::Role::Assistant,
            &ws,
            None,
            &["hello"],
            "sleep_text_user".to_string(),
            "gui".to_string(),
            false,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .await;

        assert_eq!(
            response.as_deref(),
            Some("here is the answer"),
            "the text must be resent as a plain round after the rejection"
        );
        assert!(!agent.sleep_ended, "a rejected sleep must not end the run");
        assert_eq!(
            fake.request_messages.lock().unwrap().len(),
            2,
            "the rejection round plus the plain resend round"
        );

        // The sleep round committed a rejection Tool result, not a "Zzz..." one.
        let history = agent.session.history();
        let rejection_tools: Vec<String> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::Tool)
            .map(|m| {
                serde_json::from_str::<crate::ToolResultPayload>(&m.content)
                    .expect("tool message must be a ToolResultPayload")
                    .content
            })
            .collect();
        assert!(
            rejection_tools
                .iter()
                .any(|c| c.contains("NOT delivered") && !c.contains("Zzz...")),
            "the sleep rejection must be committed without a Zzz... result: {rejection_tools:?}"
        );

        assert!(
            !crate::session::store().get_sleep_ended(&agent_id).await,
            "the durable sleep-ended flag must stay clear"
        );
    }

    /// A `sleep` call bundled with another tool call is rejected at runtime
    /// (subsequent silent end), while the sibling call executes normally — the
    /// loop continues and the model can resend as a plain round.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn sleep_bundled_with_other_tool_is_rejected_but_sibling_executes() {
        crate::util::test::init_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("seed.txt"), "bundle-marker\n").unwrap();
        let ws = crate::Workspace::from_path(dir.path());

        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_text_and_tool_calls(
                    "",
                    &[
                        ("read", serde_json::json!({"path": "seed.txt"})),
                        ("sleep", serde_json::json!({})),
                    ],
                )
                .ok("done"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let agent_id = "sleep_bundle_agent".to_string();

        let (agent, response) = run_agent(
            agent_id.clone(),
            crate::Role::Assistant,
            &ws,
            None,
            &["hello"],
            "sleep_bundle_user".to_string(),
            "gui".to_string(),
            false,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .await;

        assert_eq!(response.as_deref(), Some("done"));
        assert!(!agent.sleep_ended, "a bundled sleep must not end the run");
        assert_eq!(fake.request_messages.lock().unwrap().len(), 2);

        let history = agent.session.history();
        let tool_results: Vec<String> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::Tool)
            .map(|m| {
                serde_json::from_str::<crate::ToolResultPayload>(&m.content)
                    .expect("tool message must be a ToolResultPayload")
                    .content
            })
            .collect();
        assert!(
            tool_results.iter().any(|c| c.contains("bundle-marker")),
            "the read sibling must execute normally: {tool_results:?}"
        );
        assert!(
            tool_results
                .iter()
                .any(|c| c.contains("bundled with other tool calls")),
            "the bundled sleep must be rejected: {tool_results:?}"
        );
    }

    /// The sleep-delivery guard also covers the resume-completion path: a
    /// dangling tool-call frame carrying text alongside a `sleep` call settles
    /// the rejection (the ORIGINAL call id) instead of re-executing it into a
    /// "Zzz..." result over text the user never saw.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn resume_completion_rejects_text_sleep_frame() {
        crate::util::test::init_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let ws = crate::Workspace::from_path(dir.path());

        let mut session = Session::default();
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some("the answer"),
            &[crate::ToolCall {
                id: "call_sleep_r".to_string(),
                name: "sleep".to_string(),
                arguments: serde_json::json!({}),
            }],
            None,
        )
        .to_string();
        session
            .persist_messages(
                "test-sleep-resume",
                &[ChatMessage::user("wake"), ChatMessage::assistant(frame)],
            )
            .await
            .unwrap();

        let mut agent = make_agent_on(
            vec![Box::new(crate::tools::SleepTool)],
            "test-sleep-resume",
            ws,
        );
        agent.session = session;
        let settled = agent.complete_pending_tool_calls().await.unwrap();
        assert!(
            settled,
            "a text-accompanied sleep frame settles the rejection"
        );

        let history = agent.session.history();
        let roles: Vec<crate::ChatRole> = history.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                crate::ChatRole::User,
                crate::ChatRole::Assistant,
                crate::ChatRole::Tool
            ],
            "result contiguous after the frame: {roles:?}"
        );
        let payload: crate::ToolResultPayload = serde_json::from_str(&history[2].content).unwrap();
        assert_eq!(
            payload.tool_call_id, "call_sleep_r",
            "rejection keeps the ORIGINAL call id"
        );
        assert!(
            payload.content.contains("NOT delivered"),
            "sleep rejection content: {}",
            payload.content
        );
        assert!(!payload.content.contains("Zzz..."));
        assert!(agent.session.pending_tool_frame().is_none());
    }

    /// The class predicate: raw empty content + no parsed tool calls, regardless of finish
    /// reason. Tool-call turns and real answers are excluded; fully-empty responses are
    /// consciously in class.
    #[test]
    fn reasoning_only_stop_classification() {
        let reasoning = || {
            Some(crate::Reasoning {
                reasoning: Some("thinking".into()),
                reasoning_content: Some("thinking".into()),
                reasoning_details: None,
            })
        };

        // Empty/absent content, no tools → in class for ANY finish reason.
        for finish in [None, Some("stop"), Some("length"), Some("tool_calls")] {
            let resp = crate::ChatResponse {
                text: None,
                reasoning: reasoning(),
                finish_reason: finish.map(str::to_string),
                ..crate::ChatResponse::default()
            };
            assert!(is_reasoning_only_stop(&resp), "finish_reason={finish:?}");
        }
        // Explicit empty-string content → in class.
        let resp = crate::ChatResponse {
            text: Some(String::new()),
            reasoning: reasoning(),
            ..crate::ChatResponse::default()
        };
        assert!(is_reasoning_only_stop(&resp));
        // Whitespace-only content → in class.
        let resp = crate::ChatResponse {
            text: Some("   \n ".into()),
            ..crate::ChatResponse::default()
        };
        assert!(is_reasoning_only_stop(&resp));
        // Fully-empty response (no reasoning attached) → in class.
        let resp = crate::ChatResponse::default();
        assert!(is_reasoning_only_stop(&resp));
        // Tool calls with empty text → NOT in class (valid tool-call turn).
        let resp = crate::ChatResponse {
            text: None,
            tool_calls: vec![crate::ToolCall {
                id: "t1".into(),
                name: "read".into(),
                arguments: serde_json::json!({}),
            }],
            ..crate::ChatResponse::default()
        };
        assert!(!is_reasoning_only_stop(&resp));
        // Real visible text (even with reasoning present) → NOT in class.
        let resp = crate::ChatResponse {
            text: Some("real answer".into()),
            reasoning: reasoning(),
            ..crate::ChatResponse::default()
        };
        assert!(!is_reasoning_only_stop(&resp));
    }

    /// A reasoning-only stop resolves via the continuation: the answer is returned, the
    /// continuation request carries the appended tail (reasoning payload + nudge), and the
    /// nudge never appears in the original request.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_recovers_reasoning_only_stop_via_continuation() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("draft plan then execute tool", Some("stop"))
                .ok("final answer"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        let resp = agent.llm_call().await.expect("continuation must resolve");
        assert_eq!(resp.text_or_empty(), "final answer");

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(fingerprints.len(), 2, "original call + one continuation");
        assert!(
            !fingerprints[0].contains("Resume your unfinished turn"),
            "original request must not carry the continuation tail"
        );
        assert!(
            fingerprints[1].contains("Resume your unfinished turn"),
            "continuation request must carry the appended nudge"
        );
        assert!(
            fingerprints[1].contains("draft plan then execute tool"),
            "continuation must echo the previous reasoning as the assistant turn"
        );
    }

    /// Consecutive reasoning-only responses accumulate one pair each, so the request
    /// prefix stays byte-stable and only the tail grows.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_continuation_accumulates_tail_until_answer() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("thinking 1", Some("stop"))
                .ok_reasoning_only("thinking 2", Some("stop"))
                .ok("answer after two continuations"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        let resp = agent.llm_call().await.expect("continuation must resolve");
        assert_eq!(resp.text_or_empty(), "answer after two continuations");

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(fingerprints.len(), 3);
        // Appended-only: one nudge after the first continuation, two after the second.
        assert_eq!(
            fingerprints[1]
                .matches("Resume your unfinished turn")
                .count(),
            1,
            "attempt 2 carries exactly the first appended pair"
        );
        assert_eq!(
            fingerprints[2]
                .matches("Resume your unfinished turn")
                .count(),
            2,
            "attempt 3 carries both appended pairs"
        );
        assert!(fingerprints[2].contains("thinking 1"));
        assert!(fingerprints[2].contains("thinking 2"));

        // Direct byte-prefix pin of the KV-cache property: attempt 3's messages begin with
        // attempt 2's verbatim (nothing rewritten).
        let messages = fake.request_messages.lock().unwrap().clone();
        assert_eq!(messages.len(), 3);
        assert!(
            messages[2].starts_with(&messages[1]),
            "byte-stable prefix: attempt 3's messages begin with attempt 2's verbatim"
        );
    }

    /// Exhaustion fails the turn safely: a NoResponse [`RetryExhausted`] with
    /// `last_raw: None`, no reasoning text, no retry marker, no transcript trace, and a
    /// granular "no_response" class.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_continuation_exhaustion_fails_safely_without_leaking() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("secret thinking alpha", Some("stop"))
                .ok_reasoning_only("secret thinking beta", Some("stop"))
                .ok_reasoning_only("secret thinking gamma", Some("stop"))
                .ok_reasoning_only("secret thinking delta", Some("stop")),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        let err = agent
            .llm_call()
            .await
            .expect_err("continuation must exhaust");
        let exhausted = err
            .chain()
            .find_map(|c| c.downcast_ref::<crate::retry::RetryExhausted>())
            .expect("RetryExhausted must survive in the error chain");
        assert_eq!(
            exhausted.final_class,
            crate::retry::FailureClass::NoResponse,
            "granular no-response classification"
        );
        assert_eq!(
            exhausted.last_raw, None,
            "no raw text on the exhausted error"
        );
        let last_failure = exhausted
            .failures
            .last()
            .expect("failure trail is non-empty");
        assert_eq!(
            last_failure.finish_reason.as_deref(),
            Some("stop"),
            "in-class NoResponse records carry the response finish_reason into the telemetry trail"
        );
        let rendered = format!("{err:#}");
        assert!(
            !rendered.contains("secret thinking"),
            "the thinking must never leak into the failure error"
        );
        assert!(
            !rendered.contains(RETRY_EXHAUSTION_MARKER),
            "must not be misclassified as LLM provider retry exhaustion"
        );
        assert!(
            agent.session.history().is_empty(),
            "the continuation tail must never reach the session transcript"
        );
        assert_eq!(failure_classification(&agent, Some(&err)), "no_response");
    }

    /// A transport error mid-recovery never duplicates the tail: the retried request is
    /// byte-identical, and the final class comes from the last failure (Transport).
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_continuation_transport_error_does_not_duplicate_tail() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("thinking 1", Some("stop"))
                .err(crate::retry::FailureClass::Transport, "connection reset")
                .err(crate::retry::FailureClass::Transport, "connection reset")
                .err(crate::retry::FailureClass::Transport, "connection reset"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        let err = agent
            .llm_call()
            .await
            .expect_err("continuation must exhaust");
        let exhausted = err
            .chain()
            .find_map(|c| c.downcast_ref::<crate::retry::RetryExhausted>())
            .expect("RetryExhausted must survive in the error chain");
        assert_eq!(
            exhausted.final_class,
            crate::retry::FailureClass::Transport,
            "final class derives from the last recorded failure, not NoResponse"
        );

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(
            fingerprints.len(),
            4,
            "original call + 3 continuation attempts"
        );
        assert_eq!(
            fingerprints[1], fingerprints[2],
            "attempt 2 re-sends the byte-identical request after a transport error"
        );
        assert_eq!(fingerprints[2], fingerprints[3]);
        let rendered = format!("{err:#}");
        assert!(
            !rendered.contains("thinking"),
            "the thinking must never leak into the failure error"
        );
    }

    /// A non-retryable error breaks immediately and keeps the granular class.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_continuation_non_retryable_error_breaks_immediately() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("thinking 1", Some("stop"))
                .err(crate::retry::FailureClass::NonRetryable, "invalid model"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        let err = agent.llm_call().await.expect_err("continuation must fail");
        let exhausted = err
            .chain()
            .find_map(|c| c.downcast_ref::<crate::retry::RetryExhausted>())
            .expect("RetryExhausted must survive in the error chain");
        assert_eq!(
            exhausted.final_class,
            crate::retry::FailureClass::NonRetryable,
            "non-retryable class survives to the terminal error"
        );
        assert_eq!(
            fake.request_fingerprints.lock().unwrap().len(),
            2,
            "original call + exactly one continuation attempt (no budget burn)"
        );
    }

    /// A global abort between attempts classifies the terminal error as Shutdown, not
    /// NoResponse — the real cause must not be masked, even before the first attempt (no
    /// request sent, nothing in telemetry).
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn recover_reasoning_only_stop_abort_classifies_as_shutdown() {
        let agent = make_agent(vec![]);
        let first = crate::ChatResponse {
            text: None,
            reasoning: Some(crate::Reasoning {
                reasoning: Some("thinking".into()),
                reasoning_content: Some("thinking".into()),
                reasoning_details: None,
            }),
            finish_reason: Some("stop".into()),
            ..crate::ChatResponse::default()
        };

        // The drain flag is process-global: set, then restore (never fire the token here).
        crate::shutdown::drain_begin();
        let exhausted = agent
            .recover_reasoning_only_stop(vec![], first, "agent-continuation")
            .await
            .expect_err("the drain must break the continuation immediately");
        crate::shutdown::drain_clear();

        assert_eq!(
            exhausted.final_class,
            crate::retry::FailureClass::Shutdown,
            "a global abort must classify as shutdown, never no_response"
        );
    }

    /// An exhausted reasoning-only turn is a FAILED call: it must never update the session
    /// token length, even though the invisible in-class response carried real usage.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_continuation_exhaustion_does_not_update_session_length() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only_with_usage("thinking a", Some("stop"), 1_000, 500)
                .ok_reasoning_only_with_usage("thinking b", Some("stop"), 1_000, 500)
                .ok_reasoning_only_with_usage("thinking c", Some("stop"), 1_000, 500)
                .ok_reasoning_only_with_usage("thinking d", Some("stop"), 1_000, 500),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        assert_eq!(agent.session.token_length(), None);
        let err = agent
            .llm_call()
            .await
            .expect_err("continuation must exhaust");
        assert_eq!(
            agent.session.token_length(),
            None,
            "a failed turn (continuation exhaustion) must not update the session length"
        );
        let rendered = format!("{err:#}");
        assert!(
            !rendered.contains("thinking"),
            "the thinking must never leak into the failure error"
        );
    }

    /// On success the session length comes from the RESOLVING response only — the
    /// in-class response is never recorded.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_continuation_success_records_only_resolving_usage() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only_with_usage("thinking a", Some("stop"), 1_000, 500)
                .ok_with_usage("final answer", 200, 300),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent(vec![]);
        let resp = agent.llm_call().await.expect("continuation must resolve");
        assert_eq!(resp.text_or_empty(), "final answer");
        assert_eq!(
            agent.session.token_length(),
            Some(500),
            "only the resolving continuation response (200 + 300) updates the session length"
        );
    }

    /// A normal answer or a tool-call turn never enters the continuation path.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn llm_call_skips_continuation_for_normal_and_tool_call_turns() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        {
            let fake = std::sync::Arc::new(FakeProvider::new().ok("normal answer"));
            let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
            let _provider_guard = install_fake_provider(provider);
            let mut agent = make_agent(vec![]);
            let resp = agent.llm_call().await.expect("normal answer");
            assert_eq!(resp.text_or_empty(), "normal answer");
            assert_eq!(
                fake.request_fingerprints.lock().unwrap().len(),
                1,
                "normal answer must not trigger continuation"
            );
        }
        {
            let fake = std::sync::Arc::new(FakeProvider::new().ok_tool_call("read"));
            let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
            let _provider_guard = install_fake_provider(provider);
            let mut agent = make_agent(vec![]);
            let resp = agent.llm_call().await.expect("tool-call turn");
            assert!(resp.text_or_empty().is_empty());
            assert_eq!(resp.tool_calls.len(), 1);
            assert_eq!(
                fake.request_fingerprints.lock().unwrap().len(),
                1,
                "tool-call turn (empty text) must not trigger continuation"
            );
        }
    }

    /// Summarize recovers a reasoning-only stop the same way; the answer is the summary.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn summarize_recovers_reasoning_only_stop_via_continuation() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("thinking about the summary", Some("stop"))
                .ok("the summary"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let agent = make_agent(vec![]);
        let summary = agent.summarize().await.expect("continuation must resolve");
        assert_eq!(summary, "the summary");

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(
            fingerprints.len(),
            2,
            "original summary call + one continuation"
        );
        assert!(fingerprints[1].contains("Resume your unfinished turn"));
    }

    /// Summarize exhaustion keeps failing open: the empty-response error fires and the
    /// thinking never leaks into it.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn summarize_continuation_exhaustion_fails_open_without_leaking() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_reasoning_only("summary thinking", Some("stop"))
                .ok_reasoning_only("summary thinking", Some("stop"))
                .ok_reasoning_only("summary thinking", Some("stop"))
                .ok_reasoning_only("summary thinking", Some("stop")),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let agent = make_agent(vec![]);
        let err = agent
            .summarize()
            .await
            .expect_err("continuation must exhaust");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("summarization"),
            "fail-open path surfaces the summarization error for maybe_summarize"
        );
        assert!(
            !rendered.contains("summary thinking"),
            "the thinking must never leak into the summarization error"
        );
    }

    /// A summarisation answer that is a tool call carries no text: it is re-asked
    /// byte-identically (verbatim, tool set included) instead of being accepted.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn summarize_reasks_a_tool_call_answer_byte_identically() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(FakeProvider::new().ok_tool_call("read").ok("the summary"));
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let agent = make_agent(vec![]);
        let summary = agent.summarize().await.expect("the re-ask must resolve");
        assert_eq!(summary, "the summary");

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(fingerprints.len(), 2, "tool-call answer + one re-ask");
        assert_eq!(
            fingerprints[0], fingerprints[1],
            "the re-ask must re-send the identical request"
        );
    }

    /// Tool-call answers exhaust the budget at [`SUMMARIZE_ATTEMPTS`] with the same
    /// outcome: the empty-response error fires.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn summarize_tool_call_exhaustion_keeps_the_empty_response_error() {
        crate::util::test::init_test_stores().await;
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .ok_tool_call("read")
                .ok_tool_call("read")
                .ok_tool_call("read"),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let agent = make_agent(vec![]);
        let err = agent
            .summarize()
            .await
            .expect_err("the budget must exhaust");
        assert!(
            format!("{err:#}").contains("summarization produced empty response"),
            "the fail-open error is unchanged: {err:#}"
        );
        assert_eq!(
            fake.request_fingerprints.lock().unwrap().len(),
            SUMMARIZE_ATTEMPTS as usize,
            "the re-ask must stop at the attempt bound"
        );
    }

    /// Both-or-nothing / keep-last / never-zero rule of [`Agent::record_session_usage`].
    #[tokio::test]
    async fn record_session_usage_keeps_last_on_missing_or_partial_usage() {
        crate::util::test::init_test_stores().await;
        let mut agent = make_agent(vec![]);
        agent.session.set_token_length(Some(7_000));

        // No usage data at all → keep the last known value.
        agent
            .record_session_usage(&crate::ChatResponse::default())
            .await;
        assert_eq!(agent.session.token_length(), Some(7_000));

        // Only one side present → keep the last known value.
        let partial_input = crate::ChatResponse {
            usage: Some(crate::ProviderUsage {
                input_tokens: Some(1_000),
                ..crate::ProviderUsage::default()
            }),
            ..crate::ChatResponse::default()
        };
        agent.record_session_usage(&partial_input).await;
        assert_eq!(agent.session.token_length(), Some(7_000));

        let partial_output = crate::ChatResponse {
            usage: Some(crate::ProviderUsage {
                output_tokens: Some(1_000),
                ..crate::ProviderUsage::default()
            }),
            ..crate::ChatResponse::default()
        };
        agent.record_session_usage(&partial_output).await;
        assert_eq!(agent.session.token_length(), Some(7_000));

        // Both present → input + output; a later usage-less response never
        // resets the value to zero.
        let full = crate::ChatResponse {
            usage: Some(crate::ProviderUsage {
                input_tokens: Some(12_000),
                output_tokens: Some(300),
                ..crate::ProviderUsage::default()
            }),
            ..crate::ChatResponse::default()
        };
        agent.record_session_usage(&full).await;
        assert_eq!(agent.session.token_length(), Some(12_300));
        agent
            .record_session_usage(&crate::ChatResponse::default())
            .await;
        assert_eq!(
            agent.session.token_length(),
            Some(12_300),
            "a usage-less response must never reset the length to zero"
        );
    }

    /// Saturating sum: an overflowing pair cannot wrap into a small bogus length.
    #[tokio::test]
    async fn record_session_usage_overflow_saturates() {
        crate::util::test::init_test_stores().await;
        let mut agent = make_agent(vec![]);
        let huge = crate::ChatResponse {
            usage: Some(crate::ProviderUsage {
                input_tokens: Some(u64::MAX),
                output_tokens: Some(u64::MAX),
                ..crate::ProviderUsage::default()
            }),
            ..crate::ChatResponse::default()
        };
        agent.record_session_usage(&huge).await;
        assert_eq!(agent.session.token_length(), Some(u64::MAX));
    }

    /// The failure context carries BOTH the run-local tool round and the session depth
    /// (history length + tokens, the latter omitted when unknown).
    #[tokio::test]
    async fn llm_step_failure_context_reports_run_round_and_session_depth() {
        crate::util::test::init_test_stores().await;
        let mut agent = make_agent(vec![]);
        agent
            .session
            .push_messages_unpersisted(&[ChatMessage::user("hello"), ChatMessage::assistant("hi")]);
        agent.session.set_token_length(Some(42_110));

        let ctx = llm_step_failure_context(&agent.session, 0);
        assert!(ctx.contains("tool round 0"), "run-local round: {ctx}");
        assert!(ctx.contains("2 messages"), "session depth: {ctx}");
        assert!(ctx.contains("42110 tokens"), "token segment: {ctx}");

        agent.session.set_token_length(None);
        let ctx = llm_step_failure_context(&agent.session, 3);
        assert!(ctx.contains("tool round 3"), "round after increment: {ctx}");
        assert!(ctx.contains("2 messages"), "session depth: {ctx}");
        assert!(
            !ctx.contains("tokens"),
            "None omits the token segment: {ctx}"
        );
    }

    /// An unknown session length only disables the token trigger; with no images present
    /// the check exits before the summarize call. No provider is installed on purpose —
    /// reaching [`Agent::summarize`] would panic.
    #[tokio::test]
    async fn maybe_summarize_none_and_below_threshold_are_noops() {
        crate::util::test::init_test_stores().await;
        let mut agent = make_agent(vec![]);

        // None (new / pre-migration session): early exit, nothing fires.
        agent.maybe_summarize(agent.session.history().len()).await;
        assert_eq!(agent.session.token_length(), None);

        // A real value below the fixed threshold is equally a no-op.
        agent
            .session
            .set_token_length(Some(crate::session::SUMMARIZATION_THRESHOLD / 2));
        agent.maybe_summarize(agent.session.history().len()).await;
    }

    /// Post-delivery compaction for the long-lived chat roles: only they may compact with
    /// the reply delivered, and never on a turn that settled dangling results, was
    /// cancelled, or ran while going down. Both triggers fire on the next turn's own
    /// terms, and the recorded length goes with the compacted history in one transaction.
    #[tokio::test]
    #[serial_test::serial(active_models, provider, drain)]
    async fn post_delivery_summarization_compacts_chat_roles_and_clears_the_length() {
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = std::sync::Arc::new(FakeProvider::new().ok("the summary"));
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let over_threshold = |role| {
            let mut agent = make_agent_with_role(vec![], role);
            agent
                .session
                .set_token_length(Some(crate::session::SUMMARIZATION_THRESHOLD + 1));
            agent
        };
        let calls = || fake.request_fingerprints.lock().unwrap().len();

        // Pipeline working roles and worker branches keep the pre-turn behaviour.
        for role in [
            crate::Role::Engineer,
            crate::Role::Analyst,
            crate::Role::Worker,
            crate::Role::Maintainer,
        ] {
            over_threshold(role).maybe_summarize_after_delivery().await;
        }

        // A settled turn is resume-like: the window would drop the just-settled result.
        let mut settled = over_threshold(crate::Role::Assistant);
        settled.turn_state.settled_tool_results = true;
        settled.maybe_summarize_after_delivery().await;

        // A cancelled run never spends a summarization call on itself.
        let mut cancelled = over_threshold(crate::Role::Assistant);
        cancelled.cancel_token.cancel();
        cancelled.maybe_summarize_after_delivery().await;

        // Going down: no destructive compaction during the drain window.
        crate::shutdown::drain_begin();
        over_threshold(crate::Role::Assistant)
            .maybe_summarize_after_delivery()
            .await;
        crate::shutdown::drain_clear();

        assert_eq!(
            calls(),
            0,
            "no excluded role or turn may spend a summarization call"
        );

        // Each chat role does compact, and its recorded length goes with the history,
        // through the production writer.
        for (role, agent_id) in [
            (crate::Role::Assistant, "e2e_post_delivery_assistant"),
            (crate::Role::Manager, "e2e_post_delivery_manager"),
            (crate::Role::Custom, "e2e_post_delivery_custom"),
        ] {
            let mut agent = make_agent_with_role(vec![], role);
            agent.agent_id = agent_id.to_string();
            agent.session.push_messages_unpersisted(&[
                ChatMessage::user("hello"),
                ChatMessage::assistant("hi"),
            ]);
            agent
                .record_session_usage(&crate::ChatResponse {
                    usage: Some(crate::ProviderUsage {
                        input_tokens: Some(crate::session::SUMMARIZATION_THRESHOLD),
                        output_tokens: Some(1),
                        ..crate::ProviderUsage::default()
                    }),
                    ..crate::ChatResponse::default()
                })
                .await;
            assert_eq!(
                reload_token_length(agent_id).await,
                Some(crate::session::SUMMARIZATION_THRESHOLD + 1),
                "the over-limit length is the one the next turn loads"
            );

            let before = calls();
            agent.maybe_summarize_after_delivery().await;
            assert_eq!(
                calls(),
                before + 1,
                "an over-limit {role} session must compact here"
            );
            assert_eq!(
                agent.session.token_length(),
                None,
                "the pre-compaction length must be cleared"
            );
            assert_eq!(
                reload_token_length(agent_id).await,
                None,
                "the next turn must compact nothing again"
            );

            // The image trigger fires on its own terms: images pushed after the compaction
            // leave the session over the count, and delivery carries no "already
            // compacted" clause.
            agent.session.push_messages_unpersisted(
                &(0..crate::session::SUMMARIZATION_IMAGE_COUNT)
                    .map(|i| ChatMessage::user(format!("[IMAGE:data:image/jpeg;base64,img{i}]")))
                    .collect::<Vec<_>>(),
            );
            let image_count = session_image_marker_count(agent.session.history());
            assert_eq!(
                summarization_trigger(agent.session.token_length(), image_count),
                Some("image count"),
                "the image trigger must be the one firing here"
            );
            agent.maybe_summarize_after_delivery().await;
            assert_eq!(
                calls(),
                before + 2,
                "an over-image-count {role} session must compact here too"
            );
        }
    }

    /// The cap holds on the request the agent actually sends: the conversation keeps every picture,
    /// the request carries the newest ones and no more, and the retired ones read as the omission
    /// note.
    #[tokio::test]
    async fn build_chat_request_bounds_the_pictures_it_sends() {
        crate::util::test::init_test_stores().await;
        let mut agent = make_agent_with_role(vec![], crate::Role::Assistant);
        let total = crate::session::SUMMARIZATION_IMAGE_COUNT + 11;
        let pictures: Vec<ChatMessage> = (0..total)
            .map(|serial| {
                ChatMessage::user(format!(
                    "shot {serial} [IMAGE:data:image/png;base64,{serial:0>8}]"
                ))
            })
            .collect();
        agent.session.push_messages_unpersisted(&pictures);

        let request = agent.build_chat_request(agent.session.history().to_vec(), "agent");

        // Past thirty pictures the cap is crossed twice over, so two whole batches step aside.
        assert_eq!(
            crate::session::image_cap::count(&request.messages),
            total - 2 * crate::session::image_cap::RETIRE_BATCH
        );
        assert!(
            request.messages[total - 1]
                .content
                .contains("data:image/png;base64,00000030"),
            "the newest picture is always among the ones sent"
        );
        assert!(!request.messages[0].content.contains("data:image/"));
        assert!(request.messages[0].content.contains("data-uri omitted"));
        assert_eq!(
            crate::session::image_cap::count(agent.session.history()),
            total,
            "the conversation itself keeps every picture"
        );
    }

    /// The observed provider rejection body (OpenRouter HTTP 400
    /// `data_inspection_failed`, image variant).
    const IMAGE_REJECTION_BODY: &str = r#"{"error":{"message":"Input image data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#;

    /// Same code, text variant — must NOT trigger the strip.
    const TEXT_REJECTION_BODY: &str = r#"{"error":{"message":"Input data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#;

    /// The recorded session length as a fresh turn's session init would load it.
    async fn reload_token_length(agent_id: &str) -> Option<u64> {
        let mut session = crate::session::Session::default();
        session.init(agent_id).await.expect("session init");
        session.token_length()
    }

    /// Seed empty model catalogs so the Assistant active-models block renders
    /// nothing and performs no network fetch (hermetic e2e).
    fn seed_empty_catalogs() {
        crate::tools::media_catalog::image::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::image::ImageCatalog::default(),
        )));
        crate::tools::media_catalog::video::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::video::VideoCatalog::default(),
        )));
    }

    /// A rejected image must not fail the run: it is durably stripped from the most recent
    /// user message (replaced by the phrase) and the loop continues — the retried call
    /// carries the corrected message and the agent answers.
    #[tokio::test]
    #[serial_test::serial(active_models, provider, drain)] // seeds the process-global catalog caches + process-global fake provider + shutdown drain flag
    async fn rejected_input_image_is_stripped_and_run_continues() {
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        let agent_id = "e2e_assistant_image_reject";
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .err_http(400, IMAGE_REJECTION_BODY)
                .ok("The image was rejected by the provider's content check."),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent_with_role(vec![], crate::Role::Assistant);
        agent.agent_id = agent_id.to_string();
        let resp = agent
            .work(&["[IMAGE:/tmp/photo.png] describe this photo"], false)
            .await
            .expect("the run must not fail wholesale on a rejected input image");
        assert_eq!(
            resp,
            "The image was rejected by the provider's content check."
        );

        // The corrected message reached the model: the phrase and no marker in the user
        // message (the system prompt mentions "[IMAGE:path]" literally, so assert on the
        // user segment).
        let messages = fake.request_messages.lock().unwrap().clone();
        assert_eq!(messages.len(), 2, "rejected attempt + normal-loop retry");
        assert!(
            messages[0].contains("[IMAGE:/tmp/photo.png]"),
            "first request carried the image"
        );
        let retried_user = messages[1]
            .split('\u{0}')
            .next_back()
            .expect("retried request has a user segment");
        assert!(
            !retried_user.contains("[IMAGE:"),
            "retried user message no longer carries the image"
        );
        assert!(
            retried_user.contains("rejected by the provider's content-inspection check"),
            "retried user message carries the explanatory phrase"
        );
        assert!(
            retried_user.contains("Input image data may contain inappropriate content."),
            "phrase embeds the provider reason"
        );

        // Durable rewrite: the session DB no longer holds the rejected image.
        let history = crate::session::store().load(agent_id).await;
        let last_user = history
            .iter()
            .rev()
            .find(|m| m.role == crate::ChatRole::User)
            .expect("user message exists");
        assert!(
            !last_user.content.contains("[IMAGE:"),
            "rejected image durably removed from the session"
        );
        assert!(
            last_user.content.contains("describe this photo"),
            "user's accompanying text preserved"
        );
        // The original text plus the phrase, with NO extra message injected.
        assert_eq!(
            history
                .iter()
                .filter(|m| m.role == crate::ChatRole::User)
                .count(),
            1,
            "no separate notification message was added"
        );
    }

    /// A text-content rejection with the same code does not trigger the strip: the run
    /// takes the normal failure path and the image stays.
    #[tokio::test]
    #[serial_test::serial(active_models, provider, drain)] // seeds the process-global catalog caches + process-global fake provider + shutdown drain flag
    async fn text_content_rejection_follows_normal_failure_path() {
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        let agent_id = "e2e_assistant_text_reject";
        let fake = std::sync::Arc::new(FakeProvider::new().err_http(400, TEXT_REJECTION_BODY));
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent_with_role(vec![], crate::Role::Assistant);
        agent.agent_id = agent_id.to_string();
        let result = agent
            .work(&["[IMAGE:/tmp/photo.png] describe this photo"], false)
            .await;
        assert!(
            result.is_err(),
            "text-content rejection must fail the run normally"
        );

        // No strip: the image marker survives in the session.
        let history = crate::session::store().load(agent_id).await;
        let last_user = history
            .iter()
            .rev()
            .find(|m| m.role == crate::ChatRole::User)
            .expect("user message exists");
        assert!(
            last_user.content.contains("[IMAGE:/tmp/photo.png]"),
            "image untouched on a text-content rejection"
        );
    }

    /// After a successful strip, a later failure does NOT re-strip: the message carries no
    /// marker, so the normal path applies (no infinite retry loop).
    #[tokio::test]
    #[serial_test::serial(active_models, provider, drain)] // seeds the process-global catalog caches + process-global fake provider + shutdown drain flag
    async fn subsequent_failure_after_strip_takes_normal_failure_path() {
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        let agent_id = "e2e_assistant_second_failure";
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .err_http(400, IMAGE_REJECTION_BODY)
                .err_http(400, IMAGE_REJECTION_BODY),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent_with_role(vec![], crate::Role::Assistant);
        agent.agent_id = agent_id.to_string();
        let result = agent
            .work(&["[IMAGE:/tmp/photo.png] describe this photo"], false)
            .await;
        assert!(
            result.is_err(),
            "a second failure after the strip must fail normally"
        );

        let messages = fake.request_messages.lock().unwrap().clone();
        assert_eq!(
            messages.len(),
            2,
            "exactly one strip, then the normal failure"
        );
        let retried_user = messages[1]
            .split('\u{0}')
            .next_back()
            .expect("retried request has a user segment");
        assert!(
            !retried_user.contains("[IMAGE:"),
            "retried user message is already stripped"
        );

        // The message was stripped exactly once (phrase present, no marker).
        let history = crate::session::store().load(agent_id).await;
        let last_user = history
            .iter()
            .rev()
            .find(|m| m.role == crate::ChatRole::User)
            .expect("user message exists");
        assert!(!last_user.content.contains("[IMAGE:"));
        assert!(
            last_user
                .content
                .contains("rejected by the provider's content-inspection check"),
            "phrase present after a single strip"
        );
    }

    /// DISTINCT `data:image/`-prefixed IMAGE markers in User-role messages. The gate is
    /// prefix-only: real PNG data-URIs count, duplicates collapse, file paths, non-IMAGE
    /// kinds and assistant messages do not.
    #[test]
    fn session_image_marker_count_prefix_gate() {
        let png = crate::util::test::tiny_png_data_uri();
        let history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user(format!("[IMAGE:{png}] first")),
            ChatMessage::user(format!("[IMAGE:{png}] duplicate — same URI")),
            ChatMessage::user("[IMAGE:/local/path.png] file path"),
            ChatMessage::user("[AUDIO:data:audio/ogg;base64,zzz]"),
            ChatMessage::assistant(format!("[IMAGE:{png}] assistant role must not count")),
        ];
        assert_eq!(
            session_image_marker_count(&history),
            1,
            "only the distinct native data-URI IMAGE marker in User messages counts"
        );
    }

    /// The decision is asymmetric: strict `>` for tokens, `>=` for images; the token
    /// trigger reports the reason when both fire.
    #[test]
    fn summarization_trigger_decides_reason() {
        use crate::session::{SUMMARIZATION_IMAGE_COUNT, SUMMARIZATION_THRESHOLD};
        let just_below = SUMMARIZATION_IMAGE_COUNT - 1;
        assert_eq!(summarization_trigger(None, just_below), None);
        assert_eq!(
            summarization_trigger(None, SUMMARIZATION_IMAGE_COUNT),
            Some("image count")
        );
        assert_eq!(
            summarization_trigger(Some(SUMMARIZATION_THRESHOLD), 0),
            None
        );
        assert_eq!(
            summarization_trigger(Some(SUMMARIZATION_THRESHOLD + 1), 0),
            Some("token threshold")
        );
        assert_eq!(
            summarization_trigger(Some(SUMMARIZATION_THRESHOLD + 1), SUMMARIZATION_IMAGE_COUNT,),
            Some("token threshold")
        );
        assert_eq!(summarization_trigger(Some(1), just_below), None);
        assert_eq!(
            summarization_trigger(Some(1), SUMMARIZATION_IMAGE_COUNT),
            Some("image count")
        );
    }

    /// The synthetic image message lands AFTER the round's tool-result block and dedups a
    /// repeat read of an attached image.
    #[tokio::test]
    #[expect(clippy::too_many_lines)]
    async fn commit_tool_results_injects_image_after_tool_results_with_dedup() {
        crate::util::test::init_test_stores().await;

        let mut agent = make_agent(vec![]);
        // Seed a prior round: an image with this data-URI is already in the conversation.
        agent.session.push_messages_unpersisted(&[ChatMessage::user(
            "[IMAGE:data:image/jpeg;base64,prior]",
        )]);
        // Emission-time durability: the frame is persisted before the results commit.
        agent
            .session
            .push_assistant("assistant with tool_calls".to_string());

        let tool_calls = vec![
            ToolCall {
                id: "call1".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.png"}),
            },
            ToolCall {
                id: "call2".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "b.png"}),
            },
        ];
        let outcomes = vec![
            ToolExecutionOutcome {
                output: "Read image file /a.png (1x1, PNG).".into(),
                success: true,
                image_payloads: vec![ImagePayload {
                    path: "/a.png".into(),
                    data_uri: "data:image/jpeg;base64,prior".into(),
                    width: 1,
                    height: 1,
                    format: "PNG".into(),
                    recovery_note: None,
                    note: None,
                    source: crate::tools::ImagePayloadSource::Read,
                }],
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
            ToolExecutionOutcome {
                output: "Read image file /b.png (1x1, PNG).".into(),
                success: true,
                image_payloads: vec![ImagePayload {
                    path: "/b.png".into(),
                    data_uri: "data:image/jpeg;base64,fresh".into(),
                    width: 1,
                    height: 1,
                    format: "PNG".into(),
                    recovery_note: None,
                    note: None,
                    source: crate::tools::ImagePayloadSource::Read,
                }],
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
        ];

        agent
            .commit_tool_results(&tool_calls, outcomes)
            .await
            .expect("commit_tool_results must succeed");

        let history = agent.session.history();
        let roles: Vec<crate::ChatRole> = history.iter().map(|m| m.role).collect();
        // Order: prior user, assistant(tool_calls), both tool results, then user(fresh) —
        // the 'prior' image is NOT re-injected.
        assert_eq!(
            roles,
            vec![
                crate::ChatRole::User,
                crate::ChatRole::Assistant,
                crate::ChatRole::Tool,
                crate::ChatRole::Tool,
                crate::ChatRole::User,
            ],
            "unexpected message ordering: {roles:?}"
        );

        let image_users: Vec<&str> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::User && m.content.contains("[IMAGE:data:"))
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            image_users.len(),
            2,
            "prior + fresh image messages, got: {image_users:?}"
        );
        assert!(
            image_users.contains(&"[IMAGE:data:image/jpeg;base64,prior]"),
            "prior image survives in history: {image_users:?}"
        );
        assert!(
            image_users
                .iter()
                .any(|s| s.contains("[IMAGE:data:image/jpeg;base64,fresh]")),
            "fresh image injected once: {image_users:?}"
        );
        let fresh_msg = image_users
            .iter()
            .find(|s| s.contains("base64,fresh"))
            .expect("fresh image message present");
        assert!(
            fresh_msg.starts_with("<injected-tool-result-image>\n[IMAGE:data:image/jpeg;base64,"),
            "injected image carries the provenance tag: {fresh_msg:?}"
        );
        // The repeat read of 'prior' is not re-injected; the fresh image exactly once.
        let fresh_count = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::User && m.content.contains("base64,fresh"))
            .count();
        assert_eq!(fresh_count, 1, "a fresh image is injected exactly once");
        // The text comes from the payload metadata: the deduped read returns the
        // 'already in the conversation' reference, the fresh one the 'attached' wording. Content is a
        // JSON envelope, so parse out the `content` field.
        let tool_results: Vec<String> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::Tool)
            .map(|m| {
                let v: serde_json::Value =
                    serde_json::from_str(&m.content).expect("tool result is valid JSON");
                crate::util::json::get_str(&v, "content")
                    .expect("tool result has content")
                    .to_string()
            })
            .collect();
        assert_eq!(
            tool_results,
            vec![
                "Read image file /a.png (1x1, PNG). Image content is already in the conversation as a native image and is not attached again.",
                "Read image file /b.png (1x1, PNG). Image content attached to the conversation as a native image.",
            ],
            "unexpected tool-result text: {tool_results:?}"
        );
    }

    /// Two identical reads within one round are deduped by `seen_data_uris`: only the
    /// first is injected.
    #[tokio::test]
    async fn commit_tool_results_dedups_identical_reads_in_same_round() {
        crate::util::test::init_test_stores().await;

        let mut agent = make_agent(vec![]);

        let tool_calls = vec![
            ToolCall {
                id: "callA".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.png"}),
            },
            ToolCall {
                id: "callB".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.png"}),
            },
        ];
        let outcomes = vec![
            ToolExecutionOutcome {
                output: "Read image file /a.png (PNG).".into(),
                success: true,
                image_payloads: vec![ImagePayload {
                    path: "/a.png".into(),
                    data_uri: "data:image/jpeg;base64,same".into(),
                    width: 1,
                    height: 1,
                    format: "PNG".into(),
                    recovery_note: None,
                    note: None,
                    source: crate::tools::ImagePayloadSource::Read,
                }],
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
            ToolExecutionOutcome {
                output: "Read image file /a.png (PNG).".into(),
                success: true,
                image_payloads: vec![ImagePayload {
                    path: "/a.png".into(),
                    data_uri: "data:image/jpeg;base64,same".into(),
                    width: 1,
                    height: 1,
                    format: "PNG".into(),
                    recovery_note: None,
                    note: None,
                    source: crate::tools::ImagePayloadSource::Read,
                }],
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
        ];

        agent
            .commit_tool_results(&tool_calls, outcomes)
            .await
            .expect("commit_tool_results must succeed");

        let history = agent.session.history();
        let image_users: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.role == crate::ChatRole::User
                    && m.content.contains("[IMAGE:data:image/jpeg;base64,same]")
            })
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            image_users.len(),
            1,
            "identical reads in one round inject exactly once, got: {image_users:?}"
        );

        // The first claims a fresh attachment; the second an already-attached reference.
        let tool_results: Vec<String> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::Tool)
            .map(|m| {
                let v: serde_json::Value =
                    serde_json::from_str(&m.content).expect("tool result is valid JSON");
                crate::util::json::get_str(&v, "content")
                    .expect("tool result has content")
                    .to_string()
            })
            .collect();
        assert_eq!(tool_results.len(), 2, "two tool results persisted");
        assert!(
            tool_results[0].contains("attached to the conversation as a native image")
                && !tool_results[0].contains("already"),
            "first read claims a fresh attachment: {tool_results:?}"
        );
        assert!(
            tool_results[1].contains("already in the conversation"),
            "second read returns a reference: {tool_results:?}"
        );
    }

    /// Every fresh image of one round lands in a SINGLE synthetic user message, so a
    /// rejected page from any call is clearable.
    #[tokio::test]
    async fn commit_tool_results_injects_one_image_message_per_round() {
        crate::util::test::init_test_stores().await;

        let mut agent = make_agent(vec![]);
        let tool_calls = vec![
            ToolCall {
                id: "callA".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.pdf"}),
            },
            ToolCall {
                id: "callB".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "b.pdf"}),
            },
        ];
        let payload = |suffix: &str| ImagePayload {
            path: format!("/tmp/{suffix}.jpg"),
            data_uri: format!("data:image/jpeg;base64,{suffix}"),
            width: 1,
            height: 1,
            format: "JPEG".into(),
            recovery_note: None,
            note: None,
            source: crate::tools::ImagePayloadSource::Read,
        };
        let outcomes = vec![
            ToolExecutionOutcome {
                output: "Read image file /tmp/a.jpg.".into(),
                success: true,
                image_payloads: vec![payload("a1"), payload("a2")],
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
            ToolExecutionOutcome {
                output: "Read image file /tmp/b.jpg.".into(),
                success: true,
                image_payloads: vec![payload("b1")],
                text_is_content: false,
                preserve_text: false,
                suspended: false,
                ends_turn: false,
            },
        ];

        agent
            .commit_tool_results(&tool_calls, outcomes)
            .await
            .expect("commit_tool_results must succeed");

        let history = agent.session.history();
        let image_messages: Vec<&str> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::User && m.content.contains("[IMAGE:"))
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            image_messages.len(),
            1,
            "one message for the whole round, got: {image_messages:?}"
        );
        let message = image_messages[0];
        assert!(
            message.starts_with("<injected-tool-result-image>"),
            "the batch keeps the provenance tag: {message:?}"
        );
        for suffix in ["a1", "a2", "b1"] {
            assert!(
                message.contains(&format!("[IMAGE:data:image/jpeg;base64,{suffix}]")),
                "every fresh image of the round is in the batch: {message:?}"
            );
        }
    }

    /// The persisted content of the history's tool results, parsed out of their JSON envelope.
    fn tool_result_texts(history: &[ChatMessage]) -> Vec<String> {
        history
            .iter()
            .filter(|m| m.role == crate::ChatRole::Tool)
            .map(|m| {
                let v: serde_json::Value =
                    serde_json::from_str(&m.content).expect("tool result is valid JSON");
                crate::util::json::get_str(&v, "content")
                    .expect("tool result has content")
                    .to_string()
            })
            .collect()
    }

    /// The content side of the image invariant: a result whose text is content — a
    /// document's extracted text — is neither read as a claim nor replaced by an
    /// annotation. A marker inside it stays literal and injects nothing, while a real
    /// payload SUPPLEMENTS the text: the annotation is appended.
    #[tokio::test]
    async fn commit_tool_results_handles_content_text() {
        crate::util::test::init_test_stores().await;

        // A real on-disk raster, so derivation would succeed if it were allowed.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let png_path = dir.path().join("page.png");
        std::fs::write(&png_path, crate::util::test::noisy_png(4, 4)).expect("write png");
        let abs = std::fs::canonicalize(&png_path).expect("canonicalize");

        let mut agent = make_agent(vec![]);
        let marker = format!("[IMAGE:{}]", abs.display());
        let body = format!("[scan.pdf: extracted text follows]\n\nsee {marker} in the report");
        let tool_calls = vec![ToolCall {
            id: "call1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "scan.pdf"}),
        }];
        let outcomes = vec![ToolExecutionOutcome {
            output: body.clone(),
            success: true,
            image_payloads: Vec::new(),
            text_is_content: true,
            preserve_text: false,
            suspended: false,
            ends_turn: false,
        }];

        agent
            .commit_tool_results(&tool_calls, outcomes)
            .await
            .expect("commit_tool_results must succeed");

        let history = agent.session.history();
        let tool_result = tool_result_texts(history).remove(0);
        assert!(
            tool_result.contains(&marker),
            "a marker inside content text stays literal: {tool_result:?}"
        );
        assert!(
            !history
                .iter()
                .any(|m| m.role == crate::ChatRole::User && m.content.contains("[IMAGE:")),
            "nothing is derived from a marker in content text: {history:?}"
        );

        let mut agent = make_agent(vec![]);
        let tool_calls = vec![ToolCall {
            id: "call1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "scan.pdf"}),
        }];
        let outcomes = vec![ToolExecutionOutcome {
            output: "[scan.pdf: no text could be extracted — the content was provided as images]"
                .into(),
            success: true,
            image_payloads: vec![ImagePayload {
                path: "/tmp/read_1/page_1.jpg".into(),
                data_uri: "data:image/jpeg;base64,page1".into(),
                width: 800,
                height: 1000,
                format: "JPEG".into(),
                recovery_note: None,
                note: None,
                source: crate::tools::ImagePayloadSource::Generated,
            }],
            text_is_content: true,
            preserve_text: false,
            suspended: false,
            ends_turn: false,
        }];

        agent
            .commit_tool_results(&tool_calls, outcomes)
            .await
            .expect("commit_tool_results must succeed");

        let history = agent.session.history();
        let tool_result = tool_result_texts(history).remove(0);
        assert_eq!(
            tool_result,
            "[scan.pdf: no text could be extracted — the content was provided as images]\n\
             Generated image file /tmp/read_1/page_1.jpg (800x1000, JPEG). Image content attached \
             to the conversation as a native image.",
            "the answer keeps its text and gains the image annotation: {tool_result:?}"
        );
        assert!(
            history.iter().any(|m| m.role == crate::ChatRole::User
                && m.content.contains("[IMAGE:data:image/jpeg;base64,page1]")),
            "the page is injected as a synthetic user message"
        );
    }

    #[tokio::test]
    async fn commit_tool_results_derives_generated_image_and_tags_it() {
        crate::util::test::init_test_stores().await;

        // A real on-disk raster at an absolute path so the derivation can compress it.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let png_path = dir.path().join("generated.png");
        std::fs::write(&png_path, crate::util::test::noisy_png(4, 4)).expect("write png");
        let abs = std::fs::canonicalize(&png_path).expect("canonicalize");

        // A media tool with no payload of its own: the derived-image path exists for such
        // output — image_gen, chrome and computer ship their own and never reach it.
        let tool = Box::new(MediaTestTool {
            name: "media_probe",
            marker: "[IMAGE:",
        }) as Box<dyn Tool>;
        let mut agent = make_agent(vec![tool]);

        let marker = format!("[IMAGE:{}]", abs.display());
        let tool_calls = vec![ToolCall {
            id: "callgen".into(),
            name: "media_probe".into(),
            arguments: serde_json::json!({}),
        }];
        // The trait default: a media tool's marker line is a claim about its image.
        let outcomes = vec![ToolExecutionOutcome {
            output: marker.clone(),
            success: true,
            image_payloads: Vec::new(),
            text_is_content: false,
            preserve_text: false,
            suspended: false,
            ends_turn: false,
        }];

        // Delivery via the raw-output marker is preserved; read before the commit.
        let media = extract_media_from_outcomes(&agent.tools, &tool_calls, &outcomes);
        assert_eq!(
            media,
            vec![("[IMAGE:", abs.display().to_string())],
            "marker preserved for user delivery: {media:?}"
        );

        agent
            .commit_tool_results(&tool_calls, outcomes)
            .await
            .expect("commit_tool_results must succeed");

        let history = agent.session.history();
        // The derived image is injected as a synthetic user message AFTER the tool result.
        let image_users: Vec<&str> = history
            .iter()
            .filter(|m| {
                m.role == crate::ChatRole::User
                    && m.content.contains("[IMAGE:data:image/jpeg;base64,")
            })
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            image_users.len(),
            1,
            "generated image injected once: {image_users:?}"
        );
        assert!(
            image_users[0]
                .starts_with("<injected-tool-result-image>\n[IMAGE:data:image/jpeg;base64,"),
            "injected image carries the provenance tag: {image_users:?}"
        );

        // The tool-result uses the Generated verb, not "Read".
        let tool_results: Vec<String> = history
            .iter()
            .filter(|m| m.role == crate::ChatRole::Tool)
            .map(|m| {
                let v: serde_json::Value =
                    serde_json::from_str(&m.content).expect("tool result JSON");
                crate::util::json::get_str(&v, "content")
                    .expect("content")
                    .to_string()
            })
            .collect();
        assert_eq!(tool_results.len(), 1);
        assert!(
            tool_results[0].starts_with("Generated image file"),
            "generated annotation: {tool_results:?}"
        );
        assert!(
            !tool_results[0].starts_with("Read image file"),
            "must not be a Read annotation: {tool_results:?}"
        );
    }

    /// (c) A non-durable dangling call is re-executed and settles with the ORIGINAL call
    /// id, contiguously after the frame.
    #[tokio::test]
    async fn complete_pending_tool_calls_reexecutes_non_durable_call() {
        crate::util::test::init_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("seed.txt"), "completion-marker\n").unwrap();
        let ws = crate::Workspace::from_path(dir.path());

        // Emission-time frame: the caller's session durably records the read
        // call; its result is absent (the "dangling" state).
        let mut session = Session::default();
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_read_c".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({"path": "seed.txt"}),
            }],
            None,
        )
        .to_string();
        session
            .persist_messages(
                "test-read-reexec",
                &[
                    ChatMessage::user("read the file"),
                    ChatMessage::assistant(frame),
                ],
            )
            .await
            .unwrap();

        let mut agent = make_agent_on(
            vec![Box::new(crate::tools::ReadTool::general())],
            "test-read-reexec",
            ws,
        );
        agent.session = session;
        let settled = agent.complete_pending_tool_calls().await.unwrap();
        assert!(settled, "a re-executed non-durable call settles a result");

        // The result sits contiguous after the frame with the ORIGINAL call id.
        let history = agent.session.history();
        let roles: Vec<crate::ChatRole> = history.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                crate::ChatRole::User,
                crate::ChatRole::Assistant,
                crate::ChatRole::Tool
            ],
            "result contiguous after the frame: {roles:?}"
        );
        let payload: crate::ToolResultPayload = serde_json::from_str(&history[2].content).unwrap();
        assert_eq!(
            payload.tool_call_id, "call_read_c",
            "re-execution keeps the ORIGINAL call id"
        );
        assert!(
            payload.content.contains("completion-marker"),
            "real tool output recorded: {}",
            payload.content
        );
        assert!(agent.session.pending_tool_frame().is_none());
    }

    /// (d) Full drain lifecycle: a sync analyze dispatch under an active drain
    /// leaves the frame durably persisted with NO tool-result row and the
    /// caller-owned job `launched` (the result is absent); after `drain_clear`,
    /// the completion resumes the durable job — a reconstructable Done slot is
    /// replayed (zero LLM calls), the consolidated tool result settles
    /// contiguously after the frame, and only then is the resumed job terminalized.
    #[tokio::test]
    #[serial_test::serial(provider, drain)] // serializes the process-global fake provider (providers::PROVIDER) + shutdown drain flag
    async fn drain_mid_sync_analyze_leaves_call_dangling_then_completion_resumes() {
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new());
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_drain_analyze");
        let pin = "drain_sync_analyze_pin";
        let conn = &crate::session::store().conn;

        // Emission-time frame: the caller's session durably records the analyze
        // call BEFORE the drain-cut execution leaves its result absent.
        let mut session = Session::default();
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_analyze_d".to_string(),
                name: "analyze".to_string(),
                arguments: serde_json::json!({"message": "analyze this"}),
            }],
            None,
        )
        .to_string();
        session
            .persist_messages(
                pin,
                &[
                    ChatMessage::user("analyze this"),
                    ChatMessage::assistant(frame),
                ],
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
            .scope(Some(pin.to_string()), async {
                tool.execute(&ws, serde_json::json!({"message": "analyze this"}))
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
                crate::db::params![pin],
            )
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1, "one launched analyze job");
        assert_eq!(jobs[0].get::<String>(1).unwrap(), "launched");
        assert_eq!(
            jobs[0].get::<String>(2).unwrap(),
            pin,
            "job is caller-owned by the session pin"
        );
        let job_id = jobs[0].get::<String>(0).unwrap();
        let session_rows = conn
            .query(
                "SELECT role FROM sessions WHERE agent_id = ?1 ORDER BY id",
                crate::db::params![pin],
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

        // Run the completion: it resumes the durable job (Done slot replayed)
        // and settles the consolidated result after the frame, then
        // terminalizes the resumed job.
        let mut agent = make_agent_on(vec![], pin, ws);
        agent.session = session;
        let settled = agent.complete_pending_tool_calls().await.unwrap();
        assert!(settled, "resumed durable job settles a result");

        // The resumed job is terminalized; zero LLM calls were made.
        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = ?1",
                crate::db::params![job_id.clone()],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "resumed job must be terminalized");
        assert!(
            fake.request_fingerprints.lock().unwrap().is_empty(),
            "replaying a Done slot must not call the model"
        );

        // Tool result contiguous after the frame (in-memory + durable).
        let history = agent.session.history();
        let roles: Vec<crate::ChatRole> = history.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                crate::ChatRole::User,
                crate::ChatRole::Assistant,
                crate::ChatRole::Tool
            ],
            "tool result after the frame: {roles:?}"
        );
        let payload: crate::ToolResultPayload = serde_json::from_str(&history[2].content).unwrap();
        assert_eq!(payload.tool_call_id, "call_analyze_d");
        assert!(
            payload.content.contains("ANALYST_RAW"),
            "{}",
            payload.content
        );
        let pending_jobs = conn
            .query(
                "SELECT id FROM pending_jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(pending_jobs.is_empty(), "no envelope pending after resume");
    }

    /// (e) Two same-kind (analyze) calls in one dangling frame bind to the two
    /// OLDEST matching caller-owned jobs (rposition — jobs are newest-first,
    /// frame calls are in dispatch order), so the first frame call carries the
    /// older job's result and the second the newer; each resumed job is
    /// terminalized and its checkpointed outcome settles against its own call
    /// id. A THIRD launched job for the pin that binds no call is dropped with the
    /// settle, so a later, unrelated call can never bind it.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn complete_pending_tool_calls_binds_same_kind_calls_to_distinct_jobs() {
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new());
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_two_analyze_jobs");
        let pin = "two_analyze_jobs_pin";
        let conn = &crate::session::store().conn;

        // Caller session with a TWO-call analyze frame (both ids dangling).
        let mut session = Session::default();
        let frame = assistant_replay_payload(
            Some(""),
            &[
                crate::ToolCall {
                    id: "call_analyze_g1".to_string(),
                    name: "analyze".to_string(),
                    arguments: serde_json::json!({"message": "analyze task one"}),
                },
                crate::ToolCall {
                    id: "call_analyze_g2".to_string(),
                    name: "analyze".to_string(),
                    arguments: serde_json::json!({"message": "analyze task two"}),
                },
            ],
            None,
        )
        .to_string();
        session
            .persist_messages(
                pin,
                &[
                    ChatMessage::user("analyze this"),
                    ChatMessage::assistant(frame),
                ],
            )
            .await
            .unwrap();

        // The frame row is backdated so all three seeded jobs fall inside the
        // frame's OWN window (created after it): the pool itself has no age gate,
        // but a pool entry older than the frame belongs to an earlier frame and
        // is abandoned with the settle rather than bound. Seed THREE caller-owned
        // analyze jobs, each with an all-Done roster slot (distinct stored
        // outcomes) so each resume core replays Terminal with zero LLM calls.
        // The two frame calls bind the two OLDEST jobs (first call →
        // `two_analyze_job_0`, second → `two_analyze_job_1`);
        // `two_analyze_job_2` binds no call and is dropped with the settle.
        conn.execute(
            "UPDATE sessions SET created_at = '2024-01-01T00:00:00+00:00' \
             WHERE agent_id = ?1 AND role = 'assistant'",
            crate::db::params![pin],
        )
        .await
        .unwrap();
        let jobs: [(&str, &str); 3] = [
            ("two_analyze_job_0", "RESULT_ONE"),
            ("two_analyze_job_1", "RESULT_TWO"),
            ("two_analyze_job_2", "RESULT_THREE"),
        ];
        for (i, (job_id, outcome)) in jobs.iter().copied().enumerate() {
            let task = format!("analyze task {i}");
            let analyst_id = format!("{job_id}_analyst");
            crate::jobs::spawn_job(
                conn,
                job_id,
                &task,
                &ws.name,
                "caller-user",
                "telegram",
                crate::Role::Engineer,
                &[crate::jobs::NewAgent {
                    agent_id: analyst_id.clone(),
                    kind: crate::jobs::AgentKind::Analyst,
                    idx: Some(0),
                    task: task.clone(),
                }],
                &crate::jobs::SpawnChild::Analyze,
                Some(pin),
                None,
                None,
            )
            .await
            .unwrap();
            crate::jobs::write_agent_outcome(
                conn,
                job_id,
                &analyst_id,
                crate::jobs::RowStatus::Done,
                Some(outcome),
            )
            .await
            .unwrap();
            // Distinct, strictly-increasing created_at so the rposition
            // newest-first binding is deterministic at second precision.
            let created = format!("2024-01-0{}T00:00:00+00:00", i + 2);
            conn.execute(
                "UPDATE jobs SET created_at = ?1 WHERE id = ?2",
                crate::db::params![created, job_id],
            )
            .await
            .unwrap();
        }

        // Run the completion: both frame calls are resumed (one per call,
        // DISTINCT — the OLDEST first), the two bound jobs terminalized, and
        // the two results settled with their own call ids.
        let mut agent = make_agent_on(vec![], pin, ws);
        agent.session = session;
        let settled = agent.complete_pending_tool_calls().await.unwrap();
        assert!(settled, "two resumed jobs settle results");

        assert!(
            fake.request_fingerprints.lock().unwrap().is_empty(),
            "replaying Done slots must not call the model"
        );
        assert!(agent.session.pending_tool_frame().is_none());

        // Both bound jobs are terminalized and the third is dropped with the
        // settle — no call can ever bind it (a call's round kind is resolved from
        // the call itself), so it must not stay in the pool.
        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE caller_agent_id = ?1 AND kind = 'analyze' AND status = 'launched'",
                crate::db::params![pin],
            )
            .await
            .unwrap();
        assert!(
            jobs.is_empty(),
            "the unbound job is dropped with the settle: a later call must never bind it"
        );

        // Both results settled with the frame's ORIGINAL call ids and the
        // per-call mapping is exact: the FIRST frame call (call_analyze_g1)
        // carries the OLDEST job's result (RESULT_ONE), the second
        // (call_analyze_g2) the newer job's (RESULT_TWO) — rposition binding,
        // not set equality.
        let history = agent.session.history();
        let mut per_call: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for msg in history {
            if msg.role != crate::ChatRole::Tool {
                continue;
            }
            let payload: crate::ToolResultPayload = serde_json::from_str(&msg.content).unwrap();
            per_call.insert(payload.tool_call_id, payload.content);
        }
        assert_eq!(per_call.len(), 2, "both calls settled");
        assert_eq!(
            per_call.get("call_analyze_g1").map(String::as_str),
            Some("RESULT_ONE"),
            "first frame call binds the OLDEST job"
        );
        assert_eq!(
            per_call.get("call_analyze_g2").map(String::as_str),
            Some("RESULT_TWO"),
            "second frame call binds the newer job"
        );
    }

    /// (f) A frame whose calls bind no job at all still drops the pool's
    /// leftovers: the launched row of a cut-mid-flight call the tool no longer
    /// knows is unreachable (nothing matches it by name any more), so it is
    /// abandoned with the settle instead of staying in the pool for a later,
    /// unrelated call to bind and settle an old report as its own result. The cut
    /// call itself is simply repeated — never aliased.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn complete_pending_tool_calls_drops_a_leftover_no_call_can_bind() {
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new());
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_leftover_pool");
        let pin = "leftover_pool_pin";
        let conn = &crate::session::store().conn;

        // One dangling call to a tool this build does not have.
        let mut session = Session::default();
        let frame = assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_retired".to_string(),
                name: "retired_tool".to_string(),
                arguments: serde_json::json!({"message": "old work"}),
            }],
            None,
        )
        .to_string();
        session
            .persist_messages(
                pin,
                &[
                    ChatMessage::user("do the old thing"),
                    ChatMessage::assistant(frame),
                ],
            )
            .await
            .unwrap();

        // The launched row that call left behind.
        crate::jobs::spawn_job(
            conn,
            "leftover_job",
            "old work",
            &ws.name,
            "caller-user",
            "telegram",
            crate::Role::Engineer,
            &[crate::jobs::NewAgent {
                agent_id: "leftover_job_analyst".to_string(),
                kind: crate::jobs::AgentKind::Analyst,
                idx: Some(0),
                task: "old work".to_string(),
            }],
            &crate::jobs::SpawnChild::Analyze,
            Some(pin),
            None,
            None,
        )
        .await
        .unwrap();

        let mut agent = make_agent_on(vec![], pin, ws);
        agent.session = session;
        let settled = agent.complete_pending_tool_calls().await.unwrap();
        assert!(settled, "the re-executed call settles its own result");

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE id = 'leftover_job'",
                crate::db::params![],
            )
            .await
            .unwrap();
        assert!(
            jobs.is_empty(),
            "the unbound launched row is abandoned with the settle, not left for a later call"
        );
    }

    /// (g) A pool entry OLDER than the frame belongs to an earlier frame, and this
    /// frame's call must never bind it — that would settle the old round's report as its
    /// own result. It is abandoned with the settle and the call runs fresh.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn complete_pending_tool_calls_never_binds_a_job_older_than_its_frame() {
        let fake = std::sync::Arc::new(crate::util::test::FakeProvider::new());
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/test_ws_frame_window");
        let pin = "frame_window_pin";
        let conn = &crate::session::store().conn;

        // One dangling `send` worker call, emitted AFTER the stale round below.
        let mut session = Session::default();
        let frame = assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_worker_now".to_string(),
                name: "send".to_string(),
                arguments: serde_json::json!({"message": "the new task", "addressee": "worker"}),
            }],
            None,
        )
        .to_string();
        session
            .persist_messages(
                pin,
                &[
                    ChatMessage::user("code the new task"),
                    ChatMessage::assistant(frame),
                ],
            )
            .await
            .unwrap();

        // The leftover of an EARLIER frame: a launched round with a stored outcome.
        crate::jobs::spawn_job(
            conn,
            "stale_frame_job",
            "the old task",
            &ws.name,
            "caller-user",
            "telegram",
            crate::Role::Engineer,
            &[crate::jobs::NewAgent {
                agent_id: "stale_frame_job_worker".to_string(),
                kind: crate::jobs::AgentKind::Worker,
                idx: Some(0),
                task: "the old task".to_string(),
            }],
            &crate::jobs::SpawnChild::Implement,
            Some(pin),
            None,
            None,
        )
        .await
        .unwrap();
        crate::jobs::write_agent_outcome(
            conn,
            "stale_frame_job",
            "stale_frame_job_worker",
            crate::jobs::RowStatus::Done,
            Some("STALE_RESULT"),
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE jobs SET created_at = '2024-01-01T00:00:00+00:00' WHERE id = 'stale_frame_job'",
            crate::db::params![],
        )
        .await
        .unwrap();

        let mut agent = make_agent_on(vec![], pin, ws);
        agent.session = session;
        let settled = agent.complete_pending_tool_calls().await.unwrap();
        assert!(settled, "the call settles a result of its own");

        let mut per_call: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for msg in agent.session.history() {
            if msg.role != crate::ChatRole::Tool {
                continue;
            }
            let payload: crate::ToolResultPayload = serde_json::from_str(&msg.content).unwrap();
            per_call.insert(payload.tool_call_id, payload.content);
        }
        let content = per_call.get("call_worker_now").expect("the call settles");
        assert!(
            !content.contains("STALE_RESULT"),
            "the earlier frame's report must never become this call's result: {content}"
        );

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE caller_agent_id = ?1 AND status = 'launched'",
                crate::db::params![pin],
            )
            .await
            .unwrap();
        assert!(
            jobs.is_empty(),
            "the pre-frame row is abandoned with the settle, not left for a later call"
        );
    }

    /// Seed a spill file for a fresh test-only agent id plus a run session tracker
    /// holding one name; the temp dir outlives the guard that reclaims it.
    async fn seed_run_owned_state(
        agent_id: &str,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        std::sync::Arc<crate::tools::chrome::ChromeRunSessions>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let spill = dir.path().join("spill.txt");
        std::fs::write(&spill, "spilled output").unwrap();
        CURRENT_TOOL_AGENT_ID
            .scope(Some(agent_id.to_string()), async {
                crate::tools::shell::record_spill_owner(spill.clone());
            })
            .await;
        // The run's own tracker (`for_run`), holding a name as a `ChromeTool` would.
        let sessions = crate::tools::chrome::ChromeRunSessions::for_run(agent_id);
        sessions.track(&format!("{}default", sessions.namespace()));
        (dir, spill, sessions)
    }

    /// A run whose future is dropped — a deadline abort or a panic — still reclaims its
    /// spill files and hands its chrome sessions to the release queue.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn run_end_guard_cleans_up_a_dropped_run() {
        let agent_id = format!("run-end-guard-{:016x}", rand::random::<u64>());
        let (_dir, spill, sessions) = seed_run_owned_state(&agent_id).await;
        chrome_release::clear_pending_releases();
        // The hand-off persists its record, and this test has no store of its own.
        let _no_store = chrome_release::no_release_store();

        drop(RunEndCleanup::new(
            agent_id,
            std::sync::Arc::clone(&sessions),
        ));

        assert!(!spill.exists(), "the run's spill file is reclaimed");
        assert_eq!(
            chrome_release::pending_names_and_attempts(),
            vec![(vec![format!("{}default", sessions.namespace())], 0)],
            "the ended run handed its session over for release"
        );
        chrome_release::clear_pending_releases();
    }

    /// Every end hands its sessions over, and the ends `chrome_release::holds_run_end`
    /// names hold them back.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_ended_run_hands_its_sessions_over_and_holds_only_a_resumed_run() {
        use crate::tools::chrome_release::RELEASE_HOLD_REDRIVEN_RUN;

        chrome_release::clear_pending_releases();
        // Every end here persists its record, and this test has no store of its own.
        let _no_store = chrome_release::no_release_store();
        let cut_id = format!("run-end-keep-{:016x}", rand::random::<u64>());
        let (_cut_dir, _cut_spill, cut_sessions) = seed_run_owned_state(&cut_id).await;
        let mut cut = RunEndCleanup::new(cut_id, std::sync::Arc::clone(&cut_sessions));
        cut.ended("drain");
        drop(cut);
        assert!(
            chrome_release::pending_release_delays()[0]
                >= RELEASE_HOLD_REDRIVEN_RUN.saturating_sub(std::time::Duration::from_secs(5)),
            "a run the daemon cut off holds its release back for its resumed segment"
        );

        chrome_release::clear_pending_releases();
        let pause_id = format!("run-end-pause-{:016x}", rand::random::<u64>());
        let (_pause_dir, _pause_spill, paused_sessions) = seed_run_owned_state(&pause_id).await;
        let mut paused = RunEndCleanup::new(pause_id, std::sync::Arc::clone(&paused_sessions));
        paused.ended("pause");
        drop(paused);
        assert!(
            chrome_release::pending_release_delays()[0]
                >= RELEASE_HOLD_REDRIVEN_RUN.saturating_sub(std::time::Duration::from_secs(5)),
            "a run frozen by a workspace pause resumes at unpause under its own id"
        );

        chrome_release::clear_pending_releases();
        let final_id = format!("run-end-release-{:016x}", rand::random::<u64>());
        let (_dir, _spill, sessions) = seed_run_owned_state(&final_id).await;
        let mut ended = RunEndCleanup::new(final_id, std::sync::Arc::clone(&sessions));
        ended.ended("transport");
        drop(ended);
        assert!(
            chrome_release::pending_release_delays()[0] < std::time::Duration::from_secs(1),
            "an end that is final releases at once"
        );
        chrome_release::clear_pending_releases();
    }

    /// Both read-only, so they run in one parallel group; `hang` never returns and
    /// announces that it started.
    struct UrgentTestTool {
        tool_name: &'static str,
        result: String,
        started: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    }

    #[async_trait]
    impl Tool for UrgentTestTool {
        fn name(&self) -> &'static str {
            self.tool_name
        }

        fn description(&self) -> String {
            "test".into()
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }

        fn side_effects(&self) -> bool {
            false
        }

        async fn execute(
            &self,
            _ws: &crate::Workspace,
            _args: serde_json::Value,
        ) -> anyhow::Result<String> {
            match &self.started {
                Some(started) => {
                    started.store(true, std::sync::atomic::Ordering::SeqCst);
                    std::future::pending().await
                }
                None => Ok(self.result.clone()),
            }
        }
    }

    /// A turn cut by an urgent message: the returned call keeps its result, the in-flight
    /// one settles with the fixed interrupted result, nothing is replayed, and the phrase is
    /// the session's whole record of the interruption — the reply is empty, not a failure,
    /// not a cancellation.
    #[tokio::test]
    #[serial_test::serial(provider, drain)]
    async fn urgent_interrupt_records_the_phrase_and_returns_no_reply() {
        crate::util::test::init_test_stores().await;
        let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fake = std::sync::Arc::new(FakeProvider::new().ok_text_and_tool_calls(
            "",
            &[
                ("quick", serde_json::json!({})),
                ("hang", serde_json::json!({})),
            ],
        ));
        let _provider_guard = install_fake_provider(fake);
        let mut agent = make_agent_with_role(
            vec![
                Box::new(UrgentTestTool {
                    tool_name: "quick",
                    result: "the sibling's own result".into(),
                    started: None,
                }),
                Box::new(UrgentTestTool {
                    tool_name: "hang",
                    result: String::new(),
                    started: Some(std::sync::Arc::clone(&started)),
                }),
            ],
            crate::Role::Engineer,
        );
        agent.agent_id = "e2e_urgent_interrupt".to_string();
        let interrupt = agent.interrupt_token.clone();
        let turn = tokio::spawn(async move {
            let answer = agent.work(&["do two things"], false).await;
            (agent, answer)
        });
        while !started.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        interrupt.cancel();

        let (agent, answer) = turn.await.expect("the turn task must not panic");
        assert_eq!(
            answer.expect("an interrupted turn is not a failure"),
            "",
            "the turn hands back no reply: nothing is delivered"
        );
        assert!(
            agent.turn_state.interrupted,
            "the router must skip delivery for this turn"
        );
        assert!(
            !agent.cancel_token.is_cancelled(),
            "the interrupt is not a cancellation"
        );

        let history = agent.session.history().to_vec();
        let results: Vec<String> = history
            .iter()
            .filter(|message| message.role == crate::ChatRole::Tool)
            .map(|message| {
                let payload: serde_json::Value =
                    serde_json::from_str(&message.content).expect("tool result is JSON");
                crate::util::json::get_str(&payload, "content")
                    .expect("tool result has content")
                    .to_string()
            })
            .collect();
        assert_eq!(results.len(), 2, "every call of the cut round has a result");
        assert_eq!(
            results[0], "the sibling's own result",
            "a call that had already returned keeps its real result"
        );
        assert_eq!(
            results[1],
            crate::prompt::load_prompt("tool_interrupted.md").trim(),
            "the cut call is settled with the one fixed interrupted result"
        );
        assert_eq!(
            history.last().map(|message| message.content.as_str()),
            Some("interrupted by the user"),
            "the phrase is the session's whole record of the interruption"
        );
        assert!(
            agent.session.pending_tool_frame().is_none(),
            "no call is left dangling for a later turn to replay"
        );
    }
}
