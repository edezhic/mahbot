use std::fmt::Write;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use tracing::Instrument;

use crate::providers::reasoning_roundtrip::assistant_replay_payload;
use crate::session::Session;
use crate::tools::{
    ToolExecutionOutcome, find_tool, format_tool_failure_feedback, normalize_tool_call,
    scrub_tool_output,
};
use crate::util::{MEDIA_MARKER_RE, UnwrapPoison, parse_media_marker, scrub_credentials};
use crate::{Agent, ChatMessage, ChatRequest, ChatResponse, Tool, ToolCall};

// ── Per-tool-call user context ─────────────────────────────────────
// Set by the Agent work loop before each tool execute(), read by tools
// that need user context (e.g. AnalyzeTool async dispatch).
//
// # Contract
// - Set by the Agent work loop before each tool execution.
// - Must be read synchronously (before any `tokio::spawn` boundary) when
//   the value is needed across an async boundary.
// - Falls back to `unwrap_or_default()` → empty string for background
//   agents (management, maintainer, workspace) that have no user context.
//
// # Example (in a tool's execute method)
// ```ignore
// let user_name = CURRENT_TOOL_USER_NAME
//     .try_with(|n| n.clone())
//     .unwrap_or_default();
// ```
tokio::task_local! {
    pub(crate) static CURRENT_TOOL_USER_NAME: String;
    pub(crate) static CURRENT_TOOL_CHANNEL: String;
    /// The calling agent's DIRECT PARENT INVOCATION grouping key (ticket /
    /// analyze round / research run) — set for the duration of tool execution so
    /// tools that spawn sub-agents (e.g. implement) can propagate the caller's
    /// group to the Running Agents view. `None` = workspace singleton caller.
    pub(crate) static CURRENT_TOOL_PARENT_KEY: Option<crate::registry::ParentKey>;
    /// The calling agent's DIRECT PARENT INVOCATION human-readable label
    /// (ticket title / analyze question / research question) — set alongside
    /// [`CURRENT_TOOL_PARENT_KEY`] for the duration of tool execution so tools
    /// that spawn sub-agents can propagate the caller's group header label to
    /// the Running Agents view. Purely presentational; `None` for workspace
    /// singletons.
    pub(crate) static CURRENT_TOOL_PARENT_LABEL: Option<String>;
    /// The calling agent's background shell session registry — set for the
    /// duration of tool execution so the shell tool (Full roles only) can
    /// register/stop background sessions that are force-killed when the agent
    /// is dropped. `None` outside an agent run (management diagnostics,
    /// tests) — background mode is unavailable there.
    pub(crate) static CURRENT_TOOL_BACKGROUND_SESSIONS:
        Option<std::sync::Arc<crate::tools::shell::BackgroundSessions>>;
    /// The calling agent's id — set for the duration of tool execution so the
    /// shell tool can attribute the spill files it creates to the owning agent
    /// (owner-deletes-at-end: the run's spill files are deleted when the agent
    /// run ends). `None` outside an agent run (management diagnostics, tests).
    pub(crate) static CURRENT_TOOL_AGENT_ID: Option<String>;
    /// The calling agent's registry identity + generation — set for the
    /// duration of tool execution so tool-internal LLM calls (e.g. media
    /// transcription in video tool results) can attribute themselves to the
    /// owning agent's Running Agents card and telemetry. `None` outside an
    /// agent run (inbound enrichment, tests) — such calls register in
    /// [`crate::call_registry::NON_AGENT_CALLS`] instead.
    pub(crate) static CURRENT_TOOL_AGENT_TRACKING:
        Option<crate::registry::AgentTracking>;
}

/// Maximum number of completed tool rounds before the agent loop bails out.
///
/// A tool round is the LLM call(s) that returned tool calls plus the execution
/// and commit of that tool group. The counter increments only after a committed
/// tool group; the final-answer LLM call and the image-rejection strip
/// `continue` do not increment it, so an agent can legitimately make more LLM
/// calls than this cap. These semantics are deliberate — do not "fix" the
/// counter (or the strip path) to count raw LLM calls while touching this file.
///
/// **DO NOT REDUCE this value without benchmarking against real ticket iteration
/// distributions in this codebase.**
///
/// # Why this is intentionally generous (1000)
///
/// * Agents routinely consume hundreds of tool rounds in normal, legitimate
///   work: multi-file editing with compile-error feedback loops, research →
///   implement → review cycles, and sequential tool dependencies all require
///   many turns — this is deliberate problem-solving, not a runaway loop.
///
/// * Cost is not a concern. Even a full 1000-round run with the default model
///   (DeepSeek V4 Flash) costs well under $1, so there is zero cost reason to
///   lower the limit.
///
/// * Running to the tool-round cap is EXTREMELY rare with modern models. The
///   limit exists only as a safety net for pathological edge cases — it is not
///   protecting against a common or recurring problem.
///
/// * Reducing the cap prematurely would cause legitimate long-running agents to
///   fail mid-task, wasting more time and tokens on restarts than the tool
///   rounds themselves would have consumed.
///
/// Value last intentionally reviewed: 2026-07-11
const MAX_LLM_ITERATIONS: usize = 1000;

/// Maximum length of serialized arguments stored in per-call stats.
/// Longer arguments are truncated at a UTF-8-safe boundary.
const MAX_STATS_ARG_LENGTH: usize = 500;

/// Retry-exhaustion marker shared by the producer contexts
/// (`llm_call`/`summarize`) and the `engineer_failure_comment` match; drift is
/// cosmetic-only (typed classification uses `RetryExhausted`).
pub(crate) const RETRY_EXHAUSTION_MARKER: &str = "exhausted retry budget";

/// Extract file paths from successful media-generation tool outcomes.
///
/// Scans the zipped tool calls and outcomes for media-generation tools,
/// parsing the output for their media marker prefixes (e.g. `[IMAGE:path]`,
/// `[VIDEO:path]`) and returning `(marker_prefix, path)` pairs.
///
/// Only successfully-executed tools with a defined [`Tool::media_marker`] are
/// inspected. Non-media tools and failed outcomes are silently skipped.
///
/// Limitation: file paths containing `]` are truncated by the regex — not a
/// concern in practice, as media-generation tools produce temporary files with safe names.
fn extract_media_from_outcomes(
    tools: &[Box<dyn Tool>],
    tool_calls: &[ToolCall],
    outcomes: &[ToolExecutionOutcome],
) -> Vec<(&'static str, String)> {
    let mut paths = Vec::new();
    for (call, outcome) in tool_calls.iter().zip(outcomes.iter()) {
        if outcome.success
            && let Some(marker_prefix) = find_tool(tools, &call.name).and_then(Tool::media_marker)
        {
            // Derive the regex kind from the marker prefix, e.g. "[IMAGE:" -> "IMAGE".
            // This relies on the documented invariant that media_marker() returns "[KIND:".
            let kind = &marker_prefix[1..marker_prefix.len() - 1];
            let mut matched = false;
            for caps in MEDIA_MARKER_RE.captures_iter(&outcome.output) {
                let (captured_kind, path) = parse_media_marker(&caps);
                if captured_kind == kind {
                    matched = true;
                    paths.push((marker_prefix, path.to_string()));
                }
            }
            if !matched {
                tracing::warn!(
                    media_tool = %call.name,
                    marker = %marker_prefix,
                    "Could not parse media path from tool output — skipping media marker",
                );
            }
        }
    }
    paths
}

/// One-pass derivation of a role's advertised tools and their specs — the
/// single source for both [`Agent::new`] and the research wrap-up snapshot
/// (`.1`), so the frozen post-deadline replay can never drift from the live
/// agent's tools (KV-cache byte identity).
#[must_use]
pub(crate) fn role_tools_and_specs(
    role: crate::Role,
    ws: &crate::Workspace,
) -> (Vec<Box<dyn Tool>>, Vec<crate::ToolSpec>) {
    let tools: Vec<Box<dyn Tool>> = role
        .tools(ws)
        .into_iter()
        .filter(|t| t.is_advertised())
        .collect();
    let tool_specs = tools.iter().map(|t| t.spec()).collect();
    (tools, tool_specs)
}

/// Build the byte-relevant chat params (model, tools, reasoning_effort,
/// routing, max_tokens) for a role — the single source shared by
/// [`Agent::build_chat_request`], the research wrap-up snapshot, and
/// [`crate::tools::research::orchestrator_params`], so all three replay the
/// same KV-cache prefix. `meta` is telemetry-only (never part of the provider
/// request body) and attached by call sites.
#[must_use]
pub(crate) fn chat_request(
    role: crate::Role,
    tool_specs: Option<Vec<crate::ToolSpec>>,
    messages: Vec<ChatMessage>,
    allow_image_parts: bool,
) -> ChatRequest {
    let model = crate::config::CONFIG.role_model(role);
    let routing = crate::config::CONFIG.model_routing(&model);
    ChatRequest {
        messages,
        tools: tool_specs,
        model,
        allow_image_parts,
        max_tokens: Some(crate::DEFAULT_MAX_TOKENS),
        reasoning_effort: Some(
            crate::role::role_info(&role)
                .default_reasoning_effort
                .to_string(),
        ),
        provider_order: routing.provider_order,
        meta: None,
    }
}

impl Agent {
    /// Create a new agent with the given agent_id, role, workspace, and optional ticket.
    ///
    /// Tools are derived from [`crate::Role`] via [`crate::Role::tools`].
    /// Automatically registers with [`crate::registry::AGENT_REGISTRY`] and creates an
    /// internal [`tokio_util::sync::CancellationToken`]. The agent is deregistered on [`Drop`].
    ///
    /// `parent_key` carries the DIRECT PARENT INVOCATION grouping key for the
    /// Running Agents view (ticket / analyze round / research run). `None` means
    /// the agent is a workspace singleton (manager / maintainer / discovery /
    /// direct chat) — ticket agents pass the ticket and get
    /// [`ParentKey::Ticket`] implicitly.
    ///
    /// `parent_label` is the human-readable label of that parent invocation
    /// (ticket title / analyze question / research question) — purely
    /// presentational. `None` falls back to the ticket title when the effective
    /// parent is a ticket; callers of analyze/research rounds pass the
    /// question/task text explicitly.
    #[must_use]
    #[expect(clippy::too_many_arguments)] // one positional arg per construction field; callers use literals
    pub fn new(
        agent_id: String,
        role: crate::Role,
        ws: &crate::Workspace,
        ticket: Option<crate::board::Ticket>,
        user_name: String,
        channel: String,
        parent_key: Option<crate::registry::ParentKey>,
        parent_label: Option<String>,
    ) -> Self {
        let (tools, tool_specs) = role_tools_and_specs(role, ws);

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let label = if let Some(ref t) = ticket {
            format!("{}: {}", role.as_str(), t.title)
        } else {
            role.to_string()
        };
        // Ticket agents group by their ticket; an explicit parent key (analyze
        // round / research run) takes precedence when both are present.
        let parent_key = parent_key.or_else(|| {
            ticket
                .as_ref()
                .map(|t| crate::registry::ParentKey::Ticket(t.id.clone()))
        });
        // The group header label: an explicit parent label (analyze/research
        // question) wins; a ticket-parented agent without one falls back to
        // the ticket title. Purely presentational — never affects behavior.
        let parent_label = parent_label.or_else(|| match parent_key {
            Some(crate::registry::ParentKey::Ticket(_)) => ticket.as_ref().map(|t| t.title.clone()),
            _ => None,
        });
        // Registration-time manual-cancel check: a sub-agent of a research
        // run whose cancel signal already fired (the user cancelled the run
        // from the Running Agents page) must never run a round — even when it
        // registers AFTER the registry's cancel sweep (the late-register
        // race). The token is pre-cancelled, so the agent's llm_loop bails at
        // its first iteration-top check and the round yields no response.
        if let Some(crate::registry::ParentKey::Research(run_id)) = &parent_key
            && crate::research_cancel::is_cancelled(run_id)
        {
            cancel_token.cancel();
        }
        let generation = crate::registry::AGENT_REGISTRY.register(
            agent_id.clone(),
            role.to_string(),
            ticket.as_ref().map(|t| t.id.clone()),
            ws,
            label,
            cancel_token.clone(),
            parent_key.clone(),
            parent_label.clone(),
        );

        Self {
            agent_id,
            role,
            session: Session::default(),
            workspace: Arc::new(ws.clone()),
            tools,
            tool_specs,
            cancel_token,
            ticket,
            generation,
            tool_stats: std::sync::Mutex::new(Vec::new()),
            user_name,
            channel,
            parent_key,
            parent_label,
            incoming_rx: None,
            round_ts: None,
            first_call_notify: None,
            failure: None,
            failure_class: None,
            background_sessions: std::sync::Arc::new(
                crate::tools::shell::BackgroundSessions::default(),
            ),
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if self.generation > 0 {
            crate::registry::AGENT_REGISTRY.deregister(&self.agent_id, self.generation);
        }
        // Teardown kill: the agent's background shell sessions must not
        // outlive it. Force-kill only (no grace), mirroring the existing
        // teardown-kill behavior for in-flight shell children. Synchronous —
        // SIGKILL to each live process group.
        self.background_sessions.terminate_all();
    }
}

impl Agent {
    /// Flush accumulated tool stats to the logs store, then persist the final
    /// assistant message via `session.finalize(&self.agent_id)`. Flush failures are logged
    /// but do not abort finalization.
    ///
    /// An empty unpersisted tail is logged here with full agent/role/workspace/
    /// ticket attribution: at INFO when the graceful drain cut the turn
    /// (expected and lossless — committed tool frames are durable, ticket
    /// rounds resume at boot), at WARN otherwise (genuine anomaly).
    pub async fn finalize_session(&mut self) -> anyhow::Result<()> {
        // Drain accumulated tool usage stats
        let stats = {
            let mut guard = self.tool_stats.lock().unwrap_poison();
            std::mem::take(&mut *guard)
        };
        if !stats.is_empty()
            && let Some(store) = crate::logs::LOG_STORE.get()
            && let Err(e) = store
                .flush_batch(
                    &self.agent_id,
                    self.role.as_str(),
                    &self.workspace.path,
                    &stats,
                )
                .await
        {
            tracing::warn!(
                agent_id = %self.agent_id,
                role = %self.role.as_str(),
                error = %e,
                "Failed to flush tool usage stats"
            );
        }

        // When cancelled (by user /stop or by global shutdown), no assistant
        // message is expected — skip finalization to avoid the "finalize called
        // but no assistant message" warning. Tool results were already
        // persisted via commit_tool_results inside llm_loop; any unpersisted
        // tail (failed drain) is in-memory only and dropped at the next init
        // (comments remain recoverable from the board DB).
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
                    // Graceful drain cut the turn right after the last
                    // tool-group commit: nothing was left unpersisted.
                    // Expected and lossless — committed tool frames are
                    // durable, ticket rounds resume at boot (job stays
                    // status='launched'), direct sessions resume on the next
                    // user message.
                    tracing::info!(
                        agent_id = %self.agent_id,
                        role = %self.role,
                        workspace = %self.workspace.name,
                        ticket = self.ticket.as_ref().map(|t| t.id.as_str()),
                        "Session finalize no-op: turn cut by graceful drain — \
                         committed frames are durable; resumes at boot or on the next user message"
                    );
                } else {
                    // Genuine anomaly: the turn ended with no persisted output
                    // outside a drain (e.g. an LLM call failing with no output
                    // produced). Keep the byte-identical historical message so
                    // external correlation keeps working.
                    tracing::warn!(
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

    /// Check whether cancellation has been triggered on this agent.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Human-readable failure reason with the same global-token-first ordering
    /// as [`failure_classification`]: shutdown cancels every per-agent token,
    /// so the global token must be checked before a user cancel is reported.
    #[must_use]
    pub(crate) fn failure_reason(&self, fallback: &str) -> String {
        if crate::shutdown::shutdown_token().is_cancelled() {
            "service shutting down".to_string()
        } else if self.is_cancelled() {
            "agent cancelled by user".to_string()
        } else {
            self.failure.clone().unwrap_or_else(|| fallback.to_string())
        }
    }

    /// Run a complete agent turn: initialize session, work loop (with shutdown
    /// cancellation), finalize session.
    pub async fn work(&mut self, msg: &str, resume: bool) -> anyhow::Result<String> {
        // Open or resume a session for this agent turn.
        self.session
            .init(
                &self.agent_id,
                msg,
                &self.workspace,
                &self.role,
                self.ticket.as_ref(),
                &self.channel,
                &self.user_name,
                self.round_ts.as_deref(),
            )
            .await?;

        // Surface the persisted session length on the live card: the registry
        // entry is created synchronously at `Agent::new` (no DB access there),
        // so resumed sessions would otherwise show no length until the first
        // successful LLM call of the turn. Purely observational — the page
        // reads the registry, never the database.
        if let Some(token_length) = self.session.token_length() {
            crate::registry::AGENT_REGISTRY.set_session_tokens(
                &self.agent_id,
                self.generation,
                token_length,
            );
        }

        // Pre-maybe_summarize drain check: a fresh dispatch that starts during
        // the graceful drain (or a fired shutdown token) must not destructively
        // compact an over-threshold session before the llm_loop drain
        // check fires. The round is cut before any LLM work; boot resume
        // continues the session.
        if crate::shutdown::aborting() {
            anyhow::bail!("Agent round cut short by shutdown/drain — resumes at boot");
        }

        // Mandatory: destructive over-threshold compaction is SKIPPED on resumed
        // turns — the pre-crash trail the resume was meant to preserve must
        // survive (the design's resume-flag rule; do NOT infer resume from an
        // empty message — RecoveryRetry semantics are unchanged).
        if !resume {
            self.maybe_summarize().await;
        }

        let shutdown = crate::shutdown::shutdown_token();
        let response_result = tokio::select! {
            () = shutdown.cancelled() => {
                Err(anyhow::anyhow!("Shutting down"))
            }
            result = self.llm_loop() => result,
        };

        // Always finalize session — even on global shutdown, persist progress
        if let Err(e) = self.finalize_session().await {
            tracing::error!(error = %e, "Session finalize failed");
        }

        let response = response_result?;
        Ok(response)
    }

    /// Run the full agent loop: LLM calls → tool execution → loop until final answer.
    async fn llm_loop(&mut self) -> anyhow::Result<String> {
        let span = tracing::info_span!("agent", agent_id = %self.agent_id, role = %self.role, workspace = %self.workspace.path);
        async {
            let mut iteration = 0usize;
            let mut accumulated_media_paths: Vec<(&'static str, String)> = Vec::new();
            loop {
                if self.cancel_token.is_cancelled() {
                    anyhow::bail!("Agent cancelled by user");
                }
                // Shutdown/drain: checked at iteration top (after the previous
                // tool group's commit) so the CURRENT tool group completes —
                // no new LLM call starts once the drain begins (or the token
                // fires). The round is resumed at boot via the job row
                // (status='launched').
                if crate::shutdown::aborting() {
                    anyhow::bail!("Agent round cut short by shutdown/drain — resumes at boot");
                }
                if iteration >= MAX_LLM_ITERATIONS {
                    anyhow::bail!(
                        "Agent exceeded maximum of {MAX_LLM_ITERATIONS} tool rounds \
                         — model may be stuck in a tool-calling loop"
                    );
                }

                // Drain incoming messages (e.g., ticket comments from the Manager
                // or comment tool). Messages are injected as user messages with a
                // descriptive prefix so the agent understands the source.
                self.drain_incoming_messages().await?;

                // Leader-stagger signal: fire after the FIRST LLM call
                // completes, success or failure — the wait is fail-open;
                // Notify's stored permit covers a fire-before-wait race.
                let llm_result = self.llm_call().await;
                if iteration == 0
                    && let Some(notify) = &self.first_call_notify
                {
                    notify.notify_one();
                }

                // Provider input-image content-inspection rejection (e.g.
                // OpenRouter HTTP 400 `data_inspection_failed`): the rejected
                // image must not fail the run wholesale nor stay sticky in the
                // session. Strip the image markers from the most recent user
                // message (durable-first) and continue the loop — the next
                // iteration re-runs `llm_call` with the corrected message as
                // part of the NORMAL loop, not a dedicated retry step. No
                // retry guard: a subsequent failure whose most recent user
                // message no longer contains image markers takes the normal
                // failure path (no repeated stripping).
                //
                // The leader-stagger signal above has already fired (first
                // call completed, success or failure): a stripped leader's next
                // request diverges from its followers' byte-stable prefix —
                // fail-open by design, and moot for the direct-chat Artist
                // scenario that exercises this path (management rounds never
                // send images). Note: the retried call re-runs the iteration-0
                // `notify_one` above, but the stagger has a single waiter (the
                // `select!` in `spawn_staggered_round`, already consumed) — the
                // second notification stores an unconsumed permit and is
                // dropped with the Notify. No follower is released twice.
                //
                // The downcast is extracted to a local first: `anyhow::Chain`
                // is not `Send`, so it must not be held across the await below
                // (the agent runs inside a spawned, Send-bound task).
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
                        iteration,
                        "Stripped provider-rejected input image from the most recent user \
                         message — continuing the normal loop"
                    );
                    continue;
                }

                let PreparedAssistantTurn {
                    mut display_text,
                    tool_calls,
                    history_content,
                } = prepare_assistant_turn(
                    llm_result
                        .with_context(|| format!("LLM step failed at iteration {iteration}"))?,
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

                // Execute tool calls with ordering: read-only tools can run in
                // parallel within a group; side-effecting tools run one at a time.
                // Groups execute sequentially in order — the original ordering is
                // preserved in `all_outcomes`.
                let all_outcomes = self.execute_tool_group(&tool_calls).await;

                // Track media generation outcomes for marker fallback
                accumulated_media_paths.extend(extract_media_from_outcomes(
                    &self.tools,
                    &tool_calls,
                    &all_outcomes,
                ));

                self.commit_tool_results(&tool_calls, &all_outcomes, &history_content)
                    .await?;

                iteration += 1;
            }
        }
        .instrument(span)
        .await
    }

    /// Execute a batch of tool calls respecting side-effect ordering.
    ///
    /// Read-only tools run in parallel within groups; side-effecting tools
    /// run one at a time. Groups execute sequentially in the original call
    /// order. The returned outcomes correspond one-to-one with `tool_calls`.
    async fn execute_tool_group(&self, tool_calls: &[ToolCall]) -> Vec<ToolExecutionOutcome> {
        // Determine side_effects for each tool call. Unknown tools are
        // conservatively treated as side-effecting (default: true).
        let side_flags: Vec<bool> = tool_calls
            .iter()
            .map(|call| find_tool(&self.tools, &call.name).is_none_or(super::Tool::side_effects))
            .collect();

        let mut outcomes: Vec<ToolExecutionOutcome> = Vec::with_capacity(tool_calls.len());
        let mut i = 0usize;

        // Set task-locals once for the entire batch of tool executions, rather
        // than per-call inside execute_tool.  This avoids a race when multiple
        // read-only tools execute concurrently via join_all in the same task:
        // per-call scopes would interleave, causing concurrent tools to read
        // each other's user context.
        let user_name = self.user_name.clone();
        let channel = self.channel.clone();
        let parent_key = self.parent_key.clone();
        let parent_label = self.parent_label.clone();
        let background_sessions = Some(self.background_sessions.clone());
        let agent_id = Some(self.agent_id.clone());
        let agent_tracking = Some(crate::registry::AgentTracking {
            agent_id: self.agent_id.clone(),
            generation: self.generation,
            role: self.role.as_str().to_string(),
            workspace: self.workspace.name.clone(),
        });
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
                                                                while i < tool_calls.len() {
                                                                    if side_flags[i] {
                                                                        // Side-effecting: single-call group, executed alone.
                                                                        let outcome = self
                                                                            .execute_tool(
                                                                                &tool_calls[i].name,
                                                                                tool_calls[i]
                                                                                    .arguments
                                                                                    .clone(),
                                                                            )
                                                                            .await;
                                                                        outcomes.push(outcome);
                                                                        i += 1;
                                                                    } else {
                                                                        // Read-only group: extend while consecutive calls are also read-only.
                                                                        let group_start = i;
                                                                        while i < tool_calls.len()
                                                                            && !side_flags[i]
                                                                        {
                                                                            i += 1;
                                                                        }
                                                                        let group_calls =
                                                                            &tool_calls
                                                                                [group_start..i];

                                                                        // Execute the entire read-only group in parallel.
                                                                        let group_outcomes: Vec<_> =
                                                                    futures_util::future::join_all(
                                                                        group_calls.iter().map(
                                                                            |call| {
                                                                                self.execute_tool(
                                                                                    &call.name,
                                                                                    call.arguments
                                                                                        .clone(),
                                                                                )
                                                                            },
                                                                        ),
                                                                    )
                                                                    .await;

                                                                        outcomes
                                                                            .extend(group_outcomes);
                                                                    }
                                                                }
                                                            })
                                                            .await;
                                                    })
                                                    .await;
                                            })
                                            .await;
                                    })
                                    .await;
                            })
                            .await;
                    })
                    .await;
            })
            .await;

        outcomes
    }

    /// Construct a failure outcome tuple for an error reason.
    ///
    /// Both error arms in [`Self::execute_tool`] (unknown tool and execution error) produce
    /// the same `(ToolExecutionOutcome, String)` shape; this helper
    /// eliminates the byte-for-byte duplicated construction.
    ///
    /// Assumes `reason` may contain sensitive data; scrubs before use in feedback
    /// text, tracing logs, and stats.
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
            },
            reason,
        )
    }

    /// Execute a single tool call and return the result.
    async fn execute_tool(
        &self,
        call_name: &str,
        call_arguments: serde_json::Value,
    ) -> ToolExecutionOutcome {
        // Pre-flight cancellation check: if the agent has been cancelled (e.g., by
        // user pressing Stop), bail immediately without executing the tool.  This
        // prevents side-effecting tools (shell commands, file edits, sub-agents)
        // from running after cancellation.  The `llm_loop` checks cancellation at
        // the top of each iteration, but tool calls dispatched before that check
        // can still reach this method.
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

        // Two distinct log levels for error arms (info vs debug) distinguish
        // unknown-tool failures from execution errors without string-prefix matching.
        let (outcome, error_reason) = match find_tool(&self.tools, &tool_name) {
            None => {
                let reason = format!("Unknown tool: {tool_name}");
                let duration = start.elapsed();
                tracing::info!(
                    tool = %tool_name,
                    duration_ms = duration.as_millis(),
                    success = false,
                    "Unknown tool call"
                );
                Self::failure_outcome(&tool_name, &tool_arguments, &reason)
            }
            Some(tool) => {
                // Live-view instrumentation: register this tool as currently
                // executing (purely observational — no cancellation semantics).
                // Registered ONLY here, after the pre-flight cancellation check
                // and after `find_tool` succeeded — unknown tools and
                // pre-flight-cancelled calls never show a phantom tool. The
                // guard removes the entry when execution completes (RAII).
                // Parallel read-only tools each carry their own instance, so a
                // single "current tool" slot is never a lie.
                let _live_tool = crate::registry::AGENT_REGISTRY.tool_started(
                    &self.agent_id,
                    self.generation,
                    &tool_name,
                    &tool_arguments,
                );
                let exec_result = tool.execute(&self.workspace, tool_arguments.clone()).await;
                let duration = start.elapsed();
                match exec_result {
                    Ok(output) => {
                        let output_text = if output.is_empty() {
                            String::from("(no output)")
                        } else {
                            output
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
                            },
                            String::new(),
                        )
                    }
                    Err(e) => {
                        let (outcome, error_reason) = Self::failure_outcome(
                            &tool_name,
                            &tool_arguments,
                            &format!("Error executing {tool_name}: {e}"),
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
        };

        // Inlined per-call stats recording — each tool invocation produces
        // one record with arguments, duration, success/failure, and error.
        {
            let elapsed_ms = start.elapsed().as_millis();
            let duration_ms = i64::try_from(elapsed_ms).unwrap_or(0);
            let args_str =
                serde_json::to_string(&tool_arguments).expect("Value is always serializable");
            let args_scrubbed = scrub_credentials(&args_str);
            let arguments =
                crate::util::truncate_bytes(&args_scrubbed, MAX_STATS_ARG_LENGTH).to_string();

            let mut guard = self.tool_stats.lock().unwrap_poison();
            guard.push(crate::ToolCallRecord {
                tool_name,
                arguments,
                duration_ms,
                success: outcome.success,
                error_message: (!error_reason.is_empty()).then_some(error_reason),
            });
        }
        outcome
    }

    async fn llm_call(&mut self) -> anyhow::Result<ChatResponse> {
        let messages = self.session.history().to_vec();
        let request =
            self.build_chat_request(messages.clone(), self.role.requires_multimodal(), "agent");

        let policy = crate::retry::RetryPolicy::current();
        let response = crate::retry::agent_chat(request, &policy)
            .await
            .with_context(|| format!("LLM call {RETRY_EXHAUSTION_MARKER}"))?;

        // Reasoning-only stop: recovered here — before any promotion,
        // persistence or display — via the bounded continuation (leak-safety
        // invariants are documented on `recover_if_reasoning_only_stop`).
        let response = self
            .recover_if_reasoning_only_stop(messages, response, Self::AGENT_REASONING_RECOVERY)
            .await?;

        // A SUCCESSFUL agent-purpose call updates the session length; failures
        // never reach here (the error above propagates), and a reasoning-only
        // turn that exhausted its continuation returned early above without
        // recording anything — so only the response actually returned to the
        // loop updates the value.
        self.record_session_usage(&response).await;
        Ok(response)
    }

    /// Detect a provider input-image content-inspection rejection on the error
    /// trail and durably strip the image from the most recent user message.
    /// Returns `true` when the strip was applied (the caller continues the
    /// normal loop with the corrected message), `false` for a conservative
    /// no-op (the caller keeps the original error path).
    ///
    /// Persist ordering: the store rewrite completes before the in-memory
    /// swap, so a persist failure (or an unpersisted-tail target) leaves the
    /// session untouched and the original LLM error propagates — no
    /// half-stripped continuation, no swallowed errors. Telemetry note: the
    /// rejected attempt already recorded one `non_retryable` `llm_requests`
    /// row (from the retry loop's `fail_exhausted` tail) before this runs;
    /// the corrected call records its own success row — by design, not a
    /// swallowed error.
    async fn strip_rejected_input_image(
        &mut self,
        exhausted: &crate::retry::RetryExhausted,
    ) -> bool {
        let Some(idx) =
            crate::image_strip::detect_input_image_rejection(exhausted, self.session.history())
        else {
            return false;
        };
        let reason = crate::image_strip::extract_provider_reason(exhausted);
        // Invariant guard: detection and the strip share one marker predicate
        // (see `image_strip::has_image_marker`), so a detected rejection always
        // changes the content. If they ever drift — e.g. a malformed `[IMAGE:`
        // fragment passes detection but the regex leaves it verbatim — a
        // no-change rewrite would report `Rewritten` and the loop `continue`
        // below would skip `iteration += 1`, an unbounded retry loop. Treat a
        // no-change strip as a non-strip: keep the original error path.
        let content = {
            let original = &self.session.history()[idx].content;
            let stripped = crate::image_strip::strip_image_markers(original, reason.as_deref());
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

    /// Bounded continuation recovery for the reasoning-only-stop class
    /// (empty content, no parsed tool calls — see [`is_reasoning_only_stop`]).
    ///
    /// Re-requests with the already-generated thinking attached as the
    /// assistant's reasoning for that turn (empty content) plus an appended
    /// continuation prompt. **Strictly appended-only**: the tail is seeded
    /// with the first in-class response's reasoning, and each LATER in-class
    /// response appends its own (assistant reasoning payload + nudge) pair —
    /// a transport failure NEVER grows the tail (the next attempt re-sends
    /// the PREVIOUS request object verbatim, byte-identical even across a
    /// concurrent config hot-reload). The request prefix stays byte-stable
    /// across attempts (prompt-prefix cache preserved). The tail exists only
    /// in the re-request message list — it is never pushed to the session, so
    /// a failed attempt leaves no transcript trace and the raw thinking can
    /// never reach the user.
    ///
    /// Bounded by [`crate::retry::RetryPolicy::continuation`] (3 attempts,
    /// 90 s wall clock), checked against [`crate::shutdown::aborting`], the
    /// agent's cancel token, and the deadline between attempts. Each attempt
    /// is a single [`crate::providers::chat_scoped`] call; retryable provider
    /// errors re-send the same bytes, non-retryable errors break immediately
    /// (a payload-rejecting 400 never burns the budget).
    ///
    /// Telemetry is operation-level — one `llm_requests` row per continuation
    /// operation, matching the single-row-per-operation convention of the
    /// other retry loops: a resolution records one success row with
    /// `retry_attempts` = the resolving attempt index; exhaustion records one
    /// failure row via the shared `fail_exhausted` tail with
    /// `retry_attempts` = the total attempt count and the last attempt's
    /// finish_reason carried on its failure record. Any response with visible
    /// text or parsed tool calls resolves the turn and is persisted normally
    /// by the caller.
    ///
    /// Exhaustion returns a [`crate::retry::RetryExhausted`] with
    /// `last_raw: None` and a final class derived from the last recorded
    /// failure ([`FailureClass::NoResponse`] for a pure thinking-only
    /// exhaustion, [`FailureClass::Shutdown`] when the global abort fired,
    /// [`FailureClass::WallClockExceeded`] when the budget ran out) — the
    /// thinking text is never embedded in the error, so failure
    /// comments/logs cannot leak it, and it is not misclassified as LLM
    /// provider retry exhaustion (no
    /// [`crate::agent::RETRY_EXHAUSTION_MARKER`]).
    async fn recover_reasoning_only_stop(
        &self,
        base: Vec<ChatMessage>,
        first: ChatResponse,
        purpose: &'static str,
    ) -> Result<ChatResponse, crate::retry::RetryExhausted> {
        let policy = crate::retry::RetryPolicy::continuation();
        let deadline = Instant::now() + policy.operation_timeout;
        let operation_started = Instant::now();
        let nudge = crate::prompt::load_prompt("resume_unfinished_turn.md")
            .trim()
            .to_string();
        let mut failures: Vec<crate::retry::RetryFailureRecord> = Vec::new();
        let mut last_request: Option<ChatRequest> = None;

        // Seed the tail with the FIRST in-class response's reasoning (empty
        // content) + the continuation nudge. Each LATER in-class response
        // appends its own pair — the tail grows only when the model actually
        // produced a new response, so a transport failure never duplicates
        // the previous thinking.
        let mut tail: Vec<ChatMessage> = vec![
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
            if Instant::now() >= deadline {
                failures.push(crate::retry::RetryFailureRecord::new_simple(
                    crate::retry::FailureClass::WallClockExceeded,
                    &anyhow::anyhow!("continuation wall-clock budget exceeded"),
                    None,
                ));
                break;
            }

            let request = if tail_grew {
                tail_grew = false;
                let mut messages = base.clone();
                messages.extend(tail.iter().cloned());
                let built =
                    self.build_chat_request(messages, self.role.requires_multimodal(), purpose);
                last_request = Some(built.clone());
                built
            } else {
                last_request
                    .clone()
                    .expect("the first iteration always builds a request")
            };

            match crate::providers::chat_scoped(request.clone(), policy.idle_timeout, deadline)
                .await
            {
                // Still thinking with no answer — record the attempt (with its
                // finish_reason, so the terminal failure row keeps it), append
                // the NEW reasoning as the next tail pair, and continue.
                Ok(resp) if is_reasoning_only_stop(&resp) => {
                    failures.push(crate::retry::RetryFailureRecord::with_metadata(
                        crate::retry::FailureClass::NoResponse,
                        &anyhow::anyhow!(
                            "model returned only reasoning with no answer \
                             (continuation attempt {attempt})"
                        ),
                        resp.finish_reason.clone(),
                        None,
                    ));
                    tail.push(ChatMessage::assistant(
                        assistant_replay_payload(None, &[], resp.reasoning.as_ref()).to_string(),
                    ));
                    tail.push(ChatMessage::user(nudge.clone()));
                    tail_grew = true;
                }
                // Real answer or tool calls — the turn resolves normally.
                Ok(resp) => {
                    crate::stats::record_llm_success(&request, operation_started, attempt, &resp)
                        .await;
                    return Ok(resp);
                }
                // Provider failure — break on non-retryable (a payload-rejecting
                // 400 must not burn the budget); otherwise the next iteration
                // re-sends the byte-identical request (tail untouched).
                Err(err) => {
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
            Some(request) => {
                crate::retry::fail_exhausted(&request, operation_started, exhausted).await
            }
            None => Err(exhausted),
        }
    }

    /// Recovery policy for the agent loop ([`Self::llm_call`]): continuation
    /// exhaustion fails the turn.
    const AGENT_REASONING_RECOVERY: ReasoningOnlyStopRecovery = ReasoningOnlyStopRecovery {
        purpose: "agent-continuation",
        exhausted_ctx: "model returned only reasoning without an answer after continuation attempts",
    };

    /// Recovery policy for summarization ([`Self::summarize`]): continuation
    /// exhaustion fails open with the full history.
    const SUMMARIZE_REASONING_RECOVERY: ReasoningOnlyStopRecovery = ReasoningOnlyStopRecovery {
        purpose: "summarize-continuation",
        exhausted_ctx: "summarization continuation exhausted — failing open with full history",
    };

    /// Recover a reasoning-only stop via the bounded continuation, or pass
    /// the response through unchanged.
    ///
    /// This is the single classification point for the reasoning-only stop
    /// (empty content, no parsed tool calls), reached by both LLM paths before
    /// any promotion, persistence or display. The provider-layer reasoning→text
    /// promotion is gone, so the response is the model's honest output.
    ///
    /// On continuation exhaustion the returned error carries
    /// `recovery.exhausted_ctx` — the caller's failure semantics: the agent
    /// loop fails the turn safely (nothing persisted, nothing displayed; the
    /// existing failure indicators — direct chat emoji / pipeline comment —
    /// fire), while the summarize path fails open, warning and continuing with
    /// the full history.
    ///
    /// Leak-safety invariants (shared by both paths): the continuation is
    /// appended-only and never touches the session transcript, so the thinking
    /// can never leak to the user — and the exhaustion error never embeds the
    /// thinking text either (see [`Self::recover_reasoning_only_stop`]).
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

    /// Record the real provider-reported session length after a SUCCESSFUL
    /// agent-purpose LLM call: `input_tokens + output_tokens` of the full
    /// request envelope plus the response that just arrived.
    ///
    /// Only agent-purpose calls pass through here — extraction, summarize,
    /// consolidate, synthesis, and research sub-calls use other builders and
    /// are excluded (they are not session context).
    ///
    /// Post-compaction note: [`Session::apply_summary`] deliberately does NOT
    /// reset the value — the fixed scope says no logic changes beyond
    /// replacing estimated tokens with real tokens. After a compaction the
    /// stale pre-compaction length may re-fire `maybe_summarize` on a
    /// following turn if no successful agent call lands in between; that
    /// edge is spec-compliant (accepted, not fixed).
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
        crate::registry::AGENT_REGISTRY.set_session_tokens(
            &self.agent_id,
            self.generation,
            token_length,
        );
    }

    /// Persist tool results to the session store and push them into the in-memory
    /// history (via the session) for the next LLM iteration.
    ///
    /// All messages (assistant call + tool results) are batch-persisted in a single
    /// DB transaction, eliminating orphaned assistant calls on crash mid-loop.
    async fn commit_tool_results(
        &mut self,
        tool_calls: &[ToolCall],
        outcomes: &[ToolExecutionOutcome],
        history_content: &str,
    ) -> anyhow::Result<()> {
        let tools = &self.tools;

        // Build DB messages for batch persistence.
        let assistant_call = ChatMessage::assistant(history_content.to_string());
        let mut db_messages = Vec::with_capacity(1 + outcomes.len());
        db_messages.push(assistant_call);

        for (call, outcome) in tool_calls.iter().zip(outcomes.iter()) {
            let tool = find_tool(tools, &call.name);
            let output = match tool {
                Some(t) => t.format_output(&outcome.output),
                None => crate::util::truncate_tool_output(&outcome.output),
            };
            db_messages.push(ChatMessage::tool_result(&call.id, &output));
        }

        // Batch-persist all messages in a single transaction (preceded by
        // any unpersisted history tail) and push them into in-memory history.
        self.session
            .persist_messages(&self.agent_id, &db_messages)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to persist tool results: {e}"))?;

        Ok(())
    }

    /// Drain any incoming messages from the router (e.g., ticket comments)
    /// and inject them into the session history.
    ///
    /// Called at the start of each LLM loop iteration, before the LLM call.
    /// Messages are persisted to the session DB AND pushed to in-memory
    /// history so they survive crashes and are visible to the model.
    ///
    /// On persist failure the messages are still delivered to the in-memory
    /// history (the model sees them this turn) and left in the unpersisted
    /// tail; the next successful persist — tool round, later drain, or
    /// finalize (which flushes the tail even on aborted turns) — writes
    /// them, so nothing is lost once the turn is finalized. A cancel before
    /// finalize drops the in-memory tail at the next turn — comments remain
    /// in the board DB.
    ///
    /// This is a no-op when no receiver is configured (e.g., chat agents
    /// that use the consumer_loop instead).
    async fn drain_incoming_messages(&mut self) -> anyhow::Result<()> {
        let Some(rx) = &mut self.incoming_rx else {
            return Ok(());
        };

        let mut messages = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(job) => {
                    let content = match job.kind {
                        crate::message_router::JobKind::TicketComment => {
                            format!(
                                "[Comment from {} on ticket]: {}",
                                job.user_name, job.content
                            )
                        }
                        _ => job.content,
                    };
                    messages.push(crate::session::user_msg_with_ts(&content, None));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    // Channel closed — no more messages will arrive.
                    // Drop the receiver so we don't try again.
                    self.incoming_rx = None;
                    break;
                }
            }
        }

        if messages.is_empty() {
            return Ok(());
        }

        // Persist to session DB — best-effort. The comment is already saved in
        // the board DB, so losing the session copy is recoverable. Log and
        // continue rather than aborting the LLM iteration.
        if let Err(e) = self
            .session
            .persist_messages(&self.agent_id, &messages)
            .await
        {
            tracing::warn!(
                agent_id = %self.agent_id,
                error = %e,
                "Failed to persist incoming messages to session DB — continuing without persistence",
            );
            // Still deliver to in-memory history so the model sees them this
            // iteration; the unpersisted tail is flushed next persist.
            self.session.push_messages_unpersisted(&messages);
        }

        Ok(())
    }

    /// Build a [`ChatRequest`] from the given messages and image-parts flag,
    /// using the agent's current model, tools, reasoning-effort, and
    /// provider-routing settings.
    ///
    /// All parameter sources are lazily resolved each call so that runtime
    /// hot-reload (model, routing) is reflected immediately. Reasoning effort
    /// is baked into the role metadata since mahbot-1819 — it is not
    /// hot-reloadable.
    ///
    /// # KV-cache preservation
    ///
    /// Every call site that produces a chat request for the same logical
    /// conversation *must* use exactly the same parameter sources — model,
    /// reasoning_effort, tools (critically tools!), and provider routing — so
    /// that the provider can reuse the cached prefix computed
    /// during the original agent call.  Any deviation (including dropping
    /// tools) forces the provider to recompute the entire KV-cache prefix.
    ///
    /// [`Self::extract_verdict`] calls this method internally to derive its
    /// parameter set.  [`Self::summarize`] calls this method directly.
    fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
        allow_image_parts: bool,
        purpose: &'static str,
    ) -> ChatRequest {
        ChatRequest {
            meta: Some(crate::ChatRequestMeta {
                purpose,
                agent_id: self.agent_id.clone(),
                role: self.role.as_str().to_string(),
                workspace: self.workspace.name.clone(),
                ticket_id: self.ticket.as_ref().map(|t| t.id.clone()),
            }),
            ..chat_request(
                self.role,
                Some(self.tool_specs.clone()),
                messages,
                allow_image_parts,
            )
        }
    }

    // ── Extraction / summarisation ──

    /// Extract a structured verdict with the hardened scoped retry loop.
    ///
    /// KV-cache requirements: the params come from [`Self::build_chat_request`]
    /// (model, reasoning_effort, tools, provider routing) so all
    /// attempts are byte-identical except the parse-failure re-prompt.
    ///
    /// `validate` runs fail-closed on the parsed value (e.g. score ∈ [0,10]):
    /// a rejected value is treated as a parse failure and re-prompted.
    /// Pass `validate = None` for plain structured extraction (diagnostics
    /// discovery). `policy_override` supplies a shorter retry schedule for
    /// fail-open callers; `None` uses the default verdict budget.
    pub(crate) async fn extract_verdict<T: serde::de::DeserializeOwned>(
        &self,
        extraction_prompt: &str,
        validate: Option<&crate::ExtractionValidator<T>>,
        policy_override: Option<&crate::retry::RetryPolicy>,
    ) -> Result<T, crate::retry::RetryExhausted> {
        // Live-view indicator: the agent's card is the single tracker for
        // extractions (they never register a separate non-agent call row) —
        // without it the card would look idle while the extraction LLM call
        // runs. Purely observational; the guard clears on every exit path.
        let _activity = crate::registry::AGENT_REGISTRY.activity_started(
            &self.agent_id,
            self.generation,
            "extracting",
        );
        let params = self.build_chat_request(vec![], false, "extraction");
        crate::extraction::retry_extract_structured_scoped(
            self.session.history(),
            extraction_prompt,
            &params,
            validate,
            policy_override,
        )
        .await
    }

    /// Summarise the agent's session history.
    ///
    /// KV-cache requirements: see [`Self::build_chat_request`].
    pub(crate) async fn summarize(&self) -> anyhow::Result<String> {
        // Live-view indicator: same single-tracker contract as extraction —
        // the agent's card shows the summarization phase instead of looking
        // idle during the (potentially large) compaction call.
        let _activity = crate::registry::AGENT_REGISTRY.activity_started(
            &self.agent_id,
            self.generation,
            "summarizing",
        );
        let mut history = self.session.history().to_vec();
        history.push(crate::ChatMessage::user(self.role.summary_prompt()));

        let policy = crate::retry::RetryPolicy::current();
        let chat_resp = crate::retry::agent_chat(
            self.build_chat_request(
                history.clone(),
                self.role.requires_multimodal(),
                "summarize",
            ),
            &policy,
        )
        .await
        .with_context(|| format!("summarization LLM call {RETRY_EXHAUSTION_MARKER}"))?;

        // Reasoning-only stop: the same bounded continuation as the agent loop
        // (leak-safety invariants on `recover_if_reasoning_only_stop`). On
        // exhaustion, fail open below — the empty-response error path warns
        // and continues with the full history.
        let chat_resp = self
            .recover_if_reasoning_only_stop(history, chat_resp, Self::SUMMARIZE_REASONING_RECOVERY)
            .await?;

        if let Some(ref u) = chat_resp.usage {
            tracing::debug!(
                input_tokens = u.input_tokens,
                cached_input_tokens = u.cached_input_tokens,
                output_tokens = u.output_tokens,
                "Summarization token usage",
            );
        }

        let summary_text = chat_resp
            .text
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("summarization produced empty response"))?;

        Ok(crate::util::truncate(&summary_text, 32_000))
    }

    /// Check the session length against [`SUMMARIZATION_THRESHOLD`] and
    /// summarise if necessary.
    ///
    /// The session length is the REAL provider-reported value (input +
    /// output tokens of the last successful agent LLM call), loaded at
    /// session init. `None` — no successful usage-bearing call ever recorded
    /// a value (new sessions, pre-migration sessions; approved no-backfill)
    /// — is treated as below the threshold: no summarization. A FAILED call
    /// never updates the value (the session did not change).
    ///
    /// KV-cache preservation: [`Agent::summarize`] keeps all parameters identical
    /// (see [`Agent::build_chat_request`]) so the cached prefix is reusable.
    async fn maybe_summarize(&mut self) {
        let Some(token_length) = self.session.token_length() else {
            return;
        };
        tracing::debug!(
            agent_id = %self.agent_id,
            role = %self.role,
            token_length,
            "Session token length",
        );
        if token_length > crate::session::SUMMARIZATION_THRESHOLD {
            tracing::info!(
                agent_id = %self.agent_id,
                role = %self.role,
                token_length,
                "Session exceeded summarization threshold",
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
                        )
                        .await;
                }
                Err(e) => {
                    tracing::warn!(
                        error_chain = %crate::util::truncate_sandwich(
                            &crate::util::scrub_credentials(&format!("{e:#}")),
                            crate::util::FAILURE_DETAIL_CAP,
                            "summarization failure",
                        ),
                        "Summarization failed — continuing with full history"
                    );
                }
            }
        }
    }
}

/// Parameters for a reasoning-only-stop recovery: the telemetry purpose tag
/// forwarded to [`Agent::recover_reasoning_only_stop`] and the context
/// attached to the exhaustion error. The two strings travel as a single
/// named-field struct (rather than two adjacent `&'static str` arguments) so
/// they cannot be silently swapped at a call site — see the per-path
/// constants on [`Agent`].
struct ReasoningOnlyStopRecovery {
    /// Purpose tag for the continuation LLM request (telemetry).
    purpose: &'static str,
    /// Context attached to the `anyhow` error when continuation attempts are
    /// exhausted.
    exhausted_ctx: &'static str,
}

/// Result of preparing an assistant turn from the LLM response.
struct PreparedAssistantTurn {
    display_text: String,
    tool_calls: Vec<ToolCall>,
    history_content: String,
}

/// Classify the "reasoning-only stop" class: raw empty content (think tags
/// stripped — they are reasoning, not content) with no parsed tool calls,
/// REGARDLESS of finish reason. The provider-layer reasoning→text promotion
/// is gone, so an empty `text` is the model's honest empty content.
///
/// Tool-call turns (empty text by design) are excluded by the tool-call
/// check. Degenerate members — fully-empty responses (no reasoning attached)
/// and think-tag-only content — are consciously in-class: the continuation
/// then carries just the nudge, and a previously-`Ok("")` silent answer now
/// resolves via the continuation or fails the turn safely (never CoT-as-text).
#[must_use]
fn is_reasoning_only_stop(response: &ChatResponse) -> bool {
    response.tool_calls.is_empty() && response.text.as_deref().is_none_or(|t| t.trim().is_empty())
}

/// Prepare assistant response data from the LLM response.
fn prepare_assistant_turn(response: ChatResponse) -> PreparedAssistantTurn {
    let response_text = response.text_or_empty().to_string();
    let tool_calls = response.tool_calls;
    let reasoning = response.reasoning.as_ref();

    // The structured payload faithfully captures the model's original response
    // (empty content for a reasoning-only return). Reasoning-only stops never
    // reach this point — the agent loop classifies and recovers them in
    // llm_call before any persistence/display — so this builder only ever
    // serializes real answers and tool-call turns.
    let json_payload =
        assistant_replay_payload(Some(&response_text), &tool_calls, reasoning).to_string();

    // Dispatch on whether tool calls and/or reasoning are present.
    // Three arms: plain answer (no tools, no reasoning), reasoning+text
    // (reasoning present alongside visible text), and tool calls (tools
    // present). Reasoning-only responses (reasoning present, empty text) are
    // the recovered class and never arrive here.
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
    }
}

/// Round-scoped agent options for parallel-round members.
///
/// `round_ts` pins the first user message's timestamp (one value per round —
/// byte-identical task messages across members); `first_call_notify` is the
/// leader-stagger signal — fired after the leader's first LLM call completes
/// (success or failure, after the full retry budget: a leader whose call
/// exceeds the stagger bound releases followers via the sleep arm and misses
/// the cached prefix) or when the leader bails at the phase gate before any
/// LLM call.
#[derive(Clone)]
pub(crate) struct RoundOpts {
    pub(crate) round_ts: String,
    pub(crate) first_call_notify: Option<std::sync::Arc<tokio::sync::Notify>>,
}

/// Default fail-open bound on the leader-stagger wait: how long followers wait
/// for the leader's first LLM call before starting anyway. First calls take
/// ~4-6 s; the bound is a safety cap, not the expected wait. Overridable via
/// env for tuning (mirrors [`crate::tools::analyze::round_timeout`]).
const DEFAULT_STAGGER_WAIT_SECS: u64 = 8;

fn leader_stagger_wait() -> std::time::Duration {
    crate::util::env_duration_secs("MAHBOT_STAGGER_WAIT_SECS", DEFAULT_STAGGER_WAIT_SECS)
}

/// Spawn a round's member tasks with a leader-stagger: member 0 (the leader)
/// starts immediately; the rest start after the leader's first LLM call
/// completes (the notify fires) or the bounded wait elapses — whichever
/// first. The wait is fail-open (a failed/never-starting leader still releases
/// the followers) and interrupted by daemon shutdown or the drain flag.
///
/// Single-member rounds are a no-op stagger, and boot-resumed rounds
/// (`resume = true`) skip the stagger entirely — their sessions already
/// diverged, so the leader's prefix would not be shared anyway and the wait
/// would only add latency.
///
/// Renders the round timestamp once and hands every member factory a
/// ready-made [`RoundOpts`] (only the leader carries the first-call signal).
///
/// The first-call signal has a single waiter — the `select!` below — so
/// `notify_one()` releases all followers at once; its stored permit also
/// covers a fire-before-wait race that `notify_waiters()` would lose.
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
        // Resume (sessions already diverged — no shared prefix to hit) or a
        // single-member round: no stagger, spawn everything immediately.
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
    // Drain/shutdown arms still spawn the followers — their Session::init
    // append runs before the aborting() bail. Deliberate: it pre-seeds the
    // session so boot resume's has_session guard skips the re-append
    // (consistent with the leader's init-then-abort drain path).
    handles.extend(
        followers
            .into_iter()
            .map(|m| tokio::spawn(m(follower_opts.clone()))),
    );
    handles
}

/// Core agent lifecycle: create agent (auto-registers with its own
/// CancellationToken), run work, handle cancellation and errors.
/// Returns the agent (even on failure) and the response on success.
///
/// `user_name` and `channel` identify the origin of the message being
/// processed — used by tools (e.g. AnalyzeTool async dispatch) to route
/// sub-agent results to the correct user.
///
/// **Cancellation safety**: Even if `agent.work()` completes before the token
/// fires (the classic race), we check `is_cancelled()` after work completes
/// and discard the result — preventing overwrites of externally-set `cancelled`
/// status in downstream code.
///
/// Returns `(agent, Some(response))` on success.
/// Returns `(agent, None)` on cancellation (discard result) or error (already logged).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn run_agent(
    agent_id: String,
    role: crate::Role,
    ws: &crate::Workspace,
    ticket: Option<&crate::board::Ticket>,
    message: &str,
    user_name: String,
    channel: String,
    incoming_rx: Option<tokio::sync::mpsc::UnboundedReceiver<crate::message_router::AgentJob>>,
    resume: bool,
    round: Option<RoundOpts>,
    parent_key: Option<crate::registry::ParentKey>,
    parent_label: Option<String>,
) -> (Agent, Option<String>) {
    // Unregister from the message router on EVERY exit path, including a
    // panic mid-work (Drop runs during unwind) — a leaked router entry would
    // keep routing comments to a dead agent. Only for caller-registered paths
    // (Some(incoming_rx)): the persistent consumer_loop path (None) owns its
    // router entry via route()'s catch_unwind cleanup and must keep it across
    // jobs. Idempotent.
    struct UnregisterOnDrop(String);
    impl Drop for UnregisterOnDrop {
        fn drop(&mut self) {
            crate::message_router::unregister_agent(&self.0);
        }
    }
    let _router_guard = incoming_rx
        .is_some()
        .then(|| UnregisterOnDrop(agent_id.clone()));
    let agent_id_for_cleanup = agent_id.clone();
    let mut agent = Agent::new(
        agent_id,
        role,
        ws,
        ticket.cloned(),
        user_name,
        channel,
        parent_key,
        parent_label,
    );
    agent.incoming_rx = incoming_rx;
    if let Some(round) = round {
        agent.round_ts = Some(round.round_ts);
        agent.first_call_notify = round.first_call_notify;
    }
    let result = agent.work(message, resume).await;

    // Cancellation safety: if the token fired after work() completed but
    // before we checked, discard the result to prevent overwriting of
    // externally-set cancelled status in downstream code.
    let outcome = if agent.is_cancelled() {
        tracing::debug!(
            agent_id = %agent.agent_id,
            workspace = %ws.name,
            role = %role,
            ticket = ticket.map(|t| t.id.as_str()),
            classification = failure_classification(&agent, None),
            "Agent cancelled"
        );
        (agent, None)
    } else {
        match result {
            Ok(response) => (agent, Some(response)),
            Err(e) => {
                // Capture the real cause (full chain) so ticket dispatchers can
                // persist it in failure comments instead of a generic placeholder.
                agent.failure = Some(format!("{e:#}"));
                agent.failure_class = failure_class_from_error(&e);
                let classification = failure_classification(&agent, Some(&e));
                let error_chain = crate::util::truncate_sandwich(
                    &crate::util::scrub_credentials(&format!("{e:#}")),
                    crate::util::FAILURE_DETAIL_CAP,
                    "agent failure log",
                );
                // During global (SIGTERM/SIGINT) shutdown, in-flight agents return
                // errors from work() — expected, not real failures. The global
                // token fires before `shutdown_all()` cancels per-agent tokens, so
                // either path resolves to classification "shutdown". Log at debug
                // level to avoid misleading ERROR noise on clean shutdown. The
                // graceful drain maps to classification "drain" — same treatment.
                if classification == "shutdown" || classification == "drain" {
                    tracing::debug!(
                        agent_id = %agent.agent_id,
                        workspace = %ws.name,
                        role = %role,
                        ticket = ticket.map(|t| t.id.as_str()),
                        classification,
                        error_chain,
                        "Agent failed during shutdown"
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
    };

    // Owner-deletes-at-end: remove the spill files this agent run created
    // (safe on macOS — the run is over, nothing will read them again).
    crate::tools::shell::cleanup_agent_spills(&agent_id_for_cleanup);
    outcome
}

/// Default dispatch: no ticket, empty user/channel, no inbox, no resume.
///
/// `parent_key` carries the DIRECT PARENT INVOCATION grouping key for the
/// Running Agents view (ticket / analyze round / research run); `None` for
/// workspace singletons (manager / maintainer / discovery / direct chat).
/// `parent_label` is the human-readable label of that parent invocation
/// (analyze question / research question) — purely presentational.
pub(crate) async fn run_default_agent(
    agent_id: &str,
    role: crate::Role,
    ws: &crate::Workspace,
    message: &str,
    round: Option<RoundOpts>,
    parent_key: Option<crate::registry::ParentKey>,
    parent_label: Option<String>,
) -> (Agent, Option<String>) {
    run_agent(
        agent_id.to_string(),
        role,
        ws,
        None,
        message,
        String::new(),
        String::new(),
        None,
        false,
        round,
        parent_key,
        parent_label,
    )
    .await
}

/// Extract the typed provider failure class from an error chain, when the
/// chain carries a [`RetryExhausted`] (the terminal retry-loop error that
/// preserves the granular [`crate::retry::FailureClass`]).
///
/// `None` means the error is not provider-classified (runtime errors, panics,
/// plain I/O failures) — callers treat that as a genuine non-provider failure.
pub(crate) fn failure_class_from_error(
    error: &anyhow::Error,
) -> Option<crate::retry::FailureClass> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<crate::retry::RetryExhausted>())
        .map(|exhausted| exhausted.final_class)
}

/// Stable failure classification for the agent-failure log.
///
/// Order matters: the global shutdown token fires on SIGTERM/SIGINT and
/// dashboard close, the per-agent token on /stop — check the global token
/// first so shutdown isn't mislabeled as a user cancel. The global token also
/// cancels every per-agent token via
/// [`crate::registry::AgentRegistry::shutdown_all`], so the cancelled branch
/// is only a user cancel when the global token has not fired. `error` is
/// `None` on the cancelled early-return path (no error exists to classify);
/// otherwise [`RetryExhausted`] (kept as an error source by
/// [`llm_call`]/[`summarize`]) carries the granular
/// [`crate::retry::FailureClass`]; everything else (I/O, tool errors, panics)
/// falls back to `runtime`.
fn failure_classification(agent: &Agent, error: Option<&anyhow::Error>) -> &'static str {
    if crate::shutdown::is_draining() {
        // The graceful drain cut the round short — not a failure. Checked
        // before the token so drained agents are never mislabeled as
        // cancelled/shutdown (no AGENT_FAILURE_EMOJI, no exit rollback).
        "drain"
    } else if crate::shutdown::shutdown_token().is_cancelled() {
        "shutdown"
    } else if agent.is_cancelled() {
        "cancelled"
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
            ticket: None,
            generation: 0,
            tool_stats: std::sync::Mutex::new(Vec::new()),
            user_name: String::new(),
            channel: String::new(),
            parent_key: None,
            parent_label: None,
            incoming_rx: None,
            round_ts: None,
            first_call_notify: None,
            failure: None,
            failure_class: None,
            background_sessions: std::sync::Arc::new(
                crate::tools::shell::BackgroundSessions::default(),
            ),
        }
    }

    #[tokio::test]
    async fn tool_with_scrub_disabled_preserves_output() {
        assert_scrubbed(false).await;
    }

    #[test]
    fn failure_classification_recovers_retry_exhaustion() {
        // failure_classification consults the process-global drain flag before
        // everything else — serialize against the drain-flag writers
        // (project convention: retry_tests_lock).
        let _guard = crate::util::test::retry_tests_lock();
        // Mirror llm_call's error construction: the RetryExhausted must
        // survive as a source (via .context, not string flattening) so the
        // granular FailureClass is recoverable from the chain.
        let exhausted = crate::retry::RetryExhausted::with_last_raw(
            vec![],
            crate::retry::FailureClass::NonRetryable,
            None,
        );
        let err = anyhow::Error::new(exhausted).context("LLM call exhausted retry budget");
        // The {:#} chain must keep the marker engineer_failure_comment matches.
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

        // User cancellation (per-agent token) is distinct from shutdown.
        let cancelled = make_agent(vec![]);
        cancelled.cancel_token.cancel();
        assert_eq!(
            failure_classification(&cancelled, Some(&runtime_err)),
            "cancelled"
        );

        // The cancelled early-return path classifies with no error present.
        assert_eq!(failure_classification(&cancelled, None), "cancelled");
    }

    /// The graceful-drain cut is classified as "drain" (NOT cancelled/
    /// shutdown/failure) — the consumer suppresses the failure emoji and the
    /// dispatch tails skip the exit-time ticket rollback for drained agents.
    /// Serialized against other tests: the drain flag is process-global, so a
    /// concurrent test consulting is_draining() could be misclassified during
    /// the assertion window (project convention: retry_tests_lock).
    #[tokio::test]
    async fn failure_classification_recognizes_drain() {
        let _guard = crate::util::test::retry_tests_lock();
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
        // A per-agent cancel during drain still classifies as drain — the
        // drain is checked first so no failure emoji fires on exit.
        agent.cancel_token.cancel();
        assert_eq!(failure_classification(&agent, None), "drain");
        // Restore isolation — the drain flag is process-global; the token
        // must never be fired from a test (it would cancel every parallel
        // retry loop).
        crate::shutdown::drain_clear();
    }

    /// Shared helper: create an agent with a TestTool, execute_tool it, and assert scrubbing behavior.
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

    #[tokio::test]
    async fn tool_with_scrub_enabled_scrubs_sensitive_output() {
        assert_scrubbed(true).await;
    }

    /// A media-generation test tool that returns a configurable [`Tool::media_marker`].
    ///
    /// Used to test [`extract_media_from_outcomes`] without real image/video generation.
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

    // ── extract_media_from_outcomes tests ───────────────────────────────────

    #[test]
    #[expect(clippy::too_many_lines)]
    fn extract_media_outcomes_consolidated() {
        enum ToolDef {
            /// [`Media`] carries a marker prefix; [`NonMedia`] uses a unit variant because
            /// `extract_media_from_outcomes` only consults [`Tool::media_marker`], never
            /// [`Tool::execute`] — the output value on [`TestTool`] has no behavioral effect.
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
            // Marker prefix with nothing parseable after it (e.g. at end of output)
            // produces an empty path which the filter rejects.
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
                        // Output value is irrelevant — extract_media_from_outcomes
                        // only consults media_marker(), never execute().
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
        /// Tool that reads CURRENT_TOOL_USER_NAME and CURRENT_TOOL_CHANNEL
        /// from task-locals and returns them in its output.
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
                // Read task-locals synchronously (no async boundary) — the
                // contract is: the Agent work loop sets these before each
                // tool execution group, and tools must read them before any
                // tokio::spawn if crossing an async boundary.
                let user_name = CURRENT_TOOL_USER_NAME
                    .try_with(String::clone)
                    .unwrap_or_default();
                let channel = CURRENT_TOOL_CHANNEL
                    .try_with(String::clone)
                    .unwrap_or_default();
                Ok(format!("user={user_name},channel={channel}"))
            }
        }

        let tool: Box<dyn Tool> = Box::new(ReadTaskLocalsTool);
        let name = tool.name();
        let mut agent = make_agent(vec![tool]);
        agent.user_name = "alice".into();
        agent.channel = "gui".into();

        // Call through execute_tool_group so task-locals are set once for
        // the entire batch — this is the path used by the agent work loop.
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
        assert!(outcomes[0].success, "ReadTaskLocalsTool should succeed");
        assert!(
            outcomes[0].output.contains("user=alice"),
            "should propagate user_name: {}",
            outcomes[0].output
        );
        assert!(
            outcomes[0].output.contains("channel=gui"),
            "should propagate channel: {}",
            outcomes[0].output
        );
    }

    // ── drain_incoming_messages tests ─────────────────────────────────

    /// drain_incoming_messages injects TicketComment messages into the session
    /// as user messages with the correct prefix.
    #[tokio::test]
    async fn test_drain_incoming_messages_injects_ticket_comment() {
        crate::util::test::init_management_test_stores().await;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut agent = make_agent(vec![]);
        agent.incoming_rx = Some(rx);
        agent.agent_id = "_test_drain_ticket_comment".into();

        // Send a TicketComment job
        let job = crate::message_router::AgentJob {
            content: "Please fix the formatting".to_string(),
            workspace_name: "test_ws".to_string(),
            user_name: "manager".to_string(),
            channel: String::new(),
            kind: crate::message_router::JobKind::TicketComment,
            role: crate::Role::Manager,
            reply_target: None,
            pending_job_id: None,
        };
        let _ = tx.send(job);

        agent
            .drain_incoming_messages()
            .await
            .expect("drain should succeed");

        let history = agent.session.history();
        assert!(!history.is_empty(), "should have at least one message");
        let last = history.last().unwrap();
        assert_eq!(last.role, crate::ChatRole::User, "should be a user message");
        assert!(
            last.content.contains("Please fix the formatting"),
            "should contain the comment content: {}",
            last.content,
        );
        assert!(
            last.content.contains("[Comment from manager on ticket]"),
            "should include comment prefix: {}",
            last.content,
        );
    }

    /// drain_incoming_messages handles unregistered job kinds by passing
    /// content through without prefix formatting.
    #[tokio::test]
    async fn test_drain_incoming_messages_non_comment() {
        crate::util::test::init_management_test_stores().await;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut agent = make_agent(vec![]);
        agent.incoming_rx = Some(rx);
        agent.agent_id = "_test_drain_non_comment".into();

        // Send a plain UserMessage job
        let job = crate::message_router::AgentJob {
            content: "Hello agent".to_string(),
            workspace_name: "test_ws".to_string(),
            user_name: "user".to_string(),
            channel: String::new(),
            kind: crate::message_router::JobKind::UserMessage,
            role: crate::Role::Assistant,
            reply_target: None,
            pending_job_id: None,
        };
        let _ = tx.send(job);

        agent
            .drain_incoming_messages()
            .await
            .expect("drain should succeed");

        let history = agent.session.history();
        assert!(!history.is_empty(), "should have at least one message");
        let last = history.last().unwrap();
        assert!(
            last.content.contains("Hello agent"),
            "should contain the raw content: {}",
            last.content,
        );
    }

    /// drain_incoming_messages handles a closed receiver gracefully
    /// (sets incoming_rx to None, returns Ok without error).
    #[tokio::test]
    async fn test_drain_incoming_messages_disconnected() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<crate::message_router::AgentJob>();
        let mut agent = make_agent(vec![]);
        agent.incoming_rx = Some(rx);
        agent.agent_id = "_test_drain_disconnected".into();

        // Drop the sender so the channel is disconnected
        drop(tx);

        // Should not panic or error
        agent
            .drain_incoming_messages()
            .await
            .expect("drain should succeed on disconnected channel");
        assert!(
            agent.incoming_rx.is_none(),
            "incoming_rx should be set to None after disconnect",
        );
    }

    // ── Leader-stagger mechanics ───────────────────────────────────

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
    /// still releases the followers after the bounded wait — the round never
    /// hangs.
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
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn leader_first_call_failure_still_fires_signal() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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
            "task",
            String::new(),
            String::new(),
            None,
            false,
            Some(RoundOpts {
                round_ts: crate::session::render_timestamp(),
                first_call_notify: Some(notify.clone()),
            }),
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

    // ── Reasoning-only stop recovery ─────────────────────────────────

    /// The class predicate: raw empty content + no parsed tool calls,
    /// regardless of finish reason. Tool-call turns (empty text by design)
    /// and real answers are excluded. Degenerate members (fully-empty, no
    /// reasoning) are consciously in-class.
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

    /// A reasoning-only stop resolves via the continuation: the final answer
    /// is returned, the continuation request carries the appended tail
    /// (assistant reasoning payload + the resume nudge), and the nudge never
    /// appears in the original request.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_recovers_reasoning_only_stop_via_continuation() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// Consecutive reasoning-only responses accumulate: each failed attempt
    /// appends its own (assistant reasoning + nudge) pair, so the request
    /// prefix stays byte-stable and only the tail grows.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_continuation_accumulates_tail_until_answer() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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
        // Strictly appended-only: one nudge after the first continuation, two
        // after the second (the first pair is part of the byte-stable prefix).
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

        // Direct byte-prefix pin of the KV-cache property: attempt 3's message
        // list begins with attempt 2's messages verbatim (append-only growth,
        // nothing rewritten). The capture joins per-message Debug with NUL, so
        // a plain prefix check holds.
        let messages = fake.request_messages.lock().unwrap().clone();
        assert_eq!(messages.len(), 3);
        assert!(
            messages[2].starts_with(&messages[1]),
            "byte-stable prefix: attempt 3's messages begin with attempt 2's verbatim"
        );
    }

    /// Continuation exhaustion fails the turn safely: a NoResponse
    /// [`RetryExhausted`] with `last_raw: None`, no reasoning text in the
    /// error, no provider-retry-exhaustion marker, no transcript trace, and
    /// granular "no_response" classification.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_continuation_exhaustion_fails_safely_without_leaking() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// A transport error mid-recovery never duplicates the thinking tail: the
    /// retried request is byte-identical, and exhaustion derives the final
    /// class from the last recorded failure (Transport, not NoResponse).
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_continuation_transport_error_does_not_duplicate_tail() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// A non-retryable provider error mid-recovery breaks immediately instead
    /// of burning the remaining budget, and keeps the granular class.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_continuation_non_retryable_error_breaks_immediately() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// A global abort (drain) between recovery attempts classifies the
    /// terminal error as Shutdown, not NoResponse — the real cause must not be
    /// masked, even when the break happens before the first attempt (no
    /// request was ever sent, so there is also nothing to record in telemetry).
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn recover_reasoning_only_stop_abort_classifies_as_shutdown() {
        let _guard = crate::util::test::retry_tests_lock();
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

        // The drain flag is process-global — set it for the duration of the
        // recovery, then restore (never fire the global token from a test).
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

    /// A reasoning-only turn whose continuation exhausts is a FAILED call: it
    /// must never update the session token length (the `maybe_summarize`
    /// "a FAILED call never updates the value" invariant), even though the
    /// invisible in-class response carried real usage.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_continuation_exhaustion_does_not_update_session_length() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// On continuation success, the session length is updated from the
    /// RESOLVING response only — the in-class response the continuation
    /// consumed is never recorded (no inflated value, no wasted double write).
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_continuation_success_records_only_resolving_usage() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn llm_call_skips_continuation_for_normal_and_tool_call_turns() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// Summarize recovers a reasoning-only stop via the same bounded
    /// continuation; the continuation answer becomes the summary.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn summarize_recovers_reasoning_only_stop_via_continuation() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    /// Summarize continuation exhaustion keeps the fail-open behavior: the
    /// empty-response error fires (warn + full history in `maybe_summarize`),
    /// and the thinking never leaks into the error.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn summarize_continuation_exhaustion_fails_open_without_leaking() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
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

    // ── Real provider-token session length semantics ───────────────────

    /// Both-or-nothing / keep-last / never-zero rule of
    /// [`Agent::record_session_usage`].
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

    /// Saturating sum: an overflowing usage pair cannot wrap into a small
    /// bogus session length.
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

    /// An unknown session length (`None` — no successful usage-bearing call
    /// ever recorded) is below the summarization threshold: the check exits
    /// before the summarize LLM call. No provider is installed on purpose —
    /// reaching [`Agent::summarize`] would panic on the unset provider.
    #[tokio::test]
    async fn maybe_summarize_none_and_below_threshold_are_noops() {
        crate::util::test::init_test_stores().await;
        let mut agent = make_agent(vec![]);

        // None (new / pre-migration session): early exit, nothing fires.
        agent.maybe_summarize().await;
        assert_eq!(agent.session.token_length(), None);

        // A real value below the fixed threshold is equally a no-op.
        agent
            .session
            .set_token_length(Some(crate::session::SUMMARIZATION_THRESHOLD / 2));
        agent.maybe_summarize().await;
    }

    // ── Provider input-image rejection strip (ticket mahbot-1788) ────────

    /// The observed provider rejection body (OpenRouter HTTP 400
    /// `data_inspection_failed`, image variant).
    const IMAGE_REJECTION_BODY: &str = r#"{"error":{"message":"Input image data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#;

    /// Same code, text variant — must NOT trigger the strip.
    const TEXT_REJECTION_BODY: &str = r#"{"error":{"message":"Input data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#;

    /// Seed empty model catalogs so the Artist active-models block renders
    /// nothing and performs no network fetch (hermetic e2e).
    fn seed_empty_catalogs() {
        crate::tools::image_catalog::seed_cache(Some(std::sync::Arc::new(
            crate::tools::image_catalog::ImageCatalog::default(),
        )));
        crate::tools::video_catalog::seed_cache(Some(std::sync::Arc::new(
            crate::tools::video_catalog::VideoCatalog::default(),
        )));
    }

    /// A rejected input image must not fail the run: the image is durably
    /// stripped from the most recent user message (replaced by the explanatory
    /// phrase) and the run continues through its normal loop — the retried
    /// call carries the corrected message and the agent answers.
    #[tokio::test]
    #[serial_test::serial(active_models)] // seeds the process-global catalog caches
    #[expect(clippy::await_holding_lock)] // retry_tests_lock serializes process-global test seams
    async fn rejected_input_image_is_stripped_and_run_continues() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        let agent_id = "e2e_artist_image_reject";
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .err_http(400, IMAGE_REJECTION_BODY)
                .ok("The image was rejected by the provider's content check."),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent_with_role(vec![], crate::Role::Artist);
        agent.agent_id = agent_id.to_string();
        let resp = agent
            .work("[IMAGE:/tmp/photo.png] describe this photo", false)
            .await
            .expect("the run must not fail wholesale on a rejected input image");
        assert_eq!(
            resp,
            "The image was rejected by the provider's content check."
        );

        // The corrected message reached the model: the second request's
        // message list carries the phrase and no image marker in the user
        // message (the Artist system prompt mentions "[IMAGE:path]" markers
        // literally — the assertion must target the user segment).
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
        // The user's original text plus the phrase, and NO extra assistant/system
        // message was injected for the rejection (single phrase inside the
        // user's own message).
        assert_eq!(
            history
                .iter()
                .filter(|m| m.role == crate::ChatRole::User)
                .count(),
            1,
            "no separate notification message was added"
        );
    }

    /// A text-content rejection with the same provider code ("Input data ..."
    /// without "image") does not trigger the strip — the run takes the normal
    /// failure path and the image stays in the session.
    #[tokio::test]
    #[serial_test::serial(active_models)] // seeds the process-global catalog caches
    #[expect(clippy::await_holding_lock)] // retry_tests_lock serializes process-global test seams
    async fn text_content_rejection_follows_normal_failure_path() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        let agent_id = "e2e_artist_text_reject";
        let fake = std::sync::Arc::new(FakeProvider::new().err_http(400, TEXT_REJECTION_BODY));
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent_with_role(vec![], crate::Role::Artist);
        agent.agent_id = agent_id.to_string();
        let result = agent
            .work("[IMAGE:/tmp/photo.png] describe this photo", false)
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

    /// After a successful strip, a subsequent failure in the same run does NOT
    /// re-strip: the most recent user message no longer contains image markers,
    /// so the normal failure path applies (no repeated stripping, no infinite
    /// retry loop).
    #[tokio::test]
    #[serial_test::serial(active_models)] // seeds the process-global catalog caches
    #[expect(clippy::await_holding_lock)] // retry_tests_lock serializes process-global test seams
    async fn subsequent_failure_after_strip_takes_normal_failure_path() {
        use crate::util::test::{
            FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
        };
        let _lock = retry_tests_lock();
        crate::util::test::init_test_stores().await;
        seed_empty_catalogs();
        let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());

        let agent_id = "e2e_artist_second_failure";
        let fake = std::sync::Arc::new(
            FakeProvider::new()
                .err_http(400, IMAGE_REJECTION_BODY)
                .err_http(400, IMAGE_REJECTION_BODY),
        );
        let provider: std::sync::Arc<dyn crate::Provider> = fake.clone();
        let _provider_guard = install_fake_provider(provider);

        let mut agent = make_agent_with_role(vec![], crate::Role::Artist);
        agent.agent_id = agent_id.to_string();
        let result = agent
            .work("[IMAGE:/tmp/photo.png] describe this photo", false)
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
}
