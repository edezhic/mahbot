use std::time::Instant;

use anyhow::Context;

use crate::providers::chat;
use crate::providers::plaintext_for_display;
use crate::providers::reasoning_roundtrip::assistant_replay_payload;
use crate::session::Session;
use crate::tools::{
    ToolExecutionOutcome, find_tool, format_tool_failure_feedback, normalize_tool_call,
    scrub_tool_output,
};
use crate::util::{MEDIA_MARKER_RE, UnwrapPoison, parse_media_marker, scrub_credentials};
use crate::{Agent, ChatMessage, ChatRequest, ChatResponse, Tool, ToolCall, ToolOutputPhase};
use std::fmt::Write;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// Maximum LLM iterations before the agent loop bails out.
///
/// **DO NOT REDUCE this value without benchmarking against real ticket iteration
/// distributions in this codebase.**
///
/// # Why this is intentionally generous (1000)
///
/// * Agents routinely consume hundreds of iterations in normal, legitimate work:
///   multi-file editing with compile-error feedback loops, research → implement →
///   review cycles, and sequential tool dependencies all require many turns — this
///   is deliberate problem-solving, not a runaway loop.
///
/// * Cost is not a concern. Even a full 1000-iteration run with the default model
///   (DeepSeek V4 Flash) costs well under $1, so there is zero cost reason to
///   lower the limit.
///
/// * Running to the iteration cap is EXTREMELY rare with modern models. The limit
///   exists only as a safety net for pathological edge cases — it is not protecting
///   against a common or recurring problem.
///
/// * Reducing the cap prematurely would cause legitimate long-running agents to
///   fail mid-task, wasting more time and tokens on restarts than the iterations
///   themselves would have consumed.
///
/// Value last intentionally reviewed: 2026-07-11
const MAX_LLM_ITERATIONS: usize = 1000;

/// Maximum length of serialized arguments stored in per-call stats.
/// Longer arguments are truncated at a UTF-8-safe boundary.
const MAX_STATS_ARG_LENGTH: usize = 500;

/// Extract file paths from successful media-generation tool outcomes.
///
/// Scans the zipped tool calls and outcomes for media-generation tools,
/// parsing the output for their media marker prefixes (e.g. `[IMAGE:path]`,
/// `[VIDEO:path]`) and returning `(marker_prefix, path)` pairs.
///
/// Only successfully-executed tools with a defined [`Tool::media_marker`] are
/// inspected. Non-media tools and failed outcomes are silently skipped.
///
/// # Parse rules
///
/// * The output is scanned using [`MEDIA_MARKER_RE`] for markers matching the
///   tool's media kind (e.g. `IMAGE` for a tool returning `[IMAGE:`).
/// * Only markers whose `kind` matches the tool's [`Tool::media_marker`] prefix
///   are extracted.
/// * The regex inherently requires a closing `]` and a non-empty `path` group,
///   so malformed markers (e.g. `[IMAGE:bogus`) or empty paths (`[IMAGE:]`) are
///   correctly skipped.
///
/// # Pre-existing limitation
///
/// File paths containing `]` would be truncated because the regex path group
/// is `[^\]]+` (one or more characters that are not `]`). All media-generation
/// tools produce paths that never contain `]` in practice (temporary files with
/// safe names), so this is not a concern.
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

impl Agent {
    /// Create a new agent with the given session_key, role, workspace, and optional ticket.
    ///
    /// Tools are derived from [`crate::Role`] via [`crate::Role::tools`].
    /// Automatically registers with [`crate::registry::AGENT_REGISTRY`] and creates an
    /// internal [`CancellationToken`]. The agent is deregistered on [`Drop`].
    #[must_use]
    pub fn new(
        session_key: String,
        role: crate::Role,
        ws: &crate::Workspace,
        ticket: Option<crate::board::Ticket>,
    ) -> Self {
        let tools = role.tools();
        let tool_specs = tools.iter().map(|t| t.spec()).collect();

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let label = if let Some(ref t) = ticket {
            format!("{}: {}", role.as_str(), t.title)
        } else {
            role.to_string()
        };
        let generation = crate::registry::AGENT_REGISTRY.register(
            session_key.clone(),
            role.to_string(),
            ticket.as_ref().map(|t| t.id.clone()),
            ws,
            label,
            cancel_token.clone(),
        );

        Self {
            id: session_key,
            role,
            session: Session::default(),
            workspace: Arc::new(ws.clone()),
            tools,
            tool_specs,
            cancel_token,
            ticket,
            generation,
            tool_stats: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if self.generation > 0 {
            crate::registry::AGENT_REGISTRY.deregister(&self.id, self.generation);
        }
    }
}

impl Agent {
    /// Flush accumulated tool stats to `stats.db`, then persist the final
    /// assistant message via `session.finalize(&self.id)`. Flush failures are logged
    /// but do not abort finalization.
    pub async fn finalize_session(&mut self) -> anyhow::Result<()> {
        // Drain accumulated tool usage stats
        let stats = {
            let mut guard = self.tool_stats.lock().unwrap_poison();
            std::mem::take(&mut *guard)
        };
        if !stats.is_empty()
            && let Err(e) = crate::stats::store()
                .flush_batch(&self.id, self.role.as_str(), &self.workspace.path, &stats)
                .await
        {
            tracing::warn!(
                agent_id = %self.id,
                role = %self.role.as_str(),
                error = %e,
                "Failed to flush tool usage stats"
            );
        }

        // When cancelled (by user /stop or by global shutdown), no assistant
        // message is expected — skip finalization to avoid the "finalize called
        // but no assistant message" warning. Intermediate messages are already
        // persisted via commit_tool_results inside llm_loop, so no data loss.
        if self.cancel_token.is_cancelled() || crate::shutdown::shutdown_token().is_cancelled() {
            tracing::debug!(
                agent_id = %self.id,
                "Session finalize skipped (agent cancelled or shutdown)"
            );
            return Ok(());
        }

        self.session.finalize(&self.id).await
    }

    /// Return a clone of the cancellation token for external use
    /// (e.g., `tokio::select!` racing, typing task ownership).
    #[must_use]
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Check whether cancellation has been triggered on this agent.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Run a complete agent turn: initialize session, work loop (with shutdown
    /// cancellation), finalize session, and optionally introspect.
    pub async fn work(&mut self, msg: &str) -> anyhow::Result<String> {
        // Open or resume a session for this agent turn.
        self.session
            .init(
                &self.id,
                msg,
                &self.workspace,
                &self.role,
                self.ticket.as_ref(),
            )
            .await?;

        // Summarize if context window is getting long.
        // KV-cache preservation: self.summarize() keeps all parameters
        // identical (see build_chat_request) so the cached prefix is reusable.
        let history_tokens = crate::session::estimate_tokens(self.session.history());
        if history_tokens > crate::session::SUMMARIZATION_THRESHOLD {
            match self.summarize().await {
                Ok(summary) => {
                    self.session
                        .apply_summary(
                            &self.id,
                            msg,
                            &summary,
                            &self.workspace,
                            &self.role,
                            self.ticket.as_ref(),
                        )
                        .await;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Summarization failed — continuing with full history");
                }
            }
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
        let span = tracing::info_span!("agent", agent_id = %self.id, role = %self.role, workspace = %self.workspace.path);
        async {
            let mut iteration = 0usize;
            let mut accumulated_media_paths: Vec<(&'static str, String)> = Vec::new();
            loop {
                if self.cancel_token.is_cancelled() {
                    anyhow::bail!("Agent cancelled by user");
                }
                if iteration >= MAX_LLM_ITERATIONS {
                    anyhow::bail!(
                        "Agent exceeded maximum of {MAX_LLM_ITERATIONS} LLM iterations \
                         — model may be stuck in a tool-calling loop"
                    );
                }
                let PreparedAssistantTurn {
                    mut display_text,
                    tool_calls,
                    history_content,
                } = prepare_assistant_turn(
                    self.llm_call()
                        .await
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
                self.log_tool_notifications(&tool_calls, None, ToolOutputPhase::Before);

                let all_outcomes = self.execute_tool_group(&tool_calls).await;

                self.log_tool_notifications(
                    &tool_calls,
                    Some(&all_outcomes),
                    ToolOutputPhase::After,
                );

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
        while i < tool_calls.len() {
            if side_flags[i] {
                // Side-effecting: single-call group, executed alone.
                let outcome = self
                    .execute_tool(&tool_calls[i].name, tool_calls[i].arguments.clone())
                    .await;
                outcomes.push(outcome);
                i += 1;
            } else {
                // Read-only group: extend while consecutive calls are also read-only.
                let group_start = i;
                while i < tool_calls.len() && !side_flags[i] {
                    i += 1;
                }
                let group_calls = &tool_calls[group_start..i];

                // Execute the entire read-only group in parallel.
                let group_outcomes: Vec<_> = futures_util::future::join_all(
                    group_calls
                        .iter()
                        .map(|call| self.execute_tool(&call.name, call.arguments.clone())),
                )
                .await;

                outcomes.extend(group_outcomes);
            }
        }
        outcomes
    }

    /// Construct a failure outcome tuple for an error reason.
    ///
    /// Both error arms in [`Self::execute_tool`] (unknown tool and execution error) produce
    /// the same `(ToolExecutionOutcome, Option<String>)` shape; this helper
    /// eliminates the byte-for-byte duplicated construction.
    ///
    /// Assumes `reason` may contain sensitive data; scrubs before use in feedback
    /// text, tracing logs, and stats.
    #[must_use]
    fn failure_outcome(
        call_name: &str,
        call_arguments: &serde_json::Value,
        reason: &str,
    ) -> (ToolExecutionOutcome, Option<String>) {
        let reason = scrub_credentials(reason);
        (
            ToolExecutionOutcome {
                output: format_tool_failure_feedback(call_name, call_arguments, &reason),
                success: false,
            },
            Some(reason),
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
            tracing::info!(
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
                            None,
                        )
                    }
                    Err(e) => {
                        let (outcome, error_reason) = Self::failure_outcome(
                            &tool_name,
                            &tool_arguments,
                            &format!("Error executing {tool_name}: {e}"),
                        );
                        let reason = error_reason
                            .as_deref()
                            .expect("failure_outcome always returns Some");
                        tracing::debug!(
                            tool = %tool_name,
                            duration_ms = duration.as_millis(),
                            success = false,
                            "Tool execution error: {reason}"
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
            let arguments = args_scrubbed
                [..args_scrubbed.floor_char_boundary(MAX_STATS_ARG_LENGTH)]
                .to_string();

            let mut guard = self.tool_stats.lock().unwrap_poison();
            guard.push(crate::ToolCallRecord {
                tool_name,
                arguments,
                duration_ms,
                success: outcome.success,
                error_message: error_reason,
            });
        }
        outcome
    }

    async fn llm_call(&self) -> anyhow::Result<ChatResponse> {
        let request = self.build_chat_request(
            self.session.history().to_vec(),
            self.role.requires_multimodal(),
        );

        chat(request).await
    }

    /// Log tool-call notifications
    fn log_tool_notifications(
        &self,
        calls: &[ToolCall],
        outcomes: Option<&[ToolExecutionOutcome]>,
        phase: ToolOutputPhase,
    ) {
        let tools = &self.tools;
        for (i, call) in calls.iter().enumerate() {
            let outcome = outcomes.and_then(|o| o.get(i));
            let msg = if let Some(tool) = find_tool(tools, &call.name) {
                tool.debug_output(
                    phase,
                    &call.arguments,
                    outcome.map(|o| (o.output.as_str(), o.success)),
                )
            } else {
                let args_preview = crate::util::summarize_args(&call.arguments);
                match outcome {
                    None => Some(format!("🔧 `{}`({})", call.name, args_preview)),
                    Some(outcome) => {
                        let status = if outcome.success { "✅" } else { "❌" };
                        Some(format!("{status} `{}`({})", call.name, args_preview))
                    }
                }
            };
            if let Some(msg) = msg {
                tracing::info!("{msg}");
            }
        }
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

        // Batch-persist all messages in a single transaction.
        crate::session::store()
            .batch_append(&self.id, &db_messages)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to persist tool results: {e}"))?;

        // Push to in-memory history (infallible — error above aborts the turn).
        self.session.push_messages(&db_messages);

        Ok(())
    }

    /// Build a [`ChatRequest`] from the given messages and image-parts flag,
    /// using the agent's current model, tools, temperature, reasoning-effort,
    /// and provider-routing settings.
    ///
    /// All parameter sources are lazily resolved each call so that runtime
    /// hot-reload (model, routing, reasoning-effort) is reflected immediately.
    ///
    /// # KV-cache preservation
    ///
    /// Every call site that produces a chat request for the same logical
    /// conversation *must* use exactly the same parameter sources — model,
    /// temperature, reasoning_effort, tools (critically tools!), and provider
    /// routing — so that the provider can reuse the cached prefix computed
    /// during the original agent call.  Any deviation (including dropping
    /// tools) forces the provider to recompute the entire KV-cache prefix.
    ///
    /// [`Self::extract_structured`] calls this method internally to derive its
    /// parameter set.  [`Self::summarize`] calls this method directly.
    fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
        allow_image_parts: bool,
    ) -> ChatRequest {
        let model = crate::config::CONFIG.role_model(self.role);
        let routing = crate::config::CONFIG.model_routing(&model);
        ChatRequest {
            messages,
            tools: Some(self.tool_specs.clone()),
            model,
            allow_image_parts,
            // temperature is a compile-time constant from role metadata — not hot-reloadable
            temperature: crate::role::role_info(&self.role).temperature,
            max_tokens: Some(crate::DEFAULT_MAX_TOKENS),
            reasoning_effort: Some(crate::config::CONFIG.role_reasoning_effort(self.role)),
            provider_order: routing.provider_order,
            provider_allow_fallbacks: routing.allow_fallbacks,
        }
    }

    // ── Extraction / summarisation ──

    /// Extract a structured `T` from the agent's session history.
    ///
    /// KV-cache requirements: uses the same parameter sources as
    /// [`Self::build_chat_request`] (model, temperature, reasoning_effort,
    /// tools, provider routing), obtained by calling that method and passing
    /// the resulting [`ChatRequest`] to
    /// [`crate::extraction::retry_extract_structured`].
    pub(crate) async fn extract_structured<T: serde::de::DeserializeOwned>(
        &self,
        extraction_prompt: &str,
        retry_prompt: &str,
        max_attempts: usize,
    ) -> anyhow::Result<T> {
        // Build a params-only ChatRequest (messages will be substituted by
        // retry_extract_structured with the extraction history).
        let params = self.build_chat_request(vec![], false);
        crate::extraction::retry_extract_structured(
            self.session.history(),
            extraction_prompt,
            retry_prompt,
            &params,
            max_attempts,
        )
        .await
    }

    /// Summarise the agent's session history.
    ///
    /// KV-cache requirements: see [`Self::build_chat_request`].
    pub(crate) async fn summarize(&self) -> anyhow::Result<String> {
        let mut history = self.session.history().to_vec();
        history.push(crate::ChatMessage::user(self.role.summary_prompt()));

        let chat_resp = crate::providers::chat(self.build_chat_request(history, false)).await?;

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
}

/// Result of preparing an assistant turn from the LLM response.
struct PreparedAssistantTurn {
    display_text: String,
    tool_calls: Vec<ToolCall>,
    history_content: String,
}

/// Prepare assistant response data from the LLM response.
fn prepare_assistant_turn(response: ChatResponse) -> PreparedAssistantTurn {
    let mut response_text = response.text_or_empty().to_string();
    let tool_calls = response.tool_calls;
    let reasoning = response.reasoning.as_ref();

    // Build structured payload BEFORE the reasoning fallback below, so the
    // content field faithfully captures the model's original response (empty
    // for reasoning-only returns like DeepSeek with content=null).
    let json_payload =
        assistant_replay_payload(Some(&response_text), &tool_calls, reasoning).to_string();

    // When the model returns only reasoning (e.g. DeepSeek with content=null),
    // fall back to plaintext reasoning as the display text.
    if response_text.is_empty()
        && tool_calls.is_empty()
        && let Some(fb) = plaintext_for_display(reasoning)
    {
        response_text = fb;
    }

    // Dispatch on whether tool calls and/or reasoning are present.
    // Three arms: plain answer (no tools, no reasoning), reasoning-only
    // (reasoning present but no tools), and tool calls (tools present).
    let (display_text, history_content) = match (tool_calls.is_empty(), reasoning.is_some()) {
        // Plain final answer — both display and history use response text directly.
        (true, false) => (response_text.clone(), response_text),
        // Reasoning present — show reasoning/answer text to user,
        // persist structured JSON payload with empty content + reasoning fields.
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

/// Core agent lifecycle: create agent (auto-registers with its own
/// CancellationToken), run work, handle cancellation and errors.
/// Returns the agent (even on failure) and the response on success.
///
/// **Cancellation safety**: Even if `agent.work()` completes before the token
/// fires (the classic race), we check `is_cancelled()` after work completes
/// and discard the result — preventing overwrites of externally-set `cancelled`
/// status in downstream code.
///
/// Returns `(agent, Some(response))` on success.
/// Returns `(agent, None)` on cancellation (discard result) or error (already logged).
pub(crate) async fn run_agent(
    session_key: String,
    role: crate::Role,
    ws: &crate::Workspace,
    ticket: Option<&crate::board::Ticket>,
    message: &str,
) -> (Agent, Option<String>) {
    let mut agent = Agent::new(session_key, role, ws, ticket.cloned());

    let result = agent.work(message).await;

    // Cancellation safety: if the token fired after work() completed but
    // before we checked, discard the result to prevent overwriting of
    // externally-set cancelled status in downstream code.
    if agent.is_cancelled() {
        return (agent, None);
    }

    match result {
        Ok(response) => (agent, Some(response)),
        Err(e) => {
            // During global (SIGTERM/SIGINT) shutdown, every in-flight agent
            // hits the `tokio::select!` shutdown branch in work() and returns
            // an error — this is expected, not a real failure. Log at debug!
            // level to avoid misleading ERROR noise on clean shutdown.
            if crate::shutdown::shutdown_token().is_cancelled() {
                tracing::debug!(
                    workspace = %ws.name,
                    role = %role,
                    ticket = ticket.map(|t| t.id.as_str()),
                    error = %e,
                    "Agent failed during shutdown"
                );
            } else {
                tracing::error!(
                    workspace = %ws.name,
                    role = %role,
                    ticket = ticket.map(|t| t.id.as_str()),
                    error = %e,
                    "Agent failed"
                );
            }
            (agent, None)
        }
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
        let tool_specs = tools.iter().map(|t| t.spec()).collect();
        Agent {
            id: "test-agent".into(),
            role: crate::Role::Engineer,
            session: Session::default(),
            workspace: std::sync::Arc::new(crate::Workspace::default()),
            tools,
            tool_specs,
            cancel_token: CancellationToken::new(),
            ticket: None,
            generation: 0,
            tool_stats: std::sync::Mutex::new(Vec::new()),
        }
    }

    #[tokio::test]
    async fn tool_with_scrub_disabled_preserves_output() {
        assert_scrubbed(false).await;
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
    #[allow(clippy::too_many_lines)]
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
}
