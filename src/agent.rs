use std::time::Instant;

use crate::providers::compatible::strip_think_tags;
use crate::providers::reasoning_roundtrip::{
    assistant_replay_payload, coalesce_streamed_reasoning_details, merge_reasoning_details_delta,
    push_reasoning_delta,
};
use crate::providers::{chat, stream_chat};
use crate::session::Session;
use crate::tools::{
    AskTool, ToolExecutionOutcome, find_tool, format_tool_failure_feedback, normalize_tool_call,
    sanitize_success_tool_output, unknown_tool_message,
};
use crate::util::{UnwrapPoison, plaintext_for_display, scrub_credentials};
use crate::{
    Agent, ChatMessage, ChatRequest, ChatResponse, Reasoning, Role, StreamChunk, StreamEvent, Tool,
    ToolCall, ToolOutputPhase,
};
use std::fmt::Write;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// Maximum LLM iterations per agent turn.
///
/// A misbehaving/adversarial model that keeps emitting tool calls will loop
/// indefinitely without this guard.  1000 iterations is generous enough for
/// any legitimate run (typical agents finish in 3–8 turns) while catching
/// truly stuck loops.
const MAX_LLM_ITERATIONS: usize = 1000;

/// Maximum length of serialized arguments stored in per-call stats.
/// Longer arguments are truncated at a UTF-8-safe boundary.
const MAX_STATS_ARG_LENGTH: usize = 500;

/// Extract file paths from successful media-generation tool outcomes.
///
/// Scans the zipped tool calls and outcomes for media-generation tools,
/// parsing the output for their media marker prefixes (e.g. `[IMAGE:path]`,
/// `[VIDEO:path]`) and returning `(tool_name, path)` pairs.
fn extract_media_from_outcomes(
    tools: &[Box<dyn Tool>],
    tool_calls: &[ToolCall],
    outcomes: &[ToolExecutionOutcome],
) -> Vec<(String, String)> {
    let mut paths = Vec::new();
    for (call, outcome) in tool_calls.iter().zip(outcomes.iter()) {
        if outcome.success
            && let Some(marker_prefix) = find_tool(tools, &call.name).and_then(Tool::media_marker)
        {
            // Output may be "text\n\n[IMAGE:path]" or "[IMAGE:path]" — extract path
            let path = outcome
                .output
                .split(marker_prefix)
                .nth(1)
                .and_then(|s| s.split(']').next())
                .unwrap_or(&outcome.output)
                .to_string();
            paths.push((call.name.clone(), path));
        }
    }
    paths
}

impl Agent {
    /// Create a new agent with the given session_key, role, workspace, and optional ticket.
    ///
    /// Tools are derived from the [`Role`] via [`Role::tools`], with the Manager
    /// role receiving its async [`AskTool`] here where the agent identity is known.
    /// Automatically registers with [`crate::registry::AGENT_REGISTRY`] and creates an
    /// internal [`CancellationToken`]. The agent is deregistered on [`Drop`].
    #[must_use]
    pub fn new(
        session_key: String,
        role: crate::Role,
        ws: &crate::Workspace,
        ticket: Option<crate::board::Ticket>,
    ) -> Self {
        let mut tools = role.tools();
        // Manager gets an async AskTool — Analyst-only dispatching, results
        // delivered via wake-up (run_agent). Does not block the caller.
        if role == Role::Manager {
            tools.push(Box::new(AskTool::new(
                vec![Role::Analyst],
                Some(session_key.clone()),
            )));
        }
        let tool_specs = tools.iter().map(|t| t.spec()).collect();

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let label = if let Some(ref t) = ticket {
            format!("{}: {}", role.as_str(), t.title)
        } else {
            role.to_string()
        };
        let generation = crate::registry::AGENT_REGISTRY.register_unguarded(
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
        let history_tokens = crate::session::summarization::estimate_tokens(self.session.history());
        if history_tokens > crate::session::summarization::SUMMARIZATION_THRESHOLD {
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
            let mut accumulated_media_paths: Vec<(String, String)> = Vec::new();
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
                let (mut display_text, tool_calls, history_content) = match self.llm_call().await {
                    Ok(resp) => prepare_assistant_turn(resp),
                    Err(e) => {
                        return Err(e.context(format!("LLM step failed at iteration {iteration}")));
                    }
                };

                if tool_calls.is_empty() {
                    self.session.push_assistant(history_content);
                    // Append any pending media markers the model may have omitted
                    for (tool_name, path) in &accumulated_media_paths {
                        if let Some(prefix) =
                            find_tool(&self.tools, tool_name).and_then(Tool::media_marker)
                        {
                            let marker = format!("{prefix}{path}]");
                            if !display_text.contains(&marker) {
                                let _ = write!(display_text, "\n{marker}");
                            }
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
            .map(|call| {
                find_tool(&self.tools, &call.name)
                    .is_none_or(|tool| tool.side_effects(&call.arguments))
            })
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
        let start = Instant::now();
        let (tool_name, tool_arguments) = normalize_tool_call(call_name, call_arguments);
        if tool_name != call_name {
            tracing::debug!(
                original = %call_name,
                normalized = %tool_name,
                "Repaired tool call name"
            );
        }

        // Two distinct log levels for error arms (info vs warn) distinguish
        // unknown-tool failures from execution errors without string-prefix matching.
        let (outcome, error_reason) = match find_tool(&self.tools, &tool_name) {
            None => {
                let reason = unknown_tool_message(&tool_name);
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
                                output: sanitize_success_tool_output(
                                    tool,
                                    &tool_arguments,
                                    &output_text,
                                ),
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

        let response = match stream_assistant_response(request.clone()).await {
            Ok(response) => response,
            Err(stream_err) => {
                tracing::warn!(
                    "provider streaming failed, falling back to non-streaming chat: {stream_err}"
                );
                chat(request).await?
            }
        };

        Ok(response)
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
                tool.debug_output(phase, &call.arguments, outcome)
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
                None => crate::util::format_tool_output(&outcome.output),
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

    pub(crate) fn model(&self) -> String {
        crate::config::CONFIG.role_model(self.role)
    }

    /// Resolve reasoning effort for this agent's role from current config.
    fn reasoning_effort(&self) -> Option<String> {
        crate::config::CONFIG.role_reasoning_effort(self.role)
    }

    /// Resolve temperature for this agent's role from static role metadata.
    const fn temperature(&self) -> f32 {
        crate::role::role_info(&self.role).temperature
    }

    /// Resolve provider order and allow_fallbacks for this agent's model from live config.
    /// Lazily resolved each call to respect runtime hot-reload.
    fn provider_routing(&self) -> crate::config::ModelRouting {
        crate::config::CONFIG.model_routing(&self.model())
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
    /// [`Self::extract_structured`] uses the same parameter sources (routed through
    /// [`crate::extraction::ExtractionConfig`] rather than calling this method
    /// directly).  [`Self::summarize`] calls this method directly.
    fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
        allow_image_parts: bool,
    ) -> ChatRequest {
        let routing = self.provider_routing();
        ChatRequest {
            messages,
            tools: Some(self.tool_specs.clone()),
            model: self.model(),
            allow_image_parts,
            temperature: self.temperature(),
            reasoning_effort: self.reasoning_effort(),
            provider_order: routing.provider_order,
            provider_allow_fallbacks: routing.allow_fallbacks,
        }
    }

    // ── Extraction / summarisation ──

    /// Extract a structured `T` from the agent's session history.
    ///
    /// KV-cache requirements: uses the same parameter sources as
    /// [`Self::build_chat_request`] (model, temperature, reasoning_effort,
    /// tools, provider routing), routed through
    /// [`crate::extraction::ExtractionConfig`] instead of calling that method
    /// directly.
    pub(crate) async fn extract_structured<T: serde::de::DeserializeOwned>(
        &self,
        extraction_prompt: &str,
        retry_prompt: &str,
        max_attempts: usize,
    ) -> anyhow::Result<T> {
        // Bind to local so the borrow lives across the .await.
        let model = self.model();
        let config = crate::extraction::ExtractionConfig {
            model: &model,
            tool_specs: Some(&self.tool_specs),
            temperature: self.temperature(),
            reasoning_effort: self.reasoning_effort(),
            max_attempts,
        };
        crate::extraction::retry_extract_structured(
            self.session.history(),
            extraction_prompt,
            retry_prompt,
            config,
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

/// Try to stream a full assistant response from the provider.
///
/// Accumulates deltas, reasoning, and tool calls from the stream,
/// then coalesces reasoning details into the response.
async fn stream_assistant_response(request: ChatRequest) -> anyhow::Result<ChatResponse> {
    let mut provider_stream = stream_chat(request);
    let mut response_text = String::new();
    let mut reasoning: Option<String> = None;
    let mut reasoning_content: Option<String> = None;
    let mut reasoning_details: Vec<serde_json::Value> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    while let Some(event_result) = provider_stream.next().await {
        let event = event_result.map_err(|err| anyhow::anyhow!("provider stream error: {err}"))?;
        match event {
            StreamEvent::Final => break,
            StreamEvent::ToolCall(tool_call) => {
                tool_calls.push(tool_call);
            }
            StreamEvent::TextDelta(chunk) => {
                let StreamChunk {
                    delta,
                    reasoning: reason,
                } = chunk;
                if let Some(patch) = reason {
                    push_reasoning_delta(&mut reasoning, patch.reasoning);
                    push_reasoning_delta(&mut reasoning_content, patch.reasoning_content);
                    if let Some(d) = patch.reasoning_details {
                        merge_reasoning_details_delta(&mut reasoning_details, d);
                    }
                }
                response_text.push_str(&delta);
            }
        }
    }

    let coalesced_details = coalesce_streamed_reasoning_details(&reasoning_details);
    let reasoning = Reasoning::from_optional_parts(reasoning, reasoning_content, coalesced_details);

    // Strip <think>...</think> blocks that some models embed inline in content.
    // Mirror the non-streaming path (effective_content_optional) which returns
    // None when stripping leaves an empty string (the model only emitted
    // reasoning wrapped in think tags — fall through to reasoning display).
    let text = {
        let stripped = strip_think_tags(&response_text);
        if stripped.is_empty() {
            None
        } else {
            Some(stripped)
        }
    };

    Ok(ChatResponse {
        text,
        tool_calls,
        usage: None,
        reasoning,
    })
}

/// Result of preparing an assistant turn from the LLM response.
/// `(display_text, tool_calls, history_content)`
type PreparedAssistantTurn = (String, Vec<ToolCall>, String);

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

    (display_text, tool_calls, history_content)
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
}
