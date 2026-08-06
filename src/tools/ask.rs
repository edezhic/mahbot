//! AskTool — spawns a sub-agent to ask a question.
//!
//! Available to the Engineer and Maintainer agents (sync mode), and to the
//! Manager and Assistant agents (async mode). In sync mode the caller blocks
//! until the sub-agent completes. In async mode the sub-agent is dispatched
//! in a background task and the result is injected back to the caller's
//! agent channel via [`crate::message_router::route`].

use crate::agent::run_agent;
use crate::config::CONFIG;
use crate::message_router::{self, AgentJob, JobKind};
use crate::prompt::{load_prompt, substitute};
use crate::session::{ask_agent_id, resolve_agent_id};
use crate::tools::Tool;
use crate::{ChatMessage, ChatRequest, DEFAULT_MAX_TOKENS, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde_json::json;

/// Controls sub-agent dispatch behaviour.
///
/// [`Sync`](DispatchMode::Sync) blocks the caller until the sub-agent completes.
/// [`Async`](DispatchMode::Async) dispatches the sub-agent in a background task
/// and injects the result via the caller's agent queue.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DispatchMode {
    Sync,
    Async,
}

impl DispatchMode {
    /// Returns `true` when this dispatch mode is [`Async`](DispatchMode::Async).
    #[must_use]
    pub const fn is_async(self) -> bool {
        matches!(self, Self::Async)
    }
}

pub struct AskTool {
    pub allowed_roles: Vec<Role>,
    /// Controls how the sub-agent is dispatched.
    /// - [`DispatchMode::Sync`] — blocks the caller until the sub-agent completes.
    /// - [`DispatchMode::Async`] — dispatches in a background task, result
    ///   delivered via the caller's agent queue.
    dispatch_mode: DispatchMode,
    /// The role of the calling agent. Used to route async results to the
    /// correct agent channel (Manager → manager_{ws}, Assistant → direct_{...}).
    pub caller_role: Role,
}

impl AskTool {
    #[must_use]
    pub const fn new(
        allowed_roles: Vec<Role>,
        dispatch_mode: DispatchMode,
        caller_role: Role,
    ) -> Self {
        Self {
            allowed_roles,
            dispatch_mode,
            caller_role,
        }
    }

    fn formatted_allowed_roles(&self) -> String {
        self.allowed_roles
            .iter()
            .map(|r| format!("'{}'", r.as_str()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[async_trait]
impl Tool for AskTool {
    fn name(&self) -> &'static str {
        "ask"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let role_str = self.formatted_allowed_roles();
        super::tool_params_schema(
            &json!({
                "role": {
                    "type": "string",
                    "description": format!("Role name of the agent: {role_str}")
                },
                "ask": {
                    "type": "string",
                    "description": "The ask to delegate to the agent"
                }
            }),
            &["role", "ask"],
        )
    }

    /// Async sub-agents could theoretically produce side effects through their
    /// own tool sets, but in practice the sub-agents dispatched by AskTool
    /// (Analysts) have no mutation tools (no edit, no full shell — only
    /// read-only shell, which reports [`Self::side_effects`] = false). This
    /// classification is coupled to `Role::tools()`; if Analyst ever gains
    /// side-effecting tools, this must be reconsidered.
    fn side_effects(&self) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let role_str = super::get_str(&args, "role")?;
        let ask = super::get_str(&args, "ask")?;

        let allowed_str = self.formatted_allowed_roles();
        let role: Role = match role_str.parse() {
            Ok(r) if self.allowed_roles.contains(&r) => r,
            Ok(_) => {
                anyhow::bail!("Cannot delegate to '{role_str}'. Only {allowed_str} are supported.");
            }
            Err(_) => {
                anyhow::bail!("Unknown role '{role_str}'. Use {allowed_str}.");
            }
        };

        // Async dispatch path — delegate to sub-agent in background.
        // Read user context from task-locals (set by Agent work loop
        // before each tool.execute() call) so the queued result carries
        // the correct user identity for per-user delivery.
        if self.dispatch_mode.is_async() {
            let ws = ws.clone();
            let ask = ask.to_string();
            let caller_role = self.caller_role;
            let user_name = crate::agent::CURRENT_TOOL_USER_NAME
                .try_with(String::clone)
                .unwrap_or_default();
            let channel = crate::agent::CURRENT_TOOL_CHANNEL
                .try_with(String::clone)
                .unwrap_or_default();

            tokio::spawn(async move {
                let message = build_async_ask_message(run_sub_agent(&ws, role, &ask).await);

                // Route result to the caller's agent channel.
                let target_agent_id =
                    resolve_agent_id(&channel, &user_name, caller_role.as_str(), &ws.name);
                message_router::route(
                    &target_agent_id,
                    AgentJob {
                        content: message,
                        workspace_name: ws.name.clone(),
                        user_name,
                        channel,
                        kind: JobKind::AskToolResult,
                        role: caller_role,
                        reply_target: None,
                    },
                );
            });

            return Ok("Sub-agent dispatched. Results will follow shortly.".to_string());
        }

        // Sync path — blocks caller until sub-agent completes.
        run_sub_agent(ws, role, ask).await
    }
}

/// Build the `<ask-tool-result>` envelope message for an async ask dispatch.
///
/// Shared by the async dispatch path (the `tokio::spawn` body in
/// [`AskTool::execute`]) and tests — the envelope shape that reaches the
/// caller's agent channel is production code, not a test re-wrap.
fn build_async_ask_message(result: anyhow::Result<String>) -> String {
    match result {
        Ok(text) => wrap_ask_tool_result(&text),
        Err(e) => {
            tracing::debug!(error = %e, "async AskTool sub-agent failed");
            wrap_ask_tool_result(&format!("An error occurred: {e}"))
        }
    }
}

/// Shared sub-agent runner — delegates to [`run_agent`] for the given role
/// and ask. Used by both sync and async paths of [`AskTool::execute`].
///
/// For [`Role::Analyst`], spawns 3 parallel analysts and consolidates their
/// responses via a single LLM synthesis call. For all other roles, dispatches a
/// single agent.
async fn run_sub_agent(ws: &Workspace, role: Role, ask: &str) -> Result<String> {
    if role == Role::Analyst {
        return run_parallel_analysts_and_consolidate(ws, ask).await;
    }

    // Single-agent path for non-Analyst roles — delegate lifecycle to run_agent.
    let agent_id = ask_agent_id(&ws.name, role.as_str());
    let (agent, response) = run_agent(
        agent_id,
        role,
        ws,
        None,
        ask,
        String::new(),
        String::new(),
        None,
    )
    .await;

    if let Some(response) = response {
        Ok(response)
    } else if agent.is_cancelled() || crate::shutdown::shutdown_token().is_cancelled() {
        anyhow::bail!("Sub-agent cancelled");
    } else {
        anyhow::bail!(
            "Sub-agent failed: {}",
            agent.failure.as_deref().unwrap_or("unknown error")
        );
    }
}

/// Spawn 3 parallel analyst agents, then consolidate their responses into a
/// single comprehensive answer.
async fn run_parallel_analysts_and_consolidate(ws: &Workspace, ask: &str) -> Result<String> {
    const PARALLEL_ANALYST_COUNT: usize = 3;

    let responses = run_parallel_analysts(ws, ask, PARALLEL_ANALYST_COUNT).await;
    consolidate_analyst_responses(ws, ask, responses).await
}

/// Run `count` parallel analyst agents, returning their responses.
/// Failed / cancelled agents produce `None`.
async fn run_parallel_analysts(ws: &Workspace, ask: &str, count: usize) -> Vec<Option<String>> {
    let suffix = crate::generate_suffix();
    let futures: Vec<_> = (0..count)
        .map(|i| {
            let ask = ask.to_string();
            let ws = ws.clone();
            let suffix = suffix.clone();
            let agent_id = format!("ask_{}_{}_{}_analyst", ws.name, suffix, i);
            async move {
                let (_, response) = run_agent(
                    agent_id,
                    Role::Analyst,
                    &ws,
                    None,
                    &ask,
                    String::new(),
                    String::new(),
                    None,
                )
                .await;
                response
            }
        })
        .collect();
    join_all(futures).await
}

/// Wrap a sub-agent result in the `<ask-tool-result>` envelope delivered to
/// the caller's agent channel.
fn wrap_ask_tool_result(text: &str) -> String {
    format!("<ask-tool-result>\n\n{text}</ask-tool-result>")
}

/// Wrap escaped analyst reports in the canonical markdown shape used by the
/// consolidation request AND the fail-open raw dump: `### Report from Analyst N`
/// sections joined by blank lines (the caller adds the surrounding heading).
/// Triple-backtick escaping is applied by the caller before this function.
fn format_analyst_reports_markdown(escaped: &[String]) -> String {
    let mut parts = Vec::new();
    for (i, report) in escaped.iter().enumerate() {
        parts.push(format!("### Report from Analyst {}", i + 1));
        parts.push(report.clone());
    }
    parts.join("\n\n")
}

/// Consolidate parallel analyst responses via a single LLM synthesis call.
///
/// Message structure (per manager refinement):
/// - System message: synthesis instructions + original question (from prompt template)
/// - User message: only the analyst reports (programmatically built)
///
/// Graceful degradation:
/// - 0 valid responses → error
/// - 1 valid response → returned directly (no consolidation call)
/// - 2–3 valid responses → consolidated via [`crate::retry::retry_chat`]
///
/// Fail-open: when consolidation fails after all
/// retries (or on any immediate non-retryable error), the raw VALID analyst
/// reports are delivered with an explicit "unconsolidated" marker instead of
/// an error — findings are never lost. The result is free-form text consumed
/// only by LLMs; nothing parses it.
async fn consolidate_analyst_responses(
    ws: &Workspace,
    ask: &str,
    responses: Vec<Option<String>>,
) -> Result<String> {
    // Filter out empty / failed responses
    let valid: Vec<String> = responses
        .into_iter()
        .flatten()
        .filter(|r| !r.trim().is_empty())
        .collect();

    match valid.len() {
        0 => {
            anyhow::bail!("All parallel analysts failed to produce a response");
        }
        1 => {
            // Only one analyst responded — return directly, no consolidation call
            Ok(valid.into_iter().next().expect("just checked len == 1"))
        }
        _n => {
            // 2 or 3 responses — consolidate via a single LLM call.
            // System message: instructions + original question
            // User message: reports only (no duplicate instructions)
            let model = CONFIG.role_model(Role::Analyst);
            let routing = CONFIG.model_routing(&model);
            let prompt_template = load_prompt("consolidate/analyst.md");

            // Escape triple-backtick code fences in analyst responses to prevent
            // accidental markdown-structure corruption in the input text.
            let escaped: Vec<String> = valid
                .iter()
                .map(|r| r.replace("```", "\\`\\`\\`"))
                .collect();

            // System message = filled template (instructions + original_ask only)
            let system_content = substitute(&prompt_template, &[("{{original_ask}}", ask)]);

            // User message = reports section only, built for the exact count
            // of responding analysts (no empty headers for missing analysts).
            let reports_markdown = format_analyst_reports_markdown(&escaped);
            let user_content = format!("## Analyst Reports\n\n{reports_markdown}");

            let request = ChatRequest {
                messages: vec![
                    ChatMessage::system(&system_content),
                    ChatMessage::user(&user_content),
                ],
                tools: None,
                model,
                allow_image_parts: false,
                max_tokens: Some(DEFAULT_MAX_TOKENS),
                reasoning_effort: Some("xhigh".to_owned()),
                provider_order: routing.provider_order,
                provider_allow_fallbacks: routing.allow_fallbacks,
                response_format_json_object: false,
                meta: Some(crate::ChatRequestMeta {
                    purpose: "consolidate",
                    agent_id: format!("ask_{}_consolidation", ws.name),
                    role: Role::Analyst.as_str().to_string(),
                    workspace: ws.name.clone(),
                    ticket_id: None,
                }),
            };

            // Hardened outer retry loop: the
            // request is byte-identical across ALL attempts (no re-prompt
            // exists here); 13 attempts, backoff 5/10/20/40/60/60… s, 720 s
            // wall cap. The outer loop is the single retry authority —
            // provider-internal retries are suppressed on scoped calls, so at
            // most 13 HTTP calls happen.
            let policy = crate::retry::RetryPolicy::current();
            match crate::retry::retry_chat(request, &policy).await {
                // retry_chat only returns Ok with non-empty (trimmed) text —
                // empty responses are classified as NoResponse and retried.
                Ok(response) => Ok(response.text_or_empty().to_string()),
                Err(exhausted) => {
                    // Fail-open (Amendment A): deliver the raw valid analyst
                    // reports with an explicit marker instead of discarding
                    // them. Precedent: n=1 valid reports already pass through
                    // raw without consolidation.
                    tracing::warn!(
                        error = %exhausted,
                        "Analyst consolidation failed — delivering raw analyst reports"
                    );
                    Ok(format!(
                        "## Analyst Reports\n\n(unconsolidated — consolidation failed: {exhausted})\n\n{reports_markdown}"
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{FakeProvider, install_fake_provider, retry_tests_lock};
    use crate::workspace::test_ws;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_ask_missing_args() {
        let tool = AskTool::new(
            vec![Role::Analyst, Role::Coder, Role::Qa],
            DispatchMode::Sync,
            Role::Engineer,
        );
        let ws = test_ws("/tmp/test_ws");

        // Missing role
        let result = tool.execute(&ws, json!({"ask": "do something"})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: role"),
            "Should mention missing role"
        );

        // Missing ask
        let result = tool.execute(&ws, json!({"role": "analyst"})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: ask"),
            "Should mention missing ask"
        );
    }

    #[tokio::test]
    async fn test_ask_unsupported_role() {
        let tool = AskTool::new(vec![Role::Analyst], DispatchMode::Sync, Role::Engineer);
        let ws = test_ws("/tmp/test_ws");
        // "manager" is a valid Role but not one that AskTool can delegate to
        let args = json!({"role": "manager", "ask": "do something"});
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Invalid role should fail");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("Cannot delegate"),
            "Should mention cannot delegate: {err}"
        );
    }

    #[tokio::test]
    async fn test_ask_unknown_role() {
        let tool = AskTool::new(vec![], DispatchMode::Sync, Role::Engineer);
        let ws = test_ws("/tmp/test_ws");
        // Truly unknown role string — returns bail!
        let args = json!({"role": "nonexistent", "ask": "do something"});
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Unknown role should fail");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("Unknown role") || err.contains("nonexistent"),
            "Should mention 'Unknown role': {err}"
        );
        assert!(
            !err.contains("sage") && !err.contains("discovery") && !err.contains("maintainer"),
            "Error message should not leak internal role names: {err}"
        );
    }

    /// Tests that ask dispatches correctly to each supported role.
    /// Requires an LLM provider to be configured.
    #[tokio::test]
    #[ignore = "requires LLM provider"]
    async fn test_ask_all_roles() {
        struct Case {
            role: &'static str,
            ask: &'static str,
        }

        let cases = [
            Case {
                role: "analyst",
                ask: "Say 'hello analyst' and nothing else.",
            },
            Case {
                role: "coder",
                ask: "Say 'hello coder' and nothing else.",
            },
            Case {
                role: "qa",
                ask: "Say 'hello qa' and nothing else.",
            },
        ];

        for c in &cases {
            let tool = AskTool::new(
                vec![Role::Analyst, Role::Coder, Role::Qa],
                DispatchMode::Sync,
                Role::Engineer,
            );
            let ws = test_ws("/tmp/test_ws");
            let args = json!({"role": c.role, "ask": c.ask});
            let result = tool.execute(&ws, args).await.expect("execute");
            assert!(
                result.contains("hello"),
                "{} output should contain hello",
                c.role
            );
        }
    }

    // ── Consolidation edge-case tests (no LLM provider needed) ──

    #[tokio::test]
    async fn test_consolidate_zero_responses_returns_error() {
        let ws = test_ws("/tmp/test_ws");
        let result =
            consolidate_analyst_responses(&ws, "test question", vec![None, None, None]).await;
        assert!(result.is_err(), "0 responses should error");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("All parallel analysts failed"),
            "Error should mention analyst failure"
        );
    }

    #[tokio::test]
    async fn test_consolidate_one_response_returned_directly() {
        let ws = test_ws("/tmp/test_ws");
        let result = consolidate_analyst_responses(
            &ws,
            "test question",
            vec![Some("only answer".to_string()), None, None],
        )
        .await;
        assert!(result.is_ok(), "1 response should succeed");
        assert_eq!(
            result.unwrap(),
            "only answer",
            "Should return the single response directly without consolidation"
        );
    }

    #[tokio::test]
    async fn test_consolidate_empty_responses_filtered() {
        // Empty/whitespace-only strings should be filtered the same as None
        let ws = test_ws("/tmp/test_ws");
        let result = consolidate_analyst_responses(
            &ws,
            "test question",
            vec![
                Some("valid".to_string()),
                Some("".to_string()),
                Some("   ".to_string()),
            ],
        )
        .await;
        assert!(
            result.is_ok(),
            "1 valid after filtering empty should succeed"
        );
        assert_eq!(
            result.unwrap(),
            "valid",
            "Should return the only non-empty response"
        );
    }

    // ── consolidation fail-open ────────────────────────────────────────

    /// Helper: run consolidation with the given fake provider outcomes
    /// scripted, returning the consolidation result.
    async fn consolidate_with_script(
        responses: Vec<Option<String>>,
        fake: FakeProvider,
    ) -> anyhow::Result<String> {
        let provider: Arc<dyn crate::Provider> = Arc::new(fake);
        let _guard = install_fake_provider(provider);
        let ws = test_ws("/tmp/test_ws");
        consolidate_analyst_responses(&ws, "test question", responses).await
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_fail_open_delivers_raw_reports() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Consolidation retries (transport failures) exhaust → fail open.
        let fake = FakeProvider::new()
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            );
        let result = consolidate_with_script(
            vec![
                Some("report one".to_string()),
                Some("report two".to_string()),
                None,
            ],
            fake,
        )
        .await;
        let text = result.expect("fail-open must succeed with raw reports");
        assert!(
            text.contains("unconsolidated — consolidation failed"),
            "must carry the unconsolidated marker: {text}"
        );
        assert!(text.contains("## Analyst Reports"), "{text}");
        assert!(text.contains("### Report from Analyst 1"), "{text}");
        assert!(text.contains("report one"), "{text}");
        assert!(text.contains("### Report from Analyst 2"), "{text}");
        assert!(text.contains("report two"), "{text}");
        assert!(!text.contains("### Report from Analyst 3"), "{text}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_non_retryable_fails_open() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Even an immediate non-retryable error fails open (Amendment A).
        let fake = FakeProvider::new().err(
            crate::retry::FailureClass::NonRetryable,
            "insufficient balance",
        );
        let result = consolidate_with_script(
            vec![
                Some("alpha".to_string()),
                Some("beta".to_string()),
                Some("gamma".to_string()),
            ],
            fake,
        )
        .await;
        let text = result.expect("non-retryable consolidation failure fails open");
        assert!(
            text.contains("unconsolidated — consolidation failed"),
            "{text}"
        );
        assert!(text.contains("alpha"), "{text}");
        assert!(text.contains("beta"), "{text}");
        assert!(text.contains("gamma"), "{text}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_success_unchanged() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = FakeProvider::new().ok("synthesized answer");
        let result = consolidate_with_script(
            vec![
                Some("report a".to_string()),
                Some("report b".to_string()),
                None,
            ],
            fake,
        )
        .await;
        assert_eq!(
            result.expect("success"),
            "synthesized answer",
            "successful consolidation output unchanged"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_fail_open_mixed_failure_classes() {
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Mixed transport + NoResponse (empty text) failures exercise both
        // retry_chat branches in one script; exhaustion still fails open
        // (Amendment A). Script: transport error, then two empty responses.
        let fake = FakeProvider::new()
            .err(
                crate::retry::FailureClass::Transport,
                "error reading response body",
            )
            .ok("")
            .ok("");
        let result = consolidate_with_script(
            vec![
                Some("only usable".to_string()),
                Some("second".to_string()),
                None,
            ],
            fake,
        )
        .await;
        let text = result.expect("mixed failures still fail open");
        assert!(
            text.contains("unconsolidated — consolidation failed"),
            "{text}"
        );
        assert!(text.contains("only usable"), "{text}");
        assert!(text.contains("second"), "{text}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn test_consolidation_async_envelope_carries_marker() {
        // The async dispatch path (tokio::spawn in AskTool::execute) builds
        // its envelope via build_async_ask_message — this test drives the REAL
        // fail-open consolidation result through that production builder
        // (not a manual re-wrap), asserting the exact envelope + marker shape
        // that reaches the caller's agent channel.
        let _guard = retry_tests_lock();
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let fake = FakeProvider::new()
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down")
            .err(crate::retry::FailureClass::Transport, "down");
        let result = consolidate_with_script(
            vec![
                Some("raw report".to_string()),
                Some("raw report 2".to_string()),
                None,
            ],
            fake,
        )
        .await;
        let envelope = build_async_ask_message(result);
        assert!(envelope.contains("<ask-tool-result>"), "{envelope}");
        assert!(
            envelope.contains("unconsolidated — consolidation failed"),
            "{envelope}"
        );
        assert!(envelope.contains("raw report"), "{envelope}");
        assert!(
            envelope.ends_with("</ask-tool-result>"),
            "envelope must close: {envelope}"
        );
    }

    #[tokio::test]
    async fn async_envelope_wraps_sub_agent_errors() {
        // The async dispatch path's error branch: a failed sub-agent is
        // wrapped in the same envelope with the error text.
        let envelope = build_async_ask_message(Err(anyhow::anyhow!("sub-agent exploded")));
        assert!(envelope.contains("<ask-tool-result>"), "{envelope}");
        assert!(
            envelope.contains("An error occurred: sub-agent exploded"),
            "{envelope}"
        );
        assert!(
            envelope.ends_with("</ask-tool-result>"),
            "envelope must close: {envelope}"
        );
    }
}
