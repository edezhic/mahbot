//! AskTool — spawns a sub-agent to ask a question.
//!
//! Available to the Engineer and Maintainer agents (sync mode), and to the
//! Manager and Assistant agents (async mode). In sync mode the caller blocks
//! until the sub-agent completes. In async mode the sub-agent is dispatched
//! in a background task and the result is injected back to the caller's
//! agent channel via [`crate::message_router::route`].

use crate::message_router::{self, AgentJob, JobKind};
use crate::session::{ask_agent_id, direct_agent_id, manager_agent_id};
use crate::tools::Tool;
use crate::{Agent, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
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
                let result = run_sub_agent(&ws, role, &ask).await;
                let message = match result {
                    Ok(text) => format!("<ask-tool-result>\n\n{text}</ask-tool-result>"),
                    Err(e) => {
                        tracing::debug!(error = %e, "async AskTool sub-agent failed");
                        format!("<ask-tool-result>\n\nAn error occurred: {e}</ask-tool-result>")
                    }
                };

                // Route result to the caller's agent channel.
                // Manager callers route to `manager_{ws_name}`.
                // Assistant callers route to `{channel}_{user_name}_{ws_name}_assistant`.
                let target_agent_id = if caller_role == Role::Manager {
                    manager_agent_id(&ws.name)
                } else {
                    direct_agent_id(&channel, &user_name, caller_role.as_str(), &ws.name)
                };
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

/// Shared sub-agent runner — creates and executes a sub-agent for the given
/// role and ask. Used by both sync and async paths of [`AskTool::execute`].
async fn run_sub_agent(ws: &Workspace, role: Role, ask: &str) -> Result<String> {
    let agent_id = ask_agent_id(&ws.name, role.as_str());

    let mut agent = Agent::new(agent_id, role, ws, None, String::new(), String::new());
    let cancel = agent.cancel_token();

    let response = tokio::select! {
        () = cancel.cancelled() => {
            anyhow::bail!("Sub-agent cancelled");
        }
        result = agent.work(ask) => match result {
            Ok(response) => response,
            Err(e) => anyhow::bail!("Sub-agent failed: {e}"),
        },
    };

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use serde_json::json;

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
}
