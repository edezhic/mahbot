//! AskTool — spawns a sub-agent to ask a question.
//!
//! Available to the Engineer and Maintainer agents (sync mode), and to the
//! Manager agent (async mode). In sync mode the caller blocks until the
//! sub-agent completes. In async mode the sub-agent is dispatched in a
//! background task and the result is injected back into the Manager's session
//! via the serialized [`ManagerQueue`](crate::manager_queue::ManagerQueue).

use crate::manager_queue::{self, JobKind, ManagerJob};
use crate::session::ask_session_key;
use crate::tools::Tool;
use crate::{Agent, Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

pub struct AskTool {
    pub allowed_roles: Vec<Role>,
    /// When `Some`, dispatches the sub-agent asynchronously and injects
    /// the result into the Manager's session via the serialized Manager queue.
    /// When `None` (default), blocks the caller until the sub-agent completes.
    /// Only set by the Manager role — Engineer and Maintainer pass `None`.
    caller_agent_id: Option<String>,
}

impl AskTool {
    #[must_use]
    pub const fn new(allowed_roles: Vec<Role>, caller_agent_id: Option<String>) -> Self {
        Self {
            allowed_roles,
            caller_agent_id,
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
    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let role_str = super::get_str(&args, "role")?;
        let ask = super::get_str(&args, "ask")?;

        let role: Role = match role_str.parse() {
            Ok(r) if self.allowed_roles.contains(&r) => r,
            Ok(_) => {
                let allowed_str = self.formatted_allowed_roles();
                anyhow::bail!("Cannot delegate to '{role_str}'. Only {allowed_str} are supported.");
            }
            Err(_) => {
                let allowed_str = self.formatted_allowed_roles();
                anyhow::bail!("Unknown role '{role_str}'. Use {allowed_str}.");
            }
        };

        // Async dispatch path — Manager delegates to Analyst in background.
        // Routing is handled by the consumer loop via DB lookup — no reply
        // context needed in the job.
        if self.caller_agent_id.is_some() {
            let ws = ws.clone();
            let ask = ask.to_string();

            tokio::spawn(async move {
                let result = run_sub_agent(&ws, role, &ask).await;
                let message = match result {
                    Ok(text) => format!("<ask-tool-result>\n\n{text}</ask-tool-result>"),
                    Err(e) => {
                        tracing::debug!(error = %e, "async AskTool sub-agent failed");
                        format!("<ask-tool-result>\n\nAn error occurred: {e}</ask-tool-result>")
                    }
                };

                manager_queue::manager_queue().enqueue(ManagerJob {
                    content: message,
                    workspace_name: ws.name.clone(),
                    kind: JobKind::AskToolResult,
                });
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
    let sub_id = ask_session_key(&ws.name, role.as_str());

    let mut agent = Agent::new(sub_id, role, ws, None);
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
        let tool = AskTool::new(vec![Role::Analyst, Role::Coder, Role::Qa], None);
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
        let tool = AskTool::new(vec![Role::Analyst], None);
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
        let tool = AskTool::new(vec![], None);
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
            let tool = AskTool::new(vec![Role::Analyst, Role::Coder, Role::Qa], None);
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
