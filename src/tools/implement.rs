//! ImplementTool — spawns a single coder sub-agent to carry out a clearly-scoped
//! implementation task. Engineer-only: the coder has full shell/read/edit/search
//! access and mutates the workspace, so this is a side-effecting tool that never
//! runs concurrently with other tools.

use crate::agent::run_agent;
use crate::session::ask_agent_id;
use crate::tools::Tool;
use crate::{Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

pub struct ImplementTool;

#[async_trait]
impl Tool for ImplementTool {
    fn name(&self) -> &'static str {
        "implement"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "task": {
                    "type": "string",
                    "description": "The implementation task to delegate to the coder sub-agent"
                }
            }),
            &["task"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let task = super::get_str(&args, "task")?;

        // Single-coder path — delegate lifecycle to run_agent. Blocks the
        // caller until the coder completes.
        let agent_id = ask_agent_id(&ws.name, Role::Coder.as_str());
        let (agent, response) = run_agent(
            agent_id,
            Role::Coder,
            ws,
            None,
            task,
            String::new(),
            String::new(),
            None,
            false,
        )
        .await;

        if let Some(response) = response {
            Ok(response)
        } else if agent.is_cancelled() || crate::shutdown::aborting() {
            anyhow::bail!("Sub-agent cancelled");
        } else {
            anyhow::bail!(
                "Sub-agent failed: {}",
                agent.failure.as_deref().unwrap_or("unknown error")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use serde_json::json;

    #[tokio::test]
    async fn test_implement_missing_args() {
        let tool = ImplementTool;
        let ws = test_ws("/tmp/test_ws");

        let result = tool.execute(&ws, json!({})).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing required field: task"),
            "Should mention missing task"
        );
    }
}
