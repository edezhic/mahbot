//! Sleep tool — end the agent's turn gracefully and go quiet.
//!
//! Called when there is nothing left to do right now and the agent is only
//! waiting for new input (a user message, an async sub-agent result, or a
//! fired alarm). The session is preserved and resumed by the next such event;
//! the tool never waits inside a turn.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::tools::Tool;

pub struct SleepTool;

#[async_trait]
impl Tool for SleepTool {
    fn name(&self) -> &'static str {
        "sleep"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(&json!({}), &[])
    }

    fn ends_turn_on_success(&self) -> bool {
        true
    }

    async fn execute(&self, _ws: &Workspace, _args: serde_json::Value) -> Result<String> {
        Ok("Zzz...".to_string())
    }
}
