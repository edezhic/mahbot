//! Assistant→Manager communication tool — send a message to a workspace
//! Manager.
//!
//! The Assistant addresses the Manager on the user's behalf via an internal
//! agent message (wrapped in an `<assistant-message>` envelope). The send tool
//! mirrors the message into each workspace user's chat AND their channel
//! bindings, symmetric with the Manager broadcast. The tool is gated to the
//! full-access Assistant (see `Role::Assistant` toolset).

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::tools::Tool;

/// The `send_message_to_manager` tool: deliver a message to the Manager agent
/// of a project workspace, and mirror it into the workspace users' chat.
pub struct SendMessageToManagerTool;

#[async_trait]
impl Tool for SendMessageToManagerTool {
    fn name(&self) -> &'static str {
        "send_message_to_manager"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "workspace": {
                    "type": "string",
                    "description": "Name of the target workspace. Only project/shared workspaces have a Manager — personal workspaces cannot be targeted."
                },
                "message": {
                    "type": "string",
                    "description": "The text to deliver. Make it self-contained: the Manager does not see your conversation with the user."
                }
            }),
            &["workspace", "message"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let (_, user_name) = crate::agent::tool_identity()?;

        let workspace = super::get_str(&args, "workspace")?;
        let message = super::get_str(&args, "message")?;

        // Only project/shared workspaces are registered — personal workspaces
        // are synthesized on the fly and have no Manager to address.
        if crate::workspace::get_by_name(workspace).await?.is_none() {
            anyhow::bail!(
                "Workspace '{workspace}' does not exist or is a personal workspace — there is no Manager to message there."
            );
        }

        // Persist the RAW message to each workspace user's chat for visibility
        // (symmetric with the Manager broadcast), attributed as the Assistant,
        // then transport-deliver it (shared broadcast id — the workspace chat
        // stream dedupes the per-user copies). If no workspace users exist,
        // skip silently — the Manager job still routes.
        let users = match crate::users::USER_STORE.get() {
            Some(store) => store.find_by_workspace(workspace).await.unwrap_or_default(),
            None => Vec::new(),
        };
        crate::agent::message_router::deliver_agent_response_to_workspace(
            message,
            &users,
            crate::Role::Assistant,
            workspace,
        )
        .await;

        // Envelope wrapping happens only on the routed Manager-bound job.
        let envelope = crate::prompt::substitute(
            &crate::prompt::load_prompt("assistant_message.md"),
            &[
                ("{{user_name}}", user_name.as_str()),
                ("{{message}}", message),
            ],
        );

        crate::agent::message_router::route_agent_message_to_manager(
            envelope,
            workspace.to_string(),
            user_name,
        )
        .await;

        Ok(format!(
            "Message delivered to the manager of workspace '{workspace}'."
        ))
    }
}
