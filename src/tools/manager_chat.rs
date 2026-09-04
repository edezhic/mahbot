//! Assistant↔Manager communication tools — send a message to a workspace
//! Manager and read a workspace's user-facing manager chat.
//!
//! These bridge the user-facing Assistant session and the workspace Manager:
//! the Assistant addresses the Manager on the user's behalf via an internal
//! agent message (wrapped in an `<assistant-message>` envelope), and can read
//! back the manager chat that the workspace users see. The send tool mirrors
//! the message into each workspace user's chat AND their channel bindings,
//! symmetric with the Manager broadcast. The tools are gated to the
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
                },
                "wait": {
                    "type": "boolean",
                    "description": "When true, end your turn after sending and stay quiet until the Manager's reply arrives."
                }
            }),
            &["workspace", "message"],
        )
    }

    /// The `wait` flag is the only turn-end trigger — a plain send must not end
    /// the turn (so `ends_turn_on_success` stays `false`).
    fn ends_turn_for_args(&self, args: &serde_json::Value) -> bool {
        super::get_bool(args, "wait", false)
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let (agent_id, user_name) = crate::agent::tool_identity()?;

        let workspace = super::get_str(&args, "workspace")?;
        let message = super::get_str(&args, "message")?;
        let wait = super::get_bool(&args, "wait", false);

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
            ws.name.clone(),
            user_name,
            agent_id,
        )
        .await;

        if wait {
            Ok(format!(
                "Message delivered to the manager of workspace '{workspace}' — going quiet until the reply arrives."
            ))
        } else {
            Ok(format!(
                "Message delivered to the manager of workspace '{workspace}'."
            ))
        }
    }
}

/// The `read_manager_chat` tool: read the recent user-facing chat of a
/// workspace's manager chat in chronological order.
pub struct ReadManagerChatTool;

#[async_trait]
impl Tool for ReadManagerChatTool {
    fn name(&self) -> &'static str {
        "read_manager_chat"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "workspace": {
                    "type": "string",
                    "description": "Name of the workspace whose chat to read."
                },
                "limit": {
                    "type": "integer",
                    "description": "How many recent messages to return.",
                    "minimum": 1,
                    "maximum": 50,
                }
            }),
            &["workspace"],
        )
    }

    /// Read-only — safe to run in parallel with other tools.
    fn side_effects(&self) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let workspace = super::get_str(&args, "workspace")?;

        if crate::workspace::get_by_name(workspace).await?.is_none() {
            anyhow::bail!(
                "Workspace '{workspace}' does not exist or is a personal workspace — there is no Manager chat there."
            );
        }

        let limit = super::get_opt_u64(&args, "limit").unwrap_or(5).clamp(1, 50);
        // The loader grows its fetch window on shortfall (see
        // `load_workspace_stream`), so the caller just requests the desired
        // message count — no user-count multiplication needed here.
        let rows = crate::channels::chat_history::store()
            .load_workspace_stream(workspace, limit as usize)
            .await?;

        if rows.is_empty() {
            return Ok(format!(
                "No messages yet in the '{workspace}' manager chat."
            ));
        }

        let lines: Vec<String> = rows
            .into_iter()
            .map(|row| {
                if row.direction == crate::ChatDirection::Agent {
                    format!(
                        "[{}]: {}",
                        row.agent_role.as_deref().unwrap_or("agent"),
                        row.content
                    )
                } else {
                    format!("{}: {}", row.user_name, row.content)
                }
            })
            .collect();

        Ok(lines.join("\n\n"))
    }
}
