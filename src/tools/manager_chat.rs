//! Assistant→Manager communication tool — send a message to a workspace
//! Manager.
//!
//! The Assistant addresses the Manager on the user's behalf via an internal
//! agent message (wrapped in an `<assistant-message>` envelope). The send is
//! fully internal: nothing reaches the workspace users' chat history or
//! channel bindings. The tool is gated to the full-access Assistant (see
//! `Role::Assistant` toolset).

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::tools::Tool;

/// The `send_message_to_manager` tool: deliver an internal message to the
/// Manager agent of a project workspace.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::UnwrapPoison;
    use std::sync::Arc;

    async fn chat_history_rows(workspace: &str) -> i64 {
        crate::session::store()
            .conn
            .query_row(
                "SELECT COUNT(*) FROM chat_history WHERE workspace = ?1",
                crate::db::params![workspace],
                |row| row.get::<i64>(0),
            )
            .await
            .unwrap()
    }

    /// Regression: the send must stay fully internal — the
    /// `<assistant-message>` envelope still routes to the workspace Manager
    /// (live route + durable pending_jobs copy), while NOTHING is mirrored
    /// into the workspace users' chat_history or their channel bindings.
    /// The admin user is attached to the workspace with a live spy-channel
    /// binding — exactly the setup the removed mirroring delivered to.
    #[tokio::test]
    #[serial_test::serial(channel_registry, drain)] // serializes the process-global channel registry + shutdown drain flag
    async fn manager_send_stays_internal() {
        crate::util::test::init_management_test_stores().await;

        let ws =
            crate::util::test::create_test_workspace("/tmp/mahbot/ws_mirror", "ws_mirror").await;

        // An admin workspace member bound to a spy channel — the removed
        // mirroring would have persisted + transport-delivered for exactly
        // this user.
        let store = crate::users::store();
        store
            .add_user("mirror_user", Some("full"), crate::Role::Assistant)
            .await
            .unwrap();
        store
            .update_user(
                "mirror_user",
                crate::users::FieldUpdate::Unchanged,
                crate::users::FieldUpdate::Set(&ws.name),
                crate::users::FieldUpdate::Unchanged,
            )
            .await
            .unwrap();
        store
            .bind_channel("mirror_user", "spy", "mirror_user")
            .await
            .unwrap();

        let (spy, sent) = crate::util::test::SpyChannel::new("spy");
        let registry = crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
        registry.register(Arc::new(spy) as Arc<dyn crate::Channel>);

        // Register the Manager consumer so the routed envelope lands here
        // instead of spawning a live consumer loop.
        let target = crate::session::manager_agent_id(&ws.name);
        let mut rx = crate::agent::message_router::register_agent(&target);

        let before = chat_history_rows(&ws.name).await;

        let res = crate::agent::CURRENT_TOOL_USER_NAME
            .scope("mirror_user".to_string(), async {
                crate::agent::CURRENT_TOOL_AGENT_ID
                    .scope(Some("agent_mirror".to_string()), async {
                        SendMessageToManagerTool
                            .execute(
                                &ws,
                                serde_json::json!({"workspace": &ws.name, "message": "delegated work"}),
                            )
                            .await
                    })
                    .await
            })
            .await;
        assert!(res.is_ok(), "tool must succeed: {:#}", res.unwrap_err());

        // The envelope reached the Manager consumer. Bounded because
        // `stamp_and_route` skips the live route while the drain flag is set (see
        // the drain serialization on this test): an unrouted envelope must fail
        // here instead of hanging the whole test binary.
        let job = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("the manager envelope must be routed within 30s")
            .expect("envelope must reach the manager");
        assert_eq!(
            job.kind,
            crate::agent::message_router::MessageKind::AgentMessage
        );
        assert_eq!(job.workspace_name, ws.name);
        assert!(job.content.contains("delegated work"));

        // A durable pending_jobs copy exists for boot resume.
        let pending = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .unwrap();
        assert!(
            pending
                .iter()
                .any(|p| p.target_agent_id == target && p.envelope.contains("delegated work")),
            "durable envelope copy expected in pending_jobs"
        );

        // Nothing mirrored into chat history or channel bindings.
        assert_eq!(chat_history_rows(&ws.name).await, before);
        assert!(
            sent.lock().unwrap_poison().is_empty(),
            "no channel delivery expected"
        );
    }
}
