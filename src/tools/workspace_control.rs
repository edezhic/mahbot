//! Admin-only control of the registered workspaces' pipelines: pause, resume
//! and list.
//!
//! The write is exactly the desktop footer toggle's
//! ([`WorkspaceStore::set_paused`](crate::workspace::WorkspaceStore::set_paused)) —
//! one flag, one in-flight freeze — so a pause started from a chat is
//! indistinguishable from one started in the dashboard. Nothing else is
//! touched: in particular the admin's *active* workspace stays where it is
//! (unlike `mahbot_config`'s `add_workspace`, which switches it).
use anyhow::anyhow;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write as _;

use crate::{Tool, Workspace, WorkspaceStatus};

/// The `workspace_control` tool: pause/resume a named workspace's pipeline and
/// list the registered workspaces with their status and pause state.
pub(crate) struct WorkspaceControlTool;

#[async_trait]
impl Tool for WorkspaceControlTool {
    fn name(&self) -> &'static str {
        "workspace_control"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": ["pause", "resume", "list"],
                    "description": "`pause` / `resume` freeze or lift the pipeline of the workspace named in `name`; `list` reports every registered workspace with its status and whether its pipeline is paused."
                },
                "name": {
                    "type": "string",
                    "description": "(pause / resume) The workspace's registered name, exactly as the `<registered-workspaces>` block spells it — matching is exact, never approximate."
                }
            }),
            &["action"],
        )
    }

    /// Names, paths and states only — nothing here is a credential.
    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        match super::get_str(&args, "action")? {
            "list" => list_workspaces().await,
            "pause" => set_pipeline_paused(super::get_str(&args, "name")?, true).await,
            "resume" => set_pipeline_paused(super::get_str(&args, "name")?, false).await,
            other => Err(anyhow!(
                "unknown action '{other}' — expected one of: pause, resume, list"
            )),
        }
    }
}

/// The registered workspaces — never a personal space, including a legacy
/// `personal:*` row — one line each in the context block's own shape plus the
/// Telegram picker's `, paused` marker. A fresh store read on every call: the
/// context block is a session-start snapshot.
async fn list_workspaces() -> anyhow::Result<String> {
    let workspaces = crate::users::registered_workspaces().await?;
    if workspaces.is_empty() {
        return Ok("No workspace is registered.".to_string());
    }
    let mut listing = String::from("Registered workspaces:\n");
    for ws in workspaces {
        let paused = if ws.paused { ", paused" } else { "" };
        let _ = writeln!(
            listing,
            "- {} ({}{paused}): {}",
            ws.name, ws.status, ws.path
        );
    }
    Ok(listing.trim_end().to_string())
}

/// Pause or resume the named workspace's pipeline.
///
/// Refused unless `name` names a registered workspace that is `ready`: only a
/// `ready` row is dispatchable (the poll gates every claim and phase dispatch on
/// the status), so a non-ready row's `paused` flag is the discovery's own
/// analysis pause rather than a freeze of running work, and the request is
/// refused with the status instead of being written into a flag the admin does
/// not own.
async fn set_pipeline_paused(name: &str, paused: bool) -> anyhow::Result<String> {
    let name = name.trim();
    if crate::users::is_personal_workspace(name) {
        return Err(anyhow!(
            "'{name}' is a personal space, not a registered workspace — personal spaces have no pipeline."
        ));
    }
    let Some(ws) = crate::workspace::get_by_name(name).await? else {
        // Name the live set in the refusal: the admin names workspaces from a
        // session-start context block, which may predate an add or a delete.
        let known = crate::users::registered_workspaces()
            .await?
            .into_iter()
            .map(|ws| ws.name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow!(
            "No workspace is registered under the name '{name}' — registered: {}",
            if known.is_empty() { "none" } else { &known }
        ));
    };
    if ws.status != WorkspaceStatus::Ready {
        return Err(anyhow!(
            "Workspace '{}' is {} — only a ready workspace has a pipeline to pause or resume.",
            ws.name,
            ws.status
        ));
    }
    // A request that matches the state it is already in is reported as such
    // rather than as a freeze lifted or applied — and writes nothing.
    if ws.paused == paused {
        return Ok(if paused {
            format!("The pipeline of workspace '{}' is already paused.", ws.name)
        } else {
            format!(
                "The pipeline of workspace '{}' is not paused — nothing to resume.",
                ws.name
            )
        });
    }
    crate::workspace::store()
        .set_paused(&ws.name, paused)
        .await?;
    let verb = if paused { "paused" } else { "resumed" };
    Ok(format!("Pipeline {verb} for workspace '{}'.", ws.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_args(action: &str, name: Option<&str>) -> serde_json::Value {
        match name {
            Some(name) => json!({ "action": action, "name": name }),
            None => json!({ "action": action }),
        }
    }

    /// The refusals that keep the tool from reporting a change it did not make.
    #[tokio::test]
    async fn refusals_are_plain_and_named() {
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/wsctl_refusals");
        let tool = WorkspaceControlTool;

        let err = tool
            .execute(&ws, tool_args("pause", Some("wsctl_absent")))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("No workspace is registered"), "got: {err}");

        let err = tool
            .execute(&ws, tool_args("resume", Some("personal:admin")))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("personal space"), "got: {err}");

        // A freshly registered workspace is `pending` — it has no pipeline yet,
        // so nothing may be frozen or resumed.
        crate::util::test::create_test_workspace("/tmp/wsctl_pending", "wsctl_pending").await;
        let err = tool
            .execute(&ws, tool_args("pause", Some("wsctl_pending")))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("pending") && err.contains("ready"),
            "got: {err}"
        );
        assert!(
            !crate::workspace::get_by_name("wsctl_pending")
                .await
                .expect("read")
                .expect("exists")
                .paused,
            "the refusal must not pause the workspace"
        );
    }

    #[tokio::test]
    async fn pause_and_resume_a_ready_workspace() {
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/wsctl_ready");
        let tool = WorkspaceControlTool;
        crate::util::test::create_test_workspace("/tmp/wsctl_ready", "wsctl_ready").await;
        crate::workspace::store()
            .set_status("wsctl_ready", &WorkspaceStatus::Ready)
            .await
            .expect("mark ready");

        let reply = tool
            .execute(&ws, tool_args("pause", Some("wsctl_ready")))
            .await
            .expect("pause");
        assert!(
            reply.contains("wsctl_ready") && reply.contains("paused"),
            "got: {reply}"
        );
        assert!(
            crate::workspace::get_by_name("wsctl_ready")
                .await
                .expect("read")
                .expect("exists")
                .paused
        );

        let listing = tool
            .execute(&ws, tool_args("list", None))
            .await
            .expect("list");
        assert!(
            listing.contains("- wsctl_ready (ready, paused)"),
            "got: {listing}"
        );

        // Re-pausing reports the state it is already in, never a second freeze.
        let reply = tool
            .execute(&ws, tool_args("pause", Some("wsctl_ready")))
            .await
            .expect("re-pause");
        assert!(reply.contains("already paused"), "got: {reply}");

        let reply = tool
            .execute(&ws, tool_args("resume", Some("wsctl_ready")))
            .await
            .expect("resume");
        assert!(reply.contains("resumed"), "got: {reply}");
        assert!(
            !crate::workspace::get_by_name("wsctl_ready")
                .await
                .expect("read")
                .expect("exists")
                .paused
        );
        assert!(
            !tool
                .execute(&ws, tool_args("list", None))
                .await
                .expect("list")
                .contains("- wsctl_ready (ready, paused)"),
            "a resumed workspace must not carry the pause marker"
        );
    }
}
