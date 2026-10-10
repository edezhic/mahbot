//! Launch tool — start a custom role with a message.
//!
//! A custom role is an administrator-defined agent (name, authorial prompt,
//! tool subset, behaviour). This tool starts one, delivering `message` to it as
//! the role's own user message; the role's final answer is delivered to the
//! admin's assistant conversation automatically, naming the role and the folder
//! it ran in.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::tools::Tool;

/// The `launch_role` tool — start an existing custom role.
pub struct LaunchRoleTool;

#[async_trait]
impl Tool for LaunchRoleTool {
    fn name(&self) -> &'static str {
        "launch_role"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "name": {
                    "type": "string",
                    "description": "The custom role's name."
                },
                "message": {
                    "type": "string",
                    "description": "What the role is asked to work on. It is delivered to the role as an ordinary user message."
                },
                "identifier": {
                    "type": "string",
                    "description": "Optional branch identifier (letters, digits, `_`, `-`): with it the launch continues that named branch of the role and its history is restored; without it the role's own default behaviour applies."
                },
                "workspace": {
                    "type": "string",
                    "description": "Optional folder the role works in: the exact name of a registered workspace, or an absolute path to any folder on the machine. Omit to run in your own personal folder."
                }
            }),
            &["name", "message"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        // The tool validates nothing itself: whether the role exists, whether
        // its name/identifier are well-formed and whether the workspace
        // resolves are all decided in `custom_role::launch` — including the
        // trim-and-empty-as-absent normalization of both optional arguments.
        // Whether the caller may launch at all is the boundary of the tool set —
        // a role only holds this tool if the administrator granted it.
        crate::custom_role::launch(
            super::get_str(&args, "name")?,
            super::get_str(&args, "message")?,
            super::get_opt_str(&args, "identifier"),
            super::get_opt_str(&args, "workspace"),
        )
        .await
    }
}
