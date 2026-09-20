//! Alarm/reminder tools — add, list, and remove the Assistant's own reminders.
//!
//! These tools let the Assistant create a reminder that fires a notification
//! back into its conversation when due, list active reminders, and remove one.
//!
//! What they return is a tool result like any other, so it keeps the
//! product-wide agent-level credential scrub (see [`crate::alarms`]).

use std::fmt::Write as _;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::alarms::{Trigger, add_alarm, format_fire_time, list_alarms, remove_alarm};
use crate::tools::Tool;

/// The calling Assistant's personal-session identity: the agent id plus the
/// raw user name (for storing the alarm owner and routing the fired reminder).
struct AssistantIdentity {
    agent_id: String,
    user_name: String,
}

/// Read the calling Assistant's identity from the tool task-locals, or bail if
/// there is no user/agent context.
fn identity() -> Result<AssistantIdentity> {
    let (agent_id, user_name) = crate::agent::tool_identity()?;
    Ok(AssistantIdentity {
        agent_id,
        user_name,
    })
}

/// The `add_alarm` tool. An alarm may carry a `trigger`: one custom tool
/// available to its owner plus that tool's arguments, run at fire time so the
/// Assistant is woken only when the check reports something.
pub struct AddAlarmTool;

#[async_trait]
impl Tool for AddAlarmTool {
    fn name(&self) -> &'static str {
        "add_alarm"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        properties.insert(
            "text".to_string(),
            json!({
                "type": "string",
                "description": "The reminder text to store"
            }),
        );
        properties.insert(
            "fire_at".to_string(),
            json!({
                "type": "string",
                "description": "RFC3339/ISO-8601 absolute fire time in UTC (e.g. 2026-08-28T10:30:00Z). Convert user-provided local times to UTC."
            }),
        );
        properties.insert(
            "interval_seconds".to_string(),
            json!({
                "type": "integer",
                "description": "Periodic interval in seconds (minimum 5). Exactly one of fire_at or interval_seconds must be provided."
            }),
        );
        properties.insert(
            "trigger".to_string(),
            json!({
                "type": "object",
                "description": "Optional check to run at fire time: a custom tool available to you and the arguments it receives. Wakes you only when the check reports something, and removes the alarm when the check cannot run cleanly.",
                "properties": {
                    "tool": { "type": "string", "description": "Name of the custom tool, as listed in the <custom-tools> block" },
                    "args": { "type": "object", "description": "The tool's arguments, keyed by parameter name; only the parameters it declares are accepted" }
                },
                "required": ["tool"]
            }),
        );
        super::tool_params_schema(&json!(properties), &["text"])
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        // The shell-command arm was removed: refuse the parameter rather than
        // quietly arming a plain reminder. An explicit null counts as not passing
        // it, as elsewhere in the product; any other value is refused.
        if args.get("command").is_some_and(|v| !v.is_null()) {
            anyhow::bail!(
                "forbidden: the `command` parameter was removed — hint: a trigger names a \
                 custom tool; pass `trigger` with the tool's name and its arguments"
            );
        }
        let ident = identity()?;
        let text = super::get_str(&args, "text")?;
        let fire_at = super::get_opt_str(&args, "fire_at");
        let interval_seconds = super::get_opt_u64(&args, "interval_seconds")?;
        let trigger = read_trigger(&args, &ident.user_name).await?;
        let alarm = add_alarm(
            &ident.agent_id,
            &ident.user_name,
            text,
            fire_at,
            interval_seconds,
            trigger,
        )
        .await?;
        let display = format_fire_time(&alarm.next_fire_at)?;
        let mut base = if let Some(interval) = alarm.interval_seconds {
            format!(
                "Alarm scheduled every {interval} seconds. Next fire at {display}.\nText: {text}"
            )
        } else {
            format!("Alarm set for {display}.\nText: {text}")
        };
        if let Some(trigger) = &alarm.trigger {
            let _ = write!(base, "\nTrigger: {}", trigger.render());
        }
        Ok(base)
    }
}

/// Read and validate the optional `trigger` parameter for `caller`: the named
/// custom tool must be one the caller can call and the stored arguments must
/// fit its declared interface exactly — an argument the tool does not declare
/// is refused here rather than stored, because an alarm that sits there passing
/// arguments which do nothing is exactly the dead state this exists to avoid.
async fn read_trigger(args: &serde_json::Value, caller: &str) -> Result<Option<Trigger>> {
    let trigger = match args.get("trigger") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(v @ serde_json::Value::Object(_)) => v,
        Some(other) => return Err(super::wrong_type("trigger", "an object", other)),
    };
    let tool = super::get_str(trigger, "tool")?;
    let supplied = super::get_object(trigger, "args")?;
    let resolved = crate::tools::custom::resolve_tool_call(caller, tool, &supplied, true)
        .await
        .map_err(crate::tools::custom::CallRefusal::into_error)?;
    Ok(Some(Trigger {
        tool: tool.to_string(),
        args: resolved.args,
    }))
}

pub struct ListAlarmsTool;

#[async_trait]
impl Tool for ListAlarmsTool {
    fn name(&self) -> &'static str {
        "list_alarms"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(&json!({}), &[])
    }

    fn side_effects(&self) -> bool {
        false // read-only listing
    }

    fn preserve_full_output(&self) -> bool {
        // The Assistant asks to see its alarms, so the whole listing must reach
        // it: sandwich-truncating would drop every alarm between the head and
        // the tail, invisibly to both the Assistant and the user.
        true
    }

    async fn execute(&self, _ws: &Workspace, _args: serde_json::Value) -> Result<String> {
        let session_id = identity()?.agent_id;
        let alarms = list_alarms(&session_id).await?;
        if alarms.is_empty() {
            return Ok("(no active alarms)".to_string());
        }
        let mut out = String::new();
        for alarm in alarms {
            let display = format_fire_time(&alarm.next_fire_at)?;
            match &alarm.trigger {
                Some(trigger) => {
                    let _ = writeln!(
                        out,
                        "- {}, {}, {}, next fire: {}, trigger: {}",
                        alarm.id,
                        alarm.kind,
                        alarm.text,
                        display,
                        trigger.render()
                    );
                }
                None => {
                    let _ = writeln!(
                        out,
                        "- {}, {}, {}, next fire: {}",
                        alarm.id, alarm.kind, alarm.text, display
                    );
                }
            }
        }
        Ok(out.trim().to_string())
    }
}

pub struct RemoveAlarmTool;

#[async_trait]
impl Tool for RemoveAlarmTool {
    fn name(&self) -> &'static str {
        "remove_alarm"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "id": {
                    "type": "string",
                    "description": "The alarm id to remove/stop"
                }
            }),
            &["id"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let session_id = identity()?.agent_id;
        let id = super::get_str(&args, "id")?;
        match remove_alarm(&session_id, id).await? {
            Some(alarm) => {
                let display = format_fire_time(&alarm.next_fire_at)?;
                Ok(format!("Removed alarm {id} (was set for {display})."))
            }
            None => Ok("No active alarm with that id.".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::ProbeFile;

    /// Arm an alarm as `user`, under the agent identity a live call carries.
    async fn arm(user: &str, args: serde_json::Value) -> Result<String> {
        let ws = crate::workspace::test_ws("arming-probe");
        crate::agent::CURRENT_TOOL_AGENT_ID
            .scope(Some(format!("assistant:{user}")), async {
                crate::agent::CURRENT_TOOL_USER_NAME
                    .scope(user.to_string(), AddAlarmTool.execute(&ws, args))
                    .await
            })
            .await
    }

    /// The arming gate: the removed `command` parameter is refused for any
    /// value (an explicit null counts as not passing it), a tool the caller
    /// cannot call is refused before anything is looked up, and an argument the
    /// named tool does not declare is refused rather than stored.
    #[tokio::test]
    async fn arming_refuses_the_removed_command_an_unavailable_tool_and_undeclared_arguments() {
        crate::util::test::init_management_test_stores().await;
        let (text, fire) = ("check the city", "2099-01-01T00:00:00Z");

        for command in [json!("echo hi"), json!("")] {
            let err = arm(
                "admin",
                json!({ "text": text, "fire_at": fire, "command": command }),
            )
            .await
            .unwrap_err()
            .to_string();
            assert!(err.contains("was removed"), "got: {err}");
        }
        // An explicit null is not passing it, so a plain reminder arms.
        arm(
            "admin",
            json!({ "text": text, "fire_at": fire, "command": null }),
        )
        .await
        .expect("a null command is not a command");

        // A guest with no grants cannot name anything...
        let err = arm(
            "arming_guest",
            json!({ "text": text, "fire_at": fire, "trigger": { "tool": "arming_probe" } }),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("is not granted to you"), "got: {err}");

        // ...and an argument the nominated tool does not declare is refused,
        // rather than stored to do nothing.
        let dir =
            crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME).join("shared");
        std::fs::create_dir_all(&dir).expect("create the shared folder");
        let probe = ProbeFile(dir.join("arming_probe.ts"));
        std::fs::write(
            &probe.0,
            "// @description Probe.\n// @param city string required the city\n",
        )
        .expect("write the probe tool");
        let err = arm(
            "admin",
            json!({
                "text": text,
                "fire_at": fire,
                "trigger": { "tool": "arming_probe", "args": { "extra": 1 } },
            }),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.starts_with("usage: "), "got: {err}");
        assert!(err.contains("[ignored arguments: extra]"), "got: {err}");

        // The declared arguments arm, and the confirmation shows the trigger.
        let arming = json!({
            "text": text,
            "fire_at": fire,
            "trigger": { "tool": "arming_probe", "args": { "city": "Minsk" } },
        });
        let out = arm("admin", arming.clone())
            .await
            .expect("a caller with the tool arms its trigger");
        assert!(
            out.contains("Trigger: arming_probe {\"city\":\"Minsk\"}"),
            "got: {out}"
        );

        // A grant is what makes the same name reachable for a guest.
        let store = crate::users::USER_STORE.get().expect("user store");
        store.add_user("arming_grantee").await.unwrap();
        store
            .add_grant("arming_grantee", "arming_probe")
            .await
            .unwrap();
        arm("arming_grantee", arming)
            .await
            .expect("a guest granted the tool arms its trigger");
    }
}
