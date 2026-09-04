//! Alarm/reminder tools — add, list, and remove the Assistant's own reminders.
//!
//! These tools let the Assistant create a reminder that fires a notification
//! back into its conversation when due, list active reminders, and remove one.

use std::fmt::Write as _;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::alarms::{add_alarm, format_fire_time, list_alarms, remove_alarm};
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

/// The `add_alarm` tool. `admin` (full-access Assistant) unlocks the
/// optional `command` parameter — command-armed alarms that wake the
/// Assistant only when the command produces output or fails.
pub struct AddAlarmTool {
    admin: bool,
}

impl AddAlarmTool {
    pub(crate) const fn new(admin: bool) -> Self {
        Self { admin }
    }
}

#[async_trait]
impl Tool for AddAlarmTool {
    fn name(&self) -> &'static str {
        "add_alarm"
    }

    fn description(&self) -> String {
        let base = crate::prompt::load_prompt("tool/add_alarm.md");
        if self.admin {
            format!(
                "{base}\n\n{}",
                crate::prompt::load_prompt("tool/add_alarm_command.md")
            )
        } else {
            base
        }
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
                "description": "Periodic interval in seconds (minimum 10). Exactly one of fire_at or interval_seconds must be provided."
            }),
        );
        if self.admin {
            properties.insert(
                "command".to_string(),
                json!({
                    "type": "string",
                    "description": "Shell command run at fire time in your personal workspace. The reminder wakes you only when the command produces output or fails; a successful command with empty output stays silent. Maximum 2000 characters."
                }),
            );
        }
        super::tool_params_schema(&json!(properties), &["text"])
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        // Reject BEFORE reading anything else: a non-admin must never get a
        // silently-degraded plain alarm when it asked for a command-armed one.
        // Explicit null is treated as absent, matching the admin path below.
        if !self.admin && args.get("command").is_some_and(|v| !v.is_null()) {
            anyhow::bail!(
                "The `command` parameter is only available in full-access (admin) sessions."
            );
        }
        // An admin passing a non-string `command` is a caller error, never a
        // silent plain alarm.
        let command = match args.get("command") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.as_str()),
            Some(_) => anyhow::bail!("The `command` parameter must be a string."),
        };
        let ident = identity()?;
        let text = super::get_str(&args, "text")?;
        let fire_at = super::get_opt_str(&args, "fire_at");
        let interval_seconds = super::get_opt_u64(&args, "interval_seconds");

        let alarm = add_alarm(
            &ident.agent_id,
            &ident.user_name,
            text,
            fire_at,
            interval_seconds,
            command,
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
        if let Some(cmd) = command {
            let _ = write!(base, "\nCommand: {cmd}");
        }
        Ok(base)
    }
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

    async fn execute(&self, _ws: &Workspace, _args: serde_json::Value) -> Result<String> {
        let session_id = identity()?.agent_id;
        let alarms = list_alarms(&session_id).await?;
        if alarms.is_empty() {
            return Ok("(no active alarms)".to_string());
        }
        let mut out = String::new();
        for alarm in alarms {
            let display = format_fire_time(&alarm.next_fire_at)?;
            match alarm.command {
                Some(cmd) => {
                    let _ = writeln!(
                        out,
                        "- {}, {}, {}, next fire: {}, command: {}",
                        alarm.id, alarm.kind, alarm.text, display, cmd
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
