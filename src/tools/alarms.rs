//! Alarm/reminder tools — add, list, and remove the Assistant's own reminders.
//!
//! These tools let the Assistant create a reminder that fires a notification
//! back into its conversation when due, list active reminders, and remove one.

use std::fmt::Write as _;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::agent::{CURRENT_TOOL_AGENT_ID, CURRENT_TOOL_USER_NAME};
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
    let user_name = CURRENT_TOOL_USER_NAME
        .try_with(String::clone)
        .unwrap_or_default();
    let agent_id = CURRENT_TOOL_AGENT_ID
        .try_with(Option::clone)
        .ok()
        .flatten()
        .unwrap_or_default();
    anyhow::ensure!(
        !agent_id.is_empty() && !user_name.is_empty(),
        "No agent identity available to associate the alarm — the Assistant must run in a user session."
    );
    Ok(AssistantIdentity {
        agent_id,
        user_name,
    })
}

pub struct AddAlarmTool;

#[async_trait]
impl Tool for AddAlarmTool {
    fn name(&self) -> &'static str {
        "add_alarm"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "text": {
                    "type": "string",
                    "description": "The reminder text to store"
                },
                "fire_at": {
                    "type": "string",
                    "description": "RFC3339/ISO-8601 absolute fire time in UTC (e.g. 2026-08-28T10:30:00Z). Convert user-provided local times to UTC."
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "Periodic interval in seconds (minimum 10). Exactly one of fire_at or interval_seconds must be provided."
                }
            }),
            &["text"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
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
        )
        .await?;
        let display = format_fire_time(&alarm.next_fire_at)?;
        if let Some(interval) = alarm.interval_seconds {
            Ok(format!(
                "Alarm scheduled every {interval} seconds. Next fire at {display}.\nText: {text}"
            ))
        } else {
            Ok(format!("Alarm set for {display}.\nText: {text}"))
        }
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
            let _ = writeln!(
                out,
                "- {}, {}, {}, next fire: {}",
                alarm.id, alarm.kind, alarm.text, display
            );
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
