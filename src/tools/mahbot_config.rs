//! Merged admin config tool — one `mahbot_config` action dispatcher for the
//! five setup actions previously exposed as separate Support tools
//! (`setup_telegram_bot`, `bind_telegram`, `add_workspace`, `add_user`,
//! `setup_web_search`). It is now the full-access Assistant's sole
//! configuration surface (the Support role is gone).
use crate::config::{
    CONFIG_KEY_EXA_KEY, CONFIG_KEY_FIRECRAWL_KEY, CONFIG_KEY_TELEGRAM_BOT_TOKEN,
    CONFIG_KEY_WEB_SEARCH_PROVIDER,
};
use crate::users::FieldUpdate;
use crate::{Role, Tool, Workspace};
use anyhow::{Context, anyhow};
use async_trait::async_trait;
use serde_json::json;

/// The user configuration operates as, derived from the personal workspace it
/// runs in (`personal:<user>`). The full-access Assistant always operates in a
/// personal workspace, so this resolves to the admin without a separate
/// identity lookup.
fn acting_user(ws: &Workspace) -> &str {
    crate::users::personal_user_name(&ws.name).unwrap_or("admin")
}

fn err(msg: impl Into<String>) -> anyhow::Error {
    anyhow!(msg.into())
}

/// Dispatch a single configuration action for the admin user.
pub(crate) struct MahbotConfigTool;

#[async_trait]
impl Tool for MahbotConfigTool {
    fn name(&self) -> &'static str {
        "mahbot_config"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": [
                        "setup_telegram_bot",
                        "bind_telegram",
                        "add_workspace",
                        "add_user",
                        "setup_web_search",
                    ],
                    "description": "Which configuration action to perform."
                },
                "token": {
                    "type": "string",
                    "description": "(setup_telegram_bot) The Telegram bot token from BotFather (the `NNN:AAA...` string)."
                },
                "handle": {
                    "type": "string",
                    "description": "(bind_telegram) The admin's Telegram @username (with or without the leading @)."
                },
                "name": {
                    "type": "string",
                    "description": "(add_workspace / add_user) A short unique name for the workspace (used in ticket ids and the GUI), or the new user's display name."
                },
                "path": {
                    "type": "string",
                    "description": "(add_workspace) Absolute path to the project directory to manage."
                },
                "telegram": {
                    "type": "string",
                    "description": "(add_user) The new user's Telegram @username (with or without the leading @)."
                },
                "default_agent": {
                    "type": "string",
                    "description": "(add_user) The default agent for this user: 'assistant'."
                },
                "provider": {
                    "type": "string",
                    "description": "(setup_web_search) The web-search provider: 'firecrawl' or 'exa'."
                },
                "key": {
                    "type": "string",
                    "description": "(setup_web_search) The API key for the chosen provider."
                }
            }),
            &["action"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let action = super::get_str(&args, "action")?;
        match action {
            "setup_telegram_bot" => self.exec_setup_telegram_bot(args).await,
            "bind_telegram" => self.exec_bind_telegram(args).await,
            "add_workspace" => self.exec_add_workspace(ws, args).await,
            "add_user" => self.exec_add_user(args).await,
            "setup_web_search" => self.exec_setup_web_search(args).await,
            other => Err(err(format!(
                "unknown action '{other}' — expected one of: setup_telegram_bot, bind_telegram, \
                 add_workspace, add_user, setup_web_search"
            ))),
        }
    }
}

impl MahbotConfigTool {
    async fn exec_setup_telegram_bot(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let token = super::get_str(&args, "token")?;
        crate::config::persist_settled_string_field(CONFIG_KEY_TELEGRAM_BOT_TOKEN, token).await?;
        Ok(
            "Telegram bot token saved — the Telegram listener hot-reloads it immediately. \
             Next, send `/start` to your bot in Telegram, then use `bind_telegram` to bind \
             your @username so messages are routed to you."
                .to_string(),
        )
    }

    async fn exec_bind_telegram(&self, args: serde_json::Value) -> anyhow::Result<String> {
        // bind_telegram always targets the `admin` user: this is a single-admin
        // model, and the config agent only runs for the admin, so the personal
        // workspace it operates in resolves to the same identity. Never the
        // operating full-permissions user.
        let user = "admin";
        let handle = super::get_str(&args, "handle")?;

        let store = crate::users::store();
        let handle = store.validate_telegram_bind(user, handle).await?;

        store.bind_channel(user, "telegram", &handle).await?;
        store
            .update_channel_contact("telegram", &handle, &handle)
            .await?;
        Ok(format!(
            "Bound @{handle} to your account. Messages sent to the bot from that @username \
             will now be routed to you."
        ))
    }

    async fn exec_add_workspace(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        let path = super::get_str(&args, "path")?;

        let store = crate::workspace::store();
        store.add(name, path).await?;

        crate::users::store()
            .update_user(
                acting_user(ws),
                FieldUpdate::Unchanged,
                FieldUpdate::Set(name),
                FieldUpdate::Unchanged,
            )
            .await?;

        Ok(format!(
            "Registered workspace '{name}' at {path} and switched your active workspace to it. \
             The workspace is being picked up — if the LLM provider is configured the pipeline \
             will claim and discover it shortly."
        ))
    }

    async fn exec_add_user(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        let handle = super::get_str(&args, "telegram")?;

        // default_agent is required (as the original tool had it); the only
        // valid value is Assistant — the single user-facing role. Checked before
        // any store access so required-field errors stay store-free.
        let agent: Role = super::get_str(&args, "default_agent")?
            .parse::<Role>()
            .context("default_agent must be 'assistant'")?;
        if agent != Role::Assistant {
            return Err(err("default_agent must be 'assistant'"));
        }

        let store = crate::users::store();
        // Normalize + guard the handle (reserved sentinel, anti-steal) before
        // the duplicate/admin checks so a rejected handle wins over e.g.
        // "user already exists".
        let handle = store.validate_telegram_bind(name, handle).await?;

        let mut existing_unbound = false;
        if store.user_exists(name).await? {
            // Reject a name that already exists AND is already Telegram-bound:
            // `add_user` is INSERT OR IGNORE, so a duplicate would silently keep the
            // existing row while re-binding the handle, producing a misleading
            // "Created user" report. A leftover unbound row (a prior run where
            // `add_user` succeeded but `bind_channel` failed) is allowed through so
            // the retry completes the binding instead of being permanently rejected.
            let bound = store
                .get_user_channels(name)
                .await?
                .iter()
                .any(|c| c.channel == "telegram");
            if bound {
                return Err(err(format!("A user named '{name}' already exists")));
            }
            // add_user only creates regular users — never mutate a full-permissions
            // row (the single 'admin' installer), which must not be re-rolable or
            // misreported as a regular user.
            if store.get_permissions(name).await?.as_deref() == Some("full") {
                return Err(err(format!(
                    "'{name}' is an admin — add_user only creates regular (non-admin) users"
                )));
            }
            existing_unbound = true;
        }

        if existing_unbound {
            // Restore the intended default agent since `add_user` won't update an
            // existing row.
            store
                .update_user(
                    name,
                    FieldUpdate::Set(agent.as_str()),
                    FieldUpdate::Unchanged,
                    FieldUpdate::Unchanged,
                )
                .await?;
        }

        store.add_user(name, None, agent).await?;
        store.bind_channel(name, "telegram", &handle).await?;

        let role_note = "They are a regular (non-admin) user: they can chat with the \
                         Assistant agent only.";
        if existing_unbound {
            Ok(format!(
                "Bound @{handle} to the existing user '{name}' and set their default agent to \
                 '{}'. {role_note}",
                agent.as_str()
            ))
        } else {
            Ok(format!(
                "Created user '{name}' with default agent '{}' and bound @{handle} to them. \
                 {role_note}",
                agent.as_str()
            ))
        }
    }

    async fn exec_setup_web_search(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let provider = super::get_str(&args, "provider")?;
        let provider = provider.to_ascii_lowercase();
        let firecrawl = provider == "firecrawl";
        if !firecrawl && provider != "exa" {
            return Err(err("provider must be 'firecrawl' or 'exa'"));
        }
        let key = super::get_str(&args, "key")?;

        crate::config::persist_settled_string_field(CONFIG_KEY_WEB_SEARCH_PROVIDER, &provider)
            .await?;
        crate::config::persist_settled_string_field(
            if firecrawl {
                CONFIG_KEY_FIRECRAWL_KEY
            } else {
                CONFIG_KEY_EXA_KEY
            },
            key,
        )
        .await?;

        Ok(format!(
            "Web-search backend registered: {provider}. Agents can now use `web_search`."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use serde_json::json;

    /// The merged tool's error paths that fire before any store/persist access
    /// (action dispatch, unknown action, and per-action required-field and
    /// provider validation — including `add_user`'s `default_agent`, validated
    /// ahead of the user store) are pure and cheaply testable.
    #[tokio::test]
    async fn action_dispatch_and_unknown_action() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        for action in [
            "setup_telegram_bot",
            "bind_telegram",
            "add_workspace",
            "add_user",
            "setup_web_search",
        ] {
            let args = json!({ "action": action });
            // Each action reports its own missing required field, proving the
            // dispatcher routes to the right handler.
            let err = tool.execute(&ws, args).await.unwrap_err().to_string();
            assert!(
                err.contains("Missing required field:"),
                "action '{action}' should report a missing field, got: {err}"
            );
        }

        let err = tool
            .execute(&ws, json!({ "action": "bogus_action" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown action 'bogus_action'"),
            "unknown action must be reported, got: {err}"
        );
        assert!(
            err.contains("setup_telegram_bot")
                && err.contains("bind_telegram")
                && err.contains("add_workspace")
                && err.contains("add_user")
                && err.contains("setup_web_search"),
            "unknown-action error must list the valid actions, got: {err}"
        );
    }

    #[tokio::test]
    async fn action_required_fields_are_checked() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        let cases: Vec<(serde_json::Value, &str)> = vec![
            (json!({ "action": "setup_telegram_bot" }), "token"),
            (json!({ "action": "bind_telegram" }), "handle"),
            (json!({ "action": "add_workspace", "name": "x" }), "path"),
            (json!({ "action": "add_workspace", "path": "/x" }), "name"),
            (json!({ "action": "add_user", "telegram": "@a" }), "name"),
            (json!({ "action": "add_user", "name": "x" }), "telegram"),
            (
                json!({ "action": "add_user", "name": "x", "telegram": "@a" }),
                "default_agent",
            ),
            (json!({ "action": "setup_web_search" }), "provider"),
            (
                json!({ "action": "setup_web_search", "provider": "exa" }),
                "key",
            ),
        ];

        for (args, field) in cases {
            let case_desc = format!("{args}");
            let err = tool.execute(&ws, args).await.unwrap_err().to_string();
            assert!(
                err.contains(&format!("Missing required field: {field}")),
                "case {case_desc} should report missing '{field}', got: {err}"
            );
        }
    }

    #[tokio::test]
    async fn provider_must_be_firecrawl_or_exa() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        let err = tool
            .execute(
                &ws,
                json!({ "action": "setup_web_search", "provider": "bing", "key": "k" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("provider must be 'firecrawl' or 'exa'"),
            "rejected provider must be reported, got: {err}"
        );
    }
}
