//! Merged admin config tool — one `mahbot_config` action dispatcher for the
//! five setup actions previously exposed as separate Support tools
//! (`setup_telegram_bot`, `bind_telegram`, `add_workspace`, `add_user`,
//! `setup_web_search`) plus the per-user custom-tool grants
//! (`grant_tool` / `revoke_tool` / `list_grants`). It is now the admin's
//! Assistant's sole configuration surface (the Support role is gone).
use crate::config::{
    CONFIG_KEY_EXA_KEY, CONFIG_KEY_FIRECRAWL_KEY, CONFIG_KEY_TELEGRAM_BOT_TOKEN,
    CONFIG_KEY_WEB_SEARCH_PROVIDER,
};
use crate::users::{GrantChange, format_grants};
use crate::{Tool, Workspace};
use anyhow::anyhow;
use async_trait::async_trait;
use serde_json::json;

/// The user configuration operates as, derived from the personal workspace it
/// runs in (`personal:<user>`). The Assistant always operates in a personal
/// workspace, so this resolves to the admin without a separate identity lookup.
fn acting_user(ws: &Workspace) -> &str {
    crate::users::personal_user_name(&ws.name).unwrap_or(crate::users::ADMIN_USER_NAME)
}

fn err(msg: impl Into<String>) -> anyhow::Error {
    anyhow!(msg.into())
}

/// Read a Telegram binding value: the @username or the numeric id arrives as a
/// string, but a number passed as a bare JSON number is that person's id just
/// the same, so it is accepted rather than refused.
fn get_binding_value(args: &serde_json::Value, key: &str) -> anyhow::Result<String> {
    match args.get(key) {
        Some(value) if value.is_i64() || value.is_u64() => Ok(value.to_string()),
        // Everything else is the shared string reader's call — a string, or its
        // missing/wrong-type usage error.
        _ => Ok(super::get_str(args, key)?.to_string()),
    }
}

/// Dispatch a single configuration action for the admin.
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
                        "grant_tool",
                        "revoke_tool",
                        "list_grants",
                    ],
                    "description": "Which configuration action to perform."
                },
                "token": {
                    "type": "string",
                    "description": "(setup_telegram_bot) The Telegram bot token from BotFather (the `NNN:AAA...` string)."
                },
                "handle": {
                    "type": "string",
                    "description": "(bind_telegram) The admin's Telegram @username (with or without the leading @), or their numeric Telegram id."
                },
                "name": {
                    "type": "string",
                    "description": "(add_workspace / add_user) A short unique name for the workspace (used in ticket ids and the GUI), or the new user's display name."
                },
                "user": {
                    "type": "string",
                    "description": "(grant_tool / revoke_tool / list_grants) The target user's name. Optional for list_grants — omit it to list every user's grants."
                },
                "tool": {
                    "type": "string",
                    "description": "(grant_tool / revoke_tool) The custom tool's name — the file name without its extension of a script in the `shared` folder of the admin's personal workspace."
                },
                "path": {
                    "type": "string",
                    "description": "(add_workspace) Absolute path to the project directory to manage."
                },
                "telegram": {
                    "type": "string",
                    "description": "(add_user) The new user's Telegram @username (with or without the leading @), or their numeric Telegram id."
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
            "grant_tool" => self.exec_grant_tool(args).await,
            "revoke_tool" => self.exec_revoke_tool(args).await,
            "list_grants" => self.exec_list_grants(args).await,
            other => Err(err(format!(
                "unknown action '{other}' — expected one of: setup_telegram_bot, bind_telegram, \
                 add_workspace, add_user, setup_web_search, grant_tool, revoke_tool, list_grants"
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
             your Telegram account — your @username or your numeric id — so messages are \
             routed to you."
                .to_string(),
        )
    }

    async fn exec_bind_telegram(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let value = get_binding_value(&args, "handle")?;

        let identifier = crate::users::store()
            .bind_telegram_for_admin(&value)
            .await?;

        Ok(format!(
            "Bound Telegram {} to your account. Messages sent to the bot from that account \
             will now be routed to you.",
            crate::users::describe_telegram_binding(&identifier)
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
            .set_selected_workspace(acting_user(ws), Some(name))
            .await?;

        Ok(format!(
            "Registered workspace '{name}' at {path} and switched your active workspace to it. \
             The workspace is being picked up — if the LLM provider is configured the pipeline \
             will claim and discover it shortly."
        ))
    }

    async fn exec_add_user(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        let value = get_binding_value(&args, "telegram")?;

        // The name IS the admin marker, so the reserved admin name is refused
        // before any store access.
        crate::users::validate_new_user_name(name)?;

        let store = crate::users::store();
        let exists = store.user_exists(name).await?;
        // A name that already exists and already holds a Telegram binding is a
        // duplicate account, and the retry cannot complete it: say what is
        // actually there, and how to free it for a different binding.
        if exists && let Some(existing) = store.telegram_binding(name).await? {
            return Err(err(format!(
                "A user named '{name}' already exists, with the Telegram binding {} — remove \
                 that binding on the Settings → Users page first, then attach the new one",
                crate::users::describe_telegram_binding(&existing)
            )));
        }
        // Guard the value: the reserved sentinel, the service identities, an
        // existing binding of this account, and a value owned by another
        // account. After that, `exists` only decides what is reported, since a
        // row holding no Telegram binding is a half-finished earlier run.
        let identifier = store.validate_telegram_bind(name, &value).await?;

        store.add_user(name).await?;
        store.attach_telegram_binding(name, &identifier).await?;

        let binding = crate::users::describe_telegram_binding(&identifier);
        let guest_note = "They are a guest — they chat with the Assistant agent only.";
        if exists {
            Ok(format!(
                "Bound {binding} to the existing user '{name}'. {guest_note}"
            ))
        } else {
            Ok(format!(
                "Created guest account '{name}' and bound {binding} to them. {guest_note}"
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

    async fn exec_grant_tool(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let user = super::get_str(&args, "user")?;
        let tool = super::get_str(&args, "tool")?;
        // A grant is recorded whatever the catalogue holds — it does not depend
        // on the file — but a name that could never be a tool name (a path, a
        // dot-file, empty) is a typo worth reporting now rather than storing a
        // grant that shows on the user's card and can never resolve.
        if !crate::tools::custom::is_tool_name(tool) {
            return Err(err(format!(
                "usage: '{tool}' cannot be a custom tool name — hint: a tool's name is its \
                 file name without the extension, with no path separators or leading dot"
            )));
        }

        let store = crate::users::store();
        match store.add_grant(user, tool).await? {
            GrantChange::NoUser => {
                return Err(err(format!(
                    "not-found: no user '{user}' — hint: grants need an existing user; create \
                     them first"
                )));
            }
            // An idempotent re-grant changes nothing, so there is nothing to
            // announce; only a real change wakes the account's Assistant.
            GrantChange::Unchanged => {}
            GrantChange::Changed => {
                crate::tools::custom::notify_grant_change(user, tool, true).await;
            }
        }
        let grants = store.get_grants(user).await?;
        Ok(format!(
            "Granted custom tool '{tool}' to '{user}'. Their custom tools are now: {}.",
            format_grants(&grants)
        ))
    }

    async fn exec_revoke_tool(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let user = super::get_str(&args, "user")?;
        let tool = super::get_str(&args, "tool")?;

        let store = crate::users::store();
        match store.remove_grant(user, tool).await? {
            // A missing user is not an error, but reporting a revoke that
            // touched nothing would be a false confirmation.
            GrantChange::NoUser => {
                return Ok(format!("No user '{user}' — nothing was revoked."));
            }
            // Only a real change is announced — see `exec_grant_tool`.
            GrantChange::Unchanged => {}
            GrantChange::Changed => {
                crate::tools::custom::notify_grant_change(user, tool, false).await;
            }
        }
        let grants = store.get_grants(user).await?;
        Ok(format!(
            "Revoked custom tool '{tool}' from '{user}'. Their custom tools are now: {}.",
            format_grants(&grants)
        ))
    }

    async fn exec_list_grants(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let user = match args.get("user") {
            None | Some(serde_json::Value::Null) => None,
            // `get_opt_str` is deliberately silent, so a wrong-typed user would
            // masquerade as "every user" — reject it here, before the store is
            // touched, like the other argument checks.
            Some(v) => Some(
                v.as_str()
                    .ok_or_else(|| super::wrong_type("user", "a string", v))?,
            ),
        };
        let store = crate::users::store();
        let Some(user) = user else {
            let all = store.list_grants().await?;
            if all.is_empty() {
                return Ok("No custom tools are granted to any user.".to_string());
            }
            return Ok(all
                .iter()
                .map(|(user, grants)| format!("{user}: {}", format_grants(grants)))
                .collect::<Vec<String>>()
                .join("\n"));
        };
        let grants = store.get_grants(user).await?;
        if grants.is_empty() {
            // "No grants" and "no such user" read the same out of the store,
            // and reporting the first for a name that does not exist would be a
            // false confirmation — as `revoke_tool` avoids too.
            if !store.user_exists(user).await? {
                return Ok(format!("No user '{user}' — nothing to list."));
            }
            return Ok(format!("No custom tools are granted to {user}."));
        }
        Ok(format!("{user}: {}", format_grants(&grants)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use serde_json::json;

    /// The merged tool's error paths that fire before any store/persist access
    /// (action dispatch, unknown action, and per-action required-field and
    /// provider validation) are pure and cheaply testable.
    #[tokio::test]
    async fn action_dispatch_and_unknown_action() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        // `list_grants` is absent: it has no required field, so it would
        // legitimately succeed with these args — its name is pinned by the
        // unknown-action message asserted below instead.
        for action in [
            "setup_telegram_bot",
            "bind_telegram",
            "add_workspace",
            "add_user",
            "setup_web_search",
            "grant_tool",
            "revoke_tool",
        ] {
            let args = json!({ "action": action });
            // Each action reports its own missing required field, proving the
            // dispatcher routes to the right handler.
            let err = tool.execute(&ws, args).await.unwrap_err().to_string();
            assert!(
                err.contains("usage: missing required argument"),
                "action '{action}' should report a missing field, got: {err}"
            );
        }

        // A present but wrong-typed `user` must not silently read as "every
        // user" — the one argument check `list_grants` performs before touching
        // any store.
        let err = tool
            .execute(&ws, json!({ "action": "list_grants", "user": 5 }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be a string"), "got: {err}");

        // A grant names a tool the way a call does: a name that could never be
        // one is rejected instead of being recorded.
        let err = tool
            .execute(
                &ws,
                json!({ "action": "grant_tool", "user": "someone", "tool": "a/b" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot be a custom tool name"), "got: {err}");

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
                && err.contains("setup_web_search")
                && err.contains("grant_tool")
                && err.contains("revoke_tool")
                && err.contains("list_grants"),
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
            (json!({ "action": "setup_web_search" }), "provider"),
            (
                json!({ "action": "setup_web_search", "provider": "exa" }),
                "key",
            ),
            (json!({ "action": "grant_tool" }), "user"),
            (json!({ "action": "grant_tool", "user": "alice" }), "tool"),
            (json!({ "action": "revoke_tool" }), "user"),
            (json!({ "action": "revoke_tool", "user": "alice" }), "tool"),
        ];

        for (args, field) in cases {
            let case_desc = format!("{args}");
            let err = tool.execute(&ws, args).await.unwrap_err().to_string();
            assert!(
                err.contains(&format!("usage: missing required argument \"{field}\"")),
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

    /// The name IS the admin marker, so `add_user` refuses the reserved admin
    /// name — this path only ever mints guest accounts.
    #[tokio::test]
    async fn add_user_refuses_the_admin_name() {
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        let err = tool
            .execute(
                &ws,
                json!({ "action": "add_user", "name": "admin", "telegram": "@a" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("only guest accounts can be created"),
            "the reserved admin name must be refused, got: {err}"
        );
    }
}
