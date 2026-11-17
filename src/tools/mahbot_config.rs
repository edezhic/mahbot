//! Merged admin config tool — one `mahbot_config` action dispatcher for the four setup actions
//! that were separate Support tools (`setup_telegram_bot`, `bind_telegram`, `add_user`,
//! `setup_web_search`) plus the per-user custom-tool grants (`grant_tool` / `revoke_tool` /
//! `list_grants`). It is the admin's Assistant's sole configuration surface. Adding a managed
//! workspace is not one of its actions — that lives in the `workspace` tool.
//!
//! `setup` is the exception: the command line's one-shot setup (`mahbot config setup`), carried
//! into the instance by this tool so the terminal and the window share one implementation. It
//! refuses outside a command-line call — the Assistant may use every other action, but writing the
//! provider credential is the owner's own act at a terminal. It is therefore absent from the tool's
//! schema, and `src/prompt/tool/mahbot_config.md` names the command instead of the action.
use crate::config::{
    CONFIG_KEY_DEEPSEEK_KEY, CONFIG_KEY_EXA_KEY, CONFIG_KEY_TELEGRAM_BOT_TOKEN,
    CONFIG_KEY_WEB_SEARCH_PROVIDER,
};
use crate::custom_role::CustomRoleBehavior;
use crate::tools::ToolAnswer;
use crate::users::{ADMIN_USER_NAME, AddUserOutcome, GrantChange, format_grants};
use crate::{Tool, Workspace};
use anyhow::anyhow;
use async_trait::async_trait;
use serde_json::json;

fn err(msg: impl Into<String>) -> anyhow::Error {
    anyhow!(msg.into())
}

/// The @username or numeric id arrive as a string, but a number passed as a bare JSON number is
/// that person's id just the same, so it is accepted rather than refused.
fn get_binding_value(args: &serde_json::Value, key: &str) -> anyhow::Result<String> {
    match args.get(key) {
        Some(value) if value.is_i64() || value.is_u64() => Ok(value.to_string()),
        // Everything else is the shared string reader's call — a string, or its
        // missing/wrong-type usage error.
        _ => Ok(super::get_str(args, key)?.to_string()),
    }
}

/// The `tools` name list: absent (or null) yields `None`, a present value must be an array of
/// strings.
fn optional_tools(args: &serde_json::Value) -> anyhow::Result<Option<Vec<String>>> {
    match args.get("tools") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(_) => Ok(Some(super::get_str_array(args, "tools")?)),
    }
}

/// The `behavior` field: absent (or null) yields `None`, a present value must be `continuous`
/// or `one_off`.
fn read_behavior(args: &serde_json::Value) -> anyhow::Result<Option<CustomRoleBehavior>> {
    match super::get_opt_str_strict(args, "behavior")? {
        None => Ok(None),
        Some(value) => Ok(Some(CustomRoleBehavior::parse(value)?)),
    }
}

/// Render a custom role's chosen tool subset for a human: the entries joined, or "none" for the
/// empty subset. Each entry is spelled with the kinds it holds
/// ([`crate::tools::grants::describe_all`]), so a bare entry shows the narrowest kind it implies.
fn format_tools(tools: &[String]) -> String {
    let described = crate::tools::grants::describe_all(tools);
    if described.is_empty() {
        "none".to_string()
    } else {
        described.join(", ")
    }
}

/// The arguments of this tool that carry a credential: each is a value the tool only writes, never
/// reads back as data.
const SECRET_ARGUMENTS: &[&str] = &[
    "token",
    "key",
    "provider_key",
    "provider_endpoint_key",
    "telegram_token",
];

pub(crate) struct MahbotConfigTool;

#[async_trait]
impl Tool for MahbotConfigTool {
    fn name(&self) -> &'static str {
        "mahbot_config"
    }

    /// The action word alone for a call that carries a credential — and nothing at all when the
    /// call names no action, since a call can carry a secret without one.
    fn recorded_arguments(&self, args: &serde_json::Value) -> Option<serde_json::Value> {
        if !SECRET_ARGUMENTS
            .iter()
            .any(|secret| args.get(*secret).is_some())
        {
            return None;
        }
        Some(
            match args.get("action").and_then(serde_json::Value::as_str) {
                Some(action) => json!({ "action": action }),
                None => json!({}),
            },
        )
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": [
                        "setup_telegram_bot",
                        "bind_telegram",
                        "add_user",
                        "setup_web_search",
                        "grant_tool",
                        "revoke_tool",
                        "list_grants",
                        "create_custom_role",
                        "update_custom_role",
                        "list_custom_roles",
                        "delete_custom_role",
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
                    "description": format!(
                        "(add_user / create_custom_role / update_custom_role / \
                         delete_custom_role) For a new user: their display name, which is also the \
                         name of their personal workspace folder, so it must be one every platform \
                         the service runs on can create: {}. {} For a custom role: the role's \
                         name — short, ASCII letters, digits, '_' and '-' only, starting with a \
                         letter, and not colliding with a built-in agent namespace (so not \
                         `ticket`, `maintainer`, `manager`, `role`, ...).",
                        crate::util::folder_name::RULE_SUMMARY,
                        crate::users::NAME_COLLISION_RULE
                    )
                },
                "prompt": {
                    "type": "string",
                    "description": "(create_custom_role / update_custom_role) The authorial text that fully defines the role. It is the whole definition of the role: it is the only thing the role is told besides the folder it works in, so everything the role must know and do has to be written in it."
                },
                "tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "(create_custom_role / update_custom_role) The chosen subset of your own tools: each entry is a tool name, optionally `name:kind1,kind2` for a tool with kinds (`analyze:quick,deep`, `send:worker,worker_branch,manager,role`); a bare name grants the tool's narrowest kind. An unknown name, an unknown kind and a duplicate are refused. Omit for no tools."
                },
                "behavior": {
                    "type": "string",
                    "enum": ["continuous", "one_off"],
                    "description": "(create_custom_role / update_custom_role) What a launch does when no branch identifier is given: `continuous` keeps one common branch of the role, `one_off` runs once and reports back."
                },
                "user": {
                    "type": "string",
                    "description": "(grant_tool / revoke_tool / list_grants) The target user's name. Optional for list_grants — omit it to list every user's grants."
                },
                "tool": {
                    "type": "string",
                    "description": "(grant_tool / revoke_tool) The custom tool's name — the file name without its extension of a script in the `shared` folder of the admin's personal workspace."
                },
                "telegram": {
                    "type": "string",
                    "description": "(add_user) The new user's Telegram @username (with or without the leading @), or their numeric Telegram id."
                },
                "provider": {
                    "type": "string",
                    "description": "(setup_web_search) The web-search provider: 'deepseek' or 'exa'."
                },
                "key": {
                    "type": "string",
                    "description": "(setup_web_search) The API key for the chosen provider."
                }
            }),
            &["action"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        Ok(self.answer(args).await?.text)
    }

    async fn execute_answer(
        &self,
        _ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<ToolAnswer> {
        self.answer(args).await
    }
}

impl MahbotConfigTool {
    /// The one dispatch both faces of the tool go through: `execute` renders it to its text and
    /// [`Tool::execute_answer`] hands it over with the partiality flag a command-line caller reads.
    ///
    /// `add_user` alone can half-succeed — the account is created whatever happens, and its
    /// personal workspace folder separately — so it builds its own answer; every other action
    /// happens in full or fails.
    async fn answer(&self, args: serde_json::Value) -> anyhow::Result<ToolAnswer> {
        match super::get_str(&args, "action")? {
            crate::setup::ACTION => Ok(ToolAnswer::done(crate::setup::run(&args).await?)),
            "add_user" => self.answer_add_user(args).await,
            "setup_telegram_bot" => Ok(ToolAnswer::done(self.exec_setup_telegram_bot(args).await?)),
            "bind_telegram" => Ok(ToolAnswer::done(self.exec_bind_telegram(args).await?)),
            "setup_web_search" => Ok(ToolAnswer::done(self.exec_setup_web_search(args).await?)),
            "grant_tool" => Ok(ToolAnswer::done(self.exec_grant_tool(args).await?)),
            "revoke_tool" => Ok(ToolAnswer::done(self.exec_revoke_tool(args).await?)),
            "list_grants" => Ok(ToolAnswer::done(self.exec_list_grants(args).await?)),
            "create_custom_role" => Ok(ToolAnswer::done(self.exec_create_custom_role(args).await?)),
            "update_custom_role" => Ok(ToolAnswer::done(self.exec_update_custom_role(args).await?)),
            "list_custom_roles" => Ok(ToolAnswer::done(self.exec_list_custom_roles().await?)),
            "delete_custom_role" => Ok(ToolAnswer::done(self.exec_delete_custom_role(args).await?)),
            other => Err(err(format!(
                "unknown action '{other}' — expected one of: {}, setup_telegram_bot, \
                 bind_telegram, add_user, setup_web_search, grant_tool, revoke_tool, list_grants, \
                 create_custom_role, update_custom_role, list_custom_roles, delete_custom_role",
                crate::setup::ACTION
            ))),
        }
    }

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
            .bind_telegram(ADMIN_USER_NAME, &value)
            .await?;

        Ok(format!(
            "Bound Telegram {} to your account. Messages sent to the bot from that account \
             will now be routed to you.",
            crate::users::describe_telegram_binding(&identifier)
        ))
    }

    /// Create guest account `name` and bind `telegram` to it, reporting both halves truthfully:
    /// the folder can be refused while the account is still created, and the answer is then
    /// partial rather than a whole success.
    async fn answer_add_user(&self, args: serde_json::Value) -> anyhow::Result<ToolAnswer> {
        let name = super::get_str(&args, "name")?;
        let value = get_binding_value(&args, "telegram")?;

        // The name IS the admin marker, so the reserved admin name is refused before any store
        // access. A name that would become a new account's folder is checked by `add_user`, which
        // refuses it before its row exists — the folder rule and the case/composition collision
        // alike; a name that already identifies one is never re-judged there.
        crate::users::refuse_admin_name(name)?;

        let store = crate::users::store();
        let exists = store.user_exists(name).await?;
        // A name that already exists and already holds a Telegram binding is a duplicate account
        // the retry cannot complete: say what is there, and how to free it.
        if exists && let Some(existing) = store.telegram_binding(name).await? {
            return Err(err(format!(
                "A user named '{name}' already exists, with the Telegram binding {} — remove \
                 that binding on the Settings → Users page first, then attach the new one",
                crate::users::describe_telegram_binding(&existing)
            )));
        }
        // Guard the reserved sentinel, the service identities, an existing binding of this
        // account, and a value owned by another account. `exists` then only decides what is
        // reported, since a row holding no binding is a half-finished earlier run.
        let identifier = store.validate_telegram_bind(name, &value).await?;

        let outcome = store.add_user(name).await?;
        store.attach_telegram_binding(name, &identifier).await?;

        let binding = crate::users::describe_telegram_binding(&identifier);
        let account = if exists {
            format!("Bound {binding} to the existing user '{name}'.")
        } else {
            format!("Created guest account '{name}' and bound {binding} to them.")
        };
        let answer =
            format!("{account} They are a guest — they chat with the Assistant agent only.");
        // The account half is always true — the row exists either way; the folder half is
        // stated only when it did not happen, and then the answer is partial.
        match outcome {
            AddUserOutcome::Complete => Ok(ToolAnswer::done(answer)),
            AddUserOutcome::WithoutPersonalWorkspace => Ok(ToolAnswer::partial(format!(
                "{answer} {}",
                crate::users::missing_personal_workspace_note(name)
            ))),
        }
    }

    async fn exec_setup_web_search(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let provider = super::get_str(&args, "provider")?;
        let provider = provider.to_ascii_lowercase();
        let deepseek = provider == "deepseek";
        if !deepseek && provider != "exa" {
            return Err(err("provider must be 'deepseek' or 'exa'"));
        }
        let key = super::get_str(&args, "key")?;

        crate::config::persist_settled_string_field(CONFIG_KEY_WEB_SEARCH_PROVIDER, &provider)
            .await?;
        crate::config::persist_settled_string_field(
            if deepseek {
                CONFIG_KEY_DEEPSEEK_KEY
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
        // A grant is recorded whatever the catalogue holds, but a name that could never be a
        // tool name (a path, a dot-file, empty) is a typo worth reporting now rather than storing
        // a grant that can never resolve.
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
            // An idempotent re-grant changes nothing to announce; only a real change wakes the
            // account's Assistant.
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
            // A missing user is not an error, but reporting a revoke that touched nothing would
            // be a false confirmation.
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
        // Refuse a wrong-typed user before the store is touched, like the other argument checks.
        let user = super::get_opt_str_strict(&args, "user")?;
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
            // "No grants" and "no such user" read the same out of the store, and the first would
            // be a false confirmation — as `revoke_tool` avoids too.
            if !store.user_exists(user).await? {
                return Ok(format!("No user '{user}' — nothing to list."));
            }
            return Ok(format!("No custom tools are granted to {user}."));
        }
        Ok(format!("{user}: {}", format_grants(&grants)))
    }

    async fn exec_create_custom_role(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        let prompt = super::get_str(&args, "prompt")?;
        let tools = optional_tools(&args)?.unwrap_or_default();
        let behavior = read_behavior(&args)?.unwrap_or(CustomRoleBehavior::Continuous);
        crate::custom_role::create(name, prompt, &tools, behavior).await?;
        Ok(format!(
            "Created custom role '{name}' (behaviour: {}) with tools: {}.",
            behavior.as_str(),
            format_tools(&tools)
        ))
    }

    async fn exec_update_custom_role(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        let prompt = super::get_opt_str_strict(&args, "prompt")?;
        let tools = optional_tools(&args)?;
        let behavior = read_behavior(&args)?;
        anyhow::ensure!(
            prompt.is_some() || tools.is_some() || behavior.is_some(),
            "usage: update_custom_role needs at least one of `prompt`, `tools` or `behavior` to \
             change"
        );
        crate::custom_role::update(name, prompt, tools.as_deref(), behavior).await?;
        Ok(format!("Updated custom role '{name}'."))
    }

    async fn exec_list_custom_roles(&self) -> anyhow::Result<String> {
        let roles = crate::custom_role::list().await?;
        if roles.is_empty() {
            return Ok("No custom roles are defined.".to_string());
        }
        Ok(roles
            .iter()
            .map(|role| {
                format!(
                    "{} — behaviour: {}, tools: {}\n  prompt: {}",
                    role.name,
                    role.behavior.as_str(),
                    format_tools(&role.tools),
                    role.prompt
                )
            })
            .collect::<Vec<String>>()
            .join("\n\n"))
    }

    async fn exec_delete_custom_role(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        crate::custom_role::delete(name).await?;
        Ok(format!(
            "Deleted custom role '{name}'. Automations addressed to it can no longer wake anyone, and \
             the owner was notified through the assistant."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use serde_json::json;

    /// Every credential this tool can be handed is kept out of the record, on both surfaces that
    /// record one — the row is a log kept for hours, and what it keeps is the action that ran.
    #[test]
    fn a_call_is_recorded_without_its_credentials() {
        let tool = MahbotConfigTool;
        let args = json!({
            "action": "setup",
            "provider_key": "sk-or-v1-secret",
            "provider_endpoint_key": "local-secret",
            "telegram_token": "123:ABC",
            "token": "456:DEF",
            "key": "exa-secret",
            "owner": "12345",
        });
        let recorded = tool
            .recorded_arguments(&args)
            .expect("this call takes credentials");
        assert_eq!(recorded, json!({ "action": "setup" }));

        // A call that carries no credential is not copied at all: the recorder keeps what it was
        // handed, and a tool that declares no credentials is never touched.
        assert_eq!(
            tool.recorded_arguments(&json!({ "action": "list_grants" })),
            None
        );
        assert_eq!(
            crate::tools::shell::ShellTool::new(crate::tools::shell::ShellMode::Full)
                .recorded_arguments(&json!({ "command": "ls" })),
            None
        );

        // A credential can arrive without a usable action — a call that never reached the dispatch
        // has no action to keep, and is still never recorded with its secret.
        for args in [
            json!({ "provider_key": "sk-or-v1-secret" }),
            json!({ "action": 7, "token": "456:DEF" }),
        ] {
            assert_eq!(
                tool.recorded_arguments(&args),
                Some(json!({})),
                "a credential with no action must still be kept out: {args}"
            );
        }
    }

    /// The rule is the credential, not the action: every action that takes one is covered, and a
    /// call that takes none is recorded as it was made — `owner` included.
    #[test]
    fn every_action_that_takes_a_credential_is_covered() {
        let tool = MahbotConfigTool;
        for (action, secret) in [
            ("setup", "provider_key"),
            ("setup_telegram_bot", "token"),
            ("setup_web_search", "key"),
        ] {
            let args = json!({ "action": action, secret: "secret-value", "other": "kept" });
            assert_eq!(
                tool.recorded_arguments(&args),
                Some(json!({ "action": action })),
                "'{action}' takes a credential and must not be recorded with it"
            );
        }
        // A value that is no credential is not a reason to drop the call.
        assert_eq!(
            tool.recorded_arguments(&json!({ "action": "bind_telegram", "owner": "12345" })),
            None
        );
    }

    /// The error paths that fire before any store access (action dispatch, unknown action,
    /// per-action required-field and provider validation) are pure and cheaply testable.
    #[tokio::test]
    async fn action_dispatch_and_unknown_action() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        // `list_grants` and `list_custom_roles` are absent: neither has a required field, so
        // they would succeed with these args — their names are pinned by the unknown-action
        // message below.
        for action in [
            "setup_telegram_bot",
            "bind_telegram",
            "add_user",
            "setup_web_search",
            "grant_tool",
            "revoke_tool",
            "create_custom_role",
            "update_custom_role",
            "delete_custom_role",
        ] {
            let args = json!({ "action": action });
            // Each action reports its own missing required field, proving the dispatcher routes
            // correctly.
            let err = tool.execute(&ws, args).await.unwrap_err().to_string();
            assert!(
                err.contains("usage: missing required argument"),
                "action '{action}' should report a missing field, got: {err}"
            );
        }

        // A present but wrong-typed `user` must not read as "every user" — the one argument check
        // `list_grants` makes before touching a store.
        let err = tool
            .execute(&ws, json!({ "action": "list_grants", "user": 5 }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be a string"), "got: {err}");

        // A grant names a tool the way a call does: a name that could never be one is rejected
        // instead of recorded.
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
                && err.contains("add_user")
                && err.contains("setup_web_search")
                && err.contains("grant_tool")
                && err.contains("revoke_tool")
                && err.contains("list_grants")
                && err.contains("create_custom_role")
                && err.contains("update_custom_role")
                && err.contains("list_custom_roles")
                && err.contains("delete_custom_role")
                && err.contains(&format!("one of: {}", crate::setup::ACTION)),
            "unknown-action error must list every action the dispatch accepts, the terminal one \
             included, got: {err}"
        );
    }

    #[tokio::test]
    async fn action_required_fields_are_checked() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        let cases: Vec<(serde_json::Value, &str)> = vec![
            (json!({ "action": "setup_telegram_bot" }), "token"),
            (json!({ "action": "bind_telegram" }), "handle"),
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
            (
                json!({ "action": "create_custom_role", "prompt": "p" }),
                "name",
            ),
            (
                json!({ "action": "create_custom_role", "name": "a" }),
                "prompt",
            ),
            (json!({ "action": "update_custom_role" }), "name"),
            (json!({ "action": "delete_custom_role" }), "name"),
        ];

        for (args, field) in cases {
            let case_desc = format!("{args}");
            let err = tool.execute(&ws, args).await.unwrap_err().to_string();
            assert!(
                err.contains(&format!("usage: missing required argument \"{field}\"")),
                "case {case_desc} should report missing '{field}', got: {err}"
            );
        }

        // Not a missing argument but a usage error of its own: `update_custom_role` must be told
        // what to update.
        let err = tool
            .execute(&ws, json!({ "action": "update_custom_role", "name": "a" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("usage:")
                && err.contains("prompt")
                && err.contains("tools")
                && err.contains("behavior"),
            "update with nothing to change must be a usage error, got: {err}"
        );
    }

    #[tokio::test]
    async fn provider_must_be_deepseek_or_exa() {
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
            err.contains("provider must be 'deepseek' or 'exa'"),
            "rejected provider must be reported, got: {err}"
        );
    }

    enum AddUserExpect {
        AdminName,
        /// A name an existing account matches, ignoring case.
        NameTaken,
    }

    struct AddUserRefusalCase {
        name: &'static str,
        /// An account created before the call, if any.
        existing: Option<&'static str>,
        args: serde_json::Value,
        expected: AddUserExpect,
    }

    #[tokio::test]
    async fn add_user_refusal_cases() {
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        for case in [
            AddUserRefusalCase {
                name: "the reserved admin name is refused",
                existing: None,
                args: json!({ "action": "add_user", "name": "admin", "telegram": "@a" }),
                expected: AddUserExpect::AdminName,
            },
            AddUserRefusalCase {
                name: "a case-insensitive collision with an existing account is refused",
                existing: Some("CaseGuest"),
                args: json!({ "action": "add_user", "name": "caseguest", "telegram": "@case_guest" }),
                expected: AddUserExpect::NameTaken,
            },
        ] {
            if let Some(existing) = case.existing {
                crate::users::store().add_user(existing).await.unwrap();
            }
            let err = tool.execute(&ws, case.args).await.unwrap_err().to_string();
            match case.expected {
                AddUserExpect::AdminName => assert!(
                    err.contains("only guest accounts can be created"),
                    "[{}] the reserved admin name must be refused, got: {err}",
                    case.name
                ),
                AddUserExpect::NameTaken => {
                    assert!(
                        err.contains(&crate::users::name_taken_message("caseguest")),
                        "[{}] a colliding name must be refused the way an exact duplicate is, got: {err}",
                        case.name
                    );
                    assert!(
                        !crate::users::store()
                            .user_exists("caseguest")
                            .await
                            .unwrap(),
                        "[{}] a refused name must not leave an account row",
                        case.name
                    );
                }
            }
        }
    }

    /// A bare entry is spelled with the narrowest kind it implies, and an explicit
    /// `tool:k1,k2` entry reads back as stored.
    #[tokio::test]
    async fn list_custom_roles_shows_the_kinds_a_role_holds() {
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        crate::custom_role::create(
            "kind_grants_probe",
            "you hold kinds",
            &["analyze".to_string(), "send:worker,manager".to_string()],
            CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();

        let out = tool
            .execute(&ws, json!({ "action": "list_custom_roles" }))
            .await
            .unwrap();
        assert!(out.contains("kind_grants_probe"), "{out}");
        assert!(
            out.contains("tools: analyze:quick, send:worker,manager"),
            "the listing spells a bare entry's narrowest kind and keeps the stored kinds: {out}"
        );
    }

    /// A created account whose personal workspace folder is in place answers exactly as it always
    /// has, and is not partial.
    #[tokio::test]
    async fn add_user_answers_a_whole_success_when_the_folder_is_created() {
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        let answer = tool
            .execute_answer(
                &ws,
                json!({ "action": "add_user", "name": "WholeGuest", "telegram": "@whole_guest" }),
            )
            .await
            .unwrap();

        assert!(
            !answer.partial,
            "a folder in place is a whole success, got: {}",
            answer.text
        );
        assert_eq!(
            answer.text,
            "Created guest account 'WholeGuest' and bound @whole_guest to them. They are a \
             guest — they chat with the Assistant agent only."
        );
        assert!(
            crate::users::personal_workspace_path("WholeGuest").is_dir(),
            "the account's personal workspace folder is created with it"
        );
    }

    /// The account is still created and the answer says so, partial rather than a whole success.
    #[tokio::test]
    async fn add_user_answers_a_partial_result_when_the_folder_cannot_be_created() {
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");
        // A file where the folder must go makes `create_dir_all` fail deterministically on
        // every platform.
        let name = "FolderRefusedGuest";
        let marker = crate::users::personal_workspace_path(name);
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, b"a file, not a folder").unwrap();

        let answer = tool
            .execute_answer(
                &ws,
                json!({ "action": "add_user", "name": name, "telegram": "@folder_refused" }),
            )
            .await
            .unwrap();

        assert!(
            answer.partial,
            "a folder that was not created is not a whole success, got: {}",
            answer.text
        );
        assert!(
            answer
                .text
                .contains("Created guest account 'FolderRefusedGuest'"),
            "the account half must still be reported: {}",
            answer.text
        );
        assert!(
            answer
                .text
                .contains(&crate::users::missing_personal_workspace_note(name)),
            "the folder half must be reported as not created: {}",
            answer.text
        );
        assert!(
            crate::users::store().user_exists(name).await.unwrap(),
            "the account exists even though its folder does not"
        );

        std::fs::remove_file(&marker).unwrap();
    }
}
