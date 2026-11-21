//! The product's own configuration surface — the admin's Assistant's one way to change it
//! without a window. It dispatches `mahbot_config`'s actions: the settings themselves
//! (`get` / `list` / `set`, implemented in [`crate::settings`]), the provider/Telegram/owner
//! setup, accounts, tool grants and custom roles, plus the two texts it serves whole — the
//! `reference` and, through `about`, the product's own README. Adding a managed workspace is not
//! one of its actions — that lives in the `workspace` tool.
//!
//! The standing description is deliberately short and each schema field carries its name and a
//! short description: the whole text — every action, every field, and what is inert where — is the
//! `reference` action, one embedded asset ([`crate::settings::reference`]) that the terminal's
//! `mahbot config reference` prints too, so the two cannot drift.
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

/// The `telegram` field of `add_user`: absent (or null) yields `None` — an account created with
/// no way in yet — while a number passed as a bare JSON number is that person's id.
fn optional_binding_value(args: &serde_json::Value, key: &str) -> anyhow::Result<Option<String>> {
    match args.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) if value.is_i64() || value.is_u64() => Ok(Some(value.to_string())),
        Some(_) => Ok(Some(super::get_str(args, key)?.to_string())),
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
/// reads back as data. `value` is not among them: [`MahbotConfigTool::recorded_arguments`] drops
/// every `set`'s value whatever setting it names, so no credential can ride along.
const SECRET_ARGUMENTS: &[&str] = &[
    "token",
    "key",
    "provider_key",
    "provider_endpoint_key",
    "telegram_token",
];

/// This tool's schema, built once: [`Tool::parameters_schema`] hands out a copy, and the
/// unknown-field guard reads the same list without rebuilding it on every call.
fn schema() -> &'static serde_json::Value {
    static SCHEMA: std::sync::LazyLock<serde_json::Value> = std::sync::LazyLock::new(build_schema);
    &SCHEMA
}

#[expect(clippy::too_many_lines)]
fn build_schema() -> serde_json::Value {
    super::tool_params_schema(
        &json!({
            "action": {
                "type": "string",
                "enum": [
                    "reference",
                    "about",
                    "get",
                    "list",
                    "set",
                    "setup",
                    "setup_telegram_bot",
                    "bind_telegram",
                    "unbind_telegram",
                    "add_user",
                    "delete_user",
                    "list_users",
                    "setup_web_search",
                    "grant_tool",
                    "revoke_tool",
                    "list_grants",
                    "create_custom_role",
                    "update_custom_role",
                    "list_custom_roles",
                    "delete_custom_role",
                ],
                "description": "Which action. Its `reference` action gives every action and field in full."
            },
            "setting": {
                "type": "string",
                "description": "(get / set) The setting's name; its `reference` action lists them."
            },
            "value": {
                "type": "string",
                "description": "(set) Its new value."
            },
            "provider_key": {
                "type": "string",
                "description": "(setup) The OpenRouter key."
            },
            "provider_endpoint": {
                "type": "string",
                "description": "(setup) A custom OpenAI-compatible chat endpoint."
            },
            "provider_endpoint_key": {
                "type": "string",
                "description": "(setup) That endpoint's own key."
            },
            "clear_provider_endpoint": {
                "type": "boolean",
                "description": "(setup) Settle back to OpenRouter's own endpoint, dropping the custom key with it."
            },
            "telegram_token": {
                "type": "string",
                "description": "(setup) The Telegram bot token."
            },
            "owner": {
                "type": "string",
                "description": "(setup) The owner's Telegram @username or numeric id."
            },
            "unbind_owner": {
                "type": "boolean",
                "description": "(setup) Remove the owner's Telegram binding."
            },
            "token": {
                "type": "string",
                "description": "(setup_telegram_bot) The bot token from BotFather."
            },
            "handle": {
                "type": "string",
                "description": "(bind_telegram) The @username or numeric id to bind — the owner's own, or a guest named by `user`."
            },
            "name": {
                "type": "string",
                "description": "(add_user) The account's name, also its folder; (custom roles) the role's name."
            },
            "telegram": {
                "type": "string",
                "description": "(add_user) Its Telegram @username or numeric id; omit for an account with no way in yet."
            },
            "user": {
                "type": "string",
                "description": "(bind_telegram / unbind_telegram / delete_user / grant_tool / revoke_tool / list_grants) The account. `bind_telegram` defaults to the owner and `list_grants` to every account; the other four require it."
            },
            "provider": {
                "type": "string",
                "description": "(setup_web_search) `deepseek` or `exa`."
            },
            "key": {
                "type": "string",
                "description": "(setup_web_search) That backend's key."
            },
            "tool": {
                "type": "string",
                "description": "(grant_tool / revoke_tool) The custom tool's name."
            },
            "tools": {
                "type": "array",
                "items": { "type": "string" },
                "description": "(custom roles) The tools it may use: a catalogue tool name, optionally `name:kind`, or `custom:<script>` for one script in your `shared` folder (one entry per script). An empty array clears them."
            },
            "behavior": {
                "type": "string",
                "enum": ["continuous", "one_off"],
                "description": "(custom roles) Its launch behaviour."
            },
            "prompt": {
                "type": "string",
                "description": "(custom roles) Its whole definition — everything the role is told."
            }
        }),
        &["action"],
    )
}

pub(crate) struct MahbotConfigTool;

#[async_trait]
impl Tool for MahbotConfigTool {
    fn name(&self) -> &'static str {
        "mahbot_config"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        schema().clone()
    }

    /// Every `set` is recorded as the setting's name alone, never with its `value`: a `set` writes
    /// whatever setting it names, so the value may be a credential and the name may be a misspelling
    /// of one — the rule recognises nothing and has nothing to misspell. Any other call is the
    /// action word alone, and not copied at all when it carries no credential.
    fn recorded_arguments(&self, args: &serde_json::Value) -> Option<serde_json::Value> {
        let action = args.get("action").and_then(serde_json::Value::as_str);
        if action == Some("set") {
            return Some(
                match args.get("setting").and_then(serde_json::Value::as_str) {
                    Some(setting) => json!({ "action": "set", "setting": setting }),
                    None => json!({ "action": "set" }),
                },
            );
        }
        if !SECRET_ARGUMENTS
            .iter()
            .any(|secret| args.get(*secret).is_some())
        {
            return None;
        }
        Some(match action {
            Some(action) => json!({ "action": action }),
            None => json!({}),
        })
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

    /// The `reference` and `about` answers must reach the model whole — each is one text, and a
    /// cut reference or README is worse than none — while every other answer keeps the framework's
    /// ~5 KB budget, so a role's whole prompt cannot dump into the context. The flag rides the
    /// call, not the tool.
    async fn execute_with_payloads(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<crate::tools::ToolOutput> {
        let preserve_text = matches!(
            args.get("action").and_then(serde_json::Value::as_str),
            Some("reference" | "about")
        );
        let text = self.execute(ws, args.clone()).await?;
        let mut output = crate::tools::with_image_payload(self, text, ws, &args).await;
        output.preserve_text = preserve_text;
        Ok(output)
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
        super::refuse_unknown_fields(schema(), &args, super::REFERENCE_ACTION_HINT)?;
        match super::get_str(&args, "action")? {
            "reference" => Ok(ToolAnswer::done(crate::settings::reference())),
            "about" => Ok(ToolAnswer::done(crate::settings::about())),
            "get" => Ok(ToolAnswer::done(
                crate::settings::get(super::get_str(&args, "setting")?).await?,
            )),
            "list" => Ok(ToolAnswer::done(crate::settings::list().await)),
            "set" => Ok(ToolAnswer::done(
                crate::settings::set(
                    super::get_str(&args, "setting")?,
                    super::get_str(&args, "value")?,
                )
                .await?,
            )),
            crate::setup::ACTION => Ok(ToolAnswer::done(crate::setup::run(&args).await?)),
            "add_user" => self.answer_add_user(args).await,
            "setup_telegram_bot" => Ok(ToolAnswer::done(self.exec_setup_telegram_bot(args).await?)),
            "bind_telegram" => Ok(ToolAnswer::done(self.exec_bind_telegram(args).await?)),
            "unbind_telegram" => Ok(ToolAnswer::done(self.exec_unbind_telegram(args).await?)),
            "delete_user" => Ok(ToolAnswer::done(self.exec_delete_user(args).await?)),
            "list_users" => Ok(ToolAnswer::done(self.exec_list_users().await?)),
            "setup_web_search" => Ok(ToolAnswer::done(self.exec_setup_web_search(args).await?)),
            "grant_tool" => Ok(ToolAnswer::done(self.exec_grant_tool(args).await?)),
            "revoke_tool" => Ok(ToolAnswer::done(self.exec_revoke_tool(args).await?)),
            "list_grants" => Ok(ToolAnswer::done(self.exec_list_grants(args).await?)),
            "create_custom_role" => Ok(ToolAnswer::done(self.exec_create_custom_role(args).await?)),
            "update_custom_role" => Ok(ToolAnswer::done(self.exec_update_custom_role(args).await?)),
            "list_custom_roles" => Ok(ToolAnswer::done(self.exec_list_custom_roles().await?)),
            "delete_custom_role" => Ok(ToolAnswer::done(self.exec_delete_custom_role(args).await?)),
            other => Err(err(format!(
                "unknown action '{other}' — call the `reference` action (`mahbot config \
                 reference`) for every action and field"
            ))),
        }
    }

    /// Remove an account's Telegram binding. Idempotent: an account holding none is the state
    /// asked for.
    async fn exec_unbind_telegram(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let user = super::get_str(&args, "user")?;
        let store = crate::users::store();
        if !store.user_exists(user).await? {
            return Err(err(format!("not-found: no user '{user}'")));
        }
        Ok(if store.unbind_telegram(user).await? {
            format!(
                "Unbound {user}'s Telegram account — messages from it no longer reach the service."
            )
        } else {
            format!("{user} held no Telegram binding; nothing was removed.")
        })
    }

    /// Delete an account and its channel bindings, naming what stays on disk.
    async fn exec_delete_user(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let user = super::get_str(&args, "user")?;
        // The admin is the owner: deleting that account would take away the only way in, and the
        // window suppresses the control for the same reason.
        if user == ADMIN_USER_NAME {
            return Err(err(format!(
                "'{user}' is the owner's own account and cannot be deleted — bind or unbind it, \
                 but it is the account the service answers"
            )));
        }
        let store = crate::users::store();
        if !store.user_exists(user).await? {
            return Err(err(format!("not-found: no user '{user}'")));
        }
        store.delete_user(user).await?;
        let folder = crate::users::personal_workspace_path(user);
        Ok(format!(
            "Deleted account '{user}' and its channel bindings. What stays on disk: its sessions \
             and chat history, and its personal workspace folder '{}' with everything in it — \
             nothing was removed from there.",
            folder.display()
        ))
    }

    async fn exec_list_users(&self) -> anyhow::Result<String> {
        let users = crate::users::store().list_users().await?;
        if users.is_empty() {
            return Ok("No accounts exist.".to_string());
        }
        Ok(users
            .iter()
            .map(|entry| {
                let telegram = match entry
                    .record
                    .channels
                    .iter()
                    .find(|c| c.channel == "telegram")
                {
                    Some(binding) => crate::users::describe_telegram_binding(&binding.identifier),
                    None => "no Telegram binding — no way in yet".to_string(),
                };
                let role = if entry.record.name == ADMIN_USER_NAME {
                    "owner"
                } else {
                    "guest"
                };
                format!("{} ({role}): {telegram}", entry.record.name)
            })
            .collect::<Vec<String>>()
            .join("\n"))
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

    /// Bind an account's Telegram identity. The owner's own account is the default target; a
    /// guest is named with `user`, which is also how an account created without a binding gets one.
    /// The target is checked first, so a mistyped or guessed name is a plain not-found rather than
    /// the channel row's foreign-key error.
    async fn exec_bind_telegram(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let value = optional_binding_value(&args, "handle")?.ok_or_else(|| {
            err("usage: missing required argument \"handle\" — the Telegram @username or numeric id")
        })?;
        let user = super::get_opt_named_str(&args, "user")?.unwrap_or(ADMIN_USER_NAME);
        let store = crate::users::store();
        if !store.user_exists(user).await? {
            return Err(err(format!("not-found: no user '{user}'")));
        }

        let identifier = store.bind_telegram(user, &value).await?;

        Ok(format!(
            "Bound Telegram {} to '{user}'. Messages sent to the bot from that account will now \
             be routed there{}.",
            crate::users::describe_telegram_binding(&identifier),
            if user == ADMIN_USER_NAME {
                ""
            } else {
                " — a guest chats with the Assistant agent only"
            }
        ))
    }

    /// Create guest account `name` and, when `telegram` is given, bind it — reporting both halves
    /// truthfully: the folder can be refused while the account is still created, and the answer is
    /// then partial rather than a whole success. With no `telegram` the account exists with no way
    /// in yet, and the answer says so.
    async fn answer_add_user(&self, args: serde_json::Value) -> anyhow::Result<ToolAnswer> {
        let name = super::get_str(&args, "name")?;
        let value = optional_binding_value(&args, "telegram")?;

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
                 that binding with `unbind_telegram` first, then attach the new one",
                crate::users::describe_telegram_binding(&existing)
            )));
        }

        let Some(value) = value else {
            // No binding: the account is created and left with no way in until one is attached.
            let outcome = store.add_user(name).await?;
            let answer = if exists {
                format!("The account '{name}' already exists and holds no Telegram binding.")
            } else {
                format!(
                    "Created guest account '{name}' with no Telegram binding — they have no way \
                     in yet. Bind their Telegram account when you know their @username or numeric \
                     id, and ask them to send /start to the bot."
                )
            };
            return Ok(match outcome {
                AddUserOutcome::Complete => ToolAnswer::done(answer),
                AddUserOutcome::WithoutPersonalWorkspace => ToolAnswer::partial(format!(
                    "{answer} {}",
                    crate::users::missing_personal_workspace_note(name)
                )),
            });
        };

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
        use crate::tools::web_search::WebSearchBackend;
        let provider = super::get_str(&args, "provider")?;
        let provider = provider.to_ascii_lowercase();
        let deepseek = provider == WebSearchBackend::DEEPSEEK;
        if !deepseek && provider != WebSearchBackend::EXA {
            return Err(err(format!(
                "provider must be '{}' or '{}'",
                WebSearchBackend::DEEPSEEK,
                WebSearchBackend::EXA
            )));
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

    /// The reference and the README arrive whole while every other answer keeps the framework's
    /// budget: the flag rides the call, so the text arrives whole without un-truncating a listing.
    #[tokio::test]
    async fn the_static_texts_arrive_whole_and_other_answers_do_not() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");

        let reference = tool
            .execute_with_payloads(&ws, json!({ "action": "reference" }))
            .await
            .expect("the reference touches no store");
        assert!(
            reference.preserve_text,
            "a cut reference is worse than none"
        );

        // The README is the other text served whole, and it is the file this product ships: read
        // back from the tree, so a hand-copied replacement fails here.
        let about = tool
            .execute_with_payloads(&ws, json!({ "action": "about" }))
            .await
            .expect("the README touches no store");
        assert!(about.preserve_text, "a cut README is worse than none");
        assert_eq!(
            about.text,
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
                .expect("the README this product ships"),
            "`about` answers with the shipped README, not a copy of it"
        );
        assert!(
            about.text.len() > crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "the README is longer than an ordinary answer, so the flag is what delivers it"
        );
        // Embedded as the repository holds it: a checkout that rewrote line endings would ship
        // bytes the file does not have (`.gitattributes` pins this one to LF).
        assert!(
            !about.text.contains('\r'),
            "the embedded README carries no carriage return"
        );

        let single = tool
            .execute_with_payloads(&ws, json!({ "action": "get", "setting": "manager_model" }))
            .await
            .expect("reading one setting touches no store");
        assert!(
            !single.preserve_text,
            "every other answer keeps the framework's budget"
        );
    }

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

        // The rule is the credential, not the action: every action that takes one is covered, and a
        // value that is no credential is not a reason to drop the call — `owner` included.
        for (action, secret) in [("setup_telegram_bot", "token"), ("setup_web_search", "key")] {
            let args = json!({ "action": action, secret: "secret-value", "other": "kept" });
            assert_eq!(
                tool.recorded_arguments(&args),
                Some(json!({ "action": action })),
                "'{action}' takes a credential and must not be recorded with it"
            );
        }
        assert_eq!(
            tool.recorded_arguments(&json!({ "action": "bind_telegram", "owner": "12345" })),
            None
        );

        // Every `set` is recorded as the setting's name alone: the value may be a credential, and
        // the rule must not depend on recognising the name it sits beside.
        for args in [
            json!({ "action": "set", "setting": "provider_key", "value": "sk-or-v1-0123456789" }),
            json!({ "action": "set", "setting": "manager_model", "value": "acme/model" }),
            // A misspelt name, a missing one and a non-string one are all the same call.
            json!({ "action": "set", "setting": "provider_ky", "value": "sk-or-v1-0123456789" }),
            json!({ "action": "set", "value": "sk-or-v1-0123456789" }),
            json!({ "action": "set", "setting": 7, "value": "sk-or-v1-0123456789" }),
        ] {
            let recorded = tool
                .recorded_arguments(&args)
                .unwrap_or_else(|| panic!("a set is always recorded: {args}"));
            assert!(
                recorded.get("value").is_none(),
                "a set's value never reaches the log row: {recorded}"
            );
            let named = args.get("setting").and_then(serde_json::Value::as_str);
            assert_eq!(
                recorded.get("setting").and_then(serde_json::Value::as_str),
                named,
                "the record keeps the setting's name exactly when there is one: {recorded}"
            );
        }
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
            err.contains("`reference` action"),
            "an unknown action must name where every action and field is listed, got: {err}"
        );
    }

    /// A guessed *field* name is refused and names the reference too — the call would otherwise
    /// do less than it asked for, silently.
    #[tokio::test]
    async fn a_guessed_field_names_the_reference() {
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");
        let err = tool
            .execute(
                &ws,
                json!({ "action": "get", "setting": "provider_key", "settign": "x" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown field 'settign'") && err.contains("`reference` action"),
            "a guessed field must name the reference, got: {err}"
        );
        // A null is a provider's padding, not a guess.
        assert!(
            tool.execute(
                &ws,
                json!({ "action": "list", "settign": serde_json::Value::Null })
            )
            .await
            .is_ok()
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
            (json!({ "action": "get" }), "setting"),
            (json!({ "action": "set" }), "setting"),
            (
                json!({ "action": "set", "setting": "manager_model" }),
                "value",
            ),
            (json!({ "action": "unbind_telegram" }), "user"),
            (json!({ "action": "delete_user" }), "user"),
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
        let _guard = crate::util::test::STORE_LOCK.lock().await;
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

    /// The account lifecycle, in the order an owner meets it: an account created with no way in,
    /// bound afterwards, listed, unbound and deleted — each answer naming what it did, and the
    /// deletion what it leaves on disk.
    #[tokio::test]
    async fn an_account_lives_through_creation_binding_and_deletion() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");
        let name = "probe_guest";
        let handle = "77700042";

        let created = tool
            .execute(&ws, json!({ "action": "add_user", "name": name }))
            .await
            .unwrap();
        assert!(created.contains("no way in yet"), "{created}");
        let store = crate::users::store();
        assert!(store.user_exists(name).await.unwrap());
        assert_eq!(
            store.telegram_binding(name).await.unwrap(),
            None,
            "an account created without a binding holds none"
        );

        let listed = tool
            .execute(&ws, json!({ "action": "list_users" }))
            .await
            .unwrap();
        assert!(
            listed.contains(&format!("{name} (guest): no Telegram binding")),
            "{listed}"
        );

        let bound = tool
            .execute(
                &ws,
                json!({ "action": "bind_telegram", "user": name, "handle": handle }),
            )
            .await
            .unwrap();
        assert!(bound.contains(name) && bound.contains(handle), "{bound}");
        assert_eq!(
            store.telegram_binding(name).await.unwrap().as_deref(),
            Some(handle)
        );

        // A guessed account is a plain not-found, never the channel row's foreign-key error.
        let missing = tool
            .execute(
                &ws,
                json!({ "action": "bind_telegram", "user": "no_such_guest", "handle": "77700043" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains("not-found: no user 'no_such_guest'"),
            "{missing}"
        );

        let unbound = tool
            .execute(&ws, json!({ "action": "unbind_telegram", "user": name }))
            .await
            .unwrap();
        assert!(unbound.contains("no longer reach"), "{unbound}");
        assert_eq!(store.telegram_binding(name).await.unwrap(), None);

        let deleted = tool
            .execute(&ws, json!({ "action": "delete_user", "user": name }))
            .await
            .unwrap();
        assert!(
            deleted.contains(name) && deleted.contains("personal workspace folder"),
            "a deletion names what it leaves on disk: {deleted}"
        );
        assert!(!store.user_exists(name).await.unwrap());

        let owner = tool
            .execute(
                &ws,
                json!({ "action": "delete_user", "user": ADMIN_USER_NAME }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(owner.contains("cannot be deleted"), "{owner}");
    }

    /// A bare entry is spelled with the narrowest kind it implies, an explicit `tool:k1,k2` entry
    /// reads back as stored, and a script entry as itself.
    #[tokio::test]
    async fn list_custom_roles_shows_the_kinds_a_role_holds() {
        crate::util::test::init_test_stores().await;
        let tool = MahbotConfigTool;
        let ws = test_ws("/tmp/test_ws");
        let dir =
            crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME).join("shared");
        std::fs::create_dir_all(&dir).unwrap();
        let probe = crate::util::test::ProbeFile(dir.join("kind_script_probe.ts"));
        std::fs::write(&probe.0, "// @description Probe.\n").unwrap();

        crate::custom_role::create(
            "kind_grants_probe",
            "you hold kinds",
            &[
                "analyze".to_string(),
                "send:worker,manager".to_string(),
                "custom:kind_script_probe".to_string(),
            ],
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
            out.contains("tools: analyze:quick, send:worker,manager, custom:kind_script_probe"),
            "the listing spells a bare entry's narrowest kind, keeps the stored kinds and names \
             each script: {out}"
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
