//! Support-agent setup tools: onboarding for the admin user (Telegram binding,
//! workspaces, other users, web search, chrome-use, finalize).
use crate::config::{
    CONFIG_KEY_EXA_KEY, CONFIG_KEY_FIRECRAWL_KEY, CONFIG_KEY_TELEGRAM_BOT_TOKEN,
    CONFIG_KEY_WEB_SEARCH_PROVIDER,
};
use crate::users::FieldUpdate;
use crate::{Role, Tool, Workspace};
use anyhow::{Context, anyhow};
use async_trait::async_trait;
use serde_json::json;

/// The user the Support agent operates as, derived from the personal workspace
/// it runs in (`personal:<user>`). Support always pins to a personal workspace,
/// so this resolves to the admin without a separate identity lookup.
fn acting_user(ws: &Workspace) -> &str {
    crate::users::personal_user_name(&ws.name).unwrap_or("admin")
}

fn err(msg: impl Into<String>) -> anyhow::Error {
    anyhow!(msg.into())
}

/// Persist the Telegram bot token so the daemon can receive admin messages.
pub(crate) struct SetupTelegramBotTool;

#[async_trait]
impl Tool for SetupTelegramBotTool {
    fn name(&self) -> &'static str {
        "setup_telegram_bot"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "bot_token": {
                    "type": "string",
                    "description": "The Telegram bot token from BotFather (the `NNN:AAA...` string)."
                }
            }),
            &["bot_token"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let token = super::get_str(&args, "bot_token")?;
        crate::config::persist_settled_string_field(CONFIG_KEY_TELEGRAM_BOT_TOKEN, token).await?;
        Ok(
            "Telegram bot token saved — the Telegram listener hot-reloads it immediately. \
             Next, send `/start` to your bot in Telegram, then use `bind_telegram` to bind \
             your @username so messages are routed to you."
                .to_string(),
        )
    }
}

/// Bind the admin's Telegram @username so incoming messages route to them.
pub(crate) struct BindTelegramTool;

#[async_trait]
impl Tool for BindTelegramTool {
    fn name(&self) -> &'static str {
        "bind_telegram"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "telegram_handle": {
                    "type": "string",
                    "description": "The admin's Telegram @username (with or without the leading @)."
                }
            }),
            &["telegram_handle"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        // bind_telegram always targets the `admin` user: this is a single-admin
        // model, and Support only runs for the admin, so the personal workspace
        // it operates in resolves to the same identity. Never the operating
        // full-permissions user.
        let user = "admin";
        let handle = super::get_str(&args, "telegram_handle")?;

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
}

/// Register a workspace (name + path) and switch the admin's active workspace to it.
pub(crate) struct AddWorkspaceTool;

#[async_trait]
impl Tool for AddWorkspaceTool {
    fn name(&self) -> &'static str {
        "add_workspace"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "name": {
                    "type": "string",
                    "description": "A short unique name for the workspace (used in ticket ids and the GUI)."
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the project directory to manage."
                }
            }),
            &["name", "path"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
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
}

/// Create a new (non-admin) user bound to Telegram.
pub(crate) struct AddUserTool;

#[async_trait]
impl Tool for AddUserTool {
    fn name(&self) -> &'static str {
        "add_user"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "name": {
                    "type": "string",
                    "description": "The new user's display name."
                },
                "telegram_handle": {
                    "type": "string",
                    "description": "The user's Telegram @username (with or without the leading @)."
                },
                "default_agent": {
                    "type": "string",
                    "description": "The default agent for this user: 'assistant' or 'artist'."
                }
            }),
            &["name", "telegram_handle", "default_agent"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let name = super::get_str(&args, "name")?;
        let handle = super::get_str(&args, "telegram_handle")?;
        let agent = super::get_str(&args, "default_agent")?
            .parse::<Role>()
            .context("default_agent must be 'assistant' or 'artist'")?;
        if !matches!(agent, Role::Assistant | Role::Artist) {
            return Err(err("default_agent must be 'assistant' or 'artist'"));
        }

        let store = crate::users::store();
        // Normalize + guard the handle (reserved sentinel, anti-steal) before any
        // other check so a rejected handle wins over e.g. "user already exists".
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
            // The Support add_user tool only creates regular users — never mutate a
            // full-permissions row (the single 'admin' installer), which must not be
            // re-rolable or misreported as a regular user.
            if store.get_permissions(name).await?.as_deref() == Some("full") {
                return Err(err(format!(
                    "'{name}' is an admin — add_user only creates regular (non-admin) users"
                )));
            }
            existing_unbound = true;
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
                         Assistant and Artist agents only.";
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
}

/// Register a web-search backend (Firecrawl or Exa) with its API key.
pub(crate) struct SetupWebSearchTool;

#[async_trait]
impl Tool for SetupWebSearchTool {
    fn name(&self) -> &'static str {
        "setup_web_search"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "provider": {
                    "type": "string",
                    "description": "The web-search provider: 'firecrawl' or 'exa'."
                },
                "api_key": {
                    "type": "string",
                    "description": "The API key for the chosen provider."
                }
            }),
            &["provider", "api_key"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let provider = super::get_str(&args, "provider")?;
        let provider = provider.to_ascii_lowercase();
        let firecrawl = provider == "firecrawl";
        if !firecrawl && provider != "exa" {
            return Err(err("provider must be 'firecrawl' or 'exa'"));
        }
        let api_key = super::get_str(&args, "api_key")?;

        crate::config::persist_settled_string_field(CONFIG_KEY_WEB_SEARCH_PROVIDER, &provider)
            .await?;
        crate::config::persist_settled_string_field(
            if firecrawl {
                CONFIG_KEY_FIRECRAWL_KEY
            } else {
                CONFIG_KEY_EXA_KEY
            },
            api_key,
        )
        .await?;

        Ok(format!(
            "Web-search backend registered: {provider}. Agents can now use `web_search`."
        ))
    }
}

/// Install chrome-use: the release binary and native-messaging host.
pub(crate) struct InstallChromeUseTool;

#[async_trait]
impl Tool for InstallChromeUseTool {
    fn name(&self) -> &'static str {
        "install_chrome_use"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(&json!({}), &[])
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, _args: serde_json::Value) -> anyhow::Result<String> {
        crate::tools::browser_daemon::install_chrome_use()
            .await
            .map_err(err)?;

        Ok(
            "chrome-use is installed: the release binary was downloaded directly and \
             SHA-256-verified, and the native-messaging host was registered via \
             `chrome-use extension install --no-profile` — no managed-Chrome configuration \
             profile is ever created, so Chrome never enters \"managed by your organization\" \
             mode.\n\
             One MANUAL step remains — you must install the chrome-use browser extension \
             yourself:\n\
             1. Open Chrome and go to the Chrome Web Store.\n\
             2. Install the chrome-use extension (per its install docs).\n\
             3. Pin it once installed so the native host can reach it.\n\
             Only after that will agents be able to drive your normal browser via \
             `browser`.\n\
             Note: this setup is intentionally invasive — chrome-use gets full control \
             over the user's real browser. It must only be run after you (the Support \
             agent) have explained what it does and obtained the user's explicit consent.\n\
             mahbot auto-updates the chrome-use binary in place (checksum-verified release \
             download) after service startup — binary only; the extension and native host \
             are never re-registered."
                .to_string(),
        )
    }
}

/// Mark onboarding complete and switch the admin to their chosen agent.
pub(crate) struct FinalizeTool;

#[async_trait]
impl Tool for FinalizeTool {
    fn name(&self) -> &'static str {
        "finalize"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "agent": {
                    "type": "string",
                    "description": "The agent to switch to after onboarding: 'assistant', 'manager', or 'artist'."
                }
            }),
            &["agent"],
        )
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let user = acting_user(ws).to_string();
        let agent = super::get_str(&args, "agent")?.parse::<Role>()?;
        if !matches!(agent, Role::Assistant | Role::Manager | Role::Artist) {
            return Err(err("agent must be 'assistant', 'manager', or 'artist'"));
        }

        // Switch the active role BEFORE persisting `Finished` (mirrors
        // `kickoff_support`: the role action runs first, then the durable state
        // is recorded). If the role switch fails the state stays `Welcomed`, so
        // the Support agent can retry `finalize`; if the persist fails the user
        // is already on the chosen agent (idempotent) and a retry re-records it.
        crate::users::switch_active_role(&user, agent).await?;
        crate::config::persist_settled_string_field(
            crate::config::CONFIG_KEY_ONBOARDING_STATE,
            crate::config::OnboardingState::Finished.as_str(),
        )
        .await?;

        let label = crate::agent::role::role_info(&agent).display_label;
        Ok(format!(
            "Onboarding complete. You're now chatting with the {label} agent. Switch agents \
             anytime from the role picker, or open Settings via the gear icon to adjust anything."
        ))
    }
}
