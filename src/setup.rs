//! One setup path for a running instance: `mahbot config setup`, and the `setup` action of the
//! `mahbot_config` tool that command line carries into the instance.
//!
//! It is the whole of "put this install's credentials and its owner in place" in one call: the
//! model provider (an OpenRouter key, or a custom OpenAI-compatible endpoint with an optional key
//! of its own), the Telegram bot token, and which Telegram account is the owner. It changes only
//! what the call named, so it is idempotent and safe to run again at any time; a run that names
//! nothing reports the current state and changes nothing but the onboarding step a configured
//! provider already earns.
//!
//! It is not a terminal UI and never becomes one: no prompt, no read from stdin, no loop — the
//! whole configuration is in the arguments, so the same code serves a script and a person. The
//! desktop onboarding flow ([`crate::onboarding`]) is the other writer of that state and stays the
//! GUI's own path; this one deliberately does not send the greeting that flow closes with, since
//! nothing is listening in a window the terminal command never opened.
//!
//! A run that fails partway still reports both halves: what landed, and the step it stopped at.
//!
//! The action is terminal-only: it refuses outside a command-line call, so the admin Assistant
//! cannot write the provider credential (it keeps every other `mahbot_config` action).
use crate::config::{
    CONFIG, CONFIG_KEY_PROVIDER_ENDPOINT, CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
    CONFIG_KEY_PROVIDER_KEY, OnboardingState,
};
use crate::users::{ADMIN_USER_NAME, describe_telegram_binding};
use crate::util::json::{get_bool, get_opt_str_strict};
use anyhow::Context as _;
use std::fmt::Write as _;
use std::time::Duration;

/// The action word the terminal command carries into the instance (`mahbot config setup`). It is
/// not in the Assistant's tool schema: [`run`] refuses every other caller.
pub(crate) const ACTION: &str = "setup";

/// OpenRouter's own key record: the one authenticated call that tells a known key from an unknown
/// one without spending a generation.
const OPENROUTER_KEY_CHECK: &str = "https://openrouter.ai/api/v1/key";

/// How long that check may take. One round trip to a platform that answers in milliseconds; a
/// socket that never answers is no reason to hold the command.
const KEY_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// The arguments of one run: exactly the command line's flags, absent for a run that names nothing.
#[derive(Debug, Default, PartialEq, Eq)]
struct SetupArgs {
    provider_key: Option<String>,
    provider_endpoint: Option<String>,
    provider_endpoint_key: Option<String>,
    telegram_token: Option<String>,
    owner: Option<String>,
    unbind_owner: bool,
}

/// One optional text argument. Absent (or null) is no input; a present non-string is the shared
/// reader's usage error; text that is blank once trimmed is refused rather than silently ignored,
/// because a blank secret is never what the caller meant.
fn text_arg(args: &serde_json::Value, key: &str) -> anyhow::Result<Option<String>> {
    match get_opt_str_strict(args, key)? {
        Some(value) if value.trim().is_empty() => anyhow::bail!(
            "usage: argument \"{key}\" is blank — give it a value, or leave the argument out"
        ),
        Some(value) => Ok(Some(value.trim().to_string())),
        None => Ok(None),
    }
}

impl SetupArgs {
    fn parse(args: &serde_json::Value) -> anyhow::Result<Self> {
        let parsed = Self {
            provider_key: text_arg(args, "provider_key")?,
            provider_endpoint: text_arg(args, "provider_endpoint")?,
            provider_endpoint_key: text_arg(args, "provider_endpoint_key")?,
            telegram_token: text_arg(args, "telegram_token")?,
            owner: text_arg(args, "owner")?,
            unbind_owner: get_bool(args, "unbind_owner", false)?,
        };
        // A custom endpoint's key names nothing on its own, and binding and unbinding the owner are
        // one decision: each pair is refused rather than resolved by an order the caller cannot
        // see.
        anyhow::ensure!(
            parsed.provider_endpoint.is_some() || parsed.provider_endpoint_key.is_none(),
            "usage: \"provider_endpoint_key\" needs \"provider_endpoint\" — a custom endpoint's key \
             is saved with the endpoint it belongs to"
        );
        // A key beside OpenRouter's own address names nothing: settling that endpoint drops the
        // custom key row, so the write would be undone before the report was rendered.
        anyhow::ensure!(
            parsed.provider_endpoint_key.is_none()
                || parsed
                    .provider_endpoint
                    .as_deref()
                    .is_some_and(crate::config::is_custom_endpoint),
            "usage: \"provider_endpoint_key\" is a custom endpoint's own key — give it with a \
             \"provider_endpoint\" that is not OpenRouter's (an OpenRouter key goes in \
             \"provider_key\")"
        );
        anyhow::ensure!(
            !(parsed.owner.is_some() && parsed.unbind_owner),
            "usage: \"owner\" and \"unbind_owner\" are one decision — give one or the other"
        );
        // The owner's shape is pure and cheap to read here: a value that can never be a person
        // (`unknown`, a service identity) is refused before any provider or token write lands,
        // rather than after them on the most likely typo.
        if let Some(owner) = &parsed.owner {
            crate::users::normalize_telegram_binding(owner)
                .map_err(|e| anyhow::anyhow!("usage: \"owner\": {e:#}"))?;
        }
        Ok(parsed)
    }
}

/// The model provider as a run left it — one state, not three independent flags, since the
/// endpoint's kind is what decides which key is the chat credential.
#[derive(Debug)]
enum ProviderState {
    /// Nothing configured: no OpenRouter key and no custom endpoint.
    NotConfigured,
    /// OpenRouter itself (the default endpoint). Reaching this state needs a saved key, so it
    /// carries none of its own.
    OpenRouter,
    /// A custom OpenAI-compatible endpoint, with or without a key of its own. The OpenRouter key
    /// is tracked beside it because media generation and editing ride that account whatever the
    /// chat endpoint is.
    Custom { key: bool, openrouter_key: bool },
}

impl ProviderState {
    /// Whether an OpenRouter key is saved, on either kind of endpoint.
    fn openrouter_key(&self) -> bool {
        match self {
            Self::NotConfigured => false,
            Self::OpenRouter => true,
            Self::Custom { openrouter_key, .. } => *openrouter_key,
        }
    }
}

/// The owner's Telegram binding as the run could read it back — three outcomes, because "no
/// binding" and "the store could not be read" are not the same answer and must not be shown as one.
#[derive(Debug)]
enum OwnerBinding {
    /// The admin has no Telegram binding.
    None,
    /// The binding, described for a person.
    Bound(String),
    /// The store could not be read back after the run's writes; those writes are unaffected.
    Unreadable(String),
}

/// What one run did and found, rendered by [`Report::render`]. Plain data, so the report's wording
/// is testable without a store, a provider or a network.
#[derive(Debug)]
struct Report {
    /// One line per change this run made, in the order it made them.
    changed: Vec<String>,
    /// Non-fatal notes: an endpoint unreachable right now, a key that did not answer, a bot whose
    /// identity could not be read.
    warnings: Vec<String>,
    /// The step a run stopped at, when one failed — the changes above it still landed.
    stopped: Option<String>,
    /// The provider after this run.
    provider: ProviderState,
    /// The endpoint in force after this run.
    endpoint: String,
    /// Whether a bot token is saved (whether or not this run named it).
    bot_token_saved: bool,
    /// The bot's identity, when this run checked it.
    bot: Option<String>,
    /// The owner's Telegram binding as the store holds it — or why it could not be read back.
    owner: OwnerBinding,
    /// The desktop onboarding state after this run.
    onboarding: OnboardingState,
}

impl Report {
    fn provider_line(&self) -> String {
        let detail = match &self.provider {
            ProviderState::NotConfigured => return "  provider: not configured".to_string(),
            ProviderState::OpenRouter => "OpenRouter, key saved",
            ProviderState::Custom {
                key,
                openrouter_key,
            } => match (key, openrouter_key) {
                (true, true) => {
                    "custom endpoint, its own key saved, plus an OpenRouter key for media"
                }
                (false, true) => "custom endpoint, keyless, plus an OpenRouter key for media",
                (true, false) => "custom endpoint, its own key saved",
                (false, false) => "custom endpoint, keyless",
            },
        };
        format!("  provider: configured — {} ({detail})", self.endpoint)
    }

    fn bot_line(&self) -> String {
        match (&self.bot, self.bot_token_saved) {
            (Some(identity), _) => format!("  telegram bot: {identity}"),
            (None, true) => "  telegram bot: a token is saved; its identity is checked only by a \
                             run that saves it"
                .to_string(),
            (None, false) => "  telegram bot: no token saved".to_string(),
        }
    }

    fn owner_line(&self) -> String {
        match &self.owner {
            OwnerBinding::None => "  owner: no Telegram binding".to_string(),
            OwnerBinding::Bound(binding) => format!("  owner: {binding}"),
            OwnerBinding::Unreadable(e) => format!("  owner: could not be read back ({e})"),
        }
    }

    fn onboarding_line(&self) -> String {
        let state = match self.onboarding {
            OnboardingState::Finished => "finished",
            OnboardingState::Init => "still open — a provider credential finishes it",
        };
        format!("  onboarding: {state}")
    }

    /// The warnings, plus the one gap that is not this run's failure: media generation and editing
    /// ride the OpenRouter account, so a custom endpoint without an OpenRouter key leaves them
    /// unserved while chat keeps working.
    fn notes(&self) -> Vec<String> {
        let mut notes = self.warnings.clone();
        if crate::config::is_custom_endpoint(&self.endpoint) && !self.provider.openrouter_key() {
            notes.push(
                "This instance uses a custom endpoint and no OpenRouter key is saved: media \
                 generation and editing need an OpenRouter key."
                    .to_string(),
            );
        }
        notes
    }

    /// What the caller does next. The last step is always the one the product cannot do for
    /// itself: Telegram forbids a bot to open a conversation, so nothing arrives until the owner
    /// writes first.
    fn next_steps(&self) -> Vec<String> {
        let mut steps = Vec::new();
        if matches!(self.provider, ProviderState::NotConfigured) {
            steps.push(
                "Set the model provider: mahbot config setup --provider-key @key.txt for an \
                 OpenRouter key, or --provider-endpoint URL [--provider-endpoint-key @key.txt] for \
                 a custom OpenAI-compatible endpoint."
                    .to_string(),
            );
        }
        if !self.bot_token_saved {
            steps.push(
                "Save the Telegram bot token from BotFather: mahbot config setup --telegram-token \
                 @token.txt."
                    .to_string(),
            );
        }
        if matches!(self.owner, OwnerBinding::None) {
            steps.push(
                "Bind your own Telegram account as the owner: mahbot config setup --owner \
                 ./@yourname (or --owner <numeric id>)."
                    .to_string(),
            );
        }
        if matches!(self.owner, OwnerBinding::Unreadable(_)) {
            steps.push(
                "Read the owner binding back with `mahbot config setup` (no arguments) — this run \
                 could not read it after its writes."
                    .to_string(),
            );
        }
        steps.push(format!(
            "Send /start to {} from the owner's Telegram account — a bot cannot start the \
             conversation itself, so nothing reaches you until that first message.",
            self.bot.as_deref().unwrap_or("the bot")
        ));
        steps
    }

    fn render(&self) -> String {
        let mut out = String::from("Changed:\n");
        if self.changed.is_empty() {
            out.push_str("  (nothing — this run changed nothing)\n");
        } else {
            for change in &self.changed {
                let _ = writeln!(out, "  - {change}");
            }
        }

        if let Some(stopped) = &self.stopped {
            out.push_str("\nStopped:\n");
            let _ = writeln!(out, "  - {stopped}");
            if self.changed.is_empty() {
                out.push_str("  Nothing was written before this step.\n");
            } else {
                out.push_str(
                    "  The changes listed above were applied; the run ended at this step.\n",
                );
            }
        }

        out.push_str("\nCurrent state:\n");
        let _ = writeln!(out, "{}", self.provider_line());
        let _ = writeln!(out, "{}", self.bot_line());
        let _ = writeln!(out, "{}", self.owner_line());
        let _ = writeln!(out, "{}", self.onboarding_line());

        let notes = self.notes();
        if !notes.is_empty() {
            out.push_str("\nWarnings:\n");
            for note in &notes {
                let _ = writeln!(out, "  - {note}");
            }
        }

        out.push_str("\nNext:\n");
        for step in self.next_steps() {
            let _ = writeln!(out, "  - {step}");
        }
        out
    }
}

/// What a run says about the owner binding it just made. A number is a real delivery address from
/// the start; a nickname is only that account's claim until the account writes to the bot — that
/// first message is what proves it, and the bot's reply there confirms it.
fn owner_binding_line(identifier: &str) -> String {
    let described = describe_telegram_binding(identifier);
    if crate::users::telegram_identifier_is_number(identifier) {
        format!("owner bound to {described}")
    } else {
        format!(
            "owner bound to {described} — accepted; the binding is confirmed at the first message \
             from that account, and the bot's reply there says so"
        )
    }
}

/// Whether OpenRouter answers the key: one authenticated `GET /key`. `Ok(true)` when it answered
/// at all (any status but 401/403 — the check is whether the key is known, not whether it has
/// credit), `Ok(false)` when the platform refused the key, `Err` when nothing answered.
async fn openrouter_key_answers(key: &str) -> anyhow::Result<bool> {
    let client = crate::util::http::build_http_client(KEY_CHECK_TIMEOUT);
    let response = client
        .get(OPENROUTER_KEY_CHECK)
        .bearer_auth(key)
        .send()
        .await
        .context("OpenRouter could not be reached")?;
    let status = response.status();
    Ok(!(status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN))
}

/// Save the OpenRouter key and check it live.
///
/// Saved first and checked after: the window never rejects a key (a provider outage is no reason to
/// refuse the owner's only way in), so a failed check is a warning on a saved key, naming what will
/// not work until a working one is in place.
async fn save_openrouter_key(
    key: &str,
    changed: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> anyhow::Result<()> {
    // A key already in force is no write and no probe: the live check would warn about a key the
    // report says this run did not touch.
    if CONFIG.provider_key().as_deref() == Some(key) {
        return Ok(());
    }
    crate::config::persist_settled_string_field(CONFIG_KEY_PROVIDER_KEY, key).await?;
    changed.push("OpenRouter key saved".to_string());
    match openrouter_key_answers(key).await {
        Ok(true) => {}
        Ok(false) => warnings.push(
            "The key was saved, but OpenRouter does not accept it: the Assistant cannot answer \
             until a working key is saved."
                .to_string(),
        ),
        Err(e) => warnings.push(format!(
            "The key was saved, but OpenRouter could not be reached to check it ({e:#}): the \
             Assistant cannot answer until a working key is saved."
        )),
    }
    Ok(())
}

/// Persist the provider arguments. The endpoint arm is the live reachability check for a custom
/// endpoint — persisting it runs the provider warmup, keyless included, where reaching it is all
/// there is to prove — and its warning is the window's own wording, reused rather than re-judged
/// here. A write that leaves the value in force where it was is not reported as a change: the
/// header promises one line per change the run made.
async fn apply_provider(
    args: &SetupArgs,
    changed: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> anyhow::Result<()> {
    if let Some(endpoint) = &args.provider_endpoint {
        // The endpoint before its key: it is validated before it is written, so a URL the config
        // refuses stops the run with nothing saved, rather than leaving an orphan key row whose own
        // line the report already claimed. A keyless custom endpoint is a state this module
        // reports, so a refused key after a good endpoint is no worse. A key beside OpenRouter's
        // own address is refused at parse time, since settling that endpoint drops the key row. A
        // value already in force is no write, like the OpenRouter key below.
        let before = crate::config::effective_chat_endpoint(&CONFIG.snapshot());
        if before != endpoint.trim() {
            // Persisting it runs the provider warmup — a custom endpoint that is down right now
            // warns here, which is the one thing this flag can prove on its own.
            let outcome =
                crate::config::persist_settled_string_field(CONFIG_KEY_PROVIDER_ENDPOINT, endpoint)
                    .await?;
            let after = crate::config::effective_chat_endpoint(&CONFIG.snapshot());
            if after != before {
                changed.push(format!("provider endpoint set to {after}"));
            }
            warnings.extend(outcome.warning);
        }
        if let Some(key) = &args.provider_endpoint_key
            && CONFIG.provider_endpoint_key().as_deref() != Some(key.as_str())
        {
            crate::config::persist_settled_string_field(CONFIG_KEY_PROVIDER_ENDPOINT_KEY, key)
                .await?;
            changed.push("custom endpoint key saved".to_string());
        }
    }
    // The OpenRouter key is its own account, never the chat credential of a custom endpoint: media
    // generation and editing ride it whatever endpoint chat uses, so a key given beside a custom
    // endpoint is saved too.
    if let Some(key) = &args.provider_key {
        save_openrouter_key(key, changed, warnings).await?;
    }
    Ok(())
}

/// Save the bot token and name the bot it belongs to. The save carries the window's own live check
/// — a token that CHANGES is validated, an unchanged one is no write and no probe, the way the
/// provider flags answer a re-run — and that check answers with the bot's identity, so naming it
/// costs no second call. A local Bot API server that is not up yet leaves the token unchecked: a
/// save the listener will pick up regardless is not a reason to fail.
async fn apply_telegram_token(
    token: &str,
    changed: &mut Vec<String>,
    warnings: &mut Vec<String>,
    bot: &mut Option<String>,
) -> anyhow::Result<()> {
    if CONFIG.telegram_bot_token().as_deref() == Some(token) {
        return Ok(());
    }
    *bot = crate::config::persist_telegram_bot_token(token).await?;
    changed.push("Telegram bot token saved".to_string());
    if bot.is_none() {
        warnings.push(
            "The Telegram bot token was saved, but its identity could not be checked now: the \
             local Bot API server has no endpoint yet, so the bot cannot be named."
                .to_string(),
        );
    }
    Ok(())
}

/// Every write one run makes, in order, each recorded as it lands. A failing step ends the run
/// where it is; what already landed stays recorded, and the report says both.
async fn apply(
    args: &SetupArgs,
    changed: &mut Vec<String>,
    warnings: &mut Vec<String>,
    bot: &mut Option<String>,
) -> anyhow::Result<()> {
    apply_provider(args, changed, warnings).await?;

    if let Some(token) = &args.telegram_token {
        apply_telegram_token(token, changed, warnings, bot).await?;
    }

    if args.unbind_owner && crate::users::store().unbind_telegram_for_admin().await? {
        changed.push("owner Telegram binding removed".to_string());
    }
    if let Some(value) = &args.owner {
        // The store says whether it wrote: the value already in place writes nothing, and the one
        // repair that does (clearing a legacy address) is a change the report must name.
        let rebound = crate::users::store()
            .rebind_telegram_for_admin(value)
            .await?;
        if rebound.changed {
            changed.push(owner_binding_line(&rebound.identifier));
        }
    }

    // The desktop flow's one-shot transition, through the same writer: a configured provider is
    // what finishes onboarding, and this run may be what configured it.
    if crate::config::provider_configured() && CONFIG.onboarding_stage() == OnboardingState::Init {
        crate::onboarding::finish_onboarding().await?;
        changed.push("onboarding finished".to_string());
    }
    Ok(())
}

/// Run the setup for the arguments given and render what it did.
///
/// Terminal-only: this is the action behind `mahbot config setup`, and it refuses any other caller
/// so the admin Assistant cannot write the provider credential.
pub(crate) async fn run(args: &serde_json::Value) -> anyhow::Result<String> {
    anyhow::ensure!(
        crate::agent::tool_command_line(),
        "forbidden: `setup` is the terminal setup command's action (`mahbot config setup`) — the \
         Assistant must not use it"
    );
    let args = SetupArgs::parse(args)?;

    let mut changed: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut bot = None;
    let stopped = apply(&args, &mut changed, &mut warnings, &mut bot)
        .await
        .err()
        .map(|e| format!("{e:#}"));

    let config = CONFIG.snapshot();
    let endpoint = crate::config::effective_chat_endpoint(&config);
    let openrouter_key = config.provider_key.is_some();
    let provider = if !crate::config::provider_configured() {
        ProviderState::NotConfigured
    } else if crate::config::is_custom_endpoint(&endpoint) {
        ProviderState::Custom {
            key: config.provider_endpoint_key.is_some(),
            openrouter_key,
        }
    } else {
        ProviderState::OpenRouter
    };
    let report = Report {
        changed,
        warnings,
        stopped,
        provider,
        endpoint,
        bot_token_saved: config.telegram_bot_token.is_some(),
        bot,
        // Read back after the writes, and never with `?`: a store failure here must not replace the
        // report of what already landed, and it is its own answer rather than a missing binding.
        owner: match crate::users::store()
            .telegram_binding(ADMIN_USER_NAME)
            .await
        {
            Ok(Some(identifier)) => OwnerBinding::Bound(describe_telegram_binding(&identifier)),
            Ok(None) => OwnerBinding::None,
            Err(e) => OwnerBinding::Unreadable(format!("{e:#}")),
        },
        onboarding: CONFIG.onboarding_stage(),
    };
    if report.stopped.is_some() {
        // The rendered report is the failure: a caller must see what landed, and exit non-zero.
        anyhow::bail!("{}", report.render());
    }
    Ok(report.render())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A report as a run leaves it with nothing to warn about, so a test names only the fields it
    /// is about.
    fn report() -> Report {
        Report {
            changed: Vec::new(),
            warnings: Vec::new(),
            stopped: None,
            provider: ProviderState::OpenRouter,
            endpoint: "https://openrouter.ai/api/v1".to_string(),
            bot_token_saved: true,
            bot: None,
            owner: OwnerBinding::Bound("@owner".to_string()),
            onboarding: OnboardingState::Finished,
        }
    }

    /// The one guard that makes this action terminal-only: outside a command-line call the run
    /// stops before any store, any network or any write.
    #[tokio::test]
    async fn setup_refuses_outside_a_command_line_call() {
        let err = run(&json!({}))
            .await
            .expect_err("the Assistant must not reach the setup action");
        let message = err.to_string();
        assert!(
            message.contains("terminal setup command") && message.contains("Assistant"),
            "the refusal must say whose action this is: {message}"
        );
    }

    /// The command line's flags read into exactly the fields the run acts on; the pairs that cannot
    /// be meant together are refused instead of resolved by an invisible order.
    #[test]
    fn argument_parsing_reads_the_terminal_flags() {
        assert_eq!(SetupArgs::parse(&json!({})).unwrap(), SetupArgs::default());

        let parsed = SetupArgs::parse(&json!({
            "provider_key": " sk-or-v1-abc ",
            "telegram_token": "123:abc",
            "owner": "@someone",
        }))
        .unwrap();
        assert_eq!(parsed.provider_key.as_deref(), Some("sk-or-v1-abc"));
        assert_eq!(parsed.telegram_token.as_deref(), Some("123:abc"));
        assert_eq!(parsed.owner.as_deref(), Some("@someone"));
        assert!(!parsed.unbind_owner);

        assert!(
            SetupArgs::parse(&json!({
                "provider_endpoint": "https://llm.example/v1",
                "provider_endpoint_key": "local-key",
                "unbind_owner": true,
            }))
            .unwrap()
            .unbind_owner
        );

        // A blank secret is refused, not read as absent.
        let err = SetupArgs::parse(&json!({ "provider_key": "   " }))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("usage:") && err.contains("provider_key"),
            "{err}"
        );

        // A custom endpoint's key names nothing without its endpoint.
        let err = SetupArgs::parse(&json!({ "provider_endpoint_key": "k" }))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("usage:") && err.contains("provider_endpoint_key"),
            "{err}"
        );

        // Nor beside OpenRouter's own address: settling that endpoint drops the key row, so the
        // run would report a value that no longer exists.
        let err = SetupArgs::parse(&json!({
            "provider_endpoint": "https://openrouter.ai/api/v1",
            "provider_endpoint_key": "k",
        }))
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("usage:") && err.contains("provider_endpoint_key"),
            "{err}"
        );

        // Binding and unbinding the owner are one decision.
        let err = SetupArgs::parse(&json!({ "owner": "@a", "unbind_owner": true }))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("usage:") && err.contains("unbind_owner"),
            "{err}"
        );

        // An owner that can never be a person is refused here, before any write lands.
        let err = SetupArgs::parse(&json!({ "owner": "unknown" }))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("usage:") && err.contains("owner"),
            "the owner's shape must be read with the other usage rules: {err}"
        );
    }

    /// A run that stopped partway says both halves: what landed, and the step it ended at — a
    /// failure must never hide a change that was already made.
    #[test]
    fn a_stopped_run_reports_what_landed_before_it() {
        let stopped = Report {
            changed: vec!["provider endpoint set to https://llm.example/v1".to_string()],
            stopped: Some("Telegram rejected the bot token — Unauthorized".to_string()),
            ..report()
        };
        let rendered = stopped.render();
        assert!(rendered.contains("Stopped:"), "{rendered}");
        assert!(
            rendered.contains("Telegram rejected the bot token"),
            "{rendered}"
        );
        assert!(
            rendered.contains("provider endpoint set to"),
            "the change that landed must still be reported: {rendered}"
        );
        assert!(
            rendered.contains("The changes listed above were applied"),
            "{rendered}"
        );

        // A run that stopped before writing anything must not claim changes that were applied.
        let empty = Report {
            stopped: Some("Telegram rejected the bot token — Unauthorized".to_string()),
            ..report()
        }
        .render();
        assert!(
            !empty.contains("changes listed above were applied"),
            "{empty}"
        );
        assert!(
            empty.contains("Nothing was written before this step"),
            "{empty}"
        );
    }

    /// The terminal command's flags are not a list of its own to drift from the parser: every
    /// `mahbot config setup` flag is an argument [`SetupArgs::parse`] reads, and it reads no
    /// argument the command line cannot fill.
    #[test]
    fn the_terminal_flags_are_exactly_what_the_parser_reads() {
        let operation = crate::cli::FAMILIES
            .iter()
            .find(|family| family.tool == "mahbot_config")
            .and_then(|family| match family.surface {
                crate::cli::Surface::Operations(operations) => operations
                    .iter()
                    .find(|operation| operation.action == Some(ACTION)),
                crate::cli::Surface::Standalone(_) => None,
            })
            .expect("the command line exposes 'mahbot config setup'");
        let empty = SetupArgs::parse(&json!({})).unwrap();
        let probes = [
            ("provider_key", json!("sk-or-v1-x"), json!({})),
            (
                "provider_endpoint",
                json!("https://llm.example/v1"),
                json!({}),
            ),
            (
                "provider_endpoint_key",
                json!("local-key"),
                json!({ "provider_endpoint": "https://llm.example/v1" }),
            ),
            ("telegram_token", json!("1:A"), json!({})),
            ("owner", json!("12345"), json!({})),
            ("unbind_owner", json!(true), json!({})),
        ];
        for (name, value, beside) in &probes {
            let mut args = beside.clone();
            args[*name] = value.clone();
            let parsed = SetupArgs::parse(&args)
                .unwrap_or_else(|e| panic!("the parser must read '{name}': {e}"));
            assert_ne!(parsed, empty, "the parser ignores '{name}'");
        }
        // One comparison both ways: a flag the parser does not read, an argument the command line
        // cannot fill, and a repeated flag all fail here.
        let mut declared: Vec<&str> = operation.flags.iter().map(|flag| flag.arg).collect();
        let mut read: Vec<&str> = probes.iter().map(|(name, _, _)| *name).collect();
        declared.sort_unstable();
        read.sort_unstable();
        assert_eq!(
            declared, read,
            "the command line's flags must be exactly the arguments the parser reads"
        );
    }

    /// A run that names nothing says so, and still answers with the state and the one step the
    /// product cannot take for itself.
    #[test]
    fn a_run_that_names_nothing_changes_nothing_and_still_advises() {
        let rendered = report().render();
        assert!(
            rendered.contains("(nothing — this run changed nothing)"),
            "{rendered}"
        );
        assert!(rendered.contains("provider: configured"), "{rendered}");
        assert!(rendered.contains("owner: @owner"), "{rendered}");
        assert!(
            rendered.contains("a bot cannot start the conversation itself"),
            "{rendered}"
        );
        assert!(!rendered.contains("Warnings:"), "{rendered}");
    }

    /// Media ride the OpenRouter account: a custom endpoint without an OpenRouter key leaves them
    /// unserved, and the report says so once — with a key saved, or on OpenRouter itself, it does
    /// not.
    #[test]
    fn the_report_names_the_media_gap_only_for_a_keyless_custom_endpoint() {
        let keyless = Report {
            endpoint: "https://llm.example/v1".to_string(),
            provider: ProviderState::Custom {
                key: false,
                openrouter_key: false,
            },
            ..report()
        };
        let rendered = keyless.render();
        assert!(
            rendered.contains("media generation and editing need an OpenRouter key"),
            "{rendered}"
        );
        assert!(
            rendered.contains("custom endpoint, keyless"),
            "the state must say the endpoint is keyless: {rendered}"
        );

        let keyed = Report {
            endpoint: "https://llm.example/v1".to_string(),
            provider: ProviderState::Custom {
                key: true,
                openrouter_key: true,
            },
            ..report()
        };
        assert!(
            !keyed
                .render()
                .contains("media generation and editing need an OpenRouter key")
        );

        assert!(
            !report()
                .render()
                .contains("media generation and editing need an OpenRouter key"),
            "the default endpoint IS OpenRouter"
        );
    }

    /// A binding the store could not be read back is neither "no binding" nor a state: it says so
    /// where the binding would be, and names the run that reads it again.
    #[test]
    fn an_unreadable_owner_is_not_reported_as_absent() {
        let unreadable = Report {
            owner: OwnerBinding::Unreadable("disk i/o error".to_string()),
            ..report()
        }
        .render();
        assert!(
            unreadable.contains("owner: could not be read back (disk i/o error)"),
            "{unreadable}"
        );
        assert!(
            !unreadable.contains("owner: no Telegram binding"),
            "{unreadable}"
        );
        assert!(
            !unreadable.contains("Bind your own Telegram account"),
            "a failed read is not a missing binding: {unreadable}"
        );
        assert!(
            unreadable.contains("Read the owner binding back"),
            "{unreadable}"
        );
    }

    /// A provider that is not configured names how to set it, and a missing bot or owner names its
    /// own step — each one a command line the caller can paste.
    #[test]
    fn missing_pieces_each_name_their_own_command() {
        let bare = Report {
            provider: ProviderState::NotConfigured,
            bot_token_saved: false,
            owner: OwnerBinding::None,
            ..report()
        };
        let rendered = bare.render();
        assert!(rendered.contains("provider: not configured"), "{rendered}");
        assert!(
            rendered.contains("mahbot config setup --provider-key @key.txt"),
            "{rendered}"
        );
        assert!(
            rendered.contains("mahbot config setup --telegram-token @token.txt"),
            "{rendered}"
        );
        assert!(
            rendered.contains("mahbot config setup --owner ./@yourname"),
            "{rendered}"
        );

        // A saved token whose identity this run did not check is reported as exactly that.
        let unchecked = Report {
            bot: None,
            ..report()
        };
        assert!(
            unchecked
                .render()
                .contains("a token is saved; its identity is checked only by a run that saves it"),
            "{}",
            unchecked.render()
        );
    }

    /// A number is a delivery address at once; a nickname is only confirmed by that account's first
    /// message, and the report says which of the two it just bound.
    #[test]
    fn an_owner_binding_says_whether_it_still_needs_the_first_message() {
        assert_eq!(owner_binding_line("123456"), "owner bound to id 123456");
        // The identifier arrives normalized — no `@` — and is described back with one.
        let nickname = owner_binding_line("alice");
        assert!(nickname.starts_with("owner bound to @alice"), "{nickname}");
        assert!(
            nickname.contains("confirmed at the first message"),
            "{nickname}"
        );
    }
}
