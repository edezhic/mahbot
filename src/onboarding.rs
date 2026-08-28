//! Onboarding flow: the Phase-1 scripted provider-setup scenario (GUI Home
//! chat) and the Phase-2 Support kickoff ("hi mah bot" auto-send).
//!
//! Phase 1 runs in the GUI Home chat while no LLM provider is configured
//! (`provider_configured() == false`): a hard-coded scripted scenario guides
//! the admin to enter an OpenRouter key or a custom endpoint+key, persisting
//! it via the public config path so the live in-memory CONFIG updates. The
//! scripted exchange is transient (never persisted to chat_history).
//!
//! Phase 2 fires once a provider is configured and the onboarding state is
//! `Init`: the admin's active role is set to Support and a single real
//! "hi mah bot" user message is routed through the normal pipeline.

use crate::{ChannelMessage, Role};
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-local guard ensuring `kickoff_support` fires exactly once per
/// process lifetime, closing the narrow window where two concurrent Home
/// renders both read `Init` before either persists `Welcomed`. The persisted
/// `Welcomed` state is the durable guard across restarts; this only covers the
/// in-process race.
static KICKOFF_FIRED: AtomicBool = AtomicBool::new(false);

/// Parsed Phase-1 provider input.
#[derive(Debug)]
pub enum ProviderInput {
    /// An OpenRouter API key (a token containing `sk-or-v1-`).
    OpenRouterKey(String),
    /// A custom endpoint URL, with an optional per-endpoint key.
    CustomEndpoint { url: String, key: Option<String> },
    /// Input that is neither a recognized key nor a URL.
    Invalid,
}

/// Parse the user's Phase-1 provider entry.
///
/// Rules:
/// - OpenRouter key = a token containing `sk-or-v1-…`.
/// - Custom endpoint = URL on line 1, optional key on line 2.
/// - A bare URL alone (keyless custom endpoint) counts as configured.
#[must_use]
pub fn parse_provider_input(raw: &str) -> ProviderInput {
    let trimmed = raw.trim();
    let lines: Vec<&str> = trimmed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() >= 2 && is_http_url(lines[0]) {
        let key = lines[1].to_string();
        // A default OpenRouter endpoint + a `sk-or-v1-` key is an OpenRouter key:
        // the user meant "use OpenRouter", not a custom endpoint.
        if crate::config::is_default_endpoint(lines[0]) && key.contains("sk-or-v1-") {
            return ProviderInput::OpenRouterKey(key);
        }
        // The default endpoint is not a custom endpoint — a bare default URL (or
        // default URL with a non-OpenRouter key) can't configure the provider, so
        // reject it rather than persisting something that never becomes configured.
        if crate::config::is_default_endpoint(lines[0]) {
            return ProviderInput::Invalid;
        }
        if !is_clean_url_line(lines[0]) {
            return ProviderInput::Invalid;
        }
        return ProviderInput::CustomEndpoint {
            url: lines[0].to_string(),
            key: Some(key),
        };
    }
    if lines.len() == 1 && is_http_url(lines[0]) {
        if crate::config::is_default_endpoint(lines[0]) {
            return ProviderInput::Invalid;
        }
        // A URL can't contain a literal space — a single line with trailing junk
        // (e.g. a key pasted on the same line) is a malformed entry outside the
        // documented two-line grammar; reject it rather than persisting a broken
        // endpoint that would report as configured.
        if !is_clean_url_line(lines[0]) {
            return ProviderInput::Invalid;
        }
        return ProviderInput::CustomEndpoint {
            url: lines[0].to_string(),
            key: None,
        };
    }
    // A line carrying a `sk-or-v1-` token is an OpenRouter key. Extract the key
    // line rather than the whole input, so a reversed entry (key line, then URL)
    // never persists a multi-line blob as the provider key.
    if let Some(key_line) = lines.iter().find(|l| l.contains("sk-or-v1-")) {
        return ProviderInput::OpenRouterKey(key_line.to_string());
    }
    ProviderInput::Invalid
}

fn is_http_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// A URL cannot contain a literal space, so a candidate URL line must be a
/// single whitespace-delimited token. A line with trailing junk (e.g. a key
/// pasted on the same line as the URL) is malformed.
fn is_clean_url_line(s: &str) -> bool {
    s.split_whitespace().count() == 1
}

/// Persist a parsed provider input via the public config path (updates the
/// live in-memory CONFIG and the `config_kv` row). Returns an error for
/// [`ProviderInput::Invalid`].
pub async fn persist_provider_input(input: &ProviderInput) -> anyhow::Result<()> {
    match input {
        ProviderInput::OpenRouterKey(key) => {
            crate::config::persist_settled_string_field("provider_key", key).await?;
        }
        ProviderInput::CustomEndpoint { url, key } => {
            if let Some(k) = key {
                crate::config::persist_settled_string_field("provider_endpoint_key", k).await?;
            }
            // Persist the endpoint LAST — it is the gating field for
            // `provider_configured()`, so a failure here leaves the provider
            // unconfigured and the script re-prompts instead of proceeding with a
            // keyless endpoint the user did not intend.
            crate::config::persist_settled_string_field("provider_endpoint", url).await?;
        }
        ProviderInput::Invalid => anyhow::bail!("invalid provider input"),
    }
    Ok(())
}

/// Phase-1 scripted intro messages (shown in the GUI Home chat).
#[must_use]
pub fn intro_messages() -> [&'static str; 2] {
    [
        "Welcome to MahBot! Before we start, I need an LLM provider to power your agents.",
        "Enter your OpenRouter API key (it starts with `sk-or-v1-`), or paste a custom endpoint. For a custom endpoint, put the URL on the FIRST line and your API key on the SECOND line (the key is optional).",
    ]
}

/// Phase-1 success message after the provider is configured.
#[must_use]
pub fn success_message() -> &'static str {
    "Provider configured! Setting up your Support assistant — one moment."
}

/// Phase-1 re-prompt for unparseable input.
#[must_use]
pub fn invalid_message() -> &'static str {
    "That doesn't look right. Enter an OpenRouter key (starts with `sk-or-v1-`) or a custom endpoint URL (first line) + optional API key (second line)."
}

/// Phase-2 kickoff: if onboarding state is `Init` and a provider is
/// configured, mark it `Welcomed` (the idempotency guard), set the admin's
/// active role to Support, and auto-send a single real "hi mah bot" user
/// message through the normal GUI pipeline. No-op when the state is already
/// `Welcomed`/`Finished`, when no provider is configured, or when the user
/// has no Support in their role pool.
pub async fn kickoff_support(user_name: &str) -> anyhow::Result<()> {
    use crate::config::OnboardingState;
    if crate::config::CONFIG.onboarding_stage() != OnboardingState::Init {
        return Ok(());
    }
    if !crate::config::provider_configured() {
        return Ok(());
    }
    // The Support/onboarding flow is admin-only: bail before touching the
    // global state when the user's pool has no Support. This keeps a non-admin
    // (created via the Settings bypass pre-provider) from consuming the
    // onboarding state or emitting the auto-message.
    let pool = crate::users::role_pool(user_name).await;
    if !pool.contains(&Role::Support) {
        return Ok(());
    }
    // Compare-and-set: only one kickoff fires per process lifetime.
    if KICKOFF_FIRED.swap(true, Ordering::SeqCst) {
        return Ok(());
    }
    // Switch the active role BEFORE persisting `Welcomed`: if the role switch
    // fails the durable state is still `Init`, so the next render retries
    // (role-setting is idempotent). Persisting first would strand the state at
    // `Welcomed` with `hi mah bot` never sent.
    if let Err(e) = crate::users::switch_active_role(user_name, Role::Support).await {
        KICKOFF_FIRED.store(false, Ordering::SeqCst);
        return Err(e);
    }
    // Mark Welcomed — the durable guard against double-fire on reload/render.
    // If the persist fails, reset the in-process CAS so a later render can
    // retry (the durable `Init` state is unchanged; the role stays Support).
    if let Err(e) = crate::config::persist_settled_string_field(
        crate::config::CONFIG_KEY_ONBOARDING_STATE,
        OnboardingState::Welcomed.as_str(),
    )
    .await
    {
        KICKOFF_FIRED.store(false, Ordering::SeqCst);
        return Err(e);
    }
    let msg = ChannelMessage {
        user_name: user_name.to_string(),
        reply_target: user_name.to_string(),
        content: "hi mah bot".to_string(),
        channel: "gui".to_string(),
        workspace: format!("personal:{user_name}"),
        optimistic_id: None,
        callback_query_id: None,
    };
    if let Some(tx) = crate::GUI_MESSAGE_TX.get()
        && let Err(e) = tx.send(msg)
    {
        // The state is already `Welcomed`; the message is best-effort.
        tracing::error!("kickoff: failed to send 'hi mah bot' via GUI_MESSAGE_TX: {e}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openrouter_key() {
        // A token containing sk-or-v1- is an OpenRouter key.
        match parse_provider_input(" sk-or-v1-abc123def456 ") {
            ProviderInput::OpenRouterKey(k) => assert_eq!(k, "sk-or-v1-abc123def456"),
            other => panic!("expected OpenRouterKey, got {other:?}"),
        }
    }

    #[test]
    fn parses_keyless_custom_endpoint_url() {
        match parse_provider_input("https://ollama.local:11434/v1") {
            ProviderInput::CustomEndpoint { url, key } => {
                assert_eq!(url, "https://ollama.local:11434/v1");
                assert_eq!(key, None);
            }
            other => panic!("expected keyless CustomEndpoint, got {other:?}"),
        }
    }

    #[test]
    fn parses_two_line_custom_endpoint_with_key() {
        match parse_provider_input("http://localhost:8080/v1\nsk-local-123") {
            ProviderInput::CustomEndpoint { url, key } => {
                assert_eq!(url, "http://localhost:8080/v1");
                assert_eq!(key.as_deref(), Some("sk-local-123"));
            }
            other => panic!("expected CustomEndpoint with key, got {other:?}"),
        }
    }

    #[test]
    fn parses_default_openrouter_url_with_key_as_openrouter() {
        // The default OpenRouter endpoint on line 1 + `sk-or-v1-` key on line 2
        // is an OpenRouter key, not a custom endpoint (a default endpoint never
        // makes `provider_configured()` true, which would re-prompt forever).
        match parse_provider_input("https://openrouter.ai/api/v1\nsk-or-v1-abc123def456") {
            ProviderInput::OpenRouterKey(k) => assert_eq!(k, "sk-or-v1-abc123def456"),
            other => panic!("expected OpenRouterKey, got {other:?}"),
        }
    }

    #[test]
    fn parses_reversed_key_then_url_as_openrouter_key() {
        // Key on line 1, URL on line 2 (reversed from the documented order) — the
        // `sk-or-v1-` line is extracted instead of persisting the whole blob as
        // the provider key (which would falsely report `provider_configured()`).
        match parse_provider_input("sk-or-v1-abc123def456\nhttps://custom.example/v1") {
            ProviderInput::OpenRouterKey(k) => assert_eq!(k, "sk-or-v1-abc123def456"),
            other => panic!("expected OpenRouterKey, got {other:?}"),
        }
    }

    #[test]
    fn rejects_bare_default_openrouter_url() {
        // The default endpoint alone is not a custom endpoint: it never makes
        // `provider_configured()` true, so accept no key-less default URL.
        assert!(matches!(
            parse_provider_input("https://openrouter.ai/api/v1"),
            ProviderInput::Invalid
        ));
    }

    #[test]
    fn rejects_single_line_url_with_embedded_junk() {
        // A URL can't contain a literal space — a key pasted on the same line as
        // the URL is malformed (outside the documented two-line grammar) and must
        // not be persisted as a broken endpoint that reports as configured.
        assert!(matches!(
            parse_provider_input("https://custom.example/v1 sk-local-123"),
            ProviderInput::Invalid
        ));
        assert!(matches!(
            parse_provider_input("https://custom.example/v1 extra"),
            ProviderInput::Invalid
        ));
    }

    #[test]
    fn rejects_default_openrouter_url_with_unknown_key() {
        // Default endpoint + a non-`sk-or-v1-` key is not a usable provider either.
        assert!(matches!(
            parse_provider_input("https://openrouter.ai/api/v1\nmy-random-key"),
            ProviderInput::Invalid
        ));
    }

    #[test]
    fn rejects_unknown_input() {
        assert!(matches!(
            parse_provider_input("just some chat text"),
            ProviderInput::Invalid
        ));
    }
}
