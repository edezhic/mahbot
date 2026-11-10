//! Onboarding flow: the Phase-1 scripted provider-setup scenario (GUI Home chat) and the Phase-2
//! Assistant kickoff ("hi mah bot" auto-send).
//!
//! Phase 1 runs in the GUI Home chat while no LLM provider is configured: a hard-coded scripted
//! scenario guides the admin to enter an OpenRouter key or a custom endpoint+key, persisted through
//! the public config path so the live in-memory CONFIG updates. That exchange is transient (never
//! written to chat_history).
//!
//! Phase 2 fires once a provider is configured and the onboarding state is `Init` for an admin: the
//! kickoff persists `OnboardingState::Finished` (closing the flow) and routes a single real "hi mah
//! bot" user message through the normal pipeline. The Assistant drives onboarding now; the guide
//! reaches the greeting session through the pending-guide bridge, not the onboarding state.

use crate::ChannelMessage;
use crate::config::{
    CONFIG_KEY_PROVIDER_ENDPOINT, CONFIG_KEY_PROVIDER_ENDPOINT_KEY, CONFIG_KEY_PROVIDER_KEY,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-local CAS guard ensuring only one `kickoff_onboarding` attempt is in flight per process;
/// reset on a persist failure so a later attempt can retry. The persisted `Finished` state is the
/// durable guard across restarts.
static KICKOFF_FIRED: AtomicBool = AtomicBool::new(false);

/// Bridge over the greeting race: `kickoff_onboarding` persists `Finished` BEFORE the async
/// pipeline builds the session that handles the greeting (the greeting crosses many channel hops;
/// the persist is one fast `config_kv` upsert), so `build_context_messages` cannot rely on
/// `onboarding_stage() != Finished` alone to inject the guide. The kickoff arms this flag; the
/// first Assistant session built for the admin afterwards consumes it — in practice the greeting
/// session. If the greeting send fails, the next admin Assistant session receives the guide
/// instead.
static GREETING_GUIDE_PENDING: AtomicBool = AtomicBool::new(false);

/// Arm the pending-guide bridge (see [`GREETING_GUIDE_PENDING`]).
pub fn mark_greeting_guide_pending() {
    GREETING_GUIDE_PENDING.store(true, Ordering::SeqCst);
}

/// Consume the pending-guide bridge; true exactly once per kickoff.
pub fn take_greeting_guide_pending() -> bool {
    GREETING_GUIDE_PENDING.swap(false, Ordering::SeqCst)
}

#[derive(Debug)]
pub enum ProviderInput {
    /// An OpenRouter API key (a token containing `sk-or-v1-`).
    OpenRouterKey(String),
    /// A custom endpoint URL, with an optional per-endpoint key.
    CustomEndpoint { url: String, key: Option<String> },
    /// Input that is neither a recognized key nor a URL.
    Invalid,
}

/// Parse the user's Phase-1 provider entry: an OpenRouter key (a token containing `sk-or-v1-…`), a
/// custom endpoint (URL on line 1, optional key on line 2), or a bare URL alone (a keyless custom
/// endpoint, which counts as configured).
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
        // A default OpenRouter endpoint + a `sk-or-v1-` key is an OpenRouter key: the user meant
        // "use OpenRouter", not a custom endpoint.
        if crate::config::is_default_endpoint(lines[0]) && key.contains("sk-or-v1-") {
            return ProviderInput::OpenRouterKey(key);
        }
        return custom_endpoint(lines[0], Some(key));
    }
    if lines.len() == 1 && is_http_url(lines[0]) {
        return custom_endpoint(lines[0], None);
    }
    // A line carrying a `sk-or-v1-` token is an OpenRouter key. Extract the line rather than the
    // whole input, so a reversed entry (key first, then URL) never persists a multi-line blob as
    // the key.
    if let Some(key_line) = lines.iter().find(|l| l.contains("sk-or-v1-")) {
        return ProviderInput::OpenRouterKey(key_line.to_string());
    }
    ProviderInput::Invalid
}

/// Shared validation tail for both URL branches: reject the default endpoint (a bare default URL,
/// or one with a non-OpenRouter key, can never become configured) and malformed URL lines;
/// otherwise build the endpoint.
fn custom_endpoint(url_line: &str, key: Option<String>) -> ProviderInput {
    if crate::config::is_default_endpoint(url_line) {
        return ProviderInput::Invalid;
    }
    if !is_clean_url_line(url_line) {
        return ProviderInput::Invalid;
    }
    ProviderInput::CustomEndpoint {
        url: url_line.to_string(),
        key,
    }
}

// Deliberately case-insensitive, unlike the prefix-only `util::is_http_url` (whose case-sensitive
// contract config validation relies on) — onboarding accepts pasted endpoint URLs leniently
// regardless of scheme casing.
fn is_http_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// A URL cannot contain a literal space, so a candidate line must be a single whitespace-delimited
/// token. A line with trailing junk (a key pasted beside the URL) is malformed.
fn is_clean_url_line(s: &str) -> bool {
    s.split_whitespace().count() == 1
}

/// Persist a parsed provider input via the public config path (updating the live CONFIG and the
/// `config_kv` row). Errors for [`ProviderInput::Invalid`].
pub async fn persist_provider_input(input: &ProviderInput) -> anyhow::Result<()> {
    match input {
        ProviderInput::OpenRouterKey(key) => {
            crate::config::persist_settled_string_field(CONFIG_KEY_PROVIDER_KEY, key).await?;
        }
        ProviderInput::CustomEndpoint { url, key } => {
            if let Some(k) = key {
                crate::config::persist_settled_string_field(CONFIG_KEY_PROVIDER_ENDPOINT_KEY, k)
                    .await?;
            }
            // Persist the endpoint LAST — it is the gating field for `provider_configured()`, so a
            // failure here leaves the provider unconfigured and the script re-prompts instead of
            // proceeding with a keyless endpoint.
            crate::config::persist_settled_string_field(CONFIG_KEY_PROVIDER_ENDPOINT, url).await?;
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
    "Provider configured! Setting up your admin Assistant — one moment."
}

/// Phase-1 re-prompt for unparseable input.
#[must_use]
pub fn invalid_message() -> &'static str {
    "That doesn't look right. Enter an OpenRouter key (starts with `sk-or-v1-`) or a custom endpoint URL (first line) + optional API key (second line)."
}

/// Phase-2 kickoff: if the onboarding state is `Init` and a provider is configured, persist
/// `OnboardingState::Finished` and auto-send a single real "hi mah bot" user message through the
/// normal GUI pipeline. No-op when the state is not `Init`, no provider is configured, or the user
/// is not an admin. Onboarding is strictly one-shot: the `== Init` gate plus the `KICKOFF_FIRED`
/// CAS make a re-trigger impossible, even after a provider reconnection or a second workspace — the
/// state only moves `Init` → `Finished`, and only this route moves it. A guest never triggers
/// onboarding: the `is_admin` gate fails closed.
pub async fn kickoff_onboarding(user_name: &str) -> anyhow::Result<()> {
    use crate::config::OnboardingState;
    if crate::config::CONFIG.onboarding_stage() != OnboardingState::Init {
        return Ok(());
    }
    if !crate::config::provider_configured() {
        return Ok(());
    }
    // The onboarding flow is admin-only: bail before touching the
    // global state when the triggering user is not an admin (fails closed).
    // This keeps a guest (created via the Settings bypass pre-provider)
    // from consuming the onboarding state or emitting the auto-message.
    if !crate::users::is_admin(user_name).await {
        return Ok(());
    }
    // Compare-and-set: one kickoff attempt per process; reset below on a persist failure so a later
    // attempt retries.
    if KICKOFF_FIRED.swap(true, Ordering::SeqCst) {
        return Ok(());
    }
    // Persist `Finished` FIRST: if it fails, nothing has been sent and the
    // CAS is reset, so the next kickoff attempt (the Home path re-fires it on
    // a later HistoryLoaded render) retries cleanly — no duplicate greeting
    // is possible. On success the durable one-shot guard is in place before
    // any side effect.
    if let Err(e) = crate::config::persist_settled_string_field(
        crate::config::CONFIG_KEY_ONBOARDING_STATE,
        OnboardingState::Finished.as_str(),
    )
    .await
    {
        KICKOFF_FIRED.store(false, Ordering::SeqCst);
        return Err(e);
    }
    // The greeting session is built only after several async hops, by which
    // time the `Finished` write above is long since visible — arm the pending
    // bridge so `Session::build_context_messages` still injects the onboarding
    // guide into exactly that session.
    mark_greeting_guide_pending();
    let msg = ChannelMessage {
        user_name: user_name.to_string(),
        reply_target: user_name.to_string(),
        content: "hi mah bot".to_string(),
        channel: "gui".to_string(),
        workspace: crate::users::personal_workspace_name(user_name),
        ..Default::default()
    };
    if let Some(tx) = crate::GUI_MESSAGE_TX.get()
        && let Err(e) = tx.send(msg)
    {
        // Best-effort: the onboarding state is already `Finished`, so a later
        // kickoff cannot re-send; the next admin Assistant session simply
        // receives the guide instead.
        tracing::error!("kickoff: failed to send 'hi mah bot' via GUI_MESSAGE_TX: {e}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shape the owner's Phase-1 entry can take, and the one form it is
    /// read as: the `sk-or-v1-` token wherever it stands in a pasted blob, the
    /// two-line and keyless custom endpoints, and the entries that must be
    /// refused rather than persisted — a bare default endpoint, a URL with junk
    /// pasted on its line, a default endpoint under a key that is not
    /// OpenRouter's, and text that is neither. Refusing rather than persisting is
    /// what keeps `provider_configured()` from turning true on a value that
    /// cannot reach a provider (which would re-prompt forever).
    #[test]
    fn provider_input_is_read_as_the_one_form_it_is() {
        /// What an entry must be read as, compared field for field.
        #[derive(Debug, PartialEq, Eq)]
        enum Read {
            Key(String),
            Endpoint { url: String, key: Option<String> },
            Invalid,
        }

        let key = |key: &str| Read::Key(key.to_string());
        let endpoint = |url: &str, name: Option<&str>| Read::Endpoint {
            url: url.to_string(),
            key: name.map(str::to_string),
        };
        let read = |raw: &str| match parse_provider_input(raw) {
            ProviderInput::OpenRouterKey(key) => Read::Key(key),
            ProviderInput::CustomEndpoint { url, key } => Read::Endpoint { url, key },
            ProviderInput::Invalid => Read::Invalid,
        };

        for (raw, expected) in [
            // A token containing `sk-or-v1-` is an OpenRouter key however it was pasted — alone,
            // with the default endpoint above it, or with a custom URL below it (the reversed order
            // the grammar does not document, whose whole blob must never become the provider key).
            (" sk-or-v1-abc123def456 ", key("sk-or-v1-abc123def456")),
            (
                "https://openrouter.ai/api/v1\nsk-or-v1-abc123def456",
                key("sk-or-v1-abc123def456"),
            ),
            (
                "sk-or-v1-abc123def456\nhttps://custom.example/v1",
                key("sk-or-v1-abc123def456"),
            ),
            // A custom endpoint is a URL on line 1 and an optional key on line 2; a bare URL is a
            // keyless one.
            (
                "https://ollama.local:11434/v1",
                endpoint("https://ollama.local:11434/v1", None),
            ),
            (
                "http://localhost:8080/v1\nsk-local-123",
                endpoint("http://localhost:8080/v1", Some("sk-local-123")),
            ),
            // The default endpoint alone never configures the provider, and neither does it under a
            // foreign key.
            ("https://openrouter.ai/api/v1", Read::Invalid),
            ("https://openrouter.ai/api/v1\nmy-random-key", Read::Invalid),
            // A URL cannot contain a literal space: a key pasted onto its line is outside the
            // documented grammar.
            ("https://custom.example/v1 sk-local-123", Read::Invalid),
            ("https://custom.example/v1 extra", Read::Invalid),
            ("just some chat text", Read::Invalid),
        ] {
            assert_eq!(read(raw), expected, "entry {raw:?}");
        }
    }
}
