//! Provider subsystem for model inference backends.
//!
//! This module implements the factory pattern for AI model providers. Each provider
//! implements the [`Provider`] trait. Currently only OpenAI-compatible providers
//! are supported, wrapped with automatic retry logic.

pub mod compatible;
pub mod reasoning;
pub mod reasoning_roundtrip;
pub mod reliable;
pub mod transcribe;

pub use reasoning::plaintext_for_display;

use crate::config::{CONFIG, non_empty, resolve_list_or, resolve_or, trimmed_or_none};
use crate::util::UnwrapPoison;
pub use crate::{ChatMessage, ChatRequest, ChatResponse, Provider};

/// Helper for tests that constructs a `ChatRequest` with sensible defaults.
/// Callers override specific fields via struct update syntax:
/// ```ignore
/// let req = ChatRequest { model: "test-model".into(), temperature: 0.7, ..test_request(messages, None) };
/// ```
#[cfg(test)]
pub(crate) fn test_request(
    messages: Vec<ChatMessage>,
    tools: Option<Vec<crate::ToolSpec>>,
) -> ChatRequest {
    ChatRequest {
        messages,
        tools,
        model: "test".to_string(),
        allow_image_parts: false,
        temperature: 0.1,
        max_tokens: None,
        reasoning_effort: None,
        provider_order: None,
        provider_allow_fallbacks: None,
    }
}

use std::sync::{Arc, RwLock};

pub use crate::providers::transcribe::ImageTranscriber;

use compatible::OpenAiCompatibleProvider;
use reliable::ReliableProvider;

/// Ensure a base URL includes the `/chat/completions` path segment.
/// If the URL already ends with `/chat/completions`, it is returned as-is.
pub(crate) fn ensure_chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

/// Strip the `/chat/completions` suffix from an endpoint URL to obtain the API base URL.
///
/// This is the complement of [`ensure_chat_completions_url`] — it undoes the addition
/// of `/chat/completions` so that sibling API paths (e.g. `/videos` or `/embeddings`)
/// can be appended. Image generation uses the chat-completions endpoint directly
/// (it mimics a chat-format tool-use API), while video generation uses a dedicated
/// `/videos` endpoint under the same API base.
pub(crate) fn ensure_base_url(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    trimmed
        .strip_suffix("/chat/completions")
        .unwrap_or(trimmed)
        .to_string()
}

/// Build a `provider` routing JSON value for OpenAI-compatible chat requests.
///
/// Splits `order` on commas, trims whitespace, and filters empty strings.
/// Returns `None` when the resulting provider list is empty, so callers can
/// skip inserting the routing block entirely (matching the pre-existing
/// behaviour in [`compatible::build_http_request`]).
///
/// This works for both comma-separated provider lists (chat completions) and
/// single-provider strings (transcription) — a single slug survives the
/// split/trim/filter cycle unchanged.
///
/// # Example
///
/// ```ignore
/// let routing = provider_routing_json("openai,   anthropic  ", true);
/// assert_eq!(
///     routing,
///     Some(serde_json::json!({
///         "order": ["openai", "anthropic"],
///         "allow_fallbacks": true,
///     })),
/// );
/// ```
pub(crate) fn provider_routing_json(
    order: &str,
    allow_fallbacks: bool,
) -> Option<serde_json::Value> {
    let providers: Vec<&str> = order
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if providers.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "order": providers,
        "allow_fallbacks": allow_fallbacks,
    }))
}

// ── Global singletons (recreatable via RwLock) ─────────────────

/// Global provider instance. Recreatable when config changes at runtime.
/// Wrapped in `Arc` so we can clone-and-drop the lock before awaiting.
static PROVIDER: RwLock<Option<Arc<dyn Provider>>> = RwLock::new(None);

/// Global image transcriber (vision model for image descriptions).
static IMAGE_TRANSCRIBER: RwLock<Option<ImageTranscriber>> = RwLock::new(None);

/// Global audio transcriber.
static AUDIO_TRANSCRIBER: RwLock<Option<transcribe::AudioTranscriber>> = RwLock::new(None);

/// Controls whether warmup failure should propagate or be non-fatal.
///
/// Used by [`setup_provider_and_transcribers`] to differentiate the two
/// call sites without a raw boolean parameter.
enum WarmupMode {
    /// Warmup failure is non-fatal — logged as a warning, init proceeds.
    NonFatal,
    /// Warmup failure propagates as an error.
    Fatal,
}

/// Shared provider and transcriber setup logic.
///
/// Extracts config from the given [`ConfigData`], creates the provider and
/// constructs both transcribers (synchronous, no I/O), then optionally warms
/// the provider up (async HTTP call). After warmup (or warmup skip/graceful
/// failure), all three globals — [`PROVIDER`], [`IMAGE_TRANSCRIBER`],
/// [`AUDIO_TRANSCRIBER`] — are swapped in together.
///
/// Used by [`init_global`] (startup, non-fatal warmup) and [`recreate_all`]
/// (config reload, fatal warmup) to eliminate ~28 lines of duplication.
async fn setup_provider_and_transcribers(
    warmup_mode: WarmupMode,
    config: &crate::config::ConfigData,
) -> anyhow::Result<()> {
    let endpoint_str = resolve_or(
        config.provider_endpoint.clone(),
        crate::config::DEFAULT_PROVIDER_ENDPOINT,
    );
    let provider: Arc<dyn Provider> =
        create_provider(config.provider_key.as_deref(), Some(endpoint_str.as_str()))?.into();

    // Construct transcribers early — purely synchronous CPU work with no I/O,
    // so there's no reason to wait until after the warmup HTTP call.
    let image_transcriber = create_transcriber(
        Some(&endpoint_str),
        config.provider_key.as_deref(),
        Some(
            resolve_or(
                config.image_transcription_model.clone(),
                crate::config::DEFAULT_IMAGE_TRANSCRIPTION_MODEL,
            )
            .as_str(),
        ),
        non_empty(config.image_transcription_provider.clone()).as_deref(),
        ImageTranscriber::from_inner,
    );
    let audio_transcriber = {
        let has_key = config
            .provider_key
            .as_deref()
            .and_then(trimmed_or_none)
            .is_some();
        if has_key {
            let models = resolve_list_or(
                config.audio_transcription_models.as_deref(),
                config.audio_transcription_model.clone(),
                crate::config::DEFAULT_AUDIO_TRANSCRIPTION_MODEL,
            );
            let route = non_empty(config.audio_transcription_provider.clone());
            Some(transcribe::AudioTranscriber::new(
                endpoint_str.clone(),
                models,
                route,
            ))
        } else {
            None
        }
    };

    // Now warm up the provider (costly HTTP round-trip).
    match warmup_mode {
        WarmupMode::Fatal => {
            provider.warmup().await?;
        }
        WarmupMode::NonFatal => {
            if let Err(e) = provider.warmup().await {
                tracing::warn!(endpoint = %endpoint_str, "Provider warmup failed (non-fatal): {e}");
            }
        }
    }

    // Atomically swap all three globals after warmup verification.
    *PROVIDER.write().unwrap_poison() = Some(provider);
    *IMAGE_TRANSCRIBER.write().unwrap_poison() = image_transcriber;
    *AUDIO_TRANSCRIBER.write().unwrap_poison() = audio_transcriber;

    Ok(())
}

/// Initialize the global provider and transcribers from CONFIG.
///
/// Warmup failures are non-fatal at startup — the system can still operate;
/// retries happen at request time.
pub async fn init_global() -> anyhow::Result<()> {
    let config = CONFIG.snapshot();
    setup_provider_and_transcribers(WarmupMode::NonFatal, &config).await
}

/// Warm up a provider from a config snapshot without swapping globals.
///
/// Returns `Ok(())` if the new API key, endpoint, and models are valid
/// (the provider responds to a warmup request). Does **not** modify the
/// global `PROVIDER`, `IMAGE_TRANSCRIBER`, or `AUDIO_TRANSCRIBER`.
/// Used by [`save_and_reload`](crate::config::save_and_reload) as a
/// pre-commit validation step.
pub async fn warmup_provider_from_config(config: &crate::config::ConfigData) -> anyhow::Result<()> {
    let endpoint = config
        .provider_endpoint
        .as_deref()
        .and_then(trimmed_or_none);
    let provider = create_provider(config.provider_key.as_deref(), endpoint.as_deref())?;
    provider.warmup().await?;
    Ok(())
}

/// Recreate all provider and transcriber singletons from the given config.
///
/// Called after a GUI-driven config save to make provider key/endpoint/model
/// changes take effect without restart. Warmup failures are fatal here
/// because the config has already been validated by
/// [`warmup_provider_from_config`] before this point.
pub async fn recreate_all(config: &crate::config::ConfigData) -> anyhow::Result<()> {
    setup_provider_and_transcribers(WarmupMode::Fatal, config).await?;
    tracing::info!("Provider and transcriber singletons recreated");
    Ok(())
}

/// Get the global image transcriber, if a vision model is configured.
#[must_use]
pub fn image_transcriber() -> Option<ImageTranscriber> {
    IMAGE_TRANSCRIBER.read().unwrap_poison().clone()
}

/// Get the global audio transcriber, if an audio model is configured.
#[must_use]
pub fn audio_transcriber() -> Option<transcribe::AudioTranscriber> {
    AUDIO_TRANSCRIBER.read().unwrap_poison().clone()
}

/// Delegate `Provider` trait for the global static.
///
/// # Panics
/// Panics if the provider has not been initialized.
pub async fn chat(request: ChatRequest) -> anyhow::Result<ChatResponse> {
    let provider = PROVIDER
        .read()
        .unwrap_poison()
        .clone()
        .expect("PROVIDER not initialized");
    provider.chat(request).await
}

/// Create a resilient OpenAI-compatible provider from flat config.
///
/// When `provider_endpoint` is unset, defaults to [OpenRouter](https://openrouter.ai)
/// and sets OpenRouter-specific headers (`X-Title`, `HTTP-Referrer`). A non-empty
/// `provider_endpoint` overrides the base URL — the same headers are still sent (most
/// providers ignore them harmlessly).
///
/// Returns a reliable provider wrapping an [`OpenAiCompatibleProvider`].
pub fn create_provider(
    api_key: Option<&str>,
    endpoint: Option<&str>,
) -> anyhow::Result<Box<dyn Provider>> {
    let key_owned = api_key.and_then(trimmed_or_none);
    let resolved_key = key_owned.as_deref();
    let base_url = endpoint
        .and_then(trimmed_or_none)
        .unwrap_or(crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string());

    let mut extra_headers = std::collections::HashMap::new();
    extra_headers.insert("X-Title".to_string(), "MahBot".to_string());
    extra_headers.insert(
        "HTTP-Referrer".to_string(),
        "https://github.com/edezhic".to_string(),
    );

    let base = OpenAiCompatibleProvider::new("OpenRouter", base_url.as_str(), resolved_key)
        .with_extra_headers(extra_headers);

    let provider: Box<dyn Provider> = Box::new(base);

    let reliable: Box<dyn Provider> = Box::new(ReliableProvider::new(
        "openrouter".to_string(),
        provider,
        4,
        500,
    ));
    Ok(reliable)
}

/// Generic helper to build a transcriber from flat config options.
#[must_use]
fn create_transcriber<T>(
    api_url: Option<&str>,
    api_key: Option<&str>,
    model: Option<&str>,
    provider: Option<&str>,
    wrapper: impl FnOnce(transcribe::MediaTranscriber) -> T,
) -> Option<T> {
    api_key.and_then(trimmed_or_none)?;
    let model = model.and_then(trimmed_or_none)?;
    let route = provider.and_then(trimmed_or_none);
    let base_url = api_url
        .unwrap_or(crate::config::DEFAULT_PROVIDER_ENDPOINT)
        .to_string();
    let inner = transcribe::MediaTranscriber::new(base_url, model, route);
    Some(wrapper(inner))
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_roundtrips() {
        // Cases exercise both ensure_chat_completions_url and ensure_base_url
        // on the same input (they are string-based inverses).
        struct Case {
            name: &'static str,
            input: &'static str,
            expected_chat: &'static str,
            expected_base: &'static str,
        }

        let cases = [
            Case {
                name: "already_has_suffix",
                input: "https://api.example.com/v1/chat/completions",
                expected_chat: "https://api.example.com/v1/chat/completions",
                expected_base: "https://api.example.com/v1",
            },
            Case {
                name: "no_suffix",
                input: "https://api.example.com/v1",
                expected_chat: "https://api.example.com/v1/chat/completions",
                expected_base: "https://api.example.com/v1",
            },
            Case {
                name: "trailing_slash",
                input: "https://api.example.com/v1/",
                expected_chat: "https://api.example.com/v1/chat/completions",
                expected_base: "https://api.example.com/v1",
            },
            Case {
                // Multiple trailing slashes are collapsed by trim_end_matches('/').
                name: "double_trailing_slash",
                input: "https://api.example.com/v1//",
                expected_chat: "https://api.example.com/v1/chat/completions",
                expected_base: "https://api.example.com/v1",
            },
            Case {
                name: "trailing_slash_before_suffix",
                input: "https://api.example.com/v1/chat/completions/",
                expected_chat: "https://api.example.com/v1/chat/completions",
                expected_base: "https://api.example.com/v1",
            },
            // Edge case: URL where /chat/completions appears in the domain, not a path segment.
            // This is a shared limitation of both helpers — they operate on strings, not URL
            // components. We document the current behaviour rather than asserting correctness.
            Case {
                name: "domain_containing_chat_completions",
                input: "https://chat.completions.com/api",
                expected_chat: "https://chat.completions.com/api/chat/completions",
                expected_base: "https://chat.completions.com/api",
            },
            // Regression: ensure_base_url must strip exactly one suffix, not all.
            // trim_end_matches would strip both; strip_suffix stops after one.
            Case {
                name: "repeated_suffix",
                input: "https://api.example.com/v1/chat/completions/chat/completions",
                expected_chat: "https://api.example.com/v1/chat/completions/chat/completions",
                expected_base: "https://api.example.com/v1/chat/completions",
            },
        ];

        for c in &cases {
            assert_eq!(
                ensure_chat_completions_url(c.input),
                c.expected_chat,
                "case '{}': ensure_chat_completions_url({:?})",
                c.name,
                c.input,
            );
            assert_eq!(
                ensure_base_url(c.input),
                c.expected_base,
                "case '{}': ensure_base_url({:?})",
                c.name,
                c.input,
            );
        }

        // Roundtrip property: base -> chat -> base and chat -> base -> chat
        // should both be identity.
        let roundtrip_inputs = &[
            "https://api.example.com/v1",
            "https://api.example.com/v1/",
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/chat/completions/",
        ];
        for &url in roundtrip_inputs {
            let base = ensure_base_url(url);
            let chat = ensure_chat_completions_url(&base);
            let roundtripped = ensure_base_url(&chat);
            assert_eq!(
                roundtripped, base,
                "roundtrip(base->chat->base) should be identity for '{url}'",
            );

            let chat = ensure_chat_completions_url(url);
            let base = ensure_base_url(&chat);
            let roundtripped = ensure_chat_completions_url(&base);
            assert_eq!(
                roundtripped, chat,
                "roundtrip(chat->base->chat) should be identity for '{url}'",
            );
        }
    }

    #[test]
    fn provider_routing() {
        struct Case {
            name: &'static str,
            order: &'static str,
            allow_fallbacks: bool,
            expected: Option<serde_json::Value>,
        }

        let cases = [
            Case {
                name: "single_provider",
                order: "openai",
                allow_fallbacks: false,
                expected: Some(serde_json::json!({
                    "order": ["openai"],
                    "allow_fallbacks": false,
                })),
            },
            Case {
                name: "multiple_providers",
                order: "openai, anthropic, google",
                allow_fallbacks: true,
                expected: Some(serde_json::json!({
                    "order": ["openai", "anthropic", "google"],
                    "allow_fallbacks": true,
                })),
            },
            Case {
                name: "whitespace_only_yields_none",
                order: "  , ,  ",
                allow_fallbacks: false,
                expected: None,
            },
            Case {
                name: "empty_string_yields_none",
                order: "",
                allow_fallbacks: true,
                expected: None,
            },
            Case {
                name: "leading_trailing_whitespace",
                order: "  openai  ",
                allow_fallbacks: false,
                expected: Some(serde_json::json!({
                    "order": ["openai"],
                    "allow_fallbacks": false,
                })),
            },
            // Transcription call sites pass a single provider slug; the
            // split/trim/filter cycle must leave it unchanged.
            Case {
                name: "single_slug_survives_split",
                order: "google-gemini",
                allow_fallbacks: false,
                expected: Some(serde_json::json!({
                    "order": ["google-gemini"],
                    "allow_fallbacks": false,
                })),
            },
        ];

        for c in &cases {
            assert_eq!(
                provider_routing_json(c.order, c.allow_fallbacks),
                c.expected,
                "case '{}': provider_routing_json({:?}, {})",
                c.name,
                c.order,
                c.allow_fallbacks,
            );
        }
    }
}
