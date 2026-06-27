//! Provider subsystem for model inference backends.
//!
//! This module implements the factory pattern for AI model providers. Each provider
//! implements the [`Provider`] trait. Currently only OpenAI-compatible providers
//! are supported, wrapped with automatic retry logic.

pub mod compatible;
pub mod compatible_streaming;
pub mod error;
pub mod reasoning_roundtrip;
pub mod reliable;
pub mod transcribe;

use crate::config::{CONFIG, trimmed_or_none};
pub use crate::{ChatMessage, ChatRequest, ChatResponse, Provider};
use crate::{StreamEvent, StreamResult};

use futures_util::stream;
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
    endpoint
        .trim_end_matches('/')
        .trim_end_matches("/chat/completions")
        .to_string()
}

/// Build a `provider` routing JSON value for OpenAI-compatible chat requests.
///
/// Splits `order` on commas, trims whitespace, and filters empty strings.
/// Returns `None` when the resulting provider list is empty, so callers can
/// skip inserting the routing block entirely (matching the pre-existing
/// behaviour in [`compatible::build_chat_request_raw`]).
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
/// Extracts config from [`CONFIG`], creates the provider and constructs both
/// transcribers (synchronous, no I/O), then optionally warms the provider up
/// (async HTTP call). After warmup (or warmup skip/graceful failure), all three
/// globals — [`PROVIDER`], [`IMAGE_TRANSCRIBER`], [`AUDIO_TRANSCRIBER`] — are
/// swapped in together.
///
/// Used by [`init_global`] (startup, non-fatal warmup) and [`recreate_all`]
/// (config reload, fatal warmup) to eliminate ~28 lines of duplication.
async fn setup_provider_and_transcribers(warmup_mode: WarmupMode) -> anyhow::Result<()> {
    let api_key = CONFIG.provider_key();
    let endpoint = CONFIG.provider_endpoint();
    let endpoint_opt = if endpoint == crate::config::DEFAULT_PROVIDER_ENDPOINT {
        None
    } else {
        Some(endpoint.as_str())
    };

    let provider: Arc<dyn Provider> = create_provider(api_key.as_deref(), endpoint_opt)?.into();

    // Construct transcribers early — purely synchronous CPU work with no I/O,
    // so there's no reason to wait until after the warmup HTTP call.
    let image_transcriber = create_transcriber(
        Some(&endpoint),
        api_key.as_deref(),
        Some(CONFIG.image_transcription_model().as_str()),
        CONFIG.transcription_provider().as_deref(),
        ImageTranscriber::from_inner,
    );
    let audio_transcriber = create_transcriber(
        Some(&endpoint),
        api_key.as_deref(),
        Some(CONFIG.audio_transcription_model().as_str()),
        CONFIG.audio_transcription_provider().as_deref(),
        transcribe::AudioTranscriber::from_inner,
    );

    // Now warm up the provider (costly HTTP round-trip).
    match warmup_mode {
        WarmupMode::Fatal => {
            provider.warmup().await?;
        }
        WarmupMode::NonFatal => {
            if let Err(e) = provider.warmup().await {
                tracing::warn!(endpoint = %endpoint, "Provider warmup failed (non-fatal): {e}");
            }
        }
    }

    // Atomically swap all three globals after warmup verification.
    *PROVIDER.write().expect("PROVIDER poisoned") = Some(provider);
    *IMAGE_TRANSCRIBER
        .write()
        .expect("IMAGE_TRANSCRIBER poisoned") = image_transcriber;
    *AUDIO_TRANSCRIBER
        .write()
        .expect("AUDIO_TRANSCRIBER poisoned") = audio_transcriber;

    Ok(())
}

/// Initialize the global provider and transcribers from CONFIG.
///
/// Warmup failures are non-fatal at startup — the system can still operate;
/// retries happen at request time.
pub async fn init_global() -> anyhow::Result<()> {
    setup_provider_and_transcribers(WarmupMode::NonFatal).await
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
    let endpoint_opt = endpoint.filter(|e| e.as_str() != crate::config::DEFAULT_PROVIDER_ENDPOINT);
    let provider = create_provider(config.provider_key.as_deref(), endpoint_opt.as_deref())?;
    provider.warmup().await?;
    Ok(())
}

/// Recreate all provider and transcriber singletons from current CONFIG.
///
/// Called after a GUI-driven config save to make provider key/endpoint/model
/// changes take effect without restart. Warmup failures are fatal here
/// because the config has already been validated by
/// [`warmup_provider_from_config`] before this point.
pub async fn recreate_all() -> anyhow::Result<()> {
    setup_provider_and_transcribers(WarmupMode::Fatal).await?;
    tracing::info!("Provider and transcriber singletons recreated");
    Ok(())
}

/// Get the global image transcriber, if a vision model is configured.
#[must_use]
pub fn image_transcriber() -> Option<ImageTranscriber> {
    IMAGE_TRANSCRIBER
        .read()
        .expect("IMAGE_TRANSCRIBER poisoned")
        .clone()
}

/// Get the global audio transcriber, if an audio model is configured.
#[must_use]
pub fn audio_transcriber() -> Option<transcribe::AudioTranscriber> {
    AUDIO_TRANSCRIBER
        .read()
        .expect("AUDIO_TRANSCRIBER poisoned")
        .clone()
}

/// Delegate `Provider` trait for the global static.
///
/// # Panics
/// Panics if the provider has not been initialized.
pub async fn chat(request: ChatRequest) -> anyhow::Result<ChatResponse> {
    let provider = PROVIDER
        .read()
        .expect("PROVIDER poisoned")
        .clone()
        .expect("PROVIDER not initialized");
    provider.chat(request).await
}

pub fn stream_chat(request: ChatRequest) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
    let provider = PROVIDER
        .read()
        .expect("PROVIDER poisoned")
        .clone()
        .expect("PROVIDER not initialized");
    provider.stream_chat(request)
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
        .unwrap_or_else(|| crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string());

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
        10,
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
    let _key = api_key.and_then(trimmed_or_none)?;
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
    fn ensure_chat_completions_url_already_has_suffix() {
        assert_eq!(
            ensure_chat_completions_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions",
        );
    }

    #[test]
    fn ensure_chat_completions_url_no_suffix() {
        assert_eq!(
            ensure_chat_completions_url("https://api.example.com/v1"),
            "https://api.example.com/v1/chat/completions",
        );
    }

    #[test]
    fn ensure_chat_completions_url_trailing_slash() {
        assert_eq!(
            ensure_chat_completions_url("https://api.example.com/v1/"),
            "https://api.example.com/v1/chat/completions",
        );
    }

    #[test]
    fn ensure_chat_completions_url_double_trailing_slash() {
        assert_eq!(
            ensure_chat_completions_url("https://api.example.com/v1//"),
            "https://api.example.com/v1/chat/completions",
            "multiple trailing slashes are collapsed by trim_end_matches('/')",
        );
    }

    #[test]
    fn ensure_base_url_already_clean() {
        assert_eq!(
            ensure_base_url("https://api.example.com/v1"),
            "https://api.example.com/v1",
        );
    }

    #[test]
    fn ensure_base_url_strips_chat_completions() {
        assert_eq!(
            ensure_base_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1",
        );
    }

    #[test]
    fn ensure_base_url_trailing_slash() {
        assert_eq!(
            ensure_base_url("https://api.example.com/v1/"),
            "https://api.example.com/v1",
        );
    }

    #[test]
    fn ensure_base_url_trailing_slash_before_suffix() {
        assert_eq!(
            ensure_base_url("https://api.example.com/v1/chat/completions/"),
            "https://api.example.com/v1",
        );
    }

    #[test]
    fn roundtrip_base_to_chat_to_base() {
        let cases = &[
            "https://api.example.com/v1",
            "https://api.example.com/v1/",
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/chat/completions/",
        ];
        for &url in cases {
            let base = ensure_base_url(url);
            let chat = ensure_chat_completions_url(&base);
            let roundtripped = ensure_base_url(&chat);
            assert_eq!(
                roundtripped, base,
                "roundtrip(ensure_base_url -> ensure_chat_completions_url -> ensure_base_url) \
                 should be identity for input '{url}'",
            );
        }
    }

    #[test]
    fn roundtrip_chat_to_base_to_chat() {
        let cases = &[
            "https://api.example.com/v1",
            "https://api.example.com/v1/",
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/chat/completions/",
        ];
        for &url in cases {
            let chat = ensure_chat_completions_url(url);
            let base = ensure_base_url(&chat);
            let roundtripped = ensure_chat_completions_url(&base);
            assert_eq!(
                roundtripped, chat,
                "roundtrip(ensure_chat_completions_url -> ensure_base_url -> \
                 ensure_chat_completions_url) should be identity for input '{url}'",
            );
        }
    }

    #[test]
    fn domain_name_containing_slash_chat_completions() {
        // Edge case: URL where /chat/completions appears in the domain, not a path segment.
        // This is a shared limitation of both helpers — they operate on strings, not URL components.
        // We document the current behaviour rather than asserting correctness.
        let url = "https://chat.completions.com/api";
        let chat = ensure_chat_completions_url(url);
        // trim_end_matches does not match '/chat/completions' here because the path
        // is '/api', not '/chat/completions'. The domain contains 'chat.completions.com'
        // but there's no '/chat/completions' at the end.
        assert_eq!(chat, "https://chat.completions.com/api/chat/completions");

        let base = ensure_base_url(url);
        // Same reasoning — '/chat/completions' is not a suffix.
        assert_eq!(base, "https://chat.completions.com/api");
    }

    // ── provider_routing_json tests ─────────────────────────────

    #[test]
    fn routing_single_provider() {
        assert_eq!(
            provider_routing_json("openai", false),
            Some(serde_json::json!({
                "order": ["openai"],
                "allow_fallbacks": false,
            })),
        );
    }

    #[test]
    fn routing_multiple_providers() {
        assert_eq!(
            provider_routing_json("openai, anthropic, google", true),
            Some(serde_json::json!({
                "order": ["openai", "anthropic", "google"],
                "allow_fallbacks": true,
            })),
        );
    }

    #[test]
    fn routing_whitespace_only_yields_none() {
        assert_eq!(provider_routing_json("  , ,  ", false), None);
    }

    #[test]
    fn routing_empty_string_yields_none() {
        assert_eq!(provider_routing_json("", true), None);
    }

    #[test]
    fn routing_leading_trailing_whitespace() {
        assert_eq!(
            provider_routing_json("  openai  ", false),
            Some(serde_json::json!({
                "order": ["openai"],
                "allow_fallbacks": false,
            })),
        );
    }

    #[test]
    fn routing_single_slug_survives_split() {
        // Transcription call sites pass a single provider slug; the
        // split/trim/filter cycle must leave it unchanged.
        assert_eq!(
            provider_routing_json("google-gemini", false),
            Some(serde_json::json!({
                "order": ["google-gemini"],
                "allow_fallbacks": false,
            })),
        );
    }
}
