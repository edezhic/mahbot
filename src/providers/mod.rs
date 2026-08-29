//! Provider subsystem for model inference backends.
//!
//! This module implements the factory pattern for AI model providers. Each provider
//! implements the [`Provider`] trait. Currently only OpenAI-compatible providers
//! are supported; the outer retry orchestration lives in [`crate::retry`].

pub(crate) mod compatible;
pub(crate) mod reasoning;
pub(crate) mod reliable;
pub(crate) mod transcribe;

pub(crate) use reasoning::plaintext_for_display;

use crate::config::{CONFIG, normalize_endpoint_url, trimmed_or_none};
use crate::util::UnwrapPoison;
pub(crate) use crate::{ChatRequest, ChatResponse, Provider};

#[cfg(test)]
use crate::ChatMessage;

/// Helper for tests that constructs a `ChatRequest` with sensible defaults.
/// Callers override specific fields via struct update syntax:
/// ```ignore
/// let req = ChatRequest { model: "test-model".into(), ..test_request(messages, None) };
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
        max_tokens: None,
        reasoning_effort: None,
        provider_order: None,
        meta: None,
    }
}

use std::sync::{Arc, RwLock};
use std::time::Instant;

pub(crate) use crate::providers::transcribe::{MediaTranscriber, transcribe_video_file};

use crate::retry::{FailureClass, RetryFailureRecord};
use compatible::OpenAiCompatibleProvider;

// ── Scoped call error ────────────────────────────────────

/// Error from a scoped (single-attempt) provider call.
///
/// Carries the underlying error, a granular [`FailureClass`], and the
/// per-attempt diagnostics [`RetryFailureRecord`] so the outer retry loop can
/// classify and build human-readable failure trails without re-stringifying.
#[derive(Debug)]
pub(crate) struct ScopedCallError {
    pub inner: anyhow::Error,
    pub record: RetryFailureRecord,
    pub class: FailureClass,
}

impl ScopedCallError {
    #[must_use]
    pub(crate) fn new(
        inner: anyhow::Error,
        record: RetryFailureRecord,
        class: FailureClass,
    ) -> Self {
        Self {
            inner,
            record,
            class,
        }
    }
}

impl std::fmt::Display for ScopedCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::error::Error for ScopedCallError {}

/// Map a provider [`ErrorClass`] to a granular [`FailureClass`]. Retryable
/// errors refine to [`FailureClass::TruncatedEnvelope`] when the response
/// envelope was truncated mid-read (the defect class this hardening targets);
/// a non-retryable status (auth, quota) still aborts immediately.
#[must_use]
pub(crate) fn failure_class(class: reliable::ErrorClass, truncated: bool) -> FailureClass {
    match class {
        reliable::ErrorClass::NonRetryable => FailureClass::NonRetryable,
        reliable::ErrorClass::Retryable if truncated => FailureClass::TruncatedEnvelope,
        reliable::ErrorClass::Retryable => FailureClass::Transport,
    }
}

/// Ensure a base URL includes the `/chat/completions` path segment.
/// Delegates to [`crate::config::normalize_endpoint_url`] for suffix handling,
/// so this also trims surrounding whitespace (both ends, not just trailing `/`)
/// and lowercases scheme/host. Intentional narrow behavior change vs the
/// previous `trim_end_matches('/')`-only impl — stored values are already
/// rejected for uppercase scheme via `is_http_url` and pre-trimmed via
/// `trimmed_or_none`, so divergence was latent/hygiene.
/// SSoT coupling: future changes to `normalize_endpoint_url` intentionally
/// affect fetch URL construction at all call sites (catalog, transcribe,
/// compatible provider).
/// Suffix matching remains case-sensitive (`/Chat/Completions` is not stripped).
/// Edge cases: scheme-less input is returned largely unchanged (no `://` → no
/// lowercasing, just trimmed/stripped); empty/whitespace-only input normalizes
/// to `""` and this wrapper then yields `"/chat/completions"`.
pub(crate) fn ensure_chat_completions_url(url: &str) -> String {
    format!("{}/chat/completions", normalize_endpoint_url(url))
}

/// Strip the `/chat/completions` suffix from an endpoint URL to obtain the API base URL.
///
/// This is the complement of [`ensure_chat_completions_url`] — it undoes the addition
/// of `/chat/completions` so that sibling API paths (e.g. `/videos` or `/embeddings`)
/// can be appended. Image generation uses the chat-completions endpoint directly
/// (it mimics a chat-format tool-use API), while video generation uses a dedicated
/// `/videos` endpoint under the same API base.
/// Delegates to [`crate::config::normalize_endpoint_url`] for suffix handling,
/// so this also trims surrounding whitespace (both ends, not just trailing `/`)
/// and lowercases scheme/host. Intentional narrow behavior change vs the
/// previous `trim_end_matches('/')`-only impl — stored values are already
/// rejected for uppercase scheme via `is_http_url` and pre-trimmed via
/// `trimmed_or_none`, so divergence was latent/hygiene.
/// SSoT coupling: future changes to `normalize_endpoint_url` intentionally
/// affect fetch URL construction at all call sites (catalog, transcribe,
/// compatible provider).
/// Suffix matching remains case-sensitive (`/Chat/Completions` is not stripped).
/// Edge cases: scheme-less input is returned largely unchanged (no `://` → no
/// lowercasing, just trimmed/stripped); empty/whitespace-only input normalizes
/// to `""`.
pub(crate) fn ensure_base_url(url: &str) -> String {
    normalize_endpoint_url(url)
}

/// Build a `provider` routing JSON value for OpenAI-compatible chat requests.
///
/// Splits `order` on commas, trims whitespace, and filters empty strings.
/// Returns `None` when the resulting provider list is empty, so callers can
/// skip inserting the routing block entirely (matching the behaviour of the
/// OpenAI-compatible request builder).
///
/// This works for both comma-separated provider lists (chat completions) and
/// single-provider strings (transcription) — a single slug survives the
/// split/trim/filter cycle unchanged.
///
/// Fallbacks are explicitly pinned to `false` in the emitted JSON — the
/// Allow Fallbacks option was removed from the settings and the runtime.
///
/// # Example
///
/// ```ignore
/// let routing = provider_routing_json("openai,   anthropic  ");
/// assert_eq!(
///     routing,
///     Some(serde_json::json!({
///         "order": ["openai", "anthropic"],
///         "allow_fallbacks": false,
///     })),
/// );
/// ```
pub(crate) fn provider_routing_json(order: &str) -> Option<serde_json::Value> {
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
        "allow_fallbacks": false,
    }))
}

// ── Global singletons (recreatable via RwLock) ─────────────────

/// Global provider instance. Recreatable when config changes at runtime.
/// Wrapped in `Arc` so we can clone-and-drop the lock before awaiting.
static PROVIDER: RwLock<Option<Arc<dyn Provider>>> = RwLock::new(None);

/// Global media transcriber (vision model for video descriptions).
static MEDIA_TRANSCRIBER: RwLock<Option<MediaTranscriber>> = RwLock::new(None);

/// Build the provider and media transcriber from config (synchronous, no I/O).
///
/// Extracted from the shared setup logic so both the boot path ([`init_global`])
/// and the config-save path ([`recreate_all`]) construct the singletons once
/// and differ only in warmup handling.
fn build_provider_and_transcriber(
    config: &crate::config::ConfigData,
) -> (Arc<dyn Provider>, Option<MediaTranscriber>) {
    // The effective endpoint + credential come from config: a custom
    // chat-completions endpoint when one is persisted, else the
    // default OpenRouter endpoint + OpenRouter key. The custom endpoint's own
    // key (if any) is used only for the custom endpoint — never the OpenRouter
    // key.
    let endpoint = crate::config::effective_chat_endpoint(config);
    let credential = crate::config::chat_credential(config);
    let provider: Arc<dyn Provider> =
        create_provider(credential.as_deref(), Some(&endpoint)).into();

    // Construct the transcriber eagerly — purely synchronous CPU work with no
    // I/O, so there's no reason to wait until after the warmup HTTP call.
    let media_transcriber = build_media_transcriber(config);

    (provider, media_transcriber)
}

/// Initialize the global provider and transcriber singletons from CONFIG.
///
/// Non-blocking by design (decision 4): the globals are swapped in
/// BEFORE the warmup HTTP round-trip so boot never waits on it (worst case the
/// endpoint is blackholed for minutes — a failure is non-fatal, retries happen
/// at request time). The warmup runs as a detached background task and only
/// pre-warms the connection pool; `chat_scoped` must never observe an unset
/// [`PROVIDER`], hence the swap-before-warmup ordering.
///
/// The local Qwen3-ASR transcriber init no longer lives here — the boot path
/// spawns it separately (see
/// [`crate::audio::local_transcriber::spawn_background_init_if_enabled`])
/// after `config::reload_from_db` so its ~4 s load overlaps with the rest of
/// boot instead of being awaited here.
pub fn init_global() -> anyhow::Result<()> {
    let config = CONFIG.snapshot();
    let (provider, media_transcriber) = build_provider_and_transcriber(&config);

    // Swap both globals now — warmup is a pool pre-warm, not a readiness gate.
    *PROVIDER.write().unwrap_poison() = Some(provider.clone());
    *MEDIA_TRANSCRIBER.write().unwrap_poison() = media_transcriber;

    // Background warmup (non-fatal). The provider is Arc-cloned so the task
    // outlives init_global; the endpoint string is captured for the log.
    // The endpoint is the effective chat endpoint — a persisted custom value
    // is honored.
    let endpoint_str = crate::config::effective_chat_endpoint(&config);
    tokio::spawn(async move {
        if let Err(e) = provider.warmup().await {
            tracing::warn!(endpoint = %endpoint_str, "Provider warmup failed (non-fatal): {e}");
        }
    });

    Ok(())
}

/// Warm up a provider from a config snapshot without swapping globals.
///
/// Returns `Ok(())` if the new API key, endpoint, and models are valid
/// (the provider responds to a warmup request). Does **not** modify the
/// global `PROVIDER` or `MEDIA_TRANSCRIBER`.
///
/// Best-effort reachability probe: the persist path
/// ([`crate::config::persist_settled_string_field`]) warns on failure
/// instead of rejecting the save, so a self-hosted endpoint can be
/// configured before it is reachable.
pub(crate) async fn warmup_provider_from_config(
    config: &crate::config::ConfigData,
) -> anyhow::Result<()> {
    // Warm up against the effective endpoint — a persisted custom value is
    // honored — with the credential that endpoint would use.
    let endpoint = crate::config::effective_chat_endpoint(config);
    let credential = crate::config::chat_credential(config);
    let provider = create_provider(credential.as_deref(), Some(&endpoint));
    provider.warmup().await?;
    Ok(())
}

/// Recreate all provider and transcriber singletons from the given config.
///
/// Called after a GUI-driven config save to make provider key/endpoint/model
/// changes take effect without restart.
///
/// The runtime MUST switch to the newly saved endpoint even when it is
/// unreachable: the singletons are swapped in BEFORE the
/// warmup, which then runs as a non-fatal background task (mirroring
/// [`init_global`]) unless `background_warmup` is false. A failed warmup
/// never leaves the old provider live — only structural validation and DB
/// writes can fail a save now, so this function cannot fail and returns
/// `()`.
///
/// `background_warmup`: the provider-endpoint persist arm performs its own
/// foreground warmup to surface the req-9 warning (or deliberately skips
/// it when reverting to the default), so it passes `false` to avoid a
/// duplicate network call; the provider-key persist arm passes `true` so
/// the new credential's pool is pre-warmed in the background.
///
/// Also attempts to load the local Qwen3-ASR transcriber from cache if
/// `audio_transcription_use_local` is enabled and the transcriber isn't
/// already loaded. If cached files are missing, a background download is
/// spawned — subsequent transcription requests return a placeholder until
/// the download completes.
pub(crate) async fn recreate_all(config: &crate::config::ConfigData, background_warmup: bool) {
    let (provider, media_transcriber) = build_provider_and_transcriber(config);

    // Config-save path: swap the globals FIRST so the runtime switches to the
    // new endpoint even when it is unreachable. The warmup is
    // then a non-fatal background pool pre-warm, exactly like boot.
    *PROVIDER.write().unwrap_poison() = Some(provider.clone());
    *MEDIA_TRANSCRIBER.write().unwrap_poison() = media_transcriber;
    tracing::info!("Provider and transcriber singletons recreated");

    if background_warmup {
        // Background warmup (non-fatal): the new provider may be unreachable right
        // now (self-hosted endpoint configured before its server is up) — retries
        // happen at request time, and the saved value stands regardless.
        let endpoint_str = crate::config::effective_chat_endpoint(config);
        tokio::spawn(async move {
            if let Err(e) = provider.warmup().await {
                tracing::warn!(endpoint = %endpoint_str, "Provider warmup failed (non-fatal): {e}");
            }
        });
    }

    // Re-init local transcriber if config enables it and it's not already ready.
    let use_local = config.audio_transcription_use_local.as_deref() != Some("false");
    if use_local && !crate::audio::local_transcriber::is_loaded() {
        if crate::audio::local_transcriber::try_init_from_cache().await {
            tracing::info!("Local Qwen3-ASR transcriber loaded from cache after config reload");
        } else {
            tracing::info!(
                "Local Qwen3-ASR transcriber will be downloaded in background after config reload"
            );
        }
    }
}

/// Get the global media transcriber, if a provider key is configured
/// (used for video transcription).
#[must_use]
pub(crate) fn media_transcriber() -> Option<MediaTranscriber> {
    MEDIA_TRANSCRIBER.read().unwrap_poison().clone()
}

/// Single-attempt scoped chat for the outer retry loops (see [`crate::retry`]).
///
/// Suppresses provider-internal retries (the outer loop is the single retry
/// authority), applies idle-timeout semantics, and bounds the attempt by the
/// remaining operation deadline. See [`Provider::chat_scoped`].
pub(crate) async fn chat_scoped(
    request: ChatRequest,
    idle_timeout: std::time::Duration,
    deadline: Instant,
) -> Result<ChatResponse, ScopedCallError> {
    let provider = PROVIDER
        .read()
        .unwrap_poison()
        .clone()
        .expect("PROVIDER not initialized");
    provider.chat_scoped(request, idle_timeout, deadline).await
}

/// Swap the global provider for tests, returning the previous value so the
/// caller can restore it (see `util::test::install_fake_provider`'s RAII
/// guard). Test doubles override [`Provider::chat_scoped`] to control failure
/// classes and request bytes without touching the network.
#[cfg(test)]
pub(crate) fn swap_provider_for_test(provider: Arc<dyn Provider>) -> Option<Arc<dyn Provider>> {
    let mut guard = PROVIDER.write().unwrap_poison();
    let previous = guard.clone();
    *guard = Some(provider);
    previous
}

/// Restore a previously swapped-out global provider (test isolation).
#[cfg(test)]
pub(crate) fn restore_provider_for_test(previous: Option<Arc<dyn Provider>>) {
    *PROVIDER.write().unwrap_poison() = previous;
}

/// Restore the previous global provider ONLY if the current provider is still
/// the one this guard installed.
///
/// A concurrent test may have swapped a newer fake in over ours; clobbering it
/// back to a stale `previous` would unset the provider under that test (the
/// intermittent "PROVIDER not initialized" panic). If a newer provider is
/// present, this guard's restore is a no-op — the newer guard owns the slot.
#[cfg(test)]
pub(crate) fn restore_provider_for_test_if(
    installed: &Arc<dyn Provider>,
    previous: Option<Arc<dyn Provider>>,
) {
    let mut guard = PROVIDER.write().unwrap_poison();
    if guard
        .as_ref()
        .is_some_and(|cur| Arc::ptr_eq(cur, installed))
    {
        *guard = previous;
    }
}

/// Snapshot the current global provider for later restore (test isolation
/// — the persist path rebuilds the singleton; pairs with
/// [`restore_provider_for_test`]).
#[cfg(test)]
pub(crate) fn snapshot_provider_for_test() -> Option<Arc<dyn Provider>> {
    PROVIDER.read().unwrap_poison().clone()
}

/// Snapshot the current global media transcriber for later restore (test
/// isolation — pairs with [`restore_transcriber_for_test`]).
#[cfg(test)]
pub(crate) fn snapshot_transcriber_for_test() -> Option<MediaTranscriber> {
    MEDIA_TRANSCRIBER.read().unwrap_poison().clone()
}

/// Restore a previously snapshotted global media transcriber (test isolation).
#[cfg(test)]
pub(crate) fn restore_transcriber_for_test(previous: Option<MediaTranscriber>) {
    *MEDIA_TRANSCRIBER.write().unwrap_poison() = previous;
}

/// Create a resilient OpenAI-compatible provider from flat config.
///
/// Identity headers (`X-Title`, `HTTP-Referrer`) are sent unconditionally
/// (most providers ignore them harmlessly). Upstream-provider attribution
/// needs no request-side opt-in: OpenRouter's top-level `provider` response
/// field (undocumented in the API reference but consumed by OpenRouter's
/// own SDK) carries it.
///
/// The display name is derived from the endpoint: "OpenRouter"
/// for the default endpoint, "Custom endpoint" otherwise — so custom-endpoint
/// errors never falsely say "OpenRouter".
///
/// Returns an [`OpenAiCompatibleProvider`]; retry orchestration lives in
/// [`crate::retry`].
pub(crate) fn create_provider(api_key: Option<&str>, endpoint: Option<&str>) -> Box<dyn Provider> {
    let key_owned = api_key.and_then(trimmed_or_none);
    let resolved_key = key_owned.as_deref();
    let base_url = endpoint
        .and_then(trimmed_or_none)
        .unwrap_or(crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string());

    let mut headers = std::collections::HashMap::new();
    headers.insert("X-Title".to_string(), "MahBot".to_string());
    headers.insert(
        "HTTP-Referrer".to_string(),
        "https://github.com/edezhic/mahbot".to_string(),
    );
    let name = if crate::config::is_default_endpoint(&base_url) {
        "OpenRouter"
    } else {
        "Custom endpoint"
    };
    let base = OpenAiCompatibleProvider::new(name, base_url.as_str(), resolved_key)
        .with_extra_headers(headers);

    Box::new(base)
}

/// Build the media transcriber from a config snapshot (synchronous, no I/O).
///
/// The video-transcription model is hardcoded (`VIDEO_TRANSCRIPTION_MODEL`) and
/// media always targets the default OpenRouter endpoint regardless of a custom
/// chat endpoint; the request never carries a routing block. Returns `None`
/// when no API key is configured (no video transcription possible).
#[must_use]
fn build_media_transcriber(config: &crate::config::ConfigData) -> Option<MediaTranscriber> {
    config.provider_key.as_deref().and_then(trimmed_or_none)?;
    Some(MediaTranscriber::new(
        crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string(),
        crate::config::VIDEO_TRANSCRIPTION_MODEL.to_string(),
    ))
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
            expected: Option<serde_json::Value>,
        }

        let cases = [
            Case {
                name: "single_provider",
                order: "openai",
                expected: Some(serde_json::json!({
                    "order": ["openai"],
                    "allow_fallbacks": false,
                })),
            },
            Case {
                name: "multiple_providers",
                order: "openai, anthropic, google",
                expected: Some(serde_json::json!({
                    "order": ["openai", "anthropic", "google"],
                    "allow_fallbacks": false,
                })),
            },
            Case {
                name: "whitespace_only_yields_none",
                order: "  , ,  ",
                expected: None,
            },
            Case {
                name: "empty_string_yields_none",
                order: "",
                expected: None,
            },
            Case {
                name: "leading_trailing_whitespace",
                order: "  openai  ",
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
                expected: Some(serde_json::json!({
                    "order": ["google-gemini"],
                    "allow_fallbacks": false,
                })),
            },
        ];

        for c in &cases {
            assert_eq!(
                provider_routing_json(c.order),
                c.expected,
                "case '{}': provider_routing_json({:?})",
                c.name,
                c.order,
            );
        }
    }
}
