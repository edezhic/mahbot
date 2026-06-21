//! Shared `reqwest::Client` construction with timeouts and graceful fallback.
//!
//! All call sites that need an HTTP client should use this helper instead of
//! building one from scratch.  `reqwest::Client` is designed to be created once
//! and reused — it maintains an internal connection pool, caches DNS
//! resolutions, and reuses TLS sessions.

use std::sync::OnceLock;
use std::time::Duration;

/// HTTP client shared by [`crate::tools::image_gen::ImageGenTool`],
/// [`crate::tools::video_gen::VideoGenTool`],
/// [`crate::tools::web_search::WebSearchTool`], and
/// [`crate::providers::transcribe::MediaTranscriber`] — all call their
/// respective APIs with a 2-minute timeout.
///
/// If a future requirement needs different timeouts for a particular consumer,
/// simply remove this static and re-add separate `OnceLock` statics in the
/// relevant files (a trivial change — exactly the original pattern).
static MEDIA_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// Return a Bearer Authorization header value built from the configured
/// provider API key.
///
/// Used by [`crate::tools::image_gen::ImageGenTool`],
/// [`crate::tools::video_gen::VideoGenTool`], and the
/// [`ImageTranscriber`](crate::providers::transcribe::ImageTranscriber) /
/// [`AudioTranscriber`](crate::providers::transcribe::AudioTranscriber) — all
/// OpenRouter-based tools that require this header.  Any future
/// OpenRouter-based media tools should reuse this helper as well.
///
/// The header name is always `"Authorization"` and the value is
/// `"Bearer {key}"` where `{key}` is the configured provider key
/// (falling back to an empty string if none is set, preserving the existing
/// behaviour).
#[must_use]
pub fn bearer_auth_header() -> String {
    format!(
        "Bearer {}",
        crate::config::CONFIG.provider_key().unwrap_or_default()
    )
}

/// Return the shared media-generation HTTP client, initialising it on first
/// call with a 2-minute request timeout and a 10-second connection timeout.
///
/// Used by [`crate::tools::image_gen::ImageGenTool`], [`crate::tools::video_gen::VideoGenTool`], [`crate::tools::web_search::WebSearchTool`] (Exa API),
/// and `MediaTranscriber` — all of which need the same timeout.  If a future
/// consumer requires a different timeout it should call
/// [`build_http_client`] directly with the appropriate duration.
#[must_use]
pub fn media_http_client() -> &'static reqwest::Client {
    MEDIA_HTTP_CLIENT.get_or_init(|| build_http_client(Duration::from_mins(2)))
}

/// Build a configured `reqwest::Client` with the given request `timeout` and a
/// 10-second connection timeout.
///
/// # Graceful fallback
///
/// If the builder fails (e.g. TLS/OpenSSL initialization issue), we log a
/// warning and fall back to [`reqwest::Client::new()`], which uses reqwest's
/// built-in defaults (including a 30-second request timeout but **no**
/// connection timeout).
#[must_use]
pub fn build_http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|error| {
            tracing::warn!(
                "Failed to build custom HTTP client: {error}; falling back to Client::new()"
            );
            reqwest::Client::new()
        })
}
