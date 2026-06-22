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

/// Check that an HTTP response has a successful status code.
///
/// If the status is 2xx the response is returned unmodified for further
/// processing (body reading, parsing, etc.).  On non-2xx the response body is
/// consumed, truncated to 500 characters, and included in the error message.
///
/// # Error format
///
/// `"{error_context} API error ({status}): {preview}"`
///
/// **Important:** [`crate::tools::video_gen`] string-matches `"(402)"` in this
/// exact format to detect insufficient credits.  Any format change will silently
/// break that logic.
///
/// # Errors
///
/// - Non-2xx status: the formatted error described above.
async fn check_response(
    response: reqwest::Response,
    error_context: &str,
) -> anyhow::Result<reqwest::Response> {
    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        let preview = crate::util::truncate(&error_text, 500);
        anyhow::bail!("{error_context} API error ({status}): {preview}");
    }
    Ok(response)
}

/// Parse a JSON response body string, producing a detailed error message on
/// failure that includes the body length and a preview.
///
/// # Error format
///
/// `"{error_context} response parse error: {e}\nraw response body (N): {body:.500}"`
///
/// # Errors
///
/// - Invalid JSON: the formatted error described above.
pub(crate) fn parse_json_response(
    body_text: &str,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    serde_json::from_str(body_text).map_err(|e| {
        anyhow::anyhow!(
            "{error_context} response parse error: {e}\nraw response body ({}): {body_text:.500}",
            body_text.len(),
        )
    })
}

/// POST JSON to a provider endpoint, check the status, and parse the response
/// as JSON.
///
/// Uses [`bearer_auth_header()`] for the Authorization header and
/// [`media_http_client()`] for the HTTP client.  Future media tools that need
/// the same POST → status-check → parse pattern should reuse this helper
/// instead of duplicating the boilerplate.
///
/// # Errors
///
/// - Transport errors: `"{error_context} request failed: {err}"`
/// - Non-2xx status: `"{error_context} API error ({status}): {preview}"` (first 500 chars)
/// - JSON parse failure: includes the raw response body length and a preview in
///   the error message for easier debugging.
pub async fn post_json_to_provider(
    url: &str,
    body: &serde_json::Value,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    let auth = bearer_auth_header();
    let client = media_http_client();

    let response = client
        .post(url)
        .header("Authorization", &auth)
        .json(body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} request failed: {e}"))?;

    let response = check_response(response, error_context).await?;

    let body_text = response
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))?;

    parse_json_response(&body_text, error_context)
}

/// GET JSON from a provider endpoint, check the status, and parse the response
/// as JSON.
///
/// Uses [`bearer_auth_header()`] for the Authorization header and
/// [`media_http_client()`] for the HTTP client.  Future consumers that need the
/// GET → status-check → parse pattern should reuse this helper instead of
/// duplicating the boilerplate.
///
/// # Errors
///
/// - Transport errors: `"{error_context} request failed: {err}"`
/// - Non-2xx status: `"{error_context} API error ({status}): {preview}"` (first 500 chars)
/// - JSON parse failure: includes the raw response body length and a preview in
///   the error message for easier debugging.
pub async fn get_json_from_provider(
    url: &str,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    let auth = bearer_auth_header();
    let client = media_http_client();

    let response = client
        .get(url)
        .header("Authorization", &auth)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} request failed: {e}"))?;

    let response = check_response(response, error_context).await?;

    let body_text = response
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))?;

    parse_json_response(&body_text, error_context)
}

/// GET bytes from a provider endpoint, check the status, and return the raw
/// binary response.
///
/// Uses [`bearer_auth_header()`] for the Authorization header and
/// [`media_http_client()`] for the HTTP client.  Useful for downloading
/// generated media files or other binary content from provider endpoints.
///
/// # Errors
///
/// - Transport errors: `"{error_context} request failed: {err}"`
/// - Non-2xx status: `"{error_context} API error ({status}): {preview}"` (first 500 chars)
/// - Body read failure: `"{error_context} failed to read response body: {err}"`
pub async fn get_bytes_from_provider(url: &str, error_context: &str) -> anyhow::Result<Vec<u8>> {
    let auth = bearer_auth_header();
    let client = media_http_client();

    let response = client
        .get(url)
        .header("Authorization", &auth)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} request failed: {e}"))?;

    let response = check_response(response, error_context).await?;

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))
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
