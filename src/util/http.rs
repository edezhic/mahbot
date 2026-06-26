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
/// Returns `None` when no provider key is configured (the key is missing or
/// empty).  Callers should propagate this as a clear error message rather than
/// sending a request with a blank `"Bearer "` header that would produce an
/// opaque 401 response.
///
/// Used by [`crate::tools::image_gen::ImageGenTool`],
/// [`crate::tools::video_gen::VideoGenTool`], and the
/// [`ImageTranscriber`](crate::providers::transcribe::ImageTranscriber) /
/// [`AudioTranscriber`](crate::providers::transcribe::AudioTranscriber) — all
/// OpenRouter-based tools that require this header.  Any future
/// OpenRouter-based media tools should reuse this helper as well.
#[must_use]
pub fn bearer_auth_header() -> Option<String> {
    let key = crate::config::CONFIG.provider_key()?;
    if key.is_empty() {
        return None;
    }
    Some(format!("Bearer {key}"))
}

/// Extract the first 4xx HTTP status code from a formatted error message.
///
/// Status codes appear in messages like
/// "OpenAI API error (400): ..." or "429 Too Many Requests".
/// This scans for any 4xx number; it is the only viable approach since
/// the typed `reqwest::Error` chain is not always preserved by the caller.
///
/// Used by:
/// - [`classify_err`](crate::providers::reliable) (string-fallback error path)
/// - [`VideoGenTool`](crate::tools::video_gen::VideoGenTool) (402 credit detection)
///
/// Both consumers depend on the error format produced by [`check_response`]:
/// `"{error_context} API error ({status}): {preview}"`.
/// This shared extractor centralises the string-based status parsing so
/// that format changes only need updating in one place.
///
/// # Limitations
///
/// - May produce false positives if error messages contain other 400-500
///   range numbers (e.g. a field value of 400).  This is an accepted risk
///   shared by all string-parsing approaches.
/// - A more robust solution would refactor `check_response` to return a
///   typed `ProviderError` with a structured status code — this is a
///   future improvement left for when the error handling infrastructure
///   is revisited more broadly.
#[must_use]
pub(crate) fn extract_http_status(msg: &str) -> Option<u16> {
    msg.split(|c: char| !c.is_ascii_digit())
        .filter_map(|w| w.parse::<u16>().ok())
        .find(|&code| (400..500).contains(&code))
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
/// **Important:** Both [`extract_http_status`] and callers that parse status
/// codes from this error format (e.g. [`VideoGenTool`](crate::tools::video_gen::VideoGenTool))
/// depend on exactly this format.  Any format change will silently break
/// those consumers.
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
        let error_text = response.text().await.unwrap_or_else(|e| {
            tracing::warn!(?e, "Failed to read response body");
            "failed to read response body".to_string()
        });
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
    let auth = bearer_auth_header()
        .ok_or_else(|| anyhow::anyhow!("{error_context}: provider API key is not configured"))?;
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
    let auth = bearer_auth_header()
        .ok_or_else(|| anyhow::anyhow!("{error_context}: provider API key is not configured"))?;
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
    let auth = bearer_auth_header()
        .ok_or_else(|| anyhow::anyhow!("{error_context}: provider API key is not configured"))?;
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
/// Used by [`crate::tools::image_gen::ImageGenTool`], [`crate::tools::video_gen::VideoGenTool`], [`crate::tools::web_search::WebSearchTool`] (Firecrawl API),
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

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_http_status_detects_4xx() {
        // ── Table-driven tests for extract_http_status ──
        // Format: (message_snippet, expected_status_code, description)
        //
        // The test covers:
        // - Standard 4xx codes from check_response format
        // - Two-digit and three-digit 4xx codes
        // - Non-4xx numbers (2xx, 5xx) that should be ignored
        // - No status code at all
        // - Numbers embedded in other contexts (field values, etc.)
        let cases: Vec<(&str, Option<u16>, &str)> = vec![
            (
                "API error (402): Insufficient credits",
                Some(402),
                "402 payment required — primary use case",
            ),
            (
                "Video generation submission API error (402):",
                Some(402),
                "402 in video_gen format",
            ),
            (
                "OpenAI API error (400): Bad Request",
                Some(400),
                "400 bad request",
            ),
            (
                "API error (401): Unauthorized",
                Some(401),
                "401 unauthorized",
            ),
            ("API error (403): Forbidden", Some(403), "403 forbidden"),
            ("API error (404): Not Found", Some(404), "404 not found"),
            (
                "API error (408): Request Timeout",
                Some(408),
                "408 request timeout",
            ),
            (
                "API error (429): Too Many Requests",
                Some(429),
                "429 too many requests",
            ),
            (
                "500 Server Error",
                None,
                "5xx ignored — not in 400-500 range",
            ),
            ("502 Bad Gateway", None, "5xx ignored"),
            ("200 OK", None, "2xx ignored"),
            ("connection reset", None, "no status code at all"),
            ("", None, "empty string"),
            (
                "field value is 400 but should be rejected",
                Some(400),
                "number in body text within range — accepted false positive",
            ),
            ("error code 1113", None, "four-digit number outside range"),
            (
                "HTTP 402 Payment Required",
                Some(402),
                "bare status in message without parens",
            ),
            ("Status 429", Some(429), "bare two-digit-then-three-digit"),
        ];

        for (msg, expected, description) in cases {
            let result = extract_http_status(msg);
            assert_eq!(
                result, expected,
                "extract_http_status({msg:?}): expected {expected:?}, got {result:?} — {description}",
            );
        }
    }

    #[test]
    fn extract_http_status_handles_adjacent_text() {
        // Numbers adjacent to other text without delimiters
        assert_eq!(
            extract_http_status("API error (402)"),
            Some(402),
            "parenthesised status"
        );
        assert_eq!(
            extract_http_status("code402"),
            Some(402),
            "digits adjacent to text without delimiter"
        );
        assert_eq!(
            extract_http_status("402error"),
            Some(402),
            "digits followed by text"
        );
        assert_eq!(
            extract_http_status("error402error"),
            Some(402),
            "digits surrounded by text"
        );
        assert_eq!(
            extract_http_status("value_is_400"),
            Some(400),
            "400 as part of identifier — accepted false positive"
        );
    }
}
