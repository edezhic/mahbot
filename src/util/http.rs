//! Shared `reqwest::Client` construction with configurable timeouts.
//!
//! All call sites that need an HTTP client should use this helper instead of
//! building one from scratch.  `reqwest::Client` is designed to be created once
//! and reused — it maintains an internal connection pool, caches DNS
//! resolutions, and reuses TLS sessions.

use std::sync::OnceLock;
use std::time::Duration;

/// HTTP client shared by [`crate::tools::image_gen::ImageGenTool`],
/// [`crate::tools::video_gen::VideoGenTool`],
/// [`crate::tools::web_search::WebSearchTool`], [`crate::tools::exa_search::ExaSearchTool`], and
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
/// - [`classify_err`](crate::providers::reliable) (string-fallback error path for
///   errors that arrive without a [`HttpError`](super::error::HttpError) wrapper)
///
/// # Limitations
///
/// - May produce false positives if error messages contain other 400-500
///   range numbers (e.g. a field value of 400).  This is an accepted risk
///   shared by all string-parsing approaches.
/// - Primary use case (re-parsing [`check_response`] string errors) has been
///   eliminated — [`check_response`] now returns a typed
///   [`HttpError`](super::error::HttpError) directly.  This function
///   remains as a general-purpose fallback for genuinely string-only errors.
#[must_use]
pub(crate) fn extract_http_status(msg: &str) -> Option<u16> {
    msg.split(|c: char| !c.is_ascii_digit())
        .filter_map(|w| w.parse::<u16>().ok())
        .find(|&code| (400..500).contains(&code))
}

/// Safely read a response body on failure, returning a fallback string.
/// Logs a warning with the provided context and the underlying error.
pub(crate) async fn read_error_body(response: reqwest::Response, context: &str) -> String {
    response.text().await.unwrap_or_else(|e| {
        tracing::warn!(?e, "Failed to read {context} response body");
        "failed to read response body".to_string()
    })
}

/// Check that an HTTP response has a successful status code.
///
/// If the status is 2xx the response is returned unmodified for further
/// processing (body reading, parsing, etc.).  On non-2xx the response body is
/// consumed, truncated to 500 characters, and wrapped in a
/// [`HttpError`](super::error::HttpError) that preserves the status
/// code and body as typed fields (accessible via `err.downcast_ref`).
///
/// # Errors
///
/// - Non-2xx status: returns the response body as a [`HttpError`](super::error::HttpError).
async fn check_response(
    response: reqwest::Response,
    error_context: &str,
) -> anyhow::Result<reqwest::Response> {
    let status = response.status();
    if !status.is_success() {
        let error_text = read_error_body(response, error_context).await;
        let preview = crate::util::truncate(&error_text, 500);
        return Err(anyhow::Error::from(super::error::HttpError::new(
            status.as_u16(),
            error_context,
            preview,
            None,
        )));
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

/// Shared request boilerplate for provider API calls.
///
/// Extracts the Bearer auth header (from [`bearer_auth_header()`]), gets the
/// shared HTTP client (from [`media_http_client()`]), uses `build_request` to
/// construct the request, sends it, and checks the response status.  The
/// Authorization header is injected automatically — the closure only needs to
/// set the HTTP method, URL, and optional body.
async fn provider_request(
    error_context: &str,
    build_request: impl FnOnce(&reqwest::Client) -> reqwest::RequestBuilder,
) -> anyhow::Result<reqwest::Response> {
    let auth = bearer_auth_header()
        .ok_or_else(|| anyhow::anyhow!("{error_context}: provider API key is not configured"))?;
    let client = media_http_client();
    let response = build_request(client)
        .header("Authorization", &auth)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} request failed: {e}"))?;
    check_response(response, error_context).await
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
/// - Non-2xx status: returns a [`HttpError`](super::error::HttpError) with the status code and response body (first 500 chars), accessible via `err.downcast_ref::<HttpError>()`
/// - JSON parse failure: includes the raw response body length and a preview in the error message for easier debugging.
pub async fn post_json_to_provider(
    url: &str,
    body: &serde_json::Value,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    let response = provider_request(error_context, |client| client.post(url).json(body)).await?;

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
/// - Non-2xx status: returns a [`HttpError`](super::error::HttpError) with the status code and response body (first 500 chars), accessible via `err.downcast_ref::<HttpError>()`
/// - JSON parse failure: includes the raw response body length and a preview in
///   the error message for easier debugging.
pub async fn get_json_from_provider(
    url: &str,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    let response = provider_request(error_context, |client| client.get(url)).await?;

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
/// - Non-2xx status: returns a [`HttpError`](super::error::HttpError) with the status code and response body (first 500 chars), accessible via `err.downcast_ref::<HttpError>()`
/// - Body read failure: `"{error_context} failed to read response body: {err}"`
pub async fn get_bytes_from_provider(url: &str, error_context: &str) -> anyhow::Result<Vec<u8>> {
    let response = provider_request(error_context, |client| client.get(url)).await?;

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))
}

/// Return the shared media-generation HTTP client, initialising it on first
/// call with a 2-minute request timeout and a 10-second connection timeout.
///
/// Used by [`crate::tools::image_gen::ImageGenTool`], [`crate::tools::video_gen::VideoGenTool`], [`crate::tools::web_search::WebSearchTool`] / [`crate::tools::exa_search::ExaSearchTool`] (web search APIs),
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
/// # Panics
///
/// Panics if `reqwest::Client::builder()` fails (typically a TLS
/// initialization failure).  TLS failure is non-recoverable — if the
/// system's TLS stack is broken, nothing will work — so the process
/// should stop immediately rather than silently producing wrong
/// behaviour at runtime.
#[must_use]
pub fn build_http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to build HTTP client (TLS initialization failure)")
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

    #[tokio::test]
    async fn check_response_returns_http_error_on_non_2xx() {
        // Construct a mock HTTP response with a non-2xx status using
        // http::Response::builder() + the unconditional From impl on reqwest::Response.
        let http_resp = http::Response::builder()
            .status(402)
            .body("Insufficient credits: please top up your account".to_string())
            .unwrap();
        let resp = reqwest::Response::from(http_resp);

        let result = check_response(resp, "Video generation submission").await;
        assert!(result.is_err(), "expected error for 402 status");

        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<crate::util::error::HttpError>()
                .map(|e| e.status),
            Some(402),
        );
    }

    #[tokio::test]
    async fn check_response_truncates_long_body() {
        // Verify that bodies longer than 500 chars are truncated.
        let long_body = "x".repeat(1000);
        let http_resp = http::Response::builder()
            .status(400)
            .body(long_body.clone())
            .unwrap();
        let resp = reqwest::Response::from(http_resp);

        let result = check_response(resp, "test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let http_err = err.downcast_ref::<crate::util::error::HttpError>().unwrap();

        // The body should be truncated to 500 Unicode chars + "…" (1 char, 3 bytes).
        assert!(
            http_err.body.len() <= 503,
            "body should be truncated, got {} bytes",
            http_err.body.len()
        );
        assert!(
            http_err.body.len() < long_body.len(),
            "truncated body ({}) should be shorter than original ({})",
            http_err.body.len(),
            long_body.len(),
        );
        assert!(
            http_err.body.ends_with('…'),
            "truncated body should end with ellipsis"
        );
        assert_eq!(http_err.status, 400);
    }

    #[tokio::test]
    async fn check_response_returns_ok_on_2xx() {
        let http_resp = http::Response::builder()
            .status(200)
            .body(r#"{"ok": true}"#.to_string())
            .unwrap();
        let resp = reqwest::Response::from(http_resp);

        let result = check_response(resp, "test").await;
        assert!(result.is_ok(), "expected success for 200 status");
    }
}
