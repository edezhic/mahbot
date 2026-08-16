//! Shared `reqwest::Client` construction with configurable timeouts.
//!
//! All call sites that need an HTTP client should use this helper instead of
//! building one from scratch.  `reqwest::Client` is designed to be created once
//! and reused — it maintains an internal connection pool, caches DNS
//! resolutions, and reuses TLS sessions.

use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

/// Install the ring crypto provider for rustls.
///
/// reqwest 0.13's `rustls-no-provider` feature leaves the process without a
/// default `CryptoProvider`; building any `reqwest::Client` without one fails
/// at build time ("No rustls crypto provider is configured"). Every client
/// factory must call this before constructing a client — including tests and
/// benches that build clients directly.
///
/// Idempotent: a process-local `OnceLock` guarantees the underlying
/// `install_default` runs at most once. The `AlreadyInstalled` error is also
/// ignored as a belt-and-suspenders — it can only surface if another code
/// path already installed a provider, in which case the process already has
/// one. Root certificates come from the OS trust store (macOS Keychain) via
/// rustls-platform-verifier — bundled roots are deliberately not used.
pub(crate) fn install_ring_provider() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// HTTP client shared by [`crate::tools::video_gen::VideoGenTool`],
/// [`crate::tools::web_search::WebSearchTool`], and
/// [`crate::providers::transcribe::MediaTranscriber`] — all call their
/// respective APIs with a 2-minute timeout.
///
/// If a future requirement needs different timeouts for a particular consumer,
/// simply remove this static and re-add separate `OnceLock` statics in the
/// relevant files (a trivial change — exactly the original pattern).
static MEDIA_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// Dedicated HTTP client for image-generation POSTs — a 10-minute request
/// timeout (generations routinely take 92–99 s and load spikes push further).
/// Kept separate from [`MEDIA_HTTP_CLIENT`] so the 2-minute cap of the other
/// media consumers (search, video, transcribe, catalog) is never raised.
static IMAGE_GEN_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

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
/// [`MediaTranscriber`](crate::providers::transcribe::MediaTranscriber) — all
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
    if !response.status().is_success() {
        let mut err = super::error::HttpError::from_response(response, error_context).await;
        // Truncate the body to keep error messages concise.
        err.body = crate::util::truncate(&err.body, 500);
        return Err(anyhow::Error::from(err));
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
/// [`media_http_client()`] for the HTTP client. `request_timeout` overrides
/// the client's 2-minute cap per request — the async-video tools pass the
/// remaining job window so large downloads on slow connections are not cut
/// short. Useful for downloading generated media files or other binary
/// content from provider endpoints.
///
/// # Errors
///
/// - Transport errors: `"{error_context} request failed: {err}"`
/// - Non-2xx status: returns a [`HttpError`](super::error::HttpError) with the status code and response body (first 500 chars), accessible via `err.downcast_ref::<HttpError>()`
/// - Body read failure: `"{error_context} failed to read response body: {err}"`
pub async fn get_bytes_from_provider(
    url: &str,
    error_context: &str,
    request_timeout: Duration,
) -> anyhow::Result<Vec<u8>> {
    let response = provider_request(error_context, |client| {
        client.get(url).timeout(request_timeout)
    })
    .await?;

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))
}

/// Return the shared media-generation HTTP client, initialising it on first
/// call with a 2-minute request timeout and a 10-second connection timeout.
///
/// Used by [`crate::tools::video_gen::VideoGenTool`], [`crate::tools::web_search::WebSearchTool`] (web search APIs),
/// and `MediaTranscriber` — all of which need the same timeout.  Image
/// generation uses its own dedicated 10-minute client
/// ([`image_gen_http_client`]).  If a future consumer requires a different
/// timeout it should call [`build_http_client`] directly with the
/// appropriate duration.
#[must_use]
pub fn media_http_client() -> &'static reqwest::Client {
    MEDIA_HTTP_CLIENT.get_or_init(|| build_http_client(Duration::from_mins(2)))
}

/// Return the dedicated image-generation HTTP client with a 10-minute request
/// timeout and a 10-second connection timeout.
///
/// Used exclusively by [`crate::tools::image_gen::ImageGenTool`]'s POST — slow
/// generations routinely take 92–99 s, so the shared 2-minute media client
/// would cut valid in-flight requests short. The image tool performs its own
/// retry/error handling on top of this client (see `image_gen.rs`).
#[must_use]
pub fn image_gen_http_client() -> &'static reqwest::Client {
    IMAGE_GEN_HTTP_CLIENT.get_or_init(|| build_http_client(Duration::from_mins(10)))
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
    install_ring_provider();
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to build HTTP client (rustls TLS initialization failure)")
}

/// Build an HTTP client for large model downloads: `total_timeout` per
/// request plus a deliberate 30-second connect timeout.
///
/// The 30s connect timeout intentionally diverges from [`build_http_client`]'s
/// 10s — model CDNs can be slow to accept connections, so a 10s fail-fast
/// would abort large downloads. Do not "simplify" this back to 10s.
///
/// Returns `Result` so each caller keeps its own error handling (propagate via
/// `?` or degrade gracefully). `total_timeout` is a per-site argument — the
/// call sites' values are load-bearing (their retry loops wrap downloads in
/// matching outer `tokio::time::timeout`) and must not be unified.
pub(crate) fn build_download_client(total_timeout: Duration) -> anyhow::Result<reqwest::Client> {
    install_ring_provider();
    reqwest::Client::builder()
        .timeout(total_timeout)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(anyhow::Error::from)
}

/// Size-check policy applied after a streaming download completes.
pub(crate) enum DownloadSizeCheck {
    /// Require downloaded bytes to equal the Content-Length header (when present).
    Exact,
    /// Require at least `min_bytes` downloaded (for servers that omit Content-Length).
    Min(u64),
    /// No size validation.
    None,
}

/// Stream a file download with on-the-fly SHA256 verification and atomic
/// tmp+rename — the single canonical path for model downloads.
///
/// * `expected_sha256` — empty skips verification (same semantics as
///   `verify_sha256`).
/// * `timeout` — per-request timeout; `None` defers to the client's.
/// * `progress` — invoked `progress(0, total)` before the first byte
///   ("started"), then per chunk; callers throttle as needed. `total` is 0
///   when the server omits Content-Length.
///
/// On failure the temp file is removed; on success it is atomically renamed
/// into place. (The bench-only sync downloader in `voice_pipeline_e2e_test.rs`
/// stays separate.)
pub(crate) async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    expected_sha256: &str,
    timeout: Option<Duration>,
    size_check: DownloadSizeCheck,
    mut progress: impl FnMut(u64, u64),
) -> anyhow::Result<()> {
    use anyhow::Context as _;
    use futures_util::StreamExt;
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncWriteExt;

    let mut request = client.get(url);
    if let Some(t) = timeout {
        request = request.timeout(t);
    }
    let response = request
        .send()
        .await
        .context("Failed to send download request")?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {} from {url}", response.status());
    }

    let content_length = response.content_length();
    let total_size = content_length.unwrap_or(0);

    // "Started" signal before the first byte is streamed.
    progress(0, total_size);

    let tmp = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .context("Failed to create temp file")?;

    let mut hasher = (!expected_sha256.is_empty()).then_some(Sha256::new());
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        let len = chunk.len() as u64;
        downloaded += len;
        if let Some(h) = &mut hasher {
            h.update(&chunk);
        }
        file.write_all(&chunk)
            .await
            .context("Failed to write download chunk")?;
        progress(downloaded, total_size);
    }

    file.flush().await?;
    file.sync_all().await?;
    drop(file);

    match size_check {
        DownloadSizeCheck::Exact => {
            if let Some(expected) = content_length
                && downloaded != expected
            {
                let _ = tokio::fs::remove_file(&tmp).await;
                anyhow::bail!(
                    "Download size mismatch: expected {expected} bytes, got {downloaded} bytes"
                );
            }
        }
        DownloadSizeCheck::Min(min) if downloaded < min => {
            let _ = tokio::fs::remove_file(&tmp).await;
            anyhow::bail!("Downloaded file too small: {downloaded} bytes");
        }
        DownloadSizeCheck::Min(_) | DownloadSizeCheck::None => {}
    }

    if let Some(h) = hasher {
        let actual_hash = format!("{:x}", h.finalize());
        if actual_hash != expected_sha256 {
            let _ = tokio::fs::remove_file(&tmp).await;
            anyhow::bail!(
                "SHA256 mismatch for {}: expected {expected_sha256}, got {actual_hash}",
                dest.display()
            );
        }
    }

    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("Failed to rename temp file to {}", dest.display()))?;
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn client_build_installs_ring_provider() {
        // reqwest 0.13's rustls-no-provider panics at Client build time if no
        // rustls crypto provider is installed. Every client factory must
        // install the ring provider first — this exercises the factory path
        // (and the idempotent double-install, since the process-global is
        // already set by earlier factories in the same test binary).
        install_ring_provider();
        let _client = build_http_client(Duration::from_secs(5));
        let _ = build_download_client(Duration::from_secs(60)).expect("download client builds");
    }
}
