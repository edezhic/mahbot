use anyhow::Context;
use base64::Engine;
use std::path::Path;

use crate::util::error::HttpError;

/// Shared internal fields for media transcribers (image/audio).
///
/// The API key is read from the live config by [`bearer_auth_header()`](crate::util::http::bearer_auth_header)
/// at request time, so config reloads take effect immediately without recreating
/// the transcriber.
#[derive(Clone)]
pub(crate) struct MediaTranscriber {
    api_url: String,
    model: String,
    provider_route: Option<String>,
}

impl MediaTranscriber {
    pub(crate) fn new(api_url: String, model: String, provider_route: Option<String>) -> Self {
        Self {
            api_url,
            model,
            provider_route,
        }
    }

    fn chat_url(&self) -> String {
        crate::providers::ensure_chat_completions_url(&self.api_url)
    }
}

/// Transcribes images (via a vision-capable model) into text descriptions
/// during the enrichment phase, so the main agent loop only sees text.
#[derive(Clone)]
pub struct ImageTranscriber {
    inner: MediaTranscriber,
}

impl ImageTranscriber {
    #[must_use]
    pub(crate) const fn from_inner(inner: MediaTranscriber) -> Self {
        Self { inner }
    }

    /// Call the vision-capable model to describe the image, returning a text
    /// description suitable for embedding inline.
    pub async fn transcribe(&self, image_data_uri: &str) -> anyhow::Result<String> {
        let mut body = serde_json::json!({
            "model": self.inner.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image concisely."},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            "max_tokens": 512,
        });

        if let Some(route) = &self.inner.provider_route
            && let Some(routing) = crate::providers::provider_routing_json(route, false)
        {
            body["provider"] = routing;
        }

        // NOTE: `post_json_to_provider` returns non-2xx responses as typed
        // [`HttpError`](crate::util::error::HttpError) (accessible via
        // `downcast_ref`).  This is safe because the error is caught by
        // `handle_non_multimodal_image` (in channels/mod.rs) which logs a
        // warning and falls back to a generic annotation — it never reaches
        // the retry logic in the provider layer.
        let result = crate::util::http::post_json_to_provider(
            &self.inner.chat_url(),
            &body,
            "transcription",
        )
        .await?;

        let text = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(text)
    }
}

/// Transcribes audio files via API into text
/// during message enrichment, so the main agent sees only text.
#[derive(Clone)]
pub struct AudioTranscriber {
    inner: MediaTranscriber,
}

impl AudioTranscriber {
    #[must_use]
    pub(crate) const fn from_inner(inner: MediaTranscriber) -> Self {
        Self { inner }
    }

    /// Transcribe an audio file, returning the transcription text.
    ///
    /// Uses OpenRouter's JSON API format: base64-encodes the audio file and
    /// sends it as `input_audio.data` with the appropriate format string.
    ///
    /// Retry behaviour follows the same pattern as [`ReliableProvider`](crate::providers::reliable::ReliableProvider):
    /// exponential backoff with jitter, Retry-After header support,
    /// non-retryable error classification, and shutdown coordination.
    /// Audio payloads are larger than typical LLM calls so we use fewer
    /// retries (3) and a more conservative base backoff (1000 ms).
    #[allow(clippy::too_many_lines)]
    pub async fn transcribe(&self, file_path: &Path) -> anyhow::Result<String> {
        // ── Retry parameters ──────────────────────────────────────────
        // Mirrors the pattern in ReliableProvider::chat but adapted for
        // audio transcription (larger payloads → conservative backoff).
        const MAX_RETRIES: u32 = 3;
        const BASE_BACKOFF_MS: u64 = 1000;

        let file_bytes = tokio::fs::read(file_path)
            .await
            .context("failed to read audio file")?;

        // Determine the audio format from the file extension.
        let format = match file_path.extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("oga") => "ogg",
            Some(e) => e,
            None => "wav",
        }
        .to_lowercase();

        // Base64-encode the audio bytes (done once, cached across retries).
        let encoded = base64::engine::general_purpose::STANDARD.encode(&file_bytes);

        let mut body = serde_json::json!({
            "model": self.inner.model,
            "input_audio": {
                "data": encoded,
                "format": format,
            },
        });

        if let Some(route) = &self.inner.provider_route
            && let Some(routing) = crate::providers::provider_routing_json(route, false)
        {
            body["provider"] = routing;
        }

        let base = crate::providers::ensure_base_url(&self.inner.api_url);
        let url = format!("{base}/audio/transcriptions");

        let mut failures: Vec<String> = Vec::new();
        let mut backoff_ms = BASE_BACKOFF_MS;

        for attempt in 0..=MAX_RETRIES {
            match crate::util::http::post_json_to_provider(&url, &body, "audio transcription").await
            {
                Ok(json) => {
                    let text = json
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| anyhow::anyhow!("empty transcription response"))?;

                    if attempt > 0 {
                        tracing::info!(attempt, "Audio transcription recovered after retry");
                    }
                    return Ok(text);
                }
                Err(e) => {
                    let class = crate::providers::reliable::classify_err(&e);
                    let error_detail = e.to_string();
                    let reason = class.reason_label();

                    failures.push(format!(
                        "attempt {}/{}: {}; error={}",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        reason,
                        error_detail,
                    ));

                    // Permanent errors abort immediately — no point retrying.
                    if class == crate::providers::reliable::ErrorClass::NonRetryable {
                        tracing::warn!(
                            error = %error_detail,
                            "Audio transcription failed with non-retryable error, aborting"
                        );
                        break;
                    }

                    // Last attempt exhausted — exit loop.
                    if attempt == MAX_RETRIES {
                        tracing::warn!(
                            error = %error_detail,
                            "Audio transcription exhausted retries"
                        );
                        break;
                    }

                    // When a 429 body doesn't match any known non-retryable
                    // hint, classify_err falls through to Retryable silently.
                    // Log the body at debug so operators can detect provider-side
                    // error-format changes (e.g., "quota_exhausted" → "credit_limit_reached").
                    if class == crate::providers::reliable::ErrorClass::Retryable
                        && let Some(http_err) = e.downcast_ref::<HttpError>()
                        && http_err.status == 429
                    {
                        tracing::debug!(
                            body = %http_err.body,
                            "HTTP 429 body did not match any non-retryable \
                             hint — treating as retryable"
                        );
                    }

                    // Compute backoff duration with Retry-After header support.
                    // Reuses ReliableProvider::compute_backoff which applies
                    // jitter and Retry-After clamping with the same formula.
                    let wait = crate::providers::reliable::ReliableProvider::compute_backoff(
                        backoff_ms, &e,
                    );

                    if !crate::shutdown::sleep_or_shutdown(std::time::Duration::from_millis(wait))
                        .await
                    {
                        tracing::info!("Audio transcription retry aborted — shutdown in progress");
                        break;
                    }

                    tracing::warn!(
                        attempt = attempt + 1,
                        reason,
                        error = %error_detail,
                        "Audio transcription failed, retrying"
                    );

                    backoff_ms = backoff_ms.saturating_mul(2);
                }
            }
        }

        anyhow::bail!(
            "Audio transcription failed after {} attempts.\n{}",
            failures.len(),
            failures.join("\n"),
        )
    }
}
