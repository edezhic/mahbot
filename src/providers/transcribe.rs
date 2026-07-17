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
///
/// Supports a configurable fallback chain: multiple models are tried in order,
/// each with its own retry loop.  Billing/quota errors abort the entire chain;
/// model-specific errors (404, 400) skip to the next model; transient errors
/// (rate limits, 5xx) retry with exponential backoff on the current model.
#[derive(Clone)]
pub struct AudioTranscriber {
    api_url: String,
    models: Vec<String>,
    provider_route: Option<String>,
}

impl AudioTranscriber {
    #[must_use]
    pub(crate) fn new(
        api_url: String,
        models: Vec<String>,
        provider_route: Option<String>,
    ) -> Self {
        Self {
            api_url,
            models,
            provider_route,
        }
    }

    /// The API base URL (e.g. `https://openrouter.ai/api/v1`).
    fn base_url(&self) -> String {
        crate::providers::ensure_base_url(&self.api_url)
    }

    /// Transcribe an audio file, returning the transcription text.
    ///
    /// Uses OpenRouter's JSON API format: base64-encodes the audio file and
    /// sends it as `input_audio.data` with the appropriate format string.
    ///
    /// Models are tried in order from the configured list (or the single
    /// fallback model if the list is empty).  Each model gets its own retry
    /// loop with exponential backoff and jitter.  Billing/quota errors abort
    /// the entire chain; model-specific errors skip to the next model.
    #[allow(clippy::too_many_lines)]
    pub async fn transcribe(&self, file_path: &Path) -> anyhow::Result<String> {
        // ── Retry parameters (per-model) ────────────────────────────
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

        // Base64-encode the audio bytes (done once, cached across all models).
        let encoded = base64::engine::general_purpose::STANDARD.encode(&file_bytes);

        let url = format!("{}/audio/transcriptions", self.base_url());

        // ── Per-model fallback chain ────────────────────────────────
        let mut all_failures: Vec<String> = Vec::new();

        for (model_idx, model) in self.models.iter().enumerate() {
            let mut body = serde_json::json!({
                "model": model,
                "input_audio": {
                    "data": encoded,
                    "format": format,
                },
            });

            if let Some(route) = &self.provider_route
                && let Some(routing) = crate::providers::provider_routing_json(route, false)
            {
                body["provider"] = routing;
            }

            let mut backoff_ms = BASE_BACKOFF_MS;
            let mut model_failures: Vec<String> = Vec::new();

            tracing::debug!(
                model = %model,
                model_index = model_idx,
                total_models = self.models.len(),
                "Audio transcription: trying model"
            );

            for attempt in 0..=MAX_RETRIES {
                match crate::util::http::post_json_to_provider(&url, &body, "audio transcription")
                    .await
                {
                    Ok(json) => {
                        let text = json
                            .get("text")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| anyhow::anyhow!("empty transcription response"))?;

                        if attempt > 0 {
                            tracing::info!(
                                attempt,
                                model = %model,
                                "Audio transcription succeeded after retry"
                            );
                        } else if model_idx > 0 {
                            tracing::info!(
                                model = %model,
                                model_index = model_idx,
                                "Audio transcription succeeded with fallback model",
                            );
                        }
                        return Ok(text);
                    }
                    Err(e) => {
                        let class = crate::providers::reliable::classify_transcriber_err(&e);
                        let error_detail = e.to_string();
                        let reason = class.reason_label();

                        model_failures.push(format!(
                            "attempt {}/{}: {}; error={}",
                            attempt + 1,
                            MAX_RETRIES + 1,
                            reason,
                            error_detail,
                        ));

                        // AbortChain → fatal (auth, billing), abort everything.
                        if class == crate::providers::reliable::TranscriberErrorClass::AbortChain {
                            tracing::warn!(
                                model = %model,
                                error = %error_detail,
                                "Audio transcription fatal error (billing/auth), aborting chain"
                            );
                            all_failures
                                .push(format!("[model: {model}] {}", model_failures.join("; ")));
                            anyhow::bail!(
                                "Audio transcription failed (fatal) after {} model(s).\n{}",
                                model_idx + 1,
                                all_failures.join("\n"),
                            );
                        }

                        // SkipModel → model-specific error, try next model.
                        if class == crate::providers::reliable::TranscriberErrorClass::SkipModel {
                            tracing::warn!(
                                model = %model,
                                error = %error_detail,
                                "Audio transcription model-specific error, trying next model"
                            );
                            break;
                        }

                        // Last attempt exhausted for this model → next model.
                        if attempt == MAX_RETRIES {
                            tracing::warn!(
                                model = %model,
                                error = %error_detail,
                                "Audio transcription exhausted retries for model"
                            );
                            break;
                        }

                        // ── RetryModel: transient, backoff and retry ──
                        if let Some(http_err) = e.downcast_ref::<HttpError>()
                            && http_err.status == 429
                        {
                            tracing::debug!(
                                body = %http_err.body,
                                "HTTP 429 body did not match any non-retryable hint"
                            );
                        }

                        let wait = crate::providers::reliable::ReliableProvider::compute_backoff(
                            backoff_ms, &e,
                        );

                        if !crate::shutdown::sleep_or_shutdown(std::time::Duration::from_millis(
                            wait,
                        ))
                        .await
                        {
                            tracing::info!(
                                "Audio transcription retry aborted — shutdown in progress"
                            );
                            all_failures
                                .push(format!("[model: {model}] {}", model_failures.join("; ")));
                            anyhow::bail!(
                                "Audio transcription cancelled (shutdown) after {} model(s).\n{}",
                                model_idx + 1,
                                all_failures.join("\n"),
                            );
                        }

                        tracing::warn!(
                            model = %model,
                            attempt = attempt + 1,
                            reason,
                            error = %error_detail,
                            "Audio transcription failed, retrying"
                        );

                        backoff_ms = backoff_ms.saturating_mul(2);
                    }
                }
            }

            // Accumulate model-level failures for the final error message.
            all_failures.push(format!("[model: {model}] {}", model_failures.join("; ")));
        }

        anyhow::bail!(
            "Audio transcription failed after {} model(s).\n{}",
            self.models.len(),
            all_failures.join("\n"),
        )
    }
}
