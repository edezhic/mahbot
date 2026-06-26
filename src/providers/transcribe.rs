use anyhow::Context;
use base64::Engine;
use std::path::Path;

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

        // NOTE: This helper switches from `ProviderError` to `anyhow::bail` for
        // non-2xx responses.  This is safe because the error is caught by
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
    pub async fn transcribe(&self, file_path: &Path) -> anyhow::Result<String> {
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

        // Base64-encode the audio bytes.
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

        // NOTE: post_json_to_provider switches from ProviderError to anyhow::bail
        // for non-2xx responses. This is safe because the caller
        // (transcribe_audio_marker in channels/mod.rs) catches all errors and
        // falls back to "[Audio: ...]" — it never reaches the retry logic in the
        // provider layer.
        let json =
            crate::util::http::post_json_to_provider(&url, &body, "audio transcription").await?;

        json.get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("empty transcription response"))
    }
}
