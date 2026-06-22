use crate::providers::error::ProviderError;
use anyhow::Context;
use std::path::Path;

/// Shared internal fields for media transcribers (image/audio).
///
/// The API key is read from the live config by [`bearer_auth_header()`](crate::util::http::bearer_auth_header)
/// at request time, so config reloads take effect immediately without recreating
/// the transcriber.
#[derive(Clone)]
pub(crate) struct MediaTranscriber {
    client: reqwest::Client,
    api_url: String,
    model: String,
    provider_route: Option<String>,
}

impl MediaTranscriber {
    pub(crate) fn new(api_url: String, model: String, provider_route: Option<String>) -> Self {
        Self {
            client: crate::util::http::media_http_client().clone(),
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
    pub(crate) fn from_inner(inner: MediaTranscriber) -> Self {
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
    pub(crate) fn from_inner(inner: MediaTranscriber) -> Self {
        Self { inner }
    }

    /// Transcribe an audio file, returning the transcription text.
    pub async fn transcribe(&self, file_path: &Path) -> anyhow::Result<String> {
        let file_bytes = tokio::fs::read(file_path)
            .await
            .context("failed to read audio file")?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio")
            .to_string();

        let file_part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.inner.model.clone());

        if let Some(route) = &self.inner.provider_route
            && let Some(routing) = crate::providers::provider_routing_json(route, false)
        {
            form = form.text("provider", routing.to_string());
        }

        let base = crate::providers::ensure_base_url(&self.inner.api_url);
        let url = format!("{base}/audio/transcriptions");
        let response = self
            .inner
            .client
            .post(&url)
            .header("Authorization", crate::util::http::bearer_auth_header())
            .multipart(form)
            .send()
            .await
            .context("audio transcription request failed")?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            anyhow::anyhow!("audio transcription failed to read response body: {e}")
        })?;

        if !status.is_success() {
            let provider_err =
                ProviderError::new(status.as_u16(), "audio transcription", &body, None);
            return Err(anyhow::Error::from(provider_err));
        }

        let json = crate::util::http::parse_json_response(&body, "audio transcription")?;

        json.get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("empty transcription response"))
    }
}
