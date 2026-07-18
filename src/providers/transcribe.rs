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
