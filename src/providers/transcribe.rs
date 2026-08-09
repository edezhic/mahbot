/// Transcribes media (images via data URIs, videos via public URLs) into text
/// descriptions during the enrichment phase, so the main agent loop only sees
/// text. The API key is read from the live config by
/// [`bearer_auth_header()`](crate::util::http::bearer_auth_header) at request
/// time, so config reloads take effect immediately without recreating the
/// transcriber.
#[derive(Clone)]
pub struct MediaTranscriber {
    api_url: String,
    model: String,
    provider_route: Option<String>,
}

/// Overall cap for one video transcription (ephemeral upload + model call).
/// Bounds the fail-open stall of both callers — the enrichment annotation and
/// the video tool-result path — well under the worst-case sum of the upload
/// bridge's per-host timeouts.
const VIDEO_TRANSCRIPTION_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(3);

impl MediaTranscriber {
    #[must_use]
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

    /// Call the vision-capable model to describe the image, returning a text
    /// description suitable for embedding inline.
    pub async fn transcribe(&self, image_data_uri: &str) -> anyhow::Result<String> {
        self.transcribe_media(
            serde_json::json!({"type": "image_url", "image_url": {"url": image_data_uri}}),
        )
        .await
    }

    /// Call the vision-capable model to describe a video referenced by a public
    /// URL, returning a text description suitable for embedding inline.
    async fn transcribe_video(&self, video_url: &str) -> anyhow::Result<String> {
        self.transcribe_media(
            serde_json::json!({"type": "video_url", "video_url": {"url": video_url}}),
        )
        .await
    }

    /// Single-attempt media transcription: POST the content part to the
    /// provider and parse the assistant text. Reasoning is disabled so a
    /// reasoning model cannot burn the token budget and return empty content.
    /// Empty model output is treated as a failure so every caller falls back
    /// to its annotation instead of rendering an empty description.
    async fn transcribe_media(&self, content_part: serde_json::Value) -> anyhow::Result<String> {
        let prompt = crate::prompt::load_prompt("media_transcription.md");
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        content_part
                    ]
                }
            ],
            "max_tokens": 2048,
            "reasoning": {"enabled": false},
        });

        if let Some(route) = &self.provider_route
            && let Some(routing) = crate::providers::provider_routing_json(route, false)
        {
            body["provider"] = routing;
        }

        // NOTE: `post_json_to_provider` returns non-2xx responses as typed
        // [`HttpError`](crate::util::error::HttpError) (accessible via
        // `downcast_ref`).  This is safe because the error is caught by the
        // fail-open callers (enrichment annotation, video tool result) which
        // log a warning and fall back — it never reaches the retry logic in
        // the provider layer.
        let result =
            crate::util::http::post_json_to_provider(&self.chat_url(), &body, "transcription")
                .await?;

        let text = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        if text.is_empty() {
            anyhow::bail!("media transcription returned empty content");
        }

        // Transcription text is embedded in tool outcomes and enriched messages
        // that the trusted marker scans parse — neutralize any marker-shaped
        // substring the model quoted from the media (the maximally-detailed
        // prompt explicitly captures on-screen text) so it cannot be misparsed
        // as a media reference. Applies to image and video alike.
        Ok(scrub_marker_like(&text))
    }
}

/// Transcribe a local video file — the shared fail-open helper for the
/// inbound-enrichment annotation and the video tool-result paths. Uploads the
/// file via the ephemeral host bridge and describes the returned public URL,
/// returning `None` on unavailable transcriber, unsupported format,
/// upload/model failure, or empty output so callers degrade to their plain
/// annotation. The whole flow is capped by [`VIDEO_TRANSCRIPTION_TIMEOUT`] so
/// both callers are uniformly bounded.
pub(crate) async fn transcribe_video_file(path: &std::path::Path) -> Option<String> {
    let transcriber = crate::providers::media_transcriber()?;
    if !crate::util::is_transcribable_video(path) {
        tracing::debug!(
            path = %path.display(),
            "Video format not supported by the transcription provider — skipping"
        );
        return None;
    }
    match tokio::time::timeout(VIDEO_TRANSCRIPTION_TIMEOUT, async {
        let url = crate::util::upload_bridge::upload_video_ephemeral_typed(path).await?;
        transcriber.transcribe_video(&url).await
    })
    .await
    {
        Ok(Ok(text)) => Some(text),
        Ok(Err(e)) => {
            tracing::warn!(path = %path.display(), error = %e, "Video transcription failed");
            None
        }
        Err(_) => {
            tracing::warn!(path = %path.display(), "Video transcription timed out");
            None
        }
    }
}

/// Neutralize marker-shaped substrings (e.g. `[IMAGE:`, `[video:`) in LLM
/// transcription text so the media-marker scans cannot treat quoted text as a
/// real media reference. Case-insensitive: the Telegram mirror path's marker
/// regex matches any case.
fn scrub_marker_like(text: &str) -> String {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\[(image|audio|video):").expect("marker scrub regex must compile")
    });
    RE.replace_all(text, "($1:").to_string()
}
