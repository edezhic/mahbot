use crate::Tool;
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;

/// Tool for generating videos via OpenRouter's async videos API.
///
/// Submits a video generation job, polls for completion, downloads the
/// resulting video file, and returns its path so the agent can send it via
/// `[VIDEO:path]` in its reply.
pub struct VideoGenTool;

#[async_trait]
impl Tool for VideoGenTool {
    fn name(&self) -> &'static str {
        "video_gen"
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[VIDEO:")
    }

    fn format_output(&self, output: &str) -> String {
        // The result is a small marker plus a bounded transcription
        // description — the default 5 KB truncation would elide the middle of
        // the description the Artist needs to reason about its own output.
        output.to_string()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "prompt": {
                    "type": "string",
                    "description": "Text description of the video to generate"
                },
                "images": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Path to a reference/start image for image-to-video generation (single image only)",
                    "maxItems": 1
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in seconds (model-dependent; valid values come from the active model's capability block)"
                },
                "resolution": {
                    "type": "string",
                    "description": "Desired resolution for the generated video (model-dependent; valid values come from the active model's capability block)"
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio for the generated video (model-dependent; valid values come from the active model's capability block)"
                },
                "size": {
                    "type": "string",
                    "pattern": "\\d+x\\d+",
                    "description": "Exact size in WxH format (e.g. 1280x720; model-dependent)"
                },
                "generate_audio": {
                    "type": "boolean",
                    "description": "Whether to generate an audio track (model-dependent)"
                },
                "seed": {
                    "type": "integer",
                    "description": "Seed for reproducible generation"
                }
            }),
            &["prompt"],
        )
    }

    async fn execute(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        let prompt = super::get_str(&args, "prompt")?;

        let model = crate::config::CONFIG.video_model();

        let duration = super::get_opt_i64(&args, "duration");
        let resolution = super::get_opt_str(&args, "resolution");
        let aspect_ratio = super::get_opt_str(&args, "aspect_ratio");
        let size = super::get_opt_str(&args, "size");
        let generate_audio = super::get_opt_bool(&args, "generate_audio");
        let seed = super::get_opt_i64(&args, "seed");

        // Build the API base URL (strip /chat/completions if present)
        let endpoint = crate::config::CONFIG.provider_endpoint();
        let api_base = crate::providers::ensure_base_url(&endpoint);

        // ── Build the video generation request body ─────────────────────
        let mut body = json!({
            "model": model,
            "prompt": prompt,
        });

        if let Some(d) = duration {
            body["duration"] = json!(d);
        }

        if let Some(r) = resolution {
            body["resolution"] = json!(r);
        }

        if let Some(a) = aspect_ratio {
            body["aspect_ratio"] = json!(a);
        }

        if let Some(s) = size {
            body["size"] = json!(s);
        }

        if let Some(g) = generate_audio {
            body["generate_audio"] = json!(g);
        }

        if let Some(s) = seed {
            body["seed"] = json!(s);
        }

        // Optional: add reference image via input_references
        let images: Vec<String> = super::get_str_array(&args, "images");

        if images.len() > 1 {
            anyhow::bail!(
                "Video generation supports only a single reference image (received {}). \
                 Retry with exactly one image.",
                images.len(),
            );
        }

        if let Some(img_path) = images.first() {
            match crate::util::load_reference_image(
                Path::new(img_path),
                super::MAX_REFERENCE_IMAGE_BYTES,
            )
            .await
            {
                Ok(data_uri) => {
                    body["input_references"] = json!([{
                        "type": "image_url",
                        "image_url": { "url": data_uri }
                    }]);
                }
                Err(e) => {
                    tracing::warn!(%img_path, error = %e, "Failed to load reference image for video gen");
                }
            }
        }

        let video_bytes =
            super::fetch_async_video(&api_base, &body, super::VideoJobLabels::GENERATION).await?;

        // Save to workspace/generated/ and format the media marker. The marker
        // stays first; the transcription is appended for the Artist to reason
        // about its own output (fail-open: marker-only on transcription failure).
        let output_path = super::save_generated_file(ws, &video_bytes, "video", "mp4").await?;
        let marker = self.format_media_result(&output_path);
        Ok(super::format_video_result(marker, &output_path).await)
    }
}
