use crate::Tool;
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;

/// Number of polling attempts (10 min timeout = 20 attempts × 30s).
const MAX_POLL_ATTEMPTS: u32 = 20;

/// Tool for generating videos via OpenRouter's async videos API.
///
/// Submits a video generation job, polls for completion, downloads the
/// resulting video file, and returns its path so the agent can send it via
/// `[VIDEO:path]` in its reply.
pub struct VideoGenTool;

#[async_trait]
#[allow(clippy::too_many_lines)]
impl Tool for VideoGenTool {
    fn name(&self) -> &'static str {
        "video_gen"
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[VIDEO:")
    }

    fn format_output(&self, output: &str) -> String {
        let prefix = self
            .media_marker()
            .expect("VideoGenTool always has a media marker");
        format!("{prefix}{output}]")
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
                    "description": "Paths to reference/start images for image-to-video generation"
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in seconds (model-dependent, typically 4-15)"
                },
                "resolution": {
                    "type": "string",
                    "enum": ["480p", "720p", "1080p", "1K", "2K", "4K"],
                    "description": "Desired resolution for the generated video"
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"],
                    "description": "Aspect ratio for the generated video"
                },
                "size": {
                    "type": "string",
                    "pattern": "\\d+x\\d+",
                    "description": "Exact size in WxH format (e.g. 1280x720)"
                },
                "generate_audio": {
                    "type": "boolean",
                    "description": "Whether to generate audio track (default: true)"
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

        let model = crate::config::CONFIG.video_gen_model();

        let duration = super::get_opt_i64(&args, "duration");
        let resolution = super::get_opt_str(&args, "resolution");
        let aspect_ratio = super::get_opt_str(&args, "aspect_ratio");
        let size = super::get_opt_str(&args, "size");
        let generate_audio = super::get_opt_bool(&args, "generate_audio");
        let seed = super::get_opt_i64(&args, "seed");

        // Build the API base URL (strip /chat/completions if present)
        let endpoint = crate::config::CONFIG.provider_endpoint();
        let api_base = crate::providers::ensure_base_url(&endpoint);

        let auth = crate::util::http::bearer_auth_header();

        let client = crate::util::http::media_http_client();

        // ── Step 1: Submit video generation job ─────────────────────────
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

        let videos_url = format!("{api_base}/videos");
        let submit_resp = match client
            .post(&videos_url)
            .header("Authorization", &auth)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                anyhow::bail!("Video generation submission failed: {e}");
            }
        };

        let submit_status = submit_resp.status();

        // Handle insufficient credits specifically
        if submit_status.as_u16() == 402 {
            let error_text = submit_resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Insufficient OpenRouter credits for video generation (HTTP 402). \
                 Please add credits to your OpenRouter account and try again.\nAPI response: {error_text}"
            );
        }

        if !submit_status.is_success() {
            let error_text = submit_resp.text().await.unwrap_or_default();
            anyhow::bail!("Video generation submission error ({submit_status}): {error_text}");
        }

        let submit_body: serde_json::Value = match submit_resp.json().await {
            Ok(v) => v,
            Err(e) => {
                anyhow::bail!("Failed to parse submission response: {e}");
            }
        };

        // OpenRouter returns: { id, polling_url, status, ... }
        let job_id = match submit_body.get("id").and_then(|v| v.as_str()) {
            Some(id) => id.to_string(),
            None => {
                anyhow::bail!("No job ID in submission response: {submit_body}");
            }
        };

        let polling_url = match submit_body.get("polling_url").and_then(|v| v.as_str()) {
            Some(url) => url.to_string(),
            None => format!("{api_base}/videos/{job_id}"),
        };

        tracing::info!(%job_id, "Video generation job submitted");

        // ── Step 2: Poll for completion (~10 min timeout) ───────────────
        let mut result_url: Option<String> = None;

        for attempt in 1..=MAX_POLL_ATTEMPTS {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            let poll_resp = match client
                .get(&polling_url)
                .header("Authorization", &auth)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!(%job_id, attempt, error = %e, "Poll failed");
                    continue;
                }
            };

            let poll_body: serde_json::Value = match poll_resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!(%job_id, attempt, error = %e, "Poll parse failed");
                    continue;
                }
            };

            let status = poll_body
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            tracing::info!(%job_id, %status, attempt, "Video generation poll");

            if status == "completed" {
                // Download URL: OpenRouter provides unsigned_urls array or
                // a content endpoint at /api/v1/videos/{jobId}/content?index=0
                result_url = poll_body
                    .get("unsigned_urls")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or_else(|| {
                        // Fallback: use content endpoint
                        Some(format!("{api_base}/videos/{job_id}/content?index=0"))
                    });
                break;
            }

            if status == "failed" || status == "cancelled" || status == "expired" {
                let err_msg = poll_body
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                anyhow::bail!("Video generation failed: {err_msg}");
            }
        }

        let download_url = result_url;
        let Some(download_url) = download_url else {
            anyhow::bail!("Video generation did not complete within the 10-minute timeout period");
        };

        // ── Step 3: Download the video ──────────────────────────────────
        let response = match client
            .get(&download_url)
            .header("Authorization", &auth)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                anyhow::bail!("Failed to download generated video: {e}");
            }
        };

        let download_status = response.status();
        if !download_status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to download video (HTTP {download_status}): {error_text}");
        }

        let video_bytes = match response.bytes().await {
            Ok(b) => b.to_vec(),
            Err(e) => {
                anyhow::bail!("Failed to read downloaded video: {e}");
            }
        };

        // ── Step 4: Save to workspace/generated/ ────────────────────────
        let output_path = super::save_generated_file(ws, &video_bytes, "video", "mp4").await?;

        Ok(output_path.to_string_lossy().to_string())
    }
}
