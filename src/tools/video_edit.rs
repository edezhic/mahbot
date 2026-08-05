use crate::Tool;
use async_trait::async_trait;
use serde_json::json;

/// Number of polling attempts (10 min timeout = 20 attempts × 30s).
const MAX_POLL_ATTEMPTS: u32 = 20;

/// Maximum length of the edit instruction in characters.
const MAX_INSTRUCTION_CHARS: usize = 1000;

/// Tool for editing an existing video clip via OpenRouter's async videos API.
///
/// Submits exactly one video edit job (public source clip URL + text
/// instruction), polls for completion, downloads the edited clip, and returns
/// its path so the agent can send it via `[VIDEO:path]` in its reply.
pub struct VideoEditTool;

#[async_trait]
#[allow(clippy::too_many_lines)]
impl Tool for VideoEditTool {
    fn name(&self) -> &'static str {
        "video_edit"
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[VIDEO:")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "video_url": {
                    "type": "string",
                    "description": "Public HTTPS URL of the source video clip (2–30 seconds) to edit"
                },
                "instruction": {
                    "type": "string",
                    "description": "Text instruction describing the edit to apply (max 1000 chars)"
                }
            }),
            &["video_url", "instruction"],
        )
    }

    async fn execute(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        let video_url = super::get_str(&args, "video_url")?;
        let instruction = super::get_str(&args, "instruction")?;

        if !(video_url.starts_with("https://") || video_url.starts_with("http://")) {
            anyhow::bail!("video_url must be a public absolute HTTP(S) URL, got: {video_url}");
        }
        let char_count = instruction.chars().count();
        if char_count > MAX_INSTRUCTION_CHARS {
            anyhow::bail!(
                "Instruction is too long: {char_count} chars (max {MAX_INSTRUCTION_CHARS}). \
                 Retry with a shorter instruction."
            );
        }

        let model = crate::config::CONFIG.video_edit_model();

        // Build the API base URL (strip /chat/completions if present)
        let endpoint = crate::config::CONFIG.provider_endpoint();
        let api_base = crate::providers::ensure_base_url(&endpoint);

        // ── Step 1: Submit video edit job (exactly one POST — no retry) ──
        // Each submission is a billable job; the endpoint has no idempotency key.
        let body = json!({
            "model": model,
            "prompt": instruction,
            "input_references": [{
                "type": "video_url",
                "video_url": { "url": video_url }
            }],
        });

        let submit_url = format!("{api_base}/videos");
        let submit_body: serde_json::Value = match crate::util::http::post_json_to_provider(
            &submit_url,
            &body,
            "Video edit submission",
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                if e.downcast_ref::<crate::util::error::HttpError>()
                    .map(|e| e.status)
                    == Some(402)
                {
                    anyhow::bail!(
                        "Insufficient OpenRouter credits for video editing (HTTP 402). \
                             Please add credits to your OpenRouter account and try again."
                    );
                }
                return Err(e);
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

        tracing::info!(%job_id, "Video edit job submitted");

        // ── Step 2: Poll for completion (~10 min timeout) ───────────────
        let mut result_url: Option<String> = None;

        for attempt in 1..=MAX_POLL_ATTEMPTS {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            let poll_body =
                match crate::util::http::get_json_from_provider(&polling_url, "Video edit poll")
                    .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::debug!(%job_id, attempt, error = %e, "Poll failed");
                        continue;
                    }
                };

            let status = poll_body
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            tracing::info!(%job_id, %status, attempt, "Video edit poll");

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
                anyhow::bail!("Video edit failed: {err_msg}");
            }
        }

        let Some(download_url) = result_url else {
            anyhow::bail!("Video edit did not complete within the 10-minute timeout period");
        };

        // ── Step 3: Download the edited video ───────────────────────────
        // The result URL requires the bearer key despite the "unsigned" name.
        let video_bytes =
            crate::util::http::get_bytes_from_provider(&download_url, "Video edit download")
                .await?;

        // Validate the payload is a real MP4 (no Content-Length on the response;
        // an error page would slip through otherwise).
        if video_bytes.len() <= 100_000 || &video_bytes[4..8] != b"ftyp" {
            anyhow::bail!(
                "Video edit download returned an invalid file ({} bytes, no ftyp header)",
                video_bytes.len(),
            );
        }

        // ── Step 4: Save to workspace/generated/ ────────────────────────
        let output_path = super::save_generated_file(ws, &video_bytes, "video", "mp4").await?;

        let path_str = output_path.to_string_lossy();
        let marker_prefix = self
            .media_marker()
            .expect("VideoEditTool always has a media marker");
        Ok(format!("{marker_prefix}{path_str}]"))
    }
}
