use crate::Tool;
use async_trait::async_trait;
use serde_json::json;

/// Maximum length of the edit instruction in characters.
const MAX_INSTRUCTION_CHARS: usize = 1000;

/// Tool for editing an existing video clip via OpenRouter's async videos API.
///
/// Submits exactly one video edit job (public source clip URL + text
/// instruction), polls for completion, downloads the edited clip, and returns
/// its path so the agent can send it via `[VIDEO:path]` in its reply.
pub struct VideoEditTool;

#[async_trait]
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

        let body = json!({
            "model": model,
            "prompt": instruction,
            "input_references": [{
                "type": "video_url",
                "video_url": { "url": video_url }
            }],
        });

        let video_bytes =
            super::fetch_async_video(&api_base, &body, super::VideoJobLabels::EDIT).await?;

        // Save to workspace/generated/ and format the media marker.
        let output_path = super::save_generated_file(ws, &video_bytes, "video", "mp4").await?;

        let path_str = output_path.to_string_lossy();
        let marker_prefix = self
            .media_marker()
            .expect("VideoEditTool always has a media marker");
        Ok(format!("{marker_prefix}{path_str}]"))
    }
}
