//! The `video_gen` tool: one job on OpenRouter's async videos API — submit,
//! poll, download — returning the path the reply can embed.
//!
//! Not video editing (`video_edit`) and not local encoding.

use super::media_catalog::video::VideoModelInfo;
use crate::Tool;
use async_trait::async_trait;
use serde_json::json;

/// The resolution asked for when the caller names none: the omni default's native tier.
const DEFAULT_RESOLUTION: &str = "720p";

/// Tool for generating videos via OpenRouter's async videos API: submit, poll, download, and
/// return the path so the agent can send it via `[VIDEO:path]` in its reply.
pub struct VideoGenTool;

/// Insert an optional value into the request body under `key`, skipping parameters the caller
/// did not supply.
fn set_opt<T: Into<serde_json::Value>>(body: &mut serde_json::Value, key: &str, value: Option<T>) {
    if let Some(v) = value {
        body[key] = v.into();
    }
}

/// The submit body: the model and prompt plus every optional parameter the caller named, with
/// `resolution` already resolved by [`requested_resolution`].
fn request_body(
    model: &str,
    args: &serde_json::Value,
    resolution: Option<String>,
) -> anyhow::Result<serde_json::Value> {
    let mut body = json!({
        "model": model,
        "prompt": super::get_str(args, "prompt")?,
    });
    set_opt(&mut body, "duration", super::get_opt_i64(args, "duration")?);
    set_opt(&mut body, "resolution", resolution);
    set_opt(
        &mut body,
        "aspect_ratio",
        super::get_opt_str(args, "aspect_ratio"),
    );
    set_opt(&mut body, "size", super::get_opt_str(args, "size"));
    set_opt(
        &mut body,
        "generate_audio",
        super::get_opt_bool(args, "generate_audio")?,
    );
    set_opt(&mut body, "seed", super::get_opt_i64(args, "seed")?);
    Ok(body)
}

/// The resolution the request must name: the caller's own verbatim, else [`DEFAULT_RESOLUTION`]
/// when the caller named neither a resolution nor an exact size — the two say the same thing —
/// and the model's declared capabilities offer it.
fn requested_resolution(args: &serde_json::Value, info: Option<&VideoModelInfo>) -> Option<String> {
    if let Some(named) = super::get_opt_str(args, "resolution") {
        return Some(named.to_string());
    }
    if super::get_opt_str(args, "size").is_some() {
        return None;
    }
    if !offers_default(info) {
        return None;
    }
    Some(DEFAULT_RESOLUTION.to_string())
}

/// Whether the model declares [`DEFAULT_RESOLUTION`], compared by value so another spelling
/// (`720P`) counts. No catalog entry, no declared resolutions, or a set without the tier all
/// answer `false`, leaving the request exactly as it is today.
fn offers_default(info: Option<&VideoModelInfo>) -> bool {
    info.and_then(|info| info.resolutions.as_ref())
        .is_some_and(|resolutions| {
            resolutions
                .iter()
                .any(|r| r.eq_ignore_ascii_case(DEFAULT_RESOLUTION))
        })
}

#[async_trait]
impl Tool for VideoGenTool {
    fn name(&self) -> &'static str {
        "video_gen"
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[VIDEO:")
    }

    fn preserve_full_output(&self) -> bool {
        // The result is the marker plus the clip's timeline — the frames as pictures of four with
        // their marks, the speech with marks and text — which can exceed the 5 KB budget: the
        // Assistant needs the whole timeline to reason about its own output.
        true
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
                    "description": "Path to a reference/start image for image-to-video generation (single image only; a local path is accepted only from the workspace uploads dir — received attachments — or the generated dir — previously generated images)",
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
        let (marker, path) = self.generate(ws, args).await?;
        Ok(
            super::video_output(marker, &path, super::TimelinePictures::TextOnly)
                .await
                .text,
        )
    }

    async fn execute_with_payloads(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<crate::tools::ToolOutput> {
        let (marker, path) = self.generate(ws, args).await?;
        Ok(super::video_output(marker, &path, super::TimelinePictures::Attached).await)
    }
}

impl VideoGenTool {
    /// The job itself: submit, poll, download, save — answering the `[VIDEO:path]` marker and the
    /// file it names, so the one run serves both the text-only and the payload-carrying answer.
    async fn generate(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<(String, std::path::PathBuf)> {
        let user_name = crate::agent::tool_user_name();
        let model = crate::users::resolve_video_model(&user_name).await;

        // Build the API base URL (strip /chat/completions if present). Video generation always
        // targets OpenRouter — a custom chat endpoint never serves video models.
        let api_base = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);

        let catalog = super::media_catalog::video::get_catalog().await;
        let info = catalog.as_deref().and_then(|catalog| catalog.find(&model));
        let mut body = request_body(&model, &args, requested_resolution(&args, info))?;

        // Reference-load failures are hard errors — never silently degrade to text-to-video.
        // `get_str_array` rejects malformed values instead of dropping them.
        let images: Vec<String> = super::get_str_array(&args, "images")?;

        if images.len() > 1 {
            anyhow::bail!(
                "Video generation supports only a single reference image (received {}). \
                 Retry with exactly one image.",
                images.len(),
            );
        }

        let mut references: Vec<crate::util::ReferenceImage> = Vec::new();
        if let Some(img_path) = images.first() {
            let path = super::resolve_local_media_path(img_path, ws, "reference image").await?;
            let reference =
                crate::util::load_reference_image(&path, super::MAX_REFERENCE_IMAGE_BYTES).await?;
            references.push(reference);
            body[super::INPUT_REFERENCES_KEY] = super::reference_json(&references);
        }

        // Aggregate body budget: one reference at the per-image cap already approaches the
        // ~2 MB client-side body budget once base64 + JSON overhead is added.
        super::fit_request_body_budget(&mut body, &mut references, super::MAX_REQUEST_BODY_BYTES)?;

        let video_bytes =
            super::fetch_async_video(&api_base, &body, super::VideoJobLabels::GENERATION).await?;

        // Save to the caller's output directory and format the media marker; the caller appends
        // the clip's own timeline.
        let output_path = super::save_generated_file(ws, &video_bytes, "video", "mp4").await?;
        let marker = self.format_media_result(&output_path);
        Ok((marker, output_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaring(resolutions: &[&str]) -> VideoModelInfo {
        VideoModelInfo {
            resolutions: Some(resolutions.iter().map(|r| (*r).to_string()).collect()),
            ..VideoModelInfo::default()
        }
    }

    #[test]
    fn default_resolution_needs_the_model_to_declare_it() {
        assert!(offers_default(Some(&declaring(&["480p", "720p"]))));
        assert!(
            offers_default(Some(&declaring(&["720P"]))),
            "the comparison is by value, not by spelling"
        );
        assert!(
            !offers_default(Some(&declaring(&["2K"]))),
            "a model without the tier keeps today's request"
        );
        assert!(!offers_default(Some(&VideoModelInfo::default())));
        assert!(!offers_default(None), "no catalog entry offers nothing");
    }

    #[test]
    fn a_named_resolution_or_size_leaves_the_request_alone() {
        let named = json!({ "resolution": "1080p" });
        assert_eq!(
            requested_resolution(&named, Some(&declaring(&["720p"]))).as_deref(),
            Some("1080p"),
            "the caller's own value wins verbatim"
        );
        let sized = json!({ "size": "1280x720" });
        assert_eq!(
            requested_resolution(&sized, Some(&declaring(&["720p"]))),
            None,
            "a size says the same thing as a resolution, so none is added"
        );
    }

    #[test]
    fn body_carries_the_resolution_it_is_given() {
        let args = json!({ "prompt": "a wave", "duration": 5 });
        let body = request_body("m", &args, Some(DEFAULT_RESOLUTION.to_string())).unwrap();
        assert_eq!(body["resolution"], DEFAULT_RESOLUTION);
        assert_eq!(body["duration"], 5);
        let body = request_body("m", &args, None).unwrap();
        assert!(body.get("resolution").is_none());
    }
}
