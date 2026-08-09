use crate::Tool;
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;

/// Maximum length of the edit instruction in characters.
const MAX_INSTRUCTION_CHARS: usize = 1000;

/// Input clip size cap for every video_edit model. Guards the local-file
/// upload path against unbounded reads (hailuo-3's own model bound is 50 MB).
const MAX_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Input image size cap (provider-declared 30 MB bound for hailuo-3).
const MAX_IMAGE_BYTES: u64 = 30 * 1024 * 1024;

/// Maximum reference images per request (native hailuo-3 limit of 9).
const MAX_REFERENCE_IMAGES: usize = 9;

/// Per-model video edit capability classification, driving validation and the
/// tool description. The active model is user-switchable via
/// `video_model` in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoEditModel {
    /// minimax/hailuo-3 (default): 5–15 s output, 2K fixed, always audio, no seed.
    Hailuo3,
    /// runway/aleph-2: output mirrors input, preserves audio, seed best-effort.
    Aleph2,
    /// Any other configured model — permissive validation.
    Unknown,
}

fn classify_model(model: &str) -> VideoEditModel {
    let m = model.to_ascii_lowercase();
    // Substring match (not exact) so vendor-prefixed IDs like
    // "minimax/hailuo-3" keep resolving; Unknown covers every other model
    // permissively.
    if m.contains("hailuo") {
        VideoEditModel::Hailuo3
    } else if m.contains("aleph") {
        VideoEditModel::Aleph2
    } else {
        VideoEditModel::Unknown
    }
}

/// Per-model parameter validation, run before any upload or billing.
fn validate_params(spec: VideoEditModel, duration: Option<i64>) -> anyhow::Result<()> {
    match spec {
        VideoEditModel::Hailuo3 => {
            if let Some(d) = duration
                && !(5..=15).contains(&d)
            {
                anyhow::bail!(
                    "hailuo-3 output duration must be 5–15 seconds, got {d}. \
                     Retry with a duration in that range."
                );
            }
        }
        VideoEditModel::Aleph2 => {
            if duration.is_some() {
                anyhow::bail!(
                    "aleph-2 output mirrors the input clip duration — the duration \
                     parameter is not supported. Remove it and retry."
                );
            }
        }
        VideoEditModel::Unknown => {
            if let Some(d) = duration
                && d <= 0
            {
                anyhow::bail!("duration must be a positive integer, got {d}");
            }
        }
    }
    Ok(())
}

/// Input mode of a video_edit request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditMode {
    /// Video clip edit, optionally guided by reference images.
    VideoRef,
    /// Image-to-video from first/last frame anchors (no video or reference
    /// images — frame and reference roles are mutually exclusive).
    FrameAnchor,
}

/// Validate the input-mode combination and the per-model image gate. Frame
/// anchors (first/last frame) take precedence over references on the
/// provider, which silently drops the losing inputs while still billing —
/// so any mixed-mode request is rejected client-side. aleph-2 declares no
/// image support; unknown models stay permissive.
fn validate_mode(
    spec: VideoEditModel,
    video_url: Option<&str>,
    images: &[String],
    first_frame: Option<&str>,
    last_frame: Option<&str>,
) -> anyhow::Result<EditMode> {
    let has_anchors = first_frame.is_some() || last_frame.is_some();
    if spec == VideoEditModel::Aleph2 && (!images.is_empty() || has_anchors) {
        anyhow::bail!(
            "aleph-2 does not support image inputs (reference images or frame \
             anchors). Retry without images or switch to hailuo-3."
        );
    }
    if has_anchors {
        if video_url.is_some() {
            anyhow::bail!(
                "Frame anchors are mutually exclusive with a video reference — a \
                 request cannot mix frame and reference roles. Remove video_url and retry."
            );
        }
        if !images.is_empty() {
            anyhow::bail!(
                "Frame anchors are mutually exclusive with reference images — a \
                 request cannot mix frame and reference roles. Remove images and retry."
            );
        }
        Ok(EditMode::FrameAnchor)
    } else {
        if video_url.is_none() {
            anyhow::bail!(
                "Missing required field: video_url. Provide a video to edit \
                 (optionally with reference images), or use first_frame/last_frame \
                 for image-to-video."
            );
        }
        if images.len() > MAX_REFERENCE_IMAGES {
            anyhow::bail!(
                "At most {MAX_REFERENCE_IMAGES} reference images are supported, got {}. \
                 Reduce the number of reference images and retry.",
                images.len()
            );
        }
        Ok(EditMode::VideoRef)
    }
}

/// Validate that a canonicalized local path lives under the workspace
/// `uploads/` directory (received media attachments) or the `generated/`
/// directory (outputs of the generation tools). Arbitrary daemon-readable
/// files must never reach the anonymous upload host.
fn check_local_containment(
    canonical: &std::path::Path,
    ws: &crate::Workspace,
    kind: &str,
) -> anyhow::Result<()> {
    // Roots are canonicalized when they exist; a missing dir simply cannot
    // contain a canonicalized (existing) file, so it is skipped.
    let roots: Vec<std::path::PathBuf> = ["uploads", "generated"]
        .iter()
        .map(|d| ws.as_path().join(d))
        .filter_map(|r| std::fs::canonicalize(&r).ok())
        .collect();
    if !super::path::is_path_under_roots(canonical, &roots) {
        tracing::warn!(
            path = %canonical.display(),
            "Local {kind} rejected: not inside workspace uploads or generated dirs"
        );
        anyhow::bail!(
            "Local {kind} must be inside the workspace uploads directory \
             (received media attachments) or the generated directory \
             (previously generated media), got: {}",
            canonical.display()
        );
    }
    Ok(())
}

/// Validate a canonicalized local clip path before upload: it must live under
/// the workspace `uploads/` directory (received video attachments) or the
/// `generated/` directory (previously generated clips) and carry a recognized
/// video extension.
fn check_local_clip(canonical: &std::path::Path, ws: &crate::Workspace) -> anyhow::Result<()> {
    check_local_containment(canonical, ws, "clip")?;
    if !crate::util::is_video_extension(canonical) {
        anyhow::bail!(
            "Local clip must have a recognized video extension (mp4, mov, mkv, avi, webm), got: {}",
            canonical.display()
        );
    }
    Ok(())
}

/// Validate a canonicalized local image path before upload: it must live under
/// the workspace `uploads/` directory (received images) or the `generated/`
/// directory (previously generated images) and carry a recognized image
/// extension (jpg, jpeg, png, webp, heic, heif).
fn check_local_image(canonical: &std::path::Path, ws: &crate::Workspace) -> anyhow::Result<()> {
    check_local_containment(canonical, ws, "image")?;
    if !crate::util::is_image_extension(canonical) {
        anyhow::bail!(
            "Local image must have a recognized image extension \
             (jpg, jpeg, png, webp, heic, heif), got: {}",
            canonical.display()
        );
    }
    Ok(())
}

/// Resolve the video reference: public URL as-is, local file → upload bridge.
async fn resolve_video_source(video_url: &str, ws: &crate::Workspace) -> anyhow::Result<String> {
    if video_url.starts_with("https://") || video_url.starts_with("http://") {
        return Ok(video_url.to_string());
    }
    // Only clips saved into the workspace uploads dir (received attachments)
    // or the generated dir (previous generation outputs) may be uploaded —
    // arbitrary local files are an exfiltration risk.
    let path = std::path::Path::new(video_url);
    let canonical = tokio::fs::canonicalize(path)
        .await
        .with_context(|| format!("Local clip not found: {video_url}"))?;
    check_local_clip(&canonical, ws)?;
    let len = tokio::fs::metadata(&canonical).await?.len();
    if len > MAX_INPUT_BYTES {
        anyhow::bail!("Source clip is limited to 50 MB, got {len} bytes. Trim the clip and retry.");
    }
    crate::util::upload_bridge::upload_video_ephemeral(&canonical).await
}

/// Resolve an image input: public URL (GET-validated — a broken image
/// reference bills anyway on the provider) or a local file in workspace
/// uploads or generated (uploaded to an ephemeral host). `label` names the
/// input in error messages ("reference image", "first-frame anchor", ...).
async fn resolve_image_input(
    input: &str,
    ws: &crate::Workspace,
    label: &str,
) -> anyhow::Result<String> {
    if input.starts_with("https://") || input.starts_with("http://") {
        crate::util::upload_bridge::verify_media_url(input, "image/")
            .await
            .with_context(|| format!("Failed to validate {label} URL {input}"))?;
        return Ok(input.to_string());
    }
    let path = std::path::Path::new(input);
    let canonical = tokio::fs::canonicalize(path)
        .await
        .with_context(|| format!("Local {label} not found: {input}"))?;
    check_local_image(&canonical, ws)?;
    let len = tokio::fs::metadata(&canonical).await?.len();
    if len > MAX_IMAGE_BYTES {
        anyhow::bail!(
            "{label} is limited to 30 MB, got {len} bytes. Use a smaller image and retry."
        );
    }
    crate::util::upload_bridge::upload_image_ephemeral(&canonical).await
}

/// Tool for editing an existing video clip via OpenRouter's async videos API.
///
/// Accepts a public source clip URL or a local file path (from the workspace
/// `uploads/` received-attachments dir or the `generated/` output dir,
/// uploaded to an ephemeral anonymous host at job time), submits exactly one
/// video edit job, polls for completion, downloads the edited clip, and
/// returns its path so the agent can send it via `[VIDEO:path]` in its reply.
pub struct VideoEditTool;

#[async_trait]
impl Tool for VideoEditTool {
    fn name(&self) -> &'static str {
        "video_edit"
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

    fn description(&self) -> String {
        crate::prompt::load_prompt("tool/video_edit.md")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "video_url": {
                    "type": "string",
                    "description": "Public HTTPS URL, or the path of a local video clip to edit — a received attachment in the workspace uploads dir (shown as [Saved video: /path] in the chat) or a previously generated video in the workspace generated dir (shown as [VIDEO:path]). Required unless first_frame/last_frame are used for image-to-video"
                },
                "instruction": {
                    "type": "string",
                    "description": "Text instruction describing the edit to apply (max 1000 chars)"
                },
                "duration": {
                    "type": "integer",
                    "description": "Output duration in seconds (model-dependent)"
                },
                "images": {
                    "type": "array",
                    "items": { "type": "string" },
                    "maxItems": 9,
                    "description": "Paths or public HTTPS URLs of reference images guiding style/subject (max 9). Local paths are accepted from the workspace uploads dir (received attachments) or the generated dir (previously generated images). Requires video_url; mutually exclusive with first_frame/last_frame"
                },
                "first_frame": {
                    "type": "string",
                    "description": "Path or public HTTPS URL of an image to use as the exact first frame (image-to-video). Local paths are accepted from the workspace uploads dir (received attachments) or the generated dir (previously generated images). Mutually exclusive with video_url and images"
                },
                "last_frame": {
                    "type": "string",
                    "description": "Path or public HTTPS URL of an image to use as the exact last frame (image-to-video). Local paths are accepted from the workspace uploads dir (received attachments) or the generated dir (previously generated images). Mutually exclusive with video_url and images"
                }
            }),
            &["instruction"],
        )
    }

    #[allow(clippy::too_many_lines)]
    async fn execute(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        // Empty strings are treated as absent optionals (a blank field carries
        // no intent and must not trigger mode/exclusivity validation).
        let video_url = super::get_opt_str(&args, "video_url").filter(|s| !s.is_empty());
        let instruction = super::get_str(&args, "instruction")?;
        let duration = super::get_opt_i64(&args, "duration");
        // Reject a malformed `images` value (bare string or non-string
        // elements) instead of silently omitting the reference — silent
        // input drops are exactly what the exclusivity rules exist to
        // prevent. Null is treated as absent, like the other optionals.
        if let Some(v) = args.get("images")
            && !v.is_null()
            && !v
                .as_array()
                .is_some_and(|a| a.iter().all(serde_json::Value::is_string))
        {
            anyhow::bail!(
                "images must be an array of image paths or URLs, got: {}",
                crate::util::truncate(&v.to_string(), 200)
            );
        }
        let images: Vec<String> = super::get_str_array(&args, "images")
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let first_frame = super::get_opt_str(&args, "first_frame").filter(|s| !s.is_empty());
        let last_frame = super::get_opt_str(&args, "last_frame").filter(|s| !s.is_empty());

        let model = crate::config::CONFIG.video_model();
        let spec = classify_model(&model);

        let char_count = instruction.chars().count();
        if char_count == 0 {
            anyhow::bail!("Instruction must not be empty. Describe the edit to apply.");
        }
        if char_count > MAX_INSTRUCTION_CHARS {
            anyhow::bail!(
                "Instruction is too long: {char_count} chars (max {MAX_INSTRUCTION_CHARS}). \
                 Retry with a shorter instruction."
            );
        }

        // ── Per-model parameter validation (fail fast, before any upload) ──
        validate_params(spec, duration)?;

        // ── Mode validation: frame anchors vs references, model image gate ──
        let mode = validate_mode(spec, video_url, &images, first_frame, last_frame)?;

        // ── Resolve sources and build the request body ────────────────────
        let mut body = json!({
            "model": model,
            "prompt": instruction,
        });

        match mode {
            EditMode::FrameAnchor => {
                let mut frame_images: Vec<serde_json::Value> = Vec::new();
                if let Some(first) = first_frame {
                    let url = resolve_image_input(first, ws, "first-frame anchor").await?;
                    frame_images.push(json!({
                        "type": "image_url",
                        "image_url": { "url": url },
                        "frame_type": "first_frame"
                    }));
                }
                if let Some(last) = last_frame {
                    let url = resolve_image_input(last, ws, "last-frame anchor").await?;
                    frame_images.push(json!({
                        "type": "image_url",
                        "image_url": { "url": url },
                        "frame_type": "last_frame"
                    }));
                }
                body["frame_images"] = json!(frame_images);
            }
            EditMode::VideoRef => {
                // validate_mode guarantees a video reference in this mode; the
                // defensive error keeps this panic-free if that ever changes.
                let video_url = video_url
                    .ok_or_else(|| anyhow::anyhow!("video_url is required for a video edit"))?;
                let source_url = resolve_video_source(video_url, ws).await?;
                let mut references = vec![json!({
                    "type": "video_url",
                    "video_url": { "url": source_url }
                })];
                for img in &images {
                    let url = resolve_image_input(img, ws, "reference image").await?;
                    references.push(json!({
                        "type": "image_url",
                        "image_url": { "url": url }
                    }));
                }
                body["input_references"] = json!(references);
            }
        }

        if let Some(d) = duration {
            body["duration"] = json!(d);
        }

        let endpoint = crate::config::CONFIG.provider_endpoint();
        let api_base = crate::providers::ensure_base_url(&endpoint);

        let video_bytes =
            super::fetch_async_video(&api_base, &body, super::VideoJobLabels::EDIT).await?;

        // Save to workspace/generated/ and format the media marker. The marker
        // stays first; the transcription is appended for the Artist to reason
        // about its own output (fail-open: marker-only on transcription failure).
        let output_path = super::save_generated_file(ws, &video_bytes, "video", "mp4").await?;
        let marker = self.format_media_result(&output_path);
        Ok(super::format_video_result(marker, &output_path).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_model_matches_known_and_unknown() {
        assert_eq!(classify_model("minimax/hailuo-3"), VideoEditModel::Hailuo3);
        assert_eq!(classify_model("MINIMAX/HAILUO-3"), VideoEditModel::Hailuo3);
        assert_eq!(classify_model("runway/aleph-2"), VideoEditModel::Aleph2);
        assert_eq!(
            classify_model("runway/aleph-2-20260729"),
            VideoEditModel::Aleph2
        );
        assert_eq!(classify_model("some/other-model"), VideoEditModel::Unknown);
    }

    #[test]
    fn validate_mode_enforces_exclusivity_and_model_gate() {
        let none: Vec<String> = Vec::new();
        // Video-only flow (unchanged) and video + reference images.
        assert_eq!(
            validate_mode(VideoEditModel::Hailuo3, Some("clip.mp4"), &none, None, None).unwrap(),
            EditMode::VideoRef
        );
        assert_eq!(
            validate_mode(
                VideoEditModel::Hailuo3,
                Some("clip.mp4"),
                &["ref.png".to_string()],
                None,
                None
            )
            .unwrap(),
            EditMode::VideoRef
        );
        // Pure frame-anchor image-to-video.
        assert_eq!(
            validate_mode(VideoEditModel::Hailuo3, None, &none, Some("f.png"), None).unwrap(),
            EditMode::FrameAnchor
        );
        assert_eq!(
            validate_mode(VideoEditModel::Hailuo3, None, &none, None, Some("l.png")).unwrap(),
            EditMode::FrameAnchor
        );
        // Mixed modes are rejected (provider silently drops one while billing).
        assert!(
            validate_mode(
                VideoEditModel::Hailuo3,
                Some("clip.mp4"),
                &none,
                Some("f.png"),
                None
            )
            .is_err()
        );
        assert!(
            validate_mode(
                VideoEditModel::Hailuo3,
                None,
                &["ref.png".to_string()],
                Some("f.png"),
                None
            )
            .is_err()
        );
        // Reference-image cap.
        let ten: Vec<String> = (0..10).map(|i| format!("r{i}.png")).collect();
        assert!(
            validate_mode(VideoEditModel::Hailuo3, Some("clip.mp4"), &ten, None, None).is_err()
        );
        // aleph-2 rejects image inputs; unknown models stay permissive.
        assert!(
            validate_mode(
                VideoEditModel::Aleph2,
                Some("clip.mp4"),
                &["ref.png".to_string()],
                None,
                None
            )
            .is_err()
        );
        assert!(validate_mode(VideoEditModel::Aleph2, None, &none, Some("f.png"), None).is_err());
        assert_eq!(
            validate_mode(VideoEditModel::Aleph2, Some("clip.mp4"), &none, None, None).unwrap(),
            EditMode::VideoRef
        );
        assert_eq!(
            validate_mode(
                VideoEditModel::Unknown,
                Some("clip.mp4"),
                &["ref.png".to_string()],
                None,
                None
            )
            .unwrap(),
            EditMode::VideoRef
        );
        // No mode selected at all.
        assert!(validate_mode(VideoEditModel::Hailuo3, None, &none, None, None).is_err());
    }

    #[test]
    fn check_local_media_requires_workspace_uploads_or_generated_containment() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = crate::Workspace::from_path(tmp.path());
        // A clip inside generated (a previous generation output) passes even
        // when the uploads dir is absent — the live video-edit failure mode.
        let generated = tmp.path().join("generated");
        std::fs::create_dir_all(&generated).unwrap();
        let gen_clip = generated.join("video_1786201173303.mp4");
        std::fs::write(&gen_clip, b"clip").unwrap();
        let canonical_gen = std::fs::canonicalize(&gen_clip).unwrap();
        assert!(check_local_clip(&canonical_gen, &ws).is_ok());
        // A clip inside uploads with a video extension passes.
        let uploads = tmp.path().join("uploads");
        std::fs::create_dir_all(&uploads).unwrap();
        let clip = uploads.join("clip.mp4");
        std::fs::write(&clip, b"clip").unwrap();
        let canonical = std::fs::canonicalize(&clip).unwrap();
        assert!(check_local_clip(&canonical, &ws).is_ok());
        // A file outside uploads/generated is rejected (arbitrary readable file).
        let outside = tmp.path().join("config.toml");
        std::fs::write(&outside, b"secret").unwrap();
        let canonical_outside = std::fs::canonicalize(&outside).unwrap();
        assert!(check_local_clip(&canonical_outside, &ws).is_err());
        // A non-video extension is rejected even inside uploads.
        let txt = uploads.join("notes.txt");
        std::fs::write(&txt, b"text").unwrap();
        let canonical_txt = std::fs::canonicalize(&txt).unwrap();
        assert!(check_local_clip(&canonical_txt, &ws).is_err());
        // Images: accepted extensions pass, non-image extensions are rejected.
        let img = uploads.join("photo.heic");
        std::fs::write(&img, b"image").unwrap();
        let canonical_img = std::fs::canonicalize(&img).unwrap();
        assert!(check_local_image(&canonical_img, &ws).is_ok());
        let gen_img = generated.join("image_1786201173303.png");
        std::fs::write(&gen_img, b"image").unwrap();
        let canonical_gen_img = std::fs::canonicalize(&gen_img).unwrap();
        assert!(check_local_image(&canonical_gen_img, &ws).is_ok());
        assert!(check_local_image(&canonical_outside, &ws).is_err());
        assert!(check_local_image(&canonical_txt, &ws).is_err());
    }
}
