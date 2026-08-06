use crate::Tool;
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;

/// Maximum length of the edit instruction in characters.
const MAX_INSTRUCTION_CHARS: usize = 1000;

/// Input clip size cap for every video_edit model. Guards the local-file
/// upload path against unbounded reads (hailuo-3's own model bound is 50 MB).
const MAX_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Per-model video edit capability classification, driving validation and the
/// tool description. The active model is user-switchable via
/// `video_edit_model` in the config.
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
fn validate_params(
    spec: VideoEditModel,
    duration: Option<i64>,
    seed: Option<i64>,
) -> anyhow::Result<()> {
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
            if seed.is_some() {
                anyhow::bail!("hailuo-3 does not support a seed parameter. Remove it and retry.");
            }
        }
        VideoEditModel::Aleph2 => {
            if duration.is_some() {
                anyhow::bail!(
                    "aleph-2 output mirrors the input clip duration — the duration \
                     parameter is not supported. Remove it and retry."
                );
            }
            // seed is accepted best-effort (passed through below).
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

/// Validate a canonicalized local clip path before upload: it must live under
/// the workspace `uploads/` directory (where received video attachments are
/// saved) and carry a recognized video extension. Arbitrary daemon-readable
/// files must never reach the anonymous upload host.
fn check_local_clip(
    canonical: &std::path::Path,
    uploads_root: &std::path::Path,
) -> anyhow::Result<()> {
    let canonical_root = std::fs::canonicalize(uploads_root).with_context(|| {
        format!(
            "Workspace uploads dir not found: {}",
            uploads_root.display()
        )
    })?;
    if !super::path::is_path_under_roots(canonical, &[canonical_root]) {
        anyhow::bail!(
            "Local clip must be inside the workspace uploads directory \
             (received video attachments are saved there), got: {}",
            canonical.display()
        );
    }
    if !crate::util::is_video_extension(canonical) {
        anyhow::bail!(
            "Local clip must have a recognized video extension (mp4, mov, mkv, avi, webm), got: {}",
            canonical.display()
        );
    }
    Ok(())
}

/// Tool for editing an existing video clip via OpenRouter's async videos API.
///
/// Accepts a public source clip URL or a local file path (uploaded to an
/// ephemeral anonymous host at job time), submits exactly one video edit job,
/// polls for completion, downloads the edited clip, and returns its path so
/// the agent can send it via `[VIDEO:path]` in its reply.
pub struct VideoEditTool;

#[async_trait]
impl Tool for VideoEditTool {
    fn name(&self) -> &'static str {
        "video_edit"
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[VIDEO:")
    }

    fn description(&self) -> String {
        // The description must match the active model (switchable via config).
        // The default `tool/video_edit.md` describes hailuo-3; aleph-2 differs
        // in every capability axis and lives in a non-tool/* prompt asset.
        match classify_model(&crate::config::CONFIG.video_edit_model()) {
            VideoEditModel::Aleph2 => crate::prompt::load_prompt("video_edit_aleph2.md"),
            VideoEditModel::Hailuo3 => crate::prompt::load_prompt("tool/video_edit.md"),
            VideoEditModel::Unknown => crate::prompt::load_prompt("video_edit_unknown.md"),
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "video_url": {
                    "type": "string",
                    "description": "Public HTTPS URL, or the path of a received video clip (shown as [Saved video: /path] in the chat) to edit"
                },
                "instruction": {
                    "type": "string",
                    "description": "Text instruction describing the edit to apply (max 1000 chars)"
                },
                "duration": {
                    "type": "integer",
                    "description": "Output duration in seconds (hailuo-3: 5–15; not supported by aleph-2)"
                },
                "seed": {
                    "type": "integer",
                    "description": "Seed for reproducibility (aleph-2 best-effort; not supported by hailuo-3)"
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
        let duration = super::get_opt_i64(&args, "duration");
        let seed = super::get_opt_i64(&args, "seed");

        let model = crate::config::CONFIG.video_edit_model();
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
        validate_params(spec, duration, seed)?;

        // ── Resolve the source: https URL as-is, local file → upload bridge ──
        let source_url = if video_url.starts_with("https://") || video_url.starts_with("http://") {
            video_url.to_string()
        } else {
            // Only clips saved into the workspace uploads dir by enrichment
            // may be uploaded — arbitrary local files are an exfiltration risk.
            let path = std::path::Path::new(video_url);
            let canonical = tokio::fs::canonicalize(path)
                .await
                .with_context(|| format!("Local clip not found: {video_url}"))?;
            check_local_clip(&canonical, &ws.as_path().join("uploads"))?;
            let len = tokio::fs::metadata(&canonical).await?.len();
            if len > MAX_INPUT_BYTES {
                anyhow::bail!(
                    "Source clip is limited to 50 MB, got {len} bytes. Trim the clip and retry."
                );
            }
            super::upload_bridge::upload_video_ephemeral(&canonical).await?
        };

        // ── Build and submit the edit job ────────────────────────────────
        let endpoint = crate::config::CONFIG.provider_endpoint();
        let api_base = crate::providers::ensure_base_url(&endpoint);

        let mut body = json!({
            "model": model,
            "prompt": instruction,
            "input_references": [{
                "type": "video_url",
                "video_url": { "url": source_url }
            }],
        });
        if let Some(d) = duration {
            body["duration"] = json!(d);
        }
        if let Some(s) = seed {
            body["seed"] = json!(s);
        }

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
    fn validate_params_enforces_per_model_rules() {
        // hailuo-3: duration 5–15, no seed
        assert!(validate_params(VideoEditModel::Hailuo3, Some(10), None).is_ok());
        assert!(validate_params(VideoEditModel::Hailuo3, None, None).is_ok());
        assert!(validate_params(VideoEditModel::Hailuo3, Some(4), None).is_err());
        assert!(validate_params(VideoEditModel::Hailuo3, Some(16), None).is_err());
        assert!(validate_params(VideoEditModel::Hailuo3, Some(10), Some(42)).is_err());
        // aleph-2: no duration, seed best-effort
        assert!(validate_params(VideoEditModel::Aleph2, None, None).is_ok());
        assert!(validate_params(VideoEditModel::Aleph2, None, Some(7)).is_ok());
        assert!(validate_params(VideoEditModel::Aleph2, Some(5), None).is_err());
        // unknown: permissive, positive duration only
        assert!(validate_params(VideoEditModel::Unknown, Some(30), Some(1)).is_ok());
        assert!(validate_params(VideoEditModel::Unknown, Some(0), None).is_err());
        assert!(validate_params(VideoEditModel::Unknown, Some(-3), None).is_err());
    }

    #[test]
    fn check_local_clip_requires_workspace_uploads_containment() {
        let tmp = tempfile::tempdir().unwrap();
        let uploads = tmp.path().join("uploads");
        std::fs::create_dir_all(&uploads).unwrap();
        let clip = uploads.join("clip.mp4");
        std::fs::write(&clip, b"clip").unwrap();
        // A clip inside uploads with a video extension passes.
        let canonical = std::fs::canonicalize(&clip).unwrap();
        assert!(check_local_clip(&canonical, &uploads).is_ok());
        // A clip outside uploads is rejected (e.g. an arbitrary readable file).
        let outside = tmp.path().join("config.toml");
        std::fs::write(&outside, b"secret").unwrap();
        let canonical_outside = std::fs::canonicalize(&outside).unwrap();
        assert!(check_local_clip(&canonical_outside, &uploads).is_err());
        // A non-video extension is rejected even inside uploads.
        let txt = uploads.join("notes.txt");
        std::fs::write(&txt, b"text").unwrap();
        let canonical_txt = std::fs::canonicalize(&txt).unwrap();
        assert!(check_local_clip(&canonical_txt, &uploads).is_err());
    }
}
