use crate::Tool;
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use std::io::{Read, Seek, SeekFrom};

/// Maximum length of the edit instruction in characters (tool-level ceiling;
/// per-model native limits lower than this surface via the model nuances).
const MAX_INSTRUCTION_CHARS: usize = 5000;

/// Input clip size cap for every video_edit model. Guards the local-file
/// upload path against unbounded reads (hailuo-3's own model bound is 50 MB).
const MAX_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Input image size cap (provider-declared 30 MB bound for hailuo-3).
const MAX_IMAGE_BYTES: u64 = 30 * 1024 * 1024;

/// Maximum reference images per request: hailuo-3 natively allows 9;
/// seedance-2.0-mini documents 9 (BytePlus/mirrors; Cloudflare's relay
/// schema says 4 — not adopted). Applied uniformly.
const MAX_REFERENCE_IMAGES: usize = 9;

/// Per-model video edit capability classification, driving validation and the
/// tool description. The active model is user-switchable via
/// `video_model` in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoEditModel {
    /// minimax/hailuo-3 (default): 5-15s output, 2K fixed, always audio, no seed.
    Hailuo3,
    /// runway/aleph-2: output mirrors input, preserves audio, seed best-effort.
    Aleph2,
    /// bytedance/seedance-2.5: whole-frame restyle, 480p/720p via OpenRouter;
    /// edit tasks require duration 'auto' (integer durations rejected).
    Seedance,
    /// bytedance/seedance-2.0-mini: whole-frame restyle, 480p/720p via OpenRouter;
    /// edit tasks accept explicit 4-15s durations (unlike 2.5's auto-only).
    SeedanceMini,
    /// Any other configured model — permissive validation.
    Unknown,
}

fn classify_model(model: &str) -> VideoEditModel {
    let m = model.to_ascii_lowercase();
    // Substring match (not exact) so vendor-prefixed IDs like
    // "minimax/hailuo-3" keep resolving; Unknown covers every other model
    // permissively. seedance-2.5 keeps its auto-duration edit rule;
    // seedance-2.0-mini gets explicit-duration edits; seedance-2.0/2.0-fast/
    // 1.5-pro stay Unknown.
    if m.contains("hailuo") {
        VideoEditModel::Hailuo3
    } else if m.contains("aleph") {
        VideoEditModel::Aleph2
    } else if m.contains("seedance") && m.contains("2.0-mini") {
        VideoEditModel::SeedanceMini
    } else if m.contains("seedance") && m.contains("2.5") {
        VideoEditModel::Seedance
    } else {
        VideoEditModel::Unknown
    }
}

impl VideoEditModel {
    /// Verified input-clip duration bounds in milliseconds — the single
    /// per-model table backing the pre-flight local-clip check.
    ///
    /// `None` fails open (no check): the limit is unverified or unknown, or
    /// the model trims over-long inputs instead of rejecting them (seedance
    /// family). Deleting a row here removes the check everywhere — no other
    /// copy of a limit exists.
    fn input_duration_limit_ms(self) -> Option<std::ops::RangeInclusive<i64>> {
        match self {
            // minimax/hailuo-3: 2000–15000 ms per clip — confirmed by live
            // provider errors ("expected [2000, 15000] ms") and MiniMax docs;
            // the same bound behind the 50 MB input cap.
            VideoEditModel::Hailuo3 => Some(2_000..=15_000),
            // aleph-2's 2–30 s range is NOT verified by a live provider error.
            // seedance-2.5/2.0-mini TRIM over-long inputs — never hard-reject.
            VideoEditModel::Aleph2
            | VideoEditModel::Seedance
            | VideoEditModel::SeedanceMini
            | VideoEditModel::Unknown => None,
        }
    }
}

/// Static per-model video-editing nuances for the `<active-models-opts>`
/// block (surfaced on session start and on model switch). Informational
/// facts only — never parameter-style: `video_edit` accepts no resolution or
/// editing-mode argument, and the block is shared with `video_gen`. Keyed via
/// [`classify_model`] so rendering can never drift from validation.
pub(crate) fn video_edit_nuances(model: &str) -> Option<&'static str> {
    match classify_model(model) {
        VideoEditModel::Hailuo3 => Some(
            "localized instruction edits + video-to-video motion transfer; 2K output; \
             source style preserved by default",
        ),
        VideoEditModel::Seedance => Some(
            "whole-frame restyle; output 480p/720p via OpenRouter; edit-mode duration \
             is automatic ('auto') — the 4-30s duration range applies to generation, \
             not edits",
        ),
        VideoEditModel::SeedanceMini => Some(
            "whole-frame restyle; output 480p/720p via OpenRouter; edit-mode duration \
             is explicit (4-15s, same range as generation)",
        ),
        VideoEditModel::Aleph2 | VideoEditModel::Unknown => None,
    }
}

/// Whether the request body must omit `duration`: seedance-2.5 edit-classified
/// (video-reference) tasks natively require duration 'auto', and forwarding an
/// explicit integer is provider-rejected. Omission is schema-valid and lets
/// the provider default to 'auto'. Image-to-video (frame-anchor) keeps the
/// catalog duration range. seedance-2.0-mini edit tasks accept explicit
/// 4-15s durations, so it never omits.
fn omit_duration(spec: VideoEditModel, mode: EditMode) -> bool {
    matches!((spec, mode), (VideoEditModel::Seedance, EditMode::VideoRef))
}

/// Validate an optional integer duration against a per-model range.
fn ensure_duration_in(
    duration: Option<i64>,
    range: std::ops::RangeInclusive<i64>,
    label: &str,
) -> anyhow::Result<()> {
    if let Some(d) = duration
        && !range.contains(&d)
    {
        anyhow::bail!(
            "{label} must be {}–{} seconds, got {d}. Retry with a duration in that range.",
            range.start(),
            range.end()
        );
    }
    Ok(())
}

/// Per-model parameter validation, run before any upload or billing. The
/// seedance-2.5 duration rule is mode-dependent, so this runs after
/// [`validate_mode`].
fn validate_params(
    spec: VideoEditModel,
    mode: EditMode,
    duration: Option<i64>,
) -> anyhow::Result<()> {
    match spec {
        VideoEditModel::Hailuo3 => {
            ensure_duration_in(duration, 5..=15, "hailuo-3 output duration")?;
        }
        VideoEditModel::Aleph2 => {
            if duration.is_some() {
                anyhow::bail!(
                    "aleph-2 output mirrors the input clip duration — the duration \
                     parameter is not supported. Remove it and retry."
                );
            }
        }
        VideoEditModel::SeedanceMini => {
            // Explicit 4-15s edit duration per native docs/catalog; the relay's
            // seedance-edit duration handling is unverified — rejection surfaces at submit.
            ensure_duration_in(duration, 4..=15, "seedance-2.0-mini duration")?;
        }
        VideoEditModel::Seedance => match mode {
            // Native edit tasks require duration 'auto' — the field is omitted
            // at body build (see [`omit_duration`]), so any value is accepted.
            EditMode::VideoRef => {}
            EditMode::FrameAnchor => {
                ensure_duration_in(duration, 4..=30, "seedance-2.5 image-to-video duration")?;
            }
        },
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

/// Pre-flight input-duration gate for a local clip, run after containment,
/// extension, and size validation and before any upload or job submission.
///
/// Hard-rejects only when the active model has a verified input-duration
/// limit AND the parsed MP4/MOV container duration falls outside it. Fails
/// open for everything uncertain: non-MP4/MOV containers (webm/mkv/avi),
/// fragmented or corrupt files, unreadable metadata, and models without a
/// verified limit.
fn check_local_input_duration(
    canonical: &std::path::Path,
    model: &str,
    spec: VideoEditModel,
) -> anyhow::Result<()> {
    let Some(limit_ms) = spec.input_duration_limit_ms() else {
        return Ok(());
    };
    // Only ISO-BMFF containers carry a readable mvhd duration — the other
    // accepted local extensions (webm/mkv/avi) are skipped by design.
    if !crate::util::has_extension(canonical, &["mp4", "mov"]) {
        return Ok(());
    }
    let Some(duration_ms) = mp4_duration_ms(canonical) else {
        // Fragmented, corrupt, or unreadable — cannot verify the provider
        // would reject it, so the submission proceeds (fail-open).
        tracing::debug!(
            path = %canonical.display(),
            "video_edit pre-flight duration unreadable — failing open"
        );
        return Ok(());
    };
    if duration_ms > *limit_ms.end() {
        // Rounded tenths of a second for readability; the exact ms the
        // provider validates against is stated too.
        let tenths = (duration_ms + 50) / 100;
        anyhow::bail!(
            "Source clip duration is {}.{} s ({duration_ms} ms), which exceeds {model}'s \
             verified input limit of {} ms ({}–{} ms). Trim the clip and retry.",
            tenths / 10,
            tenths % 10,
            limit_ms.end(),
            limit_ms.start(),
            limit_ms.end()
        );
    }
    if duration_ms < *limit_ms.start() {
        let tenths = (duration_ms + 50) / 100;
        anyhow::bail!(
            "Source clip duration is {}.{} s ({duration_ms} ms), which is below {model}'s \
             verified input minimum of {} ms ({}–{} ms). Use a longer clip and retry.",
            tenths / 10,
            tenths % 10,
            limit_ms.start(),
            limit_ms.start(),
            limit_ms.end()
        );
    }
    Ok(())
}

/// One ISO-BMFF box: its 4-cc type and the span of its payload.
struct BoxSpan {
    box_type: [u8; 4],
    payload_start: u64,
    payload_end: u64,
}

/// Read a box header at `offset` and return its payload span, bounded by
/// `parent_end` (a size==0 box runs to the end of its containing scope).
/// size==1 boxes carry a 64-bit largesize. Returns `None` on any malformed
/// header: undersized size, overflow, or a span outside the parent.
fn read_box(file: &mut std::fs::File, offset: u64, parent_end: u64) -> Option<BoxSpan> {
    let mut header = [0u8; 8];
    file.seek(SeekFrom::Start(offset)).ok()?;
    file.read_exact(&mut header).ok()?;
    let size32 = u32::from_be_bytes(header[..4].try_into().ok()?);
    let (payload_start, payload_end) = match size32 {
        0 => (offset.checked_add(8)?, parent_end),
        1 => {
            let mut large = [0u8; 8];
            file.read_exact(&mut large).ok()?;
            let size64 = u64::from_be_bytes(large);
            if size64 < 16 {
                return None;
            }
            (offset.checked_add(16)?, offset.checked_add(size64)?)
        }
        n if n < 8 => return None,
        n => (offset.checked_add(8)?, offset.checked_add(u64::from(n))?),
    };
    if payload_end < payload_start || payload_end > parent_end {
        return None;
    }
    Some(BoxSpan {
        box_type: header[4..8].try_into().ok()?,
        payload_start,
        payload_end,
    })
}

/// Read an MP4/MOV file's duration in milliseconds from its `mvhd` metadata
/// box, using only the standard library.
///
/// The `moov` box may sit anywhere in the file — end-of-file for iPhone
/// MOVs — so top-level boxes are walked by seek rather than reading a fixed
/// header window. Returns `None` (fail-open) for anything untrustworthy:
/// non-ISO-BMFF data, fragmented files (`moof`/`mvex` — mvhd duration does
/// not cover fragments), a zero or missing/unreadable `mvhd` duration, and
/// I/O errors.
fn mp4_duration_ms(path: &std::path::Path) -> Option<i64> {
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    // Pass 1: scan the full top-level layout for fragmentation. A moof AFTER
    // moov (a non-conformant fragmented layout without mvex) must fail open
    // too, so no moov is parsed until the whole file is confirmed unfragmented.
    let mut offset: u64 = 0;
    while offset.checked_add(8).is_some_and(|o| o <= file_len) {
        let span = read_box(&mut file, offset, file_len)?;
        if matches!(&span.box_type, b"moof") {
            // Fragmented: the real duration lives in moof/traf, not mvhd.
            return None;
        }
        offset = span.payload_end;
    }
    // Pass 2: parse any moov boxes.
    let mut offset: u64 = 0;
    while offset.checked_add(8).is_some_and(|o| o <= file_len) {
        let span = read_box(&mut file, offset, file_len)?;
        if matches!(&span.box_type, b"moov")
            && let Some(ms) = parse_moov_duration(&mut file, span.payload_start, span.payload_end)
        {
            return Some(ms);
        }
        offset = span.payload_end;
    }
    None
}

/// Walk the children of a `moov` box looking for a usable `mvhd`; `mvex`
/// marks a fragmented movie (fail-open). The whole payload is scanned before
/// returning because `mvex` typically sits AFTER `mvhd` in a fragmented moov
/// — returning on the first mvhd would trust a duration that does not cover
/// the fragments. `None` when no usable mvhd is found.
fn parse_moov_duration(
    file: &mut std::fs::File,
    moov_payload_start: u64,
    moov_payload_end: u64,
) -> Option<i64> {
    let mut offset = moov_payload_start;
    let mut mvhd: Option<i64> = None;
    while offset.checked_add(8).is_some_and(|o| o <= moov_payload_end) {
        let span = read_box(file, offset, moov_payload_end)?;
        match &span.box_type {
            b"mvhd" => {
                if mvhd.is_none() {
                    mvhd = parse_mvhd_duration(file, span.payload_start, span.payload_end);
                }
            }
            b"mvex" => return None, // fragmented — mvhd duration not meaningful
            _ => {}
        }
        offset = span.payload_end;
    }
    mvhd
}

/// Parse an `mvhd` payload into a duration in milliseconds
/// (`duration * 1000 / timescale` — the exact value the provider validates
/// against). Supports mvhd version 0 (32-bit fields) and version 1 (64-bit).
/// A zero duration is indistinguishable from 'unknown' (live/streamed or
/// broken encoders) and returns `None` — the gate must fail open rather than
/// hard-reject on metadata it cannot trust.
fn parse_mvhd_duration(
    file: &mut std::fs::File,
    payload_start: u64,
    payload_end: u64,
) -> Option<i64> {
    let payload_len = payload_end.checked_sub(payload_start)?;
    let mut version_flags = [0u8; 4];
    file.seek(SeekFrom::Start(payload_start)).ok()?;
    file.read_exact(&mut version_flags).ok()?;
    let (timescale_offset, duration_offset, duration_width) = match version_flags[0] {
        0 => (12u64, 16u64, 4u64),
        1 => (20u64, 24u64, 8u64),
        _ => return None, // unknown version — fail open
    };
    if payload_len < duration_offset + duration_width {
        return None;
    }
    let mut timescale_buf = [0u8; 4];
    file.seek(SeekFrom::Start(payload_start + timescale_offset))
        .ok()?;
    file.read_exact(&mut timescale_buf).ok()?;
    let timescale = u32::from_be_bytes(timescale_buf);
    let mut duration_buf = [0u8; 8];
    let duration_width = usize::try_from(duration_width).ok()?;
    file.seek(SeekFrom::Start(payload_start + duration_offset))
        .ok()?;
    file.read_exact(&mut duration_buf[..duration_width]).ok()?;
    let duration = if duration_width == 8 {
        u64::from_be_bytes(duration_buf)
    } else {
        u64::from(u32::from_be_bytes(duration_buf[..4].try_into().ok()?))
    };
    if timescale == 0 {
        return None;
    }
    let ms = duration
        .checked_mul(1000)?
        .checked_div(u64::from(timescale))?;
    if ms == 0 {
        return None; // zero duration = unknown — fail open
    }
    i64::try_from(ms).ok()
}

/// Resolve the video reference: public URL as-is, local file → pre-flight
/// validation (containment, extension, size, input-duration gate) → upload
/// bridge. `model`/`spec` drive the duration gate's per-model limit.
async fn resolve_video_source(
    video_url: &str,
    ws: &crate::Workspace,
    model: &str,
    spec: VideoEditModel,
) -> anyhow::Result<String> {
    if crate::util::is_http_url(video_url) {
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
    // Pre-flight duration gate: reject clips the active model would reject
    // (hailuo-3: 2000–15000 ms) before any upload or billable submission —
    // a doomed clip otherwise costs an upload, a job, and a retry round-trip.
    check_local_input_duration(&canonical, model, spec)?;
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
    if crate::util::is_http_url(input) {
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

    fn preserve_full_output(&self) -> bool {
        // Result is the marker plus an LLM transcription capped at
        // 2048 tokens/3-min timeout that can exceed the 5 KB budget —
        // the Artist needs the full description to reason about its output.
        true
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
                    "description": "Text instruction describing the edit to apply (max 5000 chars)"
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

        // ── Mode validation: frame anchors vs references, model image gate ──
        let mode = validate_mode(spec, video_url, &images, first_frame, last_frame)?;

        // ── Per-model parameter validation (fail fast, before any upload) ──
        // Runs after mode validation — the seedance-2.5 duration rule is
        // mode-dependent (edit vs image-to-video).
        validate_params(spec, mode, duration)?;

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
                // validate_mode returns VideoRef only when video_url is present.
                let source_url = resolve_video_source(
                    video_url.expect("VideoRef mode implies video_url"),
                    ws,
                    &model,
                    spec,
                )
                .await?;
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

        // seedance-2.5 edit tasks natively require duration 'auto' — omit the
        // field (schema-valid; provider defaults to auto) instead of
        // forwarding a provider-rejected integer.
        if let Some(d) = duration
            && !omit_duration(spec, mode)
        {
            body["duration"] = json!(d);
        }

        // Video editing always targets OpenRouter (mahbot-1884) — a custom
        // chat endpoint never serves video models.
        let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string();
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
        // seedance-2.5 keeps its auto-duration edit rule; seedance-2.0-mini
        // gets explicit-duration edits; seedance-2.0/2.0-fast/1.5-pro stay
        // permissive (Unknown) so they never inherit either rule.
        assert_eq!(
            classify_model("bytedance/seedance-2.5"),
            VideoEditModel::Seedance
        );
        assert_eq!(
            classify_model("bytedance/seedance-2.0-mini"),
            VideoEditModel::SeedanceMini
        );
        assert_eq!(
            classify_model("bytedance/seedance-2.0"),
            VideoEditModel::Unknown
        );
        assert_eq!(
            classify_model("bytedance/seedance-2.0-fast"),
            VideoEditModel::Unknown
        );
        assert_eq!(
            classify_model("bytedance/seedance-1.5-pro"),
            VideoEditModel::Unknown
        );
        assert_eq!(classify_model("some/other-model"), VideoEditModel::Unknown);
    }

    #[test]
    fn seedance_duration_rules() {
        // Edit-classified (video-reference) requests omit duration entirely —
        // any value is accepted and dropped at body build.
        assert!(omit_duration(VideoEditModel::Seedance, EditMode::VideoRef));
        assert!(validate_params(VideoEditModel::Seedance, EditMode::VideoRef, Some(8)).is_ok());
        assert!(validate_params(VideoEditModel::Seedance, EditMode::VideoRef, None).is_ok());
        // Image-to-video (frame-anchor) keeps the catalog 4-30s range.
        assert!(!omit_duration(
            VideoEditModel::Seedance,
            EditMode::FrameAnchor
        ));
        assert!(validate_params(VideoEditModel::Seedance, EditMode::FrameAnchor, Some(8)).is_ok());
        assert!(validate_params(VideoEditModel::Seedance, EditMode::FrameAnchor, Some(3)).is_err());
        assert!(
            validate_params(VideoEditModel::Seedance, EditMode::FrameAnchor, Some(31)).is_err()
        );
        assert!(validate_params(VideoEditModel::Seedance, EditMode::FrameAnchor, None).is_ok());
        // seedance-2.0-mini accepts explicit durations (4-15s) and never omits
        // the field; validation does not branch on mode, so VideoRef covers it.
        assert!(!omit_duration(
            VideoEditModel::SeedanceMini,
            EditMode::VideoRef
        ));
        assert!(validate_params(VideoEditModel::SeedanceMini, EditMode::VideoRef, Some(4)).is_ok());
        assert!(
            validate_params(VideoEditModel::SeedanceMini, EditMode::VideoRef, Some(15)).is_ok()
        );
        assert!(
            validate_params(VideoEditModel::SeedanceMini, EditMode::VideoRef, Some(3)).is_err()
        );
        assert!(
            validate_params(VideoEditModel::SeedanceMini, EditMode::VideoRef, Some(16)).is_err()
        );
        assert!(validate_params(VideoEditModel::SeedanceMini, EditMode::VideoRef, None).is_ok());
        // Other models never omit duration.
        assert!(!omit_duration(VideoEditModel::Hailuo3, EditMode::VideoRef));
        assert!(!omit_duration(VideoEditModel::Unknown, EditMode::VideoRef));
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

    // ── Pre-flight input-duration gate ────────────────────────────────

    /// Build one ISO-BMFF box: big-endian 32-bit size + 4-cc type + payload.
    fn box_bytes(box_type: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = u32::try_from(8 + payload.len()).unwrap();
        let mut v = Vec::with_capacity(8 + payload.len());
        v.extend_from_slice(&size.to_be_bytes());
        v.extend_from_slice(&box_type);
        v.extend_from_slice(payload);
        v
    }

    /// Build an `mvhd` payload for the given version (0: 32-bit, 1: 64-bit
    /// timescale/duration fields).
    fn mvhd_payload(version: u8, timescale: u32, duration: u64) -> Vec<u8> {
        let mut payload = vec![version, 0, 0, 0]; // version + flags
        match version {
            0 => {
                payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
                payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
                payload.extend_from_slice(&timescale.to_be_bytes());
                payload.extend_from_slice(&u32::try_from(duration).unwrap().to_be_bytes());
            }
            1 => {
                payload.extend_from_slice(&0u64.to_be_bytes()); // creation_time
                payload.extend_from_slice(&0u64.to_be_bytes()); // modification_time
                payload.extend_from_slice(&timescale.to_be_bytes());
                payload.extend_from_slice(&duration.to_be_bytes());
            }
            _ => unreachable!(),
        }
        payload
    }

    /// Minimal mp4 byte layout: ftyp + mdat, then moov with the given children
    /// appended last — the iPhone-MOV end-of-file layout the parser must walk.
    fn mp4_bytes(moov_children: &[Vec<u8>]) -> Vec<u8> {
        let moov_payload: Vec<u8> = moov_children.iter().flatten().copied().collect();
        let mut bytes = box_bytes(*b"ftyp", b"isom");
        bytes.extend_from_slice(&box_bytes(*b"mdat", &[0u8; 64]));
        bytes.extend_from_slice(&box_bytes(*b"moov", &moov_payload));
        bytes
    }

    /// A fragmented mp4 layout: ftyp, a top-level moof, mdat, then moov with
    /// an over-long mvhd. Shared by the parser- and gate-level tests so the
    /// byte construction lives in exactly one place.
    fn fragmented_mp4_bytes() -> Vec<u8> {
        let mut bytes = box_bytes(*b"ftyp", b"isom");
        bytes.extend_from_slice(&box_bytes(*b"moof", &[]));
        bytes.extend_from_slice(&box_bytes(*b"mdat", &[0u8; 32]));
        bytes.extend_from_slice(&box_bytes(
            *b"moov",
            &box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_000)),
        ));
        bytes
    }

    /// Write an mp4 built from `moov_children` to a temp file; returns the
    /// tempdir (kept alive) and the file path.
    fn write_mp4(moov_children: &[Vec<u8>]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.mp4");
        std::fs::write(&path, mp4_bytes(moov_children)).unwrap();
        (dir, path)
    }

    /// Write raw bytes as `clip.mp4` under the given tempdir.
    fn write_raw(dir: &tempfile::TempDir, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join("clip.mp4");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn input_duration_limit_table_verified_only() {
        // Verified: hailuo-3 input clips 2000–15000 ms.
        assert_eq!(
            VideoEditModel::Hailuo3.input_duration_limit_ms(),
            Some(2_000..=15_000)
        );
        // Unverified (aleph-2) and trimming (seedance) models fail open.
        assert_eq!(VideoEditModel::Aleph2.input_duration_limit_ms(), None);
        assert_eq!(VideoEditModel::Seedance.input_duration_limit_ms(), None);
        assert_eq!(VideoEditModel::SeedanceMini.input_duration_limit_ms(), None);
        assert_eq!(VideoEditModel::Unknown.input_duration_limit_ms(), None);
    }

    #[test]
    fn mp4_duration_reads_mvhd_in_end_of_file_moov() {
        // iPhone-MOV layout: moov (with mvhd) sits at the very end of the
        // file, after mdat — the parser must seek, not read a header window.
        let (_dir, path) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_167))]);
        assert_eq!(mp4_duration_ms(&path), Some(35_167));
        // Non-1000 timescale: 9760 ticks @ 600/s = 16.266 s → 16266 ms.
        let (_dir, path) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 600, 9760))]);
        assert_eq!(mp4_duration_ms(&path), Some(16_266));
        // Version 1 (64-bit duration fields).
        let (_dir, path) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(1, 1000, 22_000))]);
        assert_eq!(mp4_duration_ms(&path), Some(22_000));
        // mvhd need not be the first moov child.
        let (_dir, path) = write_mp4(&[
            box_bytes(*b"udta", &[0u8; 8]),
            box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 8_000)),
        ]);
        assert_eq!(mp4_duration_ms(&path), Some(8_000));
    }

    #[test]
    fn mp4_duration_fails_open_on_fragmented_files() {
        // Any top-level moof marks the file fragmented — mvhd duration does
        // not cover fragments, so the check must fail open.
        let dir = tempfile::tempdir().unwrap();
        let path = write_raw(&dir, &fragmented_mp4_bytes());
        assert_eq!(mp4_duration_ms(&path), None);
        // mvex inside moov is the fragmented-initialization marker.
        let (_dir, path) = write_mp4(&[
            box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_000)),
            box_bytes(*b"mvex", &box_bytes(*b"trex", &[0u8; 24])),
        ]);
        assert_eq!(mp4_duration_ms(&path), None);
        // Non-conformant fragmented layout: moov WITHOUT mvex, then a moof
        // later in the file. The deferred top-level scan still detects the
        // fragment and fails open instead of trusting the mvhd duration.
        let mut bytes = box_bytes(*b"ftyp", b"isom");
        bytes.extend_from_slice(&box_bytes(
            *b"moov",
            &box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_000)),
        ));
        bytes.extend_from_slice(&box_bytes(*b"mdat", &[0u8; 32]));
        bytes.extend_from_slice(&box_bytes(*b"moof", &[]));
        let path = write_raw(&dir, &bytes);
        assert_eq!(mp4_duration_ms(&path), None);
    }

    #[test]
    fn mp4_duration_fails_open_on_unparseable_input() {
        let dir = tempfile::tempdir().unwrap();
        // Truncated header.
        assert_eq!(
            mp4_duration_ms(&write_raw(&dir, b"\x00\x00\x00\x18ftyp")),
            None
        );
        // Box size overruns the file.
        let mut bytes = vec![0u8; 16];
        bytes[..4].copy_from_slice(&1000u32.to_be_bytes());
        bytes[4..8].copy_from_slice(b"moov");
        assert_eq!(mp4_duration_ms(&write_raw(&dir, &bytes)), None);
        // No moov at all.
        let mut bytes = box_bytes(*b"ftyp", b"isom");
        bytes.extend_from_slice(&box_bytes(*b"mdat", &[0u8; 16]));
        assert_eq!(mp4_duration_ms(&write_raw(&dir, &bytes)), None);
        // Non-ISO-BMFF data (mkv-like bytes).
        assert_eq!(
            mp4_duration_ms(&write_raw(&dir, b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81")),
            None
        );
        // Zero timescale is unreadable (fail-open).
        let (_dir, path) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 0, 35_000))]);
        assert_eq!(mp4_duration_ms(&path), None);
        // Zero duration is indistinguishable from 'unknown' (live/streamed or
        // broken encoders) — fail open, never a hard-reject below the floor.
        let (_dir, path) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 0))]);
        assert_eq!(mp4_duration_ms(&path), None);
        // Missing file.
        assert_eq!(mp4_duration_ms(&dir.path().join("nope.mp4")), None);
    }

    #[test]
    fn check_local_input_duration_hard_rejects_only_verified_ranges() {
        let hailuo = ("minimax/hailuo-3", VideoEditModel::Hailuo3);
        // The observed live failure: a 35.2 s clip rejected by hailuo-3.
        let (_dir, overlong) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_200))]);
        let err = check_local_input_duration(&overlong, hailuo.0, hailuo.1).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("35.2 s"), "message: {msg}");
        assert!(msg.contains("35200 ms"), "message: {msg}");
        assert!(msg.contains("minimax/hailuo-3"), "message: {msg}");
        assert!(
            msg.contains("2000") && msg.contains("15000"),
            "message: {msg}"
        );
        assert!(msg.contains("exceeds"), "message: {msg}");
        assert!(msg.contains("Trim"), "message: {msg}");

        // Sub-minimum (1.5 s) is equally doomed — same verified range; the
        // error names the violated lower bound and the actionable direction.
        let (_dir, tooshort) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 1_500))]);
        let err = check_local_input_duration(&tooshort, hailuo.0, hailuo.1).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("1.5 s"), "message: {msg}");
        assert!(msg.contains("below"), "message: {msg}");
        assert!(msg.contains("2000"), "message: {msg}");
        assert!(msg.contains("longer clip"), "message: {msg}");

        // Boundary values pass: 2 s and 15 s exactly.
        let (_dir, min_ok) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 2_000))]);
        assert!(check_local_input_duration(&min_ok, hailuo.0, hailuo.1).is_ok());
        let (_dir, max_ok) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 15_000))]);
        assert!(check_local_input_duration(&max_ok, hailuo.0, hailuo.1).is_ok());

        // Case-insensitive extension: an over-long .MOV clip is rejected too.
        let dir = tempfile::tempdir().unwrap();
        let mov_path = dir.path().join("clip.MOV");
        std::fs::write(
            &mov_path,
            mp4_bytes(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_200))]),
        )
        .unwrap();
        assert!(check_local_input_duration(&mov_path, hailuo.0, hailuo.1).is_err());

        // Unknown model: no check, even for the same over-long clip.
        assert!(
            check_local_input_duration(&overlong, "some/other-model", VideoEditModel::Unknown)
                .is_ok()
        );
        // seedance family trims over-long inputs — never hard-rejected.
        assert!(
            check_local_input_duration(
                &overlong,
                "bytedance/seedance-2.5",
                VideoEditModel::Seedance
            )
            .is_ok()
        );
    }

    #[test]
    fn check_local_input_duration_fails_open_on_uncertain_inputs() {
        let hailuo = ("minimax/hailuo-3", VideoEditModel::Hailuo3);
        // Non-MP4/MOV containers (webm/mkv/avi) are never parsed.
        let dir = tempfile::tempdir().unwrap();
        let mkv = dir.path().join("clip.mkv");
        std::fs::write(&mkv, b"\x1a\x45\xdf\xa3garbage").unwrap();
        assert!(check_local_input_duration(&mkv, hailuo.0, hailuo.1).is_ok());
        // Corrupt mp4 fails open (cannot verify the provider would reject).
        let corrupt = dir.path().join("corrupt.mp4");
        std::fs::write(&corrupt, b"not a real mp4").unwrap();
        assert!(check_local_input_duration(&corrupt, hailuo.0, hailuo.1).is_ok());
        // Fragmented mp4 (moof) fails open even with an over-long mvhd.
        let frag = write_raw(&dir, &fragmented_mp4_bytes());
        assert!(check_local_input_duration(&frag, hailuo.0, hailuo.1).is_ok());
        // Zero-duration mvhd is 'unknown' — fails open below the floor.
        let (_dir, zero) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 0))]);
        assert!(check_local_input_duration(&zero, hailuo.0, hailuo.1).is_ok());
    }
}
