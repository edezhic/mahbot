//! The `video_gen` tool: one job on OpenRouter's async videos API — submit, poll, download —
//! returning the path the reply can embed. One tool, three mutually exclusive shapes decided by
//! the inputs: text-to-video (one optional reference image), image-to-video anchored on the first
//! and/or last frame, and editing a referenced clip. Not local encoding — cutting, trimming and
//! retiming a real recording is `hyperframes`.

use super::media_catalog::video::VideoModelInfo;
use crate::Tool;
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use std::io::{Read, Seek, SeekFrom};

/// The resolution asked for when the caller names none: the omni default's native tier.
const DEFAULT_RESOLUTION: &str = "720p";

/// Maximum length of the prompt in characters (per-model limits surface via the model nuances).
const MAX_PROMPT_CHARS: usize = 5000;

/// Input clip size cap for every video model: guards the local-file upload path against
/// unbounded reads.
const MAX_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Input image size cap (provider-declared 30 MB bound for hailuo-3).
const MAX_IMAGE_BYTES: u64 = 30 * 1024 * 1024;

/// Maximum reference images guiding one edit: hailuo-3 allows 9, seedance-2.0-mini documents 9
/// (Cloudflare's relay schema says 4 — not adopted). Applied uniformly.
const MAX_REFERENCE_IMAGES: usize = 9;

/// The parameters only a fresh generation takes. With a video reference or frame anchors they are
/// refused, never silently dropped: a value the provider never sees is exactly the ambiguity this
/// one tool must not have.
const GENERATION_ONLY: &[&str] = &[
    "resolution",
    "aspect_ratio",
    "size",
    "generate_audio",
    "seed",
];

/// Per-model video capability classification, driving validation and the description. The active
/// model is user-switchable via `/video_models`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoModel {
    /// minimax/hailuo-3: 5-15s output, 2K fixed, always audio, no seed.
    Hailuo3,
    /// runway/aleph-2: output mirrors input, preserves audio, seed best-effort.
    Aleph2,
    /// bytedance/seedance-2.5: whole-frame restyle, 480p/720p; edits require duration 'auto'.
    Seedance,
    /// bytedance/seedance-2.0-mini: whole-frame restyle; edits accept explicit 4-15s durations.
    SeedanceMini,
    /// Any other configured model — permissive validation.
    Unknown,
}

fn classify_model(model: &str) -> VideoModel {
    let m = model.to_ascii_lowercase();
    // Substring match (not exact) so vendor-prefixed IDs like "minimax/hailuo-3" resolve;
    // Unknown covers every other model permissively.
    if m.contains("hailuo") {
        VideoModel::Hailuo3
    } else if m.contains("aleph") {
        VideoModel::Aleph2
    } else if m.contains("seedance") && m.contains("2.0-mini") {
        VideoModel::SeedanceMini
    } else if m.contains("seedance") && m.contains("2.5") {
        VideoModel::Seedance
    } else {
        VideoModel::Unknown
    }
}

impl VideoModel {
    /// The input-clip duration bounds in milliseconds — the single table backing the pre-flight
    /// local-clip check.
    ///
    /// `None` fails open: the seedance family trims over-long inputs instead of rejecting them,
    /// and an unrecognised model lands on the same arm.
    fn input_duration_limit_ms(self) -> Option<std::ops::RangeInclusive<i64>> {
        match self {
            // minimax/hailuo-3: 2000–15000 ms per clip (provider and MiniMax docs); the same bound
            // behind the 50 MB input cap.
            VideoModel::Hailuo3 => Some(2_000..=15_000),
            // aleph-2 mirrors its input's duration; the seedance models TRIM over-long inputs —
            // never hard-reject; an unrecognised model has no bound.
            VideoModel::Aleph2
            | VideoModel::Seedance
            | VideoModel::SeedanceMini
            | VideoModel::Unknown => None,
        }
    }
}

/// Static per-model editing nuances for the `<active-models-opts>` block: informational facts
/// only — an edit takes no resolution or editing-mode argument. Keyed via [`classify_model`] so
/// rendering cannot drift from validation.
pub(crate) fn video_model_nuances(model: &str) -> Option<&'static str> {
    match classify_model(model) {
        VideoModel::Hailuo3 => Some(
            "localized instruction edits + video-to-video motion transfer; 2K output; \
             source style preserved by default",
        ),
        VideoModel::Seedance => Some(
            "whole-frame restyle; output 480p/720p via OpenRouter; edit-mode duration \
             is automatic ('auto') — the 4-30s duration range applies to generation, \
             not edits",
        ),
        VideoModel::SeedanceMini => Some(
            "whole-frame restyle; output 480p/720p via OpenRouter; edit-mode duration \
             is explicit (4-15s, same range as generation)",
        ),
        VideoModel::Aleph2 | VideoModel::Unknown => None,
    }
}

/// Which of the three mutually exclusive shapes one call has, decided by the inputs it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// Text-to-video, optionally from one reference image.
    Generate,
    /// Image-to-video anchored on the exact first and/or last frame.
    Anchors,
    /// Editing the referenced clip, optionally guided by reference images.
    Edit,
}

impl Shape {
    /// How a refusal names what the caller passed instead of a fresh generation — `None` when
    /// that is exactly what it asked for, and nothing is refused on that account.
    const fn passed_input(self) -> Option<&'static str> {
        match self {
            Self::Generate => None,
            Self::Anchors => Some("frame anchors"),
            Self::Edit => Some("a video reference"),
        }
    }
}

/// One optional string input, read strictly: absent or explicit `null` is no input, a present
/// non-string is a wrong-type refusal, and a present empty string is a refusal too — it names
/// nothing, and treating it as absent would change the shape of a call that bills.
fn named_input<'a>(args: &'a serde_json::Value, key: &str) -> anyhow::Result<Option<&'a str>> {
    match super::get_opt_str_strict(args, key)? {
        Some("") => anyhow::bail!("`{key}` is empty. Give it a value, or leave the argument out."),
        named => Ok(named),
    }
}

/// One parsed call: the shape it asks for and every value that shape accepts.
#[derive(Debug)]
struct Request<'a> {
    shape: Shape,
    /// The one text input, shared by all three shapes.
    prompt: &'a str,
    /// The clip to edit — present only on [`Shape::Edit`].
    video_url: Option<&'a str>,
    /// One start picture for a fresh clip, or up to [`MAX_REFERENCE_IMAGES`] guiding an edit.
    images: Vec<String>,
    /// The exact first and/or last frame — present only on [`Shape::Anchors`].
    first_frame: Option<&'a str>,
    last_frame: Option<&'a str>,
    /// The output length; a fresh generation's range is the catalog's to state.
    duration: Option<i64>,
    /// The fresh-generation parameters, refused on the other two shapes.
    resolution: Option<&'a str>,
    aspect_ratio: Option<&'a str>,
    size: Option<&'a str>,
    generate_audio: Option<bool>,
    seed: Option<i64>,
}

impl<'a> Request<'a> {
    /// Read the arguments strictly and decide the shape. Every refusal here happens before any
    /// upload or billable submission.
    fn parse(args: &'a serde_json::Value) -> anyhow::Result<Self> {
        let prompt = super::get_str(args, "prompt")?;
        let char_count = prompt.chars().count();
        if char_count == 0 {
            anyhow::bail!("The prompt must not be empty. Describe the clip or the edit to apply.");
        }
        if char_count > MAX_PROMPT_CHARS {
            anyhow::bail!(
                "The prompt is too long: {char_count} chars (max {MAX_PROMPT_CHARS}). \
                 Retry with a shorter prompt."
            );
        }
        // A present input names a real one or is a refusal: an empty value would otherwise read
        // as absent and silently turn the call into a different, paid shape.
        let video_url = named_input(args, "video_url")?;
        let images: Vec<String> = super::get_str_array(args, "images")?;
        if let Some(empty) = images.iter().position(String::is_empty) {
            anyhow::bail!(
                "`images` entry {} is empty. Name a picture, or leave the entry out.",
                empty + 1
            );
        }
        let first_frame = named_input(args, "first_frame")?;
        let last_frame = named_input(args, "last_frame")?;
        let shape = match (video_url, first_frame.is_some() || last_frame.is_some()) {
            (Some(_), true) => anyhow::bail!(
                "A video reference and frame anchors are mutually exclusive — a request cannot \
                 mix frame and reference roles. Remove `video_url` or the frames and retry."
            ),
            (Some(_), false) => Shape::Edit,
            (None, true) => Shape::Anchors,
            (None, false) => Shape::Generate,
        };
        let request = Self {
            shape,
            prompt,
            video_url,
            images,
            first_frame,
            last_frame,
            duration: super::get_opt_i64(args, "duration")?,
            resolution: super::get_opt_str(args, "resolution"),
            aspect_ratio: super::get_opt_str(args, "aspect_ratio"),
            size: super::get_opt_str(args, "size"),
            generate_audio: super::get_opt_bool(args, "generate_audio")?,
            seed: super::get_opt_i64(args, "seed")?,
        };
        request.check_shape(args)?;
        Ok(request)
    }

    /// Refuse the inputs the decided shape does not take.
    fn check_shape(&self, args: &serde_json::Value) -> anyhow::Result<()> {
        if let Some(passed) = self.shape.passed_input() {
            for key in GENERATION_ONLY {
                if args.get(*key).is_some_and(|value| !value.is_null()) {
                    anyhow::bail!(
                        "`{key}` belongs to a fresh generation and cannot be used with {passed} — \
                         remove it and retry."
                    );
                }
            }
        }
        match self.shape {
            Shape::Generate if self.images.len() > 1 => anyhow::bail!(
                "A fresh generation starts from at most one reference image (received {}). \
                 Retry with exactly one image.",
                self.images.len(),
            ),
            Shape::Anchors if !self.images.is_empty() => anyhow::bail!(
                "Frame anchors are mutually exclusive with reference images — a request cannot \
                 mix frame and reference roles. Remove `images` and retry."
            ),
            Shape::Edit if self.images.len() > MAX_REFERENCE_IMAGES => anyhow::bail!(
                "At most {MAX_REFERENCE_IMAGES} reference images are supported, got {}. \
                 Reduce the number of reference images and retry.",
                self.images.len()
            ),
            _ => Ok(()),
        }
    }

    /// The per-model gates, run before any upload: an image-incompatible model and a duration
    /// outside the model's own range are refused here rather than by the provider after a bill.
    fn validate(&self, spec: VideoModel) -> anyhow::Result<()> {
        let image_inputs =
            !self.images.is_empty() || self.first_frame.is_some() || self.last_frame.is_some();
        if spec == VideoModel::Aleph2 && self.shape != Shape::Generate && image_inputs {
            anyhow::bail!(
                "aleph-2 does not support image inputs (reference images or frame \
                 anchors). Retry without images or switch to hailuo-3."
            );
        }
        validate_duration(spec, self.shape, self.duration)
    }
}

/// Whether the request body must omit `duration`: seedance-2.5 edit-classified tasks require
/// 'auto' and an explicit integer is provider-rejected (omission is schema-valid). Image-to-video
/// keeps the catalog range; seedance-2.0-mini accepts 4-15s, so it never omits.
fn omit_duration(spec: VideoModel, shape: Shape) -> bool {
    matches!((spec, shape), (VideoModel::Seedance, Shape::Edit))
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

/// Per-model duration validation: a fresh generation's range is the catalog's to state, and the
/// seedance-2.5 rule is shape-dependent, so the two decide the arm together.
fn validate_duration(spec: VideoModel, shape: Shape, duration: Option<i64>) -> anyhow::Result<()> {
    match (spec, shape) {
        // A fresh generation's range is the catalog's to state, and a seedance-2.5 edit takes
        // 'auto' — the field is omitted at body build (see [`omit_duration`]) — so neither is
        // bounded here.
        (_, Shape::Generate) | (VideoModel::Seedance, Shape::Edit) => {}
        (VideoModel::Hailuo3, _) => {
            ensure_duration_in(duration, 5..=15, "hailuo-3 output duration")?;
        }
        (VideoModel::Aleph2, _) => {
            if duration.is_some() {
                anyhow::bail!(
                    "aleph-2 output mirrors the input clip duration — the duration \
                     parameter is not supported. Remove it and retry."
                );
            }
        }
        // Explicit 4-15s edit duration per native docs/catalog; the check here refuses anything
        // outside it.
        (VideoModel::SeedanceMini, _) => {
            ensure_duration_in(duration, 4..=15, "seedance-2.0-mini duration")?;
        }
        (VideoModel::Seedance, Shape::Anchors) => {
            ensure_duration_in(duration, 4..=30, "seedance-2.5 image-to-video duration")?;
        }
        (VideoModel::Unknown, _) => {
            if let Some(d) = duration
                && d <= 0
            {
                anyhow::bail!("duration must be a positive integer, got {d}");
            }
        }
    }
    Ok(())
}

/// Insert an optional value into the request body under `key`, skipping parameters the caller
/// did not supply.
fn set_opt<T: Into<serde_json::Value>>(body: &mut serde_json::Value, key: &str, value: Option<T>) {
    if let Some(v) = value {
        body[key] = v.into();
    }
}

/// The resolution the request must name: the caller's own verbatim, else [`DEFAULT_RESOLUTION`]
/// when the caller named neither a resolution nor an exact size — the two say the same thing —
/// and the model's declared capabilities offer it.
fn requested_resolution(request: &Request<'_>, info: Option<&VideoModelInfo>) -> Option<String> {
    if let Some(named) = request.resolution {
        return Some(named.to_string());
    }
    if request.size.is_some() {
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

/// The containers the editing provider takes as an uploaded clip. Narrower than the product's own
/// [`crate::util::VIDEO_EXTENSIONS`], which is what an arrival reads: a local clip is uploaded as
/// `video/mp4`, so a container outside this set would be refused by the provider after the upload
/// was paid for.
const UPLOADABLE_CLIPS: &[&str] = &["mp4", "mov", "mkv", "avi", "webm"];

/// Resolve the video reference: public URL as-is, local file → pre-flight validation
/// (containment, extension, size, duration gate) → upload bridge.
async fn resolve_video_source(
    video_url: &str,
    ws: &crate::Workspace,
    model: &str,
    spec: VideoModel,
) -> anyhow::Result<String> {
    if crate::util::is_http_url(video_url) {
        return Ok(video_url.to_string());
    }
    // Only clips already inside the workspace media dirs may be uploaded — arbitrary local files
    // are an exfiltration risk.
    let canonical = super::resolve_local_media_path(video_url, ws, "clip").await?;
    if !crate::util::has_extension(&canonical, UPLOADABLE_CLIPS) {
        anyhow::bail!(
            "Local clip must be one of {} for the editing provider, got: {}",
            UPLOADABLE_CLIPS.join(", "),
            canonical.display()
        );
    }
    let len = tokio::fs::metadata(&canonical).await?.len();
    if len > MAX_INPUT_BYTES {
        anyhow::bail!("Source clip is limited to 50 MB, got {len} bytes. Trim the clip and retry.");
    }
    // Pre-flight duration gate: reject clips the active model would reject before any upload or
    // billable submission — a doomed clip otherwise costs an upload, a job, and a retry.
    check_local_input_duration(&canonical, model, spec)?;
    crate::util::upload_bridge::upload_video_ephemeral(&canonical).await
}

/// Resolve an image input: public URL (GET-validated — a broken reference bills anyway) or a
/// local file in workspace uploads or generated. `label` names the input in error messages.
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
    let canonical = super::resolve_local_media_path(input, ws, label).await?;
    if !crate::util::is_image_extension(&canonical) {
        anyhow::bail!(
            "Local image must have a recognized image extension \
             (jpg, jpeg, png, webp, heic, heif), got: {}",
            canonical.display()
        );
    }
    let len = tokio::fs::metadata(&canonical).await?.len();
    if len > MAX_IMAGE_BYTES {
        anyhow::bail!(
            "{label} is limited to 30 MB, got {len} bytes. Use a smaller image and retry."
        );
    }
    crate::util::upload_bridge::upload_image_ephemeral(&canonical).await
}

/// Pre-flight input-duration gate for a local clip, run after containment, extension and size
/// validation and before any upload.
///
/// Hard-rejects only when the active model has a verified input-duration limit AND the parsed
/// MP4/MOV duration falls outside it. Fails open for everything uncertain: other containers,
/// fragmented or corrupt files, unreadable metadata, and models without a verified limit.
fn check_local_input_duration(
    canonical: &std::path::Path,
    model: &str,
    spec: VideoModel,
) -> anyhow::Result<()> {
    let Some(limit_ms) = spec.input_duration_limit_ms() else {
        return Ok(());
    };
    // Only ISO-BMFF containers carry a readable mvhd duration; the other accepted extensions
    // (webm/mkv/avi) are skipped by design.
    if !crate::util::has_extension(canonical, &["mp4", "mov"]) {
        return Ok(());
    }
    let Some(duration_ms) = mp4_duration_ms(canonical) else {
        // Fragmented, corrupt, or unreadable — cannot verify the provider
        // would reject it, so the submission proceeds (fail-open).
        tracing::debug!(
            path = %canonical.display(),
            "video_gen pre-flight duration unreadable — failing open"
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

/// Read a box header at `offset` and return its payload span, bounded by `parent_end` (a
/// size==0 box runs to the end of its scope; size==1 carries a 64-bit largesize). `None` on
/// any malformed header.
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

/// Read an MP4/MOV file's duration in milliseconds from its `mvhd` box, using only the
/// standard library. The `moov` box may sit anywhere (end-of-file for iPhone MOVs), so
/// top-level boxes are walked by seek rather than from a fixed header window. `None`
/// (fail-open) for non-ISO-BMFF data, fragmented files (`moof`/`mvex` — mvhd does not cover
/// fragments), a zero or unreadable duration, and I/O errors.
fn mp4_duration_ms(path: &std::path::Path) -> Option<i64> {
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    // Pass 1: scan the full top-level layout for fragmentation. A moof AFTER moov must fail
    // open too, so no moov is parsed until the whole file is confirmed unfragmented.
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

/// Walk the children of a `moov` box looking for a usable `mvhd`; `mvex` marks a fragmented
/// movie. The whole payload is scanned before returning, because `mvex` typically sits AFTER
/// `mvhd` — returning on the first mvhd would trust a duration that does not cover the fragments.
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

/// Parse an `mvhd` payload into milliseconds (`duration * 1000 / timescale`, the exact value
/// the provider validates), versions 0 and 1. A zero duration is indistinguishable from
/// 'unknown' and returns `None` — the gate must fail open.
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

/// The submit body for one call: the model and prompt plus every parameter the shape accepts.
/// Reference material is resolved and uploaded here, so every refusal that needs no network has
/// already happened.
async fn request_body(
    ws: &crate::Workspace,
    request: &Request<'_>,
    model: &str,
    spec: VideoModel,
    info: Option<&VideoModelInfo>,
) -> anyhow::Result<serde_json::Value> {
    let mut body = json!({
        "model": model,
        "prompt": request.prompt,
    });
    match request.shape {
        Shape::Generate => {
            set_opt(&mut body, "resolution", requested_resolution(request, info));
            set_opt(&mut body, "aspect_ratio", request.aspect_ratio);
            set_opt(&mut body, "size", request.size);
            set_opt(&mut body, "generate_audio", request.generate_audio);
            set_opt(&mut body, "seed", request.seed);
            // Reference-load failures are hard errors — never silently degrade to text-to-video.
            let mut references: Vec<crate::util::ReferenceImage> = Vec::new();
            if let Some(image) = request.images.first() {
                let path = super::resolve_local_media_path(image, ws, "reference image").await?;
                references.push(
                    crate::util::load_reference_image(&path, super::MAX_REFERENCE_IMAGE_BYTES)
                        .await?,
                );
            }
            // Aggregate body budget: one reference at the per-image cap already approaches the
            // ~2 MB client-side body budget once base64 + JSON overhead is added.
            super::fit_request_body_budget(
                &mut body,
                &mut references,
                super::MAX_REQUEST_BODY_BYTES,
            )?;
        }
        Shape::Anchors => {
            let mut frames: Vec<serde_json::Value> = Vec::new();
            for (anchor, label, frame_type) in [
                (request.first_frame, "first-frame anchor", "first_frame"),
                (request.last_frame, "last-frame anchor", "last_frame"),
            ] {
                if let Some(anchor) = anchor {
                    let url = resolve_image_input(anchor, ws, label).await?;
                    frames.push(json!({
                        "type": "image_url",
                        "image_url": { "url": url },
                        "frame_type": frame_type
                    }));
                }
            }
            body["frame_images"] = json!(frames);
        }
        Shape::Edit => {
            let source = request
                .video_url
                .expect("the edit shape implies a video reference");
            let source_url = resolve_video_source(source, ws, model, spec).await?;
            let mut references = vec![json!({
                "type": "video_url",
                "video_url": { "url": source_url }
            })];
            for image in &request.images {
                let url = resolve_image_input(image, ws, "reference image").await?;
                references.push(json!({
                    "type": "image_url",
                    "image_url": { "url": url }
                }));
            }
            body["input_references"] = json!(references);
        }
    }
    // seedance-2.5 edit tasks require 'auto' — omit the field (schema-valid) instead of
    // forwarding a provider-rejected integer.
    if let Some(duration) = request.duration
        && !omit_duration(spec, request.shape)
    {
        body["duration"] = json!(duration);
    }
    Ok(body)
}

/// Tool for one model-driven video job — generate a clip, or edit a referenced one — through
/// OpenRouter's async videos API: submit, poll, download, and return the path so the agent can
/// send it via `[VIDEO:path]` in its reply.
pub struct VideoGenTool;

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
                    "description": format!(
                        "The text description of the clip to produce, or of the edit to apply \
                         (max {MAX_PROMPT_CHARS} chars)"
                    )
                },
                "video_url": {
                    "type": "string",
                    "description": "The clip to edit: a public HTTPS URL, or a local path from the workspace uploads dir (shown as [Saved video: /path] in the chat) or the generated dir (shown as [VIDEO:path]). Present, the call edits that clip; omitted, a new clip is generated. Mutually exclusive with first_frame/last_frame"
                },
                "images": {
                    "type": "array",
                    "items": { "type": "string" },
                    "maxItems": MAX_REFERENCE_IMAGES,
                    "description": format!(
                        "A new clip's start picture (ONE, a local path from the workspace uploads \
                         dir — received attachments — or the generated dir — previously generated \
                         media; never a URL), or an edit's guiding pictures (up to \
                         {MAX_REFERENCE_IMAGES}, those local paths or public HTTPS URLs) for \
                         style/subject/identity. An edit's guiding pictures are JPG/JPEG/PNG/WEBP/\
                         HEIC/HEIF, at most 30 MB each. Mutually exclusive with \
                         first_frame/last_frame"
                    )
                },
                "duration": {
                    "type": "integer",
                    "description": "Output duration in seconds (model-dependent; valid values come from the active model's capability block)"
                },
                "resolution": {
                    "type": "string",
                    "description": "Desired resolution for a new clip (model-dependent; valid values come from the active model's capability block)"
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio for a new clip (model-dependent; valid values come from the active model's capability block)"
                },
                "size": {
                    "type": "string",
                    "pattern": "\\d+x\\d+",
                    "description": "Exact size of a new clip in WxH format (e.g. 1280x720; model-dependent)"
                },
                "generate_audio": {
                    "type": "boolean",
                    "description": "Whether a new clip carries an audio track (model-dependent)"
                },
                "seed": {
                    "type": "integer",
                    "description": "Seed for a repeatable generation of a new clip"
                },
                "first_frame": {
                    "type": "string",
                    "description": "Path or public HTTPS URL of an image to use as the exact first frame (image-to-video) — JPG/JPEG/PNG/WEBP/HEIC/HEIF, at most 30 MB. Local paths are accepted from the workspace uploads dir (received attachments) or the generated dir (previously generated images). Mutually exclusive with video_url and images"
                },
                "last_frame": {
                    "type": "string",
                    "description": "Path or public HTTPS URL of an image to use as the exact last frame (image-to-video) — JPG/JPEG/PNG/WEBP/HEIC/HEIF, at most 30 MB. Local paths are accepted from the workspace uploads dir (received attachments) or the generated dir (previously generated images). Mutually exclusive with video_url and images"
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
        let (marker, path) = self.run(ws, args).await?;
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
        let (marker, path) = self.run(ws, args).await?;
        Ok(super::video_output(marker, &path, super::TimelinePictures::Attached).await)
    }
}

impl VideoGenTool {
    /// The job itself: parse, validate, submit, poll, download, save — answering the `[VIDEO:path]`
    /// marker and the file it names, so the one run serves both the text-only and the
    /// payload-carrying answer.
    async fn run(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<(String, std::path::PathBuf)> {
        let request = Request::parse(&args)?;
        let user_name = crate::agent::tool_user_name();
        let model = crate::users::resolve_video_model(&user_name).await;
        let spec = classify_model(&model);
        request.validate(spec)?;

        // The catalog only matters to a fresh generation: it states the default resolution tier.
        let catalog = match request.shape {
            Shape::Generate => super::media_catalog::video::get_catalog().await,
            Shape::Anchors | Shape::Edit => None,
        };
        let info = catalog.as_deref().and_then(|catalog| catalog.find(&model));
        let body = request_body(ws, &request, &model, spec, info).await?;

        // Video work always targets OpenRouter — a custom chat endpoint never serves video models.
        let api_base = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);
        let labels = match request.shape {
            // An edit is named apart from a fresh generation, so a failure says what was asked.
            Shape::Edit => super::VideoJobLabels::EDIT,
            Shape::Generate | Shape::Anchors => super::VideoJobLabels::GENERATION,
        };
        let video_bytes = super::fetch_async_video(&api_base, &body, labels).await?;

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

    fn shape(args: &serde_json::Value) -> Shape {
        Request::parse(args).expect("a valid call").shape
    }

    fn refusal(args: &serde_json::Value) -> String {
        Request::parse(args)
            .expect_err("a refused call")
            .to_string()
    }

    #[test]
    fn classify_model_matches_known_and_unknown() {
        assert_eq!(classify_model("minimax/hailuo-3"), VideoModel::Hailuo3);
        assert_eq!(classify_model("MINIMAX/HAILUO-3"), VideoModel::Hailuo3);
        assert_eq!(classify_model("runway/aleph-2"), VideoModel::Aleph2);
        assert_eq!(
            classify_model("runway/aleph-2-20260729"),
            VideoModel::Aleph2
        );
        // seedance-2.5 keeps its auto-duration edit rule; seedance-2.0-mini gets explicit-duration
        // edits; the rest stay permissive (Unknown).
        assert_eq!(
            classify_model("bytedance/seedance-2.5"),
            VideoModel::Seedance
        );
        assert_eq!(
            classify_model("bytedance/seedance-2.0-mini"),
            VideoModel::SeedanceMini
        );
        assert_eq!(
            classify_model("bytedance/seedance-2.0"),
            VideoModel::Unknown
        );
        assert_eq!(
            classify_model("bytedance/seedance-2.0-fast"),
            VideoModel::Unknown
        );
        assert_eq!(
            classify_model("bytedance/seedance-1.5-pro"),
            VideoModel::Unknown
        );
        assert_eq!(classify_model("some/other-model"), VideoModel::Unknown);
    }

    #[test]
    fn the_inputs_decide_the_shape() {
        assert_eq!(shape(&json!({"prompt": "a wave"})), Shape::Generate);
        assert_eq!(
            shape(&json!({"prompt": "a wave", "images": ["a.png"]})),
            Shape::Generate
        );
        assert_eq!(
            shape(&json!({"prompt": "a wave", "first_frame": "f.png"})),
            Shape::Anchors
        );
        assert_eq!(
            shape(&json!({"prompt": "a wave", "last_frame": "l.png"})),
            Shape::Anchors
        );
        assert_eq!(
            shape(&json!({"prompt": "restyle", "video_url": "c.mp4"})),
            Shape::Edit
        );
        assert_eq!(
            shape(&json!({"prompt": "restyle", "video_url": "c.mp4", "images": ["a.png"]})),
            Shape::Edit
        );
        // An explicit null is absence, the shared helper's rule — never an empty input.
        for args in [
            json!({"prompt": "a wave", "video_url": null}),
            json!({"prompt": "a wave", "first_frame": null}),
            json!({"prompt": "a wave", "last_frame": null}),
            json!({"prompt": "a wave", "images": null}),
        ] {
            assert_eq!(shape(&args), Shape::Generate);
        }
    }

    #[test]
    fn a_present_input_is_a_value_or_a_refusal() {
        // An empty input names nothing: read as absent it would silently turn the call into a
        // different — paid — shape, so every one of them is refused.
        for args in [
            json!({"prompt": "restyle", "video_url": ""}),
            json!({"prompt": "p", "first_frame": ""}),
            json!({"prompt": "p", "last_frame": ""}),
            json!({"prompt": "p", "images": ["a.png", ""]}),
        ] {
            let empty = refusal(&args);
            assert!(empty.contains("empty"), "{empty}");
        }
        for (args, named) in [
            (json!({"prompt": "restyle", "video_url": 7}), "video_url"),
            (json!({"prompt": "p", "first_frame": 3}), "first_frame"),
            (
                json!({"prompt": "p", "last_frame": ["l.png"]}),
                "last_frame",
            ),
        ] {
            let wrong = refusal(&args);
            assert!(wrong.contains(named), "{wrong}");
            assert!(wrong.contains("a string"), "{wrong}");
        }
    }

    #[test]
    fn the_shapes_refuse_each_other() {
        let mixed = refusal(&json!({"prompt": "p", "video_url": "c.mp4", "first_frame": "f.png"}));
        assert!(mixed.contains("mutually exclusive"), "{mixed}");
        let anchored =
            refusal(&json!({"prompt": "p", "first_frame": "f.png", "images": ["a.png"]}));
        assert!(anchored.contains("mutually exclusive"), "{anchored}");
        let two = refusal(&json!({"prompt": "p", "images": ["a.png", "b.png"]}));
        assert!(two.contains("at most one"), "{two}");
        let ten: Vec<String> = (0..10).map(|i| format!("r{i}.png")).collect();
        let too_many = refusal(&json!({"prompt": "p", "video_url": "c.mp4", "images": ten}));
        assert!(too_many.contains("At most 9"), "{too_many}");
        let nine: Vec<String> = (0..9).map(|i| format!("r{i}.png")).collect();
        assert_eq!(
            shape(&json!({"prompt": "p", "video_url": "c.mp4", "images": nine})),
            Shape::Edit
        );
    }

    #[test]
    fn a_parameter_the_shape_does_not_take_is_refused() {
        for (args, named) in [
            (
                json!({"prompt": "p", "video_url": "c.mp4", "resolution": "720p"}),
                "resolution",
            ),
            (
                json!({"prompt": "p", "video_url": "c.mp4", "generate_audio": true}),
                "generate_audio",
            ),
            (
                json!({"prompt": "p", "video_url": "c.mp4", "seed": 7}),
                "seed",
            ),
            (
                json!({"prompt": "p", "first_frame": "f.png", "size": "1280x720"}),
                "size",
            ),
            (
                json!({"prompt": "p", "last_frame": "l.png", "aspect_ratio": "16:9"}),
                "aspect_ratio",
            ),
        ] {
            let reason = refusal(&args);
            assert!(reason.contains(named), "{reason}");
            assert!(reason.contains("fresh generation"), "{reason}");
        }
        // A fresh generation keeps every one of them.
        assert_eq!(
            shape(&json!({"prompt": "p", "resolution": "720p", "seed": 7})),
            Shape::Generate
        );
    }

    #[test]
    fn the_prompt_is_shared_and_bounded() {
        let empty = refusal(&json!({"prompt": ""}));
        assert!(empty.contains("must not be empty"), "{empty}");
        let too_long = refusal(&json!({"prompt": "x".repeat(MAX_PROMPT_CHARS + 1)}));
        assert!(too_long.contains("too long"), "{too_long}");
        assert_eq!(
            shape(&json!({"prompt": "x".repeat(MAX_PROMPT_CHARS), "video_url": "c.mp4"})),
            Shape::Edit
        );
        let missing = refusal(&json!({}));
        assert!(missing.contains("prompt"), "{missing}");
    }

    #[test]
    fn seedance_duration_rules() {
        // Edit-classified (video-reference) requests omit duration entirely —
        // any value is accepted and dropped at body build.
        assert!(omit_duration(VideoModel::Seedance, Shape::Edit));
        assert!(validate_duration(VideoModel::Seedance, Shape::Edit, Some(8)).is_ok());
        assert!(validate_duration(VideoModel::Seedance, Shape::Edit, None).is_ok());
        // Image-to-video (frame-anchor) keeps the catalog 4-30s range.
        assert!(!omit_duration(VideoModel::Seedance, Shape::Anchors));
        assert!(validate_duration(VideoModel::Seedance, Shape::Anchors, Some(8)).is_ok());
        assert!(validate_duration(VideoModel::Seedance, Shape::Anchors, Some(3)).is_err());
        assert!(validate_duration(VideoModel::Seedance, Shape::Anchors, Some(31)).is_err());
        assert!(validate_duration(VideoModel::Seedance, Shape::Anchors, None).is_ok());
        // seedance-2.0-mini accepts explicit durations (4-15s) and never omits the field.
        assert!(!omit_duration(VideoModel::SeedanceMini, Shape::Edit));
        assert!(validate_duration(VideoModel::SeedanceMini, Shape::Edit, Some(4)).is_ok());
        assert!(validate_duration(VideoModel::SeedanceMini, Shape::Edit, Some(15)).is_ok());
        assert!(validate_duration(VideoModel::SeedanceMini, Shape::Edit, Some(3)).is_err());
        assert!(validate_duration(VideoModel::SeedanceMini, Shape::Edit, Some(16)).is_err());
        assert!(validate_duration(VideoModel::SeedanceMini, Shape::Edit, None).is_ok());
        // Other models never omit duration.
        assert!(!omit_duration(VideoModel::Hailuo3, Shape::Edit));
        assert!(!omit_duration(VideoModel::Unknown, Shape::Edit));
        // A fresh generation's duration is the catalog's to bound, never this table's.
        assert!(validate_duration(VideoModel::Hailuo3, Shape::Generate, Some(3)).is_ok());
        assert!(validate_duration(VideoModel::Aleph2, Shape::Generate, Some(8)).is_ok());
    }

    fn validated(args: &serde_json::Value, spec: VideoModel) -> anyhow::Result<()> {
        Request::parse(args)?.validate(spec)
    }

    #[test]
    fn a_call_validates_against_its_model() {
        let err = validated(
            &json!({"prompt": "p", "video_url": "c.mp4", "duration": 3}),
            VideoModel::Hailuo3,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("5–15"), "{err}");
        assert!(
            validated(
                &json!({"prompt": "p", "video_url": "c.mp4", "duration": 5}),
                VideoModel::Aleph2
            )
            .is_err()
        );
        // aleph-2 takes no image inputs; a fresh generation's single reference keeps today's path.
        assert!(
            validated(
                &json!({"prompt": "p", "video_url": "c.mp4", "images": ["a.png"]}),
                VideoModel::Aleph2
            )
            .is_err()
        );
        assert!(
            validated(
                &json!({"prompt": "p", "first_frame": "f.png"}),
                VideoModel::Aleph2
            )
            .is_err()
        );
        assert!(
            validated(
                &json!({"prompt": "p", "images": ["a.png"]}),
                VideoModel::Aleph2
            )
            .is_ok()
        );
        assert!(
            validated(
                &json!({"prompt": "p", "video_url": "c.mp4"}),
                VideoModel::Aleph2
            )
            .is_ok()
        );
        assert!(
            validated(
                &json!({"prompt": "p", "video_url": "c.mp4"}),
                VideoModel::Unknown
            )
            .is_ok()
        );
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
        let named = json!({ "prompt": "p", "resolution": "1080p" });
        let request = Request::parse(&named).unwrap();
        assert_eq!(
            requested_resolution(&request, Some(&declaring(&["720p"]))).as_deref(),
            Some("1080p"),
            "the caller's own value wins verbatim"
        );
        let sized = json!({ "prompt": "p", "size": "1280x720" });
        let request = Request::parse(&sized).unwrap();
        assert_eq!(
            requested_resolution(&request, Some(&declaring(&["720p"]))),
            None,
            "a size says the same thing as a resolution, so none is added"
        );
    }

    #[tokio::test]
    async fn body_carries_the_shape_it_parsed() {
        let ws = crate::workspace::test_ws("video-gen-body");

        let args = json!({ "prompt": "a wave", "duration": 5 });
        let request = Request::parse(&args).unwrap();
        let body = request_body(&ws, &request, "m", VideoModel::Unknown, None)
            .await
            .unwrap();
        assert_eq!(body["prompt"], "a wave");
        assert_eq!(body["duration"], 5);
        assert!(body.get("resolution").is_none());
        assert!(body.get("frame_images").is_none());

        // The default tier is added when the model declares it.
        let body = request_body(
            &ws,
            &request,
            "m",
            VideoModel::Unknown,
            Some(&declaring(&["720p"])),
        )
        .await
        .unwrap();
        assert_eq!(body["resolution"], DEFAULT_RESOLUTION);

        // A public reference needs no upload, so an edit body is checkable without files.
        let args = json!({ "prompt": "restyle", "video_url": "https://example.test/c.mp4", "duration": 8 });
        let request = Request::parse(&args).unwrap();
        let body = request_body(&ws, &request, "m", VideoModel::Unknown, None)
            .await
            .unwrap();
        assert_eq!(body["input_references"][0]["type"], "video_url");
        assert_eq!(body["duration"], 8);
        // seedance-2.5 edit tasks require 'auto': the field is omitted instead.
        let body = request_body(&ws, &request, "m", VideoModel::Seedance, None)
            .await
            .unwrap();
        assert!(body.get("duration").is_none());
    }

    #[test]
    fn input_duration_limit_table_bounds_hailuo3_only() {
        // Hailuo-3 input clips: 2000–15000 ms.
        assert_eq!(
            VideoModel::Hailuo3.input_duration_limit_ms(),
            Some(2_000..=15_000)
        );
        // Aleph-2 (its output mirrors the input) and the trimming seedance models
        // fail open.
        assert_eq!(VideoModel::Aleph2.input_duration_limit_ms(), None);
        assert_eq!(VideoModel::Seedance.input_duration_limit_ms(), None);
        assert_eq!(VideoModel::SeedanceMini.input_duration_limit_ms(), None);
        assert_eq!(VideoModel::Unknown.input_duration_limit_ms(), None);
    }

    fn box_bytes(box_type: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = u32::try_from(8 + payload.len()).unwrap();
        let mut v = Vec::with_capacity(8 + payload.len());
        v.extend_from_slice(&size.to_be_bytes());
        v.extend_from_slice(&box_type);
        v.extend_from_slice(payload);
        v
    }

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

    fn mp4_bytes(moov_children: &[Vec<u8>]) -> Vec<u8> {
        let moov_payload: Vec<u8> = moov_children.iter().flatten().copied().collect();
        let mut bytes = box_bytes(*b"ftyp", b"isom");
        bytes.extend_from_slice(&box_bytes(*b"mdat", &[0u8; 64]));
        bytes.extend_from_slice(&box_bytes(*b"moov", &moov_payload));
        bytes
    }

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

    fn write_mp4(moov_children: &[Vec<u8>]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.mp4");
        std::fs::write(&path, mp4_bytes(moov_children)).unwrap();
        (dir, path)
    }

    fn write_raw(dir: &tempfile::TempDir, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join("clip.mp4");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn mp4_duration_reads_mvhd_in_end_of_file_moov() {
        // iPhone-MOV layout: moov (with mvhd) sits at the very end, after mdat — the parser must
        // seek, not read a header window.
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
        // Any top-level moof marks the file fragmented — mvhd does not cover fragments, so the
        // check must fail open.
        let dir = tempfile::tempdir().unwrap();
        let path = write_raw(&dir, &fragmented_mp4_bytes());
        assert_eq!(mp4_duration_ms(&path), None);
        // mvex inside moov is the fragmented-initialization marker.
        let (_dir, path) = write_mp4(&[
            box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 35_000)),
            box_bytes(*b"mvex", &box_bytes(*b"trex", &[0u8; 24])),
        ]);
        assert_eq!(mp4_duration_ms(&path), None);
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
        // Zero duration is indistinguishable from 'unknown' — fail open, never a hard-reject.
        let (_dir, path) = write_mp4(&[box_bytes(*b"mvhd", &mvhd_payload(0, 1000, 0))]);
        assert_eq!(mp4_duration_ms(&path), None);
        // Missing file.
        assert_eq!(mp4_duration_ms(&dir.path().join("nope.mp4")), None);
    }

    #[test]
    fn check_local_input_duration_hard_rejects_only_bounded_models() {
        let hailuo = ("minimax/hailuo-3", VideoModel::Hailuo3);
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

        // Sub-minimum (1.5 s) is equally doomed — the same bound; the
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
            check_local_input_duration(&overlong, "some/other-model", VideoModel::Unknown).is_ok()
        );
        // seedance family trims over-long inputs — never hard-rejected.
        assert!(
            check_local_input_duration(&overlong, "bytedance/seedance-2.5", VideoModel::Seedance)
                .is_ok()
        );
    }

    #[test]
    fn check_local_input_duration_fails_open_on_uncertain_inputs() {
        let hailuo = ("minimax/hailuo-3", VideoModel::Hailuo3);
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
