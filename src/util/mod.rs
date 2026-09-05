//! Utility modules for shared helper functions.

pub(crate) mod catalog_cache;
pub(crate) mod error;
pub(crate) mod html;
pub(crate) mod http;
pub(crate) mod json;
pub mod lock;
pub(crate) mod macros;
pub(crate) mod managed_bin;
pub(crate) mod media_target;
pub(crate) mod model_state;
#[cfg(test)]
pub(crate) mod test;
pub(crate) mod tree_sitter;
pub(crate) mod upload_bridge;

use directories::UserDirs;
use regex::Regex;
use regex::RegexBuilder;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context as _, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use rand::RngExt;
#[cfg(any(test, feature = "voice-tests"))]
use rand::SeedableRng;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use tracing::error;

/// Extension trait to unwrap poisoned lock results, replacing
/// `.unwrap_or_else(std::sync::PoisonError::into_inner)` with `.unwrap_poison()`.
pub trait UnwrapPoison {
    type Inner;
    /// Unwrap the lock result, recovering the inner value even if the lock is poisoned.
    #[must_use]
    fn unwrap_poison(self) -> Self::Inner;
}

impl<T> UnwrapPoison for Result<T, std::sync::PoisonError<T>> {
    type Inner = T;
    fn unwrap_poison(self) -> T {
        self.unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The regex pattern for `[KIND:path]` media markers.
///
/// This is the single source of truth for the marker pattern. Both the case-sensitive
/// [`MEDIA_MARKER_RE`] and the case-insensitive [`TELEGRAM_MEDIA_MARKER_RE`] are built
/// from this constant, so adding a new marker kind here automatically keeps both in sync.
const MEDIA_MARKER_PATTERN: &str = r"\[(?P<kind>IMAGE|AUDIO|VIDEO):(?P<path>[^\]\r\n]+)\]";

/// Matches `[IMAGE:path]`, `[AUDIO:path]`, or `[VIDEO:path]` markers in message content.
///
/// **Invariant — marker stripping:** When enriching messages, IMAGE markers
/// are ALWAYS preserved — they're needed for native image-part integration via
/// `to_message_content()` — while all non-IMAGE markers (AUDIO, VIDEO, and any
/// future marker kinds) are stripped from the content by `enrich_message`,
/// which mirrors the `parse_image_markers()` pattern. Adding a new marker kind
/// to this regex will cause it to be automatically stripped unless the closure
/// is explicitly updated to preserve it.
pub(crate) static MEDIA_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(MEDIA_MARKER_PATTERN).expect("MEDIA_MARKER_RE must compile"));

/// Case-insensitive variant of [`MEDIA_MARKER_RE`] used by `telegram.rs` to
/// accept `[image:...]`, `[Image:...]`, etc. Built from the same
/// [`MEDIA_MARKER_PATTERN`] constant to stay in sync.
pub(crate) static TELEGRAM_MEDIA_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(MEDIA_MARKER_PATTERN)
        .case_insensitive(true)
        .build()
        .expect("TELEGRAM_MEDIA_MARKER_RE must compile")
});

/// Extract the `kind` and `path` named groups from a [`MEDIA_MARKER_RE`] / [`TELEGRAM_MEDIA_MARKER_RE`] capture.
///
/// Returns `(kind, path)` as string slices borrowed from the original haystack.
///
/// # Panics
///
/// Panics if either named group is missing — this should never happen with the
/// well-formed regex since the pattern requires both groups to match.
#[must_use]
pub(crate) fn parse_media_marker<'h>(caps: &regex::Captures<'h>) -> (&'h str, &'h str) {
    let kind = caps
        .name("kind")
        .expect("parse_media_marker: expected 'kind' group")
        .as_str();
    let path = caps
        .name("path")
        .expect("parse_media_marker: expected 'path' group")
        .as_str();
    (kind, path)
}

/// Provenance tag prepended to every synthetic User-role image message the agent
/// loop injects, so a vision-capable model can distinguish a tool-injected image
/// from a user-uploaded one. Consumed (with the image marker) by the image-strip
/// path so a stripped message reads cleanly.
pub(crate) const INJECTED_IMAGE_TAG: &str = "<injected-tool-result-image>";

/// Compose the synthetic User-role message carrying a tool-injected image: the
/// provenance tag followed by the `[IMAGE:{data_uri}]` marker the provider layer
/// turns into a native image part.
#[must_use]
pub(crate) fn injected_image_user_message(data_uri: &str) -> String {
    format!("{INJECTED_IMAGE_TAG}\n[IMAGE:{data_uri}]")
}

/// Truncate a string to `max_chars` Unicode characters, appending "…" if truncated.
#[must_use]
pub fn truncate(input: &str, max_chars: usize) -> String {
    match input.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", input[..idx].trim_end()),
        None => input.to_string(),
    }
}

/// Truncate to at most `max_bytes` bytes at a UTF-8 char boundary (no ellipsis).
#[must_use]
pub(crate) fn truncate_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        &s[..s.floor_char_boundary(max_bytes)]
    }
}

/// Convert a string reference to `None` if empty, otherwise `Some(s.to_string())`.
///
/// Useful when building query structs where empty filters mean "no filter".
#[must_use]
pub(crate) fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Word character classification shared by the shell grep engine's `-w`
/// handling and editor word navigation.
///
/// The grep side is a conservative approximation of unicode `\w` that gates
/// the `-w` → `\b(?:pat)\b` translation; the editor side uses it for
/// word-boundary detection. Currently identical; a deliberate one-sided
/// divergence (e.g. ASCII-only for GNU-grep parity) must split this helper
/// rather than silently change the other subsystem.
#[must_use]
pub(crate) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Current Unix timestamp in milliseconds since the epoch.
///
/// Returns `0` if the system clock is set before the Unix epoch (January 1, 1970).
///
/// Returns `u64` — sufficient for timestamps up to ~500 million years from now.
#[must_use]
#[expect(clippy::cast_possible_truncation)]
pub(crate) fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Parse an env var as a whole number of seconds, falling back to `default_secs`.
///
/// Shared by the bounded tool I/O waits (shell output drain, FIFO reads, round
/// consolidation) so their env-override pattern stays in one place.
/// A value of `0` produces an immediate timeout — deliberate for tests;
/// operators should set a positive value.
#[must_use]
pub(crate) fn env_duration_secs(name: &str, default_secs: u64) -> std::time::Duration {
    let secs = std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_secs);
    std::time::Duration::from_secs(secs)
}

/// Format a byte slice as a lowercase hex string.
///
/// Each byte is written as two hex digits, yielding a string of length
/// `bytes.len() * 2`.
#[must_use]
pub(crate) fn hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

/// Standard base64 (RFC 4648, `+`/`/`, `=` padding) encoding of a byte slice.
#[must_use]
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

/// Standard base64 (RFC 4648, `+`/`/`, `=` padding) decoding of a byte slice.
///
/// Returns `None` on malformed input (wrong length, invalid characters, or
/// non-canonical padding).
#[must_use]
pub(crate) fn base64_decode(s: &str) -> Option<Vec<u8>> {
    STANDARD.decode(s).ok()
}

/// Verify a file's SHA256 hash matches the expected hex string.
///
/// Shared model-integrity verifier extracted from the three near-identical
/// private streaming copies in `audio::tts`, `audio::local_transcriber`, and
/// `audio::voice` (whose empty-`expected` skip semantics and error wording had
/// drifted).  Model-download integrity is security-adjacent, so the copies are
/// consolidated here.
///
/// * If `expected` is empty, verification is skipped (returns `Ok`) — the
///   canonical "no hash configured" semantics that won the reconciliation.
/// * Uses streaming SHA256 via [`Sha256::update`] to avoid loading the entire
///   file into memory (model files can be multiple GB).
pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    if expected.is_empty() {
        return Ok(()); // no hash configured — skip verification
    }

    let mut hasher = Sha256::new();
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open {} for SHA256 verification", path.display()))?;
    let mut buf = vec![0u8; 65536]; // 64 KB heap buffer
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex_string(&hasher.finalize());
    if actual != expected {
        anyhow::bail!(
            "SHA256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

/// Expand a leading tilde (`~`) to the user's home directory.
///
/// Checks `$HOME` first (Unix, Git Bash on Windows), then `$USERPROFILE`
/// (cmd.exe / PowerShell). If neither is set, returns the path unchanged
/// (which means `~`-prefixed entries will be skipped by callers that
/// check for expansion success).
#[must_use]
pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix('~') {
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
        if let Ok(home) = home {
            return PathBuf::from(home).join(stripped.trim_start_matches('/'));
        }
    }
    PathBuf::from(path)
}

/// Resolve the shared `~/.mahbot/models/` directory via the CONFIG storage root.
///
/// Returns `None` if the storage root hasn't been initialized yet.  Per-model
/// subdirectories are joined by each consumer (e.g. `audio::models_subdir`).
#[must_use]
pub(crate) fn models_dir() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .map(|root| root.join("models"))
}

/// Run a blocking I/O operation with awareness of the current Tokio runtime.
///
/// - **Multi-threaded runtime:** wraps the call in
///   [`tokio::task::block_in_place`] so the runtime can re-schedule the
///   blocking thread to other tasks.
/// - **Current-thread runtime** or **no runtime:** calls `f()` directly —
///   blocking is safe in those contexts, and `block_in_place` would panic
///   on a current-thread runtime.
///
/// Use this instead of a bare `std::fs::canonicalize` (or other fast blocking
/// syscall) inside async functions that may run on a multi-threaded worker
/// pool. Prefer this over [`tokio::task::spawn_blocking`] for operations that
/// complete in < ~1 ms (where thread-spawn overhead dominates).
#[must_use]
pub(crate) fn with_block_in_place<T>(f: impl FnOnce() -> T) -> T {
    if let Ok(handle) = tokio::runtime::Handle::try_current()
        && handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread
    {
        return tokio::task::block_in_place(f);
    }
    f()
}

/// Extract a human-readable message from a panic payload returned by
/// [`catch_unwind`](futures_util::FutureExt::catch_unwind).
#[must_use]
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Log panics/cancellations from a `join_all`-aggregated batch of spawned
/// task results, keeping the enclosing loop alive.
pub(crate) fn log_join_failures(
    results: Vec<Result<(), tokio::task::JoinError>>,
    panic_log: &str,
    cancelled_log: &str,
) {
    for result in results {
        if let Err(e) = result {
            if e.is_panic() {
                let payload = e.into_panic();
                error!(error = %panic_message(&*payload), "{panic_log}");
            } else {
                error!("{cancelled_log}");
            }
        }
    }
}

/// Byte cap for failure-detail dumps (error chains, raw verdict responses).
/// Retry-exhaustion chains embed up to 13 per-attempt errors, and comment
/// dumps are read verbatim by downstream agents — the sandwich truncation
/// keeps the head (outermost context) and tail (last attempt's cause).
pub(crate) const FAILURE_DETAIL_CAP: usize = 24_000;

/// Canonical scrub+truncate failure-comment detail: scrub secrets, then
/// sandwich-truncate to [`FAILURE_DETAIL_CAP`]. Failure dumps rendered
/// through this path keep one canonical ordering and cap. (Some adjacent
/// sites deliberately diverge — scrub-only early-returns and a
/// truncate-then-scrub raw dump — and are intentionally not unified here.)
#[must_use]
pub(crate) fn failure_detail(text: &str, label: &str) -> String {
    truncate_sandwich(&scrub_credentials(text), FAILURE_DETAIL_CAP, label)
}

/// Truncate a string to at most `max_bytes` bytes using a head/tail
/// "sandwich" strategy: keeps the first ~2/3 and last ~1/3, inserting an
/// omission marker between them. Returns the input unchanged if it fits
/// within the limit.
///
/// The marker format is `"... (N bytes omitted at {label} truncation)\n"`,
/// where `label` provides context for the truncation (e.g., `"shell output"`,
/// `"tool output"`, `"stderr"`).
///
/// Slicing respects UTF-8 character boundaries via `floor_char_boundary`.
/// An overlap guard is included as defense-in-depth; it only triggers if
/// the head and tail ranges would intersect (impossible under the 2/3 + 1/3
/// split, but guards against future ratio changes).
#[must_use]
pub(crate) fn truncate_sandwich(s: &str, max_bytes: usize, label: &str) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head_bytes = max_bytes * 2 / 3;
    let tail_bytes = max_bytes / 3;
    let head_end = s.floor_char_boundary(head_bytes);
    let tail_start = s.floor_char_boundary(s.len().saturating_sub(tail_bytes));
    if head_end < tail_start {
        let omitted = s[head_end..tail_start].len();
        format!(
            "{}... ({} bytes omitted at {label} truncation)\n{}",
            &s[..head_end],
            omitted,
            &s[tail_start..]
        )
    } else {
        // Head and tail would overlap — simple truncation fallback
        let boundary = s.floor_char_boundary(max_bytes);
        let mut out = s[..boundary].to_string();
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("\n... [{label} truncated at {max_bytes} bytes]"),
        );
        out
    }
}

/// Shared byte budget for tool output passed to the LLM: used by the shell
/// spill threshold, the read tool preview cap, and [`truncate_tool_output`].
pub(crate) const TOOL_OUTPUT_BUDGET_BYTES: usize = 5_000;

/// Truncate tool output for LLM consumption (delegates to [`truncate_sandwich`]
/// with the shared [`TOOL_OUTPUT_BUDGET_BYTES`] limit). Returns input unchanged if within limit.
#[must_use]
pub(crate) fn truncate_tool_output(output: &str) -> String {
    truncate_sandwich(output, TOOL_OUTPUT_BUDGET_BYTES, "tool output")
}

/// Map a read image file's actual bytes to a raster MIME label, or `None` for a
/// non-native raster. Used by [`local_image_to_data_uri`] so the MIME matches
/// the bytes (not the extension), and the native set stays single-sourced in
/// [`image_format_native_label`].
fn mime_for_raster_bytes(bytes: &[u8]) -> Option<String> {
    let label = image_format_native_label(image::guess_format(bytes).ok()?)?;
    Some(format!("image/{}", label.to_ascii_lowercase()))
}

/// Read a local image file and return a base64 data URI suitable for native
/// image-part model input (e.g., `data:image/png;base64,...`). The MIME subtype
/// is derived from the file's actual raster bytes (magic sniff), falling back
/// to the path extension when the bytes aren't a recognised native raster.
pub(crate) async fn local_image_to_data_uri(path: &std::path::Path) -> anyhow::Result<String> {
    let bytes = tokio::fs::read(path).await?;
    let mime = mime_for_raster_bytes(&bytes).unwrap_or_else(|| mime_for_extension(path).to_owned());
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(&bytes)))
}

/// Inbound-photo compression bounds for non-Artist roles: the
/// longest-side cap and JPEG quality of the single ingestion-time re-encode.
pub(crate) const INBOUND_IMAGE_MAX_SIDE: u32 = 1024;
const INBOUND_IMAGE_JPEG_QUALITY: u8 = 85;

/// Input-size ceiling for the inbound-photo decode, aligned with the
/// reference-image path's [`MAX_REFERENCE_INPUT_BYTES`] pattern: a decoded
/// bitmap can be far larger than its compressed file, so over-cap files are
/// refused from metadata BEFORE the file is read (the fail-open caller falls
/// back to the original bytes as a data URI instead). Shared by the media-target
/// classifier and the GUI render path so a single 50 MiB cap cannot drift.
pub(crate) const INBOUND_IMAGE_MAX_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Read a local image file and return a bounded-JPEG data URI: longest side
/// capped at [`INBOUND_IMAGE_MAX_SIDE`], quality
/// [`INBOUND_IMAGE_JPEG_QUALITY`], alpha flattened onto white, EXIF
/// orientation applied. Fail-open callers fall back to
/// [`local_image_to_data_uri`] on any decode/encode error.
pub(crate) async fn local_image_to_compressed_data_uri(
    path: &std::path::Path,
) -> anyhow::Result<String> {
    Ok(local_image_to_compressed_data_uri_with_meta(path)
        .await?
        .data_uri)
}

/// Result of bounded inbound-image compression, with the post-EXIF/post-resize
/// dimensions and the source-format label for annotation rendering.
pub(crate) struct CompressedImageMeta {
    pub data_uri: String,
    pub width: u32,
    pub height: u32,
    pub format: String,
}

/// Read a local image file and return a bounded-JPEG data URI together with the
/// post-EXIF/post-resize dimensions of the encoded JPEG. Single source of truth
/// for the image payload's annotation dims (they describe the very encode the
/// data-URI carries).
pub(crate) async fn local_image_to_compressed_data_uri_with_meta(
    path: &std::path::Path,
) -> anyhow::Result<CompressedImageMeta> {
    // Metadata-first cap check (matches the reference-image path's bounded
    // read): refuse over-cap files BEFORE the read, so a huge file never
    // enters memory just to be refused.
    let meta = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to access inbound image {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!("Inbound image {} is not a regular file", path.display());
    }
    if meta.len() > INBOUND_IMAGE_MAX_INPUT_BYTES {
        anyhow::bail!(
            "Inbound image {} is {} bytes — over the {} MiB decode cap; passing the original through",
            path.display(),
            meta.len(),
            INBOUND_IMAGE_MAX_INPUT_BYTES / (1024 * 1024),
        );
    }
    let bytes = tokio::fs::read(path).await?;
    let (out, width, height, format) = with_block_in_place(|| compress_inbound_image(&bytes))?;
    Ok(CompressedImageMeta {
        data_uri: format!("{JPEG_DATA_URI_PREFIX}{}", STANDARD.encode(&out)),
        width,
        height,
        format,
    })
}

/// Map a decodable raster format to its uppercase source-format label used in
/// image annotations (`"PNG" | "JPEG" | "WEBP"`). Returns `None` for any format
/// mahbot does not attach natively (GIF/BMP/...). This is the single source of
/// the native PNG/JPEG/WebP set: both the read tool's content sniff and the
/// inbound-compression path derive their format decision here, so a future
/// change to the native set only touches this one function (no drift risk).
#[must_use]
pub(crate) fn image_format_native_label(fmt: image::ImageFormat) -> Option<&'static str> {
    use image::ImageFormat;
    match fmt {
        ImageFormat::Png => Some("PNG"),
        ImageFormat::Jpeg => Some("JPEG"),
        ImageFormat::WebP => Some("WEBP"),
        _ => None,
    }
}

/// One bounded compression step for inbound photos: decode, apply EXIF
/// orientation, downscale the longest side to [`INBOUND_IMAGE_MAX_SIDE`]
/// (aspect-preserving, Triangle filter, min 1 px), flatten alpha onto white,
/// and re-encode as JPEG at [`INBOUND_IMAGE_JPEG_QUALITY`]. Returns the
/// encoded bytes, the final (post-resize/post-flatten) dimensions, and the
/// uppercase source-format label (`"PNG" | "JPEG" | "WEBP"`, falling back to
/// `"IMAGE"`). Reuses the existing `exif_orientation` /
/// `flatten_alpha_onto_white` helpers. The input-size ceiling is enforced by
/// the caller (`local_image_to_compressed_data_uri_with_meta`, metadata-first).
fn compress_inbound_image(bytes: &[u8]) -> anyhow::Result<(Vec<u8>, u32, u32, String)> {
    use image::GenericImageView;
    let mut img = image::load_from_memory(bytes).context("Failed to decode inbound image")?;
    let format = match image::guess_format(bytes) {
        Ok(f) => image_format_native_label(f).unwrap_or("IMAGE"),
        Err(_) => "IMAGE",
    };
    if let Some(orientation) = exif_orientation(bytes) {
        img.apply_orientation(orientation);
    }
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    let img = if longest > INBOUND_IMAGE_MAX_SIDE {
        #[expect(clippy::cast_precision_loss)]
        let scale = INBOUND_IMAGE_MAX_SIDE as f32 / longest as f32;
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let nw = (w as f32 * scale).round().max(1.0) as u32;
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let nh = (h as f32 * scale).round().max(1.0) as u32;
        img.resize(nw, nh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgb = flatten_alpha_onto_white(&img);
    let (width, height) = rgb.dimensions();
    let mut out = Vec::new();
    {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            &mut out,
            INBOUND_IMAGE_JPEG_QUALITY,
        );
        encoder
            .encode(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                image::ExtendedColorType::Rgb8,
            )
            .context("Failed to encode compressed inbound image")?;
    }
    Ok((out, width, height, format.to_string()))
}

// ── Reference-image loading & compression (image_gen / video_gen) ────────

/// Input-size ceiling for the reference-image compression path (aligned with
/// video_edit's 50 MB input cap). Over-cap files must be read fully before
/// compression, so this keeps the path bounded on pathological inputs.
const MAX_REFERENCE_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Combined input-size ceiling across all references of one request: the
/// per-image ceiling does not bound a multi-reference total, and the fail-open
/// path (no catalog cap) would otherwise hold up to 16 × 50 MB of source bytes
/// in memory before the body budget runs.
pub(crate) const MAX_TOTAL_REFERENCE_INPUT_BYTES: u64 = 100 * 1024 * 1024;

/// Bound on a reference-image read: guards the narrow metadata→read window
/// where a path swapped to a FIFO/special file could otherwise block forever
/// (the is_file check runs on the pre-read metadata). The blocked read task
/// lingers until the special file resolves, but the tool call itself errors
/// visibly instead of hanging.
const REFERENCE_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(1);

/// Data-URI prefix for ladder-compressed references (the ladder always
/// produces JPEG).
const JPEG_DATA_URI_PREFIX: &str = "data:image/jpeg;base64,";

/// Compression ladder: (downscale factor, JPEG quality) steps, mildest first.
/// A same-format PNG re-encode can GROW the file (verified: +3% on the exact
/// failing image), so every step crosses to JPEG with alpha flattened onto
/// white; later steps downscale. The ladder is a small fixed bound (~6 steps).
const REFERENCE_COMPRESSION_LADDER: &[(f32, u8)] = &[
    (1.0, 85),
    (1.0, 70),
    (1.0, 55),
    (0.75, 70),
    (0.5, 70),
    (0.5, 45),
];

/// A validated reference image for a generation request: under-cap images pass
/// through unchanged (original bytes, original format); over-cap images are
/// compressed via the bounded ladder. Held in memory only — no disk artifacts.
pub(crate) struct ReferenceImage {
    data_uri: String,
    /// Original file bytes kept for later ladder steps (aggregate body budget);
    /// holding them avoids re-reading the file (no TOCTOU window).
    source_bytes: Vec<u8>,
    next_step: usize,
    /// Terminal state after [`ReferenceImage::release_source_bytes`]: the
    /// request body is final, so no further compression is possible.
    released: bool,
}

impl ReferenceImage {
    /// The base64 data URI to embed in the request.
    #[must_use]
    pub(crate) fn data_uri(&self) -> &str {
        &self.data_uri
    }

    /// True while the compression ladder still has steps left.
    #[must_use]
    pub(crate) fn has_compression_left(&self) -> bool {
        !self.released && self.next_step < REFERENCE_COMPRESSION_LADDER.len()
    }

    /// Apply the next compression step (used by the aggregate body budget).
    /// Errors — loudly, not panicking — when no steps remain or the source
    /// bytes were released, so a caller contract violation surfaces instead of
    /// corrupting state. The guards are unreachable from the current
    /// budget-loop caller (which filters on `has_compression_left` first);
    /// they exist to bound any future caller.
    pub(crate) fn compress_more(&mut self) -> anyhow::Result<()> {
        if self.released {
            anyhow::bail!("Reference image is final — its source bytes were released");
        }
        if self.next_step >= REFERENCE_COMPRESSION_LADDER.len() {
            anyhow::bail!("Reference image compression ladder exhausted");
        }
        let out =
            with_block_in_place(|| compress_reference_step(&self.source_bytes, self.next_step))?;
        self.next_step += 1;
        self.data_uri = format!("{JPEG_DATA_URI_PREFIX}{}", STANDARD.encode(&out));
        Ok(())
    }

    /// Drop the retained original bytes once the request body is final — no
    /// further compression is possible (or needed) after this point.
    pub(crate) fn release_source_bytes(&mut self) {
        self.released = true;
        self.source_bytes = Vec::new();
    }
}

/// Load a reference image for generation: validate existence, the input-size
/// ceiling, and regular-file-ness via metadata BEFORE any read; then the
/// format (PNG/JPEG/WebP by extension AND content sniff); compress over-cap
/// images via the bounded ladder until they fit `max_bytes`.
pub(crate) async fn load_reference_image(
    path: &std::path::Path,
    max_bytes: u64,
) -> anyhow::Result<ReferenceImage> {
    // Metadata-first: a missing file must report not-found (not a format
    // error), pathological inputs are refused without being read, and special
    // files (FIFOs etc.) are refused so the read below cannot block forever.
    let meta = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to access reference image {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!(
            "Reference image {} is not a regular file — refusing to read it.",
            path.display(),
        );
    }
    if meta.len() > MAX_REFERENCE_INPUT_BYTES {
        anyhow::bail!(
            "Reference image {} is limited to 50 MB, got {} bytes. Use a smaller image.",
            path.display(),
            meta.len(),
        );
    }
    // Extension gate before reading the file: HEIC/HEIF, GIF, BMP and unknown
    // files are rejected without reading them (a 49 MB HEIC must not be read
    // just to be refused).
    check_reference_extension(path)?;
    let bytes = tokio::time::timeout(REFERENCE_READ_TIMEOUT, tokio::fs::read(path))
        .await
        .map_err(|_| anyhow::anyhow!("Timed out reading reference image {}", path.display()))?
        .map_err(|e| anyhow::anyhow!("Failed to read reference image {}: {e}", path.display()))?;

    // Content sniff: catches mislabeled or undecodable files that the
    // extension gate let through (the format gate rejects them up front
    // instead of sending undecoded bytes to the provider).
    let format = sniff_reference_content(path, &bytes)?;

    // Under the cap → pass through unchanged (current behavior preserved).
    if bytes.len() as u64 <= max_bytes {
        return Ok(ReferenceImage {
            data_uri: format!(
                "data:{};base64,{}",
                format.to_mime_type(),
                STANDARD.encode(&bytes)
            ),
            source_bytes: bytes,
            next_step: 0,
            released: false,
        });
    }

    // Over the cap → bounded compression ladder, in-memory only.
    let mut step = 0;
    loop {
        if step >= REFERENCE_COMPRESSION_LADDER.len() {
            // Exact bytes + decimal MB ("1500000 bytes (1.5 MB)") — errors
            // must not confuse MiB with MB.
            #[expect(clippy::cast_precision_loss)]
            let cap = format!(
                "{} bytes ({:.1} MB)",
                max_bytes,
                max_bytes as f64 / 1_000_000.0
            );
            anyhow::bail!(
                "Reference image {} is {} bytes and cannot be compressed under the {} cap \
                 after {} bounded steps. Use a smaller or simpler image.",
                path.display(),
                bytes.len(),
                cap,
                step,
            );
        }
        let out = with_block_in_place(|| compress_reference_step(&bytes, step))?;
        if out.len() as u64 <= max_bytes {
            return Ok(ReferenceImage {
                data_uri: format!("{JPEG_DATA_URI_PREFIX}{}", STANDARD.encode(&out)),
                source_bytes: bytes,
                next_step: step + 1,
                released: false,
            });
        }
        step += 1;
    }
}

/// Pre-flight combined-size check for multi-reference requests: sums metadata
/// lengths BEFORE any file is read, so a pathological total is refused without
/// loading the references into memory (the per-image ceiling does not bound
/// the sum).
pub(crate) async fn check_reference_total_input(paths: &[PathBuf]) -> anyhow::Result<()> {
    let mut total: u64 = 0;
    for path in paths {
        total += tokio::fs::metadata(path)
            .await
            .with_context(|| format!("Failed to access reference image {}", path.display()))?
            .len();
    }
    if total > MAX_TOTAL_REFERENCE_INPUT_BYTES {
        anyhow::bail!(
            "Combined reference images are limited to 100 MB, got {total} bytes total. \
             Use fewer or smaller images.",
        );
    }
    Ok(())
}

/// Reject non-PNG/JPEG/WebP extensions before reading the file.
fn check_reference_extension(path: &Path) -> anyhow::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg" | "webp")) {
        anyhow::bail!(
            "Reference image {}: unsupported format ({}). Only PNG, JPEG, or WebP \
             images are accepted.",
            path.display(),
            ext.as_deref().unwrap_or("unknown extension"),
        );
    }
    Ok(())
}

/// Content-sniff the actual image format (via the in-tree `image` crate),
/// accepting only PNG/JPEG/WebP.
fn sniff_reference_content(path: &Path, bytes: &[u8]) -> anyhow::Result<image::ImageFormat> {
    let format = image::guess_format(bytes).map_err(|_| {
        anyhow::anyhow!(
            "Reference image {}: content is not a decodable image (PNG/JPEG/WebP). \
             HEIC/HEIF and other unsupported formats are not accepted.",
            path.display(),
        )
    })?;
    match format {
        image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP => Ok(format),
        other => anyhow::bail!(
            "Reference image {}: unsupported image format ({other:?}). Only PNG, JPEG, \
             or WebP images are accepted.",
            path.display(),
        ),
    }
}

/// One bounded compression step: decode, optionally downscale, flatten alpha
/// onto white, and re-encode as JPEG at the ladder's quality.
fn compress_reference_step(bytes: &[u8], step: usize) -> anyhow::Result<Vec<u8>> {
    use image::GenericImageView;
    // Both call sites guarantee `step < REFERENCE_COMPRESSION_LADDER.len()`.
    let (scale, quality) = REFERENCE_COMPRESSION_LADDER[step];
    let mut img = image::load_from_memory(bytes).context("Failed to decode reference image")?;
    // EXIF orientation is metadata, not pixels: `load_from_memory` returns the
    // stored pixels as-is, and the JPEG encoder below starts from an empty EXIF
    // buffer — so over-cap phone JPEGs would otherwise be re-encoded silently
    // rotated from the user's intent. Apply the tag before downscaling so the
    // output dimensions reflect the true orientation.
    if let Some(orientation) = exif_orientation(bytes) {
        img.apply_orientation(orientation);
    }
    let img = if scale < 1.0 {
        let (w, h) = img.dimensions();
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let nw = (w as f32 * scale).round().max(1.0) as u32;
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let nh = (h as f32 * scale).round().max(1.0) as u32;
        img.resize(nw, nh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgb = flatten_alpha_onto_white(&img);
    let mut out = Vec::new();
    {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
        encoder
            .encode(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                image::ExtendedColorType::Rgb8,
            )
            .context("Failed to encode compressed reference image")?;
    }
    Ok(out)
}

/// Read the EXIF orientation tag from a raw image without decoding pixels
/// (header-only parse via the decoder's `orientation()`).
fn exif_orientation(bytes: &[u8]) -> Option<image::metadata::Orientation> {
    use image::ImageDecoder;
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    match decoder.orientation().ok()? {
        image::metadata::Orientation::NoTransforms => None,
        orientation => Some(orientation),
    }
}

/// Flatten any alpha channel onto a white background and return RGB pixels
/// (JPEG has no alpha channel; user uploads may carry transparency).
fn flatten_alpha_onto_white(img: &image::DynamicImage) -> image::RgbImage {
    // Opaque sources (JPEG, WebP-without-alpha) need no flattening — a single
    // copy instead of to_rgba8 + a second RgbImage pass.
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (x, y, px) in rgba.enumerate_pixels() {
        let [r, g, b, a] = px.0;
        let alpha = f32::from(a) / 255.0;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let blend = |c: u8| (f32::from(c) * alpha + 255.0 * (1.0 - alpha)).round() as u8;
        rgb.put_pixel(x, y, image::Rgb([blend(r), blend(g), blend(b)]));
    }
    rgb
}

/// Recognized video file extensions (single source of truth for inbound
/// Telegram routing and the video_edit local-clip guard).
pub(crate) const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "mkv", "avi", "webm"];

/// Video formats the transcription provider accepts by URL — intentionally
/// narrower than [`VIDEO_EXTENSIONS`] (which also admits mkv/avi for local
/// editing); unsupported formats silently fall back to the plain annotation.
/// The transcription path uploads with the extension-derived MIME, so every
/// whitelisted format is served with its real content type.
pub(crate) const TRANSCRIBABLE_VIDEO_EXTENSIONS: &[&str] = &["mp4", "mpeg", "mov", "webm"];

/// Recognized image file extensions for video_edit image inputs (reference
/// images and frame anchors), matching the provider-declared formats. It
/// matches the Telegram routing list (`telegram::IMAGE_EXTENSIONS`) for the
/// PNG/JPEG/WebP codecs (the only locally decodable ones) and additionally
/// admits heic/heif, which are passed through to the provider unmangled
/// (never locally decoded).
pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "heic", "heif"];

/// Daemon-owned subdir under the system temp dir where inbound Telegram
/// attachments are downloaded before enrichment. Enrichment only copies
/// video clips from here into workspace uploads — one of the containment
/// roots for the video-edit flow (the other being `generated/`).
pub(crate) const TELEGRAM_FILES_DIR: &str = "mahbot_telegram_files";

/// Check whether a path's extension (case-insensitive) belongs to `table`.
#[must_use]
pub(crate) fn has_extension(path: &std::path::Path, table: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| table.contains(&ext.to_ascii_lowercase().as_str()))
}

/// Check whether a file path has a recognized video extension.
#[must_use]
pub(crate) fn is_video_extension(path: &std::path::Path) -> bool {
    has_extension(path, VIDEO_EXTENSIONS)
}

/// Check whether a file path has a video extension the transcription provider
/// accepts (OpenRouter chat-completions video input).
#[must_use]
pub(crate) fn is_transcribable_video(path: &std::path::Path) -> bool {
    has_extension(path, TRANSCRIBABLE_VIDEO_EXTENSIONS)
}

/// Check whether a file path has a recognized image extension.
#[must_use]
pub(crate) fn is_image_extension(path: &std::path::Path) -> bool {
    has_extension(path, IMAGE_EXTENSIONS)
}

/// Check whether a string is an http(s) URL (case-sensitive prefix match).
#[must_use]
pub(crate) fn is_http_url(target: &str) -> bool {
    target.starts_with("http://") || target.starts_with("https://")
}

/// Map a file path's extension to a MIME type string.
pub(crate) fn mime_for_extension(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("heic") => "image/heic",
        Some("heif") => "image/heif",
        Some("mp4") => "video/mp4",
        Some("mpeg") => "video/mpeg",
        Some("mov") => "video/quicktime",
        Some("webm") => "video/webm",
        _ => "application/octet-stream",
    }
}

/// Extract the file name (last path component) from a path string, falling
/// back to the raw path when the path has no file name component.
#[must_use]
pub(crate) fn file_name_or_path(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

/// Strip ANSI escape sequences from a string.
///
/// Removes common ANSI escape codes used for terminal text formatting (colors,
/// bold, underline, cursor movement, etc.) while preserving the visible content.
/// This is useful when processing shell command output or any text that may
/// contain terminal control sequences.
#[must_use]
pub(crate) fn strip_ansi_escapes(input: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"\x1B\[[0-9;]*[a-zA-Z]|\x1B\][0-9;]*[^\x1B]*\x1B\\|\x1B[\(\)\[\]KM]|\x1B\][0-9;]*\x07",
        )
        .unwrap()
    });
    RE.replace_all(input, "").to_string()
}

/// Redact sensitive values for safe logging. Shows first 4 characters + "*[REDACTED]" suffix.
/// Uses char-boundary-safe indexing to avoid panics on multi-byte UTF-8 strings.
static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(token|api[_-]?key|password|secret|user[_-]?key|bearer|credential)["']?\s*[:=]\s*(?:"([^"]{8,})"|'([^']{8,})'|([a-zA-Z0-9_\-\./+=]{8,}))"#).expect("hardcoded regex is valid")
});

/// Whether `input` is "effectively empty" after trimming whitespace: ANSI
/// escapes are stripped first, and credential matches are removed outright
/// (not redacted), so ANSI-escape-only or fully credential-redacted output
/// counts as empty. Used for the command-armed alarm wake/no-wake decision.
#[must_use]
pub(crate) fn is_blank_after_redaction(input: &str) -> bool {
    let plain = strip_ansi_escapes(input);
    SENSITIVE_KV_REGEX.replace_all(&plain, "").trim().is_empty()
}

/// Scrub credentials from tool output to prevent accidental exfiltration.
/// Replaces known credential patterns with a redacted placeholder while preserving
/// a small prefix for context.
#[must_use]
pub(crate) fn scrub_credentials(input: &str) -> String {
    SENSITIVE_KV_REGEX
        .replace_all(input, |caps: &regex::Captures| {
            let full_match = &caps[0];
            let key = &caps[1];
            let val = caps
                .get(2)
                .or(caps.get(3))
                .or(caps.get(4))
                .map_or("", |m| m.as_str());

            // Preserve first 4 chars for context, then redact.
            debug_assert!(val.len() >= 8, "regex guarantees values >= 8 chars");
            let prefix = val
                .char_indices()
                .nth(4)
                .map_or(val, |(byte_idx, _)| &val[..byte_idx]);

            // Determine quote style from which capture group matched the value.
            // Group 2 = double-quoted, Group 3 = single-quoted, Group 4 (else) = unquoted.
            // Using capture-group identity avoids false positives from quotes/apostrophes
            // appearing elsewhere in the match (e.g., a double-quoted key name with a
            // single-quoted value, or an apostrophe in a key like `don't_share`).
            let quote = if caps.get(2).is_some() {
                Some('"')
            } else if caps.get(3).is_some() {
                Some('\'')
            } else {
                None
            };

            let redacted = format!("{prefix}*[REDACTED]");

            if full_match.contains(':') {
                match quote {
                    Some('"') => format!("\"{key}\": \"{redacted}\""),
                    Some('\'') => format!("{key}: '{redacted}'"),
                    _ => format!("{key}: {redacted}"),
                }
            } else {
                match quote {
                    Some('"') => format!("{key}=\"{redacted}\""),
                    Some('\'') => format!("{key}='{redacted}'"),
                    _ => format!("{key}={redacted}"),
                }
            }
        })
        .to_string()
}

/// Extract a provider error detail string from an error-body JSON value with
/// a conservative cascade over the envelope fields providers actually use.
///
/// Handles both the wrapped shape (`{"error": {...}}` / `{"error": "msg"}`)
/// and the bare error object (`{"code": ..., "message": ...}`), plus a bare
/// JSON string. Cascade:
/// 1. `message` — the standard field (the existing output contract: callers
///    that previously surfaced `error.message` keep their exact behavior).
/// 2. `metadata.raw` — OpenRouter forwards the upstream provider's raw error
///    body here when the top-level message is generic; the raw text is often
///    itself JSON, whose detail is preferred over the raw text.
/// 3. `metadata.provider_error_code` — the upstream provider's error code.
/// 4. `code`, then `type` — bare identifiers when nothing richer exists.
///
/// Used by the image-generation and web-search tool error paths (so the full
/// provider code/message reaches the model even in a nested envelope) and by
/// the input-image-rejection phrase builder.
#[must_use]
pub(crate) fn extract_provider_error_detail(error: &serde_json::Value) -> Option<String> {
    if let serde_json::Value::String(s) = error
        && !s.trim().is_empty()
    {
        return Some(s.clone());
    }

    // Unwrap a top-level `error` wrapper when present (the common envelope
    // shape across providers) — the string form is a complete message.
    match error.get("error") {
        Some(serde_json::Value::String(s)) if !s.trim().is_empty() => return Some(s.clone()),
        Some(inner) if !inner.is_null() => {
            if let Some(detail) = extract_provider_error_detail(inner) {
                return Some(detail);
            }
        }
        _ => {}
    }

    let text = |v: Option<&serde_json::Value>| -> Option<String> {
        v.and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };

    if let Some(msg) = text(error.get("message")) {
        return Some(msg);
    }
    if let Some(raw) = text(error.get("metadata").and_then(|m| m.get("raw"))) {
        if let Ok(raw_json) = serde_json::from_str::<serde_json::Value>(&raw)
            && let Some(detail) = extract_provider_error_detail(&raw_json)
        {
            return Some(detail);
        }
        return Some(raw);
    }
    if let Some(code) = text(
        error
            .get("metadata")
            .and_then(|m| m.get("provider_error_code")),
    ) {
        return Some(code);
    }
    if let Some(code) = text(error.get("code")) {
        return Some(code);
    }
    text(error.get("type"))
}

/// True when `path` can be executed: on Unix, a file with at least one
/// execute bit set (owner, group, or other — `PermissionsExt::mode() &
/// 0o111`); on Windows, a file with a `.exe` extension (Windows
/// executability is determined by extension and content, not permission
/// bits).
pub(crate) fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.is_file() && std::fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"))
    }
}

/// Resolve the cargo bin directory.
///
/// Resolution order:
/// 1. `$CARGO_HOME/bin` if `CARGO_HOME` environment variable is set and non-empty.
/// 2. `~/.cargo/bin` using `directories::UserDirs`.
/// 3. `None` — no home directory available.
#[must_use]
pub(crate) fn cargo_bin_dir() -> Option<PathBuf> {
    if let Ok(cargo_home) = std::env::var("CARGO_HOME")
        && !cargo_home.is_empty()
    {
        return Some(PathBuf::from(cargo_home).join("bin"));
    }

    let dirs = UserDirs::new()?;
    Some(dirs.home_dir().join(".cargo").join("bin"))
}

/// Strip surrounding double-quotes and unescape C-style escapes.
///
/// If the input starts with `"` and ends with `"`, strips the quotes and
/// calls `unescape_c_style` on the inner content. Otherwise returns the
/// input as-is (no unescaping needed — git only C-quotes paths that contain
/// trigger characters).
///
/// This is the standard pattern for handling git's quoted path output
/// (the same approach as git's own `unquote_c_style`).
#[must_use]
pub(crate) fn unquote_c_style(raw: &str) -> Option<String> {
    if let Some(inner) = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        unescape_c_style(inner)
    } else {
        Some(raw.to_string())
    }
}

/// Unescape C-style escape sequences from a git path name.
///
/// Supports the same escapes as git's `unquote_c_style`:
/// - `\"` → literal `"`, `\\` → literal `\`
/// - `\t` → tab, `\n` → newline, `\a` → bell, `\b` → backspace
/// - `\f` → form feed, `\r` → carriage return, `\v` → vertical tab
/// - `\0`–`\3` followed by 1–3 octal digits → byte value
///
/// Malformed escapes cause this function to return `None`:
/// - `\` at end of string (dangling backslash)
/// - `\x` or any other unrecognized escape letter
/// - `\4`–`\7` followed by a digit (git rejects these as invalid octal prefixes)
///
/// Non-UTF-8 bytes produced by octal escapes are handled via
/// `String::from_utf8_lossy` — pragmatic for macOS where non-UTF-8 paths
/// are filesystem-impossible.
fn unescape_c_style(input: &str) -> Option<String> {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i: usize = 0;

    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 1; // consume backslash
            if i >= bytes.len() {
                tracing::warn!(
                    input = %input,
                    "unescape_c_style: dangling backslash at end of string"
                );
                return None;
            }
            match bytes[i] {
                b'"' => result.push('"'),
                b'\\' => result.push('\\'),
                b't' => result.push('\t'),
                b'n' => result.push('\n'),
                b'a' => result.push('\x07'),
                b'b' => result.push('\x08'),
                b'f' => result.push('\x0c'),
                b'r' => result.push('\r'),
                b'v' => result.push('\x0b'),
                b'0'..=b'3' => {
                    // Octal escape: 1–3 octal digits. The 0–7 range check
                    // leaves '8'/'9' unconsumed so they emit as literal digits
                    // on the next outer iteration.
                    let digits_start = i;
                    i += 1;
                    let mut digit_count = 1;
                    while digit_count < 3 && i < bytes.len() && (b'0'..=b'7').contains(&bytes[i]) {
                        i += 1;
                        digit_count += 1;
                    }
                    let octal_str = std::str::from_utf8(&bytes[digits_start..i]).ok()?;
                    let Ok(byte_val) = u8::from_str_radix(octal_str, 8) else {
                        tracing::warn!(
                            input = %input, octal = %octal_str,
                            "unescape_c_style: invalid octal escape"
                        );
                        return None;
                    };
                    result.push_str(&String::from_utf8_lossy(&[byte_val]));
                    continue; // skip the i += 1 at end of loop
                }
                b'4'..=b'7' => {
                    // \4–\7 are not valid octal prefixes in git's unquote_c_style.
                    // If followed by a digit, it's a malformed octal attempt.
                    if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                        tracing::warn!(
                            input = %input,
                            ch = %(bytes[i] as char),
                            "unescape_c_style: invalid octal prefix \\4–\\7 followed by digit"
                        );
                        return None;
                    }
                    // Otherwise: literal digit (backslash consumed, no special meaning).
                    result.push(bytes[i] as char);
                }
                _ => {
                    tracing::warn!(
                        input = %input,
                        ch = %(bytes[i] as char),
                        "unescape_c_style: unrecognized escape sequence"
                    );
                    return None;
                }
            }
        } else {
            result.push(bytes[i] as char);
        }
        i += 1;
    }

    Some(result)
}

/// Resample PCM audio from one sample rate to another using linear
/// interpolation with a 3-tap binomial anti-aliasing filter for downsampling.
///
/// This is the canonical implementation. All other resample call sites
/// delegate to this one, except `audio::local_transcriber`, which uses
/// `qwen_asr::audio::resample` directly.
///
/// When `from_rate > to_rate` (downsampling), a simple binomial low-pass filter
/// is applied to attenuate frequencies above the new Nyquist before decimation.
/// Without this filter, linear interpolation introduces aliasing — high-frequency
/// content above `to_rate / 2` folds back into the audible range as noise.
///
/// The 3-tap binomial `[0.25, 0.5, 0.25]` gives reasonable stopband attenuation
/// (~6 dB at 0.25 normalised) for speech audio. For 48 kHz → 16 kHz this
/// attenuates content above ~8 kHz.
///
/// # Aliasing trade-off
///
/// Linear interpolation introduces aliasing when downsampling even with the
/// pre-filter — the filter only provides ~6 dB stopband attenuation. This is
/// acceptable for speech processing and wake word training data augmentation.
/// If aliasing artifacts prove problematic, a sinc-based resampler can be
/// substituted here — all call sites except `local_transcriber` (which
/// resamples via `qwen_asr` itself) benefit automatically.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;

    // Anti-aliasing filter for downsampling
    let filtered: Vec<f32> = if from_rate > to_rate && samples.len() >= 3 {
        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0] * 0.75 + samples[1] * 0.25);
        for i in 1..samples.len() - 1 {
            out.push(samples[i - 1] * 0.25 + samples[i] * 0.5 + samples[i + 1] * 0.25);
        }
        out.push(samples[samples.len() - 2] * 0.25 + samples[samples.len() - 1] * 0.75);
        out
    } else {
        samples.to_vec()
    };

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < filtered.len() {
            output.push(
                (f64::from(filtered[src_idx]) * (1.0 - frac)
                    + f64::from(filtered[src_idx + 1]) * frac) as f32,
            );
        } else if src_idx < filtered.len() {
            output.push(filtered[src_idx]);
        } else {
            output.push(0.0);
        }
    }
    output
}

#[cfg(test)]
mod media_mime_tests {
    use super::*;

    // A mislabelled file (photo.png holding JPEG bytes) must produce a data-URI
    // whose MIME matches the actual bytes, so the decoder's declared-subtype
    // check accepts it instead of silently dropping the image downstream.
    #[tokio::test]
    async fn data_uri_mime_follows_bytes_not_extension() {
        let img = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        let mut jpeg = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .expect("test JPEG must encode");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.png");
        tokio::fs::write(&path, &jpeg).await.unwrap();

        let uri = local_image_to_data_uri(&path).await.unwrap();
        assert!(
            uri.starts_with("data:image/jpeg;base64,"),
            "MIME must come from the actual bytes; ext says png: {uri}"
        );
        assert!(
            uri.ends_with(&STANDARD.encode(&jpeg)),
            "payload bytes must pass through unchanged"
        );
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::*;

    // ── truncate_sandwich: passthrough ────────────────────────────────────

    #[test]
    fn passthrough_under_limit() {
        let input = "hello world";
        let result = truncate_sandwich(input, 5_000, "test");
        assert_eq!(
            result, input,
            "should pass through unchanged when under limit"
        );
    }

    #[test]
    fn passthrough_at_exact_limit() {
        let input = "a".repeat(5_000);
        assert_eq!(input.len(), 5_000);
        let result = truncate_sandwich(&input, 5_000, "test");
        assert_eq!(result, input, "exact limit should pass through unchanged");
    }

    // ── truncate_sandwich: head/tail sandwich ─────────────────────────────

    #[test]
    fn sandwich_just_over_limit() {
        // Input is exactly limit+1 bytes — sandwich marker may add overhead
        // making output longer than input, which is expected for tiny overshoot.
        let input = "x".repeat(5_001);
        let result = truncate_sandwich(&input, 5_000, "test");
        assert!(
            result.starts_with("xxx"),
            "head portion should be preserved"
        );
        assert!(
            result.contains("bytes omitted at test truncation"),
            "should contain the omission marker"
        );
        assert!(result.ends_with('x'), "tail should contain input suffix");
    }

    #[test]
    fn sandwich_large_input() {
        // Input well over the limit — classic head/tail sandwich with label
        let line = "hello world\n".repeat(200_000);
        assert!(line.len() > 1_048_576, "input should exceed 1MB");
        let result = truncate_sandwich(&line, 1_048_576, "output");
        assert!(result.len() < line.len(), "should truncate");
        assert!(
            result.contains("bytes omitted at output truncation"),
            "should contain label in omission marker"
        );
        // Head portion appears
        assert!(
            result.starts_with("hello world"),
            "head should be preserved"
        );
        // Tail portion appears
        let last_line = result.lines().last().unwrap_or("");
        assert_eq!(last_line, "hello world", "tail should be preserved");
    }

    #[test]
    fn sandwich_preserves_utf8_boundaries() {
        // Place a multibyte char ('🐱', 4 bytes) right at the head/tail
        // boundary so it straddles the cut point. floor_char_boundary must
        // back up to the character boundary. Build: 3329 'x's, then 🐱
        // (bytes 3329-3332), then 'y's. head_bytes ≈ 3333, so 🐱 is the
        // last complete char in head. Verify it appears intact.
        let mut input = String::new();
        input.push_str(&"x".repeat(3_329));
        input.push('🐱'); // bytes 3329..=3332
        input.push_str(&"y".repeat(20_000));
        let result = truncate_sandwich(&input, 5_000, "test");
        assert!(
            result.contains('🐱'),
            "multibyte char at boundary should survive intact"
        );
    }

    #[test]
    fn sandwich_line_boundaries_intact() {
        // Lines should not be concatenated across truncation boundaries
        let line = "hello world!\n".repeat(100_000);
        let result = truncate_sandwich(&line, 500_000, "test");
        assert!(result.len() < line.len(), "should truncate");
        for l in result.lines().filter(|l| !l.starts_with("...")) {
            assert!(
                !l.contains("hello world!hello"),
                "lines should not be concatenated"
            );
        }
    }

    // ── truncate_sandwich: custom label ───────────────────────────────────

    #[test]
    fn custom_label_appears_in_marker() {
        let input = "x".repeat(10_000);
        let result = truncate_sandwich(&input, 5_000, "my custom label");
        assert!(
            result.contains("bytes omitted at my custom label truncation"),
            "custom label should appear verbatim in marker"
        );
    }

    #[test]
    fn empty_label() {
        let input = "x".repeat(10_000);
        let result = truncate_sandwich(&input, 5_000, "");
        assert!(
            result.contains("bytes omitted at  truncation"),
            "empty label should still produce coherent marker"
        );
    }

    // ── truncate_tool_output compatibility ──────────────────────────────────

    #[test]
    fn truncate_tool_output_appends_correct_label() {
        let input = "abc".repeat(2_000); // 6_000 bytes > 5_000 limit
        let result = truncate_tool_output(&input);
        assert!(result.len() < input.len(), "should truncate");
        assert!(
            result.contains("bytes omitted at tool output truncation"),
            "should use 'tool output' label"
        );
        assert!(result.starts_with("abcabc"), "head should be preserved");
    }
}

// ── scrub_credentials tests ────────────────────────────────────────────

#[cfg(test)]
mod scrub_tests {
    use super::scrub_credentials;

    #[test]
    fn scrub_redacts_credentials() {
        /// Cases verifying `[REDACTED]` appears, with optional negative and
        /// prefix checks. Fields: (name, input, must_not_contain, must_start_with).
        /// Empty string for must_not_contain/must_start_with = skip check.
        const CASES: &[(&str, &str, &str, &str)] = &[
            (
                "alphanumeric unquoted value",
                "API_KEY=sk-1234567890abcdef",
                "1234567890abcdef",
                "API_KEY=sk-1",
            ),
            // Standard Base64-encoded secret containing +, /, =
            (
                "Base64 unquoted value with plus and slash",
                "api_key=u2FsdGVkX1+h/wZ/L3Y+Q==",
                "u2FsdGVkX1+h/wZ/L3Y+Q==",
                "api_key=u2Fs",
            ),
            (
                "double-quoted value with colon separator",
                r#"token: "abcdefgh1234567890""#,
                "1234567890",
                "",
            ),
            (
                "bearer colon-separated value",
                "bearer: eyJhbGciOiJIUzI1NiJ9",
                "eyJhbG",
                "",
            ),
            // Hyphen-key variant: regex `user[_-]?key` also matches `user-key`.
            (
                "hyphen-key variant",
                "user-key=abcdefgh12345678",
                "12345678",
                "user-key=abcd",
            ),
        ];

        for &(name, input, not_contains, prefix) in CASES {
            let out = scrub_credentials(input);
            assert!(out.contains("[REDACTED]"), "{name}: should redact: {out}");
            if !not_contains.is_empty() {
                assert!(
                    !out.contains(not_contains),
                    "{name}: should not leak value: {out}"
                );
            }
            if !prefix.is_empty() {
                assert!(out.starts_with(prefix), "{name}: should keep prefix: {out}");
            }
        }
    }

    #[test]
    fn scrub_exact_output() {
        /// Cases verifying exact output strings. Exact match is the strictest
        /// assertion — it subsumes containment and non-leakage checks.
        /// Fields: (name, input, expected_output).
        const CASES: &[(&str, &str, &str)] = &[
            // Single quotes must be preserved (the bug this test guards against).
            (
                "single-quoted value with colon separator",
                "password: 's3cr3t_p@ssw0rd!!'",
                "password: 's3cr*[REDACTED]'",
            ),
            (
                "single-quoted value with equals separator",
                "password='mysecretvalue123'",
                "password='myse*[REDACTED]'",
            ),
            // Edge case: the key-level optional quote in the regex can produce
            // full_match containing a double-quote from the key suffix, e.g.
            // "password": 'secretvalue1234'. The capture-group approach correctly
            // identifies this as a single-quoted value despite the double-quote
            // appearing in the full match string.
            // Note: the key-suffix " is consumed by the regex match and not
            // reconstructed — this is a pre-existing cosmetic issue also present
            // in the double-quote path, and out of scope for this fix.
            (
                "double-quoted key with single-quoted value",
                r#""password": 'secretvalue123'"#,
                "\"password: 'secr*[REDACTED]'",
            ),
        ];

        for &(name, input, expected) in CASES {
            assert_eq!(scrub_credentials(input), expected, "{name}");
        }
    }

    #[test]
    fn scrub_passthrough() {
        /// Cases where the input is not a credential pattern and must pass
        /// through unchanged. Fields: (name, input).
        const CASES: &[(&str, &str)] = &[
            ("short unquoted values (under 8 chars)", "key=short"),
            (
                "non-secret lines with = and /",
                "normal line with = equals and / slash",
            ),
        ];

        for &(name, input) in CASES {
            assert_eq!(scrub_credentials(input), input, "{name}");
        }
    }

    #[test]
    fn blank_after_redaction_treats_credential_only_output_as_empty() {
        use super::is_blank_after_redaction;
        // Split literals so this source does not itself look like a credential
        // to output scrubbers.
        let credential = format!("api_{}={:?}", "key", "supersecretvalue123");
        assert!(is_blank_after_redaction(""));
        assert!(is_blank_after_redaction("   \n\t "));
        // ANSI-escape-only output.
        assert!(is_blank_after_redaction("\x1B[32m\x1B[0m"));
        // Output that consists only of a credential counts as empty.
        assert!(is_blank_after_redaction(&credential));
        // But real content does not.
        assert!(!is_blank_after_redaction(&format!(
            "build ok\n{credential}"
        )));
    }
}

// ── extract_provider_error_detail tests ─────────────────────────────────

#[cfg(test)]
mod extract_provider_error_detail_tests {
    use super::extract_provider_error_detail;
    use serde_json::json;

    #[test]
    fn standard_message_field() {
        // Wrapped envelope — the common shape.
        let body = json!({"error": {"message": "upstream busy"}});
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("upstream busy")
        );
        // Bare object without the `error` wrapper.
        let body = json!({"message": "bare detail"});
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("bare detail")
        );
    }

    #[test]
    fn string_error_field() {
        let body = json!({"error": "plain message"});
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("plain message")
        );
    }

    #[test]
    fn nested_envelope_raw_field() {
        let body = json!({
            "error": {
                "code": "data_inspection_failed",
                "metadata": {"raw": "Input image data may contain inappropriate content."}
            }
        });
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("Input image data may contain inappropriate content.")
        );
    }

    #[test]
    fn nested_raw_is_itself_json_preferred() {
        let body = json!({
            "error": {
                "message": "generic",
                "metadata": {"raw": r#"{"error":{"message":"deep upstream detail"}}"#}
            }
        });
        // `message` exists — it wins (existing output contract).
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("generic")
        );

        // Without a top-level message, the nested raw JSON's detail wins.
        let body = json!({
            "error": {
                "metadata": {"raw": r#"{"error":{"message":"deep upstream detail"}}"#}
            }
        });
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("deep upstream detail")
        );
    }

    #[test]
    fn provider_error_code_field() {
        let body = json!({
            "error": {
                "metadata": {"provider_error_code": "upstream_shared_pool"}
            }
        });
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("upstream_shared_pool")
        );
    }

    #[test]
    fn bare_code_then_type_fallbacks() {
        let body = json!({"error": {"code": "invalid_request_error"}});
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("invalid_request_error")
        );
        let body = json!({"error": {"type": "rate_limit_exceeded"}});
        assert_eq!(
            extract_provider_error_detail(&body).as_deref(),
            Some("rate_limit_exceeded")
        );
    }

    #[test]
    fn empty_and_absent_fields_yield_none() {
        assert_eq!(extract_provider_error_detail(&json!({})), None);
        assert_eq!(extract_provider_error_detail(&json!({"error": null})), None);
        assert_eq!(extract_provider_error_detail(&json!({"error": {}})), None);
        // A blank message is skipped, not surfaced.
        assert_eq!(
            extract_provider_error_detail(&json!({"error": {"message": "  "}})),
            None
        );
    }
}

#[cfg(test)]
mod unescape_c_style_tests {
    use super::unescape_c_style;

    #[test]
    fn test_unescape_c_style() {
        // Cases: (input, expected_output).
        // Uses Option<&str> — compared via .as_deref() against the
        // function's Option<String> return type.
        let cases: &[(&str, Option<&str>)] = &[
            // ── basic escapes ──
            (
                r#"hello\"world\\test\nline\there"#,
                Some("hello\"world\\test\nline\there"),
            ),
            // Bell, backspace, formfeed, CR, vertical tab.
            (r"\a\b\f\r\v", Some("\x07\x08\x0c\r\x0b")),
            // ── octal escapes, 1–3 digits ──
            // \0 → NUL (0x00), \1 → SOH (0x01)
            (r"\0\1", Some("\0\x01")),
            // \12 → newline (0x0a), \37 → unit separator (0x1f)
            (r"\12\37", Some("\n\x1f")),
            // \101 → 'A' (0x41), \377 → 0xff → U+FFFD (from_utf8_lossy replacement)
            (r"\101\377", Some("A\u{FFFD}")),
            // Octal stops at non-octal-digit: \12x → newline + 'x'
            (r"\12x", Some("\nx")),
            // \18 → \1 (SOH, 0x01) then '8' (8 is not an octal digit)
            (r"\18", Some("\x018")),
            // ── no-op cases (no escape sequences) ──
            ("plain/path.rs", Some("plain/path.rs")),
            ("", Some("")),
            // ── error cases (return None) ──
            // Dangling backslash at end of string.
            (r"path\", None),
            // \x looks like a hex escape prefix but git's unquote_c_style rejects it.
            (r"\x", None),
            // \q is not a recognized escape sequence.
            (r"\q", None),
            // \40 — \4 followed by digit (git rejects as invalid octal prefix).
            (r"\40", None),
            // \77 — \7 followed by digit (git rejects as invalid octal prefix).
            (r"\77", None),
            // \70 — \7 followed by digit (git rejects as invalid octal prefix).
            (r"\70", None),
            // ── literal-digit cases (\4/\7 not followed by digit) ──
            // \4 at end of string → literal '4'.
            (r"\4", Some("4")),
            // \7 followed by non-digit → literal '7' then 'x'.
            (r"\7x", Some("7x")),
        ];
        for (i, (input, expected)) in cases.iter().enumerate() {
            let result = unescape_c_style(input);
            assert_eq!(
                result.as_deref(),
                *expected,
                "case {i}: unescape_c_style({input:?})"
            );
        }
    }
}

// Audio utility functions (canonical implementations)

/// Generate pink noise (1/f spectrum) using the Voss-McCartney algorithm.
///
/// Uses a seeded RNG for reproducibility.  The high-pass delta variant
/// naturally removes DC bias.  Output is normalized to unit RMS.
///
/// # Voss-McCartney variants
///
/// There are several common variants of Voss-McCartney pink noise:
///
/// * **High-pass delta** (this implementation): stores previous value per
///   octave, emits the difference (new - prev).  Removes DC bias naturally.
/// * **Direct-sum**: sums all octave values directly.  May accumulate DC bias.
///
/// The canonical implementation uses 16 octaves (~3 dB/octave rolloff down
/// to 0.03 Hz at 16 kHz) and the high-pass delta variant for DC-free output.
///
/// Used by the voice-pipeline benchmark (`voice-tests`) and default-build unit tests.
#[cfg(any(test, feature = "voice-tests"))]
pub(crate) fn generate_pink_noise(len: usize, mut rng: impl rand::Rng) -> Vec<f32> {
    const NUM_OCTAVES: usize = 16;
    let mut values = [0.0f32; NUM_OCTAVES];
    let mut outputs = [0.0f32; NUM_OCTAVES];
    let mut sample_count = 0u64;
    let mut noise = Vec::with_capacity(len);

    for _ in 0..len {
        sample_count += 1;
        let mut sum = 0.0;
        for octave in 0..NUM_OCTAVES {
            // Update this octave's generator at intervals of 2^octave samples.
            if sample_count.is_multiple_of(1u64 << octave) {
                values[octave] = rng.random::<f32>() * 2.0 - 1.0;
            }
            // High-pass delta: emit difference instead of direct value.
            let new_val = values[octave];
            let delta = new_val - outputs[octave];
            outputs[octave] = new_val;
            sum += delta;
        }
        noise.push(sum);
    }

    // Normalize to unit RMS
    let rms = compute_rms(&noise).max(1e-10);
    for s in &mut noise {
        *s /= rms;
    }

    noise
}

/// Pink-noise alias of [`add_noise_color`] — kept for the recipe's variant-4
/// call sites; the SNR-scaling/clamp arithmetic lives in one place.
/// Used by the voice-pipeline bench (`voice-tests`) and default-build unit tests.
#[cfg(any(test, feature = "voice-tests"))]
pub(crate) fn add_noise(pcm: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    add_noise_color(pcm, snr_db, NoiseColor::Pink, seed)
}

/// Apply a fixed gain to PCM audio.
///
/// DETERMINISTIC — no RNG involved. The gain is `10^(gain_db / 20)`.
/// Negative values attenuate, positive values amplify.
/// Used by the voice-pipeline bench (`voice-tests`) and default-build unit tests.
#[cfg(any(test, feature = "voice-tests"))]
pub(crate) fn apply_gain(pcm: &[f32], gain_db: f32) -> Vec<f32> {
    let amp = 10.0_f32.powf(gain_db / 20.0);
    pcm.iter().map(|&s| s * amp).collect()
}

/// Noise colors supported by the noise-mixing helpers (voice-tests bench and
/// default-build unit tests).
#[cfg(any(test, feature = "voice-tests"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoiseColor {
    Pink,
    // Constructed only by the voice-tests bench augmentation; the default
    // test build exercises Pink only.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    Brown,
}

/// Add color noise to PCM audio at the given SNR (deterministic, seeded).
///
/// Mirrors [`add_noise`]'s arithmetic (unit-RMS noise scaled to the SNR
/// target, clamped mix) with the color selector.  `Brown` is a leaky
/// integration of white noise (DC-free-ish, low-frequency dominant).
/// Used by the voice-pipeline bench (`voice-tests`) and default-build unit tests.
#[cfg(any(test, feature = "voice-tests"))]
pub(crate) fn add_noise_color(pcm: &[f32], snr_db: f32, color: NoiseColor, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let signal_rms = compute_rms(pcm).max(1e-10);
    let noise: Vec<f32> = match color {
        NoiseColor::Pink => generate_pink_noise(pcm.len(), &mut rng),
        NoiseColor::Brown => {
            let mut acc = 0.0f32;
            (0..pcm.len())
                .map(|_| {
                    acc = 0.999 * acc + (rng.random::<f32>() * 2.0 - 1.0);
                    acc
                })
                .collect()
        }
    };
    let noise_rms_target = signal_rms * 10.0_f32.powf(-snr_db / 20.0);
    let noise_rms_current = compute_rms(&noise).max(1e-10);
    let scale = noise_rms_target / noise_rms_current;
    pcm.iter()
        .zip(noise.iter())
        .map(|(&s, &n)| (s + n * scale).clamp(-1.0, 1.0))
        .collect()
}

/// Compute the RMS (root mean square) of audio samples.
///
/// Returns `0.0` for empty input.  Ungated so production hot
/// paths (AGC, noise generation, utterance quality) share one implementation
/// instead of hand-rolling `sum(x²)/n → sqrt`.  Callers that need a
/// divide-by-zero floor for degenerate all-zero input apply `.max(1e-10)` at
/// the call site — it is deliberately NOT part of this function.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute both Box-Muller branches from two uniform draws in `(0, 1]`.
///
/// `z1 = sqrt(-2 ln u1) cos(2π u2)`, `z2 = sqrt(-2 ln u1) sin(2π u2)`.
/// Shared math for the bench's EPSILON-clamp pair sampler below.
#[must_use]
#[inline]
pub(crate) fn gaussian_pair_from_uniforms(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Draw a standard-normal pair via Box-Muller with EPSILON-clamped draws
/// (2 draws per 2 samples).  Preserves the bench's draw sequence and
/// degenerate-input semantics (`u1`/`u2` floored at [`f32::EPSILON`] rather
/// than re-rolled).  Used by the `voice-tests` bench and the seeded-sequence
/// equivalence test.
#[cfg_attr(not(any(feature = "voice-tests", test)), allow(dead_code))]
pub(crate) fn sample_gaussian_pair_clamped(rng: &mut impl rand::Rng) -> (f32, f32) {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
    gaussian_pair_from_uniforms(u1, u2)
}

/// Apply speed perturbation by resampling.
///
/// Changes both speed and pitch (time-domain resampling). For wake word
/// training data diversity, this is acceptable and does not require
/// pitch-preserving time-stretching.
///
/// # Arguments
///
/// * `samples` — Audio PCM f32 at `sample_rate`.
/// * `sample_rate` — Original sample rate in Hz.
/// * `factor` — Speed factor: >1.0 = faster (fewer samples), <1.0 = slower
///   (more samples). Typical range: 0.8-1.2 (±20%).
///
/// # Returns
///
/// Speed-adjusted audio at the original `sample_rate`.
/// Used by the voice-pipeline bench (`voice-tests`) and default-build unit tests.
#[must_use]
#[cfg(any(test, feature = "voice-tests"))]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn speed_perturbation(samples: &[f32], sample_rate: u32, factor: f32) -> Vec<f32> {
    if samples.is_empty() || (factor - 1.0).abs() < 1e-6 {
        return samples.to_vec();
    }
    // Speed perturbation via resampling: change the effective rate
    // new_rate = sample_rate * factor
    let effective_rate = (sample_rate as f32 * factor) as u32;
    // Resample from effective_rate back to sample_rate
    // This produces the same duration as original but with shifted pitch
    resample_audio(samples, effective_rate, sample_rate)
}

#[cfg(test)]
mod strip_ansi_escapes_tests {
    use super::strip_ansi_escapes;

    #[test]
    fn test_ansi_escape_cases() {
        let cases: &[(&str, &str)] = &[
            ("\x1B[31mred\x1B[0m \x1B[1mbold\x1B[22m", "red bold"),
            ("hello world", "hello world"),
            ("\x1B[32mgreen\x1B[0m", "green"),
            ("no escapes here", "no escapes here"),
            ("", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(strip_ansi_escapes(input), *expected, "input: {input:?}");
        }
    }
}

// ── Shared model-integrity verifier ─────────────────────────
// Tests consolidated from the three former private copies (tts,
// local_transcriber, voice).  The file-not-found error path is covered with a
// NON-empty expected hash because the shared empty-hash skip returns Ok before
// opening the file (sanctioned reconciliation of the transcriber's original
// `verify_sha256(path, "").is_err()` assertion).

#[cfg(test)]
mod verify_sha256_tests {
    use super::verify_sha256;

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        super::hex_string(&hasher.finalize())
    }

    #[test]
    fn verify_sha256_cases() {
        // Matching and mismatching hashes over a written file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"test data").unwrap();
        let hash = sha256_hex(b"test data");
        assert!(
            verify_sha256(&path, &hash).is_ok(),
            "matching hash should pass"
        );
        assert!(
            verify_sha256(&path, &sha256_hex(b"other data")).is_err(),
            "mismatching hash should fail"
        );

        // Empty expected hash → Ok without opening the file (skip semantics).
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.bin");
        assert!(
            verify_sha256(&missing, "").is_ok(),
            "empty hash should skip verification"
        );
        // The not-found error path requires a non-empty expected hash.
        assert!(
            verify_sha256(&missing, &sha256_hex(b"anything")).is_err(),
            "missing file with non-empty hash should fail"
        );
    }
}

// ── Audio utility regression tests ──────────────────────────────────────────
// Moved verbatim from voice.rs (post-1029 stranded util tests) so the util
// module has an in-place regression net for speed perturbation, gain, noise,
// and pink noise.  The utilities are gated `any(test, voice-tests)`, so the
// tests run in the default build.

#[cfg(test)]
mod audio_util_tests {
    use super::{add_noise, apply_gain, generate_pink_noise, speed_perturbation};
    use rand::SeedableRng;

    #[test]
    fn test_speed_perturbation_identity() {
        // rate=1.0 should return approximately the original
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..100 — exact in f32
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let result = speed_perturbation(&pcm, 16000, 1.0);
        assert_eq!(result.len(), pcm.len(), "identity should preserve length");
        for (a, b) in pcm.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5, "identity should preserve values");
        }
    }

    #[test]
    fn test_speed_perturbation_rates() {
        // Slow down: rate=0.5 should produce more samples
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..100 — exact in f32
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let slowed = speed_perturbation(&pcm, 16000, 0.5);
        assert!(
            slowed.len() > pcm.len(),
            "rate < 1 should increase sample count"
        );
        // Speed up: rate=2.0 should produce fewer samples
        let sped_up = speed_perturbation(&pcm, 16000, 2.0);
        assert!(
            sped_up.len() < pcm.len(),
            "rate > 1 should decrease sample count"
        );
    }

    #[test]
    fn test_speed_perturbation_determinism() {
        // Same input + same rate → same output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..100 — exact in f32
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let a = speed_perturbation(&pcm, 16000, 0.95);
        let b = speed_perturbation(&pcm, 16000, 0.95);
        assert_eq!(a, b, "deterministic speed perturbation");
    }

    #[test]
    fn test_apply_gain_determinism() {
        // apply_gain must be deterministic: same PCM + same gain_db → same output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..50 — exact in f32
        let pcm: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) / 25.0).collect();
        let a = apply_gain(&pcm, -3.0);
        let b = apply_gain(&pcm, -3.0);
        assert_eq!(a, b, "apply_gain must be deterministic");
    }

    #[test]
    fn test_apply_gain_db_conversion() {
        // 0 dB → unity gain (output == input)
        let pcm: Vec<f32> = vec![0.5, -0.3, 0.1, -0.7, 0.9];
        let unity = apply_gain(&pcm, 0.0);
        for (a, b) in pcm.iter().zip(unity.iter()) {
            assert!((a - b).abs() < 1e-6, "0 dB gain should be unity");
        }
        // -6 dB → amplitude halved (10^(-6/20) ≈ 0.5)
        let attenuated = apply_gain(&pcm, -6.0);
        let expected_amp = 10.0_f32.powf(-6.0 / 20.0);
        for (orig, atten) in pcm.iter().zip(attenuated.iter()) {
            assert!(
                (atten - orig * expected_amp).abs() < 1e-6,
                "-6 dB gain should multiply by {expected_amp}"
            );
        }
        // +6 dB → amplitude doubled (10^(6/20) ≈ 2.0)
        let amplified = apply_gain(&pcm, 6.0);
        let expected_amp2 = 10.0_f32.powf(6.0 / 20.0);
        for (orig, amp) in pcm.iter().zip(amplified.iter()) {
            assert!(
                (amp - orig * expected_amp2).abs() < 1e-6,
                "+6 dB gain should multiply by {expected_amp2}"
            );
        }
    }

    #[test]
    fn test_add_noise_determinism() {
        // Same PCM + same SNR + same seed → same output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..200 — exact in f32
        let pcm: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 100.0).collect();
        let a = add_noise(&pcm, 25.0, 42);
        let b = add_noise(&pcm, 25.0, 42);
        assert_eq!(a.len(), b.len(), "add_noise output length should match");
        assert_eq!(a, b, "add_noise with same seed must be deterministic");
    }

    #[test]
    fn test_add_noise_seed_variation() {
        // Different seed → different output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..200 — exact in f32
        let pcm: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 100.0).collect();
        let a = add_noise(&pcm, 25.0, 42);
        let b = add_noise(&pcm, 25.0, 99);
        assert_ne!(a, b, "different seeds should produce different output");
    }

    #[test]
    fn test_add_noise_snr_approximation() {
        // At very high SNR (100 dB), the output should be very close to input
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..1000 — exact in f32
        let pcm: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();
        let noisy = add_noise(&pcm, 100.0, 42);
        // Compute actual SNR of output vs input
        let signal_power: f32 = pcm.iter().map(|&s| s * s).sum();
        let noise_power: f32 = pcm
            .iter()
            .zip(noisy.iter())
            .map(|(&s, &n)| (n - s) * (n - s))
            .sum();
        let actual_snr_db = 10.0 * (signal_power / noise_power.max(1e-10)).log10();
        assert!(
            actual_snr_db > 80.0,
            "at 100 dB target, actual SNR should be high (was {actual_snr_db:.1} dB)"
        );
    }

    #[test]
    fn test_generate_pink_noise_properties() {
        // Pink noise should have positive and negative values
        let noise = generate_pink_noise(1000, rand::rngs::StdRng::seed_from_u64(42));
        assert_eq!(noise.len(), 1000);
        let has_positive = noise.iter().any(|&s| s > 0.0);
        let has_negative = noise.iter().any(|&s| s < 0.0);
        assert!(has_positive, "pink noise should have positive values");
        assert!(has_negative, "pink noise should have negative values");
        // RMS should be near 1.0 (normalized)
        let rms = super::compute_rms(&noise);
        assert!(
            (rms - 1.0).abs() < 0.1,
            "pink noise RMS should be near 1.0 (was {rms})"
        );
    }

    #[test]
    fn test_augmentation_variants_are_different() {
        // Verify that the 4 augmentation strategies produce different results
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..500 — exact in f32
        let pcm: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) / 250.0).collect();
        let speed_down = speed_perturbation(&pcm, 16000, 0.95);
        let speed_up = speed_perturbation(&pcm, 16000, 1.05);
        let volume_down = apply_gain(&pcm, -3.0);
        let noise = add_noise(&pcm, 25.0, 42);

        // Each variant should differ from original AND each other
        assert_ne!(speed_down, pcm, "speed-down should differ from original");
        assert_ne!(speed_up, pcm, "speed-up should differ from original");
        assert_ne!(volume_down, pcm, "volume-down should differ from original");
        assert_ne!(noise, pcm, "noise should differ from original");

        // Verify they also differ from each other (different transforms)
        assert_ne!(
            speed_down, speed_up,
            "speed-down and speed-up should differ"
        );
        assert_ne!(
            speed_down, volume_down,
            "speed-down and volume-down should differ"
        );
        assert_ne!(speed_down, noise, "speed-down and noise should differ");
    }
}

// ── Shared Gaussian sampler seeded-sequence equivalence ─────
// MANDATORY verification from the manager pin: the extraction must consume
// exactly the same RNG draws at every site so seeded outputs stay byte-identical.
// These tests prove the shared helper reproduces the pre-extraction inline
// EPSILON-clamp bench-family formula over ~8000 seeded draw pairs.

#[cfg(test)]
mod gaussian_sampler_tests {
    use super::{gaussian_pair_from_uniforms, sample_gaussian_pair_clamped};
    use rand::RngExt;
    use rand::SeedableRng;

    /// Reference: the bench's pre-extraction EPSILON-clamp pair sampler
    /// (2 draws per 2 samples, cos+sin).
    fn reference_bench_pair(rng: &mut impl rand::Rng) -> (f32, f32) {
        let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
        let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).sin();
        (z1, z2)
    }

    #[test]
    fn seeded_bench_sequence_byte_identical() {
        // Bench seed 43; 8000 samples = 4000 draw pairs (2 draws per 2 samples).
        const DRAW_PAIRS: usize = 4000;
        let mut shared_rng = rand::rngs::StdRng::seed_from_u64(43);
        let mut ref_rng = rand::rngs::StdRng::seed_from_u64(43);
        for i in 0..DRAW_PAIRS {
            let shared = sample_gaussian_pair_clamped(&mut shared_rng);
            let reference = reference_bench_pair(&mut ref_rng);
            #[expect(clippy::float_cmp)] // same-seed RNG must be byte-identical
            {
                assert_eq!(shared.0, reference.0, "bench z1 divergence at pair {i}");
            }
            #[expect(clippy::float_cmp)] // same-seed RNG must be byte-identical
            {
                assert_eq!(shared.1, reference.1, "bench z2 divergence at pair {i}");
            }
        }
    }

    #[test]
    fn pair_math_matches_inline_formulas() {
        // gaussian_pair_from_uniforms must equal the inline cos/sin formulas.
        for &(u1, u2) in &[(0.1, 0.2), (0.5, 0.9), (0.001, 0.999), (0.25, 0.75)] {
            let (z1, z2) = gaussian_pair_from_uniforms(u1, u2);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            #[expect(clippy::float_cmp)]
            // identical expression ordering — bit-identical by construction
            {
                assert_eq!(z1, r * theta.cos(), "z1 for ({u1}, {u2})");
            }
            #[expect(clippy::float_cmp)]
            // identical expression ordering — bit-identical by construction
            {
                assert_eq!(z2, r * theta.sin(), "z2 for ({u1}, {u2})");
            }
        }
    }
}

// ── Reference-image loader (validation + compression ladder) ─────────────

#[cfg(test)]
mod reference_image_tests {
    use super::*;
    use crate::util::test::noisy_png;

    #[tokio::test]
    async fn under_cap_passes_through_unchanged() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("small.png");
        let bytes = noisy_png(64, 64);
        std::fs::write(&path, &bytes).unwrap();

        let img = load_reference_image(&path, 1_500_000).await.unwrap();
        assert_eq!(
            img.data_uri(),
            format!("data:image/png;base64,{}", STANDARD.encode(&bytes))
        );
    }

    #[tokio::test]
    async fn over_cap_is_compressed_under_cap() {
        const CAP: u64 = 500_000;
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("big.png");
        let bytes = noisy_png(512, 512);
        assert!(
            bytes.len() as u64 > CAP,
            "noise PNG should exceed the cap (got {} bytes)",
            bytes.len()
        );
        std::fs::write(&path, &bytes).unwrap();

        let img = load_reference_image(&path, CAP).await.unwrap();
        assert!(img.data_uri().starts_with("data:image/jpeg;base64,"));
        let decoded = STANDARD
            .decode(img.data_uri().split(',').nth(1).unwrap())
            .unwrap();
        assert!(decoded.len() as u64 <= CAP);
    }

    #[tokio::test]
    async fn unsupported_extension_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("photo.heic");
        std::fs::write(&path, b"\x00\x00\x00\x18ftypheic").unwrap();
        let Err(err) = load_reference_image(&path, 1_500_000).await else {
            panic!("HEIC reference should be rejected");
        };
        assert!(err.to_string().contains("unsupported format"), "{err}");
    }

    #[tokio::test]
    async fn undecodable_content_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("fake.png");
        std::fs::write(&path, b"this is not an image").unwrap();
        let Err(err) = load_reference_image(&path, 1_500_000).await else {
            panic!("undecodable reference should be rejected");
        };
        assert!(err.to_string().contains("not a decodable image"), "{err}");
    }

    /// Inject an EXIF orientation tag (0x0112 = 6 → Rotate90) into a JPEG right
    /// after the SOI marker. The stored pixels stay unrotated; a viewer (or the
    /// compression path) must rotate them per the tag.
    fn jpeg_with_exif_orientation(jpeg: &[u8]) -> Vec<u8> {
        assert!(jpeg.starts_with(&[0xFF, 0xD8]));
        let mut out = jpeg[..2].to_vec();
        // APP1: FF E1, length 0x0022 = 2 (len) + 6 ("Exif\0\0") + 26 (TIFF).
        out.extend_from_slice(&[0xFF, 0xE1, 0x00, 0x22]);
        out.extend_from_slice(b"Exif\0\0");
        // Little-endian TIFF: header (8) + IFD (2 count + 12 entry + 4 next).
        out.extend_from_slice(&[
            0x49, 0x49, 0x2A, 0x00, // "II", 42
            0x08, 0x00, 0x00, 0x00, // IFD offset = 8
            0x01, 0x00, // 1 entry
            0x12, 0x01, 0x03, 0x00, // tag 0x0112, type SHORT
            0x01, 0x00, 0x00, 0x00, // count = 1
            0x06, 0x00, 0x00, 0x00, // value = 6 (Rotate90)
            0x00, 0x00, 0x00, 0x00, // next IFD = none
        ]);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn over_cap_jpeg_exif_orientation_applied() {
        use image::GenericImageView;
        use image::ImageDecoder;
        // 2x1 JPEG stored with EXIF orientation=6 (Rotate90): the compressed
        // output must come out 1x2 — a silently-rotated reference is exactly
        // the class of invisible corruption this pipeline exists to prevent.
        let src = image::RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        });
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode_image(&image::DynamicImage::ImageRgb8(src))
            .unwrap();
        let oriented = jpeg_with_exif_orientation(&jpeg);

        // Sanity: the crafted file's decoder reports the intended rotation.
        let reader = image::ImageReader::new(std::io::Cursor::new(&oriented))
            .with_guessed_format()
            .unwrap();
        assert_eq!(
            reader.into_decoder().unwrap().orientation().unwrap(),
            image::metadata::Orientation::Rotate90
        );

        let out = compress_reference_step(&oriented, 0).unwrap();
        let decoded = image::load_from_memory(&out).unwrap();
        assert_eq!(decoded.dimensions(), (1, 2));
    }
}

#[cfg(test)]
mod executable_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn test_is_executable_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_exe");

        // File doesn't exist — should not be executable.
        assert!(!is_executable(&file_path));

        // Create a non-executable file.
        std::fs::write(&file_path, "content").unwrap();
        std::fs::set_permissions(&file_path, PermissionsExt::from_mode(0o644)).unwrap();
        assert!(
            !is_executable(&file_path),
            "File with mode 644 should not be executable"
        );

        // Set executable bit.
        crate::util::test::make_executable(&file_path);
        assert!(
            is_executable(&file_path),
            "File with mode 755 should be executable"
        );

        // Also test with only owner execute bit.
        std::fs::set_permissions(&file_path, PermissionsExt::from_mode(0o100)).unwrap();
        assert!(
            is_executable(&file_path),
            "File with mode 100 should be executable"
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_is_executable_on_windows() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_exe.exe");

        // File doesn't exist — should not be executable.
        assert!(!is_executable(&file_path));

        // Create an exe file.
        std::fs::write(&file_path, "content").unwrap();
        assert!(
            is_executable(&file_path),
            "File with .exe extension should be executable"
        );

        // Non-exe file should not be executable.
        let txt_path = dir.path().join("test.txt");
        std::fs::write(&txt_path, "content").unwrap();
        assert!(
            !is_executable(&txt_path),
            "File with .txt extension should not be executable"
        );
    }
}
