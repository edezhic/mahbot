//! Channel message enrichment: media marker processing, link enrichment, file
//! operations, and per-role media handling.
//!
//! This module transforms [`ChannelMessage`] content before it reaches the
//! agent pipeline. It handles:
//! - **Media markers** (`[IMAGE: ...]`, `[AUDIO: ...]`, `[VIDEO: ...]`)
//!   → inbound local images become native data-URI parts for EVERY role
//!   (byte-identical for Artist, bounded-JPEG compressed for all others);
//!   audio is transcribed to text; video handling is workspace copy +
//!   transcription for every role (no role split)
//! - **Link enrichment** → prepends webpage summaries for URLs in the message
//! - **File operations** → downloading/saving images to workspace, cleaning
//!   up temporary files
//!
//! **Containment invariant**: all local file reads, copies, and deletes are
//! scoped to the daemon's Telegram temp dir. Marker paths outside it
//! (user-typed `[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]` annotations) degrade
//! to plain-text annotations and are never read, transcribed, copied into
//! workspace uploads, or deleted.
//!
//! The public entry points are [`enrich_message`] and [`enrich_links`],
//! re-exported from [`crate::channels`]. The [`EnrichmentStrategy`] struct
//! carries the per-role knobs: image and video handling are unconditional
//! (native data-URI parts for images, workspace copy + transcription for
//! videos — every role), while image compression is role-dependent.

use crate::ChannelMessage;
use crate::tools::browser::BrowserTool;
use crate::util::{MEDIA_MARKER_RE, file_name_or_path, is_http_url, parse_media_marker};
use regex::Regex;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Write;
use std::sync::LazyLock;

/// URL regex: matches http:// and https:// URLs, stopping at whitespace, angle
/// brackets, or double-quotes.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"']+"#).expect("URL regex must compile"));

/// The audio-transcription icon combo (sound written into text). Used both as
/// the transcription-failure fallback and as the annotation for out-of-scope
/// `[AUDIO:...]` markers that must never be read or deleted.
const AUDIO_ICON: &str = "🔊✍️";

/// Transcribe an audio file referenced by a `[AUDIO:...]` marker and return
/// the content to embed in the message: the audio-transcription icon combo
/// (`🔊✍️` — sound written into text) followed by the transcription text.
///
/// The audio file is a pure intermediate artifact — the caller deletes it
/// regardless of outcome (scoped to the Telegram temp dir), so the returned
/// string never contains the file name or path. On failure just the icon
/// combo is returned (no text).
async fn transcribe_audio_marker(path: &str) -> String {
    let path_buf = std::path::PathBuf::from(path);

    // ── Step 1: Try local Qwen3-ASR transcription ────────────────────
    // Default to enabled; only explicitly "false" disables local transcription.
    let use_local = crate::config::CONFIG
        .snapshot()
        .audio_transcription_use_local
        .as_deref()
        != Some("false");

    if use_local {
        match crate::audio::local_transcriber::transcribe_file_async(
            &path_buf,
            // 10-minute timeout for enrichment path — attached audio can be
            // arbitrarily long (voice memos, meeting recordings, etc.).
            crate::audio::local_transcriber::INFERENCE_TIMEOUT,
        )
        .await
        {
            Ok(text) => {
                tracing::debug!("Local audio transcription succeeded");
                let text = text.trim();
                return if text.is_empty() {
                    AUDIO_ICON.to_string()
                } else {
                    format!("{AUDIO_ICON} {text}")
                };
            }
            Err(e) => {
                tracing::warn!(error = %e, "Local audio transcription failed");
            }
        }
    }

    // ── Step 2: Icon-only fallback (no text, no filename) ────────────
    tracing::warn!("Audio transcription unavailable");
    AUDIO_ICON.to_string()
}

/// A media file copied into the workspace `uploads/` directory: the
/// `[Saved {label}: path]` annotation for the message and the destination
/// path for agent tool references.
struct SavedMedia {
    annotation: String,
    dest: std::path::PathBuf,
}

/// Copy a media file (image/video) into the workspace `uploads/` directory so
/// the agent can reference it via tool calls. Returns `None` when no uploads
/// dir is available or the copy fails.
async fn save_media_to_workspace(
    media_path: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    label: &str,
    fallback_ext: &str,
) -> Option<SavedMedia> {
    let dir = uploads_dir?;
    tokio::fs::create_dir_all(dir).await.ok()?;
    let ext = media_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or(fallback_ext);
    let timestamp = crate::util::unix_millis();
    let dest_name = format!("upload_{timestamp}.{ext}");
    let dest_path = dir.join(&dest_name);
    tokio::fs::copy(media_path, &dest_path).await.ok()?;
    Some(SavedMedia {
        annotation: format!("[Saved {label}: {}]", dest_path.display()),
        dest: dest_path,
    })
}

/// Per-message media-enrichment behavior, decided at the channel boundary for
/// the routed role. Image handling is unconditional (native data-URI parts for
/// every role) and video handling is unconditional too (workspace copy +
/// transcription for every role); only image compression is role-dependent.
#[derive(Debug, Clone)]
pub struct EnrichmentStrategy {
    /// Workspace uploads dir for saved full-resolution media copies (`None`
    /// disables copies).
    pub workspace_path: Option<std::path::PathBuf>,
    /// Downscale/compress inbound local images to a bounded JPEG before they
    /// enter the session — every role EXCEPT Artist. Artist passes through
    /// full-resolution byte-identical.
    pub compress_images: bool,
}

/// Outcome of processing an IMAGE marker.
enum ImageAction {
    /// Keep the marker unchanged (e.g. HTTP/HTTPS URL).
    Keep,
    /// Replace the marker with the given text, optionally including an
    /// upload-path annotation for agent tool references. `delete_temp` is set
    /// only when the source file was consumed from the daemon's Telegram temp
    /// dir (copied/read) — out-of-scope and missing files are never deleted.
    Replace {
        replacement: String,
        upload_annotation: Option<String>,
        delete_temp: bool,
    },
}

/// Handle an IMAGE marker — convert to a data URI, invalid reference, or (for
/// out-of-scope paths) a plain-text annotation. Saves a workspace copy if
/// `uploads_dir` is available. When `compress` is set the data URI is a
/// bounded-JPEG re-encode (non-Artist roles); otherwise the original bytes
/// pass through byte-identical (Artist). The returned action's `delete_temp`
/// tells the caller whether the source temp file was consumed from the
/// Telegram temp dir and may be cleaned up.
async fn handle_image(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    compress: bool,
) -> ImageAction {
    // HTTP/HTTPS URLs can be sent as-is.
    if is_http_url(path) {
        return ImageAction::Keep;
    }

    let invalid_ref = format!("[Invalid image reference: {path}]");
    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Image file not found for enrichment");
        return ImageAction::Replace {
            replacement: invalid_ref,
            upload_annotation: None,
            delete_temp: false,
        };
    }

    // Containment: only Telegram-temp-dir files may be read, copied, or deleted.
    if !is_under_telegram_files(path_obj).await {
        tracing::warn!(%path, "Image path outside telegram temp dir — annotating without copy");
        return ImageAction::Replace {
            replacement: format!("[Image: {} attached]", file_name_or_path(path)),
            upload_annotation: None,
            delete_temp: false,
        };
    }

    // Save a copy to workspace uploads so the agent can reference it
    let saved = save_media_to_workspace(path_obj, uploads_dir, "image", "png")
        .await
        .map(|saved| saved.annotation);

    // Convert to data URI for the API request. The compressed path is
    // fail-open: any decode/encode error falls back to the original bytes
    // untouched; a total conversion failure degrades to the invalid ref.
    let data_uri = if compress {
        match crate::util::local_image_to_compressed_data_uri(path_obj).await {
            Ok(uri) => Ok(uri),
            Err(e) => {
                tracing::warn!(%path, error = %e, "Image compression failed — falling back to original bytes");
                crate::util::local_image_to_data_uri(path_obj).await
            }
        }
    } else {
        crate::util::local_image_to_data_uri(path_obj).await
    };
    let replacement = match data_uri {
        Ok(data_uri) => format!("[IMAGE:{data_uri}]"),
        Err(e) => {
            tracing::warn!(%path, error = %e, "Failed to convert image to data URI");
            invalid_ref
        }
    };

    ImageAction::Replace {
        replacement,
        upload_annotation: saved,
        delete_temp: true,
    }
}

/// Whether a local media path resolves inside the daemon's Telegram
/// attachment temp dir — the only legitimate source of inbound media
/// (video clips and voice messages). Arbitrary marker paths must never
/// reach workspace uploads or be deleted.
async fn is_under_telegram_files(path: &std::path::Path) -> bool {
    let Ok(canonical) = tokio::fs::canonicalize(path).await else {
        return false;
    };
    let root = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
    let Ok(canonical_root) = tokio::fs::canonicalize(&root).await else {
        return false;
    };
    crate::tools::path::is_path_under_roots(&canonical, &[canonical_root])
}

/// Outcome of processing a VIDEO marker: the replacement text, whether the
/// source temp file was copied into the workspace uploads (so the agent can
/// feed the clip to the video-edit flow) and can be cleaned up, and the
/// optional "[Video transcription of <name>]: <text>" annotation
/// (prepended as an annotation block by the caller). HTTP(S) URLs, missing
/// files, and out-of-scope paths degrade to plain annotations; in-scope temp
/// files are always cleaned up (whether or not the workspace copy succeeded).
struct VideoAction {
    replacement: String,
    delete_temp: bool,
    transcription: Option<String>,
}

impl VideoAction {
    /// Plain annotation: no workspace copy, no transcription.
    fn annotation(replacement: String) -> Self {
        Self {
            replacement,
            delete_temp: false,
            transcription: None,
        }
    }
}

async fn handle_video(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    workspace: &str,
) -> VideoAction {
    if is_http_url(path) {
        return VideoAction::annotation(format!("[Video: {path}]"));
    }
    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Video file not found for enrichment");
        return VideoAction::annotation(format!("[Invalid video reference: {path}]"));
    }
    if !is_under_telegram_files(path_obj).await {
        tracing::warn!(%path, "Video path outside telegram temp dir — annotating without copy");
        return VideoAction::annotation(format!("[Video: {} attached]", file_name_or_path(path)));
    }
    if let Some(saved) = save_media_to_workspace(path_obj, uploads_dir, "video", "mp4").await {
        // Transcribe the persistent workspace copy — never the temp file
        // (deleted after a successful copy). The annotation keeps the original
        // Telegram filename; fail-open: any failure degrades to the plain
        // [Saved video: ...] annotation.
        let transcription =
            transcribe_saved_video(&saved.dest, file_name_or_path(path), workspace).await;
        return VideoAction {
            replacement: saved.annotation,
            delete_temp: true,
            transcription,
        };
    }
    // Copy failed (or no uploads dir — e.g. a no-role message): annotate
    // without transcription, but the in-scope temp file is still a pure
    // intermediate artifact and is cleaned up. `is_under_telegram_files` was
    // verified above, so the delete stays inside the containment boundary.
    VideoAction {
        replacement: format!("[Video: {} attached]", file_name_or_path(path)),
        delete_temp: true,
        transcription: None,
    }
}

/// Transcribe a saved workspace video copy for the routed role, returning the
/// "[Video transcription of <name>]: <text>" annotation (using the original
/// source `file_name`). Fail-open: returns `None` (plain annotation) when the
/// transcription fails (unavailable transcriber, unsupported format, upload
/// or model error, timeout, empty output) — the overall timeout lives inside
/// [`transcribe_video_file`](crate::providers::transcribe_video_file),
/// bounding both callers. `workspace` names the workspace for telemetry and
/// the live non-agent call row (the inbound path has no agent card).
async fn transcribe_saved_video(
    path: &std::path::Path,
    file_name: &str,
    workspace: &str,
) -> Option<String> {
    let text = crate::providers::transcribe_video_file(path, Some(workspace)).await?;
    Some(format!("[Video transcription of {file_name}]: {text}"))
}

/// Process all media markers (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`)
/// in a single pass. Each marker kind is handled according to the strategy:
///
/// | Kind | Behavior |
/// |------|----------|
/// | IMAGE | data URI conversion (byte-identical for Artist, bounded-JPEG compression for every other role) + workspace copy when in scope |
/// | AUDIO | transcription (unchanged for all roles) |
/// | VIDEO | workspace copy + `[Saved video: path]` + transcription (every role) |
///
/// After processing, markers that were handled are stripped from the content
/// and annotations are prepended. Temp files are cleaned up after processing,
/// scoped to the daemon's Telegram temp dir — out-of-scope marker paths are
/// never read, copied, or deleted and degrade to plain-text annotations.
// Marker dispatch hub (3 kinds, per-kind handling is extracted into the
// handler functions above, keeping this loop flat on purpose).
pub async fn enrich_message(msg: &mut ChannelMessage, strategy: &EnrichmentStrategy) {
    let mut annotations: Vec<String> = Vec::new();
    let mut result = msg.content.clone();
    // Accumulates upload path annotations across the for-loop below.
    // Populated whenever a local in-scope image was copied to workspace
    // uploads — any role.
    let mut upload_annotations: Vec<String> = Vec::new();

    // Temp files to remove after the loop — only ever queued for files under
    // the daemon's Telegram temp dir, so user-typed markers can never delete
    // arbitrary local files.
    let mut files_to_delete: Vec<std::path::PathBuf> = Vec::new();

    let uploads_dir = strategy.workspace_path.as_ref().map(|p| p.join("uploads"));

    for caps in MEDIA_MARKER_RE.captures_iter(&msg.content) {
        let whole = caps.get_match();
        let (kind, path) = parse_media_marker(&caps);
        let path_obj = std::path::Path::new(path);

        match kind {
            "IMAGE" => {
                match handle_image(
                    path,
                    path_obj,
                    uploads_dir.as_deref(),
                    strategy.compress_images,
                )
                .await
                {
                    ImageAction::Keep => {
                        // HTTP/HTTPS URL — no local file to clean up.
                    }
                    ImageAction::Replace {
                        replacement,
                        upload_annotation,
                        delete_temp,
                    } => {
                        result = result.replacen(whole.as_str(), &replacement, 1);
                        if let Some(ann) = upload_annotation {
                            upload_annotations.push(ann);
                        }
                        // Local IMAGE temp files are cleaned up only when
                        // consumed from the daemon's Telegram temp dir (delete_temp).
                        if delete_temp {
                            files_to_delete.push(path_obj.to_path_buf());
                        }
                    }
                }
            }
            "AUDIO" => {
                // Containment: only Telegram-temp-dir files are transcribed or
                // deleted; out-of-scope markers degrade to the icon only.
                if is_under_telegram_files(path_obj).await {
                    annotations.push(transcribe_audio_marker(path).await);
                    files_to_delete.push(path_obj.to_path_buf());
                } else {
                    tracing::warn!(%path, "Audio path outside telegram temp dir — annotating without transcription");
                    annotations.push(AUDIO_ICON.to_string());
                }
            }
            "VIDEO" => {
                let VideoAction {
                    replacement,
                    delete_temp,
                    transcription,
                } = handle_video(path, path_obj, uploads_dir.as_deref(), &msg.workspace).await;
                result = result.replacen(whole.as_str(), &replacement, 1);
                if let Some(annotation) = transcription {
                    annotations.push(annotation);
                }
                if delete_temp {
                    files_to_delete.push(path_obj.to_path_buf());
                }
            }
            // NOTE: If a new marker kind is added to MEDIA_MARKER_RE in
            // util/mod.rs, a corresponding arm MUST be added here for enrichment
            // behavior (transcription, annotation, etc.). The unified stripping
            // at the end of this function handles marker removal: IMAGE markers
            // are always preserved (native image parts), all other markers are
            // stripped. The `_ =>` arm is unreachable for well-formed markers
            // (the regex only matches IMAGE|AUDIO|VIDEO), but exists as a
            // defensive guard during development.
            _ => {
                tracing::warn!(kind, %path, "Unknown media marker kind");
            }
        }
    }

    // ── File cleanup ────────────────────────────────────────────────
    // Delete queued temp files (only Telegram-temp-dir paths are ever queued).
    // Deletion errors are logged (not silently discarded).
    for file_path in &files_to_delete {
        if let Err(e) = tokio::fs::remove_file(file_path).await {
            tracing::warn!(
                path = %file_path.display(),
                error = %e,
                "Failed to delete temp file after enrichment"
            );
        }
    }

    // ── Upload-annotation post-processing ──
    // Append upload path annotations so the model can reference saved files.
    // `upload_annotations` accumulates across the for-loop; it is populated
    // whenever a local in-scope image was copied to the workspace uploads
    // directory (any role).
    if !upload_annotations.is_empty() {
        let annotation_block = upload_annotations.join("\n");
        let _ = write!(result, "\n\n{annotation_block}");
    }

    // ── Marker stripping and annotation prepending ──
    // Strip media markers from the enriched content. IMAGE markers are always
    // preserved — they carry the native image parts the routed role's model
    // consumes via `to_message_content`; AUDIO/VIDEO (and any future marker
    // kind) are stripped. The MEDIA_MARKER_PATTERN constant in util/mod.rs is
    // the single canonical source of truth for the marker pattern; both
    // MEDIA_MARKER_RE (case-sensitive) and TELEGRAM_MEDIA_MARKER_RE
    // (case-insensitive) are built from it to stay in sync.
    let cleaned = MEDIA_MARKER_RE
        .replace_all(&result, |caps: &regex::Captures| {
            if parse_media_marker(caps).0 == "IMAGE" {
                caps.get_match().as_str().to_string()
            } else {
                String::new()
            }
        })
        .to_string();
    let cleaned = cleaned.trim().to_string();

    // ── Prepend text annotations (if any) ──
    // These are accumulated text descriptions: transcribed AUDIO content and
    // VIDEO transcription annotations.
    msg.content = if annotations.is_empty() {
        cleaned
    } else {
        let prefix = annotations.join("\n");
        if cleaned.is_empty() {
            prefix
        } else {
            format!("{prefix}\n\n{cleaned}")
        }
    };
}

/// Whether `content` carries `[AUDIO:...]` markers and no other media markers.
///
/// Used by the caller to decide when enriched content (icon + transcription)
/// can be persisted to chat history instead of the raw original: audio-only
/// messages never leak temp file paths (raw audio markers) or embed image
/// data URIs (IMAGE markers). Mixed audio+image/audio+video
/// messages fall back to the raw persist — the raw `[AUDIO:path]` marker
/// still reaches chat history for those (data-URI avoidance takes precedence).
/// Purely syntactic: a hand-typed `[AUDIO:<URL>]` marker qualifies too, so its
/// icon-only enriched content is persisted and the URL is dropped.
#[must_use]
pub fn has_only_audio_markers(content: &str) -> bool {
    let mut has_audio = false;
    for caps in MEDIA_MARKER_RE.captures_iter(content) {
        if parse_media_marker(&caps).0 == "AUDIO" {
            has_audio = true;
        } else {
            return false;
        }
    }
    has_audio
}

/// Extract all unique URLs from message text.
///
/// Strips common trailing punctuation (commas, periods, closing brackets,
/// colons, semicolons, exclamation/question marks) that naturally appears
/// around URLs in prose.
fn extract_urls(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for m in URL_RE.find_iter(text) {
        let mut url = m.as_str().to_string();
        // Strip trailing punctuation that isn't part of the actual URL
        while url.ends_with(&[',', '.', ')', ']', '}', ':', ';', '!', '?'][..]) {
            url.pop();
        }
        if seen.insert(url.clone()) {
            result.push(url);
        }
    }
    result
}

/// Enrich a message by prepending link summaries for any URLs found in the text.
///
/// If no URLs are found, the original message is returned unchanged.
/// Links are fetched concurrently using the shared `BrowserTool` — each URL
/// gets its own isolated session tab that is closed after text extraction.
pub async fn enrich_links(content: &str) -> Cow<'_, str> {
    // Truncate very long snippets to keep messages manageable.
    const MAX_TEXT_LEN: usize = 5000;
    let urls = extract_urls(content);
    if urls.is_empty() {
        return Cow::Borrowed(content);
    }

    // Gate on the cached (non-probing) daemon advertisement first — the cheap
    // in-memory check short-circuits the `--version` spawn below while the
    // daemon is confirmed down. A stale/unknown state passes optimistically
    // and the concurrent fetch tasks re-discover liveness (bounded by the
    // probe timeout) without failing the message.
    if !(crate::tools::browser_daemon::is_advertised()
        && matches!(
            crate::tools::browser_daemon::cli_probe().await,
            crate::tools::browser_daemon::CliStatus::Available
        ))
    {
        tracing::debug!("chrome-use not available, skipping link enrichment");
        return Cow::Borrowed(content);
    }

    // Fetch all URLs concurrently.
    let browser = std::sync::Arc::new(BrowserTool::default());
    let mut tasks = Vec::with_capacity(urls.len());
    for (i, url) in urls.iter().enumerate() {
        let url = url.clone();
        let tab = format!("link-enricher-{i}");
        let browser = std::sync::Arc::clone(&browser);
        tasks.push(tokio::spawn(async move {
            let result = browser.fetch_page_text(&url, &tab).await;
            // Close the tab (best-effort) regardless of fetch outcome.
            browser.close_session(&tab).await;
            (url, result)
        }));
    }

    let mut enrichments: Vec<String> = Vec::new();
    for task in tasks {
        match task.await {
            Ok((url, Ok(body_text))) => {
                if body_text.trim().is_empty() {
                    // Blank/empty page — don't insert an empty snippet.
                    tracing::debug!(url, "Link enricher: page text is empty, skipping snippet");
                    continue;
                }
                let snippet = if body_text.len() > MAX_TEXT_LEN {
                    format!("{}…", crate::util::truncate_bytes(&body_text, MAX_TEXT_LEN))
                } else {
                    body_text
                };
                enrichments.push(format!("📄 [{url}]\n{snippet}"));
            }
            Ok((url, Err(e))) => {
                tracing::debug!(url, error = %e, "Link enricher: failed to fetch page text");
            }
            Err(e) => {
                tracing::debug!("Link enricher task panicked: {e}");
            }
        }
    }

    if enrichments.is_empty() {
        return Cow::Borrowed(content);
    }

    let prefix = enrichments.join("\n\n");
    Cow::Owned(format!("{prefix}\n\n{content}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;

    #[test]
    fn extract_urls_finds_http_and_https() {
        let urls = extract_urls("Check https://example.com and http://test.org/page for info");
        assert_eq!(urls, vec!["https://example.com", "http://test.org/page"]);
    }

    #[test]
    fn extract_urls_deduplicates() {
        let urls = extract_urls("Visit https://example.com and https://example.com again");
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn extract_urls_strips_trailing_punctuation() {
        let urls = extract_urls("See https://example.com, and https://test.org.");
        assert_eq!(urls, vec!["https://example.com", "https://test.org"]);
    }

    #[test]
    fn extract_urls_handles_urls_in_parens() {
        let urls = extract_urls("(https://example.com) and [https://test.org]");
        assert_eq!(urls, vec!["https://example.com", "https://test.org"]);
    }

    #[tokio::test]
    async fn enrich_links_returns_borrowed_when_no_urls() {
        let content = "Hello, this is a plain message without any URLs.";
        let result = enrich_links(content).await;
        // No URLs → should borrow the input, not allocate a new String.
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.as_ref(), content);
    }

    // ── Enrichment strategy tests ─────────────────────────────────────

    /// Helper: quick ChannelMessage for enrichment tests.
    fn test_msg(content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: "test".into(),
            reply_target: "test".into(),
            content: content.to_string(),
            channel: "test".into(),
            workspace: "test".into(),
            optimistic_id: None,
            callback_query_id: None,
        }
    }

    /// Create the daemon's Telegram temp dir. The containment root must exist
    /// before path canonicalization — a missing root makes every path look
    /// out of scope.
    async fn ensure_telegram_files_dir() {
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
    }

    /// Set up an out-of-scope fixture: a unique scratch dir under the system
    /// temp dir (outside the Telegram temp dir) containing a fake workspace
    /// (`ws_path`) and a single arbitrary media file. Returns
    /// `(tmp_root, ws_path, arbitrary_file)`.
    async fn out_of_scope_fixture(
        prefix: &str,
        file_name: &str,
        contents: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        ensure_telegram_files_dir().await;
        let tmp_root = std::env::temp_dir().join(format!("{prefix}_{}", std::process::id()));
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();
        let arbitrary = tmp_root.join(file_name);
        tokio::fs::write(&arbitrary, contents).await.unwrap();
        (tmp_root, ws_path, arbitrary)
    }

    /// Generate a real decodable PNG of the given dimensions (solid gradient).
    fn real_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                u8::try_from(x % 256).expect("x % 256 fits in u8"),
                u8::try_from(y % 256).expect("y % 256 fits in u8"),
                128,
            ])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("fixture PNG must encode");
        bytes
    }

    /// Extract the image payload embedded in a `[IMAGE:...]` data URI inside
    /// `content` and decode it, returning the decoded image dimensions.
    fn embedded_image_dimensions(content: &str) -> (u32, u32) {
        use image::GenericImageView;
        let data_uri = content
            .split("[IMAGE:")
            .nth(1)
            .expect("data URI marker must be present")
            .split(']')
            .next()
            .expect("data URI marker must be closed");
        let b64 = data_uri
            .split(',')
            .nth(1)
            .expect("data URI must carry a base64 payload");
        let bytes = STANDARD.decode(b64).expect("data URI base64 must decode");
        let img = image::load_from_memory(&bytes).expect("embedded image must decode");
        img.dimensions()
    }

    #[tokio::test]
    async fn enrich_image_http_url_passthrough() {
        let mut msg = test_msg("Check this [IMAGE:https://example.com/img.png] out");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        assert_eq!(
            msg.content,
            "Check this [IMAGE:https://example.com/img.png] out"
        );
    }

    #[tokio::test]
    async fn enrich_image_file_not_found() {
        let mut msg = test_msg("Here is [IMAGE:/tmp/nonexistent_xyz_img.png] an image");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        assert!(
            msg.content
                .contains("[Invalid image reference: /tmp/nonexistent_xyz_img.png]")
        );
    }

    #[tokio::test]
    async fn enrich_audio_annotation_and_strip() {
        let mut msg = test_msg("Listen [AUDIO:/tmp/audio_xyz.mp3] to this");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        // AUDIO marker stripped; annotation prepended (icon-only fallback since
        // no audio transcriber is configured in the test environment)
        assert!(
            msg.content.contains("🔊✍️"),
            "Audio annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
        // No file name may survive in any form
        assert!(
            !msg.content.contains("audio_xyz"),
            "Audio temp file name must not appear, got: {}",
            msg.content
        );
        // The original text is preserved
        assert!(msg.content.contains("Listen"), "Original text preserved");
        assert!(msg.content.contains("to this"), "Original text preserved");
    }

    #[tokio::test]
    async fn enrich_image_valid_file_converts_to_data_uri_and_deletes_temp() {
        // The fixture must live under the daemon's Telegram temp dir to be in
        // scope for reading (data URI) and cleanup.
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp = tg_dir.join(format!("test_enrich_img_{}.png", std::process::id()));
        let png_header: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0x60, 0x60, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE7, 0x21, 0x33, 0x7C,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        tokio::fs::write(&tmp, png_header).await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Image: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Marker replaced with data URI
        assert!(
            msg.content.contains("[IMAGE:data:image/png;base64,"),
            "Expected data URI, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(&path_str),
            "Raw file path must not remain in content"
        );
        // Temp file deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
    }

    #[tokio::test]
    async fn enrich_image_with_workspace_creates_upload_annotation() {
        let tmp_root = std::env::temp_dir().join(format!("test_enrich_ws_{}", std::process::id()));
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // The fixture must live under the daemon's Telegram temp dir to be in
        // scope for reading (data URI) and cleanup.
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp_img = tg_dir.join(format!("test_enrich_ws_img_{}.png", std::process::id()));
        let png_header: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0x60, 0x60, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE7, 0x21, 0x33, 0x7C,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        tokio::fs::write(&tmp_img, png_header).await.unwrap();
        let img_path_str = tmp_img.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Image: [IMAGE:{img_path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Data URI present and upload annotation added
        assert!(msg.content.contains("[IMAGE:data:image/png;base64,"));
        assert!(
            msg.content.contains("[Saved image:"),
            "Upload annotation must be present, got: {}",
            msg.content
        );
        // Temp file deleted
        assert!(
            !tmp_img.exists(),
            "Temp file must be deleted after enrichment"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_with_workspace_copies_and_annotates() {
        let tmp_root =
            std::env::temp_dir().join(format!("test_enrich_video_ws_{}", std::process::id()));
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // Only clips in the daemon's Telegram temp dir are eligible for copy.
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp_video = tg_dir.join(format!("test_enrich_video_{}.mp4", std::process::id()));
        let mp4_header: &[u8] = &[
            0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D, 0x00, 0x00,
            0x00, 0x00, 0x69, 0x73, 0x6F, 0x6D, 0x69, 0x73, 0x6F, 0x32,
        ];
        tokio::fs::write(&tmp_video, mp4_header).await.unwrap();
        let video_path_str = tmp_video.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Edit this clip: [VIDEO:{video_path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Marker replaced with a [Saved video: ...] annotation pointing at the
        // workspace uploads copy so the agent can feed it to video_edit.
        assert!(
            msg.content.contains("[Saved video:"),
            "Video upload annotation must be present, got: {}",
            msg.content
        );
        assert!(
            msg.content
                .contains(&ws_path.join("uploads").display().to_string()),
            "Annotation must point into workspace uploads, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
        // Temp file deleted after the workspace copy
        assert!(
            !tmp_video.exists(),
            "Temp video file must be deleted after enrichment"
        );
        // Cleanup
        let _ = tokio::fs::remove_file(&tmp_video).await;
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_outside_telegram_files_annotates_without_copy() {
        // An injected marker pointing at an arbitrary readable file must not
        // be copied into uploads (exfiltration vector) or deleted.
        let (tmp_root, ws_path, arbitrary) =
            out_of_scope_fixture("test_enrich_video_outside", "secret.txt", b"top secret").await;
        let marker = format!("Edit [VIDEO:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[Video: secret.txt attached]"),
            "Out-of-scope path must degrade to a plain-text annotation, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[Saved video:"));
        assert!(!msg.content.contains("[VIDEO:"));
        assert!(
            arbitrary.exists(),
            "Source file outside the telegram temp dir must not be deleted"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for out-of-scope paths"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_image_outside_telegram_files_annotates_without_read_or_copy() {
        // An injected marker pointing at an arbitrary readable file must not
        // be read into model context (data URI), copied into uploads, or
        // deleted.
        let (tmp_root, ws_path, arbitrary) = out_of_scope_fixture(
            "test_enrich_img_outside",
            "secret.png",
            b"top secret image bytes",
        )
        .await;
        let marker = format!("Look at [IMAGE:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[Image: secret.png attached]"),
            "Out-of-scope path must degrade to a plain-text annotation, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("data:image"),
            "No data URI may be produced for out-of-scope paths (would read the file)"
        );
        assert!(!msg.content.contains("[Saved image:"));
        assert!(!msg.content.contains("[IMAGE:"));
        assert!(
            arbitrary.exists(),
            "Source file outside the telegram temp dir must not be deleted"
        );
        assert_eq!(
            tokio::fs::read(&arbitrary).await.unwrap(),
            b"top secret image bytes",
            "Source file contents must be unchanged"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for out-of-scope paths"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_http_url_kept_as_plain_text() {
        let mut msg = test_msg("Edit [VIDEO:https://example.com/clip.mp4] this");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        // HTTP URL video reference is preserved as plain text (no marker strip)
        assert!(
            msg.content
                .contains("[Video: https://example.com/clip.mp4]"),
            "HTTP video URL must be kept as plain-text reference, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
    }

    #[tokio::test]
    async fn enrich_video_without_workspace_deletes_in_scope_temp() {
        // No workspace path (e.g. a no-role user's message): the video gets a
        // plain annotation, no transcription, but the in-scope Telegram temp
        // file is still a pure intermediate artifact and must be cleaned up —
        // a regression guard for the no-role gating in main.rs.
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp_video = tg_dir.join(format!(
            "test_enrich_video_norole_{}.mp4",
            std::process::id()
        ));
        tokio::fs::write(&tmp_video, b"fake mp4").await.unwrap();
        let video_path_str = tmp_video.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Watch [VIDEO:{video_path_str}] this clip"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains(&format!(
                "[Video: {} attached]",
                tmp_video.file_name().unwrap().to_string_lossy()
            )),
            "Plain video annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[Saved video:"),
            "No workspace copy may be made without an uploads dir, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
        assert!(
            !tmp_video.exists(),
            "In-scope temp video must be deleted even without a workspace copy"
        );
    }

    #[tokio::test]
    async fn enrich_video_outside_telegram_files_annotates_without_copy_or_delete() {
        // An injected marker pointing at an arbitrary readable file must not
        // be copied into uploads (exfiltration vector) or deleted.
        let (tmp_root, ws_path, arbitrary) = out_of_scope_fixture(
            "test_enrich_video_nonmm",
            "secret.mp4",
            b"top secret video bytes",
        )
        .await;
        let marker = format!("Watch [VIDEO:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[Video: secret.mp4 attached]"),
            "Out-of-scope path must degrade to a plain-text annotation, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
        assert!(
            arbitrary.exists(),
            "Source file outside the telegram temp dir must not be deleted"
        );
        assert_eq!(
            tokio::fs::read(&arbitrary).await.unwrap(),
            b"top secret video bytes",
            "Source file contents must be unchanged"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for out-of-scope paths"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_non_artist_image_compressed_to_jpeg_data_uri() {
        // Real 1500x1000 PNG: the longest side exceeds the 1024 px cap, so the
        // ingestion-time re-encode must downscale it to a bounded JPEG while
        // the workspace copy stays the full-resolution original.
        let source_bytes = real_png(1500, 1000);
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp = tg_dir.join(format!("test_enrich_compress_{}.png", std::process::id()));
        tokio::fs::write(&tmp, &source_bytes).await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let tmp_root =
            std::env::temp_dir().join(format!("test_enrich_compress_ws_{}", std::process::id()));
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        let mut msg = test_msg(&format!("Photo: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[IMAGE:data:image/jpeg;base64,"),
            "Compressed JPEG data URI expected, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("data:image/png"),
            "Original PNG data URI must not appear, got: {}",
            msg.content
        );
        let (dw, dh) = embedded_image_dimensions(&msg.content);
        assert!(
            dw.max(dh) <= crate::util::INBOUND_IMAGE_MAX_SIDE,
            "Compressed image longest side {} must be ≤ 1024",
            dw.max(dh)
        );
        // The workspace copy is the full-resolution original, byte-identical.
        let uploads_dir = ws_path.join("uploads");
        let mut entries = tokio::fs::read_dir(&uploads_dir).await.unwrap();
        let entry = entries
            .next_entry()
            .await
            .unwrap()
            .expect("one upload copy");
        let copy_bytes = tokio::fs::read(entry.path()).await.unwrap();
        assert_eq!(
            copy_bytes, source_bytes,
            "Workspace copy must be byte-identical to the source PNG"
        );
        // Temp file deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_artist_image_byte_identical_data_uri() {
        let source_bytes = real_png(64, 48);
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp = tg_dir.join(format!("test_enrich_artist_{}.png", std::process::id()));
        tokio::fs::write(&tmp, &source_bytes).await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Art: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        let expected = format!(
            "[IMAGE:data:image/png;base64,{}]",
            STANDARD.encode(&source_bytes)
        );
        assert!(
            msg.content.contains(&expected),
            "Artist data URI must be byte-identical to the source, got: {}",
            msg.content
        );
        // Temp file deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
    }

    #[tokio::test]
    async fn enrich_image_compression_failure_passes_original_through() {
        // Undecodable bytes: the compressed path fails open and the original
        // bytes pass through untouched as a data URI.
        let source_bytes = b"not an image";
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp = tg_dir.join(format!(
            "test_enrich_compress_fail_{}.png",
            std::process::id()
        ));
        tokio::fs::write(&tmp, source_bytes).await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Photo: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        let expected = format!(
            "[IMAGE:data:image/png;base64,{}]",
            STANDARD.encode(source_bytes)
        );
        assert!(
            msg.content.contains(&expected),
            "Compression failure must pass the original bytes through, got: {}",
            msg.content
        );
        // Temp file deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
    }

    #[tokio::test]
    async fn enrich_audio_file_deleted_on_failure() {
        // Only files in the daemon's Telegram temp dir are eligible for
        // cleanup (user-typed markers must never delete arbitrary files).
        let tg_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        tokio::fs::create_dir_all(&tg_dir).await.unwrap();
        let tmp = tg_dir.join(format!("test_enrich_audio_{}.mp3", std::process::id()));
        tokio::fs::write(&tmp, b"fake audio content").await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Audio: [AUDIO:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Temp file must be deleted even when transcription fails — the audio
        // file is a pure intermediate artifact (no transcriber in tests).
        assert!(
            !tmp.exists(),
            "Audio temp file must be deleted on transcription failure"
        );
        // Defensive cleanup in case the assertion above fails.
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn enrich_combined_image_preserved_audio_annotated() {
        let msg_content = "Here [IMAGE:https://example.com/img.png] and [AUDIO:/tmp/sound_xyz.mp3]";
        let mut msg = test_msg(msg_content);
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // IMAGE http URL kept
        assert!(
            msg.content.contains("[IMAGE:https://example.com/img.png]"),
            "IMAGE with http URL must be preserved, got: {}",
            msg.content
        );
        // AUDIO marker stripped, annotation present
        assert!(
            msg.content.contains("🔊✍️"),
            "Audio annotation must be present"
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
    }

    async fn assert_no_markers_unchanged(strategy: EnrichmentStrategy, content: &str) {
        let mut msg = test_msg(content);
        let original = msg.content.clone();
        enrich_message(&mut msg, &strategy).await;
        assert_eq!(msg.content, original, "No markers = no changes");
    }

    #[tokio::test]
    async fn enrich_no_annotations_when_no_markers() {
        assert_no_markers_unchanged(
            EnrichmentStrategy {
                workspace_path: None,
                compress_images: false,
            },
            "Just a plain message with no markers",
        )
        .await;
    }
}
