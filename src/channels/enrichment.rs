//! Channel message enrichment: media marker processing, link enrichment, file
//! operations, and multimodal annotation strategies.
//!
//! This module transforms [`ChannelMessage`] content before it reaches the
//! agent pipeline. It handles:
//! - **Media markers** (`[IMAGE: ...]`, `[AUDIO: ...]`, `[VIDEO: ...]`)
//!   → transcription for audio and video, data URI conversion for images
//!   (multimodal strategy) or strippping with annotation (non-multimodal
//!   strategy)
//! - **Link enrichment** → prepends webpage summaries for URLs in the message
//! - **File operations** → downloading/saving images to workspace, cleaning
//!   up temporary files
//!
//! The public entry points are [`enrich_message`] and [`enrich_links`],
//! re-exported from [`crate::channels`]. The two [`EnrichmentStrategy`]
//! variants control how image media markers are handled: `Multimodal`
//! preserves them as data URIs for vision-capable models, while
//! `NonMultimodal` strips them and adds a textual annotation.

use crate::ChannelMessage;
use crate::tools::browser::BrowserTool;
use crate::util::{MEDIA_MARKER_RE, is_http_url, parse_media_marker};
use regex::Regex;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Write;
use std::sync::LazyLock;

/// URL regex: matches http:// and https:// URLs, stopping at whitespace, angle
/// brackets, or double-quotes.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"']+"#).expect("URL regex must compile"));

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
                    "🔊✍️".to_string()
                } else {
                    format!("🔊✍️ {text}")
                };
            }
            Err(e) => {
                tracing::warn!(error = %e, "Local audio transcription failed");
            }
        }
    }

    // ── Step 2: Icon-only fallback (no text, no filename) ────────────
    tracing::warn!("Audio transcription unavailable");
    "🔊✍️".to_string()
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

/// Strategy for message enrichment, determining how each media marker kind
/// (IMAGE, AUDIO, VIDEO) is handled.
#[derive(Debug, Clone)]
pub enum EnrichmentStrategy {
    /// Multimodal mode:
    /// - IMAGE markers are converted to base64 data URIs (for vision model)
    /// - AUDIO markers are transcribed to text
    /// - VIDEO markers are copied to workspace `uploads/`, replaced with a
    ///   plain-text `[Saved video: path]` annotation, and transcribed to a
    ///   text description (media transcriber) so the Artist can understand the
    ///   clip and feed it into the video-edit flow
    ///
    /// When `workspace_path` is provided, copies of media files are saved to
    /// `uploads/` for agent tool references.
    Multimodal {
        workspace_path: Option<std::path::PathBuf>,
    },
    /// Non-multimodal mode: all media markers are transcribed/extracted to
    /// text annotations and the raw markers are stripped from the content.
    NonMultimodal,
}

/// Outcome of processing an IMAGE marker in multimodal mode.
enum MultimodalImageAction {
    /// Keep the marker unchanged (e.g. HTTP/HTTPS URL).
    Keep,
    /// Replace the marker with the given text, optionally including an
    /// upload-path annotation for agent tool references.
    Replace {
        replacement: String,
        upload_annotation: Option<String>,
    },
}

/// Handle an IMAGE marker in multimodal mode — convert to data URI or invalid
/// reference. Saves a workspace copy if `uploads_dir` is available.
/// The caller is responsible for cleaning up the source temp file.
async fn handle_multimodal_image(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
) -> MultimodalImageAction {
    // HTTP/HTTPS URLs can be sent as-is.
    if is_http_url(path) {
        return MultimodalImageAction::Keep;
    }

    let invalid_ref = format!("[Invalid image reference: {path}]");
    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Image file not found for multimodal enrichment");
        return MultimodalImageAction::Replace {
            replacement: invalid_ref,
            upload_annotation: None,
        };
    }

    // Save a copy to workspace uploads so the agent can reference it
    let saved = save_media_to_workspace(path_obj, uploads_dir, "image", "png")
        .await
        .map(|saved| saved.annotation);

    // Convert to data URI for the API request
    let replacement = match crate::util::local_image_to_data_uri(path_obj).await {
        Ok(data_uri) => format!("[IMAGE:{data_uri}]"),
        Err(e) => {
            tracing::warn!(%path, error = %e, "Failed to convert image to data URI");
            invalid_ref
        }
    };

    MultimodalImageAction::Replace {
        replacement,
        upload_annotation: saved,
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

/// Outcome of processing a VIDEO marker in multimodal mode: the replacement
/// text, whether the source temp file was copied into the workspace uploads
/// (so the Artist can feed the clip to the video-edit flow) and can be cleaned
/// up, and the optional "[Video transcription of <name>]: <text>" annotation
/// (prepended as an annotation block by the caller). HTTP(S) URLs become
/// plain-text references; missing files, out-of-scope paths, and copy failures
/// degrade to annotations while preserving the source file.
struct MultimodalVideoAction {
    replacement: String,
    delete_temp: bool,
    transcription: Option<String>,
}

impl MultimodalVideoAction {
    /// Plain annotation: no workspace copy, no transcription.
    fn annotation(replacement: String) -> Self {
        Self {
            replacement,
            delete_temp: false,
            transcription: None,
        }
    }
}

async fn handle_multimodal_video(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
) -> MultimodalVideoAction {
    if is_http_url(path) {
        return MultimodalVideoAction::annotation(format!("[Video: {path}]"));
    }
    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Video file not found for multimodal enrichment");
        return MultimodalVideoAction::annotation(format!("[Invalid video reference: {path}]"));
    }
    if !is_under_telegram_files(path_obj).await {
        tracing::warn!(%path, "Video path outside telegram temp dir — annotating without copy");
        return MultimodalVideoAction::annotation(format!(
            "[Video: {} attached]",
            extract_file_name(path)
        ));
    }
    if let Some(saved) = save_media_to_workspace(path_obj, uploads_dir, "video", "mp4").await {
        // Transcribe the persistent workspace copy — never the temp file
        // (deleted after a successful copy). The annotation keeps the original
        // Telegram filename; fail-open: any failure degrades to the plain
        // [Saved video: ...] annotation.
        let transcription = transcribe_saved_video(&saved.dest, extract_file_name(path)).await;
        return MultimodalVideoAction {
            replacement: saved.annotation,
            delete_temp: true,
            transcription,
        };
    }
    // Copy failed — annotate and preserve the temp file.
    MultimodalVideoAction::annotation(format!("[Video: {} attached]", extract_file_name(path)))
}

/// Transcribe a saved workspace video copy for the Artist, returning the
/// "[Video transcription of <name>]: <text>" annotation (using the original
/// source `file_name`). Fail-open: returns `None` (plain annotation) when the
/// transcription fails (unavailable transcriber, unsupported format, upload
/// or model error, timeout, empty output) — the overall timeout lives inside
/// [`transcribe_video_file`](crate::providers::transcribe_video_file),
/// bounding both callers.
async fn transcribe_saved_video(path: &std::path::Path, file_name: &str) -> Option<String> {
    let text = crate::providers::transcribe_video_file(path).await?;
    Some(format!("[Video transcription of {file_name}]: {text}"))
}

/// Handle an IMAGE marker in non-multimodal mode — transcribe to text
/// description or fall back to a generic attachment annotation.
async fn handle_non_multimodal_image(path_obj: &std::path::Path, file_name: &str) -> String {
    if let Some(ref transcriber) = crate::providers::media_transcriber() {
        match transcribe_image_file(path_obj, transcriber).await {
            Ok(description) => format!("[Image: {description}]"),
            Err(e) => {
                tracing::warn!(path = %path_obj.display(), error = %e, "Image transcription failed");
                format!("[Image: {file_name} attached]")
            }
        }
    } else {
        format!("[Image: {file_name} attached]")
    }
}

/// Extract the file name portion from a media marker path, falling back to
/// the raw path string if the path has no file name component.
fn extract_file_name(path: &str) -> &str {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

/// Process all media markers (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`)
/// in a single pass. Each marker kind is handled according to the strategy:
///
/// | Kind | Multimodal | NonMultimodal |
/// |------|-----------|---------------|
/// | IMAGE | data URI conversion, workspace copy | text transcription |
/// | AUDIO | transcription | transcription |
/// | VIDEO | workspace copy + `[Saved video: path]` + transcription | text annotation |
///
/// After processing, markers that were handled are stripped from the content
/// and annotations are prepended. Temp files are cleaned up after processing —
/// audio unconditionally (temp-dir scoped), video once copied to the workspace
/// — only the transcription (or icon) survives.
// Marker dispatch hub (3 kinds × 2 strategies); per-kind handling is extracted
// into the handler functions above, keeping this loop flat on purpose.
#[allow(clippy::too_many_lines)]
pub async fn enrich_message(msg: &mut ChannelMessage, strategy: &EnrichmentStrategy) {
    let mut annotations: Vec<String> = Vec::new();
    let mut result = msg.content.clone();
    // Accumulates upload path annotations across the for-loop below.
    // Only ever populated in Multimodal/IMAGE branch — always empty otherwise.
    let mut upload_annotations: Vec<String> = Vec::new();

    // Tracks temp files to remove after the loop; each kind decides its own
    // cleanup. Audio is unconditional (temp-dir scoped) — the file is a pure
    // intermediate artifact, only the transcription (or icon) survives.
    let mut files_to_delete: Vec<std::path::PathBuf> = Vec::new();

    let uploads_dir = match strategy {
        EnrichmentStrategy::Multimodal { workspace_path } => {
            workspace_path.as_ref().map(|p| p.join("uploads"))
        }
        EnrichmentStrategy::NonMultimodal => None,
    };

    for caps in MEDIA_MARKER_RE.captures_iter(&msg.content) {
        let whole = caps.get_match();
        let (kind, path) = parse_media_marker(&caps);
        let path_obj = std::path::Path::new(path);

        match kind {
            "IMAGE" => match strategy {
                EnrichmentStrategy::Multimodal { .. } => {
                    match handle_multimodal_image(path, path_obj, uploads_dir.as_deref()).await {
                        MultimodalImageAction::Keep => {
                            // HTTP/HTTPS URL — no local file to clean up.
                        }
                        MultimodalImageAction::Replace {
                            replacement,
                            upload_annotation,
                        } => {
                            result = result.replacen(whole.as_str(), &replacement, 1);
                            if let Some(ann) = upload_annotation {
                                upload_annotations.push(ann);
                            }
                            // Local IMAGE temp files are always cleaned up
                            // (legacy behaviour; image retries tracked in a follow-up).
                            files_to_delete.push(path_obj.to_path_buf());
                        }
                    }
                }
                EnrichmentStrategy::NonMultimodal => {
                    let file_name = extract_file_name(path);
                    annotations.push(handle_non_multimodal_image(path_obj, file_name).await);
                    // IMAGE temp files cleaned up regardless of outcome.
                    files_to_delete.push(path_obj.to_path_buf());
                }
            },
            "AUDIO" => {
                annotations.push(transcribe_audio_marker(path).await);
                // Audio temp files are always cleaned up, scoped to the daemon's
                // Telegram temp dir so user-typed [AUDIO:...] markers can never
                // delete arbitrary local files.
                if is_under_telegram_files(path_obj).await {
                    files_to_delete.push(path_obj.to_path_buf());
                }
            }
            "VIDEO" => match strategy {
                EnrichmentStrategy::Multimodal { .. } => {
                    let MultimodalVideoAction {
                        replacement,
                        delete_temp,
                        transcription,
                    } = handle_multimodal_video(path, path_obj, uploads_dir.as_deref()).await;
                    result = result.replacen(whole.as_str(), &replacement, 1);
                    if let Some(annotation) = transcription {
                        annotations.push(annotation);
                    }
                    if delete_temp {
                        files_to_delete.push(path_obj.to_path_buf());
                    }
                }
                EnrichmentStrategy::NonMultimodal => {
                    annotations.push(format!("[Video: {} attached]", extract_file_name(path)));
                    // VIDEO temp files are always cleaned up.
                    files_to_delete.push(path_obj.to_path_buf());
                }
            },
            // NOTE: If a new marker kind is added to MEDIA_MARKER_RE in
            // util/mod.rs, a corresponding arm MUST be added here for enrichment
            // behavior (transcription, annotation, etc.). The unified stripping
            // at the end of this function handles marker removal: in multimodal mode,
            // only IMAGE markers are preserved (all others are stripped); in
            // non-multimodal mode, all markers are stripped. The `_ =>` arm is
            // unreachable for well-formed markers (the regex only matches
            // IMAGE|AUDIO|VIDEO), but exists as a defensive guard during development.
            _ => {
                tracing::warn!(kind, %path, "Unknown media marker kind");
            }
        }
    }

    // ── File cleanup ────────────────────────────────────────────────
    // Temp files queued above are deleted here — per kind: audio
    // unconditionally, video only after a successful workspace copy.
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

    // ── Multimodal-specific post-processing ──
    // Append upload path annotations so the model can reference saved files.
    // `upload_annotations` accumulates across the for-loop; it is only ever
    // populated in Multimodal mode when a local IMAGE file was successfully
    // copied to the workspace uploads directory.
    if !upload_annotations.is_empty() {
        let annotation_block = upload_annotations.join("\n");
        let _ = write!(result, "\n\n{annotation_block}");
    }

    // ── Marker stripping and annotation prepending ──
    // Strip media markers from the enriched content. In multimodal mode,
    // IMAGE markers are preserved (needed for vision API integration via
    // to_message_content); all other markers are stripped. In non-multimodal
    // mode, all markers are stripped. The MEDIA_MARKER_PATTERN constant in
    // util/mod.rs is the single canonical source of truth for the marker
    // pattern; both MEDIA_MARKER_RE (case-sensitive) and TELEGRAM_MEDIA_MARKER_RE
    // (case-insensitive) are built from it to stay in sync.
    //
    // Note: using matches!() with a boolean guard means a future
    // EnrichmentStrategy variant would silently default to marker-stripping
    // (conservative behavior) rather than producing a compile error. This is
    // intentional — stripping unknown markers is the safe default.
    let keep_image = matches!(strategy, EnrichmentStrategy::Multimodal { .. });
    let cleaned = MEDIA_MARKER_RE
        .replace_all(&result, |caps: &regex::Captures| {
            if keep_image && parse_media_marker(caps).0 == "IMAGE" {
                caps.get_match().as_str().to_string()
            } else {
                String::new()
            }
        })
        .to_string();
    let cleaned = cleaned.trim().to_string();

    // ── Prepend text annotations (if any) ──
    // These are accumulated text descriptions for non-multimodal image files,
    // transcribed AUDIO content, and VIDEO annotations.
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
/// data URIs (multimodal IMAGE markers). Mixed audio+image/audio+video
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

/// Transcribe a local image file into a text description.
async fn transcribe_image_file(
    path: &std::path::Path,
    transcriber: &crate::providers::transcribe::MediaTranscriber,
) -> anyhow::Result<String> {
    if !path.exists() || !path.is_file() {
        anyhow::bail!("image file not found: {}", path.display());
    }

    let data_uri = crate::util::local_image_to_data_uri(path).await?;
    transcriber.transcribe(&data_uri).await
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

    #[tokio::test]
    async fn enrich_multimodal_image_http_url_passthrough() {
        let mut msg = test_msg("Check this [IMAGE:https://example.com/img.png] out");
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
        };
        enrich_message(&mut msg, &strategy).await;
        assert_eq!(
            msg.content,
            "Check this [IMAGE:https://example.com/img.png] out"
        );
    }

    #[tokio::test]
    async fn enrich_multimodal_image_file_not_found() {
        let mut msg = test_msg("Here is [IMAGE:/tmp/nonexistent_xyz_img.png] an image");
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
        };
        enrich_message(&mut msg, &strategy).await;
        assert!(
            msg.content
                .contains("[Invalid image reference: /tmp/nonexistent_xyz_img.png]")
        );
    }

    #[tokio::test]
    async fn enrich_multimodal_audio_annotation_and_strip() {
        let mut msg = test_msg("Listen [AUDIO:/tmp/audio_xyz.mp3] to this");
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
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
    async fn enrich_multimodal_image_valid_file_converts_to_data_uri_and_deletes_temp() {
        let tmp = std::env::temp_dir().join(format!("test_enrich_img_{}.png", std::process::id()));
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
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
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
    async fn enrich_multimodal_image_with_workspace_creates_upload_annotation() {
        let tmp_root = std::env::temp_dir().join(format!("test_enrich_ws_{}", std::process::id()));
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        let tmp_img =
            std::env::temp_dir().join(format!("test_enrich_ws_img_{}.png", std::process::id()));
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
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: Some(ws_path.clone()),
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
    async fn enrich_non_multimodal_image_annotation() {
        let mut msg = test_msg("Here is [IMAGE:/tmp/photo_xyz.jpg] from the camera");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // IMAGE marker stripped, annotation prepended (fallback since no transcriber)
        assert!(
            msg.content.contains("[Image:"),
            "Image annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[IMAGE:"),
            "IMAGE marker must be stripped"
        );
        assert!(msg.content.contains("from the camera"));
    }

    #[tokio::test]
    async fn enrich_non_multimodal_http_image_url_passthrough() {
        let mut msg = test_msg("Check [IMAGE:https://example.com/photo.png] online");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // HTTP image URL treated as attachment, annotation prepended
        assert!(
            msg.content.contains("[Image:"),
            "Image annotation must be present despite HTTP URL"
        );
        assert!(
            !msg.content.contains("[IMAGE:"),
            "IMAGE marker must be stripped"
        );
    }

    #[tokio::test]
    async fn enrich_non_multimodal_video_annotation() {
        let mut msg = test_msg("Watch [VIDEO:/tmp/clip_xyz.mp4] this video");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // VIDEO marker stripped, generic annotation prepended
        assert!(
            msg.content.contains("[Video: clip_xyz.mp4 attached]"),
            "Video annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
    }

    #[tokio::test]
    async fn enrich_multimodal_video_with_workspace_copies_and_annotates() {
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
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: Some(ws_path.clone()),
        };
        enrich_message(&mut msg, &strategy).await;

        // Marker replaced with a [Saved video: ...] annotation pointing at the
        // workspace uploads copy so the Artist can feed it to video_edit.
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
    async fn enrich_multimodal_video_outside_telegram_files_annotates_without_copy() {
        let tmp_root =
            std::env::temp_dir().join(format!("test_enrich_video_outside_{}", std::process::id()));
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // An injected marker pointing at an arbitrary readable file must not
        // be copied into uploads (exfiltration vector) or deleted.
        let arbitrary = tmp_root.join("secret.txt");
        tokio::fs::write(&arbitrary, b"top secret").await.unwrap();
        let marker = format!("Edit [VIDEO:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: Some(ws_path.clone()),
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
    async fn enrich_multimodal_video_http_url_kept_as_plain_text() {
        let mut msg = test_msg("Edit [VIDEO:https://example.com/clip.mp4] this");
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
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
    async fn enrich_non_multimodal_all_markers_stripped_and_annotated() {
        // _xyz-suffixed paths: nonexistent by convention, so the AUDIO branch
        // deterministically falls back to the annotation (never a loaded-model
        // transcription) and the IMAGE/VIDEO cleanup passes are no-ops.
        let mut msg = test_msg(
            "Check [IMAGE:/tmp/img_xyz.png] and listen [AUDIO:/tmp/audio_xyz.mp3] and watch [VIDEO:/tmp/vid_xyz.mp4]",
        );
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // All markers stripped
        assert!(!msg.content.contains("[IMAGE:"));
        assert!(!msg.content.contains("[AUDIO:"));
        assert!(!msg.content.contains("[VIDEO:"));
        // Annotations for all three
        assert!(msg.content.contains("[Image:"), "Image annotation missing");
        assert!(msg.content.contains("🔊✍️"), "Audio annotation missing");
        assert!(msg.content.contains("[Video:"), "Video annotation missing");
        // Original text preserved
        assert!(msg.content.contains("Check"));
        assert!(msg.content.contains("listen"));
        assert!(msg.content.contains("watch"));
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
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;

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
    async fn enrich_multimodal_combined_image_preserved_audio_annotated() {
        let msg_content = "Here [IMAGE:https://example.com/img.png] and [AUDIO:/tmp/sound_xyz.mp3]";
        let mut msg = test_msg(msg_content);
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
        };
        enrich_message(&mut msg, &strategy).await;

        // IMAGE http URL kept
        assert!(
            msg.content.contains("[IMAGE:https://example.com/img.png]"),
            "IMAGE with http URL must be preserved in multimodal mode, got: {}",
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
    async fn enrich_multimodal_no_annotations_when_no_markers() {
        assert_no_markers_unchanged(
            EnrichmentStrategy::Multimodal {
                workspace_path: None,
            },
            "Just a plain message with no markers",
        )
        .await;
    }

    #[tokio::test]
    async fn enrich_non_multimodal_no_annotations_when_no_markers() {
        assert_no_markers_unchanged(
            EnrichmentStrategy::NonMultimodal,
            "Plain text, no markers here",
        )
        .await;
    }
}
