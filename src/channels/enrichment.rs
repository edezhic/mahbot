//! Channel message enrichment: media marker processing, link enrichment, file
//! operations, and multimodal annotation strategies.
//!
//! This module transforms [`ChannelMessage`] content before it reaches the
//! agent pipeline. It handles:
//! - **Media markers** (`[IMAGE: ...]`, `[AUDIO: ...]`, `[VIDEO: ...]`)
//!   → transcription for audio, data URI conversion for images (multimodal
//!   strategy) or strippping with annotation (non-multimodal strategy)
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
use crate::util::{MEDIA_MARKER_RE, parse_media_marker};
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
/// the annotation text to embed in the message.
async fn transcribe_audio_marker(path: &str) -> String {
    let file_name = extract_file_name(path);

    if let Some(ref transcriber) = crate::providers::audio_transcriber() {
        match transcriber.transcribe(std::path::Path::new(path)).await {
            Ok(text) => format!("[Audio transcription of {file_name}]: {text}"),
            Err(e) => {
                tracing::warn!(%path, error = %e, "Audio transcription failed");
                format!("[Audio: {file_name} attached]")
            }
        }
    } else {
        format!("[Audio: {file_name} attached]")
    }
}

/// Save a copy of the image to `workspace/uploads/` for agent tool references.
/// Returns an annotation string on success, `None` if no workspace is configured
/// or I/O fails.
async fn save_image_to_workspace(
    image_path: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
) -> Option<String> {
    let dir = uploads_dir?;
    tokio::fs::create_dir_all(dir).await.ok()?;
    let ext = image_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png");
    let timestamp = crate::util::unix_millis();
    let dest_name = format!("upload_{timestamp}.{ext}");
    let dest_path = dir.join(&dest_name);
    tokio::fs::copy(image_path, &dest_path).await.ok()?;
    Some(format!("[Saved image: {}]", dest_path.display()))
}

/// Strategy for message enrichment, determining how each media marker kind
/// (IMAGE, AUDIO, VIDEO) is handled.
#[derive(Debug, Clone)]
pub enum EnrichmentStrategy {
    /// Multimodal mode:
    /// - IMAGE markers are converted to base64 data URIs (for vision model)
    /// - AUDIO markers are transcribed to text
    /// - VIDEO markers are stripped silently (no native video in chat completions)
    ///
    /// When `workspace_path` is provided, copies of images are saved to
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
    if path.starts_with("http://") || path.starts_with("https://") {
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
    let saved = save_image_to_workspace(path_obj, uploads_dir).await;

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

/// Handle an IMAGE marker in non-multimodal mode — transcribe to text
/// description or fall back to a generic attachment annotation.
async fn handle_non_multimodal_image(path_obj: &std::path::Path, file_name: &str) -> String {
    if let Some(ref transcriber) = crate::providers::image_transcriber() {
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
/// | VIDEO | stripped silently | text annotation |
///
/// After processing, markers that were handled are stripped from the content
/// and annotations are prepended. Audio/video/image temp files are deleted
/// after processing.
pub async fn enrich_message(msg: &mut ChannelMessage, strategy: &EnrichmentStrategy) {
    let mut annotations: Vec<String> = Vec::new();
    let mut result = msg.content.clone();
    // Accumulates upload path annotations across the for-loop below.
    // Only ever populated in Multimodal/IMAGE branch — always empty otherwise.
    let mut upload_annotations: Vec<String> = Vec::new();

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
                        MultimodalImageAction::Keep => {}
                        MultimodalImageAction::Replace {
                            replacement,
                            upload_annotation,
                        } => {
                            result = result.replacen(whole.as_str(), &replacement, 1);
                            if let Some(ann) = upload_annotation {
                                upload_annotations.push(ann);
                            }
                        }
                    }
                }
                EnrichmentStrategy::NonMultimodal => {
                    let file_name = extract_file_name(path);
                    let annotation = handle_non_multimodal_image(path_obj, file_name).await;
                    annotations.push(annotation);
                }
            },
            "AUDIO" => {
                let annotation = transcribe_audio_marker(path).await;
                annotations.push(annotation);
            }
            "VIDEO" => match strategy {
                EnrichmentStrategy::Multimodal { .. } => {
                    // No native video support in chat completions — strip silently.
                    // The marker will be stripped by the marker-stripping logic
                    // below (all non-IMAGE markers are removed in multimodal mode).
                }
                EnrichmentStrategy::NonMultimodal => {
                    let file_name = extract_file_name(path);
                    annotations.push(format!("[Video: {file_name} attached]"));
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

        // Single cleanup point — all media marker temp files are removed here,
        // regardless of kind or strategy. The helper functions (e.g.
        // handle_multimodal_image, handle_non_multimodal_image) no longer
        // perform cleanup themselves.
        let _ = tokio::fs::remove_file(path_obj).await;
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

/// Transcribe a local image file into a text description.
async fn transcribe_image_file(
    path: &std::path::Path,
    transcriber: &crate::providers::transcribe::ImageTranscriber,
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
    let urls = extract_urls(content);
    if urls.is_empty() {
        return Cow::Borrowed(content);
    }

    // Check chrome-use availability once before spawning concurrent work.
    if !BrowserTool::is_available().await {
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
                // Truncate very long text to keep messages manageable.
                const MAX_TEXT_LEN: usize = 5000;
                let snippet = if body_text.len() > MAX_TEXT_LEN {
                    let end = body_text.floor_char_boundary(MAX_TEXT_LEN);
                    format!("{}…", &body_text[..end])
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
            source_channel: "test".into(),
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
        // AUDIO marker stripped; annotation prepended (fallback since no audio transcriber)
        assert!(
            msg.content.contains("[Audio:"),
            "Audio annotation must be present"
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
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
        let mut msg = test_msg("Here is [IMAGE:/tmp/photo.jpg] from the camera");
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
        let mut msg = test_msg("Watch [VIDEO:/tmp/clip.mp4] this video");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // VIDEO marker stripped, generic annotation prepended
        assert!(
            msg.content.contains("[Video: clip.mp4 attached]"),
            "Video annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
    }

    #[tokio::test]
    async fn enrich_non_multimodal_all_markers_stripped_and_annotated() {
        let mut msg = test_msg(
            "Check [IMAGE:/tmp/img.png] and listen [AUDIO:/tmp/audio.mp3] and watch [VIDEO:/tmp/vid.mp4]",
        );
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // All markers stripped
        assert!(!msg.content.contains("[IMAGE:"));
        assert!(!msg.content.contains("[AUDIO:"));
        assert!(!msg.content.contains("[VIDEO:"));
        // Annotations for all three
        assert!(msg.content.contains("[Image:"), "Image annotation missing");
        assert!(msg.content.contains("[Audio:"), "Audio annotation missing");
        assert!(msg.content.contains("[Video:"), "Video annotation missing");
        // Original text preserved
        assert!(msg.content.contains("Check"));
        assert!(msg.content.contains("listen"));
        assert!(msg.content.contains("watch"));
    }

    #[tokio::test]
    async fn enrich_audio_file_deleted_after_transcription() {
        let tmp =
            std::env::temp_dir().join(format!("test_enrich_audio_{}.mp3", std::process::id()));
        tokio::fs::write(&tmp, b"fake audio content").await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Audio: [AUDIO:{path_str}]"));
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;

        // Temp file must be deleted after processing
        assert!(!tmp.exists(), "Audio temp file must be deleted");
    }

    #[tokio::test]
    async fn enrich_multimodal_combined_image_preserved_audio_video_stripped() {
        let msg_content = "Here [IMAGE:https://example.com/img.png] and [AUDIO:/tmp/sound.mp3] and [VIDEO:/tmp/clip.mp4]";
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
            msg.content.contains("[Audio:"),
            "Audio annotation must be present"
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
        // VIDEO stripped silently with no annotation
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
        assert!(
            !msg.content.contains("[Video:"),
            "No video annotation in multimodal mode (silent strip)"
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
