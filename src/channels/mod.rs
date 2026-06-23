pub mod gui;
pub mod telegram;
use crate::tools::browser::BrowserTool;
use crate::turso;
use crate::util::MEDIA_MARKER_RE;
use crate::{ChannelMessage, ChatDirection, SendMessage};
use regex::Regex;
use std::collections::HashSet;
use std::fmt::Write;
use std::sync::LazyLock;
use tokio_util::sync::CancellationToken;

const CHANNEL_TYPING_REFRESH_INTERVAL_SECS: u64 = 4;

/// Entry for a single chat message that should be both broadcast to the GUI
/// dashboard and persisted to chat_history. Fields map directly to the
/// [`crate::ChatEvent::Message`] and [`crate::chat_history::ChatHistoryStore::insert`] parameters.
#[derive(Debug, Clone)]
struct BroadcastPersistEntry {
    user_name: String,
    channel_name: String,
    content: String,
    direction: ChatDirection,
    agent_role: Option<String>,
    workspace: String,
    optimistic_id: Option<String>,
    reply_markup: Option<serde_json::Value>,
}

impl BroadcastPersistEntry {
    /// Broadcast this entry to [`crate::CHAT_BROADCAST`] and persist it to
    /// `chat_history`.
    async fn broadcast_and_persist(self) {
        use crate::ChatEvent;

        // Invariant: direction=Agent must carry a non-None agent_role
        debug_assert!(
            self.direction != ChatDirection::Agent || self.agent_role.is_some(),
            "BroadcastPersistEntry: direction=Agent but agent_role is None"
        );

        let message_id = crate::generate_id();
        let timestamp = turso::now();

        let (db_role, db_direction) = match self.direction {
            ChatDirection::Agent => (self.agent_role.as_deref().unwrap_or(""), "agent"),
            ChatDirection::User => ("user", "user"),
        };

        if let Some(tx) = crate::CHAT_BROADCAST.get() {
            let _ = tx.send(ChatEvent::Message {
                message_id: message_id.clone(),
                user_name: self.user_name.clone(),
                content: self.content.clone(),
                direction: self.direction,
                timestamp: timestamp.clone(),
                agent_role: self.agent_role.clone(),
                workspace: self.workspace.clone(),
                optimistic_id: self.optimistic_id,
                reply_markup: self.reply_markup,
            });
        }

        let store = crate::chat_history::store();
        let _ = store
            .insert(
                &message_id,
                &self.user_name,
                &self.channel_name,
                db_role,
                db_direction,
                &self.content,
                self.agent_role.as_deref(),
                &self.workspace,
                &timestamp,
            )
            .await;
    }
}

/// Broadcast an agent response to CHAT_BROADCAST for live GUI display and
/// persist it to chat_history. This is the canonical entry point for all
/// agent responses — both the non-Manager path
/// ([`send_channel_reply_with_buttons`]) and the Manager queue consumer
/// in [`crate::manager_queue`].
///
/// Takes explicit `user_name` (canonical user name), `channel` (e.g. "telegram", "gui"),
/// and primitive fields — does **not** depend on [`SendMessage`], so it can be used
/// from the Manager queue which works from [`crate::users::UserRecord`].
pub async fn broadcast_and_persist_agent_response(
    user_name: &str,
    channel: &str,
    content: &str,
    agent_role: Option<String>,
    workspace: &str,
    reply_markup: Option<serde_json::Value>,
) {
    BroadcastPersistEntry {
        user_name: user_name.to_string(),
        channel_name: channel.to_string(),
        content: content.to_string(),
        direction: ChatDirection::Agent,
        agent_role,
        workspace: workspace.to_string(),
        optimistic_id: None, // agent messages must not carry one
        reply_markup,
    }
    .broadcast_and_persist()
    .await;
}

/// Write an incoming user message to CHAT_BROADCAST for immediate GUI display
/// and persist it to chat_history. Uses `msg.source_channel` for the channel
/// field so it works for both Telegram and GUI-originated messages.
pub async fn write_incoming_to_broadcast(msg: &ChannelMessage) {
    BroadcastPersistEntry {
        user_name: msg.user_name.clone(),
        channel_name: msg.source_channel.clone(),
        content: msg.content.clone(),
        direction: ChatDirection::User,
        agent_role: None, // user messages have no agent role
        workspace: msg.workspace.clone(),
        optimistic_id: msg.message_id.clone(), // GUI uses this for replacement
        reply_markup: None,                    // user messages have no reply markup
    }
    .broadcast_and_persist()
    .await;
}

/// Send a reply through a channel directly.
pub async fn send_channel_reply(content: String, msg: &ChannelMessage) {
    send_channel_reply_with_buttons(content, msg, None, None).await;
}

/// Send a reply through a channel with an inline keyboard. When `buttons` is `Some`,
/// the reply is rendered with that inline keyboard.
pub async fn send_channel_reply_with_buttons(
    content: String,
    msg: &ChannelMessage,
    buttons: Option<Vec<serde_json::Value>>,
    agent_role: Option<String>,
) {
    let Some(channel) = crate::channel_registry().get(&msg.source_channel) else {
        tracing::warn!(
            source_channel = %msg.source_channel,
            "Channel not found in registry — cannot send reply"
        );
        return;
    };
    let reply_markup = buttons.map(|b| serde_json::json!({ "inline_keyboard": [b] }));

    // ── Broadcast agent response for live GUI display and chat_history ──
    broadcast_and_persist_agent_response(
        &msg.user_name,
        &msg.source_channel,
        &content,
        agent_role.clone(),
        &msg.workspace,
        reply_markup.clone(),
    )
    .await;

    let reply = SendMessage {
        content,
        recipient: msg.reply_target.clone(),
        reply_markup,
        agent_role,
        workspace: msg.workspace.clone(),
    };

    if let Err(e) = channel.send(&reply).await {
        tracing::error!("Failed to reply on {}: {e}", channel.name());
    }
}

#[must_use]
pub fn spawn_scoped_typing_task(
    recipient: String,
    source_channel: String,
    cancellation_token: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let stop_signal = cancellation_token;
    let refresh_interval = std::time::Duration::from_secs(CHANNEL_TYPING_REFRESH_INTERVAL_SECS);
    tokio::spawn(async move {
        let Some(channel) = crate::channel_registry().get(&source_channel) else {
            tracing::warn!(
                source_channel = %source_channel,
                "Channel not found in registry — skipping typing indicator"
            );
            return;
        };
        let mut interval = tokio::time::interval(refresh_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                () = stop_signal.cancelled() => break,
                _ = interval.tick() => {
                    if let Err(e) = channel.start_typing(&recipient).await {
                        tracing::debug!("Failed to start typing on {}: {e}", channel.name());
                    }
                }
            }
        }
    })
}

/// Cancel the typing task (via token) and await its completion.
pub async fn stop_typing(handle: tokio::task::JoinHandle<()>) {
    if let Err(error) = handle.await {
        tracing::error!("Typing task crashed: {error}");
    }
}

/// URL regex: matches http:// and https:// URLs, stopping at whitespace, angle
/// brackets, or double-quotes.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"']+"#).expect("URL regex must compile"));

/// Transcribe an audio file referenced by a `[AUDIO:...]` marker and return
/// the annotation text to embed in the message.
async fn transcribe_audio_marker(path: &str) -> String {
    let file_name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);

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
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
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
/// Always deletes the source temp file.
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
        remove_temp_file(path_obj).await;
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

    remove_temp_file(path_obj).await;
    MultimodalImageAction::Replace {
        replacement,
        upload_annotation: saved,
    }
}

/// Handle an IMAGE marker in non-multimodal mode — transcribe to text
/// description or fall back to a generic attachment annotation.
async fn handle_non_multimodal_image(path_obj: &std::path::Path, file_name: &str) -> String {
    let annotation = if let Some(ref transcriber) = crate::providers::image_transcriber() {
        match transcribe_image_file(path_obj, transcriber).await {
            Ok(description) => format!("[Image: {description}]"),
            Err(e) => {
                tracing::warn!(path = %path_obj.display(), error = %e, "Image transcription failed");
                format!("[Image: {file_name} attached]")
            }
        }
    } else {
        format!("[Image: {file_name} attached]")
    };
    remove_temp_file(path_obj).await;
    annotation
}

/// Extract the file name portion from a media marker path, falling back to
/// the raw path string if the path has no file name component.
fn extract_file_name(path: &str) -> &str {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

/// Delete a temporary file, ignoring errors (best-effort cleanup).
async fn remove_temp_file(path: &std::path::Path) {
    let _ = tokio::fs::remove_file(path).await;
}

/// Apply enrichment post-processing: strip markers according to strategy
/// and prepend text annotations. This is a pure function that takes the
/// content (with any replacements already applied) and returns the final
/// content string.
fn finish_enrichment(
    annotations: &[String],
    strategy: &EnrichmentStrategy,
    content: &str,
) -> String {
    // ── Strip markers ──
    // In multimodal mode, IMAGE markers are preserved (needed for vision API
    // integration via to_message_content); all other markers are stripped.
    // In non-multimodal mode, all markers are stripped. The MEDIA_MARKER_RE
    // is the single canonical source of truth for all media marker patterns.
    //
    // Note: using matches!() with a boolean guard means a future
    // EnrichmentStrategy variant would silently default to marker-stripping
    // (conservative behavior) rather than producing a compile error. This is
    // intentional — stripping unknown markers is the safe default.
    let keep_image = matches!(strategy, EnrichmentStrategy::Multimodal { .. });
    let cleaned = MEDIA_MARKER_RE
        .replace_all(content, |caps: &regex::Captures| {
            if keep_image && caps.get(1).unwrap().as_str() == "IMAGE" {
                caps.get(0).unwrap().as_str().to_string()
            } else {
                String::new()
            }
        })
        .to_string();
    let cleaned = cleaned.trim().to_string();

    // ── Prepend annotations (if any) ──
    if annotations.is_empty() {
        cleaned
    } else {
        let prefix = annotations.join("\n");
        if cleaned.is_empty() {
            prefix
        } else {
            format!("{prefix}\n\n{cleaned}")
        }
    }
}

/// Enrich an inbound message according to the provided strategy.
///
/// Processes all media markers (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`)
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
    // Only populated/used in Multimodal mode — always empty in NonMultimodal.
    let mut upload_annotations: Vec<String> = Vec::new();

    let uploads_dir = match strategy {
        EnrichmentStrategy::Multimodal { workspace_path } => {
            workspace_path.as_ref().map(|p| p.join("uploads"))
        }
        EnrichmentStrategy::NonMultimodal => None,
    };

    for caps in MEDIA_MARKER_RE.captures_iter(&msg.content) {
        let whole = caps.get(0).unwrap();
        let kind = caps.get(1).unwrap().as_str();
        let path = caps.get(2).unwrap().as_str();
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
                remove_temp_file(path_obj).await;
            }
            "VIDEO" => match strategy {
                EnrichmentStrategy::Multimodal { .. } => {
                    // No native video support in chat completions — strip silently.
                    // The marker will be stripped by finish_enrichment below
                    // (all non-IMAGE markers are removed in multimodal mode).
                    remove_temp_file(path_obj).await;
                }
                EnrichmentStrategy::NonMultimodal => {
                    let file_name = extract_file_name(path);
                    annotations.push(format!("[Video: {file_name} attached]"));
                    remove_temp_file(path_obj).await;
                }
            },
            // NOTE: If a new marker kind is added to MEDIA_MARKER_RE in
            // util/mod.rs, a corresponding arm MUST be added here for enrichment
            // behavior (transcription, annotation, etc.). The unified stripping
            // in `finish_enrichment` handles marker removal: in multimodal mode,
            // only IMAGE markers are preserved (all others are stripped); in
            // non-multimodal mode, all markers are stripped. The `_ =>` arm is
            // unreachable for well-formed markers (the regex only matches
            // IMAGE|AUDIO|VIDEO), but exists as a defensive guard during development.
            _ => {
                tracing::warn!(kind, %path, "Unknown media marker kind");
            }
        }
    }

    // ── Multimodal-specific post-processing ──
    // Append upload path annotations so the model can reference saved files.
    // Invariant: upload_annotations is only populated in Multimodal mode,
    // when a local IMAGE file was successfully copied to the workspace uploads
    // directory.
    if !upload_annotations.is_empty() {
        let annotation_block = upload_annotations.join("\n");
        let _ = write!(result, "\n\n{annotation_block}");
    }

    msg.content = finish_enrichment(&annotations, strategy, &result);
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
/// gets its own isolated session tab that is closed after snapshot extraction.
pub async fn enrich_links(content: &str) -> String {
    let urls = extract_urls(content);
    if urls.is_empty() {
        return content.to_string();
    }

    // Check agent-browser availability once before spawning concurrent work.
    if !BrowserTool::is_available().await {
        tracing::debug!("agent-browser not available, skipping link enrichment");
        return content.to_string();
    }

    // Fetch all URLs concurrently.
    let browser = std::sync::Arc::new(BrowserTool::default());
    let mut tasks = Vec::with_capacity(urls.len());
    for (i, url) in urls.iter().enumerate() {
        let url = url.clone();
        let tab = format!("link-enricher-{i}");
        let browser = std::sync::Arc::clone(&browser);
        tasks.push(tokio::spawn(async move {
            let result = browser.fetch_snapshot(&url, &tab).await;
            // Close the tab (best-effort) regardless of fetch outcome.
            browser.close_session(&tab).await;
            (url, result)
        }));
    }

    let mut enrichments: Vec<String> = Vec::new();
    for task in tasks {
        match task.await {
            Ok((url, Ok(snapshot))) => {
                // Truncate very long snapshots to keep messages manageable.
                const MAX_SNAPSHOT_LEN: usize = 5000;
                let snippet = if snapshot.len() > MAX_SNAPSHOT_LEN {
                    let end = snapshot.floor_char_boundary(MAX_SNAPSHOT_LEN);
                    format!("{}…", &snapshot[..end])
                } else {
                    snapshot
                };
                enrichments.push(format!("📄 [{url}]\n{snippet}"));
            }
            Ok((url, Err(e))) => {
                tracing::debug!(url, error = %e, "Link enricher: failed to fetch page snapshot");
            }
            Err(e) => {
                tracing::debug!("Link enricher task panicked: {e}");
            }
        }
    }

    if enrichments.is_empty() {
        return content.to_string();
    }

    let prefix = enrichments.join("\n\n");
    format!("{prefix}\n\n{content}")
}

/// URL extraction for messages with links that should be enriched.
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

    // ── Enrichment strategy tests ─────────────────────────────────────

    /// Helper: quick ChannelMessage for enrichment tests.
    fn test_msg(content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: "test".into(),
            reply_target: "test".into(),
            content: content.to_string(),
            source_channel: "test".into(),
            workspace: "test".into(),
            message_id: None,
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
        let mut msg = test_msg("Here is [IMAGE:/tmp/nonexistent_test_img.png] an image");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        assert!(
            msg.content.contains("[Image:"),
            "Image annotation must be present"
        );
        assert!(
            !msg.content.contains("[IMAGE:"),
            "IMAGE marker must be stripped"
        );
        assert!(msg.content.contains("Here is"), "Original text preserved");
        assert!(msg.content.contains("an image"), "Original text preserved");
    }

    #[tokio::test]
    async fn enrich_non_multimodal_http_image_url_passthrough() {
        // HTTP image URLs in non-multimodal mode: the marker is stripped (since
        // all markers are stripped in non-multimodal), and a fallback annotation
        // is produced (the file can't be transcribed remotely — same behavior
        // as any non-existent local path).
        let mut msg = test_msg("[IMAGE:https://example.com/photo.png]");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        assert!(!msg.content.contains("[IMAGE:"), "IMAGE marker stripped");
        // Fallback annotation: the filename portion "photo.png" is used
        assert!(
            msg.content.contains("[Image:"),
            "Fallback image annotation produced"
        );
        assert!(msg.content.contains("photo.png"), "Filename in annotation");
    }

    #[tokio::test]
    async fn enrich_non_multimodal_video_annotation() {
        let mut msg = test_msg("Check [VIDEO:/tmp/nonexistent_test_video.mp4] out");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        assert!(
            msg.content.contains("[Video:"),
            "Video annotation must be present"
        );
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
    }

    #[tokio::test]
    async fn enrich_non_multimodal_all_markers_stripped_and_annotated() {
        let mut msg =
            test_msg("Img [IMAGE:/tmp/i.png] Aud [AUDIO:/tmp/a.mp3] Vid [VIDEO:/tmp/v.mp4] end");
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        // All raw markers stripped
        assert!(!msg.content.contains("[IMAGE:"), "IMAGE marker stripped");
        assert!(!msg.content.contains("[AUDIO:"), "AUDIO marker stripped");
        assert!(!msg.content.contains("[VIDEO:"), "VIDEO marker stripped");
        // Annotations present (image + audio + video)
        assert!(msg.content.contains("[Image:"), "Image annotation");
        assert!(msg.content.contains("[Audio:"), "Audio annotation");
        assert!(msg.content.contains("[Video:"), "Video annotation");
        // Original text preserved
        assert!(msg.content.contains("Img"), "Original text preserved");
        assert!(msg.content.contains("Aud"), "Original text preserved");
        assert!(msg.content.contains("Vid"), "Original text preserved");
        assert!(msg.content.contains("end"), "Original text preserved");
    }

    #[tokio::test]
    async fn enrich_audio_file_deleted_after_transcription() {
        // Create a temp audio file
        let tmp =
            std::env::temp_dir().join(format!("test_enrich_audio_{}.mp3", std::process::id()));
        tokio::fs::write(&tmp, b"fake audio content").await.unwrap();
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = test_msg(&format!("Audio: [AUDIO:{path_str}]"));
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;

        // Audio temp file must be deleted
        assert!(
            !tmp.exists(),
            "Audio temp file must be deleted after enrichment"
        );
    }

    #[tokio::test]
    async fn enrich_multimodal_strips_video_but_not_image_markers() {
        // Verify that multimodal mode preserves IMAGE markers (needed for
        // to_message_content) while stripping VIDEO markers.
        let mut msg =
            test_msg("before [IMAGE:https://example.com/i.png] mid [VIDEO:/tmp/v.mp4] after");
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
        };
        enrich_message(&mut msg, &strategy).await;

        // IMAGE marker preserved (HTTP URL)
        assert!(
            msg.content.contains("[IMAGE:https://example.com/i.png]"),
            "IMAGE HTTP URL must be preserved in multimodal"
        );
        // VIDEO marker stripped
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
        assert!(
            !msg.content.contains("[Video:"),
            "No Video annotation in multimodal"
        );
        // Text preserved
        assert!(msg.content.contains("before"), "Text preserved");
        assert!(msg.content.contains("mid"), "Text preserved");
        assert!(msg.content.contains("after"), "Text preserved");
    }

    #[tokio::test]
    async fn enrich_multimodal_combined_image_preserved_audio_video_stripped() {
        // Verify the multimodal invariant: all three marker kinds in one message,
        // IMAGE is preserved, AUDIO and VIDEO are stripped (with annotations).
        let mut msg = test_msg(
            "Start [IMAGE:https://example.com/i.png] mid [AUDIO:/tmp/a.mp3] end [VIDEO:/tmp/v.mp4]",
        );
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
        };
        enrich_message(&mut msg, &strategy).await;

        // IMAGE HTTP URL preserved unchanged
        assert!(
            msg.content.contains("[IMAGE:https://example.com/i.png]"),
            "IMAGE HTTP URL must be preserved in multimodal"
        );
        // AUDIO marker stripped, annotation present (fallback transcription)
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
        assert!(
            msg.content.contains("[Audio:"),
            "Audio annotation must be present"
        );
        // VIDEO marker stripped, no video annotation in multimodal
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
        assert!(
            !msg.content.contains("[Video:"),
            "No Video annotation in multimodal"
        );
        // Text preserved
        assert!(msg.content.contains("Start"), "Text preserved");
        assert!(msg.content.contains("mid"), "Text preserved");
        assert!(msg.content.contains("end"), "Text preserved");
    }

    #[tokio::test]
    async fn enrich_multimodal_no_annotations_when_no_markers() {
        let mut msg = test_msg("Hello, this is a plain text message with no markers");
        let original = msg.content.clone();
        let strategy = EnrichmentStrategy::Multimodal {
            workspace_path: None,
        };
        enrich_message(&mut msg, &strategy).await;
        assert_eq!(msg.content, original, "No markers = no changes");
    }

    #[tokio::test]
    async fn enrich_non_multimodal_no_annotations_when_no_markers() {
        let mut msg = test_msg("Plain text, no markers here");
        let original = msg.content.clone();
        enrich_message(&mut msg, &EnrichmentStrategy::NonMultimodal).await;
        assert_eq!(msg.content, original, "No markers = no changes");
    }
}
