pub mod gui;
pub mod telegram;
use crate::chat_history::ChatHistoryInsert;
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
/// [`crate::ChatEvent::Message`] and [`ChatHistoryInsert`] parameters.
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
            .insert(&ChatHistoryInsert {
                message_id: message_id.clone(),
                user_name: self.user_name.clone(),
                channel: self.channel_name.clone(),
                role: db_role.to_string(),
                direction: db_direction.to_string(),
                content: self.content.clone(),
                agent_role: self.agent_role.clone(),
                workspace: self.workspace.clone(),
                created_at: timestamp.clone(),
            })
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
            if keep_image
                && caps
                    .name("kind")
                    .expect("MEDIA_MARKER_RE: expected 'kind' group")
                    .as_str()
                    == "IMAGE"
            {
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
        let kind = caps
            .name("kind")
            .expect("MEDIA_MARKER_RE: expected 'kind' group")
            .as_str();
        let path = caps
            .name("path")
            .expect("MEDIA_MARKER_RE: expected 'path' group")
            .as_str();
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
/// gets its own isolated session tab that is closed after text extraction.
pub async fn enrich_links(content: &str) -> String {
    let urls = extract_urls(content);
    if urls.is_empty() {
        return content.to_string();
    }

    // Check chrome-use availability once before spawning concurrent work.
    if !BrowserTool::is_available().await {
        tracing::debug!("chrome-use not available, skipping link enrichment");
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
        return content.to_string();
    }

    let prefix = enrichments.join("\n\n");
    format!("{prefix}\n\n{content}")
}

/// Mirror a GUI-originated user message to the user's Telegram chat(s)
/// as a blockquote, so conversation history is readable from both surfaces.
///
/// This should be called before enrichment to preserve the original
/// user-typed text (pre-link-summary, pre-transcription).
///
/// # Guards
///
/// * Only mirrors messages where `source_channel == "gui"` (prevents echo loops).
/// * Skips empty or whitespace-only messages.
/// * Silently returns when no Telegram channel is registered or the user has no
///   Telegram binding with a `reply_target` (no error, no crash).
/// * Sends to **all** Telegram bindings if the user has multiple.
///
/// # Quote format
///
/// Uses `<blockquote>` HTML tags, which `markdown_to_telegram_html` in the
/// Telegram channel's `send()` pipeline passes through unchanged. The user's
/// text retains markdown formatting through the standard inline parser.
/// Media markers (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`) are stripped
/// so raw marker syntax does not appear in the quote; purely media-only
/// messages are skipped entirely.
pub async fn mirror_gui_message_to_telegram(msg: &ChannelMessage) {
    // Guard: only mirror GUI-originated user messages (prevents echo loops).
    if msg.source_channel != "gui" {
        return;
    }

    // Guard: skip empty or whitespace-only messages.
    let trimmed = msg.content.trim();
    if trimmed.is_empty() {
        return;
    }

    // Guard: Telegram channel must be available.
    let Some(channel) = crate::channel_registry().get("telegram") else {
        return;
    };

    // Look up the user's channel bindings.
    let bindings = match crate::users::store()
        .get_user_channels(&msg.user_name)
        .await
    {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                user = %msg.user_name,
                error = %e,
                "Failed to look up user channels for GUI message mirror"
            );
            return;
        }
    };

    // Filter to Telegram bindings (reply_target checked per binding below).
    let telegram_bindings: Vec<_> = bindings
        .into_iter()
        .filter(|b| b.channel == "telegram")
        .collect();

    if telegram_bindings.is_empty() {
        return; // No Telegram binding — silently skip.
    }

    // Strip media markers so users don't see raw `[IMAGE:...]` syntax in the quote.
    let content = MEDIA_MARKER_RE.replace_all(trimmed, "").to_string();
    let content = content.trim().to_string();
    if content.is_empty() {
        return; // Media-only message — nothing to quote.
    }

    // Wrap in <blockquote> — these tags pass through markdown_to_telegram_html
    // unchanged, while the user's text retains markdown formatting.
    let quoted = format!("<blockquote>\n{content}\n</blockquote>");

    for binding in &telegram_bindings {
        let Some(reply_target) = &binding.reply_target else {
            continue; // skip bindings without a reply target
        };
        let reply = SendMessage {
            content: quoted.clone(),
            recipient: reply_target.clone(),
            reply_markup: None,
            agent_role: None,
            workspace: String::new(),
        };

        if let Err(e) = channel.send(&reply).await {
            tracing::error!(
                user = %msg.user_name,
                recipient = %reply_target,
                error = %e,
                "Failed to mirror GUI message to Telegram"
            );
        }
    }
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

    // ── GUI message → Telegram mirror tests ──────────────────────

    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    /// Serialization lock for all mirror tests — these tests share the global
    /// [`CHANNEL_REGISTRY`] and store singletons, so they must run one at a time.
    /// Uses `tokio::sync::Mutex` to avoid blocking worker threads while held
    /// across await points.
    static MIRROR_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

    async fn acquire_mirror_lock() -> tokio::sync::MutexGuard<'static, ()> {
        MIRROR_TEST_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await
    }

    use crate::util::UnwrapPoison;

    /// A spy channel that records sent messages in a shared Vec.
    struct SpyChannel {
        sent: Arc<Mutex<Vec<SendMessage>>>,
    }

    #[async_trait]
    impl crate::Channel for SpyChannel {
        async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
            self.sent.lock().unwrap_poison().push(message.clone());
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn name(&self) -> &str {
            "telegram"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Set up the channel registry with a spy Telegram channel and return a
    /// shared sent-messages buffer. Idempotent — safe to call from every test.
    fn setup_spy_channel() -> &'static Arc<Mutex<Vec<SendMessage>>> {
        static SPY_SENT: OnceLock<Arc<Mutex<Vec<SendMessage>>>> = OnceLock::new();
        SPY_SENT.get_or_init(|| {
            let sent = Arc::new(Mutex::new(Vec::new()));
            let registry = crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
            registry.register(Arc::new(SpyChannel {
                sent: Arc::clone(&sent),
            }) as Arc<dyn crate::Channel>);
            sent
        })
    }

    /// Ensure the user store has a test user with a Telegram binding and
    /// reply_target. Idempotent.
    async fn setup_user_with_telegram_binding(user_name: &str, reply_target: &str) {
        use crate::users::store;
        let store = store();
        store
            .add_user(user_name, Some("full"))
            .await
            .expect("add_user");
        store
            .bind_channel(user_name, "telegram", user_name)
            .await
            .expect("bind_channel");
        store
            .update_channel_contact("telegram", user_name, reply_target)
            .await
            .expect("update_channel_contact");
    }

    fn gui_msg(user_name: &str, content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: user_name.to_string(),
            reply_target: String::new(),
            content: content.to_string(),
            source_channel: "gui".to_string(),
            workspace: "test".to_string(),
            message_id: None,
            callback_query_id: None,
        }
    }

    fn telegram_msg(user_name: &str, content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: user_name.to_string(),
            reply_target: "chat:thread".to_string(),
            content: content.to_string(),
            source_channel: "telegram".to_string(),
            workspace: "test".to_string(),
            message_id: None,
            callback_query_id: None,
        }
    }

    // ── Guard tests: early-return conditions ─────────────────────
    //
    // These tests verify that `mirror_gui_message_to_telegram` returns
    // early (without sending) for each guard condition. They are serialized
    // via [`MIRROR_TEST_LOCK`] because the channel registry and store
    // singletons are global. Each uses a unique reply target so assertions
    // filter only the current test's messages from the shared spy buffer.

    #[tokio::test]
    async fn skip_non_gui_source() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("skip_telegram", "target_non_gui").await;

        let msg = telegram_msg("skip_telegram", "hello from telegram");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_non_gui")
            .collect();
        assert!(our_msgs.is_empty(), "non-GUI source should not send");
    }

    #[tokio::test]
    async fn skip_empty_content() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("skip_empty", "target_empty").await;

        let msg = gui_msg("skip_empty", "");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_empty")
            .collect();
        assert!(our_msgs.is_empty(), "empty content should not send");
    }

    #[tokio::test]
    async fn skip_whitespace_content() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("skip_ws", "target_ws").await;

        let msg = gui_msg("skip_ws", "   \t\n  ");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_ws")
            .collect();
        assert!(
            our_msgs.is_empty(),
            "whitespace-only content should not send"
        );
    }

    #[tokio::test]
    async fn skip_user_with_no_bindings() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        // Create user but DO NOT bind a Telegram channel.
        let store = crate::users::store();
        store.add_user("no_binding", None).await.expect("add_user");

        // Use the user's name as the recipient filter — no bindings means
        // no messages should be sent for this user at all.
        let user_name = "no_binding";
        let msg = gui_msg(user_name, "hello");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard.iter().filter(|m| m.recipient == user_name).collect();
        assert!(our_msgs.is_empty(), "user with no bindings should not send");
    }

    #[tokio::test]
    async fn skip_binding_without_reply_target() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        // Bind a Telegram channel but don't set reply_target.
        let store = crate::users::store();
        store.add_user("no_target", None).await.expect("add_user");
        store
            .bind_channel("no_target", "telegram", "no_target")
            .await
            .expect("bind_channel");
        // Note: skip update_channel_contact → reply_target stays NULL.

        let msg = gui_msg("no_target", "hello");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "no_target")
            .collect();
        assert!(
            our_msgs.is_empty(),
            "binding without reply_target should not send"
        );
    }

    #[tokio::test]
    async fn skip_media_only_content() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("media_only", "target_media").await;

        let msg = gui_msg("media_only", "[IMAGE:/path/to/img.png]");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_media")
            .collect();
        assert!(our_msgs.is_empty(), "media-only content should not send");
    }

    // ── Happy path tests ─────────────────────────────────────────

    #[tokio::test]
    async fn sends_blockquote_to_single_binding() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("single_user", "unique_single").await;

        let msg = gui_msg("single_user", "Hello, world!");
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        // Filter to our test's messages by recipient.
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_single")
            .collect();
        assert_eq!(our_msgs.len(), 1, "expected exactly one message");
        assert_eq!(
            our_msgs[0].content,
            "<blockquote>\nHello, world!\n</blockquote>"
        );
        assert!(our_msgs[0].reply_markup.is_none());
        assert!(our_msgs[0].agent_role.is_none());
    }

    #[tokio::test]
    async fn sends_to_multiple_telegram_bindings() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        let store = crate::users::store();
        store.add_user("multi_user", None).await.expect("add_user");
        // Bind two Telegram accounts with unique recipients.
        store
            .bind_channel("multi_user", "telegram", "multi_user_1")
            .await
            .expect("bind_channel_1");
        store
            .bind_channel("multi_user", "telegram", "multi_user_2")
            .await
            .expect("bind_channel_2");
        store
            .update_channel_contact("telegram", "multi_user_1", "unique_multi_a")
            .await
            .expect("update_channel_contact_1");
        store
            .update_channel_contact("telegram", "multi_user_2", "unique_multi_b")
            .await
            .expect("update_channel_contact_2");

        let msg = gui_msg("multi_user", "Hi both!");
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_multi_a" || m.recipient == "unique_multi_b")
            .collect();
        assert_eq!(our_msgs.len(), 2, "expected two messages (one per binding)");
        // Both should have the same content.
        for m in &our_msgs {
            assert_eq!(m.content, "<blockquote>\nHi both!\n</blockquote>");
        }
        let recipients: Vec<&str> = our_msgs.iter().map(|m| m.recipient.as_str()).collect();
        assert!(recipients.contains(&"unique_multi_a"));
        assert!(recipients.contains(&"unique_multi_b"));
    }

    #[tokio::test]
    async fn strips_media_markers_from_content() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("strip_markers", "unique_markers").await;

        let msg = gui_msg(
            "strip_markers",
            "Check this [IMAGE:/tmp/screenshot.png] and my [AUDIO:/tmp/recording.mp3]",
        );
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_markers")
            .collect();
        assert_eq!(our_msgs.len(), 1);
        // Markers should be stripped entirely (trailing whitespace is trimmed).
        assert_eq!(
            our_msgs[0].content,
            "<blockquote>\nCheck this  and my\n</blockquote>"
        );
    }

    #[tokio::test]
    async fn preserves_markdown_formatting_in_blockquote() {
        let _lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        setup_user_with_telegram_binding("md_user", "unique_md").await;

        let msg = gui_msg("md_user", "**bold** and `code` and *italic*");
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_md")
            .collect();
        assert_eq!(our_msgs.len(), 1);
        // Markdown syntax inside the blockquote passes through — the Telegram
        // channel's markdown_to_telegram_html will handle formatting later.
        assert_eq!(
            our_msgs[0].content,
            "<blockquote>\n**bold** and `code` and *italic*\n</blockquote>"
        );
    }
}
