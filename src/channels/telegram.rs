use crate::util::html::{decode_html_entities, escape_html, push_escaped};
use crate::util::media_target::{self, MediaTarget};
use crate::util::{TELEGRAM_MEDIA_MARKER_RE, UnwrapPoison, is_http_url, parse_media_marker};
use crate::{Channel, ChannelMessage, Role, SendMessage};
use anyhow::Context;
use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

/// Telegram's maximum message length for text messages
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;
/// Reserve space for continuation markers added by `send_text_chunks`:
/// worst case is "(continued)\n\n" + chunk + "\n\n(continues...)" = 30 extra chars
const TELEGRAM_CONTINUATION_OVERHEAD: usize = 30;

/// Description for the `/clear` command — used in `setMyCommands` API and `/start` welcome message.
pub const CLEAR_COMMAND_DESC: &str = "Reset your session";
/// Description for the `/image_models` command (Artist role).
pub const IMAGE_MODELS_COMMAND_DESC: &str = "Select image generation model";
/// Description for the `/video_models` command (Artist role).
pub const VIDEO_MODELS_COMMAND_DESC: &str = "Select video model";
/// Description for the `/board` command (admin).
pub const BOARD_COMMAND_DESC: &str = "List active workspace tickets";
/// Description for the `/archive` command (admin).
pub const ARCHIVE_COMMAND_DESC: &str = "Archive done & cancelled tickets";
/// Description for the `/pause` command (admin).
pub const PAUSE_COMMAND_DESC: &str = "Pause the workspace pipeline";
/// Description for the `/unpause` command (admin).
pub const UNPAUSE_COMMAND_DESC: &str = "Resume the workspace pipeline";
/// Description for the `/maintenance_on` command (admin, menu form).
pub const MAINTENANCE_ON_COMMAND_DESC: &str = "Enable workspace maintenance";
/// Description for the `/maintenance_off` command (admin, menu form).
pub const MAINTENANCE_OFF_COMMAND_DESC: &str = "Disable workspace maintenance";
/// Description for the `/update` command (admin, menu form).
pub const UPDATE_COMMAND_DESC: &str = "Update MahBot to the latest version";

// ── Action prefixes (__act__) ───────────────────────────────────────

/// Callback data prefix for action callbacks (e.g., model selection, clear session).
pub(crate) const ACTION_PREFIX: &str = "__act__";

/// Decode action callback data.
///
/// Returns `(action, payload)` on success, `None` when `content` does not
/// carry the `ACTION_PREFIX`.
///
/// # Format
///
/// `__act__<action>|<payload>` where `<action>` is the action name and
/// `<payload>` is the action-specific data (may be empty).
///
/// **Examples:**
/// - `__act__set_image_model|google/gemini-3.1-flash-image-preview`
///   → `("set_image_model", "google/gemini-3.1-flash-image-preview")`
/// - `__act__clear_session|` → `("clear_session", "")`
/// - `__act__clear_session` → `("clear_session", "")`
#[must_use]
pub fn decode_action(content: &str) -> Option<(String, String)> {
    let rest = content.strip_prefix(ACTION_PREFIX)?;
    match rest.split_once('|') {
        Some((action, payload)) => Some((action.to_string(), payload.to_string())),
        None => Some((rest.to_string(), String::new())),
    }
}

/// Metadata for an incoming document or photo attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IncomingAttachment {
    file_id: String,
    file_name: Option<String>,
    file_size: Option<u64>,
    caption: Option<String>,
    kind: IncomingAttachmentKind,
    mime_type: Option<String>,
}

/// The kind of incoming attachment (document, photo, video, or audio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncomingAttachmentKind {
    Document,
    Photo,
    Video,
    Audio,
}
/// Split a message into chunks that respect Telegram's 4096 character limit.
/// Tries to split at word boundaries when possible, and handles continuation.
/// The effective per-chunk limit is reduced to leave room for continuation markers.
/// When the input contains HTML tags, avoids splitting mid-tag unless the tag
/// would push the chunk past the limit, in which case the split stays at the
/// 4066-char boundary (the plain-text fallback in `send_text_chunks` tolerates
/// the resulting malformed HTML).
fn split_message_for_telegram(message: &str) -> Vec<String> {
    if message.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH {
        return vec![message.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = message;
    let chunk_limit = TELEGRAM_MAX_MESSAGE_LENGTH - TELEGRAM_CONTINUATION_OVERHEAD;

    while !remaining.is_empty() {
        // Find a good split point within the chunk_limit region.
        let hard_split = remaining
            .char_indices()
            .nth(chunk_limit)
            .map_or(remaining.len(), |(idx, _)| idx);

        let mut chunk_end = if hard_split == remaining.len() {
            hard_split
        } else {
            // Try to find a good break point (newline, then space) within hard_split.
            find_split_boundary(remaining, hard_split)
        };

        // If we split inside an HTML tag, extend past the '>'.
        // Clamp the extension to `hard_split` — the byte offset of the
        // 4066-char boundary, always a char boundary. Letting the tag push
        // the chunk past the sendable limit gets the message rejected by
        // the API (the HTML send and the plain-text retry both fail), while
        // a mid-tag split only degrades formatting — the existing HTML→plain
        // fallback in `send_text_chunks` already tolerates it. Clamping to a
        // char count instead would be unit-mismatched (byte offset vs chars)
        // and panic on multibyte text.
        if let Some(adjusted) = extend_past_open_tag(remaining, chunk_end) {
            chunk_end = adjusted.min(hard_split);
        }

        chunks.push(remaining[..chunk_end].to_string());
        remaining = &remaining[chunk_end..];
    }

    chunks
}

/// Apply continuation markers to a chunk in a multi-part Telegram message.
///
/// * First chunk: `"...\n\n(continues...)"`
/// * Middle chunk: `"(continued)\n\n...\n\n(continues...)"`
/// * Last chunk: `"(continued)\n\n..."`
/// * Single chunk: passed through unchanged.
fn wrap_chunk(chunk: &str, index: usize, total: usize) -> String {
    if total > 1 {
        if index == 0 {
            format!("{chunk}\n\n(continues...)")
        } else if index == total - 1 {
            format!("(continued)\n\n{chunk}")
        } else {
            format!("(continued)\n\n{chunk}\n\n(continues...)")
        }
    } else {
        chunk.to_string()
    }
}

/// Find the best split point within the first `hard_split` bytes of `text`.
/// Returns a byte offset ≤ `hard_split`, preferring the natural break
/// (newline or space) closest to `hard_split`, or a hard character-boundary
/// split when neither exists.
fn find_split_boundary(text: &str, hard_split: usize) -> usize {
    let search_area = &text[..hard_split];
    search_area
        .rfind('\n')
        .max(search_area.rfind(' '))
        .map_or(hard_split, |p| p + 1)
}

/// If `pos` is inside an HTML tag (the last `<` before `pos` has no matching `>`),
/// return the byte offset just past the closing `>`. Otherwise return `None`.
///
/// Handles `>` inside quoted attribute values correctly — a `>` inside a
/// single- or double-quoted string is not treated as a tag closer.
fn extend_past_open_tag(text: &str, pos: usize) -> Option<usize> {
    let prefix = &text[..pos];
    let last_open = prefix.rfind('<')?;

    // Scan forward from last_open in one pass, tracking quote state,
    // to find the first unquoted '>' (the real tag closer).
    let mut in_quote = false;
    let mut quote_char = '"';

    for (i, c) in text[last_open..].char_indices() {
        match c {
            '"' | '\'' if !in_quote => {
                in_quote = true;
                quote_char = c;
            }
            '"' | '\'' if in_quote && c == quote_char => {
                in_quote = false;
            }
            '>' if !in_quote => {
                let gt_absolute = last_open + i;
                if gt_absolute < pos {
                    return None; // tag properly closed before pos
                }
                return Some(gt_absolute + 1); // past the closing '>'
            }
            _ => {}
        }
    }

    // No unquoted '>' found at all.
    None
}

fn extract_sender_user_name(message: &serde_json::Value) -> String {
    message
        .get("from")
        .and_then(|from| from.get("username"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(crate::users::TELEGRAM_UNKNOWN_SENTINEL)
        .to_string()
}

/// Extracted metadata common to both text and attachment message parsing.
struct MessageContext {
    user_name: String,
    chat_id: String,
    message_id: i64,
    reply_target: String,
}

impl MessageContext {
    fn into_channel_message(self, content: String, cq_id: Option<String>) -> ChannelMessage {
        ChannelMessage {
            user_name: self.user_name,
            reply_target: self.reply_target,
            content,
            channel: "telegram".to_string(),
            workspace: String::new(),
            optimistic_id: None,
            callback_query_id: cq_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TelegramAttachmentKind {
    Image,
    Document,
    Video,
    Audio,
    Voice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TelegramAttachment {
    kind: TelegramAttachmentKind,
    target: String,
}

/// Metadata associated with each attachment kind.
#[derive(Debug, Clone, Copy)]
struct AttachmentMeta {
    api_method: &'static str,
    form_field: &'static str,
    default_filename: &'static str,
    label: &'static str,
    /// Send `disable_content_type_detection=true` with the multipart form so
    /// Telegram does not re-classify image uploads as photos (which would
    /// re-encode them despite the sendDocument path).
    disable_content_type_detection: bool,
}

impl TelegramAttachmentKind {
    fn from_marker(marker: &str) -> Option<Self> {
        match marker.trim().to_ascii_uppercase().as_str() {
            "IMAGE" => Some(Self::Image),
            "VIDEO" => Some(Self::Video),
            "AUDIO" => Some(Self::Audio),
            _ => None,
        }
    }

    const fn meta(self) -> AttachmentMeta {
        match self {
            Self::Image => AttachmentMeta {
                api_method: "sendPhoto",
                form_field: "photo",
                default_filename: "photo.jpg",
                label: "Image",
                disable_content_type_detection: false,
            },
            Self::Document => AttachmentMeta {
                api_method: "sendDocument",
                form_field: "document",
                default_filename: "file",
                label: "Document",
                disable_content_type_detection: false,
            },
            Self::Video => AttachmentMeta {
                api_method: "sendVideo",
                form_field: "video",
                default_filename: "video.mp4",
                label: "Video",
                disable_content_type_detection: false,
            },
            Self::Audio => AttachmentMeta {
                api_method: "sendAudio",
                form_field: "audio",
                default_filename: "audio.mp3",
                label: "Audio",
                disable_content_type_detection: false,
            },
            Self::Voice => AttachmentMeta {
                api_method: "sendVoice",
                form_field: "voice",
                default_filename: "voice.ogg",
                label: "Voice",
                disable_content_type_detection: false,
            },
        }
    }

    /// Metadata for local-file sends. Image files use the general-file path
    /// (sendDocument) so recipients get the original bytes — sendPhoto always
    /// re-encodes server-side. URL sends keep `meta()` (sendDocument-by-URL
    /// only accepts .PDF/.ZIP), so this routing must not touch that path.
    const fn file_meta(self) -> AttachmentMeta {
        match self {
            Self::Image => AttachmentMeta {
                api_method: "sendDocument",
                form_field: "document",
                default_filename: "image",
                label: "Image",
                disable_content_type_detection: true,
            },
            _ => self.meta(),
        }
    }
}

/// Recognized image file extensions for Telegram receive-path routing.
/// PNG/JPEG/WebP only — gif/bmp are deliberately NOT routed as images
/// anywhere (codec support is trimmed; see the image dependency's feature
/// list). GIF-picker animations arrive as `video`-kind attachments and take
/// the video path instead.
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// MIME prefixes routed as images by the MIME fallback in
/// [`format_attachment_content`]. Mirrors [`IMAGE_EXTENSIONS`]: only the
/// PNG/JPEG/WebP MIME types are admitted, so gif/bmp (including the legacy
/// `image/x-ms-bmp` alias) never route as images.
const IMAGE_MIME_PREFIXES: &[&str] = &["image/png", "image/jpeg", "image/jpg", "image/webp"];

/// Format a sender label for display: `@username` if a username is present,
/// otherwise the display name (first_name, or `"unknown"` as ultimate fallback).
#[must_use]
fn format_sender_label(from: &serde_json::Value) -> String {
    if let Some(username) = from.get("username").and_then(serde_json::Value::as_str) {
        format!("@{username}")
    } else {
        from.get("first_name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(crate::users::TELEGRAM_UNKNOWN_SENTINEL)
            .to_string()
    }
}

/// Build the user-facing content string for an incoming attachment.
///
/// Photos with a recognized image extension use `[IMAGE:/path]` so
/// enrichment can convert them to native image parts for the routed role's
/// model.  When the extension is not recognized the optional `mime_type` is
/// consulted as a secondary signal (e.g. Document + no extension +
/// "image/jpeg" → still `[IMAGE:]`).
/// Videos (native `video`/`video_note`/`animation` messages, or documents
/// with a video MIME/extension) use `[VIDEO:/path]` so enrichment can copy
/// them into the workspace uploads and route them into the video-edit flow.
/// Voice and audio messages use `[AUDIO:/path]`. Other attachment types use
/// `[Document: name] /path`.
fn format_attachment_content(
    kind: IncomingAttachmentKind,
    local_filename: &str,
    local_path: &Path,
    mime_type: Option<&str>,
) -> String {
    // MIME fallback mirrors the extension whitelist: only the PNG/JPEG/WebP
    // MIME types route as images (no gif/bmp image support anywhere).
    let is_image = crate::util::has_extension(local_path, IMAGE_EXTENSIONS)
        || mime_type.is_some_and(|m| IMAGE_MIME_PREFIXES.iter().any(|p| m.starts_with(p)));
    let is_video = crate::util::is_video_extension(local_path)
        || mime_type.is_some_and(|m| m.starts_with("video/"));
    match kind {
        IncomingAttachmentKind::Photo | IncomingAttachmentKind::Document if is_image => {
            format!("[IMAGE:{}]", local_path.display())
        }
        IncomingAttachmentKind::Video | IncomingAttachmentKind::Document if is_video => {
            format!("[VIDEO:{}]", local_path.display())
        }
        IncomingAttachmentKind::Audio => {
            format!("[AUDIO:{}]", local_path.display())
        }
        _ => {
            format!("[Document: {}] {}", local_filename, local_path.display())
        }
    }
}

/// Normalize a Video-kind or video-MIME attachment's filename to a
/// recognized video extension. Telegram's GIF picker sends animations as
/// H.264 MP4 bytes under a ".gif" file_name, which would bounce at
/// video_edit's extension guard after enrichment copies the clip into
/// workspace uploads. Documents with a `video/*` MIME get the same
/// treatment so routing and the guard see the same signal.
fn normalize_video_filename(
    kind: IncomingAttachmentKind,
    filename: &str,
    mime_type: Option<&str>,
) -> String {
    let is_video =
        kind == IncomingAttachmentKind::Video || mime_type.is_some_and(|m| m.starts_with("video/"));
    if is_video && !crate::util::is_video_extension(std::path::Path::new(filename)) {
        let stem = std::path::Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename);
        // Derive the extension from the declared MIME when available so a
        // video/webm document isn't mislabeled as mp4.
        let ext = match mime_type {
            Some("video/webm") => "webm",
            Some("video/quicktime") => "mov",
            _ => "mp4",
        };
        format!("{stem}.{ext}")
    } else {
        filename.to_string()
    }
}

fn infer_attachment_kind_from_target(target: &str) -> Option<TelegramAttachmentKind> {
    let normalized = target.split(['?', '#']).next().unwrap();

    let extension = Path::new(normalized)
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();

    match extension.as_str() {
        ext if IMAGE_EXTENSIONS.contains(&ext) => Some(TelegramAttachmentKind::Image),
        ext if crate::util::VIDEO_EXTENSIONS.contains(&ext) => Some(TelegramAttachmentKind::Video),
        "mp3" | "m4a" | "wav" | "flac" => Some(TelegramAttachmentKind::Audio),
        "ogg" | "oga" | "opus" => Some(TelegramAttachmentKind::Voice),
        "pdf" | "txt" | "md" | "csv" | "json" | "zip" | "tar" | "gz" | "doc" | "docx" | "xls"
        | "xlsx" | "ppt" | "pptx" => Some(TelegramAttachmentKind::Document),
        _ => None,
    }
}

fn parse_path_only_attachment(message: &str) -> Option<TelegramAttachment> {
    let trimmed = message.trim();
    if trimmed.is_empty() || trimmed.contains('\n') {
        return None;
    }

    let candidate = trimmed.trim_matches(|c| matches!(c, '`' | '"' | '\''));
    if candidate.chars().any(char::is_whitespace) {
        return None;
    }

    let candidate = candidate.strip_prefix("file://").unwrap_or(candidate);
    let kind = infer_attachment_kind_from_target(candidate)?;

    // Only a real image target may be attached as a photo — a bare path to a
    // non-image or an existing non-raster file stays plain text, matching the
    // `[IMAGE:...]` marker gate. Audio/video/document keep their existing
    // URL-or-existing-file semantics.
    if kind == TelegramAttachmentKind::Image {
        if !matches!(
            media_target::classify_media_image_target(candidate),
            MediaTarget::LocalImage | MediaTarget::RemoteUrl
        ) {
            return None;
        }
    } else if !is_http_url(candidate) && !Path::new(candidate).exists() {
        return None;
    }

    Some(TelegramAttachment {
        kind,
        target: candidate.to_string(),
    })
}

/// Parse `[KIND:path]` media markers from a message, returning cleaned text
/// (with markers removed) and extracted attachments.
///
/// Uses the case-insensitive [`TELEGRAM_MEDIA_MARKER_RE`] to match markers.
/// Markers whose target is neither an http(s) URL nor an existing regular
/// file are kept as literal text — prose quoting the syntax never aborts
/// delivery. Known limitation: prose referencing a real existing file still
/// delivers as an attachment. Unknown or unrecognized markers are left intact.
fn parse_attachment_markers(message: &str) -> (String, Vec<TelegramAttachment>) {
    let mut attachments: Vec<TelegramAttachment> = Vec::new();

    let cleaned = TELEGRAM_MEDIA_MARKER_RE
        .replace_all(message, |caps: &regex::Captures| {
            let (kind_str, path) = parse_media_marker(caps);
            let path = path.trim();

            // For IMAGE markers, only a valid image target is attached — an
            // existing regular raster file or an http(s) URL. A data-URI IMAGE
            // marker (or a non-image target) stays as literal text, because
            // telegram's `send_attachment` reads a local file path or sends
            // by URL and cannot send a data URI. The kind gate is
            // case-insensitive to match TELEGRAM_MEDIA_MARKER_RE (lowercase
            // `[image:...]` must not bypass the classifier into the loose
            // is_file check). AUDIO/VIDEO markers keep their existing
            // semantics (an http(s) URL or an existing regular file); prose
            // quoting the marker syntax must stay visible and never abort the
            // send of other attachments in the same message.
            if matches!(
                TelegramAttachmentKind::from_marker(kind_str),
                Some(TelegramAttachmentKind::Image)
            ) {
                // Telegram cannot attach an inline data URI — keep it literal
                // WITHOUT a wasted bounded decode (classifying it would fully
                // decode, then reject it). Local raster / URL targets use the
                // shared classifier.
                let ok = if path.starts_with("data:") {
                    false
                } else {
                    matches!(
                        media_target::classify_media_image_target(path),
                        MediaTarget::LocalImage | MediaTarget::RemoteUrl
                    )
                };
                if !ok {
                    return caps.get_match().as_str().to_string();
                }
            } else if path.is_empty() || (!is_http_url(path) && !Path::new(path).is_file()) {
                return caps.get_match().as_str().to_string();
            }

            if let Some(kind) = TelegramAttachmentKind::from_marker(kind_str) {
                attachments.push(TelegramAttachment {
                    kind,
                    target: path.to_string(),
                });
            }
            String::new()
        })
        .to_string();

    (cleaned.trim().to_string(), attachments)
}

/// Base URL for the Telegram Bot API.
const API_BASE: &str = "https://api.telegram.org";

/// Telegram Bot API maximum file download size (20 MB).
const TELEGRAM_MAX_FILE_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;

/// `config_kv` key prefix for the per-chat role-switch pin message id.
/// Leftover entries for unbound chats are tiny and never re-read — harmless.
pub(crate) const ROLE_PIN_KV_PREFIX: &str = "telegram_role_pin:";

/// Change-detection state for a chat's per-user command menu refresh.
enum ChatMenuState {
    /// Last successfully registered command payload (`None` = never registered).
    Registered(Option<String>),
    /// A refresh is already in flight — skip until it completes.
    InFlight,
}

/// Telegram channel — long-polls the Bot API for updates
pub struct TelegramChannel {
    bot_token: String,
    /// Shared HTTP client with connection reuse across all Telegram API calls.
    http_client: reqwest::Client,

    /// Per-instance cancellation token — cancelling this stops only this
    /// channel's listener, not the entire application.
    cancel: std::sync::Arc<tokio_util::sync::CancellationToken>,

    /// Last confirmed `update_id + 1` offset. Shared across old/new listener
    /// instances during hot-reload so the new listener doesn't replay old
    /// updates from Telegram's server.
    offset: std::sync::Arc<std::sync::atomic::AtomicI64>,

    /// Per-chat command menu state — change detection + in-flight coalescing
    /// for the per-user `setMyCommands` refresh, so outbound message floods
    /// (manager broadcasts, parallel agent responses) don't trip Telegram's
    /// rate limiting. Menu refreshes are fire-and-forget and fail-open.
    menu_cache: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, ChatMenuState>>>,

    /// Mutual exclusion for role-switch pin flows: two rapid switches run in
    /// detached tasks and must not interleave (both could read "no pin yet"
    /// and pin two messages). Serialization is per instance — a bot-token
    /// hot-reload replaces the channel, so an in-flight flow on the old
    /// instance can interleave with one on the new (rare and self-healing).
    pin_lock: tokio::sync::Mutex<()>,
}

/// Extract chat_id and reply_target from a Telegram message sub-object
/// (e.g., `update["callback_query"]["message"]` or `update["message"]`).
fn extract_chat_context(message: &serde_json::Value) -> Option<(String, String)> {
    let chat_id = message.get("chat")?.get("id")?.as_i64()?.to_string();
    let thread_id = message
        .get("message_thread_id")
        .and_then(serde_json::Value::as_i64)
        .map(|id| id.to_string());
    let reply_target = match &thread_id {
        Some(tid) => format!("{chat_id}:{tid}"),
        None => chat_id.clone(),
    };
    Some((chat_id, reply_target))
}

/// Inject `message_thread_id` into a JSON request body if present.
fn set_thread_id_on_json(body: &mut serde_json::Value, thread_id: Option<&str>) {
    if let Some(tid) = thread_id {
        body["message_thread_id"] = serde_json::Value::String(tid.to_string());
    }
}

/// Parse a Telegram recipient string into `(chat_id, optional thread_id)`.
///
/// Supports two formats:
/// - `"chat_id"` → `("chat_id", None)`
/// - `"chat_id:thread_id"` → `("chat_id", Some("thread_id"))`
fn parse_recipient(recipient: &str) -> (&str, Option<&str>) {
    match recipient.split_once(':') {
        Some((chat, thread)) => (chat, Some(thread)),
        None => (recipient, None),
    }
}

/// Extract sender info, verify authorization, and update contact metadata.
/// Returns `None` if the user is not authorized or if chat context is missing.
/// On auth failure, the caller is responsible for logging (e.g., caller may want to
/// log the username). Contact info is updated only on success.
///
/// Returns a 3-tuple `(canonical_user, chat_id, reply_target)` where:
/// - `canonical_user`: the resolved system username for the Telegram sender
/// - `chat_id`: the raw chat ID (e.g., `"123456"`)
/// - `reply_target`: the reply target string (e.g., `"123456"` or `"123456:789"` for threads)
async fn resolve_authorized_sender(
    sender_source: &serde_json::Value,
    chat_source: &serde_json::Value,
) -> Option<(String, String, String)> {
    let username = extract_sender_user_name(sender_source);
    // Fail-closed: the sentinel is never a valid binding identity. Even if a
    // legacy "unknown" row exists in user_channels, a username-less sender
    // must stay unauthorized.
    if username == crate::users::TELEGRAM_UNKNOWN_SENTINEL {
        return None;
    }
    // Look up the canonical user name via user_channels binding
    let canonical_user = crate::users::resolve_user_by_channel("telegram", &username).await?;
    let (chat_id, reply_target) = extract_chat_context(chat_source)?;
    // Update reply_target for future message delivery
    let _ = crate::users::update_channel_contact("telegram", &username, &reply_target).await;
    Some((canonical_user, chat_id, reply_target))
}

/// If the text at position `i` starts with `delim`, finds the matching closing
/// `delim`, HTML-escapes the content, and wraps it in `<tag>...</tag>`.
///
/// On success, advances `i` past the closing delimiter and returns `true`.
/// Returns `false` if `delim` is not found or when the content between
/// delimiters is empty (the `end > 0` guard prevents zero-length formatting
/// spans like `****` or `*` with no content between delimiters).
///
/// This is a helper to deduplicate the 5 structurally identical inline
/// formatting branches (bold, italic, code, strikethrough). Callers that
/// need to guard against matching a single character when the previous
/// character is the same (e.g. the second `*` of `**` for italic, or the
/// second `` ` `` of ` `` ` for inline code) must apply that guard before
/// calling this helper.
fn try_format_inline(text: &str, i: &mut usize, out: &mut String, delim: &str, tag: &str) -> bool {
    if text[*i..].starts_with(delim) {
        let content_start = *i + delim.len();
        if let Some(end) = text[content_start..].find(delim)
            && end > 0
        {
            let inner = escape_html(&text[content_start..content_start + end]);
            let _ = write!(out, "<{tag}>{inner}</{tag}>");
            *i += delim.len() * 2 + end;
            return true;
        }
    }
    false
}

/// Convert Markdown to Telegram HTML format.
/// Telegram HTML supports: &lt;b&gt;, &lt;i&gt;, &lt;u&gt;, &lt;s&gt;, &lt;code&gt;, &lt;pre&gt;, &lt;a href="..."&gt;
/// Convert a subset of Markdown to Telegram's HTML parse_mode format.
///
/// Supported: headers (`# …`, `## …`), bold (`**…**`, `__…__`), italic (`*…*`),
/// inline code (`` `…` ``), links (`[…](url)`), strikethrough (`~~…~~`),
/// fenced code blocks (` ``` … ``` `), and `<blockquote>` pass-through.
///
/// Code block fences are detected first so inline formatting inside them is
/// never interpreted (single-pass with code-block tracking).
fn markdown_to_telegram_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code_block = false;
    let mut code_buf = String::new();

    for line in text.split('\n') {
        let trimmed = line.trim_start();

        // ── Fenced code blocks ────────────────────────────────
        if trimmed.starts_with("```") {
            if in_code_block {
                in_code_block = false;
                let escaped = escape_html(code_buf.trim_end_matches('\n'));
                let _ = writeln!(out, "<pre><code>{escaped}</code></pre>");
            } else {
                in_code_block = true;
            }
            code_buf.clear();
            continue;
        }

        if in_code_block {
            code_buf.push_str(line);
            code_buf.push('\n');
            continue;
        }

        // ── Blockquotes — pass through as-is ──────────────────
        if trimmed == "<blockquote>" || trimmed == "</blockquote>" {
            out.push_str(trimmed);
            out.push('\n');
            continue;
        }

        // ── Headers: ## Title → <b>Title</b> ───────────────────
        let stripped = line.trim_start_matches('#');
        let header_level = line.len() - stripped.len();
        if header_level > 0 && stripped.starts_with(' ') {
            let title = escape_html(stripped.trim());
            let _ = writeln!(out, "<b>{title}</b>");
            continue;
        }

        // ── Inline formatting per line ────────────────────────
        let mut line_out = String::new();
        let bytes = line.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            // Bold: **text**
            if try_format_inline(line, &mut i, &mut line_out, "**", "b") {
                continue;
            }
            // Bold: __text__
            if try_format_inline(line, &mut i, &mut line_out, "__", "b") {
                continue;
            }
            // Italic: *text* — guard against matching second `*` of `**`
            if (i == 0 || bytes[i - 1] != b'*')
                && try_format_inline(line, &mut i, &mut line_out, "*", "i")
            {
                continue;
            }
            // Inline code: `code` — guard against matching second `` ` `` of ` `` `
            if (i == 0 || bytes[i - 1] != b'`')
                && try_format_inline(line, &mut i, &mut line_out, "`", "code")
            {
                continue;
            }
            // Markdown link: [text](url)
            if bytes[i] == b'['
                && let Some(bracket_end) = line[i + 1..].find(']')
            {
                let text_part = &line[i + 1..i + 1 + bracket_end];
                let after_bracket = i + 1 + bracket_end + 1;
                if after_bracket < len
                    && bytes[after_bracket] == b'('
                    && let Some(paren_end) = line[after_bracket + 1..].find(')')
                {
                    let url = &line[after_bracket + 1..after_bracket + 1 + paren_end];
                    if is_http_url(url) {
                        let text_html = escape_html(text_part);
                        let url_html = escape_html(url);
                        let _ = write!(line_out, "<a href=\"{url_html}\">{text_html}</a>");
                        i = after_bracket + 1 + paren_end + 1;
                        continue;
                    }
                }
            }
            // Strikethrough: ~~text~~
            if try_format_inline(line, &mut i, &mut line_out, "~~", "s") {
                continue;
            }
            // Default: escape HTML entities
            let ch = line[i..].chars().next().unwrap();
            push_escaped(ch, &mut line_out);
            i += ch.len_utf8();
        }
        line_out.push('\n');
        out.push_str(&line_out);
    }

    // Unclosed code block at EOF — emit what we have.
    if in_code_block && !code_buf.is_empty() {
        let _ = writeln!(
            out,
            "<pre><code>{}</code></pre>",
            escape_html(code_buf.trim_end())
        );
    }

    out.trim_end_matches('\n').to_string()
}

/// Strip all HTML tags from a string, leaving only the text content.
/// Used when falling back from HTML `parse_mode` to plain text so users
/// don't see raw tags like `<b>`, `<code>`, `<pre>` etc.
///
/// Correctly handles `>` inside quoted attribute values — a `>` inside a
/// single- or double-quoted string is not treated as a tag closer.
fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    let mut in_quote = false;
    let mut quote_char = '"';
    for c in s.chars() {
        match c {
            '<' if !in_tag => in_tag = true,
            '>' if in_tag && !in_quote => in_tag = false,
            '"' | '\'' if in_tag => {
                if in_quote && c == quote_char {
                    in_quote = false;
                } else if !in_quote {
                    in_quote = true;
                    quote_char = c;
                }
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

/// Classification of an `editMessageText` failure for the role-switch pin
/// flow — matched on stable substrings independent of Telegram's wording;
/// a reworded error hits `Other` (delivery unaffected, no recovery).
enum EditMessageFailure {
    /// Target message was deleted — re-send + pin + persist a fresh id.
    NotFound,
    /// Past Telegram's 48-hour edit window — re-send + pin + persist a fresh id.
    CannotEdit,
    /// Text is identical ("message is not modified") — success/no-op.
    NotModified,
    /// Any other failure — deliver a plain notification, keep the stored id
    /// (only 'not found'/'can't be edited' recover).
    Other,
}

fn classify_edit_failure(description: &str) -> EditMessageFailure {
    let lower = description.to_lowercase();
    if lower.contains("not found") {
        EditMessageFailure::NotFound
    } else if lower.contains("not modified") {
        EditMessageFailure::NotModified
    } else if lower.contains("can't be edited") || lower.contains("cant be edited") {
        EditMessageFailure::CannotEdit
    } else {
        EditMessageFailure::Other
    }
}

/// Shared conversion for outbound text: decode HTML entities (e.g. &#39;
/// that LLMs may emit) before markdown→HTML conversion so they don't get
/// double-escaped. The role-pin flow reuses this so pinned content matches
/// plain sends.
fn to_telegram_html(text: &str) -> String {
    markdown_to_telegram_html(&decode_html_entities(text))
}

impl TelegramChannel {
    /// Internal constructor shared by [`new`](Self::new) and
    /// [`with_offset`](Self::with_offset).
    #[must_use]
    fn new_with(bot_token: String, offset: std::sync::Arc<std::sync::atomic::AtomicI64>) -> Self {
        Self {
            bot_token,
            http_client: crate::util::http::build_http_client(Duration::from_mins(1)),
            cancel: std::sync::Arc::new(tokio_util::sync::CancellationToken::new()),
            offset,
            menu_cache: std::sync::Arc::new(
                std::sync::Mutex::new(std::collections::HashMap::new()),
            ),
            pin_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// # Panics
    ///
    /// Panics if `reqwest::Client::build()` fails — with reqwest 0.13's
    /// `rustls-no-provider` TLS stack this happens when no rustls crypto
    /// provider is installed. `util::http::install_ring_provider` installs
    /// the ring provider before the client is built (the TLS stack is
    /// rustls/ring — OpenSSL is not involved).
    #[must_use]
    pub fn new(bot_token: String) -> Self {
        Self::new_with(
            bot_token,
            std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0)),
        )
    }

    /// Create a new channel that inherits the update offset from a
    /// previous instance. Used during hot-reload to avoid replaying
    /// already-processed Telegram updates.
    #[must_use]
    pub fn with_offset(
        bot_token: String,
        inherited_offset: std::sync::Arc<std::sync::atomic::AtomicI64>,
    ) -> Self {
        Self::new_with(bot_token, inherited_offset)
    }

    /// Answer a callback query to dismiss the loading spinner.
    /// When `text` is provided, shows a toast notification to the user.
    /// Errors are logged so users don't get stuck on an infinite spinner.
    pub async fn answer_callback_query(&self, callback_query_id: &str, text: Option<&str>) {
        let mut body = serde_json::json!({
            "callback_query_id": callback_query_id,
        });
        if let Some(txt) = text {
            body["text"] = serde_json::Value::String(txt.to_string());
        }
        if let Err((status, error)) = self
            .post_telegram_json("answerCallbackQuery", body, "answerCallbackQuery error")
            .await
        {
            tracing::warn!(
                callback_query_id = %callback_query_id,
                status = %status,
                error = %error,
                "answerCallbackQuery failed"
            );
        }
    }

    /// Parse a `callback_query` update into a `ChannelMessage`.
    /// The callback data becomes the message content.
    async fn parse_callback_query(&self, cq: &serde_json::Value) -> Option<ChannelMessage> {
        let data = cq.get("data").and_then(serde_json::Value::as_str)?;
        let msg = cq.get("message")?;
        let callback_query_id = cq
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        // Auth is clicker-based (cq.from), never msg author; chat_id/message_id unused.
        let Some((user_name, _, reply_target)) = resolve_authorized_sender(cq, msg).await else {
            tracing::debug!(
                "Telegram: ignoring callback query from unknown user '{}'",
                extract_sender_user_name(cq)
            );
            return None;
        };

        let ctx = MessageContext {
            user_name,
            chat_id: String::new(),
            message_id: 0,
            reply_target,
        };
        Some(ctx.into_channel_message(data.to_string(), callback_query_id))
    }

    fn extract_update_message_target(update: &serde_json::Value) -> Option<(String, i64)> {
        let message = update.get("message")?;
        let chat_id = extract_chat_context(message)?.0;
        let message_id = message
            .get("message_id")
            .and_then(serde_json::Value::as_i64)?;
        Some((chat_id, message_id))
    }

    /// Extract sender info, user allow check, chat/message/thread IDs, and
    /// reply target. Returns `None` if the sender is not allowed.
    async fn extract_message_context(&self, message: &serde_json::Value) -> Option<MessageContext> {
        let Some((user_name, chat_id, reply_target)) =
            resolve_authorized_sender(message, message).await
        else {
            tracing::debug!(
                "Telegram: ignoring message from unknown user '{}'",
                extract_sender_user_name(message)
            );
            return None;
        };

        let message_id = message
            .get("message_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        Some(MessageContext {
            user_name,
            chat_id,
            message_id,
            reply_target,
        })
    }

    /// Prepend reply context and forwarding attribution to content.
    fn prepend_reply_metadata(content: String, message: &serde_json::Value) -> String {
        let content = if let Some(quote) = Self::extract_reply_context(message) {
            format!("{quote}\n\n{content}")
        } else {
            content
        };
        if let Some(attr) = Self::format_forward_attribution(message) {
            format!("{attr}{content}")
        } else {
            content
        }
    }

    fn try_add_ack_reaction_nonblocking(&self, chat_id: String, message_id: i64) {
        let client = self.http_client().clone();
        let url = self.api_url("setMessageReaction");
        let body = serde_json::json!({
            "chat_id": &chat_id,
            "message_id": message_id,
            "reaction": [{"type": "emoji", "emoji": "👀"}]
        });

        tokio::spawn(async move {
            let response = match client.post(&url).json(&body).send().await {
                Ok(resp) => resp,
                Err(err) => {
                    tracing::warn!(
                        "Telegram: failed to add ACK reaction to chat_id={chat_id}, message_id={message_id}: {err}"
                    );
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let err_body =
                    crate::util::http::read_error_body(response, "ACK reaction error").await;
                tracing::warn!(
                    "Telegram: add ACK reaction failed for chat_id={chat_id}, message_id={message_id}: status={status}, body={err_body}"
                );
            }
        });
    }

    const fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    fn api_url(&self, method: &str) -> String {
        format!("{API_BASE}/bot{}/{method}", self.bot_token)
    }

    /// Signal this specific channel's listener to stop, without affecting
    /// the global shutdown token or other channels.
    pub fn cancel_own(&self) {
        self.cancel.cancel();
    }

    /// Register the bot's global (unscoped) commands via Telegram's
    /// `setMyCommands` API. Only `/clear` is global — per-user commands are
    /// registered per chat via [`Self::spawn_menu_refresh`].
    ///
    /// Failure is logged as a warning and does not block the caller.
    pub async fn set_my_commands(&self) {
        let body = serde_json::json!({
            "commands": [
                {"command": "clear", "description": CLEAR_COMMAND_DESC},
            ]
        });
        post_set_my_commands(self.http_client(), &self.api_url("setMyCommands"), &body).await;
    }

    /// Spawn a per-user command menu refresh for a chat, triggered after
    /// every outbound message. Fire-and-forget and fail-open: any failure
    /// (DB lookup, API error) is logged and never affects message delivery.
    ///
    /// Everything except a cheap in-flight check runs in the spawned task:
    /// the reverse-lookup of the chat's bound user (first match wins for
    /// group chats), the command payload computed from their current
    /// role/admin state, and the scoped `setMyCommands` registration when
    /// it differs from the last successful one (change detection) with
    /// in-flight coalescing.
    fn spawn_menu_refresh(&self, chat_id: &str) {
        // Cheap inline coalescing: skip spawning when a refresh for this
        // chat is already in flight.
        {
            let cache = self.menu_cache.lock().unwrap_poison();
            if matches!(cache.get(chat_id), Some(ChatMenuState::InFlight)) {
                return;
            }
        }

        let bot_token = self.bot_token.clone();
        let http_client = self.http_client.clone();
        let cache = std::sync::Arc::clone(&self.menu_cache);
        let chat_id = chat_id.to_string();
        tokio::spawn(async move {
            let Some(user_name) =
                crate::users::resolve_user_by_reply_target("telegram", &chat_id).await
            else {
                return;
            };
            let entries = user_command_entries(&user_name).await;
            let payload = serde_json::json!({
                "commands": entries
                    .iter()
                    .map(|(cmd, desc)| serde_json::json!({ "command": cmd, "description": desc }))
                    .collect::<Vec<_>>(),
            });
            let payload_str = payload.to_string();

            let should_send = {
                let mut cache = cache.lock().unwrap_poison();
                match cache.get(&chat_id) {
                    Some(ChatMenuState::Registered(last))
                        if last.as_deref() == Some(&payload_str) =>
                    {
                        false
                    }
                    Some(ChatMenuState::InFlight) => false,
                    _ => {
                        cache.insert(chat_id.clone(), ChatMenuState::InFlight);
                        true
                    }
                }
            };
            if !should_send {
                return;
            }

            let url = format!("{API_BASE}/bot{bot_token}/setMyCommands");
            let body = serde_json::json!({
                "scope": {
                    "type": "chat",
                    "chat_id": chat_id.parse::<i64>().map_or_else(
                        |_| serde_json::Value::String(chat_id.clone()),
                        serde_json::Value::from,
                    ),
                },
                "commands": payload["commands"].clone(),
            });
            let ok = post_set_my_commands(&http_client, &url, &body).await;
            let mut cache = cache.lock().unwrap_poison();
            cache.insert(
                chat_id,
                if ok {
                    ChatMenuState::Registered(Some(payload_str))
                } else {
                    // Reset so the next message retries.
                    ChatMenuState::Registered(None)
                },
            );
        });
    }

    /// Validate a Telegram bot token by calling the `getMe` endpoint.
    /// Returns `Ok(())` if the token is valid, `Err` with a descriptive
    /// message otherwise.
    pub async fn validate_token(token: &str) -> anyhow::Result<()> {
        if token.trim().is_empty() {
            anyhow::bail!("Telegram bot token is empty");
        }
        let url = format!("{API_BASE}/bot{token}/getMe");
        let client = crate::util::http::build_http_client(std::time::Duration::from_secs(10));
        let resp = client
            .get(&url)
            .send()
            .await
            .context("Failed to reach Telegram API")?;
        let status = resp.status();
        let body: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => anyhow::bail!("Failed to parse Telegram API response: {e}"),
        };
        if !status.is_success() || body.get("ok").and_then(serde_json::Value::as_bool) != Some(true)
        {
            let desc = body
                .get("description")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown error");
            anyhow::bail!("Invalid Telegram bot token: {desc}");
        }
        Ok(())
    }

    fn handle_non_parseable_message(update: &serde_json::Value) {
        let Some(message) = update.get("message") else {
            return;
        };

        let text = message
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<non-text content>");
        tracing::debug!("Telegram: message not parseable (unsupported type), skipping: {text}");
    }

    /// Get the file path for a Telegram file ID via the Bot API.
    async fn get_file_path(&self, file_id: &str) -> anyhow::Result<String> {
        let url = self.api_url("getFile");
        let resp = self
            .http_client()
            .get(&url)
            .query(&[("file_id", file_id)])
            .send()
            .await
            .context("Failed to call Telegram getFile")?;

        let data: serde_json::Value = resp.json().await?;
        data.get("result")
            .and_then(|r| r.get("file_path"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
            .context("Telegram getFile: missing file_path in response")
    }

    /// Download a file from the Telegram CDN.
    async fn download_file(&self, file_path: &str) -> anyhow::Result<Vec<u8>> {
        let url = format!("{API_BASE}/file/bot{}/{file_path}", self.bot_token);
        let resp = self
            .http_client()
            .get(&url)
            .send()
            .await
            .context("Failed to download Telegram file")?;

        if !resp.status().is_success() {
            anyhow::bail!("Telegram file download failed: {}", resp.status());
        }

        Ok(resp.bytes().await?.to_vec())
    }

    /// Extract attachment metadata from an incoming Telegram message.
    ///
    /// Handles `document`, `photo` (array — takes last element for highest
    /// resolution), `video`, `video_note`, `animation`, `audio`, and `voice`.
    /// Both `audio` and `voice` map to [`IncomingAttachmentKind::Audio`] since
    /// there's no separate variant for each.  Returns `None` for text‑only
    /// and other unsupported message types.
    ///
    /// `document` is checked first because Telegram sets both `animation` and
    /// `document` on animation messages (so those classify as Document).
    /// `video_note` is a round video with no `file_name`/`mime_type`;
    /// `animation` is GIF-like (usually webm/mp4).  `photo` is mutually
    /// exclusive with the other media keys, so it is checked last.
    fn parse_attachment_metadata(message: &serde_json::Value) -> Option<IncomingAttachment> {
        for (key, kind) in [
            ("document", IncomingAttachmentKind::Document),
            ("video", IncomingAttachmentKind::Video),
            ("video_note", IncomingAttachmentKind::Video),
            ("animation", IncomingAttachmentKind::Video),
            ("audio", IncomingAttachmentKind::Audio),
            ("voice", IncomingAttachmentKind::Audio),
        ] {
            if let Some(v) = message.get(key) {
                return Self::build_attachment(v, message, kind);
            }
        }
        // Photo (array of PhotoSize — take last = highest resolution)
        if let Some(photos) = message.get("photo").and_then(serde_json::Value::as_array) {
            let best = photos.last()?;
            return Self::build_attachment(best, message, IncomingAttachmentKind::Photo);
        }
        None
    }

    /// Build an [`IncomingAttachment`] from a pre‑resolved JSON sub‑object.
    ///
    /// * `sub_obj` — the value *inside* the attachment key (e.g. the document
    ///   object, the last photo array element, or the voice object).
    /// * `message` — the parent Telegram message object (provides `caption`).
    fn build_attachment(
        sub_obj: &serde_json::Value,
        message: &serde_json::Value,
        kind: IncomingAttachmentKind,
    ) -> Option<IncomingAttachment> {
        let file_id = sub_obj.get("file_id")?.as_str()?.to_string();
        let file_name = sub_obj
            .get("file_name")
            .and_then(serde_json::Value::as_str)
            .map(String::from);
        let file_size = sub_obj.get("file_size").and_then(serde_json::Value::as_u64);
        let caption = message
            .get("caption")
            .and_then(serde_json::Value::as_str)
            .map(String::from);
        let mime_type = sub_obj
            .get("mime_type")
            .and_then(serde_json::Value::as_str)
            .map(String::from);
        Some(IncomingAttachment {
            file_id,
            file_name,
            file_size,
            caption,
            kind,
            mime_type,
        })
    }

    /// Attempt to parse a Telegram update as a document/photo attachment.
    ///
    /// Downloads the file to a system temp directory and returns a
    /// `ChannelMessage` with the local file path. The file is later moved or
    /// cleaned up by [`enrich_message`](crate::channels::enrich_message). Returns `None` if the message
    /// is not an attachment, the sender is not authorized, or the file exceeds
    /// size limits.
    async fn try_parse_attachment_message(
        &self,
        update: &serde_json::Value,
    ) -> Option<ChannelMessage> {
        let message = update.get("message")?;
        let attachment = Self::parse_attachment_metadata(message)?;

        // Check file size limit
        if let Some(size) = attachment.file_size
            && size > TELEGRAM_MAX_FILE_DOWNLOAD_BYTES
        {
            tracing::info!(
                "Skipping attachment: file size {size} bytes exceeds {} MB limit",
                TELEGRAM_MAX_FILE_DOWNLOAD_BYTES / (1024 * 1024)
            );
            return None;
        }

        let ctx = self.extract_message_context(message).await?;

        // Save to system temp directory — cleaned up by enrich_message
        let save_dir = std::env::temp_dir().join(crate::util::TELEGRAM_FILES_DIR);
        if let Err(e) = tokio::fs::create_dir_all(&save_dir).await {
            tracing::warn!("Failed to create telegram_files directory: {e}");
            return None;
        }

        // Download file from Telegram
        let tg_file_path = match self.get_file_path(&attachment.file_id).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Failed to get attachment file path: {e}");
                return None;
            }
        };

        let file_data = match self.download_file(&tg_file_path).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to download attachment: {e}");
                return None;
            }
        };

        // Determine local filename
        let local_filename = if let Some(name) = &attachment.file_name {
            name.clone()
        } else {
            let ext = Path::new(&tg_file_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or(match attachment.kind {
                    // video_note has no file_name/mime_type — a ".jpg" default
                    // would misroute it out of the video flow.
                    IncomingAttachmentKind::Video => "mp4",
                    _ => "jpg",
                });
            let prefix = match attachment.kind {
                IncomingAttachmentKind::Photo => "photo",
                IncomingAttachmentKind::Video => "video",
                IncomingAttachmentKind::Audio => "audio",
                IncomingAttachmentKind::Document => "file",
            };
            format!("{prefix}_{}_{}.{ext}", ctx.chat_id, ctx.message_id)
        };
        let local_filename = normalize_video_filename(
            attachment.kind,
            &local_filename,
            attachment.mime_type.as_deref(),
        );

        let local_path = save_dir.join(&local_filename);
        if let Err(e) = tokio::fs::write(&local_path, &file_data).await {
            tracing::warn!("Failed to save attachment to {}: {e}", local_path.display());
            return None;
        }

        let mut content = format_attachment_content(
            attachment.kind,
            &local_filename,
            &local_path,
            attachment.mime_type.as_deref(),
        );
        if let Some(caption) = &attachment.caption
            && !caption.is_empty()
        {
            let _ = write!(content, "\n\n{caption}");
        }

        let content = Self::prepend_reply_metadata(content, message);

        Some(ctx.into_channel_message(content, None))
    }

    /// Build a forwarding attribution prefix from Telegram forward fields.
    ///
    /// Returns `Some("[Forwarded from ...] ")` when the message is forwarded,
    /// `None` otherwise.
    fn format_forward_attribution(message: &serde_json::Value) -> Option<String> {
        if let Some(from_chat) = message.get("forward_from_chat") {
            // Forwarded from a channel or group
            let title = from_chat
                .get("title")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown channel");
            Some(format!("[Forwarded from channel: {title}] "))
        } else if let Some(from_user) = message.get("forward_from") {
            // Forwarded from a user (privacy allows identity)
            let label = format_sender_label(from_user);
            Some(format!("[Forwarded from {label}] "))
        } else {
            // Forwarded from a user who hides their identity
            message
                .get("forward_sender_name")
                .and_then(serde_json::Value::as_str)
                .map(|name| format!("[Forwarded from {name}] "))
        }
    }

    /// Extract reply context from a Telegram `reply_to_message`, if present.
    fn extract_reply_context(message: &serde_json::Value) -> Option<String> {
        let reply = message.get("reply_to_message")?;

        let from = reply.get("from");
        let reply_label = from.map_or_else(|| "unknown".to_string(), format_sender_label);

        let reply_text = if let Some(text) = reply.get("text").and_then(serde_json::Value::as_str) {
            text.to_string()
        } else if reply.get("voice").is_some() || reply.get("audio").is_some() {
            "[Voice message]".to_string()
        } else if reply.get("photo").is_some() {
            "[Photo]".to_string()
        } else if reply.get("document").is_some() {
            "[Document]".to_string()
        } else if reply.get("video").is_some() {
            "[Video]".to_string()
        } else if reply.get("sticker").is_some() {
            "[Sticker]".to_string()
        } else {
            "[Message]".to_string()
        };

        // Format as blockquote with sender attribution
        let quoted_lines: String = reply_text
            .lines()
            .map(|line| format!("> {line}"))
            .collect::<Vec<_>>()
            .join("\n");

        Some(format!("> {reply_label}:\n{quoted_lines}"))
    }

    async fn parse_update_message(&self, update: &serde_json::Value) -> Option<ChannelMessage> {
        let message = update.get("message")?;
        let text = message.get("text").and_then(serde_json::Value::as_str)?;
        let ctx = self.extract_message_context(message).await?;

        // Strip @BotUsername suffix from commands (e.g. `/new@MyBot` → `/new`)
        // Telegram appends the bot username to commands in group chats.
        let text = if text.starts_with('/') {
            text.split('@').next().unwrap_or(text)
        } else {
            text
        };

        let content = text.to_string();

        let content = Self::prepend_reply_metadata(content, message);

        Some(ctx.into_channel_message(content, None))
    }

    /// POST a JSON body to a Telegram API method; returns the response on 2xx
    /// or `(status, body)` on failure. Network/transport errors map to
    /// BAD_GATEWAY (no HTTP response exists).
    async fn post_telegram_json(
        &self,
        method: &str,
        body: serde_json::Value,
        err_label: &str,
    ) -> Result<reqwest::Response, (reqwest::StatusCode, String)> {
        let resp = self
            .http_client()
            .post(self.api_url(method))
            .json(&body)
            .send()
            .await
            // Network/transport errors (connection refused, DNS failure, timeout) produce
            // no HTTP response, so we use BAD_GATEWAY as a sentinel — it signals an upstream
            // communication failure, not an actual HTTP-level error from the Telegram API.
            .map_err(|e| (reqwest::StatusCode::BAD_GATEWAY, e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            Ok(resp)
        } else {
            let err_body = crate::util::http::read_error_body(resp, err_label).await;
            Err((status, err_body))
        }
    }

    /// Send one Telegram text message, with optional `parse_mode`. Returns
    /// the Telegram message id on success (`None` when the 2xx body omits it
    /// — the message was still delivered; only the pin flow needs the id), or
    /// the HTTP status and response body on failure. Parsing the body for the
    /// id is an accepted coupling into the shared send path — the `None` case
    /// preserves status-only semantics for callers that don't need the id.
    async fn send_message_get_id(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        text: &str,
        parse_mode: Option<&str>,
        reply_markup: Option<serde_json::Value>,
    ) -> Result<Option<i64>, (reqwest::StatusCode, String)> {
        let mut body = serde_json::json!({
            "chat_id": chat_id,
            "text": text,
        });
        if let Some(mode) = parse_mode {
            body["parse_mode"] = serde_json::Value::String(mode.to_string());
        }
        set_thread_id_on_json(&mut body, thread_id);
        if let Some(markup) = reply_markup {
            body["reply_markup"] = markup;
        }

        let resp = self
            .post_telegram_json("sendMessage", body, "sendMessage error")
            .await?;

        let status = resp.status();
        // A 2xx body without a parseable message_id is still a delivered
        // message: the shared send path only checks the status, so this must
        // not be treated as a failure.
        let id = match resp.json::<serde_json::Value>().await {
            Ok(body) => body["result"]["message_id"].as_i64(),
            Err(e) => {
                tracing::warn!(
                    status = ?status,
                    error = %e,
                    "sendMessage: unparseable response body — treating as delivered"
                );
                None
            }
        };
        Ok(id)
    }

    /// Send one Telegram text message, with optional `parse_mode`.
    /// Returns the HTTP status and response body on failure, or Ok(()) on success.
    async fn send_single_message(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        text: &str,
        parse_mode: Option<&str>,
        reply_markup: Option<serde_json::Value>,
    ) -> Result<(), (reqwest::StatusCode, String)> {
        self.send_message_get_id(chat_id, thread_id, text, parse_mode, reply_markup)
            .await
            .map(|_| ())
    }

    async fn send_text_chunks(
        &self,
        message: &str,
        chat_id: &str,
        thread_id: Option<&str>,
        reply_markup: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        // Convert Markdown to Telegram HTML once, then split.
        let html = to_telegram_html(message);
        let chunks = split_message_for_telegram(&html);

        for (index, chunk) in chunks.iter().enumerate() {
            let text = wrap_chunk(chunk, index, chunks.len());

            let chunk_reply_markup = if index == chunks.len() - 1 {
                reply_markup.clone()
            } else {
                None
            };

            if let Err((html_status, html_err)) = self
                .send_single_message(
                    chat_id,
                    thread_id,
                    &text,
                    Some("HTML"),
                    chunk_reply_markup.clone(),
                )
                .await
            {
                tracing::info!(
                    status = ?html_status,
                    "Telegram sendMessage with HTML parse_mode failed; retrying without parse_mode"
                );
                // Strip HTML tags so users don't see raw `<b>`, `<code>` etc.
                let clean_text = strip_html_tags(&text);
                self.send_single_message(chat_id, thread_id, &clean_text, None, chunk_reply_markup)
                    .await
                    .map_err(|(plain_status, plain_err)| {
                        anyhow::anyhow!(
                            "Telegram sendMessage failed (html {html_status}: {html_err}; plain {plain_status}: {plain_err})"
                        )
                    })?;
            }

            if index < chunks.len() - 1 {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        Ok(())
    }

    /// Edit a previously sent message's text. Returns the HTTP status and
    /// error body on failure — the caller classifies the error.
    async fn edit_message_text(
        &self,
        chat_id: &str,
        message_id: i64,
        text: &str,
        parse_mode: Option<&str>,
    ) -> Result<(), (reqwest::StatusCode, String)> {
        let mut body = serde_json::json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        });
        if let Some(mode) = parse_mode {
            body["parse_mode"] = serde_json::Value::String(mode.to_string());
        }

        self.post_telegram_json("editMessageText", body, "editMessageText error")
            .await
            .map(|_| ())
    }

    /// Pin a message in a chat. Pins are always silent in private chats;
    /// `disable_notification` additionally suppresses any service notice.
    async fn pin_chat_message(
        &self,
        chat_id: &str,
        message_id: i64,
    ) -> Result<(), (reqwest::StatusCode, String)> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "disable_notification": true,
        });

        self.post_telegram_json("pinChatMessage", body, "pinChatMessage error")
            .await
            .map(|_| ())
    }

    /// Send the role-switch success notification in a Telegram private chat,
    /// pinned at the top: the first switch sends + pins + persists the message
    /// id (per chat in `config_kv`); later switches edit it in place and
    /// re-affirm the pin. Fail-open: the text is always delivered (barring
    /// transport failure or a shutdown dropping the detached task) — a deleted
    /// or past-48h pin is re-sent and re-pinned, other edit failures deliver a
    /// plain message while the stored id is kept (only 'not found'/'can't be
    /// edited' recover), and a failed re-pin after a successful edit is only
    /// logged. Groups and threads get a plain notification (no pin).
    pub async fn send_role_switch_notification(&self, reply_target: &str, text: &str) {
        // Serialize per channel (the lock is held across the HTTP calls —
        // role switches are human-paced, so contention is negligible).
        let _guard = self.pin_lock.lock().await;
        let (chat_id, thread_id) = parse_recipient(reply_target);
        // Same command-menu "(current)" refresh the normal send path triggers.
        self.spawn_menu_refresh(chat_id);
        let html = to_telegram_html(text);

        // Pin only in private chats (positive chat_id) — groups require admin
        // rights and pinChatMessage has no thread parameter.
        if thread_id.is_some() || !chat_id.parse::<i64>().is_ok_and(|id| id > 0) {
            self.send_plain_role_notification(chat_id, thread_id, &html)
                .await;
            return;
        }

        let kv_key = format!("{ROLE_PIN_KV_PREFIX}{chat_id}");
        let stored_id = match crate::config_db::store().get_kv(&kv_key).await {
            Ok(Some(id)) => id.parse::<i64>().ok(),
            Ok(None) => None,
            Err(e) => {
                // DB unavailable — deliver plain, skip pinning; self-heals next switch.
                tracing::warn!(chat_id, error = %e, "Failed to read Telegram role pin id");
                self.send_plain_role_notification(chat_id, thread_id, &html)
                    .await;
                return;
            }
        };

        let Some(message_id) = stored_id else {
            self.send_and_pin_role_notification(chat_id, thread_id, &html, &kv_key)
                .await;
            return;
        };

        match self
            .edit_message_text(chat_id, message_id, &html, Some("HTML"))
            .await
        {
            Ok(()) => {}
            Err((_, desc)) => match classify_edit_failure(&desc) {
                EditMessageFailure::NotModified => {} // identical text — success/no-op
                EditMessageFailure::NotFound => {
                    tracing::warn!(
                        chat_id,
                        message_id,
                        "Telegram role pin deleted — re-sending"
                    );
                    self.send_and_pin_role_notification(chat_id, thread_id, &html, &kv_key)
                        .await;
                    return;
                }
                EditMessageFailure::CannotEdit => {
                    // Past the 48h edit window: re-send + re-pin a fresh id —
                    // it auto-replaces the stale pin, and a failed fresh pin
                    // self-heals on the next switch via the stored fresh id.
                    tracing::warn!(
                        chat_id,
                        message_id,
                        error = %desc,
                        "Telegram role pin no longer editable (48h window) — re-sending"
                    );
                    self.send_and_pin_role_notification(chat_id, thread_id, &html, &kv_key)
                        .await;
                    return;
                }
                EditMessageFailure::Other => {
                    // Unclassified failure — deliver the notification as a
                    // plain message and keep the stored id (no re-send; only
                    // 'not found'/'can't be edited' recover).
                    tracing::warn!(chat_id, message_id, error = %desc, "Telegram role pin edit failed — delivering plain notification");
                    self.send_plain_role_notification(chat_id, thread_id, &html)
                        .await;
                    return;
                }
            },
        }
        // Edited in place (or no-op) — re-affirm the pin (restores an
        // unpin-without-delete).
        if let Err((status, desc)) = self.pin_chat_message(chat_id, message_id).await {
            tracing::warn!(chat_id, message_id, status = ?status, error = %desc, "Telegram role pin re-pin failed");
        }
    }

    /// Low-level role notification send (HTML parse mode, no markup); returns
    /// the message id (`None` when the 2xx body omits it — the message was
    /// delivered, only the pin flow needs the id). The fixed plain-word text
    /// can't produce HTML that Telegram rejects, so the strip-tags retry the
    /// chunked send path uses is unnecessary here. The `_plain` and `_and_pin`
    /// wrappers differ in what they do with the id.
    async fn send_role_notification_html(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        html: &str,
    ) -> Result<Option<i64>, (reqwest::StatusCode, String)> {
        self.send_message_get_id(chat_id, thread_id, html, Some("HTML"), None)
            .await
    }

    /// Send the role notification as a plain (unpinned) message — the
    /// non-pinnable-context and fail-open fallback. Logs send failures.
    async fn send_plain_role_notification(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        html: &str,
    ) {
        if let Err((status, desc)) = self
            .send_role_notification_html(chat_id, thread_id, html)
            .await
        {
            tracing::warn!(chat_id, status = ?status, error = %desc, "Telegram role notification send failed");
        }
    }

    /// Send a fresh role notification and pin it. The id is persisted BEFORE
    /// pinning — a crash between send and persist leaves only a harmless
    /// unpinned duplicate; if persistence fails the pin is skipped so it can
    /// never be orphaned (self-heals on the next switch).
    async fn send_and_pin_role_notification(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        html: &str,
        kv_key: &str,
    ) {
        let message_id = match self
            .send_role_notification_html(chat_id, thread_id, html)
            .await
        {
            Ok(Some(id)) => id,
            Ok(None) => {
                tracing::warn!(
                    chat_id,
                    "Telegram role notification sent without a message id — cannot pin"
                );
                return;
            }
            Err((status, desc)) => {
                tracing::warn!(chat_id, status = ?status, error = %desc, "Telegram role notification send failed");
                return;
            }
        };
        if let Err(e) = crate::config_db::store()
            .set_kv(kv_key, &message_id.to_string())
            .await
        {
            tracing::warn!(chat_id, message_id, error = %e, "Failed to persist Telegram role pin id — skipping pin");
            return;
        }
        if let Err((status, desc)) = self.pin_chat_message(chat_id, message_id).await {
            tracing::warn!(chat_id, message_id, status = ?status, error = %desc, "Telegram role notification pin failed");
        }
    }

    async fn send_attachment(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        attachment: &TelegramAttachment,
    ) -> anyhow::Result<()> {
        let target = attachment.target.trim();

        if is_http_url(target) {
            let result = self
                .send_media_by_url(chat_id, thread_id, attachment.kind, target)
                .await;

            // If sending media by URL failed (e.g. Telegram can't fetch the URL,
            // wrong content type, etc.), fall back to sending the URL as a text link
            // instead of losing the reply entirely.
            if let Err(e) = result {
                tracing::warn!(
                    url = target,
                    error = %e,
                    "Telegram send media by URL failed; falling back to text link"
                );
                let fallback_text = format!("{}: {target}", attachment.kind.meta().label);
                self.send_text_chunks(&fallback_text, chat_id, thread_id, None)
                    .await?;
            }

            return Ok(());
        }

        let path = Path::new(&target);
        if !path.exists() {
            anyhow::bail!("Telegram attachment path not found: {target}");
        }

        self.send_media_file(chat_id, thread_id, attachment.kind, path)
            .await
    }

    /// Post a pre-built media request, check status, and log success.
    async fn send_media(
        &self,
        chat_id: &str,
        api_method: &'static str,
        request: reqwest::RequestBuilder,
        label: &str,
    ) -> anyhow::Result<()> {
        let resp = request.send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram {api_method} failed: {err}");
        }
        tracing::info!("Telegram {api_method} sent to {chat_id}: {label}");
        Ok(())
    }

    /// Send a media file (image-as-file/document/video/audio/voice) to a
    /// Telegram chat.
    async fn send_media_file(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        kind: TelegramAttachmentKind,
        file_path: &Path,
    ) -> anyhow::Result<()> {
        let meta = kind.file_meta();
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(meta.default_filename);

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part(meta.form_field, part);

        if meta.disable_content_type_detection {
            form = form.text("disable_content_type_detection", "true");
        }

        if let Some(tid) = thread_id {
            form = form.text("message_thread_id", tid.to_string());
        }

        let request = self
            .http_client()
            .post(self.api_url(meta.api_method))
            .multipart(form);

        self.send_media(chat_id, meta.api_method, request, file_name)
            .await
    }

    /// Send a file by URL (Telegram will download it).
    async fn send_media_by_url(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        kind: TelegramAttachmentKind,
        url: &str,
    ) -> anyhow::Result<()> {
        let meta = kind.meta();
        let mut body = serde_json::json!({ "chat_id": chat_id });
        body[meta.form_field] = serde_json::Value::String(url.to_string());

        set_thread_id_on_json(&mut body, thread_id);

        let request = self
            .http_client()
            .post(self.api_url(meta.api_method))
            .json(&body);

        self.send_media(chat_id, meta.api_method, request, url)
            .await
    }
}

/// Outcome of a single `getUpdates` poll.
enum PollOutcome {
    /// Successfully fetched updates (offset already advanced past them).
    Updates(Vec<serde_json::Value>),
    /// 409 Conflict — caller decides backoff.
    Conflict,
    /// Non-409 API error with description.
    Error(String),
    /// Network or parse error (sleep already applied by helper).
    Transport,
}

impl TelegramChannel {
    /// Call `getUpdates`, advance offset, and classify the outcome.
    ///
    /// `ok_default` controls what happens when the `ok` field is missing from
    /// the response: `false` (probe) treats it as an error, `true` (main loop)
    /// assumes success to be lenient.
    async fn poll_get_updates(
        &self,
        offset: &mut i64,
        timeout: u64,
        ok_default: bool,
    ) -> PollOutcome {
        let url = self.api_url("getUpdates");
        let body = serde_json::json!({
            "offset": *offset,
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"]
        });

        let resp = match self.http_client().post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Telegram poll error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                return PollOutcome::Transport;
            }
        };

        let data: serde_json::Value = match resp.json().await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Telegram parse error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                return PollOutcome::Transport;
            }
        };

        let ok = data
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(ok_default);

        if ok {
            if let Some(results) = data.get("result").and_then(serde_json::Value::as_array) {
                // Advance offset past these updates so they aren't re-delivered.
                for update in results {
                    if let Some(uid) = update.get("update_id").and_then(serde_json::Value::as_i64) {
                        *offset = (*offset).max(uid + 1);
                    }
                }
                return PollOutcome::Updates(results.clone());
            }
            // ok=true with no result array — rare, treat as empty.
            return PollOutcome::Updates(Vec::new());
        }

        let error_code = data
            .get("error_code")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_default();
        if error_code == 409 {
            PollOutcome::Conflict
        } else {
            let desc = data
                .get("description")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown Telegram API error");
            PollOutcome::Error(desc.to_string())
        }
    }

    /// Probe: claim the `getUpdates` slot before entering the long-poll loop.
    ///
    /// A previous daemon's 30-second poll may still be active on Telegram's server.
    /// We retry with `timeout=0` until we receive a successful (non-409) response,
    /// confirming the slot is ours.
    ///
    /// Returns `true` if the probe succeeded, `false` if cancelled (caller should
    /// return `Ok(())` from `listen`).
    async fn probe_startup_slot(&self, offset: &mut i64) -> bool {
        loop {
            if self.cancel.is_cancelled() {
                tracing::info!("Telegram channel cancelled during startup probe");
                return false;
            }
            match self.poll_get_updates(offset, 0, false).await {
                PollOutcome::Updates(_) => return true,
                PollOutcome::Conflict => {
                    tracing::debug!("Startup probe: slot busy (409), retrying in 5s");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                PollOutcome::Error(desc) => {
                    tracing::warn!("Startup probe: API error: {desc}; retrying in 5s");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                PollOutcome::Transport => {} // sleep already applied by helper
            }
        }
    }

    /// Process a batch of Telegram updates, sending parsed messages through the
    /// message pipeline.
    ///
    /// Handles text messages (via [`Self::parse_update_message`] /
    /// [`Self::try_parse_attachment_message`]), callback queries, and photo album
    /// buffering (media groups are merged into a single message).
    ///
    /// Returns `true` if the pipeline is still alive, `false` if the channel was
    /// closed (`tx.send()` failed) — the caller should exit the long-poll loop.
    async fn process_updates(
        &self,
        tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
        updates: Vec<serde_json::Value>,
    ) -> bool {
        let mut album_groups: HashMap<String, Vec<ChannelMessage>> = HashMap::new();

        for update in updates {
            // Check for callback_query first — it has a different structure
            if let Some(cq) = update.get("callback_query") {
                let cq_id = cq["id"].as_str().map(ToString::to_string);
                let cq_data = cq["data"].as_str().unwrap_or("");

                // For __act__ callbacks, do NOT answer early — the action handler
                // (handle_action_callback in main.rs) will answer with the appropriate
                // toast text. Dismiss the spinner now for all other callbacks.
                if !cq_data.starts_with(ACTION_PREFIX)
                    && let Some(ref id) = cq_id
                {
                    self.answer_callback_query(id, None).await;
                }

                let Some(msg) = self.parse_callback_query(cq).await else {
                    continue;
                };
                if tx.send(msg).await.is_err() {
                    return false;
                }
                continue;
            }

            let msg = if let Some(m) = self.parse_update_message(&update).await {
                m
            } else if let Some(m) = self.try_parse_attachment_message(&update).await {
                m
            } else {
                Self::handle_non_parseable_message(&update);
                continue;
            };

            // Send ACK reaction for every individual update (fire-and-forget)
            if let Some((reaction_chat_id, reaction_message_id)) =
                Self::extract_update_message_target(&update)
            {
                self.try_add_ack_reaction_nonblocking(reaction_chat_id, reaction_message_id);
            }

            // Check for media group (album) membership
            let media_group_id = update
                .get("message")
                .and_then(|m| m.get("media_group_id"))
                .and_then(|v| v.as_str())
                .map(String::from);

            if let Some(group_id) = media_group_id {
                // Buffer — combine after collecting all group members
                album_groups.entry(group_id).or_default().push(msg);
            } else {
                // Not part of a media group — send immediately
                if tx.send(msg).await.is_err() {
                    return false;
                }
            }
        }

        // Flush all buffered album groups — combine content with \n separator
        for (_group_id, group_messages) in album_groups.drain() {
            // Merge messages: use the first message as template, concatenate content
            let merged = group_messages
                .into_iter()
                .reduce(|mut acc, next| {
                    acc.content.push('\n');
                    acc.content.push_str(&next.content);
                    acc
                })
                .unwrap();
            if tx.send(merged).await.is_err() {
                return false;
            }
        }

        true
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &'static str {
        "telegram"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
        let content = message.content.trim();
        if content.is_empty() {
            tracing::warn!("TelegramChannel: attempted to send empty message – skipping");
            return Ok(()); // nothing to send, not an error
        }

        // Parse recipient: "chat_id" or "chat_id:thread_id" format
        let (chat_id, thread_id) = parse_recipient(&message.recipient);

        // Per-user command menu refresh — fire-and-forget, fail-open; never
        // blocks or affects message delivery.
        self.spawn_menu_refresh(chat_id);

        // Look for inline attachment markers like [IMAGE:path/to/file.png].
        // Marker parsing now runs the shared classifier, whose local-file branch
        // is a blocking raster decode — offload it so a Tokio worker is not
        // parked while a local image is decoded for attachment.
        let (text_without_markers, attachments) =
            crate::util::with_block_in_place(|| parse_attachment_markers(content));

        if !attachments.is_empty() {
            if !text_without_markers.is_empty() {
                self.send_text_chunks(
                    &text_without_markers,
                    chat_id,
                    thread_id,
                    message.reply_markup.clone(),
                )
                .await?;
            }

            for attachment in &attachments {
                self.send_attachment(chat_id, thread_id, attachment).await?;
            }

            return Ok(());
        }

        if let Some(attachment) =
            crate::util::with_block_in_place(|| parse_path_only_attachment(content))
        {
            self.send_attachment(chat_id, thread_id, &attachment)
                .await?;
            return Ok(());
        }

        self.send_text_chunks(content, chat_id, thread_id, message.reply_markup.clone())
            .await
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        let mut offset = self.offset.load(Ordering::Acquire);
        if offset > 0 {
            tracing::info!(offset, "Telegram channel resuming from previous offset");
        }

        tracing::info!("Telegram channel listening for messages...");

        // Startup probe: claim the getUpdates slot before entering the long-poll loop.
        if !self.probe_startup_slot(&mut offset).await {
            return Ok(());
        }

        tracing::debug!("Startup probe succeeded; entering main long-poll loop.");
        let shutdown_token = crate::shutdown::shutdown_token();
        let per_channel_cancel = self.cancel.clone();

        loop {
            tokio::select! {
                () = shutdown_token.cancelled() => {
                    tracing::info!("Telegram channel shutting down (global shutdown)");
                    self.offset.store(offset, Ordering::Release);
                    return Ok(());
                }
                () = per_channel_cancel.cancelled() => {
                    tracing::info!("Telegram channel shutting down (token hot-reload)");
                    self.offset.store(offset, Ordering::Release);
                    return Ok(());
                }
                poll_result = self.poll_get_updates(&mut offset, 30, true) => {
                    // Persist offset after each successful poll so a
                    // hot-reloaded listener can resume from here.
                    self.offset.store(offset, Ordering::Release);

                    let updates = match poll_result {
                        PollOutcome::Updates(updates) => updates,
                        PollOutcome::Conflict => {
                            tracing::warn!(
                                "Telegram polling conflict (409). \
                                 Ensure only one `mahbot` process is using this bot token."
                            );
                            tokio::time::sleep(std::time::Duration::from_secs(35)).await;
                            continue;
                        }
                        PollOutcome::Error(desc) => {
                            tracing::warn!("Telegram getUpdates API error: {desc}");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            continue;
                        }
                        PollOutcome::Transport => continue,
                    };

                    if !self.process_updates(&tx, updates).await {
                        return Ok(());
                    }
                }
            }
        }
    }

    // -- typing indicators --

    async fn start_typing(&self, recipient: &str) -> anyhow::Result<()> {
        let url = self.api_url("sendChatAction");
        let (chat_id, thread_id) = parse_recipient(recipient);
        let mut body = serde_json::json!({
            "chat_id": chat_id,
            "action": "typing"
        });
        set_thread_id_on_json(&mut body, thread_id);
        self.http_client().post(&url).json(&body).send().await?;
        Ok(())
    }
}
fn cancel_old_listener(old: Option<&std::sync::Arc<dyn Channel>>) {
    if let Some(old) = old
        && let Some(tc) = old.as_any().downcast_ref::<TelegramChannel>()
    {
        tc.cancel_own();
    }
}
/// Hot-reload the Telegram bot listener with a new token.
///
/// Used when the user changes the bot token in Settings — no full application
/// restart required.
///
/// # Behaviour by token state
///
/// - **New token provided** (`Some(token)`):
///   1. Validate the token via `getMe`
///   2. Inherit the update offset from the old listener (if any) to avoid replay
///   3. Atomically swap the channel in the registry (no gap)
///   4. Cancel the old listener
///   5. Spawn the new listener on the shared message pipeline
///
/// - **Token cleared** (`None`):
///   1. Cancel the old listener
///   2. Remove the channel from the registry
///
/// # Cancel-safety
///
/// Cancelling the returned future may leave the listener in an intermediate
/// state. This function should be awaited to completion.
pub async fn restart_telegram_listener(new_token: Option<&str>) -> anyhow::Result<()> {
    let registry = crate::channel_registry();
    let old_channel = registry.get("telegram");

    if let Some(token) = new_token.filter(|t| !t.trim().is_empty()) {
        // Validate the new token before touching anything.
        TelegramChannel::validate_token(token).await?;

        // Inherit the update offset from the old listener to avoid
        // replaying already-processed updates from Telegram's server.
        let offset = old_channel
            .as_ref()
            .and_then(|c| c.as_any().downcast_ref::<TelegramChannel>())
            .map(|tc| std::sync::Arc::clone(&tc.offset));

        // Create the new channel with the inherited offset.
        let telegram_channel = if let Some(inherited_offset) = offset {
            TelegramChannel::with_offset(token.to_string(), inherited_offset)
        } else {
            TelegramChannel::new(token.to_string())
        };

        let telegram_arc = std::sync::Arc::new(telegram_channel);

        // Register commands with Telegram API (fire-and-forget, non-blocking).
        tokio::spawn({
            let tc = std::sync::Arc::clone(&telegram_arc);
            async move {
                tc.set_my_commands().await;
            }
        });

        let new_channel: std::sync::Arc<dyn Channel> = telegram_arc;

        // Atomically replace in the registry — no gap where
        // "telegram" returns None.
        registry.replace(std::sync::Arc::clone(&new_channel));

        // Cancel the old listener now that the registry has the new one.
        cancel_old_listener(old_channel.as_ref());

        // Spawn the new listener on the shared message pipeline.
        if let Some(tx) = crate::MESSAGE_TX.get() {
            let tx = tx.clone();
            tokio::spawn(async move {
                if let Err(e) = new_channel.listen(tx).await {
                    tracing::error!(error = %e, "Telegram listener error after hot-reload");
                }
            });
        } else {
            tracing::error!("MESSAGE_TX not set — cannot spawn Telegram listener");
        }

        tracing::info!("Telegram bot listener restarted with new token");
    } else {
        // Token cleared — stop the old listener and unregister.
        cancel_old_listener(old_channel.as_ref());
        registry.unregister("telegram");
        tracing::info!("Telegram bot token cleared — listener stopped");
    }

    Ok(())
}

/// Send a message directly through the registered Telegram channel, bypassing
/// the router (no broadcast/persist). Errors if the channel is missing from
/// the registry or the transport send fails.
pub async fn send_direct(
    recipient: &str,
    content: String,
    reply_markup: Option<serde_json::Value>,
) -> anyhow::Result<()> {
    let Some(channel) = crate::channel_registry().get("telegram") else {
        anyhow::bail!("Telegram channel not found in registry");
    };
    let reply = SendMessage {
        content,
        recipient: recipient.to_string(),
        reply_markup,
    };
    channel.send(&reply).await
}

/// Send a plain-text message directly via the registered Telegram channel,
/// ignoring transport/channel errors (best-effort). Used for command replies
/// and notifications where a send failure should not fail the caller.
pub async fn send_reply(recipient: &str, content: &str) {
    let _ = send_direct(recipient, content.to_string(), None).await;
}

/// Mirror a local user's message to their Telegram chats as a blockquote, so conversation history is readable from both surfaces.
///
/// This should be called before enrichment to preserve the original
/// user-typed text (pre-link-summary, pre-transcription).
///
/// # Guards
///
/// * Only mirrors messages where `channel == "gui"` or `channel == "voice"`
///   (prevents echo loops: voice is a strictly local source that can never
///   originate from Telegram, so accepting it cannot create feedback;
///   Telegram-originated messages remain excluded).
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
    // Guard: only mirror local-originated user messages — GUI text and local
    // voice transcripts (prevents echo loops: voice is a strictly local source
    // that can never originate from Telegram, so accepting it cannot create
    // feedback; Telegram-originated messages remain excluded).
    if msg.channel != "gui" && msg.channel != "voice" {
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
                "Failed to look up user channels for message mirror"
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
    let content = TELEGRAM_MEDIA_MARKER_RE
        .replace_all(trimmed, "")
        .to_string();
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
        };

        if let Err(e) = channel.send(&reply).await {
            tracing::error!(
                user = %msg.user_name,
                recipient = %reply_target,
                error = %e,
                "Failed to mirror local message to Telegram"
            );
        }
    }
}

/// POST a `setMyCommands` request. Returns `true` on success; failures are
/// logged (warn) and reported as `false` so callers can reset their
/// change-detection state and retry on the next message.
async fn post_set_my_commands(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
) -> bool {
    match client.post(url).json(body).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("Telegram bot commands registered successfully");
            true
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(
                status = %status,
                body = %body,
                "Telegram setMyCommands returned unsuccessful status"
            );
            false
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to call Telegram setMyCommands");
            false
        }
    }
}

/// Format one `/board` listing line: bold state, monospace ticket ID, then
/// the title. Shared by the Telegram `/board` handler and its tests so the
/// format cannot silently drift between them.
#[must_use]
pub fn format_board_line(
    phase: &crate::pipeline::board::TicketPhase,
    id: &str,
    title: &str,
) -> String {
    format!("• **{}** `{}` {}", phase.display_name(), id, title)
}

/// (command, description) entries for a user's Telegram command menu,
/// derived from their role pool, active role, admin status, and the state of
/// their selected shared workspace. Shared by the per-chat `setMyCommands`
/// refresh and the `/start` welcome message.
///
/// State-aware entries: exactly one of `/pause` or `/unpause` appears (the
/// one matching the workspace's paused state), and exactly one of
/// `/maintenance_on` or `/maintenance_off`. Admins without a selected shared
/// workspace get neither pair (there is no workspace state to reflect).
#[must_use]
pub async fn user_command_entries(user_name: &str) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::new();

    let pool = crate::users::role_pool(user_name).await;
    let active_role = crate::users::resolve_active_role_from_pool(user_name, &pool).await;

    // Each pool role is a direct command; the active role's entry is marked.
    // Role switches lead the menu — the most frequent action.
    for role in &pool {
        let label = crate::agent::role::role_info(role).display_label;
        let desc = if Some(*role) == active_role {
            format!("{label} (current)")
        } else {
            label.to_string()
        };
        entries.push((role.as_str().to_string(), desc));
    }

    if crate::users::is_admin(user_name).await {
        entries.push(("board".to_string(), BOARD_COMMAND_DESC.to_string()));
        entries.push(("archive".to_string(), ARCHIVE_COMMAND_DESC.to_string()));

        // `/update` is global (any full-permission admin) and shown only when
        // the shared availability cache confirms an update — always in
        // local-checkout mode, registry mode only when a strictly newer stable
        // version exists. The menu reflects the cached state, never a network
        // call per refresh.
        if crate::self_update::should_show_update(crate::self_update::update_availability(), true) {
            entries.push(("update".to_string(), UPDATE_COMMAND_DESC.to_string()));
        }

        // Workspace-state entries follow the selected shared workspace; a
        // personal workspace (or lookup failure) omits both pairs.
        if let Ok(Some(ws_name)) = crate::users::get_raw_selected_workspace(user_name).await
            && let Ok(Some(ws)) = crate::workspace::get_by_name(&ws_name).await
        {
            if ws.paused {
                entries.push(("unpause".to_string(), UNPAUSE_COMMAND_DESC.to_string()));
            } else {
                entries.push(("pause".to_string(), PAUSE_COMMAND_DESC.to_string()));
            }
            if ws.maintenance_enabled {
                entries.push((
                    "maintenance_off".to_string(),
                    MAINTENANCE_OFF_COMMAND_DESC.to_string(),
                ));
            } else {
                entries.push((
                    "maintenance_on".to_string(),
                    MAINTENANCE_ON_COMMAND_DESC.to_string(),
                ));
            }
        }
    }

    // Artist model-selection commands are available whenever Artist is in the
    // user's pool (they can switch to Artist and use them).
    if pool.contains(&Role::Artist) {
        entries.push((
            "image_models".to_string(),
            IMAGE_MODELS_COMMAND_DESC.to_string(),
        ));
        entries.push((
            "video_models".to_string(),
            VIDEO_MODELS_COMMAND_DESC.to_string(),
        ));
    }

    // Session clear is the least frequent action — keep it last.
    entries.push(("clear".to_string(), CLEAR_COMMAND_DESC.to_string()));
    entries
}

#[cfg(test)]
#[path = "telegram_tests.rs"]
mod tests;
