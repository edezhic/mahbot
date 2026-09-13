use crate::channels::ReplyReference;
use crate::channels::reply::normalize_reply_text;
use crate::util::html::{decode_html_entities, escape_html, push_escaped};
use crate::util::media_target::{self, MediaTarget};
use crate::util::{
    FILE_MAX_BYTES, MediaMarkerKind, TELEGRAM_MEDIA_MARKER_RE, UnwrapPoison, file_name_or_path,
    is_http_url, parse_media_marker,
};
use crate::{Channel, ChannelMessage, SendMessage};
use anyhow::Context;
use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;
use strum::IntoEnumIterator;

/// Telegram's maximum message length for text messages
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;
/// Reserve space for continuation markers added by `send_text_chunks`:
/// worst case is "(continued)\n\n" + chunk + "\n\n(continues...)" = 29 extra
/// chars (middle chunk); 30 keeps 4066+29 = 4095 within the 4096 limit.
const TELEGRAM_CONTINUATION_OVERHEAD: usize = 30;

/// Byte budget for a sanitized attachment filename, extension included. Bytes
/// rather than characters because the filesystem's own per-component limit
/// (`NAME_MAX`, 255 here) counts bytes: a character cap would admit a name that
/// only fails at the write. See [`sanitize_attachment_filename`].
const MAX_ATTACHMENT_FILENAME_BYTES: usize = 180;

/// Per-request timeout for a media transfer, overriding the channel client's
/// one-minute default: [`FILE_MAX_BYTES`] in either direction takes longer than
/// a minute over a slow link. Still bounded, so a stalled transfer cannot hold
/// the request open forever.
const MEDIA_TRANSFER_TIMEOUT: Duration = Duration::from_mins(5);

/// Description for the `/clear` command — used in `setMyCommands` API and `/start` welcome message.
const CLEAR_COMMAND_DESC: &str = "Reset your session";
/// Description for the `/image_models` command.
const IMAGE_MODELS_COMMAND_DESC: &str = "Select image generation model";
/// Description for the `/video_models` command.
const VIDEO_MODELS_COMMAND_DESC: &str = "Select video model";
/// Description for the `/board` command (admin).
const BOARD_COMMAND_DESC: &str = "List active workspace tickets";
/// Description for the `/archive` command (admin).
const ARCHIVE_COMMAND_DESC: &str = "Archive done & cancelled tickets";
/// Description for the `/pause` command (admin).
const PAUSE_COMMAND_DESC: &str = "Pause the workspace pipeline";
/// Description for the `/unpause` command (admin).
const UNPAUSE_COMMAND_DESC: &str = "Resume the workspace pipeline";
/// Description for the `/maintenance_on` command (admin, menu form).
const MAINTENANCE_ON_COMMAND_DESC: &str = "Enable workspace maintenance";
/// Description for the `/maintenance_off` command (admin, menu form).
const MAINTENANCE_OFF_COMMAND_DESC: &str = "Disable workspace maintenance";
/// Description for the `/update` command (admin, menu form).
const UPDATE_COMMAND_DESC: &str = "Update MahBot to the latest version";

// ── Action prefixes (__act__) ───────────────────────────────────────

/// Callback data prefix for action callbacks (e.g., model selection, clear session).
const ACTION_PREFIX: &str = "__act__";

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
    /// Reply reference parsed from the update's `reply_to_message` — `None`
    /// when this message does not reply to anything.
    reply_reference: Option<ReplyReference>,
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
            reply_reference: self.reply_reference,
            chat_id: (!self.chat_id.is_empty()).then_some(self.chat_id),
            message_id: (self.message_id != 0).then_some(self.message_id),
            attachment_dirs: Vec::new(),
        }
    }

    /// Build the channel message for a downloaded attachment, recording the
    /// staging directory the bytes were written into as the only path
    /// enrichment may later read, copy from, and delete.
    fn into_attachment_message(self, content: String, staging_dir: String) -> ChannelMessage {
        ChannelMessage {
            attachment_dirs: vec![staging_dir],
            ..self.into_channel_message(content, None)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TelegramAttachmentKind {
    Image,
    /// The `[FILE:...]` outbound kind — sent via `sendDocument`. Local image
    /// files also route through `sendDocument`; see [`Self::file_meta`].
    Document,
    Video,
    Audio,
}

impl From<MediaMarkerKind> for TelegramAttachmentKind {
    fn from(kind: MediaMarkerKind) -> Self {
        match kind {
            MediaMarkerKind::Image => Self::Image,
            MediaMarkerKind::Audio => Self::Audio,
            MediaMarkerKind::Video => Self::Video,
            MediaMarkerKind::File => Self::Document,
        }
    }
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
    /// Telegram does not re-classify the upload server-side: an image sent as a
    /// general file would be re-encoded as a photo despite the sendDocument
    /// path, and a file the assistant promised to send as a document would come
    /// back as media.
    disable_content_type_detection: bool,
}

impl TelegramAttachmentKind {
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
                disable_content_type_detection: true,
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

/// Shared username → `@u`, else `first_name` label chain. Used by both
/// [`format_sender_label`] and the non-bot branch of [`replied_to_sender_label`];
/// they differ only in the terminal fallback sentinel.
fn sender_label_with_fallback(from: &serde_json::Value, fallback: &str) -> String {
    from.get("username")
        .and_then(serde_json::Value::as_str)
        .map_or_else(
            || {
                from.get("first_name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(fallback)
                    .to_string()
            },
            |u| format!("@{u}"),
        )
}

/// Format a sender label for display: `@username` if a username is present,
/// otherwise the display name (first_name, or `"unknown"` as ultimate fallback).
#[must_use]
fn format_sender_label(from: &serde_json::Value) -> String {
    sender_label_with_fallback(from, crate::users::TELEGRAM_UNKNOWN_SENTINEL)
}

/// Resolve the display label for a replied-to message's author.
///
/// The fallback chain is deliberately distinct from [`format_sender_label`],
/// which keeps the `'unknown'` sentinel: a reply quote header must never
/// surface `'unknown'`, so the ultimate fallback is `"user"`.
///
/// 1. `sender_chat` present → its `title`, else `"channel"`.
/// 2. `from` present → if `is_bot`: `@username` else `"bot"`; otherwise
///    `@username` → `first_name` → `"user"`.
/// 3. Neither → `"user"`.
///
/// `<` / `>` are stripped from the resulting label since `first_name` /
/// `title` may contain user-controlled angle brackets.
#[must_use]
fn replied_to_sender_label(message: &serde_json::Value) -> String {
    let label = if let Some(sender_chat) = message.get("sender_chat") {
        sender_chat
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("channel")
            .to_string()
    } else if let Some(from) = message.get("from") {
        if from
            .get("is_bot")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            from.get("username")
                .and_then(serde_json::Value::as_str)
                .map_or_else(|| "bot".to_string(), |u| format!("@{u}"))
        } else {
            sender_label_with_fallback(from, "user")
        }
    } else {
        "user".to_string()
    };
    label.replace(['<', '>'], "")
}

/// Placeholder text for a media message reply whose text/caption is empty.
///
/// The mapping mirrors the original reply-quote logic: photo / voice/audio /
/// video / document / sticker get a specific label, everything else
/// (animation, video_note, etc.) falls back to `[Message]`.
#[must_use]
fn media_kind_placeholder(reply_to: &serde_json::Value) -> String {
    if reply_to.get("voice").is_some() || reply_to.get("audio").is_some() {
        "[Voice message]".to_string()
    } else if reply_to.get("photo").is_some() {
        "[Photo]".to_string()
    } else if reply_to.get("document").is_some() {
        "[Document]".to_string()
    } else if reply_to.get("video").is_some() {
        "[Video]".to_string()
    } else if reply_to.get("sticker").is_some() {
        "[Sticker]".to_string()
    } else {
        "[Message]".to_string()
    }
}

/// Derive a reply snippet from a `reply_to_message` object.
///
/// Text/caption snippets are normalized via [`normalize_reply_text`]; media
/// messages with no text/caption use [`media_kind_placeholder`].
#[must_use]
fn reply_to_snippet(reply_to: &serde_json::Value) -> String {
    if let Some(text) = reply_to.get("text").and_then(serde_json::Value::as_str)
        && !text.is_empty()
    {
        normalize_reply_text(text)
    } else if let Some(caption) = reply_to.get("caption").and_then(serde_json::Value::as_str)
        && !caption.is_empty()
    {
        normalize_reply_text(caption)
    } else {
        media_kind_placeholder(reply_to)
    }
}

/// Build a [`ReplyReference`] from a Telegram message's `reply_to_message`,
/// when present. Returns `None` when the message does not reply to another
/// message.
///
/// The snippet is derived from the replied-to message's text/caption
/// (normalized via [`normalize_reply_text`]) or, for media messages with no
/// text, a media-kind placeholder. The author label comes from
/// [`replied_to_sender_label`].
#[must_use]
fn build_reply_reference(message: &serde_json::Value) -> Option<ReplyReference> {
    let reply_to = message.get("reply_to_message")?;
    if !reply_to.is_object() {
        return None;
    }
    Some(ReplyReference {
        author: replied_to_sender_label(reply_to),
        snippet: reply_to_snippet(reply_to),
    })
}

/// Build the user-facing content string for an incoming attachment.
///
/// The kind and the file name/MIME type decide the marker: photos with a
/// recognized image extension (or image MIME type) use `[IMAGE:/path]` so
/// enrichment can convert them to native image parts; videos (native
/// `video`/`video_note`/`animation` messages, or documents with a video
/// MIME/extension) use `[VIDEO:/path]`; voice and audio messages use
/// `[AUDIO:/path]`. Everything else uses `[FILE:/path]` — after one last check
/// of the file's own bytes, since a sender can omit both the name and the MIME
/// type.
fn format_attachment_content(
    kind: IncomingAttachmentKind,
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
            // Neither the name nor the declared MIME type classified this
            // attachment. The bytes still can, through the same decodable-raster
            // gate enrichment applies to `[IMAGE:...]`, so an image the sender
            // stripped the name and type from is still fed to the model as an
            // image instead of being reported as an unconvertible file. That gate
            // reads and decodes the file, so it is offloaded — this runs on the
            // Telegram listener task.
            let target = local_path.to_string_lossy().to_string();
            if crate::util::with_block_in_place(|| {
                crate::util::media_target::classify_media_image_target(&target)
            }) == MediaTarget::LocalImage
            {
                format!("[IMAGE:{}]", local_path.display())
            } else {
                format!("[FILE:{}]", local_path.display())
            }
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
        // No stem (`.`, `..`, `""`): nothing to swap, and the sanitizer replaces
        // the name with the generated fallback either way.
        let Some(stem) = std::path::Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
        else {
            return filename.to_string();
        };
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

/// Make an attachment filename safe to use as a single path component under the
/// Telegram temp root: only the final path component survives
/// (`../../etc/passwd` → `passwd`), [`crate::util::neutralized_name`] maps out
/// the characters that would break a path component or the `[FILE:<path>]`
/// marker carrying it, and the result fits [`MAX_ATTACHMENT_FILENAME_BYTES`]
/// bytes on a char boundary. The sender's name is used when it yields one, and
/// the caller's `fallback` (cleaned the same way) otherwise.
fn sanitize_attachment_filename(name: &str, fallback: &str) -> String {
    let mut cleaned = [name, fallback]
        .into_iter()
        .find_map(|name| {
            // `Path::file_name` yields only a normal component, so it is already
            // `None` for `""`, `.`, `..`, a bare separator and a non-UTF-8 name.
            let component = Path::new(name).file_name().and_then(|n| n.to_str())?;
            Some(crate::util::neutralized_name(component))
        })
        .unwrap_or_else(|| "file".to_string());
    // Prefix before truncating so the `_` counts toward the budget too.
    if cleaned.starts_with('.') {
        cleaned.insert(0, '_');
    }
    if cleaned.len() > MAX_ATTACHMENT_FILENAME_BYTES {
        // Split off the extension (dot included) so truncating the stem never
        // eats it; a name with no extension has the whole name as its stem and
        // truncates wholesale.
        let stem = crate::util::name_stem(&cleaned);
        let ext = &cleaned[stem.len()..];
        // A stem with no room left in the budget leaves a bare extension — a
        // dotfile — so the whole name truncates instead, as it does when there
        // is no stem to keep.
        let truncated = crate::util::truncate_bytes(
            stem,
            MAX_ATTACHMENT_FILENAME_BYTES.saturating_sub(ext.len()),
        );
        cleaned = if truncated.is_empty() {
            crate::util::truncate_bytes(cleaned.as_str(), MAX_ATTACHMENT_FILENAME_BYTES).to_string()
        } else {
            format!("{truncated}{ext}")
        };
    }
    cleaned
}

/// Reduce a `getFile`-supplied extension to an ASCII-alphanumeric token, or
/// `None` when nothing survives: it is remote input that reaches the user's
/// filename, so separators, control characters and `]` must not survive into
/// it.
fn sanitize_remote_extension(remote_ext: &str) -> Option<String> {
    let cleaned: String = remote_ext
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Resolve the local save name for an inbound attachment: the sender-supplied
/// `file_name` when Telegram provided one, otherwise a generated
/// `<kind>_<chat>_<message>.<ext>` fallback from `remote_ext` (the extension
/// from `getFile`, which is absent before that call). Either way the name goes
/// through [`sanitize_attachment_filename`], so omitting `file_name` cannot
/// smuggle a hostile extension past it.
fn local_attachment_name(
    attachment: &IncomingAttachment,
    chat_id: &str,
    message_id: i64,
    remote_ext: Option<&str>,
) -> String {
    let remote_ext = remote_ext.and_then(sanitize_remote_extension);
    let ext = remote_ext.as_deref().unwrap_or(match attachment.kind {
        // video_note has no file_name/mime_type — a ".jpg" default
        // would misroute it out of the video flow.
        IncomingAttachmentKind::Video => "mp4",
        // A document with no name and no MIME type cannot be routed by
        // `format_attachment_content`, so it takes the generic file path; the
        // default stays format-neutral rather than claiming "jpg".
        IncomingAttachmentKind::Document => "bin",
        _ => "jpg",
    });
    let prefix = match attachment.kind {
        IncomingAttachmentKind::Photo => "photo",
        IncomingAttachmentKind::Video => "video",
        IncomingAttachmentKind::Audio => "audio",
        IncomingAttachmentKind::Document => "file",
    };
    let fallback = format!("{prefix}_{chat_id}_{message_id}.{ext}");
    // The video extension swap runs before sanitizing: it can lengthen the name
    // (`.bin` → `.webm`), and the sanitizer's byte budget must cover the name
    // that is actually written.
    let chosen = normalize_video_filename(
        attachment.kind,
        attachment.file_name.as_deref().unwrap_or(&fallback),
        attachment.mime_type.as_deref(),
    );
    sanitize_attachment_filename(&chosen, &fallback)
}

/// Agent-facing content for a rejected attachment: the caption (when non-empty)
/// first, then the `[File <name>: <outcome> — <reason>]` note.
#[must_use]
fn attachment_rejection_content(
    display_name: &str,
    outcome: &str,
    reason: &str,
    caption: Option<&str>,
) -> String {
    let note = format!("[File {display_name}: {outcome} — {reason}]");
    match caption.filter(|c| !c.is_empty()) {
        Some(caption) => format!("{caption}\n\n{note}"),
        None => note,
    }
}

/// Attachment-target acceptance gate used by `parse_attachment_markers`:
/// image markers may only attach a real raster or an http(s) URL (via
/// `media_target::classify_media_image_target`); video and audio accept an
/// http(s) URL or any existing regular file.
///
/// Deliberately looser than `[FILE:...]`, which [`resolve_file_target`] resolves
/// against the authoring workspace roots: only that kind is workspace-confined.
fn is_attachable_target(kind: TelegramAttachmentKind, target: &str) -> bool {
    match kind {
        TelegramAttachmentKind::Image => matches!(
            media_target::classify_media_image_target(target),
            MediaTarget::LocalImage | MediaTarget::RemoteUrl
        ),
        TelegramAttachmentKind::Video | TelegramAttachmentKind::Audio => {
            is_http_url(target) || Path::new(target).is_file()
        }
        // Unreachable: only `[FILE:...]` yields a document, and it returns above.
        // Refused rather than accepted so a new marker route cannot attach one
        // without the containment this gate does not do.
        TelegramAttachmentKind::Document => false,
    }
}

/// Build the user-facing notice for a refused outbound file delivery.
fn file_refusal(name: &str, reason: &str) -> String {
    format!("Could not send \"{name}\": {reason}")
}

/// Resolve an outbound `[FILE:...]` target against the roots that authorize it.
/// A URL is rejected: the marker is local-only.
///
/// The target is canonicalized (a symlink cannot smuggle out a file from
/// outside a root) and must be inside one of the roots BEFORE the existence,
/// directory and size checks, so a refusal does not disclose which of those the
/// target is. The notice names a bounded prefix of the target, which a
/// malformed marker can make an arbitrary blob. `Err` is that notice.
fn resolve_file_target(target: &str, file_roots: &[PathBuf]) -> Result<PathBuf, String> {
    let name = crate::util::truncate(file_name_or_path(target), 60);

    // `[FILE:...]` is a local-only mechanism; a URL belongs to the other kinds.
    if is_http_url(target) {
        return Err(file_refusal(
            &name,
            "FILE targets are local files, not URLs.",
        ));
    }
    if file_roots.is_empty() {
        return Err(file_refusal(
            &name,
            "no workspace is available for file delivery.",
        ));
    }

    // Containment compares like with like, so the roots are canonicalized too:
    // a workspace living under a symlinked path (macOS `/tmp`, `/var`) would
    // otherwise never contain the canonical target. A root that cannot be
    // canonicalized is kept as given.
    let roots: Vec<PathBuf> = file_roots
        .iter()
        .map(|root| std::fs::canonicalize(root).unwrap_or_else(|_| root.clone()))
        .collect();

    let raw = Path::new(target);
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        // A relative target belongs to whichever root actually has it: relayed
        // content names a file in the originating workspace, which is the
        // second root. When no root has it, the first one backs the
        // "file not found." refusal.
        roots
            .iter()
            .map(|root| root.join(raw))
            .find(|candidate| candidate.exists())
            .unwrap_or_else(|| roots[0].join(raw))
    };

    // Containment comes before the directory and size checks: a path outside
    // the workspace must not have those disclosed to the end user. (Whether it
    // exists at all is still visible in the refusal text.)
    let Ok(canonical) = std::fs::canonicalize(&joined) else {
        return Err(file_refusal(&name, "file not found."));
    };
    if !crate::tools::path::is_path_under_roots(&canonical, &roots) {
        return Err(file_refusal(&name, "it is outside your workspace."));
    }

    let Ok(meta) = std::fs::metadata(&canonical) else {
        return Err(file_refusal(&name, "file not found."));
    };
    if meta.is_dir() {
        return Err(file_refusal(&name, "it is a directory."));
    }
    if meta.len() > FILE_MAX_BYTES {
        return Err(file_refusal(
            &name,
            &format!(
                "it is {} MB, larger than the {} MB limit.",
                meta.len().div_ceil(1024 * 1024),
                mb(FILE_MAX_BYTES)
            ),
        ));
    }

    Ok(canonical)
}

/// Parse `[KIND:path]` media markers from a message, returning cleaned text
/// (with markers removed), extracted attachments, and user-facing refusal
/// notices for `[FILE:...]` markers.
///
/// A matched marker is only consumed when the delivery layer can act on it: a
/// kind whose target is not attachable stays literal, so prose quoting the
/// syntax never aborts delivery and never silently disappears. The gate is
/// structural, so marker-shaped prose naming a real existing file still
/// delivers as an attachment.
///
/// `[FILE:...]` markers are the exception: they are resolved against
/// `file_roots` (see [`resolve_file_target`]) and stripped either way — a
/// delivered file leaves nothing behind, a refused one is reported in the
/// notices instead of echoing raw marker syntax at the user.
fn parse_attachment_markers(
    message: &str,
    file_roots: &[PathBuf],
) -> (String, Vec<TelegramAttachment>, Vec<String>) {
    let mut attachments: Vec<TelegramAttachment> = Vec::new();
    let mut refusals: Vec<String> = Vec::new();

    let cleaned = TELEGRAM_MEDIA_MARKER_RE
        .replace_all(message, |caps: &regex::Captures| {
            let (marker_kind, path) = parse_media_marker(caps);
            let path = path.trim();

            if marker_kind == MediaMarkerKind::File {
                match resolve_file_target(path, file_roots) {
                    Ok(canonical) => attachments.push(TelegramAttachment {
                        kind: TelegramAttachmentKind::Document,
                        target: canonical.to_string_lossy().into_owned(),
                    }),
                    Err(notice) => refusals.push(notice),
                }
                return String::new();
            }

            // Only a valid image target is attached as a photo; everything
            // else — including a data-URI, which the classifier rejects
            // cheaply (no decode, and Telegram cannot send inline data URIs)
            // — stays as literal text.
            let kind = TelegramAttachmentKind::from(marker_kind);
            if !is_attachable_target(kind, path) {
                return caps.get_match().as_str().to_string();
            }

            attachments.push(TelegramAttachment {
                kind,
                target: path.to_string(),
            });
            String::new()
        })
        .to_string();

    (cleaned.trim().to_string(), attachments, refusals)
}

/// Base URL for the Telegram Bot API.
const API_BASE: &str = "https://api.telegram.org";

/// Bot-API URL for `method` on the bot identified by `token`.
fn bot_api_url(token: &str, method: &str) -> String {
    format!("{API_BASE}/bot{token}/{method}")
}

/// Telegram Bot API maximum file download size.
const TELEGRAM_MAX_FILE_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;

/// A byte cap as the whole-MB figure the user-facing notices quote, derived from
/// the cap itself so the text cannot drift from the number enforced.
fn mb(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

/// User-facing reason when a declared attachment size exceeds the
/// product-level [`FILE_MAX_BYTES`] cap.
fn file_too_large_reason() -> String {
    format!("the file exceeds the {} MB limit", mb(FILE_MAX_BYTES))
}

/// User-facing reason when a declared attachment size exceeds Telegram's own
/// bot-download limit (which is stricter than [`FILE_MAX_BYTES`]).
fn telegram_download_limit_reason() -> String {
    format!(
        "Telegram only lets the bot download files up to {} MB",
        mb(TELEGRAM_MAX_FILE_DOWNLOAD_BYTES)
    )
}

/// User-facing reason when Telegram declared no size (or one under the limit)
/// but the `getFile`/download step still failed — in practice the download
/// limit the transport does not always report. Naming no figure keeps it
/// independent of [`TELEGRAM_MAX_FILE_DOWNLOAD_BYTES`].
const TELEGRAM_TRANSFER_REFUSED_REASON: &str =
    "Telegram refused the transfer (it may exceed what the bot can download)";

/// User-facing reason when the download ran into [`MEDIA_TRANSFER_TIMEOUT`]:
/// the request was stopped on this side, so it is not described as a refusal by
/// the transport.
const ATTACHMENT_DOWNLOAD_TIMEOUT_REASON: &str = "the download took too long and was stopped";

/// User-facing reason when the download got no answer out of Telegram at all —
/// the connection failed before a response. Like the timeout, that is this
/// side's failure and not a refusal by the transport.
const ATTACHMENT_DOWNLOAD_FAILED_REASON: &str = "the connection to Telegram failed";

/// Reported when Telegram did not return a download path for an attachment the
/// bot is otherwise allowed to fetch — an API or network failure, not a size
/// refusal.
const TELEGRAM_LOOKUP_REFUSED_REASON: &str = "Telegram could not provide the file";

/// What happened to an attachment in the agent-facing note when the transport
/// never handed it over.
const NOT_RECEIVED: &str = "not received";

/// What happened to an attachment in the agent-facing note when it arrived but
/// the daemon could not keep it, so the agent cannot be given it.
const NOT_STORED: &str = "could not be stored";

/// User-facing reason when the staging directory for a received attachment
/// could not be created locally — a mahbot-side failure, so it must not be
/// reported as a transfer refusal.
const ATTACHMENT_DIR_REFUSED_REASON: &str =
    "the local temp directory for the file could not be created";

/// User-facing reason when a fully downloaded attachment could not be written
/// to that directory — also a local failure, reported separately from the
/// transport's own refusals.
const ATTACHMENT_SAVE_REFUSED_REASON: &str = "the file could not be saved locally";

/// User-facing reason when the transport failed to deliver a local file that
/// passed the product-level size check.
///
/// Deliberately static: the transport error text embeds the request URL, which
/// carries the bot token, so it is logged and returned as an `Err` but never
/// rendered into the user's chat.
const ATTACHMENT_UPLOAD_FAILED_REASON: &str = "the upload failed";

/// The refusal reason for an attachment whose *declared* size exceeds a limit,
/// or `None` when it is within both. The product cap is checked first, so a
/// size over both reports that one.
#[must_use]
fn declared_size_refusal(size: u64) -> Option<String> {
    if size > FILE_MAX_BYTES {
        Some(file_too_large_reason())
    } else if size > TELEGRAM_MAX_FILE_DOWNLOAD_BYTES {
        Some(telegram_download_limit_reason())
    } else {
        None
    }
}

/// The user-facing reason for a failed inbound download. Only what Telegram
/// itself answered — an error status, or a body over the product cap — is a
/// refusal; a transfer that got no answer at all (a `reqwest` error in the
/// chain, whether it timed out or the connection failed) is reported as the
/// failure on this side that it is.
fn download_failure_reason(error: &anyhow::Error) -> &'static str {
    match error
        .chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
    {
        Some(e) if e.is_timeout() => ATTACHMENT_DOWNLOAD_TIMEOUT_REASON,
        Some(_) => ATTACHMENT_DOWNLOAD_FAILED_REASON,
        None => TELEGRAM_TRANSFER_REFUSED_REASON,
    }
}

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
    /// (parallel agent responses) don't trip Telegram's rate limiting. Menu
    /// refreshes are fire-and-forget and fail-open.
    menu_cache: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, ChatMenuState>>>,
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

/// Parse a Markdown link `[label](url)` starting at byte offset `i`, which must
/// point at the `[`. Returns the label, the URL, and the offset just past the
/// closing `)`. `None` when there is no http(s) link here or the label names a
/// media marker rather than prose (see [`is_marker_label`]).
fn parse_markdown_link(text: &str, i: usize) -> Option<(&str, &str, usize)> {
    let after_open = text.get(i..)?.strip_prefix('[')?;
    let bracket_end = after_open.find(']')?;
    let label = &after_open[..bracket_end];
    let after_bracket = i + bracket_end + 2;
    let after_url_open = text.get(after_bracket..)?.strip_prefix('(')?;
    let url = &after_url_open[..after_url_open.find(')')?];
    if !is_http_url(url) || is_marker_label(label) {
        return None;
    }
    Some((label, url, after_bracket + url.len() + 2))
}

/// True when a bracketed label names a media marker rather than prose: a known
/// [`MediaMarkerKind`], in any casing (`[FILE:…]`, `[image:…]`).
///
/// [`parse_attachment_markers`] keeps an unattachable marker verbatim in the
/// outgoing text, and such a marker immediately followed by a parenthesised URL
/// is delivery scaffolding: an anchor labelled with the marker's text would
/// fabricate a link out of it.
fn is_marker_label(label: &str) -> bool {
    let Some((kind, _)) = label.split_once(':') else {
        return false;
    };
    MediaMarkerKind::iter().any(|marker| marker.token().eq_ignore_ascii_case(kind))
}

/// Prefix of the anchor `render_inline` emits for a markdown link, shared with
/// the [`render_if_link`] probe so the two cannot drift.
const ANCHOR_PREFIX: &str = "<a href=\"";

/// Render `text` for Telegram HTML, or `None` when the rendering carries no
/// link, probed via [`ANCHOR_PREFIX`]. The `Some` arm means an enclosing
/// formatting span has to give way to the top-level anchor (see
/// [`markdown_to_telegram_html`]); inline code renders no anchor, so a span that
/// only wraps code keeps its formatting.
fn render_if_link(text: &str) -> Option<String> {
    let rendered = render_inline(text);
    rendered.contains(ANCHOR_PREFIX).then_some(rendered)
}

/// If the text at position `i` starts with `delim`, finds the matching closing
/// `delim`, renders the content, and wraps it in `<tag>...</tag>`.
///
/// With `linkify`, a span whose content renders a link yields to it: the span's
/// formatting is dropped and its content is rendered by [`render_inline`]
/// instead (see [`markdown_to_telegram_html`]).
///
/// On success, advances `i` past the closing delimiter and returns `true`.
/// Returns `false` if `delim` is not found or when the content between
/// delimiters is empty (the `end > 0` guard prevents zero-length formatting
/// spans like `****` or `*` with no content between delimiters).
///
/// This is a helper to deduplicate the 5 structurally identical inline
/// formatting branches (bold `**`/`__`, italic, code, strikethrough). Callers that
/// need to guard against matching a single character when the previous
/// character is the same (e.g. the second `*` of `**` for italic, or the
/// second `` ` `` of ` `` ` for inline code) must apply that guard before
/// calling this helper.
fn try_format_inline(
    text: &str,
    i: &mut usize,
    out: &mut String,
    delim: &str,
    tag: &str,
    linkify: bool,
) -> bool {
    if text[*i..].starts_with(delim) {
        let content_start = *i + delim.len();
        if let Some(end) = text[content_start..].find(delim)
            && end > 0
        {
            let inner = &text[content_start..content_start + end];
            if linkify && let Some(rendered) = render_if_link(inner) {
                out.push_str(&rendered);
            } else {
                let _ = write!(out, "<{tag}>{}</{tag}>", escape_html(inner));
            }
            *i += delim.len() * 2 + end;
            return true;
        }
    }
    false
}

/// Render one line of Markdown as Telegram HTML: bold (`**`, `__`), italic
/// (`*`), inline code (`` ` ``), strikethrough (`~~`), and links (`[text](url)`).
fn render_inline(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // Bold: **text**
        if try_format_inline(text, &mut i, &mut out, "**", "b", true) {
            continue;
        }
        // Bold: __text__
        if try_format_inline(text, &mut i, &mut out, "__", "b", true) {
            continue;
        }
        // Italic: *text* — guard against matching second `*` of `**`
        if (i == 0 || bytes[i - 1] != b'*')
            && try_format_inline(text, &mut i, &mut out, "*", "i", true)
        {
            continue;
        }
        // Inline code: `code` — guard against matching second `` ` `` of ` `` `
        // and never linkify: a URL inside code is code, not a link.
        if (i == 0 || bytes[i - 1] != b'`')
            && try_format_inline(text, &mut i, &mut out, "`", "code", false)
        {
            continue;
        }
        // Markdown link: [text](url)
        if bytes[i] == b'['
            && let Some((label, url, next)) = parse_markdown_link(text, i)
        {
            let _ = write!(
                out,
                "{ANCHOR_PREFIX}{}\">{}</a>",
                escape_html(url),
                escape_html(label)
            );
            i = next;
            continue;
        }
        // Strikethrough: ~~text~~
        if try_format_inline(text, &mut i, &mut out, "~~", "s", true) {
            continue;
        }
        // Default: escape HTML entities
        let ch = text[i..].chars().next().unwrap();
        push_escaped(ch, &mut out);
        i += ch.len_utf8();
    }
    out
}

/// Convert a subset of Markdown to Telegram's HTML parse_mode format.
/// Telegram HTML supports: &lt;b&gt;, &lt;i&gt;, &lt;u&gt;, &lt;s&gt;, &lt;code&gt;, &lt;pre&gt;, &lt;a href="..."&gt;
///
/// Supported: headers (`# …`, `## …`), bold (`**…**`, `__…__`), italic (`*…*`),
/// inline code (`` `…` ``), links (`[…](url)`), strikethrough (`~~…~~`),
/// fenced code blocks (` ``` … ``` `), and `<blockquote>` pass-through.
///
/// A formatting span (including a heading) whose content renders a link drops
/// its formatting and emits the anchor at the top level instead of nesting it:
/// the anchor is what carries clickability, not the span. Code content is never
/// linkified.
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
            let title = stripped.trim();
            // A heading is a formatting span too: it yields to a link in its
            // content, which would otherwise be delivered as literal text.
            match render_if_link(title) {
                Some(rendered) => {
                    let _ = writeln!(out, "{rendered}");
                }
                None => {
                    let _ = writeln!(out, "<b>{}</b>", escape_html(title));
                }
            }
            continue;
        }

        // ── Inline formatting per line ────────────────────────
        out.push_str(&render_inline(line));
        out.push('\n');
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

/// Classification of a Telegram edit failure (`editMessageReplyMarkup`) —
/// matched on stable substrings independent of
/// Telegram's wording; a reworded error hits `Other` (delivery unaffected,
/// no recovery).
#[derive(Debug)]
pub enum EditMessageFailure {
    /// Target message was deleted.
    NotFound,
    /// Past Telegram's 48-hour edit window.
    CannotEdit,
    /// Message is unchanged ("message is not modified") — success/no-op.
    NotModified,
    /// Any other failure (delivery unaffected, no recovery).
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
/// double-escaped.
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
    fn with_offset(
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

        // Auth is clicker-based (cq.from), never msg author; chat_id/message_id
        // identify the pressed keyboard's message so handlers can edit it in place.
        let Some((user_name, chat_id, reply_target)) = resolve_authorized_sender(cq, msg).await
        else {
            tracing::debug!(
                "Telegram: ignoring callback query from unknown user '{}'",
                extract_sender_user_name(cq)
            );
            return None;
        };

        let message_id = msg
            .get("message_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        let ctx = MessageContext {
            user_name,
            chat_id,
            message_id,
            reply_target,
            reply_reference: None,
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
            reply_reference: build_reply_reference(message),
        })
    }

    /// Prepend forwarding attribution (from `forward_from` /
    /// `forward_from_chat` / `forward_sender_name`) to content.
    fn prepend_forward_attribution(content: String, message: &serde_json::Value) -> String {
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
        bot_api_url(&self.bot_token, method)
    }

    /// Signal this specific channel's listener to stop, without affecting
    /// the global shutdown token or other channels.
    fn cancel_own(&self) {
        self.cancel.cancel();
    }

    /// Register the bot's global (unscoped) commands via Telegram's
    /// `setMyCommands` API. `/clear` is global — per-user commands are
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

            let url = bot_api_url(&bot_token, "setMyCommands");
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
        let url = bot_api_url(token, "getMe");
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
    ///
    /// Bounded by [`FILE_MAX_BYTES`] on both sides: a declared `Content-Length`
    /// over the cap is refused before the body is buffered, and the received
    /// body is checked again because the transport does not always declare one.
    /// The request carries [`MEDIA_TRANSFER_TIMEOUT`]: the cap itself takes
    /// longer than the client's one-minute default to pull over a slow link.
    async fn download_file(&self, file_path: &str) -> anyhow::Result<Vec<u8>> {
        let url = format!("{API_BASE}/file/bot{}/{file_path}", self.bot_token);
        let resp = self
            .http_client()
            .get(&url)
            .timeout(MEDIA_TRANSFER_TIMEOUT)
            .send()
            .await
            .context("Failed to download Telegram file")?;

        if !resp.status().is_success() {
            anyhow::bail!("Telegram file download failed: {}", resp.status());
        }

        anyhow::ensure!(
            resp.content_length()
                .is_none_or(|len| len <= FILE_MAX_BYTES),
            "Telegram file is larger than the {} MB product cap",
            mb(FILE_MAX_BYTES)
        );
        let bytes = resp.bytes().await?;
        anyhow::ensure!(
            bytes.len() as u64 <= FILE_MAX_BYTES,
            "Telegram file is larger than the {} MB product cap",
            mb(FILE_MAX_BYTES)
        );
        Ok(bytes.to_vec())
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
    /// Downloads the file into this message's own staging directory in the
    /// Telegram temp root and returns a `ChannelMessage` recording it (see
    /// [`ChannelMessage::attachment_dirs`]). Returns `None` only when the
    /// message is not an attachment or the sender is unauthorized: an attachment
    /// the bot cannot retrieve or store still yields a message, so the caption
    /// reaches the agent with the failure called out in the content.
    async fn try_parse_attachment_message(
        &self,
        update: &serde_json::Value,
    ) -> Option<ChannelMessage> {
        let message = update.get("message")?;

        // Authorization comes first: an unauthorized sender is ignored
        // entirely and silently, before any attachment work happens.
        let ctx = self.extract_message_context(message).await?;

        let attachment = Self::parse_attachment_metadata(message)?;

        // Declared-size gate. Telegram reports the size on most attachments,
        // so reject locally before spending a download; the two limits differ
        // and the user must be told which one was hit.
        if let Some(size) = attachment.file_size
            && let Some(reason) = declared_size_refusal(size)
        {
            tracing::info!("Rejecting attachment: declared size {size} bytes ({reason})");
            return Some(
                self.reject_attachment(ctx, &attachment, None, NOT_RECEIVED, &reason)
                    .await,
            );
        }

        // Download file from Telegram. The transport does not always report a
        // size, so a transfer failure is the other way the 20 MB limit shows
        // up — surface it instead of dropping the update. A failure on this side
        // (a timeout, or a connection that broke) is reported as that, not as a
        // refusal by the transport.
        let tg_file_path = match self.get_file_path(&attachment.file_id).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Failed to get attachment file path: {e}");
                return Some(
                    self.reject_attachment(
                        ctx,
                        &attachment,
                        None,
                        NOT_RECEIVED,
                        TELEGRAM_LOOKUP_REFUSED_REASON,
                    )
                    .await,
                );
            }
        };
        let remote_ext = Path::new(&tg_file_path)
            .extension()
            .and_then(|e| e.to_str());

        let file_data = match self.download_file(&tg_file_path).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to download attachment: {e}");
                return Some(
                    self.reject_attachment(
                        ctx,
                        &attachment,
                        remote_ext,
                        NOT_RECEIVED,
                        download_failure_reason(&e),
                    )
                    .await,
                );
            }
        };

        // The directory is created only once the bytes are here so a refused
        // transfer leaves nothing behind.
        let staging_dir = crate::util::telegram_staging_dir_name(&ctx.chat_id, ctx.message_id);
        let save_dir = crate::util::telegram_files_root().join(&staging_dir);
        if let Err(e) = tokio::fs::create_dir_all(&save_dir).await {
            tracing::warn!("Failed to create telegram attachment directory: {e}");
            return Some(
                self.reject_attachment(
                    ctx,
                    &attachment,
                    remote_ext,
                    NOT_STORED,
                    ATTACHMENT_DIR_REFUSED_REASON,
                )
                .await,
            );
        }

        let local_filename =
            local_attachment_name(&attachment, &ctx.chat_id, ctx.message_id, remote_ext);
        let local_path = save_dir.join(&local_filename);
        if let Err(e) = tokio::fs::write(&local_path, &file_data).await {
            tracing::warn!("Failed to save attachment to {}: {e}", local_path.display());
            // A partial write must not linger: nothing downstream will ever
            // reference this per-message directory.
            let _ = tokio::fs::remove_dir_all(&save_dir).await;
            return Some(
                self.reject_attachment(
                    ctx,
                    &attachment,
                    remote_ext,
                    NOT_STORED,
                    ATTACHMENT_SAVE_REFUSED_REASON,
                )
                .await,
            );
        }

        let mut content = format_attachment_content(
            attachment.kind,
            &local_path,
            attachment.mime_type.as_deref(),
        );
        if let Some(caption) = &attachment.caption
            && !caption.is_empty()
        {
            let _ = write!(content, "\n\n{caption}");
        }

        let content = Self::prepend_forward_attribution(content, message);

        Some(ctx.into_attachment_message(content, staging_dir))
    }

    /// Tell the user an attachment could not be taken and return the
    /// caption-preserving channel message, so an oversized or refused file still
    /// produces a user-visible reply and an agent turn. `outcome` is
    /// [`NOT_RECEIVED`] or [`NOT_STORED`] — it decides whether the note may say
    /// the file never arrived. The name in the note is
    /// [`local_attachment_name`]'s; the direct Telegram notice is best-effort (a
    /// send failure is only logged).
    async fn reject_attachment(
        &self,
        ctx: MessageContext,
        attachment: &IncomingAttachment,
        remote_ext: Option<&str>,
        outcome: &str,
        reason: &str,
    ) -> ChannelMessage {
        let display_name =
            local_attachment_name(attachment, &ctx.chat_id, ctx.message_id, remote_ext);
        let notice = format!("Could not process the file \"{display_name}\": {reason}.");
        let (_, thread_id) = parse_recipient(&ctx.reply_target);
        if let Err(e) = self
            .send_text_chunks(&notice, &ctx.chat_id, thread_id, None)
            .await
        {
            tracing::warn!("Failed to send attachment-failure notice: {e}");
        }

        let content = attachment_rejection_content(
            &display_name,
            outcome,
            reason,
            attachment.caption.as_deref(),
        );
        ctx.into_channel_message(content, None)
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

        let content = Self::prepend_forward_attribution(content, message);

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
    /// — the message was still delivered), or the HTTP status and response
    /// body on failure. Parsing the body for the id couples this method to the
    /// shared send path's response shape; the `None` case preserves status-only
    /// semantics for callers that don't need the id.
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
        // Telegram attaches a web-page preview to the first URL unless the send
        // disables it: Bot API 7.0 replaced `disable_web_page_preview` with
        // `link_preview_options`, whose object form requires `is_disabled`.
        body["link_preview_options"] = serde_json::json!({ "is_disabled": true });

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

    /// Replace a previously sent message's inline keyboard in place
    /// (`editMessageReplyMarkup`). `reply_markup` must be a full
    /// `{"inline_keyboard": [...]}` value (or `Value::Null` to remove the
    /// keyboard). Failures are classified with the shared edit-failure
    /// classifier; callers typically treat every failure as a no-op.
    pub async fn edit_reply_markup(
        &self,
        chat_id: &str,
        message_id: i64,
        reply_markup: &serde_json::Value,
    ) -> Result<(), EditMessageFailure> {
        self.post_telegram_json(
            "editMessageReplyMarkup",
            serde_json::json!({
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup,
            }),
            "editMessageReplyMarkup error",
        )
        .await
        .map(|_| ())
        .map_err(|(_, desc)| classify_edit_failure(&desc))
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
        if !path.is_file() {
            anyhow::bail!("Telegram attachment target is not an existing regular file: {target}");
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

    /// Send a media file (image-as-file/document/video/audio) to a Telegram
    /// chat.
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
            .multipart(form)
            .timeout(MEDIA_TRANSFER_TIMEOUT);

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
            let Some(merged) = merge_album_members(group_messages) else {
                continue;
            };
            if tx.send(merged).await.is_err() {
                return false;
            }
        }

        true
    }
}

/// Merge a Telegram album (media group) into the single message the pipeline
/// processes: members arrive as separate updates, so their contents are
/// concatenated and every member's inbound staging directory is carried over
/// (see [`ChannelMessage::attachment_dirs`]); addressing fields come from the
/// first member. `None` for an empty group (an internal invariant failure that
/// must not abort the inbound listener).
fn merge_album_members(members: Vec<ChannelMessage>) -> Option<ChannelMessage> {
    let mut members = members.into_iter();
    let mut acc = members.next()?;
    for next in members {
        acc.content.push('\n');
        acc.content.push_str(&next.content);
        acc.attachment_dirs.extend(next.attachment_dirs);
    }
    Some(acc)
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
        // parked while a local image is decoded for attachment. `[FILE:...]`
        // markers are resolved (and possibly refused) here too.
        let (text_without_markers, attachments, refusals) =
            crate::util::with_block_in_place(|| {
                parse_attachment_markers(content, &message.file_roots)
            });

        if attachments.is_empty() && refusals.is_empty() {
            return self
                .send_text_chunks(content, chat_id, thread_id, message.reply_markup.clone())
                .await;
        }

        // Refusal notices ride in the same body as the model's remaining text:
        // a separate best-effort send would lose the file, the marker AND the
        // explanation on a transient failure, while the body send's `?` makes
        // it a real delivery error.
        let body = if refusals.is_empty() {
            text_without_markers
        } else {
            let notices = refusals.join("\n\n");
            if text_without_markers.is_empty() {
                notices
            } else {
                format!("{text_without_markers}\n\n{notices}")
            }
        };

        if !body.is_empty() {
            self.send_text_chunks(&body, chat_id, thread_id, message.reply_markup.clone())
                .await?;
        }

        // One failed attachment must not cost the user the others; the first
        // error is returned after the loop so delivery-failure logging still
        // fires, and every failure is surfaced in-chat.
        let mut first_error: Option<anyhow::Error> = None;
        for attachment in &attachments {
            if let Err(e) = self.send_attachment(chat_id, thread_id, attachment).await {
                tracing::warn!(
                    error = %e,
                    file = %attachment.target,
                    "Telegram: attachment delivery failed"
                );
                let notice = file_refusal(
                    file_name_or_path(&attachment.target),
                    ATTACHMENT_UPLOAD_FAILED_REASON,
                );
                if let Err(notice_err) = self
                    .send_text_chunks(&notice, chat_id, thread_id, None)
                    .await
                {
                    tracing::warn!(
                        error = %notice_err,
                        "Telegram: failed to report attachment failure to the user"
                    );
                }
                first_error.get_or_insert(e);
            }
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
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
        file_roots: Vec::new(),
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
/// Media markers of every kind the pattern matches (`[IMAGE:...]`,
/// `[AUDIO:...]`, `[VIDEO:...]`, `[FILE:...]`) are stripped so raw marker
/// syntax does not appear in the quote; a marker-shaped word of any other kind
/// is not a marker and stays literal. Purely media-only messages are skipped
/// entirely.
/// When the message carries a
/// [`ReplyReference`], a `↩ {author}: {snippet}` header line leads the
/// blockquote (before the user's text), and a media-only message with a
/// reference still mirrors — the blockquote then holds only the `↩` line.
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

    // Strip media markers so users don't see raw `[IMAGE:...]` syntax in the
    // quote. Every kind the pattern matches is stripped; a marker-shaped word of
    // an unknown kind is not a marker at all and survives rather than being
    // deleted. A reply reference is preserved as a `↩` header line inside the
    // blockquote; when a reference is present the media-only guard below is
    // relaxed so a reply to a media message still mirrors the `↩` line.
    let content = TELEGRAM_MEDIA_MARKER_RE
        .replace_all(trimmed, "")
        .to_string();
    let content = content.trim().to_string();

    let quoted = if let Some(reply) = &msg.reply_reference {
        let header = format!("↩ {}: {}", reply.author, reply.snippet);
        if content.is_empty() {
            // Media-only message with a reply — emit just the `↩` line.
            format!("<blockquote>\n{header}\n</blockquote>")
        } else {
            format!("<blockquote>\n{header}\n{content}\n</blockquote>")
        }
    } else {
        if content.is_empty() {
            return; // Media-only message — nothing to quote.
        }
        // Wrap in <blockquote> — these tags pass through markdown_to_telegram_html
        // unchanged, while the user's text retains markdown formatting.
        format!("<blockquote>\n{content}\n</blockquote>")
    };

    for binding in &telegram_bindings {
        let Some(reply_target) = &binding.reply_target else {
            continue; // skip bindings without a reply target
        };
        let reply = SendMessage {
            content: quoted.clone(),
            recipient: reply_target.clone(),
            reply_markup: None,
            // The quote is already marker-free (every mapped kind was stripped
            // above), so an empty root list cannot turn the user's own text into
            // a "Could not send ..." refusal; it only states that the mirror
            // delivers no files.
            file_roots: Vec::new(),
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
/// derived from their admin status and the state of their selected shared
/// workspace. Shared by the per-chat `setMyCommands` refresh and the
/// `/start` welcome message.
///
/// State-aware entries: exactly one of `/pause` or `/unpause` appears (the
/// one matching the workspace's paused state), and exactly one of
/// `/maintenance_on` or `/maintenance_off`. Admins without a selected shared
/// workspace get neither pair (there is no workspace state to reflect).
#[must_use]
pub async fn user_command_entries(user_name: &str) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::new();

    if crate::users::is_admin(user_name).await {
        entries.push(("board".to_string(), BOARD_COMMAND_DESC.to_string()));
        entries.push(("archive".to_string(), ARCHIVE_COMMAND_DESC.to_string()));

        // `/update` is global (any full-permission admin) and shown only when
        // the shared availability cache confirms an update — always in
        // local-checkout mode, registry mode only when a strictly newer stable
        // version exists. The menu reflects the cached state, never a network
        // call per refresh.
        if crate::self_update::should_show_update(crate::self_update::update_availability()) {
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

    // Media model-selection commands are available to every user (the media
    // models are stored per-user, and every user has the Assistant role).
    entries.push((
        "image_models".to_string(),
        IMAGE_MODELS_COMMAND_DESC.to_string(),
    ));
    entries.push((
        "video_models".to_string(),
        VIDEO_MODELS_COMMAND_DESC.to_string(),
    ));

    // `/clear` is anchored at the very bottom of the menu. The `/agents`
    // role-switch entry is gone: the pool is the constant single Assistant,
    // so there is nothing to switch.
    entries.push(("clear".to_string(), CLEAR_COMMAND_DESC.to_string()));
    entries
}

#[cfg(test)]
#[path = "telegram_tests.rs"]
mod tests;
