//! The Telegram transport: Bot API long polling, update decoding, send and
//! edit with the Bot API's own HTML subset, media upload and download, and the
//! forward-only intake ledger that stops an update being handled twice.
//!
//! Not the shared message plane (`crate::channels`) and not the bot command
//! handlers — this file turns updates into intake claims and `ChannelMessage`
//! values, and turns replies into Bot API calls.

use crate::channels::ReplyReference;
use crate::channels::reply::normalize_reply_text;
use crate::channels::telegram_intake::{Intake, TelegramIntakeStore, listener_claim};
use crate::pipeline::board::{Ticket, TicketPhase};
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
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
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
const BOARD_COMMAND_DESC: &str = "List tickets from all workspaces";
/// Description for the `/archive` command (admin).
const ARCHIVE_COMMAND_DESC: &str = "Archive done & cancelled tickets (all workspaces)";
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
pub(crate) fn decode_action(content: &str) -> Option<(String, String)> {
    let rest = content.strip_prefix(ACTION_PREFIX)?;
    match rest.split_once('|') {
        Some((action, payload)) => Some((action.to_string(), payload.to_string())),
        None => Some((rest.to_string(), String::new())),
    }
}

/// The control-plane input a message carries: something the dispatch loop handles
/// itself, without the assistant. `None` for a message the agent path handles —
/// including a button press whose payload is neither kind, which has always gone
/// to the agent.
#[derive(Debug)]
pub enum ControlInput {
    /// An `__act__` action payload, as `(action, payload)`.
    Action((String, String)),
    /// A bot command.
    Command(crate::BotCommand),
}

/// Which control-plane input an inbound message carries, or `None` when the agent
/// path handles it. A bot command counts only on Telegram: other channels route
/// `/`-prefixed text as ordinary text.
#[must_use]
pub fn control_input(msg: &ChannelMessage) -> Option<ControlInput> {
    if let Some(action) = decode_action(&msg.content) {
        return Some(ControlInput::Action(action));
    }
    if msg.channel == "telegram"
        && let Some(cmd) = crate::parse_bot_command(&msg.content)
    {
        return Some(ControlInput::Command(cmd));
    }
    None
}

/// True for a message the burst collector must never hold: one carrying control
/// input ([`control_input`]), or a button press. This is a superset of
/// [`control_input`] — the dispatch loop lets a press whose payload is neither
/// kind fall through to the agent, but a burst must not hold it either.
#[must_use]
pub(crate) fn is_control_message(msg: &ChannelMessage) -> bool {
    msg.callback_query_id.is_some() || control_input(msg).is_some()
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

/// The kind of incoming attachment (document, photo, video, audio, or recorded
/// voice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncomingAttachmentKind {
    Document,
    Photo,
    Video,
    /// A sound file (the `audio` key). On macOS it is transcribed like a
    /// recorded voice note; with no local audio subsystem elsewhere it takes
    /// the ordinary file path instead.
    Audio,
    /// A recorded voice message (the `voice` key). Telegram carries no file
    /// name for it — OGG/Opus bytes only.
    Voice,
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

/// A Telegram sender that is a person, as Telegram presents the event.
struct TelegramSender {
    /// The numeric id Telegram reports, as digits.
    numeric_id: Option<String>,
    /// The nickname, when the sender has one.
    nickname: Option<String>,
}

/// The person a Telegram event came from, or `None` when the sender is not a
/// person: a message sent on behalf of a chat (`sender_chat` — a channel post,
/// the anonymous group administrator, an auto-forwarded channel post), a bot
/// (Telegram's own bots included), or one of Telegram's own service identities.
/// Decided by the sender's identity as Telegram presents it, never by the
/// spelling of a nickname, so no nickname list can outrun it.
fn telegram_sender_person(source: &serde_json::Value) -> Option<TelegramSender> {
    if source.get("sender_chat").is_some() {
        return None;
    }
    let from = source.get("from")?;
    if from.get("is_bot").and_then(serde_json::Value::as_bool) == Some(true) {
        return None;
    }
    let numeric_id = from
        .get("id")
        .and_then(serde_json::Value::as_i64)
        .map(|id| id.to_string());
    if let Some(id) = &numeric_id
        && crate::users::is_telegram_service_id(id)
    {
        return None;
    }
    // The reserved sentinel means "no nickname": a sender whose reported
    // username is that literal word is folded to nickname-less here, and the
    // store's row scan skips a stored row holding it — either alone keeps a
    // legacy "unknown" binding inert.
    let nickname = from
        .get("username")
        .and_then(serde_json::Value::as_str)
        .filter(|nickname| *nickname != crate::users::TELEGRAM_UNKNOWN_SENTINEL)
        .map(String::from);
    Some(TelegramSender {
        numeric_id,
        nickname,
    })
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
    /// The `[Forwarded from …]` prefix this message's content opens with, when
    /// it arrived as forwarded content — rendered once, here, so a path that
    /// hands the message on in another shape (a refusal notice included)
    /// cannot lose it.
    forward_prefix: Option<String>,
    /// Whether the message came in as forwarded content — the fact the burst
    /// collector's longer window rests on, kept apart from the rendered
    /// `forward_prefix` so a forward whose origin this product has no label for
    /// is still forwarded.
    forwarded: bool,
    /// Telegram's own time for the message (see `ChannelMessage::update_date`).
    update_date: Option<i64>,
}

impl MessageContext {
    fn into_channel_message(self, content: String, cq_id: Option<String>) -> ChannelMessage {
        let content = match self.forward_prefix {
            Some(prefix) => format!("{prefix}{content}"),
            None => content,
        };
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
            parts: Vec::new(),
            update_date: self.update_date,
            forwarded: self.forwarded,
            update_ids: Vec::new(),
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
/// MIME/extension) use `[VIDEO:/path]`; audio messages use `[AUDIO:/path]` on
/// macOS and `[FILE:/path]` (the ordinary attachment path) where no local
/// audio subsystem exists; a recorded voice message is only ever
/// transcribed on macOS, so elsewhere it is refused before it reaches this
/// function. Everything else uses `[FILE:/path]` — after one last check
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
        IncomingAttachmentKind::Audio | IncomingAttachmentKind::Voice => {
            // macOS transcribes this through the AUDIO marker enrichment
            // consumes; elsewhere there is no local audio, so a sound file is
            // handed over as an ordinary attachment (a recorded voice message,
            // which has no file name, is refused before it gets here).
            let kind = if cfg!(target_os = "macos") {
                "AUDIO"
            } else {
                "FILE"
            };
            format!("[{kind}:{}]", local_path.display())
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
        // A recorded voice message arrives with no `file_name`, so it gets the
        // generated `audio_<chat>_<message>.<ext>` name — with the generic
        // fallback extension above, not a recording-specific one.
        IncomingAttachmentKind::Audio | IncomingAttachmentKind::Voice => "audio",
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
/// first, then the `[File <name>: <outcome> — <reason>]` note. `display_name`
/// is `None` for an attachment with no usable name to show (a recorded voice
/// message has none, and [`local_attachment_name`] would invent one with the
/// wrong extension), which renders the name-less `[File: <outcome> — <reason>]`
/// instead.
#[must_use]
fn attachment_rejection_content(
    display_name: Option<&str>,
    outcome: &str,
    reason: &str,
    caption: Option<&str>,
) -> String {
    let note = match display_name {
        Some(name) => format!("[File {name}: {outcome} — {reason}]"),
        None => format!("[File: {outcome} — {reason}]"),
    };
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

    // The plain spelling, not the platform's verbatim form: this value becomes
    // the attachment's `target`, which reaches the delivery log.
    Ok(crate::util::strip_verbatim_prefix(&canonical))
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

/// Bot-API URL for `method` on the bot identified by `token`, at `api_base`.
fn bot_api_url(api_base: &str, token: &str, method: &str) -> String {
    format!("{api_base}/bot{token}/{method}")
}

/// The `getMe` answer for `token` at `api_base`, once the platform has confirmed
/// the token is one of its own.
///
/// Validating a token and naming the account it belongs to are the same call and
/// the same answer; the caller is what differs — the intake keys its record on the
/// account's own identity rather than on the token's text, so a rotated token for
/// the same account still recovers.
async fn get_me(
    client: &reqwest::Client,
    api_base: &str,
    token: &str,
) -> anyhow::Result<serde_json::Value> {
    let response = client
        .get(bot_api_url(api_base, token, "getMe"))
        .send()
        .await
        .context("Failed to reach Telegram API")?;
    let status = response.status();
    let data: serde_json::Value = match response.json().await {
        Ok(data) => data,
        Err(err) => anyhow::bail!("Failed to parse Telegram API response: {err}"),
    };
    if !status.is_success() || data["ok"].as_bool() != Some(true) {
        let desc = data["description"].as_str().unwrap_or("unknown error");
        anyhow::bail!("Invalid Telegram bot token: {desc}");
    }
    Ok(data)
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

/// User-facing reason when a recorded voice message arrives off macOS, where
/// no local audio subsystem exists to transcribe it: it is refused rather than
/// silently handed on useless.
#[cfg(not(target_os = "macos"))]
const VOICE_UNSUPPORTED_REASON: &str = "voice messages are not supported on this platform";

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

    /// The Bot API base this listener calls — [`API_BASE`] in production, and an
    /// endpoint of its own when a check drives the intake.
    api_base: String,
    /// The intake's durable store, resolved the first time a listener runs (the
    /// stores open after the channel is built).
    intake_store: std::sync::OnceLock<TelegramIntakeStore>,
    /// The claim this listener's rows are recorded under. Every listener mints its
    /// own, so the listener that replaces this one is a different run and takes over
    /// what this one left held instead of leaving it for the rest of the process's
    /// life.
    intake_claim: std::sync::Arc<str>,
    /// Whether a listener frame of this channel is running, and the signal for when it
    /// stops — what a replacement waits on ([`TelegramChannel::wait_stopped`]). Several
    /// waiters can be parked on one stop: the teardown of a cleared token, and a hand-over
    /// that lands while that teardown waits.
    listener_running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    listener_stopped: std::sync::Arc<tokio::sync::Notify>,

    /// The messages this listener is preparing right now — taken up and not yet
    /// handed to the burst collector, which holds that sender's group for that
    /// chat while a mark is on it (`telegram_group::Preparations`).
    preparations: std::sync::Arc<crate::channels::telegram_group::Preparations>,

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

/// Authorize a Telegram event by the identity of the person it came from, and
/// derive the reply context it arrived in.
///
/// `sender_source` is the object Telegram presents the sender in — the message
/// itself for a normal message (so `sender_chat` is read there), the
/// `callback_query` for a button press, so a keyboard attached to a
/// chat-attributed message still authorizes the person who pressed it.
/// `chat_source` is the object carrying the `chat` the event arrived in; it
/// only supplies the delivery address, never the identity.
///
/// Returns `Ok(None)` when the sender is not a person ([`telegram_sender_person`]),
/// matches no binding, or the chat context is missing; otherwise a 3-tuple
/// `(canonical_user, chat_id, reply_target)` where `canonical_user` is the
/// account the binding resolved to, `chat_id` the raw chat ID (e.g. `"123456"`)
/// and `reply_target` the reply target string (e.g. `"123456"`, or
/// `"123456:789"` for threads). The contact is updated only on success, and only
/// on the binding that actually matched.
///
/// The error is the read of the user bindings failing, which is not a decision:
/// it is what keeps a caller from turning "cannot decide right now" into a
/// refusal and dropping the event for good.
async fn resolve_authorized_sender(
    sender_source: &serde_json::Value,
    chat_source: &serde_json::Value,
) -> anyhow::Result<Option<(String, String, String)>> {
    // Authorization keys on the identity of the person who sent the event — a
    // number first, since it survives a rename or a dropped nickname, then the
    // nickname. The conversation the event arrived in only supplies the
    // delivery address.
    let Some(sender) = telegram_sender_person(sender_source) else {
        return Ok(None);
    };
    let Some(matched) = crate::users::store()
        .resolve_telegram_user(sender.numeric_id.as_deref(), sender.nickname.as_deref())
        .await?
    else {
        return Ok(None);
    };
    let Some((chat_id, reply_target)) = extract_chat_context(chat_source) else {
        return Ok(None);
    };
    // Update reply_target for future message delivery
    let _ =
        crate::users::update_channel_contact("telegram", &matched.identifier, &reply_target).await;
    Ok(Some((matched.user_name, chat_id, reply_target)))
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

/// Prefix of the `[Forwarded from …]` attribution a forwarded message's content
/// opens with. The fact behind it, which the burst collector's longer window
/// rests on, is a message's `forwarded` flag — never this text.
const FORWARD_ATTRIBUTION_PREFIX: &str = "[Forwarded from ";

/// What `TelegramChannel::forward` makes of a message's forward fields: whether it
/// is forwarded content at all, and the attribution to render for it.
struct Forward {
    /// Whether any forward field of the message is set. The flag is the fact and
    /// the prefix only its rendering: an origin the product has no name for is
    /// still forwarded.
    forwarded: bool,
    /// The whole `[Forwarded from …] ` prefix, when one can be rendered.
    prefix: Option<String>,
}

/// The label one value of a legacy forward field renders as.
type ForwardLabel = fn(&serde_json::Value) -> Option<String>;

/// The older fields Telegram attributes a forwarded message in, in the order the
/// attribution reads them: each field, and the label a value of it renders as.
/// The flag and the label both come from this one list, because a field can be
/// present without the product being able to name what it points at — such a
/// message is still forwarded.
const LEGACY_FORWARD_FIELDS: [(&str, ForwardLabel); 3] = [
    // A channel or a group: its title, with the historical "channel: " shape
    // (these fields carried no reliable kind).
    ("forward_from_chat", |chat| {
        Some(format!(
            "channel: {}",
            chat.get("title")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown channel")
        ))
    }),
    // A user whose identity the privacy settings allow.
    ("forward_from", |user| Some(format_sender_label(user))),
    // A user who hides their identity.
    ("forward_sender_name", |name| {
        name.as_str().map(String::from)
    }),
];

impl TelegramChannel {
    /// Internal constructor every other constructor builds on.
    #[must_use]
    fn new_with(bot_token: String, api_base: String) -> Self {
        Self {
            bot_token,
            api_base,
            http_client: crate::util::http::build_http_client(Duration::from_mins(1)),
            cancel: std::sync::Arc::new(tokio_util::sync::CancellationToken::new()),
            intake_store: std::sync::OnceLock::new(),
            intake_claim: std::sync::Arc::from(listener_claim()),
            listener_running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            listener_stopped: std::sync::Arc::new(tokio::sync::Notify::new()),
            preparations: std::sync::Arc::new(crate::channels::telegram_group::Preparations::new()),
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
        Self::new_with(bot_token, API_BASE.to_string())
    }

    /// The intake's store for this process.
    fn intake_store(&self) -> TelegramIntakeStore {
        self.intake_store
            .get_or_init(|| crate::channels::telegram_intake::store().clone())
            .clone()
    }

    /// A channel pointed at an endpoint of its own and recording into `store`
    /// instead of the process's database — how the intake's checks drive a listener
    /// without the running service. The claim is the channel's own, exactly as it is
    /// for a listener the product builds.
    #[cfg(test)]
    #[must_use]
    fn for_test(bot_token: String, api_base: String, store: TelegramIntakeStore) -> Self {
        Self {
            intake_store: std::sync::OnceLock::from(store),
            ..Self::new_with(bot_token, api_base)
        }
    }

    /// Whether a listener frame of this channel is running right now.
    fn running(&self) -> bool {
        self.listener_running
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Wait, briefly, for the listener frame of this channel to end — the state a
    /// replacement asks for before it takes over what the frame left behind
    /// ([`stop_listeners`]), because a frame whose collector has drained means
    /// everything it held is settled or left held in the record. The one thing no
    /// frame covers is the process itself ending, a crash ([`ListenerFrame`]).
    ///
    /// Bounded, so a frame that will not end cannot hold a replacement up for good.
    /// Past the bound the replacement goes ahead and may take over events the
    /// still-open frame is about to hand on, which risks the same event reaching the
    /// pipeline twice — the side to err on, and the reason the bound is generous.
    async fn wait_stopped(&self) {
        // Enrolled before the check below rather than left to the first poll:
        // `Notify::notify_waiters` wakes only waiters that are already registered, so a
        // stop landing in between would be missed and the wait would sit out the whole
        // bound.
        let mut stopped = std::pin::pin!(self.listener_stopped.notified());
        stopped.as_mut().enable();
        if !self.running() {
            return;
        }
        if tokio::time::timeout(LISTENER_STOP_WAIT, stopped)
            .await
            .is_err()
        {
            tracing::warn!("Telegram listener: the replaced listener did not stop in time");
        }
    }

    /// Wait out a pause in the intake's work, ending it early when this listener is
    /// being stopped: a failing store or a contended one must not hold a token change or
    /// a wind-down up for the whole pause.
    async fn intake_pause(&self, duration: Duration) {
        tokio::select! {
            () = self.cancel.cancelled() => {}
            _ = crate::shutdown::sleep_or_shutdown_or_drain(duration) => {}
        }
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
    async fn parse_callback_query(
        &self,
        cq: &serde_json::Value,
    ) -> anyhow::Result<Option<ChannelMessage>> {
        let Some(data) = cq.get("data").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let Some(msg) = cq.get("message") else {
            return Ok(None);
        };
        let callback_query_id = cq
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        // Auth is clicker-based (cq.from), never msg author; chat_id/message_id
        // identify the pressed keyboard's message so handlers can edit it in place.
        let Some((user_name, chat_id, reply_target)) = resolve_authorized_sender(cq, msg).await?
        else {
            tracing::debug!(
                "Telegram: ignoring callback query from unknown user '{}'",
                extract_sender_user_name(cq)
            );
            return Ok(None);
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
            forward_prefix: None,
            forwarded: false,
            update_date: None,
        };
        Ok(Some(
            ctx.into_channel_message(data.to_string(), callback_query_id),
        ))
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
    /// reply target. `Ok(None)` if the sender is not allowed; the error is the
    /// binding read failing, which is not a decision and must not be turned into
    /// a refusal by the caller — it is logged there, where the event it belongs to
    /// is known.
    async fn extract_message_context(
        &self,
        message: &serde_json::Value,
    ) -> anyhow::Result<Option<MessageContext>> {
        let Some((user_name, chat_id, reply_target)) =
            resolve_authorized_sender(message, message).await?
        else {
            tracing::debug!(
                "Telegram: ignoring message from unknown user '{}'",
                extract_sender_user_name(message)
            );
            return Ok(None);
        };

        let message_id = message
            .get("message_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        let forward = Self::forward(message);
        Ok(Some(MessageContext {
            user_name,
            chat_id,
            message_id,
            reply_target,
            reply_reference: build_reply_reference(message),
            forward_prefix: forward.prefix,
            forwarded: forward.forwarded,
            update_date: message.get("date").and_then(serde_json::Value::as_i64),
        }))
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
        bot_api_url(&self.api_base, &self.bot_token, method)
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
    /// admin state, and the scoped `setMyCommands` registration when
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
        let api_base = self.api_base.clone();
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

            let url = bot_api_url(&api_base, &bot_token, "setMyCommands");
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
        let client = crate::util::http::build_http_client(std::time::Duration::from_secs(10));
        get_me(&client, API_BASE, token).await.map(|_| ())
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
        let url = format!("{}/file/bot{}/{file_path}", self.api_base, self.bot_token);
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
    /// `audio` maps to [`IncomingAttachmentKind::Audio`] and `voice` to
    /// [`IncomingAttachmentKind::Voice`]; the two differ in what the receive
    /// path can do with them off macOS.  Returns `None` for text‑only
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
            ("voice", IncomingAttachmentKind::Voice),
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
    /// [`ChannelMessage::attachment_dirs`]). `Ok(None)` only when the message is
    /// not an attachment or the sender is unauthorized — the error is the binding
    /// read failing, which the caller keeps the event held for. An attachment the
    /// bot cannot retrieve or store still yields a message, so the caption reaches
    /// the agent with the failure called out in the content.
    ///
    /// Off macOS a recorded voice message is refused (see
    /// `reject_voice_message`) before any fetch — nothing there can transcribe
    /// it, and the note names no file. An `audio` sound file is not refused: it
    /// takes the ordinary path below, so the agent gets the file itself.
    ///
    /// The whole preparation — the fetch included — is marked in
    /// [`TelegramChannel::preparations`], so the collector holds that sender's
    /// group for that chat until the part it produces has been handed over. The
    /// `[Forwarded from …]` attribution (and the `forwarded` flag) is applied by
    /// [`MessageContext::into_channel_message`], so it survives every refusal
    /// path below as well as the success one.
    #[expect(clippy::too_many_lines)] // one linear fetch-and-refuse run, each gate a distinct outcome
    async fn try_parse_attachment_message(
        &self,
        update: &serde_json::Value,
    ) -> anyhow::Result<Option<ChannelMessage>> {
        let Some(message) = update.get("message") else {
            return Ok(None);
        };

        // Authorization comes first: an unauthorized sender is ignored
        // entirely and silently, before any attachment work happens.
        let Some(ctx) = self.extract_message_context(message).await? else {
            return Ok(None);
        };

        let Some(attachment) = Self::parse_attachment_metadata(message) else {
            return Ok(None);
        };

        // Held for the whole preparation below — the fetch included — so the group
        // this sender has open in this chat waits for the part this call produces
        // (see `Preparations`).
        let _preparing = self.preparations.begin(&ctx.user_name, &ctx.reply_target);

        // No local audio here: a recorded voice message cannot be transcribed,
        // so refuse it before anything is fetched. A sound file (`audio`) is
        // not affected — it takes the ordinary attachment path below.
        #[cfg(not(target_os = "macos"))]
        if attachment.kind == IncomingAttachmentKind::Voice {
            return Ok(Some(
                self.reject_voice_message(ctx, attachment.caption.as_deref())
                    .await,
            ));
        }

        // Declared-size gate. Telegram reports the size on most attachments,
        // so reject locally before spending a download; the two limits differ
        // and the user must be told which one was hit.
        if let Some(size) = attachment.file_size
            && let Some(reason) = declared_size_refusal(size)
        {
            tracing::info!("Rejecting attachment: declared size {size} bytes ({reason})");
            return Ok(Some(
                self.reject_attachment(ctx, &attachment, None, NOT_RECEIVED, &reason)
                    .await,
            ));
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
                return Ok(Some(
                    self.reject_attachment(
                        ctx,
                        &attachment,
                        None,
                        NOT_RECEIVED,
                        TELEGRAM_LOOKUP_REFUSED_REASON,
                    )
                    .await,
                ));
            }
        };
        let remote_ext = Path::new(&tg_file_path)
            .extension()
            .and_then(|e| e.to_str());

        let file_data = match self.download_file(&tg_file_path).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to download attachment: {e}");
                return Ok(Some(
                    self.reject_attachment(
                        ctx,
                        &attachment,
                        remote_ext,
                        NOT_RECEIVED,
                        download_failure_reason(&e),
                    )
                    .await,
                ));
            }
        };

        // The directory is created only once the bytes are here so a refused
        // transfer leaves nothing behind.
        let staging_dir = crate::util::telegram_staging_dir_name(&ctx.chat_id, ctx.message_id);
        let save_dir = crate::util::telegram_files_root().join(&staging_dir);
        if let Err(e) = tokio::fs::create_dir_all(&save_dir).await {
            tracing::warn!("Failed to create telegram attachment directory: {e}");
            return Ok(Some(
                self.reject_attachment(
                    ctx,
                    &attachment,
                    remote_ext,
                    NOT_STORED,
                    ATTACHMENT_DIR_REFUSED_REASON,
                )
                .await,
            ));
        }

        let local_filename =
            local_attachment_name(&attachment, &ctx.chat_id, ctx.message_id, remote_ext);
        let local_path = save_dir.join(&local_filename);
        if let Err(e) = tokio::fs::write(&local_path, &file_data).await {
            tracing::warn!("Failed to save attachment to {}: {e}", local_path.display());
            // A partial write must not linger: nothing downstream will ever
            // reference this per-message directory.
            let _ = tokio::fs::remove_dir_all(&save_dir).await;
            return Ok(Some(
                self.reject_attachment(
                    ctx,
                    &attachment,
                    remote_ext,
                    NOT_STORED,
                    ATTACHMENT_SAVE_REFUSED_REASON,
                )
                .await,
            ));
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

        Ok(Some(ctx.into_attachment_message(content, staging_dir)))
    }

    /// Tell the user an attachment could not be taken and return the
    /// caption-preserving channel message, so an oversized or refused file still
    /// produces a user-visible reply and an agent turn. `outcome` is
    /// [`NOT_RECEIVED`] or [`NOT_STORED`] — it decides whether the note may say
    /// the file never arrived. The name in the note is
    /// [`local_attachment_name`]'s.
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
        let content = attachment_rejection_content(
            Some(&display_name),
            outcome,
            reason,
            attachment.caption.as_deref(),
        );
        self.refuse_with_notice(ctx, &notice, content).await
    }

    /// Send the user-visible refusal `notice` — best-effort, a send failure is
    /// only logged — and return the channel message carrying the agent-facing
    /// `content`.
    async fn refuse_with_notice(
        &self,
        ctx: MessageContext,
        notice: &str,
        content: String,
    ) -> ChannelMessage {
        let (_, thread_id) = parse_recipient(&ctx.reply_target);
        if let Err(e) = self
            .send_text_chunks(notice, &ctx.chat_id, thread_id, None)
            .await
        {
            tracing::warn!(%notice, "Failed to send refusal notice: {e}");
        }
        ctx.into_channel_message(content, None)
    }

    /// Refuse a recorded voice message where the local audio subsystem does
    /// not exist, before the file is fetched. Same shape as
    /// [`Self::reject_attachment`], except that the note names no file: the
    /// message carries no file name, and the fallback name
    /// [`local_attachment_name`] builds has the wrong extension, so naming it
    /// would mislead.
    #[cfg(not(target_os = "macos"))]
    async fn reject_voice_message(
        &self,
        ctx: MessageContext,
        caption: Option<&str>,
    ) -> ChannelMessage {
        let notice = format!("Could not process the voice message: {VOICE_UNSUPPORTED_REASON}.");
        let content =
            attachment_rejection_content(None, NOT_RECEIVED, VOICE_UNSUPPORTED_REASON, caption);
        self.refuse_with_notice(ctx, &notice, content).await
    }

    /// What Telegram says about a message being forwarded content.
    ///
    /// The modern `forward_origin` is read first and the older fields (see
    /// [`LEGACY_FORWARD_FIELDS`]) only where it names nothing — an update carrying
    /// both must still yield ONE attribution, and an origin of a kind this product
    /// has no label for falls back to the older fields rather than dropping the
    /// attribution. The message kinds the product does not handle at all (a
    /// sticker, a poll, a location) produce no message, so nothing here has to
    /// attribute them.
    fn forward(message: &serde_json::Value) -> Forward {
        let mut forwarded = false;
        let mut label = None;
        if let Some(origin) = message.get("forward_origin") {
            forwarded = true;
            label = Self::forward_origin_label(origin);
        }
        for (field, render) in LEGACY_FORWARD_FIELDS {
            let Some(value) = message.get(field) else {
                continue;
            };
            forwarded = true;
            if label.is_none() {
                label = render(value);
            }
        }
        Forward {
            forwarded,
            prefix: label.map(|label| format!("{FORWARD_ATTRIBUTION_PREFIX}{label}] ")),
        }
    }

    /// The label the modern `forward_origin` names, if it names one this product
    /// knows: an origin of a kind it does not know yields none, so the older fields
    /// still get their chance to attribute the message.
    fn forward_origin_label(origin: &serde_json::Value) -> Option<String> {
        origin
            .get("type")
            .and_then(serde_json::Value::as_str)
            .and_then(|kind| match kind {
                // MessageOriginUser
                "user" => origin.get("sender_user").map(format_sender_label),
                // MessageOriginHiddenUser
                "hidden_user" => origin
                    .get("sender_user_name")
                    .and_then(serde_json::Value::as_str)
                    .map(String::from),
                // MessageOriginChat / MessageOriginChannel: the chat's title, named
                // as a channel when the origin says it is one.
                "chat" | "channel" => origin
                    .get("sender_chat")
                    .and_then(|chat| chat.get("title").and_then(serde_json::Value::as_str))
                    .map(|title| {
                        let kind = if kind == "channel" { "channel: " } else { "" };
                        format!("{kind}{title}")
                    }),
                _ => None,
            })
    }

    async fn parse_update_message(
        &self,
        update: &serde_json::Value,
    ) -> anyhow::Result<Option<ChannelMessage>> {
        let Some(message) = update.get("message") else {
            return Ok(None);
        };
        let Some(text) = message.get("text").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let Some(ctx) = self.extract_message_context(message).await? else {
            return Ok(None);
        };

        // Strip @BotUsername suffix from commands (e.g. `/new@MyBot` → `/new`)
        // Telegram appends the bot username to commands in group chats.
        let text = if text.starts_with('/') {
            text.split('@').next().unwrap_or(text)
        } else {
            text
        };

        let content = text.to_string();

        Ok(Some(ctx.into_channel_message(content, None)))
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
        // Debug: one record per successfully delivered attachment — routine traffic
        // on a path whose failures are reported separately, at warn.
        tracing::debug!("Telegram {api_method} sent to {chat_id}: {label}");
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
    /// Successfully fetched updates. Still unconfirmed on the platform's side —
    /// the caller's next poll's offset is what confirms them.
    Updates(Vec<serde_json::Value>),
    /// 409 Conflict — caller decides backoff.
    Conflict,
    /// Non-409 API error with description.
    Error(String),
    /// Network or parse error (sleep already applied by helper).
    Transport,
}

impl TelegramChannel {
    /// Call `getUpdates` and classify the outcome.
    ///
    /// The position is the caller's: it asks for everything at or past `offset`,
    /// and what the next poll resumes from is computed from the intake's record
    /// rather than from this call.
    ///
    /// `ok_default` controls what happens when the `ok` field is missing from
    /// the response: `false` (handshake) treats it as an error, `true` (main
    /// loop) assumes success to be lenient.
    async fn poll_get_updates(&self, offset: i64, timeout: u64, ok_default: bool) -> PollOutcome {
        let url = self.api_url("getUpdates");
        let body = serde_json::json!({
            "offset": offset,
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

    /// The intake's opening handshake: resolve the account the token belongs to
    /// and claim the `getUpdates` slot. Both must hold before the intake may take
    /// an event on — the record is keyed by the account's own identity, so the
    /// platform has to name it before anything is polled, and a previous daemon's
    /// 30-second poll may still be active on the platform's side.
    ///
    /// Retries until the listener is cancelled or the daemon drains, `None` when
    /// either happens while waiting. The batch the successful claim fetched is
    /// returned rather than discarded: it is unconfirmed on the platform's side,
    /// so discarding it would throw away an event the platform has not given up
    /// on.
    async fn open_intake(&self) -> Option<IntakeSlot> {
        loop {
            if self.cancel.is_cancelled() || crate::shutdown::is_draining() {
                tracing::info!("Telegram intake stopping during its startup handshake");
                return None;
            }
            let account_id = match self.bot_account_id().await {
                Ok(account_id) => account_id,
                Err(err) => {
                    tracing::warn!("Startup handshake: getMe failed: {err}; retrying in 5s");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
            };
            match self.poll_get_updates(0, 0, false).await {
                PollOutcome::Updates(updates) => {
                    return Some(IntakeSlot {
                        account_id,
                        updates,
                    });
                }
                PollOutcome::Conflict => {
                    tracing::debug!("Startup handshake: slot busy (409), retrying in 5s");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                PollOutcome::Error(desc) => {
                    tracing::warn!("Startup handshake: API error: {desc}; retrying in 5s");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                PollOutcome::Transport => {} // sleep already applied by helper
            }
        }
    }

    /// The account the token belongs to, as the platform reports it.
    ///
    /// The intake keys its record on this rather than on the token's text, so
    /// rotating the token of the same account still recovers what that account is
    /// holding, while a token pointing at another account neither reads nor is held
    /// back by what the first one left.
    async fn bot_account_id(&self) -> anyhow::Result<String> {
        let data = get_me(self.http_client(), &self.api_base, &self.bot_token).await?;
        let id = data["result"]["id"]
            .as_i64()
            .ok_or_else(|| anyhow::anyhow!("getMe response carries no bot id"))?;
        Ok(id.to_string())
    }

    /// Parse one incoming update into a message — polled or recovered: a text
    /// message, an attachment, or (logged and skipped) nothing. A parsed update
    /// gets its acknowledgement reaction, as it always has. The error is the
    /// sender-resolution read failing, which is not "nothing to make of it" and
    /// is left for the caller to keep the event held for.
    async fn parse_incoming_update(
        &self,
        update: &serde_json::Value,
    ) -> anyhow::Result<Option<ChannelMessage>> {
        let msg = if let Some(msg) = self.parse_update_message(update).await? {
            msg
        } else if let Some(msg) = self.try_parse_attachment_message(update).await? {
            msg
        } else {
            Self::handle_non_parseable_message(update);
            return Ok(None);
        };

        // Send ACK reaction for every individual update (fire-and-forget)
        if let Some((reaction_chat_id, reaction_message_id)) =
            Self::extract_update_message_target(update)
        {
            self.try_add_ack_reaction_nonblocking(reaction_chat_id, reaction_message_id);
        }
        Ok(Some(msg))
    }

    /// Take one polled batch on: record every event, then hand on only the events
    /// this call was the first to record. `None` when the pipeline is gone (the
    /// listener stops).
    ///
    /// The batch is recorded unit by unit — an album's members as one unit, every
    /// other update on its own (see [`poll_plan`]) — and a unit the record could not
    /// be written for holds the position where the unit starts (at the offset the
    /// request started from, when the platform never identified it), so the platform
    /// offers the unit again rather than the intake handing on the parts of it that
    /// could be recorded.
    async fn take_up(
        &self,
        parts_tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
        intake: &Intake,
        offset: i64,
        updates: Vec<serde_json::Value>,
        origin: BatchOrigin,
    ) -> Option<Polled> {
        // A long poll that timed out with nothing to offer leaves the position
        // where it is: there is nothing behind it the platform has not given up
        // on already.
        if updates.is_empty() {
            return Some(Polled {
                position: offset,
                reoffered: false,
            });
        }
        let plan = poll_plan(&updates);
        let mut fresh = HashSet::new();
        // The unit the record could not be written for — an unwritable store, or an
        // update the platform never identified, which can be neither held nor
        // confirmed past (there is no `update_id` to move past): it holds the
        // position where the request started, and the platform's own offering window
        // is what retires it.
        let mut unrecorded: Option<i64> = None;
        let mut record_error: Option<anyhow::Error> = None;
        let mut failed_units = 0usize;
        for (at, placement) in plan.iter().enumerate() {
            let unit: Vec<&serde_json::Value> = match placement {
                PollPlacement::Skip => continue,
                PollPlacement::Album(members) => {
                    members.iter().map(|&member| &updates[member]).collect()
                }
                PollPlacement::Single | PollPlacement::Callback => vec![&updates[at]],
            };
            let first = unit
                .first()
                .and_then(|update| update["update_id"].as_i64())
                .unwrap_or(offset);
            match intake.hold(&unit).await {
                Ok(recorded) => fresh.extend(recorded),
                Err(err) => {
                    unrecorded = Some(unrecorded.map_or(first, |smallest| smallest.min(first)));
                    failed_units += 1;
                    record_error.get_or_insert(err);
                }
            }
        }

        // One warning per batch, not one per event: the pinned position re-offers the
        // same page every pause, so a persistently unwritable store would otherwise log
        // a warning per update per poll.
        if let Some(err) = record_error {
            tracing::warn!(
                first_unrecorded = ?unrecorded,
                units = failed_units,
                %err,
                "Telegram intake: could not record part of a batch"
            );
        }

        // The position stops at the first event the record could not be written
        // for: the platform then offers that one, and everything behind it, again.
        let position = unrecorded.unwrap_or_else(|| {
            updates
                .iter()
                .filter_map(|update| update["update_id"].as_i64())
                .max()
                .map_or(offset, |newest| newest + 1)
        });

        if !self
            .process_updates(parts_tx, intake, &mut fresh, origin, updates, &plan)
            .await
        {
            return None;
        }
        // The platform offers the batch again immediately, so a record that could
        // not be written for would otherwise spin.
        if unrecorded.is_some() {
            self.intake_pause(RECORD_FAILURE_BACKOFF).await;
        }
        Some(Polled {
            position,
            reoffered: unrecorded.is_some(),
        })
    }

    /// Parse a recorded batch and hand it to the collector.
    ///
    /// `fresh` are the events this call was the first to record; the rest are
    /// skipped. An event nothing can be made of is settled here as the record's
    /// own decision, so it neither comes back nor grows the record. Returns
    /// `false` when the collector is gone.
    ///
    /// The one thing parsed out of turn is an album (media group): its members are
    /// parsed together at the position of the album's first member, so an album is
    /// delivered where the user sent it instead of after the rest of the poll. The
    /// grouping is the caller's ([`poll_plan`]), which is also what recorded the
    /// batch, so both walk one batch the same way.
    async fn process_updates(
        &self,
        tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
        intake: &Intake,
        fresh: &mut HashSet<i64>,
        origin: BatchOrigin,
        updates: Vec<serde_json::Value>,
        plan: &[PollPlacement],
    ) -> bool {
        for (update, placement) in updates.iter().zip(plan) {
            match placement {
                // A button press is not a message: its `message` is the message
                // the button sits on (see `poll_plan`).
                PollPlacement::Callback => {
                    let Some(update_id) = Self::fresh_update(update, fresh) else {
                        continue;
                    };
                    if !self
                        .handle_callback_press(tx, intake, origin, update_id, update)
                        .await
                    {
                        return false;
                    }
                }
                PollPlacement::Skip => {}
                PollPlacement::Single => {
                    let Some(update_id) = Self::fresh_update(update, fresh) else {
                        continue;
                    };
                    if let Some(msg) = self.take_event(intake, origin, update_id, update).await
                        && tx.send(msg).await.is_err()
                    {
                        return false;
                    }
                }
                PollPlacement::Album(members) => {
                    let mut album = None;
                    for &member in members {
                        let update = &updates[member];
                        let Some(update_id) = Self::fresh_update(update, fresh) else {
                            continue;
                        };
                        // A member nothing can be made of is settled by `take_event`,
                        // and one it cannot decide about yet stays held; either way
                        // the rest of the album goes on.
                        if let Some(msg) = self.take_event(intake, origin, update_id, update).await
                        {
                            album = Some(merge_album_member(album, msg));
                        }
                    }
                    if let Some(album) = album
                        && tx.send(album).await.is_err()
                    {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// The record entry of a batch update, when this call is the first to take it
    /// up — taken out of `fresh`, so a batch that repeats an `update_id` walks
    /// the event once.
    ///
    /// `fresh` is the set of ids the record accepted rather than one flag per unit:
    /// an album is recorded whole or not at all, but what `hold` answers is which of
    /// the unit's ids it inserted, and a member the record already held is not handed
    /// on with its album.
    fn fresh_update(update: &serde_json::Value, fresh: &mut HashSet<i64>) -> Option<i64> {
        let id = update["update_id"].as_i64()?;
        fresh.remove(&id).then_some(id)
    }

    /// Take one recorded event on: parse it, and settle it here when there is
    /// nothing to hand on — an event nothing can be made of, or control input of
    /// unknown age, which is not executed out of context.
    ///
    /// Three things can come of the parse, and they are not the same decision. A
    /// message is handed on. Nothing can be made of the event — `Ok(None)` — which
    /// is the record's own decision: it is settled and neither comes back nor grows
    /// the record. The read the decision needs failed — `Err` — which is no decision
    /// at all: the event stays held, payload and all, exactly as one a crash left
    /// behind, and a later run takes it up.
    async fn take_event(
        &self,
        intake: &Intake,
        origin: BatchOrigin,
        update_id: i64,
        update: &serde_json::Value,
    ) -> Option<ChannelMessage> {
        let mut msg = match self.parse_incoming_update(update).await {
            Ok(Some(msg)) => msg,
            // Nothing can be made of this event: that is a decision of the record's
            // own, so it is settled and neither comes back nor grows the record.
            Ok(None) => {
                intake.settle(&[update_id]).await;
                return None;
            }
            // The read this decision needs failed, so no decision was made: the
            // event stays held, payload and all, and a later run takes it up — the
            // same way what a crash left held is taken up. Refusing it here would
            // drop it for good, and receiving it again is the side to err on.
            Err(err) => {
                tracing::warn!(
                    update_id,
                    %err,
                    "Telegram intake: could not decide an event; it stays held for a later run"
                );
                return None;
            }
        };
        msg.update_ids.push(update_id);
        if origin == BatchOrigin::Stale && is_control_message(&msg) {
            self.refuse_unacted_control(intake, &msg).await;
            return None;
        }
        Some(msg)
    }

    /// Tell the user a control input the product never acted on was not acted on,
    /// and settle it. Control input in a batch of unknown age
    /// ([`BatchOrigin::Stale`]) is not executed out of context, and a press of that
    /// age had its press-time answer expire just as long ago — so the plain notice
    /// is all it gets.
    ///
    /// An event whose notice could not be sent stays held: being told is the whole
    /// of what the event is for, so a later run produces the notice again rather than
    /// the user never hearing it.
    async fn refuse_unacted_control(&self, intake: &Intake, msg: &ChannelMessage) {
        let (chat_id, thread_id) = parse_recipient(&msg.reply_target);
        if let Err(err) = self
            .send_text_chunks(UNACTED_CONTROL_NOTICE, chat_id, thread_id, None)
            .await
        {
            tracing::warn!(%err, "Telegram: could not report an unacted command to its user");
            return;
        }
        intake.settle(&msg.update_ids).await;
    }

    /// Handle one button press, and report whether the pipeline is still alive.
    /// The press is parsed on its own — never as the message it sits on — and
    /// handed over as control input.
    ///
    /// The spinner is dismissed only for a press this run polled — the answer of
    /// a press of unknown age expired just as long ago, so that would be a doomed
    /// API call. A press nothing can be made of is settled here, so it is not
    /// offered again; a press the intake cannot yet decide about stays held, and a
    /// later run takes it up.
    async fn handle_callback_press(
        &self,
        tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
        intake: &Intake,
        origin: BatchOrigin,
        update_id: i64,
        update: &serde_json::Value,
    ) -> bool {
        let cq = &update["callback_query"];
        let cq_id = cq["id"].as_str().map(ToString::to_string);
        let cq_data = cq["data"].as_str().unwrap_or("");

        // For __act__ callbacks, do NOT answer early — the action handler
        // (handle_action_callback in main.rs) will answer with the appropriate
        // toast text. Dismiss the spinner now for all other callbacks.
        if origin == BatchOrigin::Fresh
            && !cq_data.starts_with(ACTION_PREFIX)
            && let Some(ref id) = cq_id
        {
            self.answer_callback_query(id, None).await;
        }

        let mut msg = match self.parse_callback_query(cq).await {
            Ok(Some(msg)) => msg,
            // Nothing can be made of this press: that is a decision of the record's
            // own, so it is settled and neither comes back nor grows the record.
            Ok(None) => {
                intake.settle(&[update_id]).await;
                return true;
            }
            // No decision was made, so none is written: the press stays held,
            // payload and all, and a later run takes it up (see `take_event`).
            Err(err) => {
                tracing::warn!(
                    update_id,
                    %err,
                    "Telegram intake: could not decide an event; it stays held for a later run"
                );
                return true;
            }
        };
        msg.update_ids.push(update_id);
        // A press is control input by construction, so it is refused by the same
        // rule `take_event` applies.
        if origin == BatchOrigin::Stale {
            self.refuse_unacted_control(intake, &msg).await;
            return true;
        }
        tx.send(msg).await.is_ok()
    }

    /// Take up the events this account's record holds and no run has handed on:
    /// what a crash, a self-update or a kill in the wrong instant left behind.
    ///
    /// A recovered event joins the same collector as a fresh one, which is why an
    /// event from the record and one from a poll by the same person can land in
    /// one burst. Returns `false` when the pipeline is gone.
    async fn recover(
        &self,
        intake: &Intake,
        parts_tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
    ) -> bool {
        // A take-over that fails leaves every row it would have claimed held, and the
        // only other thing that takes them up is a whole listener rebuild — so a
        // failure that is only transient is worth another try before the event waits
        // that long.
        let mut attempt = 0;
        let updates = loop {
            match intake.take_over().await {
                Ok(updates) => break updates,
                Err(err) => {
                    attempt += 1;
                    if attempt == TAKE_OVER_ATTEMPTS {
                        tracing::warn!(
                            %err,
                            "Telegram intake: could not take up what the record holds; \
                             what it holds stays for the next run"
                        );
                        return true;
                    }
                    tracing::warn!(%err, "Telegram intake: could not take up what the record holds; retrying");
                    self.intake_pause(TAKE_OVER_RETRY).await;
                }
            }
        };
        if updates.is_empty() {
            return true;
        }
        tracing::info!(
            account_id = %intake.account_id,
            held = updates.len(),
            "Telegram intake: taking up events the record holds"
        );
        let mut fresh: HashSet<i64> = updates
            .iter()
            .filter_map(|update| update["update_id"].as_i64())
            .collect();
        // The album regrouping the stored updates came back in is the same planning
        // the poll path uses, so a recovered album is one event here too.
        let plan = poll_plan(&updates);
        self.process_updates(
            parts_tx,
            intake,
            &mut fresh,
            BatchOrigin::Stale,
            updates,
            &plan,
        )
        .await
    }

    /// Age the record out — the intake's own retention job, over the events it
    /// holds and the releases it recorded.
    async fn age_out_record(&self) {
        match self.intake_store().age_out().await {
            Ok(0) => {}
            Ok(dropped) => tracing::debug!(
                dropped,
                "Telegram intake: aged rows out of the intake's record"
            ),
            Err(err) => tracing::warn!(%err, "Telegram intake: could not age the record"),
        }
    }

    /// The intake proper, once the handshake has resolved the account and fetched the
    /// opening batch: the sweep, what the record holds, and the long-poll loop.
    ///
    /// Returns when the listener stops, however it stops; the frame stays the collector's
    /// until the burst it still held has been handed on ([`ListenerFrame`]).
    async fn intake_loop(
        &self,
        parts_tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
        intake: &Intake,
        handshake: Vec<serde_json::Value>,
    ) {
        // The sweep runs before anything is taken up: a row that outlived the
        // retention bound must never stand in for a later event the platform has
        // re-used the id of, and dropping it is the only thing that makes that so.
        self.age_out_record().await;
        let mut aged_at = Instant::now();

        if !self.recover(intake, parts_tx).await {
            tracing::warn!("Telegram listener stopping: the message pipeline is closed");
            return;
        }

        // The batch the handshake fetched was fetched at offset 0, before any
        // position existed, so it is input of unknown age — as is every batch behind
        // it: a fetch that fills the platform's own page may not have come to the end
        // of what it was already holding, and only a fetch that comes back short says
        // the platform has nothing more from before this run. Control input that
        // arrived in any of those batches must not be executed out of context.
        let page_full = handshake.len() >= UPDATES_PAGE;
        let Some(taken) = self
            .take_up(parts_tx, intake, 0, handshake, BatchOrigin::Stale)
            .await
        else {
            tracing::warn!("Telegram listener stopping: the message pipeline is closed");
            return;
        };
        let mut offset = taken.position;
        let mut backlog_open = page_full || taken.reoffered;

        let shutdown_token = crate::shutdown::shutdown_token();
        let per_channel_cancel = self.cancel.clone();

        loop {
            tokio::select! {
                () = shutdown_token.cancelled() => {
                    tracing::info!("Telegram channel shutting down (global shutdown)");
                    break;
                }
                () = per_channel_cancel.cancelled() => {
                    tracing::info!("Telegram channel shutting down (token hot-reload)");
                    break;
                }
                // The intake stops fetching before it stops handling: an event
                // arriving during the wind-down is either left unconfirmed for
                // the next start or already held durably. The channel itself
                // stays registered — returning from here does not unregister it
                // — and outbound sending keeps working.
                () = crate::shutdown::drain_wait() => {
                    tracing::info!("Telegram channel stopping the intake (daemon draining)");
                    break;
                }
                poll_result = self.poll_get_updates(offset, 30, true) => {
                    // Every completed poll ages resolved rows out, whatever it
                    // carried: a run that only ever sees errors or empty batches
                    // must still trim the record.
                    if aged_at.elapsed() >= AGE_OUT_INTERVAL {
                        self.age_out_record().await;
                        aged_at = Instant::now();
                    }
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
                            // Informational: the poll is retried every 5s, so an
                            // unhealthy API would otherwise keep the issues view full.
                            tracing::info!("Telegram getUpdates API error: {desc}");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            continue;
                        }
                        PollOutcome::Transport => continue,
                    };

                    let page_full = updates.len() >= UPDATES_PAGE;
                    let Some(taken) = self
                        .take_up(parts_tx, intake, offset, updates, batch_origin(backlog_open))
                        .await
                    else {
                        tracing::warn!(
                            "Telegram listener stopping: the message pipeline is closed"
                        );
                        return;
                    };
                    offset = taken.position;
                    backlog_open = (backlog_open && page_full) || taken.reoffered;
                }
            }
        }
    }
}

/// What the opening handshake resolves to.
struct IntakeSlot {
    /// The bot account the token belongs to, as the platform names it.
    account_id: String,
    /// The batch the successful claim fetched — still unconfirmed on the
    /// platform's side, so exactly an event the intake now has to hold.
    updates: Vec<serde_json::Value>,
}

/// Fold a parsed album member into the album message built so far: content
/// joined with a newline (the album reads as one message) and the member's
/// staging directories kept, since each member downloads into its own. The
/// first member starts the album.
///
/// A member being forwarded content makes the album forwarded content: whichever
/// member carries the attribution, the group the album lands in is owed the same
/// longer window. The album's own record time stays the first member's — the
/// platform sends an album's members together.
///
/// The member's record entries come along too: an album is one incoming event
/// whose every part is its own entry, and the merged message is what settles
/// them — so no member is left held by the record after the album is handed on.
fn merge_album_member(album: Option<ChannelMessage>, member: ChannelMessage) -> ChannelMessage {
    match album {
        Some(mut album) => {
            album.content.push('\n');
            album.content.push_str(&member.content);
            album.attachment_dirs.extend(member.attachment_dirs);
            album.update_ids.extend(member.update_ids);
            album.forwarded |= member.forwarded;
            album
        }
        None => member,
    }
}

/// Whether a batch is one this intake's own poll is seeing for the first time.
///
/// Control input is acted on only for a fresh batch: everything else may be older
/// than this run — the opening handshake's batch was fetched before the intake had
/// ever run, a batch whose record could not be written for is offered again to the
/// next poll, and what recovery takes up is what a previous run held and never
/// handed on — so control input in it is not executed out of context. A message is
/// handed over regardless; only control input is affected.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchOrigin {
    /// A batch this run's own poll fetched with nothing of what the platform was
    /// already holding left behind it.
    Fresh,
    /// A batch that may hold events this run already fetched, or events the platform
    /// was holding before this run started.
    Stale,
}
/// What one taken-up batch leaves behind.
struct Polled {
    /// The position the next `getUpdates` resumes from: past the batch once
    /// every event in it is held, or at the unit the record could not be written
    /// for — at the offset the request itself started from, when a unit carries
    /// no `update_id` to pin it at — which the platform then offers again.
    position: i64,
    /// True when part of the batch could not be recorded, so the platform is about
    /// to offer it again — one of the two reasons the batch the next poll fetches is
    /// of unknown age ([`batch_origin`]).
    reoffered: bool,
}

/// The origin of the batch a poll fetches next, from what the batch just taken up
/// left unknown: it filled the platform's own page, so the platform may still be
/// handing over what it was holding before this run (the backlog is open until a
/// fetch comes back short), or part of it could not be recorded, so the platform
/// offers it to the next poll again. Either way the next answer may repeat what this
/// run has already fetched, and control input in it must not run out of context.
fn batch_origin(unknown_age: bool) -> BatchOrigin {
    if unknown_age {
        BatchOrigin::Stale
    } else {
        BatchOrigin::Fresh
    }
}

/// Marks a listener frame as running for the whole of `listen`, so a listener that
/// replaces this one can wait for what this frame held to be handed on or left
/// behind before it takes over ([`TelegramChannel::wait_stopped`]).
///
/// The frame is a value the collector owns, not the `listen` future: a listener
/// whose future is dropped or that panics still leaves the frame open until its
/// collector has finished — so the frame reads as running for exactly as long as
/// anything it held may still be handed on. What no frame covers is the process
/// itself ending, a crash, which is the record's own case.
pub(crate) struct ListenerFrame {
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    stopped: std::sync::Arc<tokio::sync::Notify>,
}

impl ListenerFrame {
    pub(crate) fn enter(
        running: &std::sync::Arc<std::sync::atomic::AtomicBool>,
        stopped: &std::sync::Arc<tokio::sync::Notify>,
    ) -> Self {
        running.store(true, std::sync::atomic::Ordering::Release);
        Self {
            running: std::sync::Arc::clone(running),
            stopped: std::sync::Arc::clone(stopped),
        }
    }
}

impl Drop for ListenerFrame {
    fn drop(&mut self) {
        self.running
            .store(false, std::sync::atomic::Ordering::Release);
        // A running→stopped transition is every waiter's business, not one's: the
        // frame can be waited on by more than one at a time, and a waiter left
        // unnotified would sit out the whole bound.
        self.stopped.notify_waiters();
    }
}

/// How long the intake pauses after a poll whose record could not be written for.
/// The platform offers the batch again immediately, so the pause is what keeps
/// a failing store from becoming a tight loop. Only the unit that could not be
/// recorded holds the position back — the rest of the batch is still recorded and
/// handed on — and the wedge ends with the platform's own offering window, which
/// is what retires the event.
const RECORD_FAILURE_BACKOFF: Duration = Duration::from_secs(5);

/// How often the intake drops the record's rows past the retention bound.
const AGE_OUT_INTERVAL: Duration = Duration::from_hours(1);

/// The platform's own `getUpdates` page size: with no `limit` asked for, a poll
/// hands back at most this many updates, so a batch this size means the platform
/// may still be holding more.
const UPDATES_PAGE: usize = 100;

/// How long a replacement waits for one listener it cancelled to end its frame.
const LISTENER_STOP_WAIT: Duration = Duration::from_secs(30);

/// How many times the intake tries to take up what the record holds before it leaves
/// those rows for the next run, and how long it pauses between tries: a read that
/// fails on contention is worth a retry, and a store that is really unusable must not
/// delay the listener.
const TAKE_OVER_ATTEMPTS: u32 = 3;
const TAKE_OVER_RETRY: Duration = Duration::from_secs(1);

/// What the intake tells a user whose control input it never acted on: the input
/// arrived in a batch of unknown age ([`BatchOrigin::Stale`]), so it is not
/// executed out of context and the user is asked to repeat it.
const UNACTED_CONTROL_NOTICE: &str = "MahBot did not act on that. Please repeat it.";

/// How one update of a poll is parsed.
///
/// An album's members can be separated by other updates in the poll, so an album
/// is parsed whole at the position of its first member and its other members are
/// skipped where they sit — that is what keeps the user's send order.
#[derive(Debug, PartialEq, Eq)]
enum PollPlacement {
    /// Parse and hand over this update on its own.
    Single,
    /// An album's first member: parse every listed update (this one first) and
    /// merge them into the one message an album is.
    Album(Vec<usize>),
    /// A later album member, already carried by its album's first member.
    Skip,
    /// A button press, handled as control input. It carries the message its
    /// button sits on, which is why it is placed apart: reading that message here
    /// would put an album photo into its album a second time, or — when the press
    /// is the album's first member of the batch — leave the album's other members
    /// skipped and lost.
    Callback,
}

/// Read each album's (media group's) member positions from the raw updates, so
/// the poll can be walked once: nothing has to be parsed for this, album
/// membership is a field of the raw update.
fn poll_plan(updates: &[serde_json::Value]) -> Vec<PollPlacement> {
    let mut plan: Vec<PollPlacement> = updates
        .iter()
        .map(|update| match update.get("callback_query") {
            Some(_) => PollPlacement::Callback,
            None => PollPlacement::Single,
        })
        .collect();
    let mut albums: HashMap<&str, Vec<usize>> = HashMap::new();
    for (at, update) in updates.iter().enumerate() {
        if plan[at] == PollPlacement::Callback {
            continue;
        }
        if let Some(group) = update
            .get("message")
            .and_then(|message| message.get("media_group_id"))
            .and_then(serde_json::Value::as_str)
        {
            albums.entry(group).or_default().push(at);
        }
    }
    for members in albums.into_values() {
        let (first, rest) = members.split_first().expect("an album has a member");
        let first = *first;
        for &later in rest {
            plan[later] = PollPlacement::Skip;
        }
        plan[first] = PollPlacement::Album(members);
    }
    plan
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
        let frame = ListenerFrame::enter(&self.listener_running, &self.listener_stopped);
        // A listener starting inside a wind-down takes nothing on: the pipeline
        // is being stopped, so an event fetched now would only have to be held
        // until the next start.
        if crate::shutdown::is_draining() {
            tracing::info!("Telegram channel not starting: the daemon is draining");
            return Ok(());
        }

        tracing::info!("Telegram channel listening for messages...");

        // Opening handshake: the account the token belongs to and the getUpdates
        // slot, before entering the long-poll loop.
        let Some(slot) = self.open_intake().await else {
            return Ok(());
        };

        tracing::debug!("Startup handshake succeeded; entering main long-poll loop.");

        let intake = Intake::new(
            self.intake_store(),
            self.intake_claim.clone(),
            &slot.account_id,
        );

        // Collected bursts are assembled off the poll loop, so getUpdates keeps being
        // issued while a group waits; the queue's bound and what dropping its sender
        // does are `telegram_group`'s. The collector also settles the record: a message
        // it has handed to the pipeline has its events resolved, while one the pipeline
        // never took leaves them held. It takes the frame, so its life is the collector's.
        let (parts_tx, parts_rx) =
            tokio::sync::mpsc::channel(crate::channels::telegram_group::QUEUE_CAPACITY);
        let collector = crate::channels::telegram_group::spawn_collector(
            parts_rx,
            tx,
            intake.clone(),
            std::sync::Arc::clone(&self.preparations),
            frame,
        );

        self.intake_loop(&parts_tx, &intake, slot.updates).await;

        // Every way the intake returns passes through this drain: dropping the sender
        // releases an open burst, which the collector hands to the pipeline and settles,
        // so the frame of this listener ends only once that collector has finished
        // ([`ListenerFrame`]).
        drop(parts_tx);
        if let Err(err) = collector.await {
            tracing::error!(
                %err,
                "Telegram intake: the burst collector failed; what it still held stays in the record"
            );
        }
        Ok(())
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

/// Stop and wait out every listener of the intake's own that may still hold events: the
/// channels a replace has moved past and no hand-over has confirmed stopped
/// ([`crate::ReplacedChannels`]). A recorded channel that is not one of them — the registry
/// is not the intake's alone — has no frame to wait out and is only taken out of the list.
///
/// `starting` is the channel the caller is about to start, which is never stopped
/// ([`hand_over_listener`]); it is left in the list rather than retired, so a hand-over
/// that lands later still finds it there and stops it.
///
/// Taken from the name's own list rather than from what the channels carry, so every
/// hand-over waits for the same frames: a chain the channels carry is consumed by whichever
/// hand-over walks it first, and a save landing while an earlier one waited would then start
/// a listener with the frame that holds undelivered events still open — handing that content
/// on twice.
///
/// A frame still open past the wait bound is left in the list, so the next hand-over waits it
/// again rather than taking its place behind its back.
async fn stop_listeners(
    frames: &crate::ReplacedChannels,
    starting: Option<&std::sync::Arc<dyn Channel>>,
) {
    for frame in frames.snapshot() {
        if starting.is_some_and(|starting| std::sync::Arc::ptr_eq(starting, &frame)) {
            continue;
        }
        if let Some(telegram) = frame.as_any().downcast_ref::<TelegramChannel>() {
            telegram.cancel_own();
            telegram.wait_stopped().await;
            if telegram.running() {
                continue;
            }
        }
        frames.retire(&frame);
    }
}

/// Hot-reload the Telegram bot listener with a new token.
///
/// Used when the user changes the bot token in Settings — no full application
/// restart required.
///
/// The hand-over runs in a task of its own, so the caller gets its answer as soon as
/// the registry holds the new channel: whether the caller is a settings save or the
/// assistant's config tool, dropping it can no longer leave the registry holding a
/// channel nothing listens on while the listener it replaced is already stopped.
///
/// - **New token provided** (`Some(token)`): the pipeline is resolved (the listener
///   cannot run without it), the token is validated and the new channel takes the old
///   one's place in the registry, which records what it moved past; the hand-over task
///   then cancels and waits out every such listener — one whose collector has drained
///   means everything it held is settled or left held, so what is left for the take-over
///   is exactly what it could not hand on — and listens, never stopping the channel it
///   starts itself ([`hand_over_listener`]).
/// - **Token cleared** (`None`): the old listener is cancelled and the channel
///   unregistered once it has ended its frame — unless a token saved in the meantime
///   has already put another channel in its place, which must stay registered. The
///   channel keeps serving outbound sends while it winds down, so a reply that is
///   already on its way still leaves through the token being removed.
pub async fn restart_telegram_listener(new_token: Option<&str>) -> anyhow::Result<()> {
    let registry = crate::channel_registry();

    if let Some(token) = new_token.filter(|t| !t.trim().is_empty()) {
        // The listener cannot be started without the pipeline, so the pipeline is
        // resolved before anything is touched — no platform call, no registry entry —
        // and a missing one is reported to the caller rather than leaving a token the
        // product never applied looking saved.
        let Some(tx) = crate::MESSAGE_TX.get().cloned() else {
            anyhow::bail!(
                "the message pipeline is not running — the Telegram listener cannot start"
            );
        };

        // Validate the new token before touching anything.
        TelegramChannel::validate_token(token).await?;

        // The record's rows are keyed by the account and claimed by the listener
        // that holds them, so the replacement records and recovers on its own: what
        // this process already has in flight is left to finish, and what the
        // listener it replaced left held is taken over by the new one.
        let telegram_arc = std::sync::Arc::new(TelegramChannel::new(token.to_string()));

        // Register commands with Telegram API (fire-and-forget, non-blocking).
        tokio::spawn({
            let tc = std::sync::Arc::clone(&telegram_arc);
            async move {
                tc.set_my_commands().await;
            }
        });

        let new_channel: std::sync::Arc<dyn Channel> = telegram_arc;

        // Atomically replace in the registry — no gap where "telegram" returns None —
        // which records what it moved past: the listeners the hand-over waits out.
        let frames = registry.replaced("telegram");
        registry.replace(std::sync::Arc::clone(&new_channel));

        tokio::spawn(hand_over_listener(frames, new_channel, tx));
        tracing::info!("Telegram bot listener restarting with new token");
    } else {
        // Token cleared — stop the old listener and unregister it once it has ended,
        // unless a token saved while it wound down has already taken its place.
        let frames = registry.replaced("telegram");
        let old_channel = registry.get("telegram");
        tokio::spawn(async move { retire_listener(registry, frames, old_channel).await });
    }

    Ok(())
}

/// Stop the listener `old` runs and, once its frame has ended, take it out of the
/// registry — unless the registry holds a channel registered since (a token saved
/// while the teardown waited), which is the one that is live by then.
///
/// The cleared channel is recorded among the frames a hand-over waits out: nothing
/// displaced it, so the teardown is what moves past it, and a save landing meanwhile
/// has to wait it out before it starts its own listener.
async fn retire_listener(
    registry: &crate::ChannelRegistry,
    frames: std::sync::Arc<crate::ReplacedChannels>,
    old: Option<std::sync::Arc<dyn Channel>>,
) {
    let Some(old) = old else {
        return;
    };
    frames.record(std::sync::Arc::clone(&old));
    stop_listeners(&frames, None).await;
    if registry.unregister_if(&old) {
        tracing::info!("Telegram bot token cleared — listener stopped");
    } else {
        tracing::info!("Telegram bot token cleared — a channel registered since is live");
    }
}

/// The hand-over half of [`restart_telegram_listener`], in a task of its own so a
/// caller that goes away cannot desynchronise the registry from the listener: the
/// listeners the replace moved past are stopped and waited out, then the new channel
/// listens on the shared pipeline.
///
/// `new` is the channel this hand-over is about to start, and is passed so the hand-over
/// never stops it: a stop fires the channel's cancel token, so the listener started right
/// after would return at once and the token this save applied would have no listener.
/// `new` is in the list only when a later save's replace has already moved past it, and
/// that needs no case of its own: the superseding hand-over stops `new` — `new` is not the
/// channel *that* hand-over starts — while the channel the last save built is the one that
/// listens.
async fn hand_over_listener(
    frames: std::sync::Arc<crate::ReplacedChannels>,
    new: std::sync::Arc<dyn Channel>,
    tx: tokio::sync::mpsc::Sender<ChannelMessage>,
) {
    stop_listeners(&frames, Some(&new)).await;
    if let Err(e) = new.listen(tx).await {
        tracing::error!(error = %e, "Telegram listener error after hot-reload");
    }
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
/// * Only mirrors messages where `channel == "gui"`, `channel == "voice"` or
///   `channel == "cli"` (prevents echo loops: those are strictly local sources
///   that can never originate from Telegram, so accepting them cannot create
///   feedback; Telegram-originated messages remain excluded). A command-line
///   message is the owner's own, so it mirrors as one of his — with nothing added
///   to its text, exactly as one he typed into the dashboard carries none.
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
    // Guard: only mirror local-originated user messages — text typed into the
    // dashboard, spoken locally, or sent from the command line (prevents echo
    // loops: those are strictly local sources that can never originate from
    // Telegram, so accepting them cannot create feedback; Telegram-originated
    // messages remain excluded).
    if msg.channel != "gui" && msg.channel != "voice" && msg.channel != crate::cli::CHANNEL {
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

/// The phase glyph standing in for the phase word on one `/board` listing
/// line. Keyed by the phase itself — the match is exhaustive with no fallback
/// arm, so a newly added phase is a compile error rather than a silently
/// missing (or transposed) row.
///
/// All ten glyphs already default to emoji presentation, so none carries a
/// variation selector. This is deliberately not `TicketPhase::display_name()`:
/// the desktop board badge keeps the phase words.
fn phase_emoji(phase: TicketPhase) -> &'static str {
    match phase {
        TicketPhase::Backlog => "📥",
        TicketPhase::Analysis => "🔍",
        TicketPhase::Planning => "🧭",
        TicketPhase::Queued => "⏳",
        TicketPhase::InDevelopment => "🔨",
        TicketPhase::InSanitation => "🧹",
        TicketPhase::Verification => "🎯",
        TicketPhase::Done => "✅",
        TicketPhase::Cancelled => "🚫",
        TicketPhase::Failed => "❌",
    }
}

/// Format one `/board` listing line: the phase emoji, monospace ticket ID,
/// then the title. Shared by the Telegram `/board` handler and its tests so
/// the format cannot silently drift between them.
///
/// The line opens with the phase emoji rather than a `•`/`*` bullet: a leading
/// `*` would pair with a `*` in a ticket title and swallow the id and title
/// into an italic span. The ticket ID is monospace, and each line converts to
/// Telegram HTML independently, so markdown-special characters in a title
/// cannot corrupt another line's formatting.
#[must_use]
pub fn format_board_line(phase: &TicketPhase, id: &str, title: &str) -> String {
    format!("{} `{}` {}", phase_emoji(*phase), id, title)
}

/// The complete `/board` reply text for `tickets` (already in the board's
/// display order): an all-workspaces header followed by one
/// [`format_board_line`] per ticket.
///
/// The listing spans every workspace, so the header names none of them and the
/// workspace marker of each entry is the ticket id's own prefix. The empty
/// listing says so in the same all-workspaces voice.
#[must_use]
pub fn board_listing_text(tickets: &[&Ticket]) -> String {
    if tickets.is_empty() {
        return "All workspaces — no tickets".to_string();
    }
    let listing = tickets
        .iter()
        .map(|t| format_board_line(&t.phase, &t.id, &t.title))
        .collect::<Vec<_>>()
        .join("\n");
    let noun = if tickets.len() == 1 {
        "ticket"
    } else {
        "tickets"
    };
    format!("All workspaces — {} {noun}\n{listing}", tickets.len())
}

/// (command, description) entries for a user's Telegram command menu, derived
/// from their admin status. Shared by the per-chat `setMyCommands` refresh and
/// the `/start` welcome message.
#[must_use]
pub async fn user_command_entries(user_name: &str) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::new();

    if crate::users::is_admin(user_name).await {
        entries.push(("board".to_string(), BOARD_COMMAND_DESC.to_string()));
        entries.push(("archive".to_string(), ARCHIVE_COMMAND_DESC.to_string()));

        // `/update` is global (the single admin) and shown only when the shared
        // availability cache confirms an update — always for a copy built from
        // sources, and for a downloaded copy only when the release host holds a
        // newer version. The menu reflects the cached state, never a network call
        // per refresh.
        if crate::self_update::should_show_update(crate::self_update::update_availability()) {
            entries.push(("update".to_string(), UPDATE_COMMAND_DESC.to_string()));
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
    // role-switch entry is gone: the Assistant is every account's only role,
    // so there is nothing to switch.
    entries.push(("clear".to_string(), CLEAR_COMMAND_DESC.to_string()));
    entries
}

#[cfg(test)]
#[path = "telegram_tests.rs"]
mod tests;
