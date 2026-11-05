//! The Telegram transport: Bot API long polling, update decoding, send and edit with Telegram's
//! HTML subset, media up/down, and the forward-only intake ledger that stops an update being
//! handled twice.

use crate::channels::ReplyReference;
use crate::channels::reply::normalize_reply_text;
use crate::channels::telegram_intake::{Intake, TelegramIntakeStore, listener_claim};
use crate::pipeline::board::{Ticket, TicketPhase};
use crate::util::html::{decode_html_entities, escape_html, push_escaped};
use crate::util::media_target::{self, MarkerTarget, MediaTarget};
use crate::util::{
    MEDIA_MARKER_RE, MediaMarkerKind, UnwrapPoison, is_http_url, parse_media_marker,
};
use crate::{Channel, ChannelMessage, RefusedMarker, SendFailure, SendMessage, SendOutcome};
use anyhow::Context;
use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;

const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;
/// Room for the continuation markers `send_text_chunks` adds: 29 chars worst case (a middle chunk),
/// so 30 keeps 4066+29 = 4095 within the 4096 limit.
const TELEGRAM_CONTINUATION_OVERHEAD: usize = 30;

/// Byte budget for a sanitized attachment filename, extension included: bytes rather than
/// characters, because `NAME_MAX` counts bytes.
const MAX_ATTACHMENT_FILENAME_BYTES: usize = 180;

/// Per-media-request timeout, overriding the channel client's one minute: a 50 MB cloud upload can
/// outlast it. Bounded, so a stall still ends.
const MEDIA_TRANSFER_TIMEOUT: Duration = Duration::from_mins(5);

const CLEAR_COMMAND_DESC: &str = "Reset your session";
const IMAGE_MODELS_COMMAND_DESC: &str = "Select image generation model";
const VIDEO_MODELS_COMMAND_DESC: &str = "Select video model";
const BOARD_COMMAND_DESC: &str = "List tickets from all workspaces";
const ARCHIVE_COMMAND_DESC: &str = "Archive done & cancelled tickets (all workspaces)";
const UPDATE_COMMAND_DESC: &str = "Update MahBot to the latest version";

const ACTION_PREFIX: &str = "__act__";

/// Decode `__act__<action>|<payload>` into `(action, payload)`; `None` when the prefix is absent,
/// an empty payload when there is no `|`.
#[must_use]
pub(crate) fn decode_action(content: &str) -> Option<(String, String)> {
    let rest = content.strip_prefix(ACTION_PREFIX)?;
    match rest.split_once('|') {
        Some((action, payload)) => Some((action.to_string(), payload.to_string())),
        None => Some((rest.to_string(), String::new())),
    }
}

/// The control-plane input a message carries, handled without the assistant; `None` for the agent
/// path, a button press of no known kind included.
#[derive(Debug)]
pub enum ControlInput {
    Action((String, String)),
    Command(crate::BotCommand),
}

/// Which control-plane input a message carries. A bot command counts only on Telegram, and a
/// message that opened with the urgent mark is never one.
#[must_use]
pub fn control_input(msg: &ChannelMessage) -> Option<ControlInput> {
    // The mark was consumed from `!/clear`, which arrives here as `/clear`.
    if msg.urgent {
        return None;
    }
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

/// True for input the burst collector must never hold: anything [`control_input`] recognizes, or a
/// button press — a superset of [`control_input`].
#[must_use]
pub(crate) fn is_control_message(msg: &ChannelMessage) -> bool {
    msg.callback_query_id.is_some() || control_input(msg).is_some()
}

/// One inbound attachment: a document, photo, video, audio or voice message.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IncomingAttachment {
    file_id: String,
    file_name: Option<String>,
    file_size: Option<u64>,
    caption: Option<String>,
    kind: IncomingAttachmentKind,
    mime_type: Option<String>,
    /// The caption opened with the urgent mark, which was consumed from it.
    urgent: bool,
}

/// The kind of an inbound attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncomingAttachmentKind {
    Document,
    Photo,
    Video,
    /// A sound file (the `audio` key): transcribed on macOS, a plain file elsewhere.
    Audio,
    /// A recorded voice message (the `voice` key): OGG/Opus bytes, no file name.
    Voice,
}
/// Split a message into chunks within Telegram's 4096 limit, preferring a word break and reserving
/// room for the continuation markers.
fn split_message_for_telegram(message: &str) -> Vec<String> {
    if message.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH {
        return vec![message.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = message;
    let chunk_limit = TELEGRAM_MAX_MESSAGE_LENGTH - TELEGRAM_CONTINUATION_OVERHEAD;

    while !remaining.is_empty() {
        let hard_split = remaining
            .char_indices()
            .nth(chunk_limit)
            .map_or(remaining.len(), |(idx, _)| idx);

        let mut chunk_end = if hard_split == remaining.len() {
            hard_split
        } else {
            find_split_boundary(remaining, hard_split)
        };

        // Extend past the tag's `>`, clamped to `hard_split` (a byte offset, always a char
        // boundary): a chunk past the sendable limit is rejected outright, while a mid-tag split
        // only degrades formatting.
        if let Some(adjusted) = extend_past_open_tag(remaining, chunk_end) {
            chunk_end = adjusted.min(hard_split);
        }

        chunks.push(remaining[..chunk_end].to_string());
        remaining = &remaining[chunk_end..];
    }

    chunks
}

/// Add the continuation markers (`(continued)`, `(continues...)`) a multi-part message needs; a
/// single chunk passes through unchanged.
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

/// Best split offset within `hard_split`: a newline, else a space, else a char boundary.
fn find_split_boundary(text: &str, hard_split: usize) -> usize {
    let search_area = &text[..hard_split];
    search_area
        .rfind('\n')
        .max(search_area.rfind(' '))
        .map_or(hard_split, |p| p + 1)
}

/// Byte offset of the first unquoted `>` at or after `from`; `None` when the tag never closes. Both
/// HTML readers in this module share this scan.
fn tag_end(s: &str, from: usize) -> Option<usize> {
    let mut in_quote = false;
    let mut quote_char = '"';
    for (i, c) in s[from..].char_indices() {
        match c {
            '"' | '\'' if !in_quote => {
                in_quote = true;
                quote_char = c;
            }
            '"' | '\'' if in_quote && c == quote_char => in_quote = false,
            '>' if !in_quote => return Some(from + i),
            _ => {}
        }
    }
    None
}

/// Byte offset just past the `>` closing the tag `pos` sits in, when it is in one.
fn extend_past_open_tag(text: &str, pos: usize) -> Option<usize> {
    let open = text[..pos].rfind('<')?;
    let gt = tag_end(text, open)?;
    (gt >= pos).then_some(gt + 1)
}

fn extract_sender_user_name(message: &serde_json::Value) -> String {
    message
        .get("from")
        .and_then(|from| from.get("username"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(crate::users::TELEGRAM_UNKNOWN_SENTINEL)
        .to_string()
}

struct TelegramSender {
    numeric_id: Option<String>,
    nickname: Option<String>,
}

/// The person a Telegram event came from, or `None` when the sender is not one: a `sender_chat`
/// post, a bot, or a Telegram service identity — decided by the sender's identity, never by the
/// spelling of a nickname.
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
    // The reserved sentinel means "no nickname": it is folded away here and skipped by the store's
    // row scan, so either alone keeps a legacy row inert.
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
    /// The reply this message carries, when it replies to one.
    reply_reference: Option<ReplyReference>,
    /// The `[Forwarded from …]` prefix this message's content opens with, rendered once here so any
    /// path that hands the message on keeps it.
    forward_prefix: Option<String>,
    /// Whether the message is forwarded — the fact the burst collector's longer window rests on,
    /// kept apart from the rendered `forward_prefix`.
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
            display_content: None,
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
            dictated: false,
            urgent: false,
        }
    }

    /// Build the channel message for a downloaded attachment, recording its staging directory as
    /// the only path enrichment may read and delete.
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
    /// The `[FILE:...]` outbound kind — `sendDocument`; see [`Self::file_meta`].
    Document,
    /// The `[VIDEO:...]` kind: delivered like [`Self::Document`], but from any file or address, and
    /// named in the chat.
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

#[derive(Debug, Clone, Copy)]
struct AttachmentMeta {
    api_method: &'static str,
    form_field: &'static str,
    /// Send `disable_content_type_detection=true` so Telegram does not re-classify the upload: an
    /// image sent as a file must not come back as a photo.
    disable_content_type_detection: bool,
}

impl TelegramAttachmentKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Image => "Image",
            Self::Document => "Document",
            Self::Video => "Video",
            Self::Audio => "Audio",
        }
    }

    const fn meta(self) -> AttachmentMeta {
        match self {
            Self::Image => AttachmentMeta {
                api_method: "sendPhoto",
                form_field: "photo",
                disable_content_type_detection: false,
            },
            Self::Document | Self::Video => AttachmentMeta {
                api_method: "sendDocument",
                form_field: "document",
                disable_content_type_detection: true,
            },
            Self::Audio => AttachmentMeta {
                api_method: "sendAudio",
                form_field: "audio",
                disable_content_type_detection: false,
            },
        }
    }

    /// `sendDocument` for a local image: `sendPhoto` re-encodes server-side.
    const fn file_meta(self) -> AttachmentMeta {
        match self {
            Self::Image => Self::Document.meta(),
            _ => self.meta(),
        }
    }
}

/// Image extensions the receive path routes as images; PNG/JPEG/WebP only — gif and bmp have no
/// codec here, and the GIF picker's clips take the video path.
pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// MIME prefixes routed as images, mirroring [`IMAGE_EXTENSIONS`]: PNG/JPEG/WebP only, so no gif
/// and no legacy bmp alias.
const IMAGE_MIME_PREFIXES: &[&str] = &["image/png", "image/jpeg", "image/jpg", "image/webp"];

/// Shared `@username` → `first_name` → `fallback` label chain; the two callers differ only in the
/// sentinel they pass.
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

/// `@username`, else `first_name`, else `"unknown"`.
#[must_use]
fn format_sender_label(from: &serde_json::Value) -> String {
    sender_label_with_fallback(from, crate::users::TELEGRAM_UNKNOWN_SENTINEL)
}

/// The label for a replied-to message's author: a `sender_chat` title, else the
/// `@username`/`first_name` of `from`, with `"user"`/`"channel"` as the terminal fallbacks — never
/// `"unknown"`, which a reply header must not surface.
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

/// Placeholder text for a media reply with no caption: one label per media kind, `[Message]` for
/// everything else.
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

/// A reply snippet: the text or caption, normalized; a media placeholder otherwise.
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

/// The reply reference for a message's `reply_to_message`: its author label plus a normalized
/// snippet or a media-kind placeholder.
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

/// The user-facing content for an inbound attachment: `[IMAGE:]` for a PNG/JPEG/ WebP, `[VIDEO:]`
/// for video, `[AUDIO:]` for sound, `[FILE:]` otherwise — the bytes deciding when name and MIME do
/// not.
fn format_attachment_content(
    kind: IncomingAttachmentKind,
    local_path: &Path,
    mime_type: Option<&str>,
) -> String {
    // MIME fallback mirrors the extension whitelist: PNG/JPEG/WebP only.
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
            // A recording and a sound file both arrive as an AUDIO marker, which enrichment
            // transcribes.
            format!("[AUDIO:{}]", local_path.display())
        }
        _ => {
            // Neither name nor MIME classified it, so the bytes decide — the same decodable- raster
            // gate enrichment uses. It reads and decodes, so it runs off this task.
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

/// Give a Video-kind or `video/*` attachment a recognized video extension: the GIF picker sends MP4
/// bytes named ".gif", which would bounce at video_edit's extension guard.
fn normalize_video_filename(
    kind: IncomingAttachmentKind,
    filename: &str,
    mime_type: Option<&str>,
) -> String {
    let is_video =
        kind == IncomingAttachmentKind::Video || mime_type.is_some_and(|m| m.starts_with("video/"));
    if is_video && !crate::util::is_video_extension(std::path::Path::new(filename)) {
        // No stem (`.`, `..`, `""`): nothing to swap, and the sanitizer falls back.
        let Some(stem) = std::path::Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
        else {
            return filename.to_string();
        };
        // From the declared type, so webm or quicktime is not mislabeled `mp4`.
        let extension = match mime_type {
            Some("video/webm") => "webm",
            Some("video/quicktime") => "mov",
            _ => "mp4",
        };
        format!("{stem}.{extension}")
    } else {
        filename.to_string()
    }
}

/// Make a sender-supplied name safe as one path component under the Telegram temp root: only the
/// final component survives (`../../etc/passwd` → `passwd`), the characters that would break a
/// component or the `[FILE:<path>]` marker are neutralized, and the result fits
/// [`MAX_ATTACHMENT_FILENAME_BYTES`] on a char boundary; `fallback` (cleaned the same way) when the
/// name yields nothing usable.
fn sanitize_attachment_filename(name: &str, fallback: &str) -> String {
    let mut cleaned = [name, fallback]
        .into_iter()
        .find_map(|name| {
            // `Path::file_name` yields only a normal component: `None` for `""`, `.`, `..`.
            let component = Path::new(name).file_name().and_then(|n| n.to_str())?;
            Some(crate::util::neutralized_name(component))
        })
        .unwrap_or_else(|| "file".to_string());
    // Prefix before truncating so the `_` counts toward the budget too.
    if cleaned.starts_with('.') {
        cleaned.insert(0, '_');
    }
    if cleaned.len() > MAX_ATTACHMENT_FILENAME_BYTES {
        // Split off the extension so truncating the stem never eats it.
        let stem = crate::util::name_stem(&cleaned);
        let ext = &cleaned[stem.len()..];
        // A bare extension left by a truncated stem would be a dotfile: the whole name truncates
        // instead.
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

/// Reduce a `getFile` extension to an ASCII-alphanumeric token: it reaches the user's filename, so
/// separators, control characters and `]` must not survive.
fn sanitize_remote_extension(remote_ext: &str) -> Option<String> {
    let cleaned: String = remote_ext
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// The local save name for an inbound attachment: the sender's `file_name`, else a generated
/// `<kind>_<chat>_<message>.<ext>`, either way through [`sanitize_attachment_filename`].
fn local_attachment_name(
    attachment: &IncomingAttachment,
    chat_id: &str,
    message_id: i64,
    remote_ext: Option<&str>,
) -> String {
    let remote_ext = remote_ext.and_then(sanitize_remote_extension);
    let ext = remote_ext.as_deref().unwrap_or(match attachment.kind {
        // video_note has no file_name or mime_type; a ".jpg" default would misroute it.
        IncomingAttachmentKind::Video => "mp4",
        // No name and no MIME type: the generic file path, so the default stays neutral.
        IncomingAttachmentKind::Document => "bin",
        _ => "jpg",
    });
    let prefix = match attachment.kind {
        IncomingAttachmentKind::Photo => "photo",
        IncomingAttachmentKind::Video => "video",
        // A recorded voice message has no `file_name`, so it takes the generated audio name.
        IncomingAttachmentKind::Audio | IncomingAttachmentKind::Voice => "audio",
        IncomingAttachmentKind::Document => "file",
    };
    let fallback = format!("{prefix}_{chat_id}_{message_id}.{ext}");
    // The video extension swap runs before sanitizing: it can lengthen the name (`.bin` → `.webm`)
    // and the byte budget must cover what is written.
    let chosen = normalize_video_filename(
        attachment.kind,
        attachment.file_name.as_deref().unwrap_or(&fallback),
        attachment.mime_type.as_deref(),
    );
    sanitize_attachment_filename(&chosen, &fallback)
}

/// Agent-facing content for a rejected attachment: the caption, then the `[File <name>: <outcome> —
/// <reason>]` note (`display_name` `None` renders it name-less). The kind word never precedes a
/// colon, so it is no media marker.
#[must_use]
fn attachment_rejection_content(
    display_name: Option<&str>,
    outcome: &str,
    reason: &str,
    caption: Option<&str>,
) -> String {
    let note = match display_name {
        Some(name) => format!("[File {name}: {outcome} — {reason}]"),
        None => format!("[File {outcome} — {reason}]"),
    };
    match caption.filter(|c| !c.is_empty()) {
        Some(caption) => format!("{caption}\n\n{note}"),
        None => note,
    }
}

/// How an outbound attachment reaches Telegram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachmentRoute {
    LocalFile,
    /// Telegram fetches the address itself: the image and audio methods.
    ByAddress,
    /// The product fetches the bytes and uploads them: a video by URL, which Telegram's by-address
    /// document form cannot deliver as a video.
    FetchThenUpload,
}

fn attachment_route(kind: TelegramAttachmentKind, target: &str) -> AttachmentRoute {
    if !is_http_url(target) {
        return AttachmentRoute::LocalFile;
    }
    if kind == TelegramAttachmentKind::Video {
        return AttachmentRoute::FetchThenUpload;
    }
    AttachmentRoute::ByAddress
}

/// The address's last segment as a path component, else `fallback`.
fn address_file_name(url: &str, fallback: &str) -> String {
    let parsed = url::Url::parse(url).ok();
    parsed
        .as_ref()
        .and_then(url::Url::path_segments)
        .and_then(|mut segments| segments.next_back())
        .filter(|name| !name.is_empty())
        .map_or_else(
            || fallback.to_string(),
            |name| sanitize_attachment_filename(name, fallback),
        )
}

/// The name a fetched video uploads under: the address's last segment, normalized to a video
/// extension.
fn video_upload_name(url: &str, content_type: Option<&str>) -> String {
    let content_type = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    normalize_video_filename(
        IncomingAttachmentKind::Video,
        &address_file_name(url, "video"),
        content_type,
    )
}

/// The refusal only a `[FILE:...]` marker produces: the other kinds accept any location on disk.
const OUTSIDE_WORKSPACE_REASON: &str = "it is outside your workspace.";

/// The Issues-view record a refused local marker leaves: the path as written and why.
const MARKER_REFUSED_MESSAGE: &str = "a media marker was refused";
/// The Issues-view record a marker delivered through a directory repair leaves: the path as
/// written and where it was delivered from.
const MARKER_REPAIRED_MESSAGE: &str = "a media marker's directory was repaired";
/// What the record's `target` column names — the surface that reported it.
const MARKER_ISSUE_TARGET: &str = "media-marker";
/// The longest path a marker record keeps: a marker may carry an arbitrary blob, and the row is a
/// pointer to what was tried, not a copy of it.
const MARKER_RECORD_PATH_MAX_CHARS: usize = 400;

/// The path a refusal quotes and records, bounded: the target as written, minus what only an
/// address carries. `None` for an inline payload, which names no path. A refusal computes it once,
/// so the note to the conversation and the Issues-view row name the same path.
fn recorded_path(target: &str) -> Option<String> {
    media_target::recordable_target(target)
        .map(|path| crate::util::truncate(&path, MARKER_RECORD_PATH_MAX_CHARS))
}

/// Record one local marker's refusal in the Issues view — the durable trace of a delivery that
/// failed — with the path as written and the reason. `written` is the path the refusal quotes,
/// computed once by the caller; `path` is the local target that failed, and `file_roots` the
/// workspaces it may belong to.
fn record_marker_refusal(written: Option<&str>, reason: &str, path: &Path, file_roots: &[PathBuf]) {
    let Some(written) = written else {
        return;
    };
    crate::logs::record_warn_row(
        MARKER_ISSUE_TARGET,
        MARKER_REFUSED_MESSAGE,
        serde_json::json!({ "path": written, "reason": reason }),
        &workspace_of(path, file_roots),
    );
}

/// Record one repair in the Issues view: a delivery from a path the marker did not name is
/// otherwise invisible, so this row is how a repair is counted and explained.
fn record_marker_repair(target: &str, repaired: &Path, file_roots: &[PathBuf]) {
    let Some(written) = recorded_path(target) else {
        return;
    };
    crate::logs::record_warn_row(
        MARKER_ISSUE_TARGET,
        MARKER_REPAIRED_MESSAGE,
        serde_json::json!({
            "path": written,
            "repaired_to": crate::util::truncate(
                &repaired.to_string_lossy(),
                MARKER_RECORD_PATH_MAX_CHARS,
            ),
        }),
        &workspace_of(repaired, file_roots),
    );
}

/// The workspace a marker belongs to: the root holding its target — a relative target resolves
/// against the first root that has it, so a relayed marker belongs to the workspace it came from.
/// Empty when no root holds it: a path outside every workspace is recorded with no workspace,
/// never attributed to one that does not contain it. The path is lexically normalized first, so a
/// `..` cannot smuggle an outside path into a workspace's count. [`crate::logs`] normalises a path
/// to the workspace's name.
fn workspace_of(path: &Path, file_roots: &[PathBuf]) -> String {
    let normalized = crate::tools::path::normalize_path(path);
    file_roots
        .iter()
        .find(|root| crate::util::is_within(&normalized, root))
        .map(|root| root.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Resolve an outbound `[FILE:...]` target within the roots that authorize it. The target is
/// canonicalized (a symlink cannot smuggle a file out) and must be inside a root BEFORE the
/// existence, directory and size checks, so a refusal discloses none of them. `Err` is why the
/// target was refused.
fn resolve_file_target(
    path: &Path,
    file_roots: &[PathBuf],
    upload_max: u64,
) -> Result<PathBuf, String> {
    let refuse = |reason: &str| Err(reason.to_string());

    // Nothing authorizes a local file without a workspace: a relative target never found a root.
    if file_roots.is_empty() {
        return refuse(media_target::NO_WORKSPACE_REASON);
    }

    // Roots are canonicalized too, so a workspace under a symlinked path (macOS `/tmp`, `/var`)
    // still contains the canonical target.
    let roots: Vec<PathBuf> = file_roots
        .iter()
        .map(|root| std::fs::canonicalize(root).unwrap_or_else(|_| root.clone()))
        .collect();

    // Containment precedes the directory and size checks: a path outside the workspace must not
    // have those disclosed.
    let Ok(canonical) = std::fs::canonicalize(path) else {
        return refuse(media_target::FILE_NOT_FOUND_REASON);
    };
    if !crate::tools::path::is_path_under_roots(&canonical, &roots) {
        return refuse(OUTSIDE_WORKSPACE_REASON);
    }

    let Ok(meta) = std::fs::metadata(&canonical) else {
        return refuse(media_target::FILE_NOT_FOUND_REASON);
    };
    if meta.is_dir() {
        return refuse(media_target::DIRECTORY_REASON);
    }
    if meta.len() > upload_max {
        return Err(UploadTooLarge {
            size: meta.len(),
            max: upload_max,
        }
        .to_string());
    }

    // The plain spelling, not the platform's verbatim form: this becomes `target`.
    Ok(crate::util::strip_verbatim_prefix(&canonical))
}

/// What one parsed `[KIND:...]` marker became.
enum MarkerOutcome {
    /// A bare word naming no file — prose explaining the syntax. Kept as written.
    Literal,
    /// Delivered from this target: an address, or a resolved absolute path. `repair` is set when
    /// the target was placed through a corrected directory.
    Attach {
        target: String,
        repair: Option<MarkerRepair>,
    },
    /// Undeliverable: the marker the conversation is told about.
    Refuse(RefusedMarker),
}

/// A marker delivered from a path it did not name: the target as written, and the file the
/// corrected directory names.
struct MarkerRepair {
    written: String,
    repaired: PathBuf,
}

/// The refusal one undeliverable marker produces: the marker as written (bounded, credential-free)
/// and the reason. `written` is the path the refusal record keeps, already bounded, so the note can
/// never name something other than what was recorded.
fn refused_marker(kind: MediaMarkerKind, written: Option<&str>, reason: &str) -> RefusedMarker {
    let marker = match written {
        Some(written) => format!("[{}:{written}]", kind.token()),
        None => format!("[{}]", kind.token()),
    };
    RefusedMarker {
        marker,
        reason: reason.to_string(),
    }
}

/// The refusal for an undeliverable local target: recorded in the Issues view and returned for the
/// conversation that wrote it. Only a local target reaches here — an address or a payload is
/// refused without a record.
fn refused(
    target: &str,
    kind: MediaMarkerKind,
    reason: &str,
    path: &Path,
    file_roots: &[PathBuf],
) -> MarkerOutcome {
    // One bounded, credential-free path is shared by the record and the note, so the note names
    // exactly what was recorded.
    let written = recorded_path(target);
    record_marker_refusal(written.as_deref(), reason, path, file_roots);
    MarkerOutcome::Refuse(refused_marker(kind, written.as_deref(), reason))
}

/// Decide one marker: the target is placed, then the kind's own gate decides whether it can be
/// delivered as what the marker claims.
fn marker_outcome(
    marker_kind: MediaMarkerKind,
    target: &str,
    file_roots: &[PathBuf],
    limits: TransferLimits,
) -> MarkerOutcome {
    let path = match media_target::resolve_marker_target(target, file_roots) {
        MarkerTarget::NotAReference => return MarkerOutcome::Literal,
        MarkerTarget::Inline => {
            // A data URI has no name worth quoting: name the kind instead.
            return MarkerOutcome::Refuse(refused_marker(
                marker_kind,
                None,
                media_target::INLINE_PAYLOAD_REASON,
            ));
        }
        MarkerTarget::Remote => {
            let written = recorded_path(target);
            return match media_target::remote_marker(marker_kind, target) {
                media_target::RemoteMarker::Refused(reason) => {
                    MarkerOutcome::Refuse(refused_marker(marker_kind, written.as_deref(), reason))
                }
                media_target::RemoteMarker::Deliverable => MarkerOutcome::Attach {
                    target: target.to_string(),
                    repair: None,
                },
            };
        }
        MarkerTarget::Local(path) => path,
    };

    // A marker whose only defect is a mistyped directory names a file that exists: the repair is
    // the same one the desktop's display layer applies, and it happens before the kind's gate so
    // that gate judges the file actually meant (and, for `[FILE:]`, the path actually delivered).
    // The repair travels with a delivered attachment; a refusal records the marker as written and
    // discloses nothing about what the repair found.
    let repaired = media_target::repaired_local_target(&path);
    let delivered = repaired.as_deref().unwrap_or(&path);
    let attach = |delivered_target: String| MarkerOutcome::Attach {
        target: delivered_target,
        repair: repaired.as_ref().map(|repaired| MarkerRepair {
            written: target.to_string(),
            repaired: repaired.clone(),
        }),
    };

    if marker_kind == MediaMarkerKind::File {
        return match resolve_file_target(delivered, file_roots, limits.upload) {
            Ok(canonical) => attach(canonical.to_string_lossy().into_owned()),
            Err(reason) => refused(target, marker_kind, &reason, delivered, file_roots),
        };
    }

    // Only `[FILE:...]` is workspace-confined; the kind's own gate still applies (an image marker
    // needs a raster signature). Nothing is decoded here — a local image goes out as a document, so
    // a large valid raster is delivered, not refused.
    if let Some(reason) = media_target::local_refusal(marker_kind, delivered) {
        return refused(target, marker_kind, reason, delivered, file_roots);
    }
    attach(delivered.to_string_lossy().into_owned())
}

/// Parse `[KIND:path]` markers, returning the text with them removed, the attachments, and the
/// markers that were refused. Targets resolve against `file_roots`; markers are independent, so one
/// refusal costs the user neither the others nor the surrounding text. A bare word naming no file
/// stays as prose. A file the message would deliver more than once — written twice, or once
/// mistyped and once correctly — is delivered once, and leaves one repair record.
fn parse_attachment_markers(
    message: &str,
    file_roots: &[PathBuf],
    limits: TransferLimits,
) -> (String, Vec<TelegramAttachment>, Vec<RefusedMarker>) {
    let mut attachments: Vec<TelegramAttachment> = Vec::new();
    let mut refusals: Vec<RefusedMarker> = Vec::new();
    let mut repairs: Vec<MarkerRepair> = Vec::new();

    let cleaned = MEDIA_MARKER_RE
        .replace_all(message, |caps: &regex::Captures| {
            let (marker_kind, path) = parse_media_marker(caps);
            match marker_outcome(marker_kind, path.trim(), file_roots, limits) {
                MarkerOutcome::Literal => caps.get_match().as_str().to_string(),
                MarkerOutcome::Attach { target, repair } => {
                    repairs.extend(repair);
                    if !already_attached(&attachments, &target) {
                        attachments.push(TelegramAttachment {
                            kind: TelegramAttachmentKind::from(marker_kind),
                            target,
                        });
                    }
                    String::new()
                }
                MarkerOutcome::Refuse(notice) => {
                    refusals.push(notice);
                    String::new()
                }
            }
        })
        .to_string();

    // One repair record per delivered file: the same file named under two kinds, or misspelled
    // twice, is one delivery. A repaired marker the gate refused never reaches this list, so that
    // marker is not what records a repair.
    let mut recorded: HashSet<PathBuf> = HashSet::new();
    for repair in repairs {
        let key = media_target::file_key(&repair.repaired);
        let delivered = attachments
            .iter()
            .any(|attachment| media_target::file_key(Path::new(&attachment.target)) == key);
        if delivered && recorded.insert(key) {
            record_marker_repair(&repair.written, &repair.repaired, file_roots);
        }
    }

    (cleaned.trim().to_string(), attachments, refusals)
}

/// Whether this message already delivers `target`. The comparison is on the file itself, not on
/// the text: `[FILE:...]` carries a canonical path while the media kinds carry the resolved one, so
/// a correct marker and its repaired twin would otherwise read as two files.
fn already_attached(attachments: &[TelegramAttachment], target: &str) -> bool {
    let key = media_target::file_key(Path::new(target));
    attachments
        .iter()
        .any(|attachment| media_target::file_key(Path::new(&attachment.target)) == key)
}

pub(crate) const CLOUD_API_BASE: &str = "https://api.telegram.org";

/// The base in force: the cloud, or the local server's endpoint. `None` while the local mode has
/// none — no listener may run, the only other base being the cloud.
fn active_base() -> Option<String> {
    match crate::channels::telegram_local::serving_mode() {
        crate::channels::telegram_local::Mode::Cloud => Some(CLOUD_API_BASE.to_string()),
        crate::channels::telegram_local::Mode::Local => {
            crate::channels::telegram_local::local_endpoint()
        }
    }
}

/// The wait between two checks for a base, before it doubles.
const TRANSPORT_WAIT_FIRST: Duration = Duration::from_millis(100);

/// What that wait doubles towards: an absent server must not cost ten checks a second.
const TRANSPORT_WAIT_CAP: Duration = Duration::from_secs(5);

/// Wait until the transport has a base, for as long as that takes. `false` means shutdown ended the
/// wait; a transport not up yet is no fault, so nothing is sent.
#[must_use]
pub(crate) async fn wait_for_transport() -> bool {
    wait_until(|| active_base().is_some()).await
}

/// The wait itself over any predicate: it re-asks rather than waiting for a signal, because the
/// base moves with the mode as well as with the endpoint.
async fn wait_until(ready: impl Fn() -> bool) -> bool {
    let mut wait = TRANSPORT_WAIT_FIRST;
    while !ready() {
        if !crate::shutdown::sleep_or_shutdown(wait).await {
            return false;
        }
        wait = wait.saturating_mul(2).min(TRANSPORT_WAIT_CAP);
    }
    true
}

/// What one transfer may be in the mode in force: the cloud caps a bot at 20 MB down and 50 MB up,
/// the local server removes the download cap and accepts 2000 MB up. Every refusal quotes the limit
/// that actually applies.
#[derive(Clone, Copy, Debug)]
struct TransferLimits {
    /// The largest file the bot may take in, `None` when uncapped.
    download: Option<u64>,
    upload: u64,
    /// The bound on one transfer request, `None` when the transport sets none.
    request: Option<Duration>,
}

/// The total bound on one local transfer: no real file fails on it, a hung socket ends.
const LOCAL_TRANSFER_BOUND: Duration = Duration::from_hours(24);

impl TransferLimits {
    const CLOUD: Self = Self {
        download: Some(TELEGRAM_MAX_FILE_DOWNLOAD_BYTES),
        upload: TELEGRAM_CLOUD_MAX_UPLOAD_BYTES,
        request: Some(MEDIA_TRANSFER_TIMEOUT),
    };

    /// The local server's limits: no download cap, 2000 MB up, no request bound.
    const LOCAL: Self = Self {
        download: None,
        upload: TELEGRAM_LOCAL_MAX_UPLOAD_BYTES,
        request: None,
    };

    /// The transfer bound: the transport's own, else the day-long client bound.
    fn request_bound(self) -> Duration {
        self.request.unwrap_or(LOCAL_TRANSFER_BOUND)
    }
}

/// The client the local server's transfers run on, built once.
fn local_transfer_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| crate::util::http::build_http_client(LOCAL_TRANSFER_BOUND))
}

/// The client a fetched address runs on: no redirect of its own (the shared SSRF guard screens each
/// hop) and a day-long bound that ends a hung socket.
fn fetch_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| crate::util::http::build_no_redirect_http_client(LOCAL_TRANSFER_BOUND))
}

fn bot_api_url(api_base: &str, token: &str, method: &str) -> String {
    format!("{api_base}/bot{token}/{method}")
}

/// What a refused `getMe` starts with when the platform rejected the token, as opposed to failing
/// to answer — one spelling, so the two are tellable.
pub(crate) const REJECTED_TOKEN_PREFIX: &str = "Telegram rejected the bot token — ";

/// The `getMe` answer for `token`, once the platform has confirmed it: validating a token and
/// naming its account are one call, and the intake keys its record on the account, so a rotated
/// token for the same account still recovers.
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
        anyhow::bail!("{REJECTED_TOKEN_PREFIX}{desc}");
    }
    Ok(data)
}

/// Cloud download cap; the enrichment gate reuses it as its processing bound.
pub(crate) const TELEGRAM_MAX_FILE_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;

const TELEGRAM_CLOUD_MAX_UPLOAD_BYTES: u64 = 50 * 1024 * 1024;

const TELEGRAM_LOCAL_MAX_UPLOAD_BYTES: u64 = 2000 * 1024 * 1024;

/// A byte cap as whole MB, derived from the cap so notices cannot drift.
fn mb(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

/// User-facing reason when a declared size exceeds the inbound limit, `None` when it fits.
#[must_use]
fn declared_size_refusal(limits: TransferLimits, size: u64) -> Option<String> {
    let download = limits.download?;
    (size > download).then(|| telegram_download_limit_reason(download))
}

/// User-facing reason when a declared size exceeds the cloud's bot-download limit.
fn telegram_download_limit_reason(download: u64) -> String {
    format!(
        "Telegram only lets the bot download files up to {} MB",
        mb(download)
    )
}

/// Reason when Telegram declared no size but the download still failed; naming no figure keeps it
/// independent of [`TELEGRAM_MAX_FILE_DOWNLOAD_BYTES`].
fn transfer_refused_reason(limits: TransferLimits) -> &'static str {
    if limits.download.is_some() {
        "Telegram refused the transfer (it may exceed what the bot can download)"
    } else {
        "Telegram refused the transfer"
    }
}

/// Reason when the download hit [`MEDIA_TRANSFER_TIMEOUT`]: stopped on this side.
const ATTACHMENT_DOWNLOAD_TIMEOUT_REASON: &str = "the download took too long and was stopped";

/// Reason when the connection failed before any answer — this side's failure too.
const ATTACHMENT_DOWNLOAD_FAILED_REASON: &str = "the connection to Telegram failed";

/// The local server did not hand over a file it said it had (it serves no route).
const LOCAL_FILE_MISSING_REASON: &str =
    "the local Bot API server did not hand the file over — it is not on this machine";

/// The local server named a file outside the directory it was given.
const LOCAL_FILE_OUTSIDE_REASON: &str =
    "the local Bot API server handed over a file that is not its own";

/// How long a hand-over may take in total: a bound on the whole wait, not on a pause, and only a
/// hand-over that never reaches the declared size is refused.
const LOCAL_HANDOVER_WAIT: Duration = Duration::from_mins(30);

/// How long the file may take to appear: a broken answer, not a slow transfer, and waiting
/// [`LOCAL_HANDOVER_WAIT`] for it would hold the batch half an hour.
const LOCAL_HANDOVER_APPEAR: Duration = Duration::from_mins(1);

/// Reason when Telegram returned no download path — an API or network failure.
const TELEGRAM_LOOKUP_REFUSED_REASON: &str = "Telegram could not provide the file";

/// The agent-facing outcome when the transport never handed the file over.
const NOT_RECEIVED: &str = "not received";

/// The agent-facing outcome when the file arrived but could not be kept.
const NOT_STORED: &str = "could not be stored";

/// Reason when the staging directory could not be created — a local failure, not a transfer
/// refusal.
const ATTACHMENT_DIR_REFUSED_REASON: &str =
    "the local temp directory for the file could not be created";

/// Reason when a downloaded attachment could not be written — a local failure.
const ATTACHMENT_SAVE_REFUSED_REASON: &str = "the file could not be saved locally";

/// An outbound file over the upload limit in force. Its text names the limit and nothing else, so
/// it is safe to render into the chat.
#[derive(Debug)]
struct UploadTooLarge {
    size: u64,
    max: u64,
}

fn upload_too_large(err: &anyhow::Error) -> Option<&UploadTooLarge> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<UploadTooLarge>())
}

impl std::fmt::Display for UploadTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "it is {} MB, larger than the {} MB limit.",
            self.size.div_ceil(1024 * 1024),
            mb(self.max)
        )
    }
}

impl std::error::Error for UploadTooLarge {}

/// Reason when the transport failed to deliver a local file that passed the size check.
/// Deliberately static: the transport error embeds the request URL, which carries the token, so it
/// is logged but never rendered into the chat.
const ATTACHMENT_UPLOAD_FAILED_REASON: &str = "the upload failed.";

/// Reason when an addressed attachment never arrived. Static for the same reason as
/// [`ATTACHMENT_UPLOAD_FAILED_REASON`]: the error may carry the URL.
const ATTACHMENT_FETCH_FAILED_REASON: &str = "it could not be fetched or sent.";

/// The local server's hand-over produced no usable file: one variant per fault, so the refusal
/// reaches the owner as what happened — missing, outside, short — rather than one sentence for all
/// of them.
#[derive(Debug)]
enum LocalHandOver {
    /// It named no path on this machine, or one that is not there — an unresolvable hand-over root
    /// lands here too.
    Missing,
    Outside,
    Incomplete {
        got: u64,
        expected: u64,
    },
}

impl LocalHandOver {
    fn reason(&self) -> String {
        match self {
            Self::Missing => LOCAL_FILE_MISSING_REASON.to_string(),
            Self::Outside => LOCAL_FILE_OUTSIDE_REASON.to_string(),
            Self::Incomplete { got, expected } => local_file_incomplete_reason(*got, *expected),
        }
    }
}

impl std::fmt::Display for LocalHandOver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason())
    }
}

impl std::error::Error for LocalHandOver {}

/// Reason when the hand-over stored fewer bytes than declared: both counts are in the sentence,
/// which is why the text is built.
fn local_file_incomplete_reason(got: u64, expected: u64) -> String {
    format!(
        "the local Bot API server handed over {got} of {expected} bytes — the transfer did not finish"
    )
}

/// Wait for a handed-over path to reach the size Telegram declared. Completeness is decided by
/// size, never by a pause: a size that stops short is a write stall and the wait goes on until
/// [`LOCAL_HANDOVER_WAIT`] runs out. With no declared size, a size that stops changing is all the
/// evidence there is; a file that never appears at all is given up on after
/// [`LOCAL_HANDOVER_APPEAR`].
async fn wait_for_hand_over(path: &Path, expected: Option<u64>) -> anyhow::Result<()> {
    let started = tokio::time::Instant::now();
    let mut previous: Option<u64> = None;
    loop {
        let size = tokio::fs::metadata(path)
            .await
            .ok()
            .filter(std::fs::Metadata::is_file)
            .map(|meta| meta.len());
        match (size, expected) {
            // A declared size reached (or exceeded) means the copy is whole.
            (Some(size), Some(expected)) if size >= expected => return Ok(()),
            // Nothing to compare against: a size that no longer changes is all the evidence.
            (Some(size), None) if previous == Some(size) => return Ok(()),
            _ => {}
        }
        let bound = if size.is_some() {
            LOCAL_HANDOVER_WAIT
        } else {
            LOCAL_HANDOVER_APPEAR
        };
        if started.elapsed() >= bound {
            return Err(match (size, expected) {
                // There but short is an unfinished transfer; never there names nothing.
                (Some(size), Some(expected)) => LocalHandOver::Incomplete {
                    got: size,
                    expected,
                },
                _ => LocalHandOver::Missing,
            }
            .into());
        }
        previous = size;
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// The user-facing reason for a failed inbound download: a write that failed here is storage, only
/// what Telegram answered — an error status, a body over the limit — is a refusal, and a transfer
/// that got no answer at all is this side's failure. The local hand-over's own failures come after
/// those ([`LocalHandOver`]); anything unrecognised is still the transport's.
fn download_failure_reason(error: &anyhow::Error, limits: TransferLimits) -> String {
    // The transport first: a `reqwest` error wraps the socket's own io error, so testing io first
    // would report a broken connection as a local write failure.
    if let Some(e) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
    {
        let reason = if e.is_timeout() {
            ATTACHMENT_DOWNLOAD_TIMEOUT_REASON
        } else {
            ATTACHMENT_DOWNLOAD_FAILED_REASON
        };
        return reason.to_string();
    }
    if error
        .chain()
        .any(|cause| cause.downcast_ref::<std::io::Error>().is_some())
    {
        return ATTACHMENT_SAVE_REFUSED_REASON.to_string();
    }
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<LocalHandOver>())
        .map_or_else(
            || transfer_refused_reason(limits).to_string(),
            LocalHandOver::reason,
        )
}

/// Change-detection state for a chat's per-user command menu refresh.
enum ChatMenuState {
    /// Last successfully registered command payload (`None` = never registered).
    Registered(Option<String>),
    /// A refresh is already in flight — skip until it completes.
    InFlight,
}

pub struct TelegramChannel {
    bot_token: String,
    /// Shared HTTP client, reusing connections across API calls.
    http_client: reqwest::Client,

    /// Cancelling this stops only this channel's listener, not the whole application.
    cancel: std::sync::Arc<tokio_util::sync::CancellationToken>,

    /// The Bot API base this listener calls: the cloud, or the local server's endpoint.
    api_base: String,
    /// Whether that base is the local server's — its limits, hand-over and unbounded transfers
    /// differ. Not `api_base != CLOUD_API_BASE`, so a channel pointed at an endpoint of its own in
    /// a check is still a cloud channel.
    local: bool,
    /// The intake's durable store, resolved on the first listener run.
    intake_store: std::sync::OnceLock<TelegramIntakeStore>,
    /// The claim this listener's rows are recorded under: minted per listener, so the one that
    /// replaces this takes over what it left held.
    intake_claim: std::sync::Arc<str>,
    /// Whether a listener frame is running, and the signal for when it stops, which a replacement
    /// waits on. Several waiters can be parked on one stop.
    listener_running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    listener_stopped: std::sync::Arc<tokio::sync::Notify>,

    /// The messages this listener is preparing now, held until the burst collector takes them
    /// (`telegram_group::Preparations`).
    preparations: std::sync::Arc<crate::channels::telegram_group::Preparations>,

    /// Per-chat command-menu state: change detection plus in-flight coalescing, so a message flood
    /// does not trip Telegram's rate limiting. Fail-open.
    menu_cache: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, ChatMenuState>>>,
}

/// `(chat_id, reply_target)` from a message sub-object.
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

fn set_thread_id_on_json(body: &mut serde_json::Value, thread_id: Option<&str>) {
    if let Some(tid) = thread_id {
        body["message_thread_id"] = serde_json::Value::String(tid.to_string());
    }
}

/// Split a recipient into `(chat_id, thread_id)`: `"chat"` or `"chat:thread"`.
fn parse_recipient(recipient: &str) -> (&str, Option<&str>) {
    match recipient.split_once(':') {
        Some((chat, thread)) => (chat, Some(thread)),
        None => (recipient, None),
    }
}

/// Authorize an event by the identity of the person it came from, and derive the reply context it
/// arrived in.
///
/// `sender_source` is where Telegram presents the sender — the message, or the `callback_query` for
/// a press, so a keyboard on a chat-attributed message still authorizes the presser; `chat_source`
/// supplies only the delivery address. `Ok(None)` for a sender that is not a person or matches no
/// binding, else `(canonical_user, chat_id, reply_target)` — `"123456"` or `"123456:789"` for
/// threads. `Err` is the bindings read failing, which is no decision and must not become a refusal:
/// that would drop the event for good.
async fn resolve_authorized_sender(
    sender_source: &serde_json::Value,
    chat_source: &serde_json::Value,
) -> anyhow::Result<Option<(String, String, String)>> {
    // Authorization keys on the person: a number first, which survives a rename or a dropped
    // nickname, then the nickname. The chat only supplies the address.
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

/// Parse a Markdown link `[label](url)` at `i`, returning the label, the URL and the offset past
/// `)`; `None` for a non-http(s) link or a label naming a media marker.
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

/// True when a bracketed label names a media marker rather than prose: a known [`MediaMarkerKind`]
/// in any casing. Such a marker followed by a parenthesised URL is delivery scaffolding — an anchor
/// would fabricate a link out of it.
fn is_marker_label(label: &str) -> bool {
    let Some((kind, _)) = label.split_once(':') else {
        return false;
    };
    MediaMarkerKind::iter().any(|marker| marker.token().eq_ignore_ascii_case(kind))
}

/// Prefix of the anchor `render_inline` emits, shared with [`render_if_link`].
const ANCHOR_PREFIX: &str = "<a href=\"";

/// The rendering of `text` when it carries a link, else `None`: the `Some` arm makes an enclosing
/// span give way to the top-level anchor.
fn render_if_link(text: &str) -> Option<String> {
    let rendered = render_inline(text);
    rendered.contains(ANCHOR_PREFIX).then_some(rendered)
}

/// If `text[i..]` starts with `delim`, find the closing one, render the content and wrap it in
/// `<tag>`; advances `i` and returns `true`. `false` when `delim` is missing or the content is
/// empty, so `****` is no span. With `linkify` a span whose content renders a link yields to it.
/// Callers guard a single-character delimiter that repeats.
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

/// Render one line of Markdown as Telegram HTML.
fn render_inline(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if try_format_inline(text, &mut i, &mut out, "**", "b", true) {
            continue;
        }
        if try_format_inline(text, &mut i, &mut out, "__", "b", true) {
            continue;
        }
        // Italic: *text* — guard against matching the second `*` of `**`
        if (i == 0 || bytes[i - 1] != b'*')
            && try_format_inline(text, &mut i, &mut out, "*", "i", true)
        {
            continue;
        }
        // Inline code: guard the second backtick of a fence; never linkify.
        if (i == 0 || bytes[i - 1] != b'`')
            && try_format_inline(text, &mut i, &mut out, "`", "code", false)
        {
            continue;
        }
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
        if try_format_inline(text, &mut i, &mut out, "~~", "s", true) {
            continue;
        }
        let ch = text[i..].chars().next().unwrap();
        push_escaped(ch, &mut out);
        i += ch.len_utf8();
    }
    out
}

/// Convert the Markdown subset the model emits to Telegram's HTML parse_mode: headings, bold,
/// italic, code, strikethrough, links and fenced blocks, plus `<blockquote>` pass-through. Fences
/// are read first, so their content is never interpreted; a span whose content renders a link
/// yields to the top-level anchor.
fn markdown_to_telegram_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code_block = false;
    let mut code_buf = String::new();

    for line in text.split('\n') {
        let trimmed = line.trim_start();

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

        if trimmed == "<blockquote>" || trimmed == "</blockquote>" {
            out.push_str(trimmed);
            out.push('\n');
            continue;
        }

        let stripped = line.trim_start_matches('#');
        let header_level = line.len() - stripped.len();
        if header_level > 0 && stripped.starts_with(' ') {
            let title = stripped.trim();
            // A heading yields to a link in its content like any other formatting span.
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

/// Strip all HTML tags, leaving the text: the plain-text fallback. Text from an unterminated `<` to
/// the end is dropped.
fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(open) = s[i..].find('<').map(|o| i + o) {
        out.push_str(&s[i..open]);
        let Some(gt) = tag_end(s, open) else {
            return out;
        };
        i = gt + 1;
    }
    out.push_str(&s[i..]);
    out
}

/// Decode HTML entities (`&#39;`) before markdown→HTML, so they are not escaped twice.
fn to_telegram_html(text: &str) -> String {
    markdown_to_telegram_html(&decode_html_entities(text))
}

/// Prefix of the `[Forwarded from …]` attribution. The fact behind it is the message's `forwarded`
/// flag — never this text.
const FORWARD_ATTRIBUTION_PREFIX: &str = "[Forwarded from ";

/// What `TelegramChannel::forward` makes of a message's forward fields.
struct Forward {
    /// Whether any forward field is set: an origin the product cannot name is still forwarded.
    forwarded: bool,
    prefix: Option<String>,
}

type ForwardLabel = fn(&serde_json::Value) -> Option<String>;

/// The older forward fields, in the order the attribution reads them. The flag and the label both
/// come from this list: a field can be present without a nameable target.
const LEGACY_FORWARD_FIELDS: [(&str, ForwardLabel); 3] = [
    // A channel or a group: its title, with the historical "channel: " shape.
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
    ///
    /// # Panics
    ///
    /// When `reqwest::Client::build()` fails — with no rustls provider installed;
    /// `util::http::install_ring_provider` installs ring (TLS is rustls/ring).
    #[must_use]
    fn new_with(bot_token: String, api_base: String, local: bool) -> Self {
        Self {
            bot_token,
            api_base,
            local,
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

    /// The intake's store for this process.
    fn intake_store(&self) -> TelegramIntakeStore {
        self.intake_store
            .get_or_init(|| crate::channels::telegram_intake::store().clone())
            .clone()
    }

    /// A channel pointed at an endpoint of its own and recording into `store` instead of the
    /// process's database, so the intake's checks drive a listener without the running service. Its
    /// claim is minted as for a product listener.
    #[cfg(test)]
    #[must_use]
    fn for_test(bot_token: String, api_base: String, store: TelegramIntakeStore) -> Self {
        Self {
            intake_store: std::sync::OnceLock::from(store),
            ..Self::new_with(bot_token, api_base, false)
        }
    }

    /// The channel the configured mode serves, or `None` while the local mode has no endpoint —
    /// then no listener may run at all.
    #[must_use]
    fn from_config(bot_token: String) -> Option<Self> {
        let local = crate::channels::telegram_local::serving_mode()
            == crate::channels::telegram_local::Mode::Local;
        Some(Self::new_with(bot_token, active_base()?, local))
    }

    /// The client one transfer runs on: the shared one, or the local mode's.
    fn transfer_client(&self) -> &reqwest::Client {
        if self.local {
            local_transfer_client()
        } else {
            &self.http_client
        }
    }

    /// `request` under the transport's own bound: the cloud's, or none in local mode, where the
    /// client's day-long bound ends a hung socket.
    fn with_transfer_bound(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.limits().request {
            Some(timeout) => request.timeout(timeout),
            None => request,
        }
    }

    fn limits(&self) -> TransferLimits {
        if self.local {
            TransferLimits::LOCAL
        } else {
            TransferLimits::CLOUD
        }
    }

    fn running(&self) -> bool {
        self.listener_running
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Wait, briefly, for this channel's listener frame to end — what a replacement asks before it
    /// takes over what the frame left behind ([`stop_listeners`]). Bounded, so a frame that will
    /// not end cannot hold it up for good: past the bound the replacement may take over events the
    /// open frame is about to hand on, and see them twice.
    async fn wait_stopped(&self) {
        // Enrolled before the check: `notify_waiters` wakes only registered waiters, so a stop
        // landing in between would be missed and the wait would sit out the whole bound.
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

    /// Wait out a pause in the intake's work, ending it early when this listener is being stopped,
    /// so a failing or contended store cannot hold a token change up for it.
    async fn intake_pause(&self, duration: Duration) {
        tokio::select! {
            () = self.cancel.cancelled() => {}
            _ = crate::shutdown::sleep_or_shutdown_or_drain(duration) => {}
        }
    }

    /// Dismiss the loading spinner, with a toast when `text` is given. Errors are logged, so a user
    /// is never stuck on an infinite spinner.
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

    /// Parse a `callback_query` into a `ChannelMessage`; its data is the content.
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

        // Auth is clicker-based (cq.from), never msg author; chat_id/message_id identify the
        // pressed keyboard's message so handlers can edit it in place.
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

    /// Sender info, the allow check, chat/message/thread ids and the reply target. `Ok(None)` when
    /// the sender is not allowed; `Err` is the binding read failing, which is no decision and must
    /// not become a refusal by the caller.
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

    /// Signal this channel's listener to stop, leaving the global token and others alone.
    fn cancel_own(&self) {
        self.cancel.cancel();
    }

    /// Register the bot's global commands via `setMyCommands`; per-user commands go per chat
    /// ([`Self::spawn_menu_refresh`]). Failure only warns and never blocks the caller.
    pub async fn set_my_commands(&self) {
        let body = serde_json::json!({
            "commands": [
                {"command": "clear", "description": CLEAR_COMMAND_DESC},
            ]
        });
        post_set_my_commands(self.http_client(), &self.api_url("setMyCommands"), &body).await;
    }

    /// Spawn a per-user command-menu refresh for a chat, after every outbound message.
    /// Fire-and-forget and fail-open. Everything but a cheap in-flight check runs in the task: the
    /// chat's bound user (first match wins in a group), their command payload, and the scoped
    /// registration when it differs from the last successful one.
    fn spawn_menu_refresh(&self, chat_id: &str) {
        // Cheap inline coalescing: skip spawning when a refresh for this chat is in flight.
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

    /// Validate a bot token by asking the Bot API the current mode serves for it. There is no
    /// fallback to the cloud: a token-bearing cloud request logs the bot back in there and splits
    /// its updates. With no published endpoint nothing is checked — a save must stay possible, the
    /// token field may be what fixes the bot.
    pub async fn validate_token(token: &str) -> anyhow::Result<()> {
        if token.trim().is_empty() {
            anyhow::bail!("Telegram bot token is empty");
        }
        let Some(base) = active_base() else {
            tracing::info!(
                "Telegram: the local server has no endpoint yet, so the token is kept unchecked"
            );
            return Ok(());
        };
        let client = crate::util::http::build_http_client(std::time::Duration::from_secs(10));
        get_me(&client, &base, token).await.map(|_| ())
    }

    /// The bot's own account id at `api_base` — the same call as validation, read for its answer.
    pub(crate) async fn probe_bot(api_base: &str, token: &str) -> anyhow::Result<i64> {
        let client = crate::util::http::build_http_client(std::time::Duration::from_secs(10));
        let data = get_me(&client, api_base, token).await?;
        data["result"]["id"]
            .as_i64()
            .ok_or_else(|| anyhow::anyhow!("the Bot API named no bot account"))
    }

    /// Take the bot out of a Bot API server (`logOut`), which lets another serve it. A server that
    /// already considers it logged out is the state asked for, not a failure: the switch is
    /// resumable.
    pub(crate) async fn log_out(api_base: &str, token: &str) -> anyhow::Result<()> {
        let client = crate::util::http::build_http_client(std::time::Duration::from_secs(30));
        let response = client
            .post(bot_api_url(api_base, token, "logOut"))
            .send()
            .await
            .context("the Bot API could not be reached")?;
        let data: serde_json::Value = response.json().await.unwrap_or_default();
        if data["ok"].as_bool() == Some(true) {
            return Ok(());
        }
        let description = data["description"].as_str().unwrap_or("no reason given");
        // Only "already out" counts: a bare `401` is a token the server does not know, and taking
        // it for a completed logOut would record leaving a server the bot is still logged into.
        if description.to_ascii_lowercase().contains("logged out") {
            return Ok(());
        }
        anyhow::bail!("{description}")
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

    /// The file path for a file ID, with the size the server reports when it does.
    async fn get_file_path(&self, file_id: &str) -> anyhow::Result<(String, Option<u64>)> {
        let url = self.api_url("getFile");
        // A local server answers only once it has the file, so the request carries the mode's own
        // bound: one minute would cut a large fetch off before the transfer was asked for.
        let request = self.with_transfer_bound(
            self.transfer_client()
                .get(&url)
                .query(&[("file_id", file_id)]),
        );
        let resp = request
            .send()
            .await
            .context("Failed to call Telegram getFile")?;

        let data: serde_json::Value = resp.json().await?;
        let result = data
            .get("result")
            .context("Telegram getFile: missing result in response")?;
        let path = result
            .get("file_path")
            .and_then(serde_json::Value::as_str)
            .map(String::from)
            .context("Telegram getFile: missing file_path in response")?;
        // The size is what makes a hand-over with no declared size checkable.
        let size = result.get("file_size").and_then(serde_json::Value::as_u64);
        Ok((path, size))
    }

    /// Fetch one file into `dest`, streaming on both paths so a multi-gigabyte file is never held
    /// in memory. A local server hands over an absolute path instead of serving bytes: the copy is
    /// confined to the directory the product gave it BEFORE anything else is done, so a bad answer
    /// can copy and delete nothing else and a file outside that tree is refused at once rather than
    /// waited for and called truncated. `expected` is the size Telegram declared, checked twice on
    /// the local path — while waiting and on what was stored.
    async fn download_file(
        &self,
        file_path: &str,
        dest: &Path,
        expected: Option<u64>,
    ) -> anyhow::Result<()> {
        if self.local {
            // A local server has no `/file/…` route: it hands over a path. Anything else is a
            // server failure, reported as that and not as a size refusal.
            let Some(root) = crate::channels::telegram_local::incoming_files_dir() else {
                return Err(LocalHandOver::Missing.into());
            };
            let source = Path::new(file_path);
            if !source.is_absolute() {
                return Err(LocalHandOver::Missing.into());
            }
            // Containment first, on the path as named, before any wait: both the configured root
            // and its canonical form count (the directory may sit under a symlinked path), and the
            // check is lexical because the path may not exist yet.
            let mut roots = vec![root.clone()];
            if let Ok(canonical) = tokio::fs::canonicalize(&root).await {
                roots.push(canonical);
            }
            if !crate::tools::path::is_path_under_roots(source, &roots) {
                return Err(LocalHandOver::Outside.into());
            }
            wait_for_hand_over(source, expected).await?;
            // And again on the file that actually appeared: a link inside the tree points anywhere.
            let source = tokio::fs::canonicalize(source)
                .await
                .map_err(|_| anyhow::Error::from(LocalHandOver::Missing))?;
            if !crate::tools::path::is_path_under_roots(&source, &roots) {
                return Err(LocalHandOver::Outside.into());
            }
            let copied = tokio::fs::copy(&source, dest).await?;
            // What was stored must be whole: a short copy would leave the agent a truncated file
            // and nothing that could ever complete it.
            if let Some(expected) = expected
                && copied < expected
            {
                let _ = tokio::fs::remove_file(dest).await;
                return Err(LocalHandOver::Incomplete {
                    got: copied,
                    expected,
                }
                .into());
            }
            // The server's copy stays: its record still names it as downloaded, so dropping it
            // would make a second `getFile` wait for a file that never appears. The age sweep
            // bounds the disk.
            return Ok(());
        }

        let url = format!("{}/file/bot{}/{file_path}", self.api_base, self.bot_token);
        let request = self.with_transfer_bound(self.transfer_client().get(&url));
        let resp = request
            .send()
            .await
            .context("Failed to download Telegram file")?;

        if !resp.status().is_success() {
            anyhow::bail!("Telegram file download failed: {}", resp.status());
        }

        // The cloud reports no size for a large file, so the bound is enforced on the stream too.
        stream_capped(resp, dest, self.limits().download, |_, max| {
            anyhow::anyhow!("{}", telegram_download_limit_reason(max))
        })
        .await
    }

    /// The attachment metadata of a message: `document`, `photo` (the last array element, the
    /// highest resolution), `video`, `video_note`, `animation`, `audio` or `voice`; `None` for
    /// text-only and other kinds. `document` is checked first because an animation sets both keys,
    /// `photo` is exclusive with the rest, so it is checked last.
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
        if let Some(photos) = message.get("photo").and_then(serde_json::Value::as_array) {
            let best = photos.last()?;
            return Self::build_attachment(best, message, IncomingAttachmentKind::Photo);
        }
        None
    }

    /// Build an [`IncomingAttachment`] from a pre-resolved sub-object of `message`, which supplies
    /// the caption.
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
        // The caption is the text the user typed, so the urgent mark is read and consumed here.
        let (caption, urgent) = match message.get("caption").and_then(serde_json::Value::as_str) {
            Some(caption) => {
                let (caption, urgent) = crate::channels::take_urgent(caption);
                (Some(caption.into_owned()), urgent)
            }
            None => (None, false),
        };
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
            urgent,
        })
    }

    /// Parse an update as an attachment message: download it into this message's own staging
    /// directory (recorded in the returned message's `attachment_dirs`) and return it. `Ok(None)`
    /// when it is no attachment or the sender is unauthorized — `Err` is the binding read failing,
    /// which the caller keeps the event held for. An attachment the bot cannot retrieve still
    /// yields a message, so the caption reaches the agent with the failure in it.
    ///
    /// A recorded voice message and an `audio` sound file both take the ordinary path. The whole
    /// preparation, fetch included, is marked in [`TelegramChannel::preparations`], so the
    /// collector holds that sender's group in that chat until the part is handed over.
    async fn try_parse_attachment_message(
        &self,
        update: &serde_json::Value,
    ) -> anyhow::Result<Option<ChannelMessage>> {
        let Some(message) = update.get("message") else {
            return Ok(None);
        };

        // Authorization first: an unauthorized sender is ignored silently, before any work.
        let Some(ctx) = self.extract_message_context(message).await? else {
            return Ok(None);
        };

        let Some(attachment) = Self::parse_attachment_metadata(message) else {
            return Ok(None);
        };

        // Held for the whole preparation, the fetch included, so the group this sender has open in
        // this chat waits for the part this call produces.
        let _preparing = self.preparations.begin(&ctx.user_name, &ctx.reply_target);

        // Declared-size gate: a file the mode in force cannot carry is refused before a transfer is
        // spent on it, in the words of the limit that applies here.
        if let Some(size) = attachment.file_size
            && let Some(reason) = declared_size_refusal(self.limits(), size)
        {
            tracing::info!("Rejecting attachment: declared size {size} bytes ({reason})");
            return Ok(Some(
                self.reject_attachment(ctx, &attachment, None, NOT_RECEIVED, &reason)
                    .await,
            ));
        }

        // The message does not always carry a size; the server's answer does, and that is what the
        // local hand-over is checked against. A transfer failure is the other way a limit shows up
        // — surface it instead of dropping the update.
        let (tg_file_path, server_size) = match self.get_file_path(&attachment.file_id).await {
            Ok(answer) => answer,
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
        let expected = attachment.file_size.or(server_size);
        let remote_ext = Path::new(&tg_file_path)
            .extension()
            .and_then(|e| e.to_str());

        // Created up front because the transfer streams into it; a refused transfer takes it with
        // it.
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
        if let Err(e) = self
            .download_file(&tg_file_path, &local_path, expected)
            .await
        {
            tracing::warn!("Failed to download attachment: {e}");
            let _ = tokio::fs::remove_dir_all(&save_dir).await;
            let reason = download_failure_reason(&e, self.limits());
            let outcome = if reason == ATTACHMENT_SAVE_REFUSED_REASON {
                NOT_STORED
            } else {
                NOT_RECEIVED
            };
            return Ok(Some(
                self.reject_attachment(ctx, &attachment, remote_ext, outcome, &reason)
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

        let mut msg = ctx.into_attachment_message(content, staging_dir);
        msg.urgent = attachment.urgent;
        Ok(Some(msg))
    }

    /// Tell the user an attachment could not be taken and return the caption-preserving message, so
    /// the file still produces a reply and an agent turn. `outcome` is [`NOT_RECEIVED`] or
    /// [`NOT_STORED`] and decides whether the note says it arrived.
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
        let mut msg = self.refuse_with_notice(ctx, &notice, content).await;
        msg.urgent = attachment.urgent;
        msg
    }

    /// Send the user-visible refusal (best-effort) and return the message carrying `content`.
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

    /// What Telegram says about a message being forwarded content: the modern `forward_origin`
    /// first, the older fields only where it names nothing, so an update carrying both yields ONE
    /// attribution. A kind this product cannot label falls back rather than dropping it.
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

    /// The label a `forward_origin` names, when this product knows the kind — `None` lets the older
    /// fields attribute the message instead.
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
                // MessageOriginChat / Channel: the title, named as a channel when the origin says
                // so.
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

        // Strip the `@BotUsername` suffix Telegram appends in group chats (`/new@MyBot`).
        let text = if text.starts_with('/') {
            text.split('@').next().unwrap_or(text)
        } else {
            text
        };

        // Read on the typed text, before the forward attribution is composed onto content.
        let (text, urgent) = crate::channels::take_urgent(text);
        let mut msg = ctx.into_channel_message(text.into_owned(), None);
        msg.urgent = urgent;
        Ok(Some(msg))
    }

    /// POST a JSON body to an API method: the response on 2xx, `(status, body)` otherwise.
    /// Transport errors map to BAD_GATEWAY, no HTTP response existing.
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
            // No HTTP response exists for a transport error, so BAD_GATEWAY is the sentinel.
            .map_err(|e| (reqwest::StatusCode::BAD_GATEWAY, e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            Ok(resp)
        } else {
            let err_body = crate::util::http::read_error_body(resp, err_label).await;
            Err((status, err_body))
        }
    }

    /// Send one text message, returning the message id (the 2xx body may omit it — the message was
    /// still delivered), or the status and body on failure.
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
        // Telegram attaches a web-page preview to the first URL unless disabled: Bot API 7.0
        // replaced `disable_web_page_preview` with `link_preview_options.is_disabled`.
        body["link_preview_options"] = serde_json::json!({ "is_disabled": true });

        let resp = self
            .post_telegram_json("sendMessage", body, "sendMessage error")
            .await?;

        let status = resp.status();
        // A 2xx body without a parseable message_id is still a delivered message: the shared send
        // path only checks the status.
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

    /// Send one text message; status and body on failure, `Ok(())` on success.
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

    /// Replace a sent message's inline keyboard in place. `reply_markup` is a full
    /// `{"inline_keyboard": [...]}` value or `Value::Null`. An identical keyboard errors.
    pub async fn edit_reply_markup(
        &self,
        chat_id: &str,
        message_id: i64,
        reply_markup: &serde_json::Value,
    ) -> anyhow::Result<()> {
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
        .map_err(|(status, desc)| {
            anyhow::anyhow!("editMessageReplyMarkup failed ({status}): {desc}")
        })
    }

    async fn send_attachment(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        attachment: &TelegramAttachment,
    ) -> anyhow::Result<()> {
        let kind = attachment.kind;
        let target = attachment.target.trim();
        let route = attachment_route(kind, target);

        // A failure is returned, not papered over: the caller reports every failed attachment with
        // a reason. The delivery line is written once, on success only.
        let (file, bytes, source_url) = match route {
            AttachmentRoute::ByAddress => {
                self.send_media_by_url(chat_id, thread_id, kind, target)
                    .await?;
                (address_file_name(target, "file"), None, Some(target))
            }
            AttachmentRoute::FetchThenUpload => {
                let fetched = self.fetch_remote_upload(target).await?;
                let size = self
                    .send_media_file(chat_id, thread_id, kind, &fetched.path)
                    .await?;
                (fetched.name, Some(size), Some(target))
            }
            AttachmentRoute::LocalFile => {
                let path = Path::new(target);
                if !path.is_file() {
                    anyhow::bail!(
                        "Telegram attachment target is not an existing regular file: {target}"
                    );
                }
                let size = self.send_media_file(chat_id, thread_id, kind, path).await?;
                (target.to_string(), Some(size), None)
            }
        };

        record_attachment_delivery(chat_id, kind, route, &file, bytes, source_url);
        Ok(())
    }

    /// Post a built media request and check its status; success is the caller's to judge.
    async fn send_media(
        &self,
        api_method: &'static str,
        request: reqwest::RequestBuilder,
    ) -> anyhow::Result<()> {
        let resp = request.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            anyhow::bail!("Telegram {api_method} failed: {text}");
        }
        Ok(())
    }

    /// Send a media file (image-as-file, document, video, audio), returning the bytes uploaded. The
    /// body streams from disk with its own length; an oversized file quotes the mode's upload
    /// limit, and the local mode sets no total request bound.
    async fn send_media_file(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        kind: TelegramAttachmentKind,
        file_path: &Path,
    ) -> anyhow::Result<u64> {
        let meta = kind.file_meta();
        let limits = self.limits();
        let size = tokio::fs::metadata(file_path).await?.len();
        if size > limits.upload {
            return Err(anyhow::Error::new(UploadTooLarge {
                size,
                max: limits.upload,
            }));
        }

        let part = Part::file(file_path).await?;

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part(meta.form_field, part);

        if meta.disable_content_type_detection {
            form = form.text("disable_content_type_detection", "true");
        }

        if let Some(tid) = thread_id {
            form = form.text("message_thread_id", tid.to_string());
        }

        let request = self.with_transfer_bound(
            self.transfer_client()
                .post(self.api_url(meta.api_method))
                .multipart(form),
        );

        self.send_media(meta.api_method, request).await?;
        Ok(size)
    }

    /// Send a file by address (Telegram will download it).
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

        let request = self.with_transfer_bound(
            self.transfer_client()
                .post(self.api_url(meta.api_method))
                .json(&body),
        );

        self.send_media(meta.api_method, request).await
    }

    /// Fetch a remote attachment into the daemon's private temp root so it can be uploaded as a
    /// file. The address is model-supplied, so every hop is screened by the shared SSRF guard; the
    /// bytes stream to disk under the mode's upload ceiling, and the whole transfer lives under the
    /// bound this mode applies to any other.
    async fn fetch_remote_upload(&self, url: &str) -> anyhow::Result<FetchedUpload> {
        let bound = self.limits().request_bound();
        tokio::time::timeout(bound, async {
            let max = self.limits().upload;
            let response = crate::util::http::get_guarded(fetch_client(), url).await?;
            if !response.status().is_success() {
                anyhow::bail!("the address answered {}", response.status());
            }
            // The address last fetched: a redirect chain may end elsewhere, and that one names the
            // file.
            let fetched_url = response.url().as_str().to_string();
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);

            let dir = tempfile::Builder::new()
                .prefix("telegram-upload-")
                .tempdir()
                .context("the temp directory for the fetched file could not be created")?;
            let name = video_upload_name(&fetched_url, content_type.as_deref());
            let path = dir.path().join(&name);
            stream_capped(response, &path, Some(max), |size, max| {
                UploadTooLarge { size, max }.into()
            })
            .await?;

            Ok::<_, anyhow::Error>(FetchedUpload {
                path,
                name,
                _dir: dir,
            })
        })
        .await
        .with_context(|| {
            format!(
                "fetching {} did not finish within {bound:?}",
                crate::util::http::redact_address(url)
            )
        })?
    }
}

/// A fetched upload, removed with this value whatever the upload did; a process that dies
/// mid-transfer leaves a directory the temp cleaner reclaims.
struct FetchedUpload {
    path: PathBuf,
    name: String,
    _dir: tempfile::TempDir,
}

/// Stream a body into `dest`, refusing one over `ceiling` on the declared length and again on the
/// stream; `too_large` renders the refusal for the caller's own transport.
async fn stream_capped(
    response: reqwest::Response,
    dest: &Path,
    ceiling: Option<u64>,
    too_large: impl Fn(u64, u64) -> anyhow::Error,
) -> anyhow::Result<()> {
    use futures_util::StreamExt as _;
    use tokio::io::AsyncWriteExt as _;

    if let (Some(max), Some(declared)) = (ceiling, response.content_length())
        && declared > max
    {
        return Err(too_large(declared, max));
    }
    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        // The transport's own error quotes the address it fetched, query and all; strip it here so
        // no caller can log a caller-supplied address whole.
        let chunk = chunk
            .map_err(reqwest::Error::without_url)
            .context("the transfer was cut off")?;
        written += chunk.len() as u64;
        if let Some(max) = ceiling
            && written > max
        {
            return Err(too_large(written, max));
        }
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    Ok(())
}

/// One debug line per delivered attachment: what went out, on which route, from where. A successful
/// delivery is routine, so ordinary operation keeps no record of it — every failure still is.
/// `file` is the name the delivery carries (path, generated name, or the address's last segment)
/// and `source_url` that address for both remote routes; both are redacted by
/// [`crate::util::http::redact_address`], so no query, fragment or signed credential outlives the
/// delivery. `bytes` is what the product uploaded — `None` by address, where Telegram fetched and
/// may have re-encoded the file.
fn record_attachment_delivery(
    chat_id: &str,
    kind: TelegramAttachmentKind,
    route: AttachmentRoute,
    file: &str,
    bytes: Option<u64>,
    source_url: Option<&str>,
) {
    let (method, delivery) = match route {
        AttachmentRoute::ByAddress => (kind.meta().api_method, "by address"),
        AttachmentRoute::LocalFile | AttachmentRoute::FetchThenUpload => {
            (kind.file_meta().api_method, "uploaded attachment")
        }
    };
    let file = crate::util::http::redact_address(file);
    let source_url = source_url.map(crate::util::http::redact_address);
    tracing::debug!(
        chat_id,
        kind = kind.label(),
        file,
        method,
        delivery,
        bytes = ?bytes,
        source_url = ?source_url,
        "Telegram: attachment delivered"
    );
}

/// Outcome of a single `getUpdates` poll.
enum PollOutcome {
    /// Fetched updates, still unconfirmed: the next poll's offset is what confirms them.
    Updates(Vec<serde_json::Value>),
    /// 409 Conflict — caller decides backoff.
    Conflict,
    /// A non-409 API error, with the description it answered.
    Error(String),
    /// Network or parse error (sleep already applied by helper).
    Transport,
}

impl TelegramChannel {
    /// Call `getUpdates` for everything at or past `offset` and classify the outcome; what the next
    /// poll resumes from comes from the intake's record, not from here. `ok_default` decides
    /// whether a missing `ok` field is an error (handshake) or assumed success (main loop).
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

    /// The intake's opening handshake: the account the token belongs to and a claimed `getUpdates`
    /// slot (a previous daemon's 30-second poll may still be active). Retries until the listener is
    /// cancelled or the daemon drains (`None` then); the claim's batch is returned rather than
    /// discarded, being unconfirmed on the platform's side.
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

    /// The account the token belongs to. The record is keyed on it rather than on the token text,
    /// so a rotated token for the same account still recovers what that account is holding.
    async fn bot_account_id(&self) -> anyhow::Result<String> {
        let data = get_me(self.http_client(), &self.api_base, &self.bot_token).await?;
        let id = data["result"]["id"]
            .as_i64()
            .ok_or_else(|| anyhow::anyhow!("getMe response carries no bot id"))?;
        Ok(id.to_string())
    }

    /// Parse one polled or recovered update into a message, or (logged and skipped) nothing. A
    /// parsed update gets its acknowledgement reaction. `Err` is the sender-resolution read
    /// failing, which the caller keeps the event held for.
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

    /// Take one polled batch on: record every event, then hand the processor the ones this call was
    /// the first to record. The record is written here, on the poll loop, because writing it is
    /// what lets the offset advance; the parse, which may download for minutes, is not. Polling
    /// continues only while the processor has room ([`BATCH_QUEUE`]), the real responsiveness bound
    /// during a large transfer, and nothing is lost at it: a batch left unconfirmed is offered
    /// again and deduped against the record. A unit the record could not be written for holds the
    /// position where it starts, so the platform offers it again instead of the intake handing on
    /// the parts of it that could be recorded.
    async fn hold_up(
        &self,
        intake: &Intake,
        offset: i64,
        updates: Vec<serde_json::Value>,
        origin: BatchOrigin,
    ) -> (Polled, Option<HeldBatch>) {
        // A long poll that timed out with nothing to offer leaves the position where it is: there
        // is nothing behind it the platform has not given up on already.
        if updates.is_empty() {
            return (
                Polled {
                    position: offset,
                    reoffered: false,
                },
                None,
            );
        }
        let plan = poll_plan(&updates);
        let mut fresh = HashSet::new();
        // The unit the record could not be written for — an unwritable store, or an update with no
        // `update_id` to move past — holds the position where the request started.
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

        // One warning per batch, not per event: a pinned position re-offers the same page every
        // pause, so a broken store would otherwise log a warning per update per poll.
        if let Some(err) = record_error {
            tracing::warn!(
                first_unrecorded = ?unrecorded,
                units = failed_units,
                %err,
                "Telegram intake: could not record part of a batch"
            );
        }

        // The position stops at the first event the record could not be written for.
        let position = unrecorded.unwrap_or_else(|| {
            updates
                .iter()
                .filter_map(|update| update["update_id"].as_i64())
                .max()
                .map_or(offset, |newest| newest + 1)
        });

        // The batch is handed to the processor rather than parsed here: parsing may spend minutes
        // on a download and the poll loop must keep acknowledging (see [`HeldBatch`]).
        if unrecorded.is_some() {
            self.intake_pause(RECORD_FAILURE_BACKOFF).await;
        }
        let polled = Polled {
            position,
            reoffered: unrecorded.is_some(),
        };
        (
            polled,
            Some(HeldBatch {
                updates,
                plan,
                origin,
                fresh,
            }),
        )
    }

    /// Parse a recorded batch and hand it to the collector. `fresh` are the events this call was
    /// the first to record; the rest are skipped, and one nothing can be made of is settled by its
    /// taker ([`Self::settle_or_hold`]). An album is parsed whole at its first member's position,
    /// so it is delivered where the user sent it. `false` when the collector is gone.
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
                // A button press is not a message: its `message` is the one the button sits on.
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
                        // A member nothing can be made of is settled by `take_event`; one it cannot
                        // decide yet stays held. Either way the rest of the album goes on.
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

    /// The record entry of a batch update when this call is the first to take it up — taken out of
    /// `fresh`, so a batch that repeats an `update_id` walks the event once. `fresh` holds the ids
    /// the record accepted: an album is recorded whole or not at all.
    fn fresh_update(update: &serde_json::Value, fresh: &mut HashSet<i64>) -> Option<i64> {
        let id = update["update_id"].as_i64()?;
        fresh.remove(&id).then_some(id)
    }

    /// Turn one recorded unit's parse into the message to hand on, or write the record's decision
    /// about it: `Ok(None)` is settled and never comes back, `Err` is no decision, so the unit
    /// stays held — payload and all — exactly as one a crash left behind ([`Intake::settle`]).
    async fn settle_or_hold(
        intake: &Intake,
        update_id: i64,
        parsed: anyhow::Result<Option<ChannelMessage>>,
    ) -> Option<ChannelMessage> {
        let mut msg = match parsed {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                intake.settle(&[update_id]).await;
                return None;
            }
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
        Some(msg)
    }

    /// Take one recorded event on: parse it, hand it over or settle it, and refuse control input of
    /// unknown age ([`BatchOrigin::Stale`]) instead of executing it out of context.
    async fn take_event(
        &self,
        intake: &Intake,
        origin: BatchOrigin,
        update_id: i64,
        update: &serde_json::Value,
    ) -> Option<ChannelMessage> {
        let msg = Self::settle_or_hold(intake, update_id, self.parse_incoming_update(update).await)
            .await?;
        if origin == BatchOrigin::Stale && is_control_message(&msg) {
            self.refuse_unacted_control(intake, &msg).await;
            return None;
        }
        Some(msg)
    }

    /// Tell the user a control input was not acted on, and settle it: input in a batch of unknown
    /// age is not executed out of context, and its press-time answer expired just as long ago. An
    /// event whose notice could not be sent stays held, so a later run tries again.
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

    /// Handle one button press, reporting whether the pipeline is still alive. The press is parsed
    /// on its own, never as the message it sits on. The spinner is dismissed only for a press this
    /// run polled: a stale press's answer expired long ago, so that would be a doomed API call.
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

        // For __act__ callbacks, answer nothing here — the action handler answers with its own
        // toast. Dismiss the spinner now for all other callbacks.
        if origin == BatchOrigin::Fresh
            && !cq_data.starts_with(ACTION_PREFIX)
            && let Some(ref id) = cq_id
        {
            self.answer_callback_query(id, None).await;
        }

        let Some(msg) =
            Self::settle_or_hold(intake, update_id, self.parse_callback_query(cq).await).await
        else {
            return true;
        };
        // A press is control input by construction, refused by the same rule `take_event` uses.
        if origin == BatchOrigin::Stale {
            self.refuse_unacted_control(intake, &msg).await;
            return true;
        }
        tx.send(msg).await.is_ok()
    }

    /// Take up the events this account's record holds and no run handed on: what a crash, a
    /// self-update or a kill in the wrong instant left behind. A recovered event joins the same
    /// collector as a fresh one. `false` when the pipeline is gone.
    async fn recover(
        &self,
        intake: &Intake,
        parts_tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
    ) -> bool {
        // A failed take-over leaves every row it would have claimed held until a whole listener
        // rebuild, so a transient failure is worth another try before the event waits that long.
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
        // The album regrouping the stored updates came back in is the same planning the poll path
        // uses.
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

    /// Age the record out: the intake's own retention job.
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

    /// The intake proper, once the handshake has resolved the account and fetched the opening
    /// batch: the sweep, the long-poll loop, and the queue of held batches the processor takes its
    /// work from. This half never parses or downloads anything — a batch is recorded and queued,
    /// the offset advances, the next poll is issued — so a multi-gigabyte attachment cannot stop
    /// the platform being polled. Returns when the listener stops, however it stops; the frame
    /// stays the collector's until the burst it held has been handed on ([`ListenerFrame`]).
    async fn intake_loop(
        &self,
        batch_tx: &tokio::sync::mpsc::Sender<HeldBatch>,
        intake: &Intake,
        handshake: Vec<serde_json::Value>,
    ) {
        // The sweep runs before anything is taken up: a row that outlived the retention bound must
        // never stand in for a later event the platform re-used the id of.
        self.age_out_record().await;
        let mut aged_at = Instant::now();

        // The handshake's batch was fetched at offset 0, before any position existed, so it is
        // input of unknown age — as is every batch behind it until a fetch comes back short.
        // Control input in any of them must not be executed out of context.
        let page_full = handshake.len() >= UPDATES_PAGE;
        let (taken, batch) = self.hold_up(intake, 0, handshake, BatchOrigin::Stale).await;
        let mut offset = taken.position;
        let mut backlog_open = page_full || taken.reoffered;
        if let Some(batch) = batch
            && batch_tx.send(batch).await.is_err()
        {
            return;
        }

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
                // The intake stops fetching before it stops handling: an event arriving during the
                // wind-down is either left unconfirmed for the next start or already held durably.
                // Outbound sending keeps working and the channel stays registered.
                () = crate::shutdown::drain_wait() => {
                    tracing::info!("Telegram channel stopping the intake (daemon draining)");
                    break;
                }
                poll_result = self.poll_get_updates(offset, 30, true) => {
                    // Every completed poll ages resolved rows out, whatever it carried: a run that
                    // only ever sees errors or empty batches must still trim the record.
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
                            // Informational: the poll is retried every 5s, so an unhealthy API
                            // would flood the issues view.
                            tracing::info!("Telegram getUpdates API error: {desc}");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            continue;
                        }
                        PollOutcome::Transport => continue,
                    };

                    let page_full = updates.len() >= UPDATES_PAGE;
                    let (taken, batch) = self
                        .hold_up(intake, offset, updates, batch_origin(backlog_open))
                        .await;
                    offset = taken.position;
                    backlog_open = (backlog_open && page_full) || taken.reoffered;
                    if let Some(batch) = batch
                        && batch_tx.send(batch).await.is_err()
                    {
                        tracing::info!("Telegram listener stopping: the batch processor is gone");
                        return;
                    }
                }
            }
        }
    }
}

/// One polled batch the poll loop recorded and the processor has yet to parse, so the record is
/// written (and the offset advanced) on one task while the parse happens on the other. Every event
/// in it is held durably, so a stop between the two loses nothing.
struct HeldBatch {
    updates: Vec<serde_json::Value>,
    /// The album grouping the record wrote the batch by, and the parse walks it by.
    plan: Vec<PollPlacement>,
    origin: BatchOrigin,
    fresh: HashSet<i64>,
}

/// How many recorded batches may wait for the processor: deep enough that one long download leaves
/// polling running, bounded so a stuck processor cannot grow it without limit.
const BATCH_QUEUE: usize = 32;

/// What the opening handshake resolves to.
struct IntakeSlot {
    account_id: String,
    /// The claim's batch, still unconfirmed: exactly what the intake now has to hold.
    updates: Vec<serde_json::Value>,
}

/// Fold a parsed album member into the album built so far: content joined with a newline (the album
/// reads as one message) and the member's staging directories kept, since each member downloads
/// into its own. A forwarded member makes the album forwarded, so the group it lands in is owed the
/// same longer window; the record time stays the first member's. The merged message settles every
/// member's entry.
fn merge_album_member(album: Option<ChannelMessage>, member: ChannelMessage) -> ChannelMessage {
    match album {
        Some(mut album) => {
            album.content.push('\n');
            album.content.push_str(&member.content);
            album.attachment_dirs.extend(member.attachment_dirs);
            album.update_ids.extend(member.update_ids);
            album.forwarded |= member.forwarded;
            // An album is one message: it is urgent when any of its members is.
            album.urgent |= member.urgent;
            album
        }
        None => member,
    }
}

/// Whether a batch is one this run's own poll is seeing for the first time. Control input is acted
/// on only for a fresh batch: the handshake's, a re-offered one and a recovered one may all be
/// older than this run. A message is handed over regardless.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchOrigin {
    /// A batch this run's poll fetched with nothing of the platform's backlog behind it.
    Fresh,
    /// A batch that may repeat what this run fetched, or predate it.
    Stale,
}
/// What one taken-up batch leaves behind.
struct Polled {
    /// The position the next `getUpdates` resumes from: past the batch once every event is held, or
    /// at the unit that could not be recorded — which the platform then offers again.
    position: i64,
    /// True when part of the batch could not be recorded, so the platform re-offers it — one of the
    /// two reasons the next batch is of unknown age ([`batch_origin`]).
    reoffered: bool,
}

/// The origin of the next batch, from what this one left unknown: it filled the platform's own
/// page, so the backlog may still be open, or part of it could not be recorded and is offered
/// again. Either way control input in the next answer must not run out of context.
fn batch_origin(unknown_age: bool) -> BatchOrigin {
    if unknown_age {
        BatchOrigin::Stale
    } else {
        BatchOrigin::Fresh
    }
}

/// Marks a listener frame as running for the whole of `listen`, so a replacement can wait for what
/// this frame held to be handed on ([`TelegramChannel::wait_stopped`]). The frame is a value the
/// collector owns, not the `listen` future, so a dropped or panicking listener still reads as
/// running until its collector has finished. Only a crash is uncovered, and that is the record's.
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
        // A running→stopped transition is every waiter's business: a waiter left unnotified would
        // sit out the whole bound.
        self.stopped.notify_waiters();
    }
}

/// How long the intake pauses after a poll whose record could not be written for, so a failing
/// store cannot become a tight loop. Only the unrecorded unit holds the position back — the rest of
/// the batch is recorded and handed on — and the platform's own window retires the event.
const RECORD_FAILURE_BACKOFF: Duration = Duration::from_secs(5);

const AGE_OUT_INTERVAL: Duration = Duration::from_hours(1);

/// The platform's `getUpdates` page size: a batch this size means the platform may still be holding
/// more.
const UPDATES_PAGE: usize = 100;

const LISTENER_STOP_WAIT: Duration = Duration::from_secs(30);

/// How many times the intake tries to take up what the record holds before leaving those rows to
/// the next run, and the pause between tries: contention is worth a retry, a really unusable store
/// must not delay the listener.
const TAKE_OVER_ATTEMPTS: u32 = 3;
const TAKE_OVER_RETRY: Duration = Duration::from_secs(1);

/// What a user whose control input was never acted on is told: it arrived in a batch of unknown
/// age, so it is not executed out of context and must be repeated.
const UNACTED_CONTROL_NOTICE: &str = "MahBot did not act on that. Please repeat it.";

/// How one update of a poll is parsed: an album is parsed whole at its first member's position, its
/// other members skipped where they sit, which keeps the user's send order.
#[derive(Debug, PartialEq, Eq)]
enum PollPlacement {
    Single,
    /// An album's first member: parse every listed update (this one first) and merge them.
    Album(Vec<usize>),
    Skip,
    /// A button press, handled as control input. It carries the message its button sits on, which
    /// is why it is placed apart: reading that message here would put an album photo into its album
    /// twice, or — when the press is the album's first member — lose the album's other members.
    Callback,
}

/// Read each album's (media group's) member positions from the raw updates, so the poll can be
/// walked once: album membership is a field of the raw update, nothing has to be parsed.
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

    async fn send(&self, message: &SendMessage) -> Result<SendOutcome, SendFailure> {
        let content = message.content.trim();
        if content.is_empty() {
            tracing::warn!("TelegramChannel: attempted to send empty message – skipping");
            return Ok(SendOutcome::default()); // nothing to send, not an error
        }

        let (chat_id, thread_id) = parse_recipient(&message.recipient);

        // Per-user command menu refresh — fire-and-forget, fail-open, never blocks delivery.
        self.spawn_menu_refresh(chat_id);

        // The marker gates stat and head-read local files, so this runs off the async worker. A
        // relative target resolves against `message.file_roots`.
        let (text_without_markers, attachments, refusals) =
            crate::util::with_block_in_place(|| {
                parse_attachment_markers(content, &message.file_roots, self.limits())
            });

        if attachments.is_empty() && refusals.is_empty() {
            return self
                .send_text_chunks(content, chat_id, thread_id, message.reply_markup.clone())
                .await
                .map(|()| SendOutcome::default())
                .map_err(|error| SendFailure {
                    error,
                    delivered: false,
                    refused: Vec::new(),
                });
        }

        // A refused marker leaves nothing in the delivered body — the conversation that wrote it
        // is told instead — so the owner gets no broken attachment. An attachment that fails after
        // acceptance is reported in the chat by the loop below.
        let mut delivered = false;
        if !text_without_markers.is_empty() {
            if let Err(error) = self
                .send_text_chunks(
                    &text_without_markers,
                    chat_id,
                    thread_id,
                    message.reply_markup.clone(),
                )
                .await
            {
                return Err(SendFailure {
                    error,
                    delivered: false,
                    refused: refusals,
                });
            }
            delivered = true;
        }

        // One failed attachment must not cost the user the others: every failure is surfaced
        // in-chat, and the first error is returned after the loop so delivery-failure logging still
        // fires.
        let mut first_error: Option<anyhow::Error> = None;
        for attachment in &attachments {
            if let Err(e) = self.send_attachment(chat_id, thread_id, attachment).await {
                tracing::warn!(
                    error = %e,
                    file = %crate::util::http::redact_address(&attachment.target),
                    "Telegram: attachment delivery failed"
                );
                // The notice names a bounded last path component: an address's query, fragment and
                // userinfo are dropped, so a signed URL's credentials never come back into the
                // chat.
                let name = media_target::marker_name(&attachment.target);
                let notice = match upload_too_large(&e) {
                    Some(too_large) => media_target::marker_refusal(&name, &too_large.to_string()),
                    None if is_http_url(&attachment.target) => {
                        media_target::marker_refusal(&name, ATTACHMENT_FETCH_FAILED_REASON)
                    }
                    None => media_target::marker_refusal(&name, ATTACHMENT_UPLOAD_FAILED_REASON),
                };
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
            Some(error) => Err(SendFailure {
                error,
                delivered,
                refused: refusals,
            }),
            None => Ok(SendOutcome { refused: refusals }),
        }
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        let frame = ListenerFrame::enter(&self.listener_running, &self.listener_stopped);
        // A listener starting inside a wind-down takes nothing on: an event fetched now would only
        // have to be held until the next start.
        if crate::shutdown::is_draining() {
            tracing::info!("Telegram channel not starting: the daemon is draining");
            return Ok(());
        }

        tracing::info!("Telegram channel listening for messages...");

        // Opening handshake: the token's account and the getUpdates slot, before the long-poll
        // loop.
        let Some(slot) = self.open_intake().await else {
            return Ok(());
        };

        tracing::debug!("Startup handshake succeeded; entering main long-poll loop.");

        let intake = Intake::new(
            self.intake_store(),
            self.intake_claim.clone(),
            &slot.account_id,
        );

        // Collected bursts are assembled off the poll loop, so getUpdates keeps being issued while
        // a group waits; the queue's bound and drop behaviour are `telegram_group`'s. The collector
        // also settles the record — a message it handed to the pipeline has its events resolved,
        // one the pipeline never took leaves them held — and takes the frame, so its life is the
        // collector's.
        let (parts_tx, parts_rx) =
            tokio::sync::mpsc::channel(crate::channels::telegram_group::QUEUE_CAPACITY);
        let collector = crate::channels::telegram_group::spawn_collector(
            parts_rx,
            tx,
            intake.clone(),
            std::sync::Arc::clone(&self.preparations),
            frame,
        );

        // Two halves, one future: the poll loop records a batch and queues it, the processor parses
        // it and hands the parts to the collector. A large attachment therefore keeps the platform
        // polled while the processor has queue room, and the offset advances only on what was
        // recorded, so nothing is lost or acknowledged early.
        let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<HeldBatch>(BATCH_QUEUE);
        // Each half owns what it needs: the sender dies with the poll loop, ending the receiver.
        let fetch_intake = intake.clone();
        let fetch = async move {
            self.intake_loop(&batch_tx, &fetch_intake, slot.updates)
                .await;
        };
        let process_intake = intake.clone();
        let process = async move {
            // What a previous run held is taken over first: it is older than anything now on offer.
            if self.recover(&process_intake, &parts_tx).await {
                while let Some(HeldBatch {
                    updates,
                    plan,
                    origin,
                    mut fresh,
                }) = batch_rx.recv().await
                {
                    // A stop refuses new work as the poll loop does: the batch stays recorded, so
                    // the next listener finds it instead of it being half-handled here.
                    if crate::shutdown::aborting() {
                        tracing::info!(
                            "Telegram listener stopping: recorded batches left for the next run"
                        );
                        break;
                    }
                    if !self
                        .process_updates(
                            &parts_tx,
                            &process_intake,
                            &mut fresh,
                            origin,
                            updates,
                            &plan,
                        )
                        .await
                    {
                        tracing::warn!(
                            "Telegram listener stopping: the message pipeline is closed"
                        );
                        break;
                    }
                }
            } else {
                tracing::warn!("Telegram listener stopping: the message pipeline is closed");
            }
            // Dropping the last sender releases an open burst: the collector hands it to the
            // pipeline and settles it, so this frame ends only once that collector has finished
            // ([`ListenerFrame`]).
            drop(parts_tx);
        };
        tokio::pin!(fetch, process);
        tokio::select! {
            // The poll loop closing the queue ends the processor once it has drained what it holds.
            () = &mut fetch => { process.await; }
            // The pipeline is gone: nothing fetched could be handed on, so the poll loop is
            // cancelled rather than left polling a bot no one reads — two pollers on one token the
            // moment a replacement starts.
            () = &mut process => {
                self.cancel.cancel();
                fetch.await;
            }
        }

        if let Err(err) = collector.await {
            tracing::error!(
                %err,
                "Telegram intake: the burst collector failed; what it still held stays in the record"
            );
        }
        Ok(())
    }

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

/// The Bot API base the registered listener was built for: the supervisor compares it with the
/// endpoint serving, so a server restarted on a fresh port is followed.
pub(crate) fn listener_base() -> Option<String> {
    let channel = crate::channel_registry().get("telegram")?;
    channel
        .as_any()
        .downcast_ref::<TelegramChannel>()
        .map(|channel| channel.api_base.clone())
}

/// Start the listener the current mode serves. When the mode is the local server this is the step
/// that makes a switch real: a token-bearing request is what logs the bot into that server. `Err`
/// therefore means the bot is not being served at all.
pub(crate) async fn start_telegram_listener() -> anyhow::Result<()> {
    let token = crate::config::CONFIG
        .telegram_bot_token()
        .ok_or_else(|| anyhow::anyhow!("Telegram has no bot token configured"))?;
    // The switch's own first token-bearing request against the base it just moved the bot to, and
    // what logs it in there: a rejected token is reported before any listener is built.
    TelegramChannel::validate_token(&token).await?;
    restart_telegram_listener(Some(&token))
}

/// Stop the listener and wait until the frame holding its events has ended: everything it held is
/// then handed on or left held, which is the state a switch needs before it takes the bot out of
/// one server. The retirement itself is [`retire_channel`]'s.
pub(crate) async fn stop_telegram_listener() -> anyhow::Result<()> {
    let registry = crate::channel_registry();
    let frames = registry.replaced("telegram");
    match registry.get("telegram") {
        Some(current) => {
            retire_channel(registry, std::sync::Arc::clone(&current)).await;
        }
        // Nothing is registered, but a frame an earlier attempt left is still polling the side we
        // leave.
        None => stop_listeners(&frames, None).await,
    }
    // The wait is bounded, so the caller has to know about a listener still finishing: logging the
    // bot out of the cloud meanwhile would leave two pollers on one token.
    let still_running = frames
        .snapshot()
        .iter()
        .filter_map(|frame| frame.as_any().downcast_ref::<TelegramChannel>())
        .any(TelegramChannel::running);
    if still_running {
        return Err(anyhow::anyhow!(
            "the Telegram listener is still finishing the batch it held"
        ));
    }
    Ok(())
}

/// Run the Telegram channel the configured mode serves, for the life of the process. The boot-time
/// preparation is here because the base URL is only known once the server runs, and a switch
/// interrupted by the last shutdown has to finish first.
pub async fn run_configured_channel(tx: tokio::sync::mpsc::Sender<ChannelMessage>) {
    use crate::channels::telegram_local::Boot;

    let Some(token) = crate::config::CONFIG.telegram_bot_token() else {
        tracing::info!("No Telegram bot token configured — running in dashboard-only mode");
        return;
    };

    match crate::channels::telegram_local::prepare_for_boot().await {
        Boot::Start => {}
        Boot::Running => {
            tracing::info!("Telegram channel already started by the resumed switch");
            return;
        }
        Boot::Waiting => {
            tracing::warn!(
                "Telegram: the local Bot API server is not ready — the bot connects once it is"
            );
            return;
        }
    }

    let Some(channel) = TelegramChannel::from_config(token) else {
        tracing::warn!("Telegram: the configured mode has no Bot API endpoint yet");
        return;
    };
    let channel = std::sync::Arc::new(channel);
    // Register commands with the Bot API (fire-and-forget, non-blocking).
    tokio::spawn({
        let tc = std::sync::Arc::clone(&channel);
        async move {
            tc.set_my_commands().await;
        }
    });
    let channel: std::sync::Arc<dyn Channel> = channel;
    crate::channel_registry().register(std::sync::Arc::clone(&channel));
    if let Err(e) = channel.listen(tx).await {
        tracing::error!(error = %e, "Telegram listener error");
    }
}

/// Stop and wait out every listener of the intake's own that may still hold events: the channels a
/// replace moved past ([`crate::ReplacedChannels`]). `starting` is never stopped
/// ([`hand_over_listener`]) and is left in the list, so a later hand-over still finds it. The list
/// is taken from the name's own record so every hand-over waits for the same frames — a chain the
/// channels carried would be consumed by whichever walked it first. A frame still open past the
/// wait bound is left there, so the next hand-over waits it again.
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

/// Hot-reload the bot listener with a new token, no application restart needed.
///
/// The hand-over runs in a task of its own, so the caller's answer comes as soon as the registry
/// holds the new channel. With `Some(token)` the pipeline is resolved (no listener without it) and
/// the token is used as given — the paths that must prove a listener starts validate it first — and
/// the hand-over then cancels and waits out every listener the replace moved past. With `None` the
/// old listener is cancelled and the channel unregistered once its frame has ended, unless a token
/// saved meanwhile took its place; it keeps serving outbound sends while it winds down.
pub fn restart_telegram_listener(new_token: Option<&str>) -> anyhow::Result<()> {
    let registry = crate::channel_registry();

    if let Some(token) = new_token.filter(|t| !t.trim().is_empty()) {
        // The pipeline is resolved before anything is touched — no platform call, no registry entry
        // — so a missing one is reported rather than leaving a token the product never applied
        // looking saved.
        let Some(tx) = crate::MESSAGE_TX.get().cloned() else {
            anyhow::bail!(
                "the message pipeline is not running — the Telegram listener cannot start"
            );
        };

        // Validating here would be a second token-bearing request for one save: the proving paths
        // do that, and a rebuild reuses a token that has already served.
        let telegram_arc = std::sync::Arc::new(
            TelegramChannel::from_config(token.to_string()).ok_or_else(|| {
                anyhow::anyhow!(
                    "the local Telegram Bot API server has no endpoint yet — the listener cannot start"
                )
            })?,
        );

        // Register commands with Telegram API (fire-and-forget, non-blocking).
        tokio::spawn({
            let tc = std::sync::Arc::clone(&telegram_arc);
            async move {
                tc.set_my_commands().await;
            }
        });

        let new_channel: std::sync::Arc<dyn Channel> = telegram_arc;

        // Atomically replace in the registry — no gap where "telegram" returns None.
        let frames = registry.replaced("telegram");
        registry.replace(std::sync::Arc::clone(&new_channel));

        tokio::spawn(hand_over_listener(frames, new_channel, tx));
        tracing::info!("Telegram bot listener restarting with new token");
    } else {
        // Token cleared — stop the old listener and unregister it once it has ended.
        let old_channel = registry.get("telegram");
        tokio::spawn(async move { retire_listener(registry, old_channel).await });
    }

    Ok(())
}

/// Stop the listener `channel` runs and take it out of the registry once its frame has ended,
/// reporting whether the registry still held it. The channel is recorded among the frames a
/// hand-over waits out, so a save landing meanwhile waits for it before starting its own.
async fn retire_channel(
    registry: &crate::ChannelRegistry,
    channel: std::sync::Arc<dyn Channel>,
) -> bool {
    let frames = registry.replaced("telegram");
    frames.record(std::sync::Arc::clone(&channel));
    stop_listeners(&frames, None).await;
    registry.unregister_if(&channel)
}

/// Stop `old` and, once its frame has ended, take it out of the registry — unless the registry
/// holds a channel registered since, which is the live one by then. The two log lines are the whole
/// of it; stopping and unregistering are [`retire_channel`]'s.
async fn retire_listener(
    registry: &crate::ChannelRegistry,
    old: Option<std::sync::Arc<dyn Channel>>,
) {
    let Some(old) = old else {
        return;
    };
    if retire_channel(registry, old).await {
        tracing::info!("Telegram bot token cleared — listener stopped");
    } else {
        tracing::info!("Telegram bot token cleared — a channel registered since is live");
    }
}

/// The hand-over half of [`restart_telegram_listener`], in a task of its own so a caller that goes
/// away cannot desynchronise the registry from the listener.
///
/// `new` is the channel about to start and is never stopped here: a stop fires its cancel token, so
/// the listener started right after would return at once.
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

/// Send a message through the registered channel, bypassing the router (no broadcast/persist).
///
/// With no channel registered — a switch has just taken the listener away, exactly when the owner
/// has to be told — the message goes out through a one-off channel on the base the mode serves.
pub async fn send_direct(
    recipient: &str,
    content: String,
    reply_markup: Option<serde_json::Value>,
) -> anyhow::Result<()> {
    let reply = SendMessage {
        content,
        recipient: recipient.to_string(),
        reply_markup,
        file_roots: Vec::new(),
    };
    if let Some(channel) = crate::channel_registry().get("telegram") {
        return channel.send(&reply).await.map(|_| ()).map_err(Into::into);
    }
    let token = crate::config::CONFIG
        .telegram_bot_token()
        .ok_or_else(|| anyhow::anyhow!("Telegram channel not found in registry"))?;
    let channel = TelegramChannel::from_config(token)
        .ok_or_else(|| anyhow::anyhow!("the Bot API this mode serves has no endpoint"))?;
    channel.send(&reply).await.map(|_| ()).map_err(Into::into)
}

/// Best-effort plain-text send through the registered channel, for replies and notifications.
pub async fn send_reply(recipient: &str, content: &str) {
    let _ = send_direct(recipient, content.to_string(), None).await;
}

/// Mirror a local user's message to their Telegram chats as a blockquote, before enrichment, so the
/// original typed text is preserved (pre-link-summary, pre-transcription).
///
/// Only `gui`, `voice` and `cli` messages mirror — strictly local sources, so accepting them cannot
/// create an echo loop; Telegram-originated messages stay excluded. Empty messages, and users with
/// no Telegram binding, are skipped silently, and **all** of a user's bindings get the quote.
///
/// Media markers are stripped so no raw marker syntax reaches the quote, and a message that is
/// nothing but media is skipped — unless it carries a [`ReplyReference`], which leads as a `↩
/// {author}: {snippet}` line even then.
pub async fn mirror_gui_message_to_telegram(msg: &ChannelMessage) {
    // Only local-originated user messages mirror: those sources can never come from Telegram, so
    // accepting them cannot create feedback.
    if msg.channel != "gui" && msg.channel != "voice" && msg.channel != crate::cli::CHANNEL {
        return;
    }

    let trimmed = msg.content.trim();
    if trimmed.is_empty() {
        return;
    }

    let Some(channel) = crate::channel_registry().get("telegram") else {
        return;
    };

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

    let telegram_bindings: Vec<_> = bindings
        .into_iter()
        .filter(|b| b.channel == "telegram")
        .collect();

    if telegram_bindings.is_empty() {
        return; // No Telegram binding — silently skip.
    }

    // Strip media markers so users don't see raw `[IMAGE:...]` syntax; every kind the pattern
    // matches goes, an unknown marker-shaped word survives, and a reply reference is preserved as a
    // `↩` header line — which relaxes the media-only guard below.
    let content = MEDIA_MARKER_RE.replace_all(trimmed, "").to_string();
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
        // <blockquote> passes through markdown_to_telegram_html unchanged.
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
            // The quote is already marker-free, so an empty root list cannot turn the user's own
            // text into a "Could not send ..." refusal.
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

/// POST a `setMyCommands` request: `true` on success, a logged warning and `false` on failure, so
/// callers reset their change detection and retry on the next message.
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

/// The phase glyph standing in for the phase word on one `/board` line. Keyed by the phase with no
/// fallback arm, so a new phase is a compile error. All ten already default to emoji presentation,
/// and this is deliberately not `TicketPhase::display_name()`: the desktop badge keeps the words.
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

/// One `/board` line: the phase emoji, monospace ticket id, title. No `•`/`*` bullet — a leading
/// `*` would pair with one in a title and swallow the id into italics — and each line converts to
/// HTML on its own, so a title cannot corrupt another line's formatting.
#[must_use]
pub fn format_board_line(phase: &TicketPhase, id: &str, title: &str) -> String {
    format!("{} `{}` {}", phase_emoji(*phase), id, title)
}

/// The complete `/board` reply for `tickets`, already in display order: an all-workspaces header,
/// then one [`format_board_line`] per ticket, whose id prefix is the workspace marker. An empty
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

/// `(command, description)` pairs for a user's menu, from their admin status: shared by the
/// per-chat refresh and the `/start` welcome.
#[must_use]
pub async fn user_command_entries(user_name: &str) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::new();

    if crate::users::is_admin(user_name).await {
        entries.push(("board".to_string(), BOARD_COMMAND_DESC.to_string()));
        entries.push(("archive".to_string(), ARCHIVE_COMMAND_DESC.to_string()));

        // `/update` is global (the single admin) and shown only when the shared availability cache
        // confirms an update — always for a copy built from sources. The menu reflects the cache,
        // never a network call per refresh.
        if crate::self_update::should_show_update(crate::self_update::update_availability()) {
            entries.push(("update".to_string(), UPDATE_COMMAND_DESC.to_string()));
        }
    }

    // The media model commands are per-user, and every user has the Assistant role.
    entries.push((
        "image_models".to_string(),
        IMAGE_MODELS_COMMAND_DESC.to_string(),
    ));
    entries.push((
        "video_models".to_string(),
        VIDEO_MODELS_COMMAND_DESC.to_string(),
    ));

    // `/clear` sits at the very bottom. The `/agents` role-switch entry is gone: the Assistant is
    // every account's only role.
    entries.push(("clear".to_string(), CLEAR_COMMAND_DESC.to_string()));
    entries
}

#[cfg(test)]
#[path = "telegram_tests.rs"]
mod tests;
