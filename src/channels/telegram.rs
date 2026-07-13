use crate::util::html::{decode_html_entities, escape_html, push_escaped};
use crate::util::{MEDIA_MARKER_RE, TELEGRAM_MEDIA_MARKER_RE, parse_media_marker};
use crate::{Channel, ChannelMessage, SendMessage};
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

// ── Telegram callback/action button decoding ─────────────────────────

/// Callback data prefix for dynamic option buttons.
pub(crate) const CALLBACK_PREFIX: &str = "__opt__";

/// Decode callback data from inline keyboard interactions.
///
/// Returns `(ticket_id, label)` on success (`ticket_id` is `None` when the
/// callback data was generated without one).  Returns `None` when `content`
/// does not carry the `CALLBACK_PREFIX`.
///
/// # Format contract
///
/// The callback data uses `|` as a delimiter between the optional ticket-id
/// and the label.  The join and split therefore assume that `ticket_id` must
/// not contain `|`; the label may contain `|`.
///
/// **Examples:**
/// - `__opt__ticket-id|Label` → `(Some("ticket-id"), "Label")`
/// - `__opt__|Label` → `(None, "Label")`
/// - `__opt__BareLabel` → `(None, "BareLabel")`
#[must_use]
pub fn decode_callback(content: &str) -> Option<(Option<String>, String)> {
    let rest = content.strip_prefix(CALLBACK_PREFIX)?;
    Some(match rest.split_once('|') {
        Some((tid, lbl)) if !tid.is_empty() => (Some(tid.to_string()), lbl.to_string()),
        Some((_, lbl)) => (None, lbl.to_string()),
        None => (None, rest.to_string()),
    })
}

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

/// The kind of incoming attachment (document, photo, or audio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncomingAttachmentKind {
    Document,
    Photo,
    Audio,
}
/// Split a message into chunks that respect Telegram's 4096 character limit.
/// Tries to split at word boundaries when possible, and handles continuation.
/// The effective per-chunk limit is reduced to leave room for continuation markers.
/// When the input contains HTML tags, avoids splitting mid-tag.
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
        if let Some(adjusted) = extend_past_open_tag(remaining, chunk_end) {
            chunk_end = adjusted;
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
        .unwrap_or("unknown")
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
    fn into_channel_message(self, content: String) -> ChannelMessage {
        ChannelMessage {
            user_name: self.user_name,
            reply_target: self.reply_target,
            content,
            source_channel: "telegram".to_string(),
            workspace: String::new(),
            optimistic_id: None,
            callback_query_id: None,
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
            },
            Self::Document => AttachmentMeta {
                api_method: "sendDocument",
                form_field: "document",
                default_filename: "file",
                label: "Document",
            },
            Self::Video => AttachmentMeta {
                api_method: "sendVideo",
                form_field: "video",
                default_filename: "video.mp4",
                label: "Video",
            },
            Self::Audio => AttachmentMeta {
                api_method: "sendAudio",
                form_field: "audio",
                default_filename: "audio.mp3",
                label: "Audio",
            },
            Self::Voice => AttachmentMeta {
                api_method: "sendVoice",
                form_field: "voice",
                default_filename: "voice.ogg",
                label: "Voice",
            },
        }
    }
}

/// Recognized image file extensions.
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];

/// Check whether a file path has a recognized image extension.
fn is_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

/// Format a sender label for display: `@username` if a username is present,
/// otherwise the display name (first_name, or `"unknown"` as ultimate fallback).
#[must_use]
fn format_sender_label(from: &serde_json::Value) -> String {
    if let Some(username) = from.get("username").and_then(serde_json::Value::as_str) {
        format!("@{username}")
    } else {
        from.get("first_name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string()
    }
}

/// Build the user-facing content string for an incoming attachment.
///
/// Photos with a recognized image extension use `[IMAGE:/path]` so the
/// multimodal pipeline can validate vision capability.  When the extension
/// is not recognized the optional `mime_type` is consulted as a secondary
/// signal (e.g. Document + no extension + "image/jpeg" → still `[IMAGE:]`).
/// Voice and audio messages use `[AUDIO:/path]`. Other attachment types use
/// `[Document: name] /path`.
fn format_attachment_content(
    kind: IncomingAttachmentKind,
    local_filename: &str,
    local_path: &Path,
    mime_type: Option<&str>,
) -> String {
    let is_image =
        is_image_extension(local_path) || mime_type.is_some_and(|m| m.starts_with("image/"));
    match kind {
        IncomingAttachmentKind::Photo | IncomingAttachmentKind::Document if is_image => {
            format!("[IMAGE:{}]", local_path.display())
        }
        IncomingAttachmentKind::Audio => {
            format!("[AUDIO:{}]", local_path.display())
        }
        _ => {
            format!("[Document: {}] {}", local_filename, local_path.display())
        }
    }
}

fn is_http_url(target: &str) -> bool {
    target.starts_with("http://") || target.starts_with("https://")
}

fn infer_attachment_kind_from_target(target: &str) -> Option<TelegramAttachmentKind> {
    let normalized = target.split(['?', '#']).next().unwrap();

    let extension = Path::new(normalized)
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();

    match extension.as_str() {
        ext if IMAGE_EXTENSIONS.contains(&ext) => Some(TelegramAttachmentKind::Image),
        "mp4" | "mov" | "mkv" | "avi" | "webm" => Some(TelegramAttachmentKind::Video),
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

    if !is_http_url(candidate) && !Path::new(candidate).exists() {
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
/// Unknown or unrecognized markers are left intact in the cleaned text.
fn parse_attachment_markers(message: &str) -> (String, Vec<TelegramAttachment>) {
    let mut attachments: Vec<TelegramAttachment> = Vec::new();

    let cleaned = TELEGRAM_MEDIA_MARKER_RE
        .replace_all(message, |caps: &regex::Captures| {
            let (kind_str, path) = parse_media_marker(caps);
            let path = path.trim();

            // Preserve markers with whitespace-only paths (e.g. `[IMAGE: ]`)
            // as original text, mirroring the old hand-rolled parser's behavior
            // where `target.is_empty()` after trim caused the marker to be kept.
            if path.is_empty() {
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

/// Telegram Bot API maximum file download size (20 MB).
const TELEGRAM_MAX_FILE_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;

/// Telegram channel — long-polls the Bot API for updates
pub struct TelegramChannel {
    bot_token: String,
    /// Shared HTTP client with connection reuse across all Telegram API calls.
    http_client: reqwest::Client,

    /// Base URL for the Telegram Bot API. Defaults to `https://api.telegram.org`.
    /// Override for local Bot API servers or testing.
    api_base: String,

    /// Per-instance cancellation token — cancelling this stops only this
    /// channel's listener, not the entire application.
    cancel: std::sync::Arc<tokio_util::sync::CancellationToken>,

    /// Last confirmed `update_id + 1` offset. Shared across old/new listener
    /// instances during hot-reload so the new listener doesn't replay old
    /// updates from Telegram's server.
    offset: std::sync::Arc<std::sync::atomic::AtomicI64>,
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
                    if url.starts_with("http://") || url.starts_with("https://") {
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

impl TelegramChannel {
    /// Internal constructor shared by [`new`](Self::new) and
    /// [`with_offset`](Self::with_offset).
    #[must_use]
    fn new_with(bot_token: String, offset: std::sync::Arc<std::sync::atomic::AtomicI64>) -> Self {
        Self {
            bot_token,
            http_client: crate::util::http::build_http_client(Duration::from_mins(1)),
            api_base: "https://api.telegram.org".to_string(),
            cancel: std::sync::Arc::new(tokio_util::sync::CancellationToken::new()),
            offset,
        }
    }

    /// # Panics
    ///
    /// Panics if `reqwest::Client::build()` fails — typically due to TLS/OpenSSL
    /// initialization failure. Check your system's TLS library installation.
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
        match self
            .http_client()
            .post(self.api_url("answerCallbackQuery"))
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => {
                tracing::warn!(
                    callback_query_id = %callback_query_id,
                    status = %resp.status(),
                    "answerCallbackQuery returned non-success status"
                );
            }
            Err(e) => {
                tracing::warn!(
                    callback_query_id = %callback_query_id,
                    error = %e,
                    "Failed to send answerCallbackQuery"
                );
            }
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

        // chat_id is intentionally unused here: callback queries don't need
        // to route replies by chat context, only the reply_target is used.
        let Some((user_name, _, reply_target)) = resolve_authorized_sender(cq, msg).await else {
            tracing::debug!(
                "Telegram: ignoring callback query from unknown user '{}'",
                extract_sender_user_name(cq)
            );
            return None;
        };

        Some(ChannelMessage {
            user_name,
            reply_target,
            content: data.to_string(),
            source_channel: "telegram".to_string(),
            workspace: String::new(),
            optimistic_id: None,
            callback_query_id,
        })
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
        format!("{}/bot{}/{method}", self.api_base, self.bot_token)
    }

    /// Signal this specific channel's listener to stop, without affecting
    /// the global shutdown token or other channels.
    pub fn cancel_own(&self) {
        self.cancel.cancel();
    }

    /// Validate a Telegram bot token by calling the `getMe` endpoint.
    /// Returns `Ok(())` if the token is valid, `Err` with a descriptive
    /// message otherwise.
    pub async fn validate_token(token: &str) -> anyhow::Result<()> {
        if token.trim().is_empty() {
            anyhow::bail!("Telegram bot token is empty");
        }
        let url = format!("https://api.telegram.org/bot{token}/getMe");
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
        let url = format!("{}/file/bot{}/{file_path}", self.api_base, self.bot_token);
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
    /// resolution), `audio`, and `voice`.  Both map to [`IncomingAttachmentKind::Audio`]
    /// since there's no separate variant for each.  Returns `None` for text‑only
    /// and other unsupported message types.
    fn parse_attachment_metadata(message: &serde_json::Value) -> Option<IncomingAttachment> {
        // Document
        if let Some(doc) = message.get("document") {
            return Self::build_attachment(doc, message, IncomingAttachmentKind::Document);
        }

        // Photo (array of PhotoSize — take last = highest resolution)
        if let Some(photos) = message.get("photo").and_then(serde_json::Value::as_array) {
            let best = photos.last()?;
            return Self::build_attachment(best, message, IncomingAttachmentKind::Photo);
        }

        // Audio — maps to Audio kind (same variant handles both audio and voice)
        if let Some(audio) = message.get("audio") {
            return Self::build_attachment(audio, message, IncomingAttachmentKind::Audio);
        }

        // Voice message
        if let Some(voice) = message.get("voice") {
            return Self::build_attachment(voice, message, IncomingAttachmentKind::Audio);
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
        let save_dir = std::env::temp_dir().join("mahbot_telegram_files");
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
                .unwrap_or("jpg");
            format!("photo_{}_{}.{ext}", ctx.chat_id, ctx.message_id)
        };

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

        Some(ctx.into_channel_message(content))
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

        Some(ctx.into_channel_message(content))
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
            .http_client()
            .post(self.api_url("sendMessage"))
            .json(&body)
            .send()
            .await
            // Network/transport errors (connection refused, DNS failure, timeout) produce
            // no HTTP response, so we use BAD_GATEWAY as a sentinel — it signals an upstream
            // communication failure, not an actual HTTP-level error from the Telegram API.
            .map_err(|e| (reqwest::StatusCode::BAD_GATEWAY, e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            Ok(())
        } else {
            let err_body = crate::util::http::read_error_body(resp, "sendMessage error").await;
            Err((status, err_body))
        }
    }

    async fn send_text_chunks(
        &self,
        message: &str,
        chat_id: &str,
        thread_id: Option<&str>,
        reply_markup: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        // Decode HTML entities (e.g. &#39;) that LLM may emit, before
        // markdown-to-HTML conversion so they don't get double-escaped.
        let message = decode_html_entities(message);
        // Convert Markdown to Telegram HTML once, then split.
        let html = markdown_to_telegram_html(&message);
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
                tracing::warn!(
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
        kind: TelegramAttachmentKind,
        request: reqwest::RequestBuilder,
        label: &str,
    ) -> anyhow::Result<()> {
        let resp = request.send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram {} failed: {err}", kind.meta().api_method);
        }
        tracing::info!(
            "Telegram {} sent to {chat_id}: {label}",
            kind.meta().api_method
        );
        Ok(())
    }

    /// Send a media file (photo/document/video/audio/voice) to a Telegram chat.
    async fn send_media_file(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        kind: TelegramAttachmentKind,
        file_path: &Path,
    ) -> anyhow::Result<()> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_else(|| kind.meta().default_filename);

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part(kind.meta().form_field, part);

        if let Some(tid) = thread_id {
            form = form.text("message_thread_id", tid.to_string());
        }

        let request = self
            .http_client()
            .post(self.api_url(kind.meta().api_method))
            .multipart(form);

        self.send_media(chat_id, kind, request, file_name).await
    }

    /// Send a file by URL (Telegram will download it).
    async fn send_media_by_url(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        kind: TelegramAttachmentKind,
        url: &str,
    ) -> anyhow::Result<()> {
        let mut body = serde_json::json!({ "chat_id": chat_id });
        body[kind.meta().form_field] = serde_json::Value::String(url.to_string());

        set_thread_id_on_json(&mut body, thread_id);

        let request = self
            .http_client()
            .post(self.api_url(kind.meta().api_method))
            .json(&body);

        self.send_media(chat_id, kind, request, url).await
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
                tracing::warn!("Telegram poll error: {e}");
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
                // toast text. For __opt__ and other callbacks, dismiss the spinner now.
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

        // Look for inline attachment markers like [IMAGE:path/to/file.png]
        let (text_without_markers, attachments) = parse_attachment_markers(content);

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

        if let Some(attachment) = parse_path_only_attachment(content) {
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
        let new_channel: std::sync::Arc<dyn Channel> = if let Some(inherited_offset) = offset {
            std::sync::Arc::new(TelegramChannel::with_offset(
                token.to_string(),
                inherited_offset,
            ))
        } else {
            std::sync::Arc::new(TelegramChannel::new(token.to_string()))
        };

        // Atomically replace in the registry — no gap where
        // "telegram" returns None.
        registry.replace(std::sync::Arc::clone(&new_channel));

        // Cancel the old listener now that the registry has the new one.
        if let Some(old) = old_channel
            && let Some(tc) = old.as_any().downcast_ref::<TelegramChannel>()
        {
            tc.cancel_own();
        }

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
        if let Some(old) = old_channel
            && let Some(tc) = old.as_any().downcast_ref::<TelegramChannel>()
        {
            tc.cancel_own();
        }
        registry.unregister("telegram");
        tracing::info!("Telegram bot token cleared — listener stopped");
    }

    Ok(())
}

/// Mirror a GUI user's message to their Telegram chats as a blockquote, so conversation history is readable from both surfaces.
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
#[path = "telegram_tests.rs"]
mod tests;
