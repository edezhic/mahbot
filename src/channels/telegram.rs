use crate::util::html::{decode_html_entities, escape_html, push_escaped};
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

/// Metadata for an incoming document or photo attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IncomingAttachment {
    file_id: String,
    file_name: Option<String>,
    file_size: Option<u64>,
    caption: Option<String>,
    kind: IncomingAttachmentKind,
}

/// The kind of incoming attachment (document vs photo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncomingAttachmentKind {
    Document,
    Photo,
    Voice,
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
        // If the remainder fits within chunk_limit, push the last chunk.
        // chunk_limit = TELEGRAM_MAX_MESSAGE_LENGTH - TELEGRAM_CONTINUATION_OVERHEAD = 4066.
        // The 30-char overhead covers the worst case (middle chunk: prefix + suffix).
        // For the last chunk, send_text_chunks only adds the "(continued)\n\n" prefix
        // (13 chars), so 4066+13=4079 ≤ 4096 — we have margin even for the last chunk.
        if remaining.chars().count() <= chunk_limit {
            chunks.push(remaining.to_string());
            break;
        }

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

fn extract_sender_username(message: &serde_json::Value) -> String {
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
            message_id: None,
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
            "IMAGE" | "PHOTO" => Some(Self::Image),
            "DOCUMENT" | "FILE" => Some(Self::Document),
            "VIDEO" => Some(Self::Video),
            "AUDIO" => Some(Self::Audio),
            "VOICE" => Some(Self::Voice),
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
/// multimodal pipeline can validate vision capability. Voice and audio
/// messages use `[AUDIO:/path]`. Other attachment types use
/// `[Document: name] /path`.
fn format_attachment_content(
    kind: IncomingAttachmentKind,
    local_filename: &str,
    local_path: &Path,
) -> String {
    match kind {
        IncomingAttachmentKind::Photo | IncomingAttachmentKind::Document
            if is_image_extension(local_path) =>
        {
            format!("[IMAGE:{}]", local_path.display())
        }
        IncomingAttachmentKind::Voice => {
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

fn find_matching_close(s: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_attachment_markers(message: &str) -> (String, Vec<TelegramAttachment>) {
    let mut cleaned = String::with_capacity(message.len());
    let mut attachments = Vec::new();
    let mut cursor = 0;

    while cursor < message.len() {
        let Some(open_rel) = message[cursor..].find('[') else {
            cleaned.push_str(&message[cursor..]);
            break;
        };

        let open = cursor + open_rel;
        cleaned.push_str(&message[cursor..open]);

        let Some(close_rel) = find_matching_close(&message[open + 1..]) else {
            cleaned.push_str(&message[open..]);
            break;
        };

        let close = open + 1 + close_rel;
        let marker = &message[open + 1..close];

        let parsed = marker.split_once(':').and_then(|(kind, target)| {
            let kind = TelegramAttachmentKind::from_marker(kind)?;
            let target = target.trim();
            if target.is_empty() {
                return None;
            }
            Some(TelegramAttachment {
                kind,
                target: target.to_string(),
            })
        });

        if let Some(attachment) = parsed {
            attachments.push(attachment);
        } else {
            cleaned.push_str(&message[open..=close]);
        }

        cursor = close + 1;
    }

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
    let username = extract_sender_username(sender_source);
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
                code_buf.clear();
            } else {
                in_code_block = true;
                code_buf.clear();
            }
            continue;
        }

        if in_code_block {
            code_buf.push_str(line);
            code_buf.push('\n');
            continue;
        }

        // ── Blockquotes — pass through as-is ──────────────────
        if trimmed.starts_with("<blockquote") || trimmed == "</blockquote>" {
            out.push_str(trimmed);
            out.push('\n');
            continue;
        }

        // ── Headers: ## Title → <b>Title</b> ───────────────────
        let stripped = line.trim_start_matches('#');
        let header_level = line.len() - stripped.len();
        if header_level > 0 && line.starts_with('#') && stripped.starts_with(' ') {
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
    async fn parse_callback_query(&self, update: &serde_json::Value) -> Option<ChannelMessage> {
        let cq = update.get("callback_query")?;
        let data = cq.get("data").and_then(serde_json::Value::as_str)?;
        let msg = cq.get("message")?;
        let callback_query_id = cq
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        // chat_id is intentionally unused here: callback queries don't need
        // to route replies by chat context, only the reply_target is used.
        let (user_name, _, reply_target) = resolve_authorized_sender(cq, msg).await?;

        Some(ChannelMessage {
            user_name,
            reply_target,
            content: data.to_string(),
            source_channel: "telegram".to_string(),
            workspace: String::new(),
            message_id: None,
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
                extract_sender_username(message)
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
                let err_body = response.text().await.unwrap_or_else(|e| {
                    tracing::warn!(?e, "Failed to read ACK reaction error response body");
                    "failed to read response body".to_string()
                });
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
    /// resolution), `audio`, and `voice`.  Audio maps to [`IncomingAttachmentKind::Voice`]
    /// because there is no separate audio variant.  Returns `None` for text‑only
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

        // Audio — maps to Voice kind (no separate Audio variant)
        if let Some(audio) = message.get("audio") {
            return Self::build_attachment(audio, message, IncomingAttachmentKind::Voice);
        }

        // Voice message
        if let Some(voice) = message.get("voice") {
            return Self::build_attachment(voice, message, IncomingAttachmentKind::Voice);
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
        Some(IncomingAttachment {
            file_id,
            file_name,
            file_size,
            caption,
            kind,
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
            let ext = tg_file_path.rsplit('.').next().unwrap_or("jpg");
            format!("photo_{}_{}.{ext}", ctx.chat_id, ctx.message_id)
        };

        let local_path = save_dir.join(&local_filename);
        if let Err(e) = tokio::fs::write(&local_path, &file_data).await {
            tracing::warn!("Failed to save attachment to {}: {e}", local_path.display());
            return None;
        }

        let mut content = format_attachment_content(attachment.kind, &local_filename, &local_path);
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
            let err_body = resp.text().await.unwrap_or_else(|e| {
                tracing::warn!(?e, "Failed to read sendMessage error response body");
                "failed to read response body".to_string()
            });
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

            // Attach the inline keyboard only to the last chunk.
            let chunk_reply_markup = if index == chunks.len() - 1 {
                reply_markup.clone()
            } else {
                None
            };

            // Try HTML first, fall back to plain text on failure.
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
            if update.get("callback_query").is_some() {
                let cq_id = update["callback_query"]["id"]
                    .as_str()
                    .map(ToString::to_string);
                let cq_data = update["callback_query"]["data"].as_str().unwrap_or("");

                // For __act__ callbacks, do NOT answer early — the action handler
                // (handle_action_callback in main.rs) will answer with the appropriate
                // toast text. For __opt__ and other callbacks, dismiss the spinner now.
                if !cq_data.starts_with("__act__")
                    && let Some(ref id) = cq_id
                {
                    self.answer_callback_query(id, None).await;
                }

                let Some(msg) = self.parse_callback_query(&update).await else {
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
                () = tokio::time::sleep(Duration::from_secs(90)) => {
                    tracing::warn!("getUpdates poll timed out (90s), retrying");
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
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn telegram_api_url() {
        let ch = TelegramChannel::new("123:ABC".into());
        assert_eq!(
            ch.api_url("getMe"),
            "https://api.telegram.org/bot123:ABC/getMe"
        );
    }

    #[test]
    fn parse_recipient_parses_chat_id_only() {
        assert_eq!(parse_recipient("12345"), ("12345", None));
    }

    #[test]
    fn parse_recipient_parses_chat_id_with_thread() {
        assert_eq!(parse_recipient("12345:678"), ("12345", Some("678")));
    }

    #[test]
    fn parse_recipient_handles_empty_string() {
        assert_eq!(parse_recipient(""), ("", None));
    }

    #[test]
    fn test_markdown_to_telegram_html() {
        // escapes quotes in link href
        let r = markdown_to_telegram_html("[click](https://example.com?q=\"x\"&a='b')");
        assert_eq!(
            r,
            "<a href=\"https://example.com?q=&quot;x&quot;&amp;a=&#39;b&#39;\">click</a>"
        );
        // escapes quotes/ampersand in plain text
        let r = markdown_to_telegram_html("say \"hi\" & <tag> 'ok'");
        assert_eq!(r, "say &quot;hi&quot; &amp; &lt;tag&gt; &#39;ok&#39;");
        // drops language attribute from code blocks
        let r = markdown_to_telegram_html("```rust\" onclick=\"alert(1)\nlet x = 1;\n```");
        assert_eq!(r, "<pre><code>let x = 1;</code></pre>");
        assert!(!r.contains("language-"));
        assert!(!r.contains("onclick"));

        // Inline formatting inside code blocks is preserved literally
        let r = markdown_to_telegram_html("```\nsome **bold** and `code`\n```");
        assert_eq!(r, "<pre><code>some **bold** and `code`</code></pre>");

        // HTML special characters in code blocks are escaped
        let r = markdown_to_telegram_html("```\n<div> & \"it\" 'works'\n```");
        assert_eq!(
            r,
            "<pre><code>&lt;div&gt; &amp; &quot;it&quot; &#39;works&#39;</code></pre>"
        );

        // Literal </code> in code block must not break the HTML
        let r = markdown_to_telegram_html("```\nuse &lt;/code&gt;\n```");
        assert_eq!(r, "<pre><code>use &amp;lt;/code&amp;gt;</code></pre>");
    }

    // ── Inline formatting tests ──────────────────────────────────────

    #[test]
    fn test_inline_formatting() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: &'static str,
        }
        let cases = vec![
            Case {
                name: "bold double asterisk",
                input: "**hello** world",
                expected: "<b>hello</b> world",
            },
            Case {
                name: "bold double underscore",
                input: "__hello__ world",
                expected: "<b>hello</b> world",
            },
            Case {
                name: "italic",
                input: "*hello* world",
                expected: "<i>hello</i> world",
            },
            Case {
                name: "inline code",
                input: "use `hello()` in your code",
                expected: "use <code>hello()</code> in your code",
            },
            Case {
                name: "strikethrough",
                input: "this is ~~wrong~~ fixed",
                expected: "this is <s>wrong</s> fixed",
            },
            Case {
                name: "combined",
                input: "**bold** and *italic* and `code` and ~~strike~~",
                expected: "<b>bold</b> and <i>italic</i> and <code>code</code> and <s>strike</s>",
            },
            Case {
                name: "bold inside text",
                input: "before **middle** after",
                expected: "before <b>middle</b> after",
            },
            Case {
                name: "escaping inner HTML",
                input: "**a < b & c > d**",
                expected: "<b>a &lt; b &amp; c &gt; d</b>",
            },
            // `**` without closing should be rendered literally
            Case {
                name: "unmatched double asterisk",
                input: "hello ** world",
                expected: "hello ** world",
            },
            // `*` without closing should be rendered literally
            Case {
                name: "unmatched single asterisk",
                input: "hello * world",
                expected: "hello * world",
            },
            // `***` is not a valid bold or italic construct; rendered literally
            Case {
                name: "triple asterisk",
                input: "***",
                expected: "***",
            },
            // `` ` `` without closing should be rendered literally (the opening ` is pushed as text)
            Case {
                name: "unmatched backtick",
                input: "hello ` world",
                expected: "hello ` world",
            },
            // ` `` ` (two backticks) — the first opens, the second closes (empty content), and since
            // the `end > 0` guard rejects empty matches, both are rendered literally.
            Case {
                name: "double backtick",
                input: "hello `` world",
                expected: "hello `` world",
            },
            // `~` without matching pair should be rendered literally
            Case {
                name: "unmatched tilde",
                input: "hello ~ world",
                expected: "hello ~ world",
            },
            // Bold takes priority over italic for `**`
            Case {
                name: "bold and italic overlap",
                input: "***bold**",
                expected: "<b>*bold</b>",
            },
        ];
        for case in cases {
            let result = markdown_to_telegram_html(case.input);
            assert_eq!(result, case.expected, "case: {}", case.name);
        }
    }

    #[tokio::test]
    async fn parse_update_message_uses_chat_id_as_reply_target() {
        crate::users::test_util::init_test_store().await;
        let ch = TelegramChannel::new("token".into());
        let update = serde_json::json!({
            "update_id": 1,
            "message": {
                "message_id": 33,
                "text": "hello",
                "from": {
                    "id": 555,
                    "username": "alice"
                },
                "chat": {
                    "id": -100_200_300
                }
            }
        });

        let msg = ch
            .parse_update_message(&update)
            .await
            .expect("message should parse");

        assert_eq!(msg.user_name, "alice");
        assert_eq!(msg.reply_target, "-100200300");
        assert_eq!(msg.content, "hello");
    }

    #[tokio::test]
    async fn parse_attachment_markers_tests() {
        let (cleaned, att) = parse_attachment_markers(
            "Here are files [IMAGE:/tmp/a.png] and [DOCUMENT:https://example.com/a.pdf]",
        );
        assert_eq!(cleaned, "Here are files  and");
        assert_eq!(att.len(), 2);
        assert_eq!(att[0].kind, TelegramAttachmentKind::Image);
        assert_eq!(att[1].kind, TelegramAttachmentKind::Document);
        // invalid markers kept as text
        let (cleaned, att) = parse_attachment_markers("Report [UNKNOWN:/tmp/a.bin]");
        assert_eq!(cleaned, "Report [UNKNOWN:/tmp/a.bin]");
        assert!(att.is_empty());
    }

    #[tokio::test]
    async fn parse_path_only_attachment_tests() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("snap.png");
        std::fs::write(&p, b"fake-png").unwrap();
        let parsed = parse_path_only_attachment(p.to_string_lossy().as_ref()).unwrap();
        assert_eq!(parsed.kind, TelegramAttachmentKind::Image);
        assert_eq!(parsed.target, p.to_string_lossy());
        assert!(parse_path_only_attachment("Screenshot saved to /tmp/snap.png").is_none());
    }

    #[test]
    fn infer_attachment_kind_from_target_detects_document_extension() {
        assert_eq!(
            infer_attachment_kind_from_target("https://example.com/files/specs.pdf?download=1"),
            Some(TelegramAttachmentKind::Document)
        );
    }

    #[tokio::test]
    async fn parse_update_message_denies_user_without_username() {
        crate::users::test_util::init_test_store().await;
        let ch = TelegramChannel::new("token".into());
        let update = serde_json::json!({
            "update_id": 2,
            "message": {
                "message_id": 9,
                "text": "ping",
                "from": {
                    "id": 555
                },
                "chat": {
                    "id": 12345
                }
            }
        });

        assert!(
            ch.parse_update_message(&update).await.is_none(),
            "user without username should be denied"
        );
    }

    #[tokio::test]
    async fn parse_update_message_extracts_thread_id_for_forum_topic() {
        crate::users::test_util::init_test_store().await;
        let ch = TelegramChannel::new("token".into());
        let update = serde_json::json!({
            "update_id": 3,
            "message": {
                "message_id": 42,
                "text": "hello from topic",
                "from": {
                    "id": 555,
                    "username": "alice"
                },
                "chat": {
                    "id": -100_200_300
                },
                "message_thread_id": 789
            }
        });

        let msg = ch
            .parse_update_message(&update)
            .await
            .expect("message with thread_id should parse");

        assert_eq!(msg.user_name, "alice");
        assert_eq!(msg.reply_target, "-100200300:789");
        assert_eq!(msg.content, "hello from topic");
    } // ── Message splitting tests ─────────────────────────────────────

    #[test]
    fn telegram_message_splitting() {
        // basic: exact limit → no split
        assert_eq!(
            split_message_for_telegram(&"a".repeat(TELEGRAM_MAX_MESSAGE_LENGTH)).len(),
            1
        );
        assert!(
            split_message_for_telegram(&"a".repeat(TELEGRAM_MAX_MESSAGE_LENGTH + 1)).len() >= 2
        );
        let long = "a".repeat(5000);
        let parts = split_message_for_telegram(&long);
        assert!(parts.len() >= 2);
        assert_eq!(parts.join(""), long);
        assert!(split_message_for_telegram("   \n\n\t  ").len() <= 1);

        // edge: code block spanning boundary
        let msg = format!("```python\n{}```\nMore text", "x".repeat(4085));
        for p in &split_message_for_telegram(&msg) {
            assert!(p.len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
        }
        // emoji at boundary
        let msg = format!("{}🎉🎊", "a".repeat(4094));
        for p in &split_message_for_telegram(&msg) {
            assert!(p.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH);
        }
    }

    #[test]
    fn newline_split_fallback_prevents_mid_word_break() {
        // Regression: when the only newline is in the first half of the
        // search window and no spaces exist, the old code would hard-split
        // mid-word. The newline fallback (tier 3) prevents this.
        let msg = format!("{}\n{}", "a".repeat(1000), "x".repeat(5000));
        let chunks = split_message_for_telegram(&msg);

        // All chunks must respect the length limit
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH,
                "chunk {} has {} chars (limit {})",
                i,
                chunk.chars().count(),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            );
        }

        // Concatenation must reconstruct the original message
        assert_eq!(chunks.join(""), msg);

        // The first chunk must end with the newline (not split mid-word)
        assert!(
            chunks[0].ends_with('\n'),
            "first chunk should end with newline, got: {:?}",
            chunks[0].chars().rev().take(10).collect::<String>()
        );
    }

    #[test]
    fn wrapped_chunks_respect_telegram_limit() {
        // Simulate send_text_chunks wrapping to verify no chunk exceeds the
        // 4096-char limit after continuation prefixes are prepended.
        //
        // Middle chunk overhead: "(continued)\n\n...\n\n(continues...)" = 28 chars
        // Last chunk overhead:  "(continued)\n\n..."                   = 13 chars
        // First chunk (multi):  "...\n\n(continues...)"               = 15 chars

        // Build a message designed to force 3+ chunks, then verify wrapping.
        let msg = format!("X{}X", "a".repeat(9000));
        let chunks = split_message_for_telegram(&msg);
        assert!(
            chunks.len() >= 3,
            "expected 3+ chunks to exercise all continuation variants"
        );

        for (i, chunk) in chunks.iter().enumerate() {
            let wrapped = wrap_chunk(chunk, i, chunks.len());
            assert!(
                wrapped.chars().count() <= 4096,
                "chunk {} wrapped length {} exceeds 4096",
                i,
                wrapped.chars().count()
            );
        }

        // Boundary: last chunk exactly at 4066 chars (new limit) — wrapping
        // produces 4066+13=4079 ≤ 4096.  Under the old code, a 4095-char last
        // chunk would wrap to 4108 > 4096.
        //
        // Intentionally uses raw format! rather than wrap_chunk() — this tests
        // the splitter's TELEGRAM_CONTINUATION_OVERHEAD constant, not the
        // wrapping helper. wrap_chunk would return the chunk bare for total==1.
        let boundary = "b".repeat(4066);
        let chunks = split_message_for_telegram(&boundary);
        assert_eq!(chunks.len(), 1, "4066-char message should not split");
        let wrapped = format!("(continued)\n\n{}", chunks[0]);
        assert!(
            wrapped.chars().count() <= 4096,
            "boundary wrapped: {} > 4096",
            wrapped.chars().count()
        );

        // Old bug reproducer: a 4095-char last chunk would wrap to 4108.
        // With the fix, a message that produces a ~4095-char last chunk
        // shouldn't exist; the splitter caps non-first chunks at 4066.
        let near_limit = "c".repeat(4096);
        let chunks = split_message_for_telegram(&near_limit);
        assert_eq!(chunks.len(), 1, "4096-char message should not split");
    }

    // ─────────────────────────────────────────────────────────────────────
    // extract_sender_username tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_extract_sender_username() {
        let username =
            extract_sender_username(&serde_json::json!({"from": {"id": 123, "username": "alice"}}));
        assert_eq!(username, "alice");
        let username = extract_sender_username(&serde_json::json!({"from": {"id": 42}}));
        assert_eq!(username, "unknown");
    }

    // ─────────────────────────────────────────────────────────────────────
    // extract_reply_context tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn extract_reply_context() {
        // text message reply
        let msg = serde_json::json!({
            "reply_to_message": {
                "from": { "username": "alice" },
                "text": "Hello world"
            }
        });
        let ctx = TelegramChannel::extract_reply_context(&msg).unwrap();
        assert_eq!(ctx, "> @alice:\n> Hello world");

        // voice message reply
        let msg = serde_json::json!({
            "reply_to_message": {
                "from": { "username": "bob" },
                "voice": { "file_id": "abc", "duration": 5 }
            }
        });
        let ctx = TelegramChannel::extract_reply_context(&msg).unwrap();
        assert_eq!(ctx, "> @bob:\n> [Voice message]");

        // no reply
        let msg = serde_json::json!({
            "text": "just a regular message"
        });
        assert!(TelegramChannel::extract_reply_context(&msg).is_none());

        // no username, uses first_name
        let msg = serde_json::json!({
            "reply_to_message": {
                "from": { "id": 999, "first_name": "Charlie" },
                "text": "Hi there"
            }
        });
        let ctx = TelegramChannel::extract_reply_context(&msg).unwrap();
        assert_eq!(ctx, "> Charlie:\n> Hi there");
    }

    #[tokio::test]
    async fn parse_update_message_includes_reply_context() {
        crate::users::test_util::init_test_store().await;
        let ch = TelegramChannel::new("t".into());
        let update = serde_json::json!({
            "message": {
                "message_id": 10,
                "text": "translate this",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 100, "type": "private" },
                "reply_to_message": {
                    "from": { "username": "bot" },
                    "text": "Bonjour le monde"
                }
            }
        });
        let parsed = ch.parse_update_message(&update).await.unwrap();
        assert!(
            parsed.content.starts_with("> @bot:"),
            "content should start with quote: {}",
            parsed.content
        );
        assert!(
            parsed.content.contains("translate this"),
            "content should contain user text"
        );
        assert!(
            parsed.content.contains("Bonjour le monde"),
            "content should contain quoted text"
        );
    }

    // ── IncomingAttachment / parse_attachment_metadata tests ─────────

    #[tokio::test]
    async fn parse_attachment_metadata() {
        // Document with all fields
        let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "document": {"file_id": "BQ", "file_name": "report.pdf", "file_size": 12345}
        }))
        .unwrap();
        assert_eq!(att.kind, IncomingAttachmentKind::Document);
        assert_eq!(att.file_id, "BQ");
        assert_eq!(att.file_name.as_deref(), Some("report.pdf"));
        assert_eq!(att.file_size, Some(12345));
        assert!(att.caption.is_none());
        // Photo (picks largest by file_size)
        let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "photo": [{"file_id": "small_id", "file_size": 100}, {"file_id": "large_id", "file_size": 2000}]
        })).unwrap();
        assert_eq!(att.kind, IncomingAttachmentKind::Photo);
        assert_eq!(att.file_id, "large_id");
        assert_eq!(att.file_size, Some(2000));
        // Caption extraction
        let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "document": {"file_id": "doc_id", "file_name": "data.csv"}, "caption": "Monthly report"
        }))
        .unwrap();
        assert_eq!(att.caption.as_deref(), Some("Monthly report"));
        let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "photo": [{"file_id": "photo_id", "file_size": 1000}], "caption": "Look at this"
        }))
        .unwrap();
        assert_eq!(att.caption.as_deref(), Some("Look at this"));
        // Document without optional fields
        let att = TelegramChannel::parse_attachment_metadata(&serde_json::json!({
            "document": {"file_id": "doc_no_name"}
        }))
        .unwrap();
        assert_eq!(att.kind, IncomingAttachmentKind::Document);
        assert_eq!(att.file_id, "doc_no_name");
        assert!(att.file_name.is_none());
        assert!(att.file_size.is_none());
        // Voice message
        let att = TelegramChannel::parse_attachment_metadata(
            &serde_json::json!({"voice": {"file_id": "v", "duration": 5}}),
        )
        .unwrap();
        assert_eq!(att.kind, IncomingAttachmentKind::Voice);
        assert_eq!(att.file_id, "v");
        assert!(att.file_name.is_none());
        // Audio message
        let att = TelegramChannel::parse_attachment_metadata(
            &serde_json::json!({"audio": {"file_id": "a", "file_name": "song.mp3", "file_size": 999}}),
        )
        .unwrap();
        assert_eq!(att.kind, IncomingAttachmentKind::Voice);
        assert_eq!(att.file_id, "a");
        assert_eq!(att.file_name.as_deref(), Some("song.mp3"));
        assert_eq!(att.file_size, Some(999));
        // No attachment cases
        assert!(
            TelegramChannel::parse_attachment_metadata(&serde_json::json!({"text": "Hello"}))
                .is_none()
        );
        assert!(
            TelegramChannel::parse_attachment_metadata(&serde_json::json!({"photo": []})).is_none()
        );
    }

    // ── Attachment content format tests ──────────────────────────────

    #[tokio::test]
    async fn attachment_content_format_rules() {
        // photo → [IMAGE:]
        let c = format_attachment_content(
            IncomingAttachmentKind::Photo,
            "photo.jpg",
            std::path::Path::new("/tmp/workspace/photo.jpg"),
        );
        assert_eq!(c, "[IMAGE:/tmp/workspace/photo.jpg]");
        // document → [Document: name] /path
        let c = format_attachment_content(
            IncomingAttachmentKind::Document,
            "report.pdf",
            std::path::Path::new("/tmp/workspace/report.pdf"),
        );
        assert_eq!(c, "[Document: report.pdf] /tmp/workspace/report.pdf");
        assert!(!c.contains("[IMAGE:"));
        // markdown files never produce [IMAGE:] even when classified as Photo
        let c = format_attachment_content(
            IncomingAttachmentKind::Photo,
            "notes.md",
            std::path::Path::new("/tmp/workspace/notes.md"),
        );
        assert!(!c.contains("[IMAGE:"));
        assert!(c.starts_with("[Document:"));
        // non-image files classified as Photo fall back to [Document:]
        for (filename, path) in [
            ("file.md", "/tmp/workspace/file.md"),
            ("file.txt", "/tmp/workspace/file.txt"),
            ("file.pdf", "/tmp/workspace/file.pdf"),
            ("file.csv", "/tmp/workspace/file.csv"),
            ("file.json", "/tmp/workspace/file.json"),
            ("file.zip", "/tmp/workspace/file.zip"),
            ("file", "/tmp/workspace/file"),
        ] {
            let c = format_attachment_content(
                IncomingAttachmentKind::Photo,
                filename,
                std::path::Path::new(path),
            );
            assert!(
                !c.contains("[IMAGE:"),
                "{filename}: should not get [IMAGE:]"
            );
            assert!(
                c.starts_with("[Document:"),
                "{filename}: should use [Document:]"
            );
        }
        // image extensions produce [IMAGE:]
        for ext in ["png", "jpg", "jpeg", "gif", "webp", "bmp"] {
            let filename = format!("photo.{ext}");
            let c = format_attachment_content(
                IncomingAttachmentKind::Photo,
                &filename,
                std::path::Path::new(&format!("/tmp/workspace/{filename}")),
            );
            assert!(c.starts_with("[IMAGE:"), "{ext}: should get [IMAGE:]");
        }
    }

    #[tokio::test]
    async fn attachment_multimodal_and_helpers() {
        // is_image_extension
        for p in [
            "photo.png",
            "photo.jpg",
            "photo.jpeg",
            "photo.gif",
            "photo.webp",
            "photo.bmp",
            "PHOTO.PNG",
        ] {
            assert!(is_image_extension(std::path::Path::new(p)));
        }
        for p in ["file.md", "file.txt", "file.pdf", "file.csv", "file"] {
            assert!(!is_image_extension(std::path::Path::new(p)));
        }
        // photo with caption
        let content = format!(
            "[IMAGE:{}]\n\nLook at this screenshot",
            std::path::Path::new("/tmp/workspace/photo.jpg").display()
        );
        assert_eq!(
            content,
            "[IMAGE:/tmp/workspace/photo.jpg]\n\nLook at this screenshot"
        );
    }

    // ── Forwarded message tests ─────────────────────────────────────

    #[tokio::test]
    async fn forward_attribution() {
        crate::users::test_util::init_test_store().await;
        let ch = TelegramChannel::new("token".into());

        // forwarded from user with username
        let update = serde_json::json!({
            "update_id": 100,
            "message": {
                "message_id": 50,
                "text": "Check this out",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 },
                "forward_from": {
                    "id": 42,
                    "first_name": "Bob",
                    "username": "bob"
                },
                "forward_date": 1_700_000_000
            }
        });
        let msg = ch.parse_update_message(&update).await.unwrap();
        assert_eq!(msg.content, "[Forwarded from @bob] Check this out");

        // forwarded from channel
        let update = serde_json::json!({
            "update_id": 101,
            "message": {
                "message_id": 51,
                "text": "Breaking news",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 },
                "forward_from_chat": {
                    "id": -1_001_234_567_890_i64,
                    "title": "Daily News",
                    "username": "dailynews",
                    "type": "channel"
                },
                "forward_date": 1_700_000_000
            }
        });
        let msg = ch.parse_update_message(&update).await.unwrap();
        assert_eq!(
            msg.content,
            "[Forwarded from channel: Daily News] Breaking news"
        );

        // forwarded hidden sender
        let update = serde_json::json!({
            "update_id": 102,
            "message": {
                "message_id": 52,
                "text": "Secret tip",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 },
                "forward_sender_name": "Hidden User",
                "forward_date": 1_700_000_000
            }
        });
        let msg = ch.parse_update_message(&update).await.unwrap();
        assert_eq!(msg.content, "[Forwarded from Hidden User] Secret tip");

        // non-forwarded unaffected
        let update = serde_json::json!({
            "update_id": 103,
            "message": {
                "message_id": 53,
                "text": "Normal message",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 }
            }
        });
        let msg = ch.parse_update_message(&update).await.unwrap();
        assert_eq!(msg.content, "Normal message");

        // forwarded from user without username
        let update = serde_json::json!({
            "update_id": 104,
            "message": {
                "message_id": 54,
                "text": "Hello there",
                "from": { "id": 1, "username": "alice" },
                "chat": { "id": 999 },
                "forward_from": {
                    "id": 77,
                    "first_name": "Charlie"
                },
                "forward_date": 1_700_000_000
            }
        });
        let msg = ch.parse_update_message(&update).await.unwrap();
        assert_eq!(msg.content, "[Forwarded from Charlie] Hello there");

        // forwarded photo with attribution
        let message = serde_json::json!({
            "message_id": 60,
            "from": { "id": 1, "username": "alice" },
            "chat": { "id": 999 },
            "photo": [
                { "file_id": "abc123", "file_unique_id": "u1", "width": 320, "height": 240 }
            ],
            "forward_from": {
                "id": 42,
                "username": "bob"
            },
            "forward_date": 1_700_000_000
        });
        let attr =
            TelegramChannel::format_forward_attribution(&message).expect("should detect forward");
        assert_eq!(attr, "[Forwarded from @bob] ");
        let photo_content = "[IMAGE:/tmp/photo.jpg]".to_string();
        let content = format!("{attr}{photo_content}");
        assert_eq!(content, "[Forwarded from @bob] [IMAGE:/tmp/photo.jpg]");
    }

    // ── strip_html_tags tests ──────────────────────────────────

    #[test]
    fn test_strip_html_tags() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: &'static str,
        }
        let cases = vec![
            Case {
                name: "empty string",
                input: "",
                expected: "",
            },
            Case {
                name: "plain text",
                input: "hello world",
                expected: "hello world",
            },
            Case {
                name: "simple tag",
                input: "<b>bold</b>",
                expected: "bold",
            },
            Case {
                name: "nested tags",
                input: "<div><span>text</span></div>",
                expected: "text",
            },
            Case {
                name: "self-closing tag",
                input: "before<br/>after",
                expected: "beforeafter",
            },
            // Regression: '<' in tag starts in_tag, then '>' inside attribute
            // value must NOT close the tag.
            Case {
                name: "gt in double-quoted attribute",
                input: "<a title=\"a > b\">link</a>",
                expected: "link",
            },
            Case {
                name: "gt in single-quoted attribute",
                input: "<a title='a > b'>link</a>",
                expected: "link",
            },
            // Double-quoted attribute containing single quotes
            Case {
                name: "mixed quotes - double with single inside",
                input: "<a title=\"he said 'hello'\">text</a>",
                expected: "text",
            },
            // Single-quoted attribute containing double quotes
            Case {
                name: "mixed quotes - single with double inside",
                input: "<a title='he said \"hello\"'>text</a>",
                expected: "text",
            },
            // Multiple attributes, some with '>' inside quoted values
            Case {
                name: "multiple attrs with gt",
                input: "<input type=\"text\" value=\"a > b\" placeholder=\"x > y\">",
                expected: "",
            },
            // '>' that is not inside a tag should be preserved as text
            Case {
                name: "gt outside tag",
                input: "a > b",
                expected: "a > b",
            },
            // '<' that is not part of a tag should start tag mode
            // (pre-existing behavior: bare '<' starts tag stripping)
            Case {
                name: "lt outside tag",
                input: "a < b",
                expected: "a ",
            },
            Case {
                name: "html comment",
                input: "<!-- comment -->visible",
                expected: "visible",
            },
            // Realistic mixed content with tags and text
            Case {
                name: "mixed content",
                input: "Hello <b>world</b>, check <a href=\"https://example.com?q=a > b\">this</a> out!",
                expected: "Hello world, check this out!",
            },
        ];
        for case in cases {
            let result = strip_html_tags(case.input);
            assert_eq!(result, case.expected, "case: {}", case.name);
        }
    }

    // ── extend_past_open_tag tests ─────────────────────────────

    #[test]
    fn test_extend_past_open_tag() {
        struct Case {
            name: &'static str,
            input: &'static str,
            pos: usize,
            expected: Option<usize>,
        }
        let cases = vec![
            // No '<' before pos → None
            Case {
                name: "no tag near pos",
                input: "hello world",
                pos: 5,
                expected: None,
            },
            // <b>hello
            // 01234567
            // pos=1 (inside <b>, before '>') → extend past '>' at 2
            Case {
                name: "inside simple tag before gt",
                input: "<b>hello",
                pos: 1,
                expected: Some(3),
            },
            // pos=2 (at the '>' itself) → extend past it
            Case {
                name: "inside simple tag at gt",
                input: "<b>hello",
                pos: 2,
                expected: Some(3),
            },
            // <b>hello
            // 01234567
            // pos=3 (at 'h', past the '>') → None
            Case {
                name: "after simple tag at h",
                input: "<b>hello",
                pos: 3,
                expected: None,
            },
            // pos further into text → None
            Case {
                name: "after simple tag further",
                input: "<b>hello",
                pos: 5,
                expected: None,
            },
            // No '>' exists after the '<' → None (can't extend)
            Case {
                name: "no closing gt",
                input: "<div",
                pos: 3,
                expected: None,
            },
            // Regression: '>' inside double-quoted attribute must not be
            // treated as tag closer. The real '>' is at index 16.
            // <a title="a > b">text
            // 012345678901234567890   (indices)
            //           111111111122
            //      '>' at 12 is inside attribute, real '>' at 16
            // pos=13 (inside tag, past the quoted '>', before real '>') → extend to 17
            Case {
                name: "gt in double-quoted attr before real gt",
                input: "<a title=\"a > b\">text",
                pos: 13,
                expected: Some(17),
            },
            // pos=16 (at the real '>') → extend past it
            Case {
                name: "gt in double-quoted attr at real gt",
                input: "<a title=\"a > b\">text",
                pos: 16,
                expected: Some(17),
            },
            // pos after the real closing '>' → None
            // <a title="a > b">text
            // 012345678901234567890
            //           111111111122
            // real '>' at 16, 't' at 17
            Case {
                name: "after closed tag with gt in attr at 17",
                input: "<a title=\"a > b\">text",
                pos: 17,
                expected: None,
            },
            Case {
                name: "after closed tag with gt in attr at 20",
                input: "<a title=\"a > b\">text",
                pos: 20,
                expected: None,
            },
            // Same scenario with single quotes
            // <a title='a > b'>text
            // real '>' at 16
            Case {
                name: "gt in single-quoted attr",
                input: "<a title='a > b'>text",
                pos: 13,
                expected: Some(17),
            },
            // Double-quoted attr containing single quotes: '>' inside should still be
            // treated as quoted because we're inside double quotes, not single.
            // <a title="he said 'stop'">text
            // real '>' at 25
            // pos=17 (inside the attribute, past the single quotes) → extend past real '>'
            Case {
                name: "mixed quotes",
                input: "<a title=\"he said 'stop'\">text",
                pos: 17,
                expected: Some(26),
            },
            // <div><span>text
            // 012345678901234
            // last '<' at 5 (<span>), its '>' at 10.
            // pos=11 (after both tags are closed) → None
            Case {
                name: "after nested tags at 11",
                input: "<div><span>text",
                pos: 11,
                expected: None,
            },
            // pos=15 (end of string, still after both tags) → None
            Case {
                name: "after nested tags at 15",
                input: "<div><span>text",
                pos: 15,
                expected: None,
            },
            // <div><span>text
            // pos=6 (inside <span>, before its '>') → extend past '>' at 10
            Case {
                name: "inside nested tag",
                input: "<div><span>text",
                pos: 6,
                expected: Some(11),
            },
            // |<b>text → pos=0 is before any '<'
            Case {
                name: "pos at start",
                input: "<b>text",
                pos: 0,
                expected: None,
            },
        ];
        for case in cases {
            let result = extend_past_open_tag(case.input, case.pos);
            assert_eq!(result, case.expected, "case: {}", case.name);
        }
    }
}
