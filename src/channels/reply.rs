//! Reply-reference plumbing: the shared snippet normalization and marker
//! formatting used by both the GUI and the Telegram channel to render a
//! "reply to" quote header on a routed message.

use crate::Role;
use crate::util::TELEGRAM_MEDIA_MARKER_RE;
use crate::util::html::decode_html_entities;
use regex::Captures;
use serde::{Deserialize, Serialize};

/// Maximum length (in Unicode chars) of a normalized reply snippet.
pub const REPLY_SNIPPET_MAX_CHARS: usize = 100;

/// A reference to the message being replied to: the author's canonical name
/// and a short normalized snippet of what they said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplyReference {
    pub author: String,
    pub snippet: String,
}

/// Truncate `input` to at most `max_chars` chars, appending `…` only when a
/// truncation actually occurred. Unlike [`crate::util::truncate`] (which may
/// return `max_chars + 1` chars), the total length never exceeds `max_chars`.
#[must_use]
fn truncate_capped(input: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let byte_idx = input
        .char_indices()
        .nth(max_chars - 1)
        .map_or(input.len(), |(idx, _)| idx);
    format!("{}…", &input[..byte_idx])
}

/// Normalize a raw reply snippet into the deterministic shared form used by
/// both the GUI quote header and the Telegram channel.
///
/// Order matters (pinned by tests):
/// 1. Decode HTML entities (Telegram provenance).
/// 2. Strip a `<blockquote>…</blockquote>` wrapper (replies to GUI-mirrored
///    messages whose raw text is the mirror's own blockquote HTML).
/// 3. Remove all `<` / `>` characters.
/// 4. Collapse `\r\n` and `\n` to single spaces (other whitespace untouched).
/// 5. Map media markers inline (IMAGE→`[Photo]`, AUDIO→`[Voice message]`,
///    VIDEO→`[Video]`) via [`TELEGRAM_MEDIA_MARKER_RE`].
/// 6. Cap at [`REPLY_SNIPPET_MAX_CHARS`].
///
/// The mapping operates on the raw message content — callers invoke this
/// before any markdown preprocessing so a live data-URI marker still maps.
#[must_use]
pub fn normalize_reply_text(raw: &str) -> String {
    // 1. Decode HTML entities.
    let mut text = decode_html_entities(raw);

    // 2. Strip a `<blockquote>…</blockquote>` wrapper when present (the mirror
    //    format is `<blockquote>\n{content}\n</blockquote>`).
    let trimmed = text.trim();
    let unwrapped: String =
        if trimmed.starts_with("<blockquote>") && trimmed.ends_with("</blockquote>") {
            let start = "<blockquote>".len();
            let end = trimmed.len() - "</blockquote>".len();
            &trimmed[start..end]
        } else {
            trimmed
        }
        .trim()
        .to_string();
    text = unwrapped;

    // 3. Remove all angle brackets.
    text.retain(|c| c != '<' && c != '>');

    // 4. Collapse newlines to single spaces.
    text = text.replace("\r\n", " ").replace('\n', " ");

    // 5. Map media markers inline, preserving surrounding text. The regex
    //    recognizes only IMAGE|AUDIO|VIDEO, so the match is exhaustive.
    text = TELEGRAM_MEDIA_MARKER_RE
        .replace_all(&text, |caps: &Captures| {
            let kind = caps
                .name("kind")
                .expect("normalize_reply_text: media marker 'kind' group")
                .as_str()
                .to_ascii_uppercase();
            match kind.as_str() {
                "IMAGE" => "[Photo]".to_string(),
                "AUDIO" => "[Voice message]".to_string(),
                "VIDEO" => "[Video]".to_string(),
                _ => unreachable!("media marker regex matches only IMAGE|AUDIO|VIDEO"),
            }
        })
        .into_owned();

    // 6. Cap.
    truncate_capped(&text, REPLY_SNIPPET_MAX_CHARS)
}

/// Render the reply marker prefix prepended to a routed message. The trailing
/// colon is intentional (user-specified marker format).
#[must_use]
fn format_reply_marker(reply: &ReplyReference) -> String {
    format!(
        "<reply-to>{}: {}</reply-to>:\n",
        reply.author, reply.snippet
    )
}

/// Prepend the reply marker to routed content. The marker sits at the very
/// front, before any Telegram `[Forwarded from …]` prefix embedded in
/// `content`.
#[must_use]
pub fn apply_reply_marker(content: &str, reply: &ReplyReference) -> String {
    format!("{}{content}", format_reply_marker(reply))
}

/// Strip a trailing `_<digits>` role suffix (e.g. `"analyst_3"` → `"analyst"`).
///
/// The rule is `rsplit_once('_')` where the suffix consists solely of ASCII
/// digits; otherwise the role is returned unchanged. This is the single shared
/// rule consumed by both the GUI and [`agent_author_label`].
#[must_use]
pub fn strip_agent_role_suffix(agent_role: &str) -> &str {
    agent_role
        .rsplit_once('_')
        .and_then(|(base, suffix)| {
            if suffix.chars().all(|c| c.is_ascii_digit()) {
                Some(base)
            } else {
                None
            }
        })
        .unwrap_or(agent_role)
}

/// Resolve the display label for an agent-role string, falling back to
/// `"Agent"` when the role is absent or unparseable.
#[must_use]
pub fn agent_author_label(agent_role: Option<&str>) -> String {
    agent_role
        .map(strip_agent_role_suffix)
        .and_then(|stripped| stripped.parse::<Role>().ok())
        .map_or_else(
            || "Agent".to_string(),
            |role| role.display_label().to_string(),
        )
}

#[cfg(test)]
#[path = "reply_tests.rs"]
mod tests;
