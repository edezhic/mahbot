//! Reply-reference plumbing: the shared snippet normalization and marker formatting the GUI and the
//! Telegram channel use to render a "reply to" quote header on a routed message.

use crate::Role;
use crate::util::html::decode_html_entities;
use crate::util::{MEDIA_MARKER_RE, MediaMarkerKind, parse_media_marker};
use regex::Captures;
use serde::{Deserialize, Serialize};

/// Maximum length (in Unicode chars) of a normalized reply snippet.
pub const REPLY_SNIPPET_MAX_CHARS: usize = 100;

/// A reference to the message being replied to: the author's canonical name and a short snippet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplyReference {
    pub author: String,
    pub snippet: String,
}

/// Normalize a raw reply snippet into the deterministic shared form both the GUI quote header and
/// the Telegram channel use. Order matters (pinned by tests): decode HTML entities; strip a
/// `<blockquote>…</blockquote>` wrapper (GUI-mirrored messages); remove all `<`/`>`; collapse
/// `\r\n` and
/// `\n` to single spaces (other whitespace untouched); map media markers inline (IMAGE→`[Photo]`,
/// AUDIO→`[Voice message]`, VIDEO→`[Video]`, FILE→`[File]`); cap at [`REPLY_SNIPPET_MAX_CHARS`],
/// the ellipsis inside the cap and whitespace at the cut trimmed. It operates on the raw content —
/// callers invoke it before any markdown preprocessing so a live data-URI marker still maps.
#[must_use]
pub fn normalize_reply_text(raw: &str) -> String {
    let mut text = decode_html_entities(raw);

    // 2. Strip a `<blockquote>…</blockquote>` wrapper (format:
    // `<blockquote>\n{content}\n</blockquote>`).
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

    text.retain(|c| c != '<' && c != '>');

    text = text.replace("\r\n", " ").replace('\n', " ");

    text = MEDIA_MARKER_RE
        .replace_all(&text, |caps: &Captures| match parse_media_marker(caps).0 {
            MediaMarkerKind::Image => "[Photo]",
            MediaMarkerKind::Audio => "[Voice message]",
            MediaMarkerKind::Video => "[Video]",
            MediaMarkerKind::File => "[File]",
        })
        .into_owned();

    crate::util::truncate(&text, REPLY_SNIPPET_MAX_CHARS)
}

/// Render the reply marker prefix prepended to a routed message; the trailing colon is intentional.
#[must_use]
fn format_reply_marker(reply: &ReplyReference) -> String {
    format!(
        "<reply-to>{}: {}</reply-to>:\n",
        reply.author, reply.snippet
    )
}

/// Prepend the reply marker to routed content — at the very front, before any Telegram
/// `[Forwarded from …]` prefix embedded in `content`.
#[must_use]
pub fn apply_reply_marker(content: &str, reply: &ReplyReference) -> String {
    format!("{}{content}", format_reply_marker(reply))
}

/// Strip a trailing `_<digits>` role suffix (`"analyst_3"` → `"analyst"`): `rsplit_once('_')` where
/// the suffix is all ASCII digits, else unchanged. The one rule the GUI and [`agent_author_label`]
/// share.
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

/// The display label for an agent-role string, `"Agent"` when the role is absent or unparseable.
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
