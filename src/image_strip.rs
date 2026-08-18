//! Provider input-image rejection: detection and durable strip.
//!
//! When a provider rejects an input image at its content-inspection stage
//! (observed: OpenRouter HTTP 400 `data_inspection_failed` — "Input image data
//! may contain inappropriate content."), the rejection happens pre-inference
//! (no tokens billed) and is **sticky**: the rejected image stays in the
//! session history and is re-sent on every later turn, so every subsequent
//! message in that session fails the same way until the session is reset.
//!
//! This module detects that rejection from the retry trail and rewrites the
//! most recent User-role message — durably, via the session store — replacing
//! all of its image markers with a single explanatory phrase (authored in
//! `src/prompt/context/image_rejected.md`; all LLM-sent text lives under
//! `src/prompt/`). The agent run then continues through its normal loop with
//! the corrected message; there is **no dedicated retry step**.
//!
//! Scope contract (ticket mahbot-1788):
//! - Only the most recent User-role message is touched; images in earlier
//!   user messages are left alone.
//! - The strip fires only on genuine input-image content rejections: the
//!   full failure trail must contain the provider code AND an image-specific
//!   message fragment. A text-content rejection with the same code (message
//!   says "Input data ..." without "image") never triggers the strip.
//! - Pre-existing sticky sessions (from before this fix) are **not** healed
//!   retroactively: healing requires the image-bearing message to BE the most
//!   recent User-role message when the rejection is detected.
//! - If another User-role message was injected after the image-bearing one
//!   (e.g. a ticket comment), the most recent message is that comment; with
//!   no image markers it does not match, so the strip does not fire and the
//!   normal failure path applies.
//! - A later injected message that itself carries an `[IMAGE:` marker is
//!   stripped instead of the earlier image-bearing one — an inherent
//!   consequence of the "most recent User-role message" rule (accepted; a
//!   ticket comment carrying an image marker is not a realistic Artist-chat
//!   scenario).

use crate::prompt::{load_prompt, substitute};
use crate::retry::RetryExhausted;
use crate::{ChatMessage, ChatRole};

/// Provider content-inspection error code that identifies the rejection
/// (case-insensitive substring match against the failure trail).
const INPUT_INSPECTION_CODE: &str = "data_inspection_failed";

/// Stable core of the observed image-specific message fragment ("Input image
/// data may contain inappropriate content."). Matched case-insensitively
/// together with the word "image" within the same failure record — a
/// text-content rejection ("Input data ...") lacks "image" and never matches.
/// The literal observed sentence also matches this core; a provider rephrasing
/// that drops either part silently disables detection (documented limitation —
/// the strip never false-positives on text rejections).
const INAPPROPRIATE_CONTENT_FRAGMENT: &str = "inappropriate content";

/// Fallback reason embedded in the phrase when the trail yields no usable
/// provider message.
const REASON_FALLBACK: &str = "no specific reason was provided";

/// Char cap for the provider reason embedded in the phrase.
const REASON_CAP: usize = 500;

/// Phrase asset key (all LLM-sent text lives under `src/prompt/`).
const PHRASE_ASSET: &str = "context/image_rejected.md";

/// Detect a provider input-image content-inspection rejection on the failure
/// trail, returning the index of the most recent User-role message when the
/// strip should fire (`None` = no strip).
///
/// Detection contract (documented):
/// - The **full** failure trail (all recorded attempts) must contain the
///   provider content-inspection code AND the image-specific fragment. In
///   practice a 400-class rejection is [`crate::retry::FailureClass::NonRetryable`],
///   so the trail has exactly one record; scanning all records keeps the rule
///   correct if a retryable attempt ever precedes the rejection.
/// - The fragment is a stable core, not the full observed sentence: any
///   provider wording containing "inappropriate content" together with
///   "image" matches (case-insensitively); the literal observed string also
///   matches. Provider rewording that drops either core disables detection
///   (documented limitation — a false positive would strip a message whose
///   image was never rejected).
/// - The most recent User-role message must contain at least one IMAGE marker
///   that [`strip_image_markers`] will actually strip — the two share one
///   predicate ([`has_image_marker`]), so a detected rejection is guaranteed
///   to change the content. Malformed/empty `[IMAGE:` fragments never match
///   the marker regex and never trigger (matching the provider-layer marker
///   parser); otherwise no strip (e.g. an injected ticket comment after the
///   image-bearing message).
#[must_use]
pub(crate) fn detect_input_image_rejection(
    exhausted: &RetryExhausted,
    history: &[ChatMessage],
) -> Option<usize> {
    let trail: Vec<String> = exhausted
        .failures
        .iter()
        .map(|f| f.error_chain.to_lowercase())
        .collect();
    let has_code = trail.iter().any(|t| t.contains(INPUT_INSPECTION_CODE));
    let has_image_fragment = trail
        .iter()
        .any(|t| t.contains(INAPPROPRIATE_CONTENT_FRAGMENT) && t.contains("image"));
    if !has_code || !has_image_fragment {
        return None;
    }
    let idx = history.iter().rposition(|m| m.role == ChatRole::User)?;
    has_image_marker(&history[idx].content).then_some(idx)
}

/// True when the content carries at least one IMAGE marker that
/// [`strip_image_markers`] would strip — the SAME predicate the strip uses
/// (`MEDIA_MARKER_RE`, IMAGE kind only). Keeping detection and strip on one
/// predicate guarantees a detected rejection always changes the content: a
/// no-change rewrite would report success and the caller's loop `continue`
/// skips the iteration counter, risking an unbounded retry loop. Empty
/// `[IMAGE:]` or unclosed `[IMAGE:...` fragments never match the regex and
/// never trigger (the provider-layer marker parser ignores them the same way).
fn has_image_marker(content: &str) -> bool {
    crate::util::MEDIA_MARKER_RE
        .captures_iter(content)
        .any(|caps| crate::util::parse_media_marker(&caps).0 == "IMAGE")
}

/// Extract the provider's reason from the failure trail, if any.
///
/// The trail's `error_chain` text carries the HttpError display
/// (`"{context} API error ({status}): {body}"`); the body is the provider's
/// JSON envelope. Extraction prefers the structured `error.message`, then the
/// nested-envelope fields (see [`crate::util::extract_provider_error_detail`]);
/// `None` means the caller substitutes [`REASON_FALLBACK`].
#[must_use]
pub(crate) fn extract_provider_reason(exhausted: &RetryExhausted) -> Option<String> {
    exhausted
        .failures
        .iter()
        .find_map(|f| reason_from_chain(&f.error_chain))
}

/// Pull the provider reason out of a single formatted error-chain string.
fn reason_from_chain(chain: &str) -> Option<String> {
    // HttpError displays as "{context} API error ({status}): {body}" — the
    // body starts after the first "): " separator (provider names never
    // contain it). Non-HttpError chains (timeouts, parse failures) lack the
    // separator and yield None.
    let sep = chain.find("): ")?;
    let body = &chain[sep + 3..];
    let body_json = serde_json::from_str::<serde_json::Value>(body).ok()?;
    crate::util::extract_provider_error_detail(&body_json)
}

/// Build the corrected content of the rejected user message: all `[IMAGE:...]`
/// markers are removed and replaced by a single explanatory phrase placed at
/// the position of the first marker, with the user's accompanying text
/// preserved around it. Non-image markers (`[AUDIO:` / `[VIDEO:`) are left
/// untouched. Empty `[IMAGE:]` markers do not match the marker regex and are
/// preserved verbatim (matching the provider-layer marker parser).
///
/// Literal replacement rule: the first image marker becomes the phrase, every
/// later image marker is deleted (the provider error does not identify which
/// image was rejected). A marker-separated sentence like
/// `see [IMAGE:a] and [IMAGE:b] here` becomes `see <phrase> and  here` — the
/// user's text is preserved verbatim; no separator trimming is attempted.
#[must_use]
pub(crate) fn strip_image_markers(content: &str, reason: Option<&str>) -> String {
    let reason = reason.unwrap_or(REASON_FALLBACK);
    // Trim trailing punctuation from the embedded reason so a provider message
    // ending in "." does not double up against the phrase's own period.
    let sanitized = crate::util::truncate(&crate::util::scrub_credentials(reason), REASON_CAP);
    let sanitized = sanitized
        .trim_end_matches(|c: char| c == '.' || c.is_whitespace())
        .to_string();
    let phrase = substitute(
        &load_prompt(PHRASE_ASSET),
        &[("{{reason}}", sanitized.as_str())],
    );

    let mut out = String::with_capacity(content.len() + phrase.len());
    let mut first = true;
    let mut last_end = 0usize;
    for caps in crate::util::MEDIA_MARKER_RE.captures_iter(content) {
        let (kind, _) = crate::util::parse_media_marker(&caps);
        if kind != "IMAGE" {
            continue;
        }
        let whole = caps.get_match();
        out.push_str(&content[last_end..whole.start()]);
        if first {
            out.push_str(&phrase);
            first = false;
        }
        last_end = whole.end();
    }
    out.push_str(&content[last_end..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::{FailureClass, RetryFailureRecord};

    /// Build a single-failure trail from a raw error-chain string (the shape
    /// `FakeProvider::err_http` produces in end-to-end tests).
    fn trail(error_chain: &str) -> RetryExhausted {
        RetryExhausted::with_last_raw(
            vec![RetryFailureRecord::new_simple(
                FailureClass::NonRetryable,
                &anyhow::anyhow!("{error_chain}"),
                None,
            )],
            FailureClass::NonRetryable,
            None,
        )
    }

    const REJECTION_CHAIN: &str = r#"OpenRouter API error (400): {"error":{"message":"Input image data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#;

    fn history_with_image() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("role description"),
            ChatMessage::user("earlier text without image"),
            ChatMessage::user("[IMAGE:/tmp/photo.png]\n\ndescribe this"),
        ]
    }

    #[test]
    fn image_rejection_detected_on_most_recent_user_message() {
        let idx = detect_input_image_rejection(&trail(REJECTION_CHAIN), &history_with_image());
        assert_eq!(
            idx,
            Some(2),
            "the image-bearing user message must be targeted"
        );
    }

    #[test]
    fn text_content_rejection_with_same_code_does_not_trigger_strip() {
        // Same code, message says "Input data ..." without "image".
        let chain = r#"OpenRouter API error (400): {"error":{"message":"Input data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#;
        assert_eq!(
            detect_input_image_rejection(&trail(chain), &history_with_image()),
            None
        );
    }

    #[test]
    fn missing_inspection_code_does_not_trigger_strip() {
        let chain = r#"OpenRouter API error (400): {"error":{"message":"Input image data may contain inappropriate content.","code":"other_code","type":"invalid_request_error"}}"#;
        assert_eq!(
            detect_input_image_rejection(&trail(chain), &history_with_image()),
            None
        );
    }

    #[test]
    fn case_insensitive_matching() {
        let chain = r#"OpenRouter API error (400): {"error":{"message":"INPUT IMAGE DATA MAY CONTAIN INAPPROPRIATE CONTENT.","code":"DATA_INSPECTION_FAILED"}}"#;
        assert_eq!(
            detect_input_image_rejection(&trail(chain), &history_with_image()),
            Some(2)
        );
    }

    #[test]
    fn strip_targets_only_the_most_recent_user_message() {
        // Markers in EARLIER user messages alone do not trigger: the most
        // recent User-role message is a later injected text-only one.
        let history = vec![
            ChatMessage::system("role description"),
            ChatMessage::user("[IMAGE:/tmp/old.png] old image"),
            ChatMessage::user("a later text-only message"),
        ];
        assert_eq!(
            detect_input_image_rejection(&trail(REJECTION_CHAIN), &history),
            None
        );
        // Markers in earlier messages do not prevent targeting the most
        // recent user message when THAT one carries the image.
        let history = vec![
            ChatMessage::user("[IMAGE:/tmp/old.png] old"),
            ChatMessage::user("[IMAGE:/tmp/new.png] new"),
        ];
        let idx = detect_input_image_rejection(&trail(REJECTION_CHAIN), &history);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn malformed_or_empty_marker_does_not_trigger_strip() {
        // Regression pinning the detection/strip gate: detection and the strip
        // share ONE predicate (MEDIA_MARKER_RE, IMAGE kind). Fragments the
        // marker regex leaves verbatim (empty `[IMAGE:]`, unclosed `[IMAGE:foo`,
        // trailing `[IMAGE:`) or non-IMAGE markers (`[AUDIO:...]`) must NOT
        // trigger — a detected rejection is guaranteed to change the content,
        // so a no-op rewrite plus the caller's `continue` (which skips the
        // iteration counter) cannot form an unbounded retry loop.
        for content in [
            "hello [IMAGE:] world",
            "unclosed [IMAGE:foo",
            "trailing [IMAGE:",
            "[AUDIO:/tmp/sound.mp3] audio-only",
        ] {
            let history = vec![ChatMessage::system("role"), ChatMessage::user(content)];
            assert_eq!(
                detect_input_image_rejection(&trail(REJECTION_CHAIN), &history),
                None,
                "content with no strippable IMAGE marker must not trigger: {content:?}"
            );
        }
    }

    #[test]
    fn malformed_latest_marker_with_earlier_image_is_conservative_noop() {
        // Sticky-session shape from review: an earlier user message's
        // well-formed marker keeps the provider rejecting, while the most
        // recent message's malformed `[IMAGE:` text would (under a loose
        // `contains("[IMAGE:")` gate) fire the strip as a no-op — the exact
        // unbounded-loop shape. The aligned gate must not fire: no strip, the
        // normal failure path applies.
        let history = vec![
            ChatMessage::user("[IMAGE:/tmp/old.png] earlier image"),
            ChatMessage::user("see [IMAGE: here"),
        ];
        assert_eq!(
            detect_input_image_rejection(&trail(REJECTION_CHAIN), &history),
            None
        );
    }

    #[test]
    fn strip_replaces_all_image_markers_with_single_phrase() {
        let content = "see [IMAGE:/tmp/a.png] and [IMAGE:/tmp/b.png] here";
        let out = strip_image_markers(content, Some("the image is blocked"));
        assert!(
            out.starts_with("see "),
            "user text before the first marker is preserved"
        );
        assert!(!out.contains("[IMAGE:"), "all image markers removed");
        assert!(
            out.contains("rejected by the provider's content-inspection check"),
            "phrase explains the rejection: {out}"
        );
        assert!(
            out.contains("the image is blocked"),
            "sanitized reason embedded: {out}"
        );
        assert!(
            out.contains("here"),
            "user text after the last marker is preserved"
        );
        // The literal rule leaves the inter-marker separator text in place.
        assert!(
            out.contains("and "),
            "separator text preserved verbatim: {out}"
        );
    }

    #[test]
    fn strip_preserves_non_image_markers() {
        let content = "[AUDIO:/tmp/sound.mp3] listen and [IMAGE:/tmp/img.png] look";
        let out = strip_image_markers(content, None);
        assert!(
            out.contains("[AUDIO:/tmp/sound.mp3]"),
            "audio marker untouched"
        );
        assert!(!out.contains("[IMAGE:"), "image marker removed");
        assert!(out.contains("listen and"), "user text preserved");
        assert!(out.contains("look"));
    }

    #[test]
    fn strip_fallback_reason_when_absent() {
        let out = strip_image_markers("[IMAGE:/tmp/a.png] hi", None);
        assert!(out.contains("no specific reason was provided"));
    }

    #[test]
    fn strip_scrubs_credentials_from_reason() {
        let content = "[IMAGE:/tmp/a.png] hi";
        let out = strip_image_markers(content, Some("leaked API_KEY=sk-1234567890abcdef"));
        assert!(
            !out.contains("sk-1234567890abcdef"),
            "credential scrubbed: {out}"
        );
    }

    #[test]
    fn strip_empty_marker_preserved_verbatim() {
        // Empty `[IMAGE:]` does not match the marker regex — preserved.
        let content = "hello [IMAGE:] world";
        assert_eq!(strip_image_markers(content, None), content);
    }

    #[test]
    fn reason_extracted_from_http_error_body() {
        let reason = extract_provider_reason(&trail(REJECTION_CHAIN));
        assert_eq!(
            reason.as_deref(),
            Some("Input image data may contain inappropriate content.")
        );
    }

    #[test]
    fn reason_extracted_from_nested_envelope() {
        let chain = r#"OpenRouter API error (400): {"error":{"code":"data_inspection_failed","metadata":{"raw":"Input image data may contain inappropriate content."}}}"#;
        assert_eq!(
            extract_provider_reason(&trail(chain)).as_deref(),
            Some("Input image data may contain inappropriate content.")
        );
    }

    #[test]
    fn reason_absent_for_non_http_chain() {
        let exhausted = RetryExhausted::with_last_raw(
            vec![RetryFailureRecord::new_simple(
                FailureClass::Transport,
                &anyhow::anyhow!("connection timed out"),
                None,
            )],
            FailureClass::Transport,
            None,
        );
        assert_eq!(extract_provider_reason(&exhausted), None);
    }

    #[test]
    fn reason_falls_back_to_code_when_no_message() {
        let chain = r#"OpenRouter API error (400): {"error":{"code":"data_inspection_failed","type":"invalid_request_error"}}"#;
        assert_eq!(
            extract_provider_reason(&trail(chain)).as_deref(),
            Some("data_inspection_failed")
        );
    }
}
