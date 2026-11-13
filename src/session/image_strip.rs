//! Provider input-image rejection: detection and durable strip.
//!
//! When a provider rejects an input image at its content-inspection stage (observed: OpenRouter
//! HTTP 400 `data_inspection_failed` — "Input image data may contain inappropriate content."), the
//! rejection happens pre-inference (no tokens billed) and is **sticky**: the rejected image stays
//! in the session history and is re-sent every later turn, so every subsequent message in that
//! session fails the same way until the session is reset.
//! This module detects that rejection from the retry trail and rewrites the most recent User-role
//! message — durably, via the session store — replacing all its image markers with one explanatory
//! phrase (authored in `src/prompt/context/image_rejected.md`; all LLM-sent text lives under
//! `src/prompt/`). The run then continues through its normal loop; there is **no dedicated retry
//! step**.
//!
//! Scope contract: only the most recent User-role message is touched; the strip fires only on
//! genuine input-image content rejections (the failure trail must hold the provider code AND an
//! image-specific message fragment — a text-content rejection with the same code never triggers);
//! pre-existing sticky sessions are **not** healed retroactively; and when another User-role
//! message was injected after the image-bearing one (e.g. a ticket comment) with no image markers,
//! the strip does not fire and the normal failure path applies. Accepted: a LATER injected message
//! that itself carries an `[IMAGE:` marker is stripped instead — a consequence of the "most recent"
//! rule, and a ticket comment carrying an image marker is not a realistic Assistant-chat scenario.

use crate::prompt::{load_prompt, substitute};
use crate::retry::RetryExhausted;
use crate::util::INJECTED_IMAGE_TAG;
use crate::{ChatMessage, ChatRole};

/// Provider content-inspection error code (case-insensitive substring match on the failure trail).
const INPUT_INSPECTION_CODE: &str = "data_inspection_failed";

/// Stable core of the observed image-specific fragment ("Input image data may contain inappropriate
/// content."), matched case-insensitively together with the word "image" in the same failure record
/// — a text-content rejection ("Input data …") lacks "image" and never matches. A provider
/// rephrasing that drops either part silently disables detection (documented limit — the strip
/// never false-positives on text rejections).
const INAPPROPRIATE_CONTENT_FRAGMENT: &str = "inappropriate content";

/// Fallback reason embedded in the phrase when the trail yields no usable provider message.
const REASON_FALLBACK: &str = "no specific reason was provided";

/// Char cap for the provider reason embedded in the phrase.
const REASON_CAP: usize = 500;

const PHRASE_ASSET: &str = "context/image_rejected.md";

/// Detect a provider input-image content-inspection rejection on the trail, returning the index of
/// the most recent User-role message when the strip should fire (`None` = no strip).
///
/// The **full** trail (all recorded attempts) must contain the content-inspection code AND the
/// image-specific fragment: the rejection is terminal
/// ([`crate::retry::FailureClass::ContentRejection`] in practice), so there is usually one
/// record, but scanning all of them keeps the rule correct if a retryable attempt ever precedes the
/// rejection. The most recent User-role message must carry at least one IMAGE marker
/// [`strip_image_markers`] will actually strip — the two share one predicate
/// ([`has_image_marker`]), so a detected rejection is guaranteed to change the content, and
/// malformed/empty `[IMAGE:` fragments never trigger.
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

/// True when the content carries at least one IMAGE marker [`strip_image_markers`] would strip —
/// the SAME predicate (`MEDIA_MARKER_RE`, IMAGE kind only). One predicate guarantees a detected
/// rejection always changes the content: a no-change rewrite would report success while the
/// caller's loop `continue` skips the iteration counter, risking an unbounded retry loop. Empty
/// `[IMAGE:]` or unclosed `[IMAGE:...` fragments never match (the provider-layer parser ignores
/// them the same way).
fn has_image_marker(content: &str) -> bool {
    crate::util::MEDIA_MARKER_RE
        .captures_iter(content)
        .any(|caps| crate::util::parse_media_marker(&caps).0 == crate::util::MediaMarkerKind::Image)
}

/// Extract the provider's reason from the failure trail, if any. The trail's `error_chain` carries
/// the HttpError display (`"{context} API error ({status}): {body}"`) whose body is the provider's
/// JSON envelope; extraction prefers the structured `error.message`, then the nested fields (see
/// [`crate::util::extract_provider_error_detail`]). `None` means the caller substitutes
/// [`REASON_FALLBACK`].
#[must_use]
pub(crate) fn extract_provider_reason(exhausted: &RetryExhausted) -> Option<String> {
    exhausted
        .failures
        .iter()
        .find_map(|f| reason_from_chain(&f.error_chain))
}

fn reason_from_chain(chain: &str) -> Option<String> {
    // HttpError displays as "{context} API error ({status}): {body}", so the body starts after
    // the first "): " separator (provider names never contain it). Non-HttpError chains
    // (timeouts, parse failures) lack it and yield None.
    let sep = chain.find("): ")?;
    let body = &chain[sep + 3..];
    let body_json = serde_json::from_str::<serde_json::Value>(body).ok()?;
    crate::util::extract_provider_error_detail(&body_json)
}

/// Build the corrected content of the rejected user message: every `[IMAGE:...]` marker goes,
/// replaced by one explanatory phrase at the position of the first, with the user's text preserved
/// around it. Non-image markers (`[AUDIO:` / `[VIDEO:`) are untouched, and empty `[IMAGE:]` markers
/// do not match the marker regex (matching the provider-layer parser).
///
/// The first image marker becomes the phrase and later ones are deleted (the provider error does
/// not say which image was rejected), so `see [IMAGE:a] and [IMAGE:b] here` becomes `see <phrase>
/// and here` — the user's text verbatim, with no separator trimming.
#[must_use]
pub(crate) fn strip_image_markers(content: &str, reason: Option<&str>) -> String {
    let reason = reason.unwrap_or(REASON_FALLBACK);
    // Trim trailing punctuation from the reason so a provider message ending in "." does not
    // double up against the phrase's own period.
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
        if kind != crate::util::MediaMarkerKind::Image {
            continue;
        }
        let whole = caps.get_match();
        if first {
            let seg = &content[last_end..whole.start()];
            // A synthetic tool-injected image message carries a provenance tag that
            // must not dangle next to the replacement phrase.
            if seg.contains(INJECTED_IMAGE_TAG) {
                let cleaned = seg.replace(INJECTED_IMAGE_TAG, "").trim().to_string();
                out.push_str(&cleaned);
            } else {
                out.push_str(seg);
            }
            out.push_str(&phrase);
            first = false;
        } else {
            out.push_str(&content[last_end..whole.start()]);
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
                FailureClass::InvalidRequest,
                &anyhow::anyhow!("{error_chain}"),
                None,
            )],
            FailureClass::InvalidRequest,
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

    /// Both halves of the gate — the inspection code and the image wording — decide the strip,
    /// and both match case-insensitively: a text-content rejection with the same code, that
    /// code on another error, and the all-caps spelling of a genuine rejection.
    #[test]
    fn detection_gate_keys_on_the_image_inspection_code_and_wording() {
        let cases: &[(&str, Option<usize>)] = &[
            (
                r#"OpenRouter API error (400): {"error":{"message":"Input data may contain inappropriate content.","code":"data_inspection_failed","type":"invalid_request_error"}}"#,
                None,
            ),
            (
                r#"OpenRouter API error (400): {"error":{"message":"Input image data may contain inappropriate content.","code":"other_code","type":"invalid_request_error"}}"#,
                None,
            ),
            (
                r#"OpenRouter API error (400): {"error":{"message":"INPUT IMAGE DATA MAY CONTAIN INAPPROPRIATE CONTENT.","code":"DATA_INSPECTION_FAILED"}}"#,
                Some(2),
            ),
        ];
        for (chain, expected) in cases {
            assert_eq!(
                detect_input_image_rejection(&trail(chain), &history_with_image()),
                *expected,
                "unexpected decision for: {chain}"
            );
        }
    }

    #[test]
    fn strip_targets_only_the_most_recent_user_message() {
        let history = vec![
            ChatMessage::system("role description"),
            ChatMessage::user("[IMAGE:/tmp/old.png] old image"),
            ChatMessage::user("a later text-only message"),
        ];
        assert_eq!(
            detect_input_image_rejection(&trail(REJECTION_CHAIN), &history),
            None
        );
        let history = vec![
            ChatMessage::user("[IMAGE:/tmp/old.png] old"),
            ChatMessage::user("[IMAGE:/tmp/new.png] new"),
        ];
        let idx = detect_input_image_rejection(&trail(REJECTION_CHAIN), &history);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn malformed_or_empty_marker_does_not_trigger_strip() {
        // Pins the detection/strip gate: both share ONE predicate (MEDIA_MARKER_RE, IMAGE
        // kind). Fragments the regex leaves verbatim (empty `[IMAGE:]`, unclosed
        // `[IMAGE:foo`, trailing `[IMAGE:`) or non-IMAGE markers (`[AUDIO:...]`) must NOT
        // trigger — otherwise a no-op rewrite plus the caller's `continue` (which skips the
        // iteration counter) forms an unbounded retry loop.
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
        // Sticky-session shape from review: an earlier message's well-formed marker keeps
        // the provider rejecting while the most recent message's malformed `[IMAGE:` text
        // would, under a loose `contains("[IMAGE:")` gate, strip to a no-op — the
        // unbounded-loop shape. The aligned gate must not fire.
        let history = vec![
            ChatMessage::user("[IMAGE:/tmp/old.png] earlier image"),
            ChatMessage::user("see [IMAGE: here"),
        ];
        assert_eq!(
            detect_input_image_rejection(&trail(REJECTION_CHAIN), &history),
            None
        );
    }

    /// One rewrite rule over the marker's shapes: every `[IMAGE:…]` becomes the explanatory
    /// phrase, the user's own text and non-image markers keep their place (inter-marker
    /// separator included), and the provenance tag wrapping an injected tool result goes with
    /// the marker it annotated.
    #[test]
    fn strip_rewrites_image_markers_and_keeps_the_rest() {
        type Case<'a> = (&'a str, Option<&'a str>, &'a [&'a str], &'a [&'a str]);
        let phrase = "rejected by the provider's content-inspection check";
        let cases: &[Case] = &[
            (
                "see [IMAGE:/tmp/a.png] and [IMAGE:/tmp/b.png] here",
                Some("the image is blocked"),
                &["see ", "and ", "here", phrase, "the image is blocked"],
                &["[IMAGE:"],
            ),
            (
                "[AUDIO:/tmp/sound.mp3] listen and [IMAGE:/tmp/img.png] look",
                None,
                &["[AUDIO:/tmp/sound.mp3]", "listen and", "look", phrase],
                &["[IMAGE:"],
            ),
            (
                "<injected-tool-result-image>\n[IMAGE:data:image/jpeg;base64,abc]",
                Some("blocked"),
                &[phrase],
                &["[IMAGE:", "<injected-tool-result-image>", "\n"],
            ),
        ];
        for (content, reason, present, absent) in cases {
            let out = strip_image_markers(content, *reason);
            for needle in *present {
                assert!(out.contains(needle), "missing {needle:?}: {out}");
            }
            for needle in *absent {
                assert!(!out.contains(needle), "unexpected {needle:?}: {out}");
            }
            assert!(
                !out.starts_with([' ', '\n']),
                "no dangling whitespace before the phrase: {out:?}"
            );
        }
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
    fn reason_extracted_from_http_error_body() {
        let reason = extract_provider_reason(&trail(REJECTION_CHAIN));
        assert_eq!(
            reason.as_deref(),
            Some("Input image data may contain inappropriate content.")
        );
    }
}
