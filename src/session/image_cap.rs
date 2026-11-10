//! The picture cap: no model request may carry more pictures than the product's one picture
//! number, [`SUMMARIZATION_IMAGE_COUNT`](super::SUMMARIZATION_IMAGE_COUNT).
//!
//! Every request re-sends the whole conversation, so pictures accumulate across tool rounds and
//! turns until a provider refuses the request outright. The two request builders that carry
//! conversation history — [`crate::Agent::build_chat_request`] (the agent loop, the reasoning
//! continuation and summarisation) and
//! [`crate::agent::extraction::retry_extract_structured_scoped`] (verdict extraction, research
//! wrap-up, comment summaries) — call [`apply`], which replaces the OLDEST pictures in place with
//! the omission note the product already uses for a picture that was not delivered
//! ([`crate::util::strip_data_uris`]), a batch at a time. The conversation itself is never touched:
//! only what is sent is bounded.
//!
//! A picture is what the provider receives as a picture: an `[IMAGE:data:image/…]` marker in a
//! User-role message. A local path or a remote link stays text and is never retired. The prefix
//! gate is the same documented superset the picture-based summarisation trigger counts, so counting
//! costs no decode: a malformed `data:image/` payload may count and be replaced by the note, which
//! over-counts only and so never lets a request past the cap.

use crate::util::{MEDIA_MARKER_RE, MediaMarkerKind, parse_media_marker};
use crate::{ChatMessage, ChatRole};
use std::collections::HashSet;

/// Pictures retired per crossing — half the cap, so a retirement leaves the request well under it.
/// Whole batches: the retirement point then jumps by ten instead of following every new picture, so
/// the provider's cached prefix survives between retirements.
pub(crate) const RETIRE_BATCH: usize = super::SUMMARIZATION_IMAGE_COUNT / 2;

/// Values of the `[IMAGE:…]` markers in User-role messages — the only role whose markers become
/// provider vision parts, so the only place a picture can be. Paths and links are included; see
/// [`is_picture`].
pub(crate) fn user_image_values(messages: &[ChatMessage]) -> impl Iterator<Item = &str> {
    messages
        .iter()
        .filter(|msg| {
            msg.role == ChatRole::User
                && crate::util::has_media_marker(&msg.content, MediaMarkerKind::Image)
        })
        .flat_map(|msg| {
            MEDIA_MARKER_RE
                .captures_iter(&msg.content)
                .filter_map(|caps| {
                    let (kind, path) = parse_media_marker(&caps);
                    (kind == MediaMarkerKind::Image).then_some(path)
                })
        })
}

/// How many pictures these messages would send as image parts.
#[must_use]
pub(crate) fn count(messages: &[ChatMessage]) -> usize {
    user_image_values(messages)
        .filter(|value| is_picture(value))
        .count()
}

/// The pictures a request built from these messages would still send: the newest, after the
/// retirement [`apply`] would make. A caller that must not claim a picture is attached when it is
/// not (the tool-image dedup) keys off this rather than the whole history.
pub(crate) fn sent_picture_values(messages: &[ChatMessage]) -> HashSet<String> {
    let pictures: Vec<&str> = user_image_values(messages)
        .filter(|value| is_picture(value))
        .collect();
    let retire = if pictures.len() > super::SUMMARIZATION_IMAGE_COUNT {
        retire_count(pictures.len())
    } else {
        0
    };
    pictures
        .into_iter()
        .skip(retire)
        .map(str::to_string)
        .collect()
}

/// Retire the oldest pictures when the request would carry more than the cap: whole batches of
/// [`RETIRE_BATCH`] until it is under the cap again, which leaves at least a batch and at most
/// `SUMMARIZATION_IMAGE_COUNT - 1` of the newest pictures.
///
/// `known` is the conversation's recorded picture count
/// ([`Session::image_count`](super::Session::image_count)): `Some(n)` at or below the cap skips the
/// walk entirely — the per-round path. `None` (a caller whose history has no session behind it)
/// counts the messages. The recorded count is a gate, never the authority: a stale value must not be
/// able to size a retirement, so an over-cap request counts its own pictures and a `known` that
/// overstates them only costs the walk. A request at or below the cap is sent byte-for-byte as it is
/// today.
pub(crate) fn apply(messages: &mut [ChatMessage], known: Option<usize>) {
    if known.is_some_and(|known| known <= super::SUMMARIZATION_IMAGE_COUNT) {
        return;
    }
    let total = count(messages);
    if total <= super::SUMMARIZATION_IMAGE_COUNT {
        return;
    }
    let mut retire = retire_count(total);
    for message in messages {
        if retire == 0 {
            break;
        }
        if message.role != ChatRole::User
            || !crate::util::has_media_marker(&message.content, MediaMarkerKind::Image)
        {
            continue;
        }
        let (content, retired) = retire_oldest(&message.content, retire);
        if retired > 0 {
            message.content = content;
            retire -= retired;
        }
    }
}

/// How many of `total` pictures step aside: whole batches until the rest is under the cap. Only
/// called for a request over the cap.
fn retire_count(total: usize) -> usize {
    let mut retire = RETIRE_BATCH;
    while total - retire >= super::SUMMARIZATION_IMAGE_COUNT {
        retire += RETIRE_BATCH;
    }
    retire
}

/// Replace the oldest `limit` pictures of one message with the omission note, returning the
/// rewritten content and how many actually stepped aside.
fn retire_oldest(content: &str, limit: usize) -> (String, usize) {
    let mut retired = 0;
    let rewritten = MEDIA_MARKER_RE.replace_all(content, |caps: &regex::Captures| {
        let (kind, path) = parse_media_marker(caps);
        if retired < limit && kind == MediaMarkerKind::Image && is_picture(path) {
            retired += 1;
            crate::util::data_uri_omitted_note(caps.get_match().as_str())
        } else {
            caps.get_match().as_str().to_string()
        }
    });
    (rewritten.into_owned(), retired)
}

/// True when the marker's value is a picture the provider receives as an image part: an inline data
/// URI. A local path or a remote link stays text and is never retired.
fn is_picture(value: &str) -> bool {
    value.trim().starts_with("data:image/")
}

#[cfg(test)]
mod tests {
    use super::{RETIRE_BATCH, apply, count};
    use crate::session::SUMMARIZATION_IMAGE_COUNT;
    use crate::{ChatMessage, ChatRole};
    use std::collections::HashSet;

    /// The marker VALUE (the data URI) of one picture.
    fn value(serial: usize) -> String {
        format!("data:image/png;base64,{serial:0>8}")
    }

    fn picture(serial: usize) -> String {
        format!("[IMAGE:{}]", value(serial))
    }

    /// `pictures` pictures spread one per user message, oldest first.
    fn spread(pictures: usize) -> Vec<ChatMessage> {
        (0..pictures)
            .map(|serial| ChatMessage::user(format!("shot {serial} {}", picture(serial))))
            .collect()
    }

    #[test]
    fn at_or_below_the_cap_is_untouched() {
        let mut messages = spread(SUMMARIZATION_IMAGE_COUNT);
        let before: Vec<String> = messages.iter().map(|m| m.content.clone()).collect();
        apply(&mut messages, Some(SUMMARIZATION_IMAGE_COUNT));
        let after: Vec<String> = messages.iter().map(|m| m.content.clone()).collect();
        assert_eq!(before, after, "a request at the cap is sent as it is today");
    }

    #[test]
    fn over_the_cap_keeps_the_newest_and_notes_the_rest() {
        let total = SUMMARIZATION_IMAGE_COUNT + 1;
        let mut messages = spread(total);
        apply(&mut messages, Some(total));
        assert_eq!(count(&messages), total - RETIRE_BATCH);
        assert!(messages[total - 1].content.contains(&picture(total - 1)));
        assert!(!messages[0].content.contains("data:image/"));
        assert!(messages[0].content.contains("data-uri omitted"));
    }

    #[test]
    fn one_message_over_the_cap_retires_inside_it() {
        let total = SUMMARIZATION_IMAGE_COUNT + 4;
        let mut messages = vec![ChatMessage::user(
            (0..total)
                .map(|serial| format!("shot {serial} {}", picture(serial)))
                .collect::<Vec<_>>()
                .join(" and "),
        )];
        apply(&mut messages, None);
        assert_eq!(count(&messages), total - RETIRE_BATCH);
        assert!(messages[0].content.contains(&picture(total - 1)));
        assert!(!messages[0].content.contains(&picture(0)));
    }

    #[test]
    fn paths_links_and_non_user_roles_are_never_retired() {
        let total = SUMMARIZATION_IMAGE_COUNT + 1;
        let mut messages = spread(total);
        messages.push(ChatMessage::user(
            "[IMAGE:/tmp/shot.png] [IMAGE:https://x/y.png]",
        ));
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: picture(0),
        });
        apply(&mut messages, None);
        assert_eq!(count(&messages), total - RETIRE_BATCH);
        let last = messages[messages.len() - 2].content.clone();
        assert_eq!(last, "[IMAGE:/tmp/shot.png] [IMAGE:https://x/y.png]");
        assert_eq!(messages[messages.len() - 1].content, picture(0));
    }

    #[test]
    fn a_known_count_over_the_cap_still_lets_the_messages_decide() {
        let mut messages = spread(RETIRE_BATCH);
        apply(&mut messages, Some(SUMMARIZATION_IMAGE_COUNT + 1));
        assert_eq!(count(&messages), RETIRE_BATCH);
    }

    /// The retirement point is a batch, not a sliding window: pictures appended one at a time leave
    /// the retired prefix alone until the cap is crossed again, which is what keeps the provider's
    /// cached prefix intact between retirements.
    #[test]
    fn the_retirement_point_moves_once_per_batch() {
        let mut messages = spread(SUMMARIZATION_IMAGE_COUNT + 1);
        apply(&mut messages, Some(SUMMARIZATION_IMAGE_COUNT + 1));
        assert_eq!(
            count(&messages),
            SUMMARIZATION_IMAGE_COUNT + 1 - RETIRE_BATCH
        );
        let first_retired = messages[0].content.clone();

        let mut total = SUMMARIZATION_IMAGE_COUNT + 1;
        // Pictures appended one at a time grow the request back to the cap without moving anything.
        while total < SUMMARIZATION_IMAGE_COUNT + RETIRE_BATCH {
            messages.push(ChatMessage::user(picture(total)));
            total += 1;
            apply(&mut messages, Some(total));
        }
        assert_eq!(
            count(&messages),
            SUMMARIZATION_IMAGE_COUNT,
            "the request may sit at the cap between retirements"
        );
        assert_eq!(messages[0].content, first_retired, "nothing older moved");

        messages.push(ChatMessage::user(picture(total)));
        apply(&mut messages, Some(total + 1));
        assert_eq!(
            count(&messages),
            SUMMARIZATION_IMAGE_COUNT + 1 - RETIRE_BATCH,
            "crossing the cap retires the next whole batch"
        );
    }

    /// The dedup key is what the request will carry, so a re-read of a picture the cap retired is
    /// attached again instead of being reported as present.
    #[test]
    fn sent_picture_values_drops_what_the_cap_retires() {
        let total = SUMMARIZATION_IMAGE_COUNT + 1;
        let mut messages = spread(total);
        let sent = super::sent_picture_values(&messages);
        assert_eq!(sent.len(), total - RETIRE_BATCH);
        assert!(!sent.contains(&value(0)), "the oldest stepped aside");
        assert!(sent.contains(&value(total - 1)), "the newest is sent");

        // Exactly the pictures apply leaves in place.
        apply(&mut messages, Some(total));
        let kept: HashSet<String> = (0..total)
            .map(value)
            .filter(|value| {
                let marker = format!("[IMAGE:{value}]");
                messages
                    .iter()
                    .any(|message| message.content.contains(&marker))
            })
            .collect();
        assert_eq!(kept, sent);

        // At or below the cap nothing steps aside, and a path marker is not a picture.
        let mut at_cap = spread(SUMMARIZATION_IMAGE_COUNT);
        at_cap.push(ChatMessage::user("[IMAGE:/tmp/shot.png]"));
        assert_eq!(
            super::sent_picture_values(&at_cap).len(),
            SUMMARIZATION_IMAGE_COUNT
        );
    }
}
