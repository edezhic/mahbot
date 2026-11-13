//! Leaked model control tokens in the session history: the repair plan for a refused session.
//!
//! The outbound boundary ([`crate::util::control_token`]) is the first line of defence — a fresh
//! request never carries the token shape. A token already persisted is still sticky: every later
//! turn re-sends it, and a provider that refuses the request's *content* leaves the session dead
//! until it is reset. This module turns that refusal class into the row rewrites that clear it, so
//! the run continues through its normal loop instead of failing forever.

use std::borrow::Cow;

use crate::ChatMessage;
use crate::retry::RetryExhausted;
use crate::util::control_token;

/// The repair plan for a history a provider refused: every token-bearing row's index paired with
/// its neutralized content, or empty when no repair applies — a retryable failure class says
/// nothing about the history's content, and a refusal over a token-free history is a different
/// problem. Every matching row is returned: one leaked token can be re-emitted into a later reply.
#[must_use]
pub(crate) fn leaked_token_rewrites(
    exhausted: &RetryExhausted,
    history: &[ChatMessage],
) -> Vec<(usize, String)> {
    if !exhausted.final_class.is_deterministic() {
        return Vec::new();
    }
    history
        .iter()
        .enumerate()
        .filter_map(
            |(index, message)| match control_token::neutralize(&message.content) {
                Cow::Borrowed(_) => None,
                Cow::Owned(cleaned) => Some((index, cleaned)),
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::{FailureClass, RetryFailureRecord};

    /// A single-failure trail of the given class, the shape a provider refusal leaves.
    fn trail(class: FailureClass) -> RetryExhausted {
        let inner = anyhow::anyhow!("the provider refused this request");
        RetryExhausted::with_last_raw(
            vec![RetryFailureRecord::new_simple(class, &inner, None)],
            class,
            None,
        )
    }

    /// A token in a model reply or a tool result is found at its own index — the two places a
    /// leaked token actually lands, and neither is a user message.
    #[test]
    fn detection_returns_the_rewrite_of_every_token_bearing_message() {
        let history = vec![
            ChatMessage::system("role description"),
            ChatMessage::user("a clean question"),
            ChatMessage::assistant("here it is <|image|> as text"),
            ChatMessage::tool_result("call_1", "tool result with <|endoftext|> inside"),
            ChatMessage::assistant("a clean reply"),
        ];
        for class in [
            FailureClass::ContentRejection,
            FailureClass::Guardrail,
            FailureClass::InvalidRequest,
        ] {
            let rewrites = leaked_token_rewrites(&trail(class), &history);
            assert_eq!(
                rewrites.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
                vec![2, 3],
                "every token-bearing row is reported for {class:?}"
            );
            assert_eq!(
                rewrites[0].1, "here it is [model-control-token] as text",
                "a reply is rewritten in place for {class:?}"
            );
            assert!(
                !rewrites[1].1.contains("<|"),
                "a tool-result envelope is rewritten for {class:?}"
            );
        }
    }

    /// A retryable class says nothing about the history's content: the transport failure that
    /// preceded a refusal must not trigger a rewrite.
    #[test]
    fn detection_does_not_fire_on_a_retryable_class() {
        let history = vec![
            ChatMessage::user("question"),
            ChatMessage::assistant("reply with <|image|> in it"),
        ];
        assert!(leaked_token_rewrites(&trail(FailureClass::Transport), &history).is_empty());
        assert!(leaked_token_rewrites(&trail(FailureClass::Parse), &history).is_empty());
    }

    /// A refusal over a token-free history is some other problem: nothing to repair.
    #[test]
    fn detection_does_not_fire_without_a_token() {
        let history = vec![
            ChatMessage::system("role description"),
            ChatMessage::user("a clean question"),
            ChatMessage::assistant("a clean reply"),
        ];
        assert!(leaked_token_rewrites(&trail(FailureClass::ContentRejection), &history).is_empty());
        // Prose that looks like an unclosed envelope is not a token.
        let prose = vec![ChatMessage::user("compare a <| b here")];
        assert!(leaked_token_rewrites(&trail(FailureClass::ContentRejection), &prose).is_empty());
    }

    /// The repair removes every token it matched, so a second plan over the repaired history is
    /// empty — the loop-safety guarantee the caller relies on.
    #[test]
    fn a_repaired_history_cannot_produce_a_second_repair() {
        let history = vec![ChatMessage::assistant("reply with <|image|> in it")];
        let trail = trail(FailureClass::ContentRejection);
        let rewrites = leaked_token_rewrites(&trail, &history);
        assert_eq!(
            rewrites,
            vec![(0, "reply with [model-control-token] in it".to_string())]
        );

        let repaired = vec![ChatMessage::assistant(rewrites[0].1.clone())];
        assert!(leaked_token_rewrites(&trail, &repaired).is_empty());
    }
}
