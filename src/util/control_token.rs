//! Model control tokens that leak into plain text.
//!
//! A model can emit one of its own control tokens as ordinary reply text (observed: an image
//! token, which a provider then rejects with an invalid-request error on every later request
//! carrying it — the rejection is sticky because the text lives in the session history forever).
//! Such a token must never leave the product toward a provider, and must never be handed out raw
//! to an agent, a ticket or the debug console.
//!
//! One shape, one replacement: [`CONTROL_TOKEN_RE`] matches the token envelope (ASCII or fullwidth
//! bars) and [`PLACEHOLDER`] replaces it. The replacement is deterministic and idempotent — the
//! request body is rebuilt on every retry attempt, so an unstable or absent substitution would
//! either break the provider's prompt-prefix cache or re-expose the token — and it is plain ASCII
//! with no quote or backslash, so it cannot break a JSON envelope it lands in (a tool call's
//! JSON-string arguments included).

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

/// What a token becomes. Visible rather than deleted (a silently removed span would change the
/// text's meaning without saying so), and not parseable as one of the product's own attachment
/// markers, which all require a `KIND:` prefix inside the brackets.
pub(crate) const PLACEHOLDER: &str = "[model-control-token]";

/// The token envelope: `<` + bar + inner payload + bar + `>`. The payload is deliberately
/// unbounded — control tokens differ in length between models, and a miss here leaves the session
/// permanently refused, while a false positive only costs a visible placeholder. `[^>]` keeps a
/// match from crossing the end of a bracketed span.
static CONTROL_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<[|｜][^>]*[|｜]>").expect("CONTROL_TOKEN_RE must compile"));

/// Replace every control token in `text` with [`PLACEHOLDER`], borrowing when there is none — so
/// `Cow::Owned` is itself the "this text changed" answer.
#[must_use]
pub(crate) fn neutralize(text: &str) -> Cow<'_, str> {
    CONTROL_TOKEN_RE.replace_all(text, PLACEHOLDER)
}

/// [`neutralize`] applied in place, writing only when something matched.
pub(crate) fn neutralize_in_place(text: &mut String) {
    if let Cow::Owned(cleaned) = neutralize(text) {
        *text = cleaned;
    }
}

/// [`neutralize`] for a string the caller already owns.
#[must_use]
pub(crate) fn neutralize_owned(text: String) -> String {
    let mut text = text;
    neutralize_in_place(&mut text);
    text
}

/// [`neutralize`] every string inside a JSON value, in place. The outbound request's tool
/// descriptions and parameter schemas are the reason this exists: they are free text assembled from
/// prompt assets and admin-defined custom tools, so no per-field walk can promise coverage. Object
/// keys are left alone — they are schema property names and provider vocabulary, never model text
/// (a tool call's JSON-string `arguments` is a single string leaf, so its own keys are covered by
/// the rewrite of that leaf).
pub(crate) fn neutralize_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(text) => neutralize_in_place(text),
        serde_json::Value::Array(items) => items.iter_mut().for_each(neutralize_json),
        serde_json::Value::Object(map) => map.values_mut().for_each(neutralize_json),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_token_envelope_is_replaced_with_the_placeholder() {
        let token = "<|image|>";
        assert_eq!(
            neutralize(&format!("a {token} b")),
            "a [model-control-token] b"
        );
        // Fullwidth bars, and an arbitrarily long inner payload: the length is a property of the
        // model, not of the shape.
        assert_eq!(neutralize("<｜deepseek_special｜>"), PLACEHOLDER);
        let long_inner = format!("<|{}|>", "x".repeat(500));
        assert_eq!(neutralize(&long_inner), PLACEHOLDER);
    }

    #[test]
    fn the_replacement_is_idempotent_and_borrows_when_clean() {
        let token = "<|image|>";
        let once = neutralize(&format!("x {token} y")).into_owned();
        assert_eq!(neutralize(&once), once, "a second pass must change nothing");
        assert!(matches!(neutralize("plain text"), Cow::Borrowed(_)));
        assert!(matches!(neutralize(&once), Cow::Borrowed(_)));
        assert_eq!(neutralize_owned(once.clone()), once);
    }

    #[test]
    fn product_markers_and_ordinary_text_are_untouched() {
        for text in [
            "[IMAGE:/tmp/a.png]",
            "[VIDEO:https://example.test/v.mp4]",
            "a < b | c > d",
            "<|unclosed",
            "|>",
        ] {
            assert!(
                matches!(neutralize(text), Cow::Borrowed(_)),
                "{text} must not be rewritten"
            );
        }
    }

    /// The placeholder must never be produced in a form that breaks the JSON a tool call's
    /// arguments carry as a string: it lands inside `{"q":"<|image|>"}` and that string must still
    /// parse back to an object. Object keys are provider/schema vocabulary and stay untouched.
    #[test]
    fn neutralizing_json_keeps_a_nested_arguments_string_parseable() {
        let mut value = serde_json::json!({
            "q": "look <|image|> here",
            "nested": ["<|a|>", 1, null],
            "schema": { "note": { "description": "x <|b|>" } },
        });
        neutralize_json(&mut value);
        let arguments = serde_json::to_string(&value).unwrap();
        assert!(!arguments.contains("<|"));
        let reparsed: serde_json::Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(reparsed["q"], format!("look {PLACEHOLDER} here"));
        assert_eq!(
            reparsed["schema"]["note"]["description"],
            format!("x {PLACEHOLDER}")
        );
    }
}
