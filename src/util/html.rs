//! Shared HTML escaping/unescaping utilities.
//!
//! Provides `escape_html` for full-string escaping, `push_escaped`
//! for use in character-at-a-time hot loops (e.g. Markdown-to-HTML converters),
//! and `decode_html_entities` for the reverse transformation.

/// HTML entity pairs: (character, encoded_entity).
///
/// **Ordering invariant:** `&amp;` must come before `&#39;` before named
/// entities (`&lt;`, `&gt;`, `&quot;`). This ensures double-encoded input
/// like `&amp;#39;` decodes correctly: first `&amp;` → `&`, yielding `&#39;`,
/// then `&#39;` → `'`. Similarly `&amp;lt;` → `&lt;` → `<`.
///
/// When adding new entities, append them **after** the existing entries to
/// preserve the current decode ordering.
const HTML_ENTITIES: &[(char, &str)] = &[
    ('&', "&amp;"),
    ('\'', "&#39;"),
    ('<', "&lt;"),
    ('>', "&gt;"),
    ('"', "&quot;"),
];

/// Escape HTML-special characters as their corresponding entities.
///
/// Handles `&`, `<`, `>`, `"`, and `'` → `&amp;`, `&lt;`, `&gt;`,
/// `&quot;`, `&#39;` respectively.
#[must_use]
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        push_escaped(ch, &mut out);
    }
    out
}

/// Push a single character to the output buffer, escaping HTML-special
/// characters as their corresponding entities.
///
/// Entity mappings are sourced from [`HTML_ENTITIES`]; any character not
/// in the map is pushed unchanged.
#[inline]
pub(crate) fn push_escaped(ch: char, out: &mut String) {
    for &(c, entity) in HTML_ENTITIES {
        if ch == c {
            out.push_str(entity);
            return;
        }
    }
    out.push(ch);
}

/// Decode the 5 standard HTML entities back to their literal characters.
///
/// # Decode order
///
/// Entities are decoded in [`HTML_ENTITIES`] order (`&amp;` → `&#39;` →
/// named entities), which correctly handles double-encoded input such as
/// `&amp;#39;` (→ `&#39;` → `'`) and `&amp;lt;` (→ `&lt;` → `<`).
///
/// # Fast path
///
/// Returns `s.to_string()` immediately when no `&` is present, avoiding
/// allocations in the common case of plain text.
#[must_use]
pub(crate) fn decode_html_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut result = s.to_string();
    for &(ch, entity) in HTML_ENTITIES {
        let mut buf = [0u8; 4];
        let replacement = ch.encode_utf8(&mut buf);
        result = result.replace(entity, replacement);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_html_all_special_chars() {
        // All five special characters in one string
        let r = escape_html("<div class=\"test\">AT&T 'hello'</div>");
        assert_eq!(
            r,
            "&lt;div class=&quot;test&quot;&gt;AT&amp;T &#39;hello&#39;&lt;/div&gt;"
        );
    }

    #[test]
    fn escape_html_no_special_chars() {
        // Plain text with no special characters passes through unchanged
        let r = escape_html("hello world 123");
        assert_eq!(r, "hello world 123");
    }

    #[test]
    fn escape_html_empty_string() {
        let r = escape_html("");
        assert_eq!(r, "");
    }

    #[test]
    fn escape_html_only_ampersand() {
        // The order-dependency test: & must not be double-escaped
        let r = escape_html("&amp; &lt; &gt; &quot; &#39;");
        assert_eq!(r, "&amp;amp; &amp;lt; &amp;gt; &amp;quot; &amp;#39;");
    }

    // --- decode_html_entities tests (moved from telegram.rs) ---

    #[test]
    fn decode_html_entities_no_change() {
        // Fast path: no ampersand → returned unchanged (also covers empty string)
        let r = decode_html_entities("hello world 123");
        assert_eq!(r, "hello world 123");
        assert_eq!(decode_html_entities(""), "");
    }

    #[test]
    fn decode_html_entities_lone_ampersand() {
        // Ampersand with no valid entity passes through unchanged
        let r = decode_html_entities("a & b");
        assert_eq!(r, "a & b");
    }

    #[test]
    fn decode_html_entities_all() {
        // All five entities decoded in realistic text
        let r = decode_html_entities("say &quot;hi&quot; &amp; &lt;tag&gt; &#39;ok&#39;");
        assert_eq!(r, "say \"hi\" & <tag> 'ok'");
    }

    #[test]
    fn decode_html_entities_double_encoded() {
        // Order dependency: &amp; before &#39; so &amp;#39; → &#39; → '
        let r = decode_html_entities("&amp;#39;");
        assert_eq!(r, "'");
    }

    // --- Round-trip: decode(encode(c)) == c for all entities ---

    #[test]
    fn round_trip_all_entities() {
        for &(ch, _entity) in HTML_ENTITIES {
            let encoded = {
                let mut s = String::new();
                push_escaped(ch, &mut s);
                s
            };
            let decoded = decode_html_entities(&encoded);
            assert_eq!(decoded, ch.to_string(), "round-trip failed for {ch:?}");
        }
    }

    #[test]
    fn round_trip_all_entities_in_context() {
        // Verify that encoding and then decoding a string containing all special
        // characters returns the original string.
        let original = "<div class=\"test\">AT&T 'hello'</div>";
        let encoded = escape_html(original);
        let decoded = decode_html_entities(&encoded);
        assert_eq!(decoded, original);
    }
}
