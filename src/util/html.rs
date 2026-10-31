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

    /// Every special character is written as its entity, and `&` is escaped
    /// before anything else so an already-encoded document is not
    /// double-decoded: `&amp;` becomes `&amp;amp;`, not `&`.
    #[test]
    fn escape_html_writes_every_special_character_as_its_entity() {
        let cases: [(&str, &str); 2] = [
            (
                "<div class=\"test\">AT&T 'hello'</div>",
                "&lt;div class=&quot;test&quot;&gt;AT&amp;T &#39;hello&#39;&lt;/div&gt;",
            ),
            (
                "&amp; &lt; &gt; &quot; &#39;",
                "&amp;amp; &amp;lt; &amp;gt; &amp;quot; &amp;#39;",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(escape_html(input), expected, "input: {input:?}");
        }
    }

    /// All five entities decode, in the order that makes double-encoded input
    /// come out right: `&amp;#39;` decodes `&amp;` first, yielding `&#39;`,
    /// which the next pass turns into `'`.
    #[test]
    fn decode_html_entities_walks_the_table_in_order() {
        let cases: [(&str, &str); 2] = [
            (
                "say &quot;hi&quot; &amp; &lt;tag&gt; &#39;ok&#39;",
                "say \"hi\" & <tag> 'ok'",
            ),
            ("&amp;#39;", "'"),
        ];
        for (input, expected) in cases {
            assert_eq!(decode_html_entities(input), expected, "input: {input:?}");
        }
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
}
