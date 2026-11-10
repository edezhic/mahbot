//! Shared HTML escaping/unescaping utilities. `push_escaped` is the
//! character-at-a-time form for hot loops such as Markdown converters.

/// HTML entity pairs: (character, encoded_entity).
///
/// **Ordering invariant:** `&amp;` must come before `&#39;` before the named
/// entities, else double-encoded input (`&amp;#39;`, `&amp;lt;`) decodes wrong.
/// Append new entities **after** the existing entries.
const HTML_ENTITIES: &[(char, &str)] = &[
    ('&', "&amp;"),
    ('\'', "&#39;"),
    ('<', "&lt;"),
    ('>', "&gt;"),
    ('"', "&quot;"),
];

#[must_use]
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        push_escaped(ch, &mut out);
    }
    out
}

/// Push one character, escaped per [`HTML_ENTITIES`]; others pass through.
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

/// Decode the five entities back to characters, in [`HTML_ENTITIES`] order so
/// double-encoded input (`&amp;#39;`) comes out right. A string with no `&` is
/// returned as-is, skipping the work.
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
