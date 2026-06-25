//! Shared HTML escaping utilities.
//!
//! Provides `escape_html` for full-string escaping and `push_escaped`
//! for use in character-at-a-time hot loops (e.g. Markdown-to-HTML converters).

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
/// # Entity reference
///
/// | Input | Output   |
/// |-------|----------|
/// | `<`   | `&lt;`   |
/// | `>`   | `&gt;`   |
/// | `&`   | `&amp;`  |
/// | `"`   | `&quot;` |
/// | `'`   | `&#39;`  |
/// | other | unchanged |
#[inline]
pub(crate) fn push_escaped(ch: char, out: &mut String) {
    match ch {
        '<' => out.push_str("&lt;"),
        '>' => out.push_str("&gt;"),
        '&' => out.push_str("&amp;"),
        '"' => out.push_str("&quot;"),
        '\'' => out.push_str("&#39;"),
        _ => out.push(ch),
    }
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
}
