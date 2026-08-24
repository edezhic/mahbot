//! Convert single newlines (CommonMark soft breaks) in chat content into hard
//! line breaks so they render as visible line breaks, without touching code.
//!
//! A chat message entered with Shift+Enter stores a literal newline. The
//! markdown renderer (iced_widget, via `iced_selection`) treats a single
//! newline within a paragraph as a *soft break*, which it renders as a single
//! space — so multi-line messages lose their intended line breaks. Injecting
//! two trailing spaces before that newline turns it into a *hard break*, which
//! the renderer displays as a visible line break.
//!
//! The conversion is code-aware. It reuses pulldown-cmark (the same parser
//! iced_widget's markdown renderer drives) to locate exactly the `SoftBreak`
//! events, which occur only inside prose paragraphs — never inside fenced or
//! indented code blocks or inline code spans, whose newlines are part of code
//! text (a `CodeBlock`/`Code` event, not a `SoftBreak`). Paragraph gaps (blank
//! lines) are block boundaries, not soft breaks, so they are preserved, and an
//! existing hard break (two trailing spaces or a backslash) is left as-is.

use std::borrow::Cow;

/// Markdown options iced's `markdown::parse` runs with (iced_widget
/// `src/markdown.rs`, `parse_with`). The Sessions `escape_html_blocks`
/// pre-scan and the soft-break scan in [`hard_breaks`] both use this exact set
/// so they classify regions identically to the real parse; keep in sync if
/// iced's defaults change.
pub(crate) fn iced_markdown_options() -> pulldown_cmark::Options {
    pulldown_cmark::Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | pulldown_cmark::Options::ENABLE_PLUSES_DELIMITED_METADATA_BLOCKS
        | pulldown_cmark::Options::ENABLE_TABLES
        | pulldown_cmark::Options::ENABLE_STRIKETHROUGH
        | pulldown_cmark::Options::ENABLE_TASKLISTS
}

/// Convert single-newline soft breaks into hard breaks so multi-line chat
/// messages render with visible line breaks, leaving code text untouched.
///
/// Returns a borrowed `Cow` when the text has no line endings (the common
/// single-line fast path) or when a scan finds no prose soft breaks (e.g. the
/// only newlines are code/paragraph boundaries); otherwise it owns a new
/// string with two trailing spaces inserted before each soft break.
pub(crate) fn hard_breaks(text: &str) -> Cow<'_, str> {
    // Fast path: a single-line message needs no scan.
    if !text.contains('\n') && !text.contains('\r') {
        return Cow::Borrowed(text);
    }

    // Normalize CRLF / CR to LF so a lone '\r' (CommonMark also treats it as a
    // line ending) becomes a single newline rather than a raw character, and a
    // CRLF is not split into a paragraph gap.
    let normalized: Cow<'_, str> = if text.contains('\r') {
        Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(text)
    };
    let src = normalized.as_ref();

    let positions = soft_break_positions(src);
    if positions.is_empty() {
        return normalized;
    }

    // Insert two trailing spaces before each soft-break newline. CommonMark
    // turns "line  \n" into a hard break (the trailing spaces are consumed, so
    // they never appear as visible characters; only the line break shows).
    let mut out = String::with_capacity(src.len() + positions.len() * 2);
    let mut last = 0usize;
    for pos in positions {
        out.push_str(&src[last..pos]);
        out.push_str("  ");
        last = pos;
    }
    out.push_str(&src[last..]);
    Cow::Owned(out)
}

/// Byte offsets of the `SoftBreak` events in `src`.
///
/// A soft break is a single newline inside a prose paragraph. Newlines inside
/// fenced/indented code blocks and inline code spans are never soft breaks
/// (they are part of code text), and paragraph gaps / existing hard breaks are
/// block boundaries or `HardBreak` events, so they are automatically excluded.
fn soft_break_positions(src: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    for (event, range) in
        pulldown_cmark::Parser::new_ext(src, iced_markdown_options()).into_offset_iter()
    {
        if let pulldown_cmark::Event::SoftBreak = event {
            positions.push(range.start);
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::hard_breaks;

    #[test]
    fn hard_breaks_converts_prose_soft_breaks() {
        assert_eq!(hard_breaks("foo\nbar").as_ref(), "foo  \nbar");
        assert_eq!(hard_breaks("a\nb\nc").as_ref(), "a  \nb  \nc");
    }

    #[test]
    fn hard_breaks_preserves_paragraph_gaps() {
        assert_eq!(hard_breaks("foo\n\nbar").as_ref(), "foo\n\nbar");
        assert_eq!(hard_breaks("a\nb\n\npara2").as_ref(), "a  \nb\n\npara2");
    }

    #[test]
    fn hard_breaks_leaves_code_untouched() {
        // fenced code block (incl. a blank line inside it)
        assert_eq!(
            hard_breaks("x\n```\ncode\n\nnext\n```\ny").as_ref(),
            "x\n```\ncode\n\nnext\n```\ny"
        );
        // indented code block
        assert_eq!(
            hard_breaks("    code\n    line2\nplain").as_ref(),
            "    code\n    line2\nplain"
        );
        // inline code span
        assert_eq!(hard_breaks("foo `a\nb` bar").as_ref(), "foo `a\nb` bar");
    }

    #[test]
    fn hard_breaks_normalizes_crlf_and_cr() {
        assert_eq!(hard_breaks("foo\r\nbar").as_ref(), "foo  \nbar");
        assert_eq!(hard_breaks("foo\rbar").as_ref(), "foo  \nbar");
    }
}
