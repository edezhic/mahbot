//! Convert single newlines (CommonMark soft breaks) in chat content into hard
//! line breaks so they render as visible line breaks, without touching code.
//!
//! A chat message entered with Shift+Enter stores a literal newline. The
//! markdown renderer (iced_widget, via the selectable markdown viewer) treats a
//! single newline within a paragraph as a *soft break*, which it renders as a
//! single space — so multi-line messages lose their intended line breaks.
//! Injecting two trailing spaces before that newline turns it into a *hard
//! break*, which the renderer displays as a visible line break.
//!
//! The conversion is code-aware. It reuses pulldown-cmark (the same parser
//! iced_widget's markdown renderer drives) to locate exactly the `SoftBreak`
//! events, which occur only inside prose paragraphs — never inside fenced or
//! indented code blocks or inline code spans, whose newlines are part of code
//! text (a `CodeBlock`/`Code` event, not a `SoftBreak`). Paragraph gaps (blank
//! lines) are block boundaries, not soft breaks, so they are preserved, and an
//! existing hard break (two trailing spaces or a backslash) is left as-is.
//!
//! The module also owns [`escape_html_blocks`], the `<`-escape both chat views
//! run before the parse so angle-bracket text renders literally instead of
//! being dropped as HTML.

use std::borrow::Cow;
use std::ops::Range;

/// Markdown options iced's `markdown::parse` runs with (iced_widget
/// `src/markdown.rs`, `parse_with`). [`escape_html_blocks`] and the soft-break
/// scan in [`hard_breaks`] both use this exact set so they classify regions
/// identically to the real parse; keep in sync if iced's defaults change.
pub(crate) fn iced_markdown_options() -> pulldown_cmark::Options {
    pulldown_cmark::Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | pulldown_cmark::Options::ENABLE_PLUSES_DELIMITED_METADATA_BLOCKS
        | pulldown_cmark::Options::ENABLE_TABLES
        | pulldown_cmark::Options::ENABLE_STRIKETHROUGH
        | pulldown_cmark::Options::ENABLE_TASKLISTS
}

/// Escape `<` → `&lt;` only inside the regions the markdown parser classifies
/// as HTML (block-level `Event::Html` plus inline `Event::InlineHtml`), so
/// angle-bracket tag blocks (`<system-notification>`, `<analyze-tool-result>`,
/// `<timestamp>` …) render as visible literal text instead of being dropped
/// wholesale.
///
/// This is deliberately a `<`-only escape: `util::escape_html` is not reused
/// (it also escapes `>` — breaking blockquotes — and `&`, which would
/// double-encode existing entities).
pub(crate) fn escape_html_blocks(text: &str) -> Cow<'_, str> {
    if !text.contains('<') {
        return Cow::Borrowed(text);
    }
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (event, range) in
        pulldown_cmark::Parser::new_ext(text, iced_markdown_options()).into_offset_iter()
    {
        match event {
            pulldown_cmark::Event::Html(_) | pulldown_cmark::Event::InlineHtml(_) => {
                ranges.push(range);
            }
            _ => {}
        }
    }
    if ranges.is_empty() {
        return Cow::Borrowed(text);
    }
    // Events arrive in document order with disjoint ranges; merge overlapping
    // or adjacent ones so multi-line HTML blocks collapse into single
    // replacement regions.
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = merged.last_mut() {
            if range.start <= last.end {
                last.end = last.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    let mut out = String::with_capacity(text.len() + merged.len() * 4);
    let mut pos = 0;
    for range in &merged {
        out.push_str(&text[pos..range.start]);
        out.push_str(&text[range.clone()].replace('<', "&lt;"));
        pos = range.end;
    }
    out.push_str(&text[pos..]);
    Cow::Owned(out)
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
    fn hard_breaks_cases() {
        let cases: &[(&str, &str, &str)] = &[
            ("prose soft break", "foo\nbar", "foo  \nbar"),
            ("several prose soft breaks", "a\nb\nc", "a  \nb  \nc"),
            ("paragraph gap preserved", "foo\n\nbar", "foo\n\nbar"),
            (
                "soft break before a paragraph gap",
                "a\nb\n\npara2",
                "a  \nb\n\npara2",
            ),
            // fenced code block (incl. a blank line inside it)
            (
                "fenced code untouched",
                "x\n```\ncode\n\nnext\n```\ny",
                "x\n```\ncode\n\nnext\n```\ny",
            ),
            (
                "indented code untouched",
                "    code\n    line2\nplain",
                "    code\n    line2\nplain",
            ),
            (
                "inline code span untouched",
                "foo `a\nb` bar",
                "foo `a\nb` bar",
            ),
            ("crlf normalized", "foo\r\nbar", "foo  \nbar"),
            ("cr normalized", "foo\rbar", "foo  \nbar"),
        ];
        for &(name, input, expected) in cases {
            assert_eq!(hard_breaks(input).as_ref(), expected, "case {name}");
        }
    }
}
