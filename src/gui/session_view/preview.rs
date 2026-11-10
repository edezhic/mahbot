//! Session transcript message collapse measurement.
//!
//! The sessions page collapses messages that render more than [`MAX_PREVIEW_LINES`]
//! wrapped lines into a short plain-text preview with an expand/collapse toggle. This
//! module measures that: a flat wrapped-line count of the display text at body metrics and
//! the actual transcript content width.
//!
//! The measurement errs toward collapsing:
//! - a safety margin narrows the measured width, so markdown structure the flat model cannot
//!   see (list/quote indentation) forces an extra wrap instead of slipping a 4-line message
//!   through;
//! - heading lines (ATX and setext) are measured at their rendered heading size;
//! - fenced code-block lines count as one source line each (the renderer places them
//!   unwrapped in a horizontally scrollable block), fence delimiters included — a 2-line code
//!   block with fences measures 4 and collapses;
//! - a markdown body has its media markers and pictures extracted into attachments before
//!   measuring, their labels measuring where the expanded body draws them (above the text),
//!   so a base64 data-URI cannot skew the count; a body drawn as raw text measures it unchanged.
//!
//! Preview extraction is BiDi-safe: cosmic-text lays glyphs out in visual order (L2
//! reordering), so an RTL run can appear with decreasing byte offsets. Each wrapped line's
//! preview segment is therefore the logical span (min start .. max end) of its glyphs — never
//! a `start > end` slice.
//!
//! Shaping is chunked per source line with early exit once the budget is exceeded, so a
//! message of hundreds of KB costs a few shaped lines. A source line longer than the
//! worst-case budget is shaped as a bounded prefix; when that proves the collapse its first
//! lines come from true wrap points, and when it cannot (a giant unbreakable word) the count
//! saturates past the budget. Every preview line is capped to the characters that fit the
//! transcript width in the body font, so the 3-line preview renders as at most 3 visual
//! lines. The result is cached per message and width by
//! [`crate::gui::sessions::SessionsState`].

use std::path::Path;
use std::sync::Arc;

use cosmic_text::{Attrs, AttrsList, BufferLine, Family, LineEnding, Shaping, Wrap};
use iced::advanced::graphics::text::cosmic_text;

use crate::gui::media_markers;
use crate::gui::text_rendering::with_font_system;
#[cfg(test)]
use crate::gui::theme::MARKDOWN_TEXT_SIZE;

/// Line budget: a message rendering more wrapped lines than this collapses to a
/// [`MAX_PREVIEW_LINES`]-line preview.
pub(crate) const MAX_PREVIEW_LINES: u32 = 3;

/// Safety margin (px) subtracted from the available width before measuring.
///
/// The flat measurement cannot see markdown list/quote indentation, so lines close to the
/// boundary wrap in the real render but not here. Narrowing the width forces them to wrap a
/// line earlier — err toward collapsing, never leaving an over-budget message expanded.
pub(crate) const WRAP_SAFETY_MARGIN_PX: f32 = 36.0;

/// Maximum characters per preview line segment from a laid line.
///
/// Laid lines that fit are already bounded by the width (~59 mono chars), and even a giant
/// unbreakable word renders as one overflowing line in the preview, so this is a hard safety
/// bound only. The width-aware [`preview_line_char_budget`] cap applies separately to
/// unwrapped fence/code lines, which wrap at spaces in the preview and would otherwise spill
/// past the 3-line budget.
const MAX_PREVIEW_LINE_CHARS: usize = 600;

/// Markdown heading sizes relative to the measurement body size, matching iced 0.14's
/// `markdown::Settings` scale exactly (h1 = 2.0×, h2 = 1.75×, h3 = 1.5×, h4 = 1.25×, h5/h6 =
/// body). Derived from the caller's `font_size` so heading wrap stays in lockstep with the
/// renderer's scaled headings.
const fn heading_sizes(font_size: f32) -> [f32; 6] {
    [
        font_size * 2.0,
        font_size * 1.75,
        font_size * 1.5,
        font_size * 1.25,
        font_size,
        font_size,
    ]
}

/// Tab width used when shaping (matches cosmic-text's default).
const TAB_WIDTH: u16 = 4;

/// Upper bound (bytes) on the shaped prefix of a single source line. A laid line holds at most
/// `avail` characters (1 px/char worst case) and a character at most 4 bytes, so
/// `need * avail * 4` bytes cannot fit in fewer than `need` lines — a longer line definitely
/// wraps past the budget. The +4 covers the char-boundary floor of the cut.
fn shape_ceiling(avail: f32, need: u32) -> usize {
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let per_line = (avail as usize).max(1) * 4;
    per_line.saturating_mul(need as usize).saturating_add(4)
}

/// Maximum characters per preview line, measured from the body font's glyph advance (mono:
/// uniform) so it tracks the real rendered width. The truncate ellipsis counts inside the cap,
/// so a capped line never exceeds the transcript width and cannot wrap to a 4th visual line.
fn preview_line_char_budget(
    font_sys: &mut cosmic_text::FontSystem,
    attrs: &Attrs,
    avail: f32,
    size: f32,
) -> usize {
    let mut probe = BufferLine::new(
        "M",
        LineEnding::None,
        AttrsList::new(attrs),
        Shaping::Advanced,
    );
    let advance = probe
        .shape(font_sys, TAB_WIDTH)
        .layout(size, Some(1_000.0), Wrap::None, None, None)
        .first()
        .map_or(0.0, |l| l.w);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let budget = (avail / advance.max(1.0)) as usize;
    budget.max(1)
}

/// Per-message measurement result, cached in [`SessionsState`] keyed by (message index, width
/// bucket). [`SessionsState`]: crate::gui::sessions::SessionsState
pub(crate) struct MessageMeasure {
    /// Width bucket the measurement was computed for (see [`width_bucket`]).
    pub(crate) width_bucket: u32,
    pub(crate) wrapped_lines: u32,
    /// First [`MAX_PREVIEW_LINES`] wrapped lines as plain text — `Some` only when the message
    /// collapses.
    pub(crate) preview: Option<String>,
    /// The display text this measurement came from, so a width change re-measures without
    /// re-decoding the message.
    pub(crate) source: Arc<str>,
    /// The media-placeholder-reduced text the measurement actually used, so a width-bucket
    /// change re-measures via [`re_measure`] without re-running media preprocessing on every
    /// resize. A [`BodyText::Plain`] body's processed text is its source.
    pub(crate) processed: Arc<str>,
    /// Body font size (px) the measurement was computed at; carried forward by
    /// [`re_measure`], so a narration body keeps the narration size on a resize.
    pub(crate) font_size: f32,
}

/// Quantize a text width into a bucket so sub-pixel layout jitter does not invalidate the
/// cache; a real resize moves the bucket and triggers re-measurement.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn width_bucket(width: f32) -> u32 {
    (width.max(0.0) / 16.0) as u32
}

/// How a collapsible body draws its text, so the measurement matches what is drawn: a markdown
/// body has its markers extracted into attachment labels, a plain one (a tool result, a thinking
/// block) shows the raw text.
#[derive(Clone, Copy)]
pub(crate) enum BodyText<'a> {
    /// A bubble body parsed as markdown; relative markers resolve against the workspace.
    Markdown(Option<&'a Path>),
    Plain,
}

/// Measure the wrapped line count of a body at `width`, producing a
/// [`MAX_PREVIEW_LINES`]-line plain-text preview when the message exceeds the budget.
///
/// `display_text` is the raw body (the text the Sessions ledger's `parse_entry_bodies` feeds to
/// `markdown::parse`); [`BodyText`] says how it is drawn, so a markdown body measures
/// [`media_markers::Preprocessed::display_text`] (attachment labels above the text, the order
/// the expanded body draws) and a plain one its raw text. The source is owned as [`Arc<str>`]
/// so a width-bucket change re-measures without copying it. `font_size` is the body size the
/// text renders at (body or narration), used for wrap and preview shaping and heading scaling.
pub(crate) fn measure_message(
    display_text: impl Into<Arc<str>>,
    body: BodyText<'_>,
    width: f32,
    font_size: f32,
) -> MessageMeasure {
    let source = display_text.into();
    let processed: Arc<str> = match body {
        BodyText::Markdown(workspace) => media_markers::preprocess(&source, workspace)
            .display_text()
            .into(),
        BodyText::Plain => source.clone(),
    };
    measure_processed(source, processed, width, font_size)
}

/// Re-measure a cached entry at a new width, reusing the cached processed text and
/// [`MessageMeasure::font_size`].
pub(crate) fn re_measure(cached: &MessageMeasure, width: f32) -> MessageMeasure {
    measure_processed(
        cached.source.clone(),
        cached.processed.clone(),
        width,
        cached.font_size,
    )
}

/// Shared core of [`measure_message`] / [`re_measure`]: lay the already media-processed text at
/// `width` and count wrapped lines.
#[expect(clippy::too_many_lines)]
fn measure_processed(
    source: Arc<str>,
    processed: Arc<str>,
    width: f32,
    font_size: f32,
) -> MessageMeasure {
    let bucket = width_bucket(width);
    let text = &processed;
    let avail = (width - WRAP_SAFETY_MARGIN_PX).max(1.0);

    // Family-only attrs: italic shares the mono glyph advance, so no style change is needed.
    let attrs = Attrs::new().family(Family::Name("JetBrains Mono"));
    let mut total_lines = 0u32;
    let mut preview = String::new();
    let mut preview_lines = 0u32;
    // Current fenced code block: (delimiter char, minimum closing length).
    let mut fence: Option<(u8, usize)> = None;

    with_font_system(|font_sys| {
        // Measured from the body font's glyph advance rather than hardcoded: a longer preview
        // line would wrap, or spill past the 3-line budget visually.
        let preview_line_chars = preview_line_char_budget(font_sys, &attrs, avail, font_size);
        let mut line = BufferLine::new(
            "",
            LineEnding::None,
            AttrsList::new(&attrs),
            Shaping::Advanced,
        );

        // Peekable so a setext heading's size can look at the underline on the next line.
        let mut lines = text.split('\n').peekable();
        while let Some(src_line) = lines.next() {
            let next_line = lines.peek().copied();
            // Early exit: the budget is exceeded, so the message collapses — stop shaping.
            if total_lines > MAX_PREVIEW_LINES {
                break;
            }

            match fence {
                Some((open_ch, open_count)) => {
                    // Every source line counts once (the renderer places code lines unwrapped
                    // in a scrollable block); a matching delimiter line closes the block.
                    let closing = fence_delimiter(src_line)
                        .is_some_and(|(ch, count)| ch == open_ch && count >= open_count)
                        && fence_has_only_trailing_whitespace(src_line);
                    if closing {
                        fence = None;
                    }
                    count_source_line(
                        src_line,
                        &mut total_lines,
                        &mut preview,
                        &mut preview_lines,
                        preview_line_chars,
                    );
                }
                None => {
                    if let Some((ch, count)) = fence_delimiter(src_line) {
                        // Opening fence line.
                        fence = Some((ch, count));
                        count_source_line(
                            src_line,
                            &mut total_lines,
                            &mut preview,
                            &mut preview_lines,
                            preview_line_chars,
                        );
                    } else {
                        // Wrappable line: shape and lay it at its rendered font size (headings
                        // render larger than the body).
                        let size = heading_size(src_line, font_size)
                            .or_else(|| setext_heading_size(src_line, next_line, font_size))
                            .unwrap_or(font_size);
                        // A source line over `shape_ceiling(avail, need)` bytes definitely wraps
                        // past the budget, so a bounded prefix proves the collapse (a giant
                        // unbreakable word saturates below). Shaping a 200KB single-line tool
                        // result in full on the UI thread would jank the transcript.
                        let need = MAX_PREVIEW_LINES + 1 - total_lines;
                        let ceiling = shape_ceiling(avail, need);
                        let shaped_src = if src_line.len() > ceiling {
                            &src_line[..src_line.floor_char_boundary(ceiling)]
                        } else {
                            src_line
                        };
                        line.reset_new(
                            shaped_src,
                            LineEnding::None,
                            AttrsList::new(&attrs),
                            Shaping::Advanced,
                        );
                        let laid = line.shape(font_sys, TAB_WIDTH).layout(
                            size,
                            Some(avail),
                            Wrap::Word,
                            None,
                            None,
                        );
                        let truncated = src_line.len() > ceiling;
                        // The prefix did not prove the collapse (fewer than `need` lines), so the
                        // rest can only add wrapped lines — err toward collapsing.
                        let line_count = if truncated && laid.len() < need as usize {
                            need
                        } else {
                            u32::try_from(laid.len()).unwrap_or(u32::MAX)
                        };
                        total_lines = total_lines.saturating_add(line_count);
                        for laid_line in laid {
                            if preview_lines >= MAX_PREVIEW_LINES {
                                break;
                            }
                            // Glyphs are laid out in visual (BiDi-reordered) order: inside an
                            // RTL run byte offsets decrease, so a wrapped line can begin after it
                            // ends — `first().start` may exceed `last().end`, which panicked the
                            // old first..last slice. Take the logical span (min start .. max
                            // end): the correct text in every direction case.
                            let (glyph_start, glyph_end) = laid_line
                                .glyphs
                                .iter()
                                .fold((usize::MAX, 0usize), |(s, e), g| {
                                    (s.min(g.start), e.max(g.end))
                                });
                            if preview_lines > 0 {
                                preview.push('\n');
                            }
                            if glyph_start <= glyph_end && glyph_start < shaped_src.len() {
                                let start = glyph_start.min(shaped_src.len());
                                let end = glyph_end.min(shaped_src.len());
                                // A fitting laid line is already bounded by the width; only one
                                // overflowing it needs a cap. That cap hits genuinely unbreakable
                                // words (a minified JSON blob), which would otherwise render far
                                // past the width; capping lines with break opportunities would
                                // truncate mid-word an RTL phrase that straddles the wrap.
                                let segment = &shaped_src[start..end];
                                let cap = if laid_line.w > avail
                                    && !segment.chars().any(char::is_whitespace)
                                {
                                    preview_line_chars
                                } else {
                                    MAX_PREVIEW_LINE_CHARS
                                };
                                preview.push_str(&crate::util::truncate(segment, cap));
                            }
                            preview_lines += 1;
                        }
                    }
                }
            }
        }
    });

    MessageMeasure {
        width_bucket: bucket,
        wrapped_lines: total_lines,
        preview: (total_lines > MAX_PREVIEW_LINES).then_some(preview),
        source,
        processed,
        font_size,
    }
}

/// Append one source line to the running preview, counting it as a single rendered line.
/// `preview_line_chars` caps each preview line to what fits the width — code lines render
/// unwrapped in the transcript but are plain wrapping text in the preview.
fn count_source_line(
    src_line: &str,
    total_lines: &mut u32,
    preview: &mut String,
    preview_lines: &mut u32,
    preview_line_chars: usize,
) {
    *total_lines += 1;
    if *preview_lines < MAX_PREVIEW_LINES {
        if *preview_lines > 0 {
            preview.push('\n');
        }
        preview.push_str(&crate::util::truncate(
            src_line.trim_end(),
            preview_line_chars,
        ));
        *preview_lines += 1;
    }
}

/// Detect a fenced code-block delimiter line (```` ``` ```` or `~~~`, at least 3 chars,
/// optionally indented). Returns the delimiter char and run length.
fn fence_delimiter(line: &str) -> Option<(u8, usize)> {
    let trimmed = line.trim_start();
    let ch = *trimmed.as_bytes().first()?;
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let count = trimmed.bytes().take_while(|&b| b == ch).count();
    (count >= 3).then_some((ch, count))
}

/// A closing fence line holds only the delimiter run plus trailing whitespace (pulldown-cmark
/// rule).
fn fence_has_only_trailing_whitespace(line: &str) -> bool {
    let trimmed = line.trim_start();
    let Some(&ch) = trimmed.as_bytes().first() else {
        return false;
    };
    let run = trimmed.bytes().take_while(|&b| b == ch).count();
    trimmed[run..].chars().all(char::is_whitespace)
}

/// ATX heading line → rendered font size, scaled from the measurement's `font_size`. `None`
/// for non-heading lines.
///
/// CommonMark allows up to three spaces of indentation, stripped before the `#` scan so `'  #
/// heading'` measures at heading size — body size would under-count a long indented heading and
/// could leave an over-budget message expanded.
fn heading_size(line: &str, font_size: f32) -> Option<f32> {
    let trimmed = line.trim_start_matches(' ');
    // CommonMark: at most three leading spaces; a tab (4 columns) exceeds that and never
    // starts an ATX heading.
    if line.len() - trimmed.len() > 3 {
        return None;
    }
    let bytes = trimmed.as_bytes();
    let hashes = bytes.iter().take_while(|&&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    (bytes.get(hashes) == Some(&b' ')).then(|| heading_sizes(font_size)[hashes - 1])
}

/// Setext heading line → rendered font size: a paragraph line immediately followed by a line of
/// `=` (H1) or `-` (H2). The underline is consumed by the parser, so the title is measured at
/// heading size — body size would under-count a long title. The underline itself still counts
/// as one flat source line, which over-counts in the safe direction; a `-`/`=` line after a
/// blank line is a thematic break, and the immediate-next-line check handles that.
///
/// Up to three leading spaces are allowed on the underline (a tab never starts one) and are
/// stripped before the scan, symmetric with [`heading_size`].
fn setext_heading_size(current: &str, next: Option<&str>, font_size: f32) -> Option<f32> {
    if current.is_empty() {
        return None;
    }
    let underline = next?.trim_end();
    if underline.is_empty() {
        return None;
    }
    let body = underline.trim_start_matches(' ');
    if underline.len() - body.len() > 3 {
        return None;
    }
    let ch = *body.as_bytes().first()?;
    if ch != b'=' && ch != b'-' {
        return None;
    }
    // pulldown-cmark maps `=` to H1 and `-` to H2; `heading_sizes` is ordered H1..H6.
    body.bytes()
        .all(|b| b == ch)
        .then(|| heading_sizes(font_size)[usize::from(ch == b'-')])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::UnwrapPoison;
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use std::sync::OnceLock;

    /// Load JetBrains Mono into the global font system so wrap counts are deterministic.
    fn ensure_fonts_loaded() {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            let mut guard = iced::advanced::graphics::text::font_system()
                .write()
                .unwrap_poison();
            guard.load_font(std::borrow::Cow::Borrowed(include_bytes!(
                "../JetBrainsMono-Regular.ttf"
            )));
            guard.load_font(std::borrow::Cow::Borrowed(include_bytes!(
                "../JetBrainsMono-Italic.ttf"
            )));
        });
    }

    /// Test width: 500px → ~59 body chars per wrapped line at 13px JetBrains Mono.
    const TEST_WIDTH: f32 = 500.0;

    fn measure(text: &str) -> MessageMeasure {
        ensure_fonts_loaded();
        measure_message(
            text,
            BodyText::Markdown(None),
            TEST_WIDTH,
            MARKDOWN_TEXT_SIZE,
        )
    }

    fn assert_preview_lines(preview: &str, expected: usize) {
        assert_eq!(
            preview.split('\n').count(),
            expected,
            "preview: {preview:?}"
        );
    }

    enum Preview {
        Expanded,
        Shown,
        Prefix,
        Text(&'static str),
    }

    fn run_measure_cases(cases: &[(&str, String, Option<usize>, Preview)]) {
        for (name, text, lines, preview) in cases {
            let m = measure(text);
            match lines {
                Some(expected) => assert_eq!(
                    m.wrapped_lines,
                    u32::try_from(*expected).expect("preview sizes fit u32"),
                    "case {name}: wrapped lines"
                ),
                None => assert!(
                    m.wrapped_lines > MAX_PREVIEW_LINES,
                    "case {name}: {} lines, expected past the budget",
                    m.wrapped_lines
                ),
            }
            match preview {
                Preview::Expanded => {
                    assert!(m.preview.is_none(), "case {name}: must stay expanded");
                }
                Preview::Shown => assert_preview_lines(
                    &m.preview.expect("the message must collapse"),
                    MAX_PREVIEW_LINES as usize,
                ),
                Preview::Prefix => {
                    let preview = m.preview.as_deref().expect("a prefix case must collapse");
                    assert_preview_lines(preview, MAX_PREVIEW_LINES as usize);
                    let flat_preview: String =
                        preview.chars().filter(|c| !c.is_whitespace()).collect();
                    let flat_text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
                    assert!(
                        flat_text.starts_with(&flat_preview),
                        "case {name}: the preview must be the start of the wrapped text: {preview:?}"
                    );
                }
                Preview::Text(expected) => {
                    let preview = m.preview.expect("the message must collapse");
                    assert_eq!(preview, *expected, "case {name}: preview");
                    assert_preview_lines(&preview, MAX_PREVIEW_LINES as usize);
                }
            }
        }
    }

    /// Collapsing is the bias: past the budget a message collapses to its first lines, within it
    /// stays whole.
    #[test]
    fn wrapping_and_collapsing_cases() {
        run_measure_cases(&[
            (
                "short message stays expanded",
                "hello".to_string(),
                Some(1),
                Preview::Expanded,
            ),
            (
                "exactly three short lines stay expanded",
                "one\ntwo\nthree".to_string(),
                Some(3),
                Preview::Expanded,
            ),
            (
                "four short lines collapse",
                "one\ntwo\nthree\nfour".to_string(),
                Some(4),
                Preview::Text("one\ntwo\nthree"),
            ),
            (
                "long prose paragraph wraps and collapses",
                "The quick brown fox jumps over the lazy dog. ".repeat(4),
                None,
                Preview::Prefix,
            ),
            (
                "blank line between lines",
                "a\n\nb\nc".to_string(),
                None,
                Preview::Text("a\n\nb"),
            ),
            (
                "blank lines with more after them",
                "a\n\nb\nc\nd".to_string(),
                None,
                Preview::Text("a\n\nb"),
            ),
            (
                "leading blank lines",
                "\n\na\n\nb".to_string(),
                None,
                Preview::Text("\n\na"),
            ),
            (
                "trailing blank line",
                "a\n\nb\n".to_string(),
                None,
                Preview::Text("a\n\nb"),
            ),
            (
                "code fence lines count as source lines",
                "```\nline1\nline2\n```".to_string(),
                None,
                Preview::Text("```\nline1\nline2"),
            ),
            (
                "short code fence stays expanded",
                "```\nline1\n```".to_string(),
                Some(3),
                Preview::Expanded,
            ),
            (
                "long code line does not wrap",
                format!("```\n{}\n```", "x".repeat(300)),
                Some(3),
                Preview::Expanded,
            ),
        ]);

        // An extracted picture contributes one label line, never its base64 payload.
        let png = {
            let img = image::RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 0, 255]));
            let mut buf = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .expect("test PNG must encode");
            buf
        };
        let uri = format!("data:image/png;base64,{}", STANDARD.encode(&png));
        let text = format!("before [IMAGE:{uri}] after");
        let m = measure(&text);
        assert!(
            m.wrapped_lines <= MAX_PREVIEW_LINES,
            "{} lines",
            m.wrapped_lines
        );
        assert!(m.preview.is_none());
        let measured = media_markers::preprocess(&text, None).display_text();
        assert!(!measured.contains("base64"), "got: {measured:?}");
        assert_eq!(measured, "🖼️ image\nbefore after");
    }

    /// Headings measure at their own size: a flat body-size count under-counts a long heading.
    #[test]
    fn heading_measurement_cases() {
        run_measure_cases(&[
            (
                "long ATX heading wraps past the budget",
                format!("# {}", "heading ".repeat(15)),
                None,
                Preview::Shown,
            ),
            (
                "long indented ATX heading",
                format!("  # {}", "heading ".repeat(15)),
                None,
                Preview::Shown,
            ),
            (
                "4+ spaces of indent is not a heading",
                format!("    # {}", "h".repeat(5)),
                Some(1),
                Preview::Expanded,
            ),
            (
                "body-sized H5 stays short",
                format!("##### {}", "h".repeat(20)),
                Some(1),
                Preview::Expanded,
            ),
            (
                "long setext title measured at heading size",
                format!("{}\n===", "setext heading title ".repeat(5)),
                None,
                Preview::Shown,
            ),
            (
                "short setext title stays expanded",
                "short title\n===".to_string(),
                Some(2),
                Preview::Expanded,
            ),
            (
                "setext title boundary pins H1 vs H2",
                format!("{}\n===", "setext title ".repeat(5)),
                None,
                Preview::Shown,
            ),
        ]);

        let sizes = heading_sizes(MARKDOWN_TEXT_SIZE);
        let mapping: &[(&str, Option<&str>, Option<f32>)] = &[
            ("= is H1", Some("==="), Some(sizes[0])),
            ("- is H2", Some("---"), Some(sizes[1])),
            ("up to three leading spaces", Some("  ==="), Some(sizes[0])),
            ("4+ leading spaces is no underline", Some("    ==="), None),
            ("trailing whitespace is fine", Some("---  "), Some(sizes[1])),
            ("mixed characters are no underline", Some("=-="), None),
            ("a blank underline is none", Some(""), None),
        ];
        for &(name, underline, expected) in mapping {
            assert_eq!(
                setext_heading_size("title", underline, MARKDOWN_TEXT_SIZE),
                expected,
                "case {name}: heading size"
            );
        }
        assert_eq!(
            setext_heading_size("", Some("==="), MARKDOWN_TEXT_SIZE),
            None,
            "case blank title: none"
        );
    }

    #[test]
    fn mixed_direction_wrap_never_panics() {
        // BLOCKER regression: a reversed RTL run left alone on a wrapped line is stored in
        // visual (L2-reordered) order with decreasing byte offsets, and the old
        // `first().start..last().end` slice panicked ("byte range starts at X but ends at Y")
        // during layout inside the responsive closure, aborting the whole GUI. Scanning a range
        // of prefix lengths lands at least one wrap there (the exact boundary depends on
        // metrics): the measurement must not panic.
        for n in 40..75 {
            let text = format!("{} שלום עולם", "x".repeat(n));
            let _ = measure(&text);
        }
    }

    #[test]
    fn mixed_direction_collapsing_message_preview_keeps_logical_order() {
        // Truncation regression: slicing visual-order glyphs dropped the first RTL letters.
        for n in 45..70 {
            let text = format!("{} שלום עולם {}", "x".repeat(n), "pad ".repeat(50));
            let m = measure(&text);
            assert!(m.wrapped_lines > 3, "n={n}: {} lines", m.wrapped_lines);
            let preview = m.preview.expect("n={n}: long message must collapse");
            let flat: String = preview.chars().filter(|c| !c.is_whitespace()).collect();
            assert!(
                flat.contains("שלוםעולם"),
                "n={n}: preview must keep the RTL phrase in logical order: {preview:?}"
            );
        }
    }
}
