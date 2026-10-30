//! Session transcript message collapse measurement.
//!
//! The sessions page collapses messages that render more than
//! [`MAX_PREVIEW_LINES`] wrapped lines into a short plain-text preview with
//! an expand/collapse toggle. This module implements the line-count
//! measurement: a flat wrapped-line count of the message's display text at body
//! metrics and the actual transcript content width.
//!
//! The measurement intentionally errs toward collapsing:
//! - a safety margin narrows the measured width so markdown structure the
//!   flat model cannot see (list/quote indentation) forces an extra wrap
//!   instead of slipping a 4-line message through;
//! - heading lines (ATX and setext) are measured at their rendered heading
//!   size;
//! - fenced code-block lines count as one source line each (the renderer
//!   places them unwrapped in a horizontally scrollable block); fence
//!   delimiter lines count too — a 2-line code block with fences measures 4
//!   and collapses, a deliberate over-collapse in the collapse direction;
//! - a markdown body has its media markers and pictures extracted into their
//!   attachments before measuring, and their labels measure in the place the
//!   expanded body draws them (above the text), so a base64 data-URI blob
//!   cannot skew the count; a body drawn as raw text measures that text
//!   unchanged.
//!
//! Preview extraction is BiDi-safe: cosmic-text lays glyphs out in visual
//! order (L2 reordering), so an RTL run can appear with decreasing byte
//! offsets. The preview segment of each wrapped line is therefore taken as
//! the logical span (min start .. max end) of the line's glyphs — correct in
//! every direction case, never a `start > end` slice.
//!
//! Shaping is chunked per source line with early exit once the line budget
//! is exceeded, so a huge message (system prompts reach hundreds of KB)
//! costs only a few shaped lines. A single source line longer than the
//! worst-case budget is shaped as a bounded prefix; when the prefix proves
//! the collapse its first lines come from true wrap points, and when it
//! cannot (a giant unbreakable word overflowing one line) the count
//! saturates past the budget — err toward collapsing. Every preview line is
//! capped to the characters that fit the transcript width in the body font
//! (fence/code lines and overflowing unbreakable words included), so the
//! 3-line preview renders as at most 3 visual lines (an unbreakable word
//! yields a single overflowing line). The result is cached per message and
//! width by [`crate::gui::sessions::SessionsState`].

use std::path::Path;
use std::sync::Arc;

use cosmic_text::{Attrs, AttrsList, BufferLine, Family, LineEnding, Shaping, Wrap};
use iced::advanced::graphics::text::cosmic_text;

use crate::gui::media_markers;
use crate::gui::text_rendering::with_font_system;
#[cfg(test)]
use crate::gui::theme::MARKDOWN_TEXT_SIZE;

/// Line budget: any message rendering more than this many wrapped lines is
/// collapsed to a [`MAX_PREVIEW_LINES`]-line preview by default.
pub(crate) const MAX_PREVIEW_LINES: u32 = 3;

/// Safety margin (px) subtracted from the available width before measuring.
///
/// The flat measurement cannot see markdown list/quote indentation, so lines
/// sitting close to the width boundary wrap in the real render but not in
/// the measurement. Narrowing the measured width forces those lines to wrap
/// one line earlier — the design constraint is to err toward collapsing (never
/// leave a message that renders longer than the budget expanded).
pub(crate) const WRAP_SAFETY_MARGIN_PX: f32 = 36.0;

/// Maximum characters per preview line segment from a laid line.
///
/// Laid lines that fit the transcript width are already bounded by it (~59
/// mono chars at the body size), and even a pathological overflowing line
/// (a giant unbreakable word) renders as a single non-wrapping line in the
/// preview — so this is a hard safety bound only. The width-aware
/// [`preview_line_char_budget`] cap applies separately to unwrapped
/// fence/code lines (which wrap at spaces in the plain-text preview and
/// would otherwise spill past the 3-line budget visually).
const MAX_PREVIEW_LINE_CHARS: usize = 600;

/// Markdown heading sizes relative to the measurement body size, matching
/// iced 0.14's `markdown::Settings` scale exactly (h1 = 2.0×, h2 = 1.75×,
/// h3 = 1.5×, h4 = 1.25×, h5/h6 = body). Derived from the caller's `font_size`
/// so heading wrap stays in lockstep with the renderer's scaled headings
/// (narration bodies scale from the larger narration size).
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

/// Upper bound (bytes) on the shaped prefix of a single source line.
///
/// Each laid line holds at most `avail` characters (the absolute worst case
/// is 1 px/char) and a UTF-8 character is at most 4 bytes, so
/// `need * avail * 4` bytes of content cannot fit in fewer than `need` laid
/// lines even before real glyph widths are considered — a line longer than
/// this ceiling definitely wraps past the remaining budget, so shaping a
/// bounded prefix is enough to prove the collapse. The +4 covers the UTF-8
/// char-boundary floor of the prefix cut.
fn shape_ceiling(avail: f32, need: u32) -> usize {
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let per_line = (avail as usize).max(1) * 4;
    per_line.saturating_mul(need as usize).saturating_add(4)
}

/// Maximum characters per preview line.
///
/// Preview lines render in the same JetBrains Mono body font as the
/// measurement, so a preview line longer than `avail / advance` would wrap
/// in the preview — for unwrapped code/fence lines that would make the
/// "3-line preview" render as many visual lines. The cap is measured from
/// the body font's glyph advance (mono: uniform), not hardcoded, so it
/// tracks the real rendered width. The truncate ellipsis counts inside the cap,
/// so a capped line (content + ellipsis) never exceeds the transcript width and
/// cannot wrap to a 4th visual line.
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

/// Per-message measurement result, cached in [`SessionsState`] keyed by
/// (message index, width bucket).
///
/// [`SessionsState`]: crate::gui::sessions::SessionsState
pub(crate) struct MessageMeasure {
    /// Width bucket the measurement was computed for (see [`width_bucket`]).
    pub(crate) width_bucket: u32,
    /// Total wrapped line count of the display text.
    pub(crate) wrapped_lines: u32,
    /// First [`MAX_PREVIEW_LINES`] wrapped lines as plain text — `Some` only
    /// when the message exceeds the budget and therefore collapses.
    pub(crate) preview: Option<String>,
    /// The display text this measurement was derived from, kept so a width
    /// change can re-measure without re-decoding the message.
    pub(crate) source: Arc<str>,
    /// Media-placeholder-reduced display text the measurement was actually
    /// computed from. Kept so a width-bucket change re-measures via
    /// [`re_measure`] without re-running media preprocessing (two full
    /// String copies + regex passes per message) on every window resize. A
    /// [`BodyText::Plain`] body's processed text is its source, unchanged.
    pub(crate) processed: Arc<str>,
    /// Body font size (px) the measurement was computed at. A width-bucket
    /// re-measure via [`re_measure`] carries it forward, so a narration body
    /// measured at the narration size keeps that size on a resize.
    pub(crate) font_size: f32,
}

/// Quantize a text width into a bucket so sub-pixel layout jitter does not
/// invalidate the per-message measurement cache; a real window resize moves
/// the bucket and triggers re-measurement.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn width_bucket(width: f32) -> u32 {
    (width.max(0.0) / 16.0) as u32
}

/// How a collapsible body draws its text, so the measurement matches what is
/// drawn: a markdown body has its markers extracted into attachment labels,
/// while a plain one (a tool result, a thinking block, promoted reasoning) shows
/// the raw text.
#[derive(Clone, Copy)]
pub(crate) enum BodyText<'a> {
    /// A bubble body parsed as markdown; relative markers resolve against the
    /// workspace.
    Markdown(Option<&'a Path>),
    /// A body drawn as raw selectable text.
    Plain,
}

/// Measure the wrapped line count of a message body at `width`, producing a
/// [`MAX_PREVIEW_LINES`]-line plain-text preview when the message exceeds
/// the budget.
///
/// `display_text` is the raw message body (the same text that the Sessions
/// ledger's `parse_entry_bodies` feeds to `markdown::parse`); [`BodyText`] says
/// how it is drawn, so the measured text is what the element shows — a markdown
/// body measures [`media_markers::Preprocessed::display_text`] (the attachment
/// labels above the text, the order the expanded body draws, so a base64 payload
/// counts as one short label instead of a blob of characters), a plain one
/// measures its own raw text. The source is owned as [`Arc<str>`] so a
/// width-bucket change can re-measure without copying the (potentially hundreds
/// of KB) display text again. `font_size` is the body size the text renders at
/// (body or narration), used for wrap and preview-line-budget shaping and
/// heading scaling.
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

/// Re-measure a cached entry at a new width, reusing the cached processed
/// display text so a width-bucket change does not re-run media
/// preprocessing (two full String copies + regex passes per message). The
/// cached [`MessageMeasure::font_size`] is carried forward.
pub(crate) fn re_measure(cached: &MessageMeasure, width: f32) -> MessageMeasure {
    measure_processed(
        cached.source.clone(),
        cached.processed.clone(),
        width,
        cached.font_size,
    )
}

/// Shared core of [`measure_message`] / [`re_measure`]: layout the already
/// media-processed display text at `width` and count wrapped lines.
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

    // Family-only attrs: italic shares the mono glyph advance, so a style
    // change is unnecessary for measurement.
    let attrs = Attrs::new().family(Family::Name("JetBrains Mono"));
    let mut total_lines = 0u32;
    let mut preview = String::new();
    let mut preview_lines = 0u32;
    // Current fenced code block: (delimiter char, minimum closing length).
    let mut fence: Option<(u8, usize)> = None;

    with_font_system(|font_sys| {
        // Preview lines render in the same JetBrains Mono body font, so a
        // preview line longer than `avail / advance` characters would wrap
        // in the preview (or, for unwrapped code lines, spill past the
        // 3-line budget visually). Measured from the body font's glyph
        // advance rather than hardcoded.
        let preview_line_chars = preview_line_char_budget(font_sys, &attrs, avail, font_size);
        let mut line = BufferLine::new(
            "",
            LineEnding::None,
            AttrsList::new(&attrs),
            Shaping::Advanced,
        );

        // Peekable so a setext heading's size can look at the underline on
        // the immediately following line.
        let mut lines = text.split('\n').peekable();
        while let Some(src_line) = lines.next() {
            let next_line = lines.peek().copied();
            // Early exit: once the budget is exceeded we already know the
            // message collapses; the preview is complete, so stop shaping.
            if total_lines > MAX_PREVIEW_LINES {
                break;
            }

            match fence {
                Some((open_ch, open_count)) => {
                    // Inside a fenced code block: every source line counts
                    // once (the renderer places code lines unwrapped in a
                    // horizontally scrollable block). A matching delimiter
                    // line closes the block.
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
                        // Wrappable line: shape it and lay it out at its
                        // rendered font size (headings render larger than
                        // the body, so they are measured at their own size).
                        let size = heading_size(src_line, font_size)
                            .or_else(|| setext_heading_size(src_line, next_line, font_size))
                            .unwrap_or(font_size);
                        // Bound the shaped text: a source line longer than
                        // `shape_ceiling(avail, need)` bytes holds more than
                        // `need * avail` characters (UTF-8 ≤ 4 bytes/char),
                        // and even at the absolute worst 1 px/char each laid
                        // line fits at most `avail` chars — such a line
                        // definitely wraps past the remaining budget, so the
                        // collapse is proven by shaping a bounded prefix
                        // (when the prefix cannot prove it — a giant
                        // unbreakable word — the count below saturates past
                        // the budget instead). Shaping a 200KB single-line
                        // tool result in full on the UI thread would jank
                        // the transcript.
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
                        // When the prefix alone does not prove the collapse
                        // (fewer than `need` lines: the line starts with a
                        // giant unbreakable word overflowing a single line),
                        // the rest of the line can only add wrapped lines —
                        // err toward collapsing. A single source line cannot
                        // produce more wrapped lines than `u32::MAX` in
                        // memory; saturate for the budget check.
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
                            // Glyphs are laid out in visual (BiDi-reordered)
                            // order: inside an RTL run byte offsets decrease,
                            // so a wrapped line can begin after it ends —
                            // `first().start` may exceed `last().end`, which
                            // panicked the old first..last slice. Take the
                            // logical span of the line's glyphs (min start ..
                            // max end): it is the correct text of this wrapped
                            // line in every direction case.
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
                                // A fitting laid line is already bounded by the
                                // transcript width; only a line overflowing it
                                // (w > avail) needs an extra cap. That cap is
                                // applied to genuinely unbreakable words (e.g.
                                // a minified JSON blob — no whitespace), which
                                // would otherwise render as one overflowing
                                // line far past the width; lines with break
                                // opportunities wrap in the preview at their
                                // spaces, and capping them would truncate
                                // mid-word an RTL phrase that straddles the
                                // wrap (the BiDi guard: every Hebrew char in
                                // logical order).
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

/// Append one source line to the running preview, counting it as a single
/// rendered line (fence and in-fence code lines). `preview_line_chars` caps
/// each preview line to what fits the transcript width in the body font —
/// code lines render unwrapped in the transcript but are plain wrapping
/// text in the preview, so an uncapped line would break the 3-line preview.
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

/// Detect a fenced code-block delimiter line (```` ``` ```` or `~~~`, at
/// least 3 chars, optionally indented). Returns the delimiter char and run
/// length.
fn fence_delimiter(line: &str) -> Option<(u8, usize)> {
    let trimmed = line.trim_start();
    let ch = *trimmed.as_bytes().first()?;
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let count = trimmed.bytes().take_while(|&b| b == ch).count();
    (count >= 3).then_some((ch, count))
}

/// A closing fence line may contain only the delimiter run plus trailing
/// whitespace (pulldown-cmark rule).
fn fence_has_only_trailing_whitespace(line: &str) -> bool {
    let trimmed = line.trim_start();
    let Some(&ch) = trimmed.as_bytes().first() else {
        return false;
    };
    let run = trimmed.bytes().take_while(|&b| b == ch).count();
    trimmed[run..].chars().all(char::is_whitespace)
}

/// ATX heading line → rendered font size (iced `markdown::Settings` sizes),
/// scaled from the measurement's `font_size`. Returns `None` for non-heading
/// lines.
///
/// CommonMark allows up to three spaces of indentation before an ATX
/// heading; the indent is stripped before the `#` scan so `'  # heading'`
/// (a valid heading) is measured at heading size. Measuring it at body size
/// would under-count a long indented heading and could leave an over-budget
/// message expanded — the one direction the collapse bias forbids.
fn heading_size(line: &str, font_size: f32) -> Option<f32> {
    let trimmed = line.trim_start_matches(' ');
    // CommonMark: at most three leading spaces; a tab (4 columns) already
    // exceeds that and never starts an ATX heading.
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

/// Setext heading line → rendered font size: a paragraph line immediately
/// followed by a line of `=` (H1) or `-` (H2) characters. The underline is
/// consumed by the parser (not a rendered line of its own), so the title is
/// measured at its rendered heading size — measuring it at body size would
/// under-count a long title and could leave an over-budget message expanded,
/// the one direction the collapse bias forbids. The underline itself still
/// counts as one flat source line, which over-counts and errs toward
/// collapsing (the safe direction); a `-`/`=` line after a blank line is a
/// thematic break / text, not an underline, and the immediate-next-line
/// check handles that (an empty `current` can never be a heading).
///
/// CommonMark allows up to three leading spaces on the underline (a tab is
/// four columns and never starts one); they are stripped before the scan,
/// symmetric with the ATX rule in [`heading_size`], so `'title\n  ==='` is a
/// valid heading and must measure at heading size — body-size measurement
/// would under-collapse a long indented title.
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
    // CommonMark (and pulldown-cmark's scan_setext_heading) maps `=` to H1
    // and `-` to H2; `heading_sizes` is ordered H1..H6.
    body.bytes()
        .all(|b| b == ch)
        .then(|| heading_sizes(font_size)[usize::from(ch == b'-')])
}

// ── Tests ─────────────────────────────────────────────────────────

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

    /// Collapsing is the bias: past the budget a message collapses to its first lines, within it stays whole.
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

    /// Headings are measured at their own size: a flat body-size count under-counts a long heading.
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

    /// Huge inputs: the measurement stops early and the preview stays bounded.
    #[test]
    fn bounded_preview_cases() {
        #[derive(Default)]
        struct Case {
            name: &'static str,
            text: String,
            preview_lines: Option<usize>,
            first_preview_line: Option<&'static str>,
            second_line_max: Option<usize>,
            max_preview_chars: Option<usize>,
            is_prefix: bool,
        }

        let cases: &[Case] = &[
            Case {
                name: "huge two-line message",
                text: format!("{}\n{}", "p ".repeat(50_000), "q ".repeat(50_000)),
                preview_lines: Some(MAX_PREVIEW_LINES as usize),
                max_preview_chars: Some(2_000),
                ..Case::default()
            },
            Case {
                name: "huge single-line message",
                text: format!("{} end", "word ".repeat(20_000)),
                preview_lines: Some(MAX_PREVIEW_LINES as usize),
                is_prefix: true,
                ..Case::default()
            },
            Case {
                name: "unbreakable word",
                text: format!("{} tail", "x".repeat(200_000)),
                ..Case::default()
            },
            Case {
                name: "code block",
                text: format!("```\n{}\n```\nafter", "code ".repeat(200)),
                preview_lines: Some(MAX_PREVIEW_LINES as usize),
                first_preview_line: Some("```"),
                second_line_max: Some(64),
                ..Case::default()
            },
        ];

        for case in cases {
            let name = case.name;
            let m = measure(&case.text);
            assert!(
                m.wrapped_lines > MAX_PREVIEW_LINES,
                "case {name}: {} lines, expected past the budget",
                m.wrapped_lines
            );
            let Some(preview) = m.preview.as_deref() else {
                panic!("case {name}: the message must collapse");
            };
            if let Some(lines) = case.preview_lines {
                assert_eq!(
                    preview.split('\n').count(),
                    lines,
                    "case {name}: {preview:?}"
                );
            }
            if let Some(first) = case.first_preview_line {
                assert_eq!(
                    preview.lines().next(),
                    Some(first),
                    "case {name}: first line"
                );
            }
            if let Some(max) = case.second_line_max {
                let second = preview
                    .lines()
                    .nth(1)
                    .expect("second preview line is the code line");
                assert!(
                    second.chars().count() <= max,
                    "case {name}: second line too long: {} chars",
                    second.chars().count()
                );
            }
            if let Some(max) = case.max_preview_chars {
                assert!(
                    preview.chars().count() < max,
                    "case {name}: preview too large: {} chars",
                    preview.chars().count()
                );
            }
            if case.is_prefix {
                let flat_preview: String = preview.chars().filter(|c| !c.is_whitespace()).collect();
                let flat_text: String = case.text.chars().filter(|c| !c.is_whitespace()).collect();
                assert!(
                    flat_text.starts_with(&flat_preview),
                    "case {name}: preview must be the start of the wrapped line: {preview:?}"
                );
            }
        }
    }

    #[test]
    fn width_bucket_quantizes_for_cache_invalidation() {
        // 16px buckets: sub-pixel jitter within a bucket keeps the cache
        // valid; a real resize moves the bucket and triggers re-measurement.
        assert_eq!(width_bucket(0.0), 0);
        assert_eq!(width_bucket(15.9), 0);
        assert_eq!(width_bucket(16.0), 1);
        assert_eq!(width_bucket(500.0), 31);
        assert_eq!(width_bucket(508.0), 31);
        assert_eq!(width_bucket(540.0), 33);
    }

    #[test]
    fn mixed_direction_wrap_never_panics() {
        // BLOCKER regression: when the wrap leaves a reversed RTL run alone
        // on a wrapped line, cosmic-text stores its glyphs in visual
        // (L2-reordered) order with decreasing byte offsets, and the old
        // `first().start..last().end` preview slice panicked ("byte range
        // starts at X but ends at Y") during layout inside the responsive
        // closure, aborting the whole GUI. Scan a range of prefix lengths so
        // at least one wrap lands on that position (the exact boundary
        // depends on font metrics) — the measurement must not panic.
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
