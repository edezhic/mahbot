//! Session transcript message collapse measurement.
//!
//! The sessions page collapses messages that render more than
//! [`MAX_PREVIEW_LINES`] wrapped lines into a short plain-text preview with
//! an expand/collapse toggle. This module implements the line-count
//! measurement: a flat wrapped-line count of the message's display text
//! (after media-marker processing) at body metrics and the actual transcript
//! content width.
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
//! - media markers are reduced to short text placeholders before measuring,
//!   so base64 data-URI blobs cannot skew the count.
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

use std::sync::Arc;
use std::sync::LazyLock;

use cosmic_text::{Attrs, AttrsList, BufferLine, Family, LineEnding, Shaping, Wrap};
use iced::advanced::graphics::text::cosmic_text;

use super::media_markers;
use super::text_rendering::with_font_system;
use super::theme::MARKDOWN_TEXT_SIZE;
use crate::util::file_name_or_path;

/// Body text size of the transcript markdown — derived from
/// [`theme::MARKDOWN_TEXT_SIZE`], the single source of truth shared with
/// the actual renderer (`theme::markdown_settings`), so a theme font-size
/// change cannot silently drift the measured wrap count from the render.
pub(crate) const BODY_FONT_SIZE: f32 = MARKDOWN_TEXT_SIZE;

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

/// Markdown heading sizes relative to the body size, matching iced 0.14's
/// `markdown::Settings` scale exactly (h1 = 2.0×, h2 = 1.75×, h3 = 1.5×,
/// h4 = 1.25×, h5/h6 = body). Derived from [`MARKDOWN_TEXT_SIZE`] so the
/// measurement stays in lockstep with the renderer.
const HEADING_SIZES: [f32; 6] = [
    MARKDOWN_TEXT_SIZE * 2.0,
    MARKDOWN_TEXT_SIZE * 1.75,
    MARKDOWN_TEXT_SIZE * 1.5,
    MARKDOWN_TEXT_SIZE * 1.25,
    BODY_FONT_SIZE,
    BODY_FONT_SIZE,
];

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
/// tracks the real rendered width. One character is reserved for the
/// truncate ellipsis ("…"), so a capped line (content + ellipsis) never
/// exceeds the transcript width and cannot wrap to a 4th visual line.
fn preview_line_char_budget(
    font_sys: &mut cosmic_text::FontSystem,
    attrs: &Attrs,
    avail: f32,
) -> usize {
    let mut probe = BufferLine::new(
        "M",
        LineEnding::None,
        AttrsList::new(attrs),
        Shaping::Advanced,
    );
    let advance = probe
        .shape(font_sys, TAB_WIDTH)
        .layout(BODY_FONT_SIZE, Some(1_000.0), Wrap::None, None, None)
        .first()
        .map_or(0.0, |l| l.w);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let budget = ((avail / advance.max(1.0)) as usize).saturating_sub(1);
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
    /// String copies + regex passes per message) on every window resize.
    pub(crate) processed: Arc<str>,
    /// Length of the raw message content this measurement was derived from —
    /// a cheap content fingerprint. The index-keyed cache is stale when the
    /// underlying message is replaced by session compaction with content of a
    /// different length, so the entry is re-measured instead of showing a
    /// stale preview until the session is reopened.
    pub(crate) content_len: usize,
}

/// Quantize a text width into a bucket so sub-pixel layout jitter does not
/// invalidate the per-message measurement cache; a real window resize moves
/// the bucket and triggers re-measurement.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn width_bucket(width: f32) -> u32 {
    (width.max(0.0) / 16.0) as u32
}

/// Measure the wrapped line count of a message body at `width`, producing a
/// [`MAX_PREVIEW_LINES`]-line plain-text preview when the message exceeds
/// the budget.
///
/// `display_text` is the raw message body (the same text that
/// [`parse_messages_to_md_items`] feeds to `markdown::parse`); media-marker
/// preprocessing and image placeholder reduction happen here. The source is
/// owned as [`Arc<str>`] so a width-bucket change can re-measure without
/// copying the (potentially hundreds of KB) display text again.
///
/// [`parse_messages_to_md_items`]: crate::gui::sessions::parse_messages_to_md_items
pub(crate) fn measure_message(
    display_text: impl Into<Arc<str>>,
    width: f32,
    content_len: usize,
) -> MessageMeasure {
    let source = display_text.into();
    let processed: Arc<str> = placeholder_media(&media_markers::preprocess(&source)).into();
    measure_processed(source, processed, width, content_len)
}

/// Re-measure a cached entry at a new width, reusing the cached processed
/// display text so a width-bucket change does not re-run media
/// preprocessing (two full String copies + regex passes per message).
pub(crate) fn re_measure(cached: &MessageMeasure, width: f32) -> MessageMeasure {
    measure_processed(
        cached.source.clone(),
        cached.processed.clone(),
        width,
        cached.content_len,
    )
}

/// Shared core of [`measure_message`] / [`re_measure`]: layout the already
/// media-processed display text at `width` and count wrapped lines.
#[expect(clippy::too_many_lines)]
fn measure_processed(
    source: Arc<str>,
    processed: Arc<str>,
    width: f32,
    content_len: usize,
) -> MessageMeasure {
    let bucket = width_bucket(width);
    let text = &processed;
    let avail = (width - WRAP_SAFETY_MARGIN_PX).max(1.0);

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
        let preview_line_chars = preview_line_char_budget(font_sys, &attrs, avail);
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
                        let size = heading_size(src_line)
                            .or_else(|| setext_heading_size(src_line, next_line))
                            .unwrap_or(BODY_FONT_SIZE);
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
        content_len,
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

/// ATX heading line → rendered font size (iced `markdown::Settings` sizes).
/// Returns `None` for non-heading lines.
///
/// CommonMark allows up to three spaces of indentation before an ATX
/// heading; the indent is stripped before the `#` scan so `'  # heading'`
/// (a valid heading) is measured at heading size. Measuring it at body size
/// would under-count a long indented heading and could leave an over-budget
/// message expanded — the one direction the collapse bias forbids.
fn heading_size(line: &str) -> Option<f32> {
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
    (bytes.get(hashes) == Some(&b' ')).then(|| HEADING_SIZES[hashes - 1])
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
fn setext_heading_size(current: &str, next: Option<&str>) -> Option<f32> {
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
    // and `-` to H2; HEADING_SIZES is ordered H1..H6.
    body.bytes()
        .all(|b| b == ch)
        .then(|| HEADING_SIZES[usize::from(ch == b'-')])
}

/// Replace markdown image syntax (`![alt](url)`) with a short text
/// placeholder (`🖼️ filename`), so images — and especially base64 data-URI
/// blobs — count as a bounded token rather than raw text. Audio/video
/// markers are already short text after [`media_markers::preprocess`] and
/// pass through untouched.
fn placeholder_media(processed: &str) -> String {
    IMAGE_PLACEHOLDER_RE
        .replace_all(processed, |caps: &regex::Captures| {
            let url = caps.get(1).map_or("", |m| m.as_str());
            let label = if url.starts_with("data:") {
                // Base64 data-URI images have no filename — generic label.
                "image"
            } else {
                file_name_or_path(url)
            };
            format!("🖼️ {label}")
        })
        .into_owned()
}

/// Matches markdown image syntax as produced by `media_markers::preprocess`
/// (`![Image](path)`) and raw `![alt](url)` in agent text. The url group
/// stops at the first `)`, which is also where pulldown-cmark closes the
/// destination.
static IMAGE_PLACEHOLDER_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"!\[[^\]]*\]\(([^)]*)\)").expect("image placeholder regex must compile")
});

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::UnwrapPoison;
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use std::sync::OnceLock;

    /// Ensure JetBrains Mono is loaded into the global font system so wrap
    /// counts are deterministic (the app loads it at startup via
    /// `iced::application::font`, which tests do not run). `OnceLock`
    /// guarantees a single load; the font-system RwLock serializes access.
    fn ensure_fonts_loaded() {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            let mut guard = iced::advanced::graphics::text::font_system()
                .write()
                .unwrap_poison();
            guard.load_font(std::borrow::Cow::Borrowed(include_bytes!(
                "JetBrainsMono-Regular.ttf"
            )));
        });
    }

    /// Test width: 500px → ~464px after the safety margin → ~59 body chars
    /// per wrapped line at 13px JetBrains Mono (0.6em advance = 7.8px/char).
    const TEST_WIDTH: f32 = 500.0;

    fn measure(text: &str) -> MessageMeasure {
        ensure_fonts_loaded();
        measure_message(text, TEST_WIDTH, text.len())
    }

    fn assert_preview_lines(preview: &str, expected: usize) {
        assert_eq!(
            preview.split('\n').count(),
            expected,
            "preview: {preview:?}"
        );
    }

    #[test]
    fn short_message_stays_expanded() {
        let m = measure("hello");
        assert_eq!(m.wrapped_lines, 1);
        assert!(m.preview.is_none());
    }

    #[test]
    fn exactly_three_short_lines_stay_expanded() {
        let m = measure("one\ntwo\nthree");
        assert_eq!(m.wrapped_lines, 3);
        assert!(m.preview.is_none());
    }

    #[test]
    fn four_short_lines_collapse() {
        let m = measure("one\ntwo\nthree\nfour");
        assert_eq!(m.wrapped_lines, 4);
        let preview = m.preview.expect("4-line message must collapse");
        assert_preview_lines(&preview, 3);
        assert_eq!(preview, "one\ntwo\nthree");
    }

    #[test]
    fn long_prose_paragraph_wraps_and_collapses() {
        // ~180 chars of prose — at ~59 chars/line this wraps to 4 lines.
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(4);
        let m = measure(&text);
        assert!(
            m.wrapped_lines > 3,
            "long paragraph must wrap past the budget ({} lines)",
            m.wrapped_lines
        );
        let preview = m.preview.expect("long paragraph must collapse");
        assert_preview_lines(&preview, 3);
        // The preview is the first 3 wrapped lines — flat-whitespace-stripped
        // it must be a prefix of the flat source text (no source newlines
        // introduced beyond the wrap points).
        let flat_preview: String = preview.chars().filter(|c| !c.is_whitespace()).collect();
        let flat_text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            flat_text.starts_with(&flat_preview),
            "preview must be the start of the wrapped text: {preview:?}"
        );
    }

    #[test]
    fn blank_lines_kept_in_preview() {
        // Blank source lines are rendered lines: a collapsing message
        // previews exactly its first 3 lines, blank lines included — not the
        // compacted "a\nb\nc".
        for text in ["a\n\nb\n", "a\n\nb\nc"] {
            let m = measure(text);
            assert!(m.wrapped_lines > 3, "{text:?}: {} lines", m.wrapped_lines);
            let preview = m.preview.expect("4-line message must collapse");
            assert_eq!(preview, "a\n\nb");
            assert_preview_lines(&preview, 3);
        }
        // Leading blanks are kept too: "\n\na\n\nb" (5 lines) previews as the
        // first 3 rendered lines "", "", "a".
        let m = measure("\n\na\n\nb");
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        let preview = m.preview.expect("5-line message must collapse");
        assert_eq!(preview, "\n\na");
        assert_preview_lines(&preview, 3);
    }

    #[test]
    fn code_fence_lines_count_as_source_lines() {
        // 2 fences + 2 code lines = 4 lines → collapses (the count is a
        // lower bound past the budget due to early exit).
        let text = "```\nline1\nline2\n```";
        let m = measure(text);
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        let preview = m.preview.expect("4-line message must collapse");
        assert_preview_lines(&preview, 3);
        assert_eq!(preview, "```\nline1\nline2");
    }

    #[test]
    fn short_code_fence_stays_expanded() {
        // 2 fences + 1 code line = 3 lines, none wrapping → no collapse.
        let text = "```\nline1\n```";
        let m = measure(text);
        assert_eq!(m.wrapped_lines, 3);
        assert!(m.preview.is_none());
    }

    #[test]
    fn long_code_line_does_not_wrap() {
        // A 300-char single code line must count as exactly 1 line (the
        // renderer scrolls code blocks horizontally).
        let text = format!("```\n{}\n```", "x".repeat(300));
        let m = measure(&text);
        assert_eq!(m.wrapped_lines, 3);
        assert!(m.preview.is_none());
    }

    #[test]
    fn heading_lines_measured_at_heading_size() {
        // ~120 chars of spaced H1 text: at 26px (~15.6px/char, ~29 chars/line
        // at 464px) it wraps to 5 lines — must collapse. A flat body-size
        // measurement would count only ~3 lines and leave it expanded.
        let text = format!("# {}", "heading ".repeat(15));
        let m = measure(&text);
        assert!(
            m.wrapped_lines > 3,
            "long heading must wrap past the budget ({} lines)",
            m.wrapped_lines
        );
        assert!(m.preview.is_some());
    }

    #[test]
    fn indented_atx_heading_measured_at_heading_size() {
        // CommonMark allows up to 3 spaces of indentation before an ATX
        // heading; a long indented heading must be measured at heading size
        // (a body-size count would under-collapse — the forbidden direction).
        let text = format!("  # {}", "heading ".repeat(15));
        let m = measure(&text);
        assert!(
            m.wrapped_lines > 3,
            "long indented heading must wrap past the budget ({} lines)",
            m.wrapped_lines
        );
        assert!(m.preview.is_some());
        // 4+ spaces of indent is not an ATX heading — body-size measurement
        // (and the heading text fits on one line even at 26px, so no wrap).
        let m = measure(&format!("    # {}", "h".repeat(5)));
        assert_eq!(m.wrapped_lines, 1);
        assert!(m.preview.is_none());
    }

    #[test]
    fn body_sized_heading_stays_short() {
        // H5 (13px): a 30-char heading fits on one line.
        let m = measure(&format!("##### {}", "h".repeat(20)));
        assert_eq!(m.wrapped_lines, 1);
        assert!(m.preview.is_none());
    }

    #[test]
    fn setext_heading_measured_at_heading_size() {
        // A long setext title (paragraph line + "===" underline) renders at
        // H1 (26px); a flat body-size measurement would count it as ~2 lines
        // plus the underline (3, expanded) and leave it expanded — the one
        // direction the collapse bias forbids. At 26px it wraps to 4+ lines
        // and must collapse.
        let text = format!("{}\n===", "setext heading title ".repeat(5));
        let m = measure(&text);
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        assert!(m.preview.is_some());
        // A short setext title stays expanded: 1 H1 line + underline = 2.
        let m = measure("short title\n===");
        assert_eq!(m.wrapped_lines, 2);
        assert!(m.preview.is_none());
    }

    #[test]
    fn setext_heading_size_maps_equals_to_h1_and_dash_to_h2() {
        // CommonMark (and pulldown-cmark's scan_setext_heading): "=" → H1
        // (2.0×), "-" → H2 (1.75×). An inverted mapping under-counts a long
        // "=" title at H2 size and can leave an over-budget message expanded
        // — the forbidden direction — so pin the mapping directly (no font
        // system needed; the sizes are constants).
        assert_eq!(
            setext_heading_size("title", Some("===")),
            Some(HEADING_SIZES[0])
        );
        assert_eq!(
            setext_heading_size("title", Some("---")),
            Some(HEADING_SIZES[1])
        );
        // Up to three leading spaces are allowed on the underline (symmetric
        // with the ATX rule); 4+ is not a setext underline.
        assert_eq!(
            setext_heading_size("title", Some("  ===")),
            Some(HEADING_SIZES[0])
        );
        assert_eq!(setext_heading_size("title", Some("    ===")), None);
        // Trailing whitespace after the run is fine; mixed characters, a
        // blank underline, or a blank title are not.
        assert_eq!(
            setext_heading_size("title", Some("---  ")),
            Some(HEADING_SIZES[1])
        );
        assert_eq!(setext_heading_size("title", Some("=-=")), None);
        assert_eq!(setext_heading_size("title", Some("")), None);
        assert_eq!(setext_heading_size("", Some("===")), None);
    }

    #[test]
    fn setext_title_boundary_pins_h1_vs_h2_mapping() {
        // A ~65-char "=" title: measured at H1 (2.0×) the title wraps to 3
        // lines plus the underline = 4 → collapse. The inverted H2 mapping
        // would count 2 title lines plus the underline = 3 → expanded, the
        // forbidden direction. This boundary distinguishes the two mappings;
        // the 105-char title above wraps past 3 at both sizes.
        let text = format!("{}\n===", "setext title ".repeat(5));
        let m = measure(&text);
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        assert!(m.preview.is_some());
    }

    #[test]
    fn base64_data_uri_reduced_to_placeholder() {
        // A valid native data-URI image marker is reduced to a short
        // `🖼️ image` placeholder rather than its base64 payload being counted
        // as token text. (Non-raster markers stay as literal text.)
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
        // Two short words + "🖼️ image" placeholder — fits within the budget.
        assert!(m.wrapped_lines <= 3, "{} lines", m.wrapped_lines);
        assert!(m.preview.is_none());
        // The placeholder must not contain the raw base64 blob.
        let measured = placeholder_media(&media_markers::preprocess(&text));
        assert!(!measured.contains("base64"));
        assert!(measured.contains("🖼️ image"));
    }

    #[test]
    fn image_marker_placeholder_uses_filename() {
        let dir = tempfile::tempdir().unwrap();
        let cap = dir.path().join("cap.png");
        let img = image::RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("test PNG must encode");
        std::fs::write(&cap, &buf).unwrap();

        let text = format!("see [IMAGE:{}] below", cap.display());
        let m = measure(&text);
        assert_eq!(m.wrapped_lines, 1);
        assert!(m.preview.is_none());
        let measured = placeholder_media(&media_markers::preprocess(&text));
        assert!(measured.contains("🖼️ cap.png"), "got: {measured:?}");
    }

    #[test]
    fn preview_is_bounded_for_huge_messages() {
        // A huge message (system-prompt sized) must collapse with a bounded
        // preview, and the measurement must stop early (asserted implicitly
        // by the preview length).
        let text = format!("{}\n{}", "p ".repeat(50_000), "q ".repeat(50_000));
        let m = measure(&text);
        assert!(m.wrapped_lines > 3);
        let preview = m.preview.expect("huge message must collapse");
        assert_preview_lines(&preview, 3);
        assert!(
            preview.len() < 2_000,
            "preview too large: {}",
            preview.len()
        );
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
    fn huge_single_line_message_is_bounded() {
        // A single source line of ~120KB with word breaks: the bounded-prefix
        // shape must prove the collapse and produce the first 3 lines at
        // their true wrap points without shaping the whole line.
        let text = format!("{} end", "word ".repeat(20_000));
        let m = measure(&text);
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        let preview = m.preview.expect("huge single-line message must collapse");
        assert_preview_lines(&preview, 3);
        let flat_preview: String = preview.chars().filter(|c| !c.is_whitespace()).collect();
        let flat_text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            flat_text.starts_with(&flat_preview),
            "preview must be the start of the wrapped line: {preview:?}"
        );
    }

    #[test]
    fn huge_unbroken_word_line_errs_toward_collapse() {
        // A 200KB unbroken word renders as one overflowing line; the bounded
        // prefix cannot see past it, so the measurement errs toward
        // collapsing (the safe direction) instead of shaping the whole line on
        // the UI thread.
        let text = format!("{} tail", "x".repeat(200_000));
        let m = measure(&text);
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        assert!(m.preview.is_some());
    }

    #[test]
    fn long_code_line_preview_capped_to_width() {
        // A collapsing message whose first lines are code: the preview line
        // must be capped to what fits the transcript width (it renders in
        // the same font), not a generous fixed cap — otherwise the
        // "3-line preview" would render as many wrapping visual lines.
        let text = format!("```\n{}\n```\nafter", "code ".repeat(200));
        let m = measure(&text);
        assert!(m.wrapped_lines > 3, "{} lines", m.wrapped_lines);
        let preview = m.preview.expect("must collapse");
        assert_preview_lines(&preview, 3);
        let mut lines = preview.lines();
        assert_eq!(lines.next(), Some("```"));
        let code_line = lines.next().expect("second preview line is the code line");
        // ~59 body chars fit at the test width; the code line is truncated
        // (with "…") far below the old 600-char cap.
        assert!(
            code_line.chars().count() <= 64,
            "code preview line too long: {} chars",
            code_line.chars().count()
        );
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
        // Truncation regression: a mixed-direction line ending or starting
        // with a reversed RTL run sliced the visual-order glyph range,
        // truncating the phrase at its first letter ("...ש"). A collapsing
        // message's preview must keep every Hebrew char in logical order
        // (the phrase may straddle a wrap, so newlines/spaces are ignored).
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
