//! Shared text rendering helpers for the editor and diff widgets.
//!
//! These utilities were extracted from [`super::editor_widget`] so that the diff
//! widget can use the same font metrics, gutter geometry, colour conversion,
//! and rich-span merging routines without depending on editor-internal code.
//!
//! All items are `pub(crate)` except [`font_metrics`] which is `pub`.

use std::sync::Arc;

use iced::advanced::graphics::text::Raw as TextRaw;
use iced::advanced::graphics::text::cosmic_text;
use iced::advanced::layout::Layout;
use iced::advanced::mouse;
use iced::advanced::renderer;
use iced::{Color, Point, Rectangle};

use super::theme;
use crate::util::UnwrapPoison;

// ── Constants ───────────────────────────────────────────────────────

/// Font metrics used for text rendering (editor buffer, diff viewer).
#[must_use]
pub fn font_metrics() -> cosmic_text::Metrics {
    cosmic_text::Metrics::relative(14.0, 1.3)
}

/// Maximum file size in bytes for which to apply syntax highlighting via
/// tree-sitter. Files larger than this are not parsed for highlighting to
/// avoid blocking the UI thread during parsing.
///
/// Both the editor widget and the diff viewer enforce this limit, sharing
/// the same value to prevent accidental drift.
pub(crate) const MAX_HIGHLIGHT_SIZE: usize = 10 * 1024 * 1024; // 10 MB

/// Font size for line numbers in the gutter.
/// Matches the diff page styling (JetBrains Mono 11px).
pub(crate) const GUTTER_FONT_SIZE: f32 = 11.0;

/// Maximum visual lines per source line as a safety limit against
/// pathological single lines (e.g. no-whitespace megabyte).
pub(crate) const MAX_VISUAL_LINES_PER_SOURCE: usize = 10_000;

// ── Font system access ───────────────────────────────────────────

/// Acquire the global font system and invoke the closure with a mutable
/// reference to it. The font system guard is released after the closure
/// completes.
///
/// This is the canonical way to access the font system for shaping,
/// highlighting, and other text operations, extracted to eliminate
/// repeated `write().unwrap_poison()` boilerplate across editor and
/// diff widgets.
pub(crate) fn with_font_system<R>(f: impl FnOnce(&mut cosmic_text::FontSystem) -> R) -> R {
    let mut guard = iced::advanced::graphics::text::font_system()
        .write()
        .unwrap_poison();
    f(guard.raw())
}

/// Shape (or re-shape) a [`cosmic_text::Buffer`] for a given viewport.
///
/// When `scroll_y` is `Some`, [`set_scroll`] is called **before**
/// [`set_size`] / [`shape_until_scroll`] — this ordering is required by
/// cosmic_text and **must not** be inverted.
///
/// Pass `scroll_y: None` to skip the scroll reset (e.g. in draw fallbacks
/// where [`layout`] already positioned the scroll).
///
/// # Scroll parameters
///
/// `line` is always 0 and `horizontal` is always 0.0 — every current
/// caller places the cursor at the first logical line and left-aligns the
/// viewport.  Accepting these as parameters would complicate every call
/// site for no present benefit; if a future use-case needs different
/// values, add them as optional parameters.
///
/// [`set_scroll`]: cosmic_text::Buffer::set_scroll
/// [`set_size`]: cosmic_text::Buffer::set_size
/// [`shape_until_scroll`]: cosmic_text::Buffer::shape_until_scroll
/// [`layout`]: iced::advanced::widget::Widget::layout
pub(crate) fn reshape_and_shape(
    buffer: &mut cosmic_text::Buffer,
    font_sys: &mut cosmic_text::FontSystem,
    scroll_y: Option<f32>,
    text_area_width: f32,
    text_area_height: f32,
) {
    // set_scroll MUST be called before shape_until_scroll / set_size
    if let Some(scroll_y) = scroll_y {
        buffer.set_scroll(cosmic_text::Scroll {
            line: 0,
            vertical: scroll_y,
            horizontal: 0.0,
        });
    }
    buffer.set_size(font_sys, Some(text_area_width), Some(text_area_height));
    // Ensure shaping runs even if set_size was a no-op (size unchanged)
    buffer.shape_until_scroll(font_sys, false);
}

// ── Geometry helpers ────────────────────────────────────────────────

/// Compute the total height of a cosmic_text buffer, accounting for wrapped
/// lines. Each source line is capped at [`MAX_VISUAL_LINES_PER_SOURCE`]
/// visual lines as a safety limit against pathological single lines
/// (e.g. no-whitespace megabyte lines).
#[expect(clippy::cast_precision_loss)]
pub(crate) fn compute_total_height(
    buffer: &mut cosmic_text::Buffer,
    font_sys: &mut cosmic_text::FontSystem,
    metrics: cosmic_text::Metrics,
) -> f32 {
    let mut total_visual_lines: f32 = 0.0;
    for i in 0..buffer.lines.len() {
        let visual_count = buffer
            .line_layout(font_sys, i)
            .map_or(1, |ll| ll.len().min(MAX_VISUAL_LINES_PER_SOURCE));
        total_visual_lines += visual_count as f32;
    }
    total_visual_lines * metrics.line_height
}

/// Compute the text area rectangle (position and size) inside the given
/// `bounds`, accounting for `padding` and `gutter_width`.
///
/// The returned rectangle has:
/// - `x`: `bounds.x + padding + gutter_width + 4px` gap
/// - `y`: `bounds.y + padding`
/// - `width`: remainder of `bounds.width` after gutter, gap, and padding
/// - `height`: `bounds.height` minus `padding` on both sides
pub(crate) fn text_area_rect(bounds: Rectangle, padding: f32, gutter_width: f32) -> Rectangle {
    let x = bounds.x + padding + gutter_width + 4.0; // 4px gap
    let y = bounds.y + padding;
    let width = (bounds.width - (x - bounds.x) - padding).max(0.0);
    let height = (bounds.height - padding * 2.0).max(0.0);
    Rectangle {
        x,
        y,
        width,
        height,
    }
}

/// Compute `(line, line_start_byte)` for a byte offset within `text`.
/// `offset` must be ≤ `text.len()`. Shared core for the char-column and
/// byte-column variants used by the editor and find/replace helpers.
pub(crate) fn byte_line_and_start(text: &str, offset: usize) -> (usize, usize) {
    let prefix = &text[..offset];
    let line = prefix.bytes().filter(|&b| b == b'\n').count();
    let line_start = prefix.rfind('\n').map_or(0, |p| p + 1);
    (line, line_start)
}

/// Transform a cursor position into buffer-relative coordinates.
///
/// Returns `Some((buf_x, buf_y))` with coordinates relative to the text
/// buffer's origin (i.e., after subtracting padding, gutter, and the 4 px
/// gap between gutter and text).  Returns `None` if the cursor is outside
/// the text area (e.g. in the gutter or padding).
///
/// # Coordinate system
///
/// [`mouse::Cursor::position_in`] (Iced) subtracts `bounds.x` / `bounds.y`
/// from the absolute cursor position, returning coordinates **relative to
/// the widget's top-left corner**.  This function then subtracts the
/// text-area origin (`padding + gutter_width + 4 px`) to obtain
/// buffer-relative coordinates.  **Do not subtract `bounds.x` / `bounds.y`
/// again** — that would double-subtract the layout position, breaking hit
/// detection wherever the widget is not at x = 0 (e.g. beside a sidebar).
pub(crate) fn cursor_to_buffer_coords(
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    gutter_width: f32,
    padding: f32,
) -> Option<(f32, f32)> {
    let bounds = layout.bounds();
    let pos = cursor.position_in(bounds)?;
    let buf_x = pos.x - padding - gutter_width - 4.0;
    let buf_y = pos.y - padding;
    if buf_x < 0.0 || buf_y < 0.0 {
        None
    } else {
        Some((buf_x, buf_y))
    }
}

/// Compute the gutter clip rectangle for line numbers.
pub(crate) fn gutter_clip_rect(
    bounds: Rectangle,
    padding: f32,
    gutter_width: f32,
    text_area_height: f32,
) -> Rectangle {
    Rectangle {
        x: bounds.x + padding,
        y: bounds.y + padding,
        width: gutter_width,
        height: text_area_height,
    }
}

// ── Highlight background rendering ────────────────────────────────

/// Draw a highlighted background rectangle for a [`LayoutRun`] behind text,
/// clipped to the text area.  Used by both the editor and diff widgets for
/// selection, find-match, and bracket-matching highlights.
///
/// * `x_offset` / `width` — the highlight position and span returned by
///   [`LayoutRun::highlight`].
#[expect(clippy::too_many_arguments)]
pub(crate) fn draw_highlight_background<Renderer>(
    renderer: &mut Renderer,
    text_clip: Rectangle,
    text_x: f32,
    text_y: f32,
    run: &cosmic_text::LayoutRun,
    x_offset: f32,
    width: f32,
    color: Color,
) where
    Renderer: iced::advanced::Renderer,
{
    let rect = Rectangle {
        x: text_x + x_offset,
        y: text_y + run.line_top,
        width,
        height: run.line_height,
    };
    if let Some(clipped) = text_clip.intersection(&rect) {
        renderer.fill_quad(
            renderer::Quad {
                bounds: clipped,
                border: iced::Border::default(),
                ..renderer::Quad::default()
            },
            color,
        );
    }
}

/// Fill the widget background. Byte-identical in the editor and diff widgets.
pub(crate) fn draw_background<Renderer>(renderer: &mut Renderer, bounds: Rectangle)
where
    Renderer: iced::advanced::Renderer,
{
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: iced::Border::default(),
            ..renderer::Quad::default()
        },
        theme::BG_BASE,
    );
}

/// Draw buffer glyphs via `fill_raw` for syntax-coloured output.
/// A white neutral multiplier preserves per-glyph colors.
pub(crate) fn draw_buffer_text<Renderer>(
    renderer: &mut Renderer,
    buffer: &Arc<cosmic_text::Buffer>,
    position: Point,
    clip_bounds: Rectangle,
) where
    Renderer: iced::advanced::graphics::text::Renderer,
{
    renderer.fill_raw(TextRaw {
        buffer: Arc::downgrade(buffer),
        position,
        color: Color::WHITE,
        clip_bounds,
    });
}

/// Draw highlight backgrounds for layout runs selected by `filter`.
///
/// `filter` maps a run to the highlight cursor pair (as `(line, byte_index)`
/// tuples) to pass to [`LayoutRun::highlight`], or `None` to skip the run.
/// When `stop_after_first` is set, scanning stops after the first drawn
/// highlight (used by the bracket-match path, which must break after the
/// first run of a logical line).
///
/// `visible_y` optionally restricts processing to runs whose vertical span
/// (`text_y + run.line_top` … `+ run.line_height`, in the same absolute
/// coordinate space as `text_y`/`text_clip`) intersects the given
/// `(top, bottom)` range. Off-screen runs are skipped — and scanning stops
/// once runs fall entirely below the range, since runs are yielded in
/// ascending y order — so per-frame cost scales with the visible content.
/// Pass `None` to process every run (previous behaviour).
#[expect(clippy::too_many_arguments)]
pub(crate) fn draw_run_highlights<Renderer, F>(
    renderer: &mut Renderer,
    buffer: &cosmic_text::Buffer,
    text_clip: Rectangle,
    text_x: f32,
    text_y: f32,
    color: Color,
    stop_after_first: bool,
    visible_y: Option<(f32, f32)>,
    mut filter: F,
) where
    Renderer: iced::advanced::Renderer,
    F: FnMut(&cosmic_text::LayoutRun) -> Option<((usize, usize), (usize, usize))>,
{
    for run in buffer.layout_runs() {
        if let Some((top, bottom)) = visible_y {
            let run_top = text_y + run.line_top;
            let run_bottom = run_top + run.line_height;
            if run_bottom <= top {
                continue;
            }
            if run_top >= bottom {
                break;
            }
        }
        let Some(((start_line, start_idx), (end_line, end_idx))) = filter(&run) else {
            continue;
        };
        let start = cosmic_text::Cursor {
            line: start_line,
            index: start_idx,
            ..cosmic_text::Cursor::default()
        };
        let end = cosmic_text::Cursor {
            line: end_line,
            index: end_idx,
            ..cosmic_text::Cursor::default()
        };
        if let Some((x_offset, width)) = run.highlight(start, end) {
            draw_highlight_background(
                renderer, text_clip, text_x, text_y, &run, x_offset, width, color,
            );
            if stop_after_first {
                break;
            }
        }
    }
}

// ── Colour conversion ───────────────────────────────────────────────

/// Convert an [`iced::Color`] (f32 RGBA components, 0.0–1.0) to
/// [`cosmic_text::Color`] (u8 RGB).
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn iced_color_to_cosmic(c: Color) -> cosmic_text::Color {
    let r = (c.r * 255.0).round() as u8;
    let g = (c.g * 255.0).round() as u8;
    let b = (c.b * 255.0).round() as u8;
    cosmic_text::Color::rgb(r, g, b)
}

// ── Rich-span merging ───────────────────────────────────────────────

/// Push a `(text, attrs)` span to `result`, merging it with the last entry
/// if both the attributes match and the slices are **contiguous** in the
/// backing `text` allocation.
///
/// This keeps the span list as short as possible when adjacent tokens
/// happen to share the same style, which reduces allocation overhead when
/// [`cosmic_text::Buffer::set_rich_text`] processes the list.
///
/// If the slices are **not** contiguous — meaning `new_text` does not
/// immediately follow `last.0` in the source — the function safely
/// falls back to pushing a separate entry. This prevents incorrect
/// attribute application to characters in the gap region between the
/// two slices.
///
/// # Correctness
///
/// Both slices must be subslices of the same `text` allocation. The
/// contiguity check uses pointer arithmetic and would produce undefined
/// behavior if the slices came from different string allocations. All
/// current callers uphold this requirement.
pub(crate) fn push_or_merge<'a>(
    text: &'a str,
    result: &mut Vec<(&'a str, cosmic_text::Attrs<'a>)>,
    new_text: &'a str,
    new_attrs: cosmic_text::Attrs<'a>,
) {
    if let Some(last) = result.last_mut() {
        if last.1 == new_attrs {
            // Compute byte offsets relative to `text` for both slices.
            let start = (last.0.as_ptr() as usize) - (text.as_ptr() as usize);
            let last_end = start + last.0.len();
            let new_start = (new_text.as_ptr() as usize) - (text.as_ptr() as usize);
            // Only merge if the new slice immediately follows the last one
            // in `text`. Non-contiguous slices are pushed separately to
            // avoid applying the wrong attributes to the gap region.
            if last_end == new_start {
                let end = new_start + new_text.len();
                last.0 = &text[start..end];
                return;
            }
        }
    }
    result.push((new_text, new_attrs));
}

/// Build rich spans covering every byte of `text`, filling gaps between
/// `spans` with `base_attrs` and applying per-span attrs.
///
/// `spans` must be sorted by start byte and non-overlapping (the diff
/// widget sorts its `span_data` before calling; the editor emits per-line
/// spans in order). Shared by [`super::editor_widget`] and
/// [`super::diff_widget`]; both previously inlined this loop.
pub(crate) fn fill_rich_spans<'a>(
    text: &'a str,
    spans: impl IntoIterator<Item = (usize, usize, cosmic_text::Attrs<'a>)>,
    base_attrs: &cosmic_text::Attrs<'a>,
) -> Vec<(&'a str, cosmic_text::Attrs<'a>)> {
    let mut result: Vec<(&str, cosmic_text::Attrs)> = Vec::new();
    let mut byte_pos = 0usize;
    for (start, end, attrs) in spans {
        if start > byte_pos {
            push_or_merge(
                text,
                &mut result,
                &text[byte_pos..start],
                base_attrs.clone(),
            );
        }
        if end > start {
            push_or_merge(text, &mut result, &text[start..end], attrs);
            byte_pos = end;
        }
    }
    if byte_pos < text.len() {
        push_or_merge(text, &mut result, &text[byte_pos..], base_attrs.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── push_or_merge tests ────────────────────────────────────────────

    #[test]
    fn push_or_merge_contiguous_same_attrs_merges() {
        let text = "hello world";
        let attrs = cosmic_text::Attrs::new();
        let mut result = Vec::new();
        result.push((&text[0..5], attrs.clone()));
        push_or_merge(text, &mut result, &text[5..11], attrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "hello world");
    }

    #[test]
    fn push_or_merge_non_contiguous_same_attrs_pushes_separately() {
        let text = "hello---world";
        let attrs = cosmic_text::Attrs::new();
        let mut result = Vec::new();
        result.push((&text[0..5], attrs.clone()));
        // "world" starts at byte 8, not immediately after "hello" (5..8 is "---")
        push_or_merge(text, &mut result, &text[8..13], attrs);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "hello");
        assert_eq!(result[1].0, "world");
    }

    #[test]
    fn push_or_merge_contiguous_different_attrs_pushes_separately() {
        let text = "hello world";
        let attrs1 = cosmic_text::Attrs::new();
        let attrs2 = cosmic_text::Attrs::new().color(cosmic_text::Color::rgb(255, 0, 0));
        let mut result = Vec::new();
        result.push((&text[0..5], attrs1));
        push_or_merge(text, &mut result, &text[5..11], attrs2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn push_or_merge_empty_result_pushes() {
        let text = "hello";
        let attrs = cosmic_text::Attrs::new();
        let mut result = Vec::new();
        push_or_merge(text, &mut result, &text[0..5], attrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "hello");
    }
}
