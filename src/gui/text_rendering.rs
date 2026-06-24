//! Shared text rendering helpers for the editor and diff widgets.
//!
//! These utilities were extracted from [`super::editor_widget`] so that the diff
//! widget can use the same font metrics, gutter geometry, colour conversion,
//! and rich-span merging routines without depending on editor-internal code.
//!
//! All items are `pub(crate)` except [`font_metrics`] which is `pub`.

use iced::advanced::graphics::text::cosmic_text;
use iced::{Color, Rectangle};

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

// ── Geometry helpers ────────────────────────────────────────────────

/// Compute the total height of a cosmic_text buffer, accounting for wrapped
/// lines. Each source line is capped at [`MAX_VISUAL_LINES_PER_SOURCE`]
/// visual lines as a safety limit against pathological single lines
/// (e.g. no-whitespace megabyte lines).
#[allow(clippy::cast_precision_loss)]
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

// ── Colour conversion ───────────────────────────────────────────────

/// Convert an [`iced::Color`] (f32 RGBA components, 0.0–1.0) to
/// [`cosmic_text::Color`] (u8 RGB).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
/// # Safety
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
