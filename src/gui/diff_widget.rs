//! A [`cosmic_text::Buffer`]-backed widget for rendering diff file content
//! with syntax highlighting, gutter line numbers, and per-line background tints.
//!
//! Each diff file gets its own [`DiffBufferWidget`]. File headers, binary
//! placeholders, and truncation warnings remain Iced widgets interleaved
//! between per-file buffer widgets. The entire diff content panel is wrapped
//! in an Iced scrollable — per-file buffers do NOT manage their own scroll.
//!
//! ## Buffer content format
//!
//! The buffer text string does NOT include the gutter. Each logical line:
//! `"{prefix} {content}\n"` — e.g., `"+ let x = 42;\n"`. The prefix character
//! (`+`, `-`, ` `) is part of the buffer text. Hunk headers are inserted as
//! full text lines: `"@@ -10,7 +10,9 @@ fn main() {\n"`.
//!
//! ## Gutter rendering
//!
//! Gutter (old/new line numbers) is rendered entirely in `draw()` via
//! `fill_text` — it is NOT part of the buffer text at all. This avoids the
//! problem of gutter text being repeated on wrapped continuation lines.

use std::sync::Arc;

use iced::advanced::graphics::text::Raw as TextRaw;
use iced::advanced::graphics::text::cosmic_text;
use iced::advanced::layout::{self, Layout};
use iced::advanced::text;
use iced::advanced::widget::{self, Tree, Widget};
use iced::advanced::{Clipboard, Shell};
use iced::advanced::{mouse, renderer};
use iced::keyboard;
use iced::{Color, Event, Length, Point, Rectangle, Size};

use crate::diff_parse::{DiffFileStatus, DiffLineKind};

use super::text_rendering::{
    GUTTER_FONT_SIZE, compute_total_height, font_metrics, gutter_clip_rect, iced_color_to_cosmic,
    push_or_merge, text_area_rect, with_font_system,
};
use super::theme;

// ── Constants ───────────────────────────────────────────────────────

/// Hunk header text color — ayu dark sky blue (matches HighlightClass::Type).
pub(crate) const HUNK_HEADER_COLOR: Color = Color::from_rgb(0.349, 0.761, 1.0);

/// Added line foreground color.
pub(crate) const ADDED_COLOR: Color = theme::STATUS_SUCCESS;

/// Removed line foreground color.
pub(crate) const REMOVED_COLOR: Color = theme::STATUS_ERROR;

/// Context line foreground color.
pub(crate) const CONTEXT_COLOR: Color = theme::TEXT_SECONDARY;

/// Approximate pixel width of one monospace digit at [`GUTTER_FONT_SIZE`].
const GUTTER_DIGIT_WIDTH: f32 = GUTTER_FONT_SIZE * 0.62;

/// Selection highlight fill color.
const SELECTION_COLOR: Color = theme::ACCENT_DIM;

// ── Per-file buffer data (pre-computed in update) ───────────────────

/// Pre-built data for rendering one diff file via [`DiffBufferWidget`].
/// All string formatting and span computation happens when this struct
/// is built (on diff load / file selection change), not per-frame.
pub struct DiffFileBuffer {
    /// The full buffer text: hunk headers + prefixed diff lines, newline-terminated.
    pub text: String,
    /// Per-span data: `(start_byte, end_byte, iced_color)` — covers the entire text.
    /// Gaps between spans use `theme::TEXT_PRIMARY` as the default color.
    pub span_data: Vec<(usize, usize, Color)>,
    /// Per-logical-line kind: `None` for hunk headers, `Some(kind)` for diff lines.
    pub line_kinds: Vec<Option<DiffLineKind>>,
    /// Per-logical-line line numbers: `(old_num, new_num)`.
    /// Both `None` for hunk headers and lines without line numbers.
    pub line_numbers: Vec<(Option<usize>, Option<usize>)>,
    /// Digit count for the widest old/new line number (minimum 1).
    pub gutter_digits: usize,
}

// ── Widget state ─────────────────────────────────────────────────────

/// Persistent state stored in `widget::Tree::State`.
#[derive(Default)]
struct DiffBufferState {
    /// The `Arc<Buffer>` must live across frames for `fill_raw` to work.
    buffer_for_render: Option<Arc<cosmic_text::Buffer>>,
    /// Cached gutter width in pixels (computed per frame in layout).
    gutter_width: f32,
    /// Selection anchor byte offset in buffer text (`None` = no selection).
    sel_anchor: Option<usize>,
    /// Selection cursor/end byte offset in buffer text.
    sel_cursor: Option<usize>,
    /// Whether the left mouse button is held for drag-selection.
    mouse_held: bool,
}

// ── Widget ───────────────────────────────────────────────────────────

/// A custom Iced widget that renders a single diff file's content via
/// [`cosmic_text::Buffer`], with gutter line numbers and per-line background
/// tints. Designed to be used inside a parent `scrollable` — it reports
/// its full content height.
pub struct DiffBufferWidget<'a> {
    data: &'a DiffFileBuffer,
    padding: f32,
}

impl<'a> DiffBufferWidget<'a> {
    /// Create a new [`DiffBufferWidget`] from pre-computed buffer data.
    pub const fn new(data: &'a DiffFileBuffer) -> Self {
        Self { data, padding: 8.0 }
    }
}

/// Compute the number of digits needed for the widest old/new line number.
#[must_use]
pub(crate) fn compute_gutter_digits(line_numbers: &[(Option<usize>, Option<usize>)]) -> usize {
    line_numbers
        .iter()
        .flat_map(|(old, new)| [*old, *new])
        .flatten()
        .map(|n| n.to_string().len())
        .max()
        .unwrap_or(1)
}

/// Pixel width for a dual-column old/new line-number gutter.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub(crate) const fn gutter_width_from_digits(digits: usize) -> f32 {
    digits as f32 * GUTTER_DIGIT_WIDTH * 2.0 + 14.0
}

/// Right-edge anchors for the old/new gutter columns.
///
/// `fill_text` uses the point as the right edge when `align_x` is `Right`.
/// This mirrors the main editor gutter; passing each column's left edge clips
/// or shifts line numbers into the gutter padding.
fn gutter_column_right_edges(bounds_x: f32, padding: f32, gutter_width: f32) -> (f32, f32) {
    let half_gutter = gutter_width / 2.0;
    (
        bounds_x + padding + half_gutter,
        bounds_x + padding + gutter_width,
    )
}

/// Convert a shaped-buffer hit to a global byte offset in the source text.
fn hit_to_global_byte(buffer: &cosmic_text::Buffer, line: usize, index: usize) -> usize {
    let mut global = 0usize;
    for i in 0..line {
        if let Some(l) = buffer.lines.get(i) {
            global += l.text().len() + 1;
        }
    }
    global + index
}

/// Convert a global byte offset back to a cosmic_text cursor.
fn global_byte_to_cursor(buffer: &cosmic_text::Buffer, global: usize) -> cosmic_text::Cursor {
    let mut pos = 0usize;
    for (i, line) in buffer.lines.iter().enumerate() {
        let text = line.text();
        let line_len = text.len();
        let next = pos.saturating_add(line_len);
        if global <= next || i + 1 == buffer.lines.len() {
            return cosmic_text::Cursor {
                line: i,
                index: global.saturating_sub(pos).min(line_len),
                ..cosmic_text::Cursor::default()
            };
        }
        pos = next + 1;
    }
    cosmic_text::Cursor::default()
}

/// Hit-test mouse position against the diff text area.
fn hit_test(
    buffer: &cosmic_text::Buffer,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    gutter_width: f32,
    padding: f32,
) -> Option<usize> {
    let bounds = layout.bounds();
    let pos = cursor.position_in(bounds)?;

    let text_x = padding + gutter_width + 4.0;
    let text_y = padding;
    let buf_x = pos.x - bounds.x - text_x;
    let buf_y = pos.y - bounds.y - text_y;

    if buf_x < 0.0 || buf_y < 0.0 {
        return None;
    }

    let hit = buffer.hit(buf_x, buf_y)?;
    Some(hit_to_global_byte(buffer, hit.line, hit.index))
}

/// Extract selected text from global byte offsets (excludes gutter numbers).
fn selection_text(text: &str, anchor: usize, cursor: usize) -> Option<String> {
    if anchor == cursor {
        return None;
    }
    let (start, end) = if anchor < cursor {
        (anchor, cursor)
    } else {
        (cursor, anchor)
    };
    if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return None;
    }
    Some(text[start..end].to_string())
}

/// Whether a non-empty selection is active.
fn has_selection(anchor: Option<usize>, cursor: Option<usize>) -> bool {
    match (anchor, cursor) {
        (Some(a), Some(c)) => a != c,
        _ => false,
    }
}

// ── Iced Widget impl ─────────────────────────────────────────────────

impl<Theme, Renderer> Widget<super::diff::DiffMessage, Theme, Renderer> for DiffBufferWidget<'_>
where
    Renderer: iced::advanced::Renderer
        + iced::advanced::graphics::text::Renderer
        + iced::advanced::text::Renderer<Font = iced::Font>,
{
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Shrink)
    }

    fn state(&self) -> widget::tree::State {
        widget::tree::State::Some(Box::<DiffBufferState>::default())
    }

    fn tag(&self) -> widget::tree::Tag {
        widget::tree::Tag::of::<DiffBufferState>()
    }

    #[allow(clippy::cast_precision_loss)]
    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let bounds = limits.max();
        let state = tree.state.downcast_mut::<DiffBufferState>();

        // ── Gutter width ───────────────────────────────────────────
        let gutter_width = gutter_width_from_digits(self.data.gutter_digits);
        state.gutter_width = gutter_width;

        let text_x = self.padding + gutter_width + 4.0; // 4px gap
        let text_area_width = (bounds.width - text_x - self.padding).max(0.0);

        // No content — collapse to zero height
        if self.data.text.is_empty() {
            return layout::Node::new(Size::new(bounds.width, 0.0));
        }

        let metrics = font_metrics();

        let (buffer, total_height) = with_font_system(|font_sys| {
            let mut buffer = cosmic_text::Buffer::new(font_sys, font_metrics());

            // ── Build rich spans from pre-computed span data ────────────
            let text = &self.data.text;
            let base_attrs = cosmic_text::Attrs::new()
                .family(cosmic_text::Family::Name("JetBrains Mono"))
                .color(iced_color_to_cosmic(theme::TEXT_PRIMARY));

            let mut rich_spans: Vec<(&str, cosmic_text::Attrs)> = Vec::new();
            let mut byte_pos = 0usize;
            for &(start, end, color) in &self.data.span_data {
                if start > byte_pos {
                    push_or_merge(
                        text,
                        &mut rich_spans,
                        &text[byte_pos..start],
                        base_attrs.clone(),
                    );
                }
                if end > start {
                    let attrs = base_attrs.clone().color(iced_color_to_cosmic(color));
                    push_or_merge(text, &mut rich_spans, &text[start..end], attrs);
                    byte_pos = end;
                }
            }
            // Cover any remaining text after the last span
            if byte_pos < text.len() {
                push_or_merge(text, &mut rich_spans, &text[byte_pos..], base_attrs.clone());
            }

            buffer.set_rich_text(
                font_sys,
                rich_spans,
                &base_attrs,
                cosmic_text::Shaping::Advanced,
                None,
            );
            buffer.set_scroll(cosmic_text::Scroll {
                line: 0,
                vertical: 0.0,
                horizontal: 0.0,
            });
            buffer.set_size(font_sys, Some(text_area_width), None);
            buffer.shape_until_scroll(font_sys, false);

            // ── Compute total height ────────────────────────────────────
            // Cap each source line at MAX_VISUAL_LINES_PER_SOURCE visual lines
            let total_height: f32 = compute_total_height(&mut buffer, font_sys, metrics);

            (buffer, total_height)
        });

        // Move the shaped buffer into an Arc and store for fill_raw
        let arc = Arc::new(buffer);
        state.buffer_for_render = Some(arc);

        layout::Node::new(Size::new(bounds.width, total_height + self.padding * 2.0))
    }

    #[allow(clippy::too_many_lines)]
    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<DiffBufferState>();
        let bounds = layout.bounds();
        let gutter_width = state.gutter_width;

        // ── 0. Fill background ──
        renderer.fill_quad(
            renderer::Quad {
                bounds,
                border: iced::Border::default(),
                ..renderer::Quad::default()
            },
            theme::BG_BASE,
        );

        let text_rect = text_area_rect(bounds, self.padding, gutter_width);
        let text_x = text_rect.x;
        let text_y = text_rect.y;
        let text_area_width = text_rect.width;
        let text_area_height = text_rect.height;

        let text_clip = text_rect;

        let buffer_for_draw = match &state.buffer_for_render {
            Some(arc) => arc.clone(),
            None => return,
        };

        // ── 1. Draw per-line background tints (behind text) ─────────
        // Use the same pattern as gutter: only draw background for the
        // first visual run of each logical line.
        let mut last_bg_line = usize::MAX;
        for run in buffer_for_draw.layout_runs() {
            if run.line_i == last_bg_line {
                continue; // wrapped continuation — already drawn
            }
            last_bg_line = run.line_i;

            if run.line_i >= self.data.line_kinds.len() {
                continue;
            }

            let bg_color = match self.data.line_kinds[run.line_i] {
                Some(DiffLineKind::Added) => Some(Color::from_rgba(0.0, 0.902, 0.541, 0.10)),
                Some(DiffLineKind::Removed) => Some(Color::from_rgba(1.0, 0.267, 0.4, 0.10)),
                _ => None, // context or hunk header — no tint
            };

            if let Some(color) = bg_color {
                let rect = Rectangle {
                    x: text_x,
                    y: text_y + run.line_top,
                    width: text_area_width,
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
        }

        // ── 2. Draw line numbers (gutter) ───────────────────────────
        let number_color = theme::TEXT_MUTED;
        let gutter_clip = gutter_clip_rect(bounds, self.padding, gutter_width, text_area_height);

        let mut last_drawn_line = usize::MAX;
        for run in buffer_for_draw.layout_runs() {
            // Only draw gutter for the first visual line of each logical line
            if run.line_i == last_drawn_line {
                continue;
            }
            last_drawn_line = run.line_i;

            if run.line_i >= self.data.line_numbers.len() {
                continue;
            }

            let (old_num, new_num) = self.data.line_numbers[run.line_i];

            let half_gutter = gutter_width / 2.0;
            let (old_right_x, new_right_x) =
                gutter_column_right_edges(bounds.x, self.padding, gutter_width);

            // Draw old line number (right-aligned in left half)
            let old_str = old_num.map_or_else(String::new, |n| n.to_string());
            if !old_str.is_empty() {
                let num_text = text::Text {
                    content: old_str,
                    bounds: Size::new(half_gutter, run.line_height),
                    size: iced::Pixels(GUTTER_FONT_SIZE),
                    line_height: text::LineHeight::Relative(1.3),
                    font: theme::FONT_REGULAR,
                    align_x: iced::alignment::Horizontal::Right.into(),
                    align_y: iced::alignment::Vertical::Center,
                    shaping: text::Shaping::Advanced,
                    wrapping: text::Wrapping::None,
                };
                renderer.fill_text(
                    num_text,
                    Point::new(old_right_x, text_y + run.line_top + run.line_height / 2.0),
                    number_color,
                    gutter_clip,
                );
            }

            // Draw new line number (right-aligned in right half)
            let new_str = new_num.map_or_else(String::new, |n| n.to_string());
            if !new_str.is_empty() {
                let num_text = text::Text {
                    content: new_str,
                    bounds: Size::new(half_gutter, run.line_height),
                    size: iced::Pixels(GUTTER_FONT_SIZE),
                    line_height: text::LineHeight::Relative(1.3),
                    font: theme::FONT_REGULAR,
                    align_x: iced::alignment::Horizontal::Right.into(),
                    align_y: iced::alignment::Vertical::Center,
                    shaping: text::Shaping::Advanced,
                    wrapping: text::Wrapping::None,
                };
                renderer.fill_text(
                    num_text,
                    Point::new(new_right_x, text_y + run.line_top + run.line_height / 2.0),
                    number_color,
                    gutter_clip,
                );
            }
        }

        // ── 3. Draw selection highlight (behind text) ───────────────
        if has_selection(state.sel_anchor, state.sel_cursor) {
            let (anchor, cursor) = (state.sel_anchor.unwrap(), state.sel_cursor.unwrap());
            let (start, end) = if anchor < cursor {
                (anchor, cursor)
            } else {
                (cursor, anchor)
            };
            let start_cur = global_byte_to_cursor(&buffer_for_draw, start);
            let end_cur = global_byte_to_cursor(&buffer_for_draw, end);

            for run in buffer_for_draw.layout_runs() {
                if let Some(highlight) = run.highlight(start_cur, end_cur) {
                    let sel_rect = Rectangle {
                        x: text_x + highlight.0,
                        y: text_y + run.line_top,
                        width: highlight.1,
                        height: run.line_height,
                    };
                    if let Some(clipped) = text_clip.intersection(&sel_rect) {
                        renderer.fill_quad(
                            renderer::Quad {
                                bounds: clipped,
                                border: iced::Border::default(),
                                ..renderer::Quad::default()
                            },
                            SELECTION_COLOR,
                        );
                    }
                }
            }
        }

        // ── 4. Draw text via fill_raw ───────────────────────────────
        renderer.fill_raw(TextRaw {
            buffer: Arc::downgrade(&buffer_for_draw),
            position: Point::new(text_x, text_y),
            color: Color::WHITE, // neutral multiplier preserves per-glyph colors
            clip_bounds: text_clip,
        });
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, super::diff::DiffMessage>,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<DiffBufferState>();
        let buffer = match &state.buffer_for_render {
            Some(arc) => arc.clone(),
            None => return,
        };

        match event {
            Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                if let Some(byte) =
                    hit_test(&buffer, layout, cursor, state.gutter_width, self.padding)
                {
                    state.mouse_held = true;
                    state.sel_anchor = Some(byte);
                    state.sel_cursor = Some(byte);
                    shell.request_redraw();
                } else {
                    state.mouse_held = false;
                    state.sel_anchor = None;
                    state.sel_cursor = None;
                    shell.request_redraw();
                }
            }

            Event::Mouse(iced::mouse::Event::ButtonReleased(iced::mouse::Button::Left)) => {
                state.mouse_held = false;
                if let (Some(a), Some(c)) = (state.sel_anchor, state.sel_cursor) {
                    if a == c {
                        state.sel_anchor = None;
                        state.sel_cursor = None;
                    }
                }
                shell.request_redraw();
            }

            Event::Mouse(iced::mouse::Event::CursorMoved { .. }) if state.mouse_held => {
                if let Some(byte) =
                    hit_test(&buffer, layout, cursor, state.gutter_width, self.padding)
                {
                    state.sel_cursor = Some(byte);
                    shell.request_redraw();
                }
            }

            Event::Keyboard(keyboard::Event::KeyPressed {
                key: key_press,
                modifiers,
                physical_key,
                ..
            }) => {
                #[cfg(target_os = "macos")]
                let is_clipboard_mod = modifiers.command() && !modifiers.control();
                #[cfg(not(target_os = "macos"))]
                let is_clipboard_mod =
                    (modifiers.command() || modifiers.control()) && !modifiers.alt();

                if is_clipboard_mod
                    && key_press.to_latin(*physical_key) == Some('c')
                    && let (Some(anchor), Some(cursor_byte)) = (state.sel_anchor, state.sel_cursor)
                    && let Some(text) = selection_text(&self.data.text, anchor, cursor_byte)
                {
                    clipboard.write(iced::advanced::clipboard::Kind::Standard, text);
                }
            }

            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let bounds = layout.bounds();
        if cursor.position_over(bounds).is_some() {
            mouse::Interaction::Text
        } else {
            mouse::Interaction::default()
        }
    }
}

// ── Builder ──────────────────────────────────────────────────────────

/// Build per-file [`DiffFileBuffer`] data from a slice of [`super::diff::DiffFile`]s.
///
/// Called in `update()` when diff data or file selection changes.
/// The resulting buffers are consumed by [`DiffBufferWidget`] in `view()`.
///
/// When `limits` is `Some((max_hunks, max_lines))`, stops building buffers once
/// cumulative hunk/line counts would exceed the caps (matches view truncation).
pub fn build_file_buffers(
    diff_files: &[super::diff::DiffFile],
    selected_file: Option<&str>,
    limits: Option<(usize, usize)>,
) -> Vec<DiffFileBuffer> {
    // Single source of truth for truncation — shared with build_diff_content.
    // compute_truncation_index already handles `None` limits internally.
    let truncate_at = super::diff::compute_truncation_index(diff_files, selected_file, limits);
    let mut buffers: Vec<DiffFileBuffer> = Vec::new();

    for (idx, file) in diff_files.iter().enumerate() {
        if let Some(sel) = selected_file {
            if file.dfile.path != *sel {
                continue;
            }
        }

        // Truncation check — use the pre-computed index from the shared helper.
        if let Some(limit) = truncate_at {
            if idx >= limit {
                break;
            }
        }

        // File headers, binary, too-large — these are rendered as Iced
        // widgets interleaved with DiffBufferWidgets. We skip buffer
        // construction for binary and too-large files.
        if file.dfile.is_binary || file.dfile.too_large_size.is_some() {
            continue;
        }

        buffers.push(build_single_file_buffer(file));
    }

    buffers
}

/// Build the [`DiffFileBuffer`] for a single file.
fn build_single_file_buffer(file: &super::diff::DiffFile) -> DiffFileBuffer {
    let mut text = String::new();
    // Pre-allocate: rough estimate of 80 bytes per line
    let estimated_lines: usize = file
        .dfile
        .hunks
        .iter()
        .map(|h| h.lines.len() + 1) // +1 for hunk header
        .sum();
    text.reserve(estimated_lines * 80);

    let mut span_data: Vec<(usize, usize, Color)> = Vec::new();
    let mut line_kinds: Vec<Option<DiffLineKind>> = Vec::new();
    let mut line_numbers: Vec<(Option<usize>, Option<usize>)> = Vec::new();

    for hunk in &file.dfile.hunks {
        // Hunk header line
        {
            let start = text.len();
            text.push_str(&hunk.header);
            text.push('\n');
            let end = text.len();
            span_data.push((start, end, HUNK_HEADER_COLOR));
            line_kinds.push(None);
            line_numbers.push((None, None));
        }

        for line in &hunk.lines {
            let start = text.len();
            let line_start = start;

            // Prefix
            text.push(line.kind.prefix());
            text.push(' ');
            let content_start = text.len();

            // Content
            text.push_str(&line.content);
            text.push('\n');
            let end = text.len();

            let fg_color = match line.kind {
                DiffLineKind::Added => ADDED_COLOR,
                DiffLineKind::Removed => REMOVED_COLOR,
                DiffLineKind::Context => CONTEXT_COLOR,
            };
            let kind = line.kind;

            let content_len = end - content_start;

            // Select highlight source based on line kind and file status
            let (highlights, hl_line_number) = match (line.kind, file.dfile.status) {
                (DiffLineKind::Removed, _) | (DiffLineKind::Context, DiffFileStatus::Deleted) => {
                    (file.old_highlights.as_ref(), line.old_line_number)
                }
                (DiffLineKind::Added | DiffLineKind::Context, _) => {
                    (file.new_highlights.as_ref(), line.new_line_number)
                }
            };

            // Build spans for this line
            let line_hl_spans = hl_line_number
                .and_then(|n| highlights.and_then(|h| h.spans.get(n.saturating_sub(1))));

            match line_hl_spans {
                Some(hl_spans) if !hl_spans.is_empty() => {
                    let mut cursor = content_start;
                    for s in hl_spans {
                        // s.start and s.end are relative to content start (0-based)
                        if s.start >= s.end || s.start >= content_len {
                            continue;
                        }
                        let abs_start = content_start + s.start.min(content_len);
                        let abs_end = content_start + s.end.min(content_len);
                        if abs_start > cursor {
                            // Gap before this span — fill with fg color
                            span_data.push((cursor, abs_start, fg_color));
                        }
                        if abs_end > abs_start {
                            span_data.push((abs_start, abs_end, s.highlight_class.color()));
                            cursor = abs_end;
                        }
                    }
                    // Remaining content after last highlight
                    if cursor < end {
                        span_data.push((cursor, end, fg_color));
                    }
                    // Also cover the prefix portion
                    span_data.push((line_start, content_start, fg_color));
                }
                _ => {
                    // No highlights — entire line in fg color
                    span_data.push((line_start, end, fg_color));
                }
            }

            line_kinds.push(Some(kind));
            line_numbers.push((line.old_line_number, line.new_line_number));
        }
    }

    // Sort span_data by start byte (they may be out of order due to
    // prefix being pushed last in the highlighted case)
    span_data.sort_by_key(|(start, _, _)| *start);

    let gutter_digits = compute_gutter_digits(&line_numbers);

    DiffFileBuffer {
        text,
        span_data,
        line_kinds,
        line_numbers,
        gutter_digits,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff_parse::{DiffFileStatus, DiffLine, DiffLineKind};

    fn make_test_diff_file(
        path: &str,
        hunks: Vec<crate::diff_parse::DiffHunk>,
        status: DiffFileStatus,
    ) -> super::super::diff::DiffFile {
        super::super::diff::DiffFile {
            dfile: crate::diff_parse::DiffFile {
                path: path.to_string(),
                old_path: None,
                hunks,
                status,
                is_binary: false,
                too_large_size: None,
            },
            old_highlights: None,
            new_highlights: None,
            add_count: 0,
            remove_count: 0,
        }
    }

    fn make_line(
        kind: DiffLineKind,
        content: &str,
        old: Option<usize>,
        new: Option<usize>,
    ) -> DiffLine {
        DiffLine {
            kind,
            old_line_number: old,
            new_line_number: new,
            content: content.to_string(),
        }
    }

    fn make_hunk(header: &str, lines: Vec<DiffLine>) -> crate::diff_parse::DiffHunk {
        crate::diff_parse::DiffHunk {
            header: header.to_string(),
            lines,
        }
    }

    #[test]
    fn test_empty_file_has_no_buffers() {
        let files: Vec<super::super::diff::DiffFile> = Vec::new();
        let buffers = build_file_buffers(&files, None, None);
        assert!(buffers.is_empty());
    }

    #[test]
    fn test_binary_file_skipped() {
        let mut file = make_test_diff_file("binary.bin", vec![], DiffFileStatus::Modified);
        file.dfile.is_binary = true;
        let buffers = build_file_buffers(&[file], None, None);
        assert!(buffers.is_empty());
    }

    #[test]
    fn test_too_large_file_skipped() {
        let mut file = make_test_diff_file("large.bin", vec![], DiffFileStatus::Modified);
        file.dfile.too_large_size = Some(5_000_000);
        let buffers = build_file_buffers(&[file], None, None);
        assert!(buffers.is_empty());
    }

    #[test]
    fn test_selected_file_filter() {
        let file_a = make_test_diff_file(
            "a.rs",
            vec![make_hunk(
                "@@ -1,3 +1,3 @@",
                vec![make_line(DiffLineKind::Context, "line1", Some(1), Some(1))],
            )],
            DiffFileStatus::Modified,
        );
        let file_b = make_test_diff_file(
            "b.rs",
            vec![make_hunk(
                "@@ -1,2 +1,2 @@",
                vec![make_line(DiffLineKind::Added, "new", None, Some(1))],
            )],
            DiffFileStatus::Modified,
        );
        let buffers = build_file_buffers(&[file_a, file_b], Some("b.rs"), None);
        assert_eq!(buffers.len(), 1);
        assert!(buffers[0].text.contains("new"));
    }

    #[test]
    fn test_buffer_text_format() {
        let file = make_test_diff_file(
            "test.rs",
            vec![make_hunk(
                "@@ -1,3 +1,4 @@ fn main() {",
                vec![
                    make_line(DiffLineKind::Context, "let x = 1;", Some(1), Some(1)),
                    make_line(DiffLineKind::Removed, "let y = 2;", Some(2), None),
                    make_line(DiffLineKind::Added, "let z = 3;", None, Some(2)),
                ],
            )],
            DiffFileStatus::Modified,
        );
        let buffers = build_file_buffers(&[file], None, None);
        assert_eq!(buffers.len(), 1);

        let buf = &buffers[0];
        // Text should contain hunk header + prefixed lines
        assert!(buf.text.starts_with("@@ -1,3 +1,4 @@ fn main() {\n"));
        assert!(buf.text.contains("  let x = 1;\n")); // context: "  " prefix
        assert!(buf.text.contains("- let y = 2;\n")); // removed: "- " prefix
        assert!(buf.text.contains("+ let z = 3;\n")); // added: "+ " prefix

        // Line kinds: hunk header is None, then Context, Removed, Added
        assert_eq!(buf.line_kinds.len(), 4); // hunk header + 3 lines
        assert_eq!(buf.line_kinds[0], None);
        assert_eq!(buf.line_kinds[1], Some(DiffLineKind::Context));
        assert_eq!(buf.line_kinds[2], Some(DiffLineKind::Removed));
        assert_eq!(buf.line_kinds[3], Some(DiffLineKind::Added));

        // Line numbers
        assert_eq!(buf.line_numbers[0], (None, None)); // hunk header
        assert_eq!(buf.line_numbers[1], (Some(1), Some(1)));
        assert_eq!(buf.line_numbers[2], (Some(2), None));
        assert_eq!(buf.line_numbers[3], (None, Some(2)));

        // Span data should cover the entire text
        let total_span_len: usize = buf.span_data.iter().map(|(s, e, _)| e - s).sum();
        assert_eq!(total_span_len, buf.text.len());

        // Spans should be sorted by start
        let starts: Vec<usize> = buf.span_data.iter().map(|(s, _, _)| *s).collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]));

        // Span data should be non-empty with valid byte ranges
        for &(start, end, _) in &buf.span_data {
            assert!(start <= end, "start {start} > end {end}");
            assert!(
                end <= buf.text.len(),
                "end {end} > text len {}",
                buf.text.len()
            );
            assert!(
                buf.text.is_char_boundary(start),
                "start {start} not char boundary"
            );
            assert!(
                buf.text.is_char_boundary(end),
                "end {end} not char boundary"
            );
        }
    }

    #[test]
    fn test_hunk_header_color() {
        let file = make_test_diff_file(
            "test.rs",
            vec![make_hunk(
                "@@ -5,7 +5,7 @@ fn foo() {",
                vec![make_line(DiffLineKind::Context, "bar();", Some(5), Some(5))],
            )],
            DiffFileStatus::Modified,
        );
        let buffers = build_file_buffers(&[file], None, None);
        let buf = &buffers[0];

        // The first span should cover the hunk header and have HUNK_HEADER_COLOR
        assert!(!buf.span_data.is_empty());
        let (start, end, color) = buf.span_data[0];
        assert_eq!(start, 0);
        // The span should cover the hunk header line
        let hunk_line = &buf.text[start..end];
        assert!(hunk_line.starts_with("@@ -5,7"));
        assert_eq!(color, HUNK_HEADER_COLOR);
    }

    #[test]
    fn test_added_line_has_green_tint_kind() {
        let file = make_test_diff_file(
            "test.rs",
            vec![make_hunk(
                "@@ -1,0 +1,1 @@",
                vec![make_line(DiffLineKind::Added, "+ new_line", None, Some(1))],
            )],
            DiffFileStatus::Added,
        );
        let buffers = build_file_buffers(&[file], None, None);
        let buf = &buffers[0];
        // Hunk header + added line
        assert_eq!(buf.line_kinds, vec![None, Some(DiffLineKind::Added)]);
    }

    #[test]
    fn test_multiple_hunks_produce_correct_line_count() {
        let file = make_test_diff_file(
            "multi.rs",
            vec![
                make_hunk(
                    "@@ -1,2 +1,2 @@",
                    vec![
                        make_line(DiffLineKind::Context, "a", Some(1), Some(1)),
                        make_line(DiffLineKind::Context, "b", Some(2), Some(2)),
                    ],
                ),
                make_hunk(
                    "@@ -10,1 +10,1 @@",
                    vec![make_line(DiffLineKind::Removed, "old", Some(10), None)],
                ),
            ],
            DiffFileStatus::Modified,
        );
        let buffers = build_file_buffers(&[file], None, None);
        let buf = &buffers[0];
        // Two hunk headers + 2 context + 1 removed = 5 logical lines
        assert_eq!(buf.line_kinds.len(), 5);
        assert_eq!(buf.line_numbers.len(), 5);
    }

    #[test]
    fn test_gutter_digits_from_source_line_numbers_not_row_count() {
        let file = make_test_diff_file(
            "test.rs",
            vec![make_hunk(
                "@@ -12345,1 +12345,1 @@",
                vec![make_line(
                    DiffLineKind::Context,
                    "far down",
                    Some(12_345),
                    Some(12_345),
                )],
            )],
            DiffFileStatus::Modified,
        );
        let buffers = build_file_buffers(&[file], None, None);
        assert_eq!(buffers[0].gutter_digits, 5);
        assert!(
            gutter_width_from_digits(5) > gutter_width_from_digits(1),
            "high line numbers need a wider gutter"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_gutter_column_positions_are_right_edges() {
        let width = gutter_width_from_digits(3);
        let (old_right, new_right) = gutter_column_right_edges(10.0, 8.0, width);

        assert_eq!(old_right, 10.0 + 8.0 + width / 2.0);
        assert_eq!(new_right, 10.0 + 8.0 + width);
        assert!(new_right > old_right);
    }

    #[test]
    fn test_selection_text_excludes_gutter_and_handles_ranges() {
        let text = "+ hello\n- world\n";
        assert_eq!(selection_text(text, 0, 7), Some("+ hello".to_string()));
        assert_eq!(selection_text(text, 7, 0), Some("+ hello".to_string()));
        assert!(selection_text(text, 3, 3).is_none());
    }

    #[test]
    fn test_compute_gutter_digits_empty_defaults_to_one() {
        assert_eq!(compute_gutter_digits(&[]), 1);
    }
}
