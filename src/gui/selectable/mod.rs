//! Read-only selectable rich text — the widget every GUI reading surface
//! renders through.
//!
//! A highlight rectangle is a visual run's own line box (`line_top` /
//! `line_height`) with its x-range from cosmic-text's `LayoutRun::highlight`,
//! so the paint follows the real per-visual-line layout instead of a uniform
//! line pitch, and shortcuts match the physical key, so a non-Latin layout
//! copies and selects all the same. Selection ends are byte offsets, never
//! grapheme counts, so what is painted and what is copied are the same range.
//!
//! A press and drag selects by character, a double click a word and a triple
//! click a line. The keyboard follows an editor's caret map: Shift with an
//! arrow, Home or End extends the selection (Command to the line or document
//! edge, the jump modifier a word or a line), Escape clears it, and Cmd+C /
//! Cmd+A copy and select-all, whichever layout produced the key.

pub(crate) mod markdown;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use iced::advanced::graphics::text::Paragraph;
use iced::advanced::graphics::text::cosmic_text::{Affinity, BufferLine, Cursor, LayoutRun};
use iced::advanced::layout::Layout;
use iced::advanced::text::{Paragraph as _, Span};
use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::{Clipboard, Shell, Widget, clipboard, layout, mouse, renderer};
use iced::keyboard;
use iced::keyboard::key;
use iced::widget::text::{Alignment, LineHeight, Shaping, Wrapping};
use iced::{
    Color, Element, Event, Font, Length, Pixels, Point, Rectangle, Size, Vector, alignment,
};

use super::theme;
use super::widgets::{CLICK_DRAG_MAX_DISTANCE, DOUBLE_CLICK_WINDOW};
use crate::util::is_word_char;

// ── Style ────────────────────────────────────────────────────────

/// The colours of a [`Selectable`]: the text colour (inherited from the
/// renderer when `None`) and the selection highlight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Style {
    pub color: Option<Color>,
    pub selection: Color,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            color: None,
            selection: theme::ACCENT_DIM,
        }
    }
}

// ── Selection model ──────────────────────────────────────────────

/// The direction a selection was last extended in; the opposite end is the
/// anchor that stays put.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Direction {
    #[default]
    Right,
    Left,
}

/// One end of a selection: a logical line and a **byte** offset into it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SelectionEnd {
    line: usize,
    index: usize,
}

impl SelectionEnd {
    const fn new(line: usize, index: usize) -> Self {
        Self { line, index }
    }
}

/// A normalized text selection: `start` is never after `end`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Selection {
    start: SelectionEnd,
    end: SelectionEnd,
    direction: Direction,
}

impl Selection {
    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The end the next extension moves.
    fn active_end(&self) -> SelectionEnd {
        match self.direction {
            Direction::Right => self.end,
            Direction::Left => self.start,
        }
    }

    /// The end that stays put while the other moves.
    fn anchor(&self) -> SelectionEnd {
        match self.direction {
            Direction::Right => self.start,
            Direction::Left => self.end,
        }
    }

    fn select_range(&mut self, start: SelectionEnd, end: SelectionEnd) {
        self.start = start.min(end);
        self.end = start.max(end);
    }

    /// Move the active end to `new_end`, flipping the direction when it passes
    /// the anchor.
    fn change_selection(&mut self, new_end: SelectionEnd) {
        let (start, end) = match self.direction {
            Direction::Right => {
                if new_end < self.start {
                    self.direction = Direction::Left;
                }
                (self.start, new_end)
            }
            Direction::Left => {
                if new_end > self.end {
                    self.direction = Direction::Right;
                }
                (new_end, self.end)
            }
        };
        self.select_range(start, end);
    }

    fn select_all(&mut self, paragraph: &Paragraph) {
        let lines = &paragraph.buffer().lines;
        let Some(last) = lines.last() else {
            *self = Self::default();
            return;
        };
        self.direction = Direction::Right;
        self.select_range(
            SelectionEnd::new(0, 0),
            SelectionEnd::new(lines.len() - 1, last.text().len()),
        );
    }

    /// Select the whole logical `line`, endings excluded.
    fn select_line(&mut self, line: usize, paragraph: &Paragraph) {
        let Some(source) = paragraph.buffer().lines.get(line) else {
            return;
        };
        self.direction = Direction::Right;
        self.select_range(
            SelectionEnd::new(line, 0),
            SelectionEnd::new(line, source.text().len()),
        );
    }

    /// Select the word around `index` on `line`.
    fn select_word(&mut self, line: usize, index: usize, paragraph: &Paragraph) {
        let Some(source) = paragraph.buffer().lines.get(line) else {
            return;
        };
        let (start, end) = word_bounds(source.text(), index);
        self.direction = Direction::Right;
        self.select_range(SelectionEnd::new(line, start), SelectionEnd::new(line, end));
    }

    /// The selected text: per-line byte slices, line endings between lines. A
    /// selection end that cuts a character of several code points in half is
    /// moved out to the whole character, exactly as [`State::selection_quads`]
    /// paints it — the copy and the highlight read one span.
    fn text(&self, paragraph: &Paragraph) -> String {
        let lines = &paragraph.buffer().lines;
        let mut value = String::new();

        for line_index in self.start.line..=self.end.line {
            let Some(line) = lines.get(line_index) else {
                break;
            };
            let text = line.text();
            let from = if line_index == self.start.line {
                let index = clamp_char_boundary(text, self.start.index);
                split_character(paragraph, line_index, index).map_or(index, |(start, _)| start)
            } else {
                0
            };
            let to = if line_index == self.end.line {
                let index = clamp_char_boundary(text, self.end.index);
                split_character(paragraph, line_index, index).map_or(index, |(_, end)| end)
            } else {
                text.len()
            };

            if from < to {
                value.push_str(&text[from..to]);
            }
            if line_index != self.end.line {
                value.push_str(line.ending().as_str());
            }
        }

        value
    }
}

/// [`str::floor_char_boundary`] with the index clamped into the text: a stored
/// selection end can run past the line it was measured on.
fn clamp_char_boundary(text: &str, index: usize) -> usize {
    text.floor_char_boundary(index.min(text.len()))
}

/// The character of several code points — an emoji, a combining mark — that
/// `index` falls inside of, as its byte span. A selection end is moved out to
/// that span by both the paint and the copy, so neither can cut one in half.
fn character_span(run: &LayoutRun<'_>, index: usize) -> Option<(usize, usize)> {
    let index = clamp_char_boundary(run.text, index);
    run.glyphs
        .iter()
        .find(|glyph| glyph.start < index && index < glyph.end)
        .map(|glyph| (glyph.start, glyph.end))
}

/// [`character_span`] for a whole logical line, whatever run `index` landed in.
fn split_character(paragraph: &Paragraph, line: usize, index: usize) -> Option<(usize, usize)> {
    paragraph
        .buffer()
        .layout_runs()
        .filter(|run| run.line_i == line)
        .find_map(|run| character_span(&run, index))
}

/// Which run a double click grabs: word characters, whitespace, or the
/// punctuation in between.
fn char_class(c: char) -> u8 {
    if is_word_char(c) {
        0
    } else if c.is_whitespace() {
        1
    } else {
        2
    }
}

/// The byte range of the run at `index`: the class of the character there, or of
/// the one before it when the caret sits just past a run — so a press at the end
/// of a word still selects that word.
fn word_bounds(text: &str, index: usize) -> (usize, usize) {
    let index = clamp_char_boundary(text, index);
    let seed = text[index..]
        .chars()
        .next()
        .filter(|c| is_word_char(*c))
        .or_else(|| text[..index].chars().next_back());
    let Some(seed) = seed else {
        return (0, 0);
    };
    let seed = char_class(seed);

    let mut start = index;
    while let Some(c) = text[..start].chars().next_back() {
        if char_class(c) != seed {
            break;
        }
        start -= c.len_utf8();
    }

    let mut end = index;
    while let Some(c) = text[end..].chars().next() {
        if char_class(c) != seed {
            break;
        }
        end += c.len_utf8();
    }

    (start, end)
}

/// The selectable the user last pressed in. iced dispatches a key press to
/// every widget in the tree, so without this one Cmd+A would select every block
/// on the page and the following Cmd+C would copy whichever block came last.
static ACTIVE: AtomicU64 = AtomicU64::new(0);

/// Source of the per-widget identity [`ACTIVE`] is compared against.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Hand the text-selection shortcuts back: called when a focus-id editor takes
/// the keyboard (a press inside it, or the page focusing it), since that field
/// now owns copy, paste and select-all.
pub(crate) fn deactivate() {
    ACTIVE.store(0, Ordering::Relaxed);
}

// ── Click tracking ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClickKind {
    Single,
    Double,
    Triple,
}

impl ClickKind {
    /// A fourth consecutive press keeps acting like a triple click.
    const fn next(self) -> Self {
        match self {
            Self::Single => Self::Double,
            Self::Double | Self::Triple => Self::Triple,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Click {
    kind: ClickKind,
    position: Point,
    time: Instant,
}

impl Click {
    /// Classify a press, given the previous one.
    fn new(previous: Option<Self>, position: Point, time: Instant) -> Self {
        let kind = match previous {
            Some(previous) if previous.is_consecutive(position, time) => previous.kind.next(),
            _ => ClickKind::Single,
        };
        Self {
            kind,
            position,
            time,
        }
    }

    fn is_consecutive(&self, position: Point, time: Instant) -> bool {
        self.position.distance(position) <= CLICK_DRAG_MAX_DISTANCE
            && time.duration_since(self.time) <= DOUBLE_CLICK_WINDOW
    }
}

/// What a drag extends by, set by the click that started it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dragging {
    Character,
    Word,
    Line,
}

// ── Shortcuts ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shortcut {
    Copy,
    SelectAll,
}

/// The shortcut a key press stands for, matched on the **latin** character so a
/// Cyrillic (or any other) layout still copies.
fn shortcut(
    key: &keyboard::Key,
    physical_key: key::Physical,
    modifiers: keyboard::Modifiers,
) -> Option<Shortcut> {
    if !modifiers.command() {
        return None;
    }
    match key.to_latin(physical_key)? {
        'c' => Some(Shortcut::Copy),
        'a' => Some(Shortcut::SelectAll),
        _ => None,
    }
}

// ── Widget ───────────────────────────────────────────────────────

/// A read-only, selectable run of rich-text spans.
pub(crate) struct Selectable<'a, Link, Message, Theme = iced::Theme, Renderer = iced::Renderer>
where
    Link: Clone + 'static,
    Theme: 'a,
    Renderer: iced::advanced::text::Renderer<Font = Font>,
{
    spans: Box<dyn AsRef<[Span<'a, Link, Renderer::Font>]> + 'a>,
    size: Option<Pixels>,
    width: Length,
    font: Option<Font>,
    wrapping: Wrapping,
    style: Box<dyn Fn(&Theme) -> Style + 'a>,
    on_link_click: Option<Box<dyn Fn(Link) -> Message + 'a>>,
}

/// Build a [`Selectable`] from rich-text spans: a slice, or anything that hands
/// one over (`Arc<[Span]>` from the markdown parser, a `Vec`, an array). The
/// spans are borrowed for the widget's life, never copied per frame.
pub(crate) fn rich_text<'a, Link, Message, Theme, Renderer>(
    spans: impl AsRef<[Span<'a, Link, Renderer::Font>]> + 'a,
) -> Selectable<'a, Link, Message, Theme, Renderer>
where
    Link: Clone + 'static,
    Theme: 'a,
    Renderer: iced::advanced::text::Renderer<Font = Font>,
{
    Selectable::<Link, Message, Theme, Renderer> {
        spans: Box::new(spans),
        size: None,
        width: Length::Shrink,
        font: None,
        wrapping: Wrapping::default(),
        style: Box::new(|_theme| Style::default()),
        on_link_click: None,
    }
}

impl<'a, Link, Message, Theme, Renderer> Selectable<'a, Link, Message, Theme, Renderer>
where
    Link: Clone + 'static,
    Theme: 'a,
    Renderer: iced::advanced::text::Renderer<Font = Font>,
{
    pub(crate) fn size(mut self, size: impl Into<Pixels>) -> Self {
        self.size = Some(size.into());
        self
    }

    pub(crate) fn width(mut self, width: impl Into<Length>) -> Self {
        self.width = width.into();
        self
    }

    pub(crate) fn font(mut self, font: impl Into<Font>) -> Self {
        self.font = Some(font.into());
        self
    }

    /// Wrap at word or glyph level; word wrapping is the default.
    pub(crate) fn wrapping(mut self, wrapping: Wrapping) -> Self {
        self.wrapping = wrapping;
        self
    }

    pub(crate) fn style(mut self, style: impl Fn(&Theme) -> Style + 'a) -> Self {
        self.style = Box::new(style);
        self
    }

    pub(crate) fn on_link_click(mut self, on_link_click: impl Fn(Link) -> Message + 'a) -> Self {
        self.on_link_click = Some(Box::new(on_link_click));
        self
    }
}

impl<'a, Link, Message, Theme, Renderer> From<Selectable<'a, Link, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Link: Clone + 'static,
    Theme: 'a,
    Renderer: iced::advanced::text::Renderer<Font = Font, Paragraph = Paragraph> + 'a,
{
    fn from(widget: Selectable<'a, Link, Message, Theme, Renderer>) -> Self {
        Self::new(widget)
    }
}

// ── Widget state ─────────────────────────────────────────────────

struct State<Link> {
    /// This widget's identity in [`ACTIVE`].
    id: u64,
    /// The spans the paragraph was laid out from; `Paragraph::compare` cannot
    /// see content changes, so this decides when to rebuild it.
    spans: Vec<Span<'static, Link, Font>>,
    paragraph: Paragraph,
    span_pressed: Option<usize>,
    hovered_link: Option<usize>,
    selection: Selection,
    dragging: Option<Dragging>,
    /// The byte offset a vertical keyboard move began at, kept while it keeps
    /// passing shorter lines.
    goal: Option<usize>,
    last_click: Option<Click>,
    modifiers: keyboard::Modifiers,
}

/// The pointer in paragraph space: clamped into the widget bounds, whose
/// top-left corner is where the paragraph starts.
fn pointer(position: Point, bounds: Rectangle) -> Point {
    Point::new(
        position.x.clamp(bounds.x, bounds.x + bounds.width) - bounds.x,
        position.y.clamp(bounds.y, bounds.y + bounds.height) - bounds.y,
    )
}

impl<Link> State<Link> {
    /// The caret position nearest `position`.
    fn hit(&self, position: Point, bounds: Rectangle) -> Option<SelectionEnd> {
        let point = pointer(position, bounds);
        let cursor = self.paragraph.buffer().hit(point.x, point.y)?;

        Some(SelectionEnd::new(cursor.line, cursor.index))
    }

    /// One rectangle per selected visual line, from the real layout: wrapped
    /// lines, per-line heights and any scroll offset come out exactly.
    fn selection_quads(&self) -> Vec<Rectangle> {
        let Selection { start, end, .. } = self.selection;

        self.paragraph
            .buffer()
            .layout_runs()
            .filter(|run| (start.line..=end.line).contains(&run.line_i))
            .filter_map(|run| {
                let from = if run.line_i == start.line {
                    let index = clamp_char_boundary(run.text, start.index);
                    character_span(&run, index).map_or(index, |(start, _)| start)
                } else {
                    0
                };
                let to = if run.line_i == end.line {
                    let index = clamp_char_boundary(run.text, end.index);
                    character_span(&run, index).map_or(index, |(_, end)| end)
                } else {
                    run.text.len()
                };
                if from >= to {
                    return None;
                }

                let (x, width) = run.highlight(
                    Cursor::new_with_affinity(run.line_i, from, Affinity::After),
                    Cursor::new_with_affinity(run.line_i, to, Affinity::Before),
                )?;
                (width > 0.0).then(|| {
                    Rectangle::new(
                        Point::new(x, run.line_top),
                        Size::new(width, run.line_height),
                    )
                })
            })
            .collect()
    }
}

/// One character left of `end`, crossing to the previous line's end at the top
/// of a line.
fn char_left(end: SelectionEnd, lines: &[BufferLine]) -> SelectionEnd {
    if end.index > 0 {
        let text = lines.get(end.line).map_or("", BufferLine::text);
        SelectionEnd::new(end.line, clamp_char_boundary(text, end.index - 1))
    } else if end.line > 0 {
        SelectionEnd::new(end.line - 1, line_len(lines, end.line - 1))
    } else {
        end
    }
}

/// One character right of `end`, crossing to the next line's start at the end
/// of a line.
fn char_right(end: SelectionEnd, lines: &[BufferLine]) -> SelectionEnd {
    let text = lines.get(end.line).map_or("", BufferLine::text);
    if let Some(next) = text.get(end.index..).and_then(|rest| rest.chars().next()) {
        SelectionEnd::new(end.line, end.index + next.len_utf8())
    } else if end.line + 1 < lines.len() {
        SelectionEnd::new(end.line + 1, 0)
    } else {
        end
    }
}

/// The start of the word before `end` (or of `end`'s own word), crossing to the
/// previous line when the caret stands at the top of one.
fn word_left(end: SelectionEnd, lines: &[BufferLine]) -> SelectionEnd {
    let line = if end.index == 0 && end.line > 0 {
        end.line - 1
    } else {
        end.line
    };
    let text = lines.get(line).map_or("", BufferLine::text);
    let mut index = if line == end.line {
        end.index.min(text.len())
    } else {
        text.len()
    };

    while let Some((start, c)) = text[..index].char_indices().next_back() {
        if is_word_char(c) {
            break;
        }
        index = start;
    }
    while let Some((start, c)) = text[..index].char_indices().next_back() {
        if !is_word_char(c) {
            break;
        }
        index = start;
    }

    SelectionEnd::new(line, index)
}

/// The end of the word after `end` (or of `end`'s own word), crossing to the
/// next line when the caret stands at the end of one.
fn word_right(end: SelectionEnd, lines: &[BufferLine]) -> SelectionEnd {
    let line = if end.index >= line_len(lines, end.line) && end.line + 1 < lines.len() {
        end.line + 1
    } else {
        end.line
    };
    let text = lines.get(line).map_or("", BufferLine::text);
    let mut index = if line == end.line {
        end.index.min(text.len())
    } else {
        0
    };

    while let Some(c) = text[index..].chars().next() {
        if is_word_char(c) {
            break;
        }
        index += c.len_utf8();
    }
    while let Some(c) = text[index..].chars().next() {
        if !is_word_char(c) {
            break;
        }
        index += c.len_utf8();
    }

    SelectionEnd::new(line, index)
}

/// The byte length of logical `line`.
fn line_len(lines: &[BufferLine], line: usize) -> usize {
    lines.get(line).map_or(0, |line| line.text().len())
}

/// The word around `end` cut at the edge a word-wise drag moves toward: its
/// stop when the drag went forward, its start when it went back.
fn word_edge(end: SelectionEnd, anchor: SelectionEnd, paragraph: &Paragraph) -> SelectionEnd {
    let text = paragraph
        .buffer()
        .lines
        .get(end.line)
        .map_or("", |line| line.text());
    let (start, stop) = word_bounds(text, end.index);
    SelectionEnd::new(end.line, if end > anchor { stop } else { start })
}

/// Where a Shift+key press moves the selection end to, following an editor's
/// caret map: Command selects to the line (or the document) edge, the jump
/// modifier a word or a line, and otherwise one character or one logical line.
/// `goal` keeps the offset a vertical move began at, so a shorter line on the
/// way does not pull the following ones back with it; the module doc above
/// names the keys.
fn extended_to(
    key: &keyboard::Key,
    modifiers: keyboard::Modifiers,
    selection: Selection,
    paragraph: &Paragraph,
    goal: &mut Option<usize>,
) -> Option<SelectionEnd> {
    let keyboard::Key::Named(name) = key.as_ref() else {
        return None;
    };
    let end = selection.active_end();
    let lines = &paragraph.buffer().lines;
    let vertical = matches!(name, key::Named::ArrowUp | key::Named::ArrowDown)
        && !modifiers.macos_command()
        && !modifiers.jump();

    if vertical {
        let line = if name == key::Named::ArrowUp {
            end.line.saturating_sub(1)
        } else {
            (end.line + 1).min(lines.len().saturating_sub(1))
        };
        // The column can fall inside a character of a shorter line, so the
        // caret lands on the nearest boundary at or before it.
        let column = *goal.get_or_insert(end.index);
        let text = lines.get(line).map_or("", BufferLine::text);
        return Some(SelectionEnd::new(line, clamp_char_boundary(text, column)));
    }
    *goal = None;

    let line_start = SelectionEnd::new(end.line, 0);
    let line_end = SelectionEnd::new(end.line, line_len(lines, end.line));
    let document_end = SelectionEnd::new(
        lines.len().saturating_sub(1),
        line_len(lines, lines.len().saturating_sub(1)),
    );

    Some(match (name, modifiers.macos_command(), modifiers.jump()) {
        // The document edge: Command on Up/Down, the jump modifier on Home/End.
        (key::Named::ArrowUp, true, _) | (key::Named::Home, _, true) => SelectionEnd::default(),
        (key::Named::ArrowDown, true, _) | (key::Named::End, _, true) => document_end,
        // The line edge: Command on Left/Right, the jump modifier on Up/Down.
        (key::Named::ArrowLeft, true, _)
        | (key::Named::Home, _, false)
        | (key::Named::ArrowUp, false, true) => line_start,
        (key::Named::ArrowRight, true, _)
        | (key::Named::End, _, false)
        | (key::Named::ArrowDown, false, true) => line_end,
        // A word with the jump modifier, otherwise one character.
        (key::Named::ArrowLeft, false, true) => word_left(end, lines),
        (key::Named::ArrowRight, false, true) => word_right(end, lines),
        (key::Named::ArrowLeft, false, false) => char_left(end, lines),
        (key::Named::ArrowRight, false, false) => char_right(end, lines),
        _ => return None,
    })
}

// ── Widget implementation ────────────────────────────────────────

impl<Link, Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for Selectable<'_, Link, Message, Theme, Renderer>
where
    Link: Clone + 'static,
    Renderer: iced::advanced::text::Renderer<Font = Font, Paragraph = Paragraph>,
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State<Link>>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::<Link> {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            spans: Vec::new(),
            paragraph: Paragraph::default(),
            span_pressed: None,
            hovered_link: None,
            selection: Selection::default(),
            dragging: None,
            goal: None,
            last_click: None,
            modifiers: keyboard::Modifiers::default(),
        })
    }

    fn size(&self) -> Size<Length> {
        Size {
            width: self.width,
            height: Length::Shrink,
        }
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let state = tree.state.downcast_mut::<State<Link>>();
        let spans = self.spans.as_ref().as_ref();

        layout::sized(limits, self.width, Length::Shrink, |limits| {
            let bounds = limits.max();
            let size = self.size.unwrap_or_else(|| renderer.default_size());
            let font = self.font.unwrap_or_else(|| renderer.default_font());

            let with_spans = || iced::advanced::Text {
                content: spans,
                bounds,
                size,
                line_height: LineHeight::default(),
                font,
                align_x: Alignment::Default,
                align_y: alignment::Vertical::Top,
                shaping: Shaping::Advanced,
                wrapping: self.wrapping,
            };

            if state.spans != spans {
                state.paragraph = Paragraph::with_spans(with_spans());
                state.spans = spans.iter().cloned().map(Span::to_static).collect();
                // The offsets belong to the old text; a replacement message
                // must not inherit them.
                state.selection = Selection::default();
                state.dragging = None;
                state.goal = None;
            } else {
                match state.paragraph.compare(with_spans().with_content(())) {
                    iced::advanced::text::Difference::None => {}
                    iced::advanced::text::Difference::Bounds => state.paragraph.resize(bounds),
                    iced::advanced::text::Difference::Shape => {
                        state.paragraph = Paragraph::with_spans(with_spans());
                    }
                }
            }

            state.paragraph.min_bounds()
        })
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        defaults: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        if !bounds.intersects(viewport) {
            return;
        }

        let state = tree.state.downcast_ref::<State<Link>>();
        let style = (self.style)(theme);
        let translation = Vector::new(bounds.x, bounds.y);

        for (index, span) in self.spans.as_ref().as_ref().iter().enumerate() {
            let hovered_link = self.on_link_click.is_some() && state.hovered_link == Some(index);
            if span.highlight.is_none() && !span.underline && !span.strikethrough && !hovered_link {
                continue;
            }

            let regions = state.paragraph.span_bounds(index);

            if let Some(highlight) = span.highlight {
                for region in &regions {
                    let region = Rectangle::new(
                        region.position() - Vector::new(span.padding.left, span.padding.top),
                        region.size() + Size::new(span.padding.x(), span.padding.y()),
                    );
                    renderer.fill_quad(
                        renderer::Quad {
                            bounds: region + translation,
                            border: highlight.border,
                            ..renderer::Quad::default()
                        },
                        highlight.background,
                    );
                }
            }

            if span.underline || span.strikethrough || hovered_link {
                let size = span
                    .size
                    .or(self.size)
                    .unwrap_or_else(|| renderer.default_size());
                let line_height = span.line_height.unwrap_or_default().to_absolute(size);
                let color = span.color.or(style.color).unwrap_or(defaults.text_color);
                let baseline =
                    translation + Vector::new(0.0, size.0 + (line_height.0 - size.0) / 2.0);

                if span.underline || hovered_link {
                    let underline_at = baseline - Vector::new(0.0, size.0 * 0.08);
                    for region in &regions {
                        renderer.fill_quad(
                            renderer::Quad {
                                bounds: Rectangle::new(
                                    region.position() + underline_at,
                                    Size::new(region.width, 1.0),
                                ),
                                ..renderer::Quad::default()
                            },
                            color,
                        );
                    }
                }

                if span.strikethrough {
                    let strike_at = baseline - Vector::new(0.0, size.0 / 2.0);
                    for region in &regions {
                        renderer.fill_quad(
                            renderer::Quad {
                                bounds: Rectangle::new(
                                    region.position() + strike_at,
                                    Size::new(region.width, 1.0),
                                ),
                                ..renderer::Quad::default()
                            },
                            color,
                        );
                    }
                }
            }
        }

        if !state.selection.is_empty() {
            for quad in state.selection_quads() {
                if let Some(quad) = bounds.intersection(&(quad + translation)) {
                    renderer.fill_quad(
                        renderer::Quad {
                            bounds: quad,
                            ..renderer::Quad::default()
                        },
                        style.selection,
                    );
                }
            }
        }

        renderer.fill_paragraph(
            &state.paragraph,
            bounds.position(),
            style.color.unwrap_or(defaults.text_color),
            *viewport,
        );
    }

    // The event arms share one piece of interlocking state (the drag, the last
    // click and the pressed span); splitting them would only scatter the
    // transitions that connect them.
    #[expect(
        clippy::too_many_lines,
        reason = "one event dispatch table over the widget's interlocking input state"
    )]
    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<State<Link>>();
        let bounds = layout.bounds();

        if viewport.intersection(&bounds).is_none()
            && state.selection.is_empty()
            && state.dragging.is_none()
        {
            return;
        }

        let selection_before = state.selection;
        let hovered_before = state.hovered_link;

        state.hovered_link = cursor
            .position_over(bounds)
            .map(|position| pointer(position, bounds))
            .and_then(|point| state.paragraph.hit_span(point))
            .filter(|span| {
                self.spans
                    .as_ref()
                    .as_ref()
                    .get(*span)
                    .is_some_and(|span| span.link.is_some())
            });

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
            | Event::Touch(iced::touch::Event::FingerPressed { .. }) => {
                if state.hovered_link.is_some() {
                    state.span_pressed = state.hovered_link;
                    shell.capture_event();
                }

                if let Some(position) = cursor.position_over(bounds) {
                    ACTIVE.store(state.id, Ordering::Relaxed);
                    state.goal = None;
                    let click = Click::new(state.last_click, position, Instant::now());
                    state.last_click = Some(click);

                    if let Some(end) = state.hit(position, bounds) {
                        match click.kind {
                            ClickKind::Single => {
                                if state.modifiers.shift() {
                                    state.selection.change_selection(end);
                                } else {
                                    state.selection.select_range(end, end);
                                }
                                state.dragging = Some(Dragging::Character);
                            }
                            ClickKind::Double => {
                                state
                                    .selection
                                    .select_word(end.line, end.index, &state.paragraph);
                                state.dragging = Some(Dragging::Word);
                            }
                            ClickKind::Triple => {
                                state.selection.select_line(end.line, &state.paragraph);
                                state.dragging = Some(Dragging::Line);
                            }
                        }
                    }

                    shell.capture_event();
                } else {
                    state.selection = Selection::default();
                    state.dragging = None;
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
            | Event::Touch(
                iced::touch::Event::FingerLifted { .. } | iced::touch::Event::FingerLost { .. },
            ) => {
                state.dragging = None;

                let clicked_link = state
                    .span_pressed
                    .take()
                    .filter(|span| Some(*span) == state.hovered_link)
                    .filter(|_| state.selection.is_empty())
                    .filter(|_| {
                        !matches!(event, Event::Touch(iced::touch::Event::FingerLost { .. }))
                    });

                if let Some((link, on_link_click)) = clicked_link.and_then(|span| {
                    let span = self.spans.as_ref().as_ref().get(span)?;
                    Some((span.link.clone()?, self.on_link_click.as_deref()?))
                }) {
                    shell.publish(on_link_click(link));
                }
            }
            Event::Mouse(mouse::Event::CursorMoved { .. })
            | Event::Touch(iced::touch::Event::FingerMoved { .. }) => {
                if let Some(dragging) = state.dragging
                    && let Some(position) = cursor.land().position()
                    && let Some(end) = state.hit(position, bounds)
                {
                    let anchor = state.selection.anchor();
                    let target = match dragging {
                        Dragging::Character => end,
                        Dragging::Word => word_edge(end, anchor, &state.paragraph),
                        Dragging::Line => SelectionEnd::new(
                            end.line,
                            if end > anchor {
                                line_len(&state.paragraph.buffer().lines, end.line)
                            } else {
                                0
                            },
                        ),
                    };

                    state.selection.change_selection(target);
                }
            }
            Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                physical_key,
                modifiers,
                ..
            }) if ACTIVE.load(Ordering::Relaxed) == state.id => {
                match shortcut(key, *physical_key, *modifiers) {
                    Some(Shortcut::Copy) if !state.selection.is_empty() => {
                        let text = state.selection.text(&state.paragraph);
                        clipboard.write(clipboard::Kind::Standard, text);
                        shell.capture_event();
                    }
                    Some(Shortcut::SelectAll) => {
                        state.selection.select_all(&state.paragraph);
                        state.goal = None;
                        shell.capture_event();
                    }
                    _ if matches!(key.as_ref(), keyboard::Key::Named(key::Named::Escape)) => {
                        // Escape drops what was dragged over, so a following
                        // copy finds nothing.
                        if !state.selection.is_empty() || state.dragging.is_some() {
                            state.dragging = None;
                            state.selection = Selection::default();
                            state.goal = None;
                            shell.capture_event();
                        }
                    }
                    _ => {
                        if modifiers.shift()
                            && let Some(end) = extended_to(
                                key,
                                *modifiers,
                                state.selection,
                                &state.paragraph,
                                &mut state.goal,
                            )
                        {
                            state.selection.change_selection(end);
                            shell.capture_event();
                        }
                    }
                }
            }
            Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                state.modifiers = *modifiers;
            }
            _ => {}
        }

        if state.selection != selection_before || state.hovered_link != hovered_before {
            shell.request_redraw();
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let state = tree.state.downcast_ref::<State<Link>>();

        if state.hovered_link.is_some() {
            mouse::Interaction::Pointer
        } else if cursor.is_over(layout.bounds()) || state.dragging.is_some() {
            mouse::Interaction::Text
        } else {
            mouse::Interaction::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Shortcut, shortcut};
    use iced::keyboard::key::{Code, Physical};
    use iced::keyboard::{Key, Modifiers};

    #[test]
    fn command_c_with_a_cyrillic_key_character_still_copies() {
        assert_eq!(
            shortcut(
                &Key::Character("с".into()),
                Physical::Code(Code::KeyC),
                Modifiers::COMMAND
            ),
            Some(Shortcut::Copy)
        );
    }

    #[test]
    fn command_a_with_a_cyrillic_key_character_still_selects_everything() {
        assert_eq!(
            shortcut(
                &Key::Character("а".into()),
                Physical::Code(Code::KeyA),
                Modifiers::COMMAND
            ),
            Some(Shortcut::SelectAll)
        );
    }
}
