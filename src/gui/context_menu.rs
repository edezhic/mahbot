//! A reusable context menu widget built on Iced's overlay system.
//!
//! Context menus rendered in the normal widget tree get clipped by parent
//! containers. This widget uses [`Widget::overlay`] and the [`Overlay`]
//! trait to render the menu above the entire widget tree, avoiding clipping.
//!
//! ## Usage
//!
//! ```ignore
//! ContextMenu::new(
//!     underlay_element,
//!     vec![
//!         ("Close".into(), MyMessage::Close),
//!         ("Close Others".into(), MyMessage::CloseOthers),
//!     ],
//! )
//! ```
//!
//! Right-clicking the underlay opens the menu at the cursor position.
//! Clicking a menu item fires its action and dismisses the menu.
//! Clicking outside or pressing Escape also dismisses the menu.
//!
//! [`Widget::overlay`]: iced::advanced::widget::Widget::overlay
//! [`Overlay`]: iced::advanced::overlay::Overlay

// Menu item counts are small (single digits to low double digits) — f32
// precision loss on usize→f32 casts is not a concern for pixel layout.
// f32→usize casts in update() are guarded by `rel_y >= 0.0` and `idx < len()` checks.

use iced::advanced::layout::{self, Layout};
use iced::advanced::overlay;
use iced::advanced::renderer;
use iced::advanced::text::{self, Paragraph};
use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::widget::{self, Widget};
use iced::advanced::{Clipboard, Shell};
use iced::mouse;
use iced::{Color, Element, Event, Length, Pixels, Point, Rectangle, Size, Vector, alignment};

use super::theme;

// ── Widget ───────────────────────────────────────────────────────────

/// A widget that wraps an underlay element and shows a context menu
/// overlay on right-click.
pub struct ContextMenu<'a, Message, Theme = iced::Theme, Renderer = iced::Renderer>
where
    Message: Clone + 'a,
{
    underlay: Element<'a, Message, Theme, Renderer>,
    menu_items: Vec<(String, Message)>,
}

impl<'a, Message, Theme, Renderer> ContextMenu<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
{
    /// Creates a new [`ContextMenu`] widget.
    ///
    /// `underlay` — the widget that responds to right-click to open the menu.
    /// `menu_items` — label/action pairs for menu items.
    #[must_use]
    pub fn new(
        underlay: impl Into<Element<'a, Message, Theme, Renderer>>,
        menu_items: Vec<(String, Message)>,
    ) -> Self {
        Self {
            underlay: underlay.into(),
            menu_items,
        }
    }
}

// ── State ────────────────────────────────────────────────────────────

/// Widget state for [`ContextMenu`], stored in the widget tree.
#[derive(Debug, Clone)]
struct ContextMenuState {
    show: bool,
    position: Point,
    /// Currently hovered menu item index, persisted across frames
    /// so the highlight remains visible when the cursor is stationary.
    hovered: Option<usize>,
}

impl ContextMenuState {
    const fn new() -> Self {
        Self {
            show: false,
            position: Point::ORIGIN,
            hovered: None,
        }
    }
}

// ── Widget impl ──────────────────────────────────────────────────────

impl<'a, Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for ContextMenu<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
    Renderer: iced::advanced::Renderer + text::Renderer,
{
    fn size(&self) -> Size<Length> {
        self.underlay.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.underlay.as_widget().size_hint()
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<ContextMenuState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(ContextMenuState::new())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.underlay)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(&[&self.underlay]);
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.underlay
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.underlay.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            cursor,
            viewport,
        );
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn widget::Operation,
    ) {
        self.underlay
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<ContextMenuState>();

        // Always forward event to underlay first, so child widgets (including
        // nested ContextMenus) process the event before we decide to capture it.
        // This allows an outer ContextMenu to act as a fallback for empty-space
        // right-clicks without overriding inner node-level ContextMenus.
        self.underlay.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );

        // Only open the menu if no child already captured the event.
        if shell.is_event_captured() {
            return;
        }

        // Check for right-click to open the menu.
        if let Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) = event {
            if let Some(pos) = cursor.position_over(layout.bounds()) {
                state.show = true;
                state.position = pos;
                state.hovered = None;
                shell.request_redraw();
                shell.capture_event();
            }
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.underlay.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        )
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        // Always forward to the underlay's overlay so nested overlays
        // (e.g., inner ContextMenus in a file tree) are rendered.
        // The underlay's tree is tree.children[0], and ContextMenu's
        // layout node IS the underlay's layout node (layout() delegates).
        let underlay_overlay = self.underlay.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        );

        let position = {
            let state = tree.state.downcast_ref::<ContextMenuState>();
            if !state.show {
                // Not showing our menu — just return underlay's overlays.
                return underlay_overlay;
            }
            // `position` was captured in content coordinates (cursor is adjusted
            // by the parent scrollable's offset). `translation` is the inverse of
            // the scroll offset. Adding them converts to viewport coordinates,
            // which is the space the overlay renders in.
            state.position + translation
        };

        let state = tree.state.downcast_mut::<ContextMenuState>();

        let own_overlay = overlay::Element::new(Box::new(ContextMenuOverlay {
            show: &mut state.show,
            hovered: &mut state.hovered,
            position,
            menu_items: &self.menu_items,
        }));

        // Combine our overlay with any underlay overlays in a Group.
        // The context menu overlay uses index 2.0 so it renders on top
        // of inner overlays when both are visible simultaneously.
        let mut overlays = Vec::new();
        if let Some(underlay) = underlay_overlay {
            overlays.push(underlay);
        }
        overlays.push(own_overlay);

        Some(overlay::Group::with_children(overlays).overlay())
    }
}

// ── From impl ────────────────────────────────────────────────────────

impl<'a, Message, Theme, Renderer> From<ContextMenu<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
    Theme: 'a,
    Renderer: iced::advanced::Renderer + text::Renderer + 'a,
{
    fn from(context_menu: ContextMenu<'a, Message, Theme, Renderer>) -> Self {
        Self::new(context_menu)
    }
}

// ── Overlay ──────────────────────────────────────────────────────────

/// The overlay that renders the context menu.
struct ContextMenuOverlay<'a, 'b, Message>
where
    Message: Clone,
{
    show: &'b mut bool,
    hovered: &'b mut Option<usize>,
    position: Point,
    menu_items: &'a [(String, Message)],
}

// Layout constants for the menu.
const MENU_ITEM_HEIGHT: f32 = 28.0;
const MENU_PADDING: f32 = 8.0;
const MENU_MIN_WIDTH: f32 = 140.0;
const MENU_FONT_SIZE: f32 = 14.0;

/// Convert a y-offset (relative to menu origin, minus padding) into a
/// menu item index.
/// Returns None if the offset is negative or beyond the last item.
fn item_index_from_y(rel_y: f32, item_count: usize) -> Option<usize> {
    if rel_y < 0.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = (rel_y / MENU_ITEM_HEIGHT) as usize;
    if idx < item_count { Some(idx) } else { None }
}

impl<Message, Theme, Renderer> overlay::Overlay<Message, Theme, Renderer>
    for ContextMenuOverlay<'_, '_, Message>
where
    Message: Clone,
    Renderer: text::Renderer,
{
    fn layout(&mut self, renderer: &Renderer, bounds: Size) -> layout::Node {
        let item_count = self.menu_items.len();
        #[allow(clippy::cast_precision_loss)]
        let menu_height = item_count as f32 * MENU_ITEM_HEIGHT + MENU_PADDING * 2.0;

        // Compute the widest label using the renderer's text measurement.
        let text_size = Pixels(MENU_FONT_SIZE);
        let max_label_width: f32 = self
            .menu_items
            .iter()
            .map(|(label, _)| {
                let paragraph = Renderer::Paragraph::with_text(text::Text {
                    content: label.as_str(),
                    bounds: Size::new(f32::MAX, MENU_ITEM_HEIGHT),
                    size: text_size,
                    line_height: text::LineHeight::Relative(1.3),
                    font: renderer.default_font(),
                    align_x: text::Alignment::Left,
                    align_y: alignment::Vertical::Top,
                    shaping: text::Shaping::Advanced,
                    wrapping: text::Wrapping::default(),
                });
                paragraph.min_bounds().width
            })
            .fold(0.0_f32, f32::max);
        let menu_width = (max_label_width + MENU_PADDING * 2.0).max(MENU_MIN_WIDTH);

        // Edge clipping: flip left/up if the menu would overflow bounds.
        let mut x = self.position.x;
        let mut y = self.position.y;

        if x + menu_width > bounds.width {
            x = (self.position.x - menu_width).max(0.0);
        }
        if y + menu_height > bounds.height {
            y = (self.position.y - menu_height).max(0.0);
        }

        layout::Node::new(Size::new(menu_width, menu_height)).move_to(Point::new(x, y))
    }

    fn draw(
        &self,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
    ) {
        let bounds = layout.bounds();

        // Draw menu background.
        renderer.fill_quad(
            renderer::Quad {
                bounds,
                border: iced::Border {
                    radius: 4.0.into(),
                    width: 1.0,
                    color: theme::BORDER_STRONG,
                },
                ..renderer::Quad::default()
            },
            theme::BG_ELEVATED,
        );

        // Draw each menu item.
        let font = renderer.default_font();
        let text_size = Pixels(MENU_FONT_SIZE);

        for (i, (label, _action)) in self.menu_items.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let item_y = bounds.y + MENU_PADDING + i as f32 * MENU_ITEM_HEIGHT;
            let item_bounds = Rectangle {
                x: bounds.x,
                y: item_y,
                width: bounds.width,
                height: MENU_ITEM_HEIGHT,
            };

            // Hover highlight.
            if *self.hovered == Some(i) {
                renderer.fill_quad(
                    renderer::Quad {
                        bounds: item_bounds,
                        border: iced::Border {
                            radius: 0.0.into(),
                            width: 0.0,
                            color: Color::TRANSPARENT,
                        },
                        ..renderer::Quad::default()
                    },
                    theme::HOVER,
                );
            }

            // Draw label text using fill_text.
            let text_color = if *self.hovered == Some(i) {
                theme::TEXT_PRIMARY
            } else {
                theme::TEXT_SECONDARY
            };

            let text = text::Text {
                content: label.clone(),
                bounds: Size::new(bounds.width - MENU_PADDING * 2.0, MENU_ITEM_HEIGHT),
                size: text_size,
                line_height: text::LineHeight::Relative(1.3),
                font,
                align_x: text::Alignment::Left,
                align_y: alignment::Vertical::Center,
                shaping: text::Shaping::Advanced,
                wrapping: text::Wrapping::default(),
            };

            renderer.fill_text(
                text,
                Point::new(item_bounds.x + MENU_PADDING, item_bounds.center_y()),
                text_color,
                item_bounds,
            );
        }
    }

    fn update(
        &mut self,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
    ) {
        let bounds = layout.bounds();

        match event {
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                // Update hovered item index.
                *self.hovered = cursor
                    .position_in(bounds)
                    .and_then(|pos| item_index_from_y(pos.y - MENU_PADDING, self.menu_items.len()));
                shell.request_redraw();
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(cursor_pos) = cursor.position() {
                    if bounds.contains(cursor_pos) {
                        if let Some(idx) = item_index_from_y(
                            cursor_pos.y - bounds.y - MENU_PADDING,
                            self.menu_items.len(),
                        ) {
                            // Fire the action and dismiss.
                            let action = self.menu_items[idx].1.clone();
                            *self.show = false;
                            shell.publish(action);
                            shell.capture_event();
                            return;
                        }
                    }
                }
                // Click outside — dismiss.
                *self.show = false;
                shell.request_redraw();
                shell.capture_event();
            }
            Event::Keyboard(iced::keyboard::Event::KeyPressed {
                key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape),
                ..
            }) => {
                *self.show = false;
                shell.request_redraw();
                shell.capture_event();
            }
            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let bounds = layout.bounds();
        if cursor.is_over(bounds) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::Idle
        }
    }

    /// Higher index ensures the context menu renders on top of any
    /// inner overlays (e.g., nested ContextMenus in a file tree)
    /// when both are visible simultaneously.
    fn index(&self) -> f32 {
        2.0
    }
}
