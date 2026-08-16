//! Role-switch dropdown rendered as an overlay, not an in-flow panel.
//!
//! The Home chat composer used to render the role list as a plain column
//! between the chat area and the composer, so opening it pushed the chat
//! layout down and left an empty strip beside the list. This widget uses
//! [`Widget::overlay`] and the [`Overlay`] trait — the same mechanism as
//! [`super::context_menu::ContextMenu`] — to float the list above the whole
//! widget tree, anchored to the role button (right-aligned) with no layout
//! impact on the composer.
//!
//! ## Interaction model
//!
//! * The wrapped underlay is the composer's role button; its own
//!   `on_press` toggles the menu open (the parent owns the open/closed
//!   flag via the `show` parameter).
//! * Clicking a non-current role fires its action and closes the menu.
//!   The current role is rendered disabled (no hover, checkmark) and its
//!   click is consumed without closing — matching the old in-flow panel.
//! * Clicking anywhere outside the popup (left *or* right button) or
//!   pressing Escape closes it via the `on_close` message. Outside clicks
//!   are captured, so the click does not leak to the widgets underneath
//!   (e.g. clicking the mic button while the menu is open closes the menu
//!   without starting a recording; clicking the role button again closes
//!   instead of toggling — standard popup behavior).
//! * Clicks on the popup's own padding or inter-item gaps are consumed
//!   without closing (only genuine outside clicks dismiss), matching the
//!   old in-flow panel — unlike ContextMenu, which dismisses on any click.
//! * Selecting a role publishes only `SwitchRole`; the Dashboard intercept
//!   answers with `RoleMenuClosed` after persisting (see `gui/mod.rs`), so
//!   the menu closes on success and would stay open on a (unreachable)
//!   rejection — exactly like the old in-flow panel.
//! * Escape also reaches the global `keyboard::listen` subscription (the
//!   overlay capture does not suppress it); on Home that handler is a
//!   no-op, matching the existing chat context-menu behavior.
//!
//! The overlay sets its [`Overlay::index`] above `ContextMenu`'s `2.0` so
//! the role list stays on top when both are visible.
//!
//! [`Widget::overlay`]: iced::advanced::widget::Widget::overlay
//! [`Overlay`]: iced::advanced::overlay::Overlay

// Role pools are tiny (single digits) — f32 precision loss on usize→f32
// casts is not a concern for pixel layout. f32→usize casts in update()
// are guarded by `rel_y >= 0.0` and `idx < len()` checks.

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

// ── Menu item ─────────────────────────────────────────────────────

/// A single role entry in the role-switch dropdown.
#[derive(Debug, Clone)]
pub struct RoleMenuItem<Message> {
    /// The role to display (drives the icon glyph, label, and colors).
    pub role: crate::Role,
    /// Whether this is the currently active role: rendered with a
    /// checkmark, without hover, and with `action == None`.
    pub is_current: bool,
    /// Action to fire when clicked. `None` for the current role (its click
    /// is consumed without closing, matching the old in-flow panel).
    pub action: Option<Message>,
}

impl<Message> RoleMenuItem<Message> {
    /// Create a role menu entry.
    #[must_use]
    pub fn new(role: crate::Role, is_current: bool, action: Option<Message>) -> Self {
        Self {
            role,
            is_current,
            action,
        }
    }
}

// ── Widget ───────────────────────────────────────────────────────────

/// Wraps the composer's role button and renders the role-switch dropdown
/// as an overlay anchored to that button.
pub struct RoleMenu<'a, Message, Theme = iced::Theme, Renderer = iced::Renderer>
where
    Message: Clone + 'a,
{
    underlay: Element<'a, Message, Theme, Renderer>,
    items: Vec<RoleMenuItem<Message>>,
    /// Whether the dropdown is open. Owned by the parent (Home state) so
    /// the close path (outside click / Escape / role selection) is a plain
    /// message round-trip with a single source of truth.
    show: bool,
    /// Message published when the popup dismisses itself (outside click or
    /// Escape) so the parent can flip `show` back to false.
    on_close: Message,
}

impl<'a, Message, Theme, Renderer> RoleMenu<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
{
    /// Creates a new [`RoleMenu`] widget.
    ///
    /// `underlay` — the role button the popup anchors to (right-aligned,
    /// opening above the button with a below-flip fallback).
    /// `items` — the role entries; the current role must carry
    /// `is_current == true` and `action == None`.
    /// `show` — whether the dropdown is currently open.
    /// `on_close` — message fired when the popup dismisses itself.
    #[must_use]
    pub fn new(
        underlay: impl Into<Element<'a, Message, Theme, Renderer>>,
        items: Vec<RoleMenuItem<Message>>,
        show: bool,
        on_close: Message,
    ) -> Self {
        Self {
            underlay: underlay.into(),
            items,
            show,
            on_close,
        }
    }
}

// ── State ────────────────────────────────────────────────────────────

/// Widget state for [`RoleMenu`], stored in the widget tree.
///
/// `hovered` persists across frames so the highlight remains visible when
/// the cursor is stationary. `show` mirrors the widget parameter so a
/// freshly-opened menu starts with no stale hover highlight.
#[derive(Debug, Clone)]
struct RoleMenuState {
    show: bool,
    hovered: Option<usize>,
}

impl RoleMenuState {
    const fn new() -> Self {
        Self {
            show: false,
            hovered: None,
        }
    }
}

// ── Widget impl ──────────────────────────────────────────────────────

impl<'a, Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for RoleMenu<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
    Renderer: iced::advanced::Renderer + text::Renderer<Font = iced::Font>,
{
    fn size(&self) -> Size<Length> {
        self.underlay.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.underlay.as_widget().size_hint()
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<RoleMenuState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(RoleMenuState::new())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.underlay)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(&[&self.underlay]);

        let state = tree.state.downcast_mut::<RoleMenuState>();
        // A fresh open must not inherit a stale hover highlight from the
        // previous open (the cursor may rest where an item used to be).
        if self.show && !state.show {
            state.hovered = None;
        }
        state.show = self.show;
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
        // The underlay is the role button; it handles its own clicks
        // (RoleMenuToggled). When the popup is open its overlay captures
        // outside clicks, so the button only toggles when the menu is
        // closed — no toggle-clash on the anchor.
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
        let underlay_overlay = self.underlay.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        );

        let state = tree.state.downcast_ref::<RoleMenuState>();
        if !state.show {
            return underlay_overlay;
        }

        // The anchor is the role button's bounds in content coordinates;
        // `translation` is the inverse scroll offset, so adding them yields
        // viewport coordinates — the space overlays render in.
        let anchor = layout.bounds() + translation;

        let state = tree.state.downcast_mut::<RoleMenuState>();
        let own_overlay = overlay::Element::new(Box::new(RoleMenuOverlay {
            hovered: &mut state.hovered,
            anchor,
            items: &self.items,
            on_close: &self.on_close,
            label_widths: Vec::new(),
        }));

        // Combine with any underlay overlays in a Group. The role menu uses
        // index 3.0 — above the chat ContextMenu's 2.0 — so it stays on top
        // when both are open.
        let mut overlays = Vec::new();
        if let Some(underlay) = underlay_overlay {
            overlays.push(underlay);
        }
        overlays.push(own_overlay);

        Some(overlay::Group::with_children(overlays).overlay())
    }
}

// ── From impl ────────────────────────────────────────────────────────

impl<'a, Message, Theme, Renderer> From<RoleMenu<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
    Theme: 'a,
    Renderer: iced::advanced::Renderer + text::Renderer<Font = iced::Font> + 'a,
{
    fn from(role_menu: RoleMenu<'a, Message, Theme, Renderer>) -> Self {
        Self::new(role_menu)
    }
}

// ── Overlay ──────────────────────────────────────────────────────────

/// The overlay that renders the role-switch dropdown.
struct RoleMenuOverlay<'a, 'b, Message>
where
    Message: Clone,
{
    hovered: &'b mut Option<usize>,
    /// Anchor bounds (the role button) in viewport coordinates.
    anchor: Rectangle,
    items: &'a [RoleMenuItem<Message>],
    on_close: &'a Message,
    /// Measured label widths, computed in [`Overlay::layout`] so `draw`
    /// can place the checkmark right after the label (matching the old
    /// in-flow `row![icon, label, check]` layout).
    label_widths: Vec<f32>,
}

// Popup geometry — mirrors the old in-flow panel: fixed 170px width,
// 5px panel padding, 2px item spacing, no scroll cap.
const MENU_WIDTH: f32 = 170.0;
const MENU_PADDING: f32 = 5.0;
const ITEM_HEIGHT: f32 = 28.0;
const ITEM_SPACING: f32 = 2.0;
/// Button-internal padding of the old in-flow menu items.
const BTN_PADDING: f32 = 5.0;
/// Gap between the popup and its anchor button (the old panel sat flush
/// against the composer; the floating popup needs a little air).
const ANCHOR_GAP: f32 = 6.0;
const ICON_SIZE: f32 = 13.0;
const CHECK_SIZE: f32 = 12.0;
const LABEL_FONT_SIZE: f32 = 13.0;
/// 6px spacing between the icon/label/check slots (old in-flow row spacing).
const ITEM_GAP: f32 = 6.0;

/// Convert a y-offset (relative to the menu origin, minus panel padding)
/// into a menu item index. Returns None for negative offsets, for the
/// spacing gap between items, or past the last item.
fn item_index_from_y(rel_y: f32, item_count: usize) -> Option<usize> {
    if rel_y < 0.0 {
        return None;
    }
    let slot = ITEM_HEIGHT + ITEM_SPACING;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = (rel_y / slot) as usize;
    if idx >= item_count {
        return None;
    }
    // Only the ITEM_HEIGHT-tall portion of each slot is a hit target; the
    // trailing ITEM_SPACING gap between items must not map to the item
    // above it (no hover highlight / action on dead space).
    #[allow(clippy::cast_precision_loss)]
    let within_item = rel_y - idx as f32 * slot < ITEM_HEIGHT;
    within_item.then_some(idx)
}

/// Compute the popup's top-left position (viewport coordinates).
///
/// Right-aligns the popup to the anchor (the role button), opens it above
/// the anchor with a small gap, flips below when there is not enough room
/// above, and finally clamps inside the viewport. Pure function — unit
/// tested below.
fn popup_position(anchor: Rectangle, menu_size: Size, viewport: Size, gap: f32) -> Point {
    // Right-align to the anchor, clamped to the viewport (narrow-window
    // safety; ContextMenuOverlay precedent).
    let x = (anchor.x + anchor.width - menu_size.width)
        .clamp(0.0, (viewport.width - menu_size.width).max(0.0));

    // Prefer opening above the anchor; flip below on overflow.
    let above = anchor.y - menu_size.height - gap;
    let y = if above >= 0.0 {
        above
    } else {
        let below = anchor.y + anchor.height + gap;
        if below + menu_size.height <= viewport.height {
            below
        } else {
            // Neither side fits — clamp inside the viewport as a last
            // resort: bottom-aligned when the menu fits vertically,
            // top-aligned (0.0) only when it is taller than the viewport.
            (viewport.height - menu_size.height).max(0.0)
        }
    };

    Point::new(x, y)
}

/// Popup height for `item_count` entries: panel padding on both sides plus
/// item slots separated by `ITEM_SPACING`.
fn menu_height(item_count: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let items = item_count as f32;
    #[allow(clippy::cast_precision_loss)]
    let gaps = item_count.saturating_sub(1) as f32;
    MENU_PADDING * 2.0 + items * ITEM_HEIGHT + gaps * ITEM_SPACING
}

impl<Message, Theme, Renderer> overlay::Overlay<Message, Theme, Renderer>
    for RoleMenuOverlay<'_, '_, Message>
where
    Message: Clone,
    Renderer: text::Renderer<Font = iced::Font>,
{
    fn layout(&mut self, renderer: &Renderer, bounds: Size) -> layout::Node {
        let menu_height = menu_height(self.items.len());

        // Measure each label so the checkmark can be placed right after it
        // (the old in-flow row was `row![icon, label, check].spacing(6)`).
        let text_size = Pixels(LABEL_FONT_SIZE);
        self.label_widths = self
            .items
            .iter()
            .map(|item| {
                let label = crate::role::role_info(&item.role).display_label;
                let paragraph = Renderer::Paragraph::with_text(text::Text {
                    content: label,
                    bounds: Size::new(f32::MAX, ITEM_HEIGHT),
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
            .collect();

        let menu_size = Size::new(MENU_WIDTH, menu_height);
        let position = popup_position(self.anchor, menu_size, bounds, ANCHOR_GAP);
        layout::Node::new(menu_size).move_to(position)
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

        // Popup background + border (elevated surface, like ContextMenu).
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

        let slot = ITEM_HEIGHT + ITEM_SPACING;
        for (i, item) in self.items.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let item_y = bounds.y + MENU_PADDING + i as f32 * slot;
            let item_bounds = Rectangle {
                x: bounds.x,
                y: item_y,
                width: bounds.width,
                height: ITEM_HEIGHT,
            };
            #[allow(clippy::cast_precision_loss)]
            let label_width = self.label_widths.get(i).copied().unwrap_or_default();
            draw_item(
                renderer,
                item,
                item_bounds,
                label_width,
                *self.hovered == Some(i),
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
                // Update hovered item index. `cursor.position_in(bounds)`
                // is already relative to the popup bounds (iced_core
                // subtracts the origin); subtracting `bounds.y` again
                // would make `rel_y` negative for menus opened above the
                // anchor and hover would never activate. ContextMenu's
                // overlay does the same (`pos.y - MENU_PADDING`).
                *self.hovered = cursor
                    .position_in(bounds)
                    .and_then(|pos| item_index_from_y(pos.y - MENU_PADDING, self.items.len()));
                shell.request_redraw();
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(cursor_pos) = cursor.position() {
                    if bounds.contains(cursor_pos) {
                        if let Some(idx) = item_index_from_y(
                            cursor_pos.y - bounds.y - MENU_PADDING,
                            self.items.len(),
                        ) {
                            if let Some(action) = self.items[idx].action.clone() {
                                // Do NOT publish on_close here: the action
                                // is always SwitchRole, which the Dashboard
                                // intercepts and answers with RoleMenuClosed
                                // after persisting (see gui/mod.rs). That
                                // single close restores the old in-flow
                                // panel's semantics — a (currently
                                // unreachable) intercept rejection would
                                // leave the menu open instead of closing it.
                                shell.publish(action);
                                shell.capture_event();
                                return;
                            }
                            // Current role: consume the click but keep the
                            // menu open (matches the disabled button of the
                            // old in-flow panel).
                            shell.capture_event();
                            return;
                        }
                        // Click on the popup's own padding or an inter-item
                        // gap: consume it but keep the menu open (the old
                        // in-flow panel did nothing on those clicks; only
                        // genuine outside clicks dismiss).
                        shell.capture_event();
                        return;
                    }
                }
                // Click outside — dismiss (and swallow the click so the
                // base tree does not also react, e.g. no toggle-clash on
                // the anchor button, no accidental mic trigger).
                shell.publish(self.on_close.clone());
                shell.request_redraw();
                shell.capture_event();
            }
            // Right-click anywhere or Escape dismisses too — the right-click
            // capture also prevents the chat context menu (which has its own
            // role-switcher section) from stacking on top of the open list.
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right))
            | Event::Keyboard(iced::keyboard::Event::KeyPressed {
                key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape),
                ..
            }) => {
                shell.publish(self.on_close.clone());
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
        if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::Idle
        }
    }

    /// Above the chat ContextMenu's `2.0` so the role list renders on top
    /// of the chat right-click menu when both are open.
    ///
    /// Keep this in sync with `ContextMenu::index` (context_menu.rs) — the
    /// 3.0-vs-2.0 layering is a silent coupling; a third overlay should
    /// move popup layering into a shared constant.
    fn index(&self) -> f32 {
        3.0
    }
}

/// Render one menu row: optional hover highlight, role icon (lucide
/// glyph), label, and the current-role checkmark.
///
/// Slots match the old in-flow `row![icon, label, check].spacing(6)`
/// layout: icon at the left padding, checkmark right after the label.
///
/// A free function (not a method) so it does not depend on `&self`.
fn draw_item<Message, Renderer>(
    renderer: &mut Renderer,
    item: &RoleMenuItem<Message>,
    item_bounds: Rectangle,
    label_width: f32,
    is_hovered: bool,
) where
    Renderer: text::Renderer<Font = iced::Font>,
{
    // Hover highlight only for switchable (non-current) roles.
    if !item.is_current && is_hovered {
        renderer.fill_quad(
            renderer::Quad {
                bounds: item_bounds,
                border: iced::Border {
                    radius: 4.0.into(),
                    width: 0.0,
                    color: Color::TRANSPARENT,
                },
                ..renderer::Quad::default()
            },
            theme::HOVER,
        );
    }

    let (icon_color, _) = theme::role_badge_color_for(&item.role);
    let label_color = if item.is_current {
        theme::TEXT_PRIMARY
    } else {
        theme::TEXT_SECONDARY
    };

    let icon_x = item_bounds.x + BTN_PADDING;
    let label_x = icon_x + ICON_SIZE + ITEM_GAP;
    let check_x = label_x + label_width + ITEM_GAP;
    let item_center_y = item_bounds.center_y();

    // Role icon (lucide glyph rendered manually — an Overlay cannot
    // contain child widgets).
    renderer.fill_text(
        text::Text {
            content: role_icon_glyph(item.role).to_string(),
            bounds: Size::new(ICON_SIZE, ITEM_HEIGHT),
            size: Pixels(ICON_SIZE),
            line_height: text::LineHeight::Relative(1.3),
            font: iced_fonts::LUCIDE_FONT,
            align_x: text::Alignment::Left,
            align_y: alignment::Vertical::Center,
            shaping: text::Shaping::Basic,
            wrapping: text::Wrapping::default(),
        },
        Point::new(icon_x, item_center_y),
        icon_color,
        item_bounds,
    );

    // Role label. Available width: panel + button padding on both ends
    // minus the icon slot, the two gaps, and the check slot.
    let label_bounds_width =
        item_bounds.width - BTN_PADDING * 2.0 - ICON_SIZE - ITEM_GAP * 2.0 - CHECK_SIZE;
    renderer.fill_text(
        text::Text {
            content: crate::role::role_info(&item.role).display_label.to_string(),
            bounds: Size::new(label_bounds_width, ITEM_HEIGHT),
            size: Pixels(LABEL_FONT_SIZE),
            line_height: text::LineHeight::Relative(1.3),
            font: renderer.default_font(),
            align_x: text::Alignment::Left,
            align_y: alignment::Vertical::Center,
            shaping: text::Shaping::Advanced,
            wrapping: text::Wrapping::default(),
        },
        Point::new(label_x, item_center_y),
        label_color,
        item_bounds,
    );

    // Current-role checkmark (non-current roles keep the reserved slot
    // width, matching the old `Space::new().width(12)`).
    if item.is_current {
        renderer.fill_text(
            text::Text {
                content: check_glyph().to_string(),
                bounds: Size::new(CHECK_SIZE, ITEM_HEIGHT),
                size: Pixels(CHECK_SIZE),
                line_height: text::LineHeight::Relative(1.3),
                font: iced_fonts::LUCIDE_FONT,
                align_x: text::Alignment::Left,
                align_y: alignment::Vertical::Center,
                shaping: text::Shaping::Basic,
                wrapping: text::Wrapping::default(),
            },
            Point::new(check_x, item_center_y),
            theme::ACCENT,
            item_bounds,
        );
    }
}

// ── Icon glyphs ──────────────────────────────────────────────────────

/// Checkmark glyph for the current-role row — the same codepoint
/// `lucide::check()` renders.
fn check_glyph() -> char {
    lucide_glyph(iced_fonts::lucide::advanced_text::check)
}

/// Lucide glyph for each role, matching [`theme::role_icon`] one-to-one.
///
/// The glyphs come from iced_fonts' generated `lucide::advanced_text`
/// module: iced_fonts_macros derives those functions from the pinned
/// `fonts/lucide.ttf` cmap at build time, and `theme::role_icon` renders
/// the very same codepoints via the `lucide::*` widget constructors — so
/// this mapping can never drift from the sidebar/chat icons. The overlay
/// cannot use child widgets, so it renders the glyphs manually with
/// [`iced_fonts::LUCIDE_FONT`].
fn role_icon_glyph(role: crate::Role) -> char {
    let glyph = match role {
        crate::Role::Manager => iced_fonts::lucide::advanced_text::bot,
        crate::Role::Engineer => iced_fonts::lucide::advanced_text::wrench,
        crate::Role::Analyst => iced_fonts::lucide::advanced_text::scan_search,
        crate::Role::Coder => iced_fonts::lucide::advanced_text::code,
        crate::Role::Qa => iced_fonts::lucide::advanced_text::gavel,
        crate::Role::Reviewer => iced_fonts::lucide::advanced_text::file_check,
        crate::Role::Discovery => iced_fonts::lucide::advanced_text::search,
        crate::Role::Artist => iced_fonts::lucide::advanced_text::palette,
        crate::Role::Maintainer => iced_fonts::lucide::advanced_text::cog,
        crate::Role::Sanitation => iced_fonts::lucide::advanced_text::spray_can,
        crate::Role::Assistant => iced_fonts::lucide::advanced_text::message_square,
    };
    lucide_glyph(glyph)
}

/// The single character of an iced_fonts generated lucide icon.
///
/// `advanced_text::*` returns `(String, Font, Shaping)`; the string is
/// the one codepoint iced_fonts_macros extracted from the font cmap.
fn lucide_glyph(glyph: fn() -> (String, iced::Font, text::Shaping)) -> char {
    glyph()
        .0
        .chars()
        .next()
        .expect("lucide glyph strings are single characters")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popup_position_opens_above_anchor_when_room() {
        let anchor = Rectangle {
            x: 800.0,
            y: 700.0,
            width: 30.0,
            height: 30.0,
        };
        let menu = Size::new(170.0, 100.0);
        let viewport = Size::new(1000.0, 800.0);

        let pos = popup_position(anchor, menu, viewport, ANCHOR_GAP);

        // Right-aligned to the anchor...
        assert_eq!(pos.x, anchor.x + anchor.width - menu.width);
        // ...and floating above it with the anchor gap.
        assert_eq!(pos.y, anchor.y - menu.height - ANCHOR_GAP);
    }

    #[test]
    fn popup_position_flips_below_when_no_room_above() {
        // Anchor near the top: not enough room above for the menu.
        let anchor = Rectangle {
            x: 400.0,
            y: 10.0,
            width: 30.0,
            height: 30.0,
        };
        let menu = Size::new(170.0, 100.0);
        let viewport = Size::new(1000.0, 800.0);

        let pos = popup_position(anchor, menu, viewport, ANCHOR_GAP);

        assert_eq!(pos.y, anchor.y + anchor.height + ANCHOR_GAP);
    }

    #[test]
    fn popup_position_clamps_when_neither_side_fits() {
        // Tiny viewport: neither above nor below fits, clamp to the top.
        let anchor = Rectangle {
            x: 10.0,
            y: 10.0,
            width: 30.0,
            height: 30.0,
        };
        let menu = Size::new(170.0, 500.0);
        let viewport = Size::new(200.0, 400.0);

        let pos = popup_position(anchor, menu, viewport, ANCHOR_GAP);

        assert_eq!(pos.y, 0.0);
        // Right-align clamps to the viewport left edge in a narrow window.
        assert_eq!(pos.x, 0.0);
    }

    #[test]
    fn popup_position_clamps_horizontally_in_narrow_windows() {
        // Anchor pokes past the viewport's right edge: the right-aligned
        // x (980 + 30 - 170 = 840) would overflow, so the clamp must pull
        // it back to viewport.width - menu.width. (The plain right-align
        // formula is covered by popup_position_opens_above_anchor_when_room.)
        let anchor = Rectangle {
            x: 980.0,
            y: 100.0,
            width: 30.0,
            height: 30.0,
        };
        let menu = Size::new(170.0, 100.0);
        let viewport = Size::new(1000.0, 800.0);

        let pos = popup_position(anchor, menu, viewport, ANCHOR_GAP);

        assert_eq!(pos.x, viewport.width - menu.width);
        assert_eq!(pos.x, 830.0);
    }

    #[test]
    fn item_index_from_y_accounts_for_spacing() {
        // Slot = ITEM_HEIGHT + ITEM_SPACING; rel_y is relative to the menu
        // origin (panel padding already subtracted by the caller). The
        // trailing ITEM_SPACING of each slot is dead space (no hit).
        assert_eq!(item_index_from_y(0.0, 4), Some(0));
        assert_eq!(item_index_from_y(ITEM_HEIGHT - 1.0, 4), Some(0));
        assert_eq!(item_index_from_y(ITEM_HEIGHT + 1.0, 4), None);
        assert_eq!(item_index_from_y(ITEM_HEIGHT + ITEM_SPACING, 4), Some(1));
        // Start of item 2 (2 slots in) — the gap before it maps to None.
        assert_eq!(
            item_index_from_y(ITEM_HEIGHT * 2.0 + ITEM_SPACING * 2.0, 4),
            Some(2)
        );
        assert_eq!(
            item_index_from_y(ITEM_HEIGHT * 2.0 + ITEM_SPACING * 2.0 - 1.0, 4),
            None
        );
        assert_eq!(item_index_from_y(-1.0, 4), None);
        // Past the last item.
        assert_eq!(
            item_index_from_y(ITEM_HEIGHT * 4.0 + ITEM_SPACING * 4.0, 4),
            None
        );
        // Empty menu.
        assert_eq!(item_index_from_y(0.0, 0), None);
    }

    #[test]
    fn role_icon_glyphs_are_pinned_to_expected_codepoints() {
        // Pins the codepoints `role_icon_glyph` must yield. These values
        // were verified once against the iced_fonts-0.3.0 fonts/lucide.ttf
        // cmap (ttf-parser 0.25.1, the same parser iced_fonts_macros uses)
        // and are now frozen: a match-arm edit (e.g. two roles swapped)
        // fails loudly, and an iced_fonts bump that moves lucide glyphs
        // fails too — review `theme::role_icon` (which moves identically,
        // same font) and update both together.
        let expected = [
            (crate::Role::Manager, '\u{E1BB}'),
            (crate::Role::Engineer, '\u{E1B1}'),
            (crate::Role::Analyst, '\u{E53C}'),
            (crate::Role::Coder, '\u{E097}'),
            (crate::Role::Qa, '\u{E0E4}'),
            (crate::Role::Reviewer, '\u{E0C5}'),
            (crate::Role::Discovery, '\u{E155}'),
            (crate::Role::Artist, '\u{E1DD}'),
            (crate::Role::Maintainer, '\u{E30B}'),
            (crate::Role::Sanitation, '\u{E49A}'),
            (crate::Role::Assistant, '\u{E11B}'),
        ];
        for (role, codepoint) in expected {
            assert_eq!(role_icon_glyph(role), codepoint, "role {role:?}");
        }
    }

    #[test]
    fn check_glyph_is_pinned_to_expected_codepoint() {
        // `lucide::check` is U+E070 in the pinned font (see the comment on
        // role_icon_glyphs_are_pinned_to_expected_codepoints).
        assert_eq!(check_glyph(), '\u{E070}');
    }
}
