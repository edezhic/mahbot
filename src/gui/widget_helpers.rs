//! Shared widget utilities — modal backdrop overlay and type-stable placeholder.

use iced::widget::{container, mouse_area, stack, text};
use iced::{Color, Element, Length};

/// Wrap dialog content in a centered modal overlay with a semi-transparent
/// backdrop that closes on click.
///
/// This is the shared helper for all modal backdrop patterns across the
/// dashboard. It creates a stack with a click-to-dismiss backdrop and a
/// centered container for the dialog content.
///
/// This helper does **not** apply `dialog_container_style` or padding —
/// callers are responsible for styling their content as needed before
/// passing it in.
///
/// # Parameters
/// - `content`: The dialog body to overlay. Should already be styled
///   (container, padding, etc.) by the caller as needed.
/// - `on_backdrop`: Message to emit when the backdrop is clicked.
/// - `opacity`: Opacity of the backdrop (e.g., `0.5` for standard
///   semi-transparent black, `0.4` for lighter).
pub fn modal_backdrop<'a, Message: 'a + Clone>(
    content: impl Into<Element<'a, Message>>,
    on_backdrop: Message,
    opacity: f32,
) -> Element<'a, Message> {
    let backdrop = mouse_area(
        container(text(""))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgba(
                    0.0, 0.0, 0.0, opacity,
                ))),
                ..container::Style::default()
            }),
    )
    .on_press(on_backdrop);

    let centered = container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill);

    stack([backdrop.into(), centered.into()]).into()
}

/// Zero-size placeholder that keeps an overlay slot's widget type stable
/// across show/hide transitions. Iced destroys widget state (scroll
/// positions, open popovers) when the widget tree tag changes between
/// frames, so the closed state must return the identical bare Container.
/// Callers whose open state is a Stack (board.rs, settings.rs modal
/// overlays) must wrap this in `iced::widget::stack([...])` themselves.
/// Known pre-existing quirks, not fixed here: git.rs's closed-branch is
/// unreachable (`view()` is gated on the modal being open), and the mod.rs
/// diff/branch overlay slots use a bare-Container placeholder against
/// Stack open states (type mismatch predates this helper).
pub fn empty_stack_placeholder<'a, Message: 'a>() -> Element<'a, Message> {
    container(text(""))
        .width(Length::Shrink)
        .height(Length::Shrink)
        .into()
}
