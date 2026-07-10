//! Shared widget utilities — modal backdrop overlay.

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
