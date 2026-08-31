//! Shared modal and confirmation-dialog builders for the GUI.

use iced::widget::{button, column, container, row, space::Space, text};
use iced::{Alignment, Element, Length};

use crate::gui::theme;

/// Standard width of the small confirm/modal family.
const CONFIRM_WIDTH: f32 = 400.0;

/// The shared modal dialog container: fixed width, padding, elevated dialog chrome.
#[must_use]
pub fn dialog_shell<'a, Message>(
    content: impl Into<Element<'a, Message>>,
    width: f32,
    padding: f32,
) -> container::Container<'a, Message> {
    container(content)
        .width(Length::Fixed(width))
        .padding(padding)
        .style(theme::dialog_container_style)
}

/// Standard confirm-dialog title: bold 16px primary text.
#[must_use]
pub fn dialog_title<'a>(
    title: impl Into<String>,
) -> iced::widget::Text<'a, iced::Theme, iced::Renderer> {
    text(title.into())
        .size(theme::TEXT_16)
        .color(theme::TEXT_PRIMARY)
        .font(theme::FONT_BOLD)
}

/// Secondary-styled dialog body paragraph(s), separated by 8px gaps.
#[must_use]
pub fn dialog_body<'a, Message: 'a>(
    paragraphs: impl IntoIterator<Item = impl Into<String>>,
) -> Element<'a, Message> {
    let mut col = column![].spacing(theme::SPACE_8);
    for paragraph in paragraphs {
        col = col.push(
            text(paragraph.into())
                .size(theme::TEXT_13)
                .color(theme::TEXT_SECONDARY)
                .width(Length::Fill),
        );
    }
    col.into()
}

/// One right-aligned action button in a confirm dialog footer.
pub struct DialogAction<'a, Message> {
    label: &'a str,
    on_press: Message,
    style: fn(&iced::Theme, button::Status) -> button::Style,
}

impl<'a, Message> DialogAction<'a, Message> {
    /// Safe/neutral action (theme::button_secondary).
    #[must_use]
    pub fn secondary(label: &'a str, on_press: Message) -> Self {
        Self {
            label,
            on_press,
            style: theme::button_secondary,
        }
    }

    /// Destructive action (theme::button_danger).
    #[must_use]
    pub fn danger(label: &'a str, on_press: Message) -> Self {
        Self {
            label,
            on_press,
            style: theme::button_danger,
        }
    }
}

/// Right-aligned dialog footer row: fill spacer followed by the buttons,
/// 8px apart, vertically centered.
#[must_use]
pub fn dialog_footer_row<'a, Message: 'a>(
    buttons: impl IntoIterator<Item = Element<'a, Message>>,
) -> iced::widget::Row<'a, Message> {
    let mut footer = row![Space::new().width(Length::Fill)].align_y(Alignment::Center);
    for (i, btn) in buttons.into_iter().enumerate() {
        if i > 0 {
            footer = footer.push(Space::new().width(theme::SPACE_8));
        }
        footer = footer.push(btn);
    }
    footer
}

/// The shared confirmation dialog: title, body, right-aligned action buttons
/// (8px apart) in the standard 400px modal shell.
///
/// `Message: Clone` is required by iced's `Button` → `Element` conversion.
#[must_use]
pub fn confirm_dialog<'a, Message: Clone + 'a>(
    title: impl Into<Element<'a, Message>>,
    body: impl Into<Element<'a, Message>>,
    actions: impl IntoIterator<Item = DialogAction<'a, Message>>,
) -> container::Container<'a, Message> {
    let footer = dialog_footer_row(actions.into_iter().map(|action| {
        button(text(action.label).size(theme::TEXT_13))
            .style(action.style)
            .on_press(action.on_press)
            .into()
    }));

    dialog_shell(
        column![
            title.into(),
            Space::new().height(12),
            body.into(),
            Space::new().height(16),
            footer,
        ]
        .width(Length::Fill),
        CONFIRM_WIDTH,
        24.0,
    )
}
