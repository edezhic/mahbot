//! Key handling for widgets we do not own — the embedded terminal.
//!
//! iced's keyboard contract is built around the widgets it ships; the terminal
//! is a third-party one that captures what it handles, so two adaptations keep
//! it consistent with the rest of the dashboard:
//!
//! - A widget that matches `keyboard::Key::Character("c")` matches the
//!   character the active layout produces, so Ctrl/Cmd+C does nothing under a
//!   Cyrillic (or any other non-Latin) layout. iced's own widgets translate the
//!   key through `Key::to_latin(physical_key)`; this wrapper applies the same
//!   translation to the events the terminal receives while a command/control
//!   shortcut is held, so the terminal behaves as it does under an English
//!   layout. Keys without that modifier are passed through untouched.
//! - The terminal types any unbound printable key press into the session, so
//!   the dashboard's Logs shortcut would become a literal character instead of
//!   reaching [`keyboard::listen`]. Its `text` is dropped here, which leaves the
//!   press uncaptured. Only macOS needs this: off macOS the chord is Ctrl+L,
//!   which the terminal binds itself and therefore already captures.

use std::borrow::Cow;

use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::{Clipboard, Layout, Shell, Widget, layout, mouse, renderer};
use iced::keyboard;
use iced::{Element, Event, Length, Rectangle, Size};

/// Show `content` with the dashboard's keyboard contract applied.
pub(crate) fn terminal_keys<'a, M: 'a>(content: impl Into<Element<'a, M>>) -> Element<'a, M> {
    Element::new(TerminalKeys {
        content: content.into(),
    })
}

struct TerminalKeys<'a, M> {
    content: Element<'a, M>,
}

/// The same event, with a shortcut key translated to its latin character and
/// the Logs shortcut left untyped.
fn normalize(event: &Event) -> Cow<'_, Event> {
    let Event::Keyboard(keyboard::Event::KeyPressed {
        key,
        physical_key,
        modifiers,
        ..
    }) = event
    else {
        return Cow::Borrowed(event);
    };
    if !modifiers.command() {
        return Cow::Borrowed(event);
    }
    // macOS only: off macOS Ctrl+L is the terminal's own clear-screen binding,
    // matched before the text fallback, so the terminal keeps it.
    let logs_shortcut = cfg!(target_os = "macos")
        && super::dashboard_shortcut(key, *modifiers, *physical_key)
            == Some(super::DashboardShortcut::Logs);
    let latin = key.to_latin(*physical_key);
    if latin.is_none() && !logs_shortcut {
        return Cow::Borrowed(event);
    }
    let mut rewritten = event.clone();
    if let Event::Keyboard(keyboard::Event::KeyPressed { key, text, .. }) = &mut rewritten {
        if let Some(latin) = latin {
            *key = keyboard::Key::Character(latin.to_string().into());
        }
        if logs_shortcut {
            *text = None;
        }
    }
    Cow::Owned(rewritten)
}

impl<Message> Widget<Message, iced::Theme, iced::Renderer> for TerminalKeys<'_, Message> {
    fn tag(&self) -> tree::Tag {
        self.content.as_widget().tag()
    }

    fn state(&self) -> tree::State {
        self.content.as_widget().state()
    }

    fn children(&self) -> Vec<Tree> {
        self.content.as_widget().children()
    }

    fn diff(&self, tree: &mut Tree) {
        self.content.as_widget().diff(tree);
    }

    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content.as_widget_mut().layout(tree, renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &iced::Renderer,
        operation: &mut dyn iced::advanced::widget::Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(tree, layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &iced::Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.content.as_widget_mut().update(
            tree,
            &normalize(event),
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &iced::Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content
            .as_widget()
            .draw(tree, renderer, theme, style, layout, cursor, viewport);
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &iced::Renderer,
    ) -> mouse::Interaction {
        self.content
            .as_widget()
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &iced::Renderer,
        viewport: &Rectangle,
        translation: iced::Vector,
    ) -> Option<iced::advanced::overlay::Element<'b, Message, iced::Theme, iced::Renderer>> {
        self.content
            .as_widget_mut()
            .overlay(tree, layout, renderer, viewport, translation)
    }
}
