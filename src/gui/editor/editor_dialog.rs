//! Dialog UI builders for the editor — overlay, confirmation, quick-open,
//! close-save-discard, delete-confirm, and new-item-input dialogs.
//!
//! These are free functions extracted from `EditorState` in `editor.rs`.
//! They take all state explicitly via parameters; none access `&self`.

use iced::widget::{Row, Space, button, column, container, row, scrollable, text, text_input};
use iced::{Alignment, Color, Element, Length, widget::Id};

use iced_fonts::lucide;

use crate::gui::theme;
use crate::gui::widgets;

use super::{
    DeleteConfirmTarget, EditorMessage, GLOBAL_SEARCH_INPUT_ID, GlobalSearchState,
    GlobalSearchStatus, NEW_ITEM_INPUT_ID, NewItemTarget, OwnedGrepMatch, QUICK_OPEN_INPUT_ID,
    QuickOpenState,
};

// ── Overlay helpers ────────────────────────────────────────────────

/// Wrap any dialog element in a centered overlay with a semi-transparent
/// backdrop that closes the dialog on click.
///
/// Delegates to the shared backdrop helper. The caller may specify a custom
/// backdrop opacity; most dialogs use `0.5`, but lighter overlays (e.g.
/// quick-open) use `0.3`.
pub(super) fn overlay_dialog<'a>(
    dialog: impl Into<Element<'a, EditorMessage>>,
    on_backdrop: EditorMessage,
    opacity: f32,
) -> Element<'a, EditorMessage> {
    super::super::widget_helpers::modal_backdrop(dialog, on_backdrop, opacity)
}

/// Wrap dialog content in the shared dialog container and overlay it.
/// All standard dialogs use this to ensure consistent container dimensions,
/// padding, style, and backdrop behavior.
pub(super) fn wrap_dialog<'a>(
    content: impl Into<Element<'a, EditorMessage>>,
    width: u32,
    cancel_msg: EditorMessage,
    opacity: f32,
) -> Element<'a, EditorMessage> {
    overlay_dialog(
        container(content)
            .width(width)
            .padding(24)
            .style(theme::dialog_container_style),
        cancel_msg,
        opacity,
    )
}

/// Build the quick-open overlay: a centered dialog with search input
/// and filtered results list.
pub(super) fn build_quick_open_overlay(qo: &QuickOpenState) -> Element<'static, EditorMessage> {
    let search_input: iced::widget::TextInput<'_, EditorMessage> =
        text_input("Search files…", &qo.filter)
            .on_input(EditorMessage::QuickOpenInput)
            .on_submit(qo.results.first().map_or(EditorMessage::Escape, |_| {
                EditorMessage::QuickOpenSelect(qo.selected_index)
            }))
            .id(Id::new(QUICK_OPEN_INPUT_ID))
            .style(widgets::text_input_style)
            .size(14)
            .width(Length::Fill)
            .padding([8, 12]);

    // Convert to Element for use in column.
    let search_elem: Element<'static, EditorMessage> = search_input.into();

    // Build results list with owned data to satisfy 'static lifetime.
    let results_owned: Vec<String> = qo.results.clone();
    let selected_index = qo.selected_index;

    let results: Vec<Element<'static, EditorMessage>> = results_owned
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let is_selected = i == selected_index;
            let path_owned = path.clone();
            let bg = if is_selected {
                theme::HOVER_STRONG
            } else {
                iced::Color::TRANSPARENT
            };
            let label = text(path_owned.clone()).size(12).color(if is_selected {
                theme::ACCENT
            } else {
                theme::TEXT_SECONDARY
            });
            let entry = container(label).padding([4, 12]).width(Length::Fill).style(
                move |_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(bg)),
                    ..Default::default()
                },
            );
            button(entry)
                .on_press(EditorMessage::QuickOpenSelect(i))
                .style(theme::button_transparent)
                .width(Length::Fill)
                .padding(0)
                .into()
        })
        .collect();

    let results_column = column(results).spacing(0).width(Length::Fill);

    let empty_hint: Option<Element<'static, EditorMessage>> =
        if qo.filter.is_empty() && qo.results.is_empty() {
            Some(
                text("Type to filter files…")
                    .size(12)
                    .color(theme::TEXT_FAINT)
                    .into(),
            )
        } else if qo.results.is_empty() {
            Some(
                text("No matches found")
                    .size(12)
                    .color(theme::TEXT_MUTED)
                    .into(),
            )
        } else {
            None
        };

    let content: Element<'static, EditorMessage> = if let Some(hint) = empty_hint {
        column![search_elem, hint].spacing(4).into()
    } else {
        column![
            search_elem,
            scrollable(results_column)
                .height(Length::Fixed(300.0))
                .style(theme::scrollbar_style),
        ]
        .spacing(4)
        .into()
    };

    let dialog = container(content)
        .width(Length::Fixed(400.0))
        .padding(12)
        .style(theme::dialog_container_style);

    overlay_dialog(dialog, EditorMessage::Escape, 0.4)
}

// ── Close dialog ──────────────────────────────────────────────────

/// Shared confirmation dialog builder.
///
/// Constructs a standardised confirmation overlay with an icon row (warning icon +
/// title), a description, and a caller-provided button row. All confirmation
/// dialogs (e.g. unsaved-changes, delete-confirm) use this to avoid duplicating
/// the identical icon row, description styling, column wrapper, and wrap_dialog
/// call.
///
/// Callers supply the button row as a pre-assembled [`Row`] (with its own
/// `.align_y` / `.width` styling) so each site can freely choose button
/// composition while reusing the structural boilerplate.
fn confirmation_dialog<'a>(
    title: impl Into<String>,
    description: impl Into<String>,
    button_row: impl Into<Element<'a, EditorMessage>>,
    width: u32,
    cancel_msg: EditorMessage,
    opacity: f32,
) -> Element<'a, EditorMessage> {
    let title: String = title.into();
    let description: String = description.into();
    wrap_dialog(
        column![
            row![
                lucide::triangle_alert::<iced::Theme, iced::Renderer>()
                    .size(16)
                    .color(theme::STATUS_WARNING),
                Space::new().width(8),
                text(title).size(16).color(theme::TEXT_PRIMARY),
            ]
            .align_y(Alignment::Center),
            text(description)
                .size(14)
                .color(theme::TEXT_SECONDARY)
                .width(Length::Fill),
            button_row.into(),
        ]
        .spacing(16)
        .width(Length::Fill),
        width,
        cancel_msg,
        opacity,
    )
}

/// Create a styled dialog button with consistent size (13) and center-aligned text.
fn dialog_button(
    label: &str,
    color: Color,
    style: fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style,
    on_press: EditorMessage,
) -> Element<'_, EditorMessage> {
    button(text(label).size(13).color(color).align_x(Alignment::Center))
        .style(style)
        .on_press(on_press)
        .into()
}

/// Create a row of dialog buttons with 8px spacing between them,
/// right-aligned within the row and filling the available width.
fn dialog_button_row<'a>(
    buttons: impl IntoIterator<Item = Element<'a, EditorMessage>>,
) -> Element<'a, EditorMessage> {
    let mut row = Row::new().align_y(Alignment::End).width(Length::Fill);
    for (i, btn) in buttons.into_iter().enumerate() {
        if i > 0 {
            row = row.push(Space::new().width(8));
        }
        row = row.push(btn);
    }
    row.into()
}

/// Build the close-save-discard dialog overlay.
pub(super) fn build_close_dialog(
    on_save: EditorMessage,
    on_discard: EditorMessage,
    on_cancel: EditorMessage,
    description: String,
) -> Element<'static, EditorMessage> {
    let button_row = dialog_button_row([
        dialog_button(
            "Cancel",
            theme::TEXT_SECONDARY,
            theme::button_secondary,
            on_cancel.clone(),
        ),
        dialog_button(
            "Discard",
            theme::STATUS_ERROR,
            theme::button_danger,
            on_discard,
        ),
        dialog_button("Save", theme::ACCENT_LIGHT, theme::button_primary, on_save),
    ]);

    confirmation_dialog(
        "Unsaved changes",
        description,
        button_row,
        380,
        on_cancel,
        0.5,
    )
}

/// Build the delete confirmation dialog overlay.
pub(super) fn build_delete_confirm_dialog(
    target: &DeleteConfirmTarget,
) -> Element<'static, EditorMessage> {
    let description = if target.is_dir {
        let dirty_msg = if target.dirty_tab_count > 0 {
            format!(
                " ({} tab{} with unsaved changes will be closed)",
                target.dirty_tab_count,
                if target.dirty_tab_count == 1 { "" } else { "s" }
            )
        } else {
            String::new()
        };
        format!(
            "Delete directory \"{}\" and all its contents?{}",
            target.path, dirty_msg
        )
    } else {
        format!("Delete file \"{}\"?", target.path)
    };

    let title = if target.is_dir {
        "Delete directory"
    } else {
        "Delete file"
    };

    let button_row = dialog_button_row([
        dialog_button(
            "Cancel",
            theme::TEXT_SECONDARY,
            theme::button_secondary,
            EditorMessage::CancelDelete,
        ),
        dialog_button(
            "Delete",
            theme::STATUS_ERROR,
            theme::button_danger,
            EditorMessage::ConfirmDelete,
        ),
    ]);

    confirmation_dialog(
        title,
        description,
        button_row,
        400,
        EditorMessage::CancelDelete,
        0.5,
    )
}

/// Build the new file/directory name input overlay.
pub(super) fn build_new_item_input(target: &NewItemTarget) -> Element<'_, EditorMessage> {
    let label = if target.is_dir {
        format!("New directory in \"{}\"", target.parent_dir)
    } else {
        format!("New file in \"{}\"", target.parent_dir)
    };

    let input = text_input("Name…", &target.input_text)
        .id(Id::new(NEW_ITEM_INPUT_ID))
        .on_input(EditorMessage::NewItemInput)
        .on_submit(EditorMessage::NewItemSubmit(target.input_text.clone()))
        .style(widgets::text_input_style)
        .padding(8);

    // Dialog content.
    wrap_dialog(
        column![
            text(label).size(14).color(theme::TEXT_PRIMARY),
            input,
            dialog_button_row([
                dialog_button(
                    "Cancel",
                    theme::TEXT_SECONDARY,
                    theme::button_secondary,
                    EditorMessage::Escape,
                ),
                dialog_button(
                    "Create",
                    theme::ACCENT_LIGHT,
                    theme::button_primary,
                    EditorMessage::NewItemSubmit(target.input_text.clone()),
                ),
            ]),
        ]
        .spacing(12)
        .width(Length::Fill),
        380,
        EditorMessage::Escape,
        0.4,
    )
}

// ── Global Search overlay ──────────────────────────────────────────

/// Build the global search (find-in-files) overlay: a search input with
/// status line and results list, wrapped in a centered overlay dialog.
#[allow(clippy::too_many_lines)]
pub(super) fn build_global_search_overlay(
    gs: &GlobalSearchState,
) -> Element<'static, EditorMessage> {
    let search_input: iced::widget::TextInput<'_, EditorMessage> =
        text_input("Search across workspace…", &gs.query)
            .on_input(EditorMessage::GlobalSearchInput)
            .on_submit(if gs.results.is_empty() {
                EditorMessage::Escape
            } else {
                EditorMessage::GlobalSearchSelect(gs.selected_index)
            })
            .id(Id::new(GLOBAL_SEARCH_INPUT_ID))
            .style(widgets::text_input_style)
            .size(14)
            .width(Length::Fill)
            .padding([8, 12]);

    let search_elem: Element<'static, EditorMessage> = search_input.into();

    // Status/hint line below the input.
    let status_elem: Element<'static, EditorMessage> = match &gs.status {
        GlobalSearchStatus::Idle => text("Type to search across workspace files")
            .size(12)
            .color(theme::TEXT_FAINT)
            .into(),
        GlobalSearchStatus::Searching => row![
            text("Searching…").size(12).color(theme::TEXT_MUTED),
            Space::new().width(Length::Fill),
        ]
        .into(),
        GlobalSearchStatus::NoResults => text("No matches found")
            .size(12)
            .color(theme::TEXT_MUTED)
            .into(),
        GlobalSearchStatus::Error(e) => text(format!("Search error: {e}"))
            .size(12)
            .color(theme::STATUS_ERROR)
            .into(),
        GlobalSearchStatus::Done => {
            let count = gs.results.len();
            text(format!(
                "{count} result{}",
                if count == 1 { "" } else { "s" }
            ))
            .size(12)
            .color(theme::TEXT_FAINT)
            .into()
        }
    };

    // Build results list with owned data to satisfy 'static lifetime.
    let results_owned: Vec<OwnedGrepMatch> = gs.results.clone();
    let selected_index = gs.selected_index;

    let results: Vec<Element<'static, EditorMessage>> = results_owned
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let is_selected = i == selected_index;
            let bg = if is_selected {
                theme::HOVER_STRONG
            } else {
                iced::Color::TRANSPARENT
            };

            // Clone owned data into the closure to satisfy 'static.
            let path_text = format!("{}:{}", m.rel_path, m.line_number);
            let line_content = m.line_content.clone();
            let offsets = m.match_byte_offsets.clone();
            let accent = if is_selected {
                theme::ACCENT
            } else {
                theme::TEXT_SECONDARY
            };

            let path_label = text(path_text.clone())
                .size(12)
                .color(accent)
                .font(iced::Font {
                    weight: iced::font::Weight::Bold,
                    ..theme::FONT_REGULAR
                });

            // Build the snippet with match portion highlighted in bold.
            let trimmed = line_content.trim().to_string();
            let snippet_label: Element<'static, EditorMessage> =
                if let Some(&(start, end)) = offsets.first() {
                    let start_us = start as usize;
                    let end_us = end as usize;
                    let content = &line_content;
                    let pre = if start_us <= content.len() {
                        content[..start_us.min(content.len())]
                            .trim_start()
                            .to_string()
                    } else {
                        String::new()
                    };
                    let matched = if start_us < content.len() && end_us <= content.len() {
                        content[start_us..end_us].to_string()
                    } else {
                        String::new()
                    };
                    let post = if end_us < content.len() {
                        content[end_us..].trim_end().to_string()
                    } else {
                        String::new()
                    };

                    let text_color = if is_selected {
                        theme::TEXT_PRIMARY
                    } else {
                        theme::TEXT_MUTED
                    };

                    row![
                        text(pre).size(12).color(text_color),
                        text(matched)
                            .size(12)
                            .color(theme::ACCENT_LIGHT)
                            .font(iced::Font {
                                weight: iced::font::Weight::Bold,
                                ..theme::FONT_REGULAR
                            }),
                        text(post).size(12).color(text_color),
                    ]
                    .spacing(0)
                    .into()
                } else {
                    text(trimmed)
                        .size(12)
                        .color(if is_selected {
                            theme::TEXT_PRIMARY
                        } else {
                            theme::TEXT_MUTED
                        })
                        .into()
                };

            let entry_content = column![path_label, snippet_label].spacing(1);

            let entry = container(entry_content)
                .padding([4, 12])
                .width(Length::Fill)
                .style(move |_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(bg)),
                    ..Default::default()
                });

            button(entry)
                .on_press(EditorMessage::GlobalSearchSelect(i))
                .style(theme::button_transparent)
                .width(Length::Fill)
                .padding(0)
                .into()
        })
        .collect();

    let results_column = column(results).spacing(0).width(Length::Fill);

    let has_results = !gs.results.is_empty();

    let content: Element<'static, EditorMessage> = if !has_results {
        column![search_elem, status_elem].spacing(4).into()
    } else {
        column![
            search_elem,
            status_elem,
            scrollable(results_column)
                .height(Length::Fixed(400.0))
                .style(theme::scrollbar_style),
        ]
        .spacing(4)
        .into()
    };

    let dialog = container(content)
        .width(Length::Fixed(600.0))
        .padding(12)
        .style(theme::dialog_container_style);

    overlay_dialog(dialog, EditorMessage::Escape, 0.4)
}
