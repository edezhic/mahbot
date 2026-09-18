//! Custom Tools dashboard page — the admin's tool files, read-only.
//!
//! The page lists the files the admin defines custom tools in, together with
//! their headers: nothing here creates, edits, grants or runs a tool. A file
//! whose header does not parse is listed as not usable instead of being skipped,
//! and the folder is re-read once a second, so a file added, fixed or deleted on
//! disk shows up without a restart.

use iced::widget::{Column, column, container, row, text};
use iced::{Alignment, Element, Length, Task};
use iced_fonts::lucide;

use crate::tools::custom::{Param, ToolListing};

use super::common::PolledList;
use super::{theme, widgets};

/// What a tool no file under whose name has a readable header shows in place of
/// its description and arguments.
const BROKEN_TOOL_NOTE: &str =
    "Not usable — the file cannot be read or its header cannot be parsed";

#[derive(Debug, Clone)]
pub(crate) enum CustomToolsMessage {
    /// One refresh interval elapsed.
    Tick,
    /// A read finished. `Err` keeps the last loaded list on screen.
    Refreshed(Result<Vec<ToolListing>, String>),
}

pub(crate) struct CustomToolsState {
    list: PolledList<ToolListing>,
}

impl CustomToolsState {
    pub(crate) const fn new() -> Self {
        Self {
            list: PolledList::new(),
        }
    }

    /// Read the tool folder now, unless a read is in flight. Called on
    /// navigation, so opening the page always starts a fresh read.
    pub(crate) fn refresh(&mut self) -> Task<CustomToolsMessage> {
        if !self.list.begin() {
            return Task::none();
        }
        Task::perform(
            crate::tools::custom::list_tool_listings(),
            CustomToolsMessage::Refreshed,
        )
    }

    pub(crate) fn update(&mut self, message: CustomToolsMessage) -> Task<CustomToolsMessage> {
        match message {
            CustomToolsMessage::Tick => self.refresh(),
            CustomToolsMessage::Refreshed(result) => {
                self.list.settle(result);
                Task::none()
            }
        }
    }

    pub(crate) fn view(&self) -> Element<'_, CustomToolsMessage> {
        let mut content = column![];

        // Error display — inset to align with the vscroll-wrapped list below.
        content = widgets::push_error_banner_inset(content, self.list.error());

        if !self.list.loaded() {
            content = content.push(widgets::scroll_h_inset(widgets::loading_text()));
        } else if self.list.entries().is_empty() {
            // A failed read must not masquerade as a legitimately empty list:
            // the error banner above is the only thing rendered then.
            if self.list.error().is_none() {
                content = content.push(widgets::empty_state_placeholder(
                    lucide::hammer::<iced::Theme, iced::Renderer>(),
                    "No custom tools",
                    theme::TEXT_MUTED,
                ));
            }
        } else {
            // A step apart, so each tool reads as its own block.
            let mut list = Column::new().spacing(theme::SPACE_4);
            for listing in self.list.entries() {
                list = list.push(render_tool(listing));
            }
            content = content.push(widgets::vscroll(list));
        }

        widgets::page(content)
    }
}

fn render_tool(listing: &ToolListing) -> Element<'_, CustomToolsMessage> {
    let mut entry = Column::new().spacing(theme::SPACE_4).push(
        text(listing.name())
            .size(theme::TEXT_14)
            .font(super::JETBRAINS_MONO)
            .color(theme::TEXT_PRIMARY),
    );
    match listing {
        ToolListing::Broken(_) => {
            entry = entry.push(
                text(BROKEN_TOOL_NOTE)
                    .size(theme::TEXT_12)
                    .color(theme::STATUS_WARNING),
            );
        }
        ToolListing::Usable(tool) => {
            entry = entry.push(
                text(&tool.description)
                    .size(theme::TEXT_13)
                    .color(theme::TEXT_SECONDARY)
                    .width(Length::Fill),
            );
            for param in &tool.params {
                entry = entry.push(render_param(param));
            }
        }
    }
    // The log row's card: one block per tool, spanning the entry width.
    container(entry)
        .padding(theme::PAD_6)
        .width(Length::Fill)
        .style(theme::surface_card_style)
        .into()
}

fn render_param(param: &Param) -> Element<'_, CustomToolsMessage> {
    row![
        text(&param.name)
            .size(theme::TEXT_12)
            .font(super::JETBRAINS_MONO)
            .color(theme::TEXT_PRIMARY),
        widgets::badge_pill(
            param.ty.as_str().to_string(),
            (theme::TEXT_SECONDARY, theme::HOVER),
            widgets::PILL_COMPACT,
        ),
        widgets::badge_pill(
            if param.required {
                "required"
            } else {
                "optional"
            }
            .to_string(),
            (theme::TEXT_SECONDARY, theme::HOVER),
            widgets::PILL_COMPACT,
        ),
        text(&param.description)
            .size(theme::TEXT_12)
            .color(theme::TEXT_SECONDARY)
            .width(Length::Fill),
    ]
    .spacing(theme::SPACE_6)
    .align_y(Alignment::Center)
    .into()
}
