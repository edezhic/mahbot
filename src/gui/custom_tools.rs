//! Custom tools area of the Settings page — the admin's tool files, read-only.
//!
//! The area lists the files the admin defines custom tools in, with a details
//! window per tool: nothing here creates, edits, grants or runs a tool. A file
//! whose header does not parse is listed as not usable instead of being skipped,
//! and the folder is re-read once a second while the page is open, so a file
//! added, fixed or deleted on disk shows up without a restart.

use iced::Task;

use crate::tools::custom::ToolListing;

use super::common::PolledList;

#[derive(Debug, Clone)]
pub(crate) enum CustomToolsMessage {
    /// One refresh interval elapsed.
    Tick,
    /// A read finished. `Err` keeps the last loaded list on screen.
    Refreshed(Result<Vec<ToolListing>, String>),
    /// Open the read-only details window for the named tool.
    OpenDetails(String),
    /// Close the details window — the Close button, Escape or a backdrop click.
    CloseDetails,
}

pub(crate) struct CustomToolsState {
    /// The polled listing the area renders; the ticking/keeping rules are
    /// [`PolledList`]'s.
    pub(crate) list: PolledList<ToolListing>,
    /// Name of the tool whose details window is open. Every read replaces the
    /// list wholesale, so the window follows the name rather than an entry, and
    /// a name the latest read no longer lists is dropped.
    details: Option<String>,
}

impl CustomToolsState {
    pub(crate) const fn new() -> Self {
        Self {
            list: PolledList::new(),
            details: None,
        }
    }

    /// Read the tool folder now, unless a read is in flight. Called on entering
    /// Settings and on every refresh tick.
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
            CustomToolsMessage::OpenDetails(name) => {
                self.details = Some(name);
                Task::none()
            }
            CustomToolsMessage::CloseDetails => {
                self.close_details();
                Task::none()
            }
            CustomToolsMessage::Refreshed(result) => {
                self.list.settle(result);
                // The selection follows the listing: re-deriving it drops a tool
                // the read no longer lists, so a tool that comes back later
                // cannot resurrect its window.
                self.details = self.details().map(|tool| tool.name().to_owned());
                Task::none()
            }
        }
    }

    /// Close the details window (its Close button, Escape or a backdrop click).
    pub(crate) fn close_details(&mut self) {
        self.details = None;
    }

    /// The tool the details window shows, or `None` when no window is open or
    /// the tool is no longer listed. The single source of truth for whether the
    /// window is open, and total by construction — an absent tool is never
    /// looked up and never panics.
    pub(crate) fn details(&self) -> Option<&ToolListing> {
        let name = self.details.as_deref()?;
        self.list.entries().iter().find(|l| l.name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The listing is replaced wholesale by every read, so a tool dropped from a
    /// read closes its window — and a later read settling it back must not
    /// reopen it.
    #[test]
    fn a_read_that_drops_the_open_tool_closes_its_window() {
        let mut state = CustomToolsState::new();
        let _ = state.update(CustomToolsMessage::Refreshed(Ok(weather())));
        let _ = state.update(CustomToolsMessage::OpenDetails("weather".to_string()));
        assert!(state.details().is_some());

        let _ = state.update(CustomToolsMessage::Refreshed(Ok(Vec::new())));
        let _ = state.update(CustomToolsMessage::Refreshed(Ok(weather())));
        assert!(state.details().is_none());
    }

    fn weather() -> Vec<ToolListing> {
        vec![ToolListing::Broken("weather".to_string())]
    }
}
