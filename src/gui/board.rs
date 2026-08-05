//! Board dashboard page — ticket management.

use std::collections::HashSet;
use std::path::Path;
use std::str::FromStr;

use crate::Role;
use crate::board::{Ticket, TicketPhase};
use crate::git_commands::{parse_numstat_lines, run_git_output};

use iced::widget::{
    Column, Row, Space, button, column, container, markdown, row, scrollable, text, text_editor,
    tooltip,
};
use iced::{Alignment, Element, Length, Task, keyboard};

use iced_fonts::lucide;

use super::common::MAX_INPUT_CHARS;
use super::theme;
use super::widget_helpers;
use super::widgets::{badge_pill, diff_stats_row, selectable_text};

/// Per-file stat from `git show --numstat`.
#[derive(Debug, Clone)]
pub struct FileStat {
    path: String,
    additions: i64,
    deletions: i64,
}

/// Parsed commit stats for a ticket's associated commit.
#[derive(Debug, Clone)]
pub struct CommitStats {
    files: Vec<FileStat>,
}

#[derive(Debug, Clone)]
pub enum BoardMessage {
    Refreshed(Vec<Ticket>),
    RefreshError(String),
    TicketDetails(Box<Ticket>),
    DetailError(String),
    PerformAction(String, String), // ticket_id, new_phase
    ActionResult(Result<(), String>),

    /// Open the ticket detail modal.
    OpenModal(String),

    /// Close the detail modal.
    CloseModal,

    /// Dismiss modals/panels (Escape key).
    Escape,

    /// A link was clicked in rendered markdown.
    LinkClicked(String),

    /// Request toast notification.
    Toast(super::ToastMessage),

    /// Batch-archive all done and cancelled tickets.
    ArchiveAllCompleted,

    /// Result of batch archive operation.
    ArchiveAllCompletedResult(Result<u64, String>),

    /// Archive a single ticket (sets is_archived = 1).
    ArchiveTicket(String),

    /// Trigger async load of commit stats for a ticket.
    FetchCommitStats(String),
    /// Commit stats loaded (or error) — carries generation for stale-callback guard.
    CommitStatsLoaded(String, u64, Result<CommitStats, String>),

    /// Navigate to the commit diff view for this ticket.
    ViewCommitDiff {
        commit_hash: String,
        workspace_name: String,
    },

    /// Toggle expansion of a diagnostics comment.
    ToggleCommentExpand(usize),

    /// Comment text input changed in the ticket detail modal.
    CommentInputChanged(text_editor::Action),
    /// Send a comment on the current ticket.
    SendComment,
    /// Result of adding a comment via the backend. Carries the generation
    /// counter captured at send-time for stale-callback detection.
    CommentSent(u64, Result<(), String>),
    /// Keyboard modifier state change (Shift, Ctrl, Cmd) — tracked for
    /// Shift+Click→Drag conversion in the comment input.
    CommentModifiersChanged(keyboard::Modifiers),
    /// Undo the last text edit in the comment input.
    Undo,
    /// Redo a previously undone text edit in the comment input.
    Redo,

    // ── Ticket search (sidebar) ──────────────────────────────────
    /// Search query text changed (user typing in the sidebar search input).
    SearchInputChanged(String),
    /// Search results returned from async FTS query.
    /// Carries a generation counter for stale-callback detection.
    SearchResults(Vec<Ticket>, u64),
    /// Clear the active search and restore the normal ticket list.
    SearchCleared,
}

pub struct BoardState {
    pub(crate) tickets: Vec<Ticket>,
    pub(crate) load_state: super::common::AsyncLoadState,
    selected_ticket: Option<Ticket>,
    selected_loading: bool,
    action_loading: Option<String>,
    /// Cached parsed markdown for the selected ticket description.
    description_md: Option<Vec<markdown::Item>>,
    /// Cached parsed markdown for comments (re-parsed when ticket changes).
    comments_md: Vec<(usize, Vec<markdown::Item>)>,
    /// Current workspace name filter (set by global picker).
    pub(crate) workspace_name: Option<String>,
    /// Loaded commit stats for the open ticket.
    commit_stats: Option<CommitStats>,
    /// Whether a commit stats fetch is in progress.
    commit_stats_loading: bool,
    /// Incremented on each new fetch; stale callbacks discarded.
    commit_stats_generation: u64,
    /// Tracks which comment indices are expanded (for diagnostics collapse).
    expanded_comments: HashSet<usize>,
    /// Stores the last detail-load error message for display in the modal.
    detail_error: Option<String>,

    /// Comment text input state for the ticket detail modal.
    comment_input: text_editor::Content,
    /// Whether a comment is currently being sent (prevents double-send).
    sending_comment: bool,
    /// Whether the comment input is focused (for Escape-key blur behavior).
    ///
    /// Note: tracking is imperfect — clicking elsewhere (natural focus loss)
    /// won't clear this flag, so the first Escape after a natural blur still
    /// acts as "blur" instead of "close". A second Escape always closes the
    /// modal, which is acceptable UX for Iced 0.14's widget focus API limits.
    comment_focused: bool,
    /// Current user name used for comment attribution (`"user:{name}"`).
    pub(crate) current_user_name: Option<String>,
    /// Monotonic counter incremented when the modal context changes
    /// (modal close via `reset_modal()`, or ticket switch via
    /// `TicketDetails`). Captured in `SendComment` and verified in
    /// `CommentSent` to detect stale async callbacks.
    comment_generation: u64,
    /// Latest keyboard modifiers, tracked for Shift+Click→Drag conversion
    /// in the comment input (matching the Home chat input pattern).
    modifiers: keyboard::Modifiers,
    /// Undo/redo stack for the comment input text editor.
    undo_stack: super::common::UndoStack,

    // ── Ticket search (sidebar) ──────────────────────────────────
    /// Current search query (empty = no search active).
    pub(crate) search_query: String,
    /// Search results, populated when `search_query` is non-empty.
    pub(crate) search_results: Vec<Ticket>,
    /// Incremented on each new search; stale callbacks check this before
    /// applying results. Follows the same pattern as `commit_stats_generation`.
    pub(crate) search_generation: u64,
}

impl BoardState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tickets: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            selected_ticket: None,
            selected_loading: false,
            action_loading: None,
            description_md: None,
            comments_md: Vec::new(),
            workspace_name: None,
            commit_stats: None,
            commit_stats_loading: false,
            commit_stats_generation: 0,
            expanded_comments: HashSet::new(),
            detail_error: None,
            comment_input: text_editor::Content::new(),
            sending_comment: false,
            comment_focused: false,
            current_user_name: None,
            comment_generation: 0,
            modifiers: keyboard::Modifiers::empty(),
            undo_stack: super::common::UndoStack::new(),
            search_query: String::new(),
            search_results: Vec::new(),
            search_generation: 0,
        }
    }

    /// Reset all modal-related state fields (close detail modal).
    fn reset_modal(&mut self) {
        self.selected_ticket = None;
        self.selected_loading = false;
        self.detail_error = None;
        self.description_md = None;
        self.comments_md.clear();
        self.expanded_comments.clear();
        self.commit_stats = None;
        self.commit_stats_loading = false;
        self.commit_stats_generation += 1;
        self.comment_input = text_editor::Content::new();
        self.sending_comment = false;
        self.comment_focused = false;
        self.comment_generation += 1;
        self.undo_stack.clear();
    }

    pub fn refresh(&self) -> Task<BoardMessage> {
        let ws_name = self.workspace_name.clone();
        Task::perform(
            async move {
                let board = crate::board::store();
                board
                    .list_all_tickets(ws_name.as_deref(), None)
                    .await
                    .map_err(|e| e.to_string())
            },
            |res| match res {
                Ok(tickets) => BoardMessage::Refreshed(tickets),
                Err(e) => BoardMessage::RefreshError(e),
            },
        )
    }

    #[allow(clippy::unused_self)]
    pub fn subscription(&self) -> iced::Subscription<BoardMessage> {
        // Track keyboard modifiers for Shift+Click→Drag conversion in the
        // comment input, and handle Cmd+Z / Cmd+Shift+Z for undo/redo,
        // mirroring the Home chat input pattern.
        use iced::keyboard;
        keyboard::listen().filter_map(|event| {
            super::common::composer_keyboard_event(
                event,
                BoardMessage::CommentModifiersChanged,
                || BoardMessage::Undo,
                || BoardMessage::Redo,
            )
        })
    }

    /// Phase transition actions (ported from Board.tsx `availableActions`)
    fn available_actions(phase: TicketPhase) -> Vec<(&'static str, TicketPhase)> {
        match phase {
            TicketPhase::ReadyForDevelopment => vec![
                ("⏸ Pause", TicketPhase::Planning),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::Reviewed => vec![
                ("✅ Send to QA", TicketPhase::InQa),
                ("🔄 Redo Dev", TicketPhase::ReadyForDevelopment),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::Planning => vec![
                ("✅ Ready for Dev", TicketPhase::ReadyForDevelopment),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::Done | TicketPhase::Cancelled => {
                vec![]
            }
            _ => vec![("🛑 Cancel", TicketPhase::Cancelled)],
        }
    }

    /// Map an action label to the appropriate lucide icon element (16px).
    fn action_icon<'a>(label: &str) -> iced::widget::Text<'a, iced::Theme, iced::Renderer> {
        match label {
            l if l.contains("Cancel") => lucide::circle_x(),
            l if l.contains("Redo") => lucide::refresh_cw(),
            l if l.contains("QA") => lucide::shield_check(),
            l if l.contains("Pause") => lucide::pause(),
            l if l.contains("Dev") => lucide::play(),
            _ => lucide::circle_check(),
        }
    }

    /// Build a row of icon-only action buttons for the given ticket and actions.
    /// Icons are 16px with 4px spacing. Cancel gets red [`theme::button_text_danger`]
    /// treatment; all others use [`theme::button_text`]. When `is_disabled` is true
    /// all buttons dim to [`theme::TEXT_MUTED`] and become non-interactive.
    fn action_icon_row<'a>(
        ticket_id: &str,
        actions: &[(&'static str, TicketPhase)],
        is_disabled: bool,
    ) -> Row<'a, BoardMessage> {
        let mut icon_row = Row::new().spacing(4);
        for (label, phase) in actions {
            let is_cancel = label.contains("Cancel");
            let icon = Self::action_icon(label);
            let icon_color = if is_disabled {
                theme::TEXT_MUTED
            } else if is_cancel {
                theme::STATUS_ERROR
            } else {
                theme::TEXT_PRIMARY
            };
            let style_fn: fn(
                &iced::Theme,
                iced::widget::button::Status,
            ) -> iced::widget::button::Style = if is_cancel {
                theme::button_text_danger
            } else {
                theme::button_text
            };
            icon_row = icon_row.push(
                button(icon.size(16).color(icon_color))
                    .style(style_fn)
                    .on_press_maybe(if is_disabled {
                        None
                    } else {
                        Some(BoardMessage::PerformAction(
                            ticket_id.to_string(),
                            phase.to_string(),
                        ))
                    }),
            );
        }
        icon_row
    }

    /// Phase badge pill (ticket cards, modal header); colors from [`theme::ticket_phase_color`].
    fn phase_badge<'a>(
        phase: TicketPhase,
        text_size: u32,
        padding: [u16; 2],
    ) -> Element<'a, BoardMessage> {
        let (bg, fg) = theme::ticket_phase_color(phase);
        badge_pill(phase.display_name(), (bg, fg), text_size, padding)
    }

    /// Priority chip pill; colors from [`theme::ticket_priority_color`].
    fn priority_badge<'a>(
        priority: i64,
        text_size: u32,
        padding: [u16; 2],
    ) -> Element<'a, BoardMessage> {
        let (bg, fg) = theme::ticket_priority_color(priority);
        badge_pill(format!("P{priority}"), (bg, fg), text_size, padding)
    }

    /// Compute how many of this ticket's prerequisites are still unfulfilled.
    /// A prerequisite is considered fulfilled if its ticket cannot be found in the
    /// loaded set (per manager clarification: missing = archived = fulfilled) or if
    /// its phase passes [`TicketPhase::is_unblocking()`].
    fn unfulfilled_prereq_count(&self, ticket: &Ticket) -> (usize, Vec<String>) {
        if ticket.prerequisites.is_empty() {
            return (0, Vec::new());
        }
        let phase_map: std::collections::HashMap<&str, TicketPhase> = self
            .tickets
            .iter()
            .map(|t| (t.id.as_str(), t.phase))
            .collect();
        let mut unfulfilled_ids = Vec::new();
        for prereq_id in &ticket.prerequisites {
            let is_unfulfilled = match phase_map.get(prereq_id.as_str()) {
                Some(phase) => !phase.is_unblocking(),
                None => false, // missing = archived = fulfilled
            };
            if is_unfulfilled {
                unfulfilled_ids.push(prereq_id.clone());
            }
        }
        let count = unfulfilled_ids.len();
        (count, unfulfilled_ids)
    }

    /// Fetch a single ticket by ID. Returns a Task that resolves to TicketDetails or DetailError.
    fn fetch_ticket(id: String) -> Task<BoardMessage> {
        Task::perform(
            async move {
                let board = crate::board::store();
                board.get_ticket(&id).await.map_err(|e| e.to_string())
            },
            |res| match res {
                Ok(Some(ticket)) => BoardMessage::TicketDetails(Box::new(ticket)),
                Ok(None) => BoardMessage::DetailError("Ticket not found".into()),
                Err(e) => BoardMessage::DetailError(e),
            },
        )
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: BoardMessage) -> Task<BoardMessage> {
        match msg {
            BoardMessage::Refreshed(tickets) => {
                self.tickets = tickets;
                self.load_state.finish_loading();
                Task::none()
            }
            BoardMessage::RefreshError(e) => {
                self.load_state.fail(e);
                Task::none()
            }
            BoardMessage::OpenModal(id) => {
                self.selected_loading = true;
                self.detail_error = None;
                // Bump generation to invalidate any in-flight comment
                // callbacks for the previous ticket before the fetch
                // even completes.
                self.comment_generation += 1;
                Self::fetch_ticket(id)
            }
            BoardMessage::CloseModal => {
                self.reset_modal();
                Task::none()
            }
            BoardMessage::Escape => {
                if self.comment_focused {
                    // On first Escape, blur the comment input (clear focus flag)
                    // so a second Escape closes the modal.
                    self.comment_focused = false;
                    Task::none()
                } else {
                    self.reset_modal();
                    Task::none()
                }
            }
            BoardMessage::TicketDetails(ticket) => {
                let ticket = *ticket;
                // Defensively clear any stale error; if we got details, we're good.
                self.detail_error = None;
                // Cache parsed markdown for description and comments
                self.description_md = if ticket.description.is_empty() {
                    None
                } else {
                    Some(markdown::parse(&ticket.description).collect())
                };
                self.comments_md = ticket
                    .comments
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, markdown::parse(&c.content).collect()))
                    .collect();
                self.selected_ticket = Some(ticket);
                self.selected_loading = false;
                // Bump generation to invalidate any in-flight comment
                // callbacks for the previous ticket.
                self.comment_generation += 1;
                // Clear comment input state when switching to a new ticket
                // to prevent a draft from the previous ticket being
                // accidentally sent to the new one.
                self.comment_input = text_editor::Content::new();
                self.sending_comment = false;
                self.comment_focused = false;
                self.undo_stack.clear();

                // Trigger commit stats fetch if commit_hash is set
                if self
                    .selected_ticket
                    .as_ref()
                    .and_then(|t| t.commit_hash.as_ref())
                    .is_some()
                {
                    self.commit_stats = None;
                    self.commit_stats_loading = true;
                    self.commit_stats_generation += 1;
                    let ticket_id = self.selected_ticket.as_ref().unwrap().id.clone();
                    Task::done(BoardMessage::FetchCommitStats(ticket_id))
                } else {
                    self.commit_stats = None;
                    self.commit_stats_loading = false;
                    Task::none()
                }
            }
            BoardMessage::DetailError(e) => {
                self.detail_error = Some(e);
                self.selected_loading = false;
                Task::none()
            }
            BoardMessage::PerformAction(ticket_id, new_phase) => {
                self.action_loading = Some(ticket_id.clone());
                Task::perform(
                    async move {
                        let board = crate::board::store();
                        let phase: TicketPhase = new_phase
                            .parse()
                            .map_err(|_| format!("Invalid phase: {new_phase}"))?;
                        board
                            .transition_to(&ticket_id, None, phase, None)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    BoardMessage::ActionResult,
                )
            }
            BoardMessage::ActionResult(Ok(())) => {
                self.action_loading = None;
                // Refresh ticket list and detail
                let refresh = self.refresh();
                let detail_refetch = self
                    .selected_ticket
                    .as_ref()
                    .map(|t| t.id.clone())
                    .map_or(Task::none(), Self::fetch_ticket);
                let toast = Task::done(BoardMessage::Toast(super::ToastMessage::Saved));
                Task::batch([refresh, detail_refetch, toast])
            }
            BoardMessage::ActionResult(Err(e)) => {
                self.action_loading = None;
                Task::done(BoardMessage::Toast(super::ToastMessage::Error(e)))
            }
            BoardMessage::ToggleCommentExpand(i) => {
                if !self.expanded_comments.remove(&i) {
                    self.expanded_comments.insert(i);
                }
                Task::none()
            }
            BoardMessage::CommentInputChanged(action) => {
                // Any interaction means the input is focused.
                self.comment_focused = true;
                super::common::apply_editor_action(
                    &mut self.comment_input,
                    &mut self.undo_stack,
                    action,
                    self.modifiers.shift(),
                );
                Task::none()
            }
            BoardMessage::Undo => {
                let snapshot = self.undo_stack.undo(&self.comment_input);
                super::common::restore_undo_snapshot(&mut self.comment_input, snapshot);
                Task::none()
            }
            BoardMessage::Redo => {
                let snapshot = self.undo_stack.redo(&self.comment_input);
                super::common::restore_undo_snapshot(&mut self.comment_input, snapshot);
                Task::none()
            }
            BoardMessage::CommentModifiersChanged(modifiers) => {
                self.modifiers = modifiers;
                Task::none()
            }
            BoardMessage::SendComment => {
                let text = self.comment_input.text();
                let trimmed =
                    match super::common::send_guard(&text, self.sending_comment, false, |count| {
                        Task::done(BoardMessage::Toast(super::ToastMessage::Warning(format!(
                            "Comment too long: {count} characters (maximum {MAX_INPUT_CHARS}). \
                                 Please shorten your comment."
                        ))))
                    }) {
                        Ok(t) => t.to_string(),
                        Err(task) => return task,
                    };

                let Some(ref user_name) = self.current_user_name else {
                    return Task::done(BoardMessage::Toast(super::ToastMessage::Warning(
                        "No user selected — cannot add comment.".into(),
                    )));
                };

                let Some(ticket_id) = self.selected_ticket.as_ref().map(|t| t.id.clone()) else {
                    return Task::none();
                };

                let role = format!("user:{user_name}");
                let now = crate::turso::now();
                let content = trimmed;

                // Optimistically add the comment to local state so it appears
                // immediately in the comments list.
                if let Some(ref mut ticket) = self.selected_ticket {
                    ticket.comments.push(crate::board::TicketComment {
                        role: role.clone(),
                        content: content.clone(),
                        created_at: now,
                    });
                    // Rebuild cached markdown for all comments.
                    self.comments_md = ticket
                        .comments
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, markdown::parse(&c.content).collect()))
                        .collect();
                }

                self.sending_comment = true;
                self.comment_input = text_editor::Content::new();
                self.undo_stack.clear();
                let generation = self.comment_generation;

                Task::perform(
                    async move {
                        let board = crate::board::store();
                        let result = board
                            .add_comment(&ticket_id, &role, &content)
                            .await
                            .map_err(|e| e.to_string());
                        (generation, result)
                    },
                    |(g, result)| BoardMessage::CommentSent(g, result),
                )
            }
            BoardMessage::CommentSent(generation, Ok(())) => {
                // Stale callback guard: if the modal closed or the user switched
                // tickets since this async was dispatched, drop the result.
                if generation != self.comment_generation {
                    self.sending_comment = false;
                    return Task::none();
                }
                self.sending_comment = false;
                // Refresh ticket detail from DB for consistency.
                if let Some(ref ticket) = self.selected_ticket {
                    let refresh = Self::fetch_ticket(ticket.id.clone());
                    let toast = Task::done(BoardMessage::Toast(super::ToastMessage::SuccessMsg(
                        "Comment added.".into(),
                    )));
                    Task::batch([refresh, toast])
                } else {
                    Task::none()
                }
            }
            BoardMessage::CommentSent(generation, Err(e)) => {
                // Stale callback guard
                if generation != self.comment_generation {
                    self.sending_comment = false;
                    return Task::none();
                }
                self.sending_comment = false;
                tracing::warn!(error = %e, "Board: failed to add comment");
                // Remove the optimistic comment by refetching from DB.
                let refetch = if let Some(ref ticket) = self.selected_ticket {
                    Self::fetch_ticket(ticket.id.clone())
                } else {
                    Task::none()
                };
                let toast = Task::done(BoardMessage::Toast(super::ToastMessage::Error(format!(
                    "Failed to add comment: {e}"
                ))));
                Task::batch([refetch, toast])
            }
            BoardMessage::LinkClicked(_)
            | BoardMessage::Toast(_)
            | BoardMessage::ViewCommitDiff { .. } => {
                // Intercepted by the Dashboard before reaching board_state.update().
                // Arms must remain for match exhaustiveness even though functionally dead.
                Task::none()
            }
            BoardMessage::ArchiveAllCompleted => {
                let ws = self.workspace_name.clone();
                Task::perform(
                    async move {
                        let board = crate::board::store();
                        board
                            .archive_all_done_and_cancelled(ws.as_deref())
                            .await
                            .map_err(|e| e.to_string())
                    },
                    BoardMessage::ArchiveAllCompletedResult,
                )
            }
            BoardMessage::ArchiveAllCompletedResult(Ok(count)) => {
                let toast = Task::done(BoardMessage::Toast(super::ToastMessage::SuccessMsg(
                    format!(
                        "Archived {count} ticket{}",
                        if count == 1 { "" } else { "s" }
                    ),
                )));
                Task::batch([self.refresh(), toast])
            }
            BoardMessage::ArchiveAllCompletedResult(Err(e)) => {
                Task::done(BoardMessage::Toast(super::ToastMessage::Error(e)))
            }
            BoardMessage::ArchiveTicket(ticket_id) => {
                self.action_loading = Some(ticket_id.clone());
                Task::perform(
                    async move {
                        let board = crate::board::store();
                        board
                            .set_archived(&ticket_id)
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    },
                    BoardMessage::ActionResult,
                )
            }
            BoardMessage::FetchCommitStats(ticket_id) => {
                let Some(ticket) = &self.selected_ticket else {
                    return Task::none();
                };
                let Some(ref commit_hash) = ticket.commit_hash else {
                    return Task::none();
                };
                let generation = self.commit_stats_generation;
                let ws_name = ticket.workspace_name.clone();
                let hash = commit_hash.clone();
                let id = ticket_id;
                Task::perform(
                    async move {
                        // Resolve workspace name to a filesystem path for git.
                        let ws_path = match crate::workspace::get_by_name(&ws_name).await {
                            Ok(Some(ws)) => ws.path,
                            Ok(None) => {
                                return Err(format!("Workspace '{ws_name}' not found"));
                            }
                            Err(e) => {
                                return Err(format!("{e:#}"));
                            }
                        };
                        Self::run_git_numstat(&ws_path, &hash)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |res| BoardMessage::CommitStatsLoaded(id.clone(), generation, res),
                )
            }
            BoardMessage::CommitStatsLoaded(id, generation, result) => {
                if self.selected_ticket.as_ref().map(|t| t.id.as_str()) != Some(id.as_str())
                    || generation != self.commit_stats_generation
                {
                    // Stale callback — ticket changed or modal reopened
                    return Task::none();
                }
                self.commit_stats_loading = false;
                match result {
                    Ok(stats) => {
                        self.commit_stats = Some(stats);
                    }
                    Err(_) => {
                        // Non-critical: silently leave stats as None
                        self.commit_stats = None;
                    }
                }
                Task::none()
            }

            // ── Ticket search (sidebar) ──────────────────────────────
            BoardMessage::SearchInputChanged(query) => {
                self.search_query.clone_from(&query);
                self.search_generation += 1;

                if query.is_empty() {
                    self.search_results.clear();
                    return Task::none();
                }

                let generation = self.search_generation;
                let ws = self.workspace_name.clone();
                Task::perform(
                    async move {
                        // Debounce: wait 300ms before executing the FTS query
                        // so rapid typing doesn't trigger per-keystroke DB hits.
                        //
                        // Effective latency is 300–1300ms depending on Tick
                        // timing (the 1-second Iced Tick that drives view
                        // updates acts as the lower bound for the next render
                        // after results arrive). In-flight tasks from earlier
                        // keystrokes are NOT cancelled — they run to completion
                        // and their results are discarded by the generation
                        // counter guard below. For typical typing speeds this
                        // creates at most a handful of short-lived sleep-only
                        // tasks, which is acceptably cheap.
                        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                        let board = crate::board::store();
                        board
                            .search_active_by_fts(&query, 20, ws.as_deref())
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |res| match res {
                        Ok(tickets) => BoardMessage::SearchResults(tickets, generation),
                        Err(_) => BoardMessage::SearchResults(vec![], generation),
                    },
                )
            }
            BoardMessage::SearchResults(tickets, generation) => {
                if generation == self.search_generation {
                    // Apply only if still the latest generation (stale callback guard)
                    self.search_results = tickets;
                }
                Task::none()
            }
            BoardMessage::SearchCleared => {
                self.search_query.clear();
                self.search_results.clear();
                self.search_generation += 1;
                Task::none()
            }
        }
    }

    /// Run `git show --numstat` (or `-m --numstat` for merges) and parse the output.
    async fn run_git_numstat(
        ws_path: &str,
        commit_hash: &str,
    ) -> Result<CommitStats, anyhow::Error> {
        // Detect merge commits: `git rev-list --parents -n 1 <hash>` outputs
        // `<hash> <parent>` for non-merge, or `<hash> <parent1> <parent2> ...` for merges.
        let is_merge = match run_git_output(
            Path::new(ws_path),
            &["rev-list", "--parents", "-n", "1", commit_hash],
        )
        .await
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.split_whitespace().collect();
                parts.len() > 2 // hash + parent1 + parent2... = merge
            }
            Err(_) => false, // if rev-list fails, assume non-merge
        };

        let mut args: Vec<&str> = vec!["show", "--numstat", "--format="];
        if is_merge {
            args.push("-m");
        }
        args.push(commit_hash);

        let output = run_git_output(Path::new(ws_path), &args)
            .await
            .map_err(|e| anyhow::anyhow!("git show failed: {e}"))?;

        if !output.status.success() {
            anyhow::bail!(
                "git show failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut files: Vec<FileStat> = Vec::new();

        for entry in parse_numstat_lines(&stdout) {
            // Skip binary files (additions/deletions are None)
            let Some(additions) = entry.additions else {
                continue;
            };
            let Some(deletions) = entry.deletions else {
                continue;
            };
            // Skip rename-only (0 additions, 0 deletions)
            if additions == 0 && deletions == 0 {
                continue;
            }

            files.push(FileStat {
                path: entry.path,
                additions,
                deletions,
            });
        }

        Ok(CommitStats { files })
    }

    /// Partition tickets into the three kanban columns.
    pub(crate) fn partition_tickets(
        tickets: &[Ticket],
    ) -> (Vec<&Ticket>, Vec<&Ticket>, Vec<&Ticket>) {
        let mut pending = Vec::new();
        let mut pipeline = Vec::new();
        let mut completed = Vec::new();

        for ticket in tickets {
            if ticket.is_archived {
                continue; // hidden from board
            }
            match ticket.phase {
                TicketPhase::Backlog
                | TicketPhase::Analysis
                | TicketPhase::Planning
                | TicketPhase::Failed => pending.push(ticket),
                TicketPhase::ReadyForDevelopment
                | TicketPhase::InDevelopment
                | TicketPhase::InDiagnostics
                | TicketPhase::DiagnosticsDone
                | TicketPhase::InSanitation
                | TicketPhase::SanitationPassed
                | TicketPhase::InReview
                | TicketPhase::Reviewed
                | TicketPhase::InQa
                | TicketPhase::QaPassed => pipeline.push(ticket),
                TicketPhase::Done | TicketPhase::Cancelled => completed.push(ticket),
            }
        }

        // Sort: pending and pipeline by priority (ASC), then oldest-first (ASC);
        // completed newest-first (DESC).
        // Priority is an integer — 0 = highest, so ASC puts urgent tickets first.
        // Ticket created_at is an ISO 8601 string, so lexical sort = chronological sort
        pending.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then(a.created_at.cmp(&b.created_at))
        });
        pipeline.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then(a.created_at.cmp(&b.created_at))
        });
        completed.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        (pending, pipeline, completed)
    }

    /// Render a single ticket card: clickable title, ID, phase badge, and action icons.
    #[allow(clippy::too_many_lines)]
    pub fn render_ticket_card<'a>(&'a self, ticket: &'a Ticket) -> Element<'a, BoardMessage> {
        let is_action_disabled = self.action_loading.as_deref() == Some(&ticket.id);

        let actions = Self::available_actions(ticket.phase);
        let icon_row = Self::action_icon_row(&ticket.id, &actions, is_action_disabled);

        let (unfulfilled_count, unfulfilled_ids) = self.unfulfilled_prereq_count(ticket);

        let mut badge_row = row![
            Self::priority_badge(ticket.priority, 10, [1, 6]),
            Self::phase_badge(ticket.phase, 10, [1, 6]),
        ]
        .spacing(6);

        if unfulfilled_count > 0 {
            let tooltip_text = format!("Blocked by: {}", unfulfilled_ids.join(", "));
            let pause_icon = lucide::pause::<iced::Theme, iced::Renderer>()
                .size(12)
                .color(theme::STATUS_WARNING);
            let count_text = text(format!("{unfulfilled_count}"))
                .size(10)
                .color(theme::STATUS_WARNING);
            let indicator = row![pause_icon, count_text]
                .spacing(2)
                .align_y(Alignment::Center);
            badge_row = badge_row.push(
                tooltip(
                    indicator,
                    text(tooltip_text).size(11),
                    tooltip::Position::Top,
                )
                .style(theme::tooltip_style),
            );
        }

        // Inline commit stats: +added/−removed with color coding,
        // positioned after prereq indicator and before fill spacer.
        // Zero-valued sides are hidden; only the non-zero side displays.
        if let (Some(hash), Some(ws_name)) = (&ticket.commit_hash, &self.workspace_name) {
            let added = ticket.lines_added.unwrap_or(0);
            let removed = ticket.lines_removed.unwrap_or(0);
            let stats_parts = row![
                text("\u{2387} ").size(10).color(theme::TEXT_SECONDARY),
                diff_stats_row::<BoardMessage>(added, removed, 10.0),
            ]
            .spacing(0)
            .align_y(Alignment::Center);
            let stats_button = button(stats_parts)
                .padding([2, 6])
                .style(theme::button_text)
                .on_press(BoardMessage::ViewCommitDiff {
                    commit_hash: hash.clone(),
                    workspace_name: ws_name.clone(),
                });
            badge_row = badge_row.push(stats_button);
        }

        badge_row = badge_row.push(Space::new().width(Length::Fill));
        badge_row = badge_row.push(icon_row);

        // Per-ticket archive button for done/cancelled tickets
        if matches!(ticket.phase, TicketPhase::Done | TicketPhase::Cancelled) && !ticket.is_archived
        {
            let archive_btn = button(
                lucide::archive::<iced::Theme, iced::Renderer>()
                    .size(16)
                    .color(theme::TEXT_MUTED),
            )
            .style(theme::button_text)
            .on_press_maybe(if is_action_disabled {
                None
            } else {
                Some(BoardMessage::ArchiveTicket(ticket.id.clone()))
            });
            badge_row = badge_row.push(archive_btn);
        }

        let mut card_children: Vec<Element<'_, BoardMessage>> = vec![
            // Title + ID row: both clickable
            button(
                column![
                    text(&ticket.title).size(13).color(theme::TEXT_PRIMARY),
                    text(&ticket.id).size(10).color(theme::TEXT_MUTED),
                ]
                .spacing(2),
            )
            .padding(iced::Padding::new(8.0).bottom(2.0))
            .width(Length::Fill)
            .style(theme::button_text)
            .on_press(BoardMessage::OpenModal(ticket.id.clone()))
            .into(),
        ];

        // Badge + optional prereq indicator + icon row (below the clickable area)
        card_children.push(badge_row.align_y(Alignment::Center).padding([0, 8]).into());

        let card = Column::from_vec(card_children)
            .spacing(0)
            .width(Length::Fill);

        container(card)
            .style(theme::elevated_card_style)
            .width(Length::Fill)
            .into()
    }

    /// Whether a ticket detail modal is currently open (or loading).
    #[must_use]
    pub const fn is_modal_open(&self) -> bool {
        self.selected_ticket.is_some() || self.selected_loading || self.detail_error.is_some()
    }

    /// Build a centered dialog with a semi-transparent backdrop that closes on click.
    fn centered_dialog<'a>(
        content: impl Into<Element<'a, BoardMessage>>,
        on_backdrop: BoardMessage,
    ) -> Element<'a, BoardMessage> {
        widget_helpers::modal_backdrop(content, on_backdrop, 0.5)
    }

    /// Render the modal overlay for ticket detail.
    /// Includes the empty-case placeholder for `Stack` widget type stability.
    #[must_use]
    pub fn render_modal_overlay(&self) -> Element<'_, BoardMessage> {
        if self.is_modal_open() {
            if self.selected_loading {
                let dialog = container(
                    column![
                        text("Loading details...")
                            .size(16)
                            .color(theme::TEXT_SECONDARY),
                        Space::new().height(12),
                        text("Fetching ticket information\u{2026}")
                            .size(13)
                            .color(theme::TEXT_SECONDARY),
                    ]
                    .align_x(Alignment::Center),
                )
                .width(Length::Fixed(400.0))
                .padding(24)
                .style(theme::dialog_container_style);

                Self::centered_dialog(dialog, BoardMessage::CloseModal)
            } else if self.selected_ticket.is_none()
                && let Some(ref err) = self.detail_error
            {
                let dialog = container(
                    column![
                        text("Failed to load ticket")
                            .size(16)
                            .color(theme::STATUS_ERROR),
                        Space::new().height(12),
                        text(err).size(13).color(theme::TEXT_SECONDARY),
                        Space::new().height(16),
                        button(
                            text("Close")
                                .size(13)
                                .color(theme::TEXT_PRIMARY)
                                .align_x(Alignment::Center),
                        )
                        .style(theme::button_secondary)
                        .on_press(BoardMessage::CloseModal),
                    ]
                    .align_x(Alignment::Center),
                )
                .width(Length::Fixed(400.0))
                .padding(24)
                .style(theme::dialog_container_style);

                Self::centered_dialog(dialog, BoardMessage::CloseModal)
            } else {
                let detail = self.modal_detail();
                let dialog = container(detail)
                    .width(Length::Fixed(600.0))
                    .padding(24)
                    .style(theme::dialog_container_style);

                Self::centered_dialog(dialog, BoardMessage::CloseModal)
            }
        } else {
            // Keep Stack widget type stable to prevent MouseArea state
            // from becoming orphaned when the modal closes.
            iced::widget::stack([widget_helpers::empty_stack_placeholder()]).into()
        }
    }

    /// Render the ticket detail modal content.
    fn modal_detail(&self) -> Element<'_, BoardMessage> {
        let Some(ticket) = &self.selected_ticket else {
            return text("No ticket selected.")
                .size(13)
                .color(theme::TEXT_MUTED)
                .into();
        };

        let is_action_disabled = self.action_loading.as_deref() == Some(&ticket.id);

        let mut sections: Vec<Element<'_, BoardMessage>> = Vec::new();
        sections.push(Self::render_header_metadata(ticket, is_action_disabled));

        if let Some(el) = Self::render_commit_stats(
            ticket,
            self.commit_stats.as_ref(),
            self.commit_stats_loading,
        ) {
            sections.push(el);
        }

        if let Some(el) = Self::render_description(ticket, self.description_md.as_deref()) {
            sections.push(Space::new().height(8).into());
            sections.push(el);
        }

        if let Some(el) = Self::render_comments(ticket, &self.expanded_comments, &self.comments_md)
        {
            sections.push(Space::new().height(12).into());
            sections.push(
                text("Comments:")
                    .size(13)
                    .color(theme::TEXT_SECONDARY)
                    .into(),
            );
            sections.push(el);
        }

        // ── Scrollable content area ──────────────────────────────
        let scrollable_content =
            scrollable(Column::from_vec(sections).spacing(4).width(Length::Fill))
                .width(Length::Fill)
                .height(Length::Fill)
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style);

        // ── Comment input area (pinned at the bottom) ────────────
        let input_area = self.render_comment_input();

        column![scrollable_content.width(Length::Fill), input_area,]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Render the modal header: title, ticket ID, phase badge, action icons, and metadata lines.
    fn render_header_metadata(
        ticket: &Ticket,
        is_action_disabled: bool,
    ) -> Element<'_, BoardMessage> {
        let actions = Self::available_actions(ticket.phase);
        let icon_row = Self::action_icon_row(&ticket.id, &actions, is_action_disabled);

        let reporter_display = if ticket.reporter.is_empty() {
            "Legacy".to_string()
        } else {
            Role::from_str(&ticket.reporter).map_or_else(
                |_| {
                    let mut chars = ticket.reporter.chars();
                    let first = chars.next().expect("non-empty checked above");
                    first.to_uppercase().to_string() + chars.as_str()
                },
                |role| crate::role::role_info(&role).display_label.to_string(),
            )
        };
        let created = theme::format_timestamp(&ticket.created_at);
        let updated = theme::format_timestamp(&ticket.updated_at);

        let meta_els: Vec<Element<'_, BoardMessage>> = vec![
            text(format!("Created: {created}"))
                .size(12)
                .color(theme::TEXT_MUTED)
                .into(),
            text(" · ").size(12).color(theme::TEXT_MUTED).into(),
            text(format!("Updated: {updated}"))
                .size(12)
                .color(theme::TEXT_MUTED)
                .into(),
            text(" · ").size(12).color(theme::TEXT_MUTED).into(),
            text(format!("Reporter: {reporter_display}"))
                .size(12)
                .color(theme::TEXT_MUTED)
                .into(),
        ];

        let mut secondary: Vec<String> = Vec::new();
        if let Some(ref assignee) = ticket.assigned_to {
            secondary.push(format!("Assigned: {assignee}"));
        }
        if !ticket.prerequisites.is_empty() {
            secondary.push(format!(
                "Prerequisites: {}",
                ticket.prerequisites.join(", ")
            ));
        }
        if let Some(ref supersedes) = ticket.supersedes {
            secondary.push(format!("Supersedes: {supersedes}"));
        }
        if let Some(ref superseded_by) = ticket.superseded_by {
            secondary.push(format!("Superseded by: {superseded_by}"));
        }

        let first_row: Element<'_, BoardMessage> =
            Row::from_vec(meta_els).align_y(Alignment::Center).into();

        let metadata_block: Element<'_, BoardMessage> = if secondary.is_empty() {
            first_row
        } else {
            let second_row: Element<'_, BoardMessage> = text(secondary.join(" · "))
                .size(12)
                .color(theme::TEXT_MUTED)
                .into();
            column![first_row, second_row].spacing(2).into()
        };

        column![
            row![
                text(&ticket.title)
                    .size(16)
                    .color(theme::TEXT_PRIMARY)
                    .font(theme::FONT_BOLD),
                Space::new().width(Length::Fill),
                button(
                    lucide::x::<iced::Theme, iced::Renderer>()
                        .size(16)
                        .color(theme::TEXT_MUTED),
                )
                .style(theme::button_text)
                .on_press(BoardMessage::CloseModal),
            ]
            .align_y(Alignment::Center),
            text(&ticket.id).size(12).color(theme::TEXT_MUTED),
            Space::new().height(6),
            row![
                Self::priority_badge(ticket.priority, 12, [2, 8]),
                Self::phase_badge(ticket.phase, 12, [2, 8]),
                Space::new().width(Length::Fill),
                icon_row,
            ]
            .align_y(Alignment::Center)
            .spacing(8)
            .padding([4, 0]),
            metadata_block,
        ]
        .spacing(4)
        .into()
    }

    /// Render commit stats: summary header + per-file rows, or loading indicator.
    /// Returns `None` when the ticket has no commit hash.
    #[allow(clippy::too_many_lines)]
    fn render_commit_stats<'a>(
        ticket: &'a Ticket,
        stats: Option<&'a CommitStats>,
        loading: bool,
    ) -> Option<Element<'a, BoardMessage>> {
        let hash = ticket.commit_hash.as_ref()?;

        if loading {
            return Some(
                column![
                    Space::new().height(8),
                    text("Loading commit stats\u{2026}")
                        .size(12)
                        .color(theme::TEXT_MUTED),
                ]
                .spacing(4)
                .into(),
            );
        }

        let stats = stats?;

        let total_additions: i64 = stats.files.iter().map(|f| f.additions).sum();
        let total_deletions: i64 = stats.files.iter().map(|f| f.deletions).sum();
        let file_count = stats.files.len();
        let mut summary_parts: Vec<Element<'_, BoardMessage>> = Vec::new();
        summary_parts.push(
            text(format!("{hash:.8}"))
                .size(11)
                .color(theme::TEXT_MUTED)
                .into(),
        );
        summary_parts.push(Space::new().width(6).into());
        summary_parts.push(
            text(format!(
                "{file_count} file{} changed",
                if file_count == 1 { "" } else { "s" }
            ))
            .size(11)
            .color(theme::TEXT_SECONDARY)
            .into(),
        );
        if total_additions > 0 {
            summary_parts.push(
                text(format!("+{total_additions}"))
                    .size(11)
                    .color(theme::STATUS_SUCCESS)
                    .into(),
            );
        }
        if total_deletions > 0 {
            summary_parts.push(
                text(format!("\u{2212}{total_deletions}"))
                    .size(11)
                    .color(theme::STATUS_ERROR)
                    .into(),
            );
        }
        let summary_header = container(row(summary_parts).spacing(4).align_y(Alignment::Center))
            .padding([4, 8])
            .width(Length::Fill);

        // File stat rows — hide zero-valued sides
        let mut file_col = Column::new().spacing(2);
        for f in &stats.files {
            let mut row_parts: Vec<Element<'_, BoardMessage>> = vec![
                container(text(&f.path).size(11).font(theme::FONT_REGULAR))
                    .width(Length::Fixed(400.0))
                    .clip(true)
                    .into(),
                Space::new().width(Length::Fill).into(),
            ];
            if f.additions > 0 {
                row_parts.push(
                    text(format!("+{}", f.additions))
                        .size(11)
                        .font(theme::FONT_REGULAR)
                        .color(theme::STATUS_SUCCESS)
                        .into(),
                );
            }
            if f.additions > 0 && f.deletions > 0 {
                row_parts.push(Space::new().width(6).into());
            }
            if f.deletions > 0 {
                row_parts.push(
                    text(format!("-{}", f.deletions))
                        .size(11)
                        .font(theme::FONT_REGULAR)
                        .color(theme::STATUS_ERROR)
                        .into(),
                );
            }
            let row = row(row_parts).align_y(Alignment::Center);
            file_col = file_col.push(row);
        }

        Some(
            column![
                summary_header,
                container(file_col)
                    .padding([4, 8])
                    .style(theme::surface_card_style),
            ]
            .spacing(4)
            .into(),
        )
    }

    /// Render the description section: markdown (if cached) or plain text.
    /// Returns `None` when the ticket description is empty.
    fn render_description<'a>(
        ticket: &'a Ticket,
        description_md: Option<&'a [markdown::Item]>,
    ) -> Option<Element<'a, BoardMessage>> {
        if ticket.description.is_empty() {
            return None;
        }

        Some(if let Some(items) = description_md {
            container(
                scrollable(
                    markdown::view(items, theme::markdown_settings())
                        .map(BoardMessage::LinkClicked),
                )
                .width(Length::Fill)
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style),
            )
            .width(Length::Fill)
            .padding(8)
            .style(theme::surface_card_style)
            .into()
        } else {
            container(selectable_text(&ticket.description, theme::TEXT_PRIMARY).size(13))
                .padding(8)
                .style(theme::surface_card_style)
                .into()
        })
    }

    /// Render the comment input widget (text editor + send button) pinned at
    /// the bottom of the ticket detail modal.
    fn render_comment_input(&self) -> Element<'_, BoardMessage> {
        container(super::widgets::chat_composer(
            &self.comment_input,
            BoardMessage::CommentInputChanged,
            BoardMessage::SendComment,
            "Add a comment… (Enter to send, Shift+Enter for newline)",
            self.sending_comment,
            44.0,
            132.0,
        ))
        .width(Length::Fill)
        .into()
    }

    /// Render the comments list: per-comment role badge, timestamp, content,
    /// and diagnostics expand/collapse toggle.
    /// Returns `None` when the ticket has no comments.
    fn render_comments<'a>(
        ticket: &'a Ticket,
        expanded: &'a HashSet<usize>,
        comments_md: &'a [(usize, Vec<markdown::Item>)],
    ) -> Option<Element<'a, BoardMessage>> {
        if ticket.comments.is_empty() {
            return None;
        }

        let mut cmt_col = Column::new().spacing(4);
        for (i, comment) in ticket.comments.iter().enumerate().rev() {
            let role_color = theme::role_badge_color(&comment.role).0;

            // For diagnostics comments, optionally show only the summary
            let is_diag = comment.role == crate::role::DIAGNOSTICS_ROLE;
            let is_expanded = expanded.contains(&i);

            let summary = if is_diag {
                comment
                    .content
                    .rfind("\n---\n")
                    .map(|pos| &comment.content[pos + 5..])
            } else {
                None
            };

            let comment_content: Element<'_, BoardMessage> = if is_diag && !is_expanded {
                selectable_text(
                    summary.unwrap_or(&comment.content).trim(),
                    theme::TEXT_PRIMARY,
                )
                .size(13)
                .into()
            } else if let Some((_, items)) = comments_md.iter().find(|(idx, _)| *idx == i) {
                markdown::view(items, theme::markdown_settings()).map(BoardMessage::LinkClicked)
            } else {
                selectable_text(&comment.content, theme::TEXT_PRIMARY)
                    .size(13)
                    .into()
            };

            // Toggle button for diagnostics comments
            let toggle_button: Option<Element<'_, BoardMessage>> = if is_diag {
                let (icon, label) = if is_expanded {
                    (
                        lucide::chevron_up::<iced::Theme, iced::Renderer>().size(12),
                        " Collapse",
                    )
                } else {
                    (
                        lucide::chevron_down::<iced::Theme, iced::Renderer>().size(12),
                        " Show full output",
                    )
                };
                Some(
                    button(
                        row![
                            icon.color(theme::TEXT_SECONDARY),
                            text(label).size(11).color(theme::TEXT_SECONDARY),
                        ]
                        .spacing(2)
                        .align_y(Alignment::Center),
                    )
                    .style(theme::button_text)
                    .on_press(BoardMessage::ToggleCommentExpand(i))
                    .into(),
                )
            } else {
                None
            };

            let mut comment_col = Column::new().spacing(4);
            comment_col = comment_col.push(
                row![
                    container(text(&comment.role).size(11).color(role_color))
                        .padding([1, 6])
                        .style(move |t| theme::role_badge_pill_style(t, role_color)),
                    Space::new().width(8),
                    text(theme::format_timestamp(&comment.created_at))
                        .size(10)
                        .color(theme::TEXT_MUTED),
                ]
                .align_y(Alignment::Center),
            );
            comment_col = comment_col.push(comment_content);
            if let Some(btn) = toggle_button {
                comment_col = comment_col.push(btn);
            }

            cmt_col = cmt_col.push(
                container(comment_col)
                    .width(Length::Fill)
                    .padding(8)
                    .style(theme::surface_card_style),
            );
        }

        Some(cmt_col.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::Point;

    fn make_board_state() -> BoardState {
        let mut state = BoardState::new();
        state.current_user_name = Some("admin".into());
        state.selected_ticket = Some(Ticket {
            id: "T-1".into(),
            title: "Test ticket".into(),
            description: String::new(),
            phase: TicketPhase::Backlog,
            assigned_to: None,
            workspace_name: "test_ws".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            comments: Vec::new(),
            prerequisites: Vec::new(),
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            pipeline_reservation: false,
            priority: 0,
            reviewed_head: None,
            reviewed_tree: None,
        });
        state
    }

    // ── SendComment: empty rejection ──────────────────────────────

    #[test]
    fn test_comment_empty_is_noop() {
        let mut state = make_board_state();
        // Empty content
        state.comment_input = text_editor::Content::new();
        let _task = state.update(BoardMessage::SendComment);
        assert!(!state.sending_comment);
        assert!(state.comment_input.text().is_empty());

        // Whitespace-only should also be treated as empty.
        state.comment_input = text_editor::Content::with_text("   ");
        let _task = state.update(BoardMessage::SendComment);
        assert!(!state.sending_comment);
    }

    // ── SendComment: character limit ──────────────────────────────

    #[test]
    fn test_comment_within_limit_clears_input() {
        let mut state = make_board_state();
        state.comment_input = text_editor::Content::with_text("hello world");
        let _task = state.update(BoardMessage::SendComment);
        // Editor must be cleared on accepted message
        assert!(
            state.comment_input.text().is_empty(),
            "input should be cleared after accepting a within-limit comment"
        );
    }

    #[test]
    fn test_comment_at_limit_sends() {
        let mut state = make_board_state();
        let text = "a".repeat(MAX_INPUT_CHARS);
        state.comment_input = text_editor::Content::with_text(&text);
        let _task = state.update(BoardMessage::SendComment);
        // Exactly at the limit: should be accepted (editor cleared).
        assert!(
            state.comment_input.text().is_empty(),
            "input should be cleared when comment is exactly at the limit"
        );
    }

    #[test]
    fn test_comment_exceeds_limit_preserves_content() {
        let mut state = make_board_state();
        let long_text = "a".repeat(MAX_INPUT_CHARS + 1);
        state.comment_input = text_editor::Content::with_text(&long_text);
        let _task = state.update(BoardMessage::SendComment);
        // Editor content must be preserved — user needs to edit it down.
        assert_eq!(
            state.comment_input.text(),
            long_text,
            "input must be preserved when comment exceeds character limit"
        );
        assert!(
            !state.sending_comment,
            "sending_comment should remain false after rejected comment"
        );
    }

    // ── SendComment: double-send guard ────────────────────────────

    #[test]
    fn test_comment_double_send_guard() {
        let mut state = make_board_state();
        state.sending_comment = true;
        state.comment_input = text_editor::Content::with_text("hello");
        let _task = state.update(BoardMessage::SendComment);
        // When sending_comment is true, the message should not be processed.
        assert!(
            !state.comment_input.text().is_empty(),
            "input should not be cleared when double-send guard prevents sending"
        );
        assert!(
            state.sending_comment,
            "sending_comment should remain true when guard was active"
        );
    }

    // ── SendComment: missing user ─────────────────────────────────

    #[test]
    fn test_comment_no_user_returns_warning() {
        let mut state = make_board_state();
        state.current_user_name = None;
        state.comment_input = text_editor::Content::with_text("hello");
        let _task = state.update(BoardMessage::SendComment);

        // Editor should NOT be cleared — message was rejected.
        assert_eq!(state.comment_input.text(), "hello");
        assert!(!state.sending_comment);
    }

    // ── SendComment: no ticket selected ───────────────────────────

    #[test]
    fn test_comment_no_ticket_is_noop() {
        let mut state = make_board_state();
        state.selected_ticket = None;
        state.comment_input = text_editor::Content::with_text("hello");
        let _task = state.update(BoardMessage::SendComment);
        assert!(!state.sending_comment);
        assert_eq!(
            state.comment_input.text(),
            "hello",
            "input should not be cleared when no ticket is selected"
        );
    }

    // ── CommentInputChanged: cursor actions pass through ──────────

    #[test]
    fn test_comment_input_changed_non_edit_passes_through() {
        let mut state = make_board_state();
        state.comment_input = text_editor::Content::with_text("hello world");

        // Cursor movement (non-edit action) must be passed to perform().
        // After a Click, the cursor position should reflect the click.
        state.update(BoardMessage::CommentInputChanged(
            text_editor::Action::Click(Point { x: 0.0, y: 0.0 }),
        ));
        // perform() was called — the editor state accepted the action
        // (no crash, content unchanged).
        assert_eq!(state.comment_input.text(), "hello world");
        assert!(
            state.comment_focused,
            "comment_focused should be set to true on any input action"
        );
    }

    // ── CommentInputChanged: Shift+Click → Drag conversion ───────

    #[test]
    fn test_comment_shift_click_converts_to_drag() {
        let mut state = make_board_state();
        state.comment_input = text_editor::Content::with_text("hello world");
        // Set Shift modifier
        state.modifiers = keyboard::Modifiers::SHIFT;

        // Click at position should be converted to Drag (which extends selection).
        // This is a behavioral contract — we can't easily assert the internal
        // conversion happened, but we can verify no crash and that the editor
        // state was updated.
        state.update(BoardMessage::CommentInputChanged(
            text_editor::Action::Click(Point { x: 0.0, y: 0.0 }),
        ));
        assert!(
            state.comment_focused,
            "comment_focused should be true after Click with Shift"
        );
    }

    // ── CommentInputChanged: without Shift, Click stays Click ─────

    #[test]
    fn test_comment_click_without_shift_stays_click() {
        let mut state = make_board_state();
        state.comment_input = text_editor::Content::with_text("hello world");
        // No modifier — Click should remain as Click
        state.modifiers = keyboard::Modifiers::empty();

        state.update(BoardMessage::CommentInputChanged(
            text_editor::Action::Click(Point { x: 0.0, y: 0.0 }),
        ));
        // No crash — content unchanged.
        assert_eq!(state.comment_input.text(), "hello world");
    }

    // ── CommentModifiersChanged: updates state ────────────────────

    #[test]
    fn test_comment_modifiers_changed_updates_state() {
        let mut state = make_board_state();
        assert_eq!(state.modifiers, keyboard::Modifiers::empty());

        state.update(BoardMessage::CommentModifiersChanged(
            keyboard::Modifiers::SHIFT,
        ));
        assert_eq!(state.modifiers, keyboard::Modifiers::SHIFT);

        state.update(BoardMessage::CommentModifiersChanged(
            keyboard::Modifiers::CTRL,
        ));
        assert_eq!(state.modifiers, keyboard::Modifiers::CTRL);
    }

    // ── OpenModal bumps generation ────────────────────────────────

    #[test]
    fn test_open_modal_bumps_comment_generation() {
        let mut state = make_board_state();
        let gen_before = state.comment_generation;
        let _task = state.update(BoardMessage::OpenModal("T-2".into()));
        assert!(
            state.comment_generation > gen_before,
            "comment_generation should be bumped on OpenModal"
        );
    }

    // ── TicketDetails bumps generation and clears input ───────────

    #[test]
    fn test_ticket_details_clears_comment_input() {
        let mut state = make_board_state();
        state.comment_input = text_editor::Content::with_text("draft comment");
        state.sending_comment = true;
        state.comment_focused = true;

        let gen_before = state.comment_generation;
        let ticket = Ticket {
            id: "T-2".into(),
            title: "New ticket".into(),
            description: String::new(),
            phase: TicketPhase::Backlog,
            assigned_to: None,
            workspace_name: "test_ws".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            comments: Vec::new(),
            prerequisites: Vec::new(),
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            pipeline_reservation: false,
            priority: 0,
            reviewed_head: None,
            reviewed_tree: None,
        };
        let _task = state.update(BoardMessage::TicketDetails(Box::new(ticket)));

        assert!(
            state.comment_input.text().is_empty(),
            "comment input should be cleared when switching tickets"
        );
        assert!(
            !state.sending_comment,
            "sending_comment should be reset when switching tickets"
        );
        assert!(
            !state.comment_focused,
            "comment_focused should be reset when switching tickets"
        );
        assert!(
            state.comment_generation > gen_before,
            "comment_generation should be bumped on ticket switch"
        );
    }

    // ── Escape: first blurs, second closes ────────────────────────

    #[test]
    fn test_escape_first_blurs_then_closes() {
        let mut state = make_board_state();
        state.comment_focused = true;
        state.selected_ticket = Some(Ticket {
            id: "T-1".into(),
            title: "Test".into(),
            description: String::new(),
            phase: TicketPhase::Backlog,
            assigned_to: None,
            workspace_name: "test_ws".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            comments: Vec::new(),
            prerequisites: Vec::new(),
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            pipeline_reservation: false,
            priority: 0,
            reviewed_head: None,
            reviewed_tree: None,
        });

        // First Escape: focused → blur (clear flag), modal stays open
        let _task = state.update(BoardMessage::Escape);
        assert!(
            !state.comment_focused,
            "comment_focused should be cleared on first Escape"
        );
        assert!(
            state.selected_ticket.is_some(),
            "modal should remain open on first Escape"
        );

        // Second Escape: not focused → close modal
        let _task = state.update(BoardMessage::Escape);
        assert!(
            state.selected_ticket.is_none(),
            "modal should close on second Escape"
        );
    }

    // ── Undo / Redo for comment input ────────────────────────────

    #[test]
    fn test_comment_undo_restores_content() {
        let mut state = make_board_state();
        // Simulate typing "abc" by performing edit actions.
        for ch in &['a', 'b', 'c'] {
            let _task = state.update(BoardMessage::CommentInputChanged(
                text_editor::Action::Edit(text_editor::Edit::Insert(*ch)),
            ));
        }
        let text_before = state.comment_input.text();
        assert_eq!(text_before, "abc", "input should have typed text");

        // Undo restores previous state.
        let _task = state.update(BoardMessage::Undo);
        let text_after_undo = state.comment_input.text();
        assert_ne!(
            text_before, text_after_undo,
            "undo should change the content"
        );
        // Redo restores the undone state.
        let _task = state.update(BoardMessage::Redo);
        assert_eq!(
            state.comment_input.text(),
            text_before,
            "redo should restore undone content"
        );
    }

    #[test]
    fn test_comment_redo_no_undo_is_noop() {
        let mut state = make_board_state();
        let text = state.comment_input.text();
        let _task = state.update(BoardMessage::Redo);
        assert_eq!(
            state.comment_input.text(),
            text,
            "redo with no undo history should be a no-op"
        );
    }

    #[test]
    fn test_comment_undo_cleared_on_reset_modal() {
        let mut state = make_board_state();
        // Build some history
        let _task = state.update(BoardMessage::CommentInputChanged(
            text_editor::Action::Edit(text_editor::Edit::Insert('h')),
        ));
        // Simulate more typing to build real undo history
        let _task = state.update(BoardMessage::CommentInputChanged(
            text_editor::Action::Edit(text_editor::Edit::Insert('e')),
        ));
        let _task = state.update(BoardMessage::CommentInputChanged(
            text_editor::Action::Edit(text_editor::Edit::Insert('l')),
        ));
        state.reset_modal();
        // After reset, undo should be a no-op.
        let text = state.comment_input.text();
        let _task = state.update(BoardMessage::Undo);
        assert_eq!(
            state.comment_input.text(),
            text,
            "undo should be no-op after modal reset"
        );
    }

    // ── CommentSent: stale callback guard ─────────────────────────
    //
    // Note: there is only a single stale-callback test because the
    // generation guard fires *before* the Result branch (Ok vs Err),
    // so both variants exercise identical code. A single Ok test
    // covers both.

    #[test]
    fn test_comment_sent_stale_callback() {
        let mut state = make_board_state();
        state.sending_comment = true;
        state.comment_generation = 42;

        // Callback with wrong generation should be dropped gracefully
        // regardless of Ok/Err (the stale check fires before the branch).
        let _task = state.update(BoardMessage::CommentSent(0, Ok(())));
        assert!(
            !state.sending_comment,
            "sending_comment should be reset in stale callback"
        );
    }

    // ── Ticket search (sidebar) ──────────────────────────────────

    #[test]
    fn test_search_empty_input_clears_results() {
        let mut state = make_board_state();
        state.search_query = "old".into();
        state.search_results = vec![Ticket {
            id: "T-1".into(),
            title: "Old result".into(),
            description: String::new(),
            phase: TicketPhase::Backlog,
            assigned_to: None,
            workspace_name: "test_ws".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            comments: Vec::new(),
            prerequisites: Vec::new(),
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            pipeline_reservation: false,
            priority: 0,
            reviewed_head: None,
            reviewed_tree: None,
        }];
        state.search_generation = 5;

        let _task = state.update(BoardMessage::SearchInputChanged(String::new()));

        assert!(state.search_query.is_empty());
        assert!(state.search_results.is_empty());
        // Generation bumped even on empty — invalidates any in-flight tasks
        assert_eq!(state.search_generation, 6);
    }

    #[test]
    fn test_search_non_empty_stores_query_and_bumps_generation() {
        let mut state = make_board_state();
        state.search_generation = 3;

        let _task = state.update(BoardMessage::SearchInputChanged("network".into()));

        assert_eq!(state.search_query, "network");
        // Generation bumped so stale callbacks are discarded
        assert!(
            state.search_generation > 3,
            "generation should be incremented"
        );
    }

    #[test]
    fn test_search_results_stale_callback_discarded() {
        let mut state = make_board_state();
        state.search_generation = 99;

        // A stale result (generation 50 < current 99) should be ignored
        let _task = state.update(BoardMessage::SearchResults(
            vec![Ticket {
                id: "T-stale".into(),
                title: "Stale".into(),
                description: String::new(),
                phase: TicketPhase::Backlog,
                assigned_to: None,
                workspace_name: "test_ws".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                updated_at: "2026-01-01T00:00:00Z".into(),
                comments: Vec::new(),
                prerequisites: Vec::new(),
                supersedes: None,
                superseded_by: None,
                commit_hash: None,
                lines_added: None,
                lines_removed: None,
                reporter: "test".into(),
                is_archived: false,
                pipeline_reservation: false,
                priority: 0,
                reviewed_head: None,
                reviewed_tree: None,
            }],
            50,
        ));

        assert!(
            state.search_results.is_empty(),
            "stale callback should not populate results"
        );
    }

    #[test]
    fn test_search_results_current_generation_accepted() {
        let mut state = make_board_state();
        state.search_query = "network".into();
        state.search_generation = 42;

        let _task = state.update(BoardMessage::SearchResults(
            vec![Ticket {
                id: "T-fresh".into(),
                title: "Fresh result".into(),
                description: String::new(),
                phase: TicketPhase::Backlog,
                assigned_to: None,
                workspace_name: "test_ws".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                updated_at: "2026-01-01T00:00:00Z".into(),
                comments: Vec::new(),
                prerequisites: Vec::new(),
                supersedes: None,
                superseded_by: None,
                commit_hash: None,
                lines_added: None,
                lines_removed: None,
                reporter: "test".into(),
                is_archived: false,
                pipeline_reservation: false,
                priority: 0,
                reviewed_head: None,
                reviewed_tree: None,
            }],
            42,
        ));

        assert_eq!(state.search_results.len(), 1);
        assert_eq!(state.search_results[0].id, "T-fresh");
    }

    #[test]
    fn test_search_cleared_resets_query_and_results() {
        let mut state = make_board_state();
        state.search_query = "something".into();
        state.search_results = vec![Ticket {
            id: "T-1".into(),
            title: "Result".into(),
            description: String::new(),
            phase: TicketPhase::Backlog,
            assigned_to: None,
            workspace_name: "test_ws".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            comments: Vec::new(),
            prerequisites: Vec::new(),
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            pipeline_reservation: false,
            priority: 0,
            reviewed_head: None,
            reviewed_tree: None,
        }];
        state.search_generation = 7;

        let _task = state.update(BoardMessage::SearchCleared);

        assert!(state.search_query.is_empty());
        assert!(state.search_results.is_empty());
        assert_eq!(state.search_generation, 8);
    }
}
