//! Board dashboard page — ticket management.

use std::collections::HashSet;
use std::path::Path;

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
use super::widgets::{badge_pill, diff_stats_row, role_badge, selectable_text};

/// Phases where an agent is actively running on the ticket. Cancelling a
/// ticket in one of these phases aborts an in-flight agent run — the
/// technical-failure auto-pause trigger (see `agent_in_flight`). The phase
/// alone is a coarse proxy: `assigned_to` must also be set, so queued
/// tickets in an agent phase with no running agent are not treated as
/// in-flight cancels.
const AGENT_RUNNING_PHASES: &[TicketPhase] = &[
    TicketPhase::Analysis,
    TicketPhase::InDevelopment,
    TicketPhase::InDiagnostics,
    TicketPhase::InReview,
    TicketPhase::InQa,
    TicketPhase::InSanitation,
];

/// Whether a cancel of this ticket aborts an in-flight agent run — an
/// agent-running phase with an assigned agent. Shared by the
/// `PerformAction` workspace-pause gate and the `RequestCancel`
/// confirmation-eligibility check so the two sites can't drift (a drift
/// would desync the pause behavior and make the modal's consequence
/// text untrue).
fn agent_in_flight(ticket: &Ticket) -> bool {
    AGENT_RUNNING_PHASES.contains(&ticket.phase) && ticket.assigned_to.is_some()
}

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
    /// Periodic re-fetch of the open ticket detail (see
    /// [`BoardState::refresh_selected_ticket`]). Carries the comment
    /// generation captured at dispatch so stale callbacks (modal closed
    /// or ticket switched while in flight) are dropped. Unlike
    /// `TicketDetails`, this is non-destructive: an in-progress comment
    /// draft survives.
    TicketDetailsRefreshed(u64, Box<Ticket>),
    /// Periodic re-fetch of the open ticket detail failed. Carries the
    /// generation for the same stale-callback detection as
    /// `TicketDetailsRefreshed`. Kept separate from `DetailError` so a
    /// stale refresh error cannot clear `selected_loading` mid-open.
    TicketDetailsRefreshError(u64, String),
    PerformAction(String, String), // ticket_id, new_phase
    ActionResult(Result<(), String>),

    /// User clicked the cancel action on a ticket. Checks the freshest
    /// cached ticket state synchronously ([`BoardState::ticket_in_flight`]):
    /// an in-flight agent opens the mid-pipeline confirmation modal
    /// ([`BoardMessage::ConfirmCancel`]); otherwise the cancel executes
    /// directly (non-agent phases stay single-click). `PerformAction`
    /// re-checks at execution, so the real pause decision is always
    /// authoritative.
    RequestCancel(String),
    /// The user confirmed the mid-pipeline cancel modal — execute the cancel.
    ConfirmCancel(String),
    /// The user dismissed the mid-pipeline cancel modal (backdrop or button).
    /// The ticket is left untouched.
    DismissCancel,

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
    /// Ticket awaiting mid-pipeline cancel confirmation. `Some(ticket_id)`
    /// while the cancel-confirmation modal is shown; the cancel only runs
    /// on [`BoardMessage::ConfirmCancel`].
    pending_cancel: Option<String>,
    /// Tracks which comment indices are expanded (for diagnostics collapse).
    expanded_comments: HashSet<usize>,
    /// Stores the last detail-load error message for display in the modal.
    detail_error: Option<String>,
    /// True while a periodic ticket-detail refresh is in flight. Guards
    /// against overlapping refreshes on slow reads (an older snapshot
    /// landing after a newer one); cleared on every refresh completion.
    detail_refresh_in_flight: bool,

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
            pending_cancel: None,
            expanded_comments: HashSet::new(),
            detail_error: None,
            detail_refresh_in_flight: false,
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

    /// Dismiss an open mid-pipeline cancel confirmation without touching
    /// the ticket. Shared by `Escape` and the modal's own dismiss actions
    /// (`DismissCancel`), so both paths stay in sync.
    fn dismiss_pending_cancel(&mut self) {
        self.pending_cancel = None;
    }

    /// Whether `ticket_id` currently has an in-flight agent, using the
    /// freshest synchronous state available: the open detail modal's
    /// `selected_ticket` (a fresh DB fetch, refreshed every second even
    /// during search), then the search results (the exact cards the user
    /// sees while a search is active), then the board list (refreshed every
    /// second; paused during search). Decides between the confirmation
    /// modal and a single-click cancel in `RequestCancel`. `PerformAction`
    /// re-fetches and re-checks `agent_in_flight` at execution, so the real
    /// pause decision is always authoritative regardless of staleness here.
    fn ticket_in_flight(&self, ticket_id: &str) -> bool {
        if let Some(ref ticket) = self.selected_ticket {
            if ticket.id == ticket_id {
                return agent_in_flight(ticket);
            }
        }
        if let Some(ticket) = self.search_results.iter().find(|t| t.id == ticket_id) {
            return agent_in_flight(ticket);
        }
        self.tickets
            .iter()
            .any(|t| t.id == ticket_id && agent_in_flight(t))
    }

    /// Reset all modal-related state fields (close detail modal).
    fn reset_modal(&mut self) {
        // Dismiss any open cancel confirmation too. Unreachable today
        // (Escape's first branch and the confirm backdrop guard it), but
        // the reset contract should hold for any future caller.
        self.dismiss_pending_cancel();
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

    /// Re-fetch the currently open ticket detail (periodic refresh).
    ///
    /// Fired alongside the ticket-list refresh so an open ticket window
    /// always shows the latest phase/comments without close/reopen.
    /// No-ops while the initial modal load is in flight, while a comment
    /// send is in progress (the DB snapshot would predate the optimistic
    /// comment push — `CommentSent` refetches on completion), or while a
    /// previous periodic refresh is still in flight (so an older snapshot
    /// cannot land after a newer one on slow reads).
    ///
    /// Carries the comment generation captured at dispatch; the
    /// [`BoardMessage::TicketDetailsRefreshed`] handler drops stale
    /// callbacks (modal closed or ticket switched while in flight).
    pub fn refresh_selected_ticket(&mut self) -> Task<BoardMessage> {
        if self.selected_loading || self.sending_comment || self.detail_refresh_in_flight {
            return Task::none();
        }
        let Some(ticket) = &self.selected_ticket else {
            return Task::none();
        };
        self.detail_refresh_in_flight = true;
        let id = ticket.id.clone();
        let generation = self.comment_generation;
        Task::perform(
            async move {
                let board = crate::board::store();
                board.get_ticket(&id).await.map_err(|e| e.to_string())
            },
            move |res| match res {
                Ok(Some(ticket)) => {
                    BoardMessage::TicketDetailsRefreshed(generation, Box::new(ticket))
                }
                Ok(None) => {
                    BoardMessage::TicketDetailsRefreshError(generation, "Ticket not found".into())
                }
                Err(e) => BoardMessage::TicketDetailsRefreshError(generation, e),
            },
        )
    }

    /// Apply fresh ticket detail to the modal display state.
    ///
    /// Clears the stale error, rebuilds the description/comment markdown
    /// caches, replaces the selected ticket, and reconciles the
    /// commit-stats display with the incoming ticket:
    ///
    /// - a commit hash that appeared or changed triggers a fresh stats
    ///   fetch (gated on the new hash being `Some`, so a hash can never
    ///   leave the loading state stuck),
    /// - an unchanged hash keeps the already-loaded stats (the periodic
    ///   refresh runs every second — unconditional fetches would hammer
    ///   the git CLI and flash "Loading commit stats" each tick),
    /// - no hash clears stale stats and any stuck loading flag.
    ///
    /// Returns the task to run (stats fetch or none). Callers must have
    /// verified the incoming ticket belongs to the open modal (id +
    /// generation guards). Non-destructive by contract: the comment
    /// input, undo stack, and generation counter are managed by the
    /// caller.
    fn apply_ticket_display(&mut self, ticket: Ticket) -> Task<BoardMessage> {
        self.detail_error = None;
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
        let hash_changed = self
            .selected_ticket
            .as_ref()
            .and_then(|t| t.commit_hash.as_ref())
            != ticket.commit_hash.as_ref();
        let ticket_id = ticket.id.clone();
        let has_hash = ticket.commit_hash.is_some();
        self.selected_ticket = Some(ticket);

        if hash_changed && has_hash {
            self.commit_stats = None;
            self.commit_stats_loading = true;
            self.commit_stats_generation += 1;
            Task::done(BoardMessage::FetchCommitStats(ticket_id))
        } else if !has_hash {
            self.commit_stats = None;
            self.commit_stats_loading = false;
            Task::none()
        } else {
            // Unchanged hash — keep the loaded stats (if any).
            Task::none()
        }
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
                ("↩ Back to Backlog", TicketPhase::Backlog),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::Failed => vec![
                ("✅ Ready for Dev", TicketPhase::ReadyForDevelopment),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::Done | TicketPhase::Cancelled => {
                vec![]
            }
            _ => vec![("🛑 Cancel", TicketPhase::Cancelled)],
        }
    }

    /// Map an action label to its lucide icon (16px) and tooltip text.
    ///
    /// Keyed off the action label (not the target phase): "Redo Dev" and
    /// "Ready for Dev" both transition to ReadyForDevelopment but need
    /// different texts, so `Redo` must be matched before `Dev`. The single
    /// keyword chain (Cancel → Redo → QA → Pause → Dev → Backlog) drives
    /// both the icon and the tooltip, so the two can never drift apart.
    /// The trailing `circle_check`/"move phase" pair is a defensive
    /// catch-all for labels outside [`Self::available_actions`].
    fn action_icon_and_tooltip<'a>(
        label: &str,
    ) -> (
        iced::widget::Text<'a, iced::Theme, iced::Renderer>,
        &'static str,
    ) {
        if label.contains("Cancel") {
            (lucide::circle_x(), "cancel ticket")
        } else if label.contains("Redo") {
            (lucide::refresh_cw(), "redo dev")
        } else if label.contains("QA") {
            (lucide::shield_check(), "send to QA")
        } else if label.contains("Pause") {
            (lucide::pause(), "move to planning")
        } else if label.contains("Dev") {
            (lucide::play(), "ready for dev")
        } else if label.contains("Backlog") {
            (lucide::rotate_ccw(), "back to backlog")
        } else {
            (lucide::circle_check(), "move phase")
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
            let (icon, tooltip_text) = Self::action_icon_and_tooltip(label);
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
                tooltip(
                    button(icon.size(16).color(icon_color))
                        .style(style_fn)
                        .on_press_maybe(if is_disabled {
                            None
                        } else if is_cancel {
                            // Cancels go through the eligibility gate first
                            // (`RequestCancel`): mid-pipeline cancels (agent
                            // running) require explicit confirmation; other
                            // cancels execute directly, exactly as before.
                            Some(BoardMessage::RequestCancel(ticket_id.to_string()))
                        } else {
                            Some(BoardMessage::PerformAction(
                                ticket_id.to_string(),
                                phase.to_string(),
                            ))
                        }),
                    text(tooltip_text).size(11),
                    tooltip::Position::Top,
                )
                .style(theme::tooltip_style),
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

    #[expect(clippy::too_many_lines)]
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
                if self.pending_cancel.is_some() {
                    // Dismiss the cancel-confirmation modal first — the
                    // ticket is left untouched.
                    self.dismiss_pending_cancel();
                    Task::none()
                } else if self.comment_focused {
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
                let stats_task = self.apply_ticket_display(ticket);
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
                stats_task
            }
            BoardMessage::TicketDetailsRefreshed(generation, ticket) => {
                let ticket = *ticket;
                // The fetch completed — allow the next periodic refresh.
                self.detail_refresh_in_flight = false;
                // Stale callback: the modal closed or the user switched
                // tickets while this fetch was in flight.
                if generation != self.comment_generation {
                    return Task::none();
                }
                // A comment send is in flight: the DB snapshot predates
                // the optimistic comment push, so applying it would make
                // the new comment flicker out of the list. `CommentSent`
                // refetches on completion.
                if self.sending_comment || self.selected_ticket.is_none() {
                    return Task::none();
                }
                // Display state only — the comment input, undo stack, and
                // generation counter stay untouched so a draft survives.
                self.apply_ticket_display(ticket)
            }
            BoardMessage::TicketDetailsRefreshError(generation, e) => {
                self.detail_refresh_in_flight = false;
                // Stale callback: the modal context changed while the
                // refresh was in flight — do not touch the display.
                if generation != self.comment_generation {
                    return Task::none();
                }
                // Keep the last known good detail visible; the next
                // periodic refresh retries. Unlike `DetailError`, this
                // never clears `selected_loading`.
                self.detail_error = Some(e);
                Task::none()
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
                        let ticket = board
                            .get_ticket(&ticket_id)
                            .await
                            .map_err(|e| e.to_string())?
                            .ok_or_else(|| format!("Ticket {ticket_id} not found"))?;
                        let source = ticket.phase;
                        // Cancelling a ticket with an agent mid-flight aborts the
                        // run — pause the workspace so queued development tickets
                        // don't cascade. Supersede auto-cancels and Manager-tool
                        // cancels don't go through here; shutdown and
                        // already-paused are handled inside the helper.
                        if phase == TicketPhase::Cancelled && agent_in_flight(&ticket) {
                            let notice = crate::management::pause_workspace_on_failure(
                                &ticket,
                                "user cancelled the ticket while an agent was running",
                            )
                            .await;
                            if !notice.is_empty() {
                                let _ = board
                                    .add_comment(
                                        &ticket_id,
                                        crate::role::SYSTEM_ROLE,
                                        notice.trim(),
                                    )
                                    .await;
                            }
                        }
                        // Manual "Redo Dev" (Reviewed → ReadyForDevelopment)
                        // is a bounce-back into development that consumes the
                        // same breaker budget as pipeline bounce-backs. The
                        // transition is phase-guarded (Reviewed only): a guard
                        // miss means the ticket was already moved externally —
                        // an expected no-op (no error, no bogus hop).
                        let applied = if source == TicketPhase::Reviewed
                            && phase == TicketPhase::ReadyForDevelopment
                        {
                            board
                                .bounce_back_to_dev(&ticket_id)
                                .await
                                .map_err(|e| e.to_string())?
                        } else {
                            board
                                .transition_to(&ticket_id, None, phase, None)
                                .await
                                .map_err(|e| e.to_string())?;
                            true
                        };
                        // Skip the hop when the ticket already reached the
                        // requested phase between render and click — the
                        // transition was a no-op and a self-transition entry
                        // would be a bogus hop.
                        if source != phase && applied {
                            crate::ticket_buffer::push(
                                &ticket.workspace_name,
                                &ticket_id,
                                source,
                                phase,
                                crate::ticket_buffer::TransitionOrigin::User,
                            );
                        }
                        Ok(())
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
            BoardMessage::RequestCancel(ticket_id) => {
                // Synchronous eligibility check against the freshest cached
                // state (`ticket_in_flight`): an in-flight agent requires
                // explicit confirmation; otherwise the cancel stays
                // single-click, exactly as before. There is no async fetch
                // to race with, so no stale-callback machinery is needed —
                // `PerformAction` re-checks `agent_in_flight` at execution.
                if self.ticket_in_flight(&ticket_id) {
                    self.pending_cancel = Some(ticket_id);
                    Task::none()
                } else {
                    Task::done(BoardMessage::PerformAction(
                        ticket_id,
                        TicketPhase::Cancelled.to_string(),
                    ))
                }
            }
            BoardMessage::ConfirmCancel(ticket_id) => {
                self.pending_cancel = None;
                Task::done(BoardMessage::PerformAction(
                    ticket_id,
                    TicketPhase::Cancelled.to_string(),
                ))
            }
            BoardMessage::DismissCancel => {
                self.dismiss_pending_cancel();
                Task::none()
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
                            .search_by_fts(&query, 20, ws.as_deref())
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

    /// Partition tickets into the four board columns in display order:
    /// In Progress, Ready, Pending, Completed.
    pub(crate) fn board_sections(tickets: &[Ticket]) -> [Vec<&Ticket>; 4] {
        crate::board::BoardStore::board_sections(tickets)
    }

    /// Render a single ticket card: clickable title, ID, phase badge, and action icons.
    #[expect(clippy::too_many_lines)]
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
            let stats_button = tooltip(
                button(stats_parts)
                    .padding([2, 6])
                    .style(theme::button_text)
                    .on_press(BoardMessage::ViewCommitDiff {
                        commit_hash: hash.clone(),
                        workspace_name: ws_name.clone(),
                    }),
                text("ticket commit diff").size(11),
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style);
            badge_row = badge_row.push(stats_button);
        }

        badge_row = badge_row.push(Space::new().width(Length::Fill));
        badge_row = badge_row.push(icon_row);

        // Per-ticket archive button for done/cancelled tickets
        if matches!(ticket.phase, TicketPhase::Done | TicketPhase::Cancelled) && !ticket.is_archived
        {
            let archive_btn = tooltip(
                button(
                    lucide::archive::<iced::Theme, iced::Renderer>()
                        .size(16)
                        .color(theme::TEXT_MUTED),
                )
                .style(theme::button_text)
                .on_press_maybe(if is_action_disabled {
                    None
                } else {
                    Some(BoardMessage::ArchiveTicket(ticket.id.clone()))
                }),
                text("archive ticket").size(11),
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style);
            badge_row = badge_row.push(archive_btn);
        }

        let mut card_children: Vec<Element<'_, BoardMessage>> = vec![
            // Title + ID row: both clickable
            tooltip(
                button(
                    column![
                        text(&ticket.title).size(13).color(theme::TEXT_PRIMARY),
                        text(&ticket.id).size(10).color(theme::TEXT_SECONDARY),
                    ]
                    .spacing(2),
                )
                .padding(iced::Padding::new(8.0).bottom(2.0))
                .width(Length::Fill)
                .style(theme::button_text)
                .on_press(BoardMessage::OpenModal(ticket.id.clone())),
                text("open ticket details").size(11),
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style)
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

    /// Render the modal overlay for ticket detail.
    ///
    /// Always a two-layer `Stack`: the ticket detail modal (or a
    /// type-stable placeholder) at child 0 and the cancel-confirmation
    /// layer (or a type-stable placeholder) at child 1. Both slots keep a
    /// `Stack` shape whether empty or populated, so the outer widget's
    /// shape never changes when the confirmation is shown or dismissed —
    /// child 0's tag stays stable and the detail subtree's transient
    /// widget state (e.g. scroll position) survives the confirm cycle.
    #[must_use]
    pub fn render_modal_overlay(&self) -> Element<'_, BoardMessage> {
        let detail_layer = if self.is_modal_open() {
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

                widget_helpers::modal_backdrop(dialog, BoardMessage::CloseModal, 0.5)
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

                widget_helpers::modal_backdrop(dialog, BoardMessage::CloseModal, 0.5)
            } else {
                let detail = self.modal_detail();
                let dialog = container(detail)
                    .width(Length::Fixed(720.0))
                    .padding(24)
                    .style(theme::dialog_container_style);

                widget_helpers::modal_backdrop(dialog, BoardMessage::CloseModal, 0.5)
            }
        } else {
            // Keep Stack widget type stable to prevent MouseArea state
            // from becoming orphaned when the modal closes.
            iced::widget::stack([widget_helpers::empty_stack_placeholder()]).into()
        };

        let confirm_layer = if let Some(ref ticket_id) = self.pending_cancel {
            Self::cancel_confirm_dialog(ticket_id)
        } else {
            // Type-stable placeholder so child 1's tag — and therefore the
            // outer Stack's shape — does not change when the confirmation
            // is dismissed.
            iced::widget::stack([widget_helpers::empty_stack_placeholder()]).into()
        };

        iced::widget::stack([detail_layer, confirm_layer]).into()
    }

    /// Build the mid-pipeline cancel confirmation dialog for a ticket.
    ///
    /// Shown when the eligibility check (`RequestCancel` →
    /// [`BoardState::ticket_in_flight`]) reports an in-flight agent; the
    /// listed consequences are real as of that check. If the agent finishes
    /// before the user confirms, `PerformAction` re-checks `agent_in_flight`
    /// at execution, so the pause just doesn't apply — the cancel itself is
    /// always phase-CAS-guarded.
    fn cancel_confirm_dialog(ticket_id: &str) -> Element<'_, BoardMessage> {
        let dialog = container(
            column![
                text(format!("Cancel ticket {ticket_id}?"))
                    .size(16)
                    .color(theme::TEXT_PRIMARY)
                    .font(theme::FONT_BOLD),
                Space::new().height(12),
                text("An agent is currently running on this ticket. Confirming will:")
                    .size(13)
                    .color(theme::TEXT_SECONDARY),
                Space::new().height(8),
                text(
                    "• cancel the ticket immediately — it is archived \
                     automatically afterwards;\n\
                     • stop the running agent — the current tool may finish, \
                     but no further work happens;\n\
                     • leave uncommitted changes in the working tree — not \
                     committed, not reverted;\n\
                     • pause the workspace pipeline — new analysis and \
                     development claims are blocked until you manually \
                     resume it (review and QA of other tickets continue).",
                )
                .size(13)
                .color(theme::TEXT_SECONDARY),
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Keep ticket").size(13))
                        .style(theme::button_secondary)
                        .on_press(BoardMessage::DismissCancel),
                    Space::new().width(8),
                    button(text("Cancel ticket").size(13))
                        .style(theme::button_danger)
                        .on_press(BoardMessage::ConfirmCancel(ticket_id.to_string())),
                ]
                .align_y(Alignment::Center),
            ]
            .width(Length::Fill),
        )
        .width(Length::Fixed(480.0))
        .padding(24)
        .style(theme::dialog_container_style);

        widget_helpers::modal_backdrop(dialog, BoardMessage::DismissCancel, 0.5)
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

        let created = theme::format_timestamp(&ticket.created_at);
        let updated = theme::format_timestamp(&ticket.updated_at);

        let meta_els: Vec<Element<'_, BoardMessage>> = vec![
            text(format!("Created: {created}"))
                .size(12)
                .color(theme::TEXT_SECONDARY)
                .into(),
            text(" · ").size(12).color(theme::TEXT_SECONDARY).into(),
            text(format!("Updated: {updated}"))
                .size(12)
                .color(theme::TEXT_SECONDARY)
                .into(),
        ];

        let mut secondary: Vec<String> = Vec::new();
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
                .color(theme::TEXT_SECONDARY)
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
                tooltip(
                    button(
                        lucide::x::<iced::Theme, iced::Renderer>()
                            .size(16)
                            .color(theme::TEXT_SECONDARY),
                    )
                    .style(theme::button_text)
                    .on_press(BoardMessage::CloseModal),
                    text("close").size(11),
                    tooltip::Position::Top,
                )
                .style(theme::tooltip_style),
            ]
            .align_y(Alignment::Center),
            text(&ticket.id).size(12).color(theme::TEXT_SECONDARY),
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
        summary_parts
            .push(diff_stats_row::<BoardMessage>(total_additions, total_deletions, 11.0).into());
        let summary_header = container(row(summary_parts).spacing(4).align_y(Alignment::Center))
            .padding([4, 8])
            .width(Length::Fill);

        // File stat rows — hide zero-valued sides
        let mut file_col = Column::new().spacing(2);
        for f in &stats.files {
            file_col = file_col.push(
                row![
                    container(text(&f.path).size(11).font(theme::FONT_REGULAR))
                        .width(Length::Fixed(400.0))
                        .clip(true),
                    Space::new().width(Length::Fill),
                    diff_stats_row::<BoardMessage>(f.additions, f.deletions, 11.0),
                ]
                .align_y(Alignment::Center),
            );
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
                    iced_selection::markdown::view(items, theme::markdown_settings())
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
            super::widgets::ChatComposerOptions {
                sending: self.sending_comment,
                min_height: 44.0,
                max_height: 132.0,
                controls: Vec::new(),
                // Board comment input is out of ticket scope — keep its
                // legacy always-active send button look.
                grey_on_empty: false,
                send_tooltip: "send comment",
            },
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
            let role_colors = theme::role_badge_color(&comment.role);

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
                iced_selection::markdown::view(items, theme::markdown_settings())
                    .map(BoardMessage::LinkClicked)
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
                    role_badge(comment.role.clone(), role_colors, 20, [2, 12], false),
                    Space::new().width(8),
                    text(theme::format_timestamp(&comment.created_at))
                        .size(10)
                        .color(theme::TEXT_SECONDARY),
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
            done_at: None,
            bounce_count: 0,
        });
        state
    }

    // ── SendComment: accepted-send smoke ──────────────────────────

    #[test]
    fn test_comment_within_limit_clears_input() {
        let mut state = make_board_state();
        state.comment_input = text_editor::Content::with_text("hello world");
        let _task = state.update(BoardMessage::SendComment);
        // Editor must be cleared on accepted message.
        assert!(
            state.comment_input.text().is_empty(),
            "input should be cleared after accepting a within-limit comment"
        );
        // Optimistic push + sending_comment are synchronous — no tx dependency.
        assert!(
            state.sending_comment,
            "sending_comment should be set after accept"
        );
        assert_eq!(
            state
                .selected_ticket
                .as_ref()
                .map_or(0, |t| t.comments.len()),
            1,
            "comment should be optimistically pushed to the selected ticket"
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
        let _ = state.update(BoardMessage::CommentInputChanged(
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
        let _ = state.update(BoardMessage::CommentInputChanged(
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

        let _ = state.update(BoardMessage::CommentInputChanged(
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

        let _ = state.update(BoardMessage::CommentModifiersChanged(
            keyboard::Modifiers::SHIFT,
        ));
        assert_eq!(state.modifiers, keyboard::Modifiers::SHIFT);

        let _ = state.update(BoardMessage::CommentModifiersChanged(
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
            done_at: None,
            bounce_count: 0,
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

    // ── TicketDetailsRefreshed (periodic refresh of the open modal) ─

    fn make_ticket(id: &str, phase: TicketPhase) -> Ticket {
        Ticket {
            id: id.into(),
            title: "Test ticket".into(),
            description: String::new(),
            phase,
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
            done_at: None,
            bounce_count: 0,
        }
    }

    /// Same-ticket refresh carrying a new phase plus an engineer comment.
    fn refreshed_ticket(phase: TicketPhase) -> Ticket {
        let mut ticket = make_ticket("T-1", phase);
        ticket.comments = vec![crate::board::TicketComment {
            role: "engineer".into(),
            content: "fixed the flaky test".into(),
            created_at: "2026-01-01T00:00:05Z".into(),
        }];
        ticket
    }

    #[test]
    fn test_ticket_details_refreshed_updates_and_preserves_draft() {
        let mut state = make_board_state(); // selected_ticket: T-1, Backlog
        state.comment_input = text_editor::Content::with_text("draft comment");
        let gen_before = state.comment_generation;

        // The periodic refresh delivers the same ticket with a new phase
        // and a new engineer comment — the exact staleness the feature fixes.
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            gen_before,
            Box::new(refreshed_ticket(TicketPhase::InReview)),
        ));

        let ticket = state.selected_ticket.as_ref().expect("modal still open");
        assert_eq!(
            ticket.phase,
            TicketPhase::InReview,
            "phase should update to the latest state"
        );
        assert_eq!(
            ticket.comments.len(),
            1,
            "comments should include the engineer comment"
        );
        assert_eq!(
            state.comment_input.text(),
            "draft comment",
            "an in-progress comment draft must survive the periodic refresh"
        );
        assert_eq!(
            state.comment_generation, gen_before,
            "periodic refresh must not bump the generation (would invalidate in-flight callbacks)"
        );
        assert!(!state.sending_comment);
        assert!(!state.commit_stats_loading);
    }

    #[test]
    fn test_ticket_details_refreshed_stale_generation_dropped() {
        let mut state = make_board_state();
        state.comment_generation = 42;
        // Stale callback (generation 0 < 42): the modal closed or the user
        // switched tickets while the fetch was in flight — must be dropped.
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            0,
            Box::new(refreshed_ticket(TicketPhase::InReview)),
        ));

        let ticket = state.selected_ticket.as_ref().expect("modal still open");
        assert_eq!(
            ticket.phase,
            TicketPhase::Backlog,
            "stale data must not apply"
        );
        assert!(ticket.comments.is_empty(), "stale comments must not apply");
        assert!(
            !state.detail_refresh_in_flight,
            "the completed fetch must clear the in-flight flag even when stale"
        );
    }

    #[test]
    fn test_ticket_details_refreshed_skipped_while_sending_comment() {
        let mut state = make_board_state();
        state.sending_comment = true;
        let generation = state.comment_generation;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            generation,
            Box::new(refreshed_ticket(TicketPhase::InReview)),
        ));

        let ticket = state.selected_ticket.as_ref().expect("modal still open");
        assert_eq!(
            ticket.phase,
            TicketPhase::Backlog,
            "refresh must not clobber the optimistic comment send"
        );
    }

    #[test]
    fn test_ticket_details_refreshed_stats_reconciliation() {
        const HASH: &str = "abcdef0123456789abcdef0123456789abcdef01";

        // Hash appeared → fresh stats fetch dispatched.
        let mut state = make_board_state();
        let mut with_hash = make_ticket("T-1", TicketPhase::Done);
        with_hash.commit_hash = Some(HASH.into());
        let stats_gen_before = state.commit_stats_generation;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            state.comment_generation,
            Box::new(with_hash),
        ));
        assert!(state.commit_stats_loading, "new hash should fetch stats");
        assert!(
            state.commit_stats_generation > stats_gen_before,
            "stats generation should bump for the new fetch"
        );

        // Unchanged hash → keep the loaded stats, no refetch.
        let mut state = make_board_state();
        state.selected_ticket.as_mut().unwrap().commit_hash = Some(HASH.into());
        state.commit_stats = Some(CommitStats { files: Vec::new() });
        state.commit_stats_generation = 5;
        let mut same_hash = make_ticket("T-1", TicketPhase::Done);
        same_hash.commit_hash = Some(HASH.into());
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            state.comment_generation,
            Box::new(same_hash),
        ));
        assert_eq!(
            state.commit_stats_generation, 5,
            "unchanged hash must not re-run git numstat every tick"
        );
        assert!(
            state.commit_stats.is_some(),
            "already-loaded stats should be kept"
        );

        // Hash vanished (latent) → loading must not stay stuck.
        let mut state = make_board_state();
        state.selected_ticket.as_mut().unwrap().commit_hash = Some(HASH.into());
        state.commit_stats_loading = true;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            state.comment_generation,
            Box::new(make_ticket("T-1", TicketPhase::Done)),
        ));
        assert!(
            !state.commit_stats_loading,
            "a hash-less ticket must clear the stats loading state"
        );
    }

    #[test]
    fn test_ticket_details_refresh_error_is_non_destructive() {
        let mut state = make_board_state();
        let generation = state.comment_generation;

        // A refresh error keeps the last known good detail visible and
        // must not clear the modal loading state (that is `DetailError`'s
        // job, for the initial modal load).
        let _task = state.update(BoardMessage::TicketDetailsRefreshError(
            generation,
            "db hiccup".into(),
        ));
        assert_eq!(
            state.selected_ticket.as_ref().unwrap().phase,
            TicketPhase::Backlog,
            "the displayed ticket must survive a refresh error"
        );
        assert_eq!(state.detail_error.as_deref(), Some("db hiccup"));
        assert!(!state.detail_refresh_in_flight);

        // Stale refresh error (generation mismatch) is dropped entirely.
        state.comment_generation += 1; // simulate modal close/switch
        let _task = state.update(BoardMessage::TicketDetailsRefreshError(0, "late".into()));
        assert_eq!(
            state.detail_error.as_deref(),
            Some("db hiccup"),
            "stale error must not overwrite the current error"
        );
    }

    #[test]
    fn test_refresh_selected_ticket_dispatch_gating() {
        // No modal open → nothing dispatched.
        let mut state = BoardState::new();
        let _task = state.refresh_selected_ticket();
        assert!(!state.detail_refresh_in_flight);

        // Modal open → dispatched (in-flight flag set).
        let mut state = make_board_state();
        let _task = state.refresh_selected_ticket();
        assert!(state.detail_refresh_in_flight);

        // Another refresh while one is in flight → no-op.
        let _task = state.refresh_selected_ticket();
        assert!(state.detail_refresh_in_flight);

        // In-flight flag cleared on completion.
        let generation = state.comment_generation;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(
            generation,
            Box::new(make_ticket("T-1", TicketPhase::Backlog)),
        ));
        assert!(!state.detail_refresh_in_flight);

        // Initial modal load in flight → no periodic refresh dispatched.
        let mut state = make_board_state();
        state.selected_loading = true;
        let _task = state.refresh_selected_ticket();
        assert!(!state.detail_refresh_in_flight);

        // Comment send in flight → no periodic refresh dispatched.
        let mut state = make_board_state();
        state.sending_comment = true;
        let _task = state.refresh_selected_ticket();
        assert!(!state.detail_refresh_in_flight);
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
            done_at: None,
            bounce_count: 0,
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

    // ── Mid-pipeline cancel confirmation ─────────────────────────

    fn ticket_with(id: &str, phase: TicketPhase, assigned_to: Option<String>) -> Ticket {
        Ticket {
            assigned_to,
            ..make_ticket(id, phase)
        }
    }

    #[test]
    fn test_mid_pipeline_cancel_confirmation_cycle() {
        // A ticket with an in-flight agent (agent-running phase + assigned
        // agent) in the cached board list → the confirmation modal opens.
        // (No detail modal open, so the list is the eligibility source.)
        let mut state = BoardState::new();
        state.tickets = vec![ticket_with(
            "T-1",
            TicketPhase::InDevelopment,
            Some("engineer".into()),
        )];
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        assert_eq!(
            state.pending_cancel.as_deref(),
            Some("T-1"),
            "in-flight cancel must open the confirmation modal"
        );

        // Dismissal (backdrop / Keep ticket) leaves the ticket untouched.
        let _task = state.update(BoardMessage::DismissCancel);
        assert!(state.pending_cancel.is_none());

        // A plain cancel (no in-flight agent) never opens the modal.
        state.tickets = vec![ticket_with("T-1", TicketPhase::Backlog, None)];
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        assert!(
            state.pending_cancel.is_none(),
            "non-pipeline cancel must stay single-click"
        );

        // Confirming closes the modal (the PerformAction task carries the
        // actual transition).
        state.tickets = vec![ticket_with(
            "T-1",
            TicketPhase::InDevelopment,
            Some("engineer".into()),
        )];
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        let _task = state.update(BoardMessage::ConfirmCancel("T-1".into()));
        assert!(state.pending_cancel.is_none());

        // Escape dismisses the confirmation modal first, even when the
        // comment input is focused — with the detail modal open showing an
        // in-flight ticket.
        let mut state = make_board_state();
        state.selected_ticket = Some(ticket_with(
            "T-1",
            TicketPhase::InDevelopment,
            Some("engineer".into()),
        ));
        state.comment_focused = true;
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        let _task = state.update(BoardMessage::Escape);
        assert!(
            state.pending_cancel.is_none(),
            "Escape must close the cancel-confirmation modal first"
        );
        assert!(
            state.comment_focused,
            "Escape on the confirm modal must not blur the comment input"
        );
        assert!(
            state.selected_ticket.is_some(),
            "Escape on the confirm modal must not close the ticket detail modal"
        );
    }

    #[test]
    fn test_request_cancel_eligibility_source_precedence() {
        // The eligibility check uses the freshest synchronous state:
        // `selected_ticket` (a fresh fetch, per-second refresh even during
        // search), then search results (the exact cards the user sees while
        // a search is active), then the board list. A stale lower-priority
        // source must not decide the modal.
        let mut state = make_board_state(); // selected_ticket: T-1, Backlog, no assignee
        state.tickets = vec![ticket_with(
            "T-1",
            TicketPhase::InDevelopment,
            Some("engineer".into()),
        )];
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        assert!(
            state.pending_cancel.is_none(),
            "the fresher open detail (Backlog) must win over a stale list"
        );

        // Detail says in-flight while the list is stale → modal opens.
        let mut state = make_board_state();
        state.selected_ticket = Some(ticket_with(
            "T-1",
            TicketPhase::InDiagnostics,
            Some("diagnostics".into()),
        ));
        state.tickets = vec![ticket_with("T-1", TicketPhase::Backlog, None)];
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        assert_eq!(
            state.pending_cancel.as_deref(),
            Some("T-1"),
            "in-flight detail must open the confirmation modal"
        );

        // During search the visible cards come from search_results (the
        // list refresh is paused), so they are consulted before the list.
        let mut state = make_board_state();
        state.selected_ticket = None;
        state.search_results = vec![ticket_with("T-1", TicketPhase::InQa, Some("qa".into()))];
        state.tickets = vec![ticket_with("T-1", TicketPhase::Backlog, None)];
        let _task = state.update(BoardMessage::RequestCancel("T-1".into()));
        assert_eq!(
            state.pending_cancel.as_deref(),
            Some("T-1"),
            "search results must be consulted while a search is active"
        );

        // A ticket in none of the eligibility sources is not treated as
        // in-flight, so the cancel stays single-click.
        let mut state = make_board_state();
        let _task = state.update(BoardMessage::RequestCancel("T-unknown".into()));
        assert!(
            state.pending_cancel.is_none(),
            "unknown ticket must stay single-click"
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
            done_at: None,
            bounce_count: 0,
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
                done_at: None,
                bounce_count: 0,
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
                done_at: None,
                bounce_count: 0,
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
            done_at: None,
            bounce_count: 0,
        }];
        state.search_generation = 7;

        let _task = state.update(BoardMessage::SearchCleared);

        assert!(state.search_query.is_empty());
        assert!(state.search_results.is_empty());
        assert_eq!(state.search_generation, 8);
    }

    // ── Completed-column ordering ──────────────────────────────────

    /// Build a ticket for sort tests from the shared base literal.
    fn test_ticket(
        id: &str,
        phase: TicketPhase,
        created_at: &str,
        done_at: Option<&str>,
    ) -> Ticket {
        let base = make_board_state().selected_ticket.unwrap();
        Ticket {
            id: id.into(),
            phase,
            created_at: created_at.into(),
            done_at: done_at.map(str::to_string),
            ..base
        }
    }

    #[test]
    fn test_partition_completed_sorts_done_then_cancelled() {
        let tickets = vec![
            test_ticket("c3", TicketPhase::Cancelled, "2026-01-01T00:00:00Z", None),
            test_ticket(
                "d2",
                TicketPhase::Done,
                "2026-01-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
            test_ticket(
                "d1",
                TicketPhase::Done,
                "2026-01-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ),
            test_ticket("d3", TicketPhase::Done, "2026-01-02T00:00:00Z", None),
            test_ticket("c1", TicketPhase::Cancelled, "2026-01-03T00:00:00Z", None),
            test_ticket("c2", TicketPhase::Cancelled, "2026-01-02T00:00:00Z", None),
        ];
        let [_, _, _, completed] = BoardState::board_sections(&tickets);
        let ids: Vec<&str> = completed.iter().map(|t| t.id.as_str()).collect();
        // Done first (newest done_at, created_at fallback), then cancelled
        // (newest created_at). A cancelled ticket's stale done_at must not
        // promote it into the done block.
        assert_eq!(ids, ["d1", "d2", "d3", "c1", "c2", "c3"]);
    }
}
