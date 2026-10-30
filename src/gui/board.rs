//! Board dashboard page — ticket management.

use std::collections::HashSet;
use std::path::Path;

use crate::git::commands::{parse_numstat_lines, run_git_output};
use crate::pipeline::board::{Ticket, TicketComment, TicketPhase};

use iced::widget::{Column, Row, Space, button, column, container, markdown, row, text, tooltip};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use super::common::MAX_INPUT_CHARS;
use super::dialog;
use super::theme;
use super::widgets::{
    self, PILL_COMPACT, PILL_MODAL, PillMetrics, badge_pill, diff_stats_row, selectable_text,
    tooltip_hint,
};

/// Per-file stat from `git show --numstat`.
#[derive(Debug, Clone)]
struct FileStat {
    path: String,
    additions: i64,
    deletions: i64,
}

/// Parsed commit stats for a ticket's associated commit.
#[derive(Debug, Clone)]
pub(crate) struct CommitStats {
    files: Vec<FileStat>,
}

// The largest variants carry `Vec<Ticket>`/boxed payloads, so the enum contains
// both tiny unit variants and sizable ones; that is inherent to a page message.
#[derive(Debug, Clone)]
#[expect(clippy::large_enum_variant)]
pub(crate) enum BoardMessage {
    Refreshed {
        tickets: Vec<Ticket>,
        generation: u64,
    },
    /// A failed board read. Carries the board generation captured at dispatch,
    /// mirroring [`BoardMessage::Refreshed`], so a stale failure is dropped.
    RefreshError {
        error: String,
        generation: u64,
    },
    /// A ticket change delivered by the CDC change stream (see
    /// [`crate::db::cdc`]). The board applies the delta idempotently by ticket id.
    TicketChanged(Box<crate::db::cdc::ChangeEvent>),
    /// The change stream lagged — force a full board refresh (the delta was
    /// lost, so the snapshot is the only safe way to recover).
    BoardRefreshNeeded,
    /// The re-fetched ticket for a [`TicketChanged`] delta, ready to upsert.
    TicketUpserted(Ticket),
    TicketDetails(Box<Ticket>),
    DetailError(String),
    /// CDC-driven re-fetch of the open ticket detail (see
    /// [`BoardState::refresh_selected_ticket`]). Carries the comment
    /// generation captured at dispatch so stale callbacks (modal closed
    /// or ticket switched while in flight) are dropped. Unlike
    /// `TicketDetails`, this is non-destructive: an in-progress comment
    /// draft survives.
    TicketDetailsRefreshed(u64, Box<Ticket>),
    /// CDC-driven re-fetch of the open ticket detail failed. Carries the
    /// generation for the same stale-callback detection as
    /// `TicketDetailsRefreshed`. Kept separate from `DetailError` so a
    /// stale refresh error cannot clear `selected_loading` mid-open.
    TicketDetailsRefreshError(u64, String),
    PerformAction(String, String), // ticket_id, new_phase
    ActionResult(Result<(), String>),

    /// User clicked the cancel action on a ticket. Checks the freshest
    /// cached ticket state synchronously
    /// ([`BoardState::cancel_confirmation_needed`]): a mid-pipeline
    /// (implementation-phase) ticket opens the confirmation modal
    /// ([`BoardMessage::ConfirmCancel`]); otherwise the cancel executes
    /// directly (non-pipeline phases stay single-click). `PerformAction`
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

    /// Batch-pause all queued tickets (back to planning).
    PauseAllQueued,

    /// Result of the batch pause operation.
    PauseAllQueuedResult(Result<u64, String>),

    /// Archive a single ticket (sets is_archived = 1).
    ArchiveTicket(String),

    /// Trigger async load of commit stats for a ticket.
    FetchCommitStats(String),
    /// Commit stats loaded (or error) — carries generation for stale-callback guard.
    CommitStatsLoaded(String, u64, Result<CommitStats, String>),

    /// Navigate to the commit diff view for this ticket.
    ViewCommitDiff {
        commit_hash: String,
        /// The ticket's own workspace — the project the commit belongs to,
        /// which may not be the picked one.
        workspace_name: String,
    },

    /// Toggle expansion of a diagnostics comment.
    ToggleCommentExpand(usize),

    /// Comment text input changed in the ticket detail modal.
    CommentInputChanged(super::editor_widget::EditorAction),
    /// Send a comment on the current ticket.
    SendComment,
    /// Result of adding a comment via the backend. Carries the generation
    /// counter captured at send-time for stale-callback detection.
    CommentSent(u64, Result<(), String>),

    // ── Ticket search (sidebar) ──────────────────────────────────
    /// Search query text changed (user typing in the sidebar search input).
    SearchInputChanged(super::editor_widget::EditorAction),
    /// Search results returned from async FTS query.
    /// Carries a generation counter for stale-callback detection.
    SearchResults(Vec<Ticket>, u64),
    /// The async FTS query failed. Carries the generation counter so a stale
    /// failure cannot replace the results of a newer search.
    SearchError(String, u64),
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
    /// Bumped on every refresh request by [`Self::refresh`]. A read outcome is
    /// applied only while its generation still matches, so an older in-flight
    /// read cannot overwrite the newest one (see [`Self::applied_read`]).
    board_generation: u64,
    /// Ticket ids a CDC delta archived/removed since the last baseline. A
    /// baseline snapshot read before such a delta must not re-add the ticket,
    /// so `apply_refreshed` skips these ids from the snapshot.
    pub(crate) delta_removed_ids: HashSet<String>,
    /// True on a lag recovery ([`BoardMessage::BoardRefreshNeeded`]) so the
    /// recovery snapshot is authoritative and replaces the board, dropping
    /// tickets a lost removal delta would have removed. Cleared on the first
    /// snapshot; subsequent refreshes merge.
    pub(crate) replace_on_refresh: bool,
    /// Loaded commit stats for the open ticket.
    commit_stats: Option<CommitStats>,
    /// Failure from the last commit-stats fetch for the open ticket, rendered
    /// in the modal so a fetch failure is not silently swallowed.
    commit_stats_error: Option<String>,
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
    comment_input: super::editor_widget::EditorBuffer,
    /// Whether a comment is currently being sent (prevents double-send).
    sending_comment: bool,
    /// Whether the comment input is focused (for Escape-key blur behavior).
    ///
    /// Note: tracking is imperfect — clicking elsewhere (natural focus loss)
    /// won't clear this flag, so the first Escape after a natural blur still
    /// acts as "blur" instead of "close". A second Escape always closes the
    /// modal, which is acceptable UX for Iced 0.14's widget focus API limits.
    comment_focused: bool,
    /// Monotonic counter incremented when the modal context changes
    /// (modal close via `reset_modal()`, or ticket switch via
    /// `TicketDetails`). Captured in `SendComment` and verified in
    /// `CommentSent` to detect stale async callbacks.
    comment_generation: u64,
    /// Undo/redo stack for the comment input text editor.
    undo_stack: super::common::UndoStack,

    // ── Ticket search (sidebar) ──────────────────────────────────
    /// Current search query (empty = no search active).
    pub(crate) search_query: super::common::SingleLineEditorState,
    /// Search results, populated when `search_query` is non-empty.
    pub(crate) search_results: Vec<Ticket>,
    /// Last FTS search failure for the active query. Rendered in the sidebar
    /// instead of "No matching tickets" until a successful search or a cleared
    /// search replaces it.
    pub(crate) search_error: Option<String>,
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
            board_generation: 0,
            delta_removed_ids: HashSet::new(),
            replace_on_refresh: false,
            commit_stats: None,
            commit_stats_error: None,
            commit_stats_loading: false,
            commit_stats_generation: 0,
            pending_cancel: None,
            expanded_comments: HashSet::new(),
            detail_error: None,
            detail_refresh_in_flight: false,
            comment_input: super::editor_widget::EditorBuffer::with_text(
                "",
                Some(super::highlight::HighlightLanguage::Markdown),
            ),
            sending_comment: false,
            comment_focused: false,
            comment_generation: 0,
            undo_stack: super::common::UndoStack::new(),
            search_query: super::common::SingleLineEditorState::new(""),
            search_results: Vec::new(),
            search_error: None,
            search_generation: 0,
        }
    }

    /// Dismiss an open mid-pipeline cancel confirmation without touching
    /// the ticket. Shared by `Escape` and the modal's own dismiss actions
    /// (`DismissCancel`), so both paths stay in sync.
    fn dismiss_pending_cancel(&mut self) {
        self.pending_cancel = None;
    }

    /// Whether `ticket_id` is in a mid-pipeline (implementation) phase, using
    /// the freshest synchronous state available: the open detail modal's
    /// `selected_ticket` (a fresh DB fetch, refreshed on change/periodic
    /// detail refresh), then the search results (the exact cards the user
    /// sees while a search is active), then the board list (CDC-driven,
    /// paused during search). Decides between the confirmation
    /// modal and a single-click cancel in `RequestCancel`. `PerformAction`
    /// re-checks the phase at execution, so the real
    /// pause decision is always authoritative regardless of staleness here.
    fn cancel_confirmation_needed(&self, ticket_id: &str) -> bool {
        self.selected_ticket
            .as_ref()
            .filter(|t| t.id == ticket_id)
            .or_else(|| self.search_results.iter().find(|t| t.id == ticket_id))
            .or_else(|| self.tickets.iter().find(|t| t.id == ticket_id))
            .is_some_and(|t| t.phase.is_pipeline_occupied())
    }

    /// Reset all modal-related state fields (close detail modal).
    fn reset_modal(&mut self) {
        // The cancel confirmation is modal state, so resetting the modal clears it.
        self.dismiss_pending_cancel();
        self.selected_ticket = None;
        self.selected_loading = false;
        self.detail_error = None;
        self.description_md = None;
        self.comments_md.clear();
        self.expanded_comments.clear();
        self.commit_stats = None;
        self.commit_stats_error = None;
        self.commit_stats_loading = false;
        self.commit_stats_generation += 1;
        self.reset_comment_state();
    }

    /// Clear the comment draft and related state, and bump
    /// `comment_generation` so in-flight comment callbacks for the
    /// previous ticket no longer match.
    fn reset_comment_state(&mut self) {
        self.comment_input.clear();
        self.sending_comment = false;
        self.comment_focused = false;
        self.comment_generation += 1;
        self.undo_stack.clear();
    }

    /// Whether a read outcome carrying `generation` is the board's current read
    /// — the shape both [`BoardMessage::Refreshed`] and
    /// [`BoardMessage::RefreshError`] apply through, so a caller can tell an
    /// applied read from one that lost its race.
    pub(crate) fn applied_read(&self, generation: u64) -> bool {
        generation == self.board_generation
    }

    /// Request a full board snapshot — every workspace's non-archived
    /// tickets. Bumps [`Self::board_generation`] so only the newest request's
    /// snapshot applies.
    pub fn refresh(&mut self) -> Task<BoardMessage> {
        self.board_generation += 1;
        let generation = self.board_generation;
        Task::perform(
            async move {
                crate::pipeline::board::store()
                    .list_all_tickets(None, None)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| match res {
                Ok(tickets) => BoardMessage::Refreshed {
                    tickets,
                    generation,
                },
                Err(error) => BoardMessage::RefreshError { error, generation },
            },
        )
    }

    /// Non-archived queued tickets in the listed board — the bulk pause's
    /// scope. The search never narrows it: the bulk actions cover the list,
    /// not what a query matched.
    fn queued_tickets(&self) -> impl Iterator<Item = &Ticket> {
        self.tickets
            .iter()
            .filter(|t| !t.is_archived && t.phase == TicketPhase::Queued)
    }

    /// Whether the "Pause all queued tickets" context-menu item should be
    /// offered — at least one non-archived Queued ticket is listed.
    pub(crate) fn has_queued_tickets(&self) -> bool {
        self.queued_tickets().next().is_some()
    }

    /// Whether the board has been read successfully and holds no ticket to show
    /// — the one state in which the Home page omits its ticket column. A board
    /// that has not loaded yet and a failed read are not empty: each renders in
    /// the column instead of hiding it.
    pub(crate) fn has_nothing_to_show(&self) -> bool {
        self.load_state.has_loaded() && self.load_state.error().is_none() && self.tickets.is_empty()
    }

    /// Workspaces to drain for the bulk pause: the distinct workspaces of the
    /// listed non-archived Queued tickets, deduplicated. Each is drained once.
    fn queued_workspaces(&self) -> Vec<String> {
        let mut workspaces: Vec<String> = Vec::new();
        for ticket in self.queued_tickets() {
            if !workspaces.contains(&ticket.workspace_name) {
                workspaces.push(ticket.workspace_name.clone());
            }
        }
        workspaces
    }

    /// Apply a CDC ticket change delta to the board, idempotently by ticket id.
    ///
    /// - DELETEs (or an archived row) remove the ticket from the list and are
    ///   recorded in `delta_removed_ids` so a stale baseline snapshot cannot
    ///   re-add them.
    /// - INSERT/UPDATE re-fetches the ticket by id and upserts it. Re-fetch uses
    ///   the board's own parsing so the delta always carries a consistent, current
    ///   snapshot; the render path re-sorts each frame, so no manual sort
    ///   maintenance is needed. There is deliberately no global `change_id`
    ///   watermark: `change_id` is a WAL `NewRowid` that restarts after
    ///   `turso_cdc` is fully pruned, so a watermark would silently drop deltas.
    /// - If the changed ticket is the open detail modal and its `updated_at`
    ///   moved, the modal detail is re-refreshed. Comment inserts bump
    ///   `tickets.updated_at` (see `add_comment_tx`), so a tickets delta fires
    ///   for comment activity and this re-refresh covers it; the board LIST
    ///   does not load comment content (`LoadComments::No`).
    fn apply_ticket_change(&mut self, event: &crate::db::cdc::ChangeEvent) -> Task<BoardMessage> {
        use crate::db::cdc::ChangeType;
        let Some(id) = event.ticket_id().map(str::to_string) else {
            return Task::none();
        };
        let detail_refresh = if self.selected_ticket.as_ref().is_some_and(|t| t.id == id) {
            let changed = match &event.after {
                Some(after) => {
                    let new_at = after
                        .get("updated_at")
                        .and_then(crate::db::cdc::CdcValue::as_text);
                    new_at != self.selected_ticket.as_ref().map(|t| t.updated_at.as_str())
                }
                None => true,
            };
            if changed {
                self.refresh_selected_ticket()
            } else {
                Task::none()
            }
        } else {
            Task::none()
        };
        let is_removed = match event.change_type {
            ChangeType::Delete => true,
            _ => {
                event
                    .after
                    .as_ref()
                    .and_then(|r| r.get("is_archived"))
                    .and_then(crate::db::cdc::CdcValue::as_integer)
                    == Some(1)
            }
        };
        let upsert = if is_removed {
            self.tickets.retain(|t| t.id != id);
            self.delta_removed_ids.insert(id.clone());
            Task::none()
        } else {
            self.delta_removed_ids.remove(&id);
            // Re-fetch current state: the render path re-sorts each frame, so no
            // manual sort maintenance is needed, and a re-fetch always returns the
            // freshest snapshot (so an out-of-order delta cannot clobber the board).
            Task::perform(
                async move {
                    let board = crate::pipeline::board::store();
                    match board
                        .get_tickets_by_ids(&[id], crate::pipeline::board::LoadComments::No)
                        .await
                        .map_err(|e| e.to_string())
                    {
                        Ok(mut v) => match v.pop() {
                            Some(ticket) => BoardMessage::TicketUpserted(ticket),
                            None => BoardMessage::BoardRefreshNeeded,
                        },
                        Err(_) => BoardMessage::BoardRefreshNeeded,
                    }
                },
                |msg| msg,
            )
        };
        Task::batch([upsert, detail_refresh])
    }

    /// Apply a baseline snapshot.
    ///
    /// - On a lag recovery (`replace_on_refresh`) the snapshot is authoritative
    ///   and replaces the board, so a ticket a lost removal delta would have
    ///   removed cannot linger. This also clears `delta_removed_ids` — the
    ///   recovery snapshot carries its own removal history.
    /// - Otherwise it merges by per-ticket `updated_at` freshness: a snapshot
    ///   ticket older than the cached one is left alone (so a newer delta
    ///   already applied is never clobbered), but a newer snapshot ticket
    ///   updates the board (recovering a lost delta). This is correct even if
    ///   two snapshots arrive out of order, and — unlike a global `change_id`
    ///   watermark — is immune to `change_id` restarting after `turso_cdc` is
    ///   fully pruned.
    /// - Tickets a delta archived/removed (`delta_removed_ids`) are never
    ///   re-added by a stale snapshot. A board ticket **absent** from the
    ///   snapshot is kept: with no guaranteed snapshot/delta ordering, an
    ///   omission is ambiguous (the snapshot may predate the ticket's
    ///   creation/delta). Dropping it risks a permanent loss (a live ticket that
    ///   never changes again), while keeping it is self-healing — a removal
    ///   delta arrives and removes it, or a lag recovery snapshot replaces the
    ///   board.
    fn apply_refreshed(&mut self, tickets: Vec<Ticket>) {
        if self.replace_on_refresh {
            self.tickets = tickets;
            self.delta_removed_ids.clear();
            self.replace_on_refresh = false;
        } else {
            let mut merged: Vec<Ticket> = Vec::with_capacity(tickets.len());
            for snap in tickets {
                if self.delta_removed_ids.contains(&snap.id) {
                    continue;
                }
                match self.tickets.iter().find(|t| t.id == snap.id) {
                    Some(cur) => merged.push(if Self::is_at_least_as_fresh(cur, &snap) {
                        cur.clone()
                    } else {
                        snap
                    }),
                    None => merged.push(snap),
                }
            }
            // Keep a board ticket absent from the snapshot (see the doc above).
            for cur in &self.tickets {
                if !merged.iter().any(|t| t.id == cur.id) {
                    merged.push(cur.clone());
                }
            }
            self.tickets = merged;
        }
        // A successful snapshot replaces any previous load failure.
        self.load_state.clear_error();
        self.load_state.finish_loading();
    }

    /// Whether `cached` is at least as fresh as `snap` by parsed RFC 3339
    /// `updated_at` (each tickets write bumps it, so it is a sound per-ticket
    /// freshness signal). Falls back to lexicographic comparison so a malformed
    /// timestamp cannot panic.
    fn is_at_least_as_fresh(cached: &Ticket, snap: &Ticket) -> bool {
        let a = crate::db::parse_utc_timestamp(&cached.updated_at);
        let b = crate::db::parse_utc_timestamp(&snap.updated_at);
        match (a, b) {
            (Ok(a), Ok(b)) => a >= b,
            _ => cached.updated_at >= snap.updated_at,
        }
    }

    /// Upsert a re-fetched ticket into the board list, honoring the archive
    /// state.
    fn apply_ticket_upsert(&mut self, ticket: Ticket) {
        if ticket.is_archived {
            self.tickets.retain(|t| t.id != ticket.id);
            // Track the removal so a stale snapshot read before the archive
            // cannot re-add it.
            self.delta_removed_ids.insert(ticket.id.clone());
            return;
        }
        if let Some(pos) = self.tickets.iter().position(|t| t.id == ticket.id) {
            self.tickets[pos] = ticket;
        } else {
            self.tickets.push(ticket);
        }
    }

    /// Re-fetch the currently open ticket detail (CDC-driven refresh).
    ///
    /// Fired alongside the ticket-list refresh so an open ticket window
    /// always shows the latest phase/comments without close/reopen.
    /// No-ops while the initial modal load is in flight, while a comment
    /// send is in progress (the DB snapshot would predate the optimistic
    /// comment push — `CommentSent` refetches on completion), or while a
    /// previous refresh is still in flight (so an older snapshot cannot
    /// land after a newer one on slow reads).
    ///
    /// Carries the comment generation captured at dispatch; the
    /// [`BoardMessage::TicketDetailsRefreshed`] handler drops stale
    /// callbacks (modal closed or ticket switched while in flight).
    fn refresh_selected_ticket(&mut self) -> Task<BoardMessage> {
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
                let board = crate::pipeline::board::store();
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

    /// Parse all comments into the cached-markdown form used by
    /// `comments_md` (`(index, items)` pairs).
    fn parse_comments_md(comments: &[TicketComment]) -> Vec<(usize, Vec<markdown::Item>)> {
        comments
            .iter()
            .enumerate()
            .map(|(i, c)| (i, markdown::parse(&c.content).collect()))
            .collect()
    }

    /// Apply fresh ticket detail to the modal display state.
    ///
    /// Clears the stale error, rebuilds the description/comment markdown
    /// caches, replaces the selected ticket, and reconciles the
    /// commit-stats display with the incoming ticket:
    ///
    /// - a (project, commit) pair that appeared or changed triggers a fresh
    ///   stats fetch (gated on the hash being `Some`, so a hash can never
    ///   leave the loading state stuck),
    /// - an unchanged pair keeps the already-loaded stats (the refresh is
    ///   event-driven — unconditional fetches would hammer the git CLI and
    ///   flash "Loading commit stats" each refresh),
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
        self.comments_md = Self::parse_comments_md(&ticket.comments);
        // Commit statistics belong to a (project, commit) pair: two projects can
        // share git history, so the hash alone cannot say whether what is
        // loaded still describes the ticket being shown.
        let stats_current = self.selected_ticket.as_ref().is_some_and(|t| {
            t.workspace_name == ticket.workspace_name && t.commit_hash == ticket.commit_hash
        });
        let ticket_id = ticket.id.clone();
        let has_hash = ticket.commit_hash.is_some();
        self.selected_ticket = Some(ticket);

        if !stats_current && has_hash {
            self.commit_stats = None;
            self.commit_stats_error = None;
            self.commit_stats_loading = true;
            self.commit_stats_generation += 1;
            Task::done(BoardMessage::FetchCommitStats(ticket_id))
        } else if !has_hash {
            self.commit_stats = None;
            self.commit_stats_error = None;
            self.commit_stats_loading = false;
            Task::none()
        } else {
            // Same project and commit — keep the loaded stats (if any).
            Task::none()
        }
    }

    fn available_actions(phase: TicketPhase) -> Vec<(&'static str, TicketPhase)> {
        match phase {
            TicketPhase::Queued => vec![
                ("⏸ Pause", TicketPhase::Planning),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::Planning => vec![
                ("✅ Queued", TicketPhase::Queued),
                ("↩ Back to Backlog", TicketPhase::Backlog),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            // Manual override for Backlog/Analysis: skip the planning gate
            // and send to engineer (cancels in-flight agents via
            // `transition_to`). Failed tickets re-queue the same way.
            TicketPhase::Failed | TicketPhase::Backlog | TicketPhase::Analysis => vec![
                ("✅ Queued", TicketPhase::Queued),
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
    /// Keyed off the action label. The keyword chain
    /// (Cancel → Pause → Queued → Backlog) drives both the icon and the
    /// tooltip, so the two can never drift apart. Rework is automatic via a
    /// bounce — there is no manual rework action anymore.
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
        } else if label.contains("Pause") {
            (lucide::pause(), "move to planning")
        } else if label.contains("Queued") {
            (lucide::play(), "queued")
        } else if label.contains("Backlog") {
            (lucide::rotate_ccw(), "back to backlog")
        } else {
            (lucide::circle_check(), "move phase")
        }
    }

    /// Build a row of icon-only action buttons for the given ticket and actions.
    /// Icons are 16px with 4px spacing. Cancel buttons render by phase: while the
    /// ticket is not pipeline-occupied ([`TicketPhase::is_pipeline_occupied`] —
    /// backlog/analysis/planning/queued/failed) the cancel is the shared compact
    /// gray [`widgets::compact_x_button`]; once the ticket occupies the pipeline
    /// it keeps the red 16px `circle_x` with [`theme::button_text_danger`].
    /// All others use [`theme::button_text`]. When `is_disabled` is true all
    /// buttons dim to [`theme::TEXT_MUTED`] and become non-interactive.
    fn action_icon_row<'a>(
        ticket_id: &str,
        current_phase: TicketPhase,
        actions: &[(&'static str, TicketPhase)],
        is_disabled: bool,
    ) -> Row<'a, BoardMessage> {
        let mut icon_row = Row::new()
            .spacing(theme::SPACE_4)
            .align_y(Alignment::Center);
        for (label, phase) in actions {
            let is_cancel = label.contains("Cancel");
            if is_cancel && !current_phase.is_pipeline_occupied() {
                // Pre-pipeline cancel: compact gray X. Padding compensates the
                // smaller 12px glyph so the hit zone stays the same 36x26 box the
                // red 16px-glyph cancel renders (12 + 2*12 = 36 wide, ~12 + 2*7 = 26 tall).
                icon_row = icon_row.push(widgets::compact_x_button(
                    theme::TEXT_SECONDARY,
                    iced::Padding {
                        top: 7.0,
                        bottom: 7.0,
                        left: 12.0,
                        right: 12.0,
                    },
                    if is_disabled {
                        None
                    } else {
                        Some(BoardMessage::RequestCancel(ticket_id.to_string()))
                    },
                    Some("cancel ticket"),
                ));
                continue;
            }
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
            icon_row = icon_row.push(widgets::icon_tooltip_button(
                icon.size(widgets::ACTION_ICON_SIZE).color(icon_color),
                tooltip_text,
                if is_disabled {
                    None
                } else if is_cancel {
                    // Cancels go through the eligibility gate first
                    // (`RequestCancel`): mid-pipeline (implementation-phase)
                    // cancels require explicit confirmation; other cancels
                    // execute directly, exactly as before.
                    Some(BoardMessage::RequestCancel(ticket_id.to_string()))
                } else {
                    Some(BoardMessage::PerformAction(
                        ticket_id.to_string(),
                        phase.to_string(),
                    ))
                },
                button::DEFAULT_PADDING,
                style_fn,
                tooltip::Position::Top,
            ));
        }
        icon_row
    }

    /// Phase badge pill (ticket cards, modal header); colors from [`theme::ticket_phase_color`].
    fn phase_badge<'a>(phase: TicketPhase, metrics: PillMetrics) -> Element<'a, BoardMessage> {
        let (fg, bg) = theme::ticket_phase_color(phase);
        let pill = badge_pill(phase.display_name(), (fg, bg), metrics);
        tooltip_hint(pill, "Current phase")
    }

    /// Priority chip pill; colors from [`theme::ticket_priority_color`].
    fn priority_badge<'a>(priority: i64, metrics: PillMetrics) -> Element<'a, BoardMessage> {
        let (fg, bg) = theme::ticket_priority_color(priority);
        let pill = badge_pill(format!("P{priority}"), (fg, bg), metrics);
        tooltip_hint(pill, "Priority")
    }

    /// Bounce-count badge (icon + count + tooltip, matching the prereq indicator
    /// style). `None` when the ticket has not bounced. Muted (id color) at or
    /// below `BOUNCE_BADGE_WARNING_THRESHOLD`, then the prereq warning yellow.
    fn bounce_badge<'a>(
        bounce_count: i64,
        icon_size: f32,
        text_size: f32,
    ) -> Option<Element<'a, BoardMessage>> {
        if bounce_count <= 0 {
            return None;
        }
        let count = usize::try_from(bounce_count).unwrap_or(0);
        let color = if count > crate::pipeline::BOUNCE_BADGE_WARNING_THRESHOLD {
            theme::STATUS_WARNING
        } else {
            theme::TEXT_SECONDARY
        };
        let indicator = row![
            lucide::rotate_ccw::<iced::Theme, iced::Renderer>()
                .size(icon_size)
                .color(color),
            text(format!("{bounce_count}")).size(text_size).color(color),
        ]
        .spacing(theme::SPACE_2)
        .align_y(Alignment::Center);
        let tooltip_text = format!(
            "Verification bounces: {bounce_count} (max {})",
            crate::pipeline::MAX_BOUNCES,
        );
        Some(tooltip_hint(indicator, tooltip_text))
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
                let board = crate::pipeline::board::store();
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
            BoardMessage::Refreshed {
                tickets,
                generation,
            } => {
                // Stale-generation guard: drop a snapshot whose refresh was
                // superseded by a newer one so it cannot clobber the
                // current list.
                if self.applied_read(generation) {
                    self.apply_refreshed(tickets);
                }
                Task::none()
            }
            BoardMessage::RefreshError { error, generation } => {
                // Stale-result guard, mirroring `Refreshed`: a failed read for a
                // refresh the board has since superseded must not paint its
                // failure over the current list.
                if self.applied_read(generation) {
                    self.load_state.fail(error);
                }
                Task::none()
            }
            BoardMessage::TicketChanged(event) => self.apply_ticket_change(&event),
            BoardMessage::BoardRefreshNeeded => {
                // The CDC stream lagged (a delta was lost). Our delta-derived
                // state may be wrong (e.g. a removal delta was dropped, leaving
                // a resurrected ticket), so the recovery snapshot is
                // authoritative: mark a replace so the next Refreshed drops a
                // ticket a lost removal delta would have removed.
                self.replace_on_refresh = true;
                self.refresh()
            }
            BoardMessage::TicketUpserted(ticket) => {
                self.apply_ticket_upsert(ticket);
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
                // callbacks for the previous ticket, and clear the draft
                // so it is not accidentally sent to the new ticket.
                self.reset_comment_state();
                stats_task
            }
            BoardMessage::TicketDetailsRefreshed(generation, ticket) => {
                let ticket = *ticket;
                // The fetch completed — allow the next refresh.
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
                // refresh retries. Unlike `DetailError`, this
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
                        let board = crate::pipeline::board::store();
                        let phase: TicketPhase = new_phase
                            .parse()
                            .map_err(|_| format!("Invalid phase: {new_phase}"))?;
                        let ticket = board
                            .get_ticket(&ticket_id)
                            .await
                            .map_err(|e| e.to_string())?
                            .ok_or_else(|| format!("Ticket {ticket_id} not found"))?;
                        let source = ticket.phase;
                        // Cancelling a ticket in an implementation phase pauses
                        // the workspace so queued development tickets don't
                        // cascade. This is the phase-based gate
                        // (`is_pipeline_occupied`), independent of whether an
                        // agent is currently in flight. Supersede auto-cancels
                        // and Manager-tool cancels don't go through here;
                        // shutdown and already-paused are handled inside the
                        // helper.
                        if phase == TicketPhase::Cancelled && source.is_pipeline_occupied() {
                            // Pause the workspace so queued development tickets
                            // don't cascade. The pause is silent in ticket
                            // history.
                            crate::pipeline::pause_workspace_on_failure(
                                &ticket,
                                "user cancelled a ticket in the development pipeline",
                            )
                            .await;
                        }
                        // This is the authoritative user-cancel signal: stop any
                        // in-flight ticket agents so their work ends promptly.
                        // The GUI-path `transition_to(Cancelled)` below is the
                        // single source of the terminal state (the agent-side
                        // failure tail no longer drives a user cancel).
                        if phase == TicketPhase::Cancelled {
                            crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket_id);
                        }
                        // Re-dispatch into development is automatic on a
                        // post-development non-success — there is no manual
                        // "Redo Queued" action anymore.
                        board
                            .transition_to(&ticket_id, None, phase, "user")
                            .await
                            .map_err(|e| e.to_string())?;
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
                // state (`cancel_confirmation_needed`): a mid-pipeline ticket
                // requires explicit confirmation; otherwise the cancel stays
                // single-click, exactly as before. There is no async fetch
                // to race with, so no stale-callback machinery is needed —
                // `PerformAction` re-checks the phase at execution.
                if self.cancel_confirmation_needed(&ticket_id) {
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
                );
                Task::none()
            }
            BoardMessage::SendComment => {
                let text = self.comment_input.text();
                let trimmed = match super::common::send_guard(
                    &text,
                    self.sending_comment,
                    super::common::SendRules::Board,
                    |count| {
                        Task::done(BoardMessage::Toast(super::ToastMessage::Warning(format!(
                            "Comment too long: {count} characters (maximum {MAX_INPUT_CHARS}). \
                                 Please shorten your comment."
                        ))))
                    },
                ) {
                    Ok(t) => t.to_string(),
                    Err(task) => return task,
                };

                let Some(ticket_id) = self.selected_ticket.as_ref().map(|t| t.id.clone()) else {
                    return Task::none();
                };

                let role = format!("user:{}", crate::users::ADMIN_USER_NAME);
                let now = crate::db::now();
                let content = trimmed;

                // Optimistically add the comment to local state so it appears
                // immediately in the comments list.
                if let Some(ref mut ticket) = self.selected_ticket {
                    ticket.comments.push(crate::pipeline::board::TicketComment {
                        role: role.clone(),
                        content: content.clone(),
                        created_at: now,
                    });
                    self.comments_md = Self::parse_comments_md(&ticket.comments);
                }

                self.sending_comment = true;
                self.comment_input.clear();
                self.undo_stack.clear();
                let generation = self.comment_generation;

                Task::perform(
                    async move {
                        let board = crate::pipeline::board::store();
                        let result = board
                            .add_comment(&ticket_id, &role, &content)
                            .await
                            .map_err(|e| e.to_string());
                        (generation, result)
                    },
                    |(g, result)| BoardMessage::CommentSent(g, result),
                )
            }
            BoardMessage::CommentSent(generation, result) => {
                // Stale callback guard: if the modal closed or the user switched
                // tickets since this async was dispatched, drop the result.
                if generation != self.comment_generation {
                    self.sending_comment = false;
                    return Task::none();
                }
                self.sending_comment = false;
                if let Err(e) = &result {
                    tracing::warn!(error = %e, "Board: failed to add comment");
                }
                // Refresh ticket detail from DB for consistency; on error this
                // also removes the optimistic comment.
                let refresh = match self.selected_ticket.as_ref() {
                    Some(ticket) => Self::fetch_ticket(ticket.id.clone()),
                    None => Task::none(),
                };
                // The success toast requires an open modal; errors always surface.
                let toast = match (&result, self.selected_ticket.as_ref()) {
                    (Ok(()), Some(_)) => Task::done(BoardMessage::Toast(
                        super::ToastMessage::SuccessMsg("Comment added.".into()),
                    )),
                    (Ok(()), None) => Task::none(),
                    (Err(e), _) => Task::done(BoardMessage::Toast(super::ToastMessage::Error(
                        format!("Failed to add comment: {e}"),
                    ))),
                };
                Task::batch([refresh, toast])
            }
            BoardMessage::LinkClicked(_)
            | BoardMessage::Toast(_)
            | BoardMessage::ViewCommitDiff { .. } => {
                // Intercepted by the Dashboard before reaching board_state.update().
                // Arms must remain for match exhaustiveness even though functionally dead.
                Task::none()
            }
            BoardMessage::ArchiveAllCompleted => Task::perform(
                async move {
                    crate::pipeline::board::store()
                        .archive_all_done_and_cancelled()
                        .await
                        .map_err(|e| e.to_string())
                },
                BoardMessage::ArchiveAllCompletedResult,
            ),
            BoardMessage::ArchiveAllCompletedResult(Ok(count)) => {
                let toast = Task::done(BoardMessage::Toast(super::ToastMessage::SuccessMsg(
                    format!(
                        "Archived {count} ticket{} across all workspaces",
                        if count == 1 { "" } else { "s" }
                    ),
                )));
                Task::batch([self.refresh(), toast])
            }
            BoardMessage::ArchiveAllCompletedResult(Err(e))
            | BoardMessage::PauseAllQueuedResult(Err(e)) => {
                Task::done(BoardMessage::Toast(super::ToastMessage::Error(e)))
            }
            BoardMessage::PauseAllQueued => {
                let workspaces = self.queued_workspaces();
                Task::perform(
                    async move {
                        let board = crate::pipeline::board::store();
                        let mut total = 0u64;
                        for ws in &workspaces {
                            match board.drain_queued_to_planning(ws, "user").await {
                                Ok(moved) => total += moved,
                                Err(e) => return Err(e.to_string()),
                            }
                        }
                        Ok(total)
                    },
                    BoardMessage::PauseAllQueuedResult,
                )
            }
            BoardMessage::PauseAllQueuedResult(Ok(count)) => {
                // Count is the total moved across the drained workspaces: the
                // listed non-archived Queued tickets, each workspace drained
                // once.
                let toast = Task::done(BoardMessage::Toast(super::ToastMessage::SuccessMsg(
                    format!(
                        "Paused {count} ticket{} across all workspaces",
                        if count == 1 { "" } else { "s" }
                    ),
                )));
                Task::batch([self.refresh(), toast])
            }
            BoardMessage::ArchiveTicket(ticket_id) => {
                self.action_loading = Some(ticket_id.clone());
                Task::perform(
                    async move {
                        let board = crate::pipeline::board::store();
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
                        self.commit_stats_error = None;
                    }
                    Err(e) => {
                        // Non-critical: show the failure rather than rendering
                        // nothing under the commit-hash line.
                        self.commit_stats = None;
                        self.commit_stats_error = Some(e);
                    }
                }
                Task::none()
            }

            // ── Ticket search (sidebar) ──────────────────────────────
            BoardMessage::SearchInputChanged(action) => {
                // Single-line Tab moves focus rather than editing.
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                let changes_text = action.changes_text();
                self.search_query.apply_action(action);
                // The shared widget emits every action (cursor moves too);
                // only re-run the search/debounce pipeline on actual text
                // changes (edits and undo/redo).
                if !changes_text {
                    return Task::none();
                }
                let query = self.search_query.text();
                self.search_generation += 1;

                if query.is_empty() {
                    self.search_results.clear();
                    self.search_error = None;
                    return Task::none();
                }

                let generation = self.search_generation;
                Task::perform(
                    async move {
                        // Debounce: wait 300ms before executing the FTS query
                        // so rapid typing doesn't trigger per-keystroke DB hits.
                        //
                        // Effective latency is ~300ms plus the render after
                        // results arrive — the results render as soon as their
                        // message is delivered, so the debounce is the dominant
                        // wait. In-flight tasks from earlier
                        // keystrokes are NOT cancelled — they run to completion
                        // and their results are discarded by the generation
                        // counter guard below. For typical typing speeds this
                        // creates at most a handful of short-lived sleep-only
                        // tasks, which is acceptably cheap.
                        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                        let board = crate::pipeline::board::store();
                        board
                            .search_by_fts(&query, 20)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |res| match res {
                        Ok(tickets) => BoardMessage::SearchResults(tickets, generation),
                        Err(e) => BoardMessage::SearchError(e, generation),
                    },
                )
            }
            BoardMessage::SearchResults(tickets, generation) => {
                if generation == self.search_generation {
                    // Apply only if still the latest generation (stale callback guard)
                    self.search_results = tickets;
                    self.search_error = None;
                }
                Task::none()
            }
            BoardMessage::SearchError(e, generation) => {
                if generation == self.search_generation {
                    // A failed search is not an empty result: drop any previous
                    // results and render the failure instead.
                    self.search_results.clear();
                    self.search_error = Some(e);
                }
                Task::none()
            }
            BoardMessage::SearchCleared => {
                self.search_query.clear();
                self.search_results.clear();
                self.search_error = None;
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
    /// In Progress, Queued, Pending, Completed.
    pub(crate) fn board_sections(tickets: &[Ticket]) -> [Vec<&Ticket>; 4] {
        crate::pipeline::board::BoardStore::board_sections(tickets)
    }

    /// Render a single ticket card: clickable title, ID, phase badge, and action icons.
    #[expect(clippy::too_many_lines)]
    pub fn render_ticket_card<'a>(&'a self, ticket: &'a Ticket) -> Element<'a, BoardMessage> {
        let is_action_disabled = self.action_loading.as_deref() == Some(&ticket.id);

        let actions = Self::available_actions(ticket.phase);
        let mut icon_row =
            Self::action_icon_row(&ticket.id, ticket.phase, &actions, is_action_disabled);
        if actions.is_empty() {
            // Done/Cancelled cards render no action buttons; the ghost pins
            // the badge row's height to the button-bearing metrics.
            icon_row = icon_row.push(widgets::ghost_icon_button());
        }

        let (unfulfilled_count, unfulfilled_ids) = self.unfulfilled_prereq_count(ticket);

        let mut badge_row = row![
            Self::priority_badge(ticket.priority, PILL_COMPACT),
            Self::phase_badge(ticket.phase, PILL_COMPACT),
        ]
        .spacing(theme::SPACE_6);

        if let Some(bounce) =
            Self::bounce_badge(ticket.bounce_count, theme::TEXT_12, theme::TEXT_10)
        {
            badge_row = badge_row.push(bounce);
        }

        if unfulfilled_count > 0 {
            let tooltip_text = format!("Blocked by: {}", unfulfilled_ids.join(", "));
            let pause_icon = lucide::pause::<iced::Theme, iced::Renderer>()
                .size(theme::TEXT_12)
                .color(theme::STATUS_WARNING);
            let count_text = text(format!("{unfulfilled_count}"))
                .size(theme::TEXT_10)
                .color(theme::STATUS_WARNING);
            let indicator = row![pause_icon, count_text]
                .spacing(theme::SPACE_2)
                .align_y(Alignment::Center);
            badge_row = badge_row.push(tooltip_hint(indicator, tooltip_text));
        }

        // Inline commit stats: +added/−removed with color coding,
        // positioned after prereq indicator and before fill spacer.
        // Zero-valued sides are hidden; only the non-zero side displays.
        if let Some(hash) = &ticket.commit_hash {
            let added = ticket.lines_added.unwrap_or(0);
            let removed = ticket.lines_removed.unwrap_or(0);
            let stats_parts = row![
                text("\u{2387} ")
                    .size(theme::TEXT_10)
                    .color(theme::TEXT_SECONDARY),
                diff_stats_row::<BoardMessage>(added, removed, 10.0),
            ]
            .spacing(0)
            .align_y(Alignment::Center);
            let stats_button = widgets::icon_tooltip_button(
                stats_parts,
                "ticket commit diff",
                Some(BoardMessage::ViewCommitDiff {
                    commit_hash: hash.clone(),
                    workspace_name: ticket.workspace_name.clone(),
                }),
                [theme::PAD_2, theme::PAD_6],
                theme::button_text,
                tooltip::Position::Top,
            );
            badge_row = badge_row.push(stats_button);
        }

        badge_row = badge_row.push(Space::new().width(Length::Fill));
        badge_row = badge_row.push(icon_row);

        // Per-ticket archive button for done/cancelled tickets
        if matches!(ticket.phase, TicketPhase::Done | TicketPhase::Cancelled) && !ticket.is_archived
        {
            let archive_btn = widgets::icon_tooltip_button(
                lucide::archive::<iced::Theme, iced::Renderer>()
                    .size(widgets::ACTION_ICON_SIZE)
                    .color(theme::TEXT_MUTED),
                "archive ticket",
                if is_action_disabled {
                    None
                } else {
                    Some(BoardMessage::ArchiveTicket(ticket.id.clone()))
                },
                button::DEFAULT_PADDING,
                theme::button_text,
                tooltip::Position::Top,
            );
            badge_row = badge_row.push(archive_btn);
        }

        let mut card_children: Vec<Element<'_, BoardMessage>> = vec![
            // Title + ID row: both clickable
            tooltip_hint(
                button(
                    column![
                        text(&ticket.title)
                            .size(theme::TEXT_13)
                            .color(theme::TEXT_PRIMARY),
                        text(&ticket.id)
                            .size(theme::TEXT_10)
                            .color(theme::TEXT_SECONDARY),
                    ]
                    .spacing(theme::SPACE_2),
                )
                .padding(iced::Padding::new(theme::PAD_8).bottom(theme::PAD_2))
                .width(Length::Fill)
                .style(theme::button_text)
                .on_press(BoardMessage::OpenModal(ticket.id.clone())),
                "open ticket details",
            ),
        ];

        // Badge + optional prereq indicator + icon row (below the clickable area)
        card_children.push(
            badge_row
                .align_y(Alignment::Center)
                .padding([0.0, theme::PAD_8])
                .into(),
        );

        let card = Column::from_vec(card_children)
            .spacing(0)
            .width(Length::Fill);

        container(card)
            .style(theme::elevated_card_style)
            .width(Length::Fill)
            .into()
    }

    /// Whether a ticket detail modal is currently open (or loading).
    const fn is_modal_open(&self) -> bool {
        self.selected_ticket.is_some() || self.selected_loading || self.detail_error.is_some()
    }

    /// Shared detail-modal shell: `dialog_shell` at `width` on the
    /// close-on-click backdrop.
    fn modal_dialog<'a>(
        content: impl Into<Element<'a, BoardMessage>>,
        width: f32,
    ) -> Element<'a, BoardMessage> {
        let dialog = dialog::dialog_shell(content, width, 24.0);

        widgets::modal_backdrop(dialog, BoardMessage::CloseModal, 0.5)
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
                Self::modal_dialog(
                    column![
                        text("Loading details\u{2026}")
                            .size(theme::TEXT_16)
                            .color(theme::TEXT_SECONDARY),
                        Space::new().height(12),
                        text("Fetching ticket information\u{2026}")
                            .size(theme::TEXT_13)
                            .color(theme::TEXT_SECONDARY),
                    ]
                    .align_x(Alignment::Center),
                    400.0,
                )
            } else if self.selected_ticket.is_none()
                && let Some(ref err) = self.detail_error
            {
                Self::modal_dialog(
                    column![
                        text("Failed to load ticket")
                            .size(theme::TEXT_16)
                            .color(theme::STATUS_ERROR),
                        Space::new().height(12),
                        text(err).size(theme::TEXT_13).color(theme::TEXT_SECONDARY),
                        Space::new().height(16),
                        button(
                            text("Close")
                                .size(theme::TEXT_13)
                                .color(theme::TEXT_PRIMARY)
                                .align_x(Alignment::Center),
                        )
                        .style(theme::button_secondary)
                        .on_press(BoardMessage::CloseModal),
                    ]
                    .align_x(Alignment::Center),
                    400.0,
                )
            } else {
                Self::modal_dialog(self.modal_detail(), 720.0)
            }
        } else {
            // Keep Stack widget type stable to prevent MouseArea state
            // from becoming orphaned when the modal closes.
            iced::widget::stack([widgets::empty_stack_placeholder()]).into()
        };

        let confirm_layer = if let Some(ref ticket_id) = self.pending_cancel {
            Self::cancel_confirm_dialog(ticket_id)
        } else {
            // Type-stable placeholder so child 1's tag — and therefore the
            // outer Stack's shape — does not change when the confirmation
            // is dismissed.
            iced::widget::stack([widgets::empty_stack_placeholder()]).into()
        };

        // Stack sizes from its first child: when the details modal is closed
        // `detail_layer` is a zero-size placeholder, so without explicit Fill
        // sizing the confirm layer collapses to 0x0 and is invisible. Pin the
        // stack to the full content area so the confirm dialog renders from
        // the sidebar-card path too.
        iced::widget::stack([detail_layer, confirm_layer])
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Build the mid-pipeline cancel confirmation dialog for a ticket.
    ///
    /// Shown when the eligibility check (`RequestCancel` →
    /// [`BoardState::cancel_confirmation_needed`]) reports a mid-pipeline
    /// (implementation-phase) ticket; the listed consequences are real as of
    /// that check. If the ticket leaves the pipeline before the user confirms,
    /// `PerformAction` re-pauses authoritatively at execution, so the pause
    /// decision is always made there; the cancel transition itself is
    /// unconditional (no expected-phase guard).
    fn cancel_confirm_dialog(ticket_id: &str) -> Element<'_, BoardMessage> {
        widgets::modal_backdrop(
            dialog::confirm_dialog(
                dialog::dialog_title(format!("Cancel ticket {ticket_id}?")),
                dialog::dialog_body([
                    "This ticket is in the middle of the development pipeline. Confirming will:",
                    "• cancel the ticket immediately — it is archived \
                     automatically afterwards;\n\
                     • stop any running agent — the current tool may finish, \
                     but no further work happens;\n\
                     • leave uncommitted changes in the working tree — not \
                     committed, not reverted;\n\
                     • pause the workspace pipeline — all in-flight work stops \
                     and no pipeline stage advances until you manually resume it.",
                ]),
                [
                    dialog::DialogAction::secondary("Keep ticket", BoardMessage::DismissCancel),
                    dialog::DialogAction::danger(
                        "Cancel ticket",
                        BoardMessage::ConfirmCancel(ticket_id.to_string()),
                    ),
                ],
            ),
            BoardMessage::DismissCancel,
            0.5,
        )
    }

    /// Render the ticket detail modal content.
    fn modal_detail(&self) -> Element<'_, BoardMessage> {
        let Some(ticket) = &self.selected_ticket else {
            return text("No ticket selected.")
                .size(theme::TEXT_13)
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
            self.commit_stats_error.as_deref(),
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
                    .size(theme::TEXT_16)
                    .color(theme::TEXT_SECONDARY)
                    .into(),
            );
            sections.push(el);
        }

        // ── Scrollable content area ──────────────────────────────
        let scrollable_content = widgets::vscroll(
            Column::from_vec(sections)
                .spacing(theme::SPACE_4)
                .width(Length::Fill),
        );

        // ── Comment input area (pinned at the bottom) ────────────
        let input_area = self.render_comment_input();

        column![scrollable_content, input_area,]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Relative timestamp label falling back to the absolute format when the
    /// stored timestamp is unparseable ([`theme::format_relative_time`]
    /// yields an empty string then). Callers comparing multiple labels should
    /// pass one shared `now` so the outcome is deterministic within a render.
    fn relative_timestamp(ts: &str, now: chrono::DateTime<chrono::Local>) -> String {
        let relative = theme::format_relative_time(ts, now);
        if relative.is_empty() {
            theme::format_timestamp(ts)
        } else {
            relative
        }
    }

    /// Render the modal header: title row, ticket ID + created/updated metadata
    /// (with optional prerequisites/supersedes line below), then badges + action icons.
    fn render_header_metadata(
        ticket: &Ticket,
        is_action_disabled: bool,
    ) -> Element<'_, BoardMessage> {
        let actions = Self::available_actions(ticket.phase);
        let icon_row =
            Self::action_icon_row(&ticket.id, ticket.phase, &actions, is_action_disabled);

        let now = chrono::Local::now();
        let created = Self::relative_timestamp(&ticket.created_at, now);
        let updated = Self::relative_timestamp(&ticket.updated_at, now);

        // Hide the updated label only when created renders non-empty and both
        // display strings are identical (same relative bucket).
        let show_updated = created.is_empty() || created != updated;

        let mut meta_els: Vec<Element<'_, BoardMessage>> = vec![
            text(&ticket.id)
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
            text(" · ")
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
            text(format!("Workspace: {}", ticket.workspace_name))
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
            text(" · ")
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
            text(format!("Created: {created}"))
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
        ];
        if show_updated {
            meta_els.extend([
                text(" · ")
                    .size(theme::TEXT_12)
                    .color(theme::TEXT_SECONDARY)
                    .into(),
                text(format!("Updated: {updated}"))
                    .size(theme::TEXT_12)
                    .color(theme::TEXT_SECONDARY)
                    .into(),
            ]);
        }

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
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into();
            column![first_row, second_row]
                .spacing(theme::SPACE_2)
                .into()
        };

        column![
            text(&ticket.title)
                .size(theme::TEXT_16)
                .color(theme::TEXT_PRIMARY)
                .font(theme::FONT_BOLD),
            metadata_block,
            {
                let mut badges = vec![
                    Self::priority_badge(ticket.priority, PILL_MODAL),
                    Self::phase_badge(ticket.phase, PILL_MODAL),
                ];
                if let Some(bounce) =
                    Self::bounce_badge(ticket.bounce_count, theme::TEXT_14, theme::TEXT_12)
                {
                    badges.push(bounce);
                }
                badges.push(Space::new().width(Length::Fill).into());
                badges.push(icon_row.into());
                Row::from_vec(badges)
                    .align_y(Alignment::Center)
                    .spacing(theme::SPACE_8)
                    .padding([theme::PAD_4, 0.0])
            },
        ]
        .spacing(theme::SPACE_4)
        .into()
    }

    /// Render commit stats: summary header + per-file rows, or a loading
    /// indicator, or the fetch failure. Returns `None` when the ticket has no
    /// commit hash.
    fn render_commit_stats<'a>(
        ticket: &'a Ticket,
        stats: Option<&'a CommitStats>,
        loading: bool,
        error: Option<&'a str>,
    ) -> Option<Element<'a, BoardMessage>> {
        let hash = ticket.commit_hash.as_ref()?;

        if loading {
            return Some(
                column![Space::new().height(8), widgets::loading_text(),]
                    .spacing(theme::SPACE_4)
                    .into(),
            );
        }

        let Some(stats) = stats else {
            let error = error?;
            return Some(
                column![
                    Space::new().height(8),
                    text(format!("Commit stats unavailable: {error}"))
                        .size(theme::TEXT_12)
                        .color(theme::STATUS_WARNING),
                ]
                .spacing(theme::SPACE_4)
                .into(),
            );
        };

        let total_additions: i64 = stats.files.iter().map(|f| f.additions).sum();
        let total_deletions: i64 = stats.files.iter().map(|f| f.deletions).sum();
        let file_count = stats.files.len();
        let mut summary_parts: Vec<Element<'_, BoardMessage>> = Vec::new();
        summary_parts.push(
            text(format!("{hash:.8}"))
                .size(theme::TEXT_11)
                .color(theme::TEXT_SECONDARY)
                .into(),
        );
        summary_parts.push(Space::new().width(theme::SPACE_6).into());
        summary_parts.push(
            text(format!(
                "{file_count} file{} changed",
                if file_count == 1 { "" } else { "s" }
            ))
            .size(theme::TEXT_11)
            .color(theme::TEXT_SECONDARY)
            .into(),
        );
        summary_parts
            .push(diff_stats_row::<BoardMessage>(total_additions, total_deletions, 11.0).into());
        let summary_header = container(
            row(summary_parts)
                .spacing(theme::SPACE_4)
                .align_y(Alignment::Center),
        )
        .padding([theme::PAD_4, theme::PAD_8])
        .width(Length::Fill);

        // File stat rows — hide zero-valued sides
        let mut file_col = Column::new().spacing(theme::SPACE_2);
        for f in &stats.files {
            file_col = file_col.push(
                row![
                    container(text(&f.path).size(theme::TEXT_11).font(theme::FONT_REGULAR))
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
                    .padding([theme::PAD_4, theme::PAD_8])
                    .style(theme::surface_card_style),
            ]
            .spacing(theme::SPACE_4)
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
            container(widgets::vscroll_sized(
                super::selectable::markdown::view(items, theme::markdown_settings())
                    .map(BoardMessage::LinkClicked),
                Length::Fill,
                Length::Shrink,
            ))
            .width(Length::Fill)
            .padding([theme::PAD_10, theme::PAD_2])
            .style(theme::bubble_style(
                theme::BG_SURFACE,
                Some(theme::TEXT_PRIMARY),
            ))
            .into()
        } else {
            container(
                selectable_text(&ticket.description, theme::TEXT_PRIMARY).size(theme::TEXT_13),
            )
            .width(Length::Fill)
            .padding(theme::PAD_10)
            .style(theme::bubble_style(
                theme::BG_SURFACE,
                Some(theme::TEXT_PRIMARY),
            ))
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
                // Board keeps its legacy always-active send button look
                // (Home greys it out on empty input).
                grey_on_empty: false,
                send_tooltip: "send comment",
                id: Some(iced::widget::Id::new("board_comment_composer")),
                attachments_on_paste: false,
            },
        ))
        .width(Length::Fill)
        .into()
    }

    /// Render the comments list: per-comment chat-style author header, content,
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

        let mut cmt_col = Column::new().spacing(theme::SPACE_12);
        for (i, comment) in ticket.comments.iter().enumerate().rev() {
            // For diagnostics comments, optionally show only the summary
            let is_diag = comment.role == crate::agent::role::DIAGNOSTICS_ROLE;
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
                .size(theme::TEXT_13)
                .into()
            } else if let Some((_, items)) = comments_md.iter().find(|(idx, _)| *idx == i) {
                super::selectable::markdown::view(items, theme::markdown_settings())
                    .map(BoardMessage::LinkClicked)
            } else {
                selectable_text(&comment.content, theme::TEXT_PRIMARY)
                    .size(theme::TEXT_13)
                    .into()
            };

            // Toggle button for diagnostics comments
            let toggle_button: Option<Element<'_, BoardMessage>> = if is_diag {
                let (icon, label) = if is_expanded {
                    (
                        lucide::chevron_up::<iced::Theme, iced::Renderer>().size(theme::TEXT_12),
                        " Collapse",
                    )
                } else {
                    (
                        lucide::chevron_down::<iced::Theme, iced::Renderer>().size(theme::TEXT_12),
                        " Show full output",
                    )
                };
                Some(
                    button(
                        row![
                            icon.color(theme::TEXT_SECONDARY),
                            text(label)
                                .size(theme::TEXT_11)
                                .color(theme::TEXT_SECONDARY),
                        ]
                        .spacing(theme::SPACE_2)
                        .align_y(Alignment::Center),
                    )
                    .style(theme::button_text)
                    .on_press(BoardMessage::ToggleCommentExpand(i))
                    .into(),
                )
            } else {
                None
            };

            let author_el: Element<'_, BoardMessage> =
                if let Some((icon, color)) = theme::comment_author_icon(&comment.role) {
                    icon.size(theme::TEXT_14).color(color).into()
                } else {
                    // User comments show the bare user name; "system" and
                    // legacy labels render as plain muted text.
                    let label = comment.role.strip_prefix("user:").unwrap_or(&comment.role);
                    text(label)
                        .size(theme::TEXT_11)
                        .color(theme::TEXT_SECONDARY)
                        .into()
                };
            let mut header = row![author_el]
                .align_y(Alignment::Center)
                .spacing(theme::SPACE_6);
            let ts = theme::format_relative_time(&comment.created_at, chrono::Local::now());
            if !ts.is_empty() {
                header = header.push(text(ts).size(theme::TEXT_11).color(theme::TEXT_SECONDARY));
            }

            let mut comment_col = Column::new().spacing(theme::SPACE_4);
            comment_col = comment_col.push(header);
            comment_col = comment_col.push(comment_content);
            if let Some(btn) = toggle_button {
                comment_col = comment_col.push(btn);
            }

            cmt_col = cmt_col.push(
                container(comment_col)
                    .width(Length::Fill)
                    .padding(theme::PAD_10)
                    .style(theme::bubble_style(
                        theme::BG_SURFACE,
                        Some(theme::TEXT_PRIMARY),
                    )),
            );
        }

        Some(cmt_col.into())
    }
}

#[cfg(test)]
mod tests {
    use super::super::editor_widget::{EditorAction, EditorBuffer};
    use super::super::highlight::HighlightLanguage;
    use super::*;
    use crate::util::test::TicketFixture;

    fn make_board_state() -> BoardState {
        let mut state = BoardState::new();
        state.selected_ticket = Some(TicketFixture::new("T-1", TicketPhase::Backlog).build());
        state
    }

    fn make_ticket(id: &str, phase: TicketPhase) -> Ticket {
        TicketFixture::new(id, phase).build()
    }

    /// Build a ticket for the pause-helper tests in `workspace`, optionally
    /// archived.
    fn ticket_in(workspace: &str, id: &str, phase: TicketPhase, archived: bool) -> Ticket {
        TicketFixture::new(id, phase)
            .workspace(workspace)
            .archived(archived)
            .build()
    }

    fn draft(text: &str) -> EditorBuffer {
        EditorBuffer::with_text(text, Some(HighlightLanguage::Markdown))
    }

    // ── Comment input: cursor actions, editing and undo/redo ──────

    fn mv(line: usize, col: usize) -> EditorAction {
        EditorAction::MoveTo { line, col }
    }

    fn sel(line: usize, col: usize) -> EditorAction {
        EditorAction::SelectTo { line, col }
    }

    /// Typing `text` into the field, one insert per character.
    fn typed(text: &str) -> Vec<EditorAction> {
        text.chars().map(EditorAction::Insert).collect()
    }

    /// How a case ends the modal session, if at all.
    #[derive(Clone, Copy, PartialEq)]
    enum Close {
        /// The modal stays open.
        None,
        /// The `CloseModal` message (backdrop click or close button).
        Message,
        /// `reset_modal()` itself: the clear the message funnels through.
        Reset,
    }

    struct CursorUndoCase {
        name: &'static str,
        initial: &'static str,
        actions: Vec<EditorAction>,
        expected_text: &'static str,
        selection: bool,
        undoes: bool,
        close: Close,
    }

    #[test]
    fn test_comment_input_cursor_actions_and_undo_edits() {
        let text = "hello world";
        let cases = [
            CursorUndoCase {
                name: "a non-edit cursor move passes through",
                initial: text,
                actions: vec![mv(0, 0)],
                expected_text: text,
                selection: false,
                undoes: false,
                close: Close::None,
            },
            CursorUndoCase {
                name: "SelectTo creates a selection",
                initial: text,
                actions: vec![mv(0, 0), sel(0, 5)],
                expected_text: text,
                selection: true,
                undoes: false,
                close: Close::None,
            },
            CursorUndoCase {
                name: "a plain MoveTo keeps the content",
                initial: text,
                actions: vec![mv(0, 3)],
                expected_text: text,
                selection: false,
                undoes: false,
                close: Close::None,
            },
            CursorUndoCase {
                name: "undo restores content and redo restores the undone state",
                initial: "",
                actions: typed("abc"),
                expected_text: "abc",
                selection: false,
                undoes: true,
                close: Close::None,
            },
            CursorUndoCase {
                name: "redo with no undo history is a no-op",
                initial: "",
                actions: vec![EditorAction::Redo],
                expected_text: "",
                selection: false,
                undoes: false,
                close: Close::None,
            },
            CursorUndoCase {
                name: "undo is a no-op after a modal reset",
                initial: "",
                actions: typed("hel"),
                expected_text: "",
                selection: false,
                undoes: false,
                close: Close::Message,
            },
            CursorUndoCase {
                name: "reset_modal clears the undo history",
                initial: "",
                actions: typed("hel"),
                expected_text: "",
                selection: false,
                undoes: false,
                close: Close::Reset,
            },
        ];

        for case in cases {
            let name = case.name;
            let expected = case.expected_text;
            let mut state = make_board_state();
            state.comment_input = draft(case.initial);
            for action in case.actions {
                let _task = state.update(BoardMessage::CommentInputChanged(action));
                // Any interaction means the input is focused.
                assert!(state.comment_focused, "case {name}: comment_focused is set");
            }
            if case.undoes {
                // An undo with history must restore the content, and redo the undo.
                let before = state.comment_input.text();
                let _task = state.update(BoardMessage::CommentInputChanged(EditorAction::Undo));
                let changed = state.comment_input.text() != before;
                assert!(changed, "case {name}: undo should change the content");
                let _task = state.update(BoardMessage::CommentInputChanged(EditorAction::Redo));
            }
            // Closing the modal clears the field and its undo history.
            if case.close == Close::Reset {
                state.reset_modal();
            } else if case.close == Close::Message {
                let _task = state.update(BoardMessage::CloseModal);
            }
            if case.close != Close::None {
                assert_eq!(state.comment_input.text(), "", "case {name}: closed");
                let _task = state.update(BoardMessage::CommentInputChanged(EditorAction::Undo));
            }
            assert_eq!(state.comment_input.text(), expected, "case {name}: content");
            let selected = state.comment_input.cursor().selection.is_some();
            assert_eq!(selected, case.selection, "case {name}: selection");
        }
    }

    // ── SendComment: accept, no ticket, stale callback ────────────

    struct SendCommentCase {
        name: &'static str,
        send: bool,
        ticket_open: bool,
        draft: &'static str,
        expected_draft: &'static str,
        sending: bool,
        comments: usize,
    }

    /// A comment send needs a ticket, and a stale completion only drops the guard.
    #[test]
    fn test_comment_send_accepts_only_with_a_ticket_and_drops_stale_callbacks() {
        let cases = [
            SendCommentCase {
                name: "a within-limit comment is accepted and clears the input",
                send: true,
                ticket_open: true,
                draft: "hello world",
                expected_draft: "",
                sending: true,
                comments: 1,
            },
            SendCommentCase {
                name: "no selected ticket leaves the draft and sends nothing",
                send: true,
                ticket_open: false,
                draft: "hello",
                expected_draft: "hello",
                sending: false,
                comments: 0,
            },
            SendCommentCase {
                name: "a stale completion only releases the send guard",
                send: false,
                ticket_open: true,
                draft: "hello",
                expected_draft: "hello",
                sending: false,
                comments: 0,
            },
        ];

        for case in cases {
            let name = case.name;
            let mut state = make_board_state();
            if !case.ticket_open {
                state.selected_ticket = None;
            }
            state.comment_input = draft(case.draft);
            if case.send {
                let _task = state.update(BoardMessage::SendComment);
            } else {
                state.sending_comment = true;
                state.comment_generation = 42;
                // The stale guard fires before the Ok/Err branch, so one
                // completion covers both variants.
                let _task = state.update(BoardMessage::CommentSent(0, Ok(())));
            }
            assert_eq!(
                state.comment_input.text(),
                case.expected_draft,
                "case {name}: comment input"
            );
            assert_eq!(
                state.sending_comment, case.sending,
                "case {name}: sending_comment"
            );
            let optimistic = state
                .selected_ticket
                .as_ref()
                .map_or(0, |t| t.comments.len());
            assert_eq!(optimistic, case.comments, "case {name}: optimistic comment");
        }
    }

    // ── A new modal context invalidates the comment draft ─────────

    #[test]
    fn a_new_modal_context_bumps_the_comment_generation_and_clears_the_draft() {
        // (case, the context switch is a ticket switch rather than a fresh open)
        let cases = [
            ("opening the modal", false),
            ("switching to another ticket", true),
        ];
        for (name, switch_ticket) in cases {
            let mut state = make_board_state();
            state.comment_input = draft("draft comment");
            let before = state.comment_generation;
            let _task = if switch_ticket {
                state.sending_comment = true;
                state.comment_focused = true;
                let ticket = TicketFixture::new("T-2", TicketPhase::Backlog);
                state.update(BoardMessage::TicketDetails(Box::new(
                    ticket.title("New").build(),
                )))
            } else {
                state.update(BoardMessage::OpenModal("T-2".into()))
            };
            assert!(state.comment_generation > before, "case {name}: bumped");
            if switch_ticket {
                assert!(
                    state.comment_input.text().is_empty(),
                    "case {name}: cleared"
                );
                assert!(!state.sending_comment, "case {name}: send reset");
                assert!(!state.comment_focused, "case {name}: focus reset");
            }
        }
    }

    // ── TicketDetailsRefreshed (periodic refresh of the open modal) ─

    /// Same-ticket refresh carrying a new phase plus an engineer comment.
    fn refreshed_ticket(phase: TicketPhase) -> Ticket {
        TicketFixture::new("T-1", phase)
            .comment("engineer", "fixed the flaky test", "2026-01-01T00:00:05Z")
            .build()
    }
    struct TicketDetailsRefreshCase {
        name: &'static str,
        generation: u64,
        sending: bool,
        callback_generation: u64,
        phase: TicketPhase,
        comments: usize,
        applied: bool,
    }

    #[test]
    fn ticket_details_refresh_applies_only_when_fresh_and_idle() {
        let (backlog, verification) = (TicketPhase::Backlog, TicketPhase::Verification);
        let cases = [
            TicketDetailsRefreshCase {
                name: "a same-ticket refresh updates the detail and keeps the draft",
                generation: 0,
                sending: false,
                callback_generation: 0,
                phase: verification,
                comments: 1,
                applied: true,
            },
            TicketDetailsRefreshCase {
                name: "a stale-generation callback is dropped",
                generation: 42,
                sending: false,
                callback_generation: 0,
                phase: backlog,
                comments: 0,
                applied: false,
            },
            TicketDetailsRefreshCase {
                name: "a refresh never clobbers an optimistic comment send",
                generation: 7,
                sending: true,
                callback_generation: 7,
                phase: backlog,
                comments: 0,
                applied: false,
            },
        ];

        for case in cases {
            let name = case.name;
            let mut state = make_board_state(); // selected_ticket: T-1, Backlog
            state.comment_input = draft("draft comment");
            state.comment_generation = case.generation;
            state.sending_comment = case.sending;
            let gen_before = state.comment_generation;

            // The periodic refresh delivers the same ticket with a new phase
            // and a new engineer comment — the exact staleness the feature fixes.
            let _task = state.update(BoardMessage::TicketDetailsRefreshed(
                case.callback_generation,
                Box::new(refreshed_ticket(verification)),
            ));

            let ticket = state.selected_ticket.as_ref().expect("modal still open");
            assert_eq!(ticket.phase, case.phase, "case {name}: phase");
            assert_eq!(
                ticket.comments.len(),
                case.comments,
                "case {name}: comments"
            );
            assert!(
                !state.detail_refresh_in_flight,
                "case {name}: the completed fetch must clear the in-flight flag"
            );
            if case.applied {
                assert_eq!(
                    state.comment_input.text(),
                    "draft comment",
                    "case {name}: an in-progress comment draft must survive the refresh"
                );
                assert_eq!(
                    state.comment_generation, gen_before,
                    "case {name}: the refresh must not bump the generation (would invalidate in-flight callbacks)"
                );
                assert!(!state.sending_comment, "case {name}: no send in flight");
                assert!(
                    !state.commit_stats_loading,
                    "case {name}: no stats fetch is started"
                );
            }
        }
    }

    #[test]
    fn test_ticket_details_refreshed_stats_reconciliation() {
        const HASH: &str = "abcdef0123456789abcdef0123456789abcdef01";
        let hashed = || {
            let mut ticket = make_ticket("T-1", TicketPhase::Done);
            ticket.commit_hash = Some(HASH.into());
            ticket
        };

        // Hash appeared → fresh stats fetch dispatched.
        let mut state = make_board_state();
        let (g, stats_gen) = (state.comment_generation, state.commit_stats_generation);
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(g, Box::new(hashed())));
        assert!(state.commit_stats_loading, "new hash should fetch stats");
        assert!(
            state.commit_stats_generation > stats_gen,
            "stats generation bumps"
        );

        // Unchanged project + hash → keep the loaded stats, no refetch.
        let mut state = make_board_state();
        state.selected_ticket.as_mut().unwrap().commit_hash = Some(HASH.into());
        state.commit_stats = Some(CommitStats { files: Vec::new() });
        state.commit_stats_generation = 5;
        let g = state.comment_generation;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(g, Box::new(hashed())));
        assert_eq!(state.commit_stats_generation, 5, "no refetch");
        assert!(state.commit_stats.is_some(), "the loaded stats are kept");

        // The same hash in another project is a different (project, commit):
        // the stats loaded for the first one say nothing about the second, so
        // they are refetched rather than shown under the wrong project.
        let mut state = make_board_state();
        state.selected_ticket.as_mut().unwrap().commit_hash = Some(HASH.into());
        state.commit_stats = Some(CommitStats { files: Vec::new() });
        state.commit_stats_error = Some("Workspace 'first' not found".into());
        let mut elsewhere = hashed();
        elsewhere.workspace_name = "other_ws".into();
        let g = state.comment_generation;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(g, Box::new(elsewhere)));
        assert!(state.commit_stats_loading, "another project refetches");
        assert!(state.commit_stats.is_none());
        assert!(
            state.commit_stats_error.is_none(),
            "the first failure is gone"
        );

        // Hash vanished (latent) → loading must not stay stuck.
        let mut state = make_board_state();
        state.selected_ticket.as_mut().unwrap().commit_hash = Some(HASH.into());
        state.commit_stats_loading = true;
        let bare = make_ticket("T-1", TicketPhase::Done);
        let g = state.comment_generation;
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(g, Box::new(bare)));
        assert!(
            !state.commit_stats_loading,
            "a hash-less ticket clears loading"
        );
    }

    #[test]
    fn test_ticket_details_refresh_error_is_non_destructive() {
        let mut state = make_board_state();
        let g = state.comment_generation;

        // A refresh error keeps the last known good detail visible and must not
        // clear the modal loading state (that is `DetailError`'s job, for the
        // initial modal load).
        let _task = state.update(BoardMessage::TicketDetailsRefreshError(
            g,
            "db hiccup".into(),
        ));
        let shown = state.selected_ticket.as_ref().unwrap();
        assert_eq!(shown.phase, TicketPhase::Backlog, "the detail survives");
        assert_eq!(state.detail_error.as_deref(), Some("db hiccup"));
        assert!(!state.detail_refresh_in_flight);

        // Stale refresh error (generation mismatch) is dropped entirely.
        state.comment_generation += 1; // simulate modal close/switch
        let _task = state.update(BoardMessage::TicketDetailsRefreshError(0, "late".into()));
        assert_eq!(
            state.detail_error.as_deref(),
            Some("db hiccup"),
            "stale error"
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
        let g = state.comment_generation;
        let ticket = make_ticket("T-1", TicketPhase::Backlog);
        let _task = state.update(BoardMessage::TicketDetailsRefreshed(g, Box::new(ticket)));
        assert!(!state.detail_refresh_in_flight);

        // Initial modal load in flight → no refresh dispatched.
        let mut state = make_board_state();
        state.selected_loading = true;
        let _task = state.refresh_selected_ticket();
        assert!(!state.detail_refresh_in_flight);

        // Comment send in flight → no refresh dispatched.
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

        // First Escape: focused → blur (clear flag), modal stays open.
        let _task = state.update(BoardMessage::Escape);
        assert!(!state.comment_focused, "the first Escape blurs the input");
        assert!(state.selected_ticket.is_some(), "the modal stays open");

        // Second Escape: not focused → close modal.
        let _task = state.update(BoardMessage::Escape);
        assert!(
            state.selected_ticket.is_none(),
            "the second Escape closes it"
        );
    }

    // ── CDC change-stream delta application ───────────────────────

    fn page(id: &str, phase: TicketPhase) -> Vec<Ticket> {
        vec![make_ticket(id, phase)]
    }

    #[test]
    fn snapshots_follow_their_generation_and_lag_recovery_replaces() {
        // (case, (board generation, loading, replace on refresh, listed tickets),
        // a `Refreshed` payload or `None` for the lag-recovery marker,
        // (expected ids, expected replace flag, expected load state))
        let cases = [
            (
                "a stale-generation snapshot is dropped",
                (2, false, false, vec![]),
                Some((1, page("T-1", TicketPhase::Backlog))),
                (&[][..], false, None),
            ),
            (
                "a current-generation snapshot applies and finishes loading",
                (2, true, false, vec![]),
                Some((2, page("T-1", TicketPhase::Backlog))),
                (&["T-1"][..], false, Some(true)),
            ),
            (
                "a lag-recovery snapshot drops a lost ticket",
                (0, false, true, page("T-old", TicketPhase::Backlog)),
                Some((0, page("T-new", TicketPhase::Analysis))),
                (&["T-new"][..], false, None),
            ),
            (
                // A ticket a delta updated, later removed by a delta the lagged
                // broadcast dropped: the recovery snapshot must be able to drop it.
                "a lag recovery marks the next snapshot as authoritative",
                (0, false, false, page("T-1", TicketPhase::Done)),
                None,
                (&["T-1"][..], true, None),
            ),
        ];

        for (name, setup, snapshot, expected) in cases {
            let (generation_before, loading, replace, tickets) = setup;
            let mut state = BoardState::new();
            state.board_generation = generation_before;
            if loading {
                state.load_state.start_loading();
            }
            state.replace_on_refresh = replace;
            state.tickets = tickets;
            let _task = match snapshot {
                Some((generation, tickets)) => state.update(BoardMessage::Refreshed {
                    tickets,
                    generation,
                }),
                None => state.update(BoardMessage::BoardRefreshNeeded),
            };
            let (ids, replaces, loaded) = expected;
            let actual: Vec<&str> = state.tickets.iter().map(|t| t.id.as_str()).collect();
            assert_eq!(actual, ids, "case {name}: board tickets");
            assert_eq!(
                state.replace_on_refresh, replaces,
                "case {name}: replace flag"
            );
            if let Some(loaded) = loaded {
                assert_eq!(
                    state.load_state.has_loaded(),
                    loaded,
                    "case {name}: load state"
                );
            }
        }
    }

    #[test]
    fn refreshed_merge_keeps_newer_updated_at_and_does_not_readd_removed() {
        const OLD: &str = "2026-08-27T04:00:00.000000+00:00";
        const NEW: &str = "2026-08-27T04:00:01.000000+00:00";
        let refresh = |state: &mut BoardState, tickets: Vec<Ticket>| {
            let _task = state.update(BoardMessage::Refreshed {
                tickets,
                generation: 0,
            });
        };
        let mut state = BoardState::new();
        // A lag recovery already consumed: the next refresh merges.
        let mut ticket = make_ticket("T-1", TicketPhase::Backlog);
        ticket.updated_at = OLD.into();
        state.tickets = vec![ticket];

        // A newer snapshot updates the stale cached ticket.
        let mut newer = make_ticket("T-1", TicketPhase::Analysis);
        newer.updated_at = NEW.into();
        refresh(&mut state, vec![newer]);
        assert_eq!(state.tickets[0].phase, TicketPhase::Analysis);
        assert_eq!(state.tickets[0].updated_at, NEW);

        // A removed ticket is not re-added by a stale snapshot.
        state.delta_removed_ids.insert("T-2".into());
        let mut removed = make_ticket("T-2", TicketPhase::Done);
        removed.updated_at = OLD.into();
        refresh(&mut state, vec![removed]);
        assert!(
            state.tickets.iter().all(|t| t.id != "T-2"),
            "an archived ticket must not be re-added from a stale snapshot"
        );

        // A board ticket absent from a stale snapshot is kept (an omission is
        // ambiguous — the snapshot may predate its creation/delta), so a
        // delta-created ticket survives an older snapshot that omits it.
        let mut created = make_ticket("T-3", TicketPhase::Backlog);
        created.updated_at = OLD.into();
        state.tickets = vec![created];
        refresh(&mut state, vec![]);
        assert_eq!(state.tickets.len(), 1, "an omission preserves the ticket");
        assert_eq!(state.tickets[0].id, "T-3");
    }

    #[test]
    fn ticket_change_tracks_removal_for_snapshot_merge() {
        let mut state = BoardState::new();
        // The delta carries no workspace filter: a removal is applied to a
        // ticket of any workspace.
        let mut ticket = make_ticket("T-1", TicketPhase::Done);
        ticket.workspace_name = "other_ws".into();
        state.tickets = vec![ticket];
        let event = crate::db::cdc::ChangeEvent {
            table: "tickets".into(),
            change_id: 5,
            change_time: 1_750_000_000,
            change_type: crate::db::cdc::ChangeType::Delete,
            pk: crate::db::cdc::Pk::Text("T-1".into()),
            before: None,
            after: None,
        };
        let _task = state.apply_ticket_change(&event);
        assert!(
            state.tickets.is_empty(),
            "a DELETE (or archived row) is removed from the board list"
        );
        assert!(
            state.delta_removed_ids.contains("T-1"),
            "a removed ticket is tracked so a stale snapshot cannot re-add it"
        );
    }

    #[test]
    fn ticket_upsert_accepts_other_workspaces_and_dedupes_by_id() {
        let mut state = BoardState::new();
        let mut other = make_ticket("T-a", TicketPhase::Backlog);
        other.workspace_name = "other_ws".into();
        state.apply_ticket_upsert(other);
        assert_eq!(
            state.tickets.len(),
            1,
            "a ticket of any workspace is accepted"
        );

        let ticket = make_ticket("T-1", TicketPhase::InDevelopment);
        state.apply_ticket_upsert(ticket.clone());
        assert_eq!(state.tickets.len(), 2);
        // Idempotent upsert by id — no duplicate.
        state.apply_ticket_upsert(ticket);
        assert_eq!(state.tickets.len(), 2);

        // Archiving removes the ticket from the list.
        let mut archived = make_ticket("T-1", TicketPhase::Done);
        archived.is_archived = true;
        state.apply_ticket_upsert(archived);
        assert_eq!(state.tickets.len(), 1);
        assert!(state.tickets.iter().all(|t| t.id != "T-1"));
    }

    // ── Mid-pipeline cancel confirmation ─────────────────────────

    #[derive(Clone, Copy, PartialEq)]
    enum Follow {
        None,
        Dismiss,
        Confirm,
        Escape,
    }

    struct CancelConfirmCase {
        name: &'static str,
        ticket_id: &'static str,
        detail_phase: Option<TicketPhase>,
        listed_phase: Option<TicketPhase>,
        search_phase: Option<TicketPhase>,
        follow_up: Follow,
        opens_confirm: bool,
        stays_pending: bool,
    }

    const CANCEL_CONFIRM_CASES: &[CancelConfirmCase] = &[
        CancelConfirmCase {
            name: "a pipeline-occupied cancel opens the confirmation modal",
            ticket_id: "T-1",
            detail_phase: None,
            listed_phase: Some(TicketPhase::InDevelopment),
            search_phase: None,
            follow_up: Follow::None,
            opens_confirm: true,
            stays_pending: true,
        },
        CancelConfirmCase {
            name: "dismissal leaves the ticket untouched",
            ticket_id: "T-1",
            detail_phase: None,
            listed_phase: Some(TicketPhase::InDevelopment),
            search_phase: None,
            follow_up: Follow::Dismiss,
            opens_confirm: true,
            stays_pending: false,
        },
        CancelConfirmCase {
            name: "a non-pipeline cancel stays single-click",
            ticket_id: "T-1",
            detail_phase: None,
            listed_phase: Some(TicketPhase::Backlog),
            search_phase: None,
            follow_up: Follow::None,
            opens_confirm: false,
            stays_pending: false,
        },
        CancelConfirmCase {
            name: "confirming closes the modal",
            ticket_id: "T-1",
            detail_phase: None,
            listed_phase: Some(TicketPhase::InDevelopment),
            search_phase: None,
            follow_up: Follow::Confirm,
            opens_confirm: true,
            stays_pending: false,
        },
        CancelConfirmCase {
            name: "Escape dismisses the confirm modal before anything else",
            ticket_id: "T-1",
            detail_phase: Some(TicketPhase::InDevelopment),
            listed_phase: None,
            search_phase: None,
            follow_up: Follow::Escape,
            opens_confirm: true,
            stays_pending: false,
        },
        CancelConfirmCase {
            name: "the fresher open detail wins over a stale list",
            ticket_id: "T-1",
            detail_phase: Some(TicketPhase::Backlog),
            listed_phase: Some(TicketPhase::InDevelopment),
            search_phase: None,
            follow_up: Follow::None,
            opens_confirm: false,
            stays_pending: false,
        },
        CancelConfirmCase {
            name: "a pipeline-occupied detail opens the confirmation modal",
            ticket_id: "T-1",
            detail_phase: Some(TicketPhase::Verification),
            listed_phase: Some(TicketPhase::Backlog),
            search_phase: None,
            follow_up: Follow::None,
            opens_confirm: true,
            stays_pending: true,
        },
        CancelConfirmCase {
            name: "search results are consulted while a search is active",
            ticket_id: "T-1",
            detail_phase: None,
            listed_phase: Some(TicketPhase::Backlog),
            search_phase: Some(TicketPhase::Verification),
            follow_up: Follow::None,
            opens_confirm: true,
            stays_pending: true,
        },
        CancelConfirmCase {
            name: "an unknown ticket stays single-click",
            ticket_id: "T-unknown",
            detail_phase: Some(TicketPhase::Backlog),
            listed_phase: None,
            search_phase: None,
            follow_up: Follow::None,
            opens_confirm: false,
            stays_pending: false,
        },
    ];

    /// Only a pipeline-occupied ticket asks before cancelling.
    #[test]
    fn cancel_confirmation_opens_only_for_a_pipeline_occupied_ticket() {
        for case in CANCEL_CONFIRM_CASES {
            let name = case.name;
            let mut state = BoardState::new();
            state.selected_ticket = case.detail_phase.map(|phase| make_ticket("T-1", phase));
            state.tickets = case
                .listed_phase
                .map(|phase| vec![make_ticket("T-1", phase)])
                .unwrap_or_default();
            state.search_results = case
                .search_phase
                .map(|phase| vec![make_ticket("T-1", phase)])
                .unwrap_or_default();
            if case.follow_up == Follow::Escape {
                // The comment input is focused with the detail modal open.
                state.comment_focused = true;
            }

            let _task = state.update(BoardMessage::RequestCancel(case.ticket_id.into()));
            let asked = state.pending_cancel.as_deref();
            assert_eq!(
                asked,
                case.opens_confirm.then_some("T-1"),
                "case {name}: asked"
            );
            let _task = match case.follow_up {
                Follow::None => Task::none(),
                Follow::Dismiss => state.update(BoardMessage::DismissCancel),
                Follow::Confirm => state.update(BoardMessage::ConfirmCancel("T-1".into())),
                Follow::Escape => state.update(BoardMessage::Escape),
            };
            let pending_now = state.pending_cancel.as_deref();
            assert_eq!(
                pending_now,
                case.stays_pending.then_some("T-1"),
                "case {name}: pending"
            );
            if case.follow_up == Follow::Escape {
                assert!(state.comment_focused, "case {name}: Escape must not blur");
                assert!(state.selected_ticket.is_some(), "case {name}: modal stays");
            }
        }
    }

    // ── Ticket search (sidebar) ──────────────────────────────────

    struct SearchInputCase {
        name: &'static str,
        query: &'static str,
        had_results: bool,
        error: Option<&'static str>,
        generation: u64,
        pasted: Option<&'static str>,
        expected_generation: Option<u64>,
        drops_error: bool,
    }

    #[test]
    fn search_input_updates_query_and_generation() {
        let cases = [
            SearchInputCase {
                name: "an empty input clears the results",
                query: "",
                had_results: true,
                error: Some("stale"),
                generation: 5,
                pasted: Some(""),
                expected_generation: Some(6),
                drops_error: true,
            },
            SearchInputCase {
                name: "a non-empty input stores the query and bumps the generation",
                query: "",
                had_results: false,
                error: None,
                generation: 3,
                pasted: Some("network"),
                expected_generation: None,
                drops_error: false,
            },
            SearchInputCase {
                name: "clearing the search resets the query and results",
                query: "something",
                had_results: true,
                error: None,
                generation: 7,
                pasted: None,
                expected_generation: Some(8),
                drops_error: false,
            },
        ];

        for case in cases {
            let name = case.name;
            let mut state = make_board_state();
            state.search_query = super::super::common::SingleLineEditorState::new(case.query);
            if case.had_results {
                state.search_results = page("T-1", TicketPhase::Backlog);
            }
            state.search_generation = case.generation;
            state.search_error = case.error.map(str::to_string);

            match case.pasted {
                Some(query) => {
                    let action = EditorAction::Paste(query.to_string());
                    let _task = state.update(BoardMessage::SearchInputChanged(action));
                }
                None => {
                    let _task = state.update(BoardMessage::SearchCleared);
                }
            }

            // The field shows what was typed, and the results are gone.
            let expected_query = case.pasted.unwrap_or("");
            assert_eq!(
                state.search_query.text(),
                expected_query,
                "case {name}: query"
            );
            assert!(
                state.search_results.is_empty(),
                "case {name}: results empty"
            );
            // A text change always bumps the generation, so in-flight tasks
            // for the previous query are discarded.
            assert!(
                state.search_generation > case.generation,
                "case {name}: bumped"
            );
            if let Some(expected) = case.expected_generation {
                assert_eq!(state.search_generation, expected, "case {name}: generation");
            }
            assert_eq!(
                state.search_error.is_some(),
                case.error.is_some() && !case.drops_error,
                "case {name}: the previous failure is dropped"
            );
        }
    }

    #[test]
    fn search_results_and_errors_follow_the_generation() {
        let bl = TicketPhase::Backlog;
        // (case, (generation, stale results shown, error shown),
        // (the callback, expected ids, expected error))
        let cases = [
            (
                "a stale result callback is discarded",
                (99, false, None),
                (Ok((page("T-stale", bl), 50)), &[][..], None),
            ),
            (
                "a current result replaces the window and clears the failure",
                (42, false, Some("previous")),
                (Ok((page("T-fresh", bl), 42)), &["T-fresh"][..], None),
            ),
            (
                "a current failure clears the stale results",
                (7, true, None),
                (
                    Err(("fts unavailable", 7)),
                    &[][..],
                    Some("fts unavailable"),
                ),
            ),
            (
                "a stale failure is discarded",
                (9, false, None),
                (Err(("old failure", 3)), &[][..], None),
            ),
        ];

        for (name, before, callback) in cases {
            let (generation, results, error) = before;
            let (outcome, ids, wanted) = callback;
            let mut state = make_board_state();
            state.search_generation = generation;
            if results {
                state.search_results = page("T-stale", bl);
            }
            state.search_error = error.map(str::to_string);

            let _task = match outcome {
                Ok((tickets, generation)) => {
                    state.update(BoardMessage::SearchResults(tickets, generation))
                }
                Err((error, generation)) => {
                    state.update(BoardMessage::SearchError(error.to_string(), generation))
                }
            };

            let actual: Vec<&str> = state
                .search_results
                .iter()
                .map(|ticket| ticket.id.as_str())
                .collect();
            assert_eq!(actual, ids, "case {name}: results");
            assert_eq!(state.search_error.as_deref(), wanted, "case {name}: error");
        }
    }

    /// A failed board list read stays visible until a successful refresh replaces it.
    #[test]
    fn refresh_error_is_visible_until_a_successful_refresh() {
        let mut state = make_board_state();

        let _ = state.update(BoardMessage::RefreshError {
            error: "db down".to_string(),
            generation: state.board_generation,
        });
        assert_eq!(state.load_state.error(), Some("db down"));
        assert!(!state.load_state.has_loaded());

        // A failure carrying a stale generation must not overwrite the current
        // list's load state.
        let stale_generation = state.board_generation.wrapping_sub(1);
        let _ = state.update(BoardMessage::RefreshError {
            error: "stale".to_string(),
            generation: stale_generation,
        });
        assert_eq!(state.load_state.error(), Some("db down"));

        let _ = state.update(BoardMessage::Refreshed {
            tickets: Vec::new(),
            generation: state.board_generation,
        });
        assert!(state.load_state.error().is_none());
        assert!(state.load_state.has_loaded());
    }

    // ── Completed-column ordering ──────────────────────────────────

    /// Build a ticket for sort tests; defaults match the board-state fixture.
    fn test_ticket(
        id: &str,
        phase: TicketPhase,
        created_at: &str,
        done_at: Option<&str>,
    ) -> Ticket {
        TicketFixture::new(id, phase)
            .created_at(created_at)
            .done_at(done_at)
            .build()
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

    // ── Pause-all-queued helpers ──────────────────────────────────

    #[test]
    fn has_queued_tickets_reads_the_listed_board() {
        let cases = [
            (
                "true when a queued ticket is listed",
                vec![
                    ticket_in("ws1", "t1", TicketPhase::Queued, false),
                    ticket_in("ws1", "t2", TicketPhase::Backlog, false),
                ],
                false,
                true,
            ),
            (
                "false without a queued ticket",
                vec![
                    ticket_in("ws1", "t1", TicketPhase::Backlog, false),
                    ticket_in("ws1", "t2", TicketPhase::Planning, false),
                ],
                false,
                false,
            ),
            (
                "archived queued tickets do not count",
                vec![ticket_in("ws1", "t1", TicketPhase::Queued, true)],
                false,
                false,
            ),
            (
                "an active search only gates what the sidebar renders",
                vec![ticket_in("ws1", "t1", TicketPhase::Queued, false)],
                true,
                true,
            ),
        ];

        for (name, tickets, searching, expected) in cases {
            let mut state = make_board_state();
            state.tickets = tickets;
            if searching {
                // A Queued ticket in the listed board keeps the context-menu
                // item available even when the search shows none.
                state.search_query = super::super::common::SingleLineEditorState::new("foo");
                state.search_results = vec![ticket_in("ws1", "t2", TicketPhase::Backlog, false)];
            }
            assert_eq!(state.has_queued_tickets(), expected, "case {name}");
        }
    }

    #[test]
    fn the_ticket_column_is_hidden_only_by_a_loaded_empty_board() {
        let mut state = BoardState::new();
        // Not read yet: the column renders its "Loading…" state.
        assert!(!state.has_nothing_to_show());

        state.load_state.start_loading();
        assert!(!state.has_nothing_to_show());

        state.load_state.finish_loading();
        assert!(state.has_nothing_to_show(), "a read, empty board hides it");

        // A failed read never hides the column: the failure renders in it.
        state.load_state.fail("db hiccup".to_string());
        assert!(!state.has_nothing_to_show());

        // A ticket to show keeps it, whatever the read state says.
        state.load_state.clear_error();
        state.tickets = vec![ticket_in("ws1", "t1", TicketPhase::Backlog, false)];
        assert!(!state.has_nothing_to_show());
    }

    #[test]
    fn queued_workspaces_come_from_the_listed_board() {
        let cases = [
            (
                "derived from the listed queued tickets",
                vec![ticket_in("ws1", "t1", TicketPhase::Queued, false)],
                false,
                vec!["ws1"],
            ),
            (
                "derived and deduped across workspaces",
                vec![
                    ticket_in("ws1", "t1", TicketPhase::Queued, false),
                    ticket_in("ws1", "t2", TicketPhase::Queued, false),
                    ticket_in("ws2", "t3", TicketPhase::Queued, false),
                    ticket_in("ws3", "t4", TicketPhase::Planning, false),
                ],
                false,
                vec!["ws1", "ws2"],
            ),
            (
                "an active search never contributes its workspaces",
                vec![ticket_in("hidden", "t1", TicketPhase::Queued, false)],
                true,
                vec!["hidden"],
            ),
        ];

        for (name, tickets, searching, expected) in cases {
            let mut state = make_board_state();
            state.tickets = tickets;
            if searching {
                // The bulk pause covers the listed board: the workspace set
                // comes from `state.tickets`, never the search results.
                state.search_query = super::super::common::SingleLineEditorState::new("foo");
                state.search_results = vec![
                    ticket_in("ws1", "t2", TicketPhase::Queued, false),
                    ticket_in("ws2", "t3", TicketPhase::Queued, false),
                ];
            }
            let workspaces: Vec<String> = state.queued_workspaces();
            assert_eq!(workspaces, expected, "case {name}");
        }
    }
}
