//! Board dashboard page — ticket management.

use std::collections::HashSet;

use crate::board::{Ticket, TicketPhase, UNBLOCKING_STATUSES};

use iced::widget::{
    Column, Row, Space, button, column, container, markdown, row, scrollable, text, tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use super::theme;
use super::widgets;
use super::widgets::selectable_text;

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
    /// Short hash (7 chars).
    hash: String,
    files: Vec<FileStat>,
    /// Conditional summary like "3 files changed, +5" or "3 files changed, -3" or "3 files changed, +5/-2".
    summary: String,
}

#[derive(Debug, Clone)]
pub enum BoardMessage {
    Refreshed(Vec<Ticket>),
    RefreshError(String),
    TicketDetails(Box<Ticket>),
    DetailError(String),
    PerformAction(String, String), // ticket_id, new_status
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
}

pub struct BoardState {
    pub(crate) tickets: Vec<Ticket>,
    error: Option<String>,
    pub(crate) loading: bool,
    /// Whether at least one refresh has completed (prevents "Loading..." flicker
    /// on empty datasets when auto-poll Ticks).
    pub(crate) has_loaded: bool,
    selected_id: Option<String>,
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
}

impl BoardState {
    pub fn new() -> Self {
        Self {
            tickets: Vec::new(),
            error: None,
            loading: false,
            has_loaded: false,
            selected_id: None,
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
        }
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

    pub fn subscription(&self) -> iced::Subscription<BoardMessage> {
        iced::Subscription::none()
    }

    /// Status transition actions (ported from Board.tsx `availableActions`)
    fn available_actions(status: &TicketPhase) -> Vec<(&'static str, TicketPhase)> {
        match status {
            TicketPhase::Paused => vec![
                ("▶️ Resume Dev", TicketPhase::ReadyForDevelopment),
                ("↩️ Back to Backlog", TicketPhase::Backlog),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::ReadyForDevelopment => vec![("🛑 Cancel", TicketPhase::Cancelled)],
            TicketPhase::InDevelopment | TicketPhase::InQa => {
                vec![("🛑 Cancel", TicketPhase::Cancelled)]
            }
            TicketPhase::InReview => vec![("🛑 Cancel", TicketPhase::Cancelled)],
            TicketPhase::Reviewed => vec![
                ("✅ Send to QA", TicketPhase::InQa),
                ("🔄 Redo Dev", TicketPhase::ReadyForDevelopment),
                ("🛑 Cancel", TicketPhase::Cancelled),
            ],
            TicketPhase::QaPassed => {
                vec![("🛑 Cancel", TicketPhase::Cancelled)]
            }
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
    /// Match order: Cancel → Redo → Done → Backlog → Review → QA → Dev → fallback.
    fn action_icon<'a>(label: &str) -> iced::widget::Text<'a, iced::Theme, iced::Renderer> {
        if label.contains("Cancel") {
            lucide::circle_x::<iced::Theme, iced::Renderer>()
        } else if label.contains("Redo") {
            lucide::refresh_cw::<iced::Theme, iced::Renderer>()
        } else if label.contains("Done") {
            lucide::circle_check::<iced::Theme, iced::Renderer>()
        } else if label.contains("Backlog") {
            lucide::rotate_ccw::<iced::Theme, iced::Renderer>()
        } else if label.contains("Review") {
            lucide::eye::<iced::Theme, iced::Renderer>()
        } else if label.contains("QA") {
            lucide::shield_check::<iced::Theme, iced::Renderer>()
        } else if label.contains("Dev") {
            lucide::play::<iced::Theme, iced::Renderer>()
        } else {
            lucide::circle_check::<iced::Theme, iced::Renderer>()
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

    /// Compute how many of this ticket's prerequisites are still unfulfilled.
    /// A prerequisite is considered fulfilled if its ticket cannot be found in the
    /// loaded set (per manager clarification: missing = archived = fulfilled) or if
    /// its status is in [`UNBLOCKING_STATUSES`].
    fn unfulfilled_prereq_count(&self, ticket: &Ticket) -> (usize, Vec<String>) {
        if ticket.prerequisites.is_empty() {
            return (0, Vec::new());
        }
        let status_map: std::collections::HashMap<&str, &TicketPhase> = self
            .tickets
            .iter()
            .map(|t| (t.id.as_str(), &t.status))
            .collect();
        let mut unfulfilled_ids = Vec::new();
        for prereq_id in &ticket.prerequisites {
            let is_unfulfilled = match status_map.get(prereq_id.as_str()) {
                Some(status) => !UNBLOCKING_STATUSES.contains(status),
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

    pub fn update(&mut self, msg: BoardMessage) -> Task<BoardMessage> {
        match msg {
            BoardMessage::Refreshed(tickets) => {
                self.tickets = tickets;
                self.loading = false;
                self.has_loaded = true;
                Task::none()
            }
            BoardMessage::RefreshError(e) => {
                self.error = Some(e);
                self.loading = false;
                Task::none()
            }
            BoardMessage::OpenModal(id) => {
                self.selected_id = Some(id.clone());
                self.selected_loading = true;
                Self::fetch_ticket(id)
            }
            BoardMessage::CloseModal => {
                self.selected_id = None;
                self.selected_ticket = None;
                self.description_md = None;
                self.comments_md.clear();
                self.expanded_comments.clear();
                self.commit_stats = None;
                self.commit_stats_loading = false;
                self.commit_stats_generation += 1;
                Task::none()
            }
            BoardMessage::TicketDetails(ticket) => {
                let ticket = *ticket;
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
                self.error = Some(e);
                self.selected_loading = false;
                Task::none()
            }
            BoardMessage::PerformAction(ticket_id, new_status) => {
                self.action_loading = Some(ticket_id.clone());
                Task::perform(
                    async move {
                        let board = crate::board::store();
                        let phase: TicketPhase = new_status
                            .parse()
                            .map_err(|_| format!("Invalid status: {new_status}"))?;
                        board
                            .transition_to(&ticket_id, None, phase)
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
                let detail_refresh = self
                    .selected_id
                    .clone()
                    .map_or(Task::none(), Self::fetch_ticket);
                let toast = Task::done(BoardMessage::Toast(super::ToastMessage::Saved));
                Task::batch([refresh, detail_refresh, toast])
            }
            BoardMessage::ActionResult(Err(e)) => {
                self.action_loading = None;
                self.error = Some(e.clone());
                Task::done(BoardMessage::Toast(super::ToastMessage::Error(e)))
            }
            BoardMessage::Escape => {
                self.selected_id = None;
                self.selected_ticket = None;
                self.description_md = None;
                self.comments_md.clear();
                self.expanded_comments.clear();
                self.commit_stats = None;
                self.commit_stats_loading = false;
                self.commit_stats_generation += 1;
                Task::none()
            }
            BoardMessage::ToggleCommentExpand(i) => {
                if !self.expanded_comments.remove(&i) {
                    self.expanded_comments.insert(i);
                }
                Task::none()
            }
            BoardMessage::LinkClicked(_) => Task::none(),
            BoardMessage::Toast(_) => Task::none(),
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
                let id = ticket_id.clone();
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
                if self.selected_id.as_deref() != Some(&id)
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
            BoardMessage::ViewCommitDiff { .. } => {
                // Intercepted by Dashboard — cross-page navigation.
                Task::none()
            }
        }
    }

    /// Run `git show --numstat` (or `-m --numstat` for merges) and parse the output.
    async fn run_git_numstat(
        ws_path: &str,
        commit_hash: &str,
    ) -> Result<CommitStats, anyhow::Error> {
        // Detect merge commits with `git cat-file -t`
        let is_merge = match tokio::process::Command::new("git")
            .args(["cat-file", "-t", commit_hash])
            .current_dir(ws_path)
            .env("LC_ALL", "C")
            .output()
            .await
        {
            Ok(output) => output.stdout.trim_ascii_end() == b"commit",
            Err(_) => false, // if cat-file fails, assume non-merge
        };

        let mut cmd = tokio::process::Command::new("git");
        cmd.args(["show", "--numstat", "--format="]);
        if is_merge {
            cmd.arg("-m");
        }
        cmd.arg(commit_hash);

        let output = cmd.current_dir(ws_path).env("LC_ALL", "C").output().await?;

        if !output.status.success() {
            anyhow::bail!(
                "git show failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut files: Vec<FileStat> = Vec::new();
        let mut total_additions: i64 = 0;
        let mut total_deletions: i64 = 0;

        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Format: <additions>\t<deletions>\t<path>
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() != 3 {
                continue;
            }

            let additions_str = parts[0];
            let deletions_str = parts[1];
            let path = parts[2].to_string();

            // Binary files: displayed as "-\t-\t<path>"
            if additions_str == "-" && deletions_str == "-" {
                continue; // skip binary files
            }

            let additions: i64 = additions_str.parse().unwrap_or(0);
            let deletions: i64 = deletions_str.parse().unwrap_or(0);

            // Skip rename-only (0 additions, 0 deletions)
            if additions == 0 && deletions == 0 {
                continue;
            }

            total_additions += additions;
            total_deletions += deletions;
            files.push(FileStat {
                path,
                additions,
                deletions,
            });
        }

        let file_count = files.len();
        let summary = match (total_additions, total_deletions) {
            (0, 0) => format!(
                "{file_count} file{} changed, no changes",
                if file_count == 1 { "" } else { "s" }
            ),
            (a, 0) => format!(
                "{file_count} file{} changed, +{a}",
                if file_count == 1 { "" } else { "s" }
            ),
            (0, d) => format!(
                "{file_count} file{} changed, -{d}",
                if file_count == 1 { "" } else { "s" }
            ),
            (a, d) => format!(
                "{file_count} file{} changed, +{a}/-{d}",
                if file_count == 1 { "" } else { "s" }
            ),
        };

        let hash = commit_hash.chars().take(7).collect();

        Ok(CommitStats {
            hash,
            files,
            summary,
        })
    }

    pub fn view(&self) -> Element<'_, BoardMessage> {
        let mut content = column![];

        if let Some(ref err) = self.error {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(12));
        }

        if self.loading && !self.has_loaded {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
        } else if self.tickets.is_empty() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::layout_dashboard::<iced::Theme, iced::Renderer>(),
                "No tickets",
            ));
        } else {
            // Partition tickets into 3 columns
            let (pending, pipeline, completed) = Self::partition_tickets(&self.tickets);

            let kanban = row![
                self.render_kanban_column("Pending", &pending, false),
                Space::new().width(8),
                self.render_kanban_column("In Pipeline", &pipeline, false),
                Space::new().width(8),
                self.render_kanban_column("Completed", &completed, true),
            ]
            .height(Length::Fill);

            content = content.push(kanban);
        }

        let base = container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..container::Style::default()
            });

        iced::widget::stack([base.into(), self.render_modal_overlay()]).into()
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
            match ticket.status {
                TicketPhase::Backlog
                | TicketPhase::Analysis
                | TicketPhase::Planning
                | TicketPhase::Paused
                | TicketPhase::Failed => pending.push(ticket),
                TicketPhase::ReadyForDevelopment
                | TicketPhase::InDevelopment
                | TicketPhase::InDiagnostics
                | TicketPhase::DiagnosticsDone
                | TicketPhase::InReview
                | TicketPhase::Reviewed
                | TicketPhase::InQa
                | TicketPhase::QaPassed => pipeline.push(ticket),
                TicketPhase::Done | TicketPhase::Cancelled => completed.push(ticket),
            }
        }

        // Sort: pending and pipeline oldest-first (ASC), completed newest-first (DESC)
        // Ticket created_at is an ISO 8601 string, so lexical sort = chronological sort
        pending.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        pipeline.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        completed.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        (pending, pipeline, completed)
    }

    /// Render a single kanban column: header with count + scrollable ticket cards.
    /// When `is_completed` is true, a batch-archive button is shown in the header.
    fn render_kanban_column<'a>(
        &'a self,
        title: &'static str,
        tickets: &[&'a Ticket],
        is_completed: bool,
    ) -> Element<'a, BoardMessage> {
        let count = tickets.len();

        let mut header_row = row![
            text(title)
                .size(14)
                .color(theme::TEXT_SECONDARY)
                .font(theme::FONT_BOLD),
            Space::new().width(6),
            container(text(format!("{count}")).size(12).color(theme::TEXT_MUTED))
                .padding([1, 6])
                .style(|_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_SURFACE)),
                    border: iced::Border {
                        radius: 8.0.into(),
                        ..iced::Border::default()
                    },
                    ..container::Style::default()
                }),
        ]
        .align_y(Alignment::Center);

        if is_completed && !tickets.is_empty() {
            let archive_icon = lucide::archive::<iced::Theme, iced::Renderer>()
                .size(14)
                .color(theme::TEXT_SECONDARY);
            let archive_btn = tooltip(
                button(archive_icon)
                    .on_press(BoardMessage::ArchiveAllCompleted)
                    .padding(4)
                    .style(theme::button_text),
                text("Archive all done & cancelled").size(11),
                tooltip::Position::Top,
            );
            header_row = header_row.push(Space::new().width(6)).push(archive_btn);
        }

        let header = header_row;

        let column_content = if tickets.is_empty() {
            column![
                header,
                Space::new().height(16),
                container(text("No tickets").size(13).color(theme::TEXT_MUTED))
                    .width(Length::Fill)
                    .center_x(Length::Fill),
            ]
        } else {
            let mut cards = Column::new().spacing(4);
            for ticket in tickets {
                cards = cards.push(self.render_ticket_card(ticket));
            }
            column![
                header,
                Space::new().height(8),
                scrollable(cards)
                    .height(Length::Fill)
                    .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
                    .style(theme::scrollbar_style)
            ]
        };

        container(column_content)
            .padding(12)
            .width(Length::FillPortion(1))
            .height(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_SURFACE)),
                border: iced::Border {
                    radius: 4.0.into(),
                    width: 1.0,
                    color: theme::BORDER,
                },
                ..container::Style::default()
            })
            .into()
    }

    /// Render a single ticket card: clickable title, ID, status badge, and action icons.
    pub fn render_ticket_card<'a>(&'a self, ticket: &'a Ticket) -> Element<'a, BoardMessage> {
        let (badge_bg, badge_text) = theme::ticket_status_color(ticket.status);
        let is_action_disabled = self.action_loading.as_deref() == Some(&ticket.id);

        let actions = Self::available_actions(&ticket.status);
        let icon_row = Self::action_icon_row(&ticket.id, &actions, is_action_disabled);

        let (unfulfilled_count, unfulfilled_ids) = self.unfulfilled_prereq_count(ticket);

        let mut badge_row = row![
            container(
                text(ticket.status.display_name())
                    .size(10)
                    .color(badge_text),
            )
            .padding([1, 6])
            .style(move |_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(badge_bg)),
                border: iced::Border {
                    radius: 4.0.into(),
                    ..iced::Border::default()
                },
                ..container::Style::default()
            }),
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
            badge_row = badge_row.push(tooltip(
                indicator,
                text(tooltip_text).size(11),
                tooltip::Position::Top,
            ));
        }

        // Inline commit stats: +added/−removed with color coding,
        // positioned after prereq indicator and before fill spacer.
        // Zero-valued sides are hidden; only the non-zero side displays.
        if let (Some(hash), Some(ws_name)) = (&ticket.commit_hash, &self.workspace_name) {
            let added = ticket.lines_added.unwrap_or(0);
            let removed = ticket.lines_removed.unwrap_or(0);
            let mut stats_parts: Vec<Element<'_, BoardMessage>> = vec![
                text("\u{2387} ")
                    .size(10)
                    .color(theme::TEXT_SECONDARY)
                    .into(),
            ];
            if added > 0 {
                stats_parts.push(
                    text(format!("+{added}"))
                        .size(10)
                        .color(theme::STATUS_SUCCESS)
                        .into(),
                );
            }
            if added > 0 && removed > 0 {
                stats_parts.push(text("/").size(10).color(theme::TEXT_MUTED).into());
            }
            if removed > 0 {
                stats_parts.push(
                    text(format!("\u{2212}{removed}"))
                        .size(10)
                        .color(theme::STATUS_ERROR)
                        .into(),
                );
            }
            let stats_button = button(row(stats_parts).spacing(0).align_y(Alignment::Center))
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
        if matches!(ticket.status, TicketPhase::Done | TicketPhase::Cancelled)
            && !ticket.is_archived
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
            .padding(8)
            .width(Length::Fill)
            .style(theme::button_text)
            .on_press(BoardMessage::OpenModal(ticket.id.clone()))
            .into(),
        ];

        // Badge + optional prereq indicator + icon row (below the clickable area)
        card_children.push(badge_row.align_y(Alignment::Center).padding([4, 8]).into());

        let card = Column::from_vec(card_children)
            .spacing(2)
            .width(Length::Fill);

        container(card)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                border: iced::Border {
                    radius: 4.0.into(),
                    width: 1.0,
                    color: theme::BORDER,
                },
                ..container::Style::default()
            })
            .width(Length::Fill)
            .into()
    }

    /// Whether a ticket detail modal is currently open (or loading).
    #[must_use]
    pub fn is_modal_open(&self) -> bool {
        self.selected_ticket.is_some() || self.selected_loading
    }

    /// Render the modal overlay for ticket detail.
    /// Includes the empty-case placeholder for `Stack` widget type stability.
    #[must_use]
    pub fn render_modal_overlay(&self) -> Element<'_, BoardMessage> {
        if self.selected_ticket.is_some() || self.selected_loading {
            let backdrop = iced::widget::mouse_area(
                container(text(""))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(|_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(iced::Color::from_rgba(
                            0.0, 0.0, 0.0, 0.5,
                        ))),
                        ..container::Style::default()
                    }),
            )
            .on_press(BoardMessage::CloseModal);

            if self.selected_loading {
                let dialog = container(
                    column![
                        text("Loading details...").size(16).color(theme::TEXT_MUTED),
                        Space::new().height(12),
                        text("Fetching ticket information\u{2026}")
                            .size(13)
                            .color(theme::TEXT_MUTED),
                    ]
                    .align_x(Alignment::Center),
                )
                .width(Length::Fixed(400.0))
                .padding(24)
                .style(|_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                    border: iced::Border {
                        radius: 8.0.into(),
                        width: 1.0,
                        color: theme::BORDER_STRONG,
                    },
                    ..container::Style::default()
                });

                let centered = container(dialog)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center);

                iced::widget::stack([backdrop.into(), centered.into()]).into()
            } else {
                let detail = self.modal_detail();
                let dialog = container(detail)
                    .width(Length::Fixed(600.0))
                    .padding(24)
                    .style(|_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                        border: iced::Border {
                            radius: 8.0.into(),
                            width: 1.0,
                            color: theme::BORDER_STRONG,
                        },
                        ..container::Style::default()
                    });

                let centered = container(dialog)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center);

                iced::widget::stack([backdrop.into(), centered.into()]).into()
            }
        } else {
            // Keep Stack widget type stable to prevent MouseArea state
            // from becoming orphaned when the modal closes.
            iced::widget::stack([container(text(""))
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()])
            .into()
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

        let (badge_bg, badge_text) = theme::ticket_status_color(ticket.status);
        let is_action_disabled = self.action_loading.as_deref() == Some(&ticket.id);

        let actions = Self::available_actions(&ticket.status);
        let icon_row = Self::action_icon_row(&ticket.id, &actions, is_action_disabled);

        let mut detail = column![
            // Modal header row with title and close button
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
            // Ticket ID below title, matching board card layout
            text(&ticket.id).size(12).color(theme::TEXT_MUTED),
            Space::new().height(6),
            // Status badge + action icons row
            row![
                container(
                    text(ticket.status.display_name())
                        .size(12)
                        .color(badge_text)
                )
                .padding([2, 8])
                .style(move |_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(badge_bg)),
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..iced::Border::default()
                    },
                    ..container::Style::default()
                }),
                Space::new().width(Length::Fill),
                icon_row,
            ]
            .align_y(Alignment::Center)
            .spacing(8)
            .padding([4, 0]),
            text(format!(
                "Created: {}",
                theme::format_timestamp(&ticket.created_at)
            ))
            .size(12)
            .color(theme::TEXT_MUTED),
            {
                let reporter_label = if ticket.reporter.is_empty() {
                    "legacy"
                } else {
                    &ticket.reporter
                };
                text(format!("Reporter: {reporter_label}"))
                    .size(12)
                    .color(theme::TEXT_MUTED)
            },
            text(format!(
                "Updated: {}",
                theme::format_timestamp(&ticket.updated_at)
            ))
            .size(12)
            .color(theme::TEXT_MUTED),
        ]
        .spacing(2);

        if let Some(ref assignee) = ticket.assigned_to {
            detail = detail.push(
                text(format!("Assigned: {assignee}"))
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        }

        // Prerequisites
        if !ticket.prerequisites.is_empty() {
            detail = detail.push(
                text(format!(
                    "Prerequisites: {}",
                    ticket.prerequisites.join(", ")
                ))
                .size(12)
                .color(theme::TEXT_MUTED),
            );
        }

        // Supersedes
        if let Some(ref supersedes) = ticket.supersedes {
            detail = detail.push(
                text(format!("Supersedes: {supersedes}"))
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        }

        // Superseded by
        if let Some(ref superseded_by) = ticket.superseded_by {
            detail = detail.push(
                text(format!("Superseded by: {superseded_by}"))
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        }

        // Commit stats section
        if ticket.commit_hash.is_some() {
            if self.commit_stats_loading {
                detail = detail.push(Space::new().height(8));
                detail = detail.push(
                    text("Loading commit stats\u{2026}")
                        .size(12)
                        .color(theme::TEXT_MUTED),
                );
            } else if let Some(ref stats) = self.commit_stats {
                detail = detail.push(Space::new().height(10));
                detail = detail.push(
                    text(format!("Commit `{}`", stats.hash))
                        .size(12)
                        .color(theme::TEXT_PRIMARY),
                );

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

                detail = detail.push(container(file_col).padding([4, 8]).style(
                    |_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_SURFACE)),
                        border: iced::Border {
                            radius: 4.0.into(),
                            width: 1.0,
                            color: theme::BORDER,
                        },
                        ..container::Style::default()
                    },
                ));

                // Summary line
                detail = detail.push(Space::new().height(4));
                detail = detail.push(text(&stats.summary).size(10).color(theme::TEXT_MUTED));
            }
            // If loading is done but stats is None (error) → render nothing
        }

        // Description
        if !ticket.description.is_empty() {
            detail = detail.push(Space::new().height(8));
            detail = detail.push(text("Description:").size(13).color(theme::TEXT_SECONDARY));
            let desc_md: Element<'_, BoardMessage> = if let Some(ref items) = self.description_md {
                container(
                    scrollable(
                        iced_selection::markdown::view(items, theme::markdown_settings())
                            .map(BoardMessage::LinkClicked),
                    )
                    .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
                    .style(theme::scrollbar_style),
                )
                .padding(8)
                .style(|_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_SURFACE)),
                    border: iced::Border {
                        radius: 4.0.into(),
                        width: 1.0,
                        color: theme::BORDER,
                    },
                    ..container::Style::default()
                })
                .into()
            } else {
                container(selectable_text(&ticket.description, theme::TEXT_PRIMARY).size(13))
                    .padding(8)
                    .style(|_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_SURFACE)),
                        border: iced::Border {
                            radius: 4.0.into(),
                            width: 1.0,
                            color: theme::BORDER,
                        },
                        ..container::Style::default()
                    })
                    .into()
            };
            detail = detail.push(desc_md);
        }

        // Comments
        if !ticket.comments.is_empty() {
            detail = detail.push(Space::new().height(12));
            detail = detail.push(text("Comments:").size(13).color(theme::TEXT_SECONDARY));
            let mut cmt_col = Column::new().spacing(4);
            for (i, comment) in ticket.comments.iter().enumerate().rev() {
                let role_color = theme::role_badge_color(&comment.role).0;

                // For diagnostics comments, optionally show only the summary
                let is_diag = comment.role == "diagnostics";
                let is_expanded = self.expanded_comments.contains(&i);

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
                } else if let Some((_, items)) = self.comments_md.iter().find(|(idx, _)| *idx == i)
                {
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
                        container(text(&comment.role).size(11).color(role_color))
                            .padding([1, 6])
                            .style(move |_theme: &iced::Theme| container::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgba(
                                    role_color.r,
                                    role_color.g,
                                    role_color.b,
                                    0.1,
                                ),)),
                                border: iced::Border {
                                    radius: 4.0.into(),
                                    ..iced::Border::default()
                                },
                                ..container::Style::default()
                            }),
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

                cmt_col = cmt_col.push(container(comment_col).padding(8).style(
                    |_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_SURFACE)),
                        border: iced::Border {
                            radius: 4.0.into(),
                            width: 1.0,
                            color: theme::BORDER,
                        },
                        ..container::Style::default()
                    },
                ));
            }
            detail = detail.push(cmt_col);
        }

        scrollable(detail.spacing(4))
            .height(Length::Fill)
            .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
            .style(theme::scrollbar_style)
            .into()
    }
}
