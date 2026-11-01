//! Sessions dashboard page — view and manage conversation sessions. No rail entry: the
//! dashboard's Cmd+K shortcut is the only way in.
//!
//! The list is a live projection of the registry, not a snapshot: a session whose agent is running
//! is pinned to the top for as long as it runs — including one that has not stored its first
//! message yet — and rows are re-read on every runtime tick, so the page converges on its own.
//! See [`session_rows`].

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::ChatMessage;
use crate::Workspace;
use crate::session::SessionMetadata;

use iced::widget::{
    Column, Id, Space, button, column, container, responsive, row, scrollable, text,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;
use std::path::{Path, PathBuf};

use super::ToastMessage;
use super::media_markers;
use super::menus::{ContextMenu, MenuItem};
use super::session_view::preview::{
    BodyText, MAX_PREVIEW_LINES, MessageMeasure, measure_message, re_measure, width_bucket,
};
use super::session_view::{self, ParsedBody, SessionEntry, ToolCallEntry};
use super::theme;
use super::widgets::{self, BubbleSide, align_bubble, selectable_text};

/// Fraction of the transcript row width a chat bubble occupies (FillPortion 3 of a 3:1 row); the
/// collapse measurement narrows the measured width to the bubble body.
const BUBBLE_BODY_RATIO: f32 = 0.75;

/// Delay before a click-scheduled collapse applies: the widget's double-click window (300 ms) plus
/// a small margin, so a double-click's first click does not collapse before the second lands.
const COLLAPSE_COMMIT_DELAY: Duration = Duration::from_millis(350);

/// Max characters for the card title before an ellipsis.
const CARD_TITLE_MAX_CHARS: usize = 32;

/// Max characters of the branch identifier inside a custom-role card title — half the title
/// budget, so the role name keeps the rest.
const CARD_TITLE_BRANCH_MAX_CHARS: usize = CARD_TITLE_MAX_CHARS / 2;

/// Shared transcript render context: the flat session ledger, its per-entry parsed bodies, the
/// element expansion set, the collapse-measurement cache, the text measurement width, and whether
/// the ledger can be truncated (a stored, idle session).
struct TranscriptCtx<'a> {
    entries: &'a [SessionEntry],
    entry_md: &'a [Option<ParsedBody>],
    /// Root folder of the conversation's workspace; a relative file marker resolves against it.
    /// Owned because the `responsive` closure that builds the transcript is a `move` closure.
    workspace_root: Option<PathBuf>,
    expanded: &'a HashSet<(usize, usize)>,
    measure_cache: &'a RefCell<HashMap<(usize, usize), MessageMeasure>>,
    text_width: f32,
    truncatable: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum SessionsMessage {
    Settled(Result<Vec<SessionMetadata>, String>),
    SelectSession(String),
    SessionMessages(String, Vec<ChatMessage>),
    /// A transcript read failure. `key` is the session the failed read belongs to.
    SessionError {
        key: String,
        error: String,
    },
    /// Toggle an element's collapsed preview / full content, keyed by `id`.
    ToggleExpand(usize, usize),

    /// The second click of a double-click on a collapsible element: cancel the scheduled collapse.
    ToggleCancel(usize, usize),

    /// Apply a collapse previously scheduled by [`SessionsMessage::ToggleExpand`].
    ToggleCommit(usize, usize),

    ScrollChanged(scrollable::Viewport),

    Escape,

    LinkClicked(String),

    /// A `[FILE:…]` marker was clicked: reveal `target` in the file manager.
    RevealFile {
        target: String,
        workspace: Option<PathBuf>,
    },

    /// A [`SessionsMessage::RevealFile`] finished: an error becomes an error toast.
    FileRevealed(Result<(), String>),

    Toast(ToastMessage),
    DeleteSession(String),
    /// A session delete finished — remove it from the list. The bool is whether a row was removed.
    SessionDeleted(String, bool),

    /// Truncate the selected session: remove the ledger from this index on.
    TruncateSession(usize),
    /// A session truncate finished: the store removed `deleted` rows from index `from`.
    SessionTruncated {
        key: String,
        entry: usize,
        deleted: usize,
    },

    /// Coalesced runtime-change tick (250 ms): re-read the list so rows that started or stopped
    /// are reflected.
    RuntimeChanged,
}

#[derive(Debug, Clone)]
struct CachedSessionItem {
    agent_id: String,
    /// Agent role parsed from the metadata role string (`None` renders the fallback).
    role: Option<crate::Role>,
    /// User name when present and non-empty, otherwise the workspace name.
    title: String,
    /// The workspace the agent belongs to, as recorded in the session context.
    workspace_name: Option<String>,
    /// The custom role this session belongs to, when it is a custom-role session.
    custom_role: Option<String>,
    /// The named branch this continuable session continues, when named.
    custom_identifier: Option<String>,
    /// Pre-formatted message count, `None` only for a running agent whose first message is not
    /// stored yet.
    msg_count_label: Option<String>,
    /// Pre-formatted compact token count (the Running Agents format).
    token_label: Option<String>,
    timestamp_label: String,
    activity: SessionActivity,
}

/// What a session's agent is doing right now — the whole difference between a live row and a
/// stored one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionActivity {
    Idle,
    /// A registered agent is working in it: the row is pinned to the top of the list.
    Working,
    /// A registered agent was frozen by a workspace pause. It stops at its next checkpoint, so the
    /// row keeps its normal place.
    Paused,
}

impl CachedSessionItem {
    /// The row's title as the list shows it. A custom-role session is titled by its role — and its
    /// branch, when named — instead of the user/workspace name every other role shows; a worker
    /// branch is titled `Worker · <branch>` so it is distinguishable from a single worker run.
    ///
    /// A workspace's manager and maintainer belong to exactly one managed workspace, so their rows
    /// are titled by its managed name (never the folder basename, never the account name); once it
    /// is no longer managed the row shows the role alone rather than a stale name.
    fn display_title(&self, workspaces: &HashMap<String, Workspace>) -> String {
        match (self.role, self.custom_role.as_deref()) {
            (Some(crate::Role::Custom), Some(name)) => {
                custom_role_title(name, self.custom_identifier.as_deref())
            }
            (Some(crate::Role::Worker), None) => match self.custom_identifier.as_deref() {
                Some(branch) => format!("Worker{}", branch_suffix(branch)),
                None => self.title.clone(),
            },
            (Some(role @ (crate::Role::Manager | crate::Role::Maintainer)), _) => self
                .workspace_name
                .as_deref()
                .and_then(|name| workspaces.get(name))
                .map_or_else(
                    || role.display_label().to_string(),
                    |ws| crate::util::truncate(&ws.name, CARD_TITLE_MAX_CHARS),
                ),
            _ => self.title.clone(),
        }
    }
}

/// The compact ` · <branch>` suffix a branch row carries, under the title budget.
fn branch_suffix(identifier: &str) -> String {
    format!(
        " · {}",
        crate::util::truncate(identifier, CARD_TITLE_BRANCH_MAX_CHARS)
    )
}

/// Card title for a custom-role row: the compact `name · branch` spelling under the title budget,
/// the identifier capped at half of it.
fn custom_role_title(name: &str, identifier: Option<&str>) -> String {
    let Some(identifier) = identifier else {
        return crate::util::truncate(name, CARD_TITLE_MAX_CHARS);
    };
    let suffix = branch_suffix(identifier);
    let room = CARD_TITLE_MAX_CHARS - suffix.chars().count();
    format!("{}{suffix}", crate::util::truncate(name, room))
}

/// The session list's rows, in display order.
///
/// A working agent's session is pinned to the top for as long as it runs (ordered by run start, so
/// the block stays still) and settles into its last-activity place once it stops: a run in progress
/// must never be lost below the fold, and the list is not re-sorted mid-run. A paused (frozen)
/// agent is not working, so its row keeps its normal place.
///
/// A registered agent whose first message is not stored yet gets a synthesised row from what the
/// registry knows. Rows are keyed by `agent_id`, so it is replaced by the stored one as soon as the
/// first message lands.
fn session_rows(
    sessions: &[SessionMetadata],
    agents: &[crate::agent::registry::AgentHandle],
    now: chrono::DateTime<chrono::Local>,
) -> Vec<CachedSessionItem> {
    let mut live: HashMap<&str, &crate::agent::registry::AgentHandle> = agents
        .iter()
        .map(|handle| (handle.agent_id.as_str(), handle))
        .collect();
    let mut rows: Vec<ListRow> = Vec::with_capacity(sessions.len() + agents.len());
    for session in sessions {
        let handle = live.remove(session.agent_id.as_str());
        let activity = session_activity(handle);
        let name = session
            .user_name
            .as_deref()
            .filter(|n| !n.is_empty())
            .or_else(|| session.workspace_name.as_deref().filter(|n| !n.is_empty()))
            .unwrap_or("");
        rows.push(ListRow {
            pinned_at: pinned_at(handle, activity),
            last_activity: session.last_activity,
            row: CachedSessionItem {
                agent_id: session.agent_id.clone(),
                role: session
                    .role
                    .as_deref()
                    .and_then(|r| r.parse::<crate::Role>().ok()),
                title: crate::util::truncate(name, CARD_TITLE_MAX_CHARS),
                workspace_name: session.workspace_name.clone().filter(|n| !n.is_empty()),
                custom_role: session.custom_role.clone(),
                custom_identifier: session.custom_identifier.clone(),
                msg_count_label: Some(format!("{} msgs", session.message_count)),
                token_label: session.token_length.map(theme::format_compact_tokens),
                timestamp_label: theme::format_relative_time(
                    &session.last_activity.to_rfc3339(),
                    now,
                ),
                activity,
            },
        });
    }
    for handle in live.into_values() {
        let activity = session_activity(Some(handle));
        let workspace_name = Some(handle.workspace_name.clone()).filter(|n| !n.is_empty());
        let title = workspace_name
            .as_deref()
            .and_then(crate::users::personal_user_name)
            .unwrap_or(handle.workspace_name.as_str());
        rows.push(ListRow {
            pinned_at: pinned_at(Some(handle), activity),
            last_activity: handle.started_at,
            row: CachedSessionItem {
                agent_id: handle.agent_id.clone(),
                role: handle.role.parse::<crate::Role>().ok(),
                title: crate::util::truncate(title, CARD_TITLE_MAX_CHARS),
                workspace_name,
                custom_role: None,
                custom_identifier: None,
                msg_count_label: None,
                token_label: None,
                timestamp_label: theme::format_relative_time(&handle.started_at.to_rfc3339(), now),
                activity,
            },
        });
    }
    // A running row sorts by the run's start; the rest keep last-activity order (a
    // synthesised row's start stands in for activity).
    rows.sort_by(|a, b| match (a.pinned_at, b.pinned_at) {
        (Some(x), Some(y)) => x.cmp(&y).then_with(|| a.row.agent_id.cmp(&b.row.agent_id)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => b
            .last_activity
            .cmp(&a.last_activity)
            .then_with(|| a.row.agent_id.cmp(&b.row.agent_id)),
    });
    rows.into_iter().map(|ListRow { row, .. }| row).collect()
}

struct ListRow {
    row: CachedSessionItem,
    /// Start of the run the row is pinned for; `None` when it keeps its last-activity place.
    pinned_at: Option<chrono::DateTime<chrono::Utc>>,
    last_activity: chrono::DateTime<chrono::Utc>,
}

fn session_activity(handle: Option<&crate::agent::registry::AgentHandle>) -> SessionActivity {
    match handle {
        Some(handle) if handle.paused => SessionActivity::Paused,
        Some(_) => SessionActivity::Working,
        None => SessionActivity::Idle,
    }
}

/// The pin key of a row: the start of the run it is pinned for, `None` when it is not pinned.
fn pinned_at(
    handle: Option<&crate::agent::registry::AgentHandle>,
    activity: SessionActivity,
) -> Option<chrono::DateTime<chrono::Utc>> {
    handle
        .filter(|_| activity == SessionActivity::Working)
        .map(|handle| handle.started_at)
}

/// The root a transcript's relative `[FILE:…]` markers resolve against: the conversation's
/// workspace, or the selected one.
fn workspace_root_for(name: &str, workspaces: &HashMap<String, Workspace>) -> Option<PathBuf> {
    if let Some(user) = crate::users::personal_user_name(name) {
        return Some(crate::users::personal_workspace_path(user));
    }
    workspaces
        .get(name)
        .map(|ws| PathBuf::from(&ws.path))
        .or_else(|| Path::new(name).is_absolute().then(|| PathBuf::from(name)))
}

pub(crate) struct SessionsState {
    /// The stored session rows, re-read on page entry and on every runtime tick
    /// so rows that appear, grow or finish converge without the user leaving the page.
    sessions: super::common::PolledList<SessionMetadata>,
    selected_session: Option<String>,
    /// The flat session ledger built from the selected session's messages.
    entries: Vec<SessionEntry>,
    /// Per-entry parsed bodies, parallel to `entries` (parsed from the
    /// ledger via `session_view::parse_entry_bodies`).
    entry_md: Vec<Option<ParsedBody>>,
    /// The root this transcript's relative marker targets resolve against: the
    /// selected session's workspace, resolved when its messages load.
    workspace_root: Option<PathBuf>,
    /// For each entry, the index of the stored `sessions` row it was built from
    /// (see `session_view::build_ledger_with_sources`). `None` while the ledger
    /// has no stored-row identity: before the first stored load, and after any
    /// live merge — those indices are in-memory history positions, and a
    /// running agent may have rewritten the stored rows since the load. The
    /// truncate action is offered only while this is `Some` on an idle session,
    /// and comes back only on re-select, which reloads the stored ledger.
    entry_rows: Option<Vec<usize>>,
    selected_loading: bool,
    /// Last transcript read failure for the selected session. Rendered in the
    /// transcript pane (instead of the previous/empty transcript) until a
    /// successful load or a new selection clears it.
    selected_error: Option<String>,
    /// Elements the user expanded beyond their collapsed preview, keyed by
    /// (entry index, element index within entry): 0 = body/narration,
    /// 1 = thinking block, 2+j = result block of call `j`. Cleared on
    /// session switch.
    expanded: HashSet<(usize, usize)>,
    /// Expanded elements whose click-scheduled collapse is waiting out the
    /// double-click window (keyed like `expanded`). The second click of a
    /// double-click removes the entry — the double-click belongs to word/line
    /// selection — and a later [`SessionsMessage::ToggleCommit`] applies the
    /// collapse only if the entry survived. Cleared on session switch.
    pending_collapses: HashSet<(usize, usize)>,
    /// Per-element collapse measurement cache, keyed like `expanded`. The
    /// ledger only appends or drops its tail, so a surviving entry keeps its
    /// index and its key stays valid — only new elements are measured lazily.
    /// Mutated during layout (the bubble width is only known there), hence `RefCell`.
    measure_cache: RefCell<HashMap<(usize, usize), MessageMeasure>>,
    /// Cached session list display data. Rebuilt whenever the row set changes —
    /// a settled read or a runtime tick (which also moves relative-time labels
    /// and the live rows). `view()` builds widgets from this data on every frame.
    cached_session_items: Option<Vec<CachedSessionItem>>,
    /// A read was asked for while one was in flight: its rows are stale by the
    /// time they land, so one more read follows the settled one. This is what
    /// makes the last event of a burst count.
    refresh_pending: bool,

    // Transcript viewport fields
    /// Stable scrollable ID for the transcript area, preserves scroll position
    /// across widget rebuilds.
    scrollable_id: Id,
    /// Whether auto-scroll-to-bottom is enabled (user is at or near the bottom).
    auto_scroll_enabled: bool,
}

impl SessionsState {
    pub(crate) fn new() -> Self {
        Self {
            sessions: super::common::PolledList::new(),
            selected_session: None,
            entries: Vec::new(),
            entry_md: Vec::new(),
            workspace_root: None,
            entry_rows: None,
            selected_loading: false,
            selected_error: None,
            expanded: HashSet::new(),
            pending_collapses: HashSet::new(),
            measure_cache: RefCell::new(HashMap::new()),
            cached_session_items: None,
            refresh_pending: false,
            scrollable_id: Id::new("session_transcript_scroll"),
            auto_scroll_enabled: false,
        }
    }

    /// Re-read the session list: entering the page, every runtime tick, and after a delete or
    /// truncation. A read already in flight makes this a no-op that queues one more read for when
    /// it settles, so ticks never pile up and the last change of a burst is never lost. A failed
    /// read keeps the rows already on screen.
    pub(crate) fn refresh(&mut self) -> Task<SessionsMessage> {
        if !self.sessions.begin() {
            self.refresh_pending = true;
            return Task::none();
        }
        Task::perform(
            async {
                crate::session::store()
                    .list_sessions_with_metadata_checked()
                    .await
                    .map_err(|e| e.to_string())
            },
            SessionsMessage::Settled,
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per transcript interaction; the per-session selection bookkeeping is only readable inline"
    )]
    pub(crate) fn update(
        &mut self,
        msg: SessionsMessage,
        workspaces: &HashMap<String, Workspace>,
    ) -> Task<SessionsMessage> {
        match msg {
            SessionsMessage::Settled(result) => {
                self.sessions.settle(result);
                // The selected session's workspace may only be named now: adopt it.
                if self.adopt_workspace_root(workspaces) {
                    self.entry_md = session_view::parse_entry_bodies(
                        &self.entries,
                        self.workspace_root.as_deref(),
                    );
                }
                self.rebuild_session_cache();
                // Changes that landed while this read was in flight are older than the reply.
                if std::mem::take(&mut self.refresh_pending) {
                    return self.refresh();
                }
                Task::none()
            }
            SessionsMessage::SelectSession(key) => {
                self.selected_session = Some(key.clone());
                self.selected_loading = true;
                self.selected_error = None;
                self.expanded.clear();
                self.pending_collapses.clear();
                // The measurement cache is cleared in SessionMessages when the new session's
                // messages arrive; a failure renders the error rather than the previous transcript.
                // Do NOT set auto_scroll_enabled here — ScrollChanged owns it.
                Task::perform(
                    async move {
                        let store = crate::session::store();
                        match store.load_checked(&key).await {
                            Ok(messages) => Ok::<_, (String, String)>((key, messages)),
                            Err(e) => Err((key, e.to_string())),
                        }
                    },
                    |res| match res {
                        Ok((key, messages)) => SessionsMessage::SessionMessages(key, messages),
                        Err((key, error)) => SessionsMessage::SessionError { key, error },
                    },
                )
            }
            SessionsMessage::SessionMessages(key, messages) => {
                if self.selected_session.as_deref() == Some(&key) {
                    let (entries, rows) = session_view::build_ledger_with_sources(&messages);
                    self.entries = entries;
                    self.entry_rows = Some(rows);
                    self.adopt_workspace_root(workspaces);
                    self.entry_md = session_view::parse_entry_bodies(
                        &self.entries,
                        self.workspace_root.as_deref(),
                    );
                    // Fresh session load: reset the measurement cache.
                    self.measure_cache.borrow_mut().clear();
                    self.pending_collapses.clear();
                    self.selected_loading = false;
                    self.selected_error = None;
                    // Snap to bottom so the user sees the most recent messages.
                    return iced::widget::operation::snap_to_end(self.scrollable_id.clone());
                }
                Task::none()
            }
            SessionsMessage::SessionError { key, error } => {
                // A slow read for a session the user has since left must not land.
                if self.selected_session.as_deref() == Some(&key) {
                    self.selected_error = Some(error);
                    self.selected_loading = false;
                }
                Task::none()
            }
            SessionsMessage::ScrollChanged(viewport) => {
                // The fit check must stay first: `relative_offset` divides by the content height.
                self.auto_scroll_enabled = viewport.content_bounds().height
                    <= viewport.bounds().height
                    || viewport.relative_offset().y >= 0.99;
                Task::none()
            }
            SessionsMessage::ToggleExpand(i, j) => self.handle_toggle_expand(i, j),
            SessionsMessage::ToggleCancel(i, j) => {
                // Double-click word/line selection: the scheduled collapse is canceled.
                self.pending_collapses.remove(&(i, j));
                Task::none()
            }
            SessionsMessage::ToggleCommit(i, j) => {
                if self.pending_collapses.remove(&(i, j)) {
                    self.expanded.remove(&(i, j));
                }
                Task::none()
            }
            SessionsMessage::DeleteSession(key) => Task::perform(
                async move {
                    // User-initiated deletion: abandon the session's unfinished jobs before the
                    // rows go.
                    crate::jobs::abandon_session_jobs(&key)
                        .await
                        .map_err(|e| e.to_string())?;
                    let store = crate::session::store();
                    let deleted = store.delete(&key).await.map_err(|e| e.to_string())?;
                    Ok::<_, String>((key, deleted))
                },
                |res| match res {
                    Ok((key, deleted)) => SessionsMessage::SessionDeleted(key, deleted),
                    Err(e) => SessionsMessage::Toast(ToastMessage::Error(e)),
                },
            ),
            SessionsMessage::SessionDeleted(key, deleted) => {
                self.sessions.entries_mut().retain(|s| s.agent_id != key);
                self.rebuild_session_cache();
                if self.selected_session.as_deref() == Some(&key) {
                    self.clear_selection();
                }
                // A read in flight may predate the delete and would put the row back.
                let refresh = self.refresh();
                // Only claim a "Deleted" success when a real row removal happened; an
                // already-absent session just vanishes from the list without an inaccurate toast.
                let toast = if deleted {
                    Task::done(SessionsMessage::Toast(ToastMessage::Deleted))
                } else {
                    Task::none()
                };
                Task::batch([refresh, toast])
            }
            SessionsMessage::TruncateSession(entry) => {
                let Some(key) = self.selected_session.clone() else {
                    return Task::none();
                };
                // The stored row the entry was built from; without one the cut is a no-op.
                let Some(row) = self
                    .entry_rows
                    .as_ref()
                    .and_then(|rows| rows.get(entry).copied())
                else {
                    return Task::none();
                };
                Task::perform(
                    async move {
                        let deleted = crate::session::store()
                            .truncate_from(&key, row)
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok::<_, String>((key, entry, deleted))
                    },
                    |res| match res {
                        Ok((key, entry, deleted)) => SessionsMessage::SessionTruncated {
                            key,
                            entry,
                            deleted,
                        },
                        Err(e) => SessionsMessage::Toast(ToastMessage::Error(e)),
                    },
                )
            }
            SessionsMessage::SessionTruncated {
                key,
                entry,
                deleted,
            } => {
                // A no-op cut has nothing to drop. A live merge landing between the click and this
                // reply can at worst prune at a stale index — cosmetic: the store cut is durable
                // and a re-select corrects the view.
                if deleted > 0 && self.selected_session.as_deref() == Some(key.as_str()) {
                    self.entries.truncate(entry);
                    self.entry_md.truncate(entry);
                    if let Some(rows) = self.entry_rows.as_mut() {
                        rows.truncate(entry);
                    }
                    self.expanded.retain(|(i, _)| *i < entry);
                    self.pending_collapses.retain(|(i, _)| *i < entry);
                    self.measure_cache
                        .borrow_mut()
                        .retain(|(i, _), _| *i < entry);
                    // The card's count renders from the list snapshot: drop the cached one.
                    if let Some(session) = self
                        .sessions
                        .entries_mut()
                        .iter_mut()
                        .find(|s| s.agent_id == key)
                    {
                        session.message_count = session.message_count.saturating_sub(deleted);
                    }
                    self.rebuild_session_cache();
                    return self.refresh();
                }
                Task::none()
            }
            SessionsMessage::RuntimeChanged => self.on_runtime_changed(workspaces),
            SessionsMessage::Toast(_) | SessionsMessage::LinkClicked(_) => Task::none(),
            SessionsMessage::RevealFile { target, workspace } => Task::perform(
                // The reveal stats and spawns a launcher: keep it out of the update thread.
                async move { media_markers::reveal_file_marker(&target, workspace.as_deref()) },
                SessionsMessage::FileRevealed,
            ),
            SessionsMessage::FileRevealed(result) => match result {
                Ok(()) => Task::none(),
                Err(error) => Task::done(SessionsMessage::Toast(ToastMessage::Error(error))),
            },
            SessionsMessage::Escape => {
                self.clear_selection();
                Task::none()
            }
        }
    }

    // Click-to-toggle of one collapsible element: expanding applies immediately, collapsing waits
    // out the double-click window.
    fn handle_toggle_expand(&mut self, i: usize, j: usize) -> Task<SessionsMessage> {
        let key = (i, j);
        if self.expanded.contains(&key) {
            self.pending_collapses.insert(key);
            Task::perform(
                async { tokio::time::sleep(COLLAPSE_COMMIT_DELAY).await },
                move |()| SessionsMessage::ToggleCommit(i, j),
            )
        } else {
            self.expanded.insert(key);
            Task::none()
        }
    }

    // Clear the currently selected session and ALL per-session transcript state.
    fn clear_selection(&mut self) {
        self.selected_session = None;
        self.entries.clear();
        self.entry_md.clear();
        self.entry_rows = None;
        self.workspace_root = None;
        self.selected_loading = false;
        self.selected_error = None;
        self.expanded.clear();
        self.pending_collapses.clear();
        self.auto_scroll_enabled = false;
        self.measure_cache.borrow_mut().clear();
    }

    // Rebuild the cached session list display data from the settled rows.
    fn rebuild_session_cache(&mut self) {
        let items = session_rows(
            self.sessions.entries(),
            &crate::agent::registry::AGENT_REGISTRY.list(),
            chrono::Local::now(),
        );
        self.cached_session_items = if items.is_empty() { None } else { Some(items) };
    }

    // Coalesced runtime-change tick: re-read the list, rebuild the display data, merge the live
    // transcript.
    fn on_runtime_changed(
        &mut self,
        workspaces: &HashMap<String, Workspace>,
    ) -> Task<SessionsMessage> {
        self.rebuild_session_cache();
        Task::batch([self.merge_selected_transcript(workspaces), self.refresh()])
    }

    // The root the selected session's relative marker targets resolve against.
    fn selected_workspace_root(&self, workspaces: &HashMap<String, Workspace>) -> Option<PathBuf> {
        self.selected_session
            .as_deref()
            .and_then(|key| {
                self.sessions
                    .entries()
                    .iter()
                    .find(|session| session.agent_id == key)
            })
            .and_then(|session| session.workspace_name.as_deref())
            .and_then(|name| workspace_root_for(name, workspaces))
    }

    // Adopt the selected session's workspace root: `true` when it changed.
    fn adopt_workspace_root(&mut self, workspaces: &HashMap<String, Workspace>) -> bool {
        let root = self.selected_workspace_root(workspaces);
        if root == self.workspace_root {
            return false;
        }
        self.workspace_root = root;
        self.measure_cache.borrow_mut().clear();
        true
    }

    /// Merge the selected session's live transcript into the view while its agent is running.
    ///
    /// Append-oriented and diffed by the longest common entry prefix, so markdown re-parsing is
    /// limited to the changed tail and measurement-cache keys of unchanged entries stay valid.
    /// Index-keyed expansion sets stay valid because the tail is append-only and out-of-range keys
    /// are harmless. Snapshot lookup doubles as the running check — `None` when the agent is not
    /// running; a recycled agent id surfaces the new run's fresh holder, replacing the transcript.
    fn merge_selected_transcript(
        &mut self,
        workspaces: &HashMap<String, Workspace>,
    ) -> Task<SessionsMessage> {
        let Some(ref key) = self.selected_session else {
            return Task::none();
        };
        if self.selected_loading {
            return Task::none();
        }
        let Some(snapshot) = crate::session::TRANSCRIPT_REGISTRY.snapshot(key) else {
            return Task::none();
        };
        let live = session_view::build_ledger(&snapshot.history);
        // A workspace resolved after the first load re-parses the whole ledger.
        let root_changed = self.adopt_workspace_root(workspaces);
        let common = if root_changed {
            0
        } else {
            common_entry_prefix(&self.entries, &live)
        };
        if common == self.entries.len() && common == live.len() {
            return Task::none();
        }
        self.measure_cache
            .borrow_mut()
            .retain(|(i, _), _| *i < common);
        self.entries = live;
        // Live history positions, not stored rows: the merged ledger has no stored ids.
        self.entry_rows = None;
        self.entry_md.truncate(common);
        self.entry_md.extend(session_view::parse_entry_bodies(
            &self.entries[common..],
            self.workspace_root.as_deref(),
        ));

        // Live updates only auto-scroll when the user was already following.
        if self.auto_scroll_enabled {
            iced::widget::operation::snap_to_end(self.scrollable_id.clone())
        } else {
            Task::none()
        }
    }

    #[expect(clippy::too_many_lines)]
    pub(crate) fn view(
        &self,
        workspaces: &HashMap<String, Workspace>,
    ) -> Element<'_, SessionsMessage> {
        let mut content = column![];

        content = widgets::push_error_banner_inset(content, self.sessions.error());

        if !self.sessions.loaded() {
            content = content.push(widgets::scroll_h_inset(widgets::loading_text()));
        } else if let Some(cached) = self.cached_session_items.as_ref() {
            // Session list on the left side — built from cached display data.
            let mut session_list = Column::new().spacing(theme::SPACE_4);
            for item in cached {
                let is_selected = self.selected_session.as_deref() == Some(&item.agent_id);

                let mut title_row = iced::widget::Row::new()
                    .width(Length::Fill)
                    .spacing(theme::SPACE_6)
                    .align_y(Alignment::Center);
                if let Some(role) = item.role {
                    title_row = title_row.push(
                        theme::role_icon(&role)
                            .size(theme::TEXT_14)
                            .color(theme::role_badge_color_for(&role).0),
                    );
                }
                let title_color = if is_selected {
                    theme::ACCENT
                } else {
                    theme::TEXT_PRIMARY
                };
                title_row = title_row.push(
                    text(item.display_title(workspaces))
                        .size(theme::TEXT_13)
                        .color(title_color),
                );
                match item.activity {
                    SessionActivity::Working => {
                        title_row = title_row
                            .push(Space::new().width(Length::Fill))
                            .push(super::activity_dot());
                    }
                    // A frozen agent produces nothing: its row keeps its place.
                    SessionActivity::Paused => {
                        title_row = title_row
                            .push(Space::new().width(Length::Fill))
                            .push(text("paused").size(theme::TEXT_11).color(theme::TEXT_MUTED));
                    }
                    SessionActivity::Idle => {}
                }

                // A synthesised row (a run whose first message is not stored yet) has no stored
                // session.
                let count_label = item
                    .msg_count_label
                    .as_deref()
                    .or((item.activity == SessionActivity::Working).then_some("working"));

                let card = container(
                    column![
                        row![
                            button(
                                container(
                                    column![
                                        title_row,
                                        {
                                            // Meta row: messages left / tokens center / timestamp
                                            // right.
                                            let mut meta_row =
                                                iced::widget::Row::new().width(Length::Fill);
                                            if let Some(count) = count_label {
                                                meta_row = meta_row.push(
                                                    text(count)
                                                        .size(theme::TEXT_11)
                                                        .color(theme::TEXT_SECONDARY),
                                                );
                                            }
                                            meta_row =
                                                meta_row.push(Space::new().width(Length::Fill));
                                            if let Some(token) = &item.token_label {
                                                meta_row = meta_row.push(
                                                    text(token)
                                                        .size(theme::TEXT_11)
                                                        .color(theme::TEXT_SECONDARY),
                                                );
                                            }
                                            meta_row.push(Space::new().width(Length::Fill)).push(
                                                text(&item.timestamp_label)
                                                    .size(theme::TEXT_11)
                                                    .color(theme::TEXT_SECONDARY),
                                            )
                                        },
                                        text(&item.agent_id)
                                            .size(theme::TEXT_11)
                                            .color(theme::TEXT_MUTED),
                                    ]
                                    .width(Length::Fill)
                                    .spacing(theme::SPACE_2),
                                )
                                .padding(theme::PAD_6)
                                .width(Length::Fill),
                            )
                            .width(Length::Fill)
                            .style(theme::button_text)
                            .on_press(SessionsMessage::SelectSession(item.agent_id.clone())),
                        ]
                        .width(Length::Fill)
                        .align_y(Alignment::Center),
                    ]
                    .spacing(theme::SPACE_2),
                )
                .style(theme::surface_card_style);

                // A synthesised row has no stored session behind it, so there is nothing to act on.
                let sess_row: Element<'_, SessionsMessage> = if item.msg_count_label.is_some() {
                    ContextMenu::new(
                        card,
                        vec![MenuItem::with_icon(
                            iced_fonts::lucide::advanced_text::trash,
                            "Delete".into(),
                            SessionsMessage::DeleteSession(item.agent_id.clone()),
                        )],
                    )
                    .into()
                } else {
                    card.into()
                };

                session_list = session_list.push(sess_row);
            }

            let session_scroll =
                widgets::vscroll_sized(session_list, Length::Fixed(350.0), Length::Shrink);

            // Transcript on the right side, wrapped in `responsive` so the bubble width is
            // measured.
            let workspace_root = self.workspace_root.clone();
            let transcript: iced::Element<'_, SessionsMessage> = if self.selected_loading {
                iced::widget::container(widgets::loading_text())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(theme::PAD_16)
                    .into()
            } else if let Some(err) = self.selected_error.as_deref() {
                // The transcript read failed: render the failure where the transcript would be.
                container(widgets::error_banner_fill(err))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(theme::PAD_16)
                    .into()
            } else if self.selected_session.is_some() {
                // Truncate targets stored rows and is offered only on an idle session.
                let truncatable = self.entry_rows.is_some()
                    && self
                        .selected_session
                        .as_deref()
                        .is_some_and(|key| !crate::agent::registry::AGENT_REGISTRY.contains(key));
                let entries = &self.entries;
                let entry_md = &self.entry_md;
                let expanded = &self.expanded;
                let measure_cache = &self.measure_cache;
                let scrollable_id = self.scrollable_id.clone();
                responsive(move |size| {
                    // The transcript scrollable wrapper adds 8px padding per side, so the content
                    // width is
                    // narrower than the row.
                    let text_width = ((size.width - 16.0) * BUBBLE_BODY_RATIO - 24.0).max(0.0);
                    let ctx = TranscriptCtx {
                        entries,
                        entry_md,
                        workspace_root: workspace_root.clone(),
                        expanded,
                        measure_cache,
                        text_width,
                        truncatable,
                    };
                    container(render_transcript(&ctx, &scrollable_id))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .padding([theme::PAD_8, 0.0])
                        .into()
                })
                .into()
            } else {
                container(
                    text("Select a session to view transcript.")
                        .size(theme::TEXT_13)
                        .color(theme::TEXT_MUTED),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(theme::PAD_16)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
            };

            content = content.push(
                row![
                    session_scroll,
                    Space::new().width(theme::SPACE_12),
                    transcript
                ]
                .height(Length::Fill),
            );
        } else if self.sessions.error().is_none() {
            // A failed read must not masquerade as a legitimately empty list.
            content = content.push(widgets::empty_state_placeholder(
                lucide::layout_dashboard::<iced::Theme, iced::Renderer>(),
                "No sessions",
                theme::TEXT_MUTED,
            ));
        }

        // The page-level dialog slot is stacked above the padded page, as on every other page.
        widgets::page_with_dialog(content, widgets::no_dialog())
    }
}

/// Length of the longest common prefix of two ledgers (first differing entry index).
fn common_entry_prefix(old: &[SessionEntry], new: &[SessionEntry]) -> usize {
    old.iter().zip(new).take_while(|(a, b)| a == b).count()
}

/// Author label above a bubble for the roles that carry one, flush left like the bubble.
fn author_label<'a>(role: crate::ChatRole) -> Option<Element<'a, SessionsMessage>> {
    let name = match role {
        crate::ChatRole::System => "System",
        crate::ChatRole::User => "User",
        crate::ChatRole::Tool => "Tool",
        crate::ChatRole::Assistant => return None,
    };
    let label = text(name).size(theme::TEXT_11).color(theme::TEXT_SECONDARY);
    Some(align_bubble(label, BubbleSide::Left))
}

/// Chat-bubble row for one ledger entry: the roles that carry an author label get it, the bubble
/// is aligned by side.
fn bubble_row(
    role: crate::ChatRole,
    body: Column<'_, SessionsMessage>,
    menu: Option<MenuItem<SessionsMessage>>,
) -> Element<'_, SessionsMessage> {
    let assistant = matches!(role, crate::ChatRole::Assistant);
    let side = if assistant {
        BubbleSide::Right
    } else {
        BubbleSide::Left
    };
    let bubble: Element<'_, SessionsMessage> = container(body)
        .padding(theme::PAD_10)
        .style(theme::bubble_style(
            if assistant {
                theme::BG_ELEVATED
            } else {
                theme::BG_SURFACE
            },
            Some(theme::TEXT_PRIMARY),
        ))
        .width(Length::FillPortion(3))
        .into();
    let bubble: Element<'_, SessionsMessage> = match menu {
        Some(item) => ContextMenu::new(bubble, vec![item]).into(),
        None => bubble,
    };
    let mut row = Column::new().spacing(theme::SPACE_2);
    if let Some(label) = author_label(role) {
        row = row.push(label);
    }
    row.push(align_bubble(bubble, side)).into()
}

/// Resolve the collapse measurement for an element at the current `text_width`, measuring on
/// cache miss (cached per element and width). Returns the wrapped-line count and — only when
/// `need_preview` is set (the element is currently collapsed) — its 3-line plain-text preview, so
/// expanded elements do not pay a per-frame preview clone. `font_size` is the body size the element
/// renders at; `body` is how the text is drawn, so the measurement matches it.
///
/// Stale-fingerprint semantics: a cache hit with the same content length and `font_size` is reused
/// when its width bucket still matches, re-measured via [`re_measure`] when only the bucket moved,
/// and otherwise replaced by a fresh measurement.
#[expect(
    clippy::float_cmp,
    reason = "font sizes are theme constants passed verbatim; exact identity is intended"
)]
fn element_measurement(
    measure_cache: &RefCell<HashMap<(usize, usize), MessageMeasure>>,
    key: (usize, usize),
    text_width: f32,
    need_preview: bool,
    content: &str,
    font_size: f32,
    body: BodyText<'_>,
) -> (u32, Option<String>) {
    let bucket = width_bucket(text_width);
    let mut cache = measure_cache.borrow_mut();
    let cache_hit = cache.get(&key);
    // The same body at the same size: only a width change can be served from the cache.
    let unchanged =
        cache_hit.filter(|m| m.source.len() == content.len() && m.font_size == font_size);
    if unchanged.is_none_or(|m| m.width_bucket != bucket) {
        let m = match unchanged {
            Some(m) => re_measure(m, text_width),
            None => measure_message(content, body, text_width, font_size),
        };
        cache.insert(key, m);
    }
    let m = cache.get(&key).expect("measurement resolved above");
    (
        m.wrapped_lines,
        if need_preview {
            m.preview.clone()
        } else {
            None
        },
    )
}

/// The collapsed-preview-versus-full content of a collapsible element: when it is collapsed, the
/// preview replaces the body.
fn collapsible_text_block(
    collapses: bool,
    is_expanded: bool,
    preview: Option<String>,
    color: iced::Color,
    size: f32,
    font: iced::Font,
    expanded: Element<'_, SessionsMessage>,
) -> Element<'_, SessionsMessage> {
    if collapses && !is_expanded {
        let preview = preview.expect("a collapsing element always has a preview");
        column![
            selectable_text(preview, color).size(size).font(font),
            lucide::ellipsis::<iced::Theme, iced::Renderer>()
                .size(theme::TEXT_11)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(theme::SPACE_2)
        .into()
    } else {
        expanded
    }
}

/// Wrap an element in the click-to-toggle area when it can collapse (a short
/// click anywhere toggles, drag selection keeps working).
fn click_toggle(
    key: (usize, usize),
    active: bool,
    el: Element<'_, SessionsMessage>,
) -> Element<'_, SessionsMessage> {
    if active {
        widgets::click_to_toggle(
            el,
            SessionsMessage::ToggleExpand(key.0, key.1),
            SessionsMessage::ToggleCancel(key.0, key.1),
        )
    } else {
        el
    }
}

/// Shared collapse-state preamble for the transcript's collapsible elements:
/// returns `(is_expanded, collapses, preview)` — whether the user expanded the
/// element, whether it renders more than [`MAX_PREVIEW_LINES`] wrapped lines
/// at `font_size` (measured at the bubble body width, the way `body` draws it),
/// and the measured 3-line preview if it can collapse.
fn collapse_state(
    ctx: &TranscriptCtx<'_>,
    key: (usize, usize),
    content: &str,
    font_size: f32,
    body: BodyText<'_>,
) -> (bool, bool, Option<String>) {
    let is_expanded = ctx.expanded.contains(&key);
    let (wrapped_lines, preview) = element_measurement(
        ctx.measure_cache,
        key,
        ctx.text_width,
        !is_expanded,
        content,
        font_size,
        body,
    );
    (is_expanded, wrapped_lines > MAX_PREVIEW_LINES, preview)
}

/// A plain-text collapsible element with the 3-line measured collapse rule:
/// when it renders more than [`MAX_PREVIEW_LINES`] wrapped lines and is not
/// expanded, a 3-line preview (with an "ellipsis" marker) is shown; otherwise
/// the full selectable text. The whole element is click-to-toggle when it can
/// collapse, so the selectable text stays drag-selectable. The measurement
/// stays at [`theme::MARKDOWN_TEXT_SIZE`] (the text renders at 11px) — the
/// err-toward-collapsing bias.
fn plain_collapsible<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
    color: iced::Color,
) -> Element<'a, SessionsMessage> {
    let (is_expanded, collapses, preview) = collapse_state(
        ctx,
        key,
        content,
        theme::MARKDOWN_TEXT_SIZE,
        BodyText::Plain,
    );
    let el = collapsible_text_block(
        collapses,
        is_expanded,
        preview,
        color,
        theme::TEXT_11,
        theme::FONT_REGULAR,
        selectable_text(content, color).size(theme::TEXT_11).into(),
    );
    click_toggle(key, collapses, el)
}

/// The collapsible Thinking block: lucide `brain` + "Thinking" header above a collapsible body.
fn thinking_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
) -> Element<'a, SessionsMessage> {
    let (is_expanded, collapses, preview) = collapse_state(
        ctx,
        key,
        content,
        theme::MARKDOWN_TEXT_SIZE,
        BodyText::Plain,
    );
    let header = row![
        lucide::brain::<iced::Theme, iced::Renderer>()
            .size(theme::TEXT_11)
            .color(theme::TEXT_MUTED),
        text("Thinking")
            .size(theme::TEXT_11)
            .color(theme::TEXT_MUTED),
    ]
    .spacing(theme::SPACE_4)
    .align_y(Alignment::Center);
    let body = collapsible_text_block(
        collapses,
        is_expanded,
        preview,
        theme::TEXT_MUTED,
        theme::TEXT_11,
        theme::FONT_REGULAR,
        selectable_text(content, theme::TEXT_MUTED)
            .size(theme::TEXT_11)
            .into(),
    );
    let whole: Element<'a, SessionsMessage> = column![header, body].spacing(theme::SPACE_4).into();
    click_toggle(key, collapses, whole)
}

/// The message/narration body with the 3-line collapse rule: a collapsed body draws its
/// plain-text preview, an expanded one the parsed markdown.
fn body_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
    md: Option<&'a ParsedBody>,
    is_narration: bool,
) -> Element<'a, SessionsMessage> {
    let size = if is_narration {
        theme::NARRATION_TEXT_SIZE
    } else {
        theme::MARKDOWN_TEXT_SIZE
    };
    // A body with no parsed form — promoted reasoning, an unparsed message — drawn as plain text.
    let body = if md.is_some() {
        BodyText::Markdown(ctx.workspace_root.as_deref())
    } else {
        BodyText::Plain
    };
    let (is_expanded, collapses, preview) = collapse_state(ctx, key, content, size, body);
    let font = if is_narration {
        theme::FONT_ITALIC
    } else {
        theme::FONT_REGULAR
    };
    let plain = || {
        selectable_text(content, theme::TEXT_PRIMARY)
            .size(size)
            .font(font)
            .into()
    };
    let expanded: Element<'a, SessionsMessage> = if let Some(parsed) = md {
        let settings = if is_narration {
            theme::narration_markdown_settings()
        } else {
            theme::markdown_settings()
        };
        let workspace = ctx.workspace_root.clone();
        let link_message = move |url: String| match media_markers::file_link_target(&url) {
            Some(target) => SessionsMessage::RevealFile {
                target: target.to_string(),
                workspace: workspace.clone(),
            },
            None => SessionsMessage::LinkClicked(url),
        };
        media_markers::body(
            &parsed.attachments,
            &parsed.items,
            settings,
            link_message,
            plain,
        )
    } else {
        plain()
    };
    let el = collapsible_text_block(
        collapses,
        is_expanded,
        preview,
        theme::TEXT_PRIMARY,
        size,
        font,
        expanded,
    );
    click_toggle(key, collapses, el)
}

/// The tool-call result block: plain collapsible text at [`theme::TEXT_SECONDARY`].
fn result_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    result: &'a str,
) -> Element<'a, SessionsMessage> {
    plain_collapsible(ctx, key, result, theme::TEXT_SECONDARY)
}

/// Render one session round as a single assistant bubble containing the narration, the reasoning
/// and the tool blocks.
fn render_tool_round<'a>(
    ctx: &TranscriptCtx<'a>,
    i: usize,
    narration: Option<&'a str>,
    reasoning: Option<&'a str>,
    calls: &'a [ToolCallEntry],
) -> Element<'a, SessionsMessage> {
    let md = ctx.entry_md.get(i).and_then(|m| m.as_ref());
    let mut bubble_col = Column::new().spacing(theme::SPACE_4);

    if let Some(reasoning) = session_view::promoted_reasoning(narration, reasoning) {
        // The reasoning IS the narration for this round: render it in the narration's place.
        bubble_col = bubble_col.push(body_block(ctx, (i, 0), reasoning, None, true));
    } else {
        if let Some(reasoning) = reasoning {
            bubble_col = bubble_col.push(thinking_block(ctx, (i, 1), reasoning));
        }
        if let Some(narration) = narration {
            bubble_col = bubble_col.push(body_block(ctx, (i, 0), narration, md, true));
        }
    }

    for (j, call) in calls.iter().enumerate() {
        let key = (i, 2 + j);
        // Only a result that actually collapses makes the tool block a click target.
        let collapses = call.result.as_deref().is_some_and(|result| {
            let (wrapped_lines, _) = element_measurement(
                ctx.measure_cache,
                key,
                ctx.text_width,
                false,
                result,
                theme::MARKDOWN_TEXT_SIZE,
                BodyText::Plain,
            );
            wrapped_lines > MAX_PREVIEW_LINES
        });
        // The tool block is wrapped in a click-to-toggle (not a button) so the whole block is the
        // target.
        let tool_block = container(session_view::tool_block(
            &call.tool,
            session_view::ToolBlockView::Full,
        ))
        .width(Length::Fill)
        .padding([theme::PAD_2, theme::PAD_4]);
        if collapses {
            bubble_col = bubble_col.push(click_toggle(key, true, tool_block.into()));
        } else {
            bubble_col = bubble_col.push(tool_block);
        }

        match call.result.as_deref() {
            Some(result) if !result.is_empty() => {
                bubble_col = bubble_col.push(result_block(ctx, key, result));
            }
            // Empty or missing ToolResult record: the explicit placeholder.
            _ => {
                bubble_col = bubble_col.push(
                    row![
                        lucide::arrow_down_to_line::<iced::Theme, iced::Renderer>()
                            .size(theme::TEXT_11)
                            .color(theme::TEXT_MUTED),
                        text("(no result)")
                            .size(theme::TEXT_10)
                            .color(theme::TEXT_SECONDARY),
                    ]
                    .spacing(theme::SPACE_4)
                    .align_y(Alignment::Center),
                );
            }
        }
    }

    // One bubble for the whole round, same container as assistant text rounds.
    bubble_row(crate::ChatRole::Assistant, bubble_col, None)
}

/// Render one ledger entry as a chat-bubble row: a `Message` renders its own bubble, a tool round
/// its assistant bubble, a narration its divider.
fn render_entry<'a>(ctx: &TranscriptCtx<'a>, i: usize) -> Option<Element<'a, SessionsMessage>> {
    let entry = &ctx.entries[i];
    match entry {
        SessionEntry::Message {
            role,
            content,
            thinking,
        } => {
            if thinking.is_none() && content.is_none() {
                // Nothing to show but the author: the row is the author label.
                return author_label(*role);
            }
            let mut bubble_col = Column::new().spacing(theme::SPACE_4);
            if let Some(thinking) = thinking {
                bubble_col = bubble_col.push(thinking_block(ctx, (i, 1), thinking));
            }
            if let Some(content) = content {
                let md = ctx.entry_md.get(i).and_then(Option::as_ref);
                bubble_col = bubble_col.push(body_block(ctx, (i, 0), content, md, false));
            }
            let menu = (ctx.truncatable && matches!(role, crate::ChatRole::User))
                .then(|| MenuItem::new("Truncate".into(), SessionsMessage::TruncateSession(i)));
            Some(bubble_row(*role, bubble_col, menu))
        }
        SessionEntry::ToolRound {
            narration,
            reasoning,
            calls,
        } => Some(render_tool_round(
            ctx,
            i,
            narration.as_deref(),
            reasoning.as_deref(),
            calls,
        )),
    }
}

/// Render the transcript column for the selected session: build the chat column, or the
/// placeholder when nothing is selected.
fn render_transcript<'a>(
    ctx: &TranscriptCtx<'a>,
    scrollable_id: &Id,
) -> Element<'a, SessionsMessage> {
    if ctx.entries.is_empty() {
        return text("No messages in this session.")
            .size(theme::TEXT_13)
            .color(theme::TEXT_MUTED)
            .into();
    }
    let mut items = Column::new().spacing(widgets::CHAT_VERTICAL_RHYTHM);
    for i in 0..ctx.entries.len() {
        if let Some(entry) = render_entry(ctx, i) {
            items = items.push(entry);
        }
    }
    widgets::vscroll_tracked(
        items,
        Length::Fill,
        Length::Fill,
        scrollable_id.clone(),
        SessionsMessage::ScrollChanged,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatRole;
    use crate::agent::registry::RunningTool;
    use SessionActivity::{Paused, Working};

    fn msg(role: ChatRole, content: Option<&str>) -> SessionEntry {
        SessionEntry::Message {
            role,
            content: content.map(str::to_string),
            thinking: None,
        }
    }

    fn tool_round() -> SessionEntry {
        SessionEntry::ToolRound {
            narration: Some("checking the file".to_string()),
            reasoning: None,
            calls: vec![ToolCallEntry {
                tool_call_id: "call_0".to_string(),
                tool: RunningTool {
                    name: "read".to_string(),
                    args: vec![("path".to_string(), "a.rs".to_string())],
                },
                result: Some("ok".to_string()),
            }],
        }
    }

    fn metadata(agent_id: &str, last_activity: chrono::DateTime<chrono::Utc>) -> SessionMetadata {
        SessionMetadata {
            agent_id: agent_id.to_string(),
            last_activity,
            message_count: 1,
            token_length: None,
            role: Some("engineer".to_string()),
            user_name: None,
            workspace_name: Some("ws".to_string()),
            custom_role: None,
            custom_identifier: None,
            launch_stopped: false,
        }
    }

    fn agent(
        agent_id: &str,
        role: &str,
        workspace: &str,
        started_at: chrono::DateTime<chrono::Utc>,
    ) -> crate::agent::registry::AgentHandle {
        crate::agent::registry::AgentHandle {
            agent_id: agent_id.to_string(),
            role: role.to_string(),
            ticket_id: None,
            workspace_path: format!("/ws/{workspace}"),
            workspace_name: workspace.to_string(),
            parent_key: None,
            parent_label: None,
            started_at,
            generation: 0,
            activity: None,
            paused: false,
        }
    }

    #[test]
    fn a_bodyless_assistant_row_is_skipped_but_labelled_roles_keep_it() {
        let expanded = HashSet::new();
        let cache = RefCell::new(HashMap::new());
        let entries = [
            msg(ChatRole::User, None),
            msg(ChatRole::System, None),
            msg(ChatRole::Tool, None),
            msg(ChatRole::Assistant, None),
            msg(ChatRole::Assistant, Some("done")),
        ];
        let ctx = TranscriptCtx {
            entries: &entries,
            entry_md: &[],
            workspace_root: None,
            expanded: &expanded,
            measure_cache: &cache,
            text_width: 300.0,
            truncatable: false,
        };

        let rendered: Vec<bool> = (0..entries.len())
            .map(|i| render_entry(&ctx, i).is_some())
            .collect();
        // Only the bodyless assistant row is skipped: the labelled roles keep theirs.
        assert_eq!(rendered, [true, true, true, false, true]);
    }

    const STORED: &[(&str, i64)] = &[("a", 10), ("b", 5), ("c", 0)];
    const PAUSED_STORED: &[(&str, i64)] = &[("a", 10), ("b", 0)];
    const RUNNING: &[(&str, &str, i64, bool)] =
        &[("a", "engineer", 1, false), ("zz", "analyst", 0, false)];
    const PAUSED_RUNNING: &[(&str, &str, i64, bool)] = &[("a", "engineer", 0, true)];

    type ProjRow<'a> = (&'a str, i64);
    type RunRow<'a> = (&'a str, &'a str, i64, bool);
    type OrderCase<'a> = (&'a str, &'a [ProjRow<'a>], &'a [RunRow<'a>], &'a [&'a str]);
    #[test]
    fn test_session_row_ordering() {
        let cases: &[OrderCase] = &[
            ("running", STORED, RUNNING, &["a", "zz", "c", "b"]),
            ("paused", PAUSED_STORED, PAUSED_RUNNING, &["b", "a"]),
        ];

        for &(name, sessions, running, order) in cases {
            let now = chrono::Utc::now();
            let started = |minutes: i64| now - chrono::Duration::minutes(minutes);
            let stored: Vec<_> = sessions
                .iter()
                .map(|&(id, m)| metadata(id, started(m)))
                .collect();
            let agents: Vec<_> = running
                .iter()
                .map(|&(id, role, minutes, paused)| {
                    let mut handle = agent(id, role, "ws", started(minutes));
                    handle.paused = paused;
                    handle
                })
                .collect();
            let rows = session_rows(&stored, &agents, now.into());
            let ids: Vec<&str> = rows.iter().map(|r| r.agent_id.as_str()).collect();
            assert_eq!(ids, order, "case {name}: order");

            for &(id, role, _, paused) in running {
                let row = rows
                    .iter()
                    .find(|r| r.agent_id == id)
                    .expect("a running row");
                let listed = if paused { Paused } else { Working };
                assert_eq!(row.activity, listed, "case {name}: {id} activity");
                if sessions.iter().any(|&(stored, _)| stored == id) {
                    continue;
                }
                assert_eq!(row.msg_count_label, None, "case {name}: no count");
                assert_eq!(row.title, "ws", "case {name}: titled by project");
                assert_eq!(
                    row.role,
                    role.parse::<crate::Role>().ok(),
                    "case {name}: role"
                );
            }
        }
    }

    #[test]
    fn a_workspace_named_after_the_load_reparses_the_ledger() {
        let tmp = tempfile::tempdir().unwrap();
        let photo = tmp.path().join("generated/poster.png");
        std::fs::create_dir_all(photo.parent().unwrap()).unwrap();
        std::fs::write(&photo, b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").unwrap();

        // Loaded before the workspace could be named: the marker is a reference, not a path.
        let mut state = SessionsState::new();
        state.selected_session = Some("s1".to_string());
        state.entries = vec![msg(
            ChatRole::Assistant,
            Some("[IMAGE:generated/poster.png]"),
        )];
        state.entry_md = session_view::parse_entry_bodies(&state.entries, None);
        assert_eq!(state.workspace_root, None);
        assert_eq!(
            state.entry_md[0].as_ref().unwrap().attachments,
            vec![media_markers::MediaAttachment {
                image: Some("generated/poster.png".to_string()),
                label: "🖼️ poster.png".to_string(),
                reveal: None,
            }],
            "an unresolved relative target still draws its row"
        );
        assert!(
            state.entry_md[0]
                .as_ref()
                .unwrap()
                .items
                .iter()
                .all(|item| !matches!(item, iced::widget::markdown::Item::Image { .. }))
        );

        // The workspace list arrives later: the already-loaded ledger is re-parsed.
        let workspaces = HashMap::from([(
            "ws".to_string(),
            Workspace {
                name: "ws".to_string(),
                path: tmp.path().to_string_lossy().into_owned(),
                ..Workspace::default()
            },
        )]);
        state
            .sessions
            .settle(Ok(vec![metadata("s1", chrono::Utc::now())]));
        let sessions = state.sessions.entries().to_vec();
        let _ = state.update(SessionsMessage::Settled(Ok(sessions)), &workspaces);

        assert_eq!(state.workspace_root.as_deref(), Some(tmp.path()));
        let parsed = state.entry_md[0].as_ref().unwrap();
        let expected = photo.to_string_lossy().into_owned();
        assert_eq!(
            parsed.attachments,
            vec![media_markers::MediaAttachment {
                image: Some(expected),
                label: "🖼️ poster.png".to_string(),
                reveal: None,
            }],
            "the reassembled body must carry the picture as an attachment"
        );
        assert!(
            parsed
                .items
                .iter()
                .all(|item| !matches!(item, iced::widget::markdown::Item::Image { .. })),
            "no markdown image may reach the text: {:?}",
            parsed.items
        );
    }

    #[test]
    fn test_common_entry_prefix() {
        use ChatRole::{Assistant, User};
        use Ledger::{Msg, Tool};

        enum Ledger {
            Msg(ChatRole, Option<&'static str>),
            Tool,
        }

        const HI: &[Ledger] = &[Msg(User, Some("hi")), Tool];
        const HI_YO: &[Ledger] = &[Msg(User, Some("hi")), Msg(Assistant, Some("yo"))];
        const A_B: &[Ledger] = &[Msg(User, Some("a")), Msg(Assistant, Some("b"))];
        const A_C: &[Ledger] = &[Msg(User, Some("a")), Msg(Assistant, Some("c"))];
        let cases: &[(&str, &[Ledger], &[Ledger], usize)] = &[
            ("empty", &[], &[], 0),
            ("same", HI, HI, 2),
            ("appended", &[Msg(User, Some("hi"))], HI_YO, 1),
            ("edited", A_B, A_C, 1),
            ("longer old", A_B, &[Msg(User, Some("a"))], 1),
        ];

        for &(name, old, new, shared) in cases {
            let ledger = |entries: &[Ledger]| -> Vec<SessionEntry> {
                entries
                    .iter()
                    .map(|entry| match *entry {
                        Msg(role, body) => msg(role, body),
                        Tool => tool_round(),
                    })
                    .collect()
            };
            let prefix = common_entry_prefix(&ledger(old), &ledger(new));
            assert_eq!(prefix, shared, "case {name}: shared entries");
        }
    }

    #[test]
    fn test_transcript_load_errors() {
        #[derive(Clone, Copy)]
        enum Load {
            Failed(&'static str),
            Loaded(&'static str),
        }
        use Load::{Failed, Loaded};
        type LoadCase<'a> = (&'a str, Option<&'a str>, Load, Option<&'a str>, bool);
        let cases: &[LoadCase] = &[
            ("stays", None, Failed("s1"), Some("read failed"), false),
            ("left", None, Failed("s2"), None, true),
            ("clears", Some("old"), Loaded("s1"), None, false),
        ];

        for &(name, error, msg, expected, loading) in cases {
            let mut state = SessionsState::new();
            state.selected_session = Some("s1".to_string());
            state.selected_loading = true;
            state.selected_error = error.map(str::to_string);
            let message = match msg {
                Failed(key) => SessionsMessage::SessionError {
                    key: key.to_string(),
                    error: "read failed".to_string(),
                },
                Loaded(key) => SessionsMessage::SessionMessages(key.to_string(), vec![]),
            };
            let _ = state.update(message, &HashMap::new());
            let error = state.selected_error.as_deref();
            assert_eq!(error, expected, "case {name}: error");
            assert_eq!(state.selected_loading, loading, "case {name}: loading");
            assert!(state.sessions.error().is_none(), "case {name}: list banner");
        }
    }

    fn cached_item(
        role: crate::Role,
        user: Option<&str>,
        workspace: Option<&str>,
    ) -> CachedSessionItem {
        CachedSessionItem {
            agent_id: "agent".to_string(),
            role: Some(role),
            title: user.or(workspace).unwrap_or_default().to_string(),
            workspace_name: workspace.map(str::to_string),
            custom_role: None,
            custom_identifier: None,
            msg_count_label: None,
            token_label: None,
            timestamp_label: String::new(),
            activity: SessionActivity::Idle,
        }
    }

    fn workspace_map(names: &[&str]) -> HashMap<String, Workspace> {
        names
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    Workspace {
                        name: (*name).to_string(),
                        ..Workspace::default()
                    },
                )
            })
            .collect()
    }

    #[test]
    fn test_workspace_root_and_card_titles() {
        use crate::Role;
        let admin = crate::users::ADMIN_USER_NAME;
        let personal = format!("personal:{admin}");
        let mut map = workspace_map(&["ws"]);
        map.get_mut("ws").unwrap().path = "/tmp/ws".to_string();
        let roots = [
            (
                personal.as_str(),
                Some(crate::users::personal_workspace_path(admin)),
            ),
            ("ws", Some(PathBuf::from("/tmp/ws"))),
            ("/tmp/ws/stored", Some(PathBuf::from("/tmp/ws/stored"))),
            ("gone", None),
        ];
        for (stored, root) in roots {
            let resolved = workspace_root_for(stored, &map);
            assert_eq!(resolved, root, "case {stored}: stored name shape");
        }

        let titles = [
            (Role::Manager, Some("admin"), Some("ws"), Some("ws")),
            (Role::Manager, None, Some("ws"), Some("ws")),
            (Role::Manager, Some("admin"), Some("gone"), None),
            (Role::Manager, Some("admin"), None, None),
            (Role::Maintainer, Some("admin"), Some("ws"), Some("ws")),
            (Role::Maintainer, None, Some("ws"), Some("ws")),
            (Role::Maintainer, Some("admin"), Some("gone"), None),
            (Role::Maintainer, Some("admin"), None, None),
            (Role::Assistant, Some("admin"), Some("ws"), Some("admin")),
            (Role::Engineer, Some("alice"), Some("ws"), Some("alice")),
        ];
        for (role, user, workspace, expected) in titles {
            let item = cached_item(role, user, workspace);
            let label = role.display_label();
            let title = expected.map_or_else(|| label.to_string(), str::to_string);
            let shown = item.display_title(&map);
            assert_eq!(shown, title, "case {role:?}: a card title");
        }
    }
}
