//! Sessions dashboard page — view and manage conversation sessions.
//!
//! The list is a live projection of the registry, not a snapshot: a session
//! whose agent is running is pinned to the top of the list for as long as it
//! runs — including an agent that has not stored its first message yet — and the
//! rows are re-read on every runtime tick rather than only on page entry, so the
//! page converges on its own. See [`session_rows`].

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
    MAX_PREVIEW_LINES, MessageMeasure, measure_message, re_measure, width_bucket,
};
use super::session_view::{self, ParsedBody, SessionEntry, ToolCallEntry};
use super::theme;
use super::widgets::{self, BubbleSide, align_bubble, selectable_text};

/// Fraction of the transcript row width a chat bubble occupies (FillPortion 3
/// of a 3:1 row). The collapse measurement uses this to narrow the measured
/// width to the actual bubble body (container padding + a safety margin).
const BUBBLE_BODY_RATIO: f32 = 0.75;

/// Delay before a click-scheduled collapse of an expanded element actually
/// applies: the widget's double-click window (300 ms) plus a small margin, so
/// a canceling second click published at the window boundary always arrives
/// before the commit. A click inside the margin (beyond the double-click
/// window) is a deliberate second click and re-affirms the still-scheduled
/// collapse.
const COLLAPSE_COMMIT_DELAY: Duration = Duration::from_millis(350);

/// Max characters for the card title (a user/workspace name) before an ellipsis
/// is appended — the card column is 350px wide. A custom-role row carries its
/// own compact spelling under the same bound (see [`custom_role_title`]).
const CARD_TITLE_MAX_CHARS: usize = 32;

/// Max characters of the branch identifier inside a custom-role card title — a
/// branch is the discriminating half of the label, so it keeps half the budget.
const CARD_TITLE_BRANCH_MAX_CHARS: usize = CARD_TITLE_MAX_CHARS / 2;

/// Shared transcript render context: the flat session ledger, its per-entry
/// parsed bodies, the element expansion set, the collapse-measurement
/// cache, the single text measurement width (bubble body width, used for
/// all collapsible-element measurement), and whether the ledger can be
/// truncated (a stored, idle session).
struct TranscriptCtx<'a> {
    entries: &'a [SessionEntry],
    entry_md: &'a [Option<ParsedBody>],
    /// Root folder of the conversation's workspace; a relative file marker
    /// resolves against it. Owned because the `responsive` closure that builds
    /// the transcript is a `move` closure — a `Fn` closure cannot lend a
    /// captured value to the element it returns.
    workspace_root: Option<PathBuf>,
    expanded: &'a HashSet<(usize, usize)>,
    measure_cache: &'a RefCell<HashMap<(usize, usize), MessageMeasure>>,
    text_width: f32,
    truncatable: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum SessionsMessage {
    /// A finished list read — page entry and the runtime tick alike.
    Settled(Result<Vec<SessionMetadata>, String>),
    SelectSession(String),
    SessionMessages(String, Vec<ChatMessage>),
    /// A transcript read failure. `key` is the session the failed read
    /// belonged to, so a slow read for a session the user has since left does
    /// not render in the newly selected transcript.
    SessionError {
        key: String,
        error: String,
    },
    /// Toggle an element's collapsed preview / full content, keyed by
    /// (entry index, element index within entry). Expanding applies
    /// immediately; collapsing is deferred past the double-click window (see
    /// `SessionsState::update`) and confirmed by [`SessionsMessage::ToggleCommit`].
    ToggleExpand(usize, usize),

    /// The second click of a double-click on a collapsible element: a
    /// word/line selection, so any collapse scheduled by the first click is
    /// dropped.
    ToggleCancel(usize, usize),

    /// Apply a collapse previously scheduled by [`SessionsMessage::ToggleExpand`]
    /// once the double-click window has passed without a canceling second
    /// click.
    ToggleCommit(usize, usize),

    /// Scroll position changed in the transcript viewport.
    ScrollChanged(scrollable::Viewport),

    /// Dismiss modals/panels (Escape key).
    Escape,

    /// A link was clicked in rendered markdown.
    LinkClicked(String),

    /// A `[FILE:…]` marker was clicked: reveal `target` in the file manager,
    /// resolving a relative target against `workspace`.
    RevealFile {
        target: String,
        workspace: Option<PathBuf>,
    },

    /// A [`SessionsMessage::RevealFile`] finished: an error becomes an error
    /// toast, success needs no action.
    FileRevealed(Result<(), String>),

    /// A toast notification to surface from the dashboard.
    Toast(ToastMessage),
    /// Delete a session (context-menu action).
    DeleteSession(String),
    /// A session delete finished — remove it from the list. The bool is the
    /// store-reported row deletion (`true` when a real removal happened).
    SessionDeleted(String, bool),

    /// Truncate the selected session (context-menu action): remove the ledger
    /// entry at this index and every stored message from it on.
    TruncateSession(usize),
    /// A session truncate finished: the store removed `deleted` rows from
    /// `entry` on (`0` = the cut was a no-op).
    SessionTruncated {
        key: String,
        entry: usize,
        deleted: usize,
    },

    /// Coalesced runtime-change tick (250 ms): re-read the list so rows that
    /// appear, grow or finish converge, refresh the cached rows and their
    /// relative-time labels, and merge the selected session's live transcript
    /// when its agent is running.
    RuntimeChanged,
}

#[derive(Debug, Clone)]
struct CachedSessionItem {
    agent_id: String,
    /// Agent role parsed from the metadata role string (`None` renders the
    /// title without an icon); for a running agent without a stored row, the
    /// role the registry reports.
    role: Option<crate::Role>,
    /// User name when present (Some and non-empty), otherwise the workspace
    /// name (may be empty if both are absent). Workspace-scoped roles ignore
    /// this and are titled by [`Self::workspace_name`] — see
    /// [`Self::display_title`].
    title: String,
    /// The workspace the agent belongs to, as recorded in the session context.
    /// `None` when the session has no workspace (or an empty one).
    workspace_name: Option<String>,
    /// The custom role this session belongs to, when it is a custom-role
    /// session — its name titles the row instead of `title`. `None` for a worker
    /// branch.
    custom_role: Option<String>,
    /// The named branch this continuable session continues, when the branch was
    /// named — a custom role's identifier, or a worker branch's name.
    custom_identifier: Option<String>,
    /// Pre-formatted message count, `None` only for a running agent whose first
    /// message is not stored yet — its row is synthesised from the registry, so
    /// there is no count and no session to delete.
    msg_count_label: Option<String>,
    /// Pre-formatted compact token count (same format as the Running Agents
    /// card), when the session ever recorded a provider-reported length.
    token_label: Option<String>,
    /// Pre-formatted timestamp string.
    timestamp_label: String,
    /// What the session's agent is doing right now.
    activity: SessionActivity,
}

/// What a session's agent is doing right now — the whole difference between a
/// row pinned on top of the list and a row in its last-activity place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionActivity {
    /// No agent holds the session.
    Idle,
    /// A registered agent is working in it: the row is pinned to the top of the
    /// list and carries the spinner.
    Working,
    /// A registered agent was frozen by a workspace pause. It stops at its next
    /// round boundary, so it is not working: the row keeps its place and shows
    /// the state instead of the spinner.
    Paused,
}

impl CachedSessionItem {
    /// The row's title as the list shows it. A custom-role session is titled by
    /// the role it belongs to — and its branch, when the branch was named —
    /// instead of the user/workspace name every other role shows; a worker branch
    /// is titled `Worker · <branch>` so it is distinguishable from a single worker
    /// run.
    ///
    /// A workspace's manager and maintainer belong to exactly one managed
    /// workspace, so their rows are titled by its managed name (`mahbot_rlx`,
    /// never the folder basename and never the account name). When that
    /// workspace is no longer a managed workspace the row shows the role
    /// alone rather than a stale name.
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

/// The compact ` · <branch>` suffix a branch row carries, under the same
/// [`CARD_TITLE_MAX_CHARS`] bound every other row's title has.
fn branch_suffix(identifier: &str) -> String {
    format!(
        " · {}",
        crate::util::truncate(identifier, CARD_TITLE_BRANCH_MAX_CHARS)
    )
}

/// Card title for a custom-role row: the compact `name · branch` spelling under
/// the same [`CARD_TITLE_MAX_CHARS`] bound every other row's title has. A named
/// branch keeps its identifier: the role's own name is truncated (a role name
/// may itself be [`crate::custom_role::MAX_ROLE_NAME_CHARS`] long) so the
/// suffix always fits. The full [`crate::launch::display_label`] wording is
/// used where the surface has room for it — the Running Agents card.
fn custom_role_title(name: &str, identifier: Option<&str>) -> String {
    let Some(identifier) = identifier else {
        return crate::util::truncate(name, CARD_TITLE_MAX_CHARS);
    };
    // The identifier is capped at half the budget, so the rest of the budget is
    // always enough for the separator and a usable share of the name.
    let suffix = branch_suffix(identifier);
    let room = CARD_TITLE_MAX_CHARS - suffix.chars().count();
    format!("{}{suffix}", crate::util::truncate(name, room))
}

/// The session list's rows, in display order.
///
/// A session whose agent is working is pinned to the top for as long as it runs
/// (ordered by run start, so the block stays still while the runs go on) and
/// settles into its last-activity place once it stops: a run in progress must
/// never be lost below the fold of a long list, and the list is not re-sorted
/// between a run's start and its end. A paused (frozen) agent is not working, so
/// its row keeps its normal place.
///
/// A registered agent whose first message is not stored yet has no row of its
/// own and gets a synthesised one from what the registry knows — its role, its
/// project and that it is working. Rows are keyed by `agent_id`, so the
/// synthesised row is replaced by the stored one as soon as the first message
/// lands.
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
    // What remains in `live` has no stored row yet.
    for handle in live.into_values() {
        let activity = session_activity(Some(handle));
        let workspace_name = Some(handle.workspace_name.clone()).filter(|n| !n.is_empty());
        // A personal workspace is titled by its user, like a stored row is.
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
    // A running row sorts by the run's start; the rest keep last-activity order
    // (a synthesised row has no stored activity of its own, so its start stands
    // in for one).
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

/// One row of the list with the two keys its order is decided by.
struct ListRow {
    row: CachedSessionItem,
    /// Start of the run the row is pinned for; `None` when it keeps its
    /// last-activity place.
    pinned_at: Option<chrono::DateTime<chrono::Utc>>,
    /// The row's last activity, or the run's start for a synthesised row.
    last_activity: chrono::DateTime<chrono::Utc>,
}

/// What a session's agent is doing right now, from the live registry alone.
fn session_activity(handle: Option<&crate::agent::registry::AgentHandle>) -> SessionActivity {
    match handle {
        Some(handle) if handle.paused => SessionActivity::Paused,
        Some(_) => SessionActivity::Working,
        None => SessionActivity::Idle,
    }
}

/// The pin key of a row: the start of the run it is pinned for, `None` when it
/// keeps its last-activity place.
fn pinned_at(
    handle: Option<&crate::agent::registry::AgentHandle>,
    activity: SessionActivity,
) -> Option<chrono::DateTime<chrono::Utc>> {
    handle
        .filter(|_| activity == SessionActivity::Working)
        .map(|handle| handle.started_at)
}

/// The root a transcript's relative `[FILE:…]` markers resolve against: the
/// session's own workspace. A personal workspace is synthetic (never in the
/// map), so it resolves by name; a stored name that is itself an absolute path
/// still resolves when the workspace row is gone.
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
    /// so rows that appear, grow or finish converge without the user leaving the
    /// page.
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
    /// Mutated during layout (the bubble width is only known there), hence
    /// `RefCell`.
    measure_cache: RefCell<HashMap<(usize, usize), MessageMeasure>>,
    /// Cached session list display data. Rebuilt whenever the row set changes —
    /// a settled read or a runtime tick (which also moves relative-time labels
    /// and the live rows). `view()` builds widgets from this data on every
    /// frame.
    cached_session_items: Option<Vec<CachedSessionItem>>,
    /// A read was asked for while one was in flight: its rows are stale by the
    /// time they land, so one more read follows the settled one. This is what
    /// makes the last event of a burst count.
    refresh_pending: bool,

    // ── Transcript viewport fields ───────────────────────────────
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

    /// Re-read the session list: entering the page, every runtime tick, and
    /// after a delete or truncation, so a session created, grown, removed or
    /// finished while the page is open converges on its own. A read already in
    /// flight makes this a no-op that queues one more read for when it settles,
    /// so ticks never pile up and the last change of a burst is never lost. A
    /// failed read keeps the rows already on screen.
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
                // The selected session's workspace may only be named now: adopt
                // it so its relative marker targets resolve, and re-parse the
                // bodies already loaded.
                if self.adopt_workspace_root(workspaces) {
                    self.entry_md = session_view::parse_entry_bodies(
                        &self.entries,
                        self.workspace_root.as_deref(),
                    );
                }
                self.rebuild_session_cache();
                // Changes that landed while this read was in flight are older
                // than the read's own rows: one more read settles them.
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
                // The measurement cache is cleared in SessionMessages when the
                // new session's messages actually arrive; a failure for the new
                // session renders the error rather than the previous
                // transcript, so stale cache entries stay unused until the next
                // successful load clears them. Do NOT set auto_scroll_enabled
                // here — ScrollChanged owns it; the initial snap to bottom
                // happens in SessionMessages.
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
                    // Fresh session load: reset the measurement cache (the
                    // previous session's keys are index-stale) and any pending
                    // collapse from the old ledger.
                    self.measure_cache.borrow_mut().clear();
                    self.pending_collapses.clear();
                    self.selected_loading = false;
                    self.selected_error = None;
                    // Snap to bottom so the user sees the most recent messages
                    // immediately.
                    return iced::widget::operation::snap_to_end(self.scrollable_id.clone());
                }
                Task::none()
            }
            SessionsMessage::SessionError { key, error } => {
                // A slow read for a session the user has since left must not
                // render its failure in the newly selected transcript.
                if self.selected_session.as_deref() == Some(&key) {
                    self.selected_error = Some(error);
                    self.selected_loading = false;
                }
                Task::none()
            }
            SessionsMessage::ScrollChanged(viewport) => {
                // The fit check must stay first: `relative_offset` divides by
                // the content/viewport height difference, so a fitting
                // transcript reads as not-pinned there (exact fit is 0/0).
                self.auto_scroll_enabled = viewport.content_bounds().height
                    <= viewport.bounds().height
                    || viewport.relative_offset().y >= 0.99;
                Task::none()
            }
            SessionsMessage::ToggleExpand(i, j) => self.handle_toggle_expand(i, j),
            SessionsMessage::ToggleCancel(i, j) => {
                // Double-click word/line selection: the scheduled collapse is
                // dropped and the block stays put.
                self.pending_collapses.remove(&(i, j));
                Task::none()
            }
            SessionsMessage::ToggleCommit(i, j) => {
                // Apply the collapse only if no double-click canceled it.
                if self.pending_collapses.remove(&(i, j)) {
                    self.expanded.remove(&(i, j));
                }
                Task::none()
            }
            SessionsMessage::DeleteSession(key) => Task::perform(
                async move {
                    // User-initiated deletion: abandon the session's unfinished
                    // sync jobs first — a delete must not orphan them. A failure
                    // surfaces via the error toast and the delete is skipped so
                    // the user can retry.
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
                // A read in flight may predate the delete and would put the row
                // back; refreshing here settles the store's own answer.
                let refresh = self.refresh();
                // Only claim a "Deleted" success when a real row removal
                // happened; an already-absent session (cleaned up elsewhere)
                // just vanishes from the list without an inaccurate toast.
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
                // The stored row the entry was built from; without one the
                // ledger has no stored-row identity and the cut is a no-op.
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
                // A no-op cut has nothing to drop. A live merge landing between
                // the click and this reply can at worst prune at a stale index —
                // cosmetic: the store cut is durable and a re-select corrects
                // the view.
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
                    // The card's count renders from the list snapshot: drop the
                    // removed messages here and let the store confirm both the
                    // count and the cut (a read in flight may predate them).
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
                // The reveal stats and spawns a launcher: keep it out of the
                // update handler.
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

    /// Click-to-toggle of one collapsible element: expanding applies
    /// immediately; collapsing is scheduled and confirmed by a delayed
    /// [`SessionsMessage::ToggleCommit`]. The widget's second-click message
    /// ([`SessionsMessage::ToggleCancel`]) drops a scheduled collapse so a
    /// word/line-selection double-click never moves the block.
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

    /// Clear the currently selected session and ALL per-session transcript
    /// state, returning the transcript column to its placeholder.
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

    /// Rebuild the cached session list display data from the settled rows and
    /// the live registry. `view()` builds widgets from this data on every frame.
    fn rebuild_session_cache(&mut self) {
        let items = session_rows(
            self.sessions.entries(),
            &crate::agent::registry::AGENT_REGISTRY.list(),
            chrono::Local::now(),
        );
        self.cached_session_items = if items.is_empty() { None } else { Some(items) };
    }

    /// Coalesced runtime-change tick: re-read the list, rebuild the display data
    /// and merge the selected session's live transcript into the view.
    ///
    /// The registry publishes this tick on every agent start and stop, which is
    /// the whole of the list's convergence: rows are re-read here rather than on
    /// a timer because a session is only ever created or grown by a running
    /// agent. Being event-driven also keeps the order still while agents work —
    /// nothing re-sorts under the cursor between a run's start and its end.
    fn on_runtime_changed(
        &mut self,
        workspaces: &HashMap<String, Workspace>,
    ) -> Task<SessionsMessage> {
        self.rebuild_session_cache();
        Task::batch([self.merge_selected_transcript(workspaces), self.refresh()])
    }

    /// The root the selected session's relative marker targets resolve against:
    /// its workspace, by stored name. `None` while no session is selected or
    /// when its workspace row is gone and the name is not a path of its own.
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

    /// Adopt the selected session's workspace root: `true` when it changed, in
    /// which case the bodies and measurements made under the old root must be
    /// rebuilt by the caller.
    fn adopt_workspace_root(&mut self, workspaces: &HashMap<String, Workspace>) -> bool {
        let root = self.selected_workspace_root(workspaces);
        if root == self.workspace_root {
            return false;
        }
        self.workspace_root = root;
        self.measure_cache.borrow_mut().clear();
        true
    }

    /// Merge the selected session's live transcript into the view while its
    /// agent is running.
    ///
    /// The merge is append-oriented and diffed by the longest common entry
    /// prefix, so markdown re-parsing is limited to the changed tail and
    /// measurement-cache keys of unchanged entries stay valid (built entries
    /// drop their key and re-measure lazily; an in-place edit changes
    /// `content_len`, which the stale fingerprint re-measures anyway).
    /// Index-keyed expansion sets stay valid because the tail is append-only
    /// and out-of-range keys are harmless. Snapshot lookup doubles as the
    /// running check — `None` when the agent is not running; a recycled agent
    /// id surfaces the new run's fresh (empty) holder, replacing the displayed
    /// transcript wholesale.
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
        // A workspace resolved after the first load re-parses the whole ledger
        // and re-measures every bubble under it.
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
        // Live history positions, not stored rows: the merged ledger has no
        // stored-row identity (see `entry_rows`).
        self.entry_rows = None;
        self.entry_md.truncate(common);
        self.entry_md.extend(session_view::parse_entry_bodies(
            &self.entries[common..],
            self.workspace_root.as_deref(),
        ));

        // Live updates only auto-scroll when the user was already following
        // the bottom (same rule as the fresh-load path).
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

        // Error display — inset to align with the vscroll-wrapped list below.
        content = widgets::push_error_banner_inset(content, self.sessions.error());

        if !self.sessions.loaded() {
            content = content.push(widgets::scroll_h_inset(widgets::loading_text()));
        } else if let Some(cached) = self.cached_session_items.as_ref() {
            // Session list on the left side — built from cached display data
            // (rebuilt on every settled read and runtime tick); cards are
            // rebuilt from it on every frame.
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
                            .push(super::spinner(theme::TEXT_14));
                    }
                    // A frozen agent produces nothing: its row keeps its place
                    // and says so rather than spinning.
                    SessionActivity::Paused => {
                        title_row = title_row
                            .push(Space::new().width(Length::Fill))
                            .push(text("paused").size(theme::TEXT_11).color(theme::TEXT_MUTED));
                    }
                    SessionActivity::Idle => {}
                }

                // A synthesised row (a run whose first message is not stored
                // yet) has no count of its own: it says what it is doing
                // instead.
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
                                            // Meta row: messages left / tokens center /
                                            // time right, spread via two Fill gaps
                                            // across the card. Sessions that never
                                            // recorded a token length leave the center
                                            // empty.
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

                // A synthesised row has no stored session behind it, so there is
                // nothing to delete.
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

            // Transcript on the right side. Wrapped in `responsive` so the
            // collapse measurement uses the real bubble body width, correct on
            // first render and after every window resize.
            let workspace_root = self.workspace_root.clone();
            let transcript: iced::Element<'_, SessionsMessage> = if self.selected_loading {
                iced::widget::container(widgets::loading_text())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(theme::PAD_16)
                    .into()
            } else if let Some(err) = self.selected_error.as_deref() {
                // The transcript read failed: render the failure where the
                // transcript would be, instead of the previous transcript or
                // the empty-session message.
                container(widgets::error_banner_fill(err))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(theme::PAD_16)
                    .into()
            } else if self.selected_session.is_some() {
                // Truncate targets stored rows and is offered only on an idle
                // session — registered membership is what disables it, so a
                // frozen agent (registered, but not working) keeps it off too.
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
                    // The transcript scrollable wrapper adds 8px padding per
                    // side. Every transcript round (message or tool round)
                    // renders as a chat bubble; all content measures at the
                    // bubble body width = 3/4 of the transcript row minus the
                    // bubble's 10px padding per side, folded in with a +4px
                    // safety margin that errs toward collapse.
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
            // A failed read must not masquerade as a legitimately empty list:
            // the error banner above is the only thing rendered.
            content = content.push(widgets::empty_state_placeholder(
                lucide::layout_dashboard::<iced::Theme, iced::Renderer>(),
                "No sessions",
                theme::TEXT_MUTED,
            ));
        }

        widgets::page(content)
    }
}

/// Length of the longest common prefix of two ledgers (first differing
/// entry index). `build_ledger` is deterministic, so an unchanged message
/// prefix yields an identical entry prefix.
fn common_entry_prefix(old: &[SessionEntry], new: &[SessionEntry]) -> usize {
    old.iter().zip(new).take_while(|(a, b)| a == b).count()
}

/// Author label above a bubble for the roles that carry one, flush left like
/// their bubbles. One non-accent color for every labeled role, so roles stay
/// visually consistent.
///
/// `Assistant` bubbles are unlabelled — `None`, so neither the label nor the
/// gap it occupied is emitted.
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

/// Chat-bubble row for one ledger entry: the roles that carry an author label
/// render it above the bubble; the bubble itself aligns right for assistant
/// rounds and left otherwise. Same background/typography as the Home chat
/// bubbles.
///
/// `menu` wraps the BUBBLE alone — not the label, and not the spacer
/// `align_bubble` adds beside it, so the menu's hit target is the bubble.
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

/// Resolve the collapse measurement for an element at the current
/// `text_width`, measuring on cache miss (cached per element and width; the
/// session ledger is append-only, so a cached entry stays valid). Returns the
/// wrapped-line count and — only when
/// `need_preview` is set (the element is currently collapsed) — its 3-line
/// plain-text preview, so expanded elements do not pay a per-frame preview
/// clone. `font_size` is the body size the element renders at (body or
/// narration).
///
/// Stale-fingerprint semantics: a cache hit with the same width bucket,
/// `content_len`, and `font_size` is reused; the same content at a different
/// bucket (or size) is re-measured via [`re_measure`]; otherwise a fresh
/// measurement runs.
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
    markers: &media_markers::MarkerContext<'_>,
) -> (u32, Option<String>) {
    let bucket = width_bucket(text_width);
    let content_len = content.len();
    let mut cache = measure_cache.borrow_mut();
    let cache_hit = cache.get(&key);
    let stale = cache_hit.is_none_or(|m| {
        m.width_bucket != bucket || m.content_len != content_len || m.font_size != font_size
    });
    if stale {
        // Cache miss: measure. A pure width change reuses the cached
        // processed display text (an `Arc` clone); otherwise measure from the
        // caller's plain-text content.
        let m = match cache_hit {
            Some(m)
                if m.width_bucket != bucket
                    && m.content_len == content_len
                    && m.font_size == font_size =>
            {
                re_measure(m, text_width)
            }
            _ => measure_message(content, markers, text_width, content_len, font_size),
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

/// The collapsed-preview-versus-full content of a collapsible element: when
/// it can collapse and is collapsed, a [`MAX_PREVIEW_LINES`]-line preview
/// (in `size`/`font`) with an "ellipsis" cue for the hidden content on its own
/// line under the preview, flush left — never trailing the last preview line,
/// where only a line that happens to wrap would put it on a line of its own;
/// otherwise the caller's `expanded` element. Callers wrap the result in
/// [`click_toggle`] — never nesting two toggle areas (a nested double-toggle
/// cancels out).
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

/// The marker context a transcript body resolves under: its session workspace,
/// and whether an undeliverable marker becomes a refusal.
fn marker_context<'a>(
    ctx: &'a TranscriptCtx<'_>,
    refusals: bool,
) -> media_markers::MarkerContext<'a> {
    media_markers::MarkerContext {
        workspace: ctx.workspace_root.as_deref(),
        refusals,
    }
}

/// Shared collapse-state preamble for the transcript's collapsible elements:
/// returns `(is_expanded, collapses, preview)` — whether the user expanded the
/// element, whether it renders more than [`MAX_PREVIEW_LINES`] wrapped lines
/// at `font_size` (measured at the bubble body width), and the measured
/// 3-line preview if it can collapse.
fn collapse_state(
    ctx: &TranscriptCtx<'_>,
    key: (usize, usize),
    content: &str,
    font_size: f32,
    refusals: bool,
) -> (bool, bool, Option<String>) {
    let is_expanded = ctx.expanded.contains(&key);
    let (wrapped_lines, preview) = element_measurement(
        ctx.measure_cache,
        key,
        ctx.text_width,
        !is_expanded,
        content,
        font_size,
        &marker_context(ctx, refusals),
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
    let (is_expanded, collapses, preview) =
        collapse_state(ctx, key, content, theme::MARKDOWN_TEXT_SIZE, false);
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

/// The collapsible Thinking block: lucide `brain` + "Thinking" header above a
/// 3-line-collapse body measured at the bubble body width. The measurement
/// stays at [`theme::MARKDOWN_TEXT_SIZE`] (the body renders at 11px) — the
/// err-toward-collapsing bias.
fn thinking_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
) -> Element<'a, SessionsMessage> {
    let (is_expanded, collapses, preview) =
        collapse_state(ctx, key, content, theme::MARKDOWN_TEXT_SIZE, false);
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

/// The message/narration body with the 3-line collapse rule: a collapsed
/// element shows a plain clickable preview; an expanded (or short) element
/// draws the body through [`media_markers::body`] — its attachments as one
/// group above its whole text, never interleaved and neither ever dropped.
/// `content` is the plain text used for the collapse measurement (at the bubble
/// body width; the measurement runs the same preprocessing and takes
/// `display_text()`, so preview and expanded agree); `md` is the parsed body
/// (markdown items, attachments and whether its markers became refusals) for
/// the expanded view. Narration bodies render italic at
/// [`theme::NARRATION_TEXT_SIZE`]; user/assistant bodies stay at
/// [`theme::MARKDOWN_TEXT_SIZE`] regular.
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
    // The refusal rule is a property of the parsed body, so the measurement of
    // a body always resolves under the context it was parsed with.
    let refusals = md.is_some_and(|parsed| parsed.refusals);
    let (is_expanded, collapses, preview) = collapse_state(ctx, key, content, size, refusals);
    let font = if is_narration {
        theme::FONT_ITALIC
    } else {
        theme::FONT_REGULAR
    };
    // The unadorned body: what an unparsed message draws, and what a parsed one
    // falls back to when the composition leaves nothing to show.
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

/// The tool-call result block: plain collapsible text at [`theme::TEXT_SECONDARY`]
/// (11px) with the 3-line collapse — no leading icon (results are recognizable
/// from context; "(no result)" keeps its arrow below). Results render inside
/// the assistant bubble, so they measure at the bubble body width.
fn result_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    result: &'a str,
) -> Element<'a, SessionsMessage> {
    plain_collapsible(ctx, key, result, theme::TEXT_SECONDARY)
}

/// Render one session round as a single assistant bubble containing the
/// reasoning block, the narration body, and one tool block per call with its
/// collapsible result beneath. An empty narration slot with a decoded
/// Reasoning block promotes the reasoning into the narration slot (plain
/// text) instead of a separate Thinking block.
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
        // The reasoning IS the narration for this round: render it in the
        // narration slot as plain text (md None — no markdown parse, so CoT
        // characters never become formatting) and skip the Thinking block,
        // whose text would duplicate it.
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
        // Only a result that actually collapses makes the tool block a click
        // target (toggling a short result's expansion would be a no-op).
        let collapses = call.result.as_deref().is_some_and(|result| {
            let (wrapped_lines, _) = element_measurement(
                ctx.measure_cache,
                key,
                ctx.text_width,
                false,
                result,
                theme::MARKDOWN_TEXT_SIZE,
                &marker_context(ctx, false),
            );
            wrapped_lines > MAX_PREVIEW_LINES
        });
        // The tool block is wrapped in a click-to-toggle (not a button) so the
        // shared tool block's hover tooltip keeps working; a short click
        // anywhere toggles the associated result's collapse, and dragging over
        // the selectable tool-args text selects instead. click_to_toggle also
        // never captures the pointer, so the tooltip stays.
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
            // Empty or missing ToolResult record: the old view's explicit
            // "(no result)" indicator.
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

    // One bubble for the whole round, same container as assistant text rounds
    // (right-aligned, 75% width, no author label).
    bubble_row(crate::ChatRole::Assistant, bubble_col, None)
}

/// Render one ledger entry as a chat-bubble row: a `Message` renders its own
/// bubble, and a `ToolRound` renders a single assistant bubble containing the
/// round's reasoning, narration, and tool calls with their results.
///
/// `None` means the entry has nothing to render at all: a message with neither
/// a body nor thinking whose role carries no author label (i.e. an assistant
/// message) would be a blank row, so the caller skips it — a zero-size child
/// would still leave a band of inter-entry spacing.
fn render_entry<'a>(ctx: &TranscriptCtx<'a>, i: usize) -> Option<Element<'a, SessionsMessage>> {
    let entry = &ctx.entries[i];
    match entry {
        SessionEntry::Message {
            role,
            content,
            thinking,
        } => {
            if thinking.is_none() && content.is_none() {
                // Nothing to show but the author: the row is the author label,
                // which exists only for the labelled roles.
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

/// Render the transcript column for the selected session: build the chat
/// bubbles from the shared ledger and apply the 3-line collapse rule to
/// message bodies, thinking blocks, and tool results.
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
        // Only the bodyless assistant row is skipped: the labelled roles keep
        // their label-only row, and an assistant row with a body still renders.
        assert_eq!(rendered, [true, true, true, false, true]);
    }

    #[test]
    fn running_sessions_are_pinned_above_last_activity_order() {
        let now = chrono::Utc::now();
        let mut state = SessionsState::new();
        state.sessions.settle(Ok(vec![
            metadata("a", now - chrono::Duration::minutes(10)),
            metadata("b", now - chrono::Duration::minutes(5)),
            metadata("c", now),
        ]));

        let rows = session_rows(
            state.sessions.entries(),
            &[
                agent("a", "engineer", "ws", now - chrono::Duration::minutes(1)),
                // Started while the page was open, before its first message
                // landed: registered, but with no stored row of its own.
                agent("zz", "analyst", "ws", now),
            ],
            now.into(),
        );

        let ids: Vec<&str> = rows.iter().map(|r| r.agent_id.as_str()).collect();
        // Running rows on top (oldest run first), then the rest by last
        // activity.
        assert_eq!(ids, ["a", "zz", "c", "b"]);
        assert_eq!(rows[1].msg_count_label, None, "no stored row, no count");
        assert_eq!(
            rows[1].title, "ws",
            "a synthesised row is titled by project"
        );
        assert_eq!(
            rows[1].role,
            "analyst".parse::<crate::Role>().ok(),
            "and carries its role"
        );
    }

    #[test]
    fn a_paused_agent_keeps_its_last_activity_place() {
        let now = chrono::Utc::now();
        let sessions = [
            metadata("a", now - chrono::Duration::minutes(10)),
            metadata("b", now),
        ];
        let mut paused = agent("a", "engineer", "ws", now);
        paused.paused = true;

        let rows = session_rows(&sessions, &[paused], now.into());
        let ids: Vec<&str> = rows.iter().map(|r| r.agent_id.as_str()).collect();

        assert_eq!(ids, ["b", "a"], "a frozen run is not pinned");
        assert_eq!(rows[1].activity, SessionActivity::Paused);
    }

    #[test]
    fn a_workspace_named_after_the_load_reparses_the_ledger() {
        let tmp = tempfile::tempdir().unwrap();
        let photo = tmp.path().join("generated/poster.png");
        std::fs::create_dir_all(photo.parent().unwrap()).unwrap();
        std::fs::write(&photo, b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").unwrap();

        // Loaded before the workspace could be named: with no root the relative
        // picture is refused rather than drawn.
        let mut state = SessionsState::new();
        state.selected_session = Some("s1".to_string());
        state.entries = vec![msg(
            ChatRole::Assistant,
            Some("[IMAGE:generated/poster.png]"),
        )];
        state.entry_md = session_view::parse_entry_bodies(&state.entries, None);
        assert_eq!(state.workspace_root, None);
        assert!(state.entry_md[0].as_ref().unwrap().attachments.is_empty());
        assert!(
            state.entry_md[0]
                .as_ref()
                .unwrap()
                .items
                .iter()
                .all(|item| !matches!(item, iced::widget::markdown::Item::Image { .. }))
        );

        // The workspace list arrives later: the already-loaded ledger is
        // re-parsed under the root, so the picture now renders.
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
        assert!(
            parsed.refusals,
            "an assistant body is the product's own reply"
        );
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
    fn common_entry_prefix_empty_vs_empty() {
        assert_eq!(common_entry_prefix(&[], &[]), 0);
    }

    #[test]
    fn common_entry_prefix_identical() {
        let old = vec![msg(ChatRole::User, Some("hi")), tool_round()];
        let new = old.clone();
        assert_eq!(common_entry_prefix(&old, &new), old.len());
    }

    #[test]
    fn common_entry_prefix_pure_append() {
        let old = vec![msg(ChatRole::User, Some("hi"))];
        let mut new = old.clone();
        new.push(msg(ChatRole::Assistant, Some("hello")));
        new.push(tool_round());
        assert_eq!(common_entry_prefix(&old, &new), old.len());
    }

    #[test]
    fn common_entry_prefix_in_place_edit() {
        let old = vec![
            msg(ChatRole::User, Some("a")),
            msg(ChatRole::Assistant, Some("b")),
            msg(ChatRole::User, Some("c")),
            msg(ChatRole::Assistant, Some("d")),
        ];
        let mut new = old.clone();
        new[2] = msg(ChatRole::User, Some("edited"));
        assert_eq!(common_entry_prefix(&old, &new), 2);
    }

    #[test]
    fn common_entry_prefix_differing_lengths() {
        let old = vec![
            msg(ChatRole::User, Some("a")),
            msg(ChatRole::Assistant, Some("b")),
        ];
        let new = vec![msg(ChatRole::User, Some("a"))];
        assert_eq!(common_entry_prefix(&old, &new), 1);
    }

    #[test]
    fn transcript_error_stays_out_of_the_list_banner() {
        let mut state = SessionsState::new();
        state.selected_session = Some("s1".to_string());
        state.selected_loading = true;

        let _ = state.update(
            SessionsMessage::SessionError {
                key: "s1".to_string(),
                error: "read failed".to_string(),
            },
            &HashMap::new(),
        );

        assert_eq!(state.selected_error.as_deref(), Some("read failed"));
        assert!(!state.selected_loading);
        assert!(
            state.sessions.error().is_none(),
            "a transcript failure must not surface in the session-list banner"
        );

        // An error for a session the user has since left is ignored.
        state.selected_error = None;
        state.selected_loading = true;
        let _ = state.update(
            SessionsMessage::SessionError {
                key: "s2".to_string(),
                error: "stale failure".to_string(),
            },
            &HashMap::new(),
        );
        assert!(state.selected_error.is_none());
        assert!(state.selected_loading);
    }

    #[test]
    fn successful_transcript_load_clears_the_error() {
        let mut state = SessionsState::new();
        state.selected_session = Some("s1".to_string());
        state.selected_error = Some("old failure".to_string());
        state.selected_loading = true;

        let _ = state.update(
            SessionsMessage::SessionMessages("s1".to_string(), vec![]),
            &HashMap::new(),
        );

        assert!(state.selected_error.is_none());
        assert!(!state.selected_loading);
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
    fn workspace_root_for_resolves_every_stored_name_shape() {
        let admin = crate::users::ADMIN_USER_NAME;
        assert_eq!(
            workspace_root_for(&format!("personal:{admin}"), &HashMap::new()),
            Some(crate::users::personal_workspace_path(admin)),
            "a personal workspace is synthetic, so it resolves by name"
        );

        let mut map = workspace_map(&["mahbot_rlx"]);
        map.get_mut("mahbot_rlx").unwrap().path = "/tmp/ws/mahbot_rlx".to_string();
        assert_eq!(
            workspace_root_for("mahbot_rlx", &map),
            Some(PathBuf::from("/tmp/ws/mahbot_rlx")),
            "a managed workspace resolves to its path"
        );
        assert_eq!(
            workspace_root_for("/tmp/ws/stored", &map),
            Some(PathBuf::from("/tmp/ws/stored")),
            "an absolute stored name is its own root"
        );
        assert_eq!(
            workspace_root_for("gone", &map),
            None,
            "an unknown relative name has no root"
        );
    }

    #[test]
    fn workspace_scoped_cards_are_titled_by_their_managed_workspace() {
        let map = workspace_map(&["mahbot_rlx"]);

        for role in [crate::Role::Manager, crate::Role::Maintainer] {
            let scoped = cached_item(role, Some("admin"), Some("mahbot_rlx"));
            assert_eq!(scoped.display_title(&map), "mahbot_rlx");
            // The last turn's user never decides a workspace-scoped title.
            let no_user = cached_item(role, None, Some("mahbot_rlx"));
            assert_eq!(no_user.display_title(&map), "mahbot_rlx");
            // A workspace no longer in the managed set leaves the role alone.
            let orphaned = cached_item(role, Some("admin"), Some("deleted_ws"));
            assert_eq!(orphaned.display_title(&map), role.display_label());
            // A session with no recorded workspace is equally "not known".
            let no_workspace = cached_item(role, Some("admin"), None);
            assert_eq!(no_workspace.display_title(&map), role.display_label());
        }

        // Every other role keeps the user name it shows today.
        let assistant = cached_item(crate::Role::Assistant, Some("admin"), Some("mahbot_rlx"));
        assert_eq!(assistant.display_title(&map), "admin");
        let engineer = cached_item(crate::Role::Engineer, Some("alice"), Some("mahbot_rlx"));
        assert_eq!(engineer.display_title(&map), "alice");
    }
}
