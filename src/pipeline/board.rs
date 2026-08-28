//! Ticket/board system — Turso-backed task management.

use crate::db::{self, IntoParams, TxGuard, Value};
use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::LazyLock;
use tracing::{debug, info, warn};

crate::define_store! {
    /// Global board store.
    pub static BOARD: BoardStore,
    post_open = after_open,
    expect = "BOARD not initialized — call init_all_stores() first",
}

/// Background task: auto-archive cancelled tickets older than 1 hour.
///
/// Runs every 5 minutes, respects the global shutdown token via
/// [`crate::shutdown::sleep_or_shutdown_or_drain`] (same pattern as
/// [`crate::agent::maintainer::run_maintainer_loop`]).
/// Logs per-pass failures and continues.
pub async fn run_archive_cancelled_loop() {
    let interval = std::time::Duration::from_mins(5);

    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }

        // Engineer-anchor terminal deletion (S5): remove permanently-NULL
        // seats for tickets in a terminal phase — the TTL guard stops
        // protecting the accumulated engineer session once the anchor is gone
        // (idempotent, ≤5-min delay against the 8h TTL).
        crate::jobs::purge_terminal_session_pins().await;

        let Some(board) = BOARD.get() else {
            warn!("Archive cancelled loop: board not initialized");
            continue;
        };

        match board.archive_stale_cancelled(CANCELLED_ARCHIVE_HOURS).await {
            Ok(n) if n > 0 => info!(count = n, "Archived stale cancelled tickets"),
            Ok(_) => debug!("Archive cancelled loop: no stale tickets"),
            Err(e) => warn!(error = %e, "Archive cancelled loop failed"),
        }
    }
}

const TICKETS_FTS_INDEX_NAME: &str = "idx_tickets_title_fts";
const TICKETS_FTS_INDEX_DDL: &str = "\
CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets \
USING fts (title) WITH (tokenizer = 'ngram')";

/// Stale-cancelled archival window (hours): Cancelled tickets this old are
/// archived by [`run_archive_cancelled_loop`].
const CANCELLED_ARCHIVE_HOURS: i64 = 1;

// Column definitions for ticket SELECT/RETURNING queries.
crate::columns! {
    TICKET_COLUMNS [TICKET] {
        ID                     => "id",
        TITLE                  => "title",
        DESCRIPTION            => "description",
        PHASE                  => "phase",
        WORKSPACE_NAME         => "workspace_name",
        CREATED_AT             => "created_at",
        UPDATED_AT             => "updated_at",
        PREREQUISITES          => "prerequisites",
        SUPERSEDES             => "supersedes",
        SUPERSEDED_BY          => "superseded_by",
        COMMIT_HASH            => "commit_hash",
        LINES_ADDED            => "lines_added",
        LINES_REMOVED          => "lines_removed",
        REPORTER               => "reporter",
        IS_ARCHIVED            => "is_archived",
        PRIORITY               => "priority",
        REVIEWED_HEAD          => "reviewed_head",
        REVIEWED_TREE          => "reviewed_tree",
        DONE_AT                => "done_at",
        BOUNCE_COUNT           => "bounce_count",
    }
}

// Column definitions for comment SELECT queries.
// Note: `id` and `ticket_id` are intentionally excluded from the column list
// because they are not consumed by the comment rendering path:
// - `ticket_id` is already known from the parent ticket query context
// - `id` is not read by any comment consumer
// These columns remain in the database schema and are not candidates for removal.
crate::columns! {
    COMMENT_COLUMNS [COMMENT] {
        ROLE       => "role",
        CONTENT    => "content",
        CREATED_AT => "created_at",
    }
}

/// Phases where a ticket occupies the dev/review/QA pipeline.
///
/// Only one ticket at a time per workspace may be in this pipeline. Any ticket in one of these
/// phases blocks new Engineer dispatches for that workspace. The Maintainer uses a separate
/// pre-development threshold (Analysis + Planning + ReadyForDevelopment) and is no longer
/// directly suppressed by this constant.
///
/// Occupancy is owned by the phase job (a short-lived job whose `kind` is the
/// ticket's current phase) — a ticket in any of these phases always has a
/// live phase job, and that job is the single authority that runs the stage
/// then advances the ticket to the next phase. `tickets.phase` is both the
/// displayed pipeline state and the input gate for the claim step
/// (Backlog → Analysis, ReadyForDevelopment → InDevelopment) — not a mirror
/// of a long-lived job.
const PIPELINE_OCCUPIED_PHASES: &[TicketPhase] = &[
    TicketPhase::InDevelopment,
    TicketPhase::InDiagnostics,
    TicketPhase::InReview,
    TicketPhase::InQa,
    TicketPhase::InSanitation,
];

/// Phases that unblock dependent tickets.
///
/// When a ticket transitions to one of these phases, any tickets that
/// depend on it become eligible for claiming (their prerequisite filter
/// no longer blocks them).
///
/// [`TicketPhase::Failed`] is intentionally excluded — a failed ticket
/// permanently blocks its dependents, requiring manual intervention.
///
/// [`TicketPhase::is_unblocking`] delegates to this constant to ensure
/// the unblocking set is always authoritative. If a new phase is added
/// here, `is_unblocking()` automatically picks it up; if the set ever
/// needs to diverge from the unblocking set, this delegation must be
/// broken explicitly.
pub const UNBLOCKING_PHASES: &[TicketPhase] = &[TicketPhase::Done, TicketPhase::Cancelled];

/// Terminal phases — a ticket in one of these can no longer be claimed.
///
/// [`TicketPhase::is_terminal`] delegates to this constant.
///
/// Note: `Failed` is deliberately included here even though it is excluded
/// from [`UNBLOCKING_PHASES`] (a failed ticket permanently blocks dependents).
pub const TERMINAL_PHASES: &[TicketPhase] = &[
    TicketPhase::Done,
    TicketPhase::Cancelled,
    TicketPhase::Failed,
];

/// Produces an SQL fragment listing phases as quoted, comma-separated
/// strings — e.g. `'done', 'cancelled'`.
///
/// # Precondition
///
/// The input slice must be non-empty. Passing an empty slice produces
/// `WHERE phase IN ()` which is invalid SQL.
fn phase_list_sql_fragment(phases: &[TicketPhase]) -> String {
    phases
        .iter()
        .map(|p| format!("'{}'", p.as_ref()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_prereqs(raw: &str) -> Result<Vec<String>> {
    serde_json::from_str(raw).with_context(|| {
        let preview = if raw.len() > 200 {
            format!("{}…", crate::util::truncate_bytes(raw, 200))
        } else {
            raw.to_string()
        };
        format!("Corrupt prerequisites JSON in database: {preview}")
    })
}

/// Bundled parameters for ticket creation.
///
/// Reduces parameter explosion across [`BoardStore::insert_ticket_tx`],
/// [`BoardStore::create_ticket`], and [`BoardStore::supersede_and_create`].
#[derive(Debug, Clone)]
pub(crate) struct TicketParams {
    pub title: String,
    pub description: String,
    pub workspace_name: String,
    pub phase: TicketPhase,
    pub prerequisites: Vec<String>,
    pub reporter: String,
    pub embedding: Option<Vec<u8>>,
    pub priority: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TicketComment {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ticket {
    pub id: String,
    pub title: String,
    pub description: String,
    pub phase: TicketPhase,
    pub workspace_name: String,
    pub created_at: String,
    pub updated_at: String,
    pub comments: Vec<TicketComment>,
    /// IDs of tickets that must be completed before this one can be claimed.
    pub prerequisites: Vec<String>,
    /// ID of the ticket this one supersedes (set when created via supersede).
    /// The superseded ticket is cancelled atomically during creation.
    pub supersedes: Option<String>,
    /// ID of the ticket that supersedes this one (set on the old ticket when
    /// it is superseded). Purely informational — never drives logic.
    pub superseded_by: Option<String>,
    /// Full commit SHA (40 hex chars), `None` if no commit recorded.
    pub commit_hash: Option<String>,
    /// Lines added (non-negative) from the associated commit.
    pub lines_added: Option<i64>,
    /// Lines removed (non-negative) from the associated commit.
    pub lines_removed: Option<i64>,
    /// Creator/tool identity — set at construction time.
    /// Also used in the GUI board display: when the value matches a known role
    /// name, the role's display label is shown (e.g. "Manager"); otherwise the
    /// content is shown with the first character uppercased (e.g. "Test" for
    /// "test"). May be empty when no reporter is recorded.
    pub reporter: String,
    /// Whether this ticket has been archived (hidden from normal listings).
    pub is_archived: bool,
    pub priority: i64,
    /// HEAD commit hash at the last completed reviewer round on this ticket.
    /// `None` until the first reviewer pass finishes — used by the reviewer
    /// skip-gate to detect brand-new content that must never skip review.
    pub reviewed_head: Option<String>,
    /// `git write-tree` index tree hash at the last completed reviewer round
    /// (captured after the post-review auto-stage). Together with
    /// [`reviewed_head`](Self::reviewed_head) and a clean porcelain this
    /// identifies the exact content reviewers saw.
    pub reviewed_tree: Option<String>,
    /// Exact completion timestamp: set on transition to Done, cleared when the
    /// ticket leaves Done. `None` for never-done or not-currently-done tickets.
    pub done_at: Option<String>,
    /// Number of times this ticket bounced back into development from a
    /// validation-phase non-success (diagnostics/review/QA/sanitation). Drives
    /// the bounce-based circuit breaker (max 10). Engineer hard failures are
    /// pause-only (workspace pause, implementation frozen) and do not consume this
    /// budget.
    pub bounce_count: i64,
}

impl Ticket {
    /// Short single-line display for listing tickets in agent-facing output.
    ///
    /// Returns `"  [{reporter}] [{phase}] {id}: {title}"` (note the leading
    /// two-space indent for alignment within a multi-line block). The trailing
    /// newline is omitted — callers add it via `writeln!` or equivalent.
    ///
    /// The `{phase}` field uses the snake_case Display representation from
    /// [`TicketPhase`] (e.g. `"in_development"`), which is the canonical form
    /// for agent-facing output. For user-facing labels with spaces instead of
    /// underscores, use [`TicketPhase::display_name()`] directly.
    ///
    /// ## Related formatting (not duplicated here)
    ///
    /// - `crate::prompt::format_ticket_block` produces a Markdown
    ///   `<current-ticket>` block for system messages — intentionally different
    ///   format and should not be unified.
    /// - `search_archived_tickets` format omits the reporter field
    ///   (`"  [{phase}] {id}: {title}"`) — intentionally different.
    #[must_use]
    pub fn short_display(&self) -> String {
        format!(
            "  [{}] [{}] {}: {}",
            self.reporter, self.phase, self.id, self.title
        )
    }

    /// Produce a detailed multi-line display of the ticket, suitable for
    /// [`GetTicketTool`](crate::tools::ticket::GetTicketTool) and other agent-facing output.
    ///
    /// The output includes these fields (when present):
    ///
    /// - Ticket ID, Title, Description
    /// - Phase (snake_case — e.g. `ready_for_development`)
    /// - Priority (P0–P4+ label)
    /// - Reporter, Workspace, Created, Updated
    /// - Supersedes, Superseded by, Prerequisites (conditionally when non-empty)
    /// - Archived flag (conditionally when `true`)
    /// - Comments block (via [`Self::format_comments`])
    ///
    /// ## Fields *not* displayed
    ///
    /// The following [`Ticket`] fields are deliberately omitted — they are
    /// available in the board UI but not meaningful for agent context:
    ///
    /// - `commit_hash`
    /// - `lines_added` / `lines_removed`
    ///
    /// ## Output size
    ///
    /// The returned string can be arbitrarily large (unbounded descriptions
    /// and comments). Callers that need truncation should apply their own
    /// limits (see `GetTicketTool::preserve_full_output` for an example that
    /// disables the default 5 KB truncation). `GetTicketTool` uses
    /// [`Self::detailed_display_limited`] for its default truncated view.
    ///
    /// ## Changing this output
    ///
    /// Because agent tool calls depend on this exact format, changes to
    /// displayed fields or layout must be kept in sync with
    /// [`crate::tools::ticket::GetTicketTool`] (the primary consumer). If new fields are
    /// added here, update the integration test `test_get_ticket_tool`
    /// to prevent silent divergence.
    #[must_use]
    pub fn detailed_display(&self) -> String {
        let mut out = self.header(&self.description);
        out.push_str(&self.format_comments());
        out
    }

    /// Return a truncated version of [`Self::detailed_display`] for agent-facing
    /// output where the caller only needs the ticket's latest state.
    ///
    /// The description value is capped at `desc_max_chars` characters (a trailing
    /// `…` is appended when it exceeds the cap) and each comment's content is
    /// capped at `comment_max_chars` characters, EXCEPT the newest `last_n_full`
    /// comments which are always shown in full. Header metadata (id, title, phase,
    /// priority, reporter, workspace, dates, supersede/prerequisite links,
    /// archived flag) is never truncated.
    #[must_use]
    pub(crate) fn detailed_display_limited(
        &self,
        desc_max_chars: usize,
        comment_max_chars: usize,
        last_n_full: usize,
    ) -> String {
        let description = crate::util::truncate(&self.description, desc_max_chars);
        let mut out = self.header(&description);
        out.push_str(&self.format_comments_limited(comment_max_chars, last_n_full));
        out
    }

    /// Build the shared header of [`Self::detailed_display`] and
    /// [`Self::detailed_display_limited`], rendering `description` verbatim.
    #[must_use]
    fn header(&self, description: &str) -> String {
        let mut out = format!(
            "Ticket: {id}\n\
             Title: {title}\n\
             Description: {description}\n\
             Phase: {phase}\n\
             Reporter: {reporter}\n\
             Workspace: {workspace}\n\
             Created: {created}\n\
             Updated: {updated}\n\
             Priority: P{priority}\n",
            id = self.id,
            title = self.title,
            description = description,
            phase = self.phase,
            reporter = self.reporter,
            workspace = self.workspace_name,
            created = self.created_at,
            updated = self.updated_at,
            priority = self.priority,
        );
        if let Some(ref s) = self.supersedes {
            let _ = writeln!(out, "Supersedes: {s}");
        }
        if let Some(ref s) = self.superseded_by {
            let _ = writeln!(out, "Superseded by: {s}");
        }
        if !self.prerequisites.is_empty() {
            let _ = writeln!(out, "Prerequisites: {}", self.prerequisites.join(", "));
        }
        if self.is_archived {
            out.push_str("Archived: yes\n");
        }
        out
    }

    /// Format comments as a `"Comments:"` block suitable for [`crate::tools::ticket::GetTicketTool`].
    ///
    /// Returns a string starting with `"Comments:"` followed by one line per
    /// comment in the format `"\n  [{role}] ({timestamp}): {content}"`, or
    /// `"Comments:\n  (no comments)"` if the comment list is empty.
    #[must_use]
    fn format_comments(&self) -> String {
        let mut s = String::from("Comments:");
        if self.comments.is_empty() {
            s.push_str("\n  (no comments)");
        } else {
            for c in &self.comments {
                s.push_str(&Self::comment_line(c, &c.content));
            }
        }
        s
    }

    /// Like [`Self::format_comments`], but caps each comment's content at
    /// `max_chars` characters (appending `…` when truncated) except for the
    /// newest `last_n_full` comments, which are shown in full.
    #[must_use]
    fn format_comments_limited(&self, max_chars: usize, last_n_full: usize) -> String {
        let mut s = String::from("Comments:");
        if self.comments.is_empty() {
            s.push_str("\n  (no comments)");
        } else {
            let n = self.comments.len();
            for (i, c) in self.comments.iter().enumerate() {
                let content = if i + last_n_full < n {
                    crate::util::truncate(&c.content, max_chars)
                } else {
                    c.content.clone()
                };
                s.push_str(&Self::comment_line(c, &content));
            }
        }
        s
    }

    /// Format a single comment line: `"\n  [{role}] ({timestamp}): {content}"`.
    ///
    /// Timestamps are truncated to seconds (`[..19]` of the RFC 3339 string),
    /// with a defensive `min()` guard against abnormally short strings.
    #[must_use]
    fn comment_line(c: &TicketComment, content: &str) -> String {
        let end = 19.min(c.created_at.len());
        let ts = &c.created_at[..end];
        format!("\n  [{}] ({}): {}", c.role, ts, content)
    }
}
/// Lowercase snake_case strings matching the DB column values — no schema
/// migration needed. Display, AsRefStr, and EnumIter are derived via `strum`;
/// FromStr is implemented manually for user-friendly error messages.
#[derive(
    Debug, Clone, Copy, PartialEq, Serialize, strum::Display, strum::AsRefStr, strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TicketPhase {
    Backlog,
    Analysis,
    /// Ticket is waiting for Manager review. Not picked up automatically by any agent —
    /// the Manager or user must manually advance it to ReadyForDevelopment or cancel it.
    Planning,
    ReadyForDevelopment,
    InDevelopment,
    InDiagnostics,
    InSanitation,
    InReview,
    InQa,
    Done,
    Cancelled,
    Failed,
}

impl TicketPhase {
    /// Returns `true` for phases that unblock dependent tickets.
    ///
    /// Delegates to [`UNBLOCKING_PHASES`] so the unblocking set can never
    /// accidentally diverge from the prerequisite-unblocking set.
    /// [`TicketPhase::Failed`] is not in [`UNBLOCKING_PHASES`] and is
    /// therefore not unblocking — a failed ticket permanently blocks its
    /// dependents and remains visible in active views for manual triage.
    #[must_use]
    pub fn is_unblocking(&self) -> bool {
        UNBLOCKING_PHASES.contains(self)
    }

    /// Returns `true` for terminal phases (`Done`, `Cancelled`, `Failed`) — a
    /// ticket in one of these can no longer be claimed, so its phase job is
    /// finished.
    ///
    /// Delegates to [`TERMINAL_PHASES`] so the terminal set is authoritative
    /// for both the transition-clearing clause and the implementation completion.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        TERMINAL_PHASES.contains(self)
    }

    /// Returns `true` if the ticket is in a pipeline-occupied phase.
    ///
    /// Tickets in these phases occupy the dev/review/QA pipeline — only one
    /// ticket per workspace may be in the pipeline at a time. Each phase is
    /// owned by its own short-lived phase job (development, diagnostics,
    /// review, QA, sanitation). The automated
    /// create-ticket tool (when superseding an existing ticket) and the
    /// update-ticket tool refuse to modify tickets in any of these phases to
    /// prevent race conditions during phase transitions. `add_comment` is the
    /// exception: it is allowed during any phase and delivers to a running
    /// agent as a soft deferred message (or persists for the next agent).
    #[must_use]
    pub fn is_pipeline_occupied(&self) -> bool {
        PIPELINE_OCCUPIED_PHASES.contains(self)
    }

    /// Human-readable display label with spaces instead of underscores
    /// (e.g. `"in development"` from [`TicketPhase::InDevelopment`]).
    ///
    /// This is the presentation-oriented counterpart to `AsRefStr::as_ref`
    /// (which returns the machine-oriented `snake_case` form like
    /// `"in_development"`). Use `display_name()` for user-facing UI labels;
    /// keep `as_ref()` for tool output, SQL fragments, and agent-facing text
    /// where agents expect the snake_case phase string.
    #[must_use]
    pub fn display_name(&self) -> String {
        self.as_ref().replace('_', " ")
    }
}

/// Valid phase names, pre-computed once to avoid re-iteration in error paths.
static ALL_TICKET_PHASE_NAMES: LazyLock<String> = LazyLock::new(|| {
    <TicketPhase as strum::IntoEnumIterator>::iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ")
});

impl std::str::FromStr for TicketPhase {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Case-sensitive matching to preserve backward compatibility with
        // the previous `strum::EnumString` derive.
        <TicketPhase as strum::IntoEnumIterator>::iter()
            .find(|p| p.as_ref() == s)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid phase '{s}'. Valid phases: {}",
                    *ALL_TICKET_PHASE_NAMES
                )
            })
    }
}

// Display and AsRefStr are provided by strum derives. FromStr is implemented
// manually above to produce user-friendly error messages.

/// Bundles a SQL mutation statement with its parameters and ticket id.
/// Returned by [`BoardStore::build_transition_sql`]; executed via
/// [`PreparedUpdate::execute_no_cancel`], [`PreparedUpdate::execute_tx`],
/// [`PreparedUpdate::execute_and_cancel`] or
/// [`PreparedUpdate::execute_tx_matched`].
struct PreparedUpdate {
    sql: String,
    params: Vec<db::Value>,
    ticket_id: String,
}

impl PreparedUpdate {
    /// Execute within an existing transaction, verifying that a row was
    /// affected. Does NOT cancel registered agents — the caller manages
    /// transaction lifecycle and should cancel agents before starting the
    /// transaction when needed.
    async fn execute_tx(self, tx: &db::TxGuard<'_>) -> Result<()> {
        let rows = tx.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        Ok(())
    }

    /// Execute the update without cancelling any agent.
    ///
    /// Use this for post-agent operations where the caller knows no agent
    /// is running and the implicit cancellation of
    /// [`execute_and_cancel`](Self::execute_and_cancel) is unnecessary.
    async fn execute_no_cancel(self, conn: &db::Connection) -> Result<()> {
        let rows = conn.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        Ok(())
    }

    /// Execute the update, verify it affected a row, then cancel any agent
    /// registered on this ticket.
    ///
    /// This is a convenience for single-ticket mutations that follow the
    /// pattern: execute → verify → cancel stale agent.
    ///
    /// # When NOT to use
    ///
    /// Do **not** use this helper for operations where cancellation is
    /// unnecessary or has different semantics. Prefer [`execute_no_cancel`](Self::execute_no_cancel)
    /// for simple post-agent updates that do not need stale-agent cancellation.
    /// Additionally, avoid this helper for:
    /// - **`BoardStore::supersede_and_create`** — runs inside a transaction, cancels
    ///   before commit via a different pattern.
    async fn execute_and_cancel(self, conn: &db::Connection) -> Result<()> {
        let rows = conn.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&self.ticket_id);
        Ok(())
    }

    /// Execute within an existing transaction, reporting whether the CAS
    /// guard matched: `Ok(true)` when a row was updated, `Ok(false)` when no
    /// row matched — the ticket is no longer in the expected phase (moved
    /// externally) or does not exist. Follows the claim convention
    /// (guard miss = `Ok(false)`, an expected no-op), unlike the other
    /// executors which treat a no-row match as an error.
    async fn execute_tx_matched(self, tx: &db::TxGuard<'_>) -> Result<bool> {
        let rows = tx.execute(&self.sql, self.params).await?;
        Ok(rows > 0)
    }
}

/// Controls whether the pipeline-occupancy check is enforced when claiming tickets.
///
/// Pipeline-occupied tickets (those in [`PIPELINE_OCCUPIED_PHASES`]) prevent
/// multiple tickets from being worked concurrently in the same workspace.
///
/// - [`Skip`](Self::Skip): claim the next available ticket without checking
///   for pipeline occupants (used by parallel phases like analysis, review, QA).
/// - [`Enforce`](Self::Enforce): only claim if no pipeline-occupied ticket exists
///   in the workspace (used by serial phases like development, diagnostics, sanitation).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PipelineCheck {
    /// Skip pipeline occupancy check — claim the next available ticket.
    Skip,
    /// Only claim if no pipeline-occupied ticket exists in the workspace.
    Enforce,
}

/// Whether to load comments when fetching tickets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum LoadComments {
    /// Load comments alongside the ticket.
    Yes,
    /// Skip loading comments.
    No,
}

impl BoardStore {
    /// Post-open setup: ensure the FTS index exists with the correct tokenizer.
    ///
    /// The index is ALSO declared in the schema catalog
    /// ([`crate::db::migrations`]) so the consolidated schema creates it before
    /// the one-time import (guaranteeing imported tickets are FTS-searchable).
    /// This hook therefore only tokenizer-migrates an existing index (drops
    /// and recreates if the tokenizer changed); it is idempotent and runs from
    /// both [`crate::db::init_all_stores`] (production, on the shared
    /// consolidated connection) and each isolated board store open. The board
    /// schema changes are NOT run here — every schema change is an entry in the
    /// append-only catalog ([`crate::db::migrations`]) applied once by
    /// [`crate::db::open_consolidated_store`].
    pub(crate) async fn after_open(&self) -> anyhow::Result<()> {
        crate::db::ensure_fts_index(
            &self.conn,
            TICKETS_FTS_INDEX_NAME,
            "ngram",
            TICKETS_FTS_INDEX_DDL,
        )
        .await?;
        Ok(())
    }

    /// Shared INSERT logic for [`BoardStore::create_ticket`] and [`BoardStore::supersede_and_create`].
    ///
    /// The `embedding` column is write-once — stored at creation time and later
    /// read by `list_archived_with_embeddings` for vector search. It is not
    /// included in `TICKET_COLUMNS` (SELECT queries) because only the dedicated
    /// `list_archived_with_embeddings` method reads it.
    ///
    /// Computes the timestamp and serializes prerequisites internally. Does NOT
    /// commit the transaction — the caller is responsible for calling
    /// `tx.commit()` after any additional writes.
    async fn insert_ticket_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        params: &TicketParams,
        supersedes: Option<&str>,
    ) -> Result<()> {
        let now = db::now();
        let prereqs_json = serde_json::to_string(&params.prerequisites)?;
        tx.execute(
            "INSERT INTO tickets (id, title, description, phase, workspace_name, \
             created_at, updated_at, prerequisites, supersedes, reporter, embedding, priority) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            db::params![
                ticket_id,
                params.title.as_str(),
                params.description.as_str(),
                params.phase.as_ref(),
                params.workspace_name.as_str(),
                now.as_str(),
                now.as_str(),
                prereqs_json.as_str(),
                supersedes,
                params.reporter.as_str(),
                params.embedding.as_deref(),
                params.priority,
            ],
        )
        .await?;
        Ok(())
    }

    /// Rewire dependents after supersede: tickets whose prerequisites mention
    /// `supersede_id` get updated to point to `new_id`. Queried and updated within
    /// the same transaction — no TOCTOU window between SELECT and UPDATE.
    ///
    /// Uses `json_each()` for exact prerequisite matching (consistent with
    /// [`claim_ticket_in_workspace`](Self::claim_ticket_in_workspace)).
    async fn rewire_dependents_tx(
        tx: &TxGuard<'_>,
        supersede_id: &str,
        new_id: &str,
        workspace_name: &str,
    ) -> Result<()> {
        let dep_rows = tx
            .query(
                "SELECT DISTINCT t.id, t.prerequisites \
                 FROM tickets t, json_each(t.prerequisites) AS je \
                 WHERE je.value = ?1 AND t.workspace_name = ?2",
                db::params![supersede_id, workspace_name],
            )
            .await?;

        for row in &dep_rows {
            let dep_id: String = row.get(0)?;
            let raw: String = row.get(1)?;
            let mut prereqs: Vec<String> = parse_prereqs(&raw)
                .with_context(|| format!("Failed to parse prerequisites for ticket {dep_id}"))?;
            let mut changed = false;
            for p in &mut prereqs {
                if *p == supersede_id {
                    *p = new_id.to_string();
                    changed = true;
                }
            }
            if changed {
                let new_json = serde_json::to_string(&prereqs)?;
                tx.execute(
                    "UPDATE tickets SET prerequisites = ?1, updated_at = ?2 WHERE id = ?3",
                    db::params![new_json, db::now(), dep_id],
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Begin a transaction, generate a ticket ID, and validate prerequisites.
    ///
    /// Performs the shared validation preamble used by both [`BoardStore::create_ticket`]
    /// and [`BoardStore::supersede_and_create`]: starts a transaction, generates a
    /// sequential ticket ID via counter upsert, checks that the new ID
    /// doesn't appear in its own prerequisites, then validates all
    /// prerequisites exist and belong to the same workspace.
    ///
    /// Callers must not call `self.conn` methods until the guard is dropped
    /// or committed — `TxGuard` holds a tokio mutex lock.
    ///
    /// # Correctness (TOCTOU)
    ///
    /// Correctness relies on both the tokio mutex inside `conn` (serializes
    /// Rust-level writes) and the SQLite transaction `tx` (provides
    /// database-level isolation) — no concurrent write can change
    /// prerequisite tickets between validation and the caller's INSERT.
    /// Validation runs inside the transaction via `tx.query()` (which
    /// uses the upstream connection through the MutexGuard, avoiding
    /// mutex deadlock with `self.conn.query()`).
    async fn begin_tx_and_validate_prerequisites(
        &self,
        workspace_name: &str,
        prerequisites: &[String],
    ) -> Result<(TxGuard<'_>, String)> {
        let tx = self.conn.begin_tx().await?;
        let seq: i64 = tx
            .query_row(
                "INSERT INTO ticket_counters (workspace_name, next_id) VALUES (?1, 1) \
                 ON CONFLICT(workspace_name) DO UPDATE SET next_id = ticket_counters.next_id + 1 \
                 RETURNING next_id - 1",
                db::params![workspace_name],
                |row| row.get(0),
            )
            .await?;
        let id = format!("{workspace_name}-{seq}");
        anyhow::ensure!(
            !prerequisites.contains(&id),
            "Ticket cannot depend on itself: {id}"
        );
        // Validate prerequisites using the transaction's query method —
        // tx.query() uses the upstream connection through the MutexGuard
        // so it doesn't deadlock with the mutex held by TxGuard.
        Self::validate_prerequisites(&tx, prerequisites, workspace_name).await?;
        Ok((tx, id))
    }

    /// Create a new ticket at the requested phase. Returns the ticket id.
    pub(crate) async fn create_ticket(&self, params: &TicketParams) -> Result<String> {
        let (tx, id) = self
            .begin_tx_and_validate_prerequisites(&params.workspace_name, &params.prerequisites)
            .await?;

        Self::insert_ticket_tx(&tx, &id, params, None).await?;

        tx.commit().await?;
        Ok(id)
    }

    /// Create a new ticket that supersedes (replaces) an existing ticket.
    ///
    /// Atomically cancels `supersede_id`, creates the new ticket with a
    /// `supersedes` back-link, and rewires any dependent tickets' prerequisites
    /// to point to the new ID. All writes happen in a single transaction
    /// via `begin_tx()` + parameterized queries.
    ///
    /// Before commit, any running agent on the superseded ticket is cancelled.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The superseded ticket does not exist
    /// - The superseded ticket is in a different workspace
    /// - A self-reference is detected (supersede ID in the new ticket's prerequisites)
    /// - Any prerequisite is invalid (doesn't exist or cross-workspace)
    pub(crate) async fn supersede_and_create(
        &self,
        supersede_id: &str,
        params: &TicketParams,
    ) -> Result<String> {
        anyhow::ensure!(
            !params.prerequisites.iter().any(|p| p == supersede_id),
            "Ticket cannot supersede and depend on the same ticket: {supersede_id}"
        );

        let (tx, new_id) = self
            .begin_tx_and_validate_prerequisites(&params.workspace_name, &params.prerequisites)
            .await?;

        // Verify the superseded ticket exists and belongs to the same workspace.
        // This runs INSIDE the transaction (tx.query() uses the upstream
        // connection through the MutexGuard) to eliminate the TOCTOU race
        // between validation and cancellation.
        let rows = tx
            .query(
                "SELECT workspace_name FROM tickets WHERE id = ?1",
                db::params![supersede_id],
            )
            .await?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Superseded ticket not found: {supersede_id}"))?;
        let old_ws: String = row.get(0)?;
        anyhow::ensure!(
            old_ws == params.workspace_name,
            "Superseded ticket {supersede_id} belongs to workspace '{old_ws}', \
             not the current workspace '{}'. \
             Cross-workspace supersede is not allowed.",
            params.workspace_name,
        );

        let now = db::now();
        let cancelled_rows = tx
            .execute(
                "UPDATE tickets SET phase = ?1, updated_at = ?2, \
                 superseded_by = ?4, is_archived = 1, done_at = NULL \
                 WHERE id = ?3",
                db::params![
                    TicketPhase::Cancelled.as_ref(),
                    now,
                    supersede_id,
                    new_id.as_str(),
                ],
            )
            .await?;
        Self::ensure_ticket_found(cancelled_rows, supersede_id)?;

        Self::insert_ticket_tx(&tx, &new_id, params, Some(supersede_id)).await?;

        Self::rewire_dependents_tx(&tx, supersede_id, &new_id, &params.workspace_name).await?;

        // Cancel agents on the superseded ticket BEFORE the transaction commits.
        // If the process crashes between commit and cancellation, the superseded
        // ticket is Cancelled in the database but its agents remain registered and
        // keep running (orphaned agents on a cancelled ticket). Cancelling first
        // flips the trade-off: if the commit subsequently fails, agents were
        // cancelled unnecessarily but will be re-registered on re-dispatch.
        crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(supersede_id);

        // No chronicle push for the cancellation: supersede is only reachable
        // through agent tools (Manager or Maintainer CreateTicketTool) and agent
        // actions are intentionally silent — the GUI board path is the user
        // surface that notifies the Manager.
        tx.commit().await?;

        Ok(new_id)
    }

    /// Build a [`Ticket`] from a row returned by a
    /// [`TICKET_COLUMNS`] SELECT, optionally including its comments.
    async fn ticket_from_row(&self, row: &db::Row, load_comments: LoadComments) -> Result<Ticket> {
        let id: String = row.get(COL_TICKET_ID)?;
        let comments = if load_comments == LoadComments::Yes {
            self.get_comments(&id).await?
        } else {
            Vec::new()
        };
        let prerequisites_raw: String = row.get(COL_TICKET_PREREQUISITES)?;
        let prerequisites = parse_prereqs(&prerequisites_raw)
            .with_context(|| format!("Failed to parse prerequisites for ticket {id}"))?;
        Ok(Ticket {
            id,
            title: row.get(COL_TICKET_TITLE)?,
            description: row.get(COL_TICKET_DESCRIPTION)?,
            phase: row
                .get::<String>(COL_TICKET_PHASE)?
                .parse::<TicketPhase>()?,
            workspace_name: row.get(COL_TICKET_WORKSPACE_NAME)?,
            created_at: row.get(COL_TICKET_CREATED_AT)?,
            updated_at: row.get(COL_TICKET_UPDATED_AT)?,
            comments,
            prerequisites,
            supersedes: row.get(COL_TICKET_SUPERSEDES)?,
            superseded_by: row.get(COL_TICKET_SUPERSEDED_BY)?,
            commit_hash: row.get(COL_TICKET_COMMIT_HASH)?,
            lines_added: row.get(COL_TICKET_LINES_ADDED)?,
            lines_removed: row.get(COL_TICKET_LINES_REMOVED)?,
            reporter: row.get::<String>(COL_TICKET_REPORTER)?,
            is_archived: row.get::<bool>(COL_TICKET_IS_ARCHIVED)?,
            priority: row.get::<i64>(COL_TICKET_PRIORITY)?,
            reviewed_head: row.get(COL_TICKET_REVIEWED_HEAD)?,
            reviewed_tree: row.get(COL_TICKET_REVIEWED_TREE)?,
            done_at: row.get(COL_TICKET_DONE_AT)?,
            bounce_count: row.get(COL_TICKET_BOUNCE_COUNT)?,
        })
    }

    /// Grace window for Backlog→Analysis claims: tickets younger than this are
    /// not claimed into Analysis. Gives the Manager ~5s after create_ticket to
    /// move the ticket straight to Planning/ReadyForDevelopment — claiming it
    /// into Analysis within the window would spawn analysts that immediately
    /// get cancelled (plus a spurious phase-change notification).
    pub(crate) const BACKLOG_CLAIM_GRACE: Duration = Duration::seconds(5);

    /// Claim a ticket scoped to a single workspace and transition it to
    /// `target_phase`. Always filters by `workspace_name` so only tickets from
    /// that workspace are eligible.
    ///
    /// Only tickets currently in `current_phase` are eligible for claiming.
    /// The WHERE clause includes `t1.phase = ?` bound to `current_phase`,
    /// providing CAS-style atomicity for phase transitions — if no ticket
    /// matches the current phase, the claim returns `None`.
    ///
    /// When `claim_grace` is `Some(duration)`, tickets created within that
    /// duration before the claim are excluded from the candidate set via a SQL
    /// `created_at <=` cutoff — they stay in `current_phase` until the window
    /// elapses. The Backlog→Analysis claim passes [`BACKLOG_CLAIM_GRACE`] so
    /// freshly created tickets are not immediately picked up; all other claims
    /// pass `None` and are unaffected.
    ///
    /// When `pipeline_check` is [`PipelineCheck::Enforce`], the claim is rejected
    /// (returns `None`) if any pipeline-occupied ticket exists in the same workspace. The
    /// occupancy check is part of the same atomic SQL UPDATE statement (no
    /// separate SELECT + UPDATE window). Pipeline-occupied phases are defined
    /// in [`PIPELINE_OCCUPIED_PHASES`].
    ///
    /// When `pipeline_check` is [`PipelineCheck::Skip`], the claim uses a
    /// simple LIMIT 1 subquery with no pipeline gating. This is used for phases
    /// that should not be blocked by in-flight pipeline tickets (e.g., analysis,
    /// review, and QA).
    ///
    /// The subquery orders by `priority ASC, created_at ASC` so that tickets
    /// with lower priority (higher urgency) are claimed first, then the oldest
    /// ticket (earliest created_at) is claimed first.
    ///
    /// Note: the claim SELECT is kept to the tickets table, and the agents
    /// roster is a separate logical table checked by the caller (via the
    /// phase-job + launched-roster guard). The claim source phases
    /// (Backlog, ReadyForDevelopment) have no running agent.
    pub(crate) async fn claim_ticket_in_workspace(
        &self,
        current_phase: TicketPhase,
        target_phase: TicketPhase,
        workspace_name: &str,
        pipeline_check: PipelineCheck,
        claim_grace: Option<Duration>,
    ) -> Result<Option<Ticket>> {
        let now = db::now();

        // Filter that excludes tickets with unmet prerequisites.
        let prereq_filter = format!(
            "AND NOT EXISTS ( \
               SELECT 1 FROM json_each(t1.prerequisites) AS je \
               JOIN tickets t_pre ON t_pre.id = je.value \
               WHERE t_pre.phase NOT IN ({}) \
             )",
            phase_list_sql_fragment(UNBLOCKING_PHASES),
        );

        let pipeline_occupied_clause = if pipeline_check == PipelineCheck::Enforce {
            let occupied_sql = phase_list_sql_fragment(PIPELINE_OCCUPIED_PHASES);
            format!(
                "AND NOT EXISTS (SELECT 1 FROM tickets t2 \
                 WHERE t2.workspace_name = t1.workspace_name \
                 AND t2.phase IN ({occupied_sql}) \
                 AND t2.id != t1.id) "
            )
        } else {
            String::new()
        };

        // Candidate-set age cutoff: excludes tickets created within the grace
        // window so fresh tickets stay in `current_phase` a bit longer.
        let grace_clause = if claim_grace.is_some() {
            "AND t1.created_at <= ?5 "
        } else {
            ""
        };

        let sql = format!(
            "UPDATE tickets SET phase = ?1, updated_at = ?2 \
             WHERE id = (SELECT t1.id FROM tickets t1 \
             WHERE t1.phase = ?3 AND t1.workspace_name = ?4 \
             AND t1.is_archived = 0 \
             {grace_clause}{pipeline_occupied_clause}{prereq_filter} \
             ORDER BY t1.priority ASC, t1.created_at ASC LIMIT 1) \
             RETURNING {TICKET_COLUMNS}"
        );

        let mut params: Vec<Value> = vec![
            Value::from(target_phase.as_ref()),
            Value::from(now),
            Value::from(current_phase.as_ref()),
            Value::from(workspace_name),
        ];
        if let Some(grace) = claim_grace {
            params.push(Value::from((Utc::now() - grace).to_rfc3339()));
        }

        let rows = self.conn.query(&sql, params).await?;
        match rows.into_iter().next() {
            Some(row) => Ok(Some(self.ticket_from_row(&row, LoadComments::Yes).await?)),
            None => Ok(None),
        }
    }

    /// Select tickets matching a SQL suffix (everything after `FROM tickets`),
    /// parsing each row via [`ticket_from_row`](Self::ticket_from_row).
    ///
    /// This is the shared building block for all `SELECT {TICKET_COLUMNS}` queries.
    /// Accepts the full suffix — typically starting with `WHERE` and optionally
    /// including `ORDER BY`, `LIMIT`, etc. — and forwards `params` directly to
    /// the underlying query so callers can use `db::params![]` without conversions.
    pub(crate) async fn select_tickets(
        &self,
        suffix: &str,
        params: impl IntoParams + Send + 'static,
        load_comments: LoadComments,
    ) -> Result<Vec<Ticket>> {
        let sql = format!("SELECT {TICKET_COLUMNS} FROM tickets {suffix}");
        let rows = self.conn.query(&sql, params).await?;
        let mut tickets = Vec::with_capacity(rows.len());
        for row in rows {
            tickets.push(self.ticket_from_row(&row, load_comments).await?);
        }
        Ok(tickets)
    }

    /// Get a ticket by id, loading its comments.
    pub async fn get_ticket(&self, ticket_id: &str) -> Result<Option<Ticket>> {
        Ok(self
            .select_tickets("WHERE id = ?1", db::params![ticket_id], LoadComments::Yes)
            .await?
            .into_iter()
            .next())
    }

    /// Fetch multiple tickets by their IDs.
    ///
    /// Returns an empty vec if `ids` is empty (no SQL round-trip).
    /// Tickets are returned in **arbitrary order** — callers that need to
    /// preserve input ordering must re-sort after receiving the result.
    pub(crate) async fn get_tickets_by_ids(
        &self,
        ids: &[String],
        load_comments: LoadComments,
    ) -> Result<Vec<Ticket>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let (suffix, params) = Self::in_clause_for_ids(ids);
        self.select_tickets(&suffix, params, load_comments).await
    }

    /// Build a `WHERE id IN (?, ?, ...)` suffix and parameter vector from
    /// ticket IDs.
    ///
    /// Callers must ensure `ids` is non-empty — the resulting SQL is invalid
    /// (syntax error from SQLite) when the list is empty.
    fn in_clause_for_ids(ids: &[String]) -> (String, Vec<Value>) {
        let suffix = format!("WHERE id IN ({})", db::sql_in_placeholders(ids.len()));
        let params: Vec<Value> = ids.iter().map(|id| Value::Text(id.clone())).collect();
        (suffix, params)
    }

    /// Get a ticket's phase by id — lightweight, no comments loaded.
    pub async fn get_ticket_phase(&self, ticket_id: &str) -> Result<Option<TicketPhase>> {
        self.conn
            .query_optional(
                "SELECT phase FROM tickets WHERE id = ?1",
                db::params![ticket_id],
                |row| {
                    let phase: String = row.get(0)?;
                    phase.parse()
                },
            )
            .await
    }

    /// Get a ticket's priority by id — lightweight, no comments loaded.
    ///
    /// Used by the priority-inheritance path in `CreateTicketTool::execute` to
    /// read the superseded ticket's priority without loading comments. Priority
    /// is immutable after ticket creation, so this single-column read outside
    /// the supersede transaction is safe — there is no TOCTOU race with the
    /// supersede transaction's own SELECT for existence/workspace/phase checks.
    pub(crate) async fn get_ticket_priority(&self, ticket_id: &str) -> Result<Option<i64>> {
        let sql = "SELECT priority FROM tickets WHERE id = ?1";
        self.conn
            .query_optional(sql, db::params![ticket_id], |row| row.get::<i64>(0))
            .await
    }

    /// Build a [`PreparedUpdate`] for an `UPDATE tickets` statement, appending
    /// `updated_at = ?` and `WHERE id = ?` as the last two parameters.
    ///
    /// Callers provide the SET-clause-specific columns (without `updated_at` or
    /// `WHERE`) together with their parameter values.  The helper appends the
    /// current timestamp and the ticket id as the final parameters, keeping the
    /// parameter ordering consistent across all `UPDATE tickets` producers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let prep = Self::build_ticket_update_with_updated_at(
    ///     "priority = ?",
    ///     vec![Value::from(2)],
    ///     "ticket-456",
    /// );
    /// // SQL:  "UPDATE tickets SET priority = ?, updated_at = ? WHERE id = ?"
    /// // params: [2, now, ticket-456]
    /// ```
    fn build_ticket_update_with_updated_at(
        set_clause: &str,
        set_params: Vec<db::Value>,
        ticket_id: &str,
    ) -> PreparedUpdate {
        let now = db::now();
        let sql = format!("UPDATE tickets SET {set_clause}, updated_at = ? WHERE id = ?");
        let mut params = set_params;
        params.push(Value::from(now));
        params.push(Value::from(ticket_id));
        PreparedUpdate {
            sql,
            params,
            ticket_id: ticket_id.to_string(),
        }
    }

    /// Build the SQL, params, and action description for a ticket phase
    /// transition. Shared by [`transition_to`](Self::transition_to) and
    /// [`transition_to_tx`](Self::transition_to_tx).
    ///
    /// Note: this does **not** use [`Self::build_ticket_update_with_updated_at`]
    /// because it has an extra SET column (`done_at = CASE ...`) and an
    /// additional WHERE condition (`AND (?4 IS NULL OR phase = ?4)`) that
    /// don't fit the helper's fixed pattern.
    ///
    /// `done_at` is set to `?2` (now) when the target is `done` — overwriting
    /// on re-completion — and cleared when the ticket leaves `done`, so the
    /// column holds a timestamp iff the ticket is currently in the Done phase.
    /// Later non-transition activity (comments, archive) never touches it.
    fn build_transition_sql(
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
    ) -> PreparedUpdate {
        let now = db::now();
        let guard: Option<&str> = expected_phase.as_ref().map(TicketPhase::as_ref);
        let sql = "UPDATE tickets SET phase = ?1, updated_at = ?2, \
                    done_at = CASE WHEN ?1 = 'done' THEN ?2 \
                                   WHEN phase = 'done' THEN NULL \
                                   ELSE done_at END \
                    WHERE id = ?3 AND (?4 IS NULL OR phase = ?4)";
        let params: Vec<db::Value> = vec![
            Value::from(target_phase.as_ref()),
            Value::from(now),
            Value::from(ticket_id),
            Value::from(guard),
        ];
        PreparedUpdate {
            sql: sql.to_string(),
            params,
            ticket_id: ticket_id.to_string(),
        }
    }

    /// Update ticket phase, optionally guarded by an expected phase for CAS-style
    /// atomicity. Always cancels running agents.
    ///
    /// # Errors
    ///
    /// Returns an error when the UPDATE matched 0 rows or a database error occurs.
    pub async fn transition_to(
        &self,
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
    ) -> Result<()> {
        let prepared = Self::build_transition_sql(ticket_id, expected_phase, target_phase);
        prepared.execute_and_cancel(&self.conn).await?;
        // A terminal transition (Done/Cancelled/Failed) completes the ticket's
        // phase jobs (idempotent; no-op when the ticket has none) so a manual
        // Done/Cancelled does not leave a lingering 'launched' phase job. The
        // pipeline path completes the phase jobs at its own Done handoff
        // functions, so this only fires for external/manual callers.
        // Guard on the session store being initialized: this is the MANUAL
        // path, and some board-only contexts (tests) never open the session DB.
        if target_phase.is_terminal()
            && let Some(session) = crate::session::SESSIONS.get()
        {
            let _ = crate::jobs::complete_ticket_phase_jobs(&session.conn, ticket_id).await;
        }
        Ok(())
    }

    /// Transactional variant of [`transition_to`](Self::transition_to) —
    /// uses an existing transaction instead of `self.conn.execute()`.
    /// Does NOT cancel registered agents — the caller is responsible for
    /// cancelling agents **before** beginning the transaction (or at least
    /// before `tx.commit()`) to avoid orphaned agents on crash.
    ///
    /// # Return value
    ///
    /// Returns `Ok(true)` when the CAS guard matched and the row was
    /// updated; `Ok(false)` when no row matched — the ticket is no longer in
    /// the expected phase (moved externally while the stage finished) or does
    /// not exist. The guard miss follows the board layer's claim convention
    /// (expected no-op, not an error), unlike
    /// [`transition_to`](Self::transition_to) whose callers perform
    /// user-initiated actions on a ticket that must exist.
    pub(crate) async fn transition_to_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
    ) -> Result<bool> {
        let prepared = Self::build_transition_sql(ticket_id, expected_phase, target_phase);
        prepared.execute_tx_matched(tx).await
    }

    /// Verify that a mutation query affected at least one row, returning an
    /// error with a descriptive message if the ticket was not found.
    fn ensure_ticket_found(rows: u64, ticket_id: &str) -> Result<()> {
        anyhow::ensure!(rows > 0, "Ticket {ticket_id} not found");
        Ok(())
    }

    /// Record commit metadata on a ticket using an existing transaction.
    /// Does NOT commit or rollback the transaction; the caller controls that.
    pub(crate) async fn set_commit_info_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        hash: &str,
        lines_added: i64,
        lines_removed: i64,
    ) -> Result<()> {
        debug_assert!(
            lines_added >= 0,
            "lines_added must be non-negative: {lines_added}"
        );
        debug_assert!(
            lines_removed >= 0,
            "lines_removed must be non-negative: {lines_removed}"
        );
        // Build the SQL and params for setting commit info.
        let prepared = Self::build_ticket_update_with_updated_at(
            "commit_hash = ?, lines_added = ?, lines_removed = ?",
            vec![
                Value::from(hash),
                Value::from(lines_added),
                Value::from(lines_removed),
            ],
            ticket_id,
        );
        prepared.execute_tx(tx).await
    }

    /// Record the reviewed content base (HEAD + index tree) on a ticket.
    ///
    /// Set after a completed reviewer round so later rounds can skip the
    /// reviewer pass only when their content is identical to this base.
    /// `None` values clear the base (ticket becomes never-reviewed).
    pub(crate) async fn set_reviewed_base(
        &self,
        ticket_id: &str,
        head: Option<&str>,
        tree: Option<&str>,
    ) -> Result<()> {
        let prepared = Self::build_ticket_update_with_updated_at(
            "reviewed_head = ?, reviewed_tree = ?",
            vec![Value::from(head), Value::from(tree)],
            ticket_id,
        );
        prepared.execute_no_cancel(&self.conn).await
    }

    /// Increment the ticket's bounce counter inside an existing transaction.
    ///
    /// Called atomically with the bounce-back transition (review/QA bounce or
    /// engineer hard failure) so the counter can never drift from the
    /// transitions that produce it.
    pub(crate) async fn increment_bounce_count_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
    ) -> Result<i64> {
        let rows = tx
            .query(
                "UPDATE tickets SET bounce_count = bounce_count + 1, updated_at = ?1 \
                 WHERE id = ?2 RETURNING bounce_count",
                db::params![db::now(), ticket_id],
            )
            .await
            .map_err(anyhow::Error::from)?;
        match rows.into_iter().next() {
            Some(row) => row.get::<i64>(0).map_err(anyhow::Error::from),
            None => Err(anyhow::anyhow!(
                "ticket {ticket_id} not found — bounce counter not incremented"
            )),
        }
    }

    /// Shared implementation for checking if a workspace has active tickets.
    ///
    /// Returns `true` if any ticket in the workspace has a pipeline-occupied
    /// phase ([`PIPELINE_OCCUPIED_PHASES`]), or a
    /// [`ReadyForDevelopment`](TicketPhase::ReadyForDevelopment) ticket,
    /// optionally excluding a specific ticket ID.
    ///
    /// [`has_active_tickets_excluding`] delegates to this helper.
    ///
    /// # Parameters
    ///
    /// * `workspace_name` — The workspace to check.
    /// * `exclude_ticket_id` — When `Some(id)`, that ticket is excluded from
    ///   the check. When `None`, no exclusion is applied (the SQL clause
    ///   `(?2 IS NULL OR id != ?2)` short-circuits to `TRUE`).
    ///
    /// Excludes archived tickets — the only phases that ever get archived are
    /// `Done` and `Cancelled`, neither of which appears in
    /// `PIPELINE_OCCUPIED_PHASES`, so this is a defensive consistency measure.
    ///
    /// # Parameter binding note
    ///
    /// The SQL always binds three positional parameters:
    /// - `?1`: `workspace_name`
    /// - `?2`: `exclude_ticket_id` (may be `None`)
    /// - `?3`: ReadyForDevelopment phase value
    async fn has_active_tickets_internal(
        &self,
        workspace_name: &str,
        exclude_ticket_id: Option<&str>,
    ) -> Result<bool> {
        let occupied_sql = phase_list_sql_fragment(PIPELINE_OCCUPIED_PHASES);
        let sql = format!(
            "SELECT 1 FROM tickets WHERE \
             (phase IN ({occupied_sql}) OR phase = ?3) \
             AND workspace_name = ?1 AND is_archived = 0 \
             AND (?2 IS NULL OR id != ?2) LIMIT 1",
        );
        let rfd = TicketPhase::ReadyForDevelopment.as_ref();
        let rows = self
            .conn
            .query(&sql, db::params![workspace_name, exclude_ticket_id, rfd])
            .await?;
        Ok(!rows.is_empty())
    }

    /// Check if the workspace has any active tickets other than the excluded one.
    ///
    /// "Active" means a ticket whose phase is either a pipeline-occupied phase
    /// (`PIPELINE_OCCUPIED_PHASES`) or [`TicketPhase::ReadyForDevelopment`]
    /// (unstarted backlog tickets are considered active to suppress Done
    /// notifications until the pipeline is fully drained).
    ///
    /// Delegates to `has_active_tickets_internal`.
    ///
    /// Non-active phases (not matched by the query): `Done`, `Cancelled`,
    /// `Failed`, `Backlog`, `Analysis`, `Planning`.
    ///
    /// # Race condition note
    ///
    /// When multiple occupied tickets in the same workspace are finalized
    /// concurrently (each via [`tokio::spawn`] in the poller), both may see
    /// each other as active and both buffer their Done transitions. In this
    /// scenario all tickets are already in Done in the database — the only
    /// consequence is that Done notifications are delayed until the next
    /// [`crate::agent::message_router::MessageKind::UserMessage`] drains the buffer. This is an accepted trade-off:
    /// the race window is small and the buffer always drains eventually.
    pub async fn has_active_tickets_excluding(
        &self,
        workspace_name: &str,
        exclude_ticket_id: &str,
    ) -> Result<bool> {
        self.has_active_tickets_internal(workspace_name, Some(exclude_ticket_id))
            .await
    }

    /// Add a comment to a ticket (append-only).
    ///
    /// After persisting the comment, routes it to any running agents assigned
    /// to this ticket via the message router. A comment delivered to a running
    /// agent is a SOFT DEFERRED message consumed at the start of the agent's
    /// next tool round — it never cancels or aborts the agent. If no agent is
    /// registered (the agent finished before the comment arrived), the comment
    /// is surfaced by the next agent that works on the ticket: a fresh dispatch
    /// on a new session re-renders the ticket comment block, and a resumed
    /// deterministic stage session (engineer/sanitation) injects outstanding
    /// comments at the empty-message resume point.
    pub async fn add_comment(&self, ticket_id: &str, role: &str, content: &str) -> Result<()> {
        crate::db::with_tx(&self.conn, ticket_id, "add comment", async |tx| {
            Self::add_comment_tx(tx, ticket_id, role, content).await
        })
        .await?;

        // Route the comment to any running agents assigned to this ticket.
        self.route_comment_to_agents(ticket_id, role, content).await;

        Ok(())
    }

    /// Route a newly-persisted comment to any running agents assigned to the ticket.
    ///
    /// Looks up the ticket's currently-running agents (`launched` roster rows
    /// across the ticket's jobs — the replacement for the historical
    /// `tickets.assigned_to`). If agents are registered in the message router,
    /// the comment is delivered to each one. This is best-effort — failures
    /// are logged but not propagated.
    async fn route_comment_to_agents(&self, ticket_id: &str, role: &str, content: &str) {
        // Best-effort: if the sessions store isn't initialized (e.g. a
        // board-only test), there are no active agents to route to.
        let Some(sessions) = crate::session::SESSIONS.get() else {
            return;
        };
        // Fetch the ticket's workspace_name.
        let ticket = match self.get_ticket(ticket_id).await {
            Ok(Some(t)) => t,
            Ok(None) => {
                warn!(ticket = %ticket_id, "Comment routing: ticket not found");
                return;
            }
            Err(e) => {
                warn!(ticket = %ticket_id, error = %e, "Comment routing: failed to fetch ticket");
                return;
            }
        };

        let active =
            match crate::jobs::list_running_agents_for_ticket(&sessions.conn, ticket_id).await {
                Ok(agents) if !agents.is_empty() => agents,
                Ok(_) => return, // No running agents
                Err(e) => {
                    warn!(
                        ticket = %ticket_id,
                        error = %e,
                        "Comment routing: failed to read running agents",
                    );
                    return;
                }
            };

        for agent_id in active {
            if agent_id.is_empty() {
                continue;
            }

            // Use the commenter's role. If it doesn't parse as a standard Role
            // (e.g. "engineer_1" from parallel-agent verdict comments), fall
            // back to Manager. This fallback is safe: pipeline agents receive
            // comments via try_route() → direct inbox delivery, NOT via the
            // consumer loop. AgentJob.role is only used for response delivery
            // in the consumer loop path, which pipeline agents never enter.
            let commenter_role = role.parse::<crate::Role>().unwrap_or(crate::Role::Manager);

            let job = crate::agent::message_router::AgentJob {
                content: content.to_string(),
                workspace_name: ticket.workspace_name.clone(),
                user_name: role.to_string(),
                channel: String::new(),
                kind: crate::agent::message_router::MessageKind::TicketComment,
                role: commenter_role,
                reply_target: None,
                pending_job_id: None,
            };

            if crate::agent::message_router::try_route(&agent_id, job) {
                debug!(
                    ticket = %ticket_id,
                    agent = %agent_id,
                    "Routed comment to running agent",
                );
            }
        }
    }

    /// Transactional variant of [`add_comment`](Self::add_comment) —
    /// uses an existing transaction instead of opening its own.
    /// Does NOT commit or rollback; the caller controls outer transaction lifecycle.
    ///
    /// Inserts the comment record AND updates the ticket's `updated_at` timestamp.
    ///
    /// NOTE: Unlike [`add_comment`](Self::add_comment), this method does NOT route
    /// the comment to running agents via the message router. All current callers
    /// (verdict recording, system comments, failure reports) are post-agent phases
    /// where no running agent exists to receive the comment. If adding a new caller
    /// that runs mid-execution, use [`add_comment`](Self::add_comment) instead, or
    /// call [`route_comment_to_agents`](Self::route_comment_to_agents) manually
    /// after the transaction commits.
    pub(crate) async fn add_comment_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let comment_id = crate::generate_id();
        let now = db::now();
        tx.execute(
            "INSERT INTO ticket_comments (id, ticket_id, role, content, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            db::params![comment_id, ticket_id, role, content, now.as_str()],
        )
        .await?;
        tx.execute(
            "UPDATE tickets SET updated_at = ?1 WHERE id = ?2",
            db::params![now.as_str(), ticket_id],
        )
        .await?;
        Ok(())
    }

    /// Get all comments for a ticket, ordered by creation time.
    pub async fn get_comments(&self, ticket_id: &str) -> Result<Vec<TicketComment>> {
        let sql = format!(
            "SELECT {COMMENT_COLUMNS} FROM ticket_comments WHERE ticket_id = ?1 ORDER BY created_at ASC"
        );
        let rows = self.conn.query(&sql, db::params![ticket_id]).await?;
        let mut comments = Vec::new();
        for row in rows {
            comments.push(TicketComment {
                role: row.get(COL_COMMENT_ROLE)?,
                content: row.get(COL_COMMENT_CONTENT)?,
                created_at: row.get(COL_COMMENT_CREATED_AT)?,
            });
        }
        Ok(comments)
    }

    /// Validate prerequisites for a new ticket being created.
    ///
    /// Checks that every prerequisite ticket exists and belongs to the same
    /// workspace. Self-reference is checked separately by the caller (before
    /// this function is called, using the real ID generated within the transaction).
    ///
    /// At creation time, transitive cycles cannot exist because no existing
    /// ticket depends on the new ticket yet. Redundant prerequisites (e.g.,
    /// A and B where B already depends on A) are allowed — they do not form
    /// a cycle.
    async fn validate_prerequisites(
        tx: &TxGuard<'_>,
        prerequisite_ids: &[String],
        workspace_name: &str,
    ) -> Result<()> {
        // Guard against empty list — SQLite rejects WHERE id IN ().
        if prerequisite_ids.is_empty() {
            return Ok(());
        }

        // Batch query: fetch id + workspace_name for all prerequisites in one
        // round trip. Uses tx.query() — the transaction's query method operates
        // on the upstream connection through the MutexGuard, avoiding mutex
        // deadlock with conn.query().
        let (suffix, params) = Self::in_clause_for_ids(prerequisite_ids);
        let sql = format!("SELECT id, workspace_name FROM tickets {suffix}");
        let rows = tx.query(&sql, params).await?;

        // Build a lookup map for O(1) prerequisite resolution.
        let mut found: HashMap<String, String> = HashMap::new();
        for row in rows {
            let id: String = row.get(0)?;
            let ws_name: String = row.get(1)?;
            found.insert(id, ws_name);
        }

        for pid in prerequisite_ids {
            let ws_name = found
                .get(pid)
                .ok_or_else(|| anyhow::anyhow!("Prerequisite ticket not found: {pid}"))?;
            anyhow::ensure!(
                ws_name == workspace_name,
                "Prerequisite {pid} belongs to workspace '{ws_name}', \
                 not the ticket's workspace '{workspace_name}'. \
                 Cross-workspace prerequisites are not allowed.",
            );
        }

        Ok(())
    }

    /// List all tickets, optionally filtered by workspace and/or phase.
    /// Used by the dashboard to show tickets across all workspaces. The
    /// `is_archived = 0` gate means a ticket appears here iff it is live,
    /// which also backs the CDC removal decision.
    pub async fn list_all_tickets(
        &self,
        workspace_name: Option<&str>,
        phase_filter: Option<TicketPhase>,
    ) -> Result<Vec<Ticket>> {
        let phase_str: Option<&str> = phase_filter.as_ref().map(TicketPhase::as_ref);
        self.select_tickets(
            "WHERE (?1 IS NULL OR workspace_name = ?1) \
             AND (?2 IS NULL OR phase = ?2) \
             AND is_archived = 0 \
             ORDER BY priority ASC, created_at DESC",
            db::params![workspace_name, phase_str],
            LoadComments::No,
        )
        .await
    }

    /// Count how many tickets have the given phase, optionally filtered by workspace.
    ///
    /// Excludes archived tickets to stay consistent with [`list_all_tickets`](Self::list_all_tickets)
    /// and most other read paths in this module. Currently unused for `Done` or `Cancelled`
    /// (the only phases that ever get archived), so this is a defensive consistency fix —
    /// callers that pass a terminal phase will not see archived tickets in the count.
    pub async fn count_by_phase(
        &self,
        phase: TicketPhase,
        workspace_name: Option<&str>,
    ) -> Result<i64> {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM tickets \
                 WHERE phase = ?1 \
                   AND (?2 IS NULL OR workspace_name = ?2) \
                   AND is_archived = 0",
                db::params![phase.as_ref(), workspace_name],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Archive a single ticket by ID.
    ///
    /// Sets `is_archived = 1`. Returns an error if the ticket does not exist.
    ///
    /// **Ordering constraint:** The caller must transition the ticket to a
    /// terminal state (`done` or `cancelled`) *before* calling this method.
    pub async fn set_archived(&self, ticket_id: &str) -> Result<()> {
        let prepared =
            Self::build_ticket_update_with_updated_at("is_archived = 1", vec![], ticket_id);
        prepared.execute_and_cancel(&self.conn).await
    }

    /// Move all non-archived ReadyForDevelopment tickets in the given workspace
    /// to Planning.
    ///
    /// Used by the circuit breaker to drain sibling ReadyForDevelopment tickets
    /// when a ticket in the same workspace fails.
    ///
    /// Uses a single atomic UPDATE so there is no TOCTOU window between reading
    /// current ReadyForDevelopment tickets and updating them. Per-sibling
    /// notifications are intentionally suppressed — each sibling will discover
    /// its new phase on the next poll cycle via the standard poll loop.
    ///
    /// Returns the number of tickets moved.
    pub(crate) async fn drain_ready_for_development_to_planning(
        &self,
        workspace_name: &str,
    ) -> Result<u64> {
        let now = db::now();
        let updated = self
            .conn
            .execute(
                "UPDATE tickets SET phase = ?1, updated_at = ?2 \
                 WHERE phase = ?3 AND workspace_name = ?4 AND is_archived = 0",
                db::params![
                    TicketPhase::Planning.as_ref(),
                    now,
                    TicketPhase::ReadyForDevelopment.as_ref(),
                    workspace_name,
                ],
            )
            .await?;
        Ok(updated)
    }

    pub async fn archive_stale_cancelled(&self, hours: i64) -> Result<u64> {
        let now = db::now();
        let cutoff = (Utc::now() - Duration::hours(hours)).to_rfc3339();
        let updated = self
            .conn
            .execute(
                "UPDATE tickets SET is_archived = 1, updated_at = ?1 \
                 WHERE phase = ?2 AND updated_at < ?3 \
                 AND is_archived = 0",
                db::params![now, TicketPhase::Cancelled.as_ref(), cutoff],
            )
            .await
            .context("Failed to archive stale cancelled tickets")?;
        Ok(updated)
    }

    pub async fn archive_all_done_and_cancelled(
        &self,
        workspace_name: Option<&str>,
    ) -> Result<u64> {
        let now = db::now();
        let sql = format!(
            "UPDATE tickets SET is_archived = 1, updated_at = ?1 \
             WHERE phase IN ({}) AND is_archived = 0 \
             AND (?2 IS NULL OR workspace_name = ?2)",
            phase_list_sql_fragment(UNBLOCKING_PHASES),
        );
        let updated = self
            .conn
            .execute(&sql, db::params![now, workspace_name])
            .await
            .context("Failed to archive done/cancelled tickets")?;
        Ok(updated)
    }

    // ── Board display ordering (shared with the GUI) ─────────────────────
    //
    // The GUI board and the Telegram `/board` command must show tickets in
    // exactly the same order. These helpers are the single source of truth —
    // the GUI column rendering and the Telegram listing both use them.

    /// Partition tickets into the three kanban columns, in the same order the
    /// GUI board displays them: completed ([`TicketPhase::is_unblocking`]),
    /// pipeline ([`TicketPhase::is_pipeline_occupied`] plus
    /// `ReadyForDevelopment`), pending (everything else — the safe fallback
    /// for unclassified phases). Archived tickets are excluded.
    ///
    /// Non-completed tickets sort by priority ASC (0 = highest) then
    /// created_at ASC; completed tickets sort Done-first by exact done
    /// timestamp DESC (created_at fallback), then Cancelled by created_at
    /// DESC.
    #[must_use]
    pub fn partition_board_tickets(
        tickets: &[Ticket],
    ) -> (Vec<&Ticket>, Vec<&Ticket>, Vec<&Ticket>) {
        let mut pending = Vec::new();
        let mut pipeline = Vec::new();
        let mut completed = Vec::new();

        for ticket in tickets {
            if ticket.is_archived {
                continue; // hidden from board
            }
            if ticket.phase.is_unblocking() {
                completed.push(ticket);
            } else if ticket.phase.is_pipeline_occupied()
                || ticket.phase == TicketPhase::ReadyForDevelopment
            {
                pipeline.push(ticket);
            } else {
                // Unknown future phases silently default to pending — the
                // safe bucket for unclassified phases.
                pending.push(ticket);
            }
        }

        // Sort: pending and pipeline by priority (ASC), then oldest-first (ASC);
        // completed: Done tickets newest-done_first (DESC), then Cancelled
        // newest-first (DESC) below them.
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
        completed.sort_by(|a, b| {
            let (a_done, b_done) = (a.phase == TicketPhase::Done, b.phase == TicketPhase::Done);
            match (a_done, b_done) {
                // Done first, newest completion on top (created_at fallback
                // for Done tickets with no done_at, e.g. test-created ones).
                (true, true) => Self::board_done_sort_key(b).cmp(Self::board_done_sort_key(a)),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => b.created_at.cmp(&a.created_at),
            }
        });

        (pending, pipeline, completed)
    }

    /// The four board sections in display order: In Progress (pipeline minus
    /// ReadyForDevelopment), Ready, Pending, Completed. Shared by the GUI
    /// column rendering and the Telegram `/board` listing so the two surfaces
    /// can never diverge. Empty sections are included (callers skip them).
    #[must_use]
    pub fn board_sections(tickets: &[Ticket]) -> [Vec<&Ticket>; 4] {
        let (pending, pipeline, completed) = Self::partition_board_tickets(tickets);
        let in_progress = pipeline
            .iter()
            .filter(|t| t.phase != TicketPhase::ReadyForDevelopment)
            .copied()
            .collect();
        let ready = pipeline
            .iter()
            .filter(|t| t.phase == TicketPhase::ReadyForDevelopment)
            .copied()
            .collect();
        [in_progress, ready, pending, completed]
    }

    /// Flatten [`Self::board_sections`] into a single display-order list —
    /// the exact order the GUI board shows them. Used by the Telegram
    /// `/board` command so its listing can never diverge from the GUI.
    #[must_use]
    pub fn board_display_order(tickets: &[Ticket]) -> Vec<&Ticket> {
        Self::board_sections(tickets)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Completion ordering key for a completed-column ticket: its exact done
    /// timestamp, falling back to creation time when `done_at` is absent.
    fn board_done_sort_key(ticket: &Ticket) -> &str {
        ticket.done_at.as_deref().unwrap_or(&ticket.created_at)
    }

    // ── Ticket FTS/embedding search methods ───────────────────────────────
    //
    // The archived-search methods contain the FTS and embedding SQL used by
    // [`SearchArchivedTicketsTool`](crate::tools::search_archived_tickets);
    // [`search_by_fts`] backs the GUI sidebar search. The board owns the
    // schema (`ngram` tokenizer, FTS index name, blob format) and the tool
    // layer owns the hybrid RRF merge logic.

    /// Search archived tickets by FTS keyword match, scoped to a workspace.
    ///
    /// Sanitizes the input query (strips non-alphanumeric characters) before
    /// matching against the `ngram`-tokenized FTS index on `title`.
    ///
    /// Returns up to `limit` `(id, fts_score)` pairs, highest score first.
    /// On SQL error (e.g. corrupt FTS index), logs a warning and returns an
    /// empty vec — the caller may fall through to vector search as a graceful
    /// degradation strategy.
    pub async fn search_archived_by_fts(
        &self,
        query: &str,
        limit: usize,
        workspace_name: &str,
    ) -> Result<Vec<(String, f64)>> {
        let sanitized = crate::db::sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        // Param order mirrors search_by_fts: ?1 = workspace, ?2 = query.
        let sql = format!(
            "SELECT t.id, fts_score(t.title, ?2) AS score \
             FROM tickets t \
             WHERE t.is_archived = 1 \
               AND t.workspace_name = ?1 \
               AND t.title MATCH ?2 \
             ORDER BY score DESC LIMIT {limit}"
        );
        match self
            .conn
            .query_map(
                &sql,
                db::params![workspace_name, sanitized.clone()],
                |row| {
                    let id: String = row.get(0)?;
                    let score: f64 = row.get(1)?;
                    Ok::<_, anyhow::Error>((id, score))
                },
            )
            .await
        {
            Ok(items) => Ok(items
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()?),
            Err(e) => {
                tracing::warn!(
                    query = %sanitized,
                    error = %e,
                    "FTS search for archived tickets failed"
                );
                Ok(Vec::new())
            }
        }
    }

    /// Search tickets (both active and archived) by FTS keyword match, scoped
    /// to an optional workspace.
    ///
    /// Sanitizes the input query (strips non-alphanumeric characters) before
    /// matching against the `ngram`-tokenized FTS index on `title`.
    ///
    /// Returns full [`Ticket`] objects (without comments) ordered by FTS
    /// relevance (`fts_score DESC`), up to `limit` results.
    /// On SQL error, logs a warning and returns an empty vec.
    pub async fn search_by_fts(
        &self,
        query: &str,
        limit: usize,
        workspace_name: Option<&str>,
    ) -> Result<Vec<Ticket>> {
        let sanitized = crate::db::sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        // Use an explicit `FROM tickets t` alias so `fts_score(t.title, ?2)`
        // resolves correctly — the `select_tickets` helper does not support
        // table aliases in FTS scoring expressions.
        let sql = format!(
            "SELECT {TICKET_COLUMNS} \
             FROM tickets t \
             WHERE (?1 IS NULL OR t.workspace_name = ?1) \
               AND t.title MATCH ?2 \
             ORDER BY fts_score(t.title, ?2) DESC \
             LIMIT {limit}"
        );
        match self
            .conn
            .query(&sql, db::params![workspace_name, sanitized.clone()])
            .await
        {
            Ok(rows) => {
                let mut tickets = Vec::with_capacity(rows.len());
                for row in rows {
                    match self.ticket_from_row(&row, LoadComments::No).await {
                        Ok(t) => tickets.push(t),
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "Failed to parse ticket row from FTS search"
                            );
                        }
                    }
                }
                Ok(tickets)
            }
            Err(e) => {
                tracing::warn!(
                    query = %sanitized,
                    error = %e,
                    "FTS search failed"
                );
                Ok(Vec::new())
            }
        }
    }

    /// List archived tickets with non-NULL embeddings, deserialized, scoped to
    /// a workspace.
    ///
    /// Returns `(id, embedding)` pairs for all archived tickets that have
    /// a stored embedding blob. Embeddings are deserialized from the
    /// on-disk `[u8]` byte layout (4-byte little-endian `f32`) into
    /// `Vec<f32>` via [`crate::vector::bytes_to_vec`].
    ///
    /// This returns ALL archived tickets with embeddings — there is no
    /// LIMIT because the caller (the tool layer) needs all candidates for
    /// cosine-similarity ranking, and the archive size is bounded in practice
    /// by the total ticket volume of the installation.
    pub async fn list_archived_with_embeddings(
        &self,
        workspace_name: &str,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let rows = self
            .conn
            .query(
                "SELECT id, embedding FROM tickets \
                 WHERE is_archived = 1 AND workspace_name = ?1 AND embedding IS NOT NULL",
                db::params![workspace_name],
            )
            .await?;

        let mut candidates: Vec<(String, Vec<f32>)> = Vec::new();
        for row in &rows {
            let id: String = row.get(0)?;
            let stored: Vec<u8> = row.get(1)?;
            let emb = crate::vector::bytes_to_vec(&stored);
            candidates.push((id, emb));
        }
        Ok(candidates)
    }
}
