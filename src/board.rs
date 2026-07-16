//! Ticket/board system — Turso-backed task management.

use crate::role::SYSTEM_ROLE;
use crate::turso::{self, IntoParams, TxGuard, Value};
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
    db_name = "board",
    schema = SCHEMA,
    post_open = after_open,
    expect = "BOARD not initialized — call init_global() first",
}

/// Background task: auto-archive cancelled tickets older than 1 hour.
///
/// Runs every 5 minutes, respects the global shutdown token via
/// [`crate::shutdown::sleep_or_shutdown`] (same pattern as
/// [`crate::maintainer::run_maintainer_loop`]).
/// Logs per-ticket failures and continues — a ticket that was un-cancelled
/// between the SELECT and UPDATE is harmlessly skipped.
pub async fn run_archive_cancelled_loop() {
    let interval = std::time::Duration::from_mins(5);

    loop {
        if !crate::shutdown::sleep_or_shutdown(interval).await {
            break;
        }

        let Some(board) = BOARD.get() else {
            warn!("Archive cancelled loop: board not initialized");
            continue;
        };

        match board.archive_stale_cancelled(1).await {
            Ok(n) if n > 0 => info!(count = n, "Archived stale cancelled tickets"),
            Ok(_) => debug!("Archive cancelled loop: no stale tickets"),
            Err(e) => warn!(error = %e, "Archive cancelled loop failed"),
        }
    }
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS tickets (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    phase          TEXT NOT NULL DEFAULT 'backlog',
    assigned_to     TEXT,
    workspace_name  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    prerequisites   TEXT NOT NULL DEFAULT '[]',
    supersedes      TEXT,
    superseded_by   TEXT,
    commit_hash     TEXT,
    lines_added     INTEGER,
    lines_removed   INTEGER,
    reporter        TEXT NOT NULL DEFAULT '',
    is_archived     INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,
    pipeline_reservation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ticket_comments (
    id          TEXT PRIMARY KEY,
    ticket_id   TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);
CREATE INDEX IF NOT EXISTS idx_ticket_comments_ticket_id ON ticket_comments(ticket_id);
CREATE TABLE IF NOT EXISTS ticket_counters (
    workspace_name TEXT PRIMARY KEY,
    next_id        INTEGER NOT NULL DEFAULT 1
);";

const TICKETS_FTS_INDEX_NAME: &str = "idx_tickets_title_fts";
const TICKETS_FTS_INDEX_DDL: &str = "\
CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets \
USING fts (title) WITH (tokenizer = 'ngram')";

// Column definitions for ticket SELECT/RETURNING queries.
crate::columns! {
    TICKET_COLUMNS [TICKET] {
        ID                     => "id",
        TITLE                  => "title",
        DESCRIPTION            => "description",
        PHASE                  => "phase",
        ASSIGNED_TO            => "assigned_to",
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
        PIPELINE_RESERVATION   => "pipeline_reservation",
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
/// Note: [`BoardStore::reset_inflight_tickets`] (via [`BoardStore::RESET_TRANSITIONS`]) only resets a subset
/// of these (InDevelopment, InDiagnostics, InSanitation, InReview, InQa) plus Analysis — see its
/// docs for rationale. The remaining four phases (DiagnosticsDone, SanitationPassed, Reviewed, QaPassed) are transitory
/// handoff states intentionally excluded from reset. The `tests::test_pipeline_blockers_coverage`
/// test enforces that every non-transitory pipeline blocker has a corresponding reset transition.
const PIPELINE_BLOCKING_PHASES: &[TicketPhase] = &[
    TicketPhase::InDevelopment,
    TicketPhase::InDiagnostics,
    TicketPhase::DiagnosticsDone,
    TicketPhase::InReview,
    TicketPhase::Reviewed,
    TicketPhase::InQa,
    TicketPhase::QaPassed,
    TicketPhase::InSanitation,
    TicketPhase::SanitationPassed,
];

/// Pipeline-blocking phases that are transitory handoff states — no agent is
/// mid-execution in these phases, so they don't need a reset transition. The
/// poller picks them up within seconds.
///
/// This is a subset of [`PIPELINE_BLOCKING_PHASES`]. The relationship is
/// mechanically verified by `tests::test_pipeline_blockers_coverage`.
#[cfg(test)]
const TRANSITORY_HANDOFF_PHASES: &[TicketPhase] = &[
    TicketPhase::DiagnosticsDone,
    TicketPhase::SanitationPassed,
    TicketPhase::Reviewed,
    TicketPhase::QaPassed,
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
            format!("{}…", &raw[..raw.floor_char_boundary(200)])
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
    pub assigned_to: Option<String>,
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
    /// "test"). Empty string for pre-migration tickets.
    pub reporter: String,
    /// Whether this ticket has been archived (hidden from normal listings).
    pub is_archived: bool,
    /// Whether this ticket holds a pipeline reservation (bounced back from
    /// review/QA/diagnostics and awaiting rework). When set, the ticket gets
    /// priority over other ReadyForDevelopment tickets during claim.
    pub pipeline_reservation: bool,
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
    /// - `assigned_to`
    /// - `commit_hash`
    /// - `lines_added` / `lines_removed`
    ///
    /// ## Output size
    ///
    /// The returned string can be arbitrarily large (unbounded descriptions
    /// and comments). Callers that need truncation should apply their own
    /// limits (see `GetTicketTool::format_output` for an example that
    /// disables the default 5 KB truncation).
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
        let mut out = format!(
            "Ticket: {id}\n\
             Title: {title}\n\
             Description: {description}\n\
             Phase: {phase}\n\
             Reporter: {reporter}\n\
             Workspace: {workspace}\n\
             Created: {created}\n\
             Updated: {updated}\n",
            id = self.id,
            title = self.title,
            description = self.description,
            phase = self.phase,
            reporter = self.reporter,
            workspace = self.workspace_name,
            created = self.created_at,
            updated = self.updated_at,
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
        out.push_str(&self.format_comments());
        out
    }

    /// Format comments as a `"Comments:"` block suitable for [`crate::tools::ticket::GetTicketTool`].
    ///
    /// Returns a string starting with `"Comments:"` followed by one line per
    /// comment in the format `"\n  [{role}] ({timestamp}): {content}"`, or
    /// `"Comments:\n  (no comments)"` if the comment list is empty.
    ///
    /// Timestamps are truncated to seconds (`[..19]` of the RFC 3339 string),
    /// with a defensive `min()` guard against abnormally short strings.
    #[must_use]
    fn format_comments(&self) -> String {
        let mut s = String::from("Comments:");
        if self.comments.is_empty() {
            s.push_str("\n  (no comments)");
        } else {
            for c in &self.comments {
                let end = 19.min(c.created_at.len());
                let ts = &c.created_at[..end];
                let _ = write!(s, "\n  [{}] ({}): {}", c.role, ts, c.content);
            }
        }
        s
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
    DiagnosticsDone,
    InSanitation,
    SanitationPassed,
    InReview,
    Reviewed,
    InQa,
    QaPassed,
    Done,
    Cancelled,
    Failed,
}

impl TicketPhase {
    /// Returns `true` for transitory handoff phases — pipeline-blocking
    /// phases where no agent is mid-execution.
    ///
    /// Delegates to `TRANSITORY_HANDOFF_PHASES` so the transitory handoff set can never
    /// accidentally diverge from the definition used in coverage tests.
    #[cfg(test)]
    #[must_use]
    fn is_transitory_handoff(self) -> bool {
        TRANSITORY_HANDOFF_PHASES.contains(&self)
    }

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

    /// Returns `true` if the ticket is in a pipeline-blocking phase.
    ///
    /// Tickets in these phases are actively being worked on by agents
    /// (development, diagnostics, review, QA). Automated tools (create,
    /// update, add_comment) should refuse to modify them to prevent
    /// race conditions with running agents.
    #[must_use]
    pub fn is_pipeline_blocking(&self) -> bool {
        PIPELINE_BLOCKING_PHASES.contains(self)
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
/// [`PreparedUpdate::execute_no_cancel`], [`PreparedUpdate::execute_tx`] or
/// [`PreparedUpdate::execute_and_cancel`].
struct PreparedUpdate {
    sql: String,
    params: Vec<turso::Value>,
    ticket_id: String,
}

impl PreparedUpdate {
    /// Execute within an existing transaction, verifying that a row was
    /// affected. Does NOT cancel registered agents — the caller manages
    /// transaction lifecycle and should cancel agents before starting the
    /// transaction when needed.
    async fn execute_tx(self, tx: &turso::TxGuard<'_>) -> Result<()> {
        let rows = tx.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        Ok(())
    }

    /// Execute the update without cancelling any agent.
    ///
    /// Use this for post-agent operations where the caller knows no agent
    /// is running and the implicit cancellation of
    /// [`execute_and_cancel`](Self::execute_and_cancel) is unnecessary.
    async fn execute_no_cancel(self, conn: &turso::Connection) -> Result<()> {
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
    /// - **`BoardStore::claim_diagnostics`** — returns `Result<bool>`, only cancels on success.
    /// - **`BoardStore::supersede_and_create`** — runs inside a transaction, cancels
    ///   before commit via a different pattern.
    /// - **`BoardStore::claim_sanitation`** — returns `Result<bool>`, does NOT cancel
    ///   (QaPassed has no running agent).
    async fn execute_and_cancel(self, conn: &turso::Connection) -> Result<()> {
        let rows = conn.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(&self.ticket_id);
        Ok(())
    }
}

/// A single reset transition: when a ticket in `from` phase is found on startup,
/// it is rolled back to `to` phase. If `pipeline_reservation` is true, the ticket
/// gets `pipeline_reservation = 1` so it is claimed before any fresh ticket in the
/// same phase (preserving rework priority across restarts).
#[derive(Debug, Clone, Copy)]
struct ResetTransition {
    from: TicketPhase,
    to: TicketPhase,
    /// Whether to set `pipeline_reservation = 1` on the reset ticket.
    ///
    /// When `true`: the reset ticket gets priority re-dispatch — it will be
    /// claimed before any fresh ticket in the same phase.
    /// When `false`: normal queue order.
    ///
    /// See [`BoardStore::RESET_TRANSITIONS`] for the rationale behind each entry.
    pipeline_reservation: bool,
}

/// Controls whether the pipeline-occupancy check is enforced when claiming tickets.
///
/// Pipeline-blocking tickets (those in [`PIPELINE_BLOCKING_PHASES`]) prevent
/// multiple tickets from being worked concurrently in the same workspace.
///
/// - [`Skip`](Self::Skip): claim the next available ticket without checking
///   for pipeline blockers (used by parallel phases like analysis, review, QA).
/// - [`Enforce`](Self::Enforce): only claim if no pipeline-blocking ticket exists
///   in the workspace (used by serial phases like development, diagnostics, sanitation).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PipelineCheck {
    /// Skip pipeline occupancy check — claim the next available ticket.
    Skip,
    /// Only claim if no pipeline-blocking ticket exists in the workspace.
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

/// One-time schema migrations for backward compatibility.
///
/// These run only for databases created with older versions.
/// Can be safely removed once all active databases have been migrated.
async fn run_schema_migrations(conn: &turso::Connection) -> anyhow::Result<()> {
    // ── schema migration: rename `status` column to `phase` ───────────
    let version_rows = conn
        .query("PRAGMA user_version", ())
        .await
        .context("Failed to read PRAGMA user_version for schema migration")?;
    let current_version: i64 = version_rows
        .first()
        .and_then(|row| row.get::<i64>(0).ok())
        .unwrap_or(0);

    if current_version < 1 {
        // Check whether the old `status` column still exists (migration needed).
        let table_info = conn
            .query("PRAGMA table_info('tickets')", ())
            .await
            .context("Failed to read PRAGMA table_info for tickets table")?;

        let has_status_column = table_info
            .iter()
            .any(|row| row.get::<String>(1).ok().as_deref() == Some("status"));

        if has_status_column {
            info!("Schema migration: renaming tickets.status to tickets.phase");
            conn.execute("ALTER TABLE tickets RENAME COLUMN status TO phase", ())
                .await
                .context(
                    "Schema migration failed: unable to rename tickets.status to tickets.phase",
                )?;
            // ALTER TABLE auto-commits in SQLite/libSQL, so the rename is
            // already persisted. Continue to update the schema version.
        }

        // PRAGMA user_version is NOT transaction-atomic in SQLite — set it
        // after the ALTER TABLE (which has already auto-committed) to avoid
        // a version-schema mismatch if a crash occurs between the two.
        conn.execute("PRAGMA user_version = 1", ())
            .await
            .context("Schema migration failed: unable to set PRAGMA user_version to 1")?;

        // Persist immediately via WAL checkpoint.
        conn.checkpoint().await.context(
            "Schema migration failed: unable to checkpoint after renaming tickets.status",
        )?;

        if has_status_column {
            info!("Schema migration complete: tickets.status renamed to tickets.phase (version 1)");
        }
    }
    Ok(())
}

impl BoardStore {
    /// Post-open setup: run schema migrations, then create FTS index.
    async fn after_open(&self) -> anyhow::Result<()> {
        run_schema_migrations(&self.conn).await?;

        // ── FTS index setup (must run after migration) ────────────────────
        crate::turso::ensure_fts_index(
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
        let now = turso::now();
        let prereqs_json = serde_json::to_string(&params.prerequisites)?;
        tx.execute(
            "INSERT INTO tickets (id, title, description, phase, workspace_name, \
             created_at, updated_at, prerequisites, supersedes, reporter, embedding) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            turso::params![
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
                turso::params![supersede_id, workspace_name],
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
                    turso::params![new_json, turso::now(), dep_id],
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
                turso::params![workspace_name],
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
                "SELECT workspace_name, phase FROM tickets WHERE id = ?1",
                turso::params![supersede_id],
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
        let phase_str: String = row.get(1)?;
        let old_phase: TicketPhase = phase_str.parse()?;

        let now = turso::now();
        let cancelled_rows = tx
            .execute(
                "UPDATE tickets SET phase = ?1, updated_at = ?2, assigned_to = NULL, \
                 superseded_by = ?4, is_archived = 1 WHERE id = ?3",
                turso::params![
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
        crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(supersede_id);

        tx.commit().await?;

        crate::ticket_buffer::push(
            &params.workspace_name,
            supersede_id,
            old_phase,
            TicketPhase::Cancelled,
        );

        Ok(new_id)
    }

    /// Build a [`Ticket`] from a row returned by a
    /// [`TICKET_COLUMNS`] SELECT, optionally including its comments.
    async fn ticket_from_row(
        &self,
        row: &turso::Row,
        load_comments: LoadComments,
    ) -> Result<Ticket> {
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
            assigned_to: row.get(COL_TICKET_ASSIGNED_TO)?,
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
            pipeline_reservation: row.get::<bool>(COL_TICKET_PIPELINE_RESERVATION)?,
        })
    }

    /// Claim a ticket scoped to a single workspace and transition it to
    /// `target_phase`. Always filters by `workspace_name` so only tickets from
    /// that workspace are eligible.
    ///
    /// Only tickets currently in `current_phase` are eligible for claiming.
    /// The WHERE clause includes `t1.phase = ?` bound to `current_phase`,
    /// providing CAS-style atomicity for phase transitions — if no ticket
    /// matches the current phase, the claim returns `None`.
    ///
    /// When `pipeline_check` is [`PipelineCheck::Enforce`], the claim is rejected
    /// (returns `None`) if any pipeline-blocking ticket exists in the same workspace. The
    /// occupancy check is part of the same atomic SQL UPDATE statement (no
    /// separate SELECT + UPDATE window). Pipeline-blocking phases are defined
    /// in [`PIPELINE_BLOCKING_PHASES`].
    ///
    /// Note that a reserved ReadyForDevelopment ticket (one with
    /// `pipeline_reservation = 1`) is **not** treated as a pipeline blocker for
    /// the purpose of this claim — [`has_pipeline_blocker_for_workspace`] (a
    /// test-only query) considers such tickets blockers, but the claim subquery
    /// orders by `pipeline_reservation DESC` and clears reservation on claim,
    /// so a reserved ticket at ReadyForDevelopment will be claimed before any
    /// other ticket at the same phase — no pipeline blocking needed.
    ///
    /// When `pipeline_check` is [`PipelineCheck::Skip`], the claim uses a
    /// simple LIMIT 1 subquery with no pipeline gating. This is used for phases
    /// that should not be blocked by in-flight pipeline tickets (e.g., analysis,
    /// review, and QA).
    ///
    /// The subquery orders by `pipeline_reservation DESC, created_at ASC` so that
    /// tickets bounced back from review/QA/diagnostics (reservation = 1) are claimed
    /// before fresh tickets at the same phase. Among tickets with equal reservation,
    /// the oldest ticket (earliest created_at) is claimed first.
    ///
    /// Note: the UPDATE sets `assigned_to = NULL` and `pipeline_reservation = 0` —
    /// this intentionally drops the previous claimant and clears any pipeline
    /// reservation so the cleared slot is available for other tickets.
    /// Callers that require agent-level assignment
    /// (single-owner dispatches like the Engineer) should call
    /// [`set_assigned_to`](Self::set_assigned_to) after claiming. Parallel-agent
    /// dispatches (analysts, verifiers) intentionally leave `assigned_to` NULL.
    pub(crate) async fn claim_ticket_in_workspace(
        &self,
        current_phase: TicketPhase,
        target_phase: TicketPhase,
        workspace_name: &str,
        pipeline_check: PipelineCheck,
    ) -> Result<Option<Ticket>> {
        let now = turso::now();

        // Filter that excludes tickets with unmet prerequisites.
        let prereq_filter = format!(
            "AND NOT EXISTS ( \
               SELECT 1 FROM json_each(t1.prerequisites) AS je \
               JOIN tickets t_pre ON t_pre.id = je.value \
               WHERE t_pre.phase NOT IN ({}) \
             )",
            phase_list_sql_fragment(UNBLOCKING_PHASES),
        );

        let pipeline_blocker_clause = if pipeline_check == PipelineCheck::Enforce {
            let blocker_sql = phase_list_sql_fragment(PIPELINE_BLOCKING_PHASES);
            format!(
                "AND NOT EXISTS (SELECT 1 FROM tickets t2 \
                 WHERE t2.workspace_name = t1.workspace_name \
                 AND t2.phase IN ({blocker_sql}) \
                 AND t2.id != t1.id) "
            )
        } else {
            String::new()
        };

        let sql = format!(
            "UPDATE tickets SET phase = ?1, assigned_to = NULL, updated_at = ?2, \
             pipeline_reservation = 0 \
             WHERE id = (SELECT t1.id FROM tickets t1 \
             WHERE t1.phase = ?3 AND t1.assigned_to IS NULL AND t1.workspace_name = ?4 \
             AND t1.is_archived = 0 \
             {pipeline_blocker_clause}{prereq_filter} \
             ORDER BY t1.pipeline_reservation DESC, t1.created_at ASC LIMIT 1) \
             RETURNING {TICKET_COLUMNS}"
        );

        let rows = self
            .conn
            .query(
                &sql,
                turso::params![
                    target_phase.as_ref(),
                    now,
                    current_phase.as_ref(),
                    workspace_name,
                ],
            )
            .await?;
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
    /// the underlying query so callers can use `turso::params![]` without conversions.
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
            .select_tickets(
                "WHERE id = ?1",
                turso::params![ticket_id],
                LoadComments::Yes,
            )
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
        let suffix = format!("WHERE id IN ({})", turso::sql_in_placeholders(ids.len()));
        let params: Vec<Value> = ids.iter().map(|id| Value::Text(id.clone())).collect();
        (suffix, params)
    }

    /// Get a ticket's phase by id — lightweight, no comments loaded.
    pub async fn get_ticket_phase(&self, ticket_id: &str) -> Result<Option<TicketPhase>> {
        let sql = "SELECT phase FROM tickets WHERE id = ?1";
        let rows = self.conn.query(sql, turso::params![ticket_id]).await?;
        match rows.into_iter().next() {
            Some(row) => {
                let phase: String = row.get(0)?;
                Ok(Some(phase.parse()?))
            }
            None => Ok(None),
        }
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
    ///     "assigned_to = ?",
    ///     vec![Value::from("user-123")],
    ///     "ticket-456",
    /// );
    /// // SQL:  "UPDATE tickets SET assigned_to = ?, updated_at = ? WHERE id = ?"
    /// // params: [user-123, now, ticket-456]
    /// ```
    fn build_ticket_update_with_updated_at(
        set_clause: &str,
        set_params: Vec<turso::Value>,
        ticket_id: &str,
    ) -> PreparedUpdate {
        let now = turso::now();
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
    /// because it has extra SET columns (`assigned_to = NULL`,
    /// `pipeline_reservation = COALESCE(?5, pipeline_reservation)`) and an
    /// additional WHERE condition (`AND (?4 IS NULL OR phase = ?4)`) that
    /// don't fit the helper's fixed pattern.
    fn build_transition_sql(
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        reservation: Option<bool>,
    ) -> PreparedUpdate {
        let now = turso::now();
        let guard: Option<&str> = expected_phase.as_ref().map(TicketPhase::as_ref);
        let sql = "UPDATE tickets SET phase = ?1, assigned_to = NULL, updated_at = ?2, \
                    pipeline_reservation = COALESCE(?5, pipeline_reservation) \
                    WHERE id = ?3 AND (?4 IS NULL OR phase = ?4)";
        let params: Vec<turso::Value> = vec![
            Value::from(target_phase.as_ref()),
            Value::from(now),
            Value::from(ticket_id),
            Value::from(guard),
            Value::from(reservation),
        ];
        PreparedUpdate {
            sql: sql.to_string(),
            params,
            ticket_id: ticket_id.to_string(),
        }
    }

    /// Update ticket phase, optionally guarded by an expected phase for CAS-style
    /// atomicity. Always clears `assigned_to` and cancels running agents.
    ///
    /// # Note on [`pipeline_reservation`](Ticket::pipeline_reservation)
    ///
    /// When `reservation` is `None`, the column is left untouched so bounce-back
    /// transitions can set it atomically, and manual transitions leave stale
    /// reservations inert (claim/blocker queries filter by phase). When
    /// `Some(value)`, it's set in the same UPDATE to avoid a race on crash/restart
    /// recovery or rework priority.
    ///
    /// # Errors
    ///
    /// Returns an error when the UPDATE matched 0 rows or a database error occurs.
    pub async fn transition_to(
        &self,
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        reservation: Option<bool>,
    ) -> Result<()> {
        let prepared =
            Self::build_transition_sql(ticket_id, expected_phase, target_phase, reservation);
        prepared.execute_and_cancel(&self.conn).await
    }

    /// Transactional variant of [`transition_to`](Self::transition_to) —
    /// uses an existing transaction instead of `self.conn.execute()`.
    /// Does NOT cancel registered agents — the caller is responsible for
    /// cancelling agents **before** beginning the transaction (or at least
    /// before `tx.commit()`) to avoid orphaned agents on crash.
    pub(crate) async fn transition_to_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        reservation: Option<bool>,
    ) -> Result<()> {
        let prepared =
            Self::build_transition_sql(ticket_id, expected_phase, target_phase, reservation);
        prepared.execute_tx(tx).await
    }

    /// Verify that a mutation query affected at least one row, returning an
    /// error with a descriptive message if the ticket was not found.
    fn ensure_ticket_found(rows: u64, ticket_id: &str) -> Result<()> {
        anyhow::ensure!(rows > 0, "Ticket {ticket_id} not found");
        Ok(())
    }

    /// Set or clear the assignee for a ticket (does not change phase).
    ///
    /// When `assigned_to` is `Some(value)`, sets the `assigned_to` column to that
    /// value. When `None`, clears the assignee (sets `assigned_to = NULL`).
    ///
    /// Cancels any running agents registered on this ticket as a safety-in-depth
    /// measure against stale assignments. For set operations, callers typically
    /// set the assignee before spawning an agent, so the cancel is normally a
    /// no-op.
    ///
    /// ## When to use [`clear_assigned_to_no_cancel`](Self::clear_assigned_to_no_cancel)
    ///
    /// If you are clearing the assignee (`assigned_to = None`) and know that no
    /// agent is running (e.g., post-agent cleanup), prefer
    /// [`clear_assigned_to_no_cancel`](Self::clear_assigned_to_no_cancel) to avoid
    /// the misleading cancellation side-effect.
    pub async fn set_assigned_to(&self, ticket_id: &str, assigned_to: Option<&str>) -> Result<()> {
        let prepared = Self::build_ticket_update_with_updated_at(
            "assigned_to = ?",
            vec![Value::from(assigned_to)],
            ticket_id,
        );
        prepared.execute_and_cancel(&self.conn).await
    }

    /// Clear the assignee for a ticket **without** cancelling any running agent.
    ///
    /// This is the safe choice for post-agent operations (e.g., after an agent
    /// has already finished) where no agent is running and the cancellation
    /// side-effect of [`set_assigned_to`](Self::set_assigned_to) would be
    /// misleading.
    ///
    /// To set a new assignee (with stale-agent cancellation), use
    /// [`set_assigned_to`](Self::set_assigned_to).
    pub async fn clear_assigned_to_no_cancel(&self, ticket_id: &str) -> Result<()> {
        let prepared = Self::build_ticket_update_with_updated_at(
            "assigned_to = ?",
            vec![Value::from(None::<&str>)],
            ticket_id,
        );
        prepared.execute_no_cancel(&self.conn).await
    }

    /// Transactional variant of [`set_assigned_to`](Self::set_assigned_to) —
    /// uses an existing transaction instead of opening its own.
    /// Does NOT cancel registered agents — the caller is responsible
    /// for cancelling stale agents **before** beginning the transaction
    /// when a cancel is needed (e.g., via `AGENT_REGISTRY.cancel_by_ticket_id`).
    /// This is safe for post-agent operations (e.g., clearing assignment
    /// after an agent has already finished) where no cancel is needed.
    pub(crate) async fn set_assigned_to_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        assigned_to: Option<&str>,
    ) -> Result<()> {
        let prepared = Self::build_ticket_update_with_updated_at(
            "assigned_to = ?",
            vec![Value::from(assigned_to)],
            ticket_id,
        );
        prepared.execute_tx(tx).await
    }

    /// Atomically claim a ticket for diagnostics execution.
    ///
    /// Sets `assigned_to` to the caller-provided value, only when the ticket
    /// is unassigned AND still in [`TicketPhase::InDiagnostics`] — a single
    /// atomic SQL guard that prevents the TOCTOU race between the poll
    /// listing pre-filter and the subsequent claim. The assignee set and phase
    /// check are fused into one UPDATE; callers do not need a separate
    /// `set_assigned_to` after claiming.
    ///
    /// Returns `Ok(true)` if a row was updated (claim succeeded), `Ok(false)`
    /// if no row matched (already claimed by another dispatch or ticket moved
    /// out of [`TicketPhase::InDiagnostics`]). On a successful claim, cancels
    /// any agent registered on this ticket as a safety-in-depth measure against
    /// stale dispatches.
    pub async fn claim_diagnostics(&self, ticket_id: &str, assigned_to: &str) -> Result<bool> {
        let now = turso::now();
        let rows = self
            .conn
            .execute(
                "UPDATE tickets \
                 SET assigned_to = ?1, updated_at = ?2 \
                 WHERE id = ?3 \
                 AND assigned_to IS NULL \
                 AND phase = ?4 \
                 AND is_archived = 0",
                turso::params![
                    assigned_to,
                    now,
                    ticket_id,
                    TicketPhase::InDiagnostics.as_ref()
                ],
            )
            .await?;

        if rows > 0 {
            crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(ticket_id);
        }

        Ok(rows > 0)
    }

    /// Claim a QaPassed ticket for sanitation processing.
    ///
    /// Atomically transitions the ticket from [`TicketPhase::QaPassed`] to
    /// [`TicketPhase::InSanitation`], sets `assigned_to` to the caller-provided
    /// value, and enforces the per-workspace serialization invariant: only one
    /// ticket at a time may be in [`TicketPhase::InSanitation`] or
    /// [`TicketPhase::SanitationPassed`].
    ///
    /// Returns `Ok(true)` if the claim succeeded, `Ok(false)` if:
    /// - The ticket is no longer in QaPassed (already claimed by another handler), or
    /// - Another ticket is already in the sanitation pipeline for this workspace.
    ///
    /// Unlike [`transition_to`](Self::transition_to), this method does NOT
    /// cancel registered agents — QaPassed is a transitory handoff phase
    /// with no running agent, so cancellation is unnecessary.
    pub async fn claim_sanitation(&self, ticket_id: &str, assigned_to: &str) -> Result<bool> {
        let now = turso::now();
        let blocker =
            phase_list_sql_fragment(&[TicketPhase::InSanitation, TicketPhase::SanitationPassed]);
        let sql = format!(
            "UPDATE tickets SET phase = ?1, assigned_to = ?2, updated_at = ?3 \
             WHERE id = ?4 AND phase = ?5 AND is_archived = 0 \
             AND NOT EXISTS (SELECT 1 FROM tickets t2 \
               WHERE t2.workspace_name = \
                 (SELECT workspace_name FROM tickets WHERE id = ?4) \
               AND t2.id != ?4 \
               AND t2.phase IN ({blocker}))"
        );
        let rows = self
            .conn
            .execute(
                &sql,
                turso::params![
                    TicketPhase::InSanitation.as_ref(),
                    assigned_to,
                    now,
                    ticket_id,
                    TicketPhase::QaPassed.as_ref(),
                ],
            )
            .await?;
        Ok(rows > 0)
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

    /// Finalize a ticket with commit info, comment, and transition to Done
    /// within a single transaction.
    ///
    /// This encapsulates the triple-write pattern
    /// (`set_commit_info_tx` + `add_comment_tx` + `transition_to_tx`)
    /// shared by production code and tests. The caller must already have
    /// a running transaction and is responsible for committing or rolling it
    /// back after this method returns.
    ///
    /// # Parameters
    ///
    /// * `tx` — active transaction
    /// * `ticket_id` — the ticket to finalize
    /// * `hash` — git commit hash to record
    /// * `lines_added` — lines added in the commit
    /// * `lines_removed` — lines removed in the commit
    /// * `comment` — comment body to add
    /// * `source` — the ticket's expected current phase (e.g. `QaPassed`)
    pub(crate) async fn finalize_done_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        hash: &str,
        lines_added: i64,
        lines_removed: i64,
        comment: &str,
        source: TicketPhase,
    ) -> Result<()> {
        Self::set_commit_info_tx(tx, ticket_id, hash, lines_added, lines_removed).await?;
        Self::add_comment_tx(tx, ticket_id, SYSTEM_ROLE, comment).await?;
        Self::transition_to_tx(tx, ticket_id, Some(source), TicketPhase::Done, None).await?;
        Ok(())
    }

    /// Transition pairs for crash/restart recovery (extracted so tests can verify
    /// coverage against [`PIPELINE_BLOCKING_PHASES`] without duplicating the pairs).
    ///
    /// Each entry maps a phase where an agent may have crashed mid-work back to the
    /// phase the ticket should resume in. Must be kept in sync with
    /// [`PIPELINE_BLOCKING_PHASES`] — see `tests::test_pipeline_blockers_coverage`.
    ///
    /// Asymmetry: `Analysis → Backlog` is included (backlog analysts may crash mid-analysis),
    /// but `Analysis` is intentionally NOT in [`PIPELINE_BLOCKING_PHASES`] (it's a pre-flight
    /// phase, not a pipeline blocker).
    ///
    /// Transitory handoff phases (DiagnosticsDone, SanitationPassed, Reviewed, QaPassed) are pipeline
    /// blocking but don't need a reset entry — the poller picks them up within seconds
    /// of restart, so no agent session is mid-execution in those states.
    ///
    /// `pipeline_reservation` choice per entry:
    /// - `true`: expensive production-side phases (development, diagnostics, sanitation)
    ///   — losing queue position wastes significant work, so reset tickets get priority.
    /// - `false`: lighter inspection phases (analysis, review, QA) — re-queuing is cheap,
    ///   so normal queue order is fine.
    const RESET_TRANSITIONS: &[ResetTransition] = &[
        ResetTransition {
            from: TicketPhase::InDevelopment,
            to: TicketPhase::ReadyForDevelopment,
            pipeline_reservation: true,
        },
        ResetTransition {
            from: TicketPhase::InDiagnostics,
            to: TicketPhase::ReadyForDevelopment,
            pipeline_reservation: true,
        },
        ResetTransition {
            from: TicketPhase::InSanitation,
            to: TicketPhase::QaPassed,
            pipeline_reservation: true,
            // Note: pipeline_reservation = true on InSanitation → QaPassed is inert —
            // QaPassed uses list-based dispatch (for_each_ticket_in_phase), not the claim
            // loop where pipeline_reservation provides ordering. Set `true` to match
            // the production-side convention (sanitation is substantive work, like
            // development and diagnostics); the flag is harmless for list-based dispatch.
        },
        ResetTransition {
            from: TicketPhase::InQa,
            to: TicketPhase::Reviewed,
            pipeline_reservation: false,
        },
        ResetTransition {
            from: TicketPhase::InReview,
            to: TicketPhase::DiagnosticsDone,
            pipeline_reservation: false,
        },
        ResetTransition {
            from: TicketPhase::Analysis,
            to: TicketPhase::Backlog,
            pipeline_reservation: false,
        },
    ];
    /// Reset all in-flight tickets to their ready state (for crash/restart recovery).
    ///
    /// Resets:
    /// - 5 of the 9 `PIPELINE_BLOCKING_PHASES` where agents may have been mid-work
    ///   (InDevelopment, InDiagnostics, InSanitation, InReview, InQa) — roll back to
    ///   their pre-pipeline state
    /// - `Analysis` (not a pipeline blocker, but backlog analysts may crash mid-work)
    ///
    /// Tickets that are bounced to `ReadyForDevelopment` (InDevelopment and InDiagnostics)
    /// get `pipeline_reservation = 1` so they are claimed before any fresh
    /// `ReadyForDevelopment` ticket — this preserves the rework priority across restarts.
    ///
    /// Excludes DiagnosticsDone, SanitationPassed, Reviewed, and QaPassed — these are transitory handoff states
    /// that the poller picks up within the next poll cycle.
    ///
    /// Uses `Self::RESET_TRANSITIONS` (extracted as an associated const so tests
    /// can verify coverage against `PIPELINE_BLOCKING_PHASES`).
    pub async fn reset_inflight_tickets(&self) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        let now = turso::now();
        for transition in Self::RESET_TRANSITIONS {
            tx.execute(
                "UPDATE tickets SET phase = ?1, assigned_to = NULL, updated_at = ?2, \
                 pipeline_reservation = ?4 WHERE phase = ?3",
                turso::params![
                    transition.to.as_ref(),
                    now.clone(),
                    transition.from.as_ref(),
                    i64::from(transition.pipeline_reservation)
                ],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Shared implementation for checking if a workspace has active tickets.
    ///
    /// Returns `true` if any ticket in the workspace has a pipeline-blocking
    /// phase ([`PIPELINE_BLOCKING_PHASES`]), or a
    /// [`ReadyForDevelopment`](TicketPhase::ReadyForDevelopment) ticket,
    /// optionally excluding a specific ticket ID.
    ///
    /// [`has_active_tickets_excluding`] delegates to this helper. The test-only
    /// [`has_pipeline_blocker_for_workspace`] delegates too, with
    /// `require_rfd_reservation = true`.
    ///
    /// # Parameters
    ///
    /// * `workspace_name` — The workspace to check.
    /// * `exclude_ticket_id` — When `Some(id)`, that ticket is excluded from
    ///   the check. When `None`, no exclusion is applied (the SQL clause
    ///   `(?2 IS NULL OR id != ?2)` short-circuits to `TRUE`).
    /// * `require_rfd_reservation` — When `true`, only ReadyForDevelopment
    ///   tickets with `pipeline_reservation = 1` are counted as active (used
    ///   by the pipeline-blocker check). When `false`, all ReadyForDevelopment
    ///   tickets are active regardless of reservation (notification-suppression
    ///   policy for [`has_active_tickets_excluding`]).
    ///
    /// Excludes archived tickets — the only phases that ever get archived are
    /// `Done` and `Cancelled`, neither of which appears in
    /// `PIPELINE_BLOCKING_PHASES`, so this is a defensive consistency measure.
    ///
    /// # Parameter binding note
    ///
    /// The SQL always binds three positional parameters:
    /// - `?1`: `workspace_name`
    /// - `?2`: `exclude_ticket_id` (may be `None`)
    /// - `?3`: ReadyForDevelopment phase value
    ///
    /// When `require_rfd_reservation = true` the RFD branch adds
    /// `AND pipeline_reservation = 1` to the same `?3` position.
    async fn has_active_tickets_internal(
        &self,
        workspace_name: &str,
        exclude_ticket_id: Option<&str>,
        require_rfd_reservation: bool,
    ) -> Result<bool> {
        let blocker_sql = phase_list_sql_fragment(PIPELINE_BLOCKING_PHASES);
        let rfd_condition = if require_rfd_reservation {
            "(phase = ?3 AND pipeline_reservation = 1)".to_string()
        } else {
            "phase = ?3".to_string()
        };
        let sql = format!(
            "SELECT 1 FROM tickets WHERE \
             (phase IN ({blocker_sql}) OR {rfd_condition}) \
             AND workspace_name = ?1 AND is_archived = 0 \
             AND (?2 IS NULL OR id != ?2) LIMIT 1",
        );
        let rfd = TicketPhase::ReadyForDevelopment.as_ref();
        let rows = self
            .conn
            .query(&sql, turso::params![workspace_name, exclude_ticket_id, rfd])
            .await?;
        Ok(!rows.is_empty())
    }

    /// Returns `true` if the given workspace has any ticket with a
    /// pipeline-blocking phase (dev/review/QA), OR any reserved
    /// ReadyForDevelopment ticket that was bounced back and is awaiting rework.
    ///
    /// **Test-only query** — retained to provide coverage of the pipeline-blocker
    /// SQL variant. Production code uses [`has_active_tickets_excluding`] or
    /// [`count_by_phase`].
    ///
    /// Delegates to [`has_active_tickets_internal`] with
    /// `exclude_ticket_id = None` and `require_rfd_reservation = true`.
    ///
    /// Note this includes `AND pipeline_reservation = 1` for ReadyForDevelopment
    /// tickets, unlike [`has_active_tickets_excluding`] (the production entry
    /// point) which treats all ReadyForDevelopment tickets as active regardless
    /// of reservation.
    ///
    /// # Maintenance warning
    ///
    /// If a future feature needs this in production, remove the `#[cfg(test)]`
    /// gate and add a real caller. The doc comment and tests will validate the
    /// query is correct before any production use.
    ///
    /// Excludes archived tickets — the only phases that ever get archived are
    /// `Done` and `Cancelled`, neither of which appears in
    /// `PIPELINE_BLOCKING_PHASES`, so this is a defensive consistency measure.
    #[cfg(test)]
    pub(crate) async fn has_pipeline_blocker_for_workspace(
        &self,
        workspace_name: &str,
    ) -> Result<bool> {
        self.has_active_tickets_internal(workspace_name, None, true)
            .await
    }

    /// Check if the workspace has any active tickets other than the excluded one.
    ///
    /// "Active" means a ticket whose phase is either a pipeline-blocking phase
    /// (`PIPELINE_BLOCKING_PHASES`) or [`TicketPhase::ReadyForDevelopment`]
    /// (regardless of `pipeline_reservation` — unstarted backlog tickets are
    /// considered active to suppress Done notifications until the pipeline is
    /// fully drained).
    ///
    /// Delegates to [`has_active_tickets_internal`]. The test-only
    /// [`has_pipeline_blocker_for_workspace`] additionally requires
    /// `pipeline_reservation = 1` for ReadyForDevelopment tickets.
    ///
    /// Non-active phases (not matched by the query): `Done`, `Cancelled`,
    /// `Failed`, `Backlog`, `Analysis`, `Planning`.
    ///
    /// # Race condition note
    ///
    /// When multiple QaPassed tickets in the same workspace are finalized
    /// concurrently (each via [`tokio::spawn`] in the poller), both may see
    /// each other as active and both buffer their Done transitions. In this
    /// scenario all tickets are already in Done in the database — the only
    /// consequence is that Done notifications are delayed until the next
    /// [`crate::manager_queue::JobKind::UserMessage`] drains the buffer. This is an accepted trade-off:
    /// the race window is small and the buffer always drains eventually.
    pub async fn has_active_tickets_excluding(
        &self,
        workspace_name: &str,
        exclude_ticket_id: &str,
    ) -> Result<bool> {
        self.has_active_tickets_internal(workspace_name, Some(exclude_ticket_id), false)
            .await
    }

    /// Add a comment to a ticket (append-only).
    pub async fn add_comment(&self, ticket_id: &str, role: &str, content: &str) -> Result<()> {
        crate::turso::with_tx(&self.conn, ticket_id, "add comment", async |tx| {
            Self::add_comment_tx(tx, ticket_id, role, content).await
        })
        .await
    }

    /// Transactional variant of [`add_comment`](Self::add_comment) —
    /// uses an existing transaction instead of opening its own.
    /// Does NOT commit or rollback; the caller controls outer transaction lifecycle.
    ///
    /// Inserts the comment record AND updates the ticket's `updated_at` timestamp.
    pub(crate) async fn add_comment_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let comment_id = crate::generate_id();
        let now = turso::now();
        tx.execute(
            "INSERT INTO ticket_comments (id, ticket_id, role, content, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            turso::params![comment_id, ticket_id, role, content, now.as_str()],
        )
        .await?;
        tx.execute(
            "UPDATE tickets SET updated_at = ?1 WHERE id = ?2",
            turso::params![now.as_str(), ticket_id],
        )
        .await?;
        Ok(())
    }

    /// Get all comments for a ticket, ordered by creation time.
    pub async fn get_comments(&self, ticket_id: &str) -> Result<Vec<TicketComment>> {
        let sql = format!(
            "SELECT {COMMENT_COLUMNS} FROM ticket_comments WHERE ticket_id = ?1 ORDER BY created_at ASC"
        );
        let rows = self.conn.query(&sql, turso::params![ticket_id]).await?;
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
    /// Used by the dashboard to show tickets across all workspaces.
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
             ORDER BY created_at DESC",
            turso::params![workspace_name, phase_str],
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
                turso::params![phase.as_ref(), workspace_name],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Archive a single ticket by ID.
    ///
    /// Sets `is_archived = 1` and clears `assigned_to` (archived tickets should
    /// not remain assigned). Returns an error if the ticket does not exist.
    ///
    /// **Ordering constraint:** The caller must transition the ticket to a
    /// terminal state (`done` or `cancelled`) *before* calling this method.
    /// There is no `assigned_to IS NULL` guard — [`transition_to`](Self::transition_to)
    /// already clears the assignee, and a single-ticket archive on an assigned
    /// ticket is intentionally allowed to resolve stale assignments.
    pub async fn set_archived(&self, ticket_id: &str) -> Result<()> {
        let prepared = Self::build_ticket_update_with_updated_at(
            "is_archived = 1, assigned_to = NULL",
            vec![],
            ticket_id,
        );
        prepared.execute_and_cancel(&self.conn).await
    }

    /// Move all non-archived ReadyForDevelopment tickets in the given workspace
    /// to Planning, clearing their assignments.
    ///
    /// Used by the circuit breaker to drain sibling ReadyForDevelopment tickets
    /// when a ticket in the same workspace fails, ensuring pipeline reservation
    /// ordering is preserved (bounced tickets get priority over fresh ones).
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
        let now = turso::now();
        let updated = self
            .conn
            .execute(
                "UPDATE tickets SET phase = ?1, assigned_to = NULL, updated_at = ?2 \
                 WHERE phase = ?3 AND workspace_name = ?4 AND is_archived = 0",
                turso::params![
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
        let now = turso::now();
        let cutoff = (Utc::now() - Duration::hours(hours)).to_rfc3339();
        let updated = self
            .conn
            .execute(
                "UPDATE tickets SET is_archived = 1, updated_at = ?1 \
                 WHERE phase = ?2 AND updated_at < ?3 AND assigned_to IS NULL \
                 AND is_archived = 0",
                turso::params![now, TicketPhase::Cancelled.as_ref(), cutoff],
            )
            .await
            .context("Failed to archive stale cancelled tickets")?;
        Ok(updated)
    }

    pub async fn archive_all_done_and_cancelled(
        &self,
        workspace_name: Option<&str>,
    ) -> Result<u64> {
        let now = turso::now();
        let done_cancelled = [TicketPhase::Done, TicketPhase::Cancelled];
        let sql = format!(
            "UPDATE tickets SET is_archived = 1, updated_at = ?1 \
             WHERE phase IN ({}) AND assigned_to IS NULL AND is_archived = 0 \
             AND (?2 IS NULL OR workspace_name = ?2)",
            phase_list_sql_fragment(&done_cancelled),
        );
        let updated = self
            .conn
            .execute(&sql, turso::params![now, workspace_name])
            .await
            .context("Failed to archive done/cancelled tickets")?;
        Ok(updated)
    }

    // ── Archived ticket search methods ────────────────────────────────────
    //
    // These BoardStore methods contain the FTS and embedding SQL used by
    // [`SearchArchivedTicketsTool`](crate::tools::search_archived_tickets).
    // The board owns the schema (`ngram` tokenizer, FTS index name, blob format)
    // and the tool layer owns the hybrid RRF merge logic.

    /// Search archived tickets by FTS keyword match.
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
    ) -> Result<Vec<(String, f64)>> {
        let sanitized = crate::turso::sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let sql = format!(
            "SELECT t.id, fts_score(t.title, ?1) AS score \
             FROM tickets t \
             WHERE t.is_archived = 1 \
               AND t.title MATCH ?1 \
             ORDER BY score DESC LIMIT {limit}"
        );
        match self
            .conn
            .query_map(&sql, turso::params![sanitized.clone()], |row| {
                let id: String = row.get(0)?;
                let score: f64 = row.get(1)?;
                Ok::<_, anyhow::Error>((id, score))
            })
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

    /// List archived tickets with non-NULL embeddings, deserialized.
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
    pub async fn list_archived_with_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let rows = self
            .conn
            .query(
                "SELECT id, embedding FROM tickets \
                 WHERE is_archived = 1 AND embedding IS NOT NULL",
                turso::params![],
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

/// Open a [`BoardStore`] in a fresh temp directory (no global CONFIG dependency).
///
/// Thin wrapper around [`crate::open_test_store!`] that avoids touching the 32
/// call sites inside `self::tests`.  Delegates to the shared macro so the
/// actual boilerplate lives in one place.
///
/// Internal test convenience — external modules should use the macro directly.
#[cfg(test)]
async fn open_test_store() -> (BoardStore, tempfile::TempDir) {
    crate::open_test_store!(BoardStore, "board")
}

#[cfg(test)]
#[path = "board_tests.rs"]
mod tests;
