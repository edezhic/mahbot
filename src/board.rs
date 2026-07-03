//! Ticket/board system — Turso-backed task management.

use crate::role::{DIAGNOSTICS_ROLE, SYSTEM_ROLE};
use crate::turso::{self, IntoParams, TxGuard, Value, params_from_iter};
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
    status          TEXT NOT NULL DEFAULT 'backlog',
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
        STATUS                 => "status",
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

/// Statuses where a ticket occupies the dev/review/QA pipeline.
///
/// Only one ticket at a time per workspace may be in this pipeline. Any ticket in one of these
/// statuses blocks new Engineer dispatches for that workspace. The Maintainer uses a separate
/// pre-development threshold (Analysis + Planning + ReadyForDevelopment) and is no longer
/// directly suppressed by this constant.
///
/// Note: [`BoardStore::reset_inflight_tickets`] (via [`BoardStore::RESET_TRANSITIONS`]) only resets a subset
/// of these (InDevelopment, InDiagnostics, InSanitation, InReview, InQa) plus Analysis — see its
/// docs for rationale. The remaining four phases ([`TRANSITORY_HANDOFF_PHASES`]) are transitory
/// handoff states intentionally excluded from reset. The `tests::test_pipeline_blockers_coverage`
/// test enforces that every non-transitory pipeline blocker has a corresponding reset transition.
const PIPELINE_BLOCKING_STATUSES: &[TicketPhase] = &[
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
/// This is a subset of [`PIPELINE_BLOCKING_STATUSES`]. The relationship is
/// mechanically verified by `tests::test_pipeline_blockers_coverage`.
const TRANSITORY_HANDOFF_PHASES: &[TicketPhase] = &[
    TicketPhase::DiagnosticsDone,
    TicketPhase::SanitationPassed,
    TicketPhase::Reviewed,
    TicketPhase::QaPassed,
];

/// Statuses that unblock dependent tickets.
///
/// When a ticket transitions to one of these statuses, any tickets that
/// depend on it become eligible for claiming (their prerequisite filter
/// no longer blocks them).
///
/// [`TicketPhase::Failed`] is intentionally excluded — a failed ticket
/// permanently blocks its dependents, requiring manual intervention.
///
/// [`TicketPhase::is_unblocking`] delegates to this constant to ensure
/// the unblocking set is always authoritative. If a new status is added
/// here, `is_unblocking()` automatically picks it up; if the set ever
/// needs to diverge from the unblocking set, this delegation must be
/// broken explicitly.
pub const UNBLOCKING_STATUSES: &[TicketPhase] = &[TicketPhase::Done, TicketPhase::Cancelled];

/// Produces an SQL fragment listing statuses as quoted, comma-separated
/// strings — e.g. `'done', 'cancelled'`.
///
/// # Precondition
///
/// The input slice must be non-empty. Passing an empty slice produces
/// `WHERE status IN ()` which is invalid SQL.
fn status_list_sql_fragment(statuses: &[TicketPhase]) -> String {
    statuses
        .iter()
        .map(|p| format!("'{p}'"))
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

/// The default phase assigned to newly created tickets.
pub const DEFAULT_TICKET_PHASE: TicketPhase = TicketPhase::Backlog;

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

#[derive(Debug, Clone, Serialize)]
pub struct TicketComment {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Ticket {
    pub id: String,
    pub title: String,
    pub description: String,
    #[serde(rename = "status")]
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
    /// disables the default 5K truncation).
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
    pub fn format_comments(&self) -> String {
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
    /// statuses where no agent is mid-execution.
    ///
    /// Delegates to `TRANSITORY_HANDOFF_PHASES` so the transitory handoff set can never
    /// accidentally diverge from the definition used in coverage tests.
    #[must_use]
    pub fn is_transitory_handoff(&self) -> bool {
        TRANSITORY_HANDOFF_PHASES.contains(self)
    }

    /// Returns `true` for phases that unblock dependent tickets.
    ///
    /// Delegates to [`UNBLOCKING_STATUSES`] so the unblocking set can never
    /// accidentally diverge from the prerequisite-unblocking set.
    /// [`TicketPhase::Failed`] is not in [`UNBLOCKING_STATUSES`] and is
    /// therefore not unblocking — a failed ticket permanently blocks its
    /// dependents and remains visible in active views for manual triage.
    #[must_use]
    pub fn is_unblocking(&self) -> bool {
        UNBLOCKING_STATUSES.contains(self)
    }

    /// Returns `true` if the ticket is in a pipeline-blocking phase.
    ///
    /// Tickets in these phases are actively being worked on by agents
    /// (development, diagnostics, review, QA). Automated tools (create,
    /// update, add_comment) should refuse to modify them to prevent
    /// race conditions with running agents.
    #[must_use]
    pub fn is_pipeline_blocking(&self) -> bool {
        PIPELINE_BLOCKING_STATUSES.contains(self)
    }

    /// Human-readable display label with spaces instead of underscores
    /// (e.g. `"in development"` from [`TicketPhase::InDevelopment`]).
    ///
    /// This is the presentation-oriented counterpart to `AsRefStr::as_ref`
    /// (which returns the machine-oriented `snake_case` form like
    /// `"in_development"`). Use `display_name()` for user-facing UI labels;
    /// keep `as_ref()` for tool output, SQL fragments, and agent-facing text
    /// where agents expect the snake_case status string.
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
                    "Invalid status '{s}'. Valid statuses: {}",
                    *ALL_TICKET_PHASE_NAMES
                )
            })
    }
}

// Display and AsRefStr are provided by strum derives. FromStr is implemented
// manually above to produce user-friendly error messages.

/// Bundles a SQL mutation statement with its parameters and a human-readable
/// action description. Returned by [`BoardStore::build_transition_sql`] and
/// [`BoardStore::build_set_commit_info_sql`]; also accepted by
/// [`BoardStore::execute_and_cancel`], [`BoardStore::execute_update`], and
/// [`BoardStore::execute_prepared_tx`].
struct PreparedUpdate {
    sql: String,
    params: Vec<turso::Value>,
    action: String,
}

/// A single reset transition: when a ticket in `from` phase is found on startup,
/// it is rolled back to `to` phase. If `pipeline_reservation` is true, the ticket
/// gets `pipeline_reservation = 1` so it is claimed before any fresh ticket in the
/// same phase (preserving rework priority across restarts).
#[derive(Debug, Clone, Copy)]
struct ResetTransition {
    from: TicketPhase,
    to: TicketPhase,
    pipeline_reservation: bool,
}

impl BoardStore {
    /// Post-open FTS index setup.
    async fn after_open(&self) -> anyhow::Result<()> {
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
            "INSERT INTO tickets (id, title, description, status, workspace_name, \
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
                    "UPDATE tickets SET prerequisites = ?1 WHERE id = ?2",
                    turso::params![new_json, dep_id],
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
        self.validate_prerequisites(&tx, prerequisites, workspace_name)
            .await?;
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
                "SELECT workspace_name, status FROM tickets WHERE id = ?1",
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
        let status_str: String = row.get(1)?;
        let old_phase: TicketPhase = status_str.parse()?;

        let now = turso::now();
        let cancelled_rows = tx
            .execute(
                "UPDATE tickets SET status = ?1, updated_at = ?2, assigned_to = NULL, \
                 superseded_by = ?4, is_archived = 1 WHERE id = ?3",
                turso::params![
                    TicketPhase::Cancelled.as_ref(),
                    now,
                    supersede_id,
                    new_id.as_str(),
                ],
            )
            .await?;
        Self::ensure_ticket_found(cancelled_rows, supersede_id, "cancel superseded ticket")?;

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
    async fn ticket_from_row(&self, row: &turso::Row, load_comments: bool) -> Result<Ticket> {
        let id: String = row.get(COL_TICKET_ID)?;
        let comments = if load_comments {
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
                .get::<String>(COL_TICKET_STATUS)?
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
    /// Only tickets currently in `expected_phase` are eligible for claiming.
    /// The WHERE clause includes `t1.status = ?` bound to `expected_phase`,
    /// providing CAS-style atomicity for phase transitions — if no ticket
    /// matches the expected phase, the claim returns `None`.
    ///
    /// When `require_clear_pipeline` is `true`, the claim is rejected (returns `None`)
    /// if any pipeline-blocking ticket exists in the same workspace. The
    /// occupancy check is part of the same atomic SQL UPDATE statement (no
    /// separate SELECT + UPDATE window). Pipeline-blocking statuses are defined
    /// in [`PIPELINE_BLOCKING_STATUSES`].
    ///
    /// Note that a reserved ReadyForDevelopment ticket (one with
    /// `pipeline_reservation = 1`) is **not** treated as a pipeline blocker for
    /// the purpose of this claim. This differs from
    /// [`has_pipeline_blocker_for_workspace`] (a test-only query that requires
    /// `pipeline_reservation = 1` for ReadyForDevelopment tickets) because the
    /// claim subquery orders by `pipeline_reservation DESC` and clears reservation
    /// on claim, so a reserved ticket at ReadyForDevelopment will be claimed
    /// before any other ticket at the same phase — no pipeline blocking needed.
    ///
    /// When `require_clear_pipeline` is `false`, the claim uses a simple LIMIT 1
    /// subquery with no pipeline gating. This is used for phases that should
    /// not be blocked by in-flight pipeline tickets (e.g., analysis, review, and QA).
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
        expected_phase: TicketPhase,
        target_phase: TicketPhase,
        workspace_name: &str,
        require_clear_pipeline: bool,
    ) -> Result<Option<Ticket>> {
        let now = turso::now();

        // Filter that excludes tickets with unmet prerequisites.
        let prereq_filter = format!(
            "AND NOT EXISTS ( \
               SELECT 1 FROM json_each(t1.prerequisites) AS je \
               JOIN tickets t_pre ON t_pre.id = je.value \
               WHERE t_pre.status NOT IN ({}) \
             )",
            status_list_sql_fragment(UNBLOCKING_STATUSES),
        );

        let pipeline_blocker_clause = if require_clear_pipeline {
            let blocker_sql = status_list_sql_fragment(PIPELINE_BLOCKING_STATUSES);
            format!(
                "AND NOT EXISTS (SELECT 1 FROM tickets t2 \
                 WHERE t2.workspace_name = t1.workspace_name \
                 AND t2.status IN ({blocker_sql}) \
                 AND t2.id != t1.id) "
            )
        } else {
            String::new()
        };

        let sql = format!(
            "UPDATE tickets SET status = ?1, assigned_to = NULL, updated_at = ?2, \
             pipeline_reservation = 0 \
             WHERE id = (SELECT t1.id FROM tickets t1 \
             WHERE t1.status = ?3 AND t1.assigned_to IS NULL AND t1.workspace_name = ?4 \
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
                    expected_phase.as_ref(),
                    workspace_name,
                ],
            )
            .await?;
        match rows.into_iter().next() {
            Some(row) => Ok(Some(self.ticket_from_row(&row, true).await?)),
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
    async fn select_tickets(
        &self,
        suffix: &str,
        params: impl IntoParams + Send + 'static,
        load_comments: bool,
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
            .select_tickets("WHERE id = ?1", turso::params![ticket_id], true)
            .await?
            .into_iter()
            .next())
    }

    /// Get a ticket's phase by id — lightweight, no comments loaded.
    pub async fn get_ticket_phase(&self, ticket_id: &str) -> Result<Option<TicketPhase>> {
        let sql = "SELECT status FROM tickets WHERE id = ?1";
        let rows = self.conn.query(sql, turso::params![ticket_id]).await?;
        match rows.into_iter().next() {
            Some(row) => {
                let status: String = row.get(0)?;
                Ok(Some(status.parse()?))
            }
            None => Ok(None),
        }
    }

    /// List all tickets in a given phase for a workspace, ordered by created_at
    /// descending. Does NOT load comments — lightweight enough for poll loops.
    ///
    /// Delegates to [`BoardStore::list_all_tickets`] with both filters set — this function
    /// is a convenience wrapper with non-optional parameters. Because it shares
    /// the same query path, future changes to `list_all_tickets` (sorting,
    /// filtering, comment loading) will affect this function too.
    pub(crate) async fn list_tickets_in_phase(
        &self,
        phase: TicketPhase,
        workspace_name: &str,
    ) -> Result<Vec<Ticket>> {
        self.list_all_tickets(Some(workspace_name), Some(phase))
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
    /// let prep = Self::update_tickets_with_updated_at(
    ///     "assigned_to = ?",
    ///     vec![Value::from("user-123")],
    ///     "set assigned_to",
    ///     "ticket-456",
    /// );
    /// // SQL:  "UPDATE tickets SET assigned_to = ?, updated_at = ? WHERE id = ?"
    /// // params: [user-123, now, ticket-456]
    /// ```
    fn update_tickets_with_updated_at(
        set_clause: &str,
        set_params: Vec<turso::Value>,
        action: String,
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
            action,
        }
    }

    /// Build the SQL, params, and action description for a ticket status
    /// transition. Shared by [`transition_to`](Self::transition_to) and
    /// [`transition_to_tx`](Self::transition_to_tx).
    ///
    /// Note: this does **not** use [`Self::update_tickets_with_updated_at`]
    /// because it has extra SET columns (`assigned_to = NULL`,
    /// `pipeline_reservation = COALESCE(?5, pipeline_reservation)`) and an
    /// additional WHERE condition (`AND (?4 IS NULL OR status = ?4)`) that
    /// don't fit the helper's fixed pattern.
    fn build_transition_sql(
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        reservation: Option<bool>,
    ) -> PreparedUpdate {
        let now = turso::now();
        let guard: Option<&str> = expected_phase.as_ref().map(TicketPhase::as_ref);
        let action = match reservation {
            Some(v) => format!(
                "set status to {} (reservation={})",
                target_phase.as_ref(),
                v,
            ),
            None => format!("set status to {}", target_phase.as_ref()),
        };
        let sql = "UPDATE tickets SET status = ?1, assigned_to = NULL, updated_at = ?2, \
                    pipeline_reservation = COALESCE(?5, pipeline_reservation) \
                    WHERE id = ?3 AND (?4 IS NULL OR status = ?4)";
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
            action,
        }
    }

    /// Update ticket status, optionally guarded by an expected phase for CAS-style
    /// atomicity. Always clears `assigned_to` and cancels running agents.
    ///
    /// # Note on [`pipeline_reservation`](Ticket::pipeline_reservation)
    ///
    /// When `reservation` is `None`, the column is left untouched so bounce-back
    /// transitions can set it atomically, and manual transitions leave stale
    /// reservations inert (claim/blocker queries filter by status). When
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
        self.execute_and_cancel(ticket_id, prepared).await
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
        Self::execute_prepared_tx(tx, prepared, ticket_id).await
    }

    /// Verify that a mutation query affected at least one row, returning an
    /// error with a descriptive message if the ticket was not found.
    fn ensure_ticket_found(rows: u64, ticket_id: &str, action: &str) -> Result<()> {
        anyhow::ensure!(rows > 0, "Ticket {ticket_id} not found — cannot {action}");
        Ok(())
    }

    /// Execute a prepared UPDATE within an existing transaction, verify it
    /// affected a row, and return success.
    ///
    /// This is the transactional counterpart of
    /// [`execute_update`](Self::execute_update) — it skips agent
    /// cancellation because the caller manages transaction lifecycle and must
    /// cancel agents before starting the transaction when needed.
    ///
    /// Used by [`transition_to_tx`](Self::transition_to_tx),
    /// [`set_assigned_to_tx`](Self::set_assigned_to_tx), and
    /// [`set_commit_info_tx`](Self::set_commit_info_tx).
    async fn execute_prepared_tx(
        tx: &TxGuard<'_>,
        prepared: PreparedUpdate,
        ticket_id: &str,
    ) -> Result<()> {
        let rows = tx.execute(&prepared.sql, prepared.params).await?;
        Self::ensure_ticket_found(rows, ticket_id, &prepared.action)?;
        Ok(())
    }

    /// Execute a prepared UPDATE and verify it affected a row.
    ///
    /// This is the non-transactional counterpart of
    /// [`execute_prepared_tx`](Self::execute_prepared_tx) — it skips agent
    /// cancellation. Use [`execute_and_cancel`](Self::execute_and_cancel) when
    /// agent cancellation is needed.
    ///
    /// Used by [`set_commit_info`](Self::set_commit_info).
    async fn execute_update(&self, ticket_id: &str, prepared: PreparedUpdate) -> Result<()> {
        let rows = self.conn.execute(&prepared.sql, prepared.params).await?;
        Self::ensure_ticket_found(rows, ticket_id, &prepared.action)?;
        Ok(())
    }

    /// Execute a mutation SQL, verify it affected a row, then cancel any agent
    /// registered on the ticket.
    ///
    /// This is a convenience helper for single-ticket mutation methods that
    /// follow the pattern: execute UPDATE → `ensure_ticket_found` → cancel stale
    /// agent. Used by [`transition_to`](Self::transition_to),
    /// [`set_assigned_to`](Self::set_assigned_to), and
    /// [`set_archived`](Self::set_archived).
    ///
    /// # When NOT to use
    ///
    /// Do **not** use this helper for methods with different semantics:
    /// - **`execute_update`** — for updates that should not cancel agents.
    /// - **`claim_diagnostics`** — returns `Result<bool>` (conditional success),
    ///   only cancels on successful claim.
    /// - **`supersede_and_create`** — runs inside a transaction, cancels before
    ///   commit via a different pattern.
    /// - **`batch_set_archived`** — batch operation on many tickets, handles
    ///   zero-affected-rows gracefully (no error).
    /// - **`claim_sanitation`** — returns `Result<bool>` (conditional success),
    ///   does NOT cancel registered agents (QaPassed has no running agent).
    async fn execute_and_cancel(&self, ticket_id: &str, prepared: PreparedUpdate) -> Result<()> {
        self.execute_update(ticket_id, prepared).await?;
        crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(ticket_id);
        Ok(())
    }

    fn build_set_assigned_to_sql(ticket_id: &str, assigned_to: Option<&str>) -> PreparedUpdate {
        let action = if assigned_to.is_some() {
            "set assigned_to"
        } else {
            "clear assigned_to"
        };
        Self::update_tickets_with_updated_at(
            "assigned_to = ?",
            vec![Value::from(assigned_to)],
            action.to_string(),
            ticket_id,
        )
    }

    /// Set or clear the assignee for a ticket (does not change status).
    ///
    /// When `assigned_to` is `Some(value)`, sets the `assigned_to` column to that
    /// value. When `None`, clears the assignee (sets `assigned_to = NULL`).
    ///
    /// Cancels any running agents registered on this ticket as a safety-in-depth
    /// measure against stale assignments. For set operations, callers typically
    /// set the assignee before spawning an agent, so the cancel is normally a
    /// no-op. For clear operations, this ensures no stale agent remains bound to
    /// a now-unassigned ticket.
    pub async fn set_assigned_to(&self, ticket_id: &str, assigned_to: Option<&str>) -> Result<()> {
        let prepared = Self::build_set_assigned_to_sql(ticket_id, assigned_to);
        self.execute_and_cancel(ticket_id, prepared).await
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
        let prepared = Self::build_set_assigned_to_sql(ticket_id, assigned_to);
        Self::execute_prepared_tx(tx, prepared, ticket_id).await
    }

    /// Atomically claim a ticket for diagnostics execution.
    ///
    /// Sets `assigned_to = 'diagnostics'` only when the ticket is unassigned
    /// AND still in [`TicketPhase::InDiagnostics`] — a single atomic SQL guard
    /// that prevents the TOCTOU race between the poll listing pre-filter and the
    /// subsequent claim. The assignee set and phase check are fused into one
    /// UPDATE; callers do not need a separate `set_assigned_to` after claiming.
    ///
    /// Returns `Ok(true)` if a row was updated (claim succeeded), `Ok(false)`
    /// if no row matched (already claimed by another dispatch or ticket moved
    /// out of [`TicketPhase::InDiagnostics`]). On a successful claim, cancels
    /// any agent registered on this ticket as a safety-in-depth measure against
    /// stale dispatches.
    pub async fn claim_diagnostics(&self, ticket_id: &str) -> Result<bool> {
        let now = turso::now();
        let rows = self
            .conn
            .execute(
                "UPDATE tickets \
                 SET assigned_to = ?1, updated_at = ?2 \
                 WHERE id = ?3 \
                 AND assigned_to IS NULL \
                 AND status = ?4",
                turso::params![
                    DIAGNOSTICS_ROLE,
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
    /// [`TicketPhase::InSanitation`], sets `assigned_to` to the sanitation
    /// session key, and enforces the per-workspace serialization invariant:
    /// only one ticket at a time may be in [`TicketPhase::InSanitation`] or
    /// [`TicketPhase::SanitationPassed`].
    ///
    /// Returns `Ok(true)` if the claim succeeded, `Ok(false)` if:
    /// - The ticket is no longer in QaPassed (already claimed by another handler), or
    /// - Another ticket is already in the sanitation pipeline for this workspace.
    ///
    /// Unlike [`transition_to`](Self::transition_to), this method does NOT
    /// cancel registered agents — QaPassed is a transitory handoff phase
    /// with no running agent, so cancellation is unnecessary.
    pub async fn claim_sanitation(&self, ticket_id: &str) -> Result<bool> {
        let now = turso::now();
        let session_key =
            crate::session::ticket_session_key(ticket_id, crate::Role::Sanitation.as_str());
        let blocker =
            status_list_sql_fragment(&[TicketPhase::InSanitation, TicketPhase::SanitationPassed]);
        let sql = format!(
            "UPDATE tickets SET status = ?1, assigned_to = ?2, updated_at = ?3 \
             WHERE id = ?4 AND status = ?5 \
             AND NOT EXISTS (SELECT 1 FROM tickets t2 \
               WHERE t2.workspace_name = \
                 (SELECT workspace_name FROM tickets WHERE id = ?4) \
               AND t2.id != ?4 \
               AND t2.status IN ({blocker}))"
        );
        let rows = self
            .conn
            .execute(
                &sql,
                turso::params![
                    TicketPhase::InSanitation.as_ref(),
                    session_key,
                    now,
                    ticket_id,
                    TicketPhase::QaPassed.as_ref(),
                ],
            )
            .await?;
        Ok(rows > 0)
    }

    /// Build the SQL, params, and action description for setting commit info.
    /// Shared by [`set_commit_info`](Self::set_commit_info) and
    /// [`set_commit_info_tx`](Self::set_commit_info_tx).
    fn build_set_commit_info_sql(
        ticket_id: &str,
        hash: &str,
        lines_added: i64,
        lines_removed: i64,
    ) -> PreparedUpdate {
        debug_assert!(
            lines_added >= 0,
            "lines_added must be non-negative: {lines_added}"
        );
        debug_assert!(
            lines_removed >= 0,
            "lines_removed must be non-negative: {lines_removed}"
        );

        Self::update_tickets_with_updated_at(
            "commit_hash = ?, lines_added = ?, lines_removed = ?",
            vec![
                Value::from(hash),
                Value::from(lines_added),
                Value::from(lines_removed),
            ],
            "set commit info".to_string(),
            ticket_id,
        )
    }

    /// Record commit metadata on a ticket.
    ///
    /// This is pure metadata — it does NOT cancel running agents or check
    /// ticket phase (unlike `set_assigned_to`, which cancels running agents). Non-negative line counts are
    /// enforced by debug assertions; the caller is responsible for providing
    /// valid values in production.
    pub async fn set_commit_info(
        &self,
        ticket_id: &str,
        hash: &str,
        lines_added: i64,
        lines_removed: i64,
    ) -> Result<()> {
        let prepared = Self::build_set_commit_info_sql(ticket_id, hash, lines_added, lines_removed);
        self.execute_update(ticket_id, prepared).await
    }

    /// Transactional variant of [`set_commit_info`](Self::set_commit_info) —
    /// uses an existing transaction instead of its own connection write.
    /// Does NOT commit or rollback the transaction; the caller controls that.
    pub(crate) async fn set_commit_info_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        hash: &str,
        lines_added: i64,
        lines_removed: i64,
    ) -> Result<()> {
        let prepared = Self::build_set_commit_info_sql(ticket_id, hash, lines_added, lines_removed);
        Self::execute_prepared_tx(tx, prepared, ticket_id).await
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
    /// * `comment` — comment body to add (the role is fixed to [`SYSTEM_ROLE`])
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
    /// coverage against [`PIPELINE_BLOCKING_STATUSES`] without duplicating the pairs).
    ///
    /// Each entry maps a phase where an agent may have crashed mid-work back to the
    /// phase the ticket should resume in. Must be kept in sync with
    /// [`PIPELINE_BLOCKING_STATUSES`] — see `tests::test_pipeline_blockers_coverage`.
    ///
    /// Asymmetry: `Analysis → Backlog` is included (backlog analysts may crash mid-analysis),
    /// but `Analysis` is intentionally NOT in [`PIPELINE_BLOCKING_STATUSES`] (it's a pre-flight
    /// phase, not a pipeline blocker).
    ///
    /// Transitory handoff phases (see [`TicketPhase::is_transitory_handoff`]) are pipeline
    /// blocking but don't need a reset entry — the poller picks them up within seconds
    /// of restart, so no agent session is mid-execution in those states.
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
            // QaPassed uses list-based dispatch (for_tickets_in_phase), not the claim
            // loop where pipeline_reservation provides ordering. Set for consistency
            // with the crash-recovery pattern; the flag is harmless for list-based
            // dispatch.
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
    /// - 5 of the 9 `PIPELINE_BLOCKING_STATUSES` where agents may have been mid-work
    ///   (InDevelopment, InDiagnostics, InSanitation, InReview, InQa) — roll back to
    ///   their pre-pipeline state
    /// - `Analysis` (not a pipeline blocker, but backlog analysts may crash mid-work)
    ///
    /// Tickets that are bounced to `ReadyForDevelopment` (InDevelopment and InDiagnostics)
    /// get `pipeline_reservation = 1` so they are claimed before any fresh
    /// `ReadyForDevelopment` ticket — this preserves the rework priority across restarts.
    ///
    /// Excludes `TRANSITORY_HANDOFF_PHASES` — these are transitory handoff states
    /// that the poller picks up within 2 seconds of restart.
    ///
    /// Uses `Self::RESET_TRANSITIONS` (extracted as an associated const so tests
    /// can verify coverage against `PIPELINE_BLOCKING_STATUSES`).
    pub async fn reset_inflight_tickets(&self) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        let now = turso::now();
        for transition in Self::RESET_TRANSITIONS {
            tx.execute(
                "UPDATE tickets SET status = ?1, assigned_to = NULL, updated_at = ?2, \
                 pipeline_reservation = ?4 WHERE status = ?3",
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
    /// status ([`PIPELINE_BLOCKING_STATUSES`]), or a
    /// [`ReadyForDevelopment`](TicketPhase::ReadyForDevelopment) ticket
    /// (optionally filtered by `pipeline_reservation`), optionally excluding a
    /// specific ticket ID.
    ///
    /// Both [`has_pipeline_blocker_for_workspace`] and
    /// [`has_active_tickets_excluding`] delegate to this helper.
    ///
    /// # Parameters
    ///
    /// * `workspace_name` — The workspace to check.
    /// * `require_reservation` — When `true`, only `ReadyForDevelopment` tickets
    ///   with `pipeline_reservation = 1` count as active (used by the test-only
    ///   pipeline-blocker query). When `false`, all `ReadyForDevelopment` tickets
    ///   count regardless of reservation.
    /// * `exclude_ticket_id` — When `Some(id)`, that ticket is excluded from
    ///   the check (e.g., when checking if other active tickets remain after one
    ///   ticket completes). When `None`, no exclusion is applied.
    ///
    /// Excludes archived tickets — the only statuses that ever get archived are
    /// `Done` and `Cancelled`, neither of which appears in
    /// `PIPELINE_BLOCKING_STATUSES`, so this is a defensive consistency measure.
    async fn has_active_tickets_internal(
        &self,
        workspace_name: &str,
        require_reservation: bool,
        exclude_ticket_id: Option<&str>,
    ) -> Result<bool> {
        let blocker_sql = status_list_sql_fragment(PIPELINE_BLOCKING_STATUSES);
        let reservation_clause = if require_reservation {
            " AND pipeline_reservation = 1"
        } else {
            ""
        };
        let sql = format!(
            "SELECT 1 FROM tickets WHERE \
             (status IN ({blocker_sql}) OR (status = '{rfd}'{reservation_clause})) \
             AND workspace_name = ?1 AND is_archived = 0 \
             AND (?2 IS NULL OR id != ?2) LIMIT 1",
            rfd = TicketPhase::ReadyForDevelopment.as_ref(),
        );
        let rows = self
            .conn
            .query(&sql, turso::params![workspace_name, exclude_ticket_id])
            .await?;
        Ok(!rows.is_empty())
    }

    /// Returns true if the given workspace has any ticket with a pipeline-blocking
    /// status (dev/review/QA), OR any reserved ReadyForDevelopment ticket that
    /// was bounced back and is awaiting rework.
    ///
    /// **Test-only query** — retained to provide coverage of the pipeline-blocker SQL.
    /// Production code uses [`has_active_tickets_excluding`] or [`count_by_status`].
    ///
    /// # Maintenance warning
    /// If a future feature needs this in production, remove the `#[cfg(test)]`
    /// gate and add a real caller. The doc comment and tests will validate the
    /// query is correct before any production use.
    ///
    /// Excludes archived tickets — the only statuses that ever get archived are
    /// `Done` and `Cancelled`, neither of which appears in
    /// `PIPELINE_BLOCKING_STATUSES`, so this is a defensive consistency measure.
    #[cfg(test)]
    pub async fn has_pipeline_blocker_for_workspace(&self, workspace_name: &str) -> Result<bool> {
        self.has_active_tickets_internal(workspace_name, true, None)
            .await
    }

    /// Check if the workspace has any active tickets other than the excluded one.
    ///
    /// "Active" means a ticket whose status is either a pipeline-blocking status
    /// (`PIPELINE_BLOCKING_STATUSES`) or [`TicketPhase::ReadyForDevelopment`]
    /// (regardless of `pipeline_reservation` — unstarted backlog tickets are
    /// considered active to suppress Done notifications until the pipeline is
    /// fully drained).
    ///
    /// Delegates to `has_active_tickets_internal` with
    /// `require_reservation = false`. The test-only
    /// `has_pipeline_blocker_for_workspace` uses `require_reservation = true`,
    /// requiring `pipeline_reservation = 1` for `ReadyForDevelopment` tickets.
    ///
    /// Non-active statuses (not matched by the query): `Done`, `Cancelled`,
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
        self.has_active_tickets_internal(workspace_name, false, Some(exclude_ticket_id))
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
        &self,
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
        // deadlock with self.conn.query().
        let sql = format!(
            "SELECT id, workspace_name FROM tickets WHERE id IN ({})",
            turso::sql_in_placeholders(prerequisite_ids.len()),
        );
        let params: Vec<Value> = prerequisite_ids
            .iter()
            .map(|id| Value::Text(id.clone()))
            .collect();
        let rows = tx.query(&sql, params_from_iter(params)).await?;

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

    /// List all tickets, optionally filtered by workspace and/or status.
    /// Used by the dashboard to show tickets across all workspaces.
    pub async fn list_all_tickets(
        &self,
        workspace_name: Option<&str>,
        status_filter: Option<TicketPhase>,
    ) -> Result<Vec<Ticket>> {
        let status_str: Option<&str> = status_filter.as_ref().map(TicketPhase::as_ref);
        self.select_tickets(
            "WHERE (?1 IS NULL OR workspace_name = ?1) \
             AND (?2 IS NULL OR status = ?2) \
             AND is_archived = 0 \
             ORDER BY created_at DESC",
            turso::params![workspace_name, status_str],
            false,
        )
        .await
    }

    /// Count how many tickets have the given status, optionally filtered by workspace.
    ///
    /// Excludes archived tickets to stay consistent with [`list_all_tickets`](Self::list_all_tickets)
    /// and most other read paths in this module. Currently unused for `Done` or `Cancelled`
    /// (the only statuses that ever get archived), so this is a defensive consistency fix —
    /// callers that pass a terminal status will not see archived tickets in the count.
    pub async fn count_by_status(
        &self,
        phase: TicketPhase,
        workspace_name: Option<&str>,
    ) -> Result<i64> {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM tickets \
                 WHERE status = ?1 \
                   AND (?2 IS NULL OR workspace_name = ?2) \
                   AND is_archived = 0",
                turso::params![phase.as_ref(), workspace_name],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Execute a SELECT query and collect the first column as candidate IDs
    /// for archival. All returned rows are included as-is; no status or phase
    /// validation is performed.
    async fn collect_archive_candidates(
        &self,
        sql: &str,
        params: impl turso::IntoParams + Send + 'static,
    ) -> Result<Vec<String>> {
        let rows = self.conn.query(sql, params).await?;
        let mut candidates = Vec::new();
        for row in rows {
            let id: String = row.get(0)?;
            candidates.push(id);
        }
        Ok(candidates)
    }

    /// Maximum IDs per batch to stay well under SQLite's
    /// `SQLITE_MAX_VARIABLE_NUMBER` limit (default 999), with one
    /// placeholder reserved for the timestamp.
    const ARCHIVE_CHUNK_SIZE: usize = 500;

    async fn batch_set_archived(&self, items: &[String]) -> Result<u64> {
        if items.is_empty() {
            return Ok(0);
        }
        let now = turso::now();
        let mut total: u64 = 0;
        for chunk in items.chunks(Self::ARCHIVE_CHUNK_SIZE) {
            let sql = format!(
                "UPDATE tickets SET is_archived = 1, updated_at = ? \
                 WHERE id IN ({}) AND assigned_to IS NULL",
                turso::sql_in_placeholders(chunk.len()),
            );
            let mut params: Vec<Value> = vec![Value::Text(now.clone())];
            params.extend(chunk.iter().map(|id| Value::Text(id.clone())));
            total += self
                .conn
                .execute(&sql, params_from_iter(params))
                .await
                .context("Failed to batch-archive tickets")?;
        }
        Ok(total)
    }

    /// Archive a single ticket by ID.
    ///
    /// Sets `is_archived = 1` and clears `assigned_to` (archived tickets should
    /// not remain assigned). Returns an error if the ticket does not exist.
    ///
    /// **Ordering constraint:** The caller must transition the ticket to a
    /// terminal state (`done` or `cancelled`) *before* calling this method.
    /// Unlike `batch_set_archived` there is no
    /// `assigned_to IS NULL` guard — [`transition_to`](Self::transition_to)
    /// already clears the assignee, and a single-ticket archive on an assigned
    /// ticket is intentionally allowed to resolve stale assignments.
    pub async fn set_archived(&self, ticket_id: &str) -> Result<()> {
        let prepared = Self::update_tickets_with_updated_at(
            "is_archived = 1, assigned_to = NULL",
            vec![],
            "set archived".to_string(),
            ticket_id,
        );
        self.execute_and_cancel(ticket_id, prepared).await
    }

    pub async fn archive_stale_cancelled(&self, hours: i64) -> Result<u64> {
        let cutoff = (Utc::now() - Duration::hours(hours)).to_rfc3339();
        let to_archive = self
            .collect_archive_candidates(
                "SELECT id FROM tickets \
             WHERE status = ?1 AND updated_at < ?2 AND assigned_to IS NULL \
             AND is_archived = 0",
                turso::params![TicketPhase::Cancelled.as_ref(), cutoff],
            )
            .await?;
        self.batch_set_archived(&to_archive).await
    }

    pub async fn archive_all_done_and_cancelled(
        &self,
        workspace_name: Option<&str>,
    ) -> Result<u64> {
        let done_cancelled = [TicketPhase::Done, TicketPhase::Cancelled];
        let select_sql = format!(
            "SELECT id FROM tickets WHERE status IN ({}) \
             AND assigned_to IS NULL AND is_archived = 0 \
             AND (?1 IS NULL OR workspace_name = ?1)",
            status_list_sql_fragment(&done_cancelled),
        );
        let to_archive = self
            .collect_archive_candidates(&select_sql, turso::params![workspace_name])
            .await?;
        self.batch_set_archived(&to_archive).await
    }

    // ── Archived ticket search methods ────────────────────────────────────
    //
    // These encapsulate the FTS and embedding SQL that
    // [`SearchArchivedTicketsTool`](crate::tools::search_archived_tickets)
    // previously ran directly against `self.conn`. The board owns the schema
    // (`ngram` tokenizer, FTS index name, blob format) and the tool layer
    // owns the hybrid RRF merge logic.

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

/// Minimal ticket info for display purposes (id, title, status).
#[derive(Debug, Clone)]
pub(crate) struct TicketMinimal {
    pub id: String,
    pub title: String,
    pub phase: String,
}

impl BoardStore {
    /// Batch fetch minimal ticket info (id, title, status) by ID list.
    ///
    /// Results are returned in the same order as `ids`. Missing IDs are
    /// silently omitted (no error).
    ///
    /// This method exists specifically for the search-archived-ticket
    /// formatting path, which needs just `id`, `title`, and `status` for
    /// display purposes after hybrid ranking.
    pub(crate) async fn list_tickets_minimal(&self, ids: &[String]) -> Result<Vec<TicketMinimal>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "SELECT id, title, status FROM tickets WHERE id IN ({})",
            turso::sql_in_placeholders(ids.len()),
        );
        let params: Vec<Value> = ids.iter().map(|id| Value::Text(id.clone())).collect();

        let rows = self.conn.query(&sql, params_from_iter(params)).await?;

        let mut map: HashMap<String, TicketMinimal> = HashMap::new();
        for row in &rows {
            let ticket_id: String = row.get(0)?;
            let title: String = row.get(1)?;
            let phase: String = row.get(2)?;
            map.insert(
                ticket_id.clone(),
                TicketMinimal {
                    id: ticket_id,
                    title,
                    phase,
                },
            );
        }

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(entry) = map.get(id) {
                results.push(entry.clone());
            }
        }
        Ok(results)
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
mod tests {
    use super::*;
    use crate::Role;
    use crate::Tool;
    use crate::Workspace;
    use crate::role::DIAGNOSTICS_ROLE;
    use crate::util::test::TicketBuilder;
    use crate::util::test::assert_superseded_ticket;
    use crate::util::test::expect_ticket;
    use crate::util::test::init_test_stores;
    use crate::util::test::make_ticket;
    use crate::workspace::test_ws;
    use crate::workspace::test_ws_named;
    use strum::IntoEnumIterator;
    use tempfile::TempDir;

    /// Open a test store and create a default ticket.
    /// Returns (store, temp_dir, ticket_id).
    async fn setup() -> (BoardStore, TempDir, String) {
        let (store, tmp) = open_test_store().await;
        let id = make_ticket(
            &store,
            &test_ws_named("/ws", "ws"),
            "Test",
            TicketPhase::Backlog,
        )
        .await;
        (store, tmp, id)
    }

    #[tokio::test]
    async fn test_get_ticket_phase() {
        let (store, _tmp) = open_test_store().await;

        // Non-existent ticket returns None.
        assert!(
            store
                .get_ticket_phase("nonexistent")
                .await
                .expect("query")
                .is_none()
        );

        let id = make_ticket(
            &store,
            &crate::workspace::test_ws_named("/workspace", "workspace"),
            "Status Test",
            TicketPhase::Planning,
        )
        .await;

        let phase = crate::util::test::expect_ticket_phase(&store, &id).await;
        assert_eq!(phase, TicketPhase::Planning);

        // After transition, reflects new status.
        store
            .transition_to(&id, None, TicketPhase::ReadyForDevelopment, None)
            .await
            .expect("set");
        let phase = crate::util::test::expect_ticket_phase(&store, &id).await;
        assert_eq!(phase, TicketPhase::ReadyForDevelopment);
    }

    #[test]
    fn test_ticket_phase_parse_and_roundtrip() {
        // Parse roundtrip — all variants
        let variants: &[(&str, TicketPhase)] = &[
            ("backlog", TicketPhase::Backlog),
            ("analysis", TicketPhase::Analysis),
            ("planning", TicketPhase::Planning),
            ("ready_for_development", TicketPhase::ReadyForDevelopment),
            ("in_development", TicketPhase::InDevelopment),
            ("in_diagnostics", TicketPhase::InDiagnostics),
            ("diagnostics_done", TicketPhase::DiagnosticsDone),
            ("in_sanitation", TicketPhase::InSanitation),
            ("sanitation_passed", TicketPhase::SanitationPassed),
            ("in_review", TicketPhase::InReview),
            ("reviewed", TicketPhase::Reviewed),
            ("in_qa", TicketPhase::InQa),
            ("qa_passed", TicketPhase::QaPassed),
            ("done", TicketPhase::Done),
            ("cancelled", TicketPhase::Cancelled),
            ("failed", TicketPhase::Failed),
        ];
        for (s, expected) in variants {
            assert_eq!(&s.parse::<TicketPhase>().unwrap(), expected, "variant: {s}");
        }

        // Roundtrip: as_ref() -> parse() for every variant
        for v in TicketPhase::iter() {
            let parsed: TicketPhase = v.as_ref().parse().unwrap();
            assert_eq!(&parsed, &v, "roundtrip failed for {v}");
        }

        // Error case
        assert!("unknown_phase".parse::<TicketPhase>().is_err());
    }

    #[test]
    fn test_display_name_no_underscores() {
        // Every variant's display_name() must be underscore-free
        // and non-empty.
        for variant in TicketPhase::iter() {
            let name = variant.display_name();
            assert!(!name.is_empty(), "empty display_name for {variant}");
            assert!(
                !name.contains('_'),
                "display_name for {variant} still has underscore: {name}"
            );
        }
    }

    #[test]
    fn test_ticket_phase_from_str_error_message() {
        let err = "bogus_status".parse::<TicketPhase>().unwrap_err();
        let msg = format!("{err}");

        assert!(
            msg.contains("Invalid status"),
            "error should mention 'Invalid status', got: {msg}"
        );
        assert!(
            msg.contains("bogus_status"),
            "error should contain the invalid input value, got: {msg}"
        );
        assert!(
            msg.contains("backlog"),
            "error should list valid phases (e.g. backlog), got: {msg}"
        );
        assert!(
            msg.contains("cancelled"),
            "error should list valid phases (e.g. cancelled), got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_unconditional_transition_clears_assignment() {
        let (store, _tmp, id) = setup().await;

        // Claim the ticket (sets assigned_to to NULL by default)
        let claimed = store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::InDevelopment,
                "ws",
                false,
            )
            .await
            .expect("claim")
            .expect("ticket exists");

        // Set assigned_to explicitly (matching production dispatch_engineer behavior)
        store
            .set_assigned_to(&claimed.id, Some(Role::Engineer.as_str()))
            .await
            .expect("set_assigned_to");
        let ticket = store
            .get_ticket(&id)
            .await
            .expect("get")
            .expect("should exist");
        assert!(
            ticket.assigned_to.is_some(),
            "assigned_to should be set after set_assigned_to"
        );

        // Update status — this should clear assigned_to
        store
            .transition_to(&id, None, TicketPhase::DiagnosticsDone, None)
            .await
            .expect("update");

        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert_eq!(ticket.phase, TicketPhase::DiagnosticsDone);
        assert!(
            ticket.assigned_to.is_none(),
            "assigned_to should be cleared after unconditional transition"
        );
    }

    #[tokio::test]
    async fn test_guarded_transition() {
        let (store, _tmp, id) = setup().await;

        // Wrong expected phase — should fail, ticket unchanged.
        let result = store
            .transition_to(
                &id,
                Some(TicketPhase::Done),
                TicketPhase::InDevelopment,
                None,
            )
            .await;
        assert!(
            result.is_err(),
            "guarded transition with wrong phase should fail"
        );
        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert_eq!(ticket.phase, TicketPhase::Backlog);

        // Correct expected phase — should succeed.
        store
            .transition_to(
                &id,
                Some(TicketPhase::Backlog),
                TicketPhase::InDevelopment,
                None,
            )
            .await
            .expect("guarded transition with correct phase should succeed");
        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert_eq!(ticket.phase, TicketPhase::InDevelopment);
    }

    #[tokio::test]
    async fn test_add_comment() {
        let (store, _tmp, id) = setup().await;

        store
            .add_comment(&id, Role::Engineer.as_str(), "done!")
            .await
            .expect("add comment");

        let comments = store.get_comments(&id).await.expect("get comments");
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].role, Role::Engineer.as_str());
        assert_eq!(comments[0].content, "done!");
        assert!(!comments[0].created_at.is_empty());

        // Verify updated_at was bumped
        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert!(ticket.updated_at > ticket.created_at);
    }

    #[tokio::test]
    async fn test_list_tickets() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        make_ticket(&store, &ws, "A", TicketPhase::Backlog).await;
        make_ticket(&store, &ws, "B", TicketPhase::Backlog).await;
        make_ticket(&store, &ws, "C", TicketPhase::Backlog).await;

        // All tickets for the workspace
        let tickets = store
            .list_all_tickets(Some("ws"), None)
            .await
            .expect("list");
        assert_eq!(tickets.len(), 3);

        // Filter by status (none match since all are Backlog)
        let tickets = store
            .list_all_tickets(Some("ws"), Some(TicketPhase::Done))
            .await
            .expect("list");
        assert_eq!(tickets.len(), 0);
    }

    /// Verify that `reset_inflight_tickets` correctly transitions each in-flight
    /// ticket phase back to its ready state, and that non-inflight phases (e.g.
    /// Backlog) are left untouched.
    #[tokio::test]
    async fn test_reset_inflight_tickets() {
        /// A single reset transition case.
        struct Case {
            name: &'static str,
            /// Unique suffix for workspace names (isolates cases).
            suffix: &'static str,
            /// The phase the ticket starts in.
            start: TicketPhase,
            /// The expected phase after reset.
            expected: TicketPhase,
            /// Expected pipeline_reservation after reset.
            reservation: bool,
        }

        let cases = [
            Case {
                name: "Backlog unaffected (not an inflight phase)",
                suffix: "a",
                start: TicketPhase::Backlog,
                expected: TicketPhase::Backlog,
                reservation: false,
            },
            Case {
                name: "Analysis → Backlog (no reservation)",
                suffix: "b",
                start: TicketPhase::Analysis,
                expected: TicketPhase::Backlog,
                reservation: false,
            },
            Case {
                name: "InDevelopment → ReadyForDevelopment (reservation=1)",
                suffix: "c",
                start: TicketPhase::InDevelopment,
                expected: TicketPhase::ReadyForDevelopment,
                reservation: true,
            },
            Case {
                name: "InDiagnostics → ReadyForDevelopment (reservation=1)",
                suffix: "d",
                start: TicketPhase::InDiagnostics,
                expected: TicketPhase::ReadyForDevelopment,
                reservation: true,
            },
            Case {
                name: "InSanitation → QaPassed (reservation=1)",
                suffix: "e",
                start: TicketPhase::InSanitation,
                expected: TicketPhase::QaPassed,
                reservation: true,
            },
            Case {
                name: "InQa → Reviewed (no reservation)",
                suffix: "f",
                start: TicketPhase::InQa,
                expected: TicketPhase::Reviewed,
                reservation: false,
            },
            Case {
                name: "InReview → DiagnosticsDone (no reservation)",
                suffix: "g",
                start: TicketPhase::InReview,
                expected: TicketPhase::DiagnosticsDone,
                reservation: false,
            },
        ];

        let (store, _tmp) = open_test_store().await;

        for case in &cases {
            let ws = test_ws_named(&format!("/{}", case.suffix), case.suffix);

            let id = make_ticket(&store, &ws, case.name, case.start).await;

            store.reset_inflight_tickets().await.expect("reset");

            let t = expect_ticket(&store, &id).await;
            assert_eq!(
                t.phase, case.expected,
                "Case '{}': unexpected status after reset",
                case.name,
            );
            assert_eq!(
                t.pipeline_reservation, case.reservation,
                "Case '{}': unexpected pipeline_reservation after reset",
                case.name,
            );
            assert!(
                t.assigned_to.is_none(),
                "Case '{}': assigned_to should be NULL after reset",
                case.name,
            );
        }
    }

    #[tokio::test]
    async fn test_claim_prefers_reserved_ticket() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        // Create two ReadyForDevelopment tickets
        let fresh_id = make_ticket(&store, &ws, "Fresh", TicketPhase::ReadyForDevelopment).await;
        let reserved_id =
            make_ticket(&store, &ws, "Reserved", TicketPhase::ReadyForDevelopment).await;

        // Set reservation on the second ticket
        store
            .transition_to(
                &reserved_id,
                Some(TicketPhase::ReadyForDevelopment),
                TicketPhase::ReadyForDevelopment,
                Some(true),
            )
            .await
            .expect("set reservation");

        // When claiming with require_clear_pipeline, the reserved ticket should be picked first
        let claimed = store
            .claim_ticket_in_workspace(
                TicketPhase::ReadyForDevelopment,
                TicketPhase::InDevelopment,
                "ws",
                true,
            )
            .await
            .expect("claim")
            .expect("should claim a ticket");
        assert_eq!(
            claimed.id, reserved_id,
            "Reserved ticket should be claimed before fresh one"
        );
        assert!(
            !claimed.pipeline_reservation,
            "Claim should clear pipeline_reservation"
        );

        // Verify the cleared reservation is persisted in the DB
        // (the returned Ticket struct already reflects the DB state, but
        // a separate re-read explicitly tests persistence).
        let reserved_db = expect_ticket(&store, &reserved_id).await;
        assert!(
            !reserved_db.pipeline_reservation,
            "Reservation should be 0 in DB after claim"
        );

        // After the reserved ticket is claimed (now InDevelopment, pipeline-blocking),
        // the fresh ticket is still at ReadyForDevelopment but cannot be claimed
        // because the pipeline is blocked. Verify the fresh ticket remains untouched.
        let fresh = expect_ticket(&store, &fresh_id).await;
        assert_eq!(
            fresh.phase,
            TicketPhase::ReadyForDevelopment,
            "Fresh ticket should still be at ReadyForDevelopment"
        );
        assert!(
            !fresh.pipeline_reservation,
            "Fresh ticket should have no reservation"
        );
    }

    #[tokio::test]
    async fn test_has_pipeline_blocker_reserved() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        // A fresh ReadyForDevelopment ticket should NOT be a blocker
        let id = make_ticket(&store, &ws, "Fresh", TicketPhase::ReadyForDevelopment).await;
        assert!(
            !store
                .has_pipeline_blocker_for_workspace("ws")
                .await
                .expect("check"),
            "Fresh ReadyForDevelopment ticket should not be a pipeline blocker"
        );

        // After setting reservation, it should be a blocker
        store
            .transition_to(
                &id,
                Some(TicketPhase::ReadyForDevelopment),
                TicketPhase::ReadyForDevelopment,
                Some(true),
            )
            .await
            .expect("set reservation");
        assert!(
            store
                .has_pipeline_blocker_for_workspace("ws")
                .await
                .expect("check"),
            "Reserved ReadyForDevelopment ticket should be a pipeline blocker"
        );

        // After removing reservation, it should not be a blocker
        store
            .transition_to(
                &id,
                Some(TicketPhase::ReadyForDevelopment),
                TicketPhase::ReadyForDevelopment,
                Some(false),
            )
            .await
            .expect("clear reservation");
        assert!(
            !store
                .has_pipeline_blocker_for_workspace("ws")
                .await
                .expect("check"),
            "Non-reserved ReadyForDevelopment ticket should not be a pipeline blocker again"
        );
    }

    /// Assert that [`BoardStore::has_active_tickets_excluding`] returns the
    /// expected value. Supports both static and formatted messages.
    async fn assert_active_excluding(
        store: &BoardStore,
        ws_name: &str,
        exclude_id: &str,
        expected: bool,
        msg: impl std::fmt::Display,
    ) {
        assert_eq!(
            store
                .has_active_tickets_excluding(ws_name, exclude_id)
                .await
                .expect("check"),
            expected,
            "{msg}"
        );
    }

    /// Create 5 tickets in non-active phases under workspace "ws_non" (/ws_non),
    /// returning their IDs.
    ///
    /// Non-active phases covered: Done, Cancelled, Failed, Planning, Backlog.
    /// Note: Analysis is also filtered out by the SQL query but is intentionally
    /// omitted here — it has its own dedicated test coverage elsewhere.
    async fn create_non_active_tickets(store: &BoardStore) -> Vec<String> {
        let ws = test_ws_named("/ws_non", "ws_non");
        vec![
            make_ticket(store, &ws, "Done", TicketPhase::Done).await,
            make_ticket(store, &ws, "Cancelled", TicketPhase::Cancelled).await,
            make_ticket(store, &ws, "Failed", TicketPhase::Failed).await,
            make_ticket(store, &ws, "Planning", TicketPhase::Planning).await,
            make_ticket(store, &ws, "Backlog", TicketPhase::Backlog).await,
        ]
    }

    /// Verify that [`BoardStore::has_active_tickets_excluding`] correctly identifies
    /// active tickets (PIPELINE_BLOCKING_STATUSES + ReadyForDevelopment) per workspace,
    /// excluding a specified ticket ID.
    ///
    /// Active tickets include all ReadyForDevelopment tickets regardless of
    /// `pipeline_reservation`, unlike [`has_pipeline_blocker_for_workspace`] which
    /// uses `require_reservation = true` (requires reservation=1). This is
    /// intentional — unstarted backlog tickets
    /// are considered active to suppress Done notifications until the pipeline
    /// is fully drained.
    #[tokio::test]
    async fn test_has_active_tickets_excluding() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        // Create one ticket per active status: all PIPELINE_BLOCKING_STATUSES + ReadyForDevelopment
        let rfd_id = make_ticket(&store, &ws, "RFD", TicketPhase::ReadyForDevelopment).await;
        let in_dev_id = make_ticket(&store, &ws, "InDev", TicketPhase::InDevelopment).await;
        let done_id = make_ticket(&store, &ws, "Done", TicketPhase::Done).await;
        let cancelled_id = make_ticket(&store, &ws, "Cancelled", TicketPhase::Cancelled).await;

        // All non-excluded active tickets are found
        assert_active_excluding(
            &store,
            "ws",
            &done_id,
            true,
            "Should find active tickets (RFD + InDev) when excluding Done",
        )
        .await;

        // Excluding an active ticket still finds another active ticket
        assert_active_excluding(
            &store,
            "ws",
            &rfd_id,
            true,
            "Should find InDev as active when excluding RFD",
        )
        .await;
        assert_active_excluding(
            &store,
            "ws",
            &in_dev_id,
            true,
            "Should find RFD as active when excluding InDev",
        )
        .await;

        // Non-active (Done, Cancelled) exclusion should still find active tickets
        for exclude in [&done_id, &cancelled_id] {
            assert_active_excluding(
                &store,
                "ws",
                exclude,
                true,
                "Non-active exclusion should still find active tickets",
            )
            .await;
        }

        // ReadyForDevelopment without reservation counts as active
        // (rfd_id already has no reservation — it was created with default)
        assert_active_excluding(
            &store,
            "ws",
            "nonexistent",
            true,
            "Should find active tickets for nonexistent exclude ID",
        )
        .await;

        // Different workspace — no tickets
        assert_active_excluding(
            &store,
            "other_ws",
            &rfd_id,
            false,
            "Should not find active tickets in unrelated workspace",
        )
        .await;

        // Workspace with only non-active tickets — Done, Cancelled, Failed, Planning, Backlog
        let non_active_ids = create_non_active_tickets(&store).await;
        for exclude in &non_active_ids {
            assert_active_excluding(
                &store,
                "ws_non",
                exclude,
                false,
                format!("Workspace with only non-active tickets should have no active tickets (excluded {exclude})"),
            )
            .await;
        }
        // Excluding a nonexistent ID in a non-active-only workspace also returns false
        assert_active_excluding(
            &store,
            "ws_non",
            "nonexistent",
            false,
            "No active tickets for nonexistent exclude ID in non-active-only workspace",
        )
        .await;
    }

    /// Verify that every non-transitory pipeline-blocking phase has a reset transition.
    ///
    /// [`PIPELINE_BLOCKING_STATUSES`] defines 9 phases; 5 of them (InDevelopment,
    /// InDiagnostics, InSanitation, InReview, InQa) have entries in
    /// [`RESET_TRANSITIONS`]. The remaining 4 phases
    /// ([`TRANSITORY_HANDOFF_PHASES`]) are transitory handoff states that the
    /// poller picks up within seconds — no agent is mid-execution in those states,
    /// so they don't need reset entries.
    ///
    /// This test does NOT assert the reverse direction (reset → pipeline blocker),
    /// because [`RESET_TRANSITIONS`] also includes `Analysis → Backlog`, and `Analysis`
    /// is intentionally not a pipeline blocker (it's a pre-flight phase).
    ///
    /// It also mechanically verifies that [`TRANSITORY_HANDOFF_PHASES`] is a subset of
    /// [`PIPELINE_BLOCKING_STATUSES`], ensuring the two sets stay in sync.
    #[test]
    fn test_pipeline_blockers_coverage() {
        // Verify that every transitory handoff phase is a pipeline blocker.
        for phase in TRANSITORY_HANDOFF_PHASES {
            assert!(
                PIPELINE_BLOCKING_STATUSES.contains(phase),
                "\
TRANSITORY_HANDOFF_PHASES contains `{phase}` which is not in \
PIPELINE_BLOCKING_STATUSES. Every transitory handoff phase must also \
be a pipeline blocker.\
                ",
            );
        }

        // Collect all `from` phases from BoardStore::RESET_TRANSITIONS for easy lookup.
        let reset_from: Vec<TicketPhase> = BoardStore::RESET_TRANSITIONS
            .iter()
            .map(|t| t.from)
            .collect();

        for phase in PIPELINE_BLOCKING_STATUSES {
            let has_reset = reset_from.contains(phase);
            assert!(
                has_reset || phase.is_transitory_handoff(),
                "\
PIPELINE_BLOCKING_STATUSES contains `{phase}` which has no corresponding \
entry in RESET_TRANSITIONS and is not a transitory handoff phase \
(see `TicketPhase::is_transitory_handoff`). Either add a reset transition to \
RESET_TRANSITIONS, or mark the phase as transitory handoff in that method \
with a comment explaining why no agent is mid-execution in that state.\
                ",
            );
        }
    }

    #[tokio::test]
    async fn test_claim_ticket_in_workspace() {
        let (store, _tmp) = open_test_store().await;

        // Create tickets in two different workspaces
        let ws_a = test_ws_named("/ws_a", "workspace_a");
        let ws_b = test_ws_named("/ws_b", "workspace_b");

        let id_a = make_ticket(&store, &ws_a, "Ticket A", TicketPhase::Backlog).await;

        let id_b = make_ticket(&store, &ws_b, "Ticket B", TicketPhase::Backlog).await;

        // Claim ticket from workspace A — should succeed
        let claimed_a = store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::InDevelopment,
                "workspace_a",
                false,
            )
            .await
            .expect("claim in ws_a")
            .expect("should claim ticket from ws_a");
        assert_eq!(claimed_a.id, id_a);
        assert_eq!(claimed_a.workspace_name, "workspace_a");
        assert_eq!(claimed_a.phase, TicketPhase::InDevelopment);
        assert!(claimed_a.assigned_to.is_none());

        // Claim from workspace A again — should return None (no more backlog tickets)
        assert!(
            store
                .claim_ticket_in_workspace(
                    TicketPhase::Backlog,
                    TicketPhase::InDevelopment,
                    "workspace_a",
                    false,
                )
                .await
                .expect("second claim in ws_a")
                .is_none(),
            "no more tickets to claim in ws_a"
        );

        // Claim ticket from workspace B — should still succeed (different workspace)
        let claimed_b = store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::InDevelopment,
                "workspace_b",
                false,
            )
            .await
            .expect("claim in ws_b")
            .expect("should claim ticket from ws_b");
        assert_eq!(claimed_b.id, id_b);
        assert_eq!(claimed_b.workspace_name, "workspace_b");
    }

    /// Table-driven tests for `claim_ticket_in_workspace` with `require_clear_pipeline: true`.
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn test_claim_ticket_in_workspace_if_pipeline_free() {
        /// The pipeline scenario for a single test case.
        enum Scenario {
            /// Blocker in the same workspace — claim should be blocked.
            SameWorkspace(TicketPhase),
            /// Blocker in a different workspace — claim should succeed.
            DifferentWorkspace(TicketPhase),
            /// No blocker — claim should succeed.
            NoBlocker,
        }

        struct Case {
            name: &'static str,
            /// Unique suffix for workspace names (isolates cases).
            suffix: &'static str,
            scenario: Scenario,
        }

        let cases = [
            Case {
                name: "blocked by same-workspace pipeline ticket",
                suffix: "blocked",
                scenario: Scenario::SameWorkspace(TicketPhase::InReview),
            },
            Case {
                name: "not blocked by cross-workspace pipeline ticket",
                suffix: "cross",
                scenario: Scenario::DifferentWorkspace(TicketPhase::InDevelopment),
            },
            Case {
                name: "no blocker succeeds",
                suffix: "none",
                scenario: Scenario::NoBlocker,
            },
        ];

        let (store, _tmp) = open_test_store().await;

        for case in &cases {
            let suffix = case.suffix;

            // Derive workspace names from the scenario.
            let (claim_ws_name, blocker_ws_name) = match &case.scenario {
                Scenario::DifferentWorkspace(_) => (
                    format!("ws_{suffix}_claimable"),
                    format!("ws_{suffix}_blocker"),
                ),
                // SameWorkspace and NoBlocker both use a single workspace name.
                Scenario::SameWorkspace(_) | Scenario::NoBlocker => {
                    let name = format!("ws_{suffix}");
                    (name.clone(), name)
                }
            };

            let expected_claim = !matches!(case.scenario, Scenario::SameWorkspace(_));

            let blocker_ws = test_ws_named(&format!("/{blocker_ws_name}"), &blocker_ws_name);
            let claimable_ws = test_ws_named(&format!("/{claim_ws_name}"), &claim_ws_name);

            // Create a pipeline blocker (if any)
            if let Scenario::SameWorkspace(phase) | Scenario::DifferentWorkspace(phase) =
                &case.scenario
            {
                // When blocker and claimable share a workspace, place the
                // blocker in the claimable's workspace (they are the same).
                let blocker_target = match &case.scenario {
                    Scenario::DifferentWorkspace(_) => &blocker_ws,
                    Scenario::SameWorkspace(_) => &claimable_ws,
                    // Not reachable: NoBlocker is guarded by the enclosing if-let.
                    Scenario::NoBlocker => unreachable!(),
                };
                make_ticket(&store, blocker_target, "Blocker", *phase).await;
            }

            // Create a claimable ticket
            let id = make_ticket(
                &store,
                &claimable_ws,
                "Claimable",
                TicketPhase::ReadyForDevelopment,
            )
            .await;

            // Claim with require_clear_pipeline=true
            let claimed = store
                .claim_ticket_in_workspace(
                    TicketPhase::ReadyForDevelopment,
                    TicketPhase::InDevelopment,
                    &claim_ws_name,
                    true,
                )
                .await
                .expect("claim should not error");

            if expected_claim {
                let claimed = claimed.expect("should claim ticket");
                assert_eq!(claimed.id, id, "Case '{}': wrong ticket id", case.name);
                assert_eq!(
                    claimed.phase,
                    TicketPhase::InDevelopment,
                    "Case '{}': wrong status after claim",
                    case.name
                );
            } else {
                assert!(
                    claimed.is_none(),
                    "Case '{}': claim should be blocked",
                    case.name
                );
            }
        }
    }

    // ── Prerequisites ────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_ticket_with_prerequisites() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        // Create prerequisite tickets first
        let p1 = make_ticket(&store, &ws, "P1", TicketPhase::Backlog).await;
        let p2 = make_ticket(&store, &ws, "P2", TicketPhase::Backlog).await;

        // Create a ticket depending on both
        let deps = vec![p1.clone(), p2.clone()];
        let id = TicketBuilder::new(&store, &ws)
            .title("Dependent")
            .desc("needs both")
            .prereqs(&deps)
            .create()
            .await
            .expect("create dependent");

        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert_eq!(ticket.prerequisites.len(), 2);
        assert!(ticket.prerequisites.contains(&p1));
        assert!(ticket.prerequisites.contains(&p2));
    }

    /// Table-driven tests for `create_ticket` with invalid prerequisite inputs.
    #[tokio::test]
    async fn test_create_ticket_invalid_inputs() {
        enum Scenario {
            /// Prerequisite references a nonexistent ticket.
            NonExistent,
            /// Prerequisite references the ticket about to be created (self-reference).
            SelfReference,
            /// Prerequisite references a ticket in a different workspace.
            CrossWorkspace,
        }

        struct Case {
            name: &'static str,
            scenario: Scenario,
        }

        let cases = [
            Case {
                name: "nonexistent prerequisite",
                scenario: Scenario::NonExistent,
            },
            Case {
                name: "self-referencing prerequisite",
                scenario: Scenario::SelfReference,
            },
            Case {
                name: "cross-workspace prerequisite",
                scenario: Scenario::CrossWorkspace,
            },
        ];

        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");
        let ws_b = test_ws_named("/ws_b", "ws_b");
        // Isolated workspace for SelfReference — its own counter avoids
        // ordering dependencies with CrossWorkspace which also creates
        // seeds in `ws`.
        let ws_sr = test_ws_named("/ws_sr", "ws_sr");

        for case in &cases {
            let expected_error = match case.scenario {
                Scenario::NonExistent => "not found",
                Scenario::SelfReference => "cannot depend on itself",
                Scenario::CrossWorkspace => "Cross-workspace",
            };

            // Create a seed ticket for scenarios that need one.
            // NonExistent: no seed needed — uses a nonexistent ID directly.
            // CrossWorkspace: create a ticket in `ws` to use as a
            //   cross-workspace prerequisite for a ticket in `ws_b`.
            // SelfReference: create exactly one ticket in its own workspace
            //   `ws_sr` to advance the counter so the next ticket will
            //   have ID `ws_sr-1`.
            let seed: Option<String> = match &case.scenario {
                Scenario::NonExistent => None,
                Scenario::CrossWorkspace => {
                    Some(make_ticket(&store, &ws, "Existing", TicketPhase::Backlog).await)
                }
                Scenario::SelfReference => {
                    Some(make_ticket(&store, &ws_sr, "First", TicketPhase::Backlog).await)
                }
            };

            let target_ws = match case.scenario {
                Scenario::CrossWorkspace => &ws_b,
                Scenario::SelfReference => &ws_sr,
                Scenario::NonExistent => &ws,
            };

            // Build prerequisites for each scenario.
            // NonExistent: a nonexistent ticket ID.
            // CrossWorkspace: the ticket created in `ws` (different workspace).
            // SelfReference: hardcoded `{ws_sr}-1` — the ID the next ticket
            //   receives in the isolated workspace, creating a self-reference.
            let prereqs: Vec<String> = match &case.scenario {
                Scenario::NonExistent => vec!["nonexistent-1".to_string()],
                Scenario::CrossWorkspace => {
                    vec![seed.clone().expect("seed must exist for CrossWorkspace")]
                }
                Scenario::SelfReference => {
                    // After creating exactly one seed ticket above, the next
                    // ticket in this isolated workspace receives ID `ws_sr-1`.
                    vec![format!("{}-1", ws_sr.name)]
                }
            };

            let err = TicketBuilder::new(&store, target_ws)
                .title("New")
                .prereqs(&prereqs)
                .create()
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains(expected_error),
                "Case '{}': expected error containing '{}', got: {err}",
                case.name,
                expected_error
            );
        }
    }

    /// Create a 2-ticket dependency chain: A (no prereqs) → B (depends on A).
    async fn create_chain_ab(store: &BoardStore, ws: Workspace) -> (String, String) {
        let a = make_ticket(store, &ws, "A", TicketPhase::Backlog).await;
        let b = TicketBuilder::new(store, &ws)
            .title("B")
            .desc("depends on A")
            .prereqs(std::slice::from_ref(&a))
            .create()
            .await
            .expect("create b");
        (a, b)
    }

    #[tokio::test]
    async fn test_circular_dependency_rejected() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        let (a, b) = create_chain_ab(&store, ws.clone()).await;

        // Verify that A→B chain works: creating a ticket with both A and B
        // as prerequisites is NOT a cycle (it's just redundant, since A is
        // already transitively required through B). This should succeed.
        let _c = TicketBuilder::new(&store, &ws)
            .title("C")
            .desc("depends on both")
            .prereqs(&[a.clone(), b.clone()])
            .create()
            .await
            .expect("create c — A and B as prereqs is not a cycle");
    }

    #[tokio::test]
    async fn test_transitive_prerequisites_block() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        let (a, b) = create_chain_ab(&store, ws.clone()).await;

        // C depends on B
        let c = TicketBuilder::new(&store, &ws)
            .title("C")
            .desc("top")
            .prereqs(std::slice::from_ref(&b))
            .create()
            .await
            .expect("create c");

        // C should be blocked even though B is done — A is still blocking
        // First claim: A is the only unblocked one
        let claimed = store
            .claim_ticket_in_workspace(TicketPhase::Backlog, TicketPhase::Analysis, "ws", false)
            .await
            .expect("claim")
            .expect("should claim A");
        assert_eq!(claimed.id, a);

        // B should still be blocked — A is in Analysis, not Done yet
        let second = store
            .claim_ticket_in_workspace(TicketPhase::Backlog, TicketPhase::Analysis, "ws", false)
            .await
            .expect("claim");
        assert!(
            second.is_none(),
            "B should be blocked because A is in Analysis, not Done"
        );

        // Move A to done
        store
            .transition_to(&a, None, TicketPhase::Done, None)
            .await
            .expect("done a");

        // Now B should be claimable
        let claimed2 = store
            .claim_ticket_in_workspace(TicketPhase::Backlog, TicketPhase::Analysis, "ws", false)
            .await
            .expect("claim")
            .expect("should claim B");
        assert_eq!(claimed2.id, b);

        // Move B to done
        store
            .transition_to(&b, None, TicketPhase::Done, None)
            .await
            .expect("done b");

        // Now C should be claimable
        let claimed3 = store
            .claim_ticket_in_workspace(TicketPhase::Backlog, TicketPhase::Analysis, "ws", false)
            .await
            .expect("claim")
            .expect("should claim C");
        assert_eq!(claimed3.id, c);
    }

    async fn assert_archive_empty_db(store: &BoardStore) {
        let count = store
            .archive_stale_cancelled(1)
            .await
            .expect("archive_stale_cancelled");
        assert_eq!(count, 0, "Empty DB stale archive should return 0");
        let count = store
            .archive_all_done_and_cancelled(None)
            .await
            .expect("archive_all_done_and_cancelled");
        assert_eq!(count, 0, "Empty DB all archive should return 0");
    }

    #[tokio::test]
    async fn test_archive_stale_cancelled() {
        let (store, _tmp) = open_test_store().await;
        assert_archive_empty_db(&store).await;

        let ws = test_ws_named("/ws", "ws");

        // Ticket 1: cancelled, old (2h) → should be archived
        let old_cancelled_id =
            make_ticket(&store, &ws, "old-cancelled", TicketPhase::Backlog).await;

        let two_hours_ago = (Utc::now() - chrono::Duration::hours(2)).to_rfc3339();

        store
            .transition_to(&old_cancelled_id, None, TicketPhase::Cancelled, None)
            .await
            .expect("cancel");
        store
            .conn
            .execute(
                "UPDATE tickets SET updated_at = ?1 WHERE id = ?2",
                crate::turso::params![two_hours_ago.clone(), old_cancelled_id.clone()],
            )
            .await
            .expect("backdate");

        // Ticket 2: cancelled, fresh → should NOT be archived
        let fresh_cancelled_id =
            make_ticket(&store, &ws, "fresh-cancelled", TicketPhase::Backlog).await;
        store
            .transition_to(&fresh_cancelled_id, None, TicketPhase::Cancelled, None)
            .await
            .expect("cancel");
        // No backdating — updated_at is now.

        // Ticket 3: not cancelled (Backlog), old → should NOT be archived
        let old_backlog_id = make_ticket(&store, &ws, "old-backlog", TicketPhase::Backlog).await;
        store
            .conn
            .execute(
                "UPDATE tickets SET updated_at = ?1 WHERE id = ?2",
                crate::turso::params![two_hours_ago.clone(), old_backlog_id.clone()],
            )
            .await
            .expect("backdate");

        // Act
        let count = store
            .archive_stale_cancelled(1)
            .await
            .expect("archive_stale_cancelled");
        assert_eq!(count, 1, "should archive only the old cancelled ticket");

        // Assert
        let old_cancelled = crate::util::test::expect_ticket(&store, &old_cancelled_id).await;
        assert!(
            old_cancelled.is_archived,
            "old cancelled ticket should be archived"
        );
        assert_eq!(old_cancelled.phase, TicketPhase::Cancelled);

        let fresh_cancelled = crate::util::test::expect_ticket(&store, &fresh_cancelled_id).await;
        assert!(
            !fresh_cancelled.is_archived,
            "fresh cancelled ticket should NOT be archived"
        );
        assert_eq!(fresh_cancelled.phase, TicketPhase::Cancelled);

        let old_backlog = crate::util::test::expect_ticket(&store, &old_backlog_id).await;
        assert!(
            !old_backlog.is_archived,
            "old non-cancelled ticket should NOT be archived"
        );
        assert_eq!(old_backlog.phase, TicketPhase::Backlog);
    }

    #[tokio::test]
    async fn test_archive_all_done_and_cancelled() {
        let (store, _tmp) = open_test_store().await;
        assert_archive_empty_db(&store).await;

        let ws = test_ws_named("/ws", "ws");

        // Create three tickets: one Done, one Cancelled, one Backlog.
        let done_id = make_ticket(&store, &ws, "done", TicketPhase::Backlog).await;
        store
            .transition_to(&done_id, None, TicketPhase::Done, None)
            .await
            .expect("set done");

        let cancelled_id = make_ticket(&store, &ws, "cancelled", TicketPhase::Backlog).await;
        store
            .transition_to(&cancelled_id, None, TicketPhase::Cancelled, None)
            .await
            .expect("cancel");

        let backlog_id = make_ticket(&store, &ws, "backlog", TicketPhase::Backlog).await;
        // Leave in Backlog.

        // Act
        let count = store
            .archive_all_done_and_cancelled(None)
            .await
            .expect("archive");
        assert_eq!(count, 2, "should archive Done and Cancelled tickets");

        // Assert
        let done_ticket = crate::util::test::expect_ticket(&store, &done_id).await;
        assert!(done_ticket.is_archived, "Done ticket should be archived");
        assert_eq!(done_ticket.phase, TicketPhase::Done);

        let cancelled_ticket = crate::util::test::expect_ticket(&store, &cancelled_id).await;
        assert!(
            cancelled_ticket.is_archived,
            "Cancelled ticket should be archived"
        );
        assert_eq!(cancelled_ticket.phase, TicketPhase::Cancelled);

        let backlog_ticket = crate::util::test::expect_ticket(&store, &backlog_id).await;
        assert!(
            !backlog_ticket.is_archived,
            "Backlog ticket should NOT be archived"
        );
        assert_eq!(backlog_ticket.phase, TicketPhase::Backlog);
    }

    #[tokio::test]
    async fn test_archive_all_done_and_cancelled_workspace_filter() {
        let (store, _tmp) = open_test_store().await;

        // Create a done ticket in ws1 and another in ws2.
        let id1 = make_ticket(
            &store,
            &test_ws_named("/ws1", "ws1"),
            "Test",
            TicketPhase::Backlog,
        )
        .await;
        store
            .transition_to(&id1, None, TicketPhase::Done, None)
            .await
            .expect("set done");
        let id2 = make_ticket(
            &store,
            &test_ws_named("/ws2", "ws2"),
            "Test",
            TicketPhase::Backlog,
        )
        .await;
        store
            .transition_to(&id2, None, TicketPhase::Done, None)
            .await
            .expect("set done");

        // Archive only ws1.
        let count = store
            .archive_all_done_and_cancelled(Some("ws1"))
            .await
            .expect("archive_all_done_and_cancelled");
        assert_eq!(count, 1, "Should archive only ws1 ticket");

        let ticket1 = crate::util::test::expect_ticket(&store, &id1).await;
        assert!(ticket1.is_archived, "ws1 ticket should be archived");
        assert_eq!(
            ticket1.phase,
            TicketPhase::Done,
            "ws1 phase should remain Done"
        );

        let ticket2 = crate::util::test::expect_ticket(&store, &id2).await;
        assert!(!ticket2.is_archived, "ws2 ticket should NOT be archived");
        assert_eq!(
            ticket2.phase,
            TicketPhase::Done,
            "ws2 ticket should remain Done"
        );
    }

    #[tokio::test]
    async fn test_count_by_status_excludes_archived() {
        let (store, _tmp) = open_test_store().await;
        // Create a ticket and set it to Done.
        let id = make_ticket(
            &store,
            &test_ws_named("/ws", "ws"),
            "Test",
            TicketPhase::Backlog,
        )
        .await;
        store
            .transition_to(&id, None, TicketPhase::Done, None)
            .await
            .expect("set done");

        // Before archiving, count includes the Done ticket.
        let count_before = store
            .count_by_status(TicketPhase::Done, None)
            .await
            .expect("count before");
        assert_eq!(count_before, 1, "Should count Done ticket before archive");

        // Archive done tickets.
        let archived = store
            .archive_all_done_and_cancelled(None)
            .await
            .expect("archive");
        assert_eq!(archived, 1, "Should have archived 1 ticket");

        // After archiving, count_by_status(Done) should return 0.
        let count_after = store
            .count_by_status(TicketPhase::Done, None)
            .await
            .expect("count after");
        assert_eq!(count_after, 0, "Should not count archived Done tickets");

        // Archived tickets with other statuses should also be excluded.
        let count_cancelled = store
            .count_by_status(TicketPhase::Cancelled, None)
            .await
            .expect("count cancelled");
        assert_eq!(count_cancelled, 0, "No Cancelled tickets exist");
    }

    #[tokio::test]
    async fn test_create_ticket_tool_with_prerequisites() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::BOARD.get().unwrap();
        let ws = test_ws("/tmp/test_ws_tool_prereqs");

        // Create a prerequisite via the store directly
        let p_id = make_ticket(store, &ws, "Pre", TicketPhase::Backlog).await;

        let tool = crate::tools::CreateTicketTool::new("test");
        let args = serde_json::json!({
            "title": "Test with prereqs",
            "description": "depends on something",
            "prerequisites": [p_id],
        });
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains(&p_id),
            "Output should mention prerequisite ID"
        );
    }

    #[tokio::test]
    async fn test_supersede_and_create_basic() {
        init_test_stores().await;
        let store = crate::board::BOARD.get().unwrap();
        let ws = test_ws_named("/ws", "ws");
        let old_id = make_ticket(store, &ws, "Test", TicketPhase::Backlog).await;

        // Supersede it
        let new_id = TicketBuilder::new(store, &ws)
            .title("New title")
            .desc("New desc")
            .supersede(&old_id)
            .await
            .expect("supersede");

        // Old ticket is cancelled and points forward to the new ticket
        let old = expect_ticket(store, &old_id).await;
        assert_superseded_ticket(&old);
        assert_eq!(
            old.superseded_by.as_deref(),
            Some(new_id.as_str()),
            "superseded ticket should point to the new ticket"
        );

        // New ticket is in Backlog and links to old
        let new = expect_ticket(store, &new_id).await;
        assert_eq!(new.phase, TicketPhase::Backlog);
        assert_eq!(new.supersedes.as_deref(), Some(old_id.as_str()));
        assert_eq!(new.title, "New title");
    }

    #[tokio::test]
    async fn test_supersede_rewires_only_matching_prerequisite() {
        init_test_stores().await;
        let store = crate::board::BOARD.get().unwrap();
        let ws = test_ws_named("/ws", "ws");

        // Create ticket A (will be superseded) and ticket C (independent).
        let a_id = make_ticket(store, &ws, "A", TicketPhase::Backlog).await;
        let c_id = make_ticket(store, &ws, "C", TicketPhase::Backlog).await;

        // Create ticket B that depends on both A and C.
        let b_id = TicketBuilder::new(store, &ws)
            .title("B")
            .desc("dep on A and C")
            .prereqs(&[a_id.clone(), c_id.clone()])
            .create()
            .await
            .expect("create B");

        // Create ticket D with no prerequisites — should be untouched.
        let d_id = make_ticket(store, &ws, "D", TicketPhase::Backlog).await;

        // Supersede A → A2.
        let supersede_id = TicketBuilder::new(store, &ws)
            .title("A2")
            .desc("refined")
            .supersede(&a_id)
            .await
            .expect("supersede");

        // B's prerequisites: A→A2, C unchanged.
        let b = store
            .get_ticket(&b_id)
            .await
            .expect("get B")
            .expect("B exists");
        assert_eq!(b.prerequisites, vec![supersede_id.clone(), c_id.clone()]);

        // D untouched.
        let d = store
            .get_ticket(&d_id)
            .await
            .expect("get D")
            .expect("D exists");
        assert!(d.prerequisites.is_empty());
    }

    /// Table-driven tests for `supersede_and_create` with invalid inputs.
    #[tokio::test]
    async fn test_supersede_invalid_inputs() {
        enum Scenario {
            /// Supersede a nonexistent ticket.
            NonExistent,
            /// Supersede a ticket from a different workspace.
            CrossWorkspace,
            /// Supersede a ticket with a self-referencing prerequisite.
            SelfReference,
        }

        struct Case {
            name: &'static str,
            scenario: Scenario,
        }

        let cases = [
            Case {
                name: "nonexistent original",
                scenario: Scenario::NonExistent,
            },
            Case {
                name: "cross-workspace supersede",
                scenario: Scenario::CrossWorkspace,
            },
            Case {
                name: "self-referencing prerequisites",
                scenario: Scenario::SelfReference,
            },
        ];

        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");
        let ws_b = test_ws_named("/ws_b", "ws_b");

        for case in &cases {
            let expected_error = match case.scenario {
                Scenario::NonExistent => "not found",
                Scenario::CrossWorkspace => "Cross-workspace",
                Scenario::SelfReference => "supersede and depend",
            };

            let original_id = match case.scenario {
                Scenario::NonExistent => None,
                Scenario::CrossWorkspace | Scenario::SelfReference => {
                    let id = make_ticket(&store, &ws, "A", TicketPhase::Backlog).await;
                    Some(id)
                }
            };

            let target_ws = match case.scenario {
                Scenario::CrossWorkspace => &ws_b,
                Scenario::NonExistent | Scenario::SelfReference => &ws,
            };
            let supersede_id: &str = original_id.as_deref().unwrap_or("nonexistent");
            // prereqs include the original id only for SelfReference.
            let prereqs: Vec<String> = match &case.scenario {
                Scenario::SelfReference => {
                    vec![
                        original_id
                            .clone()
                            .expect("original must exist for SelfReference"),
                    ]
                }
                Scenario::NonExistent | Scenario::CrossWorkspace => vec![],
            };

            let err = TicketBuilder::new(&store, target_ws)
                .title("New")
                .prereqs(&prereqs)
                .supersede(supersede_id)
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains(expected_error),
                "Case '{}': expected error containing '{}', got: {err}",
                case.name,
                expected_error
            );
        }
    }

    #[tokio::test]
    async fn test_supersede_tool() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::BOARD.get().unwrap();
        let ws = test_ws("/tmp/test_ws_supersede_tool");

        // Create old ticket
        let old_id = make_ticket(store, &ws, "Old", TicketPhase::Backlog).await;

        let tool = crate::tools::CreateTicketTool::new("test");
        let args = serde_json::json!({
            "title": "Refined",
            "description": "refined desc",
            "supersede": old_id,
        });
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains("Superseded"),
            "Output should say Superseded: {result}"
        );
        assert!(
            result.contains(&old_id),
            "Output should mention old ID: {result}"
        );

        // Verify old is cancelled
        let old = expect_ticket(store, &old_id).await;
        assert_superseded_ticket(&old);
    }

    #[tokio::test]
    async fn test_supersede_already_cancelled() {
        init_test_stores().await;
        let store = crate::board::BOARD.get().unwrap();
        let ws = test_ws_named("/ws", "ws");
        let old_id = make_ticket(store, &ws, "Test", TicketPhase::Backlog).await;

        // Cancel the ticket
        store
            .transition_to(&old_id, None, TicketPhase::Cancelled, None)
            .await
            .expect("cancel");

        // Superseding an already-cancelled ticket should work
        let new_id = TicketBuilder::new(store, &ws)
            .title("Refined")
            .supersede(&old_id)
            .await
            .expect("supersede already-cancelled");

        let new = expect_ticket(store, &new_id).await;
        assert_eq!(new.supersedes.as_deref(), Some(old_id.as_str()));

        // Old ticket should also be archived immediately.
        let old = expect_ticket(store, &old_id).await;
        assert_superseded_ticket(&old);
    }

    // ── set_commit_info ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_set_commit_info() {
        // Successfully set commit info
        let (store, _tmp, id) = setup().await;

        store
            .set_commit_info(&id, "abcdef0123456789abcdef0123456789abcd0123", 10, 5)
            .await
            .expect("set commit info");

        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert_eq!(
            ticket.commit_hash.as_deref(),
            Some("abcdef0123456789abcdef0123456789abcd0123")
        );
        assert_eq!(ticket.lines_added, Some(10));
        assert_eq!(ticket.lines_removed, Some(5));

        // Non-existent ticket fails with appropriate error
        let (store2, _tmp2) = open_test_store().await;
        let result = store2
            .set_commit_info(
                "nonexistent",
                "0000000000000000000000000000000000000000",
                0,
                0,
            )
            .await;

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent"),
            "error should mention ticket id: {msg}"
        );
    }

    #[tokio::test]
    async fn test_transactional_triple_write() {
        for should_commit in [false, true] {
            // Exercise the full pattern used by commit_and_transition_ticket:
            // all three _tx writes (set_commit_info_tx, transition_to_tx,
            // add_comment_tx) in one transaction → commit → all visible
            // (or rollback → none persist).  This is the sole transactional
            // test for set_commit_info_tx (its standalone test was removed
            // as subsumed); the commit_hash, lines_added, and lines_removed
            // assertions below verify its behavior under both commit and
            // rollback, complementing the non-transactional coverage in
            // test_set_commit_info.
            // Now delegates to the real production method BoardStore::finalize_done_tx.
            let (store, _tmp, id) = setup().await;
            store
                .transition_to(&id, None, TicketPhase::QaPassed, None)
                .await
                .unwrap();

            let tx = store.conn.begin_tx().await.unwrap();
            BoardStore::finalize_done_tx(
                &tx,
                &id,
                "abcdef0123456789abcdef0123456789abcd0123",
                10,
                5,
                "triple write comment",
                TicketPhase::QaPassed,
            )
            .await
            .unwrap();

            let label = if should_commit { "commit" } else { "rollback" };
            if should_commit {
                tx.commit().await.unwrap();
            } else {
                tx.rollback().await.unwrap();
            }

            let ticket = crate::util::test::expect_ticket(&store, &id).await;
            let comments = store.get_comments(&id).await.expect("get comments");
            if should_commit {
                // All three changes should be visible.
                assert_eq!(
                    ticket.commit_hash.as_deref(),
                    Some("abcdef0123456789abcdef0123456789abcd0123"),
                    "({label}) commit_hash",
                );
                assert_eq!(ticket.lines_added, Some(10), "({label}) lines_added");
                assert_eq!(ticket.lines_removed, Some(5), "({label}) lines_removed");
                assert_eq!(ticket.phase, TicketPhase::Done, "({label}) phase");
                assert_eq!(comments.len(), 1, "({label}) comments.len");
                assert_eq!(
                    comments[0].content, "triple write comment",
                    "({label}) comment content"
                );
            } else {
                // None of the three changes should be visible.
                assert_eq!(
                    ticket.commit_hash, None,
                    "({label}) commit_hash after rollback"
                );
                assert_eq!(
                    ticket.lines_added, None,
                    "({label}) lines_added after rollback"
                );
                assert_eq!(
                    ticket.lines_removed, None,
                    "({label}) lines_removed after rollback"
                );
                assert_eq!(
                    ticket.phase,
                    TicketPhase::QaPassed,
                    "({label}) phase after rollback",
                );
                assert_eq!(comments.len(), 0, "({label}) comments.len after rollback");
            }
        }
    }

    // ── parse_prereqs unit tests ──

    #[test]
    fn test_parse_prereqs() {
        // ── Valid JSON cases ──
        let valid: &[(&str, &[&str])] = &[
            ("[]", &[] as &[&str]),
            (r#"["a","b","c"]"#, &["a", "b", "c"]),
        ];
        for (input, expected) in valid {
            let got = parse_prereqs(input).expect("should parse valid JSON");
            assert_eq!(got, *expected, "input: {input:?}");
        }

        // ── Invalid / corrupt JSON cases ──
        let invalid: &[&str] = &["", "not valid json {{{", r#"{"key":"value"}"#, "[1, 2, 3]"];
        for input in invalid {
            let err = parse_prereqs(input).unwrap_err();
            assert!(
                err.to_string().contains("Corrupt prerequisites JSON"),
                "input {input:?}: expected 'Corrupt prerequisites JSON' error, got: {err}",
            );
        }

        // ── Long ASCII input (>200 bytes) — preview truncated with ellipsis ──
        let long = format!(r#""{}...""#, "x".repeat(500));
        let msg = parse_prereqs(&long).unwrap_err().to_string();
        assert!(
            msg.contains('…'),
            "long input should produce truncated preview: {msg}"
        );
        assert!(
            msg.len() < 500,
            "truncated message should be <500 chars, got len={}",
            msg.len()
        );

        // ── Multi-byte character straddling byte 200 — no panic on truncation ──
        // Without floor_char_boundary, `&raw[..200]` would panic on the mid-char slice.
        let raw = format!("{}éééééééééémore", "x".repeat(199));
        assert!(raw.len() > 200, "need raw longer than 200 chars");
        // Verify byte 200 is indeed within a multi-byte character (not a boundary).
        assert!(
            !raw.is_char_boundary(200),
            "byte 200 must be mid-character for this test to be meaningful"
        );
        let msg = parse_prereqs(&raw).unwrap_err().to_string();
        assert!(
            msg.contains('…'),
            "multi-byte input should produce truncated preview: {msg}"
        );
        assert!(
            msg.len() < raw.len() + 50,
            "message too long after truncation: len={}, raw.len()={}",
            msg.len(),
            raw.len()
        );
        assert!(
            msg.contains("Corrupt prerequisites JSON"),
            "should mention corrupt JSON: {msg}"
        );
    }

    // ── Integration test: corrupt prerequisites in the database ──

    #[tokio::test]
    async fn corrupt_prerequisites_causes_get_ticket_error() {
        let (store, _tmp, id) = setup().await;

        // Directly corrupt the prerequisites column via raw SQL
        store
            .conn
            .execute(
                "UPDATE tickets SET prerequisites = ?1 WHERE id = ?2",
                crate::turso::params!["{not valid json}", id.clone()],
            )
            .await
            .expect("corrupt update");

        // get_ticket should now return an error
        let result = store.get_ticket(&id).await;
        assert!(
            result.is_err(),
            "get_ticket should fail when prerequisites are corrupt"
        );
        let err = result.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Corrupt prerequisites JSON"),
            "error should mention corrupt JSON: {msg}"
        );
        assert!(
            msg.contains(&id),
            "error should include ticket ID {id}: {msg}"
        );
    }

    #[tokio::test]
    async fn corrupt_prerequisites_causes_list_all_tickets_error() {
        let (store, _tmp, id) = setup().await;
        // Corrupt its prerequisites
        store
            .conn
            .execute(
                "UPDATE tickets SET prerequisites = ?1 WHERE id = ?2",
                crate::turso::params!["garbage{{{", id.clone()],
            )
            .await
            .expect("corrupt update");

        // list_all_tickets should fail entirely
        let result = store.list_all_tickets(Some("ws"), None).await;
        assert!(
            result.is_err(),
            "list_all_tickets should fail when any ticket has corrupt prerequisites"
        );
    }

    // ── claim_diagnostics tests ──

    /// Table-driven tests for `claim_diagnostics` covering success,
    /// pre-assignment rejection, wrong-phase rejection, and idempotency.
    #[tokio::test]
    async fn test_claim_diagnostics() {
        enum Scenario {
            /// Ticket is unassigned and in InDiagnostics — claim should succeed.
            Success,
            /// Ticket is already assigned — claim should fail.
            AlreadyAssigned,
            /// Ticket is in a different phase — claim should fail.
            WrongPhase,
        }

        struct Case {
            name: &'static str,
            scenario: Scenario,
        }

        let cases = [
            Case {
                name: "unassigned in diagnostics succeeds",
                scenario: Scenario::Success,
            },
            Case {
                name: "already assigned fails",
                scenario: Scenario::AlreadyAssigned,
            },
            Case {
                name: "wrong phase fails",
                scenario: Scenario::WrongPhase,
            },
        ];

        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        for (i, case) in cases.iter().enumerate() {
            let title = format!("claim-{i}");
            let id = make_ticket(&store, &ws, &title, TicketPhase::Backlog).await;

            if matches!(case.scenario, Scenario::Success | Scenario::AlreadyAssigned) {
                store
                    .transition_to(&id, None, TicketPhase::InDiagnostics, None)
                    .await
                    .expect("transition to InDiagnostics");
            }
            if matches!(case.scenario, Scenario::AlreadyAssigned) {
                store
                    .set_assigned_to(&id, Some(DIAGNOSTICS_ROLE))
                    .await
                    .expect("set_assigned_to");
            }

            let claimed = store
                .claim_diagnostics(&id)
                .await
                .expect("claim_diagnostics");

            match case.scenario {
                Scenario::Success => {
                    assert!(claimed, "Case '{}': expected claim to succeed", case.name);

                    // Verify post-claim state.
                    let ticket = crate::util::test::expect_ticket(&store, &id).await;
                    assert_eq!(
                        ticket.assigned_to.as_deref(),
                        Some(DIAGNOSTICS_ROLE),
                        "Case '{}': assignee should be set",
                        case.name
                    );
                    assert_eq!(
                        ticket.phase,
                        TicketPhase::InDiagnostics,
                        "Case '{}': phase should remain InDiagnostics",
                        case.name
                    );

                    // Verify idempotency (second claim returns false).
                    let second = store.claim_diagnostics(&id).await.expect("second claim");
                    assert!(
                        !second,
                        "Case '{}': second claim should return false (idempotent)",
                        case.name
                    );
                }
                Scenario::AlreadyAssigned | Scenario::WrongPhase => {
                    assert!(!claimed, "Case '{}': expected claim to fail", case.name);
                }
            }
        }
    }

    // ── claim_sanitation tests ──

    /// Table-driven tests for `claim_sanitation` covering success (QaPassed),
    /// wrong-phase rejection, and assigned_to verification on successful claim.
    #[tokio::test]
    async fn test_claim_sanitation() {
        struct Case {
            name: &'static str,
            phase: TicketPhase,
            expected_claim: bool,
        }

        let cases = [
            Case {
                name: "qa_passed succeeds",
                phase: TicketPhase::QaPassed,
                expected_claim: true,
            },
            Case {
                name: "backlog (wrong phase) fails",
                phase: TicketPhase::Backlog,
                expected_claim: false,
            },
            Case {
                name: "in_development (wrong phase) fails",
                phase: TicketPhase::InDevelopment,
                expected_claim: false,
            },
        ];

        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        for (i, case) in cases.iter().enumerate() {
            let title = format!("san-claim-{i}");
            let id = make_ticket(&store, &ws, &title, case.phase).await;

            let claimed = store.claim_sanitation(&id).await.expect("claim_sanitation");
            assert_eq!(
                claimed, case.expected_claim,
                "Case '{}': unexpected claim result",
                case.name
            );

            if case.expected_claim {
                let ticket = crate::util::test::expect_ticket(&store, &id).await;
                assert_eq!(ticket.phase, TicketPhase::InSanitation);
                let expected_key =
                    crate::session::ticket_session_key(&id, crate::Role::Sanitation.as_str());
                assert_eq!(
                    ticket.assigned_to.as_deref(),
                    Some(expected_key.as_str()),
                    "Case '{}': assigned_to should be set to sanitation session key",
                    case.name
                );
            }
        }
    }

    #[tokio::test]
    async fn test_claim_sanitation_workspace_serialization() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");

        // First ticket in QaPassed
        let first_id = make_ticket(&store, &ws, "First", TicketPhase::QaPassed).await;

        // Second ticket in QaPassed — same workspace
        let second_id = make_ticket(&store, &ws, "Second", TicketPhase::QaPassed).await;

        // Claim the first — should succeed
        let first_claimed = store
            .claim_sanitation(&first_id)
            .await
            .expect("first claim");
        assert!(first_claimed, "first claim should succeed");

        // Claim the second while first is in InSanitation — should fail (serialized)
        let second_claimed = store
            .claim_sanitation(&second_id)
            .await
            .expect("second claim");
        assert!(
            !second_claimed,
            "second claim should be blocked while first ticket is in sanitation pipeline"
        );

        // Transition first ticket out of the sanitation pipeline entirely
        // (simulating the real flow: SanitationPassed → auto-commit → Done).
        // We transition directly to Done since SanitationPassed is also in the
        // blocked set, so moving to SanitationPassed alone wouldn't clear it.
        store
            .transition_to(&first_id, None, TicketPhase::Done, None)
            .await
            .expect("transition first to Done (clears sanitation pipeline)");

        // Now second claim should succeed
        let second_claimed_retry = store
            .claim_sanitation(&second_id)
            .await
            .expect("second claim retry");
        assert!(
            second_claimed_retry,
            "second claim should succeed after pipeline clears"
        );
    }

    #[tokio::test]
    async fn test_claim_sanitation_cross_workspace_serialization() {
        let (store, _tmp) = open_test_store().await;
        let ws_a = test_ws_named("/ws_a", "ws_a");
        let ws_b = test_ws_named("/ws_b", "ws_b");

        // One ticket in each workspace, both in QaPassed
        let id_a = make_ticket(&store, &ws_a, "Workspace A", TicketPhase::QaPassed).await;
        let id_b = make_ticket(&store, &ws_b, "Workspace B", TicketPhase::QaPassed).await;

        // Both should succeed independently (different workspaces)
        let claimed_a = store.claim_sanitation(&id_a).await.expect("claim a");
        assert!(claimed_a, "workspace A claim should succeed");

        let claimed_b = store.claim_sanitation(&id_b).await.expect("claim b");
        assert!(
            claimed_b,
            "workspace B claim should succeed independently of workspace A"
        );

        let ticket_a = crate::util::test::expect_ticket(&store, &id_a).await;
        assert_eq!(ticket_a.phase, TicketPhase::InSanitation);

        let ticket_b = crate::util::test::expect_ticket(&store, &id_b).await;
        assert_eq!(ticket_b.phase, TicketPhase::InSanitation);
    }

    #[tokio::test]
    async fn test_set_assigned_to_none() {
        // Successfully clear an assigned assignee
        let (store, _tmp, id) = setup().await;

        store
            .set_assigned_to(&id, Some(DIAGNOSTICS_ROLE))
            .await
            .expect("set_assigned_to");
        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert_eq!(ticket.assigned_to.as_deref(), Some(DIAGNOSTICS_ROLE));

        store
            .set_assigned_to(&id, None)
            .await
            .expect("set_assigned_to(None) should clear assignee");
        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        assert!(ticket.assigned_to.is_none(), "assigned_to should be NULL");

        // Idempotent: clearing an already-None assignee succeeds
        store
            .set_assigned_to(&id, None)
            .await
            .expect("second set_assigned_to(None) should also succeed");

        // Non-existent ticket fails
        let (store2, _tmp2) = open_test_store().await;
        let result = store2.set_assigned_to("nonexistent", None).await;
        assert!(
            result.is_err(),
            "set_assigned_to(None) on nonexistent ticket should fail"
        );
    }

    /// Round-trip test that exercises ALL column-index constants in
    /// [`ticket_from_row`] by creating a ticket, setting every mutable field
    /// via public API, then verifying every [`Ticket`] field (including
    /// `pipeline_reservation` via its SQL `DEFAULT 0`) survives the
    /// SELECT → `ticket_from_row` deserialization path.
    ///
    /// Serves as a regression test for ticket deserialization — the
    /// [`columns!`] macro ensures single-sourcing of [`TICKET_COLUMNS`]
    /// and [`COL_TICKET_*`], so column-order drift between them is
    /// structurally impossible. This test still exercises the full
    /// `ticket_from_row` deserialization path, including manual
    /// field-by-field extraction via `row.get::<Type>(COL_TICKET_*)`
    /// and default-value handling.
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn test_ticket_roundtrip_all_fields() {
        let (store, _tmp) = open_test_store().await;

        // Non-existent ticket returns None.
        let none = store.get_ticket("nonexistent").await.expect("get");
        assert!(none.is_none(), "non-existent ticket should return None");

        let ws = crate::workspace::test_ws_named("/test_ws", "test_workspace");

        // Create ticket with known values for every TICKET_COLUMNS position.
        let id = TicketBuilder::new(&store, &ws)
            .title("Roundtrip Title")
            .desc("Roundtrip description")
            .phase(TicketPhase::Backlog)
            .reporter("test_reporter")
            .create()
            .await
            .expect("create_ticket");

        // Read back BEFORE setting any mutable fields — verify fresh-ticket defaults
        // (None for assigned_to, commit_hash, lines_added, lines_removed; empty comments).
        let fresh = store
            .get_ticket(&id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");
        assert_eq!(fresh.title, "Roundtrip Title", "fresh title");
        assert_eq!(
            fresh.description, "Roundtrip description",
            "fresh description"
        );
        assert_eq!(fresh.phase, TicketPhase::Backlog, "fresh phase");
        assert!(
            fresh.assigned_to.is_none(),
            "fresh ticket should have no assigned_to"
        );
        assert!(
            fresh.comments.is_empty(),
            "fresh ticket should have no comments"
        );
        assert!(
            fresh.commit_hash.is_none(),
            "fresh ticket should have no commit_hash"
        );
        assert!(
            fresh.lines_added.is_none(),
            "fresh ticket should have no lines_added"
        );
        assert!(
            fresh.lines_removed.is_none(),
            "fresh ticket should have no lines_removed"
        );
        assert_eq!(
            fresh.workspace_name, "test_workspace",
            "fresh workspace_name"
        );
        assert!(
            fresh.created_at.contains('T'),
            "fresh created_at should be RFC 3339: {}",
            fresh.created_at,
        );
        assert!(
            fresh.updated_at.contains('T'),
            "fresh updated_at should be RFC 3339: {}",
            fresh.updated_at,
        );
        assert_eq!(fresh.reporter, "test_reporter", "fresh reporter");
        assert!(
            fresh.prerequisites.is_empty(),
            "fresh prerequisites should be empty"
        );
        assert!(
            fresh.supersedes.is_none(),
            "fresh supersedes should be None"
        );
        assert!(
            fresh.superseded_by.is_none(),
            "fresh superseded_by should be None"
        );
        assert!(!fresh.is_archived, "fresh is_archived should be false");
        assert!(
            !fresh.pipeline_reservation,
            "fresh pipeline_reservation should be false"
        );

        // Set assigned_to (exercises COL_TICKET_ASSIGNED_TO with non-None value).
        store
            .set_assigned_to(&id, Some("test_assignee"))
            .await
            .expect("set_assigned_to");

        // Set commit_hash, lines_added, lines_removed with non-default values.
        store
            .set_commit_info(&id, "abcdef0123456789abcdef0123456789abcd0123", 42, 7)
            .await
            .expect("set_commit_info");

        // Read back BEFORE archiving (which clears assigned_to).
        let ticket = store
            .get_ticket(&id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // ── Assert every Ticket field round-trips ──────────────────────
        assert_eq!(ticket.id, id, "id mismatch");
        assert_eq!(ticket.title, "Roundtrip Title", "title mismatch");
        assert_eq!(
            ticket.description, "Roundtrip description",
            "description mismatch",
        );
        assert_eq!(ticket.phase, TicketPhase::Backlog, "phase mismatch");
        assert_eq!(
            ticket.assigned_to.as_deref(),
            Some("test_assignee"),
            "assigned_to should round-trip",
        );
        assert_eq!(ticket.workspace_name, "test_workspace");
        // Timestamps are auto-generated RFC 3339 — validate format, not value.
        assert!(
            ticket.created_at.contains('T'),
            "created_at should be RFC 3339: {}",
            ticket.created_at,
        );
        assert!(
            ticket.updated_at.contains('T'),
            "updated_at should be RFC 3339: {}",
            ticket.updated_at,
        );
        assert!(ticket.comments.is_empty(), "no comments expected");
        assert!(
            ticket.prerequisites.is_empty(),
            "prerequisites should round-trip as empty",
        );
        assert_eq!(
            ticket.commit_hash.as_deref(),
            Some("abcdef0123456789abcdef0123456789abcd0123"),
            "commit_hash mismatch",
        );
        assert_eq!(ticket.lines_added, Some(42), "lines_added mismatch");
        assert_eq!(ticket.lines_removed, Some(7), "lines_removed mismatch");
        assert_eq!(ticket.reporter, "test_reporter", "reporter mismatch");
        // Fields not set remain at their defaults.
        assert!(
            ticket.supersedes.is_none(),
            "supersedes should be None for simple ticket",
        );
        assert!(
            ticket.superseded_by.is_none(),
            "superseded_by should be None for simple ticket",
        );
        assert!(
            !ticket.is_archived,
            "is_archived should be false before archiving",
        );
        assert!(
            !ticket.pipeline_reservation,
            "pipeline_reservation should be false for fresh ticket",
        );

        // ── Exercise is_archived bool deserialization ──────────────────
        // set_archived flips is_archived to 1 in SQL, which exercises the
        // conversion: row.get::<bool>()?.
        store.set_archived(&id).await.expect("set_archived");

        let archived = store
            .get_ticket(&id)
            .await
            .expect("get_ticket")
            .expect("ticket exists after archive");
        assert!(
            archived.is_archived,
            "is_archived should be true after set_archived"
        );
        assert!(
            archived.assigned_to.is_none(),
            "assigned_to should be cleared after archive",
        );
    }

    // ── Archived ticket search methods ──────────────────────────────────

    /// Create an archived ticket with the given title in tests.
    async fn create_archived_ticket(
        store: &super::BoardStore,
        title: &str,
        workspace_name: &str,
    ) -> String {
        let ws = test_ws(workspace_name);
        let id = make_ticket(store, &ws, title, crate::board::TicketPhase::Done).await;
        store.set_archived(&id).await.expect("set_archived");
        id
    }

    #[tokio::test]
    async fn test_search_archived_by_fts_finds_matching_title() {
        let (store, _tmp) = open_test_store().await;
        let id = create_archived_ticket(&store, "Fix network timeout bug", "ws1").await;

        let results = store
            .search_archived_by_fts("network timeout", 10)
            .await
            .expect("FTS search");
        assert!(!results.is_empty(), "should find the ticket");
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&id.as_str()),
            "result should contain our ticket"
        );
    }

    #[tokio::test]
    async fn test_search_archived_by_fts_excludes_non_archived() {
        let (store, _tmp) = open_test_store().await;
        // Create a non-archived ticket — should not appear in archived search
        let ws = test_ws("ws2");
        make_ticket(
            &store,
            &ws,
            "Still active",
            crate::board::TicketPhase::Backlog,
        )
        .await;

        let results = store
            .search_archived_by_fts("active", 10)
            .await
            .expect("FTS search");
        assert!(results.is_empty(), "non-archived ticket should not appear");
    }

    #[tokio::test]
    async fn test_search_archived_by_fts_sanitize_mangles_query() {
        let (store, _tmp) = open_test_store().await;
        let results = store
            .search_archived_by_fts("!@#$%", 10)
            .await
            .expect("FTS search");
        assert!(
            results.is_empty(),
            "query with only special chars becomes empty after sanitize"
        );
    }

    /// Tests for `list_tickets_minimal`: found by ID, nonexistent IDs omitted,
    /// empty-ids early-return guard, and input order preservation.
    #[tokio::test]
    async fn test_list_tickets_minimal() {
        let (store, _tmp) = open_test_store().await;

        // Create 3 archived tickets in a shared store.
        let id_a = create_archived_ticket(&store, "Alpha", "ws").await;
        let id_b = create_archived_ticket(&store, "Beta", "ws").await;
        let id_c = create_archived_ticket(&store, "Gamma", "ws").await;

        // 1. Found by ID.
        let rows = store
            .list_tickets_minimal(std::slice::from_ref(&id_a))
            .await
            .expect("list minimal");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, id_a);
        assert_eq!(rows[0].title, "Alpha");
        assert_eq!(rows[0].phase, "done");

        // 2. Nonexistent IDs omitted.
        let rows = store
            .list_tickets_minimal(&[id_a.clone(), "nonexistent".to_string()])
            .await
            .expect("list minimal");
        assert_eq!(rows.len(), 1, "nonexistent IDs should be omitted");
        assert_eq!(rows[0].id, id_a);

        // 3. Empty ids returns empty (early-return guard).
        let rows: Vec<crate::board::TicketMinimal> = store
            .list_tickets_minimal(&[] as &[String])
            .await
            .expect("list minimal");
        assert!(rows.is_empty(), "empty ids should return empty results");

        // 4. Preserves input order.
        let rows = store
            .list_tickets_minimal(&[id_c.clone(), id_a.clone(), id_b.clone()])
            .await
            .expect("list minimal");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].id, id_c, "first result should be Gamma");
        assert_eq!(rows[1].id, id_a, "second result should be Alpha");
        assert_eq!(rows[2].id, id_b, "third result should be Beta");
    }

    /// Basic field layout of `detailed_display`: fields present, negative
    /// assertions for absent fields, and "(no comments)" when empty.
    #[tokio::test]
    async fn test_detailed_display_basic() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/test-workspace", "test-ws");

        let prereq_id = make_ticket(&store, &ws, "Prereq", TicketPhase::Backlog).await;

        let id = TicketBuilder::new(&store, &ws)
            .title("Display Test Ticket")
            .desc("A description for testing")
            .phase(TicketPhase::InDevelopment)
            .prereqs(std::slice::from_ref(&prereq_id))
            .reporter("manager")
            .create()
            .await
            .expect("create");

        let ticket = expect_ticket(&store, &id).await;
        let display = ticket.detailed_display();

        assert!(
            display.contains(&format!("Ticket: {id}")),
            "should contain ticket id"
        );
        assert!(
            display.contains("Title: Display Test Ticket"),
            "should contain title"
        );
        assert!(
            display.contains("Description: A description for testing"),
            "should contain description"
        );
        assert!(
            display.contains("Phase: in_development"),
            "should use snake_case phase"
        );
        assert!(
            display.contains("Reporter: manager"),
            "should contain reporter"
        );
        assert!(
            display.contains("Workspace: test-ws"),
            "should contain workspace"
        );
        assert!(
            display.contains("Created:"),
            "should contain created timestamp"
        );
        assert!(
            display.contains("Updated:"),
            "should contain updated timestamp"
        );
        assert!(
            display.contains(&format!("Prerequisites: {prereq_id}")),
            "should show prerequisites"
        );
        assert!(
            display.contains("Comments:"),
            "should have comments section"
        );
        assert!(display.contains("(no comments)"), "should show no comments");

        // Fields that should NOT appear when unset
        assert!(
            !display.contains("Supersedes:"),
            "no supersedes when not set"
        );
        assert!(
            !display.contains("Superseded by:"),
            "no superseded_by when not set"
        );
        assert!(
            !display.contains("Archived:"),
            "no archived line when false"
        );
        assert!(
            !display.contains("assigned_to:"),
            "assigned_to should not be displayed"
        );
        assert!(
            !display.contains("commit_hash:"),
            "commit_hash should not be displayed"
        );
        assert!(
            !display.contains("lines_added:"),
            "lines_added should not be displayed"
        );
        assert!(
            !display.contains("lines_removed:"),
            "lines_removed should not be displayed"
        );
    }

    /// `detailed_display` with comments (role labels, content) and multiple
    /// prerequisites joined by comma+space.
    #[tokio::test]
    async fn test_detailed_display_with_content() {
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/test-workspace", "test-ws");

        // ── Comment formatting: two comments with different roles ──

        let id = make_ticket(&store, &ws, "Comment Test", TicketPhase::Backlog).await;

        store
            .add_comment(&id, Role::Analyst.as_str(), "First comment")
            .await
            .expect("add_comment");
        store
            .add_comment(&id, Role::Reviewer.as_str(), "Second comment")
            .await
            .expect("add_comment");

        let ticket = expect_ticket(&store, &id).await;
        let display = ticket.detailed_display();

        assert!(
            display.contains("Comments:"),
            "should have comments section"
        );
        assert!(display.contains("[analyst]"), "should show analyst role");
        assert!(display.contains("[reviewer]"), "should show reviewer role");
        assert!(
            display.contains("First comment"),
            "should show first comment"
        );
        assert!(
            display.contains("Second comment"),
            "should show second comment"
        );
        assert!(
            !display.contains("(no comments)"),
            "should not say 'no comments' when comments exist"
        );

        // ── Multiple prerequisites: all three joined by comma+space ──

        let pre_a = make_ticket(&store, &ws, "Pre-A", TicketPhase::Backlog).await;
        let pre_b = make_ticket(&store, &ws, "Pre-B", TicketPhase::Backlog).await;
        let pre_c = make_ticket(&store, &ws, "Pre-C", TicketPhase::Backlog).await;

        let multi_id = TicketBuilder::new(&store, &ws)
            .title("Multi prereq")
            .prereqs(&[pre_a.clone(), pre_b.clone(), pre_c.clone()])
            .create()
            .await
            .expect("create");

        let ticket = expect_ticket(&store, &multi_id).await;
        let display = ticket.detailed_display();

        assert!(
            display.contains(&format!("Prerequisites: {pre_a}, {pre_b}, {pre_c}")),
            "should show all prerequisites joined with comma+space"
        );
    }

    /// `detailed_display` for supersedes chains: new ticket shows Supersedes,
    /// old ticket shows Superseded by + Archived.
    #[tokio::test]
    async fn test_detailed_display_supersedes_chain() {
        init_test_stores().await;
        let store = crate::board::BOARD.get().unwrap();
        let ws = test_ws_named("/ws", "ws");

        // Create an old ticket first
        let old_id = make_ticket(store, &ws, "Old ticket", TicketPhase::Backlog).await;

        // Supersede it — new ticket gets supersedes = old_id, old ticket gets
        // superseded_by = new_id and is archived.
        let new_id = TicketBuilder::new(store, &ws)
            .title("New ticket")
            .desc("new desc")
            .supersede(&old_id)
            .await
            .expect("supersede");

        // Check the new ticket shows Supersedes
        let new_ticket = expect_ticket(store, &new_id).await;
        let new_display = new_ticket.detailed_display();
        assert!(
            new_display.contains(&format!("Supersedes: {old_id}")),
            "new ticket should show Supersedes: old_id"
        );

        // Check the old ticket shows Superseded by + Archived
        let old_ticket = expect_ticket(store, &old_id).await;
        let old_display = old_ticket.detailed_display();
        assert!(
            old_display.contains(&format!("Superseded by: {new_id}")),
            "old ticket should show Superseded by: new_id"
        );
        assert!(
            old_display.contains("Archived: yes"),
            "old ticket should be archived"
        );
    }

    #[tokio::test]
    async fn test_list_archived_with_embeddings_returns_deserialized() {
        let (store, _tmp) = open_test_store().await;

        // Empty DB returns empty
        {
            let candidates = store.list_archived_with_embeddings().await.expect("list");
            assert!(candidates.is_empty(), "no tickets at all");
        }

        let ws = test_ws("ws");

        // Create a ticket with a known embedding blob (two small f32s)
        let embedding: Vec<f32> = vec![1.0, 2.0];
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let id = TicketBuilder::new(&store, &ws)
            .title("Embedded ticket")
            .phase(crate::board::TicketPhase::Done)
            .embedding(&blob)
            .create()
            .await
            .expect("create_ticket with embedding");
        store.set_archived(&id).await.expect("archive");

        let candidates = store.list_archived_with_embeddings().await.expect("list");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, id);
        assert_eq!(candidates[0].1, vec![1.0, 2.0]);
    }
}
