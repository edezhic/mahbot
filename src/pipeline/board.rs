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
    pub static BOARD: BoardStore,
    post_open = after_open,
    expect = "BOARD not initialized — call init_all_stores() first",
}

/// Background task: auto-archive cancelled tickets older than 1 hour.
pub async fn run_archive_cancelled_loop() {
    let interval = std::time::Duration::from_mins(5);

    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }

        // Session-pin terminal deletion (S5): remove permanently-NULL seats for tickets in a
        // terminal phase — with the pin gone the TTL guard stops protecting the accumulated
        // stage session (idempotent, ≤5-min delay against the session TTL).
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

/// Stale-cancelled archival window (hours): Cancelled tickets this old are archived by
/// [`run_archive_cancelled_loop`].
const CANCELLED_ARCHIVE_HOURS: i64 = 1;

/// The acting identity for pipeline-driven phase transitions (poller claims, breaker-trip
/// drain-to-planning). Stage finalizations, supersede cancels and the manual tool/GUI paths write
/// role labels or `"user"` instead.
///
/// Invariant: every phase-changing UPDATE on `tickets` writes `last_transition_actor` in the same
/// statement — the chronicle materializer reads the CDC after-row, so a stale column misattributes
/// the *previous* actor.
///
/// Invariant: a finished (`done`) ticket's phase is immutable. Every phase statement whose source
/// could be `done` must carry `phase <> 'done'` (writing `done` onto a `done` ticket is not a
/// transition and stays allowed), so no writer — tool, GUI, stage or future raw UPDATE — reopens a
/// finished ticket. Claim and drain are safe only because their source is pinned to
/// `backlog`/`queued`; keep any new phase writer inside this rule. Only archiving remains.
pub(crate) const PIPELINE_ACTOR: &str = "pipeline";

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
        LABEL                  => "label",
        REVIEWED_HEAD          => "reviewed_head",
        REVIEWED_TREE          => "reviewed_tree",
        DONE_AT                => "done_at",
        BOUNCE_COUNT           => "bounce_count",
    }
}

// Column definitions for comment SELECT queries: `id` and `ticket_id` are excluded because no
// comment consumer reads them (`ticket_id` is known from the parent query context). Both stay in
// the schema and are not candidates for removal.
crate::columns! {
    COMMENT_COLUMNS [COMMENT] {
        ROLE       => "role",
        CONTENT    => "content",
        CREATED_AT => "created_at",
    }
}

const PIPELINE_OCCUPIED_PHASES: &[TicketPhase] = &[
    TicketPhase::InDevelopment,
    TicketPhase::Verification,
    TicketPhase::InSanitation,
];

/// The four phases the poll loop drives with phase jobs, in dispatch order.
pub(crate) const WORKING_PHASES: &[TicketPhase] = &[
    TicketPhase::Analysis,
    TicketPhase::InDevelopment,
    TicketPhase::Verification,
    TicketPhase::InSanitation,
];

/// Phases that unblock dependent tickets: on entering one, dependents become claimable (their
/// prerequisite filter no longer blocks them).
///
/// [`TicketPhase::Failed`] is intentionally excluded — a failed ticket permanently blocks its
/// dependents, requiring manual intervention. [`TicketPhase::is_unblocking`] delegates here, so
/// a phase added to this set is picked up automatically; if the two must ever diverge, that
/// delegation has to be broken explicitly.
const UNBLOCKING_PHASES: &[TicketPhase] = &[TicketPhase::Done, TicketPhase::Cancelled];

/// Terminal phases — a ticket in one of these can no longer be claimed.
const TERMINAL_PHASES: &[TicketPhase] = &[
    TicketPhase::Done,
    TicketPhase::Cancelled,
    TicketPhase::Failed,
];

/// Produces an SQL fragment listing phases as quoted, comma-separated strings — e.g. `'done',
/// 'cancelled'`.
///
/// # Precondition
///
/// The input slice must be non-empty: an empty slice yields `WHERE phase IN ()`, invalid SQL.
fn phase_list_sql_fragment(phases: &[TicketPhase]) -> String {
    phases
        .iter()
        .map(|p| format!("'{}'", p.as_ref()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The claim-candidate eligibility predicate, shared verbatim by the claim UPDATE and its
/// read-only probe (single source — the two cannot drift). `source` selects the claim shape: only
/// Backlog→Analysis applies the [`BACKLOG_CLAIM_GRACE`] cutoff, only Queued→InDevelopment enforces
/// the single-pipeline-occupancy and failed-sibling start gates. Placeholders are numbered from
/// `base`: `?{base}` = source phase, `?{base+1}` = workspace name and, for Backlog, `?{base+2}` =
/// grace cutoff.
fn claim_candidate_where(base: usize, source: TicketPhase) -> String {
    let mut parts = vec![
        format!("t1.phase = ?{base}"),
        format!("t1.workspace_name = ?{}", base + 1),
        "t1.is_archived = 0".to_string(),
    ];
    if source == TicketPhase::Backlog {
        parts.push(format!("t1.created_at <= ?{}", base + 2));
    }
    if source == TicketPhase::Queued {
        parts.push(format!(
            "NOT EXISTS (SELECT 1 FROM tickets t2 \
                WHERE t2.workspace_name = t1.workspace_name \
                AND t2.phase IN ({}) \
                AND t2.is_archived = 0 \
                AND t2.id != t1.id)",
            phase_list_sql_fragment(PIPELINE_OCCUPIED_PHASES),
        ));
        // A failed ticket awaiting Manager triage parks the workspace until it is cancelled or
        // superseded (both move it out of Failed). Failed is deliberately NOT in
        // [`PIPELINE_OCCUPIED_PHASES`], so the gate lives here, on the start decision itself,
        // where the read-only probe inherits it through this shared builder.
        parts.push(format!(
            "NOT EXISTS (SELECT 1 FROM tickets t2 \
                WHERE t2.workspace_name = t1.workspace_name \
                AND t2.phase = '{}' \
                AND t2.is_archived = 0 \
                AND t2.id != t1.id)",
            TicketPhase::Failed.as_ref(),
        ));
    }
    parts.push(format!(
        "NOT EXISTS ( \
            SELECT 1 FROM json_each(t1.prerequisites) AS je \
            JOIN tickets t_pre ON t_pre.id = je.value \
            WHERE t_pre.phase NOT IN ({}) \
         )",
        phase_list_sql_fragment(UNBLOCKING_PHASES),
    ));
    parts.join(" AND ")
}

fn parse_prereqs(raw: &str) -> Result<Vec<String>> {
    serde_json::from_str(raw).with_context(|| {
        let preview = crate::util::truncate(raw, 200);
        format!("Corrupt prerequisites JSON in database: {preview}")
    })
}

/// Bundled parameters for ticket creation, to avoid parameter explosion across
/// [`BoardStore::insert_ticket_tx`], [`BoardStore::create_ticket`] and
/// [`BoardStore::supersede_and_create`].
#[derive(Debug, Clone)]
pub(crate) struct TicketParams {
    pub title: String,
    pub description: String,
    pub workspace_name: String,
    pub phase: TicketPhase,
    pub prerequisites: Vec<String>,
    pub reporter: String,
    pub priority: i64,
    pub label: TicketLabel,
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
    /// ID of the ticket this one supersedes (set when created via supersede); the superseded
    /// ticket is cancelled atomically during creation.
    pub supersedes: Option<String>,
    /// ID of the ticket that supersedes this one. Purely informational — never drives logic.
    pub superseded_by: Option<String>,
    /// Full commit SHA (40 hex chars), `None` if no commit recorded.
    pub commit_hash: Option<String>,
    pub lines_added: Option<i64>,
    pub lines_removed: Option<i64>,
    /// Creator/tool identity, set at construction time. Also shown by the GUI board: a value
    /// matching a role name renders that role's label (e.g. "Manager"), otherwise the value with
    /// its first character uppercased. May be empty when no reporter is recorded.
    pub reporter: String,
    /// Whether this ticket has been archived (hidden from normal listings).
    pub is_archived: bool,
    pub priority: i64,
    /// Classification label, fixed at creation and never written again. Shown as a chip on the
    /// ticket's card and detail popup, it prefixes every commit the pipeline makes for the ticket
    /// and decides whether the release changelog keeps it.
    pub label: TicketLabel,
    /// HEAD hash of the content the code reviewers were handed, as written by
    /// `verification::record_reviewed_base` for a round whose review ran and which then moved the
    /// ticket on. `None` until the first such recording; the reviewer skip-gate uses it to detect
    /// brand-new content that must never skip review.
    pub reviewed_head: Option<String>,
    /// The working tree identity of the same content: what a `git add -A` would have staged.
    /// With [`reviewed_head`](Self::reviewed_head) and a clean porcelain this identifies the
    /// exact content reviewers saw.
    pub reviewed_tree: Option<String>,
    /// Exact completion timestamp: set on transition to Done and never cleared — a finished
    /// ticket's phase is immutable. `None` for tickets that never reached Done.
    pub done_at: Option<String>,
    /// Number of times this ticket bounced back into development from a non-success in either
    /// post-development stage. Drives the bounce-based circuit breaker (max 10). Engineer hard
    /// failures are pause-only (workspace pause, implementation frozen) and do not consume this
    /// budget.
    pub bounce_count: i64,
}

impl Ticket {
    /// Whether the maintainer filed this ticket — the workspace-cleanup tickets only the
    /// maintainer creates.
    #[must_use]
    pub(crate) fn filed_by_maintainer(&self) -> bool {
        self.reporter == crate::Role::Maintainer.as_str()
    }

    /// Short single-line display for listing tickets in agent-facing output:
    /// `"  [{reporter}] [{phase}] {id}: {title}"` (the two-space indent aligns within a multi-line
    /// block). No trailing newline — callers add it. `{phase}` is the snake_case Display form of
    /// [`TicketPhase`]; for user-facing labels use [`TicketPhase::display_name()`].
    ///
    /// Not unified with two deliberately different formats: `crate::prompt::format_ticket_block`
    /// (a Markdown `<current-ticket>` block for system messages) and `search_archived_tickets`
    /// (which omits the reporter).
    #[must_use]
    pub fn short_display(&self) -> String {
        format!(
            "  [{}] [{}] {}: {}",
            self.reporter, self.phase, self.id, self.title
        )
    }

    /// Detailed multi-line display of the ticket for
    /// [`GetTicketTool`](crate::tools::ticket::GetTicketTool) and other agent-facing output:
    /// id, title, description, snake_case phase, P0–P4+ priority, reporter, workspace, dates,
    /// supersede/prerequisite links and the archived flag when set, then [`Self::format_comments`].
    ///
    /// `commit_hash` and `lines_added`/`lines_removed` are deliberately omitted — available in
    /// the board UI, not meaningful for agent context.
    ///
    /// The result can be arbitrarily large (unbounded descriptions and comments). Callers that
    /// need truncation apply their own limits (see `GetTicketTool::preserve_full_output`, which
    /// disables the default ~5 KB tool-output truncation); `GetTicketTool` uses
    /// [`Self::detailed_display_limited`] for its default truncated view. Agent tool calls depend
    /// on this exact format, so any change here must be kept in sync with that consumer.
    #[must_use]
    pub fn detailed_display(&self) -> String {
        let mut out = self.header(&self.description);
        out.push_str(&self.format_comments());
        out
    }

    /// A truncated [`Self::detailed_display`] for callers that only need the ticket's latest
    /// state: the description is capped at `desc_max_chars` (with a trailing `…`), each comment's
    /// content at `comment_max_chars`, EXCEPT the newest `last_n_full` comments, shown in full.
    /// Header metadata is never truncated.
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

    /// Comments as a `"Comments:"` block for [`crate::tools::ticket::GetTicketTool`]: every
    /// comment in full, or `"Comments:\n  (no comments)"` when there are none.
    #[must_use]
    fn format_comments(&self) -> String {
        self.format_comments_limited(usize::MAX, self.comments.len())
    }

    /// Like [`Self::format_comments`], but caps each comment's content at `max_chars`
    /// (appending `…`) except for the newest `last_n_full`, shown in full.
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

    /// A single comment line: `"\n  [{role}] ({timestamp}): {content}"`. Timestamps are cut to
    /// seconds (`[..19]`), with a `min()` guard for abnormally short strings.
    #[must_use]
    fn comment_line(c: &TicketComment, content: &str) -> String {
        let end = 19.min(c.created_at.len());
        let ts = &c.created_at[..end];
        format!("\n  [{}] ({}): {}", c.role, ts, content)
    }
}

/// The single classification label every ticket carries: `feat` for work that adds a capability,
/// `fix` for work that repairs a problem, `maint` for everything else.
///
/// Stated once, at creation (see [`CreateTicketTool`](crate::tools::ticket::CreateTicketTool));
/// no writer updates it afterwards, so it never changes for the ticket's whole life. The product
/// shows it as a chip on the ticket's card and detail popup, and prefixes every pipeline commit
/// subject with `[feat] `/`[fix] `/`[maint] `. Stored like [`TicketPhase`]: lowercase snake_case
/// matching the DB column values.
#[derive(Debug, Clone, Copy, PartialEq, strum::Display, strum::AsRefStr, strum::EnumIter)]
#[strum(serialize_all = "snake_case")]
pub enum TicketLabel {
    Feat,
    Fix,
    Maint,
}

impl TicketLabel {
    /// The tag every pipeline commit subject starts with — `[feat] `, `[fix] ` or `[maint] `,
    /// before the title.
    #[must_use]
    pub(crate) fn commit_prefix(self) -> String {
        format!("[{self}] ")
    }
}

/// Valid label names, pre-computed once for the refusal message.
static ALL_TICKET_LABEL_NAMES: LazyLock<String> = LazyLock::new(|| {
    <TicketLabel as strum::IntoEnumIterator>::iter()
        .map(|label| label.to_string())
        .collect::<Vec<_>>()
        .join(", ")
});

impl std::str::FromStr for TicketLabel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <TicketLabel as strum::IntoEnumIterator>::iter()
            .find(|label| label.as_ref() == s)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid ticket label '{s}'. Valid labels: {}",
                    *ALL_TICKET_LABEL_NAMES
                )
            })
    }
}

/// Lowercase snake_case strings matching the DB column values — no schema migration needed.
/// `strum` derives Display, `AsRefStr` (the borrow-side token), `IntoStaticStr` (the same strings
/// with a `'static` lifetime — the `jobs.kind` vocabulary) and `EnumIter`; `FromStr` is manual for
/// user-friendly error messages.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Serialize,
    strum::Display,
    strum::AsRefStr,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TicketPhase {
    Backlog,
    Analysis,
    /// Waiting for Manager review. Not picked up automatically by any agent — the Manager or
    /// user must advance it to Queued or cancel it.
    Planning,
    Queued,
    InDevelopment,
    InSanitation,
    /// The merged code-review + functional-check stage: the code reviewers and the single
    /// functional tester run concurrently on one phase job and produce one consolidated result.
    /// The round also runs the workspace's own project commands alongside its participants.
    Verification,
    /// Finished for good: the phase is immutable — only archiving remains, plus reading and
    /// commenting, which never touch the phase.
    Done,
    Cancelled,
    Failed,
}

impl TicketPhase {
    /// Whether the phase unblocks dependent tickets (`Done`, `Cancelled`).
    ///
    /// Delegates to the crate-private set so the two can never diverge.
    /// [`TicketPhase::Failed`] is deliberately excluded — a failed ticket permanently blocks its
    /// dependents and stays visible in active views for manual triage.
    #[must_use]
    pub fn is_unblocking(&self) -> bool {
        UNBLOCKING_PHASES.contains(self)
    }

    /// Whether the phase is terminal (`Done`, `Cancelled`, `Failed`) — such a ticket can no
    /// longer be claimed, so its phase job is finished.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        TERMINAL_PHASES.contains(self)
    }

    #[must_use]
    pub fn is_pipeline_occupied(&self) -> bool {
        PIPELINE_OCCUPIED_PHASES.contains(self)
    }

    /// Human-readable label with spaces instead of underscores (e.g. `"in development"`).
    #[must_use]
    pub fn display_name(&self) -> String {
        let token = self.as_ref();
        match self {
            Self::Verification => format!("in {token}"),
            _ => token.replace('_', " "),
        }
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

// Display and AsRefStr come from the strum derives; FromStr above is manual for user-friendly
// error messages.

/// Bundles an `UPDATE tickets` statement with its bound parameters and the ticket id it targets.
/// Phase transitions execute the SQL directly (their zero-row outcome needs its own explanation);
/// the other producers go through [`PreparedUpdate::execute_no_cancel`],
/// [`PreparedUpdate::execute_tx`] or [`PreparedUpdate::execute_and_cancel`], which use
/// `ticket_id` for the not-found check.
struct PreparedUpdate {
    sql: String,
    params: Vec<db::Value>,
    ticket_id: String,
}

impl PreparedUpdate {
    /// Execute within an existing transaction, verifying a row was affected. Does NOT cancel
    /// registered agents — the caller owns the transaction lifecycle and cancels agents before
    /// starting it when needed.
    async fn execute_tx(self, tx: &db::TxGuard<'_>) -> Result<()> {
        let rows = tx.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        Ok(())
    }

    /// Execute the update without cancelling any agent — for post-agent operations where the
    /// caller knows no agent is running.
    async fn execute_no_cancel(self, conn: &db::Connection) -> Result<()> {
        let rows = conn.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        Ok(())
    }

    /// Execute the update, verify it affected a row, then cancel any agent registered on this
    /// ticket — the execute → verify → cancel pattern for single-ticket mutations.
    async fn execute_and_cancel(self, conn: &db::Connection) -> Result<()> {
        let rows = conn.execute(&self.sql, self.params).await?;
        BoardStore::ensure_ticket_found(rows, &self.ticket_id)?;
        crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&self.ticket_id);
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum LoadComments {
    Yes,
    No,
}

impl BoardStore {
    /// Post-open setup: ensure the FTS index exists with the correct tokenizer.
    pub(crate) async fn after_open(&self) -> anyhow::Result<()> {
        crate::db::ensure_fts_index(
            &self.conn,
            crate::db::TICKETS_FTS_INDEX_NAME,
            "ngram",
            crate::db::TICKETS_FTS_INDEX_DDL,
        )
        .await?;
        Ok(())
    }

    /// Shared INSERT logic for [`BoardStore::create_ticket`] and
    /// [`BoardStore::supersede_and_create`]. Computes the timestamp and serializes prerequisites;
    /// does NOT commit — the caller calls `tx.commit()` after any further writes.
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
             created_at, updated_at, prerequisites, supersedes, reporter, priority, label) \
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
                params.priority,
                params.label.as_ref(),
            ],
        )
        .await?;
        Ok(())
    }

    /// Rewire dependents after supersede: tickets whose prerequisites mention `supersede_id` are
    /// updated to point to `new_id`, queried and updated in one transaction — no TOCTOU window.
    /// `json_each()` gives exact prerequisite matching, as in
    /// [`claim_backlog_for_analysis`](Self::claim_backlog_for_analysis).
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

    /// Begin a transaction, generate a ticket ID, and validate prerequisites — the shared preamble
    /// of [`BoardStore::create_ticket`] and [`BoardStore::supersede_and_create`]. The new id must
    /// not appear in its own prerequisites, and every prerequisite must exist in the same
    /// workspace.
    ///
    /// Callers must not call `self.conn` methods until the guard is dropped or committed —
    /// `TxGuard` holds a tokio mutex lock. Correctness (TOCTOU) rests on that mutex serializing
    /// Rust-level writes plus the Turso transaction isolating them at the database level, so no
    /// concurrent write can change a prerequisite between validation and the caller's INSERT.
    /// Validation runs inside the transaction through its own guard; see [`TxGuard::query`] for
    /// why that is the only read the transaction's work may use.
    async fn begin_tx_and_validate_prerequisites(
        &self,
        workspace_name: &str,
        prerequisites: &[String],
    ) -> Result<(TxGuard<'_>, String)> {
        let tx = self.conn.begin_tx().await?;
        // Seed-then-bump, not the engine's unusable `INSERT ... ON CONFLICT ... DO UPDATE` form
        // (see `db::Connection::upsert_row`); the two-step helper cannot serve this call because
        // the new id has to be read back. A fresh workspace seeds `next_id` at 0, then the bump
        // stores 1 and returns `next_id - 1` — the first ticket is still `<workspace>-0`.
        tx.execute(
            "INSERT INTO ticket_counters (workspace_name, next_id) VALUES (?1, 0) \
             ON CONFLICT(workspace_name) DO NOTHING",
            db::params![workspace_name],
        )
        .await?;
        let seq: i64 = tx
            .execute_one(
                "UPDATE ticket_counters SET next_id = next_id + 1 WHERE workspace_name = ?1 \
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
        // Validate prerequisites inside the transaction — see [`TxGuard::query`].
        Self::validate_prerequisites(&tx, prerequisites, workspace_name).await?;
        Ok((tx, id))
    }

    pub(crate) async fn create_ticket(&self, params: &TicketParams) -> Result<String> {
        let (tx, id) = self
            .begin_tx_and_validate_prerequisites(&params.workspace_name, &params.prerequisites)
            .await?;

        Self::insert_ticket_tx(&tx, &id, params, None).await?;

        tx.commit().await?;
        Ok(id)
    }

    pub(crate) async fn supersede_and_create(
        &self,
        supersede_id: &str,
        params: &TicketParams,
        actor: &str,
    ) -> Result<String> {
        anyhow::ensure!(
            !params.prerequisites.iter().any(|p| p == supersede_id),
            "Ticket cannot supersede and depend on the same ticket: {supersede_id}"
        );

        let (tx, new_id) = self
            .begin_tx_and_validate_prerequisites(&params.workspace_name, &params.prerequisites)
            .await?;

        // Verify the superseded ticket exists, belongs to the same workspace, and is not finished
        // — a finished phase is immutable and supersede is a phase write. Inside the transaction:
        // see [`TxGuard::query`].
        let rows = tx
            .query(
                "SELECT workspace_name, phase FROM tickets WHERE id = ?1",
                db::params![supersede_id],
            )
            .await?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Superseded ticket not found: {supersede_id}"))?;
        let old_ws: String = row.get(0)?;
        let old_phase: String = row.get(1)?;
        anyhow::ensure!(
            old_ws == params.workspace_name,
            "Superseded ticket {supersede_id} belongs to workspace '{old_ws}', \
             not the current workspace '{}'. \
             Cross-workspace supersede is not allowed.",
            params.workspace_name,
        );
        if old_phase == TicketPhase::Done.as_ref() {
            return Err(Self::finished_ticket_error(supersede_id, "superseded"));
        }

        let now = db::now();
        let cancelled_rows = tx
            .execute(
                "UPDATE tickets SET phase = ?1, updated_at = ?2, \
                 superseded_by = ?4, is_archived = 1, \
                 last_transition_actor = ?5 \
                 WHERE id = ?3 AND phase <> 'done'",
                db::params![
                    TicketPhase::Cancelled.as_ref(),
                    now,
                    supersede_id,
                    new_id.as_str(),
                    actor,
                ],
            )
            .await?;
        Self::ensure_ticket_found(cancelled_rows, supersede_id)?;

        Self::insert_ticket_tx(&tx, &new_id, params, Some(supersede_id)).await?;

        Self::rewire_dependents_tx(&tx, supersede_id, &new_id, &params.workspace_name).await?;

        // Cancel agents on the superseded ticket BEFORE the transaction commits. A crash between
        // commit and cancellation would leave a Cancelled ticket with agents still registered and
        // running (orphaned agents on a cancelled ticket); cancelling first flips the trade-off —
        // if the commit then fails, agents were cancelled unnecessarily but re-register on
        // re-dispatch.
        crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(supersede_id);

        // No chronicle push for the cancellation: supersede is only reachable through agent tools
        // (Manager or Maintainer CreateTicketTool) and agent actions are intentionally silent — the
        // GUI board path is the user surface that notifies the Manager.
        tx.commit().await?;

        Ok(new_id)
    }

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
            label: row
                .get::<String>(COL_TICKET_LABEL)?
                .parse::<TicketLabel>()?,
            reviewed_head: row.get(COL_TICKET_REVIEWED_HEAD)?,
            reviewed_tree: row.get(COL_TICKET_REVIEWED_TREE)?,
            done_at: row.get(COL_TICKET_DONE_AT)?,
            bounce_count: row.get(COL_TICKET_BOUNCE_COUNT)?,
        })
    }

    /// Grace window for Backlog→Analysis claims; the rationale lives on
    /// [`claim_backlog_for_analysis`](Self::claim_backlog_for_analysis).
    pub(crate) const BACKLOG_CLAIM_GRACE: Duration = Duration::seconds(5);

    /// Claim a workspace's oldest eligible [`TicketPhase::Backlog`] ticket and transition it to
    /// Analysis; see [`claim_for_phase`] for the shared semantics. The `created_at <= ?6` cutoff
    /// excludes tickets younger than [`BACKLOG_CLAIM_GRACE`], so a fresh ticket stays in Backlog
    /// ~5s: claiming it sooner would spawn analysts that are immediately cancelled (plus a spurious
    /// phase-change notification), while the Manager gets time to move it to Planning/Queued.
    pub(crate) async fn claim_backlog_for_analysis(
        &self,
        workspace_name: &str,
        now: String,
        grace_cutoff: String,
    ) -> Result<Option<Ticket>> {
        self.claim_for_phase(
            workspace_name,
            now,
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            Some(&grace_cutoff),
        )
        .await
    }

    /// Claim a workspace's oldest eligible [`TicketPhase::Queued`] ticket and transition it to
    /// InDevelopment; see [`claim_for_phase`] for the shared semantics. Two extra checks reject
    /// the claim (returning `None`), both against the committed state at the claim instant:
    ///
    /// - while any pipeline-occupied ticket ([`PIPELINE_OCCUPIED_PHASES`]) exists in the same
    ///   workspace, enforcing a single ticket at a time in the dev/verification pipeline;
    /// - while any non-archived [`TicketPhase::Failed`] ticket exists in the same workspace — a
    ///   failure awaiting Manager triage parks the workspace so no new work starts next to the
    ///   unresolved failure and its uncommitted state. The park lifts by itself: cancelling or
    ///   superseding the failed ticket moves it out of Failed. Nothing else is paused, so the
    ///   Manager keeps full triage control.
    ///
    /// The two occupancy checks' `t2.is_archived = 0` guard is a deliberate no-op for the pipeline
    /// phases (only Done/Cancelled tickets get archived, and neither is a pipeline phase) but
    /// load-bearing for `Failed`, which stays non-archived until triage.
    pub(crate) async fn claim_queued_for_development(
        &self,
        workspace_name: &str,
        now: String,
    ) -> Result<Option<Ticket>> {
        self.claim_for_phase(
            workspace_name,
            now,
            TicketPhase::Queued,
            TicketPhase::InDevelopment,
            None,
        )
        .await
    }

    /// Core claim: atomically transition a workspace's oldest eligible ticket from `source` to
    /// `target` in one `UPDATE..RETURNING`. The `workspace_name` filter keeps the claim inside one
    /// workspace, and `t1.phase = ?4` bound to `source` gives CAS-style atomicity — no matching
    /// ticket means `None`.
    ///
    /// The WHERE predicate comes from [`claim_candidate_where`] with base offset 4 (`?1`–`?3` are
    /// the SET-clause phase/updated_at/actor binds) — the same builder the read-only probe uses, so
    /// claim and probe cannot drift. Only Backlog carries the `grace_cutoff` bind (it must be
    /// `Some` only for [`TicketPhase::Backlog`]); only Queued enforces the pipeline-occupancy and
    /// failed-sibling start gates. `now` and `grace_cutoff` come from the caller, computed once per
    /// poll round and shared with the probe, so the two never disagree about clock or cutoff.
    ///
    /// The prereq filter is part of the same atomic UPDATE (no separate SELECT+UPDATE window):
    /// `NOT EXISTS` over `json_each(t1.prerequisites)` rejects a candidate unless every
    /// prerequisite ticket is in an unblocking phase ([`UNBLOCKING_PHASES`]) — so a Manager moving
    /// a Backlog ticket straight to Queued (or a prereq re-opened after its dependent entered the
    /// pipeline) cannot advance it out of dependency order.
    ///
    /// Ordering is `priority ASC, created_at ASC`: higher urgency first, then oldest.
    ///
    /// The SQL text per (source, grace_cutoff) combination is invariant, which the
    /// [`Connection::execute_rows_cached`] cache contract requires. The claim SELECT stays on the
    /// tickets table; the agents roster is a separate table checked by the caller (phase-job +
    /// launched-roster guard) since Backlog/Queued tickets have no running agent.
    async fn claim_for_phase(
        &self,
        workspace_name: &str,
        now: String,
        source: TicketPhase,
        target: TicketPhase,
        grace_cutoff: Option<&str>,
    ) -> Result<Option<Ticket>> {
        let sql = format!(
            "UPDATE tickets SET phase = ?1, updated_at = ?2, \
             last_transition_actor = ?3 \
             WHERE id = (SELECT t1.id FROM tickets t1 \
             WHERE {} \
             ORDER BY t1.priority ASC, t1.created_at ASC LIMIT 1) \
             RETURNING {TICKET_COLUMNS}",
            claim_candidate_where(4, source),
        );
        let rows = match grace_cutoff {
            Some(cutoff) => {
                self.conn
                    .execute_rows_cached(
                        &sql,
                        db::params![
                            target.as_ref(),
                            now,
                            PIPELINE_ACTOR,
                            source.as_ref(),
                            workspace_name,
                            cutoff
                        ],
                    )
                    .await?
            }
            None => {
                self.conn
                    .execute_rows_cached(
                        &sql,
                        db::params![
                            target.as_ref(),
                            now,
                            PIPELINE_ACTOR,
                            source.as_ref(),
                            workspace_name
                        ],
                    )
                    .await?
            }
        };
        match rows.into_iter().next() {
            Some(row) => Ok(Some(self.ticket_from_row(&row, LoadComments::Yes).await?)),
            None => Ok(None),
        }
    }

    /// Read-only existence probe for the Backlog→Analysis claim: `true` when at least one ticket
    /// satisfies the exact claim predicate (same shared builder, same bound grace cutoff — no
    /// drift). Existence only: no ORDER BY, no column fetch.
    pub(crate) async fn backlog_claim_candidate_exists(
        &self,
        workspace_name: &str,
        grace_cutoff: &str,
    ) -> Result<bool> {
        self.claim_candidate_exists(TicketPhase::Backlog, workspace_name, Some(grace_cutoff))
            .await
    }

    /// Read-only existence probe for the Queued→InDevelopment claim.
    pub(crate) async fn queued_claim_candidate_exists(&self, workspace_name: &str) -> Result<bool> {
        self.claim_candidate_exists(TicketPhase::Queued, workspace_name, None)
            .await
    }

    /// Core probe: `SELECT 1 FROM tickets t1 WHERE {claim_candidate_where} LIMIT 1` through the
    /// prepared-cached path. The SQL text is invariant (phase lists are compile-time constants;
    /// workspace and cutoff are bound parameters), which the [`Connection::query_cached`] cache
    /// contract requires. `query_cached` is a read entry that does not check the SQL itself, so
    /// the statement being a plain read-only SELECT is what keeps the read where it belongs.
    /// Existence only, so unlike the claim there is no ORDER BY sorter to pay for.
    async fn claim_candidate_exists(
        &self,
        source: TicketPhase,
        workspace_name: &str,
        grace_cutoff: Option<&str>,
    ) -> Result<bool> {
        let sql = format!(
            "SELECT 1 FROM tickets t1 WHERE {} LIMIT 1",
            claim_candidate_where(1, source),
        );
        let rows = match grace_cutoff {
            Some(cutoff) => {
                self.conn
                    .query_cached(&sql, db::params![source.as_ref(), workspace_name, cutoff])
                    .await?
            }
            None => {
                self.conn
                    .query_cached(&sql, db::params![source.as_ref(), workspace_name])
                    .await?
            }
        };
        Ok(!rows.is_empty())
    }

    /// Select tickets matching a SQL suffix (everything after `FROM tickets`), parsing each row via
    /// [`ticket_from_row`](Self::ticket_from_row) — the shared building block for all
    /// `SELECT {TICKET_COLUMNS}` queries. `suffix` typically starts with `WHERE` and may include
    /// `ORDER BY`, `LIMIT`, …; `params` is forwarded to the underlying query unchanged.
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

    pub async fn get_ticket(&self, ticket_id: &str) -> Result<Option<Ticket>> {
        Ok(self
            .select_tickets("WHERE id = ?1", db::params![ticket_id], LoadComments::Yes)
            .await?
            .into_iter()
            .next())
    }

    /// Fetch multiple tickets by their IDs. An empty `ids` returns an empty vec without a SQL
    /// round-trip; the result order is **arbitrary**, so callers needing input order re-sort.
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

    /// Build a `WHERE id IN (?, ?, ...)` suffix and parameter vector from ticket IDs.
    /// `ids` must be non-empty — an empty list yields invalid SQL (a Turso syntax error).
    fn in_clause_for_ids(ids: &[String]) -> (String, Vec<Value>) {
        let suffix = format!("WHERE id IN ({})", db::sql_in_placeholders(ids.len()));
        let params: Vec<Value> = ids.iter().map(|id| Value::Text(id.clone())).collect();
        (suffix, params)
    }

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
    /// Used by the priority-inheritance path in `CreateTicketTool::execute` to read the superseded
    /// ticket's priority without loading comments. Priority is immutable after ticket creation, so
    /// this single-column read outside the supersede transaction is safe — there is no TOCTOU race
    /// with the supersede transaction's own SELECT for existence/workspace/phase checks.
    pub(crate) async fn get_ticket_priority(&self, ticket_id: &str) -> Result<Option<i64>> {
        let sql = "SELECT priority FROM tickets WHERE id = ?1";
        self.conn
            .query_optional(sql, db::params![ticket_id], |row| row.get::<i64>(0))
            .await
    }

    /// Build a [`PreparedUpdate`] for an `UPDATE tickets` statement, appending `updated_at = ?`
    /// and `WHERE id = ?` as the last two parameters.
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

    /// Build the SQL and params for a ticket phase transition. Shared by
    /// [`transition_to`](Self::transition_to) and [`transition_to_tx`](Self::transition_to_tx).
    ///
    /// This does **not** use [`Self::build_ticket_update_with_updated_at`]: it has extra SET
    /// columns (`last_transition_actor = ?3`, `done_at = CASE ...`) and extra WHERE conditions (the
    /// optional CAS guard `AND (?5 IS NULL OR phase = ?5)` plus the finished-ticket guard) that do
    /// not fit that helper's fixed pattern.
    ///
    /// `done_at` is set to `?2` (now) when the target is `done` and never cleared: a finished phase
    /// is immutable, so no transition ever leaves `done`.
    ///
    /// `AND (phase <> 'done' OR ?1 = 'done')` puts the finished-ticket rule in the statement:
    /// moving a `done` ticket elsewhere matches zero rows, so no writer can slip a finished ticket
    /// back into work. Writing `done` onto an already-`done` ticket is not a transition and stays
    /// allowed (its `done_at` overwrite is kept).
    fn build_transition_sql(
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        actor: &str,
    ) -> PreparedUpdate {
        let now = db::now();
        let guard: Option<&str> = expected_phase.as_ref().map(TicketPhase::as_ref);
        let sql = "UPDATE tickets SET phase = ?1, updated_at = ?2, \
                    last_transition_actor = ?3, \
                    done_at = CASE WHEN ?1 = 'done' THEN ?2 ELSE done_at END \
                    WHERE id = ?4 AND (?5 IS NULL OR phase = ?5) \
                    AND (phase <> 'done' OR ?1 = 'done')";
        let params: Vec<db::Value> = vec![
            Value::from(target_phase.as_ref()),
            Value::from(now),
            Value::from(actor),
            Value::from(ticket_id),
            Value::from(guard),
        ];
        PreparedUpdate {
            sql: sql.to_string(),
            params,
            ticket_id: ticket_id.to_string(),
        }
    }

    /// Update ticket phase, optionally guarded by an expected phase for CAS-style atomicity. Always
    /// cancels running agents.
    pub async fn transition_to(
        &self,
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        actor: &str,
    ) -> Result<()> {
        let prepared = Self::build_transition_sql(ticket_id, expected_phase, target_phase, actor);
        let rows = self.conn.execute(&prepared.sql, prepared.params).await?;
        if rows == 0 {
            let current = self.get_ticket_phase(ticket_id).await?;
            if Self::is_finished_phase_refusal(current, target_phase) {
                return Err(Self::finished_transition_error(ticket_id, target_phase));
            }
            return Err(match current {
                Some(current) => anyhow::anyhow!(
                    "Ticket {ticket_id} is in phase '{current}', so the change to '{}' \
                     did not apply",
                    target_phase.as_ref(),
                ),
                None => anyhow::anyhow!("Ticket {ticket_id} not found"),
            });
        }
        crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(ticket_id);
        // A terminal transition (Done/Cancelled/Failed) completes the ticket's phase jobs
        // (idempotent; no-op when it has none), so a manual Done/Cancelled does not leave a
        // lingering 'launched' phase job. The pipeline path completes them at its own Done handoff
        // functions, so this only fires for external/manual callers.
        // Guard on the session store being initialized: this is the MANUAL path, and some
        // board-only contexts (tests) never open the session DB.
        if target_phase.is_terminal()
            && let Some(session) = crate::session::SESSIONS.get()
        {
            let _ = crate::jobs::complete_ticket_phase_jobs(&session.conn, ticket_id).await;
        }
        Ok(())
    }

    /// Transactional variant of [`transition_to`](Self::transition_to), using an existing
    /// transaction instead of `self.conn.execute()`. Does NOT cancel registered agents — the caller
    /// cancels them **before** beginning the transaction (or at least before `tx.commit()`) to
    /// avoid orphaned agents on crash.
    ///
    /// # Return value
    ///
    /// `Ok(true)` when the guard matched and the row was updated; `Ok(false)` when no row matched —
    /// the ticket left the expected phase (moved externally while the stage finished) or does not
    /// exist. That miss follows the board layer's claim convention (expected no-op, not an error),
    /// unlike [`transition_to`](Self::transition_to), whose callers perform user-initiated actions
    /// on a ticket that must exist.
    ///
    /// A `done` ticket asked to move elsewhere is the exception: it returns the finished-ticket
    /// refusal as an error, because "the ticket moved externally" would be a false reason for a
    /// phase that can never move. A stale guard onto `done` itself is not a transition, so it keeps
    /// the benign `Ok(false)`.
    pub(crate) async fn transition_to_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
        expected_phase: Option<TicketPhase>,
        target_phase: TicketPhase,
        actor: &str,
    ) -> Result<bool> {
        let prepared = Self::build_transition_sql(ticket_id, expected_phase, target_phase, actor);
        if tx.execute(&prepared.sql, prepared.params).await? > 0 {
            return Ok(true);
        }
        let current = tx
            .query_map_strict(
                "SELECT phase FROM tickets WHERE id = ?1",
                db::params![ticket_id],
                |row| {
                    let phase: String = row.get(0)?;
                    phase.parse()
                },
            )
            .await?
            .into_iter()
            .next();
        if Self::is_finished_phase_refusal(current, target_phase) {
            return Err(Self::finished_transition_error(ticket_id, target_phase));
        }
        Ok(false)
    }

    /// Whether a zero-row phase write was refused by the finished-ticket rule: the ticket is `done`
    /// and the target is not (`done → done` is not a transition, so it keeps CAS-miss semantics).
    fn is_finished_phase_refusal(current: Option<TicketPhase>, target: TicketPhase) -> bool {
        current == Some(TicketPhase::Done) && target != TicketPhase::Done
    }

    /// The one refusal for a phase write on a finished ticket: names the ticket, the attempted
    /// action and what still works, so no caller mistakes it for "not found" or a silent no-op.
    fn finished_ticket_error(ticket_id: &str, action: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "Ticket {ticket_id} is finished (phase 'done') — a finished ticket's phase is \
             immutable, so it cannot be {action}. Only archiving remains; the ticket can \
             still be read and commented on.",
        )
    }

    /// The refusal for a phase transition on a finished ticket — the shared action wording for
    /// both transition paths.
    fn finished_transition_error(ticket_id: &str, target: TicketPhase) -> anyhow::Error {
        Self::finished_ticket_error(ticket_id, &format!("moved to '{}'", target.as_ref()))
    }

    fn ensure_ticket_found(rows: u64, ticket_id: &str) -> Result<()> {
        anyhow::ensure!(rows > 0, "Ticket {ticket_id} not found");
        Ok(())
    }

    /// Record commit metadata on a ticket using an existing transaction; the caller commits or
    /// rolls back.
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

    /// Record the reviewed content base — the content the code reviewers were handed, read before
    /// the round's participants ran. Written only by a round whose code review ran and which then
    /// moved the ticket on; see `verification::record_reviewed_base`. `None` values clear the base
    /// (the ticket becomes never-reviewed).
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

    /// Increment the ticket's shared return-to-development counter inside an existing transaction:
    /// every non-success in either post-development stage goes through this one counter, updated
    /// atomically with the bounce-back transition so it can never drift. Returns the
    /// post-increment count. Uses the guard's row-returning change entry — see
    /// [`TxGuard::execute_one`].
    pub(crate) async fn increment_bounce_count_tx(
        tx: &TxGuard<'_>,
        ticket_id: &str,
    ) -> Result<i64> {
        tx.execute_one(
            "UPDATE tickets SET bounce_count = bounce_count + 1, updated_at = ?1 \
             WHERE id = ?2 RETURNING bounce_count",
            db::params![db::now(), ticket_id],
            |row| row.get::<i64>(0),
        )
        .await
        .map_err(|e| match e {
            turso::Error::QueryReturnedNoRows => {
                anyhow::anyhow!("ticket {ticket_id} not found — bounce counter not incremented")
            }
            other => anyhow::Error::from(other),
        })
    }

    /// Add a comment to a ticket (append-only).
    ///
    /// After persisting it, the comment is routed to any running agents assigned to this ticket via
    /// the message router. Delivery to a running agent is a SOFT DEFERRED message consumed at the
    /// start of the agent's next tool round — it never cancels or aborts the agent. With no agent
    /// registered (it finished before the comment arrived), the next agent working the ticket
    /// surfaces it: a fresh dispatch on a new session re-renders the ticket comment block, and a
    /// resumed deterministic stage session (engineer/sanitation) injects outstanding comments at
    /// the empty-message resume point.
    pub async fn add_comment(&self, ticket_id: &str, role: &str, content: &str) -> Result<()> {
        crate::db::with_tx(&self.conn, ticket_id, "add comment", async |tx| {
            Self::add_comment_tx(tx, ticket_id, role, content).await
        })
        .await?;

        self.route_comment_to_agents(ticket_id, role, content).await;

        Ok(())
    }

    /// Route a newly-persisted comment to any running agents assigned to the ticket: the ticket's
    /// `launched` roster rows across its jobs (the replacement for the historical
    /// `tickets.assigned_to`), delivered through the message router. Best-effort — failures are
    /// logged, not propagated.
    async fn route_comment_to_agents(&self, ticket_id: &str, role: &str, content: &str) {
        // Best-effort: with no sessions store (e.g. a board-only test) there are no agents to route
        // to.
        let Some(sessions) = crate::session::SESSIONS.get() else {
            return;
        };
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

            // The commenter's role. If it does not parse as a standard Role (e.g. "engineer_1" from
            // parallel-agent verdict comments), fall back to Manager — safe because pipeline agents
            // receive comments via try_route() → direct inbox delivery, NOT via the consumer loop;
            // `AgentJob.role` only matters for response delivery in that loop, which pipeline
            // agents never enter.
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
                originating_workspace: None,
                custom_role: None,
                reply_to_custom: None,
                urgent: false,
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

    /// Transactional [`add_comment`](Self::add_comment): inserts the comment and bumps
    /// `updated_at`, without committing or rollback. It does NOT route the comment to running
    /// agents — every current caller is a post-agent phase where none runs; a mid-execution
    /// caller must use `add_comment` or call
    /// [`route_comment_to_agents`](Self::route_comment_to_agents) after the commit.
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

    /// Validate prerequisites for a new ticket: each must exist and belong to the same
    /// workspace. Self-reference is checked separately by the caller (before this call, with
    /// the real ID generated inside the transaction).
    async fn validate_prerequisites(
        tx: &TxGuard<'_>,
        prerequisite_ids: &[String],
        workspace_name: &str,
    ) -> Result<()> {
        // Guard against empty list — Turso rejects `WHERE id IN ()`.
        if prerequisite_ids.is_empty() {
            return Ok(());
        }

        // One round trip for every prerequisite; read through the transaction's own entry — see
        // [`TxGuard::query`].
        let (suffix, params) = Self::in_clause_for_ids(prerequisite_ids);
        let sql = format!("SELECT id, workspace_name FROM tickets {suffix}");
        let rows = tx.query(&sql, params).await?;

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

    /// List all tickets, optionally filtered by workspace and/or phase. The `is_archived = 0`
    /// gate means a ticket appears here iff it is live, which also backs the CDC removal decision.
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

    /// The live tickets of a workspace across all working phases ([`WORKING_PHASES`]) in one
    /// snapshot, grouped phase-major in dispatch order with `priority ASC, created_at DESC`
    /// inside each phase (the global `ORDER BY` plus stable bucketing keeps each phase's
    /// order). A single snapshot, so the polling dispatch sees one consistent view per tick.
    pub(crate) async fn list_working_tickets(
        &self,
        workspace_name: &str,
    ) -> Result<Vec<(TicketPhase, Ticket)>> {
        // The IN-list placeholders derive from WORKING_PHASES so the SQL cannot drift from the
        // constant; the interpolated text stays invariant across calls, keeping the
        // statement-cache contract.
        let phase_placeholders = (2..=WORKING_PHASES.len() + 1)
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT {TICKET_COLUMNS} FROM tickets \
             WHERE workspace_name = ?1 \
             AND phase IN ({phase_placeholders}) \
             AND is_archived = 0 \
             ORDER BY priority ASC, created_at DESC"
        );
        let mut params: Vec<Value> = vec![Value::from(workspace_name)];
        params.extend(WORKING_PHASES.iter().map(|p| Value::from(p.as_ref())));
        let rows = self.conn.query_cached(&sql, params).await?;
        // Bucket each ticket by its index in WORKING_PHASES. Bucketing is stable, so the global
        // `priority ASC, created_at DESC` order is preserved within each phase.
        let mut buckets: Vec<Vec<Ticket>> = vec![Vec::new(); WORKING_PHASES.len()];
        for row in rows {
            let ticket = self.ticket_from_row(&row, LoadComments::No).await?;
            let Some(idx) = WORKING_PHASES.iter().position(|p| *p == ticket.phase) else {
                continue; // defensive: unreachable via the IN filter
            };
            buckets[idx].push(ticket);
        }
        let mut out = Vec::with_capacity(buckets.iter().map(Vec::len).sum());
        for (idx, phase) in WORKING_PHASES.iter().enumerate() {
            out.extend(buckets[idx].drain(..).map(|ticket| (*phase, ticket)));
        }
        Ok(out)
    }

    /// The maintainer's pre-development ticket count (Analysis + Planning + Queued) for one
    /// workspace in a single snapshot — the ticket-creation throttle reading. Archived tickets
    /// are excluded (the GUI can archive in any phase, and archived work must not throttle);
    /// callers treat `Err` as fail-open.
    pub(crate) async fn count_pre_dev_tickets(&self, workspace_name: &str) -> Result<i64> {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM tickets \
                 WHERE workspace_name = ?1 \
                 AND phase IN (?2, ?3, ?4) \
                 AND is_archived = 0",
                db::params![
                    workspace_name,
                    TicketPhase::Analysis.as_ref(),
                    TicketPhase::Planning.as_ref(),
                    TicketPhase::Queued.as_ref(),
                ],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Archive a single ticket by ID: sets `is_archived = 1`, erroring if the ticket does not
    /// exist.
    ///
    /// **Ordering constraint:** the caller must transition the ticket to a terminal state (`done`
    /// or `cancelled`) *before* calling this.
    pub async fn set_archived(&self, ticket_id: &str) -> Result<()> {
        let prepared =
            Self::build_ticket_update_with_updated_at("is_archived = 1", vec![], ticket_id);
        prepared.execute_and_cancel(&self.conn).await
    }

    /// Move all non-archived Queued tickets in the workspace to Planning — the circuit breaker's
    /// drain when a ticket in the same workspace fails. Returns how many moved.
    pub(crate) async fn drain_queued_to_planning(
        &self,
        workspace_name: &str,
        actor: &str,
    ) -> Result<u64> {
        let now = db::now();
        let updated = self
            .conn
            .execute(
                "UPDATE tickets SET phase = ?1, updated_at = ?2, \
                 last_transition_actor = ?3 \
                 WHERE phase = ?4 AND workspace_name = ?5 AND is_archived = 0",
                db::params![
                    TicketPhase::Planning.as_ref(),
                    now,
                    actor,
                    TicketPhase::Queued.as_ref(),
                    workspace_name,
                ],
            )
            .await?;
        Ok(updated)
    }

    async fn archive_stale_cancelled(&self, hours: i64) -> Result<u64> {
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

    /// Archive every non-archived done/cancelled ticket in the installation — every ticket row,
    /// including those whose workspace is gone or was never registered. The Telegram `/archive`
    /// command and the GUI bulk action both come through here, so the two cannot diverge.
    pub async fn archive_all_done_and_cancelled(&self) -> Result<u64> {
        let now = db::now();
        let sql = format!(
            "UPDATE tickets SET is_archived = 1, updated_at = ?1 \
             WHERE phase IN ({}) AND is_archived = 0",
            phase_list_sql_fragment(UNBLOCKING_PHASES),
        );
        let updated = self
            .conn
            .execute(&sql, db::params![now])
            .await
            .context("Failed to archive done/cancelled tickets")?;
        Ok(updated)
    }

    // The GUI board and the Telegram `/board` command must show tickets in exactly the same order;
    // these helpers are the single source of truth both use.

    /// Partition tickets into the three kanban columns in GUI board order: completed
    /// ([`TicketPhase::is_unblocking`]), pipeline ([`TicketPhase::is_pipeline_occupied`] plus
    /// `Queued`), pending (everything else — the safe fallback for unclassified phases). Archived
    /// tickets are excluded.
    /// Non-completed sort by priority ASC (0 = highest) then created_at ASC; completed sort
    /// Done-first by exact done timestamp DESC (created_at fallback), then Cancelled by created_at
    /// DESC.
    #[must_use]
    fn partition_board_tickets(tickets: &[Ticket]) -> (Vec<&Ticket>, Vec<&Ticket>, Vec<&Ticket>) {
        let mut pending = Vec::new();
        let mut pipeline = Vec::new();
        let mut completed = Vec::new();

        for ticket in tickets {
            if ticket.is_archived {
                continue; // hidden from board
            }
            if ticket.phase.is_unblocking() {
                completed.push(ticket);
            } else if ticket.phase.is_pipeline_occupied() || ticket.phase == TicketPhase::Queued {
                pipeline.push(ticket);
            } else {
                // Unknown future phases silently default to pending — the safe bucket.
                pending.push(ticket);
            }
        }

        // Pending and pipeline: priority ASC (0 = highest) then oldest-first;
        // completed: Done newest-done-first (DESC), then Cancelled newest-first below them.
        // created_at is an ISO 8601 string, so lexical sort is chronological.
        for column in [&mut pending, &mut pipeline] {
            column.sort_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then(a.created_at.cmp(&b.created_at))
            });
        }
        completed.sort_by(|a, b| {
            let (a_done, b_done) = (a.phase == TicketPhase::Done, b.phase == TicketPhase::Done);
            match (a_done, b_done) {
                // Done first, newest completion on top (created_at fallback for Done tickets with
                // no done_at, e.g. test-created ones).
                (true, true) => Self::board_done_sort_key(b).cmp(Self::board_done_sort_key(a)),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => b.created_at.cmp(&a.created_at),
            }
        });

        (pending, pipeline, completed)
    }

    /// The four board sections in display order: In Progress (pipeline minus Queued), Queued,
    /// Pending, Completed — shared by the GUI columns and the Telegram `/board` listing so the two
    /// cannot diverge. Empty sections are included (callers skip them).
    #[must_use]
    pub fn board_sections(tickets: &[Ticket]) -> [Vec<&Ticket>; 4] {
        let (pending, pipeline, completed) = Self::partition_board_tickets(tickets);
        let in_progress = pipeline
            .iter()
            .filter(|t| t.phase != TicketPhase::Queued)
            .copied()
            .collect();
        let queued = pipeline
            .iter()
            .filter(|t| t.phase == TicketPhase::Queued)
            .copied()
            .collect();
        [in_progress, queued, pending, completed]
    }

    /// Flatten [`Self::board_sections`] into one display-order list — the exact order the GUI board
    /// shows. The Telegram `/board` command uses it so its listing cannot diverge from the GUI.
    #[must_use]
    pub fn board_display_order(tickets: &[Ticket]) -> Vec<&Ticket> {
        Self::board_sections(tickets)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Completion ordering key: the exact done timestamp, falling back to creation time when
    /// `done_at` is absent.
    fn board_done_sort_key(ticket: &Ticket) -> &str {
        ticket.done_at.as_deref().unwrap_or(&ticket.created_at)
    }

    // The installation-wide archived search used by
    // [`SearchArchivedTicketsTool`](crate::tools::search_archived_tickets), and [`search_by_fts`],
    // which backs the GUI sidebar search. The board owns the FTS schema (`ngram` tokenizer and
    // index name); the tool layer owns only the presentation.

    /// Search archived tickets by FTS keyword match, installation-wide: up to `limit` ticket ids,
    /// highest score first, across every workspace's archived tickets.
    pub(crate) async fn search_archived_by_fts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let sanitized = crate::db::sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let sql = format!(
            "SELECT id FROM (SELECT t.id AS id, t.is_archived AS archived, \
             fts_score(t.title, ?1) AS score FROM tickets t WHERE t.title MATCH ?1) \
             WHERE archived = 1 ORDER BY score DESC LIMIT {limit}"
        );
        match self
            .conn
            .query_map(&sql, db::params![sanitized.clone()], |row| {
                let id: String = row.get(0)?;
                Ok::<_, anyhow::Error>(id)
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

    /// Search tickets (active and archived) by FTS keyword match across every workspace: full
    /// [`Ticket`] objects (without comments) by FTS relevance (`fts_score DESC`), up to `limit`.
    pub async fn search_by_fts(&self, query: &str, limit: usize) -> Result<Vec<Ticket>> {
        let sanitized = crate::db::sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        // An explicit `FROM tickets t` alias lets `fts_score(t.title, ?1)` resolve — the
        // `select_tickets` helper does not support table aliases in FTS scoring expressions.
        let sql = format!(
            "SELECT {TICKET_COLUMNS} \
             FROM tickets t \
             WHERE t.title MATCH ?1 \
             ORDER BY fts_score(t.title, ?1) DESC \
             LIMIT {limit}"
        );
        let rows = self
            .conn
            .query(&sql, db::params![sanitized])
            .await
            .inspect_err(|e| warn!(error = %e, "Ticket FTS search failed"))?;
        let mut tickets = Vec::with_capacity(rows.len());
        for row in rows {
            tickets.push(
                self.ticket_from_row(&row, LoadComments::No)
                    .await
                    .inspect_err(
                        |e| warn!(error = %e, "Ticket row failed to parse in FTS search"),
                    )?,
            );
        }
        Ok(tickets)
    }
}
