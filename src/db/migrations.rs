//! Centralized database migrations.
//!
//! Each migration is a ready-made SQL statement/script applied exactly once,
//! in order, tracked via the consolidated `schema_migrations` table. The
//! runner ([`crate::db::run_pending_migrations`]) fires from the
//! consolidated-store init path.
//!
//! Adding a migration: append a new [`crate::db::Migration`] to the store's
//! list with a stable unique id, the ready SQL, and — when the fresh-DB
//! SCHEMA already contains the migrated shape — an existence guard so the
//! wipe-and-recreate path records it as applied instead of failing with a
//! duplicate-column error.

use crate::db::{Migration, MigrationGuard};

/// Migrations for the sessions store (applied in the consolidated domain
/// database, alongside the merged board history).
///
/// **001 — session token length.** Adds the real provider-reported session
/// length column (`session_metadata.token_length`) — the durable backing for
/// the summarization decision and the Running Agents card metric. The
/// fresh-DB SCHEMA (`src/session/mod.rs`) already declares the column, so the
/// ALTER only fires on pre-existing databases; the existence guard makes both
/// paths converge to the same state and the migration is recorded as applied
/// either way (upgrade-in-place: the ALTER runs; wipe-and-recreate: the
/// column is already present and the SQL is skipped). Applied exactly once
/// per database.
///
/// **002 — session message count.** Adds the denormalized per-session message
/// count (`session_metadata.message_count`) so the Sessions page list reads
/// counts from lightweight metadata instead of a per-refresh LEFT JOIN +
/// GROUP BY over the full `sessions` message table. The ALTER declares the
/// column `NOT NULL DEFAULT 0` (turso follows SQLite's ADD COLUMN rule: NOT
/// NULL is allowed only with a non-NULL default, which the read path
/// relies on — a nullable read via `row.get::<i64>` would silently drop
/// sessions from the list). The one-time correlated-subquery backfill
/// (`COUNT(s.id)` per existing session) keeps the pre-migration live store on
/// the 'same counts' semantics the list had before — without it every existing
/// session would show 0 messages. The count definition matches the historical
/// list exactly: one row per `sessions` row (system prompts, tool-call
/// frames, and tool results all count — the counter is maintained in lockstep
/// with message inserts). The fresh-DB SCHEMA already declares the column, so
/// the existence guard makes both paths converge (wipe-and-recreate needs no
/// backfill — fresh databases start empty).
///
/// **003 — drop `ticket_stage_jobs.round`.** The analysis rework is one-round
/// per job, so the `round` column is dead. The fresh-DB SCHEMA no longer
/// declares it; the drop guard makes both paths converge (fresh DB: column
/// already absent, SQL skipped, recorded as applied; upgraded DB: the DROP
/// fires).
///
/// **004 — reset pre-rework ticket-jobs data.** A deliberate one-time data
/// reset for the clean upgrade: deletes the pre-rework `ticket_stage` job rows
/// (the only ticket-job kind that could exist at migration time — the reworked
/// `ticket_analysis`/`ticket_implementation` kinds are created only by the
/// rework, and `ticket_journey` never existed). Cascades to `ticket_stage_jobs`
/// child rows and agent rosters (FK ON DELETE CASCADE); non-ticket jobs
/// (research/analyze/cleanup) are left intact. No backward compat — this is a
/// deliberate one-time data reset for the clean upgrade.
///
/// **005 — drop `ticket_stage_jobs.phase`.** The phase column shipped in a
/// prior schema but is now redundant — the ticket phase is the only stored
/// representation, so it is not derivable from a job-side mirror. The fresh-DB
/// SCHEMA no longer declares
/// it; the drop guard makes both paths converge (fresh DB: column already
/// absent, SQL skipped, recorded as applied; upgraded DB: the DROP fires).
///
/// **006 — rename `ticket_stage_jobs` → `ticket_jobs`.** The child table ships
/// under the new name. On a fresh DB the SCHEMA creates `ticket_jobs`
/// directly, so the rename is skipped and recorded as applied. On an
/// upgraded-in-place DB the old `ticket_stage_jobs` table holds the data; the
/// migration first drops the empty `ticket_jobs` that the fresh-DB SCHEMA batch
/// created on this same open (it is brand new and empty — no data loss), then
/// renames the old table under the new name so the new code reads the
/// pre-existing rows. Both paths converge to a single `ticket_jobs` table.
///
/// **consolidate_002 — drop `ticket_jobs.stage`.** The phase column no longer
/// mirrors `tickets.phase`; per the single-DB consolidation the ticket phase is
/// the only stored representation, so the job-side stage mirror is dropped. On
/// an upgrade-in-place DB the 006 rename has already moved the legacy
/// `ticket_stage_jobs` (still carrying `stage`) to `ticket_jobs`;
/// consolidate_002 then drops the column. The fresh-DB SCHEMA no longer
/// declares it, so the drop guard makes both paths converge. It uses the
/// `consolidate_` post-consolidation prefix (per the documented convention)
/// and runs after 006 and before the one-time consolidation import.
pub(crate) const SESSION_MIGRATIONS: &[Migration] = &[
    Migration {
        id: "001_session_token_length",
        sql: "ALTER TABLE session_metadata ADD COLUMN token_length INTEGER",
        guard: Some(MigrationGuard::ColumnExists {
            table: "session_metadata",
            column: "token_length",
        }),
    },
    Migration {
        id: "002_session_message_count",
        sql: "ALTER TABLE session_metadata ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0; \
              UPDATE session_metadata SET message_count = (SELECT COUNT(*) FROM sessions \
              WHERE sessions.agent_id = session_metadata.agent_id)",
        guard: Some(MigrationGuard::ColumnExists {
            table: "session_metadata",
            column: "message_count",
        }),
    },
    Migration {
        id: "003_drop_ticket_stage_jobs_round",
        sql: "ALTER TABLE ticket_stage_jobs DROP COLUMN round",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "ticket_stage_jobs",
            column: "round",
        }),
    },
    Migration {
        id: "004_reset_implementation_jobs",
        sql: "DELETE FROM jobs WHERE kind = 'ticket_stage'",
        guard: None,
    },
    Migration {
        id: "005_drop_ticket_stage_jobs_phase",
        sql: "ALTER TABLE ticket_stage_jobs DROP COLUMN phase",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "ticket_stage_jobs",
            column: "phase",
        }),
    },
    Migration {
        id: "006_rename_ticket_stage_jobs_to_ticket_jobs",
        sql: "DROP TABLE IF EXISTS ticket_jobs; \
              ALTER TABLE ticket_stage_jobs RENAME TO ticket_jobs",
        guard: Some(MigrationGuard::TableExists {
            table: "ticket_stage_jobs",
        }),
    },
    Migration {
        id: "consolidate_002_drop_ticket_jobs_stage",
        sql: "ALTER TABLE ticket_jobs DROP COLUMN stage",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "ticket_jobs",
            column: "stage",
        }),
    },
    // ── Job-per-phase rework ──────────────────────────────────────────────
    // The pipeline moves from one long-lived `ticket_implementation` job to a
    // short-lived job per pipeline phase, keyed by `tickets.phase`. Jobs now
    // carry `ticket_id` directly; the `ticket_jobs` child table and
    // `jobs.paused_frozen` are removed. Old ticket_analysis/ticket_implementation
    // rows are discarded on upgrade (no backward compatibility).
    Migration {
        id: "consolidate_003_jobs_ticket_id",
        sql: "ALTER TABLE jobs ADD COLUMN ticket_id TEXT REFERENCES tickets(id)",
        guard: Some(MigrationGuard::ColumnExists {
            table: "jobs",
            column: "ticket_id",
        }),
    },
    Migration {
        id: "consolidate_004_discard_old_ticket_jobs",
        sql: "DELETE FROM jobs WHERE kind IN ('ticket_analysis', 'ticket_implementation')",
        guard: None,
    },
    Migration {
        id: "consolidate_005_drop_ticket_jobs",
        sql: "DROP TABLE IF EXISTS ticket_jobs",
        guard: None,
    },
    Migration {
        id: "consolidate_006_drop_jobs_paused_frozen",
        sql: "ALTER TABLE jobs DROP COLUMN paused_frozen",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "jobs",
            column: "paused_frozen",
        }),
    },
    Migration {
        id: "consolidate_007_jobs_phase_ticket_index",
        sql: "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_phase_ticket \
              ON jobs(kind, ticket_id) WHERE ticket_id IS NOT NULL",
        guard: None,
    },
];

/// Migrations for the board store (applied in the consolidated domain
/// database, alongside the merged session history).
///
/// **001 — drop `tickets.pipeline_reservation`.** The job-centric pipeline
/// rework removes the ticket-level pipeline reservation in favor of the
/// per-phase job (see `src/jobs.rs`). The fresh-DB SCHEMA
/// (`src/pipeline/board.rs`) no longer declares the column, so the ALTER only fires on
/// pre-existing databases; the drop guard makes both paths converge (fresh-DB
/// path: column already absent, SQL skipped, migration recorded as applied).
///
/// **002 — drop `tickets.assigned_to`.** Replaced by the `agents` roster in
/// the jobs store (`status='launched'` rows bound to the ticket's per-phase jobs),
/// which drives comment routing and the mid-execution re-dispatch guard.
///
/// **003 — reset non-terminal tickets.** A deliberate one-time data reset for
/// the reworked pipeline phase semantics: every non-terminal, non-archived
/// ticket is returned to `Backlog`. Only non-terminal, non-archived rows are
/// reset (Done/Cancelled/Failed and archived tickets are untouched). Phase is
/// stored lowercase snake_case; `TicketPhase::Backlog.as_ref() == "backlog"`.
/// No backward compat — this is a deliberate reset.
///
/// **004 — drop `tickets.review_base_count`.** A dead column that exists only
/// as drift in the live board DB — it is not declared by the current SCHEMA
/// and is referenced by no production code. The fresh-DB SCHEMA never declares
/// it, so the drop guard makes both paths converge (fresh DB: column already
/// absent, SQL skipped, recorded as applied; upgraded DB: the DROP fires).
pub(crate) const BOARD_MIGRATIONS: &[Migration] = &[
    Migration {
        id: "001_drop_ticket_pipeline_reservation",
        sql: "ALTER TABLE tickets DROP COLUMN pipeline_reservation",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "tickets",
            column: "pipeline_reservation",
        }),
    },
    Migration {
        id: "002_drop_ticket_assigned_to",
        sql: "ALTER TABLE tickets DROP COLUMN assigned_to",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "tickets",
            column: "assigned_to",
        }),
    },
    Migration {
        id: "003_reset_nonterminal_tickets",
        sql: "UPDATE tickets SET phase = 'backlog' WHERE phase NOT IN ('done','cancelled','failed') AND is_archived = 0",
        guard: None,
    },
    Migration {
        id: "004_drop_tickets_review_base_count",
        sql: "ALTER TABLE tickets DROP COLUMN review_base_count",
        guard: Some(MigrationGuard::ColumnDropped {
            table: "tickets",
            column: "review_base_count",
        }),
    },
];

/// Apply the consolidated **domain** migration history to the single
/// consolidated database.
///
/// This is the SINGLE migration runner for the one consolidated domain file.
/// It applies the board and sessions migration lists in order against the same
/// unified `schema_migrations` table.
///
/// # Id preservation (decision 2)
///
/// The ids are the ORIGINAL per-store ids, deliberately NOT renumbered — two
/// of them (`003_reset_nonterminal_tickets`, `004_reset_implementation_jobs`)
/// are unguarded one-time data resets that would re-run destructively if
/// renumbered: a renumbered id would not be found in the unified tracking
/// table and would fire again on migrated data. The tracking table therefore
/// holds the full merged history and never re-runs a reset. All other domain
/// stores (workspaces, users, config, chat_history) carry no migration
/// history.
///
/// # Runs BEFORE the one-time import
///
/// [`crate::db::open_consolidated_store`] calls this before the one-time
/// consolidation import. That ordering is intentional: the two unguarded data
/// resets fire against the EMPTY consolidated file (no-ops) and are recorded as
/// applied, so the import that follows copies the legacy user tables
/// **verbatim** — pre-rework phases and stale `kind='ticket_stage'` jobs are
/// never destructively reset/deleted (migrate without losing data).
///
/// Future schema changes run only here, in the consolidated database, using a
/// new non-colliding id prefix (e.g. `consolidate_002_...`).
pub(crate) async fn run_domain_migrations(conn: &crate::db::Connection) -> anyhow::Result<()> {
    // `run_pending_migrations` is idempotent per-migration: a second call
    // re-reads the unified tracking table and skips already-applied ids. The
    // `store` label is the consolidated scope, not a legacy per-store file.
    crate::db::run_pending_migrations(conn, "consolidated", BOARD_MIGRATIONS).await?;
    crate::db::run_pending_migrations(conn, "consolidated", SESSION_MIGRATIONS).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{column_exists, run_pending_migrations};

    async fn applied_ids(conn: &crate::db::Connection) -> Vec<String> {
        conn.query("SELECT id FROM schema_migrations ORDER BY id", ())
            .await
            .expect("read tracking table")
            .into_iter()
            .map(|row| row.get::<String>(0).expect("id column"))
            .collect()
    }

    // ── Board DROP COLUMN migrations (the inverse guard direction) ──────

    /// Old board schema: `tickets` WITH the columns that the job-centric
    /// rework drops (`pipeline_reservation`, `assigned_to`) plus the dead
    /// drift column `review_base_count` (migration 004 drops it) — simulates
    /// the pre-migration live board.db (upgrade-in-place path). Includes
    /// `phase` and `is_archived` so migration 003's reset UPDATE can run.
    const OLD_BOARD_SCHEMA: &str = "CREATE TABLE tickets (\
         id TEXT PRIMARY KEY, title TEXT NOT NULL, pipeline_reservation INTEGER, \
         assigned_to TEXT, review_base_count INTEGER, \
         phase TEXT NOT NULL DEFAULT 'backlog', \
         is_archived INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);";

    /// Fresh board schema: `tickets` WITHOUT any dropped column (the current
    /// SCHEMA no longer declares them) — simulates the wipe-and-recreate path.
    /// Includes `phase` and `is_archived` so migration 003's reset UPDATE runs.
    const FRESH_BOARD_SCHEMA: &str = "CREATE TABLE tickets (\
         id TEXT PRIMARY KEY, title TEXT NOT NULL, \
         phase TEXT NOT NULL DEFAULT 'backlog', \
         is_archived INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);";

    #[tokio::test]
    async fn board_drop_column_migrations_run_on_old_db() {
        // The DROP direction is the inverse of the tested ADD path; exercises
        // the ACTUAL `ALTER TABLE ... DROP COLUMN` DDL against turso on an
        // upgraded-in-place database.
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(&tmp.path().join("board.db"), OLD_BOARD_SCHEMA)
            .await
            .expect("open old board test store");
        assert!(
            column_exists(&conn, "tickets", "pipeline_reservation")
                .await
                .unwrap(),
            "old board DB has pipeline_reservation before migration"
        );
        assert!(
            column_exists(&conn, "tickets", "assigned_to")
                .await
                .unwrap(),
            "old board DB has assigned_to before migration"
        );
        assert!(
            column_exists(&conn, "tickets", "review_base_count")
                .await
                .unwrap(),
            "old board DB has the dead review_base_count drift column before migration"
        );

        run_pending_migrations(&conn, "board", BOARD_MIGRATIONS)
            .await
            .expect("apply board DROP COLUMN migrations");
        assert!(
            !column_exists(&conn, "tickets", "pipeline_reservation")
                .await
                .unwrap(),
            "ALTER must drop pipeline_reservation"
        );
        assert!(
            !column_exists(&conn, "tickets", "assigned_to")
                .await
                .unwrap(),
            "ALTER must drop assigned_to"
        );
        assert!(
            !column_exists(&conn, "tickets", "review_base_count")
                .await
                .unwrap(),
            "ALTER must drop the dead review_base_count column"
        );
        assert_eq!(
            applied_ids(&conn).await,
            vec![
                "001_drop_ticket_pipeline_reservation",
                "002_drop_ticket_assigned_to",
                "003_reset_nonterminal_tickets",
                "004_drop_tickets_review_base_count",
            ],
            "all DROP migrations + the reset recorded in order"
        );

        // Re-running (e.g. a later boot) is a strict no-op — never twice.
        run_pending_migrations(&conn, "board", BOARD_MIGRATIONS)
            .await
            .expect("second run is a no-op");
        assert_eq!(applied_ids(&conn).await.len(), 4, "never re-run");
    }

    #[tokio::test]
    async fn board_drop_column_migrations_skip_on_fresh_db_and_record_applied() {
        // Wipe-and-recreate: the SCHEMA no longer declares the columns, so the
        // DROPs must NOT fire (the DROP would fail on a missing column); the
        // guard turns them into recorded no-ops.
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(&tmp.path().join("board.db"), FRESH_BOARD_SCHEMA)
            .await
            .expect("open fresh board test store");
        assert!(
            !column_exists(&conn, "tickets", "pipeline_reservation")
                .await
                .unwrap(),
            "fresh board DB has no pipeline_reservation"
        );
        assert!(
            !column_exists(&conn, "tickets", "assigned_to")
                .await
                .unwrap(),
            "fresh board DB has no assigned_to"
        );
        assert!(
            !column_exists(&conn, "tickets", "review_base_count")
                .await
                .unwrap(),
            "fresh board DB has no dead review_base_count column"
        );

        run_pending_migrations(&conn, "board", BOARD_MIGRATIONS)
            .await
            .expect("guard turns pending drops into recorded no-ops");
        assert!(
            !column_exists(&conn, "tickets", "pipeline_reservation")
                .await
                .unwrap(),
            "column still absent (never created)"
        );
        assert!(
            !column_exists(&conn, "tickets", "assigned_to")
                .await
                .unwrap(),
            "column still absent (never created)"
        );
        assert!(
            !column_exists(&conn, "tickets", "review_base_count")
                .await
                .unwrap(),
            "review_base_count still absent (never created)"
        );
        assert_eq!(
            applied_ids(&conn).await,
            vec![
                "001_drop_ticket_pipeline_reservation",
                "002_drop_ticket_assigned_to",
                "003_reset_nonterminal_tickets",
                "004_drop_tickets_review_base_count",
            ],
            "all recorded as applied even though the DDL was skipped"
        );
    }
}
