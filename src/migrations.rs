//! Centralized database migrations.
//!
//! Each migration is a ready-made SQL statement/script applied exactly once,
//! in order, tracked via a per-store `schema_migrations` table. The runner
//! lives in [`crate::turso::run_pending_migrations`] and fires from the
//! store's `post_open` hook at initialization.
//!
//! Adding a migration: append a new [`crate::turso::Migration`] to the store's
//! list with a stable unique id, the ready SQL, and — when the fresh-DB
//! SCHEMA already contains the migrated shape — an existence guard so the
//! wipe-and-recreate path records it as applied instead of failing with a
//! duplicate-column error.

use crate::turso::Migration;

/// Migrations for the sessions store (sessions.db).
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
pub(crate) const SESSION_MIGRATIONS: &[Migration] = &[Migration {
    id: "001_session_token_length",
    sql: "ALTER TABLE session_metadata ADD COLUMN token_length INTEGER",
    guard: Some(("session_metadata", "token_length")),
}];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turso::{column_exists, run_pending_migrations};

    /// Old-DB schema: `session_metadata` WITHOUT the token-length column —
    /// simulates a pre-migration database (upgrade-in-place path).
    const OLD_SCHEMA: &str =
        "CREATE TABLE session_metadata (agent_id TEXT PRIMARY KEY, last_activity TEXT NOT NULL);";

    /// Fresh-DB schema: `session_metadata` already declares the column —
    /// simulates the wipe-and-recreate path where the SCHEMA produced the
    /// migrated shape.
    const FRESH_SCHEMA: &str = "CREATE TABLE session_metadata (\
         agent_id TEXT PRIMARY KEY, last_activity TEXT NOT NULL, token_length INTEGER);";

    async fn applied_ids(conn: &crate::turso::Connection) -> Vec<String> {
        conn.query("SELECT id FROM schema_migrations ORDER BY id", ())
            .await
            .expect("read tracking table")
            .into_iter()
            .map(|row| row.get::<String>(0).expect("id column"))
            .collect()
    }

    #[tokio::test]
    async fn migration_alters_old_db_exactly_once() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::turso::open_with_schema(&tmp.path().join("sessions.db"), OLD_SCHEMA)
            .await
            .expect("open old-db test store");
        assert!(
            !column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "old DB must lack the column before the migration"
        );

        run_pending_migrations(&conn, "sessions", SESSION_MIGRATIONS)
            .await
            .expect("first run applies the ALTER");
        assert!(
            column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "ALTER must add the column"
        );
        assert_eq!(applied_ids(&conn).await, vec!["001_session_token_length"]);

        // Re-running (e.g. a later boot) is a strict no-op — never twice.
        run_pending_migrations(&conn, "sessions", SESSION_MIGRATIONS)
            .await
            .expect("second run is a no-op");
        assert_eq!(applied_ids(&conn).await.len(), 1, "never re-run");
    }

    #[tokio::test]
    async fn migration_skips_sql_on_fresh_db_and_still_records_applied() {
        // The user's wipe-and-recreate sequence: the SCHEMA already contains
        // the column, so the ALTER must NOT fire (duplicate-column failure).
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::turso::open_with_schema(&tmp.path().join("sessions.db"), FRESH_SCHEMA)
            .await
            .expect("open fresh-db test store");
        assert!(
            column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "fresh DB has the column from the SCHEMA"
        );

        run_pending_migrations(&conn, "sessions", SESSION_MIGRATIONS)
            .await
            .expect("guard turns the pending migration into a recorded no-op");
        assert!(
            column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "column still present, exactly once"
        );
        assert_eq!(
            applied_ids(&conn).await,
            vec!["001_session_token_length"],
            "migration recorded as applied even though the SQL was skipped"
        );
    }
}
