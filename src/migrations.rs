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
pub(crate) const SESSION_MIGRATIONS: &[Migration] = &[
    Migration {
        id: "001_session_token_length",
        sql: "ALTER TABLE session_metadata ADD COLUMN token_length INTEGER",
        guard: Some(("session_metadata", "token_length")),
    },
    Migration {
        id: "002_session_message_count",
        sql: "ALTER TABLE session_metadata ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0; \
              UPDATE session_metadata SET message_count = (SELECT COUNT(*) FROM sessions \
              WHERE sessions.agent_id = session_metadata.agent_id)",
        guard: Some(("session_metadata", "message_count")),
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turso::{column_exists, run_pending_migrations};

    /// Old-DB schema: `session_metadata` WITHOUT the token-length or
    /// message-count columns — simulates the live pre-migration database
    /// (upgrade-in-place path). Includes the `sessions` table so the
    /// message-count backfill is exercised against real rows.
    const OLD_SCHEMA: &str = "CREATE TABLE sessions (\
         id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, \
         role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);\
         CREATE TABLE session_metadata (agent_id TEXT PRIMARY KEY, last_activity TEXT NOT NULL);";

    /// Fresh-DB schema: `session_metadata` already declares both migrated
    /// columns — simulates the wipe-and-recreate path where the SCHEMA
    /// produced the migrated shape.
    const FRESH_SCHEMA: &str = "CREATE TABLE sessions (\
         id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, \
         role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);\
         CREATE TABLE session_metadata (\
         agent_id TEXT PRIMARY KEY, last_activity TEXT NOT NULL, \
         token_length INTEGER, message_count INTEGER NOT NULL DEFAULT 0);";

    async fn applied_ids(conn: &crate::turso::Connection) -> Vec<String> {
        conn.query("SELECT id FROM schema_migrations ORDER BY id", ())
            .await
            .expect("read tracking table")
            .into_iter()
            .map(|row| row.get::<String>(0).expect("id column"))
            .collect()
    }

    async fn message_count(conn: &crate::turso::Connection, agent_id: &str) -> i64 {
        conn.query_optional(
            "SELECT message_count FROM session_metadata WHERE agent_id = ?1",
            (turso::Value::Text(agent_id.to_string()),),
            |row| row.get::<i64>(0),
        )
        .await
        .expect("read message_count")
        .expect("session exists")
    }

    #[tokio::test]
    async fn migrations_alter_old_db_exactly_once_with_backfill() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::turso::open_with_schema(&tmp.path().join("sessions.db"), OLD_SCHEMA)
            .await
            .expect("open old-db test store");
        assert!(
            !column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "old DB must lack token_length before the migration"
        );
        assert!(
            !column_exists(&conn, "session_metadata", "message_count")
                .await
                .unwrap(),
            "old DB must lack message_count before the migration"
        );

        // Seed a pre-migration live-store shape: sessions with messages but
        // no metadata count column (the migration must backfill it).
        for (agent, n) in [("sess_a", 3), ("sess_b", 1), ("sess_c", 0)] {
            conn.execute(
                "INSERT INTO session_metadata (agent_id, last_activity) VALUES (?1, '2026-01-01T00:00:00Z')",
                (turso::Value::Text(agent.to_string()),),
            )
            .await
            .expect("seed metadata");
            for i in 0..n {
                conn.execute(
                    "INSERT INTO sessions (agent_id, role, content, created_at) VALUES (?1, 'user', ?2, '2026-01-01T00:00:00Z')",
                    (turso::Value::Text(agent.to_string()), turso::Value::Text(format!("m{i}"))),
                )
                .await
                .expect("seed message");
            }
        }

        run_pending_migrations(&conn, "sessions", SESSION_MIGRATIONS)
            .await
            .expect("first run applies both migrations");
        assert!(
            column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "ALTER must add token_length"
        );
        assert!(
            column_exists(&conn, "session_metadata", "message_count")
                .await
                .unwrap(),
            "ALTER must add message_count"
        );
        // Backfill: counts match the historical COUNT(s.id) definition —
        // system prompts, tool frames, and tool results all count (here all
        // rows are plain 'user' messages; the definition is one row per
        // session row).
        assert_eq!(message_count(&conn, "sess_a").await, 3);
        assert_eq!(message_count(&conn, "sess_b").await, 1);
        assert_eq!(message_count(&conn, "sess_c").await, 0);
        assert_eq!(
            applied_ids(&conn).await,
            vec!["001_session_token_length", "002_session_message_count"],
            "both migrations recorded in order"
        );

        // Re-running (e.g. a later boot) is a strict no-op — never twice.
        run_pending_migrations(&conn, "sessions", SESSION_MIGRATIONS)
            .await
            .expect("second run is a no-op");
        assert_eq!(applied_ids(&conn).await.len(), 2, "never re-run");
    }

    #[tokio::test]
    async fn migration_skips_sql_on_fresh_db_and_still_records_applied() {
        // The user's wipe-and-recreate sequence: the SCHEMA already contains
        // the migrated columns, so the ALTERs must NOT fire (duplicate-column
        // failure).
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::turso::open_with_schema(&tmp.path().join("sessions.db"), FRESH_SCHEMA)
            .await
            .expect("open fresh-db test store");
        assert!(
            column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "fresh DB has token_length from the SCHEMA"
        );
        assert!(
            column_exists(&conn, "session_metadata", "message_count")
                .await
                .unwrap(),
            "fresh DB has message_count from the SCHEMA"
        );

        run_pending_migrations(&conn, "sessions", SESSION_MIGRATIONS)
            .await
            .expect("guard turns the pending migrations into recorded no-ops");
        assert!(
            column_exists(&conn, "session_metadata", "token_length")
                .await
                .unwrap(),
            "column still present, exactly once"
        );
        assert!(
            column_exists(&conn, "session_metadata", "message_count")
                .await
                .unwrap(),
            "column still present, exactly once"
        );
        assert_eq!(
            applied_ids(&conn).await,
            vec!["001_session_token_length", "002_session_message_count"],
            "both migrations recorded as applied even though the SQL was skipped"
        );
    }
}
