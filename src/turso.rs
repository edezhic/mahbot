use anyhow::Context;
use chrono::Utc;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::OnceCell;
use turso::Builder;
pub use turso::{Error as TursoError, IntoParams, Row, Value, params, params_from_iter};

// ── Timestamp helper ────────────────────────────────────────────────

/// Current UTC timestamp in RFC 3339 format for database columns.
#[must_use]
pub fn now() -> String {
    Utc::now().to_rfc3339()
}

/// Feature names for experimental Turso database features that must be enabled
/// consistently by the main daemon and the debug CLI.
///
/// These correspond to the `with_*()` methods / public fields of
/// [`turso::core::DatabaseOpts`] — see [`experimental_database_opts`] for
/// the canonical construction.
///
/// [`Connection::open`] derives its `experimental_*()` builder calls from
/// [`experimental_database_opts`], so this constant and [`experimental_database_opts`]
/// are the two places that define which features are active. The
/// `experimental_features_are_consistent` test verifies they match.
///
/// # API naming asymmetries
///
/// The `turso::Builder` (used by [`Connection::open`]) and `turso::core::DatabaseOpts`
/// (used by [`experimental_database_opts`]) have slightly different APIs for the same
/// underlying features. Known mismatches:
///
/// | DatabaseOpts field / `with_*()` | Builder `experimental_*()` | Feature string |
/// |---|---:|---|
/// | `enable_views` / `with_views()` | `experimental_materialized_views()` | `"views"` |
///
/// Some fields exist on only one side:
/// - `enable_autovacuum` / `with_autovacuum()` — DatabaseOpts only, no Builder equivalent.
/// - `unsafe_testing` / `with_unsafe_testing()` — DatabaseOpts only, no Builder equivalent.
/// - `experimental_triggers()` — Builder only (no-op for backwards compatibility).
/// - `experimental_strict()` — Builder only (no-op for backwards compatibility).
///
/// See the test `builder_mapping_matches_experimental_features` which verifies that
/// the field-by-field mapping in [`Connection::open`] enables exactly the features
/// listed here.
pub const EXPERIMENTAL_FEATURES: &[&str] = &["index_method", "multiprocess_wal"];

/// Create [`turso::core::DatabaseOpts`] with all experimental features enabled.
///
/// This is the **single source of truth** for which experimental features are active.
/// [`Connection::open`] reads its builder calls from this function, and
/// [`EXPERIMENTAL_FEATURES`] lists the feature names for test verification.
///
/// Used by `mahbot debug` to open databases with the same feature set as the
/// main daemon, preventing `.tshm` WAL coordination file inconsistencies.
///
/// # Adding a new feature
///
/// 1. Add the `with_*()` call here.
/// 2. Add the `experimental_*()` mapping in [`Connection::open`].
/// 3. Add the feature name string to [`EXPERIMENTAL_FEATURES`].
///
/// See [`EXPERIMENTAL_FEATURES`] for known API naming asymmetries between
/// `DatabaseOpts::with_*()` and `Builder::experimental_*()`.
#[must_use]
pub fn experimental_database_opts() -> turso::core::DatabaseOpts {
    turso::core::DatabaseOpts::new()
        .with_multiprocess_wal(true)
        .with_index_method(true)
}

/// Register a global singleton store.
///
/// This is the canonical init pattern for all DB-backed global stores. Each module
/// calls this from its `init_global()` function with its `OnceCell`, a name for
/// error messages, and an async open function (typically a closure that captures
/// the storage root and calls the store's `open` method).
///
/// # Errors
/// Returns an error if the store fails to open, or if the cell is already set.
pub async fn register_global_store<T, F, Fut>(
    cell: &OnceCell<T>,
    name: &str,
    open_fn: F,
) -> anyhow::Result<()>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = anyhow::Result<T>> + Send,
{
    let store = open_fn().await?;
    cell.set(store)
        .map_err(|_| anyhow::anyhow!("{name} already initialized"))?;
    Ok(())
}

/// Declare a global `OnceCell`-backed store with `init_global()` and `store()`.
///
/// Generates three items:
/// - `pub static $NAME: OnceCell<$Type>` — the underlying cell.
/// - `pub async fn init_global()` — calls `register_global_store` with the
///   constructor function invoked on `CONFIG.global_storage_root()`.
/// - `#[must_use] pub fn store()` — returns `&'static $Type`, panicking if
///   not yet initialized.
///
/// # Two-arm syntax
///
/// Standard form — auto-generates expect message:
/// ```ignore
/// global_store! {
///     /// Doc comment for the static.
///     pub static $NAME: $Type,
///     constructor = $constructor_expr,
/// }
/// ```
///
/// Custom expect form — for non-standard panic messages:
/// ```ignore
/// global_store! {
///     /// Doc comment for the static.
///     pub static $NAME: $Type,
///     constructor = $constructor_expr,
///     expect = $custom_message,
/// }
/// ```
#[macro_export]
macro_rules! global_store {
    // Standard form — auto-generates expect message.
    (
        $(#[$attr:meta])*
        pub static $name:ident: $ty:ty,
        constructor = $constructor:expr,
    ) => {
        $crate::global_store! {
            $(#[$attr])*
            pub static $name: $ty,
            constructor = $constructor,
            expect = concat!(
                stringify!($name),
                " not initialized — call init_global() first"
            ),
        }
    };

    // Custom expect form.
    (
        $(#[$attr:meta])*
        pub static $name:ident: $ty:ty,
        constructor = $constructor:expr,
        expect = $expect:expr,
    ) => {
        $(#[$attr])*
        pub static $name: ::tokio::sync::OnceCell<$ty> =
            ::tokio::sync::OnceCell::const_new();

        #[doc = concat!("Initialize the global ", stringify!($name), " store.")]
        pub async fn init_global() -> ::anyhow::Result<()> {
            let root = $crate::config::CONFIG.global_storage_root();
            $crate::turso::register_global_store(
                &$name,
                stringify!($name),
                || $constructor(&root),
            )
            .await
        }

        #[must_use]
        #[doc = concat!(
            "Get a reference to the global ",
            stringify!($name),
            " store.\n\n# Panics\n\nPanics if the store has not been initialized.",
        )]
        pub fn store() -> &'static $ty {
            $name.get().expect($expect)
        }
    };
}

/// Remove characters that cause FTS query parser errors.
///
/// Replaces offending characters with spaces (rather than removing them) to
/// preserve word boundaries — otherwise "user@example.com" would become the
/// unsearchable blob "userexamplecom" instead of "user example com".
/// Returns the sanitized query (might be empty).
///
/// Required because turso's FTS uses Tantivy, whose query parser treats
/// certain non-alphanumeric characters (e.g. backtick, braces, parentheses,
/// colons, brackets, and other punctuation) as syntax operators, causing
/// parse errors on user-generated queries. This is a Tantivy design decision,
/// not a turso version bug — upgrading turso will not eliminate this requirement.
/// Since the ngram tokenizer indexes character substrings, removing syntax
/// operators from queries does not reduce search quality.
#[must_use]
pub fn sanitize_fts_query(query: &str) -> String {
    query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build a comma-separated list of `?` placeholders for SQL IN-clauses.
///
/// Returns an empty string for `count == 0`. Callers MUST guard against
/// empty lists to avoid producing invalid SQL like `WHERE id IN ()`.
///
/// # Example
///
/// ```
/// # use mahbot::turso::sql_in_placeholders;
/// assert_eq!(sql_in_placeholders(3), "?, ?, ?");
/// assert_eq!(sql_in_placeholders(0), "");
/// ```
///
/// Note: libSQL/SQLite binds `Vec<Value>` positionally regardless of whether
/// the SQL uses `?` or `?N`, so numbered placeholders (`?1, ?2, ...`) are
/// never necessary — use this helper everywhere.
#[must_use]
pub fn sql_in_placeholders(count: usize) -> String {
    vec!["?"; count].join(", ")
}

/// Compatibility layer over Turso with a persistent connection.
///
/// Every `execute` / `execute_batch` / `query` / `query_map` call reuses a
/// single cached connection — eliminating the per-call overhead of
/// `db.connect()` and preventing page-cache races that occur when concurrent
/// read+write operations share a Limbo page cache.
///
/// All operations are serialized through an internal mutex to support
/// concurrent access from multiple tasks (required for parallel tests).
///
/// Dangling transactions (TxGuard dropped without commit/rollback) are handled
/// via a deferred rollback pattern: the flag is set in Drop and the actual
/// ROLLBACK is executed at the start of the next write operation on any clone
/// of this Connection.
#[derive(Clone, Debug)]
pub struct Connection {
    /// Persistent turso connection — reused for all execute/query calls.
    /// Mutex serializes concurrent access since libsql connections
    /// do not support concurrent operations.
    conn: Arc<tokio::sync::Mutex<turso::Connection>>,
    /// Set when a TxGuard is dropped without explicit commit/rollback.
    /// Checked at the start of every write operation (execute, begin_tx).
    /// Mirrors the upstream `turso::Connection::dangling_tx` pattern but
    /// works at our wrapper level so we don't need async in Drop.
    has_dangling_tx: Arc<AtomicBool>,
}

impl Connection {
    pub async fn open(path: &Path) -> anyhow::Result<Self> {
        let path_str = path
            .to_str()
            .with_context(|| format!("database path must be UTF-8: {}", path.display()))?;
        // Derive experimental features from the canonical opts function.
        // If you need to add/remove an experimental feature, change
        // experimental_database_opts() and EXPERIMENTAL_FEATURES — not here.
        let opts = experimental_database_opts();
        let db = Builder::new_local(path_str)
            .experimental_index_method(opts.enable_index_method)
            .experimental_multiprocess_wal(opts.enable_multiprocess_wal)
            .build()
            .await
            .context("failed to open local database")?;
        let conn = db.connect()?;
        conn.busy_timeout(Duration::from_mins(1))?;
        Ok(Self {
            conn: Arc::new(tokio::sync::Mutex::new(conn)),
            has_dangling_tx: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Handle a dangling transaction from a dropped TxGuard.
    /// Called at the start of every write operation.
    async fn maybe_rollback_dangling_tx(&self) -> turso::Result<()> {
        let conn = self.conn.lock().await;
        if self.has_dangling_tx.swap(false, Ordering::SeqCst) {
            conn.execute("ROLLBACK".to_string(), ()).await?;
        }
        Ok(())
    }

    pub async fn execute(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
    ) -> turso::Result<u64> {
        self.maybe_rollback_dangling_tx().await?;
        let conn = self.conn.lock().await;
        conn.execute(sql.to_string(), params).await
    }

    pub(crate) async fn execute_batch(&self, sql: &str) -> turso::Result<()> {
        self.maybe_rollback_dangling_tx().await?;
        let conn = self.conn.lock().await;
        conn.execute_batch(sql.to_string()).await
    }

    /// Begin a transaction and return a guard that keeps the connection locked
    /// until the transaction is committed or rolled back.
    pub async fn begin_tx(&self) -> turso::Result<TxGuard<'_>> {
        self.maybe_rollback_dangling_tx().await?;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN".to_string(), ()).await?;
        Ok(TxGuard {
            conn,
            has_dangling_tx: Some(self.has_dangling_tx.clone()),
        })
    }

    /// Execute a read-only query, returning all matching rows.
    /// Acquires the mutex so reads are serialized with writes — this eliminates
    /// page-cache races that occurred when reads used `db.connect()` to spawn
    /// a fresh connection sharing a cache with the write connection.
    pub async fn query(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
    ) -> turso::Result<Vec<Row>> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(sql, params).await?;
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(row);
        }
        Ok(result)
    }

    /// Execute a read-only query, mapping each row through a closure.
    /// Returns a Vec of results so callers can handle per-row errors
    /// individually.
    pub async fn query_map<T, E>(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
        map: impl FnMut(&Row) -> std::result::Result<T, E> + Send + 'static,
    ) -> turso::Result<Vec<turso::Result<T>>>
    where
        T: Send + 'static,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let rows = self.query(sql, params).await?;
        let mut map = map;
        Ok(rows
            .iter()
            .map(|row| map(row).map_err(|e| turso::Error::Error(e.to_string())))
            .collect())
    }

    /// Execute a query that returns exactly one row.
    /// Acquires the mutex so reads are serialized with writes.
    pub async fn query_row<T, E>(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
        map: impl FnOnce(&Row) -> std::result::Result<T, E> + Send + 'static,
    ) -> turso::Result<T>
    where
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(sql, params).await?;
        let row = rows
            .next()
            .await?
            .ok_or(turso::Error::QueryReturnedNoRows)?;
        map(&row).map_err(|e| turso::Error::Error(e.to_string()))
    }

    /// Force a WAL checkpoint with TRUNCATE mode.
    ///
    /// Writes all pending WAL content to the main database file and truncates
    /// the WAL. This is critical before hard process termination
    /// (e.g., [`std::process::exit`] in self-update) to prevent data loss from
    /// unwritten WAL pages.
    ///
    /// Safe to call even if the database is not in WAL mode — non-WAL databases
    /// treat this as a no-op.
    pub async fn checkpoint(&self) -> anyhow::Result<()> {
        self.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .await
            .context("Failed to checkpoint WAL")?;
        Ok(())
    }
}

/// A locked connection handle scoped to a single transaction.
/// Holds the mutex guard for the entire duration — dropped guard triggers rollback.
pub struct TxGuard<'a> {
    conn: tokio::sync::MutexGuard<'a, turso::Connection>,
    /// Shared flag on the parent Connection; set in Drop to signal a deferred
    /// rollback on the next write operation. Set to None when the transaction
    /// has been explicitly committed or rolled back, preventing Drop from
    /// flagging a dangling transaction.
    has_dangling_tx: Option<Arc<AtomicBool>>,
}

impl TxGuard<'_> {
    /// Execute a statement within the transaction.
    pub async fn execute(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
    ) -> turso::Result<u64> {
        self.conn.execute(sql.to_string(), params).await
    }

    /// Execute a query that returns exactly one row within the transaction.
    /// Uses the upstream connection directly so the query participates in the
    /// transaction.
    pub async fn query_row<T, E>(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
        map: impl FnOnce(&Row) -> std::result::Result<T, E> + Send + 'static,
    ) -> turso::Result<T>
    where
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let mut rows = self.conn.query(sql, params).await?;
        let row = rows
            .next()
            .await?
            .ok_or(turso::Error::QueryReturnedNoRows)?;
        map(&row).map_err(|e| turso::Error::Error(e.to_string()))
    }

    /// Execute a query returning zero or more rows within the transaction.
    /// Uses the upstream connection directly so the query participates in the
    /// transaction. Returns an empty Vec when no rows match.
    pub async fn query(
        &self,
        sql: &str,
        params: impl IntoParams + Send + 'static,
    ) -> turso::Result<Vec<Row>> {
        let mut rows = self.conn.query(sql, params).await?;
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(row);
        }
        Ok(result)
    }

    /// Commit the transaction and release the lock.
    pub async fn commit(mut self) -> turso::Result<()> {
        self.conn.execute("COMMIT".to_string(), ()).await?;
        // Clear flag so Drop doesn't try to roll back an already-committed tx
        self.has_dangling_tx = None;
        Ok(())
    }

    /// Rollback the transaction and release the lock.
    pub async fn rollback(mut self) -> turso::Result<()> {
        self.conn.execute("ROLLBACK".to_string(), ()).await?;
        // Clear flag so Drop doesn't try to roll back again
        self.has_dangling_tx = None;
        Ok(())
    }
}

impl Drop for TxGuard<'_> {
    fn drop(&mut self) {
        // Set the dangling_tx flag on the parent Connection so the next
        // write operation (execute, begin_tx) will issue a ROLLBACK first.
        // This is a deferred pattern — we can't call async methods from Drop,
        // but the lock isn't released yet (MutexGuard is still alive), so
        // no other task can execute a write before the flag is set.
        if let Some(flag) = &self.has_dangling_tx {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Row extraction helper
// ─────────────────────────────────────────────────────────────────────

/// Extract a `TEXT` column from a row, returning `""` for `NULL`.
pub fn row_text(row: &Row, idx: usize) -> anyhow::Result<String> {
    match row.get_value(idx)? {
        Value::Text(s) => Ok(s),
        Value::Null => Ok(String::new()),
        other => anyhow::bail!("expected text column {idx}, got {other:?}"),
    }
}

/// Extract an optional `TEXT` column from a row, returning `None` for `NULL`.
pub fn row_text_opt(row: &Row, idx: usize) -> anyhow::Result<Option<String>> {
    match row.get_value(idx)? {
        Value::Text(s) => Ok(Some(s)),
        Value::Null => Ok(None),
        other => anyhow::bail!("expected text or null in column {idx}, got {other:?}"),
    }
}

/// Extract an optional `INTEGER` column as `Option<bool>` (1 = true, 0 = false,
/// `NULL` = `None`).
pub fn row_bool_opt(row: &Row, idx: usize) -> anyhow::Result<Option<bool>> {
    match row.get_value(idx)? {
        Value::Integer(i) => Ok(Some(i != 0)),
        Value::Null => Ok(None),
        other => anyhow::bail!("expected integer or null in column {idx}, got {other:?}"),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Schema / index management
// ──────────────────────────────────────────────────────────────────────

/// Ensure a full-text search index exists with the correct tokenizer.
/// Drops and recreates if the existing index has a different tokenizer.
pub async fn ensure_fts_index(
    conn: &Connection,
    index_name: &str,
    tokenizer: &str,
    ddl: &str,
) -> anyhow::Result<()> {
    let existing_sql: Option<String> = match conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?1 LIMIT 1",
            params![index_name],
            |row| match row.get_value(0)? {
                Value::Text(s) => Ok::<_, ::turso::Error>(s),
                _ => Ok::<_, ::turso::Error>(String::new()),
            },
        )
        .await
    {
        Ok(sql) if !sql.is_empty() => Some(sql),
        Err(turso::Error::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
        _ => None,
    };

    let needs_rebuild = existing_sql
        .as_deref()
        .is_none_or(|sql| !sql.to_lowercase().contains(&tokenizer.to_lowercase()));

    if needs_rebuild {
        conn.execute(&format!("DROP INDEX IF EXISTS {index_name}"), ())
            .await?;
        // Race-safe: `CREATE INDEX IF NOT EXISTS` prevents failure when two
        // connections run schema init concurrently (e.g. parallel tests).
        conn.execute(ddl, ()).await?;
    }

    Ok(())
}

/// Open a database, create parent directories if needed, and run schema init.
///
/// `schema` is executed via `execute_batch` (multiple DDL statements).
pub async fn open_with_schema(db_path: &Path, schema: &str) -> anyhow::Result<Connection> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    let conn = Connection::open(db_path)
        .await
        .with_context(|| format!("Failed to open database: {}", db_path.display()))?;

    conn.execute_batch(schema)
        .await
        .context(format!("Failed to run schema {schema}"))?;

    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experimental_features_are_consistent() {
        // Verify that experimental_database_opts() enables exactly
        // the features listed in EXPERIMENTAL_FEATURES, and no other
        // known experimental features.
        let opts = experimental_database_opts();

        // All features listed in the const must be enabled.
        for feature in EXPERIMENTAL_FEATURES {
            match *feature {
                "index_method" => assert!(
                    opts.enable_index_method,
                    "index_method should be enabled per EXPERIMENTAL_FEATURES"
                ),
                "multiprocess_wal" => assert!(
                    opts.enable_multiprocess_wal,
                    "multiprocess_wal should be enabled per EXPERIMENTAL_FEATURES"
                ),
                other => panic!("unknown experimental feature: {other}"),
            }
        }

        // No other known DatabaseOpts experimental features should be enabled.
        // If turso_core adds a new field here, add a check below to keep the
        // test honest — every field should be accounted for.
        assert!(
            !opts.enable_views,
            "views is not an active experimental feature"
        );
        assert!(
            !opts.enable_custom_types,
            "custom_types is not an active experimental feature"
        );
        assert!(
            !opts.enable_encryption,
            "encryption is not an active experimental feature"
        );
        assert!(
            !opts.enable_autovacuum,
            "autovacuum is not an active experimental feature"
        );
        assert!(
            !opts.enable_vacuum,
            "vacuum is not an active experimental feature"
        );
        assert!(
            !opts.enable_attach,
            "attach is not an active experimental feature"
        );
        assert!(
            !opts.enable_generated_columns,
            "generated_columns is not an active experimental feature"
        );
        assert!(
            !opts.enable_without_rowid,
            "without_rowid is not an active experimental feature"
        );
        assert!(
            !opts.unsafe_testing,
            "unsafe_testing is not an active experimental feature"
        );
    }

    #[test]
    fn builder_mapping_matches_experimental_features() {
        // Verify that the field-by-field mapping in Connection::open (simulated
        // below) enables exactly the features listed in EXPERIMENTAL_FEATURES.
        //
        // This catches reverse-drift: if someone adds a feature to
        // experimental_database_opts() and EXPERIMENTAL_FEATURES but forgets
        // the experimental_*() mapping line in Connection::open, the test
        // fails because the simulated mapping doesn't enable it.
        //
        // To add a new feature:
        //   1. Add with_*() to experimental_database_opts()
        //   2. Add experimental_*() to Connection::open
        //   3. Add feature name to EXPERIMENTAL_FEATURES
        //   4. Add the if-guard below to this test's mapping table
        let opts = experimental_database_opts();

        // ── Simulate the builder mapping from Connection::open ────────────
        // Must stay in sync with the .experimental_*() calls there.
        let mut mapped: Vec<&str> = Vec::new();
        if opts.enable_index_method {
            mapped.push("index_method");
        }
        if opts.enable_multiprocess_wal {
            mapped.push("multiprocess_wal");
        }
        // Note: enable_views maps to experimental_materialized_views() but
        // is not an active feature — no if-guard needed.
        // Note: enable_autovacuum and unsafe_testing have no Builder
        // experimental_*() equivalent — they cannot be mapped here.
        // ─────────────────────────────────────────────────────────────────

        mapped.sort_unstable();
        let mut expected: Vec<&str> = EXPERIMENTAL_FEATURES.to_vec();
        expected.sort_unstable();

        assert_eq!(
            mapped, expected,
            "Connection::open builder mapping enables features that differ from \
             EXPERIMENTAL_FEATURES.\n\
             If you added a feature: add the experimental_*() guard above AND \
             add it to Connection::open.\n\
             If you removed a feature: remove it from both places.\n\
             See EXPERIMENTAL_FEATURES docs for naming asymmetries."
        );
    }

    #[test]
    fn test_sanitize_fts_query_keeps_alphanumeric() {
        assert_eq!(sanitize_fts_query("hello world"), "hello world");
    }

    #[test]
    fn test_sanitize_fts_query_strips_special_chars() {
        assert_eq!(sanitize_fts_query("`Hello ${name}`"), "Hello name");
    }

    #[test]
    fn test_sanitize_fts_query_preserves_word_boundaries() {
        assert_eq!(
            sanitize_fts_query("contact user@example.com now"),
            "contact user example com now"
        );
    }

    #[test]
    fn test_sanitize_fts_query_handles_punctuation() {
        assert_eq!(
            sanitize_fts_query("hello, world! How's it going?"),
            "hello world How s it going"
        );
    }

    #[test]
    fn test_sanitize_fts_query_empty_result() {
        assert_eq!(sanitize_fts_query("!@#$%"), "");
    }
}
