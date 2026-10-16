//! The connection layer of every store: the product's own use of the engine's
//! SDK kit.
//!
//! One statement may hold a store for at most [`statement_limit`]. The engine
//! (turso_core) checks that limit as the statement steps and terminates it once
//! the limit is reached, so the store answers the very next operation without a
//! restart and the caller learns the statement was too heavy. The limit lives on
//! the connection (`TursoConnection::set_query_timeout`) and the `turso` facade
//! exposes neither it nor the connection it wraps, so the connections this
//! product opens are created here, from the published `turso_sdk_kit` the
//! facade itself is built on — pinned to the same version, the same engine.
//!
//! Statement execution mirrors the facade: prepare, bind, step, and an IO step
//! runs the engine's IO before the next step, which then yields to the runtime.
//! What is not mirrored is the facade's statement/transaction plumbing the
//! product does not use.

use std::sync::Arc;
use std::time::Duration;

use turso::core::OpenFlags;
use turso::params::Params;
use turso::{IntoParams, Value};
use turso_sdk_kit::rsapi::{
    FromValue, TursoConnection, TursoDatabase, TursoDatabaseConfig, TursoError, TursoStatement,
    TursoStatusCode, Value as CoreValue,
};

use crate::util::UnwrapPoison;

/// Default limit on how long one statement may hold a store, in seconds.
const STATEMENT_LIMIT_DEFAULT_SECS: u64 = 180;

/// Env override for [`STATEMENT_LIMIT_DEFAULT_SECS`], in seconds — the same
/// env-override pattern the other bounded waits use, and what lets the limit's
/// behaviour be exercised without waiting out the default. The limit it yields
/// ([`statement_limit`]) is what every connection the product opens is armed
/// with, the inspection clients' own direct read-only one included
/// ([`crate::db::debug::connect_readonly`]); those clients carry their own
/// command-wide read limit on top of it ([`crate::db::read_limit`]), which is a
/// separate knob.
pub(crate) const STATEMENT_LIMIT_ENV: &str = "MAHBOT_STORE_STATEMENT_LIMIT_SECS";

/// The longest one statement may hold a store.
///
/// The engine reads a zero duration as "no limit", which would disarm the guard
/// this module exists to arm, so the override never drops below a second.
#[must_use]
pub(crate) fn statement_limit() -> Duration {
    crate::util::env_duration_secs(STATEMENT_LIMIT_ENV, STATEMENT_LIMIT_DEFAULT_SECS)
        .max(Duration::from_secs(1))
}

/// What a caller is told about a statement the limit terminated.
///
/// The engine's own wording for this outcome is "interrupted", which says
/// nothing about why; every path out of the product — the debug CLI through the
/// IPC answer, the in-process tool, the service's own log line — reads the same
/// sentence, and it tells the caller the query is too heavy to run here and
/// must be reformulated rather than sent again.
#[must_use]
pub(crate) fn too_heavy_message() -> String {
    format!(
        "the statement is too heavy to run: it held the store for longer than its {}s \
         per-statement limit and was terminated — reformulate it (narrow the scan, add a filter, \
         use an index or add a LIMIT) instead of sending the same query again",
        statement_limit().as_secs()
    )
}

/// Map an engine error to the error a caller sees.
///
/// An interrupt is always the limit terminating the statement: nothing else
/// interrupts this product's connections, and the engine's own wording for it
/// ("interrupted") does not say what happened. Every other error passes through
/// unchanged, keeping its variant and its message.
fn map_error(err: TursoError) -> turso::Error {
    match err {
        TursoError::Interrupt(_) => turso::Error::Interrupt(too_heavy_message()),
        other => other.into(),
    }
}

/// The string form of [`crate::db::EXPERIMENTAL_FEATURES`], the single source for
/// which features are active.
fn experimental_features() -> Option<String> {
    (!crate::db::EXPERIMENTAL_FEATURES.is_empty())
        .then(|| crate::db::EXPERIMENTAL_FEATURES.join(","))
}

/// The engine config a store is opened with — the facade's `Builder::build`
/// settings, in the one place that builds them (the test asserts the features
/// they carry against the read-only path's typed opts).
fn database_config(path: &str) -> TursoDatabaseConfig {
    TursoDatabaseConfig {
        path: path.to_owned(),
        experimental_features: experimental_features(),
        async_io: true,
        encryption: None,
        vfs: turso::IoBackend::Default,
        io: None,
        db_file: None,
        page_codec: None,
        open_flags: OpenFlags::default(),
    }
}

/// The engine handle is not `Debug` and the facade's connection printed nothing
/// but its type name; a store connection sits inside structs that derive `Debug`
/// (the store structs and the persistence module's own `Connection`), so it
/// renders that same name and leaves their log lines unchanged.
impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection").finish()
    }
}

/// An opened store: the handle every connection of that store is made from.
pub(crate) struct Database {
    inner: Arc<TursoDatabase>,
}

/// The engine handle is not `Debug`; see the `Connection` impl above.
impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database").finish()
    }
}

impl Database {
    /// Open the store at `path` for read-write use.
    ///
    /// `TursoDatabase` hands back the IO work of the open, so the open is
    /// finished by driving it here — the same steps the facade's `Builder::build`
    /// performs, with the same engine settings.
    pub(crate) async fn open(path: &str) -> turso::Result<Self> {
        let db = TursoDatabase::new(database_config(path));
        loop {
            let opened = db.open().map_err(map_error)?;
            let Some(completion) = opened.io() else {
                break;
            };
            let io = db.io().map_err(map_error)?;
            completion
                .wait_async(io.as_ref())
                .await
                .map_err(TursoError::from)
                .map_err(map_error)?;
        }
        Ok(Self { inner: db })
    }

    /// Connect to the store, with [`statement_limit`] armed on the connection.
    pub(crate) fn connect(&self) -> turso::Result<Connection> {
        self.connect_with(statement_limit())
    }

    /// [`Self::connect`] with an explicit limit, so a test can exercise the
    /// limit without waiting out the default.
    fn connect_with(&self, limit: Duration) -> turso::Result<Connection> {
        let conn = self.inner.connect().map_err(map_error)?;
        conn.set_query_timeout(limit);
        Ok(Connection { inner: conn })
    }
}

/// One connection of a store, with the per-statement limit armed on it.
pub(crate) struct Connection {
    inner: Arc<TursoConnection>,
}

impl Connection {
    /// Prepare and bind `sql`, returning the rows it will produce.
    ///
    /// Preparing is all this call does; the rows come out of [`Rows::next`] as
    /// it steps the statement.
    pub(crate) fn query(&self, sql: &str, params: impl IntoParams) -> turso::Result<Rows> {
        self.prepare(sql)?.query(params)
    }

    /// Execute a statement, returning how many rows it changed.
    ///
    /// The statement must not hand rows back; a change that reads its own
    /// result back is driven through [`Self::query`] instead.
    pub(crate) async fn execute(&self, sql: &str, params: impl IntoParams) -> turso::Result<u64> {
        let mut stmt = self.prepare(sql)?;
        stmt.execute(params).await
    }

    /// Execute every statement in `sql` in order, stopping at the first
    /// failure; each statement runs in autocommit, so the ones before a failure
    /// stay committed.
    pub(crate) async fn execute_batch(&self, sql: &str) -> turso::Result<()> {
        let mut sql = sql;
        while let Some((stmt, offset)) = self.inner.prepare_first(sql).map_err(map_error)? {
            let mut stmt = statement(stmt);
            stmt.execute(()).await?;
            sql = &sql[offset..];
        }
        Ok(())
    }

    /// Prepare `sql` through the connection's own statement cache — for a
    /// statement run in a loop, where re-parsing the text on every call would
    /// be the cost.
    pub(crate) fn prepare_cached(&self, sql: &str) -> turso::Result<Statement> {
        let inner = self.inner.prepare_cached(sql).map_err(map_error)?;
        Ok(statement(inner))
    }

    /// Set how long a lock wait may take before it is reported as busy.
    pub(crate) fn busy_timeout(&self, duration: Duration) {
        self.inner.set_busy_timeout(duration);
    }

    /// Whether the connection is between transactions.
    #[must_use]
    pub(crate) fn is_autocommit(&self) -> bool {
        self.inner.get_auto_commit()
    }

    fn prepare(&self, sql: &str) -> turso::Result<Statement> {
        let inner = self.inner.prepare_single(sql).map_err(map_error)?;
        Ok(statement(inner))
    }
}

fn statement(inner: Box<TursoStatement>) -> Statement {
    Statement {
        inner: Arc::new(std::sync::Mutex::new(inner)),
    }
}

/// One prepared statement, shared with the [`Rows`] a query hands back.
///
/// The engine's own statement holds the connection it was prepared on, so this
/// needs no reference to it.
#[derive(Clone)]
pub(crate) struct Statement {
    inner: Arc<std::sync::Mutex<Box<TursoStatement>>>,
}

impl Statement {
    /// Bind `params` and hand back the statement's rows.
    pub(crate) fn query(&mut self, params: impl IntoParams) -> turso::Result<Rows> {
        self.bind(params)?;
        let columns = self.column_count();
        Ok(Rows {
            stmt: self.clone(),
            columns,
        })
    }

    /// Execute this statement, returning how many rows it changed.
    pub(crate) async fn execute(&mut self, params: impl IntoParams) -> turso::Result<u64> {
        self.bind(params)?;
        match self.step().await? {
            TursoStatusCode::Done => Ok(self.change_count()),
            TursoStatusCode::Row => Err(turso::Error::Misuse(
                "unexpected row during execution".to_string(),
            )),
            TursoStatusCode::Io => unreachable!("step() runs the engine's IO"),
        }
    }

    /// Step the statement until it hands back a row or finishes.
    ///
    /// The engine reports [`TursoStatusCode::Io`] both for a step that needs IO
    /// and for its own cooperative yield, so the IO is run and control returns to
    /// the runtime before the next step: a long statement's steps never spin on
    /// one worker. The connection opens the blocking default IO backend, so the
    /// IO step completes inline and no waker is involved.
    async fn step(&self) -> turso::Result<TursoStatusCode> {
        loop {
            let status = self
                .inner
                .lock()
                .unwrap_poison()
                .step(None)
                .map_err(map_error)?;
            if status != TursoStatusCode::Io {
                return Ok(status);
            }
            self.inner
                .lock()
                .unwrap_poison()
                .run_io()
                .map_err(map_error)?;
            tokio::task::yield_now().await;
        }
    }

    /// Reset the statement and bind `params` to it.
    fn bind(&mut self, params: impl IntoParams) -> turso::Result<()> {
        let params = params.into_params()?;
        let mut inner = self.inner.lock().unwrap_poison();
        inner.reset().map_err(map_error)?;
        match params {
            Params::None => {}
            Params::Positional(values) => {
                for (index, value) in values.into_iter().enumerate() {
                    inner
                        .bind_positional(index + 1, value.into())
                        .map_err(map_error)?;
                }
            }
            Params::Named(values) => {
                for (name, value) in values {
                    let position = inner.named_position(name).map_err(map_error)?;
                    inner
                        .bind_positional(position, value.into())
                        .map_err(map_error)?;
                }
            }
        }
        Ok(())
    }

    fn change_count(&self) -> u64 {
        let inner = self.inner.lock().unwrap_poison();
        inner.n_change().try_into().unwrap_or(0)
    }

    fn column_count(&self) -> usize {
        self.inner.lock().unwrap_poison().column_count()
    }

    fn column_names(&self) -> Vec<String> {
        let inner = self.inner.lock().unwrap_poison();
        (0..inner.column_count())
            .map(|index| inner.column_name(index).unwrap_or_default())
            .collect()
    }

    fn row(&self, columns: usize) -> turso::Result<Row> {
        let inner = self.inner.lock().unwrap_poison();
        let mut values = Vec::with_capacity(columns);
        for index in 0..columns {
            values.push(inner.row_value(index).map_err(map_error)?);
        }
        Ok(Row { values })
    }
}

/// The rows of one query.
pub(crate) struct Rows {
    stmt: Statement,
    columns: usize,
}

impl Rows {
    /// The result set's column names.
    #[must_use]
    pub(crate) fn column_names(&self) -> Vec<String> {
        self.stmt.column_names()
    }

    /// Fetch the next row, or `None` once the statement has finished.
    pub(crate) async fn next(&mut self) -> turso::Result<Option<Row>> {
        match self.stmt.step().await? {
            TursoStatusCode::Done => Ok(None),
            TursoStatusCode::Row => Ok(Some(self.stmt.row(self.columns)?)),
            TursoStatusCode::Io => unreachable!("step() runs the engine's IO"),
        }
    }
}

/// One row of a query's result.
#[derive(Debug)]
pub(crate) struct Row {
    values: Vec<CoreValue>,
}

impl Row {
    /// The value at `index`, as the caller-facing [`Value`].
    pub(crate) fn get_value(&self, index: usize) -> turso::Result<Value> {
        Ok(self.value(index)?.clone().into())
    }

    /// The value at `index`, converted to `T`.
    pub(crate) fn get<T: FromValue>(&self, index: usize) -> turso::Result<T> {
        T::from_sql(self.value(index)?.clone())
            .map_err(|err| turso::Error::ConversionFailure(err.to_string()))
    }

    /// The number of columns in this row.
    #[must_use]
    pub(crate) fn column_count(&self) -> usize {
        self.values.len()
    }

    fn value(&self, index: usize) -> turso::Result<&CoreValue> {
        self.values.get(index).ok_or_else(|| {
            turso::Error::Misuse(format!(
                "column index {index} out of bounds (row has {} columns)",
                self.values.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The feature string this layer opens the engine with must produce exactly
    /// the typed opts the read-only debug path opens with. `database_opts()`
    /// silently ignores a name it does not know, so this also catches a typo in
    /// [`crate::db::EXPERIMENTAL_FEATURES`].
    #[test]
    fn feature_string_matches_the_debug_paths_typed_opts() {
        assert_eq!(
            database_config(":memory:").database_opts(),
            crate::db::experimental_database_opts()
        );
    }

    /// A statement that overruns the limit is terminated, the caller is told the
    /// query is too heavy, and the store answers the very next operation on the
    /// same connection; a cut write rolls its statement back instead of leaving
    /// the store mid-write. This is the whole point of arming the limit: without
    /// it a heavy statement holds the store for as long as it runs.
    #[tokio::test]
    async fn a_statement_past_the_limit_is_cut_and_the_store_stays_usable() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("cut.db");
        let db = Database::open(path.to_str().expect("utf-8 path"))
            .await
            .expect("open store");
        let conn = db
            .connect_with(Duration::from_secs(1))
            .expect("connect with a one-second limit");

        conn.execute("CREATE TABLE big (x INTEGER)", ())
            .await
            .expect("create table");
        conn.execute("INSERT INTO big (x) VALUES (1)", ())
            .await
            .expect("seed a row");

        // A recursive count to a hundred million is a statement the engine
        // cannot finish inside a second, and nothing predicts it beforehand —
        // the limit cuts it as it runs. Rows are stepped by `next`, so that is
        // where the cut surfaces.
        let mut cut = conn
            .query(&format!("{HEAVY}SELECT COUNT(*) FROM c"), ())
            .expect("prepare the heavy statement");
        let err = cut
            .next()
            .await
            .expect_err("the statement must be cut at the limit");
        assert!(
            matches!(err, turso::Error::Interrupt(_)),
            "a cut statement must be reported as an interrupt: {err:?}"
        );
        assert!(
            err.to_string().contains("too heavy"),
            "the caller must be told the query is too heavy: {err}"
        );

        let mut rows = conn
            .query("SELECT COUNT(*) FROM big", ())
            .expect("the store answers the next operation");
        let row = rows.next().await.expect("row").expect("one aggregate row");
        assert_eq!(row.get::<i64>(0).expect("count"), 1);
        // …and it is not left read-only: a cut statement leaves neither the
        // connection nor the store unable to change.
        conn.execute("INSERT INTO big (x) VALUES (2)", ())
            .await
            .expect("the connection still writes after a cut");

        // A write is cut the same way, and its statement is rolled back: the
        // table keeps what it had and the connection is not left mid-write.
        let err = conn
            .execute(&format!("{HEAVY}INSERT INTO big (x) SELECT i FROM c"), ())
            .await
            .expect_err("the write must be cut at the limit");
        assert!(
            err.to_string().contains("too heavy"),
            "a cut write must be told the same thing: {err}"
        );
        let mut rows = conn
            .query("SELECT COUNT(*) FROM big", ())
            .expect("the store answers after a cut write");
        let row = rows.next().await.expect("row").expect("one aggregate row");
        assert_eq!(row.get::<i64>(0).expect("count"), 2);
    }

    /// What every heavy statement here is built on: a recursive series to a
    /// hundred million, far beyond a second's work, whose tail decides whether
    /// the statement reads or writes.
    const HEAVY: &str =
        "WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i < 100000000) ";
}
