//! Shared test utilities for initializing global stores and building test
//! tickets.
//!
//! Provides a single temporary directory shared across all test initializations,
//! eliminating the duplicated `init_with_temp_dir()` pattern that was previously
//! required in each module and leaked a separate temp dir per store.
//!
//! The temp directory is intentionally leaked for the process lifetime to avoid
//! races at shutdown — since global [`OnceCell`]s can only be set once, each
//! store is initialized at most once per test run.
//!
//! Also provides [`TicketBuilder`], a builder for creating test tickets that was
//! historically defined in the `board` module and imported from there by sibling
//! modules. Moved here so all test infrastructure lives in one place.
//!
//! Also provides [`create_test_workspace`] (inserting a workspace into the test DB)
//! and [`init_management_test_stores`] (initializing all stores plus the manager
//! queue), relocated from `management.rs` tests so they are discoverable alongside
//! the rest of the shared test infrastructure.

#![cfg(test)]

use crate::board::{BoardStore, Ticket, TicketParams, TicketPhase};
use crate::turso;
use crate::workspace::test_ws_named;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Shared test root directory, created once and intentionally leaked
/// for the process lifetime.
static TEST_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Mutex serializing env-var-modifying tests to prevent thread-safety
/// issues with `std::env::set_var` (which is `unsafe` in Rust 2024).
///
/// All test modules that modify environment variables should use this
/// shared lock to prevent data races between concurrent tests.
pub fn env_lock() -> &'static std::sync::Mutex<()> {
    static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// RAII guard that restores an environment variable to its original value
/// on drop, including during a panic (unwind safety).
///
/// Created by [`set_env_var`]. Holds the shared [`env_lock()`] for the
/// entire duration to serialize concurrent env access across tests.
///
/// # Panic safety
///
/// The `Drop` implementation restores the original value even if the
/// enclosing scope panics, preventing test-isolation leaks. This is the
/// key advantage over a closure-based `with_env_var` helper.
pub struct EnvVarGuard {
    /// Serializes concurrent env access — held for the guard's entire lifetime.
    _lock: std::sync::MutexGuard<'static, ()>,
    key: String,
    /// Original value captured before mutation. Stored as [`OsString`] to
    /// preserve arbitrary (non-UTF-8) env var values on restore.
    original: Option<std::ffi::OsString>,
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: We hold the env_lock, prohibiting concurrent env writes
        // from other test threads while we restore the original value.
        unsafe {
            match &self.original {
                Some(val) => std::env::set_var(&self.key, val),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

/// Set an environment variable for the duration of the returned guard.
///
/// The environment variable `key` is immediately set to `value` (or
/// removed if `value` is `None`). When the returned [`EnvVarGuard`] is
/// dropped — including on panic — the original value is restored.
///
/// Acquires the shared [`env_lock()`] to prevent data races with other
/// tests that manipulate environment variables.
///
/// # Example
///
/// ```ignore
/// let _guard = set_env_var("CARGO_HOME", Some("/custom/cargo"));
/// let path = resolve_cargo_bin_path();
/// // _guard drops here, restoring CARGO_HOME to its original value
/// ```
#[must_use]
pub fn set_env_var(key: &str, value: Option<&str>) -> EnvVarGuard {
    let guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let original = std::env::var_os(key);
    // SAFETY: Protected by the env_lock acquired above — no concurrent
    // env writes can happen from other test threads.
    unsafe {
        match value {
            Some(val) => std::env::set_var(key, val),
            None => std::env::remove_var(key),
        }
    }
    EnvVarGuard {
        _lock: guard,
        key: key.to_owned(),
        original,
    }
}

fn test_root() -> &'static PathBuf {
    TEST_ROOT.get_or_init(|| {
        let tmp = tempfile::TempDir::new().expect("failed to create test temp dir");
        let path = tmp.path().to_path_buf();
        // Leak: avoids races on the temp directory at shutdown. The process
        // will clean it up on exit, and tests never re-initialize anyway.
        std::mem::forget(tmp);
        path
    })
}

/// Open a temporary store for testing.
///
/// Creates a temporary directory and opens the given store inside it.
/// Returns `(store, TempDir)`.  The `TempDir` MUST be held for the store's
/// lifetime (typically bound to `_tmp` / `_dir` in the calling test).
///
/// `store_name` is used in the panic message if opening the store fails,
/// so it should be a human-readable identifier (e.g. `"workspace"`, `"board"`).
///
/// This is a macro (not a generic function) because Rust's type system cannot
/// express the lifetime relationship between a closure argument and the future
/// returned by `async fn open(path: &Path) -> Result<T>` — the future captures
/// a borrow of the argument, which would require higher-ranked lifetime bounds
/// that `FnOnce` / `AsyncFnOnce` cannot express in the current edition without
/// boxing.
///
/// # Panics
///
/// Panics if the temporary directory cannot be created, or if opening the
/// store fails.
#[macro_export]
macro_rules! open_test_store {
    ($store:ty, $store_name:expr) => {{
        let tmp = ::tempfile::TempDir::new().expect("temp dir for test store");
        let store = <$store>::open(tmp.path()).await.unwrap_or_else(|e| {
            ::std::panic!(
                "failed to open test {store_name} store: {e:?}",
                store_name = $store_name
            )
        });
        (store, tmp)
    }};
}

/// Convenience helper to create a test ticket with just a title and phase.
///
/// Reduces the common boilerplate:
/// ```ignore
/// let id = TicketBuilder::new(&store, &ws)
///     .title("My Ticket")
///     .phase(TicketPhase::Backlog)
///     .create()
///     .await
///     .expect("create my ticket");
/// ```
/// to:
/// ```ignore
/// let id = make_ticket(&store, &ws, "My Ticket", TicketPhase::Backlog).await;
/// ```
///
/// For tickets that need `.desc()`, `.prereqs()`, `.reporter()`, `.embedding()`,
/// or `.supersede()`, use [`TicketBuilder`] directly.
///
/// # Panics
///
/// Panics if the ticket cannot be created. The panic message includes the title
/// and phase for debugging.
pub(crate) async fn make_ticket(
    store: &BoardStore,
    ws: &crate::Workspace,
    title: &str,
    phase: TicketPhase,
) -> String {
    TicketBuilder::new(store, ws)
        .title(title)
        .phase(phase)
        .create()
        .await
        .unwrap_or_else(|e| panic!("make_ticket({title}, {phase}) failed: {e}"))
}

/// Fetch a ticket by ID, panicking if the DB query fails or the ticket
/// does not exist.
///
/// Replaces the common test boilerplate:
/// ```ignore
/// let ticket = store.get_ticket(&id).await.expect("get").expect("exists");
/// ```
/// with the more concise:
/// ```ignore
/// let ticket = expect_ticket(&store, &id).await;
/// ```
///
/// # Panics
///
/// Panics if the DB query fails or the ticket is not found (returns `None`).
/// The panic originates from within this helper function.
pub async fn expect_ticket(store: &BoardStore, id: &str) -> Ticket {
    store
        .get_ticket(id)
        .await
        .expect("BoardStore::get_ticket query failed")
        .expect("expected ticket to exist")
}

/// Fetch a ticket's phase by ID, panicking if the DB query fails or the
/// ticket does not exist.
///
/// Replaces the common test boilerplate:
/// ```ignore
/// let phase = store.get_ticket_phase(&id).await.expect("query").expect("exists");
/// ```
/// with the more concise:
/// ```ignore
/// let phase = expect_ticket_phase(&store, &id).await;
/// ```
///
/// # Panics
///
/// Panics if the DB query fails or the ticket is not found (returns `None`).
/// The panic originates from within this helper function.
pub async fn expect_ticket_phase(store: &BoardStore, id: &str) -> TicketPhase {
    store
        .get_ticket_phase(id)
        .await
        .expect("BoardStore::get_ticket_phase query failed")
        .expect("expected ticket phase to exist")
}

/// Assert that a ticket has been superseded: its status is `Cancelled`,
/// it has no assignee, and it is archived immediately.
///
/// This checks the minimum invariants set atomically by
/// [`BoardStore::supersede_and_create`]. Individual tests MAY also assert
/// additional fields (e.g. `superseded_by`) as needed.
///
/// # Panics
///
/// Panics if any of the assertions fail.
pub fn assert_superseded_ticket(ticket: &Ticket) {
    assert_eq!(ticket.phase, TicketPhase::Cancelled);
    assert!(
        ticket.assigned_to.is_none(),
        "superseded ticket should have no assignee"
    );
    assert!(
        ticket.is_archived,
        "superseded ticket should be archived immediately"
    );
}

/// Builder for creating test tickets with common defaults.
///
/// Defaults: `desc="desc"`, `phase=Backlog`, `prerequisites=[]`, `reporter="test"`,
/// `embedding=None`. Title is required (no default) via `.title()`.
///
/// # Examples
/// ```ignore
/// // Simple ticket with defaults
/// TicketBuilder::new(&store, &ws).title("A").create().await?;
///
/// // Custom phase and prerequisites
/// TicketBuilder::new(&store, &ws)
///     .title("B")
///     .phase(TicketPhase::InDevelopment)
///     .prereqs(&[a_id, b_id])
///     .create().await?;
///
/// // Supersede an existing ticket
/// TicketBuilder::new(&store, &ws)
///     .title("New title")
///     .supersede(&old_id).await?;
///
/// // With embedding bytes
/// TicketBuilder::new(&store, &ws)
///     .title("Embedded")
///     .embedding(&blob)
///     .create().await?;
/// ```
pub(crate) struct TicketBuilder<'a> {
    store: &'a BoardStore,
    ws: crate::Workspace,
    title: String,
    desc: String,
    phase: TicketPhase,
    prereqs: Vec<String>,
    reporter: String,
    embedding: Option<Vec<u8>>,
}

impl<'a> TicketBuilder<'a> {
    /// Start building a test ticket for `store` in workspace `ws`.
    pub(crate) fn new(store: &'a BoardStore, ws: &crate::Workspace) -> Self {
        Self {
            store,
            ws: ws.clone(),
            title: String::new(),
            desc: "desc".into(),
            phase: TicketPhase::Backlog,
            prereqs: Vec::new(),
            reporter: "test".into(),
            embedding: None,
        }
    }

    /// Set the ticket title (required).
    pub(crate) fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the description (default: `"desc"`).
    pub(crate) fn desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = desc.into();
        self
    }

    /// Set the phase (default: [`TicketPhase::Backlog`]).
    pub(crate) fn phase(mut self, phase: TicketPhase) -> Self {
        self.phase = phase;
        self
    }

    /// Set prerequisites (default: empty).
    pub(crate) fn prereqs(mut self, prereqs: &[String]) -> Self {
        self.prereqs = prereqs.to_vec();
        self
    }

    /// Set the reporter (default: `"test"`).
    pub(crate) fn reporter(mut self, reporter: impl Into<String>) -> Self {
        self.reporter = reporter.into();
        self
    }

    /// Set embedding bytes (default: `None`).
    pub(crate) fn embedding(mut self, blob: &[u8]) -> Self {
        self.embedding = Some(blob.to_vec());
        self
    }

    /// Create the ticket with the accumulated parameters.
    pub(crate) async fn create(self) -> anyhow::Result<String> {
        let (store, params) = self.into_parts();
        store.create_ticket(&params).await
    }

    /// Supersede `supersede_id` with this ticket (calls `supersede_and_create`).
    pub(crate) async fn supersede(self, supersede_id: &str) -> anyhow::Result<String> {
        let (store, params) = self.into_parts();
        store.supersede_and_create(supersede_id, &params).await
    }

    fn into_parts(self) -> (&'a BoardStore, TicketParams) {
        (
            self.store,
            TicketParams {
                title: self.title,
                description: self.desc,
                workspace_name: self.ws.name,
                phase: self.phase,
                prerequisites: self.prereqs,
                reporter: self.reporter,
                embedding: self.embedding,
            },
        )
    }
}

/// Initialize all global test stores (session, board, workspace, users,
/// config, stats, chat_history) with a shared temp directory.
///
/// # Stores initialized
///
/// Note: The canonical list of all store names lives in
/// [`crate::turso::ALL_STORE_NAMES`].  This function intentionally excludes
/// `logs` (not needed for most tests) and initializes stores sequentially
/// (not concurrently like the production bootstrap).  Keep this list in sync
/// with `ALL_STORE_NAMES` when adding or removing stores.
///
/// Also initializes the search engine registry (required by workspace store)
/// and sets the CONFIG storage root.
///
/// Idempotent — subsequent calls are no-ops.
pub async fn init_test_stores() {
    use crate::chat_history::ChatHistoryStore;
    use crate::config_db::ConfigStore;
    use crate::session::SessionStore;
    use crate::stats::StatsStore;
    use crate::users::UserStore;
    use crate::workspace::WorkspaceStore;

    static INIT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    INIT.get_or_init(|| async {
        // Set CONFIG storage root (no-op if already set by another test)
        let _ = crate::config::CONFIG.try_set_storage_root(test_root().clone());

        // search_engine is sync — must be initialized before workspace
        crate::search_engine::init_global();

        // ticket_buffer is sync — lightweight allocation, no DB I/O.
        crate::ticket_buffer::init_global();

        macro_rules! init_test_store {
            ($cell:expr, $store:ty) => {
                $cell
                    .set(<$store>::open(test_root()).await.expect(concat!(
                        "failed to create ",
                        stringify!($store),
                        " for tests"
                    )))
                    .expect(concat!(
                        stringify!($cell),
                        " already initialized by another path"
                    ));
            };
        }

        init_test_store!(crate::session::SESSIONS, SessionStore);
        init_test_store!(crate::board::BOARD, BoardStore);
        init_test_store!(crate::workspace::WORKSPACES, WorkspaceStore);
        init_test_store!(crate::users::USER_STORE, UserStore);
        init_test_store!(crate::config_db::CONFIG_STORE, ConfigStore);
        init_test_store!(crate::stats::STATS_STORE, StatsStore);
        init_test_store!(crate::chat_history::CHAT_HISTORY, ChatHistoryStore);
    })
    .await;
}

/// Initialize all stores needed by management tests that interact with
/// the ticket buffer.
///
/// Calls [`init_test_stores`] (all test DBs) then initializes the global
/// manager queue. The manager queue consumer is required by callers that
/// exercise [`notify_ticket`](crate::management::notify_ticket) which
/// enqueues notifications via [`crate::manager_queue::manager_queue`].
///
/// # Panics
///
/// Panics if [`init_test_stores`] has not been called first (the manager
/// queue depends on stores being available), or if initialization of the
/// manager queue fails.
///
/// # Idempotency note
///
/// [`init_test_stores`] is idempotent (uses a [`tokio::sync::OnceCell`]).
/// [`crate::manager_queue::init_global`] spawns a consumer loop before
/// setting its [`OnceCell`](tokio::sync::OnceCell), so calling this
/// function more than once leaks a background task — ensure callers
/// initialize once per process lifetime.
pub async fn init_management_test_stores() {
    init_test_stores().await;

    let _ = crate::manager_queue::init_global();
}

/// Create a test workspace by inserting it into the test DB and returning
/// a [`Workspace`](crate::Workspace) struct with the given `path` and `name`.
///
/// Parameters are `(path, name)` to match the convention of
/// [`test_ws_named`](crate::workspace::test_ws_named).
///
/// # Precondition
///
/// [`init_test_stores`] must be called before this function — the
/// workspace store's [`OnceCell`](tokio::sync::OnceCell) panics if
/// accessed before initialization.
///
/// # Panics
///
/// Panics if the workspace store is not initialized, or if the INSERT
/// SQL query fails.
pub async fn create_test_workspace(path: &str, name: &str) -> crate::Workspace {
    let now = crate::turso::now();
    crate::workspace::store()
        .conn
        .execute(
            "INSERT INTO workspaces (name, path, created_at, updated_at, paused) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            turso::params![name, path, now.clone(), now, 0],
        )
        .await
        .expect("insert test workspace");
    test_ws_named(path, name)
}

/// Create a temporary directory initialized as a git repository with a
/// committed file named `test.txt` (containing `"line1\nline2\nline3\n"`)
/// and a single commit titled `"Initial commit"`.
///
/// The returned [`TempDir`](tempfile::TempDir) MUST be kept alive (bound to
/// `_dir` or similar) for the returned [`PathBuf`] to remain valid.
///
/// # Panics
///
/// Panics if `git` is not available, or if any git command fails.
pub(crate) fn init_temp_repo() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let repo_path = dir.path().to_path_buf();

    // git init
    let status = std::process::Command::new("git")
        .args(["init"])
        .current_dir(&repo_path)
        .status()
        .expect("git init");
    assert!(status.success());

    // Set user config (required for commit)
    for (key, value) in [("user.name", "Test"), ("user.email", "test@test.com")] {
        let status = std::process::Command::new("git")
            .args(["config", key, value])
            .current_dir(&repo_path)
            .status()
            .expect("git config");
        assert!(status.success());
    }

    // Create a file and make initial commit
    std::fs::write(repo_path.join("test.txt"), b"line1\nline2\nline3\n").expect("write test file");
    let status = std::process::Command::new("git")
        .args(["add", "-A"])
        .current_dir(&repo_path)
        .status()
        .expect("git add");
    assert!(status.success());
    let status = std::process::Command::new("git")
        .args(["commit", "-m", "Initial commit"])
        .current_dir(&repo_path)
        .status()
        .expect("git commit");
    assert!(status.success());

    (dir, repo_path)
}

// ── EnvVarGuard tests ─────────────────────────────────────────────────
//
// These tests use `unsafe { std::env::set_var/remove_var }` directly for
// setup and cleanup rather than going through the guard — ironic given the
// helper's purpose. This is an accepted trade-off: we're testing the
// wrapper's correctness, and the env_lock is independently exercised by
// every non-helper caller. Unique-per-test variable names prevent
// collisions within this module, but per std::env docs even writes to
// *different* variables are a data race without the lock.

#[cfg(test)]
mod env_var_guard_tests {
    use super::*;

    #[test]
    fn sets_and_restores_to_absent() {
        let guard = set_env_var("MAHBOT_TEST_SET_RESTORE", Some("hello"));
        assert_eq!(std::env::var("MAHBOT_TEST_SET_RESTORE"), Ok("hello".into()));
        drop(guard);

        // Variable was absent before the guard (unique name, first use).
        // The guard should restore to that state.
        assert!(
            std::env::var_os("MAHBOT_TEST_SET_RESTORE").is_none(),
            "guard should restore env var to absent on drop"
        );
    }

    #[test]
    fn removes_env_var() {
        // SAFETY: This bypasses env_lock (unique var name mitigates but
        // doesn't eliminate the race — see mod-level doc above). We need
        // pre-existing state to test the removal+restore path.
        unsafe {
            std::env::set_var("MAHBOT_TEST_REMOVE", "present");
        }

        let guard = set_env_var("MAHBOT_TEST_REMOVE", None);
        assert!(
            std::env::var_os("MAHBOT_TEST_REMOVE").is_none(),
            "set_env_var(key, None) should remove the variable"
        );
        drop(guard);

        // Original should be restored.
        assert_eq!(
            std::env::var("MAHBOT_TEST_REMOVE"),
            Ok("present".into()),
            "guard should restore the original value on drop"
        );

        // SAFETY: Bypasses env_lock (same trade-off as above).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_REMOVE");
        }
    }

    #[test]
    fn captures_and_restores_original_value() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::set_var("MAHBOT_TEST_CAPTURE", "original");
        }

        let guard = set_env_var("MAHBOT_TEST_CAPTURE", Some("override"));
        assert_eq!(std::env::var("MAHBOT_TEST_CAPTURE"), Ok("override".into()));
        drop(guard);

        assert_eq!(
            std::env::var("MAHBOT_TEST_CAPTURE"),
            Ok("original".into()),
            "guard should restore the original value on drop"
        );

        // SAFETY: Bypasses env_lock (same trade-off).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_CAPTURE");
        }
    }

    #[test]
    fn restores_on_panic() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_PANIC_ABSENT");
        }

        let result = std::panic::catch_unwind(|| {
            let _guard = set_env_var("MAHBOT_TEST_PANIC_ABSENT", Some("panic-value"));
            panic!("intentional panic");
        });
        assert!(result.is_err());

        assert!(
            std::env::var_os("MAHBOT_TEST_PANIC_ABSENT").is_none(),
            "MAHBOT_TEST_PANIC_ABSENT should be absent after panic-restore"
        );
    }

    #[test]
    fn restores_original_on_panic() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::set_var("MAHBOT_TEST_PANIC_ORIGINAL", "original");
        }

        let result = std::panic::catch_unwind(|| {
            let _guard = set_env_var("MAHBOT_TEST_PANIC_ORIGINAL", Some("panic-value"));
            panic!("intentional panic");
        });
        assert!(result.is_err());

        assert_eq!(
            std::env::var("MAHBOT_TEST_PANIC_ORIGINAL"),
            Ok("original".into()),
            "should restore original value after panic"
        );

        // SAFETY: Bypasses env_lock (same trade-off).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_PANIC_ORIGINAL");
        }
    }
}
