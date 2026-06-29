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

#![cfg(test)]

use crate::board::{BoardStore, Ticket, TicketParams, TicketPhase};
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

/// Fetch a ticket's status by ID, panicking if the DB query fails or the
/// ticket does not exist.
///
/// Replaces the common test boilerplate:
/// ```ignore
/// let status = store.get_ticket_status(&id).await.expect("query").expect("exists");
/// ```
/// with the more concise:
/// ```ignore
/// let status = expect_ticket_status(&store, &id).await;
/// ```
///
/// # Panics
///
/// Panics if the DB query fails or the ticket is not found (returns `None`).
/// The panic originates from within this helper function.
pub async fn expect_ticket_status(store: &BoardStore, id: &str) -> TicketPhase {
    store
        .get_ticket_status(id)
        .await
        .expect("BoardStore::get_ticket_status query failed")
        .expect("expected ticket status to exist")
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
    assert_eq!(ticket.status, TicketPhase::Cancelled);
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
    pub(crate) fn new(store: &'a BoardStore, ws: crate::Workspace) -> Self {
        Self {
            store,
            ws,
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
