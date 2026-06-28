//! Shared test utilities for initializing global stores.
//!
//! Provides a single temporary directory shared across all test initializations,
//! eliminating the duplicated `init_with_temp_dir()` pattern that was previously
//! required in each module and leaked a separate temp dir per store.
//!
//! The temp directory is intentionally leaked for the process lifetime to avoid
//! races at shutdown — since global [`OnceCell`]s can only be set once, each
//! store is initialized at most once per test run.

#![cfg(test)]

/// Assert that a column-string constant has exactly `LAST + 1` entries.
///
/// Every `*_COLUMNS` string constant (used for `SELECT col1, col2, ...`)
/// must stay in sync with the corresponding `COL_*` index constants.
/// This macro verifies the structural invariant: the number of comma-separated
/// entries in `$cols_str` must equal the highest `COL_*` index + 1.
///
/// # Example
///
/// ```ignore
/// assert_column_count!(TICKET_COLUMNS, COL_TICKET_PIPELINE_RESERVATION);
/// ```
///
/// # Panics
///
/// Panics if the entry count does not match `$last + 1`.
#[macro_export]
macro_rules! assert_column_count {
    ($cols_str:expr, $last:expr) => {
        let count = $cols_str.split(',').count();
        assert_eq!(
            $last + 1,
            count,
            "{} has {count} entries but {} ({}) + 1 = {}",
            stringify!($cols_str),
            stringify!($last),
            $last,
            $last + 1,
        );
    };
}

use crate::board::{BoardStore, Ticket, TicketPhase};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Shared test root directory, created once and intentionally leaked
/// for the process lifetime.
static TEST_ROOT: OnceLock<PathBuf> = OnceLock::new();

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

/// Initialize all global test stores (session, board) with a shared
/// temp directory.
///
/// Idempotent — subsequent calls are no-ops.
pub async fn init_test_stores() {
    use crate::session::SessionStore;

    static INIT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    INIT.get_or_init(|| async {
        // Set CONFIG storage root (no-op if already set by another test)
        let _ = crate::config::CONFIG.try_set_storage_root(test_root().clone());

        crate::session::SESSIONS
            .set(
                SessionStore::open(test_root())
                    .await
                    .expect("failed to create SessionStore for tests"),
            )
            .expect("SESSIONS already initialized by another path");
        crate::board::BOARD
            .set(
                BoardStore::open(test_root())
                    .await
                    .expect("failed to create BoardStore for tests"),
            )
            .expect("BOARD already initialized by another path");
    })
    .await;
}
