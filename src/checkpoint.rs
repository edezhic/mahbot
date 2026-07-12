//! WAL checkpointing for all Turso database stores.
//!
//! MahBot uses `multiprocess_wal` mode (configured in [`crate::turso::EXPERIMENTAL_FEATURES`]),
//! which relies on a `.tshm` shared-memory file for WAL coordination between
//! connections. On process crash or hard exit (`std::process::exit(0)` or signal
//! without cleanup), the `.tshm` file becomes stale and pending WAL frames are
//! permanently lost.
//!
//! This module provides the canonical [`checkpoint_all_databases`] function that
//! forces a WAL checkpoint (TRUNCATE mode) on every initialized store. It should
//! be called:
//!
//! - Before any hard process termination (self-update restart, GUI exit).
//! - On graceful shutdown signals (SIGTERM/SIGINT) to flush pending writes.
//! - Periodically as defense-in-depth against crashes (the auto-checkpoint
//!   loop spawned by the binary's background task set).
//!
//! # Why keep `multiprocess_wal`?
//!
//! `multiprocess_wal` (via Turso) forces `NoLock` on all connections, making
//! explicit WAL checkpoints the only way to guarantee durability on exit
//! (the root cause this module works around). Removing it would eliminate the
//! data-loss mechanism entirely for normal (non-`exit(0)`) exits.
//!
//! The feature is retained because `mahbot debug` (the CLI subcommand) opens
//! the same `.db` files while the daemon is running. Without
//! `multiprocess_wal`, the debug tool and the daemon would share a single
//! WAL file without coordination — strictly worse than the current approach
//! (all connections share a single WAL with `.tshm` coordination). A future
//! refactor could eliminate the debug CLI's need to access live databases
//! (e.g., via an IPC query endpoint), making `multiprocess_wal` removable.
//!
//! In the meantime, the checkpoint orchestration in this module (exit-time +
//! periodic) ensures that under normal operation no writes are lost, and
//! crash data loss is bounded to the auto-checkpoint interval (5 minutes).

use futures_util::future::join_all;
use tracing::{error, info, warn};

/// Checkpoint all Turso database stores before hard process termination.
///
/// `std::process::exit(0)` bypasses Rust destructors, so Turso WAL connections
/// are never properly closed. Without this explicit checkpoint, pending WAL
/// writes are silently lost and `.tshm` coordination files are left inconsistent.
///
/// Skips stores that haven't been initialized yet. Logs and swallows per-store
/// errors to avoid blocking shutdown.
///
/// The store entries come from [`crate::turso::iter_checkpoint_stores`] — the
/// single source of truth for which stores get checkpointed.
pub async fn checkpoint_all_databases() {
    let futs: Vec<_> = crate::turso::iter_checkpoint_stores()
        .filter_map(|(name, conn_opt)| {
            let conn = conn_opt?;
            Some(async move {
                match conn.checkpoint().await {
                    Ok(()) => info!(db = %name, "Database WAL checkpointed"),
                    Err(e) => warn!(error = %e, db = %name, "Failed to checkpoint database WAL"),
                }
            })
        })
        .collect();
    join_all(futs).await;
}

/// Run PRAGMA quick_check on all initialized database stores.
///
/// Iterates all stores via [`crate::turso::iter_checkpoint_stores`], runs
/// `quick_check` on each in parallel, and logs the results. Corruption
/// errors are logged at `error!` level for operator visibility in the
/// dashboard Logs page. Successes are logged at `info!` level.
///
/// Skips stores that haven't been initialized yet. This is a fire-and-forget
/// function: all per-store errors are logged and swallowed to avoid blocking
/// the caller (matching the pattern of [`checkpoint_all_databases`]).
pub async fn verify_all_databases() {
    let futs: Vec<_> = crate::turso::iter_checkpoint_stores()
        .filter_map(|(name, conn_opt)| {
            let conn = conn_opt?;
            Some(async move {
                match conn.quick_check().await {
                    Ok(()) => info!(db = %name, "Database integrity check passed"),
                    Err(e) => error!(error = %e, db = %name, "Database integrity check failed"),
                }
            })
        })
        .collect();
    join_all(futs).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `checkpoint_all_databases` is a no-op (no panic) when
    /// no stores are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        // No stores initialized — all get() calls return None.
        // This should not panic or error.
        checkpoint_all_databases().await;
    }

    /// Verify that `verify_all_databases` is a no-op (no panic) when
    /// no stores are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn verify_noop_when_no_stores() {
        // No stores initialized — all get() calls return None.
        // This should not panic or error.
        verify_all_databases().await;
    }
}
