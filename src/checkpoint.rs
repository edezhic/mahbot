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
use std::future::Future;
use tracing::{error, info, warn};

/// Iterate all stores via [`crate::turso::iter_checkpoint_stores`] and run an
/// async operation on each initialized store in parallel.
///
/// This is the shared iteration pattern used by both
/// [`checkpoint_all_databases`] and [`verify_all_databases`]. Stores that
/// haven't been initialized yet (connection is `None`) are silently skipped.
///
/// The operation closure receives `(&'static str, &'static Connection)` — the
/// store name and the canonical connection — and should return a `Future` that
/// completes the operation and logs the result.
async fn for_each_store<F, Fut>(op: F)
where
    F: Fn(&'static str, &'static crate::turso::Connection) -> Fut,
    Fut: Future<Output = ()>,
{
    let futs: Vec<_> = crate::turso::iter_checkpoint_stores()
        .filter_map(|(name, conn_opt)| {
            let conn = conn_opt?;
            Some(op(name, conn))
        })
        .collect();
    join_all(futs).await;
}

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
    for_each_store(|name, conn| async move {
        match conn.checkpoint().await {
            Ok(()) => info!(db = %name, "Database WAL checkpointed"),
            Err(e) => warn!(error = %e, db = %name, "Failed to checkpoint database WAL"),
        }
    })
    .await;
}

/// Run PRAGMA quick_check on all initialized database stores.
///
/// Iterates all stores via [`crate::turso::iter_checkpoint_stores`], runs
/// `quick_check` on each in parallel, and logs the results. Corruption
/// errors are logged at `error!` level for operator visibility in the
/// dashboard Logs page, and a full `PRAGMA integrity_check` is
/// automatically triggered on the affected store so the complete
/// diagnostic report is available without manual intervention.
/// Successes are logged at `info!` level.
///
/// Skips stores that haven't been initialized yet. This is a fire-and-forget
/// function: all per-store errors are logged and swallowed to avoid blocking
/// the caller (matching the pattern of [`checkpoint_all_databases`]).
pub async fn verify_all_databases() {
    for_each_store(|name, conn| async move {
        match conn.quick_check().await {
            Ok(()) => info!(db = %name, "Database integrity check passed"),
            Err(e) => {
                error!(error = %e, db = %name, "Database integrity check failed, running full diagnostic");

                // Run the full integrity_check to get the complete diagnostic
                // report so operators can triage without running debug CLI.
                match conn.integrity_check().await {
                    Ok(problems) if problems.is_empty() => {
                        // quick_check reported corruption but integrity_check
                        // found nothing — unexpected but handle gracefully.
                        warn!(
                            db = %name,
                            "Full integrity check returned no problems after quick_check failure"
                        );
                    }
                    Ok(problems) => {
                        for problem in &problems {
                            error!(db = %name, problem = %problem, "Integrity issue");
                        }
                        let count = problems.len();
                        error!(
                            db = %name, count,
                            "Full integrity check found {} issue(s) in {}",
                            count, name,
                        );
                    }
                    Err(diag_err) => {
                        error!(
                            error = %diag_err, db = %name,
                            "Full integrity check also failed — database corruption may be severe"
                        );
                    }
                }
            }
        }
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that both `checkpoint_all_databases` and `verify_all_databases` are
    /// no-ops (no panic) when no stores are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        checkpoint_all_databases().await;
        verify_all_databases().await;
    }
}
