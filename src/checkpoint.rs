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
//! - Periodically as defense-in-depth against crashes (see
//!   [`spawn_auto_checkpoint_loop`]).
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

/// How often to auto-checkpoint all databases as defense-in-depth.
const AUTO_CHECKPOINT_INTERVAL: std::time::Duration = std::time::Duration::from_mins(5);

/// Spawn a background task that periodically checkpoints all databases.
///
/// Uses [`tokio::task::JoinSet`] tracking and [`std::panic::catch_unwind`] panic protection,
/// matching the style of `spawn_cancellable` in `main.rs`.
///
/// This is defense-in-depth: even if the process crashes or a shutdown path
/// misses the explicit checkpoint, the periodic checkpoint ensures that at
/// most `AUTO_CHECKPOINT_INTERVAL` worth of writes can be lost.
///
/// The loop races against the supplied cancellation token and exits when the
/// token is cancelled (graceful shutdown). The task is inserted into
/// `tasks` so the caller can await completion during teardown.
pub fn spawn_auto_checkpoint_loop(
    tasks: &mut tokio::task::JoinSet<()>,
    shutdown_token: &tokio_util::sync::CancellationToken,
) {
    use futures_util::FutureExt;
    use std::panic::AssertUnwindSafe;

    let cancel = shutdown_token.clone();
    tasks.spawn(async move {
        loop {
            tokio::select! {
                biased; // check cancellation before sleeping on every iteration
                () = cancel.cancelled() => {
                    info!("Auto-checkpoint loop stopped (shutdown)");
                    break;
                }
                () = tokio::time::sleep(AUTO_CHECKPOINT_INTERVAL) => {
                    let result = AssertUnwindSafe(checkpoint_all_databases())
                        .catch_unwind()
                        .await;
                    if let Err(payload) = result {
                        error!(
                            "Background task panicked [auto-checkpoint]: {}",
                            crate::util::panic_message(&*payload),
                        );
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::task::JoinSet;

    /// Verify that `spawn_auto_checkpoint_loop` exits promptly when the
    /// cancellation token is fired (without waiting for the sleep interval).
    #[tokio::test]
    async fn exits_on_cancellation() {
        let token = tokio_util::sync::CancellationToken::new();
        let mut tasks = JoinSet::new();
        spawn_auto_checkpoint_loop(&mut tasks, &token);

        // Give the spawned task a moment to enter the select! loop.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Cancel — the loop should exit on the next iteration.
        token.cancel();

        // The task should complete within a reasonable timeout.
        let result = tokio::time::timeout(Duration::from_millis(500), tasks.join_next()).await;
        assert!(result.is_ok(), "task did not exit after cancellation");
    }

    /// Verify that `checkpoint_all_databases` is a no-op (no panic) when
    /// no stores are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        // No stores initialized — all get() calls return None.
        // This should not panic or error.
        checkpoint_all_databases().await;
    }
}
