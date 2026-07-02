//! Global shutdown infrastructure.
//!
//! Provides a global shutdown token and signal handling for graceful daemon
//! shutdown. Used by provider, agent, management, storage, and channel
//! code to race futures against shutdown signals.
//!
//! Extracted from `self_update` where it was a layer violation — shutdown
//! coordination is not self-update.

use std::sync::OnceLock;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

// ── Global shutdown token ─────────────────────────────────────────────────

static GLOBAL_SHUTDOWN: OnceLock<CancellationToken> = OnceLock::new();

fn global_shutdown() -> &'static CancellationToken {
    GLOBAL_SHUTDOWN.get_or_init(CancellationToken::new)
}

/// Get a clone of the global shutdown token.
#[must_use]
pub fn shutdown_token() -> CancellationToken {
    global_shutdown().clone()
}

/// Trigger a graceful shutdown of the daemon.
pub fn shutdown() {
    global_shutdown().cancel();
}

/// Error returned by [`race_shutdown`] when the global shutdown token fires.
pub struct Shutdown;

/// Race a future against the global shutdown token.
/// Returns `Ok(T)` if the future completes first, `Err(Shutdown)` if shutdown is signaled.
pub async fn race_shutdown<F, T>(fut: F) -> Result<T, Shutdown>
where
    F: std::future::Future<Output = T>,
{
    let token = shutdown_token();
    tokio::select! {
        result = fut => Ok(result),
        () = token.cancelled() => Err(Shutdown),
    }
}

/// Sleep for the given duration, or return early if shutdown is signaled.
/// Returns `true` if the sleep completed normally, `false` if shutdown was signaled.
#[must_use]
pub async fn sleep_or_shutdown(duration: Duration) -> bool {
    race_shutdown(tokio::time::sleep(duration)).await.is_ok()
}

// ── Signal handling ───────────────────────────────────────────────────────

/// Wait for a shutdown signal (SIGINT or SIGTERM), then return so the
/// caller can trigger graceful shutdown via the global cancellation token.
///
/// This is spawned as a background task in `spawn_background_tasks()`. When
/// a signal arrives, the task calls [`shutdown()`], which cancels the global
/// token. The Iced dashboard subscription picks this up and closes the
/// window, which unblocks `iced::application::run` so the
/// process can tear down cleanly via `shutdown_after_dashboard()`.
///
/// SIGHUP is explicitly ignored so the daemon survives terminal/SSH disconnects.
pub async fn wait_for_shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut sigint = signal(SignalKind::interrupt())?;
        let mut sigterm = signal(SignalKind::terminate())?;
        let mut sighup = signal(SignalKind::hangup())?;

        loop {
            tokio::select! {
                _ = sigint.recv() => {
                    info!("Received SIGINT, shutting down...");
                    break;
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, shutting down...");
                    break;
                }
                _ = sighup.recv() => {
                    debug!("Received SIGHUP, ignoring (daemon stays running)");
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
        info!("Received Ctrl+C, shutting down...");
    }

    Ok(())
}

// ── Fatal signal handlers ─────────────────────────────────────────────────

/// Install bare-metal signal handlers for fatal signals (SIGBUS, SIGABRT).
///
/// These are separate from the tokio-based [`wait_for_shutdown_signal`] — that
/// handles graceful shutdown (SIGINT/SIGTERM). The handlers here catch
/// *unexpected* fatal signals that would otherwise kill the process silently
/// with no diagnostic output.
///
/// On first call, installs handlers via `libc::signal`. Safe to call
/// multiple times — only the first call installs handlers.
///
/// The handlers write a one-line diagnostic message to stderr using the
/// async-signal-safe `write(2)` syscall, then `_exit(1)`. No heap allocation,
/// no locks, no stdio — safe to call from within a signal handler.
#[cfg(unix)]
pub fn install_fatal_signal_handlers() {
    use std::sync::Once;
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: `libc::signal` is async-signal-safe. The handler functions
        // use STATIC string constants (no heap allocation) and call only
        // `libc::write` (raw syscall) and `libc::_exit` — both
        // async-signal-safe per POSIX.
        unsafe {
            libc::signal(
                libc::SIGBUS,
                fatal_signal_handler as *const () as libc::sighandler_t,
            );
            libc::signal(
                libc::SIGABRT,
                fatal_signal_handler as *const () as libc::sighandler_t,
            );
        }
    });
}

const SIGBUS_MSG: &str = "mahbot: caught SIGBUS (bus error), terminating\n";
const SIGABRT_MSG: &str = "mahbot: caught SIGABRT (abort), terminating\n";

extern "C" fn fatal_signal_handler(sig: i32) {
    let msg = match sig {
        libc::SIGBUS => SIGBUS_MSG,
        libc::SIGABRT => SIGABRT_MSG,
        _ => "mahbot: caught unknown fatal signal, terminating\n",
    };
    // SAFETY: write(2) and _exit(2) are async-signal-safe per POSIX.
    unsafe {
        let _ = libc::write(
            libc::STDERR_FILENO,
            msg.as_ptr().cast::<libc::c_void>(),
            msg.len(),
        );
        libc::_exit(1);
    }
}

#[cfg(not(unix))]
pub fn install_fatal_signal_handlers() {
    // No-op on non-Unix platforms.
}
