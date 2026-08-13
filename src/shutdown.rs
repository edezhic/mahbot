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

/// Global graceful-drain flag (decision 2): set by the first shutdown signal
/// (SIGINT / window-close / self-update). Distinct from the cancellation
/// token — during the drain the token is NOT fired, so in-flight LLM calls
/// (which race the token around the HTTP send) survive to complete their
/// current round. Background loops fold this flag into their sleep/shutdown
/// races; the second signal maps to force-cancel.
static DRAINING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Mark the daemon as draining (graceful-shutdown window). Idempotent.
pub fn drain_begin() {
    DRAINING.store(true, std::sync::atomic::Ordering::SeqCst);
    info!("Draining: in-flight work completes before exit");
}

/// Whether the graceful-drain window is active.
#[must_use]
pub fn is_draining() -> bool {
    DRAINING.load(std::sync::atomic::Ordering::SeqCst)
}

/// Whether the daemon is aborting: the shutdown token fired OR the graceful
/// drain is active. Loops gate NEW work on this (during the drain, in-flight
/// work completes but nothing new starts).
#[must_use]
pub fn aborting() -> bool {
    shutdown_token().is_cancelled() || is_draining()
}

/// Force-cancel the drain: fire the global token immediately (in-flight
/// agents are cancelled and boot-resume via status='launched'), then the normal
/// exit path (checkpoint + join + exit) runs.
pub fn force_cancel() {
    global_shutdown().cancel();
}

/// Clear the drain flag. Production code never clears it (drains are
/// one-way); tests use this to restore isolation after asserting drain
/// behavior.
pub fn drain_clear() {
    DRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
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

/// Sleep for the given duration, breaking early on the shutdown token OR the
/// graceful-drain flag. Background loops fold the drain into their sleep
/// cycles so they stop spawning new work when the drain begins.
/// Returns `true` if the sleep completed normally, `false` if shutdown or
/// draining was signaled.
#[must_use]
pub async fn sleep_or_shutdown_or_drain(duration: Duration) -> bool {
    let token = shutdown_token();
    let drain = drain_wait();
    tokio::pin!(drain);
    tokio::select! {
        () = token.cancelled() => false,
        () = &mut drain => false,
        () = tokio::time::sleep(duration) => true,
    }
}

/// Completes when the drain flag flips. Polled inside
/// [`sleep_or_shutdown_or_drain`] every select round; the 100 ms poll cadence
/// is negligible against 10-minute cleanup loops. Also used by the
/// leader-stagger wait so a graceful drain releases followers immediately.
pub(crate) async fn drain_wait() {
    while !is_draining() {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ── Signal handling ───────────────────────────────────────────────────────

/// Wait for shutdown signals, then drive the two-signal drain protocol.
///
/// First signal (SIGINT or SIGTERM): begins the graceful drain via
/// [`drain_begin`] — the global token is NOT fired, so in-flight LLM calls
/// and tool groups complete. Background loops break on the drain flag.
/// The signal streams stay alive (never dropped) so a SECOND signal during
/// the drain maps to [`force_cancel`] — abort the drain, checkpoint, exit.
///
/// Returns only after a second signal (force-cancel); the clean-drain exit
/// path is driven by the drain-watch task in the binary (fires the token
/// when no in-flight agents or orchestrator calls remain).
///
/// SIGHUP is explicitly ignored so the daemon survives terminal/SSH disconnects.
pub async fn wait_for_shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut sigint = signal(SignalKind::interrupt())?;
        let mut sigterm = signal(SignalKind::terminate())?;
        let mut sighup = signal(SignalKind::hangup())?;

        let mut first = true;
        loop {
            let signal = tokio::select! {
                _ = sigint.recv() => "SIGINT",
                _ = sigterm.recv() => "SIGTERM",
                _ = sighup.recv() => {
                    debug!("Received SIGHUP, ignoring (daemon stays running)");
                    continue;
                }
            };
            if first {
                info!("Received {signal} — draining (second signal force-cancels)");
                drain_begin();
                first = false;
            } else {
                info!("Received second {signal} — force-cancelling drain");
                return Ok(());
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
        info!("Received Ctrl+C — draining");
        drain_begin();
        // Non-unix: no second-signal path; the drain-watch drives completion.
        tokio::future::pending::<()>().await;
        Ok(())
    }
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
