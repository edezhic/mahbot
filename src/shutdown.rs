//! Global shutdown infrastructure.
//!
//! Provides a global shutdown token and signal handling for graceful daemon
//! shutdown. Used by provider, agent, management, storage, and channel
//! code to race futures against shutdown signals.
//!
//! Extracted from `self_update` where it was a layer violation — shutdown
//! coordination is not self-update.

use crate::util::UnwrapPoison;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::info;

// ── Global shutdown token ─────────────────────────────────────────────────

static GLOBAL_SHUTDOWN: OnceLock<CancellationToken> = OnceLock::new();

fn global_shutdown() -> &'static CancellationToken {
    GLOBAL_SHUTDOWN.get_or_init(CancellationToken::new)
}

/// Global graceful-drain state: a `watch` channel of `bool` set by the first
/// shutdown signal (SIGINT / window-close / self-update). Distinct from the
/// cancellation token — during the drain the token is NOT fired, so in-flight
/// LLM calls (which race the token around the HTTP send) survive to complete
/// their current round. Background loops fold this flag into their
/// sleep/shutdown races; the second signal maps to force-cancel.
static DRAIN: OnceLock<tokio::sync::watch::Sender<bool>> = OnceLock::new();

/// The process-lifetime drain sender.
fn drain_sender() -> &'static tokio::sync::watch::Sender<bool> {
    DRAIN.get_or_init(|| tokio::sync::watch::channel(false).0)
}

/// A receiver tracking the current drain value. Each call subscribes fresh, so
/// a waiter that registers before a drain begins is notified the instant it
/// flips; one that registers after reads the current value immediately.
fn drain_receiver() -> tokio::sync::watch::Receiver<bool> {
    drain_sender().subscribe()
}

/// Mark the daemon as draining (graceful-shutdown window). Idempotent.
pub fn drain_begin() {
    drain_sender().send_replace(true);
    info!("Draining: in-flight work completes before exit");
}

/// Whether the graceful-drain window is active.
#[must_use]
pub fn is_draining() -> bool {
    *drain_sender().borrow()
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
    drain_sender().send_replace(false);
}

/// Get a clone of the global shutdown token.
#[must_use]
pub fn shutdown_token() -> CancellationToken {
    global_shutdown().clone()
}

/// Clean-drain-completion trigger: fires the global token once the drain-watch
/// sees no in-flight work, ending the graceful-drain window. Gracefulness comes
/// from [`drain_begin`] — this does not start a drain (if the token already
/// fired, the watch returns before reaching this). Contrast [`force_cancel`],
/// the abort path that fires the token mid-drain to cancel in-flight work.
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

/// Completes when the drain flag flips. Event-driven: waits on the drain watch
/// channel rather than polling. Borrowing before awaiting makes it race-free —
/// a drain that began before this waiter registered resolves immediately, and
/// it is cancel-safe for `select!` use.
pub(crate) async fn drain_wait() {
    let mut rx = drain_receiver();
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            return; // sender is a process-lifetime static — unreachable
        }
    }
}

// ── Stop requests ─────────────────────────────────────────────────────────

/// A stop request the platform delivered to the daemon.
///
/// Unix has two, on an async signal stream. Windows has three, from the console
/// control handler below, and they are not the same request: only Ctrl+C is the
/// platform's "wind down" gesture.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StopRequest {
    /// Ctrl+C — the Windows counterpart of SIGINT/SIGTERM.
    Interrupt,
    /// Ctrl+Break — terminates the process by default, with no timeout.
    Break,
    /// The console window is closing; the task manager's "end task" raises the
    /// same event. The platform kills the process when its grace elapses.
    ConsoleClose,
}

#[cfg(windows)]
impl StopRequest {
    /// The name the request is logged and recorded under.
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::Interrupt => "Ctrl+C",
            Self::Break => "Ctrl+Break",
            Self::ConsoleClose => "console close",
        }
    }
}

/// What a stop request does to the process.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StopAction {
    /// Begin the graceful drain: in-flight work finishes its current round.
    Drain,
    /// Abandon the drain and run the exit path now.
    ForceCancel,
}

/// The request→action rule — the Windows half of the protocol Unix gets from
/// SIGINT/SIGTERM. Ctrl+C drains, unless a drain is already running: then it is
/// the "second request" that force-cancels. Ctrl+Break is that same hard stop in
/// its own right, and a console close is one the platform's kill deadline may cut
/// off — so neither begins a drain.
#[cfg(any(windows, test))]
#[must_use]
fn stop_action(request: StopRequest, draining: bool) -> StopAction {
    if request == StopRequest::Interrupt && !draining {
        StopAction::Drain
    } else {
        StopAction::ForceCancel
    }
}

/// The instant the platform's close grace expires: the close event's delivery plus
/// the grace it grants, from the first close that reported one.
static CLOSE_DEADLINE: OnceLock<Instant> = OnceLock::new();

/// Record the platform's close grace for the exit path's budget (first close wins)
/// and return the deadline this event grants, which its handler thread holds until —
/// returning earlier would kill the process sooner than the platform would.
#[cfg(windows)]
fn record_close_grace(grace: Duration) -> Instant {
    let deadline = Instant::now() + grace;
    let _ = CLOSE_DEADLINE.set(deadline);
    deadline
}

/// The exit path's budget from the platform's close deadline and the moment a stage
/// starts: half the grace still left, so the checkpoint keeps the rest. A deadline that
/// has passed leaves nothing.
#[must_use]
fn stage_budget(deadline: Instant, now: Instant) -> Duration {
    deadline.saturating_duration_since(now) / 2
}

/// The bound on the exit path's stage before the store checkpoint, for a stop that
/// runs under the platform's own kill deadline (a Windows console close) — `None`
/// for every other stop, whose stage stays unbounded (macOS/Linux, a dashboard
/// window close, a first Ctrl+C).
///
/// Sampled once, when that stage starts, from the deadline fixed by the first close
/// (see [`stage_budget`]): the later in the exit path the stage starts, the less of that
/// deadline is still left for it. The checkpoint that follows is never bounded — it is
/// the durable step the exit path exists for — so the rest of the grace is left for it.
/// What the bound cuts loses nothing durable: the browser releases a cut flush did not
/// settle are durable records retried at boot, and its closing sweep is best-effort by
/// design.
#[must_use]
pub fn urgent_release_budget() -> Option<Duration> {
    CLOSE_DEADLINE
        .get()
        .map(|deadline| stage_budget(*deadline, Instant::now()))
}

/// The last stop request the platform delivered, for the exit path's log lines — `None`
/// when nothing was recorded, since nothing else writes one, and the line then keeps the
/// wording it always had. A `Mutex`, not a `OnceLock`: each request overwrites the record
/// as the console protocol loop acts on it, so the last write names the last request that
/// loop acted on.
static EXIT_TRIGGER: Mutex<Option<&'static str>> = Mutex::new(None);

/// The last recorded stop request, if the platform recorded one.
#[must_use]
pub fn exit_trigger() -> Option<&'static str> {
    *EXIT_TRIGGER.lock().unwrap_poison()
}

// ── Signal handling ───────────────────────────────────────────────────────

/// Wait for stop requests, then drive the two-request drain protocol.
///
/// Unix: first signal (SIGINT or SIGTERM) begins the graceful drain via
/// [`drain_begin`] — the global token is NOT fired, so in-flight LLM calls
/// and tool groups complete. Background loops break on the drain flag.
/// The signal streams stay alive (never dropped) so a SECOND signal during
/// the drain maps to [`force_cancel`] — abort the drain, checkpoint, exit.
/// SIGHUP is explicitly ignored so the daemon survives terminal/SSH disconnects.
///
/// Windows: the same protocol, fed by the console control handler installed by
/// [`install_console_stop_handler`] (each event arrives on a fresh OS thread, so
/// the handler queues it here and, for a closing console, holds that thread; the
/// `console` module documents the platform rules and what start-up does before this
/// loop exists). Ctrl+C is the drain request, a second Ctrl+C force-cancels, and
/// Ctrl+Break and a console close are force-cancel class outright.
///
/// The "second request" is read off the global drain flag rather than a per-source
/// count, so a first Ctrl+C that follows a drain begun elsewhere — the dashboard
/// window close, a self-update's finalizing drain — force-cancels, where on Unix a
/// first SIGINT after those is a no-op. The window-close path applies the same rule
/// to itself, so this is deliberate and not a defect.
///
/// Returns only when the drain must be abandoned (a force-cancel-class request);
/// the clean-drain exit path is driven by the drain-watch task in the binary
/// (fires the token when no in-flight agents or orchestrator calls remain).
pub async fn wait_for_shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        use tracing::debug;

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

    #[cfg(windows)]
    {
        loop {
            let request = console::next_request().await?;
            let label = request.label();
            *EXIT_TRIGGER.lock().unwrap_poison() = Some(label);
            match stop_action(request, is_draining()) {
                StopAction::Drain => {
                    info!("Received {label} — draining (a second request force-cancels)");
                    drain_begin();
                }
                // The caller's signal task logs this one, naming the trigger.
                StopAction::ForceCancel => return Ok(()),
            }
        }
    }
}

// ── Windows console control handler ───────────────────────────────────────

/// Install the console stop-request handler — a no-op on macOS/Linux.
///
/// Called from `main` before boot: the handler needs no runtime (it queues for the
/// async protocol loop, see the `console` module), so the subscription is never
/// what a stop request goes missing on. A registration failure is reported, never
/// fatal: Ctrl+C then still stops the daemon through tokio's own handler.
pub fn install_console_stop_handler() {
    #[cfg(windows)]
    console::install();
}

/// The Windows console control handler — the platform's stop-request source.
///
/// Ctrl+C, Ctrl+Break and the console window closing arrive as console control
/// events, each on a **new OS thread the system creates in this process**. That
/// thread has no async context and a closing console must not be left to a task the
/// console might kill first, so the handler queues the request for
/// [`wait_for_shutdown_signal`] instead of serving it. That loop is spawned only
/// after a successful boot: while it does not exist, a request carrying a platform
/// deadline (the close) is still queued and one that carries none keeps the
/// platform's default handling (see the dead ends below).
///
/// # The platform's rules, and what they force
///
/// - A close event brings a grace the system starts counting down on delivery, from
///   the user-settable `SPI_GETHUNGAPPTIMEOUT` parameter (5000 ms by default, so it
///   is read here and never assumed). The system kills the process when it elapses,
///   and that grace is the only window a close leaves for an exit.
/// - The grace is usable only by NOT returning: the contract is "return TRUE and the
///   system terminates the process". So the close handler holds its thread while the
///   async side runs the exit path, and the system kills the process if that path
///   does not finish — the accepted hard death. Holding the full grace is
///   deliberate: returning sooner only kills the process sooner than the platform
///   would, and the hold ends at the deadline the platform is already charging for.
/// - Ctrl+C and Ctrl+Break have no timeout, and returning TRUE leaves the process
///   running, so once the loop exists they queue and return immediately.
/// - Two rapid requests may reach the process as one, because the console can
///   collapse them before the handler runs (as Unix may) — so a "second request"
///   that is never observed is not a defect. Every event the handler does receive is
///   queued on its own.
///
/// # Accepted dead ends (documented, deliberately not fixed)
///
/// - Log-off and machine shutdown: `CTRL_LOGOFF_EVENT` and `CTRL_SHUTDOWN_EVENT` are
///   not delivered to a console application that loads the GUI libraries (user32 and
///   gdi32, as this one does for its dashboard) — Windows treats it as a Windows
///   application — and are otherwise received only by services. So the handler
///   declines them and the default termination stands; the daemon is deliberately not
///   registered as a service either.
/// - External termination (the task manager's "end process", `TerminateProcess`): no
///   process can intercept it. Its "end task" is a different command, raises the
///   close event, and does become graceful here.
/// - Process-tree containment: children the daemon starts (shells, the browser CLI)
///   survive a stop when they are not stopped with it. That asymmetry belongs to
///   those subsystems, not to termination, and is left alone.
/// - Ctrl+C and Ctrl+Break before the loop exists: nothing can act on them, so they
///   keep today's platform default handling, as macOS/Linux does before its signal
///   task registers its streams. Swallowing them instead would make Ctrl+C a dead key
///   wherever boot never reaches the loop — the start-failure screen, which consumes
///   neither the drain flag nor the token.
/// - Once the loop exists the handler suppresses the default handling for Ctrl+C and
///   Ctrl+Break, so those would be inert if that loop died while the process lived. Unix
///   is no better: tokio keeps its handler installed for the whole process even once the
///   signal streams are dropped, so a died-out signal task there swallows the signal the
///   same way. No machinery is warranted: the console close and the dashboard window
///   close remain as escapes.
/// - A kill mid-exit, when a close's grace expires first: what the bound cuts is
///   covered by the durability note on [`urgent_release_budget`], and a kill that lands
///   mid-checkpoint cuts only hygiene — committed work is durable on disk. Whatever
///   state that leaves the store in is the next start's to sort out, through the same
///   bring-up gate as any other hard death here: `db::open_store` shape-checks the file
///   and then runs its data-preserving repairs.
///
/// # Remaining silent hard deaths
///
/// Those the dead ends above name, plus a stop request that arrives with the handler
/// unregistered (Ctrl+C alone still reaches tokio's handler then) and a close during a
/// start-up that never reached the loop. The detached instance after a self-update
/// receives no console events at all, so it keeps today's triggers: the dashboard
/// window close and the self-update path.
#[cfg(windows)]
mod console {
    use super::{StopRequest, record_close_grace};
    use crate::util::UnwrapPoison;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tokio::sync::Notify;
    use windows_sys::Win32::Foundation::{FALSE, TRUE};
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, CTRL_CLOSE_EVENT, SetConsoleCtrlHandler,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SPI_GETHUNGAPPTIMEOUT, SystemParametersInfoW,
    };

    /// The documented default of `SPI_GETHUNGAPPTIMEOUT`, used when it cannot be read.
    const DEFAULT_CLOSE_GRACE: Duration = Duration::from_secs(5);

    /// Requests queued by handler threads, drained in order by the protocol loop.
    static REQUESTS: Mutex<VecDeque<StopRequest>> = Mutex::new(VecDeque::new());

    /// Wakes the protocol loop. `notify_one` keeps its permit, so a request
    /// queued before the loop waits is never lost.
    static QUEUED: Notify = Notify::const_new();

    /// Whether the handler is registered. The protocol loop falls back to tokio's
    /// Ctrl+C when it is not, so a failed registration cannot cost the daemon a
    /// stop request it had before.
    static INSTALLED: AtomicBool = AtomicBool::new(false);

    /// Whether the protocol loop has started, i.e. whether the queue has a consumer
    /// (see the module docs).
    static LOOP_RUNNING: AtomicBool = AtomicBool::new(false);

    /// Register the console control handler for this process.
    pub(super) fn install() {
        // SAFETY: `SetConsoleCtrlHandler` only records the function pointer in
        // this process's handler list; `handler` has the required
        // `extern "system" fn(u32) -> i32` signature and lives for the process's
        // lifetime.
        let installed = unsafe { SetConsoleCtrlHandler(Some(handler), TRUE) != 0 };
        INSTALLED.store(installed, Ordering::SeqCst);
        if !installed {
            // Through the boot diagnostic rather than `error!`: this runs before
            // boot opens the stores, so it reaches stderr now and the logs store
            // once tracing exists — where an `error!` would be dropped.
            crate::boot::boot_diagnostic(
                "console control handler not registered — Ctrl+C still stops the daemon, but a \
                 console close and Ctrl+Break will kill it without a shutdown"
                    .to_string(),
            );
        }
    }

    /// The console control handler: queues the request for the async protocol loop
    /// and, for a close, holds this thread (see the module docs).
    extern "system" fn handler(event: u32) -> i32 {
        let Some(request) = request_for(event) else {
            // CTRL_LOGOFF_EVENT / CTRL_SHUTDOWN_EVENT and anything else: not ours,
            // so the default handler's termination stands — the dead end documented
            // on this module.
            return FALSE;
        };
        // A close event starts the platform's countdown now, so its grace is read
        // here rather than by the async side, and this thread holds until then.
        let close = request == StopRequest::ConsoleClose;
        let deadline = close.then(|| record_close_grace(close_grace()));
        // A close brings a deadline the platform charges for either way, so it is
        // queued even before the loop exists; a request that carries none is left to
        // the platform (see the dead ends on this module).
        if !LOOP_RUNNING.load(Ordering::SeqCst) && !close {
            return FALSE;
        }
        // `unwrap_poison` rather than a panic: a panic on a control-handler thread
        // aborts the process instead of shutting it down.
        REQUESTS.lock().unwrap_poison().push_back(request);
        QUEUED.notify_one();
        if let Some(deadline) = deadline {
            // Hold this thread for the grace the platform is counting down (the module
            // docs say why returning is not an option).
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        }
        TRUE
    }

    /// The next queued request, awaiting one when the queue is empty.
    ///
    /// With no handler registered, Ctrl+C through tokio's own handler is the only
    /// request source left — today's behaviour, kept so a failed registration cannot
    /// cost the daemon the stop request it had. Its failure to subscribe is a real
    /// error, so the caller's signal task reports it rather than inventing a request.
    pub(super) async fn next_request() -> anyhow::Result<StopRequest> {
        LOOP_RUNNING.store(true, Ordering::SeqCst);
        if !INSTALLED.load(Ordering::SeqCst) {
            tokio::signal::ctrl_c().await?;
            return Ok(StopRequest::Interrupt);
        }
        loop {
            let queued = REQUESTS.lock().unwrap_poison().pop_front();
            if let Some(request) = queued {
                return Ok(request);
            }
            QUEUED.notified().await;
        }
    }

    /// Map a console event onto the request class it belongs to, `None` for the
    /// events this handler deliberately leaves to the default handler.
    fn request_for(event: u32) -> Option<StopRequest> {
        match event {
            CTRL_C_EVENT => Some(StopRequest::Interrupt),
            CTRL_BREAK_EVENT => Some(StopRequest::Break),
            CTRL_CLOSE_EVENT => Some(StopRequest::ConsoleClose),
            _ => None,
        }
    }

    /// The grace this process's closing console grants (see the module docs). A stored
    /// `0` is taken at face value — no grace at all, i.e. today's hard kill — so the
    /// exit path cannot run past a deadline that has already passed.
    fn close_grace() -> Duration {
        let mut millis: u32 = 0;
        // SAFETY: `SPI_GETHUNGAPPTIMEOUT` writes a single `u32` through `pvparam`;
        // `uiparam` and `fWinIni` are unused for this action.
        let read =
            unsafe { SystemParametersInfoW(SPI_GETHUNGAPPTIMEOUT, 0, (&raw mut millis).cast(), 0) };
        if read == 0 {
            DEFAULT_CLOSE_GRACE
        } else {
            Duration::from_millis(u64::from(millis))
        }
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

#[cfg(unix)]
const SIGBUS_MSG: &str = "mahbot: caught SIGBUS (bus error), terminating\n";
#[cfg(unix)]
const SIGABRT_MSG: &str = "mahbot: caught SIGABRT (abort), terminating\n";

#[cfg(unix)]
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

// ── Panic hook ────────────────────────────────────────────────────────────

/// Install a global panic hook that prints a timestamped marker line before
/// the default panic report, so panics captured in update.log (the replacement
/// daemon's stderr) are time-attributable. The default hook (message +
/// backtrace) still runs.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        crate::boot::timestamped_stderr(&info.to_string());
        default_hook(info);
    }));
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// The decision logic above is platform-independent by construction, so the host
/// lane can exercise it even though only the Windows branch calls it.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_first_interrupt_drains() {
        for (request, draining, expected) in [
            (StopRequest::Interrupt, false, StopAction::Drain),
            (StopRequest::Interrupt, true, StopAction::ForceCancel),
            (StopRequest::Break, false, StopAction::ForceCancel),
            (StopRequest::ConsoleClose, false, StopAction::ForceCancel),
        ] {
            assert_eq!(
                stop_action(request, draining),
                expected,
                "{request:?}, draining={draining}"
            );
        }
    }

    #[test]
    fn the_release_budget_samples_half_the_remaining_grace() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(5);
        assert_eq!(stage_budget(deadline, now), Duration::from_millis(2500));
        // Sampled from what is left when the stage starts, not from the original
        // grace: the later that is, the less the stage gets.
        assert_eq!(
            stage_budget(deadline, now + Duration::from_secs(4)),
            Duration::from_millis(500)
        );
        // A grace already spent leaves nothing to spend, so the stage is cut at once
        // and the exit path goes straight to the checkpoint.
        assert_eq!(stage_budget(now, now), Duration::ZERO);
    }
}
