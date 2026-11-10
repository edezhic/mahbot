//! Global shutdown infrastructure: the shutdown token, the graceful-drain flag and the
//! platform's stop-request sources, used to race futures against shutdown.
//!
//! # Exits that run none of the closing path
//!
//! The closing path is the library's `gui::Dashboard::save_and_exit` (draft flush, window
//! state, the token fired before the iced runtime drops) plus the binary's
//! `shutdown_after_dashboard` (browser release, background-task join, store checkpoint, the
//! stop of the supervised Telegram Bot API server child). These exits run none of it, listed
//! here once so the rest of the tree points at this list:
//!
//! - a platform-forced termination (Force Quit, `SIGKILL`): nothing can intercept it; a plain
//!   `kill` is SIGTERM, which is handled once the signal task is installed;
//! - the self-update tail (its own drain and `exit(0)` after its checkpoint);
//! - the fatal-signal and abort paths (`install_fatal_signal_handlers`, the panic hook);
//! - an iced `run()` that returns an error;
//! - macOS log-out, restart or shut-down: the quit hook hands the platform's ask back and
//!   AppKit ends the process, losing the last moment of typing, the exit-time tidying (the
//!   next start is slower) and the browser release (which the next start does anyway);
//! - an ordinary macOS quit the hook does not take over (the interface is already past its
//!   exit, or the interception could not be installed);
//! - a SIGTERM arriving before the signal task exists;
//! - the exit bound `macos_quit` arms for a taken-over quit: a hard `_exit` when the closing
//!   path overruns it — the same class as a force quit.
//!
//! Since none of them runs the closing path, the Telegram Bot API server child is left
//! running; the next start adopts or reaps it by verifying the recorded process, never by
//! signalling a pid it cannot verify.
//!
//! A Windows session end is not on the list: the listener stops the daemon like any other
//! stop request, so the closing path runs within the platform's deadline — unless the
//! notification arrives before the dashboard subscribes. The CLI subcommands and a launch
//! the instance lock turns away reach none of the closing path either, having exited before
//! the runtime exists. A new exit path that drops the runtime without firing the token
//! belongs on this list.
//!
//! A hangup and the catchable suspending signals are not stop requests at all and never end
//! the daemon ([`install_ignored_signals`] takes them out of play from the first
//! instructions, and [`wait_for_shutdown_signal`] drops a hangup ever after).
//! `SIGKILL`/`SIGSTOP` are nobody's to refuse.

use crate::util::UnwrapPoison;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::info;

static GLOBAL_SHUTDOWN: OnceLock<CancellationToken> = OnceLock::new();

fn global_shutdown() -> &'static CancellationToken {
    GLOBAL_SHUTDOWN.get_or_init(CancellationToken::new)
}

/// Graceful-drain state: a `watch` channel of `bool` set by the first shutdown signal
/// (SIGINT, a close that could not hide, a quit, self-update). Distinct from the token —
/// during the drain the token is NOT fired, so in-flight LLM calls racing the token around
/// the HTTP send complete their round. Background loops fold the flag into their races; the
/// second signal maps to force-cancel.
static DRAIN: OnceLock<tokio::sync::watch::Sender<bool>> = OnceLock::new();

fn drain_sender() -> &'static tokio::sync::watch::Sender<bool> {
    DRAIN.get_or_init(|| tokio::sync::watch::channel(false).0)
}

/// Each call subscribes fresh: a waiter registered before a drain begins is notified the
/// instant it flips, one registered after reads the current value.
fn drain_receiver() -> tokio::sync::watch::Receiver<bool> {
    drain_sender().subscribe()
}

/// Mark the daemon as draining (graceful-shutdown window). Idempotent.
pub fn drain_begin() {
    drain_sender().send_replace(true);
    info!("Draining: in-flight work completes before exit");
}

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
/// Force-cancel the drain: fire the global token immediately (in-flight agents are cancelled and
/// boot-resume via status='launched'), then the normal exit path (checkpoint + join + exit) runs.
pub fn force_cancel() {
    global_shutdown().cancel();
}

/// Production never clears it (drains are one-way); tests restore isolation with it.
pub fn drain_clear() {
    drain_sender().send_replace(false);
}

#[must_use]
pub fn shutdown_token() -> CancellationToken {
    global_shutdown().clone()
}

/// Fires the global token once the drain-watch sees no in-flight work, ending the drain
/// window. Gracefulness comes from [`drain_begin`] — this starts no drain. Contrast
/// [`force_cancel`], which fires the token mid-drain to cancel in-flight work.
pub fn shutdown() {
    global_shutdown().cancel();
}

/// Error returned by [`race_shutdown`] when the global shutdown token fires.
pub struct Shutdown;

/// `Ok(T)` if the future completes first, `Err(Shutdown)` if shutdown is signalled.
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

/// `true` if the sleep completed, `false` if shutdown was signalled.
#[must_use]
pub async fn sleep_or_shutdown(duration: Duration) -> bool {
    race_shutdown(tokio::time::sleep(duration)).await.is_ok()
}

/// Breaks early on the shutdown token OR the graceful-drain flag, so a background loop stops
/// spawning new work when the drain begins. `true` if the sleep completed.
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

/// Completes when the drain flag flips, waiting on the watch channel rather than polling.
/// Borrowing before awaiting makes it race-free (a drain that began earlier resolves at once)
/// and cancel-safe for `select!`.
pub(crate) async fn drain_wait() {
    let mut rx = drain_receiver();
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            return; // sender is a process-lifetime static — unreachable
        }
    }
}

/// Unix has two, on an async signal stream; Windows has three, from the console control
/// handler below, and only Ctrl+C is the platform's "wind down" gesture. A launch with no
/// console — the windowed product's normal one — receives none of them. A session end
/// (log-off, shutdown, restart) is a stop request too, but not one of these and not a drain:
/// it comes from the window listener ([`install_session_end_listener`]).
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StopRequest {
    /// Ctrl+C — the Windows counterpart of SIGINT/SIGTERM.
    Interrupt,
    /// Ctrl+Break — terminates the process by default, with no timeout.
    Break,
    /// Only a launch with a console can receive it; the task manager's "end task" raises the
    /// same event. The platform kills the process when its grace elapses.
    ConsoleClose,
}

#[cfg(windows)]
impl StopRequest {
    /// The name it is logged and recorded under.
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

/// The Windows half of the protocol Unix gets from SIGINT/SIGTERM. Ctrl+C drains, unless a
/// drain is already running — then it is the second request that force-cancels. Ctrl+Break is
/// that hard stop in its own right, and a console close may be cut off by the platform's kill
/// deadline, so neither begins a drain.
#[cfg(any(windows, test))]
#[must_use]
fn stop_action(request: StopRequest, draining: bool) -> StopAction {
    if request == StopRequest::Interrupt && !draining {
        StopAction::Drain
    } else {
        StopAction::ForceCancel
    }
}

/// Set when a stop carrying a grace arrived (a Windows console close or session end), plus the
/// grace granted; `None` until then.
static STOP_DEADLINE: Mutex<Option<Instant>> = Mutex::new(None);

/// The documented default of `SPI_GETHUNGAPPTIMEOUT`, the user-settable parameter behind a
/// console close. A session end has no readable equivalent and takes the same bound.
#[cfg(windows)]
const DEFAULT_STOP_GRACE: Duration = Duration::from_secs(5);

/// Record the deadline a stop has to finish before, for the exit path's budget. The earliest one
/// wins: that is the kill the process must beat.
#[cfg(windows)]
fn record_stop_deadline(deadline: Instant) {
    let mut recorded = STOP_DEADLINE.lock().unwrap_poison();
    if recorded.is_none_or(|current| deadline < current) {
        *recorded = Some(deadline);
    }
}

/// The exit path's budget from the platform's stop deadline and the moment a stage starts: half the
/// grace still left, so the checkpoint keeps the rest. A deadline that has passed leaves nothing.
#[must_use]
fn stage_budget(deadline: Instant, now: Instant) -> Duration {
    deadline.saturating_duration_since(now) / 2
}

/// The bound on the exit path's stage before the store checkpoint, for a stop that runs under the
/// platform's own kill deadline (a Windows console close or session end): `None` for every other
/// stop, which no platform deadline bounds (the tray's quit, a close that could not hide, a first
/// Ctrl+C, a macOS quit).
///
/// Sampled once, when that stage starts, from the deadline the platform's stops recorded
/// (`record_stop_deadline` keeps the earliest — the kill the process must beat); [`stage_budget`]
/// takes half of what is still left, so the later in the exit path the stage starts the less it
/// gets. The checkpoint that follows is never bounded — it is the durable step the exit path
/// exists for. A cut loses nothing durable: an unsettled browser release is retried at boot, and
/// the flush is best-effort.
#[must_use]
pub fn urgent_release_budget() -> Option<Duration> {
    STOP_DEADLINE
        .lock()
        .unwrap_poison()
        .map(|deadline| stage_budget(deadline, Instant::now()))
}

/// Written by the Windows stop deliveries (the console protocol loop and the session-end
/// listener), a macOS quit and the tray menu's quit — the last two only where that request
/// drives the exit, so a repeat gesture leaves the record to the request that caused it. `None`
/// when nothing was recorded (a close that could not hide, a Unix signal), and the exit line
/// then keeps its old wording. A `Mutex`, not a `OnceLock`: a later request takes over.
///
/// The exit line is dropped outright — it runs after the iced runtime whose log writer batches
/// — so this makes a stop's reason nameable by that line, not visible.
static EXIT_TRIGGER: Mutex<Option<&'static str>> = Mutex::new(None);

/// Record the stop request the exit path is running for, so the exit path can name it.
pub(crate) fn record_exit_trigger(label: &'static str) {
    *EXIT_TRIGGER.lock().unwrap_poison() = Some(label);
}

/// The last recorded stop request, if any was recorded — `EXIT_TRIGGER` above says what writes one.
#[must_use]
pub fn exit_trigger() -> Option<&'static str> {
    *EXIT_TRIGGER.lock().unwrap_poison()
}

/// `None` for a notification this daemon must not act on.
/// The platform asks twice: `WM_QUERYENDSESSION` (answered affirmatively, never acted on),
/// then the final notification. `ending` is its `wParam` — false when a started shutdown is
/// cancelled, so only a true one may stop the daemon. `closes_app_only` is the `lParam`'s
/// `ENDSESSION_CLOSEAPP` bit, the platform closing this application alone for an update: it
/// ends no session, and stopping would leave the daemon down on a continuing one. `log_off` is
/// the `ENDSESSION_LOGOFF` bit; without it the end is a shutdown or restart. The critical end
/// sets neither bit and is a session end like any other.
#[cfg(any(windows, test))]
#[must_use]
fn session_end_label(ending: bool, closes_app_only: bool, log_off: bool) -> Option<&'static str> {
    if !ending || closes_app_only {
        return None;
    }
    Some(if log_off {
        "Windows log-off"
    } else {
        "Windows shutdown/restart"
    })
}

/// Act on a session end: name it for the exit path, bound the stages before the exit-time
/// checkpoint, and force the stop.
///
/// Forced, not drained: the platform's seconds cannot hold the graceful drain, so waiting would
/// leave the stop unfinished. It waits for nothing and asks the platform for nothing (no
/// shutdown-blocking reason, the one way to gain time), running on the listener's thread and
/// returning at once — no "preventing shutdown" surface can appear. The stop sequence (draft and
/// geometry flush, browser release, journal checkpoint) still runs, inside the granted time.
#[cfg(windows)]
fn stop_for_session_end(label: &'static str) {
    record_exit_trigger(label);
    record_stop_deadline(Instant::now() + DEFAULT_STOP_GRACE);
    info!("{label} — forcing the stop: the platform's seconds cannot hold a drain");
    force_cancel();
}

/// Unix: the first SIGINT or SIGTERM begins the graceful drain via [`drain_begin`] — the token is
/// NOT fired, so in-flight LLM calls and tool groups complete. The streams stay alive, so a SECOND
/// signal maps to [`force_cancel`]. SIGHUP is ignored so the daemon survives terminal/SSH
/// disconnects.
///
/// Windows: the same protocol, fed by `install_console_stop_handler` (each event arrives on a
/// fresh OS thread, so it is queued here and, for a closing console, that thread is held).
/// Ctrl+C drains, a second Ctrl+C force-cancels, and Ctrl+Break and a console close are
/// force-cancel class outright. A session end arrives through `install_session_end_listener`.
///
/// The "second request" comes from the global drain flag, not a per-source count, so a Ctrl+C
/// after a drain begun elsewhere (a quit, a close that could not hide, a self-update drain)
/// force-cancels where on Unix a first SIGINT is a no-op — matching
/// `gui::Dashboard::request_exit`.
///
/// Returns only when the drain must be abandoned, naming that request where the platform names
/// it. The clean-drain exit is driven by the binary's drain-watch task.
pub async fn wait_for_shutdown_signal() -> anyhow::Result<Option<&'static str>> {
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
                return Ok(None);
            }
        }
    }

    #[cfg(windows)]
    {
        loop {
            let request = console::next_request().await?;
            let label = request.label();
            record_exit_trigger(label);
            match stop_action(request, is_draining()) {
                StopAction::Drain => {
                    info!("Received {label} — draining (a second request force-cancels)");
                    drain_begin();
                }
                // The caller's signal task logs it, naming the request returned here.
                StopAction::ForceCancel => return Ok(Some(label)),
            }
        }
    }
}

/// Unix: take the terminal's own signals out of play for the daemon's whole life, from its
/// first instructions — the hangup and every suspending signal it can refuse (Ctrl+Z's
/// `SIGTSTP`, `SIGTTIN`, `SIGTTOU`).
///
/// The disposition is a *caught* handler that does nothing, not `SIG_IGN`: an ignored signal
/// stays ignored across `exec`, a caught one is reset — so the agent shells, the browser, the
/// Telegram helper and the CLI subcommands keep their behaviour. This covers the window before
/// [`wait_for_shutdown_signal`] exists, where a hangup used to end the daemon. `SIGINT`/
/// `SIGTERM` stay default here: a plain kill still stops the service.
///
/// A no-op where the platform has no such signals.
pub fn install_ignored_signals() {
    #[cfg(unix)]
    {
        /// Do nothing: the handler's existence is the point, so it must never allocate or lock.
        extern "C" fn ignore(_signal: libc::c_int) {}

        use std::sync::Once;
        static INSTALLED: Once = Once::new();
        INSTALLED.call_once(|| {
            // SAFETY: `signal` is async-signal-safe and the handler only returns.
            unsafe {
                for kind in [libc::SIGHUP, libc::SIGTSTP, libc::SIGTTIN, libc::SIGTTOU] {
                    libc::signal(kind, ignore as *const () as libc::sighandler_t);
                }
            }
        });
    }
}

/// Before boot and before the interface exists: the two Windows sources below and the quit
/// source. They must be up this early because the platform may deliver a stop the moment the
/// window is up, and the request has to survive until the dashboard subscribes.
pub fn install_stop_request_sources() {
    install_console_stop_handler();
    install_session_end_listener();
    install_quit_requests();
}

/// Called from [`install_stop_request_sources`] before boot: the handler needs no runtime (it
/// queues for the async protocol loop), so a stop request never goes missing on the
/// subscription. A registration failure is reported where there is a console and never fails a
/// launch — the fallback it costs is tokio's Ctrl+C handler, which also fires only where a
/// console exists, so a console-less launch loses nothing.
fn install_console_stop_handler() {
    #[cfg(windows)]
    console::install();
}

/// Called from [`install_stop_request_sources`] before boot, next to
/// [`install_console_stop_handler`]: the platform may end the session at any time, and the
/// listener needs no runtime (it forces the stop synchronously). A failure to bring it up is
/// reported, never fatal: the daemon keeps exactly the stop requests it had.
fn install_session_end_listener() {
    #[cfg(windows)]
    session_end::install();
}

/// Ctrl+C, Ctrl+Break and the console window closing arrive as console control events, each on
/// a **new OS thread the system creates in this process**. That thread has no async context and
/// a closing console must not wait on a task the console might kill first, so the handler queues
/// the request for [`wait_for_shutdown_signal`]. While that loop does not exist (it is spawned
/// only after a successful boot), a request carrying a platform deadline is still queued and one
/// that carries none keeps the platform's default handling (see the dead ends below).
///
/// # The platform's rules, and what they force
///
/// - A close brings a grace counted down from delivery, taken from the user-settable
///   `SPI_GETHUNGAPPTIMEOUT` (5000 ms by default, so it is read and never assumed); the system
///   kills the process when it elapses, and that grace is the only window a close leaves.
/// - The grace is usable only by NOT returning: the contract is "return TRUE and the system
///   terminates the process". So the close handler holds its thread while the async side runs
///   the exit path, and the system's kill is the accepted hard death. Holding the full grace is
///   deliberate: returning sooner only kills the process sooner.
/// - Ctrl+C and Ctrl+Break have no timeout, and returning TRUE leaves the process running, so
///   once the loop exists they queue and return immediately.
/// - Two rapid requests may reach the process as one (the console can collapse them), so an
///   unseen "second request" is not a defect; every event received is queued on its own.
///
/// # Session end is not a console event
///
/// `CTRL_LOGOFF_EVENT` and `CTRL_SHUTDOWN_EVENT` are not delivered to a console application that
/// loads user32/gdi32, as this one does for its dashboard: once user32 is loaded the process is
/// treated as owning a window, and the log-off, shutdown or restart arrives as window messages.
/// This handler declines both, and the daemon hears a session end through its own hidden window.
///
/// # Accepted dead ends (deliberately not fixed)
///
/// - External termination (`TerminateProcess`): no process can intercept it. The task manager's
///   "end task" is a different command, raises the close event, and is graceful here.
/// - Process-tree containment: a stop that ends the daemon takes the command trees it started
///   (a job object's handles go with the process, `tools::shell::tree`), while the launches it
///   detaches on purpose (browser, browser-automation CLI, the self-update replacement) survive.
///   That boundary belongs to those subsystems, not to termination.
/// - Ctrl+C and Ctrl+Break before the loop exists: nothing can act on them, so they keep the
///   platform's default handling, as macOS/Linux does before its signal task registers.
///   Swallowing them would make Ctrl+C a dead key wherever boot never reaches the loop — the
///   start-failure screen answers its own window's close but has no console-event consumer.
/// - Once the loop exists the handler suppresses the default for Ctrl+C and Ctrl+Break, so those
///   would be inert if that loop died while the process lived. Unix is no better: tokio keeps
///   its handler installed for the whole process, so a died-out signal task swallows the signal
///   the same way. The console close and the tray menu's quit remain as escapes.
/// - A kill mid-exit, when a close's grace expires first: what the bound cuts is covered by
///   the durability note on [`urgent_release_budget`], and a kill mid-checkpoint cuts only
///   hygiene — committed work is durable on disk. The store's state is the next start's to sort
///   out through the usual bring-up gate (`db::open_store` shape-checks and repairs).
///
/// # Remaining silent hard deaths
///
/// Those above, plus a stop request arriving with the handler unregistered, and a console close
/// during a start-up that never reached the loop — a console event has no consumer while the
/// loop does not exist. Both need a launch with a console to arise, and there Ctrl+C still
/// reaches tokio's own handler. The dashboard's close is not among them (it is consumed in every
/// state, and hides the window unless nothing is left to keep running or no tray exists), and
/// neither is a session end, which reaches the process through its own window whether or not
/// this handler is registered.
#[cfg(windows)]
mod console {
    use super::{DEFAULT_STOP_GRACE, StopRequest, record_stop_deadline};
    use crate::util::UnwrapPoison;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tokio::sync::Notify;
    use windows_sys::Win32::Foundation::{FALSE, TRUE};
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, CTRL_CLOSE_EVENT, GetConsoleCP, SetConsoleCtrlHandler,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SPI_GETHUNGAPPTIMEOUT, SystemParametersInfoW,
    };

    /// Requests queued by handler threads, drained in order by the protocol loop.
    static REQUESTS: Mutex<VecDeque<StopRequest>> = Mutex::new(VecDeque::new());

    /// Wakes the protocol loop. `notify_one` keeps its permit, so a request queued before the loop
    /// waits is never lost.
    static QUEUED: Notify = Notify::const_new();

    /// Whether the handler is registered. The protocol loop falls back to tokio's Ctrl+C when it
    /// is not, so a failed registration cannot cost a launch that has a console the stop request
    /// it had; a launch without one receives no console event either way, which is why the failure
    /// is not diagnosed there ([`attached_to_console`]). It is registered on every launch
    /// regardless: it costs nothing, and a console-subsystem build has a console to serve.
    static INSTALLED: AtomicBool = AtomicBool::new(false);

    static LOOP_RUNNING: AtomicBool = AtomicBool::new(false);

    /// Without a console the platform delivers no console control event, so a failed registration
    /// has nothing to lose and is not worth a standing diagnostic.
    fn attached_to_console() -> bool {
        // SAFETY: `GetConsoleCP` only reads the calling process's console input
        // code page. The platform documents zero as the failure value, not "no
        // console"; a launch with no console is one of the ways it fails, and the
        // consequence here is only whether a refused registration is worth a
        // standing diagnostic line.
        unsafe { GetConsoleCP() != 0 }
    }

    pub(super) fn install() {
        // SAFETY: `SetConsoleCtrlHandler` only records the function pointer in
        // this process's handler list; `handler` has the required
        // `extern "system" fn(u32) -> i32` signature and lives for the process's lifetime.
        let installed = unsafe { SetConsoleCtrlHandler(Some(handler), TRUE) != 0 };
        INSTALLED.store(installed, Ordering::SeqCst);
        if !installed && attached_to_console() {
            // Through the boot diagnostic rather than `error!`: this runs before
            // boot opens the stores, so it reaches stderr now and the logs store
            // once tracing exists — where an `error!` would be dropped.
            crate::boot::boot_diagnostic(
                "console control handler not registered — Ctrl+C still stops the daemon \
                 through tokio's own handler; a console close and Ctrl+Break will kill \
                 the process without a shutdown."
                    .to_string(),
            );
        }
    }

    /// The console control handler: queues the request for the async protocol loop and, for a
    /// close, holds this thread.
    extern "system" fn handler(event: u32) -> i32 {
        let Some(request) = request_for(event) else {
            // CTRL_LOGOFF_EVENT / CTRL_SHUTDOWN_EVENT: not ours (a session end arrives as a window
            // message), so the default handler's termination stands.
            return FALSE;
        };
        // A close starts the platform's countdown now, so its grace is read here rather than by
        // the async side, and this thread holds until then.
        let close = request == StopRequest::ConsoleClose;
        let deadline = close.then(close_deadline);
        // A close brings a deadline the platform charges for either way, so it is queued even
        // before the loop exists; a request carrying none is left to the platform.
        if !close && !LOOP_RUNNING.load(Ordering::SeqCst) {
            return FALSE;
        }
        // `unwrap_poison` rather than a panic: a panic on a control-handler thread aborts the
        // process instead of shutting it down.
        REQUESTS.lock().unwrap_poison().push_back(request);
        QUEUED.notify_one();
        if let Some(deadline) = deadline {
            // Hold this thread for the grace being counted down (returning is not an option).
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        }
        TRUE
    }

    /// With no handler registered, Ctrl+C through tokio's own handler is the only source left —
    /// kept so a failed registration cannot cost a launch that has a console the request it had
    /// (tokio subscribes through the same `SetConsoleCtrlHandler`). Its failure to subscribe is a
    /// real error the caller reports, except on a launch with no console, which can receive no
    /// console event and waits forever instead.
    pub(super) async fn next_request() -> anyhow::Result<StopRequest> {
        LOOP_RUNNING.store(true, Ordering::SeqCst);
        if !INSTALLED.load(Ordering::SeqCst) {
            if !attached_to_console() {
                // With no console there is nothing to subscribe to: `ctrl_c` would only turn it
                // into a standing "Signal handler failed to set up" error. Such a launch's stop
                // sources are the window and the session-end listener.
                return std::future::pending().await;
            }
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

    /// Map a console event onto the request class it belongs to, `None` for the events this handler
    /// deliberately leaves to the default handler.
    fn request_for(event: u32) -> Option<StopRequest> {
        match event {
            CTRL_C_EVENT => Some(StopRequest::Interrupt),
            CTRL_BREAK_EVENT => Some(StopRequest::Break),
            CTRL_CLOSE_EVENT => Some(StopRequest::ConsoleClose),
            _ => None,
        }
    }

    /// The instant this process's closing console stops granting time: the grace read now (see the
    /// module docs), recorded for the exit path's budget and held by this handler thread until it.
    fn close_deadline() -> Instant {
        let deadline = Instant::now() + close_grace();
        record_stop_deadline(deadline);
        deadline
    }

    /// A stored `0` is taken at face value — no grace, today's hard kill — so the exit path
    /// cannot run past a deadline that has already passed.
    fn close_grace() -> Duration {
        let mut millis: u32 = 0;
        // SAFETY: `SPI_GETHUNGAPPTIMEOUT` writes a single `u32` through `pvparam`;
        // `uiparam` and `fWinIni` are unused for this action.
        let read =
            unsafe { SystemParametersInfoW(SPI_GETHUNGAPPTIMEOUT, 0, (&raw mut millis).cast(), 0) };
        if read == 0 {
            DEFAULT_STOP_GRACE
        } else {
            Duration::from_millis(u64::from(millis))
        }
    }
}

/// The session-end listener — the platform's stop request for a log-off, shutdown or restart,
/// delivered to a window of this daemon's own.
///
/// # Why a window of its own
///
/// Windows announces a session end through a process's top-level windows (`WM_QUERYENDSESSION`,
/// then `WM_ENDSESSION`). Nothing in the windowing stack this daemon draws with handles either
/// message: winit sends both to the default procedure, which answers "yes" and ignores the end,
/// leaving the process to be terminated silently. So the daemon creates its own window — hidden,
/// on its own thread, but a real top-level window, which is what the broadcast reaches. A
/// message-only window (`HWND_MESSAGE`) would never receive it.
///
/// # What it does
///
/// - Answers the preliminary question with "go ahead" and nothing else — [`session_end_label`]
///   says why acting there would be wrong.
/// - On the final notification for a session that really is ending, forces the stop and returns
///   ([`stop_for_session_end`] owns the platform rules).
/// - Hands every other message to `DefWindowProcW`: swallowing one would fail window creation,
///   and answering the question with "no" would refuse the shutdown.
///
/// # The time available
///
/// The platform allows seconds — fewer during a critical shutdown — before ending a process
/// that has not finished, so the stop steps can still be cut off, the exit-time checkpoint most
/// of all. What the bound cuts loses nothing durable (see [`urgent_release_budget`]) and in-flight
/// agent work is cut as on the other platforms; a process ended at the deadline is the same class
/// as today's silent hard death.
///
/// # Accepted, deliberately not fixed
///
/// - A session end before the dashboard is ready: the dashboard subscribes only once boot has
///   succeeded and nothing else consumes the forced stop, so the platform ends the process as
///   today. A boot that finishes inside the grace does exit properly (the token stays fired for
///   the subscription that then appears); one that fails never does.
/// - A session end while a self-update is finalizing: the update's exit path wins, spawn
///   included. The replacement it starts into a session that is ending is cut off with the
///   session like any other in-flight work, and the next start recovers. Deliberately no guard
///   against that spawn.
/// - Bringing the listener up can fail (the window class cannot be registered, the window
///   cannot be created, the thread cannot be spawned). Reported through the boot diagnostics;
///   the daemon runs with the stop requests it always had, and it must never be fatal.
/// - Fast user switching ends no session and sends no notification, so the daemon keeps running.
/// - The stop trace is best-effort, not a durable record: this route forces the stop at once and
///   the exit burst lasts milliseconds, so the listener's line is dropped with the runtime as the
///   exit path's is (see `EXIT_TRIGGER`).
#[cfg(windows)]
mod session_end {
    use super::{session_end_label, stop_for_session_end};
    use std::io::Error;
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DispatchMessageW, ENDSESSION_CLOSEAPP, ENDSESSION_LOGOFF,
        GetMessageW, MSG, RegisterClassExW, WM_ENDSESSION, WM_QUERYENDSESSION, WNDCLASSEXW,
    };
    use windows_sys::w;

    /// The window class this listener registers. Nothing looks it up by name: a class needs one.
    const WINDOW_CLASS: windows_sys::core::PCWSTR = w!("MahBotSessionEnd");

    /// On a thread of its own, never joined: a window's messages go to the thread that created
    /// it, and none of this may run on the dashboard's thread.
    pub(super) fn install() {
        if let Err(e) = std::thread::Builder::new()
            .name("session-end-listener".to_string())
            .spawn(listen)
        {
            report("thread not spawned", &e);
        }
    }

    /// The listener thread: create the window, then pump its messages for the life of the process.
    fn listen() {
        // SAFETY: a null module name asks for this process's own module, which every
        // process has; the window class below needs an instance to own its procedure.
        let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
        // SAFETY: zeroed is the documented "no icon, no cursor, no background, no menu"
        // for the fields set here, and the class is fully described for its `cbSize` by
        // the assignments.
        let mut class: WNDCLASSEXW = unsafe { zeroed() };
        // `cbSize` must report this structure's size: that is how the platform tells which of the
        // two it is handed. `allow`, not `expect` — an unfired expectation is itself a warning.
        #[allow(clippy::cast_possible_truncation)]
        let class_size = size_of::<WNDCLASSEXW>() as u32;
        class.cbSize = class_size;
        class.lpfnWndProc = Some(window_proc);
        class.hInstance = instance;
        class.lpszClassName = WINDOW_CLASS;
        // SAFETY: the class lives for the duration of the call and describes itself
        // completely, which is all the registration reads it for; the procedure it names
        // lives for the life of the process.
        if unsafe { RegisterClassExW(&raw const class) } == 0 {
            report("window class not registered", &Error::last_os_error());
            return;
        }
        // SAFETY: the class is registered to this module, the style is "nothing", the caption
        // and parent are null — which is what makes this a hidden *top-level* window, the kind
        // the session-end broadcast reaches (a message-only parent would exclude it). It is never
        // shown, moved, resized or destroyed.
        let window = unsafe {
            CreateWindowExW(
                0,
                WINDOW_CLASS,
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                instance,
                std::ptr::null(),
            )
        };
        if window == 0 {
            report("window not created", &Error::last_os_error());
            return;
        }
        // SAFETY: zeroed is a valid initial `MSG`; every field is written by the call below.
        let mut message: MSG = unsafe { zeroed() };
        loop {
            // SAFETY: the buffer is `MSG`-sized on this thread's stack and the filter arguments
            // ask for every message of this thread. The retrieval also dispatches messages *sent*
            // from another thread, which is how the platform delivers the two this listener
            // exists for.
            let retrieved = unsafe { GetMessageW(&raw mut message, 0, 0, 0) };
            if retrieved == 0 {
                // `WM_QUIT` — nothing posts one to this thread.
                break;
            }
            if retrieved < 0 {
                report("message loop failed", &Error::last_os_error());
                break;
            }
            // SAFETY: `message` was filled above; the notifications arrive sent, not queued.
            unsafe { DispatchMessageW(&raw const message) };
        }
    }

    /// The listener window's procedure: the two session-end notifications, and the default handling
    /// for everything else (see the module docs for why that is not optional).
    extern "system" fn window_proc(
        window: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        match message {
            // The preliminary question, always answered "go ahead": this daemon never holds up the
            // platform, and the answer is never acted on.
            WM_QUERYENDSESSION => LRESULT::from(true),
            WM_ENDSESSION => {
                // The flags are a 32-bit field in the parameter's low half, which the platform
                // may have sign-extended. `allow`, not `expect` — an unfired expectation warns.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let flags = lparam as u32;
                let ending = wparam != 0;
                let closes_app_only = flags & ENDSESSION_CLOSEAPP != 0;
                let log_off = flags & ENDSESSION_LOGOFF != 0;
                if let Some(label) = session_end_label(ending, closes_app_only, log_off) {
                    stop_for_session_end(label);
                }
                0
            }
            // SAFETY: the procedure delegates with its own arguments untouched, on the
            // thread the platform called it on.
            _ => unsafe { DefWindowProcW(window, message, wparam, lparam) },
        }
    }

    /// Through the boot diagnostic rather than `warn!`, because this may run before boot opens
    /// the stores: stderr now, the logs store once tracing exists.
    fn report(what: &str, detail: &Error) {
        crate::boot::boot_diagnostic(format!(
            "session-end listener: {what}: {detail} — a Windows log-off, shutdown or restart \
             will end the daemon without its stop steps"
        ));
    }
}

/// The wording the macOS quit route records for itself in [`EXIT_TRIGGER`]: every ordinary quit
/// ask — ⌘Q, the menu's Quit, the Dock's Quit — arrives as the one AppleEvent `macos_quit`
/// takes over, so they share one label.
pub(crate) const MACOS_QUIT_TRIGGER: &str = "macOS quit";

/// The tray menu's one item records its own ask, which on either platform that has a tray is
/// neither the platform's quit nor a close.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) const TRAY_QUIT_TRIGGER: &str = "tray quit";

/// The quit channel, and on macOS the interception of the platform's own asks. Called from
/// [`install_stop_request_sources`] before the interface starts. The channel exists on every
/// platform so the dashboard's subscription always has a source, but only macOS sends on it.
fn install_quit_requests() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    // The receiving half first, so a quit the platform delivers before the dashboard subscribes
    // waits here instead of being lost.
    *QUIT_REQUESTS.lock().unwrap_poison() = Some(rx);
    // The channel exists on every platform so the dashboard's subscription always has a source;
    // only macOS sends, so everywhere else the sender is dropped here and the source stays empty.
    #[cfg(target_os = "macos")]
    macos_quit::install(tx);
    #[cfg(not(target_os = "macos"))]
    drop(tx);
}

/// The dashboard's subscription takes it once, and a quit arriving before anything subscribes
/// waits here instead of being lost — the platform asks whether or not the interface is ready.
static QUIT_REQUESTS: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<()>>> = Mutex::new(None);

/// `None` before [`install_quit_requests`] ran, or once taken: iced spawns a
/// `Subscription::run` recipe — identified by its function pointer — once per process.
pub(crate) fn take_quit_requests() -> Option<tokio::sync::mpsc::UnboundedReceiver<()>> {
    QUIT_REQUESTS.lock().unwrap_poison().take()
}

/// The platform's ordinary quit asks, brought into the closing path, and the Dock click that
/// brings a hidden window back.
///
/// # Why it exists
///
/// macOS ends an application through its own machinery: an ordinary quit ask (the ones
/// [`crate::gui::Message::QuitRequested`] receives) reaches AppKit's `terminate:`, which asks the
/// application delegate whether it may terminate and then terminates the process itself. winit
/// installs that delegate but implements no `applicationShouldTerminate:`, so AppKit's default
/// answer stands: the asks never reach the closing path, the window state is never recorded, the
/// checkpoint never runs, the browser release is deferred and in-flight work is cut off hard.
///
/// # The hook
///
/// The method is added to winit's delegate class at run time (`class_addMethod`), which leaves
/// winit's delegate object untouched — it has to, because winit panics unless the delegate is the
/// one it registered. The class does not exist before the event loop is built, so a thread of its
/// own waits for it and gives up after `INSTALL_WAIT`.
///
/// # The Dock's reopen
///
/// The same class carries `applicationShouldHandleReopen:hasVisibleWindows:`, which hands a show
/// request to the dashboard (see [`crate::window_requests`]) and answers `YES`, leaving the
/// platform's own unhiding to run. The same waiter installs it — before boot finishes, so the
/// Dock works from the moment the window exists, the launch the updater restarted hidden included.
///
/// # Which ask it is
///
/// `kAEQuitReason` on the Quit AppleEvent names why the quit was sent; an ordinary quit carries no
/// reason. The delegate's `sender` cannot tell them apart — for every AppleEvent-driven route
/// it is the application itself, the Dock's Quit included. The asks are answered differently:
///
/// - An ordinary quit is answered `NSTerminateCancel`: termination is refused and the closing path
///   takes over, as with the tray menu's quit (winit answers `windowShouldClose:` with "no" for the
///   same reason). A **second** quit during the graceful drain reaches this hook again and forces
///   that drain, except while a self-update's drain owns the exit.
/// - A session end (log-out, restart, shut-down) is not the product's: answered
///   `NSTerminateNow`, what an application without this method gets, and the platform ends the
///   process at once — those losses are on the list above.
///
/// # The bound
///
/// An ordinary quit is refused at the platform level, so the product's own bound must end the
/// process or a wedged closing path would leave it running. It cannot live in the machinery the
/// closing path tears down, so `arm_exit_bound` runs it on a thread outliving the interface,
/// armed as the quit is taken over: `EXIT_BOUND`, the drain cap plus an allowance for the
/// closing steps after it.
///
/// # Accepted, deliberately not fixed
///
/// The platform-forced termination and the session end are on the `shutdown` module's list. What
/// this module accepts on its own terms:
///
/// - A quit with nothing left to act on it (the dashboard's subscription gone, the interface
///   past its exit) is handed back (`NSTerminateNow`), so a quit is never a dead key and the
///   platform never waits on an answer nothing can give.
/// - A failure to install (the delegate class never appears, a future winit renames it, the
///   class already implements the method, the thread cannot be spawned): an ordinary quit keeps
///   today's behaviour — the platform ends the process — reported, never fatal. The same
///   failure leaves the Dock's reopen uninstalled, which costs that way back and nothing else:
///   the tray and a second launch still bring the window up.
/// - The exit bound firing: a hard end with no closing steps, the same class as a force quit. A
///   quit arriving while a self-update is finalizing is armed like any other, so the bound can cut
///   that hand-off where the exit path would otherwise wait for it: a tail outrunning the
///   allowance looks no different from a wedged one.
#[cfg(target_os = "macos")]
mod macos_quit {
    use objc2::ffi::class_addMethod;
    use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
    use objc2::sel;
    use objc2_app_kit::NSApplicationTerminateReply;
    use objc2_core_services::{
        kAEQuitAll, kAEQuitReason, kAEReallyLogOut, kAERestart, kAEShutDown,
    };
    use objc2_foundation::NSAppleEventManager;
    use std::ffi::CStr;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tracing::{info, warn};

    /// The sending half of the quit channel; this module is its only producer, and a quit's value
    /// says nothing — every one arrives by the same route.
    static QUIT_TX: OnceLock<tokio::sync::mpsc::UnboundedSender<()>> = OnceLock::new();

    /// `false` when nothing will consume it (the subscription is gone), so the caller falls back
    /// on the platform ending the process.
    fn queue_quit() -> bool {
        QUIT_TX.get().is_some_and(|tx| tx.send(()).is_ok())
    }

    /// The class winit registers for its event loop, and the one AppKit asks about a termination
    /// and a Dock reopen; a rename in a future winit costs the interception and nothing else.
    const DELEGATE_CLASS: &CStr = c"WinitApplicationDelegate";

    /// `applicationShouldTerminate:`'s encoding: reply (`NSUInteger`), receiver, selector, sender.
    const SHOULD_TERMINATE_TYPES: &CStr = c"Q@:@";

    /// `applicationShouldHandleReopen:hasVisibleWindows:`'s encoding: `BOOL`, receiver, selector,
    /// application, visible-windows flag.
    const SHOULD_REOPEN_TYPES: &CStr = c"c@:@c";

    /// The wait for winit's delegate class, polled every [`INSTALL_POLL`].
    const INSTALL_WAIT: Duration = Duration::from_secs(60);
    const INSTALL_POLL: Duration = Duration::from_millis(5);

    /// The last-resort bound on a quit's exit: the drain cap plus an allowance for the closing
    /// steps that follow it.
    const EXIT_BOUND: Duration = Duration::from_secs(crate::jobs::DRAIN_CAP_SECS + 300);

    /// On a thread of its own, never joined: a failure is reported and never fatal — an ordinary
    /// quit then keeps today's behaviour.
    pub(super) fn install(tx: tokio::sync::mpsc::UnboundedSender<()>) {
        let _ = QUIT_TX.set(tx);
        if let Err(e) = std::thread::Builder::new()
            .name("quit-interception".to_string())
            .spawn(install_when_registered)
        {
            report(&format!("thread not spawned: {e}"));
        }
    }

    fn install_when_registered() {
        let give_up = Instant::now() + INSTALL_WAIT;
        loop {
            if let Some(class) = AnyClass::get(DELEGATE_CLASS) {
                // SAFETY: the IMP's signature is the method's own (`self`, `_cmd`, `sender`,
                // returning the reply type, an `NSUInteger`), and the encoding string spells it
                // out; the class outlives the process, so the method stays valid for every
                // delegate instance.
                let added = unsafe {
                    class_addMethod(
                        std::ptr::from_ref(class).cast_mut(),
                        sel!(applicationShouldTerminate:),
                        std::mem::transmute::<
                            unsafe extern "C-unwind" fn(
                                *mut AnyObject,
                                Sel,
                                *mut AnyObject,
                            )
                                -> NSApplicationTerminateReply,
                            Imp,
                        >(should_terminate),
                        SHOULD_TERMINATE_TYPES.as_ptr(),
                    )
                };
                if !added.as_bool() {
                    report(
                        "winit's delegate already implements applicationShouldTerminate: — \
                         the platform's own termination stands",
                    );
                }
                // SAFETY: as above — the IMP is the method's own signature (`self`, `_cmd`,
                // the application and the flag, answering a `BOOL`), the encoding spells it out,
                // and the class outlives the process.
                let reopened = unsafe {
                    class_addMethod(
                        std::ptr::from_ref(class).cast_mut(),
                        sel!(applicationShouldHandleReopen:hasVisibleWindows:),
                        std::mem::transmute::<
                            unsafe extern "C-unwind" fn(
                                *mut AnyObject,
                                Sel,
                                *mut AnyObject,
                                i8,
                            ) -> i8,
                            Imp,
                        >(should_reopen),
                        SHOULD_REOPEN_TYPES.as_ptr(),
                    )
                };
                if !reopened.as_bool() {
                    report(
                        "winit's delegate already answers a Dock reopen — the tray and a second \
                         launch still bring the window up",
                    );
                }
                // Success is silent, as the Windows installers are: the class appears inside
                // `run`, before boot opens the stores, so nothing would carry a line — and a
                // takeover is not reported to the owner either. What it produces is the closing
                // path itself.
                return;
            }
            if Instant::now() >= give_up {
                report(
                    "winit's delegate class was not registered in time — the platform's own \
                     termination stands",
                );
                return;
            }
            std::thread::sleep(INSTALL_POLL);
        }
    }

    /// An ordinary quit is handed to the closing path, a session end left to the platform.
    ///
    /// # Safety
    ///
    /// AppKit calls this as `-[WinitApplicationDelegate applicationShouldTerminate:]` on the
    /// main thread: `_this` is the delegate, `_sender` the application object (unused — the ask
    /// is told apart by the AppleEvent the system is handling, not by the sender).
    unsafe extern "C-unwind" fn should_terminate(
        _this: *mut AnyObject,
        _cmd: Sel,
        _sender: *mut AnyObject,
    ) -> NSApplicationTerminateReply {
        if is_session_end() {
            return NSApplicationTerminateReply::TerminateNow;
        }
        if !queue_quit() {
            warn!("the dashboard is past its exit — the platform ends the process");
            return NSApplicationTerminateReply::TerminateNow;
        }
        info!("macOS quit — the platform asked the product to quit");
        arm_exit_bound();
        NSApplicationTerminateReply::TerminateCancel
    }

    /// The delegate method AppKit asks about a reopen — the Dock icon clicked while the
    /// application runs: the dashboard brings the window up, and answering `YES` leaves the
    /// platform's own reopen (unhiding the application after ⌘H) to run.
    ///
    /// # Safety
    ///
    /// AppKit calls this as `-[WinitApplicationDelegate
    /// applicationShouldHandleReopen:hasVisibleWindows:]` on the main thread: `_this` is the
    /// delegate, `_sender` the application object and `_has_visible_windows` the flag, none of
    /// which this needs.
    unsafe extern "C-unwind" fn should_reopen(
        _this: *mut AnyObject,
        _cmd: Sel,
        _sender: *mut AnyObject,
        _has_visible_windows: i8,
    ) -> i8 {
        let _ = crate::window_requests::show();
        1
    }

    /// Whether the platform is ending the session rather than asking the product to quit: its
    /// Quit AppleEvent names a session end with `kAEQuitReason`, and an ordinary quit carries no
    /// reason at all.
    ///
    /// A reason this build does not know — and a session end whose reason the system omits —
    /// is read as an ordinary quit. The asymmetry is deliberate: a session end read as an ordinary
    /// one only starts a closing path the platform's own deadline then cuts, while an ordinary
    /// quit read as a session end would leave a quit the user is watching to the platform — the
    /// behaviour this module exists to replace. It does mean a misread session end cancels the
    /// platform's termination, and that the classification rests on Apple's documented reason
    /// codes alone.
    fn is_session_end() -> bool {
        /// The reason codes read as a session end: the documented values of `kAEQuitReason`
        /// (`AERegistry.h`) — shutdown, restart, log-out, and the system's quit-all, the last of
        /// which is read as a session end here rather than as an ordinary quit.
        const SESSION_ENDS: [u32; 4] = [kAEShutDown, kAERestart, kAEReallyLogOut, kAEQuitAll];
        NSAppleEventManager::sharedAppleEventManager()
            .currentAppleEvent()
            .and_then(|event| event.attributeDescriptorForKeyword(kAEQuitReason))
            .is_some_and(|reason| SESSION_ENDS.contains(&reason.typeCodeValue()))
    }

    /// Arm [`EXIT_BOUND`] on the first quit the product takes over.
    fn arm_exit_bound() {
        static ARMED: AtomicBool = AtomicBool::new(false);
        if ARMED.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Err(e) = std::thread::Builder::new()
            .name("quit-exit-bound".to_string())
            .spawn(|| {
                std::thread::sleep(EXIT_BOUND);
                report(&format!(
                    "the closing path did not finish within {EXIT_BOUND:?} of a quit — ending \
                     the process"
                ));
                // SAFETY: `_exit` ends the process at once, from any thread. Nothing is
                // waited for: this is the last resort, the same class of end as a force quit.
                unsafe { libc::_exit(1) }
            })
        {
            report(&format!("exit bound not armed: {e}"));
        }
    }

    /// Report a failure of the delegate hooks, or the exit bound firing, through the boot
    /// diagnostic, as the session-end listener's `report` does: its stderr leg is the one that
    /// carries it (an install failure can run before the logs store exists, and the exit-bound
    /// report is followed at once by the `_exit`).
    fn report(what: &str) {
        crate::boot::boot_diagnostic(format!("macOS delegate hooks: {what}"));
    }
}

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

thread_local! {
    /// Set while a contained pass ([`contain_panics`]) runs on this thread.
    static CONTAINED_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f`, reporting a panic it raises nowhere: the boot hook skips its report
/// for a panic contained here, and the panic is caught instead ([`Err`] carries
/// its payload).
///
/// For a caller that contains *many* panics by design — the per-page PDF text
/// pass, where one malformed page panics — the default hook would otherwise
/// turn one failed document into one report per failed page on stderr.
pub(crate) fn contain_panics<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    let previous = CONTAINED_PANIC.with(|flag| flag.replace(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    CONTAINED_PANIC.with(|flag| flag.set(previous));
    result
}

/// Install a global panic hook that prints a timestamped marker line before
/// the default panic report, so panics captured in `daemon.log` (a detached launch's own
/// output) or `update.log` (the replacement daemon's stderr) are time-attributable. The
/// default hook (message + backtrace) still runs — except for a panic contained by
/// [`contain_panics`], which is reported through the caller's own result instead.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if CONTAINED_PANIC.with(std::cell::Cell::get) {
            return;
        }
        crate::boot::timestamped_stderr(&info.to_string());
        default_hook(info);
    }));
}

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
    fn only_the_final_session_end_notification_stops_the_daemon() {
        // The final notification for a session that really is ending — a log-off, or a
        // shutdown/restart (a critical shutdown carries neither flag).
        assert_eq!(
            session_end_label(true, false, true),
            Some("Windows log-off")
        );
        assert_eq!(
            session_end_label(true, false, false),
            Some("Windows shutdown/restart")
        );
        // The question's "not ending" answer: a shutdown under way was cancelled, so a
        // stop here would leave the daemon half-stopped for a session that continues.
        assert_eq!(session_end_label(false, false, false), None);
        // `ENDSESSION_CLOSEAPP`: the platform asking this application alone to close so
        // that an update can proceed, which ends no session.
        assert_eq!(session_end_label(true, true, false), None);
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

    #[test]
    fn a_contained_panic_is_not_reported() {
        /// Only the marker this test raises is counted: the hook stays
        /// transparent for every other panic, so a test running in parallel keeps
        /// its ordinary report.
        const MARKER: &str = "mahbot: contained-panic test marker";
        static REPORTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync>;

        /// Puts the process hook back even when an assertion fails: a counting
        /// hook left installed would drop every later panic report.
        struct RestoreHook(Option<std::sync::Arc<PanicHook>>);

        impl Drop for RestoreHook {
            fn drop(&mut self) {
                if let Some(hook) = self.0.take() {
                    std::panic::set_hook(Box::new(move |info| hook.as_ref()(info)));
                }
            }
        }

        let previous = std::sync::Arc::new(std::panic::take_hook());
        let _restore = RestoreHook(Some(std::sync::Arc::clone(&previous)));
        std::panic::set_hook(Box::new(move |info| {
            // A formatted panic carries a `String` payload; a literal one a `&str`.
            let payload = info.payload();
            let text = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
            if text == Some(MARKER) {
                REPORTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                previous.as_ref()(info);
            }
        }));
        install_panic_hook();
        assert!(
            contain_panics(|| panic!("{MARKER}")).is_err(),
            "the panic is still returned to the caller"
        );
        assert_eq!(
            REPORTED.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "a contained panic must not reach the hook: one document conversion \
             cannot report once per failing page"
        );
        let _ = std::panic::catch_unwind(|| panic!("{MARKER}"));
        assert_eq!(
            REPORTED.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "every other panic is reported as before"
        );
    }
}
