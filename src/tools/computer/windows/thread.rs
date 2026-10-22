//! The dedicated multithreaded-apartment thread every accessibility and capture
//! call runs on, and the deadline every caller awaits it under.
//!
//! The thread is created lazily and rebuilt after a hang or a caught structured
//! exception. A job carries a plain-data [`Call`] and a one-shot reply channel; the
//! COM objects the thread initializes (`IUIAutomation`, the D3D device) never leave
//! it, which is why the caller side of this module needs no `unsafe` and no `Send`
//! wrappers.
//!
//! An abandoned job is an accepted limit: the action lock cannot outlive the
//! deadline that keeps the product responsive, so a job stuck in a provider may
//! still finish on the leaked thread after its caller was told it timed out. That
//! is why the timeout message reports that the call produced no result and says to
//! check the target's state before repeating it, rather than to retry blindly.

use super::super::core::{
    self, AppInfo, Capture, ElementAct, Locator, Observation, RawInput, SurfaceGeometry,
    TargetSpec, WindowInfo,
};
use super::capture::CaptureContext;
use super::{input, seh, uia, window};
use crate::util::UnwrapPoison;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance};
use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize};
use windows::Win32::UI::Accessibility::{CUIAutomation8, IUIAutomation, IUIAutomation2};
use windows::core::Interface as _;

/// Client-side bound on one provider connection (a UIA call into a window whose
/// application is busy). The platform's default is 2 s; 3 s leaves room for a
/// cold provider without letting a hung application stretch the whole job.
const CONNECTION_TIMEOUT_MS: u32 = 3_000;
/// Client-side bound on one transaction, which can span several provider calls.
const TRANSACTION_TIMEOUT_MS: u32 = 10_000;

// ── Calls and replies ───────────────────────────────────────────────────

/// One unit of work for the apartment thread. Plain data only: no COM handle is
/// ever sent or shared across a thread boundary.
pub(super) enum Call {
    ProbeCapture,
    ListApps,
    ListWindows {
        app: Option<AppInfo>,
    },
    FocusedWindow,
    SurfaceGeometry {
        target: TargetSpec,
    },
    Observe {
        target: TargetSpec,
    },
    Act {
        target: TargetSpec,
        locator: Locator,
        act: ElementAct,
    },
    RawInput {
        target: TargetSpec,
        input: RawInput,
    },
    CursorPosition,
    Capture {
        target: TargetSpec,
    },
}

/// The answer to one [`Call`]. `pub(super)` only because [`FromReply`] names it in its
/// method; nothing outside this module ever constructs or matches one. Boxed payloads
/// keep the enum small enough to move cheaply across the channel.
pub(super) enum Reply {
    Flag(bool),
    Apps(Vec<AppInfo>),
    Windows(Vec<WindowInfo>),
    Window(Box<WindowInfo>),
    Geometry(SurfaceGeometry),
    Observation(Box<Observation>),
    Unit,
    Point(f64, f64),
    Capture(Box<Capture>),
}

/// The payload one [`Call`] shape answers with. The pairing is stated exactly once,
/// below; a wrong shape is an internal bug and is reported as such rather than
/// panicking inside a tool call.
pub(super) trait FromReply: Sized {
    fn from_reply(reply: Reply) -> anyhow::Result<Self>;
}

macro_rules! from_reply {
    ($($ty:ty: $pattern:pat => $value:expr),* $(,)?) => {$(
        impl FromReply for $ty {
            fn from_reply(reply: Reply) -> anyhow::Result<Self> {
                match reply {
                    $pattern => Ok($value),
                    _ => Err(reply_mismatch()),
                }
            }
        }
    )*};
}

from_reply! {
    bool: Reply::Flag(value) => value,
    Vec<AppInfo>: Reply::Apps(value) => value,
    Vec<WindowInfo>: Reply::Windows(value) => value,
    WindowInfo: Reply::Window(value) => *value,
    SurfaceGeometry: Reply::Geometry(value) => value,
    Observation: Reply::Observation(value) => *value,
    (): Reply::Unit => (),
    (f64, f64): Reply::Point(x, y) => (x, y),
    Capture: Reply::Capture(value) => *value,
}

fn reply_mismatch() -> anyhow::Error {
    // `unsupported` rather than `degraded`: the prompt tells the model to retry a
    // `degraded` failure, and retrying cannot fix a wrong pairing inside this backend.
    core::taxonomy_error(
        core::ERR_UNSUPPORTED,
        "internal computer-use reply mismatch — a bug in this backend, not a retryable condition; \
         report it",
    )
}

struct Job {
    call: Call,
    reply: tokio::sync::oneshot::Sender<anyhow::Result<Reply>>,
}

// ── The thread ──────────────────────────────────────────────────────────

/// The live apartment thread. `alive` is cleared by a caller whose deadline
/// expired (or whose reply channel died), and by the thread itself after a caught
/// structured exception: the next submission then starts a fresh thread instead of
/// queueing behind a provider that may never return.
struct Worker {
    tx: Sender<Job>,
    alive: Arc<AtomicBool>,
}

static WORKER: LazyLock<Mutex<Option<Worker>>> = LazyLock::new(|| Mutex::new(None));

/// Submit one call and await the payload it answers with under `timeout` — the only
/// entry point from the async side, so no call can reach a platform API from the
/// caller's thread.
pub(super) async fn submit<T: FromReply>(call: Call, timeout: Duration) -> anyhow::Result<T> {
    let (tx, alive) = sender()?;
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    if tx
        .send(Job {
            call,
            reply: reply_tx,
        })
        .is_err()
    {
        alive.store(false, Ordering::SeqCst);
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "the computer-use thread stopped before it accepted the call — retry",
        ));
    }
    match tokio::time::timeout(timeout, reply_rx).await {
        Ok(Ok(result)) => result.and_then(T::from_reply),
        Ok(Err(_)) => {
            alive.store(false, Ordering::SeqCst);
            Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                "the computer-use thread stopped while running the call — retry",
            ))
        }
        Err(_) => {
            // Abandon it: the thread may be blocked inside a provider and can be
            // neither interrupted nor killed. Marking it dead makes the next call
            // start a fresh apartment immediately; this one exits on its own once
            // the provider answers.
            alive.store(false, Ordering::SeqCst);
            Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                format!(
                    "the application did not respond within {}s — the call was abandoned without a \
                     result; check the target's state (observe or screenshot) before repeating it, \
                     since an action that did land would apply twice if repeated",
                    timeout.as_secs()
                ),
            ))
        }
    }
}

/// The live thread's sender, starting a fresh thread when none is alive.
fn sender() -> anyhow::Result<(Sender<Job>, Arc<AtomicBool>)> {
    let mut guard = WORKER.lock().unwrap_poison();
    if let Some(worker) = guard.as_ref()
        && worker.alive.load(Ordering::SeqCst)
    {
        return Ok((worker.tx.clone(), Arc::clone(&worker.alive)));
    }
    let alive = Arc::new(AtomicBool::new(true));
    let (tx, rx) = std::sync::mpsc::channel();
    let thread_alive = Arc::clone(&alive);
    std::thread::Builder::new()
        .name("mahbot-computer".to_string())
        .spawn(move || run(&rx, &thread_alive))
        .map_err(|e| {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                format!("failed to start the computer-use thread: {e}"),
            )
        })?;
    *guard = Some(Worker {
        tx: tx.clone(),
        alive: Arc::clone(&alive),
    });
    Ok((tx, alive))
}

/// The thread body: initialize the apartment once, then serve jobs until the
/// channel closes or the thread has been abandoned.
fn run(rx: &Receiver<Job>, alive: &AtomicBool) {
    let mut ctx = match seh::guard(Ctx::open) {
        Ok(Ok(ctx)) => ctx,
        Ok(Err(e)) => return retire(rx, alive, &format!("{e:#}")),
        Err(code) => {
            // The apartment's own setup faulted. Nothing it built is trustworthy,
            // and the half-initialized COM state is abandoned with the thread.
            let error = framework_failed("open the computer-use apartment", code);
            let message = format!("{error:#}");
            tracing::warn!("computer use: {message}");
            return retire(rx, alive, &message);
        }
    };
    while let Ok(job) = rx.recv() {
        if !alive.load(Ordering::SeqCst) {
            break;
        }
        match seh::guard(|| ctx.execute(job.call)) {
            Ok(result) => {
                // A caller that gave up has dropped its receiver; that is expected.
                let _ = job.reply.send(result);
            }
            Err(code) => {
                let error = framework_failed("run the computer-use call", code);
                let message = format!("{error:#}");
                tracing::warn!("computer use: {message}; retiring the thread");
                let _ = job.reply.send(Err(error));
                // Released, this device would run `Release` on a state nobody can
                // vouch for, outside any exception frame — at most one leaked device
                // per caught fault is the cheaper consequence.
                std::mem::forget(ctx);
                // Nothing here can be resumed, so the thread serves no other job: this
                // caller has the fault, everything already queued gets it too, and the
                // next call starts a fresh apartment.
                return retire(rx, alive, &message);
            }
        }
    }
    // Closing the apartment runs `Release` on native objects, which is foreign code
    // like any other: the drop is guarded too, and a fault there only ends this
    // thread, which is ending anyway.
    if let Err(code) = seh::guard(move || drop(ctx)) {
        let error = framework_failed("close the computer-use apartment", code);
        tracing::warn!("computer use: {error:#}");
    }
}

/// Leave this thread dead and fail everything already queued with `message`, so the
/// next call starts a fresh apartment instead of queueing behind this one.
fn retire(rx: &Receiver<Job>, alive: &AtomicBool, message: &str) {
    alive.store(false, Ordering::SeqCst);
    for job in rx {
        let _ = job.reply.send(Err(anyhow::anyhow!(message.to_owned())));
    }
}

/// The fault a caught structured exception becomes, named by what was running.
fn framework_failed(what: &str, code: u32) -> anyhow::Error {
    core::framework_failed(
        what,
        format!("the system raised structured exception 0x{code:08X}"),
    )
}

/// Everything the apartment thread owns: the UIA client and the capture device.
struct Ctx {
    uia: IUIAutomation,
    capture: CaptureContext,
}

impl Ctx {
    fn open() -> anyhow::Result<Self> {
        // A fresh thread has no apartment yet, so this cannot conflict. WinRT
        // activation (the capture facility) needs the same initialization.
        // SAFETY: `sender` spawns a thread that runs this once, so the apartment is
        // uninitialized here; every COM object built below stays on this thread.
        unsafe { RoInitialize(RO_INIT_MULTITHREADED) }.map_err(|e| {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                format!("failed to initialize the computer-use apartment: {e}"),
            )
        })?;
        window::ensure_dpi_aware();
        // SAFETY: the apartment was initialized on the line above, the class is
        // in-process, and the client is owned by `Self`, which never leaves this thread.
        let uia: IUIAutomation = unsafe {
            CoCreateInstance(
                &CUIAutomation8,
                None::<&windows::core::IUnknown>,
                CLSCTX_INPROC_SERVER,
            )
        }
        .map_err(|e| {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                format!("UI Automation is unavailable: {e}"),
            )
        })?;
        // Bound every provider call at the client. `CUIAutomation8` always
        // implements `IUIAutomation2` on the product's Windows floor; a failure
        // here only means the platform defaults (2 s/20 s) stay in force.
        if let Ok(uia2) = uia.cast::<IUIAutomation2>() {
            // SAFETY: `uia2` is a live interface cast from the client above, and the
            // setters only store two timeouts on it, on the thread that owns it.
            unsafe {
                let _ = uia2.SetConnectionTimeout(CONNECTION_TIMEOUT_MS);
                let _ = uia2.SetTransactionTimeout(TRANSACTION_TIMEOUT_MS);
            }
        }
        Ok(Self {
            uia,
            capture: CaptureContext::default(),
        })
    }

    fn execute(&mut self, call: Call) -> anyhow::Result<Reply> {
        match call {
            Call::ProbeCapture => Ok(Reply::Flag(window::interactive_desktop_available())),
            Call::ListApps => Ok(Reply::Apps(window::enumerate_apps())),
            Call::ListWindows { app } => Ok(Reply::Windows(window::list_windows(app.as_ref()))),
            Call::FocusedWindow => Ok(Reply::Window(Box::new(window::focused_window()?))),
            Call::SurfaceGeometry { target } => {
                Ok(Reply::Geometry(window::surface_geometry(&target)?))
            }
            Call::Observe { target } => Ok(Reply::Observation(Box::new(uia::observe(
                &self.uia, &target,
            )?))),
            Call::Act {
                target,
                locator,
                act,
            } => {
                uia::act_on_element(&self.uia, &target, &locator, &act)?;
                Ok(Reply::Unit)
            }
            Call::RawInput { target, input } => {
                input::raw_input(&target, input)?;
                Ok(Reply::Unit)
            }
            Call::CursorPosition => {
                let (x, y) = input::cursor_position()?;
                Ok(Reply::Point(x, y))
            }
            Call::Capture { target } => {
                Ok(Reply::Capture(Box::new(self.capture.capture(&target)?)))
            }
        }
    }
}
