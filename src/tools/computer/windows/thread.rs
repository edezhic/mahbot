//! The dedicated multithreaded-apartment thread every accessibility and capture
//! call runs on, and the deadline every caller awaits it under.
//!
//! The thread is created lazily and rebuilt after a hang or a caught structured
//! exception. A job carries a plain-data [`Call`] and a one-shot reply channel; the
//! COM objects the thread initializes (`IUIAutomation`, the D3D device) never leave
//! it, which is why the caller side of this module needs no `unsafe` and no `Send`
//! wrappers.
//!
//! An abandoned job that had already started is an accepted limit: the action lock
//! cannot outlive the deadline that keeps the product responsive, so a job stuck in a
//! provider may still finish on the leaked thread after its caller was told it timed
//! out. A timed-out actuation is therefore answered as [`Impact::Unconfirmed`], never as
//! a failure; a timed-out read has nothing in flight, so its deadline stays an error,
//! and a job whose thread stopped mid-call answers the same unconfirmed way as a
//! deadline for the same reason. A job whose deadline expired while it was still queued
//! behind another call never runs at all: it is marked abandoned by the same swap the thread
//! uses to take it, so nothing of it is in flight, nothing was delivered and no thread is
//! retired. Both entry points answer it as the plain `degraded` failure it is (see
//! [`JobFailure::NotStarted`]) — the shared thread is still the healthy one, and only a
//! genuine hang is a reason to leak it. Each thread a hang leaves behind
//! is counted (see [`LEAKED_THREADS`]) and the count is reported in the deadline's answer
//! — the one failure that produced it — so a run that keeps hanging shows what it costs
//! instead of leaking silently.

use super::super::core::{
    self, Actuation, AppInfo, AppListing, Capture, CaptureBudget, ElementAct, FocusedElement,
    Impact, Locator, Observation, RawInput, SurfaceGeometry, TargetSpec, WindowInfo, WindowListing,
};
use super::capture::CaptureContext;
use super::{input, seh, uia, window};
use crate::util::UnwrapPoison;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
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
    ListApps,
    FocusedElement {
        target: Option<TargetSpec>,
    },
    ListWindows {
        app: Option<AppInfo>,
    },
    SurfaceWindow {
        target: TargetSpec,
    },
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
        /// The call's one pixel allowance, carried across the thread boundary so the
        /// buffers the apartment thread builds are charged to the caller's call.
        budget: CaptureBudget,
    },
}

impl Call {
    /// What this call is addressed to, in the words an error about it should use.
    /// Read before the call moves onto the thread: a deadline that names nothing tells
    /// the model less than the failure it reports.
    fn label(&self) -> String {
        match self {
            Call::ListApps => "the running applications".to_string(),
            Call::ListWindows { app } => match app {
                Some(app) => format!("the windows of {}", app.name),
                None => "the open windows".to_string(),
            },
            Call::FocusedElement { target } => match target {
                Some(target) => format!("the focus on {}", target.describe()),
                None => "the focused element".to_string(),
            },
            Call::SurfaceWindow { target }
            | Call::SurfaceGeometry { target }
            | Call::Observe { target }
            | Call::Act { target, .. }
            | Call::RawInput { target, .. }
            | Call::Capture { target, .. } => target.describe(),
            Call::CursorPosition => "the pointer".to_string(),
        }
    }
}

/// The answer to one [`Call`]. `pub(super)` only because [`FromReply`] names it in its
/// method; nothing outside this module ever constructs or matches one. Boxed payloads
/// keep the enum small enough to move cheaply across the channel.
pub(super) enum Reply {
    Apps(AppListing),
    Focus(Option<FocusedElement>),
    Windows(WindowListing),
    Window(Box<WindowInfo>),
    Geometry(SurfaceGeometry),
    Observation(Box<Observation>),
    /// What an actuating call delivered — and, when it put the pointer somewhere, where.
    /// An element action answers [`Actuation::of`], because it places no point of its own.
    Input(Box<Actuation>),
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
    AppListing: Reply::Apps(value) => value,
    Option<FocusedElement>: Reply::Focus(value) => value,
    WindowListing: Reply::Windows(value) => value,
    WindowInfo: Reply::Window(value) => *value,
    SurfaceGeometry: Reply::Geometry(value) => value,
    Observation: Reply::Observation(value) => *value,
    Actuation: Reply::Input(value) => *value,
    (f64, f64): Reply::Point(x, y) => (x, y),
    Capture: Reply::Capture(value) => *value,
}

fn reply_mismatch() -> anyhow::Error {
    // `unsupported` rather than `degraded`: the degraded failures name a retry, and
    // retrying cannot fix a wrong pairing inside this backend.
    core::taxonomy_error(
        core::ERR_UNSUPPORTED,
        "internal computer-use reply mismatch — a bug in this backend, not a retryable condition; \
         report it",
    )
}

/// What a submitted job is doing, as one value both sides agree on: the caller that gave up
/// and the thread that was about to run it settle it with one compare-and-swap each, so a
/// job is never executed after its caller was told nothing was delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum JobState {
    /// Waiting in the channel.
    Queued = 0,
    /// Taken by the thread, which is running it now and may still complete.
    Running = 1,
    /// Its caller's deadline expired while it was still queued: it must not run at all.
    Abandoned = 2,
}

impl JobState {
    /// Move `state` from `from` to `to`, answering whether this side won.
    fn settle(state: &AtomicU8, from: Self, to: Self) -> bool {
        state
            .compare_exchange(from as u8, to as u8, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

struct Job {
    call: Call,
    reply: tokio::sync::oneshot::Sender<anyhow::Result<Reply>>,
    /// The handshake above: a job the thread has begun may still complete, while one still
    /// queued when its caller gave up never runs.
    state: Arc<AtomicU8>,
    /// Where an actuating job put its point, published by the job as it maps it — before it
    /// posts anything — so a call that never answers still reports the position it used.
    /// Only [`submit_act`] fills this in.
    placed: Option<Arc<PlacementMemo>>,
}

/// Where an actuating call placed its point, written by the thread as the call maps it.
pub(super) type PlacementMemo = Mutex<Option<core::Placement>>;

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

/// Threads a genuine hang left behind. Windows offers no safe way to interrupt a thread
/// blocked inside a provider, so a deadline that expires while the thread is running a job
/// leaks one MTA thread with its apartment and UIA client. The count is reported in that
/// deadline's answer and logged with it: unbounded in principle, visible in practice. A
/// job that never started retires nothing and is not counted — and no other answer carries
/// the count, since nothing is left running for it.
static LEAKED_THREADS: AtomicU64 = AtomicU64::new(0);

/// Why a call's reply never arrived. The two entry points below differ only in
/// what this means: a read failed, an actuation may or may not have landed.
enum JobFailure {
    /// The call itself failed: its own error, with nothing else to say.
    Call(anyhow::Error),
    /// The reply channel died: the thread caught a structured exception or exited
    /// while running the job.
    Stopped,
    /// The deadline expired while the thread was running the job, which was abandoned on
    /// the leaked thread and may still complete.
    Deadline {
        timeout: Duration,
        /// What the abandoned call was addressed to, read before the call moved onto
        /// the thread.
        label: String,
    },
    /// The deadline expired while the job was still queued behind another call: it never
    /// started, so nothing is in flight, nothing was delivered and no thread was retired.
    NotStarted {
        timeout: Duration,
        /// What the call that never started was addressed to.
        label: String,
    },
}

/// Submit one call and wait for its reply under `timeout` — the only way a call
/// reaches the apartment thread. The bookkeeping is here, once: a job abandoned while the
/// thread was running it marks the thread dead, so the next call starts a fresh apartment
/// and the wedged one exits on its own once the provider answers; a job abandoned while
/// still queued behind another call retires nothing, because that thread is healthy.
async fn job_reply(
    call: Call,
    timeout: Duration,
    placed: Option<Arc<PlacementMemo>>,
) -> Result<Reply, JobFailure> {
    // Named before the call is moved onto the thread: the deadline error must say what
    // it was addressed to, and there is nothing left to ask once it is abandoned.
    let label = call.label();
    let (tx, alive) = sender().map_err(JobFailure::Call)?;
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    let state = Arc::new(AtomicU8::new(JobState::Queued as u8));
    if tx
        .send(Job {
            call,
            reply: reply_tx,
            state: Arc::clone(&state),
            placed,
        })
        .is_err()
    {
        alive.store(false, Ordering::SeqCst);
        return Err(JobFailure::Call(core::taxonomy_error(
            core::ERR_DEGRADED,
            "the computer-use thread stopped before it accepted the call — nothing was delivered; \
             retry",
        )));
    }
    match tokio::time::timeout(timeout, reply_rx).await {
        Ok(Ok(Ok(reply))) => Ok(reply),
        Ok(Ok(Err(e))) => Err(JobFailure::Call(e)),
        Ok(Err(_)) => {
            // The thread itself is gone (a caught fault retired it, or it exited):
            // nothing is still running, so nothing is leaked here.
            alive.store(false, Ordering::SeqCst);
            Err(JobFailure::Stopped)
        }
        // Giving up is a race with the thread picking the job up, and the swap settles it:
        // winning means the job will never run (the thread skips an abandoned one), so
        // nothing was delivered; losing means it is already running and may still complete,
        // which retires the thread.
        Err(_) if JobState::settle(&state, JobState::Queued, JobState::Abandoned) => {
            Err(JobFailure::NotStarted { timeout, label })
        }
        Err(_) => {
            leak(&alive);
            Err(JobFailure::Deadline { timeout, label })
        }
    }
}

/// Mark the live thread dead and account for the one it leaves behind: the next call
/// starts a fresh apartment, while this thread keeps the provider call it is stuck in.
fn leak(alive: &AtomicBool) {
    alive.store(false, Ordering::SeqCst);
    let count = LEAKED_THREADS.fetch_add(1, Ordering::SeqCst) + 1;
    tracing::warn!(
        leaked_threads = count,
        "computer use: abandoned a hung worker thread — {count} left behind since this process \
         started"
    );
}

/// How many threads hangs left behind, in the words appended to the answer one of them
/// produced. Read after [`leak`], so it always counts that thread.
fn leaked_note() -> String {
    let count = LEAKED_THREADS.load(Ordering::SeqCst);
    let (threads, have) = if count == 1 {
        ("thread", "has")
    } else {
        ("threads", "have")
    };
    format!(
        "; {count} computer-use worker {threads} {have} been left behind by hangs since start — \
         the next call starts a fresh one"
    )
}

/// Submit one read-shaped call and await the payload it answers with: every
/// abandoned reply has nothing in flight, so all of them are failures.
pub(super) async fn submit<T: FromReply>(call: Call, timeout: Duration) -> anyhow::Result<T> {
    match job_reply(call, timeout, None).await {
        Ok(reply) => T::from_reply(reply),
        Err(JobFailure::Call(e)) => Err(e),
        // Nothing was left running here, so this answer never carries the leak count.
        Err(JobFailure::Stopped) => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "the computer-use thread stopped while running the call — retry",
        )),
        Err(JobFailure::Deadline { timeout, label }) => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "{label} did not respond within {}s — the call was abandoned without a result; \
                 retry{}",
                timeout.as_secs(),
                leaked_note()
            ),
        )),
        Err(JobFailure::NotStarted { timeout, label }) => Err(not_started(timeout, &label)),
    }
}

/// The failure of a job that never reached the thread's call: it waited behind another
/// job for its whole deadline. Nothing of it ran, so nothing was delivered and no thread
/// was retired — both entry points answer it as the plain failure it is.
fn not_started(timeout: Duration, label: &str) -> anyhow::Error {
    core::taxonomy_error(
        core::ERR_DEGRADED,
        format!(
            "{label} did not start within {}s — the shared accessibility thread was busy with \
             another call, so this one never started and nothing was delivered; retry",
            timeout.as_secs()
        ),
    )
}

/// Submit one actuating call — an element action or a batch of input, which both answer
/// from this one submission. Only a job that never reached the thread, one the thread
/// itself refused, or one that never started is a failure; a deadline that expires while
/// the thread runs the job, a thread that stops mid-job and a framework exception all
/// answer [`Impact::Unconfirmed`], because the input may still land. Such an answer keeps
/// the placement the job had already published when it mapped its point, so a call that
/// ended without an answer still says where the input was put.
pub(super) async fn submit_act(call: Call, timeout: Duration) -> anyhow::Result<Actuation> {
    let placed: Arc<PlacementMemo> = Arc::new(Mutex::new(None));
    match job_reply(call, timeout, Some(Arc::clone(&placed))).await {
        Ok(reply) => Actuation::from_reply(reply),
        Err(failure) => {
            let impact = actuation_failure(failure)?;
            // A job that had already mapped its point published it before posting, so an
            // answer that never arrived still states the position the input used.
            Ok(Actuation::placed(
                impact,
                placed.lock().unwrap_poison().take(),
            ))
        }
    }
}

/// The one failure mapping of an actuating call.
fn actuation_failure(failure: JobFailure) -> anyhow::Result<Impact> {
    match failure {
        // Foreign code faulted while the job ran: where it stopped is unknown.
        JobFailure::Call(e) if core::is_framework_failed(&e) => Ok(Impact::Unconfirmed(format!(
            "the system raised a structured exception while the call was running, so its input \
             may be partly applied — {e}"
        ))),
        JobFailure::Call(e) => Err(e),
        JobFailure::Stopped => Ok(Impact::Unconfirmed(String::from(
            "the computer-use thread stopped while running the call, so its input may or may not \
             have been delivered",
        ))),
        JobFailure::Deadline { timeout, label } => Ok(Impact::Unconfirmed(format!(
            "{label} did not answer within {}s, so the call was abandoned while the job may still \
             complete on the leaked thread — the input may land after this answer{}",
            timeout.as_secs(),
            leaked_note()
        ))),
        // Nothing of the job ran, so nothing of it can land: this is not an unconfirmed
        // actuation but a call that was never delivered.
        JobFailure::NotStarted { timeout, label } => Err(not_started(timeout, &label)),
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
            // This job was abandoned before the thread picked it up. It never ran, so
            // it is answered as the failure it is — dropping it would reach an
            // actuating caller as a call that may still land. Break rather than return:
            // the apartment below must still close inside the exception frame.
            let error = unstarted();
            let message = error.to_string();
            let _ = job.reply.send(Err(error));
            retire(rx, alive, &message);
            break;
        }
        // The one swap that decides whether this job runs at all: a caller that gave up
        // while the job was queued has already marked it abandoned, and then nothing may be
        // executed on its behalf — its answer said nothing was delivered.
        if !JobState::settle(&job.state, JobState::Queued, JobState::Running) {
            continue;
        }
        match seh::guard(|| ctx.execute(job.call, job.placed.as_deref())) {
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

/// The answer for a job this thread will never run: it delivered nothing.
fn unstarted() -> anyhow::Error {
    core::taxonomy_error(
        core::ERR_DEGRADED,
        "the computer-use thread was retired before this call started — nothing was delivered; \
         retry",
    )
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

    fn execute(&mut self, call: Call, placed: Option<&PlacementMemo>) -> anyhow::Result<Reply> {
        match call {
            Call::ListApps => Ok(Reply::Apps(window::list_apps())),
            Call::FocusedElement { target } => Ok(Reply::Focus(uia::focused_element(
                &self.uia,
                target.as_ref(),
            ))),
            Call::ListWindows { app } => Ok(Reply::Windows(window::list_windows(app.as_ref()))),
            Call::SurfaceWindow { target } => {
                Ok(Reply::Window(Box::new(window::window_info(&target)?)))
            }
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
            } => Ok(Reply::Input(Box::new(Actuation::of(uia::act_on_element(
                &self.uia, &target, &locator, &act,
            )?)))),
            Call::RawInput { target, input } => Ok(Reply::Input(Box::new(input::raw_input(
                &target, input, placed,
            )?))),
            Call::CursorPosition => {
                let (x, y) = input::cursor_position()?;
                Ok(Reply::Point(x, y))
            }
            Call::Capture { target, budget } => Ok(Reply::Capture(Box::new(
                self.capture.capture(&target, &budget)?,
            ))),
        }
    }
}
