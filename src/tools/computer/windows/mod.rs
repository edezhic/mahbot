//! Windows [`Backend`] for the Computer tool: UI Automation for observe/act,
//! `SendInput` for pointer and keyboard synthesis, and Windows.Graphics.Capture
//! with GDI fallbacks for stills.
//!
//! # One dedicated apartment thread
//!
//! Every accessibility and capture call runs as a job on ONE thread of its own, created
//! lazily by the first call that needs it and initialized into a multithreaded apartment —
//! `RoInitialize(RO_INIT_MULTITHREADED)` plus the `CUIAutomation8` client with bounded
//! provider timeouts. Nothing here touches the agent's thread or the interface thread: the
//! [`Backend`] methods only build a plain-data [`thread::Call`] and await its reply, so no
//! COM interface crosses a thread boundary and the GUI cannot be blocked by a slow provider.
//!
//! # No call may hang
//!
//! Three bounds stack: each provider call is bounded client-side by the UIA timeouts this
//! thread sets; a walk runs under [`uia::WALK_DEADLINE`], a budget checked before every
//! provider read, so a stalled target costs one transaction; and the whole job is awaited
//! under a per-call deadline sized for what that shape pays. A provider that stalls on
//! several calls inside one job can still reach the job deadline: the thread is marked dead,
//! the next call starts a fresh MTA thread, and the wedged one is left to the OS (Windows
//! offers no safe way to interrupt it). A read then fails as a `degraded` timeout; an
//! actuation answers applied-but-unconfirmed. One thread is leaked per hang and none is
//! reused: the running count is reported in that answer and logged with it.
//!
//! # Structured exceptions
//!
//! Foreign code can fault rather than fail, and Rust can neither catch a structured exception
//! nor unwind past it. Every job, the apartment's setup and its teardown therefore run inside
//! the one C frame this backend owns ([`seh::guard`]). A caught exception fails the call with
//! `framework-failed`, naming the platform's exception code in hex, and retires the thread:
//! the state it owned is unknown, so the context is abandoned rather than released. [`seh`]
//! owns what a per-call frame cannot cover — a fault on a driver's own thread, and the Rust
//! panics, control-flow codes and stack overflows the frame must let pass.
//!
//! # Coordinates and DPI
//!
//! Everything here is PHYSICAL pixels: UIA rectangles, window frames, the virtual desktop rect
//! and pointer positions all agree once the process is per-monitor DPI aware. iced's winit
//! event loop makes it so at startup; the thread additionally asks for per-monitor-v2
//! awareness once, eagerly — a `cargo`-launched CLI run has no event loop, and a lazily-set
//! awareness would mix physical UIA bounds with virtualized window metrics.
//!
//! # Window identity
//!
//! A window target carries the window's own OS handle plus the pid that owned it
//! ([`window::resolve`] builds and parses that key), so it either lands on that window or
//! fails as `surface-gone`: a moved, resized or renamed window keeps resolving, a closed or
//! replaced one never silently becomes another window of the same application. An app target
//! names no window — it is the application's own focused window, re-resolved per call.
//!
//! # Actuation limits, stated rather than worked around
//!
//! [`window::ensure_actuation_allowed`] owns two refusals and names which one fired:
//!
//! * A window owned by an elevated process cannot be driven by this non-elevated one: Windows
//!   blocks both synthetic input (UIPI) and accessibility actions into it. A CAPTURE of it may
//!   still work, because the compositor serves the frame. Starting elevated is out of scope.
//! * The UAC secure desktop and the lock screen live on another desktop: nothing can be read
//!   or actuated there, and no workaround is attempted.
//!
//! [`window::ensure_not_own_window`] adds the one refusal that is ours alone: an ELEMENT
//! ACTION on a window of mahbot itself. UI Automation answers our own process in-process, and
//! this backend has no path onto the interface thread the dashboard owns, so a press or a
//! value write could freeze the dashboard or kill the service. It is refused as
//! `self-target`; our windows stay listed, readable and capturable, and raw input still
//! reaches them.
//!
//! `cmd` in a key chord is the Windows key here (Super on Linux, Command on macOS); `win` is
//! accepted as a key name for a plain Windows-key tap.
//!
//! # Capture
//!
//! [`capture`] owns the route order: Windows.Graphics.Capture first, then a desktop `BitBlt`.
//! [`capture::take_route`] is the one statement of the frame rule every route obeys.
//! Consequences accepted rather than engineered around:
//!
//! * The platform's capture facility draws its own visible indicator around what it captures;
//!   the GDI fallback has none. Removing it needs a packaged, specially entitled application,
//!   which this product deliberately is not.
//! * The facility needs Windows 10 1903; older builds fall through to GDI, so observation
//!   degrades instead of failing.
//! * Some applications hand it black frames (protected content, a surface that refuses
//!   readback); that is reported honestly.
//! * A frame is expressed against the window's surface whatever extent the compositor reports
//!   ([`capture::align_to_surface`]); one matching neither convention means the window changed
//!   size during the grab and is refused.
//! * Availability is per target and per route, never cached per window:
//!   [`Backend::capture_available`] answers for the desk itself. The tool layer caches that
//!   answer for a minute, so a failure while it is positive keeps its per-route diagnosis.
//!
//! # Manual check (run once on a real Windows machine)
//!
//! Build with `cargo build --release`, run the bare executable from a normal Command Prompt
//! with `set HOME=C:\Users\<you>\mahbot-check`, sign in as admin and talk to the Assistant.
//! Check, in order: the dashboard starts; `observe` with no target returns a
//! `[Surface: app=… window=…]` tree (not "unknown tool"); a Notepad tree carries the typed
//! sentence as a `value`; a Calculator button `click { ref }` and `press { keys }` behave;
//! `screenshot`/`zoom` return the window and a minimized one is refused as `degraded`; an
//! elevated Notepad is refused `permission-denied` for input and still captures, and a locked
//! workstation answers `desktop-unavailable` rather than hanging; and
//! `cargo test --lib tools::computer::windows` passes with the harness alive — the three
//! `seh` tests cover a null-read fault, a foreign C++ throw and a Rust panic.

#![cfg(target_os = "windows")]
// The Win32 surface this backend speaks (C-style integer geometry, i32/u32 pixel dimensions,
// RAW handles) makes these pedantic cast lints inherent.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod capture;
mod input;
mod seh;
mod thread;
mod uia;
mod window;

use self::thread::Call;
use self::window::SECURE_DESKTOP_REASON;
use super::core::{
    self, Actuation, AppInfo, AppListing, Backend, Capture, CaptureBudget, ElementAct,
    FocusedElement, Impact, Locator, Observation, RawInput, SurfaceGeometry, TargetSpec,
    WindowInfo, WindowListing,
};
use async_trait::async_trait;
use std::time::Duration;

// One bound per call shape, sized for what that shape actually pays: for the accessibility
// shapes, the common path plus ONE provider transaction. The UIA timeouts end a single
// unresponsive provider call, and [`uia::WALK_DEADLINE`] — a [`super::core::Budget`] checked
// before every provider read, whose error names the window — ends a walk before the job
// deadline fires. A provider that stalls on more than one call inside a job can still reach
// the job deadline, and that error names the same surface. An actuating call answers
// applied-but-unconfirmed there, a read stays an error, and a deadline that expires before
// the job started is the plain `degraded` failure.

/// App/window enumeration and geometry: plain Win32, no provider involved.
const LIST_TIMEOUT: Duration = Duration::from_secs(10);
/// A full accessibility read: the walk plus one transaction.
const OBSERVE_TIMEOUT: Duration = Duration::from_secs(25);
/// An element action — the walk, the pattern it obtains, the action's own call — or a raw
/// input batch, which pays none of those.
const ACT_TIMEOUT: Duration = Duration::from_secs(35);
/// One still, every route included: the screen stitch is cut at `SCREEN_BUDGET`, then the
/// GDI route runs.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(25);
/// The desktop probe: user32 calls that need no apartment and no provider, bounded on a
/// blocking thread so a wedged desktop switch costs this and not the caller's budget.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Whether an interactive desktop is attached, or `None` when the probe could not answer.
async fn desktop_probe() -> Option<bool> {
    tokio::time::timeout(
        PROBE_TIMEOUT,
        tokio::task::spawn_blocking(window::interactive_desktop_available),
    )
    .await
    .ok()?
    .ok()
}

pub(crate) static WINDOWS_BACKEND: WindowsBackend = WindowsBackend;

pub(crate) struct WindowsBackend;

#[async_trait]
impl Backend for WindowsBackend {
    /// No grant, no permission and no signature is involved: UIA, `SendInput` and the capture
    /// facility are all present on the product's Windows floor. What can still fail per call —
    /// an elevated window, the secure desktop, an uncooperative provider — is refused with its
    /// own diagnosis.
    fn is_supported(&self) -> bool {
        true
    }

    /// Windows names the state itself: the input desktop is `Default` only while the owner is
    /// looking at it, and any other name — `Winlogon`, `Screen-saver`, a detached session — is
    /// a desktop this process must not act into.
    ///
    /// The probe runs on a blocking thread of its own rather than through [`thread`]'s
    /// apartment: it needs no COM, and an apartment thread the pool had to abandon would read
    /// as this refusal instead of the hang it is.
    async fn desktop_unavailable(&self) -> Option<String> {
        (desktop_probe().await == Some(false)).then(|| SECURE_DESKTOP_REASON.to_string())
    }

    /// Whether an interactive desktop is attached — per call, never cached per target: a
    /// window that refuses capture leaves the channel available. A probe that could not answer
    /// is taken as available, and the capture attempt reports its own failure.
    async fn capture_available(&self) -> bool {
        desktop_probe().await != Some(false)
    }

    /// The desktop can change between the gate and the capture, so this names the same state
    /// the gate does: a retry cannot succeed either.
    fn capture_unavailable_error(&self) -> anyhow::Error {
        super::core::taxonomy_error(
            super::core::ERR_DESKTOP_UNAVAILABLE,
            format!(
                "{SECURE_DESKTOP_REASON} — screen capture is unavailable there; ask the owner to \
                 return to their desktop"
            ),
        )
    }

    async fn list_apps(&self) -> anyhow::Result<AppListing> {
        thread::submit(Call::ListApps, LIST_TIMEOUT).await
    }

    async fn list_windows(&self, app: Option<&AppInfo>) -> anyhow::Result<WindowListing> {
        let app = app.cloned();
        thread::submit(Call::ListWindows { app }, LIST_TIMEOUT).await
    }

    /// A plain process query, run here rather than through the apartment: no COM, no provider.
    async fn app_running(&self, app: &AppInfo) -> Option<bool> {
        app.pid.and_then(window::process_alive)
    }

    async fn surface_window(&self, target: &TargetSpec) -> anyhow::Result<WindowInfo> {
        let call = Call::SurfaceWindow {
            target: target.clone(),
        };
        thread::submit(call, LIST_TIMEOUT).await
    }

    async fn surface_geometry(&self, target: &TargetSpec) -> anyhow::Result<SurfaceGeometry> {
        let target = target.clone();
        thread::submit(Call::SurfaceGeometry { target }, LIST_TIMEOUT).await
    }

    async fn observe(&self, target: &TargetSpec) -> anyhow::Result<Observation> {
        let target = target.clone();
        thread::submit(Call::Observe { target }, OBSERVE_TIMEOUT).await
    }

    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> anyhow::Result<Impact> {
        let call = Call::Act {
            target: target.clone(),
            locator: locator.clone(),
            act,
        };
        thread::submit_act(call, ACT_TIMEOUT)
            .await
            .map(|actuation| actuation.impact)
    }

    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> anyhow::Result<Actuation> {
        // Argument bounds first, before the call reaches the interface thread.
        core::check_raw_input(&input)?;
        let call = Call::RawInput {
            target: target.clone(),
            input,
        };
        thread::submit_act(call, ACT_TIMEOUT).await
    }

    async fn cursor_position(&self) -> anyhow::Result<(f64, f64)> {
        thread::submit(Call::CursorPosition, LIST_TIMEOUT).await
    }

    /// Read best-effort, as the contract requires: anything this cannot establish is `None`,
    /// which the action's report states as unread.
    async fn focused_element(&self, target: Option<&TargetSpec>) -> Option<FocusedElement> {
        let call = Call::FocusedElement {
            target: target.cloned(),
        };
        thread::submit(call, LIST_TIMEOUT).await.ok().flatten()
    }

    async fn capture(
        &self,
        target: &TargetSpec,
        budget: &CaptureBudget,
    ) -> anyhow::Result<Capture> {
        let call = Call::Capture {
            target: target.clone(),
            budget: budget.clone(),
        };
        thread::submit(call, CAPTURE_TIMEOUT).await
    }
}
