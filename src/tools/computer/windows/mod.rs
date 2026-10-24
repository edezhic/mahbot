//! Windows [`Backend`] for the Computer tool: UI Automation for observe/act,
//! `SendInput` for pointer and keyboard synthesis, and Windows.Graphics.Capture
//! with GDI fallbacks for stills.
//!
//! # One dedicated apartment thread
//!
//! Every accessibility and capture call runs as a job on ONE thread of its own,
//! created lazily by the first call that needs it and initialized into a
//! multithreaded apartment — `RoInitialize(RO_INIT_MULTITHREADED)` plus the
//! `CUIAutomation8` client with bounded provider timeouts. Nothing in this module
//! touches the agent's thread or the interface thread: the [`Backend`] methods
//! only build a plain-data [`thread::Call`] and await its reply, so no COM
//! interface ever crosses a thread boundary and the GUI cannot be blocked by a
//! slow provider. The desktop probe is the one contact outside it — see
//! [`Backend::desktop_unavailable`].
//!
//! # No call may hang
//!
//! Three bounds stack. Each provider call is bounded client-side by the UI Automation
//! timeouts this thread sets; a whole accessibility walk runs under
//! [`uia::WALK_DEADLINE`], a budget checked before every provider read it makes, so a
//! stalled target costs one transaction rather than seven; and the whole job is awaited
//! under a per-call deadline sized for what that shape actually pays (see the deadline
//! constants below). Every one of those failures names the surface it was addressed to.
//! A provider that stalls on several calls inside one job can still reach a job
//! deadline: the thread is marked dead — the next call starts a fresh MTA thread and
//! the wedged one is left to the OS (Windows offers no safe way to interrupt a thread
//! blocked inside a provider). A read then fails with an honest `degraded` timeout; an
//! actuation answers applied-but-unconfirmed, because it may still land on that leaked
//! thread. One thread is leaked per hang and none is reused: the running count is
//! reported in that deadline's answer and logged with it, so repeated hangs are visible
//! instead of silent, and the action lock, the agent and the daemon all stay
//! responsive.
//!
//! # Structured exceptions
//!
//! Foreign code can fault rather than fail: a graphics driver, an in-process provider
//! or the UI Automation client itself can raise a structured exception from inside a
//! call this backend made, and Rust can neither catch one nor unwind past it. Every
//! job the apartment thread serves, the apartment's own setup and its teardown
//! therefore run inside the one C frame this backend owns ([`seh::guard`]).
//!
//! A caught exception fails the call with `framework-failed`, naming the platform's
//! exception code in hex, and retires the thread. The faulting code cannot resume,
//! so the state it owned is unknown: the context is abandoned rather than released,
//! and the next call starts a fresh apartment. [`seh`] owns what a per-call frame
//! cannot cover: a fault raised on a driver's own thread, and the Rust panics,
//! control-flow codes and stack overflows the frame must let pass.
//!
//! # Coordinates and DPI
//!
//! Everything here is PHYSICAL pixels: UI Automation rectangles, window frames,
//! the virtual desktop rect and pointer positions all agree once the process is
//! per-monitor DPI aware. iced's winit event loop makes it so at startup; the
//! thread additionally asks for per-monitor-v2 awareness once, eagerly, before any
//! read — a `cargo`-launched CLI run has no event loop, and a lazily-set awareness
//! would mix physical UIA bounds with virtualized window metrics.
//!
//! # Window identity
//!
//! A window target carries the window's own OS handle plus the pid that owned it
//! ([`window::resolve`] builds and parses that key), so it either lands on that
//! window or fails as `surface-gone`: a moved, resized or renamed window keeps
//! resolving, while a closed or replaced one never silently becomes another window
//! of the same application. An app target names no window — it is the
//! application's own focused window, re-resolved per call. [`input`] owns which
//! input is pinned to a named window and which shapes are global.
//!
//! # Actuation limits, stated rather than worked around
//!
//! [`window::ensure_actuation_allowed`] owns these two refusals and names which one fired:
//!
//! * A window owned by a process running elevated cannot be driven by this
//!   non-elevated one: Windows blocks both synthetic input (UIPI) and accessibility
//!   actions into it. A CAPTURE of that window may still work, because the
//!   compositor, not its owner, serves the frame. Starting elevated is out of scope.
//! * The UAC secure desktop and the lock screen live on another desktop. When either
//!   is up, nothing can be read or actuated there and no workaround is attempted.
//!
//! [`window::ensure_not_own_window`] adds the one refusal that is ours alone: an
//! ELEMENT ACTION on a window of mahbot itself. UI Automation answers our own
//! process in-process, and this backend has no path onto the interface thread the
//! dashboard owns (iced's event loop exposes no way to run a closure on itself),
//! so a press or a value write could freeze the dashboard or kill the service. It
//! is refused as `self-target`; our windows stay listed, readable and capturable,
//! and raw input still reaches them.
//!
//! `cmd` in a key chord is the Windows key here (it is Super on Linux and Command on
//! macOS); `win` is accepted as a key name for a plain Windows-key tap.
//!
//! # Capture
//!
//! [`capture`] owns the route order: Windows.Graphics.Capture first (a window item,
//! or every monitor stitched into the virtual-desktop rect), then a desktop `BitBlt`.
//! [`capture::take_route`] is the one statement of the frame rule every route obeys.
//! Consequences accepted rather than engineered around:
//!
//! * The platform's capture facility draws its own visible capture indicator around
//!   what is being captured, on its route; the GDI fallback is a plain read of the
//!   screen and has none. Removing the indicator needs a packaged, specially
//!   entitled application, which this product deliberately is not.
//! * The facility needs Windows 10 1903. Older builds fall through to the GDI
//!   route, so observation degrades instead of failing.
//! * Some applications hand that facility black frames for their own reasons
//!   (protected content, a hardware-accelerated surface that refuses readback);
//!   that is reported honestly.
//! * A frame is expressed against the window's surface whatever extent the
//!   compositor reports (see [`capture::align_to_surface`]); one that matches
//!   neither convention means the window changed size during the grab and is
//!   refused.
//! * Availability is per target and per route, never cached per window: one window's
//!   failure says nothing about the channel, so [`Backend::capture_available`] answers
//!   for the desk itself (the input desktop is the ordinary one), not for a window.
//!   The tool layer caches that desk answer for a minute (see the capture probe in
//!   `tools::computer`), so a capture failure while it is positive keeps its own
//!   per-route diagnosis instead of being relabelled.
//!
//! # Manual check (run once on a real Windows machine)
//!
//! Build with `cargo build --release`, then run the bare executable from a normal
//! (non-elevated) Command Prompt with `set HOME=C:\Users\<you>\mahbot-check` so a
//! live instance elsewhere is not disturbed. Sign in to the dashboard as the
//! admin and talk to the Assistant.
//!
//! 1. Launch: the dashboard starts, and step 2 works. (The boot capture probe does not
//!    start this module's apartment thread — the first `apps` call does; with
//!    `RUST_LOG=mahbot::tools::computer=debug` each action logs its arguments.)
//! 2. Tool offered: ask the Assistant to `observe` with no target. The reply is a
//!    `[Surface: app=… window=…]` tree — not "unknown tool" and not the stub's
//!    "not supported on this platform".
//! 3. Read a stock application: open Notepad, type a sentence by hand, then ask the
//!    Assistant for `apps`, `windows { app: "notepad" }` and `observe { target:
//!    "w1" }`. The tree contains an `[eN] textfield` carrying the sentence as its
//!    `value` (Windows 11 Notepad exposes its editor as a `document`, which gets a
//!    ref the same way), and the header's node/interactive counts are non-zero.
//! 4. Click a control: open Calculator and ask for `observe`; find the `1` button's
//!    ref and `click { ref: "eN" }`. The display shows 1. Then click `+`, `2`, `=`
//!    by ref and check for 3.
//! 5. Type: `click { ref: "<the Notepad edit ref>" }` then `type { text: "hello" }`
//!    appends visible text. With the ref on a text field (`textfield` or
//!    `document`), `type { text: … }` replaces its value.
//! 6. Press a key: `press { keys: "ctrl+a" }` selects all the text, `press { keys:
//!    "backspace" }` clears it, `press { keys: "win" }` opens the Start menu, and
//!    `press { keys: "cmd+e" }` opens File Explorer (`cmd` is the Windows key on
//!    this platform; a lone `cmd` is a modifier with no key and is rejected).
//! 7. Take a capture: `screenshot { target: "w1" }` returns an image that shows the
//!    window (the OS capture indicator appears around it — expected, keep it), and
//!    `zoom { target: "w1", region: [0,0,500,500] }` returns the top-left quarter.
//!    Minimize that window: the next action on it is refused with `degraded: … is
//!    minimized — restore it first`. An app that only renders protected content
//!    instead reports the all-black failure, naming every route it tried.
//! 8. The honest elevated-window failure: open an Administrator Command Prompt
//!    yourself (accept the UAC prompt) and start `notepad` from it. In the
//!    Assistant, list ALL windows (`windows` with no `app`) — the elevated Notepad
//!    is a second process with its own `wN` — then click it by COORDINATES
//!    (`click { target: "wN", x: 500, y: 500 }`) — the refusal is
//!    `permission-denied: window "…" belongs to notepad (pid N), which runs
//!    elevated …`. `observe { target: "wN" }` on the same window comes back thin or
//!    `permission-denied` (Windows blocks accessibility reads into elevated
//!    processes too — an honest answer, not a bug), while `screenshot { target:
//!    "wN" }` still returns its frame. Finally lock the workstation (Win+L) and try
//!    any action: `desktop-unavailable: … secure desktop, the lock screen or a
//!    non-interactive session …`, never a hang, and `apps`/`windows` still list.
//! 9. Structured-exception containment: `cargo test --lib tools::computer::windows`
//!    on that machine, where all three `seh` tests must pass with the harness still
//!    alive — were containment broken, the test process itself would die instead of
//!    failing. One raises a real null-read fault, one raises a hand-built foreign C++
//!    throw (a provider that throws across the boundary), and one asserts that a Rust
//!    panic still reaches Rust's own handler.

#![cfg(target_os = "windows")]
// The Win32 surface this backend speaks (C-style integer geometry, i32/u32 pixel
// dimensions, RAW handles) makes these pedantic cast lints inherent.
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
    self, AppInfo, Backend, Capture, CaptureBudget, ElementAct, Impact, Locator, Observation,
    RawInput, SurfaceGeometry, TargetSpec, WindowInfo, WindowListing,
};
use async_trait::async_trait;
use std::time::Duration;

// ── Deadlines ───────────────────────────────────────────────────────────
//
// One bound per call shape, sized for what that shape actually pays. For the two
// accessibility shapes that is the common path plus ONE provider transaction: the UIA
// timeouts set on the apartment thread end a single unresponsive provider call, and
// [`uia::WALK_DEADLINE`] — a [`super::core::Budget`] checked before every provider
// read, whose error names the window — ends a walk with its own honest timeout before
// the job deadline fires. A provider that stalls on more than one call inside a job —
// the `ElementFromHandle` before the walk, one `GetCurrentPatternAs` per pattern tried,
// the property reads `set_value` does before it types — can still reach the job
// deadline, and that error names the same surface (the call's own label, read before it
// is submitted). For an actuating call the deadline answers applied-but-unconfirmed
// (see [`thread::submit_act`]), because its input may still land; for a read it stays
// an error, since nothing is left in flight.

/// App/window enumeration and geometry: plain Win32, no provider involved.
const LIST_TIMEOUT: Duration = Duration::from_secs(10);
/// A full accessibility read: the walk plus one transaction.
const OBSERVE_TIMEOUT: Duration = Duration::from_secs(25);
/// An element action — the walk, the pattern it obtains, the action's own call — or a
/// raw input batch, which pays none of those and stays far below this.
const ACT_TIMEOUT: Duration = Duration::from_secs(35);
/// One still, every route included: the screen stitch is cut at `SCREEN_BUDGET` (it
/// checks the budget before each display and waits no longer than what is left), then
/// the GDI route runs.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(25);
/// The desktop probe: a handful of user32 calls that need no apartment and no provider,
/// bounded on a blocking thread so a desktop switch that wedges costs this and not the
/// calling task's own budget. A probe that does not answer in time is "cannot tell".
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

// ── Backend ─────────────────────────────────────────────────────────────

pub(crate) static WINDOWS_BACKEND: WindowsBackend = WindowsBackend;

pub(crate) struct WindowsBackend;

#[async_trait]
impl Backend for WindowsBackend {
    /// No grant, no permission and no signature is involved: UI Automation,
    /// `SendInput` and the compositor's capture facility are all present on the
    /// product's Windows floor. What can still fail per call — an elevated
    /// window, the secure desktop, a single uncooperative provider — is refused
    /// with its own diagnosis instead.
    fn is_supported(&self) -> bool {
        true
    }

    /// Windows names the state itself: the input desktop is `Default` only while the
    /// owner is looking at it, and any other name — `Winlogon`, `Screen-saver`, a
    /// session that is not attached — is a desktop this process must not act into.
    ///
    /// The probe runs on a blocking thread of its own rather than through [`thread`]'s
    /// apartment: it needs no COM and no provider, and an apartment thread the pool had to
    /// abandon would read as this refusal instead of the hang it is.
    async fn desktop_unavailable(&self) -> Option<String> {
        (desktop_probe().await == Some(false)).then(|| SECURE_DESKTOP_REASON.to_string())
    }

    /// Whether an interactive desktop is attached. Deliberately not about any
    /// one window and never cached per target: a window that refuses capture
    /// leaves the channel available for everything else. The same probe as
    /// [`Self::desktop_unavailable`], and a probe that could not answer is taken as
    /// available — the capture attempt then reports its own failure rather than a state
    /// that was never observed.
    async fn capture_available(&self) -> bool {
        desktop_probe().await != Some(false)
    }

    /// The desktop can change between the gate and the capture, so this names the same
    /// state the gate does: a retry cannot succeed either.
    fn capture_unavailable_error(&self) -> anyhow::Error {
        super::core::taxonomy_error(
            super::core::ERR_DESKTOP_UNAVAILABLE,
            format!(
                "{SECURE_DESKTOP_REASON} — screen capture is unavailable there; ask the owner to \
                 return to their desktop"
            ),
        )
    }

    async fn list_apps(&self) -> anyhow::Result<Vec<AppInfo>> {
        thread::submit(Call::ListApps, LIST_TIMEOUT).await
    }

    async fn list_windows(&self, app: Option<&AppInfo>) -> anyhow::Result<WindowListing> {
        let app = app.cloned();
        thread::submit(Call::ListWindows { app }, LIST_TIMEOUT).await
    }

    async fn focused_window(&self) -> anyhow::Result<WindowInfo> {
        thread::submit(Call::FocusedWindow, LIST_TIMEOUT).await
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
        thread::submit_act(call, ACT_TIMEOUT).await
    }

    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> anyhow::Result<Impact> {
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
