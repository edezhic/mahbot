//! macOS [`Backend`] for the Computer tool: AX accessibility (observe/act),
//! CGEvent raw input, and ScreenCaptureKit capture. Non-`Send` platform objects
//! are created and dropped inside the blocking closure of the call that uses them
//! ([`with_platform`]) — never in struct state or across an await. The one
//! path outside such a closure is the grant diagnosis ([`permission_error`]),
//! which resolves the responsible application's name synchronously, with no await
//! in between.
//!
//! # Foreign failures
//!
//! System frameworks fail in their own terms: an Objective-C exception raised
//! inside a framework call is not a Rust panic, so Rust cannot contain it — an
//! unwind that is not caught at the call ends the process, and the panic guard
//! aborts on it outright. Every async [`Backend`] entry therefore runs behind
//! [`with_platform`] — the blocking body off the async worker, behind
//! [`catching_foreign`] — which turns such an exception into a `framework-failed`
//! failure of the call. The AX C functions are not messages, so the bindings offer
//! no per-site wrapper for them: the boundary of the call is the widest point an
//! exception can still be caught at.
//!
//! What runs outside that boundary is guarded where it runs, and a failure there has
//! no call left to fail, so it is logged and dropped: the grant prompt thread, the
//! grant-name lookup of [`permission_error`], the main-queue exchange of
//! [`on_ax_thread`] and the completion handlers the system invokes ([`contained`] —
//! the waiter may already have given up). The one synchronous entry without a call to
//! fail into, [`capture_unavailable_error`], guards its own platform contacts.
//!
//! Answers are checked before they are used: an AX array is filtered to the member
//! type it is read as ([`typed_members`]) instead of being cast wholesale, an AX
//! string is cut at the product's cap by its length before a character of it is
//! copied ([`attribute_text`]), and a capture size is validated before it reaches the
//! capture framework. This backend charges every frame copy it makes and the
//! composite it builds to the call's budget; the framework's own texture is bounded
//! by the same geometry, while the copy this tool keeps is the one budgeted — the
//! reasoning lives in [`core::CAPTURE_BUDGET_PIXELS`].
//!
//! # Deadlines
//!
//! One AX exchange is bounded by the process-wide messaging timeout this backend sets
//! on the system-wide element before its first one ([`install_messaging_timeout`],
//! [`core::EXCHANGE_DEADLINE`], deliberately not shorter than the default applications
//! rely on). One [`core::Budget`] per call bounds the whole of it; the walk cannot be
//! bounded from outside — it runs inside the blocking frame no timer can interrupt — so
//! it checks the budget between exchanges, and a responsive target never reaches it.
//!
//! # Surfaces and identity
//!
//! A window target carries the identity this backend gave it when it was
//! enumerated — a retained AX window element ([`HELD_WINDOWS`]) — so a stored id
//! resolves to that very window, live: a moved, resized or renamed window keeps
//! working, and a closed or replaced one fails as `surface-gone` instead of
//! becoming another window of the same application. An app target names no
//! window: it is the application's own focused window, re-resolved per call.
//!
//! Stills of a window are matched to ScreenCaptureKit's list by owning process and
//! frame (`find_sc_window`), never by title alone or by position, and a minimized
//! window is refused wherever its frame would be reported or used — capture,
//! observe and coordinate mapping. The `screen` target is composited at the densest
//! attached display's scale, so a mixed-density arrangement yields one faithful
//! image.
//!
//! Input pinned to a window — typed text, a drag, a wheel at a point — is raised
//! and then verified to be the window the server would deliver to
//! ([`ensure_window_receives`]); when it could not be brought forward the call
//! fails with nothing sent. Keyboard chords, a wheel with no point and a coordinate
//! click are global input and are described as such.
//!
//! # Availability
//!
//! macOS 14 is the product's floor, and the ScreenCaptureKit still API
//! (`SCScreenshotManager` + `SCContentFilter.pointPixelScale`) is exactly that
//! old, so [`sck_available`] guards every use of it: on a Mac below the floor
//! capture degrades to an honest error while observe/act keep working. The guard
//! is a runtime OS-version test in a build whose deployment target is lower and a
//! compile-time constant in the published build, whose own floor is 14.0; the
//! point of having it is that messaging a class or selector the running system
//! lacks aborts the whole process instead of failing the call, so nothing newer
//! than the floor may ever be called unguarded.
//!
//! # Privacy grants
//!
//! A bare binary has no privacy identity: macOS attributes every grant to the
//! RESPONSIBLE PROCESS, which for this product is the terminal the daemon was
//! started from — a detached daemon keeps it. The owner grants Accessibility
//! (tree reads and element actions), input sending (synthesized clicks, typing
//! and key chords) and Screen Recording (stills) to that terminal once, and the
//! grants survive rebuilds and self-updates because the terminal's signature
//! never changes. Accepted consequences: every process started from the same
//! terminal — the shells the agent itself runs included — shares them, and a
//! daemon started some other way has none until that other thing is granted
//! them too. A missing grant is requested and, when still missing, reported by
//! [`permission_error`]; nothing degrades silently.
//!
//! # Our own windows
//!
//! An accessibility request addressed to mahbot's own window is answered inside
//! this very process, on its main thread. From a worker thread it is not a clean
//! failure: raising the window or acting on an element traps the process with
//! SIGTRAP (uncatchable, and fatal on current macOS), while writing a value waits
//! forever. Every AX exchange aimed at our own process therefore runs on the main
//! queue and is awaited for a bounded time — see [`on_ax_thread`] — and our
//! windows stay listed, readable, actionable and capturable like any other
//! application's.

#![cfg(target_os = "macos")]
// The many casts and raw-pointer borrows are inherent to the C ABI/geometry
// types the FFI exposes (u32 display IDs, usize pixel dims, CGFloat floats);
// suppressing the pedantic cast/pointer lints here keeps the file focused.
#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    clippy::semicolon_if_nothing_returned
)]

use super::core::{
    self, AppInfo, Backend, Capture, CaptureBudget, ElementAct, Impact, Locator, Modifier,
    MouseButton, Observation, RawInput, ScrollDirection, SurfaceGeometry, TargetSpec, Truncation,
    UiNode, UnreadWindows, WindowInfo, WindowKey, WindowListing,
};
use crate::util::{UnwrapPoison, with_block_in_place};
use anyhow::anyhow;
use block2::RcBlock;
use dispatch2::{DispatchRetained, DispatchSemaphore, DispatchTime};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_app_kit::{NSApplicationActivationPolicy, NSRunningApplication, NSWorkspace};
use objc2_application_services::{
    self, AXError, AXIsProcessTrusted, AXIsProcessTrustedWithOptions, AXUIElement, AXValue,
    AXValueType, kAXTrustedCheckOptionPrompt,
};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFDictionary, CFRange, CFRetained, CFString, CFType, CGPoint, CGRect,
    CGSize, ConcreteType, Type,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGBitmapContextGetData, CGColorSpace, CGContext, CGDirectDisplayID,
    CGDisplayBounds, CGError, CGEvent, CGEventField, CGEventFlags, CGEventSource,
    CGEventSourceStateID, CGEventTapLocation, CGEventType, CGGetActiveDisplayList, CGImage,
    CGImageAlphaInfo, CGImageByteOrderInfo, CGMouseButton, CGPreflightPostEventAccess,
    CGPreflightScreenCaptureAccess, CGRequestPostEventAccess, CGRequestScreenCaptureAccess,
    CGScrollEventUnit,
};
use objc2_foundation::{NSArray, NSError, NSString};
use objc2_screen_capture_kit::{
    SCContentFilter, SCScreenshotManager, SCShareableContent, SCStreamConfiguration,
    SCStreamErrorCode, SCWindow,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Once};
use std::time::Duration;

pub(crate) static MACOS_BACKEND: MacOsBackend = MacOsBackend;

pub(crate) struct MacOsBackend;

// ── AX exchange deadline ─────────────────────────────────────────────────
//
// The contract lives in the module header; this is the one place it is set.

/// Set the AX messaging timeout, once per process, on the system-wide element.
///
/// Apple honours a process-wide setting only when it is applied to the system-wide
/// element — one set on an application or window element bounds that element alone, and
/// this backend makes its exchanges through fresh per-application elements. Installed
/// from [`with_platform`], the funnel every AX entry point passes through, rather than
/// from the grant check: a missing grant must not be what decides whether the setting
/// was ever applied.
fn install_messaging_timeout() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: both calls only configure this process's own accessibility client —
        // creating the system-wide element allocates it, and the timeout is a positive
        // number of seconds; no pointer outlives the statement and no other process is
        // contacted.
        let err = unsafe {
            AXUIElement::new_system_wide()
                .set_messaging_timeout(core::EXCHANGE_DEADLINE.as_secs_f32())
        };
        if err != AXError::Success {
            tracing::warn!(
                limit_secs = core::EXCHANGE_DEADLINE.as_secs(),
                "computer use: could not set the AX messaging timeout ({err:?}) — an \
                 exchange that never answers is then unbounded"
            );
        }
    });
}

// ── Foreign failures ─────────────────────────────────────────────────────
//
// The contract lives in the module header; these are the helpers it names.

/// Run a `Result`-returning body behind Objective-C's own exception handler: an
/// exception the system raises anywhere under it is caught here and returned as a
/// failed call. A Rust panic passes through untouched (it is not a foreign exception);
/// where both guards are wanted, [`contained`] owns their order.
fn catching_foreign<T>(
    what: &str,
    f: impl FnOnce() -> Result<T, anyhow::Error>,
) -> Result<T, anyhow::Error> {
    objc2::exception::catch(std::panic::AssertUnwindSafe(f))
        .map_err(|exc| {
            let detail = exc
                .as_ref()
                .map_or_else(|| "an unnamed exception".to_string(), ToString::to_string);
            core::framework_failed(what, format!("the system raised an exception: {detail}"))
        })
        .and_then(std::convert::identity)
}

/// Both guards in the one order that works — the exception firewall inside the panic
/// guard, because a foreign exception reaching that guard aborts the process — for a
/// body that runs in frames we do not own: a dispatch queue, or a callback the system
/// invokes. Either failure becomes the error this returns.
fn contained<T>(
    what: &str,
    body: impl FnOnce() -> Result<T, anyhow::Error>,
) -> Result<T, anyhow::Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        catching_foreign(what, body)
    }))
    .unwrap_or_else(|payload| {
        Err(anyhow!(
            "{what} panicked: {}",
            crate::util::panic_message(payload.as_ref())
        ))
    })
}

/// One entry into the system from the async side: the blocking body off the async
/// worker, behind the Objective-C firewall. `what` names the operation for the
/// failure it reports.
fn with_platform<T>(
    what: &str,
    f: impl FnOnce() -> Result<T, anyhow::Error>,
) -> Result<T, anyhow::Error> {
    with_block_in_place(|| {
        catching_foreign(what, || {
            install_messaging_timeout();
            f()
        })
    })
}

// ── kAX* attribute/action string constants ───────────────────────────────
//
// The objc2 bindings do not ship the kAX* string constants, so the ones we use
// are declared here. `CFString::from_static_str` reuses the static buffer, so
// the fresh retain per call is cheap; drop happens at the end of the statement.

const AX_ROLE: &str = "AXRole";
const AX_TITLE: &str = "AXTitle";
const AX_VALUE: &str = "AXValue";
const AX_SUBROLE: &str = "AXSubrole";
const AX_POSITION: &str = "AXPosition";
const AX_SIZE: &str = "AXSize";
const AX_CHILDREN: &str = "AXChildren";
const AX_WINDOWS: &str = "AXWindows";
const AX_FOCUSED_WINDOW: &str = "AXFocusedWindow";
const AX_FOCUSED: &str = "AXFocused";
const AX_PRESS: &str = "AXPress";
const AX_OPEN: &str = "AXOpen";
const AX_RAISE: &str = "AXRaise";
const AX_MINIMIZED: &str = "AXMinimized";

#[inline]
fn axc(s: &'static str) -> CFRetained<CFString> {
    CFString::from_static_str(s)
}

// ── Privacy grants (TCC) ─────────────────────────────────────────────────
//
// See the module docs for the model. Each grant is probed cheaply, requested at
// most once per process (the system's dialog is the owner's to answer, and a
// retry loop must not re-open it), and reported through one actionable error
// that names the terminal macOS holds the grant against.

/// The three grants a computer-use action can need.
#[derive(Clone, Copy)]
enum Grant {
    /// Reading the AX tree and acting on elements.
    Accessibility,
    /// Posting synthesized input events.
    Input,
    /// Capturing stills of windows and displays.
    ScreenRecording,
}

impl Grant {
    /// What the grant enables, as the error and the log name it.
    fn what(self) -> &'static str {
        match self {
            Self::Accessibility => "Accessibility",
            Self::Input => "permission to send input (clicks, typing and key chords)",
            Self::ScreenRecording => "Screen Recording",
        }
    }

    /// The privacy pane the owner grants it in.
    fn pane(self) -> &'static str {
        match self {
            // macOS files synthesized input under the same grant that lets an
            // application control others.
            Self::Accessibility | Self::Input => "Accessibility",
            Self::ScreenRecording => "Screen Recording",
        }
    }

    /// Cheap, prompt-free check of this process's own grant.
    fn held(self) -> bool {
        // SAFETY: each call only reads this process's own privacy state.
        unsafe {
            match self {
                Self::Accessibility => AXIsProcessTrusted(),
                Self::Input => CGPreflightPostEventAccess(),
                Self::ScreenRecording => CGPreflightScreenCaptureAccess(),
            }
        }
    }

    /// The flag recording that this process already asked for this grant.
    fn asked(self) -> &'static AtomicBool {
        match self {
            Self::Accessibility => &ACCESSIBILITY_ASKED,
            Self::Input => &INPUT_ASKED,
            Self::ScreenRecording => &SCREEN_RECORDING_ASKED,
        }
    }

    /// Ask the system to show its grant dialog, attributed to the responsible
    /// process (the terminal) and not to this binary. The answer arrives outside
    /// the call — the caller re-probes. The Accessibility variant returns at once;
    /// the two CG requests wait for the owner, so the prompt runs on its own
    /// thread and the caller waits at most [`PROMPT_WAIT`], which bounds how long
    /// the computer-action lock is held while the dialog sits unanswered.
    fn request(self) {
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // A fresh thread is outside the caller's firewall and nothing there can
            // fail the call: the answer is read by the caller's re-probe, so a failure
            // is logged and dropped, and a failing body returns instead of leaving the
            // caller to its timeout.
            if let Err(e) = contained("ask for the grant", || {
                self.prompt();
                Ok(())
            }) {
                tracing::warn!("computer use: {e:#}");
            }
            let _ = done.send(());
        });
        let _ = wait.recv_timeout(PROMPT_WAIT);
    }

    /// The blocking half of [`Grant::request`]; the answer is read by the caller's
    /// re-probe, not from the return values here.
    fn prompt(self) {
        // SAFETY: each call only asks the system to prompt; nothing is granted
        // behind the owner's back and no pointer outlives the call.
        unsafe {
            match self {
                Self::Accessibility => {
                    // `kAXTrustedCheckOptionPrompt: true` is what makes the call
                    // show the system's grant dialog instead of only answering.
                    let options = CFDictionary::<CFString, CFBoolean>::from_slices(
                        &[kAXTrustedCheckOptionPrompt],
                        &[CFBoolean::new(true)],
                    );
                    let _ = AXIsProcessTrustedWithOptions(Some(options.as_opaque()));
                }
                Self::Input => {
                    let _ = CGRequestPostEventAccess();
                }
                Self::ScreenRecording => {
                    let _ = CGRequestScreenCaptureAccess();
                }
            }
        }
    }
}

/// How long an unanswered grant dialog may hold the action that asked for it
/// before the call reports the grant as missing; the dialog's late answer still
/// applies to the next call.
const PROMPT_WAIT: Duration = Duration::from_secs(20);

/// Set once this process has asked the system for the grant, so the owner answers
/// each dialog once instead of once per agent retry.
static ACCESSIBILITY_ASKED: AtomicBool = AtomicBool::new(false);
static INPUT_ASKED: AtomicBool = AtomicBool::new(false);
static SCREEN_RECORDING_ASKED: AtomicBool = AtomicBool::new(false);

/// Require `grant`, asking the system to prompt for it when this process has not
/// asked yet. The error is the actionable diagnosis — never a silent fallback and
/// never a retry behind the owner's back.
fn ensure(grant: Grant) -> anyhow::Result<()> {
    if grant.held() {
        return Ok(());
    }
    if !grant.asked().swap(true, Ordering::Relaxed) {
        // The log carries the same facts as the error the caller will see — the
        // restart for Screen Recording and its periodic re-ask included.
        tracing::warn!(
            "computer use: asking the system to prompt — {}",
            permission_error(grant)
        );
        grant.request();
    }
    if grant.held() {
        return Ok(());
    }
    Err(permission_error(grant))
}

/// The actionable diagnosis for a grant this process does not hold: what is
/// unavailable, which application the owner must authorize, where, and what the
/// grant needs before it applies.
fn permission_error(grant: Grant) -> anyhow::Error {
    let app = responsible_app();
    let consequence = match grant {
        Grant::Accessibility => {
            "Nothing was observed or acted on. The grant covers every process started from that \
             terminal, this daemon's own shell commands included."
        }
        Grant::Input => {
            "Clicks, typing and key chords stay unavailable; reading the accessibility tree does \
             not need this grant."
        }
        Grant::ScreenRecording => {
            "Restart mahbot before it applies — macOS gives this grant only to a newly started \
             process. Stills stay unavailable meanwhile; observe and act keep working. macOS asks \
             for this grant again periodically by design — grant it again the same way. Every \
             process started from that terminal shares it."
        }
    };
    core::permission_denied(format!(
        "{} is not granted to {app}. macOS attributes privacy grants to the terminal this \
         daemon was started from, not to the binary itself — open System Settings → Privacy & \
         Security → {} and enable {app}. {consequence}",
        grant.what(),
        grant.pane(),
    ))
}

/// The application macOS holds this daemon's grants against: the terminal it was
/// started from. The process tree no longer reaches it once the daemon detaches,
/// but the environment it inherited names it — terminal emulators export their
/// own bundle identifier (`__CFBundleIdentifier`, set by AppKit) and program name
/// (`TERM_PROGRAM`). The last fallback says what to look for rather than claiming
/// a name it does not know.
fn responsible_app() -> String {
    let non_empty = |key: &str| {
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    };
    if let Some(bundle_id) = non_empty("__CFBundleIdentifier") {
        // The running application's localized name is what System Settings shows.
        let id = NSString::from_str(&bundle_id);
        let resolved = catching_foreign("resolve the responsible application", || {
            Ok(
                NSRunningApplication::runningApplicationsWithBundleIdentifier(&id)
                    .iter()
                    .find_map(|app| {
                        app.localizedName()
                            .map(|name| name.to_string())
                            .filter(|name| !name.is_empty())
                    }),
            )
        })
        .inspect_err(|e| tracing::warn!("computer use: {e:#}"))
        .ok()
        .flatten();
        return resolved.unwrap_or(bundle_id);
    }
    non_empty("TERM_PROGRAM").unwrap_or_else(|| "the terminal you started mahbot from".to_string())
}

// ── Availability guards ──────────────────────────────────────────────────

/// ScreenCaptureKit stills (`SCScreenshotManager`, `SCContentFilter`'s pixel
/// scale) need macOS 14. Below it no SCK symbol is touched: capture reports
/// [`screen_capture_needs_14`] and the AX channel stays fully usable.
fn sck_available() -> bool {
    objc2::available!(macos = 14.0, ..)
}

fn screen_capture_needs_14() -> anyhow::Error {
    core::taxonomy_error(
        core::ERR_UNSUPPORTED,
        "screen capture needs macOS 14 or newer; observing and acting through the accessibility \
         tree keep working",
    )
}

// ── Our own process ─────────────────────────────────────────────────────
//
// The thread decision (see the module header) is taken where the target
// process becomes known, never earlier from NSWorkspace state: the frontmost app
// can change between that decision and the exchange. Target-addressed exchanges
// get the pid from [`resolve_window`], the one place a target becomes a platform
// handle, which raises [`NeedsAxThread`] before its first AX call; the window
// listing knows the pid up front and enters through [`on_ax_thread`] directly.
// [`with_target`] turns the marker into a re-run on the main queue.

/// The longest a self-targeted AX exchange waits for the main queue.
const SELF_AX_WAIT: Duration = Duration::from_secs(5);

/// Marker for "this exchange resolved to mahbot itself while off the main
/// thread". Never reaches the caller: [`with_target`] catches it and re-runs the
/// exchange on the main queue.
#[derive(Debug)]
struct NeedsAxThread;

impl std::fmt::Display for NeedsAxThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mahbot's own window is answered on the main thread")
    }
}

impl std::error::Error for NeedsAxThread {}

/// Whether `err` is the main-thread marker (robust to a `.context()` layer).
fn is_ax_thread(err: &anyhow::Error) -> bool {
    err.downcast_ref::<NeedsAxThread>().is_some()
}

/// Raise [`NeedsAxThread`] when `pid` is our own process and this is not the main
/// thread. `pid` must come from the resolved target — for an app element that is a
/// local getter, not an AX exchange.
fn ensure_ax_thread(pid: u32) -> Result<(), anyhow::Error> {
    if pid == std::process::id() && objc2::MainThreadMarker::new().is_none() {
        return Err(anyhow::Error::new(NeedsAxThread));
    }
    Ok(())
}

/// The wait for mahbot's own main thread ran out. Typed so an actuating call can
/// report the possibly-applied request as applied-but-unconfirmed, while a read
/// reports it as the `degraded` failure it is (nothing was read, nothing changed).
#[derive(Debug)]
struct AxBusy(u64);

impl std::fmt::Display for AxBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mahbot's own window did not answer within {}s — its main thread is busy, so the call \
             gave up instead of hanging",
            self.0
        )
    }
}

impl std::error::Error for AxBusy {}

/// A read's rendering of [`AxBusy`]: the taxonomy failure whose guidance is to
/// retry a read (an element action converts the same marker into the middle
/// outcome instead — see [`Backend::act_on_element`]).
fn degraded_read(e: anyhow::Error) -> anyhow::Error {
    match e.downcast_ref::<AxBusy>() {
        Some(busy) => core::taxonomy_error(core::ERR_DEGRADED, format!("{busy}; retry the read")),
        None => e,
    }
}

/// Run one accessibility exchange where the system requires it: on the main
/// thread when `pid` is mahbot itself, on the calling thread otherwise. A main
/// thread that stays busy past [`SELF_AX_WAIT`] gives up as [`AxBusy`] instead of
/// hanging the call; the request itself may still be applied afterwards.
fn on_ax_thread<T: Send + 'static>(
    pid: u32,
    f: impl FnOnce() -> Result<T, anyhow::Error> + Send + 'static,
) -> Result<T, anyhow::Error> {
    if pid != std::process::id() || objc2::MainThreadMarker::new().is_some() {
        return f();
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    dispatch2::DispatchQueue::main().exec_async(move || {
        let result = contained("access to mahbot's own window", f);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(SELF_AX_WAIT) {
        Ok(result) => result,
        Err(_) => Err(anyhow::Error::new(AxBusy(SELF_AX_WAIT.as_secs()))),
    }
}

/// Run an exchange aimed at `target`, on the main thread when it resolves to
/// mahbot's own process. The exchange reports that by raising
/// [`NeedsAxThread`] before it does anything, so the discarded first attempt has
/// performed nothing.
fn with_target<T: Send + 'static>(
    target: &TargetSpec,
    f: impl Fn(&TargetSpec) -> Result<T, anyhow::Error> + Send + 'static,
) -> Result<T, anyhow::Error> {
    match f(target) {
        Err(e) if is_ax_thread(&e) => {
            let owned = target.clone();
            on_ax_thread(std::process::id(), move || f(&owned))
        }
        other => other,
    }
}

// ── AX attribute readers ─────────────────────────────────────────────────

fn attribute_value(el: &AXUIElement, attr: &CFString) -> Result<CFRetained<CFType>, AXError> {
    let mut value: *const CFType = std::ptr::null();
    // SAFETY: `value` points to valid storage the binding fills on success.
    let err = unsafe {
        el.copy_attribute_value(
            attr,
            std::ptr::NonNull::new(&mut value as *mut *const CFType).unwrap(),
        )
    };
    if err != AXError::Success {
        return Err(err);
    }
    // SAFETY: AXUIElementCopyAttributeValue returns a +1 retained pointer (Create
    // rule); null indicates a missing value.
    match std::ptr::NonNull::new(value.cast_mut()) {
        Some(p) => Ok(unsafe { CFRetained::from_raw(p) }),
        None => Err(AXError::NoValue),
    }
}

/// Read an AX string attribute, copying at most `max` characters of it. The
/// framework has already materialised the whole value before this is called, so the
/// cap is on what this copies and keeps: the length decides before a single unit is
/// read out, and at most `max` characters' worth of UTF-16 units are ever copied.
/// The caller sets its own ceiling on the result — a value or a name through
/// [`Truncation::keep_value`]/[`Truncation::keep_name`], a label through
/// [`core::cap_label`].
fn attribute_text(el: &AXUIElement, attr: &CFString, max: usize) -> Option<String> {
    let val = attribute_value(el, attr).ok()?;
    let string = val.downcast_ref::<CFString>()?;
    let length = string.length();
    if length <= 0 {
        return Some(String::new());
    }
    // Two UTF-16 units per character at worst, plus one character past the cap so a
    // string that is merely longer still reads as cut.
    let wanted = length.min(2 * (max as isize + 2));
    let mut units = vec![0u16; wanted as usize];
    // SAFETY: `units` holds exactly `wanted` UTF-16 units, the number of units asked
    // for, and `wanted` is at most the string's own length.
    unsafe {
        string.characters(CFRange::new(0, wanted), units.as_mut_ptr());
    }
    Some(String::from_utf16_lossy(&units))
}

/// A string this backend uses as a label rather than as content — a role, a
/// subrole, an application or window name — kept to [`core::MAX_NAME_CHARS`], the
/// ellipsis marking the cut ([`core::cap_label`]). A label is shown, not matched on,
/// and a window is matched on its frame, so the cap only costs title-only matches.
fn attribute_name(el: &AXUIElement, attr: &CFString) -> Option<String> {
    attribute_text(el, attr, core::MAX_NAME_CHARS).map(|text| core::cap_label(&text))
}

fn attribute_point(el: &AXUIElement, attr: &CFString) -> Option<CGPoint> {
    let val = attribute_value(el, attr).ok()?;
    let axval = val.downcast_ref::<AXValue>()?;
    let mut p = CGPoint::new(0.0, 0.0);
    // SAFETY: `axval` is an AXValue; the destination matches AXValueType::CGPoint.
    let ok = unsafe {
        axval.value(
            AXValueType::CGPoint,
            std::ptr::NonNull::new(&mut p as *mut _ as *mut _).unwrap(),
        )
    };
    ok.then_some(p)
}

fn attribute_size(el: &AXUIElement, attr: &CFString) -> Option<CGSize> {
    let val = attribute_value(el, attr).ok()?;
    let axval = val.downcast_ref::<AXValue>()?;
    let mut s = CGSize::new(0.0, 0.0);
    // SAFETY: `axval` is an AXValue; the destination matches AXValueType::CGSize.
    let ok = unsafe {
        axval.value(
            AXValueType::CGSize,
            std::ptr::NonNull::new(&mut s as *mut _ as *mut _).unwrap(),
        )
    };
    ok.then_some(s)
}

fn attribute_bool(el: &AXUIElement, attr: &CFString) -> Option<bool> {
    let val = attribute_value(el, attr).ok()?;
    Some(val.downcast_ref::<CFBoolean>()?.value())
}

/// The members of `array` that really are `T`. An array arrives with whatever its
/// owner put in it, so each member is checked against the type it is used as; a
/// member of another type is skipped rather than read as a `T`.
fn typed_members<T: ConcreteType>(array: &CFArray) -> Vec<CFRetained<T>> {
    // SAFETY: an array's members are Core Foundation objects, and `CFType` is the
    // root of them all — the cast makes no claim about a member's own type, which
    // is what `downcast` below checks.
    let members: CFRetained<CFArray<CFType>> =
        unsafe { CFRetained::cast_unchecked(array.retain()) };
    members
        .iter()
        .filter_map(|member| member.downcast::<T>().ok())
        .collect()
}

/// Raw (AX-prefixed) action names. The AX error is the caller's to report — a tree
/// walk degrades to no actions, while acting on an element must not turn a failed
/// read into "element has no press/open action".
///
/// `Ok(None)` is the third answer: the system replied with something that is not a
/// list of names — a list whose members are none of them strings — so the caller
/// knows it learned nothing rather than that the element has no actions.
fn raw_action_names(el: &AXUIElement) -> Result<Option<Vec<String>>, AXError> {
    let mut names: *const CFArray = std::ptr::null();
    // SAFETY: `names` points to valid storage the binding fills on success.
    let err = unsafe {
        el.copy_action_names(std::ptr::NonNull::new(&mut names as *mut *const CFArray).unwrap())
    };
    if err != AXError::Success {
        return Err(err);
    }
    // SAFETY: AXUIElementCopyActionNames returns a +1 retained Core Foundation
    // object (Create rule); "Core Foundation object" is all this claims — the
    // `downcast` below checks which kind it really is. Null indicates an element
    // with no actions.
    let Some(raw) = std::ptr::NonNull::new(names.cast_mut()) else {
        return Ok(Some(Vec::new()));
    };
    // SAFETY: as above — `from_raw` takes over the +1 reference, `cast_unchecked`
    // only widens it to the root type, and the type check follows.
    let object: CFRetained<CFType> =
        unsafe { CFRetained::cast_unchecked(CFRetained::from_raw(raw)) };
    // An answer that is not an array is not a list of names, so it taught us
    // nothing — the same conclusion as an array with no usable member.
    let Ok(array) = object.downcast::<CFArray>() else {
        return Ok(None);
    };
    let names: Vec<String> = typed_members::<CFString>(&array)
        .iter()
        .map(ToString::to_string)
        .collect();
    // An empty array really is "no actions"; a non-empty array with nothing usable
    // in it taught us nothing.
    if names.is_empty() && !array.is_empty() {
        return Ok(None);
    }
    Ok(Some(names))
}

fn attribute_children(el: &AXUIElement) -> Vec<CFRetained<AXUIElement>> {
    let Ok(val) = attribute_value(el, &axc(AX_CHILDREN)) else {
        return Vec::new();
    };
    let Ok(array) = val.downcast::<CFArray>() else {
        return Vec::new();
    };
    typed_members::<AXUIElement>(&array)
}

fn normalize_action_name(name: &str) -> String {
    name.strip_prefix("AX").unwrap_or(name).to_lowercase()
}

/// Arms shared by [`ax_error`] and [`ax_window_error`] — identical messages
/// and taxonomy codes. `None` falls through to caller-specific handling.
fn ax_error_shared(err: AXError, ctx: impl std::fmt::Display) -> Option<anyhow::Error> {
    match err {
        AXError::CannotComplete => Some(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("transient AX failure, retry: {ctx}"),
        )),
        AXError::InvalidUIElement => Some(core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("window closed or changed — re-enumerate: {ctx}"),
        )),
        AXError::APIDisabled => Some(permission_error(Grant::Accessibility)),
        _ => None,
    }
}

/// Map an AXError to a taxonomy error per the contract.
fn ax_error(err: AXError, ctx: impl std::fmt::Display) -> anyhow::Error {
    if let Some(mapped) = ax_error_shared(err, &ctx) {
        return mapped;
    }
    match err {
        AXError::AttributeUnsupported
        | AXError::ActionUnsupported
        | AXError::NoValue
        | AXError::NotImplemented
        | AXError::ParameterizedAttributeUnsupported => core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("unsupported accessibility operation: {ctx}"),
        ),
        // Plain (non-taxonomy) errors — not actionable by the model.
        _ => anyhow!("AX error {err:?}: {ctx}"),
    }
}

/// Window-level error mapping (CannotComplete/InvalidUIElement are fatal).
fn ax_window_error(err: AXError, ctx: impl std::fmt::Display) -> anyhow::Error {
    ax_error_shared(err, &ctx).unwrap_or_else(|| {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            format!("no windows enumerated — re-enumerate: {ctx}"),
        )
    })
}

// ── AX tree walk ─────────────────────────────────────────────────────────

const MAX_DEPTH: usize = 25;

/// True when the element is a secure (e.g. password) field. macOS reports the
/// secure marker in either AXRole or AXSubrole — a password field is
/// AXTextField with the marker in AXSubrole — so both are checked. One predicate
/// keeps the value-masking in [`walk_element`] and the SetValue refusal in
/// [`set_element_value`] in agreement.
fn is_secure_field(role: &str, subrole: Option<&str>) -> bool {
    role.to_lowercase().contains("secure")
        || subrole.is_some_and(|s| s.to_lowercase().contains("secure"))
}

/// One node and its subtree, read within `budget`. `Ok(None)` is a node not read
/// because the node ceiling was reached; `Err` is the call's budget spent, which is
/// the only failure a walk raises — a refused or timed-out attribute stays the tolerant
/// `None`/empty it has always been, and the walk goes on.
fn walk_element(
    element: &AXUIElement,
    origin: (f64, f64),
    depth: usize,
    handles: &mut Vec<CFRetained<AXUIElement>>,
    nodes: &mut usize,
    truncation: &mut Truncation,
    budget: &core::Budget,
) -> anyhow::Result<Option<UiNode>> {
    budget.check()?;
    if *nodes >= core::MAX_TREE_NODES {
        truncation.hit_node_ceiling(core::MAX_TREE_NODES);
        return Ok(None);
    }
    *nodes += 1;
    handles.push(element.retain());

    // Each group below is one or two AX exchanges, and the budget is re-checked
    // between them: a target that stalls across a node's own reads must not spend the
    // whole walk's budget inside that node.
    budget.check()?;
    let role = attribute_name(element, &axc(AX_ROLE)).unwrap_or_else(|| "unknown".to_string());
    // Never read a value out of a secure element. A password field reports
    // AXRole="AXTextField" with the secure marker in AXSubrole, so check the
    // subrole (or role) for "secure" before pulling the value.
    let subrole = attribute_name(element, &axc(AX_SUBROLE));
    budget.check()?;
    let value = if is_secure_field(&role, subrole.as_deref()) {
        None
    } else {
        attribute_text(element, &axc(AX_VALUE), core::MAX_VALUE_CHARS)
            .map(|value| truncation.keep_value(&value))
    };
    budget.check()?;
    let frame = match (
        attribute_point(element, &axc(AX_POSITION)),
        attribute_size(element, &axc(AX_SIZE)),
    ) {
        (Some(p), Some(s)) => Some((p.x - origin.0, p.y - origin.1, s.width, s.height)),
        _ => None,
    };
    budget.check()?;
    // A failed or unusable read only leaves the node's actions absent: the walk
    // degrades, acting on the element is what must not.
    let actions = raw_action_names(element)
        .ok()
        .flatten()
        .unwrap_or_default()
        .iter()
        .map(|a| normalize_action_name(a))
        .collect();
    budget.check()?;
    let focused = attribute_bool(element, &axc(AX_FOCUSED)).unwrap_or(false);

    let mut children = Vec::new();
    if depth < MAX_DEPTH {
        budget.check()?;
        for child in attribute_children(element) {
            if let Some(child_node) = walk_element(
                &child,
                origin,
                depth + 1,
                handles,
                nodes,
                truncation,
                budget,
            )? {
                children.push(child_node);
            }
        }
    } else if !attribute_children(element).is_empty() {
        // The node's subtree is not read. Only a node that has one is a cut, so a
        // leaf at the ceiling is not reported as a truncation.
        truncation.cut_depth(MAX_DEPTH);
    }
    Ok(Some(UiNode {
        role,
        name: attribute_text(element, &axc(AX_TITLE), core::MAX_NAME_CHARS)
            .map(|name| truncation.keep_name(&name)),
        value,
        frame,
        actions,
        focused,
        children,
    }))
}

/// Build the accessibility tree, returning the root node, a pre-order-aligned
/// `Vec` of platform handles (used by `act_on_element` to map a matched node back
/// to its element) and the ceilings the walk hit.
fn build_tree(
    root: &AXUIElement,
    origin: (f64, f64),
    budget: &core::Budget,
) -> anyhow::Result<(UiNode, Vec<CFRetained<AXUIElement>>, Truncation)> {
    let mut handles = Vec::new();
    let mut nodes = 0usize;
    let mut truncation = Truncation::default();
    let node = walk_element(
        root,
        origin,
        0,
        &mut handles,
        &mut nodes,
        &mut truncation,
        budget,
    )?
    .ok_or_else(|| {
        // The root is the first node, so only its own reads can leave it unread — and an
        // empty tree would present "nothing was read" as a window with no content.
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "{} did not answer for its accessibility root, so nothing was read — retry, \
                 or use screenshot instead",
                budget.target()
            ),
        )
    })?;
    Ok((node, handles, truncation))
}

/// The application a target points at: the name the target carries, else the resolved
/// application's own (one exchange when the target names none).
fn resolved_app_name(target: &TargetSpec, app: &AXUIElement) -> Result<String, anyhow::Error> {
    match target.app() {
        Some((_, name)) => Ok(name.to_string()),
        None => Ok(app_display_name(app_pid(app)?, app)),
    }
}

/// The label of the surface a call resolved, for the failures that follow the reads that
/// name it — the same rule [`core::name_surface`] applies where a budget survives to be
/// pointed at it. The title is passed in, so a caller that already read it does not read it
/// again from a target that may be the hung one.
fn resolved_surface_label(
    target: &TargetSpec,
    app: &AXUIElement,
    title: Option<&str>,
) -> Result<String, anyhow::Error> {
    Ok(core::surface_label(&resolved_app_name(target, app)?, title))
}

fn observe_blocking(target: &TargetSpec) -> Result<Observation, anyhow::Error> {
    // One clock for the whole call, resolution and the reads that name the surface
    // included, so the budget the model is told about is the one the call pays.
    let mut budget = core::Budget::new(core::OBSERVE_BUDGET, "observation", target.describe());
    let (window, app) = resolve_window(target, &budget)?;
    budget.check()?;
    // The identity is read before the reads that can go unanswered: a failure has to
    // name the window, and a title is an exchange it can no longer afford then.
    let app_name = resolved_app_name(target, &app)?;
    // The header title is a name, not a label: read past the cap so a cut is counted.
    let title = attribute_text(&window, &axc(AX_TITLE), core::MAX_NAME_CHARS);
    let label = core::name_surface(&mut budget, &app_name, title.as_deref());
    // The header reports the window's frame, and a minimized window's is the one it
    // had before it was minimized: refuse rather than describe a surface that is not
    // on screen.
    refuse_minimized(&window)?;
    budget.check()?;
    let origin =
        window_origin(&window).map_err(|_| core::unanswered(&label, "for its position"))?;
    let surface =
        window_frame(&window, &budget).map_err(|_| core::unanswered(&label, "for its frame"))?;
    budget.check()?;
    let (root, _handles, mut truncation) = build_tree(&window, origin, &budget)?;
    // The title is a name like any node's, so a cut one is counted in the same header
    // ceiling instead of being shortened silently.
    let title = title.map(|title| truncation.keep_name(&title));
    Ok(Observation {
        truncation,
        app_name,
        window_title: title,
        surface,
        root,
    })
}

// ── Window identity ─────────────────────────────────────────────────────

/// A retained AX window element, addressed by the token that names it in the
/// target registry. The element itself is the identity: it answers with the
/// window's live title, frame and process, so nothing here has to re-find a
/// window by title or by its place in an enumeration. The owner is the agent
/// that enumerated it, so the cap can drop one agent's handles only.
struct HeldWindow {
    element: CFRetained<AXUIElement>,
    owner: String,
}

// SAFETY: an AX exchange is not thread-safe, so an element is only ever used on
// the thread that owns it — the tool serializes every call through one action
// lock, and exchanges aimed at our own windows run on the main queue under a
// bounded wait. The wrapper makes the map a static; it does not make concurrent
// AX calls safe, and none are performed.
unsafe impl Send for HeldWindow {}

/// Held windows by token. Ordered by token, so the first entry is the oldest.
static HELD_WINDOWS: LazyLock<Mutex<BTreeMap<u64, HeldWindow>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

static NEXT_WINDOW_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Cap on windows retained per owning agent — the same number the target registry
/// keeps ([`core::MAX_STORED_TARGETS`]) — so a live id always has a handle to
/// resolve through. Beyond it that agent's oldest are dropped; an id whose token
/// went with them reads as a window that can no longer be identified, which is an
/// honest error rather than a wrong window.
const MAX_HELD_WINDOWS: usize = core::MAX_STORED_TARGETS;

/// The identity of `window`: the token it already holds, else a fresh one.
/// `CFEqual` compares accessibility elements by the object they address, so the
/// same window is recognised across enumerations even though every enumeration
/// hands back a new element reference. Tokens are per agent, so one agent's cleanup
/// can never drop a handle another agent's id resolves through. An element id an
/// application recycles after its window closed is the one case this cannot tell
/// apart; catching it would need a liveness probe per held window on every listing.
///
/// `owner` is read by the caller — never inside an exchange routed to the main
/// queue, where this thread's task-locals are absent.
fn window_key(window: &AXUIElement, owner: &str) -> WindowKey {
    let mut held = HELD_WINDOWS.lock().unwrap_poison();
    let known = held
        .iter()
        .find(|(_, existing)| existing.owner == owner && &*existing.element == window)
        .map(|(token, _)| *token);
    let token = if let Some(token) = known {
        token
    } else {
        let token = NEXT_WINDOW_TOKEN.fetch_add(1, Ordering::Relaxed);
        held.insert(
            token,
            HeldWindow {
                element: window.retain(),
                owner: owner.to_string(),
            },
        );
        // Tokens increase monotonically, so the owner's first tokens are its
        // oldest. Collected first: removing while iterating would hold a borrow.
        let mine: Vec<u64> = held
            .iter()
            .filter(|(_, entry)| entry.owner == owner)
            .map(|(token, _)| *token)
            .collect();
        for token in mine
            .iter()
            .take(mine.len().saturating_sub(MAX_HELD_WINDOWS))
        {
            held.remove(token);
        }
        token
    };
    WindowKey::new(format!("ax:{token}"))
}

/// The window a stored key names, refusing rather than substituting another one.
/// A token that no longer resolves, or an element whose window is gone, is
/// `surface-gone`: the call is about a window this process can no longer
/// identify.
fn window_by_key(
    key: &WindowKey,
    app_name: &str,
) -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    let token = key
        .as_str()
        .strip_prefix("ax:")
        .and_then(|raw| raw.parse::<u64>().ok())
        .ok_or_else(|| core::surface_gone(app_name))?;
    let window = {
        let held = HELD_WINDOWS.lock().unwrap_poison();
        held.get(&token)
            .map(|held| held.element.retain())
            .ok_or_else(|| core::surface_gone(app_name))?
    };
    // A retained element outlives its window: reading it is what tells the two apart.
    // Only the framework's own "no longer an element" answer is a gone window — a read
    // that timed out is a live window that stopped answering, and calling that
    // `surface-gone` would send the model re-enumerating a window that is still there.
    match attribute_value(&window, &axc(AX_ROLE)) {
        Ok(_) => Ok(window),
        Err(AXError::InvalidUIElement) => Err(core::surface_gone(app_name)),
        Err(_) => Err(core::unanswered(
            &core::surface_label(app_name, None),
            "for its role",
        )),
    }
}

/// Drop every window `agent` holds: its ids are gone with its registry entries.
pub(crate) fn forget_agent_windows(agent: &str) {
    let mut held = HELD_WINDOWS.lock().unwrap_poison();
    held.retain(|_, entry| entry.owner != agent);
}

/// Refuse a minimized window: it has no visible pixels, and its frame is the one
/// it had before it was minimized.
fn refuse_minimized(window: &AXUIElement) -> Result<(), anyhow::Error> {
    if !attribute_bool(window, &axc(AX_MINIMIZED)).unwrap_or(false) {
        return Ok(());
    }
    Err(core::minimized_window(
        &window_title(window).unwrap_or_default(),
    ))
}

// ── Window resolution ────────────────────────────────────────────────────

/// The frontmost application's AX element, resolved through `NSWorkspace`.
///
/// The system-wide AX element's own focus attributes are deliberately not used:
/// a process that holds its grant through the terminal gets
/// `kAXErrorCannotComplete` from them, which would make the DEFAULT surface (a
/// call with no explicit target) fail while every per-application query works.
fn frontmost_app() -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    let app = NSWorkspace::sharedWorkspace()
        .frontmostApplication()
        .ok_or_else(|| {
            core::taxonomy_error(
                core::ERR_NOT_MATCHED,
                "no frontmost application — enumerate apps/windows first",
            )
        })?;
    // SAFETY: AXUIElementCreateApplication never returns null; a pid that went
    // away surfaces as InvalidUIElement on the first attribute read.
    Ok(unsafe { AXUIElement::new_application(app.processIdentifier()) })
}

fn focused_window_of_app(app: &AXUIElement) -> Option<CFRetained<AXUIElement>> {
    attribute_value(app, &axc(AX_FOCUSED_WINDOW))
        .ok()?
        .downcast::<AXUIElement>()
        .ok()
}

fn running_app_pid(name: &str) -> Option<u32> {
    let needle = name.to_lowercase();
    NSWorkspace::sharedWorkspace()
        .runningApplications()
        .iter()
        .find(|app| {
            app.activationPolicy() == NSApplicationActivationPolicy::Regular
                && app
                    .localizedName()
                    .is_some_and(|n| n.to_string().to_lowercase() == needle)
        })
        .map(|app| app.processIdentifier() as u32)
}

/// App display name for a pid: the workspace's authoritative localized name (the
/// same source `list_apps` uses) with an AXTitle fallback when the running-app
/// lookup misses. An AXTitle read on the AXApplication element is usually empty,
/// so relying on it alone yields an app="".
fn app_display_name(pid: u32, app: &AXUIElement) -> String {
    NSWorkspace::sharedWorkspace()
        .runningApplications()
        .iter()
        .find(|a| a.processIdentifier() == pid as i32)
        .and_then(|a| a.localizedName().map(|n| n.to_string()))
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| attribute_name(app, &axc(AX_TITLE)).unwrap_or_default())
}

fn app_element(pid: Option<u32>, name: &str) -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    let pid = match pid {
        Some(p) => p,
        None => running_app_pid(name).ok_or_else(|| core::surface_gone(name))?,
    };
    // SAFETY: `pid` is a running app (or stale); AXUIElementCreateApplication
    // always returns a non-null element — attribute reads surface InvalidUIElement.
    Ok(unsafe { AXUIElement::new_application(pid as libc::pid_t) })
}

/// An application's windows. A target that stopped answering is named as such: the
/// generic "transient AX failure, retry" would send the model re-enumerating a hung
/// application instead of reporting it.
fn read_windows(
    app: &AXUIElement,
    app_name: &str,
) -> Result<Vec<CFRetained<AXUIElement>>, anyhow::Error> {
    let val = attribute_value(app, &axc(AX_WINDOWS)).map_err(|e| {
        if e == AXError::CannotComplete {
            core::unanswered(&core::surface_label(app_name, None), "for its window list")
        } else {
            ax_window_error(e, "enumerate windows")
        }
    })?;
    let array: CFRetained<CFArray> = val
        .downcast::<CFArray>()
        .map_err(|_| anyhow!("AXWindows was not a CFArray"))?;
    Ok(typed_members::<AXUIElement>(&array))
}

/// One application's listed windows: the AX window array plus each window's frame,
/// title and identity. A window with neither a title nor a frame is not a surface
/// and is dropped. The budget is checked before the window list and before every
/// window, so an application that stalls is caught at the check after its stall.
///
/// The listing is the one path that knows the target process without resolving a
/// `TargetSpec`, so it calls [`on_ax_thread`] itself with the app's pid. `owner` is
/// the enumerating agent's key, read before the exchange for the same reason.
fn windows_of_app(
    pid: u32,
    app_name: &str,
    owner: &str,
    budget: &core::Budget,
) -> Result<Vec<WindowInfo>, anyhow::Error> {
    // SAFETY: `pid` is a running app (or stale); AXUIElementCreateApplication
    // always returns a non-null element — attribute reads surface InvalidUIElement.
    let app = unsafe { AXUIElement::new_application(pid as libc::pid_t) };
    budget.check()?;
    let windows = read_windows(&app, app_name)?;
    let mut out = Vec::new();
    for window in &windows {
        budget.check()?;
        let frame = match window_frame(window, budget) {
            Ok(frame) => frame,
            // A refused frame loses this window; a spent share ends this application's
            // answer instead, so the window is never dropped as if it had none.
            Err(e) if budget.expired() => return Err(e),
            Err(_) => continue,
        };
        // A window costs its frame and its title: a stall on either is caught here.
        budget.check()?;
        let title = window_title(window).unwrap_or_default();
        if title.is_empty() && (frame.width <= 0.0 || frame.height <= 0.0) {
            continue;
        }
        out.push(WindowInfo {
            app_pid: Some(pid),
            app_name: app_name.to_string(),
            title,
            key: window_key(window, owner),
            surface: frame,
        });
    }
    Ok(out)
}

/// Resolve a target to its AX window and application element.
///
/// The only place a target becomes a platform handle, and therefore the one place
/// the "our own process off the main thread" check belongs ([`ensure_ax_thread`]):
/// it runs after the app element exists but before the first AX exchange, so a
/// worker attempt stops here with nothing performed. An app element's pid is a
/// local getter, not an exchange.
fn resolve_window(
    target: &TargetSpec,
    budget: &core::Budget,
) -> Result<(CFRetained<AXUIElement>, CFRetained<AXUIElement>), anyhow::Error> {
    budget.check()?;
    match target {
        TargetSpec::Focused => {
            let app = frontmost_app()?;
            ensure_ax_thread(app_pid(&app)?)?;
            // Strictly the frontmost app's focused window: a coordinate action on
            // a fallback surface would land on whatever is visible, which may not
            // be that window (another Space, a minimized window).
            let Some(window) = focused_window_of_app(&app) else {
                // Both leave AXFocusedWindow unset, and only a live application's answer
                // says it has no focused window: one that does not answer at all must be
                // named as such instead.
                return Err(match attribute_value(&app, &axc(AX_ROLE)) {
                    Ok(_) => core::taxonomy_error(
                        core::ERR_NOT_MATCHED,
                        "no focused window (desktop frontmost/screensaver) — use \
                         'apps'/'windows' to pick a surface",
                    ),
                    Err(_) => core::unanswered(
                        &format!(
                            "the frontmost application ({})",
                            app_display_name(app_pid(&app)?, &app)
                        ),
                        "for its focused window",
                    ),
                });
            };
            Ok((window, app))
        }
        TargetSpec::App {
            app_pid: pinned,
            app_name,
        } => {
            // An a-id names the application's own focused window and is
            // deliberately re-resolved per call, so it follows that window.
            let app = app_element(*pinned, app_name)?;
            let pid = app_pid(&app)?;
            ensure_ax_thread(pid)?;
            budget.check()?;
            let window = match focused_window_of_app(&app) {
                Some(window) => window,
                // A read that failed is the application's answer, not an application
                // without a window: only a successful read that found none means that.
                None => read_windows(&app, app_name)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| core::surface_gone(app_name))?,
            };
            Ok((window, app))
        }
        TargetSpec::Window {
            app_pid: pinned,
            app_name,
            key,
            ..
        } => {
            // A w-id names one window: the key resolves that window or fails,
            // never another one of the same application.
            let app = app_element(*pinned, app_name)?;
            ensure_ax_thread(app_pid(&app)?)?;
            Ok((window_by_key(key, app_name)?, app))
        }
        TargetSpec::Screen => Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "observe targets an accessibility surface — use screenshot for the screen",
        )),
    }
}

fn window_origin(window: &AXUIElement) -> Result<(f64, f64), anyhow::Error> {
    let p = attribute_point(window, &axc(AX_POSITION))
        .ok_or_else(|| anyhow!("window position unavailable"))?;
    Ok((p.x, p.y))
}

/// A window's frame: position and size, two AX reads, each charged to `budget` so a target
/// that stalls between them does not outlive the call by a whole extra read.
fn window_frame(
    window: &AXUIElement,
    budget: &core::Budget,
) -> Result<SurfaceGeometry, anyhow::Error> {
    budget.check()?;
    let p = attribute_point(window, &axc(AX_POSITION))
        .ok_or_else(|| anyhow!("window position unavailable"))?;
    budget.check()?;
    let s =
        attribute_size(window, &axc(AX_SIZE)).ok_or_else(|| anyhow!("window size unavailable"))?;
    Ok(SurfaceGeometry {
        x: p.x,
        y: p.y,
        width: s.width,
        height: s.height,
    })
}

fn app_pid(app: &AXUIElement) -> Result<u32, anyhow::Error> {
    let mut pid: libc::pid_t = 0;
    // SAFETY: `pid` points to valid storage the binding fills on success.
    let err = unsafe { app.pid(std::ptr::NonNull::new(&mut pid).unwrap()) };
    if err != AXError::Success {
        return Err(ax_error(err, "read application pid"));
    }
    Ok(pid as u32)
}

// ── Raw input (CGEvent) ──────────────────────────────────────────────────

fn cg_mouse_button(button: MouseButton) -> CGMouseButton {
    match button {
        MouseButton::Left => CGMouseButton::Left,
        MouseButton::Right => CGMouseButton::Right,
        MouseButton::Middle => CGMouseButton::Center,
    }
}

fn cg_type_for(button: MouseButton, down: bool) -> CGEventType {
    match (button, down) {
        (MouseButton::Left, true) => CGEventType::LeftMouseDown,
        (MouseButton::Left, false) => CGEventType::LeftMouseUp,
        (MouseButton::Right, true) => CGEventType::RightMouseDown,
        (MouseButton::Right, false) => CGEventType::RightMouseUp,
        (MouseButton::Middle, true) => CGEventType::OtherMouseDown,
        (MouseButton::Middle, false) => CGEventType::OtherMouseUp,
    }
}

fn cg_flags(modifiers: &[Modifier]) -> CGEventFlags {
    modifiers.iter().fold(CGEventFlags::empty(), |acc, m| {
        acc | match m {
            Modifier::Cmd => CGEventFlags::MaskCommand,
            Modifier::Ctrl => CGEventFlags::MaskControl,
            Modifier::Alt => CGEventFlags::MaskAlternate,
            Modifier::Shift => CGEventFlags::MaskShift,
        }
    })
}

fn post_event(event: &CGEvent) {
    CGEvent::post(CGEventTapLocation::HIDEventTap, Some(event));
}

fn source() -> Option<CFRetained<CGEventSource>> {
    CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
}

/// A mouse event of `kind` at `point`. A failed creation is an honest error: an
/// action that posts nothing must never report success.
fn mouse_event(
    source: Option<&CGEventSource>,
    kind: CGEventType,
    point: CGPoint,
    button: CGMouseButton,
) -> Result<CFRetained<CGEvent>, anyhow::Error> {
    CGEvent::new_mouse_event(source, kind, point, button)
        .ok_or_else(|| anyhow!("failed to create a mouse event"))
}

/// A keyboard event for `code`; see [`mouse_event`] on the failure rule.
fn keyboard_event(
    source: Option<&CGEventSource>,
    code: u16,
    down: bool,
) -> Result<CFRetained<CGEvent>, anyhow::Error> {
    CGEvent::new_keyboard_event(source, code, down)
        .ok_or_else(|| anyhow!("failed to create a keyboard event"))
}

/// Every event of the gesture is created before any is posted, so an event that
/// cannot be made leaves nothing half-done: nothing was delivered, and the call is
/// a plain failure rather than a click whose release never went out.
fn click(
    point: CGPoint,
    button: MouseButton,
    double: bool,
    modifiers: &[Modifier],
) -> Result<Impact, anyhow::Error> {
    let src = source();
    let flags = cg_flags(modifiers);
    // Both halves of a double click carry click-state 2.
    let state = if double { 2 } else { 1 };
    let mut events = Vec::with_capacity(if double { 4 } else { 2 });
    for _ in 0..if double { 2 } else { 1 } {
        for down in [true, false] {
            let ev = mouse_event(
                src.as_deref(),
                cg_type_for(button, down),
                point,
                cg_mouse_button(button),
            )?;
            CGEvent::set_flags(Some(&ev), flags);
            CGEvent::set_integer_value_field(Some(&ev), CGEventField::MouseEventClickState, state);
            events.push(ev);
        }
    }
    for ev in &events {
        post_event(ev);
    }
    Ok(Impact::Done)
}

fn sleep_millis(ms: u64) {
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

fn ascii_key(c: char) -> Option<i64> {
    // kVK_ANSI_* for the 26 letters and top-row digits.
    Some(match c {
        '0' => 29,
        '1' => 18,
        '2' => 19,
        '3' => 20,
        '4' => 21,
        '5' => 23,
        '6' => 22,
        '7' => 26,
        '8' => 28,
        '9' => 25,
        'a' => 0,
        's' => 1,
        'd' => 2,
        'f' => 3,
        'h' => 4,
        'g' => 5,
        'z' => 6,
        'x' => 7,
        'c' => 8,
        'v' => 9,
        'b' => 11,
        'q' => 12,
        'w' => 13,
        'e' => 14,
        'r' => 15,
        'y' => 16,
        't' => 17,
        'i' => 34,
        'o' => 31,
        'p' => 35,
        'l' => 37,
        'j' => 38,
        'k' => 40,
        'n' => 45,
        'm' => 46,
        'u' => 32,
        _ => return None,
    })
}

/// Key name → macOS virtual keycode. Unknown names return `None`.
fn key_code(key: &str) -> Option<i64> {
    let k = key.to_lowercase();
    if k.len() == 1
        && let Some(c) = k.chars().next().and_then(ascii_key)
    {
        return Some(c);
    }
    Some(match k.as_str() {
        "return" | "enter" => 36,
        "tab" => 48,
        "space" => 49,
        "delete" => 51, // backspace
        "forwarddelete" => 117,
        "escape" => 53,
        "home" => 115,
        "end" => 119,
        "pageup" => 116,
        "pagedown" => 121,
        "up" => 126,
        "down" => 125,
        "left" => 123,
        "right" => 124,
        "f1" => 122,
        "f2" => 120,
        "f3" => 99,
        "f4" => 118,
        "f5" => 96,
        "f6" => 97,
        "f7" => 98,
        "f8" => 100,
        "f9" => 101,
        "f10" => 109,
        "f11" => 103,
        "f12" => 111,
        "minus" => 27,
        "equal" => 24,
        "comma" => 43,
        "period" => 47,
        "slash" => 44,
        "semicolon" => 41,
        "quote" => 39,
        "backslash" => 42,
        "backtick" => 50,
        "leftbracket" => 33,
        "rightbracket" => 30,
        "capslock" => 57,
        _ => return None,
    })
}

const KNOWN_KEYS: &str = "return, enter, tab, space, delete, forwarddelete, escape, home, end, \
     pageup, pagedown, up, down, left, right, f1..f12, minus, equal, comma, period, slash, \
     semicolon, quote, backslash, backtick, leftbracket, rightbracket, capslock, a-z, 0-9";

/// The virtual keycode for a parsed key name, with the one error message every
/// caller — the argument pre-check included — reports.
fn lookup_key(key: &str) -> Result<i64, anyhow::Error> {
    key_code(key).ok_or_else(|| anyhow!("unknown key '{key}' — supported keys: {KNOWN_KEYS}"))
}

/// One key tap: both events are built before either is posted, so a build failure
/// delivers nothing.
fn press_key(key: &str) -> Result<(), anyhow::Error> {
    let code = lookup_key(key)?;
    let src = source();
    let down = keyboard_event(src.as_deref(), code as u16, true)?;
    let up = keyboard_event(src.as_deref(), code as u16, false)?;
    post_event(&down);
    post_event(&up);
    Ok(())
}

/// Typing is one character at a time, so a failure part-way through leaves the
/// earlier characters typed: that is the middle outcome, not a failure. A failure
/// on the FIRST character typed nothing and stays a failure.
fn type_text(text: &str) -> Result<Impact, anyhow::Error> {
    fn type_char(src: Option<&CGEventSource>, ch: char) -> Result<(), anyhow::Error> {
        let mut buf = [0u16; 2];
        let encoded = ch.encode_utf16(&mut buf);
        // Both events are built before either is posted, as in `press_key`: a build
        // failure must not leave the character half-posted while it is counted untyped.
        let down = keyboard_event(src, 0, true)?;
        let up = keyboard_event(src, 0, false)?;
        for event in [&down, &up] {
            // SAFETY: `encoded` is a valid UTF-16 buffer.
            unsafe {
                CGEvent::keyboard_set_unicode_string(
                    Some(event),
                    encoded.len() as u64,
                    encoded.as_ptr(),
                )
            };
        }
        post_event(&down);
        post_event(&up);
        Ok(())
    }

    let src = source();
    let total = text.chars().count();
    for (typed, ch) in text.chars().enumerate() {
        let made = if ch == '\n' {
            press_key("return")
        } else {
            type_char(src.as_deref(), ch)
        };
        match made {
            Ok(()) => sleep_millis(1),
            Err(e) if typed == 0 => return Err(e),
            Err(e) => {
                return Ok(Impact::Unconfirmed(format!(
                    "{typed} of {total} characters were typed before the next keyboard event \
                     could not be created ({e})"
                )));
            }
        }
    }
    Ok(Impact::Done)
}

fn key_chord(chord: &str) -> Result<Impact, anyhow::Error> {
    let (modifiers, key) = core::parse_key_chord(chord)?;
    let code = lookup_key(&key)?;
    let src = source();
    let flags = cg_flags(&modifiers);
    let down = keyboard_event(src.as_deref(), code as u16, true)?;
    let up = keyboard_event(src.as_deref(), code as u16, false)?;
    CGEvent::set_flags(Some(&down), flags);
    // Clear flags on key-up so the modifier doesn't stick.
    CGEvent::set_flags(Some(&up), CGEventFlags::empty());
    post_event(&down);
    post_event(&up);
    Ok(Impact::Done)
}

/// Scroll: wheel 1 = vertical (down positive), wheel 2 = horizontal (right
/// positive). CGEvent deltas use +ve for down/right on macOS.
fn scroll(
    point: Option<CGPoint>,
    direction: ScrollDirection,
    amount: u32,
) -> Result<Impact, anyhow::Error> {
    let src = source();
    // The tool bounds the amount before it gets here; the conversion is still total so
    // no magnitude can wrap the direction.
    let ticks = i32::try_from(amount).unwrap_or(i32::MAX);
    let (w1, w2) = match direction {
        ScrollDirection::Down => (ticks, 0),
        ScrollDirection::Up => (-ticks, 0),
        ScrollDirection::Right => (0, ticks),
        ScrollDirection::Left => (0, -ticks),
    };
    let ev =
        CGEvent::new_scroll_wheel_event2(src.as_deref(), CGScrollEventUnit::Line, 2, w1, w2, 0)
            .ok_or_else(|| anyhow!("failed to create a scroll event"))?;
    if let Some(p) = point {
        CGEvent::set_location(Some(&ev), p);
    }
    post_event(&ev);
    Ok(Impact::Done)
}

/// The whole gesture — down, interpolation, up — is created before any of it is
/// posted, so a gesture that cannot be built delivers nothing instead of leaving
/// the button down mid-drag.
fn drag(from: CGPoint, to: CGPoint) -> Result<Impact, anyhow::Error> {
    let src = source();
    let mut events = Vec::with_capacity(14);
    events.push(mouse_event(
        src.as_deref(),
        CGEventType::LeftMouseDown,
        from,
        CGMouseButton::Left,
    )?);
    for i in 1..=12 {
        let t = i as f64 / 12.0;
        events.push(mouse_event(
            src.as_deref(),
            CGEventType::LeftMouseDragged,
            CGPoint::new(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t),
            CGMouseButton::Left,
        )?);
    }
    events.push(mouse_event(
        src.as_deref(),
        CGEventType::LeftMouseUp,
        to,
        CGMouseButton::Left,
    )?);
    for event in &events {
        post_event(event);
        sleep_millis(2);
    }
    Ok(Impact::Done)
}

// ── Surface geometry ─────────────────────────────────────────────────────

fn active_display_ids() -> Vec<CGDirectDisplayID> {
    let mut displays = [0u32; 32];
    let mut count = 0u32;
    // SAFETY: `displays` and `count` are valid storage the binding fills.
    let err =
        unsafe { CGGetActiveDisplayList(displays.len() as u32, displays.as_mut_ptr(), &mut count) };
    if err != CGError::Success {
        return Vec::new();
    }
    displays[..count as usize].to_vec()
}

fn screen_surface() -> Result<SurfaceGeometry, anyhow::Error> {
    let ids = active_display_ids();
    if ids.is_empty() {
        anyhow::bail!("no active displays found");
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for id in ids {
        let b = CGDisplayBounds(id);
        min_x = min_x.min(b.origin.x);
        min_y = min_y.min(b.origin.y);
        max_x = max_x.max(b.origin.x + b.size.width);
        max_y = max_y.max(b.origin.y + b.size.height);
    }
    Ok(SurfaceGeometry {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    })
}

/// The target's surface. A window frame is an AX read, so the grant gate belongs
/// to [`surface_geometry_with`], which keeps it on the calling thread: a missing
/// grant is requested there, never from the main queue.
fn surface_geometry_blocking(
    target: &TargetSpec,
    budget: &core::Budget,
) -> Result<SurfaceGeometry, anyhow::Error> {
    if matches!(target, TargetSpec::Screen) {
        return screen_surface();
    }
    // Coordinate input maps normalized points through this frame, so a
    // pre-minimization frame must be refused here too.
    let (window, app) = resolve_window(target, budget)?;
    budget.check()?;
    let label = resolved_surface_label(target, &app, window_title(&window).as_deref())?;
    budget.check()?;
    refuse_minimized(&window)?;
    // `window_frame` checks the budget itself before each of its two reads.
    window_frame(&window, budget).map_err(|_| core::unanswered(&label, "for its frame"))
}

/// [`surface_geometry_blocking`] with the grant gate hoisted and the read routed
/// to the thread the system answers a self target on. The budget is owned so it can
/// cross to that thread; a caller that already holds one passes a [`core::Budget`]
/// clone, so the lookup cannot outlive the call it belongs to.
fn surface_geometry_with(
    target: &TargetSpec,
    budget: core::Budget,
) -> Result<SurfaceGeometry, anyhow::Error> {
    if !matches!(target, TargetSpec::Screen) {
        ensure(Grant::Accessibility)?;
    }
    with_target(target, move |target| {
        surface_geometry_blocking(target, &budget)
    })
}

/// Raise a window so input aimed at it lands there; `false` when it exposes no
/// usable raise action.
fn raise_window(window: &AXUIElement) -> Result<bool, anyhow::Error> {
    let Some(names) = raw_action_names(window).ok().flatten() else {
        return Ok(false);
    };
    if !names.iter().any(|a| a == AX_RAISE) {
        return Ok(false);
    }
    // SAFETY: AXRaise was verified to exist on the window.
    let err = unsafe { window.perform_action(&axc(AX_RAISE)) };
    if err != AXError::Success {
        return Err(ax_error(err, "raise window"));
    }
    Ok(true)
}

/// Best-effort raise for input that is not pinned to a window (a coordinate
/// click, a wheel at a point on `screen`): a refusal is not a failure there,
/// because the input is allowed to land wherever its point is. Returns whether a
/// raise was performed, so the caller settles only when there is something to
/// settle. Pinned input uses [`raise_window`] plus [`ensure_window_receives`]
/// instead.
fn activate_window(target: &TargetSpec, budget: &core::Budget) -> Result<bool, anyhow::Error> {
    let (TargetSpec::Focused | TargetSpec::App { .. } | TargetSpec::Window { .. }) = target else {
        return Ok(false);
    };
    let (window, _app) = match resolve_window(target, budget) {
        Ok(resolved) => resolved,
        // Nothing to raise, but the main-thread marker must reach [`with_target`].
        Err(e) if !is_ax_thread(&e) => return Ok(false),
        Err(e) => return Err(e),
    };
    raise_window(&window)
}

/// Whether `window` is where the window server delivers input right now: its
/// application is frontmost and the window is that application's focused one.
fn window_receives_input(window: &AXUIElement) -> bool {
    let Ok(app) = frontmost_app() else {
        return false;
    };
    focused_window_of_app(&app).is_some_and(|focused| &*focused == window)
}

/// The verification half of pinned input: the window input was aimed at must be
/// the one the window server would deliver to. A raise completes asynchronously,
/// so the check is retried over the shared settle budget — the pauses run here, on
/// the calling thread, and only each accessibility read is routed to the thread
/// that answers it.
fn ensure_window_receives(target: &TargetSpec, budget: &core::Budget) -> Result<(), anyhow::Error> {
    for attempt in 0..=core::RAISE_SETTLE_STEPS {
        if attempt > 0 {
            sleep_millis(core::RAISE_SETTLE_MS);
        }
        budget.check()?;
        let probe = budget.clone();
        let receives = with_target(target, move |target| {
            let (window, _app) = resolve_window(target, &probe)?;
            Ok(window_receives_input(&window))
        })?;
        if receives {
            return Ok(());
        }
    }
    Err(core::not_brought_forward(target.describe()))
}

/// Reject an argument mistake before any grant is requested: a call that can
/// never succeed must not raise a system permission dialog.
fn check_input_arguments(input: &RawInput) -> Result<(), anyhow::Error> {
    core::check_raw_input(input)?;
    // The chord's key is resolved here, so an unknown key is refused before any event.
    match input {
        RawInput::KeyChord { chord } => lookup_key(&core::parse_key_chord(chord)?.1).map(|_| ()),
        _ => Ok(()),
    }
}

fn raw_input_blocking(target: &TargetSpec, input: &RawInput) -> Result<Impact, anyhow::Error> {
    let pinned = core::pinned_input(target, input);
    // A chord goes to whatever already has focus, so nothing is raised for it.
    let raises = !matches!(input, RawInput::KeyChord { .. });
    // One lookup budget for the whole call: raising a window, proving it receives the
    // input and reading the frame the input is mapped through all fit in it.
    let lookup = core::Budget::new(core::LIST_BUDGET, "surface lookup", target.describe());
    let geometry = lookup.clone();
    if pinned {
        let raise = lookup.clone();
        with_target(target, move |target| {
            let (window, _app) = resolve_window(target, &raise)?;
            raise_window(&window).map(|_| ())
        })?;
        with_target(target, move |target| {
            ensure_window_receives(target, &lookup)
        })?;
    } else if raises {
        let raise = lookup.clone();
        if with_target(target, move |target| activate_window(target, &raise))? {
            // Unpinned input has no delivery to verify, so a best-effort raise gets the
            // same settle budget before the event is posted.
            sleep_millis((core::RAISE_SETTLE_STEPS as u64) * core::RAISE_SETTLE_MS);
        }
    }
    match input {
        RawInput::Click {
            point,
            button,
            double,
            modifiers,
        } => {
            let (x, y) = core::normalized_to_surface(
                point.0,
                point.1,
                &surface_geometry_with(target, geometry)?,
            )?;
            click(CGPoint::new(x, y), *button, *double, modifiers)
        }
        RawInput::TypeText { text } => type_text(text),
        RawInput::KeyChord { chord } => key_chord(chord),
        RawInput::Scroll {
            point,
            direction,
            amount,
        } => {
            let p = if let Some((nx, ny)) = point {
                let (x, y) = core::normalized_to_surface(
                    *nx,
                    *ny,
                    &surface_geometry_with(target, geometry)?,
                )?;
                Some(CGPoint::new(x, y))
            } else {
                None
            };
            scroll(p, *direction, *amount)
        }
        RawInput::Drag { from, to } => {
            let surface = surface_geometry_with(target, geometry)?;
            let (fx, fy) = core::normalized_to_surface(from.0, from.1, &surface)?;
            let (tx, ty) = core::normalized_to_surface(to.0, to.1, &surface)?;
            drag(CGPoint::new(fx, fy), CGPoint::new(tx, ty))
        }
    }
}

// ── Capture (ScreenCaptureKit) ───────────────────────────────────────────

/// Hand-off wrappers letting SCK completion handlers deliver non-`Send` retained
/// objects across the handler-to-waiter thread boundary.
struct SendShareable(Retained<SCShareableContent>);
unsafe impl Send for SendShareable {}

struct SendImage(Retained<CGImage>);
unsafe impl Send for SendImage {}

/// The failure a completion handler's `NSError` stands for, classified by its
/// `SCStreamError` code.
fn sc_error(err: &NSError) -> anyhow::Error {
    let code = err.code();
    if code == SCStreamErrorCode::UserDeclined.0 || code == SCStreamErrorCode::MissingEntitlements.0
    {
        permission_error(Grant::ScreenRecording)
    } else if code == SCStreamErrorCode::NoWindowList.0 {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "no window found — re-enumerate windows",
        )
    } else {
        anyhow!(
            "screen capture failed (code {code}): {}",
            err.localizedDescription()
        )
    }
}

/// Block on an SCK completion handler (5s timeout) and return the handed-off
/// result. The caller spawns the provider call; this consumes the shared slot.
fn sc_wait<T: Send>(
    slot: &Arc<Mutex<Option<Result<T, anyhow::Error>>>>,
    semaphore: &Arc<DispatchRetained<DispatchSemaphore>>,
    what: &str,
) -> Result<T, anyhow::Error> {
    if semaphore.wait(DispatchTime::NOW.time(5_000_000_000)) != 0 {
        return Err(anyhow!("timed out waiting for {what}"));
    }
    slot.lock()
        .unwrap_poison()
        .take()
        .ok_or_else(|| anyhow!("{what} handler never invoked"))?
}

/// Hand one completion handler its answer and release the waiter — the only way a
/// handler body ends.
///
/// The system invokes these bodies on its own frames, where nothing may unwind: a
/// Rust panic cannot cross into them, and a framework exception raised inside the
/// body cannot reach the caller's firewall either (the body runs on the system's
/// thread, possibly after the wait gave up). [`contained`] holds both guards, and a
/// failure becomes the value the waiter reads — the call that is waiting reports it,
/// so it is logged at debug rather than as an issue row; a body that fails after the
/// wait returned is the case with no reader.
fn sc_complete<T>(
    slot: &Mutex<Option<Result<T, anyhow::Error>>>,
    semaphore: &DispatchSemaphore,
    what: &str,
    body: impl FnOnce() -> Result<T, anyhow::Error>,
) {
    let outcome = contained(what, body);
    if let Err(e) = &outcome {
        tracing::debug!("computer use: {e:#}");
    }
    *slot.lock().unwrap_poison() = Some(outcome);
    semaphore.signal();
}

/// Fetch the shareable-content snapshot (availability probe + window/display
/// capture source).
fn get_shareable_content() -> Result<Retained<SCShareableContent>, anyhow::Error> {
    let semaphore = Arc::new(DispatchSemaphore::new(0));
    let sem2 = Arc::clone(&semaphore);
    let slot: Arc<Mutex<Option<Result<SendShareable, anyhow::Error>>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);
    let block = RcBlock::new(move |content: *mut SCShareableContent, err: *mut NSError| {
        sc_complete(&slot2, &sem2, "shareable content", || {
            if !err.is_null() {
                // SAFETY: `err` is valid for the handler's duration.
                return Err(sc_error(unsafe { &*err }));
            }
            if content.is_null() {
                return Err(anyhow!("no shareable content"));
            }
            // SAFETY: a retained object the handler hands us; `retain` takes
            // ownership beyond the callback.
            let content = unsafe { Retained::retain(content) }
                .ok_or_else(|| anyhow!("null shareable content"))?;
            Ok(SendShareable(content))
        });
    });
    // SAFETY: `block` is a live stack block; the framework copies it for the
    // asynchronous call, and the wait below keeps the original alive until the
    // handler has run.
    //
    // Off-screen windows are included on purpose: a window on another Space, behind
    // another app or minimized is still the surface a call may name, and excluding
    // it made the capture path report a missing window as "re-enumerate".
    unsafe {
        SCShareableContent::getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler(false, false, &block);
    }
    sc_wait::<SendShareable>(&slot, &semaphore, "shareable content").map(|SendShareable(c)| c)
}

/// A single still of `filter`'s content, at the pixel size the caller asked for.
///
/// Both capture routes derive that size from what the system reported — a window's
/// frame, a display's frame — so it is put through [`core::capture_size`] BEFORE the
/// request reaches the framework: zero is what a minimized, off-screen or vanished
/// surface reduces to, and the framework must never be asked for a size the buffer
/// cannot match.
///
/// Callers guarantee [`sck_available`]: the `SCScreenshotManager` class itself
/// only exists from macOS 14 on, and messaging a missing class aborts the
/// process.
fn capture_filter_to_rgba(
    filter: &SCContentFilter,
    px_w: usize,
    px_h: usize,
    ignore_shadows: bool,
    budget: &CaptureBudget,
) -> Result<Capture, anyhow::Error> {
    let (px_w, px_h) = core::capture_size("the target", px_w, px_h)?;
    let (px_w, px_h) = (px_w as usize, px_h as usize);
    let config = unsafe { SCStreamConfiguration::new() };
    // SAFETY: the config setters are ObjC messages against a live object.
    // No source rect is set: SCK interprets one filter-locally and would crop
    // off-content, so the full filter content is captured and `core::crop_rgba`
    // does the zoom cropping instead.
    unsafe {
        config.setWidth(px_w);
        config.setHeight(px_h);
        config.setShowsCursor(false);
        config.setIgnoreShadowsSingleWindow(ignore_shadows);
        config.setIgnoreShadowsDisplay(ignore_shadows);
    }

    let semaphore = Arc::new(DispatchSemaphore::new(0));
    let sem2 = Arc::clone(&semaphore);
    let slot: Arc<Mutex<Option<Result<SendImage, anyhow::Error>>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);
    let block = RcBlock::new(move |image: *mut CGImage, err: *mut NSError| {
        sc_complete(&slot2, &sem2, "screenshot", || {
            if !err.is_null() {
                // SAFETY: `err` is valid for the handler's duration.
                return Err(sc_error(unsafe { &*err }));
            }
            if image.is_null() {
                return Err(anyhow!("no image in the screenshot"));
            }
            // SAFETY: a +0 object the handler hands us; `retain` takes ownership
            // beyond the callback, which ends when the waiter returns.
            let image = unsafe { Retained::retain(image) }
                .ok_or_else(|| anyhow!("null screenshot image"))?;
            Ok(SendImage(image))
        });
    });
    // SAFETY: filter/config/block are valid; the framework copies the block for
    // the asynchronous call, and the wait below keeps the original alive until
    // the handler has run.
    unsafe {
        SCScreenshotManager::captureImageWithFilter_configuration_completionHandler(
            filter,
            &config,
            Some(&block),
        );
    }
    let SendImage(image) = sc_wait::<SendImage>(&slot, &semaphore, "screenshot")?;
    capture_image_to_rgba(&image, budget)
}

/// Turn a CGImage into top-down RGBA bytes. CG contexts use a bottom-left
/// origin, so the image is flipped (translate+scale) before drawing to yield
/// top-down rows. The buffer is the call's: its geometry goes through
/// [`core::capture_size`], and it is charged to `budget` with a fallible reserve,
/// then filled in one copy from the context's own pixels.
fn capture_image_to_rgba(
    image: &CGImage,
    budget: &CaptureBudget,
) -> Result<Capture, anyhow::Error> {
    let w = CGImage::width(Some(image));
    let h = CGImage::height(Some(image));
    let (px_w, px_h) = core::capture_size("a captured frame", w, h)?;
    let space = CGColorSpace::new_device_rgb()
        .ok_or_else(|| anyhow!("failed to create RGB color space"))?;
    let bitmap_info = CGImageAlphaInfo::PremultipliedLast.0 | CGImageByteOrderInfo::Order32Big.0;
    // SAFETY: sizes/stride are valid; NULL data → CG allocates the buffer.
    let ctx = unsafe {
        CGBitmapContextCreate(
            std::ptr::null_mut(),
            w,
            h,
            8,
            w * 4,
            Some(&space),
            bitmap_info,
        )
    }
    .ok_or_else(|| anyhow!("failed to create bitmap context"))?;
    CGContext::translate_ctm(Some(&ctx), 0.0, h as f64);
    CGContext::scale_ctm(Some(&ctx), 1.0, -1.0);
    CGContext::draw_image(
        Some(&ctx),
        CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(w as f64, h as f64)),
        Some(image),
    );
    let data = CGBitmapContextGetData(Some(&ctx));
    if data.is_null() {
        anyhow::bail!("bitmap context returned no data");
    }
    // SAFETY: `data` points to `w*h*4` bytes owned by the context — the buffer's own
    // size, as the charge records — and it is copied before the context is dropped.
    let pixels = unsafe { std::slice::from_raw_parts(data as *const u8, w * h * 4) };
    Capture::from_pixels(budget, "a captured frame", px_w, px_h, pixels)
}

/// The ScreenCaptureKit window a target's AX window is matched to, or a refusal:
/// the two are tied together by owning process and frame, never by the order of a
/// list or a title alone.
fn find_sc_window(
    content: &SCShareableContent,
    pid: u32,
    title: Option<&str>,
    frame: SurfaceGeometry,
) -> Result<Retained<SCWindow>, anyhow::Error> {
    let near = |window: &SCWindow| {
        // SAFETY: ObjC message against a live SCWindow.
        let f = unsafe { window.frame() };
        (f.origin.x - frame.x).abs() < 2.0
            && (f.origin.y - frame.y).abs() < 2.0
            && (f.size.width - frame.width).abs() < 2.0
            && (f.size.height - frame.height).abs() < 2.0
    };
    let titled = |windows: &[Retained<SCWindow>], title: &str| {
        windows
            .iter()
            .filter(|window| {
                // SAFETY: ObjC message against a live SCWindow.
                unsafe { window.title() }.map(|s| s.to_string()).as_deref() == Some(title)
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    // SAFETY: each call below is an ObjC message against a live object.
    let of_app = unsafe { content.windows() }
        .iter()
        .filter(|window| {
            let owner = unsafe { window.owningApplication() };
            owner.as_ref().map(|a| unsafe { a.processID() } as u32) == Some(pid)
        })
        .collect::<Vec<_>>();
    let of_frame = of_app
        .iter()
        .filter(|window| near(window))
        .cloned()
        .collect::<Vec<_>>();
    // The frame is the window's own truth; a title only breaks a tie, and a tie
    // that survives it means the two cannot be told apart — refusing beats
    // capturing the wrong one. With no frame match the frameworks merely disagree
    // about geometry, so a unique exact title in the same process still names the
    // window honestly. The title this is handed is the AX one, capped at
    // [`core::MAX_NAME_CHARS`], so a longer one no longer compares equal here: the
    // frame is the primary key and the cap only costs title-only matches.
    let matching = match title.filter(|t| !t.is_empty()) {
        Some(title) if !of_frame.is_empty() => {
            let by_title = titled(&of_frame, title);
            if by_title.is_empty() {
                of_frame
            } else {
                by_title
            }
        }
        Some(title) => titled(&of_app, title),
        None => of_frame,
    };
    match matching.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(core::taxonomy_error(
            core::ERR_SURFACE_GONE,
            format!(
                "{} this target names is not in the capture list — it closed, or the capture \
                 facility does not expose it; enumerate windows again to pick a surface",
                core::window_description("", title.unwrap_or_default())
            ),
        )),
        many => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "{} windows of the same application match this target's title or frame, so the \
                 capture cannot tell which one to shoot — move or resize one of them, or \
                 screenshot the screen instead",
                many.len()
            ),
        )),
    }
}

/// The AX half of a window still: which process owns the window, its title and
/// its frame — the facts ScreenCaptureKit is then matched against. A minimized
/// window is refused here: it has no visible pixels, and its frame is the one it
/// had before it was minimized.
fn window_capture_facts(
    target: &TargetSpec,
    budget: &core::Budget,
) -> Result<(u32, Option<String>, SurfaceGeometry), anyhow::Error> {
    let (window, app) = resolve_window(target, budget)?;
    budget.check()?;
    let pid = app_pid(&app)?;
    let title = window_title(&window);
    budget.check()?;
    let label = resolved_surface_label(target, &app, title.as_deref())?;
    refuse_minimized(&window)?;
    // `window_frame` checks the budget itself before each of its two reads.
    let frame =
        window_frame(&window, budget).map_err(|_| core::unanswered(&label, "for its frame"))?;
    Ok((pid, title, frame))
}

fn capture_window_blocking(
    target: &TargetSpec,
    budget: &CaptureBudget,
) -> Result<Capture, anyhow::Error> {
    // The window is resolved through AX (pid, title, frame) before SCK is asked
    // for pixels; only that half belongs on the AX thread, never the capture.
    let lookup = core::Budget::new(core::LIST_BUDGET, "surface lookup", target.describe());
    let (pid, title, frame) =
        with_target(target, move |target| window_capture_facts(target, &lookup))?;
    let content = get_shareable_content()?;
    let sc_window = find_sc_window(&content, pid, title.as_deref(), frame)?;
    // SAFETY: `alloc()` is the fresh allocation consumed by the init.
    let filter = unsafe {
        SCContentFilter::initWithDesktopIndependentWindow(SCContentFilter::alloc(), &sc_window)
    };
    let scale = unsafe { filter.pointPixelScale() } as f64;
    let px_w = (frame.width * scale).round();
    let px_h = (frame.height * scale).round();
    capture_filter_to_rgba(&filter, px_w as usize, px_h as usize, true, budget)
}

/// Composite each active display into one RGBA buffer at the virtual-desktop
/// bounding rect, each display drawn at its global-frame offset.
///
/// One scale governs the whole buffer: the densest display's. Every display is
/// captured AT that size — ScreenCaptureKit renders a filter's content at the size
/// it is asked for — so a 1x display beside a 2x one lands at its logical offset ×
/// that scale instead of covering half the buffer, and a mixed-density arrangement
/// produces one faithful composite rather than a distorted one.
///
/// The composite is the call's largest buffer, so it is charged to `budget` before
/// it exists — as is each display's frame, which the framework renders at that size.
fn capture_screen_blocking(budget: &CaptureBudget) -> Result<Capture, anyhow::Error> {
    let content = get_shareable_content()?;
    // SAFETY: SCK getter is an ObjC message against a live object.
    let displays = unsafe { content.displays() };
    if displays.is_empty() {
        anyhow::bail!("no shareable displays found");
    }
    let empty: Retained<NSArray<SCWindow>> = NSArray::new();
    let mut filters = Vec::with_capacity(displays.len());
    let mut ref_scale: f64 = 0.0;
    for display in &displays {
        // SAFETY: `alloc()` is the fresh allocation consumed by the init.
        let filter = unsafe {
            SCContentFilter::initWithDisplay_excludingWindows(
                SCContentFilter::alloc(),
                &display,
                &empty,
            )
        };
        // SAFETY: ObjC message against a live SCDisplay.
        let f = unsafe { display.frame() };
        ref_scale = ref_scale.max(unsafe { filter.pointPixelScale() } as f64);
        filters.push((
            filter,
            SurfaceGeometry {
                x: f.origin.x,
                y: f.origin.y,
                width: f.size.width,
                height: f.size.height,
            },
        ));
    }
    if ref_scale <= 0.0 {
        anyhow::bail!("no display reported a usable pixel scale");
    }
    let bbox = screen_surface()?;
    let out_w = (bbox.width * ref_scale).round() as usize;
    let out_h = (bbox.height * ref_scale).round() as usize;
    let (out_px_w, out_px_h) = core::capture_size("the virtual desktop", out_w, out_h)?;
    let mut composite = Capture::alloc(budget, "the screen composite", out_px_w, out_px_h)?;
    for (filter, frame) in filters {
        let px_w = (frame.width * ref_scale).round() as usize;
        let px_h = (frame.height * ref_scale).round() as usize;
        let cap = capture_filter_to_rgba(&filter, px_w, px_h, false, budget)?;
        // A display that comes back at another size cannot be placed at this
        // scale, and blitting it would leave part of the buffer black without a
        // word — refuse instead of composing a wrong image.
        if (cap.width as usize, cap.height as usize) != (px_w, px_h) {
            return Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                format!(
                    "a display asked for {px_w}x{px_h} pixels came back at {}x{}, so this \
                     mixed-density arrangement cannot be composited faithfully — screenshot a \
                     window instead, or capture with the display arrangement made uniform",
                    cap.width, cap.height
                ),
            ));
        }
        let dx = ((frame.x - bbox.x) * ref_scale).round();
        let dy = ((frame.y - bbox.y) * ref_scale).round();
        if dx >= 0.0 && dy >= 0.0 {
            core::blit_rgba(
                &mut composite.rgba,
                out_w,
                out_h,
                &cap,
                dx as usize,
                dy as usize,
            );
        }
    }
    Ok(composite)
}

fn capture_blocking(target: &TargetSpec, budget: &CaptureBudget) -> Result<Capture, anyhow::Error> {
    if !sck_available() {
        return Err(screen_capture_needs_14());
    }
    // A window still is resolved through AX (its frame and title) before SCK is
    // asked for pixels, so the Accessibility grant is needed as well.
    if !matches!(target, TargetSpec::Screen) {
        ensure(Grant::Accessibility)?;
    }
    let result = match target {
        TargetSpec::Screen => capture_screen_blocking(budget),
        _ => capture_window_blocking(target, budget),
    };
    // A failed still is diagnosed from the grant, never assumed: the attempt
    // itself is the authority (a preflight answer is only a hint), so a
    // grantless capture asks for the grant and reports the actionable error,
    // while a transient failure keeps its own.
    result.map_err(|e| ensure(Grant::ScreenRecording).err().unwrap_or(e))
}

fn capture_available_blocking() -> bool {
    // Preflight only: the boot probe must never put a TCC dialog on the owner's
    // screen, and fetching shareable content can raise one.
    sck_available() && Grant::ScreenRecording.held()
}

// ── Element actions ──────────────────────────────────────────────────────

fn press_element(element: &AXUIElement) -> Result<Impact, anyhow::Error> {
    let names = raw_action_names(element).map_err(|e| ax_error(e, "read element actions"))?;
    let Some(names) = names else {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "the accessibility API answered without usable action names for this element — act \
             on it by coordinates instead of by ref",
        ));
    };
    if names.iter().any(|a| a == AX_PRESS) {
        // SAFETY: AXPress was verified to exist on the element.
        map_action_err(
            unsafe { element.perform_action(&axc(AX_PRESS)) },
            "perform press",
        )
    } else if names.iter().any(|a| a == AX_OPEN) {
        // SAFETY: AXOpen was verified to exist on the element.
        map_action_err(
            unsafe { element.perform_action(&axc(AX_OPEN)) },
            "perform open",
        )
    } else {
        let available = names
            .iter()
            .map(|a| normalize_action_name(a))
            .collect::<Vec<_>>()
            .join(", ");
        Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("element has no press/open action; available actions: {available}"),
        ))
    }
}

fn set_element_value(element: &AXUIElement, text: &str) -> Result<Impact, anyhow::Error> {
    let role = attribute_name(element, &axc(AX_ROLE)).unwrap_or_default();
    let subrole = attribute_name(element, &axc(AX_SUBROLE));
    if is_secure_field(&role, subrole.as_deref()) {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element is a secure text field — SetValue refused (use click by ref then type)",
        ));
    }
    let cf: CFRetained<CFType> = CFRetained::from(CFString::from_str(text));
    // SAFETY: `cf` is a valid CFString (a CFType).
    map_action_err(
        unsafe { element.set_attribute_value(&axc(AX_VALUE), &cf) },
        "set value",
    )
}

fn act_on_element_blocking(
    target: &TargetSpec,
    locator: &Locator,
    act: &ElementAct,
) -> Result<Impact, anyhow::Error> {
    // One clock for the whole call: resolution, the reads that name the surface, the
    // walk and the action itself.
    let mut budget = core::Budget::new(core::ACT_BUDGET, "element action", target.describe());
    let (window, app) = resolve_window(target, &budget)?;
    budget.check()?;
    let app_name = resolved_app_name(target, &app)?;
    // The header title is a name, not a label: read past the cap so a cut is counted.
    let title = attribute_text(&window, &axc(AX_TITLE), core::MAX_NAME_CHARS);
    let label = core::name_surface(&mut budget, &app_name, title.as_deref());
    let origin =
        window_origin(&window).map_err(|_| core::unanswered(&label, "for its position"))?;
    let (root, handles, _truncation) = build_tree(&window, origin, &budget)?;
    let matched = core::resolve_locator_checked(&root, locator)?;
    let element = &handles[core::pre_order_index(&root, matched)];
    match act {
        ElementAct::Press => press_element(element),
        ElementAct::SetValue { text } => set_element_value(element, text),
    }
}

/// Map the result of an AX ACTION onto the outcome the model is owed. The one
/// action-specific arm is `CannotComplete`: the request reached the target without
/// confirmation, so it is the middle outcome, [`Impact::Unconfirmed`].
fn map_action_err(err: AXError, ctx: &str) -> Result<Impact, anyhow::Error> {
    match err {
        AXError::Success => Ok(Impact::Done),
        AXError::AttributeUnsupported => Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!(
                "{ctx} — element does not support SetValue; use click by ref then type without ref"
            ),
        )),
        AXError::InvalidUIElement => Err(core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("{ctx} — window closed or changed — re-enumerate"),
        )),
        AXError::APIDisabled => Err(permission_error(Grant::Accessibility)),
        AXError::CannotComplete => Ok(Impact::Unconfirmed(format!(
            "{ctx} — the application did not confirm the accessibility action, which may already \
             have been performed"
        ))),
        _ => Err(anyhow!("AX error {err:?} while {ctx}")),
    }
}

// ── Backend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl Backend for MacOsBackend {
    fn is_supported(&self) -> bool {
        // macOS 14 is the floor, so the capability itself is always present.
        true
    }

    async fn capture_available(&self) -> bool {
        // A probe that cannot answer counts as "no": the capture attempt carries the
        // real diagnosis, and the failure is logged instead of being lost.
        with_platform("probe capture", || Ok(capture_available_blocking())).unwrap_or_else(|e| {
            tracing::warn!("computer use: {e:#}");
            false
        })
    }

    /// The diagnosis for a capture that is unavailable. It has no call to fail into,
    /// so a raised exception *becomes* the answer, as does the diagnosis itself; its
    /// platform contacts are the version check and the grant-name lookup.
    fn capture_unavailable_error(&self) -> anyhow::Error {
        catching_foreign("diagnose capture", || {
            Ok(if sck_available() {
                permission_error(Grant::ScreenRecording)
            } else {
                screen_capture_needs_14()
            })
        })
        .unwrap_or_else(std::convert::identity)
    }

    async fn list_apps(&self) -> Result<Vec<AppInfo>, anyhow::Error> {
        with_platform("list applications", || {
            let mut out = Vec::new();
            for app in NSWorkspace::sharedWorkspace().runningApplications() {
                if app.activationPolicy() != NSApplicationActivationPolicy::Regular {
                    continue;
                }
                let name = app.localizedName().map_or(String::new(), |s| s.to_string());
                if !name.is_empty() {
                    out.push(AppInfo {
                        pid: Some(app.processIdentifier() as u32),
                        name,
                    });
                }
            }
            out.sort_by(|a, b| a.name.cmp(&b.name));
            out.dedup_by(|a, b| a.pid == b.pid);
            Ok(out)
        })
    }

    async fn list_windows(&self, app: Option<&AppInfo>) -> Result<WindowListing, anyhow::Error> {
        with_platform("list windows", || {
            ensure(Grant::Accessibility)?;
            let app_infos: Vec<AppInfo> = if let Some(a) = app {
                vec![a.clone()]
            } else {
                let mut v = Vec::new();
                for run in NSWorkspace::sharedWorkspace().runningApplications() {
                    if run.activationPolicy() != NSApplicationActivationPolicy::Regular {
                        continue;
                    }
                    let name = run.localizedName().map_or(String::new(), |s| s.to_string());
                    if !name.is_empty() {
                        v.push(AppInfo {
                            pid: Some(run.processIdentifier() as u32),
                            name,
                        });
                    }
                }
                v
            };
            let mut listing = WindowListing::default();
            let owner = core::agent_key();
            // One clock for the whole enumeration, and one slice of it per application:
            // a hung application costs its own part rather than the healthy ones read
            // before and after it.
            let budget = core::Budget::new(
                core::LIST_BUDGET,
                "window listing",
                app.map_or_else(
                    || "the running applications".to_string(),
                    |a| core::windows_label(&a.name),
                ),
            );
            for ai in app_infos {
                let Some(pid) = ai.pid else { continue };
                let name = ai.name;
                // A listing that runs out of time keeps what it read and names the
                // applications it did not reach, instead of discarding healthy windows.
                if let Err(e) = budget.check() {
                    listing.unread.push(UnreadWindows {
                        app_name: name,
                        reason: format!("{e:#}"),
                    });
                    continue;
                }
                let owner = owner.clone();
                let app_name = name.clone();
                // An application whose accessibility cannot be read is named in the
                // listing with what went wrong — never left out as if it had no
                // windows, and never allowed to fail the whole listing. The AX thread
                // only makes our OWN window's reads possible at all, so that app's
                // failure is the app's, listed like any other.
                let app_budget = budget.slice(core::APP_SLICE, core::windows_label(&name));
                // The copy outlives the closure, so the listing can ask afterwards whether
                // it was this application's share that ran out.
                let share = app_budget.clone();
                match on_ax_thread(pid, move || windows_of_app(pid, &name, &owner, &app_budget)) {
                    Ok(rows) => listing.windows.extend(rows),
                    // A spent share is one skipped application, said on the line naming it.
                    Err(e) => listing.unread.push(UnreadWindows {
                        app_name,
                        reason: core::unread_reason(&e, &share),
                    }),
                }
            }
            Ok(listing)
        })
        .map_err(degraded_read)
    }

    async fn focused_window(&self) -> Result<WindowInfo, anyhow::Error> {
        with_platform("read the focused window", || {
            ensure(Grant::Accessibility)?;
            let owner = core::agent_key();
            let budget = core::Budget::new(
                core::LIST_BUDGET,
                "focused-window search",
                "the running applications",
            );
            with_target(&TargetSpec::Focused, move |target| {
                let (window, app) = resolve_window(target, &budget)?;
                budget.check()?;
                let pid = app_pid(&app)?;
                let app_name = resolved_app_name(target, &app)?;
                let title = window_title(&window);
                let label = resolved_surface_label(target, &app, title.as_deref())?;
                Ok(WindowInfo {
                    app_pid: Some(pid),
                    app_name,
                    title: title.unwrap_or_default(),
                    key: window_key(&window, &owner),
                    surface: window_frame(&window, &budget)
                        .map_err(|_| core::unanswered(&label, "for its frame"))?,
                })
            })
        })
        .map_err(degraded_read)
    }

    async fn surface_geometry(
        &self,
        target: &TargetSpec,
    ) -> Result<SurfaceGeometry, anyhow::Error> {
        with_platform("read the target's geometry", || {
            let budget = core::Budget::new(core::LIST_BUDGET, "surface lookup", target.describe());
            surface_geometry_with(target, budget)
        })
        .map_err(degraded_read)
    }

    async fn observe(&self, target: &TargetSpec) -> Result<Observation, anyhow::Error> {
        with_platform("observe", || {
            ensure(Grant::Accessibility)?;
            with_target(target, observe_blocking)
        })
        .map_err(degraded_read)
    }

    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> Result<Impact, anyhow::Error> {
        with_platform("act on an element", || {
            if matches!(target, TargetSpec::Screen) {
                return Err(core::screen_act_error());
            }
            ensure(Grant::Accessibility)?;
            let locator = locator.clone();
            match with_target(target, move |target| {
                act_on_element_blocking(target, &locator, &act)
            }) {
                // The request was handed to mahbot's own main queue and the wait ran
                // out: it may still be applied when that thread catches up, so this is
                // the middle outcome rather than a failure.
                Err(e) if e.downcast_ref::<AxBusy>().is_some() => Ok(Impact::Unconfirmed(format!(
                    "{e}; the request may still be applied once that thread catches up"
                ))),
                // A framework exception is raised from inside the exchange: where it
                // stopped is unknown, so the element may already have acted.
                Err(e) if core::is_framework_failed(&e) => Ok(Impact::Unconfirmed(format!(
                    "the system raised an exception while the element action was running, so it \
                     may already have been performed — {e}"
                ))),
                other => other,
            }
        })
    }

    async fn raw_input(
        &self,
        target: &TargetSpec,
        input: RawInput,
    ) -> Result<Impact, anyhow::Error> {
        match with_platform("synthesize input", || {
            check_input_arguments(&input)?;
            ensure(Grant::Input)?;
            raw_input_blocking(target, &input)
        }) {
            // An own-window exchange that timed out before any event was posted:
            // nothing was actuated, so it is a plain retryable failure.
            Err(e) if e.downcast_ref::<AxBusy>().is_some() => Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                format!("{e} — nothing was sent to the window; retry the call"),
            )),
            // An exception raised under the call: the events already posted stay
            // posted, so this may have landed and is not a failure.
            Err(e) if core::is_framework_failed(&e) => Ok(Impact::Unconfirmed(format!(
                "the system raised an exception while the input was being synthesized, so part of \
                 it may already have been delivered — {e}"
            ))),
            other => other,
        }
    }

    async fn cursor_position(&self) -> Result<(f64, f64), anyhow::Error> {
        with_platform("read the pointer position", || {
            let event =
                CGEvent::new(None).ok_or_else(|| anyhow!("failed to create dummy cursor event"))?;
            let p = CGEvent::location(Some(&event));
            Ok((p.x, p.y))
        })
    }

    async fn capture(
        &self,
        target: &TargetSpec,
        budget: &CaptureBudget,
    ) -> Result<Capture, anyhow::Error> {
        with_platform("capture", || capture_blocking(target, budget)).map_err(degraded_read)
    }
}

/// The window's title as a capped label ([`core::cap_label`]): a cut is marked with an
/// ellipsis and counted only where a tree header is at hand, which is
/// [`observe_blocking`] reading the title through [`Truncation::keep_name`] itself.
fn window_title(window: &AXUIElement) -> Option<String> {
    attribute_name(window, &axc(AX_TITLE))
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The evidence for this backend's exception route: a framework call that
    /// raises must come back as a failed call, and raising it must not kill the
    /// process running this test.
    #[test]
    fn a_framework_exception_becomes_a_failed_call() {
        let err = with_platform("probe", || -> Result<(), anyhow::Error> {
            let array = NSArray::<objc2::runtime::NSObject>::new();
            // An out-of-range read is the framework's own failure (an
            // NSRangeException raised by NSArray), not something Rust maps.
            let _: *mut objc2::runtime::AnyObject =
                unsafe { objc2::msg_send![&array, objectAtIndex: 0usize] };
            Ok(())
        })
        .expect_err("the exception must not escape the firewall");
        let text = format!("{err:#}");
        assert!(
            text.starts_with(core::ERR_FRAMEWORK_FAILED),
            "expected a framework-failed call, got: {text}"
        );
        assert!(text.contains("exception"), "{text}");
    }

    /// A wrongly-typed member of a system array must not be read as the type it
    /// was assumed to be, and must not take the well-typed members with it.
    #[test]
    fn typed_members_skip_a_member_of_another_type() {
        let name = CFString::from_str("AXPress");
        let number = objc2_core_foundation::CFNumber::new_i64(7);
        let array = CFArray::<CFType>::from_objects(&[&name, &number]);
        let names: Vec<String> = typed_members::<CFString>(array.as_ref())
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(names, vec!["AXPress".to_string()]);
    }

    #[test]
    fn key_table_covers_chord_parser_keys() {
        for key in [
            "a",
            "z",
            "5",
            "0",
            "return",
            "space",
            "backslash",
            "f1",
            "f12",
            "capslock",
        ] {
            let (mods, parsed) = core::parse_key_chord(key).unwrap();
            assert!(mods.is_empty());
            assert_eq!(parsed, key.to_lowercase(), "chord parser mangles '{key}'");
            assert!(key_code(key).is_some(), "kVK table missing '{key}'");
        }
        let (mods, parsed) = core::parse_key_chord("cmd+shift+t").unwrap();
        assert_eq!(mods, vec![Modifier::Cmd, Modifier::Shift]);
        assert_eq!(parsed, "t");
    }
}
