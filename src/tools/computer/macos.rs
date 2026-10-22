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
//! type it is read as ([`typed_members`]) instead of being cast wholesale, and a
//! capture size is validated before it reaches the capture framework.
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
    self, AppInfo, Backend, Capture, ElementAct, Locator, Modifier, MouseButton, Observation,
    RawInput, ScrollDirection, SurfaceGeometry, TargetSpec, UiNode, WindowInfo,
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
    CFArray, CFBoolean, CFDictionary, CFRetained, CFString, CFType, CGPoint, CGRect, CGSize,
    ConcreteType, Type,
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub(crate) static MACOS_BACKEND: MacOsBackend = MacOsBackend;

pub(crate) struct MacOsBackend;

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
    with_block_in_place(|| catching_foreign(what, f))
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

/// Run one accessibility exchange where the system requires it: on the main
/// thread when `pid` is mahbot itself, on the calling thread otherwise. A main
/// thread that stays busy past [`SELF_AX_WAIT`] fails the call honestly instead
/// of hanging it; the request itself may still be applied afterwards.
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
        Err(_) => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "mahbot's own window did not answer within {}s — its main thread is busy, so the \
                 call gave up instead of hanging; the request may still be applied once that \
                 thread catches up, so re-observe before repeating a mutation",
                SELF_AX_WAIT.as_secs()
            ),
        )),
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

fn attribute_string(el: &AXUIElement, attr: &CFString) -> Option<String> {
    let val = attribute_value(el, attr).ok()?;
    Some(val.downcast_ref::<CFString>()?.to_string())
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

fn walk_element(
    element: &AXUIElement,
    origin: (f64, f64),
    depth: usize,
    handles: &mut Vec<CFRetained<AXUIElement>>,
    nodes: &mut usize,
) -> Option<UiNode> {
    if *nodes >= core::MAX_RENDER_NODES {
        return None;
    }
    *nodes += 1;
    handles.push(element.retain());

    let role = attribute_string(element, &axc(AX_ROLE)).unwrap_or_else(|| "unknown".to_string());
    // Never read a value out of a secure element. A password field reports
    // AXRole="AXTextField" with the secure marker in AXSubrole, so check the
    // subrole (or role) for "secure" before pulling the value.
    let subrole = attribute_string(element, &axc(AX_SUBROLE));
    let value = if is_secure_field(&role, subrole.as_deref()) {
        None
    } else {
        attribute_string(element, &axc(AX_VALUE))
    };
    let frame = match (
        attribute_point(element, &axc(AX_POSITION)),
        attribute_size(element, &axc(AX_SIZE)),
    ) {
        (Some(p), Some(s)) => Some((p.x - origin.0, p.y - origin.1, s.width, s.height)),
        _ => None,
    };
    // A failed or unusable read only leaves the node's actions absent: the walk
    // degrades, acting on the element is what must not.
    let actions = raw_action_names(element)
        .ok()
        .flatten()
        .unwrap_or_default()
        .iter()
        .map(|a| normalize_action_name(a))
        .collect();
    let focused = attribute_bool(element, &axc(AX_FOCUSED)).unwrap_or(false);

    let mut children = Vec::new();
    if depth < MAX_DEPTH {
        for child in attribute_children(element) {
            if let Some(child_node) = walk_element(&child, origin, depth + 1, handles, nodes) {
                children.push(child_node);
            }
        }
    }
    Some(UiNode {
        role,
        name: attribute_string(element, &axc(AX_TITLE)),
        value,
        frame,
        actions,
        focused,
        children,
    })
}

/// Build the accessibility tree, returning the root node plus a pre-order-aligned
/// `Vec` of platform handles (used by `act_on_element` to map a matched node back
/// to its element).
fn build_tree(root: &AXUIElement, origin: (f64, f64)) -> (UiNode, Vec<CFRetained<AXUIElement>>) {
    let mut handles = Vec::new();
    let mut nodes = 0usize;
    let node = walk_element(root, origin, 0, &mut handles, &mut nodes).unwrap_or_else(|| UiNode {
        role: "unknown".to_string(),
        name: None,
        value: None,
        frame: None,
        actions: Vec::new(),
        focused: false,
        children: Vec::new(),
    });
    (node, handles)
}

fn observe_blocking(target: &TargetSpec) -> Result<Observation, anyhow::Error> {
    let (window, app) = resolve_window(target)?;
    let origin = window_origin(&window)?;
    let surface = window_frame(&window)?;
    let app_name = match target {
        TargetSpec::Window { app_name, .. } => app_name.clone(),
        _ => attribute_string(&app, &axc(AX_TITLE)).unwrap_or_default(),
    };
    let (root, _handles) = build_tree(&window, origin);
    Ok(Observation {
        app_name,
        window_title: window_title(&window),
        surface,
        root,
    })
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
        .unwrap_or_else(|| attribute_string(app, &axc(AX_TITLE)).unwrap_or_default())
}

fn app_element(pid: Option<u32>, name: &str) -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    let pid = match pid {
        Some(p) => p,
        None => running_app_pid(name).ok_or_else(|| {
            anyhow!("could not resolve running application '{name}' to a process id")
        })?,
    };
    // SAFETY: `pid` is a running app (or stale); AXUIElementCreateApplication
    // always returns a non-null element — attribute reads surface InvalidUIElement.
    Ok(unsafe { AXUIElement::new_application(pid as libc::pid_t) })
}

fn read_windows(app: &AXUIElement) -> Result<Vec<CFRetained<AXUIElement>>, anyhow::Error> {
    let val = attribute_value(app, &axc(AX_WINDOWS))
        .map_err(|e| ax_window_error(e, "enumerate windows"))?;
    let array: CFRetained<CFArray> = val
        .downcast::<CFArray>()
        .map_err(|_| anyhow!("AXWindows was not a CFArray"))?;
    Ok(typed_members::<AXUIElement>(&array))
}

/// One application's listed windows: the AX window array plus each window's frame
/// and title. A window with neither is not a surface and is dropped.
///
/// The listing is the one path that knows the target process without resolving a
/// `TargetSpec`, so it calls [`on_ax_thread`] itself with the app's pid.
fn windows_of_app(pid: u32, app_name: &str) -> Result<Vec<WindowInfo>, anyhow::Error> {
    // SAFETY: `pid` is a running app (or stale); AXUIElementCreateApplication
    // always returns a non-null element — attribute reads surface InvalidUIElement.
    let app = unsafe { AXUIElement::new_application(pid as libc::pid_t) };
    let windows = read_windows(&app)?;
    let mut out = Vec::new();
    for (index, window) in windows.iter().enumerate() {
        let Ok(frame) = window_frame(window) else {
            continue;
        };
        let title = window_title(window).unwrap_or_default();
        if title.is_empty() && (frame.width <= 0.0 || frame.height <= 0.0) {
            continue;
        }
        out.push(WindowInfo {
            app_pid: Some(pid),
            app_name: app_name.to_string(),
            title,
            index,
            surface: frame,
        });
    }
    Ok(out)
}

fn resolve_window_in_app(
    app: &AXUIElement,
    title: &str,
    index: usize,
) -> Result<CFRetained<AXUIElement>, anyhow::Error> {
    // App-level target (empty title, index 0) resolves the app's focused window
    // at ACT time via kAXFocusedWindowAttribute first; only if that is
    // unavailable does it fall back to the first window in the AXWindows array.
    // This is deliberately re-resolved per action, so a ref observed against an
    // app-level target follows the currently focused window. A locator mismatch
    // inside the so-resolved window then degrades to not-matched (a safe error),
    // never to a wrong-window action.
    if title.is_empty()
        && index == 0
        && let Some(w) = focused_window_of_app(app)
    {
        return Ok(w);
    }
    let windows = read_windows(app)?;
    if windows.is_empty() {
        return Err(core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "no windows found for this app — re-enumerate windows",
        ));
    }
    if !title.is_empty()
        && let Some(w) = windows
            .iter()
            .find(|w| attribute_string(w, &axc(AX_TITLE)).as_deref() == Some(title))
    {
        return Ok(w.retain());
    }
    if index < windows.len() {
        return Ok(windows[index].retain());
    }
    Err(core::taxonomy_error(
        core::ERR_NOT_MATCHED,
        format!("window '{title}' (index {index}) not found — re-enumerate windows"),
    ))
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
) -> Result<(CFRetained<AXUIElement>, CFRetained<AXUIElement>), anyhow::Error> {
    match target {
        TargetSpec::Focused => {
            let app = frontmost_app()?;
            ensure_ax_thread(app_pid(&app)?)?;
            // Strictly the frontmost app's focused window: a coordinate action on
            // a fallback surface would land on whatever is visible, which may not
            // be that window (another Space, a minimized window).
            let window = focused_window_of_app(&app).ok_or_else(|| {
                core::taxonomy_error(
                    core::ERR_NOT_MATCHED,
                    "no focused window (desktop frontmost/screensaver) — use 'apps'/'windows' \
                     to pick a surface",
                )
            })?;
            Ok((window, app))
        }
        TargetSpec::Window {
            app_pid: pinned,
            app_name,
            title,
            index,
        } => {
            let app = app_element(*pinned, app_name)?;
            ensure_ax_thread(app_pid(&app)?)?;
            Ok((resolve_window_in_app(&app, title, *index)?, app))
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

fn window_frame(window: &AXUIElement) -> Result<SurfaceGeometry, anyhow::Error> {
    let p = attribute_point(window, &axc(AX_POSITION))
        .ok_or_else(|| anyhow!("window position unavailable"))?;
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

fn click(
    point: CGPoint,
    button: MouseButton,
    double: bool,
    modifiers: &[Modifier],
) -> Result<(), anyhow::Error> {
    let src = source();
    let flags = cg_flags(modifiers);
    let flap = |down: bool, state: i64| -> Result<CFRetained<CGEvent>, anyhow::Error> {
        let ev = mouse_event(
            src.as_deref(),
            cg_type_for(button, down),
            point,
            cg_mouse_button(button),
        )?;
        CGEvent::set_flags(Some(&ev), flags);
        CGEvent::set_integer_value_field(Some(&ev), CGEventField::MouseEventClickState, state);
        Ok(ev)
    };
    // Both halves of a double click carry click-state 2.
    let state = if double { 2 } else { 1 };
    for _ in 0..if double { 2 } else { 1 } {
        let down = flap(true, state)?;
        post_event(&down);
        release(flap(false, state))?;
    }
    Ok(())
}

/// Post a created mouse-up event, or report why it could not be created. Once the
/// matching down is posted the attempt is never skipped, so a failed action
/// cannot leave the button held down and the desktop mid-drag.
fn release(made: Result<CFRetained<CGEvent>, anyhow::Error>) -> Result<(), anyhow::Error> {
    if let Ok(ev) = &made {
        post_event(ev);
    }
    made.map(drop)
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

fn press_key(key: &str) -> Result<(), anyhow::Error> {
    let code = lookup_key(key)?;
    let src = source();
    let down = keyboard_event(src.as_deref(), code as u16, true)?;
    post_event(&down);
    let up = keyboard_event(src.as_deref(), code as u16, false)?;
    post_event(&up);
    Ok(())
}

fn type_text(text: &str) -> Result<(), anyhow::Error> {
    let src = source();
    for ch in text.chars() {
        if ch == '\n' {
            press_key("return")?;
            continue;
        }
        let mut buf = [0u16; 2];
        let encoded = ch.encode_utf16(&mut buf);
        let down = keyboard_event(src.as_deref(), 0, true)?;
        // SAFETY: `encoded` is a valid UTF-16 buffer.
        unsafe {
            CGEvent::keyboard_set_unicode_string(
                Some(&down),
                encoded.len() as u64,
                encoded.as_ptr(),
            )
        };
        post_event(&down);
        let up = keyboard_event(src.as_deref(), 0, false)?;
        // SAFETY: `encoded` is a valid UTF-16 buffer.
        unsafe {
            CGEvent::keyboard_set_unicode_string(Some(&up), encoded.len() as u64, encoded.as_ptr())
        };
        post_event(&up);
        sleep_millis(1);
    }
    Ok(())
}

fn key_chord(chord: &str) -> Result<(), anyhow::Error> {
    let (modifiers, key) = core::parse_key_chord(chord)?;
    let code = lookup_key(&key)?;
    let src = source();
    let flags = cg_flags(&modifiers);
    let down = keyboard_event(src.as_deref(), code as u16, true)?;
    CGEvent::set_flags(Some(&down), flags);
    post_event(&down);
    let up = keyboard_event(src.as_deref(), code as u16, false)?;
    // Clear flags on key-up so the modifier doesn't stick.
    CGEvent::set_flags(Some(&up), CGEventFlags::empty());
    post_event(&up);
    Ok(())
}

/// Scroll: wheel 1 = vertical (down positive), wheel 2 = horizontal (right
/// positive). CGEvent deltas use +ve for down/right on macOS.
fn scroll(
    point: Option<CGPoint>,
    direction: ScrollDirection,
    amount: u32,
) -> Result<(), anyhow::Error> {
    let src = source();
    let (w1, w2) = match direction {
        ScrollDirection::Down => (amount as i32, 0),
        ScrollDirection::Up => (-(amount as i32), 0),
        ScrollDirection::Right => (0, amount as i32),
        ScrollDirection::Left => (0, -(amount as i32)),
    };
    let ev =
        CGEvent::new_scroll_wheel_event2(src.as_deref(), CGScrollEventUnit::Line, 2, w1, w2, 0)
            .ok_or_else(|| anyhow!("failed to create a scroll event"))?;
    if let Some(p) = point {
        CGEvent::set_location(Some(&ev), p);
    }
    post_event(&ev);
    Ok(())
}

fn drag(from: CGPoint, to: CGPoint) -> Result<(), anyhow::Error> {
    let src = source();
    let down = mouse_event(
        src.as_deref(),
        CGEventType::LeftMouseDown,
        from,
        CGMouseButton::Left,
    )?;
    post_event(&down);
    let moved = (|| -> Result<(), anyhow::Error> {
        for i in 1..=12 {
            let t = i as f64 / 12.0;
            let step = mouse_event(
                src.as_deref(),
                CGEventType::LeftMouseDragged,
                CGPoint::new(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t),
                CGMouseButton::Left,
            )?;
            post_event(&step);
            sleep_millis(2);
        }
        Ok(())
    })();
    let released = release(mouse_event(
        src.as_deref(),
        CGEventType::LeftMouseUp,
        to,
        CGMouseButton::Left,
    ));
    moved?;
    released
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
/// to [`surface_geometry_routed`], which keeps it on the calling thread: a missing
/// grant is requested there, never from the main queue.
fn surface_geometry_blocking(target: &TargetSpec) -> Result<SurfaceGeometry, anyhow::Error> {
    if matches!(target, TargetSpec::Screen) {
        return screen_surface();
    }
    window_frame(&resolve_window(target)?.0)
}

/// [`surface_geometry_blocking`] with the grant gate hoisted and the read routed
/// to the thread the system answers a self target on.
fn surface_geometry_routed(target: &TargetSpec) -> Result<SurfaceGeometry, anyhow::Error> {
    if !matches!(target, TargetSpec::Screen) {
        ensure(Grant::Accessibility)?;
    }
    with_target(target, surface_geometry_blocking)
}

/// Raise a window target so raw input aimed at it lands there. Best-effort: no
/// focused window (e.g. desktop frontmost) is fine for key chords/typing — the
/// geometry is only required by the point-based actions. Returns whether a raise
/// was actually performed, so the caller can let it settle off the main thread.
fn activate_window(target: &TargetSpec) -> Result<bool, anyhow::Error> {
    let (TargetSpec::Focused | TargetSpec::Window { .. }) = target else {
        return Ok(false);
    };
    let (window, _app) = match resolve_window(target) {
        Ok(resolved) => resolved,
        // Nothing to raise, but the main-thread marker must reach [`with_target`].
        Err(e) if !is_ax_thread(&e) => return Ok(false),
        Err(e) => return Err(e),
    };
    // Best-effort by design: an unusable answer only skips the raise.
    let Some(names) = raw_action_names(&window).ok().flatten() else {
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

/// Reject an argument mistake before any grant is requested: a call that can
/// never succeed must not raise a system permission dialog.
fn check_input_arguments(input: &RawInput) -> Result<(), anyhow::Error> {
    match input {
        RawInput::Click { point, .. } => core::check_normalized(point.0, point.1),
        RawInput::Scroll { point, .. } => {
            point.map_or(Ok(()), |(x, y)| core::check_normalized(x, y))
        }
        RawInput::Drag { from, to } => {
            core::check_normalized(from.0, from.1)?;
            core::check_normalized(to.0, to.1)
        }
        RawInput::KeyChord { chord } => lookup_key(&core::parse_key_chord(chord)?.1).map(|_| ()),
        RawInput::TypeText { .. } => Ok(()),
    }
}

fn raw_input_blocking(target: &TargetSpec, input: &RawInput) -> Result<(), anyhow::Error> {
    // The 80 ms settle after a raise runs here, on the worker: only the raise
    // itself needs the target's own thread.
    if with_target(target, activate_window)? {
        sleep_millis(80);
    }
    match input {
        RawInput::Click {
            point,
            button,
            double,
            modifiers,
        } => {
            let (x, y) =
                core::normalized_to_surface(point.0, point.1, &surface_geometry_routed(target)?)?;
            click(CGPoint::new(x, y), *button, *double, modifiers)?;
        }
        RawInput::TypeText { text } => type_text(text)?,
        RawInput::KeyChord { chord } => key_chord(chord)?,
        RawInput::Scroll {
            point,
            direction,
            amount,
        } => {
            let p = if let Some((nx, ny)) = point {
                let (x, y) =
                    core::normalized_to_surface(*nx, *ny, &surface_geometry_routed(target)?)?;
                Some(CGPoint::new(x, y))
            } else {
                None
            };
            scroll(p, *direction, *amount)?;
        }
        RawInput::Drag { from, to } => {
            let geometry = surface_geometry_routed(target)?;
            let (fx, fy) = core::normalized_to_surface(from.0, from.1, &geometry)?;
            let (tx, ty) = core::normalized_to_surface(to.0, to.1, &geometry)?;
            drag(CGPoint::new(fx, fy), CGPoint::new(tx, ty))?;
        }
    }
    Ok(())
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
    unsafe {
        SCShareableContent::getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler(false, true, &block);
    }
    sc_wait::<SendShareable>(&slot, &semaphore, "shareable content").map(|SendShareable(c)| c)
}

/// A single still of `filter`'s content, at the pixel size the caller asked for.
///
/// The pixel size is validated here because it is what reaches the framework: both
/// capture routes derive it from what the system reported — a window's frame, a
/// display's frame — and zero is what a minimized, off-screen or vanished surface
/// reduces to, so it must be an honest failure rather than a request handed to
/// ScreenCaptureKit.
///
/// Callers guarantee [`sck_available`]: the `SCScreenshotManager` class itself
/// only exists from macOS 14 on, and messaging a missing class aborts the
/// process.
fn capture_filter_to_rgba(
    filter: &SCContentFilter,
    px_w: usize,
    px_h: usize,
    ignore_shadows: bool,
) -> Result<Capture, anyhow::Error> {
    if px_w == 0 || px_h == 0 {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "the target has no capture area ({px_w}x{px_h} px) — it is minimized, off-screen \
                 or was closed; restore it and re-observe"
            ),
        ));
    }
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
    capture_image_to_rgba(&image)
}

/// Turn a CGImage into top-down RGBA bytes. CG contexts use a bottom-left
/// origin, so the image is flipped (translate+scale) before drawing to yield
/// top-down rows.
fn capture_image_to_rgba(image: &CGImage) -> Result<Capture, anyhow::Error> {
    let w = CGImage::width(Some(image));
    let h = CGImage::height(Some(image));
    if w == 0 || h == 0 {
        anyhow::bail!("captured image has zero size");
    }
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
    // SAFETY: `data` points to `w*h*4` bytes owned by the context; copied before drop.
    let rgba = unsafe { std::slice::from_raw_parts(data as *const u8, w * h * 4) }.to_vec();
    Ok(Capture {
        width: w as u32,
        height: h as u32,
        rgba,
    })
}

fn find_sc_window(
    content: &SCShareableContent,
    pid: u32,
    title: Option<&str>,
    frame: SurfaceGeometry,
) -> Option<Retained<SCWindow>> {
    // SAFETY: SCK getter methods are ObjC messages against live objects.
    unsafe { content.windows() }.iter().find(|window| {
        // SAFETY: ObjC message against a live SCWindow/SCRunningApplication.
        let owner = unsafe { window.owningApplication() };
        let owner_pid = owner.as_ref().map(|a| unsafe { a.processID() } as u32);
        if owner_pid != Some(pid) {
            return false;
        }
        if let Some(t) = title {
            // SAFETY: ObjC message against a live SCWindow.
            if unsafe { window.title() }.map(|s| s.to_string()).as_deref() == Some(t) {
                return true;
            }
        }
        // Fall back to frame proximity when the title didn't match.
        // SAFETY: ObjC message against a live SCWindow.
        let f = unsafe { window.frame() };
        (f.origin.x - frame.x).abs() < 2.0
            && (f.origin.y - frame.y).abs() < 2.0
            && (f.size.width - frame.width).abs() < 2.0
            && (f.size.height - frame.height).abs() < 2.0
    })
}

/// The AX half of a window still: which process owns the window, its title and
/// its frame — the facts ScreenCaptureKit is then matched against.
fn window_capture_facts(
    target: &TargetSpec,
) -> Result<(u32, Option<String>, SurfaceGeometry), anyhow::Error> {
    let (window, app) = resolve_window(target)?;
    Ok((
        app_pid(&app)?,
        window_title(&window),
        window_frame(&window)?,
    ))
}

fn capture_window_blocking(target: &TargetSpec) -> Result<Capture, anyhow::Error> {
    // The window is resolved through AX (pid, title, frame) before SCK is asked
    // for pixels; only that half belongs on the AX thread, never the capture.
    let (pid, title, frame) = with_target(target, window_capture_facts)?;
    let content = get_shareable_content()?;
    let sc_window = find_sc_window(&content, pid, title.as_deref(), frame).ok_or_else(|| {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "window not found in shareable content — re-enumerate windows",
        )
    })?;
    // SAFETY: `alloc()` is the fresh allocation consumed by the init.
    let filter = unsafe {
        SCContentFilter::initWithDesktopIndependentWindow(SCContentFilter::alloc(), &sc_window)
    };
    let scale = unsafe { filter.pointPixelScale() } as f64;
    let px_w = (frame.width * scale).round();
    let px_h = (frame.height * scale).round();
    capture_filter_to_rgba(&filter, px_w as usize, px_h as usize, true)
}

/// Composite each active display into one RGBA buffer at the virtual-desktop
/// bounding rect, each display drawn at its global-frame offset. Each display is
/// captured and placed at ITS OWN pixel scale, so a mixed-density setup renders
/// each screen at its native density; when densities differ the composite is
/// sized at the first display's reference scale and a lower-density display
/// occupies a proportionally smaller region of the buffer (an accepted residual
/// limitation of a single-buffer composite).
fn capture_screen_blocking() -> Result<Capture, anyhow::Error> {
    let content = get_shareable_content()?;
    // SAFETY: SCK getter is an ObjC message against a live object.
    let displays = unsafe { content.displays() };
    if displays.is_empty() {
        anyhow::bail!("no shareable displays found");
    }
    let empty: Retained<NSArray<SCWindow>> = NSArray::new();
    let first = displays.iter().next().expect("displays non-empty");
    // SAFETY: `alloc()` is the fresh allocation consumed by the init.
    let first_filter = unsafe {
        SCContentFilter::initWithDisplay_excludingWindows(SCContentFilter::alloc(), &first, &empty)
    };
    let ref_scale = unsafe { first_filter.pointPixelScale() } as f64;
    let bbox = screen_surface()?;
    let out_w = (bbox.width * ref_scale).round() as usize;
    let out_h = (bbox.height * ref_scale).round() as usize;
    if out_w == 0 || out_h == 0 {
        anyhow::bail!("virtual desktop has zero size");
    }
    let mut composite = vec![0u8; out_w * out_h * 4];
    for display in displays {
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
        let scale = unsafe { filter.pointPixelScale() } as f64;
        let px_w = (f.size.width * scale).round() as usize;
        let px_h = (f.size.height * scale).round() as usize;
        let cap = capture_filter_to_rgba(&filter, px_w, px_h, false)?;
        let dx = ((f.origin.x - bbox.x) * scale).round();
        let dy = ((f.origin.y - bbox.y) * scale).round();
        if dx >= 0.0 && dy >= 0.0 {
            core::blit_rgba(&mut composite, out_w, out_h, &cap, dx as usize, dy as usize);
        }
    }
    Ok(Capture {
        width: out_w as u32,
        height: out_h as u32,
        rgba: composite,
    })
}

fn capture_blocking(target: &TargetSpec) -> Result<Capture, anyhow::Error> {
    if !sck_available() {
        return Err(screen_capture_needs_14());
    }
    // A window still is resolved through AX (its frame and title) before SCK is
    // asked for pixels, so the Accessibility grant is needed as well.
    if !matches!(target, TargetSpec::Screen) {
        ensure(Grant::Accessibility)?;
    }
    let result = match target {
        TargetSpec::Screen => capture_screen_blocking(),
        _ => capture_window_blocking(target),
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

fn press_element(element: &AXUIElement) -> Result<(), anyhow::Error> {
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

fn set_element_value(element: &AXUIElement, text: &str) -> Result<(), anyhow::Error> {
    let role = attribute_string(element, &axc(AX_ROLE)).unwrap_or_default();
    let subrole = attribute_string(element, &axc(AX_SUBROLE));
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
) -> Result<(), anyhow::Error> {
    let (window, _app) = resolve_window(target)?;
    let (root, handles) = build_tree(&window, window_origin(&window)?);
    let matched = core::resolve_locator_checked(&root, locator)?;
    let element = &handles[core::pre_order_index(&root, matched)];
    match act {
        ElementAct::Press => press_element(element),
        ElementAct::SetValue { text } => set_element_value(element, text),
    }
}

fn map_action_err(err: AXError, ctx: &str) -> Result<(), anyhow::Error> {
    match err {
        AXError::Success => Ok(()),
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
        AXError::CannotComplete => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("{ctx} — transient AX failure, retry"),
        )),
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

    async fn list_windows(&self, app: Option<&AppInfo>) -> Result<Vec<WindowInfo>, anyhow::Error> {
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
            let mut out = Vec::new();
            for ai in app_infos {
                let Some(pid) = ai.pid else { continue };
                let name = ai.name;
                // An app whose accessibility is unavailable is skipped, as
                // before; the AX thread only makes our OWN window's reads
                // possible at all. A listing of ONE app is the caller's whole
                // request, so its failure is reported instead.
                match on_ax_thread(pid, move || windows_of_app(pid, &name)) {
                    Ok(rows) => out.extend(rows),
                    // A single-app listing of our OWN window says why it is
                    // missing; every other failure keeps the old "answer what
                    // was read" shape, and the unfiltered listing must never
                    // fail as a whole because of one app.
                    Err(e) if app.is_some() && pid == std::process::id() => return Err(e),
                    Err(_) => {}
                }
            }
            Ok(out)
        })
    }

    async fn focused_window(&self) -> Result<WindowInfo, anyhow::Error> {
        with_platform("read the focused window", || {
            ensure(Grant::Accessibility)?;
            with_target(&TargetSpec::Focused, |target| {
                let (window, app) = resolve_window(target)?;
                let pid = app_pid(&app)?;
                Ok(WindowInfo {
                    app_pid: Some(pid),
                    app_name: app_display_name(pid, &app),
                    title: window_title(&window).unwrap_or_default(),
                    index: 0,
                    surface: window_frame(&window)?,
                })
            })
        })
    }

    async fn surface_geometry(
        &self,
        target: &TargetSpec,
    ) -> Result<SurfaceGeometry, anyhow::Error> {
        with_platform("read the target's geometry", || {
            surface_geometry_routed(target)
        })
    }

    async fn observe(&self, target: &TargetSpec) -> Result<Observation, anyhow::Error> {
        with_platform("observe", || {
            ensure(Grant::Accessibility)?;
            with_target(target, observe_blocking)
        })
    }

    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> Result<(), anyhow::Error> {
        with_platform("act on an element", || {
            if matches!(target, TargetSpec::Screen) {
                return Err(core::screen_act_error());
            }
            ensure(Grant::Accessibility)?;
            let locator = locator.clone();
            with_target(target, move |target| {
                act_on_element_blocking(target, &locator, &act)
            })
        })
    }

    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> Result<(), anyhow::Error> {
        with_platform("synthesize input", || {
            check_input_arguments(&input)?;
            ensure(Grant::Input)?;
            raw_input_blocking(target, &input)
        })
    }

    async fn cursor_position(&self) -> Result<(f64, f64), anyhow::Error> {
        with_platform("read the pointer position", || {
            let event =
                CGEvent::new(None).ok_or_else(|| anyhow!("failed to create dummy cursor event"))?;
            let p = CGEvent::location(Some(&event));
            Ok((p.x, p.y))
        })
    }

    async fn capture(&self, target: &TargetSpec) -> Result<Capture, anyhow::Error> {
        with_platform("capture", || capture_blocking(target))
    }
}

fn window_title(window: &AXUIElement) -> Option<String> {
    attribute_string(window, &axc(AX_TITLE))
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
