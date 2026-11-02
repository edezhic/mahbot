//! Platform-agnostic core of the Computer (GUI observe/act) tool: shared types, coordinate
//! math, the element-ref lifecycle, the remembered look of a surface, the target registry,
//! tree rendering and the [`Backend`] trait — no platform imports (`macos.rs`, `linux.rs`,
//! `windows`, `stub.rs` implement it).
//!
//! Three honesty rules every backend owes: a read states every ceiling it hit ([`Truncation`]);
//! an actuating call states which of its three outcomes it reached ([`Impact`]) and, having
//! placed a point, where ([`Actuation`]); and what a surface is or was is judged against the one
//! remembered look of it, never a second copy. Platform-scoped `expect(dead_code)` marks items
//! only a real backend or a test calls: no caller on a stub platform, but they are the real
//! backends' contract, so they stay.

use crate::util::UnwrapPoison;
use anyhow::anyhow;
use serde::Deserialize;
use serde::Serialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

pub(crate) const NORMALIZED_MAX: f64 = 1000.0;

pub(crate) const AX_THIN_INTERACTIVE_ELEMENTS: usize = 5;

#[cfg(any(target_os = "macos", target_os = "linux", test))]
pub(crate) const MAX_TREE_NODES: usize = 4000;

/// Depth every tree read walks to. The backends stop there and record the cut, so a
/// narrow look cannot ask for levels the read could not have produced.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) const MAX_TREE_DEPTH: usize = 25;

/// Ceiling on the children ONE node may contribute: a target advertising an absurd child list
/// (a spreadsheet's cell list, a count near 2^31) would otherwise be paid for inside a single
/// exchange, so the advertised count is read first and a node beyond this is not walked.
#[cfg(any(target_os = "linux", test))]
pub(crate) const MAX_CHILDREN: usize = 1000;

/// Character cap on a node value shown in a tree dump — the dump is a map of the surface, not a
/// copy of its text, so the cut is counted in the truncation header.
pub(crate) const RENDER_VALUE_CHARS: usize = 120;

/// Most characters one node value keeps, on every platform: the read's cap, not the dump's — a
/// document-shaped field reports a whole file, which would inflate the tree per node. A cut value
/// keeps its beginning and is counted in the header ([`Truncation::keep_value`]); Linux, reading
/// one character past the dump's cap, never reaches this.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) const MAX_VALUE_CHARS: usize = 2000;

/// Most characters one accessible name keeps, as with a value ([`Truncation::keep_name`]): a
/// title can be a document too. A ref matches the name it saw against a freshly read one, cut the
/// same way both sides; the child-index path is the primary key, so a capped name only decides the
/// fallback — names differing past the cap resolve as [`LocatorMatch::Ambiguous`].
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) const MAX_NAME_CHARS: usize = 200;

/// Longest string one computer action accepts, in characters — `text`, `keys`, `target`, `app`
/// and `ref` alike. Typing pays one synthetic event per character under the action lock, so a
/// longer argument is refused before the call starts work.
pub(crate) const MAX_ARG_CHARS: usize = 2000;

/// The scroll range in wheel ticks, on every platform: one tick is one notch, and it
/// costs one wheel event on macOS and Windows but a press/release pair per tick on
/// Linux — under the same lock. Zero is outside the range too: it names no motion.
pub(crate) const MAX_SCROLL_AMOUNT: u32 = 100;

/// Ceiling on the pixels ONE capture call may hold at once, chosen by the product rather than taken
/// from what the machine reports: this process aborts on a failed allocation, so no reported figure
/// may decide how much is allocated. Every capture buffer — a display's frame, the whole-desktop
/// composite, the zoom crop, a portal screenshot's pixels — holds a [`Claim`] on it, weighed before
/// the buffer exists and given back when dropped. 128 Mi pixels is 512 MiB of RGBA: a
/// two-6K-display wall fits, four 6K displays are refused naming this ceiling. A display's frame
/// lives only while blitted, so displays do not multiply the cost; platform-owned memory fails with
/// a null handle or an `HRESULT` instead of aborting.
pub(crate) const CAPTURE_BUDGET_PIXELS: u64 = 128 * 1024 * 1024;

// Every exchange is bounded by a deadline the product chose, and every call by a bound on top of
// that, so a target that does not answer costs a lost detail instead of the action lock. The model
// meets one only in the failure that reports the bound it hit.

/// How long ONE accessibility exchange may take before its target counts as not
/// answering: the D-Bus method timeout on Linux and the AX messaging timeout on macOS,
/// deliberately not shorter than the OS default applications rely on (1.5 s measured on
/// a stopped application here). Windows sets its own transaction timeout.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const EXCHANGE_DEADLINE: Duration = Duration::from_secs(10);

/// The aggregate budget of one observation, resolution included: well clear of what a responsive
/// target at the node ceiling pays (~2.5 s of AX exchanges), so only a stopped target reaches it.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const OBSERVE_BUDGET: Duration = Duration::from_secs(120);

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const ACT_BUDGET: Duration = OBSERVE_BUDGET.saturating_add(EXCHANGE_DEADLINE);

/// One listing, application walk or surface lookup — a handful of exchanges per
/// application, where a hung one must not hold the call either.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const LIST_BUDGET: Duration = Duration::from_secs(20);

/// What ONE application may spend of a listing's budget: one stalled exchange, checked before
/// every read it makes, so a hung application is caught at the check after its stall. A share is
/// overshot only by the read in flight; the healthy applications keep the rest of the call.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const APP_SLICE: Duration = EXCHANGE_DEADLINE;

/// Why a listing skips one item it names: it did not answer within its share of the call.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const SKIPPED_NOTE: &str = "did not answer within its share of the call — skipped";

/// Why a listing calls one item unread: a spent share is one skipped item, said in
/// [`SKIPPED_NOTE`] since the line names it and the call returned; anything else is stated in full.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn unread_reason(err: &anyhow::Error, share: &Budget) -> String {
    if share.expired() {
        SKIPPED_NOTE.to_string()
    } else {
        format!("{err:#}")
    }
}

/// One call's aggregate deadline, checked between exchanges, because a platform walk runs inside
/// a blocking frame no external timer can interrupt. Linux and macOS thread one through resolution
/// and the walk; Windows budgets its walk and bounds the rest with the job deadline.
#[derive(Clone)]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) struct Budget {
    end: Instant,
    limit: Duration,
    what: &'static str,
    target: String,
    part: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Budget {
    /// Start the clock for `what` (the operation the model asked for) against `target`.
    pub(crate) fn new(limit: Duration, what: &'static str, target: impl Into<String>) -> Self {
        Self {
            end: Instant::now() + limit,
            limit,
            what,
            target: target.into(),
            part: false,
        }
    }

    /// One application's share: at most `share` from now, never past the call's own end, so the
    /// step it bounds cannot outlive the call. Running out skips that application, not the call.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn slice(&self, share: Duration, target: impl Into<String>) -> Self {
        let now = Instant::now();
        let end = self.end.min(now + share);
        Self {
            end,
            limit: end.saturating_duration_since(now),
            what: self.what,
            target: target.into(),
            part: true,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn target(&self) -> &str {
        &self.target
    }

    /// Point the budget at the surface the call resolved: every failure from here names the window
    /// rather than the target. [`name_surface`] is the only caller, setting both together.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn retarget(&mut self, target: &str) {
        target.clone_into(&mut self.target);
    }

    pub(crate) fn expired(&self) -> bool {
        Instant::now() >= self.end
    }

    pub(crate) fn check(&self) -> anyhow::Result<()> {
        if self.expired() {
            Err(self.error())
        } else {
            Ok(())
        }
    }

    pub(crate) fn note(&self) -> String {
        tracing::warn!(
            target = %self.target,
            limit_secs = self.limit.as_secs(),
            part = self.part,
            "computer use: the {} budget ran out",
            self.what
        );
        if self.part {
            format!(
                "{} stopped answering accessibility requests — it was skipped, so the {} went on \
                 without it",
                self.target, self.what
            )
        } else {
            format!(
                "{} stopped answering accessibility requests — the {} was abandoned after {}s; {}",
                self.target,
                self.what,
                self.limit.as_secs(),
                unanswered_next_step()
            )
        }
    }

    fn error(&self) -> anyhow::Error {
        taxonomy_error(ERR_DEGRADED, self.note())
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn surface_label(app_name: &str, title: Option<&str>) -> String {
    window_description(&cap_label(app_name), &cap_label(title.unwrap_or_default()))
}

/// The label a listing or search uses for one application's windows, capped like any name.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn windows_label(app_name: &str) -> String {
    format!("the windows of {}", cap_label(app_name))
}

/// The same label, and the budget pointed at it: the one place the identity → label → budget rule
/// lives, so a call cannot report a surface one way and bound it under another.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn name_surface(budget: &mut Budget, app_name: &str, title: Option<&str>) -> String {
    let label = surface_label(app_name, title);
    budget.retarget(&label);
    label
}

/// The step a surface that stopped answering leaves the caller: a still when stills are known to
/// work, else the retry that needs no capture channel — the one owner of that choice.
pub(crate) fn unanswered_next_step() -> &'static str {
    if super::capture_known_available() {
        "retry, or use screenshot instead"
    } else {
        "retry it"
    }
}

/// The failure for a surface that did not answer one of the reads a call makes before it can show,
/// act on or capture anything. AT-SPI and AX both report an unanswered request as a missing value,
/// so the target is named with what was asked of it, not as an absent attribute.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn unanswered(surface: &str, what: &str) -> anyhow::Error {
    taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{surface} did not answer {what} — it may be busy or hung; {}",
            unanswered_next_step()
        ),
    )
}

const INTERACTIVE_ROLES: &[&str] = &[
    "button",
    "checkbox",
    "radiobutton",
    "menuitem",
    "menu",
    "combobox",
    "textfield",
    "textarea",
    "searchfield",
    "slider",
    "tab",
    "tabgroup",
    "link",
    "listitem",
    "popupbutton",
    "stepper",
    "colorwell",
    "datefield",
    "tokenfield",
    "outline",
    "table",
];

// Backend/runtime failures (permission, stale/ambiguous elements, unsupported surfaces) MUST
// start with a taxonomy word (via [`taxonomy_error`]) so a model can branch on the category;
// argument-validation errors are plain and self-describing — no tag needed.

pub(crate) const ERR_PERMISSION_DENIED: &str = "permission-denied";
pub(crate) const ERR_UNSUPPORTED: &str = "unsupported";
pub(crate) const ERR_DEGRADED: &str = "degraded";
/// The session cannot receive the call at all — see [`Backend::desktop_unavailable`].
pub(crate) const ERR_DESKTOP_UNAVAILABLE: &str = "desktop-unavailable";
pub(crate) const ERR_STALE_ELEMENT: &str = "stale-element";
/// The system's framework (on Windows, a structured exception inside it) failed while serving the
/// call: abandoned, process intact. Retrying is unlikely to help — the fault is foreign code.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) const ERR_FRAMEWORK_FAILED: &str = "framework-failed";
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) const ERR_AMBIGUOUS_LOCATOR: &str = "ambiguous-locator";
pub(crate) const ERR_NOT_MATCHED: &str = "not-matched";
/// The surface a call names is gone: the window closed or was replaced, or its application quit.
/// Deliberately not `not-matched` (the name did not match; re-enumerate apps/windows): that would
/// report a followed surface's disappearance as a misnamed target. Raised through [`surface_gone`]
/// and [`app_absent`], whose typed [`Gone`] lets the tool drop a dead id.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) const ERR_SURFACE_GONE: &str = "surface-gone";
/// The target is mahbot's own interface and this platform cannot act on itself from inside the
/// process. Distinct from `unsupported`, whose failures name a route that would keep it spinning.
#[cfg_attr(not(target_os = "windows"), expect(dead_code))]
pub(crate) const ERR_SELF_TARGET: &str = "self-target";
/// The point a click was aimed at lies over another application's window, so nothing was sent.
/// Its own word because it is the only refusal a plain repeat clears: the covering window may go
/// away on its own, and every call raises the addressed window again.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) const ERR_BLOCKED: &str = "blocked";

#[must_use]
pub(crate) fn taxonomy_error(kind: &str, msg: impl fmt::Display) -> anyhow::Error {
    anyhow!("{kind}: {msg}")
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn framework_failed(what: &str, detail: impl fmt::Display) -> anyhow::Error {
    anyhow::Error::new(FrameworkFailed(taxonomy_error(
        ERR_FRAMEWORK_FAILED,
        format!("{what} — {detail}"),
    )))
}

/// Whether `err` is the typed [`framework_failed`] failure — robust to a `.context()` layer above
/// it. The answer path asks on every platform, since only macOS and Windows raise it.
pub(crate) fn is_framework_failed(err: &anyhow::Error) -> bool {
    err.downcast_ref::<FrameworkFailed>().is_some()
}

#[derive(Debug)]
struct FrameworkFailed(anyhow::Error);

impl fmt::Display for FrameworkFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for FrameworkFailed {}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
#[derive(Debug)]
struct Gone(anyhow::Error);

impl fmt::Display for Gone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for Gone {}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn is_surface_gone(err: &anyhow::Error) -> bool {
    err.downcast_ref::<Gone>().is_some()
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn surface_gone(app_name: &str) -> anyhow::Error {
    anyhow::Error::new(Gone(taxonomy_error(
        ERR_SURFACE_GONE,
        format!(
            "the window of '{app_name}' this target names can no longer be identified — it \
             closed, was replaced, or its application quit; list windows again with the \
             'windows' action"
        ),
    )))
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum AppAbsence {
    Quit,
    Windowless,
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "windows", test)),
        expect(dead_code)
    )]
    Unreachable,
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn app_absent(app_name: &str, absence: AppAbsence) -> anyhow::Error {
    match absence {
        AppAbsence::Quit => anyhow::Error::new(Gone(taxonomy_error(
            ERR_SURFACE_GONE,
            format!(
                "'{app_name}' is no longer running — the application this identifier names has \
                 quit; the 'apps' action lists what is running now"
            ),
        ))),
        AppAbsence::Windowless => taxonomy_error(
            ERR_SURFACE_GONE,
            format!(
                "'{app_name}' is running but has no window to act on — the 'windows' action lists \
                 its windows"
            ),
        ),
        AppAbsence::Unreachable => taxonomy_error(
            ERR_SURFACE_GONE,
            format!(
                "no window of '{app_name}' could be found, and this call could not establish \
                 whether the application is still running — the 'apps' action lists what is \
                 reachable"
            ),
        ),
    }
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn minimized_window(title: &str) -> anyhow::Error {
    taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{} is minimized — restore it first; a minimized window has no visible pixels",
            window_description("", title)
        ),
    )
}

/// The `degraded` failure for pinned input whose surface could not be brought to the front:
/// nothing was sent, so the call cannot claim it landed there. `surface` is its spelled name.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn not_brought_forward(surface: impl fmt::Display) -> anyhow::Error {
    taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{surface} could not be brought to the front — the input would have gone to another \
             window, so nothing was sent; bring the window forward and repeat the action"
        ),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowAtPoint {
    pub app_pid: Option<u32>,
    pub app_name: String,
    pub title: String,
    pub frame: Option<SurfaceGeometry>,
}

impl WindowAtPoint {
    /// Model-facing name of this window, never an empty pair of quotes.
    #[must_use]
    pub(crate) fn describe(&self) -> String {
        window_description(&self.app_name, &self.title)
    }
}

pub(crate) const SCREEN_POINT_NOTE: &str =
    "the click names the screen, so the window under the point was not checked";

/// What a click's answer says where the platform cannot attribute a window to a point.
pub(crate) const UNCHECKED_POINT_NOTE: &str =
    "the window under the point was not identified, so the click's landing could not be checked";

/// What the point of a coordinate click would land on, decided as the click is sent. Only a
/// platform that can attribute a window to an application decides the three middle cases.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum UnderPoint {
    Screen,
    /// The platform cannot attribute the window at that point: the click goes, and the answer
    /// says the point could not be checked.
    Unknown,
    Clear,
    ThatWindow(WindowAtPoint),
    Blocked(WindowAtPoint),
}

impl UnderPoint {
    /// Decide from what the platform answered; `None` is "could not say", and the click goes with
    /// the answer saying the point was not checked. Windows are first told apart by owning process
    /// (two maximized windows share a frame exactly), then the addressed window by its frame (a
    /// title could not: systems spell it differently). Its own dialog, menu or tooltip, and an
    /// element with no window of its own, are not obstacles. An unattributable window is the
    /// unknown case — the click goes, the answer says it was not checked.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "windows", test)),
        expect(dead_code)
    )]
    #[must_use]
    pub(crate) fn classify(
        surface: &TargetSpec,
        addressed_pid: Option<u32>,
        addressed_frame: Option<&SurfaceGeometry>,
        at: Option<&WindowAtPoint>,
    ) -> Self {
        if matches!(surface, TargetSpec::Screen) {
            return Self::Screen;
        }
        let Some(at) = at else {
            return Self::Unknown;
        };
        match (addressed_pid, at.app_pid) {
            (Some(addressed), Some(over)) if addressed != over => Self::Blocked(at.clone()),
            (Some(_), Some(_)) => match (addressed_frame, at.frame.as_ref()) {
                (Some(addressed), Some(over)) if same_frame(addressed, over) => Self::Clear,
                (Some(_), Some(_)) => Self::ThatWindow(at.clone()),
                // The application is the same, so nothing of another application is in the way;
                // only which of its windows the point is on cannot be said.
                _ => Self::Clear,
            },
            // The owner of a window could not be read, so the point cannot be attributed.
            _ => Self::Unknown,
        }
    }

    fn disclosure(&self, check: &ClickCheck<'_>) -> anyhow::Result<Option<String>> {
        match self {
            Self::Clear => Ok(None),
            Self::Screen => Ok(Some(SCREEN_POINT_NOTE.to_string())),
            Self::Unknown => Ok(Some(UNCHECKED_POINT_NOTE.to_string())),
            Self::ThatWindow(at) => Ok(Some(format!(
                "the point lay over {}, another window of the same application",
                at.describe()
            ))),
            // The refusal states surface and position, names the application in the way, says
            // nothing was sent and gives the two routes that work: an element of the addressed
            // surface, or a retry once that window is out of the way.
            Self::Blocked(at) => Err(taxonomy_error(
                ERR_BLOCKED,
                format!(
                    "nothing was sent — the point at {} on {} lay over {}, which belongs to \
                     another application, so the click would have landed there; act on an \
                     element of the surface instead, or click again once that window is out of \
                     the way",
                    placed_point(check.local, check.geometry),
                    check.surface.describe(),
                    at.describe()
                ),
            )),
        }
    }
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) struct ClickCheck<'a> {
    pub surface: &'a TargetSpec,
    pub local: (f64, f64),
    pub geometry: &'a SurfaceGeometry,
    pub addressed_pid: Option<u32>,
    pub addressed_frame: Option<&'a SurfaceGeometry>,
    pub under: Option<&'a WindowAtPoint>,
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn click_under_point(check: &ClickCheck<'_>) -> anyhow::Result<Option<String>> {
    UnderPoint::classify(
        check.surface,
        check.addressed_pid,
        check.addressed_frame,
        check.under,
    )
    .disclosure(check)
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn permission_denied(msg: impl fmt::Display) -> anyhow::Error {
    anyhow::Error::new(PermissionDenied(taxonomy_error(ERR_PERMISSION_DENIED, msg)))
}

/// The `degraded` refusal for a capture over the call's [`CAPTURE_BUDGET_PIXELS`] allowance. TYPED
/// so the Windows route classifier can tell a ceiling refusal (remedy: a smaller surface) from
/// every other degraded cause without reading the message.
fn ceiling_refused(what: &str, width: u32, height: u32, held: u64) -> anyhow::Error {
    anyhow::Error::new(CeilingRefusal(taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{what} needs {} MiB of RGBA ({width}x{height}) and this call already holds {} MiB \
             of its {} MiB — a capture this large cannot be held whole; name a smaller surface \
             if one exists",
            mib(u64::from(width) * u64::from(height)),
            mib(held),
            mib(CAPTURE_BUDGET_PIXELS),
        ),
    )))
}

#[cfg_attr(not(any(target_os = "windows", test)), expect(dead_code))]
pub(crate) fn is_ceiling_refusal(err: &anyhow::Error) -> bool {
    err.downcast_ref::<CeilingRefusal>().is_some()
}

/// A capture refused at the ceiling, as a type (see [`ceiling_refused`]).
#[derive(Debug)]
struct CeilingRefusal(anyhow::Error);

impl fmt::Display for CeilingRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for CeilingRefusal {}

pub(crate) fn is_permission_denied(err: &anyhow::Error) -> bool {
    err.downcast_ref::<PermissionDenied>().is_some()
}

#[derive(Debug)]
struct PermissionDenied(anyhow::Error);

impl fmt::Display for PermissionDenied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for PermissionDenied {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurfaceGeometry {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Normalized 0-1000 region (consumers normalise corner order). Constructed at runtime only by
/// serde from `ComputerAction::Zoom` arguments; `Default` yields an all-zero (invalid) region and
/// exists only so the `EnumIter` schema lockstep test can build a `Zoom` via `Default::default()`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Default)]
pub(crate) struct Region {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl<'de> Deserialize<'de> for Region {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RegionVisitor;

        impl<'de> Visitor<'de> for RegionVisitor {
            type Value = Region;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a 4-element array [x0, y0, x1, y1] or object {x0, y0, x1, y1}")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let x0 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("x0"))?;
                let y0 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("y0"))?;
                let x1 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("x1"))?;
                let y1 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("y1"))?;
                if seq.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(
                        "a region is exactly [x0, y0, x1, y1] — extra values name nothing",
                    ));
                }
                Ok(Region { x0, y0, x1, y1 })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut x0 = None;
                let mut y0 = None;
                let mut x1 = None;
                let mut y1 = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "x0" => x0 = Some(map.next_value()?),
                        "y0" => y0 = Some(map.next_value()?),
                        "x1" => x1 = Some(map.next_value()?),
                        "y1" => y1 = Some(map.next_value()?),
                        other => {
                            return Err(de::Error::unknown_field(other, &["x0", "y0", "x1", "y1"]));
                        }
                    }
                }
                Ok(Region {
                    x0: x0.ok_or_else(|| de::Error::missing_field("x0"))?,
                    y0: y0.ok_or_else(|| de::Error::missing_field("y0"))?,
                    x1: x1.ok_or_else(|| de::Error::missing_field("x1"))?,
                    y1: y1.ok_or_else(|| de::Error::missing_field("y1"))?,
                })
            }
        }

        deserializer.deserialize_any(RegionVisitor)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Locator {
    pub path: Vec<usize>,
    pub role: String,
    pub name: Option<String>,
    pub frame: Option<(f64, f64, f64, f64)>,
}

#[derive(Debug, Clone)]
pub(crate) struct UiNode {
    pub role: String,
    pub name: Option<String>,
    pub value: Option<String>,
    pub frame: Option<(f64, f64, f64, f64)>,
    pub actions: Vec<String>,
    pub focused: bool,
    pub children: Vec<UiNode>,
}

/// What a platform backend can honestly say about one actuating call.
///
/// Input synthesis is fire-and-forget everywhere: [`Impact::Done`] means the call was delivered
/// without error — never that its effect happened. [`Impact::DoneWith`] appends the one thing a
/// delivered call still has to disclose. [`Impact::Unconfirmed`] is the middle state — part or all
/// of it may have been applied and the platform cannot say which — and is a normal result, not a
/// failure: a failure reads "nothing happened, do it again", and a repeated click is a second
/// actuation.
#[derive(Debug, Clone)]
pub(crate) enum Impact {
    Done,
    DoneWith(String),
    /// Carries what is known: how much went through, and why it stopped.
    Unconfirmed(String),
}

#[must_use]
pub(crate) fn impact_text(done: String, impact: Impact) -> String {
    match impact {
        Impact::Done => done,
        Impact::DoneWith(note) => format!("{done} — {note}"),
        Impact::Unconfirmed(detail) => format!(
            "applied but unconfirmed — {done}: {detail}. The action was delivered to the system \
             but not confirmed; it may already have taken effect, so check the target's state \
             before repeating it, and never repeat it blind, because a second attempt would act \
             twice."
        ),
    }
}

/// Add one fact to an outcome without changing what it says: a delivered call stays delivered, an
/// unconfirmed one stays unconfirmed (a failed probe must not make a delivered action ambiguous).
#[must_use]
pub(crate) fn impact_note(impact: Impact, note: String) -> Impact {
    match impact {
        Impact::Done => Impact::DoneWith(note),
        Impact::DoneWith(existing) => Impact::DoneWith(format!("{existing}; {note}")),
        Impact::Unconfirmed(detail) => Impact::Unconfirmed(format!("{detail}; {note}")),
    }
}

/// What one input call delivered, and where it put the pointer. The geometry comes from the backend
/// that posted the event, so an answer cannot quote a point from a different read of the surface.
#[derive(Debug, Clone)]
pub(crate) struct Actuation {
    pub impact: Impact,
    pub placed: Option<Placement>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement {
    pub geometry: SurfaceGeometry,
    pub from: (f64, f64),
    /// Where a drag ended — `None` for a click, a wheel, and a drag whose end was never placed.
    pub to: Option<(f64, f64)>,
}

impl Placement {
    #[must_use]
    pub(crate) fn describe(&self) -> String {
        match self.to {
            Some(to) => format!(
                "from {} to {}",
                placed_point(self.from, &self.geometry),
                placed_point(to, &self.geometry)
            ),
            None => placed_point(self.from, &self.geometry),
        }
    }
}

impl Actuation {
    #[must_use]
    pub(crate) fn of(impact: Impact) -> Self {
        Self {
            impact,
            placed: None,
        }
    }

    #[must_use]
    pub(crate) fn at(impact: Impact, placement: Placement) -> Self {
        Self {
            impact,
            placed: Some(placement),
        }
    }

    #[must_use]
    pub(crate) fn placed(impact: Impact, placement: Option<Placement>) -> Self {
        Self {
            impact,
            placed: placement,
        }
    }
}

fn cap_text(text: &str, max: usize) -> (String, bool) {
    let kept = crate::util::truncate(text, max);
    let cut = kept != text;
    (kept, cut)
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn cap_label(text: &str) -> String {
    cap_text(text, MAX_NAME_CHARS).0
}

/// The pixel size of a buffer this tool may hold, from a figure the machine reported. An empty area
/// is an honest failure (minimized, off-screen or gone) and a dimension [`Capture`] cannot express
/// is refused here, so no caller hands a platform an impossible size; the charge path runs too.
pub(crate) fn capture_size(what: &str, width: usize, height: usize) -> anyhow::Result<(u32, u32)> {
    if width == 0 || height == 0 {
        return Err(taxonomy_error(
            ERR_DEGRADED,
            format!(
                "{what} has no capture area ({width}x{height} px) — it is minimized, off-screen \
                 or was closed; restore it and re-observe"
            ),
        ));
    }
    match (u32::try_from(width), u32::try_from(height)) {
        (Ok(width), Ok(height)) => Ok((width, height)),
        _ => Err(taxonomy_error(
            ERR_DEGRADED,
            format!(
                "{what} measures {width}x{height} px, which is more than a capture can express — \
                 name a smaller surface instead"
            ),
        )),
    }
}

/// Where a tree read hit a ceiling; every `None` is a ceiling not hit. [`render_tree`] states each
/// one that was, in the header — the part of a large observation a spill preview guarantees reads.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Truncation {
    pub nodes: Option<usize>,
    pub depth: Option<Cut>,
    pub children: Option<Cut>,
    pub children_cut: Option<Cut>,
    /// This many values were shortened at the read's own character cap.
    pub values: Option<Cut>,
    /// This many names were shortened at the read's own character cap.
    pub names: Option<Cut>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Cut {
    pub count: usize,
    pub limit: usize,
}

/// The pluralisation rule of the ceiling statements: `one` for a count of one, `many` otherwise.
fn counted(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 { one } else { many }
}

impl Truncation {
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
        expect(dead_code)
    )]
    pub(crate) fn hit_node_ceiling(&mut self, limit: usize) {
        self.nodes = Some(limit);
    }

    #[cfg_attr(
        not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
        expect(dead_code)
    )]
    pub(crate) fn cut_depth(&mut self, limit: usize) {
        self.depth.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn cut_children(&mut self, limit: usize) {
        self.children.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn trim_children(&mut self, limit: usize) {
        self.children_cut
            .get_or_insert(Cut { count: 0, limit })
            .count += 1;
    }

    #[cfg_attr(
        not(any(target_os = "macos", target_os = "windows", test)),
        expect(dead_code)
    )]
    pub(crate) fn keep_value(&mut self, text: &str) -> String {
        let (kept, cut) = cap_text(text, MAX_VALUE_CHARS);
        if cut {
            self.cut_value(MAX_VALUE_CHARS);
        }
        kept
    }

    #[cfg_attr(
        not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
        expect(dead_code)
    )]
    pub(crate) fn keep_name(&mut self, text: &str) -> String {
        let (kept, cut) = cap_text(text, MAX_NAME_CHARS);
        if cut {
            self.cut_name(MAX_NAME_CHARS);
        }
        kept
    }

    /// Record one value shortened at the read's own character cap.
    fn cut_value(&mut self, limit: usize) {
        self.values.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    /// Record one name shortened at the read's own character cap.
    fn cut_name(&mut self, limit: usize) {
        self.names.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    /// One line per ceiling that was hit, each naming it. `values_cut` comes from the renderer: a
    /// read cap (`values`) and a display cap are different ceilings and both are stated.
    fn statements(&self, values_cut: usize) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(ceiling) = self.nodes {
            out.push(format!(
                "[Tree truncated: the {ceiling}-node ceiling was reached — nodes beyond it were \
                 not read]"
            ));
        }
        if let Some(cut) = self.depth {
            out.push(format!(
                "[Tree truncated: {} {} at the {}-deep ceiling — {} children were not read]",
                cut.count,
                counted(cut.count, "node", "nodes"),
                cut.limit,
                counted(cut.count, "its", "their")
            ));
        }
        if let Some(cut) = self.children {
            out.push(format!(
                "[Tree truncated: {} {} {} more than {} children — {} children were not read]",
                cut.count,
                counted(cut.count, "node", "nodes"),
                counted(cut.count, "claims", "claim"),
                cut.limit,
                counted(cut.count, "its", "their")
            ));
        }
        if let Some(cut) = self.children_cut {
            out.push(format!(
                "[Tree truncated: {} {} listed more than {} children — only the first {} were read]",
                cut.count,
                counted(cut.count, "node", "nodes"),
                cut.limit,
                cut.limit
            ));
        }
        if let Some(cut) = self.values {
            out.push(format!(
                "[Tree truncated: {} {} cut at {} chars]",
                cut.count,
                counted(cut.count, "value", "values"),
                cut.limit
            ));
        }
        if values_cut > 0 {
            out.push(format!(
                "[Tree truncated: {values_cut} {} cut at {RENDER_VALUE_CHARS} chars]",
                counted(values_cut, "value", "values")
            ));
        }
        if let Some(cut) = self.names {
            out.push(format!(
                "[Tree truncated: {} {} cut at {} chars]",
                cut.count,
                counted(cut.count, "name", "names"),
                cut.limit
            ));
        }
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Observation {
    pub app_name: String,
    pub window_title: Option<String>,
    pub surface: SurfaceGeometry,
    pub root: UiNode,
    pub truncation: Truncation,
}

#[derive(Debug, Default)]
pub(crate) struct WindowListing {
    pub windows: Vec<WindowInfo>,
    pub unread: Vec<UnreadWindows>,
    pub frontmost: Option<String>,
    pub order: WindowOrder,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowOrder {
    /// This platform cannot attribute an order to the windows listed.
    #[default]
    Unknown,
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "windows", test)),
        expect(dead_code)
    )]
    All,
    #[cfg_attr(not(any(target_os = "macos", test)), expect(dead_code))]
    Partial,
    #[cfg_attr(not(any(target_os = "macos", test)), expect(dead_code))]
    Unread,
}

/// Part of a listing that is not listed, and why: per application, with the count of windows a
/// failed read dropped. The renderer words it without claiming anything about the listed windows.
#[derive(Debug)]
pub(crate) struct UnreadWindows {
    pub app_name: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AppInfo {
    pub pid: Option<u32>,
    pub name: String,
}

#[derive(Debug, Default)]
pub(crate) struct AppListing {
    pub apps: Vec<AppInfo>,
    /// The application in front — also when it is not among `apps` (a menu-bar accessory, a process
    /// with no enumerable window). `None` is stated as "could not be named", never as a guess.
    pub frontmost: Option<String>,
}

/// Backend-defined identity of one window surface, opaque to the core: each backend formats it so
/// resolving it lands on that very window or fails — never on another of the same application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowKey(String);

impl WindowKey {
    pub(crate) fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowInfo {
    pub app_pid: Option<u32>,
    pub app_name: String,
    pub title: String,
    pub key: WindowKey,
    pub surface: SurfaceGeometry,
    /// This is the application's focused window. Only the frontmost application's is marked in an
    /// all-applications listing: a background application's cannot receive input.
    pub focused: bool,
    pub rank: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// The pixels this buffer holds of its call's [`CaptureBudget`], given back on drop.
    claim: Claim,
}

impl Capture {
    /// A blank `width`×`height` RGBA buffer: the size is charged before the buffer exists and the
    /// memory is reserved FALLIBLY, so a buffer this tool chose to make fails the call instead of
    /// aborting. It comes back zeroed, because a platform that draws into it needs an initialised
    /// `&mut [u8]`; a path that fills every byte itself ([`crop_rgba`]) takes pixels over with
    /// [`Capture::adopt`] instead.
    pub(crate) fn alloc(
        budget: &CaptureBudget,
        what: &str,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let claim = budget.charge(what, width, height)?;
        let bytes = rgba_bytes(width, height);
        let mut rgba = reserved_bytes(bytes, what, (width, height))?;
        rgba.resize(bytes, 0);
        Ok(Self {
            width,
            height,
            rgba,
            claim,
        })
    }

    /// Pixels a platform already produced, copied into a buffer charged to `budget`: the fallible
    /// reserve of [`Capture::alloc`] with the copy standing in for its zero-fill.
    #[cfg_attr(not(any(target_os = "macos", test)), expect(dead_code))]
    pub(crate) fn from_pixels(
        budget: &CaptureBudget,
        what: &str,
        width: u32,
        height: u32,
        src: &[u8],
    ) -> anyhow::Result<Self> {
        debug_assert_eq!(src.len(), rgba_bytes(width, height));
        let claim = budget.charge(what, width, height)?;
        let mut rgba = reserved_bytes(src.len(), what, (width, height))?;
        rgba.extend_from_slice(src);
        Ok(Self::adopt(claim, width, height, rgba))
    }

    pub(crate) fn adopt(claim: Claim, width: u32, height: u32, rgba: Vec<u8>) -> Self {
        debug_assert_eq!(rgba.len(), rgba_bytes(width, height));
        Self {
            width,
            height,
            rgba,
            claim,
        }
    }

    pub(crate) fn into_parts(self) -> (u32, u32, Vec<u8>, Claim) {
        (self.width, self.height, self.rgba, self.claim)
    }
}

fn reserved_bytes(
    bytes: usize,
    what: &str,
    (width, height): (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(bytes).map_err(|e| {
        taxonomy_error(
            ERR_DEGRADED,
            format!(
                "{what} could not be allocated ({bytes} bytes, {width}x{height}: {e}) — the \
                 machine is out of memory; retry, or capture a smaller surface"
            ),
        )
    })?;
    Ok(buffer)
}

// reason: a claimed buffer is at most the ceiling, 512 MiB, which every usize holds.
#[expect(clippy::cast_possible_truncation)]
fn rgba_bytes(width: u32, height: u32) -> usize {
    (u64::from(width) * u64::from(height) * 4) as usize
}

/// One capture call's allowance of pixels, shared by every buffer the call holds at once — the async
/// caller and any platform thread it hands the capture to draw on the same allowance. A [`Claim`] is
/// weighed before its buffer exists and refused without spending anything, and gives its pixels back
/// when the buffer is dropped, so the counter is what the call holds rather than what it has ever
/// held. The call is serialized (one action at a time, and a platform route is awaited), so claiming
/// and releasing need no stronger guarantee.
#[derive(Debug, Clone, Default)]
pub(crate) struct CaptureBudget(Arc<AtomicU64>);

impl CaptureBudget {
    /// Claim `width`×`height` pixels of RGBA for `what`, or refuse naming the geometry, what the
    /// call already holds and [`CAPTURE_BUDGET_PIXELS`] — checked before the buffer exists.
    pub(crate) fn charge(&self, what: &str, width: u32, height: u32) -> anyhow::Result<Claim> {
        let (width, height) = capture_size(what, width as usize, height as usize)?;
        // Two `u32` dimensions multiply into a `u64` without overflow.
        let pixels = u64::from(width) * u64::from(height);
        let spent = self.0.fetch_add(pixels, Ordering::Relaxed) + pixels;
        if spent > CAPTURE_BUDGET_PIXELS {
            self.0.fetch_sub(pixels, Ordering::Relaxed);
            return Err(ceiling_refused(what, width, height, spent - pixels));
        }
        Ok(Claim {
            budget: Arc::clone(&self.0),
            pixels,
        })
    }

    #[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))]
    pub(crate) fn remaining(&self) -> u64 {
        CAPTURE_BUDGET_PIXELS.saturating_sub(self.0.load(Ordering::Relaxed))
    }

    #[cfg(test)]
    pub(crate) fn held(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub(crate) struct Claim {
    budget: Arc<AtomicU64>,
    pixels: u64,
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.budget.fetch_sub(self.pixels, Ordering::Relaxed);
    }
}

fn mib(pixels: u64) -> u64 {
    pixels.saturating_mul(4) / (1024 * 1024)
}

/// The surface a call addresses. `Focused` is where global input goes and what a call with nothing
/// observed yet falls back to; `App` is an `a`-id — the application's own focused window,
/// re-resolved per call by design; `Window` is one exact window, identified by the backend's own
/// [`WindowKey`] so it can never land on another window.
#[derive(Debug, Clone)]
pub(crate) enum TargetSpec {
    Focused,
    App {
        app_pid: Option<u32>,
        app_name: String,
    },
    Window {
        app_pid: Option<u32>,
        app_name: String,
        key: WindowKey,
        /// Title when enumerated — model-facing text, never identity: a rename keeps its key
        label: String,
    },
    Screen,
}

impl TargetSpec {
    #[must_use]
    pub(crate) fn from_window(win: &WindowInfo) -> Self {
        TargetSpec::Window {
            app_pid: win.app_pid,
            app_name: win.app_name.clone(),
            key: win.key.clone(),
            label: win.title.clone(),
        }
    }

    #[must_use]
    pub(crate) fn app(&self) -> Option<(Option<u32>, &str)> {
        match self {
            TargetSpec::App {
                app_pid, app_name, ..
            }
            | TargetSpec::Window {
                app_pid, app_name, ..
            } => Some((*app_pid, app_name)),
            TargetSpec::Focused | TargetSpec::Screen => None,
        }
    }

    #[must_use]
    pub(crate) fn describe(&self) -> String {
        match self {
            TargetSpec::Focused => "the focused window of the frontmost app".to_string(),
            TargetSpec::App { app_name, .. } => {
                format!("the focused window of {app_name}")
            }
            TargetSpec::Window {
                app_name, label, ..
            } => window_description(app_name, label),
            TargetSpec::Screen => "the screen".to_string(),
        }
    }
}

#[must_use]
pub(crate) fn same_application(a: &TargetSpec, b: &TargetSpec) -> bool {
    match (a.app(), b.app()) {
        (Some((a_pid, a_name)), Some((b_pid, b_name))) => {
            (a_pid == b_pid && a_pid.is_some()) || a_name.eq_ignore_ascii_case(b_name)
        }
        _ => false,
    }
}

#[must_use]
pub(crate) fn window_description(app_name: &str, title: &str) -> String {
    match (app_name.is_empty(), title.is_empty()) {
        (false, false) => format!("window \"{title}\" of {app_name}"),
        (false, true) => format!("a window of {app_name}"),
        (true, false) => format!("window \"{title}\""),
        (true, true) => "the window".to_string(),
    }
}

/// The definition of surface identity: two specs are equal when they name the same surface, and the
/// registry hands the same id to equal specs. An `App` is its application, a `Window` its backend
/// key alone, which already carries the application and is what every backend resolves through. The
/// pid, app name and label are snapshots for messages, so a renamed window keeps its identity.
impl PartialEq for TargetSpec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TargetSpec::Focused, TargetSpec::Focused)
            | (TargetSpec::Screen, TargetSpec::Screen) => true,
            (
                TargetSpec::App {
                    app_pid: a,
                    app_name: a_name,
                },
                TargetSpec::App {
                    app_pid: b,
                    app_name: b_name,
                },
            ) => a == b && a_name == b_name,
            (TargetSpec::Window { key: a_key, .. }, TargetSpec::Window { key: b_key, .. }) => {
                a_key == b_key
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ElementAct {
    Press,
    SetValue { text: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Modifier {
    Cmd,
    Ctrl,
    Alt,
    Shift,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ScrollDirection {
    Up,
    #[default]
    Down,
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RawInput {
    Click {
        point: (f64, f64),
        button: MouseButton,
        double: bool,
        modifiers: Vec<Modifier>,
    },
    TypeText {
        text: String,
    },
    KeyChord {
        chord: String,
    },
    Scroll {
        point: Option<(f64, f64)>,
        direction: ScrollDirection,
        amount: u32,
    },
    Drag {
        from: (f64, f64),
        to: (f64, f64),
    },
}

/// Whether `input` on `target` has to reach a named window: typed text, a drag gesture and a wheel
/// aimed at a point, and only when the call names a window (a `w` id or an `a` id) rather than the
/// focused window or the screen.
///
/// Everything else is global input — a key chord and a wheel with no point go wherever the focus is
/// — and no result may report it as having acted on the observed surface. A click at a point is not
/// listed here: it is sent to the point, and what lies under that point is checked separately when
/// it is sent ([`UnderPoint`]). This is the one definition of that rule, so the three backends
/// cannot drift apart.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn pinned_input(target: &TargetSpec, input: &RawInput) -> bool {
    matches!(target, TargetSpec::Window { .. } | TargetSpec::App { .. })
        && matches!(
            input,
            RawInput::TypeText { .. }
                | RawInput::Drag { .. }
                | RawInput::Scroll { point: Some(_), .. }
        )
}

/// Steps a raise is given to settle before the window is declared unable to come
/// forward: a window manager completes a raise asynchronously.
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), expect(dead_code))]
pub(crate) const RAISE_SETTLE_STEPS: usize = 4;

#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), expect(dead_code))]
pub(crate) const RAISE_SETTLE_MS: u64 = 25;

/// Parse a key chord string ("cmd+shift+t", "return", "ctrl+c") into its modifiers and a single key
/// name, lowercased: `cmd`/`command`, `ctrl`/`control`, `alt`/`option`, `shift` (case-insensitive);
/// the non-modifier segment is the key name (e.g. `"return"`, `"a"`, `"f5"`, `"5"`). Pure — the
/// backend maps the name to a platform virtual keycode.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn parse_key_chord(chord: &str) -> anyhow::Result<(Vec<Modifier>, String)> {
    let parts: Vec<&str> = chord.split('+').map(str::trim).collect();
    if parts.is_empty() || parts.iter().any(|p| p.is_empty()) {
        anyhow::bail!(
            "invalid key chord '{chord}' — expected e.g. 'cmd+shift+t', 'return', 'ctrl+c'"
        );
    }
    let mut mods = Vec::new();
    let mut key: Option<String> = None;
    for part in parts {
        match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" => mods.push(Modifier::Cmd),
            "ctrl" | "control" => mods.push(Modifier::Ctrl),
            "alt" | "option" => mods.push(Modifier::Alt),
            "shift" => mods.push(Modifier::Shift),
            other => {
                if key.is_some() {
                    anyhow::bail!(
                        "invalid key chord '{chord}' — too many key names; expected exactly one"
                    );
                }
                key = Some(other.to_string());
            }
        }
    }
    let key = key.ok_or_else(|| {
        anyhow::anyhow!("invalid key chord '{chord}' — no key name given (only modifiers)")
    })?;
    Ok((mods, key))
}

// The tool checks every argument of an action here before taking the action lock, and each backend
// rechecks its input at entry, so a bad call costs a round trip rather than a lock: an out-of-range
// argument is refused with the range, text past its cap with the count and the cap (never with the
// rejected value, which would repeat the oversized payload back).

/// Refuse a string argument longer than [`MAX_ARG_CHARS`], naming its length and the cap.
pub(crate) fn check_arg_len(field: &str, value: &str) -> anyhow::Result<()> {
    let len = value.chars().count();
    if len > MAX_ARG_CHARS {
        anyhow::bail!("{field} is too long: {len} chars (max {MAX_ARG_CHARS})");
    }
    Ok(())
}

pub(crate) fn check_scroll_amount(amount: u32) -> anyhow::Result<()> {
    if !(1..=MAX_SCROLL_AMOUNT).contains(&amount) {
        anyhow::bail!("scroll amount must be 1-{MAX_SCROLL_AMOUNT} ticks — got {amount}");
    }
    Ok(())
}

pub(crate) fn check_point(field: &str, nx: f64, ny: f64) -> anyhow::Result<()> {
    if !nx.is_finite()
        || !ny.is_finite()
        || !(0.0..=NORMALIZED_MAX).contains(&nx)
        || !(0.0..=NORMALIZED_MAX).contains(&ny)
    {
        anyhow::bail!(
            "{field} ({nx}, {ny}) is out of range — coordinates are normalized 0-{NORMALIZED_MAX}"
        );
    }
    Ok(())
}

/// Refuse a region whose corners are outside 0-1000 or that encloses no area. Reversed corners name
/// the same region (ordered where it is mapped), so only a zero-extent one is refused — it would
/// otherwise become a single pixel standing for an area the call never named.
#[expect(clippy::float_cmp)] // an exact repeat of a corner encloses nothing
pub(crate) fn check_region(region: &Region) -> anyhow::Result<()> {
    for (field, v) in [
        ("x0", region.x0),
        ("y0", region.y0),
        ("x1", region.x1),
        ("y1", region.y1),
    ] {
        if !v.is_finite() || !(0.0..=NORMALIZED_MAX).contains(&v) {
            anyhow::bail!(
                "region corner {field} is {v} — corners are normalized 0-{NORMALIZED_MAX}"
            );
        }
    }
    if region.x0 == region.x1 || region.y0 == region.y1 {
        anyhow::bail!(
            "region [{}, {}, {}, {}] encloses no area — x0 and x1, and y0 and y1, must differ",
            region.x0,
            region.y0,
            region.x1,
            region.y1
        );
    }
    Ok(())
}

/// The bounds a backend's input entry checks before it starts any work — every point, the scroll
/// magnitude and the typed text. A chord's length is bounded upstream; its syntax is the backend's.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn check_raw_input(input: &RawInput) -> anyhow::Result<()> {
    match input {
        RawInput::Click { point, .. } => check_point("click point", point.0, point.1),
        RawInput::Scroll { point, amount, .. } => {
            check_scroll_amount(*amount)?;
            point.map_or(Ok(()), |(x, y)| check_point("scroll anchor", x, y))
        }
        RawInput::Drag { from, to } => {
            check_point("drag start", from.0, from.1)?;
            check_point("drag end", to.0, to.1)
        }
        RawInput::TypeText { text } => check_arg_len("text", text),
        RawInput::KeyChord { .. } => Ok(()),
    }
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn normalized_to_surface(
    nx: f64,
    ny: f64,
    geo: &SurfaceGeometry,
) -> anyhow::Result<(f64, f64)> {
    check_point("coordinates", nx, ny)?;
    Ok((
        geo.x + nx / NORMALIZED_MAX * geo.width,
        geo.y + ny / NORMALIZED_MAX * geo.height,
    ))
}

#[must_use]
pub(crate) fn surface_local(global: (f64, f64), geo: &SurfaceGeometry) -> (f64, f64) {
    (global.0 - geo.x, global.1 - geo.y)
}

#[must_use]
pub(crate) fn placed_point(local: (f64, f64), geo: &SurfaceGeometry) -> String {
    format!(
        "({:.1}, {:.1}) of {:.0}x{:.0}",
        local.0, local.1, geo.width, geo.height
    )
}

pub(crate) fn surface_to_normalized(
    ax: f64,
    ay: f64,
    geo: &SurfaceGeometry,
) -> anyhow::Result<(f64, f64)> {
    if !ax.is_finite() || !ay.is_finite() || geo.width <= 0.0 || geo.height <= 0.0 {
        anyhow::bail!("invalid cursor position or surface geometry");
    }
    let nx = ((ax - geo.x) / geo.width * NORMALIZED_MAX).clamp(0.0, NORMALIZED_MAX);
    let ny = ((ay - geo.y) / geo.height * NORMALIZED_MAX).clamp(0.0, NORMALIZED_MAX);
    Ok((nx, ny))
}

#[must_use]
pub(crate) fn contains_point(geo: &SurfaceGeometry, ax: f64, ay: f64) -> bool {
    ax >= geo.x && ax < geo.x + geo.width && ay >= geo.y && ay < geo.y + geo.height
}

/// Map a normalized region to an integer pixel rect `(x, y, w, h)` of the captured image (corners
/// ordered, clamped). The image's own size is the space, so a crop cannot target another surface.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn region_to_pixels(
    region: &Region,
    px_w: u32,
    px_h: u32,
) -> anyhow::Result<(u32, u32, u32, u32)> {
    check_region(region)?;
    let (x0, x1) = (region.x0.min(region.x1), region.x0.max(region.x1));
    let (y0, y1) = (region.y0.min(region.y1), region.y0.max(region.y1));
    let px = |n: f64, total: u32| -> u32 {
        ((n / NORMALIZED_MAX * f64::from(total)).round() as u32).min(total)
    };
    let left = px(x0, px_w);
    let right = px(x1, px_w);
    let top = px(y0, px_h);
    let bottom = px(y1, px_h);
    let x = left.min(px_w.saturating_sub(1));
    let y = top.min(px_h.saturating_sub(1));
    let w = right
        .saturating_sub(left)
        .max(1)
        .min(px_w.saturating_sub(x));
    let h = bottom
        .saturating_sub(top)
        .max(1)
        .min(px_h.saturating_sub(y));
    Ok((x, y, w, h))
}

/// Crop a rect `(x, y, w, h)` out of a capture, clamping to the image bounds. `what` names the crop
/// in a refusal, and the copy is charged to the call — a zoom crop is as large as its surface. The
/// rows are copied straight into the claimed buffer, so it costs one copy and no zero-fill.
pub(crate) fn crop_rgba(
    cap: &Capture,
    rect: (u32, u32, u32, u32),
    budget: &CaptureBudget,
    what: &str,
) -> anyhow::Result<Capture> {
    let (x, y, w, h) = rect;
    if cap.width == 0 || cap.height == 0 {
        anyhow::bail!("crop_rgba: source capture is empty");
    }
    let x = x.min(cap.width.saturating_sub(1));
    let y = y.min(cap.height.saturating_sub(1));
    let w = w.min(cap.width - x).max(1);
    let h = h.min(cap.height - y).max(1);

    let claim = budget.charge(what, w, h)?;
    let stride = cap.width as usize * 4;
    let row_bytes = w as usize * 4;
    let mut rgba = reserved_bytes(row_bytes * h as usize, what, (w, h))?;
    for index in 0..h as usize {
        let start = (y as usize + index) * stride + x as usize * 4;
        rgba.extend_from_slice(&cap.rgba[start..start + row_bytes]);
    }
    Ok(Capture::adopt(claim, w, h, rgba))
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn blit_rgba(
    dest: &mut [u8],
    dest_w: usize,
    dest_h: usize,
    src: &Capture,
    dx: usize,
    dy: usize,
) {
    let (src_w, src_h) = (src.width as usize, src.height as usize);
    if dx >= dest_w || dy >= dest_h {
        return;
    }
    let row_copy = (dest_w - dx).min(src_w);
    let rows = (dest_h - dy).min(src_h);
    for r in 0..rows {
        let src_start = r * src_w * 4;
        let dst_start = (dy + r) * dest_w * 4 + dx * 4;
        dest[dst_start..dst_start + row_copy * 4]
            .copy_from_slice(&src.rgba[src_start..src_start + row_copy * 4]);
    }
}

/// Scale between an accessibility coordinate space and captured pixels: `a11y_screen` is the whole
/// screen as the accessibility stack reports it, `pixels` a full-screen capture's size. `None` when
/// the two cannot be reconciled — non-positive extents, axes that disagree, or a space larger than
/// the capture — so a caller can refuse instead of cropping the wrong part of the screen. Only
/// extents are compared: an origin offset between the two spaces is not detectable without a second
/// anchor, and both ordinary arrangements share the screen's own origin.
#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))]
pub(crate) fn a11y_scale(a11y_screen: &SurfaceGeometry, pixels: (u32, u32)) -> Option<f64> {
    let (px, py) = (f64::from(pixels.0), f64::from(pixels.1));
    if a11y_screen.width <= 0.0 || a11y_screen.height <= 0.0 || px <= 0.0 || py <= 0.0 {
        return None;
    }
    let (sx, sy) = (px / a11y_screen.width, py / a11y_screen.height);
    if !sx.is_finite() || !sy.is_finite() || sx < 0.99 {
        return None;
    }
    if (sx - sy).abs() > 0.02 * sx {
        return None;
    }
    Some(f64::midpoint(sx, sy))
}

/// Map a geometry from an accessibility coordinate space into captured pixels.
#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))]
pub(crate) fn scale_geometry(geo: SurfaceGeometry, scale: f64) -> SurfaceGeometry {
    SurfaceGeometry {
        x: geo.x * scale,
        y: geo.y * scale,
        width: geo.width * scale,
        height: geo.height * scale,
    }
}

/// ZPixmap tile (`depth` 24/32, 4 bytes/pixel) → one RGBA capture, charged like every other buffer.
#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))] // only the Linux backend calls this; tests exercise it everywhere
pub(crate) fn x11_pixels_to_rgba(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u8,
    lsb_first: bool,
    budget: &CaptureBudget,
) -> Result<Capture, anyhow::Error> {
    if depth != 24 && depth != 32 {
        return Err(taxonomy_error(
            ERR_DEGRADED,
            format!("unsupported X11 depth {depth} (need 24/32)"),
        ));
    }
    let mut cap = Capture::alloc(budget, "a display tile", width, height)?;
    if data.len() < cap.rgba.len() {
        anyhow::bail!("short X11 image body: {} < {}", data.len(), cap.rgba.len());
    }
    for (out, px) in cap
        .rgba
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(data.as_chunks::<4>().0)
    {
        *out = if lsb_first {
            [px[2], px[1], px[0], 0xff]
        } else {
            [px[1], px[2], px[3], 0xff]
        };
    }
    Ok(cap)
}

#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))] // only the Linux backend calls this; tests exercise it everywhere
pub(crate) fn portal_uri_to_path(uri: &str) -> Result<PathBuf, anyhow::Error> {
    let stripped = uri.strip_prefix("file://").ok_or_else(|| {
        taxonomy_error(ERR_DEGRADED, format!("portal uri is not a file uri: {uri}"))
    })?;
    let path = if stripped.starts_with('/') {
        stripped
    } else {
        match stripped.find('/') {
            Some(idx) => {
                let (host, path) = stripped.split_at(idx);
                if host != "localhost" {
                    return Err(taxonomy_error(
                        ERR_DEGRADED,
                        "portal uri has unsupported host",
                    ));
                }
                path
            }
            None => {
                return Err(taxonomy_error(
                    ERR_DEGRADED,
                    "portal uri has unsupported host",
                ));
            }
        }
    };
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let bad_escape = || {
        taxonomy_error(
            ERR_DEGRADED,
            format!("bad percent-escape in portal uri: {uri}"),
        )
    };
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3).ok_or_else(&bad_escape)?;
            let hex = std::str::from_utf8(hex).map_err(|_| bad_escape())?;
            let byte = u8::from_str_radix(hex, 16).map_err(|_| bad_escape())?;
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    let s = String::from_utf8(out).map_err(|_| {
        taxonomy_error(
            ERR_DEGRADED,
            format!("portal uri is not valid UTF-8: {uri}"),
        )
    })?;
    Ok(PathBuf::from(s))
}

/// Window extents → pixel rect inside a screen-channel capture: true rect intersection in buffer
/// space (buffer origin = screen origin). `crop_rgba` clamps the far edge, so w/h are floored at 1.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)] // the Linux and Windows backends call this; tests exercise it everywhere
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn window_capture_rect(
    win: &SurfaceGeometry,
    screen: &SurfaceGeometry,
) -> Result<(u32, u32, u32, u32), anyhow::Error> {
    if win.width <= 0.0 || win.height <= 0.0 {
        return Err(taxonomy_error(
            ERR_DEGRADED,
            "the window has no visible area — it is minimized or off-screen; restore it and retry",
        ));
    }
    let x0 = (win.x - screen.x).max(0.0);
    let y0 = (win.y - screen.y).max(0.0);
    let x1 = (win.x + win.width - screen.x).min(screen.width);
    let y1 = (win.y + win.height - screen.y).min(screen.height);
    if x1 <= x0 || y1 <= y0 {
        return Err(taxonomy_error(
            ERR_DEGRADED,
            "the window lies entirely outside the captured screen — it is on another display or \
             minimized; screenshot with target: \"screen\" shows the desktop instead",
        ));
    }
    Ok((
        x0.round() as u32,
        y0.round() as u32,
        (x1 - x0).round().max(1.0) as u32,
        (y1 - y0).round().max(1.0) as u32,
    ))
}

// Three lifetimes. Element references and the observation they were minted from belong to the turn
// and are cleared when it ends. Application and window ids, the surface last observed, and
// each surface's remembered look belong to the session: only [`end_session`] ends them early, and a
// call outside an agent run has no session, so nothing session-scoped is stored for it.

struct ObservationCache {
    target: TargetSpec,
    refs: HashMap<String, Locator>,
}

static LAST_OBSERVATION: LazyLock<Mutex<HashMap<String, ObservationCache>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) const MAX_LOOKS: usize = 4;

/// What one full observation showed of one surface: the baseline every statement about that surface
/// having moved, resized or changed is measured against. One record per surface, newest first; only
/// a full observation writes one ([`remember_look`]) — a still, a capture, a narrow look and a
/// change answer replace nothing.
#[derive(Debug)]
pub(crate) struct Look {
    pub target: TargetSpec,
    pub obs: Observation,
    pub counts: Counts,
    pub taken_at: Instant,
}

/// The remembered look of each session's surfaces (see [`Look`]). A record is shared, not
/// cloned per reader, so a view built from it can be rendered after the lock is released.
static LOOKS: LazyLock<Mutex<HashMap<String, Vec<Arc<Look>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The surface each session last observed or captured (see [`last_surface`]).
static LAST_SURFACE: LazyLock<Mutex<HashMap<String, TargetSpec>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[must_use]
pub(crate) fn agent_key() -> String {
    crate::agent::current_agent_id()
        .unwrap_or_else(|| crate::agent::role::DIAGNOSTICS_ROLE.to_string())
}

pub(crate) fn store_observation(target: TargetSpec, obs: &Observation) {
    let mut refs = HashMap::new();
    let mut counter = 0usize;
    let mut path = Vec::new();
    assign_refs(&obs.root, &mut path, &mut refs, &mut counter);
    let cache = ObservationCache { target, refs };
    let key = agent_key();
    let mut guard = LAST_OBSERVATION.lock().unwrap_poison();
    guard.insert(key, cache);
}

fn assign_refs(
    node: &UiNode,
    path: &mut Vec<usize>,
    refs: &mut HashMap<String, Locator>,
    counter: &mut usize,
) {
    *counter += 1;
    refs.insert(
        format!("e{counter}"),
        Locator {
            path: path.clone(),
            role: node.role.clone(),
            name: node.name.clone(),
            frame: node.frame,
        },
    );
    for (idx, child) in node.children.iter().enumerate() {
        path.push(idx);
        assign_refs(child, path, refs, counter);
        path.pop();
    }
}

pub(crate) fn resolve_ref(reference: &str) -> anyhow::Result<(Locator, TargetSpec)> {
    let key = agent_key();
    let guard = LAST_OBSERVATION.lock().unwrap_poison();
    let cache = guard.get(&key).ok_or_else(|| {
        taxonomy_error(
            ERR_STALE_ELEMENT,
            format!("ref {reference} is not resolvable — no observation cached; run observe first"),
        )
    })?;
    let locator = cache.refs.get(reference).ok_or_else(|| {
        taxonomy_error(
            ERR_STALE_ELEMENT,
            format!("ref {reference} is not resolvable — the observed tree changed or expired; run observe again"),
        )
    })?;
    Ok((locator.clone(), cache.target.clone()))
}

/// Remember `obs` as the look of `target`'s surface: the record every later statement about that
/// surface having moved, resized or changed is measured against.
///
/// Only a full observation that succeeded calls this, and it replaces the previous record of the
/// same surface — a surface's record is keyed by the window it was taken from, so an application
/// identifier that now names another window has nothing to compare against rather than a fabricated
/// move. When the session holds more than [`MAX_LOOKS`] surfaces, the least recently looked at is
/// dropped. The observation is moved in, so the caller renders it before remembering it.
pub(crate) fn remember_look(target: &TargetSpec, obs: Observation, counts: Counts) {
    let look = Look {
        target: target.clone(),
        obs,
        counts,
        taken_at: Instant::now(),
    };
    let mut guard = LOOKS.lock().unwrap_poison();
    let entries = guard.entry(agent_key()).or_default();
    entries.retain(|existing| existing.target != *target);
    entries.insert(0, Arc::new(look));
    entries.truncate(MAX_LOOKS);
}

/// Read one surface's remembered look. Reading never reorders: only a full observation counts as
/// looking at a surface, which is exactly the order [`remember_look`] keeps, so the surface a full
/// set drops first is the one least recently *looked at*. The record comes back shared, so a caller
/// that builds a view of it (a narrow look, a change answer) holds no lock while doing so.
pub(crate) fn with_look(target: &TargetSpec) -> Option<Arc<Look>> {
    let guard = LOOKS.lock().unwrap_poison();
    let entries = guard.get(&agent_key())?;
    entries.iter().find(|look| look.target == *target).cloned()
}

pub(crate) fn remembered_geometry(target: &TargetSpec) -> Option<(SurfaceGeometry, Instant)> {
    with_look(target).map(|look| (look.obs.surface, look.taken_at))
}

/// Forget the remembered look of a surface confirmed gone — however the call addressed
/// it: by an id, by a ref, or as the surface it last observed. A window takes its own
/// record; an application that quit takes every record of its windows. A surface this
/// session never observed is ignored.
pub(crate) fn forget_look(target: &TargetSpec) {
    forget_where(|look| match target {
        TargetSpec::Window { .. } => look.target == *target,
        TargetSpec::App { .. } => same_application(&look.target, target),
        TargetSpec::Focused | TargetSpec::Screen => false,
    });
}

/// Forget the remembered look of an application a failed call named without an id — by
/// its name or its pid. That is all such a call identified it by, and it is the surface
/// confirmed gone; falling back to the session's last surface here would forget an
/// unrelated record.
pub(crate) fn forget_look_of_app_argument(arg: &str) {
    let name = arg.trim();
    let pid = name.parse::<u32>().ok();
    forget_where(|look| {
        let named_by_pid = pid.is_some_and(|pid| {
            look.target
                .app()
                .is_some_and(|(look_pid, _)| look_pid == Some(pid))
        });
        named_by_pid || look.obs.app_name.eq_ignore_ascii_case(name)
    });
}

fn forget_where(gone: impl Fn(&Look) -> bool) {
    let mut guard = LOOKS.lock().unwrap_poison();
    if let Some(entries) = guard.get_mut(&agent_key()) {
        entries.retain(|look| !gone(look));
    }
}

#[must_use]
pub(crate) fn look_age(since: Instant) -> String {
    let secs = since.elapsed().as_secs();
    match (secs / 60, secs % 60) {
        (0, s) => format!("{s}s ago"),
        (m, s) => format!("{m}m {s}s ago"),
    }
}

/// Whether the caller's live refs belong to `target` — the one condition under which a
/// narrow look may print refs. The refs a caller holds come from the latest full
/// observation and end with the turn, so a narrow look in a later turn prints none and
/// says that acting needs a fresh observation.
#[must_use]
pub(crate) fn refs_belong_to(target: &TargetSpec) -> bool {
    let guard = LAST_OBSERVATION.lock().unwrap_poison();
    guard
        .get(&agent_key())
        .is_some_and(|cache| cache.target == *target)
}

/// The surface the calling session last observed or captured. A call with no explicit target
/// addresses it, so a focus shift between observe and the action never retargets a different
/// surface; it lasts as long as the session's ids do, so a later turn still addresses the surface
/// the model last saw, and a call whose surface is gone fails instead of silently using whatever is
/// frontmost. `None` means nothing has been pinned yet.
#[must_use]
pub(crate) fn last_surface() -> Option<TargetSpec> {
    let guard = LAST_SURFACE.lock().unwrap_poison();
    guard.get(&crate::agent::current_agent_id()?).cloned()
}

/// Pin the surface the calling agent last observed or captured. Only a call that
/// succeeded pins: a failed observation leaves the previous surface in place. A call
/// with no session pins nothing, since no later call could read it.
pub(crate) fn set_last_surface(target: TargetSpec) {
    let Some(key) = crate::agent::current_agent_id() else {
        return;
    };
    LAST_SURFACE.lock().unwrap_poison().insert(key, target);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdClass {
    App,
    Window,
}

impl IdClass {
    /// The class an argument of one of this tool's id shapes names (`a12`, `w3`).
    /// `None` for anything else, so a name or a pid is never read as an id.
    fn of_id(text: &str) -> Option<Self> {
        let mut chars = text.chars();
        let class = match chars.next()? {
            'a' => Self::App,
            'w' => Self::Window,
            _ => return None,
        };
        let rest = chars.as_str();
        (!rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())).then_some(class)
    }

    /// The next id of this class. Opaque to the model: the digits carry the process
    /// and the count, and only equality with a stored entry ever matters.
    fn mint(self) -> String {
        let counter = match self {
            Self::App => &NEXT_APP,
            Self::Window => &NEXT_WINDOW,
        };
        let next = counter.fetch_add(1, Ordering::Relaxed) + 1;
        let prefix = match self {
            Self::App => 'a',
            Self::Window => 'w',
        };
        format!("{prefix}{next}")
    }
}

/// Where this process's numbering starts: the epoch time in milliseconds at the moment
/// it first minted an id. Live ids therefore stay above every id an earlier process
/// handed out, so an id a durable conversation still holds reads as unknown after a
/// restart instead of naming a surface minted after it.
static ID_BASE: LazyLock<u64> = LazyLock::new(|| {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            elapsed.as_millis().try_into().unwrap_or(u64::MAX)
        })
});

/// The id counters, seeded from [`ID_BASE`]. They only ever move forward — a listing
/// and a reset both leave them where they are — so an id is never handed to a second surface.
static NEXT_APP: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(*ID_BASE));
static NEXT_WINDOW: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(*ID_BASE));

type SessionTargets = Vec<(String, TargetSpec)>;

/// The ids each session holds, keyed by the session's agent id. Nothing is evicted
/// from it: the table holds only surfaces the caller listed or observed, an entry dies
/// with its surface ([`forget_target`]) or with the session ([`end_session`]), and no
/// size limit ever costs the caller an id it was given.
static TARGETS: LazyLock<Mutex<HashMap<String, SessionTargets>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn store_app_targets(apps: &[TargetSpec]) -> Vec<String> {
    store_targets(IdClass::App, apps)
}

pub(crate) fn store_window_targets(windows: &[TargetSpec]) -> Vec<String> {
    store_targets(IdClass::Window, windows)
}

/// Store one enumeration's surfaces and return their ids, in the same order: a surface already
/// enumerated keeps its id (re-enumeration refreshes its message-only snapshot, not its identity),
/// a new one gets a fresh id, and an id of a surface this enumeration did not include is kept.
fn store_targets(class: IdClass, surfaces: &[TargetSpec]) -> Vec<String> {
    let Some(key) = crate::agent::current_agent_id() else {
        // No session, so no later call could resolve what this one mints: the ids only print.
        return surfaces.iter().map(|_| class.mint()).collect();
    };
    let mut guard = TARGETS.lock().unwrap_poison();
    let entries = guard.entry(key).or_default();
    let mut ids = Vec::with_capacity(surfaces.len());
    for surface in surfaces {
        if let Some(position) = entries.iter().position(|(_, stored)| stored == surface) {
            let (id, stored) = &mut entries[position];
            ids.push(id.clone());
            *stored = surface.clone();
            continue;
        }
        let id = class.mint();
        entries.push((id.clone(), surface.clone()));
        ids.push(id);
    }
    ids
}

pub(crate) fn get_target(id: &str) -> Option<TargetSpec> {
    let guard = TARGETS.lock().unwrap_poison();
    let entries = guard.get(&crate::agent::current_agent_id()?)?;
    entries
        .iter()
        .find(|(stored, _)| stored == id)
        .map(|(_, surface)| surface.clone())
}

/// Drop the id a call confirmed gone, so the table follows only surfaces that still exist.
pub(crate) fn forget_target(id: &str) {
    let Some(key) = crate::agent::current_agent_id() else {
        return;
    };
    let mut guard = TARGETS.lock().unwrap_poison();
    if let Some(entries) = guard.get_mut(&key) {
        entries.retain(|(stored, _)| stored != id);
    }
}

pub(crate) fn resolve_target(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    let Some(target) = target else {
        return Ok(TargetSpec::Focused);
    };
    if target == "screen" {
        return Ok(TargetSpec::Screen);
    }
    get_target(target).ok_or_else(|| unknown_identifier(target))
}

#[must_use]
pub(crate) fn is_identifier(text: &str) -> bool {
    IdClass::of_id(text).is_some()
}

/// The refusal for an argument that is not a target this tool knows. An id of this tool's shape
/// says the tool no longer knows a surface by it, and gives the listing that mints ids of its
/// class; anything else says what a target may be. Neither claims anything about an application's
/// state.
pub(crate) fn unknown_identifier(arg: &str) -> anyhow::Error {
    let (what, next) = match IdClass::of_id(arg) {
        Some(IdClass::App) => (
            "an application",
            "list applications again with the 'apps' action",
        ),
        Some(IdClass::Window) => ("a window", "list windows again with the 'windows' action"),
        None => {
            return anyhow!(
                "unknown target '{arg}' — a target is an id printed by the 'apps' or 'windows' \
                 action, or \"screen\""
            );
        }
    };
    anyhow!(
        "unknown identifier '{arg}' — the tool no longer knows {what} by this id; {next} and use \
         the id it prints"
    )
}

/// Clear the state one turn owns, at agent-run end: element refs belong to the full observation they
/// came from — replaced by the next one, and gone with the turn that minted them — so they never
/// outlive the run. Stored ids, the remembered looks and the last observed or captured surface are
/// deliberately kept: they last as long as the session, and [`end_session`] is what ends those.
pub(crate) fn cleanup_turn(key: &str) {
    let mut guard = LAST_OBSERVATION.lock().unwrap_poison();
    guard.remove(key);
}

/// End everything a session minted: its application and window ids, the surface it last observed or
/// captured, and every remembered look. This is the single statement of what a reset does, and the
/// tool's `end_session` its single caller — reached only where the session store deletes the
/// conversation, which every reset path goes through (`/clear`, the Sessions page, the maintainer's
/// cleanup). A service restart ends the same state by ending the process.
pub(crate) fn end_session(key: &str) {
    TARGETS.lock().unwrap_poison().remove(key);
    LAST_SURFACE.lock().unwrap_poison().remove(key);
    LOOKS.lock().unwrap_poison().remove(key);
}

fn normalized_role(role: &str) -> &str {
    role.strip_prefix("AX").unwrap_or(role)
}

fn is_interactive(node: &UiNode) -> bool {
    if !node.actions.is_empty() {
        return true;
    }
    let role = normalized_role(&node.role);
    INTERACTIVE_ROLES
        .iter()
        .any(|r| r.eq_ignore_ascii_case(role))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Counts {
    pub nodes: usize,
    pub interactive: usize,
}

#[must_use]
pub(crate) fn counts(root: &UiNode) -> Counts {
    fn walk(node: &UiNode, counts: &mut Counts) {
        counts.nodes += 1;
        counts.interactive += usize::from(is_interactive(node));
        for child in &node.children {
            walk(child, counts);
        }
    }
    let mut counts = Counts {
        nodes: 0,
        interactive: 0,
    };
    walk(root, &mut counts);
    counts
}

/// How close two positions must be to count as the same place. A few points absorbs the sub-pixel
/// difference two reads of one control report, without hiding a reflow that moved it.
pub(crate) const POSITION_TOLERANCE: f64 = 2.0;

#[must_use]
pub(crate) fn same_place(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() <= POSITION_TOLERANCE && (a.1 - b.1).abs() <= POSITION_TOLERANCE
}

#[must_use]
pub(crate) fn same_size(a: &SurfaceGeometry, b: &SurfaceGeometry) -> bool {
    (a.width - b.width).abs() <= POSITION_TOLERANCE
        && (a.height - b.height).abs() <= POSITION_TOLERANCE
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
#[must_use]
pub(crate) fn same_frame(a: &SurfaceGeometry, b: &SurfaceGeometry) -> bool {
    same_place((a.x, a.y), (b.x, b.y)) && same_size(a, b)
}

/// Render the observation as an indented element tree with pre-order refs. The header carries the
/// node/interactive totals ([`counts`]) and every ceiling the read hit — a tree that was cut short
/// must never read as complete.
#[must_use]
pub(crate) fn render_tree(obs: &Observation, counts: Counts) -> String {
    let mut tree = String::new();
    let mut counter = 0usize;
    let mut values_cut = 0usize;
    render_node(&obs.root, 0, &mut tree, &mut counter, &mut values_cut);

    let mut out = surface_header(
        &obs.app_name,
        obs.window_title.as_deref(),
        &obs.surface,
        counts,
    );
    for statement in obs.truncation.statements(values_cut) {
        out.push_str(&statement);
        out.push('\n');
    }
    out.push_str(&tree);
    out
}

fn surface_header(
    app_name: &str,
    title: Option<&str>,
    surface: &SurfaceGeometry,
    counts: Counts,
) -> String {
    format!(
        "[Surface: app=\"{app_name}\" window=\"{}\" {}x{} @ ({}, {}) — {} nodes, {} interactive]\n",
        title.unwrap_or(""),
        surface.width,
        surface.height,
        surface.x,
        surface.y,
        counts.nodes,
        counts.interactive
    )
}

/// One element as every view prints it: the same fields in the same order, with a ref when the view
/// has one, and no line ending — so all views describe one element alike, and a change answer
/// compares exactly what they print. Returns whether the value had to be cut to the display cap.
fn push_node_line(out: &mut String, node: &UiNode, depth: usize, reference: Option<usize>) -> bool {
    let indent = "  ".repeat(depth);
    let _ = write!(out, "{indent}");
    if let Some(number) = reference {
        let _ = write!(out, "[e{number}] ");
    }
    let _ = write!(out, "{}", normalized_role(&node.role));
    if let Some(name) = &node.name {
        let _ = write!(out, " \"{name}\"");
    }
    if !node.actions.is_empty() {
        let _ = write!(out, " actions=[{}]", node.actions.join(","));
    }
    if let Some((x, y, _, _)) = node.frame {
        let _ = write!(out, " @ ({x}, {y})");
    }
    let mut cut = false;
    if let Some(value) = &node.value {
        let shown = crate::util::truncate(value, RENDER_VALUE_CHARS);
        cut = shown != *value;
        let _ = write!(out, " value=\"{shown}\"");
    }
    if node.focused {
        out.push_str(" focused");
    }
    cut
}

fn render_node(
    node: &UiNode,
    depth: usize,
    out: &mut String,
    counter: &mut usize,
    values_cut: &mut usize,
) {
    *counter += 1;
    if push_node_line(out, node, depth, Some(*counter)) {
        *values_cut += 1;
    }
    out.push('\n');
    for child in &node.children {
        render_node(child, depth + 1, out, counter, values_cut);
    }
}

/// How a narrow look is limited, after its arguments were resolved.
#[derive(Debug, Clone, Default)]
pub(crate) struct Narrow {
    pub at: Option<Locator>,
    pub depth: Option<usize>,
    pub kind: Option<String>,
}

impl Narrow {
    #[must_use]
    pub(crate) fn is_narrow(&self) -> bool {
        self.at.is_some() || self.depth.is_some() || self.kind.is_some()
    }
}

struct Placed<'a> {
    index: usize,
    depth: usize,
    path: Vec<usize>,
    node: &'a UiNode,
}

fn placed(root: &UiNode) -> Vec<Placed<'_>> {
    fn rec<'a>(node: &'a UiNode, depth: usize, path: &mut Vec<usize>, out: &mut Vec<Placed<'a>>) {
        out.push(Placed {
            index: out.len(),
            depth,
            path: path.clone(),
            node,
        });
        for (idx, child) in node.children.iter().enumerate() {
            path.push(idx);
            rec(child, depth + 1, path, out);
            path.pop();
        }
    }
    let mut out = Vec::new();
    rec(root, 0, &mut Vec::new(), &mut out);
    out
}

fn path_text(path: &[usize]) -> String {
    if path.is_empty() {
        "the root".to_string()
    } else {
        path.iter()
            .map(|index| (index + 1).to_string())
            .collect::<Vec<_>>()
            .join(".")
    }
}

const MAX_LISTED: usize = 12;

/// The kinds present in a tree, most frequent first, for an answer that must name them.
fn kinds_present<'a>(nodes: impl Iterator<Item = &'a UiNode>) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for node in nodes {
        let kind = normalized_role(&node.role);
        match counts
            .iter_mut()
            .find(|(name, _)| name.eq_ignore_ascii_case(kind))
        {
            Some((_, count)) => *count += 1,
            None => counts.push((kind.to_string(), 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let shown = counts.len().min(MAX_LISTED);
    let mut text = counts[..shown]
        .iter()
        .map(|(name, count)| format!("{name} ({count})"))
        .collect::<Vec<_>>()
        .join(", ");
    if counts.len() > shown {
        let _ = write!(text, " and {} more kinds", counts.len() - shown);
    }
    text
}

/// Render a narrow look over a remembered tree: the subtree beneath a ref, the levels to a depth,
/// or a flat list of one kind — always stating that it covers part of the surface. `refs_valid`
/// says whether the caller's live refs belong to this surface and turn; when not, no ref is printed
/// (it would not resolve) and the answer says acting needs a fresh observation.
pub(crate) fn render_narrow(
    look: &Look,
    narrow: &Narrow,
    refs_valid: bool,
) -> anyhow::Result<String> {
    let nodes = placed(&look.obs.root);
    let mut out = surface_header(
        &look.obs.app_name,
        look.obs.window_title.as_deref(),
        &look.obs.surface,
        look.counts,
    );
    match &narrow.kind {
        Some(kind) => render_kind_list(&mut out, &nodes, narrow, kind, refs_valid)?,
        None => render_subtree(&mut out, look.counts.nodes, &nodes, narrow, refs_valid)?,
    }
    Ok(out)
}

fn narrow_base(nodes: &[Placed<'_>], at: Option<&Locator>) -> anyhow::Result<(Vec<usize>, usize)> {
    let Some(locator) = at else {
        return Ok((Vec::new(), 0));
    };
    let found = nodes
        .iter()
        .find(|placed| placed.path == locator.path && node_matches(placed.node, locator))
        .ok_or_else(|| {
            taxonomy_error(
                ERR_STALE_ELEMENT,
                format!(
                    "the element at {} is no longer {} — run observe again for refs that match \
                     this surface",
                    path_text(&locator.path),
                    locator_describe(locator)
                ),
            )
        })?;
    Ok((found.path.clone(), found.depth))
}

fn inside_look(placed: &Placed<'_>, base: &(Vec<usize>, usize), depth: Option<usize>) -> bool {
    let (base_path, base_depth) = base;
    placed.path.starts_with(base_path)
        && placed.depth >= *base_depth
        && placed.depth - base_depth < depth.unwrap_or(usize::MAX)
}

fn render_kind_list(
    out: &mut String,
    nodes: &[Placed<'_>],
    narrow: &Narrow,
    kind: &str,
    refs_valid: bool,
) -> anyhow::Result<()> {
    let base = narrow_base(nodes, narrow.at.as_ref())?;
    let covered = nodes
        .iter()
        .filter(|placed| inside_look(placed, &base, narrow.depth))
        .collect::<Vec<&Placed<'_>>>();
    let matched = covered
        .iter()
        .filter(|placed| normalized_role(&placed.node.role).eq_ignore_ascii_case(kind))
        .copied()
        .collect::<Vec<&Placed<'_>>>();
    if matched.is_empty() {
        let _ = writeln!(
            out,
            "[Narrow: no element of kind \"{kind}\" in this part of the surface — kinds present: \
             {}. The look covers part of the surface only]",
            kinds_present(covered.iter().map(|placed| placed.node))
        );
        return Ok(());
    }
    let _ = writeln!(
        out,
        "[Narrow: elements of kind \"{kind}\" — {} of {} nodes; it covers part of the surface only]",
        matched.len(),
        covered.len()
    );
    for placed in matched {
        push_node_line(out, placed.node, 0, refs_valid.then_some(placed.index + 1));
        let _ = writeln!(out, " — at path {}", path_text(&placed.path));
    }
    push_refs_note(out, refs_valid);
    Ok(())
}

fn render_subtree(
    out: &mut String,
    total: usize,
    nodes: &[Placed<'_>],
    narrow: &Narrow,
    refs_valid: bool,
) -> anyhow::Result<()> {
    let base = narrow_base(nodes, narrow.at.as_ref())?;
    let base_depth = base.1;
    let shown = nodes
        .iter()
        .filter(|placed| inside_look(placed, &base, narrow.depth))
        .collect::<Vec<_>>();
    let _ = writeln!(
        out,
        "[Narrow: {} beneath {} — {} of {} nodes; it covers part of the surface only]",
        match narrow.depth {
            Some(depth) => format!("{depth} {}", counted(depth, "level", "levels")),
            None => "the subtree".to_string(),
        },
        match &narrow.at {
            Some(locator) => path_text(&locator.path),
            None => "the surface root".to_string(),
        },
        shown.len(),
        total
    );
    for placed in shown {
        push_node_line(
            out,
            placed.node,
            placed.depth - base_depth,
            refs_valid.then_some(placed.index + 1),
        );
        out.push('\n');
    }
    push_refs_note(out, refs_valid);
    Ok(())
}

fn push_refs_note(out: &mut String, refs_valid: bool) {
    if !refs_valid {
        out.push_str(
            "[No refs: they expire with the turn that made them — run observe for refs to act on]\n",
        );
    }
}

fn locator_describe(locator: &Locator) -> String {
    match &locator.name {
        Some(name) => format!("{} \"{name}\"", normalized_role(&locator.role)),
        None => normalized_role(&locator.role).to_string(),
    }
}

/// One element as the observation prints it — the whole comparison key of a change answer. Two
/// elements are the same when their kind and name match, paired in the order they appear, which is
/// why a rename reads as one element gone and another appeared. Everything is borrowed from the
/// tree.
struct PrintedNode<'a> {
    node: &'a UiNode,
    kind: &'a str,
    match_kind: String,
    value: Option<String>,
    value_cut: bool,
    path: Vec<usize>,
}

impl PrintedNode<'_> {
    fn describe(&self) -> String {
        match &self.node.name {
            Some(name) => format!("{} \"{name}\"", self.kind),
            None => self.kind.to_string(),
        }
    }

    fn where_text(&self) -> String {
        let path = path_text(&self.path);
        match self.node.frame {
            Some((x, y, _, _)) => format!(" at ({x}, {y}), path {path}"),
            None => format!(" at path {path}"),
        }
    }
}

fn printed_nodes(root: &UiNode) -> Vec<PrintedNode<'_>> {
    placed(root)
        .into_iter()
        .map(|placed| {
            let (value, value_cut) = match placed.node.value.as_deref() {
                Some(value) => {
                    let shown = crate::util::truncate(value, RENDER_VALUE_CHARS);
                    let cut = shown != value;
                    (Some(shown), cut)
                }
                None => (None, false),
            };
            let kind = normalized_role(&placed.node.role);
            PrintedNode {
                node: placed.node,
                kind,
                match_kind: kind.to_lowercase(),
                value,
                value_cut,
                path: placed.path,
            }
        })
        .collect()
}

struct Moved<'a> {
    node: &'a PrintedNode<'a>,
    from: (f64, f64),
    to: (f64, f64),
}

#[derive(Default)]
struct Changes<'a> {
    appeared: Vec<&'a PrintedNode<'a>>,
    disappeared: Vec<&'a PrintedNode<'a>>,
    moved: Vec<Moved<'a>>,
    revalued: Vec<(&'a PrintedNode<'a>, &'a PrintedNode<'a>)>,
    actions_changed: Vec<(&'a PrintedNode<'a>, &'a PrintedNode<'a>)>,
    refocused: Vec<&'a PrintedNode<'a>>,
}

impl Changes<'_> {
    fn is_empty(&self) -> bool {
        self.appeared.is_empty()
            && self.disappeared.is_empty()
            && self.moved.is_empty()
            && self.revalued.is_empty()
            && self.actions_changed.is_empty()
            && self.refocused.is_empty()
    }
}

/// Pair the trees by kind and name in the order they print, then classify each pair: an element
/// with no position on either side can appear or disappear but never moved. The pairing is a queue
/// per identity, linear in the size of the trees, so the answer holds no more than the two trees
/// themselves.
fn diff_trees<'a>(before: &'a [PrintedNode<'a>], after: &'a [PrintedNode<'a>]) -> Changes<'a> {
    type Identity<'a> = (&'a str, Option<&'a str>);
    let mut unpaired: HashMap<Identity<'_>, std::collections::VecDeque<(usize, &'a PrintedNode)>> =
        HashMap::new();
    for (position, node) in before.iter().enumerate() {
        unpaired
            .entry((node.match_kind.as_str(), node.node.name.as_deref()))
            .or_default()
            .push_back((position, node));
    }
    let mut changes = Changes::default();
    let mut paired: Vec<(&PrintedNode, &PrintedNode)> = Vec::new();
    for node in after {
        match unpaired
            .get_mut(&(node.match_kind.as_str(), node.node.name.as_deref()))
            .and_then(std::collections::VecDeque::pop_front)
        {
            Some((_, old)) => paired.push((old, node)),
            None => changes.appeared.push(node),
        }
    }
    let mut gone = unpaired
        .into_values()
        .flatten()
        .collect::<Vec<(usize, &PrintedNode)>>();
    gone.sort_unstable_by_key(|(position, _)| *position);
    changes.disappeared = gone.into_iter().map(|(_, node)| node).collect();
    for (old, new) in paired {
        if let (Some(a), Some(b)) = (old.node.frame, new.node.frame)
            && !same_place((a.0, a.1), (b.0, b.1))
        {
            changes.moved.push(Moved {
                node: new,
                from: (a.0, a.1),
                to: (b.0, b.1),
            });
        }
        if old.value != new.value {
            changes.revalued.push((old, new));
        }
        if old.node.actions != new.node.actions {
            changes.actions_changed.push((old, new));
        }
        if old.node.focused != new.node.focused {
            changes.refocused.push(new);
        }
    }
    changes
}

pub(crate) fn render_changes(look: &Look, now: &Observation, counts: Counts) -> String {
    let before = printed_nodes(&look.obs.root);
    let after = printed_nodes(&now.root);
    let changes = diff_trees(&before, &after);
    let mut out = surface_header(
        &now.app_name,
        now.window_title.as_deref(),
        &now.surface,
        counts,
    );
    // The comparison states every ceiling the read it compares reached, exactly as the full
    // observation would: a tree that was cut short must never read as a surface whose tail
    // disappeared.
    let values_cut = after.iter().filter(|node| node.value_cut).count();
    for statement in now.truncation.statements(values_cut) {
        out.push_str(&statement);
        out.push('\n');
    }
    let _ = writeln!(
        out,
        "[Changes since the observation of this surface taken {}: {} appeared, {} disappeared, {} \
         moved, {} changed value, {} changed actions, {} changed focus]",
        look_age(look.taken_at),
        changes.appeared.len(),
        changes.disappeared.len(),
        changes.moved.len(),
        changes.revalued.len(),
        changes.actions_changed.len(),
        changes.refocused.len()
    );
    if changes.is_empty() {
        let _ = writeln!(
            out,
            "nothing changed on this surface since the observation taken {}",
            look_age(look.taken_at)
        );
        return out;
    }
    push_changes(&mut out, &changes);
    out.push_str(
        "[No refs: a change answer compares, it does not hand out references — run observe for \
         refs to act on]\n",
    );
    out
}

fn push_changes(out: &mut String, changes: &Changes<'_>) {
    push_change_section(
        out,
        "appeared",
        changes
            .appeared
            .iter()
            .map(|node| format!("{}{}", node.describe(), node.where_text())),
    );
    push_change_section(
        out,
        "disappeared",
        changes
            .disappeared
            .iter()
            .map(|node| format!("{}{}", node.describe(), node.where_text())),
    );
    push_change_section(
        out,
        "moved",
        changes.moved.iter().map(|moved| {
            format!(
                "{} from ({}, {}) to ({}, {})",
                moved.node.describe(),
                moved.from.0,
                moved.from.1,
                moved.to.0,
                moved.to.1
            )
        }),
    );
    push_change_section(
        out,
        "changed value",
        changes.revalued.iter().map(|(old, new)| {
            let shown = |value: Option<&str>| match value {
                Some(value) => format!("\"{value}\""),
                None => "no value".to_string(),
            };
            format!(
                "{} — {} → {}",
                new.describe(),
                shown(old.value.as_deref()),
                shown(new.value.as_deref())
            )
        }),
    );
    push_change_section(
        out,
        "changed actions",
        changes.actions_changed.iter().map(|(old, new)| {
            format!(
                "{} — actions [{}] → [{}]",
                new.describe(),
                old.node.actions.join(","),
                new.node.actions.join(",")
            )
        }),
    );
    push_change_section(
        out,
        "changed focus",
        changes.refocused.iter().map(|node| {
            format!(
                "{} is {}",
                node.describe(),
                if node.node.focused {
                    "focused now"
                } else {
                    "no longer focused"
                }
            )
        }),
    );
}

fn push_change_section(out: &mut String, title: &str, lines: impl Iterator<Item = String>) {
    let lines = lines.collect::<Vec<_>>();
    if lines.is_empty() {
        return;
    }
    let _ = writeln!(out, "{title}:");
    for line in lines.iter().take(MAX_LISTED) {
        let _ = writeln!(out, "  {line}");
    }
    if lines.len() > MAX_LISTED {
        let _ = writeln!(out, "  … and {} more", lines.len() - MAX_LISTED);
    }
}

#[must_use]
pub(crate) fn geometry_change(before: &SurfaceGeometry, now: &SurfaceGeometry) -> Option<String> {
    let (dx, dy) = (now.x - before.x, now.y - before.y);
    let moved = !same_place((before.x, before.y), (now.x, now.y));
    let resized = !same_size(before, now);
    match (moved, resized) {
        (false, false) => None,
        (true, false) => Some(format!("moved by ({dx:.1}, {dy:.1})")),
        (false, true) => Some(format!(
            "resized from {}x{} to {}x{}",
            before.width, before.height, now.width, now.height
        )),
        (true, true) => Some(format!(
            "moved by ({dx:.1}, {dy:.1}) and resized from {}x{} to {}x{}",
            before.width, before.height, now.width, now.height
        )),
    }
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
#[must_use]
pub(crate) fn pre_order_index(root: &UiNode, target: &UiNode) -> usize {
    fn rec(node: &UiNode, target: &UiNode, counter: &mut usize, found: &mut Option<usize>) {
        if found.is_some() {
            return;
        }
        if std::ptr::eq(node, target) {
            *found = Some(*counter);
        }
        *counter += 1;
        for c in &node.children {
            rec(c, target, counter, found);
        }
    }
    let (mut counter, mut found) = (0, None);
    rec(root, target, &mut counter, &mut found);
    found.unwrap_or(0)
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) enum LocatorMatch<'a> {
    Path(&'a UiNode),
    Unique(&'a UiNode),
    Ambiguous,
    NotFound,
}

/// Resolve a locator against a tree. The recorded child-index path is followed first; if it reaches
/// a node matching the role+name (same matching rules as a search) it wins even when the search
/// would match several nodes. Otherwise the tree is searched for role+name matches: exactly one is
/// unique, more than one is ambiguous, and none is not-found.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
#[must_use]
pub(crate) fn resolve_locator<'a>(root: &'a UiNode, locator: &Locator) -> LocatorMatch<'a> {
    let mut node = root;
    for idx in &locator.path {
        let Some(child) = node.children.get(*idx) else {
            return search_locator(root, locator);
        };
        node = child;
    }
    if node_matches(node, locator) {
        return LocatorMatch::Path(node);
    }
    search_locator(root, locator)
}

/// Resolve a ref and confirm it is still the element the caller saw ([`confirm_locator`]) — the
/// contract every backend's `act_on_element` follows before it sends. An `Err` is a refusal;
/// nothing was sent.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn resolve_locator_confirmed<'a>(
    root: &'a UiNode,
    locator: &Locator,
) -> Result<(&'a UiNode, LocatorConfirmation), anyhow::Error> {
    match resolve_locator(root, locator) {
        LocatorMatch::Path(node) | LocatorMatch::Unique(node) => {
            Ok((node, confirm_locator(node, locator)?))
        }
        LocatorMatch::Ambiguous => Err(taxonomy_error(
            ERR_AMBIGUOUS_LOCATOR,
            format!(
                "more than one element now matches {} — nothing was sent; run observe again and \
                 act on the ref it prints now",
                locator_describe(locator)
            ),
        )),
        LocatorMatch::NotFound => {
            let now = node_at_path(root, &locator.path);
            Err(taxonomy_error(
                ERR_STALE_ELEMENT,
                match now {
                    Some(node) => format!(
                        "{} is no longer on this surface — {} sits where it was; nothing was sent, \
                         so run observe again and act on what it prints now",
                        locator_describe(locator),
                        locator_describe(&Locator {
                            path: locator.path.clone(),
                            role: node.role.clone(),
                            name: node.name.clone(),
                            frame: node.frame,
                        })
                    ),
                    None => format!(
                        "{} is no longer on this surface — nothing was sent; run observe again and \
                         act on what it prints now",
                        locator_describe(locator)
                    ),
                },
            ))
        }
    }
}

fn node_at_path<'a>(root: &'a UiNode, path: &[usize]) -> Option<&'a UiNode> {
    let mut node = root;
    for idx in path {
        node = node.children.get(*idx)?;
    }
    Some(node)
}

/// Shared refusal for observing the `screen` target: an observation reads an accessibility surface,
/// and the screen itself is read by a still — named only when stills work, since listing windows
/// always does.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn screen_observe_error() -> anyhow::Error {
    taxonomy_error(
        ERR_UNSUPPORTED,
        if super::capture_known_available() {
            "observe targets an accessibility surface — use screenshot for the screen"
        } else {
            "observe targets an accessibility surface — use apps/windows to pick one"
        },
    )
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn screen_act_error() -> anyhow::Error {
    taxonomy_error(
        ERR_UNSUPPORTED,
        if super::capture_known_available() {
            "act targets a window — use apps/windows to pick one (use screenshot for the screen)"
        } else {
            "act targets a window — use apps/windows to pick one"
        },
    )
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
fn node_matches(node: &UiNode, locator: &Locator) -> bool {
    let role_ok = normalized_role(&node.role).eq_ignore_ascii_case(normalized_role(&locator.role));
    let name_ok = match (&locator.name, &node.name) {
        (None, _) => true,
        (Some(l), Some(n)) => l == n,
        (Some(_), None) => false,
    };
    role_ok && name_ok
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LocatorConfirmation {
    SamePlace,
    PositionUnconfirmed,
}

impl LocatorConfirmation {
    #[must_use]
    pub(crate) fn note(self, locator: &Locator) -> String {
        match self {
            Self::SamePlace => format!(
                "{} is still where the observation showed it",
                locator_describe(locator)
            ),
            Self::PositionUnconfirmed => format!(
                "{} matched by kind and name, but a position could not be compared — one side of \
                 the read carries none — so its place could not be confirmed",
                locator_describe(locator)
            ),
        }
    }
}

/// Whether a freshly read node is still the element the ref was minted for: the same kind, the same
/// name and — against the surface, so a window that merely moved invalidates nothing — the same
/// place. `Err` names what changed, and nothing was sent; this keeps a press off whatever reflowed
/// there.
pub(crate) fn confirm_locator(
    node: &UiNode,
    locator: &Locator,
) -> anyhow::Result<LocatorConfirmation> {
    let (Some((x, y, _, _)), Some(frame)) = (node.frame, locator.frame) else {
        return Ok(LocatorConfirmation::PositionUnconfirmed);
    };
    if same_place((frame.0, frame.1), (x, y)) {
        return Ok(LocatorConfirmation::SamePlace);
    }
    Err(taxonomy_error(
        ERR_STALE_ELEMENT,
        format!(
            "{} is no longer where the observation showed it — it was at ({}, {}) and is now at \
             ({x}, {y}); nothing was sent, so run observe again and act on what it prints now",
            locator_describe(locator),
            frame.0,
            frame.1
        ),
    ))
}

#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
fn search_locator<'a>(root: &'a UiNode, locator: &Locator) -> LocatorMatch<'a> {
    let mut found: Option<&'a UiNode> = None;
    let mut count = 0usize;
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node_matches(node, locator) {
            count += 1;
            if count > 1 {
                return LocatorMatch::Ambiguous;
            }
            found = Some(node);
        }
        for child in node.children.iter().rev() {
            stack.push(child);
        }
    }
    match found {
        Some(node) => LocatorMatch::Unique(node),
        None => LocatorMatch::NotFound,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FocusedElement {
    pub kind: String,
    pub name: Option<String>,
}

impl FocusedElement {
    #[must_use]
    pub(crate) fn describe(&self) -> String {
        let kind = normalized_role(&self.kind);
        match &self.name {
            Some(name) if !name.is_empty() => format!("{kind} \"{name}\""),
            _ => format!("an unnamed {kind}"),
        }
    }
}

#[async_trait::async_trait]
pub(crate) trait Backend: Send + Sync {
    fn is_supported(&self) -> bool;
    /// Why this session cannot receive an interactive call — the state the owner has to change,
    /// named in the error the model reads — or `None` when it can. Answer only from a signal the
    /// platform publishes: `None` for a state it cannot tell, since a false refusal on a working
    /// session is worse.
    async fn desktop_unavailable(&self) -> Option<String>;
    /// Prompt-free capture-channel probe (the screen-recording grant on macOS); cached with a TTL.
    async fn capture_available(&self) -> bool;
    /// Actionable diagnosis for an unavailable capture channel (missing grant, or an OS too old).
    fn capture_unavailable_error(&self) -> anyhow::Error;
    async fn list_apps(&self) -> anyhow::Result<AppListing>;
    async fn list_windows(&self, app: Option<&AppInfo>) -> anyhow::Result<WindowListing>;
    /// Whether the application these [`AppInfo`] name is still running: `Some(false)` only when this
    /// platform positively established that it is not, `None` when it cannot tell. The `windows`
    /// action asks it for an `a` id before listing, so an application that went away is reported as
    /// gone rather than as one with no windows.
    async fn app_running(&self, _app: &AppInfo) -> Option<bool> {
        None
    }
    /// The concrete window a target names right now: an application id re-resolves to the window it
    /// has focused, `Focused` to the frontmost application's. Every statement about a surface is
    /// about one window, so an application address is made concrete before anything is recorded
    /// about it.
    async fn surface_window(&self, target: &TargetSpec) -> anyhow::Result<WindowInfo>;
    async fn surface_geometry(&self, target: &TargetSpec) -> anyhow::Result<SurfaceGeometry>;
    async fn observe(&self, target: &TargetSpec) -> anyhow::Result<Observation>;
    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> anyhow::Result<Impact>;
    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> anyhow::Result<Actuation>;
    async fn cursor_position(&self) -> anyhow::Result<(f64, f64)>;
    async fn focused_element(&self, _target: Option<&TargetSpec>) -> Option<FocusedElement> {
        None
    }
    async fn capture(&self, target: &TargetSpec, budget: &CaptureBudget)
    -> anyhow::Result<Capture>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(x: f64, y: f64, w: f64, h: f64) -> SurfaceGeometry {
        SurfaceGeometry {
            x,
            y,
            width: w,
            height: h,
        }
    }

    fn node(role: &str, name: Option<&str>) -> UiNode {
        UiNode {
            role: role.to_string(),
            name: name.map(String::from),
            value: None,
            frame: None,
            actions: Vec::new(),
            focused: false,
            children: Vec::new(),
        }
    }

    fn leaf(role: &str, name: &str) -> UiNode {
        let mut n = node(role, Some(name));
        n.actions = vec!["press".to_string()];
        n
    }

    fn sample_observation() -> Observation {
        let root = UiNode {
            role: "AXWindow".to_string(),
            name: Some("Sample".to_string()),
            value: None,
            frame: Some((0.0, 0.0, 800.0, 600.0)),
            actions: Vec::new(),
            focused: true,
            children: vec![
                leaf("AXButton", "OK"),
                UiNode {
                    role: "AXGroup".to_string(),
                    name: None,
                    value: None,
                    frame: None,
                    actions: Vec::new(),
                    focused: false,
                    children: vec![leaf("AXButton", "Cancel")],
                },
            ],
        };
        Observation {
            app_name: "TestApp".to_string(),
            window_title: Some("Sample".to_string()),
            surface: surface(0.0, 0.0, 800.0, 600.0),
            root,
            truncation: Truncation::default(),
        }
    }

    fn framed_observation() -> Observation {
        let mut obs = sample_observation();
        obs.root.frame = Some((0.0, 0.0, 800.0, 600.0));
        obs.root.children[0].frame = Some((10.0, 10.0, 80.0, 24.0));
        obs.root.children[1].children[0].frame = Some((10.0, 50.0, 80.0, 24.0));
        obs
    }

    /// A point in normalized coordinates and the pixel it must name, or `None` when unexpressible.
    struct PointCase {
        name: &'static str,
        geometry: SurfaceGeometry,
        point: (f64, f64),
        pixels: Option<(f64, f64)>,
    }

    #[test]
    fn normalized_point_cases() {
        let cases = [
            PointCase {
                name: "an_offset_surface",
                geometry: surface(100.0, 200.0, 800.0, 600.0),
                point: (500.0, 250.0),
                pixels: Some((500.0, 350.0)),
            },
            PointCase {
                name: "a_negative_origin_surface",
                geometry: surface(0.0, -100.0, 3840.0, 1180.0),
                point: (500.0, 500.0),
                pixels: Some((1920.0, 490.0)),
            },
            PointCase {
                name: "below_the_range",
                geometry: surface(100.0, 200.0, 800.0, 600.0),
                point: (-1.0, 0.0),
                pixels: None,
            },
            PointCase {
                name: "above_the_range",
                geometry: surface(100.0, 200.0, 800.0, 600.0),
                point: (0.0, 1001.0),
                pixels: None,
            },
            PointCase {
                name: "not_a_number",
                geometry: surface(100.0, 200.0, 800.0, 600.0),
                point: (f64::NAN, 0.0),
                pixels: None,
            },
        ];
        for case in cases {
            let result = normalized_to_surface(case.point.0, case.point.1, &case.geometry);
            let Some((x, y)) = case.pixels else {
                assert!(result.is_err(), "{}: the point must be refused", case.name);
                continue;
            };
            let (px, py) = result.unwrap_or_else(|e| panic!("{}: {e}", case.name));
            assert!((px - x).abs() < 1e-9, "{}: x = {px}", case.name);
            assert!((py - y).abs() < 1e-9, "{}: y = {py}", case.name);
        }
    }

    #[test]
    fn blit_rgba_clips_and_copies() {
        let mut dest = vec![0u8; 4 * 4 * 4];
        let mut src = Capture::alloc(&CaptureBudget::default(), "a test frame", 2, 2).unwrap();
        src.rgba
            .copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]);
        blit_rgba(&mut dest, 4, 4, &src, 1, 1);
        assert_eq!(&dest[4 * 4 + 4..4 * 4 + 8], &[1, 2, 3, 255]);
        assert_eq!(&dest[4 * 4 + 8..4 * 4 + 12], &[4, 5, 6, 255]);
    }

    /// Every backend entry refuses the bounds it cannot honour before it starts work.
    #[test]
    fn raw_input_bounds_are_checked_at_the_backend_entry() {
        let refused = [
            RawInput::Click {
                point: (NORMALIZED_MAX + 1.0, 0.0),
                button: MouseButton::Left,
                double: false,
                modifiers: Vec::new(),
            },
            RawInput::Scroll {
                point: None,
                direction: ScrollDirection::Down,
                amount: MAX_SCROLL_AMOUNT + 1,
            },
            RawInput::Scroll {
                point: None,
                direction: ScrollDirection::Down,
                amount: 0,
            },
            RawInput::Drag {
                from: (0.0, 0.0),
                to: (0.0, -1.0),
            },
            RawInput::TypeText {
                text: "t".repeat(MAX_ARG_CHARS + 1),
            },
        ];
        for input in refused {
            assert!(
                check_raw_input(&input).is_err(),
                "{input:?} must be refused"
            );
        }
        assert!(
            check_raw_input(&RawInput::Scroll {
                point: Some((1.0, NORMALIZED_MAX)),
                direction: ScrollDirection::Up,
                amount: MAX_SCROLL_AMOUNT,
            })
            .is_ok()
        );
    }

    /// Where a rect comes from: a region cut out of an image, or a window on a captured screen.
    enum RectSource {
        Region {
            region: Region,
            image: (u32, u32),
        },
        Window {
            window: SurfaceGeometry,
            screen: SurfaceGeometry,
        },
    }

    struct RectCase {
        name: &'static str,
        source: RectSource,
        /// The `(x, y, width, height)` the rect must name, or the words of the refusal.
        rect: Result<(u32, u32, u32, u32), &'static str>,
    }

    #[expect(clippy::too_many_lines)] // one rect shape per source and refusal
    #[test]
    fn rect_cases() {
        let cases = [
            RectCase {
                name: "a_region_of_an_image",
                source: RectSource::Region {
                    region: Region {
                        x0: 200.0,
                        y0: 100.0,
                        x1: 800.0,
                        y1: 500.0,
                    },
                    image: (1000, 1000),
                },
                rect: Ok((200, 100, 600, 400)),
            },
            RectCase {
                name: "reversed_corners_name_the_same_region",
                source: RectSource::Region {
                    region: Region {
                        x0: 800.0,
                        y0: 500.0,
                        x1: 200.0,
                        y1: 100.0,
                    },
                    image: (1000, 1000),
                },
                rect: Ok((200, 100, 600, 400)),
            },
            RectCase {
                name: "a_region_outside_the_normalized_range",
                source: RectSource::Region {
                    region: Region {
                        x0: 1100.0,
                        y0: 0.0,
                        x1: 200.0,
                        y1: 100.0,
                    },
                    image: (1000, 1000),
                },
                rect: Err(""),
            },
            RectCase {
                name: "a_region_enclosing_no_area",
                source: RectSource::Region {
                    region: Region {
                        x0: 100.0,
                        y0: 100.0,
                        x1: 900.0,
                        y1: 100.0,
                    },
                    image: (1000, 1000),
                },
                rect: Err(""),
            },
            RectCase {
                name: "a_region_beyond_the_image_is_clamped",
                source: RectSource::Region {
                    region: Region {
                        x0: 500.0,
                        y0: 500.0,
                        x1: 1000.0,
                        y1: 1000.0,
                    },
                    image: (200, 200),
                },
                rect: Ok((100, 100, 100, 100)),
            },
            RectCase {
                name: "a_window_on_a_negative_origin_screen",
                source: RectSource::Window {
                    window: surface(-1920.0, 100.0, 800.0, 600.0),
                    screen: surface(-1920.0, 0.0, 3840.0, 1080.0),
                },
                rect: Ok((0, 100, 800, 600)),
            },
            RectCase {
                name: "a_window_partly_off_screen",
                source: RectSource::Window {
                    window: surface(1500.0, 900.0, 800.0, 400.0),
                    screen: surface(-1920.0, 0.0, 3840.0, 1080.0),
                },
                rect: Ok((3420, 900, 420, 180)),
            },
            RectCase {
                name: "a_window_off_the_screen",
                source: RectSource::Window {
                    window: surface(5000.0, 0.0, 100.0, 100.0),
                    screen: surface(-1920.0, 0.0, 3840.0, 1080.0),
                },
                rect: Err("outside the captured screen"),
            },
            RectCase {
                name: "a_window_with_no_area",
                source: RectSource::Window {
                    window: surface(0.0, 0.0, 0.0, 0.0),
                    screen: surface(-1920.0, 0.0, 3840.0, 1080.0),
                },
                rect: Err(""),
            },
        ];

        for case in &cases {
            let result = match case.source {
                RectSource::Region { region, image } => region_to_pixels(&region, image.0, image.1),
                RectSource::Window { window, screen: on } => window_capture_rect(&window, &on),
            };
            match case.rect {
                Ok(expected) => assert_eq!(
                    result.unwrap_or_else(|e| panic!("{}: {e}", case.name)),
                    expected,
                    "{}: the rect",
                    case.name
                ),
                Err(words) => {
                    let err = format!(
                        "{:?}",
                        result
                            .err()
                            .unwrap_or_else(|| panic!("{}: the rect must be refused", case.name))
                    );
                    assert!(
                        err.contains(words),
                        "{}: the refusal must say {words:?}: {err}",
                        case.name
                    );
                }
            }
        }

        // A region object carries only its corners: an unknown key is a mistake, not a value.
        assert!(
            serde_json::from_value::<Region>(
                serde_json::json!({"x0": 0, "y0": 0, "x1": 1, "y1": 1, "x2": 2})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<Region>(
                serde_json::json!({"x0": 0, "y0": 0, "x1": 1, "y1": 1})
            )
            .is_ok()
        );
        assert!(serde_json::from_value::<Region>(serde_json::json!([0, 0, 1, 1, 2])).is_err());
        assert!(serde_json::from_value::<Region>(serde_json::json!([0, 0, 1, 1])).is_ok());
    }

    #[test]
    fn crop_rgba_extracts_sub_rect() {
        let budget = CaptureBudget::default();
        let mut cap = Capture::alloc(&budget, "a test frame", 4, 2).unwrap();
        for (i, (r, c)) in [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
        ]
        .into_iter()
        .enumerate()
        {
            cap.rgba[i * 4..(i + 1) * 4].copy_from_slice(&[r * 40, c * 10, 0, 255]);
        }
        let crop = crop_rgba(&cap, (1, 0, 2, 2), &budget, "the zoom crop").unwrap();
        assert_eq!(crop.width, 2);
        assert_eq!(crop.height, 2);
        assert_eq!(crop.rgba.len(), 2 * 2 * 4);
        assert_eq!(budget.held(), 4 * 2 + 2 * 2);
        assert_eq!(&crop.rgba[..4], &[0, 10, 0, 255]);
        // Out-of-bounds rect is clamped, never panics.
        let clamp = crop_rgba(&cap, (10, 10, 50, 50), &budget, "the zoom crop").unwrap();
        assert!(clamp.width >= 1 && clamp.height >= 1);
    }

    #[test]
    fn x11_pixels_convert_both_byte_orders() {
        let budget = CaptureBudget::default();
        let tile =
            x11_pixels_to_rgba(&[10, 20, 30, 0, 40, 50, 60, 0], 2, 1, 24, true, &budget).unwrap();
        assert_eq!(tile.rgba, vec![30, 20, 10, 255, 60, 50, 40, 255]);
        let tile = x11_pixels_to_rgba(&[0, 30, 20, 10], 1, 1, 32, false, &budget).unwrap();
        assert_eq!(tile.rgba, vec![30, 20, 10, 255]);
        let err = x11_pixels_to_rgba(&[0; 4], 1, 1, 16, true, &budget).unwrap_err();
        assert!(err.to_string().contains("16"), "{err:?}");
    }

    #[test]
    fn portal_uri_decodes_to_path() {
        assert_eq!(
            portal_uri_to_path("file:///tmp/shot%20one.png").unwrap(),
            PathBuf::from("/tmp/shot one.png")
        );
        assert_eq!(
            portal_uri_to_path("file://localhost/tmp/shot%20one.png").unwrap(),
            PathBuf::from("/tmp/shot one.png")
        );
        assert!(
            portal_uri_to_path("file:///tmp/a+b.png")
                .unwrap()
                .to_str()
                .is_some_and(|p| p.contains('+'))
        );
        assert!(portal_uri_to_path("https://x/y.png").is_err());
        assert!(portal_uri_to_path("file://otherhost/tmp/a.png").is_err());
        assert!(portal_uri_to_path("file://localhost").is_err());
        assert!(portal_uri_to_path("file:///tmp/a%2").is_err());
        assert!(portal_uri_to_path("file:///tmp/a%").is_err());
        assert!(portal_uri_to_path("file:///tmp/a%zz.png").is_err());
    }

    /// A tree and what its rendering must show: the lines, the header above them, and the counts.
    struct RenderCase {
        name: &'static str,
        observation: Observation,
        /// The prefix each of the first lines must start with.
        lines: &'static [&'static str],
        /// Substrings the whole rendering must carry.
        contains: &'static [&'static str],
        /// Statements the header (above the first element line) must carry.
        header: Vec<String>,
        /// Substrings the rendering must not carry.
        absent: Vec<String>,
        counts: Option<(usize, usize)>,
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn render_tree_cases() {
        let pre_order = RenderCase {
            name: "pre_order_refs_and_flags",
            observation: sample_observation(),
            lines: &[
                "[Surface: app=\"TestApp\" window=\"Sample\" 800x600 @ (0, 0) — 4 nodes, 2 interactive]",
                "[e1] Window \"Sample\"",
                "  [e2] Button \"OK\"",
                "  [e3] Group",
                "    [e4] Button \"Cancel\"",
            ],
            contains: &["actions=[press]"],
            header: Vec::new(),
            absent: Vec::new(),
            counts: Some((4, 2)),
        };

        let long = "x".repeat(200);
        let truncated = RenderCase {
            name: "every_ceiling_is_stated_in_the_header",
            observation: Observation {
                app_name: "A".to_string(),
                window_title: None,
                surface: surface(0.0, 0.0, 10.0, 10.0),
                root: UiNode {
                    value: Some(long.clone()),
                    ..node("AXTextField", Some("field"))
                },
                truncation: Truncation {
                    nodes: Some(MAX_TREE_NODES),
                    depth: Some(Cut {
                        count: 2,
                        limit: 25,
                    }),
                    values: Some(Cut {
                        count: 1,
                        limit: MAX_VALUE_CHARS,
                    }),
                    names: Some(Cut {
                        count: 3,
                        limit: MAX_NAME_CHARS,
                    }),
                    children: Some(Cut {
                        count: 4,
                        limit: MAX_CHILDREN,
                    }),
                    children_cut: Some(Cut {
                        count: 1,
                        limit: MAX_CHILDREN,
                    }),
                },
            },
            lines: &[],
            contains: &["value=\""],
            // Every cut is stated in the header: a tree near a ceiling reaches the agent as a
            // preview of its head, so a statement after the tree would never be read.
            header: vec![
                format!("the {MAX_TREE_NODES}-node ceiling"),
                "2 nodes at the 25-deep ceiling".to_string(),
                format!("1 value cut at {RENDER_VALUE_CHARS} chars"),
                // The read's own value cap is a separate ceiling from the display cap.
                format!("1 value cut at {MAX_VALUE_CHARS} chars"),
                format!("3 names cut at {MAX_NAME_CHARS} chars"),
                format!("4 nodes claim more than {MAX_CHILDREN} children"),
                format!(
                    "1 node listed more than {MAX_CHILDREN} children — only the first {MAX_CHILDREN} \
                     were read"
                ),
            ],
            absent: vec![long],
            counts: None,
        };

        for case in [pre_order, truncated] {
            let text = render_tree(&case.observation, counts(&case.observation.root));
            let lines: Vec<&str> = text.lines().collect();
            for (index, prefix) in case.lines.iter().enumerate() {
                assert!(
                    lines[index].starts_with(prefix),
                    "{}: line {index} should start with {prefix:?}: {}",
                    case.name,
                    lines[index]
                );
            }
            for expected in case.contains {
                assert!(
                    text.contains(expected),
                    "{}: the rendering does not carry {expected:?}: {text}",
                    case.name
                );
            }
            for unexpected in &case.absent {
                assert!(
                    !text.contains(unexpected),
                    "{}: the rendering carries {unexpected:?}: {text}",
                    case.name
                );
            }
            if !case.header.is_empty() {
                let header_end = lines
                    .iter()
                    .position(|line| line.starts_with("[e"))
                    .expect("the tree renders under the header");
                let header = lines[..header_end].join("\n");
                for statement in &case.header {
                    assert!(
                        header.contains(statement.as_str()),
                        "{}: the header does not state {statement:?}: {header}",
                        case.name
                    );
                }
            }
            if let Some((nodes, interactive)) = case.counts {
                let measured = counts(&case.observation.root);
                assert_eq!(measured.nodes, nodes, "{}: nodes", case.name);
                assert_eq!(
                    measured.interactive, interactive,
                    "{}: interactive elements",
                    case.name
                );
                assert!(
                    measured.interactive < AX_THIN_INTERACTIVE_ELEMENTS,
                    "{}: fewer interactive elements than the thin-surface threshold",
                    case.name
                );
            }
        }
    }

    fn look_of(obs: &Observation, target: TargetSpec) -> Look {
        Look {
            target,
            counts: counts(&obs.root),
            obs: obs.clone(),
            taken_at: Instant::now(),
        }
    }

    fn window_target(label: &str) -> TargetSpec {
        TargetSpec::Window {
            app_pid: Some(7),
            app_name: "TestApp".to_string(),
            key: WindowKey::new(label),
            label: label.to_string(),
        }
    }

    /// A narrowed look and what it must answer: its matches, what it must not print, or the
    /// refusal.
    struct NarrowCase {
        name: &'static str,
        narrow: Narrow,
        refs: bool,
        present: &'static [&'static str],
        absent: &'static [&'static str],
        refusal: &'static str,
    }

    fn subtree_narrow() -> Narrow {
        Narrow {
            at: Some(Locator {
                path: vec![1],
                role: "AXGroup".to_string(),
                name: None,
                frame: None,
            }),
            depth: Some(2),
            kind: None,
        }
    }

    #[expect(clippy::too_many_lines)] // one cohesive table over the narrowed looks
    #[test]
    fn narrow_look_cases() {
        let obs = sample_observation();
        let look = look_of(&obs, window_target("k"));
        let cases = vec![
            NarrowCase {
                name: "by_kind_lists_matches_and_names_the_kinds_present",
                narrow: Narrow {
                    kind: Some("button".to_string()),
                    ..Narrow::default()
                },
                refs: true,
                present: &[
                    "elements of kind \"button\" — 2 of 4 nodes",
                    "[e2] Button \"OK\"",
                    "[e4] Button \"Cancel\"",
                    "— at path 1",
                    "— at path 2.1",
                    "it covers part of the surface only",
                ],
                absent: &[],
                refusal: "",
            },
            NarrowCase {
                // Narrowing composes: a kind inside a ref's subtree is only that subtree's.
                name: "a_kind_inside_a_subtree_is_only_that_subtrees",
                narrow: Narrow {
                    at: Some(Locator {
                        path: vec![1],
                        role: "AXGroup".to_string(),
                        name: None,
                        frame: None,
                    }),
                    kind: Some("button".to_string()),
                    ..Narrow::default()
                },
                refs: true,
                present: &["— 1 of 2 nodes", "Button \"Cancel\""],
                absent: &["Button \"OK\""],
                refusal: "",
            },
            NarrowCase {
                name: "a_mistyped_kind_names_the_kinds_present",
                narrow: Narrow {
                    kind: Some("buton".to_string()),
                    ..Narrow::default()
                },
                refs: true,
                present: &["no element of kind \"buton\"", "kinds present: Button (2)"],
                absent: &[],
                refusal: "",
            },
            NarrowCase {
                name: "a_subtree_keeps_its_shape",
                narrow: subtree_narrow(),
                refs: true,
                present: &[
                    "[e3] Group",
                    "  [e4] Button \"Cancel\" actions=[press]",
                    "2 levels beneath 2",
                ],
                absent: &[],
                refusal: "",
            },
            NarrowCase {
                name: "a_subtree_prints_no_refs_once_the_turn_ended",
                narrow: subtree_narrow(),
                refs: false,
                present: &["[No refs:"],
                absent: &["[e3]"],
                refusal: "",
            },
            NarrowCase {
                name: "a_ref_that_no_longer_names_the_element_is_refused",
                narrow: Narrow {
                    at: Some(Locator {
                        path: vec![1],
                        role: "AXButton".to_string(),
                        name: Some("Gone".to_string()),
                        frame: None,
                    }),
                    ..Narrow::default()
                },
                refs: true,
                present: &[],
                absent: &[],
                refusal: ERR_STALE_ELEMENT,
            },
        ];
        for case in cases {
            let result = render_narrow(&look, &case.narrow, case.refs);
            if !case.refusal.is_empty() {
                let err = format!(
                    "{}",
                    result
                        .err()
                        .unwrap_or_else(|| panic!("{}: the look must be refused", case.name))
                );
                assert!(
                    err.starts_with(case.refusal),
                    "{}: the refusal: {err}",
                    case.name
                );
                continue;
            }
            let answer = result.unwrap_or_else(|e| panic!("{}: {e}", case.name));
            for expected in case.present {
                assert!(
                    answer.contains(expected),
                    "{}: the answer does not say {expected:?}: {answer}",
                    case.name
                );
            }
            for unexpected in case.absent {
                assert!(
                    !answer.contains(unexpected),
                    "{}: the answer prints {unexpected:?}: {answer}",
                    case.name
                );
            }
        }
    }

    #[test]
    #[serial_test::serial(computer_looks)]
    fn the_remembered_look_keeps_a_few_surfaces_and_drops_the_least_recently_looked_at() {
        let obs = sample_observation();
        let counts = counts(&obs.root);
        let keys = (0..=MAX_LOOKS)
            .map(|i| window_target(&format!("k{i}")))
            .collect::<Vec<_>>();
        for key in &keys {
            remember_look(key, obs.clone(), counts);
        }
        assert!(
            with_look(&keys[0]).is_none(),
            "the least recently looked at surface is dropped first"
        );
        for key in &keys[1..] {
            assert!(with_look(key).is_some(), "{} is kept", key.describe());
        }
        with_look(&keys[2]);
        remember_look(&window_target("fresh"), obs.clone(), counts);
        assert!(
            with_look(&keys[1]).is_none(),
            "a read did not keep the least recently observed surface"
        );
        assert!(with_look(&keys[2]).is_some());
        remember_look(&keys[3], obs.clone(), counts);
        remember_look(&window_target("fresh2"), obs.clone(), counts);
        assert!(
            with_look(&keys[3]).is_some(),
            "the re-observed record survived"
        );
        assert!(with_look(&keys[2]).is_none(), "the older one went");
        forget_look(&keys[3]);
        assert!(with_look(&keys[3]).is_none(), "a gone surface is forgotten");
        for key in &keys {
            forget_look(key);
        }
        forget_look(&window_target("fresh"));
        forget_look(&window_target("fresh2"));
    }

    /// A tree after the change, and what the change answer must say about it.
    struct ChangeCase {
        name: &'static str,
        after: Observation,
        present: &'static [&'static str],
        absent: &'static [&'static str],
    }

    #[test]
    #[serial_test::serial(computer_looks)]
    fn change_answer_cases() {
        let before = framed_observation();
        assert!(
            render_changes(
                &look_of(&before, window_target("k")),
                &before,
                counts(&before.root)
            )
            .contains("nothing changed on this surface since the observation taken")
        );
        let same = render_changes(
            &look_of(&before, window_target("k")),
            &before,
            counts(&before.root),
        );
        assert!(
            !same.contains("[e"),
            "a change answer prints no refs: {same}"
        );
        let mut moved = framed_observation();
        moved.root.children[0].frame = Some((70.0, 10.0, 80.0, 24.0));
        moved.root.children[0].value = Some("typed".to_string());
        moved.root.children[1].children[0].name = Some("Cancelled".to_string());
        moved.root.children.push(leaf("AXTextField", "Search"));

        let mut restated = framed_observation();
        restated.root.children[0].actions.clear();
        restated.root.children[1].children[0].focused = true;

        // A control whose position the read did not report is never called moved, and a side with
        // no value says so instead of a pair of quotes the observation never printed.
        let mut unplaced = sample_observation();
        assert!(unplaced.root.children[1].children[0].frame.is_none());
        unplaced.root.children[1].children[0].value = Some("other".to_string());

        let cases = vec![
            ChangeCase {
                name: "moves_renames_and_new_elements_are_paired",
                after: moved,
                present: &[
                    "2 appeared, 1 disappeared, 1 moved, 1 changed value",
                    "Button \"OK\" from (10, 10) to (70, 10)",
                    "Button \"Cancel\"",
                    "Button \"Cancelled\"",
                    "TextField \"Search\"",
                    "changed value:",
                ],
                absent: &[],
            },
            ChangeCase {
                name: "actions_and_focus_are_restated",
                after: restated,
                present: &[
                    "1 changed actions, 1 changed focus",
                    "Button \"OK\" — actions [press] → []",
                    "Button \"Cancel\" is focused now",
                ],
                absent: &[],
            },
            ChangeCase {
                name: "an_unplaced_element_is_never_reported_as_moved",
                after: unplaced,
                present: &["changed value:", "no value → \"other\""],
                absent: &["moved:"],
            },
        ];
        for case in cases {
            let answer = render_changes(
                &look_of(&before, window_target("k")),
                &case.after,
                counts(&case.after.root),
            );
            for expected in case.present {
                assert!(
                    answer.contains(expected),
                    "{}: the answer does not say {expected:?}: {answer}",
                    case.name
                );
            }
            for unexpected in case.absent {
                assert!(
                    !answer.contains(unexpected),
                    "{}: the answer prints {unexpected:?}: {answer}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn a_ref_is_acted_on_only_while_it_still_names_what_was_seen() {
        let obs = framed_observation();
        let root = &obs.root;
        let locator = Locator {
            path: vec![0],
            role: "AXButton".to_string(),
            name: Some("OK".to_string()),
            frame: None,
        };
        let (_, confirmed) =
            resolve_locator_confirmed(root, &locator).expect("the button is there");
        assert_eq!(confirmed, LocatorConfirmation::PositionUnconfirmed);
        assert!(
            confirmed
                .note(&locator)
                .contains("its place could not be confirmed"),
            "{}",
            confirmed.note(&locator)
        );

        let placed = Locator {
            frame: Some((10.0, 10.0, 80.0, 24.0)),
            ..locator.clone()
        };
        let (_, confirmed) = resolve_locator_confirmed(root, &placed).expect("the button is there");
        assert_eq!(confirmed, LocatorConfirmation::SamePlace);
        assert!(
            confirmed
                .note(&placed)
                .contains("is still where the observation showed it"),
            "{}",
            confirmed.note(&placed)
        );

        let reflowed = Locator {
            frame: Some((300.0, 300.0, 80.0, 24.0)),
            ..locator
        };
        let err = resolve_locator_confirmed(root, &reflowed).expect_err("it moved away");
        assert!(err.to_string().starts_with(ERR_STALE_ELEMENT), "{err}");
        assert!(err.to_string().contains("nothing was sent"), "{err}");
    }

    #[test]
    fn a_surfaces_move_is_measured_against_one_record_with_a_tolerance() {
        let before = surface(100.0, 100.0, 800.0, 600.0);
        assert_eq!(geometry_change(&before, &before), None);
        assert_eq!(
            geometry_change(&before, &surface(101.0, 100.5, 800.0, 600.0)),
            None
        );
        assert_eq!(
            geometry_change(&before, &surface(100.0, 140.0, 800.0, 600.0)).as_deref(),
            Some("moved by (0.0, 40.0)")
        );
        assert_eq!(
            geometry_change(&before, &surface(100.0, 100.0, 900.0, 600.0)).as_deref(),
            Some("resized from 800x600 to 900x600")
        );
    }

    #[test]
    fn a_click_is_blocked_only_by_another_applications_window() {
        let geometry = surface(100.0, 100.0, 800.0, 600.0);
        let other = WindowAtPoint {
            app_pid: Some(1),
            app_name: "Slack".to_string(),
            title: "Team".to_string(),
            frame: Some(surface(200.0, 200.0, 900.0, 700.0)),
        };
        let addressed = window_target("Inbox");
        assert_eq!(
            UnderPoint::classify(&addressed, Some(2), Some(&geometry), Some(&other)),
            UnderPoint::Blocked(other.clone())
        );
        let covering = WindowAtPoint {
            frame: Some(geometry),
            ..other.clone()
        };
        assert_eq!(
            UnderPoint::classify(&addressed, Some(2), Some(&geometry), Some(&covering)),
            UnderPoint::Blocked(covering.clone())
        );
        let dialog = WindowAtPoint {
            app_pid: Some(2),
            app_name: "Mail".to_string(),
            title: "Save".to_string(),
            frame: Some(surface(300.0, 300.0, 400.0, 200.0)),
        };
        let verdict = UnderPoint::classify(&addressed, Some(2), Some(&geometry), Some(&dialog));
        assert_eq!(verdict, UnderPoint::ThatWindow(dialog.clone()));
        let check = ClickCheck {
            surface: &addressed,
            local: (10.0, 10.0),
            geometry: &geometry,
            addressed_pid: Some(2),
            addressed_frame: Some(&geometry),
            under: Some(&dialog),
        };
        let note = verdict.disclosure(&check).expect("a note, not a refusal");
        assert!(note.is_some_and(|note| note.contains("another window of the same application")));
        let itself = WindowAtPoint {
            title: "Inbox - Mail".to_string(),
            frame: Some(geometry),
            ..dialog.clone()
        };
        assert_eq!(
            UnderPoint::classify(&addressed, Some(2), Some(&geometry), Some(&itself)),
            UnderPoint::Clear
        );
        // The application is the same, but the window's own frame is unreadable (a menu-bar item, a
        // Dock tile, an element whose window is not reported): only which of its windows is
        // unknown.
        let unframed = WindowAtPoint {
            frame: None,
            ..dialog.clone()
        };
        assert_eq!(
            UnderPoint::classify(&addressed, Some(2), None, Some(&unframed)),
            UnderPoint::Clear
        );
        // A window that cannot be attributed is the unknown case, never a silent clear: the click
        // goes and the answer says it was not checked.
        let unattributed = WindowAtPoint {
            app_pid: None,
            ..other.clone()
        };
        for (addressed_pid, at) in [(None, &other), (Some(2), &unattributed)] {
            assert_eq!(
                UnderPoint::classify(&addressed, addressed_pid, Some(&geometry), Some(at)),
                UnderPoint::Unknown
            );
        }
        // Nothing to check: the screen, or a platform that cannot answer.
        assert_eq!(
            UnderPoint::classify(&TargetSpec::Screen, None, None, None),
            UnderPoint::Screen
        );
        assert_eq!(
            UnderPoint::classify(&addressed, Some(2), Some(&geometry), None),
            UnderPoint::Unknown
        );
        let surface_target = window_target("Inbox");
        let err = click_under_point(&ClickCheck {
            surface: &surface_target,
            local: (340.0, 220.0),
            geometry: &geometry,
            addressed_pid: Some(2),
            addressed_frame: Some(&geometry),
            under: Some(&other),
        })
        .expect_err("another application's window blocks the click");
        assert!(err.to_string().starts_with(ERR_BLOCKED), "{err}");
        assert!(err.to_string().contains("nothing was sent"), "{err}");
        assert!(
            err.to_string().contains("Slack"),
            "the answer names the application in the way: {err}"
        );
        assert!(
            err.to_string().contains("(340.0, 220.0) of 800x600"),
            "the refusal states the position and the surface's size: {err}"
        );
        assert!(
            err.to_string().contains("window \"Inbox\" of TestApp"),
            "the refusal names the surface: {err}"
        );
    }

    #[test]
    fn keep_value_records_only_what_it_cut() {
        let mut truncation = Truncation::default();
        let fits = "z".repeat(MAX_VALUE_CHARS);
        assert_eq!(truncation.keep_value(&fits), fits);
        assert!(truncation.values.is_none(), "nothing was cut");

        let long = "z".repeat(MAX_VALUE_CHARS + 5);
        let kept = truncation.keep_value(&long);
        assert_eq!(kept.chars().count(), MAX_VALUE_CHARS);
        assert!(kept.starts_with(&"z".repeat(MAX_VALUE_CHARS - 1)), "{kept}");
        assert!(kept.ends_with('…'), "the cut is marked: {kept}");
        let cut = truncation.values.expect("the cut is recorded");
        assert_eq!((cut.count, cut.limit), (1, MAX_VALUE_CHARS));

        // A name is cut at its own, lower cap and counted apart from the values.
        let name = truncation.keep_name(&"n".repeat(MAX_NAME_CHARS + 1));
        assert_eq!(name.chars().count(), MAX_NAME_CHARS);
        assert!(
            truncation
                .names
                .is_some_and(|cut| cut.limit == MAX_NAME_CHARS),
            "the name cut is recorded"
        );
        assert_eq!(truncation.values.expect("one value was cut").count, 1);

        // A label is cut at the same cap and only marked: labels are shown, not matched on.
        let label = cap_label(&"l".repeat(MAX_NAME_CHARS + 1));
        assert_eq!(label.chars().count(), MAX_NAME_CHARS);
        assert!(label.ends_with('…'), "{label}");
    }

    #[test]
    fn capture_budget_refuses_a_surface_it_cannot_hold() {
        let budget = CaptureBudget::default();
        let err = budget
            .charge("the screen composite", u32::MAX, u32::MAX)
            .unwrap_err();
        let text = err.to_string();
        assert!(text.starts_with(ERR_DEGRADED), "{text}");
        assert!(text.contains("the screen composite"), "{text}");
        assert!(
            text.contains("already holds 0 MiB of its 512 MiB"),
            "{text}"
        );
        assert!(budget.charge("a display frame", 3840, 2160).is_ok());

        let empty = budget.charge("a window still", 0, 1080).unwrap_err();
        assert!(empty.to_string().contains("no capture area"), "{empty}");

        let ceiling = budget
            .charge("the screen composite", u32::MAX, u32::MAX)
            .unwrap_err();
        assert!(is_ceiling_refusal(&ceiling), "{ceiling}");
        let other = taxonomy_error(ERR_DEGRADED, "a frame could not be allocated");
        assert!(!is_ceiling_refusal(&other));
    }

    #[test]
    fn capture_alloc_and_from_pixels_take_their_pixels_from_the_budget() {
        let budget = CaptureBudget::default();
        let blank = Capture::alloc(&budget, "a test frame", 4, 2).unwrap();
        assert_eq!(blank.rgba.len(), 4 * 2 * 4);
        assert!(
            blank.rgba.iter().all(|byte| *byte == 0),
            "the buffer is blank"
        );
        assert_eq!(budget.held(), 8);
        let pixels = vec![7u8; 4];
        let copied = Capture::from_pixels(&budget, "a small frame", 1, 1, &pixels).unwrap();
        assert_eq!(copied.rgba, pixels, "the pixels are taken over as they are");
        assert_eq!(
            budget.held(),
            8 + 1,
            "the claim is the pixels, not the bytes"
        );
    }

    #[test]
    fn capture_budget_follows_the_pixels_a_call_holds() {
        let budget = CaptureBudget::default();
        // Each fits the ceiling alone; together they do not, which is the point of a per-call
        // budget over a per-buffer one. Claims carry the accounting on their own — no buffer
        // allocated here.
        let half = u32::try_from(CAPTURE_BUDGET_PIXELS / 2).unwrap();
        let composite = budget.charge("the composite", half, 1).unwrap();
        let frame = budget.charge("a display frame", half, 1).unwrap();
        assert_eq!(budget.held(), CAPTURE_BUDGET_PIXELS);
        // A full-size crop cannot join them, and the refusal names what the call already holds.
        let err = budget.charge("the zoom crop", half, 1).unwrap_err();
        assert!(err.to_string().contains("the zoom crop"), "{err}");
        assert!(err.to_string().contains("already holds 512 MiB"), "{err}");

        drop(frame);
        assert_eq!(budget.held(), u64::from(half));
        drop(composite);
        assert_eq!(budget.held(), 0);
        assert!(budget.charge("the zoom crop", half, 1).is_ok());

        let claim = budget
            .charge("a decoded image", 4, 4)
            .expect("the allowance is free again");
        assert_eq!(budget.remaining(), CAPTURE_BUDGET_PIXELS - 16);
        let adopted = Capture::adopt(claim, 4, 4, vec![0; 4 * 4 * 4]);
        assert_eq!(adopted.rgba.len(), 4 * 4 * 4);
        drop(adopted);
        assert_eq!(budget.remaining(), CAPTURE_BUDGET_PIXELS);
    }

    #[test]
    fn store_and_resolve_ref_round_trip_then_stale() {
        let obs = sample_observation();
        let target = TargetSpec::Focused;
        store_observation(target.clone(), &obs);

        let (locator, resolved_target) = resolve_ref("e2").unwrap();
        assert_eq!(locator.role, "AXButton");
        assert_eq!(locator.name.as_deref(), Some("OK"));
        assert_eq!(locator.path, vec![0]);
        assert_eq!(resolved_target, TargetSpec::Focused);

        let obs2 = Observation {
            app_name: "Other".to_string(),
            window_title: None,
            surface: surface(0.0, 0.0, 10.0, 10.0),
            root: node("AXWindow", Some("other")),
            truncation: Truncation::default(),
        };
        store_observation(target, &obs2);
        let err = resolve_ref("e2").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("stale-element"), "got {msg}");
        let err = resolve_ref("zzz").unwrap_err();
        assert!(format!("{err}").contains("stale-element"));
    }

    #[serial_test::serial(targets)]
    #[test]
    fn resolve_target_resolves_focused_screen_stored_and_unknown() {
        assert_eq!(resolve_target(None).unwrap(), TargetSpec::Focused);
        assert_eq!(resolve_target(Some("screen")).unwrap(), TargetSpec::Screen);
        in_session("resolve-target-test", || {
            let app = TargetSpec::App {
                app_pid: Some(42),
                app_name: "Safari".to_string(),
            };
            let ids = store_app_targets(std::slice::from_ref(&app));
            assert_eq!(ids.len(), 1);
            assert_eq!(resolve_target(Some(&ids[0])).unwrap(), app);
            let err = resolve_target(Some("w9999")).unwrap_err();
            assert!(format!("{err}").contains("unknown"), "{err}");
            assert!(!is_surface_gone(&err));
        });
    }

    /// Only a surface a call confirmed gone is reported as gone; a running app keeps its id.
    #[test]
    fn only_a_surface_confirmed_gone_is_typed_as_gone() {
        assert!(is_surface_gone(&surface_gone("Beta")));
        assert!(is_surface_gone(&app_absent("Beta", AppAbsence::Quit)));
        assert!(!is_surface_gone(&app_absent(
            "Beta",
            AppAbsence::Windowless
        )));
        assert!(!is_surface_gone(&app_absent(
            "Beta",
            AppAbsence::Unreachable
        )));
    }

    #[serial_test::serial(targets)]
    #[test]
    fn ids_last_for_the_session_not_for_the_turn() {
        let key = "session-lifetime-test";
        let app = TargetSpec::App {
            app_pid: Some(9),
            app_name: "Beta".to_string(),
        };
        let window = TargetSpec::Window {
            app_pid: Some(9),
            app_name: "Beta".to_string(),
            key: WindowKey::new("k9"),
            label: "notes".to_string(),
        };
        in_session(key, || {
            let app_id = store_app_targets(std::slice::from_ref(&app))[0].clone();
            let window_id = store_window_targets(std::slice::from_ref(&window))[0].clone();
            set_last_surface(window.clone());

            cleanup_turn(key);
            assert_eq!(resolve_target(Some(&app_id)).unwrap(), app);
            assert_eq!(resolve_target(Some(&window_id)).unwrap(), window);
            assert_eq!(last_surface(), Some(window.clone()));

            // Only the id whose surface a call confirmed gone stops resolving.
            forget_target(&window_id);
            assert!(resolve_target(Some(&window_id)).is_err());
            assert_eq!(resolve_target(Some(&app_id)).unwrap(), app);

            end_session(key);
            assert!(resolve_target(Some(&app_id)).is_err());
            assert_eq!(last_surface(), None);

            // Numbering survives the reset, so a pre-reset id can never name a new surface.
            let fresh = store_app_targets(std::slice::from_ref(&app))[0].clone();
            assert_ne!(fresh, app_id);
            assert!(resolve_target(Some(&app_id)).is_err());
        });
    }

    #[serial_test::serial(targets)]
    #[test]
    fn store_targets_keeps_ids_for_seen_surfaces_and_never_reuses_them() {
        in_session("target-identity-test", ids_follow_surfaces);
    }

    /// Run `body` as a call inside a session: session state is stored only for a call that has one.
    fn in_session<T>(key: &'static str, body: impl FnOnce() -> T) -> T {
        crate::agent::CURRENT_TOOL_AGENT_ID.sync_scope(Some(key.to_string()), body)
    }

    fn ids_follow_surfaces() {
        let window = |name: &str, key: &str| TargetSpec::Window {
            app_pid: Some(7),
            app_name: name.to_string(),
            key: WindowKey::new(key),
            label: format!("{key} title"),
        };
        let first = store_window_targets(&[window("Beta", "k1"), window("Beta", "k2")]);
        let again = store_window_targets(&[window("Beta", "k2"), window("Beta", "k1")]);
        assert_eq!(again, vec![first[1].clone(), first[0].clone()]);

        let after = store_window_targets(&[window("Beta", "k1"), window("Beta", "k3")]);
        assert_eq!(after[0], first[0]);
        assert_ne!(after[1], first[1]);
        assert_eq!(
            resolve_target(Some(&first[1])).unwrap(),
            window("Beta", "k2"),
            "the id of a vanished window must not name its replacement"
        );

        // A rename keeps the key, so the id still names the same window and the label follows it;
        // the app name a backend reports can change with the title and must not mint a second id.
        let renamed = TargetSpec::Window {
            app_pid: Some(7),
            app_name: "Beta (renamed)".to_string(),
            key: WindowKey::new("k1"),
            label: "a new title".to_string(),
        };
        assert_eq!(store_window_targets(&[renamed])[0], first[0]);
        assert_eq!(
            resolve_target(Some(&first[0])).unwrap().describe(),
            "window \"a new title\" of Beta (renamed)"
        );
    }

    #[test]
    fn a11y_scale_maps_only_reconcilable_spaces() {
        let screen = surface(0.0, 0.0, 1440.0, 900.0);
        assert_eq!(a11y_scale(&screen, (1440, 900)), Some(1.0));
        assert_eq!(a11y_scale(&screen, (2880, 1800)), Some(2.0));
        // A space larger than the capture, per-axis disagreement and empty extents are all
        // refusals.
        assert_eq!(a11y_scale(&screen, (720, 450)), None);
        assert_eq!(a11y_scale(&screen, (2880, 900)), None);
        assert_eq!(a11y_scale(&surface(0.0, 0.0, 0.0, 0.0), (10, 10)), None);
        let scaled = scale_geometry(surface(10.0, 20.0, 100.0, 50.0), 2.0);
        assert_eq!(scaled, surface(20.0, 40.0, 200.0, 100.0));
    }

    /// A locator and the decision it must reach against the sample surface.
    struct LocatorCase {
        name: &'static str,
        locator: Locator,
        decision: &'static str,
    }

    #[test]
    fn locator_cases() {
        let obs = sample_observation();
        let root = &obs.root;
        let cases = [
            LocatorCase {
                name: "a_path_that_names_the_element",
                locator: Locator {
                    path: vec![0],
                    role: "button".to_string(),
                    name: Some("OK".to_string()),
                    frame: None,
                },
                decision: "path",
            },
            LocatorCase {
                name: "a_path_out_of_range_falls_through_to_the_search",
                locator: Locator {
                    path: vec![99],
                    role: "button".to_string(),
                    name: Some("OK".to_string()),
                    frame: None,
                },
                decision: "unique",
            },
            LocatorCase {
                name: "a_path_whose_element_moved_on_is_searched_again",
                locator: Locator {
                    path: vec![1],
                    role: "button".to_string(),
                    name: Some("OK".to_string()),
                    frame: None,
                },
                decision: "unique",
            },
            LocatorCase {
                name: "nothing_matches",
                locator: Locator {
                    path: Vec::new(),
                    role: "AXSlider".to_string(),
                    name: None,
                    frame: None,
                },
                decision: "not-found",
            },
            LocatorCase {
                name: "several_elements_match",
                locator: Locator {
                    path: Vec::new(),
                    role: "AXButton".to_string(),
                    name: None,
                    frame: None,
                },
                decision: "ambiguous",
            },
            LocatorCase {
                name: "a_pinning_path_wins_over_the_ambiguity",
                locator: Locator {
                    path: vec![0],
                    role: "AXButton".to_string(),
                    name: None,
                    frame: None,
                },
                decision: "path",
            },
        ];
        for case in cases {
            let decision = match resolve_locator(root, &case.locator) {
                LocatorMatch::Path(_) => "path",
                LocatorMatch::Unique(_) => "unique",
                LocatorMatch::Ambiguous => "ambiguous",
                LocatorMatch::NotFound => "not-found",
            };
            assert_eq!(decision, case.decision, "{}: the decision", case.name);
        }
    }

    #[test]
    fn parse_key_chord_splits_modifiers_and_key() {
        let (mods, key) = parse_key_chord("cmd+shift+t").unwrap();
        assert_eq!(mods, vec![Modifier::Cmd, Modifier::Shift]);
        assert_eq!(key, "t");

        let (mods, key) = parse_key_chord("Control+Alt+Delete").unwrap();
        assert_eq!(mods, vec![Modifier::Ctrl, Modifier::Alt]);
        assert_eq!(key, "delete");

        let (mods, key) = parse_key_chord("Return").unwrap();
        assert!(mods.is_empty());
        assert_eq!(key, "return");

        let (_, key) = parse_key_chord("5").unwrap();
        assert_eq!(key, "5");
        let (_, key) = parse_key_chord("F5").unwrap();
        assert_eq!(key, "f5");

        let err = parse_key_chord("cmd+ctrl").unwrap_err();
        assert!(format!("{err}").contains("no key name"), "got {err}");

        assert!(parse_key_chord("cmd+").is_err());
        let err = parse_key_chord("a+b").unwrap_err();
        assert!(format!("{err}").contains("too many key names"), "got {err}");
    }

    /// The middle outcome is stated as delivered-but-unconfirmed, never as a failure to repeat.
    #[test]
    fn impact_text_states_the_middle_outcome_without_advice_to_retry() {
        assert_eq!(impact_text("clicked".to_string(), Impact::Done), "clicked");
        assert_eq!(
            impact_text(
                "clicked".to_string(),
                Impact::DoneWith("ctrl may still be held down".to_string()),
            ),
            "clicked — ctrl may still be held down",
            "a delivered note is appended"
        );
        let text = impact_text(
            "clicked element e3".to_string(),
            Impact::Unconfirmed("the call was abandoned".to_string()),
        );
        assert!(text.starts_with("applied but unconfirmed"), "{text}");
        assert!(text.contains("may already have taken effect"), "{text}");
        assert!(text.contains("act twice"), "{text}");
        assert!(!text.contains("retry"), "{text}");

        // A note records one more fact without changing what the outcome says: done stays DoneWith,
        // never Unconfirmed, and unconfirmed stays unconfirmed.
        let delivered = impact_note(Impact::Done, "the probe failed".to_string());
        assert!(
            matches!(&delivered, Impact::DoneWith(note) if note == "the probe failed"),
            "impact_note never upgrades a delivered outcome"
        );
        let delivered_with = impact_note(
            Impact::DoneWith("ctrl may still be held".to_string()),
            "the probe failed".to_string(),
        );
        assert!(
            matches!(&delivered_with, Impact::DoneWith(note) if note == "ctrl may still be held; the probe failed"),
            "a delivered note accumulates as a delivered note"
        );
        let unconfirmed = impact_note(
            Impact::Unconfirmed("abandoned".to_string()),
            "the probe failed".to_string(),
        );
        assert!(
            matches!(&unconfirmed, Impact::Unconfirmed(note) if note == "abandoned; the probe failed"),
            "an unconfirmed outcome stays unconfirmed"
        );
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    struct ShareCase {
        name: &'static str,
        call: u64,
        share: u64,
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn share_budget_cases() {
        for case in [
            ShareCase {
                name: "a_share_asking_for_more_than_the_call_has",
                call: 10,
                share: 60,
            },
            ShareCase {
                name: "a_spent_share",
                call: 20,
                share: 0,
            },
        ] {
            let call = Budget::new(
                Duration::from_secs(case.call),
                "window listing",
                "the running applications",
            );
            let share = call.slice(Duration::from_secs(case.share), "the windows of Numbers");
            if case.share > case.call {
                assert_eq!(share.end, call.end, "{}: the shared end", case.name);
            }
            assert!(share.end <= call.end, "{}: the call's end", case.name);
            assert!(share.limit <= call.limit, "{}: the call's limit", case.name);

            if case.share == 0 {
                assert_eq!(
                    share.limit,
                    Duration::ZERO,
                    "{}: the spent share",
                    case.name
                );
                assert!(share.expired(), "{}: the share is spent", case.name);
                assert!(!call.expired(), "{}: the call is not", case.name);
            }

            let note = share.note();
            assert!(
                note.contains("the windows of Numbers stopped answering accessibility requests"),
                "{}: {note}",
                case.name
            );
            // A share never speaks for the call: the model must not be told to retry a call that
            // returned results, nor read the skipped application as the call failing.
            assert!(!note.contains("abandoned"), "{}: {note}", case.name);
            assert!(!note.contains("retry"), "{}: {note}", case.name);

            let refused = anyhow!("window refused");
            let reason = unread_reason(&refused, &share);
            if case.share == 0 {
                assert_eq!(reason, SKIPPED_NOTE, "{}: a spent share", case.name);
            } else {
                assert_eq!(reason, "window refused", "{}: an unspent share", case.name);
            }
            assert_eq!(
                unread_reason(&refused, &call),
                "window refused",
                "{}: the call itself",
                case.name
            );
        }
    }
}
