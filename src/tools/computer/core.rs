//! Platform-agnostic core of the Computer (GUI observe/act) tool.
//!
//! Owns the shared types, coordinate math, the element-ref/observation lifecycle,
//! the target registry, tree rendering, and the [`Backend`] trait. Contains no
//! platform imports — `macos.rs`/`linux.rs`/`windows`/`stub.rs` implement
//! [`Backend`].
//!
//! Two honesty rules live here because every backend owes them: a read states
//! every ceiling it hit ([`Truncation`], rendered into the observation header), and
//! an actuating call states which of the three outcomes it reached ([`Impact`],
//! rendered by [`impact_text`]).
//!
//! Items carrying a platform-scoped `expect(dead_code)` are the ones only a real
//! backend — or a test — calls: on a platform whose backend is the stub they
//! have no caller, and they stay, as they are the real backends' contract. Per
//! item, not module-wide, so `dead_code` stays armed everywhere else.

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

/// Maximum normalized coordinate value, inclusive on both axes ([`NORMALIZED_MAX`]).
pub(crate) const NORMALIZED_MAX: f64 = 1000.0;

/// Surfaces exposing fewer than this many interactive elements are "AX-thin" —
/// the model is nudged toward screenshots over the accessibility tree.
pub(crate) const AX_THIN_INTERACTIVE_ELEMENTS: usize = 5;

/// Node ceiling of one tree read — prevents unbounded output. A read that
/// reaches it records that in [`Truncation`], which the header states. Windows
/// walks under its own, lower ceiling (UI Automation pays one cross-process call
/// per property), so only the backends that use this one compile it.
#[cfg(any(target_os = "macos", target_os = "linux", test))]
pub(crate) const MAX_TREE_NODES: usize = 4000;

/// Ceiling on the children ONE node may contribute. A target that advertises an
/// absurd child list — a spreadsheet's cell list is the classic case, a count near
/// 2^31 — would otherwise be paid for inside a single exchange, long before the node
/// ceiling applies, so the advertised count is read first and a node beyond this
/// ceiling is not walked at all.
#[cfg(any(target_os = "linux", test))]
pub(crate) const MAX_CHILDREN: usize = 1000;

/// Character cap on a node value shown in a tree dump. The dump is a map of the
/// surface, not a copy of its text, so a long value is cut here and counted in
/// the truncation header.
pub(crate) const RENDER_VALUE_CHARS: usize = 120;

/// Most characters one node value keeps, on every platform. It is the read's cap,
/// not the dump's: a document-shaped field reports a whole file as its value, and
/// keeping it would inflate the tree by that file per node. A longer value keeps
/// its beginning with an ellipsis and is counted in the header
/// ([`Truncation::keep_value`]). Linux reads a value only one character past the
/// dump's own cap, so this one is out of its reach by design.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) const MAX_VALUE_CHARS: usize = 2000;

/// Most characters one accessible name keeps, on every platform — the same
/// discipline as a value ([`Truncation::keep_name`]), because a title can be a
/// document too. A ref records the name it saw and matches it against a freshly read
/// one, cut the same way on both sides; the recorded child-index path is the primary
/// key, so a capped name only decides the fallback — two nodes whose names differ
/// solely past the cap resolve as [`LocatorMatch::Ambiguous`] instead of one of them
/// being picked.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) const MAX_NAME_CHARS: usize = 200;

/// The longest string one computer action accepts, in characters — `text`, `keys`,
/// `target`, `app` and `ref` alike. Typing pays one synthetic event per character while
/// the action lock is held, so a longer argument is refused before the call starts work
/// instead of holding the tool for as long as it is big.
pub(crate) const MAX_ARG_CHARS: usize = 2000;

/// The scroll range in wheel ticks, on every platform: one tick is one notch, and it
/// costs one wheel event on macOS and Windows but a press/release pair per tick on
/// Linux — under the same lock. Zero is outside the range too: it names no motion.
pub(crate) const MAX_SCROLL_AMOUNT: u32 = 100;

/// Ceiling on the pixels ONE capture call may hold at once, chosen by the product
/// rather than taken from what the machine reports. This process aborts on a failed
/// allocation, so no figure the machine reports may decide how much is allocated:
/// every buffer the tool keeps for a capture — a display's frame, the composite of the
/// whole virtual desktop, the zoom crop taken out of it, the pixels a portal
/// screenshot decoded to — holds a [`Claim`] on this allowance, weighed before the
/// buffer exists and given back when it is dropped.
///
/// 128 Mi pixels is 512 MiB of RGBA. A whole-desktop capture of a two-6K-display wall
/// costs its 40.7 MP composite and a 40.7 MP zoom crop — 81 MP — and fits; a wall of
/// four 6K displays needs 163 MP and is refused, naming this ceiling. A display's
/// frame is alive only while it is blitted, so the number of displays does not
/// multiply the cost. The PNG encode holds no second pixel copy and is bounded by the
/// pixels it encodes; what a platform allocates for itself (the capture framework's
/// texture, a GDI section) fails with a null handle or an `HRESULT` instead of
/// aborting, and is bounded by the same geometry checks.
pub(crate) const CAPTURE_BUDGET_PIXELS: u64 = 128 * 1024 * 1024;

// ── Deadlines ───────────────────────────────────────────────────────────
//
// Every exchange is bounded by a deadline the product chose, and every call by a bound on
// top of that (the aggregate budget below on Linux and macOS, a job deadline on Windows),
// so a target that does not answer costs a lost detail instead of the action lock. The
// figures are these constants and each backend's header; what the model is told lives in
// the tool's prompt.

/// How long ONE accessibility exchange may take before its target counts as not
/// answering: the D-Bus method timeout on Linux and the AX messaging timeout on macOS,
/// deliberately not shorter than the OS default applications rely on (1.5 s measured on
/// a stopped application here). Windows sets its own transaction timeout.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const EXCHANGE_DEADLINE: Duration = Duration::from_secs(10);

/// The aggregate budget of one observation, resolution included: well clear of what a
/// responsive target at the node ceiling pays (~2.5 s of exchanges measured over AX,
/// tens of seconds at a slow AT-SPI round trip), so only a target that stops answering
/// reaches it.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const OBSERVE_BUDGET: Duration = Duration::from_secs(120);

/// One element action also observes the surface it addresses, so it pays the
/// observation budget plus one exchange for the action itself.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const ACT_BUDGET: Duration = OBSERVE_BUDGET.saturating_add(EXCHANGE_DEADLINE);

/// One listing, application walk or surface lookup — a handful of exchanges per
/// application, where a hung one must not hold the call either.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const LIST_BUDGET: Duration = Duration::from_secs(20);

/// What ONE application may spend of a listing's or a search's budget: one stalled
/// exchange, checked before every read that application makes, so a hung application is
/// caught at the check after its stall. A share is overshot by the read in flight, never
/// by more, and the healthy applications keep the rest of the call.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const APP_SLICE: Duration = EXCHANGE_DEADLINE;

/// The reason a listing gives for one item it skipped, where the line already names that
/// item: it did not answer within its share of the call, so nothing about it was read.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const SKIPPED_NOTE: &str = "did not answer within its share of the call — skipped";

/// The reason a listing gives for one unread item: a spent share is one skipped item, said
/// in [`SKIPPED_NOTE`] because the line already names it and the call did return; anything
/// else is that item's own failure, stated in full.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn unread_reason(err: &anyhow::Error, share: &Budget) -> String {
    if share.expired() {
        SKIPPED_NOTE.to_string()
    } else {
        format!("{err:#}")
    }
}

/// One call's aggregate deadline, checked between exchanges, because a platform walk
/// runs inside a blocking frame no external timer can interrupt. Linux and macOS make
/// one per call and thread it through resolution and the walk; Windows makes its walk
/// budget at the walk and bounds the rest with the job deadline it awaits the apartment
/// thread under. The macOS listing clones one so it can hand a copy to the AX thread and
/// still ask afterwards which application's share ran out.
#[derive(Clone)]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) struct Budget {
    end: Instant,
    limit: Duration,
    what: &'static str,
    target: String,
    /// This is one item's share of the call rather than the call itself, so running
    /// out means that item was skipped — not that the call was abandoned.
    part: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Budget {
    /// Start the clock for `what` (the operation the model asked for) against
    /// `target`, named the way results name a surface.
    pub(crate) fn new(limit: Duration, what: &'static str, target: impl Into<String>) -> Self {
        Self {
            end: Instant::now() + limit,
            limit,
            what,
            target: target.into(),
            part: false,
        }
    }

    /// One application's share of this budget: at most `share` from now, and never past the
    /// call's own end, so the step it bounds cannot outlive the call that owns it. Running
    /// out of a share skips that application; it does not abandon the call.
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

    /// The surface this budget names, once the call has resolved it.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn target(&self) -> &str {
        &self.target
    }

    /// Point the budget at the surface the call resolved: every failure from here on
    /// names the window rather than the target as the model wrote it. [`name_surface`]
    /// is the only caller, so a label and the budget it bounds are set together.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn retarget(&mut self, target: &str) {
        target.clone_into(&mut self.target);
    }

    /// Whether the budget is spent, without building the error or logging it — for the
    /// callers that record the expiry and describe it later.
    pub(crate) fn expired(&self) -> bool {
        Instant::now() >= self.end
    }

    /// Fail the call if its budget is spent — the one shape a walk checks with, so
    /// every backend notices it between exchanges and nowhere inside one.
    pub(crate) fn check(&self) -> anyhow::Result<()> {
        if self.expired() {
            Err(self.error())
        } else {
            Ok(())
        }
    }

    /// The sentence for a target that stopped answering: it names the surface, says what
    /// ran out of time and whether that ended the call or only skipped the item. Untagged,
    /// so a caller that has more to say about the result can build the answer from it.
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
                "{} stopped answering accessibility requests — the {} was abandoned after {}s; \
                 retry, or use screenshot instead",
                self.target,
                self.what,
                self.limit.as_secs()
            )
        }
    }

    /// The honest answer for a spent budget, tagged and logged ([`Budget::note`]).
    fn error(&self) -> anyhow::Error {
        taxonomy_error(ERR_DEGRADED, self.note())
    }
}

/// The label a call uses for the surface it resolved, both parts capped like any name.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn surface_label(app_name: &str, title: Option<&str>) -> String {
    window_description(&cap_label(app_name), &cap_label(title.unwrap_or_default()))
}

/// The label a listing or a search uses for one application's windows — the one way such a
/// call names them, capped like any other name.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn windows_label(app_name: &str) -> String {
    format!("the windows of {}", cap_label(app_name))
}

/// The same label, and the call's budget pointed at it: the one place the identity →
/// label → budget rule lives, so a call cannot report a surface one way and bound it under
/// another. Returns the label for the errors that name it.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn name_surface(budget: &mut Budget, app_name: &str, title: Option<&str>) -> String {
    let label = surface_label(app_name, title);
    budget.retarget(&label);
    label
}

/// The failure for a surface that did not answer one of the reads a call makes before
/// it can show, act on or capture anything. Both AT-SPI and AX report an unanswered
/// request as a missing value, so a target that stopped answering is named — with what
/// the call asked it for — instead of being reported as an attribute it does not have
/// or as a surface that went away.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn unanswered(surface: &str, what: &str) -> anyhow::Error {
    taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{surface} did not answer {what} — it may be busy or hung; retry, or use screenshot \
             instead"
        ),
    )
}

/// Accessible role names considered genuinely actionable (AX-prefix already
/// stripped, matched case-insensitively). Used alongside non-empty `actions`.
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

// ── Error taxonomy ──────────────────────────────────────────────────────
//
// Backend/runtime failures (permission, stale/ambiguous elements, unsupported
// surfaces) MUST start with one of these words (via [`taxonomy_error`]) so a
// model can branch on the category. Argument-validation errors (invalid
// coordinates, unknown targets/keys, missing refs, wait caps) are plain and
// self-describing — no taxonomy tag needed.

pub(crate) const ERR_PERMISSION_DENIED: &str = "permission-denied";
pub(crate) const ERR_UNSUPPORTED: &str = "unsupported";
pub(crate) const ERR_DEGRADED: &str = "degraded";
/// The session cannot receive the call at all — see [`Backend::desktop_unavailable`].
pub(crate) const ERR_DESKTOP_UNAVAILABLE: &str = "desktop-unavailable";
pub(crate) const ERR_STALE_ELEMENT: &str = "stale-element";
/// The system's own framework (or, on Windows, a structured exception raised
/// inside it) failed while serving the call: the call was abandoned and the
/// process is intact. Distinct from `degraded` because retrying the same call is
/// unlikely to help — the fault is in foreign code, not in the target's state.
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
/// The surface a call names is gone: the window closed or was replaced.
/// Deliberately not `not-matched`, whose guidance — coordinates or a screenshot of
/// where it used to be — cannot help when the window no longer exists.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) const ERR_SURFACE_GONE: &str = "surface-gone";
/// The target is mahbot's own interface and this platform cannot perform the
/// operation on itself from inside the process. Deliberately distinct from
/// `unsupported`, whose prompt guidance ("take another route … or coordinates")
/// would keep the model spinning on a class the platform refuses for that
/// operation.
#[cfg_attr(not(target_os = "windows"), expect(dead_code))]
pub(crate) const ERR_SELF_TARGET: &str = "self-target";

/// Build a taxonomy-prefixed error: `"<kind>: <msg>"`. The message MUST start
/// with the taxonomy word so the model can branch on the category.
#[must_use]
pub(crate) fn taxonomy_error(kind: &str, msg: impl fmt::Display) -> anyhow::Error {
    anyhow!("{kind}: {msg}")
}

/// The `framework-failed` failure: `what` names the operation, `detail` what the
/// system raised under it. Both platforms report the tag in this one shape, and it
/// is TYPED so an actuating call can tell "foreign code failed while the call was
/// running" (its input may be partly applied) from every other failure without
/// reading the text — see [`is_framework_failed`].
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

/// Whether `err` is the typed [`framework_failed`] failure — robust to a
/// `.context()` layer added above it.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn is_framework_failed(err: &anyhow::Error) -> bool {
    err.downcast_ref::<FrameworkFailed>().is_some()
}

/// The `framework-failed` category as a type, so a caller that must branch on it
/// does not depend on the message's shape.
#[derive(Debug)]
struct FrameworkFailed(anyhow::Error);

impl fmt::Display for FrameworkFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for FrameworkFailed {}

/// The `surface-gone` failure for a window of `app_name` (see [`ERR_SURFACE_GONE`]).
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn surface_gone(app_name: &str) -> anyhow::Error {
    taxonomy_error(
        ERR_SURFACE_GONE,
        format!(
            "the window of '{app_name}' this target names can no longer be identified — it \
             closed or was replaced; enumerate windows again"
        ),
    )
}

/// The `degraded` failure for a minimized window: it has no visible pixels, and
/// the frame it keeps is the one it had before it was minimized. `title` may be
/// empty (an untitled window), which [`window_description`] renders as "the window".
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

/// The `degraded` failure for pinned input whose surface could not be brought to
/// the front: nothing was sent, so the call cannot claim it landed there. `surface`
/// is a [`TargetSpec::describe`] or [`window_description`] spelling of it.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn not_brought_forward(surface: impl fmt::Display) -> anyhow::Error {
    taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{surface} could not be brought to the front — the input would have gone to another \
             window, so nothing was sent; bring the window forward (click it) and repeat the action"
        ),
    )
}

/// Build the `permission-denied` failure as a TYPED error: its text is the usual
/// `"permission-denied: …"`, and [`is_permission_denied`] tells it apart from
/// every other backend failure without reading that text.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn permission_denied(msg: impl fmt::Display) -> anyhow::Error {
    anyhow::Error::new(PermissionDenied(taxonomy_error(ERR_PERMISSION_DENIED, msg)))
}

/// The `degraded` refusal for a capture that would exceed the call's
/// [`CAPTURE_BUDGET_PIXELS`] allowance. TYPED, like [`permission_denied`], so the
/// Windows route classifier can tell a ceiling refusal — whose remedy is a smaller
/// surface — from every other degraded cause without reading the message.
fn ceiling_refused(what: &str, width: u32, height: u32, held: u64) -> anyhow::Error {
    anyhow::Error::new(CeilingRefusal(taxonomy_error(
        ERR_DEGRADED,
        format!(
            "{what} needs {} MiB of RGBA ({width}x{height}) and this call already holds {} MiB \
             of its {} MiB — name a smaller surface instead; a capture this large cannot be \
             held whole",
            mib(u64::from(width) * u64::from(height)),
            mib(held),
            mib(CAPTURE_BUDGET_PIXELS),
        ),
    )))
}

/// Whether `err` is the typed [`ceiling_refused`] failure — robust to a `.context()`
/// layer added above it.
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

/// Whether `err` is the typed [`permission_denied`] failure — robust to a
/// `.context()` layer added above it.
pub(crate) fn is_permission_denied(err: &anyhow::Error) -> bool {
    err.downcast_ref::<PermissionDenied>().is_some()
}

/// The `permission-denied` category as a type, so a caller that must branch on it
/// (see [`is_permission_denied`]) does not depend on the message's shape.
#[derive(Debug)]
struct PermissionDenied(anyhow::Error);

impl fmt::Display for PermissionDenied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for PermissionDenied {}

// ── Types ───────────────────────────────────────────────────────────────

/// Surface extent in global surface points (top-left origin): logical points on
/// macOS, physical pixels on Linux and Windows. For a window this is the window's
/// frame; for the screen target it is the virtual-desktop bounding rect across all
/// displays (origin at its min corner).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurfaceGeometry {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Normalized 0-1000 region (corner order is normalized by consumers).
///
/// Constructed at runtime only via serde deserialization of
/// `ComputerAction::Zoom` arguments. `Default` yields an all-zero
/// (degenerate/invalid) region and exists only so the `ComputerAction`
/// `EnumIter` (used by the schema lockstep test) can build a `Zoom` variant
/// via `Default::default()`.
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

/// Lightweight element reference: a DFS path from the tree root plus the
/// role/name snapshot captured at observe time (NOT a platform handle).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Locator {
    /// Child-index path from the root node.
    pub path: Vec<usize>,
    /// Raw accessibility role (may carry an `AX` prefix).
    pub role: String,
    /// Accessible name, if any.
    pub name: Option<String>,
}

/// A node in the accessibility tree. `frame` is surface-local `(x, y, w, h)` in
/// the surface's own points (see [`SurfaceGeometry`]); `children` mirrors the AX
/// hierarchy.
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
/// Input synthesis is fire-and-forget everywhere: [`Impact::Done`] means the call
/// was delivered to the system without error — never that its effect happened.
/// [`Impact::DoneWith`] appends the one thing a delivered call still has to
/// disclose. [`Impact::Unconfirmed`] is the middle state: part or all of it may
/// have been applied and the platform cannot say which. It is a normal result, not
/// a failure — a failure reads as "nothing happened, do it again", and a repeated
/// click is a second actuation.
#[derive(Debug, Clone)]
pub(crate) enum Impact {
    Done,
    /// Delivered, with one thing the result must still say: a modifier the call
    /// could not release again, which may still be held down.
    #[cfg_attr(all(not(target_os = "linux"), not(test)), expect(dead_code))]
    DoneWith(String),
    /// Carries what is known: how much went through, and why it stopped.
    Unconfirmed(String),
}

/// Render an actuating call's result: a delivered action keeps its terse
/// confirmation plus any state it left behind, and an unconfirmed one leads with
/// that state instead of speaking of failure.
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

/// Keep at most `max` characters of one accessible string, its beginning preserved
/// and an ellipsis where it was cut. The flag reports the cut, so a read can state the
/// ceiling it hit; a string that fits comes back untouched.
fn cap_text(text: &str, max: usize) -> (String, bool) {
    let kept = crate::util::truncate(text, max);
    let cut = kept != text;
    (kept, cut)
}

/// A label read from the accessibility layer — an application name, a window title,
/// a role — kept to [`MAX_NAME_CHARS`], the ellipsis marking the cut. Labels are shown
/// rather than matched on (a window resolves through its key), so the cut is marked
/// and not counted.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn cap_label(text: &str) -> String {
    cap_text(text, MAX_NAME_CHARS).0
}

/// The pixel size of a buffer this tool may hold, from a figure the machine reported
/// — a window frame, a display frame, a decoded image. An empty area is an honest
/// failure (the surface is minimized, off-screen or gone) and a dimension a [`Capture`]
/// cannot express is refused here, so no caller hands a platform a size the buffer
/// cannot match. [`CaptureBudget::charge`] runs it too, so every claimed buffer has a
/// non-empty area by construction.
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

/// Where a tree read hit a ceiling. Every `None` means that ceiling was not hit;
/// [`render_tree`] states each one that was, in the header — the only part of a
/// large observation a spill preview guarantees the agent reads.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Truncation {
    /// Nodes beyond this node ceiling were not read (the walk stopped).
    pub nodes: Option<usize>,
    /// Children of this many nodes sit at the depth ceiling and were not read.
    pub depth: Option<Cut>,
    /// This many nodes advertised more children than the child ceiling allows, so
    /// their children were not read at all.
    pub children: Option<Cut>,
    /// This many nodes' child lists were longer than the child ceiling despite what
    /// they advertised, so they were cut there.
    pub children_cut: Option<Cut>,
    /// This many values were shortened at the read's own character cap.
    pub values: Option<Cut>,
    /// This many names were shortened at the read's own character cap.
    pub names: Option<Cut>,
}

/// How many items a ceiling cut, and the ceiling itself.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cut {
    pub count: usize,
    pub limit: usize,
}

/// The one pluralisation rule of the ceiling statements: `one` for a count of one, `many`
/// otherwise — nouns, verbs and pronouns alike, so none of them can read "1 nodes".
fn counted(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 { one } else { many }
}

impl Truncation {
    /// Record that the walk stopped at its node ceiling.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
        expect(dead_code)
    )]
    pub(crate) fn hit_node_ceiling(&mut self, limit: usize) {
        self.nodes = Some(limit);
    }

    /// Record one node whose children sit at the depth ceiling.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
        expect(dead_code)
    )]
    pub(crate) fn cut_depth(&mut self, limit: usize) {
        self.depth.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    /// Record one node whose advertised children exceed the child ceiling.
    #[cfg(target_os = "linux")]
    pub(crate) fn cut_children(&mut self, limit: usize) {
        self.children.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    /// Record one node whose child list was cut at the child ceiling.
    #[cfg(target_os = "linux")]
    pub(crate) fn trim_children(&mut self, limit: usize) {
        self.children_cut
            .get_or_insert(Cut { count: 0, limit })
            .count += 1;
    }

    /// Keep at most [`MAX_VALUE_CHARS`] of a value read from the accessibility layer,
    /// recording the read's cut in the header. The beginning is kept and the cut
    /// marked, so a document-shaped field still shows the text the model came for.
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

    /// Keep at most [`MAX_NAME_CHARS`] of a node's name, recording the cut in the
    /// header as a name cut.
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

    /// One line per ceiling that was hit, each naming it. `values_cut` comes from
    /// the renderer, which is where a value is cut for display — a read cap
    /// (`values`) and a display cap are different ceilings and both are stated.
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

/// A snapshot of one observe call against a target surface.
#[derive(Debug, Clone)]
pub(crate) struct Observation {
    pub app_name: String,
    pub window_title: Option<String>,
    pub surface: SurfaceGeometry,
    pub root: UiNode,
    /// Where the read stopped short, stated in the header.
    pub truncation: Truncation,
}

/// What one window listing produced. A listing never turns a failed read into an
/// absence: whatever is not listed is named beside the windows that were.
#[derive(Debug, Default)]
pub(crate) struct WindowListing {
    pub windows: Vec<WindowInfo>,
    pub unread: Vec<UnreadWindows>,
}

/// Part of a listing that is not listed, and why: per application, with the count of
/// windows a failed read dropped. The renderer words it without claiming anything
/// about the windows it did list.
#[derive(Debug)]
pub(crate) struct UnreadWindows {
    pub app_name: String,
    pub reason: String,
}

/// An application participating in the target registry.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AppInfo {
    pub pid: Option<u32>,
    pub name: String,
}

/// Backend-defined identity of one window surface. Opaque to the core: each
/// backend formats it so that resolving it lands on that very window or fails —
/// never on another window the same application happens to own.
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

/// A window of an application.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowInfo {
    pub app_pid: Option<u32>,
    pub app_name: String,
    pub title: String,
    /// Identity of this window (see [`WindowKey`]).
    pub key: WindowKey,
    pub surface: SurfaceGeometry,
}

/// Plain RGBA pixel data (Send-safe, no platform handles).
#[derive(Debug)]
pub(crate) struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// The pixels this buffer holds of its call's [`CaptureBudget`], given back when
    /// it is dropped.
    claim: Claim,
}

impl Capture {
    /// A blank `width`×`height` RGBA buffer. The size is charged before the buffer
    /// exists — a surface the call cannot hold is refused, never attempted — and the
    /// memory is reserved FALLIBLY, so a buffer this tool chose to make fails the call
    /// instead of aborting the process.
    ///
    /// The buffer comes back zeroed: a composite leaves the area no display covers
    /// black, and a platform that draws into the buffer needs an initialised
    /// `&mut [u8]` — safe Rust has no uninitialised one. A path that fills every byte
    /// itself ([`crop_rgba`], macOS's frame copy) takes its pixels over with
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

    /// Pixels a platform already produced, copied into a buffer charged to `budget`:
    /// the fallible reserve of [`Capture::alloc`] with the copy standing in for its
    /// zero-fill, so no byte is written twice.
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

    /// Pixels a platform already produced, taken over with the [`Claim`] their
    /// conversion was reserved with.
    pub(crate) fn adopt(claim: Claim, width: u32, height: u32, rgba: Vec<u8>) -> Self {
        debug_assert_eq!(rgba.len(), rgba_bytes(width, height));
        Self {
            width,
            height,
            rgba,
            claim,
        }
    }

    /// The pixels and the allowance they hold, separately: the claim has to live as
    /// long as the pixels do.
    pub(crate) fn into_parts(self) -> (u32, u32, Vec<u8>, Claim) {
        (self.width, self.height, self.rgba, self.claim)
    }
}

/// An empty byte buffer this tool chose to make, with `bytes` reserved FALLIBLY: a
/// failed allocation fails the call, naming the buffer, instead of aborting the
/// process.
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

/// One capture call's allowance of pixels, shared by every buffer the call holds at
/// once — the async caller and any platform thread it hands the capture to draw on
/// the same allowance. A [`Claim`] is weighed before its buffer exists and refused
/// without spending anything, and gives its pixels back when the buffer is dropped, so
/// the counter is what the call holds rather than what it has ever held. The call is
/// serialized (one action at a time, and a platform route is awaited), so claiming and
/// releasing need no stronger guarantee than that.
#[derive(Debug, Clone, Default)]
pub(crate) struct CaptureBudget(Arc<AtomicU64>);

impl CaptureBudget {
    /// Claim `width`×`height` pixels of RGBA for `what`, or refuse — naming the
    /// geometry, what the call already holds and [`CAPTURE_BUDGET_PIXELS`] — when the
    /// geometry is empty or the remaining allowance cannot cover it. Checked before
    /// the buffer exists, so a refused request costs nothing.
    pub(crate) fn charge(&self, what: &str, width: u32, height: u32) -> anyhow::Result<Claim> {
        // Empty geometry is refused by the one geometry gate, so no path can claim a
        // buffer of no pixels.
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

    /// What is left of this call's allowance, in pixels — what a decode about to
    /// materialise pixels can spend without breaking the ceiling.
    #[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))]
    pub(crate) fn remaining(&self) -> u64 {
        CAPTURE_BUDGET_PIXELS.saturating_sub(self.0.load(Ordering::Relaxed))
    }

    /// What the call holds right now, in pixels.
    #[cfg(test)]
    pub(crate) fn held(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// What one buffer holds of a [`CaptureBudget`] until the buffer is dropped.
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

/// Pixels as whole MiB, for the refusals — the figure the ceiling is stated in.
fn mib(pixels: u64) -> u64 {
    pixels.saturating_mul(4) / (1024 * 1024)
}

/// The surface a call addresses.
///
/// `Focused` is where global input goes and what a call with nothing observed
/// yet falls back to; `App` is an `a`-id — the application's own focused window,
/// re-resolved per call by design; `Window` is one exact window, identified by
/// the backend's own [`WindowKey`] so it can never land on another window.
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
        /// The window's title when it was enumerated — model-facing text only,
        /// never identity: a renamed window keeps its key and keeps resolving.
        label: String,
    },
    Screen,
}

impl TargetSpec {
    /// Convert a backend [`WindowInfo`] into a concrete window target. Used to
    /// pin an observed focused window so ref-resolved actions re-resolve THIS
    /// window and not whatever happens to be focused later (see `mod.rs`).
    #[must_use]
    pub(crate) fn from_window(win: &WindowInfo) -> Self {
        TargetSpec::Window {
            app_pid: win.app_pid,
            app_name: win.app_name.clone(),
            key: win.key.clone(),
            label: win.title.clone(),
        }
    }

    /// The application this target belongs to: the backend resolves it before
    /// the window (`None` for a target that names no application).
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

    /// Model-facing description of the surface, for results and for errors about
    /// a target that does not match a ref.
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

/// How a window surface is named in results and errors — the one spelling, so a
/// backend that holds a resolved window names it exactly as its target would. An
/// empty title is an untitled window, never an empty pair of quotes.
#[must_use]
pub(crate) fn window_description(app_name: &str, title: &str) -> String {
    match (app_name.is_empty(), title.is_empty()) {
        (false, false) => format!("window \"{title}\" of {app_name}"),
        (false, true) => format!("a window of {app_name}"),
        (true, false) => format!("window \"{title}\""),
        (true, true) => "the window".to_string(),
    }
}

/// The definition of surface identity: two specs are equal when they name the same
/// surface, and the registry hands the same id to equal specs.
///
/// An `App` is its application; a `Window` is its backend key alone, because that
/// key already carries the application and is the only thing every backend
/// resolves through. The pid, app name and label are snapshots for messages, so a
/// renamed window (or one whose process-reported name changes) keeps its identity
/// instead of being handed a second id for the same surface.
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

/// An action applied to a matched element.
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

/// A raw input event whose points are normalized 0-1000 relative to the target
/// surface. The backend resolves geometry (via [`normalized_to_surface`]) before
/// converting to platform-specific coordinates.
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

/// Whether `input` on `target` has to reach a named window: typed text, a drag
/// gesture and a wheel aimed at a point, and only when the call names a window (a
/// `w` id or an `a` id) rather than the focused window or the screen.
///
/// Everything else is global input — a key chord, a wheel with no point and a
/// click at a point go where the system sends them — and no result may report it
/// as having acted on the observed surface. The one definition of that rule, so
/// the three backends cannot drift apart.
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

/// Pause between two raise checks.
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), expect(dead_code))]
pub(crate) const RAISE_SETTLE_MS: u64 = 25;

// ── Key chord parsing ───────────────────────────────────────────────────

/// Parse a key chord string ("cmd+shift+t", "return", "ctrl+c") into its
/// modifiers and a single key name (lowercased). Pure and platform-agnostic —
/// the backend maps the key name to a platform virtual keycode.
///
/// Recognized modifiers (case-insensitive): `cmd`/`command`, `ctrl`/`control`,
/// `alt`/`option`, `shift`. The non-modifier segment is the key name and is
/// returned lowercased (e.g. `"return"`, `"a"`, `"f5"`, `"5"`).
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

// ── Argument bounds ─────────────────────────────────────────────────────
//
// The tool checks every argument of an action here before it takes the action lock, and
// each backend checks the bounds of the input it was handed at its own entry, so a bad
// call costs a round trip rather than a lock: an argument outside its stated range is
// refused with the range, and text past its cap with the count and the cap (never with
// the rejected value, which would repeat the oversized payload back).

/// Refuse a string argument longer than [`MAX_ARG_CHARS`], naming its length and the cap.
pub(crate) fn check_arg_len(field: &str, value: &str) -> anyhow::Result<()> {
    let len = value.chars().count();
    if len > MAX_ARG_CHARS {
        anyhow::bail!("{field} is too long: {len} chars (max {MAX_ARG_CHARS})");
    }
    Ok(())
}

/// Refuse a scroll magnitude outside `1..=MAX_SCROLL_AMOUNT`.
pub(crate) fn check_scroll_amount(amount: u32) -> anyhow::Result<()> {
    if !(1..=MAX_SCROLL_AMOUNT).contains(&amount) {
        anyhow::bail!("scroll amount must be 1-{MAX_SCROLL_AMOUNT} ticks — got {amount}");
    }
    Ok(())
}

/// Refuse a normalized point outside 0-1000, naming the argument it came from.
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

/// Refuse a region whose corners are outside 0-1000 or that encloses no area. Reversed
/// corners name the same region (the corners are ordered where the region is mapped),
/// so only a zero-extent one is refused — it would otherwise become a single pixel
/// standing for an area the call never named.
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

/// The bounds a backend's input entry checks before it starts any work — every point,
/// the scroll magnitude and the typed text — so no value it cannot honour reaches a
/// platform call. A chord's length is bounded upstream with the action's own arguments;
/// its syntax is the backend's to refuse.
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

// ── Coordinate math ─────────────────────────────────────────────────────

/// Map a normalized 0-1000 point to absolute global surface points on the target
/// surface.
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

/// Map an absolute global surface point to normalized 0-1000 coordinates against
/// the target surface, clamped to the surface extent.
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

/// True when an absolute global surface point lies inside the surface extent.
/// Top-left corner is inclusive; the bottom-right edge is exclusive.
#[must_use]
pub(crate) fn contains_point(geo: &SurfaceGeometry, ax: f64, ay: f64) -> bool {
    ax >= geo.x && ax < geo.x + geo.width && ay >= geo.y && ay < geo.y + geo.height
}

/// Map a normalized region to an integer pixel rect `(x, y, w, h)` of the
/// captured image (corners ordered, clamped to the image bounds). The image's own
/// size is the space the region is mapped into, so a crop can never be taken
/// against a surface other than the one captured.
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

/// Crop a rect `(x, y, w, h)` out of a capture, clamping to the image bounds.
/// `what` names the crop in a refusal, and the copy is charged to the call — a
/// zoom crop is as large as the surface it came from. The rows are copied straight
/// into the claimed buffer, so the crop costs one copy and no zero-fill.
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

/// Place `src` RGBA pixels at `(dx, dy)` inside a `dest_w`×`dest_h` buffer,
/// clipping to the destination bounds. Platform-free; shared by the capture
/// composite backends.
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

/// Scale between an accessibility coordinate space and captured pixels.
///
/// `a11y_screen` is the whole screen as the accessibility stack reports it and
/// `pixels` is a full-screen capture's size. `None` when the two cannot be
/// reconciled — non-positive extents, axes that disagree, or a space larger
/// than the capture (a scale this tool cannot trust) — so a caller can refuse
/// instead of cropping the wrong part of the screen. Only extents are compared:
/// an origin offset between the two spaces is not detectable without a second
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
    // Rounding of odd extents makes the two axes differ by a pixel; a real
    // disagreement is a space this cannot map.
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

/// ZPixmap tile (`depth` 24/32, 4 bytes/pixel) → one RGBA capture, charged to the
/// call like every other buffer.
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

/// `file://` portal URI → path, percent-decoding `%XX` (`+` kept verbatim).
/// After `file://`, `[host]/path` with an empty host (`file:///...`) or
/// exactly `localhost` is accepted; anything else is rejected.
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

/// Window extents → pixel rect inside a screen-channel capture: true rect
/// intersection in buffer space (buffer origin = screen origin, size =
/// screen size). `crop_rgba` clamps the far edge, so w/h are rounded up to
/// `>= 1` (mirror `region_to_pixels` style).
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

// ── Element-ref lifecycle ───────────────────────────────────────────────

/// Cached observation with the refs resolved against it.
struct ObservationCache {
    target: TargetSpec,
    refs: HashMap<String, Locator>,
}

static LAST_OBSERVATION: LazyLock<Mutex<HashMap<String, ObservationCache>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The surface each agent last observed or captured (see [`last_surface`]).
static LAST_SURFACE: LazyLock<Mutex<HashMap<String, TargetSpec>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The per-agent registry key, derived from the tool-context agent id. Task-
/// locals are alive throughout a tool-call task (read synchronously here; no
/// spawns cross this boundary). Outside an agent run (management, tests) the key
/// falls back to the diagnostics role so those callers share one bucket.
#[must_use]
pub(crate) fn agent_key() -> String {
    crate::agent::CURRENT_TOOL_AGENT_ID
        .try_with(Clone::clone)
        .unwrap_or(None)
        .unwrap_or_else(|| crate::agent::role::DIAGNOSTICS_ROLE.to_string())
}

/// Walk the tree in DFS pre-order assigning refs `e1`, `e2`, … and replace the
/// whole cache for the calling agent, returning the element count.
pub(crate) fn store_observation(target: TargetSpec, obs: &Observation) -> usize {
    let mut refs = HashMap::new();
    let mut counter = 0usize;
    let mut path = Vec::new();
    assign_refs(&obs.root, &mut path, &mut refs, &mut counter);
    let cache = ObservationCache { target, refs };
    let key = agent_key();
    let mut guard = LAST_OBSERVATION.lock().unwrap_poison();
    guard.insert(key, cache);
    counter
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
        },
    );
    for (idx, child) in node.children.iter().enumerate() {
        path.push(idx);
        assign_refs(child, path, refs, counter);
        path.pop();
    }
}

/// Resolve a ref like `e3` to the locator/target stored at the calling agent's
/// last observe.
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

/// The target pinned by the calling agent's most recent observation or capture.
/// A call with no explicit target addresses it, so a focus shift between
/// observe and the action never retargets a different surface. `None` means
/// nothing has been pinned yet.
#[must_use]
pub(crate) fn last_surface() -> Option<TargetSpec> {
    let key = agent_key();
    let guard = LAST_SURFACE.lock().unwrap_poison();
    guard.get(&key).cloned()
}

/// Pin the surface the calling agent last observed or captured. Only a call that
/// succeeded pins: a failed observation leaves the previous surface in place.
pub(crate) fn set_last_surface(target: TargetSpec) {
    let key = agent_key();
    let mut guard = LAST_SURFACE.lock().unwrap_poison();
    guard.insert(key, target);
}

// ── Target registry ─────────────────────────────────────────────────────

/// The two id classes an enumeration hands out: `a` ids address applications,
/// `w` ids windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdClass {
    App,
    Window,
}

impl IdClass {
    fn prefix(self) -> char {
        match self {
            Self::App => 'a',
            Self::Window => 'w',
        }
    }
}

/// Most ids kept per agent, and — through the macOS backend — most window handles
/// kept per agent, so an id in the registry always has a handle to resolve
/// through. A long run that enumerates a churning desktop drops its oldest
/// entries; a dropped id then reads as unknown, which is a safe error.
pub(crate) const MAX_STORED_TARGETS: usize = 512;

/// One agent's ids: the surfaces they name, and the counters that mint them.
#[derive(Default)]
struct TargetRegistry {
    /// `(id, surface)` in the order minted.
    entries: Vec<(String, TargetSpec)>,
    /// Counters are never rewound, so an id is never handed to another surface.
    next_app: usize,
    next_window: usize,
}

impl TargetRegistry {
    fn mint(&mut self, class: IdClass) -> String {
        let counter = match class {
            IdClass::App => &mut self.next_app,
            IdClass::Window => &mut self.next_window,
        };
        *counter += 1;
        format!("{}{counter}", class.prefix())
    }

    fn position_of(&self, surface: &TargetSpec) -> Option<usize> {
        self.entries
            .iter()
            .position(|(_, stored)| stored == surface)
    }

    fn insert(&mut self, id: String, surface: TargetSpec) {
        self.entries.push((id, surface));
        if self.entries.len() > MAX_STORED_TARGETS {
            self.entries.remove(0);
        }
    }
}

static TARGETS: LazyLock<Mutex<HashMap<String, TargetRegistry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Store one enumeration's applications and return their `a` ids, in the same
/// order.
pub(crate) fn store_app_targets(apps: &[TargetSpec]) -> Vec<String> {
    store_targets(IdClass::App, apps)
}

/// Store one enumeration's windows and return their `w` ids, in the same order.
pub(crate) fn store_window_targets(windows: &[TargetSpec]) -> Vec<String> {
    store_targets(IdClass::Window, windows)
}

/// Store one enumeration's surfaces and return their ids, in the same order. A
/// surface already enumerated keeps the id it was given — re-enumerating refreshes
/// its message-only snapshot but not its identity — while a new surface gets a
/// fresh id; an id of a surface this enumeration did not include is kept, so it
/// still names that surface (and the backend reports it gone) rather than silently
/// naming whatever took its place.
fn store_targets(class: IdClass, surfaces: &[TargetSpec]) -> Vec<String> {
    let key = agent_key();
    let mut guard = TARGETS.lock().unwrap_poison();
    let registry = guard.entry(key).or_default();
    let mut ids = Vec::with_capacity(surfaces.len());
    for surface in surfaces {
        if let Some(position) = registry.position_of(surface) {
            // Same surface: keep its id, refresh what messages read from it.
            let (id, stored) = &mut registry.entries[position];
            ids.push(id.clone());
            *stored = surface.clone();
            continue;
        }
        let id = registry.mint(class);
        registry.insert(id.clone(), surface.clone());
        ids.push(id);
    }
    ids
}

/// Look up a stored target id without erroring (used when an argument may or
/// may not be a target id).
pub(crate) fn get_target(id: &str) -> Option<TargetSpec> {
    let key = agent_key();
    let guard = TARGETS.lock().unwrap_poison();
    let registry = guard.get(&key)?;
    registry
        .entries
        .iter()
        .find(|(stored, _)| stored == id)
        .map(|(_, surface)| surface.clone())
}

/// Resolve a target id or the `"screen"` literal. `None` → [`TargetSpec::Focused`].
pub(crate) fn resolve_target(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    let Some(target) = target else {
        return Ok(TargetSpec::Focused);
    };
    if target == "screen" {
        return Ok(TargetSpec::Screen);
    }
    get_target(target).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown target '{target}' — it is not a stored app/window id; enumerate targets \
             with 'apps' (a1-style) or 'windows' (w1-style) first"
        )
    })
}

/// Remove an agent's cached observation, pinned surface and target entries.
/// Called at agent-run end so a stale per-agent registry never leaks across runs
/// (see [`crate::tools::computer::cleanup_agent_state`]). No-op for unknown keys.
pub(crate) fn clear_agent_state(key: &str) {
    {
        let mut guard = LAST_OBSERVATION.lock().unwrap_poison();
        guard.remove(key);
    }
    {
        let mut guard = LAST_SURFACE.lock().unwrap_poison();
        guard.remove(key);
    }
    let mut guard = TARGETS.lock().unwrap_poison();
    guard.remove(key);
}

// ── Tree rendering ──────────────────────────────────────────────────────

/// Strip a leading `AX` from an accessibility role (display + matching).
fn normalized_role(role: &str) -> &str {
    role.strip_prefix("AX").unwrap_or(role)
}

/// Whether a node is interactive: non-empty actions or an actionable role.
fn is_interactive(node: &UiNode) -> bool {
    if !node.actions.is_empty() {
        return true;
    }
    let role = normalized_role(&node.role);
    INTERACTIVE_ROLES
        .iter()
        .any(|r| r.eq_ignore_ascii_case(role))
}

/// Number of interactive nodes in the whole tree.
#[must_use]
pub(crate) fn count_interactive(obs: &Observation) -> usize {
    count_stats(&obs.root).1
}

/// (total nodes, interactive nodes) across the whole tree, in one pre-order
/// walk. Used by [`render_tree`] for its header counts and by
/// [`count_interactive`] so the walk is single-sourced.
fn count_stats(node: &UiNode) -> (usize, usize) {
    let mut total = 1usize;
    let mut interactive = usize::from(is_interactive(node));
    for child in &node.children {
        let (t, i) = count_stats(child);
        total += t;
        interactive += i;
    }
    (total, interactive)
}

/// True when the surface exposes very few actionable elements — the signal to
/// switch from the accessibility tree to screenshots.
#[must_use]
pub(crate) fn is_ax_thin(obs: &Observation) -> bool {
    count_interactive(obs) < AX_THIN_INTERACTIVE_ELEMENTS
}

/// Render the observation as an indented element tree with pre-order refs. The
/// header carries the node/interactive totals (computed in one pre-walk, so they
/// describe everything that was read) and states every ceiling the read hit — a
/// tree that was cut short must never read as complete.
#[must_use]
pub(crate) fn render_tree(obs: &Observation) -> String {
    let (node_count, interactive_count) = count_stats(&obs.root);
    let mut tree = String::new();
    let mut counter = 0usize;
    let mut values_cut = 0usize;
    render_node(&obs.root, 0, &mut tree, &mut counter, &mut values_cut);

    let surf = &obs.surface;
    let title = obs.window_title.as_deref().unwrap_or("");
    let mut out = String::new();
    let _ = writeln!(
        out,
        "[Surface: app=\"{}\" window=\"{}\" {}x{} @ ({}, {}) — {} nodes, {} interactive]",
        obs.app_name, title, surf.width, surf.height, surf.x, surf.y, node_count, interactive_count
    );
    for statement in obs.truncation.statements(values_cut) {
        out.push_str(&statement);
        out.push('\n');
    }
    out.push_str(&tree);
    out
}

fn render_node(
    node: &UiNode,
    depth: usize,
    out: &mut String,
    counter: &mut usize,
    values_cut: &mut usize,
) {
    *counter += 1;
    let indent = "  ".repeat(depth);
    let mut line = format!("{indent}[e{counter}] {}", normalized_role(&node.role));
    if let Some(name) = &node.name {
        let _ = write!(line, " \"{name}\"");
    }
    if !node.actions.is_empty() {
        let _ = write!(line, " actions=[{}]", node.actions.join(","));
    }
    if let Some((x, y, _, _)) = node.frame {
        let _ = write!(line, " @ ({x}, {y})");
    }
    if let Some(value) = &node.value {
        let v = crate::util::truncate(value, RENDER_VALUE_CHARS);
        if v != *value {
            *values_cut += 1;
        }
        let _ = write!(line, " value=\"{v}\"");
    }
    if node.focused {
        line.push_str(" focused");
    }
    out.push_str(&line);
    out.push('\n');
    for child in &node.children {
        render_node(child, depth + 1, out, counter, values_cut);
    }
}

/// Pre-order DFS index of `target` in `root` — maps a matched node back to its
/// handle slot (the handles are kept in the same pre-order as the tree).
/// Platform-free; shared by the backends in `act_on_element`.
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

/// Outcome of resolving a locator against a tree.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) enum LocatorMatch<'a> {
    /// The recorded child-index path reached a node matching the role+name —
    /// the aligned element (the path is load-bearing even when the search would
    /// be ambiguous).
    Path(&'a UiNode),
    /// Exactly one role+name match in the whole tree.
    Unique(&'a UiNode),
    /// More than one role+name match and the path did not pin one down.
    Ambiguous,
    /// No node matches the role+name.
    NotFound,
}

/// Resolve a locator against a tree. The recorded child-index path is followed
/// first; if it reaches a node matching the role+name (same matching rules as a
/// search) it wins even when the search would match several nodes. Otherwise the
/// tree is searched for role+name matches: exactly one is unique, more than one
/// is ambiguous, and none is not-found.
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

/// Resolve a locator, mapping the ambiguous/not-found outcomes to their
/// taxonomy errors — the shared failure contract of every backend's
/// `act_on_element`.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn resolve_locator_checked<'a>(
    root: &'a UiNode,
    locator: &Locator,
) -> Result<&'a UiNode, anyhow::Error> {
    match resolve_locator(root, locator) {
        LocatorMatch::Path(node) | LocatorMatch::Unique(node) => Ok(node),
        LocatorMatch::Ambiguous => Err(taxonomy_error(
            ERR_AMBIGUOUS_LOCATOR,
            "locator matches multiple elements — re-observe and pick a more specific ref",
        )),
        LocatorMatch::NotFound => Err(taxonomy_error(
            ERR_NOT_MATCHED,
            "element no longer matches its locator — re-observe",
        )),
    }
}

/// Shared refusal for acting on the `screen` target: acting addresses an element
/// of a window. Linux and macOS pre-check it; Windows lets its own target
/// resolution raise it, so the wording is the same everywhere.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows")),
    expect(dead_code)
)]
pub(crate) fn screen_act_error() -> anyhow::Error {
    taxonomy_error(
        ERR_UNSUPPORTED,
        "act targets a window — use apps/windows to pick one (use screenshot for the screen)",
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
        // Reverse push keeps the traversal in DFS pre-order.
        for child in node.children.iter().rev() {
            stack.push(child);
        }
    }
    match found {
        Some(node) => LocatorMatch::Unique(node),
        None => LocatorMatch::NotFound,
    }
}

// ── Backend trait ───────────────────────────────────────────────────────

#[async_trait::async_trait]
pub(crate) trait Backend: Send + Sync {
    /// Whether the platform can serve the tool at all — the advertisement gate,
    /// not a grant check: macOS always can (its grants are requested per call),
    /// Windows always can (no grant exists; the refusals are per window and per
    /// call), Linux needs a GUI session with a reachable session bus.
    fn is_supported(&self) -> bool;
    /// Why this session cannot receive an interactive call — the state the owner has to
    /// change, named in the error the model reads — or `None` when it can. Implementers
    /// answer only from a signal the platform actually publishes: `None` is the answer for
    /// a state the platform cannot tell, because a false refusal on a working session is
    /// worse than the failure the call would report on its own.
    async fn desktop_unavailable(&self) -> Option<String>;
    /// Prompt-free capture-channel probe (the screen-recording grant on macOS);
    /// the caller caches it with a TTL.
    async fn capture_available(&self) -> bool;
    /// Actionable diagnosis for an unavailable capture channel (missing grant,
    /// or an OS too old for stills).
    fn capture_unavailable_error(&self) -> anyhow::Error;
    async fn list_apps(&self) -> anyhow::Result<Vec<AppInfo>>;
    /// Windows of one application (or of every application). A failed read of one
    /// application is reported in [`WindowListing::unread`], never as an empty list.
    async fn list_windows(&self, app: Option<&AppInfo>) -> anyhow::Result<WindowListing>;
    /// Focused window of the frontmost application (the DEFAULT surface).
    async fn focused_window(&self) -> anyhow::Result<WindowInfo>;
    /// Surface geometry for a target (Screen: virtual-desktop bounding rect).
    async fn surface_geometry(&self, target: &TargetSpec) -> anyhow::Result<SurfaceGeometry>;
    /// AX tree of a window target (Screen target: `unsupported` error).
    async fn observe(&self, target: &TargetSpec) -> anyhow::Result<Observation>;
    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> anyhow::Result<Impact>;
    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> anyhow::Result<Impact>;
    /// Pointer position in ABSOLUTE global surface points (logical points on
    /// macOS, physical pixels on Linux and Windows).
    async fn cursor_position(&self) -> anyhow::Result<(f64, f64)>;
    /// RGBA pixels of a target surface (window capture, or full virtual desktop for
    /// Screen). Every buffer this produces is charged to `budget` — the call's one
    /// allowance, so a surface the machine reports as enormous is refused instead
    /// of allocated.
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
        // pre-order refs: e1 window, e2 OK button, e3 group, e4 Cancel button
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

    #[test]
    fn normalized_to_surface_maps_and_validates() {
        let geo = surface(100.0, 200.0, 800.0, 600.0);
        let (sx, sy) = normalized_to_surface(500.0, 250.0, &geo).unwrap();
        assert!((sx - 500.0).abs() < 1e-9, "x = {sx}");
        assert!((sy - 350.0).abs() < 1e-9, "y = {sy}");
        // Out of bounds (finite but negative / over max) and NaN.
        assert!(normalized_to_surface(-1.0, 0.0, &geo).is_err());
        assert!(normalized_to_surface(0.0, 1001.0, &geo).is_err());
        assert!(normalized_to_surface(f64::NAN, 0.0, &geo).is_err());
    }

    #[test]
    fn normalized_to_surface_handles_negative_origins() {
        // A virtual-desktop bounding rect spanning a display with a negative origin.
        let geo = surface(0.0, -100.0, 3840.0, 1180.0);
        let (px, py) = normalized_to_surface(500.0, 500.0, &geo).unwrap();
        assert!((px - 1920.0).abs() < 1e-9, "x = {px}");
        assert!((py - 490.0).abs() < 1e-9, "y = {py}");
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

    #[test]
    fn region_to_pixels_maps_orders_and_clamps() {
        // Normalized 200..=800 on a 1000px image → 200..=800 px.
        let region = Region {
            x0: 200.0,
            y0: 100.0,
            x1: 800.0,
            y1: 500.0,
        };
        assert_eq!(
            region_to_pixels(&region, 1000, 1000).unwrap(),
            (200, 100, 600, 400)
        );
        // Reversed corners are normalized in order → same rect.
        let rev = Region {
            x0: 800.0,
            y0: 500.0,
            x1: 200.0,
            y1: 100.0,
        };
        assert_eq!(
            region_to_pixels(&rev, 1000, 1000).unwrap(),
            (200, 100, 600, 400)
        );
        // Out-of-range normalized coords are rejected.
        let bad = Region {
            x0: 1100.0,
            y0: 0.0,
            x1: 200.0,
            y1: 100.0,
        };
        assert!(region_to_pixels(&bad, 1000, 1000).is_err());
        // A region enclosing no area is refused, never turned into one pixel.
        let flat = Region {
            x0: 100.0,
            y0: 100.0,
            x1: 900.0,
            y1: 100.0,
        };
        assert!(region_to_pixels(&flat, 1000, 1000).is_err());
        // A region object carries only its corners: an unknown key is a mistake, not
        // a value to ignore.
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
        // The array form is four numbers: a fifth names nothing and is refused too.
        assert!(serde_json::from_value::<Region>(serde_json::json!([0, 0, 1, 1, 2])).is_err());
        assert!(serde_json::from_value::<Region>(serde_json::json!([0, 0, 1, 1])).is_ok());
        // A region beyond the image is clamped to the image bounds (>=1px).
        let big = Region {
            x0: 500.0,
            y0: 500.0,
            x1: 1000.0,
            y1: 1000.0,
        };
        let (x, y, w, h) = region_to_pixels(&big, 200, 200).unwrap();
        assert!(x + w <= 200 && y + h <= 200);
        assert!(w >= 1 && h >= 1);
    }

    #[test]
    fn crop_rgba_extracts_sub_rect() {
        let budget = CaptureBudget::default();
        // 4x2 image: rows are constant-color for easy verification.
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
        // The crop is a buffer of its own, claimed like every other.
        assert_eq!(budget.held(), 4 * 2 + 2 * 2);
        // First cropped pixel = original row0 col1 = [0, 10, 0, 255].
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

    #[test]
    fn window_rect_intersects_screen() {
        let screen = surface(-1920.0, 0.0, 3840.0, 1080.0);
        // Negative-origin offset into buffer space.
        let win = surface(-1920.0, 100.0, 800.0, 600.0);
        assert_eq!(
            window_capture_rect(&win, &screen).unwrap(),
            (0, 100, 800, 600)
        );
        // Partially off-screen shrinks dims to the visible intersection.
        let partial = surface(1500.0, 900.0, 800.0, 400.0);
        assert_eq!(
            window_capture_rect(&partial, &screen).unwrap(),
            (3420, 900, 420, 180)
        );
        // Fully off-screen errors instead of cropping a misleading 1x1.
        let off = surface(5000.0, 0.0, 100.0, 100.0);
        let err = window_capture_rect(&off, &screen).unwrap_err();
        assert!(
            err.to_string().contains("outside the captured screen"),
            "{err:?}"
        );
        assert!(window_capture_rect(&surface(0.0, 0.0, 0.0, 0.0), &screen).is_err());
    }

    #[test]
    fn render_tree_assigns_pre_order_refs_and_flags() {
        let obs = sample_observation();
        let text = render_tree(&obs);
        let lines: Vec<&str> = text.lines().collect();
        assert!(
            lines[0].starts_with(
                "[Surface: app=\"TestApp\" window=\"Sample\" 800x600 @ (0, 0) — 4 nodes, 2 interactive]"
            ),
            "got {}",
            lines[0]
        );
        // e1 = root window (AX stripped to Window), e2 = OK button (has action),
        // e3 = group, e4 = Cancel button.
        assert!(
            lines[1].starts_with("[e1] Window \"Sample\""),
            "got {}",
            lines[1]
        );
        assert!(
            lines[2].starts_with("  [e2] Button \"OK\""),
            "got {}",
            lines[2]
        );
        assert!(lines[2].contains("actions=[press]"), "got {}", lines[2]);
        assert!(lines[3].starts_with("  [e3] Group"), "got {}", lines[3]);
        assert!(
            lines[4].starts_with("    [e4] Button \"Cancel\""),
            "got {}",
            lines[4]
        );
        // count_interactive counts the two buttons (actions) and treats root/group as non-interactive.
        assert_eq!(count_interactive(&obs), 2);
        // 2 < AX_THIN_INTERACTIVE_ELEMENTS → the surface is AX-thin (screenshot signal).
        assert!(is_ax_thin(&obs));
    }

    #[test]
    fn render_tree_states_every_ceiling_it_hit() {
        let long = "x".repeat(200);
        let root = UiNode {
            value: Some(long.clone()),
            ..node("AXTextField", Some("field"))
        };
        let obs = Observation {
            app_name: "A".to_string(),
            window_title: None,
            surface: surface(0.0, 0.0, 10.0, 10.0),
            root,
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
        };
        let text = render_tree(&obs);
        assert!(text.contains("value=\""), "value should render");
        assert!(!text.contains(&long), "long value must be truncated");
        // Every cut is stated in the header, above the first element line: a tree
        // near a ceiling reaches the agent as a preview of its head, so a statement
        // after the tree would never be read.
        let header_end = text
            .lines()
            .position(|line| line.starts_with("[e"))
            .expect("the tree renders under the header");
        let header = text.lines().take(header_end).collect::<Vec<_>>().join("\n");
        assert!(
            header.contains(&format!("the {MAX_TREE_NODES}-node ceiling")),
            "{header}"
        );
        assert!(
            header.contains("2 nodes at the 25-deep ceiling"),
            "{header}"
        );
        assert!(
            header.contains(&format!("1 value cut at {RENDER_VALUE_CHARS} chars")),
            "{header}"
        );
        // The read's own value cap is a separate ceiling from the display cap.
        assert!(
            header.contains(&format!("1 value cut at {MAX_VALUE_CHARS} chars")),
            "{header}"
        );
        assert!(
            header.contains(&format!("3 names cut at {MAX_NAME_CHARS} chars")),
            "{header}"
        );
        assert!(
            header.contains(&format!("4 nodes claim more than {MAX_CHILDREN} children")),
            "{header}"
        );
        assert!(
            header.contains(&format!(
                "1 node listed more than {MAX_CHILDREN} children — only the first {MAX_CHILDREN} \
                 were read"
            )),
            "{header}"
        );
    }

    #[test]
    fn a_single_node_at_a_ceiling_reads_as_one_node() {
        let mut truncation = Truncation::default();
        truncation.cut_depth(25);
        assert_eq!(
            truncation.statements(0),
            ["[Tree truncated: 1 node at the 25-deep ceiling — its children were not read]"]
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

        // A label is cut at the same cap and only marked: labels are shown, not
        // matched on, so there is no header to count it in.
        let label = cap_label(&"l".repeat(MAX_NAME_CHARS + 1));
        assert_eq!(label.chars().count(), MAX_NAME_CHARS);
        assert!(label.ends_with('…'), "{label}");
    }

    #[test]
    fn capture_size_refuses_what_a_capture_cannot_express() {
        assert_eq!(
            capture_size("a window still", 800, 600).unwrap(),
            (800, 600)
        );
        assert!(capture_size("a window still", 0, 600).is_err());
        let err = capture_size("a display frame", usize::MAX, 1).unwrap_err();
        assert!(err.to_string().starts_with(ERR_DEGRADED), "{err}");
    }

    #[test]
    fn capture_budget_refuses_a_surface_it_cannot_hold() {
        let budget = CaptureBudget::default();
        // A wildly misreported display: the area a u32 rect can name is far past the
        // call's ceiling, so it is refused before anything is allocated.
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
        // Nothing was charged by the refusal: an ordinary surface still fits.
        assert!(budget.charge("a display frame", 3840, 2160).is_ok());

        // Empty geometry is refused rather than allocated as nothing — by the one
        // geometry gate, whose message says what the surface state is.
        let empty = budget.charge("a window still", 0, 1080).unwrap_err();
        assert!(empty.to_string().contains("no capture area"), "{empty}");

        // The ceiling refusal is TYPED, so a caller can tell it from every other
        // degraded failure without reading the message.
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
        // Each of these fits the ceiling alone; together they do not, and that is the
        // point of a per-call budget over a per-buffer one. Claims carry the accounting
        // on their own — no buffer is allocated here.
        let half = u32::try_from(CAPTURE_BUDGET_PIXELS / 2).unwrap();
        let composite = budget.charge("the composite", half, 1).unwrap();
        let frame = budget.charge("a display frame", half, 1).unwrap();
        assert_eq!(budget.held(), CAPTURE_BUDGET_PIXELS);
        // A full-size crop of the surface cannot join them, and the refusal names what
        // the call is already holding — the figure the model can act on.
        let err = budget.charge("the zoom crop", half, 1).unwrap_err();
        assert!(err.to_string().contains("the zoom crop"), "{err}");
        assert!(err.to_string().contains("already holds 512 MiB"), "{err}");

        // A refused claim spends nothing, and a dropped claim gives its pixels back: a
        // call that blits one display at a time is not charged for every display it has
        // ever held.
        drop(frame);
        assert_eq!(budget.held(), u64::from(half));
        drop(composite);
        assert_eq!(budget.held(), 0);
        assert!(budget.charge("the zoom crop", half, 1).is_ok());

        // Pixels a platform produced itself are taken over with the claim their
        // conversion was reserved with, and what is left of the allowance follows it.
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
    fn impact_text_states_the_middle_outcome_without_advice_to_retry() {
        assert_eq!(impact_text("clicked".to_string(), Impact::Done), "clicked");
        assert_eq!(
            impact_text(
                "clicked".to_string(),
                Impact::DoneWith("ctrl may still be held down".to_string()),
            ),
            "clicked — ctrl may still be held down"
        );
        let text = impact_text(
            "clicked element e3".to_string(),
            Impact::Unconfirmed("the call was abandoned".to_string()),
        );
        assert!(text.starts_with("applied but unconfirmed"), "{text}");
        assert!(text.contains("may already have taken effect"), "{text}");
        assert!(text.contains("act twice"), "{text}");
        assert!(!text.contains("retry"), "{text}");
    }

    #[test]
    fn store_and_resolve_ref_round_trip_then_stale() {
        let obs = sample_observation();
        let target = TargetSpec::Focused;
        let count = store_observation(target.clone(), &obs);

        let (locator, resolved_target) = resolve_ref("e2").unwrap();
        assert_eq!(locator.role, "AXButton");
        assert_eq!(locator.name.as_deref(), Some("OK"));
        assert_eq!(locator.path, vec![0]);
        assert_eq!(resolved_target, TargetSpec::Focused);
        assert_eq!(count, 4);

        // A fresh observe replaces the cache; the old ref goes stale.
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
        // Garbage ref → stale-element too.
        let err = resolve_ref("zzz").unwrap_err();
        assert!(format!("{err}").contains("stale-element"));
    }

    // The registry tests share the process-global map, so one seeding it under
    // the other's assertions would flake; serialize them.
    #[serial_test::serial(targets)]
    #[test]
    fn resolve_target_resolves_focused_screen_stored_and_unknown() {
        assert_eq!(resolve_target(None).unwrap(), TargetSpec::Focused);
        assert_eq!(resolve_target(Some("screen")).unwrap(), TargetSpec::Screen);

        let app = TargetSpec::App {
            app_pid: Some(42),
            app_name: "Safari".to_string(),
        };
        let ids = store_app_targets(std::slice::from_ref(&app));
        assert_eq!(ids.len(), 1);
        assert_eq!(resolve_target(Some(&ids[0])).unwrap(), app);
        let err = resolve_target(Some("w9999")).unwrap_err();
        assert!(format!("{err}").contains("unknown target"));
    }

    #[serial_test::serial(targets)]
    #[test]
    fn store_targets_keeps_ids_for_seen_surfaces_and_never_reuses_them() {
        let window = |name: &str, key: &str| TargetSpec::Window {
            app_pid: Some(7),
            app_name: name.to_string(),
            key: WindowKey::new(key),
            label: format!("{key} title"),
        };
        let first = store_window_targets(&[window("Beta", "k1"), window("Beta", "k2")]);
        // The same two windows, re-enumerated in the other order: the ids follow
        // the surfaces, not their position in the listing.
        let again = store_window_targets(&[window("Beta", "k2"), window("Beta", "k1")]);
        assert_eq!(again, vec![first[1].clone(), first[0].clone()]);

        // A vanished window keeps its id pointing at itself, and the id of the
        // window that took its place is a new one.
        let after = store_window_targets(&[window("Beta", "k1"), window("Beta", "k3")]);
        assert_eq!(after[0], first[0]);
        assert_ne!(after[1], first[1]);
        assert_eq!(
            resolve_target(Some(&first[1])).unwrap(),
            window("Beta", "k2"),
            "the id of a vanished window must not name its replacement"
        );

        // A rename keeps the key, so the id still names the same window — and the
        // message-only label follows the rename. Identity is the key alone: the app
        // name a backend reports can change with the title (a frame-hosted window on
        // Windows is named by it) and must not mint a second id for that window.
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
        // A space larger than the capture, per-axis disagreement, and empty
        // extents are all refusals rather than guesses.
        assert_eq!(a11y_scale(&screen, (720, 450)), None);
        assert_eq!(a11y_scale(&screen, (2880, 900)), None);
        assert_eq!(a11y_scale(&surface(0.0, 0.0, 0.0, 0.0), (10, 10)), None);
        let scaled = scale_geometry(surface(10.0, 20.0, 100.0, 50.0), 2.0);
        assert_eq!(scaled, surface(20.0, 40.0, 200.0, 100.0));
    }

    #[test]
    fn resolve_locator_path_hits_unique_and_not_found() {
        let obs = sample_observation();
        let root = &obs.root;
        // Path [0] reaches the OK button → Path match (role + name agree).
        let hit = resolve_locator(
            root,
            &Locator {
                path: vec![0],
                role: "button".to_string(),
                name: Some("OK".to_string()),
            },
        );
        assert!(matches!(hit, LocatorMatch::Path(_)));
        // Path out of range → falls through to the search (button "OK" is unique).
        let fallback = resolve_locator(
            root,
            &Locator {
                path: vec![99],
                role: "button".to_string(),
                name: Some("OK".to_string()),
            },
        );
        assert!(matches!(fallback, LocatorMatch::Unique(_)));
        // Path node exists but role+name mismatch → search again.
        let mismatch = resolve_locator(
            root,
            &Locator {
                path: vec![1],
                role: "button".to_string(),
                name: Some("OK".to_string()),
            },
        );
        assert!(matches!(mismatch, LocatorMatch::Unique(_)));
        // No node matches → NotFound.
        let none = resolve_locator(
            root,
            &Locator {
                path: Vec::new(),
                role: "AXSlider".to_string(),
                name: None,
            },
        );
        assert!(matches!(none, LocatorMatch::NotFound));
    }

    #[test]
    fn resolve_locator_ambiguous_when_many_match() {
        let obs = sample_observation();
        let root = &obs.root;
        // Two buttons match the role with no name constraint → ambiguous.
        let all_buttons = resolve_locator(
            root,
            &Locator {
                path: Vec::new(),
                role: "AXButton".to_string(),
                name: None,
            },
        );
        assert!(matches!(all_buttons, LocatorMatch::Ambiguous));
        // A path that pins one of them wins over the ambiguity.
        let pinned = resolve_locator(
            root,
            &Locator {
                path: vec![0],
                role: "AXButton".to_string(),
                name: None,
            },
        );
        assert!(matches!(pinned, LocatorMatch::Path(_)));
    }

    #[test]
    fn parse_key_chord_splits_modifiers_and_key() {
        let (mods, key) = parse_key_chord("cmd+shift+t").unwrap();
        assert_eq!(mods, vec![Modifier::Cmd, Modifier::Shift]);
        assert_eq!(key, "t");

        // Modifier names are case-insensitive; aliases accepted.
        let (mods, key) = parse_key_chord("Control+Alt+Delete").unwrap();
        assert_eq!(mods, vec![Modifier::Ctrl, Modifier::Alt]);
        assert_eq!(key, "delete");

        // A lone key word.
        let (mods, key) = parse_key_chord("Return").unwrap();
        assert!(mods.is_empty());
        assert_eq!(key, "return");

        // Digits and function keys pass through.
        let (_, key) = parse_key_chord("5").unwrap();
        assert_eq!(key, "5");
        let (_, key) = parse_key_chord("F5").unwrap();
        assert_eq!(key, "f5");

        // Modifiers only, no key → error.
        let err = parse_key_chord("cmd+ctrl").unwrap_err();
        assert!(format!("{err}").contains("no key name"), "got {err}");

        // Trailing '+' → empty part → error.
        assert!(parse_key_chord("cmd+").is_err());
        // Two key names → error.
        let err = parse_key_chord("a+b").unwrap_err();
        assert!(format!("{err}").contains("too many key names"), "got {err}");
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn a_share_cannot_outlive_its_call_or_lengthen_a_short_one() {
        let call = Budget::new(
            Duration::from_secs(10),
            "window listing",
            "the running applications",
        );
        // A share asking for more than the call has ends with the call.
        let long = call.slice(Duration::from_secs(60), "the windows of Numbers");
        assert_eq!(long.end, call.end);
        assert!(long.limit <= call.limit);
        // A share asking for less keeps its own, shorter end.
        let short = call.slice(Duration::ZERO, "the windows of Numbers");
        assert_eq!(short.limit, Duration::ZERO);
        assert!(short.expired());
        assert!(!call.expired());
    }

    #[test]
    fn a_spent_call_says_it_was_abandoned_and_names_the_target() {
        let call = Budget::new(Duration::ZERO, "window listing", "the windows of Numbers");
        let text = format!("{:#}", call.check().expect_err("the clock is spent"));
        assert!(text.starts_with("degraded: "), "{text}");
        assert!(
            text.contains("the windows of Numbers stopped answering accessibility requests"),
            "{text}"
        );
        assert!(
            text.contains("the window listing was abandoned after 0s"),
            "{text}"
        );
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn a_spent_share_says_the_item_was_skipped_not_the_call_abandoned() {
        let call = Budget::new(
            Duration::from_secs(20),
            "window listing",
            "the running applications",
        );
        let share = call.slice(Duration::ZERO, "the windows of Numbers");
        let note = share.note();
        assert!(
            note.contains("the windows of Numbers stopped answering accessibility requests"),
            "{note}"
        );
        assert!(
            note.contains("it was skipped, so the window listing went on without it"),
            "{note}"
        );
        // A share never speaks for the call: the model must not be told to retry a call
        // that returned results, nor read the skipped application as the call failing.
        assert!(!note.contains("abandoned"), "{note}");
        assert!(!note.contains("retry"), "{note}");
        // The listing renders an expired share as "skipped" and anything else as itself.
        assert_eq!(
            unread_reason(&anyhow!("window refused"), &share),
            SKIPPED_NOTE
        );
        assert_eq!(
            unread_reason(&anyhow!("window refused"), &call),
            "window refused"
        );
    }
}
