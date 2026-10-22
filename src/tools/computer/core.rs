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
use std::sync::{LazyLock, Mutex};

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

/// Character cap on a node value shown in a tree dump. The dump is a map of the
/// surface, not a copy of its text, so a long value is cut here and counted in
/// the truncation header.
pub(crate) const RENDER_VALUE_CHARS: usize = 120;

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
                        _ => {
                            let _: de::IgnoredAny = map.next_value()?;
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

/// Where a tree read hit a ceiling. Every `None` means that ceiling was not hit;
/// [`render_tree`] states each one that was, in the header — the only part of a
/// large observation a spill preview guarantees the agent reads.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Truncation {
    /// Nodes beyond this node ceiling were not read (the walk stopped).
    pub nodes: Option<usize>,
    /// Children of this many nodes sit at the depth ceiling and were not read.
    pub depth: Option<Cut>,
    /// This many names were shortened at the read's own character cap.
    pub names: Option<Cut>,
}

/// How many items a ceiling cut, and the ceiling itself.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cut {
    pub count: usize,
    pub limit: usize,
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

    /// Record one name shortened at the read's own character cap.
    #[cfg_attr(not(target_os = "windows"), expect(dead_code))]
    pub(crate) fn cut_name(&mut self, limit: usize) {
        self.names.get_or_insert(Cut { count: 0, limit }).count += 1;
    }

    /// One line per ceiling that was hit, each naming it. `values_cut` comes from
    /// the renderer, which is where a value is cut for display.
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
                "[Tree truncated: {} nodes at the {}-deep ceiling — their children were not read]",
                cut.count, cut.limit
            ));
        }
        if values_cut > 0 {
            out.push(format!(
                "[Tree truncated: {values_cut} {} cut at {RENDER_VALUE_CHARS} chars]",
                if values_cut == 1 { "value" } else { "values" }
            ));
        }
        if let Some(cut) = self.names {
            out.push(format!(
                "[Tree truncated: {} {} cut at {} chars]",
                cut.count,
                if cut.count == 1 { "name" } else { "names" },
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
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
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

// ── Coordinate math ─────────────────────────────────────────────────────

/// Reject a normalized point that is not a finite 0-1000 coordinate.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux", target_os = "windows", test)),
    expect(dead_code)
)]
pub(crate) fn check_normalized(nx: f64, ny: f64) -> anyhow::Result<()> {
    if !nx.is_finite()
        || !ny.is_finite()
        || !(0.0..=NORMALIZED_MAX).contains(&nx)
        || !(0.0..=NORMALIZED_MAX).contains(&ny)
    {
        anyhow::bail!("out-of-bounds coordinates ({nx}, {ny}) — expected normalized 0-1000");
    }
    Ok(())
}

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
    check_normalized(nx, ny)?;
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
    for v in [region.x0, region.y0, region.x1, region.y1] {
        if !v.is_finite() || !(0.0..=NORMALIZED_MAX).contains(&v) {
            anyhow::bail!("region_to_pixels: normalized coordinate {v} out of 0-1000 range");
        }
    }
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
pub(crate) fn crop_rgba(cap: &Capture, rect: (u32, u32, u32, u32)) -> anyhow::Result<Capture> {
    let (x, y, w, h) = rect;
    if cap.width == 0 || cap.height == 0 {
        anyhow::bail!("crop_rgba: source capture is empty");
    }
    let x = x.min(cap.width.saturating_sub(1));
    let y = y.min(cap.height.saturating_sub(1));
    let w = w.min(cap.width - x).max(1);
    let h = h.min(cap.height - y).max(1);

    let stride = cap.width as usize * 4;
    let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
    for row in y..(y + h) {
        let start = row as usize * stride + x as usize * 4;
        rgba.extend_from_slice(&cap.rgba[start..start + w as usize * 4]);
    }
    Ok(Capture {
        width: w,
        height: h,
        rgba,
    })
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

/// ZPixmap tile (`depth` 24/32, 4 bytes/pixel) → RGBA.
#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))] // only the Linux backend calls this; tests exercise it everywhere
pub(crate) fn x11_pixels_to_rgba(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u8,
    lsb_first: bool,
) -> Result<Vec<u8>, anyhow::Error> {
    if depth != 24 && depth != 32 {
        return Err(taxonomy_error(
            ERR_DEGRADED,
            format!("unsupported X11 depth {depth} (need 24/32)"),
        ));
    }
    let pixels = width as usize * height as usize;
    if data.len() < pixels * 4 {
        anyhow::bail!("short X11 image body: {} < {}", data.len(), pixels * 4);
    }
    let mut rgba = Vec::with_capacity(pixels * 4);
    for px in data.as_chunks::<4>().0.iter().take(pixels) {
        if lsb_first {
            rgba.extend_from_slice(&[px[2], px[1], px[0], 0xff]);
        } else {
            rgba.extend_from_slice(&[px[1], px[2], px[3], 0xff]);
        }
    }
    Ok(rgba)
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
    /// RGBA pixels of a target surface (window capture, or full virtual desktop for Screen).
    async fn capture(&self, target: &TargetSpec) -> anyhow::Result<Capture>;
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
        let src = Capture {
            width: 2,
            height: 2,
            rgba: vec![1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255],
        };
        blit_rgba(&mut dest, 4, 4, &src, 1, 1);
        assert_eq!(&dest[4 * 4 + 4..4 * 4 + 8], &[1, 2, 3, 255]);
        assert_eq!(&dest[4 * 4 + 8..4 * 4 + 12], &[4, 5, 6, 255]);
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
        // 4x2 image: rows are constant-color for easy verification.
        let mut rgba = Vec::new();
        for (r, c) in [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
        ] {
            rgba.extend_from_slice(&[r * 40, c * 10, 0, 255]);
        }
        let cap = Capture {
            width: 4,
            height: 2,
            rgba,
        };
        let crop = crop_rgba(&cap, (1, 0, 2, 2)).unwrap();
        assert_eq!(crop.width, 2);
        assert_eq!(crop.height, 2);
        assert_eq!(crop.rgba.len(), 2 * 2 * 4);
        // First cropped pixel = original row0 col1 = [0, 10, 0, 255].
        assert_eq!(&crop.rgba[..4], &[0, 10, 0, 255]);
        // Out-of-bounds rect is clamped, never panics.
        let clamp = crop_rgba(&cap, (10, 10, 50, 50)).unwrap();
        assert!(clamp.width >= 1 && clamp.height >= 1);
    }

    #[test]
    fn x11_pixels_convert_both_byte_orders() {
        let rgba = x11_pixels_to_rgba(&[10, 20, 30, 0, 40, 50, 60, 0], 2, 1, 24, true).unwrap();
        assert_eq!(rgba, vec![30, 20, 10, 255, 60, 50, 40, 255]);
        let rgba = x11_pixels_to_rgba(&[0, 30, 20, 10], 1, 1, 32, false).unwrap();
        assert_eq!(rgba, vec![30, 20, 10, 255]);
        let err = x11_pixels_to_rgba(&[0; 4], 1, 1, 16, true).unwrap_err();
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
                names: Some(Cut {
                    count: 3,
                    limit: 200,
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
        assert!(header.contains("3 names cut at 200 chars"), "{header}");
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
}
