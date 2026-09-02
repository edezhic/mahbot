//! Platform-agnostic core of the Computer (GUI observe/act) tool.
//!
//! Owns the shared types, coordinate math, the element-ref/observation lifecycle,
//! the target registry, tree rendering, and the [`Backend`] trait. Contains no
//! platform imports — `macos.rs`/`linux.rs`/`stub.rs` implement [`Backend`].

use crate::util::UnwrapPoison;
use anyhow::anyhow;
use serde::Deserialize;
use serde::Serialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::sync::{LazyLock, Mutex};

/// Maximum normalized coordinate value, inclusive on both axes ([`NORMALIZED_MAX`]).
pub(crate) const NORMALIZED_MAX: f64 = 1000.0;

/// Surfaces exposing fewer than this many interactive elements are "AX-thin" —
/// the model is nudged toward screenshots over the accessibility tree.
pub(crate) const AX_THIN_INTERACTIVE_ELEMENTS: usize = 5;

/// Cap on nodes rendered in a tree dump — prevents unbounded output.
pub(crate) const MAX_RENDER_NODES: usize = 4000;

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
pub(crate) const ERR_AMBIGUOUS_LOCATOR: &str = "ambiguous-locator";
pub(crate) const ERR_NOT_MATCHED: &str = "not-matched";

/// Build a taxonomy-prefixed error: `"<kind>: <msg>"`. The message MUST start
/// with the taxonomy word so the model can branch on the category.
#[must_use]
pub(crate) fn taxonomy_error(kind: &str, msg: impl fmt::Display) -> anyhow::Error {
    anyhow!("{kind}: {msg}")
}

// ── Types ───────────────────────────────────────────────────────────────

/// Surface extent in GLOBAL logical points (top-left origin). For a window this
/// is the window's frame; for the screen target it is the virtual-desktop
/// bounding rect across all displays (origin at its min corner).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurfaceGeometry {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Normalized 0-1000 region (corner order is normalized by consumers).
///
/// `Default` yields an all-zero (degenerate/invalid) region. It is never
/// constructed at runtime — it exists only so the `ComputerAction` `EnumIter`
/// (used by the schema lockstep test) can build a `Zoom` variant via
/// `Default::default()`.
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

/// A node in the accessibility tree. `frame` is surface-local `(x, y, w, h)`
/// in global logical points; `children` mirrors the AX hierarchy.
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

/// A snapshot of one observe call against a target surface.
#[derive(Debug, Clone)]
pub(crate) struct Observation {
    pub app_name: String,
    pub window_title: Option<String>,
    pub surface: SurfaceGeometry,
    pub root: UiNode,
}

/// An application participating in the target registry.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AppInfo {
    pub pid: Option<u32>,
    pub name: String,
}

/// A window of an application.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowInfo {
    pub app_pid: Option<u32>,
    pub app_name: String,
    pub title: String,
    pub index: usize,
    pub surface: SurfaceGeometry,
}

/// Plain RGBA pixel data (Send-safe, no platform handles).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The surface an action targets. `Focused` is the default; an app id targets
/// the app's focused window (empty title, index 0 in the registry).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TargetSpec {
    Focused,
    Window {
        app_pid: Option<u32>,
        app_name: String,
        title: String,
        index: usize,
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
            title: win.title.clone(),
            index: win.index,
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

// ── Key chord parsing ───────────────────────────────────────────────────

/// Parse a key chord string ("cmd+shift+t", "return", "ctrl+c") into its
/// modifiers and a single key name (lowercased). Pure and platform-agnostic —
/// the backend maps the key name to a platform virtual keycode.
///
/// Recognized modifiers (case-insensitive): `cmd`/`command`, `ctrl`/`control`,
/// `alt`/`option`, `shift`. The non-modifier segment is the key name and is
/// returned lowercased (e.g. `"return"`, `"a"`, `"f5"`, `"5"`).
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

/// Map a normalized 0-1000 point to absolute global logical points on the
/// target surface.
pub(crate) fn normalized_to_surface(
    nx: f64,
    ny: f64,
    geo: &SurfaceGeometry,
) -> anyhow::Result<(f64, f64)> {
    if !nx.is_finite()
        || !ny.is_finite()
        || !(0.0..=NORMALIZED_MAX).contains(&nx)
        || !(0.0..=NORMALIZED_MAX).contains(&ny)
    {
        anyhow::bail!("out-of-bounds coordinates ({nx}, {ny}) — expected normalized 0-1000");
    }
    Ok((
        geo.x + nx / NORMALIZED_MAX * geo.width,
        geo.y + ny / NORMALIZED_MAX * geo.height,
    ))
}

/// Map an absolute global logical point to normalized 0-1000 coordinates
/// against the target surface, clamped to the surface extent.
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

/// True when an absolute global logical point lies inside the surface extent.
/// Top-left corner is inclusive; the bottom-right edge is exclusive.
#[must_use]
pub(crate) fn contains_point(geo: &SurfaceGeometry, ax: f64, ay: f64) -> bool {
    ax >= geo.x && ax < geo.x + geo.width && ay >= geo.y && ay < geo.y + geo.height
}

/// Map a normalized region to an integer pixel rect `(x, y, w, h)` of the
/// captured image (corners ordered, clamped to the image bounds).
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn region_to_pixels(
    region: &Region,
    geo: &SurfaceGeometry,
    px_w: u32,
    px_h: u32,
) -> anyhow::Result<(u32, u32, u32, u32)> {
    if geo.width <= 0.0 || geo.height <= 0.0 {
        anyhow::bail!(
            "region_to_pixels: surface geometry has non-positive extent ({}, {})",
            geo.width,
            geo.height
        );
    }
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

// ── Element-ref lifecycle ───────────────────────────────────────────────

/// Cached observation with the refs resolved against it.
struct ObservationCache {
    target: TargetSpec,
    refs: HashMap<String, Locator>,
}

static LAST_OBSERVATION: LazyLock<Mutex<HashMap<String, ObservationCache>>> =
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

/// The target pinned by the calling agent's most recent observation, if any.
/// Coordinate actions without an explicit target use it so a focus shift between
/// observe and the action never retargets a different surface than the refs
/// came from. `None` means no observation has been cached yet.
#[must_use]
pub(crate) fn last_observation_target() -> Option<TargetSpec> {
    let key = agent_key();
    let guard = LAST_OBSERVATION.lock().unwrap_poison();
    guard.get(&key).map(|c| c.target.clone())
}

// ── Target registry ─────────────────────────────────────────────────────

static TARGETS: LazyLock<Mutex<HashMap<String, HashMap<String, TargetSpec>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Update the calling agent's target map. Incoming ids replace existing ids of
/// the SAME id class (leading `a` for apps, `w` for windows) while ids of the
/// OTHER class survive — an apps enumeration no longer invalidates window ids
/// and vice versa, so both enumerations coexist. An empty enumeration is a
/// no-op (the class is inferred from the first id, so there is nothing to
/// otherwise clear).
pub(crate) fn store_targets(targets: Vec<(String, TargetSpec)>) {
    let key = agent_key();
    let mut guard = TARGETS.lock().unwrap_poison();
    let map = guard.entry(key).or_default();
    if let Some(class) = targets.first().map(|(id, _)| id.chars().next()) {
        map.retain(|id, _| id.chars().next() != class);
    }
    for (id, spec) in targets {
        map.insert(id, spec);
    }
}

/// Look up a stored target id without erroring (used when an argument may or
/// may not be a target id).
pub(crate) fn get_target(id: &str) -> Option<TargetSpec> {
    let key = agent_key();
    let guard = TARGETS.lock().unwrap_poison();
    guard.get(&key)?.get(id).cloned()
}

/// Resolve a target id or the `"screen"` literal. `None` → [`TargetSpec::Focused`].
pub(crate) fn resolve_target(target: Option<&str>) -> anyhow::Result<TargetSpec> {
    let Some(target) = target else {
        return Ok(TargetSpec::Focused);
    };
    if target == "screen" {
        return Ok(TargetSpec::Screen);
    }
    let key = agent_key();
    let guard = TARGETS.lock().unwrap_poison();
    let spec = guard.get(&key).and_then(|m| m.get(target)).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown target '{target}' — it is not a stored app/window id; enumerate targets \
             with 'apps' (a1-style) or 'windows' (w1-style) first"
        )
    })?;
    Ok(spec.clone())
}

/// Remove an agent's cached observation and target entries. Called at agent-run
/// end so a stale per-agent registry never leaks across runs (see
/// [`crate::tools::computer::cleanup_agent_state`]). No-op for unknown keys.
pub(crate) fn clear_agent_state(key: &str) {
    {
        let mut guard = LAST_OBSERVATION.lock().unwrap_poison();
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
/// header carries the node/interactive totals (computed in one pre-walk so they
/// stay accurate even when the render is capped at [`MAX_RENDER_NODES`]).
#[must_use]
pub(crate) fn render_tree(obs: &Observation) -> String {
    let surf = &obs.surface;
    let title = obs.window_title.as_deref().unwrap_or("");
    let (node_count, interactive_count) = count_stats(&obs.root);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "[Surface: app=\"{}\" window=\"{}\" {}x{} @ ({}, {}) — {} nodes, {} interactive]",
        obs.app_name, title, surf.width, surf.height, surf.x, surf.y, node_count, interactive_count
    );
    let mut counter = 0usize;
    let mut truncated = false;
    render_node(&obs.root, 0, &mut out, &mut counter, &mut truncated);
    if truncated {
        out.push_str("… (truncated)\n");
    }
    out
}

fn render_node(
    node: &UiNode,
    depth: usize,
    out: &mut String,
    counter: &mut usize,
    truncated: &mut bool,
) {
    if *counter >= MAX_RENDER_NODES {
        *truncated = true;
        return;
    }
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
        let v = crate::util::truncate(value, 120);
        let _ = write!(line, " value=\"{v}\"");
    }
    if node.focused {
        line.push_str(" focused");
    }
    out.push_str(&line);
    out.push('\n');
    for child in &node.children {
        render_node(child, depth + 1, out, counter, truncated);
    }
}

/// Pre-order DFS index of `target` in `root` — maps a matched node back to its
/// handle slot (the handles are kept in the same pre-order as the tree).
/// Platform-free; shared by the backends in `act_on_element`.
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

fn node_matches(node: &UiNode, locator: &Locator) -> bool {
    let role_ok = normalized_role(&node.role).eq_ignore_ascii_case(normalized_role(&locator.role));
    let name_ok = match (&locator.name, &node.name) {
        (None, _) => true,
        (Some(l), Some(n)) => l == n,
        (Some(_), None) => false,
    };
    role_ok && name_ok
}

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
    /// Cheap synchronous accessibility-channel check (macOS: AX trust equivalent).
    fn accessibility_available(&self) -> bool;
    /// Slow capture-channel probe (screen-recording grant); caller caches with TTL.
    async fn capture_available(&self) -> bool;
    /// Actionable diagnosis for a missing capture grant.
    fn capture_unavailable_error(&self) -> anyhow::Error;
    async fn list_apps(&self) -> anyhow::Result<Vec<AppInfo>>;
    async fn list_windows(&self, app: Option<&AppInfo>) -> anyhow::Result<Vec<WindowInfo>>;
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
    ) -> anyhow::Result<()>;
    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> anyhow::Result<()>;
    /// Pointer position in ABSOLUTE global logical points.
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
    fn region_to_pixels_maps_orders_and_clamps() {
        let geo = surface(0.0, 0.0, 1000.0, 1000.0);
        // Normalized 200..=800 on a 1000px image → 200..=800 px.
        let region = Region {
            x0: 200.0,
            y0: 100.0,
            x1: 800.0,
            y1: 500.0,
        };
        assert_eq!(
            region_to_pixels(&region, &geo, 1000, 1000).unwrap(),
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
            region_to_pixels(&rev, &geo, 1000, 1000).unwrap(),
            (200, 100, 600, 400)
        );
        // Out-of-range normalized coords are rejected.
        let bad = Region {
            x0: 1100.0,
            y0: 0.0,
            x1: 200.0,
            y1: 100.0,
        };
        assert!(region_to_pixels(&bad, &geo, 1000, 1000).is_err());
        // A region beyond the image is clamped to the image bounds (>=1px).
        let big = Region {
            x0: 500.0,
            y0: 500.0,
            x1: 1000.0,
            y1: 1000.0,
        };
        let (x, y, w, h) = region_to_pixels(&big, &geo, 200, 200).unwrap();
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
    fn render_tree_truncates_value_and_nodes() {
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
        };
        let text = render_tree(&obs);
        assert!(text.contains("value=\""), "value should render");
        assert!(!text.contains(&long), "long value must be truncated");
        assert!(
            text.contains("…"),
            "truncated value should carry an ellipsis"
        );

        // Node cap: a wide tree beyond MAX_RENDER_NODES shows the truncated note
        // (kept shallow so the DFS does not recurse deeply enough to overflow).
        let mut root = node("AXWindow", Some("w"));
        for i in 0..(MAX_RENDER_NODES + 10) {
            root.children.push(node(&format!("AXButton{i}"), None));
        }
        let obs = Observation {
            app_name: "A".to_string(),
            window_title: None,
            surface: surface(0.0, 0.0, 10.0, 10.0),
            root,
        };
        let text = render_tree(&obs);
        assert!(text.contains("… (truncated)"));
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
        };
        store_observation(target, &obs2);
        let err = resolve_ref("e2").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("stale-element"), "got {msg}");
        // Garbage ref → stale-element too.
        let err = resolve_ref("zzz").unwrap_err();
        assert!(format!("{err}").contains("stale-element"));
    }

    #[test]
    fn resolve_target_resolves_focused_screen_stored_and_unknown() {
        assert_eq!(resolve_target(None).unwrap(), TargetSpec::Focused);
        assert_eq!(resolve_target(Some("screen")).unwrap(), TargetSpec::Screen);

        store_targets(vec![(
            "a1".to_string(),
            TargetSpec::Window {
                app_pid: Some(42),
                app_name: "Safari".to_string(),
                title: String::new(),
                index: 0,
            },
        )]);
        assert_eq!(
            resolve_target(Some("a1")).unwrap(),
            TargetSpec::Window {
                app_pid: Some(42),
                app_name: "Safari".to_string(),
                title: String::new(),
                index: 0,
            }
        );
        let err = resolve_target(Some("w9")).unwrap_err();
        assert!(format!("{err}").contains("unknown target"));
    }

    #[test]
    fn store_targets_merges_only_same_id_class() {
        let window = |app_pid: u32, name: &str, title: &str, index: usize| TargetSpec::Window {
            app_pid: Some(app_pid),
            app_name: name.to_string(),
            title: title.to_string(),
            index,
        };
        // Seed one app id and one window id.
        store_targets(vec![
            ("a5".to_string(), window(1, "Alpha", "", 0)),
            ("w5".to_string(), window(2, "Beta", "doc", 0)),
        ]);
        assert!(resolve_target(Some("a5")).is_ok());
        assert!(resolve_target(Some("w5")).is_ok());

        // Re-enumerating apps replaces a-ids but keeps the OTHER class (w5).
        store_targets(vec![("a5".to_string(), window(3, "Gamma", "", 0))]);
        assert!(resolve_target(Some("a5")).is_ok());
        assert!(
            resolve_target(Some("w5")).is_ok(),
            "window id survived an apps enumeration"
        );

        // Re-enumerating windows keeps a-ids and clears only stale w-ids.
        store_targets(vec![("w6".to_string(), window(2, "Beta", "doc2", 1))]);
        assert!(
            resolve_target(Some("a5")).is_ok(),
            "app id survived a windows enumeration"
        );
        assert!(
            resolve_target(Some("w5")).is_err(),
            "stale window id cleared"
        );
        assert!(resolve_target(Some("w6")).is_ok());
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
