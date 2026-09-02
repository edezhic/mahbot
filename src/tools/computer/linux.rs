//! Linux [`Backend`] for the Computer tool: AT-SPI2 accessibility (async D-Bus,
//! observe/act), XTEST input synthesis via enigo, and X11 capture via xcap. A
//! THIN adapter over the three mature crates — no re-implementation of what they
//! already provide. Raw input needs X11 (pure-Wayland surfaces degrade); capture
//! is best-effort on X11.
//!
//! Surface geometry is PIXELS on Linux: X11 has no stable logical-point space and
//! mixed-DPI makes physical↔logical translation unreliable — this is precisely
//! why the tool is accessibility-first here.

#![cfg(target_os = "linux")]
// A handful of pedantic cast lints are inherent to the C ABI-style integer
// geometry the atspi/xcap crates expose (i32 screen coords, u32 pixel dims).
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::module_name_repetitions,
    clippy::semicolon_if_nothing_returned
)]

use super::core::{
    self, AppInfo, Backend, Capture, ElementAct, Locator, Modifier, MouseButton, Observation,
    RawInput, ScrollDirection, SurfaceGeometry, TargetSpec, UiNode, WindowInfo,
};
use crate::util::with_block_in_place;
use anyhow::anyhow;
use async_recursion::async_recursion;
use atspi::proxy::accessible::{AccessibleProxy, ObjectRefExt};
use atspi::proxy::proxy_ext::{Proxies, ProxyExt};
use atspi::{AccessibilityConnection, CoordType, ObjectRefOwned, Role, State as AtspiState};
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use std::time::Duration;
use tracing::debug;

pub(crate) static LINUX_BACKEND: LinuxBackend = LinuxBackend {
    connection: tokio::sync::OnceCell::const_new(),
};

pub(crate) struct LinuxBackend {
    /// Lazily-established AT-SPI connection (`None` = connect failed → degraded).
    /// [`atspi::AccessibilityConnection`] is `Clone`/`Send + Sync`, so the handle
    /// is cloned out per call (never a `&mut` across an await).
    connection: tokio::sync::OnceCell<Option<AccessibilityConnection>>,
}

const MAX_DEPTH: usize = 25;
/// Cap on text pulled into a node's `value` so a huge document never floods the
/// tree dump (the renderer truncates, but we avoid the round-trip too).
const MAX_VALUE_CHARS: i32 = 2000;
const INTERP_STEPS: usize = 12;

const CONN_DEGRADED_MSG: &str =
    "AT-SPI2 accessibility bus unavailable — is at-spi2 installed and the session bus running?";

const TEXT_VALUE_ROLES: &[&str] = &[
    "text",
    "textfield",
    "entry",
    "paragraph",
    "label",
    "static",
    "tooltip",
];

const KNOWN_KEYS: &str = "return, enter, tab, space, delete/backspace, forwarddelete, escape, \
     home, end, pageup, pagedown, up, down, left, right, f1..f12, capslock, insert, minus, equal, \
     comma, period, slash, semicolon, quote, backslash, backtick, leftbracket, rightbracket, a-z, 0-9";

impl LinuxBackend {
    /// Lazy, one-shot connection. The computer tool serializes every action
    /// through a process lock, so two tasks can never race this init.
    async fn conn(&self) -> Result<AccessibilityConnection, anyhow::Error> {
        if let Some(conn) = self.connection.get().and_then(|o| o.as_ref()) {
            return Ok(conn.clone());
        }
        match AccessibilityConnection::new().await {
            Ok(conn) => {
                let _ = self.connection.set(Some(conn.clone()));
                Ok(conn)
            }
            Err(e) => {
                let _ = self.connection.set(None);
                debug!(error = %e, "atspi connection failed");
                Err(core::taxonomy_error(core::ERR_DEGRADED, CONN_DEGRADED_MSG))
            }
        }
    }
}

// ── Pure helpers ─────────────────────────────────────────────────────────

/// True when only a Wayland session is active — XTEST synthesis is impossible
/// and every raw-input variant must degrade.
fn is_wayland_only(wayland_display: bool, x11_display: bool) -> bool {
    wayland_display && !x11_display
}

fn wayland_only_env() -> bool {
    is_wayland_only(
        std::env::var_os("WAYLAND_DISPLAY").is_some(),
        std::env::var_os("DISPLAY").is_some(),
    )
}

/// AT-SPI `Role` → the lowercase string the core matches on. Most CamelCase enum
/// names lowercase to the expected role; a few are remapped to the conventional
/// AT-SPI role-name spelling so reflected roles read naturally.
fn role_name(role: Role) -> String {
    match role {
        Role::Button => "pushbutton".to_string(),
        Role::Entry => "textfield".to_string(),
        Role::PageTab => "tab".to_string(),
        Role::PageTabList => "tabgroup".to_string(),
        Role::ListItem => "listitem".to_string(),
        Role::ScrollBar => "scrollbar".to_string(),
        other => format!("{other:?}").to_lowercase(),
    }
}

fn is_text_role(role: &str) -> bool {
    TEXT_VALUE_ROLES
        .iter()
        .any(|r| r.eq_ignore_ascii_case(role))
}

fn is_window_role(role: Role) -> bool {
    matches!(role, Role::Window | Role::Frame | Role::Dialog)
}

/// Union bounding rect of monitor rects `(x, y, w, h)` in pixels. Origins may be
/// negative (a left/upper monitor); the result keeps them so normalized ↔ pixel
/// mapping stays exact.
fn surface_from_rects(rects: &[(i32, i32, i32, i32)]) -> SurfaceGeometry {
    if rects.is_empty() {
        return SurfaceGeometry {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        };
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y, w, h) in rects {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x + w);
        max_y = max_y.max(y + h);
    }
    SurfaceGeometry {
        x: f64::from(min_x),
        y: f64::from(min_y),
        width: f64::from(max_x - min_x),
        height: f64::from(max_y - min_y),
    }
}

/// D-Bus / atspi error mapping: access denied → permission-denied, a gone
/// app/object/method → stale-element, a dropped reply → degraded, else plain.
fn dbus_err(err: impl std::fmt::Display, ctx: &str) -> anyhow::Error {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("accessdenied")
        || lower.contains("notauthorized")
        || lower.contains("authfailed")
    {
        core::taxonomy_error(core::ERR_PERMISSION_DENIED, format!("{ctx}: {msg}"))
    } else if lower.contains("serviceunknown")
        || lower.contains("unknownobject")
        || lower.contains("namehasnoowner")
        || lower.contains("unknownmethod")
    {
        core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("{ctx}: {msg} — app or window closed; re-enumerate"),
        )
    } else if lower.contains("noreply")
        || lower.contains("failed to connect")
        || lower.contains("input/output")
        || lower.contains("connection closed")
    {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("{ctx}: {msg} — transient bus failure, retry"),
        )
    } else {
        anyhow::anyhow!("{ctx}: {msg}")
    }
}

// ── Registry / tree enumeration ──────────────────────────────────────────

struct WindowHandle {
    object_ref: ObjectRefOwned,
    surface: SurfaceGeometry,
    title: String,
    active: bool,
    focused: bool,
    index: usize,
}

struct ResolvedWindow {
    object_ref: ObjectRefOwned,
    app_info: AppInfo,
    title: String,
    index: usize,
    surface: SurfaceGeometry,
}

/// Root accessible's children = applications that expose accessibility. Apps
/// that don't simply don't appear — an accepted AX-first limitation.
async fn enumerate_apps(
    conn: &AccessibilityConnection,
) -> Result<Vec<(ObjectRefOwned, AppInfo)>, anyhow::Error> {
    let root = conn
        .root_accessible_on_registry()
        .await
        .map_err(|e| dbus_err(e, "access registry root"))?;
    let app_refs = root
        .get_children()
        .await
        .map_err(|e| dbus_err(e, "enumerate applications"))?;
    let mut out = Vec::new();
    for app_ref in app_refs {
        let Some(accessible) = app_ref.as_accessible_proxy(conn.connection()).await.ok() else {
            continue;
        };
        let name = accessible.name().await.unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        // atspi 0.30 has no Application::get_pid, so the pid stays unknown.
        out.push((app_ref, AppInfo { pid: None, name }));
    }
    Ok(out)
}

async fn app_info_of(
    conn: &AccessibilityConnection,
    app_ref: &ObjectRefOwned,
) -> Result<AppInfo, anyhow::Error> {
    let accessible = app_ref
        .as_accessible_proxy(conn.connection())
        .await
        .map_err(|e| dbus_err(e, "open application"))?;
    let name = accessible.name().await.unwrap_or_default();
    Ok(AppInfo { pid: None, name })
}

async fn resolve_app_object(
    conn: &AccessibilityConnection,
    app_name: &str,
    app_pid: Option<u32>,
) -> Result<ObjectRefOwned, anyhow::Error> {
    let apps = enumerate_apps(conn).await?;
    let needle = app_name.to_lowercase();
    let by_name = apps
        .iter()
        .find(|(_, info)| info.name.to_lowercase() == needle);
    let by_pid = apps
        .iter()
        .find(|(_, info)| app_pid.is_some() && info.pid == app_pid);
    by_name.or(by_pid).map(|(r, _)| r.clone()).ok_or_else(|| {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            format!("application '{app_name}' not found — enumerate apps first"),
        )
    })
}

/// The app's window-role children (Window/Frame/Dialog): global screen extents,
/// title, active/focused state and filtered position index.
async fn app_window_handles(
    app_ref: &ObjectRefOwned,
    conn: &AccessibilityConnection,
) -> Vec<WindowHandle> {
    let mut out = Vec::new();
    let Ok(accessible) = app_ref.as_accessible_proxy(conn.connection()).await else {
        return out;
    };
    let Ok(children) = accessible.get_children().await else {
        return out;
    };
    for child in children {
        let Ok(ca) = child.as_accessible_proxy(conn.connection()).await else {
            continue;
        };
        let role = ca.get_role().await.unwrap_or(Role::Unknown);
        if !is_window_role(role) {
            continue;
        }
        let title = ca.name().await.unwrap_or_default();
        let state = ca.get_state().await.unwrap_or_default();
        let Some(surface) = window_screen_extents(&ca).await else {
            continue;
        };
        let index = out.len();
        out.push(WindowHandle {
            object_ref: child,
            surface,
            title,
            active: state.contains(AtspiState::Active),
            focused: state.contains(AtspiState::Focused),
            index,
        });
    }
    out
}

async fn window_screen_extents(accessible: &AccessibleProxy<'_>) -> Option<SurfaceGeometry> {
    let proxies = accessible.proxies().await.ok()?;
    let comp = proxies.component().await.ok()?;
    let (x, y, w, h) = comp.get_extents(CoordType::Screen).await.ok()?;
    Some(SurfaceGeometry {
        x: f64::from(x),
        y: f64::from(y),
        width: f64::from(w),
        height: f64::from(h),
    })
}

async fn resolve_window_in_app(
    conn: &AccessibilityConnection,
    app_ref: &ObjectRefOwned,
    title: &str,
    index: usize,
) -> Result<ObjectRefOwned, anyhow::Error> {
    let wins = app_window_handles(app_ref, conn).await;
    if wins.is_empty() {
        return Err(core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "no windows found for this app — re-enumerate windows",
        ));
    }
    if !title.is_empty()
        && let Some(w) = wins.iter().find(|w| w.title == title)
    {
        return Ok(w.object_ref.clone());
    }
    // App-level target (empty title, index 0): the app's active window, else first.
    if title.is_empty()
        && index == 0
        && let Some(w) = wins.iter().find(|w| w.active)
    {
        return Ok(w.object_ref.clone());
    }
    if index < wins.len() {
        return Ok(wins[index].object_ref.clone());
    }
    Err(core::taxonomy_error(
        core::ERR_NOT_MATCHED,
        format!("window '{title}' (index {index}) not found — re-enumerate windows"),
    ))
}

async fn resolve_focused_window(
    conn: &AccessibilityConnection,
) -> Result<ResolvedWindow, anyhow::Error> {
    let apps = enumerate_apps(conn).await?;
    let mut matches = Vec::new();
    for (app_ref, app_info) in &apps {
        for w in app_window_handles(app_ref, conn).await {
            if w.active || w.focused {
                matches.push(ResolvedWindow {
                    object_ref: w.object_ref,
                    app_info: app_info.clone(),
                    title: w.title,
                    index: w.index,
                    surface: w.surface,
                });
            }
        }
    }
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "no focused window found (no window reports active/focused) — enumerate apps/windows first",
        )),
        n => Err(core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            format!(
                "{n} windows report active/focused state — enumerate apps/windows to pick a surface"
            ),
        )),
    }
}

/// Resolve a non-`Screen` target to its window object + owning app.
async fn resolve_window(
    conn: &AccessibilityConnection,
    target: &TargetSpec,
) -> Result<(ObjectRefOwned, AppInfo), anyhow::Error> {
    match target {
        TargetSpec::Screen => Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "observe targets an accessibility surface — use screenshot for the screen",
        )),
        TargetSpec::Focused => {
            let w = resolve_focused_window(conn).await?;
            Ok((w.object_ref, w.app_info))
        }
        TargetSpec::Window {
            app_pid,
            app_name,
            title,
            index,
        } => {
            let app_ref = resolve_app_object(conn, app_name, *app_pid).await?;
            let win_ref = resolve_window_in_app(conn, &app_ref, title, *index).await?;
            let app_info = app_info_of(conn, &app_ref).await?;
            Ok((win_ref, app_info))
        }
    }
}

// ── Accessibility tree walk ──────────────────────────────────────────────

#[async_recursion]
async fn walk_node(
    object_ref: &ObjectRefOwned,
    conn: &AccessibilityConnection,
    depth: usize,
    handles: &mut Vec<ObjectRefOwned>,
    nodes: &mut usize,
) -> Option<UiNode> {
    if *nodes >= core::MAX_RENDER_NODES {
        return None;
    }
    let accessible = object_ref
        .as_accessible_proxy(conn.connection())
        .await
        .ok()?;
    *nodes += 1;
    handles.push(object_ref.clone());

    let role = role_name(accessible.get_role().await.unwrap_or(Role::Unknown));
    let name = accessible.name().await.ok();
    let proxies = accessible.proxies().await.ok();

    let value = tree_value(proxies.as_ref(), &role).await;
    let frame = tree_frame(proxies.as_ref()).await;
    let actions = tree_actions(proxies.as_ref()).await;
    let focused = accessible
        .get_state()
        .await
        .map(|s| s.contains(AtspiState::Focused))
        .unwrap_or(false);

    let mut children = Vec::new();
    if depth < MAX_DEPTH {
        if let Ok(child_refs) = accessible.get_children().await {
            for child in child_refs {
                if *nodes >= core::MAX_RENDER_NODES {
                    break;
                }
                if let Some(child_node) = walk_node(&child, conn, depth + 1, handles, nodes).await {
                    children.push(child_node);
                }
            }
        }
    }

    Some(UiNode {
        role,
        name,
        value,
        frame,
        actions,
        focused,
        children,
    })
}

async fn build_tree(
    root_ref: &ObjectRefOwned,
    conn: &AccessibilityConnection,
) -> (UiNode, Vec<ObjectRefOwned>) {
    let mut handles = Vec::new();
    let mut nodes = 0usize;
    let node = walk_node(root_ref, conn, 0, &mut handles, &mut nodes)
        .await
        .unwrap_or_else(|| UiNode {
            role: "window".to_string(),
            name: None,
            value: None,
            frame: None,
            actions: Vec::new(),
            focused: false,
            children: Vec::new(),
        });
    (node, handles)
}

async fn tree_value(proxies: Option<&Proxies<'_>>, role: &str) -> Option<String> {
    let proxies = proxies?;
    if !is_text_role(role) {
        return None;
    }
    let text = proxies.text().await.ok()?;
    let count = text.character_count().await.ok()?;
    if count <= 0 || count > MAX_VALUE_CHARS {
        return None;
    }
    text.get_text(0, count).await.ok()
}

async fn tree_frame(proxies: Option<&Proxies<'_>>) -> Option<(f64, f64, f64, f64)> {
    let proxies = proxies?;
    let comp = proxies.component().await.ok()?;
    let (x, y, w, h) = comp.get_extents(CoordType::Window).await.ok()?;
    // Drop zero-size (hidden/unmapped) nodes like the AX backend does.
    if w <= 0 || h <= 0 {
        return None;
    }
    Some((f64::from(x), f64::from(y), f64::from(w), f64::from(h)))
}

async fn tree_actions(proxies: Option<&Proxies<'_>>) -> Vec<String> {
    let Some(proxies) = proxies else {
        return Vec::new();
    };
    let Ok(action) = proxies.action().await else {
        return Vec::new();
    };
    let Ok(actions) = action.get_actions().await else {
        return Vec::new();
    };
    actions
        .into_iter()
        .filter_map(|a| (!a.name.is_empty()).then(|| a.name.to_lowercase()))
        .collect()
}

// ── Element actions ──────────────────────────────────────────────────────

async fn press_element(proxies: &Proxies<'_>) -> Result<(), anyhow::Error> {
    let action = proxies
        .action()
        .await
        .map_err(|e| dbus_err(e, "read actions"))?;
    let actions = action
        .get_actions()
        .await
        .map_err(|e| dbus_err(e, "list actions"))?;
    let idx = actions
        .iter()
        .position(|a| a.name.eq_ignore_ascii_case("press"))
        .or_else(|| {
            actions
                .iter()
                .position(|a| a.name.eq_ignore_ascii_case("click"))
        })
        .or_else(|| {
            actions
                .iter()
                .position(|a| a.name.eq_ignore_ascii_case("open"))
        });
    let Some(idx) = idx else {
        let available = actions
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            format!("element has no press/open action; available actions: {available}"),
        ));
    };
    let ok = action
        .do_action(idx as i32)
        .await
        .map_err(|e| dbus_err(e, "perform action"))?;
    if ok {
        Ok(())
    } else {
        Err(anyhow!("action '{}' reported failure", actions[idx].name))
    }
}

async fn set_element_value(
    proxies: &Proxies<'_>,
    role: &str,
    text: &str,
) -> Result<(), anyhow::Error> {
    if role.to_lowercase().contains("password") {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element is a password field — SetValue refused (use raw input to type)",
        ));
    }
    if let Ok(editable) = proxies.editable_text().await {
        let ok = editable
            .set_text_contents(text)
            .await
            .map_err(|e| dbus_err(e, "set text contents"))?;
        return if ok {
            Ok(())
        } else {
            Err(anyhow!("set-text-contents returned false"))
        };
    }
    if let Ok(value) = proxies.value().await {
        if let Ok(num) = text.trim().parse::<f64>() {
            value
                .set_current_value(num)
                .await
                .map_err(|e| dbus_err(e, "set current value"))?;
            return Ok(());
        }
    }
    Err(core::taxonomy_error(
        core::ERR_UNSUPPORTED,
        "element has no editable-text or value interface — SetValue not available; use raw input",
    ))
}

// ── Raw input (enigo / XTEST) ────────────────────────────────────────────

fn enigo_mod_key(m: Modifier) -> Key {
    match m {
        Modifier::Cmd => Key::Meta,
        Modifier::Ctrl => Key::Control,
        Modifier::Alt => Key::Alt,
        Modifier::Shift => Key::Shift,
    }
}

fn enigo_button(b: MouseButton) -> Button {
    match b {
        MouseButton::Left => Button::Left,
        MouseButton::Right => Button::Right,
        MouseButton::Middle => Button::Middle,
    }
}

fn char_key(c: char) -> Option<Key> {
    match c {
        ' ' => Some(Key::Space),
        '\t' => Some(Key::Tab),
        _ if c.is_ascii() && !c.is_control() => Some(Key::Unicode(c)),
        _ => None,
    }
}

/// Key-chord name → enigo [`Key`]. Single printable chars (letters, digits,
/// punctuation) map to `Key::Unicode`, which the X keymap synthesizes; named
/// keys map to their dedicated variant.
fn enigo_key(name: &str) -> Result<Key, String> {
    if let Ok(c) = name.parse::<char>()
        && let Some(key) = char_key(c)
    {
        return Ok(key);
    }
    let k = name.to_lowercase();
    let key = match k.as_str() {
        "return" | "enter" => Key::Return,
        "tab" => Key::Tab,
        "space" => Key::Space,
        "delete" | "backspace" => Key::Backspace,
        "forwarddelete" => Key::Delete,
        "escape" => Key::Escape,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" => Key::PageUp,
        "pagedown" => Key::PageDown,
        "up" => Key::UpArrow,
        "down" => Key::DownArrow,
        "left" => Key::LeftArrow,
        "right" => Key::RightArrow,
        "capslock" => Key::CapsLock,
        "insert" => Key::Insert,
        "f1" => Key::F1,
        "f2" => Key::F2,
        "f3" => Key::F3,
        "f4" => Key::F4,
        "f5" => Key::F5,
        "f6" => Key::F6,
        "f7" => Key::F7,
        "f8" => Key::F8,
        "f9" => Key::F9,
        "f10" => Key::F10,
        "f11" => Key::F11,
        "f12" => Key::F12,
        "minus" => Key::Unicode('-'),
        "equal" => Key::Unicode('='),
        "comma" => Key::Unicode(','),
        "period" | "dot" => Key::Unicode('.'),
        "slash" => Key::Unicode('/'),
        "semicolon" => Key::Unicode(';'),
        "quote" => Key::Unicode('\''),
        "backslash" => Key::Unicode('\\'),
        "backtick" => Key::Unicode('`'),
        "leftbracket" => Key::Unicode('['),
        "rightbracket" => Key::Unicode(']'),
        _ => {
            return Err(format!(
                "unknown key '{name}' — supported keys: {KNOWN_KEYS}"
            ));
        }
    };
    Ok(key)
}

fn press_modifiers(enigo: &mut Enigo, mods: &[Modifier]) -> Result<(), anyhow::Error> {
    for m in mods {
        enigo
            .key(enigo_mod_key(*m), Direction::Press)
            .map_err(input_err)?;
    }
    Ok(())
}

fn release_modifiers(enigo: &mut Enigo, mods: &[Modifier]) -> Result<(), anyhow::Error> {
    for m in mods.iter().rev() {
        enigo
            .key(enigo_mod_key(*m), Direction::Release)
            .map_err(input_err)?;
    }
    Ok(())
}

fn scroll_params(direction: ScrollDirection, amount: u32) -> (i32, Axis) {
    let a = amount as i32;
    match direction {
        ScrollDirection::Down => (a, Axis::Vertical),
        ScrollDirection::Up => (-a, Axis::Vertical),
        ScrollDirection::Right => (a, Axis::Horizontal),
        ScrollDirection::Left => (-a, Axis::Horizontal),
    }
}

/// Normalized surface point → absolute X11 pixel point.
fn pixel_point(
    point: (f64, f64),
    geo: Option<&SurfaceGeometry>,
) -> Result<(i32, i32), anyhow::Error> {
    let geo = geo.ok_or_else(|| anyhow!("raw input needs surface geometry for this action"))?;
    let (x, y) = core::normalized_to_surface(point.0, point.1, geo)?;
    Ok((x.round() as i32, y.round() as i32))
}

fn input_err(e: enigo::InputError) -> anyhow::Error {
    anyhow!("input synthesis failed: {e}")
}

fn key_chord(enigo: &mut Enigo, chord: &str) -> Result<(), anyhow::Error> {
    let (mods, key) = core::parse_key_chord(chord)?;
    let key = enigo_key(&key).map_err(anyhow::Error::msg)?;
    press_modifiers(enigo, &mods)?;
    enigo.key(key, Direction::Click).map_err(input_err)?;
    release_modifiers(enigo, &mods)
}

fn raw_input_blocking(input: RawInput, geo: Option<&SurfaceGeometry>) -> Result<(), anyhow::Error> {
    let mut enigo =
        Enigo::new(&Settings::default()).map_err(|e| anyhow!("enigo init failed: {e}"))?;
    match input {
        RawInput::Click {
            point,
            button,
            double,
            modifiers,
        } => {
            let (x, y) = pixel_point(point, geo)?;
            enigo.move_mouse(x, y, Coordinate::Abs).map_err(input_err)?;
            let button = enigo_button(button);
            press_modifiers(&mut enigo, &modifiers)?;
            // XTEST synthesizes into whatever is under the pointer; the model
            // clicks first if focus is required (no activate-then-control).
            for _ in 0..(if double { 2 } else { 1 }) {
                enigo.button(button, Direction::Press).map_err(input_err)?;
                enigo
                    .button(button, Direction::Release)
                    .map_err(input_err)?;
            }
            release_modifiers(&mut enigo, &modifiers)
        }
        RawInput::TypeText { text } => enigo.text(&text).map_err(input_err),
        RawInput::KeyChord { chord } => key_chord(&mut enigo, &chord),
        RawInput::Scroll {
            point,
            direction,
            amount,
        } => {
            if let Some((nx, ny)) = point {
                let (x, y) = pixel_point((nx, ny), geo)?;
                enigo.move_mouse(x, y, Coordinate::Abs).map_err(input_err)?;
            }
            let (length, axis) = scroll_params(direction, amount);
            enigo.scroll(length, axis).map_err(input_err)
        }
        RawInput::Drag { from, to } => {
            let (fx, fy) = pixel_point(from, geo)?;
            let (tx, ty) = pixel_point(to, geo)?;
            enigo
                .button(Button::Left, Direction::Press)
                .map_err(input_err)?;
            for i in 1..=INTERP_STEPS {
                let t = i as f64 / INTERP_STEPS as f64;
                let x = (fx as f64 + (tx - fx) as f64 * t).round() as i32;
                let y = (fy as f64 + (ty - fy) as f64 * t).round() as i32;
                enigo.move_mouse(x, y, Coordinate::Abs).map_err(input_err)?;
                std::thread::sleep(Duration::from_millis(2));
            }
            enigo
                .button(Button::Left, Direction::Release)
                .map_err(input_err)
        }
    }
}

// ── Capture (xcap / X11) ────────────────────────────────────────────────

/// Composite all monitors into one buffer covering the virtual-desktop union.
/// Rect origins may be negative (offset); on overlaps the later monitor wins.
fn capture_screen_blocking() -> Result<Capture, anyhow::Error> {
    let monitors = xcap::Monitor::all()?;
    if monitors.is_empty() {
        anyhow::bail!("no monitors found");
    }
    let mut rects = Vec::with_capacity(monitors.len());
    for m in &monitors {
        let w = m.width()? as i32;
        let h = m.height()? as i32;
        rects.push((m.x()?, m.y()?, w, h));
    }
    let geo = surface_from_rects(&rects);
    let (out_w, out_h) = (geo.width.round() as usize, geo.height.round() as usize);
    if out_w == 0 || out_h == 0 {
        anyhow::bail!("virtual desktop has zero size");
    }
    let mut composite = vec![0u8; out_w * out_h * 4];
    for m in &monitors {
        let img = m.capture_image()?;
        let cap = Capture {
            width: img.width(),
            height: img.height(),
            rgba: img.into_raw(),
        };
        let dx = (m.x()? - geo.x.round() as i32).max(0) as usize;
        let dy = (m.y()? - geo.y.round() as i32).max(0) as usize;
        core::blit_rgba(&mut composite, out_w, out_h, &cap, dx, dy);
    }
    Ok(Capture {
        width: out_w as u32,
        height: out_h as u32,
        rgba: composite,
    })
}

fn capture_xcap_window(win: &xcap::Window) -> Result<Capture, anyhow::Error> {
    let img = win.capture_image()?;
    Ok(Capture {
        width: img.width(),
        height: img.height(),
        rgba: img.into_raw(),
    })
}

fn capture_window_blocking(pid: Option<u32>, title: &str) -> Result<Capture, anyhow::Error> {
    let windows = xcap::Window::all()?;
    let mut candidates: Vec<xcap::Window> = windows
        .into_iter()
        .filter(|w| pid.is_none_or(|p| w.pid().map(|wp| wp == p).unwrap_or(false)))
        .collect();
    if let Some(win) = candidates
        .iter()
        .find(|w| w.title().map(|t| t == title).unwrap_or(false))
    {
        return capture_xcap_window(win);
    }
    let win = candidates.into_iter().next().ok_or_else(|| {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            "no matching capture window — re-enumerate windows",
        )
    })?;
    capture_xcap_window(&win)
}

fn screen_surface_blocking() -> Result<SurfaceGeometry, anyhow::Error> {
    let monitors = xcap::Monitor::all()?;
    if monitors.is_empty() {
        anyhow::bail!("no monitors found");
    }
    let mut rects = Vec::with_capacity(monitors.len());
    for m in &monitors {
        let w = m.width()? as i32;
        let h = m.height()? as i32;
        rects.push((m.x()?, m.y()?, w, h));
    }
    Ok(surface_from_rects(&rects))
}

// ── Backend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl Backend for LinuxBackend {
    fn accessibility_available(&self) -> bool {
        let gui =
            std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some();
        let bus = std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_some()
            || std::env::var_os("XDG_RUNTIME_DIR")
                .is_some_and(|d| std::path::PathBuf::from(d).join("bus").exists());
        gui && bus
    }

    async fn capture_available(&self) -> bool {
        with_block_in_place(|| xcap::Monitor::all().map(|m| !m.is_empty()).unwrap_or(false))
    }

    fn capture_unavailable_error(&self) -> anyhow::Error {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            "screen capture is unavailable (Wayland session without portal support in v1, or \
             X11 capture init failed); accessibility observe/act remain available",
        )
    }

    async fn list_apps(&self) -> Result<Vec<AppInfo>, anyhow::Error> {
        let conn = self.conn().await?;
        let mut out: Vec<AppInfo> = enumerate_apps(&conn)
            .await?
            .into_iter()
            .map(|(_, info)| info)
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    async fn list_windows(&self, app: Option<&AppInfo>) -> Result<Vec<WindowInfo>, anyhow::Error> {
        let conn = self.conn().await?;
        let mut out = Vec::new();
        for (app_ref, app_info) in enumerate_apps(&conn).await? {
            if let Some(filter) = app {
                let name_ok = filter.name.eq_ignore_ascii_case(&app_info.name);
                let pid_ok = filter.pid.is_some() && filter.pid == app_info.pid;
                if !name_ok && !pid_ok {
                    continue;
                }
            }
            for w in app_window_handles(&app_ref, &conn).await {
                out.push(WindowInfo {
                    app_pid: app_info.pid,
                    app_name: app_info.name.clone(),
                    title: w.title,
                    index: w.index,
                    surface: w.surface,
                });
            }
        }
        Ok(out)
    }

    async fn focused_window(&self) -> Result<WindowInfo, anyhow::Error> {
        let conn = self.conn().await?;
        let w = resolve_focused_window(&conn).await?;
        Ok(WindowInfo {
            app_pid: w.app_info.pid,
            app_name: w.app_info.name,
            title: w.title,
            index: w.index,
            surface: w.surface,
        })
    }

    async fn surface_geometry(
        &self,
        target: &TargetSpec,
    ) -> Result<SurfaceGeometry, anyhow::Error> {
        match target {
            TargetSpec::Screen => with_block_in_place(screen_surface_blocking),
            other => {
                let conn = self.conn().await?;
                let (window_ref, _app) = resolve_window(&conn, other).await?;
                let proxy = window_ref
                    .as_accessible_proxy(conn.connection())
                    .await
                    .map_err(|e| dbus_err(e, "open window"))?;
                window_screen_extents(&proxy)
                    .await
                    .ok_or_else(|| anyhow!("window geometry unavailable"))
            }
        }
    }

    async fn observe(&self, target: &TargetSpec) -> Result<Observation, anyhow::Error> {
        let conn = self.conn().await?;
        let (window_ref, app_info) = match target {
            TargetSpec::Screen => {
                return Err(core::taxonomy_error(
                    core::ERR_UNSUPPORTED,
                    "observe needs a window target — use apps/windows to pick one",
                ));
            }
            other => resolve_window(&conn, other).await?,
        };
        let proxy = window_ref
            .as_accessible_proxy(conn.connection())
            .await
            .map_err(|e| dbus_err(e, "open window"))?;
        let surface = window_screen_extents(&proxy)
            .await
            .ok_or_else(|| anyhow!("window geometry unavailable"))?;
        let title = proxy.name().await.ok();
        let (root, _handles) = build_tree(&window_ref, &conn).await;
        let app_name = match target {
            TargetSpec::Window { app_name, .. } => app_name.clone(),
            _ => app_info.name,
        };
        Ok(Observation {
            app_name,
            window_title: title,
            surface,
            root,
        })
    }

    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> Result<(), anyhow::Error> {
        let conn = self.conn().await?;
        let (window_ref, _app) = match target {
            TargetSpec::Screen => {
                return Err(core::taxonomy_error(
                    core::ERR_UNSUPPORTED,
                    "act targets a window — use apps/windows to pick one",
                ));
            }
            other => resolve_window(&conn, other).await?,
        };
        let (root, handles) = build_tree(&window_ref, &conn).await;
        let matched = match core::resolve_locator(&root, locator) {
            core::LocatorMatch::Path(node) | core::LocatorMatch::Unique(node) => node,
            core::LocatorMatch::Ambiguous => {
                return Err(core::taxonomy_error(
                    core::ERR_AMBIGUOUS_LOCATOR,
                    "locator matches multiple elements — re-observe and pick a more specific ref",
                ));
            }
            core::LocatorMatch::NotFound => {
                return Err(core::taxonomy_error(
                    core::ERR_NOT_MATCHED,
                    "element no longer matches its locator — re-observe",
                ));
            }
        };
        if handles.is_empty() {
            return Err(core::taxonomy_error(
                core::ERR_STALE_ELEMENT,
                "element tree is empty — the window may have closed; re-observe",
            ));
        }
        let element_ref = &handles[core::pre_order_index(&root, matched)];
        let element = element_ref
            .as_accessible_proxy(conn.connection())
            .await
            .map_err(|e| dbus_err(e, "open element"))?;
        let proxies = element
            .proxies()
            .await
            .map_err(|e| dbus_err(e, "enumerate element interfaces"))?;
        let role = role_name(element.get_role().await.unwrap_or(Role::Unknown));
        match act {
            ElementAct::Press => press_element(&proxies).await,
            ElementAct::SetValue { text } => set_element_value(&proxies, &role, &text).await,
        }
    }

    async fn raw_input(&self, target: &TargetSpec, input: RawInput) -> Result<(), anyhow::Error> {
        if wayland_only_env() {
            return Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                "Wayland sessions do not support raw input synthesis (no XTEST); use ref-based \
                 accessibility actions where the app exposes them",
            ));
        }
        let geo = match &input {
            RawInput::Click { .. }
            | RawInput::Scroll { point: Some(_), .. }
            | RawInput::Drag { .. } => Some(self.surface_geometry(target).await?),
            _ => None,
        };
        with_block_in_place(|| raw_input_blocking(input, geo.as_ref()))
    }

    async fn cursor_position(&self) -> Result<(f64, f64), anyhow::Error> {
        if wayland_only_env() {
            return Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                "global pointer position is not exposed on Wayland",
            ));
        }
        with_block_in_place(|| {
            let enigo =
                Enigo::new(&Settings::default()).map_err(|e| anyhow!("enigo init failed: {e}"))?;
            let (x, y) = enigo.location().map_err(input_err)?;
            Ok((f64::from(x), f64::from(y)))
        })
    }

    async fn capture(&self, target: &TargetSpec) -> Result<Capture, anyhow::Error> {
        match target {
            TargetSpec::Screen => with_block_in_place(capture_screen_blocking),
            other => {
                let conn = self.conn().await?;
                let (window_ref, app_info) = resolve_window(&conn, other).await?;
                let proxy = window_ref
                    .as_accessible_proxy(conn.connection())
                    .await
                    .map_err(|e| dbus_err(e, "open window"))?;
                let title = proxy.name().await.unwrap_or_default();
                let pid = app_info.pid;
                with_block_in_place(|| capture_window_blocking(pid, &title))
            }
        }
    }
}

// ── Tests (pure logic only; no D-Bus / enigo / xcap) ─────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_name_covers_key_roles_and_falls_back() {
        assert_eq!(role_name(Role::Button), "pushbutton");
        assert_eq!(role_name(Role::Frame), "frame");
        assert_eq!(role_name(Role::Window), "window");
        assert_eq!(role_name(Role::Dialog), "dialog");
        assert_eq!(role_name(Role::CheckBox), "checkbox");
        assert_eq!(role_name(Role::PageTab), "tab");
        assert_eq!(role_name(Role::Entry), "textfield");
        // Unknown / unmapped enum variant → Debug name lowercased.
        assert_eq!(role_name(Role::Unknown), "unknown");
        assert_eq!(role_name(Role::Canvas), "canvas");
    }

    #[test]
    fn wayland_gating_requires_no_x11() {
        assert!(is_wayland_only(true, false));
        assert!(!is_wayland_only(true, true)); // XWayland: DISPLAY set → X11 synthesis works
        assert!(!is_wayland_only(false, true));
        assert!(!is_wayland_only(false, false));
    }

    #[test]
    fn surface_union_handles_negative_origins() {
        let rects = [
            (0, 0, 1920, 1080),
            (1920, 0, 1920, 1080),
            (0, -100, 800, 600),
        ];
        let geo = surface_from_rects(&rects);
        assert_eq!(
            (geo.x, geo.y, geo.width, geo.height),
            (0.0, -100.0, 3840.0, 1180.0)
        );
        let (px, py) = core::normalized_to_surface(500.0, 500.0, &geo).unwrap();
        assert!((px - 1920.0).abs() < 1e-9, "x = {px}");
        assert!((py - 490.0).abs() < 1e-9, "y = {py}");
    }
}
