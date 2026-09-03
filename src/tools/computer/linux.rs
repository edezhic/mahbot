//! Linux [`Backend`] for the Computer tool: AT-SPI2 accessibility (async D-Bus,
//! observe/act), XTEST input synthesis via enigo, X11 stills via x11rb and
//! Wayland stills via the Screenshot portal over zbus. All Rust-only: no
//! system capture packages beyond the session's own daemons.
//!
//! Surface geometry is PIXELS on Linux: X11 has no stable logical-point space and
//! mixed-DPI makes physical↔logical translation unreliable — this is precisely
//! why the tool is accessibility-first here.

#![cfg(target_os = "linux")]
// A handful of pedantic cast lints are inherent to the C ABI-style integer
// geometry the atspi/X11 crates expose (i32 screen coords, u32 pixel dims).
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
use crate::util::{UnwrapPoison, with_block_in_place};
use anyhow::anyhow;
use async_recursion::async_recursion;
use atspi::proxy::accessible::{AccessibleProxy, ObjectRefExt};
use atspi::proxy::proxy_ext::{Proxies, ProxyExt};
use atspi::{AccessibilityConnection, CoordType, ObjectRefOwned, Role, State as AtspiState};
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use futures_util::StreamExt as _;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tracing::debug;
use x11rb::connection::Connection as _;
use x11rb::protocol::{randr, xproto};
use x11rb::rust_connection::RustConnection;

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

    /// Screen-channel still: X11 full capture or the single portal attempt,
    /// caching dims for zoom. Window captures crop out of this in `capture`.
    async fn capture_screen_channel(&self) -> Result<(Capture, SurfaceGeometry), anyhow::Error> {
        if wayland_env() {
            let cap = portal_screenshot().await?;
            let geo = SurfaceGeometry {
                x: 0.0,
                y: 0.0,
                width: f64::from(cap.width),
                height: f64::from(cap.height),
            };
            // X11 geometry is always live; only Wayland zoom reads the cache.
            *SCREEN_DIMS.lock().unwrap_poison() = Some((cap.width, cap.height));
            Ok((cap, geo))
        } else {
            with_block_in_place(x11_capture_screen)
        }
    }
}

// ── Pure helpers ─────────────────────────────────────────────────────────

fn wayland_env() -> bool {
    // XWayland (both `WAYLAND_DISPLAY` and `DISPLAY` set) still counts as
    // Wayland: XTEST would only drive XWayland windows while native Wayland
    // windows ignore it, so raw input must degrade.
    std::env::var_os("WAYLAND_DISPLAY").is_some()
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

// ── X11 stills (x11rb) ───────────────────────────────────────────────────
// Blocking throughout; every entry runs inside `with_block_in_place`.
// Per-CRTC `GetImage` tiles composited over the RandR union (negative origins
// kept); screen→drawable mapping is `coord − union_min` — the root pixmap
// spans the bounding box, so capture offsets are union-relative while the
// geometry keeps the screen-space origin for coordinate mapping.

/// Monitor rects `(x, y, w, h)` from the RandR CRTC union. Any RandR failure
/// or zero-area result falls back to the root window geometry.
fn x11_monitor_rects(conn: &RustConnection, root: xproto::Window) -> Vec<(i32, i32, i32, i32)> {
    let resources = randr::get_screen_resources_current(conn, root)
        .ok()
        .and_then(|c| c.reply().ok())
        .map(|r| (r.crtcs, r.config_timestamp))
        .or_else(|| {
            randr::get_screen_resources(conn, root)
                .ok()
                .and_then(|c| c.reply().ok())
                .map(|r| (r.crtcs, r.config_timestamp))
        });
    if let Some((crtcs, stamp)) = resources {
        let mut rects = Vec::with_capacity(crtcs.len());
        for crtc in crtcs {
            let Some(info) = randr::get_crtc_info(conn, crtc, stamp)
                .ok()
                .and_then(|c| c.reply().ok())
            else {
                continue;
            };
            if info.mode == 0 || info.width == 0 || info.height == 0 {
                continue;
            }
            rects.push((
                i32::from(info.x),
                i32::from(info.y),
                i32::from(info.width),
                i32::from(info.height),
            ));
        }
        if !rects.is_empty() {
            return rects;
        }
    }
    xproto::get_geometry(conn, root)
        .ok()
        .and_then(|c| c.reply().ok())
        .map(|g| vec![(0, 0, i32::from(g.width), i32::from(g.height))])
        .unwrap_or_default()
}

/// Virtual-desktop union over the RandR CRTC rects.
fn x11_screen_geometry() -> Result<SurfaceGeometry, anyhow::Error> {
    let (conn, screen) = x11rb::connect(None)
        .map_err(|e| core::taxonomy_error(core::ERR_DEGRADED, format!("no X11 display: {e}")))?;
    let geo = surface_from_rects(&x11_monitor_rects(&conn, conn.setup().roots[screen].root));
    if geo.width <= 0.0 || geo.height <= 0.0 {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "X11 capture failed: no monitor geometry",
        ));
    }
    Ok(geo)
}

/// Full virtual-desktop still: per-CRTC `GetImage` tiles (row-tiled to the
/// server's max request size) composited over the RandR union via
/// `core::blit_rgba`. The X11 root pixmap spans the RandR bounding box with
/// drawable origin == union min (a pixmap has no negative indices), so
/// screen→drawable mapping is always `coord - union_min`: one global
/// coordinate system, no mixed spaces.
fn x11_capture_screen() -> Result<(Capture, SurfaceGeometry), anyhow::Error> {
    let (conn, screen) = x11rb::connect(None)
        .map_err(|e| core::taxonomy_error(core::ERR_DEGRADED, format!("no X11 display: {e}")))?;
    let root = conn.setup().roots[screen].root;
    let lsb_first = conn.setup().image_byte_order == xproto::ImageOrder::LSB_FIRST;
    let cap_bytes = conn
        .maximum_request_bytes()
        .min(4 * 1024 * 1024)
        .max(64 * 1024);
    let rects = x11_monitor_rects(&conn, root);
    let geo = surface_from_rects(&rects);
    let (w, h) = (geo.width.round() as u32, geo.height.round() as u32);
    if w == 0 || h == 0 {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "X11 capture failed: empty virtual desktop",
        ));
    }
    // `GetImage` takes u16 dims and i16 offsets — the union must fit `i16::MAX`.
    if w > i16::MAX as u32 || h > i16::MAX as u32 {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "X11 capture failed: virtual desktop too large for the X11 protocol",
        ));
    }
    let (ox, oy) = (geo.x.round() as i32, geo.y.round() as i32);
    let (out_w, out_h) = (w as usize, h as usize);
    let mut rgba = vec![0u8; out_w * out_h * 4];
    for &(rx, ry, rw, rh) in &rects {
        if rw <= 0 || rh <= 0 {
            continue;
        }
        let tile_w = rw as u32;
        let rows_per_tile = (cap_bytes / (tile_w as usize * 4)).max(1) as u32;
        let mut y = 0i32;
        while y < rh {
            let tile_h = rows_per_tile.min((rh - y) as u32);
            // Range-checked by the union guard above: every origin/dim passed
            // is within `[0, union]`, so the `as i16/u16/usize` casts below
            // cannot silently wrap.
            let tile_reply = xproto::get_image(
                &conn,
                xproto::ImageFormat::Z_PIXMAP,
                root,
                (rx - ox) as i16,
                (ry - oy + y) as i16,
                tile_w as u16,
                tile_h as u16,
                u32::MAX,
            )
            .ok()
            .and_then(|c| c.reply().ok());
            let Some(reply) = tile_reply else {
                return Err(core::taxonomy_error(
                    core::ERR_DEGRADED,
                    "X11 capture failed",
                ));
            };
            let tile =
                core::x11_pixels_to_rgba(&reply.data, tile_w, tile_h, reply.depth, lsb_first)?;
            let tile_cap = Capture {
                width: tile_w,
                height: tile_h,
                rgba: tile,
            };
            core::blit_rgba(
                &mut rgba,
                out_w,
                out_h,
                &tile_cap,
                (rx - ox) as usize,
                (ry - oy + y) as usize,
            );
            y += tile_h as i32;
        }
    }
    Ok((
        Capture {
            width: w,
            height: h,
            rgba,
        },
        geo,
    ))
}

// ── Wayland stills (Screenshot portal over zbus) ──────────────────────────
// One Screenshot attempt with short timeouts; deny/timeout/missing portal
// degrades to accessibility-only — no retries, no prompt spam.

/// Last known screen pixels, stored on every successful `Screen` capture so
/// zoom can size itself without triggering a second portal dialog.
static SCREEN_DIMS: Mutex<Option<(u32, u32)>> = Mutex::new(None);

static PORTAL_TOKEN_SEQ: AtomicU64 = AtomicU64::new(0);

const PORTAL_BUS: &str = "org.freedesktop.portal.Desktop";
const PORTAL_PATH: &str = "/org/freedesktop/portal/desktop";
const PORTAL_IFACE: &str = "org.freedesktop.portal.Screenshot";

/// The session bus has a portal owner. Never pops a dialog — safe as a probe.
/// Uses an untyped proxy: the generated `DBus` proxy trait is crate-private in
/// zbus, so `NameHasOwner` is called by hand.
async fn portal_available() -> bool {
    tokio::time::timeout(Duration::from_secs(3), async {
        let conn = zbus::Connection::session().await.ok()?;
        let proxy = zbus::Proxy::new(
            &conn,
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
        )
        .await
        .ok()?;
        let owned: bool = proxy.call("NameHasOwner", &PORTAL_BUS).await.ok()?;
        Some(owned)
    })
    .await
    .ok()
    .flatten()
    .unwrap_or(false)
}

/// Single Screenshot-portal attempt: subscribe for the `Response` signal first
/// (a fast compositor can't beat the subscription), then call, then wait.
async fn portal_screenshot() -> Result<Capture, anyhow::Error> {
    let conn = zbus::Connection::session().await.map_err(|e| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "Screenshot portal missing (no session bus: {e}) — no manual setup needed; \
                 accessibility observe/act remain available"
            ),
        )
    })?;
    let rule = zbus::MatchRule::builder()
        .msg_type(zbus::message::Type::Signal)
        .interface("org.freedesktop.portal.Request")?
        .member("Response")?
        .build();
    let mut stream = zbus::MessageStream::for_match_rule(rule, &conn, None)
        .await
        .map_err(|e| {
            core::taxonomy_error(core::ERR_DEGRADED, format!("portal subscribe failed: {e}"))
        })?;
    let token = format!(
        "mahbot_{}_{}",
        std::process::id(),
        PORTAL_TOKEN_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let mut options = HashMap::new();
    options.insert("handle_token", zbus::zvariant::Value::from(token));
    options.insert("modal", zbus::zvariant::Value::from(false));
    options.insert("interactive", zbus::zvariant::Value::from(false));
    let proxy = zbus::Proxy::new(&conn, PORTAL_BUS, PORTAL_PATH, PORTAL_IFACE)
        .await
        .map_err(|e| {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                format!("Screenshot portal unavailable ({e}); accessibility observe/act remain available"),
            )
        })?;
    let (req_path,): (zbus::zvariant::OwnedObjectPath,) = tokio::time::timeout(
        Duration::from_secs(10),
        proxy.call("Screenshot", &("", options)),
    )
    .await
    .map_err(|_| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            "Screenshot portal call timed out; accessibility observe/act remain available",
        )
    })?
    .map_err(|e| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "Screenshot portal call failed ({e}); accessibility observe/act remain available"
            ),
        )
    })?;
    let want = req_path.as_str().to_owned();
    let wait = async {
        while let Some(msg) = stream.next().await {
            let msg = msg.map_err(|e| {
                core::taxonomy_error(
                    core::ERR_DEGRADED,
                    format!("portal response stream failed: {e}"),
                )
            })?;
            if !msg
                .header()
                .path()
                .is_some_and(|p| p.as_str() == want.as_str())
            {
                continue;
            }
            let (code, results): (u32, HashMap<String, zbus::zvariant::OwnedValue>) =
                msg.body().deserialize().map_err(|e| {
                    core::taxonomy_error(
                        core::ERR_DEGRADED,
                        format!("portal response decode failed: {e}"),
                    )
                })?;
            return portal_response(code, results).await;
        }
        Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "portal closed the Screenshot response stream; accessibility observe/act remain available",
        ))
    };
    tokio::time::timeout(Duration::from_secs(15), wait)
        .await
        .map_err(|_| {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                "Screenshot response timed out; accessibility observe/act remain available",
            )
        })?
}

/// `Response` signal body → pixels. Response 1 is an in-dialog deny/dismiss:
/// opt-in consent, no auto-retry.
async fn portal_response(
    code: u32,
    results: HashMap<String, zbus::zvariant::OwnedValue>,
) -> Result<Capture, anyhow::Error> {
    match code {
        0 => {
            let uri_value = results.get("uri").ok_or_else(|| {
                core::taxonomy_error(core::ERR_DEGRADED, "portal Screenshot reply had no uri")
            })?;
            let uri = String::try_from(uri_value.try_clone().map_err(|e| {
                core::taxonomy_error(core::ERR_DEGRADED, format!("portal uri unreadable: {e}"))
            })?)
            .map_err(|e| {
                core::taxonomy_error(core::ERR_DEGRADED, format!("portal uri not a string: {e}"))
            })?;
            let path = core::portal_uri_to_path(&uri)?;
            let read = tokio::fs::read(&path).await;
            tokio::fs::remove_file(&path).await.ok();
            let bytes = read.map_err(|e| {
                core::taxonomy_error(
                    core::ERR_DEGRADED,
                    format!("portal screenshot unreadable: {e}"),
                )
            })?;
            let img =
                with_block_in_place(move || image::load_from_memory(&bytes)).map_err(|e| {
                    core::taxonomy_error(
                        core::ERR_DEGRADED,
                        format!("portal screenshot undecodable: {e}"),
                    )
                })?;
            let rgba = img.to_rgba8();
            Ok(Capture {
                width: rgba.width(),
                height: rgba.height(),
                rgba: rgba.into_raw(),
            })
        }
        1 => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "screenshot dismissed or denied in the system dialog — screenshots are an opt-in \
             elevated permission, no auto-retry; accessibility observe/act remain available",
        )),
        _ => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "portal Screenshot failed (response {code}); accessibility observe/act remain available"
            ),
        )),
    }
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
        if wayland_env() {
            portal_available().await
        } else {
            with_block_in_place(|| x11_screen_geometry().is_ok())
        }
    }

    fn capture_unavailable_error(&self) -> anyhow::Error {
        if wayland_env() {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                "Wayland capture unavailable (Screenshot portal missing or denied) — no manual \
                 setup needed, screenshots are an opt-in elevated permission; accessibility \
                 observe/act remain available",
            )
        } else {
            core::taxonomy_error(
                core::ERR_DEGRADED,
                "X11 capture init failed — accessibility observe/act remain available",
            )
        }
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
            TargetSpec::Screen => {
                if wayland_env() {
                    // Zoom sizes itself from the last Screen capture so it never
                    // triggers a second portal dialog.
                    match *SCREEN_DIMS.lock().unwrap_poison() {
                        Some((w, h)) => Ok(SurfaceGeometry {
                            x: 0.0,
                            y: 0.0,
                            width: f64::from(w),
                            height: f64::from(h),
                        }),
                        None => Err(core::taxonomy_error(
                            core::ERR_DEGRADED,
                            "no cached Wayland screen size — capture the screen first; \
                             accessibility observe/act remain available",
                        )),
                    }
                } else {
                    with_block_in_place(x11_screen_geometry)
                }
            }
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
        if wayland_env() {
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
        if wayland_env() {
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

    /// Window stills show current visible pixels (occluders visible; use
    /// observe for the exact tree).
    async fn capture(&self, target: &TargetSpec) -> Result<Capture, anyhow::Error> {
        match target {
            TargetSpec::Screen => Ok(self.capture_screen_channel().await?.0),
            other => {
                let conn = self.conn().await?;
                let (window_ref, _app) = resolve_window(&conn, other).await?;
                let proxy = window_ref
                    .as_accessible_proxy(conn.connection())
                    .await
                    .map_err(|e| dbus_err(e, "open window"))?;
                let win = window_screen_extents(&proxy)
                    .await
                    .ok_or_else(|| anyhow!("window geometry unavailable"))?;
                let (screen_cap, screen_geo) = self.capture_screen_channel().await?;
                core::crop_rgba(&screen_cap, core::window_capture_rect(&win, &screen_geo)?)
            }
        }
    }
}

// ── Tests (pure logic only; no D-Bus / X11 / portal) ─────────────────────

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
