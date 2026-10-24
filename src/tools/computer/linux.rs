//! Linux [`Backend`] for the Computer tool: AT-SPI2 accessibility (async D-Bus,
//! observe/act), XTEST input synthesis via enigo, X11 stills via x11rb and
//! Wayland stills via the Screenshot portal over zbus. All Rust-only: no
//! system capture packages beyond the session's own daemons.
//!
//! # Which server this is
//!
//! Wayland is decided from the display server, not from an environment variable: an X
//! connection that advertises `XWAYLAND` is XWayland, i.e. a Wayland session, and one
//! without it is a real X server; `WAYLAND_DISPLAY` decides only a session with no X
//! connection at all. A Wayland session therefore gets stills through the portal and no
//! coordinate input, whether or not `WAYLAND_DISPLAY` is set.
//!
//! # Nothing left held, and nothing left behind
//!
//! Input synthesis is fire-and-forget: a gesture that stops partway reports what may still
//! be held or half-applied rather than failing as if nothing happened, and the buttons a
//! call pressed are let go on the way out — an error return or an unwind included, though a
//! call abandoned inside a request of the server releases nothing until that request
//! returns. The Screenshot portal's request is closed on every path, the timeout included;
//! the image file it hands over is read and left where the portal put it, because only a
//! file this tool created is ever deleted.
//!
//! # A locked session is refused
//!
//! Input into a locked session is synthesized and reported as delivered, and a still of
//! it is a still of the lock screen, so the session's own lock state is asked first:
//! systemd-logind's `LockedHint`, the one lock signal the system publishes for every
//! desktop. A session that does not answer is not refused; the X screen saver's own state
//! is deliberately not read, since a screen it blanked after an idle period with no
//! locker is the ordinary session input still reaches.
//!
//! Surface geometry is PIXELS on Linux: X11 has no stable logical-point space and
//! mixed-DPI makes physical↔logical translation unreliable — this is precisely
//! why the tool is accessibility-first here.
//!
//! A window's identity is its AT-SPI object reference (`unique bus name|object
//! path`): a stored key resolves to that very window or fails, never to another
//! window of the same application.
//!
//! A window still must show the window's current visible pixels, occluders
//! included, which reading the window's own drawable would not — so a window
//! still is a crop of the screen capture. AT-SPI extents live in the accessibility
//! coordinate space, which need not be the capture's pixels: the crop is taken
//! only after the two are reconciled by `core::a11y_scale`; on X11, whose extents
//! are the root's own pixels, an unreconcilable pair falls back to 1:1, and on
//! Wayland the still is refused instead.
//!
//! Mahbot's own window publishes no AT-SPI surface, so it never appears in an
//! enumeration, has no element tree and no window still; screen-target input, key
//! chords and typing still reach it. Nothing here needs to know which process owns
//! a window.
//!
//! A read never presents a ceiling as completeness: a tree that stopped at one
//! records it in the observation's header, and an application whose windows cannot
//! be read is named as unread instead of appearing to have none. A node whose own
//! accessibility proxy cannot be read is skipped with its subtree — an accepted
//! limit of the AT-SPI walk.
//!
//! Every accessibility exchange is bounded: the connection is this backend's own,
//! opened with the product's method timeout ([`core::EXCHANGE_DEADLINE`]), so a target
//! that never answers costs one attribute or one child instead of the call. One
//! aggregate budget per call ([`core::Budget`]) bounds resolution — which walks every
//! application — and the walk alike, and each application gets a share of a listing so
//! a hung one cannot spend it. A connection the accessibility service closed is rebuilt
//! by the next call, so a restarted at-spi takes effect instead of leaving the
//! capability dead for the life of the process.
//!
//! Nor is anything sized by what the machine reports: a name or a value is cut at
//! the product's cap as it is read, and every still buffer is charged to the call's
//! allowance. The accounting and its reasoning live in [`core`].

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
    self, AppInfo, Backend, Capture, CaptureBudget, ElementAct, Impact, Locator, Modifier,
    MouseButton, Observation, RawInput, ScrollDirection, SurfaceGeometry, TargetSpec, Truncation,
    UiNode, UnreadWindows, WindowInfo, WindowKey, WindowListing,
};
use crate::util::{UnwrapPoison, with_block_in_place};
use anyhow::Context as _;
use anyhow::anyhow;
use async_recursion::async_recursion;
use atspi::proxy::accessible::{AccessibleProxy, ObjectRefExt};
use atspi::proxy::bus::BusProxy;
use atspi::proxy::proxy_ext::{Proxies, ProxyExt};
use atspi::{CoordType, ObjectRef, ObjectRefOwned, Role, State as AtspiState};
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use futures_util::StreamExt as _;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::AsyncReadExt as _;
use tracing::{debug, warn};
use x11rb::connection::{Connection as _, RequestConnection as _};
use x11rb::protocol::{randr, xproto};
use x11rb::rust_connection::RustConnection;

pub(crate) static LINUX_BACKEND: LinuxBackend = LinuxBackend {
    connection: tokio::sync::Mutex::const_new(None),
};

pub(crate) struct LinuxBackend {
    /// The a11y bus connection, cloned out per call (never held across a call
    /// body). `None` until one is established; a connection the service closed is
    /// dropped and rebuilt by the next call, and a failed attempt caches nothing.
    /// [`zbus::Connection`] carries the per-exchange deadline as its method timeout.
    connection: tokio::sync::Mutex<Option<zbus::Connection>>,
}

const MAX_DEPTH: usize = 25;
/// Cap on text read into a node's `value`: one character past what the renderer
/// shows. The tree dump is a map of the surface, not a copy of its text, so one
/// character more than the renderer displays is enough to know the value was cut —
/// and the renderer's own count states that cut in the header.
const VALUE_READ_CHARS: i32 = core::RENDER_VALUE_CHARS as i32 + 1;
const INTERP_STEPS: usize = 12;

/// What a refused or timed-out a11y connection ends in — the one message both
/// states share; the next call retries, so a restarted at-spi is picked up.
const CONN_DEGRADED_MSG: &str =
    "AT-SPI2 accessibility bus unavailable — is at-spi2 installed and the session bus running?";

/// The registry's well-known name, which the registry root is addressed by.
const A11Y_REGISTRY: &str = "org.a11y.atspi.Registry";

/// Fail a call whose connection died under it. Every attribute and child read
/// degrades to `None`/empty on a failure, so a bus that went away mid-read would
/// otherwise be reported as a window with almost nothing in it — the honest answer is
/// that the call never saw the target, and the next call re-establishes the link.
fn ensure_link(conn: &zbus::Connection) -> Result<(), anyhow::Error> {
    if conn.is_closed() {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "the accessibility connection was lost while the call ran, so what it did read is \
             incomplete — the next call reconnects; retry",
        ));
    }
    Ok(())
}

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
    /// The a11y bus connection, established on first use and rebuilt whenever the
    /// held one is closed (a killed `at-spi-bus-launcher` closes the socket — that
    /// is the evidence the service is gone). Establishment is bounded here because
    /// the session-bus round trip and the a11y handshake are not method calls, so
    /// the connection's own timeout does not cover them.
    async fn conn(&self) -> Result<zbus::Connection, anyhow::Error> {
        let mut held = self.connection.lock().await;
        if let Some(conn) = held.as_ref() {
            if conn.is_closed() {
                warn!("at-spi connection closed — rebuilding it for this call");
                *held = None;
            } else {
                return Ok(conn.clone());
            }
        }
        match tokio::time::timeout(core::EXCHANGE_DEADLINE, open_a11y_bus()).await {
            Ok(Ok(conn)) => {
                *held = Some(conn.clone());
                Ok(conn)
            }
            Ok(Err(e)) => {
                // Nothing is cached on failure: a later success must take effect.
                debug!(error = %e, "atspi connection failed");
                Err(core::taxonomy_error(core::ERR_DEGRADED, CONN_DEGRADED_MSG))
            }
            Err(_) => {
                debug!("atspi connection attempt timed out");
                Err(core::taxonomy_error(
                    core::ERR_DEGRADED,
                    format!(
                        "{CONN_DEGRADED_MSG} (connecting did not answer within {}s — the next call \
                         retries)",
                        core::EXCHANGE_DEADLINE.as_secs()
                    ),
                ))
            }
        }
    }

    /// The target's surface, resolved under the caller's budget, and named once it is
    /// resolved so a failure refers to the window rather than to the target as the model
    /// wrote it. A `Screen` target costs no accessibility exchange at all.
    async fn surface_geometry_with(
        &self,
        target: &TargetSpec,
        budget: &mut core::Budget,
    ) -> Result<SurfaceGeometry, anyhow::Error> {
        match target {
            TargetSpec::Screen => {
                if wayland() {
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
                let window = resolve_surface(&conn, other, budget).await?;
                let scale = a11y_pixel_scale(&conn, pixel_space(), budget).await?;
                Ok(core::scale_geometry(window.surface, scale))
            }
        }
    }

    /// Screen-channel still: X11 full capture or the single portal attempt,
    /// caching dims for zoom. Window stills crop out of this in `capture`, through
    /// [`window_still`] (which reconciles the accessibility and pixel spaces).
    /// Every buffer either route holds is charged to the call's `budget`.
    async fn capture_screen_channel(
        &self,
        budget: &CaptureBudget,
    ) -> Result<(Capture, SurfaceGeometry), anyhow::Error> {
        if wayland() {
            let cap = portal_screenshot(budget).await?;
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
            with_block_in_place(|| x11_capture_screen(budget))
        }
    }
}

// ── Accessibility bus ────────────────────────────────────────────────────

/// Open the a11y bus with the product's per-exchange deadline as the connection's own
/// method timeout. Every proxy here is built on this raw connection rather than on the
/// crate's convenience wrapper, because the timeout has to belong to the connection: only
/// then does a target that stops answering cost one exchange.
async fn open_a11y_bus() -> anyhow::Result<zbus::Connection> {
    let session = zbus::Connection::session()
        .await
        .context("connect to the session bus")?;
    let bus = BusProxy::new(&session)
        .await
        .context("open the a11y bus address proxy")?;
    let address: zbus::Address = bus
        .get_address()
        .await
        .context("read the a11y bus address")?
        .parse()
        .context("parse the a11y bus address")?;
    zbus::connection::Builder::address(address)
        .context("prepare the a11y connection")?
        .method_timeout(core::EXCHANGE_DEADLINE)
        .build()
        .await
        .context("connect to the a11y bus")
}

/// The registry root, built as atspi builds it: the registry's properties are
/// incompletely implemented, so they are never cached.
async fn registry_root(conn: &zbus::Connection) -> anyhow::Result<AccessibleProxy<'_>> {
    AccessibleProxy::builder(conn)
        .destination(A11Y_REGISTRY)
        .map_err(|e| dbus_err(e, "access registry root"))?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
        .map_err(|e| dbus_err(e, "access registry root"))
}

// ── Display server and session state ─────────────────────────────────────

/// The display server this process is talking to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DisplayServer {
    X11,
    Wayland,
}

static DISPLAY_SERVER: std::sync::OnceLock<DisplayServer> = std::sync::OnceLock::new();

/// systemd-logind, whose per-session `LockedHint` is the desktop-agnostic lock signal.
const LOGIND_DEST: &str = "org.freedesktop.login1";
const LOGIND_PATH: &str = "/org/freedesktop/login1";

/// How long the logind question may take, asked before every call that needs the
/// desktop: a system bus that stops answering costs this and not the call's own budget.
const STATE_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// The system bus, kept across calls the way the accessibility connection is: the probe
/// runs before every call that needs the desktop, so it must not pay a connect each time.
/// A bus that went away is reopened, and nothing is cached when the connect fails.
static SYSTEM_BUS: tokio::sync::Mutex<Option<zbus::Connection>> =
    tokio::sync::Mutex::const_new(None);

async fn system_bus() -> Option<zbus::Connection> {
    let mut held = SYSTEM_BUS.lock().await;
    if let Some(conn) = held.as_ref() {
        if !conn.is_closed() {
            return Some(conn.clone());
        }
        *held = None;
    }
    let conn = zbus::Connection::system().await.ok()?;
    *held = Some(conn.clone());
    Some(conn)
}

/// Which server this session is, from the probe the module header describes. The answer is
/// kept for the life of the process — a session does not change server under it — while a
/// probe that could not reach the server is retried by the next caller.
fn display_server() -> Option<DisplayServer> {
    if let Some(known) = DISPLAY_SERVER.get() {
        return Some(*known);
    }
    let probed = with_block_in_place(probe_display_server)?;
    let _ = DISPLAY_SERVER.set(probed);
    Some(probed)
}

fn probe_display_server() -> Option<DisplayServer> {
    if std::env::var_os("DISPLAY").is_some()
        && let Ok((conn, _screen)) = x11rb::connect(None)
    {
        let xwayland = xproto::query_extension(&conn, b"XWAYLAND")
            .ok()
            .and_then(|cookie| cookie.reply().ok())
            .is_some_and(|reply| reply.present);
        return Some(if xwayland {
            DisplayServer::Wayland
        } else {
            DisplayServer::X11
        });
    }
    std::env::var_os("WAYLAND_DISPLAY").map(|_| DisplayServer::Wayland)
}

/// Whether this is a Wayland session — the branch every still and coordinate map takes.
fn wayland() -> bool {
    display_server() == Some(DisplayServer::Wayland)
}

/// `LockedHint` of this process's logind session, read under [`STATE_PROBE_TIMEOUT`]:
/// logind is optional, so the answer is only ever what the session itself published.
async fn locked_reason() -> Option<String> {
    let locked = tokio::time::timeout(STATE_PROBE_TIMEOUT, async {
        let conn = system_bus().await?;
        let manager = zbus::Proxy::new(
            &conn,
            LOGIND_DEST,
            LOGIND_PATH,
            "org.freedesktop.login1.Manager",
        )
        .await
        .ok()?;
        // "self" is logind resolving the caller by pid, so a process that belongs to no
        // session of its own (a user service) is answered with an error — "cannot tell".
        let (session,): (zbus::zvariant::OwnedObjectPath,) =
            manager.call("GetSession", &("self",)).await.ok()?;
        let proxy = zbus::Proxy::new(
            &conn,
            LOGIND_DEST,
            session.as_str(),
            "org.freedesktop.login1.Session",
        )
        .await
        .ok()?;
        proxy.get_property::<bool>("LockedHint").await.ok()
    })
    .await
    .ok()
    .flatten()?;
    locked.then(|| "the session's lock screen is active".to_string())
}

// ── Pure helpers ─────────────────────────────────────────────────────────

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

/// What a D-Bus / atspi error string names. One classifier for both mappings:
/// [`dbus_err`] (reads, and actions that certainly did nothing) and
/// [`action_dbus_outcome`] (the action call itself) must agree on which failures
/// are transient, or the same bus failure would be a refusal in one and a silent
/// success in the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BusFault {
    /// Access denied / not authorized.
    Denied,
    /// The app or object is gone, or has no such method.
    Gone,
    /// The reply never arrived. Nothing says whether the call was performed.
    Transient,
    /// Anything else: passed through as a plain error.
    Other,
}

/// The single classification of an atspi/zbus error string, so the read mapping
/// and the action mapping below can never disagree about what is transient. A
/// timed-out call belongs with the dropped replies: the answer was never
/// received, which says nothing about whether it was performed.
///
/// The spellings are the ones the library really produces: a client-side method
/// timeout is `Error::InputOutput(TimedOut, "timed out")`, displayed as
/// `I/O error: timed out`; the bus's own error name for the same condition is
/// `org.freedesktop.DBus.Error.TimedOut`; and a connection the bus dropped under us
/// reads `I/O error: socket closed` (or `… connection reset by peer`). A dropped
/// connection says nothing about whether the request reached its owner either.
fn classify_bus_fault(msg: &str) -> BusFault {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("accessdenied")
        || lower.contains("notauthorized")
        || lower.contains("authfailed")
    {
        BusFault::Denied
    } else if lower.contains("serviceunknown")
        || lower.contains("unknownobject")
        || lower.contains("namehasnoowner")
        || lower.contains("unknownmethod")
    {
        BusFault::Gone
    } else if lower.contains("noreply")
        || lower.contains("timeout")
        || lower.contains("timedout")
        || lower.contains("timed out")
        || lower.contains("i/o error")
        || lower.contains("failed to connect")
    {
        BusFault::Transient
    } else {
        BusFault::Other
    }
}

/// Read error mapping: access denied → permission-denied, a gone app/object/method
/// → stale-element, a dropped reply → degraded, else plain. Reads only — an
/// action call maps through [`action_dbus_outcome`], because a dropped reply there
/// may have carried the action through.
fn dbus_err(err: impl std::fmt::Display, ctx: &str) -> anyhow::Error {
    let msg = err.to_string();
    bus_error(classify_bus_fault(&msg), &msg, ctx)
}

/// Build the failure for one bus fault. Split out so a caller that has already
/// classified the fault renders the same message without re-stringifying the error.
fn bus_error(fault: BusFault, msg: &str, ctx: &str) -> anyhow::Error {
    match fault {
        BusFault::Denied => {
            core::taxonomy_error(core::ERR_PERMISSION_DENIED, format!("{ctx}: {msg}"))
        }
        BusFault::Gone => core::taxonomy_error(
            core::ERR_STALE_ELEMENT,
            format!("{ctx}: {msg} — app or window closed; re-enumerate"),
        ),
        BusFault::Transient => core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("{ctx}: {msg} — transient bus failure, retry"),
        ),
        BusFault::Other => anyhow::anyhow!("{ctx}: {msg}"),
    }
}

/// Outcome of the ACTION call itself — never of the reads that precede it, which
/// stay [`dbus_err`] failures because nothing was performed yet. A transient bus
/// failure there may have carried the action through, so it is reported as
/// [`Impact::Unconfirmed`], naming the call that may have landed.
fn action_dbus_outcome(err: impl std::fmt::Display, ctx: &str) -> anyhow::Result<Impact> {
    let msg = err.to_string();
    match classify_bus_fault(&msg) {
        BusFault::Transient => Ok(Impact::Unconfirmed(format!(
            "the accessibility call '{ctx}' failed at the bus level: {msg}"
        ))),
        fault => Err(bus_error(fault, &msg, ctx)),
    }
}

/// Identity of a window: its AT-SPI object reference, stable for as long as that
/// window lives and never positional (`core::WindowKey`).
fn window_key(object_ref: &ObjectRefOwned) -> WindowKey {
    WindowKey::new(format!(
        "{}|{}",
        object_ref.name_as_str().unwrap_or_default(),
        object_ref.path_as_str()
    ))
}

// ── Registry / tree enumeration ──────────────────────────────────────────

struct WindowHandle {
    object_ref: ObjectRefOwned,
    key: WindowKey,
    surface: SurfaceGeometry,
    title: String,
    active: bool,
    focused: bool,
}

struct ResolvedWindow {
    object_ref: ObjectRefOwned,
    key: WindowKey,
    app_info: AppInfo,
    title: String,
    surface: SurfaceGeometry,
}

/// Root accessible's children = applications that expose accessibility. Apps
/// that don't simply don't appear — an accepted AX-first limitation.
async fn enumerate_apps(
    conn: &zbus::Connection,
    budget: &core::Budget,
) -> Result<Vec<(ObjectRefOwned, AppInfo)>, anyhow::Error> {
    let root = registry_root(conn).await?;
    let app_refs = root
        .get_children()
        .await
        .map_err(|e| dbus_err(e, "enumerate applications"))?;
    let mut out = Vec::new();
    for app_ref in app_refs {
        budget.check()?;
        let Some(accessible) = app_ref.as_accessible_proxy(conn).await.ok() else {
            continue;
        };
        let name = core::cap_label(&accessible.name().await.unwrap_or_default());
        if name.is_empty() {
            continue;
        }
        // atspi 0.30 has no Application::get_pid, so the pid stays unknown.
        out.push((app_ref, AppInfo { pid: None, name }));
    }
    // A listing that lost the bus mid-way would show as "no applications".
    ensure_link(conn)?;
    Ok(out)
}

async fn app_info_of(
    conn: &zbus::Connection,
    app_ref: &ObjectRefOwned,
) -> Result<AppInfo, anyhow::Error> {
    let accessible = app_ref
        .as_accessible_proxy(conn)
        .await
        .map_err(|e| dbus_err(e, "open application"))?;
    let name = core::cap_label(&accessible.name().await.unwrap_or_default());
    Ok(AppInfo { pid: None, name })
}

/// The two names a call addresses a window by: the application the target carries
/// (else the one resolution found) and the window's live title. Both are read BEFORE
/// the walk, because a walk that spends its budget has to name what it abandoned, and
/// the title is an exchange it can no longer afford then.
async fn surface_identity(
    target: &TargetSpec,
    window: &AccessibleProxy<'_>,
    resolved_app: &AppInfo,
) -> (String, Option<String>) {
    let app_name = target
        .app()
        .map_or_else(|| resolved_app.name.clone(), |(_, name)| name.to_string());
    (app_name, window.name().await.ok())
}

async fn resolve_app_object(
    conn: &zbus::Connection,
    app_name: &str,
    app_pid: Option<u32>,
    budget: &core::Budget,
) -> Result<ObjectRefOwned, anyhow::Error> {
    let apps = enumerate_apps(conn, budget).await?;
    let needle = app_name.to_lowercase();
    let by_name = apps
        .iter()
        .find(|(_, info)| info.name.to_lowercase() == needle);
    let by_pid = apps
        .iter()
        .find(|(_, info)| app_pid.is_some() && info.pid == app_pid);
    by_name
        .or(by_pid)
        .map(|(r, _)| r.clone())
        .ok_or_else(|| core::surface_gone(app_name))
}

/// Whether the element advertises more children than [`core::MAX_CHILDREN`]. Read
/// before the list itself: AT-SPI answers `GetChildren` in one exchange, so a target
/// that claims a near-2^31 child list (a spreadsheet's cell list is the classic case)
/// is refused here instead of being paid for. A count the target cannot produce is
/// not a claim, and the caller proceeds to the list.
async fn claims_too_many_children(accessible: &AccessibleProxy<'_>) -> bool {
    accessible
        .child_count()
        .await
        .is_ok_and(|count| count > core::MAX_CHILDREN as i32)
}

/// The app's window-role children (Window/Frame/Dialog): identity key, global
/// screen extents, title and active/focused state.
///
/// A failure to read the application or its children is an `Err`, never an empty
/// list: "this app has no windows" and "this app's windows could not be read" are
/// different answers, and a listing must be able to say which one it has. A child
/// whose own proxy, role or extents cannot be read is skipped — that is one
/// window's detail, not a failed read of the application.
///
/// One application's part of a listing pays a few exchanges per window, so it checks
/// its budget — the call's, or the share of it this application was given — before every
/// read it makes, including the reads inside one window: an application that stalls is
/// caught at the check after its stall, having overspent by that one read at most.
async fn app_window_handles(
    app_ref: &ObjectRefOwned,
    conn: &zbus::Connection,
    app_name: &str,
    budget: &core::Budget,
) -> Result<Vec<WindowHandle>, anyhow::Error> {
    let mut out = Vec::new();
    budget.check()?;
    let accessible = app_ref
        .as_accessible_proxy(conn)
        .await
        .map_err(|e| dbus_err(e, "read windows"))?;
    if claims_too_many_children(&accessible).await {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "{app_name} advertises more than {} children — its windows were not read",
                core::MAX_CHILDREN
            ),
        ));
    }
    // A stalled count read must not be followed by the list read it cannot afford.
    budget.check()?;
    let mut children = accessible
        .get_children()
        .await
        .map_err(|e| dbus_err(e, "read windows"))?;
    children.truncate(core::MAX_CHILDREN);
    for child in children {
        budget.check()?;
        let Ok(ca) = child.as_accessible_proxy(conn).await else {
            continue;
        };
        let role = ca.get_role().await.unwrap_or(Role::Unknown);
        budget.check()?;
        if !is_window_role(role) {
            continue;
        }
        let title = core::cap_label(&ca.name().await.unwrap_or_default());
        budget.check()?;
        let state = ca.get_state().await.unwrap_or_default();
        budget.check()?;
        let Some(surface) = window_screen_extents(&ca, budget).await? else {
            continue;
        };
        out.push(WindowHandle {
            key: window_key(&child),
            object_ref: child,
            surface,
            title,
            active: state.contains(AtspiState::Active),
            focused: state.contains(AtspiState::Focused),
        });
    }
    Ok(out)
}

/// An element's global screen extents, checked per exchange so one stalled read cannot
/// outlive the caller's budget. `None` is a surface that reports no usable extents;
/// `Err` is the budget spent.
async fn window_screen_extents(
    accessible: &AccessibleProxy<'_>,
    budget: &core::Budget,
) -> Result<Option<SurfaceGeometry>, anyhow::Error> {
    budget.check()?;
    let Some(proxies) = accessible.proxies().await.ok() else {
        return Ok(None);
    };
    budget.check()?;
    let Some(comp) = proxies.component().await.ok() else {
        return Ok(None);
    };
    budget.check()?;
    let Some((x, y, w, h)) = comp.get_extents(CoordType::Screen).await.ok() else {
        return Ok(None);
    };
    Ok(Some(SurfaceGeometry {
        x: f64::from(x),
        y: f64::from(y),
        width: f64::from(w),
        height: f64::from(h),
    }))
}

/// A window target resolved, named, refused if minimized and measured: the reads every
/// window-shaped call makes before it can show, capture or map coordinates, each charged to
/// `budget`.
struct ResolvedSurface {
    /// The window object the reads were made against.
    object_ref: ObjectRefOwned,
    app_name: String,
    title: Option<String>,
    surface: SurfaceGeometry,
}

/// The one place a window target becomes a named, measurable surface. Each step is checked
/// against `budget`, so a target that stops answering costs the read in flight and ends the
/// call there instead of at the next whole step.
async fn resolve_surface(
    conn: &zbus::Connection,
    target: &TargetSpec,
    budget: &mut core::Budget,
) -> Result<ResolvedSurface, anyhow::Error> {
    let (object_ref, app_info) = resolve_window(conn, target, budget).await?;
    let proxy = object_ref
        .as_accessible_proxy(conn)
        .await
        .map_err(|e| dbus_err(e, "open window"))?;
    let (app_name, title) = surface_identity(target, &proxy, &app_info).await;
    budget.check()?;
    let label = core::name_surface(budget, &app_name, title.as_deref());
    refuse_minimized(&proxy).await?;
    budget.check()?;
    let surface = window_screen_extents(&proxy, budget)
        .await?
        .ok_or_else(|| core::unanswered(&label, "for its window geometry"))?;
    Ok(ResolvedSurface {
        object_ref,
        app_name,
        title,
        surface,
    })
}

/// Refuse a minimized (Iconified) window before its frame is used: it keeps the
/// extents it had before it was minimized, so a crop of them, or a point mapped
/// through them, is about a different part of the screen.
async fn refuse_minimized(proxy: &AccessibleProxy<'_>) -> Result<(), anyhow::Error> {
    if proxy
        .get_state()
        .await
        .is_ok_and(|state| state.contains(AtspiState::Iconified))
    {
        return Err(core::minimized_window(&core::cap_label(
            &proxy.name().await.unwrap_or_default(),
        )));
    }
    Ok(())
}

/// Resolve a stored window key to exactly the window it names.
///
/// The key is the AT-SPI object reference (`unique name|object path`), so this
/// either lands on that window or fails with [`core::ERR_SURFACE_GONE`]: it never
/// falls back to another window of the app, and never to coordinates.
async fn resolve_window_by_key(
    conn: &zbus::Connection,
    app_name: &str,
    key: &WindowKey,
) -> Result<ObjectRefOwned, anyhow::Error> {
    let object_ref = object_ref_from_key(key).ok_or_else(|| core::surface_gone(app_name))?;
    let accessible = object_ref
        .as_accessible_proxy(conn)
        .await
        .map_err(|e| window_gone_or_silent(e, app_name))?;
    let role = accessible
        .get_role()
        .await
        .map_err(|e| window_gone_or_silent(e, app_name))?;
    if !is_window_role(role) {
        return Err(core::surface_gone(app_name));
    }
    Ok(object_ref)
}

/// The failure for a stored window key. Only the bus's own "that object is gone"
/// answer is `surface-gone`: a call that timed out is a live window that stopped
/// answering, and saying it vanished would send the model re-enumerating a window that
/// is still there.
fn window_gone_or_silent(err: impl std::fmt::Display, app_name: &str) -> anyhow::Error {
    let msg = err.to_string();
    match classify_bus_fault(&msg) {
        BusFault::Gone => core::surface_gone(app_name),
        BusFault::Transient => core::unanswered(
            &core::surface_label(app_name, None),
            "when the call looked for it",
        ),
        fault => bus_error(fault, &msg, "resolve the window"),
    }
}

/// Parse a [`WindowKey`] back into an owned object reference. `None` when the
/// text is not one this backend produced.
fn object_ref_from_key(key: &WindowKey) -> Option<ObjectRefOwned> {
    let (name, path) = key.as_str().split_once('|')?;
    let name = zbus::names::UniqueName::try_from(name).ok()?;
    let path = zbus::zvariant::ObjectPath::try_from(path).ok()?;
    Some(ObjectRef::new_owned(name.to_owned(), path.to_owned()))
}

/// The app's own window: the one reporting active/focused state, else its first.
/// Per-call by design — a `TargetSpec::App` names an application, not a window.
async fn app_current_window(
    conn: &zbus::Connection,
    app_ref: &ObjectRefOwned,
    app_name: &str,
    budget: &core::Budget,
) -> Result<ObjectRefOwned, anyhow::Error> {
    let wins = app_window_handles(app_ref, conn, app_name, budget).await?;
    if let Some(w) = wins.iter().find(|w| w.active || w.focused) {
        return Ok(w.object_ref.clone());
    }
    wins.into_iter()
        .next()
        .map(|w| w.object_ref)
        .ok_or_else(|| core::surface_gone(app_name))
}

/// The one window reporting active/focused across every application. An application
/// whose windows cannot be read is skipped, as it always has been: a targetless read
/// must not be broken by an unrelated application, and the `windows` listing is where
/// such a failure is disclosed. A share that runs out is not one of those details —
/// that application was skipped because it stopped answering — so a search that finds
/// no window reports it instead of claiming none has focus.
async fn resolve_focused_window(
    conn: &zbus::Connection,
    budget: &core::Budget,
) -> Result<ResolvedWindow, anyhow::Error> {
    let apps = enumerate_apps(conn, budget).await?;
    let mut matches = Vec::new();
    let mut skipped = None;
    for (app_ref, app_info) in &apps {
        budget.check()?;
        // One application's share: a hung one costs its own part, and the search goes on.
        let slice = budget.slice(core::APP_SLICE, core::windows_label(&app_info.name));
        match app_window_handles(app_ref, conn, &app_info.name, &slice).await {
            Ok(handles) => {
                for w in handles {
                    if w.active || w.focused {
                        matches.push(ResolvedWindow {
                            object_ref: w.object_ref,
                            key: w.key,
                            app_info: app_info.clone(),
                            title: w.title,
                            surface: w.surface,
                        });
                    }
                }
            }
            // Its share ran out: remembered, because it may be the window the caller is
            // after. Any other failure is one application's detail.
            Err(_) => {
                if slice.expired() {
                    skipped = Some(slice);
                }
            }
        }
    }
    match matches.len() {
        1 => return Ok(matches.remove(0)),
        0 => {}
        n => {
            return Err(core::taxonomy_error(
                core::ERR_NOT_MATCHED,
                format!(
                    "{n} windows report active/focused state — enumerate apps/windows to pick a \
                     surface"
                ),
            ));
        }
    }
    // Nothing reported focus. A call whose budget ran out says so; otherwise an
    // application that stopped answering is named — where "no focused window found" would
    // read as an ordinary negative answer — and the negative result is stated with it.
    budget.check()?;
    if let Some(slice) = skipped {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("{}, and no window reports active/focused", slice.note()),
        ));
    }
    Err(core::taxonomy_error(
        core::ERR_NOT_MATCHED,
        "no focused window found (no window reports active/focused) — enumerate apps/windows first",
    ))
}

/// Resolve a non-`Screen` target to its window object + owning app, under the call's
/// budget: resolution walks every application, so a hung unrelated one must not hold
/// work aimed at a healthy window.
async fn resolve_window(
    conn: &zbus::Connection,
    target: &TargetSpec,
    budget: &core::Budget,
) -> Result<(ObjectRefOwned, AppInfo), anyhow::Error> {
    match target {
        TargetSpec::Screen => Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "observe targets an accessibility surface — use screenshot for the screen",
        )),
        TargetSpec::Focused => {
            let w = resolve_focused_window(conn, budget).await?;
            Ok((w.object_ref, w.app_info))
        }
        TargetSpec::App { app_pid, app_name } => {
            let app_ref = resolve_app_object(conn, app_name, *app_pid, budget).await?;
            let win_ref = app_current_window(conn, &app_ref, app_name, budget).await?;
            let app_info = app_info_of(conn, &app_ref).await?;
            Ok((win_ref, app_info))
        }
        TargetSpec::Window {
            app_pid,
            app_name,
            key,
            ..
        } => {
            let win_ref = resolve_window_by_key(conn, app_name, key).await?;
            Ok((
                win_ref,
                AppInfo {
                    pid: *app_pid,
                    name: app_name.clone(),
                },
            ))
        }
    }
}

// ── Window stills ────────────────────────────────────────────────────────
//
// A window still must show the window's current visible pixels, occluders
// included, which reading the window's own drawable would not — so a window still
// is a crop of the screen capture. AT-SPI extents and captured pixels are
// different coordinate spaces whenever the desktop scale is not 1, so the crop is
// taken only after the whole screen is read from the accessibility stack and
// reconciled with the capture (`core::a11y_scale`) — an unreconcilable pair
// refuses instead of cropping a guess.

/// The whole screen in the accessibility coordinate space (the registry root's own
/// `Component` extents). A failed exchange means the stack did not report it (`Ok(None)`);
/// a spent budget is an error, because a target that stopped answering must not be reported
/// as nothing being reported.
async fn a11y_screen_extents(
    conn: &zbus::Connection,
    budget: &core::Budget,
) -> Result<Option<SurfaceGeometry>, anyhow::Error> {
    budget.check()?;
    let Some(root) = registry_root(conn).await.ok() else {
        return Ok(None);
    };
    window_screen_extents(&root, budget).await
}

/// The pixel space stills and coordinate mapping use: the X11 virtual desktop, or
/// the last portal capture's size on Wayland. `None` when it cannot be established.
fn pixel_space() -> Option<(u32, u32)> {
    if wayland() {
        // Copied out of the cache, so no lock is held past this call.
        *SCREEN_DIMS.lock().unwrap_poison()
    } else {
        with_block_in_place(x11_screen_geometry)
            .ok()
            .map(|geo| (geo.width.round() as u32, geo.height.round() as u32))
    }
}

/// What to do instead when an accessibility space cannot be mapped into pixels: the
/// one tail every refusal in [`a11y_pixel_scale`] ends with.
const A11Y_UNMAPPED_GUIDANCE: &str = "capture target: \"screen\" first — its still and zoom \
     always work — and prefer ref actions, which need no pixels";

/// The scale between accessibility extents and the pixel space `pixels` describes —
/// the one owner of that reconciliation, so a crop and a coordinate map can never
/// disagree about whether their spaces match.
///
/// One message per cause: an unknown pixel space is not a mismatch, an
/// accessibility screen that was never read was never compared at all (a spent budget is
/// its own, honest error), and only a pair that was compared and disagreed is a
/// reconciliation failure. On X11 the
/// accessibility screen extents are the X11 root's own pixels, and a root covering
/// one output of a larger desktop is a sub-rect rather than a scale — refusing
/// there would lose a 1:1 crop that is right, so X11 falls back to it. Wayland's
/// per-output scaling leaves no such guarantee, so an unreconcilable space there
/// is refused instead of guessed.
async fn a11y_pixel_scale(
    conn: &zbus::Connection,
    pixels: Option<(u32, u32)>,
    budget: &core::Budget,
) -> Result<f64, anyhow::Error> {
    let Some((width, height)) = pixels else {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "the screen's pixel size is not known, so accessibility extents cannot be mapped \
                 into it — {A11Y_UNMAPPED_GUIDANCE}"
            ),
        ));
    };
    let Some(extents) = a11y_screen_extents(conn, budget).await? else {
        // Nothing was compared, so this is not a mismatch: the accessibility stack
        // did not report the screen at all. X11 still answers in the root's own
        // pixels, so 1:1 remains the right mapping there.
        if !wayland() {
            return Ok(1.0);
        }
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "the accessibility stack did not report the screen's geometry, so its coordinates \
                 cannot be mapped into the captured pixels — {A11Y_UNMAPPED_GUIDANCE}"
            ),
        ));
    };
    match core::a11y_scale(&extents, (width, height)) {
        Some(scale) => Ok(scale),
        None if !wayland() => Ok(1.0),
        None => Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "the accessibility extents could not be reconciled with the {width}x{height} \
                 screen pixels, so nothing was mapped — {A11Y_UNMAPPED_GUIDANCE}"
            ),
        )),
    }
}

/// One window still: the window's AT-SPI extents mapped into the captured pixels
/// by the scale both spaces share, then cropped. `cap_geo` is the geometry the
/// backend maps coordinates with (the X11 virtual desktop, the portal image on
/// Wayland), so `cap` and `cap_geo` always describe the same pixels.
///
/// The extents are taken to be screen-relative, which is what `CoordType::Screen`
/// asks for; a toolkit that answers with window-relative extents cannot be told
/// apart from the accessibility data, so its crop is the wrong part of the screen.
/// The crop is a buffer of its own, so it is charged to the call's `budget`; `lookup`
/// bounds the accessibility screen read the scale comes from — the only accessibility
/// exchange in here.
async fn window_still(
    conn: &zbus::Connection,
    win: &SurfaceGeometry,
    cap: &Capture,
    cap_geo: &SurfaceGeometry,
    budget: &CaptureBudget,
    lookup: &core::Budget,
) -> Result<Capture, anyhow::Error> {
    let scale = a11y_pixel_scale(conn, Some((cap.width, cap.height)), lookup).await?;
    // The crop is clamped to the captured screen, so a window that hangs off it
    // yields its visible part — the same result as before the reconciliation
    // existed. A window entirely outside it is an honest failure.
    core::crop_rgba(
        cap,
        core::window_capture_rect(&core::scale_geometry(*win, scale), cap_geo)?,
        budget,
        "the window still",
    )
}

// ── Accessibility tree walk ──────────────────────────────────────────────

/// One node and its subtree, `Ok(None)` when the node is not emitted (it could not
/// be opened, or the node ceiling was reached). `Err` is the aggregate budget
/// running out: a refused or timed-out attribute or child only loses that detail
/// and the walk continues, but a target that has spent the whole budget ends the
/// call — thousands of bounded exchanges can still add up to more than the tool may
/// hold.
#[async_recursion]
async fn walk_node(
    object_ref: &ObjectRefOwned,
    conn: &zbus::Connection,
    depth: usize,
    handles: &mut Vec<ObjectRefOwned>,
    nodes: &mut usize,
    truncation: &mut Truncation,
    budget: &core::Budget,
) -> anyhow::Result<Option<UiNode>> {
    budget.check()?;
    if *nodes >= core::MAX_TREE_NODES {
        truncation.hit_node_ceiling(core::MAX_TREE_NODES);
        return Ok(None);
    }
    let Some(accessible) = object_ref.as_accessible_proxy(conn).await.ok() else {
        return Ok(None);
    };
    *nodes += 1;
    handles.push(object_ref.clone());

    let role = role_name(accessible.get_role().await.unwrap_or(Role::Unknown));
    // A name is what a ref falls back to matching on and a title can be a document,
    // so it is capped at the read's own limit and a cut is stated in the header.
    let name = accessible
        .name()
        .await
        .ok()
        .map(|name| truncation.keep_name(&name));
    let proxies = accessible.proxies().await.ok();
    // A node's own reads are several exchanges; a target that stalls across them must
    // not be allowed to spend the budget of the whole walk here.
    budget.check()?;

    // Read one character past what the renderer shows ([`VALUE_READ_CHARS`]), so the
    // value never needs the shared read cap and the renderer's own count states the
    // display cut in the header.
    let value = tree_value(proxies.as_ref(), &role).await;
    let frame = tree_frame(proxies.as_ref()).await;
    let actions = tree_actions(proxies.as_ref()).await;
    budget.check()?;
    let focused = accessible
        .get_state()
        .await
        .is_ok_and(|s| s.contains(AtspiState::Focused));

    let mut children = Vec::new();
    if depth >= MAX_DEPTH {
        // The node's subtree is not read. Only a node that has one is a cut, so a leaf
        // at the ceiling is not reported as a truncation.
        if node_has_children(&accessible).await {
            truncation.cut_depth(MAX_DEPTH);
        }
    } else {
        budget.check()?;
        if let Some(child_refs) = walkable_children(&accessible, truncation).await {
            for child in child_refs {
                if *nodes >= core::MAX_TREE_NODES {
                    truncation.hit_node_ceiling(core::MAX_TREE_NODES);
                    break;
                }
                if let Some(child_node) =
                    walk_node(&child, conn, depth + 1, handles, nodes, truncation, budget).await?
                {
                    children.push(child_node);
                }
            }
        }
    }

    Ok(Some(UiNode {
        role,
        name,
        value,
        frame,
        actions,
        focused,
        children,
    }))
}

/// Whether a node has any children, at the depth ceiling where only the yes/no is
/// wanted. A count beyond the child ceiling answers it without paying for the list,
/// and a count the target cannot be trusted with (or cannot produce) falls back to the
/// list itself — this decides whether a subtree is recorded as cut, so it must not turn
/// a wrong count into a hidden one.
async fn node_has_children(accessible: &AccessibleProxy<'_>) -> bool {
    if claims_too_many_children(accessible).await {
        return true;
    }
    accessible
        .get_children()
        .await
        .is_ok_and(|children| !children.is_empty())
}

/// A node's children, or `None` when its advertised count exceeds the child ceiling
/// — a spreadsheet's cell list advertises near 2^31, and one exchange would then pay
/// for all of it. A reply that is longer than the ceiling anyway is cut to it. Both
/// cases are recorded, each as what it is.
async fn walkable_children(
    accessible: &AccessibleProxy<'_>,
    truncation: &mut Truncation,
) -> Option<Vec<ObjectRefOwned>> {
    if claims_too_many_children(accessible).await {
        truncation.cut_children(core::MAX_CHILDREN);
        return None;
    }
    let mut children = accessible.get_children().await.ok()?;
    if children.len() > core::MAX_CHILDREN {
        children.truncate(core::MAX_CHILDREN);
        truncation.trim_children(core::MAX_CHILDREN);
    }
    Some(children)
}

async fn build_tree(
    root_ref: &ObjectRefOwned,
    conn: &zbus::Connection,
    budget: &core::Budget,
) -> anyhow::Result<(UiNode, Vec<ObjectRefOwned>, Truncation)> {
    let mut handles = Vec::new();
    let mut nodes = 0usize;
    let mut truncation = Truncation::default();
    let Some(node) = walk_node(
        root_ref,
        conn,
        0,
        &mut handles,
        &mut nodes,
        &mut truncation,
        budget,
    )
    .await?
    else {
        // The root is the one node whose absence is not a detail: without it nothing
        // was read at all, and an empty tree would present that as a window with no
        // content.
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "{} did not answer for its accessibility root, so nothing was read — \
                 retry, or use screenshot instead",
                budget.target()
            ),
        ));
    };
    // A walk that lost the bus mid-way degrades per node, so what it kept would read as
    // a thin window; say the read is incomplete instead.
    ensure_link(conn)?;
    Ok((node, handles, truncation))
}

async fn tree_value(proxies: Option<&Proxies<'_>>, role: &str) -> Option<String> {
    let proxies = proxies?;
    if !is_text_role(role) {
        return None;
    }
    let text = proxies.text().await.ok()?;
    let count = text.character_count().await.ok()?;
    if count <= 0 {
        return None;
    }
    // A text longer than the cap is read up to it and shown cut, never dropped: an
    // absent value would read as "this field is empty".
    text.get_text(0, count.min(VALUE_READ_CHARS)).await.ok()
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

async fn press_element(proxies: &Proxies<'_>) -> Result<Impact, anyhow::Error> {
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
    let ok = match action.do_action(idx as i32).await {
        Ok(ok) => ok,
        // The one actuating call: a bus failure here may have performed it.
        Err(e) => return action_dbus_outcome(e, "perform action"),
    };
    if ok {
        Ok(Impact::Done)
    } else {
        Err(anyhow!("action '{}' reported failure", actions[idx].name))
    }
}

async fn set_element_value(
    proxies: &Proxies<'_>,
    role: &str,
    text: &str,
) -> Result<Impact, anyhow::Error> {
    if role.to_lowercase().contains("password") {
        return Err(core::taxonomy_error(
            core::ERR_UNSUPPORTED,
            "element is a password field — SetValue refused (use raw input to type)",
        ));
    }
    if let Ok(editable) = proxies.editable_text().await {
        let ok = match editable.set_text_contents(text).await {
            Ok(ok) => ok,
            // The one actuating call: a bus failure here may have set the text.
            Err(e) => return action_dbus_outcome(e, "set text contents"),
        };
        return if ok {
            Ok(Impact::Done)
        } else {
            Err(anyhow!("set-text-contents returned false"))
        };
    }
    if let Ok(value) = proxies.value().await
        && let Ok(num) = text.trim().parse::<f64>()
    {
        if let Err(e) = value.set_current_value(num).await {
            return action_dbus_outcome(e, "set current value");
        }
        return Ok(Impact::Done);
    }
    Err(core::taxonomy_error(
        core::ERR_UNSUPPORTED,
        "element has no editable-text or value interface — SetValue not available; use raw input",
    ))
}

// ── Raw input (enigo / XTEST) ────────────────────────────────────────────

/// Bring a window to the front and prove it got there.
///
/// XTEST input goes to whatever holds the focus, so input aimed at a window that
/// did not come forward must fail BEFORE anything is synthesized: reporting
/// success while the keystrokes went to another window is the failure this
/// prevents. `grab_focus` is a request to the window manager, so its reply is not
/// the authority — the window's own active/focused state is, re-read briefly.
async fn raise_window(
    conn: &zbus::Connection,
    window_ref: &ObjectRefOwned,
    target: &TargetSpec,
) -> Result<(), anyhow::Error> {
    let proxy = window_ref
        .as_accessible_proxy(conn)
        .await
        .map_err(|e| dbus_err(e, "open window"))?;
    if let Ok(proxies) = proxy.proxies().await
        && let Ok(component) = proxies.component().await
    {
        let _ = component.grab_focus().await;
    }
    for _ in 0..core::RAISE_SETTLE_STEPS {
        let state = proxy
            .get_state()
            .await
            .map_err(|e| dbus_err(e, "read window state"))?;
        if state.contains(AtspiState::Active) || state.contains(AtspiState::Focused) {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(core::RAISE_SETTLE_MS)).await;
    }
    Err(core::not_brought_forward(target.describe()))
}

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

// ── Input synthesis ──────────────────────────────────────────────────────

/// One call's synthesis session: the `Enigo` it drives and the mouse buttons it has
/// pressed and not yet released. A button is recorded before it is pressed — a press that
/// errors may still have reached the server — and released by [`Drop`]; enigo's own drop
/// releases the keys it pressed.
struct InputSession {
    enigo: Enigo,
    buttons: Vec<Button>,
}

impl InputSession {
    fn new() -> Result<Self, anyhow::Error> {
        let enigo =
            Enigo::new(&Settings::default()).map_err(|e| anyhow!("enigo init failed: {e}"))?;
        Ok(Self {
            enigo,
            buttons: Vec::new(),
        })
    }

    /// Press `button`, recorded BEFORE the attempt: a press that errors may still have
    /// reached the server, so it must be released either way.
    fn press_button(&mut self, button: Button) -> Result<(), enigo::InputError> {
        self.buttons.push(button);
        self.enigo.button(button, Direction::Press)
    }

    /// Release `button`; only a release that went through drops it from the held list.
    fn release_button(&mut self, button: Button) -> Result<(), enigo::InputError> {
        let released = self.enigo.button(button, Direction::Release);
        if released.is_ok() {
            self.buttons.retain(|held| *held != button);
        }
        released
    }
}

impl Drop for InputSession {
    /// Let go of every button this call left pressed. Failures are dropped with the
    /// session — there is nothing left to report them to, and the result the call
    /// returned already named what may be held.
    fn drop(&mut self) {
        for button in self.buttons.drain(..) {
            let _ = self.enigo.button(button, Direction::Release);
        }
    }
}

/// Press `mods` in order. `Err` carries the index of the modifier whose press failed
/// and that failure: it and every modifier before it may be down, so a caller that
/// stops here must say so rather than report that nothing was delivered.
fn press_modifiers(
    session: &mut InputSession,
    mods: &[Modifier],
) -> Result<(), (usize, enigo::InputError)> {
    for (i, m) in mods.iter().enumerate() {
        if let Err(e) = session.enigo.key(enigo_mod_key(*m), Direction::Press) {
            return Err((i, e));
        }
    }
    Ok(())
}

fn release_modifiers(session: &mut InputSession, mods: &[Modifier]) -> Result<(), anyhow::Error> {
    for m in mods.iter().rev() {
        session
            .enigo
            .key(enigo_mod_key(*m), Direction::Release)
            .map_err(input_err)?;
    }
    Ok(())
}

/// Release `mods` and, when that fails, name what a held modifier costs and who can
/// clear it: the chord and every character typed after it would be modified.
fn release_note(session: &mut InputSession, mods: &[Modifier]) -> Option<String> {
    if mods.is_empty() || release_modifiers(session, mods).is_ok() {
        return None;
    }
    let names = mods
        .iter()
        .map(|m| match m {
            Modifier::Cmd => "cmd",
            Modifier::Ctrl => "ctrl",
            Modifier::Alt => "alt",
            Modifier::Shift => "shift",
        })
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "{names} may still be held down, so every later chord and typed character may be modified \
         — ask the owner to clear it before acting again"
    ))
}

/// Append a release state to a detail that already reports what landed.
fn with_held(detail: String, held: Option<String>) -> String {
    match held {
        Some(held) => format!("{detail}; {held}"),
        None => detail,
    }
}

/// The result of an actuation whose own input is delivered: done, unless letting go
/// of the modifiers failed, which the result then has to say.
fn done_after_release(session: &mut InputSession, mods: &[Modifier]) -> Impact {
    match release_note(session, mods) {
        None => Impact::Done,
        Some(note) => Impact::DoneWith(note),
    }
}

/// The failure of an actuation that stopped before its own input was delivered. It
/// says only what is known: the input did not go through, and a modifier a failed
/// press left down has to be released before trying again.
fn press_failed(
    session: &mut InputSession,
    mods: &[Modifier],
    what: &str,
    err: &enigo::InputError,
) -> anyhow::Error {
    core::taxonomy_error(
        core::ERR_DEGRADED,
        with_held(format!("{what} ({err})"), release_note(session, mods)),
    )
}

fn scroll_params(direction: ScrollDirection, amount: u32) -> (i32, Axis) {
    // The tool bounds the amount before it gets here; the conversion is still total so
    // no magnitude can wrap the direction.
    let a = i32::try_from(amount).unwrap_or(i32::MAX);
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

// Owned because it is handed to `map_err` at most call sites: borrowing would push
// a closure into each of them.
#[expect(clippy::needless_pass_by_value)]
fn input_err(e: enigo::InputError) -> anyhow::Error {
    anyhow!("input synthesis failed: {e}")
}

fn key_chord(session: &mut InputSession, chord: &str) -> Result<Impact, anyhow::Error> {
    let (mods, key) = core::parse_key_chord(chord)?;
    let key = enigo_key(&key).map_err(anyhow::Error::msg)?;
    // Nothing of the chord is in the system before the modifiers are down, so a
    // failed modifier press is only a half-pressed chord: it and the ones before it
    // may be down, but no key of it has been delivered.
    if let Err((failed, e)) = press_modifiers(session, &mods) {
        return Err(press_failed(
            session,
            &mods[..=failed],
            "the chord did not go through — a modifier press failed",
            &e,
        ));
    }
    // XTEST writes the key press before the release can fail, so a press that errors
    // may already have fired the chord: press and release are driven apart, and a
    // press failure is repaired by releasing the key again.
    if let Err(e) = session.enigo.key(key, Direction::Press) {
        let released = session.enigo.key(key, Direction::Release).is_ok();
        let key_state = if released {
            "the chord may already have fired, and the key was released again"
        } else {
            "the key may still be held down"
        };
        return Ok(Impact::Unconfirmed(with_held(
            format!("the chord's key press errored ({e}) — {key_state}"),
            release_note(session, &mods),
        )));
    }
    if let Err(e) = session.enigo.key(key, Direction::Release) {
        return Ok(Impact::Unconfirmed(with_held(
            format!(
                "the chord's key went down but its release failed ({e}) — the key may still be \
                 held down"
            ),
            release_note(session, &mods),
        )));
    }
    Ok(done_after_release(session, &mods))
}

/// What a drag that stopped after the button went down leaves behind. `released`
/// says whether the button was let go again, so the result never claims a clean
/// machine when one may still be mid-gesture.
fn drag_incomplete(fx: i32, fy: i32, released: bool) -> Impact {
    let at = format!("the button went down at ({fx}, {fy})");
    Impact::Unconfirmed(if released {
        format!(
            "the drag started ({at}) but the gesture did not complete; the button was released \
             again"
        )
    } else {
        format!(
            "the drag started ({at}) but the gesture did not complete and the button may still be \
             held down"
        )
    })
}

/// Let a button go again after a press that errored — the press may have been
/// written before the call failed — and say how that ended.
fn release_after_failed_press(session: &mut InputSession, button: Button) -> &'static str {
    if session.release_button(button).is_ok() {
        "was released again"
    } else {
        "may still be held down"
    }
}

/// A coordinate click: the pointer moves first, then the modifiers, then each
/// press/release pair (twice for a double click). XTEST synthesizes into whatever is
/// under the pointer — the model clicks first if focus is required (no
/// activate-then-control) — and every step after the move is answered for: a failure
/// names what may have landed or may still be held, because a second click is a
/// second actuation.
fn click(
    session: &mut InputSession,
    x: i32,
    y: i32,
    button: Button,
    double: bool,
    modifiers: &[Modifier],
) -> Result<Impact, anyhow::Error> {
    session
        .enigo
        .move_mouse(x, y, Coordinate::Abs)
        .map_err(input_err)?;
    if let Err((failed, e)) = press_modifiers(session, modifiers) {
        return Err(press_failed(
            session,
            &modifiers[..=failed],
            "the click did not go through — a modifier press failed",
            &e,
        ));
    }
    for press in 0..(if double { 2 } else { 1 }) {
        if let Err(e) = session.press_button(button) {
            let state = release_after_failed_press(session, button);
            let detail = if press == 0 {
                format!(
                    "the button press errored ({e}) — the target may have seen a click at ({x}, \
                     {y}), and the button {state}"
                )
            } else {
                format!(
                    "the first of the two clicks was delivered and the second's button press \
                     errored ({e}) — the button {state}"
                )
            };
            return Ok(Impact::Unconfirmed(with_held(
                detail,
                release_note(session, modifiers),
            )));
        }
        if let Err(e) = session.release_button(button) {
            let detail = format!(
                "the button was pressed but its release failed ({e}) — it may still be held down"
            );
            return Ok(Impact::Unconfirmed(with_held(
                detail,
                release_note(session, modifiers),
            )));
        }
    }
    Ok(done_after_release(session, modifiers))
}

/// Type `text` one character at a time. Enigo writes a character's key press before
/// its release can fail, so each half is driven apart: a press that errors may
/// already have typed the character, and a release that errors leaves its key down.
fn type_text(session: &mut InputSession, text: &str) -> Impact {
    let total = text.chars().count();
    for (typed, ch) in text.chars().enumerate() {
        let key = Key::Unicode(ch);
        if let Err(e) = session.enigo.key(key, Direction::Press) {
            let state = if session.enigo.key(key, Direction::Release).is_ok() {
                "that character may already have been typed, and its key was released again"
            } else {
                "that character may already have been typed and its key may still be held down"
            };
            return Impact::Unconfirmed(format!(
                "{typed} of {total} characters were typed before the next one's key press errored \
                 ({e}) — {state}"
            ));
        }
        if let Err(e) = session.enigo.key(key, Direction::Release) {
            return Impact::Unconfirmed(format!(
                "{} of {total} characters were typed, but the last one's key release failed ({e}) \
                 — its key may still be held down",
                typed + 1
            ));
        }
    }
    Impact::Done
}

/// One wheel tick is a press/release pair of a scroll button, and the ticks are
/// driven apart for the same reason: a failure names how many landed and whether a
/// wheel button is left down.
fn scroll(session: &mut InputSession, length: i32, axis: Axis) -> Impact {
    let button = match (length.is_positive(), axis) {
        (true, Axis::Vertical) => Button::ScrollDown,
        (false, Axis::Vertical) => Button::ScrollUp,
        (true, Axis::Horizontal) => Button::ScrollRight,
        (false, Axis::Horizontal) => Button::ScrollLeft,
    };
    let ticks = length.abs();
    for tick in 0..ticks {
        if let Err(e) = session.press_button(button) {
            let state = release_after_failed_press(session, button);
            return Impact::Unconfirmed(format!(
                "{tick} of {ticks} scroll ticks were sent before the next one's press errored \
                 ({e}) — its wheel button {state}"
            ));
        }
        if let Err(e) = session.release_button(button) {
            return Impact::Unconfirmed(format!(
                "{} of {ticks} scroll ticks were sent, but the last one's release failed ({e}) — \
                 its wheel button may still be held down",
                tick + 1
            ));
        }
    }
    Impact::Done
}

fn raw_input_blocking(
    input: RawInput,
    geo: Option<&SurfaceGeometry>,
) -> Result<Impact, anyhow::Error> {
    let mut session = InputSession::new()?;
    match input {
        RawInput::Click {
            point,
            button,
            double,
            modifiers,
        } => {
            let (x, y) = pixel_point(point, geo)?;
            click(&mut session, x, y, enigo_button(button), double, &modifiers)
        }
        RawInput::TypeText { text } => Ok(type_text(&mut session, &text)),
        RawInput::KeyChord { chord } => key_chord(&mut session, &chord),
        RawInput::Scroll {
            point,
            direction,
            amount,
        } => {
            if let Some((nx, ny)) = point {
                let (x, y) = pixel_point((nx, ny), geo)?;
                session
                    .enigo
                    .move_mouse(x, y, Coordinate::Abs)
                    .map_err(input_err)?;
            }
            let (length, axis) = scroll_params(direction, amount);
            Ok(scroll(&mut session, length, axis))
        }
        RawInput::Drag { from, to } => {
            let (fx, fy) = pixel_point(from, geo)?;
            let (tx, ty) = pixel_point(to, geo)?;
            // The button goes down where the gesture starts: pressing at the
            // pointer's previous position would grab whatever window that was.
            session
                .enigo
                .move_mouse(fx, fy, Coordinate::Abs)
                .map_err(input_err)?;
            // As in a click, the press may have reached the server before the call
            // failed: the button can be down and the target may have seen a click.
            if let Err(e) = session.press_button(Button::Left) {
                let state = release_after_failed_press(&mut session, Button::Left);
                return Ok(Impact::Unconfirmed(format!(
                    "the drag's button press errored ({e}) at ({fx}, {fy}) — the button {state}, \
                     so the target may have taken it as a click"
                )));
            }
            // Past the button-down the drag is under way: a failure must let the
            // button go again rather than leave the machine mid-gesture.
            for i in 1..=INTERP_STEPS {
                let t = i as f64 / INTERP_STEPS as f64;
                let x = (fx as f64 + (tx - fx) as f64 * t).round() as i32;
                let y = (fy as f64 + (ty - fy) as f64 * t).round() as i32;
                if session.enigo.move_mouse(x, y, Coordinate::Abs).is_err() {
                    let released = session.release_button(Button::Left).is_ok();
                    return Ok(drag_incomplete(fx, fy, released));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            if let Err(e) = session.release_button(Button::Left) {
                // Every step ran and the pointer reached `to`: only the release
                // failed, which is what must not read as an unfinished gesture.
                return Ok(Impact::Unconfirmed(format!(
                    "the drag ran to its end but the button's release failed ({e}) — the button \
                     may still be held down"
                )));
            }
            Ok(Impact::Done)
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
///
/// The composite and every tile are charged to the call's `budget`, so a display
/// arrangement the server reports as enormous is refused before it is allocated.
fn x11_capture_screen(budget: &CaptureBudget) -> Result<(Capture, SurfaceGeometry), anyhow::Error> {
    let (conn, screen) = x11rb::connect(None)
        .map_err(|e| core::taxonomy_error(core::ERR_DEGRADED, format!("no X11 display: {e}")))?;
    let root = conn.setup().roots[screen].root;
    let lsb_first = conn.setup().image_byte_order == xproto::ImageOrder::LSB_FIRST;
    let cap_bytes = conn
        .maximum_request_bytes()
        .clamp(64 * 1024, 4 * 1024 * 1024);
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
    let mut composite = Capture::alloc(budget, "the virtual desktop", w, h)?;
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
            let tile = core::x11_pixels_to_rgba(
                &reply.data,
                tile_w,
                tile_h,
                reply.depth,
                lsb_first,
                budget,
            )?;
            core::blit_rgba(
                &mut composite.rgba,
                out_w,
                out_h,
                &tile,
                (rx - ox) as usize,
                (ry - oy + y) as usize,
            );
            y += tile_h as i32;
        }
    }
    Ok((composite, geo))
}

// ── Wayland stills (Screenshot portal over zbus) ──────────────────────────
// One Screenshot attempt with short timeouts; deny/timeout/missing portal
// degrades to accessibility-only — no retries, no prompt spam.

/// Last known screen pixels, stored on every successful `Screen` capture so
/// zoom can size itself without triggering a second portal dialog, and so AT-SPI
/// extents can be reconciled with pixels ([`pixel_space`]).
static SCREEN_DIMS: Mutex<Option<(u32, u32)>> = Mutex::new(None);

static PORTAL_TOKEN_SEQ: AtomicU64 = AtomicU64::new(0);

const PORTAL_BUS: &str = "org.freedesktop.portal.Desktop";
const PORTAL_PATH: &str = "/org/freedesktop/portal/desktop";
const PORTAL_IFACE: &str = "org.freedesktop.portal.Screenshot";

/// Every portal call is a `Request` object; its `Close` is how a call gives one back.
const REQUEST_IFACE: &str = "org.freedesktop.portal.Request";

/// Ceiling on the encoded file the Screenshot portal hands over. The portal chooses
/// the image, so this is a FILE limit, separate from what the decode will honour
/// (the shared raster envelope, itself capped by what the call has left): a PNG of
/// any surface this tool can hold is far below it, and a larger file is refused
/// unread.
const PORTAL_FILE_MAX_BYTES: u64 = 128 * 1024 * 1024;

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

/// Single Screenshot-portal attempt: subscribe for the `Response` signal first (a fast
/// compositor can't beat the subscription), then call and wait. The request path is
/// predicted from the token and the caller's bus name before the call, so a `Screenshot`
/// call that times out — leaving no reply to read it from — still has one to close, and
/// [`PortalRequest`] closes it however the attempt ends.
async fn portal_screenshot(budget: &CaptureBudget) -> Result<Capture, anyhow::Error> {
    let conn = zbus::Connection::session().await.map_err(|e| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "Screenshot portal missing (no session bus: {e}) — no manual setup needed; \
                 accessibility observe/act remain available"
            ),
        )
    })?;
    let token = format!(
        "mahbot_{}_{}",
        std::process::id(),
        PORTAL_TOKEN_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let mut request = PortalRequest::new(&conn, &token);
    let outcome = screenshot_request(&conn, &token, &mut request, budget).await;
    request.close().await;
    outcome
}

/// One `Screenshot` request, closed however the call ends: explicitly on every path the
/// call returns on, and from its own drop on the ones it cannot — an unwind out of the
/// response handling, or a future dropped mid-call. The drop closes on a task of its own
/// because a bus round trip cannot be awaited there; the portal also drops a request when
/// the caller's connection goes away, so a missed task is not a leak.
struct PortalRequest {
    conn: zbus::Connection,
    path: Option<zbus::zvariant::OwnedObjectPath>,
}

impl PortalRequest {
    /// The request this call is about to open, at the path the portal will publish it
    /// at — `None` when the caller's own bus name is not known, in which case only a
    /// reply can name it.
    fn new(conn: &zbus::Connection, token: &str) -> Self {
        Self {
            conn: conn.clone(),
            path: request_path(conn, token),
        }
    }

    /// Take the path the portal actually published, which is authoritative over the
    /// predicted one.
    fn adopt(&mut self, path: zbus::zvariant::OwnedObjectPath) {
        self.path = Some(path);
    }

    async fn close(mut self) {
        if let Some(path) = self.path.take() {
            close_request(&self.conn, &path).await;
        }
    }
}

impl Drop for PortalRequest {
    fn drop(&mut self) {
        let Some(path) = self.path.take() else {
            return;
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let conn = self.conn.clone();
        handle.spawn(async move { close_request(&conn, &path).await });
    }
}

/// Where the portal publishes this request's `Response`, from the caller's unique bus
/// name (':' stripped, '.' as '_') and the token this tool chose: the one handle that
/// exists before the call replies.
fn request_path(conn: &zbus::Connection, token: &str) -> Option<zbus::zvariant::OwnedObjectPath> {
    let sender = conn
        .unique_name()?
        .as_str()
        .trim_start_matches(':')
        .replace('.', "_");
    format!("{PORTAL_PATH}/request/{sender}/{token}")
        .try_into()
        .ok()
}

/// Close the request, best-effort and bounded: the call's own answer is already
/// decided by the time this runs, and a portal that ignores `Close` must not hold it.
async fn close_request(conn: &zbus::Connection, path: &zbus::zvariant::ObjectPath<'_>) {
    let _ = tokio::time::timeout(Duration::from_secs(3), async {
        let proxy = zbus::Proxy::new(conn, PORTAL_BUS, path, REQUEST_IFACE)
            .await
            .ok()?;
        proxy.call::<_, _, ()>("Close", &()).await.ok()
    })
    .await;
}

/// The `Screenshot` call and its response. The request's own path is adopted as soon as
/// the portal replies, so the caller can close the request whatever the wait afterwards
/// does with it.
async fn screenshot_request(
    conn: &zbus::Connection,
    token: &str,
    request: &mut PortalRequest,
    budget: &CaptureBudget,
) -> Result<Capture, anyhow::Error> {
    let rule = zbus::MatchRule::builder()
        .msg_type(zbus::message::Type::Signal)
        .interface(REQUEST_IFACE)?
        .member("Response")?
        .build();
    let mut stream = zbus::MessageStream::for_match_rule(rule, conn, None)
        .await
        .map_err(|e| {
            core::taxonomy_error(core::ERR_DEGRADED, format!("portal subscribe failed: {e}"))
        })?;
    let mut options = HashMap::new();
    options.insert("handle_token", zbus::zvariant::Value::from(token));
    options.insert("modal", zbus::zvariant::Value::from(false));
    options.insert("interactive", zbus::zvariant::Value::from(false));
    let proxy = zbus::Proxy::new(conn, PORTAL_BUS, PORTAL_PATH, PORTAL_IFACE)
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
    let expected_path = req_path.as_str().to_owned();
    request.adopt(req_path);
    let wait = async {
        while let Some(msg) = stream.next().await {
            let msg = msg.map_err(|e| {
                core::taxonomy_error(
                    core::ERR_DEGRADED,
                    format!("portal response stream failed: {e}"),
                )
            })?;
            if msg
                .header()
                .path()
                .is_none_or(|p| p.as_str() != expected_path.as_str())
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
            return portal_response(code, results, budget).await;
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

/// The portal's image file, read through a bounded reader: at most
/// [`PORTAL_FILE_MAX_BYTES`] are taken, so a file that grew past it after it was
/// opened is refused instead of loaded whole.
async fn read_portal_file(path: &std::path::Path) -> Result<Vec<u8>, anyhow::Error> {
    let unreadable = |e: std::io::Error| {
        core::taxonomy_error(
            core::ERR_DEGRADED,
            format!("portal screenshot unreadable: {e}"),
        )
    };
    let file = tokio::fs::File::open(path).await.map_err(unreadable)?;
    let mut bytes = Vec::new();
    file.take(PORTAL_FILE_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(unreadable)?;
    if bytes.len() as u64 > PORTAL_FILE_MAX_BYTES {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            format!(
                "portal screenshot is larger than the {} MiB this tool reads — not read; \
                 accessibility observe/act remain available",
                PORTAL_FILE_MAX_BYTES / (1024 * 1024),
            ),
        ));
    }
    Ok(bytes)
}

/// `Response` signal body → pixels. Response 1 is an in-dialog deny/dismiss:
/// opt-in consent, no auto-retry. Nothing here is sized by the compositor: the file
/// is read under this product's own ceiling, the decode is bounded by what is left of
/// the call's allowance, and the RGBA this path keeps is a claimed buffer — so
/// neither a compositor nor a crafted PNG can drive an allocation this call cannot
/// survive.
async fn portal_response(
    code: u32,
    results: HashMap<String, zbus::zvariant::OwnedValue>,
    budget: &CaptureBudget,
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
            let bytes = read_portal_file(&path).await?;
            // Two ceilings on the decode: the shared raster envelope, and what is left of
            // this call's allowance (`remaining` counts RGBA pixels, four bytes each).
            // The refusal states the one that was applied.
            let mut limits = crate::util::media_target::raster_decode_limits();
            let decode_cap = crate::util::media_target::RASTER_DECODE_MAX_ALLOC_BYTES
                .min(budget.remaining().saturating_mul(4));
            limits.max_alloc = Some(decode_cap);
            let img = with_block_in_place(move || {
                crate::util::media_target::decode_raster_image(&bytes, limits)
            })
            .map_err(|e| {
                core::taxonomy_error(
                    core::ERR_DEGRADED,
                    format!(
                        "portal screenshot undecodable: {e} (this tool decodes a portal image \
                             at most {} px on a side within {} MiB)",
                        crate::util::media_target::RASTER_DECODE_MAX_DIMENSION_PX,
                        decode_cap / (1024 * 1024),
                    ),
                )
            })?;
            let (w, h) = core::capture_size(
                "the portal screenshot",
                img.width() as usize,
                img.height() as usize,
            )?;
            // The RGBA this path keeps is a claimed buffer: charging it BEFORE the
            // conversion is made refuses an image this call cannot hold instead of
            // converting it. The decode above is transient and bounded by the same
            // allowance, so this path peaks at that decode plus the claimed RGBA.
            let claim = budget.charge("the portal screenshot", w, h)?;
            Ok(Capture::adopt(claim, w, h, img.into_rgba8().into_raw()))
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
    fn is_supported(&self) -> bool {
        let gui =
            std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some();
        let bus = std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_some()
            || std::env::var_os("XDG_RUNTIME_DIR")
                .is_some_and(|d| std::path::PathBuf::from(d).join("bus").exists());
        gui && bus
    }

    /// The session's own lock state — see [`locked_reason`].
    async fn desktop_unavailable(&self) -> Option<String> {
        locked_reason().await
    }

    async fn capture_available(&self) -> bool {
        if wayland() {
            portal_available().await
        } else {
            with_block_in_place(|| x11_screen_geometry().is_ok())
        }
    }

    fn capture_unavailable_error(&self) -> anyhow::Error {
        if wayland() {
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
        let budget = core::Budget::new(
            core::LIST_BUDGET,
            "application list",
            "the running applications",
        );
        let mut out: Vec<AppInfo> = enumerate_apps(&conn, &budget)
            .await?
            .into_iter()
            .map(|(_, info)| info)
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    async fn list_windows(&self, app: Option<&AppInfo>) -> Result<WindowListing, anyhow::Error> {
        let conn = self.conn().await?;
        let mut listing = WindowListing::default();
        let budget = core::Budget::new(
            core::LIST_BUDGET,
            "window listing",
            app.map_or_else(
                || "the running applications".to_string(),
                |app| core::windows_label(&app.name),
            ),
        );
        for (app_ref, app_info) in enumerate_apps(&conn, &budget).await? {
            // The caller's filter comes first: an application it did not ask about is
            // never reported as "not reached" — only the ones this listing owes an
            // answer for are.
            if let Some(filter) = app {
                let name_ok = filter.name.eq_ignore_ascii_case(&app_info.name);
                let pid_ok = filter.pid.is_some() && filter.pid == app_info.pid;
                if !name_ok && !pid_ok {
                    continue;
                }
            }
            // A listing that runs out of time keeps what it read and names the
            // applications it did not reach, instead of discarding healthy windows.
            if let Err(e) = budget.check() {
                listing.unread.push(UnreadWindows {
                    app_name: app_info.name.clone(),
                    reason: format!("{e:#}"),
                });
                continue;
            }
            // One unreadable application must not hide the others — and must not
            // be presented as an application without windows.
            let slice = budget.slice(core::APP_SLICE, core::windows_label(&app_info.name));
            match app_window_handles(&app_ref, &conn, &app_info.name, &slice).await {
                Ok(handles) => {
                    for w in handles {
                        listing.windows.push(WindowInfo {
                            app_pid: app_info.pid,
                            app_name: app_info.name.clone(),
                            title: w.title,
                            key: w.key,
                            surface: w.surface,
                        });
                    }
                }
                // A spent share is one skipped application, said on the line naming it.
                Err(e) => listing.unread.push(UnreadWindows {
                    app_name: app_info.name.clone(),
                    reason: core::unread_reason(&e, &slice),
                }),
            }
        }
        Ok(listing)
    }

    async fn focused_window(&self) -> Result<WindowInfo, anyhow::Error> {
        let conn = self.conn().await?;
        let budget = core::Budget::new(
            core::LIST_BUDGET,
            "focused-window search",
            "the running applications",
        );
        let w = resolve_focused_window(&conn, &budget).await?;
        Ok(WindowInfo {
            app_pid: w.app_info.pid,
            app_name: w.app_info.name,
            title: w.title,
            key: w.key,
            surface: w.surface,
        })
    }

    async fn surface_geometry(
        &self,
        target: &TargetSpec,
    ) -> Result<SurfaceGeometry, anyhow::Error> {
        let mut budget = core::Budget::new(core::LIST_BUDGET, "surface lookup", target.describe());
        self.surface_geometry_with(target, &mut budget).await
    }

    async fn observe(&self, target: &TargetSpec) -> Result<Observation, anyhow::Error> {
        if matches!(target, TargetSpec::Screen) {
            return Err(core::taxonomy_error(
                core::ERR_UNSUPPORTED,
                "observe needs a window target — use apps/windows to pick one",
            ));
        }
        let conn = self.conn().await?;
        // One clock for the whole observation, resolution included, so the budget the
        // model is told about is the one the call actually pays.
        let mut budget = core::Budget::new(core::OBSERVE_BUDGET, "observation", target.describe());
        let window = resolve_surface(&conn, target, &mut budget).await?;
        let (root, _handles, mut truncation) =
            build_tree(&window.object_ref, &conn, &budget).await?;
        // The header shows the window's title, so it is capped like a tree name and a
        // cut is counted in the same ceiling.
        let window_title = window.title.map(|title| truncation.keep_name(&title));
        Ok(Observation {
            app_name: window.app_name,
            window_title,
            surface: window.surface,
            root,
            truncation,
        })
    }

    async fn act_on_element(
        &self,
        target: &TargetSpec,
        locator: &Locator,
        act: ElementAct,
    ) -> Result<Impact, anyhow::Error> {
        let conn = self.conn().await?;
        if matches!(target, TargetSpec::Screen) {
            return Err(core::screen_act_error());
        }
        // One clock for the whole call, resolution included: the action observes the
        // surface it addresses and then performs one exchange on it.
        let mut budget = core::Budget::new(core::ACT_BUDGET, "element action", target.describe());
        let (window_ref, app) = resolve_window(&conn, target, &budget).await?;
        let proxy = window_ref
            .as_accessible_proxy(&conn)
            .await
            .map_err(|e| dbus_err(e, "open window"))?;
        let (app_name, title) = surface_identity(target, &proxy, &app).await;
        core::name_surface(&mut budget, &app_name, title.as_deref());
        let (root, handles, _truncation) = build_tree(&window_ref, &conn, &budget).await?;
        let matched = core::resolve_locator_checked(&root, locator)?;
        if handles.is_empty() {
            return Err(core::taxonomy_error(
                core::ERR_STALE_ELEMENT,
                "element tree is empty — the window may have closed; re-observe",
            ));
        }
        let element_ref = &handles[core::pre_order_index(&root, matched)];
        let element = element_ref
            .as_accessible_proxy(&conn)
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

    async fn raw_input(
        &self,
        target: &TargetSpec,
        input: RawInput,
    ) -> Result<Impact, anyhow::Error> {
        // Argument bounds first, before the channel check or any window lookup.
        core::check_raw_input(&input)?;
        if wayland() {
            return Err(core::taxonomy_error(
                core::ERR_DEGRADED,
                "this is a Wayland session, which cannot receive synthesized input: XTEST would \
                 reach only the XWayland windows, never the native ones on screen — use ref-based \
                 accessibility actions where the app exposes them",
            ));
        }
        // Input that lands where the focus is: aimed at one window, it must reach
        // that window or not go at all. A coordinate click and a point-less wheel
        // are deliberately unpinned global input. One budget covers the whole
        // resolution, so raising a window and mapping its frame cannot stack two.
        let mut budget = core::Budget::new(core::LIST_BUDGET, "surface lookup", target.describe());
        if core::pinned_input(target, &input) {
            let conn = self.conn().await?;
            let (window_ref, _app) = resolve_window(&conn, target, &budget).await?;
            raise_window(&conn, &window_ref, target).await?;
        }
        let geo = match &input {
            RawInput::Click { .. }
            | RawInput::Scroll { point: Some(_), .. }
            | RawInput::Drag { .. } => Some(self.surface_geometry_with(target, &mut budget).await?),
            _ => None,
        };
        with_block_in_place(|| raw_input_blocking(input, geo.as_ref()))
    }

    async fn cursor_position(&self) -> Result<(f64, f64), anyhow::Error> {
        if wayland() {
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
    /// observe for the exact tree). Both capture routes crop the same way:
    /// through [`window_still`], which reconciles the accessibility and pixel
    /// spaces before cropping. Every buffer is charged to the call's `budget`.
    async fn capture(
        &self,
        target: &TargetSpec,
        budget: &CaptureBudget,
    ) -> Result<Capture, anyhow::Error> {
        match target {
            TargetSpec::Screen => Ok(self.capture_screen_channel(budget).await?.0),
            other => {
                let conn = self.conn().await?;
                // The window is resolved before any capture work: a hung target is
                // answered here rather than after a whole screen was read for nothing.
                let mut lookup =
                    core::Budget::new(core::LIST_BUDGET, "surface lookup", other.describe());
                let window = resolve_surface(&conn, other, &mut lookup).await?;
                let (screen_cap, screen_geo) = self.capture_screen_channel(budget).await?;
                // On Wayland that capture may have sat behind a person's consent dialog, which
                // is not the target failing to answer: the scale read gets its clock from here,
                // on the surface the call already resolved.
                let lookup = core::Budget::new(
                    core::LIST_BUDGET,
                    "surface lookup",
                    lookup.target().to_string(),
                );
                window_still(
                    &conn,
                    &window.surface,
                    &screen_cap,
                    &screen_geo,
                    budget,
                    &lookup,
                )
                .await
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
