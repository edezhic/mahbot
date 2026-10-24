//! Plain Win32 window facts for the Windows backend: which top-level windows an
//! application owns, every target's frame in physical pixels, the virtual-desktop
//! rect, process names, and the two refusals actuation owes the caller (an
//! elevated window, the secure desktop).
//!
//! Runs on the apartment thread, like every other platform call: a target is
//! resolved to an `HWND` here, used, and dropped — nothing stores a platform
//! handle across jobs.
//!
//! A window's identity is its own OS handle plus the pid that owned it when it was
//! enumerated ([`window_key`]): the handle stays valid across moves, renames and
//! re-enumeration, so a `TargetSpec::Window` either lands on that very window or
//! fails with [`core::ERR_SURFACE_GONE`] — never on another window of the
//! application, by title or by position in a listing. `TargetSpec::App` names an
//! application rather than a window and is resolved per call: its own active
//! window, else its frontmost one.

use super::super::core::{
    self, AppInfo, SurfaceGeometry, TargetSpec, UnreadWindows, WindowInfo, WindowKey, WindowListing,
};
use std::sync::{Once, OnceLock};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, POINT, RECT};
use windows::Win32::Graphics::Dwm::{
    DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute,
};
use windows::Win32::Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation};
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, DESKTOP_CONTROL_FLAGS, DESKTOP_READOBJECTS, GetUserObjectInformationW,
    OpenInputDesktop, UOI_NAME,
};
use windows::Win32::System::Threading::{
    GetCurrentProcessId, OpenProcess, OpenProcessToken, PROCESS_NAME_WIN32,
    PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GA_ROOT, GWL_EXSTYLE, GetAncestor, GetForegroundWindow, GetSystemMetrics,
    GetWindowLongW, GetWindowRect, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
    IsIconic, IsWindow, IsWindowVisible, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN, SetForegroundWindow, WS_EX_TOOLWINDOW, WindowFromPoint,
};
use windows::core::BOOL;

/// How long a foreground change is given to land, and how often it is re-checked.
const FOREGROUND_RETRIES: usize = 5;
const FOREGROUND_PAUSE: std::time::Duration = std::time::Duration::from_millis(20);

/// A resolved window target: everything the other modules need about one window.
pub(super) struct TargetWindow {
    pub hwnd: HWND,
    pub pid: u32,
    pub app_name: String,
    pub title: String,
    pub surface: SurfaceGeometry,
}

/// Ask for per-monitor-v2 DPI awareness once, eagerly, before any coordinate is
/// read. A GUI run already has it (winit sets it when the dashboard starts) and
/// the call then fails harmlessly, leaving the existing awareness in place.
pub(super) fn ensure_dpi_aware() {
    static ONCE: Once = Once::new();
    // SAFETY: no pointer or handle crosses this boundary, and the only failure (an
    // awareness already set by the GUI's event loop) is deliberately ignored.
    ONCE.call_once(|| unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    });
}

/// Whether an interactive desktop is attached at all: the channel question
/// [`super::Backend::capture_available`] answers. It is the same desktop check
/// actuation refuses on, so a secure, locked or disconnected desktop answers "no"
/// here too; a single window that refuses capture is not this.
pub(super) fn interactive_desktop_available() -> bool {
    ensure_input_desktop().is_ok()
}

// ── Target resolution ───────────────────────────────────────────────────

/// Resolve a target to its window. `Focused` follows the foreground window;
/// `App` re-resolves the application's own window per call; `Window` lands on the
/// exact window its key names or fails; `Screen` is a capture surface, not a
/// window.
pub(super) fn resolve(target: &TargetSpec) -> anyhow::Result<TargetWindow> {
    match target {
        TargetSpec::Screen => Err(core::screen_act_error()),
        TargetSpec::Focused => {
            // SAFETY: no arguments and no out-pointer; a null handle means no
            // foreground window and is refused below.
            let hwnd = unsafe { GetForegroundWindow() };
            if hwnd.0.is_null() {
                return Err(core::taxonomy_error(
                    core::ERR_DEGRADED,
                    "no foreground window — the session is locked, the UAC secure desktop is up, \
                     or no window has focus; ask the owner to return to their desktop",
                ));
            }
            describe(hwnd)
        }
        TargetSpec::App { app_pid, app_name } => {
            let windows = app_windows(*app_pid, app_name);
            let window = windows
                .iter()
                .find(|w| w.active)
                .or_else(|| windows.first())
                .ok_or_else(|| core::surface_gone(app_name))?;
            describe(window.hwnd)
        }
        TargetSpec::Window { app_name, key, .. } => resolve_key(app_name, key),
    }
}

/// Resolve a stored window key to exactly the window it names, or refuse.
///
/// The key carries the window's own handle and the pid that owned it, so a moved,
/// resized or renamed window keeps resolving while a closed or replaced one does
/// not. There is no fallback: a title match or another window of the application
/// would be a different surface than the caller named.
fn resolve_key(app_name: &str, key: &WindowKey) -> anyhow::Result<TargetWindow> {
    let Some((pid, hwnd)) = parse_key(key) else {
        return Err(core::surface_gone(app_name));
    };
    let alive =
        // SAFETY: `hwnd` is a plain handle value from the key; the call only tests it.
        unsafe { IsWindow(Some(hwnd)) }.as_bool() && pid_of(hwnd) == pid;
    if !alive {
        return Err(core::surface_gone(app_name));
    }
    describe(hwnd)
}

/// The identity a [`WindowKey`] carries (`hwnd:<pid>:<handle>`), and the one place
/// it is spelled. The handle alone is normally unique, but Windows may recycle one
/// after its window closes: the pid is the guard, and a recycled handle re-creating
/// the same pid+handle pair while a stale key is held is not addressable here.
fn window_key(hwnd: HWND, pid: u32) -> WindowKey {
    WindowKey::new(format!("hwnd:{pid}:{}", hwnd.0 as usize))
}

/// Split a [`WindowKey`] back into its owning pid and handle. `None` when the text
/// is not one this backend produced.
fn parse_key(key: &WindowKey) -> Option<(u32, HWND)> {
    let (pid, hwnd) = key.as_str().strip_prefix("hwnd:")?.split_once(':')?;
    let hwnd = hwnd.parse::<usize>().ok()?;
    Some((pid.parse().ok()?, HWND(hwnd as *mut std::ffi::c_void)))
}

/// Window geometry for a target: the window's own frame, or the virtual-desktop
/// bounding rect (which can have a negative origin) for `Screen`.
pub(super) fn surface_geometry(target: &TargetSpec) -> anyhow::Result<SurfaceGeometry> {
    match target {
        TargetSpec::Screen => screen_geometry(),
        other => Ok(resolve(other)?.surface),
    }
}

/// The virtual desktop across every display, origin at its min corner.
pub(super) fn screen_geometry() -> anyhow::Result<SurfaceGeometry> {
    // SAFETY: four index-only calls with no out-pointers; a non-positive metric is
    // refused below rather than trusted.
    unsafe {
        let (x, y) = (
            GetSystemMetrics(SM_XVIRTUALSCREEN),
            GetSystemMetrics(SM_YVIRTUALSCREEN),
        );
        let (width, height) = (
            GetSystemMetrics(SM_CXVIRTUALSCREEN),
            GetSystemMetrics(SM_CYVIRTUALSCREEN),
        );
        if width <= 0 || height <= 0 {
            return Err(core::taxonomy_error(
                core::ERR_DESKTOP_UNAVAILABLE,
                format!("{SECURE_DESKTOP_REASON} — there is no screen to describe"),
            ));
        }
        Ok(SurfaceGeometry {
            x: f64::from(x),
            y: f64::from(y),
            width: f64::from(width),
            height: f64::from(height),
        })
    }
}

/// The window a screen coordinate lands on, for the elevated-window check.
pub(super) fn top_level_at(x: i32, y: i32) -> Option<HWND> {
    // SAFETY: the point is passed by value; a null handle means no window there.
    let hwnd = unsafe { WindowFromPoint(POINT { x, y }) };
    if hwnd.0.is_null() {
        return None;
    }
    // SAFETY: `hwnd` is an OS handle this frame does not own, checked non-null just
    // above; `GA_ROOT` only answers an ancestor, null when the window has none.
    let root = unsafe { GetAncestor(hwnd, GA_ROOT) };
    Some(if root.0.is_null() { hwnd } else { root })
}

/// The window that has focus: where a key chord and synthetic typing go.
pub(super) fn foreground() -> Option<HWND> {
    // SAFETY: no arguments and no out-pointer; a null handle is filtered out below.
    let hwnd = unsafe { GetForegroundWindow() };
    (!hwnd.0.is_null()).then_some(hwnd)
}

/// Refuse input aimed at a window that is not in front: typing, a drag or a wheel at
/// a point would land in whatever is. Windows may refuse the foreground change this
/// follows (a background process), and then the honest answer is to click first.
pub(super) fn ensure_foreground(hwnd: HWND) -> anyhow::Result<()> {
    // SAFETY: no arguments; the handle is only compared, never dereferenced.
    if unsafe { GetForegroundWindow() } == hwnd {
        return Ok(());
    }
    let title = title_of(hwnd);
    Err(core::not_brought_forward(core::window_description(
        &app_name(pid_of(hwnd), &title),
        &title,
    )))
}

/// Bring a window to the foreground before input aimed at it. Best-effort: Windows
/// may refuse foreground changes for a background process. Every pinned action —
/// typing, a drag, a wheel at a point — then verifies with [`ensure_foreground`] and
/// fails honestly; a coordinate click is deliberately unpinned and only raises.
pub(super) fn activate(hwnd: HWND) {
    // SAFETY: no arguments; the handle is only compared, never dereferenced.
    if unsafe { GetForegroundWindow() } == hwnd {
        return;
    }
    // SAFETY: `hwnd` is a handle the caller resolved from the OS and does not own;
    // a refusal is a FALSE this ignores, and `ensure_foreground` is the real check.
    unsafe {
        let _ = SetForegroundWindow(hwnd);
    }
    // Windows applies some foreground changes asynchronously; give it a moment
    // before the caller's check turns a late one into a refusal.
    for _ in 0..FOREGROUND_RETRIES {
        // SAFETY: as above — a no-argument query whose result is only compared.
        if unsafe { GetForegroundWindow() } == hwnd {
            return;
        }
        std::thread::sleep(FOREGROUND_PAUSE);
    }
}

// ── Actuation refusals ──────────────────────────────────────────────────

/// The two platform refusals actuation owes the caller, checked before any input
/// or accessibility action: the secure desktop (see [`ensure_input_desktop`]) and
/// an elevated window. `hwnd` is the window the input actually lands on — `None`
/// is the desktop itself, which no process owns and which cannot be elevated.
pub(super) fn ensure_actuation_allowed(hwnd: Option<HWND>) -> anyhow::Result<()> {
    ensure_input_desktop()?;
    let Some(hwnd) = hwnd else {
        return Ok(());
    };
    if hwnd.0.is_null() || own_process_is_elevated() {
        return Ok(());
    }
    let pid = pid_of(hwnd);
    if is_own_process(pid) || !process_is_elevated(pid) {
        return Ok(());
    }
    let title = title_of(hwnd);
    let app = process_name(pid).unwrap_or_else(|| format!("pid {pid}"));
    Err(core::permission_denied(format!(
        "{} belongs to {app} (pid {pid}), which runs elevated — this non-elevated process cannot \
         send it input or drive its accessibility elements (Windows blocks both); {}ask the owner \
         to run that application without elevation",
        core::window_description("", &title),
        if super::super::capture_known_available() {
            "a capture of the window may still work — "
        } else {
            ""
        }
    )))
}

/// Refuse an accessibility ACTION aimed at a window of mahbot itself.
///
/// UI Automation answers a call addressed to our own process in-process, and this
/// backend has no path onto the interface thread the dashboard owns (iced's event
/// loop exposes no way to run a closure on itself), so an element action issued
/// from the accessibility thread can freeze the dashboard or abort the process —
/// and a trap cannot be caught. The action is refused rather than risk the
/// product. Listing, reading and capturing our own windows, and raw input that
/// reaches them, are unaffected.
pub(super) fn ensure_not_own_window(hwnd: HWND) -> anyhow::Result<()> {
    if !is_own_process(pid_of(hwnd)) {
        return Ok(());
    }
    let title = title_of(hwnd);
    Err(core::taxonomy_error(
        core::ERR_SELF_TARGET,
        format!(
            "{} belongs to mahbot itself: this process cannot drive its own accessibility elements \
             safely, because a same-process element action is answered in-process and there is no \
             way onto the interface thread — it could freeze the dashboard or kill the service, so \
             it is refused instead. Our windows stay listed, readable and capturable, and \
             coordinate input still reaches them",
            core::window_description("", &title)
        ),
    ))
}

/// Refuse while the secure desktop or the lock screen owns the input desktop.
fn ensure_input_desktop() -> anyhow::Result<()> {
    // The default desktop is not localized; when the secure desktop is up,
    // opening the input desktop fails with access denied instead.
    let desktop =
        // SAFETY: no out-pointer; the returned desktop handle is owned by this function
        // and closed below, and a failed open carries no handle at all.
        unsafe { OpenInputDesktop(DESKTOP_CONTROL_FLAGS::default(), false, DESKTOP_READOBJECTS) };
    let Ok(desktop) = desktop else {
        return Err(secure_desktop_error());
    };
    let mut name = [0u16; 64];
    // SAFETY: the object is the live desktop handle from above, which this call
    // only queries; `name` is a live `[u16; 64]` local whose own byte length is
    // passed as the cap, so the call writes at most those 64 UTF-16 units.
    let read = unsafe {
        GetUserObjectInformationW(
            windows::Win32::Foundation::HANDLE(desktop.0),
            UOI_NAME,
            Some(name.as_mut_ptr().cast()),
            (name.len() * 2) as u32,
            None,
        )
    };
    // SAFETY: `desktop` came from `OpenInputDesktop` above, was only queried since,
    // and is released here exactly once.
    unsafe {
        let _ = CloseDesktop(desktop);
    }
    if read.is_err() {
        return Err(secure_desktop_error());
    }
    let end = name.iter().position(|c| *c == 0).unwrap_or(name.len());
    if String::from_utf16_lossy(&name[..end]).eq_ignore_ascii_case("Default") {
        return Ok(());
    }
    Err(secure_desktop_error())
}

/// What a session without an interactive desktop is, named once: the core gate
/// ([`super::core::ERR_DESKTOP_UNAVAILABLE`]) and every Windows refusal spell it the
/// same way.
pub(super) const SECURE_DESKTOP_REASON: &str = "the UAC secure desktop, the lock screen or a \
     non-interactive session owns the desktop";

fn secure_desktop_error() -> anyhow::Error {
    core::taxonomy_error(
        core::ERR_DESKTOP_UNAVAILABLE,
        format!(
            "{SECURE_DESKTOP_REASON} — nothing can be read or actuated there; ask the owner to return to their desktop"
        ),
    )
}

fn own_process_is_elevated() -> bool {
    static OWN: OnceLock<bool> = OnceLock::new();
    // SAFETY: the only unsafe call is the argument, `GetCurrentProcessId`, which has
    // no arguments and no failure mode.
    *OWN.get_or_init(|| unsafe { process_is_elevated(GetCurrentProcessId()) })
}

/// Whether a process's token is elevated. A failed query answers "no": the refusal
/// must never fire on a guess, and the action itself then fails honestly.
fn process_is_elevated(pid: u32) -> bool {
    // SAFETY: every handle here is the OS's own return, checked or closed in this
    // function — `token` is only read after `OpenProcessToken` succeeded — and
    // `GetTokenInformation` fills a live `TOKEN_ELEVATION` whose byte count is
    // `size_of` of that same type.
    unsafe {
        let Ok(process) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return false;
        };
        let mut token = HANDLE::default();
        let opened = OpenProcessToken(process, TOKEN_QUERY, &raw mut token).is_ok();
        let _ = CloseHandle(process);
        if !opened {
            return false;
        }
        let mut elevation = TOKEN_ELEVATION::default();
        let mut size = 0u32;
        let read = GetTokenInformation(
            token,
            TokenElevation,
            Some(std::ptr::from_mut(&mut elevation).cast()),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &raw mut size,
        )
        .is_ok();
        let _ = CloseHandle(token);
        read && elevation.TokenIsElevated != 0
    }
}

// ── Enumeration ─────────────────────────────────────────────────────────

/// One enumerated top-level window.
struct Enumerated {
    hwnd: HWND,
    pid: u32,
    app_name: String,
    title: String,
    active: bool,
    /// The frame, or why it could not be read: a window without one is left out of
    /// a listing but is still reported there (see [`list_windows`]).
    surface: anyhow::Result<SurfaceGeometry>,
}

/// Every running application that owns at least one top-level window, in
/// front-to-back order.
pub(super) fn enumerate_apps() -> Vec<AppInfo> {
    let mut apps: Vec<AppInfo> = Vec::new();
    for win in top_level_windows() {
        // Name as well as pid: one host process (a UWP frame host) serves several apps.
        if apps
            .iter()
            .any(|a| a.pid == Some(win.pid) && a.name.eq_ignore_ascii_case(&win.app_name))
        {
            continue;
        }
        apps.push(AppInfo {
            pid: Some(win.pid),
            name: win.app_name,
        });
    }
    apps
}

/// Top-level windows, optionally narrowed to one application's. A window whose frame
/// cannot be read has no coordinates to act on, so it is not listed — a zeroed
/// surface would silently map every normalized point to the origin — but the
/// application that lost one is: [`WindowListing::unread`] carries how many of its
/// windows could not be read and the first error.
/// One application's windows that a listing could not read, accumulated to be
/// reported as one note rather than one note per lost window.
struct LostFrames {
    app_name: String,
    count: usize,
    first_error: String,
}

pub(super) fn list_windows(app: Option<&AppInfo>) -> WindowListing {
    let windows = match app {
        Some(app) => app_windows(app.pid, &app.name),
        None => top_level_windows(),
    };
    let mut listing = WindowListing::default();
    let mut lost: Vec<LostFrames> = Vec::new();
    for win in windows {
        let surface = match win.surface {
            Ok(surface) => surface,
            Err(error) => {
                match lost
                    .iter_mut()
                    .find(|entry| entry.app_name.eq_ignore_ascii_case(&win.app_name))
                {
                    Some(entry) => entry.count += 1,
                    None => lost.push(LostFrames {
                        app_name: win.app_name.clone(),
                        count: 1,
                        first_error: format!("{error:#}"),
                    }),
                }
                continue;
            }
        };
        listing.windows.push(WindowInfo {
            app_pid: Some(win.pid),
            app_name: win.app_name,
            title: win.title,
            key: window_key(win.hwnd, win.pid),
            surface,
        });
    }
    listing.unread = lost
        .into_iter()
        .map(|entry| UnreadWindows {
            app_name: entry.app_name,
            reason: format!(
                "{} {} no readable frame (first error: {})",
                entry.count,
                if entry.count == 1 {
                    "window has"
                } else {
                    "windows have"
                },
                entry.first_error
            ),
        })
        .collect();
    listing
}

/// The focused window of the frontmost application.
pub(super) fn focused_window() -> anyhow::Result<WindowInfo> {
    // SAFETY: no arguments and no out-pointer; a null handle is refused below.
    let hwnd = unsafe { GetForegroundWindow() };
    if hwnd.0.is_null() {
        return Err(core::taxonomy_error(
            core::ERR_DEGRADED,
            "no foreground window found — the session is locked, the UAC secure desktop is up, or \
             no window has focus; ask the owner to return to their desktop, or enumerate \
             apps/windows to pick a surface",
        ));
    }
    let resolved = describe(hwnd)?;
    // The key is the handle itself, so a foreground window outside the filtered
    // listing (a tool palette, an untitled surface) needs no position in it to be
    // addressable later.
    Ok(WindowInfo {
        app_pid: Some(resolved.pid),
        app_name: resolved.app_name,
        title: resolved.title,
        key: window_key(hwnd, resolved.pid),
        surface: resolved.surface,
    })
}

/// One application's top-level windows (by pid when known, else by name), in
/// front-to-back order.
fn app_windows(pid: Option<u32>, app_name: &str) -> Vec<Enumerated> {
    top_level_windows()
        .into_iter()
        .filter(|win| match pid {
            Some(pid) => {
                win.pid == pid
                    && (app_name.is_empty() || win.app_name.eq_ignore_ascii_case(app_name))
            }
            None => win.app_name.eq_ignore_ascii_case(app_name),
        })
        .collect()
}

/// All top-level application windows, front to back, plus the foreground window when
/// the filter below would hide it (see the note there).
///
/// Visible and titled, which is what tells an application window apart from the
/// helper windows every toolkit leaves lying around. Owned windows are kept: a
/// modal dialog is exactly the surface an agent needs. Cloaked windows (another
/// virtual desktop, a suspended UWP app) are dropped — they have no pixels.
fn top_level_windows() -> Vec<Enumerated> {
    let mut handles: Vec<HWND> = Vec::new();
    // SAFETY: `handles` is a live local that outlives this synchronous enumeration,
    // and `collect_window` receives its address through the `LPARAM`; nothing else
    // touches the vector until `EnumWindows` returns.
    unsafe {
        let _ = EnumWindows(
            Some(collect_window),
            LPARAM(std::ptr::from_mut(&mut handles).cast::<std::ffi::c_void>() as isize),
        );
    }
    // SAFETY: no arguments and no out-pointer.
    let foreground = unsafe { GetForegroundWindow() };
    let mut windows: Vec<Enumerated> = handles
        .into_iter()
        .filter(|hwnd| is_application_window(*hwnd))
        .map(|hwnd| enumerate(hwnd, foreground))
        .collect();
    // The foreground window joins the list even when the filter above hides it (a tool
    // palette, an untitled surface): `observe` pins the focused surface, and the
    // listing should name the surface the agent is looking at.
    if foreground_joins_listing(&windows, foreground) {
        windows.insert(0, enumerate(foreground, foreground));
    }
    windows
}

/// Whether the foreground window is a surface the filter above hid but the listing
/// must still carry: it is what `observe` pins. Invisible and cloaked windows stay
/// out — they can have no pixels.
fn foreground_joins_listing(windows: &[Enumerated], foreground: HWND) -> bool {
    if foreground.0.is_null() || windows.iter().any(|win| win.hwnd == foreground) {
        return false;
    }
    // SAFETY: `foreground` is an OS handle and visibility is a query with no pointer.
    unsafe { IsWindowVisible(foreground) }.as_bool() && !is_cloaked(foreground)
}

/// One window's enumerated facts.
fn enumerate(hwnd: HWND, foreground: HWND) -> Enumerated {
    let pid = pid_of(hwnd);
    let title = title_of(hwnd);
    Enumerated {
        hwnd,
        pid,
        app_name: app_name(pid, &title),
        title,
        active: hwnd == foreground,
        surface: surface_of(hwnd),
    }
}

/// An application's display name. UWP and WinUI apps are hosted by
/// `ApplicationFrameHost.exe`, one process for several apps, so the window's own
/// title is the app there — the exe stem would merge them into one. Consequence: a
/// frame-hosted app whose title changes after enumeration makes its stored `a`
/// target stale (a `w` target keeps the window's own handle and is unaffected), and
/// re-enumerating is the fix — ordinary apps are named by their executable.
fn app_name(pid: u32, title: &str) -> String {
    match process_name(pid) {
        Some(stem) if stem.eq_ignore_ascii_case("ApplicationFrameHost") && !title.is_empty() => {
            title.to_string()
        }
        Some(stem) => stem,
        None => format!("pid {pid}"),
    }
}

// SAFETY: only ever installed as `EnumWindows`' callback, whose `lparam` this
// module sets to the address of its own `Vec<HWND>`; `hwnd` is an OS handle this
// frame does not own.
unsafe extern "system" fn collect_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
    // Always continue: a `FALSE` here would make `EnumWindows` report an error.
    // SAFETY: `lparam` is that `&mut Vec<HWND>` from `top_level_windows`, still
    // live and exclusively reachable here, so it is aligned and pushable.
    let handles = unsafe { &mut *(lparam.0 as *mut Vec<HWND>) };
    handles.push(hwnd);
    BOOL(1)
}

fn is_application_window(hwnd: HWND) -> bool {
    // SAFETY: no out-pointers in the group; `hwnd` is a handle enumeration just
    // reported, and a window closed since then merely fails these queries.
    unsafe {
        if !IsWindowVisible(hwnd).as_bool() || GetWindowTextLengthW(hwnd) == 0 {
            return false;
        }
        // Tool windows are palettes and floating bars; they are still reachable
        // through the focused-window path when the owner actually uses one.
        let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        if ex_style & WS_EX_TOOLWINDOW.0 != 0 {
            return false;
        }
        !is_cloaked(hwnd)
    }
}

fn is_cloaked(hwnd: HWND) -> bool {
    let mut cloaked = 0u32;
    // SAFETY: `cloaked` is a live `u32` local and the byte count is `size_of` of
    // that same type, which is what `DWMWA_CLOAKED` writes.
    let queried = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            std::ptr::from_mut(&mut cloaked).cast(),
            std::mem::size_of::<u32>() as u32,
        )
    };
    queried.is_ok() && cloaked != 0
}

/// A window's surface in physical pixels: the visible frame without the invisible
/// resize border (`DWMWA_EXTENDED_FRAME_BOUNDS`), falling back to `GetWindowRect`.
fn surface_of(hwnd: HWND) -> anyhow::Result<SurfaceGeometry> {
    let mut rect = RECT::default();
    // SAFETY: `rect` is a live `RECT` local and the byte count is `size_of` of that
    // same type, the shape `DWMWA_EXTENDED_FRAME_BOUNDS` writes.
    let frame = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            std::ptr::from_mut(&mut rect).cast(),
            std::mem::size_of::<RECT>() as u32,
        )
    };
    if frame.is_err() {
        return window_rect(hwnd);
    }
    Ok(geometry(rect))
}

/// The window rect, invisible resize border included — the other size a capture
/// item may come back in (see [`super::capture::align_to_surface`]).
pub(super) fn window_rect(hwnd: HWND) -> anyhow::Result<SurfaceGeometry> {
    let mut rect = RECT::default();
    // SAFETY: `rect` is a live local of exactly the type the call fills.
    unsafe { GetWindowRect(hwnd, &raw mut rect) }.map_err(|e| {
        core::taxonomy_error(
            core::ERR_NOT_MATCHED,
            format!("window is gone or has no frame ({e}) — re-enumerate windows"),
        )
    })?;
    Ok(geometry(rect))
}

fn geometry(rect: RECT) -> SurfaceGeometry {
    SurfaceGeometry {
        x: f64::from(rect.left),
        y: f64::from(rect.top),
        width: f64::from(rect.right - rect.left),
        height: f64::from(rect.bottom - rect.top),
    }
}

/// Everything the caller needs about one window, refusing a minimized one: its
/// frame sits off-screen and its capture is black, so every action on it would be
/// a lie.
fn describe(hwnd: HWND) -> anyhow::Result<TargetWindow> {
    let pid = pid_of(hwnd);
    let title = title_of(hwnd);
    // A handle that vanished since resolution answers FALSE, which reads as "not
    // minimized".
    // SAFETY: one handle argument and no out-pointer.
    if unsafe { IsIconic(hwnd) }.as_bool() {
        return Err(core::minimized_window(&title));
    }
    Ok(TargetWindow {
        hwnd,
        pid,
        app_name: app_name(pid, &title),
        title,
        surface: surface_of(hwnd)?,
    })
}

fn pid_of(hwnd: HWND) -> u32 {
    let mut pid = 0u32;
    // SAFETY: `pid` is a live `u32` local the call fills in place; a window that
    // vanished simply leaves the zero it started with.
    unsafe {
        GetWindowThreadProcessId(hwnd, Some(&raw mut pid));
    }
    pid
}

/// Whether `pid` is this very process. A window that vanished before the query
/// reports pid 0, which does not count as ours.
fn is_own_process(pid: u32) -> bool {
    // SAFETY: `GetCurrentProcessId` takes no arguments and has no failure mode.
    pid != 0 && pid == unsafe { GetCurrentProcessId() }
}

/// A window's title as a label: kept to the product's cap, the ellipsis marking the
/// cut. Used wherever a title is shown or named.
fn title_of(hwnd: HWND) -> String {
    core::cap_label(&title_text(hwnd))
}

/// A window's title as read — bounded by a product-sized buffer, never by what the
/// window reports, so a document-length title is not materialised. `observe` keeps
/// it through [`core::Truncation::keep_name`], which is what counts a cut title in
/// the observation header; a label is only marked.
pub(super) fn title_text(hwnd: HWND) -> String {
    // SAFETY: one handle argument, no out-pointer.
    let length = unsafe { GetWindowTextLengthW(hwnd) };
    if length <= 0 {
        return String::new();
    }
    // The reported length decides how much is ASKED FOR, never how much is allocated;
    // two UTF-16 units per character plus room for the ellipsis and the terminator.
    let wanted = core::MAX_NAME_CHARS * 2 + 2;
    let mut buffer = vec![0u16; (length as usize).min(wanted) + 1];
    // SAFETY: `buffer` is alive here and the binding passes its own length as the cap,
    // so the call cannot write past it.
    let written = unsafe { GetWindowTextW(hwnd, &mut buffer) };
    String::from_utf16_lossy(&buffer[..written.max(0) as usize])
}

/// An application's display name: its executable's file stem ("notepad"), which is
/// what `windows { app: … }` matches and what the target lines show.
fn process_name(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    let mut path = vec![0u16; 1024];
    let mut length = u32::try_from(path.len()).ok()?;
    // SAFETY: `process` is the checked handle from `.ok()?` and is closed before the
    // block ends; `path` is a live 1024-unit buffer whose own length is passed as
    // the cap, and `length` is a local in/out count the call keeps within it.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let queried = QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(path.as_mut_ptr()),
            &raw mut length,
        );
        let _ = CloseHandle(process);
        queried.ok()?;
    }
    let full = String::from_utf16_lossy(&path[..length as usize]);
    let name = std::path::Path::new(&full)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())?;
    if name.is_empty() { None } else { Some(name) }
}
