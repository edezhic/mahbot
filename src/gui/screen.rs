//! The screen a fresh window opens on: its work area — what the platform's own panels leave of it
//! (the menu bar and Dock, the taskbar, a desktop panel) — in logical px, the unit the window's
//! size is in. A fresh window is capped to it (`fresh_size` in [`super`]); a saved geometry is
//! never measured against it.
//!
//! Which screen that is follows the platform: a fresh window opens at the desktop's own top-left
//! corner (`WindowState::position`), so macOS reads the main screen and Windows the primary
//! monitor, while X11 measures the desktop as one rectangle (see each arm). A platform that
//! reports no work area — a Wayland session, an X11 one whose window manager is not EWMH —
//! answers `None` and leaves a fresh size uncapped.

/// The primary screen's work area, without the menu bar and Dock.
#[cfg(target_os = "macos")]
#[must_use]
#[expect(clippy::cast_possible_truncation)]
pub fn work_area() -> Option<iced::Size> {
    use objc2_app_kit::NSScreen;

    // AppKit is used on the main thread, which is where the window is created from.
    let Some(marker) = objc2::MainThreadMarker::new() else {
        tracing::warn!("the screen's work area is unknown: AppKit is not used on the main thread");
        return None;
    };
    let area = NSScreen::mainScreen(marker)?.visibleFrame().size;
    Some(iced::Size::new(area.width as f32, area.height as f32))
}

/// The primary monitor's work area, without the taskbar and any other app bar.
#[cfg(windows)]
#[must_use]
#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn work_area() -> Option<iced::Size> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MONITOR_DEFAULTTOPRIMARY, MONITORINFO, MonitorFromWindow,
    };
    use windows::Win32::UI::WindowsAndMessaging::IsProcessDPIAware;

    // SAFETY: the calls write only into the `MONITORINFO` handed to them, sized as the call
    // requires, and a failed one is never read from.
    unsafe {
        let monitor = MonitorFromWindow(HWND::default(), MONITOR_DEFAULTTOPRIMARY);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !GetMonitorInfoW(monitor, &raw mut info).as_bool() {
            return None;
        }
        let work = info.rcWork;
        let width = (work.right - work.left) as f32;
        let height = (work.bottom - work.top) as f32;
        // This process resolves no DPI itself until iced's event loop is created (winit sets it
        // there), so the system hands it a desktop already scaled to its DPI: the rectangle is
        // logical as it stands. An aware process gets device pixels, which the monitor's own DPI
        // turns back into the logical size the window is created with.
        let scale = if IsProcessDPIAware().as_bool() {
            monitor_dpi(monitor) / 96.0
        } else {
            1.0
        };
        Some(iced::Size::new(width / scale, height / scale))
    }
}

/// The monitor's effective DPI, or the unscaled 96 where the system reports none.
#[cfg(windows)]
#[expect(clippy::cast_precision_loss)]
fn monitor_dpi(monitor: windows::Win32::Graphics::Gdi::HMONITOR) -> f32 {
    use windows::Win32::UI::HiDpi::{GetDpiForMonitor, MDT_EFFECTIVE_DPI};

    let (mut x, mut y) = (0u32, 0u32);
    // SAFETY: the two out-pointers are the call's own outputs.
    match unsafe { GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &raw mut x, &raw mut y) } {
        Ok(()) => x as f32,
        Err(_) => 96.0,
    }
}

/// The desktop's work area, without the panels its EWMH window manager reserves for itself. One
/// rectangle covers the whole desktop: a multi-monitor X11 session has no per-monitor work area,
/// and a desktop wide or tall enough for two monitors is beyond the cap's reach anyway.
#[cfg(target_os = "linux")]
#[must_use]
#[expect(clippy::cast_precision_loss)]
pub fn work_area() -> Option<iced::Size> {
    use x11rb::connection::Connection as _;
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};

    // A Wayland session reports none: its X root is XWayland's own, not the screen.
    if crate::util::display_server::wayland() {
        return None;
    }
    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let root = conn.setup().roots.get(screen_num)?.root;
    let workarea = conn
        .intern_atom(false, b"_NET_WORKAREA")
        .ok()?
        .reply()
        .ok()?
        .atom;
    let reply = conn
        .get_property(false, root, workarea, AtomEnum::CARDINAL, 0, 4)
        .ok()?
        .reply()
        .ok()?;
    let values: Vec<u32> = reply.value32()?.collect();
    // The rectangle of the first desktop: an origin, then the size.
    let [_, _, width, height] = values[..] else {
        return None;
    };
    // The window's size is logical, while the work area is in device pixels on a scaled desktop —
    // the resource is what a desktop that scales publishes for clients that cannot read XSETTINGS.
    let scale = xft_dpi(&conn, root).map_or(1.0, |dpi| dpi / 96.0);
    Some(iced::Size::new(width as f32 / scale, height as f32 / scale))
}

/// A platform with no screen of its own to measure.
#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
#[must_use]
pub fn work_area() -> Option<iced::Size> {
    None
}

/// `Xft.dpi` from the root window's resources, the value a desktop sets when it scales.
#[cfg(target_os = "linux")]
fn xft_dpi(
    conn: &x11rb::rust_connection::RustConnection,
    root: x11rb::protocol::xproto::Window,
) -> Option<f32> {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};

    let reply = conn
        .get_property(
            false,
            root,
            AtomEnum::RESOURCE_MANAGER,
            AtomEnum::STRING,
            0,
            1024,
        )
        .ok()?
        .reply()
        .ok()?;
    String::from_utf8_lossy(&reply.value)
        .lines()
        .find_map(|line| line.strip_prefix("Xft.dpi:")?.trim().parse().ok())
}
