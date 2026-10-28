//! The dashboard's window on the platforms whose window is the app's own: a close HIDES it
//! and the product keeps running, so the ways back are the tray, a second launch and the
//! Dock (see `crate::window_requests`; the Dock's reopen is one of the delegate methods the
//! macOS quit interception installs — see `crate::shutdown`).
//!
//! The window's shape stays the platform's: macOS keeps the system's titled window — the only
//! window the system rounds and shadows by itself — with its whole title strip hidden (see
//! [`opened`]), and Windows asks the system to round the corners of a window that has no frame
//! (`main`). Linux keeps the platform's frame — the system's minimize and close, and a close
//! ends the product — so nothing there is ever hidden: the flag below stays false, and only the
//! animations read it.
//!
//! A hidden window still runs everything behind it — the agents, the pipeline, the channels —
//! and what they change lands in the window's state, so what comes back is current. What stops
//! is the animation: every animation that would otherwise re-arm its own redraw reads
//! [`hidden`], so no frame is produced for the sake of an animation while it is hidden.
use std::sync::atomic::{AtomicBool, Ordering};

// The hidden window itself: only the platforms whose window is the app's own hide one.
#[cfg(any(target_os = "macos", target_os = "windows"))]
use super::Message;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use iced::Task;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use iced::window;
// AppKit, where the window's own chrome is the app's to hide.
#[cfg(target_os = "macos")]
use objc2::{MainThreadMarker, rc::Retained};
#[cfg(target_os = "macos")]
use objc2_app_kit::{NSApplication, NSWindow};

/// Whether the window is hidden.
static HIDDEN: AtomicBool = AtomicBool::new(false);

/// Whether the dashboard's window is hidden: every animation that would re-arm its own
/// redraw stops while this is true.
pub(crate) fn hidden() -> bool {
    HIDDEN.load(Ordering::Relaxed)
}

/// Set the hidden flag — what hiding and showing the window do before they touch it, and what
/// a launch the updater restarted hidden starts with (see `main`).
pub fn set_hidden(hidden: bool) {
    HIDDEN.store(hidden, Ordering::Relaxed);
}

/// Whether the system started this process at login (`crate::autostart`).
#[cfg(any(target_os = "macos", target_os = "windows"))]
static LOGIN_START: AtomicBool = AtomicBool::new(false);

/// Record that the system started the process at login (`crate::autostart`). Such a launch is
/// not a second launch and never brings the window up by itself — see [`login_start`].
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub fn set_login_start(login_start: bool) {
    LOGIN_START.store(login_start, Ordering::Relaxed);
}

/// Whether this is a login start. It comes up as the tray icon only: the system asked for a
/// quiet start, so neither a failed boot nor a missing tray may surface the window here — the
/// tray, or a hand launch when there is none, is the way back.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn login_start() -> bool {
    LOGIN_START.load(Ordering::Relaxed)
}

/// Hide the dashboard's window. Everything behind it keeps running, the window is hidden the
/// way the platform hides one (its window state is kept, so nothing re-shows it), and the same
/// window comes back as it was left — the same size, position, page and state — when [`show`]
/// brings it up.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn hide() -> Task<Message> {
    set_hidden(true);
    window::latest().and_then(|id| window::set_mode(id, window::Mode::Hidden))
}

/// Bring the window up: out of the hidden state and out of the minimized one, in front —
/// including while another application is in front. Called for a window that has never been
/// drawn too: the one a launch just created, and the launch the updater restarted hidden.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn show() -> Task<Message> {
    set_hidden(false);
    // macOS: activating the application and lifting the window out of the minimized state is
    // what iced's own `gain_focus` cannot do — the platform refuses to activate an application
    // whose windows are all minimized, and a window brought back while the application is still
    // behind stays behind. Showing the window itself is the mode change below.
    #[cfg(target_os = "macos")]
    bring_forward();
    window::latest().and_then(|id| {
        let shown = window::set_mode(id, window::Mode::Windowed);
        // A minimized window ignores a focus request, so the restore is chained ahead of it.
        // On macOS that is `bring_forward`'s deminiaturize, and the focus request is not
        // needed: the window is made key when it is shown.
        #[cfg(target_os = "windows")]
        let shown = shown
            .chain(window::minimize(id, false))
            .chain(window::gain_focus(id));
        shown
    })
}

/// macOS: the window iced has just created — its title strip hidden and the window up, unless
/// this launch starts hidden. iced creates the window hidden (`main`), so no strip is ever
/// drawn: what AppKit adds over the top of the window goes before anything is shown, and the
/// platform's own corners and shadow are the only shape the window has.
///
/// The only thing that puts a fresh window up — `Opened` is subscribed before the window is
/// created, so the event cannot be missed — and a strip that cannot be found costs only the
/// strip: the window is shown either way.
#[cfg(target_os = "macos")]
pub(crate) fn opened() -> Task<Message> {
    hide_title_strip();
    if hidden() { Task::none() } else { show() }
}

/// Hide the title strip a titled window still has: the three window buttons, the separator, and the
/// strip's own drag. The strip draws nothing and clicks still reach the content, but AppKit still
/// moves the window on a drag anywhere in that band — over the top of the content, the rail's
/// control included. The window keeps moving the way this app moves it (`iced::window::drag`, which
/// a window that cannot be moved still honours) and keeps the platform's corners and shadow.
#[cfg(target_os = "macos")]
fn hide_title_strip() {
    use objc2_app_kit::{NSTitlebarSeparatorStyle, NSWindowButton};

    // AppKit is used on the main thread, which is where the dashboard's update runs.
    let Some(marker) = MainThreadMarker::new() else {
        tracing::warn!(
            "the window's title strip is left as it is: AppKit is not used on the main thread"
        );
        return;
    };
    let Some(window) = dashboard_window(marker) else {
        tracing::warn!(
            "the window's title strip is left as it is: the dashboard's window was not found"
        );
        return;
    };
    for button in [
        NSWindowButton::CloseButton,
        NSWindowButton::MiniaturizeButton,
        NSWindowButton::ZoomButton,
    ] {
        if let Some(button) = window.standardWindowButton(button) {
            button.setHidden(true);
        }
    }
    window.setTitlebarSeparatorStyle(NSTitlebarSeparatorStyle::None);
    // The strip's drag: no press over its band moves the window, and the app's own drags still do.
    window.setMovable(false);
}

/// The dashboard's window: the process's only miniaturizable one — the status item's own window
/// and AppKit's tooltip windows cannot be minimized — which is what finds it.
#[cfg(target_os = "macos")]
fn dashboard_window(marker: MainThreadMarker) -> Option<Retained<NSWindow>> {
    NSApplication::sharedApplication(marker)
        .windows()
        .iter()
        .find(|window| window.isMiniaturizable())
}

/// macOS: activates the application and brings its window to the front, out of the minimized
/// state as well, in one step.
#[cfg(target_os = "macos")]
fn bring_forward() {
    // The dashboard's update runs on the main thread, which is where AppKit is used; a call
    // from anywhere else is simply skipped.
    let Some(marker) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(marker);
    let window = dashboard_window(marker);
    // An application hidden with ⌘H stays hidden for an activation request: its windows come
    // back only once it is unhidden, which is what the Dock's own reopen does on this platform.
    app.unhide(None);
    // Deprecated in favour of `activate()`, which the platform grants only to an application
    // it considers driven by the user right now — not to one whose windows are all minimized,
    // which is exactly the state this lifts.
    #[expect(deprecated)]
    app.activateIgnoringOtherApps(true);
    let Some(window) = window else {
        return;
    };
    window.deminiaturize(None);
}
