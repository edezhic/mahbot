//! The tray icon: the product's own icon in the platform's status area — the macOS menu bar and the
//! Windows notification area — with a click that brings the window up and a native one-item menu
//! whose item quits. It is the way back to a window a close hid, and the product's only in-app
//! quit. Linux has no tray, and nothing here is compiled there. Nothing runs periodically: each
//! platform hands a gesture to the handlers [`install`] sets, and they only enqueue it. A tray that
//! cannot be created is not fatal — it is reported, the daemon runs without it, and a window then
//! never hides (a hidden window with no tray to bring it back would leave no way to quit).
use std::sync::atomic::{AtomicBool, Ordering};
use tray_icon::TrayIconBuilder;
use tray_icon::menu::{Menu, MenuEvent, MenuItem};

use crate::window_requests;

/// The menu's single item: the id its event is matched against, and its wording.
const QUIT_ITEM: &str = "quit";
const QUIT_ITEM_LABEL: &str = "Quit mahbot";
/// The icon's own name, set as the tooltip — the only label it carries.
const TOOLTIP: &str = "mahbot";
/// How long [`remove`] waits for the thread that owns the icon to take it away. Bounded because an
/// exiting process must not wait on it: a late removal leaves a stale icon on Windows, never a
/// hang.
const REMOVAL_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// Whether the icon exists — the product's way back to a hidden window; false when creation failed.
static AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Whether the tray icon is there: a window may only be hidden while it is (see `gui`).
pub(crate) fn available() -> bool {
    AVAILABLE.load(Ordering::Relaxed)
}

/// Install the icon, once per process, before the iced application runs. The menu is the same on
/// both platforms; each supplies its own click. Gestures go to [`crate::window_requests`], whose
/// channel [`crate::window_requests::init`] fills before this runs — an early gesture waits there.
pub fn install() {
    MenuEvent::set_event_handler(Some(|event: MenuEvent| {
        if event.id.as_ref() == QUIT_ITEM {
            window_requests::quit();
        }
    }));

    #[cfg(target_os = "macos")]
    macos::install();
    #[cfg(target_os = "windows")]
    windows::install();
}

// Each platform's own `remove` is documented there: the re-export is what both ends call — after
// the iced runtime returns (`main`), and the self-update tail, whose `exit(0)` runs no destructors.
#[cfg(target_os = "macos")]
pub use macos::remove;
#[cfg(target_os = "windows")]
pub use windows::remove;

/// The builder both platforms share: the native menu with its one item, the tooltip, and the click
/// that opens
/// the menu kept to a secondary click — a plain click belongs to the window it brings up.
fn builder() -> Result<TrayIconBuilder, String> {
    let menu = Menu::new();
    let quit = MenuItem::with_id(QUIT_ITEM, QUIT_ITEM_LABEL, true, None);
    menu.append(&quit)
        .map_err(|e| format!("its menu could not be built: {e}"))?;

    Ok(TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_menu_on_left_click(false)
        .with_tooltip(TOOLTIP))
}

fn report(reason: &str) {
    crate::boot::boot_diagnostic(format!("the tray icon was not created: {reason}"));
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{AVAILABLE, REMOVAL_WAIT, builder, report};
    use crate::window_requests;
    use block2::RcBlock;
    use dispatch2::DispatchQueue;
    use objc2::MainThreadMarker;
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_app_kit::{NSEvent, NSEventMask, NSStatusItem};
    use std::cell::RefCell;
    use std::ptr::NonNull;
    use std::sync::atomic::Ordering;
    use tray_icon::TrayIcon;

    /// What the main thread keeps for as long as the icon is in the menu bar.
    struct Installed {
        /// The tray itself; dropping it takes the status item out of the menu bar.
        tray: TrayIcon,
        /// The event-stream monitor reading the plain click, taken out with the item.
        monitor: Option<Retained<AnyObject>>,
    }

    thread_local! {
        /// The icon, on the thread that owns it: an AppKit status item belongs to the main thread.
        static ICON: RefCell<Option<Installed>> = const { RefCell::new(None) };
    }

    /// The status item can only be created on the main thread, with the application's own event
    /// loop running:
    /// click handling, menu tracking and every AppKit call behind them are that thread's work.
    /// [`install`](super::install) runs before iced's loop starts, so creation is queued onto the
    /// main thread.
    pub(super) fn install() {
        DispatchQueue::main().exec_async(|| match create() {
            Ok(installed) => {
                AVAILABLE.store(true, Ordering::Relaxed);
                ICON.with(|icon| *icon.borrow_mut() = Some(installed));
            }
            Err(reason) => report(&reason),
        });
    }

    fn create() -> Result<Installed, String> {
        let icon = crate::app_icon::tray_icon_image().ok_or("its image could not be built")?;
        let tray = builder()?
            .with_icon_templated(icon)
            .build()
            .map_err(|e| e.to_string())?;
        let marker = MainThreadMarker::new().ok_or("it can only be created on the main thread")?;
        let item = tray
            .ns_status_item()
            .ok_or("its status item is gone")?
            .clone();
        // The backing-store image replaces the one above at the same size in points: the item does
        // not move.
        if let (Some(image), Some(button)) =
            (crate::app_icon::tray_status_image(), item.button(marker))
        {
            button.setImage(Some(&image));
        }

        // A plain click never reaches the crate's own view — the status-bar button handles the
        // press itself and
        // keeps the release, while a secondary click reaches the view and opens the menu — so the
        // press is read from the event stream instead, before it is dispatched to a view.
        let watched = item.clone();
        let handler = RcBlock::new(move |event: NonNull<NSEvent>| -> *mut NSEvent {
            // SAFETY: the monitor hands over a live event for the duration of the call.
            if on_icon(&watched, unsafe { event.as_ref() }) {
                window_requests::show();
            }
            // Handed back untouched: the icon's own handling still needs it.
            event.as_ptr()
        });
        // SAFETY: the block hands back the pointer to the `NSEvent` it was given.
        let monitor = unsafe {
            NSEvent::addLocalMonitorForEventsMatchingMask_handler(
                NSEventMask::LeftMouseDown,
                &handler,
            )
        };

        Ok(Installed { tray, monitor })
    }

    /// Whether the event landed on this item's own button. A monitor sees every mouse event this
    /// application gets, and the status bar gives each item a window of its own, so the window name
    /// is the answer.
    fn on_icon(item: &NSStatusItem, event: &NSEvent) -> bool {
        let Some(marker) = MainThreadMarker::new() else {
            return false;
        };
        let (Some(button), Some(clicked)) = (item.button(marker), event.window(marker)) else {
            return false;
        };
        button.window().is_some_and(|window| {
            std::ptr::eq(Retained::as_ptr(&window), Retained::as_ptr(&clicked))
        })
    }

    /// Take the icon out of the menu bar: safe from any thread, idempotent, bounded by
    /// [`REMOVAL_WAIT`].
    pub fn remove() {
        if MainThreadMarker::new().is_some() {
            take_icon();
            return;
        }
        // The exit may run on any thread, and the item can only go on the main one.
        let (sent, removed) = std::sync::mpsc::channel();
        DispatchQueue::main().exec_async(move || {
            take_icon();
            let _ = sent.send(());
        });
        let _ = removed.recv_timeout(REMOVAL_WAIT);
    }

    fn take_icon() {
        let Some(Installed { tray, monitor }) = ICON.with(|icon| icon.borrow_mut().take()) else {
            return;
        };
        if let Some(monitor) = &monitor {
            // SAFETY: a monitor this thread installed, removed once.
            unsafe { NSEvent::removeMonitor(monitor) };
        }
        // Dropping the icon takes the status item out of the menu bar.
        drop(tray);
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use super::{AVAILABLE, REMOVAL_WAIT, builder, report};
    use crate::util::UnwrapPoison;
    use crate::window_requests;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::Receiver;
    use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconEvent};
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, GetMessageW, PostThreadMessageW, TranslateMessage, WM_QUIT,
    };

    /// The notification area's icon: the thread that owns it and how to learn it is gone.
    struct Owned {
        /// The owning thread's id, set once it has built the icon; gone once an exit has claimed
        /// the slot.
        thread: Option<u32>,
        /// Answers once the icon has been dropped: a disconnected channel is the icon's removal.
        removed: Receiver<()>,
    }

    /// `install` fills the slot before the thread starts, so `remove` always finds one to claim and
    /// wait for.
    static OWNED: Mutex<Option<Owned>> = Mutex::new(None);

    /// The notification area hands a click to the thread that registered the icon and needs a
    /// message loop on
    /// it, so the tray gets a thread of its own. Nothing of ours runs on that loop beyond the
    /// dispatch, and an open menu never blocks the iced event loop.
    pub(super) fn install() {
        TrayIconEvent::set_event_handler(Some(|event: TrayIconEvent| {
            // The release, not the press: the gesture one click reports once.
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                window_requests::show();
            }
        }));

        let (held, waiting) = std::sync::mpsc::channel();
        *OWNED.lock().unwrap_poison() = Some(Owned {
            thread: None,
            removed: waiting,
        });
        let spawned = std::thread::Builder::new()
            .name("tray".to_owned())
            .spawn(move || {
                run();
                // Losing the last sender is what an exit waiting on the channel sees.
                drop(held);
            });
        if let Err(e) = spawned {
            // Nothing was started, so the slot would only make an exit wait for a signal that never
            // comes.
            *OWNED.lock().unwrap_poison() = None;
            report(&format!("its thread could not be started: {e}"));
        }
    }

    fn create() -> Result<TrayIcon, String> {
        let icon = crate::app_icon::tray_icon_image().ok_or("its image could not be built")?;
        builder()?
            .with_icon(icon)
            .build()
            .map_err(|e| e.to_string())
    }

    fn run() {
        let tray = match create() {
            Ok(tray) => tray,
            Err(reason) => {
                report(&reason);
                return;
            }
        };

        // Registering under the lock `remove` claims makes the two orders exhaustive: either the
        // exit claimed the
        // slot first and this icon is dropped below (this thread will not pump it), or this thread
        // is registered
        // first and the exit's `WM_QUIT` ends the loop. An exit waits for the channel closing
        // either way.
        let registered = {
            let mut slot = OWNED.lock().unwrap_poison();
            match slot.as_mut() {
                Some(owned) => {
                    owned.thread = Some(unsafe { GetCurrentThreadId() });
                    true
                }
                None => false,
            }
        };
        if registered {
            // The icon is in the notification area and will be pumped from here: the way back to a
            // hidden window.
            AVAILABLE.store(true, Ordering::Relaxed);
            pump();
        }
        drop(tray);
    }

    /// The platform's messages for the icon — its clicks, and the restart of the taskbar,
    /// which re-registers it — are dispatched here, on the thread that registered it. A
    /// null window pumps this thread's own queue, and nothing of ours schedules work on
    /// it: it sits blocked until a message arrives.
    fn pump() {
        let mut message = std::mem::MaybeUninit::zeroed();
        while unsafe { GetMessageW(message.as_mut_ptr(), 0, 0, 0) } > 0 {
            let message = unsafe { message.assume_init_ref() };
            unsafe {
                TranslateMessage(message);
                DispatchMessageW(message);
            }
        }
    }

    /// Take the icon out of the notification area: safe from any thread, idempotent, bounded by
    /// [`REMOVAL_WAIT`].
    pub fn remove() {
        let Some(owned) = OWNED.lock().unwrap_poison().take() else {
            return;
        };
        if let Some(thread) = owned.thread {
            // The loop returns on `WM_QUIT`, which is what takes the icon away.
            unsafe { PostThreadMessageW(thread, WM_QUIT, 0, 0) };
        }
        // Returns once the thread dropped the icon, or on the timeout — a stale icon, never a hang.
        let _ = owned.removed.recv_timeout(REMOVAL_WAIT);
    }
}
