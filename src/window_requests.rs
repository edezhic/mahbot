//! The ways into the dashboard's window from outside it, and the product's own in-app quit:
//! the tray icon and its menu, a second launch of the product, and on macOS a click on the
//! Dock icon.
//!
//! A window that a close hid is reachable only through these surfaces, so each of them hands
//! its gesture to this channel and the dashboard decides what it means (see
//! `gui::Dashboard`). Nothing here waits on a platform: the handlers only enqueue, and the
//! subscription that drains the channel runs on the interface's own runtime.
//!
//! Windows and macOS only: the platforms whose window can be hidden are the ones with a way
//! back to it.
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::util::UnwrapPoison;

/// A gesture for the dashboard.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WindowRequest {
    /// Bring the window up: out of the hidden state, restored from minimized, in front.
    Show,
    /// An ordinary exit request: the tray menu's one item, the product's only in-app quit.
    Quit,
}

/// The sending half, installed for the process lifetime by [`init`].
static SENDER: OnceLock<UnboundedSender<WindowRequest>> = OnceLock::new();

/// The gestures the dashboard's subscription reads. Filled by [`init`] before the iced
/// application runs, so the subscription always has a source.
static RECEIVER: Mutex<Option<UnboundedReceiver<WindowRequest>>> = Mutex::new(None);

/// Install the channel, once per process, before the iced application runs: a gesture that
/// arrives before anything is subscribed then waits instead of being lost.
pub fn init() {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let _ = SENDER.set(sender);
    *RECEIVER.lock().unwrap_poison() = Some(receiver);
}

fn send(request: WindowRequest) -> bool {
    SENDER
        .get()
        .is_some_and(|sender| sender.send(request).is_ok())
}

/// Ask the dashboard to bring the window up. `false` when nothing will act on it — the
/// channel was never installed — so the caller falls back on what it does without the request.
pub(crate) fn show() -> bool {
    send(WindowRequest::Show)
}

/// The tray menu's one item: an ordinary exit request, the same one the platform's own quit
/// makes.
pub(crate) fn quit() {
    send(WindowRequest::Quit);
}

/// Take the gestures for the dashboard's subscription. `None` before [`init`] ran, or once
/// it has been taken: iced spawns a `Subscription::run` recipe identified by its function
/// pointer once per process.
pub(crate) fn take() -> Option<UnboundedReceiver<WindowRequest>> {
    RECEIVER.lock().unwrap_poison().take()
}
