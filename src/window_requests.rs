//! The ways into the dashboard's window from outside it, and the product's only in-app quit: the
//! tray icon and its menu, a second launch, and on macOS a Dock click. A window a close hid is
//! reachable only through these, so each gesture is handed to this channel and the dashboard
//! decides what it means (see `gui::Dashboard`). Nothing here waits on a platform: the handlers
//! only enqueue, and the subscription draining the channel runs on the interface's own runtime.
//! Windows and macOS only — the platforms whose window can be hidden.
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::util::UnwrapPoison;

#[derive(Debug, Clone, Copy)]
pub(crate) enum WindowRequest {
    /// Bring the window up: out of the hidden state, restored from minimized, in front.
    Show,
    Quit,
}

/// The sending half, installed for the process lifetime by [`init`].
static SENDER: OnceLock<UnboundedSender<WindowRequest>> = OnceLock::new();

/// The gestures the dashboard's subscription reads, filled by [`init`] before iced runs.
static RECEIVER: Mutex<Option<UnboundedReceiver<WindowRequest>>> = Mutex::new(None);

/// Install the channel, once per process, before iced runs: a gesture arriving before
/// anything is subscribed waits instead of being lost.
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

/// Ask the dashboard to bring the window up. `false` when nothing will act on it — the channel was
/// never installed — so the caller falls back.
pub(crate) fn show() -> bool {
    send(WindowRequest::Show)
}

/// The tray menu's one item: an ordinary exit request.
pub(crate) fn quit() {
    send(WindowRequest::Quit);
}

/// Take the gestures for the subscription. `None` before [`init`] ran, or once taken: iced
/// spawns a `Subscription::run` recipe identified by its function pointer once per process.
pub(crate) fn take() -> Option<UnboundedReceiver<WindowRequest>> {
    RECEIVER.lock().unwrap_poison().take()
}
