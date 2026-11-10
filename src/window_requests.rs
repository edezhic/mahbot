//! The ways into the dashboard's window from outside it, and the product's only in-app quit: the
//! tray icon and its menu. A window a close hid is reachable only through the tray, so each
//! gesture is handed to this channel and the dashboard decides what it means (see
//! `gui::Dashboard`). Nothing else raises the window: not a launch of the executable (a keepalive
//! automation's start branch, an agent's restart, a script that "opens the app" — see `main`), not
//! a reopen of the application (a Dock click and `open -a` arrive as one event, so `shutdown`
//! raises nothing for either). Nothing here waits on a platform: the handlers only enqueue, and
//! the subscription draining the channel runs on the interface's own runtime. Windows and macOS
//! only — the platforms whose window can be hidden.
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

fn send(request: WindowRequest) {
    if let Some(sender) = SENDER.get() {
        let _ = sender.send(request);
    }
}

/// Bring the window up: the tray icon's click, the only producer of this request.
pub(crate) fn show() {
    send(WindowRequest::Show);
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
