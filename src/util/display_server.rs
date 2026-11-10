//! Which display server a Linux session talks to. Decided by probing, not by an
//! environment variable: an X connection advertising `XWAYLAND` is a Wayland
//! session, one without it is a real X server, and `WAYLAND_DISPLAY` decides
//! only a session with no X connection at all.

use std::sync::OnceLock;

use crate::util::with_block_in_place;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisplayServer {
    X11,
    Wayland,
}

static DISPLAY_SERVER: OnceLock<DisplayServer> = OnceLock::new();

/// Cached for the life of the process — a session does not change server under
/// it — but a probe that could not reach the server is retried by the next caller.
pub(crate) fn display_server() -> Option<DisplayServer> {
    if let Some(known) = DISPLAY_SERVER.get() {
        return Some(*known);
    }
    let probed = with_block_in_place(probe)?;
    let _ = DISPLAY_SERVER.set(probed);
    Some(probed)
}

/// Whether this is a Wayland session — the branch every still, coordinate map and input-idle
/// reading takes.
pub(crate) fn wayland() -> bool {
    display_server() == Some(DisplayServer::Wayland)
}

fn probe() -> Option<DisplayServer> {
    if std::env::var_os("DISPLAY").is_some()
        && let Ok((conn, _screen)) = x11rb::connect(None)
    {
        let xwayland = x11rb::protocol::xproto::query_extension(&conn, b"XWAYLAND")
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
