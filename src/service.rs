//! Whether the product runs with a window.
//!
//! The service is the product; the window is a layer on top of it that exists only where a
//! graphical system does. This module owns the one decision that follows from that: whether to
//! create a window at all ([`window_mode`]). It is a pure read — no display is touched and no probe
//! is run — so the decision can be exercised without a graphical system, and the boot path can name
//! it in one plain sentence.

/// Where the process runs, as far as window creation is concerned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Platform {
    MacOs,
    Windows,
    Linux,
    /// Any other target: no window layer is built for it.
    Other,
}

/// The graphical-system facts a window decision reads. Kept apart from the environment so the
/// decision is a pure function.
#[derive(Clone, Copy, Default)]
struct DisplayEnv<'a> {
    pub display: Option<&'a str>,
    pub wayland_display: Option<&'a str>,
    /// `MAHBOT_HEADLESS`: an operator's own decision to run a host with an unusable graphical
    /// system as a service. Not a mode switch — the service is the same either way.
    pub headless: bool,
}

/// Whether a window is created, and why not when it is not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowMode {
    /// A graphical system exists; the window is created and drives the same service.
    Window,
    /// No window: the service runs on its own. The sentence is what the boot logs once.
    Headless(&'static str),
}

impl WindowMode {
    #[must_use]
    pub const fn is_headless(self) -> bool {
        matches!(self, Self::Headless(_))
    }
}

const NO_DISPLAY: &str =
    "no graphical system was found (neither DISPLAY nor WAYLAND_DISPLAY is set)";
const FORCED_HEADLESS: &str = "MAHBOT_HEADLESS is set";
const NO_WINDOW_LAYER: &str = "this platform has no window layer";

/// Decide from facts alone. A graphical system is a platform with a window layer and — on Linux,
/// where the display is named by the environment — a display to put the window on. A host that
/// names no display is a server, not a broken desktop: nothing is guessed from a failed
/// connection, so an SSH install on a desktop is not mistaken for one.
#[must_use]
fn window_mode(platform: Platform, env: DisplayEnv<'_>) -> WindowMode {
    if env.headless {
        return WindowMode::Headless(FORCED_HEADLESS);
    }
    match platform {
        Platform::MacOs | Platform::Windows => WindowMode::Window,
        Platform::Linux => {
            if present(env.display) || present(env.wayland_display) {
                WindowMode::Window
            } else {
                WindowMode::Headless(NO_DISPLAY)
            }
        }
        Platform::Other => WindowMode::Headless(NO_WINDOW_LAYER),
    }
}

/// The decision this process runs under, read once at the daemon's start — before anything asks
/// for a display, and handed to whatever needs it rather than read a second time.
#[must_use]
pub fn detect_window_mode() -> WindowMode {
    let display = std::env::var("DISPLAY").ok();
    let wayland_display = std::env::var("WAYLAND_DISPLAY").ok();
    window_mode(
        platform(),
        DisplayEnv {
            display: display.as_deref(),
            wayland_display: wayland_display.as_deref(),
            headless: headless_requested(),
        },
    )
}

/// Whether the operator asked this host to run without a window.
#[must_use]
fn headless_requested() -> bool {
    std::env::var("MAHBOT_HEADLESS").is_ok_and(|value| value != "0" && !value.is_empty())
}

const fn platform() -> Platform {
    if cfg!(target_os = "macos") {
        Platform::MacOs
    } else if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "linux") {
        Platform::Linux
    } else {
        Platform::Other
    }
}

fn present(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_platform_with_a_window_layer_and_a_display_gets_a_window() {
        assert_eq!(
            window_mode(
                Platform::Linux,
                DisplayEnv {
                    display: Some(":0"),
                    ..DisplayEnv::default()
                }
            ),
            WindowMode::Window
        );
        assert_eq!(
            window_mode(
                Platform::Linux,
                DisplayEnv {
                    wayland_display: Some("wayland-0"),
                    ..DisplayEnv::default()
                }
            ),
            WindowMode::Window
        );
        assert_eq!(
            window_mode(Platform::MacOs, DisplayEnv::default()),
            WindowMode::Window
        );
        assert_eq!(
            window_mode(Platform::Windows, DisplayEnv::default()),
            WindowMode::Window
        );
    }

    #[test]
    fn a_linux_host_with_no_display_runs_without_a_window() {
        assert_eq!(
            window_mode(Platform::Linux, DisplayEnv::default()),
            WindowMode::Headless(NO_DISPLAY)
        );
        // An empty or blank variable names no display either.
        assert_eq!(
            window_mode(
                Platform::Linux,
                DisplayEnv {
                    display: Some("  "),
                    wayland_display: Some(""),
                    ..DisplayEnv::default()
                }
            ),
            WindowMode::Headless(NO_DISPLAY)
        );
    }

    #[test]
    fn the_operators_own_decision_wins_over_the_environment() {
        assert_eq!(
            window_mode(
                Platform::MacOs,
                DisplayEnv {
                    headless: true,
                    ..DisplayEnv::default()
                }
            ),
            WindowMode::Headless(FORCED_HEADLESS)
        );
        assert_eq!(
            window_mode(
                Platform::Linux,
                DisplayEnv {
                    display: Some(":0"),
                    headless: true,
                    ..DisplayEnv::default()
                }
            ),
            WindowMode::Headless(FORCED_HEADLESS)
        );
        assert_eq!(
            window_mode(Platform::Other, DisplayEnv::default()),
            WindowMode::Headless(NO_WINDOW_LAYER)
        );
    }
}
