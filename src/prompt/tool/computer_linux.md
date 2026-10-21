## Linux

* Rust-only, no setup: clean Ubuntu 24.04 works, no groups/rules/apt packages.
* X11 = full: screen/window/multi stills + zoom/observe via X11 protocol, AT-SPI2 observe/act, XTEST raw input.
* Wayland/XWayland (both DISPLAY+WAYLAND_DISPLAY set counts as Wayland) = accessibility + one best-effort Screenshot-portal attempt; a system consent dialog may appear — never click it yourself; deny/timeout/missing → `degraded`, continue accessibility-only.
* Raw input + cursor `degraded` on Wayland — prefer `ref` actions.
* Window stills show visible pixels (occluders visible, minimized fails — `observe` for the exact tree).
* No streaming in v1.
* Offered only when a GUI session with a reachable session D-Bus exists; an accessibility stack that starts later is picked up by newly constructed sessions.
* `permission-denied` means a D-Bus call was not authorized.
* mahbot's own window publishes no AT-SPI surface, so it is not in `apps`/`windows`, cannot be observed or acted on by `ref`, and has no window still. `target: "screen"` input, key chords and `type` still reach it.
