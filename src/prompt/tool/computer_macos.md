## macOS

* macOS 14 or newer (the product's floor); stills use ScreenCaptureKit, observe/act use the AX API. On an older macOS a still is `unsupported`.
* The three grants belong to the TERMINAL the daemon was started from, not to the product — a bare binary has no privacy identity of its own. The owner enables that terminal once under System Settings → Privacy & Security: Accessibility (tree reads and element actions), input sending (clicks, typing, key chords such as `cmd+shift+t` — macOS files it under the same Accessibility pane) and Screen Recording (stills). The grants survive rebuilds and self-updates because the terminal's signature never changes, and every process started from the same terminal — your own shell commands included — shares them.
* A missing grant is requested from the system when an action needs it; the dialog's subject is the terminal. Screen Recording applies only after mahbot is restarted, and macOS asks for it again periodically by design. There is no way around this and none is attempted. A missing grant surfaces as `permission-denied`.
* Never try to grant a permission yourself — a settings window is not yours to click.
