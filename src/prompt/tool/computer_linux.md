## Linux

* X11 serves every still and all synthetic input. On Wayland input is refused and a still needs the desktop portal, which may ask the owner; `observe` and ref actions still work.
* A window still is a crop of the full-screen capture, so it costs no less than a screen still.
* A minimized window is refused for `observe`, a window still and coordinate input; restore it first.
* Neither listing can give a front-to-back order: AT-SPI carries no stacking, so it says the order is unknown. Which application is in front is read from the active state of the applications' windows, so a listing names it when one window reports that state and says it could not name one otherwise.
* A coordinate `click` is sent without checking what is under its point — this platform cannot attribute a window to an application — and the answer says the point could not be checked.
