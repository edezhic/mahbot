Observe and act on the local GUI: read accessibility trees, click, type, press, scroll, drag, and capture a screenshot or a zoomed region.

## Which action

`apps` then `windows` picks a target; `observe` reads its tree and returns element refs (`e1`, `e2`, …). Act by `ref` where the tree exposes the control, by `x`/`y` where it does not; refs expire on the next observe. Use `screenshot` or `zoom` where the tree cannot show what you need. A tree shows at most 120 characters of a value and 200 of a name or title, cut with an ellipsis and counted in the header.

## Which surface

With no `target`, a call acts on the surface you last observed or captured, before that on the focused window of the frontmost application. A successful `observe`, `screenshot` or `zoom` that names a target makes that surface current (a `"screen"` capture leaves the screen current); a failed one leaves it as it was.

A `w` id names exactly one window: it follows that window and fails as `surface-gone` once it is gone, never falling back to another window of the same application. An `a` id is the application's focused window at the moment of the call; `"screen"` is the whole desktop.
An `a` or `w` id keeps naming the same application or window for the whole session, across turns; only resetting the conversation or restarting the service ends it. A listing keeps the id of a surface it already named, and a new surface gets a new id. An id the tool no longer knows says so — list again.

Points and regions are normalized 0–1000 of the surface the call names; a region is `[x0, y0, x1, y1]`. A ref belongs to the surface it was observed on. An observation's header names its surface.

`press` goes to the focused application, a coordinate `click` lands on whatever is under the point, and a `scroll` without x/y follows the system's wheel routing; none of them pins a window.

## Results

An observation reports its own result. An actuating call (click, type, press, scroll, drag) answers `done` when the system took the input, without checking what it did, or `applied but unconfirmed` when part of it may have landed and the platform cannot say how much: check the target's state before acting again: a repeated click is a second click. A failure means the call did not happen; its message names the reason and the next step.
