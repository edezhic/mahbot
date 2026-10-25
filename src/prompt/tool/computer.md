Observe and act on the local GUI: read accessibility trees, click, type, press, scroll, drag, and capture a screenshot or a zoomed region.

## Which action

`apps` then `windows` picks a target; in a `windows` listing `[focused]` marks the focused window of the application in front — or, in a listing of one application, that application's own — and `#N` its place among the windows listed, where the platform can attribute one. `observe` reads a surface's tree and returns element refs (`e1`, `e2`, …). Act by `ref` where the tree exposes the control, by `x`/`y` where it does not; refs expire at the next full `observe` or with the end of the turn, whichever comes first. Use `screenshot` or `zoom` where the tree cannot show what you need. A tree shows at most 120 characters of a value and 200 of a name or title, cut with an ellipsis and counted in the header.

An observation can be narrowed — `ref` to the subtree beneath an element, `depth` to that many levels, `kind` to one kind (matched against the kinds it prints) — and a narrow look reads what the tool already holds rather than the surface: it says it covers part of the surface, and a surface never fully observed has nothing to narrow. `changed` answers what differs from the remembered look of that surface — what appeared, disappeared, moved and changed, and how old the comparison is. Neither replaces that look or the refs built from it, and a change answer prints no refs.

## Which surface

With no `target`, a call acts on the surface you last observed or captured, before that on the focused window of the frontmost application. A full `observe`, and a `screenshot` or `zoom`, that names a target makes that surface current (a `"screen"` capture leaves the screen current); a failed one leaves it as it was, and a narrow look or a change answer never moves it.

A `w` id names exactly one window: it follows that window and fails as `surface-gone` once it is gone, never falling back to another window of the same application. An `a` id is the application's focused window at the moment of the call; `"screen"` is the whole desktop.
An `a` or `w` id keeps naming the same application or window for the whole session, across turns; only resetting the conversation or restarting the service ends it. A listing keeps the id of a surface it already named, and a new surface gets a new id. An id the tool no longer knows says so — list again.

Points and regions are normalized 0–1000 of the surface the call names; a region is `[x0, y0, x1, y1]`. A ref belongs to the surface it was observed on and is acted on only if it still matches what that observation showed — its kind, its name and its place on the surface. An observation's header names its surface.

`press` goes to the focused application and a `scroll` without x/y follows the system's wheel routing; neither pins a window. A coordinate `click` is checked against the window under its point at the moment it is sent, where the platform can attribute that window.

## Results

An observation reports its own result. An actuating call answers what it delivered and then what it could observe afterwards: the focused element, and — for an action that addressed a surface — that surface and whether it sits where the remembered observation found it. It never claims that the intended effect happened — check the state it reports. `applied but unconfirmed` means part of the input may have landed and the platform cannot say how much: check the target's state before acting again, so a repeated click is not a second click. A failure means the call did not happen; its message names the reason and the next step.
