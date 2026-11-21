# computer — reference

`action` is one tagged object per call, e.g. `{"click": {"ref": "e1"}}`.

Every point and region is normalized 0–1000 of the surface the call names. A region is `[x0, y0, x1, y1]`. Each string argument is capped at 2,000 characters; `depth` is 1–25; `amount` is 1–100; `seconds` is greater than 0 and at most 10.

## Actions

- `observe` — read a surface's tree and mint element refs (`e1`, `e2`, …). `target` names the surface; `ref` narrows to the subtree beneath an element; `depth` narrows to that many levels; `kind` narrows to one kind, matched against the kinds it prints; `changed` answers what differs from the remembered look.
- `screenshot` — capture a surface as an image; `target` names it.
- `zoom` — capture a crop; `region` is required and `target` names the surface.
- `apps` — list applications with their ids.
- `windows` — list windows with their ids, optionally for one application by `app`.
- `click` — click a `ref`, or a point by `x` and `y`. `button` is `left`, `right` or `middle`; `double` asks for a double click; `modifiers` is any of `cmd`, `ctrl`, `alt`, `shift`. `button`, `double` and `modifiers` apply to a point only.
- `type` — set an element's value when `ref` names one, or type into the focused control when it does not; `text` is required.
- `press` — press a key chord in `keys`, e.g. `"ctrl+c"`.
- `scroll` — scroll `direction` (`up`, `down`, `left`, `right`), optionally at a point by `x` and `y` on `target`, by `amount` steps.
- `drag` — drag from `from` to `to`, two points, on `target`.
- `cursor` — report the pointer position.
- `wait` — pause `seconds` before the next action.

## Which surface

With no `target`, a call acts on the surface you last observed or captured, before that on the focused window of the frontmost application. A full `observe`, and a `screenshot` or `zoom`, that names a target makes that surface current (a `"screen"` capture leaves the screen current); a failed one leaves it as it was, and a narrow look or a change answer never moves it.

A `w` id names exactly one window: it follows that window and fails as `surface-gone` once it is gone, never falling back to another window of the same application. An `a` id is the application's focused window at the moment of the call; `"screen"` is the whole desktop. An `a` or `w` id keeps naming the same application or window for the whole session, across turns; only resetting the conversation or restarting the service ends it. A listing keeps the id of a surface it already named, and a new surface gets a new id. An id the tool no longer knows says so — list again.

An observation's header names its surface.

## Observations and refs

A full `observe` is the only thing that records a surface and mints refs; refs expire at the next full `observe` or with the end of the turn, whichever comes first. A ref belongs to the surface it was observed on and is acted on only if it still matches what that observation showed — its kind, its name and its place on the surface.

A narrow look reads what the tool already holds rather than the surface: it says it covers part of the surface, and a surface never fully observed has nothing to narrow. `changed` reports what appeared, disappeared, moved and changed, and how old the comparison is. Neither replaces that look or the refs built from it, and a change answer prints no refs.

A tree shows at most 120 characters of a value and 200 of a name or title, cut with an ellipsis and counted in the header.

In a `windows` listing, `[focused]` marks the focused window of the application in front — or, in a listing of one application, that application's own — and `#N` its place among the windows listed, where the platform can attribute one.

## Input routing

`press` goes to the focused application and a `scroll` without `x`/`y` follows the system's wheel routing; neither pins a window. A coordinate `click` is checked against the window under its point at the moment it is sent, where the platform can attribute that window.

## Results

Beyond the observation itself, an actuating call names the focused element and — for an action that addressed a surface — that surface and whether it sits where the remembered observation found it.

`applied but unconfirmed` means part of the input may have landed and the platform cannot say how much: check the target's state before acting again, so a repeated click is not a second click.
