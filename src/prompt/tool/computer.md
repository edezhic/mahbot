Observe and act on the local GUI through the OS accessibility channel: `observe`, `screenshot`, `zoom`, `apps`, `windows`, `click`, `type`, `press`, `scroll`, `drag`, `cursor`, `wait`. One action per call.

Use `apps` then `windows` to pick a target, `observe` to read it and get element refs, then act by `ref` where the tree exposes the control or by `x`/`y` where it does not. Use `screenshot` or `zoom` where the tree cannot show what you need.

The platform backend decides what is available: a refused still or synthetic input fails before anything is sent, and a minimized window is refused for `observe`, a still and coordinate input — restore it first.

A call returns an observation — the tree with refs, the image, the applications or windows — or, for an action, what it delivered and what it then observes.
