## `for` — the steps after this call

`for` holds only the steps that come after this call: the call itself is step one of the chain, so the step you name in `for` is step two. A chain is at most ten steps, so `for` cannot go more than nine steps deep.

## Where a step works

`place` on a step is that step's own folder, and a step that names none works in the folder the nearest earlier step named — this call included, so naming a folder on the call, or on any step, moves the whole rest of the chain there. This is what `place`'s "omit to run in your own workspace" means inside a chain: only a step with no earlier step naming a folder keeps its own default — a worker your own workspace, a role the administrator's personal workspace.

A folder is handed on in the form the next step takes — a role gets the workspace it is, a worker the directory — so a step is never refused over how an earlier step spelled it. The folder rules themselves do not change, so a chain whose step would inherit a folder it cannot work in at all, a worker given a managed workspace above all, is refused before anything runs: the refusal names the step and the folder it would inherit, and the way out is to name on that step a folder it can work in, or not to chain it.
