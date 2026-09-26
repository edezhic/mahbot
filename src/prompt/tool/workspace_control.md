Control the pipelines of the registered workspaces. Pick one `action`:

- `list` — report every registered workspace: its status (`pending`, `analyzing`, `ready`, `failed` — the product's own status word) and `paused` when its pipeline is frozen. On a workspace that is not `ready`, `paused` is the analysis's own pause rather than a freeze you can lift. This is a live read: use it for any question about the current state of the projects, because the `<registered-workspaces>` block in your context is a session-start snapshot and does not carry the pause state at all.
- `pause` — freeze the pipeline of the workspace named in `name`: no stage advances, and the agents working on its tickets stop at their next round boundary.
- `resume` — lift that freeze; work in flight goes on from where it stopped. Neither direction loses anything.

`name` is the workspace's registered name, matched exactly (no case-insensitive and no approximate matching): pass it spelled as the `<registered-workspaces>` block or `list` spells it. A name with no registered workspace behind it, or a personal space, is refused.

Only a `ready` workspace can be paused or resumed. A `pending`, `analyzing` or `failed` one has no pipeline running, and the request is refused with its status instead of being reported as applied. A pause or resume of a workspace already in that state is reported as `already paused` / `not paused` — never as a fresh change.

These are administrative actions you carry out yourself — never forward them to a Manager, and never claim an outcome the call did not report. No confirmation is needed: a pause is reversible. Close with the outcome in the reply that ends your turn, naming the workspace and its resulting state.