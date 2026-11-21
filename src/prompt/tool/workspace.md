Your view into the owner's managed workspaces — the software projects — and their settings. It manages no board: ticket work stays with each project's manager.

- `action: "reference"` — every action, every field, and what is inert where.
- `list` / `show` — the managed workspaces, read live: name, status, paused, maintenance, folder; `show` adds rules, diagnostics commands and the active project. Use these for current state — `<managed-workspaces>` is a session-start snapshot.
- `add` — add a software project directory as a managed workspace and make it active. Software projects only.
- `delete`, `set_rules`, `set_diagnostics`, `rediscover`, `rediscover_diagnostics`, `maintenance`, `use` — its own settings; `action: "reference"` has each.
- `pause` / `resume` — freeze or lift a pipeline; agents stop at their next round boundary. **Without `workspace` it acts on every managed workspace at once** — a fleet-wide write, so use that form only when the owner asked for all of them, and then pass nothing else.
- `board` — the tickets of the addressed workspaces, one line each; without `workspace`, all managed workspaces together.
- `ticket` — one ticket by its full id, with its fields and comment thread; shortened by default, `full: true` for it whole.
- `comment` — leave a note on a ticket as `assistant`; it never advances or blocks a ticket and cannot be edited afterwards, so leave one only when the owner asks.

`list`, `show`, `board` and `ticket` change nothing and work for any managed workspace. Carry these out yourself — never forward them to a Manager, and never claim an outcome the call did not report. Call `action: "reference"` for every action and field in full.
