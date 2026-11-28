Your view into the owner's managed workspaces — the software projects — and their settings. It manages no board: ticket work stays with each project's manager.

- `action: "reference"` — every action, every field, and what is inert where.
- `list` / `show` — the managed workspaces, read live: name, status, paused, maintenance, folder; `show` adds rules, diagnostics commands and the active project. A workspace that is waiting — its profile unconfirmed, or another writer in its folder — says why, and its status sits beside it: a workspace whose first discovery has not run, or has failed, has nothing to claim. A workspace being re-analysed is working, not waiting: reading holds nothing. Use these for current state — `<managed-workspaces>` is a session-start snapshot.
- `add` — add a software project directory as a managed workspace and make it active. Software projects only.
- `profile`, `confirm_profile`, `set_profile`, `reset_profile` — the workspace's profile: who staffs each stage, verification's pass threshold and whether every project check must pass. A new workspace's pipeline waits until its profile is confirmed. The same four actions are on the `config` tool.
- `delete`, `set_rules`, `set_diagnostics`, `rediscover`, `rediscover_diagnostics`, `maintenance`, `use` — its own settings; `action: "reference"` has each.
- `pause` / `resume` — freeze or lift a pipeline; every writer stops at its next round boundary, while reading (a discovery pass, a separately launched analysis) runs on. A resume answers the messages the freeze kept and re-drives the rounds it stopped. **Without `workspace` it acts on every managed workspace at once** — a fleet-wide write, so use that form only when the owner asked for all of them, and then pass nothing else.
- `board` — the tickets of the addressed workspaces, one line each; without `workspace`, all managed workspaces together.
- `ticket` — one ticket by its full id, with its fields and comment thread; shortened by default, `full: true` for it whole.
- `comment` — leave a note on a ticket as `assistant`; it never advances or blocks a ticket and cannot be edited afterwards, so leave one only when the owner asks.

`list`, `show`, `board` and `ticket` change nothing and work for any managed workspace. Carry these out yourself — never forward them to a Manager, and never claim an outcome the call did not report. Call `action: "reference"` for every action and field in full.
