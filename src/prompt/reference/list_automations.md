# list_automations — reference

Takes no arguments. The listing is read in full — it is never truncated — and covers the active automations of the owner whose conversation asked.

## Each line

`- <id>, <kind>, <text>, next fire: <local time> (<utc> UTC)` — `kind` is `one-shot` or `periodic`. Two things append after the fire time:

- `, trigger: <tool> <args>` for an automation that carries a stored check.
- `, addressed to custom role "<role>"` for one that wakes a role, with `(branch "<identifier>")` and the folder when they were recorded.

Lines are ordered by next fire time, soonest first. The id is the one `remove_automation` takes.

## Empty

`(no active automations)` — an automation that already fired one-shot is no longer active and does not appear.
