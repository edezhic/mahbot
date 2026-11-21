# remove_automation — reference

## Arguments

- `id` — the automation to stop, exactly as `list_automations` prints it.

## What it does

Only an active automation of your own can be removed: one that already fired, or was already removed, is not found, and the answer says `No active automation with that id.` rather than failing. A removal is final — the automation never fires again, and its text, trigger and role target go with it.

## Result

`Removed automation <id> (was set for <local time> (<utc> UTC)).` — naming the moment it had been set for, so you can tell the user which one went.
