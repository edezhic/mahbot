Create a reminder that will wake you up when it becomes due: it fires a message back into your own conversation so you can follow up. This is useful when the user asks for a notification at a specific time or on a recurring interval, or when you need to be reminded of something later.

Exactly one of `fire_at` or `interval_seconds` must be provided:

- `fire_at` — a one-shot reminder. Must be an RFC3339/ISO-8601 UTC timestamp in the future (e.g. `2026-08-28T10:30:00Z`); a past timestamp is an error. If the user gives a local time ("at 5pm", "tomorrow at noon"), convert it to UTC before passing it here.
- `interval_seconds` — a periodic reminder. The interval is in seconds and must be at least 5.

Optional `trigger` — schedule a check instead of only delivering the reminder: `{"tool": "<name>", "args": {…}}` names one of the custom tools available to you, with the same name and arguments a normal `custom` call takes. The tool must be one you can call and the arguments must be ones it declares; a trigger names one tool, never a chain. The trigger is stored with the alarm (the whole stored trigger — name and arguments together — within 2000 characters) and is shown by `list_alarms` and in the `<user-alarms>` context block.

A triggered alarm wakes you only when the check reported something: any output wakes you, and a check that printed nothing at all stays silent. Any outcome that is not a clean run wakes you with the reason and removes the alarm — a non-zero exit, a timeout, a failure to start, a tool that is no longer available to you or no longer usable, arguments the tool no longer accepts, or a missing runtime — so a broken check never keeps firing.

A maximum of 10 active alarms may exist for you; adding beyond that fails.
