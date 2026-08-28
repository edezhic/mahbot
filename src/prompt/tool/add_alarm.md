Create a reminder that will wake you up when it becomes due: it fires a message back into your own conversation so you can follow up. This is useful when the user asks for a notification at a specific time or on a recurring interval, or when you need to be reminded of something later.

Exactly one of `fire_at` or `interval_seconds` must be provided:

- `fire_at` — a one-shot reminder. Must be an RFC3339/ISO-8601 UTC timestamp in the future (e.g. `2026-08-28T10:30:00Z`); a past timestamp is an error. If the user gives a local time ("at 5pm", "tomorrow at noon"), convert it to UTC before passing it here.
- `interval_seconds` — a periodic reminder. The interval is in seconds and must be at least 10.

A maximum of 10 active alarms may exist for you; adding beyond that fails.
