Create an automation that wakes you when due by delivering a message into this conversation. Give it `text` and exactly one of `fire_at` (one-shot RFC3339/ISO-8601 UTC, in the future — convert the user's local time) or `interval_seconds` (periodic, at least 5).

Optional `trigger` runs one custom tool at fire time and wakes you only when the check reports something; a check that breaks wakes you with the reason and removes the automation. Optional `role` addresses it to an existing custom role instead of you — the administrator alone may do that, and a role-addressed automation takes no `trigger`.

Answers with the schedule and its next fire time.
