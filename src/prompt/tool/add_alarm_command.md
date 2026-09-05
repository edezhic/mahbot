Optional `command` parameter: attach a shell command to the reminder. It runs at fire time in your personal workspace with a sanitized environment (no secrets), under the fixed shell timeout (10 minutes) with process-group kill on timeout.

Wake semantics — you are woken only when the command's raw output (stdout + stderr, after trimming whitespace) is non-empty, OR when the command fails (non-zero exit, timeout, or spawn failure). A broken poll script never stays silently broken, but a successful no-output poll stays silent. stderr counts as output. On any command failure (non-zero exit, timeout, or spawn failure) the alarm is auto-deleted after firing so a broken poll never keeps re-firing — the notification tells you it was deleted so you can recreate it after fixing the problem.

Works for both one-shot and periodic reminders; a periodic reminder re-runs the command on every firing, and a firing whose previous run is still in flight is skipped (no overlap).

The command is capped at 2000 characters, is stored with the alarm (visible via `list_alarms`), and its output is delivered scrubbed and truncated inside the `<alarm-notification>` message.

Use a command-armed alarm to schedule a meaningful re-check (a service health probe, a price check, an inbox poll) where you only want to wake when there is something to act on.
