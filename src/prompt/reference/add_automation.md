# add_automation — reference

## Arguments

- `text` — the automation's own text, stored with it and delivered when it fires. On a role-addressed automation it is the role's launch message, so it must not be empty.
- `fire_at` — the one-shot fire time: RFC3339/ISO-8601, UTC, in the future. Convert the user's local time to UTC; a time already past is refused.
- `interval_seconds` — the periodic interval in seconds: at least 5, at most about 292 years. The first fire is one interval from now.
- `trigger` — an optional check, `{"tool": …, "args": …}`: `tool` names one custom tool available to you and `args` holds the arguments a normal `custom` call would take, keyed by the parameter names that tool declares. A trigger names one tool, never a chain; an argument the named tool does not declare is refused when the automation is armed rather than stored. The whole stored trigger — the name plus its arguments as JSON — is limited to 2000 characters.
- `role` — address the automation to a custom role instead of yourself, by the role's name. Only the administrator may do this, and the role must exist.
- `identifier` — that role's branch, created on first use; omit for the role's default behaviour. Only with `role`.
- `workspace` — the folder the role works in: a managed workspace name, an absolute path, or `personal:<user>`; omit for your personal folder. Only with `role`.

`identifier` and `workspace` without `role` are refused: they are companions of `role`, never a target of their own.

## The schedule

Pass exactly one of `fire_at` and `interval_seconds` — both, or neither, is refused. A one-shot fires once and is then gone; a periodic automation fires every interval until removed. An owner may hold 50 active automations at a time; arming one more is refused, so remove an obsolete one with `remove_automation` first.

A firing that comes due while the product is not running is delivered when it next runs, marked `MISSED` and naming the moment it was due for.

## The trigger

A triggered automation runs its check at fire time instead of only delivering `text`:

- Any output from the check wakes you with that output; a check that prints nothing delivers nothing and the automation stays armed.
- A check that does not finish in time delivers nothing, stays pending and is retried after about a minute. After three such attempts you are told once that it is not producing a result, and it keeps retrying; a report that finally arrives after such retries is marked late and names the time it was due for.
- Any other non-clean outcome wakes you with the reason and removes the automation: a non-zero exit, a memory stop, a failure to start, a tool no longer available or usable, arguments it no longer accepts, or a missing runtime.

The check runs as you, in your personal workspace, with the same custom tools and grants your own `custom` calls have.

## Role-addressed automations

`role` wakes a custom role's launch with `text` as its launch message, and the role's result comes back to you. Such an automation cannot carry a `trigger` — the text is the whole message — and only the administrator may arm one.

## Result

The answer names the schedule and its next fire time in local and UTC terms, repeats the stored text, and adds the trigger or the role target when there is one. It carries no id: read it from `list_automations`, which is also where the trigger and the target are shown.
