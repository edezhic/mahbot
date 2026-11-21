# update_ticket — reference

## Arguments

- `ticket_id` — the ticket to move: a bare number (`123`, resolved against your workspace) or the fully prefixed `workspace-123`.
- `phase` — the phase to move it to.

## Manual phases

- `backlog` — return the ticket to the start; analysts re-evaluate it.
- `planning` — paused, awaiting a decision whether to proceed or cancel.
- `queued` — send it to the engineer's queue for implementation.
- `cancelled` — abandon it. Work already done in the workspace stays, so the workspace can be left dirty.
- `failed` — mark the attempt unsuccessful. A failed ticket blocks its dependents and stays visible for triage.
- `done` — mark it complete and successful; irreversible.

## Pipeline-managed phases

Never set `analysis`, `in_development`, `verification` or `in_sanitation` yourself: the board poller moves tickets through them, and a manual transition interferes with running agents. Set a phase only when asked to.

## Finished tickets

A `done` ticket's phase is frozen: every move is refused and it cannot be superseded. Only the phase is frozen — the ticket can still be read, commented on and archived. To revisit finished work, create a new standalone ticket.

## Result

The answer is a one-line confirmation naming the ticket and its new phase.
