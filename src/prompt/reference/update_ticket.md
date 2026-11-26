# update_ticket — reference

## Arguments

- `ticket_id` — the ticket to move: a bare number (`123`, resolved against your workspace) or the fully prefixed `workspace-123`.
- `phase` — the phase to move it to.

## Manual phases

- `backlog` — return the ticket to the start; analysts re-evaluate it.
- `planning` — paused, awaiting a decision whether to proceed or cancel.
- `queued` — send it to the engineer's queue for implementation.
- `cancelled` — abandon it. Work already done in the workspace stays, so the workspace can be left dirty.
- `failed` — mark the attempt unsuccessful. A failed ticket parks the workspace: while a ticket here sits in `failed`, nothing here is picked up for development. It also blocks its dependents and stays visible for triage. Do not move a failed ticket back to `backlog`, `planning` or `queued` — a failed ticket is replaced by a supersede, or cancelled.
- `done` — mark it complete and successful; irreversible.

## Pipeline-managed phases

Never set `analysis`, `in_development`, `verification` or `in_sanitation` yourself: the board poller moves tickets through them, and a manual transition interferes with running agents. Set a phase only when asked to.

## Finished tickets

A `done` ticket's phase is frozen: every move is refused and it cannot be superseded. Only the phase is frozen — the ticket can still be read and commented on. To revisit finished work, create a new standalone ticket.

## Result

The answer confirms the ticket and its new phase. A move to `queued` adds a sentence when the workspace holds a failed ticket other than the one being moved: it names them and says the queued one will not be picked up until every one of them has left `failed` — by being superseded or cancelled.
