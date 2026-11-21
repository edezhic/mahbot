# create_ticket — reference

A new ticket starts in `backlog`, where analysts validate its scope before it is planned.

## Arguments

- `title` — a short summary, the line the board shows.
- `description` — the full statement of the work: what to change and why.
- `prerequisites` — optional ticket IDs that must finish (`done`, archived or `cancelled`) before this ticket can be claimed. A `failed` prerequisite keeps blocking.
- `supersede` — optional ticket ID this ticket replaces. The old ticket is cancelled, this one is created as its replacement, and every dependent's prerequisite is rewired to the new ID. A `done` ticket cannot be superseded: its phase is immutable, so file the follow-up as a standalone ticket.
- `label` — the Manager's own argument, required: `feat` adds a capability, `fix` repairs a problem, `maint` is everything else. It is fixed at creation and cannot change; the pipeline prefixes every commit it makes for the ticket with it. A replacement states its own label — nothing is inherited.
- `priority` — the Manager's own argument, optional: 0 is the highest urgency, 1 the default, higher numbers lower. Omitted while superseding, it is inherited from the superseded ticket; otherwise the role default applies.

## Ticket IDs

A bare number (`123`) resolves against the current workspace. The fully prefixed form (`workspace-123`) is accepted only when its prefix is the current workspace; an ID from another workspace is refused. Every write tool here is confined that way; `get_ticket` is the exception, accepting a prefixed ID from any workspace.

## Result

The answer names the new ticket's ID, and any prerequisites, as `Created ticket <id>: <title>`; a supersede answers `Superseded <old> → created ticket <id>: <title>`.
