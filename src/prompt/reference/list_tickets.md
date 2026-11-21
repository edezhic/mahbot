# list_tickets — reference

## Arguments

- `phase` — optional phase filter. Without it only active tickets are returned: `done` and `cancelled` are excluded. Name a phase explicitly to include them (`phase: "done"`, `phase: "cancelled"`).

## Phases

`backlog`, `analysis`, `planning`, `queued`, `in_development`, `verification`, `in_sanitation`, `done`, `cancelled`, `failed`.

`archived` is not a phase: a `phase: "archived"` request is refused, and the archive is searched with `search_archived_tickets` instead.

## Result

One line per ticket, the board's short display. An empty board answers `No tickets found.`
