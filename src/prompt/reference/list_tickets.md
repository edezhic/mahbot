# list_tickets — reference

## Arguments

- `phase` — optional filter; the phases are listed below.

## Phases

`backlog`, `analysis`, `planning`, `queued`, `in_development`, `verification`, `in_sanitation`, `done`, `cancelled`, `failed`.

`done` and `cancelled` are finished: they get archived, and `search_archived_tickets` is how finished work is reached — naming their phase here returns only one that has not been archived yet. A `failed` ticket is not finished — it stays live and does come back.

`archived` is not a phase: a `phase: "archived"` request is refused.

## Result

One line per ticket, the board's short display. An empty board answers `No tickets found.`
