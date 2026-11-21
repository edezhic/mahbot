Transition a ticket to a new phase, named by `ticket_id` (a bare number or `<workspace>-<number>`). Valid manual phases: `backlog` (back to the queue), `planning` (paused), `queued` (send to the engineer), `cancelled` (abandon; work already done leaves the workspace dirty), `failed` (unsuccessful; blocks its dependents), `done` (complete, irreversible).

Never set the pipeline-managed phases (`analysis`, `in_development`, `verification`, `in_sanitation`) — the board poller does. A `done` ticket's phase is frozen: it cannot change or be superseded; file a new ticket to revisit the work.

Returns a one-line confirmation.
