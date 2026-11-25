File work on the board as a new ticket; it starts in `backlog` for analysts to validate its scope. Give a `title`, a `description`, and optionally `prerequisites` (tickets that must finish first) or `supersede` (the ticket this one replaces — superseding a failed ticket is how its work is resumed, and the park lifts only once nothing here is left in `failed`). A `done` ticket cannot be superseded — file the follow-up as a new ticket.

IDs take a bare number or `<workspace>-<number>`; another workspace's is refused.

Returns the new ticket's ID.
