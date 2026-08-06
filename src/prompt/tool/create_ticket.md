Create a new ticket on the work board.

New tickets start in `backlog`, where they are analyzed before development.

Parameters:
- `title`: short ticket title
- `description`: full ticket description
- `prerequisites`: optional ticket IDs that must finish before this ticket can be claimed
- `supersede`: optional ticket ID to replace
- `priority` (Manager only): optional priority level (0 = highest urgency, 1 default, 2, 3, ... higher = lower). Maintainer must NOT include this parameter.

Ticket IDs may be given as a bare number (e.g. `123`) or the fully prefixed form (e.g. `mahbot-123`); both refer to tickets in the current workspace only. IDs from other workspaces are rejected.

When `supersede` is provided, the old ticket is cancelled, this ticket is created as its replacement, and dependent prerequisites are rewired to the new ticket.

Constraints:
- prerequisites must exist in the same workspace
- the superseded ticket must be in the same workspace
- a ticket cannot supersede and depend on the same ticket