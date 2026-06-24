Create a new ticket on the work board.

New tickets start in `backlog`, where they are analyzed before development.

Parameters:
- `title`: short ticket title
- `description`: full ticket description
- `prerequisites`: optional ticket IDs that must finish before this ticket can be claimed
- `supersede`: optional ticket ID to replace

When `supersede` is provided, the old ticket is cancelled, this ticket is created as its replacement, and dependent prerequisites are rewired to the new ticket.

Constraints:
- prerequisites must exist in the same workspace
- the superseded ticket must be in the same workspace
- a ticket cannot supersede and depend on the same ticket