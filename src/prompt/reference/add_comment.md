# add_comment — reference

## Arguments

- `ticket_id` — a bare number (`123`, resolved against your workspace) or the fully prefixed `workspace-123`; only a ticket from the current workspace can be commented on.
- `content` — the comment itself.

## Visibility and attribution

Every comment is attributed to the Manager role, whichever agent calls the tool. It is visible to the user and to every agent who later works on the ticket, so it is the primary way to carry context and decisions across the ticket's lifecycle.

## Delivery to a running agent

A running agent receives a comment as a soft deferred message at the start of its next tool round; it never cancels or aborts the agent. A later dispatch — including a resumed engineer or sanitation round — surfaces comments the agent has not yet seen.

## Result

The answer confirms the comment was added to the named ticket.
