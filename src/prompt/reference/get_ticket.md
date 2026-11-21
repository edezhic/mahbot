# get_ticket — reference

## Arguments

- `ticket_id` — a bare number (`123`) resolves against your workspace; a fully prefixed `workspace-123` is taken as written.
- `full` — return the complete ticket. Default false.

## What comes back

The header metadata is always complete: id, title, phase, priority, reporter, workspace, created and updated dates, supersedes / superseded-by / prerequisites links, and the archived flag.

The default view applies only when the ticket's reporter is you. It truncates the description to 500 characters (appending `…` when longer) and each comment to 200 characters, except the three newest comments, which are shown whole. `full: true` returns the full description and the complete comment thread.

A ticket created by another agent is always returned in full, whatever `full` says.

## Comments

A ticket's thread carries analyst findings, verification results, and messages from the user, the manager, or the assistant writing on the owner's behalf. Read it before acting on a ticket.
