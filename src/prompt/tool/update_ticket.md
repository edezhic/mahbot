Transition an existing ticket to a new status or phase. Requires the ticket `ticket_id`, which you can obtain from the output of create_ticket or list_tickets. The valid phases for manual transitions are:

- **backlog** — return the ticket to the start (analysts will re-evaluate it)
- **planning** — paused state awaiting further decision whether to proceed with the ticket or cancel it
- **ready_for_development** — send the ticket to the engineer's queue for implementation
- **cancelled** — abandon the ticket without completing it (beware that if the ticket already had some work done in it then the workspace will be left in a dirty state even after cancellation).
- **done** — mark the ticket as complete and successful

In general you should not set any other statuses unless explicitly requested by the user.