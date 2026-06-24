Transition an existing ticket to a new status or phase. Requires the ticket `ticket_id`, which you can obtain from the output of create_ticket or list_tickets. The valid phases for manual transitions are:

- **backlog** — return the ticket to the queue (analysts will re-evaluate it)
- **ready_for_development** — send the ticket to an engineer for implementation
- **paused** — temporarily hold the ticket (e.g., awaiting external input)
- **cancelled** — abandon the ticket without completing it
- **failed** — mark the ticket as unsuccessful
- **done** — mark the ticket as complete and successful

Do NOT set pipeline-managed phases (analysis, planning, in_development, in_diagnostics, diagnostics_done, in_review, reviewed, in_qa, qa_passed) — the board poller manages these automatically, and manual transitions will race with running agents.