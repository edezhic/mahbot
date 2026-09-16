⚠️ Sanitation failure — workspace paused

Ticket: {{ticket_id}}

{{failure_details}}

---

The sanitation round could not finish on this ticket (the repository state could not be read or the final commit failed), so the workspace was frozen in the sanitation stage with its context preserved. No action is needed right now: once the workspace is unpaused the round is replayed from scratch and the commit is attempted again — the changes stay in the working tree. {{workspace_status}}