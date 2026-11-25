⚠️ Failure notification

{{failure_details}}

---

This failed ticket parks the workspace: while a ticket sits in `failed`, nothing here is picked up for development. The park lifts only once nothing here is left in `failed`: a failed ticket is replaced by a supersede, or cancelled, never re-queued.

Superseding is how work resumes: file a ticket on the same scope that finishes what the failed one started, or reverts its work in progress, with `supersede` set to this ticket. That cancels and archives the failed ticket and links the replacement to it as its successor. Move the replacement straight to `queued` - it is the one ticket that skips analysis and planning. Cancelling ends the failure as well, but it abandons the unverified work in the tree, so deal with the uncommitted changes before anything moves back into development.

If the replacement has already been filed as an ordinary ticket beside the failed one, cancel that stray ticket and file the replacement as a supersede of the failed one - a new ticket beside a failed one never resumes the workspace.

The board in your context is a snapshot from when this session started or was last compacted, not the current state: list the board before triaging, since a ticket it shows as in development may already have failed.

{{workspace_status}} Only after the failure reason is fully resolved - continue development in the workspace. Check what happened in the failed ticket and dispatch an analyst if anything isn't clear. It's your responsibility to diagnose what went wrong with the help of analysts and to see the failed work closed by its replacement. When a ticket finishes successfully the pipeline commits on its own - committing is never yours to request.