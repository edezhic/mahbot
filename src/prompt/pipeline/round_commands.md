The workspace's own project commands run in this round, in parallel with you — the pipeline starts them itself. They run in this order and stop at the first failure:

{{commands}}

Do not run them yourself: they are part of this round's verification and their result is recorded next to your verdict. A failing command sends the ticket back to the engineer with the full picture, so re-running one or making up for it is not your job.