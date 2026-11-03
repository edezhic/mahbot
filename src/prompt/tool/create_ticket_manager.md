Two more parameters are yours to state:

- `label` (required): the ticket's classification — `feat` when the work adds a capability, `fix` when it repairs a problem, `maint` for everything else. It is fixed when the ticket is created and cannot be changed afterwards; it is shown as a chip on the ticket's card and detail popup, and it prefixes every commit the pipeline makes for the ticket. A ticket that replaces another one states its own label; nothing is inherited.
- `priority`: optional priority level (0 = highest urgency, 1 default, 2, 3, ... higher = lower).
