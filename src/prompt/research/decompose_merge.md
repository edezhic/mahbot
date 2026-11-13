Two independent analysts produced decomposition plans for the question below. Merge them into ONE consolidated plan that covers the question's full scope: keep the strongest sub-questions, fold closely related ones into a single wider direction, drop only genuine duplicates, and keep the risk labels.

# Original Question

{{question}}

# Independent Plans

Each plan item has a GLOBAL item id, numbered flat across both plans (ids 0..N in plan order: plan 0's items first, then plan 1's).

{{plans}}

Produce a JSON plan:
{"sub_questions": [{"from_id": 0, "also_ids": [1, 2]}, {"from_id": 3, "also_ids": []}], "dropped": []}

Rules:
- "from_id" is the plan item the merged direction is primarily built from; "also_ids" lists every other item folded into that same direction — identical items from the other plan AND closely related ones you merged into one wider direction. An entry may cite any number of ids; the system composes the direction's question and evidence need from all of them (deduplicated) and gives it the highest risk label cited. So folding is expressed by citing several ids on one entry — never by dropping the folded-in item.
- "dropped" is only for a genuine duplicate — an item whose direction a merged entry already names.
- Every plan item id must be covered EXACTLY once — either merged (via from_id or also_ids) or listed in "dropped". No item may be silently omitted, and none may appear twice.
- Each merged sub-question must be independently researchable, non-overlapping, and collectively exhaustive of the original question's scope.
- The merged plan must cost at most {{max_slots}} analyst slots (a low- or medium-risk direction costs one, a high-risk one two). Where the two plans together name more directions than fit, fold related ones into one wider direction — the split stays complete, and an item is never dropped for size.
- risk reflects how hard it will be to find solid evidence (high = contested, sparse sources, or a fast-moving topic).
