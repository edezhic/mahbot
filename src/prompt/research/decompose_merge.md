Three independent analysts produced decomposition plans for the question below. Merge them into ONE consolidated plan that best covers the question's scope: keep the strongest sub-questions, drop redundant or weak ones, and keep the risk labels. Merge a question's scope well — aim for a compact plan that is neither padded with near-duplicates nor so thin it leaves the question under-covered.

# Original Question

{{question}}

# Independent Plans

Each plan item has a GLOBAL item id, numbered flat across all three plans (ids 0..N in plan order: plan 0's items first, then plan 1's, then plan 2's).

{{plans}}

Produce a JSON plan:
{"sub_questions": [{"from_id": 0, "also_ids": [1]}], "dropped": [{"id": 2}]}

Rules:
- Every merged sub-question cites the id of the plan item it is a verbatim copy of ("from_id"); "also_ids" lists the ids of identical items in the other plans (empty when none). The system resolves the id to the item's question/evidence_needed/risk.
- Every dropped item is cited by its id. Identical tuples in different plans are distinct ids — each must be covered separately (merged via also_ids or dropped).
- Every plan item id must be covered EXACTLY once — either merged (via from_id or also_ids) or listed in "dropped". No item may be silently omitted, and none may appear twice.
- Each merged sub-question must be independently researchable, non-overlapping, and collectively exhaustive of the original question's scope.
- risk reflects how hard it will be to find solid evidence (high = contested, sparse sources, or a fast-moving topic).