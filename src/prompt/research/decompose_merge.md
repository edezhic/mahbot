Three independent analysts produced decomposition plans for the question below. Merge them into ONE consolidated plan of 4–6 sub-questions that best covers the question's scope: keep the strongest sub-questions, drop redundant or weak ones, and keep the risk labels.

# Original Question

{{question}}

# Independent Plans

{{plans}}

Produce a JSON plan:
{"sub_questions": [{"question": "<sub-question>", "evidence_needed": "<what evidence would answer it>", "risk": "low|medium|high", "from_plan": 0, "also_in_plans": [1]}], "dropped": [{"question": "<dropped sub-question>", "evidence_needed": "<its evidence_needed>", "risk": "<its risk>", "reason": "<why it was dropped>"}]}

Rules:
- 4 to 6 sub-questions in "sub_questions".
- Every merged sub-question must be a VERBATIM copy (exact text of question, evidence_needed, and risk) of an item from one of the independent plans: "from_plan" is the 0-based index of that plan in the Independent Plans list; "also_in_plans" lists every OTHER plan containing the identical verbatim item (empty when none).
- Every sub-question item in every independent plan must be covered EXACTLY once — either merged (cited via from_plan or also_in_plans) or listed in "dropped". No item may be silently omitted, and none may appear twice.
- "dropped" entries are verbatim copies too (same question, evidence_needed, risk), with a reason.
- Each sub-question must be independently researchable, non-overlapping, and collectively exhaustive of the original question's scope.
- risk reflects how hard it will be to find solid evidence (high = contested, sparse sources, or a fast-moving topic).