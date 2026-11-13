Break the following question into sub-questions that together would fully answer it. Each sub-question must be independently researchable, non-overlapping, and collectively exhaustive of the original question's scope.

# Question

{{question}}

# Scratch Workspace

This run has a temporary per-run folder you may use for scratch files
(notes, dumps, intermediate artifacts). It is wiped after the run:

{{run_root}}

# How Wide

The run that works this plan can spend {{max_slots}} analyst slots on it: a low- or medium-risk sub-question costs one slot, a high-risk one costs two. Produce a plan that run can finish.

Produce a JSON plan:
{"sub_questions": [{"question": "<sub-question>", "evidence_needed": "<what evidence would answer it>", "risk": "low|medium|high"}]}

Rules:
- Cover the question's full scope — split it at its natural seams, neither padding with near-duplicates nor leaving parts under-covered.
- Where the question's scope needs more directions than fit the allowance above, merge closely related ones into a single wider sub-question. Never drop part of the scope, and never exceed the allowance.
- risk reflects how hard it will be to find solid evidence (high = contested, sparse sources, or a fast-moving topic).
- Each sub-question must name the evidence that would settle it.
