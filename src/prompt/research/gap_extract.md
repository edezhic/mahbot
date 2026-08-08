A research team has gathered evidence toward the question below. Produce the current gap list — the specific missing claims a fresh analyst could still hunt for.

# Original Question

{{question}}

# Round-0 Decomposition Plan

The sub_questions array below is 0-indexed — position 0 is the first sub-question.

{{plan}}

# Accumulated Evidence

{{evidence}}

Produce a JSON gap list:
{"gaps": [{"type": "unanswered|partially_answered|contradictory|low_evidence", "item": "<the specific missing claim or unanswered aspect>", "traces_to": 0, "evidence_seen": "<what evidence exists so far, if any>"}]}

Rules:
- item must name the missing claim concretely — something a fresh analyst could hunt for.
- traces_to is the 0-based index into the Round-0 plan's sub_questions array; it MUST reference an existing sub-question (0 to N-1, where N is the plan's sub-question count). Gaps that trace to no sub-question are dropped.
- type: unanswered = no evidence found; partially_answered = only weak or indirect evidence; contradictory = sources disagree; low_evidence = thin or single-source evidence.