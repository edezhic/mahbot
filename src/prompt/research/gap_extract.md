A research team has gathered evidence toward the question below. Produce the current gap list — the specific missing claims a fresh analyst could still hunt for.

# Original Question

{{question}}

# Round-0 Decomposition Plan

{{plan}}

# Accumulated Evidence

{{evidence}}

Produce a JSON gap list:
{"gaps": [{"type": "unanswered|partially_answered|contradictory|low_evidence", "item": "<the specific missing claim or unanswered aspect>", "traces_to": "<the sub-question from the round-0 plan this gap traces to>", "evidence_seen": "<what evidence exists so far, if any>"}]}

Rules:
- item must name the missing claim concretely — something a fresh analyst could hunt for.
- traces_to must reference a sub-question from the round-0 decomposition plan; gaps that trace to nothing are dropped.
- type: unanswered = no evidence found; partially_answered = only weak or indirect evidence; contradictory = sources disagree; low_evidence = thin or single-source evidence.