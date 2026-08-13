A research team is answering the question below. You are responsible for ONE sub-question — research it thoroughly and report your findings as structured claim-level findings (same JSON schema as the extraction prompt).

# Original Question

{{question}}

# Your Sub-question

{{sub_question}}

# What Evidence Would Answer It

{{evidence_needed}}

# Queries Already Asked (do not repeat these verbatim)

{{query_ledger}}

# Scratch Workspace

This run has a temporary per-run folder you may use for scratch files
(notes, dumps, intermediate artifacts). It is wiped after the run:

{{run_root}}

Rules:
- Research your sub-question deeply and report load-bearing claims with sources.
- Do not repeat queries already asked (list above).
- If part of the sub-question is unanswerable with available evidence, record it in "unanswered" rather than fabricating.