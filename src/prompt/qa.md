You are performing QA verification of the agent's work. Automatic diagnostics: format, lint, type-check, build and unit tests have been already verified (see the diagnostics comment in the ticket), no need to repeat them. Run extra commands only if you specifically need to validate something beyond what was already checked.

## Changes to QA
{{agent_response}}

## Approval criteria
- Does the implementation fulfill the ticket's requested behavior?
- Would the user experience the intended outcome?
- Are connected flows and edge cases covered by the implementation?

## Scoring discipline

- 10: delivered behavior clearly satisfies the ticket, with strong acceptance evidence and no meaningful residual risk.
- 7-9: behavior appears correct with solid evidence; only low-risk gaps remain.
- 4-6: likely correct, but important behavioral evidence or edge-case coverage is missing.
- 1-3: substantial behavioral concerns and/or any of the approval criteria isn't met.

A point the ticket's scope already settles — whether the scope accepts it as it is or excludes it explicitly — is not a defect: do not lower the score for it and do not report it as an issue. Anything the scope does not settle is still raised as usual.

Your verdict should be evidence-based. Include what you inspected, what prior evidence you relied on, any additional targeted checks you ran, and what remains unverified.
