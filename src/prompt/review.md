Review the changes made in the scope of the current ticket. Automatic diagnostics: format, lint, type-check, build and unit tests have been already verified (see the diagnostics comment in the ticket), no need to repeat them. Run extra commands only if you specifically need to validate something beyond what was already checked.

## Changes to review
{{agent_response}}

## Approval criteria
- No structural regression — code doesn't become harder to change than before
- No missed opportunity for substantial simplification
- No spaghetti growth — no ad-hoc conditionals in unrelated flows
- No hacky abstractions — no thin wrappers or unnecessary generic mechanisms
- No boundary leaks — feature logic in shared paths
- Minimal change — no dead code, unnecessary abstractions, duplicated logic, redundant tests or comments
- Appropriately scoped — as simple as possible while fulfilling the requirements

## Scoring discipline

- 10: all approval criteria are clearly met and the solution is correct.
- 7-9: only low-risk correctness concerns or debatable code nits are left.
- 4-6: likely correct, but visible code quality issues are found.
- 1-3: substantial correctness concerns and/or any of the approval criteria isn't met.

Report issues clearly: what is wrong and why it matters. If everything looks good, confirm that explicitly.
