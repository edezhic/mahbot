Analyze the current ticket and determine how ready it is for development:
- Read the ticket, comments, prior findings, and relevant workspace context.
- Search for related code paths, similar features, tests, configuration, docs, and project conventions.
- Trace likely dependencies and side effects across product behavior, UX, APIs, data, compatibility, rollout, and operations where applicable.
- Use web research when external docs, library behavior, platform constraints, or current best practices are relevant.
- Expose hidden assumptions, ambiguous requirements, missing acceptance criteria, and unspecified side effects. Do not invent requirements.

Return a structured analysis report with:
1. Goal and expected outcome
2. Relevant evidence gathered
3. Assumptions, ambiguities, and missing context
4. Risks, pitfalls, and side effects
5. Implementation and test considerations
6. Readiness verdict: list each concern you found with a grade (minor / major / blocker)

Grade the concerns:
- minor: a small gap / non-blocking note
- major: a material problem that should be resolved before development
- blocker: prevents implementing the ticket as proposed

Beware that the ticket author does not need to specify exact implementation details(interfaces, code references etc), and exact tech spec is not expected from the manager. Unless exact specification is implied by the ticket's expectations - do not grade missing details as major/blocker-level concerns. Remain focused on the product-level concerns.
