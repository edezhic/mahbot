Analyze the current ticket and determine how ready it is for development.

Research the ticket from the perspective of an engineer who may need to implement it next:
- Read the ticket, comments, prior findings, and relevant workspace context.
- Search for related code paths, similar features, tests, configuration, docs, and project conventions.
- Trace likely dependencies and side effects across product behavior, UX, APIs, data, errors, tests, compatibility, rollout, and operations where applicable.
- Use web research when external docs, library behavior, platform constraints, or current best practices are relevant.
- Expose hidden assumptions, ambiguous requirements, missing acceptance criteria, and unspecified side effects. Do not invent requirements.

Return a structured research report with:
1. Goal and expected outcome
2. Relevant evidence gathered
3. Assumptions, ambiguities, and missing context
4. Risks, pitfalls, and side effects
5. Implementation and test considerations
6. Readiness verdict with a 0-10 score, blocking issues, and non-blocking notes

Score according to general readiness:
- 0-2: unclear, infeasible, or missing the core goal
- 3-4: major unanswered product or technical questions
- 5-6: partly actionable but risky or underspecified
- 7-8: actionable with limited clarifications or manageable risk
- 9-10: clear, well-scoped, low-risk, and ready to implement