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
6. Readiness verdict with a 0-10 score, blocking issues, and non-blocking notes

Score according to general readiness:
- 1-3: unclear and/or infeasible
- 4-6: major unanswered product questions
- 7-9: minor underspecified product changes
- 10: crystal clear without any unspecified side-effects