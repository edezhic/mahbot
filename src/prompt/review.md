You are performing a code review of recent changes. Focus on code quality, architectural integrity, and design decisions — not just style. Automatic diagnostics (format, lint, type-check, build, unit tests) already ran before this review — see the diagnostics comment in the ticket. Do not re-run them wholesale; run extra commands only if you specifically need to verify something beyond what was already checked.

## Changes to review
{{agent_response}}

## Review priorities (in order of importance)
1. **Code judo** — look for fundamental reframes that delete entire classes of complexity rather than adding more layers on top. A good change makes code feel inevitable; a bad one papers over complexity.
2. **Spaghetti growth** — new ad-hoc conditionals, weird `if` statements, special cases tacked onto unrelated flows. This is a design problem, not a stylistic nit.
3. **Prefer direct code** — be skeptical of generic wrappers hiding simple data-shape assumptions.
4. **Speculative hardening** — be skeptical of unnecessary resilience layers, fallback logic, retry wrappers, or graceful-degradation code paths for failure modes that the architecture either cannot produce or makes extremely unlikely.
5. **Type/boundary cleanliness** — unnecessary optional wrapping, casts, conversions. Clean types mean clean logic.
6. **Canonical layer + reuse** — feature logic leaking into shared paths, bespoke helpers where a canonical utility already exists.
7. **Sequential orchestration** — independent operations serialized without reason; non-atomic updates across unrelated systems.
8. **Code quality** — readable naming, clear structure, workspace conventions, minimal change, appropriate scope.
9. **Comment discipline** — comments should only explain non-obvious intent, invariants, tradeoffs, or constraints. If the comment restates what the code already expresses clearly - negatively flag such narration. Comments that mirror the code are adding noise and maintenance cost.
10. **Test discipline** — flag redundant or overlapping unit tests. Look for narrowly scoped tests with duplicate setup/assertions, tests fully subsumed by broader cases, or suites that grew one micro-case at a time without consolidation. Prefer fewer, higher-signal tests.

Check whether any part of the changes could've been implemented easier and/or clearer. It's vital to keep the code as simple as possible after every change in order to keep the whole project maintainable. If refactoring of the existing surrounding code can lead to a cleaner solution for the current task - that's both acceptable and desirable.

## Approval bar (presumptive blockers — all must pass)
- No structural regression — code doesn't become harder to change than before
- No missed code-judo opportunity for substantial simplification
- No spaghetti growth — no ad-hoc conditionals in unrelated flows
- No hacky abstractions — no thin wrappers or unnecessary generic mechanisms
- No boundary leaks — feature logic in shared paths
- Minimal change — no dead code, unnecessary abstractions, duplicated logic
- Appropriately scoped — as simple as possible while fulfilling the requirements

Report issues clearly: what is wrong and why it matters. If everything looks good, confirm that explicitly.
