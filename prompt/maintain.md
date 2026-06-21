I need you to thoroughly investigate the codebase and look for refactoring opportunities. 
Any simplification of the code without breaking current behaviour is relevant. 
Don't rush and look where code is unnecessarily convoluted - our goal is to identify improvements. 

Splitting up existing modules/functions without actually reducing complexity and the amount of code is very low priority. 
Aligning names of variables/functions/etc, updating stale comments, trimming narration comments, consolidating overlapping tests, deduplicating things is medium priority. 
Untangling complex code while resolving subtle bugs is high priority. 

## Code quality patterns to detect
When investigating, also look for these specific design issues:
- **Code judo** — fundamental reframes that could delete entire classes of complexity. Code that could feel inevitable but currently feels overcomplicated.
- **Spaghetti growth** — ad-hoc conditionals, weird `if` statements, special cases tacked onto unrelated flows. These are design problems.
- **Prefer direct code** — thin wrappers, identity abstractions, unnecessary generic mechanisms hiding simple data-shape assumptions.
- **Sequential orchestration smell** — independent operations serialized without reason; non-atomic updates across unrelated systems.
- **Type/boundary issues** — unnecessary casts, `as` conversions, redundant `Option` wrapping.
- **Canonical layer violations** — feature logic leaking into shared paths; bespoke helpers where a canonical utility already exists.
- **Premature optimization** - most projects aren't high-load low-latency services and would benefit more from cleaner code than saving a few microseconds of CPU time.
- **Outdated docs** - comments and documentation whose contents don't match the actual behaviour/code.
- **Narration comments** — comments that restate the code line-by-line instead of explaining non-obvious intent, invariants, or tradeoffs.
- **Test suite bloat** — clusters of narrow unit tests with overlapping scenarios; cases where one broader test already covers another; opportunities to merge, parametrize, or remove subsumed tests. Overtesting - unit-tests for simple straightforward code just for the sake of coverage.
- **Confusing naming** - variable passed into the function with different arg name; same-meaning variables named differently in different places.

Do NOT make any direct code changes. Use read/search tools and the Ask tool (to spawn analyst sub-agents) to investigate signals, and create_ticket to document findings.

When you find a refactoring opportunity, create a backlog ticket on the board describing:
- What the issue is (complex code, dead code, inconsistency, etc.)
- What the improvement would be
**It's much better to focus on just a couple of refactoring opportunities and study them in-depth then to create a dozen of tickets. Make sure to create tickets only when you're sure that the change will actually lead to simplification and will not bloat the code.**

Don't worry if some investigation will lead into a dead end. One good ticket (reduces complexity, reduces total LoC, easily readable code) or even none at all is MUCH better than a bunch of bad ones. Keep in mind that there is no such thing as a perfect codebase - sometimes finding a good refactoring opportunity is hard and might require considering a lot of related pieces to avoid breaking things and to actually improve things. Don't hesitate to consider major changes, but carefully reconsider their every detail before proposing a ticket. Don't hesitate to create small tickets either - if you see a small inconsistency in the comments or even variable names it's still better to refactor than to leave as is.
