Generate a workspace context summary for the MAINTAINER role.

Explore this workspace and capture maintenance-relevant facts — where the codebase is healthy, where it drifts, and where simplification is likely safe.

Cover:
- Complexity hotspots and convoluted areas
- Duplication, stale comments, and dead-code signals
- Safe refactor boundaries — modules or layers where cleanup is low-risk
- Modules where simplification is likely to pay off
- Dependencies and config that tend to drift or accumulate cruft
- Behavior-sensitive areas where refactoring would need extra care

Output ONLY the summary text. No JSON, no markdown fences, no explanatory text.
