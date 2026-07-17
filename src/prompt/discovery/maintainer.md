Generate a workspace context summary for the MAINTAINER role.

Explore this workspace and capture maintenance-relevant facts — where the codebase is healthy, where it drifts, and where simplification is likely safe.

Cover:
- Complexity hotspots — which subsystems or layers carry the most inherent complexity
- Code organization conventions — how modules, crates, or packages are structured and named
- Testing patterns and conventions — where tests live, how they're organized, what's tested vs not
- Dependency and build architecture — how external deps are managed, what build tooling is used
- Architectural patterns and idioms — recurring patterns (actor model, event-driven, DI, etc.) the Maintainer should recognize
- Behavior-sensitive areas — modules or layers where changes risk subtle breakage or cascading side-effects

Output ONLY the summary text. No JSON, no markdown fences, no explanatory text.
