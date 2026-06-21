Generate a workspace context summary for the REVIEWER role.

Explore this workspace and capture project-specific facts that inform code review judgments.

Cover:
- Architectural boundaries that should not be crossed
- Invariants the codebase relies on
- Local anti-patterns — things that look fine generically but are wrong here
- Common risk areas: persistence, concurrency, provider integrations, shared utilities, etc.
- Dependency and API caveats specific to this project
- Areas where changes often create structural regressions or hidden coupling

Search the web for official documentation where it helps clarify version-specific API behavior or framework constraints relevant to reviewing changes in this workspace.

Output ONLY the summary text. No JSON, no markdown fences, no explanatory text.
