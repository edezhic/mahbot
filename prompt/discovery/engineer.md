Generate a workspace context summary for the ENGINEER role.

Explore this workspace and capture the architectural map an engineer needs to implement changes correctly.

Cover:
- Entrypoints and how the application starts
- Major subsystems and how they relate
- Data flow through the system
- Persistence and runtime model (databases, files, in-memory state, async tasks)
- Concurrency and task boundaries
- Configuration and environment dependencies
- Important invariants and integration points
- Core dependencies with version-grounded official docs for APIs the engineer is likely to touch

Search the web for official documentation of the languages, frameworks, and libraries detected in the workspace. Anchor to versions from project files and lockfiles.

Output ONLY the summary text. No JSON, no markdown fences, no explanatory text.
