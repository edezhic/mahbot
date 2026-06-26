Generate a workspace context summary for the SANITATION role.

Explore this workspace and capture project-specific facts that inform cleanup and code-quality maintenance.

Cover:
- Build artifacts and output directories (target/, dist/, build/, *.o, *.pyc, etc.)
- Temp and cache patterns (editor swap files, .DS_Store, __pycache__, .cache/, etc.)
- Vendored and dependency directories (node_modules/, .venv/, vendor/, .dub/, etc.)
- Project conventions for ignored/untracked files (.gitignore patterns, .dockerignore, known non-source assets)
- Known cruft areas: dead code, orphaned files, stale comments, TODO/FIXME density
- Dependency hygiene: outdated lockfiles, duplicate or unused dependencies
- File-system bloat indicators: oversized assets, generated files checked into version control

Output ONLY the summary text. No JSON, no markdown fences, no explanatory text.