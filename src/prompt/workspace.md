Your workspace:
- OS: {{operating_system}}
- Workspace path: `{{workspace}}`
Do not mutate anything outside the workspace (or OS's temp folder) unless explicitly requested by the user.
If you need to create temporary files during your work, use the OS temp directory (e.g., `/tmp/` or `$TMPDIR`) — never create temp files directly in the workspace that could be mistaken for project artifacts.

{{workspace_context}}
