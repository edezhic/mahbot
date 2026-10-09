Your workspace:
- OS: {{operating_system}}
- System locale: {{system_locale}}
- Workspace path: `{{workspace}}`
{{path_frame}}
If you need to create temporary files during your work, use the OS temp directory the shell environment points at (`$TMPDIR` on unix, `%TMP%`/`%TEMP%` on Windows) — never create temp files directly in the workspace that could be mistaken for project artifacts.

{{workspace_context}}
