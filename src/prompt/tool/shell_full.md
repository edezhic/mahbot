Execute a shell command. Commands run from the workspace directory. Use when a task needs a command-line operation that is not better handled by read, edit or search. Output is filtered: 1MB hard truncation; generic output shows first/last ~10 lines (max 200) with long lines (>500 chars) shortened; JSON output is replaced with a summary preview; `cargo test` hides passing tests, showing only failures + summary; stderr on success only shows `warning:` and error lines. Large output spills to a temp file with a `[view with: read <path>]` hint — use `read` on that path for full unfiltered output.

## Background mode

For long-running non-interactive commands (e.g. starting a dev server that must keep running), set `background: true` (default false). The command then keeps running after this tool call returns:

- The tool returns the path of the command's output file (in the OS temp area's `.agent` directory). All output — stdout and stderr — is written to that file RAW: no truncation, no filtering, no credential scrubbing. Do not run commands that print secrets in background mode.
- Do NOT append a trailing `&` (shell background operator) to the command — `background: true` already detaches it, and a `&` makes the wrapping shell exit immediately, so the session's watchdog treats the command as finished and kills the long-running process you meant to keep alive.
- Read progress with the read tool on that path. When the command exits, the line `[exit status: N]` is appended to the end of the file UNCONDITIONALLY — including exit 0 (signal kills read `[exit status: terminated by signal]`). Its presence with ANY exit code is the only signal that the command finished; its absence means the command is still running. Do not treat a non-zero exit as a launch failure — it is a normal completion.
- `timeout_secs` is IGNORED in background mode. The command runs until it exits, you stop it, or your run ends.
- Stop the session with the same tool: pass only `stop: "<output-file-path>"` (the exact path returned by the launch). Stop is two-stage: SIGTERM, ~5s grace, then SIGKILL to the whole process group. Stopping an already-finished session is a no-op. Do not combine `stop` with `background` or `command`.
- Sessions are strictly scoped to the agent that started them: no other agent can read or stop them, and they are force-killed when your run ends (success, error, or abort). Never assume a background process survives past your run — if you need it later, restart it.
- Launch failures (command not found, not executable) are returned as a tool error immediately — check the tool response before assuming the session started.

Caveats: output is unbounded — a command that prints forever will fill the temp disk. Prefer commands that write modest output, and tail large outputs via the shell tool (the read tool caps at 10 MB). Output files persist until the next daemon startup.
