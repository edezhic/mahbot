## Background mode

For long-running non-interactive commands set `background: true` (default false); the command keeps running after this tool call returns.

- The tool returns the path of the command's output file (in the OS temp area's `.agent` directory). All output — stdout and stderr — is written there RAW: no truncation, no filtering, no credential scrubbing. Do not run commands that print secrets in background mode.
- {{trailing_separator}}
- Read progress with the read tool on that path. When the command exits, `[exit status: N]` is appended to the end of the file UNCONDITIONALLY — including exit 0 (`[exit status: terminated by signal]` if a signal ended it). Its presence with ANY exit code is the only signal that the command finished; its absence means it is still running, and a non-zero exit is a normal completion, not a launch failure.
- `timeout_secs` is IGNORED in background mode: the command runs until it exits, you stop it, or your run ends.
- Stop the session with the same tool: pass only `stop: "<output-file-path>"` (the exact path returned by the launch). {{stop_semantics}} Stopping an already-finished session is a no-op. Do not combine `stop` with `background` or `command`.
- Sessions are strictly scoped to the agent that started them: no other agent can read or stop them, and when your run ends — success, error, or abort — the session's process tree is destroyed. Never assume a background process survives past your run; restart it if you need it later.
- Launch failures (command not found, not executable) are returned as a tool error immediately — check the response before assuming the session started. That check is Unix-only: on Windows a missing or non-executable command starts a session whose file holds the interpreter's error and ends with `[exit status: 1]`.

Caveats: output is unbounded — a command that prints forever will fill the temp disk. Prefer modest output and tail large outputs via the shell tool (the read tool caps text files at 10 MB). Output files persist in the temp area until the periodic temp cleaner or the OS temp sweep reclaims them (the daemon performs no startup purge).

Background-mode greps are NOT served by the engine — `background: true` bypasses the interception, so the Grep notes below do not apply: the `grep` that runs is the host's own, if the host has one.
