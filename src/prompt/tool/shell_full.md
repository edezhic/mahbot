## Background mode

For long-running non-interactive commands set `background: true`; the command keeps running after the call returns.

- The tool answers with the path of its output file, where stdout and stderr are written raw.
- {{trailing_separator}}
- Read progress with the read tool on that path: `[exit status: N]` at the end means it finished, its absence that it is still running.
- `timeout_secs` is ignored; the command runs until it exits, you stop it, or your run ends.
- Stop it with the same tool, passing only `stop: "<output-file-path>"`. {{stop_semantics}}

Output is unbounded — prefer modest output. Background greps bypass the engine and run the host's own `grep`.
