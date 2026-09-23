⚠️ READ-ONLY MODE: You are not permitted to modify the workspace. Use this tool only for inspection: reading files, listing directories, running builds/checks, git status/log/diff, searching, etc. Writing to the OS temp directory is allowed — use a literal path under the daemon temp root `{{temp_root}}`, the directory this session's temp environment variables point at.
The guard rejects before execution — these are its checks, not a complete account of every way a command can be spelled:
{{platform_checks}}
- git config/repository injection: global `-c`, `--config-*`, `--exec-path`, `--git-dir`, `--work-tree`, and `GIT_*` environment variables (`GIT_PAGER` excepted — a pager never spawns on captured output). `git -C <path> <read-only subcommand>` is allowed.
- `tar` is list-only (`tar -tzf f`, `tar tzf f`, `tar --list`) — extraction/creation is rejected in every form, including into temp directories.
- What the guard approves is the command's *words*, not what they run: your commands run in the owner's own environment, so a name the lists above approve still runs whatever his search path resolves it to. Nothing is compensated for.

Do not work around the guard: re-invoking a blocked command through a built binary, a relocated copy or another deliberate spelling is not permitted — the limits listed above are not extra permissions. If you absolutely require this action to do your job - note it in your final response instead of attempting it further.
