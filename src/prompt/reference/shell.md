# shell — reference

## Arguments

- `command` — the shell command to execute. Required for a normal run and for a background one; in full mode it is ignored when `stop` is set.
- `timeout_secs` — timeout in seconds, default 600, max 3600. Full mode only: a background run ignores it.
- `background` — full mode only: run detached and return the path of the output file. Default false.
- `stop` — full mode only: the output-file path of a background session, passed alone.

## Output filtering

- 256 KB hard truncation per stream.
- Generic output shows first/last ~10 lines (max 200), with long lines (>500 characters) shortened.
- JSON output is replaced with a summary preview.
- `cargo test` hides passing tests, showing only failures and the summary.
- stderr on success shows only `warning:` and error lines.
- Large output spills to a temp file and prints a `[view with: read <path>]` hint; read that path for the full, unfiltered output.

## Read-only guard

These are the checks behind the banner's rules, not a complete account of every spelling.

### Unix

- Redirect and mutator spellings the guard knows: `> file`, `sed -i`, `awk -i inplace`, `dd of=`, `tee`/`touch`/`cp`/`mv`/`rm`/`mkdir` against outside paths, and package managers (`npm`, `pip`, `brew`, …). A scratch directory can be bound with `NAME=$(mktemp -d -p "$TMPDIR")` and referenced as `$NAME` (a bare `mktemp -d` on macOS ignores TMPDIR and lands elsewhere, still inside the allowed roots).
- A command word bound to a literal in the same invocation passes (`BIN=cargo; "$BIN" --list`); substitution-derived bindings (`BIN=$(which cargo)`), transitive chains (`X=rm; Y="$X"`), and unbound variables are rejected.
- A destructive verb in a different case is not recognised by the guard, though on a case-insensitive filesystem it still resolves.
- What an approved program starts with is the loader's: `LD_PRELOAD`/`LD_LIBRARY_PATH` (macOS: `DYLD_INSERT_LIBRARIES`/`DYLD_LIBRARY_PATH`); `$TMPDIR`/`$HOME` are the values the product pins.

### Windows

- File and tree mutator spellings: `del`/`erase`/`rd`/`md`/`move`/`ren`/`replace`/`mklink`; `copy`/`xcopy`/`robocopy` are gated on destination.
- The parser reads these spellings differently from cmd.exe: a relative operand never satisfies the temp gate, and `del /f,x` is refused.
- Wrapping in another interpreter or launcher is admitted. A command word spelled `%DEL%`, `d^el` or `del.` passes; so does a word escaping a separator or redirect character, unless it is read as a path operand of a known verb (`del %TEMP%\a\&b.txt` is refused).
- `PATHEXT` decides which extensions a bare name may carry; `ComSpec`, which interpreter a nested command gets.

### Common

- Why `GIT_PAGER` is excepted: a pager never spawns on captured output. `git -C <path> <read-only subcommand>` is allowed.
- `tar`'s list forms: `tar -tzf f`, `tar tzf f`, `tar --list`; extraction and creation are rejected in every form, including into temp.
- The read-only calls it allows: `mahbot debug`, `__grep-engine`, `__env-dump`, `-h`/`--help`/`-V`/`--version`. Its other command lines start, change or drive the product; its `config`/`workspace` families carry configuration and workspace authority, its `send` family dispatches agents, its `audio` operations write files or fetch models, and its producing families make documents, pictures and clips with the owner's tools and account. A launch that finds it running changes nothing: no window, no service, no restart.
- What the guard approves is the command's words, not what they run: commands run in the owner's environment, so an approved name runs whatever the owner's search path resolves it to. Nothing is compensated for.

## Background sessions (full mode)

- The output file lives in the OS temp area's `.agent` directory.
- `[exit status: terminated by signal]` means a signal ended it.
- Sessions are scoped to the agent that started them; no other agent can read or stop one, and the tree is destroyed when the run ends. Never assume a background process survives the run.
- Launch failures (command not found, not executable) are tool errors immediately on Unix; on Windows a missing or non-executable command starts a session whose file holds the interpreter's error and ends with `[exit status: 1]`.
- The read tool caps text files at 10 MB. Output files persist until the periodic temp cleaner or the OS temp sweep reclaims them; the daemon performs no startup purge.

## Grep engine

- A recursive walk may list files in a different order than system `grep`; the matching lines are the same.
- Windows: the shapes it recognises but cannot serve are a grep nested in another command (another interpreter's command line, `xargs grep …`, `find … -exec grep … +`, a second grep in a pipeline), an unsupported flag, a redirect glued to a search word (`grep -n x>out.txt` — spell it `grep -n x f.txt > out.txt`), a member that sets state a later member reads, an untracked `cd`, or a `mahbot` call that is not one plain call. A bare `mahbot` runs the product's own copy from its own full path (`mahbot -V`, `mahbot debug --db board "…"`), while a spelled-out other copy (`.\mahbot.exe`, `target\debug\mahbot.exe`) is refused. Left to the platform as written: its own program runs and you see its outcome, never an empty set — a search another program owns (`git grep …`), a search in the arguments of an unlisted program, a search whose program is spelled with a path, a grep word glued to `{` or a redirect operator (`{grep x f.txt`, `grep>x f.txt`), and `find … -exec grep … \;`. A word with two `%` is refused: `%…%` expands before any program runs, while a single `%` and every `!` are literal.
