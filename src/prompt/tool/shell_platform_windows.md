## This platform

- There is no POSIX tool set: `find`, `sed` or `ls` is some other program or nothing, and `python` may be a placeholder that opens the store. A separately installed set (Git, MSYS2, WSL, a vendor distribution) may or may not be on the path — check first.
- `;` is not a command separator: write two commands with `&&` (or `&`), and quote a literal `;`. A line with an unquoted `;` is refused.
- `#` is not a comment: what follows is an argument. Quote a literal `#`; a line where an unquoted `#` starts a word is refused.
- Those refusals read the line as a Unix command: a shape only cmd.exe understands (`if exist x (a;b)`, `for %i in (…) do …`) is refused in read-only mode but handed to the interpreter as one line in full mode. Quote a literal `;` or `#` either way.
- The interpreter starts with command extensions on, delayed expansion off and this machine's auto-run scripts disabled; an interpreter started inside a command (`cmd`, `powershell`) keeps the machine's own settings.
- Output in the machine's console encoding is read back in that encoding; a program writing another encoding can read back wrong.
