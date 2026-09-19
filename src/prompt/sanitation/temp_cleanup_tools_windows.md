Your shell is `cmd.exe`, running as the daemon (the same identity as the
service), with its working directory in the daemon's private temp root. Remove
only inside the temp roots your task lists — the verbs below are what gets it
done.

| What you need | Command |
| --- | --- |
| The daemon's identity | `whoami /user` |
| Inspect a candidate — owner and mtime, size for files | `dir /a /q /tw <path>` |
| Non-removable nodes / type audit — links and reparse points | `dir /a:l /s /b <dir>` |
| List a root / a tree | `dir /a <dir>` · `dir /a /s /b <dir>` |
| Newest agent-made mtime in a tree | `forfiles /P <dir> /S /M * /C "cmd /c echo @fdate @ftime @path"` |
| Liveness — is a path held open | there is no `lsof`. Windows itself refuses to delete a file another process holds open without a delete share, so attempt the removal and treat a sharing violation as "live, leave it" |
| Delete a file | `del /f /q <file>` |
| Delete a tree | `rd /s /q <dir>` |
| Delete an empty directory | `rd <dir>` |

A Windows temp area has no sockets, FIFOs or devices — the only by-type nodes
to watch for are links and reparse points. A removal always names a
fully-qualified path, never a wildcard sweep.

**Accepted approximation of the "leave the whole tree" rule.** There is no
shipped pre-flight probe for "is anything in this tree open", so on Windows the
liveness check happens DURING the removal: `rd /s /q` walks the tree and reports
an entry it cannot remove. That means a partly-busy tree can come back partly
removed instead of untouched. Remove a tree only when it is confidently
attributed and old enough (the age rule already protects a run in progress), and
when a removal reports a sharing violation, stop, leave the rest, and report the
tree as live.
