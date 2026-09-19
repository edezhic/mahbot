Your shell runs as the daemon (the same identity as the service), with its
working directory in the daemon's private temp root.

| What you need | Command |
| --- | --- |
| The daemon's identity | `id -u` |
| Inspect a candidate — owner, type, mode, size, mtime | macOS `stat -f '%N %Su %Sp %z %Sm' <path>` · Linux `stat -c '%n %U %A %s %y' <path>` |
| Probe a path's type | `file <path>` |
| List a root / a tree | `ls -la <dir>` · `find <root> -maxdepth 3` |
| Newest agent-made mtime in a tree | macOS `find <dir> -type f -exec stat -f '%m %N' {} +` · Linux `find <dir> -type f -exec stat -c '%Y %n' {} +` |
| Non-regular nodes in a tree | `find <dir> \( -type s -o -type p -o -type b -o -type c \)` |
| Liveness — is a path held open | `lsof <path>` · `lsof +D <dir>` for a tree |
| Delete a file | `rm <file>` |
| Delete a tree | `rm -rf <dir>` |
| Delete an empty directory | `rmdir <dir>` |

`find -delete` is never used, and a removal always names a fully-qualified path.
