## Grep notes

Greps are served by a fast built-in engine instead of a system search program; a grep the engine cannot serve is handled the way the platform notes below describe. Engine-served recursive walks skip hidden content and everything the ignore rules exclude (e.g. `target/`, `.git/`, `node_modules/`) — `.ignore` files everywhere, and inside a git repository also `.gitignore`, the repository's own `.git/info/exclude` and your global git ignore. Matches under those paths are not found: to search there, pass an explicit path or `cd` into the subdir. Engine-served recursive greps may also list files in a different order than a system `grep` (parallel walk); the set of matching lines is the same.

{{platform_notes}}
