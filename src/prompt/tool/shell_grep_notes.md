## Grep notes

Greps are served by a fast built-in engine instead of a system search program; one it cannot serve is handled by the platform notes below. Engine-served recursive walks skip hidden content and everything the ignore rules exclude (`target/`, `.git/`, `node_modules/`, `.ignore` files, `.gitignore`, `.git/info/exclude` and your global git ignore), so matches under those paths are not found — pass an explicit path or `cd` into the subdir to search there. A recursive walk may list files in a different order than a system `grep`; the matching lines are the same.

{{platform_notes}}
