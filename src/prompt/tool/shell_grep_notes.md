## Grep notes

Greps are served by a fast built-in engine instead of a system search program; one it cannot serve falls to the platform's own search. Served recursive walks skip hidden content and gitignore-excluded paths (`target/`, `.git/`, `node_modules/`), so pass an explicit path or `cd` in to search there.

{{platform_notes}}
