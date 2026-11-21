Debugs the system you operate in — agents, tickets, configs, logs — and is irrelevant for most tasks. It runs one read-only SQL statement against mahbot's live databases and returns pipe-delimited text.

Safety: `query` is read-only and same-user, and the tool fails closed — a mutating statement, transaction control or a non-inspection `PRAGMA` is rejected before execution. Reach the live stores only through this tool, never through another tool or a sqlite binary.

Introspect table and column names first instead of guessing.

An answer is a pipe-delimited header then one row per line; a large result is capped and spilled to a file with a `[view with: read <path>]` hint.
