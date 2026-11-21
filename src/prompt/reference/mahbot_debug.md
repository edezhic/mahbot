# mahbot_debug — reference

One read-only SQL statement against mahbot's live databases, answered as pipe-delimited text.

## Arguments

- `query` — the statement. Pass a single statement: with several separated by `;`, only the first runs and the rest are silently ignored.
- `db` — the store to query: a logical store (`board`, `sessions`, `workspaces`, `users`, `config`, `chat_history`, `automations`, `custom_roles`, `telegram_intake`), `logs`, or `core` — the consolidated file and the default.

## Query rules

The validator is fail-closed, by design:

- String literals, comments and quoted identifiers are stripped before keyword matching, so a mutation keyword inside them is ignored; a blocklist keyword in SQL position anywhere is still rejected.
- `BEGIN`, `COMMIT`, `ROLLBACK` and other transaction control are rejected.
- `PRAGMA` is allowed only for read-only inspection — `quick_check`, `integrity_check`, `table_info`, `table_xinfo`, index and foreign-key pragmas and the like.
- A statement that holds the store longer than its per-statement limit is terminated and reported as too heavy.

Do not guess table or column names — introspect the schema first: `SELECT name, sql FROM sqlite_master WHERE type = 'table'` lists tables with their DDL, and `PRAGMA table_info(<table>)` lists a table's columns.

## Retention

The `logs` store keeps a rolling window: log and telemetry records older than 8 hours are trimmed, failed tool calls after 7 days. An empty answer for a window older than that is not a missing table.

## Output

A pipe-delimited header then one row per line. NULL is an empty cell, a Blob is lowercase hex, Text is verbatim (pipes and newlines unescaped), except that a model control token in a text cell is masked to `[model-control-token]` — stored text may predate that mask, and a token read back into your context kills the session that read it.

Results are capped at 10,000 rows with a `truncated` sentinel row; larger sets are spilled to a file, and the inline output carries a preview plus a `[view with: read <path>]` reference.
