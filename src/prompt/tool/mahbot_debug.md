This tool debugs the system (harness) you operate in — other agents, tickets, configs, logs and so on — and is irrelevant for most tasks. It runs a read-only SQL query against mahbot's live databases and returns pipe-delimited text.

Pass the read-only `query` and an optional `db` selector; the accepted values are listed in the schema. Pass a single statement: with several separated by `;`, only the first runs and the rest are silently ignored.

A statement that holds the store longer than its per-statement limit is terminated and reported as too heavy.

Do not guess table or column names — introspect the schema first: `SELECT name, sql FROM sqlite_master WHERE type = 'table'` lists tables with their DDL, and `PRAGMA table_info(<table>)` lists a table's columns.

The `logs` store keeps a rolling window: log and telemetry records older than 8 hours are trimmed (failed tool calls after 7 days), so an empty answer for an older window is that retention rather than a missing table.

Validator quirks (fail-closed, by design):
- String literals, comments and quoted identifiers are stripped before keyword matching, so a mutation keyword inside them is ignored; a blocklist keyword in SQL position anywhere is still rejected.
- `BEGIN`/`COMMIT`/`ROLLBACK` and other transaction control are rejected.
- `PRAGMA` is allowed only for read-only inspection (`quick_check`, `integrity_check`, `table_info`, `table_xinfo`, index and foreign-key pragmas and the like).

Output: a pipe-delimited header then one row per line; NULL → empty, Blob → lowercase hex, Text → verbatim (pipes and newlines unescaped). Results are capped at 10,000 rows with a `truncated` sentinel row; larger sets are spilled to a file, and the inline output carries a preview plus a `[view with: read <path>]` reference.
