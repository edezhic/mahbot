This tool is designed for cases when you need to debug the system(harness) you're(the agent) operating in - inspect other agents, tickets, configs, logs etc. It's irrelevant for most tasks, but can be useful in some rare cases when the past or ongoing work of other agents requires inspection. Runs a read-only SQL query against mahbot's live databases and returns the result as pipe-delimited text. 

Pass the read-only `query` and, optionally, a `db` selector naming the target database — the consolidated `core` store by default, or `logs`. The accepted `db` values are listed in the tool's schema. Only a single statement is supported: multi-statement queries (e.g. two `SELECT`s separated by `;`) are rejected by the engine. Do not even attempt to modify anything - read-only protection will invalidate any such query.

Do not guess table or column names — introspect the schema first: `SELECT name, sql FROM sqlite_master WHERE type = 'table'` lists tables with their DDL, and `PRAGMA table_info(<table>)` lists a table's columns. Guessed names are the most common cause of failed calls here.

Validator quirks (fail-closed, by design):
- The validator tokenizes on whitespace and SQL punctuation and checks whole words case-insensitively. A mutation keyword inside a string literal or comment (e.g. `SELECT 'DROP'` or `-- UPDATE`) is REJECTED as a false positive. Write queries so that no mutation keyword appears in a literal or comment.
- `BEGIN`/`COMMIT`/`ROLLBACK` and transaction-control statements are rejected.
- `PRAGMA` is only allowed for `quick_check`, `integrity_check`, `table_info`, `table_xinfo`, `index_info`, `index_list`, `index_xinfo`, `foreign_key_check`, and similar read-only inspect pragmas.

Output:
- Header line is the pipe-delimited column names, followed by one pipe-delimited row per line.
- NULL → empty, Blob → lowercase hex, Text → verbatim (pipes/newlines not escaped).
- Results are capped at 10,000 rows; if the cap is hit, a `truncated` sentinel row is appended.
- Large result sets are spilled to a file: the inline output shows a preview plus a `[view with: read <path>]` reference. Read that file to consume the full result.
