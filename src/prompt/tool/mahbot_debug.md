This tool is designed for cases when you need to debug the system(harness) you're(the agent) operating in - inspect other agents, tickets, configs, logs etc. It's irrelevant for most tasks, but can be useful in some rare cases when the past or ongoing work of other agents requires inspection. Runs a read-only SQL query against mahbot's live databases and returns the result as pipe-delimited text. 

Pass the read-only `query` and, optionally, a `db` selector naming the target database — the consolidated `core` store by default, or `logs`. The accepted `db` values are listed in the tool's schema. Pass a single statement: if multiple statements are separated by `;`, the engine runs only the first and silently ignores the rest — the read-only validator below is the real guard against mutation, not the engine. Do not even attempt to modify anything - read-only protection will invalidate any such query.

Do not guess table or column names — introspect the schema first: `SELECT name, sql FROM sqlite_master WHERE type = 'table'` lists tables with their DDL, and `PRAGMA table_info(<table>)` lists a table's columns. Guessed names are the most common cause of failed calls here.

Validator quirks (fail-closed, by design):
- The validator strips string literals, `--`/`/* */` comments, and quoted identifiers (`"x"`, `[x]`, `` `x` ``) before keyword matching, so a mutation keyword inside them (e.g. `WHERE tool_name = 'analyze'`) is ignored rather than rejected. Any blocklist keyword surviving in SQL position anywhere in the query is still rejected.
- `BEGIN`/`COMMIT`/`ROLLBACK` and transaction-control statements are rejected.
- `PRAGMA` is only allowed for `quick_check`, `integrity_check`, `table_info`, `table_xinfo`, `index_info`, `index_list`, `index_xinfo`, `foreign_key_check`, and similar read-only inspect pragmas.

Output:
- Header line is the pipe-delimited column names, followed by one pipe-delimited row per line.
- NULL → empty, Blob → lowercase hex, Text → verbatim (pipes/newlines not escaped).
- Results are capped at 10,000 rows; if the cap is hit, a `truncated` sentinel row is appended.
- Large result sets are spilled to a file: the inline output shows a preview plus a `[view with: read <path>]` reference. Read that file to consume the full result.
