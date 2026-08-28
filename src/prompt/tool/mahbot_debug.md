Run a read-only SQL query against mahbot's live databases and receive the result as pipe-delimited text. This runs in-process against the daemon's own connections — it does NOT spawn a subprocess and does NOT open a second instance of any store.

Parameters:
- `query` (required): the read-only SQL statement. Only a single statement is supported — multi-statement queries (e.g. two `SELECT`s separated by `;`) are rejected by the engine.
- `db` (optional): the target database — one of the logical domain store names, `core` (the consolidated store, the default), or `logs`. The accepted values are listed in the tool schema (derived from the runtime validator, so they can't drift).

Read-only guarantee:
- Mutating statements are rejected before execution: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `BEGIN`, `COMMIT`, `ROLLBACK`, `VACUUM`, `REINDEX`, `GRANT`, `REVOKE`, `ATTACH`, `DETACH`, `ANALYZE`, and any PRAGMA not on the read-only allowlist.
- The connection is additionally opened with `PRAGMA query_only=1` for the duration of the query, so a statement that slips past the validator still cannot write.

Validator quirks (fail-closed, by design):
- The validator tokenizes on whitespace and SQL punctuation and checks whole words case-insensitively. A mutation keyword inside a string literal or comment (e.g. `SELECT 'DROP'` or `-- UPDATE`) is REJECTED as a false positive. Write queries so that no mutation keyword appears in a literal or comment.
- `BEGIN`/`COMMIT`/`ROLLBACK` and transaction-control statements are rejected.
- `PRAGMA` is only allowed for `quick_check`, `integrity_check`, `table_info`, `table_xinfo`, `index_info`, `index_list`, `index_xinfo`, `foreign_key_check`, and similar read-only inspect pragmas.

Output:
- Header line is the pipe-delimited column names, followed by one pipe-delimited row per line.
- NULL → empty, Blob → lowercase hex, Text → verbatim (pipes/newlines not escaped).
- Results are capped at 10,000 rows; if the cap is hit, a `truncated` sentinel row is appended.
- Large result sets are spilled to a file: the inline output shows a preview plus a `[view with: read <path>]` reference. Read that file to consume the full result.

Data surface: the tool can read every table in the target database, including `config_kv` (which holds provider keys), `users`, and `user_channels`. Standard credential redaction is applied to the result (both the inline portion and any spill file), but it only catches `key=value` / `key: value` patterns — pipe-delimited values are not redacted, so results may still contain secrets. Treat them as sensitive in any deliverable.
