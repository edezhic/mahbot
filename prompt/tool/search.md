Search workspace files. Two modes — file name search or content grep.

**Important:** `mode` and `grep_mode` are SEPARATE parameters.
- `mode` selects the top-level operation: files or grep.
- `grep_mode` selects the matching strategy within grep mode (plain_text, regex, or fuzzy).
They are not interchangeable — `grep_mode` is ignored when `mode="files"`.

---

## mode="files" — Fuzzy file name search

Find files by their path/name. Uses frecency-ranked fuzzy matching (replaces glob).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query`   | required | Fuzzy text to match against file paths. Supports inline constraints like `*.rs` (extension) and `src/` (path segment). |
| `path`    | optional | Path segment filter — appends a trailing-slash constraint (e.g. `"src"` → `src/`). Use this or inline `src/` syntax in query. |
| `ext`     | optional | File extension filter — appends a glob constraint (e.g. `"rs"` → `*.rs`). Use this or inline `*.rs` syntax in query. |
| `max_results` | 50 (max 500) | Max files to return |
| `offset`  | 0 | Pagination offset (0-based). In 'files' mode: skips this many items. In 'grep' mode: skips this many **files** (not matches). Use `next_file_offset` from grep results for correct pagination. |

Parameters `grep_mode`, `case_sensitive`, and `context_lines` are IGNORED in this mode.

## mode="grep" — File content search

Search file contents (replaces rg). Uses an indexed engine with early termination.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query`   | required | Pattern to find in file contents. Supports inline constraints like `*.rs` (extension) and `src/` (path segment). |
| `path`    | optional | Path segment filter — appends a trailing-slash constraint (e.g. `"src"` → `src/`). Use this or inline `src/` syntax in query. |
| `ext`     | optional | File extension filter — appends a glob constraint (e.g. `"rs"` → `*.rs`). Use this or inline `*.rs` syntax in query. |
| `grep_mode` | `fuzzy` | How to interpret the pattern — see grep modes below |
| `case_sensitive` | `true` | When `true`: case-sensitive. When `false`: smart-case (case-insensitive if pattern is all lowercase) |
| `context_lines` | 0 | Lines of context before and after each match |
| `max_results` | 50 (max 500) | Max matches to return |
| `offset`  | 0 | **File-based** pagination offset (0-based). Skips this many **files** (not individual matches). ⚠ Different from 'files' mode where offset is item-based. Use `next_file_offset` from the response for correct pagination — do NOT simply increment by max_results. |

**Pagination protocol:** When results include a `── pagination ──` section with `next_file_offset: <N>`, pass `"offset": <N>` in your next call with the same query to fetch the next page. A `next_file_offset` of `0` means there are no more files to search.

**Non-empty results** also show `total_matched` (files mode) or `next_file_offset` (grep mode) as pagination hints, so you always know how to continue.

### Grep modes

- **`fuzzy`** — approximate/similar matching, the recommended starting mode for most searches. Use for typos, partial names, or when you're not sure of the exact spelling.
- **`plain_text`** — literal substring search. Use this when you need to find exact text like specific variable names, error strings, or identifiers.
- **`regex`** — full regex pattern matching. Use for complex patterns requiring alternation, anchors, character classes, groups, etc. **Note:** lookahead and lookbehind assertions (`(?=...)`, `(?!...)`, `(?<=...)`, `(?<!...)`) are not supported and will return an error. Patterns with special regex characters (`.`, `*`, `+`, `?`, `[`, `]`, `(`, `)`, `{`, `}`, `^`, `$`, `|`, `\`) may behave unexpectedly as plain_text. **Warning:** invalid regex patterns silently fall back to plain-text matching with an error note in results.

---

## Diagnostic signals (zero results)

When zero results are returned, the tool includes a `── diagnostics ──` block to help you distinguish causes and self-correct. Pay attention to these signals:

### Files mode diagnostics

| Field | Meaning |
|--------|---------|
| `total_files` | Total files in the workspace index |
| `total_matched` | Files matched before pagination (offset/limit applied) |
| `constraints` | Parsed constraints from your query (e.g. `*.rs, src/`) |
| `offset`, `limit` | Pagination values that were applied |

**Common patterns:**

- `total_matched: 0` with constraints present → constraints filtered everything. Try removing or broadening constraints.
- `total_matched > 0` but `offset` >= `total_matched` → you paginated past the end. Reset `offset` to 0.
- `total_files: 0` → workspace index is empty (scanning may not be done yet).

### Grep mode diagnostics

| Field | Meaning |
|--------|---------|
| `total_files` | Total files in the workspace index |
| `total_files_searched` | How many files were actually searched this call |
| `filtered_file_count` | Files excluded by filters (binary, too-large, etc.) |
| `files_with_matches` | Files that contained at least one match |
| `constraints` | Parsed constraints from your query |
| `offset`, `limit` | Pagination values applied |
| `next_file_offset` | Pass as `offset` for the next page (0 = no more files to search) |

**Common patterns:**

- `filtered_file_count: 0` with `total_files > 0` → all files were filtered out. Check your constraints or file size limits.
- `files_with_matches > 0` but `offset` >= `files_with_matches` → you paginated past the end. Reset `offset` to 0.
- `files_with_matches: 0` with `next_file_offset > 0` → offset landed in a gap between matching files. Try `offset=N` with the `next_file_offset` value.
- `files_with_matches: 0` with `next_file_offset: 0` and no constraints → the pattern genuinely doesn't exist. Try fuzzy mode or a different query.
- `files_with_matches: 0` with `next_file_offset: 0` and constraints → try removing constraints to check if the pattern exists at all.

### Unknown parameter warnings

If you see a `── note ──` block about "Unrecognized parameter(s)", you used parameter names the tool doesn't recognize. Check the schema above for valid names. Common mistakes: using `file_pattern` instead of `query`, `filename` instead of `query`, or `search_term` instead of `query`. Valid parameters: `mode, query, grep_mode, case_sensitive, max_results, offset, context_lines`. Aliases `pattern` (→ query), `path` (→ path: constraint), and `ext` (→ ext: constraint) are also accepted silently.