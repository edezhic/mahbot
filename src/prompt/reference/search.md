# search — reference

## Arguments

- `mode` — `files` for file-name search or `grep` for content search (default).
- `query` — the search text. For `files`, fuzzy text like `lib.rs`; for `grep`, the pattern.
- `grep_mode` — the match within grep mode: `fuzzy` (default), `plain_text` or `regex`. Ignored in `files` mode.
- `case_sensitive` — grep only, default true. False is smart-case: insensitive while the pattern is all lowercase, sensitive otherwise. Ignored in `files` mode.
- `max_results` — maximum results, default 50, max 500.
- `offset` — pagination offset, default 0. In `files` mode it skips items; in `grep` mode it skips files, not matches — pass `next_file_offset` back.
- `context_lines` — lines of context before and after each match, grep only, default 0.
- `path` — a path-segment filter; a trailing slash is appended, so `src` becomes `src/`. Additive with inline `src/`.
- `ext` — a file-extension filter; a glob is appended, so `rs` becomes `*.rs`. Additive with inline `*.rs`.

## Matching

- `fuzzy` matches approximately — typos, partial names, unsure spelling.
- `plain_text` matches a literal substring — exact identifiers, error strings.
- `regex` is a full regular expression. Lookaround (`(?=…)`, `(?!…)`, `(?<=…)`, `(?<!…)`) is refused; reformulate without it.

## Results

- A result with more pages prints a `── pagination ──` section; `0` for `next_file_offset` means no more files.
- A zero-result answer prints a `── diagnostics ──` block naming the cause and the fix.
- An unknown parameter is reported in a `── note ──` block with the valid names.
