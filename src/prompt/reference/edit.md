# edit — reference

## Arguments

- `path` — the file. Relative paths resolve from the workspace. A database that is not the product's own — yours, another product's, a dump — is an ordinary file.
- `old_string` — omitted or empty creates a new file (refused if it exists). Non-empty: the text to replace.
- `new_string` — the content to write when creating, or the replacement text when editing (empty deletes the match).
- `multiple` — replace every occurrence of `old_string` instead of requiring exactly one. Default false.

## Editing

- `old_string` must appear exactly once unless `multiple` is true.
- Line endings need not match: LF and CRLF are interchangeable for every file type. `new_string`'s line endings are rewritten to the file's own around the match.
- In code files the match is semi-insensitive to whitespace.
- `new_string` must differ from `old_string`; an identical pair is refused as a no-op.
- A `path` that does not exist is created when `old_string` is absent or empty.

## Result

`Written <n> bytes to <path>` for a new file, `Edited <path>: replaced <n> occurrence(s) (<bytes> bytes)` for an edit.

