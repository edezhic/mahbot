Search workspace files. `mode` picks file-name search (`files`) or content search (`grep`, the default); `grep_mode` picks the matching strategy within grep mode (`fuzzy`, `plain_text` or `regex`) and is ignored in files mode.

`fuzzy` matches approximately — use it for typos, partial names, or when unsure of spelling. `plain_text` matches a literal substring — use it for exact text like identifiers and error strings. `regex` is a full regular expression; lookaround (`(?=…)`, `(?!…)`, `(?<=…)`, `(?<!…)`) is refused, so reformulate without it.

A query may carry inline constraints (`*.rs`, `src/`) or use the `path` and `ext` parameters. A result with more pages prints a `── pagination ──` section with `next_file_offset` — pass it back as `offset`; 0 means no more files. A zero-result answer prints a `── diagnostics ──` block naming the cause and the fix, and an unknown parameter is reported in a `── note ──` block with the valid names.
