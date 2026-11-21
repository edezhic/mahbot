Search workspace files. `mode` picks file-name search (`files`) or content search (`grep`, default); `grep_mode` picks the match within grep mode — `fuzzy` (default), `plain_text` or `regex` — and is ignored in files mode.

A query may carry inline constraints (`*.rs`, `src/`), combined with `path` and `ext`. Results paginate: pass a `next_file_offset` back as `offset`.
