Read a file or directory; prefer it over shell `cat`. Text comes back line-numbered, with `offset`/`limit` selecting a range. Content is cut to a ~5 KB budget, so read big files in slices.

`mode` selects what is read: `content` (default) reads the file itself — images and video timelines arrive as attachments, document containers are converted to text; `symbols` lists a code file's top-level AST symbols with line ranges, the quickest way to map a large file; `zoom` returns one symbol's full source, named by `symbol`. `symbols` and `zoom` need a supported code format (Rust, JS/TS, Python, Go, C, Ruby, SQL, Markdown, JSON, TOML, CSS, HTML, shell).

Supported inputs: text and code, raster images (PNG, JPEG, WebP), video and audio recordings, PDF, Word (`.doc`/`.docx`/`.docm`), Excel (`.xls`/`.xlsx`/`.xlsm`), PowerPoint (`.ppt`/`.pptx`/`.pptm`). Anything else is read as text with lossy UTF-8.

A directory lists its contents instead; `mode`, `offset` and `limit` do not apply to it.

{{path_policy}}
