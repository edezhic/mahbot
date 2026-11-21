# rasterize — reference

Turn a file or a page into an ordinary picture. The input is rendered and saved as a PNG in the workspace's `generated/` folder, and the picture is attached to the conversation as a native image in the same step; the output carries an `[IMAGE:path]` marker — name it in your reply to send the picture to the user.

## Inputs

- `path` — what to render:
  - SVG markup, plain or gzip-compressed.
  - One page of a PDF, named with `page`.
  - A local HTML file inside your personal workspace, rendered as a page.
  - An http(s) address, rendered as a page; `file://` and every other scheme are refused.
- `page` — the 1-based page to render, for a PDF of several pages; naming a page for any other input is refused.

The picture keeps the source's aspect ratio, its long side capped at 2048 px.
