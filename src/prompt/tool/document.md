Create and edit documents. Every action is data — a fixed operation over the values you pass — and every path you pass is confined to your workspace.

**Actions**

- `create` — write a new file. `format` is `docx`, `xlsx`, `pptx` or `pdf`. For docx/pptx/pdf pass `content`; for xlsx pass `sheets`, because a created spreadsheet is its sheets of cells — images belong in a document, a presentation or a PDF.
- `fill_template` — copy a user's own sample (`template`, a .docx/.pptx/.xlsx inside the workspace) and replace its `{name}` placeholders with `values` (an object keyed by placeholder name; each value is text, a number or a boolean). The sample itself is never modified, only copied, and the copy keeps the sample's own extension.
- `pdf_merge` — merge 2+ workspace PDFs (`files`) into one, in order; at most {{merge_inputs}} files, and at most {{merge_mb}} MB in total, in one call.
- `pdf_split` — split one PDF (`path`) into one file per range in `ranges`, e.g. `["1-3", "5"]`; at most {{split_parts}} ranges in one call.
- `pdf_rotate` — add `degrees` to `pages`: a multiple of `{{degrees_step}}`, taken modulo 360, and a whole turn is refused because it would change nothing.
- `pdf_text` — draw `text` on `pages`. Optional `x`, `y` (points), `size`, `color` (`#` plus `{{color_digits}}` hex digits), `stamp` (draw a box) and `rotate`.
- `pdf_image` — place `image` (a workspace PNG or JPEG) on `pages`. Optional `x`, `y`, `width`, `height` (points); giving one of `width`/`height` keeps the image's proportions.
- `pdf_form_fill` — fill the PDF's form fields with `values` (each value is text, a number or a boolean); `flatten` bakes them into the page content.

`pages` is `"all"` or a list like `"1-3,5"`; it defaults to all pages. `file_name` is optional and only a base name — the tool appends the extension (on `fill_template`, the sample's own), sanitizes it to one path component, makes it unique, and drops a trailing extension of a kind it produces rather than doubling it. Output always lands in the workspace `generated/` directory.

**Content blocks** (docx/pptx/pdf), one object each:

- `{"type": "heading", "level": 1, "text": "…"}` — level {{heading_levels}}.
- `{"type": "paragraph", "text": "…"}`
- `{"type": "list", "items": ["…"], "ordered": true}`
- `{"type": "table", "headers": ["…"], "rows": [["…"]]}` — either may be given on its own and the table needs at least one non-empty one; every bullet and every cell is text, a number or a boolean, and an empty header list or an empty row is treated as absent rather than laid out.
- `{"type": "image", "path": "…", "width": 300, "height": 200}` — width/height in px, optional, {{image_side_px}}; giving one of them keeps the image's proportions, and with neither the image keeps its own size capped to a width the page's text has room for.
- `{"type": "notes", "text": "…"}` — a presentation's speaker notes (pptx only).

**Sheets** (xlsx): `[{"name": "Sheet1", "rows": [["Cell", 42, true, {"formula": "SUM(A1:A2)"}]]}]`. A sheet's `name` is optional — without one it becomes `Sheet<n>` by position — and must be at most {{sheet_name_max}} characters, hold none of {{sheet_name_forbidden}}, and be unique ignoring case. A cell is a string, a number, a boolean, or `{"formula": "SUM(A1:A2)"}`. Text is always text — a string that starts with `=` stays a string. A formula is written as a formula and is never evaluated here; the application that opens the file computes it.

Embedded images must be PNG or JPEG — the file's whole content, not just its name. A size you pass (`size`, `width`, `height`) must be {{pdf_size_points}} points, and a position (`x`, `y`) must be within ±{{pdf_point_abs_max}} points. Paths (`template`, `path`, `files`, `image`, and image blocks) must be inside the workspace, and an input must be under {{input_mb}} MB.

**Limits.** A PDF is laid out as a simple stream — the blocks follow one another down the page, and columns, floating objects and exact page placement are not attempted. Marks, stamps and images drawn onto an existing PDF become page content rather than annotation objects, and saving such a file drops tagging and some service metadata; text inside a finished PDF is never reflowed to make room for what you draw. A table and a presentation sample keep everything they already hold — filling only substitutes placeholders — and a substitute value never becomes a formula: in a text document or a presentation it is written as text, and in a spreadsheet a cell that holds exactly one placeholder takes a number as a number.

Each call produces at most the files its one action names; `pdf_split` produces several. The produced file is attached to your reply as a file — mention it in your answer (its `[FILE:…]` marker is what delivers it), or it may be swept away.