# document — reference

`document` is one tool over thirteen actions. Every call names one `action` and that action's own arguments; `file_name` and `metadata` are shared by all of them. Every `path` resolves in your access frame.

## Shared arguments

- `action` — one of `create`, `fill_template`, `docx_edit`, `xlsx_edit`, `pptx_edit`, `pdf_merge`, `pdf_split`, `pdf_rotate`, `pdf_crop`, `pdf_resize`, `pdf_text`, `pdf_image`, `pdf_form_fill`.
- `file_name` — at most 80 characters; the tool appends the extension and makes the name unique.
- `metadata` — a PDF's own description, taken by every PDF-writing action. Fields: `title`, `author`, `subject`, `keywords` (the single text the file carries), `creator`, `producer`, `created`, `modified`. A field left out stays as the file has it, except an existing `modified`, which advances to the write; an empty string clears a field, and no entry is invented for a file that carries none. A date is `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM[:SS]`, with an optional `Z` or `±HH:MM` offset (no offset means UTC). A file the tool builds names `mahbot` as its producer unless `producer` says otherwise.

## Pages

Every PDF action but `pdf_merge` and `pdf_form_fill` takes `pages`: an array whose entries are `"all"`, a page (`"3"`), a range (`"2-5"`) or a comma list (`"1-3,7"`); `"all"` stands alone, and absent means every page. At most 10000 pages per request. `pdf_split` makes one output file per entry, in order — `["1-3", "5"]` is two files, `["1-3,5"]` one file holding those four pages — and takes at most 50 entries.

## create

`create` writes a new file; `format` is `docx`, `xlsx`, `pptx` or `pdf`.

- `content` — the docx/pptx/pdf blocks, in order. Each block has a `type`: `heading` (`text`, `level` 1, 2 or 3), `paragraph` (`text`, `footnote`), `notes` (pptx speaker notes; `text`), `list` (`items`, `ordered`), `table` (`headers`, `rows`), `image` (`image`, `width`, `height`), `link` (`text`, `url`), `page_break`, `contents` (a table of contents over the headings, with page numbers already written in; one per document). A table needs a `headers` row or `rows`; a .docx table holds at most 63 columns. `link`, `page_break`, `footnote` and `contents` are a .docx's own. A block's `width` and `height` are points (greater than 0 and at most 20000); one side alone keeps the proportions, and with neither the picture keeps its own size, fitted whole into the page's or slide's text column. An `image` is PNG, JPEG or WebP; a PDF takes PNG or JPEG only. A `url` is an http, https, mailto or tel address.
- `sheets` — the xlsx sheets, in order. Each has `name` (at most 31 characters, none of `[]:*?/\`, unique ignoring case; unnamed becomes `Sheet<n>`), `rows` (top to bottom, each a row of `value`s) and `merges` (ranges like `A1:C1`, corners in either order, each covering at least two cells). A `value` is text, a number, a boolean, or a `formula` object (`{"formula": "SUM(A1:A2)"}`); a string starting with `=` stays text, and a number is written as the cell stores it, not as a reading displayed it.
- `size` — a pptx deck: `widescreen` (13.333 × 7.5 in), `standard` (10 × 7.5 in) or `portrait` (7.5 × 10 in). Unnamed is the 10 × 5.625 in deck this tool has always written.
- `header`, `footer` — a running header or footer, one paragraph per line; `{page}` is the page number and `{pages}` the document's page count.

A PDF is laid out as a simple stream: blocks follow one another down the page, and columns, floating objects and exact page placement are not attempted. A .docx may carry a running header and footer, page numbers, a table of contents, links, footnotes and explicit page breaks. A `create` PDF whose content is pictures only gives each its own page, fitted whole, with no empty page before the first.

## fill_template

- `path` — a `.docx`/`.pptx`/`.xlsx` sample.
- `values` — one entry per placeholder, keyed by its name; a value is text, a number or a boolean.

A Word or PowerPoint sample is filled at its `{name}` placeholders; a spreadsheet is filled cell by cell, where a cell holding exactly one placeholder takes a number as a number. Elsewhere a substitute is written as text and never becomes a formula. A placeholder you give no value is left as it was and reported back.

## docx_edit

Edits a `.docx`/`.docm`; the input must be a `.docx` package. `edits` is an array (at most 200), applied whole or not at all: every part it does not name is carried unchanged, and a signature the file carried is invalidated. `op` selects the operation; `bold`, `italic`, `underline` and `strike` are toggles — true sets the property, false switches it off.

- `replace_text` — `find`, `replace`: every occurrence in every matching paragraph.
- `insert_text` — `find`, `insert`, `position` (`before` or `after`).
- `remove_text` — `find`.
- `remove_paragraph` — `find`: removes every paragraph holding it.
- `format_text` — `find` plus at least one of `bold`, `italic`, `underline`, `strike`, `size` (between 0.5 and 819 points), `color` (6 hex digits).
- `format_paragraph` — `find` plus at least one of `align` (`left`, `center`, `right`, `justify`), `style` (a style the document defines, at most 64 characters; Normal, Heading 1, Heading 2 and Heading 3 always work), `indent_left` and `indent_right` (between -2000 and 2000 points), `indent_first_line` (the same bound; a negative one hangs), `spacing_before` and `spacing_after` (between 0 and 2000 points), `line_spacing` (between 0.05 and 2000 points), `list` (`bullet`, `number`, or false to take the paragraph's list away), `level` (0 to 8).
- `add_paragraph` — `text`, `after`.
- `add_image` — `after`, `image` (PNG, JPEG or WebP), `width`, `height` (points; one side alone keeps the proportions; with neither the picture is fitted whole into the page's text column).
- `add_table` — `after`, `headers`, `rows`; at most 63 columns, headings included.

Every occurrence in every matching paragraph is rewritten. A text box's own text and a table cell's are addressed; a header's, footer's, footnote's, endnote's, comment's, a field's cached result and a table of contents' own entries are not, and a fragment only they hold is refused as that part's text. A heading a table of contents lists keeps its text and level: an edit changing either is refused with the table named, while formatting it is not. `add_image` and `add_table` land after the first paragraph holding the `after` text — a table cell's anchor means after the whole table — and a fragment the body does not hold is refused rather than appended at the end. Formatting a fragment formats its whole run.

## xlsx_edit

Edits an `.xlsx`/`.xlsm`; the input must be an `.xlsx` package. `edits` is an array (at most 200). An edit names one `sheet` (as a reading prints it) unless it adds one.

- `set_cell` — `sheet`, `cell` (A1, like B7), `value`, `number_format` (an Excel format code, at most 200 characters). Writes a value and leaves the cell's formatting alone.
- `clear_cell` — `sheet`, `cell`. Empties a cell and keeps its formatting.
- `format_cells` — `sheet`, one target (`range` like B2:D2 with corners in either order and at most 10000 cells, `column` up to the 16384th, or `row` up to 1048576) and at least one property: `font` (at most 31 characters), `size` (between 1 and 409 points), `bold`, `italic`, `underline`, `strike`, `color` (6 hex digits), `fill` (6 hex digits, or false to take the fill away), `border` (any of `top`, `bottom`, `left`, `right`, each a boolean, at least one), `align` (`general`, `left`, `center`, `right`, `justify`, `distributed`, `centerContinuous`), `vertical` (`top`, `center`, `bottom`, `justify`, `distributed`), `wrap`, `number_format`, `width_chars` (between 0 and 255 characters), `height_points` (between 0 and 409 points). The one op that changes how cells appear without touching what they hold; a cell keeps the look its row and column give it.
- `insert_row`, `delete_row` — `sheet`, `row`. `insert_column`, `delete_column` — `sheet`, `column`. An inserted row or column may be one past the sheet's used range; a deleted one must be inside it.
- `merge_cells`, `unmerge_cells` — `sheet`, `range`. A merge never clears the covered cells' content; `unmerge_cells` names a range that must match a merge; a merge overlapping one already there is refused naming that merge, and a range of one cell is refused as too small to merge.
- `add_sheet` — `name`; `rename_sheet` — `sheet`, `name`; `delete_sheet` — `sheet`; `move_sheet` — `sheet`, `to` (the 1-based position it takes). A `name` is at most 31 characters, none of `[]:*?/\`, unique ignoring case. The workbook always keeps at least one visible sheet.
- `add_image` — `sheet`, `cell`, `image` (PNG, JPEG or WebP), `width`, `height` (points; proportions kept; with no side named, its own size). Anchors the picture's top-left corner in the named cell; it only places pictures in an existing workbook (`create` writes none), and a picture placed this way is not moved by a later row or column insert or delete.

A rename or removal does not rewrite formulas, defined names, chart series or links — they keep their text, and the answer names what it left pointing at a name that is gone.

## pptx_edit

Edits a `.pptx`/`.pptm`. `edits` is an array (at most 200); `slide` is its 1-based number as a reading prints it. `bold`, `italic`, `underline` and `strike` are toggles.

- `replace_text` — `slide`, `find`, `replace`.
- `remove_text` — `slide`, `find`.
- `format_text` — `slide`, `find` plus at least one of `bold`, `italic`, `underline`, `strike`, `size` (between 1 and 4000 points), `color` (6 hex digits), `align` (`left`, `center`, `right`, `justify`).
- `add_paragraph` — `slide`, `text`, `after`, `level` (0 to 8).
- `remove_paragraph` — `slide`, `find`.
- `add_image` — `slide`, `image` (PNG, JPEG or WebP), `x_fraction`, `y_fraction`, `width_fraction`, `height_fraction`.
- `add_table` — `slide`, `headers`, `rows`, `x_fraction`, `y_fraction`, `width_fraction`, `height_fraction`. A presentation's table has no column bound, and a headings row alone makes a table.
- `add_slide` — `after`, `title`, `bullets` (at most 100).
- `delete_slide` — `slide`; `move_slide` — `slide`, `to`; `duplicate_slide` — `slide`.
- `replace_notes` — `slide`, `find`, `replace`; `remove_notes` — `slide`, `find`; `add_notes` — `slide`, `text`.

`x_fraction` and `y_fraction` are between 0 and 1 of the slide's width and height; `width_fraction` and `height_fraction` are greater than 0 and at most 1. An unnamed place is the slide's middle, an unnamed size a picture's largest fit or a table's own, and a place named without a size can push either past the slide's edge. `add_table` places the table after what is already there, shrunk only as far as the slide forces, never grown. A text op names a fragment that must be on the slide — whitespace alone matches nothing. `create` lays out its content by the deck's own proportions. A diagram's text and a comment's text are not editable; `replace_notes`, `remove_notes` and `add_notes` address a slide's speaker notes.

## pdf_merge

- `paths` — the PDFs to merge, in order: at least 2, at most 50, at most 200 MB in total.
- `metadata` — the new file's description.

Carries only the pages: a document's description, bookmarks and named destinations, tags, page labels, embedded files and form fields — a merged form is no longer fillable — are not carried, while page content, rotation, annotations and the visible area are. A carried link or annotation may name a page the new file does not have.

## pdf_split

- `path`, `pages`, `metadata`.

One output file per page entry, in order. Only the pages are carried, as `pdf_merge` states.

## pdf_rotate

- `path`, `degrees` — a multiple of 90, not a whole turn.
- `pages`, `metadata`.

Adds a rotation to the pages it names and draws nothing.

## pdf_crop

- `path`, `area` — an object with `x`, `y`, `width`, `height`, each between 0 and 1 of what the page currently shows, measured from the top-left as the page reads (rotation included) and reaching down and right.
- `pages`, `metadata`.

Narrows the visible area to `area`: it changes only what is visible — nothing is removed — and an area outside the page or with no width or height is refused.

## pdf_resize

- `path`, `width`, `height` — both required, in points, greater than 0 and at most 20000.
- `pages`, `metadata`.

The content is fitted uniformly, never stretched; any spare strip is left blank, the page afterwards reads as the named size, and links and form widgets move with the content — an annotation that cannot follow is reported.

## pdf_text

- `path`, `text` — the text to draw.
- `pages`, `x`, `y` — position in points, within ±20000, from the page's left and bottom edge as it reads.
- `size` — font size in points, greater than 0 and at most 20000.
- `color` — 6 hex digits.
- `stamp` — draw a box around the text.
- `opacity` — greater than 0 and at most 1, applied to the text and its stamp box; unstated is fully opaque.
- `rotate` — degrees, within ±20000.
- `metadata`.

What it draws becomes page content rather than an annotation object, and existing text is never reflowed to make room. Anything landing outside the page's visible area is drawn as asked, and the answer carries a note naming the page. A picture is turned with the page, so it reads upright on a page displayed at a rotation; drawn text and its stamp are not — they follow only their own `rotate` and read sideways on such a page.

## pdf_image

- `path`, `image` — PNG or JPEG only; a PDF takes no WebP.
- `pages`, `x`, `y` — position in points, within ±20000, from the page's left and bottom edge as it reads.
- `width`, `height` — placed size in points, greater than 0 and at most 20000; one side alone keeps the proportions, and with neither named the picture keeps its own size.
- `metadata`.

What it draws becomes page content rather than an annotation object.

## pdf_form_fill

- `path` — the PDF.
- `values` — one entry per form field, keyed by its name; a value is text, a number or a boolean.
- `flatten` — bake the fields into the page content and remove them.
- `metadata`.

A PDF's form fields are separate from its page content; a placeholder you give no value is left as it was and reported back.

## Refusals

A call is refused before a file is written for an argument the action does not take, an edit whose fields do not fit its `op`, a value outside its stated bound, and a fragment found nowhere. A text op matches inside one paragraph of the document's own body text and never across paragraphs; a fragment or replacement is at most 20000 characters. A PDF's existing text is not addressable — there is no PDF find-and-replace.

## Reading marks

A Word reading prints marks and labels around the text; they are the reader's, not the file's. They are not text an edit addresses, and a fragment holding only one is refused for what it is. A tab and a line break a reading shows are markup (`<w:tab/>`, `<w:br/>`): a fragment holding one is refused, and a rewrite may not match across one.

Bracketed marks: `[text box]`, `[ins]`, `[/ins]`, `[del]`, `[/del]`, `[moved-here]`, `[/moved-here]`, `[moved-away]`, `[/moved-away]`, `[ins ¶]`, `[del ¶]`, `[moved-here ¶]`, `[moved-away ¶]`, `[fmt]`, `[fmt ¶]`, `[fmt section]`, `[heading 1]`, `[field PAGE \* MERGEFORMAT]`, `[comment 1 starts]`.

Labels and block marks: `Table 1 (formatting revised):`, `C2:`, `(spans C2-C4)`, `(part of R3C2)`, `(merge revised)`, `(link: internal)`, `Footnote 1 (not referenced in the text):`, `Header 1 (first page):`, `Unaccepted tracked changes: 1 insertion`, `(authors: Anna)`.

A chart's content and a spreadsheet's cell marks are the reading's too, listed under `read`'s reference; no edit addresses them.

A presentation reading prints `Slide <n>:`, `Slide <n> notes:` and `Slide <n> comments:`, and marks a slide `(title)`, `(hidden slide)`, `(diagram text)`, `(diagram text could not be read)` or `(no text)`. Those are that printing, not text on the slide, not a diagram's text (which no edit changes) and not a comment's text (which no edit addresses).
