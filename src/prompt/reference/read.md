# read — reference

## Arguments

- `path` — the file or directory. Relative paths resolve from the workspace; `~`/`~/…` mean your home directory where the access level reaches it.
- `mode` — `content` (default), `symbols` or `zoom`.
- `symbol` — the symbol to extract with `mode: "zoom"`.
- `offset` — first line (1-based) of a text read, or the start second of a video or recording. With `limit` omitted, a video or recording is read as an overview of the whole clip (the first ten minutes of speech).
- `limit` — maximum lines of a text read, or seconds of a video or recording (to the end when omitted).

`offset`/`limit` are not used for a converted document. `mode`, `offset` and `limit` are ignored for a directory, which lists its contents instead.

## Text

- Line-numbered. Content is truncated to a ~5 KB budget; a media timeline is not cut.
- Binary files are read with lossy UTF-8, except document containers and a background shell session's own output file (read as a program's output).
- A misspelled path that matches one file closely is recovered, with a `[Recovered path: …]` note; several matches are listed under `Did you mean:`.

## Images

A content-mode raster (PNG, JPEG, WebP) is attached as a native image; a second read references the already-attached image. The attachment note gives the file's own pixel size and, when the copy handed to the model is smaller, that copy's size separately — the file's size is what the user receives. GIF, BMP and other recognised-but-unsupported formats are reported, not decoded.

## Video and recordings

A content-mode video is a timeline: frames with the second each marks, speech with its marks and text, and a sound label where there is no speech. Frames follow as pictures of up to four moments each, left to right then the next row, each annotation naming the frames it holds. A recording returns the same timeline without pictures.

## Document containers

PDF, Word `.doc`/`.docx`/`.docm`, Excel `.xls`/`.xlsx`/`.xlsm` and PowerPoint `.ppt`/`.pptx`/`.pptm` are converted, not returned as bytes. An encrypted file reads `password-protected`; an old binary is read as its family whatever its name; text is returned, or its path when long. Files over 10 MB are rejected, except containers, accepted up to 50 MB; a video or a recording is not subject to that gate.

### Charts

A chart is described where it sits (a layout's or master's included): its kind, its title, any text or label, and one series line per series — numbered when unnamed — of category: value, with a bubble's size beside its value and excess counted. A chart with no readable values says so and stays in the report.

### What a reading does not show

Every reading says what it did not show, by kind. A missing image keeps the note saying why; what a reading never looks at is not reported. Pages and images with no text are attached as images, at most 6 per read (beyond that the reading says how many the document holds and how many it left out); a page whose text was read is not drawn; a page with neither text nor images, or that will not convert, says so.

### Old binary Office files

`.doc`/`.xls`/`.ppt` are read only, in their family's shape. Hidden sheets and slides, merged ranges, cell comments, hyperlinks, defined names and formulas are never read. A presentation gets no title, hidden-slide, diagram or comment mark. Document stories are labelled blocks (`Headers and footers:`, `Footnotes:`, `Endnotes:`, `Comments:`), with no comment author.

### Excel workbooks

`Sheet "<name>":` blocks in workbook order, one line per valued cell (`A1: 42`). A formula with no stored value is shown as its formula; with both, as `42 (=SUM(B1:B2))`. An empty sheet says so; a chart sheet says so. Write back the stored number, not its displayed text; an unreproducible format shows the stored number, marked as such. Hidden rows, columns and sheets are marked, merged ranges where they are; comments follow the sheet's cells, a hyperlink is marked on its cell, defined names follow the last sheet as `Defined names:`. A pivot table's values and a shape's or text box's text are never read.

### Word documents

A table is a `Table N:` block, one line per row under `R1:`, one `C2:` token per cell; a merged cell is marked `(spans C1-C2)` and `(part of R1C1)` on each continuation. Headers and footers follow, one block per part, then footnotes, endnotes and comments, each indented under its heading, a definition nothing refers to saying so, with `[footnote 1]`, `[comment 1]` or `[comment 1 starts]`…`[comment 1 ends]` marking where it belongs. A page number or date field is `[field PAGE]`; a text box's text is one indented block. A heading opens `[heading N]`; a link's content is followed by `(link: ADDRESS)`, `(link: internal)` or nothing when no address is known.

### PowerPoint presentations

Each slide's text under a `Slide <n>:` block, then its comments and replies (each with the author the deck names) and its notes. A declared title, a hidden slide, diagram text and a slide with no text are marked. A slide's layout and theme are not read.

### PDF

Text page by page: a `Page <n>:` block per page with text; a page with none, or unreadable, is attached as an image. Text a page's crop box hides is not its text. A `Description:` block leads when the file has one, naming its entries — `title`, `author`, `subject`, `keywords`, `creator`, `producer`, `created`, `modified` — with dates in calendar form. Marks follow as an `Annotations:` section, one entry per mark as `<kind>[ (<author>)] on page <n>`, text below. A `Form fields:` section lists each field as `name (kind; options: …) [on page <n>]: value`, with a no-value mark when empty; kinds are text, checkbox, radio and choice; options are named for radio/choice (long lists cut), a pushbutton or signature is not listed, an unnamed field and an option-less radio group or list box are marked not fillable, and a form with none says so.

### Tracked changes

Unaccepted tracked changes are never passed off as final text: `[ins]…[/ins]` inserted, `[del]…[/del]` deleted, `[moved-here]…[/moved-here]` and `[moved-away]…[/moved-away]` moved, `[ins ¶]`/`[del ¶]`/`[moved-here ¶]`/`[moved-away ¶]` revising a paragraph mark; a row or cell added or removed says `(inserted)` or `(deleted)`; `[fmt]`, `[fmt ¶]`, `[fmt section]`, `(merge revised)` and `(formatting revised)` revise formatting alone. The document opens with an `Unaccepted tracked changes: …` line counting them and naming their authors.

## Output marks

A reading prints these literal marks to describe the file; they are the reader talking, not text from the file.

- `Chart (<kind>):` — the header a chart's content prints under; a part that names no kind prints `Chart:`.
- `Title:`, `Text:`, `Label:` — a chart's field labels: its title, an axis title, a data label's own text.
- `Series "<name>":` — one line per series; a series the part leaves unnamed prints its 1-based number instead: `Series 1:`, `Series 2:`.
- `(size <size>)` — a bubble point's size, beside its value, e.g. `(size 12)`.
- `3 more value(s)` — the form a bounded list's excess takes: count, noun, `more`.
- `(no values in the chart part)`, `(values only in the attached table)`, `(the chart part could not be read)`, `(an extended chart kind, not read)` — a chart with no readable values says which case it is, where its series would stand.
- `(stored number, format not shown)` — a spreadsheet number under a format this reader cannot reproduce, shown as the file stores it.
- `(chart sheet)` — a sheet that is one chart over a whole tab; it prints first, the chart's content under it.
- `(no values)` — the empty-unit marker.
- `2 chart(s) not shown` — the shape of every "did not show" report line: count, kind, `not shown`.
- `[text box]` — a Word text box's text, one indented block. `(link: ADDRESS)` — what a Word link's own content is followed by, `(link: internal)` for an anchor in the same document.
- `(title)`, `(hidden slide)`, `(diagram text)`, `(diagram text could not be read)`, `(no text)` — a presentation slide's marks: its declared title, a hidden slide, its diagram text, a diagram whose text could not be read, and a slide holding none.
