Edit an existing image deterministically: open one document, apply a chain of the engine's own
commands in order, save the result — or start from a canvas created from nothing and save that. This
is not image generation — it changes pixels you already have. Cropping, resizing, rotating, format
conversion (HEIC included as an input), colour and tone, sharpening, layer work on PSD and TIFF,
folder batches, and inspection.

Actions:
- `commands` — the engine's command registry. Unfiltered it is every command's id, label and menu
  path, and one answer carries only its first part; `filter` reaches the rest and adds each match's
  parameter documentation — names, ranges, defaults and the result shape — as the vendor's JSON, one
  row per match, so a narrowed question takes you from an id to a callable step. A broad filter's
  answer drops whole commands that did not fit and says how many it left out, so narrow the filter
  rather than guessing at a cut-off row. The registry is the authority, and the vendor is alpha with
  no compatibility guarantee for ids, so a step that suddenly fails is usually an id that moved:
  look it up again instead of guessing.
- `inspect` — one file's size, colour mode, depth and layer tree, as JSON, with a `format` field the
  product adds from the file name (the engine's own report carries none). Run it first on a format
  you have not fed the engine before.
- `run` — `input`, an optional `chain`, and `output`. The chain is a JSON array, one object per
  step: `{"command": "filter.sharpen.smartSharpen", "params": {"amount": 80}}`. The steps run in
  order against the one open document, then it is saved; `params` is optional, and an empty chain
  is a plain open-and-convert. With no `input` at all, pass `canvas` instead — the new document's own
  parameters, like `{"width": 800, "height": 600}` — and the chain starts from a canvas created from
  nothing.
- `batch` — the same `chain` applied to every file in an `input` folder, written into an `output`
  folder. With no `chain` it converts every file into the `format` you name, which is how a folder
  of HEIC photographs becomes PNG. `quality` (1-100) sets the lossy quality, and TIFF output keeps
  its layers — naming `tif` or `tiff` in `format` does that on its own, and `tiff_layers` keeps them
  in a folder of TIFFs written without a `format`.

Order matters: a filter needs an active pixel layer, so create adjustment layers last — once one is
created it becomes the active layer and a following filter fails. Every step echoes its own JSON
result: read it, because it names the defaults that ran beside your parameters.

Everyday jobs, as chains:
- Colour and tone: `image.adjustments.levels`, `layer.newAdjustmentLayer.curves`
  (`{"points": [[0,0], [64,48], [192,212], [255,255]]}`),
  `layer.newAdjustmentLayer.hueSaturation` (`{"saturation": -100}` desaturates),
  `layer.newAdjustmentLayer.vibrance`, `image.adjustments.invert`.
- Sharpen: `filter.sharpen.smartSharpen` — `{"amount": 80}` also applies its own radius and noise
  reduction, which the result JSON names.
- Crop, resize, rotate and the rest of geometry: filter the registry for `crop` or `rotate`; resize
  is `image.imageSize`. The ids carry their menu path, so the document-wide ones start with `image.`.
- Format conversion, HEIC included: name the extension you want in `output`. HEIC is read-only, so
  a HEIC source is converted to PNG or JPG before it can be edited any further.
- A look to reuse: `file.export.colorLookupTables` writes a `.cube` from the document's adjustment
  layers.

Layers: the native `.pcraft` format, PSD and TIFF keep layers and layer styles; PNG and JPG export
flattens, and the engine says so. To keep a layered job editable, save it as `.pcraft` or `.psd`
and open that file in the next call — nothing is kept between calls.

Metadata: EXIF, XMP and ICC pass through an open-and-save. Document-level XMP fields — title,
author, keywords, copyright — are written by `file.fileInfo`, which also reads a camera's EXIF back;
camera EXIF itself is read-only. IPTC is dropped in every format.

Output: `output` names the file to write, and a path that already exists is refused — choose a new
name. With no `output`, a fresh file is written into generated/ in the input's own format, so a
layered document stays layered. An input the engine only reads — HEIC, Krita, Affinity, Paint.NET,
camera raw, PCX, DICOM — or one with no extension becomes `.png`, and so does a canvas created from
nothing; name `output` to write another format. AVIF is neither read nor written by this engine, so
convert it elsewhere first.

A `batch` writes into a folder that is new or empty, and its answer opens with how many files were
written and then lists them; a count the folder could not be read in full for is stated as a floor
("at least"), and a folder that could not be read at all says that rather than counting nothing, so a
short read is never passed off as the count. A long list is cut to its first and last lines with a
count of the ones left out, so a file it does not name was still written; the per-file lines are kept
before the warnings, so a batch too large for both counts the warning kinds it could not show. The
engine's per-file warnings are collapsed to one line per kind, `warning (<n> lines): <what it says>`
(`(1 line)` for a single one), where the count is how many warning lines of that kind it printed —
one per file, so it reads as the number of files affected; a warning naming a per-file value shows
`N` in its place, and the numbers every file agreed on are kept. The engine's own succeeded and
failed counts are kept however long the batch, and every file it failed on is named with its reason
— a batch with more failures than one answer can hold says how many more there were, so read the
counts before reporting a batch's outcome. A batch that stops partway keeps the files it already
wrote and says how many there are.

Shortened answers: a registry answer this large drops whole commands and says how many, so a command
it leaves out is one a narrower filter reaches; an inspection of a heavily layered document keeps its
size, mode and depth and cuts the layer tree to the layers that fit, counting the rest, and a report
too large for the product to capture at all says so and keeps only the format the file name gives.

A parameter that names a file — a watermark, a placed layer — must be an absolute path: the engine
resolves a relative one against its own working directory, not yours.
