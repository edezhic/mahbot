//! The `document` tool: create `.docx`/`.xlsx`/`.pptx`/`.pdf` files, fill a
//! user's own sample, edit an existing Office file, and edit PDF pages.
//!
//! All work is data — never code or an expression — handed to the embedded
//! document kit ([`crate::docgen`]) that runs on the managed bun runtime. The
//! operation set is fixed; the model chooses an action and the data for it, and
//! the tool owns every path decision the model must not make.
//!
//! # Path confinement
//!
//! Every path the model supplies — a sample, a PDF to edit, an image to embed —
//! is resolved with [`crate::tools::path::resolve_read_target`] under the
//! caller's [`PathAccess`]: workspace-only for a guest's Assistant, anywhere the
//! shell could name for the admin's. The OUTPUT path is never the model's: the
//! tool picks a name inside its output directory — the caller's own workspace's
//! `generated/` for every caller but the command line, which names the
//! destination itself — and reserves it with a create-new write before the kit
//! runs.
//!
//! # Input rules
//!
//! Every shape the kit's writers cannot render is refused here, before the
//! runtime is spawned. The bounds and enumerations the tool, the kit and the
//! presentation reader share are stated once — in `assets/docgen/rules.json`,
//! compiled in and parsed in one place by [`crate::docgen::rules`] — so a rule
//! cannot drift between them.
//!
//! # Editing
//!
//! `docx_edit`, `xlsx_edit` and `pptx_edit` change an existing Office file. Three
//! promises hold for every one: the input is never modified, the result is a
//! new file under the output directory (the workspace's `generated/`, or the
//! destination a command-line call was given), and every part an edit does not name
//! is carried into it with the content it had — a chart, a comment, a header or
//! a media file the caller never mentioned included. The raw archive is
//! re-compressed, so byte-identical bytes are not promised; the content is. The
//! honest limit is the other side of that: a package's aggregate formatting, a
//! presentation's layouts and animations, and complex objects are not rebuilt,
//! and the kit reports in its `notes` what it could not keep rather than passing
//! a loss off as success.
//!
//! # Naming
//!
//! `file_name` is sanitized to a single path component and made unique within the
//! output directory, so a model-supplied name can neither escape the directory
//! nor silently overwrite an earlier file. The tool appends the extension it
//! produces — the action's own, or the sample's or input's on a fill or an edit
//! — after dropping a trailing OOXML or PDF extension (any of `.docx`, `.docm`,
//! `.xlsx`, `.xlsm`, `.pptx`, `.pptm`, `.pdf`), so a name that already carries
//! one is not doubled.

use crate::docgen;
use crate::tools::path::PathAccess;
use crate::{Tool, Workspace};
use anyhow::{Context as _, Result};
use async_trait::async_trait;
use chrono::{Datelike as _, Timelike as _};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use super::GENERATED_DIR;

/// The marker every produced file reaches a caller through: `[FILE:<path>]`.
pub(crate) const FILE_MARKER: &str = "[FILE:";

/// Longest base name the tool keeps from `file_name`; the extension and the
/// uniqueness suffix are added on top, and a cap keeps the result well inside a
/// filesystem's component limit.
const MAX_BASE_CHARS: usize = 80;

/// Upper bound on the files ONE `pdf_split` call may produce. The selection list
/// is the model's, so without this a call could turn a document into thousands of
/// attachments; the page cap bounds the pages, not the files they are cut
/// into. More than this is more than a reply can carry.
const MAX_SPLIT_PARTS: usize = 50;

/// Upper bound on the files ONE `pdf_merge` call may take; more is refused.
const MAX_MERGE_INPUTS: usize = 50;

/// Upper bound on what one `pdf_merge` may load in total. Every input is one
/// file the tool already holds under `FILE_MAX_BYTES`, but the kit reads them all
/// at once: without a total, a caller-caused merge of many large PDFs would end
/// as a killed run reported as a product fault.
const MAX_MERGE_BYTES: u64 = 200 * 1024 * 1024;

/// Most uncovered characters the "missing from the file" note names; the kit's
/// `unsupported` array still carries every one, and the rest are counted in the
/// note rather than listed.
const UNSUPPORTED_NOTE_MAX: usize = 10;

/// The tool's fixed operation set, in the order the schema lists it. The schema's
/// `oneOf` entries and the hint for an unknown action both come from here; the
/// dispatch `match` and [`action_schema`] spell the names again, one arm per
/// action.
const ACTIONS: [&str; 13] = [
    "create",
    "fill_template",
    "docx_edit",
    "xlsx_edit",
    "pptx_edit",
    "pdf_merge",
    "pdf_split",
    "pdf_rotate",
    "pdf_crop",
    "pdf_resize",
    "pdf_text",
    "pdf_image",
    "pdf_form_fill",
];

/// One family's edit vocabulary: every op with the fields it takes, in the order
/// the description states them. The `op` enum, the `Fields by op:` text, the
/// refusal of an unknown op and the refusal of a field its op does not take all
/// read this one table, so none of them can disagree about what an edit takes.
/// The rules a field list cannot carry — one target only, at least one property
/// named — are stated beside it, in the family's own `edits_arg` sentence.
#[expect(clippy::too_many_lines)] // the one table every family's vocabulary lives in
fn edit_op_fields(
    family: crate::ooxml::Family,
) -> &'static [(&'static str, &'static [&'static str])] {
    match family {
        crate::ooxml::Family::Docx => &[
            ("replace_text", &["find", "replace"]),
            ("insert_text", &["find", "insert", "position"]),
            ("remove_text", &["find"]),
            ("remove_paragraph", &["find"]),
            (
                "format_text",
                &[
                    "find",
                    "bold",
                    "italic",
                    "underline",
                    "strike",
                    "size",
                    "color",
                ],
            ),
            (
                "format_paragraph",
                &[
                    "find",
                    "align",
                    "style",
                    "indent_left",
                    "indent_right",
                    "indent_first_line",
                    "spacing_before",
                    "spacing_after",
                    "line_spacing",
                    "list",
                    "level",
                ],
            ),
            ("add_paragraph", &["text", "after"]),
            ("add_image", &["after", "image", "width", "height"]),
            ("add_table", &["after", "headers", "rows"]),
        ],
        crate::ooxml::Family::Xlsx => &[
            ("set_cell", &["sheet", "cell", "value", "number_format"]),
            ("clear_cell", &["sheet", "cell"]),
            (
                "format_cells",
                &[
                    "sheet",
                    "range",
                    "column",
                    "row",
                    "font",
                    "size",
                    "bold",
                    "italic",
                    "underline",
                    "strike",
                    "color",
                    "fill",
                    "border",
                    "align",
                    "vertical",
                    "wrap",
                    "number_format",
                    "width_chars",
                    "height_points",
                ],
            ),
            ("insert_row", &["sheet", "row"]),
            ("delete_row", &["sheet", "row"]),
            ("insert_column", &["sheet", "column"]),
            ("delete_column", &["sheet", "column"]),
            ("merge_cells", &["sheet", "range"]),
            ("unmerge_cells", &["sheet", "range"]),
            ("add_sheet", &["name"]),
            ("rename_sheet", &["sheet", "name"]),
            ("delete_sheet", &["sheet"]),
            ("move_sheet", &["sheet", "to"]),
            ("add_image", &["sheet", "cell", "image", "width", "height"]),
        ],
        crate::ooxml::Family::Pptx => &[
            ("replace_text", &["slide", "find", "replace"]),
            ("remove_text", &["slide", "find"]),
            (
                "format_text",
                &[
                    "slide",
                    "find",
                    "bold",
                    "italic",
                    "underline",
                    "strike",
                    "size",
                    "color",
                    "align",
                ],
            ),
            ("add_paragraph", &["slide", "text", "after", "level"]),
            ("remove_paragraph", &["slide", "find"]),
            (
                "add_image",
                &[
                    "slide",
                    "image",
                    "x_fraction",
                    "y_fraction",
                    "width_fraction",
                    "height_fraction",
                ],
            ),
            (
                "add_table",
                &[
                    "slide",
                    "headers",
                    "rows",
                    "x_fraction",
                    "y_fraction",
                    "width_fraction",
                    "height_fraction",
                ],
            ),
            ("add_slide", &["after", "title", "bullets"]),
            ("delete_slide", &["slide"]),
            ("move_slide", &["slide", "to"]),
            ("duplicate_slide", &["slide"]),
            ("replace_notes", &["slide", "find", "replace"]),
            ("remove_notes", &["slide", "find"]),
            ("add_notes", &["slide", "text"]),
        ],
    }
}

/// The fields the op `op` of `family` takes, from the one table the `op` enum,
/// the op description and the unknown-field refusal read.
fn op_fields(family: crate::ooxml::Family, op: &str) -> &'static [&'static str] {
    edit_op_fields(family)
        .iter()
        .find(|(name, _)| *name == op)
        .map(|(_, fields)| *fields)
        .expect("the op was checked against the family's vocabulary")
}

/// The pptx edits that look text up on the slide itself, and so are the only
/// ones whose refusal can name a diagram's text (the kit's `slideFindRefusal`):
/// `add_paragraph` looks up its `after` anchor, the others a `find`. A notes op
/// searches the notes part instead, where no diagram is, so no diagram text is
/// read for one.
const PPTX_SLIDE_TEXT_OPS: [&str; 5] = [
    "replace_text",
    "remove_text",
    "format_text",
    "add_paragraph",
    "remove_paragraph",
];

/// The formats `create` understands: the three OOXML families (whose names the
/// kit's `format` field also uses) plus PDF; the value doubles as the extension.
const SUPPORTED_FORMATS: [&str; 4] = [
    crate::ooxml::Family::Docx.name(),
    crate::ooxml::Family::Xlsx.name(),
    crate::ooxml::Family::Pptx.name(),
    "pdf",
];

/// A closed numeric range from [`RULES`]. `min_exclusive` marks a low end the
/// value has to exceed rather than reach.
#[derive(serde::Deserialize)]
struct Span {
    min: f64,
    max: f64,
    #[serde(default)]
    min_exclusive: bool,
}

impl Span {
    /// Whether `value` sits inside the range.
    fn contains(&self, value: f64) -> bool {
        let above_min = if self.min_exclusive {
            value > self.min
        } else {
            value >= self.min
        };
        above_min && value <= self.max
    }

    /// The range in the words the refusals use, matching the kit's own wording
    /// for the same bound (see `assets/docgen/kit.js`).
    fn bounds(&self) -> String {
        if self.min_exclusive {
            format!("greater than {} and at most {}", self.min, self.max)
        } else {
            format!("between {} and {}", self.min, self.max)
        }
    }
}

/// The parsed [`crate::docgen::rules`] (see the file for what each entry
/// governs).
#[derive(serde::Deserialize)]
struct Rules {
    heading_levels: Vec<f64>,
    scalar_kinds: Vec<String>,
    image_extensions: Vec<String>,
    /// The most columns a Word table may hold: Word refuses a row of more than 63
    /// cells, so every table written into a `.docx` — a `create` block's or an
    /// `add_table`'s — is held to it. A presentation's or a PDF's table has no such
    /// bound.
    table_columns_max: usize,
    /// A length in points on a page, where zero is not a length: a placed
    /// picture's width or height — a content block's image (`width`/`height`), a
    /// docx `add_image`'s and a `pdf_image`'s placed size alike — and the font
    /// size a `pdf_text` draws at. One unit and one bound wherever a page measure
    /// is stated; a picture side derived from the one the request names is held to
    /// it too.
    dimension_points: Span,
    /// A pptx `add_image` or `add_table` placement: the fraction of the slide's
    /// width or height the shape is placed at.
    slide_position_fraction: Span,
    /// A pptx `add_image` or `add_table` size: the fraction of the slide's width
    /// or height the shape is scaled to, where zero is not a size.
    slide_size_fraction: Span,
    /// A docx `format_text` size in points. Word's `<w:sz>` counts half-points
    /// and caps the value at 1638 — 819 points — so the kit writes the doubled
    /// number and both sides refuse past this bound rather than write a
    /// `<w:sz>` a reader cannot hold.
    text_size_points: Span,
    /// A docx `format_paragraph` alignment: the word a caller names and the
    /// `<w:jc>` value ECMA-376 writes for it, the kit's writer reading the same
    /// pairs so the word and the value written for it cannot drift.
    paragraph_alignments: Vec<(String, String)>,
    /// The kinds a docx `format_paragraph` `list` may name, each paired with the
    /// list kind the paragraph's own `<w:numPr>` carries, the kit reading the
    /// same pairs the way it reads the alignments'.
    paragraph_lists: Vec<(String, String)>,
    /// A docx `format_paragraph` indent in points — the unit the tool states and
    /// the kit writes to Word's twips — for `indent_left`, `indent_right` and
    /// `indent_first_line`. The range reaches below zero because the kit writes a
    /// first-line indent as `<w:firstLine>` when it is positive and as
    /// `<w:hanging>` when it is negative.
    paragraph_indent_points: Span,
    /// A docx `format_paragraph` spacing in points (`spacing_before`,
    /// `spacing_after`), the unit Word's `<w:spacing w:before>`/`w:after` holds.
    paragraph_spacing_points: Span,
    /// A docx `format_paragraph` line height in points (`line_spacing`), the unit
    /// Word's `<w:spacing w:line>` holds. The kit writes the value in whole
    /// twips — a twentieth of a point — by rounding, so a floor below about
    /// 0.025 would round to the zero twips a reader draws as no line at all; the
    /// bound is stated at a round 0.05 above that boundary rather than resting on
    /// it.
    paragraph_line_points: Span,
    /// The longest docx `format_paragraph` `style` name. The kit resolves the
    /// name against the document's own styles and refuses an unknown one, so
    /// this is only a length no style name reaches, not a check that one exists.
    paragraph_style_max: usize,
    /// A pptx `format_text` size in points. DrawingML's `<a:rPr sz>` counts
    /// hundredths of a point, so the ruler is its own — unlike
    /// [`Rules::text_size_points`], which is Word's half-point cap.
    slide_text_size_points: Span,
    /// A xlsx `format_cells` font size in points, `<sz val>`'s own unit: Excel's
    /// own size box holds 1 to 409 points, and a size past it is a font no reader
    /// shows as asked.
    cell_font_points: Span,
    /// A xlsx `format_cells` column width in characters — the unit `<col width>`
    /// carries and Excel's own column-width box shows.
    column_width_chars: Span,
    /// A xlsx `format_cells` row height in points, `<row ht>`'s own unit.
    row_height_points: Span,
    /// The longest xlsx `format_cells` font name: Excel's own limit, and
    /// `<name val>` carries the name verbatim.
    font_name_max: usize,
    /// The most cells one xlsx `format_cells` range may address. A range is
    /// written out cell by cell — an empty one included, because a formatted cell
    /// is part of the sheet's used range — so a whole-column spelling
    /// (`"A1:A1048576"`) would write a sheet-sized body instead of the handful
    /// of cells the caller meant.
    format_cells_max: usize,
    /// The pages one request may name, read by `pages_arg`, by the tool's own
    /// explicit-selection bound and — in the kit — by a selection the document's
    /// own length resolves.
    max_pages_per_request: usize,
    pdf_point_abs_max: f64,
    /// The fraction of a page's visible area a `pdf_crop` names for one side of
    /// its `area`, measured from the displayed top-left. Zero is a legal edge —
    /// the left or top of the area — while the width and height the area states
    /// must be above zero, which the boundary checks on the area as a whole.
    pdf_page_fraction: Span,
    /// The slack a `pdf_crop` area's sides may reach past [`Self::pdf_page_fraction`]'s
    /// top end, so an area a caller computed (`1/3` wide, starting two thirds
    /// across) is not refused for the last bit a double cannot hold. The kit's
    /// own boundary check reads the same number, so the two cannot disagree.
    pdf_fraction_slack: f64,
    /// A `pdf_text` `opacity`: the ink the text and its stamp box are drawn with,
    /// where zero would draw nothing at all.
    pdf_opacity: Span,
    color_digits: usize,
    /// A pptx `format_text` alignment: the word a caller names, and the `algn`
    /// value ECMA-376 writes for it. The kit's writer reads the same pairs, so
    /// the word a call may name and the value written for it cannot drift.
    slide_alignments: Vec<(String, String)>,
    /// The values a xlsx `format_cells` may name for `align`: the `<alignment
    /// horizontal>` tokens themselves, since the word and the value are one.
    cell_alignments: Vec<String>,
    /// The values a xlsx `format_cells` may name for `vertical`, the `<alignment
    /// vertical>` tokens.
    cell_verticals: Vec<String>,
    /// The sides a xlsx `format_cells` border may name, in the order the prompt
    /// states them. The kit's reader and writer read the same list.
    cell_border_sides: Vec<String>,
    degrees_step: i64,
    sheet_name_max: usize,
    sheet_name_forbidden: String,
    edit_text_max: usize,
    edits_max: usize,
    bullets_max: usize,
    paragraph_level_max: u32,
    sheet_row_max: u32,
    sheet_column_max: u32,
    /// The longest xlsx `set_cell`/`format_cells` `number_format`: text inside
    /// the shared cap, an empty format not being one.
    number_format_max: usize,
    /// The slide sizes a created presentation may name, in the order the prompt
    /// states them. The kit reads the same list and defines the layout it writes
    /// from the same EMU, so a name and the deck it produces cannot drift.
    pptx_sizes: Vec<PptxSize>,
    /// The deck this tool has written since before a size could be named — the
    /// 10 × 5.625 in one (`LAYOUT_16x9`) every measurement the kit places itself is
    /// drawn on, and the one an unnamed `create` still writes. Stated here because
    /// the model-facing text names it and the kit's own reference geometry is read
    /// from it.
    pptx_default_size: DeckSize,
}

/// A slide size in EMU: the pair of lengths a deck's `<p:sldSz>` states, which
/// every deck-geometry measurement is scaled from.
#[derive(serde::Deserialize)]
struct DeckSize {
    width_emu: u64,
    height_emu: u64,
}

impl DeckSize {
    /// The size as the model reads it, in inches to a thousandth — the unit a
    /// caller thinks of a slide in, while the file states EMU. Integer arithmetic
    /// only, and no rounding: a thousandth of an inch is 914.4 EMU, so the sizes a
    /// deck states are spelled exactly where a thousandth can (10 × 5.625 in), and
    /// one that needs more digits — widescreen's 13.333… in — is truncated there.
    fn inches(&self) -> String {
        const EMU_PER_INCH: u64 = 914_400;
        let side = |emu: u64| {
            let fraction = format!("{:03}", emu % EMU_PER_INCH * 1000 / EMU_PER_INCH);
            let fraction = fraction.trim_end_matches('0');
            let dot = if fraction.is_empty() { "" } else { "." };
            format!("{}{dot}{fraction}", emu / EMU_PER_INCH)
        };
        format!("{} × {} in", side(self.width_emu), side(self.height_emu))
    }
}

/// One size [`Rules::pptx_sizes`] states: the word a `create` may name and the
/// deck that word means, in EMU — the unit `<p:sldSz>` states a slide in.
#[derive(serde::Deserialize)]
struct PptxSize {
    name: String,
    #[serde(flatten)]
    deck: DeckSize,
}

/// [`crate::docgen::rules`] parsed into this tool's own shape: the shared file's
/// one JSON read is there, and each consumer deserializes what it needs out of
/// it. A file this shape cannot be taken from is a build fault, caught by the
/// test below rather than by a call.
static RULES: LazyLock<Rules> = LazyLock::new(|| {
    serde::Deserialize::deserialize(crate::docgen::rules())
        .expect("assets/docgen/rules.json must state the rules the document tool enforces")
});

/// Create and edit office documents and PDFs. See the module docs.
///
/// `access` is the caller's [`PathAccess`], fixed when the role's toolset is
/// built: it decides where the tool's *inputs* may come from. Every produced
/// file lands in the caller's own workspace's `generated/` directory, whatever
/// the access — except for a command-line call, which produces into the
/// destination the caller was given ([`crate::agent::tool_output_dir`]).
pub(crate) struct DocumentTool {
    access: PathAccess,
}

impl DocumentTool {
    #[must_use]
    pub(crate) const fn new(access: PathAccess) -> Self {
        Self { access }
    }

    /// Where the model may read its inputs from, in its own access level's
    /// wording: the admin's may be read from anywhere, a guest's stay inside its
    /// workspace. Only those two levels reach this tool (it belongs to the
    /// Assistant alone); the allowlisted level is worded as the workspace it
    /// certainly covers.
    const fn input_location(&self) -> &'static str {
        match self.access {
            PathAccess::Unrestricted => {
                "readable from anywhere on this machine (the product's own live databases excepted)"
            }
            PathAccess::Workspace | PathAccess::Allowlisted => "confined to your workspace",
        }
    }
}

#[async_trait]
impl Tool for DocumentTool {
    fn name(&self) -> &'static str {
        "document"
    }

    /// Hidden while the managed runtime is missing: the tool cannot work without
    /// it, and the model should not burn a call on the refusal. An agent built
    /// during an outage keeps the verdict for its lifetime (see
    /// [`Tool::is_advertised`]), and `execute` still refuses honestly if the
    /// runtime vanishes mid-run.
    fn is_advertised(&self) -> bool {
        crate::tools::bun::bun_binary_path().is_some()
    }

    /// A produced file reaches the user only through its `[FILE:…]` marker, so
    /// the tool advertises the marker kind: an answer that omitted it gets it
    /// appended, and the media sweep keeps a file the reply mentions.
    fn media_marker(&self) -> Option<&'static str> {
        Some(FILE_MARKER)
    }

    /// Every produced file reaches the user only through its marker, and one
    /// reply can carry many (`pdf_split` lists one per range): a truncated
    /// marker list would deliver part of what the call produced.
    fn preserve_full_output(&self) -> bool {
        true
    }

    /// The prompt asset with the rules the interface cannot state rendered in:
    /// what the model cannot derive from the parameter schema — the safety
    /// promise, the reader's own printing it must not mistake for a document's
    /// text, and the losses and refusals no argument carries. Every bound lives in
    /// the schema beside the argument that carries it, so the two cannot drift.
    fn description(&self) -> String {
        let ppt = crate::reader_output::ppt_description_substitutions();
        let input_mb = megabytes(crate::util::FILE_MAX_BYTES).to_string();
        let edit_text_max = RULES.edit_text_max.to_string();
        // The Word marks are quoted from the shared list: the ones a reading
        // prints in brackets and the labels it adds without them.
        let docx_inline_marks = spelled_docx_marks(true);
        let docx_label_marks = spelled_docx_marks(false);
        let mut pairs: Vec<(&str, &str)> = vec![
            ("{{input_location}}", self.input_location()),
            ("{{input_mb}}", &input_mb),
            ("{{edit_text_max}}", &edit_text_max),
            ("{{docx_inline_marks}}", &docx_inline_marks),
            ("{{docx_label_marks}}", &docx_label_marks),
        ];
        pairs.extend(ppt.iter().map(|(key, value)| (*key, value.as_str())));
        crate::prompt::substitute(&crate::prompt::load_prompt("tool/document.md"), &pairs)
    }

    /// The machine-readable per-action contract, in the shape the other
    /// action-rich tool (`chrome`) uses: the root declares `action` as a `oneOf`
    /// with one entry per action, each built by [`super::action_entry_schema`] so
    /// the arguments it takes, the ones it requires and every selector's allowed
    /// values are the schema's own statement rather than prose. The bounds the
    /// tool enforces on a value the model passes appear in the description of the
    /// argument that carries it (see [`action_schema`]).
    fn parameters_schema(&self) -> Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "oneOf": ACTIONS.iter().map(|action| action_schema(action)).collect::<Vec<_>>()
                },
                "file_name": file_name_arg(),
                // The one description shape several actions take is stated here
                // once, beside `file_name`; an action's own entry only opts into
                // it (see [`metadata_entry`]).
                "metadata": metadata_arg(),
            }),
            &["action"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: Value) -> Result<String> {
        Ok(self.dispatch(ws, args).await?.text)
    }

    async fn execute_answer(&self, ws: &Workspace, args: Value) -> Result<super::ToolAnswer> {
        self.dispatch(ws, args).await
    }
}

/// A document call's arguments in the flat shape the dispatch reads: `action` a
/// string with the action's own arguments beside it. The nested shape the schema
/// literally describes — `{"action": {"pdf_rotate": {…}}}` — is flattened onto
/// it, and so is the chrome-shaped `{"action": "pdf_rotate", "pdf_rotate": {…}}`;
/// the shared top-level `file_name` is merged in. Anything else is refused with
/// the shape the caller was expected to use.
fn normalize_call(args: &Value) -> Result<Value> {
    let nested_shape = || {
        anyhow::anyhow!(
            "usage: \"action\" must name one of {} as a string with the action's own arguments \
             beside it, or as an object naming one action — hint: like {{\"action\": \"pdf_rotate\", \
             \"path\": \"…\", \"degrees\": 90}} or {{\"action\": {{\"pdf_rotate\": {{…}}}}}}",
            ACTIONS.join(", ")
        )
    };
    let flattened = |name: &str, fields: &serde_json::Map<String, Value>| {
        // The caller's other top-level keys come along rather than being dropped:
        // an argument stated beside the action object is the caller's, and one the
        // action does not take is refused by name afterwards — never silently lost.
        let object = args
            .as_object()
            .expect("a call whose action matched is an object");
        let mut flat = object
            .iter()
            .filter(|(key, _)| key.as_str() != "action" && key.as_str() != name)
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<serde_json::Map<String, Value>>();
        flat.extend(fields.clone());
        flat.insert("action".to_owned(), json!(name));
        if let Some(file_name) = args.get("file_name") {
            flat.insert("file_name".to_owned(), file_name.clone());
        }
        Value::Object(flat)
    };
    match args.get("action") {
        // The flat call, and the chrome-shaped one whose arguments sit under a
        // key named after the action rather than beside it.
        Some(Value::String(name)) => match args.get(name.as_str()) {
            Some(Value::Object(fields)) => Ok(flattened(name, fields)),
            _ => Ok(args.clone()),
        },
        // The nested call the schema describes: the one key names the action and
        // its value is that action's arguments.
        Some(Value::Object(action)) if action.len() == 1 => {
            let (name, fields) = action.iter().next().expect("one key");
            match fields {
                Value::Object(fields) => Ok(flattened(name, fields)),
                _ => Err(nested_shape()),
            }
        }
        _ => Err(nested_shape()),
    }
}

/// Refuse an argument the action does not take, from the action's own schema
/// entry: the interface is the contract, so a key the action has no place for is
/// named rather than silently ignored. `action` and the shared `file_name` are
/// every action's own. A `null` value is not stated, the way the whole boundary
/// reads one, so a null key is not an argument the action does not take.
fn require_declared_arguments(action: &str, args: &Value) -> Result<()> {
    let entry = action_schema(action);
    let declared = entry["properties"][action]["properties"]
        .as_object()
        .expect("every action's entry declares its arguments");
    let accepted = std::iter::once("action")
        .chain(std::iter::once("file_name"))
        .chain(declared.keys().map(String::as_str))
        .collect::<Vec<_>>();
    let Some(object) = args.as_object() else {
        return Ok(());
    };
    let unknown = object
        .keys()
        .map(String::as_str)
        .filter(|key| !accepted.contains(key) && stated(object, key))
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        let named = listed(
            &unknown
                .iter()
                .map(|key| format!("\"{key}\""))
                .collect::<Vec<_>>(),
        );
        anyhow::bail!(
            "usage: {action} takes no argument {named} — hint: it takes {}",
            listed(&accepted)
        );
    }
    Ok(())
}

// ── the parameter interface ──────────────────────────────────────────
//
// One builder per argument shape: an action's entry names the arguments it takes
// rather than restating them, and every bound a value is checked against — the
// shared rules and this tool's own caps — appears in the description of the
// argument that carries it, so what the model is told is what the boundary
// enforces.

/// A string argument.
fn text_arg(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}

/// A number argument.
fn number_arg(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}

/// A whole-number argument.
fn integer_arg(description: &str) -> Value {
    json!({ "type": "integer", "description": description })
}

/// A run toggle: its meaning is stated once, in the `op` of the family's own
/// formatting path, rather than beside each of the four names.
fn toggle_arg() -> Value {
    json!({ "type": "boolean" })
}

/// A boolean argument.
fn bool_arg(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}

/// A string argument closed to `values`.
fn choice_arg<S: AsRef<str>>(values: &[S], description: &str) -> Value {
    let values: Vec<&str> = values.iter().map(AsRef::as_ref).collect();
    json!({ "type": "string", "enum": values, "description": description })
}

/// An array argument of `items`.
fn array_arg(items: &Value, description: &str) -> Value {
    json!({ "type": "array", "items": items, "description": description })
}

/// The scalar a bullet or a cell may be — the kinds the shared rules name.
fn scalar_arg() -> Value {
    json!({ "type": RULES.scalar_kinds })
}

/// `file_name`: the produced file's base name, capped at [`MAX_BASE_CHARS`].
fn file_name_arg() -> Value {
    text_arg(&format!(
        "Base name for the produced file (max {MAX_BASE_CHARS} chars); the tool appends the \
         extension, puts the file in the workspace's generated/ directory and makes the name \
         unique."
    ))
}

/// `metadata`: the PDF's own description, which every PDF-writing action takes.
/// A field left out is left as the file has it rather than cleared (an existing
/// `modified` date advances to the write), so the absence of a key and an empty
/// string mean different things.
fn metadata_arg() -> Value {
    let properties = crate::docgen::pdf_metadata_fields()
        .iter()
        .map(|field| (field.key.clone(), text_arg(&field.description)))
        .collect::<serde_json::Map<_, _>>();
    json!({
        "type": "object",
        "minProperties": 1,
        "additionalProperties": false,
        "description": format!(
            "The PDF's own description. A field left out stays as the file has it — except an \
             existing modified date, which advances to a write; an empty string clears one; \
             keywords is the single text the file carries. A date is {}.",
            crate::docgen::pdf_date_format()
        ),
        "properties": properties,
    })
}

/// `metadata` as a PDF action's entry declares it: the action opts into the
/// tool-level argument, whose fields are stated once in
/// [`parameters_schema`](DocumentTool::parameters_schema) because a copy per PDF
/// action would cost more contract bytes than the schema has room for.
fn metadata_entry() -> Value {
    json!({ "type": "object" })
}

/// `path`: the one input document every action but `pdf_merge` takes. The file is
/// never modified, and what its own bytes hold is the format the action works on.
fn path_arg() -> Value {
    text_arg("The document to work on.")
}

/// `pdf_image`: the picture to place on a PDF's page. A PDF holds its pictures in
/// its own two encodings, so the accepted set is the action line's.
fn pdf_image_arg() -> Value {
    picture_arg(pdf_pictures().names)
}

/// `image`: the picture to place — a PNG, a JPEG or a WebP, by its content — and
/// the one sizing rule every placement shares.
///
/// Every field built here NAMES A FILE, which is what the command line reads: a
/// field inside a structured value whose schema is one of `image_args` is a
/// reference path the frontend must resolve in the caller's own directory, so the
/// two cannot disagree about which fields those are (see `crate::cli`'s field
/// tie).
fn image_arg() -> Value {
    picture_arg(embedded_pictures().names)
}

/// `image` inside a `create` content block: [`image_arg`]'s picture, with the
/// narrower set a PDF takes — one `create` entry covers every format, so the
/// exception is stated where the target may be a PDF.
fn content_image_arg() -> Value {
    picture_arg(&format!(
        "{}; a PDF takes {}",
        embedded_pictures().names,
        pdf_pictures().names
    ))
}

/// The picture arguments [`image_arg`] and its two siblings build: every one of
/// them names a file, whatever set its own target takes, so the command line's
/// field tie reads them as one family.
#[cfg(test)]
pub(crate) fn image_args() -> Vec<Value> {
    vec![image_arg(), content_image_arg(), pdf_image_arg()]
}

/// What a picture argument states, with `names` the formats its target takes.
fn picture_arg(names: &str) -> Value {
    text_arg(&format!(
        "The picture to place ({names}); naming one side's size alone keeps its own proportions."
    ))
}

/// `paths`: a merge's inputs, with the item count the interface declares and the
/// total size the boundary enforces — the count as `minItems`/`maxItems`, the
/// total as the one thing only the description can state.
fn paths_arg() -> Value {
    json!({
        "type": "array",
        "items": text_arg("Path of one PDF."),
        "minItems": 2,
        "maxItems": MAX_MERGE_INPUTS,
        "description": format!(
            "The PDFs to merge, in order: at most {} MB in total.",
            megabytes(MAX_MERGE_BYTES)
        ),
    })
}

/// `values`: the name-keyed map both fills take. At least one entry is required —
/// the schema's `minProperties` and the boundary's refusal alike.
fn values_arg(kind: &str) -> Value {
    json!({
        "type": "object",
        "minProperties": 1,
        "description": format!(
            "One entry per {kind}, keyed by its name; a value is text, a number or a boolean."
        ),
    })
}

/// `pages`: the one page grammar every PDF action that takes pages uses. An
/// element is `"all"` or a comma-separated list of `N` and `N-M`; for a split
/// each element is one output file, for the other actions their union is what the
/// action acts on.
fn pages_arg(split: bool) -> Value {
    let element = text_arg(
        "An entry: \"all\", a page, a range, or a comma list like \"1-3,5\"; \"all\" stands alone.",
    );
    if split {
        json!({
            "type": "array",
            "items": element,
            "minItems": 1,
            "maxItems": MAX_SPLIT_PARTS,
            "description": format!(
                "One output file per entry, in order: [\"1-3\", \"5\"] makes two files, and \
                 [\"1-3,5\"] one file holding those four pages. At most {MAX_SPLIT_PARTS} files \
                 and {} pages.",
                RULES.max_pages_per_request
            ),
        })
    } else {
        array_arg(
            &element,
            &format!(
                "The pages to act on, as [\"1-3\", \"5\"]; absent means all pages. At most \
                 {} pages.",
                RULES.max_pages_per_request
            ),
        )
    }
}

/// A row of a table or a sheet: the cells of one row, the scalars the shared
/// rules name (a sheet cell may also be a formula object).
fn row_arg() -> Value {
    json!({ "type": "array", "items": scalar_arg() })
}

/// The one statement of a placed picture's size: the unit and the bound, and the
/// fact that the side the request derives from the named one is held to it too —
/// the proportions that derivation keeps are stated on the picture field itself.
fn picture_size_rule() -> String {
    format!(
        "in points, {}; a side derived from the other one is held to the same bound",
        RULES.dimension_points.bounds()
    )
}

/// A size a picture is placed at, in points — a content-block image's or a docx
/// `add_image`'s `height`, and a `pdf_image`'s placed size, so the bound has one
/// statement.
fn placed_size_arg(what: &str) -> Value {
    number_arg(&format!("{what} {}.", picture_size_rule()))
}

/// With neither side of a placed picture named, a laid-out block's keeps its own
/// size, fitted whole into the page's or slide's own text column so neither side of
/// it runs past. A picture in a sheet has no column to fit and keeps its own size.
const BLOCK_IMAGE_FIT: &str = "with neither side named the picture keeps its own size, fitted \
                                whole into the page's or slide's own text column.";

/// A placed picture's width: the shared placement rule, then `fallback` — what
/// happens with neither side named, the one part a laid-out block's picture and one
/// in a sheet do not share.
fn picture_width_arg(what: &str, fallback: &str) -> Value {
    number_arg(&format!("{what} {}; {fallback}", picture_size_rule()))
}

/// A `format_cells` `border`: the sides to set, each a boolean — true puts a thin
/// border on that side, false takes that side's own away. The sides come from the
/// shared rules, so the declaration and the boundary's refusal name the same ones.
/// An object naming no side states nothing and is refused, which a declaration
/// cannot say.
fn border_arg() -> Value {
    let sides = RULES
        .cell_border_sides
        .iter()
        .map(|side| (side.clone(), bool_arg("Set this side.")))
        .collect::<serde_json::Map<String, Value>>();
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": sides,
        "description": "The sides to set — true draws a thin border on that side, false takes its \
                        own away — at least one of them.",
    })
}

/// A colour in the shared rules' hex digits, with the optional leading `#`.
fn color_arg(what: &str) -> Value {
    text_arg(&format!("{what}: {} hex digits.", RULES.color_digits))
}

/// A `format_paragraph` `list`: the words the shared rules state, whose values
/// name the kind, or `false` taking the paragraph's list away. The two shapes
/// share one declaration, so the words still come from `RULES.paragraph_lists`
/// while `false` says the paragraph loses its numbering.
fn list_arg() -> Value {
    let mut values = pair_words(&RULES.paragraph_lists)
        .into_iter()
        .map(|word| json!(word))
        .collect::<Vec<_>>();
    values.push(json!(false));
    json!({
        "type": ["string", "boolean"],
        "enum": values,
        "description": "The list kind, or false to take the paragraph's list away; absent keeps \
                        its numbering.",
    })
}

/// A `format_cells` `fill`: the shared rules' hex digits for the colour, or
/// `false` taking the cell's fill away — no colour means "no fill", so `false`
/// is the shape that removes it.
fn fill_arg() -> Value {
    json!({
        "anyOf": [
            { "type": "string" },
            { "enum": [false] },
        ],
        "description": format!(
            "Background: {} hex digits, or false to take the cell's fill away.",
            RULES.color_digits
        ),
    })
}

/// The block kinds `create`'s `content` accepts: the fields each kind takes and
/// the ones it cannot do without (beyond the `type` every block carries). One
/// owner: the declaration's variant per kind is built from this table and the
/// boundary refuses a field a block's kind does not take against it.
const CONTENT_BLOCKS: &[(&str, &[&str], &[&str])] = &[
    // kind, its fields, the fields it requires
    ("heading", &["text", "level"], &["text"]),
    ("paragraph", &["text", "footnote"], &["text"]),
    ("notes", &["text"], &["text"]),
    ("list", &["items", "ordered"], &["items"]),
    ("table", &["headers", "rows"], &[]),
    ("image", &["image", "width", "height"], &["image"]),
    ("link", &["text", "url"], &["text", "url"]),
    ("page_break", &[], &[]),
    ("contents", &[], &[]),
];

/// What a block kind's own `type` declaration says about it, for the kinds whose
/// name alone does not carry the meaning: a break, a link and a table of contents
/// are words this vocabulary did not have, so each says what it writes.
fn block_kind_note(kind: &str) -> Option<String> {
    match kind {
        "page_break" => Some("A page break: the next block starts a new page.".to_owned()),
        "link" => Some("A paragraph holding a link: its text and the address it opens.".to_owned()),
        "contents" => Some(format!(
            "A table of contents over the document's headings (levels {}), with the page numbers \
             already written in; one per document.",
            listed(&RULES.heading_levels)
        )),
        _ => None,
    }
}

/// How `create` declares one block field: the single statement of what that field
/// is, whichever kind carries it.
fn block_field(name: &str) -> Value {
    match name {
        "text" => text_arg("The block's text."),
        "level" => integer_arg(&format!("Level {}.", listed(&RULES.heading_levels))),
        "items" => array_arg(&scalar_arg(), "The bullets."),
        "ordered" => bool_arg("Number them."),
        "headers" => array_arg(
            &scalar_arg(),
            &format!(
                "The header cells; a .docx table holds at most {} columns.",
                RULES.table_columns_max
            ),
        ),
        "rows" => array_arg(
            &row_arg(),
            &format!(
                "The rows, each an array of cells; a .docx table holds at most {} columns.",
                RULES.table_columns_max
            ),
        ),
        "image" => content_image_arg(),
        "width" => picture_width_arg("Placed width", BLOCK_IMAGE_FIT),
        "height" => placed_size_arg("Placed height"),
        "footnote" => text_arg(
            "A footnote attached to this paragraph: its text is printed at the foot of the \
             page the paragraph lands on.",
        ),
        "url" => text_arg(
            "The address the link opens: an http, https, mailto or tel address, e.g. \
             https://example.com; another scheme is refused.",
        ),
        other => unreachable!("every block field has a declaration: {other}"),
    }
}

/// `content`: the blocks a docx/pptx/pdf create lays out, in order. One variant per
/// kind, so the fields a block declares are the ones its kind takes — the fields
/// the boundary refuses a block of another kind against — and every variant
/// requires what its kind cannot do without.
fn content_arg() -> Value {
    let variants = CONTENT_BLOCKS
        .iter()
        .map(|(kind, fields, required)| {
            let mut properties = serde_json::Map::new();
            properties.insert(
                "type".to_owned(),
                match block_kind_note(kind) {
                    Some(note) => json!({ "enum": [kind], "description": note }),
                    None => json!({ "enum": [kind] }),
                },
            );
            properties.extend(
                fields
                    .iter()
                    .map(|name| ((*name).to_owned(), block_field(name))),
            );
            json!({
                "type": "object",
                "additionalProperties": false,
                "required": std::iter::once("type").chain(required.iter().copied()).collect::<Vec<_>>(),
                "properties": properties,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "type": "array",
        "minItems": 1,
        "items": { "oneOf": variants },
        "description": "The blocks, in order, each with its own \"type\"; \"notes\" is pptx speaker \
                        notes, a table needs a row or a header, and a link, a page break, a \
                        footnote and a table of contents are a .docx's own.",
    })
}

/// The fields `create` declares for a sheet, and the one a sheet cannot do
/// without — the one owner of that vocabulary, read by the declaration
/// ([`sheets_arg`]) and by the boundary's refusal alike, the way
/// [`CONTENT_BLOCKS`] owns a block's.
const SHEET_FIELDS: &[(&str, bool)] = &[
    // field, required
    ("name", false),
    ("rows", true),
    ("merges", false),
];

/// The field names of [`SHEET_FIELDS`].
fn sheet_field_names() -> Vec<&'static str> {
    SHEET_FIELDS.iter().map(|(name, _)| *name).collect()
}

/// `sheets`: a workbook's sheets, each with a name, its rows and the ranges it
/// merges. The characters a name may not hold are stated inline in the name's own
/// description.
fn sheets_arg() -> Value {
    let properties = SHEET_FIELDS
        .iter()
        .map(|(name, _)| ((*name).to_owned(), sheet_field(name)))
        .collect::<serde_json::Map<String, Value>>();
    let required = SHEET_FIELDS
        .iter()
        .filter(|(_, required)| *required)
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": false,
            "required": required,
            "properties": properties,
        },
        "description": "The workbook's sheets, in order.",
    })
}

/// How `create` declares one field of a sheet (see [`sheets_arg`]).
fn sheet_field(name: &str) -> Value {
    match name {
        "name" => text_arg(&format!(
            "Sheet name: at most {} characters, none of {:?}, unique ignoring case; without one it \
             becomes Sheet<n>.",
            RULES.sheet_name_max, RULES.sheet_name_forbidden,
        )),
        "rows" => array_arg(&sheet_row_arg(), "The rows, top to bottom."),
        "merges" => array_arg(
            &text_arg("A range like A1:C1, corners in either order."),
            "Ranges to merge, each covering at least two cells; a merge never clears what its cells \
             hold.",
        ),
        other => unreachable!("every sheet field has a declaration: {other}"),
    }
}

/// A cell's value, in the one declaration `set_cell` writes and a sheet's rows
/// hold: text (a string that starts with `=` stays text), a number, a boolean,
/// or a formula object, which is written as a formula and never evaluated. A
/// number is written as the cell stores it, not as it was displayed.
fn cell_arg() -> Value {
    json!({
        "anyOf": [
            scalar_arg(),
            {
                "type": "object",
                "additionalProperties": false,
                "properties": { "formula": text_arg("The formula, without a leading =.") },
                "required": ["formula"],
            },
        ],
        "description": "Text (a string starting with `=` stays text), a number, a boolean, or \
                        {\"formula\": \"SUM(A1:A2)\"}; a number is written as the cell stores it, \
                        not as a reading displayed it.",
    })
}

/// A sheet's row: its cells.
fn sheet_row_arg() -> Value {
    array_arg(
        &cell_arg(),
        "One row, one cell after another, left to right.",
    )
}

/// What the four run toggles mean in every family's formatting path: the one
/// statement rather than a description beside each of the twelve names.
const TOGGLE_RULE: &str = "bold, italic, underline and strike are toggles: true sets the \
                            property, false switches it off.";

/// `edits`: one edit object per family, declaring that family's `op` enum and the
/// union of the fields its ops take. The `op` description carries the op -> field
/// mapping and the family's cross-cutting rules, so no op gets a branch of its
/// own.
fn edits_arg(family: crate::ooxml::Family, fields: &Value, rules: &str) -> Value {
    let op_fields = edit_op_fields(family);
    let ops: Vec<&str> = op_fields.iter().map(|(op, _)| *op).collect();
    let mut properties = fields.clone();
    // The op -> field mapping the description states, from the one table the enum
    // and the refusal read too.
    let fields_by_op = op_fields
        .iter()
        .map(|(op, taken)| format!("{op}: {}", taken.join(", ")))
        .collect::<Vec<_>>()
        .join("; ");
    // Three statements, one per line of the op's own contract: the family's
    // cross-cutting rules, what its toggles mean, and which fields each op takes.
    let statements = [
        rules,
        TOGGLE_RULE,
        &format!("Fields by op: {fields_by_op}."),
    ];
    let description = statements
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    properties["op"] = choice_arg(&ops, &description);
    json!({
        "type": "array",
        "minItems": 1,
        "maxItems": RULES.edits_max,
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": properties,
            "required": ["op"],
        },
        "description": format!(
            "The edits, in order, at most {}. Applied whole or not at all: an edit whose fields \
             do not fit its op is refused by name rather than partly applied.",
            RULES.edits_max
        ),
    })
}

/// The schema's `oneOf` entry for one action, by name: what it does (one line,
/// with the refusal a caller would otherwise learn by making the call), the
/// arguments it requires and exactly the arguments it takes.
#[expect(clippy::too_many_lines)] // the one place every action's arguments are stated
fn action_schema(action: &str) -> Value {
    match action {
        "create" => super::action_entry_schema(
            "create",
            "Write a new file: blocks for docx/pptx/pdf, sheets for xlsx. A PDF is laid out as a \
             simple stream: blocks follow one another down the page, and columns, floating objects \
             and exact page placement are not attempted. A .docx may carry a running header and \
             footer, page numbers, a table of contents over its headings, links, footnotes and \
             explicit page breaks.",
            &["format"],
            &json!({
                "format": choice_arg(&SUPPORTED_FORMATS, "The format to write."),
                "content": content_arg(),
                "sheets": sheets_arg(),
                "metadata": metadata_entry(),
                "size": choice_arg(
                    &pptx_size_words(),
                    &format!(
                        "A pptx slide size: {}. Unnamed is the {} deck this tool has always \
                         written.",
                        pptx_size_list(),
                        RULES.pptx_default_size.inches()
                    ),
                ),
                "header": text_arg(
                    "A running header, one paragraph per line: {page} is the page number and \
                     {pages} the document's page count.",
                ),
                "footer": text_arg(
                    "A running footer, one paragraph per line: {page} is the page number and \
                     {pages} the document's page count.",
                ),
            }),
        ),
        "fill_template" => super::action_entry_schema(
            "fill_template",
            "Fill a copy of the user's own .docx/.pptx/.xlsx sample's placeholders.",
            &["path", "values"],
            &json!({ "path": path_arg(), "values": values_arg("placeholder") }),
        ),
        "docx_edit" => super::action_entry_schema(
            "docx_edit",
            "Edit a .docx/.docm; a .docx package is required. Rewrite and format its text, and \
             add paragraphs, pictures and tables.",
            &["path", "edits"],
            &json!({
                "path": path_arg(),
                "edits": edits_arg(
                    crate::ooxml::Family::Docx,
                    &json!({
                        "find": text_arg("Text to find."),
                        "replace": text_arg("Text to write."),
                        "insert": text_arg("Text to insert."),
                        "position": choice_arg(&["before", "after"], "Where it lands."),
                        "text": text_arg("The paragraph's text."),
                        "after": text_arg("The paragraph to land after."),
                        "bold": toggle_arg(),
                        "italic": toggle_arg(),
                        "underline": toggle_arg(),
                        "strike": toggle_arg(),
                        "size": number_arg(&format!(
                            "Font size, {} points.",
                            RULES.text_size_points.bounds()
                        )),
                        "color": color_arg("Text colour"),
                        "align": choice_arg(
                            &pair_words(&RULES.paragraph_alignments),
                            "Paragraph alignment."
                        ),
                        "style": text_arg(&format!(
                            "A paragraph style the document defines, at most {} characters; \
                             Normal, Heading 1, Heading 2 and Heading 3 always work.",
                            RULES.paragraph_style_max
                        )),
                        "indent_left": number_arg(&format!(
                            "Left indent, {} points.",
                            RULES.paragraph_indent_points.bounds()
                        )),
                        "indent_right": number_arg(&format!(
                            "Right indent, {} points.",
                            RULES.paragraph_indent_points.bounds()
                        )),
                        "indent_first_line": number_arg(&format!(
                            "First-line indent, {} points; a negative one hangs.",
                            RULES.paragraph_indent_points.bounds()
                        )),
                        "spacing_before": number_arg(&format!(
                            "Space above, {} points.",
                            RULES.paragraph_spacing_points.bounds()
                        )),
                        "spacing_after": number_arg(&format!(
                            "Space below, {} points.",
                            RULES.paragraph_spacing_points.bounds()
                        )),
                        "line_spacing": number_arg(&format!(
                            "Line height, {} points.",
                            RULES.paragraph_line_points.bounds()
                        )),
                        "list": list_arg(),
                        "level": integer_arg(&format!("List level, 0 to {}.", RULES.paragraph_level_max)),
                        "image": image_arg(),
                        "width": picture_width_arg("Picture width", BLOCK_IMAGE_FIT),
                        "height": placed_size_arg("Picture height"),
                        "headers": array_arg(
                            &scalar_arg(),
                            &format!(
                                "add_table: the headings row; the table holds at most {} columns, \
                                 headings included.",
                                RULES.table_columns_max
                            ),
                        ),
                        "rows": array_arg(
                            &array_arg(&scalar_arg(), "A row of cells."),
                            &format!(
                                "add_table: the table's rows, each an array of cells — a table \
                                 holds at most {} columns; a headings row alone makes a table.",
                                RULES.table_columns_max
                            ),
                        ),
                    }),
                    "format_text and format_paragraph need one property beyond find. add_image and \
                     add_table land after the first paragraph that holds their after text — a \
                     table cell's anchor means after the whole table — and are refused, never \
                     appended, when no paragraph holds it.",
                ),
            }),
        ),
        "xlsx_edit" => super::action_entry_schema(
            "xlsx_edit",
            "Edit an .xlsx/.xlsm; an .xlsx package is required. Rewrite cells and formatting, shift \
             rows and columns, merge and unmerge a range, add, rename, move or delete a sheet, and \
             place a picture in a sheet.",
            &["path", "edits"],
            &json!({
                "path": path_arg(),
                "edits": edits_arg(
                    crate::ooxml::Family::Xlsx,
                    &json!({
                        "sheet": text_arg("Sheet name, as a reading prints it."),
                        "name": text_arg(&format!(
                            "A sheet name: at most {} characters, none of {:?}, unique ignoring \
                             case.",
                            RULES.sheet_name_max, RULES.sheet_name_forbidden
                        )),
                        "to": integer_arg(
                            "The 1-based position the sheet takes, counting the sheets as a reading \
                             prints them."
                        ),
                        "cell": text_arg("A1 address, like B7."),
                        "value": cell_arg(),
                        "number_format": text_arg(&format!(
                            "An Excel format code, at most {} characters.",
                            RULES.number_format_max
                        )),
                        "range": text_arg(&format!(
                            "Cells, like B2:D2, corners in either order; a format_cells range \
                             holds at most {} cells.",
                            RULES.format_cells_max
                        )),
                        "column": text_arg(&format!(
                            "Column letters, up to the {}th column.",
                            RULES.sheet_column_max
                        )),
                        "row": integer_arg(&format!(
                            "The 1-based row, up to {}.",
                            RULES.sheet_row_max
                        )),
                        "width_chars": number_arg(&format!(
                            "Column width, {} characters (as Excel shows one).",
                            RULES.column_width_chars.bounds()
                        )),
                        "height_points": number_arg(&format!(
                            "Row height, {} points.",
                            RULES.row_height_points.bounds()
                        )),
                        "font": text_arg(&format!(
                            "Font name, at most {} characters.",
                            RULES.font_name_max
                        )),
                        "size": number_arg(&format!(
                            "Font size, {} points.",
                            RULES.cell_font_points.bounds()
                        )),
                        "bold": toggle_arg(),
                        "italic": toggle_arg(),
                        "underline": toggle_arg(),
                        "strike": toggle_arg(),
                        "color": color_arg("Text colour"),
                        "fill": fill_arg(),
                        "border": border_arg(),
                        "align": choice_arg(&RULES.cell_alignments, "Horizontal alignment."),
                        "vertical": choice_arg(&RULES.cell_verticals, "Vertical alignment."),
                        "wrap": bool_arg("Wrap the text."),
                        "image": image_arg(),
                        "width": picture_width_arg(
                            "Picture width",
                            "with neither side named the picture keeps its own size.",
                        ),
                        "height": placed_size_arg("Picture height"),
                    }),
                    "An edit names one sheet unless it adds one. One target only: range, column or \
                     row, with at least one of its own properties. An inserted row or column may be \
                     one past the sheet's used range; a deleted one must be inside it.",
                ),
            }),
        ),
        "pptx_edit" => super::action_entry_schema(
            "pptx_edit",
            "Edit a .pptx/.pptm; a slide is its 1-based number.",
            &["path", "edits"],
            &json!({
                "path": path_arg(),
                "edits": edits_arg(
                    crate::ooxml::Family::Pptx,
                    &json!({
                        "slide": integer_arg("Slide number, as a reading prints it, 1-based."),
                        "find": text_arg("Text to find on the slide."),
                        "replace": text_arg("Text to write."),
                        "text": text_arg("The text to write."),
                        "after": {
                            "type": ["string", "integer"],
                            "description": "add_paragraph: the text it follows; add_slide: the slide number.",
                        },
                        "level": integer_arg(&format!("Outline level, 0 to {}.", RULES.paragraph_level_max)),
                        "bold": toggle_arg(),
                        "italic": toggle_arg(),
                        "underline": toggle_arg(),
                        "strike": toggle_arg(),
                        "size": number_arg(&format!(
                            "Font size, {} points.",
                            RULES.slide_text_size_points.bounds()
                        )),
                        "color": color_arg("Text colour"),
                        "align": choice_arg(&pair_words(&RULES.slide_alignments), "Alignment."),
                        "image": image_arg(),
                        "headers": array_arg(
                            &scalar_arg(),
                            "add_table: the headings row; a presentation's table has no column bound.",
                        ),
                        "rows": array_arg(
                            &array_arg(&scalar_arg(), "A row of cells."),
                            "add_table: the table's rows, each an array of cells — a headings row alone makes a table.",
                        ),
                        "x_fraction": number_arg(&format!(
                            "Across the slide, {} of its width; an unnamed place is the slide's \
                             middle, an unnamed size a picture's largest fit or a table's own, and \
                             a place named without a size can push either past the slide's edge.",
                            RULES.slide_position_fraction.bounds()
                        )),
                        "y_fraction": number_arg(&format!(
                            "Down the slide, {} of its height.",
                            RULES.slide_position_fraction.bounds()
                        )),
                        "width_fraction": number_arg(&format!(
                            "Width, {} of the slide's width.",
                            RULES.slide_size_fraction.bounds()
                        )),
                        "height_fraction": number_arg(&format!(
                            "Height, {} of the slide's height.",
                            RULES.slide_size_fraction.bounds()
                        )),
                        "title": text_arg("The slide's title."),
                        "bullets": array_arg(
                            &scalar_arg(),
                            &format!("The slide's bullets, at most {}.", RULES.bullets_max),
                        ),
                        "to": integer_arg("The slide it moves to."),
                    }),
                    "format_text needs one property beyond find.",
                ),
            }),
        ),
        "pdf_merge" => super::action_entry_schema(
            "pdf_merge",
            "Merge PDFs into one new file.",
            &["paths"],
            &json!({ "paths": paths_arg(), "metadata": metadata_entry() }),
        ),
        "pdf_split" => super::action_entry_schema(
            "pdf_split",
            "Split a PDF, one output file per page entry.",
            &["path", "pages"],
            &json!({ "path": path_arg(), "pages": pages_arg(true), "metadata": metadata_entry() }),
        ),
        "pdf_rotate" => super::action_entry_schema(
            "pdf_rotate",
            "Add a rotation to a PDF's pages.",
            &["path", "degrees"],
            &json!({ "path": path_arg(), "degrees": degrees_arg(), "pages": pages_arg(false),
                     "metadata": metadata_entry() }),
        ),
        "pdf_crop" => super::action_entry_schema(
            "pdf_crop",
            "Crop a PDF page's visible area.",
            &["path", "area"],
            &json!({ "path": path_arg(), "area": page_area_arg(), "pages": pages_arg(false),
                     "metadata": metadata_entry() }),
        ),
        "pdf_resize" => super::action_entry_schema(
            "pdf_resize",
            "Resize a PDF page, scaling its content to fit.",
            &["path", "width", "height"],
            &json!({
                "path": path_arg(),
                "width": number_arg(&format!(
                    "Page width, {} points.",
                    RULES.dimension_points.bounds()
                )),
                "height": number_arg(&format!(
                    "Page height, {} points.",
                    RULES.dimension_points.bounds()
                )),
                "pages": pages_arg(false),
                "metadata": metadata_entry(),
            }),
        ),
        "pdf_text" => super::action_entry_schema(
            "pdf_text",
            "Draw text on a PDF's pages, as page content rather than an annotation.",
            &["path", "text"],
            &json!({
                "path": path_arg(),
                "text": text_arg("The text to draw."),
                "pages": pages_arg(false),
                "x": point_arg("X position on the page as it reads, from its left edge"),
                "y": point_arg("Y position on the page as it reads, from its bottom edge"),
                "size": number_arg(&format!(
                    "Font size, {} points.",
                    RULES.dimension_points.bounds()
                )),
                "color": color_arg("Text colour"),
                "stamp": bool_arg("Draw a box around the text."),
                "opacity": number_arg(&format!(
                    "Opacity of the drawn text and its stamp box, {}; unstated draws them solid.",
                    RULES.pdf_opacity.bounds()
                )),
                "rotate": number_arg(&format!(
                    "Rotation in degrees, within ±{}.",
                    RULES.pdf_point_abs_max
                )),
                "metadata": metadata_entry(),
            }),
        ),
        "pdf_image" => super::action_entry_schema(
            "pdf_image",
            "Place a PNG or JPEG on a PDF's pages, as page content rather than an annotation.",
            &["path", "image"],
            &json!({
                "path": path_arg(),
                "image": pdf_image_arg(),
                "pages": pages_arg(false),
                "x": point_arg("X position on the page as it reads, from its left edge"),
                "y": point_arg("Y position on the page as it reads, from its bottom edge"),
                "width": placed_size_arg("Placed width"),
                "height": placed_size_arg("Placed height"),
                "metadata": metadata_entry(),
            }),
        ),
        "pdf_form_fill" => super::action_entry_schema(
            "pdf_form_fill",
            "Fill a PDF's form fields into a copy.",
            &["path", "values"],
            &json!({
                "path": path_arg(),
                "values": values_arg("form field"),
                "flatten": bool_arg("Bake the fields into the page content and remove them."),
                "metadata": metadata_entry(),
            }),
        ),
        other => unreachable!("the schema lists exactly the dispatched actions: {other}"),
    }
}

/// A PDF position in points, bounded by the shared rules.
fn point_arg(what: &str) -> Value {
    number_arg(&format!(
        "{what} in points, within ±{}.",
        RULES.pdf_point_abs_max
    ))
}

/// `degrees`: the rotation a `pdf_rotate` adds.
fn degrees_arg() -> Value {
    integer_arg(&format!(
        "A multiple of {}, not a whole turn.",
        RULES.degrees_step
    ))
}

/// `area`: the fraction of what a page currently shows a `pdf_crop` keeps.
fn page_area_arg() -> Value {
    let side = |what: &str| {
        number_arg(&format!(
            "{what}, as a fraction of the page ({}).",
            RULES.pdf_page_fraction.bounds()
        ))
    };
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["x", "y", "width", "height"],
        "properties": {
            "x": side("The left edge"),
            "y": side("The top edge"),
            "width": side("The width"),
            "height": side("The height"),
        },
        "description": "The area to keep, in fractions of the page's current visible area, \
                        measured from the displayed top-left and reaching down and right.",
    })
}

impl DocumentTool {
    /// The one action dispatch both faces of the tool go through: `execute`
    /// renders it to its text (so an agent's view is unchanged) and
    /// [`Tool::execute_answer`] hands it over with the partiality flag a
    /// command-line caller reads. There is no second code path to drift from
    /// this one.
    async fn dispatch(&self, ws: &Workspace, args: Value) -> Result<super::ToolAnswer> {
        let args = normalize_call(&args)?;
        let action = super::get_str(&args, "action")?;
        // Only a real action is checked against its schema entry: an action the
        // tool does not have keeps its own refusal below rather than being
        // reported as taking no arguments.
        if ACTIONS.contains(&action) {
            require_declared_arguments(action, &args)?;
        }
        let output = super::output_directory(ws, GENERATED_DIR);
        match action {
            "create" => self.create(ws, &args, &output).await,
            "fill_template" => self.fill_template(ws, &args, &output).await,
            "docx_edit" => {
                self.edit(ws, &args, &output, crate::ooxml::Family::Docx)
                    .await
            }
            "xlsx_edit" => {
                self.edit(ws, &args, &output, crate::ooxml::Family::Xlsx)
                    .await
            }
            "pptx_edit" => {
                self.edit(ws, &args, &output, crate::ooxml::Family::Pptx)
                    .await
            }
            "pdf_merge" => self.pdf_merge(ws, &args, &output).await,
            "pdf_split" => self.pdf_split(ws, &args, &output).await,
            "pdf_rotate" => self.pdf_rotate(ws, &args, &output).await,
            "pdf_crop" => self.pdf_crop(ws, &args, &output).await,
            "pdf_resize" => self.pdf_resize(ws, &args, &output).await,
            "pdf_text" => self.pdf_text(ws, &args, &output).await,
            "pdf_image" => self.pdf_image(ws, &args, &output).await,
            "pdf_form_fill" => self.pdf_form_fill(ws, &args, &output).await,
            other => anyhow::bail!(
                "usage: unknown action \"{other}\" — hint: use {}",
                ACTIONS.join(", ")
            ),
        }
    }

    /// `create` — a fresh document from content blocks or sheets.
    async fn create(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let format = super::get_str(args, "format")?;
        if !SUPPORTED_FORMATS.contains(&format) {
            anyhow::bail!(
                "usage: unknown format \"{format}\" — hint: use {}",
                SUPPORTED_FORMATS.join(", ")
            );
        }
        let metadata = opt_metadata(args)?;
        if metadata.is_some() && format != "pdf" {
            anyhow::bail!(
                "usage: a {format} file carries no PDF description — hint: pass \"metadata\" only \
                 for format \"pdf\""
            );
        }
        let mut request = json!({ "op": "create", "format": format });
        set_metadata(&mut request, metadata);
        // A running header or footer is a document's own, and the page numbers it
        // composes are written as fields: a presentation, a workbook and this
        // tool's PDF have none, so one named there is refused rather than dropped.
        for (key, what) in [
            ("header", "a running header"),
            ("footer", "a running footer"),
        ] {
            let Some(text) = super::get_opt_str_strict(args, key)? else {
                continue;
            };
            if format != "docx" {
                anyhow::bail!(
                    "usage: \"{key}\" is {what}, and only a .docx has one — hint: leave it out \
                     for format \"{format}\""
                );
            }
            // A blank one is a header the reader never sees, written as an empty
            // part of a call that reports success — the same rule a blank link text
            // and a blank footnote are refused by.
            require_visible(
                &format!("\"{key}\""),
                text,
                &format!("give its text, or leave \"{key}\" out"),
            )?;
            request[key] = json!(text);
        }
        // A slide size is a presentation's own: the page a document or a PDF is
        // laid out on is not choosable, so a `size` named for one is refused
        // rather than silently dropped.
        let size = super::get_opt_str_strict(args, "size")?;
        if let Some(size) = size {
            if format != "pptx" {
                anyhow::bail!(
                    "usage: \"size\" is a slide size, and only format \"pptx\" has slides — \
                     hint: pass \"size\" only for format \"pptx\""
                );
            }
            let sizes = pptx_size_words();
            if !sizes.contains(&size) {
                anyhow::bail!(
                    "usage: unknown slide size \"{size}\" — hint: use {}",
                    listed(&sizes)
                );
            }
            request["size"] = json!(size);
        }
        if format == "xlsx" {
            // A workbook is built from `sheets`; content blocks passed beside it
            // would be dropped in silence, which is worse than being told they do
            // not apply here.
            if args.get("content").is_some_and(|value| !value.is_null()) {
                anyhow::bail!(
                    "usage: \"content\" is not used for format \"xlsx\" — hint: pass \"sheets\" to \
                     write a spreadsheet"
                );
            }
            request["sheets"] = resolve_sheets(args)?;
        } else {
            if args.get("sheets").is_some_and(|value| !value.is_null()) {
                anyhow::bail!(
                    "usage: \"sheets\" is not used for format \"{format}\" — hint: pass \"content\" \
                     to write a {format} document"
                );
            }
            request["content"] = resolve_content(ws, args, format, self.access).await?;
        }
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &format!("{}_{}", default_base(format), crate::util::unix_millis()),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), format).await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "created").await
    }

    /// `fill_template` — a copy of the user's own sample with `{name}`
    /// placeholders replaced.
    async fn fill_template(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (sample, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let values = require_values(args, "placeholder")?;
        require_scalar_values(&values)?;
        // The kit's `format` is the sample's FAMILY: a workbook has its own
        // cell-wise filler, while a document and a presentation both go through
        // the template library (which reads the package's file type itself). The
        // name picks it, and the sample's own bytes must hold it — a sample of
        // another family, an old binary one or an encrypted package is refused
        // here, before a runtime is spawned.
        let family = require_sample_family(&sample).await?;
        let extension = copy_extension(&sample, family);
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&sample, "filled"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), &extension).await?;
        let request = json!({
            "op": "fill_template",
            // The kit's own field name for the sample it fills.
            "template": sample.to_string_lossy(),
            "format": family.name(),
            "values": values,
            "output": outputs[0].to_string_lossy(),
        });
        self.deliver(request, &outputs, "filled the sample into")
            .await
    }

    /// `docx_edit`/`xlsx_edit`/`pptx_edit` — a surgical edit of an existing
    /// package of one family, into a new file.
    async fn edit(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
        family: crate::ooxml::Family,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        // The file's own bytes decide what it is, never its name: the input must
        // really be the package family this action edits — a package of another
        // family, a PDF, a container that is no document and an old binary
        // `.doc`/`.xls`/`.ppt` are each refused naming what they are, and a
        // genuinely encrypted package is the password verdict whatever the name
        // claims.
        require_edit_family(&input, family).await?;
        let mut edits = validate_edits(family, args)?;
        // Every family's `add_image` places a picture, so its path is resolved
        // before the kit is handed it; only a presentation also needs the diagram
        // text a refusal about the slide's own text may name. What a slide's
        // diagrams show is the reader's rule, so the text is read from the input
        // — under the conversion bound, since it walks a whole deck — and the kit
        // looks it up rather than restating how a diagram is read. An edit that
        // never looks text up on the slide (a notes edit, a slide reshape) reads
        // no diagram text at all.
        let mut diagram_text = HashMap::new();
        resolve_edit_images(ws, &mut edits, self.access).await?;
        if family == crate::ooxml::Family::Pptx && edits_search_slide_text(&edits) {
            diagram_text = crate::document::read_pptx_diagram_text(&input).await;
        }
        let extension = copy_extension(&input, family);
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "edited"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), &extension).await?;
        let mut request = json!({
            "op": format!("{}_edit", family.name()),
            "input": input.to_string_lossy(),
            "edits": edits,
            "output": outputs[0].to_string_lossy(),
        });
        // A deck whose slides show no diagram text sends no field: there is
        // nothing a refusal could name.
        if !diagram_text.is_empty() {
            request["diagram_text"] = json!(diagram_text);
        }
        self.deliver(request, &outputs, "edited").await
    }

    /// `pdf_merge` — every input's pages, in order, in one file.
    async fn pdf_merge(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let paths = super::get_str_array(args, "paths")?;
        let metadata = opt_metadata(args)?;
        if paths.len() < 2 {
            anyhow::bail!(
                "usage: `paths` needs at least 2 PDFs to merge, got {} — hint: pass two or \
                 more PDF paths",
                paths.len()
            );
        }
        if paths.len() > MAX_MERGE_INPUTS {
            anyhow::bail!(
                "usage: `paths` holds {} files, more than one merge takes (limit \
                 {MAX_MERGE_INPUTS}) — hint: merge them in batches",
                paths.len()
            );
        }
        let mut inputs = Vec::with_capacity(paths.len());
        // The kit decodes every input into memory at once, so the size of the
        // whole call is bounded, not just of one file.
        let mut total = 0u64;
        for path in &paths {
            let (resolved, size) = resolve_input(ws, path, self.access).await?;
            total += size;
            inputs.push(resolved);
        }
        if total > MAX_MERGE_BYTES {
            anyhow::bail!(
                "usage: `paths` totals {} MB, more than one merge takes (limit {} MB) — hint: \
                 merge them in batches",
                megabytes(total),
                megabytes(MAX_MERGE_BYTES)
            );
        }
        // Every argument is read before anything is reserved, and the content
        // verdict comes with them: a merge of anything but PDFs is refused here,
        // naming what each input really holds.
        for input in &inputs {
            require_pdf(input).await?;
        }
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&inputs[0], "merged"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        let mut request = json!({
            "op": "pdf_merge",
            "inputs": path_strings(&inputs),
            "output": outputs[0].to_string_lossy(),
        });
        set_metadata(&mut request, metadata);
        self.deliver(request, &outputs, "merged the files into")
            .await
    }

    /// `pdf_split` — one file per page selection.
    async fn pdf_split(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let groups = split_groups(args)?;
        let metadata = opt_metadata(args)?;
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            crate::util::name_stem(&file_name_of(&input)),
        );
        let bases: Vec<String> = (0..groups.len())
            .map(|index| format!("{base}_part{}", index + 1))
            .collect();
        let outputs = reserve(output, &bases, "pdf").await?;
        let mut request = json!({
            "op": "pdf_split",
            "input": input.to_string_lossy(),
            "groups": groups,
            "outputs": path_strings(&outputs),
        });
        set_metadata(&mut request, metadata);
        self.deliver(request, &outputs, "split the file into").await
    }

    /// `pdf_rotate` — add a rotation to the selected pages.
    async fn pdf_rotate(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let degrees = match super::get_opt_i64(args, "degrees")? {
            // Reduced here, so the request carries a value bounded by the turn
            // it is: a rotation IS its angle modulo 360, and a caller can send
            // one as large as an i64 allows.
            Some(degrees) if degrees % RULES.degrees_step == 0 && degrees.rem_euclid(360) != 0 => {
                degrees.rem_euclid(360)
            }
            Some(degrees) if degrees % RULES.degrees_step == 0 => anyhow::bail!(
                "usage: {degrees} degrees would leave the pages as they are — hint: use a \
                 multiple of {} below a full turn",
                RULES.degrees_step
            ),
            Some(_) => anyhow::bail!(
                "usage: \"degrees\" must be a multiple of {}",
                RULES.degrees_step
            ),
            None => anyhow::bail!(
                "usage: missing required argument \"degrees\" — hint: pass a multiple of {} below a \
                 full turn",
                RULES.degrees_step
            ),
        };
        // Every argument is checked before anything is reserved: a call the tool
        // refuses must not leave an empty output file behind.
        let mut request = json!({
            "op": "pdf_rotate",
            "input": input.to_string_lossy(),
            "degrees": degrees,
        });
        request["pages"] = opt_pages(args)?.kit_value();
        set_metadata(&mut request, opt_metadata(args)?);
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "rotated"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "rotated").await
    }

    /// `pdf_crop` — narrow the visible area of the selected pages.
    async fn pdf_crop(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let area = require_page_area(args)?;
        let mut request = json!({
            "op": "pdf_crop",
            "input": input.to_string_lossy(),
            "area": area,
        });
        request["pages"] = opt_pages(args)?.kit_value();
        set_metadata(&mut request, opt_metadata(args)?);
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "cropped"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "cropped").await
    }

    /// `pdf_resize` — scale the selected pages to a named size.
    async fn pdf_resize(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let mut request = json!({
            "op": "pdf_resize",
            "input": input.to_string_lossy(),
        });
        set_positive_number(&mut request, args, "width", &RULES.dimension_points)?;
        set_positive_number(&mut request, args, "height", &RULES.dimension_points)?;
        // A side left out is not "keep the page's own": the kit has no such
        // reading of the request, and a resize always names the page it makes.
        if let Some(missing) = ["width", "height"]
            .into_iter()
            .find(|side| request.get(*side).is_none())
        {
            anyhow::bail!(
                "usage: missing required argument \"{missing}\" — a resize names both sides — \
                 hint: give both \"width\" and \"height\" in points"
            );
        }
        request["pages"] = opt_pages(args)?.kit_value();
        set_metadata(&mut request, opt_metadata(args)?);
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "resized"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "resized").await
    }

    /// `pdf_text` — draw text (optionally stamped and rotated) on the pages.
    async fn pdf_text(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let text = super::get_str(args, "text")?;
        let mut request = json!({
            "op": "pdf_text",
            "input": input.to_string_lossy(),
            "text": text,
        });
        request["pages"] = opt_pages(args)?.kit_value();
        set_number(&mut request, args, "x")?;
        set_number(&mut request, args, "y")?;
        set_positive_number(&mut request, args, "size", &RULES.dimension_points)?;
        set_bounded_number(
            &mut request,
            args,
            "opacity",
            &RULES.pdf_opacity,
            "give a fraction just above zero",
        )?;
        set_number(&mut request, args, "rotate")?;
        if let Some(color) = super::get_opt_str_strict(args, "color")? {
            // A pdf argument is named quoted, as the neighbouring refusals do.
            require_hex_color("\"color\"", color)?;
            request["color"] = json!(color);
        }
        if let Some(stamp) = super::get_opt_bool(args, "stamp")? {
            request["stamp"] = json!(stamp);
        }
        set_metadata(&mut request, opt_metadata(args)?);
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "text"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "stamped text into").await
    }

    /// `pdf_image` — place a PNG/JPEG on the pages.
    async fn pdf_image(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let (image, _) = resolve_input(ws, super::get_str(args, "image")?, self.access).await?;
        ensure_picture(&image, &pdf_pictures())?;
        let mut request = json!({
            "op": "pdf_image",
            "input": input.to_string_lossy(),
            "image": image.to_string_lossy(),
        });
        request["pages"] = opt_pages(args)?.kit_value();
        set_number(&mut request, args, "x")?;
        set_number(&mut request, args, "y")?;
        set_positive_number(&mut request, args, "width", &RULES.dimension_points)?;
        set_positive_number(&mut request, args, "height", &RULES.dimension_points)?;
        set_metadata(&mut request, opt_metadata(args)?);
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "image"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "inserted the image into")
            .await
    }

    /// `pdf_form_fill` — fill a PDF's AcroForm fields.
    async fn pdf_form_fill(
        &self,
        ws: &Workspace,
        args: &Value,
        output: &Path,
    ) -> Result<super::ToolAnswer> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?, self.access).await?;
        let values = require_values(args, "form field")?;
        require_scalar_values(&values)?;
        // Every argument is read before anything is reserved: a call the tool
        // refuses must not leave an empty output file behind.
        let flatten = super::get_bool(args, "flatten", false)?;
        let metadata = opt_metadata(args)?;
        require_pdf(&input).await?;
        let base = sanitize_base(
            super::get_opt_str_strict(args, "file_name")?,
            &input_base(&input, "filled"),
        );
        let outputs = reserve(output, std::slice::from_ref(&base), "pdf").await?;
        let mut request = json!({
            "op": "pdf_form_fill",
            "input": input.to_string_lossy(),
            "values": values,
            "flatten": flatten,
            "output": outputs[0].to_string_lossy(),
        });
        set_metadata(&mut request, metadata);
        self.deliver(request, &outputs, "filled the form into")
            .await
    }

    /// Run one prepared operation and turn its result into the reply: verify
    /// every reserved output exists, describe what was produced, and list the
    /// `[FILE:…]` markers. A failed run removes the reservations so no empty
    /// placeholder is left behind.
    async fn deliver(
        &self,
        request: Value,
        outputs: &[PathBuf],
        prefix: &str,
    ) -> Result<super::ToolAnswer> {
        // The runtime's absence and a failing probe are mahbot-side faults the
        // model cannot fix; `probe` reports them with that token, so a broken
        // runtime never looks like a broken request. It runs here — once the
        // call's own arguments have been checked and its outputs reserved — so a
        // malformed call is refused on its own merits whatever state the runtime
        // is in, and a probe failure still cleans up the reservations.
        if let Err(e) = docgen::probe().await {
            remove_files(outputs).await;
            return Err(e);
        }
        let outcome = match docgen::run(request).await {
            Ok(outcome) => outcome,
            Err(e) => {
                remove_files(outputs).await;
                return Err(e);
            }
        };
        let mut sizes = Vec::with_capacity(outputs.len());
        for path in outputs {
            // The kit echoes the paths it wrote; a reservation it did not report
            // back is a kit that wrote somewhere else, so nothing is delivered.
            if !outcome
                .outputs
                .iter()
                .any(|reported| reported.as_path() == path.as_path())
            {
                remove_files(outputs).await;
                return Err(super::internal_fault(&format!(
                    "the document kit did not report writing {}",
                    path.display()
                )));
            }
            match tokio::fs::metadata(path).await {
                Ok(meta) if meta.len() > 0 => sizes.push(meta.len()),
                _ => {
                    remove_files(outputs).await;
                    return Err(super::internal_fault(&format!(
                        "the document kit reported success but wrote nothing to {}",
                        path.display()
                    )));
                }
            }
        }

        let names: Vec<String> = outputs.iter().map(|path| file_name_of(path)).collect();
        let mut text = describe(prefix, &names, &sizes);
        if !outcome.missing.is_empty() {
            text = super::with_note(
                &text,
                &format!(
                    "[no values for these placeholders, left unchanged: {}]",
                    outcome.missing.join(", ")
                ),
            );
        }
        // The embedded font draws an unknown character as a blank box; the kit
        // reports the distinct characters it could not draw, so the answer says
        // they are missing from the file rather than presenting them as there.
        if !outcome.unsupported.is_empty() {
            let listed = outcome
                .unsupported
                .iter()
                .take(UNSUPPORTED_NOTE_MAX)
                .map(|character| format!("'{}'", character.escape_debug()))
                .collect::<Vec<_>>()
                .join(" ");
            let more = outcome
                .unsupported
                .len()
                .saturating_sub(UNSUPPORTED_NOTE_MAX);
            let characters = if more == 0 {
                listed
            } else {
                format!("{listed} and {more} more")
            };
            text = super::with_note(
                &text,
                &format!("[missing from the file, the embedded font cannot draw: {characters}]"),
            );
        }
        // Only a template fill can say this, and the kit's own count is what
        // makes it a fact: a copy cannot be compared against its sample to learn
        // whether anything was substituted (the kit re-serialises parts). The
        // count covers only the parts the filler rewrites, so a placeholder in a
        // chart label, a comment or a footnote is outside it and the note says
        // that rather than claiming the sample holds none.
        if outcome.placeholders == Some(0) {
            text = super::with_note(
                &text,
                "[no {name} placeholders in the parts this operation rewrites — nothing was \
                 substituted; the copy is unchanged]",
            );
        }
        // Each note is the kit's own caveat about the input's own properties or
        // about how it carried a part over — a chart or pivot whose cached value
        // goes stale, a range the delete covered, a part left naming the old cells
        // — already a bracketed, user-facing sentence, so it is appended verbatim.
        // Whether a note means part of the work is MISSING is the kit's own
        // `lossy` flag and never this text (see the partiality below): a note
        // about what the input already carried is information, not a shortfall.
        for note in &outcome.notes {
            text = super::with_note(&text, note);
        }
        for path in outputs {
            text.push('\n');
            text.push_str(&self.format_media_result(path));
        }
        // `missing` (placeholders left unchanged), `unsupported` (characters the
        // font cannot draw), `placeholders == Some(0)` (nothing was substituted)
        // and `outcome.lossy` (the kit's own losses) are the statements that part
        // of what was asked is not in the produced file.
        let partial = outcome.lossy
            || !outcome.missing.is_empty()
            || !outcome.unsupported.is_empty()
            || outcome.placeholders == Some(0);
        Ok(if partial {
            super::ToolAnswer::partial(text)
        } else {
            super::ToolAnswer::done(text)
        })
    }
}

/// Resolve a model-supplied input path under `access` and require it to be a
/// regular file the kit can load whole. The file's length is returned alongside
/// the path so a caller never stats the same file a second time.
async fn resolve_input(ws: &Workspace, path: &str, access: PathAccess) -> Result<(PathBuf, u64)> {
    let resolved = super::path::resolve_read_target(ws.as_path(), path, access).await?;
    let meta = tokio::fs::metadata(&resolved)
        .await
        .with_context(|| format!("cannot read {}", resolved.display()))?;
    if !meta.is_file() {
        anyhow::bail!(
            "usage: {} is not a file — hint: pass the path of a file",
            resolved.display()
        );
    }
    // The product's own conversion limit, not a transfer limit: the kit loads an
    // input whole. Without it an oversized file is read whole by the kit and ends
    // as a killed run — a caller's own file reported as a mahbot-side fault.
    if meta.len() > crate::util::FILE_MAX_BYTES {
        anyhow::bail!(
            "usage: {} is over the {} MB limit for a document operation, got {} — hint: shrink \
             the file and retry",
            resolved.display(),
            megabytes(crate::util::FILE_MAX_BYTES),
            super::listing::human_readable_size(meta.len())
        );
    }
    Ok((resolved, meta.len()))
}

/// The leading bytes of `path` — the five-byte PDF magic [`real_family_of`]
/// settles a PDF with, without reading a whole file for it.
async fn head_of(path: &Path) -> Result<[u8; 8]> {
    use tokio::io::AsyncReadExt as _;
    let mut magic = [0u8; 8];
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?;
    // A file shorter than the magic reads what it has and is not a container; the
    // bytes it could not read stay zero.
    let _ = file.read(&mut magic).await;
    Ok(magic)
}

/// What the file at `path` really is, from its own bytes — never its name. The
/// head settles everything a container is not needed for — a PDF's five-byte
/// magic, and bytes that are no document container at all — so a caller's own
/// PDF, the common and heaviest case, is never read whole. A container's own
/// directory lies further in, so a container goes to [`container_family`].
async fn real_family_of(path: &Path) -> Result<crate::document::RealFamily> {
    let head = head_of(path).await?;
    if !crate::document::is_zip_container(&head) && !head.starts_with(crate::document::CFB_MAGIC) {
        return Ok(crate::document::real_family(&head));
    }
    let owned = path.to_owned();
    // The whole-file read and the legacy parse run on a blocking thread under
    // the document bound (see `crate::document::run_document_parse`): the parse
    // is a third-party reader, so its panic must cost the call rather than the
    // caller's turn.
    match crate::document::run_document_parse(move || container_family(&owned)).await {
        Some(checked) => checked,
        None => anyhow::bail!(
            "usage: {} could not be read — hint: retry it, or check the file is intact",
            file_name_of(path)
        ),
    }
}

/// What a container at `path` really holds, from its own bytes: the whole-file
/// read ([`crate::util::FILE_MAX_BYTES`], already checked by [`resolve_input`])
/// and the family parse that has to see past the head.
fn container_family(path: &Path) -> Result<crate::document::RealFamily> {
    use crate::document::RealFamily;
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let real = crate::document::real_family(&bytes);
    // A CFB container whose own directory names no Office document leaves the
    // name as the only evidence, and the reading path answers such a file the
    // same way (`crate::document::classify`): an old-format name makes it that
    // old format's file, while a package's name stays the package's own
    // corruption verdict (`real_clause` reads the name for that).
    let real = match real {
        RealFamily::Other if bytes.starts_with(crate::document::CFB_MAGIC) => {
            crate::legacy::named_family(path).map_or(real, crate::legacy::Family::real)
        }
        other => other,
    };
    // A legacy `.doc`/`.xls`/`.ppt` can be genuinely encrypted, which its
    // container alone does not say: only the family's own reader can tell a
    // password from corruption, so a legacy family that is really encrypted is
    // answered with the password verdict here. `crate::document::real_family`
    // itself stays pure — the reading path asks it about every old document and
    // must not pay a second parse for a verdict only the write path needs.
    if let Some(RealFormat::Legacy(family)) = RealFormat::of(real)
        && crate::legacy::is_encrypted(&bytes, family)
    {
        return Ok(RealFamily::Encrypted);
    }
    Ok(real)
}

/// The verdict a genuinely encrypted input gets, whatever its name says: its own
/// container declares genuine encryption, and the read path gives the same file
/// the same answer. `wanted` names what the caller would have done to the file,
/// so the refusal tells it what a password-free copy is for in its own voice.
fn password_refusal(path: &Path, wanted: Wanted) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: {} is password-protected — hint: remove the protection, save an unprotected copy \
         and {}",
        file_name_of(path),
        wanted.unprotected()
    )
}

/// The middle clause of a refusal: the file's real format when its bytes name
/// one. When they name none, the name decides: a `.docx`/`.xlsx`/`.pptx` name
/// means a package of that family that cannot be read, the same verdict a
/// reading gives such a file, and any other name means bytes that are no
/// document this tool knows.
fn real_clause(real: Option<RealFormat>, path: &Path) -> String {
    if let Some(format) = real {
        return format!("which is really {}", format.phrase());
    }
    crate::ooxml::family_of(path).map_or_else(
        || "which is not a Word, Excel, PowerPoint or PDF document".to_owned(),
        |family| format!("which is {}", crate::ooxml::corrupt_reason(family)),
    )
}

/// What a refusal names a file as: the formats a sentence can speak of. The
/// password case is answered before a format is asked for ([`wrong_family`]),
/// and a container that names no format has none ([`real_clause`] reads the
/// name for those) — so this is the one mapping from
/// [`crate::document::RealFamily`]'s variants to the three kinds every refusal
/// sentence is written against.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RealFormat {
    Pdf,
    Package(crate::ooxml::Family),
    Legacy(crate::legacy::Family),
}

impl RealFormat {
    /// The format `real` is, or `None` for a file that names none.
    fn of(real: crate::document::RealFamily) -> Option<Self> {
        use crate::document::RealFamily;
        match real {
            RealFamily::Pdf => Some(Self::Pdf),
            RealFamily::Docx => Some(Self::Package(crate::ooxml::Family::Docx)),
            RealFamily::Xlsx => Some(Self::Package(crate::ooxml::Family::Xlsx)),
            RealFamily::Pptx => Some(Self::Package(crate::ooxml::Family::Pptx)),
            RealFamily::LegacyDoc => Some(Self::Legacy(crate::legacy::Family::Doc)),
            RealFamily::LegacyXls => Some(Self::Legacy(crate::legacy::Family::Xls)),
            RealFamily::LegacyPpt => Some(Self::Legacy(crate::legacy::Family::Ppt)),
            RealFamily::Encrypted | RealFamily::Other => None,
        }
    }

    /// What it is, as a refusal names it.
    fn phrase(self) -> &'static str {
        match self {
            Self::Pdf => "a PDF",
            Self::Package(crate::ooxml::Family::Docx) => "a .docx package",
            Self::Package(crate::ooxml::Family::Xlsx) => "an .xlsx package",
            Self::Package(crate::ooxml::Family::Pptx) => "a .pptx package",
            Self::Legacy(crate::legacy::Family::Doc) => "an old .doc file",
            Self::Legacy(crate::legacy::Family::Xls) => "an old .xls file",
            Self::Legacy(crate::legacy::Family::Ppt) => "an old .ppt file",
        }
    }
}

/// What the caller's action can do with a file of its own family: the one thing
/// the advice a wrong-family refusal ends with needs beyond the file's real
/// family — which door to point the caller at, and what the old formats are
/// owed. The clause naming the action is the caller's own and `takes` is what
/// the action accepts in the short form a hint carries.
#[derive(Clone, Copy)]
enum Wanted {
    /// One of the three editable packages: another package's edit action — a
    /// caller that works on files, like `Self::Pdf`.
    Package,
    /// A sample of the user's own to fill.
    Sample,
    /// A PDF, for the actions that work on one. One takes a PDF, so a real PDF is
    /// never refused for it ([`require_pdf`] returns before the advice is asked
    /// for) and this is only ever answered as a caller that works on files.
    Pdf,
}

impl Wanted {
    /// The verb a sentence about this action uses.
    const fn verb(self) -> &'static str {
        match self {
            Self::Package => "edited",
            Self::Sample => "filled",
            Self::Pdf => "written",
        }
    }

    /// What the old formats' sentence tells the caller to convert the file to.
    const fn convert(self) -> &'static str {
        match self {
            Self::Pdf => "save it as a PDF first",
            Self::Package | Self::Sample => "save it as one of the OOXML families first",
        }
    }

    /// What a wrong-family PDF is owed: a caller that fills cannot fill pages, so
    /// it is pointed at the one action that fills a PDF's own fields, and every
    /// caller that works on files at the PDF actions.
    const fn pdf(self) -> &'static str {
        match self {
            Self::Sample => "a PDF's form fields are filled with pdf_form_fill",
            Self::Package | Self::Pdf => {
                "a PDF's own actions are pdf_merge, pdf_split, pdf_rotate, pdf_crop, pdf_resize, \
                 pdf_text, pdf_image and pdf_form_fill"
            }
        }
    }

    /// What the password refusal tells the caller to do with a password-free
    /// copy, in this action's own voice.
    const fn unprotected(self) -> &'static str {
        match self {
            Self::Package => "edit that",
            Self::Sample => "fill that",
            Self::Pdf => "convert it to a PDF",
        }
    }
}

/// The advice a wrong-family refusal ends with: another OOXML package points at
/// the action that edits it (or, for a PDF's own actions, at a PDF), an
/// old-format file keeps the sentence the old formats are owed, a PDF is what
/// the PDF actions are for, and a file that names no format is passed as what
/// the action takes. `takes` names what the caller's action accepts (`pass a
/// PDF`, say) and `wanted` what that action does to a file it does take. A caller
/// never asks about a format it takes: an edit action is never given its own
/// package, a fill is never given a package it fills, a PDF action is never given
/// a PDF.
fn wrong_family_advice(real: Option<RealFormat>, takes: &str, wanted: Wanted) -> String {
    match real {
        Some(RealFormat::Package(family)) => match wanted {
            Wanted::Package => format!("use {}_edit to edit it", family.name()),
            // A package a fill can fill is filled, so the package it is refused is
            // one whose name disagrees with its bytes: the name is the one thing
            // the file is missing.
            Wanted::Sample => "rename it to match its content".to_owned(),
            Wanted::Pdf => "convert it to a PDF first".to_owned(),
        },
        Some(RealFormat::Legacy(_)) => old_format_advice(wanted),
        Some(RealFormat::Pdf) => wanted.pdf().to_owned(),
        None => format!("pass {takes}"),
    }
}

/// The sentence the old binary formats are owed — they are read, but never
/// written — in the caller's own verb, telling it what to convert one to.
fn old_format_advice(wanted: Wanted) -> String {
    format!(
        "an old .doc/.xls/.ppt file is read but never {}; {}",
        wanted.verb(),
        wanted.convert()
    )
}

/// The refusal an input of the wrong family is owed, in the action's own voice.
/// `what` states what the caller's action takes (`docx_edit edits only a
/// .docx/.docm file`), `takes` the same in the short form a hint can carry (`a
/// .docx/.docm file`), `path` the file as the caller named it, `real` what the
/// file's own bytes really hold, and `wanted` what the action would have done to
/// a file it does take. The verdict follows the bytes: only a real old binary
/// `.doc`/`.xls`/`.ppt` is owed the old formats' sentence, and the name is
/// consulted just for the bytes that are no document at all.
fn wrong_family(
    what: &str,
    takes: &str,
    path: &Path,
    real: crate::document::RealFamily,
    wanted: Wanted,
) -> anyhow::Error {
    use crate::document::RealFamily;
    // The password verdict is owed whatever the action asked for: every
    // sentence below names a format, and an encrypted file's own format is not
    // one the caller can act on.
    if real == RealFamily::Encrypted {
        return password_refusal(path, wanted);
    }
    let format = RealFormat::of(real);
    anyhow::anyhow!(
        "usage: {what}, got {}, {} — hint: {}",
        file_name_of(path),
        real_clause(format, path),
        wrong_family_advice(format, takes, wanted)
    )
}

/// Require an edit input to be the package family its action edits, from the
/// file's own bytes: a package of another family, a PDF, a container that is no
/// document and an old binary `.doc`/`.xls`/`.ppt` are each named for what they
/// really are — the old format keeping the sentence the old formats are owed —
/// and a genuinely encrypted package is the password verdict whatever the name
/// claims.
async fn require_edit_family(path: &Path, family: crate::ooxml::Family) -> Result<()> {
    let real = real_family_of(path).await?;
    if RealFormat::of(real) == Some(RealFormat::Package(family)) {
        return Ok(());
    }
    let accepted = match family {
        crate::ooxml::Family::Docx => "a .docx/.docm file",
        crate::ooxml::Family::Xlsx => "a .xlsx/.xlsm file",
        crate::ooxml::Family::Pptx => "a .pptx/.pptm file",
    };
    Err(wrong_family(
        &format!("{}_edit edits only {accepted}", family.name()),
        accepted,
        path,
        real,
        Wanted::Package,
    ))
}

/// The family a fill must treat `sample` as: the family its name claims — which
/// picks the kit's filler, the copy's extension and the file the copy is compared
/// against — that the sample's own bytes must be. A sample whose bytes are one of
/// the three packages this tool fills but whose name says otherwise is one rename
/// away, so it is refused with the name it needs rather than with the edit advice
/// a call to fill cannot use. A sample of another family, an old binary document,
/// a PDF or no document at all is refused naming what it holds, and a genuinely
/// encrypted one is the password verdict.
async fn require_sample_family(sample: &Path) -> Result<crate::ooxml::Family> {
    let real = real_family_of(sample).await?;
    let format = RealFormat::of(real);
    if let Some(RealFormat::Package(family)) = format
        && crate::ooxml::family_of(sample) == Some(family)
    {
        return Ok(family);
    }
    let accepted = "a .docx/.docm, .pptx/.pptm or .xlsx/.xlsm sample";
    let what = if matches!(format, Some(RealFormat::Package(_))) {
        "the sample's name must be the family its bytes are".to_owned()
    } else {
        format!("a sample must be {accepted}")
    };
    Err(wrong_family(&what, accepted, sample, real, Wanted::Sample))
}

/// Require a `pdf_*` input to really be a PDF, from its own bytes: a valid PDF
/// under a non-PDF name keeps working, and a file that is not one is refused
/// naming what it really holds rather than left to the kit's parser. Its
/// sentences are the caller that works on PDFs: another package is told to
/// convert to a PDF first, and an old-format file is owed the old formats'
/// sentence with a PDF as what to convert it to.
async fn require_pdf(path: &Path) -> Result<()> {
    let real = real_family_of(path).await?;
    if real == crate::document::RealFamily::Pdf {
        return Ok(());
    }
    Err(wrong_family(
        "the input must be a PDF",
        "a PDF",
        path,
        real,
        Wanted::Pdf,
    ))
}

/// Validate and normalize the `edits` argument for one family's edit action,
/// refusing every shape the kit would fail on before a runtime is spawned: the
/// ops the family accepts, the fields each op needs, and the shared bounds in
/// `assets/docgen/rules.json`. Returns the array the request carries.
fn validate_edits(family: crate::ooxml::Family, args: &Value) -> Result<Value> {
    let names = edit_op_fields(family)
        .iter()
        .map(|(op, _)| *op)
        .collect::<Vec<_>>();
    let edits = match args.get("edits") {
        Some(Value::Array(edits)) if !edits.is_empty() => edits,
        Some(Value::Array(_)) => {
            anyhow::bail!("usage: \"edits\" is empty — hint: name at least one edit")
        }
        Some(v) => return Err(super::wrong_type("edits", "an array", v)),
        None => anyhow::bail!(
            "usage: missing required argument \"edits\" — hint: pass a JSON array of edit objects, \
             e.g. [{{\"op\": \"replace_text\", \"find\": \"…\", \"replace\": \"…\"}}]"
        ),
    };
    if edits.len() > RULES.edits_max {
        anyhow::bail!(
            "usage: {} edits is more than one call takes (limit {}) — hint: split them across \
             calls",
            edits.len(),
            RULES.edits_max
        );
    }
    let mut out = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        let at = format!("edits[{index}]");
        let Some(object) = edit.as_object() else {
            anyhow::bail!("usage: {at} must be an object — hint: give every edit an \"op\"");
        };
        let Some(op) = object.get("op").and_then(Value::as_str) else {
            anyhow::bail!("usage: {at} has no \"op\" — hint: use {}", names.join(", "));
        };
        if !names.contains(&op) {
            anyhow::bail!(
                "usage: unknown {} edit \"{op}\" — hint: use {}",
                family.name(),
                names.join(", ")
            );
        }
        out.push(validate_edit(family, edit, object, op, &at)?);
    }
    Ok(Value::Array(out))
}

/// Validate one edit and return the object the request carries: an xlsx
/// `set_cell`'s formula is normalized to its text, the way `create` normalizes a
/// cell, every other family's edit is passed through unchanged, and a `null`
/// field is dropped from all three.
fn validate_edit(
    family: crate::ooxml::Family,
    edit: &Value,
    object: &serde_json::Map<String, Value>,
    op: &str,
    at: &str,
) -> Result<Value> {
    require_taken_keys(
        object,
        at,
        &format!("{} \"{op}\"", family.name()),
        op_fields(family, op),
        &["op"],
    )?;
    let validated = match family {
        crate::ooxml::Family::Docx => {
            validate_docx_edit(object, op, at)?;
            edit.clone()
        }
        crate::ooxml::Family::Xlsx => validate_xlsx_edit(edit, object, op, at)?,
        crate::ooxml::Family::Pptx => {
            validate_pptx_edit(object, op, at)?;
            edit.clone()
        }
    };
    Ok(without_nulls(validated))
}

/// `edit` with its null-valued keys dropped. The boundary reads a `null` as
/// "absent" (every `edit_*`/`require_*` above matches `Some(Value::Null)`), but
/// the kit tests only `!== undefined`: a null left in would reach it as a value
/// — a `format_text`'s `bold: null` would strip the run's own bold where the
/// model meant to leave it, and an `after: null` would be read as a text field.
/// A null is "absent" at every level, so the drop reaches into an object-valued
/// field's own keys too: an xlsx `format_cells` `border` of `{"top": null,
/// "left": true}` is forwarded as `{"left": true}` rather than reaching the kit
/// with a side it refuses. An array is left as it is — a null inside one is an
/// element, and dropping it would change what the array says. Validation runs
/// before the keys are dropped, so a `null` where a value is required is still
/// refused as a missing one.
fn without_nulls(edit: Value) -> Value {
    match edit {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key, without_nulls(value)))
                .collect(),
        ),
        other => other,
    }
}

/// One string field of an edit, bounded by the shared `edit_text_max`.
fn edit_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<Option<&'a str>> {
    match optional_string(object.get(key), at, key)? {
        Some(text) => {
            let length = text.chars().count();
            if length > RULES.edit_text_max {
                anyhow::bail!(
                    "usage: {at}.{key} must be at most {} characters, got {length} — hint: shorten \
                     the text",
                    RULES.edit_text_max
                );
            }
            Ok(Some(text))
        }
        None => Ok(None),
    }
}

/// Require an edit's string field to be present; an empty value is still a
/// value, which is what several ops accept.
fn edit_string_present<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<&'a str> {
    match edit_string(object, key, at)? {
        Some(text) => Ok(text),
        None => {
            anyhow::bail!("usage: {at} has no \"{key}\" — hint: give the edit its \"{key}\" text")
        }
    }
}

/// Require an edit's string field to be a non-empty value.
fn edit_string_non_empty<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<&'a str> {
    match edit_string(object, key, at)? {
        Some(text) if !text.is_empty() => Ok(text),
        _ => anyhow::bail!("usage: {at}.{key} must not be empty"),
    }
}

/// Require an optional string field, when present, to be a non-empty value.
fn require_optional_text(
    object: &serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<()> {
    match edit_string(object, key, at)? {
        Some("") => anyhow::bail!("usage: {at}.{key} must not be empty"),
        _ => Ok(()),
    }
}

/// One optional whole-number field of an edit; a negative, fractional or
/// non-numeric value is refused here rather than by the kit later.
fn edit_integer(
    object: &serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<Option<u64>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            super::wrong_type(&format!("{at}.{key}"), "a non-negative integer", value)
        }),
    }
}

/// One optional number field of an edit; a non-numeric or non-finite value is
/// refused here rather than by the kit later.
fn edit_number(
    object: &serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<Option<f64>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .map(Some)
            .ok_or_else(|| super::wrong_type(&format!("{at}.{key}"), "a number", value)),
    }
}

/// The docx edit vocabulary: text ops address the joined text of a paragraph's
/// runs in the body, so the field each op needs and the shapes it accepts are
/// stated here once.
fn validate_docx_edit(object: &serde_json::Map<String, Value>, op: &str, at: &str) -> Result<()> {
    match op {
        "replace_text" => {
            edit_string_non_empty(object, "find", at)?;
            edit_string_present(object, "replace", at)?;
        }
        "insert_text" => {
            edit_string_non_empty(object, "find", at)?;
            edit_string_non_empty(object, "insert", at)?;
            require_position(object, at)?;
        }
        "remove_text" | "remove_paragraph" => {
            edit_string_non_empty(object, "find", at)?;
        }
        "format_text" => {
            edit_string_non_empty(object, "find", at)?;
            require_format(object, at)?;
        }
        "format_paragraph" => {
            edit_string_non_empty(object, "find", at)?;
            require_paragraph_format(object, at)?;
        }
        "add_paragraph" => {
            edit_string_present(object, "text", at)?;
            require_optional_text(object, "after", at)?;
        }
        "add_image" => {
            edit_string_non_empty(object, "after", at)?;
            edit_string_non_empty(object, "image", at)?;
            require_span_number(
                object,
                "width",
                &RULES.dimension_points,
                "a picture's width in points",
                at,
            )?;
            require_span_number(
                object,
                "height",
                &RULES.dimension_points,
                "a picture's height in points",
                at,
            )?;
        }
        "add_table" => {
            edit_string_non_empty(object, "after", at)?;
            require_table(object, at, Some(RULES.table_columns_max))?;
        }
        _ => unreachable!("the op was checked against the family's vocabulary"),
    }
    Ok(())
}

/// Require an `insert_text`'s optional `position` to be `after` or `before`.
fn require_position(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    match optional_string(object.get("position"), at, "position")? {
        None | Some("after" | "before") => Ok(()),
        Some(position) => anyhow::bail!(
            "usage: {at}.position must be \"after\" or \"before\", got \"{position}\""
        ),
    }
}

/// Require each named toggle, when present and not null, to be a boolean;
/// returns whether any was stated, which is what a `format_text`'s "at least
/// one property" test counts.
fn require_toggles(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
    at: &str,
) -> Result<bool> {
    let mut any = false;
    for key in keys {
        match object.get(*key) {
            None | Some(Value::Null) => {}
            Some(Value::Bool(_)) => any = true,
            Some(v) => return Err(super::wrong_type(&format!("{at}.{key}"), "a boolean", v)),
        }
    }
    Ok(any)
}

/// Require an optional numeric field, when present and not null, to sit inside
/// `span`; the refusal names `what` the number is for.
fn require_span_number(
    object: &serde_json::Map<String, Value>,
    key: &str,
    span: &Span,
    what: &str,
    at: &str,
) -> Result<()> {
    match edit_number(object, key, at)? {
        Some(number) if span.contains(number) => Ok(()),
        Some(number) => anyhow::bail!(
            "usage: {at}.{key} must be {}, got {number} — hint: {what}",
            span.bounds()
        ),
        None => Ok(()),
    }
}

/// Require an optional numeric field to sit inside `span`, and report whether it
/// was stated — the presence a `format_text`'s or a `format_cells`'s "at least
/// one property" test counts, read the way [`edit_number`] reads one.
fn require_stated_number(
    object: &serde_json::Map<String, Value>,
    key: &str,
    span: &Span,
    what: &str,
    at: &str,
) -> Result<bool> {
    require_span_number(object, key, span, what, at)?;
    Ok(stated(object, key))
}

/// Require a `format_text` to name at least one property, and each property to
/// be the shape the kit writes: a boolean toggle, or a point size inside the
/// shared bound (`<w:sz>` counts half-points and tops out at 1638 of them), or a
/// colour.
fn require_format(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let mut any = require_toggles(object, &["bold", "italic", "underline", "strike"], at)?;
    any |= require_stated_number(
        object,
        "size",
        &RULES.text_size_points,
        "give a size in points",
        at,
    )?;
    any |= require_color(object, at)?;
    if !any {
        anyhow::bail!(
            "usage: {at} (format_text) needs at least one of bold, italic, underline, strike, \
             size or color — hint: a boolean toggles bold, italic, underline or strike, size is \
             in points and color is {} hex digits",
            RULES.color_digits
        );
    }
    Ok(())
}

/// Require a `format_paragraph` to name at least one property, and each to be the
/// shape the kit writes: an alignment word, a style name, an indent, a spacing or
/// a line height inside the shared point bounds, or a list — a kind with an
/// optional level that belongs to it, or `false` taking the list away.
fn require_paragraph_format(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let indent = "give an indent in points";
    let spacing = "give spacing in points";
    let mut any = require_alignment(
        object,
        "align",
        &pair_words(&RULES.paragraph_alignments),
        at,
    )?;
    // Nothing here can check that the named style exists: the kit resolves the name
    // against the document's own styles and refuses an unknown one.
    any |= require_name(object, "style", RULES.paragraph_style_max, at)?;
    for (key, span, what) in [
        ("indent_left", &RULES.paragraph_indent_points, indent),
        ("indent_right", &RULES.paragraph_indent_points, indent),
        ("indent_first_line", &RULES.paragraph_indent_points, indent),
        ("spacing_before", &RULES.paragraph_spacing_points, spacing),
        ("spacing_after", &RULES.paragraph_spacing_points, spacing),
        (
            "line_spacing",
            &RULES.paragraph_line_points,
            "give a line height in points",
        ),
    ] {
        any |= require_stated_number(object, key, span, what, at)?;
    }
    any |= require_list(object, at)?;
    require_list_level(object, at)?;
    if !any {
        anyhow::bail!(
            "usage: {at} (format_paragraph) needs at least one of align, style, indent_left, \
             indent_right, indent_first_line, spacing_before, spacing_after, line_spacing or \
             list — hint: name the property the paragraph is to take"
        );
    }
    Ok(())
}

/// Require an optional name-shaped field — a style name, a font name, a
/// number format — to be a non-empty string of at most `max` characters, the
/// shared `edit_text_max` applied first as for any other string field. Reports
/// whether it was stated.
fn require_name(
    object: &serde_json::Map<String, Value>,
    key: &str,
    max: usize,
    at: &str,
) -> Result<bool> {
    match edit_string(object, key, at)? {
        None => Ok(false),
        Some("") => anyhow::bail!("usage: {at}.{key} must not be empty"),
        Some(name) if name.chars().count() <= max => Ok(true),
        Some(name) => anyhow::bail!(
            "usage: {at}.{key} must be at most {max} characters, got {}",
            name.chars().count()
        ),
    }
}

/// Require an optional `format_paragraph` `list`: a word the shared rules state,
/// or `false` to take the paragraph's list away. A word the writer has not, or a
/// value of another type, is refused rather than written as a list the kit cannot
/// place. Reports whether the field was stated, the `false` included.
fn require_list(object: &serde_json::Map<String, Value>, at: &str) -> Result<bool> {
    let kinds = pair_words(&RULES.paragraph_lists);
    match object.get("list") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(false)) => Ok(true),
        Some(Value::String(kind)) if kinds.contains(&kind.as_str()) => Ok(true),
        Some(Value::String(kind)) => anyhow::bail!(
            "usage: {at}.list must be {}, or false to take the paragraph's list away, got \
             \"{kind}\"",
            listed(&kinds)
        ),
        Some(value) => Err(super::wrong_type(
            &format!("{at}.list"),
            "a list kind or false",
            value,
        )),
    }
}

/// Require an optional `format_paragraph` `level` to be a whole number inside the
/// shared cap and to have a list to belong to: a level beside no `list`, or
/// beside `list: false`, names a depth there is no list at.
fn require_list_level(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let Some(level) = edit_integer(object, "level", at)? else {
        return Ok(());
    };
    if !matches!(object.get("list"), Some(Value::String(_))) {
        anyhow::bail!(
            "usage: {at}.level needs a list to belong to — hint: name \"list\" as {} beside it",
            listed(&pair_words(&RULES.paragraph_lists))
        );
    }
    if level > u64::from(RULES.paragraph_level_max) {
        anyhow::bail!(
            "usage: {at}.level must be a whole number from 0 to {}, got {level}",
            RULES.paragraph_level_max
        );
    }
    Ok(())
}

/// A xlsx `format_cells`'s one target. The three are exclusive — a cell
/// rectangle, a column (its width) and a row (its height) are different things —
/// so an edit naming two of them is refused rather than settled by a precedence
/// rule the caller cannot read.
#[derive(Clone, Copy)]
enum FormatCellsTarget {
    Range,
    Column,
    Row,
}

impl FormatCellsTarget {
    /// The key that names this target.
    fn key(self) -> &'static str {
        match self {
            Self::Range => "range",
            Self::Column => "column",
            Self::Row => "row",
        }
    }

    /// The formatting properties this target takes — the keys an edit naming
    /// nothing is refused by, and the ones an unknown key is refused against.
    fn attributes(self) -> &'static [&'static str] {
        match self {
            Self::Range => &[
                "font",
                "size",
                "bold",
                "italic",
                "underline",
                "strike",
                "color",
                "fill",
                "border",
                "align",
                "vertical",
                "wrap",
                "number_format",
            ],
            Self::Column => &["width_chars"],
            Self::Row => &["height_points"],
        }
    }

    /// Every key this target's edit takes: the sheet, the target itself and the
    /// properties it formats.
    fn keys(self) -> Vec<&'static str> {
        let mut keys = vec!["sheet", self.key()];
        keys.extend_from_slice(self.attributes());
        keys
    }
}

/// Whether an edit stated a field: a `null` is absent, the way the whole
/// boundary reads one.
fn stated(object: &serde_json::Map<String, Value>, key: &str) -> bool {
    !matches!(object.get(key), None | Some(Value::Null))
}

/// The one target a xlsx `format_cells` names.
fn require_format_cells_target(
    object: &serde_json::Map<String, Value>,
    at: &str,
) -> Result<FormatCellsTarget> {
    let named = ["range", "column", "row"]
        .into_iter()
        .filter(|key| stated(object, key))
        .collect::<Vec<_>>();
    match named.as_slice() {
        [] => anyhow::bail!(
            "usage: {at} names no target — hint: name \"range\" (a cell or a range of cells), \
             \"column\" (a column's letters, for its width) or \"row\" (a 1-based row, for its \
             height)"
        ),
        ["range"] => Ok(FormatCellsTarget::Range),
        ["column"] => Ok(FormatCellsTarget::Column),
        ["row"] => Ok(FormatCellsTarget::Row),
        both => {
            let listed = both
                .iter()
                .map(|key| format!("\"{key}\""))
                .collect::<Vec<_>>()
                .join(" and ");
            anyhow::bail!(
                "usage: {at} names more than one target ({listed}) — hint: an edit formats one of \
                 \"range\", \"column\" or \"row\""
            )
        }
    }
}

/// The corners an A1 range names — one cell, or two corners in either order —
/// inside the sheet's grid. `what` names the field in a refusal.
fn range_corners(range: &str, what: &str) -> Result<((u32, u32), (u32, u32))> {
    let mut corners = range.split(':');
    let first = corners.next().unwrap_or_default();
    let second = corners.next();
    if corners.next().is_some() {
        anyhow::bail!(
            "usage: {what} must be a cell or a rectangular range like B2:D2, got \"{range}\""
        );
    }
    let first = check_cell_address(first, what)?;
    let second = match second {
        Some(second) => check_cell_address(second, what)?,
        None => first,
    };
    Ok((first, second))
}

/// Require a xlsx `format_cells`'s `range` to be a cell (`"B2"`) or a
/// rectangular range (`"B2:D2"`) inside the sheet's grid. The corners may be
/// named in either order — the rectangle is what they enclose — and it may hold
/// at most `format_cells_max` cells, because the kit writes it out cell by cell:
/// a whole-column spelling would write a sheet-sized body rather than format the
/// cells the caller meant.
fn require_format_cells_range(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let field = format!("{at}.range");
    let range = edit_string_non_empty(object, "range", at)?;
    let ((first_row, first_column), (last_row, last_column)) = range_corners(range, &field)?;
    let rows = u64::from(last_row.abs_diff(first_row)) + 1;
    let columns = u64::from(last_column.abs_diff(first_column)) + 1;
    let cells = rows * columns;
    if cells > RULES.format_cells_max as u64 {
        anyhow::bail!(
            "usage: {field} \"{range}\" holds {cells} cells, more than one edit formats ({}) — \
             hint: name the rows and columns you mean",
            RULES.format_cells_max
        );
    }
    Ok(())
}

/// Require `range` to name one merge: two corners in either order, inside the
/// sheet's grid, covering at least two cells. Overlap is the kit's to refuse —
/// only its writer sees what is merged there.
fn require_merge_text(range: &str, what: &str) -> Result<()> {
    let ((first_row, first_column), (last_row, last_column)) = range_corners(range, what)?;
    let cells = (u64::from(last_row.abs_diff(first_row)) + 1)
        * (u64::from(last_column.abs_diff(first_column)) + 1);
    if cells < 2 {
        anyhow::bail!(
            "usage: {what} \"{range}\" covers one cell, and a merge needs at least two — hint: name \
             a range like A1:B2"
        );
    }
    Ok(())
}

/// Require a xlsx `merge_cells`'s `range` to name a merge (see
/// [`require_merge_text`]).
fn require_merge_range(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let range = edit_string_non_empty(object, "range", at)?;
    require_merge_text(range, &format!("{at}.range"))
}

/// Require an optional `color` — a `format_text`'s or a xlsx `format_cells`'s
/// font colour — to be the shape the kit parses (see [`require_hex_color`]);
/// returns whether it was stated.
fn require_color(object: &serde_json::Map<String, Value>, at: &str) -> Result<bool> {
    match optional_string(object.get("color"), at, "color")? {
        None => Ok(false),
        Some(color) => {
            require_hex_color(&format!("{at}.color"), color)?;
            Ok(true)
        }
    }
}

/// Require an optional xlsx `format_cells` `fill`: a colour, or `false` for
/// taking the cell's fill away — the shape the other named properties use for
/// "off", since no colour means "no fill". Reports whether it was stated.
fn require_fill(object: &serde_json::Map<String, Value>, at: &str) -> Result<bool> {
    match object.get("fill") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(false)) => Ok(true),
        Some(Value::Bool(true)) => anyhow::bail!(
            "usage: {at}.fill must be a colour or false — hint: false takes the cell's fill away, \
             and a fill cannot be named without its colour"
        ),
        Some(Value::String(color)) => {
            // The colour is refused with the shared message, with the one other
            // shape a `fill` takes named beside it: "none" is the spelling a
            // caller reaches for when they mean "take the fill away".
            require_hex_color(&format!("{at}.fill"), color).map_err(|color_error| {
                anyhow::anyhow!("{color_error} — or pass false to take the cell's fill away")
            })?;
            Ok(true)
        }
        Some(value) => Err(super::wrong_type(
            &format!("{at}.fill"),
            "a colour or false",
            value,
        )),
    }
}

/// Require an optional xlsx `format_cells` `border`: an object naming at least
/// one side, each a boolean — true puts a thin border on that side, false takes
/// that side's own border away. An object naming no side states nothing, and an
/// unknown side would be dropped in silence, so both are refused. Returns
/// whether the property was stated.
fn require_border(object: &serde_json::Map<String, Value>, at: &str) -> Result<bool> {
    let border = match object.get("border") {
        None | Some(Value::Null) => return Ok(false),
        Some(Value::Object(border)) => border,
        Some(value) => {
            return Err(super::wrong_type(
                &format!("{at}.border"),
                "an object of sides",
                value,
            ));
        }
    };
    let stated = border
        .iter()
        .filter(|(_, value)| !value.is_null())
        .collect::<Vec<_>>();
    if stated.is_empty() {
        anyhow::bail!(
            "usage: {at}.border names no side — hint: it takes {}",
            listed(&RULES.cell_border_sides)
        );
    }
    for (side, value) in stated {
        if !RULES.cell_border_sides.contains(side) {
            anyhow::bail!(
                "usage: {at}.border.{side} is not a side — hint: it takes {}",
                listed(&RULES.cell_border_sides)
            );
        }
        if !matches!(value, Value::Bool(_)) {
            return Err(super::wrong_type(
                &format!("{at}.border.{side}"),
                "a boolean",
                value,
            ));
        }
    }
    Ok(true)
}

/// Require an optional alignment field to be one the writer has — a word the
/// caller passes in from the shared rules, whose values are the alignment
/// tokens themselves. Returns whether it was stated.
fn require_alignment<W: AsRef<str> + std::fmt::Display>(
    object: &serde_json::Map<String, Value>,
    key: &str,
    values: &[W],
    at: &str,
) -> Result<bool> {
    match optional_string(object.get(key), at, key)? {
        None => Ok(false),
        Some(value) if values.iter().any(|allowed| allowed.as_ref() == value) => Ok(true),
        Some(value) => anyhow::bail!(
            "usage: {at}.{key} must be {}, got \"{value}\"",
            listed(values)
        ),
    }
}

/// Refuse a key of the object at `at` that the shape `subject` names does not
/// take, naming the ones it does: the kit reads only those, so an extra key would
/// be dropped in silence. `fields` is the shape's own vocabulary and `always` the
/// keys every shape of that kind takes beyond it (`op` on an edit, `type` on a
/// block). A null value is not stated — the boundary reads a `null` as absent
/// everywhere (see [`stated`]) — so a key whose value is null is not a key the
/// shape does not take.
fn require_taken_keys(
    object: &serde_json::Map<String, Value>,
    at: &str,
    subject: &str,
    fields: &[&str],
    always: &[&str],
) -> Result<()> {
    for key in object.keys() {
        if !always.contains(&key.as_str()) && !fields.contains(&key.as_str()) && stated(object, key)
        {
            let taken = always.iter().chain(fields).copied().collect::<Vec<_>>();
            anyhow::bail!(
                "usage: {at}.{key} is not a field of {subject} — hint: it takes {}",
                listed(&taken)
            );
        }
    }
    Ok(())
}

/// Refuse a content block's key its own kind does not take, against the fields
/// [`CONTENT_BLOCKS`] states for that kind — the same table the declaration is
/// built from, so the two cannot drift.
fn require_block_keys(object: &serde_json::Map<String, Value>, at: &str, kind: &str) -> Result<()> {
    let fields = CONTENT_BLOCKS
        .iter()
        .find_map(|(name, fields, _)| (*name == kind).then_some(*fields))
        .expect("every block kind the boundary reads is one the declaration names");
    require_taken_keys(object, at, &format!("the {kind} block"), fields, &["type"])
}

/// Content only a Word document can carry — a link, a page break, a table of
/// contents or a paragraph's footnote: a presentation and the PDF this tool writes
/// as a simple stream have no part that holds one, so it is refused rather than
/// handed to a writer that would drop it — the one thing a caller must never be
/// told is that their content was written when it was not.
fn require_docx_block(format: &str, at: &str, kind: &str) -> Result<()> {
    if format == "docx" {
        return Ok(());
    }
    anyhow::bail!(
        "usage: {at} ({kind}) is content only a .docx holds — hint: leave it out of a {format} \
         create"
    )
}

/// A link's address must be one an office application can follow, and the schemes it
/// follows are named rather than refused one by one: `http`, `https`, `mailto` and
/// `tel`. Anything else — a target with no scheme, which a document writes and no
/// application opens, or a scheme that is not an address a document should hand a
/// reader — is refused with the reason and the address it got. The address the parse
/// produced is the one the document carries: parsing trims the space around it and
/// skips an embedded tab or newline, so writing the raw string could state an address
/// that is not the one given — and one no application follows. The `url` field states
/// the rule to the caller, and the kit restates it where the address is written.
fn link_address(raw: &str, index: usize) -> Result<String> {
    let Ok(url) = url::Url::parse(raw) else {
        anyhow::bail!(
            "usage: content[{index}].url must be an absolute URL, e.g. https://example.com — \
             got: {raw:?}"
        );
    };
    if !matches!(url.scheme(), "http" | "https" | "mailto" | "tel") {
        anyhow::bail!(
            "usage: content[{index}].url must be an http, https, mailto or tel address — got: \
             {raw:?}"
        );
    }
    Ok(url.to_string())
}

/// The xlsx edit vocabulary and the shapes its writer knows: a sheet is named by
/// every op but `add_sheet`, a cell is an A1 address, a row and a column are the
/// 1-based units the reader shows, a merge is a range of at least two cells, and
/// `format_cells` names one of a cell range, a column's width or a row's height
/// and the properties it applies to it.
#[expect(clippy::too_many_lines)] // the one place the xlsx op vocabulary is checked
fn validate_xlsx_edit(
    edit: &Value,
    object: &serde_json::Map<String, Value>,
    op: &str,
    at: &str,
) -> Result<Value> {
    // Ahead of the arm, so an op added later cannot forget it.
    if op != "add_sheet" {
        require_sheet_field(object, at)?;
    }
    match op {
        "set_cell" => {
            require_cell(object, at)?;
            require_name(object, "number_format", RULES.number_format_max, at)?;
            let mut normalized = edit.clone();
            normalized["value"] = require_cell_value(object, at)?;
            return Ok(normalized);
        }
        "clear_cell" => {
            require_cell(object, at)?;
        }
        "format_cells" => {
            let target = require_format_cells_target(object, at)?;
            // A property another target owns (`bold` on a column, which sets only a
            // width) would be dropped in silence, so it is refused by name.
            require_taken_keys(
                object,
                at,
                &format!("a {} target", target.key()),
                &target.keys(),
                &["op"],
            )?;
            let named = match target {
                FormatCellsTarget::Range => {
                    require_format_cells_range(object, at)?;
                    let mut named = require_toggles(
                        object,
                        &["bold", "italic", "underline", "strike", "wrap"],
                        at,
                    )?;
                    // Excel's own name length; `<name val>` carries the name verbatim.
                    named |= require_name(object, "font", RULES.font_name_max, at)?;
                    named |= require_stated_number(
                        object,
                        "size",
                        &RULES.cell_font_points,
                        "give a size in points",
                        at,
                    )?;
                    named |= require_color(object, at)?;
                    named |= require_fill(object, at)?;
                    named |= require_border(object, at)?;
                    named |= require_alignment(object, "align", &RULES.cell_alignments, at)?;
                    named |= require_alignment(object, "vertical", &RULES.cell_verticals, at)?;
                    if stated(object, "number_format") {
                        require_name(object, "number_format", RULES.number_format_max, at)?;
                        named = true;
                    }
                    named
                }
                FormatCellsTarget::Column => {
                    require_column(object, at)?;
                    require_stated_number(
                        object,
                        "width_chars",
                        &RULES.column_width_chars,
                        "give a width in characters, as Excel shows one",
                        at,
                    )?
                }
                FormatCellsTarget::Row => {
                    require_row(object, at)?;
                    require_stated_number(
                        object,
                        "height_points",
                        &RULES.row_height_points,
                        "give a height in points",
                        at,
                    )?
                }
            };
            if !named {
                anyhow::bail!(
                    "usage: {at} names no formatting — hint: name at least one of {}, beyond the \
                     target it applies to",
                    target.attributes().join(", ")
                );
            }
        }
        "insert_row" | "delete_row" => {
            require_row(object, at)?;
        }
        "insert_column" | "delete_column" => {
            require_column(object, at)?;
        }
        "merge_cells" => {
            require_merge_range(object, at)?;
        }
        "unmerge_cells" => {
            // The corners alone: a one-cell range is refused by the kit, which names
            // the merges the sheet really has, where `require_merge_text` would refuse
            // it without them.
            let field = format!("{at}.range");
            range_corners(edit_string_non_empty(object, "range", at)?, &field)?;
        }
        "add_sheet" | "rename_sheet" => {
            require_sheet_name_field(object, at)?;
        }
        // Nothing beyond the `sheet` checked above: the op names that one field, and
        // the last-sheet and last-visible refusals are the kit's, which sees the list.
        "delete_sheet" => {}
        "move_sheet" => {
            require_one_based_number(object, "to", at)?;
        }
        "add_image" => {
            require_cell(object, at)?;
            edit_string_non_empty(object, "image", at)?;
            require_span_number(
                object,
                "width",
                &RULES.dimension_points,
                "a picture's width in points",
                at,
            )?;
            require_span_number(
                object,
                "height",
                &RULES.dimension_points,
                "a picture's height in points",
                at,
            )?;
        }
        _ => unreachable!("the op was checked against the family's vocabulary"),
    }
    Ok(edit.clone())
}

/// Require an xlsx edit's `sheet` — every op but `add_sheet` names one — and its
/// length.
fn require_sheet_field(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let sheet = edit_string_non_empty(object, "sheet", at)?;
    require_sheet_name_length(sheet, &format!("{at}.sheet"))
}

/// Require an xlsx edit's `name` to be a non-empty sheet name of the shape the
/// shared rules state (see [`require_sheet_name_text`]). Whether the workbook
/// already holds the name is the kit's refusal — it is the side that sees the list.
fn require_sheet_name_field(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let name = edit_string_non_empty(object, "name", at)?;
    require_sheet_name_text(name, &format!("{at}.name"))
}

/// The 1-based column number A1-style letters name (`"A"` -> 1, `"XFD"` ->
/// 16384), or `None` when they are not one to three ASCII uppercase letters.
/// The letters' shape is checked here and nowhere else, so the `cell` and the
/// `column` fields cannot disagree about what a column address is, and the
/// number is the shared reader's own (`reader_output::column_index`).
fn column_number(letters: &str) -> Option<u32> {
    if !((1..=3).contains(&letters.len()) && letters.bytes().all(|byte| byte.is_ascii_uppercase()))
    {
        return None;
    }
    crate::reader_output::column_index(letters).map(|index| index + 1)
}

/// The row and column an A1 address (`"B7"`) names, or `None` when the text is
/// not one: one to three uppercase letters (the column) and a 1-based row of at
/// most seven digits, which is why the row cannot overflow a `u32`.
fn cell_address(cell: &str) -> Option<(u32, u32)> {
    let split = cell.find(|c: char| c.is_ascii_digit())?;
    let (letters, digits) = cell.split_at(split);
    let column = column_number(letters)?;
    let digits_ok = digits.len() <= 7
        && digits
            .as_bytes()
            .first()
            .is_some_and(|byte| (b'1'..=b'9').contains(byte))
        && digits.bytes().all(|byte| byte.is_ascii_digit());
    if !digits_ok {
        return None;
    }
    Some((digits.parse().ok()?, column))
}

/// The row and column an A1 address names, inside the sheet's grid: the shape
/// [`cell_address`] reads, refused with `what` naming the field a caller gave
/// it in.
fn check_cell_address(cell: &str, what: &str) -> Result<(u32, u32)> {
    let Some((row, column)) = cell_address(cell) else {
        anyhow::bail!("usage: {what} must be an A1 address like B7, got \"{cell}\"");
    };
    if row > RULES.sheet_row_max {
        anyhow::bail!(
            "usage: {what} \"{cell}\": the row must be between 1 and {}",
            RULES.sheet_row_max
        );
    }
    if column > RULES.sheet_column_max {
        anyhow::bail!(
            "usage: {what} \"{cell}\": the column must be between A and {}",
            crate::reader_output::column_letters(RULES.sheet_column_max - 1)
        );
    }
    Ok((row, column))
}

/// Require an edit's `cell` to be an A1 address inside the sheet's grid, with
/// its row and column within the shared limits.
fn require_cell(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let cell = edit_string_non_empty(object, "cell", at)?;
    check_cell_address(cell, &format!("{at}.cell")).map(|_| ())
}

/// The `value` a `set_cell` writes: a scalar the writer stringifies, or a
/// `{"formula": "…"}` object normalized to its text — the same rule `create`'s
/// cells follow. A `null` is absent, the way the whole boundary reads one, and a
/// cell's formatting is the `format_cells` op's to change without a value.
fn require_cell_value(object: &serde_json::Map<String, Value>, at: &str) -> Result<Value> {
    let Some(value) = object.get("value").filter(|value| !value.is_null()) else {
        anyhow::bail!(
            "usage: {at} has no \"value\" — hint: pass a string, a number, a boolean or \
             {{\"formula\": \"SUM(A1:A2)\"}}, or use the \"format_cells\" op to change only the \
             cell's formatting"
        );
    };
    if is_scalar(value) {
        return Ok(value.clone());
    }
    match value {
        Value::Object(formula) => normalized_formula(&format!("{at}.value"), formula),
        other => Err(super::wrong_type(
            &format!("{at}.value"),
            SCALAR_TYPE,
            other,
        )),
    }
}

/// Require an edit's `row` to be a sheet row the workbook's grid holds.
fn require_row(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    match edit_integer(object, "row", at)? {
        Some(row) if (1..=u64::from(RULES.sheet_row_max)).contains(&row) => Ok(()),
        _ => anyhow::bail!(
            "usage: {at}.row must be a whole number between 1 and {} — hint: a 1-based sheet row",
            RULES.sheet_row_max
        ),
    }
}

/// Require an edit's `column` to be column letters the workbook's grid holds.
fn require_column(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let column = edit_string_non_empty(object, "column", at)?;
    match column_number(column) {
        Some(number) if number <= RULES.sheet_column_max => Ok(()),
        _ => anyhow::bail!(
            "usage: {at}.column must be column letters between A and {}, got \"{column}\"",
            crate::reader_output::column_letters(RULES.sheet_column_max - 1)
        ),
    }
}

/// The pptx edit vocabulary: a slide is its 1-based number as the reader shows
/// it, and a text op names a fragment that must be on that slide.
fn validate_pptx_edit(object: &serde_json::Map<String, Value>, op: &str, at: &str) -> Result<()> {
    match op {
        "replace_text" | "replace_notes" => {
            require_slide(object, at)?;
            require_pptx_lookup(edit_string_non_empty(object, "find", at)?, "find", at)?;
            edit_string_present(object, "replace", at)?;
        }
        "remove_text" | "remove_paragraph" | "remove_notes" => {
            require_slide(object, at)?;
            require_pptx_lookup(edit_string_non_empty(object, "find", at)?, "find", at)?;
        }
        "format_text" => {
            require_slide(object, at)?;
            require_pptx_lookup(edit_string_non_empty(object, "find", at)?, "find", at)?;
            require_pptx_format(object, at)?;
        }
        "add_paragraph" => {
            require_slide(object, at)?;
            edit_string_present(object, "text", at)?;
            // An absent anchor places the paragraph last; a blank one is refused
            // like a blank `find`.
            if let Some(after) = edit_string(object, "after", at)? {
                require_pptx_lookup(after, "after", at)?;
            }
            require_optional_level(object, at)?;
        }
        "add_image" => {
            require_slide(object, at)?;
            edit_string_non_empty(object, "image", at)?;
            require_slide_geometry(object, at)?;
        }
        "add_table" => {
            require_slide(object, at)?;
            // A presentation's table has no column bound (see [`require_table`]).
            require_table(object, at, None)?;
            require_slide_geometry(object, at)?;
        }
        "add_slide" => {
            if edit_integer(object, "after", at)? == Some(0) {
                anyhow::bail!("usage: {at}.after must be a whole number of at least 1");
            }
            require_optional_text(object, "title", at)?;
            require_bullets(object, at)?;
        }
        "delete_slide" | "duplicate_slide" => {
            require_slide(object, at)?;
        }
        "move_slide" => {
            require_slide(object, at)?;
            require_one_based_number(object, "to", at)?;
        }
        "add_notes" => {
            require_slide(object, at)?;
            edit_string_present(object, "text", at)?;
        }
        _ => unreachable!("the op was checked against the family's vocabulary"),
    }
    Ok(())
}

/// Require a lookup a pptx edit makes — the `find` a text op searches for, or
/// the `after` anchor an inserted paragraph is placed by — to name text. A value
/// that is empty once whitespace is trimmed names nothing a reading showed (the
/// reader prints no blank line), so it is refused here, before the deck is read,
/// rather than searched for.
fn require_pptx_lookup(text: &str, key: &str, at: &str) -> Result<()> {
    if text.trim().is_empty() {
        anyhow::bail!(
            "usage: {at}.{key} must name text — whitespace alone names nothing to search for"
        );
    }
    Ok(())
}

/// Require an edit's `slide` to be a 1-based slide number; a deck's length is
/// the kit's to know, so only the lower bound is enforced here.
fn require_slide(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    require_one_based_number(object, "slide", at)
}

/// Require a 1-based number under `key` — an edit's `slide`, a `move_slide`'s
/// `to` or a `move_sheet`'s `to`.
fn require_one_based_number(
    object: &serde_json::Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<()> {
    match edit_integer(object, key, at)? {
        Some(number) if number >= 1 => Ok(()),
        _ => anyhow::bail!("usage: {at}.{key} must be a whole number of at least 1"),
    }
}

/// Require an `add_paragraph`'s optional outline `level` to be one the slide
/// writer has (0 is the slide's own level, `paragraph_level_max` the deepest).
fn require_optional_level(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    match edit_integer(object, "level", at)? {
        None => Ok(()),
        Some(level) if level <= u64::from(RULES.paragraph_level_max) => Ok(()),
        Some(level) => {
            anyhow::bail!(
                "usage: {at}.level must be a whole number from 0 to {}, got {level}",
                RULES.paragraph_level_max
            )
        }
    }
}

/// Require an `add_slide`'s optional `bullets` to be a list of bounded strings.
fn require_bullets(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let bullets = match object.get("bullets") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(bullets)) => bullets,
        Some(v) => return Err(super::wrong_type(&format!("{at}.bullets"), "an array", v)),
    };
    if bullets.len() > RULES.bullets_max {
        anyhow::bail!(
            "usage: {at}.bullets must hold at most {} bullets, got {} — hint: split the slide",
            RULES.bullets_max,
            bullets.len()
        );
    }
    for (index, bullet) in bullets.iter().enumerate() {
        let Some(text) = bullet.as_str() else {
            return Err(super::wrong_type(
                &format!("{at}.bullets[{index}]"),
                "a string",
                bullet,
            ));
        };
        let length = text.chars().count();
        if length > RULES.edit_text_max {
            anyhow::bail!(
                "usage: {at}.bullets[{index}] must be at most {} characters, got {length}",
                RULES.edit_text_max
            );
        }
    }
    Ok(())
}

/// The words of a shared `(word, value)` pair list, in the order the rules state
/// them: the one list a refusal and the model-facing hint both read, so a word
/// added to the rules cannot leave either of them stale.
fn pair_words(list: &[(String, String)]) -> Vec<&str> {
    list.iter().map(|(word, _)| word.as_str()).collect()
}

/// The slide sizes a `create` may name, as the words the model chooses from.
fn pptx_size_words() -> Vec<&'static str> {
    RULES
        .pptx_sizes
        .iter()
        .map(|size| size.name.as_str())
        .collect()
}

/// The same sizes as the model reads them: every word with the deck it means, so
/// a size is a stated fact rather than a word the model has to guess at.
fn pptx_size_list() -> String {
    RULES
        .pptx_sizes
        .iter()
        .map(|size| format!("{} ({})", size.name, size.deck.inches()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The examples of the shared Word marks a reading prints in brackets, and the
/// labels and annotations it adds without them — quoted and comma-separated for
/// the prompt, which names every one of them from this one statement (see
/// [`crate::docgen::docx_marks`]). The group a mark falls in is the shape it is
/// *printed* as, which its own example states — a pattern is a regex source and
/// says only how a mark is matched.
fn spelled_docx_marks(bracketed: bool) -> String {
    crate::docgen::docx_marks()
        .iter()
        .filter(|mark| mark.example.starts_with('[') == bracketed)
        .map(|mark| format!("`{}`", mark.example))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Require a pptx `format_text` to name at least one property, and each to be
/// the shape the kit writes: a boolean run toggle, a point size inside the
/// presentation's own bound, a hex colour, or a paragraph alignment.
fn require_pptx_format(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let mut any = require_toggles(object, &["bold", "italic", "underline", "strike"], at)?;
    any |= require_stated_number(
        object,
        "size",
        &RULES.slide_text_size_points,
        "give a size in points",
        at,
    )?;
    any |= require_color(object, at)?;
    any |= require_alignment(object, "align", &pair_words(&RULES.slide_alignments), at)?;
    if !any {
        anyhow::bail!(
            "usage: {at} (format_text) needs at least one of bold, italic, underline, strike, \
             size, color or align — hint: a boolean sets bold, italic, underline or strike (false \
             switches it off), size is in points, color is hex digits and align is {}",
            listed(&pair_words(&RULES.slide_alignments))
        );
    }
    Ok(())
}

/// Require a shape's optional placement and size: `x_fraction` and `y_fraction`
/// are fractions of the slide's own width and height and may sit on its edge,
/// while `width_fraction` and `height_fraction` are fractions of that size and
/// may not be zero. A picture and a table are both placed this way.
fn require_slide_geometry(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let place = "give the fraction of the slide the shape is placed at";
    let size = "give the fraction of the slide the shape is scaled to";
    for (key, span, what) in [
        ("x_fraction", &RULES.slide_position_fraction, place),
        ("y_fraction", &RULES.slide_position_fraction, place),
        ("width_fraction", &RULES.slide_size_fraction, size),
        ("height_fraction", &RULES.slide_size_fraction, size),
    ] {
        require_span_number(object, key, span, what, at)?;
    }
    Ok(())
}

/// Resolve and validate the `content` blocks, resolving every embedded image
/// path under the caller's [`PathAccess`]. Shapes the kit would otherwise fail
/// on are rejected here as usage errors.
async fn resolve_content(
    ws: &Workspace,
    args: &Value,
    format: &str,
    access: PathAccess,
) -> Result<Value> {
    let blocks = match args.get("content") {
        Some(Value::Array(blocks)) if !blocks.is_empty() => blocks,
        Some(Value::Array(_)) => {
            anyhow::bail!("usage: \"content\" is empty — hint: pass at least one block")
        }
        Some(v) => return Err(super::wrong_type("content", "an array", v)),
        None => anyhow::bail!(
            "usage: missing required argument \"content\" — hint: pass a JSON array of blocks, \
             e.g. [{{\"type\": \"paragraph\", \"text\": \"…\"}}]"
        ),
    };
    let mut out = Vec::with_capacity(blocks.len());
    // The kinds a block may name, from the one table that states them.
    let kinds = CONTENT_BLOCKS
        .iter()
        .map(|(kind, _, _)| *kind)
        .collect::<Vec<_>>();
    let mut contents = false;
    for (index, block) in blocks.iter().enumerate() {
        let Some(object) = block.as_object() else {
            anyhow::bail!(
                "usage: content[{index}] must be an object — hint: give every block a \"type\""
            );
        };
        let Some(kind) = object.get("type").and_then(Value::as_str) else {
            anyhow::bail!(
                "usage: content[{index}] has no \"type\" — hint: use {}",
                listed(&kinds)
            );
        };
        let at = format!("content[{index}]");
        // Every arm produces the block to push: it validates the shape and hands
        // back what the request will carry, the image's path resolved against the
        // workspace. One push after the match keeps the two in step.
        let resolved = match kind {
            "heading" => {
                require_block_keys(object, &at, "heading")?;
                require_string(object, "text", index)?;
                require_heading_level(object, index)?;
                block.clone()
            }
            "paragraph" => {
                require_block_keys(object, &at, "paragraph")?;
                require_string(object, "text", index)?;
                // A footnote is a Word document's own: a presentation and a PDF
                // have nowhere to print one, so it is refused rather than lost. An
                // explicit null is an absent footnote, the way every optional block
                // field reads one.
                if object.get("footnote").is_some_and(|value| !value.is_null()) {
                    require_docx_block(format, &at, "footnote")?;
                    require_text(object, "footnote", index)?;
                }
                block.clone()
            }
            // Speaker notes exist only on a presentation: dropping the block
            // for a document or a PDF would lose text the caller asked for.
            "notes" => {
                if format != "pptx" {
                    anyhow::bail!(
                        "usage: content[{index}] (notes) is a presentation's speaker notes — \
                         only pptx has notes — hint: write a paragraph block instead"
                    );
                }
                require_block_keys(object, &at, "notes")?;
                require_string(object, "text", index)?;
                block.clone()
            }
            "list" => {
                require_block_keys(object, &at, "list")?;
                let Some(items) = object.get("items") else {
                    anyhow::bail!("usage: content[{index}] has no \"items\"");
                };
                require_scalar_array(items, &format!("content[{index}].items"))?;
                require_optional_bool(object, "ordered", index)?;
                block.clone()
            }
            "table" => {
                require_block_keys(object, &at, "table")?;
                // The block reaches the writers as the one shape both of them take:
                // an empty header set and a cell-less row are dropped here. Only a
                // Word table has a column bound; a presentation's and a PDF's table
                // has none.
                let table = normalize_table(object);
                let max = (format == "docx").then_some(RULES.table_columns_max);
                require_table(&table, &at, max)?;
                Value::Object(table)
            }
            "image" => resolve_image_block(ws, block, object, index, format, access).await?,
            "link" | "page_break" | "contents" => {
                docx_only_block(block, object, kind, index, format, &mut contents)?
            }
            other => anyhow::bail!(
                "usage: content[{index}] has unknown type \"{other}\" — hint: use {}",
                listed(&kinds)
            ),
        };
        out.push(resolved);
    }
    Ok(Value::Array(out))
}

/// A block only a Word document holds — a link, a page break or a table of
/// contents — checked in the shape its kind declares, and handed back as the request
/// carries it: a link's address is the one its own parse produced. `contents` is the
/// document's one table of contents: a second is refused rather than written beside it.
fn docx_only_block(
    block: &Value,
    object: &serde_json::Map<String, Value>,
    kind: &str,
    index: usize,
    format: &str,
    contents: &mut bool,
) -> Result<Value> {
    let at = format!("content[{index}]");
    require_block_keys(object, &at, kind)?;
    require_docx_block(format, &at, kind)?;
    let mut resolved = block.clone();
    match kind {
        "link" => {
            require_text(object, "text", index)?;
            resolved["url"] = json!(link_address(require_string(object, "url", index)?, index)?);
        }
        "contents" if *contents => anyhow::bail!(
            "usage: content[{index}] is a second table of contents — a document has one"
        ),
        "contents" => *contents = true,
        _ => {}
    }
    Ok(resolved)
}

/// One `image` content block resolved: the picture's path under the caller's own
/// [`PathAccess`], checked against the formats the target format can hold, and
/// the block the request carries — the kit reads the placed picture from the
/// block's own `image`.
async fn resolve_image_block(
    ws: &Workspace,
    block: &Value,
    object: &serde_json::Map<String, Value>,
    index: usize,
    format: &str,
    access: PathAccess,
) -> Result<Value> {
    require_block_keys(object, &format!("content[{index}]"), "image")?;
    require_optional_size(object, "width", index)?;
    require_optional_size(object, "height", index)?;
    let Some(raw) = object.get("image").and_then(Value::as_str) else {
        anyhow::bail!(
            "usage: content[{index}] (image) has no \"image\" — hint: give the image's path"
        );
    };
    let (image, _) = resolve_input(ws, raw, access).await?;
    // A PDF's own two encodings take no WebP; the block is refused here, before
    // the call reserves an output.
    if format == "pdf" {
        ensure_picture(&image, &pdf_pictures())?;
    } else {
        ensure_picture(&image, &embedded_pictures())?;
    }
    let mut resolved = block.clone();
    resolved["image"] = json!(image.to_string_lossy());
    Ok(resolved)
}

/// Resolve every `add_image` picture in a validated edit list under the caller's
/// [`PathAccess`], the way `create`'s image block is: the kit is handed a path it
/// can open, and the same read rule keeps an edit from embedding a file the
/// caller did not name inside its own reach. The list's shape — an array of
/// objects, each naming an `op`, an `add_image` also carrying a non-empty
/// `image` — is the family validator's own contract.
async fn resolve_edit_images(ws: &Workspace, edits: &mut Value, access: PathAccess) -> Result<()> {
    let list = edits
        .as_array_mut()
        .expect("validate_edits hands an edit list back as an array");
    for edit in list {
        let object = edit
            .as_object_mut()
            .expect("validate_edits hands every edit back as an object");
        if object.get("op").and_then(Value::as_str) != Some("add_image") {
            continue;
        }
        let raw = object
            .get("image")
            .and_then(Value::as_str)
            .expect("the edit validator requires an add_image image")
            .to_owned();
        let (image, _) = resolve_input(ws, &raw, access).await?;
        ensure_picture(&image, &embedded_pictures())?;
        object.insert("image".to_owned(), json!(image.to_string_lossy()));
    }
    Ok(())
}

/// Whether one of `edits` looks text up on the slide itself: an edit whose op is
/// a slide text op (see [`PPTX_SLIDE_TEXT_OPS`]) and that names what to look up —
/// the `find` a text op matches, or the `after` anchor an inserted paragraph is
/// placed by. Only such an edit can be refused for naming a diagram's text, so
/// only they need the slides' diagram text read.
fn edits_search_slide_text(edits: &Value) -> bool {
    edits.as_array().is_some_and(|edits| {
        edits.iter().any(|edit| {
            let names_a_lookup = edit.get("find").is_some() || edit.get("after").is_some();
            names_a_lookup
                && edit
                    .get("op")
                    .and_then(Value::as_str)
                    .is_some_and(|op| PPTX_SLIDE_TEXT_OPS.contains(&op))
        })
    })
}

/// Resolve and validate the `sheets` argument: every sheet is an object with a
/// name, rows of cells and the ranges it merges, and every cell is a shape the
/// kit's writer knows.
fn resolve_sheets(args: &Value) -> Result<Value> {
    let sheets = match args.get("sheets") {
        Some(Value::Array(sheets)) if !sheets.is_empty() => sheets,
        Some(Value::Array(_)) => {
            anyhow::bail!("usage: \"sheets\" is empty — hint: pass at least one sheet")
        }
        Some(v) => return Err(super::wrong_type("sheets", "an array", v)),
        None => anyhow::bail!(
            "usage: missing required argument \"sheets\" — hint: pass a JSON array of sheets, \
             e.g. [{{\"rows\": [[\"Cell\", 1]]}}]"
        ),
    };
    let mut normalized = Vec::with_capacity(sheets.len());
    let mut seen_names = HashSet::new();
    for (index, sheet) in sheets.iter().enumerate() {
        let Some(object) = sheet.as_object() else {
            anyhow::bail!(
                "usage: sheets[{index}] must be an object — hint: {{\"name\": …, \"rows\": [[…]]}}"
            );
        };
        let at = format!("sheets[{index}]");
        require_taken_keys(object, &at, "a sheet", &sheet_field_names(), &[])?;
        let name = optional_string(object.get("name"), &at, "name")?;
        // A sheet with no name is written as `Sheet<n>` by its position, the
        // kit's own writer default. Resolving it here means the request always
        // carries an explicit name, so the kit never has to default in the
        // tool's path — and the resolved name is held to the shared rules too.
        let name = name.filter(|name| !name.is_empty()).map_or_else(
            || format!("Sheet{}", index + 1),
            std::string::ToString::to_string,
        );
        require_sheet_name(&name, index, &mut seen_names)?;
        let rows = match object.get("rows") {
            Some(Value::Array(rows)) => normalize_sheet_cells(rows, index)?,
            Some(v) => {
                return Err(super::wrong_type(
                    &format!("sheets[{index}].rows"),
                    "an array",
                    v,
                ));
            }
            None => anyhow::bail!(
                "usage: sheets[{index}] has no \"rows\" — hint: {{\"rows\": [[\"Cell\", 1]]}}"
            ),
        };
        let mut normalized_sheet = object.clone();
        normalized_sheet.insert("name".to_owned(), json!(name));
        normalized_sheet.insert("rows".to_owned(), rows);
        // The clone carries the sheet's own `merges` as it was written, so the
        // entries are checked in place. A null reads as absent everywhere else,
        // and the kit tests only `!== undefined`.
        match object.get("merges") {
            None => {}
            Some(Value::Null) => {
                normalized_sheet.remove("merges");
            }
            Some(Value::Array(merges)) => require_sheet_merges(merges, index)?,
            Some(v) => {
                return Err(super::wrong_type(
                    &format!("sheets[{index}].merges"),
                    "an array",
                    v,
                ));
            }
        }
        normalized.push(Value::Object(normalized_sheet));
    }
    Ok(Value::Array(normalized))
}

/// Require the `merges` a `create` sheet names: each entry is an A1 range, corners
/// in either order, inside the sheet's grid and covering at least two cells.
/// Overlap is the kit's to refuse — only its writer sees what is merged there.
fn require_sheet_merges(merges: &[Value], index: usize) -> Result<()> {
    for (position, merge) in merges.iter().enumerate() {
        let at = format!("sheets[{index}].merges[{position}]");
        match merge {
            Value::String(range) => require_merge_text(range, &at)?,
            other => return Err(super::wrong_type(&at, "a string", other)),
        }
    }
    Ok(())
}

/// Refuse a sheet name longer than the shared cap, naming `at` in the message.
/// The length rule is the same one `create`'s `sheets[{index}].name` enforces.
fn require_sheet_name_length(name: &str, at: &str) -> Result<()> {
    if name.chars().count() > RULES.sheet_name_max {
        anyhow::bail!(
            "usage: {at} must be at most {} characters, got \"{name}\" — hint: shorten the sheet \
             name",
            RULES.sheet_name_max
        );
    }
    Ok(())
}

/// Require a sheet name to satisfy the shared rules: at most `sheet_name_max`
/// characters, none of `sheet_name_forbidden`, and no earlier sheet sharing it
/// ignoring case. The caller has already resolved an absent name to `Sheet<n>`,
/// so only the upper bound is enforced here. The kit's own writer keeps the same
/// rule as its last line, over the names it actually writes.
fn require_sheet_name(name: &str, index: usize, seen: &mut HashSet<String>) -> Result<()> {
    let at = format!("sheets[{index}].name");
    require_sheet_name_text(name, &at)?;
    if !seen.insert(name.to_lowercase()) {
        anyhow::bail!(
            "usage: {at} is not unique, ignoring case, got \"{name}\" — hint: give every sheet a \
             distinct name"
        );
    }
    Ok(())
}

/// Require a sheet name to have the shared shape: at most `sheet_name_max`
/// characters and none of `sheet_name_forbidden`. Uniqueness is not this function's:
/// `create` checks it across the sheets it writes, and an edit's clash with a sheet
/// the workbook already holds is the kit's refusal.
fn require_sheet_name_text(name: &str, at: &str) -> Result<()> {
    require_sheet_name_length(name, at)?;
    if let Some(forbidden) = RULES
        .sheet_name_forbidden
        .chars()
        .find(|character| name.contains(*character))
    {
        // The offending character as a JSON string, as the set below is: a bare
        // `\` before a closing quote reads as an escaped quote instead.
        anyhow::bail!(
            "usage: {at} must not contain {:?} — hint: a sheet name cannot hold any of {:?}",
            forbidden.to_string(),
            RULES.sheet_name_forbidden
        );
    }
    Ok(())
}

/// A sheet cell holding a formula: an object with nothing but a `formula` whose
/// text, trimmed, with one optional leading `=` removed and trimmed again, is
/// not empty and no longer starts with `=`. It returns that normalized text as
/// `{"formula": …}`, so the request always carries the formula the cell is to
/// hold: a text like `==SUM(A1:A2)` is refused rather than reaching a cell's
/// `<f>` element, where it would no longer be a formula.
///
/// The kit's writer repeats the same rule as its own last line, so the two read
/// alike.
fn normalized_formula(at: &str, object: &serde_json::Map<String, Value>) -> Result<Value> {
    let normalized = object
        .get("formula")
        .and_then(Value::as_str)
        .map(str::trim)
        .map(|text| text.strip_prefix('=').unwrap_or(text).trim())
        .filter(|text| !text.is_empty() && !text.starts_with('='));
    match normalized {
        Some(text) if object.len() == 1 => Ok(json!({ "formula": text })),
        _ => anyhow::bail!(
            "usage: {at} must be text, a number, a boolean or {{\"formula\": \"SUM(A1:A2)\"}} — \
             hint: the formula's text alone, as in \"SUM(A1:A2)\""
        ),
    }
}

/// Normalize every row of a sheet into the array the request carries: each row
/// must be an array of cells the kit's writer knows — text, a number, a boolean,
/// or `{"formula": "..."}`. A string that starts with `=` stays text — writing a
/// formula is what the object shape is for — and any other structure would reach
/// the sheet as `[object Object]`. Formula cells go through
/// [`normalized_formula`], the one owner of that rule.
fn normalize_sheet_cells(rows: &[Value], sheet: usize) -> Result<Value> {
    let mut normalized_rows = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let Some(cells) = row.as_array() else {
            return Err(super::wrong_type(
                &format!("sheets[{sheet}].rows[{row_index}]"),
                "an array",
                row,
            ));
        };
        let mut normalized_cells = Vec::with_capacity(cells.len());
        for (cell_index, cell) in cells.iter().enumerate() {
            let at = format!("sheets[{sheet}].rows[{row_index}][{cell_index}]");
            match cell {
                value if is_scalar(value) => normalized_cells.push(value.clone()),
                Value::Object(object) => {
                    let formula = normalized_formula(&at, object)?;
                    normalized_cells.push(formula);
                }
                other => return Err(super::wrong_type(&at, SCALAR_TYPE, other)),
            }
        }
        normalized_rows.push(Value::Array(normalized_cells));
    }
    Ok(Value::Array(normalized_rows))
}

/// A table's shape, as both producers take it: `headers` (when stated) and `rows`
/// are arrays of scalars, the table holds something to lay out, and — where the
/// target format has one — it fits the column bound. `at` names the position in
/// the request, `max_columns` the bound the format has (`None` for a presentation
/// or a PDF, which has no such limit). One rule for `create` and an `add_table`,
/// so the two cannot accept different tables.
fn require_table(
    object: &serde_json::Map<String, Value>,
    at: &str,
    max_columns: Option<usize>,
) -> Result<()> {
    // `headers` is optional — null counts as absent here exactly as it does for
    // `rows` and every other optional argument.
    let headers = match object.get("headers").filter(|value| !value.is_null()) {
        None => 0,
        Some(headers) => {
            require_scalar_array(headers, &format!("{at}.headers"))?;
            headers.as_array().map_or(0, Vec::len)
        }
    };
    // The headings row is a row too: its cells count toward the columns the table
    // states, the way the kit's own writer counts them.
    let mut widest = headers;
    let mut laid_out = 0;
    if let Some(rows) = object.get("rows").filter(|value| !value.is_null()) {
        let Some(rows) = rows.as_array() else {
            return Err(super::wrong_type(&format!("{at}.rows"), "an array", rows));
        };
        for (row_index, row) in rows.iter().enumerate() {
            let Some(cells) = row.as_array() else {
                return Err(super::wrong_type(
                    &format!("{at}.rows[{row_index}]"),
                    "an array",
                    row,
                ));
            };
            if let Some((cell_index, cell)) = scalar_array_violation(cells) {
                return Err(super::wrong_type(
                    &format!("{at}.rows[{row_index}][{cell_index}]"),
                    SCALAR_TYPE,
                    cell,
                ));
            }
            widest = widest.max(cells.len());
            // A row with no cells lays out nothing — `create`'s writer drops it,
            // and a writer handed one would emit a `<w:tr/>` Word refuses to open
            // (see [`normalize_table`]).
            laid_out += usize::from(!cells.is_empty());
        }
    }
    // No writer has a presentation for a table with nothing to lay out: the docx
    // arm would emit a `<w:tbl>` with no `<w:tr>`, which Word refuses to open, and
    // the other two would draw nothing at all for it.
    if laid_out + headers == 0 {
        anyhow::bail!(
            "usage: {at} states a table with no rows and no headers — hint: give it at least one \
             row or one header"
        );
    }
    if let Some(max) = max_columns.filter(|max| widest > *max) {
        anyhow::bail!(
            "usage: {at} states {widest} columns — a table holds at most {max} — hint: split the \
             table"
        );
    }
    Ok(())
}

/// Require `value` to be an array of scalars (see [`SCALAR_TYPE`]), naming the
/// offending position by `at` — the full path, e.g. `content[1].headers`.
fn require_scalar_array(value: &Value, at: &str) -> Result<()> {
    let Some(values) = value.as_array() else {
        return Err(super::wrong_type(at, "an array", value));
    };
    match scalar_array_violation(values) {
        Some((position, scalar)) => Err(super::wrong_type(
            &format!("{at}[{position}]"),
            SCALAR_TYPE,
            scalar,
        )),
        None => Ok(()),
    }
}

/// What a bullet and a table cell can hold: the kit's writers stringify these
/// and have no rendering for a structure.
const SCALAR_TYPE: &str = "text, a number or a boolean";

/// The type name a scalar value has in the shared rules' vocabulary — the names
/// `typeof` gives in the kit — or `None` for a structure (and for null, which
/// `typeof` calls an object).
fn scalar_kind(value: &Value) -> Option<&'static str> {
    match value {
        Value::String(_) => Some("string"),
        Value::Number(_) => Some("number"),
        Value::Bool(_) => Some("boolean"),
        _ => None,
    }
}

/// Whether `value` is one of the scalar kinds the shared rules allow.
fn is_scalar(value: &Value) -> bool {
    scalar_kind(value).is_some_and(|kind| RULES.scalar_kinds.iter().any(|allowed| allowed == kind))
}

/// The first element of `values` that is not a scalar, with its position.
fn scalar_array_violation(values: &[Value]) -> Option<(usize, &Value)> {
    values
        .iter()
        .enumerate()
        .find(|(_, value)| !is_scalar(value))
}

/// Normalize a table block into the one the request carries: an empty set is an
/// absent one, so an empty `headers` array is removed and a cell-less row is
/// dropped. A writer handed a cell-less row would emit a row with no cells — a
/// `<w:tr/>` Word refuses to open — so only what the writers will draw reaches
/// them. What the block still carries is shape-checked by [`require_table`]
/// afterwards, which also refuses a table left with nothing.
fn normalize_table(object: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    let mut table = object.clone();
    if table
        .get("headers")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        table.remove("headers");
    }
    if let Some(Value::Array(rows)) = table.get_mut("rows") {
        rows.retain(|row| !row.as_array().is_some_and(Vec::is_empty));
    }
    table
}

/// `1, 2 or 3` — how a refusal hint and the description name the values a rule
/// allows.
fn listed<T: std::fmt::Display>(values: &[T]) -> String {
    crate::util::listed(values, "or")
}

/// Whole megabytes of `bytes`, the unit the tool's messages state a size in.
fn megabytes(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

/// Require a heading's `level` to be one of the levels the writers have.
fn require_heading_level(object: &serde_json::Map<String, Value>, index: usize) -> Result<()> {
    match object.get("level") {
        None | Some(Value::Null) => Ok(()),
        Some(value) => {
            let valid = value
                .as_f64()
                .is_some_and(|level| level.fract() == 0.0 && RULES.heading_levels.contains(&level));
            if valid {
                Ok(())
            } else {
                anyhow::bail!(
                    "usage: content[{index}].level must be {} — hint: a lower number is the more \
                     important heading",
                    listed(&RULES.heading_levels)
                )
            }
        }
    }
}

/// Require an optional size field of a content block (an image's `width` or
/// `height`) to be a number inside the range the writers can lay out: zero or a
/// negative number is not a size, anything past the shared rules'
/// `dimension_points` upper bound is beyond any page, and anything else is not
/// a size at all. The unit is the point, which is what every writer places a
/// picture in.
fn require_optional_size(
    object: &serde_json::Map<String, Value>,
    key: &str,
    index: usize,
) -> Result<()> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(value) => {
            let number = value.as_f64().ok_or_else(|| {
                super::wrong_type(&format!("content[{index}].{key}"), "a number", value)
            })?;
            if !RULES.dimension_points.contains(number) {
                anyhow::bail!(
                    "usage: content[{index}].{key} must be {} points, got {number} — hint: give \
                     the size the image should be placed at",
                    RULES.dimension_points.bounds()
                );
            }
            Ok(())
        }
    }
}

/// Require an optional boolean field of a content block.
fn require_optional_bool(
    object: &serde_json::Map<String, Value>,
    key: &str,
    index: usize,
) -> Result<()> {
    match object.get(key) {
        None | Some(Value::Null | Value::Bool(_)) => Ok(()),
        Some(value) => Err(super::wrong_type(
            &format!("content[{index}].{key}"),
            "a boolean",
            value,
        )),
    }
}

/// A present optional string, `at` naming the prefix of the path it sits at: an
/// absent or null value is `None`, a string is read, and a value of another type
/// is refused rather than read as absent, the way the shared
/// [`super::get_opt_str_strict`] refuses a top-level one — this helper only adds
/// the caller-built path prefix a nested field's refusal names.
fn optional_string<'a>(value: Option<&'a Value>, at: &str, key: &str) -> Result<Option<&'a str>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(other) => Err(super::wrong_type(&format!("{at}.{key}"), "a string", other)),
    }
}

/// The PDF description `metadata` names, in the kit's shape: every field it
/// names is passed through (a date in the PDF's own spelling), and a field it
/// does not name is absent rather than null, so the kit leaves the file's own
/// entry as it is — except the modification date an edit advances. Read with the
/// rest of the arguments, before any output is reserved.
fn opt_metadata(args: &Value) -> Result<Option<Value>> {
    let Some(value) = args.get("metadata") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(object) = value.as_object() else {
        return Err(super::wrong_type("metadata", "an object", value));
    };
    let fields = crate::docgen::pdf_metadata_fields();
    let keys = fields
        .iter()
        .map(|field| field.key.as_str())
        .collect::<Vec<_>>();
    if object.is_empty() {
        anyhow::bail!(
            "usage: \"metadata\" is empty — hint: name at least one of {}",
            listed(&keys)
        );
    }
    let mut metadata = serde_json::Map::with_capacity(object.len());
    for (key, value) in object {
        let Some(field) = fields.iter().find(|field| field.key == *key) else {
            anyhow::bail!(
                "usage: unknown metadata field \"{key}\" — hint: use {}",
                listed(&keys)
            );
        };
        let Some(text) = value.as_str() else {
            return Err(super::wrong_type(
                &format!("metadata.{key}"),
                "a string",
                value,
            ));
        };
        metadata.insert(
            key.clone(),
            Value::String(if field.date {
                pdf_date(text)?
            } else {
                text.to_string()
            }),
        );
    }
    Ok(Some(Value::Object(metadata)))
}

/// Attach a PDF action's `metadata` to its request, when the caller named it.
fn set_metadata(request: &mut Value, metadata: Option<Value>) {
    if let Some(metadata) = metadata {
        request["metadata"] = metadata;
    }
}

/// The PDF spelling of a metadata date: `D:YYYYMMDDHHMMSS` with `Z` for UTC or
/// `±HH'MM'` for another offset. A calendar date alone is midnight UTC, a time
/// follows `T` with optional seconds, and the only offset spellings are `Z` and
/// `±HH:MM`. An empty string clears the entry and passes through.
fn pdf_date(text: &str) -> Result<String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let refused = || {
        anyhow::anyhow!(
            "usage: \"{text}\" is not a date — hint: use {}",
            crate::docgen::pdf_date_format()
        )
    };
    let date = text.get(..10).ok_or_else(refused)?;
    let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|_| refused())?;
    let mut rest = &text[10..];
    let offset_minutes = if let Some(stripped) = rest.strip_suffix('Z') {
        rest = stripped;
        0
    } else if let Some(index) = rest.rfind(['+', '-']) {
        let (time, offset) = rest.split_at(index);
        rest = time;
        pdf_offset(offset).ok_or_else(refused)?
    } else {
        0
    };
    let time = if rest.is_empty() {
        chrono::NaiveTime::from_hms_opt(0, 0, 0).expect("midnight is a valid time")
    } else {
        let time = rest.strip_prefix('T').ok_or_else(refused)?;
        ["%H:%M", "%H:%M:%S"]
            .iter()
            .find_map(|format| chrono::NaiveTime::parse_from_str(time, format).ok())
            .ok_or_else(refused)?
    };
    let mut rendered = format!(
        "D:{:04}{:02}{:02}{:02}{:02}{:02}",
        date.year(),
        date.month(),
        date.day(),
        time.hour(),
        time.minute(),
        time.second()
    );
    if offset_minutes == 0 {
        rendered.push('Z');
    } else {
        let sign = if offset_minutes < 0 { '-' } else { '+' };
        let minutes = offset_minutes.abs();
        let _ = write!(rendered, "{sign}{:02}'{:02}'", minutes / 60, minutes % 60);
    }
    Ok(rendered)
}

/// A `±HH:MM` offset as minutes from UTC.
fn pdf_offset(text: &str) -> Option<i32> {
    let (sign, rest) = match text.as_bytes().first()? {
        b'+' => (1, &text[1..]),
        b'-' => (-1, &text[1..]),
        _ => return None,
    };
    let (hours, minutes) = rest.split_once(':')?;
    if hours.len() != 2 || minutes.len() != 2 {
        return None;
    }
    let hours: i32 = hours.parse().ok()?;
    let minutes: i32 = minutes.parse().ok()?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    Some(sign * (hours * 60 + minutes))
}

/// Require a fill's `values` to be present and non-empty, and hand the map back:
/// a fill with no values is a copy, not a fill, and reporting that as a success
/// would name an outcome where nothing happened. Both fills resolve their input
/// first, so a call that cannot read the file at all is refused for that first.
fn require_values(args: &Value, kind: &str) -> Result<serde_json::Map<String, Value>> {
    let values = super::get_object(args, "values")?;
    if values.is_empty() {
        anyhow::bail!(
            "usage: missing or empty argument \"values\" — hint: name at least one {kind}, e.g. \
             {{\"name\": \"Иван\"}}"
        );
    }
    Ok(values)
}

/// Require every value of a name-keyed map — a sample's placeholders, a form's
/// fields — to be a scalar the kit can write: a structure would reach the
/// document as `[object Object]`.
fn require_scalar_values(values: &serde_json::Map<String, Value>) -> Result<()> {
    for (name, value) in values {
        if !is_scalar(value) {
            return Err(super::wrong_type(
                &format!("values.{name}"),
                SCALAR_TYPE,
                value,
            ));
        }
    }
    Ok(())
}

/// Check that a colour is the shape the kit parses: the shared rules'
/// `color_digits` hex digits, with an optional leading `#`. `at` is the path the
/// refusal names, spelled the way the caller's other messages spell it.
fn require_hex_color(at: &str, color: &str) -> Result<()> {
    let digits = color.strip_prefix('#').unwrap_or(color);
    if digits.len() == RULES.color_digits && digits.chars().all(|digit| digit.is_ascii_hexdigit()) {
        return Ok(());
    }
    anyhow::bail!(
        "usage: {at} must be {} hex digits, with an optional leading \"#\", got \"{color}\" — \
         hint: like \"#{}\"",
        RULES.color_digits,
        "f".repeat(RULES.color_digits)
    )
}

/// Require `object[key]` to be a string, naming the block position in the error, and
/// give its text — the shape a rule about the string's own content reads next.
fn require_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    index: usize,
) -> Result<&'a str> {
    match object.get(key) {
        Some(Value::String(text)) => Ok(text),
        Some(v) => Err(super::wrong_type(
            &format!("content[{index}].{key}"),
            "a string",
            v,
        )),
        None => anyhow::bail!("usage: content[{index}] has no \"{key}\""),
    }
}

/// Require a piece of text to hold something: an empty or whitespace-only link text,
/// footnote, running header or footer is content no reader shows, so it is refused
/// rather than written as an invisible part of an answer that reports success. `at`
/// names the field and `hint` the way out, so one rule serves every field that holds
/// text and each refusal still names its own.
fn require_visible(at: &str, text: &str, hint: &str) -> Result<()> {
    if text.trim().is_empty() {
        anyhow::bail!("usage: {at} is empty — hint: {hint}");
    }
    Ok(())
}

/// Require `object[key]` to be a string holding something (see [`require_visible`]).
fn require_text(object: &serde_json::Map<String, Value>, key: &str, index: usize) -> Result<()> {
    require_visible(
        &format!("content[{index}].{key}"),
        require_string(object, key, index)?,
        "give the text it shows",
    )
}

/// The picture formats one placement takes: the extensions a picture's path is
/// checked against and the phrase every argument and refusal states them by, so a
/// set is written once.
struct PictureFormats {
    extensions: Vec<&'static str>,
    names: &'static str,
}

/// The formats a document or a presentation embeds: the shared rules' list, which
/// holds the WebP the product's own picture tool writes.
fn embedded_pictures() -> PictureFormats {
    PictureFormats {
        extensions: RULES.image_extensions.iter().map(String::as_str).collect(),
        names: "PNG, JPEG or WebP",
    }
}

/// A PDF's own two encodings: the WebP the document writers take is refused here
/// rather than after the call's output was reserved.
fn pdf_pictures() -> PictureFormats {
    PictureFormats {
        extensions: vec!["png", "jpg", "jpeg"],
        names: "PNG or JPEG",
    }
}

/// Require a picture's path to carry an extension of `formats`' set. The bytes are
/// what really decides the kind, and the kit reads them there (see
/// `assets/docgen/kit.js`), which is the one place every image goes through.
fn ensure_picture(path: &Path, formats: &PictureFormats) -> Result<()> {
    if crate::util::has_extension(path, &formats.extensions) {
        Ok(())
    } else {
        anyhow::bail!(
            "usage: a picture must be {}, got {} — hint: convert the image and retry",
            formats.names,
            path.display()
        )
    }
}

/// The pages a PDF action acts on: the union of the `pages` elements.
enum Pages {
    /// Every page (`"all"`, an empty `pages`, or an absent one).
    All,
    /// 1-based page numbers, in the order the elements name them.
    Explicit(Vec<u32>),
}

impl Pages {
    /// The value the kit understands: `"all"`, or an array of 1-based numbers.
    #[must_use]
    fn kit_value(&self) -> Value {
        match self {
            Self::All => json!("all"),
            Self::Explicit(pages) => json!(pages),
        }
    }
}

/// The page selections a call names, from the one `pages` argument every PDF
/// action that takes it shares: absent, null or an empty list names none (every
/// page), and an element is `"all"` or a comma-separated list of `N` and `N-M`
/// that [`parse_range`] reads. An `"all"` element takes every page, so it cannot
/// sit beside another element: the two would name the same page twice, which the
/// per-element duplicate rule already refuses for named pages.
fn page_selections(args: &Value) -> Result<Vec<String>> {
    // The element's own name is what a caller can fix, so the refusal is the
    // array's, naming the shape it wants.
    let selections = super::get_str_array(args, "pages")?;
    if selections.len() > 1
        && selections
            .iter()
            .any(|selection| selection.trim().eq_ignore_ascii_case("all"))
    {
        anyhow::bail!(
            "usage: an \"all\" selection takes every page, so it cannot sit beside another \
             selection — hint: name every page explicitly, or pass \"all\" alone"
        );
    }
    Ok(selections)
}

/// Parse the optional `pages` argument into the pages the action acts on: the
/// union of every element, in the order the elements name them. An absent, null
/// or empty list means every page, and `"all"` is the only element that means
/// it — it stands alone, so no element can be widened by another.
fn opt_pages(args: &Value) -> Result<Pages> {
    let mut seen = HashSet::new();
    let mut pages = Vec::new();
    let mut all = false;
    for selection in page_selections(args)? {
        let selection = selection.trim();
        if selection.eq_ignore_ascii_case("all") {
            all = true;
            continue;
        }
        pages.extend(parse_range(selection, &mut seen)?);
    }
    if all || pages.is_empty() {
        Ok(Pages::All)
    } else {
        Ok(Pages::Explicit(pages))
    }
}

/// Parse one `pdf_split` page selection per output file, in order. The list is
/// required and non-empty, and one output file per element makes the element
/// count the attachment count. An element's grammar is the one every PDF action
/// uses, `"all"` included — and, as everywhere, an `"all"` element stands alone,
/// so no page is written into two files. The kit resolves that element against
/// the document's own page count, the only place it is known.
fn split_groups(args: &Value) -> Result<Vec<Value>> {
    let selections = page_selections(args)?;
    if selections.is_empty() {
        anyhow::bail!(
            "usage: missing required argument \"pages\" — hint: one page selection per output file, \
             e.g. [\"1-3\", \"5\"]"
        );
    }
    if selections.len() > MAX_SPLIT_PARTS {
        anyhow::bail!(
            "usage: {} output files is more than one call can produce (limit {MAX_SPLIT_PARTS}) — \
             hint: ask for fewer page selections, or wider ones",
            selections.len()
        );
    }
    // The `seen` set is the request's, not an element's: a page named in two
    // elements would be written into two files, and the page cap is on the whole
    // request rather than on any one element.
    let mut seen = HashSet::new();
    let mut groups = Vec::with_capacity(selections.len());
    for selection in &selections {
        let selection = selection.trim();
        if selection.eq_ignore_ascii_case("all") {
            groups.push(json!("all"));
            continue;
        }
        groups.push(json!(parse_range(selection, &mut seen)?));
    }
    Ok(groups)
}

/// Parse a comma-separated range string (`"1-3,5"`) into ascending 1-based page
/// numbers, rejecting 0, a reversed range and any page already claimed by
/// `seen`, so one call can never name a page twice. `seen` is shared across a
/// request, so its length bounds the pages the whole request names, not just
/// this range's own.
fn parse_range(spec: &str, seen: &mut HashSet<u32>) -> Result<Vec<u32>> {
    let mut pages = Vec::new();
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            anyhow::bail!(
                "usage: empty page range in \"{spec}\" — hint: write a selection like \
                 \"1-3,5\""
            );
        }
        let (start, end) = if let Some((start, end)) = token.split_once('-') {
            (parse_page(start)?, parse_page(end)?)
        } else {
            let page = parse_page(token)?;
            (page, page)
        };
        if end < start {
            anyhow::bail!(
                "usage: reversed page range \"{token}\" — hint: write it low-to-high, \
                 e.g. \"{end}-{start}\""
            );
        }
        for page in start..=end {
            if !seen.insert(page) {
                anyhow::bail!(
                    "usage: page {page} appears more than once — hint: list each page once"
                );
            }
            pages.push(page);
            if seen.len() > RULES.max_pages_per_request {
                anyhow::bail!(
                    "usage: too many pages requested (limit {})",
                    RULES.max_pages_per_request
                );
            }
        }
    }
    Ok(pages)
}

/// Parse one page number, rejecting 0.
fn parse_page(text: &str) -> Result<u32> {
    let page: u32 = text.trim().parse().map_err(|_| {
        anyhow::anyhow!(
            "usage: \"{text}\" is not a page number — hint: pages are 1-based, like \"3\" or \"1-3\""
        )
    })?;
    if page == 0 {
        anyhow::bail!("usage: page 0 does not exist — hint: the first page is 1");
    }
    Ok(page)
}

/// Copy a present numeric argument into the request, within the range a page can
/// carry (negative positions are legal, so the bound is on the magnitude). The
/// largest coordinate, font size or placed size a PDF writer lays out is the
/// shared rules' `pdf_point_abs_max`: nothing on a page comes near it, and
/// pdf-lib writes the number straight through, so an unbounded one becomes a
/// file no reader can open.
fn set_number(request: &mut Value, args: &Value, key: &str) -> Result<()> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(v) => {
            let number = v
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| super::wrong_type(key, "a number", v))?;
            if number.abs() > RULES.pdf_point_abs_max {
                anyhow::bail!(
                    "usage: \"{key}\" must be within ±{} points, got {number} — hint: \
                     give a position or size on the page",
                    RULES.pdf_point_abs_max
                );
            }
            request[key] = v.clone();
            Ok(())
        }
    }
}

/// [`set_number`] for a measure the writers lay out — a font size or a placed
/// size, where zero or a negative number is not a thing a document can show. The
/// caller passes the span that governs the measure, so what a size may be is
/// stated by the rule rather than here.
fn set_positive_number(request: &mut Value, args: &Value, key: &str, span: &Span) -> Result<()> {
    set_bounded_number(request, args, key, span, "give a size in points")
}

/// [`set_number`] for a value the shared rules bound, with the hint a refusal
/// gives: the bound is the rule's, and what a caller should do instead belongs to
/// the measure.
fn set_bounded_number(
    request: &mut Value,
    args: &Value,
    key: &str,
    span: &Span,
    hint: &str,
) -> Result<()> {
    set_number(request, args, key)?;
    if let Some(number) = request.get(key).and_then(Value::as_f64)
        && !span.contains(number)
    {
        anyhow::bail!(
            "usage: \"{key}\" must be {}, got {number} — hint: {hint}",
            span.bounds()
        );
    }
    Ok(())
}

/// The `area` a `pdf_crop` names: exactly `x`, `y`, `width` and `height`, each a
/// fraction of the page's current visible area, and the four together inside it.
/// The format silently clips an area past the page, so such an area is refused
/// here rather than answered with a smaller crop than the caller asked for.
fn require_page_area(args: &Value) -> Result<Value> {
    let Some(area) = args.get("area").filter(|value| !value.is_null()) else {
        anyhow::bail!(
            "usage: missing required argument \"area\" — hint: name x, y, width and height as \
             fractions of the page's visible area"
        );
    };
    let Some(object) = area.as_object() else {
        return Err(super::wrong_type(
            "area",
            "an object of x, y, width and height",
            area,
        ));
    };
    let names = ["x", "y", "width", "height"];
    if let Some(unknown) = object.keys().find(|key| !names.contains(&key.as_str())) {
        anyhow::bail!(
            "usage: unknown area key \"{unknown}\" — hint: an area names exactly {}",
            listed(&names)
        );
    }
    let mut sides = [0f64; 4];
    for (side, slot) in names.iter().zip(&mut sides) {
        let Some(value) = object.get(*side) else {
            anyhow::bail!(
                "usage: \"area\" has no \"{side}\" — hint: name all four of {}",
                listed(&names)
            );
        };
        *slot = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| super::wrong_type(&format!("area.{side}"), "a number", value))?;
        if !RULES.pdf_page_fraction.contains(*slot) {
            anyhow::bail!(
                "usage: \"area.{side}\" must be {}, got {} — hint: give a fraction of the page",
                RULES.pdf_page_fraction.bounds(),
                *slot
            );
        }
    }
    let [x, y, width, height] = sides;
    if width <= 0.0 || height <= 0.0 {
        anyhow::bail!(
            "usage: \"area\" must have a size, got {width}×{height} — hint: give a width and a \
             height above zero"
        );
    }
    // A tolerance, so an area a caller computed (1/3 wide and 2/3 across) is not
    // refused for the last bit a double cannot hold — the kit applies the same
    // slack, from the same rule.
    let slack = RULES.pdf_fraction_slack;
    if x + width > 1.0 + slack || y + height > 1.0 + slack {
        anyhow::bail!(
            "usage: \"area\" reaches past the page (x {x} + width {width}, y {y} + height \
             {height}) — hint: the format would silently clip it; name an area inside the page"
        );
    }
    Ok(area.clone())
}

/// Refuse an output directory that is a symlink: the reservations below name a
/// single component inside it and never follow a symlink for the file itself, but
/// a symlinked directory would take the whole call wherever it points — out of the
/// workspace, where the periodic sweep refuses to follow such an entry, so what
/// the call wrote would never be reclaimed either. A command-line call cannot
/// reach this: [`crate::cli`] hands over a destination its own
/// `resolve_output_dir` has already canonicalized, so it is never a link.
async fn refuse_symlinked_output(dir: &Path) -> Result<()> {
    if let Ok(meta) = tokio::fs::symlink_metadata(dir).await
        && meta.file_type().is_symlink()
    {
        anyhow::bail!(
            "forbidden: {} is a symlink — hint: document output needs a real directory there",
            dir.display()
        );
    }
    Ok(())
}

/// Reserve one output file per base with a create-new write of an empty file:
/// the create *is* the collision check, so two concurrent calls cannot pick the
/// same name (see [`crate::util::create_unique_file`]). The kit overwrites each
/// reserved file; a failure removes every reservation.
async fn reserve(dir: &Path, bases: &[String], ext: &str) -> Result<Vec<PathBuf>> {
    refuse_symlinked_output(dir).await?;
    tokio::fs::create_dir_all(dir)
        .await
        .with_context(|| format!("failed to create the output directory at {}", dir.display()))?;
    let mut reserved = Vec::with_capacity(bases.len());
    for base in bases {
        match reserve_one(dir, base, ext).await {
            Ok(path) => reserved.push(path),
            Err(e) => {
                remove_files(&reserved).await;
                return Err(e);
            }
        }
    }
    Ok(reserved)
}

/// Reserve one unique `base.<ext>` inside `dir` with a create-new write of an
/// empty file, dropping the handle: the kit overwrites the reserved path.
async fn reserve_one(dir: &Path, base: &str, ext: &str) -> Result<PathBuf> {
    let name = format!("{base}.{ext}");
    let (_file, path) = crate::util::create_unique_file(dir, &name)
        .await
        .with_context(|| format!("failed to reserve an output file in {}", dir.display()))?;
    Ok(path)
}

/// Remove reserved files, best effort — used on the failure path so a call that
/// produced nothing leaves nothing.
async fn remove_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = tokio::fs::remove_file(path).await;
    }
}

/// [`FILE_MARKER`] around one produced file's absolute path — the line every
/// producing operation's answer carries, the reader's included.
pub(crate) fn file_marker(path: &Path) -> String {
    format!("{FILE_MARKER}{}]", path.to_string_lossy())
}

/// A one-line description of what was produced.
pub(crate) fn describe(prefix: &str, names: &[String], sizes: &[u64]) -> String {
    if names.len() == 1 {
        return format!(
            "{prefix} {} ({})",
            names[0],
            super::listing::human_readable_size(sizes[0])
        );
    }
    let list = names
        .iter()
        .zip(sizes)
        .map(|(name, size)| format!("{name} ({})", super::listing::human_readable_size(*size)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{prefix} {} files: {list}", names.len())
}

/// Strip a string down to one safe file-name component: the directory part dropped,
/// what [`crate::util::neutralized_name`] rewrites neutralized, the reserved
/// filesystem punctuation removed, and the result trimmed of dots/spaces and
/// length-capped.
#[must_use]
fn clean_component(raw: &str) -> String {
    // Any directory part is dropped: the tool chooses the directory, and a name
    // that tried to steer it (a separator, `..`) must not be able to.
    let last = raw.rsplit(['/', '\\']).next().unwrap_or(raw);
    let cleaned: String = crate::util::neutralized_name(last)
        .chars()
        .filter(|c| !matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*'))
        .collect();
    let capped: String = cleaned
        .trim_matches(['.', ' '])
        .chars()
        .take(MAX_BASE_CHARS)
        .collect();
    capped.trim_matches(['.', ' ']).to_string()
}

/// `name` without a trailing extension the tool appends itself.
#[must_use]
fn strip_output_extension(name: &str) -> &str {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return name;
    };
    if stem.is_empty() {
        return name;
    }
    // The tool appends the OOXML families' own extensions (the sample's on a
    // fill, the chosen format on a create) and PDF; any other trailing
    // extension belongs to the model's name and is kept.
    let appended =
        crate::ooxml::family_of(Path::new(name)).is_some() || ext.eq_ignore_ascii_case("pdf");
    if appended { stem } else { name }
}

/// Sanitize a model-supplied `file_name` (or the action's own derived default)
/// to a single safe path component, falling back to the cleaned default when the
/// model's name leaves nothing usable. A trailing extension the tool appends
/// itself is stripped, so a name that already carries one is not doubled.
#[must_use]
pub(crate) fn sanitize_base(raw: Option<&str>, fallback: &str) -> String {
    let candidate = raw.map_or_else(|| clean_component(fallback), clean_component);
    let preferred = strip_output_extension(&candidate);
    if preferred.is_empty() {
        clean_component(fallback)
    } else {
        preferred.to_string()
    }
}

/// The default base name for a `create` call.
#[must_use]
fn default_base(format: &str) -> &'static str {
    match format {
        "xlsx" => "spreadsheet",
        "pptx" => "presentation",
        _ => "document",
    }
}

/// The default base name for an action that edits an input: the input's stem
/// plus what the action did to it.
#[must_use]
fn input_base(input: &Path, suffix: &str) -> String {
    format!("{}_{suffix}", crate::util::name_stem(&file_name_of(input)))
}

/// The file-name spelling of a path, or the path's whole spelling when it has
/// none — the `&Path` form of [`crate::util::file_name_or_path`], which names
/// every produced file in a reply.
#[must_use]
fn file_name_of(path: &Path) -> String {
    crate::util::file_name_or_path(&path.to_string_lossy()).to_string()
}

/// The extension a copy of `input` is written with: the input's own when its
/// name is the family its own bytes are — a `.docm` copy stays a `.docm` one,
/// macros and all — and the family's name otherwise, so a dotless or foreign
/// name never hands the copy an extension that says nothing about what is in
/// it.
#[must_use]
fn copy_extension(input: &Path, family: crate::ooxml::Family) -> String {
    if crate::ooxml::family_of(input) == Some(family)
        && let Some(ext) = input.extension()
    {
        return ext.to_string_lossy().to_ascii_lowercase();
    }
    family.name().to_string()
}

/// Paths as strings for a JSON request.
#[must_use]
fn path_strings(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::noisy_png;
    use crate::workspace::test_ws_named;
    use std::io::{Cursor, Write as _};

    /// The workspace-confined document tool — what every Assistant but the
    /// admin's gets, and the only access the tool's own rules depend on.
    fn confined_document() -> DocumentTool {
        DocumentTool::new(PathAccess::Workspace)
    }

    fn workspace() -> (tempfile::TempDir, crate::Workspace) {
        // The kit is materialized under the storage root, so a test reaching it
        // needs one resolved; the shared test root is set once per process, and
        // this is a no-op if another test set it first.
        let _ = crate::config::CONFIG.try_set_storage_root(crate::util::test::test_root().clone());
        let dir = tempfile::TempDir::new().expect("tempdir");
        // The workspace path must be canonical, as a stored workspace's is: a
        // raw `/tmp` spelling diverges from the resolved `/private/tmp` on macOS
        // and every read would be refused.
        let path = std::fs::canonicalize(dir.path()).expect("canonical workspace");
        let ws = test_ws_named(&path.to_string_lossy(), "document-test");
        (dir, ws)
    }

    /// Add one part with raw bytes to an in-progress package.
    fn add_bytes(zip: &mut zip::ZipWriter<Cursor<Vec<u8>>>, name: &str, body: &[u8]) {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .expect("start part");
        zip.write_all(body).expect("write part");
    }

    /// Write a fixture package into the workspace and return its path.
    fn write_fixture(ws: &Workspace, name: &str, bytes: &[u8]) -> PathBuf {
        let path = ws.as_path().join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    /// A package whose only part is `part`: the family verdict reads the
    /// container's own entry names, so a fixture standing for a family only has
    /// to be a ZIP holding that family's own main part.
    fn package_with(part: &str) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        add_bytes(&mut zip, part, b"<x/>");
        zip.finish().expect("finish package").into_inner()
    }

    /// A genuinely encrypted OOXML package: a CFB container declaring both
    /// streams genuine encryption writes, which is the pair the verdict reads.
    fn encrypted_package() -> Vec<u8> {
        crate::legacy::test_fixtures::cfb(&[
            ("EncryptionInfo", b"info".as_slice()),
            ("EncryptedPackage", b"data".as_slice()),
        ])
    }

    /// A legacy `.doc`: a CFB container declaring the old Word stream.
    fn legacy_doc() -> Vec<u8> {
        crate::legacy::test_fixtures::cfb(&[("WordDocument", b"legacy word".as_slice())])
    }

    /// A password-protected legacy `.doc`: the same container as [`legacy_doc`],
    /// but with the `.doc` reader's own encrypted fixture, whose FIB sets
    /// `fEncrypted` — a flag only the reader can see, so the classifier has to ask
    /// it.
    fn encrypted_legacy_doc() -> Vec<u8> {
        crate::legacy::test_fixtures::doc_fixture("secret", &[], true)
    }

    /// A minimal one-page PDF carrying a single fillable text field named
    /// `name`. The kit has no operation that authors a form, so the fixture is
    /// written here (a classic xref table keeps it readable by pdf-lib).
    fn fillable_pdf() -> Vec<u8> {
        crate::document::test_fixtures::assemble_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [5 0 R] /DA (/Helv 0 Tf 0 g) >> >>"
                .to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] \
              /Resources << /Font << /Helv 4 0 R >> >> /Annots [5 0 R] >>"
                .to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (name) /Rect [72 700 300 730] \
              /P 3 0 R /V () /DA (/Helv 12 Tf 0 g) >>"
                .to_vec(),
        ])
    }

    /// A block only a Word document can hold is refused for every other format,
    /// before an output is reserved.
    #[tokio::test]
    async fn a_docx_only_block_is_refused_for_another_format() {
        let (_dir, ws) = workspace();
        for (format, block) in [
            ("pptx", json!({ "type": "page_break" })),
            ("pdf", json!({ "type": "contents" })),
            (
                "pptx",
                json!({ "type": "link", "text": "x", "url": "https://example.com" }),
            ),
            (
                "pdf",
                json!({ "type": "paragraph", "text": "x", "footnote": "note" }),
            ),
        ] {
            let err = resolve_content(
                &ws,
                &json!({ "content": [block.clone()] }),
                format,
                PathAccess::Workspace,
            )
            .await
            .expect_err("a block only a .docx holds must be refused");
            assert!(
                err.to_string().contains(".docx"),
                "the refusal must name the format it belongs to: {err}"
            );
        }
    }

    /// A document holds one table of contents: a second is a second field over the
    /// same headings, refused rather than written beside the first.
    #[tokio::test]
    async fn a_second_table_of_contents_is_refused() {
        let (_dir, ws) = workspace();
        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "contents" }, { "type": "contents" }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("a document has one table of contents");
        assert!(
            err.to_string().contains("second table of contents"),
            "the refusal must name the second table: {err}"
        );
    }

    /// A running header or footer is a Word document's own, refused for a format that
    /// has none before an output is reserved.
    #[tokio::test]
    async fn a_running_header_is_refused_for_another_format() {
        let (_dir, ws) = workspace();
        for (format, key) in [("pptx", "header"), ("pdf", "footer")] {
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": "create", "format": format, "file_name": "made",
                        key: "Report", "content": [{ "type": "paragraph", "text": "x" }],
                    }),
                )
                .await
                .expect_err("a format without a running header must refuse one");
            assert!(
                err.to_string().contains("only a .docx has one"),
                "the refusal must say which format carries one: {err}"
            );
        }
    }

    /// An empty or whitespace-only link text, footnote, running header or running footer
    /// is content no reader shows: each is refused rather than written as an invisible
    /// part of a call that reports success.
    #[tokio::test]
    async fn blank_content_is_refused() {
        let (_dir, ws) = workspace();
        for block in [
            json!({ "type": "link", "text": "", "url": "https://example.com" }),
            json!({ "type": "paragraph", "text": "x", "footnote": " " }),
        ] {
            let err = resolve_content(
                &ws,
                &json!({ "content": [block] }),
                "docx",
                PathAccess::Workspace,
            )
            .await
            .expect_err("empty content must be refused");
            assert!(
                err.to_string().contains("is empty"),
                "the refusal must say the field is empty: {err}"
            );
        }
        for key in ["header", "footer"] {
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": "create", "format": "docx", "file_name": "blank",
                        key: " ", "content": [{ "type": "paragraph", "text": "x" }],
                    }),
                )
                .await
                .expect_err("a blank running header or footer must be refused");
            assert!(
                err.to_string().contains("is empty"),
                "the refusal must say the field is empty: {err}"
            );
        }
    }

    /// A link's address must be one an office application can follow: the schemes it
    /// follows are an allowlist, so a scheme outside it is refused; the document
    /// carries the address the parse produced rather than the raw string.
    #[tokio::test]
    async fn a_link_address_must_be_followable() {
        let (_dir, ws) = workspace();
        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "link", "text": "x", "url": "example.com" }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("an address without a scheme cannot be followed");
        assert!(
            err.to_string().contains("absolute URL"),
            "the refusal must say what a link's address is: {err}"
        );
        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "link", "text": "x", "url": "javascript:alert(1)" }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("a scheme outside the allowlist must be refused");
        assert!(
            err.to_string()
                .contains("must be an http, https, mailto or tel address"),
            "the refusal must name the schemes a link may use: {err}"
        );
        // Parsing trims the space around an address and skips an embedded tab or
        // newline, so the raw string can state an address that is not the one given
        // and that no application follows: the block carries the parsed one.
        let resolved = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "link", "text": "x", "url": " https://exa\nmple.com " }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect("a padded http address is the same address");
        assert_eq!(
            resolved[0]["url"],
            json!("https://example.com/"),
            "the document would state an address other than the one that parses: {resolved}"
        );
    }

    /// A crop's area is the caller's own fraction of the page, checked at the
    /// boundary: an area past the page, one with no size, an unknown side, a
    /// missing side and an absent area are refused before the kit is reached, so
    /// the format never answers with a smaller crop than was asked for.
    #[tokio::test]
    async fn pdf_crop_refuses_an_area_it_cannot_crop() {
        let (_dir, ws) = workspace();
        // The check runs before the file is opened as a PDF, so a stub is enough.
        let source = ws.as_path().join("stub.pdf");
        std::fs::write(&source, b"%PDF-1.4\n").expect("write stub pdf");
        for (area, expected) in [
            (
                json!({ "x": 0.5, "y": 0.0, "width": 0.6, "height": 1.0 }),
                "reaches past the page",
            ),
            (
                json!({ "x": 0.0, "y": 0.0, "width": 0.0, "height": 1.0 }),
                "must have a size",
            ),
            (
                json!({ "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "top": 0.0 }),
                "unknown area key \"top\"",
            ),
            (
                json!({ "x": 0.0, "y": 0.0, "width": 1.0 }),
                "\"area\" has no \"height\"",
            ),
            (Value::Null, "missing required argument \"area\""),
        ] {
            let mut args = json!({
                "action": "pdf_crop", "path": source.to_string_lossy(),
            });
            if !area.is_null() {
                args["area"] = area;
            }
            let err = confined_document()
                .execute(&ws, args)
                .await
                .expect_err("an area the tool cannot crop must be refused");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(
                message.contains(expected),
                "expected {expected:?}, got: {message}"
            );
        }
    }

    /// A resize names both sides: one side alone is not "keep the page's own",
    /// and the refusal says so before the kit is reached.
    #[tokio::test]
    async fn pdf_resize_refuses_a_single_side() {
        let (_dir, ws) = workspace();
        let source = ws.as_path().join("stub.pdf");
        std::fs::write(&source, b"%PDF-1.4\n").expect("write stub pdf");
        for side in ["width", "height"] {
            let mut args = json!({
                "action": "pdf_resize", "path": source.to_string_lossy(),
            });
            args[side] = json!(300);
            let err = confined_document()
                .execute(&ws, args)
                .await
                .expect_err("a resize with one side must be refused");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(
                message.contains("a resize names both sides"),
                "got: {message}"
            );
        }
    }

    /// A document's input follows the caller's access: the admin's Assistant
    /// reads a sample — and any other input — from anywhere, a guest's stays
    /// inside its own workspace.
    #[tokio::test]
    async fn inputs_follow_the_callers_access() {
        let (dir, ws) = workspace();
        let outside = tempfile::TempDir::new().unwrap();
        let sample = outside.path().join("sample.docx");
        std::fs::write(&sample, b"a sample the kit would take").unwrap();
        let path = sample.to_string_lossy().to_string();

        let err = resolve_input(&ws, &path, PathAccess::Workspace)
            .await
            .expect_err("a guest's input must stay inside its workspace");
        assert!(
            err.to_string().contains("outside the workspace"),
            "the refusal must name the boundary: {err}"
        );
        resolve_input(&ws, &path, PathAccess::Unrestricted)
            .await
            .expect("the admin's input may be read from anywhere");

        drop((dir, outside));
    }

    /// A sample the family verdict cannot take — a genuinely encrypted (CFB)
    /// `.docx`, whose container declares `EncryptionInfo` and `EncryptedPackage` —
    /// is the request's own input, so the answer names the file as a usage error
    /// rather than reporting an unexplained kit failure. A legacy container under
    /// the same name is the old format it is, never the password verdict.
    #[tokio::test]
    async fn an_unopenable_sample_is_refused_as_a_usage_error() {
        let (_dir, ws) = workspace();
        let sample = write_fixture(&ws, "encrypted.docx", &encrypted_package());

        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "path": sample.to_string_lossy(), "values": { "name": "Имя" },
                }),
            )
            .await
            .expect_err("a sample that is not a package must be refused");
        let message = err.to_string();
        assert!(message.starts_with("usage:"), "got: {message}");
        assert!(
            message.contains("password-protected"),
            "the refusal must give the read path's verdict: {message}"
        );
        assert!(
            message.contains("encrypted.docx"),
            "the refusal must name the file: {message}"
        );

        let legacy = write_fixture(&ws, "legacy.docx", &legacy_doc());
        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "path": legacy.to_string_lossy(), "values": { "name": "Имя" },
                }),
            )
            .await
            .expect_err("a legacy document is not a modern package");
        assert!(
            !err.to_string().contains("password-protected"),
            "an old-format document under a .docx name is not encrypted: {err}"
        );
    }

    /// A legacy `.doc` that is genuinely encrypted is the password verdict, not
    /// the old formats' "save it as OOXML first" sentence: only the family's own
    /// reader can see the password, so the classifier asks it — for an edit action
    /// and for a `fill_template` sample alike.
    #[tokio::test]
    async fn a_legacy_encrypted_document_is_the_password_verdict() {
        let (_dir, ws) = workspace();
        let secret = write_fixture(&ws, "secret.doc", &encrypted_legacy_doc());

        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": secret.to_string_lossy(),
                    "edits": [{ "op": "replace_text", "find": "x", "replace": "y" }],
                }),
            )
            .await
            .expect_err("an encrypted old document cannot be edited");
        let message = err.to_string();
        assert!(
            message.contains("password-protected"),
            "the edit refusal must give the read path's verdict: {message}"
        );
        assert!(
            !message.contains("save it as one of the OOXML families"),
            "a password is not a reason to convert the file: {message}"
        );

        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "path": secret.to_string_lossy(), "values": { "name": "Имя" },
                }),
            )
            .await
            .expect_err("an encrypted old document cannot be filled");
        let message = err.to_string();
        assert!(
            message.contains("password-protected"),
            "the fill refusal must give the read path's verdict: {message}"
        );
        assert!(
            message.contains("secret.doc"),
            "the refusal must name the file: {message}"
        );
    }

    /// A PDF action given a genuinely encrypted input is owed the password
    /// verdict in its own voice: what the caller is to save an unprotected copy
    /// as is a PDF, not something to edit or fill.
    #[tokio::test]
    async fn a_pdf_action_tells_an_encrypted_input_to_convert_it_to_a_pdf() {
        let (_dir, ws) = workspace();
        for (name, bytes) in [
            ("package.docx", encrypted_package()),
            ("secret.doc", encrypted_legacy_doc()),
        ] {
            let path = write_fixture(&ws, name, &bytes);
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": "pdf_rotate", "path": path.to_string_lossy(), "degrees": 90,
                    }),
                )
                .await
                .expect_err("an encrypted input cannot be read as a PDF");
            let message = err.to_string();
            assert!(
                message.contains("password-protected"),
                "{name}: the refusal must give the read path's verdict: {message}"
            );
            assert!(
                message.contains("and convert it to a PDF"),
                "{name}: a PDF action's password refusal must ask for a PDF: {message}"
            );
        }
    }

    /// A sample's family comes from its NAME — the name picks the filler, the
    /// copy's extension and the file the copy is compared against — and the name
    /// must be the family the bytes really are: a `.xls`-named sample holding an
    /// OOXML package is told to be renamed, while a real `.doc`/`.xls`/`.ppt` gets
    /// the old formats' answer — read, but never filled.
    #[tokio::test]
    async fn a_sample_is_answered_by_what_it_really_holds() {
        let (_dir, ws) = workspace();
        let cases = [
            (
                "report.doc",
                legacy_doc(),
                "an old .doc/.xls/.ppt file is read but never filled",
            ),
            (
                "sheet.xls",
                package_with("xl/workbook.xml"),
                "hint: rename it to match its content",
            ),
            (
                "chart.docx",
                b"not a container".to_vec(),
                "which is corrupt or unsupported .docx",
            ),
        ];
        for (name, bytes, expected) in cases {
            std::fs::write(ws.as_path().join(name), &bytes).expect("write sample");
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": "fill_template", "file_name": "copy",
                        "path": name, "values": { "name": "Имя" },
                    }),
                )
                .await
                .expect_err("a sample of another format must be refused");
            assert!(
                err.to_string().contains(expected),
                "{name}: expected {expected:?}, got {err}"
            );
        }
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// A fill with no values is a copy, not a fill: the boundary refuses an empty
    /// `values` object — after the sample's path is resolved, so a call that
    /// cannot reach its sample is refused for that first.
    #[tokio::test]
    async fn a_fill_with_no_values_is_refused() {
        let (_dir, ws) = workspace();
        let sample = write_fixture(&ws, "sample.docx", &package_with("word/document.xml"));
        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "path": sample.to_string_lossy(), "values": {},
                }),
            )
            .await
            .expect_err("an empty fill must be refused");
        let message = err.to_string();
        assert!(
            message.contains("missing or empty argument \"values\""),
            "got: {message}"
        );
        assert!(message.contains("placeholder"), "got: {message}");
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// A substitute is text, a number or a boolean: a structure would be written
    /// into the document as `[object Object]`, so it is refused at the boundary —
    /// for the placeholders of a sample and the values of a form alike.
    #[tokio::test]
    async fn a_structured_substitute_is_refused() {
        let (_dir, ws) = workspace();
        // The check runs before the file is opened as anything, so a stub is
        // enough to reach it.
        let sample = ws.as_path().join("sample.docx");
        std::fs::write(&sample, b"not a package").expect("write sample");

        for action in ["fill_template", "pdf_form_fill"] {
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": action, "path": sample.to_string_lossy(),
                        "values": { "name": { "a": 1 } },
                    }),
                )
                .await
                .expect_err("a structure is not a value");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(message.contains("values.name"), "got: {message}");
        }
    }

    /// Every action the schema advertises is dispatched: a name listed for the
    /// model but absent from the `match` would answer "unknown action" to a call
    /// the model was told to make.
    #[tokio::test]
    async fn every_advertised_action_is_dispatched() {
        let (_dir, ws) = workspace();
        for action in ACTIONS {
            let error = confined_document()
                .execute(&ws, json!({ "action": action }))
                .await
                .expect_err("an action with no arguments cannot succeed");
            let message = format!("{error:#}");
            assert!(!message.contains("unknown action"), "{action}: {message}");
        }
    }

    /// The whole schema is checked at the boundary: every shape the writers
    /// cannot render is refused with the position it sits at, so no argument
    /// reaches the kit as text in the produced document (`[object Object]`,
    /// `NaN`, an empty bullet) or as a failure the model cannot act on.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // reason: one boundary case per argument shape
    async fn the_content_and_sheet_shapes_are_checked_at_the_boundary() {
        let (_dir, ws) = workspace();

        let content = [
            // A table needs something to lay out, and the presentation writer
            // rejects an empty one outright.
            (
                json!({ "content": [{ "type": "table", "rows": [] }] }),
                "no rows and no headers",
            ),
            (
                json!({ "content": [{ "type": "table", "headers": [], "rows": [] }] }),
                "no rows and no headers",
            ),
            // A row, a cell, a header or a bullet the writers cannot render: a
            // structure reaches the document as `[object Object]`.
            (
                json!({ "content": [{ "type": "table", "rows": ["not a row"] }] }),
                "content[0].rows[0]",
            ),
            (
                json!({ "content": [{ "type": "table", "rows": [[{ "a": 1 }]] }] }),
                "content[0].rows[0][0]",
            ),
            (
                json!({ "content": [{ "type": "table", "headers": [{ "a": 1 }], "rows": [["a"]] }] }),
                "content[0].headers[0]",
            ),
            (
                json!({ "content": [{ "type": "list", "items": [{ "a": 1 }] }] }),
                "content[0].items[0]",
            ),
            // A heading level the writers do not have — refused, not clamped to
            // one they do.
            (
                json!({ "content": [{ "type": "heading", "text": "h", "level": 7 }] }),
                "level must be 1, 2 or 3",
            ),
            (
                json!({ "content": [{ "type": "heading", "text": "h", "level": "2" }] }),
                "level must be 1, 2 or 3",
            ),
            // A list's `ordered` decides a rendering only a boolean can pick.
            (
                json!({ "content": [{ "type": "list", "items": ["a"], "ordered": "yes" }] }),
                "content[0].ordered",
            ),
            // A size the layout has no meaning for.
            (
                json!({ "content": [{ "type": "image", "image": "x.png", "width": { "a": 1 } }] }),
                "content[0].width",
            ),
            (
                json!({ "content": [{ "type": "image", "image": "x.png", "height": -1 }] }),
                "content[0].height",
            ),
            (
                json!({ "content": [{ "type": "image", "image": "x.png", "width": 1e30 }] }),
                "content[0].width",
            ),
        ];
        for (args, expected) in content {
            let err = resolve_content(&ws, &args, "docx", PathAccess::Workspace)
                .await
                .expect_err("a shape the writers cannot render");
            assert!(
                err.to_string().contains(expected),
                "expected the refusal to name {expected}: {err}"
            );
        }

        // The same for a workbook's sheets and cells.
        let long_name = "a".repeat(RULES.sheet_name_max + 1);
        let sheets = [
            (
                json!({ "sheets": [{ "name": 5, "rows": [["a"]] }] }),
                "sheets[0].name",
            ),
            // A name the writer cannot hold: past the shared limit, holding a
            // forbidden character, or shared with an earlier sheet ignoring case.
            (
                json!({ "sheets": [{ "name": long_name, "rows": [["a"]] }] }),
                "sheets[0].name",
            ),
            (
                json!({ "sheets": [{ "name": "bad:name", "rows": [["a"]] }] }),
                "sheets[0].name",
            ),
            (
                json!({ "sheets": [
                    { "name": "Trips", "rows": [["a"]] },
                    { "name": "trips", "rows": [["b"]] },
                ] }),
                "sheets[1].name",
            ),
            (
                json!({ "sheets": [{ "rows": [[null]] }] }),
                "sheets[0].rows[0][0]",
            ),
            (
                json!({ "sheets": [{ "rows": [[{ "b": 1 }]] }] }),
                "sheets[0].rows[0][0]",
            ),
            (
                json!({ "sheets": [{ "rows": [[{ "formula": "  " }]] }] }),
                "sheets[0].rows[0][0]",
            ),
            // A formula that says nothing: the leading `=` is not a formula.
            (
                json!({ "sheets": [{ "rows": [[{ "formula": "=" }]] }] }),
                "sheets[0].rows[0][0]",
            ),
            // A second `=` survives normalization and would land in the cell's
            // `<f>` element as text: refused, not handed over.
            (
                json!({ "sheets": [{ "rows": [[{ "formula": "==SUM(A1:A2)" }]] }] }),
                "sheets[0].rows[0][0]",
            ),
            (json!({ "sheets": [[["a"]]] }), "sheets[0]"),
        ];
        for (args, expected) in sheets {
            let err = resolve_sheets(&args).expect_err("a shape the writer cannot write");
            assert!(
                err.to_string().contains(expected),
                "expected the refusal to name {expected}: {err}"
            );
        }

        // `headers` stays optional, null counts as absent, and a well-formed
        // table and sheet pass.
        let good = json!({ "content": [{ "type": "table", "rows": [["a", "b"]] }] });
        assert!(
            resolve_content(&ws, &good, "docx", PathAccess::Workspace)
                .await
                .is_ok()
        );
        let good =
            json!({ "content": [{ "type": "table", "headers": null, "rows": [["a", "b"]] }] });
        assert!(
            resolve_content(&ws, &good, "docx", PathAccess::Workspace)
                .await
                .is_ok()
        );
        let headers_only = json!({ "content": [{ "type": "table", "headers": ["a"] }] });
        assert!(
            resolve_content(&ws, &headers_only, "docx", PathAccess::Workspace)
                .await
                .is_ok()
        );
        let good = json!({ "sheets": [{ "name": "Лист", "rows": [["a", 1, true, { "formula": "=SUM(A1:A2)" }]] }] });
        let normalized = resolve_sheets(&good).expect("a well-formed sheet");
        assert_eq!(
            normalized[0]["rows"][0][3],
            json!({ "formula": "SUM(A1:A2)" }),
            "the formula must reach the request without its leading `=`"
        );
    }

    /// A table block takes the one shape both producers write: an empty set is a
    /// set of nothing to lay out, so the boundary drops it from the block the
    /// request carries instead of handing a writer a cell-less row — a construct
    /// Word refuses to open; a table left with nothing is refused; and a `.docx`
    /// table is held to Word's column bound, headings row included, by `create`
    /// exactly as by `add_table`.
    #[tokio::test]
    async fn a_table_block_takes_the_one_shape_both_producers_write() {
        let (_dir, ws) = workspace();

        let no_headers = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "headers": [], "rows": [["a"]] }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect("a table with rows is a table");
        assert!(
            no_headers[0].get("headers").is_none(),
            "an empty header set must be dropped: {no_headers}"
        );

        let no_empty_row = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "rows": [[], ["a"]] }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect("a table with one cell is a table");
        assert_eq!(no_empty_row[0]["rows"], json!([["a"]]));

        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "headers": [], "rows": [[], []] }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("a table of no cells must still be refused");
        assert!(
            err.to_string().contains("no rows and no headers"),
            "got: {err}"
        );

        // The column bound is the adding operation's too: one table shape, whether
        // `create` lays it down or an `add_table` places it in an existing document.
        let wide = vec![json!("c"); RULES.table_columns_max + 1];
        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "rows": [wide.clone()] }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("a table past the column bound must be refused");
        assert!(
            err.to_string().contains("a table holds at most"),
            "got: {err}"
        );
        // The headings row is a row of cells too: it is held to the same bound, so
        // the library never writes a Word-invalid heading row this boundary passed.
        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "headers": wide.clone(), "rows": [["a"]] }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("a headings row past the column bound must be refused");
        assert!(
            err.to_string().contains("a table holds at most"),
            "got: {err}"
        );
        // The bound is Word's, not every format's: a presentation's table has no
        // such limit.
        let deck = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "rows": [wide] }] }),
            "pptx",
            PathAccess::Workspace,
        )
        .await
        .expect("a presentation's table has no column bound");
        assert_eq!(
            deck[0]["rows"],
            json!([vec![json!("c"); RULES.table_columns_max + 1]])
        );
    }

    /// Every argument of an action that produces a file is checked before an
    /// output is reserved, so a call the tool refuses leaves nothing behind in
    /// the output directory — and a file that is not the image it is named as is
    /// refused rather than embedded.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // one case per argument, all refused before anything is reserved
    async fn the_arguments_are_checked_before_an_output_is_reserved() {
        let (_dir, ws) = workspace();
        let image = ws.as_path().join("picture.png");
        std::fs::write(&image, noisy_png(4, 4)).expect("write png");
        let notes = ws.as_path().join("notes.txt");
        std::fs::write(&notes, b"not an image").expect("write notes");
        let document = ws.as_path().join("document.pdf");
        std::fs::write(&document, fillable_pdf()).expect("write pdf");
        let merged = ws.as_path().join("merged.pdf");
        std::fs::write(&merged, fillable_pdf()).expect("write pdf");
        let path = document.to_string_lossy();
        // An input the kit loads whole is bounded: an oversized one must be
        // refused on its own merits rather than as a killed run reported as a
        // product fault. Sparse, so the test costs no disk.
        let huge = ws.as_path().join("huge.pdf");
        std::fs::File::create(&huge)
            .and_then(|file| file.set_len(crate::util::FILE_MAX_BYTES + 1))
            .expect("a sparse oversized input");
        // A merge holds every input at once, so the whole set is bounded too:
        // five files, each within the per-file limit and together over what one
        // call may load. Sparse, so the test costs no disk.
        let mut batch = Vec::new();
        for index in 0..5 {
            let member = ws.as_path().join(format!("batch{index}.pdf"));
            std::fs::File::create(&member)
                .and_then(|file| file.set_len(crate::util::FILE_MAX_BYTES))
                .expect("a sparse input at the per-file limit");
            batch.push(member.to_string_lossy().into_owned());
        }

        let calls = [
            (
                json!({ "action": "create", "format": "docx", "file_name": 5,
                        "content": [{ "type": "paragraph", "text": "x" }] }),
                "file_name",
            ),
            // The argument the chosen format does not use is named, never
            // dropped in silence.
            (
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "rows": [["a"]] }],
                        "content": [{ "type": "paragraph", "text": "x" }] }),
                "\"content\" is not used for format \"xlsx\"",
            ),
            (
                json!({ "action": "create", "format": "docx",
                        "content": [{ "type": "paragraph", "text": "x" }],
                        "sheets": [{ "rows": [["a"]] }] }),
                "\"sheets\" is not used for format \"docx\"",
            ),
            (
                json!({ "action": "pdf_text", "path": path, "text": "x", "color": "red" }),
                "color",
            ),
            (
                json!({ "action": "pdf_text", "path": path, "text": "x", "size": 0 }),
                "size",
            ),
            // A number no page can carry: pdf-lib writes it through, and the
            // caller would receive a file nothing can open as a success.
            (
                json!({ "action": "pdf_text", "path": path, "text": "x", "size": 1e300 }),
                "within ±20000",
            ),
            (
                json!({ "action": "pdf_image", "path": path, "image": image.to_string_lossy(),
                        "x": -1e300 }),
                "within ±20000",
            ),
            // A wrong-typed `flatten` is read before the output is reserved.
            (
                json!({ "action": "pdf_form_fill", "path": path, "values": { "name": "x" },
                        "flatten": "yes" }),
                "flatten",
            ),
            (
                json!({ "action": "pdf_image", "path": path,
                        "image": image.to_string_lossy(), "width": -5 }),
                "width",
            ),
            (
                json!({ "action": "pdf_image", "path": path, "image": notes.to_string_lossy() }),
                "PNG or JPEG",
            ),
            (
                json!({ "action": "pdf_rotate", "path": path, "degrees": 45 }),
                "multiple of 90",
            ),
            // A whole turn (or zero) is refused: the pages would come back
            // unchanged.
            (
                json!({ "action": "pdf_rotate", "path": path, "degrees": 360 }),
                "would leave the pages as they are",
            ),
            // A form filled with nothing says nothing.
            (
                json!({ "action": "pdf_form_fill", "path": path, "values": {} }),
                "empty",
            ),
            (
                json!({ "action": "pdf_rotate", "path": huge.to_string_lossy(), "degrees": 90 }),
                "MB limit",
            ),
            (
                json!({ "action": "pdf_rotate", "path": path, "degrees": 90, "pages": ["0"] }),
                "page 0 does not exist",
            ),
            (
                json!({ "action": "pdf_merge", "paths": [merged.to_string_lossy()] }),
                "at least 2",
            ),
            (
                json!({ "action": "pdf_merge",
                        "paths": (0..=MAX_MERGE_INPUTS).map(|index| format!("part{index}.pdf")).collect::<Vec<_>>() }),
                "limit 50",
            ),
            (
                json!({ "action": "pdf_merge", "paths": batch }),
                "more than one merge takes",
            ),
            (
                json!({ "action": "pdf_split", "path": path,
                        "pages": (1..=MAX_SPLIT_PARTS + 1).map(|n| n.to_string()).collect::<Vec<_>>() }),
                "limit 50",
            ),
            // The page cap is on the pages the request names, not on the
            // document's length: a one-page file still refuses a selection that
            // names more pages than one call takes.
            (
                json!({ "action": "pdf_split", "path": path,
                        "pages": [format!("1-{}", RULES.max_pages_per_request + 1)] }),
                "limit 10000",
            ),
            (
                json!({ "action": "pdf_split", "path": path, "pages": ["3-1"] }),
                "reversed",
            ),
        ];
        for (args, expected) in calls {
            let err = confined_document()
                .execute(&ws, args)
                .await
                .expect_err("a call the tool cannot carry out");
            assert!(
                err.to_string().contains(expected),
                "expected the refusal to name {expected}: {err}"
            );
        }
        let left = std::fs::read_dir(ws.as_path().join(GENERATED_DIR))
            .map_or(0, std::iter::Iterator::count);
        assert_eq!(left, 0, "a refused call left an output behind");
    }

    /// A metadata date is canonicalised to the PDF's own spelling: a calendar
    /// date is midnight UTC, `T` separates the time, seconds and offsets are
    /// optional, and a zero offset is written `Z`.
    #[test]
    fn pdf_dates_are_written_in_the_pdfs_own_spelling() {
        assert_eq!(
            pdf_date("2024-05-17").expect("a calendar date"),
            "D:20240517000000Z"
        );
        assert_eq!(
            pdf_date("2024-05-17T10:30").expect("a date and minutes"),
            "D:20240517103000Z"
        );
        assert_eq!(
            pdf_date("2024-05-17T10:30:15Z").expect("a date, seconds and Z"),
            "D:20240517103015Z"
        );
        assert_eq!(
            pdf_date("2024-05-17T10:30:00+03:00").expect("a positive offset"),
            "D:20240517103000+03'00'"
        );
        assert_eq!(pdf_date("").expect("an empty string clears"), "");
        for bad in [
            "2024-13-01",
            "2024-05-17T10:30:00 o'clock",
            "2024-05-17 10:30:15Z",
            "2024-05-17T10:30:00-0530",
        ] {
            let err = pdf_date(bad).expect_err("a bad date must be refused");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(
                message.contains(bad),
                "the refusal must echo {bad:?}: {message}"
            );
            assert!(
                message.contains("YYYY-MM-DD"),
                "the refusal must spell the accepted form: {message}"
            );
        }
    }

    /// A workspace-only tool refuses a path outside the workspace.
    #[tokio::test]
    async fn refuses_a_sample_outside_the_workspace() {
        let (_dir, ws) = workspace();
        let outside = tempfile::NamedTempFile::new().expect("temp file");
        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "fill_template",
                    "path": outside.path().to_string_lossy(),
                    "values": {},
                }),
            )
            .await
            .expect_err("a sample outside the workspace must be refused");
        assert!(
            err.to_string().contains("outside the workspace"),
            "got: {err}"
        );
    }

    /// A `generated` that is a symlink would take every output write wherever it
    /// points, so the reservation refuses it before it creates anything.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_generated_directory_is_refused() {
        let (dir, ws) = workspace();
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).expect("target directory");
        let generated = ws.as_path().join("generated");
        std::os::unix::fs::symlink(&elsewhere, &generated).expect("symlink generated");

        let err = reserve(&generated, &["report".to_string()], "docx")
            .await
            .expect_err("a symlinked generated directory must be refused");
        assert!(err.to_string().starts_with("forbidden:"), "got: {err}");
        assert!(
            std::fs::read_dir(&elsewhere)
                .expect("target directory")
                .next()
                .is_none(),
            "nothing may be written through the symlink"
        );
    }

    /// The generated directory's entry count: how a test proves a refused call
    /// left no output behind.
    fn generated_count(ws: &Workspace) -> usize {
        std::fs::read_dir(ws.as_path().join(GENERATED_DIR)).map_or(0, std::iter::Iterator::count)
    }

    /// The three edit actions address a file's own FAMILY, decided from its own
    /// bytes and never from its name: a package of another family, a PDF, a
    /// container that is no document and an old `.doc`/`.xls`/`.ppt` are each
    /// refused for what they really are — the legacy one with the hint that it is
    /// read but never edited — before a runtime is spawned.
    #[tokio::test]
    async fn editing_refuses_an_input_of_another_family() {
        let (_dir, ws) = workspace();
        let legacy = "an old .doc/.xls/.ppt file is read but never edited";
        let cases = [
            // The input's own bytes, not its name, are what each action takes.
            ("report.doc", legacy_doc(), "docx_edit", legacy),
            ("sheet.xls", legacy_doc(), "xlsx_edit", legacy),
            ("deck.ppt", legacy_doc(), "pptx_edit", legacy),
            (
                "paper.pdf",
                fillable_pdf(),
                "docx_edit",
                "edits only a .docx/.docm file",
            ),
            (
                "paper.pdf",
                fillable_pdf(),
                "xlsx_edit",
                "edits only a .xlsx/.xlsm file",
            ),
            (
                "paper.pdf",
                fillable_pdf(),
                "pptx_edit",
                "edits only a .pptx/.pptm file",
            ),
            // A real package under another family's name is refused as the family
            // it really holds, and pointed at the action that edits it.
            (
                "report.docx",
                package_with("xl/workbook.xml"),
                "docx_edit",
                "which is really an .xlsx package",
            ),
            (
                "sheet.xlsx",
                package_with("ppt/presentation.xml"),
                "xlsx_edit",
                "which is really a .pptx package",
            ),
            // A container that is no document is not called one.
            (
                "notes.msg",
                crate::legacy::test_fixtures::cfb(&[("__properties_version1.0", b"outlook")]),
                "docx_edit",
                "which is not a Word, Excel, PowerPoint or PDF document",
            ),
        ];
        for (name, bytes, action, expected) in cases {
            std::fs::write(ws.as_path().join(name), &bytes).expect("write input");
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": action, "path": name,
                        "edits": [{ "op": "replace_text", "find": "x", "replace": "y" }],
                    }),
                )
                .await
                .expect_err("a wrong-family input must be refused");
            assert!(
                err.to_string().contains(expected),
                "{action} on {name}: expected {expected:?}, got {err}"
            );
        }
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");

        // A genuinely encrypted package is the password verdict whatever its name
        // says — and an old-format document under that same name is not.
        for (name, bytes, expected) in [
            ("encrypted.docx", encrypted_package(), "password-protected"),
            ("container.doc", legacy_doc(), legacy),
        ] {
            std::fs::write(ws.as_path().join(name), &bytes).expect("write input");
            let err = confined_document()
                .execute(
                    &ws,
                    json!({
                        "action": "docx_edit", "path": name,
                        "edits": [{ "op": "replace_text", "find": "x", "replace": "y" }],
                    }),
                )
                .await
                .expect_err("a container is refused for what it really holds");
            assert!(
                err.to_string().contains(expected),
                "{name}: expected {expected:?}, got {err}"
            );
        }
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// A `.doc` name is not an old document: a ZIP container named `.doc` whose
    /// bytes hold no package is refused for what its bytes are, never with the
    /// sentence the old binary formats are owed — that sentence follows a real
    /// `LegacyDoc`/`LegacyXls`/`LegacyPpt` verdict, not the name.
    #[tokio::test]
    async fn a_doc_named_zip_is_not_an_old_doc() {
        let (_dir, ws) = workspace();
        let archive = write_fixture(&ws, "archive.doc", &package_with("mimetype"));
        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": archive.to_string_lossy(),
                    "edits": [{ "op": "replace_text", "find": "x", "replace": "y" }],
                }),
            )
            .await
            .expect_err("a ZIP that holds no package is not a docx");
        let message = err.to_string();
        assert!(
            message.contains("which is not a Word, Excel, PowerPoint or PDF document"),
            "got: {message}"
        );
        assert!(
            !message.contains("old .doc/.xls/.ppt"),
            "a container that is no old document must not be called one: {message}"
        );
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// A file's name is not its format: a valid PDF under a `.txt` name is taken
    /// by a `pdf_*` action — the verdict reads the bytes — while a file that is
    /// not a PDF is refused naming what it really holds.
    #[tokio::test]
    async fn a_pdf_is_taken_under_any_name() {
        let (_dir, ws) = workspace();
        let renamed = write_fixture(&ws, "notes.txt", &fillable_pdf());
        require_pdf(&renamed)
            .await
            .expect("a PDF under any name is a PDF");

        let workbook = write_fixture(&ws, "sheet.txt", &package_with("xl/workbook.xml"));
        let err = require_pdf(&workbook)
            .await
            .expect_err("a workbook is not a PDF, whatever it is called");
        assert!(
            err.to_string().contains("which is really an .xlsx package"),
            "got: {err}"
        );

        // The whole call follows the same verdict: what the renamed PDF meets is
        // the managed runtime, never a refusal about its format.
        if let Err(e) = confined_document()
            .execute(
                &ws,
                json!({ "action": "pdf_text", "path": "notes.txt", "text": "x" }),
            )
            .await
        {
            assert!(!e.to_string().contains("must be a PDF"), "got: {e}");
        }
    }

    /// `pages` is one argument with one grammar for every PDF action that takes
    /// it: an array of selections, whose elements are the files a split cuts and
    /// whose union is what the rotate/text/image actions act on. A page named
    /// twice anywhere in the request is refused, before the kit is spawned.
    #[tokio::test]
    async fn the_pages_argument_is_an_array_of_selections() {
        let (_dir, ws) = workspace();
        assert_eq!(
            split_groups(&json!({ "pages": ["1-3", "5"] })).expect("two selections"),
            vec![json!([1, 2, 3]), json!([5])],
            "one output file per element, in order"
        );
        // `"all"` is the same element an acting action takes, and the kit resolves
        // it against the document's own page count — the one place it is known.
        assert_eq!(
            split_groups(&json!({ "pages": ["all"] })).expect("one file, every page"),
            vec![json!("all")],
            "an element naming every page is passed through unexpanded"
        );
        assert_eq!(
            opt_pages(&json!({ "pages": ["1-3", "5"] }))
                .expect("a selection list")
                .kit_value(),
            json!([1, 2, 3, 5]),
            "the union of the elements is what the action acts on"
        );
        for pages in [
            json!({}),
            json!({ "pages": null }),
            json!({ "pages": [] }),
            json!({ "pages": ["all"] }),
        ] {
            assert_eq!(
                opt_pages(&pages).expect("every page").kit_value(),
                json!("all"),
                "{pages} names no page to narrow the call to"
            );
        }
        // An empty element names no page: `"all"` is the only element that means
        // every page, so a blank one is refused as an empty selection rather than
        // silently widening the call.
        for pages in [json!({ "pages": [""] }), json!({ "pages": ["1-3", ""] })] {
            let err = opt_pages(&pages)
                .map(|pages| pages.kit_value())
                .expect_err("a blank page element must be refused");
            assert!(
                err.to_string().contains("empty page range"),
                "{pages}: got {err}"
            );
        }
        let duplicate = split_groups(&json!({ "pages": ["1", "2", "1"] }))
            .expect_err("a page named twice is refused");
        assert!(
            duplicate.to_string().contains("appears more than once"),
            "got: {duplicate}"
        );
        // An `"all"` element stands alone: beside another element it would name the
        // same page twice, the way the named duplicate above does.
        for pages in [
            json!({ "pages": ["all", "1"] }),
            json!({ "pages": ["1-3", "all"] }),
            json!({ "pages": ["all", "all"] }),
        ] {
            let err = split_groups(&pages).expect_err("an \"all\" element stands alone");
            assert!(
                err.to_string()
                    .contains("cannot sit beside another selection"),
                "split {pages}: got {err}"
            );
            let err = opt_pages(&pages)
                .map(|pages| pages.kit_value())
                .expect_err("an \"all\" element stands alone");
            assert!(
                err.to_string()
                    .contains("cannot sit beside another selection"),
                "{pages}: got {err}"
            );
        }

        let pdf = write_fixture(&ws, "base.pdf", &fillable_pdf());
        let err = confined_document()
            .execute(
                &ws,
                json!({ "action": "pdf_rotate", "path": pdf.to_string_lossy(),
                        "degrees": 90, "pages": ["1-2", "2"] }),
            )
            .await
            .expect_err("a page named twice is refused");
        assert!(
            err.to_string().contains("appears more than once"),
            "got: {err}"
        );
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// Both call shapes are accepted: the flat one (an `action` string with the
    /// action's own arguments beside it) and the nested one the schema literally
    /// describes, whose single key names the action — and the chrome-shaped call
    /// whose arguments sit under the action's own key. Anything else is refused
    /// with the shape the caller was expected to use.
    #[tokio::test]
    async fn a_nested_action_call_is_dispatched_like_a_flat_one() {
        let (_dir, ws) = workspace();
        assert_eq!(
            normalize_call(&json!({
                "action": { "pdf_rotate": { "path": "a.pdf", "degrees": 90 } },
                "file_name": "turned",
            }))
            .expect("the nested shape"),
            json!({
                "action": "pdf_rotate", "path": "a.pdf", "degrees": 90, "file_name": "turned"
            })
        );
        assert_eq!(
            normalize_call(&json!({ "action": "pdf_rotate", "pdf_rotate": { "degrees": 90 } }))
                .expect("the chrome shape"),
            json!({ "action": "pdf_rotate", "degrees": 90 })
        );
        // An argument stated beside the action object is the caller's: it is kept
        // rather than dropped, so a nested call cannot widen into a call the
        // caller did not make.
        assert_eq!(
            normalize_call(&json!({
                "action": { "pdf_text": { "path": "a.pdf", "text": "x" } },
                "pages": ["1"],
            }))
            .expect("the nested shape with a selection beside it"),
            json!({ "action": "pdf_text", "path": "a.pdf", "text": "x", "pages": ["1"] })
        );
        for bad in [
            json!({ "action": 5 }),
            json!({ "action": ["pdf_rotate"] }),
            json!({ "action": { "a": 1, "b": 2 } }),
            json!({ "action": {} }),
        ] {
            let err = normalize_call(&bad).expect_err("a shape the tool cannot read");
            assert!(err.to_string().contains("\"action\""), "{bad}: {err}");
        }

        // Dispatch follows the normalisation: the nested call meets the same
        // refusal the flat one does.
        let pdf = write_fixture(&ws, "base.pdf", &fillable_pdf());
        let err = confined_document()
            .execute(
                &ws,
                json!({
                    "action": { "pdf_rotate": { "path": pdf.to_string_lossy(), "degrees": 45 } }
                }),
            )
            .await
            .expect_err("45 degrees is not a multiple of 90");
        assert!(err.to_string().contains("multiple of 90"), "got: {err}");
    }

    /// A call is checked against the action's own schema entry: a key the action
    /// has no place for is refused by name rather than silently ignored, so a
    /// renamed argument names itself instead of surfacing as a "missing required
    /// argument" that points at the wrong key. An action the tool does not have
    /// keeps its own refusal and is not pre-empted.
    #[tokio::test]
    async fn an_action_refuses_an_argument_it_does_not_take() {
        let (_dir, ws) = workspace();
        for (args, expected) in [
            // `pages` belongs to the PDF page actions, not to a form fill.
            (
                json!({ "action": "pdf_form_fill", "path": "form.pdf", "pages": ["1"],
                        "values": { "name": "Иван" } }),
                "takes no argument \"pages\"",
            ),
            // `path` is where the sample goes now; `template` was the kit's own
            // field name and is no argument of the action.
            (
                json!({ "action": "fill_template", "template": "sample.docx",
                        "values": { "name": "Иван" } }),
                "takes no argument \"template\"",
            ),
        ] {
            let err = confined_document()
                .execute(&ws, args)
                .await
                .expect_err("an argument the action does not take must be refused");
            assert!(
                err.to_string().contains(expected),
                "expected {expected:?}: {err}"
            );
        }
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");

        let err = confined_document()
            .execute(&ws, json!({ "action": "no_such_action" }))
            .await
            .expect_err("an action the tool does not have");
        assert!(err.to_string().contains("unknown action"), "got: {err}");
    }

    /// Every edit shape the kit would fail on is refused at the boundary,
    /// naming the offending edit and field, and before an output is reserved.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // reason: one refused edit shape per op and field
    async fn edit_shapes_are_checked_before_an_output_is_reserved() {
        let (_dir, ws) = workspace();
        // The input's own bytes decide the family, so a fixture standing for one
        // is a package holding that family's own main part.
        for (name, part) in [
            ("doc.docx", "word/document.xml"),
            ("book.xlsx", "xl/workbook.xml"),
            ("deck.pptx", "ppt/presentation.xml"),
        ] {
            std::fs::write(ws.as_path().join(name), package_with(part)).expect("write input");
        }
        let too_many: Vec<Value> = (0..=RULES.edits_max)
            .map(|_| json!({ "op": "replace_text", "find": "x", "replace": "y" }))
            .collect();
        let long_style = "a".repeat(RULES.paragraph_style_max + 1);
        let style_cap = format!("must be at most {} characters", RULES.paragraph_style_max);
        let wide_row: Vec<Value> = (0..=RULES.table_columns_max)
            .map(|column| json!(column))
            .collect();
        let calls = [
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "nope", "find": "x" }] }),
                "unknown docx edit",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "nope", "sheet": "Values", "cell": "A1" }] }),
                "unknown xlsx edit",
            ),
            (
                json!({ "action": "pptx_edit", "path": "deck.pptx",
                        "edits": [{ "op": "nope", "slide": 1 }] }),
                "unknown pptx edit",
            ),
            // A lookup of nothing but whitespace — a `find` for a text op and the
            // `after` anchor an inserted paragraph is placed by — names no text a
            // slide holds, so it is refused before the deck is read rather than
            // searched for.
            (
                json!({ "action": "pptx_edit", "path": "deck.pptx",
                        "edits": [{ "op": "replace_text", "slide": 1, "find": "  ",
                                    "replace": "x" }] }),
                "must name text",
            ),
            (
                json!({ "action": "pptx_edit", "path": "deck.pptx",
                        "edits": [{ "op": "add_paragraph", "slide": 1, "text": "X",
                                    "after": "\t" }] }),
                "must name text",
            ),
            // A `find` an op needs, a `format_text` with nothing to set, and an
            // `insert_text` in the wrong position.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "replace_text", "replace": "y" }] }),
                "must not be empty",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_text", "find": "x" }] }),
                "at least one of bold",
            ),
            // A `format_text` whose only property is a null has nothing to set
            // either: the null is "absent" at the boundary, so it is refused here
            // rather than forwarded and read by the kit as a value.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_text", "find": "x", "bold": null }] }),
                "at least one of bold",
            ),
            // A `format_text` size outside the shared bound, and one JSON cannot
            // carry as a finite number: the kit writes half-points, so either
            // would otherwise be rounded into a `<w:sz>` a reader cannot hold. A
            // non-finite float is JSON null — the shape JSON has for it — which
            // is an absent property, so the edit with nothing else to set is
            // refused rather than succeeding with no formatting.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_text", "find": "x", "size": 1e300 }] }),
                "size must be",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_text", "find": "x", "size": f64::INFINITY }] }),
                "at least one of bold",
            ),
            // A `format_text` naming an attribute the op does not take beside one
            // it does: the kit reads only the fields it knows, so the extra one
            // would be dropped in silence.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_text", "find": "x", "bold": true,
                                    "level": 1 }] }),
                "level is not a field of docx",
            ),
            // A known attribute whose value the writer has not: a colour that is
            // not hex digits, and a paragraph alignment the closed set does not
            // hold.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_text", "find": "x", "color": "red" }] }),
                "got \"red\"",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x",
                                    "align": "middle" }] }),
                "got \"middle\"",
            ),
            // A `format_paragraph` naming no property: it would otherwise rewrite
            // the paragraph and change nothing.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x" }] }),
                "at least one of align",
            ),
            // A `level` with no list to belong to: alone, and beside `list: false`.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x", "level": 2 }] }),
                "needs a list to belong to",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x", "list": false,
                                    "level": 2 }] }),
                "needs a list to belong to",
            ),
            // A list kind outside the closed set, and a style name past the cap.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x",
                                    "list": "dash" }] }),
                "got \"dash\"",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x",
                                    "style": long_style }] }),
                style_cap.as_str(),
            ),
            // An indent and a line height outside the shared bounds, the line
            // height's zero included: a line that takes no room is not a height.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x",
                                    "indent_left": 3000 }] }),
                "indent_left must be",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x",
                                    "line_spacing": 0 }] }),
                "line_spacing must be",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "format_paragraph", "find": "x",
                                    "line_spacing": 3000 }] }),
                "line_spacing must be",
            ),
            // A well-formed `format_paragraph` passes the boundary's own check:
            // the refusal it meets is the family one a path that is not a `.docx`
            // is owed, after the shape was accepted.
            (
                json!({ "action": "docx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_paragraph", "find": "x", "align": "center",
                                    "list": "bullet", "level": 2 }] }),
                "edits only a .docx/.docm file",
            ),
            // A `set_cell` needs a `value`: a value-less one is refused with a
            // pointer to the `format_cells` op, which formats without writing one.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "set_cell", "sheet": "Values", "cell": "A1",
                                    "number_format": "0.00" }] }),
                "has no \"value\"",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "insert_text", "find": "x", "insert": "y",
                                    "position": "middle" }] }),
                "position",
            ),
            // The two block-adding ops: an anchor and a picture are required, a
            // picture's width is a point measure above zero, and a table needs
            // rows of cells no wider than Word draws a row — a headings row alone
            // is a table, exactly as it is for `create`.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "add_image", "image": "picture.png" }] }),
                "after must not be empty",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "add_image", "after": "x", "image": "picture.png",
                                    "width": 0 }] }),
                "width must be",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "add_table", "after": "x" }] }),
                "a table with no rows and no headers",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "add_table", "after": "x", "rows": [[]] }] }),
                "a table with no rows and no headers",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "add_table", "after": "x", "rows": [wide_row] }] }),
                "a table holds at most",
            ),
            // A cell, a row and a column outside the workbook's grid.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "set_cell", "sheet": "Values", "cell": "A9999999",
                                    "value": 1 }] }),
                "the row must be between",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "set_cell", "sheet": "Values", "cell": "ZZZ1",
                                    "value": 1 }] }),
                "the column must be between",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "set_cell", "sheet": "Values", "cell": "A0",
                                    "value": 1 }] }),
                "must be an A1 address",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "insert_row", "sheet": "Values", "row": 0 }] }),
                "row must be a whole number",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "insert_column", "sheet": "Values", "column": "A1" }] }),
                "column must be column letters",
            ),
            // A `format_cells` target's own address is checked like every other
            // op's: a malformed or lowercase column, and a row past the grid, are
            // refused here rather than by the kit.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "column": "A1",
                                    "width_chars": 10 }] }),
                "column must be column letters",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "column": "b",
                                    "width_chars": 10 }] }),
                "column must be column letters",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "row": 99_999_999,
                                    "height_points": 20 }] }),
                "row must be a whole number",
            ),
            // A xlsx `format_cells` naming no target, more than one target, or only
            // the target and none of the properties it takes: the op would
            // otherwise rewrite the sheet and change nothing.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "bold": true }] }),
                "names no target",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "column": "B", "bold": true }] }),
                "names more than one target",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1" }] }),
                "names no formatting",
            ),
            // A property that is the wrong type, and one the target does not take:
            // the kit reads only the keys it knows, so an extra one would be
            // dropped in silence.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "bold": "yes" }] }),
                "must be a boolean",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "shade": "FF0000" }] }),
                "is not a field of xlsx",
            ),
            // The same for a docx key the op does not take: a field named for
            // another op (`position` is an `insert_text`'s) would otherwise be
            // dropped in silence.
            (
                json!({ "action": "docx_edit", "path": "doc.docx",
                        "edits": [{ "op": "replace_text", "find": "x", "replace": "y",
                                    "position": "after" }] }),
                "is not a field of docx",
            ),
            // An alignment word and a border side the closed sets do not hold, and
            // a border that names no side at all.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "align": "middle" }] }),
                "align must be",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "border": { "leftish": true } }] }),
                "is not a side",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "border": {} }] }),
                "names no side",
            ),
            // A size, a width and a height outside the shared bounds, and a range
            // holding more cells than one `format_cells` may name.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "size": 500 }] }),
                "size must be",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "column": "B",
                                    "width_chars": 300 }] }),
                "width_chars must be",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "row": 3,
                                    "height_points": 500 }] }),
                "height_points must be",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1:A999999",
                                    "bold": true }] }),
                "more than one edit formats",
            ),
            // A colour and a fill that are not six hex digits.
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "color": "red" }] }),
                "color must be",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "range": "A1",
                                    "fill": "red" }] }),
                "fill must be",
            ),
            // A slide number below one.
            (
                json!({ "action": "pptx_edit", "path": "deck.pptx",
                        "edits": [{ "op": "delete_slide", "slide": 0 }] }),
                "slide must be a whole number",
            ),
            (
                json!({ "action": "docx_edit", "path": "doc.docx", "edits": too_many }),
                "more than one call takes",
            ),
        ];
        for (args, expected) in calls {
            let err = confined_document()
                .execute(&ws, args)
                .await
                .expect_err("a call the tool cannot carry out");
            assert!(
                err.to_string().contains(expected),
                "expected the refusal to name {expected}: {err}"
            );
        }
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// A `null` for an optional edit field is the boundary's "absent", and the
    /// request spells it that way: the kit tests only `!== undefined`, so a null
    /// left in would reach it as a value — a `format_text`'s `bold: null` would
    /// strip a run's own bold, and a text field's null would be read as text. A
    /// null inside an object-valued field is dropped at that level too.
    #[test]
    fn edit_nulls_are_dropped_before_the_request_is_forwarded() {
        use crate::ooxml::Family;
        let docx = validate_edits(
            Family::Docx,
            &json!({ "edits": [
                { "op": "format_text", "find": "x", "bold": null, "size": 24 },
                { "op": "insert_text", "find": "x", "insert": "y", "position": null },
                { "op": "add_paragraph", "text": "z", "after": null },
            ]}),
        )
        .expect("valid docx edits");
        assert_eq!(
            docx,
            json!([
                { "op": "format_text", "find": "x", "size": 24 },
                { "op": "insert_text", "find": "x", "insert": "y" },
                { "op": "add_paragraph", "text": "z" },
            ])
        );

        // The set_cell object's normalized `value` survives beside the dropped
        // `number_format`, and the nested null inside the `format_cells`
        // `border` object is dropped at that level too.
        let xlsx = validate_edits(
            Family::Xlsx,
            &json!({ "edits": [
                { "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1,
                  "number_format": null },
                { "op": "format_cells", "sheet": "S", "range": "A1",
                  "border": { "top": null, "left": true } },
            ]}),
        )
        .expect("valid xlsx edits");
        assert_eq!(
            xlsx,
            json!([
                { "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1 },
                { "op": "format_cells", "sheet": "S", "range": "A1",
                  "border": { "left": true } },
            ])
        );

        let pptx = validate_edits(
            Family::Pptx,
            &json!({ "edits": [
                { "op": "add_paragraph", "slide": 1, "text": "z", "after": null,
                  "level": null },
                { "op": "add_slide", "after": null, "title": null, "bullets": null },
            ]}),
        )
        .expect("valid pptx edits");
        assert_eq!(
            pptx,
            json!([
                { "op": "add_paragraph", "slide": 1, "text": "z" },
                { "op": "add_slide" },
            ])
        );
    }

    /// The values the schema now declares are the values the boundary takes: a
    /// `list` and a `fill` of `false`, and a `set_cell` `value` that is a number
    /// or the formula object — each proved here through the validators, and each
    /// proved to be what the action's own schema entry declares.
    #[test]
    fn the_schema_declares_the_shapes_the_boundary_accepts() {
        use crate::ooxml::Family;

        let docx = validate_edits(
            Family::Docx,
            &json!({ "edits": [{ "op": "format_paragraph", "find": "x", "list": false }] }),
        )
        .expect("list: false takes the paragraph's list away");
        assert_eq!(
            docx,
            json!([{ "op": "format_paragraph", "find": "x", "list": false }])
        );

        let xlsx = validate_edits(
            Family::Xlsx,
            &json!({ "edits": [
                { "op": "format_cells", "sheet": "S", "range": "A1", "fill": false },
                { "op": "set_cell", "sheet": "S", "cell": "A1", "value": 42 },
                { "op": "set_cell", "sheet": "S", "cell": "A2",
                  "value": { "formula": "SUM(A1:A1)" } },
            ]}),
        )
        .expect("a false fill, a number and a formula object are all declared shapes");
        assert_eq!(
            xlsx,
            json!([
                { "op": "format_cells", "sheet": "S", "range": "A1", "fill": false },
                { "op": "set_cell", "sheet": "S", "cell": "A1", "value": 42 },
                { "op": "set_cell", "sheet": "S", "cell": "A2",
                  "value": { "formula": "SUM(A1:A1)" } },
            ])
        );

        // The same values the boundary just took are the ones the schema states:
        // the list's enum holds `false` beside the words, the fill is a colour or
        // `false`, and a cell's `value` is the shared scalar kinds or the formula
        // object the boundary took above.
        let fields = |action: &str| {
            action_schema(action)["properties"][action]["properties"]["edits"]["items"]
                ["properties"]
                .clone()
        };
        let docx_fields = fields("docx_edit");
        assert_eq!(docx_fields["list"]["type"], json!(["string", "boolean"]));
        assert!(
            docx_fields["list"]["enum"]
                .as_array()
                .is_some_and(|values| values.contains(&json!(false))),
            "the list enum must offer false beside the kinds: {}",
            docx_fields["list"]
        );
        let xlsx_fields = fields("xlsx_edit");
        assert_eq!(xlsx_fields["fill"]["anyOf"][0]["type"], json!("string"));
        assert_eq!(xlsx_fields["fill"]["anyOf"][1]["enum"], json!([false]));
        assert_eq!(
            xlsx_fields["value"]["anyOf"][0]["type"],
            json!(RULES.scalar_kinds)
        );
        assert_eq!(
            xlsx_fields["value"]["anyOf"][1]["required"],
            json!(["formula"])
        );
    }

    /// A copy of an edited input is named with the input's own extension when the
    /// name is the family the bytes are, and with the family's own when the name
    /// says nothing about them — the copy of a dotless or foreign-named package
    /// must not be handed an extension it does not have.
    #[test]
    fn copy_extension_follows_the_content_when_the_name_does_not() {
        use crate::ooxml::Family;
        assert_eq!(
            copy_extension(Path::new("report.docm"), Family::Docx),
            "docm"
        );
        assert_eq!(
            copy_extension(Path::new("report.docx"), Family::Docx),
            "docx"
        );
        assert_eq!(copy_extension(Path::new("report"), Family::Docx), "docx");
        assert_eq!(
            copy_extension(Path::new("report.bin"), Family::Docx),
            "docx"
        );
        assert_eq!(copy_extension(Path::new("sheet.xls"), Family::Xlsx), "xlsx");
        assert_eq!(copy_extension(Path::new("deck.ppt"), Family::Pptx), "pptx");
    }

    /// A pptx edit that names a key its op does not take is refused at the
    /// boundary: the kit reads only the fields the op knows, so a mistyped or
    /// extra parameter would be dropped in silence and the model would believe
    /// it asked for something it did not.
    #[test]
    fn pptx_edit_refuses_a_key_its_op_does_not_take() {
        let cases = [
            (
                json!([{ "op": "delete_slide", "slide": 1, "title": "x" }]),
                "edits[0].title is not a field of pptx \"delete_slide\"",
            ),
            (
                json!([{ "op": "format_text", "slide": 1, "find": "a", "shadow": true }]),
                "edits[0].shadow is not a field of pptx \"format_text\"",
            ),
        ];
        for (edits, expected) in cases {
            let err = validate_edits(crate::ooxml::Family::Pptx, &json!({ "edits": edits }))
                .expect_err("a key the op does not take must be refused");
            let message = err.to_string();
            assert!(
                message.contains("usage:") && message.contains(expected),
                "expected {expected:?}, got: {message}"
            );
        }
    }

    /// A table added to a slide takes the fields its own op states and nothing
    /// else, and it must hold something to lay out — the one table rule both
    /// producers read.
    #[test]
    fn pptx_add_table_refuses_a_key_and_a_table_its_op_does_not_take() {
        for (edits, expected) in [
            (
                json!([{ "op": "add_table", "slide": 1, "image": "x.png" }]),
                "edits[0].image is not a field of pptx \"add_table\"",
            ),
            (
                json!([{ "op": "add_table", "slide": 1, "headers": [], "rows": [] }]),
                "states a table with no rows and no headers",
            ),
        ] {
            let err = validate_edits(crate::ooxml::Family::Pptx, &json!({ "edits": edits }))
                .expect_err("a table the op cannot write must be refused");
            let message = err.to_string();
            assert!(
                message.contains("usage:") && message.contains(expected),
                "expected {expected:?}, got: {message}"
            );
        }
    }

    /// A slide size is a presentation's own: a document or a PDF naming one, and a
    /// size the product does not recognise, are refused before anything is
    /// written.
    #[tokio::test]
    async fn create_refuses_a_slide_size_another_format_cannot_have() {
        let (_dir, ws) = workspace();
        for (args, expected) in [
            (
                json!({ "action": "create", "format": "docx", "size": "widescreen",
                        "content": [{ "type": "paragraph", "text": "x" }] }),
                "only format \"pptx\" has slides",
            ),
            (
                json!({ "action": "create", "format": "pptx", "size": "a4",
                        "content": [{ "type": "paragraph", "text": "x" }] }),
                "unknown slide size \"a4\"",
            ),
        ] {
            let err = confined_document()
                .execute(&ws, args)
                .await
                .expect_err("a size no create takes");
            assert!(
                err.to_string().contains(expected),
                "expected {expected:?}, got: {err}"
            );
        }
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// A key whose value is `null` is not stated — the boundary reads a `null` as
    /// absent everywhere (see `edit_nulls_are_dropped_before_the_request_is_forwarded`)
    /// — so a `format_cells` naming `row: null` beside its `range` is the range edit
    /// it looks like: not a second target and not a key the range's op does not take.
    /// A key that IS stated still has to be one the op takes.
    #[test]
    fn xlsx_edit_reads_a_null_key_as_absent_when_checking_its_op() {
        let edits = validate_edits(
            crate::ooxml::Family::Xlsx,
            &json!({ "edits": [
                { "op": "format_cells", "sheet": "S", "range": "A1", "row": null,
                  "bold": true },
            ]}),
        )
        .expect("a null key must not be read as a key the op does not take");
        assert_eq!(
            edits,
            json!([{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }]),
            "the null key was not dropped"
        );

        let err = validate_edits(
            crate::ooxml::Family::Xlsx,
            &json!({ "edits": [
                { "op": "format_cells", "sheet": "S", "range": "A1", "bold": true,
                  "weight": 2 },
            ]}),
        )
        .expect_err("a stated key the op does not take must still be refused");
        assert!(
            err.to_string()
                .contains("edits[0].weight is not a field of xlsx \"format_cells\""),
            "got: {err}"
        );
    }

    /// A content block or a sheet naming a key its own kind does not take is
    /// refused at the boundary: the schema declares one item object for every
    /// kind, so only the boundary knows which key belongs to which, and a
    /// mistyped key would otherwise be dropped in silence.
    #[tokio::test]
    async fn a_block_refuses_a_key_its_kind_does_not_take() {
        let (_dir, ws) = workspace();
        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "paragraph", "text": "x", "level": 1 }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect_err("a paragraph block does not take a level");
        assert!(
            err.to_string().contains("usage:")
                && err
                    .to_string()
                    .contains("content[0].level is not a field of the paragraph block"),
            "got: {err}"
        );

        let err = resolve_sheets(&json!({ "sheets": [{ "rows": [["x"]], "columns": 3 }] }))
            .expect_err("a sheet does not take columns");
        assert!(
            err.to_string().contains("usage:")
                && err
                    .to_string()
                    .contains("sheets[0].columns is not a field of a sheet"),
            "got: {err}"
        );

        // The accepted kinds still pass: a paragraph naming only its own fields
        // and a named sheet with rows are the shapes their kinds declare.
        resolve_content(
            &ws,
            &json!({ "content": [{ "type": "paragraph", "text": "ok" }] }),
            "docx",
            PathAccess::Workspace,
        )
        .await
        .expect("a paragraph with only its own fields is a valid block");
        resolve_sheets(&json!({ "sheets": [{ "name": "S", "rows": [["ok"]] }] }))
            .expect("a named sheet with rows is a valid sheet");
    }

    /// Every shape a pptx `format_text` could be given that the kit cannot write
    /// is refused at the boundary: no property at all, a size outside the run's
    /// own bound, a colour that is not hex digits, an alignment the writer has
    /// not, and a toggle that is not a boolean.
    #[test]
    fn pptx_edit_format_text_refuses_every_unwritable_property() {
        let cases = [
            (
                json!({ "op": "format_text", "slide": 1, "find": "a" }),
                "needs at least one of bold",
            ),
            (
                json!({ "op": "format_text", "slide": 1, "find": "a", "size": 1e300 }),
                "size must be",
            ),
            (
                json!({ "op": "format_text", "slide": 1, "find": "a", "size": 0.5 }),
                "size must be",
            ),
            (
                json!({ "op": "format_text", "slide": 1, "find": "a", "color": "zzz" }),
                "color must be",
            ),
            (
                json!({ "op": "format_text", "slide": 1, "find": "a", "align": "middle" }),
                "align must be",
            ),
            (
                json!({ "op": "format_text", "slide": 1, "find": "a", "bold": "yes" }),
                "must be a boolean",
            ),
        ];
        for (edit, expected) in cases {
            let err = validate_edits(crate::ooxml::Family::Pptx, &json!({ "edits": [edit] }))
                .expect_err("an unwritable format_text must be refused");
            assert!(
                err.to_string().contains(expected),
                "expected {expected:?}, got: {err}"
            );
        }
    }

    /// The naming helpers drop a directory part, neutralize marker punctuation,
    /// ignore a trailing extension the tool appends itself, and fall back when
    /// nothing usable is left.
    #[test]
    fn naming_strips_extension_directory_and_markers() {
        assert_eq!(sanitize_base(Some("report.docx"), "fallback"), "report");
        assert_eq!(sanitize_base(Some("dir/sub/report"), "fallback"), "report");
        assert_eq!(sanitize_base(Some("re[po]rt"), "fallback"), "re_po_rt");
        assert_eq!(sanitize_base(Some("   "), "fallback"), "fallback");
        assert_eq!(sanitize_base(Some("..."), "fallback"), "fallback");
    }

    /// The page cap is on the total a request names, not on one selection's own
    /// size: two selections that add up past the limit must be refused before the
    /// kit ever sees them.
    #[test]
    fn split_groups_refuses_a_request_over_the_page_cap() {
        let err = split_groups(&json!({ "pages": ["1-6000", "6001-12000"] }))
            .expect_err("a request over the cap must be refused");
        assert!(
            err.to_string().contains("too many pages"),
            "unexpected error: {err}"
        );
    }

    /// The rules both this tool and the kit enforce are one committed file: a
    /// malformed one must fail loudly when it is read, and an emptied one must
    /// not silently widen every bound it states.
    #[test]
    fn the_shared_rules_parse_and_are_sane() {
        let rules = &*RULES;
        assert!(
            !rules.heading_levels.is_empty(),
            "no heading levels — every heading would be refused"
        );
        // The docx `HeadingLevel` list and the PDF size table have three
        // entries each and are indexed by `level - 1`; the PPTX writer branches
        // on the first level instead. A fourth level in the shared list would
        // pass the guard and then render an undefined style (docx/pdf) or a
        // wrong size (pptx) rather than be refused.
        assert!(
            rules
                .heading_levels
                .iter()
                .all(|level| level.fract() == 0.0 && (1.0..=3.0).contains(level)),
            "a heading level the writers cannot style"
        );
        assert!(!rules.scalar_kinds.is_empty(), "no scalar kinds");
        assert!(!rules.image_extensions.is_empty(), "no image extensions");
        // Each of these is a closed vocabulary an edit is refused against, so an
        // empty one would refuse every value the model is told to pass.
        assert!(!rules.cell_alignments.is_empty(), "no cell alignments");
        assert!(!rules.cell_verticals.is_empty(), "no cell verticals");
        assert!(!rules.cell_border_sides.is_empty(), "no cell border sides");
        assert!(
            !rules.paragraph_alignments.is_empty(),
            "no paragraph alignments"
        );
        assert!(!rules.paragraph_lists.is_empty(), "no paragraph lists");
        assert!(
            rules.dimension_points.min < rules.dimension_points.max,
            "an empty dimension range"
        );
        assert!(
            rules.slide_position_fraction.min < rules.slide_position_fraction.max,
            "an empty slide-position range"
        );
        assert!(
            rules.slide_size_fraction.min < rules.slide_size_fraction.max,
            "an empty slide-size range"
        );
        assert!(
            rules.text_size_points.min < rules.text_size_points.max,
            "an empty text-size range"
        );
        assert!(
            rules.paragraph_indent_points.min < rules.paragraph_indent_points.max,
            "an empty paragraph-indent range"
        );
        assert!(
            rules.paragraph_spacing_points.min < rules.paragraph_spacing_points.max,
            "an empty paragraph-spacing range"
        );
        assert!(
            rules.paragraph_line_points.min < rules.paragraph_line_points.max,
            "an empty paragraph-line range"
        );
        assert!(
            rules.slide_text_size_points.min < rules.slide_text_size_points.max,
            "an empty slide-text-size range"
        );
        assert!(
            rules.cell_font_points.min < rules.cell_font_points.max,
            "an empty cell-font-size range"
        );
        assert!(
            rules.column_width_chars.min < rules.column_width_chars.max,
            "an empty column-width range"
        );
        assert!(
            rules.row_height_points.min < rules.row_height_points.max,
            "an empty row-height range"
        );
        assert!(
            rules.color_digits > 0 && rules.color_digits.is_multiple_of(3),
            "a colour whose digits do not split into three channels"
        );
        assert!(
            rules.degrees_step > 0 && rules.degrees_step < 360 && 360 % rules.degrees_step == 0,
            "a step that leaves no rotation below a full turn"
        );
        assert!(
            rules.sheet_name_max > 0,
            "a sheet-name limit that refuses every name"
        );
        assert!(
            !rules.sheet_name_forbidden.is_empty(),
            "an empty forbidden set admits every character"
        );
        // Each cap is a bound the tool refuses past, so a zero would refuse every
        // value a caller could pass.
        for (name, cap) in [
            ("edit_text_max", rules.edit_text_max),
            ("edits_max", rules.edits_max),
            ("bullets_max", rules.bullets_max),
            ("paragraph_level_max", rules.paragraph_level_max as usize),
            ("sheet_row_max", rules.sheet_row_max as usize),
            ("sheet_column_max", rules.sheet_column_max as usize),
            ("number_format_max", rules.number_format_max),
            ("font_name_max", rules.font_name_max),
            ("format_cells_max", rules.format_cells_max),
            ("paragraph_style_max", rules.paragraph_style_max),
            ("table_columns_max", rules.table_columns_max),
        ] {
            assert!(cap > 0, "{name} refuses every value");
        }
    }

    /// Whether `text` states `value` as a whole number rather than as a slice of
    /// a longer number: a bare `contains` would let the paragraph-level cap `8`
    /// be satisfied by the `1048576` of the row cap, so a placeholder that moved
    /// or went stale would still pass.
    fn states_whole_number(text: &str, value: &str) -> bool {
        // An empty value is found everywhere, so it would pass this test vacuously.
        if value.is_empty() {
            return false;
        }
        let mut from = 0;
        while let Some(at) = text[from..].find(value) {
            let start = from + at;
            let end = start + value.len();
            let digit_before = text[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_digit());
            let digit_after = text[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit());
            if !digit_before && !digit_after {
                return true;
            }
            from = start + 1;
        }
        false
    }

    /// Every string the schema itself carries — its descriptions and enum words,
    /// with no JSON escaping — so a rule stated in the parameter interface is
    /// searched the same way the rendered description is.
    fn schema_text(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            Value::Array(values) => values.iter().map(schema_text).collect::<Vec<_>>().join(" "),
            Value::Object(object) => object
                .values()
                .map(schema_text)
                .collect::<Vec<_>>()
                .join(" "),
            other => other.to_string(),
        }
    }

    /// Every rule the tool enforces is stated where the model reads it: the asset
    /// carries the placeholders and [`DocumentTool::description`] renders each from
    /// `assets/docgen/rules.json`, while the bounds that live on a single argument
    /// are stated in the parameter schema — so a rule that moved cannot leave a
    /// stale number in what the model is told, and a placeholder left unreplaced
    /// would reach the model as its own literal spelling.
    #[test]
    fn the_description_states_every_rule_it_names() {
        let description = confined_document().description();
        assert!(
            !description.contains("{{"),
            "the description left a placeholder unreplaced:\n{description}"
        );
        let schema = schema_text(&confined_document().parameters_schema());
        for value in [
            listed(&RULES.heading_levels),
            RULES.dimension_points.bounds(),
            RULES.slide_position_fraction.bounds(),
            RULES.slide_size_fraction.bounds(),
            RULES.text_size_points.bounds(),
            RULES.paragraph_indent_points.bounds(),
            RULES.paragraph_spacing_points.bounds(),
            RULES.paragraph_line_points.bounds(),
            RULES.slide_text_size_points.bounds(),
            RULES.cell_font_points.bounds(),
            RULES.column_width_chars.bounds(),
            RULES.row_height_points.bounds(),
            RULES.color_digits.to_string(),
            RULES.degrees_step.to_string(),
            RULES.sheet_name_max.to_string(),
            format!("{:?}", RULES.sheet_name_forbidden),
            RULES.edit_text_max.to_string(),
            RULES.edits_max.to_string(),
            RULES.bullets_max.to_string(),
            RULES.paragraph_level_max.to_string(),
            RULES.paragraph_style_max.to_string(),
            RULES.sheet_row_max.to_string(),
            RULES.sheet_column_max.to_string(),
            RULES.number_format_max.to_string(),
            RULES.font_name_max.to_string(),
            RULES.format_cells_max.to_string(),
            RULES.table_columns_max.to_string(),
            RULES.max_pages_per_request.to_string(),
            megabytes(crate::util::FILE_MAX_BYTES).to_string(),
        ] {
            assert!(
                states_whole_number(&description, &value) || states_whole_number(&schema, &value),
                "neither the description nor the schema states the rule it names — {value:?} is \
                 missing as a whole number from:\n{description}\n{schema}"
            );
        }
        // The default deck's size is text the model reads rather than a number
        // inside a sentence, so the whole-number guard above would not cover it.
        let default_deck = RULES.pptx_default_size.inches();
        assert!(
            schema.contains(&default_deck),
            "the schema does not state the default deck's size {default_deck:?}:\n{schema}"
        );
        // A slide mark is not a number, so the whole-number guard above would
        // assert nothing for it: what the description owes the model is to name
        // the mark the reader prints — every one of them, for either family, from
        // the shared lists themselves. A label's `{n}` is the reader's own
        // placeholder, so the description states the label with `<n>` where a real
        // number goes.
        //
        // The enumerations are the one thing the schema states word by word
        // rather than as a rendered phrase, so each accepted word is checked on
        // its own.
        for word in pair_words(&RULES.paragraph_alignments)
            .into_iter()
            .chain(pair_words(&RULES.paragraph_lists))
            .chain(pair_words(&RULES.slide_alignments))
            .chain(RULES.cell_alignments.iter().map(String::as_str))
            .chain(RULES.cell_verticals.iter().map(String::as_str))
            .chain(RULES.cell_border_sides.iter().map(String::as_str))
            .chain(RULES.pptx_sizes.iter().map(|size| size.name.as_str()))
        {
            assert!(
                description.contains(word) || schema.contains(word),
                "neither the description nor the schema names the accepted word {word:?}"
            );
        }
        let marks = crate::docgen::ppt_marks();
        let labels = crate::docgen::ppt_slide_labels();
        let slide_label = crate::reader_output::slide_label(&labels.slide, "<n>");
        let notes_label = crate::reader_output::slide_label(&labels.notes, "<n>");
        let comments_label = crate::reader_output::slide_label(&labels.comments, "<n>");
        for mark in [
            &marks.title,
            &marks.hidden_slide,
            &marks.diagram_text,
            &marks.diagram_text_lost,
            &marks.no_text,
            &slide_label,
            &notes_label,
            &comments_label,
        ] {
            assert!(
                description.contains(mark.as_str()),
                "the description does not name the slide mark {mark:?}"
            );
        }
        for mark in crate::docgen::docx_marks() {
            assert!(
                description.contains(mark.example.as_str()),
                "the description does not name the Word mark {:?}",
                mark.example
            );
        }
    }

    /// What this tool's model-read contract measured: the rendered description for
    /// the path-access variant this test builds, and the description and the
    /// serialized schema together. Adding to the contract re-records both here, so
    /// growth is a decision made in this test rather than a drift.
    const BASELINE_DESCRIPTION: usize = 9_532;
    const BASELINE_TOTAL: usize = 37_402;

    /// The whole per-action contract is one compact value: every action has an
    /// entry, each entry declares exactly the arguments that action takes (and the
    /// ones it requires), and the description and schema together stay within
    /// their measured baseline.
    #[test]
    fn the_schema_is_compact_and_declares_every_action() {
        let tool = confined_document();
        let schema = tool.parameters_schema();
        let entries = schema["properties"]["action"]["oneOf"]
            .as_array()
            .expect("the action list is a oneOf");
        assert_eq!(entries.len(), ACTIONS.len(), "one entry per action");
        for (entry, action) in entries.iter().zip(ACTIONS) {
            assert_eq!(
                entry["properties"][action]["type"], "object",
                "{action} must be one entry with its own arguments"
            );
            assert_eq!(
                entry["required"],
                json!([action]),
                "{action}'s entry must require its own name"
            );
            assert_eq!(
                entry["additionalProperties"], false,
                "an entry takes only the actions it names"
            );
        }
        let description = tool.description().len();
        assert!(
            description <= BASELINE_DESCRIPTION,
            "the rendered description is {description} bytes, more than the measured \
             {BASELINE_DESCRIPTION}"
        );
        let size = serde_json::to_string(&schema)
            .expect("the schema serializes")
            .len();
        assert!(
            description + size <= BASELINE_TOTAL,
            "the description ({description}) and serialized schema ({size}) together are {} bytes, \
             more than the measured {BASELINE_TOTAL}",
            description + size
        );
    }

    /// The tool is offered exactly while the managed runtime is present: the
    /// role's tool list is what the model chooses from, so a tool whose runtime
    /// is missing must not be in it, and a host that has the runtime must not
    /// lose it. Written against the host it is given — either state is a valid
    /// one — so it runs everywhere rather than being skipped.
    #[test]
    fn the_tool_is_offered_exactly_with_the_managed_runtime() {
        let (_dir, ws) = workspace();
        let (tools, _) = crate::agent::role_tools_and_specs(
            crate::Role::Assistant,
            &ws,
            false,
            std::sync::Arc::default(),
            None,
        );
        assert_eq!(
            tools.iter().any(|tool| tool.name() == "document"),
            crate::tools::bun::bun_binary_path().is_some(),
            "the tool must be offered exactly while the managed bun runtime is present"
        );
    }

    /// The xlsx edit vocabulary takes the shapes the sheet operations state, and a
    /// merge or a name a workbook refuses is refused here, before a runtime is
    /// spawned; a `null` side is absent, the way it is for every edit.
    #[test]
    fn the_xlsx_edit_vocabulary_takes_the_new_sheet_shapes() {
        use crate::ooxml::Family;
        let edits = validate_edits(
            Family::Xlsx,
            &json!({ "edits": [
                { "op": "merge_cells", "sheet": "S", "range": "B2:C3" },
                { "op": "unmerge_cells", "sheet": "S", "range": "C3:B2" },
                { "op": "add_sheet", "name": "New" },
                { "op": "rename_sheet", "sheet": "S", "name": "Renamed" },
                { "op": "delete_sheet", "sheet": "S" },
                { "op": "move_sheet", "sheet": "S", "to": 2 },
                { "op": "add_image", "sheet": "S", "cell": "D4", "image": "a.png",
                  "width": null, "height": 40 },
            ] }),
        )
        .expect("every op the vocabulary names");
        assert_eq!(
            edits[6],
            json!({ "op": "add_image", "sheet": "S", "cell": "D4", "image": "a.png",
                    "height": 40 }),
            "a null side is absent"
        );
        for (edit, expected) in [
            (
                json!({ "op": "merge_cells", "sheet": "S", "range": "A1" }),
                "covers one cell",
            ),
            (
                json!({ "op": "add_sheet", "name": "a:b" }),
                "must not contain",
            ),
            (
                json!({ "op": "move_sheet", "sheet": "S", "to": 0 }),
                "at least 1",
            ),
            (
                json!({ "op": "add_image", "sheet": "S", "cell": "nope", "image": "a.png" }),
                "A1 address",
            ),
        ] {
            let err = validate_edits(Family::Xlsx, &json!({ "edits": [edit] }))
                .expect_err("a shape the writer cannot render");
            assert!(err.to_string().contains(expected), "{expected}: {err}");
        }
    }
}
