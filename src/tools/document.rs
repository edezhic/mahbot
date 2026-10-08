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
//! Every path the model supplies — a template, a PDF to edit, an image to embed —
//! is resolved with [`crate::tools::path::resolve_read_target`] in its strict
//! workspace-only form, for owner and guest alike. The OUTPUT path is never the
//! model's: the tool picks a name inside `<workspace>/generated/` and reserves it
//! with a create-new write before the kit runs.
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
//! new file under `<workspace>/generated/`, and every part an edit does not name
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
//! `file_name` is sanitized to a single path component and made unique against
//! `generated/`, so a model-supplied name can neither escape the directory nor
//! silently overwrite an earlier file. The tool appends the extension it
//! produces — the action's own, or the sample's or input's on a fill or an edit
//! — after dropping a trailing OOXML or PDF extension (any of `.docx`, `.docm`,
//! `.xlsx`, `.xlsm`, `.pptx`, `.pptm`, `.pdf`), so a name that already carries
//! one is not doubled.

use crate::docgen;
use crate::{Tool, Workspace};
use anyhow::{Context as _, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Subdirectory of the workspace the tool writes into, and the only one.
const GENERATED_DIR: &str = "generated";

/// Longest base name the tool keeps from `file_name`; the extension and the
/// uniqueness suffix are added on top, and a cap keeps the result well inside a
/// filesystem's component limit.
const MAX_BASE_CHARS: usize = 80;

/// Upper bound on pages ONE REQUEST may name, so a huge range (or several ranges
/// that add up) cannot make the tool allocate before the kit ever sees it.
const MAX_PAGES_PER_REQUEST: usize = 10_000;

/// Upper bound on the files ONE `pdf_split` call may produce. The range list is
/// the model's, so without this a call could turn a document into thousands of
/// attachments; the page cap above bounds the pages, not the files they are cut
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
/// `enum` and the hint for an unknown action both come from here; only the
/// dispatch `match` spells the names again.
const ACTIONS: [&str; 11] = [
    "create",
    "fill_template",
    "docx_edit",
    "xlsx_edit",
    "pptx_edit",
    "pdf_merge",
    "pdf_split",
    "pdf_rotate",
    "pdf_text",
    "pdf_image",
    "pdf_form_fill",
];

/// The ops each family's edit action accepts, in the order the refusal names
/// them. The kit owns the same vocabularies; these lists are what the boundary
/// check answers an unknown `op` with, before a runtime is spawned.
const DOCX_EDIT_OPS: [&str; 7] = [
    "replace_text",
    "insert_text",
    "remove_text",
    "format_text",
    "format_paragraph",
    "add_paragraph",
    "remove_paragraph",
];

const XLSX_EDIT_OPS: [&str; 7] = [
    "set_cell",
    "clear_cell",
    "format_cells",
    "insert_row",
    "delete_row",
    "insert_column",
    "delete_column",
];

const PPTX_EDIT_OPS: [&str; 13] = [
    "replace_text",
    "remove_text",
    "format_text",
    "add_paragraph",
    "remove_paragraph",
    "add_image",
    "add_slide",
    "delete_slide",
    "move_slide",
    "duplicate_slide",
    "replace_notes",
    "remove_notes",
    "add_notes",
];

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
    image_side_px: Span,
    /// A pptx `add_image` placement: the fraction of the slide's width or
    /// height the image is placed at.
    slide_position_fraction: Span,
    /// A pptx `add_image` size: the fraction of the slide's width or height the
    /// image is scaled to, where zero is not a size.
    slide_size_fraction: Span,
    pdf_size_points: Span,
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
    pdf_point_abs_max: f64,
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
pub(crate) struct DocumentTool;

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
        Some("[FILE:")
    }

    /// Every produced file reaches the user only through its marker, and one
    /// reply can carry many (`pdf_split` lists one per range): a truncated
    /// marker list would deliver part of what the call produced.
    fn preserve_full_output(&self) -> bool {
        true
    }

    /// The prompt asset with the shared rules rendered in: the tool's prompt
    /// states the rules it enforces, and `assets/docgen/rules.json` is their
    /// one owner, so the two cannot drift — an asset bound that the rules moved
    /// past would otherwise be told to the model as a stale number.
    fn description(&self) -> String {
        let degrees_step = RULES.degrees_step.to_string();
        let heading_levels = listed(&RULES.heading_levels);
        let image_side_px = RULES.image_side_px.bounds();
        let slide_position_fraction = RULES.slide_position_fraction.bounds();
        let slide_size_fraction = RULES.slide_size_fraction.bounds();
        let pdf_size_points = RULES.pdf_size_points.bounds();
        let text_size_points = RULES.text_size_points.bounds();
        let paragraph_indent_points = RULES.paragraph_indent_points.bounds();
        let paragraph_spacing_points = RULES.paragraph_spacing_points.bounds();
        let paragraph_line_points = RULES.paragraph_line_points.bounds();
        let slide_text_size_points = RULES.slide_text_size_points.bounds();
        let cell_font_points = RULES.cell_font_points.bounds();
        let column_width_chars = RULES.column_width_chars.bounds();
        let row_height_points = RULES.row_height_points.bounds();
        // The alignments, colours and sets the prompt spells in a JSON edit:
        // quoted and pipe-separated, in the order the rules state them.
        let slide_alignments = quoted_words(&pair_words(&RULES.slide_alignments));
        let paragraph_alignments = quoted_words(&pair_words(&RULES.paragraph_alignments));
        let paragraph_lists = quoted_words(&pair_words(&RULES.paragraph_lists));
        let cell_alignments = quoted_words(&RULES.cell_alignments);
        let cell_verticals = quoted_words(&RULES.cell_verticals);
        let cell_border_sides = quoted_words(&RULES.cell_border_sides);
        let pdf_point_abs_max = RULES.pdf_point_abs_max.to_string();
        let color_digits = RULES.color_digits.to_string();
        let merge_inputs = MAX_MERGE_INPUTS.to_string();
        let merge_mb = megabytes(MAX_MERGE_BYTES).to_string();
        let split_parts = MAX_SPLIT_PARTS.to_string();
        let input_mb = megabytes(crate::util::FILE_MAX_BYTES).to_string();
        let sheet_name_max = RULES.sheet_name_max.to_string();
        let edit_text_max = RULES.edit_text_max.to_string();
        let edits_max = RULES.edits_max.to_string();
        let bullets_max = RULES.bullets_max.to_string();
        let paragraph_level_max = RULES.paragraph_level_max.to_string();
        let paragraph_style_max = RULES.paragraph_style_max.to_string();
        let sheet_row_max = RULES.sheet_row_max.to_string();
        let sheet_column_max = RULES.sheet_column_max.to_string();
        let number_format_max = RULES.number_format_max.to_string();
        let font_name_max = RULES.font_name_max.to_string();
        let format_cells_max = RULES.format_cells_max.to_string();
        // The set with its backslash escaped, as a quoted string spells one: a
        // bare `\` before the closing quote reads as an escaped quote.
        let sheet_name_forbidden = format!("{:?}", RULES.sheet_name_forbidden);
        let ppt_marks = crate::docgen::ppt_marks();
        crate::prompt::substitute(
            &crate::prompt::load_prompt("tool/document.md"),
            &[
                ("{{degrees_step}}", &degrees_step),
                ("{{heading_levels}}", &heading_levels),
                ("{{image_side_px}}", &image_side_px),
                ("{{slide_position_fraction}}", &slide_position_fraction),
                ("{{slide_size_fraction}}", &slide_size_fraction),
                ("{{pdf_size_points}}", &pdf_size_points),
                ("{{text_size_points}}", &text_size_points),
                ("{{paragraph_alignments}}", &paragraph_alignments),
                ("{{paragraph_lists}}", &paragraph_lists),
                ("{{paragraph_indent_points}}", &paragraph_indent_points),
                ("{{paragraph_spacing_points}}", &paragraph_spacing_points),
                ("{{paragraph_line_points}}", &paragraph_line_points),
                ("{{paragraph_style_max}}", &paragraph_style_max),
                ("{{slide_text_size_points}}", &slide_text_size_points),
                ("{{cell_font_points}}", &cell_font_points),
                ("{{column_width_chars}}", &column_width_chars),
                ("{{row_height_points}}", &row_height_points),
                ("{{slide_alignments}}", &slide_alignments),
                ("{{cell_alignments}}", &cell_alignments),
                ("{{cell_verticals}}", &cell_verticals),
                ("{{cell_border_sides}}", &cell_border_sides),
                ("{{pdf_point_abs_max}}", &pdf_point_abs_max),
                ("{{color_digits}}", &color_digits),
                ("{{merge_inputs}}", &merge_inputs),
                ("{{merge_mb}}", &merge_mb),
                ("{{split_parts}}", &split_parts),
                ("{{input_mb}}", &input_mb),
                ("{{sheet_name_max}}", &sheet_name_max),
                ("{{sheet_name_forbidden}}", &sheet_name_forbidden),
                ("{{edit_text_max}}", &edit_text_max),
                ("{{edits_max}}", &edits_max),
                ("{{bullets_max}}", &bullets_max),
                ("{{paragraph_level_max}}", &paragraph_level_max),
                ("{{sheet_row_max}}", &sheet_row_max),
                ("{{sheet_column_max}}", &sheet_column_max),
                ("{{number_format_max}}", &number_format_max),
                ("{{font_name_max}}", &font_name_max),
                ("{{format_cells_max}}", &format_cells_max),
                ("{{ppt_title_mark}}", ppt_marks.title.as_str()),
                ("{{ppt_hidden_slide_mark}}", ppt_marks.hidden_slide.as_str()),
                ("{{ppt_diagram_text_mark}}", ppt_marks.diagram_text.as_str()),
                (
                    "{{ppt_diagram_text_lost_mark}}",
                    ppt_marks.diagram_text_lost.as_str(),
                ),
                ("{{ppt_no_text_mark}}", ppt_marks.no_text.as_str()),
            ],
        )
    }

    #[expect(clippy::too_many_lines)] // the one schema the tool advertises, argument by argument
    fn parameters_schema(&self) -> Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": ACTIONS,
                    "description": "Operation to perform."
                },
                "file_name": {
                    "type": "string",
                    "description": "Base name for the produced file. Sanitized to a single path component and made unique in the generated directory; a trailing extension the tool itself produces (.docx/.docm/.xlsx/.xlsm/.pptx/.pptm/.pdf) is dropped from your value instead of being doubled."
                },
                "format": {
                    "type": "string",
                    "enum": SUPPORTED_FORMATS,
                    "description": "create: the file format to write."
                },
                "content": {
                    "type": "array",
                    "items": { "type": "object" },
                    "description": "create (docx/pptx/pdf): content blocks. Each block is {\"type\": \"heading|paragraph|list|table|image|notes\", …} — see the tool description."
                },
                "sheets": {
                    "type": "array",
                    "items": { "type": "object" },
                    "description": "create (xlsx): sheets, each {\"name\": …, \"rows\": [[cell, …]]}. A cell is a string, a number, a boolean, or {\"formula\": \"SUM(A1:A2)\"}."
                },
                "template": {
                    "type": "string",
                    "description": "fill_template: workspace path of the user's own .docx/.pptx/.xlsx sample; {name} placeholders are replaced with values. The sample is never modified."
                },
                "values": {
                    "type": "object",
                    "description": "fill_template / pdf_form_fill: values keyed by placeholder or form-field name; each value is text, a number or a boolean. pdf_form_fill needs at least one field."
                },
                "edits": {
                    "type": "array",
                    "items": { "type": "object" },
                    "description": format!(
                        "docx_edit/xlsx_edit/pptx_edit: the edits to apply in order, at most {}; \
                         each is an object with an \"op\" and that op's fields — see the tool \
                         description. Applied to a copy; the file passed is never changed.",
                        RULES.edits_max
                    )
                },
                "files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": format!(
                        "pdf_merge: two or more workspace PDF paths, merged in order — at most {} \
                         files, totalling at most {} MB, in one call.",
                        MAX_MERGE_INPUTS,
                        megabytes(MAX_MERGE_BYTES)
                    )
                },
                "path": {
                    "type": "string",
                    "description": "PDF and edit actions: workspace path of the file."
                },
                "image": {
                    "type": "string",
                    "description": "pdf_image: workspace path of the PNG or JPEG to place on the pages."
                },
                "ranges": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": format!(
                        "pdf_split: one range per output file, e.g. [\"1-3\", \"5\"], at most {} \
                         per call.",
                        MAX_SPLIT_PARTS
                    )
                },
                "pages": {
                    "type": "string",
                    "description": "Pages to act on: \"all\" or a list like \"1-3,5\". Defaults to all pages."
                },
                "degrees": {
                    "type": "integer",
                    "description": format!(
                        "pdf_rotate: rotation to add, a multiple of {} taken modulo 360; a whole \
                         turn is refused.",
                        RULES.degrees_step
                    )
                },
                "text": {
                    "type": "string",
                    "description": "pdf_text: the text to draw."
                },
                "x": { "type": "number", "description": "pdf_text/pdf_image: X position in points." },
                "y": { "type": "number", "description": "pdf_text/pdf_image: Y position in points." },
                "size": {
                    "type": "number",
                    "description": format!(
                        "pdf_text: font size in points, {}.",
                        RULES.pdf_size_points.bounds()
                    )
                },
                "color": {
                    "type": "string",
                    "description": format!(
                        "pdf_text: text colour as {} hex digits, with an optional leading \"#\".",
                        RULES.color_digits
                    )
                },
                "stamp": { "type": "boolean", "description": "pdf_text: draw a box around the text." },
                "rotate": { "type": "number", "description": "pdf_text: rotation of the drawn text, in degrees." },
                "width": {
                    "type": "number",
                    "description": format!(
                        "pdf_image: placed width in points, {}.",
                        RULES.pdf_size_points.bounds()
                    )
                },
                "height": {
                    "type": "number",
                    "description": format!(
                        "pdf_image: placed height in points, {}.",
                        RULES.pdf_size_points.bounds()
                    )
                },
                "flatten": { "type": "boolean", "description": "pdf_form_fill: flatten the filled fields into the page content." }
            }),
            &["action"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: Value) -> Result<String> {
        let generated = ws.as_path().join(GENERATED_DIR);
        match super::get_str(&args, "action")? {
            "create" => self.create(ws, &args, &generated).await,
            "fill_template" => self.fill_template(ws, &args, &generated).await,
            "docx_edit" => {
                self.edit(ws, &args, &generated, crate::ooxml::Family::Docx)
                    .await
            }
            "xlsx_edit" => {
                self.edit(ws, &args, &generated, crate::ooxml::Family::Xlsx)
                    .await
            }
            "pptx_edit" => {
                self.edit(ws, &args, &generated, crate::ooxml::Family::Pptx)
                    .await
            }
            "pdf_merge" => self.pdf_merge(ws, &args, &generated).await,
            "pdf_split" => self.pdf_split(ws, &args, &generated).await,
            "pdf_rotate" => self.pdf_rotate(ws, &args, &generated).await,
            "pdf_text" => self.pdf_text(ws, &args, &generated).await,
            "pdf_image" => self.pdf_image(ws, &args, &generated).await,
            "pdf_form_fill" => self.pdf_form_fill(ws, &args, &generated).await,
            other => anyhow::bail!(
                "usage: unknown action \"{other}\" — hint: use {}",
                ACTIONS.join(", ")
            ),
        }
    }
}

impl DocumentTool {
    /// `create` — a fresh document from content blocks or sheets.
    async fn create(&self, ws: &Workspace, args: &Value, generated: &Path) -> Result<String> {
        let format = super::get_str(args, "format")?;
        if !SUPPORTED_FORMATS.contains(&format) {
            anyhow::bail!(
                "usage: unknown format \"{format}\" — hint: use {}",
                SUPPORTED_FORMATS.join(", ")
            );
        }
        let mut request = json!({ "op": "create", "format": format });
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
            request["content"] = resolve_content(ws, args, format).await?;
        }
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            &format!("{}_{}", default_base(format), crate::util::unix_millis()),
        );
        let outputs = reserve(generated, std::slice::from_ref(&base), format).await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "created").await
    }

    /// `fill_template` — a copy of the user's own sample with `{name}`
    /// placeholders replaced.
    async fn fill_template(
        &self,
        ws: &Workspace,
        args: &Value,
        generated: &Path,
    ) -> Result<String> {
        let (template, _) = resolve_input(ws, super::get_str(args, "template")?).await?;
        let values = super::get_object(args, "values")?;
        require_scalar_values(&values)?;
        let name = file_name_of(&template);
        let head = head_of(&template).await?;
        // The kit's `format` is the sample's FAMILY: a workbook has its own
        // cell-wise filler, while a document and a presentation both go through
        // the template library (which reads the package's file type itself).
        let Some(family) = crate::ooxml::family_of(&template) else {
            // Only the three OOXML families carry `{name}` placeholders; naming
            // the alternatives beats the package reader's own parse error.
            let hint = old_format_hint(
                &template,
                &head,
                "fill",
                "an old .doc/.xls/.ppt file is read but never filled or edited, so save it as one \
                 of the OOXML families first",
            )
            .unwrap_or_else(|| {
                "a PDF is annotated with the pdf_text, pdf_image or pdf_form_fill actions"
                    .to_string()
            });
            anyhow::bail!(
                "usage: a sample must be a .docx/.docm, .pptx/.pptm or .xlsx/.xlsm file, got {name} \
                 — hint: {hint}"
            );
        };
        // The OUTPUT keeps the sample's own extension — an `.xlsm` copy stays an
        // `.xlsm` one, macros and all — which a matched family guarantees it has.
        let extension = template
            .extension()
            .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
            .expect("a matched family means the sample is named with its extension");
        ensure_input_readable(&template, &head)?;
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            &input_base(&template, "filled"),
        );
        let outputs = reserve(generated, std::slice::from_ref(&base), &extension).await?;
        let request = json!({
            "op": "fill_template",
            "template": template.to_string_lossy(),
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
        generated: &Path,
        family: crate::ooxml::Family,
    ) -> Result<String> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?).await?;
        // The file's head is read once and answers both questions: the family its
        // name claims (its own family, never the name alone, is what this action
        // edits — a package of another family or a PDF is refused by name and an
        // old binary format with the hint that it is read but never edited) and
        // whether it is an encrypted package this tool cannot open at all.
        let head = head_of(&input).await?;
        require_edit_family(&input, &head, family)?;
        ensure_input_readable(&input, &head)?;
        let mut edits = validate_edits(family, args)?;
        // Only a presentation needs either of these: a resolved path for the
        // image an edit places, and the diagram text a refusal about the slide's
        // own text may name. What a slide's diagrams show is the reader's rule,
        // so the text is read from the input — under the conversion bound, since
        // it walks a whole deck — and the kit looks it up rather than restating
        // how a diagram is read. An edit that never looks text up on the slide (a
        // notes edit, a slide reshape) reads no diagram text at all.
        let mut diagram_text = HashMap::new();
        if family == crate::ooxml::Family::Pptx {
            resolve_edit_images(ws, &mut edits).await?;
            if edits_search_slide_text(&edits) {
                diagram_text = crate::document::read_pptx_diagram_text(&input).await;
            }
        }
        // The OUTPUT keeps the input's own extension — a `.docm` copy stays a
        // `.docm` one, macros and all — which a matched family guarantees it has.
        let extension = input
            .extension()
            .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
            .expect("a matched family means the input is named with its extension");
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            &input_base(&input, "edited"),
        );
        let outputs = reserve(generated, std::slice::from_ref(&base), &extension).await?;
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
    async fn pdf_merge(&self, ws: &Workspace, args: &Value, generated: &Path) -> Result<String> {
        let files = super::get_str_array(args, "files")?;
        if files.len() < 2 {
            anyhow::bail!(
                "usage: `files` needs at least 2 PDFs to merge, got {} — hint: pass two or \
                 more workspace paths",
                files.len()
            );
        }
        if files.len() > MAX_MERGE_INPUTS {
            anyhow::bail!(
                "usage: {} files is more than one merge takes (limit {MAX_MERGE_INPUTS}) — hint: \
                 merge them in batches",
                files.len()
            );
        }
        let mut inputs = Vec::with_capacity(files.len());
        // The kit decodes every input into memory at once, so the size of the
        // whole call is bounded, not just of one file.
        let mut total = 0u64;
        for file in &files {
            let (path, size) = resolve_input(ws, file).await?;
            total += size;
            inputs.push(path);
        }
        if total > MAX_MERGE_BYTES {
            anyhow::bail!(
                "usage: the files total {} MB, more than one merge takes (limit {} MB) — hint: \
                 merge them in batches",
                megabytes(total),
                megabytes(MAX_MERGE_BYTES)
            );
        }
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            &input_base(&inputs[0], "merged"),
        );
        let outputs = reserve(generated, std::slice::from_ref(&base), "pdf").await?;
        let request = json!({
            "op": "pdf_merge",
            "inputs": path_strings(&inputs),
            "output": outputs[0].to_string_lossy(),
        });
        self.deliver(request, &outputs, "merged the files into")
            .await
    }

    /// `pdf_split` — one file per range.
    async fn pdf_split(&self, ws: &Workspace, args: &Value, generated: &Path) -> Result<String> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?).await?;
        let groups = split_groups(args)?;
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            crate::util::name_stem(&file_name_of(&input)),
        );
        let bases: Vec<String> = (0..groups.len())
            .map(|index| format!("{base}_part{}", index + 1))
            .collect();
        let outputs = reserve(generated, &bases, "pdf").await?;
        let request = json!({
            "op": "pdf_split",
            "input": input.to_string_lossy(),
            "groups": groups,
            "outputs": path_strings(&outputs),
        });
        self.deliver(request, &outputs, "split the file into").await
    }

    /// `pdf_rotate` — add a rotation to the selected pages.
    async fn pdf_rotate(&self, ws: &Workspace, args: &Value, generated: &Path) -> Result<String> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?).await?;
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
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            &input_base(&input, "rotated"),
        );
        let outputs = reserve(generated, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "rotated").await
    }

    /// `pdf_text` — draw text (optionally stamped and rotated) on the pages.
    async fn pdf_text(&self, ws: &Workspace, args: &Value, generated: &Path) -> Result<String> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?).await?;
        let text = super::get_str(args, "text")?;
        let mut request = json!({
            "op": "pdf_text",
            "input": input.to_string_lossy(),
            "text": text,
        });
        request["pages"] = opt_pages(args)?.kit_value();
        set_number(&mut request, args, "x")?;
        set_number(&mut request, args, "y")?;
        set_positive_number(&mut request, args, "size")?;
        set_number(&mut request, args, "rotate")?;
        if let Some(color) = opt_string(args, "color")? {
            // A pdf argument is named quoted, as the neighbouring refusals do.
            require_hex_color("\"color\"", color)?;
            request["color"] = json!(color);
        }
        if let Some(stamp) = super::get_opt_bool(args, "stamp")? {
            request["stamp"] = json!(stamp);
        }
        let base = sanitize_base(opt_string(args, "file_name")?, &input_base(&input, "text"));
        let outputs = reserve(generated, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "stamped text into").await
    }

    /// `pdf_image` — place a PNG/JPEG on the pages.
    async fn pdf_image(&self, ws: &Workspace, args: &Value, generated: &Path) -> Result<String> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?).await?;
        let (image, _) = resolve_input(ws, super::get_str(args, "image")?).await?;
        ensure_image(&image)?;
        let mut request = json!({
            "op": "pdf_image",
            "input": input.to_string_lossy(),
            "image": image.to_string_lossy(),
        });
        request["pages"] = opt_pages(args)?.kit_value();
        set_number(&mut request, args, "x")?;
        set_number(&mut request, args, "y")?;
        set_positive_number(&mut request, args, "width")?;
        set_positive_number(&mut request, args, "height")?;
        let base = sanitize_base(opt_string(args, "file_name")?, &input_base(&input, "image"));
        let outputs = reserve(generated, std::slice::from_ref(&base), "pdf").await?;
        request["output"] = json!(outputs[0].to_string_lossy());
        self.deliver(request, &outputs, "inserted the image into")
            .await
    }

    /// `pdf_form_fill` — fill a PDF's AcroForm fields.
    async fn pdf_form_fill(
        &self,
        ws: &Workspace,
        args: &Value,
        generated: &Path,
    ) -> Result<String> {
        let (input, _) = resolve_input(ws, super::get_str(args, "path")?).await?;
        let values = super::get_object(args, "values")?;
        // A form filled with nothing is a copy, not a fill: reporting that as
        // "filled the form" would name a success where nothing happened.
        if values.is_empty() {
            anyhow::bail!(
                "usage: missing or empty argument \"values\" — hint: name at least one form field, \
                 e.g. {{\"name\": \"Иван\"}}"
            );
        }
        require_scalar_values(&values)?;
        // Every argument is read before anything is reserved: a call the tool
        // refuses must not leave an empty output file behind.
        let flatten = super::get_bool(args, "flatten", false)?;
        let base = sanitize_base(
            opt_string(args, "file_name")?,
            &input_base(&input, "filled"),
        );
        let outputs = reserve(generated, std::slice::from_ref(&base), "pdf").await?;
        let request = json!({
            "op": "pdf_form_fill",
            "input": input.to_string_lossy(),
            "values": values,
            "flatten": flatten,
            "output": outputs[0].to_string_lossy(),
        });
        self.deliver(request, &outputs, "filled the form into")
            .await
    }

    /// Run one prepared operation and turn its result into the reply: verify
    /// every reserved output exists, describe what was produced, and list the
    /// `[FILE:…]` markers. A failed run removes the reservations so no empty
    /// placeholder is left behind.
    #[expect(clippy::too_many_lines)] // the one reply builder: verify, describe, and every note an operation can raise
    async fn deliver(&self, request: Value, outputs: &[PathBuf], prefix: &str) -> Result<String> {
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
        let oversized: Vec<String> = outputs
            .iter()
            .zip(&sizes)
            .filter(|(_, size)| **size > crate::util::FILE_MAX_BYTES)
            .map(|(path, _)| file_name_of(path))
            .collect();
        if !oversized.is_empty() {
            text = super::with_note(
                &text,
                &format!(
                    "[over the {} MB send limit — may not be delivered: {}]",
                    megabytes(crate::util::FILE_MAX_BYTES),
                    oversized.join(", ")
                ),
            );
        }
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
        // Each note is the kit's own caveat about what it could not keep — a chart
        // or pivot keeping a cached value, a range the delete covered, a part left
        // naming the old cells — already a bracketed, user-facing sentence, so it
        // is appended verbatim.
        for note in &outcome.notes {
            text = super::with_note(&text, note);
        }
        for path in outputs {
            text.push('\n');
            text.push_str(&self.format_media_result(path));
        }
        Ok(text)
    }
}

/// Resolve a model-supplied input path inside the workspace (strict: no temp
/// files, no dependency sources) and require it to be a regular file the kit can
/// load whole. The file's length is returned alongside the path so a caller
/// never stats the same file a second time.
async fn resolve_input(ws: &Workspace, path: &str) -> Result<(PathBuf, u64)> {
    let resolved = super::path::resolve_read_target(ws.as_path(), path, true).await?;
    let meta = tokio::fs::metadata(&resolved)
        .await
        .with_context(|| format!("cannot read {}", resolved.display()))?;
    if !meta.is_file() {
        anyhow::bail!(
            "usage: {} is not a file — hint: pass a workspace path to a file",
            resolved.display()
        );
    }
    // The product's own file limit, the same one an input to a conversion or an
    // outgoing attachment meets. Without it an oversized file is read whole by
    // the kit and ends as a killed run — a caller's own file reported as a
    // mahbot-side fault.
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

/// The leading bytes of `path` — the shape every container this module asks
/// about is recognized by. A sample and an edit input are each read once and both
/// their questions (password, family) asked of the same head.
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

/// Refuse an encrypted OOXML input — a `fill_template` sample or an edit's
/// input. Such a file is a CFB container that no package reader can open; the
/// read path gives the same file the password-protected verdict, and the kit's
/// own failure would be a zip-level message that says nothing about the
/// password. Only a file whose name claims one of the OOXML families reaches
/// this — a legacy `.doc`/`.xls`/`.ppt` is a CFB container too, and the family
/// check refuses it before this point because an old format is read but never
/// filled or edited.
fn ensure_input_readable(path: &Path, head: &[u8]) -> Result<()> {
    if head != crate::document::CFB_MAGIC {
        return Ok(());
    }
    anyhow::bail!(
        "usage: {} is password-protected — hint: remove the protection, save an unprotected copy \
         and fill or edit that",
        path.display()
    )
}

/// The answer a file NAMED as an old binary format but holding an OOXML package
/// is owed. The legacy answer — the format is read but never written — would
/// describe a file it is not, and the name is what this tool addresses a package
/// by (it is what picks the family to fill or edit), so renaming it is the fix.
/// `action` is what the caller can then do with it (`edit`, `fill`).
fn misnamed_package(action: &str) -> String {
    format!(
        "the file's bytes are an OOXML package, so name it as the family it really is and {action} \
         that"
    )
}

/// The hint an input named in the old `.doc`/`.xls`/`.ppt` family is owed, or
/// `None` when its name says nothing about old formats. The container, not the
/// name alone, decides which of the two answers such a name gets: one whose bytes
/// are an OOXML package is not an old format — the read path would not read it as
/// one — and is owed the same "name it as the family it really is" hint a misnamed
/// package gets anywhere else. `action` is what the caller can then do with it
/// (`edit`, `fill`), and `old_format` the sentence an action that cannot write the
/// format says instead.
fn old_format_hint(path: &Path, head: &[u8], action: &str, old_format: &str) -> Option<String> {
    crate::legacy::family_of(path).is_some().then(|| {
        if crate::document::is_zip_container(head) {
            misnamed_package(action)
        } else {
            old_format.to_string()
        }
    })
}

/// Require an edit input to be a package of the family its action edits: a file
/// of another family (or a PDF) is refused by name, and an old binary
/// `.doc`/`.xls`/`.ppt` gets the answer the old formats are owed — the format is
/// read but never edited — instead of a file substituted for the one asked for.
fn require_edit_family(path: &Path, head: &[u8], family: crate::ooxml::Family) -> Result<()> {
    if crate::ooxml::family_of(path) == Some(family) {
        return Ok(());
    }
    let accepted = match family {
        crate::ooxml::Family::Docx => "a .docx/.docm file",
        crate::ooxml::Family::Xlsx => "a .xlsx/.xlsm file",
        crate::ooxml::Family::Pptx => "a .pptx/.pptm file",
    };
    let name = file_name_of(path);
    if let Some(hint) = old_format_hint(
        path,
        head,
        "edit",
        "an old .doc/.xls/.ppt file is read but never edited; say so, or write a new file with \
         create",
    ) {
        anyhow::bail!(
            "usage: {}_edit edits only {accepted}, got {name} — hint: {hint}",
            family.name()
        );
    }
    anyhow::bail!(
        "usage: {}_edit edits only {accepted}, got {name} — hint: pass the file's own family, or \
         use the pdf_* actions for a PDF",
        family.name()
    )
}

/// The ops one family's edit action accepts.
fn edit_ops(family: crate::ooxml::Family) -> &'static [&'static str] {
    match family {
        crate::ooxml::Family::Docx => &DOCX_EDIT_OPS,
        crate::ooxml::Family::Xlsx => &XLSX_EDIT_OPS,
        crate::ooxml::Family::Pptx => &PPTX_EDIT_OPS,
    }
}

/// Validate and normalize the `edits` argument for one family's edit action,
/// refusing every shape the kit would fail on before a runtime is spawned: the
/// ops the family accepts, the fields each op needs, and the shared bounds in
/// `assets/docgen/rules.json`. Returns the array the request carries.
fn validate_edits(family: crate::ooxml::Family, args: &Value) -> Result<Value> {
    let ops = edit_ops(family);
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
            anyhow::bail!("usage: {at} has no \"op\" — hint: use {}", ops.join(", "));
        };
        if !ops.contains(&op) {
            anyhow::bail!(
                "usage: unknown {} edit \"{op}\" — hint: use {}",
                family.name(),
                ops.join(", ")
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
    let keys =
        |allowed: &[&str]| require_edit_keys(object, crate::ooxml::Family::Docx, op, allowed, at);
    match op {
        "replace_text" => {
            keys(&["find", "replace"])?;
            edit_string_non_empty(object, "find", at)?;
            edit_string_present(object, "replace", at)?;
        }
        "insert_text" => {
            keys(&["find", "insert", "position"])?;
            edit_string_non_empty(object, "find", at)?;
            edit_string_non_empty(object, "insert", at)?;
            require_position(object, at)?;
        }
        "remove_text" | "remove_paragraph" => {
            keys(&["find"])?;
            edit_string_non_empty(object, "find", at)?;
        }
        "format_text" => {
            keys(&[
                "find",
                "bold",
                "italic",
                "underline",
                "strike",
                "size",
                "color",
            ])?;
            edit_string_non_empty(object, "find", at)?;
            require_format(object, at)?;
        }
        "format_paragraph" => {
            keys(&[
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
            ])?;
            edit_string_non_empty(object, "find", at)?;
            require_paragraph_format(object, at)?;
        }
        "add_paragraph" => {
            keys(&["text", "after"])?;
            edit_string_present(object, "text", at)?;
            require_optional_text(object, "after", at)?;
        }
        _ => unreachable!("the op was checked against the family's vocabulary"),
    }
    Ok(())
}

/// Require an `insert_text`'s optional `position` to be `after` or `before`.
fn require_position(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    match object.get("position") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(position)) if position == "after" || position == "before" => Ok(()),
        Some(Value::String(position)) => anyhow::bail!(
            "usage: {at}.position must be \"after\" or \"before\", got \"{position}\""
        ),
        Some(v) => Err(super::wrong_type(&format!("{at}.position"), "a string", v)),
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
                "color",
                "fill",
                "border",
                "align",
                "vertical",
                "wrap",
                "number_format",
            ],
            Self::Column => &["width"],
            Self::Row => &["height"],
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

/// Require a xlsx `format_cells`'s `range` to be a cell (`"B2"`) or a
/// rectangular range (`"B2:D2"`) inside the sheet's grid. The corners may be
/// named in either order — the rectangle is what they enclose — and it may hold
/// at most `format_cells_max` cells, because the kit writes it out cell by cell:
/// a whole-column spelling would write a sheet-sized body rather than format the
/// cells the caller meant.
fn require_format_cells_range(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let field = format!("{at}.range");
    let range = edit_string_non_empty(object, "range", at)?;
    let mut corners = range.split(':');
    let first = corners.next().unwrap_or_default();
    let second = corners.next();
    if corners.next().is_some() {
        anyhow::bail!(
            "usage: {field} must be a cell or a rectangular range like B2:D2, got \"{range}\""
        );
    }
    let (first_row, first_column) = check_cell_address(first, &field)?;
    let (last_row, last_column) = match second {
        Some(second) => check_cell_address(second, &field)?,
        None => (first_row, first_column),
    };
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

/// Require an optional `color` — a `format_text`'s or a xlsx `format_cells`'s
/// font colour — to be the shape the kit parses (see [`require_hex_color`]);
/// returns whether it was stated.
fn require_color(object: &serde_json::Map<String, Value>, at: &str) -> Result<bool> {
    match object.get("color") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(color)) => {
            require_hex_color(&format!("{at}.color"), color)?;
            Ok(true)
        }
        Some(v) => Err(super::wrong_type(&format!("{at}.color"), "a string", v)),
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
    match object.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(value)) if values.iter().any(|allowed| allowed.as_ref() == value) => {
            Ok(true)
        }
        Some(Value::String(value)) => anyhow::bail!(
            "usage: {at}.{key} must be {}, got \"{value}\"",
            listed(values)
        ),
        Some(value) => Err(super::wrong_type(&format!("{at}.{key}"), "a string", value)),
    }
}

/// Refuse an edit's key that its op does not take, naming the fields the op does:
/// the kit reads only those, so an extra one would be dropped in silence. A null
/// value is not stated — the boundary reads a `null` as absent everywhere (see
/// `stated`) — so a key whose value is null is not a key the op does not take.
fn require_edit_keys(
    object: &serde_json::Map<String, Value>,
    family: crate::ooxml::Family,
    op: &str,
    allowed: &[&str],
    at: &str,
) -> Result<()> {
    for key in object.keys() {
        if key != "op" && !allowed.contains(&key.as_str()) && stated(object, key) {
            anyhow::bail!(
                "usage: {at}.{key} is not a field of {} \"{op}\" — hint: it takes {}",
                family.name(),
                allowed.join(", ")
            );
        }
    }
    Ok(())
}

/// The xlsx edit vocabulary and the cell, row and column shapes its writer
/// knows: a sheet is named, a cell is an A1 address, a row and a column are the
/// 1-based units the reader shows, and `format_cells` names one of a cell range,
/// a column's width or a row's height and the properties it applies to it.
fn validate_xlsx_edit(
    edit: &Value,
    object: &serde_json::Map<String, Value>,
    op: &str,
    at: &str,
) -> Result<Value> {
    let keys =
        |allowed: &[&str]| require_edit_keys(object, crate::ooxml::Family::Xlsx, op, allowed, at);
    let sheet = edit_string_non_empty(object, "sheet", at)?;
    require_sheet_name_length(sheet, &format!("{at}.sheet"))?;
    match op {
        "set_cell" => {
            keys(&["sheet", "cell", "value", "number_format"])?;
            require_cell(object, at)?;
            require_name(object, "number_format", RULES.number_format_max, at)?;
            let mut normalized = edit.clone();
            normalized["value"] = require_cell_value(object, at)?;
            return Ok(normalized);
        }
        "clear_cell" => {
            keys(&["sheet", "cell"])?;
            require_cell(object, at)?;
        }
        "format_cells" => {
            let target = require_format_cells_target(object, at)?;
            keys(&target.keys())?;
            let named = match target {
                FormatCellsTarget::Range => {
                    require_format_cells_range(object, at)?;
                    let mut named = require_toggles(object, &["bold", "italic", "wrap"], at)?;
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
                        "width",
                        &RULES.column_width_chars,
                        "give a width in characters, as Excel shows one",
                        at,
                    )?
                }
                FormatCellsTarget::Row => {
                    require_row(object, at)?;
                    require_stated_number(
                        object,
                        "height",
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
            keys(&["sheet", "row"])?;
            require_row(object, at)?;
        }
        "insert_column" | "delete_column" => {
            keys(&["sheet", "column"])?;
            require_column(object, at)?;
        }
        _ => unreachable!("the op was checked against the family's vocabulary"),
    }
    Ok(edit.clone())
}

/// The 1-based column number A1-style letters name (`"A"` -> 1, `"XFD"` ->
/// 16384), or `None` when they are not one to three ASCII uppercase letters.
/// The letters' shape is checked here and nowhere else, so the `cell` and the
/// `column` fields cannot disagree about what a column address is, and the
/// number is the shared reader's own (`ooxml::column_index`).
fn column_number(letters: &str) -> Option<u32> {
    if !((1..=3).contains(&letters.len()) && letters.bytes().all(|byte| byte.is_ascii_uppercase()))
    {
        return None;
    }
    crate::ooxml::column_index(letters).map(|index| index + 1)
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
            crate::ooxml::column_letters(RULES.sheet_column_max - 1)
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
            crate::ooxml::column_letters(RULES.sheet_column_max - 1)
        ),
    }
}

/// The pptx edit vocabulary: a slide is its 1-based number as the reader shows
/// it, and a text op names a fragment that must be on that slide.
fn validate_pptx_edit(object: &serde_json::Map<String, Value>, op: &str, at: &str) -> Result<()> {
    let keys =
        |allowed: &[&str]| require_edit_keys(object, crate::ooxml::Family::Pptx, op, allowed, at);
    match op {
        "replace_text" | "replace_notes" => {
            keys(&["slide", "find", "replace"])?;
            require_slide(object, at)?;
            require_pptx_lookup(edit_string_non_empty(object, "find", at)?, "find", at)?;
            edit_string_present(object, "replace", at)?;
        }
        "remove_text" | "remove_paragraph" | "remove_notes" => {
            keys(&["slide", "find"])?;
            require_slide(object, at)?;
            require_pptx_lookup(edit_string_non_empty(object, "find", at)?, "find", at)?;
        }
        "format_text" => {
            keys(&[
                "slide",
                "find",
                "bold",
                "italic",
                "underline",
                "size",
                "color",
                "align",
            ])?;
            require_slide(object, at)?;
            require_pptx_lookup(edit_string_non_empty(object, "find", at)?, "find", at)?;
            require_pptx_format(object, at)?;
        }
        "add_paragraph" => {
            keys(&["slide", "text", "after", "level"])?;
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
            keys(&["slide", "path", "x", "y", "width", "height"])?;
            require_slide(object, at)?;
            edit_string_non_empty(object, "path", at)?;
            require_image_geometry(object, at)?;
        }
        "add_slide" => {
            keys(&["after", "title", "bullets"])?;
            if edit_integer(object, "after", at)? == Some(0) {
                anyhow::bail!("usage: {at}.after must be a whole number of at least 1");
            }
            require_optional_text(object, "title", at)?;
            require_bullets(object, at)?;
        }
        "delete_slide" | "duplicate_slide" => {
            keys(&["slide"])?;
            require_slide(object, at)?;
        }
        "move_slide" => {
            keys(&["slide", "to"])?;
            require_slide(object, at)?;
            require_slide_number(object, "to", at)?;
        }
        "add_notes" => {
            keys(&["slide", "text"])?;
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
    require_slide_number(object, "slide", at)
}

/// Require a 1-based slide number under `key` — an edit's `slide` or a
/// `move_slide`'s `to`.
fn require_slide_number(
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

/// A rule's values as the prompt spells one in a JSON edit: quoted and
/// pipe-separated, in the order the rules state them.
fn quoted_words<S: AsRef<str>>(words: &[S]) -> String {
    words
        .iter()
        .map(|word| format!("\"{}\"", word.as_ref()))
        .collect::<Vec<_>>()
        .join("|")
}

/// Require a pptx `format_text` to name at least one property, and each to be
/// the shape the kit writes: a boolean run toggle, a point size inside the
/// presentation's own bound, a hex colour, or a paragraph alignment.
fn require_pptx_format(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let mut any = require_toggles(object, &["bold", "italic", "underline"], at)?;
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
            "usage: {at} (format_text) needs at least one of bold, italic, underline, size, color \
             or align — hint: a boolean toggles bold, italic or underline, size is in points, \
             color is hex digits and align is {}",
            listed(&pair_words(&RULES.slide_alignments))
        );
    }
    Ok(())
}

/// Require an `add_image`'s optional placement and size: `x` and `y` are fractions
/// of the slide's own width and height and may sit on its edge, while `width` and
/// `height` are fractions of that size and may not be zero.
fn require_image_geometry(object: &serde_json::Map<String, Value>, at: &str) -> Result<()> {
    let place = "give the fraction of the slide the image is placed at";
    let size = "give the fraction of the slide the image is scaled to";
    for (key, span, what) in [
        ("x", &RULES.slide_position_fraction, place),
        ("y", &RULES.slide_position_fraction, place),
        ("width", &RULES.slide_size_fraction, size),
        ("height", &RULES.slide_size_fraction, size),
    ] {
        require_span_number(object, key, span, what, at)?;
    }
    Ok(())
}

/// Resolve and validate the `content` blocks, confining every embedded image
/// path to the workspace. Shapes the kit would otherwise fail on are rejected
/// here as usage errors.
async fn resolve_content(ws: &Workspace, args: &Value, format: &str) -> Result<Value> {
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
    for (index, block) in blocks.iter().enumerate() {
        let Some(object) = block.as_object() else {
            anyhow::bail!(
                "usage: content[{index}] must be an object — hint: give every block a \"type\""
            );
        };
        let Some(kind) = object.get("type").and_then(Value::as_str) else {
            anyhow::bail!(
                "usage: content[{index}] has no \"type\" — hint: use heading, paragraph, list, \
                 table, image or notes"
            );
        };
        // Every arm produces the block to push: it validates the shape and hands
        // back what the request will carry, the image's path resolved against the
        // workspace. One push after the match keeps the two in step.
        let resolved = match kind {
            "heading" => {
                require_string(object, "text", index)?;
                require_heading_level(object, index)?;
                block.clone()
            }
            "paragraph" => {
                require_string(object, "text", index)?;
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
                require_string(object, "text", index)?;
                block.clone()
            }
            "list" => {
                let Some(items) = object.get("items") else {
                    anyhow::bail!("usage: content[{index}] has no \"items\"");
                };
                require_scalar_array(items, &format!("content[{index}].items"))?;
                require_optional_bool(object, "ordered", index)?;
                block.clone()
            }
            "table" => {
                require_table_rows(object, index)?;
                // `headers` is optional — null counts as absent here exactly as
                // it does for `rows` and every other optional argument.
                match object.get("headers") {
                    None | Some(Value::Null) => {}
                    Some(headers) => {
                        require_scalar_array(headers, &format!("content[{index}].headers"))?;
                    }
                }
                normalize_table(object, index)?
            }
            "image" => {
                require_optional_size(object, "width", index)?;
                require_optional_size(object, "height", index)?;
                let Some(raw) = object.get("path").and_then(Value::as_str) else {
                    anyhow::bail!(
                        "usage: content[{index}] (image) has no \"path\" — hint: give the image's \
                         workspace path"
                    );
                };
                let (image, _) = resolve_input(ws, raw).await?;
                ensure_image(&image)?;
                let mut resolved = block.clone();
                resolved["path"] = json!(image.to_string_lossy());
                resolved
            }
            other => anyhow::bail!(
                "usage: content[{index}] has unknown type \"{other}\" — hint: use heading, \
                 paragraph, list, table, image or notes"
            ),
        };
        out.push(resolved);
    }
    Ok(Value::Array(out))
}

/// Resolve every `add_image` path in a validated pptx edit list against the
/// workspace, the way `create`'s image block is: the kit is handed a path it can
/// open, and the same strict read rule keeps an edit from embedding a file the
/// caller did not name inside the workspace. The list's shape — an array of
/// objects, each naming an `op`, an `add_image` also carrying a non-empty
/// `path` — is the pptx validator's own contract.
async fn resolve_edit_images(ws: &Workspace, edits: &mut Value) -> Result<()> {
    let list = edits
        .as_array_mut()
        .expect("validate_edits hands a pptx edit list back as an array");
    for edit in list {
        let object = edit
            .as_object_mut()
            .expect("validate_edits hands every pptx edit back as an object");
        if object.get("op").and_then(Value::as_str) != Some("add_image") {
            continue;
        }
        let raw = object
            .get("path")
            .and_then(Value::as_str)
            .expect("the pptx validator requires an add_image path")
            .to_owned();
        let (image, _) = resolve_input(ws, &raw).await?;
        ensure_image(&image)?;
        object.insert("path".to_owned(), json!(image.to_string_lossy()));
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
/// name and rows of cells, and every cell is a shape the kit's writer knows.
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
        let name = match object.get("name") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) => Some(name.as_str()),
            Some(value) => {
                return Err(super::wrong_type(
                    &format!("sheets[{index}].name"),
                    "a string",
                    value,
                ));
            }
        };
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
        normalized.push(Value::Object(normalized_sheet));
    }
    Ok(Value::Array(normalized))
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
    require_sheet_name_length(name, &at)?;
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
    if !seen.insert(name.to_lowercase()) {
        anyhow::bail!(
            "usage: {at} is not unique, ignoring case, got \"{name}\" — hint: give every sheet a \
             distinct name"
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
/// The kit's writer keeps the same rule as its own last line, and the paired
/// `the_tool_and_the_kit_refuse_the_same_shapes` drives both sides, so the two
/// cannot drift apart unnoticed.
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

/// Require a table block's `rows`, when it has any, to be an array of rows, each
/// an array of scalar cells, naming the offending position: a row that is not an
/// array, or a cell that is a structure, is a shape the kit's writers either
/// throw on or coerce to `[object Object]`. Absent (or null) is legal — a header
/// alone is a table.
fn require_table_rows(object: &serde_json::Map<String, Value>, index: usize) -> Result<()> {
    let rows = match object.get("rows") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(rows)) => rows,
        Some(v) => {
            return Err(super::wrong_type(
                &format!("content[{index}].rows"),
                "an array",
                v,
            ));
        }
    };
    for (row_index, row) in rows.iter().enumerate() {
        let Some(cells) = row.as_array() else {
            return Err(super::wrong_type(
                &format!("content[{index}].rows[{row_index}]"),
                "an array",
                row,
            ));
        };
        if let Some((cell_index, cell)) = scalar_array_violation(cells) {
            return Err(super::wrong_type(
                &format!("content[{index}].rows[{row_index}][{cell_index}]"),
                SCALAR_TYPE,
                cell,
            ));
        }
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

/// Normalize a shape-checked table block into the one the request carries: an
/// empty set is an absent one, so an empty `headers` array is removed and a
/// cell-less row is dropped. A writer handed a cell-less row would emit a row
/// with no cells — a `<w:tr/>` Word refuses to open — so only what the writers
/// will draw reaches them. A table left with nothing is still refused by
/// [`require_table_body`], which runs on the normalized block.
fn normalize_table(object: &serde_json::Map<String, Value>, index: usize) -> Result<Value> {
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
    require_table_body(&table, index)?;
    Ok(Value::Object(table))
}

/// Require a table block to hold something to lay out. No writer has a
/// presentation for a table with neither rows nor headers: the docx arm would
/// emit a `<w:tbl>` with no `<w:tr>`, which Word refuses to open, and the other
/// two would draw nothing at all for it. The shape is refused once here rather
/// than behaving differently per format.
fn require_table_body(object: &serde_json::Map<String, Value>, index: usize) -> Result<()> {
    let count = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    };
    if count("rows") + count("headers") == 0 {
        anyhow::bail!(
            "usage: content[{index}] (table) has no rows and no headers — hint: give it at least \
             one row or one header"
        );
    }
    Ok(())
}

/// `1, 2 or 3` — how a refusal hint and the description name the values a rule
/// allows.
fn listed<T: std::fmt::Display>(values: &[T]) -> String {
    match values {
        [] => String::new(),
        [only] => only.to_string(),
        [rest @ .., last] => format!(
            "{} or {last}",
            rest.iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
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
/// `image_side_px` upper bound is beyond any page, and anything else is not a
/// size at all.
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
            if !RULES.image_side_px.contains(number) {
                anyhow::bail!(
                    "usage: content[{index}].{key} must be {} pixels, got {number} — hint: give \
                     the size the image should be placed at",
                    RULES.image_side_px.bounds()
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

/// A present optional string, `at` naming where it sits for the refusal (`""`
/// for a top-level field): an absent or null value is `None`, a string is read,
/// and a value of another type is refused rather than read as absent (the shared
/// [`super::get_opt_str`] is deliberately silent, which would drop the caller's
/// value instead of telling them about it). The displayed path is built only on
/// the error path, so reading a value allocates nothing.
fn optional_string<'a>(value: Option<&'a Value>, at: &str, key: &str) -> Result<Option<&'a str>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(other) => {
            let named = if at.is_empty() {
                key.to_string()
            } else {
                format!("{at}.{key}")
            };
            Err(super::wrong_type(&named, "a string", other))
        }
    }
}

/// A present optional string of a top-level argument.
fn opt_string<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    optional_string(args.get(key), "", key)
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

/// Require `object[key]` to be a string, naming the block position in the error.
fn require_string(object: &serde_json::Map<String, Value>, key: &str, index: usize) -> Result<()> {
    match object.get(key) {
        Some(Value::String(_)) => Ok(()),
        Some(v) => Err(super::wrong_type(
            &format!("content[{index}].{key}"),
            "a string",
            v,
        )),
        None => anyhow::bail!("usage: content[{index}] has no \"{key}\""),
    }
}

/// Require an image path to carry one of the extensions the shared rules allow
/// the kit to embed; the bytes are checked against that name where the kit reads
/// them (see `assets/docgen/kit.js`), which is the one place every image goes
/// through.
fn ensure_image(path: &Path) -> Result<()> {
    let allowed: Vec<&str> = RULES.image_extensions.iter().map(String::as_str).collect();
    if crate::util::has_extension(path, &allowed) {
        Ok(())
    } else {
        anyhow::bail!(
            "usage: images must be PNG or JPEG, got {} — hint: convert the image and retry",
            path.display()
        )
    }
}

/// A parsed `pages` argument.
enum Pages {
    /// Every page (`"all"` or absent).
    All,
    /// 1-based page numbers in the order given.
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

/// Parse the optional `pages` argument.
fn opt_pages(args: &Value) -> Result<Pages> {
    match args.get("pages") {
        None | Some(Value::Null) => Ok(Pages::All),
        Some(Value::String(spec)) => parse_pages(spec),
        Some(v) => Err(super::wrong_type("pages", "a string", v)),
    }
}

/// Parse `"all"` (or empty) into [`Pages::All`], else a range list.
fn parse_pages(spec: &str) -> Result<Pages> {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
        return Ok(Pages::All);
    }
    let mut seen = HashSet::new();
    Ok(Pages::Explicit(parse_range(spec, &mut seen)?))
}

/// Parse one `pdf_split` range per output file.
fn split_groups(args: &Value) -> Result<Vec<Vec<u32>>> {
    let ranges = super::get_str_array(args, "ranges")?;
    if ranges.is_empty() {
        anyhow::bail!(
            "usage: missing required argument \"ranges\" — hint: one range per output file, \
             e.g. [\"1-3\", \"5\"]"
        );
    }
    // One output file per range, so the range count IS the attachment count: an
    // unbounded list would turn one call into as many files as the model asked
    // for, which is not something a reply can carry.
    if ranges.len() > MAX_SPLIT_PARTS {
        anyhow::bail!(
            "usage: {} output files is more than one call can produce (limit {MAX_SPLIT_PARTS}) — \
             hint: ask for fewer ranges, or wider ones",
            ranges.len()
        );
    }
    let mut seen = HashSet::new();
    let mut groups = Vec::with_capacity(ranges.len());
    for range in &ranges {
        groups.push(parse_range(range, &mut seen)?);
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
                "usage: empty page range in \"{spec}\" — hint: write ranges like \"1-3,5\""
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
            if seen.len() > MAX_PAGES_PER_REQUEST {
                anyhow::bail!("usage: too many pages requested (limit {MAX_PAGES_PER_REQUEST})");
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
/// size, where zero or a negative number is not a thing a document can show.
fn set_positive_number(request: &mut Value, args: &Value, key: &str) -> Result<()> {
    set_number(request, args, key)?;
    if let Some(number) = request.get(key).and_then(Value::as_f64)
        && !RULES.pdf_size_points.contains(number)
    {
        anyhow::bail!(
            "usage: \"{key}\" must be {}, got {number} — hint: give a size in points",
            RULES.pdf_size_points.bounds()
        );
    }
    Ok(())
}

/// Refuse a `generated/` directory that is a symlink: the reservations below name
/// a single component inside it and never follow a symlink for the file itself,
/// but a symlinked directory would take the whole call wherever it points — out
/// of the workspace, where the periodic sweep refuses to follow such an entry, so
/// what the call wrote would never be reclaimed either.
async fn refuse_symlinked_generated(dir: &Path) -> Result<()> {
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
    refuse_symlinked_generated(dir).await?;
    tokio::fs::create_dir_all(dir).await.with_context(|| {
        format!(
            "failed to create the generated directory at {}",
            dir.display()
        )
    })?;
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

/// A one-line description of what was produced.
fn describe(prefix: &str, names: &[String], sizes: &[u64]) -> String {
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
fn sanitize_base(raw: Option<&str>, fallback: &str) -> String {
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
    use crate::document::{DocOutcome, convert_document_file};
    use crate::util::test::{noisy_jpeg, noisy_png};
    use crate::workspace::test_ws_named;
    use std::io::{Cursor, Read as _, Write as _};

    /// A distinctive string that only a document with a real Cyrillic-capable
    /// font (and a correct encoding round-trip) can carry.
    const CYRILLIC: &str = "Кириллический текст";

    /// A character outside the Basic Multilingual Plane: one code point, two
    /// UTF-16 code units. An edit that measured a match's offset in code units
    /// would land before it, rewriting the wrong part of the text.
    const ASTRAL: &str = "\u{1F600}";

    /// End-to-end coverage needs the managed bun runtime, which the product
    /// installs on its first start. A check that needs it is `#[ignore]`d, so a
    /// standard test run stays green on a host that lacks the runtime; an
    /// explicit `--ignored` run on such a host prints the reason and returns
    /// instead of failing.
    fn runtime_missing() -> bool {
        if crate::tools::bun::bun_binary_path().is_some() {
            return false;
        }
        // Written to the OS stream rather than through `eprintln!`: libtest's
        // capture swallows the print macros of a passing test, and this line
        // exists to be seen.
        let _ = std::io::stderr().write_all(
            "SKIP: the managed bun runtime is not installed — start the product once to install it\n"
                .as_bytes(),
        );
        true
    }

    fn workspace() -> (tempfile::TempDir, crate::Workspace) {
        // The kit is materialized under the storage root, so a test reaching it
        // needs one resolved; the shared test root is set once per process, and
        // this is a no-op if another test set it first.
        let _ = crate::config::CONFIG.try_set_storage_root(crate::util::test::test_root().clone());
        let dir = tempfile::TempDir::new().expect("tempdir");
        // The workspace path must be canonical, as a stored workspace's is: a
        // raw `/tmp` spelling diverges from the resolved `/private/tmp` on macOS
        // and every strict read would be refused.
        let path = std::fs::canonicalize(dir.path()).expect("canonical workspace");
        let ws = test_ws_named(&path.to_string_lossy(), "document-test");
        (dir, ws)
    }

    async fn run(ws: &Workspace, args: Value) -> String {
        DocumentTool
            .execute(ws, args)
            .await
            .expect("the document call must succeed")
    }

    /// The paths carried by the `[FILE:…]` markers in a reply.
    fn file_paths(reply: &str) -> Vec<PathBuf> {
        reply
            .split("[FILE:")
            .skip(1)
            .filter_map(|rest| rest.split(']').next())
            .map(PathBuf::from)
            .collect()
    }

    /// The single path a one-file reply names.
    fn single(reply: &str) -> PathBuf {
        let mut paths = file_paths(reply);
        assert_eq!(paths.len(), 1, "expected one file marker in: {reply}");
        paths.pop().expect("one path")
    }

    fn open_zip(path: &Path) -> zip::ZipArchive<Cursor<Vec<u8>>> {
        zip::ZipArchive::new(Cursor::new(std::fs::read(path).expect("read package")))
            .expect("a readable package")
    }

    /// The package's PARTS, folder entries apart: a folder states itself by a
    /// name ending in `/`, which a zip may carry, and a folder's own entry is not
    /// a part — nothing is looked up by that name and it holds no bytes.
    fn zip_names(path: &Path) -> Vec<String> {
        let mut archive = open_zip(path);
        let mut names: Vec<String> = (0..archive.len())
            .map(|index| archive.by_index(index).expect("entry").name().to_string())
            .filter(|name| !name.ends_with('/'))
            .collect();
        names.sort();
        names
    }

    /// The `word/media/*` entries with their (decompressed) bytes, so a copy can
    /// be compared against its source.
    fn media_parts(path: &Path) -> Vec<(String, Vec<u8>)> {
        let mut archive = open_zip(path);
        let mut parts = Vec::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).expect("entry");
            if !entry.name().starts_with("word/media/") {
                continue;
            }
            let name = entry.name().to_string();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).expect("read media part");
            parts.push((name, bytes));
        }
        parts.sort_by(|a, b| a.0.cmp(&b.0));
        parts
    }

    fn assert_parts(path: &Path, names: &[&str]) {
        let archive = zip_names(path);
        for name in names {
            assert!(
                archive.iter().any(|entry| entry == name),
                "{name} missing from {}: {archive:?}",
                path.display()
            );
        }
    }

    /// One ZIP entry as text, from the package's own bytes — the primitive
    /// [`part_text`] and a fixture that must be modified before it is written
    /// out both read through.
    fn part_text_bytes(package: &[u8], name: &str) -> String {
        let mut archive = zip::ZipArchive::new(Cursor::new(package)).expect("open package");
        let mut entry = archive.by_name(name).expect("part");
        let mut text = String::new();
        entry.read_to_string(&mut text).expect("read part");
        text
    }

    /// One ZIP entry as text, read from the package at `path`.
    fn part_text(path: &Path, name: &str) -> String {
        part_text_bytes(&std::fs::read(path).expect("read package"), name)
    }

    /// One ZIP entry's decompressed bytes, so a part can be compared byte for
    /// byte against its source.
    fn part_bytes(path: &Path, name: &str) -> Vec<u8> {
        let mut archive = open_zip(path);
        let mut entry = archive.by_name(name).expect("part");
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).expect("read part");
        bytes
    }

    /// The elements named `child` inside the `container` element of `xml`, in
    /// the file's own order and each taken whole — a styles part's
    /// `<font>`/`<fill>`/`<border>`/`<xf>` entries. The self-closing shape is
    /// taken as it is and the child-bearing one with its close.
    fn element_children<'a>(xml: &'a str, container: &str, child: &str) -> Vec<&'a str> {
        let at = xml
            .find(&format!("<{container}"))
            .unwrap_or_else(|| panic!("no <{container}> in: {xml}"));
        let body_start = xml[at..].find('>').expect("a container open tag") + at + 1;
        let body_end = xml[body_start..]
            .find(&format!("</{container}>"))
            .expect("a container close tag")
            + body_start;
        let closing = format!("</{child}>");
        let mut rest = &xml[body_start..body_end];
        let mut entries = Vec::new();
        while let Some(found) = rest.find(&format!("<{child}")) {
            let element = &rest[found..];
            let open_end = element.find('>').expect("an element open tag");
            let length = if element.as_bytes()[open_end - 1] == b'/' {
                open_end + 1
            } else {
                element.find(&closing).expect("an element close tag") + closing.len()
            };
            entries.push(&element[..length]);
            rest = &element[length..];
        }
        entries
    }

    /// An element's own `name="…"` attribute, read from its open tag alone, or
    /// `None` when it states none.
    fn attribute_value<'a>(element: &'a str, name: &str) -> Option<&'a str> {
        let open = &element[..element.find('>').expect("an element open tag")];
        let needle = format!(" {name}=\"");
        let at = open.find(&needle)?;
        let value = &open[at + needle.len()..];
        Some(&value[..value.find('"').expect("a closing quote")])
    }

    /// An element's own `name="…"` attribute, read from its open tag alone.
    fn attribute_of<'a>(element: &'a str, name: &str) -> &'a str {
        attribute_value(element, name).unwrap_or_else(|| panic!("no {name} on: {element}"))
    }

    /// Assert a styles part this kit grew is one a reader can open: every block a
    /// `cellXfs`/`cellStyleXfs` entry's ids resolve into is stated, each with its
    /// entry 0, in `CT_Stylesheet` order — so no `<xf>` names a `fontId`, `fillId`,
    /// `borderId` or `xfId` of a block the part does not hold.
    fn assert_styles_part_is_openable(styles: &str) {
        // Each id a `<xf>` may state, the block it resolves into, and that block's
        // own entry element.
        let references = [
            ("fontId", "fonts", "font"),
            ("fillId", "fills", "fill"),
            ("borderId", "borders", "border"),
            ("xfId", "cellStyleXfs", "xf"),
        ];
        // The blocks a reader meets in this order, top to bottom.
        let order = [
            "fonts",
            "fills",
            "borders",
            "cellStyleXfs",
            "cellXfs",
            "cellStyles",
        ];
        let found: Vec<usize> = order
            .iter()
            .map(|block| {
                styles
                    .find(&format!("<{block}"))
                    .unwrap_or_else(|| panic!("the grown part states no <{block}>: {styles}"))
            })
            .collect();
        assert!(
            found.windows(2).all(|pair| pair[0] < pair[1]),
            "the grown part's blocks are out of CT_Stylesheet order: {styles}"
        );
        for (_, block, entry) in references {
            assert!(
                !element_children(styles, block, entry).is_empty(),
                "the grown part states no <{block}> entry 0: {styles}"
            );
        }
        for parent in ["cellStyleXfs", "cellXfs"] {
            for xf in element_children(styles, parent, "xf") {
                for (attribute, block, entry) in references {
                    let Some(index) = attribute_value(xf, attribute) else {
                        continue;
                    };
                    let index: usize = index.parse().expect("a numeric id");
                    let entries = element_children(styles, block, entry).len();
                    assert!(
                        index < entries,
                        "the <xf> names {attribute}=\"{index}\" past the {entries} <{block}> \
                         entries: {styles}"
                    );
                }
            }
        }
    }

    /// The `<c>` element addressing `address` in a sheet part, taken whole.
    fn cell_element<'a>(sheet: &'a str, address: &str) -> &'a str {
        let at = sheet
            .find(&format!("<c r=\"{address}\""))
            .unwrap_or_else(|| panic!("no cell {address} in: {sheet}"));
        let element = &sheet[at..];
        let open_end = element.find('>').expect("a cell open tag");
        let length = if element.as_bytes()[open_end - 1] == b'/' {
            open_end + 1
        } else {
            element.find("</c>").expect("a cell close tag") + "</c>".len()
        };
        &element[..length]
    }

    /// Whether `needle` sits between an `open` tag and its `close` tag in `xml`:
    /// the opens before it outnumber the closes, so the caller states the exact
    /// tags a nesting is measured with.
    fn inside_element(xml: &str, needle: &str, open: &str, close: &str) -> bool {
        let at = xml.find(needle).expect("the text to place");
        let before = &xml[..at];
        before.matches(open).count() > before.matches(close).count()
    }

    /// `xml`'s `<p:sldIdLst>` with every `<p:sldId>`'s relationship id dropped:
    /// the deck the reader can only number by file name.
    fn without_sld_id_rel_ids(xml: &str) -> String {
        let open = xml.find("<p:sldIdLst>").expect("a slide list");
        let close = xml[open..]
            .find("</p:sldIdLst>")
            .expect("a slide list close")
            + open;
        let mut list = String::new();
        let mut rest = &xml[open..close];
        while let Some(at) = rest.find(" r:id=\"") {
            list.push_str(&rest[..at]);
            let after = &rest[at + " r:id=\"".len()..];
            rest = &after[after.find('"').expect("a closing quote") + 1..];
        }
        list.push_str(rest);
        format!("{}{}{}", &xml[..open], list, &xml[close..])
    }

    /// The `<a:p>…</a:p>` whose run text is exactly `text`: the paragraph a
    /// slide's text sits in, so a caller can assert what the paragraph itself
    /// carries — an alignment, say — rather than the run.
    fn paragraph_element<'a>(xml: &'a str, text: &str) -> &'a str {
        let at = xml
            .find(&format!("<a:t>{text}</a:t>"))
            .expect("the paragraph's run text");
        let start = xml[..at].rfind("<a:p>").expect("the paragraph's open");
        let end = xml[at..].find("</a:p>").expect("the paragraph's close") + at;
        &xml[start..end]
    }

    /// The relationship id a picture names in its `r:embed`.
    fn embedding_id(picture: &str) -> &str {
        let attribute = "r:embed=\"";
        let after = &picture
            [picture.find(attribute).expect("a picture names an image") + attribute.len()..];
        &after[..after.find('"').expect("a closing quote")]
    }

    /// `xml` with every `<Relationship …/>` whose element text names `needle`
    /// dropped: a part a caller needs to lack a relationship it would otherwise
    /// declare.
    fn without_relationship(xml: &str, needle: &str) -> String {
        let mut out = String::new();
        let mut rest = xml;
        while let Some(at) = rest.find("<Relationship") {
            out.push_str(&rest[..at]);
            let element = &rest[at..];
            let end = element.find("/>").expect("a self-closing relationship") + "/>".len();
            if !element[..end].contains(needle) {
                out.push_str(&element[..end]);
            }
            rest = &element[end..];
        }
        out.push_str(rest);
        out
    }

    /// Add one part to an in-progress package.
    fn add_part(zip: &mut zip::ZipWriter<Cursor<Vec<u8>>>, name: &str, body: &str) {
        add_bytes(zip, name, body.as_bytes());
    }

    /// Add one part with raw bytes to an in-progress package.
    fn add_bytes(zip: &mut zip::ZipWriter<Cursor<Vec<u8>>>, name: &str, body: &[u8]) {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .expect("start part");
        zip.write_all(body).expect("write part");
    }

    /// A package holding the parts of `package` — the names in `dropped` left out —
    /// with `parts` written beside them, a name already there replaced. Folder
    /// entries are not parts.
    fn repacked(package: &[u8], dropped: &[&str], parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        {
            let mut archive =
                zip::ZipArchive::new(Cursor::new(package.to_vec())).expect("open package");
            for index in 0..archive.len() {
                let mut entry = archive.by_index(index).expect("entry");
                let name = entry.name().to_string();
                if name.ends_with('/')
                    || dropped.contains(&name.as_str())
                    || parts.iter().any(|(part, _)| *part == name)
                {
                    continue;
                }
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes).expect("read part");
                add_bytes(&mut zip, &name, &bytes);
            }
        }
        for (name, body) in parts {
            add_bytes(&mut zip, name, body);
        }
        zip.finish().expect("finish package").into_inner()
    }

    /// `package` with `parts` added, a name it already holds replaced. Used to
    /// give a fixture the parts an edit must carry through untouched — a header,
    /// a footer, a chart, a media file — whose bodies may be minimal XML.
    fn with_parts(package: &[u8], parts: &[(&str, &[u8])]) -> Vec<u8> {
        repacked(package, &[], parts)
    }

    /// `xml` with the first `needle` replaced by `with`: a fixture that states an
    /// element the way no writer does (a container opened and never closed).
    fn with_first(xml: &str, needle: &str, with: &str) -> String {
        let at = xml
            .find(needle)
            .unwrap_or_else(|| panic!("no {needle} in: {xml}"));
        format!("{}{}{}", &xml[..at], with, &xml[at + needle.len()..])
    }

    /// A `pptx` package whose parts are not the ones a number spells: its layout is
    /// numbered with a leading zero (`slideLayout01.xml`, and no `slideLayout1.xml`),
    /// and of the two notes masters it holds it declares and uses the second, the
    /// first being a part nothing points at. The other families' writers only ever
    /// number their parts plainly and keep one master, so this is written by hand
    /// rather than produced by one.
    fn leading_zero_deck() -> Vec<u8> {
        const LAYOUT: &str = r#"<?xml version="1.0"?><p:sldLayout xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"#;
        const MASTER: &str = r#"<?xml version="1.0"?><p:notesMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"#;
        const SLIDE: &str = r#"<?xml version="1.0"?><p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Первый</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#;
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        add_part(
            &mut zip,
            "[Content_Types].xml",
            concat!(
                r#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">"#,
                r#"<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>"#,
                r#"<Default Extension="xml" ContentType="application/xml"/>"#,
                r#"<Override PartName="/ppt/slideLayouts/slideLayout01.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>"#,
                r#"<Override PartName="/ppt/notesMasters/notesMaster01.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>"#,
                r#"<Override PartName="/ppt/notesMasters/notesMaster02.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>"#,
                r#"<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>"#,
                "</Types>",
            ),
        );
        add_part(
            &mut zip,
            "ppt/presentation.xml",
            concat!(
                r#"<?xml version="1.0"?><p:presentation xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
                r#"<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>"#,
                r#"<p:notesMasterIdLst><p:notesMasterId r:id="rId3"/></p:notesMasterIdLst>"#,
                r#"<p:sldSz cx="12192000" cy="6858000"/></p:presentation>"#,
            ),
        );
        add_part(
            &mut zip,
            "ppt/_rels/presentation.xml.rels",
            concat!(
                r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
                r#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>"#,
                r#"<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="slideLayouts/slideLayout01.xml"/>"#,
                r#"<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="notesMasters/notesMaster02.xml"/>"#,
                "</Relationships>",
            ),
        );
        add_part(&mut zip, "ppt/slides/slide1.xml", SLIDE);
        add_part(&mut zip, "ppt/slideLayouts/slideLayout01.xml", LAYOUT);
        add_part(&mut zip, "ppt/notesMasters/notesMaster01.xml", MASTER);
        add_part(&mut zip, "ppt/notesMasters/notesMaster02.xml", MASTER);
        zip.finish().expect("finish package").into_inner()
    }

    /// Write a fixture package into the workspace and return its path.
    fn write_fixture(ws: &Workspace, name: &str, bytes: &[u8]) -> PathBuf {
        let path = ws.as_path().join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    /// A freshly created one-sheet workbook (`S`, one seed row): the package the
    /// format tests start from, each writing the parts it means to edit over it.
    async fn created_xlsx(ws: &Workspace) -> Vec<u8> {
        let path = single(
            &run(
                ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        std::fs::read(&path).expect("read base package")
    }

    /// The same package with one part dropped — the shape a deck has when it
    /// declares a part it no longer holds.
    fn without_part(package: &[u8], name: &str) -> Vec<u8> {
        repacked(package, &[name], &[])
    }

    /// A deck whose presentation numbers a `<p:sldId>` its list does not hold: a
    /// slide of its own the reader shows BEFORE the list's three, so the entry at a
    /// position in the list is the slide after the one the reader shows there. The
    /// stray slide shows `STRAY` where the list's show `FIRST`, `SECOND` and `THIRD`,
    /// and its part is a copy of the first slide's, so which part an edit removes is
    /// visible in the result. Built from the writer's own deck because no writer
    /// emits a `<p:sldId>` outside the list.
    async fn stray_sld_id_deck(ws: &Workspace, name: &str) -> PathBuf {
        let created = single(
            &run(
                ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "FIRST" },
                        { "type": "heading", "level": 1, "text": "SECOND" },
                        { "type": "heading", "level": 1, "text": "THIRD" },
                    ],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let stray = part_text_bytes(&base, "ppt/slides/slide1.xml").replace("FIRST", "STRAY");
        let presentation = part_text_bytes(&base, "ppt/presentation.xml").replacen(
            "<p:sldIdLst>",
            r#"<p:sldId id="999" r:id="rId99"/><p:sldIdLst>"#,
            1,
        );
        let rels = part_text_bytes(&base, "ppt/_rels/presentation.xml.rels").replace(
            "</Relationships>",
            concat!(
                r#"<Relationship Id="rId99" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide4.xml"/>"#,
                "</Relationships>",
            ),
        );
        write_fixture(
            ws,
            name,
            &with_parts(
                &base,
                &[
                    ("ppt/presentation.xml", presentation.as_bytes()),
                    ("ppt/_rels/presentation.xml.rels", rels.as_bytes()),
                    ("ppt/slides/slide4.xml", stray.as_bytes()),
                ],
            ),
        )
    }

    /// A real-Excel-shaped workbook: its string lives in `xl/sharedStrings.xml`
    /// and the cell only indexes into it (`t="s"`), which is what Excel, Google
    /// Sheets and LibreOffice all write. A pass that reads only the cell text
    /// would leave the placeholder in place, so this shape is the case the
    /// shared table exists for. The text is written with numeric character
    /// references, as openpyxl writes every non-ASCII character: resolving them
    /// before substituting is what keeps the sample's own text readable.
    fn shared_string_workbook() -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        add_part(
            &mut zip,
            "xl/workbook.xml",
            r#"<?xml version="1.0"?><workbook><sheets><sheet name="Лист" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        );
        add_part(
            &mut zip,
            "xl/sharedStrings.xml",
            r#"<?xml version="1.0"?><sst><si><t>&#1054;&#1090;&#1095;&#1105;&#1090; для {name}</t></si><si><t>{amount}</t></si></sst>"#,
        );
        add_part(
            &mut zip,
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0"?><worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row><row r="2"><c r="A2" t="s"><v>1</v></c></row></sheetData></worksheet>"#,
        );
        // A part the fill must copy through untouched.
        add_part(&mut zip, "docProps/app.xml", "DISTINCTIVE-APP-PART-BYTES");
        zip.finish().expect("finish package").into_inner()
    }

    /// A workbook whose cells carry NO `r`: a writer need not address every cell
    /// it writes, so the filler falls back to the cell's position in its row —
    /// the rule the reader uses — instead of leaving the placeholder in place.
    fn addressless_workbook() -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        add_part(
            &mut zip,
            "xl/workbook.xml",
            r#"<?xml version="1.0"?><workbook><sheets><sheet name="Лист" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        );
        add_part(
            &mut zip,
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0"?><worksheet><sheetData><row><c t="inlineStr"><is><t>{name}</t></is></c><c><v>1</v></c></row></sheetData></worksheet>"#,
        );
        zip.finish().expect("finish package").into_inner()
    }

    async fn converted_text(ws: &Workspace, path: &Path) -> String {
        let out_dir = ws.as_path().join("extracted");
        match convert_document_file(path, &file_name_of(path), &out_dir).await {
            DocOutcome::Text { text, .. } => text,
            DocOutcome::Unreadable { reason } => {
                panic!("cannot read {}: {reason}", path.display())
            }
            DocOutcome::Unsupported => panic!("unsupported document {}", path.display()),
        }
    }

    /// A one-page PDF whose only content is `text`.
    async fn pdf_with(ws: &Workspace, name: &str, text: &str) -> PathBuf {
        let reply = run(
            ws,
            json!({
                "action": "create", "format": "pdf", "file_name": name,
                "content": [{ "type": "paragraph", "text": text }],
            }),
        )
        .await;
        single(&reply)
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

    /// A minimal PDF the library refuses to open without a password: its trailer
    /// points at an `/Encrypt` dictionary. Built from `assemble_pdf`'s objects
    /// with the key appended to the trailer, which is not itself in the xref
    /// table, so every offset stays valid.
    fn encrypted_pdf() -> Vec<u8> {
        let objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>".to_vec(),
            b"<< /Filter /Standard /V 1 /R 2 /O <00> /U <00> /P -1 >>".to_vec(),
        ];
        let assembled = crate::document::test_fixtures::assemble_pdf(&objects);
        let encrypted = String::from_utf8(assembled)
            .expect("the fixture is ASCII")
            .replace(
                "<< /Size 5 /Root 1 0 R >>",
                "<< /Size 5 /Root 1 0 R /Encrypt 4 0 R >>",
            );
        encrypted.into_bytes()
    }

    /// A Word document is created and read back: the text survives a round trip
    /// through the shared converter and the package carries its required parts.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn created_docx_reads_back() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let reply = run(
            &ws,
            json!({
                "action": "create", "format": "docx", "file_name": "report",
                "content": [
                    { "type": "heading", "level": 1, "text": "Заголовок" },
                    { "type": "paragraph", "text": CYRILLIC },
                    { "type": "list", "items": ["раз", "два"], "ordered": true },
                    { "type": "table", "headers": ["Колонка"], "rows": [["значение"]] },
                ],
            }),
        )
        .await;
        let path = single(&reply);
        assert_parts(&path, &["[Content_Types].xml", "word/document.xml"]);
        assert!(
            converted_text(&ws, &path).await.contains(CYRILLIC),
            "docx text lost"
        );
    }

    /// A table whose headers are empty and which holds a cell-less row draws
    /// only the rows that have cells: a `<w:tr/>` is a file Word refuses to
    /// open, and the call reports success, so the writer must never emit one.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn a_docx_table_never_holds_a_cell_less_row() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let reply = run(
            &ws,
            json!({
                "action": "create", "format": "docx", "file_name": "rows",
                "content": [
                    { "type": "table", "headers": [], "rows": [["a"], [], ["b"]] },
                ],
            }),
        )
        .await;
        let path = single(&reply);
        let body = part_text(&path, "word/document.xml");
        assert_eq!(
            body.matches("<w:tr").count(),
            2,
            "a cell-less row must not be written: {body}"
        );
        assert!(
            !body.contains("<w:tr/>"),
            "a cell-less row reached the file: {body}"
        );
    }

    /// A table is created and read back, cells and a formula without a stored
    /// value included.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn created_xlsx_reads_back() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let reply = run(
            &ws,
            json!({
                "action": "create", "format": "xlsx", "file_name": "table",
                "sheets": [
                    { "name": "Лист", "rows": [[CYRILLIC, 42, true, { "formula": "SUM(B1:B1)" }, { "formula": " =SUM(E1:E1)" }]] },
                    { "name": "Второй", "rows": [["а", "б"]] },
                ],
            }),
        )
        .await;
        let path = single(&reply);
        assert_parts(
            &path,
            &[
                "[Content_Types].xml",
                "xl/workbook.xml",
                "xl/worksheets/sheet1.xml",
            ],
        );
        assert!(
            converted_text(&ws, &path).await.contains(CYRILLIC),
            "xlsx text lost"
        );
        // A formula is written as its text alone: the leading `=` the caller may
        // wrap it in (with or without spacing) must not reach the `<f>` element,
        // where it is no longer a formula.
        let sheet = part_text(&path, "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains("<f>SUM(E1:E1)</f>"),
            "a formula must be written as its text: {sheet}"
        );
        // A cell nests inside its `<row>`: a sheet listing cells straight under
        // `<sheetData>` looks well-formed here but opens empty in real readers.
        let sheet = part_text(&path, "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains("<row r=\"1\">"),
            "cells must sit in their row: {sheet}"
        );
        // The styles part is the one `create` has always written — a `format_cells`
        // edit must not change what creating a workbook produces: entry 0 of each
        // block it needs, and no default style, which a workbook that names no style
        // does not state.
        let styles = part_text(&path, "xl/styles.xml");
        assert!(
            styles.contains(r#"<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>"#)
                && !styles.contains("<cellStyles"),
            "the styles part a created workbook carries changed: {styles}"
        );
    }

    /// A presentation is created and read back: the slide text and the speaker
    /// notes both survive, with the notes delivered separately from the slide.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn created_pptx_reads_back() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let reply = run(
            &ws,
            json!({
                "action": "create", "format": "pptx", "file_name": "slides",
                "content": [
                    { "type": "heading", "level": 1, "text": "Заголовок" },
                    { "type": "paragraph", "text": CYRILLIC },
                    { "type": "list", "items": ["раз", "два"], "ordered": true },
                    { "type": "table", "headers": ["А", "Б"], "rows": [["1", "2"]] },
                    { "type": "notes", "text": "Заметка" },
                ],
            }),
        )
        .await;
        let path = single(&reply);
        assert_parts(
            &path,
            &[
                "[Content_Types].xml",
                "ppt/presentation.xml",
                "ppt/slides/slide1.xml",
            ],
        );
        let slides = converted_text(&ws, &path).await;
        assert!(slides.contains(CYRILLIC), "pptx text lost: {slides}");
        assert!(
            slides.contains("Slide 1 notes:") && slides.contains("Заметка"),
            "the presenter notes must be delivered separately: {slides}"
        );
    }

    /// The `<p:sp>` shape named `name` in a slide part, taken whole — the shape a
    /// title is declared in, read out so its own frame and formatting can be
    /// asserted on.
    fn named_shape<'a>(xml: &'a str, name: &str) -> &'a str {
        let needle = format!("name=\"{name}\"");
        let at = xml
            .find(&needle)
            .unwrap_or_else(|| panic!("no shape named {name} in: {xml}"));
        let start = xml[..at].rfind("<p:sp>").expect("a shape open tag");
        let close = at + xml[at..].find("</p:sp>").expect("a shape close tag") + "</p:sp>".len();
        &xml[start..close]
    }

    /// The whole `name` element in `xml`, its body with it, or the self-closing
    /// tag on its own when it has none.
    fn element_named<'a>(xml: &'a str, name: &str) -> &'a str {
        let at = xml
            .find(&format!("<{name}"))
            .unwrap_or_else(|| panic!("no <{name}> in: {xml}"));
        let element = &xml[at..];
        let open_end = element.find('>').expect("an element open tag");
        if element.as_bytes()[open_end - 1] == b'/' {
            return &element[..=open_end];
        }
        let close = format!("</{name}>");
        let end = element.find(&close).expect("an element close tag") + close.len();
        &element[..end]
    }

    /// `name`'s open tag in `xml`, a self-closing slash dropped, so an element one
    /// writer opens and closes (`<x></x>`) and one the other self-closes (`<x/>`)
    /// read alike.
    fn open_tag(xml: &str, name: &str) -> String {
        let at = xml
            .find(&format!("<{name}"))
            .unwrap_or_else(|| panic!("no <{name}> in: {xml}"));
        let open_end = at + xml[at..].find('>').expect("an element open tag");
        let tag = &xml[at..=open_end];
        tag.strip_suffix("/>")
            .map_or_else(|| tag.to_string(), |head| format!("{head}>"))
    }

    /// The formatting a slide's title shape states, as one string both writers
    /// spell alike: the frame, the body properties, the paragraph's and the run's
    /// — the fonts the run states among them.
    fn title_shape_formatting(shape: &str) -> String {
        format!(
            "{}|{}|{}|{}",
            element_named(shape, "a:xfrm"),
            open_tag(shape, "a:bodyPr"),
            element_named(shape, "a:pPr"),
            element_named(shape, "a:rPr"),
        )
    }

    /// The frame, the declaration and the formatting a declared title states: the
    /// placeholder, the frame in EMU, the vertical centring, the left alignment,
    /// and the run's size, weight and three fonts.
    fn assert_declared_title(shape: &str) {
        assert!(
            shape.contains(r#"<p:ph type="title"/>"#),
            "the title shape declares no title placeholder: {shape}"
        );
        for fragment in [
            r#"<a:off x="365760" y="274320"/>"#,
            r#"<a:ext cx="8412480" cy="731520"/>"#,
            r#"anchor="ctr""#,
            r#"algn="l""#,
            r#"sz="2800""#,
            r#"b="1""#,
            r#"<a:latin typeface="+mn-lt"/>"#,
            r#"<a:ea typeface="+mn-ea"/>"#,
            r#"<a:cs typeface="+mn-cs"/>"#,
        ] {
            assert!(
                shape.contains(fragment),
                "the title shape states no {fragment}: {shape}"
            );
        }
    }

    /// Where a relationship's `Target` points: resolved against the directory that
    /// owns the `.rels` part — `_rels/.rels` against the package root — with `.`
    /// and `..` applied.
    fn resolve_relationship_target(rels_part: &str, target: &str) -> String {
        if let Some(absolute) = target.strip_prefix('/') {
            return absolute.to_string();
        }
        let owner = rels_part
            .split("_rels/")
            .next()
            .unwrap_or("")
            .trim_end_matches('/');
        let mut parts: Vec<&str> = owner.split('/').filter(|part| !part.is_empty()).collect();
        for component in target.split('/') {
            match component {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                other => parts.push(other),
            }
        }
        parts.join("/")
    }

    /// The note count `docProps/app.xml` declares as `<Notes>`.
    fn notes_count(app: &str) -> u32 {
        app.split("<Notes>")
            .nth(1)
            .and_then(|rest| rest.split("</Notes>").next())
            .and_then(|value| value.parse().ok())
            .expect("a <Notes> count in the application properties")
    }

    /// Whether a part is a relationships part — the one the package states at its
    /// root (`_rels/.rels`) or one beside a part, always in a `_rels` folder. Read
    /// off the name rather than `Path::extension`, which reads a name that IS the
    /// extension (`.rels`) as no extension at all.
    fn is_relationships_part(name: &str) -> bool {
        name.split('/').any(|part| part == "_rels") && name.rsplit('.').next() == Some("rels")
    }

    /// The `ppt/slides/slideN.xml` part whose text holds `needle`.
    fn slide_part_holding(path: &Path, needle: &str) -> String {
        zip_names(path)
            .into_iter()
            .find(|name| {
                name.starts_with("ppt/slides/slide")
                    && Path::new(name).extension().is_some_and(|ext| ext == "xml")
                    && part_text(path, name).contains(needle)
            })
            .unwrap_or_else(|| panic!("no slide part holds {needle}"))
    }

    /// A created deck's package is sound: every part an `<Override>` declares is
    /// in the archive and every part has a type — its own `<Override>` or a
    /// `<Default>` for its extension — every relationship in every `.rels` part
    /// resolves to a part that exists, and the archive states no folder entry.
    fn assert_created_deck_parts_resolve(path: &Path) {
        let mut archive = open_zip(path);
        let mut names: Vec<String> = Vec::new();
        let mut text: Vec<(String, String)> = Vec::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).expect("entry");
            let name = entry.name().to_string();
            assert!(
                !name.ends_with('/'),
                "the archive states a folder entry: {name}"
            );
            let mut body = String::new();
            entry.read_to_string(&mut body).expect("read part");
            names.push(name.clone());
            text.push((name, body));
        }
        let types = text
            .iter()
            .find(|(name, _)| name == "[Content_Types].xml")
            .map(|(_, body)| body.as_str())
            .expect("the package's content types");
        let defaults: Vec<&str> = element_children(types, "Types", "Default")
            .iter()
            .map(|element| attribute_of(element, "Extension"))
            .collect();
        let declared: Vec<String> = element_children(types, "Types", "Override")
            .iter()
            .map(|element| {
                attribute_of(element, "PartName")
                    .trim_start_matches('/')
                    .to_string()
            })
            .collect();
        for part in &declared {
            assert!(
                names.contains(part),
                "an <Override> declares the missing part {part}"
            );
        }
        for name in &names {
            if name == "[Content_Types].xml" {
                continue;
            }
            let extension = name.rsplit('.').next().unwrap_or("");
            assert!(
                declared.contains(name)
                    || defaults
                        .iter()
                        .any(|known| known.eq_ignore_ascii_case(extension)),
                "no <Override> and no <Default> gives {name} a type"
            );
        }
        for (rels, body) in &text {
            if !is_relationships_part(rels) {
                continue;
            }
            for element in element_children(body, "Relationships", "Relationship") {
                if attribute_value(element, "TargetMode")
                    .is_some_and(|mode| mode.eq_ignore_ascii_case("External"))
                {
                    continue;
                }
                let target = attribute_of(element, "Target");
                let resolved = resolve_relationship_target(rels, target);
                assert!(
                    names.contains(&resolved),
                    "{rels} names {target}, which resolves to the missing part {resolved}"
                );
            }
        }
    }

    /// A created deck declares the block that opens a slide as the slide's real
    /// title — a shape stating the `title` placeholder, its frame and its
    /// formatting — which the reader shows as a title; a slide a paragraph opens
    /// is declared nothing of the sort. The package it comes in is sound, and the
    /// presentation's children follow the format's sequence.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn created_pptx_declares_its_slides_titles() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let path = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "titled",
                    "content": [
                        { "type": "paragraph", "text": "Только текст" },
                        { "type": "heading", "level": 1, "text": "Заголовок" },
                    ],
                }),
            )
            .await,
        );

        // A slide a paragraph opens is no title at all.
        let untitled = part_text(&path, "ppt/slides/slide1.xml");
        assert!(
            !untitled.contains("<p:ph"),
            "the title-less slide declares a placeholder: {untitled}"
        );
        // A slide a level-1 heading opens is the slide's title: the placeholder,
        // the frame in EMU and the formatting the declaration adds.
        assert_declared_title(named_shape(
            &part_text(&path, "ppt/slides/slide2.xml"),
            "Title",
        ));

        // The reader shows the declared title as its mark, and the title-less
        // slide carries none.
        let text = converted_text(&ws, &path).await;
        assert!(
            text.contains("(title) Заголовок"),
            "the reader did not show the declared title: {text}"
        );
        assert_eq!(
            text.matches("(title)").count(),
            1,
            "only the declared title is a title: {text}"
        );

        // The package is sound, and the presentation's children are in the order
        // the format's sequence asks for.
        assert_created_deck_parts_resolve(&path);
        let presentation = part_text(&path, "ppt/presentation.xml");
        let offsets: Vec<usize> = [
            "<p:sldMasterIdLst",
            "<p:notesMasterIdLst",
            "<p:sldIdLst",
            "<p:sldSz",
            "<p:notesSz",
        ]
        .iter()
        .map(|child| {
            presentation
                .find(child)
                .unwrap_or_else(|| panic!("the presentation holds no {child}: {presentation}"))
        })
        .collect();
        for pair in offsets.windows(2) {
            assert!(
                pair[0] < pair[1],
                "the presentation's children are out of sequence: {presentation}"
            );
        }
    }

    /// A created deck holds a notes page only for a slide that has notes: one
    /// created without any declares no notes part, no slide relationship to one
    /// and no note count; one created with a notes block keeps its page and counts
    /// it.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn created_pptx_holds_notes_pages_only_for_noted_slides() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let plain = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "plain",
                    "content": [{ "type": "paragraph", "text": "Только текст" }],
                }),
            )
            .await,
        );
        let parts = zip_names(&plain);
        assert!(
            !parts
                .iter()
                .any(|name| name.starts_with("ppt/notesSlides/")),
            "a deck with no notes holds a notes page: {parts:?}"
        );
        for rels in parts.iter().filter(|name| is_relationships_part(name)) {
            assert!(
                !part_text(&plain, rels).contains("notesSlide"),
                "{rels} still declares a notes page"
            );
        }
        let app = part_text(&plain, "docProps/app.xml");
        assert_eq!(
            notes_count(&app),
            0,
            "a deck with no notes counts some: {app}"
        );

        let noted = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "noted",
                    "content": [
                        { "type": "paragraph", "text": "Только текст" },
                        { "type": "notes", "text": "Заметка" },
                    ],
                }),
            )
            .await,
        );
        let parts = zip_names(&noted);
        assert!(
            parts
                .iter()
                .any(|name| name.starts_with("ppt/notesSlides/notesSlide")),
            "a deck with notes holds no notes page: {parts:?}"
        );
        assert_ne!(
            notes_count(&part_text(&noted, "docProps/app.xml")),
            0,
            "a deck with notes counts none"
        );
    }

    /// A slide `add_slide` writes declares the same title a created slide opens
    /// with: the same frame, centring, alignment and run formatting, written by
    /// the slide arm's own title builder rather than the create arm's.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_added_slide_declares_a_title_like_a_created_one() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Заголовок" },
                        { "type": "paragraph", "text": "Тело" },
                    ],
                }),
            )
            .await,
        );
        let created_xml = part_text(&created, "ppt/slides/slide1.xml");
        let created_title = named_shape(&created_xml, "Title");
        assert_declared_title(created_title);

        let output = single(
            &run(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "edited",
                    "path": created.to_string_lossy(),
                    "edits": [
                        { "op": "add_slide", "after": 1, "title": "Добавленный",
                          "bullets": ["раз", "два"] },
                        { "op": "add_slide", "bullets": ["Без заголовка"] },
                    ],
                }),
            )
            .await,
        );
        let added_part = slide_part_holding(&output, "Добавленный");
        let added_xml = part_text(&output, &added_part);
        let added_title = named_shape(&added_xml, "Title");
        assert_declared_title(added_title);
        assert_eq!(
            title_shape_formatting(created_title),
            title_shape_formatting(added_title),
            "the added slide's title does not state the created one's formatting"
        );

        // A slide added with no title declares none: its bullets are a body
        // placeholder and nothing else.
        let untitled = part_text(&output, &slide_part_holding(&output, "Без заголовка"));
        assert!(
            untitled.contains(r#"<p:ph type="body" idx="1"/>"#),
            "the added slide's bullets are not a body placeholder: {untitled}"
        );
        assert!(
            !untitled.contains(r#"<p:ph type="title"/>"#),
            "a slide added with no title declares one: {untitled}"
        );
    }

    /// A PDF is created and read back: the magic is there, and the round trip
    /// proves the embedded font really carries the Cyrillic alphabet.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn created_pdf_reads_back() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let reply = run(
            &ws,
            json!({
                "action": "create", "format": "pdf", "file_name": "paper",
                "content": [{ "type": "paragraph", "text": CYRILLIC }],
            }),
        )
        .await;
        let path = single(&reply);
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF-"), "not a PDF: {}", path.display());
        assert!(
            converted_text(&ws, &path).await.contains(CYRILLIC),
            "the embedded font did not carry the Cyrillic text"
        );
    }

    /// A character the embedded font has no glyph for is drawn as a blank box,
    /// so the answer names it as missing from the file; a Cyrillic alphabet the
    /// font covers reports nothing missing.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn a_pdf_reports_characters_the_font_cannot_draw() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let missing = run(
            &ws,
            json!({
                "action": "create", "format": "pdf", "file_name": "cjk",
                "content": [{ "type": "paragraph", "text": "漢字" }],
            }),
        )
        .await;
        assert!(
            missing.contains("'漢'") && missing.contains("'字'"),
            "a glyph the font lacks must be named in the answer: {missing}"
        );
        // A newline is consumed by pdf-lib's own layout before the font is
        // consulted, so it is laid out as a break rather than reported as a
        // character the font cannot draw.
        let overlaid = run(
            &ws,
            json!({
                "action": "pdf_text", "file_name": "overlaid",
                "path": single(&missing).to_string_lossy(),
                "text": "first\nsecond",
            }),
        )
        .await;
        assert!(
            !overlaid.contains("missing from the file"),
            "a newline the pipeline lays out must not be reported missing: {overlaid}"
        );
        let covered = run(
            &ws,
            json!({
                "action": "create", "format": "pdf", "file_name": "cyrillic",
                "content": [{ "type": "paragraph", "text": CYRILLIC }],
            }),
        )
        .await;
        assert!(
            !covered.contains("missing from the file"),
            "a covered alphabet must not be reported missing: {covered}"
        );
    }

    /// A table may be its header alone: the request carried no `rows` for it,
    /// and all three content formats lay the headings out.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn a_headers_only_table_is_laid_out() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        for format in ["docx", "pptx", "pdf"] {
            let reply = run(
                &ws,
                json!({
                    "action": "create", "format": format,
                    "file_name": format!("header_only_{format}"),
                    "content": [{ "type": "table", "headers": ["Колонка", "Значение"] }],
                }),
            )
            .await;
            let text = converted_text(&ws, &single(&reply)).await;
            assert!(
                text.contains("Колонка") && text.contains("Значение"),
                "{format} lost a header-only table's headings: {text}"
            );
        }
    }

    /// A PDF table cell wraps over as many lines as its text needs, and a row
    /// taller than the page is broken across pages rather than drawn past the
    /// page bottom: text lost either way would be delivered as a success with
    /// part of it missing from the file.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn a_wrapping_table_cell_keeps_every_line() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        // A cell of a thousand words: many lines of the one column, and more
        // than one page of them.
        let cell = format!("{}конец", "слово ".repeat(999));
        let reply = run(
            &ws,
            json!({
                "action": "create", "format": "pdf", "file_name": "wrapping_table",
                "content": [{ "type": "table", "headers": ["Колонка"], "rows": [[cell]] }],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;
        assert!(
            text.contains("конец"),
            "the cell's last words are missing from the file: {text}"
        );
        // The read-back joins the drawn lines, so it holds one word per line of
        // a cell this long; a file that kept only a line or two would hold a
        // handful.
        let words = text.matches("слово").count();
        assert!(
            words >= 900,
            "only {words} of the cell's 999 words reached the file"
        );
    }

    /// A sample with no placeholders at all yields a copy that substituted
    /// nothing, and the answer says so instead of reporting a fill. A workbook's
    /// strings live in cells and a presentation's package is re-serialised by the
    /// template library, so both shapes must be reported honestly.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn filling_a_sample_without_placeholders_says_so() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let samples = [
            json!({
                "action": "create", "format": "xlsx", "file_name": "plain-sheet",
                "sheets": [{ "rows": [["Позиция", 1]] }],
            }),
            json!({
                "action": "create", "format": "pptx", "file_name": "plain-slides",
                "content": [{ "type": "paragraph", "text": "Без подстановок" }],
            }),
        ];
        for sample in samples {
            let created = run(&ws, sample).await;
            let sample_path = single(&created);

            let filled = run(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "template": sample_path.to_string_lossy(),
                    "values": { "name": "Имя" },
                }),
            )
            .await;
            assert!(
                filled.contains("nothing was substituted"),
                "expected the no-placeholder note: {filled}"
            );
            assert!(
                !file_paths(&filled).is_empty(),
                "the copy is still delivered: {filled}"
            );
        }
    }

    /// A placeholder named after an `Object.prototype` member is a name the
    /// caller gave no value for: reading it off the prototype would write
    /// `[object Object]` — or a function's own source — into the delivered file,
    /// which is content the request never asked for.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn a_placeholder_named_after_a_prototype_member_is_reported_missing() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let cases = [
            (
                json!({
                    "action": "create", "format": "xlsx", "file_name": "prototype-sheet",
                    "sheets": [{ "rows": [["{toString}"], ["{__proto__}"], ["Имя: {name}"]] }],
                }),
                "toString, __proto__",
            ),
            (
                json!({
                    "action": "create", "format": "pptx", "file_name": "prototype-slides",
                    "content": [{ "type": "notes", "text": "Автор: {constructor}" }],
                }),
                "constructor",
            ),
        ];
        for (sample, expected) in cases {
            let sample_path = single(&run(&ws, sample).await);
            let filled = run(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "template": sample_path.to_string_lossy(),
                    "values": { "name": "Имя" },
                }),
            )
            .await;
            assert!(
                filled.contains(&format!("left unchanged: {expected}]")),
                "the prototype's own member was not reported missing: {filled}"
            );
        }
    }

    /// Filling a sample substitutes the values, leaves (and reports) an unfilled
    /// placeholder, and does not touch any other part of the package — the
    /// sample's media included.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn fill_template_substitutes_and_leaves_the_sample_intact() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let image = ws.as_path().join("pic.png");
        std::fs::write(&image, noisy_png(8, 8)).expect("write png");

        let sample = run(
            &ws,
            json!({
                "action": "create", "format": "docx", "file_name": "sample",
                "content": [
                    { "type": "paragraph", "text": "Привет, {name}! {unfilled}" },
                    { "type": "image", "path": image.to_string_lossy() },
                ],
            }),
        )
        .await;
        let sample_path = single(&sample);

        let filled = run(
            &ws,
            json!({
                "action": "fill_template", "file_name": "filled",
                "template": sample_path.to_string_lossy(),
                "values": { "name": "Имя" },
            }),
        )
        .await;
        let filled_path = single(&filled);

        let text = converted_text(&ws, &filled_path).await;
        assert!(text.contains("Имя"), "value not substituted: {text}");
        assert!(
            text.contains("{unfilled}"),
            "the unfilled placeholder must remain: {text}"
        );
        assert!(
            filled.contains("unfilled"),
            "the answer must name the unfilled placeholder: {filled}"
        );
        assert_eq!(
            zip_names(&sample_path),
            zip_names(&filled_path),
            "filling changed the package's part set"
        );
        assert_eq!(
            media_parts(&sample_path),
            media_parts(&filled_path),
            "filling changed the sample's media"
        );
    }

    /// A real workbook keeps its strings in `xl/sharedStrings.xml` and writes
    /// only an index into the cell, so the substitution has to run over the
    /// shared table while the cell (and every other part) stays as it was. A
    /// whole-cell placeholder whose value is a number becomes a numeric cell
    /// instead of a string carrying the digits.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn fill_template_substitutes_a_shared_string_workbook() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sample = ws.as_path().join("book.xlsx");
        std::fs::write(&sample, shared_string_workbook()).expect("write fixture");

        let filled = run(
            &ws,
            json!({
                "action": "fill_template", "file_name": "book-filled",
                "template": sample.to_string_lossy(),
                "values": { "name": "Имя", "amount": 42 },
            }),
        )
        .await;
        assert!(
            !filled.contains("nothing was substituted"),
            "a shared-string placeholder is a placeholder: {filled}"
        );
        let path = single(&filled);
        assert_eq!(
            zip_names(&sample),
            zip_names(&path),
            "filling changed the package's part set"
        );
        assert_eq!(
            part_bytes(&sample, "docProps/app.xml"),
            part_bytes(&path, "docProps/app.xml"),
            "filling changed a part it does not own"
        );
        let sheet = part_text(&path, "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains("t=\"s\""),
            "the cell must keep its shared-string index: {sheet}"
        );
        let amount = sheet
            .split_once(r#"<c r="A2""#)
            .and_then(|(_, rest)| rest.split_once("</c>"))
            .map(|(cell, _)| cell)
            .expect("the sample's A2 cell must exist");
        assert!(
            amount.contains("<v>42</v>"),
            "a whole-cell number must be written as a value: {amount}"
        );
        assert!(
            !amount.contains("t=\"s\"") && !amount.contains("t=\"inlineStr\""),
            "a numeric cell must not carry a string type: {amount}"
        );
        let text = converted_text(&ws, &path).await;
        assert!(
            text.contains("Отчёт для Имя"),
            "the shared string was not substituted: {text}"
        );
    }

    /// A sample whose cells carry no `r` is filled by position, so a placeholder
    /// the reader can show is one the filler can fill; the alternative leaves it
    /// in the copy and reports that there was nothing to substitute.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn fill_template_addresses_cells_the_sample_left_unaddressed() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sample = ws.as_path().join("no-addresses.xlsx");
        std::fs::write(&sample, addressless_workbook()).expect("write fixture");

        let filled = run(
            &ws,
            json!({
                "action": "fill_template", "file_name": "no-addresses-filled",
                "template": sample.to_string_lossy(),
                "values": { "name": "Отчёт" },
            }),
        )
        .await;
        assert!(
            !filled.contains("nothing was substituted"),
            "the sample does hold a placeholder: {filled}"
        );
        let path = single(&filled);
        let sheet = part_text(&path, "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains("Отчёт"),
            "the placeholder was left in place: {sheet}"
        );
        assert!(
            sheet.contains(r#"<c r="A1""#),
            "a rewritten cell must carry the address its position names: {sheet}"
        );
        // The reader synthesises the same address for a cell without one, so the
        // filled value is what it shows.
        let text = converted_text(&ws, &path).await;
        assert!(text.contains("A1: Отчёт"), "{text}");
    }

    /// Filling a presentation substitutes both the slide text and the speaker
    /// notes and keeps the package's parts.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn fill_template_substitutes_a_presentation() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sample = run(
            &ws,
            json!({
                "action": "create", "format": "pptx", "file_name": "slides-sample",
                "content": [
                    { "type": "paragraph", "text": "Привет, {name}!" },
                    { "type": "notes", "text": "Заметка для {name}" },
                ],
            }),
        )
        .await;
        let sample_path = single(&sample);

        let filled = run(
            &ws,
            json!({
                "action": "fill_template", "file_name": "slides-filled",
                "template": sample_path.to_string_lossy(),
                "values": { "name": "Имя" },
            }),
        )
        .await;
        assert!(
            !filled.contains("nothing was substituted"),
            "the sample declares a placeholder: {filled}"
        );
        let path = single(&filled);
        assert_eq!(
            zip_names(&sample_path),
            zip_names(&path),
            "filling changed the package's part set"
        );
        let text = converted_text(&ws, &path).await;
        assert!(
            text.contains("Привет, Имя!"),
            "the slide text was not substituted: {text}"
        );
        assert!(
            text.contains("Slide 1 notes:") && text.contains("Заметка для Имя"),
            "the speaker notes were not substituted: {text}"
        );
        assert!(!text.contains("{name}"), "a placeholder was left: {text}");
    }

    /// Merging and splitting preserve each document's own text.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pdf_merge_and_split_carry_each_documents_text() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let first = pdf_with(&ws, "first", "Первый документ целиком").await;
        let second = pdf_with(&ws, "second", "Второй документ тоже").await;

        let merged = run(
            &ws,
            json!({
                "action": "pdf_merge", "file_name": "merged",
                "files": [first.to_string_lossy(), second.to_string_lossy()],
            }),
        )
        .await;
        let merged_path = single(&merged);
        let merged_text = converted_text(&ws, &merged_path).await;
        assert!(
            merged_text.contains("Первый документ"),
            "got: {merged_text}"
        );
        assert!(
            merged_text.contains("Второй документ"),
            "got: {merged_text}"
        );

        let split = run(
            &ws,
            json!({
                "action": "pdf_split", "file_name": "split",
                "path": merged_path.to_string_lossy(),
                "ranges": ["1", "2"],
            }),
        )
        .await;
        let parts = file_paths(&split);
        assert_eq!(parts.len(), 2, "one file per range: {split}");
        assert!(
            converted_text(&ws, &parts[0])
                .await
                .contains("Первый документ"),
            "first part lost its text"
        );
        assert!(
            converted_text(&ws, &parts[1])
                .await
                .contains("Второй документ"),
            "second part lost its text"
        );
    }

    /// Rotation is written to the saved bytes as a 90° page rotation — and a
    /// larger multiple of 90 is the rotation it is modulo a turn, so 450 is a
    /// 90° one.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pdf_rotate_writes_the_rotation() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = pdf_with(&ws, "flat", "Страница без поворота").await;
        let rotated = run(
            &ws,
            json!({
                "action": "pdf_rotate", "file_name": "turned",
                "path": source.to_string_lossy(), "degrees": 450,
            }),
        )
        .await;
        let bytes = std::fs::read(single(&rotated)).expect("read rotated pdf");
        let pdf = hayro::hayro_syntax::Pdf::new(bytes).expect("the produced file parses as a PDF");
        let page = pdf.pages().first().expect("one page");
        assert!(
            matches!(
                page.rotation(),
                hayro::hayro_syntax::page::Rotation::Horizontal
            ),
            "the saved page carries no 90° rotation"
        );
    }

    /// A page past the document's end is refused with a message that names the
    /// page and the document's length, not with whatever the PDF library says.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn a_page_past_the_end_is_refused_by_name() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = pdf_with(&ws, "one-page", "Единственная страница").await;

        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pdf_rotate", "file_name": "turned",
                    "path": source.to_string_lossy(), "pages": "5", "degrees": 90,
                }),
            )
            .await
            .expect_err("a page the document does not have must be refused");
        assert!(err.to_string().contains("no page 5"), "got: {err}");
        // The kit's own rejection of the request carries the `usage` token, so
        // the model can tell it apart from a mahbot-side kit fault.
        assert!(err.to_string().starts_with("usage:"), "got: {err}");

        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pdf_split", "file_name": "parts",
                    "path": source.to_string_lossy(), "ranges": ["3"],
                }),
            )
            .await
            .expect_err("a range past the document's end must be refused");
        assert!(err.to_string().contains("no page 3"), "got: {err}");
    }

    /// A PDF the library refuses to open is the caller's own file, whatever the
    /// reason, and never a product fault: a body the parser cannot read must not
    /// surface as the library's internal text under an `internal:` token, and an
    /// encrypted one must say so rather than be described as unreadable.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn an_unopenable_pdf_is_the_callers_own_input() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let broken = ws.as_path().join("broken.pdf");
        std::fs::write(&broken, b"%PDF-1.4\nnot really a pdf").expect("write a broken pdf");
        let encrypted = ws.as_path().join("encrypted.pdf");
        std::fs::write(&encrypted, encrypted_pdf()).expect("write an encrypted pdf");

        for (path, expected) in [
            (&broken, "do not form a PDF"),
            (&encrypted, "password-protected"),
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pdf_rotate", "file_name": "turned",
                        "path": path.to_string_lossy(), "pages": "all", "degrees": 90,
                    }),
                )
                .await
                .expect_err("a PDF the library cannot open must be refused");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(
                message.contains(expected),
                "{expected} is not named in: {message}"
            );
        }
    }

    /// A sample the package reader cannot open at all — an encrypted (CFB)
    /// `.docx`, which has no ZIP central directory to read — is the request's own
    /// input, so the answer names the file as a usage error rather than reporting
    /// an unexplained kit failure.
    #[tokio::test]
    async fn an_unopenable_sample_is_refused_as_a_usage_error() {
        let (_dir, ws) = workspace();
        let sample = ws.as_path().join("encrypted.docx");
        // The CFB compound-file magic an encrypted OOXML package starts with.
        let mut bytes = vec![0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1];
        bytes.resize(512, 0);
        std::fs::write(&sample, bytes).expect("write fixture");

        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "template": sample.to_string_lossy(), "values": { "name": "Имя" },
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
    }

    /// A sample's family comes from its NAME, and the name is judged by the
    /// container: a sample NAMED as an old binary format whose bytes are an OOXML
    /// package is told to be renamed, while a real `.doc`/`.xls`/`.ppt` gets the old
    /// formats' answer — read, but never filled or edited.
    #[tokio::test]
    async fn a_sample_is_answered_by_what_it_really_holds() {
        let (_dir, ws) = workspace();
        let cases = [
            (
                "report.doc",
                [crate::document::CFB_MAGIC, b"the rest of a container"].concat(),
                "an old .doc/.xls/.ppt file is read but never filled or edited",
            ),
            (
                "sheet.xls",
                b"PK\x03\x04the rest of a package".to_vec(),
                "the file's bytes are an OOXML package",
            ),
        ];
        for (name, bytes, expected) in cases {
            std::fs::write(ws.as_path().join(name), &bytes).expect("write sample");
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "fill_template", "file_name": "copy",
                        "template": name, "values": { "name": "Имя" },
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

    /// A sample the template library cannot compile — an unbalanced `{`, which
    /// is what a document a user edited by hand can end up with — is the
    /// request's own input too, and the refusal carries what makes it fixable.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: the sample, the fill and the edit voice, one case each
    async fn a_sample_that_cannot_be_compiled_is_a_usage_error() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = run(
            &ws,
            json!({
                "action": "create", "format": "docx", "file_name": "unbalanced",
                "content": [{ "type": "paragraph", "text": "Текст {name" }],
            }),
        )
        .await;
        let sample = single(&created);

        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "fill_template", "file_name": "copy",
                    "template": sample.to_string_lossy(), "values": { "name": "Имя" },
                }),
            )
            .await
            .expect_err("a sample with an unclosed tag must be refused");
        let message = err.to_string();
        assert!(message.starts_with("usage:"), "got: {message}");
        assert!(
            message.contains("unclosed"),
            "the refusal must say what is wrong with the sample: {message}"
        );

        // A package whose contents disagree with its name is named as what it
        // really is — in either direction: the library's message for the one
        // talks about its own paid modules, and a workbook filler handed a Word
        // document would report a successful fill that substituted nothing.
        let sheet = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "sheet",
                    "sheets": [{ "rows": [["Позиция", 1]] }],
                }),
            )
            .await,
        );
        let word = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "docx", "file_name": "word",
                    "content": [{ "type": "paragraph", "text": "Текст" }],
                }),
            )
            .await,
        );
        let cases = [
            (&sheet, "sheet.docx", "really a .xlsx package"),
            (&word, "word.xlsx", "really a .docx package"),
        ];
        for (source, misnamed, expected) in cases {
            let misnamed = ws.as_path().join(misnamed);
            std::fs::copy(source, &misnamed).expect("copy the package under the other name");
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "fill_template", "file_name": "copy2",
                        "template": misnamed.to_string_lossy(), "values": { "name": "Имя" },
                    }),
                )
                .await
                .expect_err("a package named as another family must be refused");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(
                message.contains("the sample is really a"),
                "a fill's sample keeps the sample wording: {message}"
            );
            assert!(
                message.contains(expected),
                "the refusal must describe the file, not the library's modules: {message}"
            );
        }

        // An edit input of another family gets that action's own voice: it is
        // told which action edits the family the package really is, rather than
        // handed the sample's rename advice, which would leave docx_edit pointed
        // at a workbook no rename makes editable.
        for (source, misnamed, action, edit, expected) in [
            (
                &sheet,
                "sheet.docx",
                "docx_edit",
                json!({ "op": "replace_text", "find": "x", "replace": "y" }),
                "use xlsx_edit",
            ),
            (
                &word,
                "word.xlsx",
                "xlsx_edit",
                json!({ "op": "clear_cell", "sheet": "S", "cell": "A1" }),
                "use docx_edit",
            ),
        ] {
            let misnamed = ws.as_path().join(misnamed);
            std::fs::copy(source, &misnamed).expect("copy the package under the other name");
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": action, "path": misnamed.to_string_lossy(),
                        "edits": [edit],
                    }),
                )
                .await
                .expect_err("an edit input of another family must be refused");
            let message = err.to_string();
            assert!(message.starts_with("usage:"), "got: {message}");
            assert!(
                message.contains("the input is really a") && message.contains(expected),
                "the refusal must point the edit at the action that fits it: {message}"
            );
        }
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
            let key = if action == "fill_template" {
                "template"
            } else {
                "path"
            };
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": action, key: sample.to_string_lossy(),
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
            let error = DocumentTool
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
                json!({ "content": [{ "type": "image", "path": "x.png", "width": { "a": 1 } }] }),
                "content[0].width",
            ),
            (
                json!({ "content": [{ "type": "image", "path": "x.png", "height": -1 }] }),
                "content[0].height",
            ),
            (
                json!({ "content": [{ "type": "image", "path": "x.png", "width": 1e30 }] }),
                "content[0].width",
            ),
        ];
        for (args, expected) in content {
            let err = resolve_content(&ws, &args, "docx")
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
        assert!(resolve_content(&ws, &good, "docx").await.is_ok());
        let good =
            json!({ "content": [{ "type": "table", "headers": null, "rows": [["a", "b"]] }] });
        assert!(resolve_content(&ws, &good, "docx").await.is_ok());
        let headers_only = json!({ "content": [{ "type": "table", "headers": ["a"] }] });
        assert!(resolve_content(&ws, &headers_only, "docx").await.is_ok());
        let good = json!({ "sheets": [{ "name": "Лист", "rows": [["a", 1, true, { "formula": "=SUM(A1:A2)" }]] }] });
        let normalized = resolve_sheets(&good).expect("a well-formed sheet");
        assert_eq!(
            normalized[0]["rows"][0][3],
            json!({ "formula": "SUM(A1:A2)" }),
            "the formula must reach the request without its leading `=`"
        );
    }

    /// An empty table set is a set of nothing to lay out, so the boundary drops
    /// it from the block the request carries instead of handing a writer a
    /// cell-less row — a construct Word refuses to open. A table left with
    /// nothing is still refused.
    #[tokio::test]
    async fn an_empty_table_set_is_normalized_to_an_absent_one() {
        let (_dir, ws) = workspace();

        let no_headers = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "headers": [], "rows": [["a"]] }] }),
            "docx",
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
        )
        .await
        .expect("a table with one cell is a table");
        assert_eq!(no_empty_row[0]["rows"], json!([["a"]]));

        let err = resolve_content(
            &ws,
            &json!({ "content": [{ "type": "table", "headers": [], "rows": [[], []] }] }),
            "docx",
        )
        .await
        .expect_err("a table of no cells must still be refused");
        assert!(
            err.to_string().contains("no rows and no headers"),
            "got: {err}"
        );
    }

    /// Every argument of an action that produces a file is checked before an
    /// output is reserved, so a call the tool refuses leaves nothing behind in
    /// `generated/` — and a file that is not the image it is named as is refused
    /// rather than embedded.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // one case per argument, all refused before anything is reserved
    async fn the_arguments_are_checked_before_an_output_is_reserved() {
        let (_dir, ws) = workspace();
        let image = ws.as_path().join("picture.png");
        std::fs::write(&image, noisy_png(4, 4)).expect("write png");
        let notes = ws.as_path().join("notes.txt");
        std::fs::write(&notes, b"not an image").expect("write notes");
        let document = ws.as_path().join("document.pdf");
        std::fs::write(&document, b"not a pdf").expect("write pdf");
        let merged = ws.as_path().join("merged.pdf");
        std::fs::write(&merged, b"not a pdf").expect("write pdf");
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
                json!({ "action": "pdf_rotate", "path": path, "degrees": 90, "pages": "0" }),
                "page 0 does not exist",
            ),
            (
                json!({ "action": "pdf_merge", "files": [merged.to_string_lossy()] }),
                "at least 2",
            ),
            (
                json!({ "action": "pdf_merge",
                        "files": (0..=MAX_MERGE_INPUTS).map(|index| format!("part{index}.pdf")).collect::<Vec<_>>() }),
                "limit 50",
            ),
            (
                json!({ "action": "pdf_merge", "files": batch }),
                "more than one merge takes",
            ),
            (
                json!({ "action": "pdf_split", "path": path,
                        "ranges": (1..=MAX_SPLIT_PARTS + 1).map(|n| n.to_string()).collect::<Vec<_>>() }),
                "limit 50",
            ),
            (
                json!({ "action": "pdf_split", "path": path, "ranges": ["3-1"] }),
                "reversed",
            ),
        ];
        for (args, expected) in calls {
            let err = DocumentTool
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

    /// Text drawn onto a page reads back through the converter.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pdf_text_overlay_reads_back() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = pdf_with(&ws, "blank", "Основа страницы документа").await;
        let overlaid = run(
            &ws,
            json!({
                "action": "pdf_text", "file_name": "stamped",
                "path": source.to_string_lossy(),
                "text": "Надпись поверх текста", "stamp": true, "size": 18, "color": "#ff0000",
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&overlaid)).await;
        assert!(text.contains("Надпись"), "overlay not readable: {text}");
    }

    /// The kit's own boundary checks are the last line for the structural shapes
    /// the tool's boundary also refuses (a structure as a value, a sheet cell, an
    /// image's bytes, a page): nothing links the two — the tool refuses before a
    /// runtime is spawned, the kit refuses before a writer is called — so every
    /// such shape is driven through BOTH here: the tool must refuse the call as
    /// the caller's own, and the kit must refuse the same shape when it is handed
    /// it directly. The value SETS and BOUNDS they share have one
    /// statement in `assets/docgen/rules.json`, read by the tool, the kit and the
    /// presentation reader; this test still covers the structural shapes and the
    /// refusal classification, so a shape widened on one side alone fails it.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: one paired case per shared rule
    async fn the_tool_and_the_kit_refuse_the_same_shapes() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sample = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "docx", "file_name": "sample",
                    "content": [{ "type": "paragraph", "text": "Имя: {name}" }],
                }),
            )
            .await,
        );
        let pdf = pdf_with(&ws, "base", "Страница").await;
        let book = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let fake = ws.as_path().join("fake.png");
        std::fs::write(&fake, b"\x89PNG\r\n\x1a\ngarbage").expect("write a fake png");
        // A frame marker with nothing after it: the walk must refuse it without
        // reading past the file's end.
        let cut = ws.as_path().join("cut.jpg");
        std::fs::write(&cut, b"\xff\xd8\xff\xc0\x00\x11\x08").expect("write a cut jpeg");
        // A whole PNG, so the width check below is reached rather than the
        // image-bytes one.
        let whole = ws.as_path().join("whole.png");
        std::fs::write(&whole, noisy_png(4, 4)).expect("write a png");
        let out = ws.as_path().join("kit-out");

        // (what, the tool's own call, the same shape as a kit request)
        let cases = [
            (
                "a structured placeholder value",
                json!({ "action": "fill_template", "template": sample.to_string_lossy(),
                        "values": { "name": { "a": 1 } } }),
                json!({ "op": "fill_template", "template": sample.to_string_lossy(), "format": "docx",
                        "values": { "name": { "a": 1 } }, "output": out.join("a.docx").to_string_lossy() }),
            ),
            (
                "a colour that is not #rrggbb",
                json!({ "action": "pdf_text", "path": pdf.to_string_lossy(), "text": "x",
                        "color": "red" }),
                json!({ "op": "pdf_text", "input": pdf.to_string_lossy(), "pages": "all", "text": "x",
                        "color": "red", "output": out.join("b.pdf").to_string_lossy() }),
            ),
            (
                "a colour whose `#` is not leading",
                json!({ "action": "pdf_text", "path": pdf.to_string_lossy(), "text": "x",
                        "color": "ff0000#" }),
                json!({ "op": "pdf_text", "input": pdf.to_string_lossy(), "pages": "all", "text": "x",
                        "color": "ff0000#", "output": out.join("b2.pdf").to_string_lossy() }),
            ),
            (
                "a sheet cell that is a structure",
                json!({ "action": "create", "format": "xlsx", "sheets": [{ "rows": [[{ "a": 1 }]] }] }),
                json!({ "op": "create", "format": "xlsx", "sheets": [{ "rows": [[{ "a": 1 }]] }],
                        "output": out.join("c.xlsx").to_string_lossy() }),
            ),
            (
                "a sheet cell that is null",
                json!({ "action": "create", "format": "xlsx", "sheets": [{ "rows": [[null]] }] }),
                json!({ "op": "create", "format": "xlsx", "sheets": [{ "rows": [[null]] }],
                        "output": out.join("c2.xlsx").to_string_lossy() }),
            ),
            (
                "a formula that is only the `=`",
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "rows": [[{ "formula": "=" }]] }] }),
                json!({ "op": "create", "format": "xlsx",
                        "sheets": [{ "rows": [[{ "formula": "=" }]] }],
                        "output": out.join("d.xlsx").to_string_lossy() }),
            ),
            (
                "a formula with a second `=`",
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "rows": [[{ "formula": "==SUM(A1:A2)" }]] }] }),
                json!({ "op": "create", "format": "xlsx",
                        "sheets": [{ "rows": [[{ "formula": "==SUM(A1:A2)" }]] }],
                        "output": out.join("d2.xlsx").to_string_lossy() }),
            ),
            (
                "two sheets sharing a name ignoring case",
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "name": "Trips", "rows": [["a"]] },
                                   { "name": "trips", "rows": [["b"]] }] }),
                json!({ "op": "create", "format": "xlsx",
                        "sheets": [{ "name": "Trips", "rows": [["a"]] },
                                   { "name": "trips", "rows": [["b"]] }],
                        "output": out.join("n.xlsx").to_string_lossy() }),
            ),
            (
                "a sheet name holding a forbidden character",
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "name": "bad:name", "rows": [["a"]] }] }),
                json!({ "op": "create", "format": "xlsx",
                        "sheets": [{ "name": "bad:name", "rows": [["a"]] }],
                        "output": out.join("n2.xlsx").to_string_lossy() }),
            ),
            (
                "image bytes that are not a whole PNG",
                json!({ "action": "create", "format": "docx",
                        "content": [{ "type": "image", "path": fake.to_string_lossy() }] }),
                json!({ "op": "create", "format": "docx",
                        "content": [{ "type": "image", "path": fake.to_string_lossy() }],
                        "output": out.join("e.docx").to_string_lossy() }),
            ),
            (
                "a JPEG cut off after its frame marker",
                json!({ "action": "create", "format": "pptx",
                        "content": [{ "type": "image", "path": cut.to_string_lossy() }] }),
                json!({ "op": "create", "format": "pptx",
                        "content": [{ "type": "image", "path": cut.to_string_lossy() }],
                        "output": out.join("f.pptx").to_string_lossy() }),
            ),
            (
                "a page the document does not have",
                json!({ "action": "pdf_rotate", "path": pdf.to_string_lossy(), "pages": "9",
                        "degrees": 90 }),
                json!({ "op": "pdf_rotate", "input": pdf.to_string_lossy(), "pages": [9],
                        "degrees": 90, "output": out.join("g.pdf").to_string_lossy() }),
            ),
            (
                "a heading level outside 1-3",
                json!({ "action": "create", "format": "docx",
                        "content": [{ "type": "heading", "level": 9, "text": "x" }] }),
                json!({ "op": "create", "format": "docx",
                        "content": [{ "type": "heading", "level": 9, "text": "x" }],
                        "output": out.join("h.docx").to_string_lossy() }),
            ),
            (
                "a table cell that is a structure",
                json!({ "action": "create", "format": "docx",
                        "content": [{ "type": "table", "rows": [[{ "a": 1 }]] }] }),
                json!({ "op": "create", "format": "docx",
                        "content": [{ "type": "table", "rows": [[{ "a": 1 }]] }],
                        "output": out.join("i.docx").to_string_lossy() }),
            ),
            (
                "a bullet that is a structure",
                json!({ "action": "create", "format": "pptx",
                        "content": [{ "type": "list", "items": [{ "a": 1 }] }] }),
                json!({ "op": "create", "format": "pptx",
                        "content": [{ "type": "list", "items": [{ "a": 1 }] }],
                        "output": out.join("j.pptx").to_string_lossy() }),
            ),
            (
                "a text block that is not text",
                json!({ "action": "create", "format": "pdf",
                        "content": [{ "type": "paragraph", "text": 7 }] }),
                json!({ "op": "create", "format": "pdf",
                        "content": [{ "type": "paragraph", "text": 7 }],
                        "output": out.join("k.pdf").to_string_lossy() }),
            ),
            (
                "an image side that is not a number",
                json!({ "action": "create", "format": "docx",
                        "content": [{ "type": "image", "path": whole.to_string_lossy(), "width": "wide" }] }),
                json!({ "op": "create", "format": "docx",
                        "content": [{ "type": "image", "path": whole.to_string_lossy(), "width": "wide" }],
                        "output": out.join("l.docx").to_string_lossy() }),
            ),
            (
                "a structured form value",
                json!({ "action": "pdf_form_fill", "path": pdf.to_string_lossy(),
                        "values": { "field": { "a": 1 } } }),
                json!({ "op": "pdf_form_fill", "input": pdf.to_string_lossy(),
                        "values": { "field": { "a": 1 } },
                        "output": out.join("m.pdf").to_string_lossy() }),
            ),
            // A sheet name past the shared limit, and one that is not text: the
            // tool and the kit both refuse them as the caller's own shape.
            (
                "a sheet name that is not text",
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "name": 5, "rows": [["a"]] }] }),
                json!({ "op": "create", "format": "xlsx",
                        "sheets": [{ "name": 5, "rows": [["a"]] }],
                        "output": out.join("n.xlsx").to_string_lossy() }),
            ),
            (
                "a sheet name past the shared limit",
                json!({ "action": "create", "format": "xlsx",
                        "sheets": [{ "name": "a".repeat(RULES.sheet_name_max + 1), "rows": [["a"]] }] }),
                json!({ "op": "create", "format": "xlsx",
                        "sheets": [{ "name": "a".repeat(RULES.sheet_name_max + 1), "rows": [["a"]] }],
                        "output": out.join("o.xlsx").to_string_lossy() }),
            ),
            // An alignment word the closed set does not hold: both sides refuse it
            // rather than writing a value no reader knows.
            (
                "an xlsx format_cells alignment the writer has not",
                json!({ "action": "xlsx_edit", "path": book.to_string_lossy(),
                        "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1",
                                    "align": "middle" }] }),
                json!({ "op": "xlsx_edit", "input": book.to_string_lossy(),
                        "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1",
                                    "align": "middle" }],
                        "output": out.join("p.xlsx").to_string_lossy() }),
            ),
        ];
        for (what, tool_args, kit_request) in cases {
            let err = DocumentTool
                .execute(&ws, tool_args)
                .await
                .expect_err("a call the tool cannot carry out");
            assert!(
                err.to_string().starts_with("usage:"),
                "{what}: the tool blamed itself: {err}"
            );
            let Err(err) = crate::docgen::run(kit_request).await else {
                panic!("{what}: the kit accepted a shape the tool refuses");
            };
            assert!(
                err.to_string().starts_with("usage:"),
                "{what}: the kit blamed itself: {err}"
            );
        }
    }

    /// A placed image survives as an extractable embedded image.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pdf_image_insert_round_trips() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = pdf_with(&ws, "base", "Страница").await;
        let image = ws.as_path().join("mark.png");
        std::fs::write(&image, noisy_png(40, 40)).expect("write png");

        let illustrated = run(
            &ws,
            json!({
                "action": "pdf_image", "file_name": "illustrated",
                "path": source.to_string_lossy(), "image": image.to_string_lossy(),
            }),
        )
        .await;
        let path = single(&illustrated);
        let out_dir = ws.as_path().join("extracted");
        match convert_document_file(&path, &file_name_of(&path), &out_dir).await {
            DocOutcome::Text { images, .. } => {
                assert!(!images.is_empty(), "no embedded image was extracted");
            }
            other => panic!("unexpected conversion outcome: {other:?}"),
        }
    }

    /// An image whose bytes are not a whole PNG or JPEG is refused for every
    /// arm: the docx and pptx writers embed what they are handed without
    /// decoding it, so a mislabelled file would otherwise reach a document the
    /// caller receives as a success while the PDF arm refused the same file.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn an_image_that_is_not_whole_is_refused_for_every_arm() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let fake = ws.as_path().join("fake.png");
        std::fs::write(&fake, b"\x89PNG\r\n\x1a\ngarbage").expect("write a fake png");
        let mislabelled = ws.as_path().join("claim.jpg");
        std::fs::write(&mislabelled, noisy_png(4, 4)).expect("write a PNG under a .jpg name");
        let source = pdf_with(&ws, "base", "Страница").await;
        let generated = ws.as_path().join(GENERATED_DIR);
        let before = std::fs::read_dir(&generated).map_or(0, std::iter::Iterator::count);

        let calls = [
            json!({ "action": "create", "format": "docx",
                    "content": [{ "type": "image", "path": fake.to_string_lossy() }] }),
            json!({ "action": "create", "format": "pptx",
                    "content": [{ "type": "image", "path": mislabelled.to_string_lossy() }] }),
            json!({ "action": "pdf_image", "path": source.to_string_lossy(),
                    "image": fake.to_string_lossy() }),
        ];
        for args in calls {
            let err = DocumentTool
                .execute(&ws, args)
                .await
                .expect_err("an image the writer cannot embed whole");
            assert!(
                err.to_string().contains("are not a whole PNG image")
                    || err.to_string().contains("are not a whole JPEG image"),
                "expected the refusal to name the bytes rather than the file: {err}"
            );
        }
        assert_eq!(
            std::fs::read_dir(&generated).map_or(0, std::iter::Iterator::count),
            before,
            "a refused image left an output behind"
        );
    }

    /// Filling a form field with a Cyrillic value and flattening it makes the
    /// value part of the page, where the converter reads it back.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pdf_form_fill_reads_back() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let form = ws.as_path().join("form.pdf");
        std::fs::write(&form, fillable_pdf()).expect("write form fixture");

        let filled = run(
            &ws,
            json!({
                "action": "pdf_form_fill", "file_name": "form-filled",
                "path": form.to_string_lossy(),
                "values": { "name": "Имя пользователя документа" }, "flatten": true,
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&filled)).await;
        assert!(text.contains("Имя"), "form value not readable: {text}");
    }

    /// A workspace-only tool refuses a path outside the workspace.
    #[tokio::test]
    async fn refuses_a_template_outside_the_workspace() {
        let (_dir, ws) = workspace();
        let outside = tempfile::NamedTempFile::new().expect("temp file");
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "fill_template",
                    "template": outside.path().to_string_lossy(),
                    "values": {},
                }),
            )
            .await
            .expect_err("a template outside the workspace must be refused");
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

    /// The body of the docx editing fixture: an untouched run whose own
    /// formatting must survive, an insert anchor, an editable paragraph, a
    /// `w:fldChar`-based field, a `w:fldSimple`-based field, a paragraph to
    /// remove or to remove whole, and a drawing that names the embedded image.
    const DOCX_EDIT_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t>UNTOUCHED-FORMAT</w:t></w:r></w:p>
<w:p><w:r><w:t>KEEP-FORMAT</w:t></w:r></w:p>
<w:p><w:r><w:t>Replace me now</w:t></w:r></w:p>
<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> DATE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>CACHED-DATE</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
<w:p><w:fldSimple w:instr=" PAGE "><w:r><w:t>CACHED-PAGE</w:t></w:r></w:fldSimple></w:p>
<w:p><w:r><w:t>Remove me too</w:t></w:r></w:p>
<w:p><w:r><w:t>PARA-TO-REMOVE</w:t></w:r></w:p>
<w:p><w:r><w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"/></a:graphic></wp:inline></w:drawing></w:r></w:p>
<w:sectPr/>
</w:body></w:document>"#;

    const DOCX_EDIT_HEADER: &str = r#"<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>HEADER-TEXT</w:t></w:r></w:p></w:hdr>"#;
    const DOCX_EDIT_FOOTER: &str = r#"<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>FOOTER-TEXT</w:t></w:r></w:p></w:ftr>"#;

    /// Minimal extra parts for the editing fixtures: a chart, a pivot cache, a
    /// table, a comment + its VML drawing, a macro and a timing part — every one
    /// an edit must carry through untouched.
    const EDIT_CHART: &str = r#"<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:chart/></c:chartSpace>"#;
    const EDIT_PIVOT: &str = r#"<pivotCacheDefinition xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>"#;
    const EDIT_TABLE: &str = r#"<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><name>Table1</name></table>"#;
    const EDIT_COMMENTS: &str = r#"<comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><authors><author>A</author></authors></comments>"#;
    const EDIT_VML: &str =
        r#"<xml xmlns:v="urn:schemas-microsoft-com:vml"><v:shape id="s"/></xml>"#;
    const EDIT_TIMING: &str =
        r#"<timing xmlns="http://schemas.openxmlformats.org/presentationml/2006/main"/>"#;

    /// The generated directory's entry count: how a test proves a refused call
    /// left no output behind.
    fn generated_count(ws: &Workspace) -> usize {
        std::fs::read_dir(ws.as_path().join(GENERATED_DIR)).map_or(0, std::iter::Iterator::count)
    }

    /// The three edit actions address a file's own family: a package of another
    /// family, a PDF, and an old `.doc`/`.xls`/`.ppt` are refused by name, the
    /// legacy one with the hint that it is read but never edited — all before a
    /// runtime is spawned.
    #[tokio::test]
    async fn editing_refuses_an_input_of_another_family() {
        let (_dir, ws) = workspace();
        let legacy = "an old .doc/.xls/.ppt file is read but never edited";
        let cases = [
            ("report.doc", "docx_edit", legacy),
            ("sheet.xls", "xlsx_edit", legacy),
            ("deck.ppt", "pptx_edit", legacy),
            ("paper.pdf", "docx_edit", "edits only a .docx/.docm file"),
            ("paper.pdf", "xlsx_edit", "edits only a .xlsx/.xlsm file"),
            ("paper.pdf", "pptx_edit", "edits only a .pptx/.pptm file"),
            ("report.docx", "xlsx_edit", "edits only a .xlsx/.xlsm file"),
            ("sheet.xlsx", "pptx_edit", "edits only a .pptx/.pptm file"),
        ];
        for (name, action, expected) in cases {
            std::fs::write(ws.as_path().join(name), b"fixture").expect("write input");
            let err = DocumentTool
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

        // The container decides which answer a `.doc`-shaped name gets: a file
        // whose bytes are an OOXML package is not an old format — the read path
        // reads it as the package it holds — while a real CFB container is.
        std::fs::write(
            ws.as_path().join("package.doc"),
            b"PK\x03\x04the-rest-of-a-package",
        )
        .expect("write input");
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": "package.doc",
                    "edits": [{ "op": "replace_text", "find": "x", "replace": "y" }],
                }),
            )
            .await
            .expect_err("a package under an old-format name is not an old format");
        assert!(
            err.to_string()
                .contains("the file's bytes are an OOXML package"),
            "got: {err}"
        );

        std::fs::write(
            ws.as_path().join("container.doc"),
            [crate::document::CFB_MAGIC, b"the rest of a container"].concat(),
        )
        .expect("write input");
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": "container.doc",
                    "edits": [{ "op": "replace_text", "find": "x", "replace": "y" }],
                }),
            )
            .await
            .expect_err("a legacy container under an old-format name is an old format");
        assert!(err.to_string().contains(legacy), "got: {err}");
        assert_eq!(generated_count(&ws), 0, "a refused call left an output");
    }

    /// Every edit shape the kit would fail on is refused at the boundary,
    /// naming the offending edit and field, and before an output is reserved.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // reason: one refused edit shape per op and field
    async fn edit_shapes_are_checked_before_an_output_is_reserved() {
        let (_dir, ws) = workspace();
        for name in ["doc.docx", "book.xlsx", "deck.pptx"] {
            std::fs::write(ws.as_path().join(name), b"fixture").expect("write input");
        }
        let too_many: Vec<Value> = (0..=RULES.edits_max)
            .map(|_| json!({ "op": "replace_text", "find": "x", "replace": "y" }))
            .collect();
        let long_style = "a".repeat(RULES.paragraph_style_max + 1);
        let style_cap = format!("must be at most {} characters", RULES.paragraph_style_max);
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
                                    "width": 10 }] }),
                "column must be column letters",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "column": "b",
                                    "width": 10 }] }),
                "column must be column letters",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "row": 99_999_999,
                                    "height": 20 }] }),
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
                                    "width": 300 }] }),
                "width must be",
            ),
            (
                json!({ "action": "xlsx_edit", "path": "book.xlsx",
                        "edits": [{ "op": "format_cells", "sheet": "Values", "row": 3,
                                    "height": 500 }] }),
                "height must be",
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
            let err = DocumentTool
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

    /// A docx edit changes only the body part: every other part keeps its exact
    /// bytes, a field's cached value and an untouched run's formatting survive,
    /// the new text reads back, and one file is delivered.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_keeps_every_part_it_does_not_touch() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "docx", "file_name": "base",
                    "content": [{ "type": "paragraph", "text": "base" }],
                }),
            )
            .await,
        );
        let png = noisy_png(4, 4);
        let package = with_parts(
            &std::fs::read(&created).expect("read base package"),
            &[
                ("word/document.xml", DOCX_EDIT_BODY.as_bytes()),
                ("word/header1.xml", DOCX_EDIT_HEADER.as_bytes()),
                ("word/footer1.xml", DOCX_EDIT_FOOTER.as_bytes()),
                ("word/charts/chart1.xml", EDIT_CHART.as_bytes()),
                ("word/media/image1.png", png.as_slice()),
            ],
        );
        let source = write_fixture(&ws, "source.docx", &package);

        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "replace_text", "find": "Replace me now", "replace": "REPLACED-TEXT" },
                    { "op": "insert_text", "find": "KEEP-FORMAT", "insert": " INSERTED",
                      "position": "after" },
                    { "op": "remove_text", "find": "Remove me too" },
                    { "op": "format_text", "find": "REPLACED-TEXT", "bold": true, "size": 24 },
                    { "op": "add_paragraph", "text": "ADDED PARAGRAPH", "after": "REPLACED-TEXT" },
                    { "op": "remove_paragraph", "find": "PARA-TO-REMOVE" },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        assert_eq!(file_paths(&reply).len(), 1, "one file marker: {reply}");

        assert_eq!(
            zip_names(&source),
            zip_names(&output),
            "a part was added or dropped"
        );
        for name in zip_names(&source) {
            if name == "word/document.xml" {
                continue;
            }
            assert_eq!(
                part_bytes(&source, &name),
                part_bytes(&output, &name),
                "{name} changed"
            );
        }

        let text = converted_text(&ws, &output).await;
        for expected in ["REPLACED-TEXT", "ADDED PARAGRAPH", "KEEP-FORMAT INSERTED"] {
            assert!(text.contains(expected), "{expected} missing from: {text}");
        }
        let body = part_text(&output, "word/document.xml");
        assert!(
            body.contains("CACHED-DATE") && body.contains("CACHED-PAGE"),
            "a field's cached value changed: {body}"
        );
        assert!(
            body.contains(r#"<w:u w:val="single"/>"#),
            "an untouched run lost its formatting: {body}"
        );
        assert!(
            body.contains("<w:b/>") && body.contains(r#"<w:sz w:val="48"/>"#),
            "format_text wrote no properties: {body}"
        );
    }

    /// Fixture bodies for the docx edge cases below: a body whose paragraph-level
    /// section break must not be mistaken for the body's own, a field whose begin
    /// and end sit in different paragraphs, a run whose own `<w:rPr>` carries a
    /// nested `<w:rPrChange>`, and a table cell holding one paragraph.
    const MULTISECTION_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>First section text</w:t></w:r></w:p><w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/></w:sectPr></w:pPr><w:r><w:t>Second section text</w:t></w:r></w:p><w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr></w:body></w:document>"#;
    const SPANNING_FIELD_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>before</w:t></w:r><w:r><w:fldChar w:fldCharType="begin"/></w:r></w:p><w:p><w:r><w:t>SPANNED-CACHE</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p><w:p><w:r><w:t>after</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
    /// A field spanning two paragraphs with real body text on BOTH sides of it:
    /// one `target` before the field's `begin` and one after its `end`, with the
    /// field's cached value between them. An edit that re-located runs without
    /// the field regions would read the cached value as body text and apply the
    /// offsets found over the shorter text to the longer one.
    const SPANNING_FIELD_TWO_TARGETS_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>head target</w:t></w:r><w:r><w:fldChar w:fldCharType="begin"/></w:r></w:p><w:p><w:r><w:t>SPANNED-CACHE</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r><w:r><w:t>tail target</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
    const NESTED_RPR_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:rPr><w:b/><w:rPrChange w:id="1" w:author="A"><w:rPr><w:b/></w:rPr></w:rPrChange></w:rPr><w:t>TARGET</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
    /// A run whose own `<w:rPr>` holds nothing and whose tracked revision holds
    /// the property: reading the revision's contents as the run's own would make
    /// a request that asked for bold change nothing at all.
    const REVISION_ONLY_RPR_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:rPr><w:rPrChange w:id="1" w:author="A"><w:rPr><w:b/><w:sz w:val="20"/></w:rPr></w:rPrChange></w:rPr><w:t>TARGET</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
    const TABLE_CELL_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:tcPr/><w:p><w:r><w:t>CELL-ONLY</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>outside</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;

    /// A table cell whose one paragraph draws a VML text box; the box holds a
    /// single paragraph, so removing it would empty the box. The cell keeps its
    /// own paragraph, the one that draws the box.
    const TEXT_BOX_ONLY_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:tcPr/><w:p><w:r><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml"><v:textbox><w:txbxContent><w:p><w:r><w:t>TXBX-ONLY</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>outside</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;

    /// The same cell with a text box holding two paragraphs: removing one leaves
    /// the box with the other.
    const TEXT_BOX_TWO_PARAGRAPHS_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:tcPr/><w:p><w:r><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml"><v:textbox><w:txbxContent><w:p><w:r><w:t>TXBX-KEEP</w:t></w:r></w:p><w:p><w:r><w:t>TXBX-GO</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>outside</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;

    /// A table cell holding two paragraphs of its own with a table nested between
    /// them: the nested table's own cells' paragraphs are that table's, not this
    /// cell's, so a cell keeps a paragraph even when every paragraph of it that
    /// holds the token goes — counting the nested ones would let the removal empty
    /// the cell and write a part Word calls damaged.
    const NESTED_TABLE_CELL_BODY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:tcPr/><w:p><w:r><w:t>CELL-GO</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:tcPr/><w:p><w:r><w:t>INNER</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>CELL-KEEP</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>outside</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;

    /// A freshly created one-paragraph docx package: the starting point every
    /// fixture builder below swaps parts of.
    async fn created_docx(ws: &Workspace, name: &str) -> PathBuf {
        single(
            &run(
                ws,
                json!({
                    "action": "create", "format": "docx", "file_name": name,
                    "content": [{ "type": "paragraph", "text": "seed" }],
                }),
            )
            .await,
        )
    }

    /// `created` with its own body part swapped for `body` and every `(part,
    /// bytes)` written beside it — a name the package already holds replaced, one
    /// it does not added — with each part named in `dropped` left out. Written back
    /// as `{name}.docx`.
    fn assembled_docx_fixture(
        ws: &Workspace,
        name: &str,
        created: &Path,
        body: &str,
        parts: &[(&str, &[u8])],
        dropped: &[&str],
    ) -> PathBuf {
        let mut named = parts.to_vec();
        named.push(("word/document.xml", body.as_bytes()));
        let package = repacked(
            &std::fs::read(created).expect("read base package"),
            dropped,
            &named,
        );
        write_fixture(ws, &format!("{name}.docx"), &package)
    }

    /// A created document with one body part swapped for `body`.
    async fn docx_body_fixture(ws: &Workspace, name: &str, body: &str) -> PathBuf {
        docx_part_fixture(ws, name, body, &[], &[]).await
    }

    /// [`docx_body_fixture`] with the package's own parts adjusted: each `(name,
    /// text)` replaces the part the created package holds or adds one it does not,
    /// and a part named in `dropped` is taken out. A document whose styles or list
    /// definitions are not the ones `create` writes — or one with no styles part at
    /// all — is what a style name, a list and its refusal have to be read against.
    async fn docx_part_fixture(
        ws: &Workspace,
        name: &str,
        body: &str,
        parts: &[(&str, &str)],
        dropped: &[&str],
    ) -> PathBuf {
        let created = created_docx(ws, name).await;
        let named: Vec<(&str, &[u8])> = parts
            .iter()
            .map(|(part, text)| (*part, text.as_bytes()))
            .collect();
        assembled_docx_fixture(ws, name, &created, body, &named, dropped)
    }

    /// [`docx_part_fixture`] with the created package's numbering part taken out
    /// and no longer named anywhere: `[Content_Types].xml` and
    /// `word/_rels/document.xml.rels` are read from the created package's own
    /// parts — the fixture must not carry a copy of a part list the writer owns —
    /// and the ONE element that names the part is dropped from each, an element of
    /// either part being a piece up to its own `>` since neither holds text. This is
    /// the state a document whose numbering part a call has to CREATE is read from.
    async fn docx_unnumbered_fixture(ws: &Workspace, name: &str, body: &str) -> PathBuf {
        let created = created_docx(ws, name).await;
        let unnumbered = |part: &str, marker: &str| {
            part.split_inclusive('>')
                .filter(|element| !element.contains(marker))
                .collect::<String>()
        };
        let content_types = unnumbered(
            &part_text(&created, "[Content_Types].xml"),
            "/word/numbering.xml",
        );
        let rels = unnumbered(
            &part_text(&created, "word/_rels/document.xml.rels"),
            "/numbering",
        );
        assembled_docx_fixture(
            ws,
            name,
            &created,
            body,
            &[
                ("[Content_Types].xml", content_types.as_bytes()),
                ("word/_rels/document.xml.rels", rels.as_bytes()),
            ],
            &["word/numbering.xml"],
        )
    }

    /// The styles part the paragraph tests resolve a name against: the default
    /// `Normal`, a paragraph style whose NAME is Russian and whose id is not, and a
    /// character style — the three answers a `style` can get.
    const FIRM_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Firm"><w:name w:val="Фирменный"/></w:style><w:style w:type="character" w:styleId="Red"><w:name w:val="Red ink"/></w:style></w:styles>"#;

    /// The settings part of a document whose tracked changes are turned on, under
    /// the element the schema — and Word, LibreOffice and the docx library this kit
    /// writes with — really states it in.
    const TRACKING_SETTINGS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:trackRevisions/></w:settings>"#;

    /// A styles part whose own paragraph style numbers the paragraphs that take it,
    /// and one deriving from it by `basedOn`: the numbering a paragraph stating no
    /// `<w:numPr>` of its own still draws, which `list: false` has to cancel rather
    /// than drop. A third style states only the cancellation `<w:numId w:val="0"/>`,
    /// numbering nothing, so `list: false` on a paragraph taking it writes nothing
    /// and reports nothing. A fourth records the cancellation in a `<w:pPrChange>`
    /// while stating no properties of its own, taking its list from `basedOn`: a
    /// recorded change is history, so the walk continues to the list it really
    /// draws and has to cancel that.
    const NUMBERED_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Bulleted"><w:name w:val="Bulleted"/><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="7"/></w:numPr></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Derived"><w:name w:val="Derived"/><w:basedOn w:val="Bulleted"/></w:style><w:style w:type="paragraph" w:styleId="Recorded"><w:name w:val="Recorded"/><w:basedOn w:val="Bulleted"/><w:pPr><w:pPrChange w:id="9" w:author="A" w:date="2024-01-01T00:00:00Z"><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:pPrChange></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Cancelled"><w:name w:val="Cancelled"/><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:style></w:styles>"#;

    /// A styles part whose `<w:docDefaults>` state a numbering every paragraph
    /// draws unless it says otherwise, and one style that already states the
    /// cancellation: `list: false` on a plain paragraph has to cancel the default
    /// numbering, while the style cancels it for its own paragraphs, so the walk
    /// stops there and nothing is written for them. A second style derives from one
    /// the document does not define: that `basedOn` is ignored (ECMA-376 17.7.2.4),
    /// so the style is its own root and only the defaults apply to it — the same
    /// numbering a paragraph naming no style the document has draws.
    const DEFAULT_NUMBERED_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:pPrDefault><w:pPr><w:numPr><w:numId w:val="3"/></w:numPr></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Vanishing"><w:name w:val="Vanishing"/><w:basedOn w:val="Vanished"/></w:style><w:style w:type="paragraph" w:styleId="Off"><w:name w:val="Off"/><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:style></w:styles>"#;

    /// A styles part whose only statement about numbering sits inside a
    /// `<w:pPrChange>` of its `<w:docDefaults>`: the document's live defaults number
    /// nothing, so a `list: false` on a paragraph taking them leaves it as it is
    /// instead of writing a cancellation for a list that was never there.
    const RECORDED_DEFAULT_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:pPrDefault><w:pPr><w:pPrChange w:id="5" w:author="A" w:date="2024-01-01T00:00:00Z"><w:pPr><w:numPr><w:numId w:val="3"/></w:numPr></w:pPr></w:pPrChange></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/></w:style></w:styles>"#;

    /// A styles part whose default paragraph style numbers the paragraphs that take
    /// it: the rung of the chain a paragraph naming no style of its own — or one the
    /// document does not define, or one deriving from it — renders with. Nothing
    /// flags the style as the default, so it is the one a reader falls back to by its
    /// built-in name, and its numbering is what a `list: false` has to cancel for all
    /// three of them. A style that states nothing of its own draws no list — a
    /// `basedOn` is what carries another style's numbering, not the fallback — so a
    /// paragraph taking it keeps its bytes, as does one whose style states the
    /// cancellation.
    const NORMAL_NUMBERED_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Body"><w:name w:val="Normal"/><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="5"/></w:numPr></w:pPr></w:style><w:style w:type="paragraph" w:styleId="OnNormal"><w:name w:val="On Normal"/><w:basedOn w:val="Body"/></w:style><w:style w:type="paragraph" w:styleId="Plain"><w:name w:val="Plain"/></w:style><w:style w:type="paragraph" w:styleId="Off"><w:name w:val="Off"/><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:style></w:styles>"#;

    /// A styles part whose rungs above a style disagree: the `<w:docDefaults>` number
    /// every paragraph no style does, while the default paragraph style states the
    /// cancellation. A style deriving from one the document does not define ignores
    /// that `basedOn` and is its own root (ECMA-376 17.7.2.4), so the default style's
    /// statement never reaches it — only the defaults do — and its paragraphs still
    /// draw the default numbering, which `list: false` has to cancel; one deriving
    /// from the default style really does draw none.
    const CANCELLED_DEFAULT_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:pPrDefault><w:pPr><w:numPr><w:numId w:val="3"/></w:numPr></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Vanishing"><w:name w:val="Vanishing"/><w:basedOn w:val="Vanished"/></w:style><w:style w:type="paragraph" w:styleId="OnNormal"><w:name w:val="On Normal"/><w:basedOn w:val="Normal"/></w:style></w:styles>"#;

    /// A styles part whose default paragraph style states no `<w:styleId>` at all:
    /// the style a reader falls back to is the ENTRY the document flags, not an id, so
    /// the numbering it states reaches a paragraph naming no style — and one naming a
    /// style the document does not define — which `list: false` has to cancel. The
    /// second style states nothing and derives from nothing: it is its own root, so no
    /// numbering reaches it and it keeps its bytes.
    const UNIDENTIFIED_DEFAULT_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1"><w:name w:val="Normal"/><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="6"/></w:numPr></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Else"><w:name w:val="Else"/></w:style></w:styles>"#;

    /// A numbering part holding ONE named definition — a numbered list at level 0
    /// and nothing else — so a bullet list has none to use and one has to be
    /// written. A second `<w:num>` states no `w:numId` at all: it names no id a
    /// call may reuse, so reading one would take an id the element never stated.
    /// The first one states the reserved 0 — the cancellation idiom's own id, which
    /// names no list at all: reusing it would take the paragraph's list away where
    /// the call meant to apply one.
    const NUMBERED_ONLY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/></w:lvl></w:abstractNum><w:num w:numId="0"><w:abstractNumId w:val="0"/></w:num><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num><w:num><w:abstractNumId w:val="0"/></w:num></w:numbering>"#;

    /// A numbering part whose one `<w:num>` carries a `<w:lvlOverride>` at level
    /// 0 with a `<w:lvl>` of its own: the abstract states `decimal` there, while
    /// the override makes numId 1 render a bullet — so the same id is a number
    /// list's and a bullet list's at once, and a call reusing it has to read the
    /// kind it RENDERS at the level it names rather than the abstract's format.
    const OVERRIDDEN_NUMBERING: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/><w:lvlOverride w:ilvl="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="●"/><w:lvlJc w:val="left"/></w:lvl></w:lvlOverride></w:num></w:numbering>"#;

    /// The document part a docx fixture is built from: `inner` is the body's own
    /// content, and the part a test writes is what an edit really reads.
    fn docx_document(inner: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{inner}<w:sectPr/></w:body></w:document>"#
        )
    }

    /// `add_paragraph` with no anchor appends before the body's OWN `<w:sectPr>`,
    /// not before the first `<w:sectPr>` anywhere in the body — a paragraph-level
    /// section break sits inside `<w:pPr>`, and appending there would nest a
    /// `<w:p>` inside it, which ECMA-376 refuses.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_add_paragraph_appends_before_the_body_section() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(&ws, "sections", MULTISECTION_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_paragraph", "text": "APPENDED" }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            !body.contains("<w:pPr><w:p"),
            "a paragraph was nested inside <w:pPr>: {body}"
        );
        assert!(
            body.contains(
                r#"APPENDED</w:t></w:r></w:p><w:sectPr><w:pgSz w:w="11906" w:h="16838"/>"#
            ),
            "the paragraph must land before the body-level sectPr: {body}"
        );
        assert!(
            body.contains("Second section text"),
            "the second section's text was lost: {body}"
        );
    }

    /// An `add_paragraph` `after` anchors after the FIRST paragraph whose text
    /// holds it — the rule the presentation's `add_paragraph` follows too — so a
    /// fragment that matches several paragraphs always lands the new one in the
    /// same place rather than after the last match.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_add_paragraph_anchors_after_the_first_match() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let body = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>first DUP-ANCHOR one</w:t></w:r></w:p><w:p><w:r><w:t>second DUP-ANCHOR two</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
        let source = docx_body_fixture(&ws, "anchors", body).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_paragraph", "text": "INSERTED", "after": "DUP-ANCHOR" }],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;
        let first = text
            .find("first DUP-ANCHOR one")
            .expect("the first paragraph");
        let inserted = text.find("INSERTED").expect("the new paragraph");
        let second = text
            .find("second DUP-ANCHOR two")
            .expect("the second paragraph");
        assert!(
            first < inserted && inserted < second,
            "the anchor must be the first match: {text}"
        );
    }

    /// A field whose begin and end sit in different paragraphs is still a field:
    /// its cached result is not body text, and a find on it is refused.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_refuses_a_find_inside_a_field_spanning_paragraphs() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(&ws, "spanning-field", SPANNING_FIELD_BODY).await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "replace_text", "find": "SPANNED-CACHE", "replace": "X" }],
                }),
            )
            .await
            .expect_err("a field's cached result must not be searched");
        assert!(
            err.to_string().contains("not in the document's body"),
            "got: {err}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// Two occurrences around a field whose `begin` and `end` sit in different
    /// paragraphs are both rewritten, and the field's cached value between them
    /// is not: the offsets come from the same field-region-excluded text the
    /// occurrence was found in, so neither the replacement nor the cached result
    /// is corrupted.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_replaces_around_a_field_spanning_paragraphs() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source =
            docx_body_fixture(&ws, "spanning-targets", SPANNING_FIELD_TWO_TARGETS_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "replace_text", "find": "target", "replace": "XYZ" }],
            }),
        )
        .await;
        let output = single(&reply);
        let body = part_text(&output, "word/document.xml");
        assert!(
            body.contains("SPANNED-CACHE"),
            "the field's cached value was rewritten: {body}"
        );
        assert!(
            !body.contains("SPANNXYZHE"),
            "the edit addressed the field region: {body}"
        );
        let text = converted_text(&ws, &output).await;
        assert!(
            text.contains("head XYZ") && text.contains("tail XYZ"),
            "an occurrence outside the field was lost: {text}"
        );
    }

    /// A match that follows a supplementary character is addressed by code
    /// point, not by UTF-16 code unit: the astral character is two code units but
    /// one character, and a code-unit offset would rewrite the character before
    /// the match.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_replaces_text_after_an_astral_character() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "docx", "file_name": "astral",
                    "content": [{ "type": "paragraph", "text": format!("{ASTRAL}abc") }],
                }),
            )
            .await,
        );
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": created.to_string_lossy(),
                "edits": [{ "op": "replace_text", "find": "abc", "replace": "XYZ" }],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;
        assert!(
            text.contains(&format!("{ASTRAL}XYZ")),
            "the character before the match was rewritten: {text}"
        );
    }

    /// `format_text` reaches the run's OWN `<w:rPr>` and never a nested
    /// `<w:rPrChange>`: a toggle the run does not have is written to it even when
    /// the tracked revision holds one (a request that changed nothing and said
    /// nothing is what this rules out), the revision keeps its own bytes, and the
    /// new properties land in front of it, the one place `<w:rPr>` accepts them.
    #[expect(clippy::too_many_lines)] // reason: set, clear and size, one case each
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_format_text_targets_the_runs_own_properties() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(&ws, "nested-rpr", NESTED_RPR_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_text", "find": "TARGET", "italic": true }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains("<w:rPr><w:b/><w:i/><w:rPrChange"),
            "the toggle did not land before the recorded revision: {body}"
        );
        assert!(
            body.contains(
                r#"<w:rPrChange w:id="1" w:author="A"><w:rPr><w:b/></w:rPr></w:rPrChange>"#
            ),
            "the tracked revision was rewritten: {body}"
        );

        let revision_only = docx_body_fixture(&ws, "revision-only", REVISION_ONLY_RPR_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": revision_only.to_string_lossy(),
                "edits": [{ "op": "format_text", "find": "TARGET", "bold": true, "size": 12 }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains(r#"<w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/><w:rPrChange"#),
            "the run did not take the properties only its revision held: {body}"
        );
        assert!(
            body.contains(r#"<w:rPr><w:b/><w:sz w:val="20"/></w:rPr>"#),
            "the revision's own properties were rewritten: {body}"
        );

        // The other direction: clearing a toggle removes the run's own property and
        // must leave the recorded revision exactly as it was — dropping the whole
        // `<w:rPr>` would delete the revision with it.
        let nested_off = docx_body_fixture(&ws, "nested-rpr-off", NESTED_RPR_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": nested_off.to_string_lossy(),
                "edits": [{ "op": "format_text", "find": "TARGET", "bold": false }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains(r#"<w:rPr><w:rPrChange w:id="1" w:author="A"><w:rPr><w:b/></w:rPr></w:rPrChange></w:rPr><w:t>TARGET"#),
            "the recorded revision did not survive the toggle being cleared: {body}"
        );

        // A run whose only property the revision holds: the request empties the
        // run's own body, and the revision is all that is left of its `<w:rPr>`.
        let revision_off =
            docx_body_fixture(&ws, "revision-only-off", REVISION_ONLY_RPR_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": revision_off.to_string_lossy(),
                "edits": [{ "op": "format_text", "find": "TARGET", "bold": false }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains(r#"<w:rPr><w:rPrChange w:id="1" w:author="A"><w:rPr><w:b/><w:sz w:val="20"/></w:rPr></w:rPrChange></w:rPr><w:t>TARGET"#),
            "a request that changed nothing removed the recorded revision: {body}"
        );

        // `<w:sz>` counts half-points, so the shared bound's low end is the smallest
        // size a document holds: a request below it is refused rather than rounded up
        // to a size the caller did not ask for.
        let below = docx_body_fixture(&ws, "size-below", NESTED_RPR_BODY).await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": below.to_string_lossy(),
                    "edits": [{ "op": "format_text", "find": "TARGET", "size": 0.05 }],
                }),
            )
            .await
            .expect_err("a size below half a point cannot be written");
        assert!(
            err.to_string().contains("between 0.5 and 819"),
            "got: {err}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let half = docx_body_fixture(&ws, "size-half", NESTED_RPR_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": half.to_string_lossy(),
                "edits": [{ "op": "format_text", "find": "TARGET", "size": 0.5 }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains(r#"<w:sz w:val="1"/>"#),
            "half a point did not write one half-point: {body}"
        );
    }

    /// A replacement takes the formatting of the run its span starts in, and the
    /// answer says so whenever ANY run the span covers is formatted differently — a
    /// differently formatted run in the middle loses exactly as much as one at the
    /// end. A removal substitutes nothing, so it takes no formatting with it and
    /// raises no such note.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_notes_formatting_loss_only_where_it_substitutes() {
        const THREE_RUNS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t xml:space="preserve">aa</w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>BOLD</w:t></w:r><w:r><w:t>zz</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
        const ACROSS: &str = "aaBOLDzz";
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(&ws, "three-runs", THREE_RUNS).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "replace_text", "find": ACROSS, "replace": "X" }],
            }),
        )
        .await;
        assert!(
            reply.contains("differently formatted runs"),
            "the formatting a replacement dropped is not named: {reply}"
        );

        let removed = docx_body_fixture(&ws, "three-runs-removed", THREE_RUNS).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": removed.to_string_lossy(),
                "edits": [{ "op": "remove_text", "find": ACROSS }],
            }),
        )
        .await;
        assert!(
            !reply.contains("differently formatted runs"),
            "a removal substituted nothing, so no formatting was lost: {reply}"
        );
    }

    /// An `add_paragraph` `after` anchors in the document's OWN paragraphs: a
    /// table cell's is one of them, a text box's is a nested one, and an `after`
    /// naming text only a text box holds is refused with a message saying where
    /// the text lies. The presentation's arm anchors only in the slide's own text
    /// and refuses a table, which the model-facing hint states.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_add_paragraph_after_says_where_the_text_lies() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(&ws, "cell-only", TABLE_CELL_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_paragraph", "text": "AFTER-CELL", "after": "CELL-ONLY" }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains(
                r#"CELL-ONLY</w:t></w:r></w:p><w:p><w:r><w:t xml:space="preserve">AFTER-CELL</w:t></w:r></w:p></w:tc>"#
            ),
            "a table cell's own paragraph must anchor the new one: {body}"
        );

        let text_box = docx_body_fixture(&ws, "text-box-only", TEXT_BOX_ONLY_BODY).await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": text_box.to_string_lossy(),
                    "edits": [{ "op": "add_paragraph", "text": "AFTER-BOX", "after": "TXBX-ONLY" }],
                }),
            )
            .await
            .expect_err("a text box is not one of the document's own paragraphs");
        assert!(err.to_string().contains("text box"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": text_box.to_string_lossy(),
                    "edits": [{ "op": "add_paragraph", "text": "AFTER-NOWHERE", "after": "NOWHERE" }],
                }),
            )
            .await
            .expect_err("text the body does not hold");
        assert!(
            err.to_string().contains("not in the document's body"),
            "got: {err}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// Removing a table cell's or a text box's only paragraph is refused: the
    /// container would be left with none, which ECMA-376 calls corrupt. A text
    /// box that keeps another paragraph can lose one.
    #[expect(clippy::too_many_lines)] // reason: a cell, a text box and a nested table, one case each
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_refuses_removing_a_containers_only_paragraph() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(&ws, "cell-only", TABLE_CELL_BODY).await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "remove_paragraph", "find": "CELL-ONLY" }],
                }),
            )
            .await
            .expect_err("a cell must keep a paragraph");
        assert!(err.to_string().contains("table cell"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let text_box = docx_body_fixture(&ws, "text-box-only", TEXT_BOX_ONLY_BODY).await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": text_box.to_string_lossy(),
                    "edits": [{ "op": "remove_paragraph", "find": "TXBX-ONLY" }],
                }),
            )
            .await
            .expect_err("a text box must keep a paragraph");
        assert!(err.to_string().contains("text box"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let two = docx_body_fixture(&ws, "text-box-two", TEXT_BOX_TWO_PARAGRAPHS_BODY).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": two.to_string_lossy(),
                "edits": [{ "op": "remove_paragraph", "find": "TXBX-GO" }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains("TXBX-KEEP"),
            "the surviving paragraph was lost: {body}"
        );
        assert!(
            !body.contains("TXBX-GO"),
            "the removed paragraph survived: {body}"
        );

        // A cell whose paragraphs sit beside a nested table: the nested cells'
        // paragraphs are not this cell's own, so the token that matches both of the
        // cell's own ones empties it and is refused, while the nested cell's own only
        // paragraph is its container's when the token names it.
        let nested = docx_body_fixture(&ws, "nested-table", NESTED_TABLE_CELL_BODY).await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": nested.to_string_lossy(),
                    "edits": [{ "op": "remove_paragraph", "find": "CELL-GO" }, { "op": "remove_paragraph", "find": "CELL-KEEP" }],
                }),
            )
            .await
            .expect_err("the cell would be left with no paragraph of its own");
        assert!(err.to_string().contains("table cell"), "got: {err}");
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": nested.to_string_lossy(),
                    "edits": [{ "op": "remove_paragraph", "find": "INNER" }],
                }),
            )
            .await
            .expect_err("the nested cell would be left with no paragraph");
        assert!(err.to_string().contains("table cell"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        // One of the cell's own paragraphs, with the nested table left standing.
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": nested.to_string_lossy(),
                "edits": [{ "op": "remove_paragraph", "find": "CELL-GO" }],
            }),
        )
        .await;
        let body = part_text(&single(&reply), "word/document.xml");
        assert!(
            body.contains("CELL-KEEP") && body.contains("INNER"),
            "the removal took the cell's own paragraph or the nested table with it: {body}"
        );
        assert!(
            !body.contains("CELL-GO"),
            "the removed paragraph survived: {body}"
        );
    }

    /// The whitespace XML allows before a close tag's `>` is read by every op that
    /// takes a position from one: a paragraph and a run whose close tags carry it
    /// are found and their text replaced rather than reported as absent from the
    /// body, and a paragraph added at the body's end goes before the body's own
    /// close tag however that close is spelled.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_reads_a_close_tag_that_carries_whitespace() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let body = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>имя</w:t></w:r ></w:p ><w:sectPr/></w:body></w:document>"#;
        let source = docx_body_fixture(&ws, "spaced-close", body).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "replace_text", "find": "имя", "replace": "фамилия" }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains("фамилия"),
            "the paragraph was not found past its close tag's whitespace: {edited}"
        );

        // A body whose own close tag carries the whitespace and which holds no
        // body-level `<w:sectPr>`: a paragraph added at the end is the body's last
        // element, written before that close and not one character short of it.
        let body = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>имя</w:t></w:r></w:p></w:body ></w:document>"#;
        let source = docx_body_fixture(&ws, "spaced-body", body).await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_paragraph", "text": "APPENDED" }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.ends_with("</w:p></w:body ></w:document>"),
            "the added paragraph did not go before the body's own close: {edited}"
        );
        assert!(
            edited.contains(r#"<w:t xml:space="preserve">APPENDED</w:t>"#),
            "the added paragraph is not in the body: {edited}"
        );
    }

    /// The colour, the underline and the strike a `format_text` names land in the
    /// run's own `<w:rPr>` at their places in `EG_RPrBase`, a `false` clears one
    /// again, and a run that already draws a richer underline or a double strike
    /// keeps exactly what it draws — while a run that states it draws no underline
    /// really takes the one a `true` asks for.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_formats_text_with_colour_underline_and_strike() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "run-colour",
            &docx_document(
                r#"<w:p><w:r><w:t>PLAIN</w:t></w:r></w:p><w:p><w:r><w:rPr><w:u w:val="double"/></w:rPr><w:t>RICH</w:t></w:r></w:p><w:p><w:r><w:rPr><w:u w:val="none"/></w:rPr><w:t>NO UNDERLINE</w:t></w:r></w:p><w:p><w:r><w:rPr><w:dstrike/></w:rPr><w:t>KEEP DOUBLE</w:t></w:r></w:p><w:p><w:r><w:rPr><w:dstrike/></w:rPr><w:t>DROP DOUBLE</w:t></w:r></w:p><w:p><w:r><w:rPr><w:dstrike w:val="false"/></w:rPr><w:t>OFF STRIKE</w:t></w:r></w:p>"#,
            ),
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_text", "find": "PLAIN", "color": "#00aaff",
                      "underline": true, "strike": true },
                    // A run's own underline may be a richer one than the plain `true`
                    // asks for, and a request never throws that away.
                    { "op": "format_text", "find": "RICH", "underline": true },
                    { "op": "format_text", "find": "PLAIN", "underline": false, "strike": false },
                    // `<w:u w:val="none"/>` is a run saying it draws no underline —
                    // `ST_Underline`'s own off spelling — so a `true` writes one.
                    { "op": "format_text", "find": "NO UNDERLINE", "underline": true },
                    // A double strike is the same property spelled another way, and a
                    // `true` leaves the one the run already draws.
                    { "op": "format_text", "find": "KEEP DOUBLE", "strike": true },
                    { "op": "format_text", "find": "DROP DOUBLE", "strike": false },
                    // A `<w:dstrike w:val="false"/>` is a run saying it draws no
                    // strike, so the `true` writes the plain one it asks for.
                    { "op": "format_text", "find": "OFF STRIKE", "strike": true },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(r#"<w:rPr><w:color w:val="00AAFF"/></w:rPr><w:t>PLAIN</w:t></w:r>"#),
            "the colour did not survive the toggles being cleared, or a toggle stayed: {edited}"
        );
        assert!(
            edited.contains(r#"<w:rPr><w:u w:val="double"/></w:rPr>"#),
            "the double underline was not kept byte for byte: {edited}"
        );
        assert!(
            edited.contains(r"<w:rPr><w:dstrike/></w:rPr><w:t>KEEP DOUBLE</w:t></w:r>"),
            "a double strike was rewritten into a single one the call did not name: {edited}"
        );
        assert!(
            edited.contains(r"<w:p><w:r><w:t>DROP DOUBLE</w:t></w:r></w:p>"),
            "a `false` left the run struck: {edited}"
        );
        assert!(
            edited.contains(r"<w:rPr><w:strike/></w:rPr><w:t>OFF STRIKE</w:t></w:r>"),
            "a run stating its strike is off reported success without one: {edited}"
        );
        assert!(
            edited.contains(r#"<w:rPr><w:u w:val="single"/></w:rPr><w:t>NO UNDERLINE</w:t></w:r>"#),
            "a run stating it draws no underline reported success without one: {edited}"
        );
        assert_eq!(
            edited.matches("<w:u ").count(),
            2,
            "a run that draws its own underline was given the plain one: {edited}"
        );
    }

    /// `format_paragraph` writes the properties it names into the paragraph's own
    /// `<w:pPr>`, at their places in `EG_PPrBase` — a paragraph inside a table cell
    /// as much as one in the body, the table keeping the shape it had — and a style
    /// is resolved by the name the document gives it. An indent written on one of
    /// `CT_Ind`'s other spellings for the same edge is cleared with it, so the
    /// indent the call named is the one the paragraph renders.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_formats_a_paragraph_inside_a_table_cell() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "paragraph-in-cell",
            &docx_document(
                r#"<w:tbl><w:tr><w:tc><w:tcPr/><w:p><w:pPr><w:ind w:left="100" w:firstLine="200"/></w:pPr><w:r><w:t>CELL TEXT</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>BODY TEXT</w:t></w:r></w:p><w:p><w:pPr><w:ind w:start="720" w:startChars="1080" w:end="1440" w:endChars="720"/></w:pPr><w:r><w:t>EDGE INDENT</w:t></w:r></w:p>"#,
            ),
            &[("word/styles.xml", FIRM_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "CELL TEXT", "align": "center",
                      "indent_left": 18, "spacing_after": 6, "style": "Фирменный" },
                    // A negative first-line indent is the hanging indent Word states,
                    // and a line height is the line's own size in points.
                    { "op": "format_paragraph", "find": "BODY TEXT", "align": "justify",
                      "indent_first_line": -12, "line_spacing": 14 },
                    { "op": "format_paragraph", "find": "EDGE INDENT", "indent_left": 18,
                      "indent_right": 9 },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Firm"/><w:spacing w:after="120"/><w:ind w:left="360" w:firstLine="200"/><w:jc w:val="center"/></w:pPr>"#
            ),
            "the cell's paragraph did not take the properties in the order a pPr holds them, or an indent the call never named was lost: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:spacing w:line="280" w:lineRule="exact"/><w:ind w:hanging="240"/><w:jc w:val="both"/></w:pPr>"#
            ),
            "the hanging indent, the line height or the justification is not what was asked: {edited}"
        );
        assert!(
            edited.contains(r#"<w:ind w:left="360" w:right="180"/></w:pPr>"#),
            "an indent stated on the leading/trailing-edge spelling kept overriding the one the call wrote: {edited}"
        );
        assert_eq!(
            (
                edited.matches("<w:tbl>").count(),
                edited.matches("<w:tr>").count(),
                edited.matches("<w:tc>").count()
            ),
            (1, 1, 1),
            "formatting a cell's paragraph changed the table: {edited}"
        );
    }

    /// A style the document does not define is refused with the styles it does
    /// define, a style that is not a paragraph style is refused as what it is, and
    /// the built-in names work in a document that defines no styles at all.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_refuses_a_style_the_document_does_not_define() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "style-answers",
            &docx_document(r"<w:p><w:r><w:t>TEXT</w:t></w:r></w:p>"),
            &[("word/styles.xml", FIRM_STYLES)],
            &[],
        )
        .await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_paragraph", "find": "TEXT", "style": "Red ink" }],
                }),
            )
            .await
            .expect_err("a character style cannot format a paragraph");
        assert!(err.to_string().contains("character style"), "got: {err}");

        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_paragraph", "find": "TEXT", "style": "Заголовок 7" }],
                }),
            )
            .await
            .expect_err("a style the document does not define");
        assert!(
            err.to_string().contains("Фирменный"),
            "the refusal does not name the styles the document defines: {err}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        // A document that saved no styles of its own still takes the built-in names.
        let bare = docx_part_fixture(
            &ws,
            "style-builtin",
            &docx_document(r"<w:p><w:r><w:t>TEXT</w:t></w:r></w:p>"),
            &[],
            &["word/styles.xml"],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": bare.to_string_lossy(),
                "edits": [{ "op": "format_paragraph", "find": "TEXT", "style": "Heading 2" }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(r#"<w:pPr><w:pStyle w:val="Heading2"/></w:pPr>"#),
            "the built-in name did not resolve in a document with no styles part: {edited}"
        );
    }

    /// A `format_paragraph` list uses the document's own definition of that kind and
    /// level, writes the one it needs when the document holds none — and the second
    /// edit of a call reuses what the first wrote — while `false` takes a
    /// paragraph's list away.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_writes_the_list_definition_it_needs() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "list-kinds",
            &docx_document(
                r#"<w:p><w:r><w:t>BULLET ONE</w:t></w:r></w:p><w:p><w:r><w:t>BULLET TWO</w:t></w:r></w:p><w:p><w:r><w:t>NUMBERED</w:t></w:r></w:p><w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>UNLISTED</w:t></w:r></w:p>"#,
            ),
            &[("word/numbering.xml", NUMBERED_ONLY)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "BULLET ONE", "list": "bullet", "level": 1 },
                    { "op": "format_paragraph", "find": "BULLET TWO", "list": "bullet", "level": 1 },
                    { "op": "format_paragraph", "find": "NUMBERED", "list": "number" },
                    { "op": "format_paragraph", "find": "UNLISTED", "list": false },
                ],
            }),
        )
        .await;
        let out = single(&reply);
        let edited = part_text(&out, "word/document.xml");
        assert_eq!(
            edited
                .matches(r#"<w:numPr><w:ilvl w:val="1"/><w:numId w:val="2"/></w:numPr>"#)
                .count(),
            2,
            "the second bullet did not reuse the definition the first edit wrote: {edited}"
        );
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>"#),
            "the document's own numbered definition was not used: {edited}"
        );
        assert_eq!(
            edited.matches("<w:numPr>").count(),
            4,
            "the paragraph's own list was not cancelled: {edited}"
        );
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr></w:pPr><w:r><w:t>UNLISTED</w:t></w:r></w:p>"#),
            "taking the list away did not state the cancelled numbering: {edited}"
        );

        let numbering = part_text(&out, "word/numbering.xml");
        assert!(
            numbering.contains(
                r#"<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/>"#
            ),
            "no bullet definition was written: {numbering}"
        );
        assert!(
            numbering
                .contains(r#"<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="bullet"/>"#),
            "the written definition is not a bullet at the level the call named: {numbering}"
        );
        assert!(
            numbering.contains(r#"<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>"#),
            "the written definition is not named by the fresh id: {numbering}"
        );
        assert!(
            numbering.contains(r#"<w:num><w:abstractNumId w:val="0"/></w:num>"#),
            "the id-less num was read from or rewritten: {numbering}"
        );
        assert!(
            numbering.find(r#"<w:abstractNum w:abstractNumId="1">"#)
                < numbering.find(r#"<w:num w:numId="1">"#),
            "the abstract definition was not written before the nums: {numbering}"
        );

        // An edit list is applied whole or not at all: the first edit here would
        // write a bullet definition and the second refuses, so the call must leave
        // neither a definition nor an output behind.
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": source.to_string_lossy(),
                    "edits": [
                        { "op": "format_paragraph", "find": "NUMBERED", "list": "bullet" },
                        { "op": "format_paragraph", "find": "NUMBERED", "style": "Заголовок 7" },
                    ],
                }),
            )
            .await
            .expect_err("the second edit names a style the document does not define");
        assert!(err.to_string().contains("Заголовок 7"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a call that refused partway left an output"
        );
    }

    /// An added paragraph takes the style, the formatting and the list membership of
    /// the paragraph it is inserted after — a heading's heading, a list item's list
    /// item — and the section break such a paragraph ends is not copied to it.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_add_paragraph_inherits_the_paragraph_it_follows() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "inherit",
            &docx_document(
                r#"<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>HEADING ONE</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/><w:numberingChange w:id="11" w:author="A" w:date="2024-01-01T00:00:00Z"/></w:numPr></w:pPr><w:r><w:t>LIST ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Normal"/><w:pPrChange w:id="3" w:author="A"><w:pPr><w:pStyle w:val="Heading1"/></w:pPr></w:pPrChange></w:pPr><w:r><w:rPr><w:ins w:id="7" w:author="A"/><w:b/></w:rPr><w:t>TRACKED ANCHOR</w:t></w:r></w:p><w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906"/></w:sectPr></w:pPr><w:r><w:t>SECTION END</w:t></w:r></w:p>"#,
            ),
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "add_paragraph", "text": "AFTER HEADING", "after": "HEADING ONE" },
                    { "op": "add_paragraph", "text": "AFTER ITEM", "after": "LIST ITEM" },
                    { "op": "add_paragraph", "text": "AFTER TRACKED", "after": "TRACKED ANCHOR" },
                    { "op": "add_paragraph", "text": "AFTER SECTION", "after": "SECTION END" },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">AFTER HEADING</w:t></w:r></w:p>"#
            ),
            "the added paragraph is not the heading the anchor is: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t xml:space="preserve">AFTER ITEM</w:t></w:r></w:p>"#
            ),
            "the added paragraph is not a list item of the anchor's own list: {edited}"
        );
        // The anchor's own history is not the new paragraph's: the recorded revisions
        // — of the properties and of the numbering — and the tracked insertion go with
        // the copy, while the style and the bold do not, and the anchor itself keeps
        // every marker.
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Normal"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">AFTER TRACKED</w:t></w:r></w:p>"#
            ),
            "the added paragraph carries a recorded revision or a tracked insertion: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:r><w:rPr><w:ins w:id="7" w:author="A"/><w:b/></w:rPr><w:t>TRACKED ANCHOR</w:t></w:r>"#
            ),
            "the anchor paragraph lost the markers of its own history: {edited}"
        );
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/><w:numberingChange w:id="11" w:author="A" w:date="2024-01-01T00:00:00Z"/></w:numPr>"#),
            "the anchor paragraph lost the marker of its own numbering's history: {edited}"
        );
        assert!(
            edited
                .contains(r#"<w:p><w:r><w:t xml:space="preserve">AFTER SECTION</w:t></w:r></w:p>"#),
            "the section break was copied to the added paragraph: {edited}"
        );
        assert_eq!(
            edited.matches("<w:sectPr").count(),
            2,
            "the section break is not left exactly where it was: {edited}"
        );
        assert!(
            reply.contains("ends a section"),
            "the answer did not say the break was not copied: {reply}"
        );
    }

    /// An edit that reaches markup the call never named says so — a hyperlink, a
    /// tracked change, a bookmark — and a document in tracked-changes mode is told
    /// that what the call wrote is plain content rather than a recorded revision.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_says_what_it_reached_beyond_the_fragment() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "reached-markup",
            &docx_document(
                r#"<w:p><w:hyperlink w:anchor="top"><w:r><w:t>LINKED</w:t></w:r></w:hyperlink></w:p><w:p><w:ins w:id="9" w:author="A"><w:r><w:t>INSERTED</w:t></w:r></w:ins></w:p><w:p><w:bookmarkStart w:id="1" w:name="BM"/><w:r><w:t>BOOKMARKED</w:t></w:r><w:bookmarkEnd w:id="1"/></w:p><w:p><w:pPr><w:pPrChange w:id="2" w:author="A"><w:pPr><w:jc w:val="center"/></w:pPr></w:pPrChange></w:pPr><w:r><w:t>REVISED</w:t></w:r></w:p>"#,
            ),
            &[("word/settings.xml", TRACKING_SETTINGS)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_text", "find": "LINKED", "bold": true },
                    { "op": "format_text", "find": "INSER", "italic": true },
                    { "op": "remove_paragraph", "find": "BOOKMARKED" },
                    { "op": "format_paragraph", "find": "REVISED", "align": "right" },
                ],
            }),
        )
        .await;
        for expected in [
            "also reached a hyperlink",
            "also reached an unaccepted tracked change",
            "also reached a bookmark",
            "tracked changes turned on",
        ] {
            assert!(
                reply.contains(expected),
                "the answer does not say {expected:?}: {reply}"
            );
        }
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:jc w:val="right"/><w:pPrChange w:id="2" w:author="A"><w:pPr><w:jc w:val="center"/></w:pPr></w:pPrChange>"#
            ),
            "the recorded revision was rewritten, or the new property did not go before it: {edited}"
        );
        assert_eq!(
            (
                edited.matches("<w:ins ").count(),
                edited.matches("<w:del ").count()
            ),
            (1, 0),
            "the call wrote a tracked change of its own: {edited}"
        );
        assert!(
            !edited.contains("BOOKMARKED"),
            "the paragraph holding the bookmark was not removed: {edited}"
        );
    }

    /// Formatting that lands wider than the fragment named says so: a fragment that
    /// matches in several places, and a fragment that covers only part of the run it
    /// sits in — where the whole run takes the formatting.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_says_when_the_formatting_landed_wider_than_named() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "wider-than-named",
            &docx_document(
                r"<w:p><w:r><w:t>SHARED and SHARED again</w:t></w:r></w:p><w:p><w:r><w:t>SHARED</w:t></w:r></w:p><w:p><w:r><w:t>LONGER RUN</w:t></w:r></w:p>",
            ),
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_text", "find": "SHARED", "bold": true },
                    { "op": "format_text", "find": "LONG", "italic": true },
                ],
            }),
        )
        .await;
        assert!(
            reply.contains("matched 3 times — every one was formatted"),
            "the answer does not say how many places took the formatting: {reply}"
        );
        assert!(
            reply.contains("covers only part of a run"),
            "the answer does not say the formatting was wider than the fragment: {reply}"
        );
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(r"<w:rPr><w:i/></w:rPr><w:t>LONGER RUN</w:t></w:r>"),
            "the run the fragment sits in did not take the formatting: {edited}"
        );
    }

    /// Removing a paragraph that holds a page break or ends a section is refused,
    /// because no note would bring the break back, while a paragraph beside a break
    /// is removed as usual.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_refuses_removing_a_paragraph_that_holds_a_break() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "breaks",
            &docx_document(
                r#"<w:p><w:r><w:t>PLAIN</w:t></w:r></w:p><w:p><w:r><w:br w:type="page"/></w:r><w:r><w:t>PAGE BREAK</w:t></w:r></w:p><w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906"/></w:sectPr></w:pPr><w:r><w:t>SECTION END</w:t></w:r></w:p>"#,
            ),
        )
        .await;
        let before = generated_count(&ws);
        for (find, expected) in [
            ("PAGE BREAK", "would delete the page break"),
            ("SECTION END", "would delete the section break"),
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "docx_edit", "path": source.to_string_lossy(),
                        "edits": [{ "op": "remove_paragraph", "find": find }],
                    }),
                )
                .await
                .expect_err("a break must not go with the paragraph");
            assert!(err.to_string().contains(expected), "got: {err}");
        }
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "remove_paragraph", "find": "PLAIN" }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(!edited.contains("PLAIN"), "the paragraph stayed: {edited}");
        assert!(
            edited.contains("PAGE BREAK") && edited.contains("SECTION END"),
            "a break went with a paragraph that did not hold it: {edited}"
        );
    }

    /// Taking a list away reaches the numbering a paragraph draws from its style — a
    /// paragraph stating no `<w:numPr>` of its own is given the cancelled id OOXML
    /// cancels an inherited number with, one deriving its style by `basedOn` too —
    /// while a paragraph that draws no numbering writes nothing at all: neither one
    /// stating none nor one whose style only states the cancellation is touched, and
    /// a call that changed no bytes reports no markup it "also reached".
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_takes_a_list_away_from_a_style_numbered_paragraph() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "style-numbered",
            &docx_document(
                r#"<w:p><w:pPr><w:pStyle w:val="Bulleted"/></w:pPr><w:r><w:t>STYLED ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Derived"/></w:pPr><w:r><w:t>DERIVED ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Recorded"/></w:pPr><w:r><w:t>RECORDED ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Cancelled"/></w:pPr><w:r><w:t>CANCELLED ITEM</w:t></w:r></w:p><w:p><w:pPr><w:numPr><w:ilvl w:val="2"/><w:numId w:val="7"/></w:numPr></w:pPr><w:r><w:t>OWN ITEM</w:t></w:r></w:p><w:p><w:r><w:t>PLAIN TEXT</w:t></w:r></w:p><w:p><w:pPr><w:jc w:val="left"/><w:sectPr><w:pgSz w:w="16838" w:h="11906"/></w:sectPr></w:pPr><w:r><w:t>SECTION END</w:t></w:r></w:p>"#,
            ),
            &[("word/styles.xml", NUMBERED_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "STYLED ITEM", "list": false },
                    { "op": "format_paragraph", "find": "DERIVED ITEM", "list": false },
                    { "op": "format_paragraph", "find": "RECORDED ITEM", "list": false },
                    { "op": "format_paragraph", "find": "CANCELLED ITEM", "list": false },
                    { "op": "format_paragraph", "find": "OWN ITEM", "list": false },
                    { "op": "format_paragraph", "find": "PLAIN TEXT", "list": false },
                    // Already what the paragraph states, so the call writes nothing —
                    // and a paragraph whose own `<w:pPr>` ends a section is then not
                    // markup the answer may claim the edit reached.
                    { "op": "format_paragraph", "find": "SECTION END", "align": "left" },
                ],
            }),
        )
        .await;
        let out = single(&reply);
        let edited = part_text(&out, "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Bulleted"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "the numbering the style gives the paragraph was not cancelled: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Derived"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "the numbering a style inherits by basedOn was not cancelled: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Recorded"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "the numbering a style draws was read from the change it recorded rather than from the list it draws: {edited}"
        );
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="2"/><w:numId w:val="0"/></w:numPr>"#),
            "the paragraph's own numbering was not cancelled in place, or its level was lost: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Cancelled"/></w:pPr><w:r><w:t>CANCELLED ITEM</w:t></w:r></w:p>"#
            ),
            "a paragraph whose style only cancels its numbering was written to: {edited}"
        );
        assert!(
            edited.contains(r"<w:p><w:r><w:t>PLAIN TEXT</w:t></w:r></w:p>"),
            "a paragraph that draws no list took a numbering statement: {edited}"
        );
        // A paragraph that draws no numbering at all and one whose style only states
        // the cancellation both keep their bytes, so the answer owes no note about
        // markup the call "reached" — nor about the section break the no-op alignment
        // edit left byte-identical.
        assert!(
            !reply.contains("also reached"),
            "a call that wrote nothing reported reaching markup: {reply}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:jc w:val="left"/><w:sectPr><w:pgSz w:w="16838" w:h="11906"/></w:sectPr></w:pPr><w:r><w:t>SECTION END</w:t></w:r></w:p>"#
            ),
            "a paragraph already stating the property the call named was rewritten: {edited}"
        );
        let styles = part_text(&out, "word/styles.xml");
        assert!(
            styles.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="7"/></w:numPr>"#),
            "the style's own numbering was rewritten: {styles}"
        );
        assert!(
            styles.contains(
                r#"<w:style w:type="paragraph" w:styleId="Cancelled"><w:name w:val="Cancelled"/><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:style>"#
            ),
            "the cancellation the style states was rewritten: {styles}"
        );
    }

    /// A `list: false` cancels numbering the document's own `<w:docDefaults>` state,
    /// while a style that itself states the cancellation stops the walk: nothing is
    /// written for its paragraphs, so the answer reports no markup reached. A style
    /// the document does not define — and a style deriving from one — falls back to
    /// the default style a reader takes for it, so the default numbering reaches
    /// those paragraphs as well and is cancelled for them too.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_cancels_numbering_the_document_defaults_state() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "default-numbered",
            &docx_document(
                r#"<w:p><w:r><w:t>DEFAULT NUMBERED</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Off"/></w:pPr><w:r><w:t>OFF ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Vanishing"/></w:pPr><w:r><w:t>VANISHING ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Gone"/></w:pPr><w:r><w:t>GONE ITEM</w:t></w:r></w:p>"#,
            ),
            &[("word/styles.xml", DEFAULT_NUMBERED_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "DEFAULT NUMBERED", "list": false },
                    { "op": "format_paragraph", "find": "OFF ITEM", "list": false },
                    { "op": "format_paragraph", "find": "VANISHING ITEM", "list": false },
                    { "op": "format_paragraph", "find": "GONE ITEM", "list": false },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr><w:r><w:t>DEFAULT NUMBERED</w:t></w:r></w:p>"#
            ),
            "the numbering the document defaults state was not cancelled: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Vanishing"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "a style deriving from one the document does not define drew no default numbering: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Gone"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "a paragraph naming a style the document does not define drew no default numbering: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Off"/></w:pPr><w:r><w:t>OFF ITEM</w:t></w:r></w:p>"#
            ),
            "a paragraph whose style states the cancellation was written to: {edited}"
        );
        assert!(
            !reply.contains("also reached"),
            "a call that wrote nothing reported reaching markup: {reply}"
        );
    }

    /// A recorded change is history, not the state a paragraph renders with: a
    /// document whose `<w:docDefaults>` only ever recorded a numbering draws none, so
    /// `list: false` on a paragraph taking the defaults writes nothing — a
    /// cancellation there would be a change the paragraph never needed.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_ignores_a_numbering_the_defaults_only_recorded() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "recorded-defaults",
            &docx_document(r"<w:p><w:r><w:t>RECORDED DEFAULT</w:t></w:r></w:p>"),
            &[("word/styles.xml", RECORDED_DEFAULT_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_paragraph", "find": "RECORDED DEFAULT", "list": false }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(r"<w:p><w:r><w:t>RECORDED DEFAULT</w:t></w:r></w:p>"),
            "a numbering the defaults only recorded was cancelled: {edited}"
        );
        assert!(
            !reply.contains("also reached"),
            "a call that wrote nothing reported reaching markup: {reply}"
        );
    }

    /// The chain a reader walks to decide whether a paragraph draws a list item
    /// marker: the paragraph's own properties, the style it is given, the styles that
    /// style derives from, the default paragraph style (the one a reader falls back to
    /// by its built-in name when the document flags none, and the one a paragraph
    /// naming no style or a style the document does not define renders with) and the
    /// document's own defaults. `list: false` cancels the numbering of every
    /// paragraph the chain finds one for, wherever it came from — while only what each
    /// rung states NOW counts, so a cancellation or a numbering a paragraph's own
    /// properties merely recorded is history rather than the list it draws — and a
    /// style that states nothing of its own draws no list at all, so its paragraph
    /// keeps its bytes.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_cancels_numbering_the_normal_style_gives() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "normal-numbered",
            &docx_document(
                r#"<w:p><w:r><w:t>NO STYLE ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Gone"/></w:pPr><w:r><w:t>GONE STYLE ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="OnNormal"/></w:pPr><w:r><w:t>DERIVED ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="OnNormal"/><w:pPrChange w:id="5" w:author="A"><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:pPrChange></w:pPr><w:r><w:t>RECORDED ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Plain"/><w:pPrChange w:id="6" w:author="A"><w:pPr><w:numPr><w:numId w:val="5"/></w:numPr></w:pPr></w:pPrChange></w:pPr><w:r><w:t>RECORDED PLAIN</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Plain"/></w:pPr><w:r><w:t>PLAIN ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Off"/></w:pPr><w:r><w:t>OFF ITEM</w:t></w:r></w:p>"#,
            ),
            &[("word/styles.xml", NORMAL_NUMBERED_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "NO STYLE ITEM", "list": false },
                    { "op": "format_paragraph", "find": "GONE STYLE ITEM", "list": false },
                    { "op": "format_paragraph", "find": "DERIVED ITEM", "list": false },
                    { "op": "format_paragraph", "find": "RECORDED ITEM", "list": false },
                    { "op": "format_paragraph", "find": "RECORDED PLAIN", "list": false },
                    { "op": "format_paragraph", "find": "PLAIN ITEM", "list": false },
                    { "op": "format_paragraph", "find": "OFF ITEM", "list": false },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr><w:r><w:t>NO STYLE ITEM</w:t></w:r></w:p>"#
            ),
            "the numbering the default paragraph style gives was not cancelled: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Gone"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "a paragraph naming a style the document does not define drew no default numbering: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="OnNormal"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "the numbering a style takes from the default style by basedOn was not cancelled: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="OnNormal"/><w:numPr><w:numId w:val="0"/></w:numPr><w:pPrChange w:id="5" w:author="A"><w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr></w:pPrChange></w:pPr>"#
            ),
            "the cancellation a paragraph's own properties only recorded was read as the list it draws, instead of the one its style gives: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Plain"/><w:pPrChange w:id="6" w:author="A"><w:pPr><w:numPr><w:numId w:val="5"/></w:numPr></w:pPr></w:pPrChange></w:pPr>"#
            ),
            "a paragraph was given a cancellation for a numbering its own properties only recorded: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Plain"/></w:pPr><w:r><w:t>PLAIN ITEM</w:t></w:r></w:p>"#
            ),
            "a paragraph whose style states nothing of its own took a numbering statement: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Off"/></w:pPr><w:r><w:t>OFF ITEM</w:t></w:r></w:p>"#
            ),
            "a paragraph whose style states the cancellation was written to: {edited}"
        );
        assert!(
            !reply.contains("also reached"),
            "a call that wrote nothing reported reaching markup: {reply}"
        );
    }

    /// An unfamiliar `w:pStyle`, and a `w:basedOn` naming a style the document does not
    /// define, are not rungs of the chain a reader walks: a paragraph naming no style
    /// the document has renders with the default one, while a style whose `basedOn`
    /// cannot be resolved is its own root, so only the document's defaults apply to it.
    /// Here the two rungs disagree — the defaults number paragraphs and the default
    /// style states the cancellation — so the two paragraphs owe opposite answers:
    /// cancelling a numbering the defaults give, and leaving untouched a paragraph the
    /// default style's own statement never reaches.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_ends_the_chain_where_the_document_does() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "cancelled-default",
            &docx_document(
                r#"<w:p><w:pPr><w:pStyle w:val="Vanishing"/></w:pPr><w:r><w:t>VANISHING ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="OnNormal"/></w:pPr><w:r><w:t>ONNORMAL ITEM</w:t></w:r></w:p>"#,
            ),
            &[("word/styles.xml", CANCELLED_DEFAULT_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "VANISHING ITEM", "list": false },
                    { "op": "format_paragraph", "find": "ONNORMAL ITEM", "list": false },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Vanishing"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "a style whose basedOn the document does not define was read as taking the default style's cancellation instead of the numbering the defaults give: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="OnNormal"/></w:pPr><w:r><w:t>ONNORMAL ITEM</w:t></w:r></w:p>"#
            ),
            "a paragraph deriving from the style that states the cancellation was written to: {edited}"
        );
        assert!(
            !reply.contains("also reached"),
            "a call that wrote one cancellation reported reaching markup: {reply}"
        );
    }

    /// The default paragraph style a reader falls back to is the ENTRY the document
    /// flags, not an id: one stating its numbering without a `w:styleId` still numbers
    /// every paragraph that renders with it — one naming no style, and one naming a
    /// style the document does not define — so `list: false` cancels them, while a
    /// paragraph whose own style derives from nothing keeps its bytes.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_cancels_numbering_an_unidentified_default_style_gives() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "unidentified-default",
            &docx_document(
                r#"<w:p><w:r><w:t>NO STYLE ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Gone"/></w:pPr><w:r><w:t>GONE ITEM</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Else"/></w:pPr><w:r><w:t>ELSE ITEM</w:t></w:r></w:p>"#,
            ),
            &[("word/styles.xml", UNIDENTIFIED_DEFAULT_STYLES)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "NO STYLE ITEM", "list": false },
                    { "op": "format_paragraph", "find": "GONE ITEM", "list": false },
                    { "op": "format_paragraph", "find": "ELSE ITEM", "list": false },
                ],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr><w:r><w:t>NO STYLE ITEM</w:t></w:r></w:p>"#
            ),
            "the numbering a default style stating no w:styleId gives was not cancelled: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:pPr><w:pStyle w:val="Gone"/><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>"#
            ),
            "a paragraph naming a style the document does not define drew no default numbering: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:p><w:pPr><w:pStyle w:val="Else"/></w:pPr><w:r><w:t>ELSE ITEM</w:t></w:r></w:p>"#
            ),
            "a paragraph whose style derives from nothing was given a numbering statement: {edited}"
        );
    }

    /// A `format_paragraph` list whose document holds no numbering part creates one:
    /// the package gains `word/numbering.xml`, the content type and the single
    /// relationship that declare it — a numbering part named twice, or one no
    /// relationship points at, is the shape Word offers to repair.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_creates_the_numbering_part_it_needs() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_unnumbered_fixture(
            &ws,
            "needs-numbering",
            &docx_document(r"<w:p><w:r><w:t>UNLISTED</w:t></w:r></w:p>"),
        )
        .await;
        let names = zip_names(&source);
        assert!(
            !names.iter().any(|name| name == "word/numbering.xml"),
            "the fixture still holds a numbering part: {names:?}"
        );
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_paragraph", "find": "UNLISTED", "list": "bullet" }],
            }),
        )
        .await;
        let out = single(&reply);
        assert_parts(&out, &["word/numbering.xml"]);
        let edited = part_text(&out, "word/document.xml");
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>"#),
            "the paragraph did not take the numbering the edit wrote: {edited}"
        );
        let numbering = part_text(&out, "word/numbering.xml");
        assert!(
            numbering.contains(
                r#"<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/>"#
            ),
            "the created part holds no bullet definition at level 0: {numbering}"
        );
        assert!(
            numbering.contains(r#"<w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num>"#),
            "the created definition is not named by a num: {numbering}"
        );
        let content_types = part_text(&out, "[Content_Types].xml");
        assert_eq!(
            content_types.matches("/word/numbering.xml").count(),
            1,
            "the numbering part is declared more than once: {content_types}"
        );
        assert!(
            content_types.contains(
                r#"ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml""#
            ),
            "the numbering part does not carry the numbering content type: {content_types}"
        );
        let rels = part_text(&out, "word/_rels/document.xml.rels");
        assert_eq!(
            rels.matches("/relationships/numbering").count(),
            1,
            "the document names more than one numbering relationship: {rels}"
        );
        assert!(
            rels.contains(r#"Target="numbering.xml""#),
            "the numbering relationship does not point at the part: {rels}"
        );
    }

    /// A body part of a document written in the strict dialect, and the
    /// relationships part that declares its own parts in it.
    const STRICT_DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://purl.oclc.org/ooxml/wordprocessingml/main"><w:body><w:p><w:r><w:t>STRICT ITEM</w:t></w:r></w:p><w:p><w:r><w:t>STRICT RIGHT</w:t></w:r></w:p><w:sectPr/></w:body></w:document>"#;
    const STRICT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://purl.oclc.org/ooxml/officeDocument/relationships/styles" Target="styles.xml"/></Relationships>"#;

    /// A document written in the strict dialect gets every part, name and value a call
    /// writes for it stated in that dialect: the numbering part a list needs, the
    /// relationship naming it and the justification of its levels (a reader of one
    /// dialect resolves only its own spellings, so a part written the other way — or a
    /// value its schema does not define — is one it never reads), an indent, which the
    /// strict schema spells `w:start`/`w:end` where the transitional one spells
    /// `w:left`/`w:right`, and an alignment, which spells the leading and trailing edge
    /// `start`/`end` the same way. The other dialect's spelling of either is cleared or
    /// never written, so only the one this document's reader resolves is left.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_writes_a_strict_documents_own_dialect() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "strict-list",
            STRICT_DOCUMENT,
            &[("word/_rels/document.xml.rels", STRICT_RELS)],
            &["word/numbering.xml"],
        )
        .await;
        let out = single(
            &run(
                &ws,
                json!({
                    "action": "docx_edit", "file_name": "edited",
                    "path": source.to_string_lossy(),
                    "edits": [
                        { "op": "format_paragraph", "find": "STRICT ITEM", "list": "bullet",
                          "indent_left": 18, "indent_right": 9, "align": "left" },
                        { "op": "format_paragraph", "find": "STRICT RIGHT", "align": "right" },
                    ],
                }),
            )
            .await,
        );
        let edited = part_text(&out, "word/document.xml");
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>"#),
            "the paragraph did not take the numbering the edit wrote: {edited}"
        );
        assert!(
            edited.contains(r#"<w:ind w:start="360" w:end="180"/>"#),
            "the indent is not stated in the dialect this document's own reader resolves: {edited}"
        );
        assert!(
            edited.contains(r#"<w:jc w:val="start"/>"#)
                && edited.contains(r#"<w:jc w:val="end"/>"#),
            "an alignment is not stated in the dialect this document's own reader resolves: {edited}"
        );
        assert!(
            !edited.contains(r"w:left=")
                && !edited.contains(r"w:right=")
                && !edited.contains(r#"<w:jc w:val="left"/>"#)
                && !edited.contains(r#"<w:jc w:val="right"/>"#),
            "a name or value is also stated in the other dialect's spelling: {edited}"
        );
        let numbering = part_text(&out, "word/numbering.xml");
        assert!(
            numbering.contains(
                r#"<w:numbering xmlns:w="http://purl.oclc.org/ooxml/wordprocessingml/main">"#
            ),
            "the created numbering part is not in the document's own dialect: {numbering}"
        );
        assert!(
            numbering.contains(r#"<w:lvlJc w:val="start"/>"#)
                && !numbering.contains(r#"<w:lvlJc w:val="left"/>"#),
            "a list level is justified in a value this document's own schema does not define: {numbering}"
        );
        let rels = part_text(&out, "word/_rels/document.xml.rels");
        assert!(
            rels.contains(
                r#"Type="http://purl.oclc.org/ooxml/officeDocument/relationships/numbering""#
            ),
            "the created part is not named by a relationship this document's reader resolves: {rels}"
        );
    }

    /// Giving a paragraph another kind of list replaces the `<w:numPr>` it had whole —
    /// and `CT_NumPr` states the revision markers of that numbering, so the request
    /// really takes them with it. The answer owes the note about the unaccepted
    /// tracked change it deleted, and the marker is gone from the file rather than
    /// reported as markup the call left alone.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_says_when_a_replaced_number_took_a_revision() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "replaced-number",
            &docx_document(
                r#"<w:p><w:pPr><w:numPr><w:numId w:val="7"/><w:numberingChange w:id="11" w:author="A" w:date="2024-01-01T00:00:00Z"/><w:ins w:id="9" w:author="A"/></w:numPr></w:pPr><w:r><w:t>NUMBERED ITEM</w:t></w:r></w:p>"#,
            ),
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_paragraph", "find": "NUMBERED ITEM", "list": "bullet" }],
            }),
        )
        .await;
        assert!(
            reply.contains("also reached an unaccepted tracked change"),
            "the answer does not say the replacement took a recorded revision: {reply}"
        );
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>"#),
            "the paragraph did not take the list the call named: {edited}"
        );
        for gone in ["<w:numberingChange", "<w:ins "] {
            assert!(
                !edited.contains(gone),
                "a revision marker of the replaced number is still in the file: {edited}"
            );
        }
    }

    /// An edit reaches only the region it rewrote: a paragraph's own section break
    /// survives byte for byte with the new property written before it, a run's
    /// recorded revision and tracked insertion marker survive with the new property
    /// beside them, and the revision markers a `<w:numPr>` states survive a list taken
    /// away from it — none is markup the answer may claim the call "also reached".
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_names_nothing_it_preserved() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "preserved",
            &docx_document(
                r#"<w:p><w:pPr><w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr></w:pPr><w:r><w:t>SECT END</w:t></w:r></w:p><w:p><w:r><w:rPr><w:rPrChange w:id="4" w:author="A"><w:rPr><w:b/></w:rPr></w:rPrChange></w:rPr><w:t>REVISED RPR</w:t></w:r></w:p><w:p><w:r><w:rPr><w:ins w:id="7" w:author="A"/><w:b/></w:rPr><w:t>INSIDE RPR</w:t></w:r></w:p><w:p><w:pPr><w:numPr><w:numId w:val="7"/><w:numberingChange w:id="11" w:author="A" w:date="2024-01-01T00:00:00Z"/><w:ins w:id="9" w:author="A"/></w:numPr></w:pPr><w:r><w:t>NUMBERED ITEM</w:t></w:r></w:p>"#,
            ),
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "SECT END", "align": "center" },
                    { "op": "format_text", "find": "REVISED RPR", "italic": true },
                    { "op": "format_text", "find": "INSIDE RPR", "italic": true },
                    { "op": "format_paragraph", "find": "NUMBERED ITEM", "list": false },
                ],
            }),
        )
        .await;
        assert!(
            !reply.contains("also reached a section break"),
            "the paragraph's own section break was reported as reached: {reply}"
        );
        assert!(
            !reply.contains("also reached an unaccepted tracked change"),
            "a recorded revision or its run's insertion marker was reported as reached: {reply}"
        );
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(
                r#"<w:jc w:val="center"/><w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>"#
            ),
            "the section break was rewritten, or the property did not go before it: {edited}"
        );
        assert!(
            edited.contains(r#"<w:rPr><w:i/><w:rPrChange w:id="4" w:author="A">"#),
            "the recorded revision was rewritten, or the new property did not go before it: {edited}"
        );
        assert!(
            edited.contains(r#"<w:rPr><w:i/><w:ins w:id="7" w:author="A"/><w:b/></w:rPr>"#),
            "the insertion marker was rewritten, or the new property did not go beside it: {edited}"
        );
        assert!(
            edited.contains(
                r#"<w:numPr><w:numId w:val="0"/><w:numberingChange w:id="11" w:author="A" w:date="2024-01-01T00:00:00Z"/><w:ins w:id="9" w:author="A"/></w:numPr>"#
            ),
            "cancelling the list rewrote the revision markers its numPr carries: {edited}"
        );
    }

    /// A run whose `<w:rPr>` holds an `<w:rPrChange>` opened and never closed no
    /// longer makes the kit throw: the call succeeds — going through `run` is
    /// itself the regression check — leaves the revision's bytes as they were, and
    /// reports no markup it "also reached".
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_survives_an_unclosed_revision() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "unclosed-revision",
            &with_first(
                &docx_document(
                    r#"<w:p><w:r><w:rPr><w:rPrChange w:id="4" w:author="A"><w:rPr><w:b/></w:rPr></w:rPrChange></w:rPr><w:t>UNCLOSED RPR</w:t></w:r></w:p>"#,
                ),
                "</w:rPrChange>",
                "",
            ),
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_text", "find": "UNCLOSED RPR", "italic": true }],
            }),
        )
        .await;
        assert!(
            !reply.contains("also reached"),
            "an unclosed revision was reported as reached: {reply}"
        );
        let edited = part_text(&single(&reply), "word/document.xml");
        assert!(
            edited.contains(r#"<w:rPrChange w:id="4" w:author="A">"#),
            "the unclosed revision's bytes were rewritten: {edited}"
        );
    }

    /// A run whose own `<w:rPr>` is opened and never closed is refused rather than
    /// given a second one beside it: the properties the call states would land inside
    /// the unclosed element, where no reader takes them, and the call would report a
    /// formatting it never applied — the refusal a paragraph's unclosed `<w:pPr>`
    /// already gets.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_refuses_a_run_whose_properties_never_close() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_body_fixture(
            &ws,
            "unclosed-run-properties",
            &docx_document(r"<w:p><w:r><w:rPr><w:b/><w:t>UNCLOSED OWN RPR</w:t></w:r></w:p>"),
        )
        .await;
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "docx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_text", "find": "UNCLOSED OWN RPR", "italic": true }],
                }),
            )
            .await
            .expect_err("a run whose properties never close cannot be formatted");
        assert!(
            err.to_string().contains("never closed"),
            "the refusal does not name the unclosed properties: {err}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A list edit reuses only a definition that RENDERS the kind it names: a
    /// `<w:num>` whose abstract states `decimal` but whose own `<w:lvlOverride>`
    /// renders a bullet is reused for a bullet list and refused for a numbered one,
    /// which writes a fresh definition of its own instead.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn docx_edit_reuses_only_a_list_definition_that_renders_the_kind() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = docx_part_fixture(
            &ws,
            "override-kinds",
            &docx_document(
                r"<w:p><w:r><w:t>BULLET OVERRIDE</w:t></w:r></w:p><w:p><w:r><w:t>NUMBER FRESH</w:t></w:r></w:p>",
            ),
            &[("word/numbering.xml", OVERRIDDEN_NUMBERING)],
            &[],
        )
        .await;
        let reply = run(
            &ws,
            json!({
                "action": "docx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_paragraph", "find": "BULLET OVERRIDE", "list": "bullet" },
                    { "op": "format_paragraph", "find": "NUMBER FRESH", "list": "number" },
                ],
            }),
        )
        .await;
        let out = single(&reply);
        let edited = part_text(&out, "word/document.xml");
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>"#),
            "the bullet list did not reuse the definition that renders one: {edited}"
        );
        assert!(
            edited.contains(r#"<w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>"#),
            "the numbered list reused an id that renders a bullet there: {edited}"
        );
        let numbering = part_text(&out, "word/numbering.xml");
        assert!(
            numbering.contains(
                r#"<w:num w:numId="1"><w:abstractNumId w:val="0"/><w:lvlOverride w:ilvl="0">"#
            ),
            "the reused definition was rewritten: {numbering}"
        );
        assert!(
            numbering.contains(
                r#"<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>"#
            ),
            "no fresh decimal definition was written: {numbering}"
        );
        assert!(
            numbering.contains(r#"<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>"#),
            "the fresh definition is not named by a num: {numbering}"
        );
    }

    /// An xlsx edit rewrites only the sheets it names: the workbook's chart,
    /// pivot, table, comment and macro parts stay byte-identical, the cells read
    /// back at their new addresses, and the reply names the stale caches.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: one workbook exercising every cell and shift op
    async fn xlsx_edit_writes_cells_and_reports_stale_charts() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [
                        { "name": "Values", "rows": [["Name", 1], ["Alpha", 7]] },
                        { "name": "Rows", "rows": [["one"], ["two"], ["three"]] },
                        { "name": "Columns", "rows": [["one", "two", "three"]] },
                    ],
                }),
            )
            .await,
        );
        let package = with_parts(
            &std::fs::read(&created).expect("read base package"),
            &[
                ("xl/charts/chart1.xml", EDIT_CHART.as_bytes()),
                (
                    "xl/pivotCache/pivotCacheDefinition1.xml",
                    EDIT_PIVOT.as_bytes(),
                ),
                ("xl/tables/table1.xml", EDIT_TABLE.as_bytes()),
                ("xl/comments1.xml", EDIT_COMMENTS.as_bytes()),
                ("xl/drawings/vmlDrawing1.vml", EDIT_VML.as_bytes()),
                ("xl/vbaProject.bin", b"macro-bytes"),
            ],
        );
        let source = write_fixture(&ws, "book.xlsx", &package);
        // The same bytes under a macro-enabled name: an edit keeps that name.
        let macro_source = write_fixture(&ws, "book.xlsm", &package);

        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "set_cell", "sheet": "Values", "cell": "B2", "value": 42 },
                    { "op": "set_cell", "sheet": "Values", "cell": "C2", "value": "added" },
                    { "op": "set_cell", "sheet": "Values", "cell": "D2", "value": true },
                    { "op": "set_cell", "sheet": "Values", "cell": "E2",
                      "value": { "formula": "=SUM(B2:B3)" } },
                    { "op": "set_cell", "sheet": "Values", "cell": "F2", "value": 1.5,
                      "number_format": "#,##0.00" },
                    { "op": "clear_cell", "sheet": "Values", "cell": "A1" },
                    { "op": "insert_row", "sheet": "Rows", "row": 2 },
                    { "op": "delete_row", "sheet": "Rows", "row": 4 },
                    { "op": "insert_column", "sheet": "Columns", "column": "B" },
                    { "op": "delete_column", "sheet": "Columns", "column": "C" },
                ],
            }),
        )
        .await;
        let output = single(&reply);

        let touched = [
            "xl/worksheets/sheet1.xml",
            "xl/worksheets/sheet2.xml",
            "xl/worksheets/sheet3.xml",
            "xl/styles.xml",
        ];
        let names = zip_names(&source);
        for name in &names {
            assert!(
                zip_names(&output).contains(name),
                "{name} vanished from the edited workbook"
            );
            if touched.contains(&name.as_str()) {
                continue;
            }
            assert_eq!(
                part_bytes(&source, name),
                part_bytes(&output, name),
                "{name} changed"
            );
        }
        assert_eq!(
            names.len(),
            zip_names(&output).len(),
            "a part was added to the edited workbook"
        );

        let text = converted_text(&ws, &output).await;
        for expected in [
            "B2: 42",
            "C2: added",
            "D2: TRUE",
            "E2: =SUM(B2:B3)",
            "F2: 1.50",
            "A3: two",
            "C1: three",
        ] {
            assert!(text.contains(expected), "{expected} missing from: {text}");
        }
        assert!(
            !text.contains("Name"),
            "the cleared cell still reads: {text}"
        );
        assert!(
            part_text(&output, "xl/styles.xml").contains("formatCode=\"#,##0.00\""),
            "the number format was not written"
        );

        assert!(
            reply.contains("charts were copied unchanged"),
            "the stale-chart caveat is missing: {reply}"
        );
        assert!(
            reply.contains("pivot tables were copied unchanged"),
            "the stale-pivot caveat is missing: {reply}"
        );

        let macro_reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "macro-edited",
                "path": macro_source.to_string_lossy(),
                "edits": [{ "op": "set_cell", "sheet": "Values", "cell": "B2", "value": 7 }],
            }),
        )
        .await;
        assert_eq!(
            single(&macro_reply)
                .extension()
                .and_then(|ext| ext.to_str()),
            Some("xlsm"),
            "a macro-enabled input must keep its extension: {macro_reply}"
        );
    }

    /// An Excel-shaped `xl/styles.xml`: `<cellXfs>` entries carrying an
    /// `<alignment>` and a `<protection>` child (legal `CT_Xf`, and what Excel
    /// writes), a `<numFmt>` whose code holds a dollar sign, and a `<cellStyles>`
    /// block after `<cellXfs>`. A `<cellXfs>` rebuilt from self-closing `<xf/>`
    /// alone would drop the child-bearing entries and shift every `s=`.
    const EXCEL_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="&quot;$&quot;#,##0.00"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyNumberFormat="0"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"><protection locked="0"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>"#;
    /// A sheet whose A1 uses `s="0"` (the bold-font entry, which states
    /// `applyNumberFormat="0"`): setting a number format must keep that font, not
    /// replace the entry with a bare one.
    const EXCEL_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B1"/><sheetData><row r="1"><c r="A1" s="0"><v>1</v></c><c r="B1" s="1" t="inlineStr"><is><t>x</t></is></c></row></sheetData></worksheet>"#;

    /// Setting a number format on an Excel-shaped workbook keeps every `<xf>`
    /// entry (children included), keeps the edited cell's own font, states
    /// `applyNumberFormat` even when the cell's own entry left it at `0` (a format
    /// written but never applied), adds the new `<numFmt>` without a dollar sign in
    /// a code corrupting the part (a replacement string reads `$&`), and points the
    /// cell at a clone of its own entry.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_set_cell_keeps_each_cell_format_and_a_dollar_code() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let package = with_parts(
            &std::fs::read(&created).expect("read base package"),
            &[
                ("xl/styles.xml", EXCEL_STYLES.as_bytes()),
                ("xl/worksheets/sheet1.xml", EXCEL_SHEET.as_bytes()),
            ],
        );
        let source = write_fixture(&ws, "styled.xlsx", &package);
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1234.5,
                            "number_format": "\"$\"#,##0.000" }],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        assert_eq!(
            styles.matches("<styleSheet").count(),
            1,
            "the part was corrupted by a dollar sign: {styles}"
        );
        assert_eq!(styles.matches("<cellXfs").count(), 1, "{styles}");
        assert!(
            styles.contains(r#"<alignment horizontal="center" wrapText="1"/>"#),
            "a child-bearing <xf> was dropped: {styles}"
        );
        assert!(
            styles.contains(r#"<protection locked="0"/>"#),
            "a child-bearing <xf> was dropped: {styles}"
        );
        assert!(
            styles.contains(r#"<numFmt numFmtId="164" formatCode="&quot;$&quot;#,##0.00"/>"#),
            "an existing dollar code was rewritten: {styles}"
        );
        assert!(
            styles.contains(r#"<numFmt numFmtId="165" formatCode="&quot;$&quot;#,##0.000"/>"#),
            "the new format was not written: {styles}"
        );
        assert_eq!(
            styles.matches("<numFmt ").count(),
            2,
            "the format table gained or lost an entry: {styles}"
        );
        assert!(
            styles.contains(r#"numFmtId="165" fontId="1""#),
            "the edited cell lost its own font: {styles}"
        );
        assert!(
            styles.contains(
                r#"<xf numFmtId="165" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyNumberFormat="1"/>"#
            ),
            "the cloned entry kept the cell's own applyNumberFormat=0, so the format would not be applied: {styles}"
        );
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains(r#"<c r="A1" s="3"><v>1234.5</v></c>"#),
            "the cell does not point at its own cloned entry: {sheet}"
        );
    }

    /// A worksheet whose rows and cells carry no `r`: the shift must address them
    /// by the position that addresses them (the rule the reader and the filler
    /// use), or the row lands in the wrong place and a deleted column's cell
    /// stays behind.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_shifts_addressless_rows_and_cells() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");

        let addressless_rows = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData></worksheet>"#;
        let rows_source = write_fixture(
            &ws,
            "no-row.xlsx",
            &with_parts(
                &base,
                &[("xl/worksheets/sheet1.xml", addressless_rows.as_bytes())],
            ),
        );
        let rows_reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "rows-edited",
                "path": rows_source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let rows_sheet = part_text(&single(&rows_reply), "xl/worksheets/sheet1.xml");
        assert!(
            rows_sheet.contains(
                r#"<sheetData><row r="1"/><row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>"#
            ),
            "the inserted row did not land in position: {rows_sheet}"
        );

        let addressless_cells = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c><c><v>2</v></c><c r="C1"><v>3</v></c></row></sheetData></worksheet>"#;
        let cells_source = write_fixture(
            &ws,
            "no-cell.xlsx",
            &with_parts(
                &base,
                &[("xl/worksheets/sheet1.xml", addressless_cells.as_bytes())],
            ),
        );
        let deleted = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "cells-deleted",
                "path": cells_source.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "B" }],
            }),
        )
        .await;
        let deleted_path = single(&deleted);
        let deleted_sheet = part_text(&deleted_path, "xl/worksheets/sheet1.xml");
        assert!(
            deleted_sheet.contains(r#"<c r="A1"><v>1</v></c><c r="B1"><v>3</v></c>"#),
            "the addressless cell in the deleted column stayed: {deleted_sheet}"
        );
        assert!(
            !deleted_sheet.contains("<c><v>2</v></c>"),
            "the deleted column's cell survived: {deleted_sheet}"
        );
        let read_back = converted_text(&ws, &deleted_path).await;
        assert!(
            read_back.contains("A1: 1") && read_back.contains("B1: 3"),
            "the reader does not show the shifted cells: {read_back}"
        );

        let inserted = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "cells-inserted",
                "path": cells_source.to_string_lossy(),
                "edits": [{ "op": "insert_column", "sheet": "S", "column": "B" }],
            }),
        )
        .await;
        let inserted_sheet = part_text(&single(&inserted), "xl/worksheets/sheet1.xml");
        assert!(
            inserted_sheet
                .contains(r#"<c r="A1"><v>1</v></c><c r="C1"><v>2</v></c><c r="D1"><v>3</v></c>"#),
            "an addressless cell did not move for the insert: {inserted_sheet}"
        );
    }

    /// A part that already states one row number twice is not healed and not
    /// refused for its own repeat, but an edit that would carry that repeat onto a
    /// number the part holds a single row at is refused: the written part would
    /// state an address twice where the part it was built from did not, which the
    /// reply would pass off as a success. The refusal names the address and no
    /// output is written.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_a_part_it_would_state_an_address_twice_in() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;

        // The middle row states no `r`, so it stands for the position after its
        // predecessor, and the last two rows both state `3`. Deleting the
        // addressless line moves both `3`s up onto the `2` the sheet holds one
        // row at.
        let repeated = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row><c r="A2"><v>2</v></c></row><row r="3"><c r="A3"><v>3</v></c></row><row r="3"><c r="A4"><v>4</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "repeated.xlsx",
            &with_parts(&base, &[("xl/worksheets/sheet1.xml", repeated.as_bytes())]),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "refused",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "delete_row", "sheet": "S", "row": 2 }],
                }),
            )
            .await
            .expect_err("a part the edit would state an address twice in must be refused");
        let message = err.to_string();
        assert!(
            message.contains("the row 2 would be stated twice"),
            "the refusal did not name the address it would state twice: {message}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        // The same delete on a sheet whose rows each state their own number: no
        // address is stated twice, the delete goes through, and the surviving row
        // moves up onto the deleted row's number with its cell.
        let addressed = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c></row><row r="3"><c r="A3"><v>3</v></c></row><row r="4"><c r="A4"><v>4</v></c></row></sheetData></worksheet>"#;
        let addressed_source = write_fixture(
            &ws,
            "addressed.xlsx",
            &with_parts(&base, &[("xl/worksheets/sheet1.xml", addressed.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": addressed_source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 3 }],
            }),
        )
        .await;
        let sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains(r#"<row r="3"><c r="A3"><v>4</v></c></row>"#),
            "the row below the deleted one did not move up onto its number: {sheet}"
        );
        assert!(
            !sheet.contains("<v>3</v>"),
            "the deleted row's cell survived: {sheet}"
        );
    }

    /// A row the writer left no `r` for stands for the position it holds, and the
    /// delete of the line above moves that row and its cells up together: the
    /// written part states the surviving row's own number and holds its cells at
    /// that number, rather than leaving a row stating a number its cells are not
    /// at, which a reader offers to repair.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_a_surviving_row_with_the_cells_it_holds() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let sheet_text = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="2"><c r="A2"><v>2</v></c></row><row><c r="A3"><v>3</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "survivor.xlsx",
            &with_parts(
                &base,
                &[("xl/worksheets/sheet1.xml", sheet_text.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 2 }],
            }),
        )
        .await;
        let sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains(r#"<row r="2"><c r="A2"><v>3</v></c></row>"#),
            "the surviving row was left stating a number its cell is not at: {sheet}"
        );
    }

    /// A cell whose own `r` is an address this kit cannot place — `$A$1` is a cell
    /// reference the format allows, and the reader places the plain form — keeps
    /// its bytes: an edit that makes a writer's implicit addresses explicit must
    /// not write a second `r` beside the one the cell already states, which no
    /// reader accepts.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_leaves_a_cell_address_it_cannot_place_alone() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let sheet_text = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="$A$1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "unplaceable.xlsx",
            &with_parts(
                &base,
                &[("xl/worksheets/sheet1.xml", sheet_text.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        for cell in sheet.split("<c ").skip(1) {
            let open = &cell[..cell.find('>').expect("a cell's open tag")];
            assert_eq!(
                open.matches("r=\"").count(),
                1,
                "a cell states its address more than once: <c {open}>"
            );
        }
        assert!(
            sheet.contains(r#"<c r="$A$1"><v>1</v></c>"#),
            "the cell's own spelling was rewritten: {sheet}"
        );
        assert!(
            sheet.contains(r#"<c r="B2"><v>2</v></c>"#),
            "the cell beside it did not move with its row: {sheet}"
        );
    }

    /// A line inside the used range the writer left NO `<row>` element for is
    /// still a line a delete takes: everything below it moves up by one, the
    /// same rule the column arm follows. Only a line past the used range is
    /// refused.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_deletes_a_line_the_writer_left_no_row_element_for() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B5"/><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c></row><row r="3"><c r="A3"><v>3</v></c></row><row r="5"><c r="A5"><v>5</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "gap.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "gap-deleted",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 4 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<row r="4"><c r="A4"><v>5</v></c></row></sheetData>"#),
            "the row below the deleted line did not move up: {output}"
        );
        let past = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "xlsx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "delete_row", "sheet": "S", "row": 6 }],
                }),
            )
            .await
            .expect_err("a row past the used range cannot be deleted");
        assert!(
            past.to_string()
                .contains("past the sheet's used range (1-5)"),
            "a row past the used range was not refused by that rule: {past}"
        );
    }

    /// The `(min, max)` of every `<col min max>` in a worksheet part.
    fn col_ranges(xml: &str) -> Vec<(u32, u32)> {
        xml.split("<col ")
            .skip(1)
            .map(|rest| {
                let min = rest
                    .split("min=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .and_then(|s| s.parse().ok())
                    .expect("a col min");
                let max = rest
                    .split("max=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .and_then(|s| s.parse().ok())
                    .expect("a col max");
                (min, max)
            })
            .collect()
    }

    /// A merge and a `<col min max>` range the deleted column falls in: the range
    /// shrinks or is dropped rather than left as a one-cell merge or a duplicate
    /// `<col>` range.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_drops_or_shrinks_merges_and_column_entries_a_delete_covers() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cols><col min="1" max="1" width="10" customWidth="1"/><col min="2" max="2" width="20" customWidth="1"/><col min="3" max="3" width="30" customWidth="1"/></cols><dimension ref="A1:D2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c><c r="C1"><v>3</v></c><c r="D1"><v>4</v></c></row><row r="2"><c r="A2"><v>5</v></c><c r="B2"><v>6</v></c><c r="C2"><v>7</v></c><c r="D2"><v>8</v></c></row></sheetData><mergeCells count="3"><mergeCell ref="B1:C1"/><mergeCell ref="C1:D1"/><mergeCell ref="C2:D2"/></mergeCells></worksheet>"#;
        let source = write_fixture(
            &ws,
            "merge.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "B" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(
                r#"<mergeCells count="2"><mergeCell ref="B1:C1"/><mergeCell ref="B2:C2"/></mergeCells>"#
            ),
            "a one-cell merge was kept or a range was not shrunk: {output}"
        );
        assert!(
            output.contains(
                r#"<cols><col min="1" max="1" width="10" customWidth="1"/><col min="2" max="2" width="30" customWidth="1"/></cols>"#
            ),
            "the <cols> container lost a surviving entry, reordered them or had a count invented for it (`CT_Cols` declares no attributes): {output}"
        );
        let ranges = col_ranges(&output);
        assert_eq!(ranges, vec![(1, 1), (2, 2)], "col ranges: {ranges:?}");
        for (index, (low, high)) in ranges.iter().enumerate() {
            for (other, (other_low, other_high)) in ranges.iter().enumerate() {
                if index != other {
                    assert!(
                        high < other_low || other_high < low,
                        "two <col> entries overlap: {ranges:?}"
                    );
                }
            }
        }
    }

    /// A delete that covers the last entry of a container the shift rewrites drops
    /// the container with it — ECMA-376 requires a child, so an emptied
    /// `<mergeCells>`/`<cols>`/`<hyperlinks>` is schema-invalid — and a hyperlink
    /// on a deleted cell goes with it rather than re-attaching to whatever cell
    /// shifts into its address.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_drops_a_container_a_delete_empties() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cols><col min="1" max="1" width="10" customWidth="1"/></cols><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:B1"/></mergeCells><hyperlinks><hyperlink ref="A2" r:id="rId1"/></hyperlinks></worksheet>"#;
        let source = write_fixture(
            &ws,
            "empties.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );

        // The header merge is the only one and the row delete covers it: the
        // emptied container goes, and the link below the header moves up with its
        // cell instead of staying behind.
        let dropped = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "row-deleted",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let rows = part_text(&single(&dropped), "xl/worksheets/sheet1.xml");
        assert!(
            !rows.contains("<mergeCells"),
            "an emptied <mergeCells> was left behind: {rows}"
        );
        assert!(
            rows.contains(r#"<hyperlinks><hyperlink ref="A1" r:id="rId1"/></hyperlinks>"#),
            "the link did not move with its cell, or a count `CT_Hyperlinks` does not declare was invented for it: {rows}"
        );

        // The column delete removes the only `<col>` entry, shrinks the header
        // merge to a non-merge, and covers the link's own cell: all three
        // containers are emptied and dropped.
        let deleted = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "column-deleted",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "A" }],
            }),
        )
        .await;
        let columns = part_text(&single(&deleted), "xl/worksheets/sheet1.xml");
        for container in ["<cols", "<mergeCells", "<hyperlinks"] {
            assert!(
                !columns.contains(container),
                "an emptied {container}> was left behind: {columns}"
            );
        }
    }

    /// A counted container one of whose children the shift rewrites may be left
    /// with its LAST child gone: `<dataValidations>`, `<protectedRanges>` and
    /// `<ignoredErrors>` require a child (ECMA-376), so an emptied one goes the
    /// way the emptied `<mergeCells>`/`<cols>`/`<hyperlinks>` do, and a count the
    /// file stated follows the children that survive rather than naming one that
    /// does not. The entries are moved inside the container's own pass — the
    /// failure this covers is an entry removed before its container was walked,
    /// which left the container empty with its old count and a reply that called
    /// the corrupt result a success.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_drops_a_counted_container_a_delete_empties() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one", "two"], ["three", "four"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");

        // Every container's only entry names the column the delete covers: all
        // three are emptied and must go, each with a note naming the drop.
        let emptied = write_fixture(
            &ws,
            "emptied-counted.xlsx",
            &with_parts(
                &base,
                &[(
                    "xl/worksheets/sheet1.xml",
                    br#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><dataValidations count="1"><dataValidation type="list" sqref="B2"/></dataValidations><protectedRanges><protectedRange sqref="B1:B2"/></protectedRanges><ignoredErrors><ignoredError sqref="B1:B2" numberStoredAsText="1"/></ignoredErrors></worksheet>"#,
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "emptied",
                "path": emptied.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "B" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        for container in ["<dataValidations", "<protectedRanges", "<ignoredErrors"] {
            assert!(
                !output.contains(container),
                "an emptied {container}> with a stale count was left behind: {output}"
            );
        }
        for named in ["data validation", "protected range", "ignored error"] {
            assert!(
                reply.contains(named),
                "the dropped {named} is not named in the answer: {reply}"
            );
        }
        assert!(
            output.contains(r#"<dimension ref="A1:A2"/>"#),
            "the shift did not run, so the test proves nothing: {output}"
        );

        // The partial case: two validations, the delete covers one, so the
        // container stays and its stated count follows the survivor; a
        // `<protectedRanges>` with no count at all keeps its open tag as written.
        let partial = write_fixture(
            &ws,
            "partial-counted.xlsx",
            &with_parts(
                &base,
                &[(
                    "xl/worksheets/sheet1.xml",
                    br#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><dataValidations count="2"><dataValidation type="list" sqref="B2"/><dataValidation type="list" sqref="A1"/></dataValidations><protectedRanges><protectedRange sqref="B1:B2"/><protectedRange sqref="A1"/></protectedRanges></worksheet>"#,
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "partial",
                "path": partial.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "B" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<dataValidations count="1"><dataValidation type="list" sqref="A1"/></dataValidations>"#),
            "the count did not follow the surviving validation: {output}"
        );
        assert!(
            output.contains(r#"<protectedRanges><protectedRange sqref="A1"/></protectedRanges>"#),
            "a count `CT_ProtectedRanges` does not declare was invented, or the survivor was lost: {output}"
        );
    }

    /// A `sqref` rewrite stays inside the tag that carries the attribute: a cell
    /// whose own TEXT reads `sqref="B2"` is content, not an attribute, and an
    /// unrelated row shift must leave it byte-identical while moving the real
    /// `sqref` with its cell.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_keeps_sqref_looking_cell_text_through_a_shift() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t xml:space="preserve">sqref="B2"</t></is></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><dataValidations count="1"><dataValidation type="list" sqref="B2"/></dataValidations></worksheet>"#;
        let source = write_fixture(
            &ws,
            "sqref.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<t xml:space="preserve">sqref="B2"</t>"#),
            "the cell's own text was rewritten as if it were an attribute: {output}"
        );
        assert!(
            output.contains(r#"sqref="B3"/>"#),
            "the real sqref attribute did not move with its cell: {output}"
        );
    }

    /// A delete shrinks the `<dimension ref>`, every `sqref` token and a
    /// `<mergeCell>` the same way it shrinks a merge: a range an end of which is
    /// the deleted line no longer names it, a range the delete covered whole is
    /// dropped (with a note when it carried an `sqref`), and a one-cell
    /// `<mergeCell>` the delete does not reach survives byte for byte.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_shrinks_the_dimension_and_sqref_and_keeps_an_untouched_one_cell_merge() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><dataValidations count="2"><dataValidation type="list" sqref="A1:B2 C2"/><dataValidation type="list" sqref="B2"/></dataValidations><mergeCells count="2"><mergeCell ref="A1:A1"/><mergeCell ref="A2:B2"/></mergeCells></worksheet>"#;
        let source = write_fixture(
            &ws,
            "shrink.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "B" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<dimension ref="A1:A2"/>"#),
            "the dimension still names the deleted column: {output}"
        );
        assert!(
            output.contains(r#"<dataValidation type="list" sqref="A1:A2 B2"/>"#),
            "the sqref range did not shrink: {output}"
        );
        assert!(
            !output.contains(r#"sqref="B2""#),
            "an sqref the delete covered whole survived: {output}"
        );
        assert!(
            output.contains(r#"<dataValidations count="1">"#),
            "the data validations count did not follow the dropped entry: {output}"
        );
        assert!(
            reply.contains("data validation's range")
                && reply.contains("B2")
                && reply.contains("gone with it"),
            "the dropped sqref was not named in a note: {reply}"
        );
        assert!(
            output.contains(r#"<mergeCells count="1"><mergeCell ref="A1:A1"/></mergeCells>"#),
            "the untouched one-cell merge was dropped or the covered one kept: {output}"
        );
    }

    /// A row or column shift moves what a sheet's rules and validations carry:
    /// the range they name AND the references their own formulas hold, so a rule
    /// keeps testing the cells it was written for. A cell's `<f>` is not rewritten
    /// (the reply names it instead), a value list is text rather than addresses,
    /// and a reference qualified with another sheet's name is left alone.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_rule_and_validation_formulas_with_their_ranges() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed", "x"], ["y", "z"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><conditionalFormatting sqref="A1:B2"><cfRule type="expression" dxfId="0" priority="1"><formula>$A$1&gt;5</formula></cfRule></conditionalFormatting><dataValidations count="2"><dataValidation type="list" sqref="B2"><formula1>"yes,no"</formula1></dataValidation><dataValidation type="list" sqref="A1:A2"><formula1>$A$1:$A$2</formula1></dataValidation></dataValidations></worksheet>"#;
        let source = write_fixture(
            &ws,
            "rules.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r"<formula>$A$2&gt;5</formula>"),
            "the conditional format's formula did not move with its range: {output}"
        );
        assert!(
            output.contains(r"<formula1>$A$2:$A$3</formula1>"),
            "the validation's formula did not move with its range: {output}"
        );
        assert!(
            output.contains(r#"<formula1>"yes,no"</formula1>"#),
            "a value list is text, not addresses: {output}"
        );

        let referencing = sheet.replace(
            "<formula1>$A$1:$A$2</formula1>",
            "<formula1>Other!$A$1</formula1><formula2>$A$2&gt;5</formula2>",
        );
        let other = write_fixture(
            &ws,
            "other.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", referencing.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": other.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 2 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r"<formula1>Other!$A$1</formula1>"),
            "a reference qualified with another sheet's name is not this sheet's: {output}"
        );
        assert!(
            output.contains(r"<formula2>#REF!&gt;5</formula2>"),
            "a reference the delete took must not name the cell that shifted in: {output}"
        );
    }

    /// The extended rule Excel writes in an `<extLst>` beside a base conditional
    /// format or data validation — the same range in an `<xm:sqref>` child and the
    /// formula in an `<xm:f>` child — moves with the base rule, and so do a
    /// whole-column range and a reference the file qualified with this sheet's own
    /// name. An extended rule whose range the delete took goes with it, and its
    /// container's `count` follows.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_whole_line_and_qualified_rule_ranges_with_their_extended_twins() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [[1, 2, 3, 4], [5, 6, 7, 8]] }],
                }),
            )
            .await,
        );
        // The base rules name whole columns and a range the file qualified with the
        // sheet's own name; the extended twins hold the same in `xm:` children, and
        // the second validation's range is the column the edit deletes.
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:D2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c><c r="C1"><v>3</v></c><c r="D1"><v>4</v></c></row><row r="2"><c r="A2"><v>5</v></c><c r="B2"><v>6</v></c><c r="C2"><v>7</v></c><c r="D2"><v>8</v></c></row></sheetData><conditionalFormatting sqref="B:D"><cfRule type="expression" dxfId="0" priority="1"><formula>SUM($B:$D)&gt;0</formula></cfRule></conditionalFormatting><dataValidations count="1"><dataValidation type="list" sqref="C:C"><formula1>S!$B$1:$B$5</formula1></dataValidation></dataValidations><extLst><ext uri="{78C0D931-6437-407d-A8EE-F0AAD7539E65}"><x14:conditionalFormattings xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main"><x14:conditionalFormatting xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:cfRule type="expression" priority="1" id="{AAAA}"><xm:f>SUM($B:$D)&gt;0</xm:f></x14:cfRule><xm:sqref>B:D</xm:sqref></x14:conditionalFormatting><x14:conditionalFormatting xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:cfRule type="expression" priority="2" id="{EEEE}"><xm:f>$A$1&gt;0</xm:f></x14:cfRule><xm:sqref>A:A</xm:sqref></x14:conditionalFormatting></x14:conditionalFormattings></ext><ext uri="{11111111-2222-3333-4444-555555555555}"><x14:conditionalFormattings xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main"><x14:conditionalFormatting xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:cfRule type="expression" priority="3" id="{FFFF}"><xm:f>$A$2&gt;0</xm:f></x14:cfRule><xm:sqref>A1:A2</xm:sqref></x14:conditionalFormatting></x14:conditionalFormattings></ext><ext uri="{CCE6A557-97BC-4b89-ADB6-D9C93CAAB3DF}"><x14:dataValidations count="2" xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:dataValidation type="list" allowBlank="1"><x14:formula1><xm:f>S!$B$1:$B$5</xm:f></x14:formula1><xm:sqref>C:C</xm:sqref></x14:dataValidation><x14:dataValidation type="list" allowBlank="1"><x14:formula1><xm:f>A1</xm:f></x14:formula1><xm:sqref>A:A</xm:sqref></x14:dataValidation></x14:dataValidations></ext></extLst></worksheet>"#;
        let source = write_fixture(
            &ws,
            "extended.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "A" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"sqref="A:C""#),
            "the whole-column rule range did not move with the delete: {output}"
        );
        assert!(
            output.contains(r"<formula>SUM($A:$C)&gt;0</formula>"),
            "a whole-column range in the rule's formula did not move: {output}"
        );
        assert!(
            output.contains(r#"<dataValidation type="list" sqref="B:B">"#),
            "the validation's whole-column range did not move: {output}"
        );
        assert!(
            output.contains(r"<formula1>S!$A$1:$A$5</formula1>"),
            "a reference the file qualified with this sheet's own name did not move: {output}"
        );
        assert!(
            output.contains(r"<xm:sqref>A:C</xm:sqref>")
                && output.contains(r"<xm:f>SUM($A:$C)&gt;0</xm:f>"),
            "the extended conditional format did not move with its base rule: {output}"
        );
        assert!(
            output.contains(r"<xm:sqref>B:B</xm:sqref>")
                && output.contains(r"<xm:f>S!$A$1:$A$5</xm:f>"),
            "the extended validation did not move with its base rule: {output}"
        );
        assert!(
            !output.contains("<xm:sqref>A:A</xm:sqref>") && output.contains(r#"count="1""#),
            "an extended rule the delete emptied was kept: {output}"
        );
        assert!(
            !output.contains("null") && output.matches("<x14:conditionalFormattings").count() == 1,
            "a dropped extended conditional format left its container (or a literal null): {output}"
        );
        assert!(
            reply.contains("extended data validation")
                && reply.contains("extended conditional format"),
            "a dropped extended range is not named: {reply}"
        );
    }

    /// A workbook whose relationships cannot name its sheets is edited where it is
    /// read: the reader falls back to the conventionally numbered worksheet part,
    /// and the edit does the same instead of refusing a file it can read.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_falls_back_to_the_conventional_sheet_part() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["before"]] }],
                }),
            )
            .await,
        );
        let source = write_fixture(
            &ws,
            "no-rels.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[(
                    "xl/_rels/workbook.xml.rels",
                    br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"#,
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "set_cell", "sheet": "S", "cell": "A1", "value": "after" }],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;
        assert!(
            text.contains("after"),
            "the edit did not reach the sheet the reader reads: {text}"
        );
    }

    /// An insert past the used range writes the empty line it was asked for and
    /// moves no address: the addresses a writer left implicit are not materialized,
    /// and the caveats about a workbook's charts are owed only for a change of the
    /// sheet's content, which this is not.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_insert_past_the_used_range_moves_no_address() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"]] }],
                }),
            )
            .await,
        );
        // The rows carry no `r` of their own — the case a materializing shift would
        // rewrite — and the workbook holds a chart, whose caveat is keyed on a real
        // change.
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1" t="inlineStr"><is><t>one</t></is></c></row><row><c r="A2" t="inlineStr"><is><t>two</t></is></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "addressless.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[
                    ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
                    (
                        "xl/charts/chart1.xml",
                        br#"<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"/>"#,
                    ),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 3 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            !output.contains(r#"<row r="1">"#) && !output.contains(r#"<row r="2">"#),
            "an insert that moved nothing materialized the sheet's addresses: {output}"
        );
        assert!(
            output.contains(r#"<row r="3"/>"#),
            "the inserted line is not in the sheet: {output}"
        );
        assert!(
            !reply.contains("charts"),
            "a chart caveat is raised for a shift that moved nothing: {reply}"
        );
    }

    /// A delete removes content even when it moves no surviving address: deleting
    /// the sheet's last used line leaves nothing to shift, and both the caveat about
    /// a workbook's charts and the parts anchored on the line it took are owed for
    /// the content it removed — the anchored part names a cell that is gone, and the
    /// shift that moved nothing must not keep that quiet.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_names_a_delete_without_a_shift_and_the_parts_anchored_on_it() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"], ["three"], ["four"], ["five"]] }],
                }),
            )
            .await,
        );
        // A product-created sheet carries no `<dimension>`, so deleting its last
        // line is a removal that shifts nothing at all, and the comment is anchored
        // on that line.
        let base = std::fs::read(&created).expect("read base package");
        let anchored = |reference: &str| {
            let comments = format!(
                r#"<comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><authors><author>a</author></authors><commentList><comment ref="{reference}" authorId="0"><text><t>note</t></text></comment></commentList></comments>"#
            );
            with_parts(
                &base,
                &[
                    (
                        "xl/charts/chart1.xml",
                        br#"<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"/>"#,
                    ),
                    (
                        "xl/worksheets/_rels/sheet1.xml.rels",
                        br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
                    ),
                    ("xl/comments1.xml", comments.as_bytes()),
                ],
            )
        };
        let source = write_fixture(&ws, "chart.xlsx", &anchored("A5"));
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 5 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            !output.contains("A5"),
            "the deleted line is still in the sheet: {output}"
        );
        assert!(
            reply.contains("charts"),
            "a delete that took the sheet's last line says nothing about its chart: {reply}"
        );
        assert!(
            reply.contains("comments"),
            "a delete that took the sheet's last line says nothing about the comment on it: {reply}"
        );

        // The same delete with the comment on a line it does not touch: the chart
        // caveat is still owed, and no part is claimed for a cell that stayed.
        let elsewhere = write_fixture(&ws, "chart-anchored.xlsx", &anchored("A3"));
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": elsewhere.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 5 }],
            }),
        )
        .await;
        assert!(
            reply.contains("charts"),
            "a delete that took content says nothing about its chart: {reply}"
        );
        assert!(
            !reply.contains("comments"),
            "a delete below the comment named a part the shift did not move: {reply}"
        );
    }

    /// A `<brk id>` names the BOUNDARY a manual page break sits at — the number of
    /// lines above it — not the line it is written for: a break at or below the line
    /// that went moves up with it, one above it stays, an insert pushes one at or
    /// beyond the point down, and a break at the sheet's own top cannot move past it.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_a_page_break_by_its_boundary() {
        use std::fmt::Write as _;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"], ["three"], ["four"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let rows = |last: u32| -> String {
            (1..=last).fold(String::new(), |mut acc, n| {
                write!(
                    acc,
                    r#"<row r="{n}"><c r="A{n}" t="inlineStr"><is><t>v{n}</t></is></c></row>"#
                )
                .expect("write");
                acc
            })
        };
        let breaks = |id: u32| {
            format!(
                r#"<rowBreaks count="1" manualBreakCount="1"><brk id="{id}" max="16383" man="1"/></rowBreaks>"#
            )
        };
        let sheet = |body: String| {
            format!(
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{body}</sheetData></worksheet>"#
            )
        };
        let with_sheet =
            |body: String| with_parts(&base, &[("xl/worksheets/sheet1.xml", body.as_bytes())]);
        // Rows 1-4 with a break between rows 3 and 4: three rows above it.
        let source = write_fixture(
            &ws,
            "breaks.xlsx",
            &with_sheet(sheet(format!("{}{}", rows(4), breaks(3)))),
        );
        let top = write_fixture(
            &ws,
            "breaks-top.xlsx",
            &with_sheet(sheet(format!("{}{}", rows(1), breaks(1)))),
        );

        for (row, op, want, what) in [
            (
                3,
                "delete_row",
                r#"id="2""#,
                "a break below the deleted line moves up",
            ),
            (
                3,
                "insert_row",
                r#"id="4""#,
                "a break at the insert point moves down",
            ),
            (
                4,
                "delete_row",
                r#"id="3""#,
                "a break above the deleted line stays",
            ),
        ] {
            let reply = run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "edited",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": op, "sheet": "S", "row": row }],
                }),
            )
            .await;
            let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
            assert!(output.contains(want), "{what}: the break reads {output}");
            assert_eq!(
                output.matches("<brk ").count(),
                1,
                "the break was lost by {op} {row}: {output}"
            );
        }

        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": top.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"id="1""#),
            "a break at the sheet's top left the grid: {output}"
        );
    }

    /// A rule element that names no range is content the edit never named, so a
    /// shift leaves it as it is; a rule whose range the delete took goes whole, and
    /// so do the container that held it and the `<ext>`/`<extLst>` wrappers around
    /// that — an emptied wrapper carries nothing.
    #[expect(clippy::too_many_lines)] // reason: a nameless rule, a dropped one and one dropped beside a surviving wrapper
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_keeps_a_rule_that_named_no_range_and_drops_the_emptied_wrappers() {
        use std::fmt::Write as _;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"], ["three"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let rows: String = (1..=3).fold(String::new(), |mut acc, n| {
            write!(
                acc,
                r#"<row r="{n}"><c r="A{n}" t="inlineStr"><is><t>v{n}</t></is></c><c r="C{n}" t="inlineStr"><is><t>w{n}</t></is></c></row>"#
            )
            .expect("write");
            acc
        });
        let rule = |sqref: &str| {
            format!(
                r#"<extLst><ext uri="{{78C0D931-6437-407d-A8EE-F0AAD7539E65}}"><x14:conditionalFormattings xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:conditionalFormatting><x14:cfRule type="expression" priority="1" id="{{AAAA}}"><xm:f>TRUE</xm:f></x14:cfRule>{sqref}</x14:conditionalFormatting></x14:conditionalFormattings></ext></extLst>"#
            )
        };
        let sheet = |body: String| {
            format!(
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{rows}</sheetData>{body}</worksheet>"#
            )
        };

        // A rule naming no range at all: nothing to move and nothing the edit
        // covered, so the element stays.
        let nameless = write_fixture(
            &ws,
            "nameless.xlsx",
            &with_parts(
                &base,
                &[(
                    "xl/worksheets/sheet1.xml",
                    sheet(rule("<xm:sqref/>")).as_bytes(),
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": nameless.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains("<x14:conditionalFormatting>"),
            "a rule naming no range was dropped by the shift: {output}"
        );
        assert!(
            !reply.contains("gone with it"),
            "a rule naming no range was reported as a loss: {reply}"
        );

        // A rule over the column the delete takes: the rule, its container and both
        // wrappers go, and the answer names the drop.
        let dropped = write_fixture(
            &ws,
            "dropped-rule.xlsx",
            &with_parts(
                &base,
                &[(
                    "xl/worksheets/sheet1.xml",
                    sheet(rule("<xm:sqref>C1:C2</xm:sqref>")).as_bytes(),
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": dropped.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "C" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        for gone in ["x14:conditionalFormatting", "<ext ", "<extLst>"] {
            assert!(
                !output.contains(gone),
                "the emptied wrapper {gone} was left behind: {output}"
            );
        }
        assert!(
            reply.contains("gone with it"),
            "the dropped rule is not named: {reply}"
        );

        // A wrapper the delete empties goes while one beside it stays: a
        // self-closing `<ext/>` the edit never named survives, so the `<extLst>`
        // still holds a wrapper and stays. A wrapper written without a `uri`
        // attribute is no different from one with it: it goes with its emptied
        // container rather than being left as an empty `<ext></ext>`.
        let beside = write_fixture(
            &ws,
            "beside-emptied.xlsx",
            &with_parts(
                &base,
                &[(
                    "xl/worksheets/sheet1.xml",
                    sheet(
                        r#"<extLst><ext uri="{AAA}"/><ext><x14:conditionalFormattings xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:conditionalFormatting><x14:cfRule type="expression" priority="1" id="{AAAA}"><xm:f>TRUE</xm:f></x14:cfRule><xm:sqref>C1:C2</xm:sqref></x14:conditionalFormatting></x14:conditionalFormattings></ext></extLst>"#
                            .to_string(),
                    )
                    .as_bytes(),
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": beside.to_string_lossy(),
                "edits": [{ "op": "delete_column", "sheet": "S", "column": "C" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<ext uri="{AAA}"/>"#),
            "a self-closing wrapper the edit never named was dropped: {output}"
        );
        assert!(
            output.contains("<extLst>") && output.contains("</extLst>"),
            "the wrapper still holding an extension was dropped with its emptied twin: {output}"
        );
        assert!(
            !output.contains("<ext></ext>") && !output.contains("x14:conditionalFormatting"),
            "an emptied wrapper written without a uri was left behind: {output}"
        );
    }

    /// An extension payload the edit never named — a plain-text `<ext>` beside a
    /// real extended conditional format — is content, not something a shift may
    /// touch: the payload and its `<ext>` stay byte-identical, the `<extLst>` that
    /// still holds both stays, and only the rule's own `<xm:sqref>` moves.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_keeps_an_extension_payload_the_edit_never_named() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"], ["three"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>v1</t></is></c><c r="B1" t="inlineStr"><is><t>w1</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>v2</t></is></c><c r="B2" t="inlineStr"><is><t>w2</t></is></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>v3</t></is></c><c r="B3" t="inlineStr"><is><t>w3</t></is></c></row></sheetData><extLst><ext uri="{AAA}">plain text payload</ext><ext uri="{78C0D931-6437-407d-A8EE-F0AAD7539E65}"><x14:conditionalFormattings xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:conditionalFormatting><x14:cfRule type="expression" priority="1" id="{AAAA}"><xm:f>TRUE</xm:f></x14:cfRule><xm:sqref>B1:B2</xm:sqref></x14:conditionalFormatting></x14:conditionalFormattings></ext></extLst></worksheet>"#;
        let source = write_fixture(
            &ws,
            "plain-ext.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<ext uri="{AAA}">plain text payload</ext>"#),
            "the extension payload the edit never named was rewritten: {output}"
        );
        assert!(
            output.contains("<xm:sqref>B2:B3</xm:sqref>"),
            "the extended rule's range did not move with the shift: {output}"
        );
        assert!(
            output.contains("<extLst>") && output.contains("</extLst>"),
            "the extension list holding an untouched payload was dropped: {output}"
        );
    }

    /// The `<xm:sqref>` and `<xm:f>` an extended rule names its range and writes
    /// its formula in are read as the elements' own text, not by subtracting a
    /// close tag spelled out again: close tags carrying the whitespace XML allows
    /// before the `>` move both like any other, where measuring the text off a
    /// literal `</xm:sqref>` writes a part no reader may hold and leaves the
    /// formula naming a cell the shift moved.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_an_extended_rule_whose_close_tags_carry_whitespace() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>v1</t></is></c><c r="B1"><v>1</v></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>v2</t></is></c><c r="B2"><v>2</v></c></row></sheetData><extLst><ext uri="{78C0D931-6437-407d-A8EE-F0AAD7539E65}"><x14:conditionalFormattings xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><x14:conditionalFormatting><x14:cfRule type="expression" priority="1" id="{AAAA}"><xm:f>B1&gt;0</xm:f ></x14:cfRule><xm:sqref>B1:B2</xm:sqref ></x14:conditionalFormatting></x14:conditionalFormattings></ext></extLst></worksheet>"#;
        let source = write_fixture(
            &ws,
            "spaced-sqref.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains("<xm:sqref>B2:B3</xm:sqref>"),
            "the extended rule's range did not move with the shift: {output}"
        );
        assert!(
            output.contains("<xm:f>B2&gt;0</xm:f>"),
            "the extended rule's formula still names the old cell: {output}"
        );
        assert!(
            !output.contains("B1:B2<"),
            "the range was cut short by the close tag's whitespace: {output}"
        );
    }

    /// A sheet that holds an empty `<extLst>` keeps it through a shift that changes
    /// the sheet — in either spelling, because the pair `<extLst></extLst>` holds
    /// nothing just as a `<extLst/>` does, and an element that held nothing is not
    /// this pass's to remove. Only a list the pass EMPTIED goes with its last
    /// wrapper (the schema requires a child), which the emptied-wrapper test covers.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_leaves_an_empty_ext_lst_alone() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one"], ["two"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        for (name, spelling) in [
            ("empty-ext-lst.xlsx", "<extLst/>"),
            ("empty-ext-lst-paired.xlsx", "<extLst></extLst>"),
        ] {
            let sheet = format!(
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c></row></sheetData>{spelling}</worksheet>"#
            );
            let source = write_fixture(
                &ws,
                name,
                &with_parts(&base, &[("xl/worksheets/sheet1.xml", sheet.as_bytes())]),
            );
            let reply = run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "edited",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
                }),
            )
            .await;
            let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
            assert!(
                output.contains(spelling),
                "an empty extension list ({spelling}) was rewritten by a shift that changed the sheet: {output}"
            );
            assert!(
                output.contains(r#"<row r="1"/>"#),
                "the shift did not change the sheet, so the test proves nothing: {output}"
            );
        }
    }

    /// A counted container may hold a member this pass never names — the `extLst`
    /// the schema allows inside `<ignoredErrors>`: a shift moves the entry's range
    /// where it stands and leaves that member, and no `count` is invented for a
    /// container whose schema declares none.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_keeps_a_member_of_a_container_it_does_not_name() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["one", "two"], ["three", "four"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><ignoredErrors><ignoredError sqref="B1:B2" numberStoredAsText="1"/><extLst><ext uri="{AAA}">keep me</ext></extLst></ignoredErrors></worksheet>"#;
        let source = write_fixture(
            &ws,
            "ignored-errors.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<ignoredError sqref="B2:B3""#),
            "the ignored error's range did not move with the shift: {output}"
        );
        assert!(
            output.contains(r#"<extLst><ext uri="{AAA}">keep me</ext></extLst>"#),
            "a member of the container the edit never named was rewritten: {output}"
        );
        assert!(
            output.contains("<ignoredErrors>") && !output.contains("ignoredErrors count="),
            "a count was invented for a container whose schema declares none: {output}"
        );
    }

    /// A formula that a shift cannot rewrite because it sits in ANOTHER sheet is
    /// named: the reference it qualifies with this sheet's name keeps naming cells
    /// that moved, and the other part is left byte-identical. A formula naming only
    /// the sheet it sits on is not this shift's business.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_names_a_formula_another_sheet_writes_about_this_one() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "Data", "rows": [[1], [2], [3]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let workbook = part_text_bytes(&base, "xl/workbook.xml").replace(
            "</sheets>",
            r#"<sheet name="Totals" sheetId="2" r:id="rId9"/></sheets>"#,
        );
        let rels = part_text_bytes(&base, "xl/_rels/workbook.xml.rels").replace(
            "</Relationships>",
            r#"<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        );
        // One cell names the shifted sheet's range, another only its own.
        let totals = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B1"/><sheetData><row r="1"><c r="A1"><f>SUM(Data!A1:A3)</f><v>6</v></c><c r="B1"><f>SUM(A1:A1)</f><v>1</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "cross-sheet.xlsx",
            &with_parts(
                &base,
                &[
                    ("xl/workbook.xml", workbook.as_bytes()),
                    ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
                    ("xl/worksheets/sheet2.xml", totals.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "Data", "row": 1 }],
            }),
        )
        .await;
        let output = single(&reply);
        assert!(
            reply.contains("other sheets' formulas"),
            "a formula another sheet writes about this one is not named: {reply}"
        );
        assert_eq!(
            part_text(&output, "xl/worksheets/sheet2.xml"),
            totals,
            "the other sheet's part was rewritten"
        );

        // Shifting the other sheet: its own formula's reference to this one is not
        // this shift's business, and its bare reference is not reported either.
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "Totals", "row": 1 }],
            }),
        )
        .await;
        assert!(
            !reply.contains("other sheets' formulas"),
            "a formula naming another sheet is claimed by the wrong shift: {reply}"
        );
    }

    /// A reference the file qualified with a sheet name carrying punctuation is
    /// spelled escaped in the part (`'R&amp;D'!…`) while the workbook's own name is
    /// not: the qualifier is unescaped before it is compared, so the rule moves
    /// with the shift and another sheet's formula naming the sheet is named, its
    /// part left byte-identical.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_a_reference_qualified_with_an_escaped_sheet_name() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "R&D", "rows": [["a"], ["b"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let workbook = part_text_bytes(&base, "xl/workbook.xml").replace(
            "</sheets>",
            r#"<sheet name="Totals" sheetId="2" r:id="rId9"/></sheets>"#,
        );
        let rels = part_text_bytes(&base, "xl/_rels/workbook.xml.rels").replace(
            "</Relationships>",
            r#"<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><conditionalFormatting sqref="B1:B2"><cfRule type="expression" dxfId="0" priority="1"><formula>COUNTIF('R&amp;D'!$A$1:$A$2,1)</formula></cfRule></conditionalFormatting></worksheet>"#;
        let totals = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1"><f>SUM('R&amp;D'!A1:A2)</f><v>3</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "escaped-sheet.xlsx",
            &with_parts(
                &base,
                &[
                    ("xl/workbook.xml", workbook.as_bytes()),
                    ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
                    ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
                    ("xl/worksheets/sheet2.xml", totals.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "R&D", "row": 1 }],
            }),
        )
        .await;
        let out = single(&reply);
        let output = part_text(&out, "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r"<formula>COUNTIF('R&amp;D'!$A$2:$A$3,1)</formula>"),
            "a reference qualified with an escaped sheet name did not move: {output}"
        );
        assert!(
            output.contains(r#"sqref="B2:B3""#),
            "the rule's range did not move: {output}"
        );
        assert!(
            reply.contains("other sheets' formulas"),
            "another sheet's formula naming this one is not named: {reply}"
        );
        assert_eq!(
            part_text(&out, "xl/worksheets/sheet2.xml"),
            totals,
            "the other sheet's part was rewritten"
        );
    }

    /// A sheet name needs no quoting when it carries no space or punctuation, so
    /// Excel writes a same-sheet validation list as `Данные!$B$2:$B$4`: the bare
    /// qualifier takes letters of any script, so the reference moves with the shift
    /// and another sheet's formula naming the sheet is named, its part untouched.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_moves_a_reference_qualified_with_a_non_ascii_sheet_name() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "Данные", "rows": [["a"], ["b"], ["c"], ["d"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let workbook = part_text_bytes(&base, "xl/workbook.xml").replace(
            "</sheets>",
            r#"<sheet name="Totals" sheetId="2" r:id="rId9"/></sheets>"#,
        );
        let rels = part_text_bytes(&base, "xl/_rels/workbook.xml.rels").replace(
            "</Relationships>",
            r#"<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:C4"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row><row r="3"><c r="A3"><v>5</v></c></row><row r="4"><c r="A4"><v>6</v></c></row></sheetData><dataValidations count="1"><dataValidation type="list" sqref="C1:C4"><formula1>Данные!$B$2:$B$4</formula1></dataValidation></dataValidations></worksheet>"#;
        let totals = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1"><f>SUM(Данные!A1:A2)</f><v>3</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "non-ascii-sheet.xlsx",
            &with_parts(
                &base,
                &[
                    ("xl/workbook.xml", workbook.as_bytes()),
                    ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
                    ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
                    ("xl/worksheets/sheet2.xml", totals.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "Данные", "row": 1 }],
            }),
        )
        .await;
        let out = single(&reply);
        let output = part_text(&out, "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r"<formula1>Данные!$B$3:$B$5</formula1>"),
            "a reference qualified with a non-ascii sheet name did not move: {output}"
        );
        assert!(
            output.contains(r#"<dataValidation type="list" sqref="C2:C5">"#),
            "the validation's range did not move: {output}"
        );
        assert!(
            reply.contains("other sheets' formulas"),
            "another sheet's formula naming this one is not named: {reply}"
        );
        assert_eq!(
            part_text(&out, "xl/worksheets/sheet2.xml"),
            totals,
            "the other sheet's part was rewritten"
        );
    }

    /// A cell's own TEXT is content, not a formula: a string that reads like a
    /// qualified reference raises no note about another sheet's formulas.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_does_not_read_a_cell_text_as_another_sheet_formula() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"], ["two"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let workbook = part_text_bytes(&base, "xl/workbook.xml").replace(
            "</sheets>",
            r#"<sheet name="Totals" sheetId="2" r:id="rId9"/></sheets>"#,
        );
        let rels = part_text_bytes(&base, "xl/_rels/workbook.xml.rels").replace(
            "</Relationships>",
            r#"<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        );
        // The cell's text reads like a qualified reference, but it is a string
        // rather than a `<f>`: the scan for another sheet's formulas must not see it.
        let totals = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t xml:space="preserve">see S!A1:A2 for the numbers</t></is></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "text-not-formula.xlsx",
            &with_parts(
                &base,
                &[
                    ("xl/workbook.xml", workbook.as_bytes()),
                    ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
                    ("xl/worksheets/sheet2.xml", totals.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let out = single(&reply);
        assert!(
            !reply.contains("other sheets' formulas"),
            "a cell's own text was read as another sheet's formula: {reply}"
        );
        assert_eq!(
            part_text(&out, "xl/worksheets/sheet2.xml"),
            totals,
            "the other sheet's part was rewritten"
        );
    }

    /// A cell's own formula keeps its text — rewriting its references is a
    /// spreadsheet engine's job — and the reply names it rather than passing the
    /// stale reference off as preserved. A formula naming nothing the shift moved
    /// is left out of that count: its text means exactly what it did.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_leaves_a_cell_formula_and_names_it() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed", "x"], ["y", "z"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><f>A2+A3</f><v>7</v></c><c r="B1"><f>A1*2</f><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "cell-formula.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 2 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains("<f>A2+A3</f>"),
            "a cell's own formula is not rewritten: {output}"
        );
        assert!(
            reply.contains("1 formula was left as it is"),
            "the cell formula was not named in the reply: {reply}"
        );
        assert!(
            !reply.contains("2 formulas"),
            "a formula naming nothing the shift moved is counted: {reply}"
        );
    }

    /// A shifted address stays inside the sheet's grid: a `<col>` covering the
    /// whole grid keeps its last column, and a page break on the grid's last line
    /// stays there instead of naming a line the grid does not have.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_keeps_a_shifted_address_inside_the_grid() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><cols><col min="1" max="16384" width="9"/></cols><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c></row></sheetData><rowBreaks count="1" manualBreakCount="1"><brk id="1048576" max="16383" man="1"/></rowBreaks><colBreaks count="1" manualBreakCount="1"><brk id="16384" max="1048575" man="1"/></colBreaks></worksheet>"#;
        let source = write_fixture(
            &ws,
            "grid.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_column", "sheet": "S", "column": "A" }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            output.contains(r#"<col min="2" max="16384" width="9"/>"#),
            "the column range was pushed past the grid: {output}"
        );
        assert!(
            !output.contains("XFE"),
            "a cell named a column the grid does not have: {output}"
        );
        assert!(
            output.contains(r#"<brk id="16384" max="1048575" man="1"/>"#),
            "a column break was pushed past the grid: {output}"
        );

        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            !output.contains(r#"r="1048577""#) && !output.contains(r#"<brk id="1048577""#),
            "a row was pushed past the grid: {output}"
        );
        assert!(
            output.contains(r#"<brk id="1048576" max="16383" man="1"/>"#),
            "the row break of the last row must stay inside the grid: {output}"
        );
    }

    /// An insert that would push a line past the sheet's grid is refused: the
    /// pushed line would have to keep the last line's own address, so the caller
    /// is told to insert where there is room instead of being handed an ambiguous
    /// sheet.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_an_insert_that_would_pass_the_grid() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        for (name, sheet, edit) in [
            (
                "last-row.xlsx",
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1048576"><c r="A1048576"><v>1</v></c></row></sheetData></worksheet>"#,
                json!({ "op": "insert_row", "sheet": "S", "row": 1_048_576 }),
            ),
            (
                "last-column.xlsx",
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="XFD1"><v>1</v></c></row></sheetData></worksheet>"#,
                json!({ "op": "insert_column", "sheet": "S", "column": "XFD" }),
            ),
        ] {
            let source = write_fixture(
                &ws,
                name,
                &with_parts(
                    &std::fs::read(&created).expect("read base package"),
                    &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
                ),
            );
            let error = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "xlsx_edit", "file_name": "edited",
                        "path": source.to_string_lossy(),
                        "edits": [edit],
                    }),
                )
                .await
                .expect_err("an insert the grid cannot hold must be refused");
            assert!(
                error.to_string().contains("past the sheet's grid"),
                "{name}: the refusal does not name the grid: {error}"
            );
        }
    }

    /// A `set_cell` widens `<dimension ref>` so it covers the written cell, the
    /// way a row or column shift keeps the ref level with the cells, while a write
    /// inside the declared reach leaves the ref alone.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_set_cell_widens_the_dimension_it_writes_outside() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "dimension.xlsx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );

        let outside = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "outside",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "set_cell", "sheet": "S", "cell": "D5", "value": 9 }],
            }),
        )
        .await;
        let widened = part_text(&single(&outside), "xl/worksheets/sheet1.xml");
        assert!(
            widened.contains(r#"<dimension ref="A1:D5"/>"#),
            "the dimension does not cover the written cell: {widened}"
        );

        let inside = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "inside",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 5 }],
            }),
        )
        .await;
        let kept = part_text(&single(&inside), "xl/worksheets/sheet1.xml");
        assert!(
            kept.contains(r#"<dimension ref="A1:B2"/>"#),
            "a write inside the reach changed the dimension: {kept}"
        );
    }

    /// A shift leaves the workbook's defined names and the sheet's table ranges
    /// pointing at the old cells, and a signed package's signature no longer
    /// matches: the reply names each rather than staying silent.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_names_the_stale_references_and_signature_it_cannot_rewrite() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["seed"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let workbook = part_text_bytes(&base, "xl/workbook.xml").replace(
            "</workbook>",
            "<definedNames><definedName name=\"Total\">S!$B$2</definedName></definedNames></workbook>",
        );
        let sheet = part_text_bytes(&base, "xl/worksheets/sheet1.xml").replace(
            "</worksheet>",
            "<tableParts count=\"1\"><tablePart r:id=\"rId9\"/></tableParts>\
             <legacyDrawing r:id=\"rId3\"/><drawing r:id=\"rId4\"/></worksheet>",
        );
        let package = with_parts(
            &base,
            &[
                ("xl/workbook.xml", workbook.as_bytes()),
                ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
                (
                    "xl/tables/table1.xml",
                    br#"<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" id="1" name="T" displayName="T" ref="A1:B2"/>"#,
                ),
                // The sheet's cell-anchored parts: a comment's own `ref`, the VML
                // shape it is drawn with and a floating drawing's anchor all name
                // a cell a row or column shift moves, and none is rewritten.
                (
                    "xl/worksheets/_rels/sheet1.xml.rels",
                    br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/vmlDrawing" Target="../drawings/vmlDrawing1.vml"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/></Relationships>"#,
                ),
                (
                    "xl/comments1.xml",
                    br#"<comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><authors><author>a</author></authors><commentList><comment ref="B2" authorId="0"><text><t>note</t></text></comment></commentList></comments>"#,
                ),
                (
                    "xl/drawings/drawing1.xml",
                    br#"<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"><xdr:twoCellAnchor><xdr:from><xdr:col>1</xdr:col><xdr:row>1</xdr:row></xdr:from><xdr:to><xdr:col>3</xdr:col><xdr:row>3</xdr:row></xdr:to></xdr:twoCellAnchor></xdr:wsDr>"#,
                ),
                (
                    "_xmlsignatures/sig1.xml",
                    br#"<Signature xmlns="http://www.w3.org/2000/09/xmldsig#"><SignedInfo/></Signature>"#,
                ),
                (
                    "_xmlsignatures/_rels/origin.sigs.rels",
                    br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"#,
                ),
            ],
        );
        let source = write_fixture(&ws, "signed.xlsx", &package);
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "insert_row", "sheet": "S", "row": 1 }],
            }),
        )
        .await;
        assert!(
            reply.contains("defined names"),
            "the stale defined names are not named: {reply}"
        );
        assert!(
            reply.contains("table ranges"),
            "the stale table ranges are not named: {reply}"
        );
        for anchored in ["comments", "comment shapes", "drawing anchors"] {
            assert!(
                reply.contains(anchored),
                "the cell-anchored {anchored} a shift left in place are not named: {reply}"
            );
        }
        assert!(
            reply.contains("digital signature"),
            "the broken signature is not named: {reply}"
        );
    }

    /// A part whose own addresses the shift moved with the cells is as valid as it
    /// was: a comment, a table range, a defined name and a drawing anchor all above
    /// the deleted line are not named, because nothing about them went stale.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_names_no_part_the_shift_did_not_move() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [{ "name": "S", "rows": [["a"], ["b"], ["c"], ["d"], ["e"]] }],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let workbook = part_text_bytes(&base, "xl/workbook.xml").replace(
            "</workbook>",
            "<definedNames><definedName name=\"Total\">S!$A$1</definedName></definedNames></workbook>",
        );
        let sheet = part_text_bytes(&base, "xl/worksheets/sheet1.xml").replace(
            "</worksheet>",
            "<tableParts count=\"1\"><tablePart r:id=\"rId9\"/></tableParts>\
             <legacyDrawing r:id=\"rId3\"/><drawing r:id=\"rId4\"/></worksheet>",
        );
        let package = with_parts(
            &base,
            &[
                ("xl/workbook.xml", workbook.as_bytes()),
                ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
                (
                    "xl/worksheets/_rels/sheet1.xml.rels",
                    br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/vmlDrawing" Target="../drawings/vmlDrawing1.vml"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table" Target="../tables/table1.xml"/></Relationships>"#,
                ),
                (
                    "xl/comments1.xml",
                    br#"<comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><authors><author>a</author></authors><commentList><comment ref="B1" authorId="0"><text><t>note</t></text></comment></commentList></comments>"#,
                ),
                (
                    "xl/drawings/vmlDrawing1.vml",
                    br#"<xml xmlns:v="urn:schemas-microsoft-com:vml"><v:shape id="s"/></xml>"#,
                ),
                (
                    "xl/drawings/drawing1.xml",
                    br#"<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"><xdr:twoCellAnchor><xdr:from><xdr:col>1</xdr:col><xdr:row>0</xdr:row></xdr:from><xdr:to><xdr:col>3</xdr:col><xdr:row>1</xdr:row></xdr:to></xdr:twoCellAnchor></xdr:wsDr>"#,
                ),
                (
                    "xl/tables/table1.xml",
                    br#"<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" id="1" name="T" displayName="T" ref="A1:B2"/>"#,
                ),
            ],
        );
        let source = write_fixture(&ws, "anchored-above.xlsx", &package);
        // The delete takes row 3 and moves rows 4-5 up, so the shift really changed
        // the sheet — every part above it names a cell it did not touch.
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_row", "sheet": "S", "row": 3 }],
            }),
        )
        .await;
        let output = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            !output.contains(">c<") && output.contains(">e<"),
            "the delete did not take the line it named: {output}"
        );
        assert!(
            !reply.contains("pointing at the old cells"),
            "a part the shift moved with its cells is named as stale: {reply}"
        );
    }

    /// An Excel-shaped styles part whose second `<cellXfs>` entry is NOT the
    /// default: a custom number format, a font that is not the workbook's first,
    /// a solid fill, a bordered cell and an `<xf>` carrying its own `<alignment>`
    /// and `<protection>` children. A format naming one property has to clone
    /// this entry for exactly that property.
    const FORMAT_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.00"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><i/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFFF00"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border/><border><left style="thin"><color indexed="64"/></left><top style="thin"><color indexed="64"/></top></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="164" fontId="1" fillId="1" borderId="1" xfId="0" applyFont="1"><alignment horizontal="right" vertical="top" wrapText="1"/><protection locked="0"/></xf></cellXfs></styleSheet>"#;

    /// The entry `A1` uses in [`FORMAT_STYLES`], so a format can be shown to
    /// leave that entry in the part untouched.
    const FORMAT_XF: &str = r#"<xf numFmtId="164" fontId="1" fillId="1" borderId="1" xfId="0" applyFont="1"><alignment horizontal="right" vertical="top" wrapText="1"/><protection locked="0"/></xf>"#;

    /// A sheet whose `A1` uses [`FORMAT_STYLES`]'s second entry and whose `B1` is
    /// an ordinary cell beside it.
    const FORMAT_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B1"/><sheetData><row r="1"><c r="A1" s="1" t="inlineStr"><is><t>KEEP</t></is></c><c r="B1" s="0"><v>7</v></c></row></sheetData></worksheet>"#;

    /// A format names one property and the cell's own `<xf>` is cloned for it: the
    /// number format, the fill, the borders and the `xfId` stay the old entry's,
    /// the font is a new entry built from the cell's own, the old `<xf>`'s
    /// child-bearing children (`<alignment>`, `<protection>`) are kept byte for
    /// byte, and the entry the cell came with is still in the part. A repeat of the
    /// same format in one call reuses the entries it made rather than appending
    /// them twice.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: one format's clone, its repeats and the entries it left, one case each
    async fn xlsx_edit_format_cells_changes_only_the_property_it_names() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "styled.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[
                    ("xl/styles.xml", FORMAT_STYLES.as_bytes()),
                    ("xl/worksheets/sheet1.xml", FORMAT_SHEET.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");

        // A1 keeps its own value bytes and moves to the appended entry.
        assert!(
            sheet.contains(r#"<c r="A1" s="2" t="inlineStr"><is><t>KEEP</t></is></c>"#),
            "the cell did not keep its value and take the new entry: {sheet}"
        );
        assert!(
            sheet.contains(r#"<c r="B1" s="0"><v>7</v></c>"#),
            "a cell outside the format changed: {sheet}"
        );

        let xfs = element_children(&styles, "cellXfs", "xf");
        assert_eq!(
            xfs.len(),
            3,
            "expected one entry beside the two the cell came with: {styles}"
        );
        let (before, after) = (xfs[1], xfs[2]);
        // Everything the request did not name is the cell's own entry's.
        for name in ["numFmtId", "fillId", "borderId", "xfId"] {
            assert_eq!(
                attribute_of(after, name),
                attribute_of(before, name),
                "the new entry changed {name}: {after}"
            );
        }
        assert_ne!(
            attribute_of(after, "fontId"),
            attribute_of(before, "fontId"),
            "the new entry reused the cell's font id: {after}"
        );
        assert_eq!(attribute_of(after, "applyFont"), "1", "{after}");
        // The children of the cell's own entry are its own bytes.
        assert!(
            after.contains(
                r#"<alignment horizontal="right" vertical="top" wrapText="1"/><protection locked="0"/>"#
            ),
            "a child of the cell's own entry was not kept: {after}"
        );
        // The font it points at is the cell's own with `<b/>` written in EG_Font order.
        let fonts = element_children(&styles, "fonts", "font");
        let font = fonts[attribute_of(after, "fontId")
            .parse::<usize>()
            .expect("a font index")];
        assert_eq!(
            font,
            r#"<font><b/><i/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font>"#,
            "the new font is not the cell's own plus <b/>: {font}"
        );
        // The entries the cell came with are still there, unchanged.
        assert_eq!(
            xfs[1], FORMAT_XF,
            "the cell's own entry was rewritten: {styles}"
        );
        assert!(
            styles.contains(r#"<numFmt numFmtId="164" formatCode="0.00"/>"#),
            "the existing number format was rewritten: {styles}"
        );
        assert!(
            styles.contains(
                r#"<font><i/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font>"#
            ),
            "the cell's own font was rewritten: {styles}"
        );

        // A second format naming the same property in the SAME call is a repeat:
        // the entries are reused, so `<cellXfs>` gained exactly what one call needs.
        let twice = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "twice",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_cells", "sheet": "S", "range": "A1", "bold": true },
                    { "op": "format_cells", "sheet": "S", "range": "A1", "bold": true },
                ],
            }),
        )
        .await;
        let twice_styles = part_text(&single(&twice), "xl/styles.xml");
        assert_eq!(
            element_children(&twice_styles, "cellXfs", "xf").len(),
            3,
            "a repeated format appended its entry twice: {twice_styles}"
        );
        assert_eq!(
            element_children(&twice_styles, "fonts", "font").len(),
            3,
            "a repeated format appended its font twice: {twice_styles}"
        );
    }

    /// A cell that names no `s=` renders as the FIRST `<cellXfs>` entry — the base
    /// style a workbook states — so a format naming one property must clone THAT
    /// entry: a hard-coded default would take away the font (or fill, or
    /// alignment) the entry states and move the cell off the look it already had.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_keeps_the_base_style_a_cell_with_no_s_inherits() {
        // Entry 0 is the base a cell with no `s=` renders as and states a bold
        // red font; entry 1 is the neutral stock entry a hard-coded default
        // would reach for.
        const STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>"#;
        const SHEET: &str = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1"><v>7</v></c></row></sheetData></worksheet>"#;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "base-style.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[
                    ("xl/styles.xml", STYLES.as_bytes()),
                    ("xl/worksheets/sheet1.xml", SHEET.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "aligned",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1",
                            "align": "center" }],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let cell = cell_element(&sheet, "A1");
        assert!(
            cell.contains("<v>7</v>"),
            "the format did not leave the value alone: {cell}"
        );
        let xfs = element_children(&styles, "cellXfs", "xf");
        let index: usize = attribute_of(cell, "s")
            .parse()
            .expect("the cell did not take a style index");
        let xf = xfs
            .get(index)
            .unwrap_or_else(|| panic!("no entry at {index}: {styles}"));
        // The font entry 0 states is the cell's own, and the alignment is the one
        // property the request named.
        assert_eq!(
            attribute_of(xf, "fontId"),
            "1",
            "the base font the cell already rendered as was dropped: {xf}"
        );
        assert_eq!(attribute_of(xf, "applyFont"), "1", "{xf}");
        assert_eq!(attribute_of(xf, "applyAlignment"), "1", "{xf}");
        assert!(
            xf.contains(r#"<alignment horizontal="center"/>"#),
            "the alignment was not written: {xf}"
        );
    }

    /// A `<cellXfs>` entry that omits `fontId` takes it from the named style its
    /// `xfId` points at, so a format naming one property must base the new entry on
    /// that style's font: resolving the omitted id to 0 would give the cell the
    /// default font and take away the one it renders as.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_keeps_the_look_a_named_style_gives_a_cell() {
        // The cell xf states no `fontId`; its `xfId` points at a `<cellStyleXfs>`
        // entry whose font is a 20-point blue Georgia — the look the cell renders
        // as — while font 0 is the neutral Calibri 11 a hard-coded default takes.
        const STYLES: &str = r#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><sz val="20"/><color rgb="FF0000FF"/><name val="Georgia"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="1" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>"#;
        const SHEET: &str = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1"><v>7</v></c></row></sheetData></worksheet>"#;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "named-style.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[
                    ("xl/styles.xml", STYLES.as_bytes()),
                    ("xl/worksheets/sheet1.xml", SHEET.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "emboldened",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let cell = cell_element(&sheet, "A1");
        assert!(
            cell.contains("<v>7</v>"),
            "the format did not leave the value alone: {cell}"
        );
        let xfs = element_children(&styles, "cellXfs", "xf");
        let index: usize = attribute_of(cell, "s")
            .parse()
            .expect("the cell did not take a style index");
        let xf = xfs
            .get(index)
            .unwrap_or_else(|| panic!("no entry at {index}: {styles}"));
        let font_id: usize = attribute_of(xf, "fontId")
            .parse()
            .expect("the new entry states no font index");
        let font = element_children(&styles, "fonts", "font")[font_id];
        // The named style's font, made bold — not the neutral font 0.
        assert!(
            font.contains(r#"<name val="Georgia"/>"#),
            "the named style's font was dropped: {font}"
        );
        assert!(
            font.contains(r#"<sz val="20"/>"#),
            "the named style's size was dropped: {font}"
        );
        assert!(
            font.contains(r#"<color rgb="FF0000FF"/>"#),
            "the named style's color was dropped: {font}"
        );
        assert!(font.contains("<b/>"), "the bold flag was not added: {font}");
    }

    /// A cell that names no `s=` of its own takes the style its `<row>` or its
    /// covering `<col>` states, so a format naming one property must materialize
    /// that style on the cell: the row wins over the column, as a reader applies
    /// them, and neither the row's nor the column's font is dropped beside the
    /// fill the request named.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_keeps_the_look_a_cell_inherits_from_its_row_or_column() {
        // Entry 1 is a bold red font (the row's), entry 2 an italic blue one (the
        // column's); entry 0 is the neutral base neither cell names.
        const STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font><font><i/><sz val="11"/><color rgb="FF0000FF"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/><xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs></styleSheet>"#;
        // Row 2 states a style for its cells; column B states one for a cell whose
        // row states none; C2 is outside the edit and must come through untouched.
        const SHEET: &str = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:C3"/><cols><col min="2" max="2" width="12" style="2" customWidth="1"/></cols><sheetData><row r="2" s="1" customFormat="1"><c r="A2"><v>7</v></c><c r="B2"><v>8</v></c><c r="C2" s="0"><v>11</v></c></row><row r="3"><c r="B3"><v>9</v></c></row></sheetData></worksheet>"#;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "inherited.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[
                    ("xl/styles.xml", STYLES.as_bytes()),
                    ("xl/worksheets/sheet1.xml", SHEET.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "filled",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_cells", "sheet": "S", "range": "A2:B2", "fill": "FFF2CC" },
                    { "op": "format_cells", "sheet": "S", "range": "B3", "fill": "FFF2CC" },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let xfs = element_children(&styles, "cellXfs", "xf");
        let xf_at = |address: &str| -> usize {
            attribute_of(cell_element(&sheet, address), "s")
                .parse()
                .expect("a style index")
        };

        // Both cells the row covers take the row's font, not the column's.
        for address in ["A2", "B2"] {
            let xf = xfs[xf_at(address)];
            assert_eq!(
                attribute_of(xf, "fontId"),
                "1",
                "{address} lost the font its row states: {xf}"
            );
            assert_eq!(attribute_of(xf, "applyFill"), "1", "{address}: {xf}");
        }
        // B3's row states none, so it takes the style its covering column states.
        let b3 = xfs[xf_at("B3")];
        assert_eq!(
            attribute_of(b3, "fontId"),
            "2",
            "B3 lost the font its column states: {b3}"
        );
        assert_eq!(attribute_of(b3, "applyFill"), "1", "{b3}");
        assert!(
            styles.contains(r#"<fgColor rgb="FFFFF2CC"/>"#),
            "the fill was not written: {styles}"
        );
        // A cell the edit does not name is byte-identical.
        assert!(
            sheet.contains(r#"<c r="C2" s="0"><v>11</v></c>"#),
            "a cell outside the edit changed: {sheet}"
        );
    }

    /// A `set_cell` that names only a number format must materialize the style the
    /// cell already renders by — its row's, or its covering column's when the row
    /// states none — exactly as `format_cells` does: deriving the new `<cellXfs>`
    /// entry from the cell's own absent `s=` would base it on the default entry and
    /// drop the font, fill or alignment the cell already had.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_set_cell_number_format_keeps_the_look_a_cell_inherits_from_its_row_or_column()
     {
        // Entry 1 is a bold red font (the style row 2 states), entry 2 a yellow
        // fill (the style the covering `<col>` states); entry 0 is the neutral base
        // neither row states.
        const STYLES: &str = r#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFFF00"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/><xf numFmtId="0" fontId="0" fillId="1" borderId="0" xfId="0" applyFill="1"/></cellXfs></styleSheet>"#;
        // B1's row states `customFormat="1"` with no `s=`, the row's own default
        // entry 0, which wins over the covering `<col style="2">`; A2's row states
        // entry 1; C3's row states none, so the covering column's entry 2 applies.
        const SHEET: &str = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:C3"/><cols><col min="2" max="3" style="2" width="9" customWidth="1"/></cols><sheetData><row r="1" customFormat="1"><c r="B1"><v>2</v></c></row><row r="2" s="1" customFormat="1"><c r="A2"><v>3</v></c></row><row r="3"><c r="C3"><v>4</v></c></row></sheetData></worksheet>"#;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "set-cell-inherited.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[
                    ("xl/styles.xml", STYLES.as_bytes()),
                    ("xl/worksheets/sheet1.xml", SHEET.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "numbered",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "set_cell", "sheet": "S", "cell": "B1", "value": 21,
                      "number_format": "0.00" },
                    { "op": "set_cell", "sheet": "S", "cell": "A2", "value": 22,
                      "number_format": "0.00" },
                    { "op": "set_cell", "sheet": "S", "cell": "C3", "value": 23,
                      "number_format": "0.00" },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let xfs = element_children(&styles, "cellXfs", "xf");
        let num_fmts = element_children(&styles, "numFmts", "numFmt");
        let xf_at = |address: &str| -> usize {
            attribute_of(cell_element(&sheet, address), "s")
                .parse()
                .expect("a style index")
        };

        // B1's row states a format of its own but no readable `s=`, so its base is
        // the entry 0 the row still wins with over the covering column.
        let b1 = xfs[xf_at("B1")];
        assert_eq!(
            attribute_of(b1, "fontId"),
            "0",
            "B1 lost the neutral font its row's default entry states: {b1}"
        );
        assert_eq!(
            attribute_of(b1, "fillId"),
            "0",
            "B1 took the covering column's fill over its row's default: {b1}"
        );
        // A2's row states entry 1 (the bold red font) and keeps it.
        let a2 = xfs[xf_at("A2")];
        assert_eq!(
            attribute_of(a2, "fontId"),
            "1",
            "A2 lost the bold red font its row states: {a2}"
        );
        // C3's row states none, so the covering column's entry 2 (the yellow fill)
        // is the base.
        let c3 = xfs[xf_at("C3")];
        assert_eq!(
            attribute_of(c3, "fillId"),
            "1",
            "C3 lost the yellow fill its covering column states: {c3}"
        );

        // Each cell kept the value it was written, and every new entry names the
        // 0.00 code the call asked for.
        for (address, value) in [("B1", "21"), ("A2", "22"), ("C3", "23")] {
            assert!(
                cell_element(&sheet, address).contains(&format!("<v>{value}</v>")),
                "{address} did not keep the value it was written: {sheet}"
            );
            let xf = xfs[xf_at(address)];
            assert_eq!(
                attribute_of(xf, "applyNumberFormat"),
                "1",
                "{address}: {xf}"
            );
            let id = attribute_of(xf, "numFmtId");
            let code = num_fmts
                .iter()
                .find(|entry| attribute_of(entry, "numFmtId") == id)
                .map(|entry| attribute_of(entry, "formatCode"));
            assert_eq!(
                code,
                Some("0.00"),
                "{address} did not take the 0.00 format it asked for: {xf}"
            );
        }
    }

    /// A styled part whose second entry's font is bold, so a `bold:false` has a
    /// `<b/>` to take away.
    const AWAY_STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><i/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFFF00"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border/><border><left style="thin"><color indexed="64"/></left><right style="thin"><color indexed="64"/></right><top style="thin"><color indexed="64"/></top><bottom style="thin"><color indexed="64"/></bottom></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="1" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="bottom" wrapText="1"/></xf></cellXfs></styleSheet>"#;

    /// Four cells sharing [`AWAY_STYLES`]'s styled entry: one format apiece can
    /// take one named property away on each.
    const AWAY_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:D1"/><sheetData><row r="1"><c r="A1" s="1"><v>1</v></c><c r="B1" s="1"><v>2</v></c><c r="C1" s="1"><v>3</v></c><c r="D1" s="1"><v>4</v></c></row></sheetData></worksheet>"#;

    /// A named property can be taken away as well as set: `bold:false` drops the
    /// cell's `<b/>`, `fill:false` points it at the "none" fill, a border side
    /// named false is left as an empty side while its siblings keep their own
    /// bytes, and `wrap:false` removes `wrapText` without touching the cell's
    /// other alignment values.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_takes_a_named_property_away() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "away.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[
                    ("xl/styles.xml", AWAY_STYLES.as_bytes()),
                    ("xl/worksheets/sheet1.xml", AWAY_SHEET.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "away-edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "format_cells", "sheet": "S", "range": "A1", "bold": false },
                    { "op": "format_cells", "sheet": "S", "range": "B1", "fill": false },
                    { "op": "format_cells", "sheet": "S", "range": "C1", "border": { "top": false } },
                    { "op": "format_cells", "sheet": "S", "range": "D1", "wrap": false },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let xfs = element_children(&styles, "cellXfs", "xf");
        let fonts = element_children(&styles, "fonts", "font");
        let fills = element_children(&styles, "fills", "fill");
        let borders = element_children(&styles, "borders", "border");
        let xf_at = |address: &str| -> usize {
            attribute_of(cell_element(&sheet, address), "s")
                .parse()
                .expect("a style index")
        };

        // `bold:false` drops the `<b/>`, off a clone of the cell's own font.
        let font = fonts[attribute_of(xfs[xf_at("A1")], "fontId")
            .parse::<usize>()
            .expect("a font index")];
        assert!(
            !font.contains("<b/>") && font.contains("<i/>"),
            "the font was not un-bolded: {font}"
        );
        assert!(
            fonts[1].contains("<b/>"),
            "the cell's own font was rewritten: {}",
            fonts[1]
        );

        // `fill:false` points the cell at the "none" fill.
        let fill = fills[attribute_of(xfs[xf_at("B1")], "fillId")
            .parse::<usize>()
            .expect("a fill index")];
        assert!(
            fill.contains(r#"patternType="none""#),
            "the fill was not taken away: {fill}"
        );

        // `border.top:false` leaves an empty `<top/>` while its siblings keep their bytes.
        let border = borders[attribute_of(xfs[xf_at("C1")], "borderId")
            .parse::<usize>()
            .expect("a border index")];
        assert!(
            border.contains("<top/>"),
            "the top side was not taken away: {border}"
        );
        for side in ["left", "right", "bottom"] {
            assert!(
                border.contains(&format!(
                    r#"<{side} style="thin"><color indexed="64"/></{side}>"#
                )),
                "the {side} side lost its own bytes: {border}"
            );
        }
        assert!(
            borders[1].contains(r#"<top style="thin"><color indexed="64"/></top>"#),
            "the cell's own border was rewritten: {}",
            borders[1]
        );

        // `wrap:false` removes wrapText while the cell's other alignment values stay.
        let wrapped = xfs[xf_at("D1")];
        assert!(
            wrapped.contains(r#"<alignment horizontal="center" vertical="bottom"/>"#),
            "the alignment was not reduced: {wrapped}"
        );
        assert!(
            !wrapped.contains("wrapText"),
            "wrapText was not removed: {wrapped}"
        );
    }

    /// One format over `A1:C1` names three cells: the two the sheet already holds
    /// and the one the rectangle creates. All three take the same new entry, the
    /// existing cells keep their values, a cell outside the rectangle does not
    /// change, the dimension covers the rectangle, and no widening is reported.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_a_range_writes_every_named_cell_and_no_other() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "range.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "filled",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1:C1", "fill": "FFFF00" }],
            }),
        )
        .await;
        let output = single(&reply);
        let filled = part_text(&output, "xl/worksheets/sheet1.xml");
        let styles = part_text(&output, "xl/styles.xml");
        for address in ["A1", "B1", "C1"] {
            assert_eq!(
                attribute_of(cell_element(&filled, address), "s"),
                "1",
                "{address} did not take the range's entry: {filled}"
            );
        }
        assert!(
            filled.contains(r#"<c r="A1" s="1"><v>1</v></c>"#),
            "A1 lost its value: {filled}"
        );
        assert!(
            filled.contains(r#"<c r="B1" s="1"><v>2</v></c>"#),
            "B1 lost its value: {filled}"
        );
        assert!(
            filled.contains(r#"<c r="C1" s="1"/>"#),
            "the created C1 did not take the style: {filled}"
        );
        assert!(
            filled.contains(r#"<c r="A2"><v>3</v></c>"#),
            "a cell outside the range changed: {filled}"
        );
        assert!(
            filled.contains(r#"<dimension ref="A1:C2"/>"#),
            "the dimension does not cover the formatted range: {filled}"
        );
        assert!(
            styles.contains(r#"<fgColor rgb="FFFFFF00"/>"#),
            "the fill was not written: {styles}"
        );
        assert!(
            !reply.contains("widened"),
            "an unmerged range was reported widened: {reply}"
        );
    }

    /// A number format moves only `s=`: a `format_cells` naming one leaves the
    /// cell's `<f>` and `<v>` byte-identical. A `set_cell` that names a value
    /// still writes that value.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_number_format_leaves_the_value_and_the_formula_alone() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B1"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1" s="0"><f>A1*2</f><v>2</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "formula.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );

        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "formatted",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "B1", "number_format": "0.00" }],
            }),
        )
        .await;
        let output = single(&reply);
        assert!(
            part_text(&output, "xl/worksheets/sheet1.xml")
                .contains(r#"<c r="B1" s="1"><f>A1*2</f><v>2</v></c>"#),
            "the format did not leave the formula and value alone: {}",
            part_text(&output, "xl/worksheets/sheet1.xml")
        );
        assert!(
            part_text(&output, "xl/styles.xml")
                .contains(r#"<numFmt numFmtId="164" formatCode="0.00"/>"#),
            "the number format was not written"
        );

        // A `set_cell` that DOES name a value still writes that value.
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "rewritten",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "set_cell", "sheet": "S", "cell": "B1", "value": 9 }],
            }),
        )
        .await;
        let rewritten = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            rewritten.contains(r#"<c r="B1" s="0"><v>9</v></c>"#),
            "a value write did not replace the formula: {rewritten}"
        );
    }

    /// Clearing a styled, filled cell empties it but keeps its own `s=`: the
    /// value children and their `t` go, the style index stays. A cell that
    /// carried no address of its own is emptied where it stands and stays
    /// addressless, so nothing the caller did not name gains an attribute.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_clear_cell_keeps_the_cells_own_style() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1" s="1"><v>5</v></c><c r="B1" s="1" t="inlineStr"><is><t>x</t></is></c></row><row r="2"><c><v>9</v></c></row></sheetData></worksheet>"#;
        let base = created_xlsx(&ws).await;
        let source = write_fixture(
            &ws,
            "cleared.xlsx",
            &with_parts(&base, &[("xl/worksheets/sheet1.xml", sheet.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "cleared-edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "clear_cell", "sheet": "S", "cell": "A1" },
                    { "op": "clear_cell", "sheet": "S", "cell": "B1" },
                    { "op": "clear_cell", "sheet": "S", "cell": "A2" },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        let cleared = part_text(&output, "xl/worksheets/sheet1.xml");
        assert!(
            cleared.contains(r#"<c r="A1" s="1"/>"#) && cleared.contains(r#"<c r="B1" s="1"/>"#),
            "a cleared cell lost its own style index: {cleared}"
        );
        assert!(
            !cleared.contains("<v>") && !cleared.contains("<is>") && !cleared.contains("</t>"),
            "a cleared cell kept its value: {cleared}"
        );
        assert!(
            cleared.contains(r#"<row r="2"><c/></row>"#),
            "the addressless cell was not emptied in place: {cleared}"
        );
        assert_eq!(
            part_bytes(&source, "xl/styles.xml"),
            part_bytes(&output, "xl/styles.xml"),
            "clearing a cell rewrote the styles part"
        );
    }

    /// A column format on `B` splits the `<col>` covering it: the spans on either
    /// side keep every attribute they had, `B` takes the new width, a sheet with
    /// no `<cols>` gains one directly before `<sheetData>`, and writing the same
    /// width again round-trips byte-identically.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_column_width_splits_a_covering_entry() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let covering = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cols><col min="1" max="4" width="10" customWidth="1" style="2" hidden="1"/></cols><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "covered.xlsx",
            &with_parts(&base, &[("xl/worksheets/sheet1.xml", covering.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "split",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "column": "B", "width": 20 }],
            }),
        )
        .await;
        let output = single(&reply);
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains(concat!(
                r#"<cols><col min="1" max="1" width="10" customWidth="1" style="2" hidden="1"/>"#,
                r#"<col min="2" max="2" width="20" customWidth="1" style="2" hidden="1"/>"#,
                r#"<col min="3" max="4" width="10" customWidth="1" style="2" hidden="1"/></cols>"#,
            )),
            "the covering entry was not split around B: {sheet}"
        );

        // A sheet with no <cols> gains one directly before <sheetData>.
        let plain = write_fixture(&ws, "bare.xlsx", &base);
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "bare-width",
                "path": plain.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "column": "B", "width": 20 }],
            }),
        )
        .await;
        let bare_sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            bare_sheet.contains(
                r#"<cols><col min="2" max="2" width="20" customWidth="1"/></cols><sheetData>"#
            ),
            "the created <cols> is not directly before <sheetData>: {bare_sheet}"
        );

        // The same width again is a no-op: a second call round-trips byte-identically.
        let again = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "split-again",
                "path": output.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "column": "B", "width": 20 }],
            }),
        )
        .await;
        assert_eq!(
            sheet,
            part_text(&single(&again), "xl/worksheets/sheet1.xml"),
            "a repeated width changed the sheet"
        );
    }

    /// A row format keeps the row element's own bytes (`spans`, `s`, its cells)
    /// and adds only `ht`/`customHeight`; a row the sheet has none of is created
    /// in order, and addressless rows are materialized before the insert so no
    /// successor is renumbered.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_row_height_keeps_the_row_and_creates_a_missing_one() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let rows = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c></row><row r="3" spans="1:2" s="1"><c r="A3"><v>3</v></c></row><row r="4"><c r="A4"><v>4</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "rows.xlsx",
            &with_parts(&base, &[("xl/worksheets/sheet1.xml", rows.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "height",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "row": 3, "height": 30 }],
            }),
        )
        .await;
        let sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            sheet.contains(r#"<row r="3" spans="1:2" s="1" ht="30" customHeight="1">"#),
            "the row's own attributes were not kept beside the height: {sheet}"
        );

        // A missing row 3 is created between 2 and 4, and the rows below keep their addresses.
        let gap = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c></row><row r="4"><c r="A4"><v>4</v></c></row></sheetData></worksheet>"#;
        let gap_source = write_fixture(
            &ws,
            "gap.xlsx",
            &with_parts(&base, &[("xl/worksheets/sheet1.xml", gap.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "gap-height",
                "path": gap_source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "row": 3, "height": 30 }],
            }),
        )
        .await;
        let gap_sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            gap_sheet.contains(concat!(
                r#"<row r="2"><c r="A2"><v>2</v></c></row>"#,
                r#"<row r="3" ht="30" customHeight="1"/>"#,
                r#"<row r="4"><c r="A4"><v>4</v></c></row>"#,
            )),
            "the created row did not land between 2 and 4: {gap_sheet}"
        );

        // Addressless rows are materialized before the insert, so their successors
        // still resolve to the numbers they held.
        let addressless = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c><v>1</v></c></row><row><c><v>2</v></c></row></sheetData></worksheet>"#;
        let addressless_source = write_fixture(
            &ws,
            "addressless.xlsx",
            &with_parts(
                &base,
                &[("xl/worksheets/sheet1.xml", addressless.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "addressless-height",
                "path": addressless_source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "row": 3, "height": 30 }],
            }),
        )
        .await;
        let addressless_sheet = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        assert!(
            addressless_sheet.contains(concat!(
                r#"<row r="1"><c r="A1"><v>1</v></c></row>"#,
                r#"<row r="2"><c r="A2"><v>2</v></c></row>"#,
                r#"<row r="3" ht="30" customHeight="1"/>"#,
            )),
            "the materialized successors were renumbered: {addressless_sheet}"
        );
    }

    /// A format aimed at part of a merged cell lands on the whole merge and says
    /// so; a range that already covers the merge is left as it is, so no note is
    /// owed and `<mergeCells>` is not rewritten.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_widens_a_merged_range_and_says_so() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:C1"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c><c r="C1"><v>3</v></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:C1"/></mergeCells></worksheet>"#;
        let source = write_fixture(
            &ws,
            "merged.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "widened",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let formatted = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        for (address, value) in [("A1", "1"), ("B1", "2"), ("C1", "3")] {
            assert!(
                formatted.contains(&format!(r#"<c r="{address}" s="1"><v>{value}</v></c>"#)),
                "{address} did not take the style of the widened range: {formatted}"
            );
        }
        assert!(
            reply.contains("the range A1 was widened to A1:C1"),
            "the widening was not named: {reply}"
        );
        assert!(
            formatted.contains(r#"<mergeCells count="1"><mergeCell ref="A1:C1"/></mergeCells>"#),
            "the merge range was rewritten: {formatted}"
        );

        // The paired `<mergeCell ref="A1:C1"></mergeCell>` spelling is the same
        // merge: it must widen the format and raise the same note, not be skipped
        // so the cell is styled alone.
        let paired_sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:C1"/><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c><c r="C1"><v>3</v></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:C1"></mergeCell></mergeCells></worksheet>"#;
        let paired_source = write_fixture(
            &ws,
            "merged-paired.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", paired_sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "widened-paired",
                "path": paired_source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let formatted = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        for (address, value) in [("A1", "1"), ("B1", "2"), ("C1", "3")] {
            assert!(
                formatted.contains(&format!(r#"<c r="{address}" s="1"><v>{value}</v></c>"#)),
                "{address} did not take the style of the paired merge's widening: {formatted}"
            );
        }
        assert!(
            reply.contains("the range A1 was widened to A1:C1"),
            "the paired merge's widening was not named: {reply}"
        );
        assert!(
            formatted.contains(r#"<mergeCell ref="A1:C1"></mergeCell>"#),
            "the paired merge element was rewritten: {formatted}"
        );

        // A range that already covers the merge moves nothing, so no note is owed.
        let covered = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "covered",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1:C1", "bold": true }],
            }),
        )
        .await;
        assert!(
            !covered.contains("widened"),
            "a range covering the merge was reported widened: {covered}"
        );
    }

    /// A format aimed at one cell of two merges stacked through a shared cell
    /// grows through both — the second reached only by the first widening — and
    /// the note names the range it reached and counts both merges, so the reply
    /// never reads as if a single merge covered the styled range.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_names_every_merge_a_widening_covers() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row><row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row></sheetData><mergeCells count="2"><mergeCell ref="A1:B1"/><mergeCell ref="B1:B2"/></mergeCells></worksheet>"#;
        let source = write_fixture(
            &ws,
            "stacked-merges.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "widened",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        assert!(
            reply.contains("the range A1 was widened to A1:B2"),
            "the widening was not named: {reply}"
        );
        assert!(
            reply.contains("covered by 2 merged cells"),
            "the note did not count both merges: {reply}"
        );
        assert!(
            !reply.contains("one merged cell"),
            "a two-merge widening was reported as a single merge: {reply}"
        );
        let formatted = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
        for address in ["A1", "B1", "A2", "B2"] {
            assert_eq!(
                attribute_of(cell_element(&formatted, address), "s"),
                "1",
                "{address} did not take the widened range's entry: {formatted}"
            );
        }
        assert!(
            formatted.contains(
                r#"<mergeCells count="2"><mergeCell ref="A1:B1"/><mergeCell ref="B1:B2"/></mergeCells>"#
            ),
            "the merge block was rewritten: {formatted}"
        );
    }

    /// A `<mergeCell>` may cover more cells than the caller's own range names, and
    /// the widening that lands on the whole merge is held to the same cap: a merge
    /// the widening takes past `format_cells_max` is refused before a cell is
    /// written, naming the range the widening reached.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_refuses_a_merge_that_widens_past_the_cell_cap() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        // Three rows by the grid's full 16384 columns — 49152 cells, well past the
        // `format_cells_max` one format may name. The caller's own `A1` is one
        // cell, so only the widening reaches the cap.
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:XFD3"/><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:XFD3"/></mergeCells></worksheet>"#;
        let source = write_fixture(
            &ws,
            "wide-merge.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "refused",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1",
                                "bold": true }],
                }),
            )
            .await
            .expect_err("a merge past the cell cap must be refused");
        let message = err.to_string();
        assert!(
            message.contains("A1:XFD3"),
            "the refusal did not name the range the widening reached: {message}"
        );
        assert!(
            message.contains("more than"),
            "the refusal did not name the cap it passed: {message}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A format brings styling to a workbook that never saved styles: the part is
    /// written, declared in `[Content_Types].xml` and named by the workbook's
    /// relationships, and the cell points at a real entry. A styles part that is
    /// there but degenerate is grown rather than refused.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_writes_the_styles_a_stripped_workbook_lacks() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;

        // The same workbook with its styles part dropped, the shape a package
        // that never saved styles has.
        let stripped = write_fixture(
            &ws,
            "stripped.xlsx",
            &repacked(&base, &["xl/styles.xml"], &[]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "restyled",
                "path": stripped.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        assert_parts(&output, &["xl/styles.xml"]);
        let types = part_text(&output, "[Content_Types].xml");
        assert!(
            types.contains(
                r#"PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml""#
            ),
            "the styles part was not declared: {types}"
        );
        let rels = part_text(&output, "xl/_rels/workbook.xml.rels");
        assert!(
            rels.contains(
                r#"Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml""#
            ),
            "the workbook does not name the styles part: {rels}"
        );
        let styles = part_text(&output, "xl/styles.xml");
        assert!(
            styles.contains(r#"<font><b/><sz val="11"/><name val="Calibri"/></font>"#),
            "the bold font was not written: {styles}"
        );
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let xfs = element_children(&styles, "cellXfs", "xf");
        let index: usize = attribute_of(cell_element(&sheet, "A1"), "s")
            .parse()
            .expect("a style index");
        assert!(
            index < xfs.len(),
            "the cell points past the styles part: {sheet} vs {styles}"
        );

        // A styles part that is there but degenerate — every block empty — must be
        // grown, not refused.
        let degenerate = write_fixture(
            &ws,
            "degenerate.xlsx",
            &with_parts(
                &base,
                &[(
                    "xl/styles.xml",
                    br#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="0"/><fills count="0"/><borders count="0"/><cellXfs count="0"/></styleSheet>"#
                        .as_slice(),
                )],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "grown",
                "path": degenerate.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let styles = part_text(&output, "xl/styles.xml");
        // The grown blocks hold their schema-default entry 0 first — the font
        // block's is the Calibri 11 default rather than a bare <font/>: entry 0
        // stays the neutral default every cell that names no style resolves to, so
        // the new bold entry lands at index 1 rather than restyling those cells.
        // That entry bases on the same font the block states first, so a part whose
        // own font 0 is missing does not leave the cell with a size-less face.
        assert!(
            styles.contains(
                r#"<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>"#
            ),
            "the fonts block was not created: {styles}"
        );
        // Every id the entry names resolves: fill 0 and border 0 come with the
        // block's first entry, grown here because the part stated them empty.
        assert!(
            styles.contains(
                r#"<fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders>"#
            ),
            "the blocks the entry's fill and border ids name were not grown: {styles}"
        );
        assert!(
            styles.contains(
                r#"<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>"#
            ),
            "the cellXfs block was not created: {styles}"
        );
        assert_eq!(
            attribute_of(
                cell_element(&part_text(&output, "xl/worksheets/sheet1.xml"), "A1"),
                "s"
            ),
            "1",
            "the cell does not point at the created entry"
        );
        // The grown part is one a reader can open: every block its entries' ids
        // resolve into is stated, each with its entry 0, in the schema's order.
        assert_styles_part_is_openable(&styles);
    }

    /// A styles part that is there but whose root is not the ordinary shape — a
    /// self-closing one holding no block, or an empty one whose close tag XML
    /// spells `</styleSheet >` — has to take the blocks an edit writes INSIDE it:
    /// a `<numFmts>` written after the root's close would leave two top-level
    /// elements in the part, a workbook Excel offers to repair. Both the number
    /// formats a `format_cells` and a `set_cell` write grow the same part.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_grows_a_styles_part_whose_root_is_not_the_ordinary_one() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let roots = [
            r#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>"#,
            r#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"></styleSheet >"#,
        ];
        let edits = [
            json!({ "op": "format_cells", "sheet": "S", "range": "A1", "number_format": "0.00" }),
            json!({ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1,
                    "number_format": "0.00" }),
        ];
        for (root_index, root) in roots.iter().enumerate() {
            for (edit_index, edit) in edits.iter().enumerate() {
                let name = format!("rooted-{root_index}-{edit_index}");
                let source = write_fixture(
                    &ws,
                    &format!("{name}.xlsx"),
                    &with_parts(&base, &[("xl/styles.xml", root.as_bytes())]),
                );
                let reply = run(
                    &ws,
                    json!({
                        "action": "xlsx_edit", "file_name": name,
                        "path": source.to_string_lossy(),
                        "edits": [edit.clone()],
                    }),
                )
                .await;
                let output = single(&reply);
                let styles = part_text(&output, "xl/styles.xml");
                let at = format!("root {root_index}, edit {edit_index}");
                assert_eq!(
                    styles.matches("<styleSheet").count(),
                    1,
                    "{at}: the part holds more than one <styleSheet>: {styles}"
                );
                assert_eq!(
                    styles.matches("</styleSheet").count(),
                    1,
                    "{at}: the part states a second close tag beside the root's: {styles}"
                );
                let num_fmts_at = styles
                    .find("<numFmts")
                    .unwrap_or_else(|| panic!("{at}: no <numFmts>: {styles}"));
                let cell_xfs_at = styles
                    .find("<cellXfs")
                    .unwrap_or_else(|| panic!("{at}: no <cellXfs>: {styles}"));
                assert!(
                    num_fmts_at < cell_xfs_at,
                    "{at}: <numFmts> was written after <cellXfs>: {styles}"
                );
                assert!(
                    styles.contains(r#"<numFmt numFmtId="164" formatCode="0.00"/>"#),
                    "{at}: the number format was not written: {styles}"
                );
                assert_styles_part_is_openable(&styles);
                let xfs = element_children(&styles, "cellXfs", "xf");
                let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
                let index: usize = attribute_of(cell_element(&sheet, "A1"), "s")
                    .parse()
                    .expect("a style index");
                assert_eq!(
                    attribute_of(
                        xfs.get(index)
                            .expect("the cell points past the styles part"),
                        "numFmtId"
                    ),
                    "164",
                    "{at}: the cell does not point at the written format: {styles}"
                );
            }
        }
    }

    /// The formatting an edit writes goes to the part the workbook's own
    /// relationships NAME, not the conventional `xl/styles.xml`: a workbook saved
    /// with its styles elsewhere has its cells' `s=` resolve in the part a reader
    /// reads, so the edit must write there — and an edit that writes no style must
    /// not invent the conventional part beside it.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_writes_the_styles_part_the_workbook_names() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let styles = part_text_bytes(&base, "xl/styles.xml");
        let rels = part_text_bytes(&base, "xl/_rels/workbook.xml.rels")
            .replace("Target=\"styles.xml\"", "Target=\"styles2.xml\"");
        let types = part_text_bytes(&base, "[Content_Types].xml").replace(
            "PartName=\"/xl/styles.xml\"",
            "PartName=\"/xl/styles2.xml\"",
        );
        let source = write_fixture(
            &ws,
            "named-elsewhere.xlsx",
            &repacked(
                &base,
                &["xl/styles.xml"],
                &[
                    ("xl/styles2.xml", styles.as_bytes()),
                    ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
                    ("[Content_Types].xml", types.as_bytes()),
                ],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "named-bold",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let names = zip_names(&output);
        assert!(
            names.iter().any(|name| name == "xl/styles2.xml"),
            "the formatting did not go to the part the relationships name: {names:?}"
        );
        assert!(
            !names.iter().any(|name| name == "xl/styles.xml"),
            "the edit invented the conventional styles part: {names:?}"
        );
        let styles2 = part_text(&output, "xl/styles2.xml");
        assert!(
            styles2.contains("<b/>"),
            "the bold was not written: {styles2}"
        );
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let index: usize = attribute_of(cell_element(&sheet, "A1"), "s")
            .parse()
            .expect("a style index");
        let xfs = element_children(&styles2, "cellXfs", "xf");
        assert!(
            index < xfs.len(),
            "the cell points past the entries the named part holds: {styles2}"
        );
        let rels_out = part_text(&output, "xl/_rels/workbook.xml.rels");
        assert!(
            rels_out.contains("Target=\"styles2.xml\""),
            "the relationships no longer name the part the styles live in: {rels_out}"
        );

        // An edit that writes no style leaves the conventional part uncreated too.
        let cleared = single(
            &run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "named-clear",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "clear_cell", "sheet": "S", "cell": "A1" }],
                }),
            )
            .await,
        );
        let cleared_names = zip_names(&cleared);
        assert!(
            !cleared_names.iter().any(|name| name == "xl/styles.xml"),
            "an edit that writes no style created the conventional styles part: {cleared_names:?}"
        );
    }

    /// An element a part opens and never closes is refused rather than written
    /// beside: a second `<sheetData>`, `<cols>`, `<dimension>` or styles block is
    /// a part a reader offers to repair, which no edit may report as a success.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_an_element_a_part_leaves_open() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let sheet = part_text_bytes(&base, "xl/worksheets/sheet1.xml");
        let styles = part_text_bytes(&base, "xl/styles.xml");
        let cases = [
            (
                "sheetData-open.xlsx",
                "xl/worksheets/sheet1.xml",
                with_first(&sheet, "</sheetData>", ""),
                json!({ "op": "set_cell", "sheet": "S", "cell": "A2", "value": 2 }),
                "<sheetData>",
            ),
            (
                "cols-open.xlsx",
                "xl/worksheets/sheet1.xml",
                with_first(
                    &sheet,
                    "<sheetData>",
                    r#"<cols><col min="1" max="1" width="9"/><sheetData>"#,
                ),
                json!({ "op": "format_cells", "sheet": "S", "column": "B", "width": 12 }),
                "<cols>",
            ),
            (
                "dimension-open.xlsx",
                "xl/worksheets/sheet1.xml",
                with_first(
                    &sheet,
                    "<sheetData>",
                    r#"<dimension ref="A1:Z9"><sheetData>"#,
                ),
                json!({ "op": "set_cell", "sheet": "S", "cell": "B2", "value": 3 }),
                "<dimension>",
            ),
            (
                "styles-open.xlsx",
                "xl/styles.xml",
                with_first(&styles, "</fonts>", ""),
                json!({ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }),
                "<fonts>",
            ),
        ];
        let before = generated_count(&ws);
        for (name, part, body, edit, element) in cases {
            let source = write_fixture(&ws, name, &with_parts(&base, &[(part, body.as_bytes())]));
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "xlsx_edit", "file_name": name,
                        "path": source.to_string_lossy(),
                        "edits": [edit],
                    }),
                )
                .await
                .expect_err("an element left open must be refused");
            let message = err.to_string();
            assert!(
                message.contains("opened and never closed") && message.contains(element),
                "the refusal for {name} does not name the element left open: {message}"
            );
        }
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A worksheet states its rows in ONE `<sheetData>` (`CT_Worksheet` says so),
    /// so a part that states a second one, or a `<row>` outside the one it states,
    /// is refused by the edits that write into the body — a row created in it, the
    /// body rebuilt — since a rebuilt body would otherwise duplicate the rows the
    /// walk found outside the region it writes. An edit that only rewrites the cells
    /// it found, or places a block beside the body (a column width's `<cols>`),
    /// still goes through: the part comes out as healthy as it arrived.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_a_sheet_whose_rows_are_not_in_one_body() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let row = r#"<row r="1"><c r="A1"><v>1</v></c></row>"#;
        let shapes = [
            (
                "two-bodies.xlsx",
                format!(
                    r#"<sheetData>{row}</sheetData><sheetData><row r="2"><c r="A2"><v>2</v></c></row></sheetData>"#
                ),
                2,
            ),
            (
                "stray-row.xlsx",
                format!(r#"<sheetData>{row}</sheetData><row r="2"><c r="A2"><v>2</v></c></row>"#),
                1,
            ),
        ];
        let before = generated_count(&ws);
        for (name, body, bodies) in &shapes {
            let sheet = format!(
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{body}</worksheet>"#
            );
            let source = write_fixture(
                &ws,
                name,
                &with_parts(&base, &[("xl/worksheets/sheet1.xml", sheet.as_bytes())]),
            );
            for edit in [
                json!({ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }),
                json!({ "op": "set_cell", "sheet": "S", "cell": "A3", "value": 3 }),
            ] {
                let err = DocumentTool
                    .execute(
                        &ws,
                        json!({
                            "action": "xlsx_edit", "file_name": name,
                            "path": source.to_string_lossy(),
                            "edits": [edit],
                        }),
                    )
                    .await
                    .expect_err("a part whose rows are not in one body must be refused");
                let message = err.to_string();
                assert!(
                    message.contains("<sheetData>"),
                    "{name} was not refused for the shape of its body: {message}"
                );
            }
            // The cell an edit finds is rewritten where it stands: the two bodies the
            // part states are the file's own, and what the edit wrote is one value.
            let reply = run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "rewritten",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 8 }],
                }),
            )
            .await;
            let written = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
            assert!(
                written.matches("<sheetData").count() == *bodies
                    && written.matches(r#"<row r="2">"#).count() == 1,
                "{name} gained, lost or duplicated a body: {written}"
            );
            assert!(
                written.contains(r#"<c r="A1"><v>8</v></c>"#),
                "{name} did not take the value: {written}"
            );
            // A column width never writes into the body — its `<cols>` goes beside it —
            // so a part whose rows are not in one body is no reason to refuse it. The
            // part still holds exactly the bodies it arrived with, and the block stands
            // before the first of them.
            let reply = run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "width",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_cells", "sheet": "S", "column": "B", "width": 12 }],
                }),
            )
            .await;
            let written = part_text(&single(&reply), "xl/worksheets/sheet1.xml");
            assert!(
                written.matches("<sheetData").count() == *bodies
                    && written.contains(r#"<col min="2" max="2" width="12" customWidth="1"/>"#)
                    && written.find("<cols>") < written.find("<sheetData"),
                "{name} did not take the width beside its body: {written}"
            );
        }
        assert_eq!(
            generated_count(&ws),
            before + shapes.len() * 2,
            "a refused call left an output"
        );
    }

    /// A `<font>` that names a THEME face (`<scheme val="minor"/>` beside its
    /// `<name>`, as Excel states it) has to leave the scheme when an edit names a
    /// `font` face: the theme's face would render instead of the requested one and
    /// be rewritten by a theme change. An edit that names no face — a `size` —
    /// keeps the scheme.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_takes_a_named_face_out_of_the_theme_scheme() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let styles = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/><scheme val="minor"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>"#;
        let source = write_fixture(
            &ws,
            "themed-font.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/styles.xml", styles.as_bytes())],
            ),
        );
        // The `<font>` the cell's own `s=` resolves to: its entry's `fontId` into
        // the `<fonts>` block.
        let font_of_cell = |styles: &str, sheet: &str| -> String {
            let index: usize = attribute_of(cell_element(sheet, "A1"), "s")
                .parse()
                .expect("a style index");
            let xfs = element_children(styles, "cellXfs", "xf");
            let xf = xfs
                .get(index)
                .expect("the cell points past the styles part");
            let font_id: usize = attribute_of(xf, "fontId").parse().expect("a font index");
            element_children(styles, "fonts", "font")
                .get(font_id)
                .expect("the entry points past the fonts block")
                .to_string()
        };
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "georgia",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "font": "Georgia" }],
            }),
        )
        .await;
        let output = single(&reply);
        let written = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let font = font_of_cell(&written, &sheet);
        assert!(
            font.contains(r#"<name val="Georgia"/>"#),
            "the requested face was not written: {font}"
        );
        assert!(
            font.contains(r#"<family val="2"/>"#),
            "a child the request did not name was dropped: {font}"
        );
        assert!(
            !font.contains("<scheme"),
            "the requested face stayed in the theme's scheme, so the theme would render: {font}"
        );

        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "sized",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "size": 14 }],
            }),
        )
        .await;
        let output = single(&reply);
        let written = part_text(&output, "xl/styles.xml");
        let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
        let font = font_of_cell(&written, &sheet);
        assert!(
            font.contains(r#"<scheme val="minor"/>"#),
            "a size names no face, so the theme's scheme stays: {font}"
        );
    }

    /// A styles part whose root never closes has nothing for an edit to write
    /// into: the kit refuses it whichever property the edit names — a style or
    /// no style at all — and no output file is left behind.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_a_styles_part_with_no_closed_root() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let unclosed = r#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>"#;
        let source = write_fixture(
            &ws,
            "unclosed-styles.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/styles.xml", unclosed.as_bytes())],
            ),
        );
        let edits = [
            json!({ "op": "format_cells", "sheet": "S", "range": "A1", "align": "center" }),
            json!({ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }),
            json!({ "op": "format_cells", "sheet": "S", "column": "A", "width": 12 }),
            json!({ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1,
                    "number_format": "0.00" }),
            json!({ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1 }),
        ];
        let before = generated_count(&ws);
        for (index, edit) in edits.into_iter().enumerate() {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "xlsx_edit", "file_name": "refused",
                        "path": source.to_string_lossy(),
                        "edits": [edit],
                    }),
                )
                .await
                .expect_err("a styles part with no closed root must be refused");
            let message = err.to_string();
            assert!(
                message.contains("its <styleSheet> root is missing or never closed"),
                "edit {index} was not refused for the part's shape: {message}"
            );
        }
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// The named cell styles a workbook states, which an edit that grows the
    /// styles part must leave byte for byte.
    const CELL_STYLES: &str = r#"<cellStyles count="3"><cellStyle name="Normal" xfId="0" builtinId="0"/><cellStyle name="Comma" xfId="1"/><cellStyle name="My Header" xfId="2"/></cellStyles>"#;

    /// A workbook that states named cell styles — the `Normal` default beside
    /// `Comma` and `My Header` — keeps every one of them through an edit that
    /// names only a `fill` and through one that names only a `number_format`: the
    /// `cellStyles` block is not the edit's to replace, and the cell's `s=` still
    /// resolves inside `<cellXfs>`.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_keeps_the_named_cell_styles_a_workbook_states() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let styles = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>{CELL_STYLES}</styleSheet>"#
        );
        let source = write_fixture(
            &ws,
            "named-styles.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/styles.xml", styles.as_bytes())],
            ),
        );
        let edits = [
            json!({ "op": "format_cells", "sheet": "S", "range": "A1", "fill": "FFEEDD" }),
            json!({ "op": "set_cell", "sheet": "S", "cell": "A1", "value": 1,
                    "number_format": "0.00" }),
        ];
        for (index, edit) in edits.into_iter().enumerate() {
            let reply = run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": format!("styled-{index}"),
                    "path": source.to_string_lossy(),
                    "edits": [edit],
                }),
            )
            .await;
            let output = single(&reply);
            let written = part_text(&output, "xl/styles.xml");
            assert!(
                written.contains(CELL_STYLES),
                "edit {index} replaced the workbook's named cell styles: {written}"
            );
            let names: Vec<&str> = element_children(&written, "cellStyles", "cellStyle")
                .iter()
                .map(|entry| attribute_of(entry, "name"))
                .collect();
            assert_eq!(
                names,
                ["Normal", "Comma", "My Header"],
                "edit {index} changed the named cell styles: {written}"
            );
            assert_styles_part_is_openable(&written);
            let sheet = part_text(&output, "xl/worksheets/sheet1.xml");
            let style_index: usize = attribute_of(cell_element(&sheet, "A1"), "s")
                .parse()
                .expect("a style index");
            let xfs = element_children(&written, "cellXfs", "xf");
            assert!(
                style_index < xfs.len(),
                "the cell points past the styles part: {sheet} vs {written}"
            );
        }
    }

    /// A workbook that holds its styles part but whose relationships name an
    /// unrelated type instead of the styles one: an edit writes the relationship
    /// list so the workbook names the part it holds, while the unrelated
    /// relationship the file already had keeps its own bytes. The naming happens
    /// before any edit runs, so an edit that writes no style at all names it too —
    /// a styles part nothing points at is one no reader reads.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_names_a_styles_part_the_workbook_holds_but_does_not_name() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let rels = part_text_bytes(&base, "xl/_rels/workbook.xml.rels").replace(
            r#"Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles""#,
            r#"Type="urn:unrelated""#,
        );
        let source = write_fixture(
            &ws,
            "unrelated-styles.xlsx",
            &with_parts(&base, &[("xl/_rels/workbook.xml.rels", rels.as_bytes())]),
        );
        let edits = [
            json!({ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }),
            json!({ "op": "clear_cell", "sheet": "S", "cell": "A1" }),
        ];
        for (index, edit) in edits.into_iter().enumerate() {
            let reply = run(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "named",
                    "path": source.to_string_lossy(),
                    "edits": [edit],
                }),
            )
            .await;
            let written = part_text(&single(&reply), "xl/_rels/workbook.xml.rels");
            assert!(
                written.contains(
                    r#"<Relationship Id="rId2" Type="urn:unrelated" Target="styles.xml"/>"#
                ),
                "edit {index} rewrote the unrelated relationship: {written}"
            );
            // A target relative to `xl/workbook.xml` resolves to `xl/styles.xml`.
            assert!(
                written.contains(
                    r#"Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml""#
                ),
                "edit {index} left the styles part the workbook holds unnamed: {written}"
            );
        }
    }

    /// A workbook with no `xl/_rels/workbook.xml.rels` at all: the kit works with
    /// the part names a reader falls back to, so an edit that writes no style goes
    /// through — while one that writes styles is refused, since a styles part the
    /// workbook's relationships cannot name would be a part no reader reads, and
    /// writing the relationship list alone would leave the workbook's own
    /// `<sheet r:id="rId1"/>` unresolvable.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_styles_a_workbook_with_no_relationships_part_cannot_name() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(
            &ws,
            "no-rels.xlsx",
            &without_part(&created_xlsx(&ws).await, "xl/_rels/workbook.xml.rels"),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "refused",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
                }),
            )
            .await
            .expect_err("a workbook whose relationships cannot name a styles part must be refused");
        assert!(
            err.to_string().contains("cannot name a styles part"),
            "the refusal does not name the shape it rejects: {err}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
        // An edit that writes no style has nothing to name and goes through.
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "cleared",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "clear_cell", "sheet": "S", "cell": "A1" }],
            }),
        )
        .await;
        assert!(
            !zip_names(&single(&reply)).contains(&"xl/_rels/workbook.xml.rels".to_string()),
            "a style-less edit invented a relationships part"
        );
    }

    /// A row whose own cells are stated out of address order — which a writer this
    /// kit did not produce may do — is written back with its cells in address
    /// order, each stated once: a format over the rect adds only the cells the row
    /// really lacks and puts every cell where a reader expects it (a duplicate or
    /// misplaced address is a workbook Excel offers to repair), and every cell of
    /// the rect takes a style index the written part holds.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_writes_a_row_that_states_its_cells_out_of_address_order() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:D2"/><sheetData><row r="1"><c r="D1"><v>4</v></c><c r="A1" t="inlineStr"><is><t>имя</t></is></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "unordered-row.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "ordered",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1:D1",
                            "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let formatted = part_text(&output, "xl/worksheets/sheet1.xml");
        for address in ["A1", "B1", "C1", "D1"] {
            assert_eq!(
                formatted.matches(&format!(r#"r="{address}""#)).count(),
                1,
                "{address} is stated more than once in the row: {formatted}"
            );
        }
        // The cells come back in address order: `D1` was stated first, and a reader
        // reaching each position finds the cell its address names.
        let stated: Vec<usize> = ["A1", "B1", "C1", "D1"]
            .iter()
            .map(|address| {
                formatted
                    .find(&format!(r#"r="{address}""#))
                    .expect("a stated cell")
            })
            .collect();
        assert!(
            stated.windows(2).all(|pair| pair[0] < pair[1]),
            "the cells were not written in address order: {formatted}"
        );
        let styles = part_text(&output, "xl/styles.xml");
        let xfs = element_children(&styles, "cellXfs", "xf");
        for address in ["A1", "B1", "C1", "D1"] {
            let index: usize = attribute_of(cell_element(&formatted, address), "s")
                .parse()
                .expect("a style index");
            assert!(
                index < xfs.len(),
                "{address} points past the styles part: {formatted} vs {styles}"
            );
        }
    }

    /// A row that states a label cell, then a gap, then the cells a range names — so
    /// the range's own left edge lies past the first cell the row holds — leaves every
    /// column it does not name alone: no cell is created for one of them and the cells
    /// outside the range keep their exact bytes.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_leaves_the_columns_before_the_range_alone() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:D1"/><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>label</t></is></c><c r="C1"><v>3</v></c><c r="D1"><v>4</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "offset-range.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "offset",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "C1:D1",
                            "fill": "FFF2CC" }],
            }),
        )
        .await;
        let output = single(&reply);
        let formatted = part_text(&output, "xl/worksheets/sheet1.xml");
        assert!(
            !formatted.contains(r#"<c r="B1""#),
            "a cell was created for a column the range does not name: {formatted}"
        );
        assert!(
            formatted.contains(r#"<c r="A1" t="inlineStr"><is><t>label</t></is></c>"#),
            "a cell outside the range changed: {formatted}"
        );
        let styles = part_text(&output, "xl/styles.xml");
        let xfs = element_children(&styles, "cellXfs", "xf");
        for address in ["C1", "D1"] {
            let index: usize = attribute_of(cell_element(&formatted, address), "s")
                .parse()
                .expect("a style index");
            assert!(
                index < xfs.len(),
                "{address} points past the styles part: {formatted} vs {styles}"
            );
        }
    }

    /// A sheet part that opens a `<row>` or a `<c>` and never closes it is a shape the
    /// readers cannot see — `elementPattern` matches the closed shapes alone — so a
    /// writer would place its content beside or inside the element it read as absent:
    /// a second `<row>` for the same number, a `<c>` nested in the one it missed. No
    /// writer produces such a part, so it is refused (never answered with a success
    /// over a file the edit made worse), and no output is written.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_a_row_or_cell_a_sheet_leaves_open() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let base = created_xlsx(&ws).await;
        let open_row = r#"<sheetData><row r="1"><c r="A1"><v>1</v></c></sheetData>"#;
        let open_cell = r#"<sheetData><row r="1"><c r="A1"><v>1</v></row></sheetData>"#;
        let cases = [
            (
                "row-open.xlsx",
                open_row,
                json!({ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }),
                "<row>",
            ),
            (
                "row-open-set.xlsx",
                open_row,
                json!({ "op": "set_cell", "sheet": "S", "cell": "A2", "value": 2 }),
                "<row>",
            ),
            (
                "cell-open.xlsx",
                open_cell,
                json!({ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }),
                "<c>",
            ),
            (
                "cell-open-set.xlsx",
                open_cell,
                json!({ "op": "set_cell", "sheet": "S", "cell": "B1", "value": 2 }),
                "<c>",
            ),
        ];
        let before = generated_count(&ws);
        for (name, body, edit, element) in cases {
            let sheet = format!(
                r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{body}</worksheet>"#
            );
            let source = write_fixture(
                &ws,
                name,
                &with_parts(&base, &[("xl/worksheets/sheet1.xml", sheet.as_bytes())]),
            );
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "xlsx_edit", "file_name": name,
                        "path": source.to_string_lossy(),
                        "edits": [edit],
                    }),
                )
                .await
                .expect_err("an element left open must be refused");
            let message = err.to_string();
            assert!(
                message.contains("opened and never closed") && message.contains(element),
                "the refusal for {name} does not name the element left open: {message}"
            );
        }
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A body whose rows are not in address order is written back in address order, but
    /// a row that states no number of its own stands for the position it sits in:
    /// reordering such a body would move the value the row holds, so the edit is
    /// refused rather than answered with a silent relocation.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_refuses_to_reorder_a_row_that_states_no_number() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="3"><c r="A3"><v>3</v></c></row><row r="2"><c r="A2"><v>2</v></c></row><row><c><v>99</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "implicit-row.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "xlsx_edit", "file_name": "implicit",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "format_cells", "sheet": "S", "range": "A2:A3",
                                "bold": true }],
                }),
            )
            .await
            .expect_err("a row stating no number must not be reordered");
        let message = err.to_string();
        assert!(
            message.contains("address order"),
            "the refusal does not name the shape it rejects: {message}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A sheet whose `<row>` elements are not in ascending order — `<row r="5">`
    /// stated before `<row r="2">`, each holding a value — is formatted over
    /// `A2:A5`: the rect's missing rows are created in address order, each of its
    /// row numbers is stated exactly once, no duplicate row is left for a reader to
    /// repair, and the values stay with the rows that held them.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_keeps_a_row_number_stated_once_when_rows_are_out_of_order() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let sheet = r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A2:A5"/><sheetData><row r="5"><c r="A5"><v>5</v></c></row><row r="2"><c r="A2"><v>2</v></c></row></sheetData></worksheet>"#;
        let source = write_fixture(
            &ws,
            "unordered-rows.xlsx",
            &with_parts(
                &created_xlsx(&ws).await,
                &[("xl/worksheets/sheet1.xml", sheet.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "ordered",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A2:A5",
                            "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let formatted = part_text(&output, "xl/worksheets/sheet1.xml");
        for number in ["2", "3", "4", "5"] {
            assert_eq!(
                formatted.matches(&format!(r#"<row r="{number}""#)).count(),
                1,
                "row {number} is stated more than once: {formatted}"
            );
        }
        // The rows go back in address order, and each keeps the value it held.
        let stated: Vec<usize> = ["2", "3", "4", "5"]
            .iter()
            .map(|number| {
                formatted
                    .find(&format!(r#"<row r="{number}""#))
                    .expect("a stated row")
            })
            .collect();
        assert!(
            stated.windows(2).all(|pair| pair[0] < pair[1]),
            "the rows were not written in address order: {formatted}"
        );
        assert!(
            formatted.contains(r#"<c r="A5" s="1"><v>5</v></c>"#)
                && formatted.contains(r#"<c r="A2" s="1"><v>2</v></c>"#),
            "a value left its row: {formatted}"
        );
    }

    /// A format writes no value and moves no address, so only the sheet it names
    /// and the styles part it grows change: every other part — a chart, a table, a
    /// comment, a macro — keeps its exact bytes, nothing is added, and no stale
    /// chart or pivot caveat is owed.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn xlsx_edit_format_cells_names_no_part_it_did_not_touch() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        // Two sheets, so the loop below proves the sheet the edit did not name is
        // carried through byte for byte beside the chart, the table, the comment
        // and the macro.
        let base = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "xlsx", "file_name": "book",
                    "sheets": [
                        { "name": "S", "rows": [["seed"]] },
                        { "name": "Other", "rows": [["kept", 1]] },
                    ],
                }),
            )
            .await,
        );
        let package = with_parts(
            &std::fs::read(&base).expect("read base package"),
            &[
                ("xl/charts/chart1.xml", EDIT_CHART.as_bytes()),
                ("xl/tables/table1.xml", EDIT_TABLE.as_bytes()),
                ("xl/comments1.xml", EDIT_COMMENTS.as_bytes()),
                ("xl/vbaProject.bin", b"macro-bytes"),
            ],
        );
        let source = write_fixture(&ws, "annotated.xlsx", &package);
        let reply = run(
            &ws,
            json!({
                "action": "xlsx_edit", "file_name": "formatted",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_cells", "sheet": "S", "range": "A1", "bold": true }],
            }),
        )
        .await;
        let output = single(&reply);
        let touched = ["xl/worksheets/sheet1.xml", "xl/styles.xml"];
        let names = zip_names(&source);
        assert!(
            names.iter().any(|name| name == "xl/worksheets/sheet2.xml"),
            "the fixture holds two sheets: {names:?}"
        );
        for name in &names {
            assert!(
                zip_names(&output).contains(name),
                "{name} vanished from the formatted workbook"
            );
            if touched.contains(&name.as_str()) {
                continue;
            }
            assert_eq!(
                part_bytes(&source, name),
                part_bytes(&output, name),
                "{name} changed"
            );
        }
        assert_eq!(
            names.len(),
            zip_names(&output).len(),
            "a part was added to the formatted workbook"
        );
        assert!(
            !reply.contains("copied unchanged"),
            "a format raised a stale-object caveat: {reply}"
        );
    }

    /// The text of a presentation's own `<p:sldIdLst>`, read up to the list's close
    /// tag however that close is spelled: what an entry appended at the list's end
    /// must land inside.
    fn slide_list(presentation: &str) -> &str {
        let open = presentation.find("<p:sldIdLst").expect("a slide list");
        let rest = &presentation[open..];
        let body = rest.find('>').expect("the list's open tag") + 1;
        let close = rest[body..].find("</p:sldIdLst").expect("the list's close") + body;
        &rest[body..close]
    }

    /// A deck whose slide list, relationships and content types all close with the
    /// whitespace XML allows before the `>` is edited like any other: the added
    /// slide's entry goes inside the list, its part is named by a relationship and
    /// declared in the content types, and a move lands inside the list too.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_registers_a_slide_in_a_list_spelled_with_whitespace() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Один" },
                        { "type": "heading", "level": 1, "text": "Два" },
                        { "type": "heading", "level": 1, "text": "Три" },
                    ],
                }),
            )
            .await,
        );
        let package = std::fs::read(&created).expect("read base package");
        let presentation = with_first(
            &part_text_bytes(&package, "ppt/presentation.xml"),
            "</p:sldIdLst>",
            "</p:sldIdLst >",
        );
        let relationships = with_first(
            &part_text_bytes(&package, "ppt/_rels/presentation.xml.rels"),
            "</Relationships>",
            "</Relationships >",
        );
        let types = with_first(
            &part_text_bytes(&package, "[Content_Types].xml"),
            "</Types>",
            "</Types >",
        );
        let source = write_fixture(
            &ws,
            "deck.pptx",
            &with_parts(
                &package,
                &[
                    ("ppt/presentation.xml", presentation.as_bytes()),
                    ("ppt/_rels/presentation.xml.rels", relationships.as_bytes()),
                    ("[Content_Types].xml", types.as_bytes()),
                ],
            ),
        );

        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "appended",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_slide", "title": "APPENDED" }],
            }),
        )
        .await;
        let output = single(&reply);
        let presentation = part_text(&output, "ppt/presentation.xml");
        assert_eq!(
            slide_list(&presentation).matches("<p:sldId ").count(),
            4,
            "the added slide is not registered inside the list: {presentation}"
        );
        let relationships = part_text(&output, "ppt/_rels/presentation.xml.rels");
        assert!(
            relationships.contains("slides/slide4.xml")
                && relationships.contains("</Relationships >"),
            "the added slide's part is not named by a relationship: {relationships}"
        );
        let types = part_text(&output, "[Content_Types].xml");
        assert!(
            types.contains(r#"PartName="/ppt/slides/slide4.xml""#) && types.contains("</Types >"),
            "the added slide's part is not declared in the content types: {types}"
        );

        let moved = single(
            &run(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "moved",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "move_slide", "slide": 1, "to": 3 }],
                }),
            )
            .await,
        );
        let presentation = part_text(&moved, "ppt/presentation.xml");
        assert_eq!(
            slide_list(&presentation).matches("<p:sldId ").count(),
            3,
            "the moved slide's entry left the list: {presentation}"
        );
    }

    /// The whitespace before a close tag's `>` reaches the other parts a slide edit
    /// writes into: a paragraph added to notes whose body closes `</p:txBody >`
    /// lands inside the body, and a picture lands inside a shape tree closing
    /// `</p:spTree >`.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_writes_beside_a_close_tag_that_carries_whitespace() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Слайд" },
                        { "type": "paragraph", "text": "Тело" },
                        { "type": "notes", "text": "Заметка" },
                    ],
                }),
            )
            .await,
        );
        let package = std::fs::read(&created).expect("read base package");
        let notes = zip_names(&created)
            .into_iter()
            .find(|name| name.starts_with("ppt/notesSlides/notesSlide"))
            .expect("the deck's notes part");
        let body = with_first(
            &part_text_bytes(&package, &notes),
            "</p:txBody>",
            "</p:txBody >",
        );
        let spaced = write_fixture(
            &ws,
            "noted.pptx",
            &with_parts(&package, &[(&notes, body.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "noted",
                "path": spaced.to_string_lossy(),
                "edits": [{ "op": "add_notes", "slide": 1, "text": "ДОБАВЛЕНО" }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), &notes);
        assert!(
            edited.contains("</a:p></p:txBody >"),
            "the added notes paragraph did not land inside the notes body: {edited}"
        );
        assert!(
            edited.contains("ДОБАВЛЕНО"),
            "the notes text was not written: {edited}"
        );

        let slide = "ppt/slides/slide1.xml";
        let tree = with_first(
            &part_text_bytes(&package, slide),
            "</p:spTree>",
            "</p:spTree >",
        );
        let spaced = write_fixture(
            &ws,
            "pictured.pptx",
            &with_parts(&package, &[(slide, tree.as_bytes())]),
        );
        let picture = write_fixture(&ws, "pic.png", &noisy_png(40, 20));
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "pictured",
                "path": spaced.to_string_lossy(),
                "edits": [{ "op": "add_image", "slide": 1, "path": picture.to_string_lossy(),
                            "x": 0.1, "y": 0.1, "width": 0.4 }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), slide);
        let placed = edited.find("<p:pic>").expect("the placed picture");
        let close = edited
            .find("</p:spTree >")
            .expect("the shape tree's close tag");
        assert!(
            placed < close,
            "the picture fell outside the shape tree: {edited}"
        );
    }

    /// A run property and the run-properties element around it, each closed with
    /// the whitespace XML allows before the `>` (`<a:rPr …>…</a:rPr >`), are read as
    /// the elements they are: a request naming the run's colour rebuilds the run's
    /// properties from the run's own body — the outline's fill stays, the run's own
    /// fill is written — rather than reading the outline's child as the run's and
    /// taking it away.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_keeps_a_run_property_whose_close_tag_carries_whitespace() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [{ "type": "heading", "level": 1, "text": "Обводка" }],
                }),
            )
            .await,
        );
        let package = std::fs::read(&created).expect("read base package");
        // The addressed run's own properties close with the whitespace and gain an
        // outline whose fill must survive the request.
        let outline = r#"<a:ln><a:solidFill><a:srgbClr val="00FF00"/></a:solidFill></a:ln >"#;
        let slide = with_first(
            &part_text_bytes(&package, "ppt/slides/slide1.xml"),
            "</a:rPr>",
            "</a:rPr >",
        );
        let text = slide.find("Обводка").expect("the heading's text");
        let properties = slide[..text].rfind("<a:rPr").expect("the run's properties");
        let open = slide[properties..].find('>').expect("its open tag") + properties + 1;
        let patched = format!("{}{outline}{}", &slide[..open], &slide[open..]);
        let source = write_fixture(
            &ws,
            "outlined.pptx",
            &with_parts(&package, &[("ppt/slides/slide1.xml", patched.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "format_text", "slide": 1, "find": "Обводка",
                            "color": "#3366CC" }],
            }),
        )
        .await;
        let edited = part_text(&single(&reply), "ppt/slides/slide1.xml");
        assert!(
            edited.contains(outline),
            "the run's outline fill was taken away: {edited}"
        );
        // The element the request rewrote holds the outline and the run's own fill,
        // in that order, and nothing the outline's close tag left behind. The run's
        // own body carries more than those two — the fonts a declared title states
        // among it — so the two are asserted, not the element's end.
        assert!(
            edited.contains(&format!(
                r#"{outline}<a:solidFill><a:srgbClr val="3366CC"/></a:solidFill>"#
            )),
            "the properties were rebuilt from something other than the run's own body: {edited}"
        );
    }

    /// A pptx edit adds and deletes slides without touching the deck's chart or
    /// timing parts, keeps the surviving slides' text and notes, and refuses to
    /// delete a deck's only slide.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: one deck exercising every slide and text op
    async fn pptx_edit_adds_and_deletes_slides_and_keeps_the_survivors() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Первый слайд" },
                        { "type": "paragraph", "text": "Первый текст" },
                        { "type": "paragraph", "text": "Удаляемый текст" },
                        { "type": "notes", "text": "Заметка один" },
                        { "type": "heading", "level": 1, "text": "Второй слайд" },
                        { "type": "paragraph", "text": "Второй текст" },
                        { "type": "notes", "text": "Заметка два" },
                    ],
                }),
            )
            .await,
        );
        let package = with_parts(
            &std::fs::read(&created).expect("read base package"),
            &[
                ("ppt/charts/chart1.xml", EDIT_CHART.as_bytes()),
                ("ppt/animations/timing1.xml", EDIT_TIMING.as_bytes()),
            ],
        );
        let source = write_fixture(&ws, "deck.pptx", &package);

        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "replace_text", "slide": 1, "find": "Первый", "replace": "Изменён" },
                    { "op": "add_paragraph", "slide": 1, "text": "Добавлено" },
                    { "op": "remove_paragraph", "slide": 1, "find": "Удаляемый" },
                    { "op": "add_slide", "after": 1, "title": "Вставленный",
                      "bullets": ["раз", "два"] },
                    { "op": "add_slide", "title": "В конце" },
                    { "op": "delete_slide", "slide": 3 },
                ],
            }),
        )
        .await;
        let output = single(&reply);

        for name in ["ppt/charts/chart1.xml", "ppt/animations/timing1.xml"] {
            assert_eq!(
                part_bytes(&source, name),
                part_bytes(&output, name),
                "{name} changed"
            );
        }
        let surviving_notes: Vec<String> = zip_names(&output)
            .into_iter()
            .filter(|name| name.starts_with("ppt/notesSlides/notesSlide"))
            .collect();
        assert!(
            !surviving_notes.is_empty(),
            "the surviving slide's notes part was dropped"
        );
        for name in &surviving_notes {
            assert_eq!(
                part_bytes(&source, name),
                part_bytes(&output, name),
                "{name} changed"
            );
        }

        let slides = converted_text(&ws, &output).await;
        assert!(
            slides.contains("Изменён текст"),
            "the surviving slide's text was lost: {slides}"
        );
        assert!(slides.contains("Добавлено"), "add_paragraph lost: {slides}");
        assert!(
            !slides.contains("Удаляемый"),
            "remove_paragraph lost: {slides}"
        );
        assert!(
            !slides.contains("Второй текст"),
            "the deleted slide's text survived: {slides}"
        );
        assert!(
            slides.contains("Заметка один"),
            "the surviving slide's notes were lost: {slides}"
        );
        assert!(
            !slides.contains("Заметка два"),
            "the deleted slide's notes survived: {slides}"
        );
        let inserted = slides.find("Вставленный").expect("the inserted slide");
        let appended = slides.find("В конце").expect("the appended slide");
        assert!(inserted < appended, "the slides are out of order: {slides}");
        assert!(
            slides.contains("Slide 2:") && slides.contains("Slide 3:"),
            "the added slides are not in the deck: {slides}"
        );
        assert!(
            !reply.to_lowercase().contains("reorder"),
            "the reply says something about reordering: {reply}"
        );

        let one = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "one",
                    "content": [{ "type": "paragraph", "text": "Только слайд" }],
                }),
            )
            .await,
        );
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": one.to_string_lossy(),
                    "edits": [{ "op": "delete_slide", "slide": 1 }],
                }),
            )
            .await
            .expect_err("a deck's only slide cannot be deleted");
        assert!(err.to_string().contains("only slide"), "got: {err}");
    }

    /// A deck's slides are reordered, duplicated and re-noted: `move_slide` puts
    /// the addressed slide first, `duplicate_slide` copies it in place with its
    /// notes, and `replace_notes` rewrites only the slide it names — the reader's
    /// own `Slide N:`/`Slide N notes:` labels say so. A notes shape whose `<p:ph>`
    /// names no type is not the body a reader draws notes from, so the notes'
    /// text is still rewritten there while a paragraph is not added to it.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_moves_duplicates_and_replaces_notes() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Первый" },
                        { "type": "paragraph", "text": "Тело один" },
                        { "type": "notes", "text": "Заметка один" },
                        { "type": "heading", "level": 1, "text": "Второй" },
                        { "type": "paragraph", "text": "Тело два" },
                        { "type": "notes", "text": "Заметка два" },
                    ],
                }),
            )
            .await,
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "move_slide", "slide": 2, "to": 1 },
                    { "op": "duplicate_slide", "slide": 1 },
                    { "op": "replace_notes", "slide": 1, "find": "Заметка два",
                      "replace": "Заметка изменена" },
                ],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;

        // The reader numbers the three slides it now holds, and the second
        // slide the deck was written with is the first of them.
        let first = text.find("Slide 1:").expect("the first slide");
        let second = text.find("Slide 2:").expect("the second slide");
        let third = text.find("Slide 3:").expect("the third slide");
        assert!(
            first < second && second < third,
            "the reader did not number three slides: {text}"
        );
        let (s1, s2, s3) = (&text[first..second], &text[second..third], &text[third..]);
        assert!(
            s1.contains("Второй") && s2.contains("Второй") && s3.contains("Первый"),
            "move_slide did not put the second slide first: {text}"
        );
        // replace_notes rewrote the first slide's notes …
        assert!(
            s1.contains("Slide 1 notes:") && s1.contains("Заметка изменена"),
            "replace_notes did not rewrite the addressed slide's notes: {s1}"
        );
        // … while the duplicate still carries the notes it copied.
        assert!(
            s2.contains("Slide 2 notes:") && s2.contains("Заметка два"),
            "the duplicate did not carry the notes of what it copied: {s2}"
        );
        assert!(
            s3.contains("Slide 3 notes:") && s3.contains("Заметка один"),
            "the untouched slide's notes changed: {s3}"
        );

        // A notes slide whose `<p:ph>` names no type holds the `obj` placeholder
        // the schema defaults to, not the body a reader draws notes from: its
        // paragraphs are still matched by text, but a paragraph added to it would
        // land where no reader shows notes, so adding one is refused.
        let notes = part_text(&source, "ppt/notesSlides/notesSlide1.xml");
        let without_type = notes.replace(r#"type="body" "#, "");
        let bodyless = write_fixture(
            &ws,
            "bodyless.pptx",
            &with_parts(
                &std::fs::read(&source).expect("read created deck"),
                &[("ppt/notesSlides/notesSlide1.xml", without_type.as_bytes())],
            ),
        );
        let noted_reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "noted",
                "path": bodyless.to_string_lossy(),
                "edits": [{ "op": "replace_notes", "slide": 1, "find": "Заметка один",
                            "replace": "Заметка без типа" }],
            }),
        )
        .await;
        let noted_text = converted_text(&ws, &single(&noted_reply)).await;
        assert!(
            noted_text.contains("Slide 1 notes:") && noted_text.contains("Заметка без типа"),
            "the notes text of a placeholder that names no type was not rewritten: {noted_text}"
        );
        let added = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "typed",
                    "path": bodyless.to_string_lossy(),
                    "edits": [{ "op": "add_notes", "slide": 1, "text": "Ещё заметка" }],
                }),
            )
            .await
            .expect_err("a notes shape no reader draws notes from must not take a paragraph");
        let refusal = added.to_string();
        let no_body = "usage: the notes of slide 1 have no body";
        assert!(refusal.starts_with(no_body), "got: {refusal}");
    }

    /// A new pptx paragraph goes into the slide's OWN text — the `<p:txBody>` of
    /// its top-level `<p:sp>` shapes — never into a table's cell, whose
    /// paragraphs a plain "last paragraph" scan would pick up. An `after` that
    /// names text only inside a table is refused rather than placed silently.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_add_paragraph_writes_into_the_slide_text_not_a_table() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Title" },
                        { "type": "paragraph", "text": "Body" },
                        { "type": "table", "headers": ["H1", "H2"],
                          "rows": [["r1c1", "r1c2"], ["r2c1", "r2c2"]] },
                    ],
                }),
            )
            .await,
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_paragraph", "slide": 1, "text": "ADDED-PARA" }],
            }),
        )
        .await;
        let output = single(&reply);
        let slide = part_text(&output, "ppt/slides/slide1.xml");
        assert!(
            !inside_element(&slide, "ADDED-PARA", "<a:tbl>", "</a:tbl>"),
            "the new paragraph landed inside the table: {slide}"
        );
        let text = converted_text(&ws, &output).await;
        assert!(
            text.contains("ADDED-PARA"),
            "the paragraph was lost: {text}"
        );
        assert!(text.contains("r2c2"), "the table was damaged: {text}");

        // Text that lives only inside the table cannot anchor a new paragraph.
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "refused",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "add_paragraph", "slide": 1, "text": "X", "after": "r1c1" }],
                }),
            )
            .await
            .expect_err("an anchor inside a table must be refused");
        assert!(err.to_string().contains("inside a table"), "got: {err}");
    }

    /// A `find` the reader showed but no slide text op changes is refused for what
    /// it really is: a diagram's own text, a label the reader prints and a slide's
    /// speaker notes all read in a reading, so none may be passed off as text the
    /// slide lacks, while real slide text still edits. What a diagram shows is
    /// decided by the reader alone — the tool hands the kit each slide's diagram
    /// text, read with the reader's own rule (its part account and the point's own
    /// `prSet`) — so the kit can only name text a reading really showed.
    /// A find copied out of a reading carries the reader's indentation, the
    /// longest reader label a find names is the one refused, and a diagram the
    /// reader reported as lost is named nothing.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: one case per diagram and label rule
    async fn pptx_edit_names_a_reader_label_and_a_diagrams_text_as_the_reason_it_refuses() {
        // A `dgm:relIds` graphic frame as a reader shows it, its relationship
        // naming the data part that holds the diagram's visible text.
        const DIAGRAM: &str = r#"<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="9" name="Diagram"/><p:cNvGraphicFramePr/></p:nvGraphicFramePr><p:xfrm/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/diagram"><dgm:relIds xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" r:dm="rId99" r:lo="rId100" r:qs="rId101" r:cs="rId102"/></a:graphicData></a:graphic></p:graphicFrame>"#;
        const DIAGRAM_REL: &str = r#"<Relationship Id="rId99" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/>"#;
        const DIAGRAM_DATA: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><dgm:dataModel xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><dgm:ptLst><dgm:pt><dgm:t><a:p><a:r><a:t>Diagram node</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:prSet phldrT="[Text]"/><dgm:t><a:p><a:r><a:t>Hinted node</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:prSet phldr="1" phldrT="[Text]"/><dgm:t><a:p><a:r><a:t>Placeholder node</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:prSet phldr="1"></dgm:prSet><dgm:t><a:p><a:r><a:t>Element placeholder</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:prSet phldr="1"/><dgm:prSet></dgm:prSet><dgm:t><a:p><a:r><a:t>Kept node</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#;
        // The notes part the slide declares through its own relationships, shaped
        // as the kit writes one (`createNotesPart`): a `<p:notes>` holding the
        // body placeholder the reader draws notes from, carrying two note
        // paragraphs.
        const NOTES_REL: &str = r#"<Relationship Id="rId98" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>"#;
        const NOTES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes Placeholder 2"/><p:cNvSpPr/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" dirty="0"/><a:t>Note A</a:t></a:r></a:p><a:p><a:r><a:rPr lang="en-US" dirty="0"/><a:t>Note B</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>"#;
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Slide title" },
                        { "type": "paragraph", "text": "Real slide text" },
                    ],
                }),
            )
            .await,
        );
        let slide = with_first(
            &part_text(&created, "ppt/slides/slide1.xml"),
            "<p:spTree>",
            &format!("<p:spTree>{DIAGRAM}"),
        );
        let rels = with_first(
            &part_text(&created, "ppt/slides/_rels/slide1.xml.rels"),
            "</Relationships>",
            &format!("{DIAGRAM_REL}{NOTES_REL}</Relationships>"),
        );
        let source = write_fixture(
            &ws,
            "diagram.pptx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[
                    ("ppt/slides/slide1.xml", slide.as_bytes()),
                    ("ppt/slides/_rels/slide1.xml.rels", rels.as_bytes()),
                    ("ppt/diagrams/data1.xml", DIAGRAM_DATA.as_bytes()),
                    ("ppt/notesSlides/notesSlide1.xml", NOTES.as_bytes()),
                ],
            ),
        );

        for (find, expected) in [
            ("Diagram node", "is inside a diagram on slide 1"),
            ("    Diagram node", "is inside a diagram on slide 1"),
            ("Hinted node", "is inside a diagram on slide 1"),
            ("Placeholder node", "is not on slide 1"),
            // A paired `<dgm:prSet phldr="1"></dgm:prSet>` is the point's own
            // `prSet`, so the point is a placeholder a reader shows no text of.
            ("Element placeholder", "is not on slide 1"),
            // A later `prSet` without the flag clears it, so the reader shows the
            // text the point holds.
            ("Kept node", "is inside a diagram on slide 1"),
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pptx_edit", "file_name": "refused",
                        "path": source.to_string_lossy(),
                        "edits": [{ "op": "replace_text", "slide": 1, "find": find, "replace": "x" }],
                    }),
                )
                .await
                .expect_err("a find the slide does not hold must be refused");
            assert!(
                err.to_string().contains(expected),
                "find {find:?} got: {err}"
            );
        }

        // Every slide text op looks its text up on the slide, so each is handed the
        // slide's diagram text and each refuses a lookup that names one: the
        // pairing between the ops the tool reads diagram text for and the ops whose
        // refusal the kit words is exercised for the whole vocabulary, not for
        // `replace_text` alone.
        for edits in [
            json!([{ "op": "remove_text", "slide": 1, "find": "Diagram node" }]),
            json!([{ "op": "format_text", "slide": 1, "find": "Diagram node", "bold": true }]),
            json!([{ "op": "remove_paragraph", "slide": 1, "find": "Diagram node" }]),
            json!([{ "op": "add_paragraph", "slide": 1, "text": "X", "after": "Diagram node" }]),
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pptx_edit", "file_name": "refused",
                        "path": source.to_string_lossy(),
                        "edits": edits,
                    }),
                )
                .await
                .expect_err("a lookup the slide does not hold must be refused");
            assert!(
                err.to_string().contains("is inside a diagram on slide 1"),
                "edits {edits} got: {err}"
            );
        }

        // A lookup of nothing but whitespace names no text a reading showed, so it
        // is refused before the deck is read — not answered as text the slide does
        // not hold, and never searched for: a search would match every space the
        // slide holds.
        for find in ["  ", "\t", " \n "] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pptx_edit", "file_name": "refused",
                        "path": source.to_string_lossy(),
                        "edits": [{ "op": "replace_text", "slide": 1, "find": find, "replace": "x" }],
                    }),
                )
                .await
                .expect_err("a find that names no text must be refused");
            assert!(
                err.to_string().contains("must name text"),
                "find {find:?} got: {err}"
            );
        }

        // Every mark the reader prints is refused as the label it is — the
        // longest label the find carries wins, so the lost-diagram mark is not
        // answered with the shorter `(diagram text)` heading written inside it.
        for mark in [
            crate::docgen::ppt_marks().title.as_str(),
            crate::docgen::ppt_marks().hidden_slide.as_str(),
            crate::docgen::ppt_marks().diagram_text.as_str(),
            crate::docgen::ppt_marks().diagram_text_lost.as_str(),
            crate::docgen::ppt_marks().no_text.as_str(),
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pptx_edit", "file_name": "refused",
                        "path": source.to_string_lossy(),
                        "edits": [{ "op": "replace_text", "slide": 1, "find": mark, "replace": "x" }],
                    }),
                )
                .await
                .expect_err("a find the slide does not hold must be refused");
            assert!(
                err.to_string()
                    .contains(&format!("{mark:?} is a label the reader prints")),
                "find {mark:?} got: {err}"
            );
        }

        // A fragment the slide's notes hold is refused as the notes it is: it
        // reads under the reader's `Slide 1 notes:` label, and the notes ops are
        // the ones that edit it. The reader indents a notes line under that
        // label, so a find copied out of a reading is matched with the
        // indentation trimmed away — and a find copied across the notes' lines
        // matches them, since the reader prints one line per note paragraph.
        for find in [
            "Note A",
            "  Note A",
            "\tNote A",
            "Note A\nNote B",
            "  Note A\n  Note B",
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pptx_edit", "file_name": "refused",
                        "path": source.to_string_lossy(),
                        "edits": [{ "op": "replace_text", "slide": 1, "find": find, "replace": "x" }],
                    }),
                )
                .await
                .expect_err("a find the slide's notes hold must be refused");
            assert!(
                err.to_string()
                    .contains("is in the speaker notes of slide 1"),
                "find {find:?} got: {err}"
            );
        }
        // The notes ops do edit it, so the refusal names the right op rather than
        // a slide that lacks the text.
        let noted = single(
            &run(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "noted",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "replace_notes", "slide": 1, "find": "Note A", "replace": "Changed note" }],
                }),
            )
            .await,
        );
        assert!(
            part_text(&noted, "ppt/notesSlides/notesSlide1.xml").contains("Changed note"),
            "replace_notes did not rewrite the notes it owns"
        );

        // The honest path is not narrowed: real slide text still edits.
        let output = single(
            &run(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "edited",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "replace_text", "slide": 1, "find": "Real slide text", "replace": "Changed slide text" }],
                }),
            )
            .await,
        );
        assert!(
            part_text(&output, "ppt/slides/slide1.xml").contains("Changed slide text"),
            "the replacement was lost"
        );

        // A diagram part the package cut short is one the reader reports as lost,
        // so its bytes hold no text a reading showed and the kit names nothing a
        // diagram's: the same find falls through to the ordinary refusal.
        let cut_source = write_fixture(
            &ws,
            "cut-diagram.pptx",
            &with_parts(
                &std::fs::read(&created).expect("read base package"),
                &[
                    ("ppt/slides/slide1.xml", slide.as_bytes()),
                    ("ppt/slides/_rels/slide1.xml.rels", rels.as_bytes()),
                    (
                        "ppt/diagrams/data1.xml",
                        br#"<?xml version="1.0"?><dgm:dataModel xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"><dgm:ptLst><dgm:pt><dgm:t><a:p><a:r><a:t>Diagram node"#,
                    ),
                ],
            ),
        );
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "refused",
                    "path": cut_source.to_string_lossy(),
                    "edits": [{ "op": "replace_text", "slide": 1, "find": "Diagram node", "replace": "x" }],
                }),
            )
            .await
            .expect_err("a find the slide does not hold must be refused");
        assert!(
            err.to_string().contains("is not on slide 1"),
            "a lost diagram's text must not be named a diagram's: {err}"
        );
    }

    /// A package whose parts are not the ones a plain number spells — the layout it
    /// holds is `slideLayout01.xml`, and the notes master it declares is the second
    /// of the two it holds — is named by the parts it really has: a relationship
    /// built from a number resolves to nothing. The model-facing description of
    /// `add_notes` promises a notes slide is written for a slide that has none, so
    /// this is also where the master it is wired to is checked: the one the deck
    /// declares, not the first the package happens to hold.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_names_the_parts_a_leading_zero_deck_really_has() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = write_fixture(&ws, "zero.pptx", &leading_zero_deck());
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "add_slide", "after": 1, "title": "NEW" },
                    { "op": "add_notes", "slide": 1, "text": "заметка" },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        let layout = part_text(&output, "ppt/slides/_rels/slide2.xml.rels");
        assert!(
            layout.contains(r#"Target="../slideLayouts/slideLayout01.xml""#),
            "the new slide names a layout the package does not have: {layout}"
        );
        let notes = part_text(&output, "ppt/notesSlides/_rels/notesSlide1.xml.rels");
        assert!(
            notes.contains(r#"Target="../notesMasters/notesMaster02.xml""#),
            "the created notes slide is not wired to the master the deck declares: {notes}"
        );
    }

    /// A deck whose slide list holds fewer entries than the reader numbers — an
    /// entry outside the list — refuses a `duplicate_slide` of the last slide with
    /// its own cause: a copy is placed where the reader shows it, and the list is
    /// what cannot take it, never a placement nobody asked for.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_duplicate_slide_refuses_a_list_that_numbers_fewer_slides() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "One" },
                        { "type": "heading", "level": 1, "text": "Two" },
                        { "type": "heading", "level": 1, "text": "Three" },
                    ],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        // A third slide the reader numbers — the `<p:sldId>` the list itself does
        // not hold, reusing the third slide's own relationship.
        let rels = part_text_bytes(&base, "ppt/_rels/presentation.xml.rels");
        let target = rels
            .find(r#"Target="slides/slide3.xml""#)
            .expect("the third slide's relationship");
        let open = rels[..target].rfind(r#"Id=""#).expect("an id") + r#"Id=""#.len();
        let id = &rels[open..][..rels[open..].find('"').expect("the id's end")];
        let presentation = part_text_bytes(&base, "ppt/presentation.xml").replace(
            "</p:sldIdLst>",
            &format!(r#"</p:sldIdLst><p:foo><p:sldId id="900" r:id="{id}"/></p:foo>"#),
        );
        let source = write_fixture(
            &ws,
            "stray.pptx",
            &with_parts(&base, &[("ppt/presentation.xml", presentation.as_bytes())]),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "duplicate_slide", "slide": 4 }],
                }),
            )
            .await
            .expect_err("a slide the list does not hold cannot be duplicated");
        let text = err.to_string();
        assert!(
            text.contains("does not number its slides the way the reader shows them"),
            "the refusal does not name its cause: {text}"
        );
        assert!(
            !text.contains("placed after"),
            "the refusal words a duplicate as a placement: {text}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A `<p:sldId>` sitting outside the presentation's list is numbered by the
    /// reader and not by the list, so the entry at a position in the list is the
    /// slide after the one the reader shows there: a move and a copy of the reader's
    /// slide are refused rather than performed on the wrong one, and a delete takes
    /// the entry the reader numbered — never one belonging to another slide, which
    /// would leave both a dangling `<p:sldId>` and an orphaned part.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_a_sld_id_outside_the_slide_list_is_not_addressable_by_position() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = stray_sld_id_deck(&ws, "stray.pptx").await;
        let before = generated_count(&ws);
        for edit in [
            json!({ "op": "move_slide", "slide": 1, "to": 3 }),
            json!({ "op": "duplicate_slide", "slide": 2 }),
        ] {
            let err = DocumentTool
                .execute(
                    &ws,
                    json!({
                        "action": "pptx_edit", "path": source.to_string_lossy(),
                        "edits": [edit],
                    }),
                )
                .await
                .expect_err("a position the list does not number is not one to act on");
            assert!(
                err.to_string()
                    .contains("does not number its slides the way the reader shows them"),
                "the refusal does not name its cause: {err}"
            );
        }
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let output = single(
            &run(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "edited",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "delete_slide", "slide": 2 }],
                }),
            )
            .await,
        );
        let names = zip_names(&output);
        assert!(
            !names.iter().any(|name| name == "ppt/slides/slide1.xml"),
            "the addressed slide's part was left behind: {names:?}"
        );
        assert!(
            ["ppt/slides/slide2.xml", "ppt/slides/slide4.xml"]
                .iter()
                .all(|kept| names.iter().any(|name| name == kept)),
            "a slide another entry numbers was removed: {names:?}"
        );
        assert_eq!(
            part_text(&output, "ppt/presentation.xml")
                .matches("<p:sldId ")
                .count(),
            3,
            "a <p:sldId> was left dangling or an extra one was removed"
        );
        let text = converted_text(&ws, &output).await;
        assert!(
            !text.contains("FIRST"),
            "the deleted slide's own text is still shown: {text}"
        );
        let (stray, second, third) = (
            text.find("STRAY").expect("the stray slide"),
            text.find("SECOND").expect("the second slide"),
            text.find("THIRD").expect("the third slide"),
        );
        assert!(
            stray < second && second < third,
            "the reader's order changed: {text}"
        );
    }

    /// A deck whose slide declares speaker notes the package does not hold cannot
    /// be duplicated: a copy repeats the source's notes, and a copy left naming the
    /// declaration the source keeps would put one notes part behind two slides, so
    /// a later edit of either would change both. The refusal names the declaration,
    /// and nothing is written.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_duplicate_slide_refuses_a_declared_notes_part_the_deck_lacks() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Один" },
                        { "type": "notes", "text": "Заметка" },
                    ],
                }),
            )
            .await,
        );
        let notes = zip_names(&created)
            .into_iter()
            .find(|name| name.starts_with("ppt/notesSlides/notesSlide"))
            .expect("the slide's notes part");
        let source = write_fixture(
            &ws,
            "dangling.pptx",
            &without_part(&std::fs::read(&created).expect("read base package"), &notes),
        );
        // The slide still declares those notes; the part they name is gone.
        assert!(
            part_text(&source, "ppt/slides/_rels/slide1.xml.rels").contains("notesSlide"),
            "the fixture lost the notes declaration it is about"
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "duplicate_slide", "slide": 1 }],
                }),
            )
            .await
            .expect_err("a slide whose notes are declared but missing cannot be copied");
        let text = err.to_string();
        assert!(
            text.contains("declares speaker notes") && text.contains(&notes),
            "the refusal does not name the notes declaration it is about: {text}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A media part `add_image` writes is declared with the content type the kit
    /// read the image as. A deck the tool created declares no `<Default>` for a
    /// media extension it holds no part for, so a JPEG added to such a deck has no
    /// `jpg` default to fall back on: its content type can only come from the
    /// part's own `<Override>`, which is what the bytes are checked against.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_declares_an_added_images_own_content_type() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [{ "type": "paragraph", "text": "Слайд" }],
                }),
            )
            .await,
        );
        let photo = write_fixture(&ws, "photo.jpg", &noisy_jpeg(60, 30));

        // The image is added to the deck the tool created, and to a copy of it whose
        // content types already declare the media part the image will take — a
        // declaration of a part that is not there, which the edit has to correct
        // rather than leave standing.
        let bytes = std::fs::read(&source).expect("read the created deck");
        let declaring = part_text_bytes(&bytes, "[Content_Types].xml").replace(
            "</Types>",
            r#"<Override PartName="/ppt/media/image1.jpg" ContentType="image/jpg"/></Types>"#,
        );
        for (name, deck) in [
            ("created.pptx", bytes.clone()),
            (
                "declaring.pptx",
                with_parts(&bytes, &[("[Content_Types].xml", declaring.as_bytes())]),
            ),
        ] {
            let deck = write_fixture(&ws, name, &deck);
            let output = single(
                &run(
                    &ws,
                    json!({
                        "action": "pptx_edit", "file_name": "edited",
                        "path": deck.to_string_lossy(),
                        "edits": [{ "op": "add_image", "slide": 1, "path": photo.to_string_lossy() }],
                    }),
                )
                .await,
            );
            let types = part_text(&output, "[Content_Types].xml");
            let media: Vec<&str> = types
                .match_indices("<Override ")
                .map(|(at, _)| types[at..].split('>').next().expect("a tag"))
                .filter(|tag| tag.contains(r#"PartName="/ppt/media/"#))
                .collect();
            let [declared] = media.as_slice() else {
                panic!("the added image is not one declared media part of {name}: {media:?}");
            };
            assert!(
                declared.contains(r#"ContentType="image/jpeg""#),
                "the added JPEG is not declared as image/jpeg in {name}: {types}"
            );
        }
    }

    /// A pptx format fragment that is exactly its paragraph widens to nothing,
    /// while a partial one raises both of the kit's caveats — the formatting took
    /// the rest of the run and the alignment the rest of the paragraph — and two
    /// added images land as the slide's last shapes: one where the request placed
    /// it, one centred and scaled to fit.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    #[expect(clippy::too_many_lines)] // reason: one deck carrying every formatting and image assertion
    async fn pptx_edit_formats_text_places_images_and_reports_a_partial_fragment() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Заголовок" },
                        { "type": "paragraph", "text": "Целый абзац" },
                        { "type": "paragraph", "text": "Второй абзац тут" },
                        { "type": "heading", "level": 1, "text": "Второй слайд" },
                    ],
                }),
            )
            .await,
        );
        let image = noisy_png(200, 100);
        let picture = write_fixture(&ws, "pic.png", &image);
        let path = source.to_string_lossy();

        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": path,
                "edits": [
                    { "op": "format_text", "slide": 1, "find": "Целый абзац",
                      "bold": true, "italic": true, "underline": true, "size": 20,
                      "color": "#FF0000", "align": "center" },
                    { "op": "format_text", "slide": 1, "find": "Второй абзац",
                      "color": "00FF00", "align": "left" },
                    { "op": "add_image", "slide": 1, "path": picture.to_string_lossy(),
                      "x": 0.1, "y": 0.2, "width": 0.4 },
                    { "op": "add_image", "slide": 1, "path": picture.to_string_lossy() },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        // "Второй абзац" is only part of its paragraph's text, so the kit says
        // the formatting widened to the rest of the run and that aligning it
        // aligned the whole paragraph.
        for note in ["part of a longer run", "alignment is the whole paragraph's"] {
            assert!(reply.contains(note), "{note} was not raised: {reply}");
        }

        // Both additions are the package's media parts, byte for byte the file
        // the request named.
        let media: Vec<String> = zip_names(&output)
            .into_iter()
            .filter(|name| name.starts_with("ppt/media/image"))
            .collect();
        assert_eq!(
            media,
            ["ppt/media/image1.png", "ppt/media/image2.png"],
            "the added images are not the deck's two media parts"
        );
        for name in &media {
            assert_eq!(
                part_bytes(&output, name),
                image,
                "{name} is not the fixture image"
            );
        }

        let slide = part_text(&output, "ppt/slides/slide1.xml");
        let pictures: Vec<&str> = slide
            .split("<p:pic>")
            .skip(1)
            .map(|rest| rest.split("</p:pic>").next().expect("a picture close"))
            .collect();
        assert_eq!(
            pictures.len(),
            2,
            "the slide does not hold exactly two pictures: {slide}"
        );
        let rels = part_text(&output, "ppt/slides/_rels/slide1.xml.rels");
        // The picture the request placed: 0.1 × 0.2 of the slide, 0.4 of its
        // width and — the image being twice as wide as it is tall — half of that
        // again as its height.
        let placed = pictures[0];
        let embed = embedding_id(placed);
        let relationship = rels
            .split("<Relationship")
            .find(|rel| rel.contains(&format!("Id=\"{embed}\"")))
            .expect("the placed picture's relationship");
        assert!(
            relationship.contains(r#"Target="../media/image1.png""#),
            "the placed picture does not draw image1.png: {relationship}"
        );
        assert!(
            placed.contains(r#"x="914400" y="1028700""#)
                && placed.contains(r#"cx="3657600" cy="1828800""#),
            "the placed picture's geometry is not the requested fraction: {placed}"
        );
        // The picture with neither a place nor a size: centred and the largest
        // that fits the slide with the image's proportions.
        let centred = pictures[1];
        let embed = embedding_id(centred);
        let relationship = rels
            .split("<Relationship")
            .find(|rel| rel.contains(&format!("Id=\"{embed}\"")))
            .expect("the centred picture's relationship");
        assert!(
            relationship.contains(r#"Target="../media/image2.png""#),
            "the centred picture does not draw image2.png: {relationship}"
        );
        assert!(
            centred.contains(r#"x="0" y="285750""#)
                && centred.contains(r#"cx="9144000" cy="4572000""#),
            "the picture with no place or size is not centred and fitted: {centred}"
        );

        let formatting = paragraph_element(&slide, "Целый абзац");
        assert!(
            formatting.contains(r#"algn="ctr""#),
            "the whole-paragraph fragment did not align its paragraph: {formatting}"
        );
        let formatted_run = &formatting[formatting
            .find("<a:rPr")
            .expect("the formatted run's properties")..];
        for attribute in [r#"b="1""#, r#"i="1""#, r#"u="sng""#, r#"sz="2000""#] {
            assert!(
                formatted_run.contains(attribute),
                "the formatted run lost {attribute}: {formatted_run}"
            );
        }
        assert!(
            formatted_run.contains(r#"<a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>"#),
            "the formatted run did not take the requested colour: {formatted_run}"
        );

        let partial = paragraph_element(&slide, "Второй абзац тут");
        assert!(
            partial.contains(r#"algn="l""#),
            "the partial fragment did not align its paragraph: {partial}"
        );
        assert!(
            partial.contains(r#"<a:srgbClr val="00FF00"/>"#),
            "the partial fragment did not colour its run: {partial}"
        );

        let text = converted_text(&ws, &output).await;
        for expected in ["Целый абзац", "Второй абзац тут", "Второй слайд"]
        {
            assert!(
                text.contains(expected),
                "{expected} was lost from the deck: {text}"
            );
        }

        // The same formatting alone, on a fragment that IS the whole paragraph:
        // it covers every point of its run, so it raises neither caveat.
        let alone = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "formatted",
                "path": path,
                "edits": [
                    { "op": "format_text", "slide": 1, "find": "Целый абзац",
                      "bold": true, "italic": false, "underline": true, "size": 20,
                      "color": "#FF0000", "align": "center" },
                ],
            }),
        )
        .await;
        for note in ["part of a longer run", "alignment is the whole paragraph's"] {
            assert!(
                !alone.contains(note),
                "a whole-paragraph fragment raised {note}: {alone}"
            );
        }
        // An explicit `false` is a written value, not an absence: the run carries
        // italic off so a placeholder, layout or theme cannot give it back.
        let formatted = single(&alone);
        let slide = part_text(&formatted, "ppt/slides/slide1.xml");
        let formatting = paragraph_element(&slide, "Целый абзац");
        assert!(
            formatting.contains(r#"i="0""#),
            "the explicit italic false was not written onto the run: {formatting}"
        );
    }

    /// A slide whose package holds no notes part is given one: `add_notes` writes
    /// a fresh notes slide whose body placeholder carries the text, wires a
    /// notes-master relationship into it, declares it in the content types and
    /// gives the slide a relationship to it — which is what the reader shows.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_add_notes_creates_the_notes_part_a_slide_lacks() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "Первый" },
                        { "type": "paragraph", "text": "Тело" },
                        { "type": "heading", "level": 1, "text": "Второй" },
                    ],
                }),
            )
            .await,
        );
        // pptxgenjs writes an empty notes slide for every slide it makes, so the
        // second slide is cooked into one whose package holds no notes at all:
        // its only relationship to them is what a reader goes by.
        let bytes = std::fs::read(&created).expect("read base package");
        let rels = part_text(&created, "ppt/slides/_rels/slide2.xml.rels");
        let without_notes = without_relationship(&rels, "notesSlide");
        let source = write_fixture(
            &ws,
            "deck.pptx",
            &with_parts(
                &bytes,
                &[("ppt/slides/_rels/slide2.xml.rels", without_notes.as_bytes())],
            ),
        );

        let output = single(
            &run(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "noted",
                    "path": source.to_string_lossy(),
                    "edits": [{ "op": "add_notes", "slide": 2, "text": "Заметка добавлена" }],
                }),
            )
            .await,
        );

        let notes: Vec<String> = zip_names(&output)
            .into_iter()
            .filter(|name| name.starts_with("ppt/notesSlides/notesSlide"))
            .filter(|name| part_text(&output, name).contains("Заметка добавлена"))
            .collect();
        assert_eq!(
            notes.len(),
            1,
            "the created notes part is not the only one holding the text: {notes:?}"
        );
        let notes_part = &notes[0];
        let xml = part_text(&output, notes_part);
        assert!(
            xml.contains(r#"<p:ph type="body""#),
            "{notes_part} has no notes body placeholder: {xml}"
        );
        assert!(
            inside_element(&xml, "Заметка добавлена", "<p:txBody>", "</p:txBody>"),
            "the added notes are not in a text body: {xml}"
        );

        let slide_rels = part_text(&output, "ppt/slides/_rels/slide2.xml.rels");
        let relationship = slide_rels
            .split("<Relationship")
            .find(|rel| rel.contains("notesSlide"))
            .expect("the slide's notes relationship");
        let target = format!(
            "../{}",
            notes_part
                .strip_prefix("ppt/")
                .expect("a package-relative part")
        );
        assert!(
            relationship.contains(&format!("Target=\"{target}\"")),
            "the slide's notes relationship does not name {notes_part}: {relationship}"
        );

        let file_name = notes_part.rsplit('/').next().expect("a notes file name");
        let notes_rels = part_text(&output, &format!("ppt/notesSlides/_rels/{file_name}.rels"));
        assert!(
            notes_rels.contains("notesMaster"),
            "{notes_part} has no notes master relationship: {notes_rels}"
        );

        let types = part_text(&output, "[Content_Types].xml");
        assert!(
            types.contains(&format!(r#"<Override PartName="/{notes_part}""#)),
            "[Content_Types].xml declares no override for {notes_part}: {types}"
        );

        let text = converted_text(&ws, &output).await;
        assert!(
            text.contains("Slide 2 notes:"),
            "the reader shows no notes for the second slide: {text}"
        );
        assert!(
            text.contains("Заметка добавлена"),
            "the created notes were lost: {text}"
        );
    }

    /// A presentation's text is written WITHOUT `xml:space`: DrawingML declares no
    /// attributes on `<a:t>`, so the attribute belongs to WordprocessingML's `w:t`
    /// (and is ordinary on SpreadsheetML's `<t>`) but not here, and a presentation
    /// that carries it offers to repair itself. A significant space needs no
    /// attribute to survive — element text reaches a reader whole — so the spaces
    /// an edit writes are in the part verbatim.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_writes_no_space_attribute_and_keeps_significant_spaces() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [{ "type": "paragraph", "text": "Seed" }],
                }),
            )
            .await,
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "replace_text", "slide": 1, "find": "Seed", "replace": "  kept  " },
                    { "op": "add_paragraph", "slide": 1, "text": "  added  " },
                ],
            }),
        )
        .await;
        let output = single(&reply);
        let slide = part_text(&output, "ppt/slides/slide1.xml");
        assert!(
            !slide.contains("xml:space"),
            "a presentation's text carries the attribute DrawingML does not declare on it: {slide}"
        );
        assert!(
            slide.contains("<a:t>  kept  </a:t>"),
            "the replacement's significant spaces were lost: {slide}"
        );
        assert!(
            slide.contains("<a:t>  added  </a:t>"),
            "the new paragraph's significant spaces were lost: {slide}"
        );
        let text = converted_text(&ws, &output).await;
        assert!(
            text.contains("kept") && text.contains("added"),
            "the edited text was lost: {text}"
        );
    }

    /// A `remove_paragraph` that would leave a slide shape's text body or a
    /// table cell's text body with no paragraph is refused: the body would be
    /// one ECMA-376 calls corrupt. A body that keeps another paragraph can lose
    /// one.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_refuses_removing_a_text_bodys_only_paragraph() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let source = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "TITLE" },
                        { "type": "table", "headers": ["H1"], "rows": [["r1c1"]] },
                    ],
                }),
            )
            .await,
        );
        let before = generated_count(&ws);
        let shape = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "remove_paragraph", "slide": 1, "find": "TITLE" }],
                }),
            )
            .await
            .expect_err("a shape's only paragraph cannot be removed");
        assert!(
            shape.to_string().contains("shape's text body"),
            "got: {shape}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let cell = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "remove_paragraph", "slide": 1, "find": "r1c1" }],
                }),
            )
            .await
            .expect_err("a table cell's only paragraph cannot be removed");
        assert!(
            cell.to_string().contains("table cell's text body"),
            "got: {cell}"
        );
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        // With another paragraph in the shape's own text, the original can go.
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [
                    { "op": "add_paragraph", "slide": 1, "text": "SECOND" },
                    { "op": "remove_paragraph", "slide": 1, "find": "TITLE" },
                ],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;
        assert!(
            !text.contains("TITLE"),
            "the removed paragraph survived: {text}"
        );
        assert!(
            text.contains("SECOND"),
            "the added paragraph was lost: {text}"
        );
    }

    /// `add_slide`'s `after` is a position in the numbering the reader shows: a
    /// `<p:sldId>` without a relationship id names no slide and takes no slot, so
    /// it must not shift the insertion point. A deck that can only be numbered by
    /// file name has no index to honour, and an explicit `after` is refused
    /// rather than appended at the end behind the caller's back.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_add_slide_after_uses_the_readers_numbering() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "FIRST" },
                        { "type": "heading", "level": 1, "text": "SECOND" },
                        { "type": "heading", "level": 1, "text": "THIRD" },
                    ],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let presentation = part_text_bytes(&base, "ppt/presentation.xml")
            .replace("<p:sldIdLst>", "<p:sldIdLst><p:sldId id=\"999\"/>");
        let source = write_fixture(
            &ws,
            "extra-sld-id.pptx",
            &with_parts(&base, &[("ppt/presentation.xml", presentation.as_bytes())]),
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_slide", "after": 2, "title": "NEW" }],
            }),
        )
        .await;
        let output = single(&reply);
        let text = converted_text(&ws, &output).await;
        let second = text.find("SECOND").expect("the second slide");
        let inserted = text.find("NEW").expect("the added slide");
        let third = text.find("THIRD").expect("the third slide");
        assert!(
            second < inserted && inserted < third,
            "the slide did not land at the reader's slide 2: {text}"
        );
        assert!(
            part_text(&output, "ppt/presentation.xml").contains(r#"<p:sldId id="999"/>"#),
            "the entry with no relationship id was rewritten"
        );

        // A deck whose slide list numbers nothing can only be read by file name.
        let unnumbered = write_fixture(
            &ws,
            "unnumbered.pptx",
            &with_parts(
                &base,
                &[(
                    "ppt/presentation.xml",
                    without_sld_id_rel_ids(&part_text_bytes(&base, "ppt/presentation.xml"))
                        .as_bytes(),
                )],
            ),
        );
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "file_name": "refused",
                    "path": unnumbered.to_string_lossy(),
                    "edits": [{ "op": "add_slide", "after": 1, "title": "NEW" }],
                }),
            )
            .await
            .expect_err("a deck with no numbered slide entry cannot honour an index");
        assert!(err.to_string().contains("does not number"), "got: {err}");
    }

    /// A slide paragraph's text is addressed by code point, as a docx
    /// paragraph's is: an astral character before the match is one character
    /// even though it is two UTF-16 code units, and a code-unit offset would
    /// rewrite it together with the match.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_replaces_text_after_an_astral_character() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "astral",
                    "content": [{ "type": "paragraph", "text": format!("{ASTRAL}abc") }],
                }),
            )
            .await,
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": created.to_string_lossy(),
                "edits": [{ "op": "replace_text", "slide": 1, "find": "abc", "replace": "XYZ" }],
            }),
        )
        .await;
        let text = converted_text(&ws, &single(&reply)).await;
        assert!(
            text.contains(&format!("{ASTRAL}XYZ")),
            "the character before the match was rewritten: {text}"
        );
    }

    /// `delete_slide` removes the addressed POSITION of the resolved slide list,
    /// not the `<p:sldId>` whose target matches the addressed part: two entries
    /// may name the same part, and deleting the other one would leave a dangling
    /// `<p:sldId>` and drop a part a remaining slide still reads.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_deletes_the_addressed_slide_position_and_keeps_a_shared_part() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "One" },
                        { "type": "paragraph", "text": "Body one" },
                        { "type": "heading", "level": 1, "text": "Two" },
                        { "type": "paragraph", "text": "Body two" },
                    ],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        // Make both slide relationships point at the first slide: the deck now
        // declares two `<p:sldId>`s naming the same part.
        let rels = part_text_bytes(&base, "ppt/_rels/presentation.xml.rels").replace(
            "Target=\"slides/slide2.xml\"",
            "Target=\"slides/slide1.xml\"",
        );
        let source = write_fixture(
            &ws,
            "dup.pptx",
            &with_parts(
                &base,
                &[("ppt/_rels/presentation.xml.rels", rels.as_bytes())],
            ),
        );
        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "delete_slide", "slide": 2 }],
            }),
        )
        .await;
        let output = single(&reply);
        let presentation = part_text(&output, "ppt/presentation.xml");
        assert_eq!(
            presentation.matches("<p:sldId ").count(),
            1,
            "a <p:sldId> was left dangling or the wrong one was removed: {presentation}"
        );
        assert!(
            presentation.contains(r#"r:id="rId2""#),
            "the first slide's entry was removed: {presentation}"
        );
        assert!(
            zip_names(&output)
                .iter()
                .any(|name| name == "ppt/slides/slide1.xml"),
            "the still-referenced slide part was removed"
        );
        // Every remaining `<p:sldId>` still resolves through the relationships.
        let presentation_rels = part_text(&output, "ppt/_rels/presentation.xml.rels");
        assert!(
            presentation_rels.contains(r#"Id="rId2""#),
            "the relationship the remaining slide names was removed: {presentation_rels}"
        );
    }

    /// A deck whose slide list names no slide cannot be addressed by the reader's
    /// numbering, so `delete_slide` is refused rather than deleting a part and
    /// leaving a `<p:sldId>` dangling.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_delete_slide_refuses_a_deck_whose_slide_list_names_nothing() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "One" },
                        { "type": "paragraph", "text": "Body one" },
                        { "type": "heading", "level": 1, "text": "Two" },
                        { "type": "paragraph", "text": "Body two" },
                    ],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        let source = write_fixture(
            &ws,
            "unnumbered.pptx",
            &with_parts(
                &base,
                &[(
                    "ppt/presentation.xml",
                    without_sld_id_rel_ids(&part_text_bytes(&base, "ppt/presentation.xml"))
                        .as_bytes(),
                )],
            ),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "delete_slide", "slide": 1 }],
                }),
            )
            .await
            .expect_err("a deck whose slide list names nothing cannot delete a slide");
        assert!(err.to_string().contains("slide list"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        // A deck whose slide list names relationship ids nothing resolves: the
        // reader numbers it by file naming, so the entry at the addressed position
        // is not the reader's own and removing it would take a `<p:sldId>`
        // belonging to another slide. The rels keep naming the real parts, so only
        // the list is the broken half.
        let presentation = part_text_bytes(&base, "ppt/presentation.xml");
        let list = presentation.find("<p:sldIdLst>").expect("a slide list");
        let end = presentation.find("</p:sldIdLst>").expect("a slide list");
        let mismatched_deck = format!(
            "{}{}{}",
            &presentation[..list],
            r#"<p:sldIdLst><p:sldId id="256" r:id="rId77"/><p:sldId id="257" r:id="rId88"/></p:sldIdLst>"#,
            &presentation[end + "</p:sldIdLst>".len()..]
        );
        let mismatched = write_fixture(
            &ws,
            "mismatched.pptx",
            &with_parts(
                &base,
                &[("ppt/presentation.xml", mismatched_deck.as_bytes())],
            ),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": mismatched.to_string_lossy(),
                    "edits": [{ "op": "delete_slide", "slide": 2 }],
                }),
            )
            .await
            .expect_err("a deck numbered by file naming cannot delete by list position");
        assert!(err.to_string().contains("slide list"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );
    }

    /// A deck whose slide list carries as many entries as there are slide files,
    /// but whose relationship ids resolve to nothing, is numbered by file naming:
    /// a count coincidence is no proof that the list is the reader's numbering, so
    /// an explicit `after` is refused and the package untouched. Without `after`
    /// there is no position to honour and the slide still appends at the end.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn pptx_edit_add_slide_refuses_an_after_the_slide_list_cannot_number() {
        if runtime_missing() {
            return;
        }
        let (_dir, ws) = workspace();
        let created = single(
            &run(
                &ws,
                json!({
                    "action": "create", "format": "pptx", "file_name": "deck",
                    "content": [
                        { "type": "heading", "level": 1, "text": "One" },
                        { "type": "paragraph", "text": "Body one" },
                        { "type": "heading", "level": 1, "text": "Two" },
                        { "type": "paragraph", "text": "Body two" },
                    ],
                }),
            )
            .await,
        );
        let base = std::fs::read(&created).expect("read base package");
        // One entry per slide file, yet neither id resolves: the list is not what
        // the reader numbers by, so no position in it can be honoured.
        let presentation = part_text_bytes(&base, "ppt/presentation.xml");
        let list = presentation.find("<p:sldIdLst>").expect("a slide list");
        let end = presentation.find("</p:sldIdLst>").expect("a slide list");
        let unnumbered = format!(
            "{}{}{}",
            &presentation[..list],
            r#"<p:sldIdLst><p:sldId id="256" r:id="rId77"/><p:sldId id="257" r:id="rId88"/></p:sldIdLst>"#,
            &presentation[end + "</p:sldIdLst>".len()..]
        );
        let source = write_fixture(
            &ws,
            "unnumbered.pptx",
            &with_parts(&base, &[("ppt/presentation.xml", unnumbered.as_bytes())]),
        );
        let before = generated_count(&ws);
        let err = DocumentTool
            .execute(
                &ws,
                json!({
                    "action": "pptx_edit", "path": source.to_string_lossy(),
                    "edits": [{ "op": "add_slide", "after": 1, "title": "Three" }],
                }),
            )
            .await
            .expect_err("a deck numbered by file naming cannot place a slide after one");
        assert!(err.to_string().contains("slide list"), "got: {err}");
        assert_eq!(
            generated_count(&ws),
            before,
            "a refused call left an output"
        );

        let reply = run(
            &ws,
            json!({
                "action": "pptx_edit", "file_name": "edited",
                "path": source.to_string_lossy(),
                "edits": [{ "op": "add_slide", "title": "Three" }],
            }),
        )
        .await;
        let output = single(&reply);
        let slides = zip_names(&output)
            .into_iter()
            .filter(|name| {
                name.starts_with("ppt/slides/slide")
                    && Path::new(name).extension().is_some_and(|ext| ext == "xml")
            })
            .count();
        assert_eq!(slides, 3, "the appended slide is not in the package");
        let presentation = part_text(&output, "ppt/presentation.xml");
        assert_eq!(
            presentation.matches("<p:sldId ").count(),
            3,
            "the appended slide is not in the slide list: {presentation}"
        );
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

    /// The page cap is on the total a request names, not on one range's own
    /// size: two ranges that add up past the limit must be refused before the
    /// kit ever sees them.
    #[test]
    fn split_groups_refuses_a_request_over_the_page_cap() {
        let err = split_groups(&json!({ "ranges": ["1-6000", "6001-12000"] }))
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
            rules.image_side_px.min < rules.image_side_px.max,
            "an empty image-side range"
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
            rules.pdf_size_points.min < rules.pdf_size_points.max,
            "an empty PDF-size range"
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

    /// Every rule the description states is the one the tool enforces: the
    /// asset carries the placeholders and [`DocumentTool::description`] renders
    /// each from `assets/docgen/rules.json`, so a rule that moved cannot leave a
    /// stale number in what the model reads — and a placeholder left unreplaced
    /// would reach the model as its own literal spelling.
    #[test]
    fn the_description_states_every_rule_it_names() {
        let description = DocumentTool.description();
        assert!(
            !description.contains("{{"),
            "the description left a placeholder unreplaced:\n{description}"
        );
        for value in [
            listed(&RULES.heading_levels),
            RULES.image_side_px.bounds(),
            RULES.slide_position_fraction.bounds(),
            RULES.slide_size_fraction.bounds(),
            RULES.pdf_size_points.bounds(),
            RULES.text_size_points.bounds(),
            RULES.paragraph_indent_points.bounds(),
            RULES.paragraph_spacing_points.bounds(),
            RULES.paragraph_line_points.bounds(),
            RULES.slide_text_size_points.bounds(),
            RULES.cell_font_points.bounds(),
            RULES.column_width_chars.bounds(),
            RULES.row_height_points.bounds(),
            quoted_words(&pair_words(&RULES.slide_alignments)),
            quoted_words(&pair_words(&RULES.paragraph_alignments)),
            quoted_words(&pair_words(&RULES.paragraph_lists)),
            quoted_words(&RULES.cell_alignments),
            quoted_words(&RULES.cell_verticals),
            quoted_words(&RULES.cell_border_sides),
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
            megabytes(crate::util::FILE_MAX_BYTES).to_string(),
        ] {
            assert!(
                states_whole_number(&description, &value),
                "the description does not state the rule it names — {value:?} is missing as a whole number from:\n{description}"
            );
        }
        // A slide mark is not a number, so the whole-number guard above would
        // assert nothing for it: what the description owes the model is to name
        // the mark the reader prints.
        let marks = crate::docgen::ppt_marks();
        for mark in [
            &marks.title,
            &marks.hidden_slide,
            &marks.diagram_text,
            &marks.diagram_text_lost,
            &marks.no_text,
        ] {
            assert!(
                description.contains(mark.as_str()),
                "the description does not name the slide mark {mark:?}"
            );
        }
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
        );
        assert_eq!(
            tools.iter().any(|tool| tool.name() == "document"),
            crate::tools::bun::bun_binary_path().is_some(),
            "the tool must be offered exactly while the managed bun runtime is present"
        );
    }
}
