//! The `document` tool: create `.docx`/`.xlsx`/`.pptx`/`.pdf` files, fill a
//! user's own sample, and edit PDF pages.
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
//! runtime is spawned. The bounds and enumerations the tool and the kit share
//! are stated once — in `assets/docgen/rules.json` ([`RULES_JSON`]), which both
//! sides read — so a rule cannot drift between them.
//!
//! # Naming
//!
//! `file_name` is sanitized to a single path component and made unique against
//! `generated/`, so a model-supplied name can neither escape the directory nor
//! silently overwrite an earlier file. The tool appends the extension it
//! produces — the action's own, or the sample's on a fill — after dropping a
//! trailing OOXML or PDF extension (any of `.docx`, `.docm`, `.xlsx`, `.xlsm`,
//! `.pptx`, `.pptm`, `.pdf`), so a name that already carries one is not doubled.

use crate::docgen;
use crate::{Tool, Workspace};
use anyhow::{Context as _, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashSet;
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
const ACTIONS: [&str; 8] = [
    "create",
    "fill_template",
    "pdf_merge",
    "pdf_split",
    "pdf_rotate",
    "pdf_text",
    "pdf_image",
    "pdf_form_fill",
];

/// The formats `create` understands: the three OOXML families (whose names the
/// kit's `format` field also uses) plus PDF; the value doubles as the extension.
const SUPPORTED_FORMATS: [&str; 4] = [
    crate::ooxml::Family::Docx.name(),
    crate::ooxml::Family::Xlsx.name(),
    crate::ooxml::Family::Pptx.name(),
    "pdf",
];

/// The shared input rules both this tool and the kit enforce: the single
/// statement of the sets and bounds below lives in `assets/docgen/rules.json`,
/// which the kit imports and bun inlines into the committed bundle. The tool
/// reads the same file, so a rule its call is checked against twice has one
/// owner rather than two copies.
const RULES_JSON: &str = include_str!("../../assets/docgen/rules.json");

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

/// The parsed [`RULES_JSON`] (see the file for what each entry governs).
#[derive(serde::Deserialize)]
struct Rules {
    heading_levels: Vec<f64>,
    scalar_kinds: Vec<String>,
    image_extensions: Vec<String>,
    image_side_px: Span,
    pdf_size_points: Span,
    pdf_point_abs_max: f64,
    color_digits: usize,
    degrees_step: i64,
    sheet_name_max: usize,
    sheet_name_forbidden: String,
}

/// [`RULES_JSON`] parsed once. A malformed file is a build fault, caught by the
/// test below rather than by a call.
static RULES: LazyLock<Rules> = LazyLock::new(|| {
    serde_json::from_str(RULES_JSON).expect("assets/docgen/rules.json must be valid JSON")
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
    /// states the rules it enforces, and [`RULES_JSON`] is their one owner, so
    /// the two cannot drift — an asset bound that the rules moved past would
    /// otherwise be told to the model as a stale number.
    fn description(&self) -> String {
        let degrees_step = RULES.degrees_step.to_string();
        let heading_levels = listed(&RULES.heading_levels);
        let image_side_px = RULES.image_side_px.bounds();
        let pdf_size_points = RULES.pdf_size_points.bounds();
        let pdf_point_abs_max = RULES.pdf_point_abs_max.to_string();
        let color_digits = RULES.color_digits.to_string();
        let merge_inputs = MAX_MERGE_INPUTS.to_string();
        let merge_mb = megabytes(MAX_MERGE_BYTES).to_string();
        let split_parts = MAX_SPLIT_PARTS.to_string();
        let input_mb = megabytes(crate::util::FILE_MAX_BYTES).to_string();
        let sheet_name_max = RULES.sheet_name_max.to_string();
        // The set with its backslash escaped, as a quoted string spells one: a
        // bare `\` before the closing quote reads as an escaped quote.
        let sheet_name_forbidden = format!("{:?}", RULES.sheet_name_forbidden);
        crate::prompt::substitute(
            &crate::prompt::load_prompt("tool/document.md"),
            &[
                ("{{degrees_step}}", &degrees_step),
                ("{{heading_levels}}", &heading_levels),
                ("{{image_side_px}}", &image_side_px),
                ("{{pdf_size_points}}", &pdf_size_points),
                ("{{pdf_point_abs_max}}", &pdf_point_abs_max),
                ("{{color_digits}}", &color_digits),
                ("{{merge_inputs}}", &merge_inputs),
                ("{{merge_mb}}", &merge_mb),
                ("{{split_parts}}", &split_parts),
                ("{{input_mb}}", &input_mb),
                ("{{sheet_name_max}}", &sheet_name_max),
                ("{{sheet_name_forbidden}}", &sheet_name_forbidden),
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
                    "description": "PDF actions: workspace path of the PDF."
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
        // The kit's `format` is the sample's FAMILY: a workbook has its own
        // cell-wise filler, while a document and a presentation both go through
        // the template library (which reads the package's file type itself).
        let Some(family) = crate::ooxml::family_of(&template) else {
            // Only the three OOXML families carry `{name}` placeholders; naming
            // the alternative beats the package reader's own parse error.
            anyhow::bail!(
                "usage: a sample must be a .docx/.docm, .pptx/.pptm or .xlsx/.xlsm file, got {name} \
                 — hint: annotate a PDF with the pdf_text, pdf_image or pdf_form_fill actions"
            );
        };
        // The OUTPUT keeps the sample's own extension — an `.xlsm` copy stays an
        // `.xlsm` one, macros and all — which a matched family guarantees it has.
        let extension = template
            .extension()
            .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
            .expect("a matched family means the sample is named with its extension");
        ensure_sample_readable(&template).await?;
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
            require_hex_color(color)?;
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

/// Refuse an encrypted OOXML sample. Such a file is a CFB container that no
/// package reader can open; the read path gives the same file the
/// password-protected verdict, and the kit's own failure would be a zip-level
/// message that says nothing about the password. Only a file whose name claims
/// one of the OOXML families reaches this — a legacy `.doc`/`.xls`/`.ppt` is a
/// CFB container too and is refused by the family check instead.
async fn ensure_sample_readable(path: &Path) -> Result<()> {
    use tokio::io::AsyncReadExt as _;
    let mut magic = [0u8; 8];
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?;
    if file.read_exact(&mut magic).await.is_err() || magic.as_slice() != crate::document::CFB_MAGIC
    {
        return Ok(());
    }
    anyhow::bail!(
        "usage: {} is password-protected — hint: remove the protection, save an unprotected copy \
         and fill that",
        path.display()
    )
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

/// Require a sheet name to satisfy the shared rules: at most `sheet_name_max`
/// characters, none of `sheet_name_forbidden`, and no earlier sheet sharing it
/// ignoring case. The caller has already resolved an absent name to `Sheet<n>`,
/// so only the upper bound is enforced here. The kit's own writer keeps the same
/// rule as its last line, over the names it actually writes.
fn require_sheet_name(name: &str, index: usize, seen: &mut HashSet<String>) -> Result<()> {
    let at = format!("sheets[{index}].name");
    let length = name.chars().count();
    if length > RULES.sheet_name_max {
        anyhow::bail!(
            "usage: {at} must be at most {} characters, got \"{name}\" — hint: shorten the sheet \
             name",
            RULES.sheet_name_max
        );
    }
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
fn listed(values: &[f64]) -> String {
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

/// A present optional string: a wrong type is refused rather than read as an
/// absent argument (the shared [`super::get_opt_str`] is deliberately silent,
/// which would drop the caller's value instead of telling them about it).
fn opt_string<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(value) => Err(super::wrong_type(key, "a string", value)),
    }
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

/// Check that a `pdf_text` colour is the shape the kit parses: the shared rules'
/// `color_digits` hex digits, with an optional leading `#`.
fn require_hex_color(color: &str) -> Result<()> {
    let digits = color.strip_prefix('#').unwrap_or(color);
    if digits.len() == RULES.color_digits && digits.chars().all(|digit| digit.is_ascii_hexdigit()) {
        Ok(())
    } else {
        anyhow::bail!(
            "usage: \"color\" must be {} hex digits, with an optional leading \"#\", got \
             \"{color}\" — hint: like \"#{}\"",
            RULES.color_digits,
            "f".repeat(RULES.color_digits)
        )
    }
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

/// Strip a string down to one safe file-name component: no directory part,
/// control characters, `[`/`]` (the marker's own punctuation) or reserved
/// filesystem punctuation, trimmed of dots/spaces and length-capped.
#[must_use]
fn clean_component(raw: &str) -> String {
    // Any directory part is dropped: the tool chooses the directory, and a name
    // that tried to steer it (a separator, `..`) must not be able to.
    let last = raw.rsplit(['/', '\\']).next().unwrap_or(raw);
    // `neutralized_name` turns the control characters and the `[`/`]` a
    // `[FILE:…]` marker is built from into `_`.
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
    use crate::util::test::noisy_png;
    use crate::workspace::test_ws_named;
    use std::io::{Cursor, Read as _, Write as _};

    /// A distinctive string that only a document with a real Cyrillic-capable
    /// font (and a correct encoding round-trip) can carry.
    const CYRILLIC: &str = "Кириллический текст";

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

    fn zip_names(path: &Path) -> Vec<String> {
        let mut archive = open_zip(path);
        let mut names: Vec<String> = (0..archive.len())
            .map(|index| archive.by_index(index).expect("entry").name().to_string())
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

    /// One ZIP entry as text, for asserting a package's internal shape.
    fn part_text(path: &Path, name: &str) -> String {
        let mut archive = open_zip(path);
        let mut entry = archive.by_name(name).expect("part");
        let mut text = String::new();
        entry.read_to_string(&mut text).expect("read part");
        text
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

    /// Add one part to an in-progress package.
    fn add_part(zip: &mut zip::ZipWriter<Cursor<Vec<u8>>>, name: &str, body: &str) {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .expect("start part");
        zip.write_all(body.as_bytes()).expect("write part");
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

    /// A sample the template library cannot compile — an unbalanced `{`, which
    /// is what a document a user edited by hand can end up with — is the
    /// request's own input too, and the refusal carries what makes it fixable.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
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
            (&sheet, "sheet.docx", "really a xlsx package"),
            (&word, "word.xlsx", "really a docx package"),
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
                message.contains(expected),
                "the refusal must describe the file, not the library's modules: {message}"
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
    /// it directly. The value SETS and BOUNDS the two share now have one
    /// statement in `assets/docgen/rules.json`, read by both sides; this test
    /// still covers the structural shapes and the refusal classification, so a
    /// shape widened on one side alone fails it.
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
                err.to_string().contains("the file is not a whole one"),
                "expected the refusal to name the file's own bytes: {err}"
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
        assert!(
            rules.image_side_px.min < rules.image_side_px.max,
            "an empty image-side range"
        );
        assert!(
            rules.pdf_size_points.min < rules.pdf_size_points.max,
            "an empty PDF-size range"
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
            RULES.pdf_size_points.bounds(),
            RULES.color_digits.to_string(),
            RULES.degrees_step.to_string(),
            RULES.sheet_name_max.to_string(),
            format!("{:?}", RULES.sheet_name_forbidden),
            megabytes(crate::util::FILE_MAX_BYTES).to_string(),
        ] {
            assert!(
                description.contains(&value),
                "the description does not state the rule it names — {value:?} is missing from:\n{description}"
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
