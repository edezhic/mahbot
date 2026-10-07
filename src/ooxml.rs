//! OOXML package reading: the ZIP half of document conversion.
//!
//! [`crate::document`] owns content-first detection and the PDF arm — the one
//! non-ZIP container. Every other format it converts is an OOXML package: a ZIP
//! of XML parts, read here with the shared `zip` + `quick_xml` plumbing. Word,
//! Excel and PowerPoint differ only in which parts hold the text, how a cell
//! address is written, and where the embedded media live — and in how much of a
//! walk the format needs: Word's is the only one that carries tables, tracked
//! changes and content beyond the body part, so its reader is the module
//! [`docx`], built on this module's package helpers.
//!
//! # Invariants
//!
//! - **No panics of its own.** A package that cannot be opened as a ZIP, or
//!   whose defining part is absent or unreadable, degrades to
//!   [`DocOutcome::Unreadable`]; a part that is merely missing costs its own
//!   sheet or slide plus a note, never the package.
//! - **Every entry is bounded** by [`MAX_ZIP_ENTRY_BYTES`], and an artifact is
//!   named from the entry's base name alone, so a crafted `../` entry path can
//!   never escape `out_dir`.

use crate::document::{DocOutcome, SkipReason, SkippedImages, ensure_out_dir};
use quick_xml::Reader;
use quick_xml::events::{BytesRef, BytesStart, Event};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

mod docx;
pub(crate) use docx::convert_docx;

// ── Package parts and bounds ────────────────────────────────────

/// Maximum bytes decompressed from a single ZIP entry, so a lying size header
/// cannot inflate the temp dir. Deliberately not an aggregate bound: the entry
/// count is unbounded.
const MAX_ZIP_ENTRY_BYTES: u64 = 64 * 1024 * 1024;

/// Part holding the workbook's sheet names and their relationship ids.
const WORKBOOK_PART: &str = "xl/workbook.xml";
/// Relationships of the workbook part: sheet relationship id -> sheet part.
const WORKBOOK_RELS_PART: &str = "xl/_rels/workbook.xml.rels";
/// Part holding the strings the `t="s"` cells index into.
const SHARED_STRINGS_PART: &str = "xl/sharedStrings.xml";
/// Prefix of the embedded-media parts in an Excel package.
const XLSX_MEDIA_PREFIX: &str = "xl/media/";
/// Charts in an Excel package: `xl/charts/chart<N>.xml`.
const XLSX_CHARTS_PREFIX: &str = "xl/charts/chart";

/// Part listing the presentation's slides in slide order.
const PPT_PRESENTATION_PART: &str = "ppt/presentation.xml";
/// Relationships of the presentation part: slide relationship id -> slide part.
const PPT_PRESENTATION_RELS_PART: &str = "ppt/_rels/presentation.xml.rels";
/// Base every presentation relationship target resolves against.
const PPT_BASE: &str = "ppt/";
/// Prefix of the slide parts.
const PPT_SLIDES_PREFIX: &str = "ppt/slides/";
/// Prefix of the per-slide relationship parts.
const PPT_SLIDES_RELS_PREFIX: &str = "ppt/slides/_rels/";
/// Prefix of the embedded-media parts in a PowerPoint package.
const PPT_MEDIA_PREFIX: &str = "ppt/media/";
/// Charts in a PowerPoint package: `ppt/charts/chart<N>.xml`.
const PPT_CHARTS_PREFIX: &str = "ppt/charts/chart";

/// Media extensions written through to `out_dir` verbatim. Everything else
/// (emf/wmf/tiff/bmp/gif/svg/...) would need transcoding this module avoids.
const EMBEDDED_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// Extensions accepted as OOXML Word packages (`docm` is a macro-enabled docx).
const DOCX_EXTENSIONS: &[&str] = &["docx", "docm"];
/// Extensions accepted as OOXML Excel packages (`xlsm` is macro-enabled xlsx).
const XLSX_EXTENSIONS: &[&str] = &["xlsx", "xlsm"];
/// Extensions accepted as OOXML PowerPoint packages (`pptm` is macro-enabled pptx).
const PPTX_EXTENSIONS: &[&str] = &["pptx", "pptm"];

/// The OOXML family a file name belongs to. The ZIP magic is shared by all
/// three, so only the name can tell them apart.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Docx,
    Xlsx,
    Pptx,
}

impl Family {
    /// The family's name — the spelling the document kit's `format` field uses.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Docx => "docx",
            Self::Xlsx => "xlsx",
            Self::Pptx => "pptx",
        }
    }
}

/// `path`'s OOXML family, or `None` when its name is not one of them. The
/// extension lists above are the only place the families' extensions live.
#[must_use]
pub(crate) fn family_of(path: &Path) -> Option<Family> {
    if crate::util::has_extension(path, DOCX_EXTENSIONS) {
        Some(Family::Docx)
    } else if crate::util::has_extension(path, XLSX_EXTENSIONS) {
        Some(Family::Xlsx)
    } else if crate::util::has_extension(path, PPTX_EXTENSIONS) {
        Some(Family::Pptx)
    } else {
        None
    }
}

// ── Excel ───────────────────────────────────────────────────────

/// Extract sheet text and embedded images from an `.xlsx`/`.xlsm` package.
///
/// Every sheet the workbook names is walked in workbook order and rendered as
/// `Sheet "<name>":` plus one indented line per valued cell. A sheet whose part
/// is missing costs only itself and a note.
#[must_use]
pub(crate) fn convert_xlsx(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return unreadable(".xlsx");
    };
    let Some(workbook) = read_zip_entry(&mut archive, WORKBOOK_PART).bytes() else {
        return unreadable(".xlsx");
    };
    let Some(sheets) = workbook_sheets(&workbook) else {
        return unreadable(".xlsx");
    };
    let shared = read_zip_entry(&mut archive, SHARED_STRINGS_PART)
        .bytes()
        .map(|xml| shared_strings(&xml));
    let shared = shared.as_deref().unwrap_or_default();
    let rels = read_zip_entry(&mut archive, WORKBOOK_RELS_PART)
        .bytes()
        .map(|xml| relationship_map(&xml))
        .unwrap_or_default();

    ensure_out_dir(out_dir);

    let mut notes = Vec::new();
    let charts = count_chart_parts(&archive, XLSX_CHARTS_PREFIX);
    if charts > 0 {
        notes.push(format!(
            "the table has {charts} chart(s), which are not extracted"
        ));
    }
    let mut skipped = SkippedImages::default();
    let images = write_media_parts(&mut archive, XLSX_MEDIA_PREFIX, out_dir, &mut skipped);

    let mut blocks = Vec::new();
    let mut lost = 0usize;
    for (index, (name, id)) in sheets.iter().enumerate() {
        // A relationship the package does not declare falls back to the
        // conventional part name for that position, so a workbook whose rels
        // cannot be read still yields its sheets instead of nothing.
        let part = rels.get(id).map_or_else(
            || format!("xl/worksheets/sheet{}.xml", index + 1),
            |target| resolve_part("xl/", target),
        );
        let rows = read_zip_entry(&mut archive, &part)
            .bytes()
            .and_then(|xml| sheet_rows(&xml, shared));
        match rows {
            Some((rows, missing_strings)) => {
                lost += missing_strings;
                blocks.push(text_block(
                    &format!("Sheet \"{name}\":"),
                    "(no values)",
                    &rows,
                ));
            }
            None => notes.push(format!("sheet \"{name}\" could not be read")),
        }
    }
    // A workbook whose string table is absent, truncated or partly unreadable
    // leaves every cell that indexes into it empty — for an Excel/Sheets/
    // LibreOffice-authored file that is all of its text, so it is said out loud
    // rather than read as a workbook with blank cells.
    if lost > 0 {
        notes.push(format!(
            "{lost} cell(s) left out: the shared string table does not provide their text"
        ));
    }
    notes.extend(skipped.notes());
    DocOutcome::Text {
        text: blocks.join("\n\n").trim_end().to_string(),
        images,
        notes,
        all_page_text_lost: false,
    }
}

/// The workbook's sheets in document order as `(name, relationship id)`.
/// `None` when the workbook part cannot be read as XML.
fn workbook_sheets(xml: &[u8]) -> Option<Vec<(String, String)>> {
    scan_elements(xml, b"sheet", |event| {
        match (attr(event, b"name"), rel_id(event)) {
            (Some(name), Some(id)) => Some((name, id)),
            _ => None,
        }
    })
}

/// Read `xl/sharedStrings.xml` into the strings a `t="s"` cell's index refers
/// to: each `<si>` is the concatenation of its `<t>` texts, rich-text runs
/// included.
fn shared_strings(xml: &[u8]) -> Vec<String> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut strings = Vec::new();
    let mut current = String::new();
    let mut in_text = false;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                b"si" => current.clear(),
                b"t" => in_text = true,
                _ => {}
            },
            Ok(Event::Text(event)) if in_text => {
                if let Ok(chunk) = event.xml10_content() {
                    current.push_str(&chunk);
                }
            }
            Ok(Event::GeneralRef(event)) if in_text => append_entity(&mut current, &event),
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                b"t" => in_text = false,
                b"si" => strings.push(std::mem::take(&mut current)),
                _ => {}
            },
            // A self-closed `<si/>` is still a shared string, so it must occupy
            // its index rather than shift every later one.
            Ok(Event::Empty(event)) if event.local_name().as_ref() == b"si" => {
                strings.push(std::mem::take(&mut current));
            }
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => {}
        }
        buffer.clear();
    }
    strings
}

/// The cell lines of one worksheet in row order: `{ref}: {value}`, or
/// `{ref}: ={formula}` / `{ref}: {value} (={formula})` for a formula cell — plus
/// how many cells held a `t="s"` string the table does not provide, which the
/// caller reports as a note.
///
/// `None` on any XML error — a sheet part that cannot be read as XML is
/// reported like a missing one.
fn sheet_rows(xml: &[u8], shared: &[String]) -> Option<(Vec<String>, usize)> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut rows = Vec::new();
    let mut lost = 0usize;
    let mut row_number = 0u32;
    // Column of the next cell that carries no `r`: real files omit `r` on
    // cells that follow one another, and the addresses still count distance.
    let mut next_column = 0u32;
    let mut cell: Option<Cell> = None;
    let mut shared_formulas: HashMap<String, String> = HashMap::new();
    let (mut in_value, mut in_text, mut in_formula) = (false, false, false);
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                b"row" => {
                    row_number = attr(&event, b"r")
                        .and_then(|r| r.parse().ok())
                        .unwrap_or(row_number.saturating_add(1));
                    next_column = 0;
                }
                b"c" => cell = Some(Cell::start(&event, row_number, &mut next_column)),
                b"v" => in_value = true,
                b"t" => in_text = true,
                b"f" => {
                    in_formula = true;
                    if let Some(cell) = &mut cell {
                        cell.start_formula(&event);
                    }
                }
                _ => {}
            },
            Ok(Event::Empty(event)) => match event.local_name().as_ref() {
                // A valueless cell still occupies its column, so the counter
                // advances even though there is nothing to emit for it.
                b"c" => {
                    let _ = cell_reference(&event, row_number, &mut next_column);
                }
                b"f" => {
                    if let Some(cell) = &mut cell {
                        cell.resolve_shared_formula(&event, &shared_formulas);
                    }
                }
                _ => {}
            },
            Ok(Event::Text(event)) => {
                if let Some(cell) = &mut cell
                    && let Ok(chunk) = event.xml10_content()
                {
                    if in_formula {
                        cell.formula.push_str(&chunk);
                    } else if in_value || in_text {
                        cell.value.push_str(&chunk);
                    }
                }
            }
            Ok(Event::GeneralRef(event)) => {
                if let Some(cell) = &mut cell {
                    if in_formula {
                        append_entity(&mut cell.formula, &event);
                    } else if in_value || in_text {
                        append_entity(&mut cell.value, &event);
                    }
                }
            }
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                b"v" => in_value = false,
                b"t" => in_text = false,
                b"f" => {
                    in_formula = false;
                    if let Some(cell) = &mut cell {
                        cell.end_formula(&mut shared_formulas);
                    }
                }
                b"c" => {
                    if let Some(mut cell) = cell.take() {
                        let line = cell.line(shared);
                        if cell.shared_lost {
                            lost += 1;
                        }
                        if let Some(line) = line {
                            rows.push(line);
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        buffer.clear();
    }
    Some((rows, lost))
}

/// A cell's address: its own `r` attribute, or one synthesised from `row` and
/// the running column counter. Either way the counter is left on the column
/// after this cell — a cell with no address still occupies its column.
fn cell_reference(event: &BytesStart<'_>, row: u32, next_column: &mut u32) -> String {
    let Some(reference) = attr(event, b"r") else {
        let reference = format!("{}{}", column_letters(*next_column), row);
        *next_column = next_column.saturating_add(1);
        return reference;
    };
    if let Some(index) = column_index(&reference) {
        *next_column = index.saturating_add(1);
    }
    reference
}

/// One `<c>` cell being read.
#[derive(Default)]
struct Cell {
    /// The cell address, from `r` or synthesised from its row and column.
    reference: String,
    /// The `t` attribute: how `value` is interpreted.
    kind: Option<String>,
    /// The `<v>`/inline `<t>` text, raw.
    value: String,
    /// The `<f>` text.
    formula: String,
    /// Whether a `<f>` element was seen.
    has_formula: bool,
    /// `si` of a shared formula definition/reference.
    formula_si: Option<String>,
    /// Whether the `<f>` was `t="shared"`.
    formula_shared: bool,
    /// The cell is a `t="s"` one whose string the shared-string table does not
    /// hold, so its text is lost — reported as a note, never as content.
    shared_lost: bool,
}

impl Cell {
    /// Start a cell from its `<c>` element.
    fn start(event: &BytesStart<'_>, row: u32, next_column: &mut u32) -> Self {
        Self {
            reference: cell_reference(event, row, next_column),
            kind: attr(event, b"t"),
            ..Self::default()
        }
    }

    fn start_formula(&mut self, event: &BytesStart<'_>) {
        self.has_formula = true;
        self.formula_si = attr(event, b"si");
        self.formula_shared = attr(event, b"t").as_deref() == Some("shared");
    }

    /// On `</f>`: a shared definition with text teaches the later references of
    /// the same sheet; a shared reference with none borrows the definition's.
    fn end_formula(&mut self, shared: &mut HashMap<String, String>) {
        if !self.formula_shared {
            return;
        }
        let Some(si) = self.formula_si.take() else {
            return;
        };
        if self.formula.is_empty() {
            if let Some(text) = shared.get(&si) {
                self.formula.clone_from(text);
            }
        } else {
            shared.insert(si, self.formula.clone());
        }
    }

    /// A `<f .../>` shared reference: it carries no text of its own.
    fn resolve_shared_formula(&mut self, event: &BytesStart<'_>, shared: &HashMap<String, String>) {
        self.has_formula = true;
        if attr(event, b"t").as_deref() == Some("shared")
            && let Some(si) = attr(event, b"si")
            && let Some(text) = shared.get(&si)
        {
            self.formula.clone_from(text);
        }
    }

    /// The cell's line, or `None` when it holds neither a value nor a formula.
    fn line(&mut self, shared: &[String]) -> Option<String> {
        let value = self.value(shared);
        if self.has_formula {
            let formula = self.formula.trim();
            return if value.is_empty() {
                Some(format!("{}: ={formula}", self.reference))
            } else {
                Some(format!("{}: {value} (={formula})", self.reference))
            };
        }
        (!value.is_empty()).then(|| format!("{}: {value}", self.reference))
    }

    /// The displayed cell value, resolved by the `t` attribute. A shared-string
    /// or inline string keeps its own spacing; every other type is trimmed of
    /// the indentation a pretty-printed part puts inside `<v>`.
    fn value(&mut self, shared: &[String]) -> String {
        match self.kind.as_deref() {
            // A `t="s"` cell holds an index into the shared-string table, so one
            // the table does not hold is text that was lost (a table that could
            // not be read at all, or a truncated one, leaves every such cell
            // here). The cell then reads empty and the loss is reported through
            // the notes — the channel this module reports every unreadable part
            // on — rather than as text that cannot be told from the workbook's
            // own content.
            Some("s") => {
                let text = self
                    .value
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| shared.get(index))
                    .cloned();
                if let Some(text) = text {
                    text
                } else {
                    self.shared_lost = true;
                    String::new()
                }
            }
            Some("b") => match self.value.trim() {
                "1" => "TRUE".to_owned(),
                "0" => "FALSE".to_owned(),
                other => other.to_owned(),
            },
            Some("inlineStr") => self.value.clone(),
            _ => self.value.trim().to_owned(),
        }
    }
}

/// The spreadsheet column letters for a zero-based column index (`0` is `A`).
/// Shared with [`crate::legacy`], so a cell address is written by one
/// implementation.
pub(crate) fn column_letters(mut index: u32) -> String {
    let mut letters: Vec<char> = Vec::new();
    loop {
        let digit = u32::from(b'A') + index % 26;
        letters.push(char::from_u32(digit).unwrap_or('A'));
        if index < 26 {
            break;
        }
        index = index / 26 - 1;
    }
    letters.reverse();
    letters.into_iter().collect()
}

/// The zero-based column index of a cell reference (`"B12"` -> `1`), or `None`
/// when it does not start with a column letter. Shared with
/// [`crate::tools::document`], so a cell reference is read by one
/// implementation.
pub(crate) fn column_index(reference: &str) -> Option<u32> {
    let mut index = 0u32;
    let mut seen = false;
    for byte in reference.bytes() {
        if !byte.is_ascii_alphabetic() {
            break;
        }
        // An address past `u32` is clamped rather than worth a panic.
        index = index
            .saturating_mul(26)
            .saturating_add(u32::from(byte.to_ascii_uppercase() - b'A') + 1);
        seen = true;
    }
    seen.then_some(index - 1)
}

// ── PowerPoint ──────────────────────────────────────────────────

/// Extract slide text, speaker notes and embedded images from a `.pptx`/`.pptm`
/// package.
#[must_use]
pub(crate) fn convert_pptx(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return unreadable(".pptx");
    };
    let Some(parts) = resolve_slide_parts(&mut archive) else {
        return unreadable(".pptx");
    };

    ensure_out_dir(out_dir);

    let mut notes = Vec::new();
    let charts = count_chart_parts(&archive, PPT_CHARTS_PREFIX);
    if charts > 0 {
        notes.push(format!(
            "the presentation has {charts} chart(s), which are not extracted"
        ));
    }
    let mut skipped = SkippedImages::default();
    let images = write_media_parts(&mut archive, PPT_MEDIA_PREFIX, out_dir, &mut skipped);

    let mut blocks = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let number = index + 1;
        // A slide the presentation declares but the package does not resolve
        // costs its own number and a note: dropping it would renumber the slides
        // after it, and label their content with someone else's number.
        let Some(part) = part.as_deref() else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        let Some(lines) = read_zip_entry(&mut archive, part)
            .bytes()
            .and_then(|xml| slide_text_lines(&xml))
        else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        blocks.push(text_block(&format!("Slide {number}:"), "(no text)", &lines));
        match slide_notes(&mut archive, part) {
            SlideNotes::Text(lines) if !lines.is_empty() => {
                blocks.push(text_lines(&format!("Slide {number} notes:"), &lines));
            }
            // The slide declares a notes part, so the text it holds is lost —
            // which is said rather than reported as a slide with no notes.
            SlideNotes::Unreadable => {
                notes.push(format!("slide {number} notes could not be read"));
            }
            _ => {}
        }
    }
    notes.extend(skipped.notes());
    DocOutcome::Text {
        text: blocks.join("\n\n").trim_end().to_string(),
        images,
        notes,
        all_page_text_lost: false,
    }
}

/// The slide parts in presentation order, one slot per slide the presentation
/// declares; a slot is `None` when that slide's own relationship cannot be
/// resolved, so the slides after it keep their own number rather than being
/// renumbered into its place. `None` for the whole list when the package has
/// neither a presentation part nor any conventionally named slide part — i.e. it
/// is not a presentation at all.
fn resolve_slide_parts<R: Read + Seek>(archive: &mut ZipArchive<R>) -> Option<Vec<Option<String>>> {
    let presentation = read_zip_entry(archive, PPT_PRESENTATION_PART).bytes();
    let parts = match presentation.as_deref() {
        Some(xml) => {
            let resolved = read_zip_entry(archive, PPT_PRESENTATION_RELS_PART)
                .bytes()
                .map(|rels| relationship_map(&rels))
                .and_then(|rels| {
                    scan_elements(xml, b"sldId", rel_id).map(|ids| {
                        ids.iter()
                            .map(|id| rels.get(id).map(|target| resolve_part(PPT_BASE, target)))
                            .collect::<Vec<_>>()
                    })
                });
            match resolved {
                // One resolvable slide is enough to trust the declared order; a
                // list with nothing resolvable at all falls back below.
                Some(parts) if parts.iter().any(Option::is_some) => parts,
                // A presentation whose own slide list cannot be resolved: the
                // parts are conventionally numbered, so their names are the order.
                _ => fallback_slide_parts(archive),
            }
        }
        None => fallback_slide_parts(archive),
    };
    if parts.is_empty() && presentation.is_none() {
        None
    } else {
        Some(parts)
    }
}

/// `ppt/slides/slide<N>.xml` in numeric order — the fallback order for a
/// presentation whose relationship list cannot be read.
fn fallback_slide_parts<R: Read + Seek>(archive: &ZipArchive<R>) -> Vec<Option<String>> {
    let mut slides: Vec<(u32, String)> = archive
        .file_names()
        .filter_map(|name| {
            let number = name
                .strip_prefix(PPT_SLIDES_PREFIX)?
                .strip_prefix("slide")?
                .strip_suffix(".xml")?
                .parse()
                .ok()?;
            Some((number, name.to_owned()))
        })
        .collect();
    slides.sort();
    slides.into_iter().map(|(_, name)| Some(name)).collect()
}

/// A slide's speaker notes, as resolved through the slide's own relationships.
enum SlideNotes {
    /// The slide declares no notes part — the usual case.
    None,
    /// The notes part's text lines.
    Text(Vec<String>),
    /// The slide declares a notes part, but it could not be read.
    Unreadable,
}

/// The speaker notes of the slide part `slide_part`, resolved through its own
/// relationships (never by slide index), as their non-empty text lines.
fn slide_notes<R: Read + Seek>(archive: &mut ZipArchive<R>, slide_part: &str) -> SlideNotes {
    let file_name = Path::new(slide_part)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let rels_part = format!("{PPT_SLIDES_RELS_PREFIX}{file_name}.rels");
    // Relationships are how a slide declares notes at all, so a slide with no
    // relationships part has none — but one that IS there and could not be read
    // may name them, and that is said rather than passed off as a slide with no
    // notes. The reader keeps the two apart rather than rescanning the names.
    let rels = match read_zip_entry(archive, &rels_part) {
        ZipEntry::Missing => return SlideNotes::None,
        ZipEntry::Unreadable => return SlideNotes::Unreadable,
        ZipEntry::Bytes(rels) => rels,
    };
    let Some(relations) = relationships(&rels) else {
        // The part is there, but its XML could not be read: it may name notes.
        return SlideNotes::Unreadable;
    };
    let Some(target) = relations
        .into_iter()
        .find(|rel| rel.kind.ends_with("/notesSlide"))
        .map(|rel| rel.target)
    else {
        return SlideNotes::None;
    };
    match read_zip_entry(archive, &resolve_part(PPT_SLIDES_PREFIX, &target)) {
        ZipEntry::Bytes(xml) => {
            slide_text_lines(&xml).map_or(SlideNotes::Unreadable, SlideNotes::Text)
        }
        ZipEntry::Missing | ZipEntry::Unreadable => SlideNotes::Unreadable,
    }
}

/// The text lines of an `<a:t>`-bearing part (a slide or a notes slide), in
/// document order: a newline at `<a:br/>` and at each `<a:p>` end, and nothing
/// from an `<a:fld>` (the slide-number/date field). Empty paragraphs are
/// dropped; `None` on any XML error.
fn slide_text_lines(xml: &[u8]) -> Option<Vec<String>> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut in_text = false;
    let mut field_depth = 0usize;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                b"t" => in_text = true,
                b"br" => current.push('\n'),
                b"fld" => field_depth += 1,
                _ => {}
            },
            Ok(Event::Empty(event)) if event.local_name().as_ref() == b"br" => current.push('\n'),
            Ok(Event::Text(event)) if in_text && field_depth == 0 => {
                if let Ok(chunk) = event.xml10_content() {
                    current.push_str(&chunk);
                }
            }
            Ok(Event::GeneralRef(event)) if in_text && field_depth == 0 => {
                append_entity(&mut current, &event);
            }
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                b"t" => in_text = false,
                b"fld" => field_depth = field_depth.saturating_sub(1),
                b"p" => push_lines(&mut lines, &mut current),
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        buffer.clear();
    }
    Some(lines)
}

/// Move the paragraph in `current` into `lines`, one trimmed line per `<a:br/>`
/// break, dropping the empty ones.
fn push_lines(lines: &mut Vec<String>, current: &mut String) {
    for line in std::mem::take(current).split('\n') {
        let line = line.trim();
        if !line.is_empty() {
            lines.push(line.to_owned());
        }
    }
}

// ── Shared package plumbing ─────────────────────────────────────

/// `header` then one indented line per entry. Shared with [`crate::legacy`] and
/// the Word reader, so the text shapes stay one implementation.
pub(crate) fn text_lines(header: &str, lines: &[String]) -> String {
    let indented = lines
        .iter()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header}\n{indented}")
}

/// [`text_lines`], or `header (marker)` when there is nothing to list. Shared
/// with [`crate::legacy`] for the same reason.
pub(crate) fn text_block(header: &str, empty_marker: &str, lines: &[String]) -> String {
    if lines.is_empty() {
        return format!("{header} {empty_marker}");
    }
    text_lines(header, lines)
}

/// Write every entry under `prefix` that names a raster file into `out_dir`,
/// counting the rest into `skipped`. An entry under a nested path (`media/a.png`
/// and `media/sub/a.png`) takes a suffixed output name rather than overwriting
/// its predecessor, since only the base name survives.
fn write_media_parts<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    out_dir: &Path,
    skipped: &mut SkippedImages,
) -> Vec<PathBuf> {
    let entries: Vec<String> = archive
        .file_names()
        // A media part names a file, i.e. carries an extension. Directory
        // entries (`media/`) and extension-less names are not media: their base
        // name would otherwise be ingested as a nonexistent image.
        .filter(|name| name.starts_with(prefix) && Path::new(name).extension().is_some())
        .map(str::to_owned)
        .collect();
    let mut images = Vec::new();
    let mut next_suffix = HashMap::new();
    for entry in entries {
        // Only the entry's file name is used: a crafted `../..` entry must
        // never escape `out_dir`.
        let file_name = crate::util::neutralized_name(crate::util::file_name_or_path(&entry));
        if !crate::util::has_extension(Path::new(&file_name), EMBEDDED_IMAGE_EXTENSIONS) {
            skipped.add(SkipReason::Unsupported);
            continue;
        }
        let Some(content) = read_zip_entry(archive, &entry).bytes() else {
            tracing::warn!(%file_name, "document: failed to read embedded image");
            skipped.add(SkipReason::Failed);
            continue;
        };
        let path = unique_out_path(out_dir, &file_name, &mut next_suffix);
        match std::fs::write(&path, &content) {
            Ok(()) => images.push(path),
            Err(e) => {
                tracing::warn!(%file_name, error = %e, "document: failed to write embedded image");
                skipped.add(SkipReason::Failed);
            }
        }
    }
    images
}

/// Count the chart parts under `prefix` (`chart<N>.xml`, the number digits
/// only) — charts are reported, not converted.
fn count_chart_parts<R: Read + Seek>(archive: &ZipArchive<R>, prefix: &str) -> usize {
    archive
        .file_names()
        .filter(|name| {
            name.strip_prefix(prefix)
                .and_then(|rest| rest.strip_suffix(".xml"))
                .is_some_and(|number| {
                    !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit())
                })
        })
        .count()
}

/// One OOXML relationship: the `Id` a part refers to, its `Type` (empty when
/// absent, which no consumer here treats as a notes relationship), and the
/// `Target` part (or part-relative path) it points at.
struct Relationship {
    id: String,
    kind: String,
    target: String,
}

/// Every relationship declared in a `.rels` part. `None` when the part cannot
/// be read as XML — reported like a missing part, never as an empty one.
fn relationships(xml: &[u8]) -> Option<Vec<Relationship>> {
    scan_elements(xml, b"Relationship", |event| {
        let (Some(id), Some(target)) = (attr(event, b"Id"), attr(event, b"Target")) else {
            return None;
        };
        Some(Relationship {
            id,
            kind: attr(event, b"Type").unwrap_or_default(),
            target,
        })
    })
}

/// Relationship id -> target, for the parts a rId alone names.
fn relationship_map(xml: &[u8]) -> HashMap<String, String> {
    relationship_targets(relationships(xml).unwrap_or_default())
}

/// [`relationship_map`] over an already-read relationship list. The Word reader
/// reads the list itself — the notes parts are named there by kind, before the
/// list becomes this lookup — and shares the mapping.
fn relationship_targets(rels: Vec<Relationship>) -> HashMap<String, String> {
    rels.into_iter().map(|rel| (rel.id, rel.target)).collect()
}

/// Resolve a relationship target against the part that owns it: a relative
/// target is appended to `base`, an absolute one (`/xl/...`) is the package
/// path with the leading slash dropped, and `..` segments (a notes part in
/// `ppt/slides/` pointing at `../notesSlides/...`) are normalized away so the
/// result names the ZIP entry it points at.
fn resolve_part(base: &str, target: &str) -> String {
    let path = match target.strip_prefix('/') {
        Some(absolute) => absolute.to_owned(),
        None => format!("{base}{target}"),
    };
    let mut parts: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// The relationship-id attribute of an element that refers to a part
/// (`r:id="rId1"`).
///
/// Matched by local name plus a namespace prefix, not by local name alone: a
/// slide carries both its own unprefixed `id` (`<p:sldId id="256" r:id="rId2"/>`)
/// and the relationship one, and only the prefix tells them apart. Attributes
/// never inherit a default namespace, so a relationship id always has one.
fn rel_id(event: &BytesStart<'_>) -> Option<String> {
    event
        .attributes()
        .with_checks(false)
        .flatten()
        .find(|attribute| {
            attribute.key.local_name().as_ref() == b"id" && attribute.key.prefix().is_some()
        })
        .and_then(|attribute| {
            attribute
                .unescape_value()
                .ok()
                .map(std::borrow::Cow::into_owned)
        })
}

/// The value of the attribute whose local name is `name` (after any namespace
/// prefix), unescaped. `None` when absent or unreadable.
fn attr(event: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    event
        .attributes()
        .with_checks(false)
        .flatten()
        .find(|attribute| attribute.key.local_name().as_ref() == name)
        .and_then(|attribute| {
            attribute
                .unescape_value()
                .ok()
                .map(std::borrow::Cow::into_owned)
        })
}

/// Walk `xml` and collect `collect`'s value for every `Start`/`Empty` element
/// whose local name is `element`, in document order.
///
/// `None` when the part cannot be read as XML at all, which is the one
/// distinction a caller needs: a broken part is reported like a missing one,
/// never as "no such element" ([`workbook_sheets`], [`relationships`]).
fn scan_elements<T>(
    xml: &[u8],
    element: &[u8],
    mut collect: impl FnMut(&BytesStart<'_>) -> Option<T>,
) -> Option<Vec<T>> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut found = Vec::new();
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event) | Event::Empty(event))
                if event.local_name().as_ref() == element =>
            {
                if let Some(value) = collect(&event) {
                    found.push(value);
                }
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        buffer.clear();
    }
    Some(found)
}

/// Resolve one XML entity reference (`&amp;`, `&#x41;`) in run text. An
/// unresolvable reference is dropped rather than aborting the whole body.
fn append_entity(text: &mut String, reference: &BytesRef<'_>) {
    if let Ok(name) = reference.decode()
        && let Some(resolved) = quick_xml::escape::resolve_predefined_entity(&name)
    {
        text.push_str(resolved);
        return;
    }
    if let Ok(Some(character)) = reference.resolve_char_ref() {
        text.push(character);
    }
}

/// A named ZIP entry as the reader found it: absent from the archive, present
/// but not readable within the bounds, or its bytes.
enum ZipEntry {
    /// The archive has no entry under the name.
    Missing,
    /// The entry is there but could not be read whole within
    /// [`MAX_ZIP_ENTRY_BYTES`] — a read failure, a size the bound rejects, or a
    /// name the reader refuses.
    Unreadable,
    /// The entry's bytes.
    Bytes(Vec<u8>),
}

impl ZipEntry {
    /// The entry's bytes when it was read, `None` otherwise. For a caller that
    /// treats absent and unreadable alike, which is most of them; a caller that
    /// must tell them apart matches on the variant.
    fn bytes(self) -> Option<Vec<u8>> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Missing | Self::Unreadable => None,
        }
    }
}

/// Read a named ZIP entry into memory, bounded by [`MAX_ZIP_ENTRY_BYTES`],
/// as [`ZipEntry`]: an absent entry is [`ZipEntry::Missing`], and one that is
/// there but cannot be read whole is [`ZipEntry::Unreadable`]. Neither is fatal
/// — the caller skips the entry (noting why) or reports the package unreadable.
fn read_zip_entry<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> ZipEntry {
    let mut entry = match archive.by_name(name) {
        Ok(entry) => entry,
        Err(zip::result::ZipError::FileNotFound) => return ZipEntry::Missing,
        Err(_) => return ZipEntry::Unreadable,
    };
    // The declared size is checked first, so a bomb that admits its size is
    // rejected before any inflation; the read then goes through the same cap so
    // a lying header cannot exceed it either.
    if entry.size() > MAX_ZIP_ENTRY_BYTES {
        tracing::warn!(
            %name,
            declared_bytes = entry.size(),
            "document: ZIP entry exceeds the decompression bound"
        );
        return ZipEntry::Unreadable;
    }
    let mut bytes = Vec::new();
    // Read one byte past the bound so an entry that is exactly at the limit is
    // still accepted whole while a larger one is detected instead of silently
    // truncated.
    if entry
        .by_ref()
        .take(MAX_ZIP_ENTRY_BYTES + 1)
        .read_to_end(&mut bytes)
        .is_err()
    {
        return ZipEntry::Unreadable;
    }
    if bytes.len() as u64 > MAX_ZIP_ENTRY_BYTES {
        tracing::warn!(%name, "document: ZIP entry exceeds the decompression bound");
        return ZipEntry::Unreadable;
    }
    ZipEntry::Bytes(bytes)
}

/// `out_dir/<file_name>`, suffixed `_2`, `_3`, … when that name is taken (see
/// [`crate::util::suffixed_name`]).
///
/// A package can hold the same base name under different directories
/// (`word/media/a.png` and `word/media/sub/a.png`), and only the base name is
/// used here: without a suffix the first entry would be overwritten and the
/// survivor ingested twice. The check is against the directory, not a counter,
/// because `out_dir` also holds the attachment itself. `next_suffix` memoizes
/// the first suffix still worth trying per name, so a package of many entries
/// sharing one base name does not rescan the names already taken.
fn unique_out_path(
    out_dir: &Path,
    file_name: &str,
    next_suffix: &mut HashMap<String, u32>,
) -> PathBuf {
    let n = next_suffix.entry(file_name.to_owned()).or_insert(1);
    while out_dir
        .join(crate::util::suffixed_name(file_name, *n))
        .exists()
    {
        *n += 1;
    }
    out_dir.join(crate::util::suffixed_name(file_name, *n))
}

/// The shared "recognized OOXML package of this kind that cannot be read"
/// outcome.
fn unreadable(extension: &str) -> DocOutcome {
    DocOutcome::Unreadable {
        reason: format!("corrupt or unsupported {extension}"),
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;
    use std::io::Write;

    pub(crate) const DOCX_BODY: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p><w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p></w:body></w:document>"#;

    /// Build a ZIP archive in memory from `(entry path, contents)` pairs.
    pub(crate) fn zip_fixture(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (path, contents) in entries {
            writer.start_file(*path, options).expect("start zip entry");
            writer.write_all(contents).expect("write zip entry");
        }
        writer.finish().expect("finish zip").into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::*;
    use super::*;

    // ── Excel ───────────────────────────────────────────────────

    const WORKBOOK: &[u8] = br#"<workbook><sheets><sheet name="Data" sheetId="1" r:id="rId1"/><sheet name="More" sheetId="2" r:id="rId2"/></sheets></workbook>"#;

    const WORKBOOK_RELS: &[u8] = br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml"/></Relationships>"#;
    /// One plain string, one rich-text one split across runs, and one written
    /// as numeric character references — the shape a producer that escapes
    /// non-ASCII writes (every Cyrillic letter becomes `&#NNNN;`).
    const SHARED_STRINGS: &[u8] = br"<sst><si><t>hello</t></si><si><r><t>rich</t></r><r><t> text</t></r></si><si><t>&#1046;&#1091;&#1082;</t></si></sst>";

    const SHEET_WITH_CELLS: &[u8] = br#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row><row r="2"><c r="A2"><f>SUM(B1:B2)</f></c><c r="B2" t="b"><v>1</v></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>inline</t></is></c><c><v>7</v></c><c t="str"><f>B1&amp;1</f><v>jog</v></c></row><row r="4"><c r="A4" t="s"><v>1</v></c></row><row r="5"><c r="A5" t="s"><v>2</v></c><c r="B5" t="inlineStr"><is><t>&#1054;&#1090;&#1095;&#1105;&#1090;</t></is></c></row></sheetData></worksheet>"#;

    #[test]
    fn xlsx_reads_cells_in_workbook_order() {
        let bytes = zip_fixture(&[
            ("xl/workbook.xml", WORKBOOK),
            ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS),
            ("xl/sharedStrings.xml", SHARED_STRINGS),
            ("xl/worksheets/sheet1.xml", SHEET_WITH_CELLS),
            (
                "xl/worksheets/sheet2.xml",
                b"<worksheet><sheetData/></worksheet>",
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text {
            text,
            images,
            notes,
            ..
        } = convert_xlsx(&bytes, dir.path())
        else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: hello\n  B1: 42\n  A2: =SUM(B1:B2)\n  B2: TRUE\n  A3: inline\n  B3: 7\n  C3: jog (=B1&1)\n  A4: rich text\n  A5: Жук\n  B5: Отчёт\n\nSheet \"More\": (no values)"
        );
        assert!(images.is_empty(), "a workbook with no media writes none");
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A workbook whose relationship list cannot be read still yields its
    /// sheets: the parts are conventionally numbered, so their names are the
    /// order.
    #[test]
    fn xlsx_falls_back_to_conventional_sheet_parts() {
        let bytes = zip_fixture(&[
            ("xl/workbook.xml", WORKBOOK),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/sheet2.xml",
                br#"<worksheet><sheetData><row r="1"><c r="B1"><v>2</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1\n\nSheet \"More\":\n  B1: 2");
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A sheet whose part is absent costs only itself and a note — the sheets
    /// around it are still delivered.
    #[test]
    fn xlsx_keeps_sheets_whose_part_is_missing() {
        let bytes = zip_fixture(&[
            ("xl/workbook.xml", WORKBOOK),
            ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1");
        assert_eq!(notes, ["sheet \"More\" could not be read"]);
    }

    /// Shared formulas: a definition teaches the references that come after it,
    /// so a reference with no text of its own still reads back as its formula.
    #[test]
    fn xlsx_resolves_shared_formulas() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="S" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><f t="shared" ref="A1:A2" si="0">B1*2</f><v>2</v></c></row><row r="2"><c r="A2"><f t="shared" si="0"/><v>4</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"S\":\n  A1: 2 (=B1*2)\n  A2: 4 (=B1*2)");
    }

    /// Media and charts are reported the same way the Word path reports its
    /// media: a raster is written through, an undecodable entry is counted, and
    /// a chart is named rather than silently dropped.
    #[test]
    fn xlsx_extracts_media_and_notes_charts_it_cannot_draw() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="S" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
            ("xl/media/pic.png", b"\x89PNG\r\n\x1a\nfake image bytes"),
            ("xl/media/diagram.emf", b"EMF bytes this stack cannot decode"),
            ("xl/charts/chart1.xml", b"<chart/>"),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { images, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(images, vec![dir.path().join("pic.png")]);
        assert_eq!(
            notes,
            [
                "the table has 1 chart(s), which are not extracted",
                "skipped 1 embedded image(s) in a format this pipeline cannot convert",
            ]
        );
    }

    /// A `t="s"` cell holds an index into the shared-string table, so one the
    /// table does not hold is text that was lost — a table that is truncated or
    /// partly unreadable, or an index past its end. The cell reads empty and the
    /// loss is reported as a note (the channel every unreadable part in this
    /// module uses), never as text the reader cannot tell from the workbook's own
    /// content.
    #[test]
    fn xlsx_reports_shared_string_cells_the_table_does_not_hold() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
            ),
            // A declared table that stops mid-`<si>`: its first entry is readable
            // and everything after it is not, which is what a truncated part
            // looks like.
            (
                "xl/sharedStrings.xml",
                br"<sst><si><t>First</t></si><si><t>Sec",
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>7</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable workbook");
        };
        assert_eq!(text, "Sheet \"Sheet1\":\n  A1: First");
        assert_eq!(
            notes,
            ["2 cell(s) left out: the shared string table does not provide their text"]
        );

        // The same for a workbook that carries no table at all.
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable workbook");
        };
        assert_eq!(text, "Sheet \"Sheet1\": (no values)");
        assert_eq!(
            notes,
            ["1 cell(s) left out: the shared string table does not provide their text"]
        );
    }

    /// A presentation that declares a slide whose own relationship cannot be
    /// resolved keeps that slide's number: dropping it would renumber the slides
    /// after it and label their content with someone else's number.
    #[test]
    fn pptx_keeps_the_number_of_a_slide_it_cannot_resolve() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/><p:sldId id="258" r:id="rId3"/></p:sldIdLst></p:presentation>"#,
            ),
            // `rId2` is never declared, so the second slide's part is unknown.
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId3" Target="slides/slide2.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>First</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
            ),
            (
                "ppt/slides/slide2.xml",
                br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Third</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a presentation");
        };
        assert_eq!(
            text, "Slide 1:\n  First\n\nSlide 3:\n  Third",
            "the unresolvable slide keeps its own number"
        );
        assert_eq!(notes, ["slide 2 could not be read"]);
    }

    // ── PowerPoint ──────────────────────────────────────────────
    /// Slides resolve in the presentation's own relationship order (never by
    /// part index), speaker notes come through the slide's own relationships
    /// (never by slide index), a slide-number field contributes nothing, a
    /// slide with only a field reads as one with no text, and the media/chart
    /// notes match the other OOXML arms.
    #[test]
    fn pptx_reads_slides_with_notes_and_media() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/><p:sldId id="258" r:id="rId3"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide2.xml"/><Relationship Id="rId2" Target="slides/slide1.xml"/><Relationship Id="rId3" Target="slides/slide3.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide2.xml",
                br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Second slide text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
            ),
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>First slide </a:t></a:r><a:br/><a:r><a:t>continued</a:t></a:r></a:p><a:p><a:fld id="1" type="slidenum"><a:t>7</a:t></a:fld></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#,
            ),
            (
                "ppt/slides/slide3.xml",
                br#"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:fld id="2" type="datetime"><a:t>2026-01-01</a:t></a:fld></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#,
            ),
            // The notes relationship is not the first one, so resolving by
            // position would pick the image instead.
            (
                "ppt/slides/_rels/slide2.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/></Relationships>"#,
            ),
            (
                "ppt/notesSlides/notesSlide1.xml",
                br"<p:notes><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Notes for the second slide</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>",
            ),
            ("ppt/media/pic.png", b"\x89PNG\r\n\x1a\nfake image bytes"),
            ("ppt/charts/chart1.xml", b"<chart/>"),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text {
            text,
            images,
            notes,
            ..
        } = convert_pptx(&bytes, dir.path())
        else {
            panic!("expected Text outcome for a well-formed pptx");
        };
        assert_eq!(
            text,
            "Slide 1:\n  Second slide text\n\nSlide 1 notes:\n  Notes for the second slide\n\nSlide 2:\n  First slide\n  continued\n\nSlide 3: (no text)"
        );
        assert_eq!(images, vec![dir.path().join("pic.png")]);
        assert_eq!(
            notes,
            ["the presentation has 1 chart(s), which are not extracted"]
        );
    }

    /// A slide's notes are announced rather than dropped in silence, however
    /// they are lost: a relationship that resolves to a part the package does
    /// not have, and a slide whose relationships part cannot be read at all
    /// (which is where a slide declares notes in the first place). Neither may
    /// pass for a slide that declares none.
    #[test]
    fn pptx_reports_notes_it_cannot_read() {
        let slide = br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
            .as_slice();
        // The notes relationship resolves to a part the package does not have,
        // and, second, a relationships part cut short before anything resolves.
        let lost: [&[u8]; 2] = [
            br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/></Relationships>"#,
            br#"<Relationships><Relationship Id="rId"#,
        ];
        for rels in lost {
            let bytes = zip_fixture(&[
                (
                    "ppt/presentation.xml",
                    br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#,
                ),
                (
                    "ppt/_rels/presentation.xml.rels",
                    br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/></Relationships>"#,
                ),
                ("ppt/slides/slide1.xml", slide),
                ("ppt/slides/_rels/slide1.xml.rels", rels),
            ]);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_pptx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable presentation");
            };
            assert_eq!(text, "Slide 1:\n  Slide text");
            assert_eq!(notes, ["slide 1 notes could not be read"]);
        }
    }

    // ── Word ────────────────────────────────────────────────────

    /// A media entry name carrying a bracket would close the `[File ...]` note
    /// (and the `[IMAGE:...]` marker built from it) early.
    #[test]
    fn docx_media_entry_names_are_marker_safe() {
        let bytes = zip_fixture(&[
            ("word/document.xml", DOCX_BODY),
            ("word/media/a]b.png", b"\x89PNG\r\n\x1a\nfake image bytes"),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { images, .. } = convert_docx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed docx");
        };
        assert_eq!(images, vec![dir.path().join("a_b.png")]);
    }

    /// Two entries can share a base name under different directories, and only
    /// the base name survives into `out_dir`: each must get its own file, or one
    /// is silently overwritten and the survivor ingested twice.
    #[test]
    fn docx_duplicate_media_base_names_get_distinct_files() {
        let bytes = zip_fixture(&[
            ("word/document.xml", DOCX_BODY),
            ("word/media/pic.png", b"\x89PNG\r\n\x1a\nfirst"),
            ("word/media/sub/pic.png", b"\x89PNG\r\n\x1a\nsecond"),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { images, .. } = convert_docx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed docx");
        };
        assert_eq!(
            images,
            vec![dir.path().join("pic.png"), dir.path().join("pic_2.png")]
        );
    }
}
