//! The Excel (`.xlsx`/`.xlsm`) reader: the workbook part, its shared-string
//! table and the worksheet parts, the cell comments and hyperlinks they name and
//! the workbook's defined names, plus the embedded media and charts.
//!
//! [`super`] owns the ZIP plumbing and the other two OOXML families; this
//! module is the whole Excel half, built on the package helpers there.
//! [`crate::document`] routes `.xlsx`/`.xlsm` files here.
//!
//! # Invariants
//!
//! - **A value is shown the way Excel shows it, or not at all.** A cell's number
//!   format decides what is printed ([`numfmt`]), so a date reads as a date and
//!   a percentage as a percentage; a format this reader cannot read, cannot
//!   reproduce, or that has no display for the value prints the stored value under
//!   its mark ([`MARK_STORED`]) — a raw number is never passed off as the display,
//!   and a wrong one is never invented. One exception is stated rather than marked:
//!   a `t="d"` cell holds its date as ISO text, shown as the workbook wrote it
//!   rather than through its number format.
//! - **A part the package left incomplete is never passed off as a whole one.** The
//!   workbook part's own walk still yields the sheets and names it got to, with the
//!   rest said not read ([`WORKBOOK_CUT_NOTE`]); a sheet, styles or comments part cut
//!   short is declined whole rather than delivered by halves; a string table cut short
//!   leaves the strings it did not reach counted as lost; and a workbook whose
//!   relationship list could not be read is read by its parts' conventional names, so
//!   a table named only there is never found and a cell naming a format past the
//!   schema's default entry 0 carries the number that table does not describe
//!   ([`WORKBOOK_RELS_NOTE`]). A part holding no element of its own kind (no
//!   workbook, no sheet, no comments element) is read like one the package does not
//!   carry; an entry the package left empty, for its part, declares nothing.
//! - **Structure is marked, never silently dropped.** Hidden sheets, hidden rows
//!   and columns, merged ranges, cell comments, hyperlinks and defined names all
//!   reach the answer, and the marks are by range rather than by cell, so a
//!   hidden block costs one line rather than one per cell in it. The merge and
//!   link lists are swept down the sheet's rows ([`RowSweep`]), so a cell is only
//!   tested against a range spanning its own row and each range costs one push and
//!   one pop to track; the tests themselves are bounded by [`MAX_RANGE_PROBES`],
//!   past which the rest of the marks are reported rather than worked through — as
//!   is a marker the sheet's own declaration does not let this reader place (a
//!   `<col>` range past the grid, a merged range or link naming no range).
//! - **The notation the model reads is documented in
//!   `src/prompt/tool/read.md` and `read_strict.md`** — a mark this module adds
//!   belongs there too.

use super::{
    Part, Relationship, ZipEntry, append_entity, attr, blank, column_index, column_letters,
    count_chart_parts, read_zip_entry, rel_id, relationship_targets, relationships_and_whole,
    resolve_part, scan_elements, text_block, text_lines, unreadable, write_media_parts,
};
use crate::document::{DocOutcome, SkippedImages, ensure_out_dir};
use crate::util::is_line_break;
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::io::{Cursor, Read, Seek};
use std::path::Path;
use zip::ZipArchive;

mod numfmt;

// ── Part names ──────────────────────────────────────────────────

/// Part holding the workbook's sheet names and their relationship ids.
const WORKBOOK_PART: &str = "xl/workbook.xml";
/// Relationships of the workbook part: sheet relationship id -> sheet part.
const WORKBOOK_RELS_PART: &str = "xl/_rels/workbook.xml.rels";
/// Part holding the strings the `t="s"` cells index into.
const SHARED_STRINGS_PART: &str = "xl/sharedStrings.xml";
/// Part holding the workbook's cell formats.
const STYLES_PART: &str = "xl/styles.xml";
/// Note pushed when the workbook declares cell formats this reader cannot read.
const STYLES_NOTE: &str =
    "the workbook's cell formats could not be read, so values are shown as stored";
/// Note pushed when the workbook's relationships part is there and cannot be read:
/// its parts are then read by their conventional names, and its sheet parts are
/// matched to their entries by position. A part named only there may be missed
/// altogether, which leaves the cells of that part showing the value as stored —
/// the one loss this note stands for, which is why it says so.
const WORKBOOK_RELS_NOTE: &str = "the workbook's relationships could not be read, so its parts are read by their conventional names and one named only there may be missed, leaving its cells showing the value as stored";
/// Note pushed when the workbook part is cut short: the sheets and defined names it
/// declares after that point were not read, and cannot be named from what is left, so
/// the ones read before it are delivered with the loss said rather than passed off as
/// the whole list.
const WORKBOOK_CUT_NOTE: &str = "the workbook part is cut short, so the sheets and names it declares after that point are not read";
/// Note pushed when the workbook part carries a workbook element declaring no sheet:
/// there is nothing in it to read, and the generic "no text" the delivery layer
/// answers with would leave the cause unnamed.
const WORKBOOK_NO_SHEETS_NOTE: &str = "the workbook declares no sheets";
/// Base a worksheet's own relationships resolve against: a comments or
/// threaded-comment target is relative to the sheet part's directory.
const SHEET_BASE: &str = "xl/worksheets/";
/// Prefix of the embedded-media parts in an Excel package.
const XLSX_MEDIA_PREFIX: &str = "xl/media/";
/// Charts in an Excel package: `xl/charts/chart<N>.xml`.
const XLSX_CHARTS_PREFIX: &str = "xl/charts/chart";

// ── Excel ───────────────────────────────────────────────────────

/// Extract sheet text and embedded images from an `.xlsx`/`.xlsm` package.
///
/// Every sheet the workbook names is walked in workbook order and rendered as
/// `Sheet "<name>":` plus one indented line per valued cell. A cell shows the
/// value its number format displays, each sheet's hidden rows and columns are
/// annotated, merged ranges are marked against the cells they cover, and a
/// hyperlink is marked on the cell it covers. A sheet's cell comments follow its
/// cells, the workbook's defined names follow the last sheet, and a sheet whose
/// part is missing costs only itself and a note.
#[must_use]
pub(crate) fn convert_xlsx(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return unreadable(".xlsx");
    };
    // The workbook part is read in one walk: its sheets, its date system and its
    // defined names come out of the same pass, which also says whether the part was
    // cut short. A part carrying no workbook element is read like one the package
    // does not carry; one cut short still yields the sheets and names it got to.
    let Some(workbook) = read_zip_entry(&mut archive, WORKBOOK_PART)
        .bytes()
        .and_then(|xml| workbook(&xml))
    else {
        return unreadable(".xlsx");
    };
    // The workbook relationships are read once: the sheet parts resolve through
    // them, the shared-string table, the styles part and the persons part are
    // named by kind, and the sheet parts then resolve their own comments through
    // their own rels. One that is there and cannot be read leaves every part at its
    // conventional name and every sheet matched to its entry by position, which
    // `rels_lost` says once rather than letting each part read as one the workbook
    // never declared.
    let workbook_rels = read_relationships(&mut archive, WORKBOOK_RELS_PART);
    let rels_lost = workbook_rels.is_none();
    let rels = workbook_rels.unwrap_or_default();
    // The shared-string table is named by kind like the styles part, so a
    // producer that lays it out elsewhere still has its strings read; only the
    // conventional name is a fallback.
    let shared_part = related_part(&rels, "/sharedStrings", "xl/")
        .unwrap_or_else(|| SHARED_STRINGS_PART.to_owned());
    let shared = read_zip_entry(&mut archive, &shared_part)
        .bytes()
        .map(|xml| shared_strings(&xml));
    let shared = shared.as_deref().unwrap_or_default();

    ensure_out_dir(out_dir);

    let mut notes = Vec::new();
    if rels_lost {
        notes.push(WORKBOOK_RELS_NOTE.to_owned());
    }
    if workbook.cut {
        notes.push(WORKBOOK_CUT_NOTE.to_owned());
    } else if workbook.sheets.is_empty() {
        notes.push(WORKBOOK_NO_SHEETS_NOTE.to_owned());
    }
    let persons = read_persons(&mut archive, &rels);
    let styles = read_styles(&mut archive, &rels, &mut notes);
    let rels = relationship_targets(&rels);
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
    let book = Book {
        shared,
        styles: &styles,
        date1904: workbook.date1904,
    };
    for (index, (name, id, state)) in workbook.sheets.iter().enumerate() {
        // A relationship the package does not declare — a workbook whose rels
        // cannot be read, or a sheet naming none — falls back to the conventional
        // part name for that position, so such a sheet still yields its cells
        // instead of nothing.
        let part = rels.get(id).map_or_else(
            || format!("{SHEET_BASE}sheet{}.xml", index + 1),
            |target| resolve_part("xl/", target),
        );
        // The sheet's own relationships name its comments and its hyperlink
        // targets; the list is kept for the parts named by kind, and the link
        // targets resolve through the shared id -> target map.
        let sheet_rels = sheet_relationships(&mut archive, &part, name, &mut notes);
        let link_targets = relationship_targets(&sheet_rels);
        // A sheet part the package does not carry is as much of a loss as one
        // that will not parse, so both are said the same way.
        let rows = read_zip_entry(&mut archive, &part)
            .bytes()
            .and_then(|xml| sheet_rows(&xml, book, &link_targets));
        match rows {
            Some(read) => {
                lost += read.lost_strings;
                sheet_losses(name, &read, &mut notes);
                let mut lines = read.lines;
                let comments = cell_comments(&mut archive, &sheet_rels, name, &persons, &mut notes);
                lines.extend(comment_lines(comments, name, &mut notes));
                blocks.push(text_block(
                    &sheet_header(name, state.as_deref()),
                    "(no values)",
                    &lines,
                ));
            }
            None => notes.push(format!("sheet \"{name}\" could not be read")),
        }
    }
    // Defined names are workbook-level, so they follow every sheet rather than
    // being repeated into the sheet whose cells they point at.
    if !workbook.defined_names.is_empty() {
        blocks.push(text_lines(
            "Defined names:",
            &defined_name_lines(&workbook.defined_names, &workbook.sheets),
        ));
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

/// The notes one sheet's reading adds: the hyperlinks whose target it could not
/// resolve, the merged ranges it could not mark, the range marks it could not
/// afford, and the column definitions — the ones it read too late for them to reach
/// a cell, the ones whose column it cannot place, and the ones whose range names an
/// end it cannot read.
fn sheet_losses(name: &str, read: &SheetRead, notes: &mut Vec<String>) {
    if read.lost_links > 0 {
        notes.push(format!(
            "sheet \"{name}\": {} link(s) left out: the target or the cell range it names is not one this reader can resolve",
            read.lost_links
        ));
    }
    if read.lost_merges > 0 {
        notes.push(format!(
            "sheet \"{name}\": {} merged range(s) left out: the cell range they name is not one this reader can mark",
            read.lost_merges
        ));
    }
    if read.late_columns > 0 {
        notes.push(format!(
            "sheet \"{name}\": {} column definition(s) left out: the sheet writes them after \
             its data, where they reach no cell of it",
            read.late_columns
        ));
    }
    if read.lost_columns > 0 {
        notes.push(format!(
            "sheet \"{name}\": {} column definition(s) left out: the columns they hide or \
             style are not ones this reader can place",
            read.lost_columns
        ));
    }
    if read.partial_columns > 0 {
        notes.push(format!(
            "sheet \"{name}\": {} column definition(s) cover a range whose end this \
             reader cannot place, so only the column each one starts at is hidden or \
             styled",
            read.partial_columns
        ));
    }
    if read.dropped_marks {
        notes.push(format!(
            "sheet \"{name}\": later merged ranges and links left out: the sheet declares more \
             of them than this reader marks"
        ));
    }
}

/// The workbook part's own content, read in the part's one walk: its sheets in
/// document order as `(name, relationship id, state)`, whether it uses the 1904
/// date system, and its defined names. The walk that reads them also says whether
/// the part was cut short, so the sheet list and the loss of what followed it are
/// read together rather than by a pass of their own each.
struct Workbook {
    /// The workbook's `<sheet>` entries, each name one line by construction
    /// ([`one_line`]): a name reaches the sheet header, the notes naming the sheet
    /// and the defined-names block, each of which is a line of its own, so a break
    /// in it must not become a line there.
    sheets: Vec<(String, String, Option<String>)>,
    /// Whether the workbook's `workbookPr` puts it in the 1904 date system.
    date1904: bool,
    /// The workbook's defined names in document order, each with its text.
    defined_names: Vec<DefinedName>,
    /// Whether the part was cut short — at an element boundary or inside a tag — so
    /// the sheets and names it would have listed after that point were not read.
    cut: bool,
}

/// The workbook part, or `None` when it carries no workbook element — an entry the
/// package left empty, an error page in its place, or one whose bytes are no
/// document are all read like a part the package does not carry, rather than as a
/// workbook declaring no sheet. Every `<sheet>` element is kept, the ones this
/// reader cannot fully place included: a sheet whose relationship the workbook does
/// not name still has a part the caller reads by its position, and a sheet that
/// vanished here would both lose its cells and shift the position every later
/// sheet's scoped defined name is matched by.
fn workbook(xml: &[u8]) -> Option<Workbook> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut read_workbook = false;
    let mut sheets = Vec::new();
    let mut date1904: Option<String> = None;
    let mut names = Vec::new();
    let mut current: Option<DefinedName> = None;
    let mut text = String::new();
    // The walk ends where the part does, so the sheets and names read before a cut are
    // delivered rather than the whole workbook being declined; whether the part counts
    // as cut — and so carries [`WORKBOOK_CUT_NOTE`] — is what its own open elements
    // say ([`Part::cut`]).
    while let Ok(event) = reader.read_event_into(&mut buffer) {
        part.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"workbook" => read_workbook = true,
                b"sheet" => sheets.push(sheet_entry(&event)),
                // The first `workbookPr` that declares a date system is the one
                // the workbook's dates are written in.
                b"workbookPr" if date1904.is_none() => {
                    date1904 = attr(&event, b"date1904");
                }
                b"definedName" => {
                    text.clear();
                    current = defined_name(&event);
                }
                _ => {}
            },
            Event::Empty(event) => match event.local_name().as_ref() {
                b"workbook" => read_workbook = true,
                b"sheet" => sheets.push(sheet_entry(&event)),
                b"workbookPr" if date1904.is_none() => {
                    date1904 = attr(&event, b"date1904");
                }
                b"definedName" => {
                    if let Some(name) = defined_name(&event) {
                        names.push(name);
                    }
                }
                _ => {}
            },
            // A defined name's text is the character data of its own element (a
            // plain string or a CDATA section), entity references resolved.
            Event::Text(event) if current.is_some() => {
                if let Ok(chunk) = event.xml10_content() {
                    text.push_str(&chunk);
                }
            }
            Event::CData(event) if current.is_some() => {
                if let Ok(chunk) = event.xml10_content() {
                    text.push_str(&chunk);
                }
            }
            Event::GeneralRef(event) if current.is_some() => {
                append_entity(&mut text, &event);
            }
            // A name whose element never closes is dropped rather than costing the
            // ones before it; the part's cut is said by the caller.
            Event::End(event) => {
                if event.local_name().as_ref() == b"definedName"
                    && let Some(mut name) = current.take()
                {
                    name.text = one_line(text.trim().to_owned());
                    names.push(name);
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    // A part carrying no workbook element — an entry the package left empty, an
    // error page in its place, or one whose bytes are no document — is read like a
    // part the package does not carry rather than as a workbook declaring no sheet.
    // A part cut short still yields the sheets and names it got to, with `cut` saying
    // the list after that point was not read.
    if !read_workbook {
        return None;
    }
    Some(Workbook {
        sheets,
        date1904: date1904.as_deref().is_some_and(is_true),
        defined_names: names,
        cut: part.cut(),
    })
}

/// One `<sheet .../>` entry as `(name, relationship id, state)`, the name as the
/// one line it must be ([`one_line`]).
fn sheet_entry(event: &BytesStart<'_>) -> (String, String, Option<String>) {
    (
        one_line(attr(event, b"name").unwrap_or_default()),
        rel_id(event).unwrap_or_default(),
        attr(event, b"state"),
    )
}

/// Whether an XML boolean attribute is true: `1` or `true`, as Excel writes it.
fn is_true(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

/// A sheet's block header, marked when the workbook hides the sheet. The state a
/// workbook writes is one of two words, and a producer's own casing is no reason to
/// pass a hidden sheet off as an ordinary one.
fn sheet_header(name: &str, state: Option<&str>) -> String {
    match state.map(str::to_ascii_lowercase).as_deref() {
        Some("veryhidden") => format!("Sheet \"{name}\" (very hidden):"),
        Some("hidden") => format!("Sheet \"{name}\" (hidden):"),
        _ => format!("Sheet \"{name}\":"),
    }
}

// ── Defined names ───────────────────────────────────────────────

/// One workbook `<definedName>`: its name, the sheet its `localSheetId` names
/// (a zero-based index into the workbook's sheets), and its text.
struct DefinedName {
    name: String,
    local_sheet: Option<u32>,
    text: String,
}

/// One `<definedName .../>` element as a name, or `None` when it carries no
/// `name` or is a service name Excel writes for itself (`_xlnm.…`: print areas,
/// print titles, `_FilterDatabase`, …), which the workbook's own UI never shows.
/// The name is one line by construction ([`one_line`]): it is printed as part of
/// the defined-names block, a line of its own.
fn defined_name(event: &BytesStart<'_>) -> Option<DefinedName> {
    let name = attr(event, b"name")?;
    if name.to_ascii_lowercase().starts_with("_xlnm.") {
        return None;
    }
    Some(DefinedName {
        local_sheet: attr(event, b"localSheetId").and_then(|id| id.trim().parse().ok()),
        name: one_line(name),
        text: String::new(),
    })
}

/// One line per defined name: a sheet-local one names the sheet its
/// `localSheetId` indexes, a workbook-level one (or one whose index resolves to
/// no sheet) is bare. A name whose element carries no text names nothing, so it is
/// left out rather than printed as a bare name and a colon.
fn defined_name_lines(
    names: &[DefinedName],
    sheets: &[(String, String, Option<String>)],
) -> Vec<String> {
    names
        .iter()
        .filter(|name| !name.text.is_empty())
        .map(|name| {
            match name
                .local_sheet
                .and_then(|index| sheets.get(index as usize))
            {
                Some((sheet, _, _)) => format!("{} (sheet \"{sheet}\"): {}", name.name, name.text),
                None => format!("{}: {}", name.name, name.text),
            }
        })
        .collect()
}

// ── Shared strings ──────────────────────────────────────────────

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

// ── Formats ─────────────────────────────────────────────────────

/// The workbook's cell formats.
#[derive(Default)]
struct Styles {
    /// Declared `numFmt` code by id.
    num_fmts: HashMap<u32, String>,
    /// The `numFmtId` of each `cellXfs` entry, in order: a cell's `s` indexes
    /// this for the format it shows, and `None` is an entry whose own declaration
    /// this reader could not read — the cell then shows its stored value, marked.
    cell_xfs: Vec<Option<u32>>,
    /// Whether the workbook's formats were there to read and no table came back: the
    /// styles part the workbook declares and the package does not carry, one that is
    /// there and will not parse, or one that declares a format but no cell format to
    /// name any of them — a `numFmt` a malformed entry included, a base-style list,
    /// or a `cellXfs` holding no entry. Every cell then shows its stored text marked
    /// as stored, whatever format it names, rather than being read as one of a
    /// workbook that declares no format at all.
    lost: bool,
}

/// Parse `xl/styles.xml`. `None` when the part cannot be read as XML, is cut short,
/// or carries anything but a stylesheet element — an error page or a text body left
/// in the part's place parses as XML just as happily, and a workbook whose formats
/// were not read is not one to pass its cells off as unformatted. An entry the
/// package left empty declares no format for any cell, which is the value a
/// stylesheet declaring no format also reads as.
fn styles(xml: &[u8]) -> Option<Styles> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut num_fmts = HashMap::new();
    // The base styles a cell format may inherit its number format from, and the
    // cell formats as the three things each declares: its own `numFmtId`, the base
    // style it names and whether it applies its declaration.
    let mut base_xfs: Vec<Option<u32>> = Vec::new();
    let mut cell_xfs: Vec<(Option<u32>, Base, bool)> = Vec::new();
    // Whether the stylesheet declares any format at all — a `numFmt` element (a
    // malformed one included: it is still a format the workbook claims to hold), a
    // cell-format list, or a base-style list, whether or not it holds an entry.
    let mut declares_formats = false;
    let (mut in_cell_xfs, mut in_base_xfs) = (false, false);
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        part.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"styleSheet" => part.kind_seen(),
                b"cellXfs" => {
                    in_cell_xfs = true;
                    declares_formats = true;
                }
                b"cellStyleXfs" => {
                    in_base_xfs = true;
                    declares_formats = true;
                }
                b"numFmt" => {
                    declares_formats = true;
                    add_num_fmt(&event, &mut num_fmts);
                }
                b"xf" if in_cell_xfs || in_base_xfs => {
                    collect_xf(&event, in_cell_xfs, &mut cell_xfs, &mut base_xfs);
                }
                _ => {}
            },
            Event::Empty(event) => match event.local_name().as_ref() {
                b"styleSheet" => part.kind_seen(),
                b"cellXfs" | b"cellStyleXfs" => declares_formats = true,
                b"numFmt" => {
                    declares_formats = true;
                    add_num_fmt(&event, &mut num_fmts);
                }
                b"xf" if in_cell_xfs || in_base_xfs => {
                    collect_xf(&event, in_cell_xfs, &mut cell_xfs, &mut base_xfs);
                }
                _ => {}
            },
            Event::End(event) => match event.local_name().as_ref() {
                b"cellXfs" => in_cell_xfs = false,
                b"cellStyleXfs" => in_base_xfs = false,
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    // A styles part cut short — or one carrying anything but a stylesheet element —
    // reads as a stylesheet declaring no cell format, which would leave every cell of
    // the workbook at the schema's default with nothing said.
    if part.unreadable() {
        return None;
    }
    // A stylesheet that declares a format but no cell format to name it from is one
    // this reader cannot derive any cell's format from: the formats it names are lost
    // rather than read as a workbook that declares no format for any cell.
    if cell_xfs.is_empty() && declares_formats {
        return None;
    }
    // A cell format that declares no format of its own and does not apply it shows
    // the one the base style it names holds — how a workbook that leaves its number
    // format to a named style writes it. Resolving here keeps every cell lookup a
    // plain index into `cell_xfs`: `None` is an entry whose format cannot be derived,
    // so the cell that shows it carries its stored value under the mark.
    let cell_xfs = cell_xfs
        .into_iter()
        .map(|(own, base, applies)| match (own, base) {
            // The format the entry would inherit is one this reader cannot read (the
            // base style's own declaration is unreadable, or its `xfId` is not one to
            // place), so the entry is read like one whose own declaration is
            // unreadable rather than as a format declaring nothing.
            (Some(0), Base::Unread) if !applies => None,
            (Some(0), Base::Index(base)) if !applies => base_xfs
                .get(base as usize)
                .copied()
                // A base list that does not hold the entry the format names: entry 0
                // stands for the schema's default base style, which is `General`, while
                // a later entry the list does not hold leaves the format underivable.
                .unwrap_or_else(|| (base == 0).then_some(0)),
            // A format that applies its own declaration keeps it, even when that
            // declaration is `General` (`numFmtId` 0): the format claims the format it
            // shows, so it is not handed the base style's.
            (own, _) => own,
        })
        .collect();
    Some(Styles {
        num_fmts,
        cell_xfs,
        lost: false,
    })
}

/// Record an `xf`: a cell format (`numFmtId`, the base style it inherits from and
/// whether it applies the format it declares — only the three together decide what
/// a cell shows) or, before `cellXfs`, a base style's own `numFmtId`.
fn collect_xf(
    event: &BytesStart<'_>,
    in_cell_xfs: bool,
    cell_xfs: &mut Vec<(Option<u32>, Base, bool)>,
    base_xfs: &mut Vec<Option<u32>>,
) {
    if in_cell_xfs {
        cell_xfs.push((
            xf_num_fmt_id(event),
            xf_base_id(event),
            attr(event, b"applyNumberFormat")
                .as_deref()
                .is_some_and(is_true),
        ));
    } else {
        base_xfs.push(xf_num_fmt_id(event));
    }
}

/// Record a `<numFmt numFmtId formatCode/>`; both attributes are required, so a
/// malformed entry is skipped.
fn add_num_fmt(event: &BytesStart<'_>, formats: &mut HashMap<u32, String>) {
    let (Some(id), Some(code)) = (attr(event, b"numFmtId"), attr(event, b"formatCode")) else {
        return;
    };
    if let Ok(id) = id.trim().parse() {
        formats.insert(id, code);
    }
}

/// The `numFmtId` an `xf` declares: `Some(0)`, the default format, when it
/// declares none, and `None` when the attribute is there and does not parse — an
/// entry whose own format this reader cannot read.
fn xf_num_fmt_id(event: &BytesStart<'_>) -> Option<u32> {
    match attr(event, b"numFmtId") {
        None => Some(0),
        Some(id) => id.trim().parse().ok(),
    }
}

/// The base style an `xf` names by its `xfId`, whose number format the entry shows
/// where it declares none of its own and does not apply it.
#[derive(Clone, Copy)]
enum Base {
    /// The entry names no base style.
    None,
    /// The base style the entry names, by the index of its `cellStyleXfs` entry.
    Index(u32),
    /// The entry names a base style this reader cannot read.
    Unread,
}

/// The base style an `xf` names.
fn xf_base_id(event: &BytesStart<'_>) -> Base {
    match attr(event, b"xfId") {
        None => Base::None,
        Some(id) => id.trim().parse().map_or(Base::Unread, Base::Index),
    }
}

/// The workbook's styles part: the first relationship whose kind names a styles
/// part, else the conventional name.
fn styles_part(rels: &[Relationship]) -> String {
    related_part(rels, "/styles", "xl/").unwrap_or_else(|| STYLES_PART.to_owned())
}

/// The workbook's cell formats, read from the part its `/styles` relationship names
/// (the conventional one as a fallback): the schema default when the workbook
/// carries no such part, plus a note when a part that was there to read yielded no
/// table — one the workbook declares through its relationships and the package does
/// not carry, or one that will not parse. A workbook that declares no styles part,
/// and carries none, declares no format for any cell: nothing is lost, nothing is
/// said, and the cells show the schema's default — bar the cell that names a format
/// the missing table would have held, which shows its stored number under the mark.
fn read_styles<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    rels: &[Relationship],
    notes: &mut Vec<String>,
) -> Styles {
    let declared = relationship_target(rels, "/styles").is_some();
    let read = match read_zip_entry(archive, &styles_part(rels)) {
        ZipEntry::Bytes(xml) => styles(&xml),
        ZipEntry::Missing if !declared => return Styles::default(),
        // A part the workbook says it carries and the package does not is as much of
        // a loss as one that will not parse.
        ZipEntry::Missing | ZipEntry::Unreadable => None,
    };
    let Some(styles) = read else {
        notes.push(STYLES_NOTE.to_owned());
        return Styles {
            lost: true,
            ..Styles::default()
        };
    };
    styles
}

// ── Sheet relationships, comments and links ─────────────────────

/// The relationships part of a worksheet part: `xl/worksheets/sheet1.xml` ->
/// `xl/worksheets/_rels/sheet1.xml.rels`.
#[must_use]
fn sheet_rels_part(part: &str) -> String {
    let (dir, file) = part.rsplit_once('/').unwrap_or(("", part));
    if dir.is_empty() {
        return format!("_rels/{file}.rels");
    }
    format!("{dir}/_rels/{file}.rels")
}

/// The relationships the part at `name` declares, or `None` when the part is there
/// and cannot be read. A part the package does not carry, and an entry it left empty,
/// both declare no relationship at all — nothing is lost; a part that is no
/// relationship list, or one the package cut short, names its remaining parts
/// nowhere, so its list is not the document's own and is read like one that could
/// not be read at all.
fn read_relationships<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Option<Vec<Relationship>> {
    match read_zip_entry(archive, name) {
        ZipEntry::Bytes(xml) if blank(&xml) => Some(Vec::new()),
        ZipEntry::Bytes(xml) => match relationships_and_whole(&xml) {
            (Some(rels), true) => Some(rels),
            _ => None,
        },
        ZipEntry::Missing => Some(Vec::new()),
        ZipEntry::Unreadable => None,
    }
}

/// A sheet's own relationships, and the note a part that is there and cannot be
/// read is said in rather than passed off as a sheet with none.
fn sheet_relationships<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    part: &str,
    name: &str,
    notes: &mut Vec<String>,
) -> Vec<Relationship> {
    let Some(rels) = read_relationships(archive, &sheet_rels_part(part)) else {
        notes.push(format!(
            "sheet \"{name}\": its relationships could not be read"
        ));
        return Vec::new();
    };
    rels
}

/// The first relationship of `rels` whose kind names `suffix`, as the raw
/// `Target` it points at.
fn relationship_target<'a>(rels: &'a [Relationship], suffix: &str) -> Option<&'a str> {
    rels.iter()
        .find(|rel| rel.kind.ends_with(suffix))
        .map(|rel| rel.target.as_str())
}

/// The package part of `base` that the first relationship of `rels` whose kind
/// names `suffix` points at, or `None` when no relationship names one. A
/// relative target is resolved against `base`, an absolute one is already a
/// package path.
fn related_part(rels: &[Relationship], suffix: &str, base: &str) -> Option<String> {
    relationship_target(rels, suffix).map(|target| resolve_part(base, target))
}

/// The workbook's person names, read from the part the workbook's `/person`
/// relationship names: a `personId` a thread names resolves here, and one it does
/// not — the part absent, unreadable, holding no `person`, or the workbook's
/// relationships unreadable in the first place — leaves that thread's author
/// unresolved, which its sheet says.
fn read_persons<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    rels: &[Relationship],
) -> HashMap<String, String> {
    let Some(part) = related_part(rels, "/person", "xl/") else {
        return HashMap::new();
    };
    read_zip_entry(archive, &part)
        .bytes()
        .and_then(|xml| scan_elements(&xml, b"person", person_entry))
        .unwrap_or_default()
        .into_iter()
        .collect()
}

/// One `<person id displayName/>` entry, or `None` when either attribute is
/// absent. The display name is one line by construction ([`one_line`]): it stands
/// inside a comment's line, so a break in it must not become a line there.
fn person_entry(event: &BytesStart<'_>) -> Option<(String, String)> {
    let id = attr(event, b"id")?;
    let name = one_line(attr(event, b"displayName")?);
    Some((id, name))
}

/// One cell comment: the cell it is attached to, its author and its text as the
/// part wrote it. The text keeps its lines — a threaded discussion's legacy stub
/// leads each of its entries with a line of its own, which is what tells it from a
/// comment written by a person — and the line the comment is shown on collapses
/// them.
struct Comment {
    reference: String,
    author: Author,
    text: String,
}

/// A comment's author: none named by the workbook, one this reader resolved, or
/// one the workbook names and this reader could not resolve — a `personId` the
/// person names hold no entry for, an `authorId` past its authors list. An author
/// that is in the file but not in this answer is said per sheet rather than passed
/// off as a comment by nobody.
#[derive(PartialEq)]
enum Author {
    None,
    Named(String),
    Lost,
}

impl Comment {
    /// The comment, or `None` when its text is blank once whitespace is collapsed
    /// — a comment that shows nothing is not worth a line of its own.
    fn new(reference: String, author: Author, text: &str) -> Option<Self> {
        if flatten(text).is_empty() {
            return None;
        }
        Some(Self {
            reference,
            author: match author {
                Author::Named(name) if name.is_empty() => Author::None,
                author => author,
            },
            text: text.to_owned(),
        })
    }

    /// The comment's line: its cell, its author when one is known, its text in one
    /// line whatever lines the part wrote it in.
    fn line(&self) -> String {
        match &self.author {
            Author::Named(author) => format!(
                "{} comment ({author}): {}",
                self.reference,
                flatten(&self.text)
            ),
            Author::None | Author::Lost => {
                format!("{} comment: {}", self.reference, flatten(&self.text))
            }
        }
    }
}

/// A text as one line: every run of whitespace becomes one space.
fn flatten(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A cell's text as the one line its cell gets: a line break a value carries would
/// otherwise stand as a break in the answer, breaking the one-line-per-cell shape
/// this reading promises and letting a document's text stand where this reader's
/// lines do. Every break a text stack may take is escaped as `\n`, which a value
/// carrying a backslash and an `n` of its own reads the same as — one line per cell
/// leaves no other spelling.
fn one_line(text: String) -> String {
    if !text.chars().any(is_line_break) {
        return text;
    }
    let mut out = String::with_capacity(text.len() + 4);
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\r' => {
                characters.next_if_eq(&'\n');
                out.push_str("\\n");
            }
            _ if is_line_break(character) => out.push_str("\\n"),
            _ => out.push(character),
        }
    }
    out
}

/// The comments in a legacy `xl/comments<N>.xml` part: its `commentList` entries,
/// each with the name its `authorId` indexes into the part's `authors` list.
/// The text is the nested character data of the entry's `<text>` element, so a
/// plain string, a `<t>` run and a run inside a rich `<r>` all read the same.
/// `None` when the part cannot be read as XML.
fn legacy_comments(xml: &[u8]) -> Option<Vec<Comment>> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut authors: Vec<String> = Vec::new();
    let mut author_text = String::new();
    let mut in_author = false;
    let mut current: Option<(String, Option<String>)> = None;
    let mut text = String::new();
    let mut in_text = false;
    let mut comments = Vec::new();
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        part.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"comments" => part.kind_seen(),
                b"author" => {
                    in_author = true;
                    author_text.clear();
                }
                b"comment" => {
                    current = Some((
                        attr(&event, b"ref").unwrap_or_default(),
                        attr(&event, b"authorId"),
                    ));
                    text.clear();
                }
                b"text" if current.is_some() => in_text = true,
                _ => {}
            },
            Event::Empty(event) if event.local_name().as_ref() == b"comments" => {
                part.kind_seen();
            }
            Event::Empty(event) if event.local_name().as_ref() == b"author" => {
                // An empty author still occupies its index, so the entries after
                // it keep their `authorId`.
                authors.push(String::new());
            }
            Event::Text(event) => {
                if in_author {
                    if let Ok(chunk) = event.xml10_content() {
                        author_text.push_str(&chunk);
                    }
                } else if in_text && let Ok(chunk) = event.xml10_content() {
                    text.push_str(&chunk);
                }
            }
            Event::GeneralRef(event) => {
                if in_author {
                    append_entity(&mut author_text, &event);
                } else if in_text {
                    append_entity(&mut text, &event);
                }
            }
            Event::End(event) => match event.local_name().as_ref() {
                b"author" => {
                    in_author = false;
                    // The author is one line by construction ([`one_line`]): the name
                    // stands inside the comment's line, so a break in it must not
                    // become a line there.
                    authors.push(one_line(std::mem::take(&mut author_text)));
                }
                b"text" => in_text = false,
                b"comment" => {
                    if let Some((reference, author_id)) = current.take() {
                        // An `authorId` the authors list does not hold names an
                        // author that is in the file but not in this answer.
                        let author = match author_id {
                            Some(id) => id
                                .trim()
                                .parse::<usize>()
                                .ok()
                                .and_then(|index| authors.get(index).cloned())
                                .map_or(Author::Lost, Author::Named),
                            None => Author::None,
                        };
                        if let Some(comment) = Comment::new(reference, author, &text) {
                            comments.push(comment);
                        }
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    // A comments part read as the comments it got to would pass the rest off as
    // comments the sheet never had, so a cut one is declined whole — as is a part
    // that is no comments document — while an empty entry declares none.
    if part.unreadable() {
        return None;
    }
    Some(comments)
}

/// The comments in a modern `xl/threadedComments/threadedComment<N>.xml` part,
/// each with the name its `personId` maps to — [`Author::Lost`] when it names a
/// person the workbook's person names do not hold. `None` when the part cannot be
/// read as XML.
fn threaded_comments(xml: &[u8], persons: &HashMap<String, String>) -> Option<Vec<Comment>> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut current: Option<(String, Author)> = None;
    let mut text = String::new();
    let mut in_text = false;
    let mut comments = Vec::new();
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        part.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"ThreadedComments" => part.kind_seen(),
                b"threadedComment" => {
                    let author = match attr(&event, b"personId") {
                        Some(id) => persons
                            .get(&id)
                            .cloned()
                            .map_or(Author::Lost, Author::Named),
                        None => Author::None,
                    };
                    current = Some((attr(&event, b"ref").unwrap_or_default(), author));
                    text.clear();
                }
                b"text" if current.is_some() => in_text = true,
                _ => {}
            },
            Event::Empty(event) if event.local_name().as_ref() == b"ThreadedComments" => {
                part.kind_seen();
            }
            Event::Text(event) if in_text => {
                if let Ok(chunk) = event.xml10_content() {
                    text.push_str(&chunk);
                }
            }
            Event::GeneralRef(event) if in_text => append_entity(&mut text, &event),
            Event::End(event) => match event.local_name().as_ref() {
                b"text" => in_text = false,
                b"threadedComment" => {
                    if let Some((reference, author)) = current.take()
                        && let Some(comment) = Comment::new(reference, author, &text)
                    {
                        comments.push(comment);
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    // As for the legacy part: a discussion cut short would read as one holding the
    // entries it got to, so it is declined whole rather than delivered by halves.
    if part.unreadable() {
        return None;
    }
    Some(comments)
}

/// The sheet's cell comments: its modern threaded comments and its legacy ones,
/// with a thread replacing the legacy stub of its own discussion. Each part the
/// sheet names that is there but cannot be read is said in a note of its own,
/// since either can be lost without the other.
fn cell_comments<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    rels: &[Relationship],
    name: &str,
    persons: &HashMap<String, String>,
    notes: &mut Vec<String>,
) -> Vec<Comment> {
    // Every comment of a cell is kept, in the order the parts read: a threaded
    // discussion is a thread, so its entries are all shown.
    let mut by_cell: BTreeMap<String, Vec<Comment>> = BTreeMap::new();
    if let Some(target) = relationship_target(rels, "/comments") {
        let part = resolve_part(SHEET_BASE, target);
        match read_zip_entry(archive, &part)
            .bytes()
            .and_then(|xml| legacy_comments(&xml))
        {
            Some(part_comments) => {
                for comment in part_comments {
                    by_cell
                        .entry(comment.reference.clone())
                        .or_default()
                        .push(comment);
                }
            }
            None => notes.push(format!("sheet \"{name}\": its comments could not be read")),
        }
    }
    if let Some(target) = relationship_target(rels, "/threadedComment") {
        let part = resolve_part(SHEET_BASE, target);
        match read_zip_entry(archive, &part)
            .bytes()
            .and_then(|xml| threaded_comments(&xml, persons))
        {
            Some(part_comments) => {
                let mut threaded: BTreeMap<String, Vec<Comment>> = BTreeMap::new();
                for comment in part_comments {
                    threaded
                        .entry(comment.reference.clone())
                        .or_default()
                        .push(comment);
                }
                for (reference, thread) in threaded {
                    let cell = by_cell.entry(reference).or_default();
                    // A real workbook keeps the discussion's own text in the legacy
                    // part as a stub; a legacy comment the thread does not carry is a
                    // comment of its own, so it keeps its line.
                    cell.retain(|comment| !is_the_stub_of(comment, &thread));
                    cell.extend(thread);
                }
            }
            None => notes.push(format!(
                "sheet \"{name}\": its threaded comments could not be read"
            )),
        }
    }
    by_cell.into_values().flatten().collect()
}

/// Whether a legacy comment on a cell is the stub of the threaded discussion on
/// it — the same discussion written twice, and so shown once.
///
/// Excel keeps a copy of a discussion in the legacy part for readers without
/// threaded comments: a `[Threaded comment]` preamble, a note to the reader, and
/// then each entry's text behind a `Comment:` (or `Reply:`) lead of its own, in
/// thread order. That copy is what this reader recognizes — the texts the leads
/// carry, in their own places, are the thread's own texts. A legacy comment
/// carrying no lead at all, or carrying texts that are not the thread's, is a
/// comment of its own and keeps its line, since dropping it would lose a person's
/// words on a guess.
fn is_the_stub_of(comment: &Comment, thread: &[Comment]) -> bool {
    let entries = stub_entries(&comment.text);
    !entries.is_empty()
        && entries.len() <= thread.len()
        && entries
            .iter()
            .zip(thread)
            .all(|(entry, comment)| flatten(entry) == flatten(&comment.text))
}

/// The entries a legacy comment's text carries as a threaded stub: the text behind
/// each `Comment:`/`Reply:` lead, up to the next lead, with everything before the
/// first lead — the `[Threaded comment]` preamble and the note to the reader Excel
/// writes ahead of them — dropped. Empty when the text carries no lead at all,
/// which is a comment of its own rather than a copy of a discussion.
fn stub_entries(text: &str) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        match stub_lead(line) {
            Some(body) => {
                entries.extend(current.take());
                current = Some(body.to_owned());
            }
            None => {
                if let Some(entry) = current.as_mut() {
                    if !entry.is_empty() {
                        entry.push(' ');
                    }
                    entry.push_str(line);
                }
            }
        }
    }
    entries.extend(current);
    entries
}

/// The text behind a `Comment:` or `Reply:` lead at the start of `line`, or `None`
/// when the line carries no lead.
fn stub_lead(line: &str) -> Option<&str> {
    let (lead, rest) = line.split_once(':')?;
    let lead = lead.trim();
    (lead.eq_ignore_ascii_case("comment") || lead.eq_ignore_ascii_case("reply"))
        .then(|| rest.trim())
}

/// The sheet's comment lines, one per comment, sorted by the (row, column) of the
/// cell they are attached to. A comment whose cell reference this reader cannot
/// parse names no cell a line could be attached to, so it is left out and counted
/// rather than printed under an address that is not one.
fn comment_lines(comments: Vec<Comment>, name: &str, notes: &mut Vec<String>) -> Vec<String> {
    let total = comments.len();
    let mut placed: Vec<((u32, u32), Comment)> = comments
        .into_iter()
        .filter_map(|comment| {
            cell_position(&comment.reference).map(|(column, row)| ((row, column), comment))
        })
        .collect();
    if placed.len() < total {
        notes.push(format!(
            "sheet \"{name}\": {} comment(s) left out: the cell they name is not one this reader can resolve",
            total - placed.len()
        ));
    }
    let lost_authors = placed
        .iter()
        .filter(|(_, comment)| comment.author == Author::Lost)
        .count();
    if lost_authors > 0 {
        notes.push(format!(
            "sheet \"{name}\": {lost_authors} comment author(s) could not be resolved"
        ));
    }
    placed.sort_by_key(|(position, _)| *position);
    placed
        .into_iter()
        .map(|(_, comment)| comment.line())
        .collect()
}

// ── Worksheet pass ──────────────────────────────────────────────

/// What a cell needs to render: the workbook's shared strings, its formats and
/// its date system.
#[derive(Clone, Copy)]
struct Book<'a> {
    shared: &'a [String],
    styles: &'a Styles,
    date1904: bool,
}

/// What one worksheet pass yields: its rendered lines, how many cells held a
/// `t="s"` string the table does not provide, how many hyperlinks were lost — a
/// link is lost when its `r:id` is not declared, its `ref` is missing, or its
/// `ref` is not a range this reader can parse — and how many merges were lost —
/// a merge is lost when its `ref` is missing or is not a range this reader can
/// mark; all are reported by the caller as notes.
struct SheetRead {
    lines: Vec<String>,
    lost_strings: usize,
    lost_links: usize,
    lost_merges: usize,
    /// The `<col>` definitions the sheet wrote after its data, where they reach no
    /// cell of it — reported by the caller.
    late_columns: usize,
    /// The `<col>` definitions whose column this reader cannot place, so nothing of
    /// them is applied — reported by the caller.
    lost_columns: usize,
    /// The `<col>` definitions whose declared end this reader cannot read, so only
    /// the column each one starts at is hidden or styled — reported by the caller.
    partial_columns: usize,
    /// The sheet declared more merged ranges and links than this reader could
    /// mark, so its later marks are missing — reported by the caller.
    dropped_marks: bool,
}

/// The lines of one worksheet: the hidden-geometry annotations first, then one
/// line per valued cell, then a line per hyperlink its cells do not show.
///
/// `None` on any XML error — a sheet part that cannot be read as XML is
/// reported like a missing one.
#[expect(clippy::too_many_lines)] // one arm per worksheet element, in a single streaming pass
fn sheet_rows(
    xml: &[u8],
    book: Book<'_>,
    link_targets: &HashMap<String, String>,
) -> Option<SheetRead> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut cells: Vec<CellLine> = Vec::new();
    let mut lost = 0usize;
    let mut row_number = 0u32;
    // Column of the next cell that carries no `r`: real files omit `r` on
    // cells that follow one another, and the addresses still count distance.
    let mut next_column = 0u32;
    let mut cell: Option<Cell> = None;
    let mut shared_formulas: HashMap<String, String> = HashMap::new();
    let (mut in_value, mut in_text, mut in_formula) = (false, false, false);
    let (mut in_cols, mut in_merges, mut in_hyperlinks) = (false, false, false);
    let mut hidden_rows = BTreeSet::new();
    let mut column_ranges: Vec<ColumnRange> = Vec::new();
    let mut columns: Option<Columns> = None;
    // How many `<col>` definitions the sheet had written when its first cell
    // settled them; the ones written after that reach no cell of it.
    let mut settled_columns: Option<usize> = None;
    let mut row_style: Option<Style> = None;
    let mut merges = Vec::new();
    let mut links = Vec::new();
    let (mut lost_links, mut lost_merges, mut lost_columns, mut partial_columns) =
        (0usize, 0usize, 0usize, 0usize);
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        part.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"worksheet" | b"chartsheet" | b"dialogsheet" | b"macrosheet" => part.kind_seen(),
                b"row" => {
                    row_number = next_row_number(&event, row_number);
                    row_style = row_default_style(&event);
                    next_column = 0;
                    if attr(&event, b"hidden").as_deref().is_some_and(is_true) {
                        hidden_rows.insert(row_number);
                    }
                }
                b"cols" => in_cols = true,
                b"col" if in_cols => {
                    collect_columns(
                        &event,
                        &mut column_ranges,
                        &mut lost_columns,
                        &mut partial_columns,
                    );
                }
                b"mergeCells" => in_merges = true,
                b"mergeCell" if in_merges => {
                    collect_merge(&event, &mut merges, &mut lost_merges);
                }
                // `<hyperlinks>` follows `<sheetData>`, so it is collected in
                // this same pass and applied once the cells are rendered.
                b"hyperlinks" => in_hyperlinks = true,
                b"hyperlink" if in_hyperlinks => {
                    collect_link(&event, link_targets, &mut links, &mut lost_links);
                }
                // `<cols>` precedes `<sheetData>` in the schema, so the ranges
                // read so far are settled the first time a cell asks for them.
                b"c" => {
                    let columns = columns.get_or_insert_with(|| {
                        settled_columns = Some(column_ranges.len());
                        resolve_columns(&column_ranges)
                    });
                    cell = Some(Cell::start(
                        &event,
                        row_number,
                        row_style,
                        columns,
                        &mut next_column,
                    ));
                }
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
            Event::Empty(event) => match event.local_name().as_ref() {
                b"worksheet" | b"chartsheet" | b"dialogsheet" | b"macrosheet" => part.kind_seen(),
                // A row with no cells still hides, so it counts like any other.
                b"row" => {
                    row_number = next_row_number(&event, row_number);
                    row_style = row_default_style(&event);
                    if attr(&event, b"hidden").as_deref().is_some_and(is_true) {
                        hidden_rows.insert(row_number);
                    }
                }
                b"col" if in_cols => {
                    collect_columns(
                        &event,
                        &mut column_ranges,
                        &mut lost_columns,
                        &mut partial_columns,
                    );
                }
                b"mergeCell" if in_merges => {
                    collect_merge(&event, &mut merges, &mut lost_merges);
                }
                b"hyperlink" if in_hyperlinks => {
                    collect_link(&event, link_targets, &mut links, &mut lost_links);
                }
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
            Event::Text(event) => {
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
            Event::GeneralRef(event) => {
                if let Some(cell) = &mut cell {
                    if in_formula {
                        append_entity(&mut cell.formula, &event);
                    } else if in_value || in_text {
                        append_entity(&mut cell.value, &event);
                    }
                }
            }
            Event::End(event) => match event.local_name().as_ref() {
                b"cols" => in_cols = false,
                b"mergeCells" => in_merges = false,
                b"hyperlinks" => in_hyperlinks = false,
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
                        let line = cell.line(book);
                        if cell.shared_lost {
                            lost += 1;
                        }
                        if let Some(line) = line {
                            cells.push(line);
                        }
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    // A sheet part the package left half written reads as one whose data simply
    // ends, so its later rows would be passed off as rows the sheet never had: it is
    // declined whole, as is a part that is no sheet document, while an entry the
    // package left empty ([`blank`]) declares a sheet with no values.
    if part.unreadable() {
        return None;
    }
    let columns = columns.unwrap_or_else(|| resolve_columns(&column_ranges));
    // A `<col>` the sheet writes after its data reaches no cell of it: the columns
    // a cell shows are settled the first time a cell asks for them. Those
    // definitions are counted here and reported rather than dropped in silence.
    let late_columns = settled_columns.map_or(0, |at| column_ranges.len() - at);
    let (rendered, dropped_marks) =
        render_lines(&cells, &hidden_rows, &columns.hidden, &merges, &links);
    Some(SheetRead {
        lines: rendered,
        lost_strings: lost,
        lost_links,
        lost_merges,
        late_columns,
        lost_columns,
        partial_columns,
        dropped_marks,
    })
}

/// The number of a `<row>`: its `r`, or one past `previous` when it has none.
fn next_row_number(event: &BytesStart<'_>, previous: u32) -> u32 {
    attr(event, b"r")
        .and_then(|row| row.trim().parse().ok())
        .unwrap_or(previous.saturating_add(1))
}

/// The style a row gives its own cells: `<row s customFormat="1"/>` styles every
/// cell of the row that declares no style of its own, and it wins over a covering
/// `<col>` — a row that marks its format custom but names no `s` takes the
/// schema's default entry 0 rather than the column's style, and a row naming an `s`
/// this reader cannot read gives its cells a format it cannot show. A row that does
/// not mark the style custom gives its cells none.
fn row_default_style(event: &BytesStart<'_>) -> Option<Style> {
    if !attr(event, b"customFormat").as_deref().is_some_and(is_true) {
        return None;
    }
    Some(match attr(event, b"s") {
        None => Style::Index(0),
        Some(style) => style.trim().parse().map_or(Style::Unread, Style::Index),
    })
}

/// The last column Excel can hold: XFD is the 16384th column. A `<col>`
/// element's declared range is document-controlled, so it is clamped to the
/// grid Excel itself has.
const MAX_COLUMNS: u32 = 16_384;

/// Where a cell's number format comes from.
#[derive(Clone, Copy, Default)]
enum Style {
    /// The cell format a cell shows: the index its own `s` names, else the one its
    /// row or its column gives it.
    Index(usize),
    /// The cell, its row or its column names a style this reader cannot read: no
    /// table holds an entry at it, not even one declaring nothing at all, so the
    /// cell's stored text is shown marked as stored rather than read as one showing
    /// the schema's default.
    Unread,
    /// The cell names no style of its own and neither its row nor its column gives
    /// it one.
    #[default]
    None,
}

/// One `<col>` element as the sheet writes it: the columns it covers, whether it
/// hides them, and the cell format a cell of those columns shows when it declares
/// no style of its own.
struct ColumnRange {
    min: u32,
    max: u32,
    hidden: bool,
    style: Option<Style>,
}

/// A sheet's `<col>` elements once resolved: the columns they hide, and every
/// column's default cell format by its zero-based index — `None` where the sheet
/// gives that column none.
#[derive(Default)]
struct Columns {
    hidden: BTreeSet<u32>,
    styles: Vec<Option<Style>>,
}

/// Record a `<col min max hidden="1" style="2"/>`: the 1-based columns it covers and
/// the style (if any) it gives them; `min` alone means a single column. An element
/// that neither hides its columns nor styles them says nothing, so nothing of it can
/// be lost; one that does and whose range this reader cannot place — a `min` missing,
/// unparsable or outside the grid, or a `max` that does not read — leaves the columns
/// it hides or styles unmarked, so it is counted as a loss. The two ways of losing
/// are counted apart: an element whose `min` it cannot place is applied to no column
/// at all, while one whose `max` does not read still hides or styles the column it
/// starts at. A `style` it names that this reader cannot read is a format the
/// column's cells show and it cannot name, so those cells are marked as stored rather
/// than read as unformatted.
fn collect_columns(
    event: &BytesStart<'_>,
    ranges: &mut Vec<ColumnRange>,
    lost: &mut usize,
    partial: &mut usize,
) {
    let hidden = attr(event, b"hidden").as_deref().is_some_and(is_true);
    let style =
        attr(event, b"style").map(|value| value.trim().parse().map_or(Style::Unread, Style::Index));
    if !hidden && style.is_none() {
        return;
    }
    let Some(min) = attr(event, b"min")
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|min| (1..=MAX_COLUMNS).contains(min))
    else {
        *lost += 1;
        return;
    };
    let max = match attr(event, b"max") {
        None => min,
        Some(value) => {
            if let Ok(max) = value.trim().parse::<u32>() {
                max.clamp(min, MAX_COLUMNS)
            } else {
                // The element's own `min` is one this reader can place, so that
                // column is hidden or styled as the element asks; only its range's
                // end — and the columns it would have covered beyond `min` — is
                // left out.
                *partial += 1;
                min
            }
        }
    };
    ranges.push(ColumnRange {
        min,
        max,
        hidden,
        style,
    });
}

/// Resolve a sheet's `<col>` elements into the columns they hide and every
/// column's default style. Every element may name the grid's whole width, so a
/// column is settled once and never looked at again: a hidden element hides the
/// columns it names, and a column's style is the one of the last element naming
/// it, so the elements are walked backwards for that.
fn resolve_columns(ranges: &[ColumnRange]) -> Columns {
    // A sheet with no `<col>` element hides nothing and styles nothing, which is
    // the shape most sheets have.
    if ranges.is_empty() {
        return Columns::default();
    }
    let mut hidden = BTreeSet::new();
    let mut styles = vec![None; MAX_COLUMNS as usize];
    let mut unhidden: BTreeSet<u32> = (1..=MAX_COLUMNS).collect();
    let mut unstyled: BTreeSet<u32> = (1..=MAX_COLUMNS).collect();
    // The columns of `range` not settled yet, taken out of `open`: a column is
    // looked at once however many elements name it.
    let settle = |open: &mut BTreeSet<u32>, range: &ColumnRange| {
        let columns: Vec<u32> = open.range(range.min..=range.max).copied().collect();
        for column in &columns {
            open.remove(column);
        }
        columns
    };
    for range in ranges.iter().filter(|range| range.hidden) {
        for column in settle(&mut unhidden, range) {
            hidden.insert(column);
        }
    }
    for range in ranges.iter().rev().filter(|range| range.style.is_some()) {
        for column in settle(&mut unstyled, range) {
            styles[column as usize - 1] = range.style;
        }
    }
    Columns { hidden, styles }
}

/// The default style of the column a cell address sits in, or `None` when the
/// sheet gives that column none.
fn column_style(columns: &Columns, reference: &str) -> Option<Style> {
    let column = column_index(reference)?;
    *columns.styles.get(column as usize)?
}

/// Record a `<mergeCell ref="A1:C1"/>`. A merge is sheet structure, so one whose
/// ref is missing or is an address this reader cannot parse (an inverted `C3:A1`
/// included) is counted as lost rather than dropped in silence — while a ref
/// naming one cell covers nothing to mark and is no loss at all.
fn collect_merge(event: &BytesStart<'_>, merges: &mut Vec<Merge>, lost: &mut usize) {
    let Some(reference) = attr(event, b"ref") else {
        *lost += 1;
        return;
    };
    match Rect::parse(&reference) {
        Some(rect) if rect.min_col < rect.max_col || rect.min_row < rect.max_row => {
            merges.push(Merge { rect, reference });
        }
        Some(_) => {}
        None => *lost += 1,
    }
}

/// Record a `<hyperlink ref="A1" r:id="rId1" location="…" display="…"/>`: a link
/// whose `r:id` the sheet's relationships do not declare, whose `ref` is missing
/// or is not a range this reader can parse, is counted as lost rather than
/// rendered pointing nowhere.
fn collect_link(
    event: &BytesStart<'_>,
    targets: &HashMap<String, String>,
    links: &mut Vec<Link>,
    lost: &mut usize,
) {
    match Link::new(event, targets) {
        Some(link) => links.push(link),
        None => *lost += 1,
    }
}

/// A cell's address: its own `r` attribute, or one synthesised from `row` and
/// the running column counter. Either way the counter is left on the column
/// after this cell — a cell with no address still occupies its column. An `r`
/// this reader cannot parse is kept as the workbook wrote it: a value is never
/// dropped for its address (a comment, which is only ever an annotation, is left
/// out and counted instead), it just carries no merge or link mark. It is one line
/// by construction ([`one_line`]): it is printed at the head of the cell's own
/// line, so a break in it must not become a line there.
fn cell_reference(event: &BytesStart<'_>, row: u32, next_column: &mut u32) -> String {
    let Some(reference) = attr(event, b"r") else {
        let reference = format!("{}{}", column_letters(*next_column), row);
        *next_column = next_column.saturating_add(1);
        return reference;
    };
    let reference = one_line(reference);
    if let Some(index) = column_index(&reference) {
        *next_column = index.saturating_add(1);
    }
    reference
}

/// The rectangle a `ref` covers. Both a merge and a hyperlink are one, so the
/// containment test and the parse live here once. A column is zero-based, a row
/// one-based.
#[derive(Clone, Copy)]
struct Rect {
    /// The rectangle's leftmost column, zero-based.
    min_col: u32,
    /// The rectangle's top row, one-based.
    min_row: u32,
    /// The rectangle's rightmost column, zero-based.
    max_col: u32,
    /// The rectangle's bottom row, one-based.
    max_row: u32,
}

impl Rect {
    /// The rectangle `reference` covers — `A1:C1` or a single `A1` — or `None`
    /// when it is not an address this reader can parse, or its minimum is past
    /// its maximum (an inverted `C3:A1` names no cell at all).
    fn parse(reference: &str) -> Option<Self> {
        if let Some((start, end)) = reference.split_once(':') {
            let (min_col, min_row) = cell_position(start)?;
            let (max_col, max_row) = cell_position(end)?;
            if min_col > max_col || min_row > max_row {
                return None;
            }
            return Some(Self {
                min_col,
                min_row,
                max_col,
                max_row,
            });
        }
        let (column, row) = cell_position(reference)?;
        Some(Self {
            min_col: column,
            min_row: row,
            max_col: column,
            max_row: row,
        })
    }

    /// Whether `(column, row)` falls inside the rectangle.
    fn holds(&self, column: u32, row: u32) -> bool {
        column >= self.min_col
            && column <= self.max_col
            && row >= self.min_row
            && row <= self.max_row
    }

    /// Whether `(column, row)` is the rectangle's top-left corner, the cell a
    /// merge's value belongs to.
    fn is_corner(&self, column: u32, row: u32) -> bool {
        column == self.min_col && row == self.min_row
    }
}

/// One `<mergeCell ref="A1:C1"/>`: the rectangle it covers and the ref text as
/// the workbook wrote it.
struct Merge {
    rect: Rect,
    reference: String,
}

/// One `<hyperlink>`: the rectangle it covers, the ref and display text as the
/// workbook wrote them, and the raw target it points at — a URL or a file path,
/// never a package part, so it is kept exactly as the workbook wrote it. Display
/// and target are one line by construction ([`one_line`]): they stand inside the
/// cell's line, so a break in either must not become a line there.
struct Link {
    rect: Rect,
    reference: String,
    display: Option<String>,
    target: String,
}

impl Link {
    /// The link, or `None` when the workbook declares no target for it or its
    /// `ref` is not a range this reader can parse — either way the link is lost
    /// rather than rendered pointing nowhere.
    fn new(event: &BytesStart<'_>, targets: &HashMap<String, String>) -> Option<Self> {
        let reference = attr(event, b"ref")?;
        let rect = Rect::parse(&reference)?;
        let target = match rel_id(event) {
            Some(id) => one_line(targets.get(&id)?.clone()),
            None => one_line(attr(event, b"location")?),
        };
        Some(Self {
            rect,
            reference,
            display: attr(event, b"display").map(one_line),
            target,
        })
    }
}

/// The mark a cell's line carries when its value is the number the cell stores
/// rather than a display of it: the cell's number format was not read, is not one
/// this reader reproduces for this value, or shows nothing at all for it.
const MARK_STORED: &str = " (stored number, format not shown)";

/// A cell's value as this reader shows it.
struct Shown {
    /// The text the cell's line carries: always one line ([`one_line`]).
    text: String,
    /// Whether `text` is the value as the cell stores it rather than a display of
    /// it, which is what makes the line carry [`MARK_STORED`].
    stored: bool,
}

impl Shown {
    /// The value as the cell's own format displays it: a format code the workbook
    /// wrote may carry a break of its own, so the display is made one line here
    /// rather than promised at the call site.
    fn displayed(text: String) -> Self {
        Self {
            text: one_line(text),
            stored: false,
        }
    }

    /// The value as the cell stores it, under [`MARK_STORED`].
    fn stored(text: String) -> Self {
        Self {
            text: one_line(text),
            stored: true,
        }
    }

    /// The mark the line carries after this value, empty when it is a display.
    fn mark(&self) -> &'static str {
        if self.stored { MARK_STORED } else { "" }
    }
}

/// What one cell holds, once its value is rendered.
enum Held {
    /// A value the cell holds, as this reader shows it.
    Value(Shown),
    /// The value and the formula that produces it.
    ValueAndFormula(Shown, String),
    /// A formula whose value is not stored, shown as its own text.
    Formula(String),
}

/// One cell of a sheet, before the marks of the ranges holding it are known: the
/// address as the workbook wrote it, what it holds, and that address as a position.
/// The line the answer carries is built from these in [`render_lines`], where those
/// marks — and the link a caption has to be weighed against — are known too.
struct CellLine {
    /// The cell address, from `r` or synthesised from its row and column.
    reference: String,
    held: Held,
    /// The address's `(column, row)`, or `None` when it is not one.
    position: Option<(u32, u32)>,
}

impl CellLine {
    /// The line before its range and link marks: the address, the value as shown
    /// when the cell holds one — with the mark a value shown as stored carries — and
    /// the formula when it holds one.
    fn text(&self) -> String {
        match &self.held {
            Held::Value(shown) => format!("{}: {}{}", self.reference, shown.text, shown.mark()),
            Held::ValueAndFormula(shown, formula) => format!(
                "{}: {} (={formula}){}",
                self.reference,
                shown.text,
                shown.mark()
            ),
            Held::Formula(formula) => format!("{}: ={formula}", self.reference),
        }
    }

    /// The value the cell shows, when it holds one.
    fn shown(&self) -> Option<&Shown> {
        match &self.held {
            Held::Value(shown) | Held::ValueAndFormula(shown, _) => Some(shown),
            Held::Formula(_) => None,
        }
    }
}

/// The position of a cell address as `(column, row)`: a zero-based column and a
/// one-based row (`B12` -> `(1, 12)`), or `None` when the address is not one.
fn cell_position(reference: &str) -> Option<(u32, u32)> {
    let column = column_index(reference)?;
    let row = reference
        .trim_start_matches(|character: char| character.is_ascii_alphabetic())
        .parse()
        .ok()?;
    Some((column, row))
}

/// The most range-to-cell tests one sheet's marks may cost. Both the ranges and
/// the cells are document-controlled, and only the ranges spanning a cell's own
/// row are ever tested against it, so an ordinary sheet — a handful of merges and
/// links over many cells — cannot spend this; one crafted to put a million ranges
/// over a million cells is marked up to here and the rest is reported, rather
/// than stalling the conversion every reader shares.
const MAX_RANGE_PROBES: usize = 1 << 24;

/// What one sheet's range marks may spend, and whether they ran out.
struct ProbeBudget {
    left: usize,
    exhausted: bool,
}

impl ProbeBudget {
    /// One range-to-cell test: `false` once the budget is spent, which leaves the
    /// rest of the marks out.
    fn spend(&mut self) -> bool {
        if self.left == 0 {
            self.exhausted = true;
            return false;
        }
        self.left -= 1;
        true
    }
}

/// The worksheet's lines: the hidden-geometry annotations first, then one line
/// per valued cell — each carrying its merge mark, then its link mark, when it
/// is part of one — and finally a line per link no cell shows. The flag says the
/// sheet's later range marks were left out because marking them would cost more
/// than [`MAX_RANGE_PROBES`].
fn render_lines(
    cells: &[CellLine],
    hidden_rows: &BTreeSet<u32>,
    hidden_columns: &BTreeSet<u32>,
    merges: &[Merge],
    hyperlinks: &[Link],
) -> (Vec<String>, bool) {
    let mut lines = Vec::new();
    if !hidden_rows.is_empty() {
        lines.push(format!(
            "(hidden rows: {})",
            compress(hidden_rows, '-', |row| row.to_string())
        ));
    }
    if !hidden_columns.is_empty() {
        lines.push(format!(
            "(hidden columns: {})",
            compress(hidden_columns, ':', |column| column_letters(
                column.saturating_sub(1)
            ))
        ));
    }
    // The cells are marked in document order but swept in position order: a range
    // is only ever tested against a cell on a row it spans, so the marks cost the
    // ranges a cell is really in rather than every range declared before it.
    let mut order: Vec<(u32, u32, usize)> = Vec::with_capacity(cells.len());
    for (index, cell) in cells.iter().enumerate() {
        if let Some((column, row)) = cell.position {
            order.push((row, column, index));
        }
    }
    order.sort_unstable();
    let mut budget = ProbeBudget {
        left: MAX_RANGE_PROBES,
        exhausted: false,
    };
    let mut merge_sweep = RowSweep::new(merges.iter().map(|merge| merge.rect));
    let mut link_sweep = RowSweep::new(hyperlinks.iter().map(|link| link.rect));
    let mut merges_at: Vec<Option<(usize, bool)>> = vec![None; cells.len()];
    let mut links_at: Vec<Option<usize>> = vec![None; cells.len()];
    for &(row, column, index) in &order {
        merges_at[index] = containing_merge(column, row, merges, &mut merge_sweep, &mut budget);
        links_at[index] = containing_link(column, row, hyperlinks, &mut link_sweep, &mut budget);
    }
    let mut shown = vec![false; hyperlinks.len()];
    for (index, cell) in cells.iter().enumerate() {
        // A covered cell that kept its own value names the range holding it;
        // the merge's own top-left names the merge instead.
        let mut text = match merges_at[index] {
            Some((merge, true)) => {
                format!("{} (merged {})", cell.text(), merges[merge].reference)
            }
            Some((merge, false)) if cell.shown().is_some() => {
                format!(
                    "{} (in merged range {})",
                    cell.text(),
                    merges[merge].reference
                )
            }
            _ => cell.text(),
        };
        if let Some(link) = links_at[index] {
            shown[link] = true;
            let link = &hyperlinks[link];
            text.push_str(" (link: ");
            // The caption a link carries is what it shows where the cell's own value
            // does not: it is named in front of the target when the two differ, so a
            // caption this reading would otherwise drop is still read.
            let caption = link.display.as_deref().filter(|caption| {
                !caption.is_empty()
                    && cell.shown().map(|shown| shown.text.as_str()) != Some(*caption)
            });
            if let Some(caption) = caption {
                text.push_str(caption);
                text.push_str(" -> ");
            }
            text.push_str(&link.target);
            text.push(')');
        }
        lines.push(text);
    }
    // A link range no cell line covers — a hyperlink over empty cells — still
    // names somewhere to go, so it gets its own line rather than being lost.
    for (index, link) in hyperlinks.iter().enumerate() {
        if shown[index] {
            continue;
        }
        let display = match &link.display {
            Some(display) if !display.is_empty() => format!(" {display}"),
            _ => String::new(),
        };
        lines.push(format!(
            "{}:{display} (link: {})",
            link.reference, link.target
        ));
    }
    (lines, budget.exhausted)
}

/// A sheet's ranges as a sweep down its rows: a range joins the candidates when
/// the first row it spans is reached and leaves once its last row is passed, so a
/// cell is only ever tested against a range that can actually hold it. A cell and
/// a range are both document-controlled, and testing every cell against every
/// range whose corner reaches it — what a sheet with a few dozen merges against a
/// few hundred thousand cells needs — costs their product; the sweep costs each
/// cell the ranges over its own row, plus one push and one pop per range.
struct RowSweep {
    /// Every range's rectangle, by its index in the sheet's own list.
    rects: Vec<Rect>,
    /// `(first row, range index)` sorted by first row: the ranges not yet reached.
    pending: Vec<(u32, usize)>,
    /// How many of `pending` have been reached.
    reached: usize,
    /// The ranges reached and not yet passed, read in the order they leave rather
    /// than the order the sheet declares them: the soonest to expire is at the
    /// front, so dropping the ones a row is past costs a pop each however many are
    /// held — a sweep whose ranges end on different rows never re-reads the ones
    /// that are still holding.
    active: BinaryHeap<Reverse<(u32, usize)>>,
}

impl RowSweep {
    /// A sweep over `rects`, ordered by the row each range begins on.
    fn new(rects: impl Iterator<Item = Rect>) -> Self {
        let rects: Vec<Rect> = rects.collect();
        let mut pending: Vec<(u32, usize)> = rects
            .iter()
            .enumerate()
            .map(|(index, rect)| (rect.min_row, index))
            .collect();
        pending.sort_unstable();
        Self {
            rects,
            pending,
            reached: 0,
            active: BinaryHeap::new(),
        }
    }

    /// The ranges spanning `row`, in no particular order. Rows are queried in
    /// increasing order.
    fn covering(&mut self, row: u32) -> impl Iterator<Item = usize> + '_ {
        while self.reached < self.pending.len() && self.pending[self.reached].0 <= row {
            let index = self.pending[self.reached].1;
            self.reached += 1;
            self.active
                .push(Reverse((self.rects[index].max_row, index)));
        }
        while self
            .active
            .peek()
            .is_some_and(|&Reverse((last_row, _))| last_row < row)
        {
            self.active.pop();
        }
        self.active.iter().map(|&Reverse((_, index))| index)
    }
}

/// The index of the first range in document order whose rectangle holds the cell
/// its `holds` closure answers for — `covering` is the ranges spanning that cell's
/// row — or `None`. Every rectangle actually tested spends one of the sheet's
/// probes.
fn first_holding(
    covering: impl Iterator<Item = usize>,
    budget: &mut ProbeBudget,
    holds: impl Fn(usize) -> bool,
) -> Option<usize> {
    let mut best = None;
    for index in covering {
        if !budget.spend() {
            break;
        }
        if holds(index) {
            best = Some(best.map_or(index, |best: usize| best.min(index)));
        }
    }
    best
}

/// The index of the merge a cell falls in, and whether the cell is that merge's
/// top-left corner — the first merge in document order that holds it.
fn containing_merge(
    column: u32,
    row: u32,
    merges: &[Merge],
    sweep: &mut RowSweep,
    budget: &mut ProbeBudget,
) -> Option<(usize, bool)> {
    let index = first_holding(sweep.covering(row), budget, |index| {
        merges[index].rect.holds(column, row)
    })?;
    Some((index, merges[index].rect.is_corner(column, row)))
}

/// The index of the link a cell falls in, so the line it is rendered on can carry
/// the target and the link is not repeated below.
fn containing_link(
    column: u32,
    row: u32,
    links: &[Link],
    sweep: &mut RowSweep,
    budget: &mut ProbeBudget,
) -> Option<usize> {
    first_holding(sweep.covering(row), budget, |index| {
        links[index].rect.holds(column, row)
    })
}

/// A sorted set of coordinates compressed into runs, each run written by `label`
/// over its endpoints and joined by `joiner`: `5, 12-14` style text.
fn compress(numbers: &BTreeSet<u32>, joiner: char, label: impl Fn(u32) -> String) -> String {
    let mut runs = Vec::new();
    let mut numbers = numbers.iter().copied().peekable();
    while let Some(start) = numbers.next() {
        let mut end = start;
        while numbers.peek().copied() == end.checked_add(1) {
            end += 1;
            numbers.next();
        }
        if start == end {
            runs.push(label(start));
        } else {
            runs.push(format!("{}{joiner}{}", label(start), label(end)));
        }
    }
    runs.join(", ")
}

/// One `<c>` cell being read.
#[derive(Default)]
struct Cell {
    /// The cell address, from `r` or synthesised from its row and column.
    reference: String,
    /// The `t` attribute: how `value` is interpreted.
    kind: Option<String>,
    /// Where this cell's number format comes from: its own `s`, else the default of
    /// its row or of its column, else nothing — or [`Style::Unread`] when the cell
    /// names a style this reader cannot read.
    style: Style,
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
    /// Start a cell from its `<c>` element. A cell with no `s` of its own shows
    /// the default style of its row, else the one of its column, else none — so
    /// its address is settled first, a cell with no `r` included.
    fn start(
        event: &BytesStart<'_>,
        row: u32,
        row_style: Option<Style>,
        columns: &Columns,
        next_column: &mut u32,
    ) -> Self {
        let reference = cell_reference(event, row, next_column);
        let style = match attr(event, b"s") {
            // A cell naming a style this reader cannot read is not one of an
            // unformatted cell: its stored text is shown marked as stored rather
            // than passed off as the display, and the cell takes no row or column
            // style either.
            Some(style) => style.trim().parse().map_or(Style::Unread, Style::Index),
            None => row_style
                .or_else(|| column_style(columns, &reference))
                .unwrap_or_default(),
        };
        Self {
            reference,
            kind: attr(event, b"t"),
            style,
            ..Self::default()
        }
    }

    /// On `<f ...>`: the formula this cell holds, and whether it is a shared
    /// definition (whose text is kept for the references that follow) or one of
    /// those references.
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

    /// What the cell holds. `None` when it holds neither a value nor a formula.
    fn line(&mut self, book: Book<'_>) -> Option<CellLine> {
        let shown = self.value(book);
        // The formula is the workbook's own text like a value, and it is printed on
        // the cell's line, so it is one line too ([`one_line`]).
        let formula = self
            .has_formula
            .then(|| one_line(self.formula.trim().to_owned()));
        let held = match (shown, formula) {
            (Some(shown), Some(formula)) => Held::ValueAndFormula(shown, formula),
            (Some(shown), None) => Held::Value(shown),
            (None, Some(formula)) => Held::Formula(formula),
            (None, None) => return None,
        };
        Some(CellLine {
            position: cell_position(&self.reference),
            reference: std::mem::take(&mut self.reference),
            held,
        })
    }

    /// The cell's value as shown, resolved by the `t` attribute — `None` when the
    /// cell holds none, or holds one that shows as nothing at all. A shared-string
    /// or inline string keeps its own spacing; every other type is trimmed of the
    /// indentation a pretty-printed part puts inside `<v>`. A value the workbook
    /// wrote with a line break in it is shown as one line (`\n`), so no value can
    /// stand where this reader's own lines do.
    fn value(&mut self, book: Book<'_>) -> Option<Shown> {
        let shown = match self.kind.as_deref() {
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
                    .and_then(|index| book.shared.get(index));
                let Some(text) = text else {
                    self.shared_lost = true;
                    return None;
                };
                Shown::displayed(text.clone())
            }
            Some("b") => Shown::displayed(match self.value.trim() {
                "1" => "TRUE".to_owned(),
                "0" => "FALSE".to_owned(),
                other => other.to_owned(),
            }),
            Some("inlineStr") => Shown::displayed(self.value.clone()),
            // A numeric cell (`t` absent or `n`) is shown as its number format
            // displays it.
            None | Some("n") => self.number(book),
            _ => Shown::displayed(self.value.trim().to_owned()),
        };
        (!shown.text.is_empty()).then_some(shown)
    }

    /// A numeric cell's value as shown, or its stored text under [`MARK_STORED`]
    /// when no display of it was produced.
    fn number(&self, book: Book<'_>) -> Shown {
        let raw = self.value.trim().to_owned();
        // A cell's own `s`, else the workbook's first cell format: `s` defaults to
        // the first entry, so a cell naming none and one naming `0` are the same
        // cell to Excel.
        let table = &book.styles.cell_xfs;
        let index = match self.style {
            Style::Unread => return Shown::stored(raw),
            Style::Index(index) => index,
            Style::None => 0,
        };
        let id = match table.get(index) {
            // The schema's default entry 0 — which a cell naming no style also shows —
            // stands in for itself where the workbook declares no format at all, the one
            // way a table this reader did read comes back empty.
            None if index == 0 && !book.styles.lost => 0,
            // Every other entry the table does not hold, and an entry whose own
            // declaration this reader could not read, is a display it cannot know: the
            // cell's stored text is marked rather than passed off as the display.
            None | Some(None) => return Shown::stored(raw),
            Some(Some(id)) => *id,
        };
        let code = book.styles.num_fmts.get(&id).map(String::as_str);
        if numfmt::is_general(id, code) {
            return Shown::displayed(raw);
        }
        let Ok(value) = raw.parse::<f64>() else {
            return Shown::displayed(raw);
        };
        match numfmt::display(value, id, code, book.date1904) {
            Some(shown) => Shown::displayed(shown),
            None => Shown::stored(raw),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ooxml::test_fixtures::zip_fixture;
    use std::fmt::Write as _;

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

    /// A sheet part the package left half written — truncated at an element
    /// boundary, where its events simply end — is reported like one that will not
    /// parse, rather than read as a sheet whose data ends where the truncation is.
    #[test]
    fn xlsx_reports_a_sheet_part_truncated_at_an_element_boundary() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable workbook");
        };
        assert_eq!(text, "");
        assert_eq!(notes, ["sheet \"Data\" could not be read"]);
    }

    /// A workbook part the package left half written — cut at an element boundary
    /// after one of its `<sheet/>` entries, or inside a tag — still yields the sheets
    /// it declared before the cut, and the loss of the sheets and names that would
    /// have followed is said once rather than passed off as the whole list.
    #[test]
    fn xlsx_keeps_the_sheets_a_cut_workbook_part_declared() {
        for workbook in [
            // Cut right after the second sheet: `</sheets>` and `</workbook>` are
            // missing, and the events simply end.
            br#"<workbook><sheets><sheet name="Data" r:id="rId1"/><sheet name="More" r:id="rId2"/>"#
                .as_slice(),
            // Cut inside the third sheet's tag, where the walk gets a parse error
            // instead of an ending.
            br#"<workbook><sheets><sheet name="Data" r:id="rId1"/><sheet name="More" r:id="rId2"/><sheet name="Th"#
                .as_slice(),
        ] {
            let bytes = zip_fixture(&[
                ("xl/workbook.xml", workbook),
                (
                    "xl/_rels/workbook.xml.rels",
                    br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>"#,
                ),
                (
                    "xl/worksheets/sheet1.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
                ),
                (
                    "xl/worksheets/sheet2.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1"><v>2</v></c></row></sheetData></worksheet>"#,
                ),
            ]);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable workbook");
            };
            assert_eq!(text, "Sheet \"Data\":\n  A1: 1\n\nSheet \"More\":\n  A1: 2");
            assert_eq!(notes, [WORKBOOK_CUT_NOTE]);
        }
    }

    /// A workbook part that is well-formed XML but no workbook — an error page in
    /// its place — carries no workbook element, so it is read like a part the package
    /// does not carry and costs the whole document. A `<workbook/>` declaring no
    /// sheet is a workbook still: it reads as an empty sheet list, with the cause of
    /// the empty answer said rather than left to the delivery layer's "no text".
    #[test]
    fn xlsx_reads_a_part_carrying_no_workbook_element_like_a_missing_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let html = zip_fixture(&[(
            "xl/workbook.xml",
            br"<html><head><title>404</title></head><body>Not Found</body></html>",
        )]);
        assert!(matches!(
            convert_xlsx(&html, dir.path()),
            DocOutcome::Unreadable { .. }
        ));
        let empty = zip_fixture(&[("xl/workbook.xml", b"<workbook/>".as_slice())]);
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&empty, dir.path()) else {
            panic!("expected Text outcome for a workbook declaring no sheet");
        };
        assert_eq!(text, "");
        assert_eq!(notes, [WORKBOOK_NO_SHEETS_NOTE]);
    }

    /// A sheet or comments part that is well-formed XML but carries no element of
    /// its own — an error page in its place — is read like a part the package does
    /// not carry: the sheet is reported as unreadable rather than as a sheet with no
    /// values, and the comments as lost rather than as a sheet without any. An entry
    /// the package left empty declares a sheet with no values and a comments part
    /// with no comments, so neither is a loss.
    #[test]
    fn xlsx_reads_a_sheet_or_comments_part_carrying_no_such_element_like_a_missing_one() {
        let error_page: &[u8] =
            b"<html><head><title>404</title></head><body>Not Found</body></html>";
        let sheet: &[u8] =
            br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#;
        for (sheet_part, comments_part, expected_text, expected_notes) in [
            (
                error_page,
                b"".as_slice(),
                "",
                &["sheet \"Data\" could not be read"][..],
            ),
            (
                sheet,
                error_page,
                "Sheet \"Data\":\n  A1: 1",
                &["sheet \"Data\": its comments could not be read"][..],
            ),
            (
                b"".as_slice(),
                b"".as_slice(),
                "Sheet \"Data\": (no values)",
                &[][..],
            ),
        ] {
            let bytes = zip_fixture(&[
                (
                    "xl/workbook.xml",
                    br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                ),
                (
                    "xl/_rels/workbook.xml.rels",
                    br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
                ),
                (
                    "xl/worksheets/_rels/sheet1.xml.rels",
                    br#"<Relationships><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
                ),
                ("xl/worksheets/sheet1.xml", sheet_part),
                ("xl/comments1.xml", comments_part),
            ]);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable workbook");
            };
            assert_eq!(text, expected_text);
            assert_eq!(notes, expected_notes);
        }
    }

    /// A sheet the workbook names but gives no relationship is read by the part its
    /// position names rather than dropped: its cells would otherwise vanish, and
    /// every sheet after it would be matched to the wrong scoped defined name.
    #[test]
    fn xlsx_reads_a_sheet_whose_relationship_is_not_named() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/><sheet name="Bare" sheetId="2"/></sheets><definedNames><definedName name="Second" localSheetId="1">First!$A$1</definedName></definedNames></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/sheet2.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>2</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 1\n\nSheet \"Bare\":\n  A1: 2\n\n\
             Defined names:\n  Second (sheet \"Bare\"): First!$A$1"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
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

    /// The shared-string table is named by a workbook relationship like the
    /// styles part, so a producer that lays it out at a non-conventional path
    /// still has its strings read — the conventional name is only the fallback.
    #[test]
    fn xlsx_resolves_the_shared_string_table_through_the_workbook_rels() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings2.xml"/></Relationships>"#,
            ),
            ("xl/sharedStrings2.xml", SHARED_STRINGS),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>2</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable workbook");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: hello\n  B1: Жук");
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A cell is shown the way its number format displays it: a date, a
    /// percentage and a custom currency resolved through `numFmtId`s, and the
    /// stored text (marked) for a format this reader cannot reproduce or one
    /// that shows nothing.
    #[test]
    fn xlsx_renders_cells_with_their_number_formats() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br##"<styleSheet><numFmts><numFmt numFmtId="164" formatCode="#,##0.00&quot; &#8364;&quot;"/><numFmt numFmtId="165" formatCode="# ?/?"/><numFmt numFmtId="166" formatCode=";;;"/></numFmts><cellXfs count="6"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="9"/><xf numFmtId="164"/><xf numFmtId="165"/><xf numFmtId="166"/></cellXfs></styleSheet>"##,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" s="1"><v>46300</v></c><c r="B1" s="2"><v>0.25</v></c><c r="C1" s="3"><v>1234.5</v></c><c r="D1" s="4"><v>1.5</v></c><c r="E1" s="5"><v>5</v></c><c r="F1"><v>42</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 2026-10-05\n  B1: 25%\n  C1: 1,234.50 €\n  D1: 1.5 (stored number, format not shown)\n  E1: 5 (stored number, format not shown)\n  F1: 42"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A workbook in the 1904 date system shows its serial as a 1904-based
    /// date.
    #[test]
    fn xlsx_reads_the_1904_date_system() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><workbookPr date1904="1"/><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellXfs count="1"><xf numFmtId="14"/></cellXfs></styleSheet>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" s="0"><v>0</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1904-01-01");
    }

    /// A cell format declaring no format of its own shows the one the base style
    /// it names holds — how a workbook that leaves its number format to a named
    /// style writes it — so its serial is never passed off as the display; a
    /// format that applies its own declaration keeps it, even when that
    /// declaration is `General`. A base style whose declaration this reader cannot
    /// read, one it names by an `xfId` that does not parse, and one past the
    /// entries the base-style list holds all leave the format the entry would show
    /// unknown — while a format declaring one of its own keeps it, the base style
    /// being only what a declaration of nothing of its own falls back to.
    #[test]
    fn xlsx_inherits_a_cell_format_from_its_base_style() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellStyleXfs count="4"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="9"/><xf numFmtId="bad"/></cellStyleXfs><cellXfs count="9"><xf numFmtId="0"/><xf numFmtId="0" xfId="1"/><xf numFmtId="0" xfId="2" applyNumberFormat="0"><alignment horizontal="center"/></xf><xf numFmtId="0" xfId="1" applyNumberFormat="1"/><xf numFmtId="0" xfId="3"/><xf numFmtId="0" xfId="9"/><xf numFmtId="14" xfId="bad"/><xf numFmtId="0" xfId="bad"/><xf numFmtId="0" xfId="bad" applyNumberFormat="1"/></cellXfs></styleSheet>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" s="0"><v>45000</v></c><c r="B1" s="1"><v>45000</v></c><c r="C1" s="2"><v>0.25</v></c><c r="D1" s="3"><v>45000</v></c><c r="E1" s="4"><v>45000</v></c><c r="F1" s="5"><v>45000</v></c><c r="G1" s="6"><v>45000</v></c><c r="H1" s="7"><v>45000</v></c><c r="I1" s="8"><v>45000</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 45000\n  B1: 2023-03-15\n  C1: 25%\n  D1: 45000\n  \
             E1: 45000 (stored number, format not shown)\n  \
             F1: 45000 (stored number, format not shown)\n  G1: 2023-03-15\n  \
             H1: 45000 (stored number, format not shown)\n  I1: 45000"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A cell with no `s` of its own shows the default style of its row — only
    /// when the row marks the style custom — else the one of its column (a cell
    /// with no `r` included: its synthesised address is what its column is read
    /// from); a cell that declares a style keeps it, and a row or column that
    /// gives none leaves the stored number.
    #[test]
    fn xlsx_shows_a_rows_or_columns_default_cell_style() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellXfs count="3"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="9"/></cellXfs></styleSheet>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><cols><col min="2" max="2" style="2"/><col min="4" max="4" style="1"/></cols><sheetData><row r="1" s="1" customFormat="1"><c r="A1"><v>45000</v></c><c r="B1"><v>45000</v></c><c r="C1" s="0"><v>45000</v></c></row><row r="2"><c r="A2"><v>45000</v></c><c><v>0.25</v></c><c r="D2"><v>45000</v></c></row><row r="3" s="1"><c r="A3"><v>45000</v></c></row><row r="4" customFormat="1"><c r="B4"><v>45000</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 2023-03-15\n  B1: 2023-03-15\n  C1: 45000\n  A2: 45000\n  B2: 25%\n  D2: 2023-03-15\n  A3: 45000\n  B4: 45000"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A styles part the workbook declares and this reader gets no format table
    /// out of — a mismatched end tag, a part that is no stylesheet at all (an error
    /// page a producer left in the part's place parses as XML just as happily), or
    /// one the package does not carry — costs every cell its format: the loss is
    /// said out loud, and a cell naming a format then reads as its stored text.
    #[test]
    fn xlsx_reports_styles_it_cannot_read() {
        for styles in [
            Some(b"<styleSheet><cellXfs></styleSheet>".as_slice()),
            Some(b"<html><head><title>404</title></head><body>Not Found</body></html>".as_slice()),
            // Truncated at an element boundary, where the events simply end.
            Some(b"<styleSheet><cellXfs>".as_slice()),
            None,
        ] {
            let mut parts: Vec<(&str, &[u8])> = vec![
                (
                    "xl/workbook.xml",
                    br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                ),
                (
                    "xl/_rels/workbook.xml.rels",
                    br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
                ),
                (
                    "xl/worksheets/sheet1.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1" s="3"><v>1234.5</v></c></row></sheetData></worksheet>"#,
                ),
            ];
            if let Some(styles) = styles {
                parts.push(("xl/styles.xml", styles));
            }
            let bytes = zip_fixture(&parts);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable workbook");
            };
            assert_eq!(
                text,
                "Sheet \"Data\":\n  A1: 1234.5 (stored number, format not shown)"
            );
            assert_eq!(notes, [STYLES_NOTE]);
        }
    }

    /// A stylesheet that declares a format but no cell format to name it from — a
    /// `numFmt` (readable or not), a base-style list, or a `cellXfs` list holding no
    /// entry — is one this reader cannot derive any cell's format from: every cell —
    /// a serial a date format would have shown included — comes back as the number
    /// it stores, marked as stored, and the loss is said. A stylesheet declaring
    /// nothing, like `<styleSheet/>`, declares no format for any cell: the cells all
    /// show the default one, and nothing is lost. An `s` that is not even a number is
    /// a different matter again: the workbook says the cell carries a format this
    /// reader cannot read, so its stored value is marked as stored even in the
    /// no-format shape — as is an `s` the empty table does not hold, since entry 0 is
    /// the only one that stands for a cell naming no format.
    #[test]
    fn xlsx_reads_a_stylesheet_declaring_no_cell_format() {
        for (styles, expected_text, expected_notes) in [
            (
                b"<styleSheet/>".as_slice(),
                "Sheet \"Data\":\n  A1: 1234.5\n  \
                 B1: 1234.5 (stored number, format not shown)\n  C1: 46300\n  \
                 D1: 46300 (stored number, format not shown)",
                &[][..],
            ),
            (
                b"<styleSheet><cellXfs/></styleSheet>".as_slice(),
                "Sheet \"Data\":\n  A1: 1234.5 (stored number, format not shown)\n  \
                 B1: 1234.5 (stored number, format not shown)\n  \
                 C1: 46300 (stored number, format not shown)\n  \
                 D1: 46300 (stored number, format not shown)",
                &[STYLES_NOTE][..],
            ),
            (
                b"<styleSheet><numFmts><numFmt numFmtId=\"14\"/></numFmts></styleSheet>"
                    .as_slice(),
                "Sheet \"Data\":\n  A1: 1234.5 (stored number, format not shown)\n  \
                 B1: 1234.5 (stored number, format not shown)\n  \
                 C1: 46300 (stored number, format not shown)\n  \
                 D1: 46300 (stored number, format not shown)",
                &[STYLES_NOTE][..],
            ),
            (
                b"<styleSheet><numFmts><numFmt numFmtId=\"14\" formatCode=\"yyyy-mm-dd\"/></numFmts></styleSheet>".as_slice(),
                "Sheet \"Data\":\n  A1: 1234.5 (stored number, format not shown)\n  \
                 B1: 1234.5 (stored number, format not shown)\n  \
                 C1: 46300 (stored number, format not shown)\n  \
                 D1: 46300 (stored number, format not shown)",
                &[STYLES_NOTE][..],
            ),
        ] {
            let bytes = zip_fixture(&[
                (
                    "xl/workbook.xml",
                    br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                ),
                (
                    "xl/_rels/workbook.xml.rels",
                    br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
                ),
                ("xl/styles.xml", styles),
                (
                    "xl/worksheets/sheet1.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1234.5</v></c><c r="B1" s="99999999999999999999"><v>1234.5</v></c><c r="C1" s="0"><v>46300</v></c><c r="D1" s="3"><v>46300</v></c></row></sheetData></worksheet>"#,
                ),
            ]);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable workbook");
            };
            assert_eq!(text, expected_text);
            assert_eq!(notes, expected_notes);
        }
    }

    /// An entry the package left empty declares nothing, so the part it stands for
    /// has no loss to report: an empty styles, comments, workbook-relationships or
    /// sheet-relationships part reads as one that declares no formats, no comments
    /// and no relationships, and no note is added. A part whose bytes are something
    /// other than a document — an error page a producer left in its place — is a
    /// loss, and is reported as one.
    #[test]
    fn xlsx_reads_an_empty_part_as_one_that_declares_nothing() {
        let sheet_rels: &[u8] = br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#;
        for (styles, comments, rels, sheet_rels, expected_notes) in [
            (
                b"".as_slice(),
                b"".as_slice(),
                b"".as_slice(),
                b"".as_slice(),
                &[][..],
            ),
            // The byte-order mark a producer may open an empty part with says how the
            // document would be encoded, not that it holds anything: the entry is as
            // empty as one the package left with no bytes at all.
            (
                b"\xEF\xBB\xBF".as_slice(),
                b"\xEF\xBB\xBF".as_slice(),
                b"\xEF\xBB\xBF".as_slice(),
                b"\xEF\xBB\xBF".as_slice(),
                &[][..],
            ),
            (
                b"Not Found".as_slice(),
                b"Not Found".as_slice(),
                b"Not Found".as_slice(),
                sheet_rels,
                &[
                    WORKBOOK_RELS_NOTE,
                    STYLES_NOTE,
                    "sheet \"Data\": its comments could not be read",
                ][..],
            ),
        ] {
            let bytes = zip_fixture(&[
                (
                    "xl/workbook.xml",
                    br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                ),
                ("xl/_rels/workbook.xml.rels", rels),
                ("xl/styles.xml", styles),
                (
                    "xl/worksheets/sheet1.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>x</t></is></c></row></sheetData></worksheet>"#,
                ),
                ("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels),
                ("xl/comments1.xml", comments),
            ]);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable workbook");
            };
            assert_eq!(text, "Sheet \"Data\":\n  A1: x");
            assert_eq!(notes, expected_notes);
        }
    }

    /// A relationships part the package cut at an element boundary names its
    /// remaining parts nowhere, so the list it got to is not the workbook's or the
    /// sheet's own: it is read like one that could not be read at all, and said
    /// rather than passed off as a workbook or a sheet declaring no relationship —
    /// which is what would lose the comments the sheet names after the cut.
    #[test]
    fn xlsx_reports_a_cut_relationships_part() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            // Cut after the sheet's own relationship: the styles part is named after
            // the cut and only found by its conventional name, if at all.
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml">"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
            // Cut before the comments relationship: the part it names is not read,
            // and the sheet says so rather than reading it as a sheet without one.
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId1""#,
            ),
            (
                "xl/comments1.xml",
                br#"<comments><authors><author>Ivan</author></authors><commentList><comment ref="A1" authorId="0"><text>note</text></comment></commentList></comments>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable workbook");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1");
        assert_eq!(
            notes,
            [
                WORKBOOK_RELS_NOTE,
                "sheet \"Data\": its relationships could not be read"
            ]
        );
    }

    /// A cell naming a format the workbook's table does not hold — an index past
    /// its `cellXfs` entries, an entry whose own declaration does not parse, an entry
    /// inheriting from a base style that does not parse either, a row or a column
    /// naming an `s` this reader cannot read — shows its stored value marked as
    /// stored: a serial the reader cannot resolve a format for is never passed off as
    /// the display. A cell showing only its formula has no stored value to mark, and a
    /// cell whose row and column declare no style keeps the schema's default.
    #[test]
    fn xlsx_marks_a_cell_whose_format_the_table_does_not_hold() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellStyleXfs count="1"><xf numFmtId="14"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0"/><xf numFmtId="14abc"/><xf numFmtId="0" xfId="bad"/></cellXfs></styleSheet>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><cols><col min="3" max="3" style="oops"/></cols><sheetData><row r="1"><c r="A1" s="0"><v>1234.5</v></c><c r="B1" s="3"><v>1234.5</v></c><c r="C1" s="9"><f>SUM(A1)</f></c><c r="D1" s="99999999999999999999"><v>1234.5</v></c><c r="E1" s="1"><v>1234.5</v></c></row><row r="2" s="abc" customFormat="1"><c r="A2"><v>1234.5</v></c><c r="B2"><v>1234.5</v></c></row><row r="3"><c r="A3" s="2"><v>45000</v></c><c r="C3"><v>45000</v></c><c r="D3"><v>45000</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 1234.5\n  B1: 1234.5 (stored number, format not shown)\n  \
             C1: =SUM(A1)\n  D1: 1234.5 (stored number, format not shown)\n  \
             E1: 1234.5 (stored number, format not shown)\n  \
             A2: 1234.5 (stored number, format not shown)\n  \
             B2: 1234.5 (stored number, format not shown)\n  \
             A3: 45000 (stored number, format not shown)\n  \
             C3: 45000 (stored number, format not shown)\n  D3: 45000"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A hidden sheet is marked in its header — the state a workbook writes is one
    /// of two words, whichever case it writes them in — and a sheet's hidden rows
    /// and columns are compressed and annotated before its cells.
    #[test]
    fn xlsx_marks_hidden_sheets_rows_and_columns() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/><sheet name="Hidden" state="hidden" r:id="rId2"/><sheet name="Secret" state="VERYHIDDEN" r:id="rId3"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Target="worksheets/sheet3.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><cols><col min="3" max="5" hidden="1"/><col min="7" hidden="1"/></cols><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="5" hidden="1"><c r="A5"><v>5</v></c></row><row r="12" hidden="1"/><row r="13" hidden="true"/><row r="14" hidden="1"/></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/sheet2.xml",
                b"<worksheet><sheetData/></worksheet>",
            ),
            (
                "xl/worksheets/sheet3.xml",
                b"<worksheet><sheetData/></worksheet>",
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  (hidden rows: 5, 12-14)\n  (hidden columns: C:E, G)\n  A1: 1\n  A5: 5\n\nSheet \"Hidden\" (hidden): (no values)\n\nSheet \"Secret\" (very hidden): (no values)"
        );
    }

    /// A `<cols>` section that names the same range over and over costs the reader
    /// its own element count, never that count times the grid's width, so a sheet
    /// whose section is huge still marks its own shape rather than stalling.
    #[test]
    fn xlsx_resolves_a_repeated_columns_section_once() {
        let mut sheet = String::from("<worksheet><cols>");
        for _ in 0..20_000u32 {
            sheet.push_str("<col min=\"1\" max=\"16384\" hidden=\"1\" style=\"1\"/>");
        }
        sheet.push_str(
            "</cols><sheetData><row r=\"1\"><c r=\"A1\"><v>45000</v></c></row></sheetData></worksheet>",
        );
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>"#,
            ),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  (hidden columns: A:XFD)\n  A1: 2023-03-15"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A `<cols>` section the sheet writes after its data reaches no cell of it —
    /// the columns a cell shows are settled the first time a cell asks for them —
    /// so it is counted and said rather than dropped in silence. A `<col>` whose
    /// `min` this reader cannot place is applied to no column at all, and one whose
    /// `max` does not read still hides or styles the column it starts at: the two
    /// are counted and said apart.
    #[test]
    fn xlsx_reports_late_and_unplaceable_column_definitions() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><cols><col min="abc" hidden="1"/><col min="16385" hidden="1"/><col min="2" max="zz" style="1"/><col min="3" max="3" hidden="1"/></cols><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData><cols><col min="4" max="5" hidden="1"/></cols></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  (hidden columns: C)\n  A1: 1");
        assert_eq!(
            notes,
            [
                "sheet \"Data\": 1 column definition(s) left out: the sheet writes them after \
                 its data, where they reach no cell of it",
                "sheet \"Data\": 2 column definition(s) left out: the columns they hide or \
                 style are not ones this reader can place",
                "sheet \"Data\": 1 column definition(s) cover a range whose end this reader \
                 cannot place, so only the column each one starts at is hidden or styled"
            ]
        );
    }

    /// A merge marks its top-left cell with the range, and a covered cell that
    /// kept its own value with the range that holds it; a covered cell with no
    /// value emits nothing, and a merge over a single cell covers nothing to mark
    /// and is no loss at all.
    #[test]
    fn xlsx_marks_merged_ranges() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Title</t></is></c><c r="B1" t="inlineStr"><is><t>extra</t></is></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>kept</t></is></c></row></sheetData><mergeCells count="3"><mergeCell ref="A1:C1"/><mergeCell ref="A3:B3"/><mergeCell ref="D4"/></mergeCells></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: Title (merged A1:C1)\n  B1: extra (in merged range A1:C1)\n  A3: kept (merged A3:B3)"
        );
        assert!(notes.is_empty(), "nothing was left out: {notes:?}");
    }

    /// A merge is sheet structure, so one whose `ref` is missing or is not a
    /// range this reader can mark is counted and reported rather than dropped in
    /// silence — while the sheet's cells and the merges around it still render.
    #[test]
    fn xlsx_reports_merges_it_cannot_mark() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData><mergeCells count="3"><mergeCell ref="A1:B1"/><mergeCell ref="C3:A1"/><mergeCell/></mergeCells></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1 (merged A1:B1)");
        assert_eq!(
            notes,
            [
                "sheet \"Data\": 2 merged range(s) left out: the cell range they name is not one this reader can mark"
            ]
        );
    }

    /// A sheet declaring far more merged ranges than any real book pays for the
    /// marks past the reader's budget and says so, rather than stalling the
    /// conversion that every reader shares.
    #[test]
    fn xlsx_reports_range_marks_it_could_not_afford() {
        let mut sheet = String::from("<worksheet><sheetData>");
        for row in 1..=4_096u32 {
            let _ = write!(sheet, "<row r=\"{row}\"><c r=\"A{row}\"><v>1</v></c></row>");
        }
        sheet.push_str("</sheetData><mergeCells count=\"8192\">");
        // Every merge spans every cell's row, so each cell's candidates are the
        // whole list and the budget is spent before the cells are through.
        for _ in 0..8_192u32 {
            sheet.push_str("<mergeCell ref=\"A1:B1048576\"/>");
        }
        sheet.push_str("</mergeCells></worksheet>");
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        // The cells before the budget still carry their mark.
        assert!(text.contains("A1: 1 (merged A1:B1048576)"), "{text}");
        assert!(
            notes
                .iter()
                .any(|note| note.contains("later merged ranges and links left out")),
            "the dropped marks are reported: {notes:?}"
        );
    }

    /// A sheet's ordinary shape — a merged label on every row of a large sheet —
    /// is marked in full rather than cut short: a range is only tested against a
    /// cell on a row it spans, so the ranges a cell is not in cost it nothing.
    #[test]
    fn xlsx_marks_a_large_sheet_within_its_budget() {
        let mut sheet = String::from("<worksheet><sheetData>");
        for row in 1..=2_048u32 {
            let _ = write!(sheet, "<row r=\"{row}\">");
            for column in 0..100u32 {
                let _ = write!(
                    sheet,
                    "<c r=\"{}{row}\"><v>1</v></c>",
                    column_letters(column)
                );
            }
            sheet.push_str("</row>");
        }
        sheet.push_str("</sheetData><mergeCells count=\"2048\">");
        for row in 1..=2_048u32 {
            let _ = write!(sheet, "<mergeCell ref=\"A{row}:B{row}\"/>");
        }
        sheet.push_str("</mergeCells></worksheet>");
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert!(notes.is_empty(), "nothing was left out: {notes:?}");
        assert!(text.contains("A1: 1 (merged A1:B1)"), "{text}");
        assert!(text.contains("B1: 1 (in merged range A1:B1)"), "{text}");
        assert!(text.contains("A2048: 1 (merged A2048:B2048)"), "{text}");
        assert!(
            text.contains("\n  C1: 1\n"),
            "a cell in no merge is left unmarked: {text}"
        );
    }

    /// A sheet whose ranges all start on the first row and end on their own: every
    /// range is held at once, and each row the sweep reaches is past one more of
    /// them than the row before. The marks must not depend on the order the sweep
    /// drops what it is past, so every cell is marked, and the first merge holding a
    /// row is the shortest one reaching it — the marks a scan of the whole list in
    /// document order would find.
    #[test]
    fn xlsx_marks_ranges_that_end_on_their_own_rows() {
        const ROWS: u32 = 1_024;
        let mut sheet = String::from("<worksheet><sheetData>");
        for row in 1..=ROWS {
            let _ = write!(sheet, "<row r=\"{row}\"><c r=\"A{row}\"><v>1</v></c></row>");
        }
        sheet.push_str("</sheetData><mergeCells count=\"1024\">");
        for row in 2..=ROWS {
            let _ = write!(sheet, "<mergeCell ref=\"A1:B{row}\"/>");
        }
        sheet.push_str("</mergeCells></worksheet>");
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert!(notes.is_empty(), "nothing was left out: {notes:?}");
        // The first merge holding a row is the shortest one reaching it.
        assert!(text.contains("A1: 1 (merged A1:B2)"), "{text}");
        assert!(text.contains("A2: 1 (in merged range A1:B2)"), "{text}");
        assert!(text.contains("A3: 1 (in merged range A1:B3)"), "{text}");
        assert!(
            text.contains(&format!("A{ROWS}: 1 (in merged range A1:B{ROWS})")),
            "{text}"
        );
    }

    /// A legacy `comments.xml` comment is surfaced after the sheet's cells, its
    /// author resolved through the part's `authorId` index and its rich text
    /// flattened to one line — while an `authorId` the part's authors list does not
    /// hold names an author that is in the file but not in this answer, so the
    /// comment says so rather than reading as one by nobody.
    #[test]
    fn xlsx_reads_a_legacy_cell_comment_with_its_author() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
            ),
            (
                "xl/comments1.xml",
                br#"<comments><authors><author>Ivan Petrov</author></authors><commentList><comment ref="A1" authorId="0"><text><r><t>check</t></r><r><t> this</t></r></text></comment><comment ref="B1" authorId="7"><text>no such author</text></comment></commentList></comments>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 1\n  B1: 2\n  \
             A1 comment (Ivan Petrov): check this\n  B1 comment: no such author"
        );
        assert_eq!(
            notes,
            ["sheet \"Data\": 1 comment author(s) could not be resolved"]
        );
    }

    /// A workbook whose relationships part is there but is no relationship list at
    /// all falls back to conventional part names: the styles part it finds there is
    /// read as usual, and one it does not find declares no format — nothing is
    /// claimed lost, since a format note must stand for a format a package really
    /// carries. The loss of the workbook's own index is said once either way.
    #[test]
    fn xlsx_reports_a_workbook_whose_relationship_list_is_lost() {
        // An error page in the relationships part reads as well-formed XML.
        let rels = b"<html><head><title>404</title></head><body>Not Found</body></html>".as_slice();
        let styles: &[u8] =
            br#"<styleSheet><cellXfs count="1"><xf numFmtId="14"/></cellXfs></styleSheet>"#;
        for styles in [None, Some(styles)] {
            let mut parts: Vec<(&str, &[u8])> = vec![
                (
                    "xl/workbook.xml",
                    br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                ),
                ("xl/_rels/workbook.xml.rels", rels),
                (
                    "xl/worksheets/sheet1.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1" s="0"><v>46300</v></c></row></sheetData></worksheet>"#,
                ),
            ];
            let expected = if let Some(styles) = styles {
                parts.push(("xl/styles.xml", styles));
                // The conventional part the fallback finds is read as usual.
                "Sheet \"Data\":\n  A1: 2026-10-05"
            } else {
                // No format is lost: the package carries none, so the cell shows
                // what it stores.
                "Sheet \"Data\":\n  A1: 46300"
            };
            let bytes = zip_fixture(&parts);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a readable workbook");
            };
            assert_eq!(text, expected);
            assert_eq!(notes, [WORKBOOK_RELS_NOTE]);
        }
    }

    /// A cell with no `s` is the same cell as one naming the first cell format: `s`
    /// defaults to `0`, so both show what the workbook's first format displays.
    #[test]
    fn xlsx_reads_an_s_less_cell_as_the_first_cell_format() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellXfs count="2"><xf numFmtId="14"/><xf numFmtId="9"/></cellXfs></styleSheet>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>46300</v></c><c r="B1" s="0"><v>46300</v></c><c r="C1" s="1"><v>0.25</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 2026-10-05\n  B1: 2026-10-05\n  C1: 25%"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A threaded discussion comes back whole, each entry with the name its person
    /// resolves to through the workbook's persons part. The copy a real workbook
    /// keeps in the legacy part as the discussion's stub — Excel's `[Threaded
    /// comment]` preamble, its note to the reader, and each entry's text behind a
    /// `Comment:`/`Reply:` lead of its own — is not repeated, while a legacy comment
    /// that is a comment of its own keeps its line, even when its words stand inside
    /// a threaded one, and a cell with no thread keeps its legacy comment.
    #[test]
    fn xlsx_reads_a_thread_without_repeating_its_legacy_stub() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/person" Target="persons/person.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c><c r="A2"><v>2</v></c><c r="A3"><v>3</v></c><c r="A4"><v>4</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/><Relationship Id="rId2" Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" Target="../threadedComments/threadedComment1.xml"/></Relationships>"#,
            ),
            // The legacy part as Excel itself writes it for a threaded discussion:
            // its own preamble and note to the reader, then each entry's text behind
            // a `Comment:`/`Reply:` lead of its own — the shape the stub has to be
            // recognized in, and the discussion then shown only once.
            (
                "xl/comments1.xml",
                br#"<comments><authors><author>Old Author</author></authors><commentList><comment ref="A1" authorId="0"><text>[Threaded comment]

Your version of Excel allows you to read this threaded comment; however, any edits to it will get removed if the file is opened in a newer version of Excel. Learn more: https://go.microsoft.com/fwlink/?linkid=870924

Comment:
threaded wins
Reply:
a reply</text></comment><comment ref="A2" authorId="0"><text>legacy two</text></comment><comment ref="A3" authorId="0"><text>an older note</text></comment><comment ref="A4" authorId="0"><text>wins</text></comment></commentList></comments>"#,
            ),
            (
                "xl/threadedComments/threadedComment1.xml",
                br#"<ThreadedComments><threadedComment ref="A1" personId="{P1}"><text>threaded wins</text></threadedComment><threadedComment ref="A1" personId="{P2}"><text>a reply</text></threadedComment><threadedComment ref="A3"><text>a newer thread</text></threadedComment><threadedComment ref="A4" personId="{P1}"><text>threaded wins</text></threadedComment></ThreadedComments>"#,
            ),
            (
                "xl/persons/person.xml",
                br#"<personList><person id="{P1}" displayName="Ivan Petrov"/><person id="{P2}" displayName="Maria"/></personList>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 1\n  A2: 2\n  A3: 3\n  A4: 4\n  \
             A1 comment (Ivan Petrov): threaded wins\n  A1 comment (Maria): a reply\n  \
             A2 comment (Old Author): legacy two\n  \
             A3 comment (Old Author): an older note\n  A3 comment: a newer thread\n  \
             A4 comment (Old Author): wins\n  A4 comment (Ivan Petrov): threaded wins"
        );
        // A thread naming no person loses no author, so nothing is said.
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A comment that names an author this reader cannot resolve — a threaded one
    /// whose `personId` the workbook's person names do not hold, whether the part is
    /// declared and absent, declared and no person list, or not declared at all —
    /// has an author that is in the file but not in this answer, which its sheet says
    /// rather than passing the comment off as one by nobody.
    #[test]
    fn xlsx_reports_a_comment_author_it_cannot_resolve() {
        for (declared, persons) in [
            (true, None),
            // An error page in the part's place reads as well-formed XML.
            (
                true,
                Some(
                    b"<html><head><title>404</title></head><body>Not Found</body></html>"
                        .as_slice(),
                ),
            ),
            (false, None),
        ] {
            let rels: &[u8] = if declared {
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/person" Target="persons/person.xml"/></Relationships>"#
            } else {
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#
            };
            let mut parts: Vec<(&str, &[u8])> = vec![
                (
                    "xl/workbook.xml",
                    br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                ),
                ("xl/_rels/workbook.xml.rels", rels),
                (
                    "xl/worksheets/sheet1.xml",
                    br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
                ),
                (
                    "xl/worksheets/_rels/sheet1.xml.rels",
                    br#"<Relationships><Relationship Id="rId2" Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" Target="../threadedComments/threadedComment1.xml"/></Relationships>"#,
                ),
                (
                    "xl/threadedComments/threadedComment1.xml",
                    br#"<ThreadedComments><threadedComment ref="A1" personId="{P1}"><text>threaded wins</text></threadedComment></ThreadedComments>"#,
                ),
            ];
            if let Some(persons) = persons {
                parts.push(("xl/persons/person.xml", persons));
            }
            let bytes = zip_fixture(&parts);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                panic!("expected Text outcome for a well-formed xlsx");
            };
            assert_eq!(
                text,
                "Sheet \"Data\":\n  A1: 1\n  A1 comment: threaded wins"
            );
            assert_eq!(
                notes,
                ["sheet \"Data\": 1 comment author(s) could not be resolved"]
            );
        }
    }

    /// A workbook that declares a person part it does not use — no sheet carrying a
    /// threaded comment — loses nothing when that part cannot be read, so it is not
    /// told about a loss it does not have.
    #[test]
    fn xlsx_ignores_a_persons_part_no_sheet_uses() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/person" Target="persons/person.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
            ),
            (
                "xl/comments1.xml",
                br#"<comments><authors><author>Ivan Petrov</author></authors><commentList><comment ref="A1" authorId="0"><text>legacy</text></comment></commentList></comments>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 1\n  A1 comment (Ivan Petrov): legacy"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A comment whose cell reference is not an address names no cell a line could
    /// hang off, so it is left out and counted — and reading the sheet neither
    /// panics nor prints it under an address that is not one.
    #[test]
    fn xlsx_leaves_out_a_comment_whose_cell_is_not_an_address() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
            ),
            (
                "xl/comments1.xml",
                br#"<comments><commentList><comment ref="A1"><text>kept</text></comment><comment><text>no address</text></comment><comment ref="1"><text>no column</text></comment></commentList></comments>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1\n  A1 comment: kept");
        assert_eq!(
            notes,
            [
                "sheet \"Data\": 2 comment(s) left out: the cell they name is not one this reader can resolve"
            ]
        );
    }

    /// A hyperlink marks the cell it covers with the raw target its `r:id`
    /// resolves to (never re-resolved as a part), a target written as `location`
    /// is used as-is, and a link over no valued cell gets its own line.
    #[test]
    fn xlsx_marks_hyperlinks_on_their_cells_and_alone() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/a" TargetMode="External"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Site</t></is></c><c r="B1"><v>2</v></c><c r="E1" t="inlineStr"><is><t>Report</t></is></c><c r="F1"><v>7</v></c></row></sheetData><hyperlinks><hyperlink ref="A1" r:id="rId1"/><hyperlink ref="C1" location="Sheet2!A1" display="Go"/><hyperlink ref="D2" location="Sheet2!B2"/><hyperlink ref="E1" location="Sheet2!C3" display="&#1054;&#1090;&#1095;&#1105;&#1090;"/><hyperlink ref="F1" location="Sheet2!D4" display="7"/></hyperlinks></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        // A caption is named in front of the target where it is not what the cell's
        // own value already says (E1), and left out where it is (F1). A link no cell
        // line covers — one over empty cells — follows every cell line.
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: Site (link: https://example.com/a)\n  B1: 2\n  \
             E1: Report (link: Отчёт -> Sheet2!C3)\n  F1: 7 (link: Sheet2!D4)\n  \
             C1: Go (link: Sheet2!A1)\n  D2: (link: Sheet2!B2)"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A value the workbook (or another producer) wrote with a line break in it is
    /// shown as one line, so no value can break the one-line-per-cell shape or stand
    /// where this reader's own lines do. Every break a text stack may take is
    /// collapsed — `\n` and `\r\n`, and the Unicode breaks a value may carry.
    #[test]
    fn xlsx_shows_a_value_with_a_line_break_as_one_line() {
        let bytes = zip_fixture(&[
            ("xl/workbook.xml", WORKBOOK),
            ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS),
            (
                "xl/sharedStrings.xml",
                br"<sst><si><t>two&#10;lines</t></si></sst>".as_slice(),
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>one
two</t></is></c><c r="C1" t="inlineStr"><is><t>cr&#13;&#10;lf</t></is></c><c r="D1" t="inlineStr"><is><t>nel&#x85;sep&#x2028;para</t></is></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/sheet2.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: two\\nlines\n  B1: one\\ntwo\n  C1: cr\\nlf\n  \
             D1: nel\\nsep\\npara\n\nSheet \"More\":\n  A1: 1"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A break in the text the workbook wrote for something other than a cell value
    /// — a hyperlink's caption, a comment's author, a defined name, a sheet name, a
    /// cell's formula and the display a cell's own number format produces — is shown
    /// as `\n` too, so none of them can add a line the workbook does not have: each
    /// keeps its own place in the answer's shape.
    #[test]
    fn xlsx_shows_a_break_in_a_document_text_as_one_line() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                // A sheet name and a defined name, each carrying a break, and the
                // defined name's own text carrying another.
                br#"<workbook><sheets><sheet name="Data&#10;Set" sheetId="1" r:id="rId1"/></sheets><definedNames><definedName name="Ra&#10;te" localSheetId="0">Data&#10;Set!$B$1</definedName></definedNames></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                // The format code the cell's display comes from carries a break of
                // its own; the first cell format is plain, so a cell showing it is
                // left alone.
                "xl/styles.xml",
                br#"<styleSheet><numFmts><numFmt numFmtId="164" formatCode="0&quot;a&#10;b&quot;"/></numFmts><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="164"/></cellXfs></styleSheet>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                // The formula carries a break of its own, as does the caption.
                br#"<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>x</t></is></c><c r="B1"><v>1</v><f>SUM(A1,&#10;B1)</f></c><c r="C1" s="1"><v>1</v></c></row></sheetData><hyperlinks><hyperlink ref="A1" location="Sheet2!A1" display="Cap&#10;tion"/></hyperlinks></worksheet>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
            ),
            (
                "xl/comments1.xml",
                br#"<comments><authors><author>Ivan&#10;Petrov</author></authors><commentList><comment ref="A1" authorId="0"><text>note</text></comment></commentList></comments>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\\nSet\":\n  A1: x (link: Cap\\ntion -> Sheet2!A1)\n  \
             B1: 1 (=SUM(A1,\\nB1))\n  \
             C1: 1a\\nb\n  \
             A1 comment (Ivan\\nPetrov): note\n\n\
             Defined names:\n  Ra\\nte (sheet \"Data\\nSet\"): Data\\nSet!$B$1"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A value the format this reader read has no display for is shown as the
    /// stored number under the same mark as one whose format was never read: the
    /// reader reports the one outcome either way.
    #[test]
    fn xlsx_marks_a_value_its_format_has_no_display_for() {
        let bytes = zip_fixture(&[
            ("xl/workbook.xml", WORKBOOK),
            ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS),
            (
                "xl/styles.xml",
                br#"<styleSheet><cellXfs count="2"><xf numFmtId="14"/><xf numFmtId="0"/></cellXfs></styleSheet>"#
                    .as_slice(),
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1" s="0"><v>-5</v></c><c r="B1" s="1"><v>1234.5</v></c><c r="C1" s="0"><v>1e19</v></c></row></sheetData></worksheet>"#,
            ),
            (
                "xl/worksheets/sheet2.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: -5 (stored number, format not shown)\n  \
             B1: 1234.5\n  C1: 1e19 (stored number, format not shown)\n\n\
             Sheet \"More\":\n  A1: 1900-01-01"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A hyperlink whose `r:id` the sheet does not declare is dropped rather
    /// than rendered pointing nowhere, and the loss is counted.
    #[test]
    fn xlsx_reports_links_whose_target_is_not_declared() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/a"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData><hyperlinks><hyperlink ref="A1" r:id="rId1"/><hyperlink ref="B1" r:id="rId7"/></hyperlinks></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\":\n  A1: 1 (link: https://example.com/a)"
        );
        assert_eq!(
            notes,
            [
                "sheet \"Data\": 1 link(s) left out: the target or the cell range it names is not one this reader can resolve"
            ]
        );
    }

    /// A range whose minimum is past its maximum (`C3:A1`) names no cell, so an
    /// inverted hyperlink is a link this reader cannot resolve — counted and
    /// reported like any other loss, rather than accepted as an empty rectangle.
    #[test]
    fn xlsx_reports_a_hyperlink_whose_range_is_inverted() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData><hyperlinks><hyperlink ref="C3:A1" location="Sheet2!A1"/></hyperlinks></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(text, "Sheet \"Data\":\n  A1: 1");
        assert_eq!(
            notes,
            [
                "sheet \"Data\": 1 link(s) left out: the target or the cell range it names is not one this reader can resolve"
            ]
        );
    }

    /// Defined names follow the last sheet: a workbook-level one is bare, a
    /// sheet-local one names the sheet its `localSheetId` indexes (a CDATA text
    /// reads the same as a plain one), and Excel's own `_xlnm.` names are left
    /// out.
    #[test]
    fn xlsx_surfaces_defined_names_with_their_sheet() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                br#"<workbook><sheets><sheet name="Data" r:id="rId1"/><sheet name="More" r:id="rId2"/></sheets><definedNames><definedName name="Rate"><![CDATA[Data!$B$1]]></definedName><definedName name="Tax" localSheetId="0" hidden="1">Data!$B$2</definedName><definedName name="Local" localSheetId="1">More!$A$1</definedName><definedName name="NoText"/><definedName name="_xlnm.print_area" localSheetId="0">Data!$A$1:$C$9</definedName></definedNames></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                b"<worksheet><sheetData/></worksheet>",
            ),
            (
                "xl/worksheets/sheet2.xml",
                b"<worksheet><sheetData/></worksheet>",
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Data\": (no values)\n\nSheet \"More\": (no values)\n\n\
             Defined names:\n  Rate: Data!$B$1\n  Tax (sheet \"Data\"): Data!$B$2\n  Local (sheet \"More\"): More!$A$1"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }

    /// A comments part the sheet names but that cannot be read — one that will not
    /// parse, one a package left half written, and the modern threaded part as
    /// well — costs the comments and is said out loud, naming the part it lost, so
    /// a sheet that lost both is never passed off as a sheet with no comments.
    #[test]
    fn xlsx_reports_a_comments_part_it_cannot_read() {
        let rels = [
            // The legacy part only.
            (br#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/>"#.as_slice(), "xl/comments1.xml", "its comments could not be read"),
            // The threaded part only.
            (br#"<Relationship Id="rId1" Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" Target="../threadedComments/threadedComment1.xml"/>"#.as_slice(), "xl/threadedComments/threadedComment1.xml", "its threaded comments could not be read"),
        ];
        for (relationship, part, message) in rels {
            for part_xml in [
                // A mismatched end tag.
                b"<comments><commentList></comments>".as_slice(),
                // Truncated at an element boundary: the events simply end.
                b"<comments><commentList>".as_slice(),
            ] {
                let sheet_rels = format!(
                    "<Relationships>{}</Relationships>",
                    String::from_utf8_lossy(relationship)
                );
                let bytes = zip_fixture(&[
                    (
                        "xl/workbook.xml",
                        br#"<workbook><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>"#,
                    ),
                    (
                        "xl/_rels/workbook.xml.rels",
                        br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
                    ),
                    (
                        "xl/worksheets/sheet1.xml",
                        br#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>"#,
                    ),
                    ("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels.as_bytes()),
                    (part, part_xml),
                ]);
                let dir = tempfile::tempdir().expect("tempdir");
                let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
                    panic!("expected Text outcome for a readable workbook");
                };
                assert_eq!(text, "Sheet \"Data\":\n  A1: 1", "{part}");
                assert_eq!(notes, [format!("sheet \"Data\": {message}")], "{part}");
            }
        }
    }

    /// A working book with the pieces together: a date, a percentage, a
    /// currency amount, a merged heading, a hidden service column, a comment, a
    /// link and a named range — every value shown the way Excel shows it and
    /// every piece of structure marked rather than dropped.
    #[test]
    fn xlsx_reads_a_working_book_end_to_end() {
        let bytes = zip_fixture(&[
            (
                "xl/workbook.xml",
                "<workbook><workbookPr date1904=\"0\"/><sheets>\
                 <sheet name=\"Отчёт\" sheetId=\"1\" r:id=\"rId1\"/>\
                 <sheet name=\"Служебный\" sheetId=\"2\" state=\"hidden\" r:id=\"rId2\"/>\
                 </sheets><definedNames>\
                 <definedName name=\"Ставка\">Отчёт!$B$2</definedName>\
                 <definedName name=\"_xlnm.Print_Area\" localSheetId=\"0\">Отчёт!$A$1:$C$3</definedName>\
                 <definedName name=\"Порог\" localSheetId=\"1\">Служебный!$B$1</definedName>\
                 </definedNames></workbook>"
                    .as_bytes(),
            ),
            (
                "xl/_rels/workbook.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
            ),
            (
                "xl/styles.xml",
                br##"<styleSheet><numFmts><numFmt numFmtId="164" formatCode="#,##0.00&quot; &#8381;&quot;"/></numFmts><cellXfs count="4"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="10"/><xf numFmtId="164"/></cellXfs></styleSheet>"##,
            ),
            (
                "xl/sharedStrings.xml",
                "<sst><si><t>Итого</t></si><si><t>Ставка</t></si><si><t>Сайт</t></si></sst>".as_bytes(),
            ),
            (
                "xl/worksheets/sheet1.xml",
                br#"<worksheet><cols><col min="3" max="3" hidden="1"/></cols><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" s="1"><v>46300</v></c></row><row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" s="2"><v>0.125</v></c><c r="C2" s="3"><v>1234.5</v></c></row><row r="4"><c r="A4" t="s"><v>2</v></c><c r="B4" t="d" s="1"><v>2024-02-29T13:45:07Z</v></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:B1"/></mergeCells><hyperlinks><hyperlink ref="A4" r:id="rId1"/></hyperlinks></worksheet>"#,
            ),
            (
                "xl/worksheets/_rels/sheet1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com" TargetMode="External"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/></Relationships>"#,
            ),
            (
                "xl/comments1.xml",
                "<comments><authors><author>Иван Петров</author></authors><commentList>\
                 <comment ref=\"B2\" authorId=\"0\"><text>Проверить ставку</text></comment>\
                 </commentList></comments>"
                    .as_bytes(),
            ),
            (
                "xl/worksheets/sheet2.xml",
                br#"<worksheet><sheetData><row r="1"><c r="B1"><v>1</v></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_xlsx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed xlsx");
        };
        assert_eq!(
            text,
            "Sheet \"Отчёт\":\n  (hidden columns: C)\n  A1: Итого (merged A1:B1)\n  \
             C1: 2026-10-05\n  A2: Ставка\n  B2: 12.50%\n  C2: 1,234.50 ₽\n  \
             A4: Сайт (link: https://example.com)\n  B4: 2024-02-29T13:45:07Z\n  \
             B2 comment (Иван Петров): Проверить ставку\n\n\
             Sheet \"Служебный\" (hidden):\n  B1: 1\n\n\
             Defined names:\n  Ставка: Отчёт!$B$2\n  Порог (sheet \"Служебный\"): Служебный!$B$1"
        );
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
    }
}
