//! Legacy binary Office reading: the CFB half of document conversion.
//!
//! [`crate::document`] owns content-first detection; [`crate::ooxml`] reads the
//! ZIP-container OOXML packages. This module reads the older, pre-OOXML Word,
//! Excel and PowerPoint files — the `.doc`/`.xls`/`.ppt` whose container is a
//! CFB (compound file) rather than a ZIP — through the `office_oxide` crate,
//! and renders them into the same text shapes the OOXML arms use, so a model
//! reading an old file and a new one reads both the same way: the Word
//! subdocument stories are labelled blocks, one per story and in the order a
//! `.docx` reading prints them, and every value or name is kept to one line.
//! Each arm also fills the shared report of what a reading did not show
//! ([`crate::reader_output::Unshown`]): its pictures counted by kind — a vector
//! metafile as a drawing, every other BLIP as an image — a `.ppt`'s objects
//! counted by what they are (a video or a sound, otherwise an embedded object; an
//! OLE object the deck only links is counted not at all, since the file holds the
//! link and not what it points at), and each `.xls` chart sheet, so no image,
//! chart or embedded object the crate hands over vanishes silently; the `.xls`
//! arm names the chart text its reader recovers in a note of its own. A media
//! object is named by its kind whether the deck holds it or links it: the reader
//! tells the two apart for OLE objects alone, and the file holds the object's own
//! record either way. What the arms deliberately do not adopt
//! is the newer arms' vocabulary: hidden geometry, hidden sheets, hidden slides,
//! merged ranges, cell comments, hyperlinks, defined names, per-note splitting and
//! formula text stay out, as does any mark the reader gives no way to place.
//!
//! # Invariants
//!
//! - **No panics of its own.** The whole parse runs inside
//!   [`crate::shutdown::contain_panics`], and every fallible step degrades to
//!   [`DocOutcome::Unreadable`] or a note. A panic inside the readers costs the
//!   document and is reported as corruption; the containment suppresses the
//!   panic hook, so no panic report reaches the product's logs.
//! - **Encrypted, unsupported and unreadable are different outcomes.** A file
//!   that needs a password is [`DocOutcome::Unreadable`] with `reason`
//!   `"password-protected"` — the same verdict the OOXML arms and the PDF arm
//!   give. A legacy *version* the reader recognizes but cannot read keeps its
//!   own reason rather than being called corrupt. Neither is ever
//!   `Unsupported` (which the callers read as "fall back to the raw bytes");
//!   everything else unreadable is reported as corrupt, never as a format this
//!   project does not convert.
//! - **Reading only.** Nothing here writes a file, touches channel state, the
//!   database or the network; the old formats are read, never edited.

use crate::document::DocOutcome;
use crate::reader_output::{
    DOC_COMMENTS, DOC_ENDNOTES, DOC_FOOTNOTES, DOC_HEADERS_FOOTERS, NO_VALUES, TEXT_BOX, Unshown,
    UnshownKind, column_letters, sheet_header, slide_header, slide_notes_header, text_block,
    text_lines,
};
use crate::util::one_line;
use office_oxide::cfb::blip::{BlipFormat, BlipImage};
use office_oxide::doc::{DocDocument, DocError, SubDocumentKind};
use office_oxide::ir::SheetKind;
use office_oxide::ppt::{OleObjectInfo, PptDocument, PptError, SlideText, TextType};
use office_oxide::xls::{CellValue, Sheet, XlsDocument, XlsError};
use std::io::Cursor;
use std::path::Path;

/// Extensions accepted as legacy Word documents.
const DOC_EXTENSIONS: &[&str] = &["doc"];
/// Extensions accepted as legacy Excel workbooks.
const XLS_EXTENSIONS: &[&str] = &["xls"];
/// Extensions accepted as legacy PowerPoint presentations.
const PPT_EXTENSIONS: &[&str] = &["ppt"];

/// `OleObjectInfo::kind` of an OLE object a deck links rather than holds: the
/// file keeps the link, and the content it points at is somewhere else, so the
/// report names nothing for it. The media kinds carry no such distinction, so a
/// linked sound or video is named like an embedded one.
const OLE_LINKED: u32 = 1;

/// The legacy family a file name belongs to. The CFB magic is shared by all
/// three (and by an encrypted OOXML package), so only the name can tell them
/// apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Doc,
    Xls,
    Ppt,
}

impl Family {
    /// The family's name — the spelling the corruption reason uses.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Doc => "doc",
            Self::Xls => "xls",
            Self::Ppt => "ppt",
        }
    }
}

/// `path`'s legacy family, or `None` when its name is not one of them. The
/// extension lists above are the only place the families' extensions live.
#[must_use]
pub(crate) fn family_of(path: &Path) -> Option<Family> {
    if crate::util::has_extension(path, DOC_EXTENSIONS) {
        Some(Family::Doc)
    } else if crate::util::has_extension(path, XLS_EXTENSIONS) {
        Some(Family::Xls)
    } else if crate::util::has_extension(path, PPT_EXTENSIONS) {
        Some(Family::Ppt)
    } else {
        None
    }
}

/// Read `bytes` as the legacy format `family`, into the shapes the OOXML arms
/// use. The whole parse is contained: a panic costs the document, not the
/// caller's turn. The containment is [`crate::shutdown::contain_panics`] rather
/// than a bare `catch_unwind`, so a panic inside the `office_oxide` readers on
/// an ordinary user file does not print a panic report and backtrace into the
/// product's logs while the caller is told the file is merely corrupt.
#[must_use]
pub(crate) fn convert(bytes: &[u8], family: Family) -> DocOutcome {
    match crate::shutdown::contain_panics(|| parse(bytes, family)) {
        Ok(outcome) => outcome,
        Err(_) => corrupt(family),
    }
}

fn parse(bytes: &[u8], family: Family) -> DocOutcome {
    match family {
        Family::Doc => convert_doc(bytes),
        Family::Xls => convert_xls(bytes),
        Family::Ppt => convert_ppt(bytes),
    }
}

/// A `.doc` reading: the body, then each subdocument story the document has as
/// a labelled block in the order a `.docx` reading prints them — headers and
/// footers, footnotes, endnotes, comments — then the text boxes and finally the
/// names of the embedded objects. The reader hands the legacy arms a whole story
/// at once, so each story is one plural, unnumbered block: the per-definition
/// numbering and the inline `[text box]` placement of the `.docx` reading have
/// no counterpart in what this reader exposes. The body and every story line
/// follow the one-line-per-unit rule ([`paragraph_lines`]).
fn convert_doc(bytes: &[u8]) -> DocOutcome {
    match DocDocument::from_reader(Cursor::new(bytes)) {
        Ok(doc) => {
            let mut sections: Vec<String> = Vec::new();
            let body = paragraph_lines(doc.plain_text_ref()).join("\n");
            if !body.is_empty() {
                sections.push(body);
            }
            for (kind, label) in [
                (SubDocumentKind::HeadersFooters, DOC_HEADERS_FOOTERS),
                (SubDocumentKind::Footnotes, DOC_FOOTNOTES),
                (SubDocumentKind::Endnotes, DOC_ENDNOTES),
                (SubDocumentKind::Comments, DOC_COMMENTS),
            ] {
                let lines = doc_story_lines(&doc, kind);
                if !lines.is_empty() {
                    sections.push(text_lines(&format!("{label}:"), &lines));
                }
            }
            // The `.docx` reader prints each text box where it is anchored; this
            // reader can only append, so the boxes come last, main-document ones
            // before the header document's.
            for kind in [SubDocumentKind::TextBoxes, SubDocumentKind::HeaderTextBoxes] {
                let lines = doc_story_lines(&doc, kind);
                if !lines.is_empty() {
                    sections.push(text_lines(TEXT_BOX, &lines));
                }
            }
            // The reader names the embedded objects only in `plain_text()`; taken
            // off that string they are the one thing telling a model an object is
            // there, so they are the last section.
            if let Some(rest) = doc
                .plain_text()
                .strip_prefix(&doc_plain_text_before_objects(&doc))
            {
                let lines = paragraph_lines(rest);
                if !lines.is_empty() {
                    sections.push(lines.join("\n"));
                }
            }
            let mut notes = Vec::new();
            if !doc.text_complete() {
                notes.push(
                    "the reader stopped early — some of the document's text may be missing"
                        .to_string(),
                );
            }
            // A `.doc` reads for its text only; the object names above are the
            // reader's own, so only its pictures are left to report.
            let mut unshown = Unshown::default();
            count_pictures(&mut unshown, doc.images());
            DocOutcome::Text {
                text: sections.join("\n").trim_end().to_string(),
                images: Vec::new(),
                notes,
                unshown,
                all_page_text_lost: false,
            }
        }
        Err(err) => doc_error(&err),
    }
}

/// One subdocument kind's story, one line per paragraph. Empty when the
/// document has no story of that kind, or its story holds nothing.
fn doc_story_lines(doc: &DocDocument, kind: SubDocumentKind) -> Vec<String> {
    doc.subdocuments()
        .iter()
        .find(|sub| sub.kind == kind)
        .map(|sub| paragraph_lines(&sub.text))
        .unwrap_or_default()
}

/// The text `plain_text()` assembles before it appends the embedded-object
/// names: the main text, then every non-empty subdocument story trimmed, in
/// `subdocuments()` order, each on its own line. The reader's `ole_objects` API
/// is crate-private, so mirroring its own assembly is the only way to reach the
/// `[<description>]` lines it appends past this prefix; a reader change that
/// makes the mirror no longer match yields no names rather than a wrong tail,
/// and the crate is pinned exactly, so only a deliberate version bump can raise
/// that reading of it.
fn doc_plain_text_before_objects(doc: &DocDocument) -> String {
    let mut out = doc.plain_text_ref().to_string();
    for sub in doc.subdocuments() {
        let text = sub.text.trim();
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(text);
        out.push('\n');
    }
    out
}

/// Every sheet in workbook order, rendered in the shape
/// [`crate::ooxml::convert_xlsx`] gives a `.xlsx` sheet: a `Sheet "<name>":`
/// block and one indented `{address}: {value}` line per valued cell, with the
/// value the way Excel displays it. Hidden geometry, merged ranges, cell comments,
/// hyperlinks and defined names are marked by the `.xlsx` arm alone. A cell whose
/// formula the reader cannot recover shows the value Excel cached for it.
fn convert_xls(bytes: &[u8]) -> DocOutcome {
    match XlsDocument::from_reader(Cursor::new(bytes)) {
        Ok(doc) => {
            let mut blocks = Vec::new();
            for sheet in &doc.sheets {
                blocks.push(text_block(
                    &sheet_header(&sheet.name, None),
                    NO_VALUES,
                    &sheet_lines(sheet),
                ));
            }
            let mut notes = Vec::new();
            if doc.truncated() {
                notes.push(
                    "the workbook was cut short — later sheets or cells may be missing".to_string(),
                );
            }
            // The reader recovers a chart's own text — a chart sheet's content
            // (`Sheet::chart_text`) or an embedded chart's series names, trendline
            // names and labels, axis titles and chart titles
            // (`XlsDocument::chart_text`) — which this reading does not print: a
            // legacy workbook is read for its cells. Named rather than passed over,
            // the way the `.xlsx` arm names the charts it cannot extract.
            if !doc.chart_text().is_empty()
                || doc.sheets.iter().any(|sheet| !sheet.chart_text.is_empty())
            {
                notes.push(
                    "the workbook holds text from its charts (chart titles, series and trendline names, and axis titles), which is not extracted".to_string(),
                );
            }
            // The report holds the rest of the loss: the pictures by kind, and each
            // chart sheet as the kind of sheet it is — a sheet the workbook prints
            // as one with no valued cell, whose whole content is the chart. The cell
            // comments, hyperlinks and defined names the reader also hands over are
            // not printed for an old format by choice (see the module docs).
            let mut unshown = Unshown::default();
            count_pictures(&mut unshown, doc.images());
            unshown.add(
                UnshownKind::ChartSheet,
                doc.sheets
                    .iter()
                    .filter(|sheet| sheet.kind == SheetKind::Chart)
                    .count(),
            );
            DocOutcome::Text {
                text: blocks.join("\n\n").trim_end().to_string(),
                images: Vec::new(),
                notes,
                unshown,
                all_page_text_lost: false,
            }
        }
        Err(err) => xls_error(&err),
    }
}

/// The valued cells of one sheet, in row then column order. The grid is
/// jagged (a row holds cells only up to its last non-empty one), so a missing
/// position is simply not iterated; a cell that is `Empty`, or whose displayed
/// text is blank, contributes no line. A value whose own text holds a line break
/// is shown as one line ([`one_line`]), so no value can break the
/// one-line-per-cell shape.
fn sheet_lines(sheet: &Sheet) -> Vec<String> {
    let mut lines = Vec::new();
    for (row, cells) in sheet.rows.iter().enumerate() {
        for (col, value) in cells.iter().enumerate() {
            if matches!(value, CellValue::Empty) {
                continue;
            }
            let Some(display) = sheet.display_text(row, col) else {
                continue;
            };
            let display = one_line(display.trim().to_owned());
            if display.is_empty() {
                continue;
            }
            // A BIFF8 sheet has at most 256 columns, so this never actually
            // exceeds `u32`; a grid that did would be corrupt rather than
            // worth a panic.
            let Ok(column) = u32::try_from(col) else {
                continue;
            };
            let address = format!("{}{}", column_letters(column), row + 1);
            lines.push(format!("{address}: {display}"));
        }
    }
    lines
}

/// Each slide's text plus its speaker notes, in the shapes
/// [`crate::ooxml::convert_pptx`] renders — without its marks: a title run
/// ([`TextType::Title`]/[`TextType::CenterTitle`]) reads here as ordinary text
/// and a hidden slide ([`SlideText::hidden`]) is not flagged, though the IR
/// holds both, while a SmartArt diagram's text has no counterpart in it at all.
fn convert_ppt(bytes: &[u8]) -> DocOutcome {
    match PptDocument::from_reader(Cursor::new(bytes)) {
        Ok(doc) => {
            let mut blocks = Vec::new();
            for (index, slide) in doc.slides.iter().enumerate() {
                let number = index + 1;
                blocks.push(text_block(
                    &slide_header(number, false),
                    &crate::docgen::ppt_marks().no_text,
                    &slide_lines(slide),
                ));
                let notes: Vec<String> = slide
                    .text_runs
                    .iter()
                    .filter(|run| run.text_type == TextType::Notes)
                    .flat_map(|run| paragraph_lines(&run.text))
                    .collect();
                if !notes.is_empty() {
                    blocks.push(text_lines(&slide_notes_header(number), &notes));
                }
            }
            let mut notes = Vec::new();
            if !doc.text_complete() {
                notes.push(
                    "the reader stopped early — some of the presentation's text may be missing"
                        .to_string(),
                );
            }
            // The deck's pictures by kind, plus the objects the reader resolves
            // per slide, counted by what the object is: a media object is named as
            // the video or the sound it holds, an embedded object or an ActiveX
            // control as an object the reading does not open. An OLE object the
            // deck only links is skipped: the file holds the link, not the content
            // it points at.
            let mut unshown = Unshown::default();
            count_pictures(&mut unshown, doc.images());
            for object in doc.slides.iter().flat_map(|slide| &slide.ole_object_refs) {
                if object.kind == OLE_LINKED {
                    continue;
                }
                unshown.add(
                    match object.kind {
                        OleObjectInfo::KIND_MEDIA_VIDEO => UnshownKind::Video,
                        OleObjectInfo::KIND_MEDIA_AUDIO => UnshownKind::Audio,
                        _ => UnshownKind::Object,
                    },
                    1,
                );
            }
            DocOutcome::Text {
                text: blocks.join("\n\n").trim_end().to_string(),
                images: Vec::new(),
                notes,
                unshown,
                all_page_text_lost: false,
            }
        }
        Err(err) => ppt_error(&err),
    }
}

/// The report kind one of an old arm's pictures belongs to: a vector metafile
/// (EMF, WMF, PICT) is a drawing the reading does not render, every other BLIP
/// an embedded image. Both are named rather than extracted, since an old format
/// is read for its text alone.
fn picture_kind(format: BlipFormat) -> UnshownKind {
    match format {
        BlipFormat::Emf | BlipFormat::Wmf | BlipFormat::Pict => UnshownKind::Drawing,
        _ => UnshownKind::Image,
    }
}

/// Note every picture in `images` by its kind — the picture loss all three arms
/// share. The count comes from the reader's lazy list: the crate extracts the
/// images on the first `images()` call, so naming the loss is what costs that
/// read.
fn count_pictures(unshown: &mut Unshown, images: &[BlipImage]) {
    for image in images {
        unshown.add(picture_kind(image.format), 1);
    }
}

/// A slide's own text lines — every run that is not speaker notes, in shape
/// order — followed by its reconstructed tables' cell text. The reader's own
/// IR keeps a table after the slide's ordinary text, cells in row-major order;
/// each cell collapses to one line here (its runs' paragraphs joined by a
/// space, the reader's within-cell joining) so a table reads as further lines
/// of the same slide.
fn slide_lines(slide: &SlideText) -> Vec<String> {
    let mut lines: Vec<String> = slide
        .text_runs
        .iter()
        .filter(|run| run.text_type != TextType::Notes)
        .flat_map(|run| paragraph_lines(&run.text))
        .collect();
    for table in &slide.tables {
        for cell in table.rows.iter().flatten() {
            let text = cell
                .iter()
                .flat_map(|run| paragraph_lines(&run.text))
                .collect::<Vec<_>>()
                .join(" ");
            if !text.is_empty() {
                lines.push(text);
            }
        }
    }
    lines
}

/// Split a text the binary readers hand over at the paragraph marks they keep
/// inside it (`\r`, `\u{0B}` and `\n` are each a line break) and drop the
/// blanks, so one line is one paragraph, the way the OOXML readers drop an
/// empty `<a:p>`/`<w:p>`.
fn paragraph_lines(text: &str) -> Vec<String> {
    text.split(['\r', '\u{0B}', '\n'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The password-protected verdict, shared by all three families and with the
/// OOXML/PDF arms.
fn encrypted() -> DocOutcome {
    DocOutcome::Unreadable {
        reason: "password-protected".to_string(),
    }
}

/// The verdict for an intact legacy file in a version the reader recognizes but
/// cannot read — its own reason, so an old format is never called corrupt. No
/// `office_oxide` 0.1.13 error carries it (see the mappers below), so this is
/// the guard for the reader version that starts doing so.
fn unsupported() -> DocOutcome {
    DocOutcome::Unreadable {
        reason: "an old Office version this tool cannot read".to_string(),
    }
}

/// A `.doc` parse error as a verdict: encryption and an unsupported version keep
/// their own reasons; everything else is corruption.
fn doc_error(err: &DocError) -> DocOutcome {
    match err {
        DocError::Encrypted => encrypted(),
        DocError::UnsupportedVersion(_) => unsupported(),
        _ => corrupt(Family::Doc),
    }
}

/// An `.xls` parse error as a verdict, mapped the same way as [`doc_error`].
fn xls_error(err: &XlsError) -> DocOutcome {
    match err {
        XlsError::Encrypted => encrypted(),
        XlsError::UnsupportedVersion(_) => unsupported(),
        _ => corrupt(Family::Xls),
    }
}

/// A `.ppt` parse error as a verdict. The reader has no
/// "recognized but unreadable version" variant, so encryption is the only
/// reason that is not corruption; mapped like [`doc_error`] otherwise.
fn ppt_error(err: &PptError) -> DocOutcome {
    match err {
        PptError::Encrypted => encrypted(),
        _ => corrupt(Family::Ppt),
    }
}

/// A legacy file that is neither encrypted, unsupported nor readable:
/// corruption or a truncated container. Named as corrupt rather than
/// unsupported, since the name said it was a format this converter accepts.
fn corrupt(family: Family) -> DocOutcome {
    DocOutcome::Unreadable {
        reason: format!("corrupt or unreadable .{}", family.name()),
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the hand-built fixtures' stream offsets and counts fit the CFB/BIFF/PPT record fields by construction"
)]
mod tests {
    use super::*;

    // ── CFB container fixture ───────────────────────────────────────────

    // The legacy files are CFB (compound file) containers. A v3 CFB is a
    // 512-byte header, as many 512-byte directory sectors as the entries need,
    // a FAT sector and then the streams' own sectors. These fixtures are built
    // in code so no binary file is committed and nothing is downloaded at test
    // time.
    //
    // The mini-stream path is avoided structurally: the root entry declares no
    // stream of its own (`start = END_OF_CHAIN`, `size = 0`), so the reader
    // finds no mini stream and reads every stream from its regular sectors —
    // which is why a stream smaller than the 4096-byte cutoff is still read
    // whole. The recipes follow the reader's own test fixtures.

    const END_OF_CHAIN: u32 = 0xFFFF_FFFE;
    const FAT_SECT: u32 = 0xFFFF_FFFD;
    const FREE_SECT: u32 = 0xFFFF_FFFF;
    const NO_ENTRY: u32 = 0xFFFF_FFFF;

    /// Assemble a minimal v3 CFB from root-level streams, however many entries
    /// the directory needs: the root plus one entry per stream fill as many
    /// 512-byte directory sectors as it takes, chained through the FAT like
    /// every other chain. The header's directory count stays 0 (a v3 reader
    /// derives the span from the chain) and its FAT-sector count stays 1.
    fn cfb(streams: &[(&str, &[u8])]) -> Vec<u8> {
        // Sector plan: directory sectors (four entries each), the FAT sector,
        // then each stream in order.
        let dir_sectors = (streams.len() + 1).div_ceil(4) as u32;
        let fat_sector = dir_sectors;
        let mut starts = Vec::new();
        let mut runs = Vec::new();
        let mut next = fat_sector + 1;
        for (_, data) in streams {
            let sectors = data.len().div_ceil(512).max(1) as u32;
            starts.push(next);
            runs.push(sectors);
            next += sectors;
        }
        let total_sectors = next as usize;
        assert!(total_sectors <= 128, "one FAT sector covers 128 sectors");

        let mut file = vec![0u8; 512 * (1 + total_sectors)];
        file[0..8].copy_from_slice(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]);
        file[0x18..0x1A].copy_from_slice(&0x003Eu16.to_le_bytes());
        file[0x1A..0x1C].copy_from_slice(&3u16.to_le_bytes());
        file[0x1C..0x1E].copy_from_slice(&0xFFFEu16.to_le_bytes());
        file[0x1E..0x20].copy_from_slice(&9u16.to_le_bytes());
        file[0x20..0x22].copy_from_slice(&6u16.to_le_bytes());
        file[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes()); // one FAT sector
        file[0x30..0x34].copy_from_slice(&0u32.to_le_bytes()); // directory at sector 0
        file[0x38..0x3C].copy_from_slice(&4096u32.to_le_bytes());
        file[0x3C..0x40].copy_from_slice(&END_OF_CHAIN.to_le_bytes());
        file[0x44..0x48].copy_from_slice(&END_OF_CHAIN.to_le_bytes());
        file[0x4C..0x50].copy_from_slice(&fat_sector.to_le_bytes()); // DIFAT[0] = FAT sector
        for i in 1..109 {
            let off = 0x4C + i * 4;
            file[off..off + 4].copy_from_slice(&FREE_SECT.to_le_bytes());
        }

        // Directory: root (entry 0) then one entry per stream, chained right.
        let dir = 512;
        let root_child = if streams.is_empty() { NO_ENTRY } else { 1 };
        write_entry(
            &mut file[dir..dir + 128],
            "Root Entry",
            5,
            root_child,
            END_OF_CHAIN,
            0,
        );
        for (index, (name, data)) in streams.iter().enumerate() {
            let right = if index + 1 < streams.len() {
                (index as u32) + 2
            } else {
                NO_ENTRY
            };
            let at = dir + (index + 1) * 128;
            write_entry(
                &mut file[at..at + 128],
                name,
                2,
                NO_ENTRY,
                starts[index],
                data.len() as u32,
            );
            file[at + 0x48..at + 0x4C].copy_from_slice(&right.to_le_bytes());
            let offset = 512 * (starts[index] as usize + 1);
            file[offset..offset + data.len()].copy_from_slice(data);
        }

        // FAT: the directory's chain, the FAT sector, then each stream's chain.
        let fat = 512 * (fat_sector as usize + 1);
        let mut entries = vec![FREE_SECT; 128];
        for k in 0..dir_sectors {
            entries[k as usize] = if k + 1 == dir_sectors {
                END_OF_CHAIN
            } else {
                k + 1
            };
        }
        entries[fat_sector as usize] = FAT_SECT;
        for (index, &start) in starts.iter().enumerate() {
            for k in 0..runs[index] {
                entries[(start + k) as usize] = if k + 1 == runs[index] {
                    END_OF_CHAIN
                } else {
                    start + k + 1
                };
            }
        }
        for (index, value) in entries.iter().enumerate() {
            file[fat + index * 4..fat + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        file
    }

    /// Write one 128-byte CFB directory entry.
    fn write_entry(buf: &mut [u8], name: &str, kind: u8, child: u32, start: u32, size: u32) {
        for (i, unit) in name.encode_utf16().enumerate() {
            buf[i * 2..i * 2 + 2].copy_from_slice(&unit.to_le_bytes());
        }
        let name_size = ((name.encode_utf16().count() + 1) * 2) as u16;
        buf[0x40..0x42].copy_from_slice(&name_size.to_le_bytes());
        buf[0x42] = kind;
        buf[0x43] = 1;
        buf[0x44..0x48].copy_from_slice(&NO_ENTRY.to_le_bytes());
        buf[0x48..0x4C].copy_from_slice(&NO_ENTRY.to_le_bytes());
        buf[0x4C..0x50].copy_from_slice(&child.to_le_bytes());
        buf[0x74..0x78].copy_from_slice(&start.to_le_bytes());
        buf[0x78..0x7C].copy_from_slice(&size.to_le_bytes());
    }

    /// One `OfficeArtBlipJPEG` record holding a tiny JPEG: record type 0xF01D
    /// ([MS-ODRAW] §2.2.27), `rh.recInstance` 0x46A, then the 17-byte header
    /// the reader's `uid_size` computes for it (16-byte UID + 1 tag byte; a
    /// bitmap BLIP has no `metafile_header_size`) and a payload starting
    /// `\xff\xd8`. The bytes mirror `office_oxide`'s own `make_blip_in_data`
    /// fixture (`doc/images.rs`) record for record: the crate pads that record
    /// with junk on both sides to imitate a mixed `.doc` `Data` stream, which
    /// is left off here because the `Pictures` walk reads from the stream's
    /// first record, while the `.doc`/`.xls` scans search byte by byte and
    /// find the record wherever it sits. The crate is pinned exactly, so a
    /// reader change to this shape would be a deliberate version bump.
    fn jpeg_blip() -> Vec<u8> {
        let img: &[u8] = b"\xff\xd8\xff\xe0JFIF";
        let rec_type: u16 = 0xF01D;
        let inst: u16 = 0x46A;
        let uid_sz = 17; // no secondary UID: bit 0 of `inst` is clear
        let rec_len = uid_sz + img.len();
        let mut buf = (inst << 4).to_le_bytes().to_vec();
        buf.extend_from_slice(&rec_type.to_le_bytes());
        buf.extend_from_slice(&(rec_len as u32).to_le_bytes());
        buf.extend(vec![0u8; uid_sz]);
        buf.extend_from_slice(img);
        buf
    }

    /// The report line an arm gives one embedded image, and one embedded
    /// object — pinned as literals so a wording drift in
    /// [`crate::reader_output::Unshown`] fails a test instead of sliding
    /// through.
    const UNSHOWN_IMAGE: &str = "1 embedded image(s) not shown";
    const UNSHOWN_OBJECT: &str = "1 embedded object part(s) not shown";

    // ── Word (.doc) ─────────────────────────────────────────────────────

    /// Offset the fixtures put the Word text at, past every FIB field.
    const DOC_TEXT_FC: u32 = 0x400;

    /// A Word 97 FIB ([MS-DOC] §2.5.1): `wIdent` 0xA5EC, the 1Table flag, the
    /// `ccp*` lengths, and the CLX pointer, with the `fEncrypted` flag for the
    /// encrypted fixture. `ccp` is the eight `FibRgLw97` lengths in the fixed
    /// `[MS-DOC]` order — text, footnotes, headers, the ignored macro length,
    /// comments, endnotes, text boxes and header text boxes. Every other
    /// FibRgFcLcb97 field stays 0.
    fn doc_fib(ccp: &[u32; 8], fc_clx: u32, lcb_clx: u32, encrypted: bool) -> Vec<u8> {
        let mut fib = vec![0u8; 1024];
        fib[0..2].copy_from_slice(&0xA5ECu16.to_le_bytes());
        fib[2..4].copy_from_slice(&0x00C1u16.to_le_bytes());
        let flags = (1u16 << 9) | (u16::from(encrypted) << 8);
        fib[0x0A..0x0C].copy_from_slice(&flags.to_le_bytes());
        for (index, length) in ccp.iter().enumerate() {
            let at = 0x4C + index * 4;
            fib[at..at + 4].copy_from_slice(&length.to_le_bytes());
        }
        fib[0x01A2..0x01A6].copy_from_slice(&fc_clx.to_le_bytes());
        fib[0x01A6..0x01AA].copy_from_slice(&lcb_clx.to_le_bytes());
        fib
    }

    /// A CLX whose Pcdt holds one Unicode piece per `(fc, char_count)` in order,
    /// the CP ranges running consecutively from 0 — the character space the
    /// `ccp*` lengths address.
    fn doc_clx(pieces: &[(u32, u32)]) -> Vec<u8> {
        let mut cps = Vec::new();
        let mut pcds = Vec::new();
        let mut cp = 0u32;
        for (fc, char_count) in pieces {
            cps.extend_from_slice(&cp.to_le_bytes());
            pcds.extend_from_slice(&0u16.to_le_bytes()); // PCD unused
            pcds.extend_from_slice(&fc.to_le_bytes()); // Unicode: fc used directly
            pcds.extend_from_slice(&0u16.to_le_bytes()); // prm
            cp += char_count;
        }
        cps.extend_from_slice(&cp.to_le_bytes());
        cps.extend_from_slice(&pcds);
        let mut clx = vec![0x02u8]; // Pcdt
        clx.extend_from_slice(&(cps.len() as u32).to_le_bytes());
        clx.extend_from_slice(&cps);
        clx
    }

    /// The subdocument stories with their `FibRgLw97` slot in the fixed
    /// `[MS-DOC]` order the reader walks (`text` is slot 0; slot 3 is the
    /// ignored macro length, which stays zero). The stories fill slots
    /// `0x50`/`0x54`/`0x5C`/`0x60`/`0x64`/`0x68` and the piece table's character
    /// space in the same order.
    const DOC_STORY_SLOTS: [(SubDocumentKind, usize); 6] = [
        (SubDocumentKind::Footnotes, 1),
        (SubDocumentKind::HeadersFooters, 2),
        (SubDocumentKind::Comments, 4),
        (SubDocumentKind::Endnotes, 5),
        (SubDocumentKind::TextBoxes, 6),
        (SubDocumentKind::HeaderTextBoxes, 7),
    ];

    /// The `(WordDocument, 1Table)` streams of a `.doc` whose main text is
    /// `body`, with the non-empty `(kind, text)` entries of `stories` as
    /// subdocuments. Each story is one UTF-16LE piece, laid out after the body
    /// in the fixed `[MS-DOC]` story order its `ccp*` length addresses. `body`
    /// is one piece at [`DOC_TEXT_FC`].
    fn doc_streams(
        body: &str,
        stories: &[(SubDocumentKind, &str)],
        encrypted: bool,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut word = vec![0u8; DOC_TEXT_FC as usize];
        let mut ccp = [0u32; 8]; // text, ftn, hdd, mcr, atn, edn, txbx, hdrtxbx
        let mut pieces = Vec::new();
        let append = |word: &mut Vec<u8>, text: &str| {
            let fc = word.len() as u32;
            let count = text.encode_utf16().count() as u32;
            word.extend(text.encode_utf16().flat_map(u16::to_le_bytes));
            (fc, count)
        };
        let (body_fc, body_len) = append(&mut word, body);
        pieces.push((body_fc, body_len));
        ccp[0] = body_len;
        for (kind, slot) in DOC_STORY_SLOTS {
            let text = stories
                .iter()
                .find(|(story_kind, _)| *story_kind == kind)
                .map_or("", |(_, text)| *text);
            let (fc, count) = append(&mut word, text);
            ccp[slot] = count;
            if count > 0 {
                pieces.push((fc, count));
            }
        }
        let clx = doc_clx(&pieces);
        word[..1024].copy_from_slice(&doc_fib(&ccp, 0, clx.len() as u32, encrypted));
        (word, clx)
    }

    /// A `.doc` package from [`doc_streams`].
    fn doc_fixture(body: &str, stories: &[(SubDocumentKind, &str)], encrypted: bool) -> Vec<u8> {
        let (word, clx) = doc_streams(body, stories, encrypted);
        cfb(&[("WordDocument", &word), ("1Table", &clx)])
    }

    /// [`doc_fixture`]'s package plus a third root stream `Data` holding one
    /// JPEG BLIP — where Word 97 keeps a picture, which the reader finds by
    /// scanning that stream.
    fn doc_image_fixture(body: &str) -> Vec<u8> {
        let (word, clx) = doc_streams(body, &[], false);
        let data = jpeg_blip();
        cfb(&[("WordDocument", &word), ("1Table", &clx), ("Data", &data)])
    }

    #[test]
    fn doc_fixture_renders_its_text() {
        let bytes = doc_fixture("Hello legacy doc", &[], false);
        let DocOutcome::Text {
            text,
            images,
            notes,
            ..
        } = convert(&bytes, Family::Doc)
        else {
            panic!("expected text");
        };
        assert_eq!(text, "Hello legacy doc");
        assert!(images.is_empty());
        assert!(notes.is_empty());
    }

    #[test]
    fn doc_embedded_image_is_reported_as_not_shown() {
        // A `.doc` reads for its text only; the picture in its `Data` stream
        // is not extracted, so the reading reports it.
        let bytes = doc_image_fixture("Body text");
        let DocOutcome::Text {
            text,
            images,
            notes,
            unshown,
            ..
        } = convert(&bytes, Family::Doc)
        else {
            panic!("expected text");
        };
        assert_eq!(text, "Body text");
        assert!(images.is_empty());
        assert!(notes.is_empty());
        assert_eq!(unshown.lines(), [UNSHOWN_IMAGE]);
    }

    #[test]
    fn doc_object_names_mirror_stays_a_prefix_of_the_readers_own_text() {
        // The crate exposes its embedded-object names only in `plain_text()`, so
        // the assembly before them is mirrored (`doc_plain_text_before_objects`).
        // No fixture here carries an object pool, so what is pinned is that the
        // mirror still reproduces the reader's own assembly on a document with
        // stories: a reader change that moved a separator would stop the mirror
        // being a prefix and fail here rather than silently dropping the names.
        let bytes = doc_fixture(
            "Body text",
            &[
                (SubDocumentKind::Footnotes, "A footnote"),
                (SubDocumentKind::HeadersFooters, "A header\rA footer"),
            ],
            false,
        );
        let doc = DocDocument::from_reader(Cursor::new(&bytes)).expect("the fixture is read");
        let plain = doc.plain_text();
        assert!(
            plain.starts_with(&doc_plain_text_before_objects(&doc)),
            "the mirrored prefix must be a prefix of {plain:?}"
        );
    }

    #[test]
    fn doc_subdocument_stories_render_as_labelled_blocks() {
        let bytes = doc_fixture(
            "Body text",
            &[
                (SubDocumentKind::Footnotes, "A footnote"),
                (SubDocumentKind::HeadersFooters, "A header\rA footer"),
                (SubDocumentKind::Comments, "A comment"),
                (SubDocumentKind::Endnotes, "An endnote"),
            ],
            false,
        );
        let DocOutcome::Text { text, notes, .. } = convert(&bytes, Family::Doc) else {
            panic!("expected text");
        };
        assert_eq!(
            text,
            "Body text\n\
             Headers and footers:\n  A header\n  A footer\n\
             Footnotes:\n  A footnote\n\
             Endnotes:\n  An endnote\n\
             Comments:\n  A comment"
        );
        assert!(notes.is_empty());
    }

    #[test]
    fn doc_empty_story_prints_no_block() {
        // A story whose extracted text trims to nothing is not a subdocument to
        // the reader at all, so its label gets no empty block.
        let bytes = doc_fixture(
            "Body only",
            &[
                (SubDocumentKind::Footnotes, " \r"),
                (SubDocumentKind::Comments, "A comment"),
            ],
            false,
        );
        let DocOutcome::Text { text, .. } = convert(&bytes, Family::Doc) else {
            panic!("expected text");
        };
        assert_eq!(text, "Body only\nComments:\n  A comment");
    }

    #[test]
    fn encrypted_doc_reports_password_protected() {
        let bytes = doc_fixture("secret", &[], true);
        assert!(matches!(
            convert(&bytes, Family::Doc),
            DocOutcome::Unreadable { reason } if reason == "password-protected"
        ));
    }

    // ── Excel (.xls) ────────────────────────────────────────────────────

    const RT_BOF: u16 = 0x0809;
    const RT_EOF: u16 = 0x000A;
    const RT_BOUNDSHEET: u16 = 0x0085;
    const RT_SST: u16 = 0x00FC;
    const RT_LABELSST: u16 = 0x00FD;
    const RT_NUMBER: u16 = 0x0203;
    const RT_FILEPASS: u16 = 0x002F;
    const RT_SERIESTEXT: u16 = 0x100D;
    /// `MSODRAWINGGROUP` ([MS-XLS] 2.4.191) — the globals record whose payload
    /// holds the workbook's OfficeArt drawing data, images included.
    const RT_MSODRAWINGGROUP: u16 = 0x00EB;

    fn biff_rec(rec_type: u16, data: &[u8]) -> Vec<u8> {
        let mut buf = rec_type.to_le_bytes().to_vec();
        buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    /// A BIFF8 `BOF` opening a substream of the given doctype.
    fn bof(doctype: u16) -> Vec<u8> {
        let mut body = 0x0600u16.to_le_bytes().to_vec();
        body.extend_from_slice(&doctype.to_le_bytes());
        body.extend_from_slice(&[0u8; 12]);
        biff_rec(RT_BOF, &body)
    }

    /// A `BOUNDSHEET` with a wide (UTF-16LE) name, so a non-Latin sheet name
    /// exercises the Unicode name path.
    fn boundsheet(name: &str) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes()); // stream offset, unused
        body.push(0); // visible
        body.push(0); // worksheet
        body.push(name.encode_utf16().count() as u8);
        body.push(0x01); // wide
        for unit in name.encode_utf16() {
            body.extend_from_slice(&unit.to_le_bytes());
        }
        biff_rec(RT_BOUNDSHEET, &body)
    }

    /// An SST holding one wide string, so the Cyrillic cell pins the Unicode
    /// string path.
    fn sst(text: &str) -> Vec<u8> {
        let mut body = 1u32.to_le_bytes().to_vec(); // total
        body.extend_from_slice(&1u32.to_le_bytes()); // unique
        body.extend_from_slice(&(text.encode_utf16().count() as u16).to_le_bytes());
        body.push(0x01); // wide
        for unit in text.encode_utf16() {
            body.extend_from_slice(&unit.to_le_bytes());
        }
        biff_rec(RT_SST, &body)
    }

    /// A `LABELSST` cell indexing shared string 0.
    fn labelsst(row: u16, col: u16) -> Vec<u8> {
        let mut body = row.to_le_bytes().to_vec();
        body.extend_from_slice(&col.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes()); // xf
        body.extend_from_slice(&0u32.to_le_bytes()); // shared string index
        biff_rec(RT_LABELSST, &body)
    }

    fn number(row: u16, col: u16, value: f64) -> Vec<u8> {
        let mut body = row.to_le_bytes().to_vec();
        body.extend_from_slice(&col.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes()); // xf
        body.extend_from_slice(&value.to_le_bytes());
        biff_rec(RT_NUMBER, &body)
    }

    /// The Workbook stream of a one-sheet workbook: cell A1 the shared string
    /// `shared`, B2 the number 3.5, plus `extra` BIFF records in the globals
    /// substream (before its `EOF`).
    fn xls_stream(shared: &str, encrypted: bool, extra: &[u8]) -> Vec<u8> {
        let mut stream = bof(0x0005); // globals
        stream.extend(boundsheet("Данные"));
        stream.extend(sst(shared));
        if encrypted {
            stream.extend(biff_rec(RT_FILEPASS, &[0x12, 0x34, 0x56, 0x78]));
        }
        stream.extend(extra);
        stream.extend(biff_rec(RT_EOF, &[]));
        stream.extend(bof(0x0010)); // the sheet's own substream
        stream.extend(labelsst(0, 0));
        stream.extend(number(1, 1, 3.5));
        stream.extend(biff_rec(RT_EOF, &[]));
        stream
    }

    /// A `.xls` with one sheet: cell A1 the shared string `shared`, B2 the
    /// number 3.5.
    fn xls_fixture(shared: &str, encrypted: bool) -> Vec<u8> {
        cfb(&[("Workbook", &xls_stream(shared, encrypted, &[]))])
    }

    /// [`xls_fixture`]'s workbook plus one `MSODRAWINGGROUP` record holding a
    /// JPEG BLIP — the drawing records are where a `.xls` keeps its pictures,
    /// which this reading does not extract.
    fn xls_image_fixture() -> Vec<u8> {
        let drawing = biff_rec(RT_MSODRAWINGGROUP, &jpeg_blip());
        cfb(&[("Workbook", &xls_stream("Привет", false, &drawing))])
    }

    /// One `SeriesText` record: 2 reserved bytes, the character count, the
    /// flags byte (compressed characters) and the characters — the shape
    /// `office_oxide` writes in its own tests.
    fn series_text(text: &str) -> Vec<u8> {
        let mut data = vec![0u8, 0u8];
        data.push(text.chars().count() as u8);
        data.push(0);
        data.extend_from_slice(text.as_bytes());
        biff_rec(RT_SERIESTEXT, &data)
    }

    /// [`xls_fixture`]'s workbook plus a chart sheet of its own: a second
    /// `BOUNDSHEET` whose substream is a chart holding one `SeriesText`, so the
    /// text the arm does not print is what its answer has to name.
    fn xls_chart_fixture() -> Vec<u8> {
        let mut stream = bof(0x0005); // globals
        stream.extend(boundsheet("Данные"));
        stream.extend(boundsheet("Chart1"));
        stream.extend(sst("Привет"));
        stream.extend(biff_rec(RT_EOF, &[]));
        stream.extend(bof(0x0010)); // the sheet's own substream
        stream.extend(labelsst(0, 0));
        stream.extend(biff_rec(RT_EOF, &[]));
        stream.extend(bof(0x0020)); // the chart sheet's own substream
        stream.extend(series_text("Revenue"));
        stream.extend(biff_rec(RT_EOF, &[]));
        cfb(&[("Workbook", &stream)])
    }

    #[test]
    fn xls_chart_text_is_named_as_left_out() {
        // A chart sheet's content is its chart's text, which this reading does
        // not print: the sheet still shows as one with no valued cell, the text
        // is named in a note rather than left to pass as an empty sheet, and the
        // sheet itself is reported as the chart sheet it is.
        let bytes = xls_chart_fixture();
        let DocOutcome::Text {
            text,
            notes,
            unshown,
            ..
        } = convert(&bytes, Family::Xls)
        else {
            panic!("expected text");
        };
        assert_eq!(
            text,
            "Sheet \"Данные\":\n  A1: Привет\n\nSheet \"Chart1\": (no values)"
        );
        assert!(
            notes
                .iter()
                .any(|note| note.contains("text from its charts")),
            "got: {notes:?}"
        );
        assert_eq!(unshown.lines(), ["1 chart sheet(s) not shown"]);
    }

    #[test]
    fn xls_fixture_renders_sheet_cells_like_xlsx() {
        let bytes = xls_fixture("Привет", false);
        let DocOutcome::Text { text, notes, .. } = convert(&bytes, Family::Xls) else {
            panic!("expected text");
        };
        assert_eq!(text, "Sheet \"Данные\":\n  A1: Привет\n  B2: 3.5");
        assert!(notes.is_empty());
    }

    #[test]
    fn xls_embedded_image_is_reported_as_not_shown() {
        // The pictures in the workbook's drawing records are not extracted, so
        // the reading reports them alongside the cells it prints.
        let bytes = xls_image_fixture();
        let DocOutcome::Text {
            text,
            images,
            notes,
            unshown,
            ..
        } = convert(&bytes, Family::Xls)
        else {
            panic!("expected text");
        };
        assert_eq!(text, "Sheet \"Данные\":\n  A1: Привет\n  B2: 3.5");
        assert!(images.is_empty());
        assert!(notes.is_empty());
        assert_eq!(unshown.lines(), [UNSHOWN_IMAGE]);
    }

    #[test]
    fn xls_shared_string_break_stays_one_line() {
        // A break inside a value is escaped to `\n` so the cell keeps its one
        // line, exactly as the `.xlsx` arm's does.
        let bytes = xls_fixture("two\nlines", false);
        let DocOutcome::Text { text, .. } = convert(&bytes, Family::Xls) else {
            panic!("expected text");
        };
        assert_eq!(text, "Sheet \"Данные\":\n  A1: two\\nlines\n  B2: 3.5");
    }

    #[test]
    fn encrypted_xls_reports_password_protected() {
        let bytes = xls_fixture("Привет", true);
        assert!(matches!(
            convert(&bytes, Family::Xls),
            DocOutcome::Unreadable { reason } if reason == "password-protected"
        ));
    }

    // ── PowerPoint (.ppt) ───────────────────────────────────────────────

    const RT_DOCUMENT: u16 = 0x03E8;
    const RT_SLIDE: u16 = 0x03EE;
    const RT_SLIDE_ATOM: u16 = 0x03EF;
    const RT_NOTES: u16 = 0x03F0;
    const RT_NOTES_ATOM: u16 = 0x03F1;
    const RT_SLIDE_PERSIST_ATOM: u16 = 0x03F3;
    const RT_SLIDE_LIST_WITH_TEXT: u16 = 0x0FF0;
    const RT_TEXT_HEADER: u16 = 0x0F9F;
    const RT_TEXT_BYTES: u16 = 0x0FA8;
    const RT_USER_EDIT_ATOM: u16 = 0x0FF5;
    const RT_CURRENT_USER_ATOM: u16 = 0x0FF6;
    const RT_PERSIST_DIRECTORY_ATOM: u16 = 0x1772;
    const SLWT_SLIDES: u16 = 0;
    const SLWT_NOTES: u16 = 2;
    /// The ClientTextbox container every shape's text lives under.
    const RT_CLIENT_TEXTBOX: u16 = 0xF00D;
    /// The group container, a shape and a cell's anchor record — what a
    /// reconstructed table's grid of shapes is built from.
    const RT_SPGR_CONTAINER: u16 = 0xF003;
    const RT_SHAPE: u16 = 0xF004;
    const RT_CHILD_ANCHOR: u16 = 0xF00F;
    /// `OfficeArtClientData` — a shape's own PPT-specific data, where its
    /// embedded-object reference lives.
    const RT_CLIENT_DATA: u16 = 0xF011;
    /// `ExObjListContainer` — the document-wide table of external objects, a
    /// child of the `Document` container.
    const RT_EXTERNAL_OBJECT_LIST: u16 = 0x0409;
    /// `ExEmbed` — the container wrapping one `ExOleObjAtom`.
    const RT_EXTERNAL_OLE_EMBED: u16 = 0x0FCC;
    /// `ExOleObjAtom` — one embedded object's identity (its `objID`, kind and
    /// subtype).
    const RT_EXTERNAL_OLE_OBJECT_ATOM: u16 = 0x0FC3;
    /// `ExObjRefAtom` — a shape's reference to an external object, inside its
    /// `ClientData`.
    const RT_EXTERNAL_OBJECT_REF_ATOM: u16 = 0x0BC1;
    /// `ExMediaAtom` — a media object's id inside its container.
    const RT_EX_MEDIA_ATOM: u16 = 0x1004;
    /// `ExVideoContainer` — the video an `ExAviMovieContainer` holds.
    const RT_EX_VIDEO: u16 = 0x1005;
    /// `ExAviMovieContainer` — a video object, which has no `ExOleObjAtom`.
    const RT_EX_AVI_MOVIE: u16 = 0x1006;
    /// `ExWAVAudioEmbeddedContainer` — a sound object, which has none either.
    const RT_EX_WAV_AUDIO_EMBEDDED: u16 = 0x100F;

    /// The one external object a fixture deck's slide references. An embedded OLE
    /// object and the media objects are resolved from different records, and which
    /// record a media one is in is what tells the report a video or a sound from
    /// an embedded object; a linked OLE object is the same record as an embedded
    /// one with the kind a deck links by.
    #[derive(Clone, Copy)]
    enum OleFixture {
        Embedded(u32),
        Linked(u32),
        Video(u32),
        Audio(u32),
    }

    impl OleFixture {
        /// The id the slide's shape references.
        const fn obj_id(self) -> u32 {
            match self {
                Self::Embedded(id) | Self::Linked(id) | Self::Video(id) | Self::Audio(id) => id,
            }
        }

        /// The `ExOleObjAtom.type` the object is written with: a media object has
        /// no such atom — [`ex_obj_list`] writes one through this only for an OLE
        /// object.
        const fn kind(self) -> u32 {
            match self {
                Self::Linked(_) => super::OLE_LINKED,
                _ => 0,
            }
        }
    }

    fn atom(rec_type: u16, instance: u16, data: &[u8]) -> Vec<u8> {
        let mut buf = (instance << 4).to_le_bytes().to_vec();
        buf.extend_from_slice(&rec_type.to_le_bytes());
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    fn container(rec_type: u16, instance: u16, children: &[u8]) -> Vec<u8> {
        let mut buf = ((instance << 4) | 0x000F).to_le_bytes().to_vec();
        buf.extend_from_slice(&rec_type.to_le_bytes());
        buf.extend_from_slice(&(children.len() as u32).to_le_bytes());
        buf.extend_from_slice(children);
        buf
    }

    /// A `ClientTextbox` holding one text run of `text_type`
    /// (`TextTypeEnum`: 0 title, 2 notes).
    fn textbox(text_type: u32, text: &str) -> Vec<u8> {
        let mut children = atom(RT_TEXT_HEADER, 0, &text_type.to_le_bytes());
        children.extend(atom(RT_TEXT_BYTES, 0, text.as_bytes()));
        container(RT_CLIENT_TEXTBOX, 0, &children)
    }

    /// A grid cell of a reconstructed table: a shape anchored at `(left, top)`
    /// whose only text is a `ClientTextbox` of type 4 (`Tx_TYPE_OTHER`).
    fn table_cell(left: i32, top: i32, text: &str) -> Vec<u8> {
        let mut anchor = left.to_le_bytes().to_vec();
        anchor.extend_from_slice(&top.to_le_bytes());
        anchor.extend_from_slice(&(left + 100).to_le_bytes());
        anchor.extend_from_slice(&(top + 50).to_le_bytes());
        let mut children = atom(RT_CHILD_ANCHOR, 0, &anchor);
        children.extend(textbox(4, text));
        container(RT_SHAPE, 0, &children)
    }

    /// A group container whose four member shapes form a clean 2x2 grid — what
    /// the reader reconstructs as one table, in row-major cell order. The
    /// group's own placeholder shape comes first and carries no anchor.
    ///
    /// Cells are emitted column-major (A1, A2, B1, B2) while the reader's cell
    /// order is row-major (A1, B1, A2, B2), so the rendered order proves the
    /// grid was reconstructed rather than walked as flat shapes in shape order.
    fn table_group(cells: [&str; 4]) -> Vec<u8> {
        let mut children = container(RT_SHAPE, 0, &[]);
        children.extend(table_cell(0, 0, cells[0]));
        children.extend(table_cell(0, 50, cells[2]));
        children.extend(table_cell(100, 0, cells[1]));
        children.extend(table_cell(100, 50, cells[3]));
        container(RT_SPGR_CONTAINER, 0, &children)
    }

    /// A shape whose `ClientData` holds an `ExObjRefAtom` naming embedded
    /// object `obj_id` — the slide-side half of an embedded object, joined
    /// against the document's `ExObjListContainer` by `objID`.
    fn ole_shape(obj_id: u32) -> Vec<u8> {
        let client_data = container(
            RT_CLIENT_DATA,
            0,
            &atom(RT_EXTERNAL_OBJECT_REF_ATOM, 0, &obj_id.to_le_bytes()),
        );
        container(RT_SHAPE, 0, &client_data)
    }

    /// The document-wide external-object table: an `ExObjListContainer` holding
    /// the one object a fixture deck names. An embedded or linked object is an
    /// `ExEmbed` → `ExOleObjAtom` whose 16-byte body is the parse-relevant prefix
    /// `drawAspect`(4) + kind(4) + `objID`(4) + `subType`(4) ([MS-PPT] 2.10.20); a
    /// media object is its own container — a movie holding the video, or a sound
    /// container holding the atom directly — whose `ExMediaAtom` carries the id a
    /// shape references, and it has no OLE atom at all.
    fn ex_obj_list(ole: OleFixture) -> Vec<u8> {
        match ole {
            OleFixture::Embedded(_) | OleFixture::Linked(_) => {
                let mut body = vec![0u8; 16];
                body[4..8].copy_from_slice(&ole.kind().to_le_bytes());
                body[8..12].copy_from_slice(&ole.obj_id().to_le_bytes());
                let ole_atom = atom(RT_EXTERNAL_OLE_OBJECT_ATOM, 0, &body);
                let embed = container(RT_EXTERNAL_OLE_EMBED, 0, &ole_atom);
                container(RT_EXTERNAL_OBJECT_LIST, 0, &embed)
            }
            OleFixture::Video(obj_id) | OleFixture::Audio(obj_id) => {
                let mut body = obj_id.to_le_bytes().to_vec();
                body.extend([0u8; 4]); // mediaType(4) + mediaId(4)
                let media_atom = atom(RT_EX_MEDIA_ATOM, 0, &body);
                let (container_type, children) = match ole {
                    OleFixture::Video(_) => {
                        (RT_EX_AVI_MOVIE, container(RT_EX_VIDEO, 0, &media_atom))
                    }
                    _ => (RT_EX_WAV_AUDIO_EMBEDDED, media_atom),
                };
                let media = container(container_type, 0, &children);
                container(RT_EXTERNAL_OBJECT_LIST, 0, &media)
            }
        }
    }

    /// A `Slide` container with a `SlideAtom` naming its notes page, one title
    /// textbox and, when given, a reconstructed table shape and a shape
    /// referencing external object `ole`.
    fn slide_container(
        text: &str,
        notes_id: u32,
        table: Option<[&str; 4]>,
        ole: Option<u32>,
    ) -> Vec<u8> {
        let mut body = vec![0u8; 24];
        body[16..20].copy_from_slice(&notes_id.to_le_bytes()); // notesIdRef
        let mut children = atom(RT_SLIDE_ATOM, 0, &body);
        children.extend(textbox(0, text));
        if let Some(cells) = table {
            children.extend(table_group(cells));
        }
        if let Some(obj_id) = ole {
            children.extend(ole_shape(obj_id));
        }
        container(RT_SLIDE, 0, &children)
    }

    /// A `Notes` container: a `NotesAtom` naming the slide it belongs to and
    /// one notes-type textbox.
    fn notes_container(text: &str, slide_id: u32) -> Vec<u8> {
        let mut children = atom(RT_NOTES_ATOM, 0, &slide_id.to_le_bytes());
        children.extend(textbox(2, text));
        container(RT_NOTES, 0, &children)
    }

    /// A `SlidePersistAtom` entry of a slide/notes list.
    fn slide_persist(persist_id: u32, slide_id: u32) -> Vec<u8> {
        let mut body = persist_id.to_le_bytes().to_vec();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&slide_id.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        atom(RT_SLIDE_PERSIST_ATOM, 0, &body)
    }

    /// A `PersistDirectoryAtom` mapping `(persist id, stream offset)` pairs,
    /// sequential from the first id.
    fn persist_directory(entries: &[(u32, u32)]) -> Vec<u8> {
        let header = entries[0].0 | ((entries.len() as u32) << 20);
        let mut body = header.to_le_bytes().to_vec();
        for (_, offset) in entries {
            body.extend_from_slice(&offset.to_le_bytes());
        }
        atom(RT_PERSIST_DIRECTORY_ATOM, 0, &body)
    }

    /// A `UserEditAtom` (one edit, so `offsetLastEdit` is 0).
    fn user_edit_atom(offset_persist_directory: u32, doc_persist_id_ref: u32) -> Vec<u8> {
        let mut body = 0u32.to_le_bytes().to_vec(); // lastSlideIdRef
        body.extend_from_slice(&0u32.to_le_bytes()); // version
        body.extend_from_slice(&0u32.to_le_bytes()); // offsetLastEdit
        body.extend_from_slice(&offset_persist_directory.to_le_bytes());
        body.extend_from_slice(&doc_persist_id_ref.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes()); // persistIdSeed
        body.extend_from_slice(&0u32.to_le_bytes()); // lastView + unused
        atom(RT_USER_EDIT_ATOM, 0, &body)
    }

    /// A `CurrentUserAtom`; the encrypted `headerToken` is the reader's first
    /// encryption signal ([MS-PPT] §2.3.2).
    fn current_user(offset_to_current_edit: u32, encrypted: bool) -> Vec<u8> {
        let mut body = 0u32.to_le_bytes().to_vec(); // size
        let token: u32 = if encrypted { 0xF3D1_C4DF } else { 0 };
        body.extend_from_slice(&token.to_le_bytes()); // headerToken
        body.extend_from_slice(&offset_to_current_edit.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.push(0);
        body.push(0);
        body.extend_from_slice(&0u16.to_le_bytes());
        atom(RT_CURRENT_USER_ATOM, 0, &body)
    }

    /// The "PowerPoint Document" stream of a one-slide deck and the offset of
    /// its user edit. The slide (and its notes page, when `notes` is given)
    /// are resolved through the persist directory, the way a real PPT97 file
    /// stores its current content. `ole` names an external object the slide's
    /// shape references, declared in the document's own `ExObjListContainer`.
    fn ppt_document_stream(
        slide_text: &str,
        notes: Option<&str>,
        table: Option<[&str; 4]>,
        ole: Option<OleFixture>,
    ) -> (Vec<u8>, u32) {
        const SLIDE_ID: u32 = 256;
        const NOTES_ID: u32 = 257;

        let mut stream = Vec::new();

        // The document container (persist id 1) names the slide list and,
        // with notes, the notes list.
        let mut doc_children = container(
            RT_SLIDE_LIST_WITH_TEXT,
            SLWT_SLIDES,
            &slide_persist(2, SLIDE_ID),
        );
        if notes.is_some() {
            let notes_list = container(
                RT_SLIDE_LIST_WITH_TEXT,
                SLWT_NOTES,
                &slide_persist(3, NOTES_ID),
            );
            doc_children.extend(notes_list);
        }
        // The document-wide external-object table the slide's shape resolves
        // its reference against.
        if let Some(ole) = ole {
            doc_children.extend(ex_obj_list(ole));
        }
        let doc_offset = stream.len() as u32;
        stream.extend(container(RT_DOCUMENT, 0, &doc_children));

        // The current slide (persist id 2) and the notes page (3), reachable
        // only through the persist directory.
        let slide_offset = stream.len() as u32;
        let notes_id = if notes.is_some() { NOTES_ID } else { 0 };
        stream.extend(slide_container(
            slide_text,
            notes_id,
            table,
            ole.map(OleFixture::obj_id),
        ));
        let mut entries = vec![(1u32, doc_offset), (2, slide_offset)];
        if let Some(notes_text) = notes {
            let notes_offset = stream.len() as u32;
            stream.extend(notes_container(notes_text, SLIDE_ID));
            entries.push((3, notes_offset));
        }

        let pd_offset = stream.len() as u32;
        stream.extend(persist_directory(&entries));
        let edit_offset = stream.len() as u32;
        stream.extend(user_edit_atom(pd_offset, 1));
        (stream, edit_offset)
    }

    /// A `.ppt` package: the "PowerPoint Document" stream, the "Current User"
    /// stream, and — when the deck has images — a "Pictures" stream of raw
    /// BLIPs.
    fn ppt_cfb(
        stream: &[u8],
        edit_offset: u32,
        encrypted: bool,
        pictures: Option<&[u8]>,
    ) -> Vec<u8> {
        let user = current_user(edit_offset, encrypted);
        let mut streams = vec![("PowerPoint Document", stream), ("Current User", &user[..])];
        if let Some(pictures) = pictures {
            streams.push(("Pictures", pictures));
        }
        cfb(&streams)
    }

    /// A `.ppt`, encrypted or not. An encrypted deck needs only the encrypted
    /// `CurrentUserAtom` token — the record stream past the edit is ciphertext
    /// in a real file, but the reader refuses the deck before reading it.
    fn ppt_fixture(slide_text: &str, notes: Option<&str>, encrypted: bool) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, notes, None, None);
        ppt_cfb(&stream, edit_offset, encrypted, None)
    }

    /// [`ppt_fixture`]'s deck plus a third root stream `Pictures` holding one
    /// JPEG BLIP — where a `.ppt` keeps its images, a flat sequence the
    /// reader's picture walk reads directly.
    fn ppt_image_fixture(slide_text: &str) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, None, None);
        let pictures = jpeg_blip();
        ppt_cfb(&stream, edit_offset, false, Some(&pictures))
    }

    /// A `.ppt` whose only slide shape references embedded object `obj_id`,
    /// which the document's `ExObjListContainer` names — the two halves a
    /// slide's embedded object is resolved from.
    fn ppt_ole_fixture(slide_text: &str, obj_id: u32) -> Vec<u8> {
        let (stream, edit_offset) =
            ppt_document_stream(slide_text, None, None, Some(OleFixture::Embedded(obj_id)));
        ppt_cfb(&stream, edit_offset, false, None)
    }

    /// A `.ppt` whose only slide references the media object `media` — a video or
    /// a sound the deck holds, resolved from the `ExMediaAtom` of its own
    /// container rather than from an `ExOleObjAtom`.
    fn ppt_media_fixture(slide_text: &str, media: OleFixture) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, None, Some(media));
        ppt_cfb(&stream, edit_offset, false, None)
    }

    /// [`ppt_ole_fixture`]'s deck plus a `Pictures` image, so one reading names
    /// both an image and an object.
    fn ppt_image_and_object_fixture(slide_text: &str, obj_id: u32) -> Vec<u8> {
        let (stream, edit_offset) =
            ppt_document_stream(slide_text, None, None, Some(OleFixture::Embedded(obj_id)));
        let pictures = jpeg_blip();
        ppt_cfb(&stream, edit_offset, false, Some(&pictures))
    }

    /// A `.ppt` whose only slide carries an ordinary text run and a 2x2 table
    /// shape — what the reader reconstructs as a `TableBlock`.
    fn ppt_table_fixture(slide_text: &str, cells: [&str; 4]) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, Some(cells), None);
        ppt_cfb(&stream, edit_offset, false, None)
    }

    #[test]
    fn ppt_fixture_renders_slide_and_notes_like_pptx() {
        let bytes = ppt_fixture("Hello deck", Some("A speaker note"), false);
        let DocOutcome::Text { text, notes, .. } = convert(&bytes, Family::Ppt) else {
            panic!("expected text");
        };
        assert_eq!(
            text,
            "Slide 1:\n  Hello deck\n\nSlide 1 notes:\n  A speaker note"
        );
        assert!(notes.is_empty());
    }

    #[test]
    fn ppt_without_notes_skips_the_notes_block() {
        let bytes = ppt_fixture("Only slide text", None, false);
        let DocOutcome::Text { text, .. } = convert(&bytes, Family::Ppt) else {
            panic!("expected text");
        };
        assert_eq!(text, "Slide 1:\n  Only slide text");
    }

    #[test]
    fn encrypted_ppt_reports_password_protected() {
        let bytes = ppt_fixture("secret", None, true);
        assert!(matches!(
            convert(&bytes, Family::Ppt),
            DocOutcome::Unreadable { reason } if reason == "password-protected"
        ));
    }

    #[test]
    fn ppt_slide_table_cells_render_after_the_slide_text() {
        let bytes = ppt_table_fixture("Slide title", ["A1", "B1", "A2", "B2"]);
        let DocOutcome::Text { text, .. } = convert(&bytes, Family::Ppt) else {
            panic!("expected text");
        };
        assert_eq!(text, "Slide 1:\n  Slide title\n  A1\n  B1\n  A2\n  B2");
    }

    #[test]
    fn ppt_embedded_image_is_reported_as_not_shown() {
        // The deck's `Pictures` stream is not extracted, so the reading reports
        // the images it holds.
        let bytes = ppt_image_fixture("Slide title");
        let DocOutcome::Text {
            text,
            images,
            notes,
            unshown,
            ..
        } = convert(&bytes, Family::Ppt)
        else {
            panic!("expected text");
        };
        assert_eq!(text, "Slide 1:\n  Slide title");
        assert!(images.is_empty());
        assert!(notes.is_empty());
        assert_eq!(unshown.lines(), [UNSHOWN_IMAGE]);
    }

    #[test]
    fn ppt_embedded_object_is_reported_as_not_shown() {
        // A slide shape's `ExObjRefAtom` resolves, through the document's
        // `ExObjListContainer`, to one embedded object this reading does not
        // extract — so it is reported.
        let bytes = ppt_ole_fixture("Slide title", 1);
        let DocOutcome::Text {
            text,
            images,
            notes,
            unshown,
            ..
        } = convert(&bytes, Family::Ppt)
        else {
            panic!("expected text");
        };
        assert_eq!(text, "Slide 1:\n  Slide title");
        assert!(images.is_empty());
        assert!(notes.is_empty());
        assert_eq!(unshown.lines(), [UNSHOWN_OBJECT]);
    }

    /// A deck's media objects are named as the media they are rather than as
    /// embedded objects: the container the `ExObjList` holds says which is which,
    /// and the report names the loss by what the deck actually holds.
    #[test]
    fn ppt_media_objects_are_reported_by_their_kind() {
        for (media, expected) in [
            (OleFixture::Video(1), "1 embedded video(s) not shown"),
            (OleFixture::Audio(1), "1 embedded audio(s) not shown"),
        ] {
            let bytes = ppt_media_fixture("Slide title", media);
            let DocOutcome::Text { text, unshown, .. } = convert(&bytes, Family::Ppt) else {
                panic!("expected text");
            };
            assert_eq!(text, "Slide 1:\n  Slide title");
            assert_eq!(unshown.lines(), [expected]);
        }
    }

    /// A linked OLE object is not content the deck holds — the file keeps the
    /// link, not what it points at — so a deck whose only object is linked reports
    /// nothing: the report names content that is really there.
    #[test]
    fn ppt_linked_object_is_not_reported_as_content() {
        let (stream, edit_offset) =
            ppt_document_stream("Slide title", None, None, Some(OleFixture::Linked(1)));
        let bytes = ppt_cfb(&stream, edit_offset, false, None);
        let DocOutcome::Text { text, unshown, .. } = convert(&bytes, Family::Ppt) else {
            panic!("expected text");
        };
        assert_eq!(text, "Slide 1:\n  Slide title");
        assert!(unshown.lines().is_empty());
    }

    /// A deck holding both a picture and an embedded object reports each loss by
    /// its own kind rather than joined into one line.
    #[test]
    fn ppt_image_and_object_are_reported_separately() {
        let bytes = ppt_image_and_object_fixture("Slide title", 1);
        let DocOutcome::Text { notes, unshown, .. } = convert(&bytes, Family::Ppt) else {
            panic!("expected text");
        };
        assert!(notes.is_empty());
        assert_eq!(unshown.lines(), [UNSHOWN_OBJECT, UNSHOWN_IMAGE]);
    }

    #[test]
    fn unsupported_version_keeps_its_own_reason() {
        // The reader declares a "recognized but unreadable version" variant for
        // each of its formats and raises NEITHER through `from_reader` in
        // `office_oxide` 0.1.13: a Word 6.0/95 `.doc` is handed to the crate's
        // own Word 6 reader before the Word 97 FIB version check runs, and
        // `XlsError::UnsupportedVersion` is constructed nowhere in it. No
        // fixture can therefore drive these, so the mapping is asserted here
        // directly — it is the guard that keeps an old format from being called
        // corrupt by the reader version that starts raising them.
        for outcome in [
            doc_error(&DocError::UnsupportedVersion("Word 6.0/95".to_string())),
            xls_error(&XlsError::UnsupportedVersion("BIFF5".to_string())),
        ] {
            assert!(matches!(
                outcome,
                DocOutcome::Unreadable { reason }
                    if reason == "an old Office version this tool cannot read"
            ));
        }
        // Encryption keeps its own reason, and an ordinary parse error is
        // still corruption.
        assert!(matches!(
            doc_error(&DocError::Encrypted),
            DocOutcome::Unreadable { reason } if reason == "password-protected"
        ));
        assert!(matches!(
            doc_error(&DocError::Corrupted("bad".to_string())),
            DocOutcome::Unreadable { reason } if reason == "corrupt or unreadable .doc"
        ));
        // The presentation reader has no unsupported-version variant, so a
        // `.ppt` maps only encryption and corruption.
        assert!(matches!(
            ppt_error(&PptError::Encrypted),
            DocOutcome::Unreadable { reason } if reason == "password-protected"
        ));
        assert!(matches!(
            ppt_error(&PptError::Corrupted("bad".to_string())),
            DocOutcome::Unreadable { reason } if reason == "corrupt or unreadable .ppt"
        ));
    }

    // ── Classification ──────────────────────────────────────────────────

    #[test]
    fn family_of_lists_exactly_the_legacy_extensions() {
        assert_eq!(family_of(Path::new("a.doc")), Some(Family::Doc));
        assert_eq!(family_of(Path::new("a.DOC")), Some(Family::Doc));
        assert_eq!(family_of(Path::new("a.xls")), Some(Family::Xls));
        assert_eq!(family_of(Path::new("a.ppt")), Some(Family::Ppt));
        assert_eq!(family_of(Path::new("a.docx")), None);
        assert_eq!(family_of(Path::new("a.txt")), None);
    }

    /// The classification of a legacy-named CFB container is `document.rs`'s
    /// own (`cfb_magic_classifies_by_name`); what this covers is that every
    /// family's reader calls a garbage container corruption rather than a format
    /// this project does not convert.
    #[test]
    fn garbage_cfb_is_unreadable_not_unsupported() {
        let mut bytes = crate::document::CFB_MAGIC.to_vec();
        bytes.extend_from_slice(b"not a compound file body");
        for family in [Family::Doc, Family::Xls, Family::Ppt] {
            assert!(matches!(
                convert(&bytes, family),
                DocOutcome::Unreadable { reason } if reason.starts_with("corrupt or unreadable .")
            ));
        }
    }
}
