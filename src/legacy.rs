//! Legacy binary Office reading: the CFB half of document conversion.
//!
//! [`crate::document`] owns content-first detection; [`crate::ooxml`] reads the ZIP-container
//! OOXML packages. This module reads the pre-OOXML Word, Excel and PowerPoint files — a CFB
//! (compound file) container rather than a ZIP — through the `office_oxide` crate, and renders
//! them into the same text shapes the OOXML arms use, so a model reads an old file and a new one
//! the same way. The pictures the pipeline converts are written into `out_dir`; every picture it
//! does not is named — a vector metafile as a drawing in the shared report
//! ([`crate::reader_output::Unshown`]), any other format in the same "cannot convert" note a
//! modern package's picture gets. The report also holds a `.ppt`'s objects counted by what they
//! are (an OLE object the deck only links is counted not at all, since the file holds the link
//! and not what it points at) and each `.xls` chart sheet, so nothing the crate hands over
//! vanishes silently. What the arms deliberately do not adopt is the newer arms' vocabulary:
//! hidden geometry, hidden sheets and slides, merged ranges, cell comments, hyperlinks, defined
//! names, per-note splitting and formula text stay out, as does any mark the reader gives no way
//! to place.
//!
//! # Invariants
//!
//! - **No panics of its own.** The whole parse runs inside
//!   [`crate::shutdown::contain_panics`], and every fallible step degrades to
//!   [`DocOutcome::Unreadable`] or a note; the containment suppresses the panic hook, so no
//!   panic report reaches the product's logs.
//! - **Encrypted, unsupported and unreadable are different outcomes.** A password-protected
//!   file is [`DocOutcome::Unreadable`] with `reason` `"password-protected"` — the verdict the
//!   OOXML and PDF arms give too. A legacy *version* the reader recognizes but cannot read
//!   keeps its own reason rather than being called corrupt. Neither is ever `Unsupported`
//!   (which callers read as "fall back to the raw bytes").
//! - **Reading only.** The source file is never modified or opened for writing; the only file
//!   created is a delivered picture inside the caller's `out_dir`.

use crate::document::{
    DELIVERABLE_IMAGE_EXTENSIONS, DRAWING_EXTENSIONS, DocOutcome, SkipReason, SkippedImages,
    ensure_out_dir, write_picture,
};
use crate::reader_output::{
    DOC_COMMENTS, DOC_ENDNOTES, DOC_FOOTNOTES, DOC_HEADERS_FOOTERS, NO_VALUES, TEXT_BOX, Unshown,
    UnshownKind, column_letters, sheet_header, slide_header, slide_notes_header, text_block,
    text_lines,
};
use crate::util::one_line;
use office_oxide::cfb::blip::BlipImage;
use office_oxide::doc::{DocDocument, DocError, SubDocumentKind};
use office_oxide::ir::SheetKind;
use office_oxide::ppt::{OleObjectInfo, PptDocument, PptError, SlideText, TextType};
use office_oxide::xls::{CellValue, Sheet, XlsDocument, XlsError};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// `OleObjectInfo::kind` of an OLE object a deck links rather than holds: the file keeps the
/// link, so the report names nothing for it. The media kinds carry no such distinction, so a
/// linked sound or video is named like an embedded one.
const OLE_LINKED: u32 = 1;

/// The legacy family a caller reads `bytes` as. The CFB magic is shared by all three (and by a
/// genuinely encrypted OOXML package), so [`crate::document::real_family`] is what tells them
/// apart, and the write path asks the family's own reader whether the file is encrypted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Doc,
    Xls,
    Ppt,
}

impl Family {
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Doc => "doc",
            Self::Xls => "xls",
            Self::Ppt => "ppt",
        }
    }

    #[must_use]
    pub(crate) const fn real(self) -> crate::document::RealFamily {
        match self {
            Self::Doc => crate::document::RealFamily::LegacyDoc,
            Self::Xls => crate::document::RealFamily::LegacyXls,
            Self::Ppt => crate::document::RealFamily::LegacyPpt,
        }
    }
}

/// The old binary family `path`'s NAME claims, or `None` when it claims none. Asked only about a
/// CFB container whose directory names no Office document, where the name is the last evidence a
/// caller has; the content decides everywhere else.
#[must_use]
pub(crate) fn named_family(path: &Path) -> Option<Family> {
    if crate::util::has_extension(path, &["doc"]) {
        Some(Family::Doc)
    } else if crate::util::has_extension(path, &["xls"]) {
        Some(Family::Xls)
    } else if crate::util::has_extension(path, &["ppt"]) {
        Some(Family::Ppt)
    } else {
        None
    }
}

/// Read `bytes` as the legacy format `family`, into the shapes the OOXML arms use, writing
/// delivered pictures into `out_dir` (see [`write_pictures`]) when it is given. The whole parse is
/// contained: a panic costs the document, not the caller's turn. The containment is
/// [`crate::shutdown::contain_panics`] rather than a bare `catch_unwind`, so a panic inside the
/// `office_oxide` readers does not print a report and backtrace into the product's logs while the
/// caller is told the file is merely corrupt.
#[must_use]
pub(crate) fn convert(bytes: &[u8], family: Family, out_dir: Option<&Path>) -> DocOutcome {
    match crate::shutdown::contain_panics(|| parse(bytes, family, out_dir)) {
        Ok(outcome) => outcome,
        Err(_) => corrupt(family),
    }
}

fn parse(bytes: &[u8], family: Family, out_dir: Option<&Path>) -> DocOutcome {
    match family {
        Family::Doc => convert_doc(bytes, out_dir),
        Family::Xls => convert_xls(bytes, out_dir),
        Family::Ppt => convert_ppt(bytes, out_dir),
    }
}

/// Whether `bytes` is a password-protected document of `family`. The family's own reader is the
/// only thing that can tell encryption from corruption — the CFB container looks the same either
/// way — so this asks it and reads the verdict a reading would give. Deliberately a whole parse:
/// only the write path calls this, and it must know about a password before it reserves an output
/// (the parse is given no output directory, so the probe writes and names nothing).
#[must_use]
pub(crate) fn is_encrypted(bytes: &[u8], family: Family) -> bool {
    matches!(
        convert(bytes, family, None),
        DocOutcome::Unreadable { reason } if reason == ENCRYPTED_REASON
    )
}

/// A `.doc` reading: the body, then each subdocument story as a labelled block in the order a
/// `.docx` reading prints them — headers and footers, footnotes, endnotes, comments — then the
/// text boxes and the names of the embedded objects. The reader hands the arms a whole story at
/// once, so each is one plural, unnumbered block: the per-definition numbering and inline
/// `[text box]` placement of the `.docx` reading have no counterpart here.
fn convert_doc(bytes: &[u8], out_dir: Option<&Path>) -> DocOutcome {
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
            // The `.docx` reader prints each text box where it is anchored; this reader can
            // only append, so the boxes come last, main-document ones before the header's.
            for kind in [SubDocumentKind::TextBoxes, SubDocumentKind::HeaderTextBoxes] {
                let lines = doc_story_lines(&doc, kind);
                if !lines.is_empty() {
                    sections.push(text_lines(TEXT_BOX, &lines));
                }
            }
            // The reader names the embedded objects only in `plain_text()`; taken off that
            // string they are the one thing telling a model an object is there, so they are last.
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
            // The reader's object names are the sections above; its pictures are delivered —
            // and the ones this pipeline does not convert named — like any other reading's.
            let mut unshown = Unshown::default();
            let images = write_pictures(|| doc.images(), out_dir, &mut unshown, &mut notes);
            let text = sections.join("\n").trim_end().to_string();
            DocOutcome::text(text, images, notes, unshown)
        }
        Err(err) => doc_error(&err),
    }
}

/// One subdocument kind's story, one line per paragraph. Empty when the document has no such
/// story or it holds nothing.
fn doc_story_lines(doc: &DocDocument, kind: SubDocumentKind) -> Vec<String> {
    doc.subdocuments()
        .iter()
        .find(|sub| sub.kind == kind)
        .map(|sub| paragraph_lines(&sub.text))
        .unwrap_or_default()
}

/// The text `plain_text()` assembles before it appends the embedded-object names: the main text,
/// then every non-empty subdocument story trimmed, in `subdocuments()` order, each on its own
/// line. The reader's `ole_objects` API is crate-private, so mirroring its assembly is the only
/// way to reach the `[<description>]` lines it appends past this prefix; a reader change that
/// breaks the mirror yields no names rather than a wrong tail, and the crate is pinned exactly.
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

/// Every sheet in workbook order, in the shape [`crate::ooxml::convert_xlsx`] gives a `.xlsx`
/// sheet: a `Sheet "<name>":` block and one indented `{address}: {value}` line per valued cell,
/// with the value the way Excel displays it. Hidden geometry, merged ranges, cell comments,
/// hyperlinks and defined names are marked by the `.xlsx` arm alone. A cell whose formula the
/// reader cannot recover shows the value Excel cached for it.
fn convert_xls(bytes: &[u8], out_dir: Option<&Path>) -> DocOutcome {
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
            // The reader recovers a chart's own text — a chart sheet's content or an embedded
            // chart's series names, labels and titles — which this reading does not print: a
            // legacy workbook is read for its cells. Named rather than passed over, the way the
            // `.xlsx` arm names the charts it cannot extract.
            if !doc.chart_text().is_empty()
                || doc.sheets.iter().any(|sheet| !sheet.chart_text.is_empty())
            {
                notes.push(
                    "the workbook holds text from its charts (chart titles, series and trendline names, and axis titles), which is not extracted".to_string(),
                );
            }
            // The report holds the rest of the loss: the pictures the pipeline does not convert,
            // and each chart sheet as the kind of sheet it is. The cell comments, hyperlinks and
            // defined names the reader also hands over are not printed for an old format by
            // choice (see the module docs).
            let mut unshown = Unshown::default();
            let images = write_pictures(|| doc.images(), out_dir, &mut unshown, &mut notes);
            unshown.add(
                UnshownKind::ChartSheet,
                doc.sheets
                    .iter()
                    .filter(|sheet| sheet.kind == SheetKind::Chart)
                    .count(),
            );
            let text = blocks.join("\n\n").trim_end().to_string();
            DocOutcome::text(text, images, notes, unshown)
        }
        Err(err) => xls_error(&err),
    }
}

/// The valued cells of one sheet, in row then column order. The grid is jagged (a row holds
/// cells only up to its last non-empty one), so a missing position is not iterated; a cell that
/// is `Empty`, or whose displayed text is blank, contributes no line. A value holding a line
/// break is shown as one line ([`one_line`]), so no value can break the one-line-per-cell shape.
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
            // A BIFF8 sheet has at most 256 columns, so this never exceeds `u32`; a grid that
            // did would be corrupt rather than worth a panic.
            let Ok(column) = u32::try_from(col) else {
                continue;
            };
            let address = format!("{}{}", column_letters(column), row + 1);
            lines.push(format!("{address}: {display}"));
        }
    }
    lines
}

/// Each slide's text plus its speaker notes, in the shapes [`crate::ooxml::convert_pptx`]
/// renders — without its marks: a title run reads here as ordinary text and a hidden slide is not
/// flagged, though the IR holds both, while a SmartArt diagram's text has no counterpart in it.
fn convert_ppt(bytes: &[u8], out_dir: Option<&Path>) -> DocOutcome {
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
            // The deck's pictures are delivered — the ones the pipeline cannot convert named
            // instead. The objects the reader resolves per slide are counted by what they are: a
            // media object as the video or sound it holds, an embedded object or ActiveX control
            // as an object the reading does not open. An OLE object the deck only links is
            // skipped: the file holds the link, not the content.
            let mut unshown = Unshown::default();
            let images = write_pictures(|| doc.images(), out_dir, &mut unshown, &mut notes);
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
            let text = blocks.join("\n\n").trim_end().to_string();
            DocOutcome::text(text, images, notes, unshown)
        }
        Err(err) => ppt_error(&err),
    }
}

/// Write the pictures a legacy reading delivers into `out_dir` (created when the first lands) and
/// return their paths, naming every other one the way a modern package's media is named: a vector
/// drawing is one the reading does not show — counted in `unshown` — and any other undelivered
/// format is added to `notes` as the reason note [`crate::document::SkippedImages`] states. The
/// delivered formats are the shared ones, so an old format and a modern package deliver the same.
/// The reader's accessor is called only when there is an `out_dir` to write into — it extracts
/// lazily — so the write path's encryption probe, which passes `None`, scans for pictures not at
/// all.
fn write_pictures<'a>(
    images: impl FnOnce() -> &'a [BlipImage],
    out_dir: Option<&Path>,
    unshown: &mut Unshown,
    notes: &mut Vec<String>,
) -> Vec<PathBuf> {
    let Some(out_dir) = out_dir else {
        return Vec::new();
    };
    let mut skipped = SkippedImages::default();
    let mut next_suffix = HashMap::new();
    let mut paths = Vec::new();
    for (slot, image) in images().iter().enumerate() {
        let file_name = format!("image_{}.{}", slot + 1, image.format.extension());
        let named = Path::new(&file_name);
        if !crate::util::has_extension(named, DELIVERABLE_IMAGE_EXTENSIONS) {
            if crate::util::has_extension(named, DRAWING_EXTENSIONS) {
                unshown.add(UnshownKind::Drawing, 1);
            } else {
                skipped.add(SkipReason::Unsupported);
            }
            continue;
        }
        ensure_out_dir(out_dir);
        paths.extend(write_picture(
            out_dir,
            &file_name,
            &image.data,
            &mut next_suffix,
            &mut skipped,
        ));
    }
    notes.extend(skipped.notes());
    paths
}

/// A slide's own text lines — every run that is not speaker notes, in shape order — followed by
/// its reconstructed tables' cell text, each cell collapsed to one line so a table reads as
/// further lines of the same slide.
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

/// Split a text the binary readers hand over at the paragraph marks they keep inside it (`\r`,
/// `\u{0B}` and `\n` are each a line break) and drop the blanks, so one line is one paragraph.
fn paragraph_lines(text: &str) -> Vec<String> {
    text.split(['\r', '\u{0B}', '\n'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The reason of the password-protected verdict: one literal the whole document layer states,
/// because a caller recognizes a password by it rather than by the arm that produced it, and
/// [`is_encrypted`] must recognize the very verdict a reading gives.
const ENCRYPTED_REASON: &str = "password-protected";

fn encrypted() -> DocOutcome {
    DocOutcome::Unreadable {
        reason: ENCRYPTED_REASON.to_string(),
    }
}

/// The verdict for an intact legacy file in a version the reader recognizes but cannot read — its
/// own reason, so an old format is never called corrupt. No `office_oxide` 0.1.13 error carries
/// it, so this guards the reader version that starts doing so.
fn unsupported() -> DocOutcome {
    DocOutcome::Unreadable {
        reason: "an old Office version this tool cannot read".to_string(),
    }
}

/// A `.doc` parse error as a verdict: encryption and an unsupported version keep their own
/// reasons; everything else is corruption.
fn doc_error(err: &DocError) -> DocOutcome {
    match err {
        DocError::Encrypted => encrypted(),
        DocError::UnsupportedVersion(_) => unsupported(),
        _ => corrupt(Family::Doc),
    }
}

fn xls_error(err: &XlsError) -> DocOutcome {
    match err {
        XlsError::Encrypted => encrypted(),
        XlsError::UnsupportedVersion(_) => unsupported(),
        _ => corrupt(Family::Xls),
    }
}

/// A `.ppt` parse error as a verdict. The reader has no "recognized but unreadable version"
/// variant, so encryption is the only reason that is not corruption.
fn ppt_error(err: &PptError) -> DocOutcome {
    match err {
        PptError::Encrypted => encrypted(),
        _ => corrupt(Family::Ppt),
    }
}

/// A legacy file that is neither encrypted, unsupported nor readable: corruption or a truncated
/// container, named as corrupt rather than unsupported since the name said the format is accepted.
fn corrupt(family: Family) -> DocOutcome {
    DocOutcome::Unreadable {
        reason: format!("corrupt or unreadable .{}", family.name()),
    }
}

/// Fixtures shared by this module's tests, [`crate::document`]'s and
/// [`crate::tools::read_document`]'s: the CFB assembler and the `.doc`/`.xls`/`.ppt` stream
/// recipes are built here once, so every side reads the very same file.
#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the hand-built fixtures' stream offsets and counts fit the CFB/BIFF/PPT record fields by construction"
)]
pub(crate) mod test_fixtures {
    use super::*;

    // The legacy files are CFB (compound file) containers. A v3 CFB is a 512-byte header, as
    // many 512-byte directory sectors as the entries need, a FAT sector and then the streams' own
    // sectors; these fixtures are built in code so no binary file is committed.
    //
    // The mini-stream path is avoided structurally: the root entry declares no stream of its own
    // (`start = END_OF_CHAIN`, `size = 0`), so the reader reads every stream from its regular
    // sectors — which is why a stream below the 4096-byte cutoff is still read whole.

    const END_OF_CHAIN: u32 = 0xFFFF_FFFE;
    const FAT_SECT: u32 = 0xFFFF_FFFD;
    const FREE_SECT: u32 = 0xFFFF_FFFF;
    const NO_ENTRY: u32 = 0xFFFF_FFFF;

    /// Assemble a minimal v3 CFB from root-level streams, however many directory entries it takes,
    /// chained through the FAT like every other chain. The header's directory count stays 0 (a v3
    /// reader derives the span from the chain) and its FAT-sector count stays 1.
    pub(crate) fn cfb(streams: &[(&str, &[u8])]) -> Vec<u8> {
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

    const DOC_TEXT_FC: u32 = 0x400;

    /// A Word 97 FIB ([MS-DOC] §2.5.1): `wIdent` 0xA5EC, the 1Table flag, the `ccp*` lengths and
    /// the CLX pointer, with `fEncrypted` for the encrypted fixture. `ccp` is the eight `FibRgLw97`
    /// lengths in `[MS-DOC]` order — text, footnotes, headers, the ignored macro length, comments,
    /// endnotes, text boxes and header text boxes.
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

    /// A CLX whose Pcdt holds one Unicode piece per `(fc, char_count)` in order, the CP ranges
    /// running consecutively from 0.
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

    /// The subdocument stories with their `FibRgLw97` slot in the fixed `[MS-DOC]` order the
    /// reader walks (`text` is slot 0; slot 3 is the ignored macro length, which stays zero).
    const DOC_STORY_SLOTS: [(SubDocumentKind, usize); 6] = [
        (SubDocumentKind::Footnotes, 1),
        (SubDocumentKind::HeadersFooters, 2),
        (SubDocumentKind::Comments, 4),
        (SubDocumentKind::Endnotes, 5),
        (SubDocumentKind::TextBoxes, 6),
        (SubDocumentKind::HeaderTextBoxes, 7),
    ];

    /// The `(WordDocument, 1Table)` streams of a `.doc` whose main text is `body`, with the
    /// non-empty `(kind, text)` entries of `stories` as subdocuments, each one UTF-16LE piece laid
    /// out in the fixed `[MS-DOC]` story order. `body` is one piece at [`DOC_TEXT_FC`].
    pub(super) fn doc_streams(
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

    pub(crate) fn doc_fixture(
        body: &str,
        stories: &[(SubDocumentKind, &str)],
        encrypted: bool,
    ) -> Vec<u8> {
        let (word, clx) = doc_streams(body, stories, encrypted);
        cfb(&[("WordDocument", &word), ("1Table", &clx)])
    }

    /// An OfficeArt record header plus `body`: `recVer` 0 with `recInstance` 0x6E0 (even, so the
    /// reader computes no secondary UID), `recType`, and `recLen`. The bytes mirror
    /// `office_oxide`'s own `make_blip` fixture header for header, and the crate is pinned exactly.
    fn record(rec_type: u16, body: &[u8]) -> Vec<u8> {
        let inst: u16 = 0x6E0;
        let mut buf = (inst << 4).to_le_bytes().to_vec();
        buf.extend_from_slice(&rec_type.to_le_bytes());
        buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
        buf.extend_from_slice(body);
        buf
    }

    /// One bitmap BLIP record: a 16-byte UID, then the 1-byte tag a bitmap BLIP carries (a
    /// metafile has none), then `picture`.
    fn blip(rec_type: u16, picture: &[u8]) -> Vec<u8> {
        let mut body = vec![0u8; 17];
        body.extend_from_slice(picture);
        record(rec_type, &body)
    }

    fn png_blip(picture: &[u8]) -> Vec<u8> {
        blip(0xF01E, picture)
    }

    /// [`doc_fixture`]'s package plus a third root stream `Data` holding one PNG BLIP — where Word
    /// 97 keeps a picture, which the reader finds by scanning that stream.
    pub(crate) fn doc_picture_fixture(body: &str, picture: &[u8]) -> Vec<u8> {
        let (word, clx) = doc_streams(body, &[], false);
        let data = png_blip(picture);
        cfb(&[("WordDocument", &word), ("1Table", &clx), ("Data", &data)])
    }

    pub(super) const RT_BOF: u16 = 0x0809;
    pub(super) const RT_EOF: u16 = 0x000A;
    pub(super) const RT_BOUNDSHEET: u16 = 0x0085;
    pub(super) const RT_SST: u16 = 0x00FC;
    pub(super) const RT_LABELSST: u16 = 0x00FD;
    pub(super) const RT_NUMBER: u16 = 0x0203;
    pub(super) const RT_FILEPASS: u16 = 0x002F;
    pub(super) const RT_SERIESTEXT: u16 = 0x100D;
    /// `MSODRAWINGGROUP` ([MS-XLS] 2.4.191) — the globals record whose payload holds the
    /// workbook's OfficeArt drawing data, images included.
    pub(super) const RT_MSODRAWINGGROUP: u16 = 0x00EB;

    pub(super) fn biff_rec(rec_type: u16, data: &[u8]) -> Vec<u8> {
        let mut buf = rec_type.to_le_bytes().to_vec();
        buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    pub(super) fn bof(doctype: u16) -> Vec<u8> {
        let mut body = 0x0600u16.to_le_bytes().to_vec();
        body.extend_from_slice(&doctype.to_le_bytes());
        body.extend_from_slice(&[0u8; 12]);
        biff_rec(RT_BOF, &body)
    }

    /// A `BOUNDSHEET` with a wide (UTF-16LE) name, so a non-Latin sheet name exercises the
    /// Unicode name path.
    pub(super) fn boundsheet(name: &str) -> Vec<u8> {
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

    /// An SST holding one wide string, so the Cyrillic cell pins the Unicode string path.
    pub(super) fn sst(text: &str) -> Vec<u8> {
        let mut body = 1u32.to_le_bytes().to_vec(); // total
        body.extend_from_slice(&1u32.to_le_bytes()); // unique
        body.extend_from_slice(&(text.encode_utf16().count() as u16).to_le_bytes());
        body.push(0x01); // wide
        for unit in text.encode_utf16() {
            body.extend_from_slice(&unit.to_le_bytes());
        }
        biff_rec(RT_SST, &body)
    }

    pub(super) fn labelsst(row: u16, col: u16) -> Vec<u8> {
        let mut body = row.to_le_bytes().to_vec();
        body.extend_from_slice(&col.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes()); // xf
        body.extend_from_slice(&0u32.to_le_bytes()); // shared string index
        biff_rec(RT_LABELSST, &body)
    }

    pub(super) fn number(row: u16, col: u16, value: f64) -> Vec<u8> {
        let mut body = row.to_le_bytes().to_vec();
        body.extend_from_slice(&col.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes()); // xf
        body.extend_from_slice(&value.to_le_bytes());
        biff_rec(RT_NUMBER, &body)
    }

    /// The Workbook stream of a one-sheet workbook: cell A1 the shared string `shared`, B2 the
    /// number 3.5, plus `extra` BIFF records in the globals substream.
    pub(super) fn xls_stream(shared: &str, encrypted: bool, extra: &[u8]) -> Vec<u8> {
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

    pub(super) fn xls_fixture(shared: &str, encrypted: bool) -> Vec<u8> {
        cfb(&[("Workbook", &xls_stream(shared, encrypted, &[]))])
    }

    /// [`xls_fixture`]'s workbook plus one `MSODRAWINGGROUP` record holding a PNG BLIP — the
    /// drawing records are where a `.xls` keeps its pictures.
    pub(crate) fn xls_picture_fixture(picture: &[u8]) -> Vec<u8> {
        let drawing = biff_rec(RT_MSODRAWINGGROUP, &png_blip(picture));
        cfb(&[("Workbook", &xls_stream("Привет", false, &drawing))])
    }

    /// One `SeriesText` record: 2 reserved bytes, the character count, the flags byte and the
    /// characters — the shape `office_oxide` writes in its own tests.
    pub(super) fn series_text(text: &str) -> Vec<u8> {
        let mut data = vec![0u8, 0u8];
        data.push(text.chars().count() as u8);
        data.push(0);
        data.extend_from_slice(text.as_bytes());
        biff_rec(RT_SERIESTEXT, &data)
    }

    /// [`xls_fixture`]'s workbook plus a chart sheet: a second `BOUNDSHEET` whose substream holds
    /// one `SeriesText`, so the text the arm does not print is what its answer has to name.
    pub(super) fn xls_chart_fixture() -> Vec<u8> {
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

    pub(super) const RT_DOCUMENT: u16 = 0x03E8;
    pub(super) const RT_SLIDE: u16 = 0x03EE;
    pub(super) const RT_SLIDE_ATOM: u16 = 0x03EF;
    pub(super) const RT_NOTES: u16 = 0x03F0;
    pub(super) const RT_NOTES_ATOM: u16 = 0x03F1;
    pub(super) const RT_SLIDE_PERSIST_ATOM: u16 = 0x03F3;
    pub(super) const RT_SLIDE_LIST_WITH_TEXT: u16 = 0x0FF0;
    pub(super) const RT_TEXT_HEADER: u16 = 0x0F9F;
    pub(super) const RT_TEXT_BYTES: u16 = 0x0FA8;
    pub(super) const RT_USER_EDIT_ATOM: u16 = 0x0FF5;
    pub(super) const RT_CURRENT_USER_ATOM: u16 = 0x0FF6;
    pub(super) const RT_PERSIST_DIRECTORY_ATOM: u16 = 0x1772;
    pub(super) const SLWT_SLIDES: u16 = 0;
    pub(super) const SLWT_NOTES: u16 = 2;
    /// The ClientTextbox container every shape's text lives under.
    pub(super) const RT_CLIENT_TEXTBOX: u16 = 0xF00D;
    /// The group container, a shape and a cell's anchor record — what a reconstructed table's grid
    /// of shapes is built from.
    pub(super) const RT_SPGR_CONTAINER: u16 = 0xF003;
    pub(super) const RT_SHAPE: u16 = 0xF004;
    pub(super) const RT_CHILD_ANCHOR: u16 = 0xF00F;
    /// `OfficeArtClientData` — a shape's own PPT-specific data, where its embedded-object
    /// reference lives.
    pub(super) const RT_CLIENT_DATA: u16 = 0xF011;
    /// `ExObjListContainer` — the document-wide table of external objects.
    pub(super) const RT_EXTERNAL_OBJECT_LIST: u16 = 0x0409;
    pub(super) const RT_EXTERNAL_OLE_EMBED: u16 = 0x0FCC;
    /// `ExOleObjAtom` — one embedded object's identity (`objID`, kind, subtype).
    pub(super) const RT_EXTERNAL_OLE_OBJECT_ATOM: u16 = 0x0FC3;
    /// `ExObjRefAtom` — a shape's reference to an external object, inside its `ClientData`.
    pub(super) const RT_EXTERNAL_OBJECT_REF_ATOM: u16 = 0x0BC1;
    pub(super) const RT_EX_MEDIA_ATOM: u16 = 0x1004;
    pub(super) const RT_EX_VIDEO: u16 = 0x1005;
    pub(super) const RT_EX_AVI_MOVIE: u16 = 0x1006;
    /// `ExWAVAudioEmbeddedContainer` — a sound object, which has none either.
    pub(super) const RT_EX_WAV_AUDIO_EMBEDDED: u16 = 0x100F;

    /// The one external object a fixture deck's slide references: an embedded OLE object and the
    /// media objects resolve from different records, which is what tells the report one from the
    /// other.
    #[derive(Clone, Copy)]
    pub(super) enum OleFixture {
        Embedded(u32),
        Video(u32),
        Audio(u32),
    }

    impl OleFixture {
        const fn obj_id(self) -> u32 {
            match self {
                Self::Embedded(id) | Self::Video(id) | Self::Audio(id) => id,
            }
        }
    }

    pub(super) fn atom(rec_type: u16, instance: u16, data: &[u8]) -> Vec<u8> {
        let mut buf = (instance << 4).to_le_bytes().to_vec();
        buf.extend_from_slice(&rec_type.to_le_bytes());
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    pub(super) fn container(rec_type: u16, instance: u16, children: &[u8]) -> Vec<u8> {
        let mut buf = ((instance << 4) | 0x000F).to_le_bytes().to_vec();
        buf.extend_from_slice(&rec_type.to_le_bytes());
        buf.extend_from_slice(&(children.len() as u32).to_le_bytes());
        buf.extend_from_slice(children);
        buf
    }

    /// A `ClientTextbox` holding one text run of `text_type` (`TextTypeEnum`: 0 title, 2 notes).
    pub(super) fn textbox(text_type: u32, text: &str) -> Vec<u8> {
        let mut children = atom(RT_TEXT_HEADER, 0, &text_type.to_le_bytes());
        children.extend(atom(RT_TEXT_BYTES, 0, text.as_bytes()));
        container(RT_CLIENT_TEXTBOX, 0, &children)
    }

    /// A grid cell of a reconstructed table: a shape anchored at `(left, top)` whose only text is
    /// a `ClientTextbox` of type 4.
    pub(super) fn table_cell(left: i32, top: i32, text: &str) -> Vec<u8> {
        let mut anchor = left.to_le_bytes().to_vec();
        anchor.extend_from_slice(&top.to_le_bytes());
        anchor.extend_from_slice(&(left + 100).to_le_bytes());
        anchor.extend_from_slice(&(top + 50).to_le_bytes());
        let mut children = atom(RT_CHILD_ANCHOR, 0, &anchor);
        children.extend(textbox(4, text));
        container(RT_SHAPE, 0, &children)
    }

    /// A group container whose four member shapes form a clean 2x2 grid — what the reader
    /// reconstructs as one table. Cells are emitted column-major while the reader's order is
    /// row-major, so the rendered order proves the grid was reconstructed.
    pub(super) fn table_group(cells: [&str; 4]) -> Vec<u8> {
        let mut children = container(RT_SHAPE, 0, &[]);
        children.extend(table_cell(0, 0, cells[0]));
        children.extend(table_cell(0, 50, cells[2]));
        children.extend(table_cell(100, 0, cells[1]));
        children.extend(table_cell(100, 50, cells[3]));
        container(RT_SPGR_CONTAINER, 0, &children)
    }

    /// A shape whose `ClientData` holds an `ExObjRefAtom` naming embedded object `obj_id` — the
    /// slide-side half, joined against the document's `ExObjListContainer`.
    pub(super) fn ole_shape(obj_id: u32) -> Vec<u8> {
        let client_data = container(
            RT_CLIENT_DATA,
            0,
            &atom(RT_EXTERNAL_OBJECT_REF_ATOM, 0, &obj_id.to_le_bytes()),
        );
        container(RT_SHAPE, 0, &client_data)
    }

    /// The document-wide external-object table holding the one object a fixture deck names. An
    /// embedded object is an `ExEmbed` → `ExOleObjAtom` whose 16-byte body is the parse-relevant
    /// prefix `drawAspect`(4) + kind(4) + `objID`(4) + `subType`(4) ([MS-PPT] 2.10.20); a media
    /// object has its own container and no OLE atom.
    pub(super) fn ex_obj_list(ole: OleFixture) -> Vec<u8> {
        match ole {
            OleFixture::Embedded(_) => {
                let mut body = vec![0u8; 16];
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

    /// A `Slide` container with a `SlideAtom` naming its notes page, one title textbox and, when
    /// given, a table shape and a shape referencing external object `ole`.
    pub(super) fn slide_container(
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

    pub(super) fn notes_container(text: &str, slide_id: u32) -> Vec<u8> {
        let mut children = atom(RT_NOTES_ATOM, 0, &slide_id.to_le_bytes());
        children.extend(textbox(2, text));
        container(RT_NOTES, 0, &children)
    }

    pub(super) fn slide_persist(persist_id: u32, slide_id: u32) -> Vec<u8> {
        let mut body = persist_id.to_le_bytes().to_vec();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&slide_id.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        atom(RT_SLIDE_PERSIST_ATOM, 0, &body)
    }

    pub(super) fn persist_directory(entries: &[(u32, u32)]) -> Vec<u8> {
        let header = entries[0].0 | ((entries.len() as u32) << 20);
        let mut body = header.to_le_bytes().to_vec();
        for (_, offset) in entries {
            body.extend_from_slice(&offset.to_le_bytes());
        }
        atom(RT_PERSIST_DIRECTORY_ATOM, 0, &body)
    }

    pub(super) fn user_edit_atom(
        offset_persist_directory: u32,
        doc_persist_id_ref: u32,
    ) -> Vec<u8> {
        let mut body = 0u32.to_le_bytes().to_vec(); // lastSlideIdRef
        body.extend_from_slice(&0u32.to_le_bytes()); // version
        body.extend_from_slice(&0u32.to_le_bytes()); // offsetLastEdit
        body.extend_from_slice(&offset_persist_directory.to_le_bytes());
        body.extend_from_slice(&doc_persist_id_ref.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes()); // persistIdSeed
        body.extend_from_slice(&0u32.to_le_bytes()); // lastView + unused
        atom(RT_USER_EDIT_ATOM, 0, &body)
    }

    /// A `CurrentUserAtom`; the encrypted `headerToken` is the reader's first encryption signal
    /// ([MS-PPT] §2.3.2).
    pub(super) fn current_user(offset_to_current_edit: u32, encrypted: bool) -> Vec<u8> {
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

    /// The "PowerPoint Document" stream of a one-slide deck and the offset of its user edit. The
    /// slide (and its notes page) resolve through the persist directory.
    pub(super) fn ppt_document_stream(
        slide_text: &str,
        notes: Option<&str>,
        table: Option<[&str; 4]>,
        ole: Option<OleFixture>,
    ) -> (Vec<u8>, u32) {
        const SLIDE_ID: u32 = 256;
        const NOTES_ID: u32 = 257;

        let mut stream = Vec::new();

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
        if let Some(ole) = ole {
            doc_children.extend(ex_obj_list(ole));
        }
        let doc_offset = stream.len() as u32;
        stream.extend(container(RT_DOCUMENT, 0, &doc_children));

        // The current slide (persist id 2) and the notes page (3), reachable only through the
        // persist directory.
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

    /// A `.ppt` package: the "PowerPoint Document" stream, the "Current User" stream, and — when
    /// the deck has images — a "Pictures" stream of raw BLIPs.
    pub(super) fn ppt_cfb(
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

    /// A `.ppt`, encrypted or not: an encrypted deck needs only the encrypted `CurrentUserAtom`
    /// token, as the reader refuses it before reading the ciphertext past the edit.
    pub(super) fn ppt_fixture(slide_text: &str, notes: Option<&str>, encrypted: bool) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, notes, None, None);
        ppt_cfb(&stream, edit_offset, encrypted, None)
    }

    /// [`ppt_fixture`]'s deck plus a third root stream `Pictures` holding one PNG BLIP.
    pub(crate) fn ppt_picture_fixture(slide_text: &str, picture: &[u8]) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, None, None);
        let pictures = png_blip(picture);
        ppt_cfb(&stream, edit_offset, false, Some(&pictures))
    }

    /// A `.ppt` whose only slide shape references embedded object `obj_id`, which the document's
    /// `ExObjListContainer` names.
    pub(super) fn ppt_ole_fixture(slide_text: &str, obj_id: u32) -> Vec<u8> {
        let (stream, edit_offset) =
            ppt_document_stream(slide_text, None, None, Some(OleFixture::Embedded(obj_id)));
        ppt_cfb(&stream, edit_offset, false, None)
    }

    /// A `.ppt` whose only slide references the media object `media` — resolved from its own
    /// container's `ExMediaAtom` rather than from an `ExOleObjAtom`.
    pub(super) fn ppt_media_fixture(slide_text: &str, media: OleFixture) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, None, Some(media));
        ppt_cfb(&stream, edit_offset, false, None)
    }

    /// A `.ppt` whose only slide carries an ordinary text run and a 2x2 table shape.
    pub(super) fn ppt_table_fixture(slide_text: &str, cells: [&str; 4]) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, Some(cells), None);
        ppt_cfb(&stream, edit_offset, false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::*;
    use super::*;
    use crate::util::test::noisy_png;

    /// The report line an arm gives one embedded object — pinned as a literal so a wording drift
    /// in [`crate::reader_output::Unshown`] fails a test.
    const UNSHOWN_OBJECT: &str = "1 embedded object part(s) not shown";

    /// The `.doc` text: the body alone, every sub-document as its own labelled block.
    #[test]
    fn doc_fixture_renders_its_text() {
        for (label, fixture, expected) in [
            (
                "a plain body",
                doc_fixture("Hello legacy doc", &[], false),
                "Hello legacy doc",
            ),
            (
                "every sub-document story",
                doc_fixture(
                    "Body text",
                    &[
                        (SubDocumentKind::Footnotes, "A footnote"),
                        (SubDocumentKind::HeadersFooters, "A header\rA footer"),
                        (SubDocumentKind::Comments, "A comment"),
                        (SubDocumentKind::Endnotes, "An endnote"),
                    ],
                    false,
                ),
                "Body text\n\
                 Headers and footers:\n  A header\n  A footer\n\
                 Footnotes:\n  A footnote\n\
                 Endnotes:\n  An endnote\n\
                 Comments:\n  A comment",
            ),
            (
                "an empty story prints no block",
                doc_fixture(
                    "Body only",
                    &[
                        (SubDocumentKind::Footnotes, " \r"),
                        (SubDocumentKind::Comments, "A comment"),
                    ],
                    false,
                ),
                "Body only\nComments:\n  A comment",
            ),
        ] {
            let DocOutcome::Text {
                text,
                images,
                notes,
                ..
            } = convert(&fixture, Family::Doc, None)
            else {
                panic!("expected text for {label}");
            };
            assert_eq!(text, expected, "{label}");
            assert!(images.is_empty(), "{label}");
            assert!(notes.is_empty(), "{label}");
        }
    }

    /// Every family's embedded picture is delivered like a modern package's media: written out to
    /// `out_dir`, decodable and byte-for-byte what the reader held, with the family's own text read
    /// unchanged beside it and nothing reported as left out.
    #[test]
    fn every_family_delivers_its_embedded_picture() {
        let picture = noisy_png(2, 2);
        for (family, bytes, text) in [
            (
                Family::Doc,
                doc_picture_fixture("Body text", &picture),
                "Body text",
            ),
            (
                Family::Xls,
                xls_picture_fixture(&picture),
                "Sheet \"Данные\":\n  A1: Привет\n  B2: 3.5",
            ),
            (
                Family::Ppt,
                ppt_picture_fixture("Slide title", &picture),
                "Slide 1:\n  Slide title",
            ),
        ] {
            let dir = tempfile::tempdir().expect("temp dir");
            let DocOutcome::Text {
                text: read,
                images,
                notes,
                unshown,
                ..
            } = convert(&bytes, family, Some(dir.path()))
            else {
                panic!("expected text");
            };
            assert_eq!(read, text, "{family:?}");
            assert_eq!(images.len(), 1, "{family:?}");
            assert_eq!(
                std::fs::read(&images[0]).expect("read the written picture"),
                picture,
                "{family:?}"
            );
            image::open(&images[0]).expect("the written picture decodes");
            assert!(notes.is_empty(), "{family:?}: {notes:?}");
            assert!(unshown.lines().is_empty(), "{family:?}");
        }
    }

    #[test]
    fn doc_object_names_mirror_stays_a_prefix_of_the_readers_own_text() {
        // The crate exposes its embedded-object names only in `plain_text()`, so the assembly
        // before them is mirrored (`doc_plain_text_before_objects`). No fixture here carries an
        // object pool, so what is pinned is that the mirror still reproduces the reader's assembly
        // on a document with stories.
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
    fn encrypted_file_reports_password_protected() {
        for (family, fixture, expected) in [
            (
                Family::Doc,
                doc_fixture("secret", &[], true),
                "password-protected",
            ),
            (
                Family::Xls,
                xls_fixture("Привет", true),
                "password-protected",
            ),
            (
                Family::Ppt,
                ppt_fixture("secret", None, true),
                "password-protected",
            ),
        ] {
            assert!(
                matches!(
                    convert(&fixture, family, None),
                    DocOutcome::Unreadable { reason } if reason == expected
                ),
                "{family:?}"
            );
        }
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
        } = convert(&bytes, Family::Xls, None)
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
        let DocOutcome::Text { text, notes, .. } = convert(&bytes, Family::Xls, None) else {
            panic!("expected text");
        };
        assert_eq!(text, "Sheet \"Данные\":\n  A1: Привет\n  B2: 3.5");
        assert!(notes.is_empty());
    }

    #[test]
    fn ppt_fixture_renders_slide_and_notes_like_pptx() {
        let bytes = ppt_fixture("Hello deck", Some("A speaker note"), false);
        let DocOutcome::Text { text, notes, .. } = convert(&bytes, Family::Ppt, None) else {
            panic!("expected text");
        };
        assert_eq!(
            text,
            "Slide 1:\n  Hello deck\n\nSlide 1 notes:\n  A speaker note"
        );
        assert!(notes.is_empty());
    }

    #[test]
    fn ppt_slide_table_cells_render_after_the_slide_text() {
        let bytes = ppt_table_fixture("Slide title", ["A1", "B1", "A2", "B2"]);
        let DocOutcome::Text { text, .. } = convert(&bytes, Family::Ppt, None) else {
            panic!("expected text");
        };
        assert_eq!(text, "Slide 1:\n  Slide title\n  A1\n  B1\n  A2\n  B2");
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
        } = convert(&bytes, Family::Ppt, None)
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
            let DocOutcome::Text { text, unshown, .. } = convert(&bytes, Family::Ppt, None) else {
                panic!("expected text");
            };
            assert_eq!(text, "Slide 1:\n  Slide title");
            assert_eq!(unshown.lines(), [expected]);
        }
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
}
