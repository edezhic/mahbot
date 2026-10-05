//! Legacy binary Office reading: the CFB half of document conversion.
//!
//! [`crate::document`] owns content-first detection; [`crate::ooxml`] reads the
//! ZIP-container OOXML packages. This module reads the older, pre-OOXML Word,
//! Excel and PowerPoint files — the `.doc`/`.xls`/`.ppt` whose container is a
//! CFB (compound file) rather than a ZIP — through the `office_oxide` crate,
//! and renders them into the *same* text shapes the OOXML arms produce, so a
//! model reading an old file and a new one sees the same vocabulary.
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
use crate::ooxml::{column_letters, text_block, text_lines};
use office_oxide::doc::{DocDocument, DocError};
use office_oxide::ppt::{PptDocument, PptError, SlideText, TextType};
use office_oxide::xls::{CellValue, Sheet, XlsDocument, XlsError};
use std::io::Cursor;
use std::path::Path;

/// Extensions accepted as legacy Word documents.
const DOC_EXTENSIONS: &[&str] = &["doc"];
/// Extensions accepted as legacy Excel workbooks.
const XLS_EXTENSIONS: &[&str] = &["xls"];
/// Extensions accepted as legacy PowerPoint presentations.
const PPT_EXTENSIONS: &[&str] = &["ppt"];

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

/// The document's text. The legacy arms are TEXT ONLY by design, so the images
/// and OLE objects the reader can decode are deliberately not surfaced.
fn convert_doc(bytes: &[u8]) -> DocOutcome {
    match DocDocument::from_reader(Cursor::new(bytes)) {
        Ok(doc) => {
            let mut notes = Vec::new();
            if !doc.text_complete() {
                notes.push(
                    "the reader stopped early — some of the document's text may be missing"
                        .to_string(),
                );
            }
            DocOutcome::Text {
                text: doc.plain_text(),
                images: Vec::new(),
                notes,
                all_page_text_lost: false,
            }
        }
        Err(err) => doc_error(&err),
    }
}

/// Every sheet in workbook order, rendered exactly as [`crate::ooxml::convert_xlsx`]
/// renders a `.xlsx` sheet: a `Sheet "<name>":` block and one indented
/// `{address}: {value}` line per valued cell, with the value the way Excel
/// displays it. A cell whose formula the reader cannot recover shows the value
/// Excel cached for it.
fn convert_xls(bytes: &[u8]) -> DocOutcome {
    match XlsDocument::from_reader(Cursor::new(bytes)) {
        Ok(doc) => {
            let mut blocks = Vec::new();
            for sheet in &doc.sheets {
                blocks.push(text_block(
                    &format!("Sheet \"{}\":", sheet.name),
                    "(no values)",
                    &sheet_lines(sheet),
                ));
            }
            let mut notes = Vec::new();
            if doc.truncated() {
                notes.push(
                    "the workbook was cut short — later sheets or cells may be missing".to_string(),
                );
            }
            DocOutcome::Text {
                text: blocks.join("\n\n").trim_end().to_string(),
                images: Vec::new(),
                notes,
                all_page_text_lost: false,
            }
        }
        Err(err) => xls_error(&err),
    }
}

/// The valued cells of one sheet, in row then column order. The grid is
/// jagged (a row holds cells only up to its last non-empty one), so a missing
/// position is simply not iterated; a cell that is `Empty`, or whose displayed
/// text is blank, contributes no line.
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
            let display = display.trim();
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

/// Each slide's text plus its speaker notes, rendered exactly as
/// [`crate::ooxml::convert_pptx`] renders a `.pptx` slide.
fn convert_ppt(bytes: &[u8]) -> DocOutcome {
    match PptDocument::from_reader(Cursor::new(bytes)) {
        Ok(doc) => {
            let mut blocks = Vec::new();
            for (index, slide) in doc.slides.iter().enumerate() {
                let number = index + 1;
                blocks.push(text_block(
                    &format!("Slide {number}:"),
                    "(no text)",
                    &slide_lines(slide),
                ));
                let notes: Vec<String> = slide
                    .text_runs
                    .iter()
                    .filter(|run| run.text_type == TextType::Notes)
                    .flat_map(|run| paragraph_lines(&run.text))
                    .collect();
                if !notes.is_empty() {
                    blocks.push(text_lines(&format!("Slide {number} notes:"), &notes));
                }
            }
            let mut notes = Vec::new();
            if !doc.text_complete() {
                notes.push(
                    "the reader stopped early — some of the presentation's text may be missing"
                        .to_string(),
                );
            }
            DocOutcome::Text {
                text: blocks.join("\n\n").trim_end().to_string(),
                images: Vec::new(),
                notes,
                all_page_text_lost: false,
            }
        }
        Err(err) => ppt_error(&err),
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

/// Split a run's text at the paragraph marks the binary readers keep inside
/// it (`\r`, `\u{0B}` and `\n` are each a line break) and drop the blanks, the
/// way the OOXML readers drop an empty `<a:p>`.
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
    // 512-byte header, a directory sector, a FAT sector and then the streams'
    // own sectors. These fixtures are built in code so no binary file is
    // committed and nothing is downloaded at test time.
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

    /// Assemble a minimal v3 CFB from root-level streams, at most two of them
    /// (one directory sector holds the root, the streams and the terminator).
    fn cfb(streams: &[(&str, &[u8])]) -> Vec<u8> {
        assert!(
            streams.len() <= 2,
            "one directory sector holds four entries"
        );

        // Sector plan: 0 = directory, 1 = FAT, then each stream in order.
        let mut starts = Vec::new();
        let mut runs = Vec::new();
        let mut next = 2u32;
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
        file[0x4C..0x50].copy_from_slice(&1u32.to_le_bytes()); // DIFAT[0] = FAT sector
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

        // FAT: the directory and FAT sectors, then each stream's chain.
        let fat = 1024;
        let mut entries = vec![FREE_SECT; 128];
        entries[0] = END_OF_CHAIN;
        entries[1] = FAT_SECT;
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

    // ── Word (.doc) ─────────────────────────────────────────────────────

    /// Offset the fixtures put the Word text at, past every FIB field.
    const DOC_TEXT_FC: u32 = 0x400;

    /// A Word 97 FIB ([MS-DOC] §2.5.1): `wIdent` 0xA5EC, the 1Table flag, the
    /// declared main-text length and the CLX pointer, with the `fEncrypted`
    /// flag for the encrypted fixture. Every other FibRgFcLcb97 field stays 0.
    fn doc_fib(ccp_text: u32, fc_clx: u32, lcb_clx: u32, encrypted: bool) -> Vec<u8> {
        let mut fib = vec![0u8; 1024];
        fib[0..2].copy_from_slice(&0xA5ECu16.to_le_bytes());
        fib[2..4].copy_from_slice(&0x00C1u16.to_le_bytes());
        let flags = (1u16 << 9) | (u16::from(encrypted) << 8);
        fib[0x0A..0x0C].copy_from_slice(&flags.to_le_bytes());
        fib[0x4C..0x50].copy_from_slice(&ccp_text.to_le_bytes());
        fib[0x01A2..0x01A6].copy_from_slice(&fc_clx.to_le_bytes());
        fib[0x01A6..0x01AA].copy_from_slice(&lcb_clx.to_le_bytes());
        fib
    }

    /// A CLX holding one Unicode piece covering `[0, char_count)` at `fc`.
    fn doc_clx(fc: u32, char_count: u32) -> Vec<u8> {
        let mut clx = vec![0x02u8]; // Pcdt
        clx.extend_from_slice(&16u32.to_le_bytes()); // (1+1)*4 + 1*8
        clx.extend_from_slice(&0u32.to_le_bytes());
        clx.extend_from_slice(&char_count.to_le_bytes());
        clx.extend_from_slice(&0u16.to_le_bytes()); // PCD unused
        clx.extend_from_slice(&fc.to_le_bytes()); // Unicode: fc used directly
        clx.extend_from_slice(&0u16.to_le_bytes()); // prm
        clx
    }

    /// A `.doc` whose main text is `text`, stored as one UTF-16LE piece.
    fn doc_fixture(text: &str, encrypted: bool) -> Vec<u8> {
        let utf16: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let char_count = text.encode_utf16().count() as u32;
        let clx = doc_clx(DOC_TEXT_FC, char_count);
        let mut word = vec![0u8; DOC_TEXT_FC as usize + utf16.len()];
        word[..1024].copy_from_slice(&doc_fib(char_count, 0, clx.len() as u32, encrypted));
        word[DOC_TEXT_FC as usize..].copy_from_slice(&utf16);
        cfb(&[("WordDocument", &word), ("1Table", &clx)])
    }

    #[test]
    fn doc_fixture_renders_its_text() {
        let bytes = doc_fixture("Hello legacy doc", false);
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
    fn encrypted_doc_reports_password_protected() {
        let bytes = doc_fixture("secret", true);
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

    /// A `.xls` with one sheet: cell A1 a Cyrillic shared string, B2 the
    /// number 3.5.
    fn xls_fixture(encrypted: bool) -> Vec<u8> {
        let mut stream = bof(0x0005); // globals
        stream.extend(boundsheet("Данные"));
        stream.extend(sst("Привет"));
        if encrypted {
            stream.extend(biff_rec(RT_FILEPASS, &[0x12, 0x34, 0x56, 0x78]));
        }
        stream.extend(biff_rec(RT_EOF, &[]));
        stream.extend(bof(0x0010)); // the sheet's own substream
        stream.extend(labelsst(0, 0));
        stream.extend(number(1, 1, 3.5));
        stream.extend(biff_rec(RT_EOF, &[]));
        cfb(&[("Workbook", &stream)])
    }

    #[test]
    fn xls_fixture_renders_sheet_cells_like_xlsx() {
        let bytes = xls_fixture(false);
        let DocOutcome::Text { text, notes, .. } = convert(&bytes, Family::Xls) else {
            panic!("expected text");
        };
        assert_eq!(text, "Sheet \"Данные\":\n  A1: Привет\n  B2: 3.5");
        assert!(notes.is_empty());
    }

    #[test]
    fn encrypted_xls_reports_password_protected() {
        let bytes = xls_fixture(true);
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

    /// A `Slide` container with a `SlideAtom` naming its notes page, one title
    /// textbox and, when given, a reconstructed table shape.
    fn slide_container(text: &str, notes_id: u32, table: Option<[&str; 4]>) -> Vec<u8> {
        let mut body = vec![0u8; 24];
        body[16..20].copy_from_slice(&notes_id.to_le_bytes()); // notesIdRef
        let mut children = atom(RT_SLIDE_ATOM, 0, &body);
        children.extend(textbox(0, text));
        if let Some(cells) = table {
            children.extend(table_group(cells));
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
    /// stores its current content.
    fn ppt_document_stream(
        slide_text: &str,
        notes: Option<&str>,
        table: Option<[&str; 4]>,
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
        let doc_offset = stream.len() as u32;
        stream.extend(container(RT_DOCUMENT, 0, &doc_children));

        // The current slide (persist id 2) and the notes page (3), reachable
        // only through the persist directory.
        let slide_offset = stream.len() as u32;
        let notes_id = if notes.is_some() { NOTES_ID } else { 0 };
        stream.extend(slide_container(slide_text, notes_id, table));
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

    /// A `.ppt`, encrypted or not. An encrypted deck needs only the encrypted
    /// `CurrentUserAtom` token — the record stream past the edit is ciphertext
    /// in a real file, but the reader refuses the deck before reading it.
    fn ppt_fixture(slide_text: &str, notes: Option<&str>, encrypted: bool) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, notes, None);
        cfb(&[
            ("PowerPoint Document", &stream),
            ("Current User", &current_user(edit_offset, encrypted)),
        ])
    }

    /// A `.ppt` whose only slide carries an ordinary text run and a 2x2 table
    /// shape — what the reader reconstructs as a `TableBlock`.
    fn ppt_table_fixture(slide_text: &str, cells: [&str; 4]) -> Vec<u8> {
        let (stream, edit_offset) = ppt_document_stream(slide_text, None, Some(cells));
        cfb(&[
            ("PowerPoint Document", &stream),
            ("Current User", &current_user(edit_offset, false)),
        ])
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
