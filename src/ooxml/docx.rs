//! The Word (`.docx`/`.docm`) reader: a WordprocessingML walk that keeps the
//! layout, the tracked changes and the non-body parts a reader needs.
//!
//! [`super`] owns the ZIP plumbing and the other two OOXML families; this
//! module is the whole Word half. The body, the headers and footers, the
//! footnotes and endnotes and the comments all go through the same walker, so a
//! marker means the same thing wherever it appears.
//!
//! # What the walk preserves
//!
//! - **Tables** as a `Table N:` label plus one line per row, two-space
//!   indented, cells joined with `" | "`, each cell spelled out as
//!   `C{column}` plus its annotations. Merge geometry is annotation, never
//!   silence: a cell covering several columns says so, and a merge
//!   continuation names the cell whose value it carries without repeating it.
//!   A table inside a cell is read — the cell token carries its own `Table N:`
//!   block — but the row-per-line shape has no room to indent it.
//! - **Tracked changes** as inline markers rather than as final-looking text:
//!   a wrapper around the content it revises, a `¶`-suffixed marker for a
//!   paragraph mark, an annotation on the table, the row or the cell whose own
//!   formatting changed, `[fmt section]` for a section — and, for the document
//!   as a whole, an `Unaccepted tracked changes: …` line that tallies every
//!   kind and names its authors. A revision in a copy the walk deliberately
//!   does not render — the second half of a duplicated text box — is not
//!   tallied either.
//! - **Note, comment and field references** where they are anchored, numbered
//!   in Word's own display order (a sequence number assigned on the first
//!   reference, never the raw `w:id`), and marked as such rather than left to
//!   pass for the note's own text.
//! - **Text boxes** as an indented `[text box]` block, read once: Word writes
//!   each one twice, as an `mc:Choice` and as the `mc:Fallback` beside it.
//!
//! The notation the model reads, every marker spelled out, is documented in
//! `src/prompt/tool/read.md` and `read_strict.md`, and every marker this walk
//! prints is listed once in `assets/docgen/rules.json` (`docx_marks`) — the list
//! the document kit refuses a `find` carrying one by, and the list the test
//! `every_docx_mark_the_shared_list_names_is_a_mark_the_reader_prints` checks
//! this reader's own output against, so a marker renamed here cannot leave the
//! kit naming one the reader no longer prints.
//!
//! # Invariants
//!
//! - **No panics of its own.** A reader error, an element nested past
//!   [`MAX_ELEMENT_DEPTH`] or a part that will not parse degrades: the body
//!   makes the package [`DocOutcome::Unreadable`], and every other part is
//!   skipped. A flow that leaves a field open reports it at the flow's end
//!   rather than letting the field swallow the rest.
//! - **Elements are matched on their local name** (namespace- and prefix-blind,
//!   like the rest of the package reader), so a producer using a prefix other
//!   than `w:` still parses.
//! - **One sink stack.** Text always lands in the innermost open sink, so a
//!   table cell or a text box renders into its own buffer and is placed into
//!   the enclosing flow as one block — nothing has to count what was written
//!   where.
//! - **The walk's own bookkeeping is bounded rather than document-decided.**
//!   The geometry a table asks for is clamped to [`MAX_TABLE_COLUMNS`] columns
//!   where its cells are placed — one cell's `w:gridSpan`, a row's
//!   `w:gridBefore`, and the total a row's cells add up to; a field marker is
//!   built from the head of its instruction rather than from a copy of it; a
//!   header/footer part is looked up by name rather than by scanning the parts
//!   named before it; and an author a revision repeats is remembered rather than
//!   searched for.

use super::{
    DocOutcome, Relationship, SkippedImages, append_entity, attr, ensure_out_dir, read_zip_entry,
    relationships, resolve_part, scan_elements, unreadable, write_media_parts,
};
use crate::reader_output::{
    TEXT_BOX, WORD_COMMENT, WORD_ENDNOTE, WORD_FOOTER, WORD_FOOTNOTE, WORD_HEADER, labeled_block,
    lines_of, text_lines,
};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{Cursor, Read, Seek};
use std::path::Path;
use std::rc::Rc;
use zip::ZipArchive;

// ── Bounds and part names ───────────────────────────────────────

/// Maximum element nesting the walk enters before it gives up on the part. The
/// walk recurses — one stack frame per level — and runs on a blocking thread
/// with a small stack, so a crafted package must not be able to exhaust it.
const MAX_ELEMENT_DEPTH: usize = 128;

/// The most columns a table may be laid out over. The geometry the document
/// asks for decides how much the walk allocates per column, so it is clamped:
/// a single `w:gridSpan`/`w:gridBefore` where it is read, and the running
/// column a row's cells add up to where the cell is placed.
const MAX_TABLE_COLUMNS: usize = 256;

/// Part holding the WordprocessingML body.
const DOCX_BODY_PART: &str = "word/document.xml";
/// Prefix of the embedded-media parts in a Word package.
const DOCX_MEDIA_PREFIX: &str = "word/media/";

/// Part holding the body's relationships: the notes parts and the header/footer
/// parts are named there, never in the body itself.
const DOCX_RELS_PART: &str = "word/_rels/document.xml.rels";

/// Base a Word relationship target resolves against.
const DOCX_BASE: &str = "word/";

/// A field instruction is reported up to this many characters — a runaway
/// instruction must not dominate the text.
const MAX_FIELD_INSTRUCTION_CHARS: usize = 80;

// ── Events ──────────────────────────────────────────────────────

/// A package XML reader over one part.
type Xml<'a> = Reader<&'a [u8]>;

/// The walker's only fault: the part being read is not usable, so nothing more
/// can be read out of it.
struct Fault;

/// One element reduced to what the walk needs: its local name and its
/// attributes, both owned so nothing borrows the event buffer while the walker
/// mutates itself.
struct Elem {
    name: Vec<u8>,
    attrs: Vec<(Vec<u8>, String)>,
}

impl Elem {
    /// The element with every attribute name's namespace/prefix stripped and
    /// every value unescaped. An unreadable value reads as empty rather than
    /// costing the part.
    fn of(event: &BytesStart<'_>) -> Self {
        Self {
            name: event.local_name().as_ref().to_vec(),
            attrs: event
                .attributes()
                .with_checks(false)
                .flatten()
                .map(|attribute| {
                    (
                        attribute.key.local_name().as_ref().to_vec(),
                        attribute
                            .unescape_value()
                            .map(std::borrow::Cow::into_owned)
                            .unwrap_or_default(),
                    )
                })
                .collect(),
        }
    }

    /// The value of the attribute whose local name is `name`.
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(key, _)| key.as_slice() == name.as_bytes())
            .map(|(_, value)| value.as_str())
    }
}

/// One XML event with every borrow resolved.
enum Ev {
    Start(Elem),
    Empty(Elem),
    /// Character data, with entity references already resolved.
    Text(String),
    End,
    Eof,
    /// An event that contributes nothing (CDATA, comments, declarations).
    Skip,
    /// The part can no longer be read.
    Fault,
}

/// Read the next event, resolving text, entity references and errors into owned
/// values.
fn next_event(reader: &mut Xml<'_>, buffer: &mut Vec<u8>) -> Ev {
    match reader.read_event_into(buffer) {
        Ok(Event::Start(event)) => Ev::Start(Elem::of(&event)),
        Ok(Event::Empty(event)) => Ev::Empty(Elem::of(&event)),
        Ok(Event::End(_)) => Ev::End,
        // A character that will not decode costs that text, not the part — the
        // same bargain the run text made before this walker existed.
        Ok(Event::Text(event)) => match event.xml10_content() {
            Ok(text) => Ev::Text(text.into_owned()),
            Err(_) => Ev::Skip,
        },
        Ok(Event::GeneralRef(reference)) => {
            let mut text = String::new();
            append_entity(&mut text, &reference);
            Ev::Text(text)
        }
        Ok(Event::Eof) => Ev::Eof,
        Ok(_) => Ev::Skip,
        Err(_) => Ev::Fault,
    }
}

/// Which dispatch table [`Word::content`] uses for a container's children.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ctx {
    /// Ordinary body content.
    Flow,
    /// `w:pPr`.
    ParagraphProps,
    /// `w:pPr/w:rPr` — the paragraph mark's own formatting.
    MarkProps,
    /// A run's own `w:rPr`.
    RunProps,
    /// `w:tblPr`.
    TableProps,
    /// `w:trPr`.
    RowProps,
    /// `w:tcPr`.
    CellProps,
    /// `w:sectPr`.
    SectionProps,
    /// An unknown child of a property container: it is not body text, and
    /// recursing into it as one would turn `w:tabs/w:tab` into a tab.
    Ignored,
}

// ── The walker ──────────────────────────────────────────────────

/// The state of one package walk.
struct Word {
    /// The body's relationships as the walk looks them up: relationship id ->
    /// the part it targets. A package may reference a part a million times, so
    /// the lookup is one map, not a scan per reference.
    rel_targets: HashMap<String, String>,
    footnotes: Notes,
    endnotes: Notes,
    comments: Notes,
    /// Text sinks; text always goes to the last one.
    buffers: Vec<String>,
    /// Paragraph-mark markers, one entry per open `w:p`.
    paras: Vec<Vec<&'static str>>,
    /// Open tables, innermost last, so a nested table works.
    tables: Vec<Table>,
    /// Open cells, innermost last.
    cells: Vec<Cell>,
    /// Open fields, innermost last.
    fields: Vec<Field>,
    /// Comment ids whose range was opened in the part being read, so a point
    /// reference does not repeat them; a range does not run between parts.
    comment_ranges: HashSet<String>,
    /// Whether the run's `w:rPr` being read carries an `rPrChange`.
    run_revised: bool,
    /// Header and footer parts the body's section properties reference, each
    /// with the variants that named it.
    headers: PartRefs,
    footers: PartRefs,
    /// Whether section properties are collected — the body part only.
    collect_sections: bool,
    /// How many `w:txbxContent` blocks are open.
    text_boxes: usize,
    /// How many tables the package has opened, and how many of those had no
    /// rows and were not rendered: a `Table N:` label counts their difference,
    /// so the labels are unique across the package and a nested table is
    /// numbered after the table that holds it.
    tables_opened: usize,
    tables_empty: usize,
    /// Whether the walk in progress is rendered: a duplicate copy the walk
    /// deliberately skips contributes no text, raises no marker and counts no
    /// revision.
    counting: bool,
    /// The tracked changes seen, for the summary line.
    revisions: Revisions,
}

impl Word {
    /// A walker over the package whose parts `rels` names, with the notes parts
    /// already located and read.
    fn new<R: Read + Seek>(rels: &[Relationship], archive: &mut ZipArchive<R>) -> Self {
        // The notes parts are named by relationship kind, so each is resolved
        // from the list before the list becomes the id lookup the walk needs.
        let footnotes_part = notes_part(rels, NoteKind::Footnote);
        let endnotes_part = notes_part(rels, NoteKind::Endnote);
        let comments_part = notes_part(rels, NoteKind::Comment);
        Self {
            rel_targets: super::relationship_targets(rels),
            footnotes: Notes::load(NoteKind::Footnote, &footnotes_part, archive),
            endnotes: Notes::load(NoteKind::Endnote, &endnotes_part, archive),
            comments: Notes::load(NoteKind::Comment, &comments_part, archive),
            buffers: Vec::new(),
            paras: Vec::new(),
            tables: Vec::new(),
            cells: Vec::new(),
            fields: Vec::new(),
            comment_ranges: HashSet::new(),
            run_revised: false,
            headers: PartRefs::default(),
            footers: PartRefs::default(),
            collect_sections: false,
            text_boxes: 0,
            tables_opened: 0,
            tables_empty: 0,
            counting: true,
            revisions: Revisions::default(),
        }
    }

    fn notes(&self, kind: NoteKind) -> &Notes {
        match kind {
            NoteKind::Footnote => &self.footnotes,
            NoteKind::Endnote => &self.endnotes,
            NoteKind::Comment => &self.comments,
        }
    }

    fn notes_mut(&mut self, kind: NoteKind) -> &mut Notes {
        match kind {
            NoteKind::Footnote => &mut self.footnotes,
            NoteKind::Endnote => &mut self.endnotes,
            NoteKind::Comment => &mut self.comments,
        }
    }

    /// The target part a relationship id names.
    fn rel(&self, id: &str) -> Option<&str> {
        self.rel_targets.get(id).map(String::as_str)
    }

    // ── Sinks ───────────────────────────────────────────────────

    /// Append `s` to the innermost open sink.
    fn push_str(&mut self, s: &str) {
        if let Some(sink) = self.buffers.last_mut() {
            sink.push_str(s);
        }
    }

    /// End the current line: a newline unless the sink is empty or already ends
    /// with one, so layout alone never leaves a blank line behind.
    fn end_line(&mut self) {
        if let Some(sink) = self.buffers.last_mut()
            && !sink.is_empty()
            && !sink.ends_with('\n')
        {
            sink.push('\n');
        }
    }

    /// Append a nested rendered block (a table, a text box): on a line of its
    /// own, with a line end after it.
    fn block(&mut self, text: &str) {
        self.end_line();
        self.push_str(text);
        self.end_line();
    }

    /// Append run text. Text between a field's `begin` and its `separate` is
    /// that field's instruction, which the marker reports instead; the field
    /// holds it until it closes, so a field that never closes cannot swallow
    /// the rest of the flow.
    fn push_text(&mut self, text: &str) {
        if let Some(field) = self.fields.last_mut()
            && field.pending
        {
            field.held.push_str(text);
            return;
        }
        self.push_str(text);
    }

    /// Count one element as a tracked change, unless the walk in progress is a
    /// duplicate copy that is not rendered.
    fn record(&mut self, elem: &Elem) {
        if self.counting {
            self.revisions.record(elem);
        }
    }

    // ── Parts ───────────────────────────────────────────────────

    /// Walk the body part.
    fn body(&mut self, xml: &[u8]) -> Option<String> {
        self.read_part(xml, true).ok()
    }

    /// Walk a whole part as ordinary content and return its rendered text.
    fn read_part(&mut self, xml: &[u8], collect_sections: bool) -> Result<String, Fault> {
        self.reset_part();
        self.collect_sections = collect_sections;
        let mut reader = Reader::from_reader(xml);
        // The part's own top level is a block like any other: a field it left
        // open is reported at its end.
        let result = self.render_block(&mut reader, 0);
        self.collect_sections = false;
        result
    }

    /// Drop what the part read before this one left open: its open tables,
    /// paragraph marks, fields and comment ranges belong to that part's own
    /// rendering.
    fn reset_part(&mut self) {
        self.paras.clear();
        self.tables.clear();
        self.cells.clear();
        self.fields.clear();
        self.comment_ranges.clear();
        self.text_boxes = 0;
    }

    /// Read a container's children, dispatching `Start`/`Empty` by name until
    /// the element's matching `End`.
    fn content(&mut self, reader: &mut Xml<'_>, level: usize, ctx: Ctx) -> Result<(), Fault> {
        if level >= MAX_ELEMENT_DEPTH {
            return Err(Fault);
        }
        let mut buffer = Vec::new();
        loop {
            match next_event(reader, &mut buffer) {
                Ev::Start(elem) => {
                    self.record(&elem);
                    self.element(reader, level + 1, ctx, &elem, false)?;
                }
                Ev::Empty(elem) => {
                    self.record(&elem);
                    self.element(reader, level + 1, ctx, &elem, true)?;
                }
                Ev::End | Ev::Eof => return Ok(()),
                Ev::Fault => return Err(Fault),
                // Character data belongs to `w:t`/`w:delText`/`w:instrText`
                // alone, all of which capture it themselves.
                Ev::Text(_) | Ev::Skip => {}
            }
            buffer.clear();
        }
    }

    /// Dispatch one element to the table its context selects.
    fn element(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        ctx: Ctx,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        // A property change is a formatting revision, and most of them also
        // raise the marker of the thing whose formatting changed.
        if is_property_change(&elem.name) {
            self.change_marker(ctx, &elem.name);
            return self.ignored(reader, level, empty);
        }
        match ctx {
            Ctx::Ignored | Ctx::RunProps | Ctx::TableProps => self.ignored(reader, level, empty),
            Ctx::Flow => self.body_element(reader, level, elem, empty),
            Ctx::ParagraphProps => self.paragraph_props(reader, level, elem, empty),
            Ctx::MarkProps => self.mark_props(reader, level, elem, empty),
            Ctx::RowProps => self.row_props(reader, level, elem, empty),
            Ctx::CellProps => self.cell_props(reader, level, elem, empty),
            Ctx::SectionProps => self.section_props(reader, level, elem, empty),
        }
    }

    /// Consume an element and everything under it, contributing nothing.
    fn ignored(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            Ok(())
        } else {
            self.content(reader, level + 1, Ctx::Ignored)
        }
    }

    /// Consume content the walk does not render — the duplicate half of a text
    /// box Word writes twice, or a construct read only to be dropped — so it
    /// contributes no text, raises no marker and counts no revision.
    fn discarded(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        let counting = self.counting;
        self.counting = false;
        let result = self.ignored(reader, level, empty);
        self.counting = counting;
        result
    }

    /// Read a plain container's children as ordinary content.
    fn flow(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            Ok(())
        } else {
            self.content(reader, level + 1, Ctx::Flow)
        }
    }

    // ── Body content ────────────────────────────────────────────

    /// One `w:p`: its content, then its paragraph-mark markers, then the line
    /// end. An empty element is already over — reading a matching `End` would
    /// take the enclosing element's.
    fn paragraph(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            self.end_line();
            return Ok(());
        }
        self.paras.push(Vec::new());
        let result = self.content(reader, level + 1, Ctx::Flow);
        for mark in self.paras.pop().unwrap_or_default() {
            self.push_str(mark);
        }
        self.end_line();
        result
    }

    /// The character data of a `w:t`/`w:delText`. Both are ordinary text: a
    /// `w:t` inside `w:del` is deleted text because the wrapper says so.
    fn run_text(&mut self, reader: &mut Xml<'_>, level: usize) -> Result<(), Fault> {
        let mut buffer = Vec::new();
        loop {
            match next_event(reader, &mut buffer) {
                Ev::Text(text) => self.push_text(&text),
                Ev::Start(elem) => {
                    self.record(&elem);
                    self.ignored(reader, level, false)?;
                }
                Ev::Empty(elem) => self.record(&elem),
                Ev::End | Ev::Eof => return Ok(()),
                Ev::Fault => return Err(Fault),
                Ev::Skip => {}
            }
            buffer.clear();
        }
    }

    /// Dispatch one element of ordinary content.
    fn body_element(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        match elem.name.as_slice() {
            b"p" => self.paragraph(reader, level, empty),
            b"t" | b"delText" => {
                if empty {
                    Ok(())
                } else {
                    self.run_text(reader, level)
                }
            }
            b"tab" | b"br" | b"cr" | b"noBreakHyphen" | b"softHyphen" | b"sym" => {
                self.leaf_text(elem);
                self.ignored(reader, level, empty)
            }
            b"tbl" => self.table(reader, level, empty),
            b"tr" if !self.tables.is_empty() => self.row(reader, level, empty),
            b"tc" if !self.tables.is_empty() => self.cell(reader, level, empty),
            b"pPr" => self.props(reader, level, Ctx::ParagraphProps, empty),
            b"rPr" => self.run_props(reader, level, empty),
            b"tblPr" => self.props(reader, level, Ctx::TableProps, empty),
            b"tblGrid" => self.table_grid(reader, level, empty),
            // The row's own properties, and the table-property exceptions Word
            // writes beside `w:trPr` rather than inside it: the row reports
            // both.
            b"trPr" | b"tblPrEx" => self.props(reader, level, Ctx::RowProps, empty),
            b"tcPr" => self.props(reader, level, Ctx::CellProps, empty),
            b"sectPr" => self.props(reader, level, Ctx::SectionProps, empty),
            b"ins" | b"del" | b"moveTo" | b"moveFrom" => self.revision(reader, level, elem, empty),
            b"footnoteReference" => {
                self.note_reference(elem, NoteKind::Footnote);
                self.ignored(reader, level, empty)
            }
            b"endnoteReference" => {
                self.note_reference(elem, NoteKind::Endnote);
                self.ignored(reader, level, empty)
            }
            b"commentRangeStart" | b"commentRangeEnd" | b"commentReference" => {
                self.comment_marker(elem);
                self.ignored(reader, level, empty)
            }
            b"fldSimple" => self.simple_field(reader, level, elem, empty),
            b"fldChar" => {
                self.field_char(elem);
                self.ignored(reader, level, empty)
            }
            b"instrText" => self.instr_text(reader, level, empty),
            // `w:delInstrText` is part of a deleted field instruction, and
            // `w:numberingChange` names a list change: neither is text. The
            // `w:sdt` placeholder properties hold no content either.
            b"delInstrText" | b"numberingChange" | b"sdtPr" | b"sdtEndPr" => {
                self.ignored(reader, level, empty)
            }
            b"txbxContent" => self.text_box(reader, level, empty),
            b"AlternateContent" => self.alternate_content(reader, level, empty),
            // Everything else is a plain container (`w:document`, `w:body`,
            // `w:r`, `w:hyperlink`, `w:smartTag`, `w:sdt`, `w:drawing`,
            // `w:pict`, the bookmarks, …): its children are body content too.
            _ => self.flow(reader, level, empty),
        }
    }

    /// The text a leaf run element contributes: `w:softHyphen` and `w:sym` are
    /// formatting marks with nothing readable to show, a tab is a tab, and a
    /// line break is a line end.
    fn leaf_text(&mut self, elem: &Elem) {
        match elem.name.as_slice() {
            b"tab" => self.push_str("\t"),
            b"br" | b"cr" => self.end_line(),
            b"noBreakHyphen" => self.push_str("-"),
            _ => {}
        }
    }

    // ── Properties ──────────────────────────────────────────────

    /// Read a property container's children with the dispatch they need.
    fn props(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        ctx: Ctx,
        empty: bool,
    ) -> Result<(), Fault> {
        if empty {
            Ok(())
        } else {
            self.content(reader, level + 1, ctx)
        }
    }

    /// `w:pPr`: the paragraph mark's formatting, and the section properties a
    /// paragraph can carry.
    fn paragraph_props(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        match elem.name.as_slice() {
            b"rPr" => self.props(reader, level, Ctx::MarkProps, empty),
            b"sectPr" => self.props(reader, level, Ctx::SectionProps, empty),
            _ => self.ignored(reader, level, empty),
        }
    }

    /// `w:pPr/w:rPr`: an insertion or deletion here is the paragraph mark's
    /// own, marked at the paragraph's end.
    fn mark_props(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        match elem.name.as_slice() {
            b"ins" => self.para_mark("[ins ¶]"),
            b"del" => self.para_mark("[del ¶]"),
            b"moveTo" => self.para_mark("[moved-here ¶]"),
            b"moveFrom" => self.para_mark("[moved-away ¶]"),
            _ => {}
        }
        self.ignored(reader, level, empty)
    }

    /// A run's own `w:rPr`: an `rPrChange` inside it marks the run's formatting
    /// right before the run's text.
    fn run_props(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        self.run_revised = false;
        let result = self.props(reader, level, Ctx::RunProps, empty);
        if self.run_revised {
            self.push_str("[fmt]");
        }
        result
    }

    /// `w:trPr`: the row's own revision, and the `w:gridBefore` that skips grid
    /// columns before its first cell.
    fn row_props(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        match elem.name.as_slice() {
            b"ins" | b"moveTo" => self.row_mark(Mark::Inserted),
            b"del" | b"moveFrom" => self.row_mark(Mark::Deleted),
            // `w:gridBefore` skips grid columns before the row's first cell.
            b"gridBefore" => {
                if let Some(table) = self.tables.last_mut() {
                    table.column = table
                        .column
                        .saturating_add(numeric_attr(elem, MAX_TABLE_COLUMNS))
                        .min(MAX_TABLE_COLUMNS);
                }
            }
            // Neither schema gives a row's mark properties (`w:rPr` is a child
            // of `w:pPr` alone), so a stray producer's `w:rPr` here is read and
            // dropped whole rather than taken for the row's.
            b"rPr" => return self.discarded(reader, level, empty),
            _ => {}
        }
        self.ignored(reader, level, empty)
    }

    /// `w:tcPr`: the cell's span, its merges and its own revision.
    fn cell_props(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        let restart = elem.attr("val") == Some("restart");
        match elem.name.as_slice() {
            b"gridSpan" => {
                if let Some(cell) = self.cells.last_mut() {
                    cell.span = numeric_attr(elem, MAX_TABLE_COLUMNS).max(1);
                }
            }
            b"vMerge" => {
                if let Some(cell) = self.cells.last_mut() {
                    cell.merge = if restart {
                        Merge::Restart
                    } else {
                        Merge::Continue
                    };
                }
            }
            b"hMerge" => {
                if let Some(cell) = self.cells.last_mut() {
                    cell.hmerge = if restart {
                        Merge::Restart
                    } else {
                        Merge::Continue
                    };
                }
            }
            b"cellIns" => self.cell_mark(Mark::Inserted),
            b"cellDel" => self.cell_mark(Mark::Deleted),
            b"cellMerge" => self.cell_mark(Mark::MergeRevised),
            _ => {}
        }
        self.ignored(reader, level, empty)
    }

    /// `w:sectPr`: record the header/footer parts it names, and mark a
    /// `sectPrChange`. A section inside a text box is the box's own layout, not
    /// the document's, so it names no parts.
    fn section_props(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        match elem.name.as_slice() {
            b"headerReference" => self.reference_part(elem, true),
            b"footerReference" => self.reference_part(elem, false),
            _ => {}
        }
        self.ignored(reader, level, empty)
    }

    /// The marker a property change raises on the thing whose formatting
    /// changed. A copy the walk is not rendering — the duplicate half of a text
    /// box — raises none.
    fn change_marker(&mut self, ctx: Ctx, name: &[u8]) {
        if !self.counting {
            return;
        }
        match ctx {
            Ctx::RunProps => self.run_revised = true,
            Ctx::MarkProps | Ctx::ParagraphProps => self.para_mark("[fmt ¶]"),
            // A paragraph carries its section's mark at its own end; the
            // document's last section is carried by no paragraph.
            Ctx::SectionProps => self.section_mark(),
            Ctx::TableProps => self.table_mark(),
            Ctx::RowProps => self.row_mark(Mark::FormatRevised),
            Ctx::CellProps => self.cell_mark(Mark::FormatRevised),
            // The one property change a table holds outside `w:tblPr`: a
            // `w:tblGridChange` written beside its `w:tblGrid` rather than
            // inside it is still that table's.
            _ if name == b"tblGridChange" => self.table_mark(),
            _ => {}
        }
    }

    /// Mark the innermost open table as formatting-revised.
    fn table_mark(&mut self) {
        if let Some(table) = self.tables.last_mut() {
            table.format_revised = true;
        }
    }

    /// The marker a section's revision carries: at the end of the paragraph
    /// whose properties hold the section, or — for the document's own last
    /// section, whose `w:body/w:sectPr` no paragraph holds — in the flow at the
    /// point the properties were read.
    fn section_mark(&mut self) {
        if self.paras.is_empty() {
            self.push_str("[fmt section]");
        } else {
            self.para_mark("[fmt section]");
        }
    }

    /// The marker a paragraph mark carries, at the end of the paragraph being
    /// read.
    fn para_mark(&mut self, mark: &'static str) {
        if let Some(marks) = self.paras.last_mut() {
            marks.push(mark);
        }
    }

    fn row_mark(&mut self, mark: Mark) {
        if let Some(table) = self.tables.last_mut() {
            table.row_marks.push(mark);
        }
    }

    fn cell_mark(&mut self, mark: Mark) {
        if let Some(cell) = self.cells.last_mut() {
            cell.marks.push(mark);
        }
    }

    // ── Tables ──────────────────────────────────────────────────

    /// One `w:tbl`: buffered in its own record and appended to the enclosing
    /// sink when it closes, as one block.
    fn table(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        // The number is taken when the table opens, so a table nested inside
        // another is numbered after it, not after the rows it happens to close
        // first.
        self.tables.push(Table {
            number: self.tables_opened + 1 - self.tables_empty,
            ..Table::default()
        });
        self.tables_opened += 1;
        let result = self.content(reader, level + 1, Ctx::Flow);
        self.finish_table();
        result
    }

    /// Close the innermost table: render its label and rows, and append them to
    /// the sink below. A table with no rows has nothing to say, and its number
    /// passes to the next table.
    fn finish_table(&mut self) {
        let Some(table) = self.tables.pop() else {
            return;
        };
        if table.rows.is_empty() {
            self.tables_empty += 1;
            return;
        }
        let columns = table.columns.max(table.grid);
        let label = if table.format_revised {
            format!("Table {} (formatting revised)", table.number)
        } else {
            format!("Table {}", table.number)
        };
        let mut text = format!(
            "{label}: {}, {}",
            plural(table.rows.len(), "row"),
            plural(columns, "column")
        );
        for row in &table.rows {
            text.push('\n');
            text.push_str(row);
        }
        self.block(&text);
    }

    /// Count a `w:tblGrid`'s declared columns.
    fn table_grid(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        let mut buffer = Vec::new();
        let mut columns = 0;
        loop {
            match next_event(reader, &mut buffer) {
                Ev::Start(elem) => {
                    self.record(&elem);
                    if self.grid_child(&elem) {
                        columns += 1;
                    }
                    self.ignored(reader, level + 1, false)?;
                }
                Ev::Empty(elem) => {
                    self.record(&elem);
                    if self.grid_child(&elem) {
                        columns += 1;
                    }
                }
                Ev::End | Ev::Eof => break,
                Ev::Fault => return Err(Fault),
                Ev::Text(_) | Ev::Skip => {}
            }
            buffer.clear();
        }
        if let Some(table) = self.tables.last_mut() {
            table.grid = columns.min(MAX_TABLE_COLUMNS);
        }
        Ok(())
    }

    /// One `w:tblGrid` child: a `w:gridCol` widens the declared grid, and a
    /// `w:tblGridChange` — whose revision is real however empty the element is —
    /// marks the table revised. Returns whether the child is a column.
    fn grid_child(&mut self, elem: &Elem) -> bool {
        match elem.name.as_slice() {
            b"gridCol" => true,
            b"tblGridChange" => {
                self.table_mark();
                false
            }
            _ => false,
        }
    }

    /// One `w:tr`: reset the running column counter, read the cells, then
    /// render the row's line.
    fn row(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        if let Some(table) = self.tables.last_mut() {
            table.column = 1;
            table.hmerge = None;
            table.cells.clear();
            table.row_marks.clear();
        }
        let result = self.content(reader, level + 1, Ctx::Flow);
        if let Some(table) = self.tables.last_mut() {
            let mut line = format!("  R{}", table.rows.len() + 1);
            if !table.row_marks.is_empty() {
                // A row the document marked twice in two ways — its own
                // properties and the table-property exception beside them —
                // still names each change once.
                table.row_marks.sort();
                table.row_marks.dedup();
                let names: Vec<&str> = table.row_marks.iter().map(|mark| mark.text()).collect();
                let _ = write!(line, " ({})", names.join(", "));
            }
            line.push(':');
            if !table.cells.is_empty() {
                line.push(' ');
                line.push_str(&table.cells.join(" | "));
            }
            table.rows.push(line);
        }
        result
    }

    /// One `w:tc`: read its content into its own sink, then place it in the
    /// table's geometry.
    fn cell(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        let column = self.tables.last().map_or(1, |table| table.column);
        self.cells.push(Cell {
            column,
            span: 1,
            ..Cell::default()
        });
        let text = self.render_block(reader, level);
        // The cell is popped whatever happened: a fault must not leave it on
        // the stack for the next cell to be read as.
        let cell = self.cells.pop().unwrap_or_default();
        let text = text?;
        if let Some(table) = self.tables.last_mut() {
            table.place(cell, &text);
        }
        Ok(())
    }

    // ── Revisions ───────────────────────────────────────────────

    /// A run-level revision (`w:ins`/`w:del`/`w:moveTo`/`w:moveFrom`): its
    /// content, wrapped in its markers. A wrapper that produced no text at all
    /// is still counted, but emits nothing.
    fn revision(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        let (open, close) = match elem.name.as_slice() {
            b"ins" => ("[ins]", "[/ins]"),
            b"del" => ("[del]", "[/del]"),
            b"moveTo" => ("[moved-here]", "[/moved-here]"),
            _ => ("[moved-away]", "[/moved-away]"),
        };
        let before = self.buffers.last().map_or(0, String::len);
        let result = self.content(reader, level + 1, Ctx::Flow);
        if let Some(sink) = self.buffers.last_mut()
            && sink.len() > before
        {
            sink.insert_str(before, open);
            sink.push_str(close);
        }
        result
    }

    // ── Notes and comments ──────────────────────────────────────

    /// `w:footnoteReference`/`w:endnoteReference`: the note's number at the
    /// point it is anchored.
    fn note_reference(&mut self, elem: &Elem, kind: NoteKind) {
        let Some(id) = elem.attr("id") else {
            return;
        };
        let number = self.notes_mut(kind).reference(id);
        let marker = note_marker(kind.marker(), number, "");
        self.push_str(&marker);
    }

    /// A comment's range or point marker. A comment whose range was opened is
    /// not reported again at its point reference.
    fn comment_marker(&mut self, elem: &Elem) {
        let Some(id) = elem.attr("id") else {
            return;
        };
        let suffix = match elem.name.as_slice() {
            b"commentRangeStart" => {
                self.comment_ranges.insert(id.to_owned());
                " starts"
            }
            b"commentRangeEnd" => " ends",
            _ if self.comment_ranges.contains(id) => return,
            _ => "",
        };
        let number = self.notes_mut(NoteKind::Comment).reference(id);
        let marker = note_marker(NoteKind::Comment.marker(), number, suffix);
        self.push_str(&marker);
    }

    // ── Fields ──────────────────────────────────────────────────

    /// `w:fldSimple`: its instruction's marker, then its cached result.
    fn simple_field(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        let marker = elem
            .attr("instr")
            .map_or_else(|| "[field]".to_owned(), field_marker);
        self.push_str(&marker);
        self.flow(reader, level, empty)
    }

    /// A complex field's `w:fldChar`: its instruction is reported at the
    /// `separate`, or when the field closes without one.
    fn field_char(&mut self, elem: &Elem) {
        match elem.attr("fldCharType") {
            Some("begin") => self.fields.push(Field {
                pending: true,
                ..Field::default()
            }),
            Some("separate") => {
                let Some(field) = self.fields.last_mut() else {
                    return;
                };
                field.pending = false;
                // What was held back was the field's own code, which the marker
                // reports instead of the text.
                field.held.clear();
                let marker = field_marker(&field.instr);
                self.push_str(&marker);
            }
            Some("end") => {
                let Some(field) = self.fields.pop() else {
                    return;
                };
                if field.pending {
                    self.close_field(&field);
                }
            }
            _ => {}
        }
    }

    /// Report a field that closed without a `separate`: its marker, then the
    /// text it was holding.
    fn close_field(&mut self, field: &Field) {
        let marker = field_marker(&field.instr);
        self.push_str(&marker);
        self.push_str(&field.held);
    }

    /// Report the fields a flow left open. A `w:fldChar begin` without its
    /// `separate` or its `end` must not go on swallowing the flow's text as its
    /// instruction.
    fn close_fields(&mut self) {
        for field in std::mem::take(&mut self.fields) {
            if field.pending {
                self.close_field(&field);
            }
        }
    }

    /// `w:instrText`: one piece of the open field's instruction. Word splits a
    /// single instruction across runs, so the pieces are concatenated and only
    /// collapsed when the field is reported.
    fn instr_text(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        let mut buffer = Vec::new();
        let mut text = String::new();
        loop {
            match next_event(reader, &mut buffer) {
                Ev::Text(piece) => text.push_str(&piece),
                Ev::Start(_) => self.ignored(reader, level + 1, false)?,
                Ev::End | Ev::Eof => break,
                Ev::Fault => return Err(Fault),
                Ev::Empty(_) | Ev::Skip => {}
            }
            buffer.clear();
        }
        if let Some(field) = self.fields.last_mut() {
            field.instr.push_str(&text);
        }
        Ok(())
    }

    // ── Text boxes ──────────────────────────────────────────────

    /// `w:txbxContent`: rendered as an indented `[text box]` block in the
    /// enclosing flow.
    fn text_box(&mut self, reader: &mut Xml<'_>, level: usize, empty: bool) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        self.text_boxes += 1;
        let inner = self.render_block(reader, level);
        self.text_boxes = self.text_boxes.saturating_sub(1);
        let inner = inner?.trim_end().to_owned();
        if inner.trim().is_empty() {
            return Ok(());
        }
        self.block(&text_lines(TEXT_BOX, &lines_of(&inner)));
        Ok(())
    }

    /// `mc:AlternateContent`: a text box is written twice, as an `mc:Choice`
    /// and as the `mc:Fallback` beside it. The first choice that renders
    /// anything wins; the fallback is read only when none of them renders
    /// anything at all.
    fn alternate_content(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        empty: bool,
    ) -> Result<(), Fault> {
        if empty {
            return Ok(());
        }
        let mut buffer = Vec::new();
        let mut choice: Option<String> = None;
        let mut fallback: Option<String> = None;
        // Every copy holds the same content, so only the copy the walk keeps
        // counts its revisions: a copy that turned out to render nothing is
        // rolled back before the next one is read, and at the end when no copy
        // rendered anything.
        let base = self.revisions.mark();
        loop {
            match next_event(reader, &mut buffer) {
                // Every choice and fallback is consumed at the depth a child of
                // this element has. A later one carries the text of one that had
                // none, and the fallback speaks only when no choice rendered
                // anything: Word writes the same content in each copy.
                Ev::Start(elem) if elem.name == b"Choice" => {
                    if renders_nothing(choice.as_deref()) {
                        self.revisions.rollback(base);
                        choice = Some(self.render_block(reader, level + 1)?);
                    } else {
                        self.discarded(reader, level + 1, false)?;
                    }
                }
                Ev::Start(elem) if elem.name == b"Fallback" => {
                    if renders_nothing(fallback.as_deref()) && renders_nothing(choice.as_deref()) {
                        self.revisions.rollback(base);
                        fallback = Some(self.render_block(reader, level + 1)?);
                    } else {
                        self.discarded(reader, level + 1, false)?;
                    }
                }
                Ev::Start(_) => self.discarded(reader, level + 1, false)?,
                Ev::End | Ev::Eof => break,
                Ev::Fault => return Err(Fault),
                Ev::Text(_) | Ev::Empty(_) | Ev::Skip => {}
            }
            buffer.clear();
        }
        // The copy that carries the text: the choice that rendered something, or
        // the fallback beside it.
        let kept = if renders_nothing(choice.as_deref()) {
            fallback
        } else {
            choice
        };
        match kept {
            Some(text) if !renders_nothing(Some(text.as_str())) => self.block(text.trim()),
            // Nothing is rendered at all, so no copy's revisions are part of the
            // document's text either.
            _ => self.revisions.rollback(base),
        }
        Ok(())
    }

    /// Render the element at `level` into its own sink and return the text: a
    /// cell, a text box, a note definition, or a whole part (whose top level
    /// sits below its document element). The block is a text flow of its own,
    /// so a field it left open is closed with it.
    fn render_block(&mut self, reader: &mut Xml<'_>, level: usize) -> Result<String, Fault> {
        self.buffers.push(String::new());
        let result = self.content(reader, level + 1, Ctx::Flow);
        self.close_fields();
        // The sink pushed above is the one popped: a nested block always pops
        // its own, fault or not, so the stack cannot grow past this block.
        let text = self.buffers.pop().unwrap_or_default();
        result.map(|()| text)
    }

    // ── Headers, footers and notes ──────────────────────────────

    /// Record a header/footer reference: the part it names, and the variant it
    /// was named under. A part serving several variants is read once and keeps
    /// them all, and the parts are held in first-encounter order with a position
    /// index, so a package that names its parts a million times costs one lookup
    /// per reference rather than a scan of everything named before it.
    fn reference_part(&mut self, elem: &Elem, header: bool) {
        if !self.collect_sections || self.text_boxes > 0 {
            return;
        }
        let label = match elem.attr("type") {
            Some("first") => "first page",
            Some("even") => "even pages",
            _ => "default",
        };
        let Some(target) = elem
            .attr("id")
            .and_then(|id| self.rel(id))
            .map(str::to_owned)
        else {
            return;
        };
        let part = resolve_part(DOCX_BASE, &target);
        let list = if header {
            &mut self.headers
        } else {
            &mut self.footers
        };
        list.record(part, label);
    }

    /// Render the parts the body's section properties reference, then read and
    /// render the notes parts: the blocks that follow the body text.
    fn extra_sections<R: Read + Seek>(&mut self, archive: &mut ZipArchive<R>) -> Vec<String> {
        let headers = std::mem::take(&mut self.headers);
        let footers = std::mem::take(&mut self.footers);
        let mut blocks = self.part_blocks(archive, &headers.parts, true);
        blocks.extend(self.part_blocks(archive, &footers.parts, false));
        for kind in [NoteKind::Footnote, NoteKind::Endnote, NoteKind::Comment] {
            self.read_definitions(kind, archive);
            blocks.extend(self.note_blocks(kind));
        }
        blocks
    }

    /// The blocks of the parts `refs` names, in first-encounter order. A part
    /// that is missing, unreadable or empty costs only itself; the ordinal
    /// appears only when more than one block is emitted.
    fn part_blocks<R: Read + Seek>(
        &mut self,
        archive: &mut ZipArchive<R>,
        refs: &[PartRef],
        header: bool,
    ) -> Vec<String> {
        let mut rendered: Vec<(&[&'static str], String)> = Vec::new();
        for entry in refs {
            let Some(xml) = read_zip_entry(archive, &entry.part).bytes() else {
                continue;
            };
            // A part that shows nothing shows nothing of its revisions either.
            let mark = self.revisions.mark();
            match self.read_part(&xml, false) {
                Ok(text) if !text.trim().is_empty() => rendered.push((&entry.labels, text)),
                _ => self.revisions.rollback(mark),
            }
        }
        let name = if header { WORD_HEADER } else { WORD_FOOTER };
        let numbered = rendered.len() > 1;
        rendered
            .into_iter()
            .enumerate()
            .map(|(index, (labels, text))| {
                let label = if numbered {
                    format!("{name} {}", index + 1)
                } else {
                    name.to_owned()
                };
                labeled_block(&format!("{label} ({})", labels.join(", ")), &text)
            })
            .collect()
    }

    /// Walk a notes part, rendering every definition it holds into its own
    /// buffer. A part that is missing or will not parse is skipped, and so is
    /// whatever the part before it left open.
    fn read_definitions<R: Read + Seek>(&mut self, kind: NoteKind, archive: &mut ZipArchive<R>) {
        // The part is read here rather than held since the body was walked: one
        // notes part at a time is resident, like every other part.
        let part = self.notes(kind).part.clone();
        let Some(xml) = read_zip_entry(archive, &part).bytes() else {
            return;
        };
        self.reset_part();
        let mut reader = Reader::from_reader(xml.as_slice());
        let _ = self.definitions(&mut reader, 0, kind);
    }

    /// The definitions of a notes part, in part order.
    fn definitions(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        kind: NoteKind,
    ) -> Result<(), Fault> {
        if level >= MAX_ELEMENT_DEPTH {
            return Err(Fault);
        }
        let mut buffer = Vec::new();
        loop {
            match next_event(reader, &mut buffer) {
                Ev::Start(elem) if elem.name == kind.element() => {
                    self.definition(reader, level + 1, kind, &elem, false)?;
                }
                Ev::Empty(elem) if elem.name == kind.element() => {
                    self.definition(reader, level + 1, kind, &elem, true)?;
                }
                Ev::Start(_) => self.definitions(reader, level + 1, kind)?,
                Ev::End | Ev::Eof => return Ok(()),
                Ev::Fault => return Err(Fault),
                Ev::Text(_) | Ev::Empty(_) | Ev::Skip => {}
            }
            buffer.clear();
        }
    }

    /// One `w:footnote`/`w:endnote`/`w:comment`. A notes part's separator and
    /// continuationSeparator entries are not shown, and neither is a definition
    /// that renders nothing and that the flow never references.
    fn definition(
        &mut self,
        reader: &mut Xml<'_>,
        level: usize,
        kind: NoteKind,
        elem: &Elem,
        empty: bool,
    ) -> Result<(), Fault> {
        let id = elem.attr("id").unwrap_or_default().to_owned();
        let author = elem.attr("author").map(str::to_owned);
        if !keeps_definition(kind, &id, elem.attr("type")) {
            return self.ignored(reader, level, empty);
        }
        let mark = self.revisions.mark();
        let text = if empty {
            String::new()
        } else {
            match self.render_block(reader, level) {
                Ok(text) => text.trim_end().to_owned(),
                // The fault drops the definition with the rest of the part, so
                // the revisions read out of it are not the document's either.
                Err(fault) => {
                    self.revisions.rollback(mark);
                    return Err(fault);
                }
            }
        };
        // A definition the flow never references is shown only when it has text:
        // an empty one would be a label with nothing under it, while a referenced
        // one must appear or its anchor would name a note with no block.
        if text.trim().is_empty() && !self.notes(kind).referenced.contains(&id) {
            // One that is not shown contributes no revision either.
            self.revisions.rollback(mark);
            return Ok(());
        }
        let notes = self.notes_mut(kind);
        let number = notes.number(&id);
        notes.defs.push(Definition {
            id,
            author,
            text,
            number,
        });
        Ok(())
    }

    /// The `Footnote`/`Endnote`/`Comment` blocks of a notes part: by assigned
    /// number, with the definitions nothing references last.
    fn note_blocks(&self, kind: NoteKind) -> Vec<String> {
        let notes = self.notes(kind);
        let mut defs: Vec<&Definition> = notes.defs.iter().collect();
        defs.sort_by_key(|definition| definition.number);
        defs.into_iter()
            .map(|definition| {
                let mut extras: Vec<String> = Vec::new();
                if let Some(author) = &definition.author {
                    extras.push(author.clone());
                }
                if !notes.referenced.contains(&definition.id) {
                    extras.push("not referenced in the text".to_owned());
                }
                let mut label = format!("{} {}", kind.label(), definition.number);
                if !extras.is_empty() {
                    let _ = write!(label, " ({})", extras.join(", "));
                }
                // A definition with no text is still a block: its anchor points
                // at it.
                if definition.text.is_empty() {
                    return format!("{label}:");
                }
                labeled_block(&label, &definition.text)
            })
            .collect()
    }

    /// The `Unaccepted tracked changes: …` line, when the document carries any.
    fn revision_summary(&self) -> Option<String> {
        let tallies = self.revisions.tallies;
        let mut parts: Vec<String> = Vec::new();
        for (count, noun) in [
            (tallies.insertions, "insertion"),
            (tallies.deletions, "deletion"),
            (tallies.moves(), "move"),
            (tallies.formatting, "formatting change"),
        ] {
            if count > 0 {
                parts.push(plural(count, noun));
            }
        }
        if parts.is_empty() {
            return None;
        }
        let mut line = format!("Unaccepted tracked changes: {}", parts.join(", "));
        if !self.revisions.authors.is_empty() {
            let _ = write!(line, " (authors: {})", self.revisions.authors.join(", "));
        }
        line.push('.');
        Some(line)
    }
}

// ── Table geometry ──────────────────────────────────────────────

/// A table being read: the rows already rendered, the geometry of the row being
/// read, and the vertical merges that reach into it.
#[derive(Default)]
struct Table {
    /// The rendered row lines.
    rows: Vec<String>,
    /// The `Table N:` number, taken when the table opens.
    number: usize,
    /// The column count, widened by the widest column covered.
    columns: usize,
    /// The number of `w:tblGrid/w:gridCol` children, at most
    /// [`MAX_TABLE_COLUMNS`]: the declared grid is document-controlled and
    /// decides nothing but the label.
    grid: usize,
    /// The running column counter within the current row.
    column: usize,
    /// The vertical merge origin of each column: `Some((row, column))` while a
    /// merge started above still covers it.
    merges: Vec<Option<(usize, usize)>>,
    /// The cell tokens of the row being read.
    cells: Vec<String>,
    /// The annotations of the row being read.
    row_marks: Vec<Mark>,
    /// The origin column of this row's `hMerge` restart, if it has one.
    hmerge: Option<usize>,
    /// Whether `w:tblPrChange` was seen.
    format_revised: bool,
}

impl Table {
    /// Place the cell that just closed: annotate it, advance the running
    /// column, and leave the vertical merge map ready for the row below.
    fn place(&mut self, cell: Cell, text: &str) {
        let row = self.rows.len() + 1;
        // One cell's span was clamped where it was read, but a row's cells add
        // up: the running column is document-controlled arithmetic too, so the
        // range is clamped here — never past [`MAX_TABLE_COLUMNS`] — and the
        // merge map never grows past that either.
        let column = cell.column.min(MAX_TABLE_COLUMNS);
        let span = cell.span.max(1).min(MAX_TABLE_COLUMNS + 1 - column);
        // A continuation is not a value of its own: it names the cell whose
        // value it carries, and that value is not repeated here.
        let mut holds_value = true;
        let mut origin: Option<(usize, usize)> = None;
        match cell.merge {
            // `restart` opens a merge over every column it covers; the map
            // keeps its origin for the rows below.
            Merge::Restart => {
                for covered in column..column + span {
                    set_merge(&mut self.merges, covered, Some((row, column)));
                }
            }
            Merge::Continue => {
                origin = self.merges.get(column).copied().flatten();
                holds_value = origin.is_none();
            }
            Merge::None => {
                for covered in column..column + span {
                    set_merge(&mut self.merges, covered, None);
                }
                match cell.hmerge {
                    Merge::Restart => self.hmerge = Some(column),
                    Merge::Continue => {
                        origin = self.hmerge.map(|origin| (row, origin));
                        holds_value = origin.is_none();
                    }
                    Merge::None => {}
                }
            }
        }
        // The annotations read in one fixed order, whatever order the
        // properties that raised them were written in: the merge a cell
        // continues, the columns it spans, then its own marks.
        let mut notes: Vec<String> = Vec::new();
        if let Some((origin_row, origin_column)) = origin {
            notes.push(format!("part of R{origin_row}C{origin_column}"));
        }
        if span > 1 {
            notes.push(format!("spans C{column}-C{}", column + span - 1));
        }
        let mut marks = cell.marks;
        marks.sort();
        marks.dedup();
        notes.extend(marks.iter().map(|mark| mark.text().to_owned()));
        // A row is one line, so a cell's own line ends are spelled as spaces.
        let value = if holds_value {
            text.trim().replace('\n', " ")
        } else {
            String::new()
        };
        let mut token = format!("C{column}");
        if !notes.is_empty() {
            let _ = write!(token, " ({})", notes.join(", "));
        }
        token.push(':');
        if !value.is_empty() {
            token.push(' ');
            token.push_str(&value);
        }
        self.cells.push(token);
        self.column = column + span;
        self.columns = self.columns.max(column + span - 1);
    }
}

/// Record `origin` — or the absence of one — against `column`, growing the map.
/// [`Table::place`] clamps the column and the span it covers, so the map is at
/// most [`MAX_TABLE_COLUMNS`] + 1 entries.
fn set_merge(
    merges: &mut Vec<Option<(usize, usize)>>,
    column: usize,
    origin: Option<(usize, usize)>,
) {
    if column >= merges.len() {
        merges.resize(column + 1, None);
    }
    merges[column] = origin;
}

/// A revision a table's row or one of its cells carries, in the fixed order a
/// line's annotations read in, whatever order the properties that raised them
/// were written in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Mark {
    Inserted,
    Deleted,
    MergeRevised,
    FormatRevised,
}

impl Mark {
    /// The word the annotation spells out.
    fn text(self) -> &'static str {
        match self {
            Self::Inserted => "inserted",
            Self::Deleted => "deleted",
            Self::MergeRevised => "merge revised",
            Self::FormatRevised => "formatting revised",
        }
    }
}

/// The cell being read.
#[derive(Default)]
struct Cell {
    /// The column the cell starts at, taken when it begins.
    column: usize,
    /// The column span, from `w:tcPr/w:gridSpan`, at most
    /// [`MAX_TABLE_COLUMNS`].
    span: usize,
    /// `w:tcPr/w:vMerge`.
    merge: Merge,
    /// `w:tcPr/w:hMerge`, the Transitional schema's horizontal merge: Word 2003
    /// XML writes it where the later output writes `w:gridSpan`.
    hmerge: Merge,
    /// The cell's own annotations.
    marks: Vec<Mark>,
}

/// A merge state, vertical or horizontal: `w:vMerge w:val="restart"` opens a
/// merge, and a bare `w:vMerge` (or `w:val="continue"`) continues the one above
/// — the transitional `w:hMerge` reads the same way, one column over.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Merge {
    #[default]
    None,
    Restart,
    Continue,
}

// ── Notes and comments ──────────────────────────────────────────

/// The three kinds of definition that live in their own part.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NoteKind {
    Footnote,
    Endnote,
    Comment,
}

impl NoteKind {
    /// The definition element's local name.
    fn element(self) -> &'static [u8] {
        match self {
            Self::Footnote => b"footnote",
            Self::Endnote => b"endnote",
            Self::Comment => b"comment",
        }
    }

    /// The label a definition block is headed with.
    fn label(self) -> &'static str {
        match self {
            Self::Footnote => WORD_FOOTNOTE,
            Self::Endnote => WORD_ENDNOTE,
            Self::Comment => WORD_COMMENT,
        }
    }

    /// The kind's own word: the one inside a reference marker (`[footnote 1]`,
    /// `[comment 2 ends]`), and the name its shared entry is keyed by — the entry
    /// states the relationship type its part is named by and the conventional part
    /// name used when the relationship list names none (see
    /// [`crate::docgen::docx_peripheral_part`]), which the document kit's refusals
    /// read for the same parts.
    fn marker(self) -> &'static str {
        match self {
            Self::Footnote => "footnote",
            Self::Endnote => "endnote",
            Self::Comment => "comment",
        }
    }
}

/// The part holding one kind of notes: the one the relationship list names by
/// type, or the conventional name the kind's shared entry carries.
fn notes_part(rels: &[Relationship], kind: NoteKind) -> String {
    let entry = crate::docgen::docx_peripheral_part(kind.marker());
    rels.iter()
        .find(|rel| rel.kind.ends_with(&entry.rel))
        .map_or_else(
            || {
                entry
                    .part
                    .clone()
                    .expect("a note kind's shared entry names its conventional part")
            },
            |rel| resolve_part(DOCX_BASE, &rel.target),
        )
}

/// One notes part: the definitions it holds, the numbering its references
/// assigned, and the ids it defines at all.
#[derive(Default)]
struct Notes {
    /// The part's definitions, in part order, once rendered.
    defs: Vec<Definition>,
    /// The ids this part defines, with the entries the walk skips left out, so
    /// a reference to one of those can say `?`.
    defined: HashSet<String>,
    /// Definition id -> the number assigned to it.
    numbers: HashMap<String, usize>,
    /// The ids a reference named, as opposed to a definition alone.
    referenced: HashSet<String>,
    /// The next free number.
    next: usize,
    /// The part the definitions live in, read up front for the ids it defines
    /// and again, after the body, for the definitions themselves.
    part: String,
}

impl Notes {
    /// Note the ids `part` defines, so a reference in the body can say `?` and
    /// be numbered; the definitions themselves are read after the body.
    fn load<R: Read + Seek>(kind: NoteKind, part: &str, archive: &mut ZipArchive<R>) -> Self {
        let mut notes = Self {
            part: part.to_owned(),
            ..Self::default()
        };
        if let Some(xml) = read_zip_entry(archive, part).bytes() {
            notes.defined = definition_ids(&xml, kind).into_iter().collect();
        }
        notes
    }

    /// The number `id` is displayed under at a reference: the one it already
    /// has, or the next free one. `None` when the part defines no such entry, so
    /// a reference to nothing can say `?`.
    fn reference(&mut self, id: &str) -> Option<usize> {
        if !self.defined.contains(id) {
            return None;
        }
        self.referenced.insert(id.to_owned());
        Some(self.number(id))
    }

    /// The number `id` is displayed under, assigning the next free one when it
    /// has none yet — a definition nothing references is numbered in part order.
    fn number(&mut self, id: &str) -> usize {
        if let Some(&number) = self.numbers.get(id) {
            return number;
        }
        self.next += 1;
        self.numbers.insert(id.to_owned(), self.next);
        self.next
    }
}

/// One `w:footnote`/`w:endnote`/`w:comment`, rendered and numbered.
struct Definition {
    id: String,
    /// `w:author`, which only comments carry.
    author: Option<String>,
    text: String,
    number: usize,
}

/// Whether a definition survives into the report: a note's type must be absent
/// or `normal` (a note part's separator entries are not text), and its id must
/// not be negative.
fn keeps_definition(kind: NoteKind, id: &str, type_attr: Option<&str>) -> bool {
    if kind != NoteKind::Comment && type_attr.is_some_and(|value| value != "normal") {
        return false;
    }
    id.parse::<i64>().is_ok_and(|number| number >= 0)
}

/// The ids a notes part defines, with the entries the walk skips left out.
fn definition_ids(xml: &[u8], kind: NoteKind) -> Vec<String> {
    scan_elements(xml, kind.element(), |event| {
        let id = attr(event, b"id")?;
        let type_attr = attr(event, b"type");
        keeps_definition(kind, &id, type_attr.as_deref()).then_some(id)
    })
    .unwrap_or_default()
}

/// A header or footer part the body references, with every variant that named
/// it.
struct PartRef {
    part: String,
    /// The variants that named the part, in the order they named it.
    labels: Vec<&'static str>,
}

/// The parts one kind of reference named: the parts in the order they were
/// first named, and where each of them sits. A package is free to name a part
/// a million times, so a naming costs one lookup rather than a scan of
/// everything named before it.
#[derive(Default)]
struct PartRefs {
    parts: Vec<PartRef>,
    /// A part's position in [`Self::parts`].
    index: HashMap<String, usize>,
}

impl PartRefs {
    /// Note that `part` was named under `label`: once, however many times it
    /// was named, with every variant that named it.
    fn record(&mut self, part: String, label: &'static str) {
        if let Some(&position) = self.index.get(&part) {
            let labels = &mut self.parts[position].labels;
            if !labels.contains(&label) {
                labels.push(label);
            }
            return;
        }
        self.index.insert(part.clone(), self.parts.len());
        self.parts.push(PartRef {
            part,
            labels: vec![label],
        });
    }
}

// ── Fields ──────────────────────────────────────────────────────

/// One open complex field.
#[derive(Default)]
struct Field {
    /// The instruction pieces collected so far.
    instr: String,
    /// Whether the instruction has not been reported yet — i.e. the field is
    /// still between its `begin` and its `separate`, where its content is the
    /// instruction and not text.
    pending: bool,
    /// The text held back while the field is pending, released by
    /// `Word::close_field` when the field turns out to have no `separate`.
    held: String,
}

/// The marker for a field: `[field PAGE]`. Runs split an instruction
/// mid-token, so its pieces are whitespace-collapsed as it is reported, and
/// copied only as far as [`MAX_FIELD_INSTRUCTION_CHARS`] allows: the
/// instruction is document-controlled, so it is never collapsed in full first.
fn field_marker(instruction: &str) -> String {
    let mut capped = String::new();
    let mut chars = 0;
    for word in instruction.split_whitespace() {
        // The separator belongs to the word that follows it: a word the cap
        // leaves no room for is dropped with its separator rather than leaving
        // a trailing space behind.
        let separator = usize::from(!capped.is_empty());
        let room = MAX_FIELD_INSTRUCTION_CHARS.saturating_sub(chars + separator);
        if room == 0 {
            break;
        }
        if separator == 1 {
            capped.push(' ');
            chars += 1;
        }
        capped.extend(word.chars().take(room));
        chars += word.chars().count().min(room);
    }
    if capped.is_empty() {
        "[field]".to_owned()
    } else {
        format!("[field {capped}]")
    }
}

// ── Revisions ───────────────────────────────────────────────────

/// The tracked changes a document carries, for its summary line.
#[derive(Default)]
struct Revisions {
    /// One tally per kind of change.
    tallies: Tallies,
    /// The `w:author` of each revision, in document order, deduplicated: the
    /// list and the set share one allocation per author.
    authors: Vec<Rc<str>>,
    /// The authors already in [`Self::authors`], so a package with a revision
    /// per author does not scan them all for each revision it records.
    seen_authors: HashSet<Rc<str>>,
}

/// One tally per kind of tracked change.
#[derive(Clone, Copy, Default)]
struct Tallies {
    insertions: usize,
    deletions: usize,
    /// The `w:moveTo` elements seen.
    moved_here: usize,
    /// The `w:moveFrom` elements seen.
    moved_away: usize,
    /// The `w:moveFromRangeStart` elements seen: Word brackets every move with
    /// one pair of range marks, so this is the move count itself rather than a
    /// count of the marks inside it.
    move_ranges: usize,
    formatting: usize,
}

impl Tallies {
    /// The move count: one moved range is one move, whatever number of marks
    /// (`w:moveTo`, `w:moveFrom`, and the paragraph marks between them) it
    /// carries. A producer that brackets a move with no range marks leaves the
    /// content marks to count instead, and the two halves of one move are then
    /// one move, so `w:moveTo` decides it and `w:moveFrom` is only the fallback.
    /// The tally covers the whole document, not one part of it.
    fn moves(&self) -> usize {
        // The range marks bracket content the walk renders a marker for. A
        // package that brackets a range and marks no moved content anywhere has
        // no move to name — a summary naming one would name a change the text
        // does not carry.
        if self.moved_here == 0 && self.moved_away == 0 {
            return 0;
        }
        if self.move_ranges > 0 {
            self.move_ranges
        } else if self.moved_here > 0 {
            self.moved_here
        } else {
            self.moved_away
        }
    }
}

/// Where a copy of an `mc:AlternateContent` may have to give its revisions
/// back: the tallies are values, and the authors are only ever appended, so a
/// discarded copy costs a checkpoint rather than a copy of the accumulator.
#[derive(Clone, Copy)]
struct RevisionsMark {
    tallies: Tallies,
    authors: usize,
}

/// The property-change elements: each holds the properties that applied before
/// a revision of the thing whose container it sits in. They are named one by
/// one rather than matched by their `Change` suffix, which also catches
/// `w:numberingChange` — a list-numbering change, which this reader leaves
/// alone like the rest of list numbering.
fn is_property_change(name: &[u8]) -> bool {
    matches!(
        name,
        b"rPrChange"
            | b"pPrChange"
            | b"tblPrChange"
            | b"tblPrExChange"
            | b"tblGridChange"
            | b"trPrChange"
            | b"tcPrChange"
            | b"sectPrChange"
    )
}

impl Revisions {
    /// Count one element when it is a revision, and remember who made it.
    fn record(&mut self, elem: &Elem) {
        match elem.name.as_slice() {
            b"ins" | b"cellIns" => self.tallies.insertions += 1,
            b"del" | b"cellDel" => self.tallies.deletions += 1,
            b"moveTo" => self.tallies.moved_here += 1,
            b"moveFrom" => self.tallies.moved_away += 1,
            b"moveFromRangeStart" => self.tallies.move_ranges += 1,
            b"cellMerge" => self.tallies.formatting += 1,
            name if is_property_change(name) => self.tallies.formatting += 1,
            _ => return,
        }
        // The set is asked before anything is allocated: a revision repeating an
        // author costs a lookup, and only a new one is allocated — once, shared
        // by the list and the set.
        let Some(author) = elem.attr("author") else {
            return;
        };
        if self.seen_authors.contains(author) {
            return;
        }
        let author: Rc<str> = Rc::from(author);
        self.seen_authors.insert(Rc::clone(&author));
        self.authors.push(author);
    }

    /// A checkpoint a copy of an `mc:AlternateContent` can be rolled back to.
    fn mark(&self) -> RevisionsMark {
        RevisionsMark {
            tallies: self.tallies,
            authors: self.authors.len(),
        }
    }

    /// Undo what was recorded since `mark`: the revisions of a copy the walk
    /// did not keep are not the document's.
    fn rollback(&mut self, mark: RevisionsMark) {
        self.tallies = mark.tallies;
        for author in self.authors.drain(mark.authors..) {
            self.seen_authors.remove(&author);
        }
    }
}

// ── Assembly helpers ────────────────────────────────────────────

/// `[footnote 1]`, `[comment 2 ends]`, `[comment ?]`: the kind, the numbered
/// reference (or `?` for a reference the part does not define) and a suffix.
fn note_marker(kind: &str, number: Option<usize>, suffix: &str) -> String {
    match number {
        Some(number) => format!("[{kind} {number}{suffix}]"),
        None => format!("[{kind} ?{suffix}]"),
    }
}

/// `1 row` / `3 rows`.
fn plural(count: usize, noun: &str) -> String {
    if count == 1 {
        format!("{count} {noun}")
    } else {
        format!("{count} {noun}s")
    }
}

/// Whether a rendered copy produced nothing at all: the walk then reads the copy
/// beside it instead.
fn renders_nothing(text: Option<&str>) -> bool {
    text.is_none_or(|text| text.trim().is_empty())
}

/// A property element's numeric attribute value, `0` when absent or unreadable
/// and at most `max`: the value is document-controlled, so it never decides how
/// much the walk allocates.
fn numeric_attr(elem: &Elem, max: usize) -> usize {
    elem.attr("val")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
        .min(max)
}

// ── Entry point ─────────────────────────────────────────────────

/// Extract the readable content and the embedded images of a `.docx`/`.docm`
/// ZIP package.
#[must_use]
pub(crate) fn convert_docx(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return unreadable(".docx");
    };
    let Some(body_xml) = read_zip_entry(&mut archive, DOCX_BODY_PART).bytes() else {
        return unreadable(".docx");
    };
    // A package that names no parts conventionally has no headers and no notes,
    // so an unreadable relationship list costs nothing but those.
    let rels = read_zip_entry(&mut archive, DOCX_RELS_PART)
        .bytes()
        .and_then(|xml| relationships(&xml))
        .unwrap_or_default();

    let mut word = Word::new(&rels, &mut archive);
    let Some(body) = word.body(&body_xml) else {
        return unreadable(".docx");
    };
    // The body's bytes are done with: the peripheral parts are read one at a
    // time from here, so nothing large stays resident beside them.
    drop(body_xml);
    // The peripheral parts are read before the summary is built: a revision in a
    // header or a footnote is a revision of the document too.
    let blocks = word.extra_sections(&mut archive);

    let mut sections: Vec<String> = Vec::new();
    if let Some(summary) = word.revision_summary() {
        // The summary leads, with a blank line after it.
        sections.push(format!("{summary}\n"));
    }
    let body = body.trim();
    if !body.is_empty() {
        sections.push(body.to_owned());
    }
    sections.extend(blocks);
    let text = sections.join("\n").trim().to_owned();

    ensure_out_dir(out_dir);

    let mut skipped = SkippedImages::default();
    let images = write_media_parts(&mut archive, DOCX_MEDIA_PREFIX, out_dir, &mut skipped);
    DocOutcome::Text {
        text,
        images,
        notes: skipped.notes(),
        all_page_text_lost: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ooxml::test_fixtures::{DOCX_BODY, zip_fixture};

    /// The namespaces a Word part is written with: the body's own, the
    /// relationship one, and the two a text box needs.
    const NAMESPACES: &str = concat!(
        r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" "#,
        r#"xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" "#,
        r#"xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" "#,
        r#"xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" "#,
        r#"xmlns:v="urn:schemas-microsoft-com:vml""#,
    );

    /// A `word/document.xml` part around `body`.
    fn document(body: &str) -> String {
        format!("<w:document {NAMESPACES}><w:body>{body}</w:body></w:document>")
    }

    /// A `word/_rels/document.xml.rels` part around `entries`.
    fn rels(entries: &str) -> String {
        format!("<Relationships>{entries}</Relationships>")
    }

    /// A relationship entry of the usual shape.
    fn rel(id: &str, kind: &str, target: &str) -> String {
        let base = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
        format!(r#"<Relationship Id="{id}" Type="{base}/{kind}" Target="{target}"/>"#)
    }

    /// A package with one body and, optionally, other parts.
    fn package(body: &str, parts: &[(&str, &str)]) -> Vec<u8> {
        let document = document(body);
        let mut entries: Vec<(&str, &[u8])> = vec![("word/document.xml", document.as_bytes())];
        entries.extend(parts.iter().map(|(path, xml)| (*path, xml.as_bytes())));
        zip_fixture(&entries)
    }

    /// The text `convert_docx` reads out of `bytes`.
    fn text_of(bytes: &[u8]) -> String {
        let dir = tempfile::tempdir().expect("tempdir");
        match convert_docx(bytes, dir.path()) {
            DocOutcome::Text { text, .. } => text,
            _ => panic!("expected a Text outcome for a well-formed docx"),
        }
    }

    // ── Media and the body ──────────────────────────────────────

    #[test]
    fn docx_extracts_paragraphs_and_media() {
        let bytes = zip_fixture(&[
            ("word/document.xml", DOCX_BODY),
            ("word/media/pic.png", b"\x89PNG\r\n\x1a\nfake image bytes"),
            // A directory entry is not media: skipping it must not produce a
            // "skipped embedded image media" note (asserted by `notes.is_empty`).
            ("word/media/", b""),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text {
            text,
            images,
            notes,
            ..
        } = convert_docx(&bytes, dir.path())
        else {
            panic!("expected Text outcome for a well-formed docx");
        };
        assert_eq!(text, "First paragraph\nSecond paragraph");
        assert_eq!(images, vec![dir.path().join("pic.png")]);
        assert!(notes.is_empty());
        assert_eq!(
            std::fs::read(&images[0]).expect("read written image"),
            b"\x89PNG\r\n\x1a\nfake image bytes"
        );
    }

    #[test]
    fn docx_notes_media_entries_it_cannot_convert() {
        let bytes = zip_fixture(&[
            ("word/document.xml", DOCX_BODY),
            (
                "word/media/diagram.emf",
                b"EMF bytes this stack cannot decode",
            ),
            (
                "word/media/vector.wmf",
                b"WMF bytes this stack cannot decode",
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { images, notes, .. } = convert_docx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a well-formed docx");
        };
        assert!(images.is_empty(), "an undecodable entry yields no image");
        assert_eq!(
            notes,
            ["skipped 2 embedded image(s) in a format this pipeline cannot convert"]
        );
    }

    // ── Tables ──────────────────────────────────────────────────

    /// A table is one block: a labelled header, one line per row, one token per
    /// cell — and a `vMerge` continuation names its origin instead of repeating
    /// the value under a new column.
    #[test]
    fn docx_renders_table_rows_cells_and_merges() {
        let body = r#"<w:p><w:r><w:t>Before</w:t></w:r></w:p><w:tbl>
<w:tblGrid><w:gridCol/><w:gridCol/><w:gridCol/></w:tblGrid>
<w:tr>
<w:tc><w:tcPr><w:vMerge w:val="restart"/><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>Merged</w:t></w:r></w:p></w:tc>
<w:tc><w:p><w:r><w:t>X</w:t></w:r></w:p></w:tc>
</w:tr>
<w:tr>
<w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>
<w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>
<w:tc><w:p><w:r><w:t>Y</w:t></w:r></w:p></w:tc>
</w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Before\n\
             Table 1: 2 rows, 3 columns\n  \
             R1: C1 (spans C1-C2): Merged | C3: X\n  \
             R2: C1 (part of R1C1): | C2 (part of R1C1): | C3: Y"
        );
    }

    /// The common single-row horizontal merge carries no `w:vMerge` at all, and
    /// its cell still says which columns it covers.
    #[test]
    fn docx_spans_a_merge_without_a_vertical_one() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/><w:gridCol/><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>Merged header</w:t></w:r></w:p></w:tc>
<w:tc><w:p><w:r><w:t>Third</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Table 1: 1 row, 3 columns\n  R1: C1 (spans C1-C2): Merged header | C3: Third"
        );
    }

    /// A continuation is not a value of its own, even when the producer stored
    /// one in it: the value belongs to the cell it names.
    #[test]
    fn docx_does_not_repeat_a_merge_continuation_value() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>Merged</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p><w:r><w:t>Merged</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Table 1: 2 rows, 1 column\n  \
             R1: C1: Merged\n  \
             R2: C1 (part of R1C1):"
        );
    }

    /// The geometry a package asks for is clamped, so a `w:gridSpan` or a
    /// `w:gridBefore` of four billion columns cannot decide how much the walk
    /// allocates.
    #[test]
    fn docx_clamps_document_controlled_table_geometry() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:tcPr><w:gridSpan w:val="4000000000"/></w:tcPr><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:trPr><w:gridBefore w:val="4000000000"/></w:trPr><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Table 1: 2 rows, 256 columns\n  \
             R1: C1 (spans C1-C256): A\n  \
             R2: C256: B"
        );
    }

    /// A row's cells add up, so the running column is clamped as well: a row of
    /// wide cells sizes the merge map no more than one wide cell does.
    #[test]
    fn docx_bounds_the_geometry_a_row_of_wide_cells_adds_up_to() {
        let cell = |text: &str| {
            format!(
                r#"<w:tc><w:tcPr><w:gridSpan w:val="4000000000"/></w:tcPr>\
<w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>"#
            )
        };
        let body = format!(
            "<w:tbl><w:tr>{}{}{}{}</w:tr></w:tbl>",
            cell("A"),
            cell("B"),
            cell("C"),
            cell("D")
        );
        assert_eq!(
            text_of(&package(&body, &[])),
            "Table 1: 1 row, 256 columns\n  \
             R1: C1 (spans C1-C256): A | C256: B | C256: C | C256: D"
        );
    }

    /// A table nested in a cell is numbered after the table that holds it, and
    /// is folded onto that cell's line: the row-per-line shape has no room for
    /// a table inside a row.
    #[test]
    fn docx_numbers_a_nested_table_after_the_one_that_holds_it() {
        let body = r"<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc>
<w:p><w:r><w:t>outer</w:t></w:r></w:p>
<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:p><w:r><w:t>inner</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
</w:tc></w:tr>
</w:tbl>";
        assert_eq!(
            text_of(&package(body, &[])),
            "Table 1: 1 row, 1 column\n  \
             R1: C1: outer Table 2: 1 row, 1 column   R1: C1: inner"
        );
    }

    /// A row's own revision is an annotation on the row line, and it is a
    /// tracked change like any other: the summary counts it.
    #[test]
    fn docx_marks_inserted_and_deleted_rows() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:trPr><w:ins w:author="Ivan Petrov"/></w:trPr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:trPr><w:del w:author="Anna"/></w:trPr><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 insertion, 1 deletion \
             (authors: Ivan Petrov, Anna).\n\n\
             Table 1: 2 rows, 1 column\n  \
             R1 (inserted): C1: A\n  \
             R2 (deleted): C1: B"
        );
    }

    /// A cell's own revision, its merge and its formatting are annotations in
    /// the fixed order, and a deleted cell keeps its text — that is what the
    /// marker is for.
    #[test]
    fn docx_annotates_cells_and_a_revised_table() {
        let body = r#"<w:tbl>
<w:tblPr><w:tblPrChange w:author="Anna"/></w:tblPr>
<w:tblGrid><w:gridCol/><w:gridCol/></w:tblGrid>
<w:tr>
<w:tc><w:tcPr><w:cellIns/><w:cellDel/><w:cellMerge/></w:tcPr><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc>
<w:tc><w:tcPr><w:tcPrChange w:author="Anna"/></w:tcPr><w:p/></w:tc>
</w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 insertion, 1 deletion, 3 formatting changes \
             (authors: Anna).\n\n\
             Table 1 (formatting revised): 1 row, 2 columns\n  \
             R1: C1 (inserted, deleted, merge revised): C | C2 (formatting revised):"
        );
    }

    /// A row's table-property exception sits beside `w:trPr`, and the row it
    /// belongs to is where its revision shows — once, even when the row's own
    /// properties changed as well.
    #[test]
    fn docx_marks_a_row_property_exception() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tblPrEx><w:tblPrExChange w:author="QA"/></w:tblPrEx><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:trPr><w:trPrChange w:author="QA"/></w:trPr><w:tblPrEx><w:tblPrExChange w:author="QA"/></w:tblPrEx><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 3 formatting changes (authors: QA).\n\n\
             Table 1: 2 rows, 1 column\n  \
             R1 (formatting revised): C1: A\n  \
             R2 (formatting revised): C1: B"
        );
    }

    /// `w:tblGridChange` carries nothing in Word's own output, and inside its
    /// `w:tblGrid`, where Word writes it, it still marks the table it revises.
    #[test]
    fn docx_marks_a_revised_grid() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/><w:tblGridChange w:author="QA"/></w:tblGrid>
<w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 formatting change (authors: QA).\n\n\
             Table 1 (formatting revised): 1 row, 1 column\n  R1: C1: A"
        );
    }

    // ── Headers, footers and fields ─────────────────────────────

    /// Every variant named is read and labelled, the part serving two variants
    /// is read once, and each is numbered only because more than one is
    /// emitted.
    #[test]
    fn docx_reads_header_and_footer_variants_and_a_field() {
        let body = r#"<w:p><w:r><w:t>Body text</w:t></w:r></w:p><w:sectPr>
<w:headerReference w:type="default" r:id="rId1"/>
<w:headerReference w:type="first" r:id="rId2"/>
<w:headerReference w:type="default" r:id="rId1"/>
<w:footerReference w:type="even" r:id="rId3"/>
</w:sectPr>"#;
        let rels = rels(&format!(
            "{}{}{}",
            rel("rId1", "header", "header1.xml"),
            rel("rId2", "header", "header2.xml"),
            rel("rId3", "footer", "footer1.xml"),
        ));
        let header1 = r"<w:hdr><w:p><w:r><w:t>Company</w:t></w:r></w:p></w:hdr>";
        let header2 = r"<w:hdr><w:p><w:r><w:t>Draft</w:t></w:r></w:p></w:hdr>";
        let footer = r#"<w:ftr><w:p><w:r><w:t xml:space="preserve">Page </w:t></w:r>\
<w:r><w:fldSimple w:instr=" PAGE "/></w:r>\
<w:r><w:t xml:space="preserve"> 5</w:t></w:r></w:p></w:ftr>"#;
        let bytes = package(
            body,
            &[
                ("word/_rels/document.xml.rels", &rels),
                ("word/header1.xml", header1),
                ("word/header2.xml", header2),
                ("word/footer1.xml", footer),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "Body text\n\
             Header 1 (default):\n  Company\n\
             Header 2 (first page):\n  Draft\n\
             Footer (even pages):\n  Page [field PAGE] 5"
        );
    }

    /// A body with no text of its own is not a document with no text: what the
    /// headers read is the text.
    #[test]
    fn docx_reads_a_body_whose_only_text_is_in_a_header() {
        let body =
            r#"<w:p/><w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId1", "header", "header1.xml")),
                ),
                (
                    "word/header1.xml",
                    r"<w:hdr><w:p><w:r><w:t>Only here</w:t></w:r></w:p></w:hdr>",
                ),
            ],
        );
        assert_eq!(text_of(&bytes), "Header (default):\n  Only here");
    }

    /// `Table N:` labels count the whole package, so a header's table takes the
    /// next number instead of repeating the body's.
    #[test]
    fn docx_numbers_tables_across_parts() {
        let body = r#"<w:p><w:r><w:t>Body</w:t></w:r></w:p>
<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
<w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId1", "header", "header1.xml")),
                ),
                (
                    "word/header1.xml",
                    r"<w:hdr><w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:p><w:r><w:t>H</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl></w:hdr>",
                ),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "Body\n\
             Table 1: 1 row, 1 column\n  R1: C1: A\n\
             Header (default):\n  Table 2: 1 row, 1 column\n    R1: C1: H"
        );
    }

    /// A part, or a definition, that shows nothing shows none of its revisions
    /// either: the summary never names a change the text does not carry.
    #[test]
    fn docx_does_not_tally_what_it_does_not_show() {
        let body = r#"<w:p><w:r><w:t>Body</w:t></w:r></w:p>
<w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&format!(
                        "{}{}",
                        rel("rId1", "header", "header1.xml"),
                        rel("rId2", "footnotes", "footnotes.xml"),
                    )),
                ),
                // A header whose only content is an empty insertion renders
                // nothing, so it is skipped.
                (
                    "word/header1.xml",
                    r#"<w:hdr><w:p><w:r><w:ins w:author="Anna"/></w:r></w:p></w:hdr>"#,
                ),
                // An unreferenced footnote that renders nothing is not shown.
                (
                    "word/footnotes.xml",
                    r#"<w:footnotes><w:footnote w:id="1"><w:p><w:ins w:author="Anna"/></w:p></w:footnote></w:footnotes>"#,
                ),
            ],
        );
        assert_eq!(text_of(&bytes), "Body");
    }

    /// A revision in a header or a footnote is a revision of the document too:
    /// the summary covers the whole package, not only the body.
    #[test]
    fn docx_summarises_revisions_from_every_part() {
        let body = r#"<w:p><w:r><w:t>Body</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r></w:p>
<w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&format!(
                        "{}{}",
                        rel("rId1", "header", "header1.xml"),
                        rel("rId2", "footnotes", "footnotes.xml"),
                    )),
                ),
                (
                    "word/header1.xml",
                    r#"<w:hdr><w:p><w:ins w:author="Anna"><w:r><w:t>Draft</w:t></w:r></w:ins></w:p></w:hdr>"#,
                ),
                (
                    "word/footnotes.xml",
                    r#"<w:footnotes><w:footnote w:id="1"><w:p><w:del w:author="Anna"><w:r><w:delText>gone</w:delText></w:r></w:del></w:p></w:footnote></w:footnotes>"#,
                ),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "Unaccepted tracked changes: 1 insertion, 1 deletion (authors: Anna).\n\n\
             Body[footnote 1]\n\
             Header (default):\n  [ins]Draft[/ins]\n\
             Footnote 1:\n  [del]gone[/del]"
        );
    }

    // ── Notes, comments and fields ──────────────────────────────
    /// A note is numbered in Word's display order, its definition is a labelled
    /// block, a comment's range is not repeated at its point reference, and a
    /// note part's separator entries are not text.
    #[test]
    fn docx_reads_footnotes_endnotes_and_comments() {
        let body = r#"<w:p>\
<w:r><w:t xml:space="preserve">Claim </w:t></w:r>\
<w:r><w:endnoteReference w:id="1"/></w:r>\
<w:commentRangeStart w:id="2"/>\
<w:r><w:t>here</w:t></w:r>\
<w:commentRangeEnd w:id="2"/>\
<w:r><w:commentReference w:id="2"/></w:r>\
<w:r><w:footnoteReference w:id="1"/></w:r>\
<w:r><w:commentReference w:id="7"/></w:r></w:p>"#;
        let footnotes = r#"<w:footnotes>\
<w:footnote w:id="-1"><w:p><w:r><w:t>separator</w:t></w:r></w:p></w:footnote>\
<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:t>continued</w:t></w:r></w:p></w:footnote>\
<w:footnote w:id="1"><w:p><w:r><w:t>see the appendix</w:t></w:r></w:p></w:footnote>\
</w:footnotes>"#;
        let endnotes = r#"<w:endnotes>\
<w:endnote w:id="1"><w:p><w:r><w:t>appendix A</w:t></w:r></w:p></w:endnote>\
</w:endnotes>"#;
        let comments = r#"<w:comments>\
<w:comment w:id="2" w:author="Ivan Petrov"><w:p><w:r><w:t>needs a citation</w:t></w:r></w:p></w:comment>\
<w:comment w:id="6" w:author="Anna"><w:p><w:r><w:t>orphan</w:t></w:r></w:p></w:comment>\
</w:comments>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId5", "footnotes", "notes/footnotes.xml")),
                ),
                ("word/notes/footnotes.xml", footnotes),
                ("word/endnotes.xml", endnotes),
                ("word/comments.xml", comments),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "Claim [endnote 1][comment 1 starts]here[comment 1 ends][footnote 1][comment ?]\n\
             Footnote 1:\n  see the appendix\n\
             Endnote 1:\n  appendix A\n\
             Comment 1 (Ivan Petrov):\n  needs a citation\n\
             Comment 2 (Anna, not referenced in the text):\n  orphan"
        );
    }

    /// A definition the flow references is reported even when it holds nothing,
    /// or its anchor would name a block that is not there. One nothing
    /// references and that holds nothing stays out.
    #[test]
    fn docx_reports_a_referenced_definition_with_no_text() {
        let body = r#"<w:p><w:r><w:t>Text</w:t></w:r>\
<w:r><w:footnoteReference w:id="1"/></w:r>\
<w:r><w:footnoteReference w:id="2"/></w:r></w:p>"#;
        let footnotes = r#"<w:footnotes>\
<w:footnote w:id="1"><w:p/></w:footnote>\
<w:footnote w:id="2"><w:p><w:r><w:t>note</w:t></w:r></w:p></w:footnote>\
<w:footnote w:id="3"><w:p/></w:footnote>\
</w:footnotes>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId5", "footnotes", "footnotes.xml")),
                ),
                ("word/footnotes.xml", footnotes),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "Text[footnote 1][footnote 2]\n\
             Footnote 1:\n\
             Footnote 2:\n  note"
        );
    }

    /// A comment range runs inside one part: opened in the body, it does not
    /// suppress the same comment's point reference in a header.
    #[test]
    fn docx_resets_a_comment_range_between_parts() {
        let body = r#"<w:p><w:commentRangeStart w:id="2"/>\
<w:r><w:t>body</w:t></w:r>\
<w:commentRangeEnd w:id="2"/>\
</w:p>\
<w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&format!(
                        "{}{}",
                        rel("rId1", "header", "header1.xml"),
                        rel("rId2", "comments", "comments.xml"),
                    )),
                ),
                (
                    "word/header1.xml",
                    r#"<w:hdr><w:p><w:r><w:commentReference w:id="2"/></w:r></w:p></w:hdr>"#,
                ),
                (
                    "word/comments.xml",
                    r#"<w:comments><w:comment w:id="2" w:author="Ivan Petrov"><w:p><w:r><w:t>note</w:t></w:r></w:p></w:comment></w:comments>"#,
                ),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "[comment 1 starts]body[comment 1 ends]\n\
             Header (default):\n  [comment 1]\n\
             Comment 1 (Ivan Petrov):\n  note"
        );
    }

    /// The reset covers the notes parts, which are walked after the body like
    /// any other: a range the body left open does not suppress a footnote's
    /// point reference to the same comment.
    #[test]
    fn docx_resets_a_comment_range_before_a_notes_part() {
        let body = r#"<w:p><w:commentRangeStart w:id="2"/>\
<w:r><w:t>body</w:t></w:r>\
<w:r><w:footnoteReference w:id="1"/></w:r></w:p>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId5", "footnotes", "footnotes.xml")),
                ),
                (
                    "word/footnotes.xml",
                    r#"<w:footnotes><w:footnote w:id="1">\
<w:p><w:r><w:commentReference w:id="2"/></w:r></w:p></w:footnote></w:footnotes>"#,
                ),
                (
                    "word/comments.xml",
                    r#"<w:comments><w:comment w:id="2" w:author="Ivan Petrov">\
<w:p><w:r><w:t>note</w:t></w:r></w:p></w:comment></w:comments>"#,
                ),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "[comment 1 starts]body[footnote 1]\n\
             Footnote 1:\n  [comment 1]\n\
             Comment 1 (Ivan Petrov):\n  note"
        );
    }

    // ── Revisions ───────────────────────────────────────────────

    /// Every marker the walk raises, and the summary line that tallies them:
    /// `w:moveFrom`/`w:moveTo` are one move, not two.
    #[test]
    fn docx_names_tracked_changes_and_their_authors() {
        let body = r#"<w:p>\
<w:pPr><w:pPrChange w:author="Anna"/></w:pPr>\
<w:r><w:rPr><w:rPrChange w:author="Ivan Petrov"/></w:rPr><w:t>kept</w:t></w:r>\
<w:ins w:author="Ivan Petrov"><w:r><w:t xml:space="preserve"> added</w:t></w:r></w:ins>\
<w:del w:author="Anna"><w:r><w:delText xml:space="preserve"> gone</w:delText></w:r></w:del>\
<w:moveFrom w:author="Anna"><w:r><w:t xml:space="preserve"> away</w:t></w:r></w:moveFrom>\
<w:moveTo w:author="Anna"><w:r><w:t xml:space="preserve"> here</w:t></w:r></w:moveTo>\
<w:ins w:author="Ivan Petrov"/></w:p>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 2 insertions, 1 deletion, 1 move, \
             2 formatting changes (authors: Anna, Ivan Petrov).\n\n\
             [fmt]kept[ins] added[/ins][del] gone[/del]\
             [moved-away] away[/moved-away][moved-here] here[/moved-here][fmt ¶]"
        );
    }

    /// Word brackets a move with its range marks, and a range holds as many
    /// marks as it likes: it is still one move.
    #[test]
    fn docx_counts_a_moved_range_once() {
        let body = r#"<w:p>\
<w:moveFromRangeStart w:id="1" w:author="Anna" w:name="move1"/>\
<w:moveFrom w:author="Anna"><w:r><w:t>away</w:t></w:r></w:moveFrom>\
<w:moveTo w:author="Anna"><w:r><w:t>here</w:t></w:r></w:moveTo>\
<w:moveToRangeEnd w:id="2"/>\
<w:moveTo w:author="Anna"><w:r><w:t> and there</w:t></w:r></w:moveTo>\
</w:p>"#;
        assert!(
            text_of(&package(body, &[]))
                .starts_with("Unaccepted tracked changes: 1 move (authors: Anna).\n\n")
        );
    }

    /// A range mark brackets content the walk marks as moved. Bracketing content
    /// that carries no such mark is not a move the text shows, so the summary
    /// does not name one.
    #[test]
    fn docx_does_not_count_a_move_range_it_never_marks() {
        let body = r#"<w:moveFromRangeStart w:id="1" w:author="Anna" w:name="move1"/>\
<w:p><w:r><w:t>Plain</w:t></w:r></w:p>\
<w:moveFromRangeEnd w:id="1"/>"#;
        assert_eq!(text_of(&package(body, &[])), "Plain");
    }

    /// The paragraph mark's own revision is marked at the paragraph's end, and a
    /// `w:sectPr/w:sectPrChange` is the section's.
    #[test]
    fn docx_marks_the_paragraph_and_section_marks() {
        let body = r#"<w:p><w:pPr>\
<w:rPr><w:del w:author="Anna"/></w:rPr>\
<w:pPrChange w:author="Anna"/>\
<w:sectPr><w:sectPrChange w:author="Anna"/></w:sectPr>\
</w:pPr><w:r><w:t>Text</w:t></w:r></w:p>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 deletion, 2 formatting changes \
             (authors: Anna).\n\n\
             Text[del ¶][fmt ¶][fmt section]"
        );
    }

    /// The document's own last section lives in `w:body/w:sectPr`, where no
    /// paragraph is open: its revision is marked in the flow rather than only
    /// counted.
    #[test]
    fn docx_marks_a_document_level_section_revision() {
        let body = concat!(
            r#"<w:p><w:r><w:t>Body</w:t></w:r></w:p>"#,
            r#"<w:sectPr><w:sectPrChange w:author="Anna"/></w:sectPr>"#,
        );
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 formatting change (authors: Anna).\n\n\
             Body\n[fmt section]"
        );
    }

    // ── Text boxes ──────────────────────────────────────────────

    /// Word writes a text box twice — as an `mc:Choice` and as the
    /// `mc:Fallback` beside it — and reading both would report it twice.
    #[test]
    fn docx_reads_a_word_text_box_once() {
        let body = r#"<w:p><w:r><mc:AlternateContent>\
<mc:Choice Requires="wps"><w:drawing><wps:txbx><w:txbxContent>\
<w:p><w:r><w:t>one</w:t></w:r></w:p><w:p><w:r><w:t>two</w:t></w:r></w:p>\
</w:txbxContent></wps:txbx></w:drawing></mc:Choice>\
<mc:Fallback><w:pict><v:textbox><w:txbxContent>\
<w:p><w:r><w:t>one</w:t></w:r></w:p><w:p><w:r><w:t>two</w:t></w:r></w:p>\
</w:txbxContent></v:textbox></w:pict></mc:Fallback>\
</mc:AlternateContent></w:r><w:r><w:t>after</w:t></w:r></w:p>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "[text box]\n  one\n  two\nafter"
        );
    }

    /// The copy Word writes twice is counted once too: the revisions inside the
    /// `mc:Fallback` the walk discards are the rendered copy's, not the
    /// document's second set.
    #[test]
    fn docx_does_not_tally_a_discarded_alternate_copy() {
        let body = r#"<w:p><w:r><mc:AlternateContent>\
<mc:Choice Requires="wps"><w:drawing><wps:txbx><w:txbxContent>\
<w:p><w:ins w:author="Anna"><w:r><w:t>once</w:t></w:r></w:ins></w:p>\
</w:txbxContent></wps:txbx></w:drawing></mc:Choice>\
<mc:Fallback><w:pict><v:textbox><w:txbxContent>\
<w:p><w:ins w:author="Anna"><w:r><w:t>once</w:t></w:r></w:ins></w:p>\
</w:txbxContent></v:textbox></w:pict></mc:Fallback>\
</mc:AlternateContent></w:r></w:p>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 insertion (authors: Anna).\n\n\
             [text box]\n  [ins]once[/ins]"
        );
    }

    /// A discarded copy contributes nothing at all: a property change inside it
    /// raises no marker on what the rendered copy sits in.
    #[test]
    fn docx_does_not_mark_from_a_discarded_copy() {
        let body = r#"<w:tbl><w:tblGrid><w:gridCol/></w:tblGrid>
<w:tr><w:tc><w:p><w:r><mc:AlternateContent>\
<mc:Choice Requires="wps"><w:drawing><wps:txbx><w:txbxContent>\
<w:p><w:r><w:t>box</w:t></w:r></w:p></w:txbxContent></wps:txbx></w:drawing></mc:Choice>\
<mc:Fallback><w:pict><v:textbox><w:txbxContent>\
<w:p><w:r><w:t>box</w:t></w:r></w:p></w:txbxContent>\
<w:tblGridChange w:author="QA"/></v:textbox></w:pict></mc:Fallback>\
</mc:AlternateContent></w:r></w:p></w:tc></w:tr>
</w:tbl>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Table 1: 1 row, 1 column\n  R1: C1: [text box]   box"
        );
    }

    /// A copy that rendered nothing is not a set of revisions of its own: the
    /// copy beside it, the one that carries the text, is the one the summary
    /// counts.
    #[test]
    fn docx_does_not_tally_a_copy_that_rendered_nothing() {
        let body = r#"<w:p><w:r><mc:AlternateContent>\
<mc:Choice Requires="wps"><w:ins w:author="Anna"/></mc:Choice>\
<mc:Choice Requires="wpg"><w:drawing><wps:txbx><w:txbxContent>\
<w:p><w:ins w:author="Anna"><w:r><w:t>once</w:t></w:r></w:ins></w:p>\
</w:txbxContent></wps:txbx></w:drawing></mc:Choice>\
</mc:AlternateContent></w:r></w:p>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "Unaccepted tracked changes: 1 insertion (authors: Anna).\n\n\
             [text box]\n  [ins]once[/ins]"
        );
    }

    /// A text box is read once wherever it sits: the header's own is a
    /// `[text box]` block inside the header's block.
    #[test]
    fn docx_reads_a_header_text_box_once() {
        let body = r#"<w:p><w:r><w:t>Body</w:t></w:r></w:p>
<w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr>"#;
        let choice = r#"<mc:Choice Requires="wps"><w:drawing><wps:txbx><w:txbxContent>\
<w:p><w:r><w:t>box</w:t></w:r></w:p></w:txbxContent></wps:txbx></w:drawing></mc:Choice>"#;
        let fallback = r"<mc:Fallback><w:pict><v:textbox><w:txbxContent>\
<w:p><w:r><w:t>box</w:t></w:r></w:p></w:txbxContent></v:textbox></w:pict></mc:Fallback>";
        let header = format!(
            "<w:hdr><w:p><w:r><mc:AlternateContent>{choice}{fallback}</mc:AlternateContent></w:r></w:p></w:hdr>"
        );
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId1", "header", "header1.xml")),
                ),
                ("word/header1.xml", &header),
            ],
        );
        assert_eq!(
            text_of(&bytes),
            "Body\nHeader (default):\n  [text box]\n    box"
        );
    }

    /// A `mc:Choice` that renders nothing at all leaves the fallback to speak
    /// for the box.
    #[test]
    fn docx_falls_back_when_a_choice_renders_nothing() {
        let body = r#"<w:p><mc:AlternateContent>\
<mc:Choice Requires="wps"><w:drawing/></mc:Choice>\
<mc:Fallback><w:pict><v:textbox><w:txbxContent><w:p><w:r><w:t>vml</w:t></w:r></w:p>\
</w:txbxContent></v:textbox></w:pict></mc:Fallback>\
</mc:AlternateContent></w:p>"#;
        assert_eq!(text_of(&package(body, &[])), "[text box]\n  vml");
    }

    /// A choice that renders nothing is not the copy that counts: a later choice
    /// carrying content is read in its place, and the fallback stays unread.
    #[test]
    fn docx_reads_a_later_choice_that_has_content() {
        let body = r#"<w:p><mc:AlternateContent>\
<mc:Choice Requires="wps"><w:drawing/></mc:Choice>\
<mc:Choice Requires="wpg"><w:drawing><wps:txbx><w:txbxContent>\
<w:p><w:r><w:t>second</w:t></w:r></w:p></w:txbxContent></wps:txbx></w:drawing></mc:Choice>\
<mc:Fallback><w:pict><v:textbox><w:txbxContent>\
<w:p><w:r><w:t>vml</w:t></w:r></w:p></w:txbxContent></v:textbox></w:pict></mc:Fallback>\
</mc:AlternateContent></w:p>"#;
        assert_eq!(text_of(&package(body, &[])), "[text box]\n  second");
    }

    // ── Fields ──────────────────────────────────────────────────

    /// A complex field's instruction is reported at its `separate` — the cached
    /// result after it is ordinary text — and a field that never separates is
    /// reported at its `end`.
    #[test]
    fn docx_reports_field_instructions_and_their_results() {
        let body = r#"<w:p>\
<w:r><w:fldChar w:fldCharType="begin"/></w:r>\
<w:r><w:instrText xml:space="preserve"> PAGE \</w:instrText></w:r>\
<w:r><w:instrText xml:space="preserve">* MERGEFORMAT </w:instrText></w:r>\
<w:r><w:fldChar w:fldCharType="separate"/></w:r>\
<w:r><w:t>7</w:t></w:r>\
<w:r><w:fldChar w:fldCharType="end"/></w:r>\
<w:r><w:fldChar w:fldCharType="begin"/></w:r>\
<w:r><w:instrText>DATE</w:instrText></w:r>\
<w:r><w:fldChar w:fldCharType="end"/></w:r>\
</w:p>"#;
        assert_eq!(
            text_of(&package(body, &[])),
            "[field PAGE \\* MERGEFORMAT]7[field DATE]"
        );
    }

    /// A field's instruction is reported up to the cap — with no separator left
    /// dangling when the cap falls between two words.
    #[test]
    fn docx_caps_a_field_instruction_without_a_trailing_space() {
        let instruction = format!("{} bcd", "a".repeat(MAX_FIELD_INSTRUCTION_CHARS - 1));
        let body = format!(
            r#"<w:p><w:r><w:fldSimple w:instr="{instruction}"><w:r><w:t>x</w:t></w:r></w:fldSimple></w:r></w:p>"#
        );
        assert_eq!(
            text_of(&package(&body, &[])),
            format!("[field {}]x", "a".repeat(MAX_FIELD_INSTRUCTION_CHARS - 1))
        );
    }

    /// A `w:fldChar begin` a part never closes must not swallow the rest of the
    /// part as its instruction: the field is reported where the part ends, with
    /// the text it was holding.
    #[test]
    fn docx_reports_a_field_a_part_left_open() {
        let body = r#"<w:p><w:r><w:t xml:space="preserve">Before </w:t></w:r>\
<w:r><w:fldChar w:fldCharType="begin"/></w:r>\
<w:r><w:instrText>PAGE</w:instrText></w:r>\
<w:r><w:t>After</w:t></w:r></w:p>"#;
        assert_eq!(text_of(&package(body, &[])), "Before \n[field PAGE]After");
    }

    // ── Degrading ───────────────────────────────────────────────

    /// A body that cannot be read as XML is the one part whose failure costs
    /// the whole package.
    #[test]
    fn docx_reports_an_unparsable_body_as_unreadable() {
        let bytes = zip_fixture(&[("word/document.xml", b"<w:document><w:body><w:p></w:body>")]);
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(matches!(
            convert_docx(&bytes, dir.path()),
            DocOutcome::Unreadable { .. }
        ));
    }

    /// A notes part that is named but will not parse costs only its own blocks,
    /// and the walk past it still reaches the body's text.
    #[test]
    fn docx_skips_a_notes_part_it_cannot_parse() {
        let body =
            r#"<w:p><w:r><w:t>Body</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r></w:p>"#;
        let bytes = package(
            body,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&rel("rId5", "footnotes", "footnotes.xml")),
                ),
                ("word/footnotes.xml", "<w:footnotes><w:footnote"),
            ],
        );
        assert_eq!(text_of(&bytes), "Body[footnote ?]");
    }

    // ── The shared mark list ────────────────────────────────────

    /// Every mark the shared list names is a spelling this reader really prints:
    /// `assets/docgen/rules.json`'s `docx_marks` is the one statement the kit's
    /// refusals and this reader both read, so an entry whose spelling no reading
    /// ever shows would have the kit name a label that is not there. The fixtures
    /// below are rich enough to raise every one of them, and each entry's
    /// `example` must appear in the joined readings.
    #[test]
    fn every_docx_mark_the_shared_list_names_is_a_mark_the_reader_prints() {
        // A body whose one paragraph carries the run-level and paragraph-mark
        // revisions, a field, a comment range and a text box, beside two headers
        // (a part per variant, so both are numbered) and a footnote nothing
        // references.
        let revised = concat!(
            r#"<w:p><w:pPr><w:pPrChange w:author="Anna"/>"#,
            r#"<w:sectPr><w:sectPrChange w:author="Anna"/></w:sectPr></w:pPr>"#,
            r#"<w:commentRangeStart w:id="2"/>"#,
            r#"<w:r><w:rPr><w:rPrChange w:author="Anna"/></w:rPr><w:t>kept</w:t></w:r>"#,
            r#"<w:ins w:author="Anna"><w:r><w:t xml:space="preserve"> added</w:t></w:r></w:ins>"#,
            r#"<w:del w:author="Anna"><w:r><w:delText xml:space="preserve"> gone</w:delText></w:r></w:del>"#,
            r#"<w:moveFrom w:author="Anna"><w:r><w:t xml:space="preserve"> away</w:t></w:r></w:moveFrom>"#,
            r#"<w:moveTo w:author="Anna"><w:r><w:t xml:space="preserve"> here</w:t></w:r></w:moveTo>"#,
            r#"<w:r><w:fldSimple w:instr="PAGE \* MERGEFORMAT"><w:t>1</w:t></w:fldSimple></w:r>"#,
            r#"<w:r><w:pict><v:shape><v:textbox><w:txbxContent><w:p><w:r><w:t>boxed</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r>"#,
            r#"</w:p>"#,
            r#"<w:sectPr><w:headerReference w:type="first" r:id="rId1"/><w:headerReference w:type="default" r:id="rId2"/></w:sectPr>"#,
        );
        let revised = package(
            revised,
            &[
                (
                    "word/_rels/document.xml.rels",
                    &rels(&format!(
                        "{}{}{}",
                        rel("rId1", "header", "header1.xml"),
                        rel("rId2", "header", "header2.xml"),
                        rel("rId5", "footnotes", "footnotes.xml"),
                    )),
                ),
                (
                    "word/header1.xml",
                    r"<w:hdr><w:p><w:r><w:t>FirstHead</w:t></w:r></w:p></w:hdr>",
                ),
                (
                    "word/header2.xml",
                    r"<w:hdr><w:p><w:r><w:t>DefaultHead</w:t></w:r></w:p></w:hdr>",
                ),
                (
                    "word/footnotes.xml",
                    r#"<w:footnotes><w:footnote w:id="1"><w:p><w:r><w:t>orphan note</w:t></w:r></w:p></w:footnote></w:footnotes>"#,
                ),
                // The conventional comments part, which the body's
                // `commentRangeStart` is numbered from: `[comment 1 starts]`.
                (
                    "word/comments.xml",
                    r#"<w:comments><w:comment w:id="2" w:author="Ivan Petrov"><w:p><w:r><w:t>needs a citation</w:t></w:r></w:p></w:comment></w:comments>"#,
                ),
            ],
        );
        // A paragraph whose own mark carries all four paragraph-mark revisions.
        let paragraph_marks = r#"<w:p><w:pPr><w:rPr><w:ins w:author="Anna"/><w:del w:author="Anna"/><w:moveTo w:author="Anna"/><w:moveFrom w:author="Anna"/></w:rPr></w:pPr><w:r><w:t>P</w:t></w:r></w:p>"#;
        // A revised table: row 1 spans C2-C4, row 2 has a merge-revised cell, and
        // row 4 continues a vertical merge that started at R3C2.
        let tables = concat!(
            r#"<w:tbl><w:tblPr><w:tblPrChange w:author="Anna"/></w:tblPr>"#,
            r#"<w:tblGrid><w:gridCol/><w:gridCol/><w:gridCol/><w:gridCol/></w:tblGrid>"#,
            r#"<w:tr><w:tc><w:p><w:r><w:t>One</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:tcPr><w:gridSpan w:val="3"/></w:tcPr><w:p><w:r><w:t>Wide</w:t></w:r></w:p></w:tc></w:tr>"#,
            r#"<w:tr><w:tc><w:tcPr><w:cellMerge/></w:tcPr><w:p><w:r><w:t>M</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>Two</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>Three</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>Four</w:t></w:r></w:p></w:tc></w:tr>"#,
            r#"<w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>Origin</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>D</w:t></w:r></w:p></w:tc></w:tr>"#,
            r#"<w:tr><w:tc><w:p><w:r><w:t>E</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>G</w:t></w:r></w:p></w:tc>"#,
            r#"<w:tc><w:p><w:r><w:t>H</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"#,
        );

        let corpus = format!(
            "{}\n{}\n{}",
            text_of(&revised),
            text_of(&package(paragraph_marks, &[])),
            text_of(&package(tables, &[])),
        );

        let marks = crate::docgen::docx_marks();
        assert!(!marks.is_empty(), "the shared list of docx marks is empty");
        for mark in marks {
            let example = mark.example.as_str();
            assert!(
                corpus.contains(example),
                "the reader never prints {example:?} (entry {})",
                mark.pattern
            );
        }
    }

    /// The literals of `source` that are shaped like a mark a reading prints: the
    /// ones it carries a character an ordinary sentence of prose does not — a
    /// bracket, a parenthesis or the `¶` a paragraph mark states. This is the
    /// narrowing the scan below rests on, so it is stated once and pinned on prose
    /// by [`the_mark_scan_leaves_a_sentence_of_prose_alone`].
    fn mark_literals(source: &str) -> Vec<String> {
        let literal = regex::Regex::new(r#""((?:[^"\\]|\\.)*)""#).expect("the literal pattern");
        literal
            .captures_iter(source)
            .map(|found| found[1].to_owned())
            .filter(|text| text.contains(['[', ']', '(', ')', '¶']))
            .collect()
    }

    /// The list's other direction: every mark the Word reader's own module writes
    /// is named by `docx_marks`, so a mark added to the walk without the shared
    /// list fails here rather than being a spelling the kit can refuse no fragment
    /// for. The source read is this reader's own code — the test module and a line
    /// comment are left out, so a mark asserted on in a test or quoted in prose is
    /// no mark the reader writes. The shared vocabulary module
    /// ([`crate::reader_output`]) is deliberately not read: it states the Excel and
    /// PowerPoint families' marks too, which no `docx_marks` entry names, and its
    /// Word names are pinned by [`every_docx_mark_the_shared_list_names_is_a_mark_the_reader_prints`],
    /// which requires the reading to print every entry the list holds. Only a
    /// mark-shaped literal is read at all ([`mark_literals`]), which is what reaches
    /// the marks a `format!` template builds (`Table {} (formatting revised)`,
    /// `[field {capped}]`): such a literal must match a pattern of the list or spell
    /// only the words the list spells. A mark written with none of those characters
    /// — a bare label such as `Table {}` — is out of this scan's reach, and a rename
    /// of one is caught by the test that reads a reading, since the list's own
    /// example stops being printed.
    #[test]
    fn every_docx_mark_the_reader_writes_is_in_the_shared_list() {
        /// The runs of three or more ASCII letters in `text` — the words a mark's
        /// pattern or the example it prints spells.
        fn words_in(text: &str) -> Vec<String> {
            text.split(|c: char| !c.is_ascii_alphabetic())
                .filter(|word| word.len() >= 3)
                .map(str::to_owned)
                .collect()
        }

        let marks = crate::docgen::docx_marks();
        let patterns: Vec<regex::Regex> = marks
            .iter()
            .map(|mark| regex::Regex::new(&mark.pattern).expect("a docx mark's pattern is a regex"))
            .collect();
        // Every word the list spells, in a pattern or in the example it prints.
        let words: Vec<String> = marks
            .iter()
            .flat_map(|mark| words_in(&format!("{} {}", mark.pattern, mark.example)))
            .collect();
        // A word the list spells, in the form it spells it: its own, or the plural
        // of it — the one form the two spell differently, the reader's tally line
        // saying `1 formatting change` where the examples state the plural.
        let spelled = |word: &str| {
            words
                .iter()
                .any(|known| known == word || known.strip_suffix('s') == Some(word))
        };
        let placeholder = regex::Regex::new(r"\{[^{}]*\}").expect("the placeholder pattern");
        let code = include_str!("docx.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("the reader's own code precedes its tests")
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let candidates = mark_literals(&code);
        assert!(
            candidates.len() > 10,
            "the scan must see this reader's marks, saw {}",
            candidates.len()
        );
        for text in &candidates {
            let written = placeholder.replace_all(text, " ");
            if patterns.iter().any(|pattern| pattern.is_match(&written)) {
                continue;
            }
            for word in words_in(&written) {
                assert!(
                    spelled(&word),
                    "{text:?} is a literal this reader writes with the word {word:?}, which no \
                     entry of the shared list spells — a mark a reading prints belongs in \
                     assets/docgen/rules.json's docx_marks"
                );
            }
        }
    }

    /// The scan above must not read a sentence of prose for a mark: a reader's own
    /// error message that happened to spell a mark word would otherwise send its
    /// author to `rules.json` for a spelling no reading prints. The sample holds the
    /// messages that did exactly that under the shape check this replaces.
    #[test]
    fn the_mark_scan_leaves_a_sentence_of_prose_alone() {
        let prose = r#"
            let a = "text is not in the body";
            let b = "Comment range is unclosed";
            let c = "part of the document";
            let d = "the field instruction is reported up to this many characters";
        "#;
        assert!(
            mark_literals(prose).is_empty(),
            "prose read as a mark: {:?}",
            mark_literals(prose)
        );
        // And a literal that IS a mark the list does not name is still read: a
        // template-built bracket mark, a parenthesis mark, and one written as a
        // raw string (the scan sees the quoted run wherever it sits).
        let marks = r#"
            let e = format!("[revision {}]", n);
            let f = "(revision started)";
            let g = r"[revision note]";
        "#;
        assert_eq!(
            mark_literals(marks),
            [
                r"[revision {}]".to_owned(),
                "(revision started)".to_owned(),
                "[revision note]".to_owned(),
            ]
        );
    }
}
