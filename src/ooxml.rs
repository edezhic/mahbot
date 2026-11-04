//! OOXML package reading: the ZIP half of document conversion.
//!
//! [`crate::document`] owns content-first detection and the PDF arm — the one non-ZIP
//! container; every other format it converts is an OOXML package, read here with the
//! shared `zip` + `quick_xml` plumbing. Word's reader is the module [`docx`], Excel's
//! [`xlsx`]; PowerPoint's slide walk (which also reaches each slide's layout, comments,
//! author registry and diagram data) still lives here. Charts are read by [`chart`].
//!
//! Invariants: a package that cannot be opened, or whose defining part is absent,
//! degrades to [`DocOutcome::Unreadable`], while a merely missing part costs its own
//! sheet or slide plus a note. Every entry is bounded by [`MAX_ZIP_ENTRY_BYTES`], and an
//! artifact is named from the entry's base name alone, so a crafted `../` path cannot
//! escape `out_dir`.

use crate::docgen::ppt_marks;
use crate::document::{
    DELIVERABLE_IMAGE_EXTENSIONS, DRAWING_EXTENSIONS, DocOutcome, SkipReason, SkippedImages,
    ensure_out_dir, write_picture,
};
use crate::reader_output::{
    Unshown, UnshownKind, slide_comments_header, slide_header, slide_notes_header, text_block,
    text_lines,
};
use crate::util::one_line;
use quick_xml::Reader;
use quick_xml::events::{BytesRef, BytesStart, Event};
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

mod chart;
mod docx;
pub(crate) use docx::{DOCX_BODY_PART, convert_docx};
/// Excel's number formats, shared by the sheet and chart readers.
mod numfmt;
mod xlsx;
pub(crate) use xlsx::{WORKBOOK_PART, convert_xlsx};

/// Maximum bytes decompressed from a single ZIP entry, so a lying size header cannot
/// inflate the temp dir. Deliberately not an aggregate bound: the entry count is open.
const MAX_ZIP_ENTRY_BYTES: u64 = 64 * 1024 * 1024;

/// Part listing the presentation's slides in slide order.
pub(crate) const PPT_PRESENTATION_PART: &str = "ppt/presentation.xml";
/// Relationships of the presentation part: slide relationship id -> slide part.
const PPT_PRESENTATION_RELS_PART: &str = "ppt/_rels/presentation.xml.rels";
/// Base every presentation relationship target resolves against.
const PPT_BASE: &str = "ppt/";
/// Prefix of the slide parts.
const PPT_SLIDES_PREFIX: &str = "ppt/slides/";
/// Prefix of the embedded-media parts in a PowerPoint package.
const PPT_MEDIA_PREFIX: &str = "ppt/media/";
/// Charts in a PowerPoint package: `ppt/charts/chart<N>.xml`.
const PPT_CHARTS_PREFIX: &str = "ppt/charts/";
/// The objects a PowerPoint package embeds (`ppt/embeddings/`).
const PPT_EMBEDDINGS_PREFIX: &str = "ppt/embeddings/";
/// The deck's legacy comment-author registry (`p:cmAuthorLst`): a slide's `p:cm`
/// entries name their author by an `authorId` into it.
const PPT_COMMENT_AUTHORS_PART: &str = "ppt/commentAuthors.xml";
/// The deck's 2018 author registry (`p:188:authorLst`): a threaded comment's
/// `authorId` is a GUID into it.
const PPT_MODERN_AUTHORS_PART: &str = "ppt/authors.xml";

/// Picture formats a package embeds that this pipeline does not convert and the
/// product's own picture list ([`crate::util::IMAGE_EXTENSIONS`]) does not name: named
/// as images, with the reason, because that is what they are.
const EMBEDDED_EXTRA_IMAGE_EXTENSIONS: &[&str] =
    &["gif", "bmp", "dib", "tif", "tiff", "ico", "jfif"];
/// Extensions of the audio a package embeds.
const EMBEDDED_AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "wav", "m4a", "aac", "ogg", "oga", "wma", "flac", "mid", "midi",
];

/// Extensions accepted as OOXML Word packages (`docm` is a macro-enabled docx).
const DOCX_EXTENSIONS: &[&str] = &["docx", "docm"];
/// Extensions accepted as OOXML Excel packages (`xlsm` is macro-enabled xlsx).
const XLSX_EXTENSIONS: &[&str] = &["xlsx", "xlsm"];
/// Extensions accepted as OOXML PowerPoint packages (`pptm` is macro-enabled pptx).
const PPTX_EXTENSIONS: &[&str] = &["pptx", "pptm"];

/// The OOXML family a file name belongs to — the ZIP magic is shared by all three.
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

/// `path`'s OOXML family, or `None`. The extension lists above are the only home of
/// the families' extensions.
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

/// Extract slide text, comments, speaker notes and embedded images from a
/// `.pptx`/`.pptm` package.
///
/// Comments and replies are read in both encodings a deck carries (legacy ISO 29500
/// `ppt/comments/comment<N>.xml` and the 2018 threaded one), each author resolved
/// through the deck's own registry. A slide's layout, master and theme are not read, so a
/// chart one of them draws is not described where it sits: the report counts it.
#[must_use]
pub(crate) fn convert_pptx(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return unreadable(Family::Pptx);
    };
    let presentation_rels = presentation_relationships(&mut archive);
    let Some(parts) = resolve_slide_parts(&mut archive, &presentation_rels) else {
        return unreadable(Family::Pptx);
    };
    let authors = read_ppt_authors(&mut archive, &presentation_rels);

    ensure_out_dir(out_dir);

    let charts = chart::read_charts(&mut archive, PPT_CHARTS_PREFIX, false);
    let mut notes = Vec::new();
    let mut unshown = Unshown::default();
    let mut skipped = SkippedImages::default();
    let images = write_media_parts(
        &mut archive,
        PPT_MEDIA_PREFIX,
        out_dir,
        &mut skipped,
        &mut unshown,
    );

    let mut blocks = Vec::new();
    // A deck's slides share a handful of layouts, so a layout's title indices and a
    // repeated diagram are read once for the whole presentation.
    let mut layouts: HashMap<String, HashSet<String>> = HashMap::new();
    let mut diagrams = DiagramCache::default();
    let mut described = HashSet::new();
    for (index, part) in parts.iter().enumerate() {
        let number = index + 1;
        // A slide the presentation declares but the package does not resolve costs its own
        // number and a note: dropping it would renumber the slides after it.
        let Some(part) = part.as_deref() else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        // Everything the slide owns — relationships, and the notes, layout and diagram parts
        // they name — resolves against the directory the package actually put the slide part
        // in, never the conventional `ppt/slides/`.
        let base = part_dir(part);
        let Some(xml) = read_zip_entry(&mut archive, part).bytes() else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        // A relationships part that could not be read loses the slide's notes, comments,
        // layout, diagrams and charts at once — the slide's own loss, said once here.
        let rels = read_relationships(&mut archive, &rels_part_of(part));
        if rels.is_none() {
            notes.push(format!("slide {number} relationships could not be read"));
        }
        let rels = rels.as_deref();
        // The layout is consulted for its title placeholders — its text never enters the answer.
        let layout_titles = layout_title_indices(&mut archive, base, rels, &mut layouts);
        // A title placeholder names its type on the slide or leaves it to the layout; one that
        // names neither takes the schema's defaults (`obj`, idx 0).
        let is_title = |kind: Option<&str>, idx: Option<&str>| match kind {
            Some("title" | "ctrTitle") => true,
            Some("obj") | None => {
                layout_titles.is_some_and(|titles| titles.contains(idx.unwrap_or("0")))
            }
            Some(_) => false,
        };
        let Some(content) = slide_text(&xml, is_title) else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        // The charts the slide draws are read where they sit: a chart whose content is printed
        // is no longer a loss. A slide whose block is not printed marks nothing, so nothing it
        // draws can vanish from both the reading and the report.
        let chart_lines = rels.map_or_else(Vec::new, |rels| {
            let parts = chart::chart_parts(&xml, rels, base);
            let (lines, chart_described) = chart::lines(&charts, &parts);
            described.extend(chart_described);
            lines
        });
        let shown = slide_diagrams(&mut archive, base, &xml, rels, &mut diagrams);
        blocks.push(slide_block(
            number,
            slide_hidden(&xml),
            &content,
            &shown,
            &chart_lines,
        ));
        // The comments block stands between the slide's own block and its notes.
        let comments = slide_comments(&mut archive, base, rels, &authors);
        comments.place(number, &mut blocks, &mut notes);
        match slide_notes(&mut archive, base, rels) {
            SlideNotes::Text(lines) if !lines.is_empty() => {
                blocks.push(text_lines(&slide_notes_header(number), &lines));
            }
            // The slide declares a notes part, so the text it holds is lost — said, rather than
            // reported as a slide with no notes.
            SlideNotes::Unreadable => {
                notes.push(format!("slide {number} notes could not be read"));
            }
            _ => {}
        }
    }
    unshown.merge(&unopened_parts(
        &mut archive,
        PPT_CHARTS_PREFIX,
        None, // a slide's diagrams are read: this arm prints their text
        PPT_EMBEDDINGS_PREFIX,
        None, // a PowerPoint package has no chart sheets
        &described,
    ));
    notes.extend(skipped.notes());
    let text = blocks.join("\n\n").trim_end().to_string();
    DocOutcome::text(text, images, notes, unshown)
}

/// The diagram text a reading shows for each slide, keyed by the slide part the
/// reading numbers: the text a slide's diagrams show — printed by the reader, never
/// changed by the editor.
///
/// Read here rather than by the document kit's own refusals, so what a diagram shows is
/// decided by the reader's rule alone (see [`slide_diagrams`]). A slide whose diagrams
/// read as lost, one whose diagrams show no text, and one whose part cannot be read are
/// absent from the map. Best effort: a package that is not a presentation, or whose
/// bytes do not open, yields an empty map.
pub(crate) fn pptx_diagram_text(bytes: &[u8]) -> HashMap<String, Vec<String>> {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return HashMap::new();
    };
    let presentation_rels = presentation_relationships(&mut archive);
    let Some(parts) = resolve_slide_parts(&mut archive, &presentation_rels) else {
        return HashMap::new();
    };
    let mut cache = DiagramCache::default();
    let mut text = HashMap::new();
    for part in parts.into_iter().flatten() {
        let Some(xml) = read_zip_entry(&mut archive, &part).bytes() else {
            continue;
        };
        let rels = read_relationships(&mut archive, &rels_part_of(&part));
        let shown: Vec<String> = slide_diagrams(
            &mut archive,
            part_dir(&part),
            &xml,
            rels.as_deref(),
            &mut cache,
        )
        .into_iter()
        .filter_map(|diagram| match diagram {
            // One entry per diagram that shows text: its points' lines joined the way a reading
            // prints them.
            DiagramRead::Text(lines) if !lines.is_empty() => Some(lines.join("\n")),
            _ => None,
        })
        .collect();
        if !shown.is_empty() {
            text.insert(part, shown);
        }
    }
    text
}

/// The presentation part's relationships: they name the slide list and the
/// comment-author registries, so both are read from one list, once per deck. A part that
/// cannot be read names nothing, and each reader falls back to conventional names.
fn presentation_relationships<R: Read + Seek>(archive: &mut ZipArchive<R>) -> Vec<Relationship> {
    read_relationships(archive, PPT_PRESENTATION_RELS_PART).unwrap_or_default()
}

/// The slide parts in presentation order, one slot per slide the presentation declares;
/// a slot is `None` when that slide's relationship cannot be resolved, so the slides
/// after it keep their own number. `None` for the whole list when the package has
/// neither a presentation part nor conventionally named slide parts.
fn resolve_slide_parts<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    presentation_rels: &[Relationship],
) -> Option<Vec<Option<String>>> {
    let presentation = read_zip_entry(archive, PPT_PRESENTATION_PART).bytes();
    let parts = match presentation.as_deref() {
        Some(xml) => {
            let rels = relationship_targets(presentation_rels);
            let resolved = scan_elements(xml, b"sldId", rel_id).map(|ids| {
                ids.iter()
                    .map(|id| rels.get(id).map(|target| resolve_part(PPT_BASE, target)))
                    .collect::<Vec<_>>()
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

/// A slide's speaker notes, as resolved through the relationships beside the
/// slide part.
enum SlideNotes {
    /// The slide declares no notes part — the usual case.
    None,
    /// The notes part's text lines.
    Text(Vec<String>),
    /// The slide declares a notes part, but it could not be read. A relationships
    /// part that could not be read names no notes part here: the caller states
    /// that wider loss once.
    Unreadable,
}

/// The speaker notes a slide declares through `rels` — their targets resolved
/// against the slide part's own directory (`base`) — as their non-empty text
/// lines: never by slide index, and never a note of another slide's. Only a notes part
/// the slide declares and the reading could not show is [`SlideNotes::Unreadable`].
fn slide_notes<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    rels: Option<&[Relationship]>,
) -> SlideNotes {
    let Some(rels) = rels else {
        return SlideNotes::None;
    };
    let Some(target) = rels
        .iter()
        .find(|rel| rel.kind.ends_with("/notesSlide"))
        .map(|rel| rel.target.as_str())
    else {
        return SlideNotes::None;
    };
    match read_zip_entry(archive, &resolve_part(base, target)) {
        ZipEntry::Bytes(xml) => slide_text(&xml, |_, _| false)
            .map_or(SlideNotes::Unreadable, |text| SlideNotes::Text(text.lines)),
        ZipEntry::Missing | ZipEntry::Unreadable => SlideNotes::Unreadable,
    }
}

/// A slide's comments and their replies, as resolved through the relationships
/// beside the slide part.
#[derive(Default)]
struct SlideComments {
    /// One line per comment or reply, in document order: its author when the deck's
    /// registry resolves the entry's id, then its text flattened to one line.
    lines: Vec<String>,
    /// Whether a comments part the slide declares could not be read.
    lost: bool,
}

impl SlideComments {
    /// Put the slide's comments where they belong: its block of lines after the slide's own
    /// block, and the note naming an unread part among the slide's losses.
    fn place(&self, number: usize, blocks: &mut Vec<String>, notes: &mut Vec<String>) {
        if !self.lines.is_empty() {
            blocks.push(text_lines(&slide_comments_header(number), &self.lines));
        }
        if self.lost {
            notes.push(format!("slide {number} comments could not be read"));
        }
    }
}

/// The comments and replies a slide declares through `rels` — either encoding, in the
/// order its relationships name them — with each entry's author resolved through the
/// deck's registry. A declared part the package does not hold, or one unreadable whole,
/// sets [`SlideComments::lost`]: losing a comment is said, not passed off as a slide
/// with none. A slide declaring both encodings shows both.
fn slide_comments<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    rels: Option<&[Relationship]>,
    authors: &HashMap<String, String>,
) -> SlideComments {
    let Some(rels) = rels else {
        return SlideComments::default();
    };
    let mut comments = SlideComments::default();
    // The legacy kind and the 2018 threaded one both end `/comments`; one walk
    // reads either, so both are collected here in the order the slide names them.
    for rel in rels.iter().filter(|rel| rel.kind.ends_with("/comments")) {
        match read_zip_entry(archive, &resolve_part(base, &rel.target))
            .bytes()
            .and_then(|xml| comment_lines(&xml, authors))
        {
            Some(lines) => comments.lines.extend(lines),
            None => comments.lost = true,
        }
    }
    comments
}

/// One comment or reply: the id of the author the deck's registry names it by, its
/// text, and its open position — a reply closes before the comment it answers, so the
/// open order is what puts it after.
struct CommentEntry {
    order: usize,
    author_id: Option<String>,
    text: String,
}

impl CommentEntry {
    /// The entry's line: its author followed by `: ` when the registry resolves the id,
    /// then its text as one line ([`flatten`]). Every entry the part holds is shown, so one
    /// with no words prints the no-text mark rather than passing for no comment. The text is
    /// document-controlled and never cut.
    fn line(&self, authors: &HashMap<String, String>) -> String {
        let text = flatten(&self.text);
        let text = if text.is_empty() {
            ppt_marks().no_text.clone()
        } else {
            text
        };
        match self.author_id.as_deref().and_then(|id| authors.get(id)) {
            Some(author) => format!("{author}: {text}"),
            None => text,
        }
    }
}

/// The comments and replies a comments part holds, one line each, in document order. One
/// walk serves both encodings: legacy ISO 29500 and the 2018 threaded one.
///
/// `None` when the part cannot be read as XML, is cut short, or is no comments document.
fn comment_lines(xml: &[u8], authors: &HashMap<String, String>) -> Option<Vec<String>> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut runs = TextRuns::default();
    // The entries open right now, innermost last: a reply is pushed onto the comment it
    // answers, and its text belongs to the innermost entry.
    let mut open: Vec<CommentEntry> = Vec::new();
    let mut entries: Vec<CommentEntry> = Vec::new();
    let mut order = 0usize;
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        part.note(&event);
        if runs.note(&event) {
            flush_paragraph(&mut runs, &mut open);
        }
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"cmLst" => part.kind_seen(),
                b"cm" | b"reply" => {
                    // Text read before the entry opened belongs to no comment: it
                    // is discarded here rather than leaking into this one.
                    runs.discard();
                    open.push(CommentEntry {
                        order,
                        author_id: attr(&event, b"authorId").map(|id| id.trim().to_owned()),
                        text: String::new(),
                    });
                    order += 1;
                }
                _ => {}
            },
            Event::Empty(event) if event.local_name().as_ref() == b"cmLst" => {
                part.kind_seen();
            }
            Event::End(event) => {
                if matches!(event.local_name().as_ref(), b"cm" | b"reply") {
                    flush_paragraph(&mut runs, &mut open);
                    if let Some(entry) = open.pop() {
                        entries.push(entry);
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    if part.unreadable() {
        return None;
    }
    entries.sort_by_key(|entry| entry.order);
    Some(entries.iter().map(|entry| entry.line(authors)).collect())
}

/// Hand the paragraph `runs` has finished to the innermost open entry, separated from
/// the one before, and clear it: text outside any entry belongs to no comment.
fn flush_paragraph(runs: &mut TextRuns, open: &mut [CommentEntry]) {
    if let Some(entry) = open.last_mut() {
        entry.text.push_str(&runs.paragraph);
        entry.text.push(' ');
    }
    runs.discard();
}

/// The names a deck's comments call their authors by, keyed by the ids they carry: the
/// legacy registry and the 2018 one, each falling back to its conventional part.
///
/// A registry the package lacks, or one unreadable, resolves no author.
fn read_ppt_authors<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    rels: &[Relationship],
) -> HashMap<String, String> {
    let legacy = related_part(
        rels,
        PPT_BASE,
        |kind| kind.ends_with("/commentAuthors"),
        PPT_COMMENT_AUTHORS_PART,
    );
    let modern = related_part(
        rels,
        PPT_BASE,
        |kind| kind.contains("2018/10") && kind.ends_with("/authors"),
        PPT_MODERN_AUTHORS_PART,
    );
    let mut authors = HashMap::new();
    for part in [legacy, modern] {
        if let Some(xml) = read_zip_entry(archive, &part).bytes() {
            authors.extend(author_names(&xml));
        }
    }
    authors
}

/// The ids and names an author registry part holds — legacy `p:cmAuthor` or 2018
/// `p:188:author` entries, which differ in element name alone. A part that will not parse
/// holds none.
fn author_names(xml: &[u8]) -> HashMap<String, String> {
    scan_names(xml, &[b"cmAuthor", b"author"], author_entry)
        .unwrap_or_default()
        .into_iter()
        .collect()
}

/// One author entry: the id its comments carry and the name they are shown by. One a
/// registry leaves empty names nobody, so the entry is left out and its comments print
/// without an author.
///
/// The name is the file's own text, printed whole as the comment it prefixes is: only what
/// a mark adds around a file's text is bounded here — a field instruction, a link's address
/// — never the text itself.
fn author_entry(event: &BytesStart<'_>) -> Option<(String, String)> {
    let id = attr(event, b"id")?.trim().to_owned();
    let name = one_line(attr(event, b"name")?);
    (!name.is_empty()).then_some((id, name))
}

/// A part's `<a:t>` text, split by whether it belongs to the title placeholder.
struct SlideText {
    /// The title placeholder's lines, in part order (empty when there is no title).
    title: Vec<String>,
    /// Every other line the part holds, unchanged from what the reader showed.
    lines: Vec<String>,
}

/// The text runs one part's walk accumulates: each `<a:t>`'s text joined into the
/// paragraph being read, `<a:br/>` breaking it into lines, and an `<a:fld>`'s text left
/// out. One implementation, so the slide, diagram and comments walks read a run alike —
/// a legacy comment's `<p:text>` is a run's text too.
#[derive(Default)]
struct TextRuns {
    /// The paragraph being read right now: the run text seen since the last paragraph
    /// end moved it out.
    paragraph: String,
    /// Whether a run's text element is open — `a:t`, or the `p:text` a legacy comment
    /// writes its body in.
    in_text: bool,
    /// How many `<a:fld>` elements are open: their text is the field's, not the run's.
    fields: usize,
}

impl TextRuns {
    /// Note one event of the walk, accumulating run text, and return `true` on the `</a:p>`
    /// that ends a paragraph.
    fn note(&mut self, event: &Event<'_>) -> bool {
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"t" | b"text" => self.in_text = true,
                b"br" => self.paragraph.push('\n'),
                b"fld" => self.fields += 1,
                _ => {}
            },
            Event::Empty(event) => {
                if event.local_name().as_ref() == b"br" {
                    self.paragraph.push('\n');
                }
            }
            Event::Text(event) if self.in_text && self.fields == 0 => {
                if let Ok(chunk) = event.xml10_content() {
                    self.paragraph.push_str(&chunk);
                }
            }
            Event::GeneralRef(event) if self.in_text && self.fields == 0 => {
                append_entity(&mut self.paragraph, event);
            }
            Event::End(event) => match event.local_name().as_ref() {
                b"t" | b"text" => self.in_text = false,
                b"fld" => self.fields = self.fields.saturating_sub(1),
                b"p" => return true,
                _ => {}
            },
            _ => {}
        }
        false
    }

    /// Clear the paragraph being read: the caller read it where it belongs to no block, so
    /// it must not leak into a later one.
    fn discard(&mut self) {
        self.paragraph.clear();
    }

    /// Move the paragraph being read into `lines` ([`push_lines`]), one line per
    /// `<a:br/>` break.
    fn lines(&mut self, lines: &mut Vec<String>) {
        push_lines(lines, &mut self.paragraph);
    }
}

/// A part's `<a:t>` text, in document order: a newline at `<a:br/>` and each `<a:p>`
/// end, nothing from an `<a:fld>`. Text of a `<p:sp>` whose `<p:ph>` declares the title
/// goes to [`SlideText::title`]; everything else to [`SlideText::lines`]. `is_title`
/// receives each `<p:ph>`'s `type` and `idx`. Empty paragraphs are dropped; `None` on any
/// XML error.
fn slide_text(
    xml: &[u8],
    is_title: impl Fn(Option<&str>, Option<&str>) -> bool,
) -> Option<SlideText> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    // One entry per open `<p:sp>`: whether that shape is the title. `<p:ph>` sets the
    // innermost entry, and a run with no shape open is body text.
    let mut shapes: Vec<bool> = Vec::new();
    let mut runs = TextRuns::default();
    let mut title = Vec::new();
    let mut lines = Vec::new();
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        // Note the runs first, so a paragraph accumulates whichever event is
        // the one that ends it.
        let ended = runs.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"sp" => shapes.push(false),
                b"ph" => mark_title(&mut shapes, &event, &is_title),
                _ => {}
            },
            Event::Empty(event) => {
                if event.local_name().as_ref() == b"ph" {
                    mark_title(&mut shapes, &event, &is_title);
                }
            }
            Event::End(event) => {
                if event.local_name().as_ref() == b"sp" {
                    shapes.pop();
                }
            }
            Event::Eof => break,
            _ => {}
        }
        if ended {
            // The shape a paragraph sits in is still open here, so its title flag is paragraph's
            // own.
            if shapes.last() == Some(&true) {
                runs.lines(&mut title);
            } else {
                runs.lines(&mut lines);
            }
        }
        buffer.clear();
    }
    Some(SlideText { title, lines })
}

/// Set the innermost open `<p:sp>`'s title flag from a `<p:ph>`; a no-op when no shape is open.
fn mark_title(
    shapes: &mut [bool],
    event: &BytesStart<'_>,
    is_title: &impl Fn(Option<&str>, Option<&str>) -> bool,
) {
    if let Some(shape) = shapes.last_mut() {
        *shape = is_title(
            attr(event, b"type").as_deref(),
            attr(event, b"idx").as_deref(),
        );
    }
}

/// Move `current` into `lines`, one trimmed line per `<a:br/>` break, dropping empties.
fn push_lines(lines: &mut Vec<String>, current: &mut String) {
    for line in std::mem::take(current).split('\n') {
        let line = line.trim();
        if !line.is_empty() {
            lines.push(line.to_owned());
        }
    }
}

/// Whether the presentation hides the slide (`<p:sld ... show="0">`). Matching the
/// attribute by local name keeps `showMasterSp`/`showMasterPhAnim` from being mistaken
/// for it.
fn slide_hidden(xml: &[u8]) -> bool {
    scan_elements(xml, b"sld", |event| attr(event, b"show"))
        .and_then(|values| values.into_iter().next())
        .is_some_and(|value| value == "0" || value.eq_ignore_ascii_case("false"))
}

/// The `idx` values of the title placeholders the slide's layout declares, memoized in
/// `cache` by layout part; the layout's target resolves against `base`.
///
/// A slide's title placeholder sometimes carries its type on the layout, so the layout is
/// consulted for its title indices and nothing else. `None` when the slide declares no
/// layout relationship; a missing, unreadable or unparsable layout reads as one
/// declaring nothing.
fn layout_title_indices<'a, R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    rels: Option<&[Relationship]>,
    cache: &'a mut HashMap<String, HashSet<String>>,
) -> Option<&'a HashSet<String>> {
    let target = rels?
        .iter()
        .find(|rel| rel.kind.ends_with("/slideLayout"))
        .map(|rel| rel.target.as_str())?;
    let part = resolve_part(base, target);
    Some(
        cache
            .entry(part.clone())
            .or_insert_with(|| read_layout_title_indices(archive, &part)),
    )
}

/// [`layout_title_indices`] for one layout part, read from the package.
fn read_layout_title_indices<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    part: &str,
) -> HashSet<String> {
    let Some(xml) = read_zip_entry(archive, part).bytes() else {
        return HashSet::new();
    };
    scan_elements(&xml, b"ph", |event| {
        matches!(attr(event, b"type").as_deref(), Some("title" | "ctrTitle"))
            .then(|| attr(event, b"idx").unwrap_or_else(|| "0".to_owned()))
    })
    .unwrap_or_default()
    .into_iter()
    .collect()
}

/// What reading one diagram on a slide produced.
#[derive(Clone)]
enum DiagramRead {
    /// The visible text of the diagram's data model, in the model's own order.
    Text(Vec<String>),
    /// The diagram's data could not be read: its text is lost, and the slide says so.
    Lost,
}

/// The text a deck's diagram data parts were read as, keyed by part: a deck that shows
/// one diagram on several slides reads its part once, like its layouts.
#[derive(Default)]
struct DiagramCache(HashMap<String, DiagramRead>);

impl DiagramCache {
    /// The text of one diagram's data part, read on first use: `Lost` for a part
    /// the package does not hold or one [`diagram_text_lines`] cannot read.
    fn read<R: Read + Seek>(&mut self, archive: &mut ZipArchive<R>, part: &str) -> DiagramRead {
        if let Some(read) = self.0.get(part) {
            return read.clone();
        }
        let read = read_zip_entry(archive, part)
            .bytes()
            .and_then(|xml| diagram_text_lines(&xml))
            .map_or(DiagramRead::Lost, DiagramRead::Text);
        self.0.insert(part.to_owned(), read.clone());
        read
    }
}

/// The diagrams a slide shows, in the order its shapes present them, read from
/// their data parts through the relationships the slide part declares, targets resolved
/// against `base` and text taken from `cache`, so a part shown on several slides is
/// parsed once.
///
/// Every diagram is read once per slide: a data part a slide refers to several times
/// yields one section.
fn slide_diagrams<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    xml: &[u8],
    rels: Option<&[Relationship]>,
    cache: &mut DiagramCache,
) -> Vec<DiagramRead> {
    // A `dgm:relIds` carries `r:dm`, `r:lo`, `r:qs` and `r:cs`; `attr` matches by local
    // name, so this is the data-model reference. Every `relIds` a slide holds is collected:
    // a frame whose `relIds` names no data model still shows a diagram, and one read as no
    // diagram would be dropped in silence.
    let Some(ids) = scan_elements(xml, b"relIds", |event| Some(attr(event, b"dm"))) else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    let mut diagrams = Vec::new();
    for id in ids {
        // No reference to resolve and nothing to dedupe on: the frame keeps its section, lost.
        let Some(id) = id else {
            diagrams.push(DiagramRead::Lost);
            continue;
        };
        let part = rels.and_then(|rels| {
            rels.iter()
                .find(|rel| rel.id == id)
                .map(|rel| resolve_part(base, &rel.target))
        });
        // Dedupe on the resolved part (never across slides, which read the same diagram with
        // their own block); an id nothing resolves keeps its own section.
        let key = part.clone().unwrap_or_else(|| format!("#{id}"));
        if !seen.insert(key) {
            continue;
        }
        let read = match part {
            Some(part) => cache.read(archive, &part),
            None => DiagramRead::Lost,
        };
        diagrams.push(read);
    }
    diagrams
}

/// One open `<dgm:pt>` of a diagram's data model: whether it is an editor
/// placeholder, and the text read from it so far.
struct DiagramPoint {
    placeholder: bool,
    lines: Vec<String>,
}

/// The text a person sees in a diagram's data model: the `<dgm:t>` text of every
/// `<dgm:pt>` not flagged as an editor placeholder, flat, in document order (the
/// `dgm:cxn` structure is not reproduced). `None` for a part that is not a diagram's data
/// model, one cut short, or one left empty — the slide states the loss.
fn diagram_text_lines(xml: &[u8]) -> Option<Vec<String>> {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut points: Vec<DiagramPoint> = Vec::new();
    let mut runs = TextRuns::default();
    let mut lines = Vec::new();
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return None;
        };
        part.note(&event);
        let ended = runs.note(&event);
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"dataModel" => part.kind_seen(),
                b"pt" => {
                    // Text read before the point opened belongs to no node: it is discarded here.
                    runs.discard();
                    points.push(DiagramPoint {
                        placeholder: false,
                        lines: Vec::new(),
                    });
                }
                b"prSet" => mark_placeholder(&mut points, &event),
                _ => {}
            },
            Event::Empty(event) => match event.local_name().as_ref() {
                // A self-closing `<dgm:dataModel/>` is a data model too.
                b"dataModel" => part.kind_seen(),
                b"prSet" => mark_placeholder(&mut points, &event),
                _ => {}
            },
            Event::End(event) => {
                if event.local_name().as_ref() == b"pt"
                    && let Some(point) = points.pop()
                    && !point.placeholder
                {
                    lines.extend(point.lines);
                }
            }
            Event::Eof => break,
            _ => {}
        }
        if ended {
            // A paragraph read with no point open belongs to no node and must not leak onward.
            match points.last_mut() {
                Some(point) => runs.lines(&mut point.lines),
                None => runs.discard(),
            }
        }
        buffer.clear();
    }
    (!part.unreadable() && !blank(xml)).then_some(lines)
}

/// Mark the innermost open `<dgm:pt>` a placeholder from its `<dgm:prSet>`: the point's
/// last `prSet` decides, and its `phldr` is `"1"`/`"true"`. A `phldrT` alone is not that
/// flag — ECMA-376 gives a point with its own text both the hint and the flag, and the
/// text is what a person sees.
fn mark_placeholder(points: &mut [DiagramPoint], event: &BytesStart<'_>) {
    let Some(point) = points.last_mut() else {
        return;
    };
    point.placeholder = attr(event, b"phldr")
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
}

/// One slide block: `Slide {n}:`, ` (hidden slide)` when hidden, then the title lines,
/// the body lines, one section per diagram (its `(diagram text)` heading over indented
/// node lines, or the unreadable mark) and one per chart, printed where it sits. A slide
/// holding a diagram or chart thus never reads `(no text)`, which [`text_block`] supplies
/// only for a slide holding nothing at all.
fn slide_block(
    number: usize,
    hidden: bool,
    content: &SlideText,
    diagrams: &[DiagramRead],
    charts: &[String],
) -> String {
    let marks = ppt_marks();
    let header = slide_header(number, hidden);
    let mut lines: Vec<String> = content
        .title
        .iter()
        .map(|line| format!("{} {line}", marks.title))
        .collect();
    lines.extend(content.lines.iter().cloned());
    for diagram in diagrams {
        match diagram {
            DiagramRead::Lost => lines.push(marks.diagram_text_lost.clone()),
            DiagramRead::Text(nodes) => {
                lines.push(marks.diagram_text.clone());
                lines.extend(nodes.iter().map(|node| format!("  {node}")));
            }
        }
    }
    lines.extend(charts.iter().cloned());
    text_block(&header, &marks.no_text, &lines)
}

/// What a package's media part is as far as its file name says: a picture this pipeline
/// converts or does not, or one of the kinds the report names. Calling a video or drawing
/// an unconvertible image would report a loss the file does not hold.
enum PackageMedia {
    /// A picture this pipeline does not convert: counted with its reason in
    /// [`SkippedImages`] rather than named by kind of its own, because that is what
    /// it is. The formats the pipeline does convert never reach here —
    /// [`write_media_parts`] writes those out first.
    Image,
    /// Content of a kind the report names as what it is.
    Unshown(UnshownKind),
}

/// [`PackageMedia`] for one media part's file name. A name of no recognized kind
/// is `Media` rather than an image: a package calling an unknown thing a media
/// part says no more about it than that, and the report repeats no more.
fn package_media(file_name: &str) -> PackageMedia {
    let path = Path::new(file_name);
    if crate::util::is_image_extension(path)
        || crate::util::has_extension(path, EMBEDDED_EXTRA_IMAGE_EXTENSIONS)
    {
        PackageMedia::Image
    } else if crate::util::has_extension(path, DRAWING_EXTENSIONS) {
        PackageMedia::Unshown(UnshownKind::Drawing)
    } else if crate::util::is_video_extension(path) {
        PackageMedia::Unshown(UnshownKind::Video)
    } else if crate::util::has_extension(path, EMBEDDED_AUDIO_EXTENSIONS) {
        PackageMedia::Unshown(UnshownKind::Audio)
    } else {
        PackageMedia::Unshown(UnshownKind::Media)
    }
}

/// Write every entry under `prefix` that names a raster file into `out_dir`, counting the
/// rest into `skipped` and non-picture media into `unshown`. A nested path takes a
/// suffixed output name rather than overwriting its predecessor (only the base name survives).
fn write_media_parts<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    out_dir: &Path,
    skipped: &mut SkippedImages,
    unshown: &mut Unshown,
) -> Vec<PathBuf> {
    let entries: Vec<String> = archive
        .file_names()
        // A media part names a file, i.e. carries an extension: directory entries and
        // extension-less names would otherwise be ingested as a nonexistent image.
        .filter(|name| name.starts_with(prefix) && Path::new(name).extension().is_some())
        .map(str::to_owned)
        .collect();
    let mut images = Vec::new();
    let mut next_suffix = HashMap::new();
    for entry in entries {
        // Only the entry's file name is used: a crafted `../..` entry must
        // never escape `out_dir`.
        let file_name = crate::util::neutralized_name(crate::util::file_name_or_path(&entry));
        if !crate::util::has_extension(Path::new(&file_name), DELIVERABLE_IMAGE_EXTENSIONS) {
            match package_media(&file_name) {
                PackageMedia::Image => skipped.add(SkipReason::Unsupported),
                PackageMedia::Unshown(kind) => unshown.add(kind, 1),
            }
            continue;
        }
        let Some(content) = read_zip_entry(archive, &entry).bytes() else {
            tracing::warn!(%file_name, "document: failed to read embedded image");
            skipped.add(SkipReason::Failed);
            continue;
        };
        images.extend(write_picture(
            out_dir,
            &file_name,
            &content,
            &mut next_suffix,
            skipped,
        ));
    }
    images
}

/// Count the chart parts under `prefix` (`chart<N>.xml`, modern `chartEx<N>.xml`) less
/// the ones a chart sheet draws and the ones a reading described: a chart whose content
/// was printed where it sits is no longer a loss. An undescribed part is still counted.
fn count_chart_parts<R: Read + Seek>(
    archive: &ZipArchive<R>,
    prefix: &str,
    chart_sheets: Option<&HashSet<String>>,
    described: &HashSet<String>,
) -> usize {
    archive
        .file_names()
        .filter(|name| {
            is_chart_part(name, prefix)
                && !chart_sheets.is_some_and(|sheets| sheets.contains(*name))
                && !described.contains(*name)
        })
        .count()
}

/// Whether `name` is a chart part one predicate for both the census and the scan for a
/// chart's companion parts, so the two cannot disagree.
fn is_chart_part(name: &str, prefix: &str) -> bool {
    is_numbered_part(name, prefix, "chart") || is_numbered_part(name, prefix, "chartEx")
}

/// Whether `name` is a `<stem><N>.xml` part — numbered parts a reading names by count.
/// A non-numeric remainder (`charts/colors1.xml`) is another kind of part.
fn is_numbered_part(name: &str, prefix: &str, stem: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(".xml"))
        .and_then(|rest| rest.strip_prefix(stem))
        .is_some_and(|number| !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()))
}

/// Count the `<stem><N>.xml` parts under `prefix`.
fn count_numbered_parts<R: Read + Seek>(
    archive: &ZipArchive<R>,
    prefix: &str,
    stem: &str,
) -> usize {
    archive
        .file_names()
        .filter(|name| is_numbered_part(name, prefix, stem))
        .count()
}

/// The parts a package keeps in one of its directories (`embeddings/`) that are parts of
/// their own: a directory entry or a relationships part only places one.
fn parts_under<R: Read + Seek>(archive: &ZipArchive<R>, prefix: &str) -> Vec<String> {
    archive
        .file_names()
        .filter(|name| {
            name.starts_with(prefix) && !name.ends_with('/') && !name.contains("/_rels/")
        })
        .map(str::to_owned)
        .collect()
}

/// How many of a package's `embedded` parts a chart keeps beside itself, read from the
/// chart's own relationships (never supposed from a name). Only a part the census counted
/// is taken, so a relationship to a part the package lacks takes nothing out.
fn chart_data_parts<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    charts: &str,
    embedded: &[String],
) -> usize {
    let chart_parts: Vec<String> = archive
        .file_names()
        .filter(|name| is_chart_part(name, charts))
        .map(str::to_owned)
        .collect();
    let counted: HashSet<&str> = embedded.iter().map(String::as_str).collect();
    let mut data_parts: HashSet<String> = HashSet::new();
    for part in chart_parts {
        let rels = read_zip_entry(archive, &rels_part_of(&part))
            .bytes()
            .and_then(|xml| relationships(&xml))
            .unwrap_or_default();
        for rel in rels {
            let target = resolve_part(part_dir(&part), &rel.target);
            if counted.contains(target.as_str()) {
                data_parts.insert(target);
            }
        }
    }
    data_parts.len()
}

/// The parts of one package that no reading opens, counted for the report: charts no
/// chart sheet draws and no reading described, the diagrams whose data parts sit under
/// `diagrams/`, and the `embeddings/` parts that are no other part's companion. Each
/// family passes its own directories, so the three arms cannot count one kind twice.
fn unopened_parts<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    charts: &str,
    diagrams: Option<&str>,
    embeddings: &str,
    chart_sheets: Option<&HashSet<String>>,
    described: &HashSet<String>,
) -> Unshown {
    let mut unshown = Unshown::default();
    unshown.add(
        UnshownKind::Chart,
        count_chart_parts(archive, charts, chart_sheets, described),
    );
    if let Some(diagrams) = diagrams {
        unshown.add(
            UnshownKind::Diagram,
            count_numbered_parts(archive, diagrams, "data"),
        );
    }
    let embedded = parts_under(archive, embeddings);
    // A chart's own data workbook is the chart's, not a part of its own: what is left of the
    // counted parts is what the document embeds.
    let data_parts = chart_data_parts(archive, charts, &embedded);
    unshown.add(UnshownKind::Object, embedded.len() - data_parts);
    unshown
}

/// One OOXML relationship: the `Id`, its `Type` (empty when absent), and the `Target`.
struct Relationship {
    id: String,
    kind: String,
    target: String,
}

/// The relationships a rels part declares, in document order, and whether the part is
/// whole: the walk that reads the list also says whether it ended inside an element, so a
/// cut part reports the loss rather than reading a shortened list as everything.
fn relationships_and_whole(xml: &[u8]) -> (Option<Vec<Relationship>>, bool) {
    let mut part = Part::new(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut entries = Vec::new();
    let mut document = false;
    loop {
        let Ok(event) = reader.read_event_into(&mut buffer) else {
            return (None, false);
        };
        part.note(&event);
        match event {
            Event::Start(event) | Event::Empty(event) => match event.local_name().as_ref() {
                b"Relationships" => document = true,
                b"Relationship" => entries.extend(relationship_entry(&event)),
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    (document.then_some(entries), !part.cut())
}

/// The relationships a rels part declares, in document order. `None` when the part
/// cannot be read as XML, or is no relationships document at all: a part that is
/// something else (an error page a producer left in its place) parses as XML just as
/// happily, and reading it as a list with no entries would pass it off as one that
/// names none of its parts.
fn relationships(xml: &[u8]) -> Option<Vec<Relationship>> {
    relationships_and_whole(xml).0
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

/// A text as one line: every run of whitespace becomes one space — what a
/// comment prints, whichever family's reader found it.
fn flatten(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One `<Relationship Id Type Target/>` entry, or `None` when it names no id or target.
fn relationship_entry(event: &BytesStart<'_>) -> Option<Relationship> {
    let (Some(id), Some(target)) = (attr(event, b"Id"), attr(event, b"Target")) else {
        return None;
    };
    Some(Relationship {
        id,
        kind: attr(event, b"Type").unwrap_or_default(),
        target,
    })
}

/// Relationship id -> target. The readers read the list themselves — notes parts are named
/// there by kind — and share the mapping.
fn relationship_targets(rels: &[Relationship]) -> HashMap<String, String> {
    rels.iter()
        .map(|rel| (rel.id.clone(), rel.target.clone()))
        .collect()
}

/// Resolve a relationship target against the part that owns it: a relative target is
/// appended to `base`, an absolute one is the package path minus its leading slash, and
/// `..` segments are normalized away so the result names the ZIP entry.
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

/// The part the first relationship whose kind `matches` names, resolved against `base`,
/// or `conventional` when the relationships declare none.
fn related_part(
    rels: &[Relationship],
    base: &str,
    matches: impl Fn(&str) -> bool,
    conventional: &str,
) -> String {
    rels.iter().find(|rel| matches(&rel.kind)).map_or_else(
        || conventional.to_owned(),
        |rel| resolve_part(base, &rel.target),
    )
}

/// The directory a part's own relationship targets resolve against
/// (`ppt/slides/slide1.xml` -> `ppt/slides/`).
fn part_dir(part: &str) -> &str {
    part.rfind('/').map_or("", |slash| &part[..=slash])
}

/// The relationships part of `part`, by the OPC convention the document kit's
/// `relsPartFor` follows: wherever the package put the part, its relationships are beside it.
fn rels_part_of(part: &str) -> String {
    let name = part.rsplit_once('/').map_or(part, |(_, name)| name);
    format!("{}_rels/{name}.rels", part_dir(part))
}

/// The relationship-id attribute of an element that refers to a part (`r:id="rId1"`).
///
/// Matched by local name plus a namespace prefix: a slide carries both its own unprefixed
/// `id` and the relationship one, and only the prefix tells them apart.
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

/// The value of the attribute whose local name is `name` (after any prefix), unescaped.
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

fn scan_elements<T>(
    xml: &[u8],
    element: &[u8],
    collect: impl FnMut(&BytesStart<'_>) -> Option<T>,
) -> Option<Vec<T>> {
    scan_names(xml, &[element], collect)
}

/// [`scan_elements`] over several local names in one pass: a part may state one kind of
/// entry under the name its schema version gives it.
fn scan_names<T>(
    xml: &[u8],
    names: &[&[u8]],
    mut collect: impl FnMut(&BytesStart<'_>) -> Option<T>,
) -> Option<Vec<T>> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut found = Vec::new();
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event) | Event::Empty(event))
                if names.contains(&event.local_name().as_ref()) =>
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

/// Whether a part's bytes carry nothing but whitespace (a byte-order mark included): an
/// entry the package left empty, which declares nothing rather than holding something unread.
fn blank(bytes: &[u8]) -> bool {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    bytes.iter().all(u8::is_ascii_whitespace)
}

/// What a part's own walk says about the part. Every reader notes each event here, so the
/// part's account of itself — whole, cut at an element boundary, or no document at all —
/// comes out of the read that reads its content.
struct Part<'a> {
    /// The part's bytes: what tells an entry the package left empty from one whose bytes
    /// are no document.
    bytes: &'a [u8],
    /// Elements open right now: what a part cut at an element boundary leaves behind.
    open: usize,
    /// Whether the walk found an element of the kind the reader reads the part for.
    kind: bool,
}

impl<'a> Part<'a> {
    /// A part's walk over its own bytes.
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            open: 0,
            kind: false,
        }
    }

    /// Note one event of the walk: only the part's own elements' start and end events
    /// build the account.
    fn note(&mut self, event: &Event<'_>) {
        match event {
            Event::Start(_) => self.open += 1,
            Event::End(_) => self.open = self.open.saturating_sub(1),
            _ => {}
        }
    }

    /// Note that the walk found an element of the kind the reader reads the part for,
    /// wherever it stands: what tells a part holding what this reader came for from one
    /// holding something else.
    fn kind_seen(&mut self) {
        self.kind = true;
    }

    /// Whether the part stopped with an element still open: quick-xml reports a part cut at an
    /// element boundary as a document that simply ends, so only the reader's own walk can tell
    /// a half-written part from a finished one.
    fn cut(&self) -> bool {
        self.open > 0
    }

    /// Whether the part is not the document it is read as: cut short, or holding no element
    /// of its own kind — an error page a producer left in its place parses as happily as the
    /// real thing. An entry the package left empty ([`blank`]) declares nothing instead.
    fn unreadable(&self) -> bool {
        self.cut() || (!self.kind && !blank(self.bytes))
    }
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

/// A named ZIP entry as the reader found it: absent, present but unreadable within
/// [`MAX_ZIP_ENTRY_BYTES`], or its bytes.
enum ZipEntry {
    /// The archive has no entry under the name.
    Missing,
    /// The entry is there but cannot be read whole within [`MAX_ZIP_ENTRY_BYTES`] — a
    /// read failure, a rejected size, or a refused name.
    Unreadable,
    /// The entry's bytes.
    Bytes(Vec<u8>),
}

impl ZipEntry {
    /// The entry's bytes when read, `None` otherwise — for callers that treat absent and
    /// unreadable alike; a caller that must tell them apart matches on the variant.
    fn bytes(self) -> Option<Vec<u8>> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Missing | Self::Unreadable => None,
        }
    }
}

/// Read a named ZIP entry into memory, bounded by [`MAX_ZIP_ENTRY_BYTES`], as
/// [`ZipEntry`]. Neither `Missing` nor `Unreadable` is fatal — the caller skips the entry
/// (noting why) or reports the package unreadable.
fn read_zip_entry<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> ZipEntry {
    let mut entry = match archive.by_name(name) {
        Ok(entry) => entry,
        Err(zip::result::ZipError::FileNotFound) => return ZipEntry::Missing,
        Err(_) => return ZipEntry::Unreadable,
    };
    // The declared size is checked first, so a bomb that admits its size is rejected before
    // any inflation; the read then goes through the same cap so a lying header cannot exceed
    // it either.
    if entry.size() > MAX_ZIP_ENTRY_BYTES {
        tracing::warn!(
            %name,
            declared_bytes = entry.size(),
            "document: ZIP entry exceeds the decompression bound"
        );
        return ZipEntry::Unreadable;
    }
    let mut bytes = Vec::new();
    // Read one byte past the bound so an entry exactly at the limit is accepted whole while a
    // larger one is detected instead of silently truncated.
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

/// The reason a reading gives a recognized OOXML package it cannot open — the container
/// unreadable, or a needed part missing or unparsable. The `document` tool's refusal quotes
/// it verbatim, so a reading and a refusal cannot name the same failure differently.
pub(crate) fn corrupt_reason(family: Family) -> String {
    format!("corrupt or unsupported .{}", family.name())
}

/// The shared `recognized OOXML package of this kind that cannot be read` outcome.
fn unreadable(family: Family) -> DocOutcome {
    DocOutcome::Unreadable {
        reason: corrupt_reason(family),
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;
    use std::io::Write;

    pub(crate) const DOCX_BODY: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p><w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p></w:body></w:document>"#;

    pub(crate) const COLUMN_CHART: &str = r#"<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>Sales</a:t></a:r></a:p></c:rich></c:tx></c:title><c:plotArea><c:barChart><c:barDir val="col"/><c:ser><c:tx><c:strRef><c:f>Sheet1!$B$1</c:f><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt></c:strCache></c:strRef></c:tx><c:cat><c:strCache><c:pt idx="0"><c:v>Jan</c:v></c:pt><c:pt idx="1"><c:v>Feb</c:v></c:pt></c:strCache></c:cat><c:val><c:numCache><c:pt idx="0"><c:v>10</c:v></c:pt><c:pt idx="1"><c:v>20</c:v></c:pt></c:numCache></c:val></c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>"#;

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
            // The notes relationship is not the first one, so resolving by position would pick the
            // image.
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
            unshown,
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
        assert_eq!(unshown.lines(), ["1 chart(s) not shown"]);
        assert!(notes.is_empty(), "the report carries the chart loss");
    }

    #[test]
    fn pptx_reports_a_deck_s_media_by_kind() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Only slide</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
            ),
            ("ppt/media/video1.mp4", b"video bytes"),
            ("ppt/media/sound.m4a", b"audio bytes"),
            ("ppt/media/image1.emf", b"EMF bytes this stack cannot decode"),
            ("ppt/media/scan.tiff", b"TIFF bytes this stack does not convert"),
            ("ppt/media/pic.png", b"\x89PNG\r\n\x1a\nfake image bytes"),
            ("ppt/embeddings/oleObject1.bin", b"OLE bytes"),
            ("ppt/charts/chart1.xml", b"<chart/>"),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text {
            images,
            notes,
            unshown,
            ..
        } = convert_pptx(&bytes, dir.path())
        else {
            panic!("expected Text outcome for a well-formed pptx");
        };
        assert_eq!(
            unshown.lines(),
            [
                "1 chart(s) not shown",
                "1 embedded object part(s) not shown",
                "1 embedded drawing(s) not shown",
                "1 embedded video(s) not shown",
                "1 embedded audio(s) not shown",
            ]
        );
        assert_eq!(
            notes,
            ["skipped 1 embedded image(s) in a format this pipeline cannot convert"]
        );
        assert_eq!(images, vec![dir.path().join("pic.png")]);
    }

    #[test]
    fn pptx_reads_a_slide_outside_the_conventional_slides_directory() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides2/slide1.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides2/slide1.xml",
                br#"<p:sld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Out-of-tree text</a:t></a:r></a:p></p:txBody></p:sp><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId2"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides2/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/></Relationships>"#,
            ),
            (
                "ppt/notesSlides/notesSlide1.xml",
                br"<p:notes><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Notes for the out-of-tree slide</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>",
            ),
            (
                "ppt/diagrams/data1.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt type="node"><dgm:t><a:p><a:r><a:t>North</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, notes, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  Out-of-tree text\n  (diagram text)\n    North\n\n\
             Slide 1 notes:\n  Notes for the out-of-tree slide"
        );
        assert!(notes.is_empty(), "no part may be reported lost: {notes:?}");
    }

    /// A hidden slide is marked, keeps its number, and still says it is hidden when it holds
    /// nothing; `showMasterSp` is not the slide's own show flag.
    #[test]
    fn pptx_marks_a_hidden_slide_and_keeps_its_number() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/><p:sldId r:id="rId3"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/><Relationship Id="rId3" Target="slides/slide3.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld show="0"><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Hidden</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld showMasterSp="0"><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Shown</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/slide3.xml",
                br#"<p:sld show="0"><p:spTree/></p:sld>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1: (hidden slide)\n  Hidden\n\n\
             Slide 2:\n  Shown\n\n\
             Slide 3: (hidden slide) (no text)"
        );
    }

    #[test]
    fn pptx_reports_the_slide_parts_it_cannot_read() {
        let presentation: &'static [u8] = br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#;
        let presentation_rels: &'static [u8] = br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/></Relationships>"#;
        let slide: &'static [u8] = br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>";
        let notes_rels: &'static [u8] = br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/></Relationships>"#;
        let comments_rels: &'static [u8] = br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments/comment1.xml"/></Relationships>"#;
        // One slide, with its own relationships part and whatever parts the case says the package
        // holds beside it.
        let deck = |rels: &'static [u8], extra: &[(&'static str, &'static [u8])]| {
            let mut entries: Vec<(&'static str, &'static [u8])> = vec![
                ("ppt/presentation.xml", presentation),
                ("ppt/_rels/presentation.xml.rels", presentation_rels),
                ("ppt/slides/slide1.xml", slide),
                ("ppt/slides/_rels/slide1.xml.rels", rels),
            ];
            entries.extend_from_slice(extra);
            entries
        };
        // A deck declaring three slides, the second of them never declared in
        // the presentation's relationships.
        let three_slides: &'static [u8] = br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/><p:sldId id="258" r:id="rId3"/></p:sldIdLst></p:presentation>"#;
        let three_slides_rels: &'static [u8] = br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId3" Target="slides/slide2.xml"/></Relationships>"#;
        let first_slide: &'static [u8] = br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>First</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>";
        let third_slide: &'static [u8] = br"<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Third</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>";
        let readable = "Slide 1:\n  Slide text";
        for (label, fixture, expected) in [
            (
                "a notes part the package does not hold",
                deck(notes_rels, &[]),
                (readable, ["slide 1 notes could not be read"].as_slice()),
            ),
            (
                "a relationships part cut mid-tag",
                deck(br#"<Relationships><Relationship Id="rId"#, &[]),
                (
                    readable,
                    ["slide 1 relationships could not be read"].as_slice(),
                ),
            ),
            (
                "a relationships part cut at an element boundary",
                deck(
                    br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>"#,
                    &[],
                ),
                (
                    readable,
                    ["slide 1 relationships could not be read"].as_slice(),
                ),
            ),
            (
                "a comments part the package does not hold",
                deck(comments_rels, &[]),
                (readable, ["slide 1 comments could not be read"].as_slice()),
            ),
            (
                "a comments part holding an error page",
                deck(
                    comments_rels,
                    &[("ppt/comments/comment1.xml", b"<html>404</html>")],
                ),
                (readable, ["slide 1 comments could not be read"].as_slice()),
            ),
            (
                "a slide whose own relationship never resolves",
                vec![
                    ("ppt/presentation.xml", three_slides),
                    ("ppt/_rels/presentation.xml.rels", three_slides_rels),
                    ("ppt/slides/slide1.xml", first_slide),
                    ("ppt/slides/slide2.xml", third_slide),
                ],
                (
                    "Slide 1:\n  First\n\nSlide 3:\n  Third",
                    ["slide 2 could not be read"].as_slice(),
                ),
            ),
        ] {
            let (expected_text, expected_notes) = expected;
            let bytes = zip_fixture(&fixture);
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_pptx(&bytes, dir.path()) else {
                panic!("{label}: expected Text outcome for a readable presentation");
            };
            assert_eq!(text, expected_text, "{label}: the slides it could read");
            assert_eq!(
                notes, expected_notes,
                "{label}: the part it could not read is named once"
            );
        }
    }

    #[test]
    fn pptx_reads_comments_with_their_authors() {
        let legacy = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/commentAuthors" Target="commentAuthors.xml"/></Relationships>"#,
            ),
            (
                "ppt/commentAuthors.xml",
                br#"<p:cmAuthorLst><p:cmAuthor id="0" name="Anna Petrova" initials="AP"/><p:cmAuthor id="1" name="Ivan Petrov" initials="IP"/></p:cmAuthorLst>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br"<p:sld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>",
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments/comment1.xml"/></Relationships>"#,
            ),
            (
                "ppt/comments/comment1.xml",
                br#"<p:cmLst><p:cm authorId="0" idx="1" dt="2026-01-01T00:00:00"><p:pos x="1" y="2"/><p:text>needs a citation</p:text></p:cm><p:cm authorId="7" idx="2"><p:text>agreed</p:text></p:cm><p:cm authorId="1" idx="3"><p:text>  </p:text></p:cm><p:cm authorId="7" idx="4"><p:text/></p:cm></p:cmLst>"#,
            ),
        ]);
        let threaded = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Type="http://schemas.microsoft.com/office/2018/10/relationships/authors" Target="authors.xml"/></Relationships>"#,
            ),
            (
                "ppt/authors.xml",
                br#"<p188:authorLst><p188:author id="{5C4A1C0A-0000-0000-0000-000000000001}" name="Anna Petrova" initials="AP" userId="anna@example.com" providerId="AD"/><p188:author id="{5C4A1C0A-0000-0000-0000-000000000002}" name="Ivan Petrov" initials="IP" userId="ivan@example.com" providerId="AD"/></p188:authorLst>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br"<p:sld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>",
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.microsoft.com/office/2018/10/relationships/comments" Target="../comments/modernComment_1.xml"/></Relationships>"#,
            ),
            (
                "ppt/comments/modernComment_1.xml",
                br#"<p188:cmLst><p188:cm authorId="{5C4A1C0A-0000-0000-0000-000000000001}" idx="1"><p188:txBody><a:p><a:r><a:t>Needs</a:t></a:r><a:br/><a:r><a:t>a citation</a:t></a:r><a:fld id="1" type="slidenum"><a:t>7</a:t></a:fld></a:p><a:p><a:r><a:t>for the figure</a:t></a:r></a:p></p188:txBody><p188:replyLst><p188:reply authorId="{5C4A1C0A-0000-0000-0000-000000000002}"><p188:txBody><a:p><a:r><a:t>Agreed</a:t></a:r></a:p></p188:txBody></p188:reply></p188:replyLst></p188:cm></p188:cmLst>"#,
            ),
        ]);
        for (label, fixture, expected) in [
            (
                "legacy comments",
                legacy,
                "Slide 1:\n  Slide text\n\n\
                 Slide 1 comments:\n  Anna Petrova: needs a citation\n  agreed\n  \
                 Ivan Petrov: (no text)\n  (no text)",
            ),
            (
                "2018 threaded comments",
                threaded,
                "Slide 1:\n  Slide text\n\n\
                 Slide 1 comments:\n  Anna Petrova: Needs a citation for the figure\n  Ivan Petrov: Agreed",
            ),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, notes, .. } = convert_pptx(&fixture, dir.path()) else {
                panic!("{label}: expected Text outcome for a readable presentation");
            };
            assert_eq!(
                text, expected,
                "{label}: the comments printed under the slide"
            );
            assert!(
                notes.is_empty(),
                "{label}: the comments are shown: {notes:?}"
            );
        }
    }

    fn own_title_placeholders() -> Vec<u8> {
        zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/><p:sldId r:id="rId3"/><p:sldId r:id="rId4"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/><Relationship Id="rId3" Target="slides/slide3.xml"/><Relationship Id="rId4" Target="slides/slide4.xml"/></Relationships>"#,
            ),
            // An own `type="title"` marks the placeholder; a shape with no
            // `<p:ph>` (a heading written as a plain text box) is ordinary text.
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Quarterly review</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:txBody><a:p><a:r><a:t>Heading as a text box</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            // `ctrTitle` is a title too.
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="ctrTitle"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Centred title</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            // An empty title placeholder adds no mark; the body still reads.
            (
                "ppt/slides/slide3.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p/></p:txBody></p:sp><p:sp><p:txBody><a:p><a:r><a:t>Only body</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            // A slide that holds only its title must not read `(no text)`.
            (
                "ppt/slides/slide4.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Title only</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
        ])
    }

    /// A deck whose five slides inherit their titles from the layout parts their
    /// own relationships name.
    fn layout_inherited_titles() -> Vec<u8> {
        zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/><p:sldId r:id="rId3"/><p:sldId r:id="rId4"/><p:sldId r:id="rId5"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/><Relationship Id="rId3" Target="slides/slide3.xml"/><Relationship Id="rId4" Target="slides/slide4.xml"/><Relationship Id="rId5" Target="slides/slide5.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph idx="10"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Inherited title</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:nvSpPr><p:nvPr><p:ph idx="11"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Body</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            // The same `idx` under a layout that declares its title elsewhere.
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph idx="10"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Not a title here</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            // A second slide of the layout the first one uses.
            (
                "ppt/slides/slide3.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph idx="10"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Shared layout title</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            // A placeholder naming neither type nor `idx`: the schema defaults
            // it to the layout's `idx` 0, which that layout declares a title.
            (
                "ppt/slides/slide4.xml",
                br"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Defaulted title</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>",
            ),
            // ...but the same layout's `idx` 1 is its body, whatever the type
            // defaults to.
            (
                "ppt/slides/slide5.xml",
                br#"<p:sld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="obj" idx="1"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Defaulted body</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/_rels/slide2.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout2.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/_rels/slide3.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/_rels/slide4.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout3.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/_rels/slide5.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout3.xml"/></Relationships>"#,
            ),
            (
                "ppt/slideLayouts/slideLayout1.xml",
                br#"<p:sldLayout><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="title" idx="10"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>LAYOUT TITLE</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="11"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>LAYOUT BODY</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sldLayout>"#,
            ),
            (
                "ppt/slideLayouts/slideLayout2.xml",
                br#"<p:sldLayout><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="title" idx="20"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>OTHER LAYOUT TITLE</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sldLayout>"#,
            ),
            // Its title carries no `idx`, which the schema defaults to 0.
            (
                "ppt/slideLayouts/slideLayout3.xml",
                br#"<p:sldLayout><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>DEFAULTED LAYOUT TITLE</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>LAYOUT BODY</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sldLayout>"#,
            ),
        ])
    }

    #[test]
    fn pptx_marks_a_title_from_the_slide_or_its_layout() {
        for (label, fixture, expected) in [
            (
                "titles the slide itself declares",
                own_title_placeholders(),
                "Slide 1:\n  (title) Quarterly review\n  Heading as a text box\n\n\
                 Slide 2:\n  (title) Centred title\n\n\
                 Slide 3:\n  Only body\n\n\
                 Slide 4:\n  (title) Title only",
            ),
            (
                "titles inherited from the slide's layout",
                layout_inherited_titles(),
                "Slide 1:\n  (title) Inherited title\n  Body\n\n\
                 Slide 2:\n  Not a title here\n\n\
                 Slide 3:\n  (title) Shared layout title\n\n\
                 Slide 4:\n  (title) Defaulted title\n\n\
                 Slide 5:\n  Defaulted body",
            ),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, .. } = convert_pptx(&fixture, dir.path()) else {
                panic!("{label}: expected Text outcome for a readable presentation");
            };
            assert_eq!(text, expected, "{label}: the title marks printed");
        }
    }

    #[test]
    fn pptx_reads_diagram_text_from_the_data_model() {
        let points = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/></Relationships>"#,
            ),
            (
                "ppt/diagrams/data1.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt type="node"><dgm:prSet phldrT="[Text]"/><dgm:t><a:p><a:r><a:t>North</a:t></a:r></a:p><a:p><a:fld id="1" type="slidenum"><a:t>7</a:t></a:fld></a:p></dgm:t></dgm:pt><dgm:pt type="node"><dgm:prSet phldr="1" phldrT="[Text]"/><dgm:t><a:p><a:r><a:t>Placeholder</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt type="asst"><dgm:t><a:p><a:r><a:t>Assistant</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:t><a:p><a:r><a:t>South</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
            // A slide whose only diagram is read but shows no text still holds
            // its diagram: it must not read as a slide with no text at all.
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide2.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data2.xml"/></Relationships>"#,
            ),
            (
                "ppt/diagrams/data2.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt type="doc"><dgm:prSet phldr="1"/><dgm:t><a:p><a:r><a:t>Hint</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
        ]);
        let once_per_slide = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/></Relationships>"#,
            ),
            // `rId1` twice, in shapes either side of `rId2`: two sections, not
            // three, and in shape order.
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId2"/></a:graphicData></a:graphic></p:graphicFrame><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            // The same data part reads again on its own slide.
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data2.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/_rels/slide2.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/></Relationships>"#,
            ),
            (
                "ppt/diagrams/data1.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt type="node"><dgm:t><a:p><a:r><a:t>North</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt type="node"><dgm:t><a:p><a:r><a:t>South</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
            (
                "ppt/diagrams/data2.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt type="node"><dgm:t><a:p><a:r><a:t>Alpha</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
        ]);
        for (label, fixture, expected) in [
            (
                "points, their types and the placeholders among them",
                points,
                "Slide 1:\n  (diagram text)\n    North\n    Assistant\n    South\n\n\
                 Slide 2:\n  (diagram text)",
            ),
            (
                "each diagram once per slide, in shape order",
                once_per_slide,
                "Slide 1:\n  (diagram text)\n    North\n    South\n  (diagram text)\n    Alpha\n\n\
                 Slide 2:\n  (diagram text)\n    North\n    South",
            ),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let DocOutcome::Text { text, .. } = convert_pptx(&fixture, dir.path()) else {
                panic!("{label}: expected Text outcome for a readable presentation");
            };
            assert_eq!(text, expected, "{label}: the diagram text printed");
        }
    }

    #[test]
    fn pptx_reports_a_diagram_whose_data_cannot_be_read() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/><p:sldId r:id="rId3"/><p:sldId r:id="rId4"/><p:sldId r:id="rId5"/><p:sldId r:id="rId6"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/><Relationship Id="rId3" Target="slides/slide3.xml"/><Relationship Id="rId4" Target="slides/slide4.xml"/><Relationship Id="rId5" Target="slides/slide5.xml"/><Relationship Id="rId6" Target="slides/slide6.xml"/></Relationships>"#,
            ),
            // The data part the relationship names is not in the package.
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/></Relationships>"#,
            ),
            // No relationship resolves the id at all.
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            // The data model is there but the package cut it short: the text
            // after the cut is not passed off as everything it held.
            (
                "ppt/slides/slide3.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide3.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data3.xml"/></Relationships>"#,
            ),
            (
                "ppt/diagrams/data3.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt type="node"><dgm:t><a:p><a:r><a:t>Cut off</a:t>"#,
            ),
            // A part that parses as XML but is no diagram data model (an error
            // page a producer left in its place) is named as lost too.
            (
                "ppt/slides/slide4.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide4.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data4.xml"/></Relationships>"#,
            ),
            ("ppt/diagrams/data4.xml", br"<html>404 Not Found</html>"),
            // No diagram at all.
            (
                "ppt/slides/slide5.xml",
                br"<p:sld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Plain</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>",
            ),
            (
                "ppt/slides/slide6.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:lo="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  (diagram text could not be read)\n\n\
             Slide 2:\n  (diagram text could not be read)\n\n\
             Slide 3:\n  (diagram text could not be read)\n\n\
             Slide 4:\n  (diagram text could not be read)\n\n\
             Slide 5:\n  Plain\n\n\
             Slide 6:\n  (diagram text could not be read)"
        );
    }

    #[test]
    fn pptx_diagram_text_is_the_text_a_reading_showed_per_slide() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/><p:sldId r:id="rId2"/><p:sldId r:id="rId3"/><p:sldId r:id="rId4"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/><Relationship Id="rId3" Target="slides/slide3.xml"/><Relationship Id="rId4" Target="slides/slide4.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId2"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data2.xml"/></Relationships>"#,
            ),
            // A placeholder point is skipped and a hint alone is not: the same
            // rule the reading marks its diagram text by.
            (
                "ppt/diagrams/data1.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt><dgm:t><a:p><a:r><a:t>North</a:t></a:r></a:p><a:p><a:r><a:t>Second line</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:prSet phldr="1"/><dgm:t><a:p><a:r><a:t>Placeholder</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt><dgm:prSet phldrT="[Text]"/><dgm:t><a:p><a:r><a:t>Hinted</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
            (
                "ppt/diagrams/data2.xml",
                br"<dgm:dataModel><dgm:ptLst><dgm:pt><dgm:t><a:p><a:r><a:t>South</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>",
            ),
            // A diagram the package cut short: its text is lost, so the slide is
            // absent rather than holding text a refusal could quote.
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide2.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data3.xml"/></Relationships>"#,
            ),
            (
                "ppt/diagrams/data3.xml",
                br"<dgm:dataModel><dgm:ptLst><dgm:pt><dgm:t><a:p><a:r><a:t>Cut off</a:t>",
            ),
            // A slide with no diagram at all, and one whose diagram is only
            // placeholders, both contribute nothing.
            (
                "ppt/slides/slide3.xml",
                br"<p:sld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Plain</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:sld>",
            ),
            (
                "ppt/slides/slide4.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide4.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data4.xml"/></Relationships>"#,
            ),
            (
                "ppt/diagrams/data4.xml",
                br#"<dgm:dataModel><dgm:ptLst><dgm:pt><dgm:prSet phldr="1"/><dgm:t><a:p><a:r><a:t>Placeholder</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#,
            ),
        ]);
        let text = pptx_diagram_text(&bytes);
        assert_eq!(text.len(), 1, "only slide 1 shows text: {text:?}");
        assert_eq!(
            text.get("ppt/slides/slide1.xml"),
            Some(&vec![
                "North\nSecond line\nHinted".to_owned(),
                "South".to_owned()
            ]),
            "one entry per diagram that shows text, in shape order: {text:?}"
        );

        // A package that is not a presentation holds no diagram text, and one
        // whose bytes are not a package at all neither.
        assert!(pptx_diagram_text(b"not a package").is_empty());
        assert!(pptx_diagram_text(&zip_fixture(&[("word/document.xml", DOCX_BODY)])).is_empty());
    }

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
}
