//! OOXML package reading: the ZIP half of document conversion.
//!
//! [`crate::document`] owns content-first detection and the PDF arm — the one
//! non-ZIP container. Every other format it converts is an OOXML package: a ZIP
//! of XML parts, read here with the shared `zip` + `quick_xml` plumbing. Word,
//! Excel and PowerPoint differ only in which parts hold the text, how a cell
//! address is written, and where the embedded media live — and in how much of a
//! walk the format needs: Word's is the only one that carries tables, tracked
//! changes and content beyond the body part, so its reader is the module
//! [`docx`]; Excel's is the module [`xlsx`]; and PowerPoint's is a slide walk
//! that reaches one part further for the two things a slide only points at —
//! its layout, whose title placeholders it borrows, and the data model of each
//! SmartArt diagram it shows — so it still lives in this module.
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

use crate::docgen::ppt_marks;
use crate::document::{DocOutcome, SkipReason, SkippedImages, ensure_out_dir};
use crate::reader_output::{slide_header, slide_notes_header, text_block, text_lines};
use quick_xml::Reader;
use quick_xml::events::{BytesRef, BytesStart, Event};
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

mod docx;
pub(crate) use docx::convert_docx;
mod xlsx;
pub(crate) use xlsx::convert_xlsx;

// ── Package parts and bounds ────────────────────────────────────

/// Maximum bytes decompressed from a single ZIP entry, so a lying size header
/// cannot inflate the temp dir. Deliberately not an aggregate bound: the entry
/// count is unbounded.
const MAX_ZIP_ENTRY_BYTES: u64 = 64 * 1024 * 1024;

/// Part listing the presentation's slides in slide order.
const PPT_PRESENTATION_PART: &str = "ppt/presentation.xml";
/// Relationships of the presentation part: slide relationship id -> slide part.
const PPT_PRESENTATION_RELS_PART: &str = "ppt/_rels/presentation.xml.rels";
/// Base every presentation relationship target resolves against.
const PPT_BASE: &str = "ppt/";
/// Prefix of the slide parts.
const PPT_SLIDES_PREFIX: &str = "ppt/slides/";
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
    // A deck's slides share a handful of layouts, so a layout's title indices are
    // read once for the whole presentation rather than once per slide; a diagram
    // a slide shows and another shows too is parsed once here for the same
    // reason.
    let mut layouts: HashMap<String, HashSet<String>> = HashMap::new();
    let mut diagrams = DiagramCache::default();
    for (index, part) in parts.iter().enumerate() {
        let number = index + 1;
        // A slide the presentation declares but the package does not resolve
        // costs its own number and a note: dropping it would renumber the slides
        // after it, and label their content with someone else's number.
        let Some(part) = part.as_deref() else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        // Everything the slide owns — its relationships, and the notes, layout
        // and diagram parts they name — resolves against the directory the
        // package actually put the slide part in, never the conventional
        // `ppt/slides/`.
        let base = part_dir(part);
        let Some(xml) = read_zip_entry(&mut archive, part).bytes() else {
            notes.push(format!("slide {number} could not be read"));
            continue;
        };
        let rels = slide_rels(&mut archive, part);
        // The layout is consulted for its title placeholders — and for nothing
        // else: its text never enters the answer.
        let layout_titles = layout_title_indices(&mut archive, base, &rels, &mut layouts);
        // A title placeholder names its type on the slide, or leaves it to the
        // layout; a placeholder that names neither takes the schema's defaults
        // (`obj` and idx 0), so the layout is consulted for those too.
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
        let shown = slide_diagrams(&mut archive, base, &xml, &rels, &mut diagrams);
        blocks.push(slide_block(number, slide_hidden(&xml), &content, &shown));
        match slide_notes(&mut archive, base, &rels) {
            SlideNotes::Text(lines) if !lines.is_empty() => {
                blocks.push(text_lines(&slide_notes_header(number), &lines));
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

/// The diagram text a reading shows for each slide of a presentation package,
/// keyed by the slide part the reading numbers (`ppt/slides/slide1.xml`): the
/// text a slide's diagrams show — text the reader prints and the editor does not
/// change.
///
/// Read here rather than by the document kit's own refusals, so what a diagram
/// shows is decided by the reader's rule alone (see [`slide_diagrams`] and
/// [`diagram_text_lines`]). A slide whose diagrams read as lost, one whose
/// diagrams show no text, and one whose part cannot be read at all are absent
/// from the map: nothing here names text a reading did not show.
///
/// Best effort by design: a package that is not a presentation, or one whose
/// bytes do not open, yields an empty map, and a refusal then keeps its generic
/// wording.
pub(crate) fn pptx_diagram_text(bytes: &[u8]) -> HashMap<String, Vec<String>> {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return HashMap::new();
    };
    let Some(parts) = resolve_slide_parts(&mut archive) else {
        return HashMap::new();
    };
    let mut cache = DiagramCache::default();
    let mut text = HashMap::new();
    for part in parts.into_iter().flatten() {
        let Some(xml) = read_zip_entry(&mut archive, &part).bytes() else {
            continue;
        };
        let rels = slide_rels(&mut archive, &part);
        let shown: Vec<String> =
            slide_diagrams(&mut archive, part_dir(&part), &xml, &rels, &mut cache)
                .into_iter()
                .filter_map(|diagram| match diagram {
                    // One entry per diagram that shows text: a diagram shows its
                    // points' lines joined, the way a reading prints them, and a
                    // diagram with nothing to show names nothing.
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

/// The relationships a slide part declares through the `.rels` part beside it.
enum SlideRels {
    /// The slide declares no relationships at all (no `.rels` part).
    None,
    /// The `.rels` part is there but could not be read.
    Unreadable,
    /// The relationships it declares, in document order.
    List(Vec<Relationship>),
}

/// The relationships of the slide part `slide_part`, read from the `.rels` part
/// beside it ([`rels_part_of`]). A slide with no relationships part declares
/// none — but one that IS there and could not be read may name a title layout,
/// diagrams or notes, and the two are kept apart rather than rescanned per
/// caller.
fn slide_rels<R: Read + Seek>(archive: &mut ZipArchive<R>, slide_part: &str) -> SlideRels {
    match read_zip_entry(archive, &rels_part_of(slide_part)) {
        ZipEntry::Missing => SlideRels::None,
        ZipEntry::Unreadable => SlideRels::Unreadable,
        // The part is there, but its XML could not be read: it may name parts.
        ZipEntry::Bytes(rels) => match relationships(&rels) {
            Some(list) => SlideRels::List(list),
            None => SlideRels::Unreadable,
        },
    }
}

/// A slide's speaker notes, as resolved through the relationships beside the
/// slide part.
enum SlideNotes {
    /// The slide declares no notes part — the usual case.
    None,
    /// The notes part's text lines.
    Text(Vec<String>),
    /// The slide declares a notes part, but it could not be read.
    Unreadable,
}

/// The speaker notes a slide declares through `rels` — their targets resolved
/// against the slide part's own directory (`base`) — as their non-empty text
/// lines: never by slide index, and never a note of another slide's.
fn slide_notes<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    rels: &SlideRels,
) -> SlideNotes {
    let rels = match rels {
        SlideRels::None => return SlideNotes::None,
        SlideRels::Unreadable => return SlideNotes::Unreadable,
        SlideRels::List(rels) => rels,
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

/// A part's `<a:t>` text, split by whether it belongs to the slide's title
/// placeholder.
struct SlideText {
    /// The title placeholder's lines, in part order (empty when the slide
    /// declares no title or the title is empty).
    title: Vec<String>,
    /// Every other line the part holds, unchanged from what the reader showed
    /// before.
    lines: Vec<String>,
}

/// The text runs one OOXML part's walk accumulates: every `<a:t>`'s text
/// joined into the paragraph being read, `<a:br/>` breaking it into lines, and
/// the text of an `<a:fld>` (the slide-number/date field) left out. One
/// implementation, so the slide walk and the diagram walk read a run alike.
#[derive(Default)]
struct TextRuns {
    /// The paragraph being read right now: the `<a:t>` text seen since its
    /// `<a:p>` opened, until a paragraph end moves it out.
    paragraph: String,
    /// Whether an `<a:t>` is open: its text is run text, and no other element's
    /// is.
    in_text: bool,
    /// How many `<a:fld>` elements are open: their text is the field's, not the
    /// run's.
    fields: usize,
}

impl TextRuns {
    /// Note one event of the part's walk, accumulating the run text, and return
    /// `true` when the event is the `</a:p>` that ends a paragraph — so the
    /// caller can read the paragraph where it belongs before the next one opens.
    fn note(&mut self, event: &Event<'_>) -> bool {
        match event {
            Event::Start(event) => match event.local_name().as_ref() {
                b"t" => self.in_text = true,
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
                b"t" => self.in_text = false,
                b"fld" => self.fields = self.fields.saturating_sub(1),
                b"p" => return true,
                _ => {}
            },
            _ => {}
        }
        false
    }

    /// Clear the paragraph being read: the caller read it where it belongs to
    /// no block of the part, so it must not leak into a later one.
    fn discard(&mut self) {
        self.paragraph.clear();
    }

    /// Move the paragraph being read into `lines` ([`push_lines`]), one line per
    /// `<a:br/>` break.
    fn lines(&mut self, lines: &mut Vec<String>) {
        push_lines(lines, &mut self.paragraph);
    }
}

/// A part's `<a:t>` text, in document order: a newline at `<a:br/>` and at each
/// `<a:p>` end, and nothing from an `<a:fld>` (the slide-number/date field).
/// Text of a `<p:sp>` whose `<p:ph>` declares the slide's title goes to
/// [`SlideText::title`]; every other line — a table cell's, a group's children's,
/// anything outside a shape — goes to [`SlideText::lines`]. `is_title` receives
/// each `<p:ph>`'s `type` and `idx`. Empty paragraphs are dropped; `None` on any
/// XML error. The runs themselves are read by [`TextRuns`].
fn slide_text(
    xml: &[u8],
    is_title: impl Fn(Option<&str>, Option<&str>) -> bool,
) -> Option<SlideText> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    // One entry per open `<p:sp>`: whether that shape is the title. `<p:ph>`
    // sets the innermost entry, and a run with no shape open is body text.
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
            // The shape a paragraph sits in is still open here, so its title
            // flag is the paragraph's own.
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

/// Set the innermost open `<p:sp>`'s title flag from a `<p:ph>`: does nothing
/// when no shape is open.
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

/// Whether the presentation hides the slide: `<p:sld ... show="0">`. Matching
/// the attribute by local name is what keeps `showMasterSp` and
/// `showMasterPhAnim` — `show`-prefixed but not the slide's own show flag — from
/// being mistaken for it.
fn slide_hidden(xml: &[u8]) -> bool {
    scan_elements(xml, b"sld", |event| attr(event, b"show"))
        .and_then(|values| values.into_iter().next())
        .is_some_and(|value| value == "0" || value.eq_ignore_ascii_case("false"))
}

/// The `idx` values of the title placeholders the slide's own layout declares,
/// memoized in `cache` by layout part; the layout's own target resolves against
/// the slide part's directory (`base`).
///
/// A slide's title placeholder sometimes carries its type on the layout rather
/// than on the slide, so the layout is consulted for its title indices — and for
/// nothing else: its text never enters the answer. `None` when the slide
/// declares no layout relationship at all; a layout that is missing, unreadable
/// or unparsable reads as one declaring nothing.
fn layout_title_indices<'a, R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    rels: &SlideRels,
    cache: &'a mut HashMap<String, HashSet<String>>,
) -> Option<&'a HashSet<String>> {
    let SlideRels::List(rels) = rels else {
        return None;
    };
    let target = rels
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
    /// The diagram's data could not be read: its text is lost, and the slide
    /// says so rather than reading as one without a diagram.
    Lost,
}

/// The text a deck's diagram data parts were read as, keyed by part: a deck that
/// shows one diagram on several slides reads its part once, the way its layouts
/// are read once for the whole presentation.
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
/// their data parts through the relationships the slide part declares, their
/// targets resolved against its own directory (`base`) and their text taken from
/// `cache`, so a part a deck shows on several slides is parsed once.
///
/// Every diagram is read once per slide: a data part a slide refers to several
/// times yields one section. A slide with no diagram reference, and a slide
/// whose own XML cannot be read, present nothing.
fn slide_diagrams<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    base: &str,
    xml: &[u8],
    rels: &SlideRels,
    cache: &mut DiagramCache,
) -> Vec<DiagramRead> {
    // A `dgm:relIds` carries `r:dm`, `r:lo`, `r:qs` and `r:cs`; `attr` matches by
    // local name, so this is the data-model reference. Every `relIds` a slide
    // holds is collected, the attribute or not: a frame whose `relIds` names no
    // data model (a package that breaks the schema, which requires all four)
    // still shows a diagram, and one read as no diagram at all would be dropped
    // in silence.
    let Some(ids) = scan_elements(xml, b"relIds", |event| Some(attr(event, b"dm"))) else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    let mut diagrams = Vec::new();
    for id in ids {
        // No reference to resolve and nothing to dedupe on: the frame keeps its
        // own section, and its text reads as lost.
        let Some(id) = id else {
            diagrams.push(DiagramRead::Lost);
            continue;
        };
        let part = match rels {
            SlideRels::List(rels) => rels
                .iter()
                .find(|rel| rel.id == id)
                .map(|rel| resolve_part(base, &rel.target)),
            SlideRels::None | SlideRels::Unreadable => None,
        };
        // Dedupe on the resolved part (never across slides, which read the same
        // diagram with their own block); an id nothing resolves keeps its own
        // section, so it is not merged with an unrelated lost diagram.
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

/// The text a person sees in a diagram's data model: the `<dgm:t>` text of
/// every `<dgm:pt>` (a point) the diagram does not flag as an editor
/// placeholder, flat, in `dgm:ptLst` document order. The structure (the
/// `dgm:cxn` tree) is not reproduced, and every point type is read (`node`,
/// `asst`, `pres`, ...; an absent `type` is the schema's `node`) — only a
/// flagged placeholder is skipped, and a point's text is kept only once the
/// whole point has been walked, so a producer that writes its `<dgm:prSet>`
/// after its `<dgm:t>` cannot leak a hint.
///
/// `None` for a part that is not a diagram's data model, one the package cut
/// short, or one the package left empty (bytes that are only whitespace, see
/// [`blank`]): a diagram's text is lost in every case, which the slide states
/// rather than reading as one with no diagram. The empty case overrides
/// [`Part::unreadable`]'s rule that a blank entry declares nothing — for a
/// diagram an entry that declares no data model holds no text either, which is
/// data absent, not a section with nothing under it.
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
                    // Text read before the point opened belongs to no node: it
                    // is discarded here rather than leaking into this point.
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
            // A paragraph read with no point open belongs to no node, and must
            // not leak into the next one.
            match points.last_mut() {
                Some(point) => runs.lines(&mut point.lines),
                None => runs.discard(),
            }
        }
        buffer.clear();
    }
    (!part.unreadable() && !blank(xml)).then_some(lines)
}

/// Mark the innermost open `<dgm:pt>` a placeholder from the `<dgm:prSet>` it
/// holds: the point's last `prSet` decides (one without a `phldr` clears the
/// flag), and PowerPoint writes one. A placeholder's carries `phldr` of
/// `"1"`/`"true"` (case-insensitive), and PowerPoint draws its hint instead of
/// the text the point holds. A `phldrT` alone is not that flag — ECMA-376 gives
/// a point carrying its own text both the hint and the flag, and the text is
/// what a person sees.
fn mark_placeholder(points: &mut [DiagramPoint], event: &BytesStart<'_>) {
    let Some(point) = points.last_mut() else {
        return;
    };
    point.placeholder = attr(event, b"phldr")
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
}

/// One slide block: `Slide {n}:`, ` (hidden slide)` when the presentation hides
/// it, then the title lines, the body lines, and one section per diagram — a
/// diagram section being the `(diagram text)` heading over its indented node
/// lines (none, for a diagram that shows no text), or the
/// `(diagram text could not be read)` mark. A slide holding a diagram thus never
/// reads `(no text)`, which [`text_block`] supplies only for a slide holding
/// nothing at all.
fn slide_block(
    number: usize,
    hidden: bool,
    content: &SlideText,
    diagrams: &[DiagramRead],
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
    text_block(&header, &marks.no_text, &lines)
}

// ── Shared package plumbing ─────────────────────────────────────

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

/// The relationships a rels part declares, in document order, and whether the part
/// is whole: the walk that reads the list also says whether it ended inside an
/// element, so a caller whose relationships part the package cut short reports the
/// loss rather than reading the shortened list as everything the part declared —
/// without a pass of its own, which is what tells the two apart at all.
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

/// One `<Relationship Id Type Target/>` entry, or `None` when it names no id or no
/// target.
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

/// Relationship id -> target, for the parts a rId alone names.
fn relationship_map(xml: &[u8]) -> HashMap<String, String> {
    relationship_targets(&relationships(xml).unwrap_or_default())
}

/// [`relationship_map`] over an already-read relationship list. The Word and
/// Excel readers read the list themselves — the notes parts are named there by
/// kind, before the list is turned into this lookup — and share the mapping.
fn relationship_targets(rels: &[Relationship]) -> HashMap<String, String> {
    rels.iter()
        .map(|rel| (rel.id.clone(), rel.target.clone()))
        .collect()
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

/// The directory a part's own relationship targets resolve against
/// (`ppt/slides/slide1.xml` -> `ppt/slides/`): a package may put a part
/// anywhere the relationship that names it points.
fn part_dir(part: &str) -> &str {
    part.rfind('/').map_or("", |slash| &part[..=slash])
}

/// The relationships part of `part` (`ppt/slides/slide1.xml` ->
/// `ppt/slides/_rels/slide1.xml.rels`), by the OPC convention the document
/// kit's `relsPartFor` follows: wherever the package put the part, its
/// relationships are beside it.
fn rels_part_of(part: &str) -> String {
    let name = part.rsplit_once('/').map_or(part, |(_, name)| name);
    format!("{}_rels/{name}.rels", part_dir(part))
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
/// never as "no such element" ([`relationships`]).
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

/// Whether a part's bytes carry nothing but whitespace — a byte-order mark included,
/// which says how a document is encoded rather than carrying any of it: an entry the
/// package left empty, which declares nothing rather than holding something unread.
fn blank(bytes: &[u8]) -> bool {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    bytes.iter().all(u8::is_ascii_whitespace)
}

/// What a part's own walk says about the part. Every reader that reads a part from
/// its first event to its last notes each event here, so the part's account of
/// itself — whole, or cut at an element boundary, or no document at all — comes out
/// of the walk that reads its content, rather than out of a pass of its own.
struct Part<'a> {
    /// The part's bytes: what tells an entry the package left empty from one whose
    /// bytes are no document.
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

    /// Note one event of the part's walk: the part's own elements are what the
    /// account is built from, so only their start and end events are noted.
    fn note(&mut self, event: &Event<'_>) {
        match event {
            Event::Start(_) => self.open += 1,
            Event::End(_) => self.open = self.open.saturating_sub(1),
            _ => {}
        }
    }

    /// Note that the walk found an element of the kind the reader reads the part
    /// for, wherever it stands in the part: what tells a part holding what this
    /// reader came for from one holding something else entirely.
    fn kind_seen(&mut self) {
        self.kind = true;
    }

    /// Whether the part stopped with an element still open: quick-xml reports a part
    /// cut at an element boundary as a document that simply ends, so only the
    /// reader's own walk can tell a half-written part from a finished one, and the
    /// content after the cut must not be passed off as content the part never had.
    fn cut(&self) -> bool {
        self.open > 0
    }

    /// Whether the part is not the document it is read as: cut short, or holding no
    /// element of its own kind ([`Self::kind_seen`]) — whether it holds elements of
    /// another kind (an error page a producer left in the part's place,
    /// `<html>404</html>`) or none at all in bytes that are no document (`Not
    /// Found`), either of which parses or reads as happily as the real thing. A
    /// reader that got nothing out of such a part reports the loss; an entry the
    /// package left empty — bytes that are nothing but whitespace ([`blank`]) —
    /// declares nothing instead, so its walk has no element to miss.
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

    /// A slide part may live anywhere the presentation's relationship points,
    /// not just `ppt/slides/`: its own `.rels` sits beside it, and the notes and
    /// diagram parts those relationships name resolve against its own directory.
    /// Read from the conventional prefix instead, the notes would be lost in
    /// silence and the diagram falsely reported as unreadable.
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

    /// A slide's own title placeholder is marked and its text appears exactly
    /// once; a shape without a `<p:ph>` is ordinary text, an empty title adds no
    /// mark, and a slide holding only its title does not read as one with no
    /// text.
    #[test]
    fn pptx_marks_the_title_placeholder_the_slide_declares() {
        let bytes = zip_fixture(&[
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
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  (title) Quarterly review\n  Heading as a text box\n\n\
             Slide 2:\n  (title) Centred title\n\n\
             Slide 3:\n  Only body\n\n\
             Slide 4:\n  (title) Title only"
        );
    }

    /// A title the slide leaves to its layout: the slide's `<p:ph idx="10"/>`
    /// carries no type, and the layout part its own relationship names declares
    /// that `idx` a title. The layout's text never enters the answer, the
    /// layout's body placeholder stays body text, and the answer is the layout's
    /// alone — a slide of another layout, whose title sits at another `idx`, is
    /// left unmarked even though its own placeholder carries the same `idx`. A
    /// placeholder naming neither type nor `idx` takes the schema's defaults,
    /// which the layout is consulted for like any other `idx`.
    #[test]
    fn pptx_marks_a_title_inherited_from_the_layout() {
        let bytes = zip_fixture(&[
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
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  (title) Inherited title\n  Body\n\n\
             Slide 2:\n  Not a title here\n\n\
             Slide 3:\n  (title) Shared layout title\n\n\
             Slide 4:\n  (title) Defaulted title\n\n\
             Slide 5:\n  Defaulted body"
        );
    }

    /// A hidden slide is marked, keeps its number, and still says it is hidden
    /// when it holds nothing; `showMasterSp` is not the slide's own show flag.
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

    /// A `dgm:relIds` shape reads its data part's point text under a
    /// `(diagram text)` line, every point type read but the points the diagram
    /// flags as its own editor placeholders, and the slide does not read as one
    /// with no text — a diagram that carries none included.
    #[test]
    fn pptx_reads_diagram_text_from_the_data_model() {
        let bytes = zip_fixture(&[
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
            // An editor's hint (`phldrT`) rides on points that carry text of
            // their own, so only a `phldr` flag hides a point; `node`, `asst`
            // and an absent type are all read. A point's second paragraph holds
            // a slide-number field: its `<a:t>` is not the point's run text.
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
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  (diagram text)\n    North\n    Assistant\n    South\n\n\
             Slide 2:\n  (diagram text)"
        );
    }

    /// Diagrams read in the order the slide's shapes present them, each data
    /// part read once on a slide however often it is referenced, and again on
    /// every other slide that shows it.
    #[test]
    fn pptx_reads_each_diagram_once_per_slide() {
        let bytes = zip_fixture(&[
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
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  (diagram text)\n    North\n    South\n  (diagram text)\n    Alpha\n\n\
             Slide 2:\n  (diagram text)\n    North\n    South"
        );
    }

    /// A diagram whose data cannot be read says so rather than letting the slide
    /// read as one without a diagram: a part the package does not hold, an id
    /// nothing resolves, a part the package cut short, a part that is no data
    /// model at all, and a frame naming no data model. A slide with no diagram
    /// carries no such mark.
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
            // A frame whose `relIds` names no data model at all (a package that
            // breaks the schema, which requires all four attributes) shows a
            // diagram whose text cannot be read, rather than none.
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

    /// A self-closing `<dgm:dataModel/>` is a data model that shows no text, so
    /// its slide reads `(diagram text)`; a data part the package left with zero
    /// bytes declares no data model, so its slide reads
    /// `(diagram text could not be read)`.
    #[test]
    fn pptx_reads_a_self_closing_data_model_and_reports_an_empty_one() {
        let bytes = zip_fixture(&[
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
            ("ppt/diagrams/data1.xml", br"<dgm:dataModel/>"),
            (
                "ppt/slides/slide2.xml",
                br#"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId1"/></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>"#,
            ),
            (
                "ppt/slides/_rels/slide2.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data2.xml"/></Relationships>"#,
            ),
            // Zero bytes: an entry the package left empty, which a diagram
            // reads as data absent rather than as a data model with no text.
            ("ppt/diagrams/data2.xml", b""),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(
            text,
            "Slide 1:\n  (diagram text)\n\nSlide 2:\n  (diagram text could not be read)"
        );
    }

    /// The diagram text the editing side is handed, keyed by slide part: one
    /// entry per diagram that shows text, in the reading's own order and read by
    /// the reading's own rule, and nothing at all for a slide whose diagram was
    /// lost or shows no text — a refusal must never name text no reading showed.
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

    /// A table cell's `<a:t>` is body text: the slide walk reads it like any
    /// other run, not as a diagram's.
    #[test]
    fn pptx_reads_table_cells_as_body_text() {
        let bytes = zip_fixture(&[
            (
                "ppt/presentation.xml",
                br#"<p:presentation><p:sldIdLst><p:sldId r:id="rId1"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                br#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                br"<p:sld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody><a:p><a:r><a:t>Cell one</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Cell two</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:sld>",
            ),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");
        let DocOutcome::Text { text, .. } = convert_pptx(&bytes, dir.path()) else {
            panic!("expected Text outcome for a readable presentation");
        };
        assert_eq!(text, "Slide 1:\n  Cell one\n  Cell two");
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
