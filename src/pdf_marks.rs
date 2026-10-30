//! PDF marks: the document's own description, its annotations and its form
//! fields, as document-text blocks.
//!
//! This is the metadata/annotation/form arm of [`crate::document`]'s PDF
//! reading. Where [`crate::document`] produces a page's text and rasters, this
//! module reads what the document says about itself (the Info dictionary), the
//! marks beside the text — comments, highlights, stamps, free text and
//! signatures in the page annotations — and the AcroForm fields a user fills in,
//! and renders them as the sections a model reads. It shares
//! [`crate::document`]'s invariants.
//!
//! # Invariants
//!
//! - **Read-only.** Nothing here writes, and nothing here holds state: the walk
//!   runs off the already-parsed [`Pdf`] the caller opened.
//! - **No panics of its own**, for any file: the whole walk runs inside
//!   [`crate::shutdown::contain_panics`], so a panic in a malformed document
//!   costs the marks and nothing else ([`Marks::default`]).
//! - **One page never costs another.** An annotation or field that cannot be
//!   read is skipped, never an error: the marks around it keep theirs.
//! - **The walking is bounded.** The field tree has a depth cap and a
//!   visited-object set, both walks are capped by [`MAX_MARKS`]/[`MAX_FIELDS`],
//!   and a field's printed option list by [`MAX_FIELD_OPTIONS`]; a cap that
//!   truncates is named in [`Marks::notes`].
//! - **A field is listed with the state it is in.** Every field a fill can reach
//!   is listed, even with no value (the shared empty marker); one it cannot
//!   reach is listed as not fillable. No value is ever invented, and a choice
//!   whose option the file does not name shows only the state it is in.
//! - **One object is one entry.** Identity is the dictionary's own bytes, so a
//!   mark listed on two pages is one mark while two dictionaries written
//!   directly into one array stay two, and a form is walked with the same
//!   `/T`-based field test the filler uses, so the two agree on its fields.

use crate::reader_output::{Unshown, UnshownKind};
use crate::util::html::decode_html_entities;
use crate::util::one_line;
use hayro::hayro_syntax::Pdf;
use hayro::hayro_syntax::object::dict::Dict;
use hayro::hayro_syntax::object::{Array, DateTime, Name, Object, String as PdfString};
use std::collections::{HashMap, HashSet};

// ── Bounds ──────────────────────────────────────────────────────

/// Maximum annotations reported for one document, so a file whose `/Annots`
/// arrays are effectively unbounded cannot make the walk (or the text it
/// produces) unbounded with it.
const MAX_MARKS: usize = 1000;

/// Maximum form fields reported for one document, for the same reason.
const MAX_FIELDS: usize = 1000;

/// Maximum options printed for one form field: a list box whose `/Opt` holds
/// thousands of entries would otherwise make every reading of the file huge.
const MAX_FIELD_OPTIONS: usize = 100;

/// Maximum depth of the field tree, so a file whose `/Kids` point in a cycle
/// that does not repeat an object still terminates.
const MAX_FIELD_DEPTH: usize = 32;

/// What tells one dictionary from another within one document: the bytes it was
/// parsed from. The engine resolves a reference to the very bytes it points at
/// and caches a decoded object stream, so one object resolves to one slice —
/// while two dictionaries written directly into the same array are two slices.
/// [`Dict::obj_id`] cannot serve as an identity for that: it is the *enclosing*
/// indirect object's id, so every direct dictionary in one array would collapse
/// onto the same one.
type DictId = (usize, usize);

/// A dictionary's identity ([`DictId`]).
fn dict_id(dict: &Dict<'_>) -> DictId {
    let data = dict.data();
    (data.as_ptr().addr(), data.len())
}

// ── Delivery notes ──────────────────────────────────────────────

/// Body of the note shown when a document carries a form but no field the
/// caller can fill: every field it has is a pushbutton, a signature, or
/// unreadable. The form is not empty — it simply offers no fillable field.
const NO_FILLABLE_FIELDS_NOTE: &str = "the document has a form, but no field that can be filled";

/// Body of the note shown when a document's `/AcroForm` exists but its
/// `/Fields` could not be read at all.
const FIELDS_UNREADABLE_NOTE: &str = "the document has a form, but its fields could not be read";

// ── Annotation flags and field flags ────────────────────────────

/// `/F` bit 2: the annotation is hidden from the reader.
const ANNOT_FLAG_HIDDEN: i32 = 2;
/// `/F` bit 5: the annotation is not shown or printed.
const ANNOT_FLAG_NO_VIEW: i32 = 32;
/// `/Ff` bit 16: a `Btn` field is a radio group.
const FIELD_FLAG_RADIO: i32 = 1 << 15;
/// `/Ff` bit 17: a `Btn` field is a pushbutton, which carries no value.
const FIELD_FLAG_PUSH_BUTTON: i32 = 1 << 16;
/// `/Ff` bit 18: a `Ch` field is a combo box (a dropdown), whose value a filler
/// may write even when the field offers no `/Opt`; an edit-less list box cannot.
const FIELD_FLAG_COMBO: i32 = 1 << 17;

// ── The public shape ────────────────────────────────────────────

/// What a PDF says beside its page text: its own description, its annotations
/// and its form fields.
#[derive(Debug, Default)]
pub(crate) struct Marks {
    /// The ready `Description:` section, already labeled and laid out; `None`
    /// when the document's Info dictionary carries nothing worth showing.
    pub(crate) description: Option<String>,
    /// The ready `Annotations:` section, already labeled and laid out; `None`
    /// when the document carries no mark.
    pub(crate) annotations: Option<String>,
    /// The ready `Form fields:` section; `None` when the document has no field a
    /// caller can fill.
    pub(crate) fields: Option<String>,
    /// Non-fatal notes, in `crate::document`'s user-facing note shape.
    pub(crate) notes: Vec<String>,
    /// The content the marks themselves carry and the reading does not show: the
    /// files a document attaches and the media its annotations hold, counted for
    /// the document's report (see [`crate::reader_output::Unshown`]) rather than
    /// passing unseen.
    pub(crate) unshown: Unshown,
}

/// Read `pdf`'s own description, its annotations and its form fields.
///
/// The walk is contained exactly like [`crate::document`]'s PDF passes: a panic
/// in a malformed object yields [`Marks::default`] rather than reaching the
/// caller, so one bad document can never cost the conversion its page text too.
#[must_use]
pub(crate) fn read(pdf: &Pdf) -> Marks {
    crate::shutdown::contain_panics(|| read_inner(pdf)).unwrap_or_default()
}

fn read_inner(pdf: &Pdf) -> Marks {
    let mut marks = Marks {
        description: read_description(pdf),
        ..Marks::default()
    };
    let walk = read_annotations(pdf);
    if !walk.lines.is_empty() {
        marks.annotations = Some(format!("Annotations:\n{}", walk.lines.join("\n")));
    }
    marks.unshown = walk.unshown;
    if walk.truncated {
        marks
            .notes
            .push(format!("only the first {MAX_MARKS} annotations were read"));
    }

    let catalog = pdf.xref().get::<Dict<'_>>(pdf.xref().root_id());
    let acroform = catalog.and_then(|catalog| catalog.get::<Dict<'_>>(b"AcroForm"));
    let mut fields_unreadable = false;
    if let Some(acroform) = &acroform {
        match read_fields(acroform, &walk.widget_pages) {
            Some(fields) => {
                if !fields.lines.is_empty() {
                    marks.fields = Some(format!(
                        "{}:\n{}",
                        crate::docgen::pdf_form_words().fields,
                        fields.lines.join("\n")
                    ));
                }
                if fields.truncated {
                    marks
                        .notes
                        .push(format!("only the first {MAX_FIELDS} form fields were read"));
                }
                if fields.options_truncated {
                    marks.notes.push(format!(
                        "only the first {MAX_FIELD_OPTIONS} options of a form field were listed"
                    ));
                }
            }
            None => fields_unreadable = true,
        }
    }
    if fields_unreadable {
        marks.notes.push(FIELDS_UNREADABLE_NOTE.to_string());
    } else if acroform.is_some() && marks.fields.is_none() {
        marks.notes.push(NO_FILLABLE_FIELDS_NOTE.to_string());
    }
    marks
}

// ── The document's description ─────────────────────────────────

/// The `Description:` block from the document's Info dictionary — only the
/// entries it actually carries, in the order the shared rules state, each on one
/// line. A date is rendered in calendar form with the offset the engine kept; a
/// date that stated no zone (or a zero one) reads as UTC, which is the engine's
/// own spelling of the two.
fn read_description(pdf: &Pdf) -> Option<String> {
    let metadata = pdf.metadata();
    let mut entries: Vec<(&str, String)> = Vec::new();
    for field in crate::docgen::pdf_metadata_fields() {
        let value = match field.key.as_str() {
            "created" => metadata.creation_date.map(format_date),
            "modified" => metadata.modification_date.map(format_date),
            "title" => metadata.title.as_deref().and_then(description_text),
            "author" => metadata.author.as_deref().and_then(description_text),
            "subject" => metadata.subject.as_deref().and_then(description_text),
            "keywords" => metadata.keywords.as_deref().and_then(description_text),
            "creator" => metadata.creator.as_deref().and_then(description_text),
            "producer" => metadata.producer.as_deref().and_then(description_text),
            _ => None,
        };
        if let Some(value) = value {
            entries.push((field.key.as_str(), value));
        }
    }
    (!entries.is_empty()).then(|| {
        let body = entries
            .into_iter()
            .map(|(label, value)| format!("{label}: {value}"))
            .collect::<Vec<_>>()
            .join("\n");
        crate::reader_output::labeled_block("Description", &body)
    })
}

/// One text entry of the description, decoded and collapsed to one line. `None`
/// for an entry that is absent or blank: an empty entry says nothing, and the
/// file carrying it is not the same as the file describing itself with nothing.
fn description_text(bytes: &[u8]) -> Option<String> {
    let text = one_line(decode_text_string(bytes).trim().to_string());
    (!text.is_empty()).then_some(text)
}

/// A PDF date as `YYYY-MM-DDTHH:MM:SS` with `Z` for a zero offset or
/// `±HH:MM` for any other. The engine keeps the offset sign only in the hour and
/// its minutes unsigned, so a file spelling `D:…-00'30'` parses as hour 0 minute
/// 30 and prints here as `+00:30`; the trailer's `/Info` is not reachable as raw
/// bytes, so the sign cannot be recovered.
fn format_date(date: DateTime) -> String {
    let calendar = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        date.year, date.month, date.day, date.hour, date.minute, date.second
    );
    if date.utc_offset_hour == 0 && date.utc_offset_minute == 0 {
        format!("{calendar}Z")
    } else {
        format!(
            "{calendar}{:+03}:{:02}",
            date.utc_offset_hour, date.utc_offset_minute
        )
    }
}

// ── Annotations ─────────────────────────────────────────────────

/// What one walk over every page's `/Annots` yields.
#[derive(Default)]
struct AnnotationWalk {
    /// One line per mark, in page order and then in each page's `/Annots` order.
    lines: Vec<String>,
    /// The page each `/Widget` annotation sits on, so a field can be named with
    /// the page its widget is bound to.
    widget_pages: HashMap<DictId, usize>,
    /// Whether [`MAX_MARKS`] cut the walk short.
    truncated: bool,
    /// The annotations that are no mark but still hold content — an attached
    /// file, a sound, a movie — counted for the document's report.
    unshown: Unshown,
}

/// Walk every page's `/Annots` once: the mark lines, the pages its `/Widget`
/// annotations sit on, and whether [`MAX_MARKS`] cut the walk short.
fn read_annotations(pdf: &Pdf) -> AnnotationWalk {
    let mut walk = AnnotationWalk::default();
    let mut reported: HashSet<DictId> = HashSet::new();
    'pages: for (index, page) in pdf.pages().iter().enumerate() {
        let Some(annots) = page.raw().get::<Array<'_>>(b"Annots") else {
            continue;
        };
        let mut annots = annots.flex_iter();
        while let Some(object) = annots.next::<Object<'_>>() {
            let Some(dict) = object.into_dict() else {
                continue;
            };
            if is_widget(&dict) {
                walk.widget_pages.entry(dict_id(&dict)).or_insert(index + 1);
            }
            // One object, one finding: an annotation listed on another page (or
            // twice on this one) is one thing the document holds, and identity is
            // the object, not the bytes of its text — identical marks on two pages
            // stay two marks, and one attachment listed twice stays one file.
            if !reported.insert(dict_id(&dict)) {
                continue;
            }
            let Some(mark) = mark_of(&dict, index + 1) else {
                // An annotation that is no mark can still carry content a reading
                // never shows — an attached file, or the media a sound, a movie
                // or a screen recording holds — and its subtype is in hand here,
                // so the document's report names it instead of the annotation
                // passing unseen.
                if !hidden(&dict)
                    && let Some(kind) = annotation_content(&dict)
                {
                    walk.unshown.add(kind, 1);
                }
                continue;
            };
            if walk.lines.len() == MAX_MARKS {
                walk.truncated = true;
                break 'pages;
            }
            walk.lines.push(mark);
        }
    }
    walk
}

/// The one line a mark contributes — `kind (author) on page n`, with the mark's
/// own text as indented lines when it carries any — or `None` when the
/// annotation is not a mark a reader would show.
fn mark_of(dict: &Dict<'_>, page_number: usize) -> Option<String> {
    let kind = mark_kind(dict)?;
    // The document itself hides this mark from the reader; a hidden note is not
    // a note the model should read.
    if hidden(dict) {
        return None;
    }
    let mut line = match mark_author(dict) {
        Some(author) => format!("{kind} ({author}) on page {page_number}"),
        None => format!("{kind} on page {page_number}"),
    };
    if let Some(text) = mark_text(dict) {
        line.push(':');
        for text_line in text.lines() {
            line.push_str("\n  ");
            line.push_str(text_line);
        }
    }
    Some(line)
}

/// The lowercase kind word for an annotation's `/Subtype`, or `None` when the
/// subtype is not one a model reads as a mark (a link, a popup, a non-signature
/// form widget, an unknown subtype).
fn mark_kind(dict: &Dict<'_>) -> Option<&'static str> {
    let subtype = dict.get::<Name<'_>>(b"Subtype")?;
    match subtype.as_str() {
        "Text" => Some("note"),
        "FreeText" => Some("free text"),
        "Highlight" => Some("highlight"),
        "Underline" => Some("underline"),
        "Squiggly" => Some("squiggly underline"),
        "StrikeOut" => Some("strikeout"),
        "Ink" => Some("ink"),
        "Square" => Some("square"),
        "Circle" => Some("circle"),
        "Polygon" => Some("polygon"),
        "PolyLine" => Some("polyline"),
        "Line" => Some("line"),
        "Caret" => Some("caret"),
        "Stamp" => Some("stamp"),
        "Widget" if is_signature(dict) => Some("signature"),
        _ => None,
    }
}

/// The document content an annotation carries that is not a mark in the text, or
/// `None` when the annotation holds none: a `/FileAttachment` holds a file, and a
/// `/Sound`, `/Movie`, `/Screen`, `/RichMedia` or `/3D` annotation holds media the
/// reading never shows. A screen annotation counts only when it really plays
/// media: one whose actions only navigate holds nothing to show. The subtypes that
/// mark nothing and hold nothing — a `/Link`, a `/Popup`, a `/Redact`, a
/// `/PrinterMark` — are none of them content a model is missing. The entry that
/// holds the content is asked for as well, so an annotation that names a media
/// subtype without carrying any is not counted: what is reported is content the
/// file really holds.
fn annotation_content(dict: &Dict<'_>) -> Option<UnshownKind> {
    let (content, kind) = match dict.get::<Name<'_>>(b"Subtype")?.as_str() {
        "FileAttachment" => (b"FS".as_slice(), UnshownKind::Attachment),
        "Sound" => (b"Sound".as_slice(), UnshownKind::MediaAnnotation),
        "Movie" => (b"Movie".as_slice(), UnshownKind::MediaAnnotation),
        "RichMedia" => (b"RichMediaContent".as_slice(), UnshownKind::MediaAnnotation),
        "3D" => (b"3DD".as_slice(), UnshownKind::MediaAnnotation),
        "Screen" => return plays_media(dict).then_some(UnshownKind::MediaAnnotation),
        _ => return None,
    };
    dict.contains_key(content).then_some(kind)
}

/// Whether a screen annotation plays media rather than doing something else with
/// the annotation: it holds the movie itself (`/Movie`), or one of its actions —
/// `/A`, or any action of `/AA` — is a media action. A screen annotation whose
/// actions only navigate (a `/GoTo`, a `/URI`) holds no media, so it is not named
/// as content the file does not hold.
fn plays_media(dict: &Dict<'_>) -> bool {
    if dict.contains_key(b"Movie") {
        return true;
    }
    if dict
        .get::<Object<'_>>(b"A")
        .is_some_and(|action| plays_media_action(&action))
    {
        return true;
    }
    dict.get::<Dict<'_>>(b"AA").is_some_and(|actions| {
        actions.keys().any(|trigger| {
            actions
                .get::<Object<'_>>(trigger)
                .is_some_and(|action| plays_media_action(&action))
        })
    })
}

/// Whether one action — or an array of them — plays media: a `/Rendition` or a
/// `/Movie` action, or one carrying the clip (`/R`) or the movie (`/Movie`)
/// itself. `/Next` chains another action to this one, so a chain is followed.
fn plays_media_action(object: &Object<'_>) -> bool {
    match object {
        Object::Array(array) => {
            let mut actions = array.flex_iter();
            while let Some(action) = actions.next::<Object<'_>>() {
                if plays_media_action(&action) {
                    return true;
                }
            }
            false
        }
        Object::Dict(action) => {
            let plays = action
                .get::<Name<'_>>(b"S")
                .is_some_and(|kind| matches!(kind.as_str(), "Rendition" | "Movie"));
            plays
                || action.contains_key(b"R")
                || action.contains_key(b"Movie")
                || action
                    .get::<Object<'_>>(b"Next")
                    .is_some_and(|next| plays_media_action(&next))
        }
        _ => false,
    }
}

/// Whether `/F` carries the Hidden or NoView flag.
fn hidden(dict: &Dict<'_>) -> bool {
    dict.get::<i32>(b"F")
        .is_some_and(|flags| flags & (ANNOT_FLAG_HIDDEN | ANNOT_FLAG_NO_VIEW) != 0)
}

/// Whether an annotation is the on-page half of a signature field: its `/FT` —
/// its own, or the one it inherits from the field it belongs to — is `/Sig`.
fn is_signature(dict: &Dict<'_>) -> bool {
    inherited(dict, b"FT")
        .and_then(Object::into_name)
        .is_some_and(|name| name.as_str() == "Sig")
}

/// A mark's author: its `/T`, on the one line ([`one_line`]) the kind word and
/// the page share it with. Never read for a signature, where `/T` is the field's
/// name rather than an author.
fn mark_author(dict: &Dict<'_>) -> Option<String> {
    if is_signature(dict) {
        return None;
    }
    let author = decode_text_string(dict.get::<PdfString>(b"T")?.as_bytes());
    let author = author.trim();
    (!author.is_empty()).then(|| one_line(author.to_string()))
}

/// A mark's text: its `/Contents`, or, when that is missing or blank, its `/RC`
/// with the markup stripped — the text under a highlight is never reconstructed.
fn mark_text(dict: &Dict<'_>) -> Option<String> {
    let contents = dict
        .get::<PdfString>(b"Contents")
        .map(|contents| decode_text_string(contents.as_bytes()));
    if let Some(contents) = contents {
        let contents = contents.trim();
        if !contents.is_empty() {
            return Some(contents.to_string());
        }
    }
    let rich = dict
        .get::<PdfString>(b"RC")
        .map(|rich| decode_text_string(rich.as_bytes()))?;
    let text = strip_markup(&rich);
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// Drop every `<…>` tag from `markup` and decode the entities that are left.
fn strip_markup(markup: &str) -> String {
    let mut text = String::with_capacity(markup.len());
    let mut in_tag = false;
    for character in markup.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(character),
            _ => {}
        }
    }
    decode_html_entities(&text)
}

// ── Form fields ─────────────────────────────────────────────────

/// The form fields of one document: the lines it lists and whether either bound
/// cut a list short.
struct FieldLines {
    /// One rendered line per field, in `/Fields` order.
    lines: Vec<String>,
    /// Whether [`MAX_FIELDS`] cut the walk short.
    truncated: bool,
    /// Whether [`MAX_FIELD_OPTIONS`] cut a field's printed options short.
    options_truncated: bool,
}

/// The form-field lines of `acroform`, in `/Fields` order. `None` when the
/// form's `/Fields` could not be read at all.
fn read_fields(acroform: &Dict<'_>, widget_pages: &HashMap<DictId, usize>) -> Option<FieldLines> {
    let fields = acroform.get::<Array<'_>>(b"Fields")?;
    let mut walk = FieldWalk {
        widget_pages,
        lines: Vec::new(),
        visited: HashSet::new(),
        truncated: false,
        options_truncated: false,
    };
    let mut fields = fields.flex_iter();
    while let Some(object) = fields.next::<Object<'_>>() {
        let Some(field) = object.into_dict() else {
            continue;
        };
        walk.visit(&field, "", 0);
        if walk.truncated {
            break;
        }
    }
    Some(FieldLines {
        lines: walk.lines,
        truncated: walk.truncated,
        options_truncated: walk.options_truncated,
    })
}

/// One pass over the field tree: the fields it reaches, and the bounds that keep
/// it finite.
struct FieldWalk<'a> {
    widget_pages: &'a HashMap<DictId, usize>,
    lines: Vec<String>,
    /// The field objects already visited, so `/Kids` that point back at an
    /// ancestor terminate instead of looping.
    visited: HashSet<DictId>,
    truncated: bool,
    /// Whether [`MAX_FIELD_OPTIONS`] cut any field's option list short.
    options_truncated: bool,
}

impl FieldWalk<'_> {
    /// Visit `field` and, through it, the named kids below it. `prefix` is the
    /// field's fully qualified parent name, `depth` the tree depth so far.
    fn visit(&mut self, field: &Dict<'_>, prefix: &str, depth: usize) {
        if depth > MAX_FIELD_DEPTH {
            return;
        }
        if !self.visited.insert(dict_id(field)) {
            return;
        }
        let name = field_name(field, prefix);
        // A dict whose kids are fields of their own is a container, not a field:
        // the filler lists it as non-terminal and never fills it, so the reading
        // does not list it either — its kids below are the fields.
        if !is_non_terminal(field)
            && let Some(kind) = field_kind(field)
        {
            if self.lines.len() == MAX_FIELDS {
                self.truncated = true;
                return;
            }
            let page = field_page(field, self.widget_pages);
            let options = field_options(field, kind);
            if options.len() > MAX_FIELD_OPTIONS {
                self.options_truncated = true;
            }
            let value = field_value(field, kind, &options);
            let fillable = !name.is_empty() && field_is_fillable(field, kind, &options);
            self.lines.push(render_field(
                &name,
                kind,
                &options,
                page,
                value.as_deref(),
                fillable,
            ));
        }
        let Some(kids) = field.get::<Array<'_>>(b"Kids") else {
            return;
        };
        let mut kids = kids.flex_iter();
        while let Some(object) = kids.next::<Object<'_>>() {
            let Some(kid) = object.into_dict() else {
                continue;
            };
            // A kid that names itself with `/T` is a field of its own — the same
            // test the filler uses to tell a sub-field from a field's on-page
            // half, so the two sides agree on which fields a form has, not just on
            // their names. Anything else is this field's widget: its value is the
            // parent's, reported above.
            if !kid.contains_key(b"T") {
                continue;
            }
            self.visit(&kid, &name, depth + 1);
            if self.truncated {
                return;
            }
        }
    }
}

/// Whether a field is a container rather than a field: a dict whose `/Kids`
/// contain a dict that names itself with `/T`, the test pdf-lib's
/// `isNonTerminalAcroField` uses. Such a dict is not listed — its named kids
/// are the fields — but the walk still descends into them.
fn is_non_terminal(field: &Dict<'_>) -> bool {
    let Some(kids) = field.get::<Array<'_>>(b"Kids") else {
        return false;
    };
    let mut kids = kids.flex_iter();
    while let Some(object) = kids.next::<Object<'_>>() {
        if object.into_dict().is_some_and(|kid| kid.contains_key(b"T")) {
            return true;
        }
    }
    false
}

/// A field's fully qualified name: its own `/T` under the `prefix` its ancestors
/// give it, joined with `.` — the spelling `pdf_form_fill` fills by.
fn field_name(field: &Dict<'_>, prefix: &str) -> String {
    let own = field
        .get::<PdfString>(b"T")
        .map(|name| one_line(decode_text_string(name.as_bytes())))
        .filter(|own| !own.is_empty());
    match own {
        None => prefix.to_string(),
        Some(own) if prefix.is_empty() => own,
        Some(own) => format!("{prefix}.{own}"),
    }
}

/// The kind of a fillable form field, in the vocabulary
/// [`crate::docgen::pdf_field_kinds`] states once for the reading and the
/// filler alike.
#[derive(Clone, Copy)]
enum FieldKind {
    Text,
    Checkbox,
    Radio,
    Choice,
}

impl FieldKind {
    /// The word the field line prints for this kind.
    fn word(self) -> &'static str {
        let kinds = crate::docgen::pdf_field_kinds();
        match self {
            Self::Text => kinds.text.as_str(),
            Self::Checkbox => kinds.checkbox.as_str(),
            Self::Radio => kinds.radio.as_str(),
            Self::Choice => kinds.choice.as_str(),
        }
    }
}

/// A field's kind from its (inherited) `/FT` and `/Ff`; `None` for a field no
/// caller can fill: a pushbutton, a signature, or a field whose `/FT` names no
/// fillable kind.
fn field_kind(field: &Dict<'_>) -> Option<FieldKind> {
    match inherited(field, b"FT")?.into_name()?.as_str() {
        "Tx" => Some(FieldKind::Text),
        "Ch" => Some(FieldKind::Choice),
        "Btn" => {
            let flags = field_flags(field);
            if flags & FIELD_FLAG_PUSH_BUTTON != 0 {
                None
            } else if flags & FIELD_FLAG_RADIO != 0 {
                Some(FieldKind::Radio)
            } else {
                Some(FieldKind::Checkbox)
            }
        }
        _ => None,
    }
}

/// A field's (inherited) `/Ff`.
fn field_flags(field: &Dict<'_>) -> i32 {
    inherited(field, b"Ff")
        .and_then(Object::into_i32)
        .unwrap_or(0)
}

/// The value a field carries, by its kind: a text field's string, a choice's
/// selected display names, a button's checked state. `None` for a text or choice
/// field that carries none — the field is still listed, with the shared empty
/// value marker.
fn field_value(field: &Dict<'_>, kind: FieldKind, options: &[String]) -> Option<String> {
    match kind {
        FieldKind::Text => text_value(field),
        FieldKind::Choice => choice_value(field),
        FieldKind::Checkbox => Some(checkbox_value(field)),
        FieldKind::Radio => Some(radio_value(field, options)),
    }
}

/// The options a field offers, as the model should read them: a radio's fill
/// values or a choice's display names. Text and checkbox fields offer none.
fn field_options(field: &Dict<'_>, kind: FieldKind) -> Vec<String> {
    match kind {
        FieldKind::Radio => radio_options(field),
        FieldKind::Choice => choice_options(field),
        FieldKind::Text | FieldKind::Checkbox => Vec::new(),
    }
}

/// Whether a filler can put a value into the field: a radio group or a list box
/// with no option to select is not one, while a combo box (a dropdown) accepts a
/// value of the caller's own even without `/Opt`. The filler's own reach is the
/// rule, so the reading never offers a field a fill would refuse.
fn field_is_fillable(field: &Dict<'_>, kind: FieldKind, options: &[String]) -> bool {
    match kind {
        FieldKind::Text | FieldKind::Checkbox => true,
        FieldKind::Radio => !options.is_empty(),
        FieldKind::Choice => !options.is_empty() || field_flags(field) & FIELD_FLAG_COMBO != 0,
    }
}

/// A text field's value: its `/V` string, or nothing when it is absent or blank.
fn text_value(field: &Dict<'_>) -> Option<String> {
    let value = inherited(field, b"V")?.into_string()?;
    let text = decode_text_string(value.as_bytes());
    let text = text.trim();
    (!text.is_empty()).then(|| one_line(text.to_string()))
}

/// A checkbox's value: whether it is checked, and — when the field offers more
/// than one on-state — the state it is in, in parentheses. A bare-number state is
/// left out: it is a service code rather than a name a reader would recognise,
/// and a checkbox fill takes only true or false, so the state names no value the
/// caller could pass. A radio's parenthetical, by contrast, is the fill value a
/// caller does pass (see [`radio_value`]).
fn checkbox_value(field: &Dict<'_>) -> String {
    let selected = selected_states(field);
    let checked = selected.iter().any(|state| state != "Off");
    let mut value = checked_word(checked);
    if !checked {
        return value;
    }
    let states = on_states(field);
    if states.len() > 1
        && let Some(state) = selected.iter().find(|state| *state != "Off")
        && !is_bare_number(state)
    {
        value = format!("{value} ({state})");
    }
    value
}

/// A radio group's value: whether it is checked, and — when the file offers more
/// than one choice — the fill value the checked state names, in parentheses.
fn radio_value(field: &Dict<'_>, options: &[String]) -> String {
    let selected = selected_states(field);
    let checked = selected.iter().any(|state| state != "Off");
    let mut value = checked_word(checked);
    if !checked {
        return value;
    }
    let states = on_states(field);
    if (options.len() > 1 || states.len() > 1)
        && let Some(state) = selected.iter().find(|state| *state != "Off")
        && let Some(name) = radio_fill_value(state, options, &states)
    {
        value = format!("{value} ({name})");
    }
    value
}

/// The word a button's checked state reads as.
fn checked_word(checked: bool) -> String {
    if checked { "checked" } else { "not checked" }.to_string()
}

/// The fill value a selected radio state names: the `/Opt` entry the state
/// spells, or the option at the state's position. `None` when the file names no
/// such option.
fn radio_fill_value(state: &str, options: &[String], states: &[String]) -> Option<String> {
    if let Some(option) = options.iter().find(|option| *option == state) {
        return Some(option.clone());
    }
    let position = states.iter().position(|known| known == state)?;
    options.get(position).cloned()
}

/// A choice field's value: each selected `/V` resolved through `/Opt`, joined
/// when the field allows several. `None` when nothing is selected.
fn choice_value(field: &Dict<'_>) -> Option<String> {
    let mut selected = selected_strings(field);
    selected.retain(|value| !value.trim().is_empty());
    if selected.is_empty() {
        return None;
    }
    let options = option_entries(field);
    let names = selected
        .iter()
        .map(|value| {
            options
                .iter()
                .find(|option| option.export == *value || option.display == *value)
                .map_or_else(|| value.clone(), |option| option.display.clone())
        })
        .collect::<Vec<_>>();
    Some(one_line(names.join(", ")))
}

/// A field's selected button states: its `/V` (a name, or the names of a
/// multi-select array), falling back to its first widget's `/AS`.
fn selected_states(field: &Dict<'_>) -> Vec<String> {
    match inherited(field, b"V") {
        Some(Object::Name(name)) => vec![name.as_str().to_string()],
        Some(Object::Array(array)) => name_strings(&array),
        _ => first_widget_state(field).into_iter().collect(),
    }
}

/// A choice field's selected `/V` strings (one, or the several of a multi-select
/// array).
fn selected_strings(field: &Dict<'_>) -> Vec<String> {
    match inherited(field, b"V") {
        Some(Object::String(value)) => vec![decode_text_string(value.as_bytes())],
        Some(Object::Array(array)) => string_items(&array),
        _ => Vec::new(),
    }
}

/// The name states an array of names holds.
fn name_strings(array: &Array<'_>) -> Vec<String> {
    let mut names = Vec::new();
    let mut items = array.flex_iter();
    while let Some(object) = items.next::<Object<'_>>() {
        if let Object::Name(name) = object {
            names.push(name.as_str().to_string());
        }
    }
    names
}

/// The text of an array's string items, in order.
fn string_items(array: &Array<'_>) -> Vec<String> {
    let mut items = Vec::new();
    let mut strings = array.flex_iter();
    while let Some(object) = strings.next::<Object<'_>>() {
        if let Object::String(text) = object {
            items.push(decode_text_string(text.as_bytes()));
        }
    }
    items
}

/// A field's first widget's `/AS` — the state a button is in when its `/V` says
/// nothing.
fn first_widget_state(field: &Dict<'_>) -> Option<String> {
    field_widget(field)?
        .get::<Name<'_>>(b"AS")
        .map(|state| state.as_str().to_string())
}

/// The widget that carries a field's own state and page: the field itself when
/// it is a widget, otherwise the first widget among its kids. `None` for a field
/// with no widget anywhere — a hidden or calculated one.
fn field_widget<'a>(field: &Dict<'a>) -> Option<Dict<'a>> {
    if is_widget(field) {
        Some(field.clone())
    } else {
        widget_kids(field).into_iter().next()
    }
}

/// The distinct on-state names a button offers, `Off` excluded — what tells a
/// lone checkbox from a group. A field's widgets are its on-page halves, so when
/// it has any the states are collected from them alone (mirroring the filler's
/// own walk); a field with no widget kids is its own widget.
fn on_states(field: &Dict<'_>) -> Vec<String> {
    let mut states = Vec::new();
    let widgets = widget_kids(field);
    if widgets.is_empty() {
        collect_on_states(field, &mut states);
    } else {
        for widget in &widgets {
            collect_on_states(widget, &mut states);
        }
    }
    states
}

fn collect_on_states(dict: &Dict<'_>, states: &mut Vec<String>) {
    let Some(appearance) = dict
        .get::<Dict<'_>>(b"AP")
        .and_then(|appearance| appearance.get::<Dict<'_>>(b"N"))
    else {
        return;
    };
    for name in appearance.keys() {
        let state = name.as_str();
        if state != "Off" && !states.iter().any(|known| known == state) {
            states.push(state.to_string());
        }
    }
}

/// One `/Opt` entry: the export value a `/V` names, and the display name the
/// model should read. A single string is both.
struct OptionEntry {
    export: String,
    display: String,
}

/// A field's `/Opt` entries, in order. A single string is one entry, the shape
/// the library's own `getOptions` reads as one option. `/Opt` is not an
/// inheritable entry, so only the field's own dictionary is read.
fn option_entries(field: &Dict<'_>) -> Vec<OptionEntry> {
    let mut entries = Vec::new();
    let mut options = match field.get::<Object<'_>>(b"Opt") {
        Some(Object::String(text)) => {
            let text = decode_text_string(text.as_bytes());
            entries.push(OptionEntry {
                export: text.clone(),
                display: text,
            });
            return entries;
        }
        Some(Object::Array(array)) => array.flex_iter(),
        _ => return entries,
    };
    while let Some(object) = options.next::<Object<'_>>() {
        match object {
            Object::String(text) => {
                let text = decode_text_string(text.as_bytes());
                entries.push(OptionEntry {
                    export: text.clone(),
                    display: text,
                });
            }
            Object::Array(pair) => {
                let parts = string_items(&pair);
                match parts.len() {
                    0 => {}
                    1 => entries.push(OptionEntry {
                        export: parts[0].clone(),
                        display: parts[0].clone(),
                    }),
                    _ => entries.push(OptionEntry {
                        export: parts[0].clone(),
                        display: parts[1].clone(),
                    }),
                }
            }
            _ => {}
        }
    }
    entries
}

/// A radio group's options: its own `/Opt` — one string, or the string items of
/// an array — or, with no `/Opt`, the on-state names it offers. A non-string
/// `/Opt` names nothing to show, so it contributes none. `/Opt` is not
/// inheritable, so a parent's copy is not the field's.
fn radio_options(field: &Dict<'_>) -> Vec<String> {
    match field.get::<Object<'_>>(b"Opt") {
        Some(Object::String(text)) => vec![one_line(decode_text_string(text.as_bytes()))],
        Some(Object::Array(array)) => string_items(&array)
            .into_iter()
            .map(one_line)
            .collect::<Vec<_>>(),
        Some(_) => Vec::new(),
        None => on_states(field),
    }
}

/// A choice field's options: the display name of each `/Opt` entry.
fn choice_options(field: &Dict<'_>) -> Vec<String> {
    option_entries(field)
        .into_iter()
        .map(|option| one_line(option.display))
        .collect()
}

/// Whether a state name is only digits, and so a service code rather than a name
/// a reader would recognise.
fn is_bare_number(state: &str) -> bool {
    !state.is_empty() && state.bytes().all(|byte| byte.is_ascii_digit())
}

/// The page a field's own widget is on. `None` for a field with no widget
/// anywhere — a hidden or calculated one.
fn field_page(field: &Dict<'_>, widget_pages: &HashMap<DictId, usize>) -> Option<usize> {
    widget_pages.get(&dict_id(&field_widget(field)?)).copied()
}

/// The `/Widget` kids of a field, in order.
fn widget_kids<'a>(field: &Dict<'a>) -> Vec<Dict<'a>> {
    let Some(kids) = field.get::<Array<'a>>(b"Kids") else {
        return Vec::new();
    };
    let mut widgets = Vec::new();
    let mut kids = kids.flex_iter();
    while let Some(object) = kids.next::<Object<'a>>() {
        if let Some(kid) = object.into_dict()
            && is_widget(&kid)
        {
            widgets.push(kid);
        }
    }
    widgets
}

/// Whether a dictionary is a `/Widget` annotation — a form field's on-page half.
fn is_widget(dict: &Dict<'_>) -> bool {
    matches!(dict.get::<Name<'_>>(b"Subtype").as_deref(), Some(b"Widget"))
}

/// `key` on `field`, or on the nearest ancestor that carries it — `/FT`, `/Ff`
/// and `/V` are all inheritable.
fn inherited<'a>(field: &Dict<'a>, key: &[u8]) -> Option<Object<'a>> {
    let mut current = field.clone();
    for _ in 0..MAX_FIELD_DEPTH {
        if let Some(value) = current.get::<Object<'a>>(key) {
            return Some(value);
        }
        current = current.get::<Dict<'a>>(b"Parent")?;
    }
    None
}

/// One field's line: `name (kind[; options: …][; not fillable])[ on page n]:
/// value`. An unnamed field says so and is marked not fillable, as is one the
/// filler cannot reach; a longer option list is cut at [`MAX_FIELD_OPTIONS`]
/// with `…`; a field the file gives no value for says the shared empty marker
/// rather than being left out.
fn render_field(
    name: &str,
    kind: FieldKind,
    options: &[String],
    page: Option<usize>,
    value: Option<&str>,
    fillable: bool,
) -> String {
    let words = crate::docgen::pdf_form_words();
    let mut label = kind.word().to_string();
    if !options.is_empty() {
        label.push_str("; options: ");
        let shown = &options[..options.len().min(MAX_FIELD_OPTIONS)];
        label.push_str(&shown.join(", "));
        if options.len() > MAX_FIELD_OPTIONS {
            label.push_str(", …");
        }
    }
    if !fillable {
        label.push_str("; ");
        label.push_str(&words.not_fillable);
    }
    let name = if name.is_empty() {
        words.unnamed.as_str()
    } else {
        name
    };
    let value = value.unwrap_or(words.no_value.as_str());
    match page {
        Some(page) => format!("{name} ({label}) on page {page}: {value}"),
        None => format!("{name} ({label}): {value}"),
    }
}

// ── Text strings ────────────────────────────────────────────────

/// A PDF text string decoded to Unicode. A leading `FE FF` BOM marks UTF-16BE,
/// a leading `EF BB BF` BOM marks UTF-8, and anything else is PDFDocEncoding.
/// Decoding is lossy on malformed input (a lone UTF-16 surrogate, invalid
/// UTF-8), because a name half-decoded still reads, while an error would cost
/// the whole mark.
fn decode_text_string(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units = rest
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes(*pair));
        return String::from_utf16_lossy(&units.collect::<Vec<_>>());
    }
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    bytes.iter().copied().map(pdf_doc_encoding).collect()
}

/// The accent characters PDFDocEncoding places at `0x18..=0x1F` (PDF 32000-1,
/// Annex D.2).
const PDF_DOC_ENCODING_ACCENTS: [char; 8] = [
    '\u{02D8}', '\u{02C7}', '\u{02C6}', '\u{02D9}', '\u{02DD}', '\u{02DB}', '\u{02DA}', '\u{02DC}',
];

/// The special range PDFDocEncoding places at `0x80..=0xA0` (PDF 32000-1, Annex
/// D.2): punctuation, ligatures and accented letters, with `0x9F` the range's
/// one undefined slot (the encoding's other, `0x7F`, falls through to the ASCII
/// branch) and `0xA0` the euro sign.
const PDF_DOC_ENCODING_SPECIAL: [char; 33] = [
    '\u{2022}', '\u{2020}', '\u{2021}', '\u{2026}', '\u{2014}', '\u{2013}', '\u{0192}', '\u{2044}',
    '\u{2039}', '\u{203A}', '\u{2212}', '\u{2030}', '\u{201E}', '\u{201C}', '\u{201D}', '\u{2018}',
    '\u{2019}', '\u{201A}', '\u{2122}', '\u{FB01}', '\u{FB02}', '\u{0141}', '\u{0152}', '\u{0160}',
    '\u{0178}', '\u{017D}', '\u{0131}', '\u{0142}', '\u{0153}', '\u{0161}', '\u{017E}', '\u{FFFD}',
    '\u{20AC}',
];

/// One byte of PDFDocEncoding: the two special ranges, and ASCII/Latin-1 for
/// every other byte.
fn pdf_doc_encoding(byte: u8) -> char {
    match byte {
        0x18..=0x1F => PDF_DOC_ENCODING_ACCENTS[usize::from(byte - 0x18)],
        0x80..=0xA0 => PDF_DOC_ENCODING_SPECIAL[usize::from(byte - 0x80)],
        _ => char::from(byte),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::test_fixtures::{PdfFixture, pdf_text_string};

    /// A PDF whose catalog carries `/Pages` and the `catalog_extra` of `build`,
    /// with one page per entry of the returned list. Each page's `/Annots`
    /// references that entry's annotation objects — or holds a raw token the test
    /// wants there, so an array can be tested with a non-object in it — and
    /// `build` pushes the objects it needs first, so one can name another.
    fn pdf_with(build: impl FnOnce(&mut PdfFixture) -> (Vec<Vec<String>>, String)) -> Vec<u8> {
        pdf_with_trailer(|fixture| {
            let (pages, catalog_extra) = build(fixture);
            (pages, catalog_extra, String::new())
        })
    }

    /// [`pdf_with`] with `trailer_extra` written into the trailer — the hook a
    /// fixture has for an `/Info` description.
    fn pdf_with_trailer(
        build: impl FnOnce(&mut PdfFixture) -> (Vec<Vec<String>>, String, String),
    ) -> Vec<u8> {
        let mut fixture = PdfFixture::new();
        let (pages, catalog_extra, trailer_extra) = build(&mut fixture);
        let page_refs: Vec<String> = pages.iter().map(|_| fixture.reserve()).collect();
        let pages_ref = fixture.push(format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            page_refs.join(" "),
            page_refs.len()
        ));
        for (page_ref, annotations) in page_refs.iter().zip(&pages) {
            let annots = if annotations.is_empty() {
                String::new()
            } else {
                format!(" /Annots [{}]", annotations.join(" "))
            };
            fixture.set(
                page_ref,
                format!("<< /Type /Page /Parent {pages_ref} /MediaBox [0 0 612 792]{annots} >>"),
            );
        }
        fixture.build_with_trailer(
            &format!("/Pages {pages_ref}{catalog_extra}"),
            &trailer_extra,
        )
    }

    fn read_pdf(bytes: Vec<u8>) -> Marks {
        let pdf = Pdf::new(bytes).expect("fixture PDF must parse");
        read(&pdf)
    }

    #[test]
    fn every_readable_mark_kind_is_reported_on_its_page() {
        let marks = read_pdf(pdf_with(|fixture| {
            let note = fixture.push("<< /Subtype /Text /Contents (please verify) >>".to_string());
            let highlight = fixture.push(format!(
                "<< /Subtype /Highlight /T {} /Contents {} >>",
                pdf_text_string("Иван Петров"),
                pdf_text_string("проверьте третью фигуру")
            ));
            let stamp = fixture.push("<< /Subtype /Stamp >>".to_string());
            let square = fixture.push("<< /Subtype /Square >>".to_string());
            let free_text =
                fixture.push("<< /Subtype /FreeText /RC (<body>Approved</body>) >>".to_string());
            (
                vec![
                    vec![note],
                    vec![highlight],
                    vec![stamp],
                    vec![square],
                    vec![free_text],
                ],
                String::new(),
            )
        }));
        assert_eq!(
            marks.annotations.as_deref(),
            Some(
                "Annotations:\nnote on page 1:\n  please verify\n\
                 highlight (Иван Петров) on page 2:\n  проверьте третью фигуру\n\
                 stamp on page 3\nsquare on page 4\n\
                 free text on page 5:\n  Approved"
            )
        );
        assert!(marks.fields.is_none());
        assert!(marks.notes.is_empty());
    }

    #[test]
    fn service_only_annotations_are_not_marks() {
        let marks = read_pdf(pdf_with(|fixture| {
            let link = fixture.push("<< /Subtype /Link /Rect [0 0 0 0] >>".to_string());
            let popup = fixture.push("<< /Subtype /Popup /Rect [0 0 0 0] >>".to_string());
            let widget = fixture.push("<< /Subtype /Widget /FT /Tx /T (field) >>".to_string());
            let hidden = fixture.push("<< /Subtype /Text /Contents (hidden) /F 2 >>".to_string());
            let no_view = fixture.push("<< /Subtype /Text /Contents (noview) /F 32 >>".to_string());
            let visible = fixture.push("<< /Subtype /Text /Contents (visible) >>".to_string());
            (
                vec![vec![link, popup, widget, hidden, no_view, visible]],
                String::new(),
            )
        }));
        assert_eq!(
            marks.annotations.as_deref(),
            Some("Annotations:\nnote on page 1:\n  visible")
        );
        assert!(
            marks.notes.is_empty(),
            "a widget without an /AcroForm is no unreadable form: {marks:?}"
        );
    }

    /// The annotations that are no mark but still hold content are named for the
    /// document's report rather than passing unseen: an attached file and the media
    /// a sound or a movie holds. A hidden one is content the reader is not shown,
    /// so it is not counted either — an annotation listed twice is one file, and
    /// one naming a media subtype while carrying no media at all (a `/Screen` with
    /// no action, or one whose action only navigates) is not content the file
    /// holds.
    #[test]
    fn marks_report_attachments_and_media_they_do_not_show() {
        let marks = read_pdf(pdf_with(|fixture| {
            let attachment = fixture.push(
                "<< /Subtype /FileAttachment /FS << /Type /Filespec /F (notes.txt) >> >>"
                    .to_string(),
            );
            let sound = fixture.push("<< /Subtype /Sound /Sound << >> >>".to_string());
            let empty_screen = fixture.push("<< /Subtype /Screen >>".to_string());
            let screen = fixture.push("<< /Subtype /Screen /A << /S /Rendition >> >>".to_string());
            let navigating =
                fixture.push("<< /Subtype /Screen /A << /S /GoTo /D [0 0 0 0] >> >>".to_string());
            let playing =
                fixture.push("<< /Subtype /Screen /AA << /PO << /S /Movie >> >> >>".to_string());
            let note = fixture.push("<< /Subtype /Text /Contents (please verify) >>".to_string());
            let hidden_movie = fixture.push("<< /Subtype /Movie /F 2 /Movie << >> >>".to_string());
            (
                vec![vec![
                    attachment.clone(),
                    attachment,
                    sound,
                    empty_screen,
                    screen,
                    navigating,
                    playing,
                    note,
                    hidden_movie,
                ]],
                String::new(),
            )
        }));
        assert_eq!(
            marks.unshown.lines(),
            [
                "1 attached file(s) not shown",
                "3 media annotation(s) not shown",
            ]
        );
        assert_eq!(
            marks.annotations.as_deref(),
            Some("Annotations:\nnote on page 1:\n  please verify")
        );
        assert!(marks.notes.is_empty());
    }

    #[test]
    fn filled_field_values_are_reported_by_kind_and_page() {
        let marks = read_pdf(pdf_with(|fixture| {
            let appearance =
                fixture.push("<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] >>".to_string());
            let name = fixture.push(format!(
                "<< /Subtype /Widget /FT /Tx /T (Full name) /V {} >>",
                pdf_text_string("Иван Петров")
            ));
            let subscribe = fixture.push(format!(
                "<< /Subtype /Widget /FT /Btn /T (Subscribe) /V /Yes \
                 /AP << /N << /Yes {appearance} /Off {appearance} >> >> /AS /Yes >>"
            ));
            let unsubscribe = fixture.push(format!(
                "<< /Subtype /Widget /FT /Btn /T (Unsubscribe) /V /Off \
                 /AP << /N << /Yes {appearance} /Off {appearance} >> >> /AS /Off >>"
            ));
            let empty = fixture.push("<< /Subtype /Widget /FT /Tx /T (Empty) >>".to_string());
            let delivery = fixture.reserve();
            let first = fixture.push(format!(
                "<< /Subtype /Widget /Parent {delivery} \
                 /AP << /N << /exp {appearance} /Off {appearance} >> >> /AS /Off >>"
            ));
            let second = fixture.push(format!(
                "<< /Subtype /Widget /Parent {delivery} \
                 /AP << /N << /std {appearance} /Off {appearance} >> >> /AS /std >>"
            ));
            fixture.set(
                &delivery,
                format!(
                    "<< /FT /Btn /T (Delivery) /V /std /Ff 32768 \
                     /Opt [[(exp) (Express)] [(std) (Standard)]] /Kids [{first} {second}] >>"
                ),
            );
            let city = fixture.push(
                "<< /Subtype /Widget /FT /Ch /T (City) /V (spb) \
                 /Opt [[(msk) (Moscow)] [(spb) (Saint Petersburg)]] >>"
                    .to_string(),
            );
            let computed = fixture.push("<< /FT /Tx /T (Computed) /V (42) >>".to_string());
            // A value whose `/V` carries a newline, written as the ASCII bytes
            // `two\nlines` a hex string holds unambiguously.
            let note = fixture.push("<< /FT /Tx /T (Note) /V <74776f0a6c696e6573> >>".to_string());
            let acroform = fixture.push(format!(
                "<< /Fields [{name} {subscribe} {unsubscribe} {delivery} {city} {computed} {note} \
                 {empty}] >>"
            ));
            (
                vec![
                    vec![name, subscribe, unsubscribe, empty, first, second],
                    vec![city],
                ],
                format!(" /AcroForm {acroform}"),
            )
        }));
        assert_eq!(
            marks.fields.as_deref(),
            Some(
                "Form fields:\nFull name (text) on page 1: Иван Петров\n\
                 Subscribe (checkbox) on page 1: checked\n\
                 Unsubscribe (checkbox) on page 1: not checked\n\
                 Delivery (radio; not fillable) on page 1: checked\n\
                 City (choice; options: Moscow, Saint Petersburg) on page 2: Saint Petersburg\n\
                 Computed (text): 42\nNote (text): two\\nlines\n\
                 Empty (text) on page 1: (no value)"
            )
        );
        assert!(marks.notes.is_empty());
    }

    /// A checkbox that offers more than one on-state says which one it is in: a
    /// bare `checked` would read the same for either, and could mean the
    /// opposite of the state the file records.
    #[test]
    fn a_multi_state_checkbox_names_the_state_it_is_in() {
        let marks = read_pdf(pdf_with(|fixture| {
            let appearance =
                fixture.push("<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] >>".to_string());
            let answer = fixture.push(format!(
                "<< /Subtype /Widget /FT /Btn /T (Answer) /V /No \
                 /AP << /N << /Yes {appearance} /No {appearance} /Off {appearance} >> >> >>"
            ));
            let acroform = fixture.push(format!("<< /Fields [{answer}] >>"));
            (vec![vec![answer]], format!(" /AcroForm {acroform}"))
        }));
        assert_eq!(
            marks.fields.as_deref(),
            Some("Form fields:\nAnswer (checkbox) on page 1: checked (No)")
        );
    }

    #[test]
    fn a_choice_with_more_options_than_the_cap_lists_the_first_ones() {
        let options = (0..=MAX_FIELD_OPTIONS)
            .map(|index| format!("[(o{index}) (Option {index})]"))
            .collect::<Vec<_>>()
            .join(" ");
        let marks = read_pdf(pdf_with(|fixture| {
            let field = fixture.push(format!(
                "<< /Subtype /Widget /FT /Ch /T (Long) /V () /Opt [{options}] >>"
            ));
            let acroform = fixture.push(format!("<< /Fields [{field}] >>"));
            (vec![vec![field]], format!(" /AcroForm {acroform}"))
        }));
        let listed = (0..MAX_FIELD_OPTIONS)
            .map(|index| format!("Option {index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let expected =
            format!("Form fields:\nLong (choice; options: {listed}, …) on page 1: (no value)");
        assert_eq!(marks.fields.as_deref(), Some(expected.as_str()));
        assert_eq!(
            marks.notes,
            vec![format!(
                "only the first {MAX_FIELD_OPTIONS} options of a form field were listed"
            )]
        );
    }

    /// Every field a form declares is listed with its kind: a field with no
    /// value prints `(no value)`, a checkbox prints the state it is in, a radio
    /// group prints the options its appearance states offer, a choice prints its
    /// options, and a field with no `/T` is shown as unnamed and marked not
    /// fillable rather than left out or read as a bare value.
    #[test]
    fn a_form_lists_every_field_it_holds() {
        let without_values: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let first = fixture.push("<< /Subtype /Widget /FT /Tx /T (One) >>".to_string());
            let second = fixture.push("<< /Subtype /Widget /FT /Tx /T (Two) >>".to_string());
            let acroform = fixture.push(format!("<< /Fields [{first} {second}] >>"));
            (vec![vec![first, second]], format!(" /AcroForm {acroform}"))
        };
        let blank: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let appearance =
                fixture.push("<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] >>".to_string());
            let name = fixture.push("<< /Subtype /Widget /FT /Tx /T (Name) >>".to_string());
            let agree =
                fixture.push("<< /Subtype /Widget /FT /Btn /T (Agree) /V /Off >>".to_string());
            let plan = fixture.reserve();
            let basic = fixture.push(format!(
                "<< /Subtype /Widget /Parent {plan} \
                 /AP << /N << /basic {appearance} /Off {appearance} >> >> /AS /Off >>"
            ));
            let pro = fixture.push(format!(
                "<< /Subtype /Widget /Parent {plan} \
                 /AP << /N << /pro {appearance} /Off {appearance} >> >> /AS /Off >>"
            ));
            fixture.set(
                &plan,
                format!("<< /FT /Btn /Ff 32768 /T (Plan) /V /Off /Kids [{basic} {pro}] >>"),
            );
            let city = fixture.push(
                "<< /Subtype /Widget /FT /Ch /T (City) \
                 /Opt [[(msk) (Moscow)] [(spb) (Saint Petersburg)]] >>"
                    .to_string(),
            );
            let acroform = fixture.push(format!("<< /Fields [{name} {agree} {plan} {city}] >>"));
            (
                vec![vec![name, agree, basic, pro, city]],
                format!(" /AcroForm {acroform}"),
            )
        };
        let unnamed: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let field = fixture.push("<< /Subtype /Widget /FT /Tx /V (secret) >>".to_string());
            let acroform = fixture.push(format!("<< /Fields [{field}] >>"));
            (vec![vec![field]], format!(" /AcroForm {acroform}"))
        };
        for (label, build, expected) in [
            (
                "a form whose fields carry no value",
                without_values,
                "Form fields:\nOne (text) on page 1: (no value)\n\
                 Two (text) on page 1: (no value)",
            ),
            (
                "a blank form",
                blank,
                "Form fields:\nName (text) on page 1: (no value)\n\
                 Agree (checkbox) on page 1: not checked\n\
                 Plan (radio; options: basic, pro) on page 1: not checked\n\
                 City (choice; options: Moscow, Saint Petersburg) on page 1: (no value)",
            ),
            (
                "a field with no /T",
                unnamed,
                "Form fields:\nunnamed field (text; not fillable) on page 1: secret",
            ),
        ] {
            let marks = read_pdf(pdf_with(build));
            assert!(
                marks.annotations.is_none(),
                "{label}: a widget is no mark: {marks:?}"
            );
            assert_eq!(
                marks.fields.as_deref(),
                Some(expected),
                "{label}: the fields it lists"
            );
            assert!(
                marks.notes.is_empty(),
                "{label}: no loss to report: {marks:?}"
            );
        }
    }

    /// A signature keeps its annotation mark and is never listed as a field
    /// value, whether it is one widget, a field whose widget is its kid, or one
    /// beside a pushbutton — and a pushbutton, which fills nothing, is not
    /// listed at all. A form of only those says so instead of looking empty.
    #[test]
    fn a_signature_is_a_mark_and_a_pushbutton_is_no_field() {
        let widget: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let signature = fixture
                .push("<< /Subtype /Widget /FT /Sig /T (Approval) /Rect [0 0 0 0] >>".to_string());
            let acroform = fixture.push(format!("<< /Fields [{signature}] >>"));
            (vec![vec![signature]], format!(" /AcroForm {acroform}"))
        };
        let beside_a_pushbutton: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let button = fixture.push(
                "<< /Subtype /Widget /FT /Btn /Ff 65536 /T (Submit) /Rect [0 0 0 0] >>".to_string(),
            );
            let signature = fixture
                .push("<< /Subtype /Widget /FT /Sig /T (Approval) /Rect [0 0 0 0] >>".to_string());
            let acroform = fixture.push(format!("<< /Fields [{button} {signature}] >>"));
            (
                vec![vec![button, signature]],
                format!(" /AcroForm {acroform}"),
            )
        };
        let split_from_its_widget: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let field = fixture.reserve();
            let widget = fixture.push(format!(
                "<< /Subtype /Widget /Parent {field} /Rect [0 0 0 0] >>"
            ));
            fixture.set(
                &field,
                format!("<< /FT /Sig /T (Approval) /Kids [{widget}] >>"),
            );
            let acroform = fixture.push(format!("<< /Fields [{field}] >>"));
            (vec![vec![widget]], format!(" /AcroForm {acroform}"))
        };
        for (label, build, expected) in [
            (
                "a signature widget",
                widget,
                "Annotations:\nsignature on page 1",
            ),
            (
                "a pushbutton beside a signature",
                beside_a_pushbutton,
                "Annotations:\nsignature on page 1",
            ),
            (
                "a signature field split from its widget",
                split_from_its_widget,
                "Annotations:\nsignature on page 1",
            ),
        ] {
            let marks = read_pdf(pdf_with(build));
            assert_eq!(
                marks.annotations.as_deref(),
                Some(expected),
                "{label}: the signature stays a mark"
            );
            assert!(
                marks.fields.is_none(),
                "{label}: no signature is a field value: {marks:?}"
            );
            assert_eq!(
                marks.notes,
                vec![NO_FILLABLE_FIELDS_NOTE.to_string()],
                "{label}: the form is said to hold nothing fillable"
            );
        }
    }

    /// A walk of the field tree and of the arrays it reads stops rather than
    /// looping or giving up: a field tree that points at an ancestor terminates
    /// (the container is not listed, the kid its own ancestor reference cannot
    /// resolve is read as the terminal field it looks like), and an array entry
    /// that is no name or no dictionary is skipped without ending the walk.
    #[test]
    fn the_field_tree_walk_terminates_and_skips_unknown_entries() {
        let ancestor: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let parent = fixture.reserve();
            let child = fixture.reserve();
            fixture.set(
                &parent,
                format!("<< /FT /Tx /T (A) /V (alpha) /Kids [{child}] >>"),
            );
            fixture.set(
                &child,
                format!("<< /FT /Tx /T (B) /V (beta) /Kids [{parent}] /Parent {parent} >>"),
            );
            let acroform = fixture.push(format!("<< /Fields [{parent}] >>"));
            (vec![vec![]], format!(" /AcroForm {acroform}"))
        };
        let unknown_entries: fn(&mut PdfFixture) -> (Vec<Vec<String>>, String) = |fixture| {
            let named_contents =
                fixture.push("<< /Subtype /Stamp /Contents /NotAString >>".to_string());
            let note = fixture.push("<< /Subtype /Text /Contents (kept) >>".to_string());
            let field = fixture.push("<< /FT /Tx /T (Name) /V (value) >>".to_string());
            let acroform = fixture.push(format!("<< /Fields [42 {field}] >>"));
            (
                vec![vec![
                    named_contents,
                    "7".to_string(),
                    "/Foo".to_string(),
                    note,
                ]],
                format!(" /AcroForm {acroform}"),
            )
        };
        for (label, build, expected) in [
            (
                "a field tree that points at an ancestor",
                ancestor,
                (None, Some("Form fields:\nA.B (text): beta")),
            ),
            (
                "unknown entries in the annotation and field arrays",
                unknown_entries,
                (
                    Some("Annotations:\nstamp on page 1\nnote on page 1:\n  kept"),
                    Some("Form fields:\nName (text): value"),
                ),
            ),
        ] {
            let (expected_annotations, expected_fields) = expected;
            let marks = read_pdf(pdf_with(build));
            assert_eq!(
                marks.annotations.as_deref(),
                expected_annotations,
                "{label}: the marks the walk kept"
            );
            assert_eq!(
                marks.fields.as_deref(),
                expected_fields,
                "{label}: the fields the walk kept"
            );
        }
    }

    /// Dictionaries written straight into an array carry no object id of their
    /// own: each is still its own mark and its own field, and a field with no
    /// `/T` is named as one rather than left as a bare value.
    #[test]
    fn direct_dictionaries_are_read_as_their_own_marks_and_fields() {
        let marks = read_pdf(pdf_with(|_fixture| {
            (
                vec![vec![
                    "<< /Subtype /Text /Contents (first) >>".to_string(),
                    "<< /Subtype /Text /Contents (second) >>".to_string(),
                ]],
                " /AcroForm << /Fields [<< /FT /Tx /T (One) /V (1) >> \
                 << /FT /Tx /T (Two) /V (2) >> << /FT /Tx /V (unnamed) >>] >>"
                    .to_string(),
            )
        }));
        assert_eq!(
            marks.annotations.as_deref(),
            Some("Annotations:\nnote on page 1:\n  first\nnote on page 1:\n  second")
        );
        assert_eq!(
            marks.fields.as_deref(),
            Some(
                "Form fields:\nOne (text): 1\nTwo (text): 2\n\
                 unnamed field (text; not fillable): unnamed"
            )
        );
    }

    /// A container whose widget kids each carry their own `/FT`/`/T`/`/V` is read
    /// as the fields it holds, named the way the filler names them — and a checked
    /// button whose `/Opt` does not name the state it is in shows only its state,
    /// never a name guessed from the state's position.
    #[test]
    fn self_contained_widget_kids_are_read_as_their_own_fields() {
        let marks = read_pdf(pdf_with(|fixture| {
            let appearance =
                fixture.push("<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] >>".to_string());
            let group = fixture.reserve();
            let first = fixture.push(format!(
                "<< /Subtype /Widget /Parent {group} /FT /Tx /T (First) /V (one) \
                 /Rect [0 0 0 0] >>"
            ));
            let plan = fixture.push(format!(
                "<< /Subtype /Widget /FT /Btn /T (Plan) /V /1 /Opt [[(one) (One)] [(two) (Two)]] \
                 /AP << /N << /1 {appearance} /2 {appearance} /Off {appearance} >> >> /AS /1 >>"
            ));
            fixture.set(&group, format!("<< /T (Group) /Kids [{first}] >>"));
            let acroform = fixture.push(format!("<< /Fields [{group} {plan}] >>"));
            (vec![vec![first, plan]], format!(" /AcroForm {acroform}"))
        }));
        assert_eq!(
            marks.fields.as_deref(),
            Some(
                "Form fields:\nGroup.First (text) on page 1: one\n\
                 Plan (checkbox) on page 1: checked"
            )
        );
    }

    /// The document's own description leads its reading: only the entries the
    /// Info dictionary carries, in a fixed order, with dates in calendar form
    /// and the file's own offset.
    #[test]
    fn a_documents_description_shows_only_what_it_carries() {
        let marks = read_pdf(pdf_with_trailer(|fixture| {
            let info = fixture.push(format!(
                "<< /Title {} /Author {} /Subject {} /Keywords {} /Creator {} /Producer {} \
                 /CreationDate (D:20240102030405+03'30') /ModDate (D:20230607080910Z) >>",
                pdf_text_string("Annual report"),
                pdf_text_string("Иван Петров"),
                pdf_text_string("Figures and notes"),
                pdf_text_string("report, figures"),
                pdf_text_string("Writer"),
                pdf_text_string("MahBot"),
            ));
            (vec![vec![]], String::new(), format!(" /Info {info}"))
        }));
        let description = marks
            .description
            .as_deref()
            .expect("the file carries a description");
        assert_eq!(
            description,
            "Description:\n  title: Annual report\n  author: Иван Петров\n  \
             subject: Figures and notes\n  keywords: report, figures\n  \
             creator: Writer\n  producer: MahBot\n  \
             created: 2024-01-02T03:04:05+03:30\n  modified: 2023-06-07T08:09:10Z"
        );
        // Every field the shared rules state is one this reading knows how to
        // print, so a field added to the rules without a value here is caught.
        for field in crate::docgen::pdf_metadata_fields() {
            assert!(
                description.contains(&format!("{}: ", field.key)),
                "the description must print every rules field, missing {:?}: {description}",
                field.key
            );
        }
    }
}
