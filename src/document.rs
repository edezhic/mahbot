//! Document conversion: turn a document's bytes into the text and image files
//! an agent can read, and report a short user-facing reason when it cannot.
//!
//! Shared by the inbound attachment flow ([`crate::channels::enrichment`]) and
//! by the read tool ([`crate::tools`]); nothing here touches channel state, the
//! database, or the network. [`convert_document_file`] is the bounded async
//! entry point both go through, and [`needs_extraction`] tells the read tool
//! whether a file is one of the containers this module extracts from.
//!
//! # Invariants
//!
//! - **Pure and CPU-bound.** No state but the shared conversion semaphore; the
//!   only I/O is reading the input bytes and writing artifacts. Conversion is
//!   synchronous on purpose: a slow rasterization never stalls an async task.
//! - **No panics of its own**, for any input including empty or truncated
//!   bytes: every fallible step degrades to a skipped artifact, a user-facing
//!   note, or [`DocOutcome::Unreadable`]. The structure parse and the
//!   `pdf-extract` text pass each run inside [`std::panic::catch_unwind`], so a
//!   panic in either one loses that pass alone. The decoders themselves are
//!   trusted, not hardened: a malicious PDF can overflow the stack or exhaust
//!   memory inside a decoder and abort the process (the filter chain inflates
//!   into an unbounded buffer), and a decoder that panics instead is only
//!   contained at the caller's blocking boundary (as [`DocOutcome::Unreadable`],
//!   losing the text pass with it). The bounds this module declares are its own
//!   (see `embedded_image_jpeg`), not the decoders'.
//! - **Content-first detection.** Magic bytes decide the format; the extension
//!   only disambiguates formats that share a container (a ZIP is a `.docx` only
//!   when the name says so) or that have no magic (plain text).
//! - **`out_dir` is created on demand** and is the only place artifacts are
//!   written, with names taken from the source/entry file name — never from a
//!   full ZIP entry path, so a crafted archive cannot write outside it.

use crate::util::media_target::{RASTER_DECODE_MAX_ALLOC_BYTES, RASTER_DECODE_MAX_DIMENSION_PX};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::content::TypedIter;
use hayro::hayro_syntax::content::ops::TypedInstruction;
use hayro::hayro_syntax::object::dict::Dict;
use hayro::hayro_syntax::object::stream::{ImageColorSpace, ImageDecodeParams, Stream};
use hayro::hayro_syntax::object::{Array, Name, Object, ObjectIdentifier};
use hayro::hayro_syntax::page::Page;
use hayro::hayro_syntax::{DecryptionError, LoadPdfError, Pdf};
use hayro::vello_cpu::color::Rgba8;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{RenderCache, RenderSettings};
use image::codecs::jpeg::JpegEncoder;
use image::{RgbImage, RgbaImage};
use quick_xml::Reader;
use quick_xml::events::{BytesRef, Event};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

/// Maximum extracted-text length (Unicode chars) inlined into the message.
pub(crate) const INLINE_TEXT_MAX_CHARS: usize = 5000;

/// Minimum trimmed text-layer length for a page to count as having real text.
/// Shorter than this is page-number/decoration noise or whitespace, so the page
/// is treated as imageless-of-text and rendered instead.
const MIN_PAGE_TEXT_CHARS: usize = 16;

/// Target pixel size of the long side of a rasterized page.
const RASTER_LONG_SIDE_PX: f32 = 1600.0;

/// Maximum rasterization scale, so a tiny page is not blown up past usefulness.
const RASTER_MAX_SCALE: f32 = 3.0;

/// JPEG quality for rasterized pages: high enough that small print stays
/// legible, low enough that a full page stays a few hundred kilobytes.
const RASTER_JPEG_QUALITY: u8 = 90;

/// Magic bytes at the start of every PDF.
const PDF_MAGIC: &[u8] = b"%PDF-";
/// Magic bytes at the start of a ZIP local file header (`.docx`/`.docm` here).
const ZIP_MAGIC: &[u8] = b"PK\x03\x04";
/// CFB/OLE container magic — what an *encrypted* OOXML package is wrapped in.
const CFB_MAGIC: &[u8] = &[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
/// Extensions accepted as OOXML Word packages (`docm` is a macro-enabled docx).
const DOCX_EXTENSIONS: &[&str] = &["docx", "docm"];
/// Maximum bytes decompressed from a single ZIP entry, so a lying size header
/// cannot inflate the temp dir. Deliberately not an aggregate bound: the entry
/// count is unbounded.
const MAX_ZIP_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
/// Part holding the WordprocessingML body.
const DOCX_BODY_PART: &str = "word/document.xml";
/// Prefix of the embedded-media parts in a Word package.
const DOCX_MEDIA_PREFIX: &str = "word/media/";
/// Media extensions written through to `out_dir` verbatim. Everything else
/// (emf/wmf/tiff/bmp/gif/svg/...) would need transcoding this module avoids.
const EMBEDDED_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];
/// Extensions treated as plain text regardless of content.
const PLAIN_TEXT_EXTENSIONS: &[&str] = &[
    "md", "markdown", "txt", "text", "rst", "adoc", "org", "csv", "json", "yaml", "yml", "toml",
    "log",
];

/// Reason reported for a document whose bytes could not be read or whose
/// conversion panicked.
const UNREADABLE_REASON: &str = "could not be read";
/// Bound on concurrently converting documents: each conversion holds the whole
/// document plus its decoded rasters, and every caller runs on its own task, so
/// a burst would otherwise multiply peak memory.
static DOCUMENT_CONVERSIONS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

// ── Delivery notes shared by both callers ───────────────────────
//
// The inbound attachment path and the read tool each wrap these bodies in their
// own `[<source>: ...]` prefix (a file name and a path respectively), so the
// sentence a model reads for the same situation cannot drift between them.

/// Body of the note shown when a document produced images but no text.
pub(crate) const NO_TEXT_LAYER_NOTE: &str =
    "no text could be extracted — the pages were provided as images";

/// Body of the note shown when a document produced neither text nor images.
pub(crate) const NO_TEXT_NOTE: &str = "no text could be extracted";

/// Body of the note pointing at the file `path` holding extracted text that was
/// too long to inline.
#[must_use]
pub(crate) fn spilled_text_note(chars: usize, path: &Path) -> String {
    format!(
        "the extracted text is too long to inline ({chars} characters); the full text was saved to {} — read that file.",
        path.display()
    )
}

/// The result of converting one document.
#[derive(Debug)]
pub(crate) enum DocOutcome {
    /// Text was extracted. `text` may be empty when the document has no text
    /// layer; `images` are files written into `out_dir`, ready for the existing
    /// inbound IMAGE pipeline; `notes` are non-fatal, already user-facing notes
    /// (e.g. skipped undecodable embedded media).
    Text {
        text: String,
        images: Vec<PathBuf>,
        notes: Vec<String>,
    },
    /// A recognized format whose bytes could not be read (corrupt, or
    /// encrypted/password-protected). `reason` is user-facing and short.
    Unreadable { reason: String },
    /// A format this project does not convert.
    Unsupported,
}

/// What [`convert_document`] found in `bytes`: the format whose container the
/// magic marks, or the extension's verdict where the container is shared.
enum DocumentKind {
    Pdf,
    Docx,
    /// An encrypted OOXML package: a CFB container, which needs a password
    /// rather than a conversion.
    EncryptedOoxml,
    PlainText,
    Unsupported,
}

/// Classify `bytes` named `path`: magic bytes first, the extension only where
/// the container is shared (a ZIP is a `.docx` only when the name says so) or
/// absent (plain text). The single dispatch behind [`convert_document`] and
/// [`needs_extraction`], so detection cannot drift between them.
fn classify(bytes: &[u8], path: &Path) -> DocumentKind {
    if bytes.starts_with(PDF_MAGIC) {
        return DocumentKind::Pdf;
    }
    // Checked before the ZIP attempt: an encrypted OOXML package is a CFB
    // container, and its bytes would otherwise look like corruption. Only a
    // Word-named one gets the password-protected verdict — CFB is also what a
    // legacy .doc/.xls/.ppt is, and those are not a format this converts.
    if bytes.starts_with(CFB_MAGIC) {
        return if crate::util::has_extension(path, DOCX_EXTENSIONS) {
            DocumentKind::EncryptedOoxml
        } else {
            DocumentKind::Unsupported
        };
    }
    if bytes.starts_with(ZIP_MAGIC) {
        // ZIP magic alone is shared by xlsx/pptx/plain archives; only the name
        // can tell a Word package apart.
        return if crate::util::has_extension(path, DOCX_EXTENSIONS) {
            DocumentKind::Docx
        } else {
            DocumentKind::Unsupported
        };
    }
    // A known text extension is decoded lossily; anything else must look like
    // text (valid UTF-8, no NUL) to qualify.
    if crate::util::has_extension(path, PLAIN_TEXT_EXTENSIONS) || is_plain_utf8(bytes) {
        return DocumentKind::PlainText;
    }
    DocumentKind::Unsupported
}

/// Whether a document's leading bytes mark one of the containers this module
/// extracts text and images from — the kinds that must be converted rather
/// than read as bytes. Only the head is needed: every extraction arm is
/// magic-local, and the text arm (plain UTF-8 / known text extension) is
/// exactly the "no extraction needed" case.
#[must_use]
pub(crate) fn needs_extraction(head: &[u8], file_name: &str) -> bool {
    matches!(
        classify(head, Path::new(file_name)),
        DocumentKind::Pdf | DocumentKind::Docx | DocumentKind::EncryptedOoxml
    )
}

/// Convert `bytes` named `file_name`, writing extracted page/embedded images
/// into `out_dir` (created if missing). Synchronous and CPU-bound — callers run
/// it on a blocking thread. Format detection is content-first (magic bytes),
/// with the file extension as a secondary signal.
#[must_use]
fn convert_document(bytes: &[u8], file_name: &str, out_dir: &Path) -> DocOutcome {
    match classify(bytes, Path::new(file_name)) {
        DocumentKind::Pdf => convert_pdf(bytes, out_dir),
        DocumentKind::EncryptedOoxml => DocOutcome::Unreadable {
            reason: "password-protected".to_string(),
        },
        DocumentKind::Docx => convert_docx(bytes, out_dir),
        DocumentKind::PlainText => DocOutcome::Text {
            text: String::from_utf8_lossy(bytes).into_owned(),
            images: Vec::new(),
            notes: Vec::new(),
        },
        DocumentKind::Unsupported => DocOutcome::Unsupported,
    }
}

/// Convert the document at `path`, named `file_name`, into text and images,
/// writing extracted rasters into `out_dir`. The single bounded entry point
/// shared by the inbound attachment path and the read tool.
///
/// `pdf-extract` panics on malformed input, so the conversion runs on a
/// blocking thread and its panic is contained at that boundary: a
/// [`tokio::task::JoinError`] degrades to an unreadable document instead of
/// taking the caller's turn down with it. The conversion semaphore is
/// deliberately shared by both callers, so peak conversion concurrency stays at
/// two daemon-wide and a local read can queue behind a busy inbound
/// conversion — the accepted trade-off, since both hold whole documents plus
/// their decoded rasters in memory.
pub(crate) async fn convert_document_file(
    path: &Path,
    file_name: &str,
    out_dir: &Path,
) -> DocOutcome {
    // Held across the read and the conversion, covering the document bytes and
    // the conversion's rasters (encoding the extracted pages into data URIs
    // happens later, outside this bound). The semaphore is never closed, so
    // acquisition cannot fail.
    let _permit = DOCUMENT_CONVERSIONS.acquire().await;
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "Failed to read the document"
            );
            return DocOutcome::Unreadable {
                reason: UNREADABLE_REASON.to_string(),
            };
        }
    };
    let name = file_name.to_string();
    let out_dir = out_dir.to_path_buf();
    match tokio::task::spawn_blocking(move || convert_document(&bytes, &name, &out_dir)).await {
        Ok(outcome) => outcome,
        Err(e) => {
            tracing::warn!(error = %e, "Document conversion failed");
            DocOutcome::Unreadable {
                reason: UNREADABLE_REASON.to_string(),
            }
        }
    }
}

/// Extract text, page rasters and embedded images from a PDF: a page with a
/// usable text layer is inlined together with every image XObject that can be
/// decoded from it — and every inline image its content stream paints — while a
/// page without one is rasterized at the scale bounded by
/// [`RASTER_LONG_SIDE_PX`] instead, which already carries its embedded images.
fn convert_pdf(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    // The structure parse also settles encryption: it decrypts with the empty
    // user password, so an owner-password-only document opens and is readable,
    // while one that needs a password reports `Decryption`. hayro also supplies
    // the page list both remaining passes walk: its pages are in document order,
    // the same order `pdf-extract`'s per-page text list uses.
    //
    // A panic in the parse is caught like one in the text pass below it: the
    // document keeps whatever the other pass can read instead of reaching the
    // caller's blocking boundary as an unreadable one.
    let pdf = match std::panic::catch_unwind(|| Pdf::new(bytes.to_vec())) {
        Ok(Ok(pdf)) => Some(pdf),
        Ok(Err(LoadPdfError::Decryption(DecryptionError::PasswordProtected))) => {
            return DocOutcome::Unreadable {
                reason: "password-protected".to_string(),
            };
        }
        Ok(Err(_)) | Err(_) => None,
    };
    // `pdf-extract` panics on some malformed structures; a panic is treated
    // exactly like its `Err` — no page text, so every page takes the visual
    // path.
    let page_texts = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem_by_pages(bytes)
    }));
    let page_texts = match page_texts {
        Ok(Ok(texts)) => texts,
        // Parsed, but the text pass failed: rasterize every page instead.
        _ if pdf.is_some() => Vec::new(),
        // Nothing can be read at all — no text and no structure to render — so
        // the document is reported as such instead of as an empty one.
        _ => {
            return DocOutcome::Unreadable {
                reason: "could not be parsed".to_string(),
            };
        }
    };
    // Either list running short degrades safely: a page past `page_texts` has no
    // text to inline, and a page past the page tree has nothing to render or
    // extract.
    let page_count = pdf.as_ref().map_or(page_texts.len(), |pdf| {
        page_texts.len().max(pdf.pages().len())
    });

    ensure_out_dir(out_dir);

    let mut text_pages: Vec<String> = Vec::new();
    let mut images: Vec<PathBuf> = Vec::new();
    // Embedded images left out of the conversion on text pages: such a page is
    // never rasterized, so without one aggregated note per cause they would
    // vanish silently.
    let mut skipped = SkippedImages::default();
    // Text pages left without a page-tree entry, because there is no structure
    // to render them from.
    let mut pages_without_structure = 0usize;
    // The images already written for the document, shared across pages (see
    // [`write_embedded_images`]).
    let mut written = WrittenImages::default();
    for index in 0..page_count {
        let page_text = page_texts.get(index).map_or("", String::as_str);
        let page = pdf.as_ref().and_then(|pdf| pdf.pages().get(index));
        if is_usable_page_text(page_text) {
            // Trim so the "\n\n" join does not double up with the extractor's
            // own trailing whitespace.
            text_pages.push(page_text.trim().to_string());
            if let Some(page) = page {
                write_embedded_images(
                    page,
                    index + 1,
                    out_dir,
                    &mut written,
                    &mut images,
                    &mut skipped,
                );
            }
        } else if let Some(page) = page {
            rasterize_page(page, index + 1, out_dir, &mut images);
        } else {
            pages_without_structure += 1;
        }
    }
    // Aggregated: a text layer without a readable page tree is one degradation,
    // not one log line per page of a long scanned document.
    if pages_without_structure > 0 {
        tracing::warn!(
            pages = pages_without_structure,
            "document: text pages missing from the PDF page tree"
        );
    }
    DocOutcome::Text {
        text: text_pages.join("\n\n"),
        images,
        notes: skipped.notes(),
    }
}

/// Extract body text and embedded images from a `.docx`/`.docm` ZIP package.
fn convert_docx(bytes: &[u8], out_dir: &Path) -> DocOutcome {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(bytes)) else {
        return unreadable_docx();
    };
    let Some(body_xml) = read_zip_entry(&mut archive, DOCX_BODY_PART) else {
        return unreadable_docx();
    };
    let Some(text) = docx_body_text(&body_xml) else {
        return unreadable_docx();
    };

    ensure_out_dir(out_dir);

    let entries: Vec<String> = archive
        .file_names()
        // A document's media parts are the entries under `word/media/` that name
        // a file, i.e. carry an extension. Directory entries (`word/media/`,
        // `word/media\`) and extension-less names are not media: their base name
        // would otherwise be ingested as a nonexistent image and counted as
        // skipped.
        .filter(|name| name.starts_with(DOCX_MEDIA_PREFIX) && Path::new(name).extension().is_some())
        .map(str::to_owned)
        .collect();
    let mut images = Vec::new();
    let mut next_suffix = HashMap::new();
    // A media entry the existing image pipeline cannot take is skipped with the
    // same aggregated note the PDF path emits: an honest count, not one line per
    // figure of a document full of them.
    let mut skipped = SkippedImages::default();
    for entry in entries {
        // Only the entry's file name is used: a crafted `word/media/../..`
        // entry must never escape `out_dir`.
        let file_name = crate::util::neutralized_name(crate::util::file_name_or_path(&entry));
        if !crate::util::has_extension(Path::new(&file_name), EMBEDDED_IMAGE_EXTENSIONS) {
            skipped.add(SkipReason::Unsupported);
            continue;
        }
        let Some(content) = read_zip_entry(&mut archive, &entry) else {
            tracing::warn!(%file_name, "document: failed to read embedded .docx image");
            skipped.add(SkipReason::Failed);
            continue;
        };
        let path = unique_out_path(out_dir, &file_name, &mut next_suffix);
        match std::fs::write(&path, &content) {
            Ok(()) => images.push(path),
            Err(e) => {
                tracing::warn!(%file_name, error = %e, "document: failed to write embedded .docx image");
                skipped.add(SkipReason::Failed);
            }
        }
    }
    DocOutcome::Text {
        text,
        images,
        notes: skipped.notes(),
    }
}

/// The shared "recognized `.docx`-shaped package that cannot be read" outcome.
fn unreadable_docx() -> DocOutcome {
    DocOutcome::Unreadable {
        reason: "corrupt or unsupported .docx".to_string(),
    }
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
/// Only the `.docx` path needs this: PDF artifacts are named from page index and
/// image slot, and a document's images are consumed before the next one converts.
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

/// Embedded images left out of the conversion, by cause.
#[derive(Default, Clone, Copy)]
struct SkippedImages {
    /// Format, colour space or bit depth this module does not convert.
    unsupported: usize,
    /// Decoding, reading or writing the image failed.
    failed: usize,
    /// Declared geometry over the shared raster envelope.
    oversized: usize,
}

impl SkippedImages {
    fn add(&mut self, reason: SkipReason) {
        match reason {
            SkipReason::Unsupported => self.unsupported += 1,
            SkipReason::Failed => self.failed += 1,
            SkipReason::Oversized => self.oversized += 1,
        }
    }

    /// One note per cause that happened, so a note never claims a decode
    /// failure for an image this module simply does not convert.
    fn notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        for (count, text) in [
            (self.unsupported, "in a format this pipeline cannot convert"),
            (self.failed, "that could not be extracted"),
            (self.oversized, "over the image size limit"),
        ] {
            if count > 0 {
                notes.push(format!("skipped {count} embedded image(s) {text}"));
            }
        }
        notes
    }
}

/// Why one embedded image is not in the conversion.
#[derive(Clone, Copy)]
enum SkipReason {
    Unsupported,
    Failed,
    Oversized,
}

/// The images already written for the document: image XObject identities, plus
/// the form XObjects whose inline images have been collected. A producer
/// references one image object from every page that shows it and paints the same
/// header form on every page, and the model needs each picture once.
#[derive(Default)]
struct WrittenImages {
    x_objects: HashSet<ObjectIdentifier>,
    forms: HashSet<ObjectIdentifier>,
}

/// Write every image in scope for `page` — its image XObjects, plus the inline
/// images its own content stream and those of the forms it paints through draw —
/// into `out_dir` as `page_<n>_img_<k>.jpg` (`k` indexing the images this page
/// wrote), pushing each path onto `images` and counting each one left out onto
/// `skipped`. A page without a usable text layer is rasterized whole instead, so
/// its embedded images are never collected here.
///
/// `written` carries what the document has already written: an image object is
/// skipped when an earlier page wrote it, and a form when an earlier page
/// collected its inline images. A form without an object id is not deduplicated.
fn write_embedded_images(
    page: &Page<'_>,
    page_number: usize,
    out_dir: &Path,
    written: &mut WrittenImages,
    images: &mut Vec<PathBuf>,
    skipped: &mut SkippedImages,
) {
    let mut slot = 0;
    let (image_streams, form_streams) = page_image_streams(page);
    for stream in image_streams {
        if let Some(id) = stream.dict().obj_id()
            && !written.x_objects.insert(id)
        {
            continue;
        }
        if let Some(reason) = write_embedded_image(&stream, page_number, &mut slot, out_dir, images)
        {
            skipped.add(reason);
        }
    }
    if let Some(content) = page.page_stream() {
        write_inline_images(content, page_number, &mut slot, out_dir, images, skipped);
    }
    // A form is a content stream an inline image can sit in just as well.
    for form in form_streams {
        if let Some(id) = form.dict().obj_id()
            && !written.forms.insert(id)
        {
            continue;
        }
        if let Ok(content) = form.decoded() {
            write_inline_images(&content, page_number, &mut slot, out_dir, images, skipped);
        }
    }
}

/// Write one decoded image as `page_<n>_img_<k>.jpg`, advancing `slot`. Reports
/// why it is not in the conversion: an image that could not be written is as
/// absent from `out_dir` as one that could not be decoded.
fn write_embedded_image(
    stream: &Stream<'_>,
    page_number: usize,
    slot: &mut usize,
    out_dir: &Path,
    images: &mut Vec<PathBuf>,
) -> Option<SkipReason> {
    let jpeg = match embedded_image_jpeg(stream) {
        Ok(jpeg) => jpeg,
        Err(reason) => return Some(reason),
    };
    let path = out_dir.join(format!("page_{page_number}_img_{slot}.jpg"));
    *slot += 1;
    match std::fs::write(&path, jpeg) {
        Ok(()) => {
            images.push(path);
            None
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "document: failed to write embedded PDF image");
            Some(SkipReason::Failed)
        }
    }
}

/// Write the inline images (`BI`/`ID`/`EI`) a content stream paints, counting
/// each one left out onto `skipped`. They live in the content stream itself, so
/// they carry no object identity to deduplicate by — and one is normally painted
/// once.
fn write_inline_images(
    content: &[u8],
    page_number: usize,
    slot: &mut usize,
    out_dir: &Path,
    images: &mut Vec<PathBuf>,
    skipped: &mut SkippedImages,
) {
    let mut operations = TypedIter::new(content);
    while let Some(operation) = operations.next() {
        if let TypedInstruction::InlineImage(image) = operation
            && let Some(reason) = write_embedded_image(image.0, page_number, slot, out_dir, images)
        {
            skipped.add(reason);
        }
    }
}

/// The `(image XObjects, form XObjects)` in scope for `page`: those its own
/// resources and the inherited page-tree resources name, plus those inside any
/// form XObject reachable from either — a form is a container the page's content
/// paints through, so its images are the page's images (and a page without a text
/// layer shows them anyway, through its raster).
fn page_image_streams<'a>(page: &Page<'a>) -> (Vec<Stream<'a>>, Vec<Stream<'a>>) {
    let mut images = Vec::new();
    let mut forms = Vec::new();
    // A form reachable down more than one path is entered once, which is also
    // what bounds the walk when forms reference each other.
    let mut visited: HashSet<ObjectIdentifier> = HashSet::new();
    let mut pending: VecDeque<Stream<'a>> = page_x_objects(page).into();
    while let Some(stream) = pending.pop_front() {
        if let Some(id) = stream.dict().obj_id()
            && !visited.insert(id)
        {
            continue;
        }
        match x_object_subtype(&stream) {
            XObjectSubtype::Image => images.push(stream),
            XObjectSubtype::Form => {
                pending.extend(form_x_objects(&stream));
                forms.push(stream);
            }
            XObjectSubtype::Other => {}
        }
    }
    (images, forms)
}

/// The XObjects the page's own resources name, followed by those inherited from
/// the page-tree resources. A name defined at more than one level is taken from
/// the most specific definition — the one the content stream paints.
fn page_x_objects<'a>(page: &Page<'a>) -> Vec<Stream<'a>> {
    let mut seen: HashSet<Name<'a>> = HashSet::new();
    let mut streams = Vec::new();
    let mut level = Some(page.resources());
    while let Some(resources) = level {
        for name in resources.x_objects.keys() {
            if seen.insert(name.clone())
                && let Some(stream) = resources.get_x_object(&name)
            {
                streams.push(stream);
            }
        }
        level = resources.parent();
    }
    streams
}

/// The XObjects a form XObject's own resources name.
fn form_x_objects<'a>(form: &Stream<'a>) -> Vec<Stream<'a>> {
    let Some(x_objects) = form
        .dict()
        .get::<Dict<'a>>(b"Resources")
        .and_then(|resources| resources.get::<Dict<'a>>(b"XObject"))
    else {
        return Vec::new();
    };
    x_objects
        .keys()
        .filter_map(|name| x_objects.get::<Stream<'a>>(&name))
        .collect()
}

/// What a `/XObject` entry holds: only images carry samples, and only forms
/// nest further resources.
enum XObjectSubtype {
    Image,
    Form,
    Other,
}

fn x_object_subtype(stream: &Stream<'_>) -> XObjectSubtype {
    match stream.dict().get::<Name<'_>>(b"Subtype").as_deref() {
        Some(b"Image") => XObjectSubtype::Image,
        Some(b"Form") => XObjectSubtype::Form,
        _ => XObjectSubtype::Other,
    }
}

/// The JPEG bytes of one image — an image XObject, or an inline image painted by
/// a content stream: a lone `DCTDecode` stream is passed through verbatim,
/// because its bytes already are a complete JPEG, and anything else goes through
/// the filter chain and is converted to an 8-bit RGB raster at the bounded raster
/// size. Every key is read in both spellings, because an inline image abbreviates
/// all of them. `Err(reason)` when the image is not in the conversion.
fn embedded_image_jpeg(stream: &Stream<'_>) -> Result<Vec<u8>, SkipReason> {
    let dict = stream.dict();
    let width = dict
        .get::<u32>(b"Width")
        .or_else(|| dict.get::<u32>(b"W"))
        .filter(|width| *width > 0)
        .ok_or(SkipReason::Unsupported)?;
    let height = dict
        .get::<u32>(b"Height")
        .or_else(|| dict.get::<u32>(b"H"))
        .filter(|height| *height > 0)
        .ok_or(SkipReason::Unsupported)?;
    // A declared size the inbound image pipeline could not decode itself is a
    // header bomb rather than a big image. Checked before the passthrough below,
    // so the bound covers every route.
    if width > RASTER_DECODE_MAX_DIMENSION_PX || height > RASTER_DECODE_MAX_DIMENSION_PX {
        return Err(SkipReason::Oversized);
    }
    // Checked before the colour space and the bit depth: a JPEG whose dictionary
    // omits or misnames either is still a complete JPEG.
    if is_lone_dct(dict) {
        return Ok(stream.raw_data().into_owned());
    }
    let is_mask = dict
        .get::<bool>(b"ImageMask")
        .or_else(|| dict.get::<bool>(b"IM"))
        .unwrap_or(false);
    let colour = if is_mask {
        // A stencil mask is one component — "paint here or do not" — and has no
        // colour space of its own.
        ColourSpace::Gray
    } else {
        let colour_space = dict
            .get::<Object<'_>>(b"ColorSpace")
            .or_else(|| dict.get::<Object<'_>>(b"CS"))
            .ok_or(SkipReason::Unsupported)?;
        parse_colour_space(colour_space).ok_or(SkipReason::Unsupported)?
    };
    let bits_per_component = dict
        .get::<u8>(b"BitsPerComponent")
        .or_else(|| dict.get::<u8>(b"BPC"))
        .unwrap_or(if is_mask { 1 } else { 8 });
    if !matches!(bits_per_component, 1 | 2 | 4 | 8 | 16) {
        return Err(SkipReason::Unsupported);
    }
    // The filter chain expands the samples at the declared size, so a declared
    // buffer over the decode budget is refused rather than allocated. A stream
    // whose real size comes from its bitstream (JBIG2) or `/DecodeParms` (CCITT
    // `/Columns`/`/Rows`) is bounded by its decoder instead.
    let declared_samples = u64::from(width)
        * u64::from(height)
        * u64::from(colour.components())
        * u64::from(bits_per_component)
        / 8;
    if declared_samples > RASTER_DECODE_MAX_ALLOC_BYTES {
        return Err(SkipReason::Oversized);
    }
    let decoded = stream
        .decoded_image(&ImageDecodeParams {
            is_indexed: matches!(colour, ColourSpace::Indexed(_)),
            bpc: Some(bits_per_component),
            num_components: Some(colour.components()),
            target_dimension: Some(scaled_dimensions(width, height)),
            width,
            height,
        })
        .map_err(|_| SkipReason::Failed)?;
    // A filter chain that returns nothing for a non-empty image failed leniently
    // (a lenient inflate reports an empty stream, not an error): the image is
    // undecodable, not blank.
    if decoded.data.is_empty() {
        return Err(SkipReason::Failed);
    }
    // What a decoder reports about its own output is authoritative over the
    // dictionary: the two disagree in the wild.
    let (width, height, bits_per_component, colour) = match &decoded.image_data {
        Some(data) => {
            // An indexed image keeps its palette: the decoder reports only the
            // component count of the palette's base space.
            let colour = if matches!(colour, ColourSpace::Indexed(_)) {
                colour
            } else {
                colour_space_of(data.color_space).map_err(|()| SkipReason::Unsupported)?
            };
            (data.width, data.height, data.bits_per_component, colour)
        }
        None => (width, height, bits_per_component, colour),
    };
    if !matches!(bits_per_component, 1 | 2 | 4 | 8 | 16) {
        return Err(SkipReason::Unsupported);
    }
    let decode = decode_ranges(dict, colour.components());
    let shape = ImageShape {
        width,
        height,
        bits_per_component,
        colour,
        decode,
        alpha: decoded
            .image_data
            .as_ref()
            .and_then(|data| data.alpha.as_deref()),
    };
    let raster = samples_to_raster(&shape, &decoded.data).ok_or(SkipReason::Failed)?;
    encode_jpeg(&raster).map_err(|()| SkipReason::Failed)
}

/// Whether `dict`'s only filter is `DCTDecode` (or its `DCT` abbreviation),
/// i.e. the stream's raw bytes are already a complete JPEG.
fn is_lone_dct(dict: &Dict<'_>) -> bool {
    /// `DCT` is the abbreviation the spec allows for `/DCTDecode`.
    fn is_dct(name: &Name<'_>) -> bool {
        name.as_ref() == b"DCTDecode" || name.as_ref() == b"DCT"
    }
    match dict
        .get::<Object<'_>>(b"Filter")
        .or_else(|| dict.get::<Object<'_>>(b"F"))
    {
        Some(Object::Name(name)) => is_dct(&name),
        Some(Object::Array(filters)) => {
            let mut entries = filters.iter::<Name<'_>>();
            entries.next().is_some_and(|name| is_dct(&name)) && entries.next().is_none()
        }
        _ => false,
    }
}

/// Map a decoder-reported colour space onto the shapes this module converts. A
/// multi-band decoder whose colour space the dictionary did not describe cannot
/// be converted from its components alone.
fn colour_space_of(colour_space: Option<ImageColorSpace>) -> Result<ColourSpace, ()> {
    match colour_space {
        Some(ImageColorSpace::Gray) => Ok(ColourSpace::Gray),
        Some(ImageColorSpace::Rgb) => Ok(ColourSpace::Rgb),
        Some(ImageColorSpace::Cmyk) => Ok(ColourSpace::Cmyk),
        Some(ImageColorSpace::Unknown(_)) | None => Err(()),
    }
}

/// A colour space an image's samples can be converted from. Anything with a
/// transform of its own — Lab, Separation, DeviceN, a pattern, an unknown ICC
/// profile — is not one of these.
enum ColourSpace {
    Gray,
    Rgb,
    Cmyk,
    /// A single index sample into an RGB palette.
    Indexed(Vec<[u8; 3]>),
}

impl ColourSpace {
    /// Samples per pixel: the component count the filter chain packs its output
    /// with.
    fn components(&self) -> u8 {
        match self {
            Self::Gray | Self::Indexed(_) => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
        }
    }
}

/// Parse an image XObject's `/ColorSpace`: a name, or a family array whose first
/// entry is the family name. `None` for every colour space this module cannot
/// convert.
fn parse_colour_space(object: Object<'_>) -> Option<ColourSpace> {
    parse_colour_space_within(object, true)
}

/// [`parse_colour_space`] with `allow_indexed` cleared for the base of an
/// `/Indexed` space, which is a device or ICC space and never another indexed
/// one: a crafted chain of nested arrays cannot deepen the walk.
fn parse_colour_space_within(object: Object<'_>, allow_indexed: bool) -> Option<ColourSpace> {
    let name = match &object {
        Object::Name(name) => name.clone(),
        Object::Array(entries) => entries.iter::<Name<'_>>().next()?,
        _ => return None,
    };
    match name.as_ref() {
        b"DeviceGray" | b"CalGray" | b"G" => Some(ColourSpace::Gray),
        b"DeviceRGB" | b"CalRGB" | b"RGB" => Some(ColourSpace::Rgb),
        b"DeviceCMYK" | b"CMYK" => Some(ColourSpace::Cmyk),
        b"ICCBased" => match icc_components(object.into_array()?.iter::<Object<'_>>().nth(1)?)? {
            1 => Some(ColourSpace::Gray),
            3 => Some(ColourSpace::Rgb),
            4 => Some(ColourSpace::Cmyk),
            _ => None,
        },
        b"Indexed" | b"I" => {
            if !allow_indexed {
                return None;
            }
            // `[/Indexed base hival lookup]`: the lookup holds `hival + 1`
            // entries of the base space's components.
            let mut entries = object.into_array()?.iter::<Object<'_>>();
            let base = parse_colour_space_within(entries.nth(1)?, false)?;
            let hival = usize::try_from(entries.next()?.into_i32()?).ok()?;
            Some(ColourSpace::Indexed(palette_entries(
                &base,
                hival,
                entries.next()?,
            )?))
        }
        _ => None,
    }
}

/// The component count (`/N`) of an `/ICCBased` profile stream, limited to the
/// gray/RGB/CMYK models this module converts.
fn icc_components(profile: Object<'_>) -> Option<u8> {
    let components = match profile {
        Object::Stream(stream) => stream.dict().get::<u8>(b"N")?,
        Object::Dict(dict) => dict.get::<u8>(b"N")?,
        _ => return None,
    };
    matches!(components, 1 | 3 | 4).then_some(components)
}

/// The RGB palette of an `/Indexed` colour space, from a string or a (filtered)
/// stream lookup. Only a gray or RGB base converts — the two entry widths whose
/// bytes map onto a channel count.
fn palette_entries(base: &ColourSpace, hival: usize, lookup: Object<'_>) -> Option<Vec<[u8; 3]>> {
    let entry_len = match base {
        ColourSpace::Gray => 1,
        ColourSpace::Rgb => 3,
        _ => return None,
    };
    let lookup = match lookup {
        Object::String(lookup) => lookup.as_bytes().to_vec(),
        Object::Stream(lookup) => lookup.decoded().ok()?.into_owned(),
        _ => return None,
    };
    let len = hival.checked_add(1)?.checked_mul(entry_len)?;
    if lookup.len() < len {
        return None;
    }
    Some(
        lookup[..len]
            .chunks_exact(entry_len)
            .map(|entry| match *entry {
                [gray] => [gray; 3],
                [red, green, blue] => [red, green, blue],
                _ => [0, 0, 0],
            })
            .collect(),
    )
}

/// The effective shape of the samples an image's filter chain returned.
struct ImageShape<'a> {
    width: u32,
    height: u32,
    bits_per_component: u8,
    colour: ColourSpace,
    /// One affine range per component, from the `/Decode` array.
    decode: Vec<(f32, f32)>,
    /// Per-pixel opacity (JPEG 2000 only), composited over white.
    alpha: Option<&'a [u8]>,
}

/// The `/Decode` ranges of an image, one per component; a component without one
/// uses the default `(0, 1)`, a pure rescale onto the byte range.
fn decode_ranges(dict: &Dict<'_>, components: u8) -> Vec<(f32, f32)> {
    let declared: Vec<(f32, f32)> = dict
        .get::<Array<'_>>(b"Decode")
        .or_else(|| dict.get::<Array<'_>>(b"D"))
        .map(|decode| decode.iter::<(f32, f32)>().collect())
        .unwrap_or_default();
    (0..components)
        .map(|component| {
            declared
                .get(usize::from(component))
                .copied()
                .unwrap_or((0.0, 1.0))
        })
        .collect()
}

/// Convert decoded samples into an RGB raster, built directly at the bounded
/// target size (see [`scaled_dimensions`]): a target pixel samples the source
/// position scaled onto it, so the allocation is bounded for any declared
/// geometry and no image is dropped for being large. Samples are packed
/// big-endian at the shape's bit depth, with every row padded to a byte boundary
/// — the layout each filter in the chain emits — and a short stream is padded
/// with zeroes rather than rejected.
fn samples_to_raster(shape: &ImageShape<'_>, data: &[u8]) -> Option<RgbImage> {
    let (width, height) = scaled_dimensions(shape.width, shape.height);
    let components = usize::from(shape.colour.components());
    let mut raster = Vec::new();
    for y in 0..height {
        let sy = scaled_index(y, shape.height, height);
        for x in 0..width {
            let sx = scaled_index(x, shape.width, width);
            let mut samples = [0u16; 4];
            for (component, sample) in samples.iter_mut().take(components).enumerate() {
                *sample = source_sample(shape, data, sx, sy, component);
            }
            // The alpha channel is indexed by the *source* pixel, which is `sy`
            // and `sx`, not the target position.
            let pixel = pixel_rgb(shape, &samples, sy * shape.width as usize + sx);
            raster.extend_from_slice(&pixel);
        }
    }
    RgbImage::from_raw(width, height, raster)
}

/// The source index one target index samples — `target * source_len /
/// target_len` — so that a target with the source's own length (every image
/// within the raster bound) maps one-to-one.
fn scaled_index(target: u32, source_len: u32, target_len: u32) -> usize {
    target as usize * source_len as usize / target_len as usize
}

/// One sample of the packed source samples at source position (`sx`, `sy`),
/// component `component`: random access, because one target row can sample any
/// source row. A row starts on a byte boundary — its last sample leaves the
/// padding the packing adds — and a position past the end of `data` reads as
/// zero, so a stream shorter than its own header claims still converts.
fn source_sample(
    shape: &ImageShape<'_>,
    data: &[u8],
    sx: usize,
    sy: usize,
    component: usize,
) -> u16 {
    let bits = usize::from(shape.bits_per_component);
    let components = usize::from(shape.colour.components());
    let row_bits = (shape.width as usize * components * bits).div_ceil(8) * 8;
    let bit = sy * row_bits + (sx * components + component) * bits;
    let byte = |index: usize| data.get(index).copied().unwrap_or(0);
    if bits == 16 {
        // A 16-bit sample is two big-endian bytes.
        return u16::from(byte(bit / 8)) << 8 | u16::from(byte(bit / 8 + 1));
    }
    // A sample is never split across two bytes, so it ends at a known distance
    // from the end of its byte.
    let shift = 8 - bits - bit % 8;
    (u16::from(byte(bit / 8)) >> shift) & ((1 << bits) - 1)
}

/// The 8-bit RGB value of the pixel whose `components` samples were just read,
/// composited over white when the image has an alpha channel. `source_pixel` is
/// the pixel's linear position in the source image (`sy * width + sx`), which is
/// where its alpha byte is — never a position in the target raster.
fn pixel_rgb(shape: &ImageShape<'_>, samples: &[u16], source_pixel: usize) -> [u8; 3] {
    // A `/Decode` entry maps the normalised sample through an affine range,
    // which for almost every image is the default `(0, 1)`.
    let channel = |component: usize| {
        let range = shape.decode.get(component).copied().unwrap_or((0.0, 1.0));
        scale_sample(samples[component], shape.bits_per_component, range)
    };
    let rgb = match &shape.colour {
        // An indexed sample is the palette index itself: this module applies no
        // `/Decode` range to it.
        ColourSpace::Indexed(palette) => palette
            .get(usize::from(samples[0]))
            .copied()
            .unwrap_or([0, 0, 0]),
        ColourSpace::Gray => [channel(0); 3],
        ColourSpace::Rgb => [channel(0), channel(1), channel(2)],
        // Ink is what a channel subtracts from white, the key channel from all
        // three.
        ColourSpace::Cmyk => {
            let key = channel(3);
            [
                subtractive(channel(0), key),
                subtractive(channel(1), key),
                subtractive(channel(2), key),
            ]
        }
    };
    match shape.alpha {
        // JPEG has no alpha channel: a translucent pixel is composited over the
        // white it would be painted on.
        Some(alpha) => over_white(rgb, alpha.get(source_pixel).copied().unwrap_or(u8::MAX)),
        None => rgb,
    }
}

/// One ink channel of a CMYK pixel: the ink and the key ink are what the channel
/// takes away from white.
fn subtractive(ink: u8, key: u8) -> u8 {
    255 - ink.saturating_add(key)
}

/// Composite one opaque RGB pixel over white with `alpha` of 255 meaning fully
/// opaque.
fn over_white(rgb: [u8; 3], alpha: u8) -> [u8; 3] {
    rgb.map(|channel| {
        let alpha = u32::from(alpha);
        let blended = (u32::from(channel) * alpha + 255 * (255 - alpha) + 127) / 255;
        u8::try_from(blended).unwrap_or(u8::MAX)
    })
}

/// One sample as a full 8-bit channel: the `/Decode` range is applied to the
/// sample normalised to `0..=1`, so the default range is a rescale from the bit
/// depth onto the byte range.
fn scale_sample(sample: u16, bits_per_component: u8, (min, max): (f32, f32)) -> u8 {
    let full = f32::from(u16::try_from((1u32 << bits_per_component) - 1).unwrap_or(u16::MAX));
    let decoded = min + (max - min) * (f32::from(sample) / full);
    to_byte(decoded)
}

/// A normalised channel value as a byte, clamped: a `/Decode` range can map
/// outside `0..=1`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to the byte range first"
)]
fn to_byte(value: f32) -> u8 {
    (value * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// The pixel size an image is written at: the long side bounded by
/// [`RASTER_LONG_SIDE_PX`], exactly like a rasterized page.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "pixel dimensions are far below the f32 integer range and the scaled sides are clamped to at least one pixel"
)]
fn scaled_dimensions(width: u32, height: u32) -> (u32, u32) {
    let long_side = width.max(height) as f32;
    if long_side <= RASTER_LONG_SIDE_PX {
        return (width, height);
    }
    let scale = RASTER_LONG_SIDE_PX / long_side;
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

/// JPEG-encode a raster at the same quality as a rasterized page.
fn encode_jpeg(image: &RgbImage) -> Result<Vec<u8>, ()> {
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, RASTER_JPEG_QUALITY)
        .encode_image(image)
        .map_err(|_| ())?;
    Ok(jpeg)
}

/// Rasterize `page` (1-based `page_number`) to `<out_dir>/page_<n>.jpg` and push
/// the path onto `images`. A page that cannot be rendered or encoded is skipped.
fn rasterize_page(page: &Page<'_>, page_number: usize, out_dir: &Path, images: &mut Vec<PathBuf>) {
    // `render_dimensions` clamps zero-area pages, so `long_side >= 1.0` and the
    // scale stays finite; the cap keeps a tiny page from being blown up.
    let (width, height) = page.render_dimensions();
    let long_side = width.max(height);
    let scale = (RASTER_LONG_SIDE_PX / long_side).min(RASTER_MAX_SCALE);
    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: WHITE,
        ..RenderSettings::default()
    };
    let pixmap = hayro::render(
        page,
        &RenderCache::new(),
        &InterpreterSettings::default(),
        &settings,
    );
    let (pixel_width, pixel_height) = (pixmap.width(), pixmap.height());
    let raw: Vec<u8> = pixmap
        .take_unpremultiplied()
        .into_iter()
        .flat_map(Rgba8::to_u8_array)
        .collect();
    let Some(image) = RgbaImage::from_raw(u32::from(pixel_width), u32::from(pixel_height), raw)
    else {
        tracing::warn!(
            page = page_number,
            "document: rasterized page has no pixels"
        );
        return;
    };
    let path = out_dir.join(format!("page_{page_number}.jpg"));
    match std::fs::File::create(&path) {
        Ok(file) => {
            let mut encoder = JpegEncoder::new_with_quality(file, RASTER_JPEG_QUALITY);
            match encoder.encode_image(&image) {
                Ok(()) => images.push(path),
                Err(e) => {
                    tracing::warn!(page = page_number, error = %e, "document: JPEG encoding failed for rasterized page");
                }
            }
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "document: failed to create rasterized page file");
        }
    }
}

/// Extract the readable body of a `word/document.xml` part: the content of
/// every `w:t` run-text element, a tab at `w:tab`, a newline at `w:br`, and a
/// newline at each `w:p` paragraph end.
///
/// Elements are matched on their local name (after any namespace prefix) so a
/// producer using a prefix other than `w:` still parses. `None` on any XML
/// error — a body that cannot be read as XML is reported as corrupt.
fn docx_body_text(xml: &[u8]) -> Option<String> {
    let mut reader = Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut text = String::new();
    let mut in_run_text = false;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                b"t" => in_run_text = true,
                b"tab" => text.push('\t'),
                b"br" => text.push('\n'),
                _ => {}
            },
            Ok(Event::Empty(event)) => match event.local_name().as_ref() {
                b"tab" => text.push('\t'),
                b"br" => text.push('\n'),
                _ => {}
            },
            Ok(Event::Text(event)) if in_run_text => {
                if let Ok(chunk) = event.xml10_content() {
                    text.push_str(&chunk);
                }
            }
            Ok(Event::GeneralRef(event)) if in_run_text => append_entity(&mut text, &event),
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                b"t" => in_run_text = false,
                b"p" => text.push('\n'),
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        buffer.clear();
    }
    // Every `w:p` leaves a trailing break; drop the document's final one so the
    // inlined text does not end with a blank line.
    Some(text.trim_end().to_string())
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

/// Read a named ZIP entry into memory, bounded by [`MAX_ZIP_ENTRY_BYTES`].
/// `None` when it is absent, unreadable, or over the bound; none of those is
/// fatal — the caller skips the entry (noting why) or reports the package
/// unreadable.
fn read_zip_entry<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Option<Vec<u8>> {
    let mut entry = archive.by_name(name).ok()?;
    // The declared size is checked first, so a bomb that admits its size is
    // rejected before any inflation; the read then goes through the same cap so
    // a lying header cannot exceed it either.
    if entry.size() > MAX_ZIP_ENTRY_BYTES {
        tracing::warn!(
            %name,
            declared_bytes = entry.size(),
            "document: ZIP entry exceeds the decompression bound"
        );
        return None;
    }
    let mut bytes = Vec::new();
    // Read one byte past the bound so an entry that is exactly at the limit is
    // still accepted whole while a larger one is detected instead of silently
    // truncated.
    entry
        .by_ref()
        .take(MAX_ZIP_ENTRY_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_ZIP_ENTRY_BYTES {
        tracing::warn!(%name, "document: ZIP entry exceeds the decompression bound");
        return None;
    }
    Some(bytes)
}

/// Create `out_dir` when missing. A failure is logged here and surfaces again
/// as an individual artifact-write failure, which is already handled.
fn ensure_out_dir(out_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        tracing::warn!(path = %out_dir.display(), error = %e, "document: failed to create extraction output dir");
    }
}

/// A page's text layer counts only when trimming leaves at least
/// [`MIN_PAGE_TEXT_CHARS`] characters.
fn is_usable_page_text(text: &str) -> bool {
    text.trim().chars().count() >= MIN_PAGE_TEXT_CHARS
}

/// Whether `bytes` are text-like with no container magic: valid UTF-8 and no
/// NUL byte.
fn is_plain_utf8(bytes: &[u8]) -> bool {
    !bytes.contains(&0) && std::str::from_utf8(bytes).is_ok()
}

/// Builders shared by this module's tests and the read tool's document tests
/// ([`crate::tools::read_document`]), so both sides convert the very same
/// fixtures.
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

    /// Assemble numbered objects into a PDF with a valid xref table, computing
    /// offsets as it writes.
    pub(crate) fn assemble_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
            pdf.extend_from_slice(object);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let start_xref = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets {
            pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::*;
    use super::*;
    use std::io::Write;

    /// One embedded image XObject for [`pdf_fixture`]: dictionary entries are
    /// written verbatim so a test can pick the filter chain and the colour space,
    /// and the stream bytes are raw so real Flate streams can be built.
    struct ImageXObject {
        width: u32,
        height: u32,
        colour_space: &'static str,
        /// The single `/Filter`, omitted entirely when the samples are raw.
        filter: Option<&'static str>,
        decode_parms: Option<&'static str>,
        data: Vec<u8>,
    }

    impl ImageXObject {
        /// An 8-bit RGB image whose samples are `data`, already encoded by
        /// `filter`.
        fn rgb(width: u32, height: u32, filter: Option<&'static str>, data: Vec<u8>) -> Self {
            Self {
                width,
                height,
                colour_space: "/DeviceRGB",
                filter,
                decode_parms: None,
                data,
            }
        }

        fn decode_parms(mut self, decode_parms: &'static str) -> Self {
            self.decode_parms = Some(decode_parms);
            self
        }

        /// The object body: dictionary plus stream.
        fn body(&self) -> Vec<u8> {
            let filter = self
                .filter
                .map(|filter| format!(" /Filter /{filter}"))
                .unwrap_or_default();
            let decode_parms = self
                .decode_parms
                .map(|parms| format!(" /DecodeParms << {parms} >>"))
                .unwrap_or_default();
            let mut body = format!(
                "<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace {} \
                 /BitsPerComponent 8{filter}{decode_parms} /Length {} >>\nstream\n",
                self.width,
                self.height,
                self.colour_space,
                self.data.len()
            )
            .into_bytes();
            body.extend_from_slice(&self.data);
            body.extend_from_slice(b"\nendstream");
            body
        }
    }

    /// A zlib stream of `data`, as a `/FlateDecode` stream carries it.
    fn flate(data: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).expect("write flate data");
        encoder.finish().expect("finish flate stream")
    }

    /// Build a one-page PDF from `content_stream` and `images` (one XObject per
    /// entry, referenced by the page's resources as `/Im0`, `/Im1`, …).
    fn pdf_fixture(content_stream: &[u8], images: &[ImageXObject]) -> Vec<u8> {
        // 1 catalog, 2 page tree, 3 font, 4 page, one object per image, then the
        // content stream.
        let contents_id = 5 + images.len();
        let xobjects = if images.is_empty() {
            String::new()
        } else {
            let refs: Vec<String> = (0..images.len())
                .map(|slot| format!(" /Im{slot} {} 0 R", 5 + slot))
                .collect();
            format!(" /XObject <<{} >>", refs.concat())
        };
        let mut objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources \
                 << /Font << /F1 3 0 R >>{xobjects} >> /Contents {contents_id} 0 R >>"
            )
            .into_bytes(),
        ];
        objects.extend(images.iter().map(ImageXObject::body));
        objects.push(
            format!(
                "<< /Length {} >>\nstream\n{}\nendstream",
                content_stream.len(),
                String::from_utf8_lossy(content_stream)
            )
            .into_bytes(),
        );
        assemble_pdf(&objects)
    }

    /// A one-page PDF with a Helvetica text layer.
    fn text_pdf(images: &[ImageXObject]) -> Vec<u8> {
        pdf_fixture(
            b"BT /F1 24 Tf 72 700 Td (Hello PDF text layer) Tj ET",
            images,
        )
    }

    /// The colour of the uniform 4x2 image the embedded-image tests use.
    const SOLID_RGB: [u8; 3] = [200, 40, 40];

    /// That image's raw samples, in row order.
    fn solid_rgb_samples() -> Vec<u8> {
        SOLID_RGB.repeat(8)
    }

    /// An inline image (`BI`/`ID`/`EI`) painting the solid 4x2 RGB samples,
    /// ASCIIHex-encoded so the assembled content stream stays printable.
    fn solid_rgb_inline_image() -> String {
        let hex = crate::util::hex_string(&solid_rgb_samples());
        format!("BI /W 4 /H 2 /CS /RGB /BPC 8 /F /AHx ID {hex}> EI")
    }

    /// A one-page PDF with a painted rectangle and no text layer: the page has
    /// to be rasterized.
    fn image_only_pdf() -> Vec<u8> {
        let image = ImageXObject::rgb(4, 2, Some("FlateDecode"), flate(&solid_rgb_samples()));
        pdf_fixture(b"0.1 0.5 0.9 rg 0 0 612 792 re f", &[image])
    }

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
        let outcome = convert_document(&bytes, "report.docx", dir.path());
        let DocOutcome::Text {
            text,
            images,
            notes,
        } = outcome
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
        let DocOutcome::Text { images, notes, .. } =
            convert_document(&bytes, "report.docx", dir.path())
        else {
            panic!("expected Text outcome for a well-formed docx");
        };
        assert!(images.is_empty(), "an undecodable entry yields no image");
        assert_eq!(
            notes,
            ["skipped 2 embedded image(s) in a format this pipeline cannot convert"]
        );
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
        let DocOutcome::Text { images, .. } = convert_document(&bytes, "report.docx", dir.path())
        else {
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
        let DocOutcome::Text { images, .. } = convert_document(&bytes, "report.docx", dir.path())
        else {
            panic!("expected Text outcome for a well-formed docx");
        };
        assert_eq!(
            images,
            vec![dir.path().join("pic.png"), dir.path().join("pic_2.png")]
        );
    }

    #[test]
    fn encrypted_docx_reports_password_protected() {
        let mut bytes = CFB_MAGIC.to_vec();
        bytes.extend_from_slice(b"OLE container body that is not a zip");
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(matches!(
            convert_document(&bytes, "secret.docx", dir.path()),
            DocOutcome::Unreadable { reason } if reason == "password-protected"
        ));
        // The same container under a legacy Word name is just a format we do not
        // convert — CFB is not by itself evidence of encryption.
        assert!(matches!(
            convert_document(&bytes, "old.doc", dir.path()),
            DocOutcome::Unsupported
        ));
    }

    #[test]
    fn garbage_pdf_is_reported_unreadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(
            b"%PDF-1.7\nthis is not a PDF body at all",
            "x.pdf",
            dir.path(),
        );
        assert!(
            matches!(outcome, DocOutcome::Unreadable { reason } if reason == "could not be parsed")
        );
    }

    #[test]
    fn markdown_and_extensionless_utf8_are_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(matches!(
            convert_document(b"# Title\n\nBody", "notes.md", dir.path()),
            DocOutcome::Text { text, .. } if text == "# Title\n\nBody"
        ));
        assert!(matches!(
            convert_document(b"plain words", "README", dir.path()),
            DocOutcome::Text { text, .. } if text == "plain words"
        ));
    }

    #[test]
    fn zip_with_non_docx_extension_is_unsupported() {
        let bytes = zip_fixture(&[("xl/workbook.xml", b"<workbook/>")]);
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(matches!(
            convert_document(&bytes, "book.xlsx", dir.path()),
            DocOutcome::Unsupported
        ));
    }

    #[test]
    fn pdf_with_text_layer_extracts_text_without_rasterizing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[]), "hello.pdf", dir.path());
        let DocOutcome::Text { text, images, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(
            text.contains("Hello PDF text layer"),
            "unexpected extracted text: {text:?}"
        );
        assert!(images.is_empty(), "text pages must not be rasterized");
    }

    /// The source colour, within JPEG's own rounding.
    fn assert_solid_rgb(actual: [u8; 3]) {
        for (channel, expected) in actual.into_iter().zip(SOLID_RGB) {
            assert!(
                channel.abs_diff(expected) <= 20,
                "expected {SOLID_RGB:?}, got {actual:?}"
            );
        }
    }

    /// Convert a text-layer page carrying `image` and decode the one image file
    /// the page wrote.
    fn decode_written_embedded_image(image: ImageXObject) -> image::RgbImage {
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[image]), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        assert_eq!(images, vec![dir.path().join("page_1_img_0.jpg")]);
        image::open(&images[0])
            .expect("decode the written image")
            .to_rgb8()
    }

    /// A text-layer page is never rasterized, so its embedded images have to be
    /// decoded and written: without that pass a document reaches the model
    /// without its figures.
    #[test]
    fn pdf_text_page_writes_embedded_flate_image() {
        let written = decode_written_embedded_image(ImageXObject::rgb(
            4,
            2,
            Some("FlateDecode"),
            flate(&solid_rgb_samples()),
        ));
        assert_eq!(written.dimensions(), (4, 2));
        assert_solid_rgb(written.get_pixel(0, 0).0);
    }

    /// A form XObject is a container the page's content paints through, so the
    /// images in its own resources are the page's images — a rasterized page
    /// shows them, and a text page has to extract them.
    #[test]
    fn pdf_text_page_writes_image_nested_in_a_form() {
        let content = "BT /F1 24 Tf 72 700 Td (Hello PDF text layer) Tj ET";
        let form = "/Im0 Do";
        let objects = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources \
              << /Font << /F1 3 0 R >> /XObject << /Fm0 5 0 R >> >> /Contents 7 0 R >>"
                .to_vec(),
            format!(
                "<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources \
                 << /XObject << /Im0 6 0 R >> >> /Length {} >>\nstream\n{form}\nendstream",
                form.len()
            )
            .into_bytes(),
            ImageXObject::rgb(4, 2, Some("FlateDecode"), flate(&solid_rgb_samples())).body(),
            format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            )
            .into_bytes(),
        ];

        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&assemble_pdf(&objects), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        assert_eq!(images, vec![dir.path().join("page_1_img_0.jpg")]);
    }

    /// An inline image lives in the content stream, not in the resources, so a
    /// text page — which is never rasterized — needs this pass to show it.
    #[test]
    fn pdf_text_page_writes_inline_image() {
        let content = format!(
            "BT /F1 24 Tf 72 700 Td (Hello PDF text layer) Tj ET {}",
            solid_rgb_inline_image()
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(
            &pdf_fixture(content.as_bytes(), &[]),
            "scan.pdf",
            dir.path(),
        );
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        assert_eq!(images, vec![dir.path().join("page_1_img_0.jpg")]);
        let written = image::open(&images[0])
            .expect("decode the written image")
            .to_rgb8();
        assert_eq!(written.dimensions(), (4, 2));
        assert_solid_rgb(written.get_pixel(0, 0).0);
    }

    /// A form XObject is a content stream an inline image can sit in just as
    /// well, so the inline pass walks the forms the page paints through.
    #[test]
    fn pdf_text_page_writes_inline_image_nested_in_a_form() {
        let content = "BT /F1 24 Tf 72 700 Td (Hello PDF text layer) Tj ET";
        let form = solid_rgb_inline_image();
        let objects = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources \
              << /Font << /F1 3 0 R >> /XObject << /Fm0 5 0 R >> >> /Contents 6 0 R >>"
                .to_vec(),
            format!(
                "<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Length {} >>\nstream\n{form}\nendstream",
                form.len()
            )
            .into_bytes(),
            format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            )
            .into_bytes(),
        ];

        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&assemble_pdf(&objects), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        assert_eq!(images, vec![dir.path().join("page_1_img_0.jpg")]);
    }

    /// A PNG predictor in `/DecodeParms` is what most producers emit, and the
    /// filter chain has to undo it before the samples can be converted.
    #[test]
    fn pdf_text_page_writes_png_predictor_image() {
        // A 4x2 image in PNG "Up" rows: the first row is subtracted from a zero
        // row, so it is raw, and the second row's differences from it are zero.
        let mut encoded = vec![2];
        encoded.extend_from_slice(&SOLID_RGB.repeat(4));
        encoded.push(2);
        encoded.extend_from_slice(&[0; 12]);
        let written = decode_written_embedded_image(
            ImageXObject::rgb(4, 2, Some("FlateDecode"), flate(&encoded))
                .decode_parms("/Predictor 12 /Colors 3 /Columns 4 /BitsPerComponent 8"),
        );
        assert_eq!(written.dimensions(), (4, 2));
        assert_solid_rgb(written.get_pixel(0, 0).0);
    }

    /// An embedded image that cannot be decoded is skipped, and a text-layer
    /// page is never rasterized — so without a note it would vanish silently.
    #[test]
    fn pdf_notes_embedded_images_that_cannot_be_decoded() {
        // JPEG 2000 bytes this module's decoder rejects outright.
        let image = ImageXObject::rgb(2, 2, Some("JPXDecode"), b"RGBRGB12".to_vec());
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[image]), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(
            images.is_empty(),
            "an undecodable embedded image yields no file"
        );
        assert_eq!(
            notes,
            ["skipped 1 embedded image(s) that could not be extracted"]
        );
    }

    /// A corrupt Flate stream is one hayro's inflate reports leniently, as an
    /// empty result rather than an error: the image must be reported like any
    /// other undecodable one, not written as a blank page image.
    #[test]
    fn pdf_notes_a_leniently_empty_decode() {
        let image = ImageXObject::rgb(4, 2, Some("FlateDecode"), b"not a flate stream".to_vec());
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[image]), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(
            images.is_empty(),
            "a decode that returned no samples yields no file"
        );
        assert_eq!(
            notes,
            ["skipped 1 embedded image(s) that could not be extracted"]
        );
    }

    /// The guard's own envelope: a declared geometry over it is refused before
    /// any sample is decoded, so the note must say that rather than claim a
    /// decode failure.
    #[test]
    fn pdf_notes_a_declared_geometry_over_the_raster_envelope() {
        let image = ImageXObject::rgb(20_000, 20_000, None, Vec::new());
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[image]), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(
            images.is_empty(),
            "an image over the envelope yields no file"
        );
        assert_eq!(
            notes,
            ["skipped 1 embedded image(s) over the image size limit"]
        );
    }

    /// A lone `DCTDecode` stream already is a complete JPEG: it is written
    /// through verbatim rather than decoded and re-encoded.
    #[test]
    fn pdf_text_page_passes_jpeg_through_verbatim() {
        let source = RgbImage::from_pixel(4, 2, image::Rgb(SOLID_RGB));
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, RASTER_JPEG_QUALITY)
            .encode_image(&source)
            .expect("encode the source JPEG");
        let image = ImageXObject::rgb(4, 2, Some("DCTDecode"), jpeg.clone());
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[image]), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        assert_eq!(images, vec![dir.path().join("page_1_img_0.jpg")]);
        assert_eq!(
            std::fs::read(&images[0]).expect("read the written image"),
            jpeg
        );
    }

    /// An `/Indexed` image's samples are palette indices, not gray levels.
    #[test]
    fn pdf_indexed_image_uses_its_palette() {
        let image = ImageXObject {
            width: 16,
            height: 4,
            colour_space: "[/Indexed /DeviceRGB 3 <FF000000FF000000FF000000FF>]",
            filter: None,
            decode_parms: None,
            // Four columns each of a palette entry: red, green, blue, white.
            data: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].repeat(4),
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&text_pdf(&[image]), "scan.pdf", dir.path());
        let DocOutcome::Text { images, notes, .. } = outcome else {
            panic!("expected Text outcome for a text-layer PDF");
        };
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        assert_eq!(images, vec![dir.path().join("page_1_img_0.jpg")]);
        let written = image::open(&images[0])
            .expect("decode the written image")
            .to_rgb8();
        assert_eq!(written.dimensions(), (16, 4));
        let [red, green, blue] = written.get_pixel(5, 0).0;
        assert!(
            green > red && green > blue,
            "the second palette entry is green, got {red},{green},{blue}"
        );
    }

    /// The only coverage of the rasterization path (render → unpremultiply →
    /// JPEG), including the scale that bounds the rendered page's long side. A
    /// page without a text layer is rendered whole, so its embedded image comes
    /// along in the raster instead of as a file of its own.
    #[test]
    fn pdf_page_without_text_layer_is_rasterized() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = convert_document(&image_only_pdf(), "scan.pdf", dir.path());
        let DocOutcome::Text {
            text,
            images,
            notes,
        } = outcome
        else {
            panic!("expected Text outcome for an image-only PDF");
        };
        assert!(text.trim().is_empty(), "no text layer to extract: {text:?}");
        assert_eq!(images, vec![dir.path().join("page_1.jpg")]);
        assert!(notes.is_empty(), "nothing was skipped: {notes:?}");
        let page = image::open(&images[0]).expect("decode the rasterized page");
        // The 792 px letter page is scaled onto the RASTER_LONG_SIDE_PX target,
        // short of it by at most the pixel-grid rounding.
        let long_side = f64::from(page.width().max(page.height()));
        let target = f64::from(RASTER_LONG_SIDE_PX);
        assert!(
            long_side <= target && long_side > target - 5.0,
            "long side {long_side} must be the RASTER_LONG_SIDE_PX target"
        );
    }
}
