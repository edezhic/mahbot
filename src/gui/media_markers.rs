//! Pre-processing for media markers in chat content.
//!
//! A body is split into the two blocks a bubble draws — its attachments and
//! its text — because the markdown parser cannot keep them apart: iced's
//! parser drains every accumulated inline span into an `Image` item's `alt`
//! (`TagEnd::Image` in `iced_widget`'s markdown), so a picture drawn *inside*
//! a paragraph swallows the text beside it. Extracting at render time means
//! already-stored chat history and transcripts render correctly too; nothing
//! about the stored content changes.
//!
//! The text that reaches `markdown::parse()` therefore never carries a
//! picture, and the stored body is never rewritten. What a marker becomes:
//!
//! - `[IMAGE:path]` → an attachment: a `data:` URI, an absolute path or a
//!   valid remote address draws; an undecodable or oversized payload degrades
//!   to the `🖼️ image` label (never raw base64 text, never a blank)
//! - `![alt](url)` → the same attachment; inside code — a fenced or indented
//!   block, an inline span — the source stays visible as written
//! - `[AUDIO:path]` → 🎵 filename, `[VIDEO:path]` → 🎬 Video: filename,
//!   `[FILE:path]` → 📄 filename (an attachment too; its click reveals the
//!   file in the OS file manager)
//! - `[Audio transcription of ...]: text` → 🔊 emoji + transcribed text
//! - `[Video transcription of ...]: text` → 🎬 emoji + transcribed text
//! - `[Saved image: path]` → stripped (the image renders separately from its
//!   data-URI marker, so the upload annotation is pure display noise)
//! - `[Saved video: path]` → `🎬 Video: filename` (there is no separate video
//!   renderer, so the clip path must stay visible rather than be hidden)
//!
//! # Marker resolution and refusals
//!
//! Targets are placed by [`media_target::resolve_marker_target`], the authority
//! Telegram delivery uses too; one that names something but cannot be shown
//! becomes a [`media_target::marker_refusal`] notice, while content the owner
//! typed keeps its structural display ([`MarkerContext::refusals`]).
//!
//! # Canonical marker pattern
//!
//! The `[KIND:path]` format is defined by the `MediaMarkerKind` enum in
//! `src/util/mod.rs`, from which [`MEDIA_MARKER_RE`] is generated — **the
//! single source of truth** for all marker kinds (`IMAGE`, `AUDIO`, `VIDEO`,
//! `FILE`, and any future additions), in any casing. This module uses the
//! shared [`MEDIA_MARKER_RE`] and [`parse_media_marker`] helper to stay in sync —
//! adding a new marker kind there automatically propagates to this module,
//! `enrichment.rs`, `agent.rs`, and `compatible.rs` without needing per-kind
//! regexes anywhere.
//!
//! # Clickable file markers
//!
//! A file attachment is a label row, not a link: it carries no colour and no
//! underline, and only a click reveals its target. The click travels as a
//! `mahbot-file:` URI ([`file_link_target`] takes it apart,
//! [`reveal_file_marker`] opens it), so both surfaces link a row the same way
//! they link a URL.

use iced::widget::{Column, image, markdown, text};
use iced::{ContentFit, Element, Length};
use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::SystemTime;

use super::{selectable, theme, widgets};
use crate::gui::markdown_breaks::iced_markdown_options;
use crate::util::media_target::{self, MarkerTarget};
use crate::util::{
    MEDIA_MARKER_RE, MediaMarkerKind, UnwrapPoison, file_name_or_path, parse_media_marker,
};

/// The `image` crate, aliased because `image` here is iced's widget module.
use ::image as image_crate;

/// What marker resolution needs from the content's surroundings.
pub(crate) struct MarkerContext<'a> {
    /// The workspace the message belongs to: a relative marker target names a
    /// file here. `None` when the surface knows no workspace. A desktop surface
    /// has exactly one root, so relayed content naming a file in its originating
    /// workspace can only resolve that far on Telegram, which carries both.
    pub workspace: Option<&'a Path>,
    /// Whether an undeliverable marker becomes a refusal reason. Only a reply
    /// the product delivers gets one; verbatim content — what the owner typed, a
    /// tool result, reasoning promoted into a narration slot — keeps the
    /// structural display it has always had.
    pub refusals: bool,
}

/// One attachment a rendered body carries, in body order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaAttachment {
    /// The picture to draw (a `data:` URI, an absolute path, or a remote
    /// address); `None` for a named file (audio, video, document) and for a
    /// picture whose payload the content no longer carries.
    pub image: Option<String>,
    /// How a plain-text rendition names it — `🖼️ image`, `🖼️ poster.png`,
    /// `🎵 song.mp3`, `🎬 Video: clip.mp4`, `📄 report.pdf`.
    pub label: String,
    /// What a `[FILE:…]` click reveals; `None` for anything else, and for a
    /// marker that names a payload rather than a path.
    pub reveal: Option<String>,
}

/// A rendered body: the text to parse and the attachments it carried.
pub(crate) struct Preprocessed {
    pub text: String,
    pub attachments: Vec<MediaAttachment>,
}

impl Preprocessed {
    /// The body as plain text: one line per attachment (its label), then the
    /// text — the order it is drawn in. Used by the Sessions collapse
    /// measurement so the preview agrees with the expanded transcript.
    #[must_use]
    pub fn display_text(&self) -> String {
        let mut out = self
            .attachments
            .iter()
            .map(|attachment| attachment.label.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if !self.text.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&self.text);
        }
        out
    }
}

/// Pre-process a content string, splitting it into the text to parse and the
/// attachments the text no longer carries.
///
/// One ordered pass over the body: the attachments come out in the order it
/// carried them, whichever source — a `[KIND:…]` marker or a markdown picture
/// — wrote each one.
pub(crate) fn preprocess(content: &str, markers: &MarkerContext<'_>) -> Preprocessed {
    // Order matters: the transcription annotations contain the words "Audio"
    // and "Video", which overlap with the raw `[AUDIO:...]`/`[VIDEO:...]`
    // patterns.  Handle them first.
    let s = replace_saved_annotations(&replace_transcription(content));
    // Fast path: nothing to extract, so the body reaches the parser as it is.
    if !MEDIA_MARKER_RE.is_match(&s) && !s.contains("![") {
        return Preprocessed {
            text: s,
            attachments: Vec::new(),
        };
    }
    let (code, mut edits) = scan(&s);
    edits.extend(marker_edits(&s, markers, &code));
    edits.sort_by_key(|edit| edit.range.start);
    apply_edits(&s, edits)
}

/// `[Audio transcription of {filename}]: {text}` → 🔊, `[Video transcription
/// of {filename}]: {text}` → 🎬 — the emoji matches the annotation kind. The
/// audio form is a legacy annotation from before the icon-combo switch (only
/// rows already persisted in chat history / sessions carry it; new messages
/// store the icon combo `🔊✍️` + transcription directly); the video form is
/// produced by `enrich_message` for inbound VIDEO markers.
/// Only the annotation header is replaced — the (often multi-line) description
/// and any following message content are left untouched.
fn replace_transcription(s: &str) -> String {
    static RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\[(Audio|Video) transcription of [^\]]+\]:\s*")
            .expect("transcription regex must compile")
    });
    RE.replace_all(s, |caps: &regex::Captures| {
        match caps.get(1).map(|m| m.as_str()) {
            Some("Audio") => "🔊 ",
            _ => "🎬 ",
        }
    })
    .to_string()
}

/// The display form of one marker: the replacement it becomes, or `None` when it
/// is not a reference and stays as the author wrote it.
fn render_marker(
    kind: MediaMarkerKind,
    target: &str,
    markers: &MarkerContext<'_>,
    roots: &[PathBuf],
) -> Option<Replacement> {
    // Verbatim content — what the owner typed, a tool result, promoted
    // reasoning — keeps the structural display: never resolved, never refused.
    if !markers.refusals {
        return render_own_text(kind, target);
    }
    match media_target::resolve_marker_target(target, roots) {
        MarkerTarget::NotAReference => None,
        MarkerTarget::Inline => match kind {
            MediaMarkerKind::Image => {
                Some(Replacement::Attachment(picture_attachment(target, target)))
            }
            MediaMarkerKind::Audio | MediaMarkerKind::Video | MediaMarkerKind::File => {
                Some(Replacement::Text(media_target::inline_marker_refusal(kind)))
            }
        },
        MarkerTarget::Remote => match media_target::remote_marker(kind, target) {
            media_target::RemoteMarker::Refused(notice) => Some(Replacement::Text(notice)),
            media_target::RemoteMarker::Deliverable => Some(Replacement::Attachment(
                delivered_marker(kind, target, target),
            )),
        },
        MarkerTarget::Local(path) => match media_target::local_refusal(kind, &path) {
            Some(reason) => Some(Replacement::Text(media_target::marker_refusal_for(
                target, reason,
            ))),
            None => Some(Replacement::Attachment(delivered_marker(
                kind,
                target,
                &path.to_string_lossy(),
            ))),
        },
    }
}

/// The attachment a marker whose target is deliverable becomes: `resolved` is
/// what the thing is reached by, while the label keeps naming the target as the
/// message spelled it.
fn delivered_marker(kind: MediaMarkerKind, target: &str, resolved: &str) -> MediaAttachment {
    match kind {
        MediaMarkerKind::Image => picture_attachment(resolved, target),
        MediaMarkerKind::File => label_attachment(kind, target, Some(resolved.to_string())),
        MediaMarkerKind::Audio | MediaMarkerKind::Video => label_attachment(kind, target, None),
    }
}

/// The attachment a marker that draws no picture becomes: its label, plus the
/// path a click reveals — `None` where nothing is revealed (audio and video have
/// no player, and a payload names no path).
fn label_attachment(
    kind: MediaMarkerKind,
    target: &str,
    reveal: Option<String>,
) -> MediaAttachment {
    MediaAttachment {
        image: None,
        label: if kind == MediaMarkerKind::File {
            file_marker_label(target)
        } else {
            media_label(kind, target)
        },
        reveal,
    }
}

/// The display form of a marker in verbatim content (what the owner typed, a
/// tool result, promoted reasoning). The product does not resolve or refuse
/// text it does not own, so this is the structural gate the page has always
/// used: an inline payload, a well-formed address or an absolute path draws;
/// anything else is left as typed.
fn render_own_text(kind: MediaMarkerKind, target: &str) -> Option<Replacement> {
    let attachment = match kind {
        MediaMarkerKind::Image => {
            let draws = target.starts_with("data:")
                || target.starts_with(crate::util::DATA_URI_OMITTED_PREFIX)
                || media_target::is_valid_remote_url(target)
                || Path::new(target).is_absolute();
            return draws.then(|| Replacement::Attachment(picture_attachment(target, target)));
        }
        MediaMarkerKind::Audio | MediaMarkerKind::Video => label_attachment(kind, target, None),
        MediaMarkerKind::File => label_attachment(
            kind,
            target,
            (!target.starts_with("data:")
                && !target.starts_with(crate::util::DATA_URI_OMITTED_PREFIX))
            .then(|| target.to_string()),
        ),
    };
    Some(Replacement::Attachment(attachment))
}

/// The attachment a picture becomes: `image` is what the renderer draws, while
/// the label names it by file name — or generically, because an inline payload
/// has no name and a stripped one has no bytes left to draw.
fn picture_attachment(image: &str, label_target: &str) -> MediaAttachment {
    let payload = image.starts_with("data:");
    let stripped = image.starts_with(crate::util::DATA_URI_OMITTED_PREFIX);
    MediaAttachment {
        image: (!stripped).then(|| image.to_string()),
        label: if payload || stripped {
            "🖼️ image".to_string()
        } else {
            format!("🖼️ {}", file_name_or_path(label_target))
        },
        reveal: None,
    }
}

/// The label a delivered `AUDIO`/`VIDEO` marker shows: neither has an inline
/// player, so the file is named rather than played.
fn media_label(kind: MediaMarkerKind, target: &str) -> String {
    match kind {
        MediaMarkerKind::Audio => format!("🎵 {}", file_name_or_path(target)),
        _ => format!("🎬 Video: {}", file_name_or_path(target)),
    }
}

/// The label a `[FILE:path]` marker shows: `📄 name`. A payload carries no file
/// name (`file_name_or_path` would return a chunk of its base64), so it is named
/// generically instead of dumping the blob into the bubble.
fn file_marker_label(path: &str) -> String {
    let path = path.trim();
    if path.starts_with("data:") || path.starts_with(crate::util::DATA_URI_OMITTED_PREFIX) {
        "📄 file".to_string()
    } else {
        format!("📄 {}", file_name_or_path(path))
    }
}

/// Handle the `[Saved <kind>: path]` upload annotations produced by the channel
/// enrichment layer. These are **not** `[KIND:path]` markers (so the canonical
/// [`MEDIA_MARKER_RE`] leaves them untouched), and they are model-facing only —
/// the GUI strips/rewrites them for display.
///
/// - `[Saved image: path]` → stripped entirely. The image already renders
///   separately from its accompanying `[IMAGE:data:…]` marker, so the upload
///   annotation (with the blank-line separator enrichment inserts before it) is
///   pure display noise.
/// - `[Saved video: path]` → `🎬 Video: filename`. Video enrichment replaces the
///   `[VIDEO:path]` marker in place with the `[Saved video: ...]` annotation and
///   has **no** separate video renderer, so blanket-stripping it would hide the
///   clip. Rewriting it to the same placeholder the raw `[VIDEO:path]` marker
///   produces keeps the clip path visible.
fn replace_saved_annotations(s: &str) -> String {
    // `\n*` eats the blank-line separator enrichment inserts before the
    // trailing image annotation so no stray whitespace remains in the display.
    static SAVED_IMAGE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\n*\[Saved image: [^\]]*\]")
            .expect("saved image annotation regex must compile")
    });
    static SAVED_VIDEO_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\[Saved video: ([^\]]*)\]")
            .expect("saved video annotation regex must compile")
    });

    let s = SAVED_IMAGE_RE.replace_all(s, "").to_string();
    SAVED_VIDEO_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let path = caps.get(1).map_or("", |m| m.as_str());
            format!("🎬 Video: {}", file_name_or_path(path))
        })
        .to_string()
}

// ── Extraction ─────────────────────────────────────────────────

/// One rewrite of the body, in body order.
struct Edit {
    range: Range<usize>,
    replacement: Replacement,
}

/// What one edit becomes.
enum Replacement {
    /// Text the body keeps in place of the range (a refusal notice).
    Text(String),
    /// The attachment the range becomes: it leaves the text and is drawn in the
    /// body's attachment group instead.
    Attachment(MediaAttachment),
}

/// One `into_offset_iter` pass over the body with iced's parser options — the
/// parser the markdown view runs — collecting what both edit sources need: the
/// byte ranges of its code, where marker syntax is source rather than content,
/// and the edits its markdown pictures become, in body order.
///
/// An agent writes `![alt](url)` directly, and the parser would draw it as an
/// `Image` item that swallows the text collected before it, so it leaves the
/// text here too. A picture needs no code check: this same scan is the render's,
/// so it reports an `Image` only where one draws — never inside an inline span
/// or a fenced/indented block, where the source must stay visible.
fn scan(text: &str) -> (Vec<Range<usize>>, Vec<Edit>) {
    let mut code = Vec::new();
    let mut pictures = Vec::new();
    // Where the open code block began, held until its matching end reports the
    // end.
    let mut block: Option<usize> = None;
    // The open picture: where it started and what it names.
    let mut picture: Option<(usize, String)> = None;
    for (event, range) in
        pulldown_cmark::Parser::new_ext(text, iced_markdown_options()).into_offset_iter()
    {
        match event {
            pulldown_cmark::Event::Code(_) => code.push(range),
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::CodeBlock(_)) => {
                block = Some(range.start);
            }
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                if let Some(start) = block.take() {
                    code.push(start..range.end);
                }
            }
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Image { dest_url, .. }) => {
                picture = Some((range.start, dest_url.into_string()));
            }
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Image) => {
                let Some((start, dest)) = picture.take() else {
                    continue;
                };
                // A `<>`-quoted destination arrives without its brackets.
                let dest = dest.trim();
                // A picture with no destination can never draw and would still
                // reach the parser as an `Image` item, so its source goes
                // without an attachment.
                let replacement = if dest.is_empty() {
                    Replacement::Text(String::new())
                } else {
                    Replacement::Attachment(picture_attachment(dest, dest))
                };
                pictures.push(Edit {
                    range: start..range.end,
                    replacement,
                });
            }
            _ => {}
        }
    }
    (code, pictures)
}

/// The edits the body's `[KIND:…]` markers become, in body order, through the
/// canonical [`MEDIA_MARKER_RE`].
///
/// A deliverable marker becomes an attachment and contributes nothing to the
/// text; a refusal notice and a marker left as written keep their place in it.
/// A target that names something but cannot be shown becomes a refusal notice
/// when the context allows one; a bare word of prose is left as written; and a
/// marker inside code is source, so it stays as written too.
fn marker_edits(text: &str, markers: &MarkerContext<'_>, code: &[Range<usize>]) -> Vec<Edit> {
    let roots: Vec<PathBuf> = markers
        .workspace
        .map(Path::to_path_buf)
        .into_iter()
        .collect();
    let mut edits = Vec::new();
    for caps in MEDIA_MARKER_RE.captures_iter(text) {
        let whole = caps.get_match();
        if code.iter().any(|range| range.contains(&whole.start())) {
            continue;
        }
        let (kind, target) = parse_media_marker(&caps);
        let Some(replacement) = render_marker(kind, target.trim(), markers, &roots) else {
            continue;
        };
        edits.push(Edit {
            range: whole.range(),
            replacement,
        });
    }
    edits
}

/// Apply the edits, sorted by where they begin, and collect the attachments in
/// walk order — one pass, so a body mixing marker kinds and markdown pictures
/// keeps its order.
fn apply_edits(text: &str, edits: Vec<Edit>) -> Preprocessed {
    let mut out = String::with_capacity(text.len());
    let mut attachments = Vec::new();
    let mut cursor = 0usize;
    let mut edits = edits.into_iter().peekable();
    while let Some(edit) = edits.next() {
        if edit.range.start < cursor {
            // Inside a range an earlier edit already rewrote.
            continue;
        }
        match edit.replacement {
            Replacement::Text(replacement) => {
                out.push_str(&text[cursor..edit.range.start]);
                out.push_str(&replacement);
                cursor = edit.range.end;
            }
            Replacement::Attachment(attachment) => {
                let next_start = edits.peek().map(|next| next.range.start);
                let removal = removal_range(text, &edit.range, cursor, next_start);
                out.push_str(&text[cursor..removal.start]);
                cursor = join_after_removal(&mut out, text, removal.end);
                attachments.push(attachment);
            }
        }
    }
    out.push_str(&text[cursor..]);
    Preprocessed {
        text: out,
        attachments,
    }
}

/// The range an attachment's removal actually takes out: the picture itself, or
/// — when the picture is all its line holds, a bullet aside — the whole line
/// including its newline. A lone picture's two surrounding line breaks are one
/// line break of the body, and a bullet left behind would dangle over nothing.
/// A line another edit already rewrote (`cursor`) or is about to rewrite
/// (`next_start`) keeps its own text: dropping it would take that edit with it.
fn removal_range(
    text: &str,
    range: &Range<usize>,
    cursor: usize,
    next_start: Option<usize>,
) -> Range<usize> {
    let line_start = text[..range.start]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let line_end = text[range.end..]
        .find('\n')
        .map_or(text.len(), |offset| range.end + offset);
    let line_shared = line_start < cursor || next_start.is_some_and(|start| start < line_end);
    let beside = format!(
        "{}{}",
        &text[line_start..range.start],
        &text[range.end..line_end]
    );
    let alone = !line_shared && (beside.trim().is_empty() || only_list_marker(&beside));
    if alone {
        line_start..(line_end + 1).min(text.len())
    } else {
        range.clone()
    }
}

/// Join the text around a removal: the whitespace run `out` ends with and the
/// one the text continues with are replaced by what they amount to — a blank
/// line for a paragraph gap, one newline for a line break, a single space
/// inside a sentence, nothing at the end of the body. A removal never sits in
/// code, so the run it touches is prose. Returns the cursor, past the run the
/// text continues with.
fn join_after_removal(out: &mut String, text: &str, cursor: usize) -> usize {
    let mut newlines = 0;
    let mut opened = false;
    while let Some(&byte) = out.as_bytes().last() {
        if !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            break;
        }
        newlines += usize::from(byte == b'\n');
        opened = true;
        out.pop();
    }
    let after = &text[cursor..];
    let run = after.len() - after.trim_start_matches([' ', '\t', '\r', '\n']).len();
    newlines += after[..run].matches('\n').count();
    let rest = &after[run..];
    match newlines {
        0 => {
            if (opened || run > 0) && !out.is_empty() && !rest.is_empty() {
                out.push(' ');
            }
        }
        1 => out.push('\n'),
        _ => out.push_str("\n\n"),
    }
    cursor + run
}

/// Whether a line is only a list marker (`-`, `*`, `+`, `1.`, `1)`) and
/// whitespace — the remains of a bullet whose whole content was extracted.
fn only_list_marker(line: &str) -> bool {
    let trimmed = line.trim();
    if matches!(trimmed, "-" | "*" | "+") {
        return true;
    }
    trimmed.strip_suffix(['.', ')']).is_some_and(|digits| {
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    })
}

// ── Raster image rendering (bounded, downscaled) ─────────────────

/// Maximum longest side (px) of the rendered image the GUI keeps per image — a
/// `MAX_IMAGE_LONGEST_SIDE_PX² · 4` = 16 MiB RGBA tile. A larger — but still
/// valid — raster is decoded under the shared generous raster budget and
/// DOWNSCALED to this cap so a tall chrome screenshot renders (not a
/// placeholder); only a genuinely corrupt / missing / non-image / header-bomb
/// target degrades to the "🖼️ …" placeholder.
const MAX_IMAGE_LONGEST_SIDE_PX: u32 = 2048;

/// Shrink a decoded raster to the display cap `MAX_IMAGE_LONGEST_SIDE_PX`.
fn downscale_raster(width: u32, height: u32, rgba: Vec<u8>) -> Option<(u32, u32, Vec<u8>)> {
    if width <= MAX_IMAGE_LONGEST_SIDE_PX && height <= MAX_IMAGE_LONGEST_SIDE_PX {
        return Some((width, height, rgba));
    }
    // `thumbnail` computes the aspect-preserving target dims internally (no
    // manual float casts) and only shrinks when the image exceeds the box.
    let img = image_crate::DynamicImage::ImageRgba8(image_crate::RgbaImage::from_raw(
        width, height, rgba,
    )?);
    let resized = img
        .thumbnail(MAX_IMAGE_LONGEST_SIDE_PX, MAX_IMAGE_LONGEST_SIDE_PX)
        .into_rgba8();
    Some((resized.width(), resized.height(), resized.into_raw()))
}

/// Decode `bytes` as a native raster and downscale it to the display cap,
/// returning `(width, height, RGBA8 pixels)` on success and `None` on any
/// failure — every `None` renders as the "🖼️ …" placeholder.
///
/// The decode step reuses the shared [`media_target::raster_decode_limits`]
/// budget (generous dimension + alloc caps, so a legitimate tall screenshot
/// still decodes) and then shrinks the result to the renderable cap so the
/// persistent handle stays small and the renderer upload happens once. The
/// transient decode is bounded by that budget; only a header-bomb target fails
/// outright.
#[must_use]
fn render_raster(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let (width, height, rgba) =
        media_target::decode_raster_bytes(bytes, media_target::raster_decode_limits())?;
    downscale_raster(width, height, rgba)
}

/// Decode and validate the image encoded in a `data:…` URI, returning
/// `(width, height, RGBA8 pixels)` on success and `None` on any failure — every
/// `None` renders as the "🖼️ image" placeholder. Vets the URI through the same
/// declared-subtype==actual-bytes gate the provider's image gate uses (a
/// `data:image/png` holding JPEG bytes is mismatched and rejected), then uses
/// the shared raster decode + downscale pipeline.
#[must_use]
fn decode_data_uri_image(uri: &str) -> Option<(u32, u32, Vec<u8>)> {
    let (width, height, rgba) =
        media_target::decode_native_data_uri(uri, media_target::raster_decode_limits())?;
    downscale_raster(width, height, rgba)
}

// ── Process-lifetime image caches ───────────────────────────────

/// A bounded least-recently-used cache: `get` bumps recency so a recently-viewed
/// entry survives eviction, `insert` evicts the oldest once the capacity cap is
/// hit. Keys are looked up by [`Borrow`] so callers can use an unowned key (e.g.
/// a `&str` media marker).
struct BoundedLruCache<K, V> {
    entries: HashMap<K, V>,
    /// LRU order of `entries` keys; back = most recently used.
    lru: VecDeque<K>,
    cap: usize,
}

impl<K, V> BoundedLruCache<K, V>
where
    K: Hash + Eq + Clone,
{
    fn with_capacity(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            cap,
        }
    }

    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hit = self.entries.get(key);
        if hit.is_some() {
            // Bump recency so recently-viewed images survive eviction.
            if let Some(pos) = self.lru.iter().position(|k| k.borrow() == key) {
                let key = self.lru.remove(pos).expect("position found by iteration");
                self.lru.push_back(key);
            }
        }
        hit
    }

    fn insert(&mut self, key: K, value: V) {
        // Remove any existing occurrence so a re-insert replaces the value and
        // moves the key to the most-recent position exactly once. Two concurrent
        // check→decode→insert passes for the same key must not duplicate it in
        // `lru`, or eviction could drop a live entry ahead of a stale copy.
        if let Some(pos) = self.lru.iter().position(|k| k == &key) {
            self.lru.remove(pos);
        }
        self.entries.insert(key.clone(), value);
        self.lru.push_back(key);
        while self.entries.len() > self.cap {
            let oldest = self
                .lru
                .pop_front()
                .expect("lru mirrors entries, so a front key always exists");
            self.entries.remove(&oldest);
        }
    }
}

/// Maximum decoded-image outcomes retained per process-lifetime cache (data-URI
/// and local-path). Each `Decoded` entry holds up to
/// `MAX_IMAGE_LONGEST_SIDE_PX² · 4` (16 MiB) of RGBA under the display budget,
/// so this bound (32) caps worst-case retained pixels at ~512 MiB per cache.
const MAX_CACHED_IMAGES: usize = 32;

/// Outcome of one cached image decode attempt.
#[derive(Debug)]
enum CachedImage {
    /// Successfully decoded; the stable handle is reused across frames, which
    /// keeps the wgpu texture Id stable too (re-building a handle in `view()`
    /// would decode every frame).
    Decoded(image::Handle),
    /// Failed validation — cached so a corrupt payload isn't re-decoded on
    /// every frame; renders as the caller's placeholder.
    Failed,
}

impl CachedImage {
    fn from_raster(raster: Option<(u32, u32, Vec<u8>)>) -> Self {
        match raster {
            Some((width, height, rgba)) => {
                Self::Decoded(image::Handle::from_rgba(width, height, rgba))
            }
            None => Self::Failed,
        }
    }

    fn handle(&self) -> Option<image::Handle> {
        match self {
            Self::Decoded(handle) => Some(handle.clone()),
            Self::Failed => None,
        }
    }
}

/// Process-lifetime cache of data-URI decode outcomes, keyed by the full
/// data-URI string. Evicts the least-recently-used entry past
/// [`MAX_CACHED_IMAGES`].
static DATA_URI_IMAGE_CACHE: LazyLock<Mutex<BoundedLruCache<String, CachedImage>>> =
    LazyLock::new(|| Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_IMAGES)));

/// Return the cached outcome for `key`, decoding and caching it on a miss.
///
/// The cached entry is inspected under the lock and returned through
/// [`CachedImage::handle`] — a cached [`CachedImage::Failed`] is a hit that
/// yields `None`, never a miss (falling through would re-decode a corrupt
/// payload on every frame). The decode itself runs with the lock released, so a
/// raster decode never blocks a concurrent frame, and the owned key is cloned
/// only when the outcome is inserted — a cache hit copies nothing.
#[must_use]
fn cached_image<K, Q>(
    cache: &Mutex<BoundedLruCache<K, CachedImage>>,
    key: &Q,
    decode: impl FnOnce() -> Option<(u32, u32, Vec<u8>)>,
) -> Option<image::Handle>
where
    K: Hash + Eq + Clone + Borrow<Q>,
    Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
{
    if let Some(cached) = cache.lock().unwrap_poison().get(key) {
        return cached.handle();
    }
    let outcome = CachedImage::from_raster(decode());
    let handle = outcome.handle();
    cache.lock().unwrap_poison().insert(key.to_owned(), outcome);
    handle
}

/// Return a stable, cached iced handle for the image in a `data:…` URI, or
/// `None` when the payload is missing, over-cap, or undecodable — the caller
/// renders the "🖼️ image" placeholder in that case (never raw base64 text,
/// never a silent blank).
///
/// Over-cap payloads are refused by `media_target::data_uri_base64_payload`
/// from length alone BEFORE the cache is consulted, so an oversized data URI is
/// never retained as a cache key.
#[must_use]
fn data_uri_image_handle(uri: &str) -> Option<image::Handle> {
    media_target::data_uri_base64_payload(uri)?;
    cached_image(&DATA_URI_IMAGE_CACHE, uri, || decode_data_uri_image(uri))
}

// ── Local-path image rendering (bounded, cached) ─────────────────

/// Process-lifetime cache of local-image decode outcomes, keyed by (canonical
/// path, mtime) — a modified file is a new entry. Evicts the least-recently-used
/// entry past [`MAX_CACHED_IMAGES`].
static LOCAL_IMAGE_CACHE: LazyLock<Mutex<BoundedLruCache<(PathBuf, SystemTime), CachedImage>>> =
    LazyLock::new(|| Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_IMAGES)));

/// Return a stable, cached iced handle for a local image file, or `None` when
/// the file is missing, relative, over-cap, or undecodable — the caller renders
/// the "🖼️ {filename}" placeholder (never a blank). The raster is decoded under
/// the shared generous budget and downscaled to the display cap ([`render_raster`]),
/// then cached at most once per (canonical path, mtime).
#[must_use]
fn local_path_image_handle(path: &str) -> Option<image::Handle> {
    let p = Path::new(path);
    if !p.is_absolute() {
        return None;
    }
    let meta = std::fs::metadata(p).ok()?;
    // Shared 50 MiB cap: a huge file is refused from metadata alone (never
    // opened), so the render thread is not asked to read it.
    if !meta.is_file() || meta.len() > crate::util::INBOUND_IMAGE_MAX_INPUT_BYTES {
        return None;
    }
    let mtime = meta.modified().ok()?;
    let canonical = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    cached_image(&LOCAL_IMAGE_CACHE, &(canonical.clone(), mtime), || {
        std::fs::read(&canonical)
            .ok()
            .and_then(|b| render_raster(&b))
    })
}

// ── Clickable file markers ──────────────────────────────────────

/// The link value a file attachment row's click carries.
const FILE_LINK_PREFIX: &str = "mahbot-file:";

/// The target a file row's click carries, or `None` for any other link.
pub(crate) fn file_link_target(uri: &str) -> Option<&str> {
    uri.strip_prefix(FILE_LINK_PREFIX)
}

/// The link a file attachment row carries: the resolved target it reveals.
fn file_link_uri(target: &str) -> String {
    format!("{FILE_LINK_PREFIX}{target}")
}

/// Resolve a marker target the way a hand-over does: an absolute target as it
/// is, a relative one against the conversation's workspace.
fn resolve_file_marker_target(target: &str, workspace: Option<&Path>) -> Result<PathBuf, String> {
    if target.is_empty() {
        return Err("The marker does not name a file.".to_string());
    }
    // `[FILE:…]` is local-only, exactly as at delivery time.
    if media_target::is_valid_remote_url(target) {
        return Err(format!(
            "Not a local file: {}",
            media_target::marker_name(target)
        ));
    }
    let raw = Path::new(target);
    if raw.is_absolute() {
        return Ok(raw.to_path_buf());
    }
    workspace
        .map(|root| root.join(raw))
        .ok_or_else(|| format!("Cannot open '{target}': the conversation's workspace is unknown."))
}

/// Reveal a marker's target in the platform file manager. Only an existing
/// regular file is handed over: the launchers report success and failure
/// unreliably, so a target that is gone, is not a file, or names no path at all
/// is reported instead of opening something.
pub(crate) fn reveal_file_marker(target: &str, workspace: Option<&Path>) -> Result<(), String> {
    let path = resolve_file_marker_target(target, workspace)?;
    if !path.exists() {
        return Err(format!("File not found: {}", path.display()));
    }
    if !path.is_file() {
        return Err(format!("Not a file: {}", path.display()));
    }
    super::reveal_in_file_manager(&path)
}

// ── Attachment group ────────────────────────────────────────────

/// The drawn body of one message: its attachment group above its whole text, in
/// the stored order — never interleaved, neither ever dropped. `fallback` draws
/// what to show when the body carries neither (content the preprocessing
/// strips, an empty body).
#[expect(
    clippy::large_types_passed_by_value,
    reason = "mirrors `markdown::view`, which hands the settings over by value"
)]
pub(crate) fn body<'a, Message: 'a>(
    attachments: &'a [MediaAttachment],
    items: &'a [markdown::Item],
    settings: markdown::Settings,
    on_link_click: impl Fn(String) -> Message + Clone + 'a,
    fallback: impl FnOnce() -> Element<'a, Message>,
) -> Element<'a, Message> {
    let text_size = f32::from(settings.text_size);
    let group = attachment_group(attachments, text_size, on_link_click.clone());
    let text =
        (!items.is_empty()).then(|| selectable::markdown::view(items, settings).map(on_link_click));
    match (group, text) {
        (Some(group), Some(text)) => Column::with_children(vec![group, text])
            .spacing(theme::SPACE_6)
            .into(),
        (Some(group), None) => group,
        (None, Some(text)) => text,
        (None, None) => fallback(),
    }
}

/// The attachment group of a rendered body: its pictures and its named files,
/// one block above the text, in the order the content carried them — drawn at
/// the text's own size. `None` when the body carries no attachment.
fn attachment_group<'a, Message: 'a>(
    attachments: &'a [MediaAttachment],
    text_size: f32,
    on_link_click: impl Fn(String) -> Message + Clone + 'a,
) -> Option<Element<'a, Message>> {
    if attachments.is_empty() {
        return None;
    }
    let rows: Vec<Element<'a, Message>> = attachments
        .iter()
        .map(|attachment| match &attachment.image {
            // A picture the content names: drawn from the cached bounded
            // decode, or named by its label when there is nothing to draw
            // (missing / corrupt / over-cap file, undrawable address, or a
            // payload the content no longer carries) — never a broken image,
            // never raw base64.
            Some(url) => picture_row(url, &attachment.label, text_size),
            None => label_row(attachment, text_size, on_link_click.clone()),
        })
        .collect();
    Some(Column::with_children(rows).spacing(theme::SPACE_6).into())
}

/// The picture `url` draws, or `label` when there is no handle for it.
fn picture_row<'a, Message: 'a>(
    url: &'a str,
    label: &'a str,
    text_size: f32,
) -> Element<'a, Message> {
    let handle = if url.starts_with("data:") {
        data_uri_image_handle(url)
    } else {
        local_path_image_handle(url)
    };
    match handle {
        Some(handle) => image::Image::new(handle)
            .width(Length::Fill)
            .content_fit(ContentFit::Contain)
            .into(),
        None => label_text(label, text_size),
    }
}

/// A named attachment's row. A revealable one is the same plain label as any
/// other — no colour, no underline, no button chrome: only a click reveals the
/// file, exactly as a `[FILE:…]` marker always behaved.
fn label_row<'a, Message: 'a>(
    attachment: &'a MediaAttachment,
    text_size: f32,
    on_link_click: impl Fn(String) -> Message + 'a,
) -> Element<'a, Message> {
    match &attachment.reveal {
        Some(target) => {
            let label = text::Span::new(attachment.label.as_str()).link(file_link_uri(target));
            selectable::rich_text([label])
                .size(text_size)
                .on_link_click(on_link_click)
                .into()
        }
        None => label_text(&attachment.label, text_size),
    }
}

/// A selectable label, in the body's own text colour.
fn label_text<'a, Message: 'a>(label: &'a str, text_size: f32) -> Element<'a, Message> {
    widgets::selectable_text(label, theme::TEXT_PRIMARY)
        .size(text_size)
        .into()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::DATA_URI_OMITTED_PREFIX;
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use iced::widget::markdown;
    use std::io::Cursor;

    /// The context an assistant-authored body renders under: markers resolve
    /// against `workspace`, and an undeliverable one becomes a reason.
    fn agent(workspace: &Path) -> MarkerContext<'_> {
        MarkerContext {
            workspace: Some(workspace),
            refusals: true,
        }
    }

    /// The context of content the owner typed: no resolution, no refusals.
    fn owner(workspace: &Path) -> MarkerContext<'_> {
        MarkerContext {
            workspace: Some(workspace),
            refusals: false,
        }
    }

    /// A picture attachment, drawn from `image`.
    fn picture(image: Option<&str>, label: &str) -> MediaAttachment {
        MediaAttachment {
            image: image.map(str::to_string),
            label: label.to_string(),
            reveal: None,
        }
    }

    /// An audio/video/document attachment, revealed at `reveal` on a click.
    fn named(label: &str, reveal: Option<&str>) -> MediaAttachment {
        MediaAttachment {
            image: None,
            label: label.to_string(),
            reveal: reveal.map(str::to_string),
        }
    }

    /// All paragraph text a parsed body carries, flattened.
    fn paragraph_text(items: &[markdown::Item]) -> String {
        let style = crate::gui::theme::markdown_settings().style;
        items
            .iter()
            .filter_map(|item| match item {
                markdown::Item::Paragraph(text) => Some(
                    text.spans(style)
                        .iter()
                        .map(|span| span.text.to_string())
                        .collect::<String>(),
                ),
                _ => None,
            })
            .collect()
    }

    #[expect(clippy::too_many_lines)] // large table-driven fixture
    #[test]
    fn preprocess_table() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let photo = root.join("photo.png");
        let spaced = root.join("my file.png");
        let recording = root.join("recording.ogg");
        let nested = root.join("dir/subdir/rec.ogg");
        let clip = root.join("video.mp4");
        let report = root.join("generated/report.pdf");
        let readme = root.join("README");
        std::fs::write(&photo, tiny_png(2, 1)).unwrap();
        std::fs::write(&spaced, tiny_png(2, 1)).unwrap();
        std::fs::write(&recording, b"ogg").unwrap();
        std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
        std::fs::write(&nested, b"ogg").unwrap();
        std::fs::write(&clip, b"mp4").unwrap();
        std::fs::create_dir_all(report.parent().unwrap()).unwrap();
        std::fs::write(&report, b"pdf").unwrap();
        std::fs::write(&readme, b"readme").unwrap();
        let photo_str = photo.to_string_lossy().into_owned();
        let spaced_str = spaced.to_string_lossy().into_owned();
        let report_str = report.to_string_lossy().into_owned();
        let readme_str = readme.to_string_lossy().into_owned();
        let valid_uri = data_uri(&tiny_png(2, 1));
        let gone = root.join("cleaned_up.png");
        let gone_str = gone.to_string_lossy().into_owned();
        let root_name = root.file_name().unwrap().to_string_lossy().into_owned();

        let cases: Vec<(&str, String, String, Vec<MediaAttachment>)> = vec![
            (
                "resolves a workspace-relative image marker",
                "Look [IMAGE:photo.png] here".into(),
                "Look here".into(),
                vec![picture(Some(&photo_str), "🖼️ photo.png")],
            ),
            (
                "uses an absolute image marker as it is",
                format!("Look [IMAGE:{photo_str}] here"),
                "Look here".into(),
                vec![picture(Some(&photo_str), "🖼️ photo.png")],
            ),
            (
                "delivers markers written in lowercase",
                "img [image:photo.png] hear [audio:recording.ogg]".into(),
                "img hear".into(),
                vec![
                    picture(Some(&photo_str), "🖼️ photo.png"),
                    named("🎵 recording.ogg", None),
                ],
            ),
            (
                "refuses a lowercase marker it cannot deliver instead of showing it",
                "see [image:cleaned_up.png]".into(),
                "see Could not send \"cleaned_up.png\": file not found.".into(),
                Vec::new(),
            ),
            (
                "resolves a relative image marker with spaces in the path",
                "img [IMAGE:my file.png]".into(),
                "img".into(),
                vec![picture(Some(&spaced_str), "🖼️ my file.png")],
            ),
            (
                "reports a relative image marker whose file is gone",
                "Look [IMAGE:cleaned_up.png] here".into(),
                "Look Could not send \"cleaned_up.png\": file not found. here".into(),
                Vec::new(),
            ),
            (
                "reports an absolute image marker whose file is gone instead of drawing a placeholder",
                format!("Look [IMAGE:{gone_str}] here"),
                "Look Could not send \"cleaned_up.png\": file not found. here".into(),
                Vec::new(),
            ),
            (
                "reports a directory image target",
                format!("[IMAGE:{}]", root.display()),
                format!("Could not send \"{root_name}\": it is a directory."),
                Vec::new(),
            ),
            (
                "reports a target that is not an image",
                {
                    let text = root.join("notes.txt");
                    std::fs::write(&text, b"not an image").unwrap();
                    format!("[IMAGE:{}]", text.display())
                },
                "Could not send \"notes.txt\": it is not an image.".into(),
                Vec::new(),
            ),
            (
                "names the stripped-data-URI placeholder generically and draws nothing",
                format!("Look [IMAGE:{DATA_URI_OMITTED_PREFIX} (35 bytes)>] here"),
                "Look here".into(),
                vec![picture(None, "🖼️ image")],
            ),
            (
                "draws an inline image payload",
                format!("Look [IMAGE:{valid_uri}] here"),
                "Look here".into(),
                vec![picture(Some(&valid_uri), "🖼️ image")],
            ),
            (
                "keeps a bare word of prose as written",
                "use `[IMAGE:path]` or `[VIDEO:...]` here".into(),
                "use `[IMAGE:path]` or `[VIDEO:...]` here".into(),
                Vec::new(),
            ),
            (
                "resolves a relative audio marker",
                "Listen [AUDIO:recording.ogg]".into(),
                "Listen".into(),
                vec![named("🎵 recording.ogg", None)],
            ),
            (
                "resolves a relative audio marker in a nested path",
                "hear [AUDIO:dir/subdir/rec.ogg]".into(),
                "hear".into(),
                vec![named("🎵 rec.ogg", None)],
            ),
            (
                "reports a missing audio marker",
                "Listen [AUDIO:gone.ogg]".into(),
                "Listen Could not send \"gone.ogg\": file not found.".into(),
                Vec::new(),
            ),
            (
                "resolves a relative video marker",
                "Watch [VIDEO:video.mp4]".into(),
                "Watch".into(),
                vec![named("🎬 Video: video.mp4", None)],
            ),
            (
                "names a file marker's resolved target",
                "Get [FILE:generated/report.pdf]".into(),
                "Get".into(),
                vec![named("📄 report.pdf", Some(&report_str))],
            ),
            (
                "delivers a workspace file whose name is a bare word",
                "Get [FILE:README]".into(),
                "Get".into(),
                vec![named("📄 README", Some(&readme_str))],
            ),
            (
                "reports a file marker whose file is gone",
                "Get [FILE:generated/gone.pdf]".into(),
                "Get Could not send \"gone.pdf\": file not found.".into(),
                Vec::new(),
            ),
            (
                "reports a file marker that names an address",
                "Get [FILE:https://example.com/a.pdf]".into(),
                "Get Could not send \"a.pdf\": an address is not a local file.".into(),
                Vec::new(),
            ),
            (
                "refuses a file marker's inline payload instead of naming the blob",
                "Get [FILE:data:text/plain;base64,aGVsbG8=]".into(),
                "Get Could not send \"file\": it carries an inline payload, not a file.".into(),
                Vec::new(),
            ),
            (
                "replaces an audio transcription block",
                "[Audio transcription of recording.ogg]: Hello world".into(),
                "🔊 Hello world".into(),
                Vec::new(),
            ),
            (
                "renders every line of a multiline audio transcription under 🔊",
                "[Audio transcription of voice.ogg]: Line one\nLine two".into(),
                "🔊 Line one\nLine two".into(),
                Vec::new(),
            ),
            (
                "renders every line of a multiline video transcription under 🎬 and leaves following message content untouched",
                "[Video transcription of clip.mp4]: Line one\nLine two\n\nEdit this".into(),
                "🎬 Line one\nLine two\n\nEdit this".into(),
                Vec::new(),
            ),
            (
                "video-transcription format contains Video and is handled before the [VIDEO:...] pattern",
                "[Video transcription of clip.mp4]: hi there [VIDEO:video.mp4]".into(),
                "🎬 hi there".into(),
                vec![named("🎬 Video: video.mp4", None)],
            ),
            (
                "audio-transcription format contains Audio and is handled before the [AUDIO:...] pattern",
                "[Audio transcription of msg.ogg]: hi there [AUDIO:recording.ogg]".into(),
                "🔊 hi there".into(),
                vec![named("🎵 recording.ogg", None)],
            ),
            (
                "separates mixed image/audio/video markers from the text around them",
                "![]() [IMAGE:photo.png] and [AUDIO:recording.ogg] end [VIDEO:video.mp4]".into(),
                "and end".into(),
                vec![
                    picture(Some(&photo_str), "🖼️ photo.png"),
                    named("🎵 recording.ogg", None),
                    named("🎬 Video: video.mp4", None),
                ],
            ),
            (
                "drops a bullet whose only content was the picture",
                "- [IMAGE:photo.png]".into(),
                String::new(),
                vec![picture(Some(&photo_str), "🖼️ photo.png")],
            ),
            (
                "keeps the rest of a list whose first bullet was a picture",
                "1. [IMAGE:photo.png]\n2. the rest".into(),
                "2. the rest".into(),
                vec![picture(Some(&photo_str), "🖼️ photo.png")],
            ),
            (
                "leaves content without markers unchanged",
                "Hello world".into(),
                "Hello world".into(),
                Vec::new(),
            ),
            (
                "returns the empty string unchanged",
                String::new(),
                String::new(),
                Vec::new(),
            ),
            (
                "strips a saved image annotation and its blank-line separator",
                format!("[IMAGE:{valid_uri}]\n\n[Saved image: /uploads/img.png]"),
                String::new(),
                vec![picture(Some(&valid_uri), "🖼️ image")],
            ),
            (
                "strips a saved image annotation at the start with no preceding separator",
                "[Saved image: /uploads/img.png]".into(),
                String::new(),
                Vec::new(),
            ),
            (
                "turns a saved video annotation into a visible clip label",
                "Clip [Saved video: /uploads/clip.mp4] here".into(),
                "Clip 🎬 Video: clip.mp4 here".into(),
                Vec::new(),
            ),
            (
                "keeps the saved video clip visible alongside its transcription",
                "[Video transcription of clip.mp4]: Line one\n\n[Saved video: /uploads/clip.mp4]"
                    .into(),
                "🎬 Line one\n\n🎬 Video: clip.mp4".into(),
                Vec::new(),
            ),
        ];

        for (i, (name, input, expected_text, expected_attachments)) in cases.iter().enumerate() {
            let rendered = preprocess(input, &agent(root));
            assert_eq!(rendered.text, *expected_text, "case {i} ({name}) text");
            assert_eq!(
                rendered.attachments, *expected_attachments,
                "case {i} ({name}) attachments"
            );
        }
    }

    #[test]
    fn own_text_keeps_the_structural_display() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let photo = root.join("photo.png");
        std::fs::write(&photo, tiny_png(2, 1)).unwrap();
        let photo_str = photo.to_string_lossy().into_owned();

        // What the owner typed is never resolved into something else and never
        // turned into a notice: a relative marker stays their words, an
        // absolute one draws exactly as before.
        let rendered = preprocess("[IMAGE:generated/x.png] [VIDEO:clip.mp4]", &owner(root));
        assert_eq!(rendered.text, "[IMAGE:generated/x.png]");
        assert_eq!(
            rendered.attachments,
            vec![named("🎬 Video: clip.mp4", None)]
        );

        let rendered = preprocess(
            &format!("[IMAGE:{photo_str}] [IMAGE:/tmp/gone.png]"),
            &owner(root),
        );
        assert_eq!(rendered.text, "", "the pictures leave no text behind");
        assert_eq!(
            rendered.attachments,
            vec![
                picture(Some(&photo_str), "🖼️ photo.png"),
                picture(Some("/tmp/gone.png"), "🖼️ gone.png"),
            ]
        );

        let rendered = preprocess("[FILE:notes.txt]", &owner(root));
        assert_eq!(rendered.text, "");
        assert_eq!(
            rendered.attachments,
            vec![named("📄 notes.txt", Some("notes.txt"))],
            "an owner-typed file keeps its own target"
        );
    }

    // ── Extracted pictures ───────────────────────────────────────

    #[test]
    fn a_picture_attachment_keeps_its_whole_target() {
        // The renderer needs the full payload or path — a truncated or escaped
        // one never draws — and the parser must see no image item at all.
        let tmp = tempfile::tempdir().unwrap();
        let spaced = tmp.path().join("my file.png");
        std::fs::write(&spaced, tiny_png(2, 1)).unwrap();
        let spaced_str = spaced.to_string_lossy().into_owned();
        let uri = data_uri(&tiny_png(2, 1));

        for (input, image, label) in [
            (format!("see [IMAGE:{uri}]"), uri.clone(), "🖼️ image"),
            (
                "see [IMAGE:my file.png]".to_string(),
                spaced_str.clone(),
                "🖼️ my file.png",
            ),
        ] {
            let rendered = preprocess(&input, &agent(tmp.path()));
            assert_eq!(
                rendered.attachments,
                vec![picture(Some(&image), label)],
                "one attachment per picture"
            );
            assert_eq!(rendered.text, "see");
            assert!(
                !rendered.text.contains("!["),
                "no markdown image reaches the parser: {:?}",
                rendered.text
            );
            let items: Vec<_> = markdown::parse(&rendered.text).collect();
            assert!(
                items
                    .iter()
                    .all(|item| !matches!(item, markdown::Item::Image { .. })),
                "the body never carries an image item: {items:?}"
            );
        }
    }

    #[test]
    fn text_beside_a_picture_survives_the_parse() {
        // The mechanism: the markdown parser drains every span collected
        // before an image into its `alt`, so the words before the picture were
        // never drawn. Extraction leaves them in the body.
        let tmp = tempfile::tempdir().unwrap();
        let uri = data_uri(&tiny_png(2, 1));
        let rendered = preprocess(&format!("8. the body [IMAGE:{uri}]"), &agent(tmp.path()));
        assert_eq!(rendered.attachments, vec![picture(Some(&uri), "🖼️ image")]);

        let items: Vec<_> = markdown::parse(&rendered.text).collect();
        let [markdown::Item::List { bullets, .. }] = items.as_slice() else {
            panic!("expected the list to survive, got {items:?}");
        };
        let [markdown::Bullet::Point { items }] = bullets.as_slice() else {
            panic!("expected one bullet, got {bullets:?}");
        };
        assert_eq!(paragraph_text(items), "the body");
    }

    #[test]
    fn a_literal_markdown_image_is_extracted() {
        // An agent writes markdown itself; the same extraction keeps the words
        // around its picture.
        let tmp = tempfile::tempdir().unwrap();
        let rendered = preprocess("Look ![shot](url) here", &agent(tmp.path()));
        assert_eq!(rendered.text, "Look here");
        assert_eq!(rendered.attachments, vec![picture(Some("url"), "🖼️ url")]);
        assert!(!rendered.text.contains("!["));
    }

    #[test]
    fn a_picture_on_its_own_line_keeps_the_paragraph() {
        // The picture sits on a line of its own between two lines of one
        // paragraph: taking that line with it leaves the one line break the
        // body had, so the text stays a single paragraph.
        let tmp = tempfile::tempdir().unwrap();
        let rendered = preprocess("before\n![shot](url)\nafter", &agent(tmp.path()));
        assert_eq!(rendered.text, "before\nafter");
        assert_eq!(rendered.attachments, vec![picture(Some("url"), "🖼️ url")]);
        let items: Vec<_> = markdown::parse(&rendered.text).collect();
        assert!(
            matches!(items.as_slice(), [markdown::Item::Paragraph(_)]),
            "one paragraph, not two: {items:?}"
        );

        // A blank line on either side is a paragraph gap and stays one.
        let rendered = preprocess("before\n\n![shot](url)\n\nafter", &agent(tmp.path()));
        assert_eq!(rendered.text, "before\n\nafter");
    }

    #[test]
    fn code_keeps_its_marker_and_image_source() {
        // Inside code the syntax is the subject, so it stays byte for byte:
        // marker and markdown picture alike, in a fenced block, an indented
        // block and an inline span.
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("shot.png");
        std::fs::write(&img, b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").unwrap();
        let img_str = img.to_string_lossy().into_owned();

        for body in [
            format!("```\n[IMAGE:{img_str}]\n![x](y)\n```"),
            format!("    [IMAGE:{img_str}]\n    ![x](y)"),
            format!("use `[IMAGE:{img_str}]` and `![x](y)` here"),
        ] {
            let rendered = preprocess(&body, &agent(tmp.path()));
            assert!(rendered.attachments.is_empty(), "nothing is extracted");
            assert_eq!(rendered.text, body);
        }
    }

    #[test]
    fn attachments_keep_body_order_across_both_sources() {
        // Both sources rewrite the one body, so the walk order is the body
        // order — no source's attachments are grouped ahead of the other's.
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("shot.png");
        std::fs::write(&img, b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").unwrap();
        let img_str = img.to_string_lossy().into_owned();

        for (body, second_first) in [
            (
                format!("[IMAGE:{img_str}] and ![second](second.png)"),
                false,
            ),
            (format!("![second](second.png) and [IMAGE:{img_str}]"), true),
        ] {
            let rendered = preprocess(&body, &agent(tmp.path()));
            let marker = picture(Some(&img_str), "🖼️ shot.png");
            let literal = picture(Some("second.png"), "🖼️ second.png");
            let expected = if second_first {
                vec![literal, marker]
            } else {
                vec![marker, literal]
            };
            assert_eq!(rendered.attachments, expected, "body order: {body}");
        }
    }

    // ── Clickable file markers ────────────────────────────────────

    #[test]
    fn file_markers_pair_labels_with_resolved_targets() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let report = root.join("generated/report.pdf");
        std::fs::create_dir_all(report.parent().unwrap()).unwrap();
        std::fs::write(&report, b"pdf").unwrap();
        let notes = root.join("notes.txt");
        std::fs::write(&notes, b"notes").unwrap();
        let readme = root.join("README");
        std::fs::write(&readme, b"readme").unwrap();

        let body = "Get [FILE:generated/report.pdf] and [FILE:notes.txt] and [FILE:README] then \
                    [FILE:generated/gone.pdf] and [FILE:data:text/plain;base64,aGVsbG8=]";
        let rendered = preprocess(body, &agent(root));
        assert_eq!(
            rendered.attachments,
            vec![
                named("📄 report.pdf", Some(&report.to_string_lossy())),
                named("📄 notes.txt", Some(&notes.to_string_lossy())),
                // A bare word the workspace has is a file, not prose.
                named("📄 README", Some(&readme.to_string_lossy())),
            ],
            "only the files the body delivers become attachments"
        );
        assert!(
            !rendered.text.contains("[FILE:"),
            "no marker survives in what the user reads: {}",
            rendered.text
        );
    }

    #[test]
    fn file_marker_targets_resolve_against_the_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        assert_eq!(
            resolve_file_marker_target("generated/report.pdf", Some(root)).unwrap(),
            root.join("generated/report.pdf"),
            "a relative target resolves against the conversation's workspace"
        );
        let absolute = root.join("elsewhere/report.pdf");
        assert_eq!(
            resolve_file_marker_target(&absolute.to_string_lossy(), Some(root)).unwrap(),
            absolute,
            "an absolute target is used as it is"
        );
        assert_eq!(
            resolve_file_marker_target("report.pdf", None).unwrap_err(),
            "Cannot open 'report.pdf': the conversation's workspace is unknown."
        );
        assert_eq!(
            resolve_file_marker_target("https://example.com/report.pdf", Some(root)).unwrap_err(),
            "Not a local file: report.pdf"
        );
        assert_eq!(
            resolve_file_marker_target("", Some(root)).unwrap_err(),
            "The marker does not name a file."
        );
    }

    #[test]
    fn reveal_file_marker_refuses_a_target_that_is_not_a_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let missing = root.join("generated/report.pdf");
        assert_eq!(
            reveal_file_marker("generated/report.pdf", Some(root)).unwrap_err(),
            format!("File not found: {}", missing.display()),
            "a relative target must resolve against the workspace"
        );

        std::fs::create_dir(root.join("folder")).unwrap();
        assert_eq!(
            reveal_file_marker("folder", Some(root)).unwrap_err(),
            format!("Not a file: {}", root.join("folder").display())
        );
    }

    #[test]
    fn file_name_or_path_table() {
        // On Unix `file_name()` additionally normalizes a trailing slash.
        let cases: &[(&str, &str, &str)] = &[
            ("unix path", "/foo/bar.txt", "bar.txt"),
            ("nested path", "/foo/bar/doc.txt", "doc.txt"),
            ("no directory component", "bar.txt", "bar.txt"),
            ("trailing slash is normalized on Unix", "/foo/bar/", "bar"),
        ];
        for (i, (name, input, expected)) in cases.iter().enumerate() {
            assert_eq!(file_name_or_path(input), *expected, "case {i} ({name})");
        }
    }

    // ── Data-URI image helpers ────────────────────────────────────

    /// Encode a solid-red PNG of the given dimensions (test helper).
    fn tiny_png(width: u32, height: u32) -> Vec<u8> {
        let img = ::image::RgbaImage::from_pixel(width, height, ::image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ::image::ImageFormat::Png)
            .expect("test PNG must encode");
        buf
    }

    fn data_uri(png: &[u8]) -> String {
        format!("data:image/png;base64,{}", STANDARD.encode(png))
    }

    #[test]
    fn decode_data_uri_image_table() {
        // URIs are built before the table: the success case needs a real
        // encoded PNG, and the two over-cap cases need a payload/dimension
        // past their cap.
        let valid_uri = data_uri(&tiny_png(2, 1));
        let invalid_base64 = "data:image/png;base64,%%%not-base64%%%".to_string();
        let non_image = format!("data:text/plain;base64,{}", STANDARD.encode(b"hello world"));
        let oversized_payload = format!(
            "data:image/png;base64,{}",
            "A".repeat(media_target::MAX_DATA_URI_ENCODED_BYTES + 1)
        );
        let oversized_dimensions = data_uri(&tiny_png(MAX_IMAGE_LONGEST_SIDE_PX + 1, 1));
        let missing_marker = "data:image/png,AAAA".to_string();
        let mut jbuf = Vec::new();
        ::image::RgbImage::from_pixel(1, 1, ::image::Rgb([255, 0, 0]))
            .write_to(&mut Cursor::new(&mut jbuf), ::image::ImageFormat::Jpeg)
            .expect("test JPEG must encode");
        let mismatched_uri = format!("data:image/png;base64,{}", STANDARD.encode(&jbuf));

        #[expect(clippy::type_complexity)] // one-off test table tuple
        let cases: &[(&str, String, Option<(u32, u32, Vec<u8>)>)] = &[
            (
                "decodes a solid-red 2×1 PNG",
                valid_uri,
                Some((2, 1, vec![255, 0, 0, 255, 255, 0, 0, 255])),
            ),
            ("rejects invalid base64", invalid_base64, None),
            ("rejects non-image bytes", non_image, None),
            (
                "rejects an oversized payload by a pure length check before any base64 work",
                oversized_payload,
                None,
            ),
            (
                "downscales a valid but over-display-cap raster to the render cap",
                oversized_dimensions,
                Some((
                    MAX_IMAGE_LONGEST_SIDE_PX,
                    1,
                    [255, 0, 0, 255].repeat(MAX_IMAGE_LONGEST_SIDE_PX as usize),
                )),
            ),
            (
                "rejects a data URI missing the base64 marker",
                missing_marker,
                None,
            ),
            (
                "rejects a data URI whose declared subtype mismatches the actual bytes",
                mismatched_uri,
                None,
            ),
        ];
        for (i, (name, uri, expected)) in cases.iter().enumerate() {
            assert_eq!(
                decode_data_uri_image(uri).as_ref(),
                expected.as_ref(),
                "case {i} ({name})"
            );
        }
    }

    #[test]
    fn data_uri_image_handle_caches_stable_handle() {
        let uri = data_uri(&tiny_png(3, 2));
        let first = data_uri_image_handle(&uri).expect("valid data URI decodes");
        let second = data_uri_image_handle(&uri).expect("cached hit still decodes");
        // iced Handle::from_rgba assigns a unique Id per construction; a
        // stable cache must return the same Id so the wgpu worker doesn't
        // re-decode and re-upload on every frame.
        assert_eq!(first.id(), second.id());
    }

    #[test]
    fn data_uri_image_handle_caches_failures() {
        let uri = "data:image/png;base64,%%%not-base64%%%";
        // A decode failure is cached so corrupt payloads aren't re-decoded on
        // every frame; the placeholder is returned on every call.
        assert_eq!(data_uri_image_handle(uri), None);
        {
            let cache = DATA_URI_IMAGE_CACHE.lock().unwrap_poison();
            assert!(
                matches!(cache.entries.get(uri), Some(CachedImage::Failed)),
                "failed decode must be cached, got {:?}",
                cache.entries.get(uri)
            );
        }
        assert_eq!(data_uri_image_handle(uri), None);
    }

    #[test]
    fn data_uri_image_handle_rejects_oversized_payload_without_caching() {
        // Over-cap payloads are refused by the pure length check before the
        // cache is consulted — a ~20 MiB key must never be retained.
        let huge = "A".repeat(media_target::MAX_DATA_URI_ENCODED_BYTES + 1);
        let uri = format!("data:image/png;base64,{huge}");
        assert_eq!(data_uri_image_handle(&uri), None);
        let cache = DATA_URI_IMAGE_CACHE.lock().unwrap_poison();
        assert!(
            !cache.entries.contains_key(&uri),
            "over-cap payload must not be cached"
        );
    }

    #[test]
    fn bounded_cache_duplicate_insert_keeps_eviction_exact() {
        // Two concurrent render passes decoding the same key and both reaching the
        // check→decode→insert path must not leave a duplicate in the LRU order, or
        // eviction could drop a live entry ahead of a stale copy.
        let mut cache = BoundedLruCache::with_capacity(2);
        cache.insert(1, "a");
        cache.insert(1, "a2"); // same-key re-insert (concurrent check→decode→insert)
        cache.insert(2, "b");
        assert_eq!(cache.lru.len(), 2, "duplicate key must not survive in lru");
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.get(&2), Some(&"b"));
        // capacity 2, lru now [1, 2] (back = most recently used). Inserting 3 evicts 1.
        cache.insert(3, "c");
        assert_eq!(cache.lru.len(), 2, "eviction must stay exact");
        assert_eq!(cache.get(&1), None, "oldest (1) must be evicted");
        assert_eq!(cache.get(&2), Some(&"b"));
        assert_eq!(cache.get(&3), Some(&"c"));
    }

    #[test]
    fn cached_image_returns_a_cached_failure_without_decoding() {
        let cache = Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_IMAGES));
        cache
            .lock()
            .unwrap_poison()
            .insert(String::from("bad"), CachedImage::Failed);
        // A cached failure is a hit: the decode must not run (a miss would
        // re-decode a corrupt payload on every frame).
        let handle = cached_image(&cache, "bad", || {
            panic!("a cached failure must not be decoded again")
        });
        assert!(handle.is_none());
    }

    #[test]
    fn cached_image_decodes_without_holding_the_cache_lock() {
        let cache = Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_IMAGES));
        // The decode runs with the cache lock released so a raster decode never
        // blocks a concurrent frame; `try_lock` fails — rather than hanging — if
        // the guard is still held here.
        let handle = cached_image(&cache, "raster", || {
            assert!(cache.try_lock().is_ok(), "decode must run unlocked");
            Some((1, 1, vec![0, 0, 0, 0]))
        });
        assert!(handle.is_some(), "the decoded handle is returned");
        assert!(matches!(
            cache.lock().unwrap_poison().get("raster"),
            Some(CachedImage::Decoded(_))
        ));
    }
}
