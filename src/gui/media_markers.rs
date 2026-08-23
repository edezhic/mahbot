//! Pre-processing for media markers in chat content.
//!
//! Converts raw media markers like `[IMAGE:path]` into forms the markdown
//! renderer can handle gracefully:
//!
//! - `[IMAGE:path]` → proper markdown image syntax `![Image](path)`
//! - `[IMAGE:data:…;base64,…]` → the decoded image itself, rendered from a
//!   process-lifetime cached handle; undecodable or oversized payloads degrade
//!   to the `🖼️ image` placeholder (never raw base64 text, never a blank)
//! - `[AUDIO:path]` → 🎵 emoji + filename (text)
//! - `[VIDEO:path]` → 🎬 emoji + placeholder text
//! - `[Audio transcription of ...]: text` → 🔊 emoji + transcribed text
//! - `[Video transcription of ...]: text` → 🎬 emoji + transcribed text
//! - `[Saved image: path]` → stripped (the image renders separately from its
//!   data-URI marker, so the upload annotation is pure display noise)
//! - `[Saved video: path]` → `🎬 Video: filename` (there is no separate video
//!   renderer, so the clip path must stay visible rather than be hidden)
//!
//! The pre-processing is applied **before** `markdown::parse()` so the
//! standard markdown pipeline naturally produces `Item::Image` from the
//! converted image markers. Audio and video markers become plain text
//! with emoji prefixes since inline audio/video playback is not supported.
//!
//! # Canonical marker pattern
//!
//! The `[KIND:path]` format is defined by `MEDIA_MARKER_PATTERN` in
//! `src/util/mod.rs`, which is **the single source of truth** for all marker
//! kinds (`IMAGE`, `AUDIO`, `VIDEO`, and any future additions).  This module
//! uses the shared [`MEDIA_MARKER_RE`] and [`parse_media_marker`] helper to
//! stay in sync — adding a new marker kind there automatically propagates
//! to this module, `enrichment.rs`, `telegram.rs`, `agent.rs`, and
//! `compatible.rs` without needing per-kind regexes here.

use std::collections::{HashMap, VecDeque};
use std::io::Cursor;
use std::sync::{LazyLock, Mutex};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use iced::advanced::{image as advanced_image, text};
use iced::widget::{image, markdown};
use iced::{ContentFit, Element, Font, Length};

use crate::util::{MEDIA_MARKER_RE, UnwrapPoison, file_name_or_path, parse_media_marker};

/// The `image` crate, aliased because `image` here is iced's widget module.
use ::image as image_crate;

/// Pre-process a content string, converting media markers before markdown parsing.
pub(crate) fn preprocess(content: &str) -> String {
    // Order matters: the transcription annotations contain the words "Audio"
    // and "Video" which overlap with the raw `[AUDIO:...]`/`[VIDEO:...]`
    // patterns.  Handle them first.
    let s = replace_transcription(content);
    let s = replace_media_markers(&s);
    replace_saved_annotations(&s)
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

/// Convert all `[KIND:path]` markers (IMAGE, AUDIO, VIDEO) to their display form
/// using the canonical [`MEDIA_MARKER_RE`].
///
/// - `[IMAGE:path]` → `![Image](path)` (markdown image syntax — the markdown
///   parser will produce `Item::Image` from this)
/// - `[AUDIO:path]` → 🎵 filename
/// - `[VIDEO:path]` → 🎬 Video: filename
fn replace_media_markers(s: &str) -> String {
    MEDIA_MARKER_RE
        .replace_all(s, |caps: &regex::Captures| {
            let (kind, path) = parse_media_marker(caps);
            match kind {
                "IMAGE" => format!("![Image]({path})"),
                "AUDIO" => format!("🎵 {}", file_name_or_path(path)),
                "VIDEO" => format!("🎬 Video: {}", file_name_or_path(path)),
                // Unreachable for well-formed markers (MEDIA_MARKER_RE only
                // matches IMAGE|AUDIO|VIDEO), but defend against future changes.
                _ => caps.get_match().as_str().to_string(),
            }
        })
        .to_string()
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

// ── Data-URI image rendering ─────────────────────────────────────

/// Maximum length (bytes) of the base64 payload inside a `data:…;base64,…`
/// marker the GUI will attempt to decode. Longer payloads are refused by a
/// pure string-length check — never decoded, never cached. ~20 MiB far
/// exceeds realistic inbound images (61 KiB–510 KiB encoded on the
/// byte-identical passthrough path; the compressed path is 1024px/q85).
const MAX_DATA_URI_ENCODED_BYTES: usize = 20 * 1024 * 1024;

/// Maximum longest side (px) of a decoded data-URI image the GUI will render.
/// Images whose header declares a longer side are refused from the header
/// alone — no pixel decode is attempted, so decompression bombs cannot
/// allocate. Over-cap → the "🖼️ image" placeholder.
const MAX_IMAGE_LONGEST_SIDE_PX: u32 = 2048;

/// Decode-allocation budget for [`decode_data_uri_image`]: the largest
/// legitimate bitmap is `MAX_IMAGE_LONGEST_SIDE_PX² · 4` bytes (16 MiB for
/// 2048×2048 RGBA); 64 MiB covers decoder intermediates while still bounding
/// pathological inputs.
const MAX_DATA_URI_DECODE_ALLOC_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum number of data-URI decode outcomes retained in the process-lifetime
/// cache. Each entry holds the data-URI key plus — for successful decodes —
/// up to 16 MiB of RGBA pixels, so the bound keeps memory in check over long
/// sessions.
const MAX_CACHED_DATA_URI_IMAGES: usize = 64;

/// Extract the base64 payload of a `data:…;base64,…` URI, or `None` when the
/// string is not a data URI with a base64 payload. The payload is everything
/// after the LAST `;base64,` token (RFC 2397 allows mediatype parameters
/// before the marker, e.g. `data:image/png;charset=utf-8;base64,AAAA`).
#[must_use]
fn data_uri_base64_payload(uri: &str) -> Option<&str> {
    let rest = uri.strip_prefix("data:")?;
    let payload_start = rest.rfind(";base64,")? + ";base64,".len();
    Some(&rest[payload_start..])
}

/// Decode and validate the image encoded in a `data:…` URI, returning
/// `(width, height, RGBA8 pixels)` on success and `None` on any failure.
///
/// Check order — every `None` renders as the "🖼️ image" placeholder:
/// 1. encoded-payload length cap (before any base64 work);
/// 2. base64 decode;
/// 3. header-only dimension read — oversized images are refused WITHOUT a
///    pixel decode;
/// 4. full decode under explicit [`image_crate::Limits`], converted to RGBA8.
#[must_use]
fn decode_data_uri_image(uri: &str) -> Option<(u32, u32, Vec<u8>)> {
    let payload = data_uri_base64_payload(uri)?;
    if payload.len() > MAX_DATA_URI_ENCODED_BYTES {
        return None;
    }
    let bytes = STANDARD.decode(payload).ok()?;

    // Content sniff from the magic bytes; unknown/unsupported formats (gif,
    // bmp, …) are refused here — only png/jpeg/webp are built in.
    let format = image_crate::guess_format(&bytes).ok()?;

    // Header-only dimension check — no pixel decode for oversized images.
    let header = image_crate::ImageReader::with_format(Cursor::new(&bytes), format);
    let (width, height) = header.into_dimensions().ok()?;
    if width.max(height) > MAX_IMAGE_LONGEST_SIDE_PX {
        return None;
    }

    // Full decode with explicit limits; the image crate's dimension caps are
    // strict, so an over-cap decode fails instead of allocating. (`Limits` is
    // `#[non_exhaustive]`, so it is built via `Default` + field mutation.)
    let mut limits = image_crate::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_LONGEST_SIDE_PX);
    limits.max_image_height = Some(MAX_IMAGE_LONGEST_SIDE_PX);
    limits.max_alloc = Some(MAX_DATA_URI_DECODE_ALLOC_BYTES);
    let mut decoder = image_crate::ImageReader::with_format(Cursor::new(&bytes), format);
    decoder.limits(limits);
    let rgba = decoder.decode().ok()?.to_rgba8();
    Some((rgba.width(), rgba.height(), rgba.into_raw()))
}

/// Outcome of a data-URI decode attempt, cached per URI string.
#[derive(Debug)]
enum CachedDataUriImage {
    /// Successfully decoded; the stable handle is reused across frames.
    Decoded(image::Handle),
    /// Failed validation — cached so a corrupt payload isn't re-decoded on
    /// every frame. Renders as the "🖼️ image" placeholder.
    Failed,
}

/// Process-lifetime cache of data-URI decode outcomes, keyed by the full
/// data-URI string.
///
/// iced [`image::Handle`]s carry a unique [`Id`](image::Handle::id) per
/// construction and the wgpu renderer re-decodes/re-uploads on every new Id,
/// so re-building a handle inside `view()` would decode on every frame.
/// Caching the handle keeps the Id — and thus the uploaded texture — stable.
/// Bounded LRU, evicting the least-recently-used entry past
/// [`MAX_CACHED_DATA_URI_IMAGES`].
static DATA_URI_IMAGE_CACHE: LazyLock<Mutex<DataUriImageCache>> = LazyLock::new(|| {
    Mutex::new(DataUriImageCache {
        entries: HashMap::new(),
        lru: VecDeque::new(),
    })
});

struct DataUriImageCache {
    entries: HashMap<String, CachedDataUriImage>,
    /// LRU order of `entries` keys; back = most recently used.
    lru: VecDeque<String>,
}

impl DataUriImageCache {
    fn get(&mut self, uri: &str) -> Option<&CachedDataUriImage> {
        let hit = self.entries.get(uri);
        if hit.is_some() {
            // Bump recency so recently-viewed images survive eviction.
            if let Some(pos) = self.lru.iter().position(|key| key.as_str() == uri) {
                let key = self.lru.remove(pos).expect("position found by iteration");
                self.lru.push_back(key);
            }
        }
        hit
    }

    fn insert(&mut self, uri: String, outcome: CachedDataUriImage) {
        self.entries.insert(uri.clone(), outcome);
        self.lru.push_back(uri);
        while self.entries.len() > MAX_CACHED_DATA_URI_IMAGES {
            let oldest = self
                .lru
                .pop_front()
                .expect("lru mirrors entries, so a front key always exists");
            self.entries.remove(&oldest);
        }
    }
}

/// Return a stable, cached iced handle for the image in a `data:…` URI, or
/// `None` when the payload is missing, over-cap, or undecodable — the caller
/// renders the "🖼️ image" placeholder in that case (never raw base64 text,
/// never a silent blank).
///
/// Over-cap payloads are refused by a pure length check BEFORE the cache is
/// consulted, so an oversized data URI is never retained as a cache key.
#[must_use]
fn data_uri_image_handle(uri: &str) -> Option<image::Handle> {
    let payload = data_uri_base64_payload(uri)?;
    if payload.len() > MAX_DATA_URI_ENCODED_BYTES {
        return None;
    }
    let mut cache = DATA_URI_IMAGE_CACHE.lock().unwrap_poison();
    match cache.get(uri) {
        Some(CachedDataUriImage::Decoded(handle)) => return Some(handle.clone()),
        Some(CachedDataUriImage::Failed) => return None,
        None => {}
    }
    let outcome = match decode_data_uri_image(uri) {
        Some((width, height, rgba)) => {
            CachedDataUriImage::Decoded(image::Handle::from_rgba(width, height, rgba))
        }
        None => CachedDataUriImage::Failed,
    };
    let handle = match &outcome {
        CachedDataUriImage::Decoded(handle) => Some(handle.clone()),
        CachedDataUriImage::Failed => None,
    };
    cache.insert(uri.to_string(), outcome);
    handle
}

// ── Selectable markdown Viewer (iced_selection + inline images) ────

/// A markdown `Viewer` rendering text through `iced_selection` (selectable,
/// copyable) while keeping the inline-image rendering of the previous viewer.
pub(crate) struct SelectableMediaViewer;

/// Render markdown items as selectable text with inline image support.
pub(crate) fn selectable_markdown_view(
    items: &[markdown::Item],
    settings: impl Into<markdown::Settings>,
) -> Element<'_, markdown::Uri, iced::Theme, iced::Renderer> {
    markdown::view_with(items, settings, &SelectableMediaViewer)
}

impl<'a, Theme, Renderer> markdown::Viewer<'a, markdown::Uri, Theme, Renderer>
    for SelectableMediaViewer
where
    Theme: markdown::Catalog + iced_selection::text::Catalog + 'a,
    Renderer: text::Renderer<Paragraph = iced::advanced::graphics::text::Paragraph, Font = Font>
        + advanced_image::Renderer<Handle = image::Handle>
        + 'a,
{
    fn on_link_click(url: markdown::Uri) -> markdown::Uri {
        url
    }

    fn image(
        &self,
        settings: markdown::Settings,
        url: &'a markdown::Uri,
        _title: &'a str,
        _alt: &markdown::Text,
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        let url_str = url.as_str();
        if url_str.starts_with("data:") {
            // Inbound images arrive as `[IMAGE:data:…;base64,…]` markers
            // (enrichment). Render from the decoded, cached handle; any
            // decode failure or size-cap breach shows the shared
            // "🖼️ image" placeholder — never raw base64 text, never a blank.
            return match data_uri_image_handle(url_str) {
                Some(handle) => image::Image::new(handle)
                    .width(Length::Fill)
                    .content_fit(ContentFit::Contain)
                    .into(),
                None => iced::widget::text("🖼️ image")
                    .size(settings.text_size)
                    .into(),
            };
        }
        let path = std::path::Path::new(url_str);
        if path.exists() {
            // Render the actual image, constrained to the bubble width.
            // Note: path.exists() is synchronous I/O on the render thread,
            // which is acceptable because file checks are fast (~µs) and
            // images only appear for agent-generated files that were just
            // created.  For sessions with many stale image references a
            // cached-existence check could be added.
            image::Image::new(url_str)
                .width(Length::Fill)
                .content_fit(ContentFit::Contain)
                .into()
        } else {
            // File doesn't exist (temp file cleaned up, or path is invalid).
            // Show a fallback with the filename.
            let filename = file_name_or_path(url_str);
            iced::widget::text(format!("🖼️ {filename}"))
                .size(settings.text_size)
                .into()
        }
    }

    fn heading(
        &self,
        settings: markdown::Settings,
        level: &'a markdown::HeadingLevel,
        text: &'a markdown::Text,
        index: usize,
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        iced_selection::markdown::heading(settings, level, text, index, |url| url)
    }

    fn paragraph(
        &self,
        settings: markdown::Settings,
        text: &markdown::Text,
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        iced_selection::markdown::paragraph(settings, text, |url| url)
    }

    fn code_block(
        &self,
        settings: markdown::Settings,
        _language: Option<&'a str>,
        _code: &'a str,
        lines: &'a [markdown::Text],
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        iced_selection::markdown::code_block(settings, lines, |url| url)
    }

    fn unordered_list(
        &self,
        settings: markdown::Settings,
        bullets: &'a [markdown::Bullet],
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        iced_selection::markdown::unordered_list(self, settings, bullets)
    }

    fn ordered_list(
        &self,
        settings: markdown::Settings,
        start: u64,
        bullets: &'a [markdown::Bullet],
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        iced_selection::markdown::ordered_list(self, settings, start, bullets)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_table() {
        let cases: &[(&str, &str, &str)] = &[
            (
                "replaces an image marker",
                "Look [IMAGE:/tmp/photo.png] here",
                "Look ![Image](/tmp/photo.png) here",
            ),
            (
                "replaces an image marker with spaces in the path",
                "img [IMAGE:/tmp/my file.png]",
                "img ![Image](/tmp/my file.png)",
            ),
            (
                "replaces an audio marker",
                "Listen [AUDIO:/tmp/recording.ogg]",
                "Listen 🎵 recording.ogg",
            ),
            (
                "replaces an audio marker in a nested path",
                "hear [AUDIO:/dir/subdir/rec.ogg]",
                "hear 🎵 rec.ogg",
            ),
            (
                "replaces a video marker",
                "Watch [VIDEO:/tmp/video.mp4]",
                "Watch 🎬 Video: video.mp4",
            ),
            (
                "replaces an audio transcription block",
                "[Audio transcription of recording.ogg]: Hello world",
                "🔊 Hello world",
            ),
            (
                "renders every line of a multiline audio transcription under 🔊",
                "[Audio transcription of voice.ogg]: Line one\nLine two",
                "🔊 Line one\nLine two",
            ),
            (
                "renders every line of a multiline video transcription under 🎬 and leaves following message content untouched",
                "[Video transcription of clip.mp4]: Line one\nLine two\n\nEdit this",
                "🎬 Line one\nLine two\n\nEdit this",
            ),
            (
                "video-transcription format contains Video and is handled before the [VIDEO:...] pattern",
                "[Video transcription of clip.mp4]: hi there [VIDEO:/tmp/other.mp4]",
                "🎬 hi there 🎬 Video: other.mp4",
            ),
            (
                "audio-transcription format contains Audio and is handled before the [AUDIO:...] pattern",
                "[Audio transcription of msg.ogg]: hi there [AUDIO:/tmp/other.ogg]",
                "🔊 hi there 🎵 other.ogg",
            ),
            (
                "replaces mixed image/audio/video markers",
                "![]() [IMAGE:/tmp/a.png] and [AUDIO:/tmp/b.ogg] end [VIDEO:/tmp/c.mp4]",
                "![]() ![Image](/tmp/a.png) and 🎵 b.ogg end 🎬 Video: c.mp4",
            ),
            (
                "leaves content without markers unchanged",
                "Hello world",
                "Hello world",
            ),
            ("returns the empty string unchanged", "", ""),
            (
                "strips a saved image annotation and its blank-line separator",
                "[IMAGE:data:image/png;base64,AAAA]\n\n[Saved image: /uploads/img.png]",
                "![Image](data:image/png;base64,AAAA)",
            ),
            (
                "strips a saved image annotation at the start with no preceding separator",
                "[Saved image: /uploads/img.png]",
                "",
            ),
            (
                "turns a saved video annotation into a visible clip placeholder",
                "Clip [Saved video: /uploads/clip.mp4] here",
                "Clip 🎬 Video: clip.mp4 here",
            ),
            (
                "keeps the saved video clip visible alongside its transcription",
                "[Video transcription of clip.mp4]: Line one\n\n[Saved video: /uploads/clip.mp4]",
                "🎬 Line one\n\n🎬 Video: clip.mp4",
            ),
        ];

        for (i, (name, input, expected)) in cases.iter().enumerate() {
            assert_eq!(preprocess(input), *expected, "case {i} ({name})");
        }
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
    fn data_uri_base64_payload_table() {
        let cases: &[(&str, &str, Option<&str>)] = &[
            (
                "extracts the base64 payload",
                "data:image/png;base64,AAAA",
                Some("AAAA"),
            ),
            (
                "handles mediatype parameters before the base64 marker",
                "data:image/png;charset=utf-8;base64,BBBB",
                Some("BBBB"),
            ),
            ("rejects a plain file path", "/tmp/photo.png", None),
            ("rejects an http URL", "https://example.com/img.png", None),
            (
                "rejects a data URI without a base64 marker",
                "data:image/png,raw-bytes",
                None,
            ),
            (
                "allows an empty payload",
                "data:image/png;base64,",
                Some(""),
            ),
        ];
        for (i, (name, uri, expected)) in cases.iter().enumerate() {
            assert_eq!(data_uri_base64_payload(uri), *expected, "case {i} ({name})");
        }
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
            "A".repeat(MAX_DATA_URI_ENCODED_BYTES + 1)
        );
        let oversized_dimensions = data_uri(&tiny_png(MAX_IMAGE_LONGEST_SIDE_PX + 1, 1));
        let missing_marker = "data:image/png,AAAA".to_string();

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
                "rejects oversized dimensions from the header alone before any pixel decode",
                oversized_dimensions,
                None,
            ),
            (
                "rejects a data URI missing the base64 marker",
                missing_marker,
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
                matches!(cache.entries.get(uri), Some(CachedDataUriImage::Failed)),
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
        let huge = "A".repeat(MAX_DATA_URI_ENCODED_BYTES + 1);
        let uri = format!("data:image/png;base64,{huge}");
        assert_eq!(data_uri_image_handle(&uri), None);
        let cache = DATA_URI_IMAGE_CACHE.lock().unwrap_poison();
        assert!(
            !cache.entries.contains_key(&uri),
            "over-cap payload must not be cached"
        );
    }

    #[test]
    fn data_uri_marker_survives_markdown_parse() {
        // The renderer must receive the full data URI — a truncated or
        // escaped URL would fall back to the placeholder even for decodable
        // images.
        let processed = preprocess("[IMAGE:data:image/png;base64,AAAA]");
        let items: Vec<_> = markdown::parse(&processed).collect();
        let [markdown::Item::Image { url, .. }] = items.as_slice() else {
            panic!("expected a single Image item, got {items:?}");
        };
        assert_eq!(url.as_str(), "data:image/png;base64,AAAA");
    }
}
