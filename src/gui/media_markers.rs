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
//! - `[FILE:path]` → 📄 emoji + filename (text), clickable: a click reveals
//!   the file in the OS file manager
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
//! The `[KIND:path]` format is defined by the `MediaMarkerKind` enum in
//! `src/util/mod.rs`, from which `MEDIA_MARKER_PATTERN` is generated — **the
//! single source of truth** for all marker kinds (`IMAGE`, `AUDIO`, `VIDEO`,
//! `FILE`, and any future additions). This module uses the shared
//! [`MEDIA_MARKER_RE`] and [`parse_media_marker`] helper to stay in sync —
//! adding a new marker kind there automatically propagates to this module,
//! `enrichment.rs`, `agent.rs`, and `compatible.rs` without needing per-kind
//! regexes anywhere.
//!
//! # Clickable file markers
//!
//! A `[FILE:…]` marker stays the plain `📄 name` label it renders as today; the
//! viewer only attaches a `mahbot-file:` link to it. Labels are found in the
//! *parsed* text, in body order, which leaves a few exotic blind spots: a name
//! the parser splits into styled spans is never found, a literal `📄 name`
//! written as prose can absorb a marker's link, and a marker rendered inside a
//! code block (deliberately never linked) still counts as the earlier candidate
//! for its label.

use iced::advanced::{image as advanced_image, text};
use iced::widget::{container, image, markdown};
use iced::{ContentFit, Element, Font, Length, Pixels, padding};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use crate::util::media_target;
use crate::util::{
    MEDIA_MARKER_RE, MediaMarkerKind, UnwrapPoison, file_name_or_path, parse_media_marker,
};

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

/// Convert all `[KIND:path]` markers (IMAGE, AUDIO, VIDEO, FILE) to their display
/// form using the canonical [`MEDIA_MARKER_RE`].
///
/// - `[IMAGE:path]` → `![Image](path)` (markdown image syntax — the markdown
///   parser will produce `Item::Image` from this)
/// - `[AUDIO:path]` → 🎵 filename
/// - `[VIDEO:path]` → 🎬 Video: filename
/// - `[FILE:path]` → 📄 filename
fn replace_media_markers(s: &str) -> String {
    MEDIA_MARKER_RE
        .replace_all(s, |caps: &regex::Captures| {
            let (kind, path) = parse_media_marker(caps);
            match kind {
                MediaMarkerKind::Image => {
                    let p = path.trim();
                    if p.starts_with(crate::util::DATA_URI_OMITTED_PREFIX) {
                        // The byte-count placeholder `util::strip_data_uris` leaves
                        // in place of an inline data URI is neither a path
                        // nor a URL: render the same affordance a data URI the
                        // viewer cannot decode falls back to.
                        "🖼️ image".to_string()
                    } else if p.starts_with("data:") {
                        // data-URI: hand off to the viewer, whose cached bounded
                        // decode draws the image or the "🖼️ image" placeholder —
                        // never a broken image. We deliberately do NOT run the
                        // shared classifier here (it would base64-decode on the
                        // render thread).
                        format!("![Image]({p})")
                    } else if media_target::is_valid_remote_url(p)
                        || std::path::Path::new(p).is_absolute()
                    {
                        // Local path (existing or cleaned-up temp file) / URL: a cheap
                        // O(1) structural gate only — NEVER the shared classifier,
                        // whose local-file decode is a blocking raster decode that must
                        // not run on the render thread. The viewer applies the
                        // authoritative bounded decode and falls back to a placeholder
                        // (🖼️ filename) for a missing / corrupt / non-image target, so
                        // a broken image is never drawn and a replayed cleaned-up temp
                        // marker still reads as a placeholder rather than raw marker text.
                        format!("![Image]({p})")
                    } else {
                        // Relative or prose marker — keep as inert text.
                        caps.get_match().as_str().to_string()
                    }
                }
                MediaMarkerKind::Audio => format!("🎵 {}", file_name_or_path(path)),
                MediaMarkerKind::File => file_marker_label(path),
                MediaMarkerKind::Video => format!("🎬 Video: {}", file_name_or_path(path)),
            }
        })
        .to_string()
}

/// The label a `[FILE:path]` marker shows: `📄 name`. A data URI carries no
/// file name (`file_name_or_path` would return a chunk of its base64 payload),
/// so it is named generically instead of dumping the blob into the bubble.
///
/// The click handling finds the marker in the parsed text by this exact label,
/// so both the preprocessor and [`file_markers`] must build it here.
fn file_marker_label(path: &str) -> String {
    if path.trim().starts_with("data:") {
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
/// data-URI string. A bounded LRU shared with the local-image cache so both
/// hold a stable iced [`image::Handle`] (whose [`Id`](image::Handle::id)
/// keeps the wgpu texture stable across frames — re-building a handle in
/// `view()` would decode every frame). Evicts the least-recently-used entry
/// past [`MAX_CACHED_IMAGES`].
static DATA_URI_IMAGE_CACHE: LazyLock<Mutex<BoundedLruCache<String, CachedDataUriImage>>> =
    LazyLock::new(|| Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_IMAGES)));

/// A bounded least-recently-used cache: `get` bumps recency so a recently-viewed
/// entry survives eviction, `insert` evicts the oldest once the capacity cap is
/// hit. Keys are looked up by [`Borrow`](std::borrow::Borrow) so callers can use
/// an unowned key (e.g. a `&str` media marker).
struct BoundedLruCache<K, V> {
    entries: HashMap<K, V>,
    /// LRU order of `entries` keys; back = most recently used.
    lru: VecDeque<K>,
    cap: usize,
}

impl<K, V> BoundedLruCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
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
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq + ?Sized,
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
    {
        let mut cache = DATA_URI_IMAGE_CACHE.lock().unwrap_poison();
        match cache.get(uri) {
            Some(CachedDataUriImage::Decoded(handle)) => return Some(handle.clone()),
            Some(CachedDataUriImage::Failed) => return None,
            None => {}
        }
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
    DATA_URI_IMAGE_CACHE
        .lock()
        .unwrap_poison()
        .insert(uri.to_string(), outcome);
    handle
}

// ── Local-path image rendering (bounded, cached) ─────────────────

/// Maximum decoded-image outcomes retained per process-lifetime cache (data-URI
/// and local-path). Each `Decoded` entry holds up to
/// `MAX_IMAGE_LONGEST_SIDE_PX² · 4` (16 MiB) of RGBA under the display budget,
/// so this bound (32) caps worst-case retained pixels at ~512 MiB per cache.
const MAX_CACHED_IMAGES: usize = 32;

/// Outcome of a local-image decode attempt, cached per (canonical path, mtime).
#[derive(Debug)]
enum CachedLocalImage {
    /// Successfully decoded; the stable handle is reused across frames.
    Decoded(image::Handle),
    /// Failed validation — cached so a corrupt file isn't re-decoded every frame.
    Failed,
}

/// Process-lifetime cache of local-image decode outcomes, keyed by (canonical
/// path, mtime). Same rationale as [`DATA_URI_IMAGE_CACHE`]: a stable iced
/// handle keeps the texture Id — and the uploaded texture — stable across
/// frames, and the result is bounded by an LRU.
static LOCAL_IMAGE_CACHE: LazyLock<
    Mutex<BoundedLruCache<(std::path::PathBuf, std::time::SystemTime), CachedLocalImage>>,
> = LazyLock::new(|| Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_IMAGES)));

/// Return a stable, cached iced handle for a local image file, or `None` when
/// the file is missing, relative, over-cap, or undecodable — the caller renders
/// the "🖼️ {filename}" placeholder (never a blank). The raster is decoded under
/// the shared generous budget and downscaled to the display cap ([`render_raster`]),
/// then cached at most once per (canonical path, mtime).
#[must_use]
fn local_path_image_handle(path: &str) -> Option<image::Handle> {
    let p = std::path::Path::new(path);
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
    let key = (canonical.clone(), mtime);
    {
        let mut cache = LOCAL_IMAGE_CACHE.lock().unwrap_poison();
        match cache.get(&key) {
            Some(CachedLocalImage::Decoded(handle)) => return Some(handle.clone()),
            Some(CachedLocalImage::Failed) => return None,
            None => {}
        }
    }
    // Blocking read + bounded decode on the render thread, deliberately OUTSIDE
    // the cache lock so a concurrent caller isn't blocked for the whole decode;
    // the size/dimension caps bound the work and the result is cached, so it
    // happens once per (canonical path, mtime).
    let outcome = match std::fs::read(&canonical)
        .ok()
        .and_then(|b| render_raster(&b))
    {
        Some((width, height, rgba)) => {
            CachedLocalImage::Decoded(image::Handle::from_rgba(width, height, rgba))
        }
        None => CachedLocalImage::Failed,
    };
    let handle = match &outcome {
        CachedLocalImage::Decoded(handle) => Some(handle.clone()),
        CachedLocalImage::Failed => None,
    };
    LOCAL_IMAGE_CACHE
        .lock()
        .unwrap_poison()
        .insert(key, outcome);
    handle
}

// ── Clickable file markers ──────────────────────────────────────

/// A `[FILE:…]` marker in a transcript body: the label [`preprocess`] renders
/// for it and the target it names. `target` is empty when the marker names a
/// payload (a data URI) rather than a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileMarker {
    pub label: String,
    pub target: String,
}

/// The link value a file marker's click carries.
const FILE_LINK_PREFIX: &str = "mahbot-file:";

/// The `[FILE:…]` markers of a body, in body order. The labels are exactly
/// what `preprocess` renders, so the viewer can find them in the parsed text.
pub(crate) fn file_markers(content: &str) -> Vec<FileMarker> {
    MEDIA_MARKER_RE
        .captures_iter(content)
        .filter_map(|caps| {
            let (kind, path) = parse_media_marker(&caps);
            if kind != MediaMarkerKind::File {
                return None;
            }
            Some(FileMarker {
                label: file_marker_label(path),
                // A data URI names a payload, not a path: no target to reveal,
                // so the click can only report that there is nothing to open.
                target: if path.trim().starts_with("data:") {
                    String::new()
                } else {
                    path.to_string()
                },
            })
        })
        .collect()
}

/// The target a `[FILE:…]` click carries, or `None` for any other link.
pub(crate) fn file_link_target(uri: &str) -> Option<&str> {
    uri.strip_prefix(FILE_LINK_PREFIX)
}

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
        return Err("FILE targets are local files, not URLs.".to_string());
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

// ── Selectable markdown Viewer (iced_selection + inline images) ────

/// A markdown `Viewer` rendering text through `iced_selection` (selectable,
/// copyable) while keeping the inline-image rendering of the previous viewer.
struct SelectableMediaViewer<'a> {
    files: &'a [FileMarker],
    /// Markers already linked during this build. Labels are matched per
    /// paragraph, so without this a later marker with the same label would
    /// resolve to the first marker's target.
    linked: RefCell<Vec<bool>>,
}

/// Render markdown items as selectable text with inline image support.
pub(crate) fn selectable_markdown_view<'a>(
    items: &'a [markdown::Item],
    settings: impl Into<markdown::Settings>,
    files: &'a [FileMarker],
) -> Element<'a, markdown::Uri, iced::Theme, iced::Renderer> {
    markdown::view_with(items, settings, &SelectableMediaViewer::new(files))
}

impl<'a> SelectableMediaViewer<'a> {
    fn new(files: &'a [FileMarker]) -> Self {
        Self {
            files,
            linked: RefCell::new(vec![false; files.len()]),
        }
    }

    /// The spans to render for one block, with file-marker labels linked. A
    /// block with nothing to link keeps the parsed spans as they are.
    fn marker_spans(
        &self,
        spans: &Arc<[text::Span<'static, markdown::Uri>]>,
    ) -> Arc<[text::Span<'static, markdown::Uri>]> {
        self.linkified_spans(spans)
            .map_or_else(|| spans.clone(), Arc::from)
    }

    /// Replace each file marker's rendered label in `spans` with a linked copy
    /// of the same span: same text, font and colour (the link carries no
    /// colour, so a marker never becomes a visibly coloured link), with the
    /// marker URI attached. `None` when no label is there to link, so the caller
    /// keeps the parsed spans.
    fn linkified_spans(
        &self,
        spans: &[text::Span<'static, markdown::Uri>],
    ) -> Option<Vec<text::Span<'static, markdown::Uri>>> {
        let mut linked = self.linked.borrow_mut();
        if !self.files.iter().enumerate().any(|(index, file)| {
            !linked[index]
                && spans
                    .iter()
                    .any(|span| can_link(span) && span.text.contains(&file.label))
        }) {
            return None;
        }
        let mut out = Vec::with_capacity(spans.len() + self.files.len());
        for span in spans {
            if !can_link(span) {
                out.push(span.clone());
                continue;
            }
            let mut rest: &str = span.text.as_ref();
            let mut split = false;
            while let Some((at, index)) = self
                .files
                .iter()
                .enumerate()
                .filter(|(i, _)| !linked[*i])
                .filter_map(|(i, file)| rest.find(&file.label).map(|at| (at, i)))
                .min_by_key(|(at, i)| (*at, std::cmp::Reverse(self.files[*i].label.len())))
            {
                split = true;
                if at > 0 {
                    out.push(span_with_text(span, &rest[..at]));
                }
                let label = &self.files[index].label;
                let mut link = span_with_text(span, label);
                link.link = Some(file_link_uri(&self.files[index].target));
                out.push(link);
                linked[index] = true;
                rest = &rest[at + label.len()..];
            }
            if !split {
                out.push(span.clone());
            } else if !rest.is_empty() {
                out.push(span_with_text(span, rest));
            }
        }
        Some(out)
    }
}

/// Whether a span may take a file marker's link: code spans (the inline-code
/// highlight) and real links keep their own behaviour.
fn can_link(span: &text::Span<'static, markdown::Uri>) -> bool {
    span.link.is_none() && span.highlight.is_none()
}

fn span_with_text(
    span: &text::Span<'static, markdown::Uri>,
    text: &str,
) -> text::Span<'static, markdown::Uri> {
    text::Span {
        text: std::borrow::Cow::Owned(text.to_owned()),
        ..span.clone()
    }
}

impl<'a, Theme, Renderer> markdown::Viewer<'a, markdown::Uri, Theme, Renderer>
    for SelectableMediaViewer<'a>
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
        // Local path (or a URL that cannot be a local file): a bounded, cached
        // decode draws the actual image, or the "🖼️ {filename}" placeholder
        // when the file is missing / corrupt / over-cap — never a broken image.
        if let Some(handle) = local_path_image_handle(url_str) {
            image::Image::new(handle)
                .width(Length::Fill)
                .content_fit(ContentFit::Contain)
                .into()
        } else {
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
        let size = match level {
            markdown::HeadingLevel::H1 => settings.h1_size,
            markdown::HeadingLevel::H2 => settings.h2_size,
            markdown::HeadingLevel::H3 => settings.h3_size,
            markdown::HeadingLevel::H4 => settings.h4_size,
            markdown::HeadingLevel::H5 => settings.h5_size,
            markdown::HeadingLevel::H6 => settings.h6_size,
        };
        let spans = text.spans(settings.style);
        let rich = iced_selection::rich_text(self.marker_spans(&spans)).size(size);
        container(rich.on_link_click(|url| url))
            .padding(padding::top(if index > 0 {
                settings.text_size / 2.0
            } else {
                Pixels::ZERO
            }))
            .into()
    }

    fn paragraph(
        &self,
        settings: markdown::Settings,
        text: &markdown::Text,
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        let spans = text.spans(settings.style);
        iced_selection::rich_text(self.marker_spans(&spans))
            .size(settings.text_size)
            .on_link_click(|url| url)
            .into()
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
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use std::io::Cursor;

    #[expect(clippy::too_many_lines)] // large table-driven fixture
    #[test]
    fn preprocess_table() {
        // Real fixture files so the classifier accepts the local image markers
        // (a marker whose target is not a real raster stays as literal text).
        let tmp = tempfile::tempdir().unwrap();
        let photo = tmp.path().join("photo.png");
        let spaced = tmp.path().join("my file.png");
        std::fs::write(&photo, tiny_png(2, 1)).unwrap();
        std::fs::write(&spaced, tiny_png(2, 1)).unwrap();
        let photo_str = photo.to_string_lossy().into_owned();
        let spaced_str = spaced.to_string_lossy().into_owned();
        let valid_uri = data_uri(&tiny_png(2, 1));
        let cleaned = tmp.path().join("cleaned_up.png"); // does not exist
        let cleaned_str = cleaned.to_string_lossy().into_owned();

        let cases: Vec<(&str, String, String)> = vec![
            (
                "replaces an image marker",
                format!("Look [IMAGE:{photo_str}] here"),
                format!("Look ![Image]({photo_str}) here"),
            ),
            (
                "replaces an image marker with spaces in the path",
                format!("img [IMAGE:{spaced_str}]"),
                format!("img ![Image]({spaced_str})"),
            ),
            (
                "renders a cleaned-up absolute temp-file marker as an image item (placeholder in the viewer)",
                format!("Look [IMAGE:{cleaned_str}] here"),
                format!("Look ![Image]({cleaned_str}) here"),
            ),
            (
                "renders the omitted-data-URI placeholder as the image affordance, not raw marker text",
                format!(
                    "Look [IMAGE:{} (35 bytes)>] here",
                    crate::util::DATA_URI_OMITTED_PREFIX
                ),
                "Look 🖼️ image here".into(),
            ),
            (
                "replaces an audio marker",
                "Listen [AUDIO:/tmp/recording.ogg]".into(),
                "Listen 🎵 recording.ogg".into(),
            ),
            (
                "replaces an audio marker in a nested path",
                "hear [AUDIO:/dir/subdir/rec.ogg]".into(),
                "hear 🎵 rec.ogg".into(),
            ),
            (
                "replaces a video marker",
                "Watch [VIDEO:/tmp/video.mp4]".into(),
                "Watch 🎬 Video: video.mp4".into(),
            ),
            (
                "replaces a file marker",
                "Get [FILE:/tmp/report.pdf]".into(),
                "Get 📄 report.pdf".into(),
            ),
            (
                "names a data-URI file marker generically instead of dumping its payload",
                "Get [FILE:data:text/plain;base64,aGVsbG8=]".into(),
                "Get 📄 file".into(),
            ),
            (
                "replaces an audio transcription block",
                "[Audio transcription of recording.ogg]: Hello world".into(),
                "🔊 Hello world".into(),
            ),
            (
                "renders every line of a multiline audio transcription under 🔊",
                "[Audio transcription of voice.ogg]: Line one\nLine two".into(),
                "🔊 Line one\nLine two".into(),
            ),
            (
                "renders every line of a multiline video transcription under 🎬 and leaves following message content untouched",
                "[Video transcription of clip.mp4]: Line one\nLine two\n\nEdit this".into(),
                "🎬 Line one\nLine two\n\nEdit this".into(),
            ),
            (
                "video-transcription format contains Video and is handled before the [VIDEO:...] pattern",
                "[Video transcription of clip.mp4]: hi there [VIDEO:/tmp/other.mp4]".into(),
                "🎬 hi there 🎬 Video: other.mp4".into(),
            ),
            (
                "audio-transcription format contains Audio and is handled before the [AUDIO:...] pattern",
                "[Audio transcription of msg.ogg]: hi there [AUDIO:/tmp/other.ogg]".into(),
                "🔊 hi there 🎵 other.ogg".into(),
            ),
            (
                "replaces mixed image/audio/video markers",
                format!("![]() [IMAGE:{photo_str}] and [AUDIO:/tmp/b.ogg] end [VIDEO:/tmp/c.mp4]"),
                format!("![]() ![Image]({photo_str}) and 🎵 b.ogg end 🎬 Video: c.mp4"),
            ),
            (
                "leaves content without markers unchanged",
                "Hello world".into(),
                "Hello world".into(),
            ),
            (
                "returns the empty string unchanged",
                String::new(),
                String::new(),
            ),
            (
                "strips a saved image annotation and its blank-line separator",
                format!("[IMAGE:{valid_uri}]\n\n[Saved image: /uploads/img.png]"),
                format!("![Image]({valid_uri})"),
            ),
            (
                "strips a saved image annotation at the start with no preceding separator",
                "[Saved image: /uploads/img.png]".into(),
                String::new(),
            ),
            (
                "turns a saved video annotation into a visible clip placeholder",
                "Clip [Saved video: /uploads/clip.mp4] here".into(),
                "Clip 🎬 Video: clip.mp4 here".into(),
            ),
            (
                "keeps the saved video clip visible alongside its transcription",
                "[Video transcription of clip.mp4]: Line one\n\n[Saved video: /uploads/clip.mp4]"
                    .into(),
                "🎬 Line one\n\n🎬 Video: clip.mp4".into(),
            ),
        ];

        for (i, (name, input, expected)) in cases.iter().enumerate() {
            assert_eq!(preprocess(input), *expected, "case {i} ({name})");
        }
    }

    // ── Clickable file markers ────────────────────────────────────

    #[test]
    fn file_markers_pair_labels_with_targets() {
        let body = "Get [FILE:generated/report.pdf] and [FILE:/tmp/notes.txt] then \
                    [FILE:data:text/plain;base64,aGVsbG8=]";
        let markers = file_markers(body);
        assert_eq!(
            markers,
            vec![
                FileMarker {
                    label: "📄 report.pdf".to_string(),
                    target: "generated/report.pdf".to_string(),
                },
                FileMarker {
                    label: "📄 notes.txt".to_string(),
                    target: "/tmp/notes.txt".to_string(),
                },
                FileMarker {
                    label: "📄 file".to_string(),
                    target: String::new(),
                },
            ]
        );
    }

    #[test]
    fn linkified_spans_links_only_the_marker_label() {
        let span = text::Span::new("Get 📄 report.pdf now");
        let files = [FileMarker {
            label: "📄 report.pdf".to_string(),
            target: "/tmp/report.pdf".to_string(),
        }];
        let spans = SelectableMediaViewer::new(&files)
            .linkified_spans(&[span])
            .unwrap();
        let texts: Vec<&str> = spans.iter().map(|s| s.text.as_ref()).collect();
        assert_eq!(texts, ["Get ", "📄 report.pdf", " now"]);
        assert_eq!(spans[0].link, None);
        assert_eq!(spans[1].link, Some(file_link_uri("/tmp/report.pdf")));
        assert_eq!(
            spans[1].color, None,
            "a file marker must keep its plain colour"
        );
        assert!(
            !spans[1].underline,
            "a file marker must not be underlined at rest"
        );
        assert_eq!(spans[2].link, None);
    }

    #[test]
    fn linkified_spans_links_same_named_markers_to_their_own_targets() {
        let files = [
            FileMarker {
                label: "📄 notes.txt".to_string(),
                target: "first/notes.txt".to_string(),
            },
            FileMarker {
                label: "📄 notes.txt".to_string(),
                target: "second/notes.txt".to_string(),
            },
        ];
        let viewer = SelectableMediaViewer::new(&files);
        let first = viewer
            .linkified_spans(&[text::Span::new("Saved 📄 notes.txt")])
            .unwrap();
        let second = viewer
            .linkified_spans(&[text::Span::new("Again 📄 notes.txt")])
            .unwrap();
        assert_eq!(
            first[1].link,
            Some(file_link_uri("first/notes.txt")),
            "each block must link the next marker, not always the first"
        );
        assert_eq!(second[1].link, Some(file_link_uri("second/notes.txt")));
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
            "FILE targets are local files, not URLs."
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
    fn data_uri_marker_survives_markdown_parse() {
        // The renderer must receive the full, valid data URI — a truncated or
        // escaped URL would fall back to the placeholder even for decodable
        // images. A non-decodable `data:` payload still reaches the markdown
        // Image item (replace_media_markers emits `![Image](...)` for every
        // `data:` URI), and the viewer degrades it to the 🖼️ placeholder rather
        // than drawing a broken image.
        let uri = data_uri(&tiny_png(2, 1));
        let processed = preprocess(&format!("[IMAGE:{uri}]"));
        let items: Vec<_> = markdown::parse(&processed).collect();
        let [markdown::Item::Image { url, .. }] = items.as_slice() else {
            panic!("expected a single Image item, got {items:?}");
        };
        assert_eq!(url.as_str(), uri);
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
}
