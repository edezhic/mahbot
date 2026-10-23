//! Pre-processing for media markers in chat content.
//!
//! Converts raw media markers like `[IMAGE:path]` into forms the markdown
//! renderer can handle gracefully:
//!
//! - `[IMAGE:path]` → proper markdown image syntax `![Image](path)`
//! - `[IMAGE:data:…;base64,…]` → the decoded image itself, rendered from a
//!   process-lifetime cached handle; an undecodable or oversized payload
//!   degrades to the `🖼️ image` placeholder (never raw base64 text, never a
//!   blank)
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
//! A `[FILE:…]` marker stays the plain `📄 name` label it renders as; the
//! viewer only attaches a `mahbot-file:` link to it, naming the resolved path.
//! Labels are found in the *parsed* text, in body order, which leaves a few
//! exotic blind spots: a name the parser splits into styled spans is never
//! found, a literal `📄 name` written as prose can absorb a marker's link, and a
//! marker rendered inside a code block (deliberately never linked) still counts
//! as the earlier candidate for its label.

use iced::advanced::{image as advanced_image, text};
use iced::widget::{container, image, markdown};
use iced::{ContentFit, Element, Font, Length, Pixels, padding};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

use super::selectable;
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

/// A rendered body: the text to parse and the `[FILE:…]` markers whose label it
/// carries, in body order.
pub(crate) struct Preprocessed {
    pub text: String,
    pub files: Vec<FileMarker>,
}

/// Pre-process a content string, converting media markers before markdown parsing.
pub(crate) fn preprocess(content: &str, markers: &MarkerContext<'_>) -> Preprocessed {
    // Order matters: the transcription annotations contain the words "Audio"
    // and "Video" which overlap with the raw `[AUDIO:...]`/`[VIDEO:...]`
    // patterns.  Handle them first.
    let s = replace_transcription(content);
    let rendered = replace_media_markers(&s, markers);
    Preprocessed {
        text: replace_saved_annotations(&rendered.text),
        files: rendered.files,
    }
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
/// - `[FILE:path]` → 📄 filename, recorded for the click handler
///
/// A target that names something but cannot be shown becomes a refusal notice
/// when the context allows one; a bare word of prose is left as written.
fn replace_media_markers(s: &str, markers: &MarkerContext<'_>) -> Preprocessed {
    let roots: Vec<PathBuf> = markers
        .workspace
        .map(Path::to_path_buf)
        .into_iter()
        .collect();
    let mut files = Vec::new();
    let text = MEDIA_MARKER_RE
        .replace_all(s, |caps: &regex::Captures| {
            let (kind, target) = parse_media_marker(caps);
            let target = target.trim();
            render_marker(kind, target, markers, &roots, &mut files)
                .unwrap_or_else(|| caps.get_match().as_str().to_string())
        })
        .to_string();
    Preprocessed { text, files }
}

/// The display form of one marker, or `None` when it is not a reference and
/// stays as the author wrote it.
fn render_marker(
    kind: MediaMarkerKind,
    target: &str,
    markers: &MarkerContext<'_>,
    roots: &[PathBuf],
    files: &mut Vec<FileMarker>,
) -> Option<String> {
    // Verbatim content — what the owner typed, a tool result, promoted
    // reasoning — keeps the structural display: never resolved, never refused.
    if !markers.refusals {
        return render_own_text(kind, target, files);
    }
    match media_target::resolve_marker_target(target, roots) {
        MarkerTarget::NotAReference => None,
        MarkerTarget::Inline => Some(match kind {
            MediaMarkerKind::Image => inline_image(target),
            MediaMarkerKind::Audio | MediaMarkerKind::Video | MediaMarkerKind::File => {
                media_target::inline_marker_refusal(kind)
            }
        }),
        MarkerTarget::Remote => Some(match media_target::remote_marker(kind, target) {
            media_target::RemoteMarker::Refused(notice) => notice,
            media_target::RemoteMarker::Deliverable => {
                delivered_marker(kind, target, target, files)
            }
        }),
        MarkerTarget::Local(path) => Some(match media_target::local_refusal(kind, &path) {
            Some(reason) => media_target::marker_refusal_for(target, reason),
            None => delivered_marker(kind, target, &path.to_string_lossy(), files),
        }),
    }
}

/// The display form of a marker whose target is deliverable: `resolved` is what
/// the thing is reached by, while the label keeps naming the target as the
/// message spelled it.
fn delivered_marker(
    kind: MediaMarkerKind,
    target: &str,
    resolved: &str,
    files: &mut Vec<FileMarker>,
) -> String {
    match kind {
        MediaMarkerKind::Image => image_item(resolved),
        MediaMarkerKind::Audio | MediaMarkerKind::Video => media_label(kind, target),
        MediaMarkerKind::File => push_file_marker(target, resolved.to_string(), files),
    }
}

/// The display form of a marker in verbatim content (what the owner typed, a
/// tool result, promoted reasoning). The product does not resolve or refuse
/// text it does not own, so this is the structural gate the page has always
/// used: an inline payload, a well-formed address or an absolute path draws;
/// anything else is left as typed.
fn render_own_text(
    kind: MediaMarkerKind,
    target: &str,
    files: &mut Vec<FileMarker>,
) -> Option<String> {
    match kind {
        MediaMarkerKind::Image => {
            if target.starts_with("data:")
                || target.starts_with(crate::util::DATA_URI_OMITTED_PREFIX)
            {
                Some(inline_image(target))
            } else if media_target::is_valid_remote_url(target) || Path::new(target).is_absolute() {
                Some(image_item(target))
            } else {
                None
            }
        }
        MediaMarkerKind::Audio | MediaMarkerKind::Video => Some(media_label(kind, target)),
        MediaMarkerKind::File => {
            // A payload names no path to reveal.
            let click_target = if target.starts_with("data:")
                || target.starts_with(crate::util::DATA_URI_OMITTED_PREFIX)
            {
                String::new()
            } else {
                target.to_string()
            };
            Some(push_file_marker(target, click_target, files))
        }
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

/// Record a delivered `[FILE:…]` marker for the click handler and return the
/// `📄 name` label the body carries.
fn push_file_marker(target: &str, click_target: String, files: &mut Vec<FileMarker>) -> String {
    let label = file_marker_label(target);
    files.push(FileMarker {
        label: label.clone(),
        target: click_target,
    });
    label
}

/// The display form of an inline-payload `[IMAGE:…]` marker: the byte-count
/// placeholder `util::strip_data_uris` leaves behind shows the generic image
/// affordance, while a real payload renders through the viewer's cached bounded
/// decode — never raw base64, never a broken image.
fn inline_image(target: &str) -> String {
    if target.starts_with(crate::util::DATA_URI_OMITTED_PREFIX) {
        "🖼️ image".to_string()
    } else {
        image_item(target)
    }
}

/// A markdown image item for `target`. A destination with a space or a
/// parenthesis only parses inside `<>` — without it the markdown parser leaves
/// the item as literal text, so a picture in a workspace whose path has a space
/// would never render.
fn image_item(target: &str) -> String {
    if target
        .chars()
        .any(|character| character.is_whitespace() || matches!(character, '(' | ')'))
    {
        format!("![Image](<{target}>)")
    } else {
        format!("![Image]({target})")
    }
}

/// The label a `[FILE:path]` marker shows: `📄 name`. A payload carries no file
/// name (`file_name_or_path` would return a chunk of its base64), so it is named
/// generically instead of dumping the blob into the bubble.
///
/// The click handling finds the marker in the parsed text by this exact label,
/// so the preprocessor is the only place it is built.
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

/// A `[FILE:…]` marker in a transcript body: the label [`preprocess`] renders
/// for it and the target it names. `target` is empty when the marker names a
/// payload (a data URI) rather than a path, and for product content it is the
/// resolved absolute path, so a click needs no workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileMarker {
    pub label: String,
    pub target: String,
}

/// The link value a file marker's click carries.
const FILE_LINK_PREFIX: &str = "mahbot-file:";

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

// ── Selectable markdown Viewer (selectable text + inline images) ───

/// A markdown `Viewer` rendering text through [`super::selectable`] (selectable,
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
    Theme: markdown::Catalog + 'a,
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
        let rich = selectable::rich_text(self.marker_spans(&spans)).size(size);
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
        selectable::rich_text(self.marker_spans(&spans))
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
        selectable::markdown::code_block(settings, lines, |url| url)
    }

    fn unordered_list(
        &self,
        settings: markdown::Settings,
        bullets: &'a [markdown::Bullet],
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        selectable::markdown::unordered_list(self, settings, bullets)
    }

    fn ordered_list(
        &self,
        settings: markdown::Settings,
        start: u64,
        bullets: &'a [markdown::Bullet],
    ) -> Element<'a, markdown::Uri, Theme, Renderer> {
        selectable::markdown::ordered_list(self, settings, start, bullets)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine as _, engine::general_purpose::STANDARD};
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
        std::fs::write(&photo, tiny_png(2, 1)).unwrap();
        std::fs::write(&spaced, tiny_png(2, 1)).unwrap();
        std::fs::write(&recording, b"ogg").unwrap();
        std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
        std::fs::write(&nested, b"ogg").unwrap();
        std::fs::write(&clip, b"mp4").unwrap();
        std::fs::create_dir_all(report.parent().unwrap()).unwrap();
        std::fs::write(&report, b"pdf").unwrap();
        std::fs::write(root.join("README"), b"readme").unwrap();
        let photo_str = photo.to_string_lossy().into_owned();
        let spaced_str = spaced.to_string_lossy().into_owned();
        let valid_uri = data_uri(&tiny_png(2, 1));
        let gone = root.join("cleaned_up.png");
        let gone_str = gone.to_string_lossy().into_owned();
        let root_name = root.file_name().unwrap().to_string_lossy().into_owned();

        let cases: Vec<(&str, String, String)> = vec![
            (
                "resolves a workspace-relative image marker",
                "Look [IMAGE:photo.png] here".into(),
                format!("Look ![Image]({photo_str}) here"),
            ),
            (
                "uses an absolute image marker as it is",
                format!("Look [IMAGE:{photo_str}] here"),
                format!("Look ![Image]({photo_str}) here"),
            ),
            (
                "delivers markers written in lowercase",
                "img [image:photo.png] hear [audio:recording.ogg]".into(),
                format!("img ![Image]({photo_str}) hear 🎵 recording.ogg"),
            ),
            (
                "refuses a lowercase marker it cannot deliver instead of showing it",
                "see [image:cleaned_up.png]".into(),
                "see Could not send \"cleaned_up.png\": file not found.".into(),
            ),
            (
                "resolves a relative image marker with spaces in the path, quoted for the markdown parser",
                "img [IMAGE:my file.png]".into(),
                format!("img ![Image](<{spaced_str}>)"),
            ),
            (
                "reports a relative image marker whose file is gone",
                "Look [IMAGE:cleaned_up.png] here".into(),
                "Look Could not send \"cleaned_up.png\": file not found. here".into(),
            ),
            (
                "reports an absolute image marker whose file is gone instead of drawing a placeholder",
                format!("Look [IMAGE:{gone_str}] here"),
                "Look Could not send \"cleaned_up.png\": file not found. here".into(),
            ),
            (
                "reports a directory image target",
                format!("[IMAGE:{}]", root.display()),
                format!("Could not send \"{root_name}\": it is a directory."),
            ),
            (
                "reports a target that is not an image",
                {
                    let text = root.join("notes.txt");
                    std::fs::write(&text, b"not an image").unwrap();
                    format!("[IMAGE:{}]", text.display())
                },
                "Could not send \"notes.txt\": it is not an image.".into(),
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
                "renders an inline image payload through the viewer",
                format!("Look [IMAGE:{valid_uri}] here"),
                format!("Look ![Image]({valid_uri}) here"),
            ),
            (
                "keeps a bare word of prose as written",
                "use `[IMAGE:path]` or `[VIDEO:...]` here".into(),
                "use `[IMAGE:path]` or `[VIDEO:...]` here".into(),
            ),
            (
                "resolves a relative audio marker",
                "Listen [AUDIO:recording.ogg]".into(),
                "Listen 🎵 recording.ogg".into(),
            ),
            (
                "resolves a relative audio marker in a nested path",
                "hear [AUDIO:dir/subdir/rec.ogg]".into(),
                "hear 🎵 rec.ogg".into(),
            ),
            (
                "reports a missing audio marker",
                "Listen [AUDIO:gone.ogg]".into(),
                "Listen Could not send \"gone.ogg\": file not found.".into(),
            ),
            (
                "resolves a relative video marker",
                "Watch [VIDEO:video.mp4]".into(),
                "Watch 🎬 Video: video.mp4".into(),
            ),
            (
                "names a file marker's resolved target",
                "Get [FILE:generated/report.pdf]".into(),
                "Get 📄 report.pdf".into(),
            ),
            (
                "delivers a workspace file whose name is a bare word",
                "Get [FILE:README]".into(),
                "Get 📄 README".into(),
            ),
            (
                "reports a file marker whose file is gone",
                "Get [FILE:generated/gone.pdf]".into(),
                "Get Could not send \"gone.pdf\": file not found.".into(),
            ),
            (
                "reports a file marker that names an address",
                "Get [FILE:https://example.com/a.pdf]".into(),
                "Get Could not send \"a.pdf\": an address is not a local file.".into(),
            ),
            (
                "refuses a file marker's inline payload instead of naming the blob",
                "Get [FILE:data:text/plain;base64,aGVsbG8=]".into(),
                "Get Could not send \"file\": it carries an inline payload, not a file.".into(),
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
                "[Video transcription of clip.mp4]: hi there [VIDEO:video.mp4]".into(),
                "🎬 hi there 🎬 Video: video.mp4".into(),
            ),
            (
                "audio-transcription format contains Audio and is handled before the [AUDIO:...] pattern",
                "[Audio transcription of msg.ogg]: hi there [AUDIO:recording.ogg]".into(),
                "🔊 hi there 🎵 recording.ogg".into(),
            ),
            (
                "replaces mixed image/audio/video markers",
                "![]() [IMAGE:photo.png] and [AUDIO:recording.ogg] end [VIDEO:video.mp4]".into(),
                format!("![]() ![Image]({photo_str}) and 🎵 recording.ogg end 🎬 Video: video.mp4"),
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
            assert_eq!(
                preprocess(input, &agent(root)).text,
                *expected,
                "case {i} ({name})"
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
        assert_eq!(
            preprocess("[IMAGE:generated/x.png] [VIDEO:clip.mp4]", &owner(root)).text,
            "[IMAGE:generated/x.png] 🎬 Video: clip.mp4"
        );
        assert_eq!(
            preprocess(
                &format!("[IMAGE:{photo_str}] [IMAGE:/tmp/gone.png]"),
                &owner(root)
            )
            .text,
            format!("![Image]({photo_str}) ![Image](/tmp/gone.png)")
        );
        assert_eq!(
            preprocess("[FILE:notes.txt]", &owner(root)).files,
            vec![FileMarker {
                label: "📄 notes.txt".to_string(),
                target: "notes.txt".to_string(),
            }]
        );
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
            rendered.files,
            vec![
                FileMarker {
                    label: "📄 report.pdf".to_string(),
                    target: report.to_string_lossy().into_owned(),
                },
                FileMarker {
                    label: "📄 notes.txt".to_string(),
                    target: notes.to_string_lossy().into_owned(),
                },
                // A bare word the workspace has is a file, not prose.
                FileMarker {
                    label: "📄 README".to_string(),
                    target: readme.to_string_lossy().into_owned(),
                },
            ],
            "only the labels the text carries are link candidates"
        );
        assert!(
            !rendered.text.contains("[FILE:"),
            "no marker survives in what the user reads: {}",
            rendered.text
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
    fn data_uri_marker_survives_markdown_parse() {
        // The renderer must receive the full, valid data URI — a truncated or
        // escaped URL would fall back to the placeholder even for decodable
        // images. A non-decodable `data:` payload still reaches the markdown
        // Image item (replace_media_markers emits `![Image](...)` for every
        // `data:` URI), and the viewer degrades it to the 🖼️ placeholder rather
        // than drawing a broken image.
        let uri = data_uri(&tiny_png(2, 1));
        let tmp = tempfile::tempdir().unwrap();
        let processed = preprocess(&format!("[IMAGE:{uri}]"), &agent(tmp.path())).text;
        let items: Vec<_> = markdown::parse(&processed).collect();
        let [markdown::Item::Image { url, .. }] = items.as_slice() else {
            panic!("expected a single Image item, got {items:?}");
        };
        assert_eq!(url.as_str(), uri);
    }

    #[test]
    fn spaced_path_marker_survives_markdown_parse() {
        // A resolved picture in a workspace whose path has a space is emitted as
        // a `<>`-quoted destination; the markdown parser must still produce one
        // Image item whose URL is the plain path, or the picture never renders.
        let tmp = tempfile::tempdir().unwrap();
        let spaced = tmp.path().join("my file.png");
        std::fs::write(&spaced, tiny_png(2, 1)).unwrap();
        let processed = preprocess("[IMAGE:my file.png]", &agent(tmp.path())).text;
        assert_eq!(
            processed,
            format!("![Image](<{}>)", spaced.to_string_lossy())
        );
        let items: Vec<_> = markdown::parse(&processed).collect();
        let [markdown::Item::Image { url, .. }] = items.as_slice() else {
            panic!("expected a single Image item, got {items:?}");
        };
        assert_eq!(url.as_str(), spaced.to_string_lossy());
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
