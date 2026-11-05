//! Shared semantics for `[KIND:…]` media-marker targets.
//!
//! [`resolve_marker_target`] is the single authority on what a target names, used by the
//! delivery surfaces and by the desktop's display layer (which canonicalises an address's
//! scheme casing first): an http(s) address, an inline payload, a bare word of prose, or a
//! local path — a relative one resolved against the workspace the message belongs to, exactly
//! as `[FILE:…]` targets are. It reads a scheme as written, so only a canonical lowercase
//! http(s) target is an address, and it performs no containment: only `[FILE:…]` is
//! workspace-confined; the other kinds deliberately accept any location on disk. A local path
//! that does not exist is offered [`repaired_local_target`] next: one directory component the
//! sender clearly misspelled is corrected, and a file name never is. [`target_file_key`] reduces
//! a marker to the one file it names, so a reply can tell which of a turn's renders it chose.
//! What a placed target becomes is decided here too — [`local_refusal`] gates delivery (Telegram,
//! which must ship bytes), [`remote_marker`] and [`inline_marker_refusal`] answer wherever a
//! marker of an unservable kind is refused, and [`marker_refusal`] words the refusal shared by
//! the display layer and the transport's post-accept failure notices.
//!
//! Consumers deciding whether a target is a "real image" use
//! [`classify_media_image_target`] (Telegram and enrichment, which offload the blocking decode
//! off the async path). Two deliberately do not: the GUI render path (a full decode would run
//! on the render thread — the display layer decides from [`resolve_marker_target`] and
//! [`repaired_local_target`] whether a marker is a row, decoding nothing, and the viewer does
//! the bounded decode itself) and the provider's agent-context scanner (`compatible.rs`, which
//! must never read a local file during request serialization and calls [`is_native_data_uri`]
//! directly). No workspace-boundary containment — a valid raster anywhere on disk is valid.
//! Local files are validated by a real raster decode (memoized per canonical path + mtime),
//! remote URLs by well-formedness. A `data:...` URI is NOT classified here: it is an inline
//! payload validated by its own [`is_native_data_uri`] gate, which only the provider uses
//! directly.

use crate::util::{MediaMarkerKind, UnwrapPoison};
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE, URL_SAFE_NO_PAD},
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// A classified `[IMAGE:...]` marker target: what consumers match on. The value itself
/// (canonical path / URL / URI) is re-read from the marker text by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaTarget {
    /// Absolute path to an existing regular file that decodes as a genuine PNG/JPEG/WebP
    /// raster — a corrupt-but-magic-valid file is not accepted.
    LocalImage,
    /// Well-formed http(s) URL for an image reference.
    RemoteUrl,
    /// Anything else — not deliverable as an image reference. A `data:...` URI is not
    /// classified here: it is an inline payload validated by [`is_native_data_uri`].
    Invalid,
}

/// Encoded-payload ceiling for any `data:...;base64,...` URI. Over-cap payloads are refused from
/// length alone — never decoded — so no thread buffers a huge payload.
pub(crate) const MAX_DATA_URI_ENCODED_BYTES: usize = 20 * 1024 * 1024;

/// Shared raster-decode allocation budget: bounded so a header bomb is refused, generous enough
/// that a legitimate tall screenshot decodes. With [`RASTER_DECODE_MAX_DIMENSION_PX`] it is the
/// one raster envelope the inbound pipeline decodes under.
pub(crate) const RASTER_DECODE_MAX_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// Longest side (px) the shared decode accepts — rejects a dimension bomb from the header
/// before any pixel buffer is allocated.
pub(crate) const RASTER_DECODE_MAX_DIMENSION_PX: u32 = 16384;

/// The shared [`image::Limits`]: a dimension and alloc cap so a header bomb is refused before
/// any pixel buffer. Used by the provider's data-URI gate ([`is_native_data_uri`]), the
/// classifier's local decode, and the GUI render decode + downscale.
#[must_use]
pub(crate) fn raster_decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(RASTER_DECODE_MAX_ALLOC_BYTES);
    limits.max_image_width = Some(RASTER_DECODE_MAX_DIMENSION_PX);
    limits.max_image_height = Some(RASTER_DECODE_MAX_DIMENSION_PX);
    limits
}

/// Classify a raw `[IMAGE:...]` marker target. A `data:...` URI is not a target this
/// classifier handles (it is an inline payload for [`is_native_data_uri`]).
#[must_use]
pub(crate) fn classify_media_image_target(target: &str) -> MediaTarget {
    let trimmed = target.trim();
    if is_valid_remote_url(trimmed) {
        MediaTarget::RemoteUrl
    } else if is_decodable_local_raster(trimmed) {
        MediaTarget::LocalImage
    } else {
        MediaTarget::Invalid
    }
}

/// Where a `[KIND:…]` marker target points after workspace-relative resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MarkerTarget {
    /// A bare word naming no file — `[IMAGE:path]`, the form prose uses to explain the
    /// syntax. Kept as written.
    NotAReference,
    /// An inline `data:` payload, or the placeholder left where one was stripped.
    Inline,
    /// An http(s) address.
    Remote,
    /// A local path: an absolute target as written, a relative one joined to the workspace
    /// root that has it (the first root when none does).
    Local(PathBuf),
}

/// What an http(s) address becomes for a marker of `kind`: the address itself (the surface
/// renders its own form) or the notice a user reads instead.
#[must_use]
pub(crate) fn resolve_marker_target(target: &str, roots: &[PathBuf]) -> MarkerTarget {
    let target = target.trim();
    // The stripped-payload placeholder is a data URI's stand-in: still a value, never a path
    // or an address.
    if is_inline_payload(target) {
        return MarkerTarget::Inline;
    }
    if crate::util::is_http_url(target) {
        return MarkerTarget::Remote;
    }
    let raw = Path::new(target);
    if raw.is_absolute() {
        return MarkerTarget::Local(raw.to_path_buf());
    }
    if let Some(joined) = roots
        .iter()
        .map(|root| root.join(raw))
        .find(|candidate| candidate.exists())
    {
        return MarkerTarget::Local(joined);
    }
    if !names_a_reference(target) {
        return MarkerTarget::NotAReference;
    }
    MarkerTarget::Local(
        roots
            .first()
            .map_or_else(|| raw.to_path_buf(), |root| root.join(raw)),
    )
}

/// How far ahead of the next candidate the nearest directory must be: a second one an edit behind
/// the best is a crowded field, not a clear intent, so nothing is guessed.
const REPAIR_MARGIN: usize = 2;

/// The existing file a local target names when exactly one directory component in it is a
/// misspelled name: the first component that does not exist is compared with the directories
/// beside it, and the single clearly-nearest one is substituted when the remainder of the path
/// exists under it. `None` when the path exists as written, when the missing component is the
/// file's own name — a wrong name is a wrong file, never a repair — when no directory beside it
/// comes near, when more than one does, or when the rest of the path is not there.
///
/// This is the one repair both delivery and the desktop's display layer apply, so a marker they
/// show is the same file.
#[must_use]
pub(crate) fn repaired_local_target(path: &Path) -> Option<PathBuf> {
    // A trailing separator names a directory, so repairing it would turn it into the file of that
    // name — never the marker's meaning.
    if !path.is_absolute() || path.to_string_lossy().ends_with(['/', '\\']) || !is_missing(path) {
        return None;
    }
    let components: Vec<Component<'_>> = path.components().collect();
    let mut prefix = PathBuf::new();
    let missing = components.iter().position(|component| {
        prefix.push(component);
        is_missing(&prefix)
    })?;
    // A drive, a root, `.` or `..` is not a name a sender mistyped; the last component is the
    // file's own name; and a `..` in what follows names a directory the substitution did not
    // produce, so it is no longer the path as written.
    if !matches!(components[missing], Component::Normal(_))
        || missing + 1 == components.len()
        || components[missing + 1..].contains(&Component::ParentDir)
    {
        return None;
    }
    let name = components[missing].as_os_str().to_str()?;
    let directory: PathBuf = components[..missing].iter().collect();
    let candidate = unique_near_directory(&directory, name)?;
    let repaired = directory
        .join(candidate)
        .join(components[missing + 1..].iter().collect::<PathBuf>());
    // The rest of the path must be there as written, or the substitution is not the one meant.
    std::fs::metadata(&repaired).is_ok().then_some(repaired)
}

/// True when nothing at `path` exists at all. The entry itself is asked (`lstat`), so a symlink
/// whose target is gone is a link that exists, not a mistyped name, and a failed `stat` of any
/// other kind is not a missing file either — the distinction [`local_target_refusal`] draws.
fn is_missing(path: &Path) -> bool {
    matches!(
        std::fs::symlink_metadata(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
    )
}

/// The one directory in `directory` that `name` is a clear misspelling of, judged by
/// [`edit_distance_within`] over [`crate::util::caseless_name_key`] so case and composition are
/// not the typo. The nearest must be within half the name's length and the next-nearest at least
/// [`REPAIR_MARGIN`] edits behind it.
fn unique_near_directory(directory: &Path, name: &str) -> Option<OsString> {
    let written: Vec<char> = crate::util::caseless_name_key(name).chars().collect();
    let limit = (written.len() / 2).max(1);
    let mut best: Option<(usize, OsString)> = None;
    let mut runner_up = usize::MAX;
    for entry in std::fs::read_dir(directory).ok()?.flatten() {
        // Only a directory can stand where the mistyped component stood, and the entry is followed
        // (`stat`) the way the repaired path itself will be, so a directory a link names counts.
        if !std::fs::metadata(entry.path()).is_ok_and(|meta| meta.is_dir()) {
            continue;
        }
        let candidate = entry.file_name();
        let Some(text) = candidate.to_str() else {
            continue;
        };
        let lowered: Vec<char> = crate::util::caseless_name_key(text).chars().collect();
        let Some(distance) = edit_distance_within(&written, &lowered, limit) else {
            continue;
        };
        match &best {
            Some((best_distance, _)) if *best_distance <= distance => {
                runner_up = runner_up.min(distance);
            }
            _ => {
                if let Some((best_distance, _)) = best.replace((distance, candidate)) {
                    runner_up = runner_up.min(best_distance);
                }
            }
        }
    }
    let (distance, candidate) = best?;
    (runner_up >= distance + REPAIR_MARGIN).then_some(candidate)
}

/// The Levenshtein distance between `a` and `b` when it is at most `limit`, `None` otherwise. A
/// row whose every cell is past the limit cannot come back under it, so a long name costs a band
/// rather than a full matrix against every neighbour.
fn edit_distance_within(a: &[char], b: &[char], limit: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > limit {
        return None;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0_usize; b.len() + 1];
    for (row, from) in a.iter().enumerate() {
        current[0] = row + 1;
        let mut row_best = current[0];
        for (column, to) in b.iter().enumerate() {
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + usize::from(from != to));
            row_best = row_best.min(current[column + 1]);
        }
        if row_best > limit {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[b.len()] <= limit).then_some(previous[b.len()])
}

/// True when `target` is a value rather than a location: a `data:` payload, or the placeholder
/// left where one was stripped.
#[must_use]
pub(crate) fn is_inline_payload(target: &str) -> bool {
    target.starts_with("data:") || target.starts_with(crate::util::DATA_URI_OMITTED_PREFIX)
}

/// The identity two targets share when they name one file: a local target resolved the way
/// delivery resolves it — a relative one joined to the workspace that has it, a clearly misspelled
/// directory repaired — then [`crate::util::canonical_or_self`]. An address and an inline payload
/// name no local file and answer `None`, as does prose that names nothing (`[IMAGE:path]` with no
/// workspace file of that name).
#[must_use]
pub(crate) fn target_file_key(target: &str, roots: &[PathBuf]) -> Option<PathBuf> {
    let MarkerTarget::Local(path) = resolve_marker_target(target, roots) else {
        return None;
    };
    Some(crate::util::canonical_or_self(
        &repaired_local_target(&path).unwrap_or(path),
    ))
}

/// True when a target names something: it has a path separator, or its last segment carries an
/// extension (`photo.png`). Only `[IMAGE:path]`-shaped prose — a bare word with neither, and no
/// workspace file of that name — does not.
fn names_a_reference(target: &str) -> bool {
    target.contains('/')
        || target.contains('\\')
        || target
            .rsplit_once('.')
            .is_some_and(|(stem, extension)| !stem.is_empty() && !extension.is_empty())
}

/// What an http(s) address becomes for a marker of `kind`.
pub(crate) enum RemoteMarker {
    /// The address is the whole target; the surface renders its own form of it.
    Deliverable,
    /// Not a supported target for this kind: why it was refused.
    Refused(&'static str),
}

/// Judge an http(s) address for `kind`: a `[FILE:…]` address is never a local file, and an
/// `[IMAGE:…]` address must be well-formed. Every delivering surface asks this, so none states
/// the rule itself.
#[must_use]
pub(crate) fn remote_marker(kind: MediaMarkerKind, target: &str) -> RemoteMarker {
    let reason = match kind {
        MediaMarkerKind::File => FILE_URL_REASON,
        MediaMarkerKind::Image if !is_valid_remote_url(target) => INVALID_ADDRESS_REASON,
        MediaMarkerKind::Image | MediaMarkerKind::Audio | MediaMarkerKind::Video => {
            return RemoteMarker::Deliverable;
        }
    };
    RemoteMarker::Refused(reason)
}

/// The notice for a target that is a value rather than a location: an inline payload cannot be
/// attached as the kind it claims.
#[must_use]
pub(crate) fn inline_marker_refusal(kind: MediaMarkerKind) -> String {
    marker_refusal(&kind.noun(), INLINE_PAYLOAD_REASON)
}

/// Why a resolved local target cannot be delivered as `kind`, or `None` when it can: an image
/// must carry a native raster signature, any other kind only needs an existing regular file.
/// No size cap applies — an outgoing attachment's only size gate is the transport's own.
/// Delivery only: the desktop display layer never asks this.
#[must_use]
pub(crate) fn local_refusal(kind: MediaMarkerKind, path: &Path) -> Option<&'static str> {
    if kind == MediaMarkerKind::Image {
        image_target_refusal(path)
    } else {
        local_target_refusal(path)
    }
}

/// Why a resolved local target cannot be handed over, or `None` when it is an existing regular
/// file. A still-relative path names a file in a workspace the message has none of; an
/// unreadable one is reported as unreadable, not missing.
#[must_use]
fn local_target_refusal(path: &Path) -> Option<&'static str> {
    if path.is_relative() {
        return Some(NO_WORKSPACE_REASON);
    }
    match std::fs::metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(FILE_NOT_FOUND_REASON),
        Err(_) => Some(UNREADABLE_REASON),
        Ok(metadata) if metadata.is_dir() => Some(DIRECTORY_REASON),
        Ok(metadata) if !metadata.is_file() => Some(NOT_A_FILE_REASON),
        Ok(_) => None,
    }
}

/// Why a local image target cannot be delivered, or `None` when it is a regular file whose head
/// carries a native raster signature. Metadata plus a bounded head read — no decode. A file that
/// passes the signature without decoding is still delivered as a document, and the desktop shows
/// the viewer's placeholder.
#[must_use]
fn image_target_refusal(path: &Path) -> Option<&'static str> {
    use std::io::Read as _;
    if let Some(reason) = local_target_refusal(path) {
        return Some(reason);
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return Some(UNREADABLE_REASON);
    };
    let mut head = [0_u8; 16];
    let Ok(read) = file.read(&mut head) else {
        return Some(UNREADABLE_REASON);
    };
    if matches!(
        image::guess_format(&head[..read]),
        Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP)
    ) {
        None
    } else {
        Some(NOT_AN_IMAGE_REASON)
    }
}

/// Why a marker could not be delivered. Every surface quotes the same reasons, so a refusal
/// reads identically wherever it is shown.
pub(crate) const FILE_NOT_FOUND_REASON: &str = "file not found.";
pub(crate) const DIRECTORY_REASON: &str = "it is a directory.";
pub(crate) const NO_WORKSPACE_REASON: &str = "the message's workspace is unknown.";
pub(crate) const UNREADABLE_REASON: &str = "it could not be read.";
pub(crate) const INLINE_PAYLOAD_REASON: &str = "it carries an inline payload, not a file.";
pub(crate) const NOT_AN_IMAGE_REASON: &str = "it is not an image.";
pub(crate) const NOT_A_FILE_REASON: &str = "it is not a regular file.";
pub(crate) const INVALID_ADDRESS_REASON: &str = "it is not a valid address.";
pub(crate) const FILE_URL_REASON: &str = "an address is not a local file.";

/// The name a refusal quotes: the target's last path component, bounded so a malformed marker
/// cannot dump an arbitrary blob into the chat.
#[must_use]
pub(crate) fn marker_name(target: &str) -> String {
    crate::util::truncate(
        crate::util::file_name_or_path(without_credentials(target).as_ref()),
        60,
    )
}

/// The target as a durable record may keep it: a local path as written, without what only an
/// address carries (its query, fragment and userinfo). `None` for an inline payload — a `data:`
/// URI or its omitted-placeholder form names no path, and its text is a payload rather than
/// something to record. The caller bounds the result.
#[must_use]
pub(crate) fn recordable_target(target: &str) -> Option<Cow<'_, str>> {
    let target = target.trim();
    // An inline payload names no path, and every other reader takes a scheme as written, so a
    // `DATA:`-cased payload is guarded here too.
    if target.to_ascii_lowercase().starts_with("data:") || is_inline_payload(target) {
        return None;
    }
    Some(without_credentials(target))
}

/// `target` without what can carry credentials but names nothing: an address's query, fragment
/// and authority userinfo. Only an address has them — a local path keeps its own `#` and `?` —
/// and a protocol-relative one (`//user:pass@host/x`) carries them too.
fn without_credentials(target: &str) -> Cow<'_, str> {
    let (prefix, rest) = if let Some((scheme, rest)) = target.split_once("://") {
        (&target[..scheme.len() + 3], rest)
    } else if let Some(rest) = target.strip_prefix("//") {
        ("//", rest)
    } else {
        return Cow::Borrowed(target);
    };
    let without_value = rest.split(['?', '#']).next().unwrap_or(rest);
    let (authority, path) = without_value.split_once('/').unwrap_or((without_value, ""));
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    Cow::Owned(format!("{prefix}{host}/{path}"))
}

/// The refusal wording shared by the transport's post-accept failure notices and the desktop
/// display layer. A marker refused for delivery is left out of the delivered body, and the
/// conversation that wrote it is told instead — the note quotes the reason, not this line. The
/// desktop still draws this line where it refuses a marker itself.
#[must_use]
pub(crate) fn marker_refusal(name: &str, reason: &str) -> String {
    format!("Could not send \"{name}\": {reason}")
}

/// [`marker_refusal`] for a raw marker target, named by [`marker_name`].
#[must_use]
pub(crate) fn marker_refusal_for(target: &str, reason: &str) -> String {
    marker_refusal(&marker_name(target), reason)
}

/// True when `target` is a well-formed http(s) URL for an image reference. Rejects whitespace
/// anywhere, an empty or malformed authority, userinfo (`@`), a bad or empty port, and malformed
/// brackets. Deliberately STRICTER than the config endpoint check (which defers embedded-space
/// hosts to warmup).
#[must_use]
pub(crate) fn is_valid_remote_url(target: &str) -> bool {
    let Some(rest) = target
        .strip_prefix("https://")
        .or_else(|| target.strip_prefix("http://"))
    else {
        return false;
    };
    if rest.chars().any(char::is_whitespace) {
        return false;
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() || authority.starts_with(':') || authority.contains('@') {
        return false;
    }
    let Some((host, port)) = split_authority_host_port(authority) else {
        return false;
    };
    // A port, when present, must be non-empty, all decimal digits, within u16, and non-zero.
    // An absent port is `""` and passes.
    if !port.is_empty()
        && (!port.bytes().all(|b| b.is_ascii_digit()) || !port.parse::<u16>().is_ok_and(|x| x > 0))
    {
        return false;
    }
    // A bracketed host is `[...]` with a non-empty inner and exactly one pair of brackets; a
    // non-bracketed one is non-empty with no `:` (a second would be an extra port separator) and
    // no `[` (a stray unmatched bracket is malformed).
    if host.starts_with('[') {
        host.ends_with(']')
            && host.len() > 2
            && host.matches('[').count() == 1
            && host.matches(']').count() == 1
    } else {
        !host.is_empty() && !host.contains(':') && !host.contains('[')
    }
}

/// Split an authority into `(host, port)`; an absent port is `""`. A bracketed IPv6 host may
/// contain `:`, so its port is the `:` right after `]`. `None` for a malformed authority: an
/// empty trailing port, a `[...]` host not at the very start, or bare text after `]`.
fn split_authority_host_port(authority: &str) -> Option<(&str, &str)> {
    if let Some(close) = authority.find(']') {
        if !authority.starts_with('[') {
            return None;
        }
        let after = &authority[close + 1..];
        if after.is_empty() {
            return Some((&authority[..=close], ""));
        }
        let port = after.strip_prefix(':')?;
        if port.is_empty() {
            return None; // trailing `]:`
        }
        Some((&authority[..=close], port))
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) if !port.is_empty() => Some((host, port)),
            Some(_) => None, // trailing `:` with no port
            None => Some((authority, "")),
        }
    }
}

/// The base64 payload of a `data:...;base64,...` URI — everything after the LAST `;base64,`
/// token (RFC 2397 allows mediatype parameters before it). `None` when the string is not such a
/// URI, or when the payload exceeds [`MAX_DATA_URI_ENCODED_BYTES`] (refused from length alone).
/// Shared with the GUI viewer so the cap and extraction live in one place.
#[must_use]
pub(crate) fn data_uri_base64_payload(uri: &str) -> Option<&str> {
    let rest = uri.strip_prefix("data:")?;
    let payload_start = rest.rfind(";base64,")? + ";base64,".len();
    let payload = &rest[payload_start..];
    (payload.len() <= MAX_DATA_URI_ENCODED_BYTES).then_some(payload)
}

/// The declared raster subtype of a `data:image/<subtype>;base64,...` URI, or `None` for a
/// non-native or malformed subtype. The payload itself is validated by
/// [`data_uri_base64_payload`] and [`decode_base64_payload`].
fn declared_data_uri_format(target: &str) -> Option<image::ImageFormat> {
    let rest = target.strip_prefix("data:image/")?;
    let subtype = rest.split(';').next().unwrap_or("").trim();
    let declared = image::ImageFormat::from_extension(subtype)?;
    crate::util::image_format_native_label(declared)
        .is_some()
        .then_some(declared)
}

/// Parse a `data:image/<native-subtype>;base64,...` URI into its format and decoded bytes,
/// verifying the declared subtype matches the actual raster. `None` for a non-native or
/// malformed subtype, an over-cap payload, invalid base64, or a mismatch. Shared by the
/// provider's gate and the GUI's pixel decode so the check stays single-sourced.
fn parse_native_data_uri(uri: &str) -> Option<(image::ImageFormat, Vec<u8>)> {
    let declared = declared_data_uri_format(uri)?;
    let payload = data_uri_base64_payload(uri)?;
    let bytes = decode_base64_payload(payload)?;
    if image::guess_format(&bytes).ok()? != declared {
        return None;
    }
    Some((declared, bytes))
}

/// Decode such a URI into a renderable raster, verifying the declared subtype matches the bytes
/// so the GUI renderer and the provider's gate reach the same verdict. `None` also on a decode
/// failure under `limits`.
#[must_use]
pub(crate) fn decode_native_data_uri(
    uri: &str,
    limits: image::Limits,
) -> Option<(u32, u32, Vec<u8>)> {
    let (_, bytes) = parse_native_data_uri(uri)?;
    decode_raster_bytes(&bytes, limits)
}

/// The shared [`image::Limits`]: a dimension and alloc cap so a header bomb is refused before any
/// pixel buffer.
#[must_use]
pub(crate) fn is_native_data_uri(target: &str) -> bool {
    parse_native_data_uri(target)
        .is_some_and(|(_, bytes)| decode_raster(&bytes, raster_decode_limits()).is_some())
}

/// Decode standard or URL-safe base64, stripping ASCII whitespace. Rejects alphabet lookalikes
/// that are not decodable (truncated pad). Reached by the GUI viewer transitively, so a URL-safe
/// payload the provider injects renders too.
#[must_use]
fn decode_base64_payload(s: &str) -> Option<Vec<u8>> {
    let compact: Cow<'_, [u8]> = if s.as_bytes().iter().any(u8::is_ascii_whitespace) {
        Cow::Owned(
            s.bytes()
                .filter(|b| !b.is_ascii_whitespace())
                .collect::<Vec<u8>>(),
        )
    } else {
        Cow::Borrowed(s.as_bytes())
    };
    if compact.is_empty() {
        return None;
    }
    STANDARD
        .decode(compact.as_ref())
        .ok()
        .or_else(|| URL_SAFE.decode(compact.as_ref()).ok())
        .or_else(|| URL_SAFE_NO_PAD.decode(compact.as_ref()).ok())
}

/// Decode `bytes` as a native raster (PNG/JPEG/WebP) under `limits`: the single shared decode
/// path, gated through [`image_format_native_label`]. The failure is kept for a caller that
/// reports it (computer-use's portal capture names the cap that refused).
pub(crate) fn decode_raster_image(
    bytes: &[u8],
    limits: image::Limits,
) -> anyhow::Result<image::DynamicImage> {
    let format = image::guess_format(bytes)
        .map_err(|e| anyhow::anyhow!("not a recognisable image format ({e})"))?;
    if crate::util::image_format_native_label(format).is_none() {
        anyhow::bail!(
            "{} is not one of the formats this tool decodes (PNG, JPEG, WEBP)",
            format
                .extensions_str()
                .first()
                .copied()
                .unwrap_or("unknown")
        );
    }
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format);
    reader.limits(limits);
    Ok(reader.decode()?)
}

/// [`decode_raster_image`] for a caller that only needs to know the bytes decode.
fn decode_raster(bytes: &[u8], limits: image::Limits) -> Option<image::DynamicImage> {
    decode_raster_image(bytes, limits).ok()
}

/// Decode `bytes` as a native raster under `limits`, returning `(width, height, RGBA8 pixels)`.
/// Used where actual pixels are needed (the GUI viewer).
#[must_use]
pub(crate) fn decode_raster_bytes(
    bytes: &[u8],
    limits: image::Limits,
) -> Option<(u32, u32, Vec<u8>)> {
    let img = decode_raster(bytes, limits)?;
    let rgba = img.to_rgba8();
    Some((rgba.width(), rgba.height(), rgba.into_raw()))
}

/// Process-lifetime memo of local-raster decode results, keyed by (canonical path, mtime): a
/// stable path decodes at most once per process, while a replaced file gets a new mtime and is
/// re-evaluated. Bounded — once it exceeds [`LOCAL_RASTER_CACHE_MAX_ENTRIES`] it is fully
/// cleared, since re-decoding a handful of paths beats an unbounded map.
static LOCAL_RASTER_DECODE_CACHE: LazyLock<Mutex<HashMap<(PathBuf, std::time::SystemTime), bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Entry cap for [`LOCAL_RASTER_DECODE_CACHE`] before it is cleared.
const LOCAL_RASTER_CACHE_MAX_ENTRIES: usize = 4096;

/// True when `target` is an absolute path to an existing regular file that decodes as a genuine
/// PNG/JPEG/WebP raster. The single "is a real image" gate must be authoritative: an O(1) magic
/// sniff would accept a corrupt-but-magic-valid file, hence a real decode, memoized per
/// (canonical path, mtime).
#[must_use]
fn is_decodable_local_raster(target: &str) -> bool {
    let path = Path::new(target);
    if !path.is_absolute() {
        return false;
    }
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() || meta.len() > crate::util::INBOUND_IMAGE_MAX_INPUT_BYTES {
        return false;
    }
    let Ok(mtime) = meta.modified() else {
        return false;
    };
    // Canonicalize so the same file reached through different spellings (the /tmp →
    // /private/tmp symlink) shares one cache slot.
    let canonical = crate::util::canonical_or_self(path);
    let key = (canonical.clone(), mtime);
    {
        let cache = LOCAL_RASTER_DECODE_CACHE.lock().unwrap_poison();
        if let Some(&result) = cache.get(&key) {
            return result;
        }
    }
    // Blocking decode, deliberately outside the cache lock.
    let result = is_decodable_raster_file(&canonical);
    let mut cache = LOCAL_RASTER_DECODE_CACHE.lock().unwrap_poison();
    if cache.len() >= LOCAL_RASTER_CACHE_MAX_ENTRIES {
        cache.clear();
    }
    cache.insert(key, result);
    result
}

/// True when `path` reads as a native PNG/JPEG/WebP raster that decodes under the shared budget.
/// Reusing [`decode_raster`] keeps the native-format restriction and decode budget identical to
/// the data-URI / provider gate, so a local file the classifier accepts is also accepted once
/// inlined. Memoized by the classifier; must not run on the render thread.
#[must_use]
fn is_decodable_raster_file(path: &Path) -> bool {
    std::fs::read(path).is_ok_and(|bytes| decode_raster(&bytes, raster_decode_limits()).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;

    /// A real 1×1 PNG written to a unique path in the system temp dir.
    fn write_temp_png(tag: &str) -> std::path::PathBuf {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "mahbot_media_target_{tag}_{}.png",
            std::process::id()
        ));
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("test PNG must encode");
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&buf))
            .expect("test PNG must write");
        path
    }

    #[test]
    fn remote_url_valid_and_malformed() {
        assert_eq!(
            classify_media_image_target("https://example.com/b.jpg"),
            MediaTarget::RemoteUrl
        );
        assert_eq!(
            classify_media_image_target("http://127.0.0.1:8080/img.png"),
            MediaTarget::RemoteUrl
        );
        // Malformed: empty host, whitespace host, authority starting with ':'.
        assert_eq!(classify_media_image_target("http://"), MediaTarget::Invalid);
        assert_eq!(
            classify_media_image_target("https://exa mple.com/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("https:// example.com/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("http://:8080/x.png"),
            MediaTarget::Invalid
        );
        // Bad port (out of u16 range) and a bare trailing colon.
        assert_eq!(
            classify_media_image_target("https://example.com:99999/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("https://example.com:/x.png"),
            MediaTarget::Invalid
        );
        // Unterminated IPv6 bracket and a userinfo host (the `:pass` is not a port).
        assert_eq!(
            classify_media_image_target("https://[::1/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("https://user:pass@host/x.png"),
            MediaTarget::Invalid
        );
        // Userinfo without a password/port is equally malformed (previously a false positive:
        // the port check was skipped).
        assert_eq!(
            classify_media_image_target("https://user@host/x.png"),
            MediaTarget::Invalid
        );
        // A well-formed bracketed IPv6 host (with a port) is still valid.
        assert_eq!(
            classify_media_image_target("http://[::1]:8080/img.png"),
            MediaTarget::RemoteUrl
        );
        // Double-port authorities (unbracketed and bracketed) are malformed.
        assert_eq!(
            classify_media_image_target("https://host:8080:9999/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("http://[::1]:80:90/img.png"),
            MediaTarget::Invalid
        );
        // Degenerate bracketed hosts: empty IPv6 host, bare text after the host, and a host not
        // at the start.
        assert_eq!(
            classify_media_image_target("http://[]/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("http://[::1]foo:80/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("https://host[::1]:80/x.png"),
            MediaTarget::Invalid
        );
        // A well-formed bracketed IPv6 host with no port is still valid.
        assert_eq!(
            classify_media_image_target("https://[::1]/img.png"),
            MediaTarget::RemoteUrl
        );
        // A space anywhere in the URL is malformed.
        assert_eq!(
            classify_media_image_target("https://example.com/foo bar.png"),
            MediaTarget::Invalid
        );
        // A stray unmatched `[` in a non-bracketed host is malformed (the bracket-balance guard
        // must not regress).
        assert_eq!(
            classify_media_image_target("http://host[/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("http://host[foo:80"),
            MediaTarget::Invalid
        );
    }

    #[test]
    fn data_uri_native_and_fake() {
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("test PNG must encode");
        let uri = format!("data:image/png;base64,{}", STANDARD.encode(&buf));

        // A data-URI is NOT classified: it is an inline payload, validated by the dedicated
        // `is_native_data_uri` gate.
        assert_eq!(classify_media_image_target(&uri), MediaTarget::Invalid);

        // Valid base64 + real raster decode -> a native data-URI.
        assert!(is_native_data_uri(&uri));
        // Valid base64 but non-image bytes.
        assert!(!is_native_data_uri("data:image/png;base64,abcd"));
        // Truncated PNG (magic + IHDR, no IDAT): a data-URI must decode to a genuine raster, so
        // the provider never forwards a magic-valid-but-undecodable part.
        let truncated = {
            let payload = uri
                .strip_prefix("data:image/png;base64,")
                .expect("tiny png uri");
            let bytes = STANDARD
                .decode(payload.as_bytes())
                .expect("tiny png base64");
            let t = &bytes[..bytes.len().min(24)];
            format!("data:image/png;base64,{}", STANDARD.encode(t))
        };
        assert!(!is_native_data_uri(&truncated));
        // Unsupported subtype (gif decoder not compiled) -> not native.
        assert!(!is_native_data_uri("data:image/gif;base64,R0lGOD"));
        // Declared subtype must match the bytes: a data:image/png holding JPEG is rejected, since
        // a provider may decode strictly by the declared MIME.
        let mut jbuf = Vec::new();
        image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]))
            .write_to(
                &mut std::io::Cursor::new(&mut jbuf),
                image::ImageFormat::Jpeg,
            )
            .expect("test JPEG must encode");
        let mismatched = format!("data:image/png;base64,{}", STANDARD.encode(&jbuf));
        assert!(!is_native_data_uri(&mismatched));
    }

    #[test]
    fn data_uri_base64_payload_extracts_and_caps() {
        let cases: &[(&str, Option<&str>)] = &[
            ("data:image/png;base64,AAAA", Some("AAAA")),
            ("data:image/png;charset=utf-8;base64,BBBB", Some("BBBB")),
            ("/tmp/photo.png", None),
            ("https://example.com/img.png", None),
            ("data:image/png,raw-bytes", None),
            ("data:image/png;base64,", Some("")),
        ];
        for (uri, expected) in cases {
            assert_eq!(data_uri_base64_payload(uri), *expected, "case: {uri}");
        }
        // An over-cap payload is refused from length alone, never decoded.
        let over = format!(
            "data:image/png;base64,{}",
            "A".repeat(MAX_DATA_URI_ENCODED_BYTES + 1)
        );
        assert_eq!(data_uri_base64_payload(&over), None);
    }

    /// A local target is a path the classifier can decode. A placeholder, a relative basename, a
    /// missing path, a non-image file and a magic-valid-but-truncated raster are all refused — the
    /// classifier requires a real decode, so nothing renders a broken image or emits a junk
    /// data-URI.
    #[test]
    fn local_targets_classify_by_a_real_decode() {
        let png = write_temp_png("valid");
        let dir = png.parent().expect("temp path has a parent");
        let text = dir.join(format!(
            "mahbot_media_target_{}_notimage.txt",
            std::process::id()
        ));
        std::fs::write(&text, b"top secret").unwrap();
        let corrupt = dir.join(format!(
            "mahbot_media_target_{}_trunc.png",
            std::process::id()
        ));
        let bytes = std::fs::read(&png).unwrap();
        std::fs::write(&corrupt, &bytes[..bytes.len().min(24)]).unwrap();

        assert_eq!(
            classify_media_image_target(png.to_str().unwrap()),
            MediaTarget::LocalImage
        );
        for invalid in [
            "",
            "   ",
            "...",
            // Relative basename is never a valid local image target.
            "photo.png",
            "/tmp/definitely_missing_mahbot.png",
            text.to_str().unwrap(),
            corrupt.to_str().unwrap(),
        ] {
            assert_eq!(
                classify_media_image_target(invalid),
                MediaTarget::Invalid,
                "target: {invalid:?}"
            );
        }

        let _ = std::fs::remove_file(&png);
        let _ = std::fs::remove_file(&text);
        let _ = std::fs::remove_file(&corrupt);
    }

    #[test]
    fn an_oversized_raster_is_not_refused_as_a_target() {
        // The transport's own upload limit is the only size gate on an outgoing attachment, so
        // the inbound decode budget must not refuse one. Sparse, so the test costs no disk.
        let png = write_temp_png("oversized");
        std::fs::File::options()
            .write(true)
            .open(&png)
            .and_then(|file| file.set_len(crate::util::INBOUND_IMAGE_MAX_INPUT_BYTES + 1))
            .expect("a sparse oversized raster");
        assert_eq!(local_refusal(MediaMarkerKind::Image, &png), None);
        let _ = std::fs::remove_file(&png);
    }

    #[test]
    fn refusal_names_hide_credentials() {
        // A refusal quotes a bounded name, never the target's payload: an address's query,
        // fragment and userinfo can all carry a secret.
        assert_eq!(
            marker_name("https://cdn.example.com/a/poster.png?token=s3cret"),
            "poster.png"
        );
        assert_eq!(
            marker_name("https://user:pass@cdn.example.com/poster.png"),
            "poster.png"
        );
        assert_eq!(
            marker_name("https://user:pass@cdn.example.com#s3cret"),
            "cdn.example.com"
        );
        assert_eq!(marker_name("generated/some-poster.png"), "some-poster.png");
        // A local name keeps its own `#` and `?`: only an address has a payload.
        assert_eq!(marker_name("generated/poster#2.png"), "poster#2.png");

        // A record keeps the whole path, and strips the same address parts: a scheme-written
        // target is still a path here, so a credential in it must not be stored.
        assert_eq!(
            recordable_target("/tmp/a.png").as_deref(),
            Some("/tmp/a.png")
        );
        assert_eq!(
            recordable_target("HTTPS://user:pass@cdn.example.com/a.png?token=s3cret").as_deref(),
            Some("HTTPS://cdn.example.com/a.png")
        );
        // A protocol-relative target carries the same userinfo.
        assert_eq!(
            recordable_target("//user:pass@cdn.example.com/a.png?token=s3cret").as_deref(),
            Some("//cdn.example.com/a.png")
        );
        // An inline payload names no path, in any casing.
        assert_eq!(recordable_target("data:image/png;base64,AAAA"), None);
        assert_eq!(recordable_target("DATA:image/png;base64,AAAA"), None);
    }

    #[test]
    fn a_mistyped_directory_resolves_to_the_file_it_meant() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let generated = root.join("userspaces").join("lyuba").join("generated");
        std::fs::create_dir_all(&generated).unwrap();
        let png = generated.join("shot.png");
        std::fs::write(&png, b"png").unwrap();

        // The typo is in an ancestor directory, and the rest of the path exists as written.
        let ancestor_typo = root
            .join("userseas")
            .join("lyuba")
            .join("generated")
            .join("shot.png");
        assert_eq!(repaired_local_target(&ancestor_typo), Some(png.clone()));

        // The same repair one level down.
        let parent_typo = root
            .join("userspaces")
            .join("lyuba")
            .join("generatd")
            .join("shot.png");
        assert_eq!(repaired_local_target(&parent_typo), Some(png.clone()));

        // A path that exists as written is never repaired, and the last component is the file's
        // own name: a similar name is a different file.
        assert_eq!(repaired_local_target(&png), None);
        assert_eq!(repaired_local_target(&generated.join("shot.pgn")), None);
        // A trailing separator names a directory: repairing it would deliver a file of that name.
        assert_eq!(
            repaired_local_target(&PathBuf::from(format!("{}/", png.display()))),
            None
        );
        // A `..` after the mistyped component names a directory the substitution did not produce.
        assert_eq!(
            repaired_local_target(&root.join("userseas").join("..").join("shot.png")),
            None
        );

        // A component with no near neighbour, and a repair whose remainder is not there.
        assert_eq!(
            repaired_local_target(&root.join("zzzz").join("shot.png")),
            None
        );
        assert_eq!(
            repaired_local_target(&root.join("userseas").join("lyuba").join("gone.png")),
            None
        );

        // An extra component is not a mistyped one: repairing it leaves the rest of the path
        // missing, so nothing is guessed.
        assert_eq!(
            repaired_local_target(
                &root
                    .join("userseas")
                    .join("userspaces")
                    .join("lyuba")
                    .join("generated")
                    .join("shot.png")
            ),
            None
        );

        // Two equally near directories could be meant: nothing is guessed. Both candidates hold
        // the file, so the ambiguity guard alone is what refuses the repair.
        std::fs::create_dir_all(root.join("alpha")).unwrap();
        std::fs::create_dir_all(root.join("alphb")).unwrap();
        std::fs::write(root.join("alpha").join("shot.png"), b"png").unwrap();
        std::fs::write(root.join("alphb").join("shot.png"), b"png").unwrap();
        assert_eq!(
            repaired_local_target(&root.join("alphx").join("shot.png")),
            None
        );

        // A directory reached through a link is a candidate like any other directory: the entry is
        // followed, and the repaired path resolves through the link.
        #[cfg(unix)]
        {
            let storage = root.join("storage");
            std::fs::create_dir_all(&storage).unwrap();
            std::fs::write(storage.join("shot.png"), b"png").unwrap();
            std::os::unix::fs::symlink(&storage, root.join("pictures")).unwrap();
            assert_eq!(
                repaired_local_target(&root.join("picturrs").join("shot.png")),
                Some(root.join("pictures").join("shot.png"))
            );
        }
    }

    #[test]
    fn workspace_files_decide_bare_word_references() {
        // Existence decides whether a bare word is a reference: `[FILE:Makefile]` names the
        // workspace's file, so it must not fall back to prose just because its name has no
        // extension (or an unusual one).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Makefile"), b"all:").unwrap();
        std::fs::write(dir.path().join("poster.thumbnail"), b"x").unwrap();
        let roots = [dir.path().to_path_buf()];
        for name in ["Makefile", "poster.thumbnail"] {
            assert_eq!(
                resolve_marker_target(name, &roots),
                MarkerTarget::Local(dir.path().join(name)),
                "{name} is a file the workspace has"
            );
        }
        // Nothing by that name anywhere: a bare word stays prose, while a path-shaped target
        // still names a (missing) file.
        assert_eq!(
            resolve_marker_target("path", &roots),
            MarkerTarget::NotAReference
        );
        assert_eq!(
            resolve_marker_target("plot.png", &roots),
            MarkerTarget::Local(dir.path().join("plot.png"))
        );
    }
}
