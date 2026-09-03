//! Shared classifier for `[IMAGE:...]` media-marker targets.
//!
//! Consumers that must decide whether a target is a "real image" use
//! [`classify_media_image_target`] (Telegram and enrichment, which offload the
//! blocking local-file decode off the async path); an invalid target stays inert
//! literal text. Two consumers deliberately do NOT route through it: the GUI
//! render path (a classifier decode would run on the render thread —
//! `replace_media_markers` uses a cheap O(1) structural gate and the viewer
//! decodes boundedly / downscales itself) and the provider's agent-context
//! scanner (`compatible.rs`, which must never read a local file during request
//! serialization and calls [`is_native_data_uri`] directly). No
//! workspace-boundary containment — a valid raster anywhere on disk is valid.
//! Local files are validated by a real raster decode (memoized per canonical
//! path + mtime); remote URLs by well-formedness. A `data:...` URI is NOT
//! classified here — it is an inline payload validated by its own
//! [`is_native_data_uri`] gate, which only the provider uses directly; other
//! consumers route through the classifier, which rejects a `data:` target
//! cheaply before any decode.

use crate::util::UnwrapPoison;
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE, URL_SAFE_NO_PAD},
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// A classified `[IMAGE:...]` marker target.
///
/// The variant is what every production consumer matches on; the validated
/// value itself (canonical path / URL / URI) is re-read from the marker text
/// by the caller, so no payload is carried here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaTarget {
    /// Absolute path to an existing regular file that decodes as a genuine
    /// PNG/JPEG/WebP raster — a corrupt-but-magic-valid file is not accepted.
    LocalImage,
    /// Well-formed http(s) URL for an image reference.
    RemoteUrl,
    /// Anything else — the marker stays as inert literal text. (A `data:...`
    /// URI is NOT classified here: it is an inline payload validated by
    /// [`is_native_data_uri`], which the provider uses directly.)
    Invalid,
}

/// Encoded-payload ceiling for any `data:...;base64,...` URI (shared by the
/// classifier and the GUI viewer). Over-cap payloads are refused from length
/// alone — never base64-decoded — so neither the render thread nor an async
/// worker buffers a huge payload.
pub(crate) const MAX_DATA_URI_ENCODED_BYTES: usize = 20 * 1024 * 1024;

/// Shared raster-decode allocation budget (data-URI classifier and GUI render
/// downscale). Bounded so a header-bomb target is refused, but generous enough
/// that a legitimate tall screenshot still decodes.
const CLASSIFY_DECODE_MAX_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// Longest side (px) the shared raster decode will accept — rejects a dimension
/// bomb from the header before any pixel buffer is allocated.
const CLASSIFY_DECODE_MAX_DIMENSION_PX: u32 = 16384;

/// The shared generous raster-decode [`image::Limits`]: a dimension and alloc
/// cap so a header bomb is refused before any pixel buffer, while a legitimate
/// tall screenshot still decodes. Used by the classifier's data-URI branch AND
/// the GUI render decode + downscale step. Built via `Default` + field mutation
/// (`Limits` is `#[non_exhaustive]`).
#[must_use]
pub(crate) fn raster_decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(CLASSIFY_DECODE_MAX_ALLOC_BYTES);
    limits.max_image_width = Some(CLASSIFY_DECODE_MAX_DIMENSION_PX);
    limits.max_image_height = Some(CLASSIFY_DECODE_MAX_DIMENSION_PX);
    limits
}

/// Classify a raw `[IMAGE:...]` marker target. A `data:...` URI is not a target
/// this classifier handles (it is an inline payload validated by
/// [`is_native_data_uri`]); pass it there instead.
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

/// True when `target` is a well-formed http(s) URL for an image reference.
/// Rejects whitespace anywhere, an empty or malformed authority, userinfo
/// (`@`), a bad or empty port, and malformed brackets. Deliberately STRICTER
/// than the config endpoint check (which defers embedded-space hosts to
/// warmup) — a space, a bad/duplicate port, or userinfo is malformed for an
/// image reference and must stay inert text.
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
    // A port, when present, must be non-empty, all decimal digits, within u16,
    // and non-zero. An absent port is `""` and passes.
    if !port.is_empty()
        && (!port.bytes().all(|b| b.is_ascii_digit()) || !port.parse::<u16>().is_ok_and(|x| x > 0))
    {
        return false;
    }
    // A bracketed host is `[...]` with a non-empty inner and exactly one pair of
    // brackets; a non-bracketed host is non-empty with no `:` (a second `:`
    // would have been an extra port separator) and no `[` (a stray unmatched
    // bracket — e.g. `host[` — is malformed and stays inert text).
    if host.starts_with('[') {
        host.ends_with(']')
            && host.len() > 2
            && host.matches('[').count() == 1
            && host.matches(']').count() == 1
    } else {
        !host.is_empty() && !host.contains(':') && !host.contains('[')
    }
}

/// Split an authority into `(host, port)`; an absent port is `""`. A bracketed
/// IPv6 host may contain `:` inside `[...]`, so its port (if any) is the `:`
/// right after `]`; a non-bracketed host has at most one `:port`. Returns
/// `None` for a malformed authority: a trailing-colon empty port (`host:` or
/// `[...]:`), a `[...]` host not at the very start (`host[::1]:80`), or bare
/// text after `]` (`[::1]foo:80`).
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

/// Extract the base64 payload of a `data:...;base64,...` URI — everything after
/// the LAST `;base64,` token (RFC 2397 allows mediatype parameters before the
/// marker, e.g. `data:image/png;charset=utf-8;base64,...`). `None` when the
/// string is not a data URI with a base64 payload, or when the payload exceeds
/// [`MAX_DATA_URI_ENCODED_BYTES`] (refused from length alone, never decoded).
/// Shared with the GUI viewer so the cap and extraction live in one place.
#[must_use]
pub(crate) fn data_uri_base64_payload(uri: &str) -> Option<&str> {
    let rest = uri.strip_prefix("data:")?;
    let payload_start = rest.rfind(";base64,")? + ";base64,".len();
    let payload = &rest[payload_start..];
    (payload.len() <= MAX_DATA_URI_ENCODED_BYTES).then_some(payload)
}

/// The declared raster subtype of a `data:image/<subtype>;base64,...` URI, or
/// `None` for a non-native or malformed subtype. (The base64 payload itself is
/// validated by [`data_uri_base64_payload`] and [`decode_base64_payload`].)
fn declared_data_uri_format(target: &str) -> Option<image::ImageFormat> {
    let rest = target.strip_prefix("data:image/")?;
    let subtype = rest.split(';').next().unwrap_or("").trim();
    let declared = image::ImageFormat::from_extension(subtype)?;
    crate::util::image_format_native_label(declared)
        .is_some()
        .then_some(declared)
}

/// Parse a `data:image/<native-subtype>;base64,...` URI into its declared
/// format and decoded bytes, verifying the declared subtype matches the actual
/// raster format. `None` for a non-native / malformed subtype, an over-cap
/// payload, invalid base64, or a declared/actual mismatch. Shared by the
/// provider's validity gate ([`is_native_data_uri`]) and the GUI's pixel
/// decode ([`decode_native_data_uri`]) so the declared/actual check stays
/// single-sourced.
fn parse_native_data_uri(uri: &str) -> Option<(image::ImageFormat, Vec<u8>)> {
    let declared = declared_data_uri_format(uri)?;
    let payload = data_uri_base64_payload(uri)?;
    let bytes = decode_base64_payload(payload)?;
    if image::guess_format(&bytes).ok()? != declared {
        return None;
    }
    Some((declared, bytes))
}

/// Decode a `data:image/<native-subtype>;base64,...` URI into a renderable
/// raster, verifying the declared subtype matches the actual bytes so the GUI
/// renderer and the provider's image gate reach the same verdict. Returns
/// `None` for a non-native or malformed subtype, an over-cap payload, invalid
/// base64, a declared/actual mismatch, or a decode failure under `limits`.
#[must_use]
pub(crate) fn decode_native_data_uri(
    uri: &str,
    limits: image::Limits,
) -> Option<(u32, u32, Vec<u8>)> {
    let (_, bytes) = parse_native_data_uri(uri)?;
    decode_raster_bytes(&bytes, limits)
}

/// True when `target` is a `data:image/<native-subtype>;base64,...` URI whose
/// payload base64-decodes to a genuine raster of that declared subtype. A
/// `data:image/png` holding JPEG bytes is mismatched (declared != actual) and
/// rejected, so the provider never forwards a MIME it may reject. Shared
/// [`parse_native_data_uri`] + [`decode_raster`] path (no RGBA8 conversion) so
/// the provider gate's verdict matches the GUI pixel decode's verdict. The GUI
/// render thread never calls this (it hands `data:` markers to the viewer).
#[must_use]
pub(crate) fn is_native_data_uri(target: &str) -> bool {
    parse_native_data_uri(target)
        .is_some_and(|(_, bytes)| decode_raster(&bytes, raster_decode_limits()).is_some())
}

/// Decode standard or URL-safe base64, stripping ASCII whitespace. Rejects
/// alphabet lookalikes that are not actually decodable (`...`, truncated pad).
/// Shared with the GUI viewer so its data-URI base64 decode matches the
/// classifier/provider (a URL-safe payload the provider injects must render too).
#[must_use]
pub(crate) fn decode_base64_payload(s: &str) -> Option<Vec<u8>> {
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

/// Decode `bytes` as a native raster (PNG/JPEG/WebP) under `limits`, returning
/// the decoded image. The single shared decode path used by a validity check
/// (the classifier and the provider's data-URI gate, which only need to know
/// the bytes decode) and by the GUI viewer, which converts to RGBA8 for an iced
/// handle. `None` for a non-native format or a decode exceeding `limits`.
#[must_use]
fn decode_raster(bytes: &[u8], limits: image::Limits) -> Option<image::DynamicImage> {
    let format = image::guess_format(bytes).ok()?;
    crate::util::image_format_native_label(format)?;
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format);
    reader.limits(limits);
    reader.decode().ok()
}

/// Decode `bytes` as a native raster under `limits`, returning
/// `(width, height, RGBA8 pixels)` on success. Used where actual pixels are
/// needed (the GUI viewer); a validity-only consumer uses [`decode_raster`]
/// so it never pays for the RGBA8 conversion.
#[must_use]
pub(crate) fn decode_raster_bytes(
    bytes: &[u8],
    limits: image::Limits,
) -> Option<(u32, u32, Vec<u8>)> {
    let img = decode_raster(bytes, limits)?;
    let rgba = img.to_rgba8();
    Some((rgba.width(), rgba.height(), rgba.into_raw()))
}

/// Process-lifetime memo of local-raster decode results, keyed by
/// (canonical path, mtime). A stable path decodes at most once per process — so
/// the GUI render thread / async worker pays for a single real decode per unique
/// image rather than repeatedly on every repaint/check — while a file replaced
/// on disk gets a new mtime and is re-evaluated. Bounded: once it exceeds
/// [`LOCAL_RASTER_CACHE_MAX_ENTRIES`] it is fully cleared (a degenerate file
/// churn is the only way to grow it, and re-decoding a handful of paths is
/// cheaper than a permanent unbounded map).
static LOCAL_RASTER_DECODE_CACHE: LazyLock<Mutex<HashMap<(PathBuf, std::time::SystemTime), bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Entry cap for [`LOCAL_RASTER_DECODE_CACHE`] before it is cleared. A daemon
/// that touches many unique temp/agent images stays bounded; a map clear just
/// costs a handful of re-decodes for the next unique paths.
const LOCAL_RASTER_CACHE_MAX_ENTRIES: usize = 4096;

/// True when `target` is an absolute path to an existing regular file that
/// decodes as a genuine PNG/JPEG/WebP raster. The classifier is the single
/// "is a real image" gate for every consumer, so it must be authoritative: an
/// O(1) magic sniff is not enough (a corrupt-but-magic-valid file would be
/// accepted), hence a real decode via [`is_decodable_raster_file`], memoized per
/// (canonical path, mtime) in [`LOCAL_RASTER_DECODE_CACHE`].
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
    // Canonicalize so the same file reached via different spellings (e.g. the
    // /tmp -> /private/tmp symlink) shares a single cache slot.
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
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

/// Decode validity for a local file: true when `path` reads as a native
/// PNG/JPEG/WebP raster that decodes under the shared [`raster_decode_limits`]
/// budget. Reusing the single [`decode_raster`] pipeline keeps the
/// native-format restriction and the decode budget identical to the data-URI /
/// provider gate, so a local file the classifier accepts is also accepted by the
/// provider once it is inlined as a data-URI (no mismatch). Memoized per
/// canonical path + mtime by the classifier; must not run on the render thread.
#[must_use]
fn is_decodable_raster_file(path: &Path) -> bool {
    std::fs::read(path).is_ok_and(|bytes| decode_raster(&bytes, raster_decode_limits()).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;

    /// A real 1×1 PNG file written to a unique path in the system temp dir.
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
        // Malformed: empty host, whitespace host (interior + leading), authority
        // starting with ':'.
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
        // Unterminated IPv6 bracket and a userinfo host (the `:pass` is not a
        // port) are malformed for an image reference.
        assert_eq!(
            classify_media_image_target("https://[::1/x.png"),
            MediaTarget::Invalid
        );
        assert_eq!(
            classify_media_image_target("https://user:pass@host/x.png"),
            MediaTarget::Invalid
        );
        // Userinfo without a password/port is equally malformed for an image
        // reference (previously a false positive: the port check was skipped).
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
        // Degenerate bracketed hosts are malformed: empty IPv6 host, bare text
        // after the host, and a host not at the start.
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
        // A space anywhere in the URL (path/query) is malformed.
        assert_eq!(
            classify_media_image_target("https://example.com/foo bar.png"),
            MediaTarget::Invalid
        );
        // A stray unmatched `[` in a non-bracketed host is malformed and stays
        // inert text (the bracket-balance guard must not regress: `host[` and
        // `host[foo:80` are not valid remote URL hosts).
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

        // A data-URI is NOT classified by `classify_media_image_target` (it is
        // an inline payload, not a local path / URL); its validity is the
        // dedicated `is_native_data_uri` gate, which the provider uses directly.
        assert_eq!(classify_media_image_target(&uri), MediaTarget::Invalid);

        // Valid base64 + real raster decode -> a native data-URI.
        assert!(is_native_data_uri(&uri));
        // Valid base64 but non-image bytes.
        assert!(!is_native_data_uri("data:image/png;base64,abcd"));
        // Truncated PNG (valid magic + IHDR, no IDAT): a data-URI must decode to
        // a genuine raster — a truncated payload is rejected, so the provider
        // never forwards a magic-valid-but-undecodable image part.
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
        // Declared subtype must match the actual bytes: a data:image/png holding
        // JPEG bytes is rejected (a provider may decode strictly by the declared
        // MIME, so a mismatched payload would be the same class of 400).
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

    #[test]
    fn local_raster_and_non_image() {
        let png = write_temp_png("valid");
        let text = std::env::temp_dir().join(format!(
            "mahbot_media_target_{}_notimage.txt",
            std::process::id()
        ));
        std::fs::write(&text, b"top secret").unwrap();

        assert_eq!(
            classify_media_image_target(png.to_str().unwrap()),
            MediaTarget::LocalImage
        );
        assert_eq!(
            classify_media_image_target(text.to_str().unwrap()),
            MediaTarget::Invalid
        );

        let _ = std::fs::remove_file(&png);
        let _ = std::fs::remove_file(&text);
    }

    #[test]
    fn placeholders_and_relative_are_invalid() {
        assert_eq!(classify_media_image_target(""), MediaTarget::Invalid);
        assert_eq!(classify_media_image_target("   "), MediaTarget::Invalid);
        assert_eq!(classify_media_image_target("..."), MediaTarget::Invalid);
        // Relative basename is never a valid local image target.
        assert_eq!(
            classify_media_image_target("photo.png"),
            MediaTarget::Invalid
        );
        // Nonexistent absolute path.
        assert_eq!(
            classify_media_image_target("/tmp/definitely_missing_mahbot.png"),
            MediaTarget::Invalid
        );
    }

    #[test]
    fn truncated_local_raster_is_invalid() {
        // A local file whose leading bytes sniff as PNG (valid magic + IHDR) but
        // whose payload is truncated must classify as INVALID, not LocalImage:
        // the classifier is the single authoritative gate and requires a real
        // decode, so a corrupt-but-magic-valid file is never accepted (it would
        // render as a broken image / emit a junk data-URI downstream).
        let png = write_temp_png("trunc");
        let bytes = std::fs::read(&png).unwrap();
        let truncated = &bytes[..bytes.len().min(24)];
        let corrupt = std::env::temp_dir().join(format!(
            "mahbot_media_target_{}_trunc.png",
            std::process::id()
        ));
        std::fs::write(&corrupt, truncated).unwrap();
        assert_eq!(
            classify_media_image_target(corrupt.to_str().unwrap()),
            MediaTarget::Invalid
        );
        let _ = std::fs::remove_file(&png);
        let _ = std::fs::remove_file(&corrupt);
    }
}
