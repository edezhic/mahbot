pub(crate) mod catalog_cache;
pub(crate) mod desktop_entry;
pub(crate) mod disk;
#[cfg(target_os = "linux")]
pub(crate) mod display_server;
pub(crate) mod error;
pub(crate) mod folder_name;
pub(crate) mod html;
pub(crate) mod http;
pub(crate) mod json;
pub(crate) mod leftover_channels;
pub mod lock;
pub(crate) mod macros;
pub(crate) mod managed_bin;
pub(crate) mod markdown;
pub(crate) mod media_target;
pub(crate) mod model_state;
pub(crate) mod owner_path;
pub(crate) mod provenance;
#[cfg(windows)]
pub(crate) mod registry;
#[cfg(test)]
pub(crate) mod test;
pub(crate) mod tree_sitter;
pub(crate) mod upload_bridge;

use directories::UserDirs;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use strum::IntoEnumIterator;

use anyhow::{Context as _, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use tracing::error;

/// Extension trait to unwrap poisoned lock results, replacing the long form with
/// `.unwrap_poison()`.
pub trait UnwrapPoison {
    type Inner;
    /// Unwrap the lock result, recovering the inner value even if the lock is poisoned.
    #[must_use]
    fn unwrap_poison(self) -> Self::Inner;
}

impl<T> UnwrapPoison for Result<T, std::sync::PoisonError<T>> {
    type Inner = T;
    fn unwrap_poison(self) -> T {
        self.unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
pub(crate) enum MediaMarkerKind {
    Image,
    Audio,
    Video,
    File,
}

impl MediaMarkerKind {
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::Image => "IMAGE",
            Self::Audio => "AUDIO",
            Self::Video => "VIDEO",
            Self::File => "FILE",
        }
    }

    pub(crate) fn noun(self) -> String {
        self.token().to_ascii_lowercase()
    }
}

/// Matches `[IMAGE:path]`, `[AUDIO:path]`, `[VIDEO:path]`, or `[FILE:path]` markers in message
/// content, in any casing. The kind list is generated from [`MediaMarkerKind`], so every pipeline
/// matches exactly the kinds the system knows; casing is insensitive because no prompt tells the
/// model it matters. **Invariant:** `enrich_message` strips only the kinds it consumes (AUDIO,
/// VIDEO) and passes every other kind through untouched; IMAGE markers are preserved for native
/// image-part integration via `to_message_content()`, FILE markers because they carry a workspace
/// file handle the routed role uses.
pub(crate) static MEDIA_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    let kinds = MediaMarkerKind::iter()
        .map(MediaMarkerKind::token)
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!(r"(?i)\[(?P<kind>{kinds}):(?P<path>[^\]\r\n]+)\]"))
        .expect("MEDIA_MARKER_RE must compile")
});

#[must_use]
pub(crate) fn has_media_marker(text: &str, kind: MediaMarkerKind) -> bool {
    let prefix = format!("[{}:", kind.token());
    text.as_bytes()
        .windows(prefix.len())
        .any(|window| window.eq_ignore_ascii_case(prefix.as_bytes()))
}

/// Extract the kind and `path` named groups from a [`MEDIA_MARKER_RE`] capture: the
/// [`MediaMarkerKind`] and the path slice borrowed from the original haystack.
///
/// # Panics
///
/// If the `kind` or `path` group is missing, or the token names no kind — impossible, since the
/// pattern is generated from [`MediaMarkerKind`] and the two cannot drift.
#[must_use]
pub(crate) fn parse_media_marker<'h>(caps: &regex::Captures<'h>) -> (MediaMarkerKind, &'h str) {
    let token = caps
        .name("kind")
        .expect("parse_media_marker: expected 'kind' group")
        .as_str();
    let kind = MediaMarkerKind::iter()
        .find(|kind| kind.token().eq_ignore_ascii_case(token))
        .expect("the pattern is generated from these kinds, so one of them matched");
    let path = caps
        .name("path")
        .expect("parse_media_marker: expected 'path' group")
        .as_str();
    (kind, path)
}

/// The media markers a message carries outside markdown code, as the kind and the path slice —
/// inside code a marker is source text the user reads, not an attachment.
#[must_use]
pub(crate) fn markers_outside_code(text: &str) -> Vec<(MediaMarkerKind, &str)> {
    let code = markdown::scan(text).code;
    MEDIA_MARKER_RE
        .captures_iter(text)
        .filter(|caps| {
            let whole = caps.get(0).expect("the whole match is always present");
            !code.iter().any(|range| range.contains(&whole.start()))
        })
        .map(|caps| parse_media_marker(&caps))
        .collect()
}

/// Placeholder that replaces an inline image data URI wherever bounded, displayable text is
/// persisted (see [`strip_data_uris`]). `gui::media_markers` recognises it by this prefix alone;
/// the byte count that follows is display-only.
pub(crate) const DATA_URI_OMITTED_PREFIX: &str = "<data-uri omitted";

/// The note a picture leaves behind when it is not delivered: the marker's byte size, so the model
/// can see a picture was attached at this point and how big it was. Shared by [`strip_data_uris`]
/// and the request picture cap ([`crate::session::image_cap`]), so there is one omission shape.
#[must_use]
pub(crate) fn data_uri_omitted_note(marker: &str) -> String {
    format!(
        "[IMAGE:{DATA_URI_OMITTED_PREFIX} ({} bytes)>]",
        marker.len()
    )
}

/// Strip unbounded data-URI payloads from `[IMAGE:...]` media markers, replacing them with a
/// byte-count placeholder.
///
/// Data URIs are the only unbounded payloads in history, and stripping them keeps compaction dumps
/// under the read tool's file-size cap; a path-based marker stays verbatim because the file it points
/// at stays openable, and a non-IMAGE kind carrying a `data:image/` path is a malformed marker that
/// is left alone.
///
/// Called for compaction dumps and, through [`crate::channels::persist_content`], for the enriched
/// content of a message whose markers name inbound attachments (document ingestion produces
/// `[IMAGE:data:...]` parts that must not reach chat history).
pub(crate) fn strip_data_uris(text: &str) -> String {
    MEDIA_MARKER_RE
        .replace_all(text, |caps: &regex::Captures| {
            let (kind, path) = parse_media_marker(caps);
            if kind == MediaMarkerKind::Image && path.starts_with("data:image/") {
                data_uri_omitted_note(caps.get_match().as_str())
            } else {
                caps.get_match().as_str().to_string()
            }
        })
        .into_owned()
}

pub(crate) fn listed<T: std::fmt::Display>(values: &[T], conjunction: &str) -> String {
    match values {
        [] => String::new(),
        [only] => only.to_string(),
        [rest @ .., last] => format!(
            "{} {conjunction} {last}",
            rest.iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod data_uri_strip_tests {
    use super::{DATA_URI_OMITTED_PREFIX, strip_data_uris};

    #[test]
    fn strips_only_image_data_uris() {
        assert_eq!(
            strip_data_uris("[IMAGE:data:image/jpeg;base64,AAAA]"),
            format!("[IMAGE:{DATA_URI_OMITTED_PREFIX} (35 bytes)>]")
        );
        assert_eq!(strip_data_uris("[IMAGE:/tmp/x.png]"), "[IMAGE:/tmp/x.png]");
        assert_eq!(
            strip_data_uris("[AUDIO:data:image/png;base64,AAAA]"),
            "[AUDIO:data:image/png;base64,AAAA]"
        );
    }
}

/// Provenance tag prepended to every synthetic User-role image message the agent loop injects, so a
/// vision-capable model can tell a tool-injected image from a user upload. Consumed by the
/// image-strip path so a stripped message reads cleanly.
pub(crate) const INJECTED_IMAGE_TAG: &str = "<injected-tool-result-image>";

/// Compose the synthetic User-role message carrying the images one round attached: the provenance
/// tag, then one marker per picture, in the order they were attached. Every marker is an
/// `[IMAGE:…]`: the conversation keeps the picture like any other it was sent.
///
/// One message per round, not per image: the provider-input-image strip rewrites only the most
/// recent User-role message, so batching keeps every image of the round clearable when the provider
/// rejects one — including the images it did not object to, since that request is the one that
/// failed.
#[must_use]
pub(crate) fn injected_image_user_message(markers: &[String]) -> String {
    let mut message = String::from(INJECTED_IMAGE_TAG);
    for marker in markers {
        message.push('\n');
        message.push_str(marker);
    }
    message
}

/// Truncate to at most `max_chars` Unicode characters, appending "…" if truncated — the ellipsis
/// counts against the cap, trailing whitespace at the cut is stripped, `max_chars == 0` yields
/// empty, and an input that already fits comes back unchanged.
#[must_use]
pub fn truncate(input: &str, max_chars: usize) -> String {
    let Some(last_kept) = max_chars.checked_sub(1) else {
        return String::new();
    };
    let mut chars = input.char_indices();
    let Some((cut, _)) = chars.nth(last_kept) else {
        return input.to_string();
    };
    if chars.next().is_none() {
        return input.to_string();
    }
    format!("{}…", input[..cut].trim_end())
}

#[must_use]
pub(crate) fn truncate_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        &s[..s.floor_char_boundary(max_bytes)]
    }
}

/// [`truncate_bytes`] with the cut marked by an ellipsis that counts against `max_bytes`, so a
/// caller composing byte-bounded output can trust the bound; nothing is left of a text with no room
/// for it.
#[must_use]
pub(crate) fn cut_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    match max_bytes.checked_sub('…'.len_utf8()) {
        Some(room) => format!("{}…", truncate_bytes(text, room)),
        None => String::new(),
    }
}

#[must_use]
pub(crate) fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Word character classification shared by the shell grep engine's `-w` handling and editor word
/// navigation. The grep side is a conservative approximation of unicode `\w` that gates the `-w` →
/// `(?:pat)` translation. The two are identical today; a deliberate one-sided divergence (e.g.
/// ASCII-only for GNU-grep parity) must split this helper rather than silently change the other
/// subsystem.
#[must_use]
pub(crate) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Current Unix timestamp in milliseconds since the epoch, `0` if the clock is set before it. `u64`
/// carries timestamps ~500 million years out.
#[must_use]
#[expect(clippy::cast_possible_truncation)]
pub(crate) fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Whether a process with this id is still there: a process this one may not ask about counts as
/// there on unix (only a pid the system reports no such process for is gone), and on Windows a
/// handle that cannot be opened for a plain existence query reads as gone.
#[must_use]
#[cfg(unix)]
pub(crate) fn process_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 sends nothing — it asks the system whether the process exists,
    // and touches no memory.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// See the unix definition: [`OpenProcess`] with a query-only right answers existence.
#[must_use]
#[cfg(windows)]
pub(crate) fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    // SAFETY: a plain open by pid, asking only to query; no code in the target runs.
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process == 0 {
        return false;
    }
    // SAFETY: `process` came from the `OpenProcess` above and is not used again.
    unsafe { CloseHandle(process) };
    true
}

#[must_use]
pub(crate) fn millis_i64(elapsed: std::time::Duration) -> i64 {
    i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
}

/// Parse an env var as a whole number of seconds, falling back to `default_secs`. Shared by every
/// bounded wait armed from an `MAHBOT_*_SECS` setting, so the override pattern lives in one place;
/// `0` produces an immediate timeout — deliberate for tests.
#[must_use]
pub(crate) fn env_duration_secs(name: &str, default_secs: u64) -> std::time::Duration {
    let secs = std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_secs);
    std::time::Duration::from_secs(secs)
}

#[must_use]
pub(crate) fn hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

/// Verify a file's SHA-256 against the hex digest it is pinned to, streaming it rather
/// than loading it (model files run to gigabytes). An empty `expected` skips the check —
/// the "no hash configured" semantics every downloader in the product shares.
pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    use sha2::{Digest as _, Sha256};
    use std::io::Read as _;

    if expected.is_empty() {
        return Ok(());
    }
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {} for SHA256 verification", path.display()))?;
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex_string(&hasher.finalize());
    if actual != expected {
        anyhow::bail!(
            "SHA256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

#[must_use]
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

#[must_use]
pub(crate) fn base64_decode(s: &str) -> Option<Vec<u8>> {
    STANDARD.decode(s).ok()
}

#[must_use]
pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix('~') {
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
        if let Ok(home) = home {
            return PathBuf::from(home).join(stripped.trim_start_matches('/'));
        }
    }
    PathBuf::from(path)
}

#[cfg(windows)]
#[must_use]
pub(crate) fn wide_path(path: &Path) -> Option<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt as _;
    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    (!wide.contains(&0)).then_some(wide)
}

/// Strip the Windows verbatim (`\\?\`) prefix that `std::fs::canonicalize` produces, so the path
/// can be handed to a shell child or rendered into a prompt. Only the two forms a canonicalized
/// *local* path takes are rewritten — a drive path (`\\?\C:\…`) and the UNC form
/// (`\\?\UNC\srv\share`) — and the drive form only when a separator follows the colon: `\\?\C:`
/// alone is drive-relative. Anything else (a volume-GUID or `GLOBALROOT` device path), a path
/// without the prefix, and one that cannot be spelled exactly comes back unchanged. The trade-off
/// is deliberate: a child process cannot reach past the platform's classic path-length limit once
/// the verbatim form is gone, which is accepted.
#[must_use]
pub(crate) fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    let Some(raw) = path.to_str() else {
        return path.to_path_buf();
    };
    if let Some(rest) = raw.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = raw.strip_prefix(r"\\?\")
        && rest.as_bytes().get(1) == Some(&b':')
        && matches!(rest.as_bytes().get(2), Some(b'\\' | b'/'))
    {
        return PathBuf::from(rest);
    }
    path.to_path_buf()
}

/// Whether `candidate` is `base` itself or lies below it, compared after both sides have dropped
/// the Windows verbatim prefix ([`strip_verbatim_prefix`]) — that strip is what makes a
/// canonicalized candidate and a stored base comparable. Lexical, component-wise and case-sensitive
/// (a drive letter's case is folded by the platform while a path is parsed), so a sibling sharing a
/// name prefix stays outside.
#[must_use]
pub(crate) fn is_within(candidate: &Path, base: &Path) -> bool {
    strip_verbatim_prefix(candidate).starts_with(strip_verbatim_prefix(base))
}

/// Whether this process may create a file in the directory `dir`: `access(2)` answers it — through
/// the group/other-write bit and through a read-only mount, as a shell's `[ -w dir ]` does — and
/// nothing is written. Unix-only: the other answer is a probe file in the directory, and both
/// callers are unix-only.
#[cfg(unix)]
#[must_use]
pub(crate) fn dir_is_writable(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt as _;

    let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `path` is a valid NUL-terminated C string that outlives the call
    // and `access` only reads it.
    unsafe { libc::access(path.as_ptr(), libc::W_OK) == 0 }
}

#[must_use]
pub(crate) fn strip_suffix_ignoring_case<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let cut = s.len().checked_sub(suffix.len())?;
    s.get(cut..)
        .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
        .then(|| &s[..cut])
}

/// What makes two paths one file: canonicalized when it exists, as written otherwise. Every file
/// identity — a decode-cache key, two markers naming one target — is compared through this, so two
/// spellings of one path never read as two files.
#[must_use]
pub(crate) fn canonical_or_self(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod verbatim_prefix_tests {
    use super::strip_verbatim_prefix;
    use std::path::{Path, PathBuf};

    #[test]
    fn strips_only_the_verbatim_prefix() {
        assert_eq!(
            strip_verbatim_prefix(Path::new("/tmp/mahbot")),
            PathBuf::from("/tmp/mahbot")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\C:\a\b")),
            PathBuf::from(r"C:\a\b")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\C:\")),
            PathBuf::from(r"C:\")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\UNC\srv\share")),
            PathBuf::from(r"\\srv\share")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\C:")),
            PathBuf::from(r"\\?\C:")
        );
        for device in [
            r"\\?\Volume{0000}\dir",
            r"\\?\GLOBALROOT\Device\HarddiskVolume1",
        ] {
            assert_eq!(
                strip_verbatim_prefix(Path::new(device)),
                PathBuf::from(device)
            );
        }
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\")),
            PathBuf::from(r"\\?\")
        );
    }

    /// A path that cannot be spelled exactly comes back untouched: a lossy rewrite would name a
    /// *different* path. The path here is a verbatim drive form with one invalid byte appended.
    #[cfg(unix)]
    #[test]
    fn leaves_a_path_it_cannot_spell_exactly() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt as _;

        let mut bytes = br"\\?\C:\a\b".to_vec();
        bytes.push(0xff);
        let raw = OsStr::from_bytes(&bytes);
        assert_eq!(
            strip_verbatim_prefix(Path::new(raw)),
            PathBuf::from(raw),
            "an unspellable path must not be rewritten lossily"
        );
    }
}

/// Write a JSON record to `path` atomically: the whole payload to a sibling `json.tmp`, then a
/// rename. A reader never sees a torn file, and a crash — a self-update kill included — leaves
/// the previous record in place rather than half of a new one. The caller owns the payload and
/// whatever lock serializes its writers: two writers sharing the one `.tmp` could publish a torn
/// file. The failure is returned rather than logged — each record decides how loudly to say it.
pub(crate) fn write_json_record(path: &Path, json: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&tmp, json)?;
    std::fs::rename(tmp, path)
}

/// Run a blocking I/O operation with awareness of the current Tokio runtime: on a multi-threaded
/// runtime it wraps the call in [`tokio::task::block_in_place`] so the blocking thread can be
/// rescheduled; on a current-thread runtime or with no runtime it calls `f()` directly
/// (`block_in_place` would panic there).
///
/// Use this instead of a bare `std::fs::canonicalize` inside async functions; prefer it over
/// [`tokio::task::spawn_blocking`] for operations under ~1 ms.
#[must_use]
pub(crate) fn with_block_in_place<T>(f: impl FnOnce() -> T) -> T {
    if let Ok(handle) = tokio::runtime::Handle::try_current()
        && handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread
    {
        return tokio::task::block_in_place(f);
    }
    f()
}

/// Extract a human-readable message from a panic payload returned by
/// [`catch_unwind`](futures_util::FutureExt::catch_unwind).
#[must_use]
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Write `text` followed by one newline to stdout. A failed write is discarded.
///
/// **Policy — a notice is never a panic.** The prints that belong to a launch — top-level usage and
/// version notices, and the line a subcommand returning an error prints — go through these two
/// rather than the standard macros, which panic on a write that really fails (a closed pipe, a full
/// disk). The subcommand CLIs that always run with wired stdio (`debug`'s own usage and argument
/// errors, chrome, the OpenRouter benchmark) still print through the macros. The families of
/// [`crate::cli`] and `mahbot audio …` print through these writers too: their exit code, not the
/// delivered text, reports whether the operation was carried out, so a closed pipe must not turn it
/// into a crash.
/// A windowed launch may have no standard streams — the library discards a write to an absent
/// handle (`handle_ebadf`), a lost notice and never a fatal one. A detached Unix launch writes to
/// `daemon.log` (`crate::detach`); the update hand-off's stderr is `update.log`. `print_stderr` is
/// also the panic hook's writer, and everything printed before tracing exists goes through
/// `boot::timestamped_stderr`.
#[doc(hidden)]
pub fn print_stdout(text: &str) {
    use std::io::Write as _;
    let _ = writeln!(std::io::stdout(), "{text}");
}

#[doc(hidden)]
pub fn print_stderr(text: &str) {
    use std::io::Write as _;
    let _ = writeln!(std::io::stderr(), "{text}");
}

/// Log panics/cancellations from a `join_all` batch of task results, keeping the loop alive.
pub(crate) fn log_join_failures(
    results: Vec<Result<(), tokio::task::JoinError>>,
    panic_log: &str,
    cancelled_log: &str,
) {
    for result in results {
        if let Err(e) = result {
            if e.is_panic() {
                let payload = e.into_panic();
                error!(error = %panic_message(&*payload), "{panic_log}");
            } else {
                error!("{cancelled_log}");
            }
        }
    }
}

/// Byte trigger for failure-detail dumps (error chains, raw verdict responses). Retry-exhaustion
/// chains embed up to 13 per-attempt errors and comment dumps are read verbatim by downstream
/// agents, so the sandwich keeps the head (outermost context) and tail (last attempt's cause); the
/// omission marker it inserts is additional to this number (see [`truncate_sandwich`]).
pub(crate) const FAILURE_DETAIL_CAP: usize = 24_000;

/// Canonical scrub+truncate failure-comment detail: scrub secrets, then sandwich-truncate at
/// [`FAILURE_DETAIL_CAP`], so dumps rendered here keep one ordering and cap. Some adjacent sites
/// deliberately diverge — scrub-only early-returns, a truncate-then-scrub raw dump — and are
/// intentionally not unified here.
#[must_use]
pub(crate) fn failure_detail(text: &str, label: &str) -> String {
    truncate_sandwich(&scrub_credentials(text), FAILURE_DETAIL_CAP, label)
}

/// Truncate a string using a head/tail "sandwich": the first ~2/3 and last ~1/3 of `max_bytes` with
/// an omission marker between them, or the input unchanged if it fits.
///
/// `max_bytes` is a trigger, not a bound: head + tail + marker exceeds it by the marker's own
/// length. The
/// marker is `"... (N bytes omitted at {label} truncation)\n"`. The head cut floors and the tail
/// cut ceils,
/// so a multi-byte character the tail lands inside moves into the omitted span, never split.
#[must_use]
pub(crate) fn truncate_sandwich(s: &str, max_bytes: usize, label: &str) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head_bytes = max_bytes * 2 / 3;
    let tail_bytes = max_bytes / 3;
    let head_end = s.floor_char_boundary(head_bytes);
    let tail_start = s.ceil_char_boundary(s.len() - tail_bytes);
    // `head_bytes + tail_bytes <= max_bytes < s.len()`, and flooring the head
    // only shrinks it while ceiling the tail only grows it, so the cuts stay
    // ordered; a ratio change that breaks that counts zero omitted bytes instead
    // of panicking inside tool output formatting.
    debug_assert!(
        head_end < tail_start,
        "sandwich cuts overlap: head_end={head_end} tail_start={tail_start} max_bytes={max_bytes}"
    );
    let omitted = tail_start.saturating_sub(head_end);
    format!(
        "{}... ({} bytes omitted at {label} truncation)\n{}",
        &s[..head_end],
        omitted,
        &s[tail_start..]
    )
}

pub(crate) const TOOL_OUTPUT_BUDGET_BYTES: usize = 5_000;

#[must_use]
pub(crate) fn truncate_tool_output(output: &str) -> String {
    truncate_sandwich(output, TOOL_OUTPUT_BUDGET_BYTES, "tool output")
}

fn mime_for_raster_bytes(bytes: &[u8]) -> Option<String> {
    let label = image_format_native_label(image::guess_format(bytes).ok()?)?;
    Some(format!("image/{}", label.to_ascii_lowercase()))
}

pub(crate) async fn local_image_to_data_uri(path: &std::path::Path) -> anyhow::Result<String> {
    let bytes = tokio::fs::read(path).await?;
    Ok(data_uri_of(path, &bytes))
}

fn data_uri_of(path: &std::path::Path, bytes: &[u8]) -> String {
    let mime = mime_for_raster_bytes(bytes).unwrap_or_else(|| mime_for_extension(path).to_owned());
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

/// Longest-side cap for each bounded-JPEG re-encode in this module: the shared
/// re-encode behind [`local_image_to_compressed_data_uri_with_meta`] (used by
/// every ingestion path — the `read` tool, chrome/computer screenshots,
/// document page images, inbound channel media) scales the longest side down to
/// this many pixels, aspect-preserving. Callers that must tell the model an
/// image was reduced read the cap from here.
pub(crate) const INBOUND_IMAGE_MAX_SIDE: u32 = 1024;
const INBOUND_IMAGE_JPEG_QUALITY: u8 = 85;

/// Input-size ceiling for the inbound-photo decode, aligned with the reference-image path's
/// pattern: a
/// decoded bitmap can be far larger than its file, so over-cap files are refused from metadata
/// BEFORE the
/// read. Shared by inbound compression, the media-target classifier and the GUI render path; a
/// decode budget, never a transfer limit — no outgoing attachment is refused on it.
pub(crate) const INBOUND_IMAGE_MAX_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Result of bounded inbound-image compression, with the post-EXIF/post-resize
/// dimensions and the source-format label for annotation rendering.
pub(crate) struct CompressedImageMeta {
    pub data_uri: String,
    pub width: u32,
    pub height: u32,
    pub format: String,
}

/// Read a local image file and return a bounded-JPEG data URI plus the post-EXIF/post-resize
/// dimensions of the encoded JPEG (longest side [`INBOUND_IMAGE_MAX_SIDE`], quality
/// [`INBOUND_IMAGE_JPEG_QUALITY`], alpha flattened onto white, EXIF orientation applied) — the
/// single source of the annotation dims, which describe the very encode the data URI carries.
pub(crate) async fn local_image_to_compressed_data_uri_with_meta(
    path: &std::path::Path,
) -> anyhow::Result<CompressedImageMeta> {
    // Metadata-first cap check, as the reference-image path does: refuse over-cap files BEFORE the
    // read, so a huge file never enters memory just to be refused.
    let meta = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to access inbound image {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!("Inbound image {} is not a regular file", path.display());
    }
    if meta.len() > INBOUND_IMAGE_MAX_INPUT_BYTES {
        anyhow::bail!(
            "Inbound image {} is {} bytes — over the {} MiB decode cap, so it is not compressed",
            path.display(),
            meta.len(),
            INBOUND_IMAGE_MAX_INPUT_BYTES / (1024 * 1024),
        );
    }
    let bytes = tokio::fs::read(path).await?;
    let (out, width, height, format) = with_block_in_place(|| compress_inbound_image(&bytes))?;
    Ok(CompressedImageMeta {
        data_uri: jpeg_data_uri(&out),
        width,
        height,
        format,
    })
}

/// Map a decodable raster format to its uppercase annotation label (`"PNG" | "JPEG" | "WEBP"`);
/// `None` for a format mahbot does not attach natively (GIF/BMP/...). The read tool's content
/// sniff, the inbound-compression path and `mime_for_raster_bytes` all derive their decision here.
#[must_use]
pub(crate) fn image_format_native_label(fmt: image::ImageFormat) -> Option<&'static str> {
    use image::ImageFormat;
    match fmt {
        ImageFormat::Png => Some("PNG"),
        ImageFormat::Jpeg => Some("JPEG"),
        ImageFormat::WebP => Some("WEBP"),
        _ => None,
    }
}

/// One bounded compression step for inbound photos: decode, apply EXIF orientation, downscale the
/// longest side to [`INBOUND_IMAGE_MAX_SIDE`] (aspect-preserving, Triangle filter, min 1 px),
/// flatten alpha onto white and re-encode as JPEG at [`INBOUND_IMAGE_JPEG_QUALITY`]. Returns the
/// bytes, the final dimensions and the uppercase source label (`"PNG" | "JPEG" | "WEBP"`, else
/// `"IMAGE"`); the input-size ceiling is the caller's.
fn compress_inbound_image(bytes: &[u8]) -> anyhow::Result<(Vec<u8>, u32, u32, String)> {
    let format = match image::guess_format(bytes) {
        Ok(f) => image_format_native_label(f).unwrap_or("IMAGE"),
        Err(_) => "IMAGE",
    };
    let (out, width, height) = compress_jpeg_core(
        bytes,
        "inbound",
        |w, h| {
            let longest = w.max(h);
            #[expect(clippy::cast_precision_loss)]
            let scale = INBOUND_IMAGE_MAX_SIDE as f32 / longest as f32;
            scale
        },
        INBOUND_IMAGE_JPEG_QUALITY,
    )?;
    Ok((out, width, height, format.to_string()))
}

fn compress_jpeg_core(
    bytes: &[u8],
    label: &str,
    scale_for: impl FnOnce(u32, u32) -> f32,
    quality: u8,
) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    use image::GenericImageView;
    let mut img = image::load_from_memory(bytes)
        .with_context(|| format!("Failed to decode {label} image"))?;
    // EXIF orientation is metadata, not pixels: `load_from_memory` returns the stored pixels and
    // the encoder starts from an empty EXIF buffer, so over-cap phone JPEGs would be re-encoded
    // rotated. Apply the tag before downscaling so the output dimensions reflect the true
    // orientation.
    if let Some(orientation) = exif_orientation(bytes) {
        img.apply_orientation(orientation);
    }
    let (w, h) = img.dimensions();
    let scale = scale_for(w, h);
    let img = if scale < 1.0 {
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let nw = (w as f32 * scale).round().max(1.0) as u32;
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let nh = (h as f32 * scale).round().max(1.0) as u32;
        img.resize(nw, nh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgb = flatten_alpha_onto_white(&img);
    let (width, height) = rgb.dimensions();
    let mut out = Vec::new();
    {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
        encoder
            .encode(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                image::ExtendedColorType::Rgb8,
            )
            .with_context(|| format!("Failed to encode compressed {label} image"))?;
    }
    Ok((out, width, height))
}

/// Input-size ceiling for the reference-image compression path (aligned with video_edit's 50 MB
/// cap): over-cap files are read fully before compression, so this bounds pathological inputs.
const MAX_REFERENCE_INPUT_BYTES: u64 = 50 * 1024 * 1024;

/// Combined input-size ceiling across all references of one request: the per-image ceiling does not
/// bound a multi-reference total, and the fail-open path (no catalog cap) would otherwise hold up
/// to 16 × 50 MB of source bytes in memory before the body budget runs.
const MAX_TOTAL_REFERENCE_INPUT_BYTES: u64 = 100 * 1024 * 1024;

/// Bound on a reference-image read: guards the metadata→read window where a path swapped to a FIFO
/// could otherwise block forever (the is_file check runs on the pre-read metadata). The blocked
/// read task lingers until the special file resolves, but the tool call errors visibly instead of
/// hanging.
const REFERENCE_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(1);

const JPEG_DATA_URI_PREFIX: &str = "data:image/jpeg;base64,";

/// JPEG bytes as the data URI an `[IMAGE:]` marker carries.
#[must_use]
pub(crate) fn jpeg_data_uri(bytes: &[u8]) -> String {
    format!("{JPEG_DATA_URI_PREFIX}{}", STANDARD.encode(bytes))
}

/// Compression ladder: (downscale factor, JPEG quality) steps, mildest first. A same-format PNG
/// re-encode can GROW the file (+3% measured), so every step crosses to JPEG with alpha flattened;
/// later steps downscale. The ladder is a small fixed bound (~6 steps).
const REFERENCE_COMPRESSION_LADDER: &[(f32, u8)] = &[
    (1.0, 85),
    (1.0, 70),
    (1.0, 55),
    (0.75, 70),
    (0.5, 70),
    (0.5, 45),
];

/// A validated reference image for a generation request: under-cap images pass through unchanged,
/// over-cap ones are compressed via the bounded ladder. Held in memory only — no disk artifacts.
pub(crate) struct ReferenceImage {
    data_uri: String,
    /// Original file bytes kept for later ladder steps (aggregate body budget); holding them avoids
    /// a re-read (no TOCTOU window). Emptied by [`ReferenceImage::release_source_bytes`] and never
    /// empty before that (the content sniffer refuses empty input), so empty means the image is
    /// final.
    source_bytes: Vec<u8>,
    next_step: usize,
}

impl ReferenceImage {
    #[must_use]
    pub(crate) fn data_uri(&self) -> &str {
        &self.data_uri
    }

    #[must_use]
    pub(crate) fn has_compression_left(&self) -> bool {
        !self.source_bytes.is_empty() && self.next_step < REFERENCE_COMPRESSION_LADDER.len()
    }

    /// Apply the next compression step (used by the aggregate body budget). The guard keeps a
    /// caller that
    /// skipped [`ReferenceImage::has_compression_left`] from panicking on the ladder index.
    pub(crate) fn compress_more(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.has_compression_left(),
            "Reference image compression is already final"
        );
        let out =
            with_block_in_place(|| compress_reference_step(&self.source_bytes, self.next_step))?;
        self.next_step += 1;
        self.data_uri = jpeg_data_uri(&out);
        Ok(())
    }

    pub(crate) fn release_source_bytes(&mut self) {
        self.source_bytes = Vec::new();
    }
}

/// Load a reference image for generation: validate existence, the input-size
/// ceiling, and regular-file-ness via metadata BEFORE any read; then the
/// format (PNG/JPEG/WebP by extension AND content sniff); compress over-cap
/// images via the bounded ladder until they fit `max_bytes`.
pub(crate) async fn load_reference_image(
    path: &std::path::Path,
    max_bytes: u64,
) -> anyhow::Result<ReferenceImage> {
    // Metadata-first: a missing file reports not-found (not a format error), pathological inputs
    // are refused unread, and special files (FIFOs) are refused so the read cannot block forever.
    let meta = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to access reference image {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!(
            "Reference image {} is not a regular file — refusing to read it.",
            path.display(),
        );
    }
    if meta.len() > MAX_REFERENCE_INPUT_BYTES {
        anyhow::bail!(
            "Reference image {} is limited to 50 MB, got {} bytes. Use a smaller image.",
            path.display(),
            meta.len(),
        );
    }
    // Extension gate before reading: HEIC/HEIF, GIF, BMP and unknown files are rejected without
    // reading them (a 49 MB HEIC must not be read just to be refused).
    check_reference_extension(path)?;
    let bytes = tokio::time::timeout(REFERENCE_READ_TIMEOUT, tokio::fs::read(path))
        .await
        .map_err(|_| anyhow::anyhow!("Timed out reading reference image {}", path.display()))?
        .map_err(|e| anyhow::anyhow!("Failed to read reference image {}: {e}", path.display()))?;

    // Content sniff: catches mislabeled or undecodable files the extension gate let through, so
    // undecoded bytes never reach the provider.
    let format = sniff_reference_content(path, &bytes)?;

    if bytes.len() as u64 <= max_bytes {
        return Ok(ReferenceImage {
            data_uri: format!(
                "data:{};base64,{}",
                format.to_mime_type(),
                STANDARD.encode(&bytes)
            ),
            source_bytes: bytes,
            next_step: 0,
        });
    }

    // Over the cap → bounded compression ladder, in-memory only.
    let mut step = 0;
    loop {
        if step >= REFERENCE_COMPRESSION_LADDER.len() {
            // Exact bytes + decimal MB ("1500000 bytes (1.5 MB)"): never confuse MiB with MB.
            #[expect(clippy::cast_precision_loss)]
            let cap = format!(
                "{} bytes ({:.1} MB)",
                max_bytes,
                max_bytes as f64 / 1_000_000.0
            );
            anyhow::bail!(
                "Reference image {} is {} bytes and cannot be compressed under the {} cap \
                 after {} bounded steps. Use a smaller or simpler image.",
                path.display(),
                bytes.len(),
                cap,
                step,
            );
        }
        let out = with_block_in_place(|| compress_reference_step(&bytes, step))?;
        if out.len() as u64 <= max_bytes {
            return Ok(ReferenceImage {
                data_uri: jpeg_data_uri(&out),
                source_bytes: bytes,
                next_step: step + 1,
            });
        }
        step += 1;
    }
}

/// Pre-flight combined-size check for multi-reference requests: sums metadata lengths BEFORE any
/// file is
/// read, so a pathological total is refused without loading the references (the per-image ceiling
/// does not bound the sum).
pub(crate) async fn check_reference_total_input(paths: &[PathBuf]) -> anyhow::Result<()> {
    let mut total: u64 = 0;
    for path in paths {
        total += tokio::fs::metadata(path)
            .await
            .with_context(|| format!("Failed to access reference image {}", path.display()))?
            .len();
    }
    if total > MAX_TOTAL_REFERENCE_INPUT_BYTES {
        anyhow::bail!(
            "Combined reference images are limited to 100 MB, got {total} bytes total. \
             Use fewer or smaller images.",
        );
    }
    Ok(())
}

fn check_reference_extension(path: &Path) -> anyhow::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg" | "webp")) {
        anyhow::bail!(
            "Reference image {}: unsupported format ({}). Only PNG, JPEG, or WebP \
             images are accepted.",
            path.display(),
            ext.as_deref().unwrap_or("unknown extension"),
        );
    }
    Ok(())
}

/// Content-sniff the actual image format, accepting only PNG/JPEG/WebP.
fn sniff_reference_content(path: &Path, bytes: &[u8]) -> anyhow::Result<image::ImageFormat> {
    let format = image::guess_format(bytes).map_err(|_| {
        anyhow::anyhow!(
            "Reference image {}: content is not a decodable image (PNG/JPEG/WebP). \
             HEIC/HEIF and other unsupported formats are not accepted.",
            path.display(),
        )
    })?;
    if image_format_native_label(format).is_none() {
        anyhow::bail!(
            "Reference image {}: unsupported image format ({format:?}). Only PNG, JPEG, \
             or WebP images are accepted.",
            path.display(),
        );
    }
    Ok(format)
}

/// One bounded compression step: decode, optionally downscale, flatten alpha, re-encode as JPEG.
fn compress_reference_step(bytes: &[u8], step: usize) -> anyhow::Result<Vec<u8>> {
    // Both call sites guarantee `step < REFERENCE_COMPRESSION_LADDER.len()`.
    let (scale, quality) = REFERENCE_COMPRESSION_LADDER[step];
    let (out, _, _) = compress_jpeg_core(bytes, "reference", |_, _| scale, quality)?;
    Ok(out)
}

/// Read the EXIF orientation tag without decoding pixels (the decoder's header-only
/// `orientation()`).
pub(crate) fn exif_orientation(bytes: &[u8]) -> Option<image::metadata::Orientation> {
    use image::ImageDecoder;
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    match decoder.orientation().ok()? {
        image::metadata::Orientation::NoTransforms => None,
        orientation => Some(orientation),
    }
}

fn flatten_alpha_onto_white(img: &image::DynamicImage) -> image::RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (x, y, px) in rgba.enumerate_pixels() {
        let [r, g, b, a] = px.0;
        let alpha = f32::from(a) / 255.0;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let blend = |c: u8| (f32::from(c) * alpha + 255.0 * (1.0 - alpha)).round() as u8;
        rgb.put_pixel(x, y, image::Rgb([blend(r), blend(g), blend(b)]));
    }
    rgb
}

/// The containers a picture stream rides in, by name — what an arrival calls a video before the
/// probe settles what the file holds. `.ts` and `.mts` are deliberately absent: they are source
/// files here, and one must not pay a probe — or a refusal — on every read.
pub(crate) const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mov", "mkv", "avi", "webm", "flv", "wmv", "mpg", "mpeg", "ogv", "3gp", "3g2",
    "vob", "rm", "rmvb", "asf", "f4v", "divx", "m2ts",
];

/// The containers a sound track rides in, as the managed ffmpeg reads them — the counterpart of
/// [`VIDEO_EXTENSIONS`] for the media path. Distinct from `speech::AUDIO_FORMATS`, which is what the
/// in-process decoder takes; a name only decides whether the probe is asked at all.
pub(crate) const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "ogg", "oga", "m4a", "m4b", "aac", "flac", "opus", "wma", "aiff", "aif", "amr",
    "caf", "ac3",
];

/// Recognized image extensions for video_edit image inputs (reference images and frame anchors),
/// matching the Telegram routing list for PNG/JPEG/WebP — the only locally decodable codecs — plus
/// heic/heif, passed to the provider unmangled (never locally decoded).
pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "heic", "heif"];

/// The product-level cap on one file wherever the product bounds one: the composer's attach limit,
/// the read tool's document-container cap and the document tool's input cap. Not a transfer limit.
pub(crate) const FILE_MAX_BYTES: u64 = 50 * 1024 * 1024;

pub(crate) const TELEGRAM_FILES_DIR: &str = "mahbot_telegram_files";

/// The temp directory inbound Telegram attachments are downloaded into: the receive path writes
/// each message's attachments into its own `telegram_staging_dir_name` subdirectory below this
/// root, and that is the inbound containment root for reading, copying and deleting. Under
/// `cfg(test)` this resolves below the process-level test root: a test process can inherit the
/// daemon's pinned `TMPDIR`, and a fixture must never touch a live daemon's directory.
#[must_use]
pub(crate) fn telegram_files_root() -> std::path::PathBuf {
    #[cfg(test)]
    {
        crate::util::test::test_root().join(TELEGRAM_FILES_DIR)
    }
    #[cfg(not(test))]
    {
        std::env::temp_dir().join(TELEGRAM_FILES_DIR)
    }
}

/// Name of the per-message subdirectory of `telegram_files_root` holding one message's inbound
/// attachments. The producer — the Telegram receive path or the desktop composer — names the
/// directory with it and records it on the message, so containment compares against the name
/// created.
#[must_use]
pub(crate) fn telegram_staging_dir_name(chat_id: &str, message_id: i64) -> String {
    format!("msg_{chat_id}_{message_id}")
}

#[must_use]
pub(crate) fn has_extension(path: &std::path::Path, table: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| table.contains(&ext.to_ascii_lowercase().as_str()))
}

#[must_use]
pub(crate) fn is_video_extension(path: &std::path::Path) -> bool {
    has_extension(path, VIDEO_EXTENSIONS)
}

#[must_use]
pub(crate) fn is_audio_extension(path: &std::path::Path) -> bool {
    has_extension(path, AUDIO_EXTENSIONS)
}

#[must_use]
pub(crate) fn is_image_extension(path: &std::path::Path) -> bool {
    has_extension(path, IMAGE_EXTENSIONS)
}

#[must_use]
pub(crate) fn is_http_url(target: &str) -> bool {
    target.starts_with("http://") || target.starts_with("https://")
}

pub(crate) fn mime_for_extension(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("heic") => "image/heic",
        Some("heif") => "image/heif",
        Some("mp4") => "video/mp4",
        Some("mpeg") => "video/mpeg",
        Some("mov") => "video/quicktime",
        Some("webm") => "video/webm",
        _ => "application/octet-stream",
    }
}

#[must_use]
pub(crate) fn file_name_or_path(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

/// `NFD(toCasefold(NFD(name)))`, the Unicode canonical caseless matching: the full fold, so `ß`
/// and `ss` (and `ﬃ` and `ffi`) are one key, and canonical decomposition, so composed and
/// decomposed accents are one key. Two names are the same name exactly when their keys are.
#[must_use]
pub(crate) fn caseless_name_key(name: &str) -> String {
    use unicode_normalization::UnicodeNormalization as _;
    let decomposed: String = name.nfd().collect();
    unicase::UniCase::new(decomposed)
        .to_folded_case()
        .nfd()
        .collect()
}

/// Whether `character` is a break a text stack may start a line at: the mandatory breaks of UAX
/// #14. A rendered answer's lines come from untrusted text, so escaping must account for every one.
#[must_use]
pub(crate) fn is_line_break(character: char) -> bool {
    matches!(
        character,
        '\n' | '\r' | '\u{0B}' | '\u{0C}' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// `text` as one line, for a value or name taken from an untrusted document: every break a text
/// stack may take ([`is_line_break`]) becomes `\n`, so a document's own text cannot stand where
/// this reader's lines do. A value carrying its own backslash and `n` reads the same — one line
/// leaves no other spelling.
#[must_use]
pub(crate) fn one_line(text: String) -> String {
    if !text.chars().any(is_line_break) {
        return text;
    }
    let mut out = String::with_capacity(text.len() + 4);
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\r' => {
                characters.next_if_eq(&'\n');
                out.push_str("\\n");
            }
            _ if is_line_break(character) => out.push_str("\\n"),
            _ => out.push(character),
        }
    }
    out
}

/// Neutralize a name taken from an untrusted source before reuse: every control character, every
/// break a
/// text stack may take ([`is_line_break`]), both path separators, and the `[`/`]` a media marker is
/// built from become `_` — safe both as a path component and inside a `[KIND:...]` marker.
#[must_use]
pub(crate) fn neutralized_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_control() || is_line_break(c) || matches!(c, '/' | '\\' | '[' | ']') {
                '_'
            } else {
                c
            }
        })
        .collect()
}

#[must_use]
pub(crate) fn name_stem(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => stem,
        _ => name,
    }
}

#[must_use]
pub(crate) fn suffixed_name(name: &str, n: u32) -> String {
    let stem = name_stem(name);
    let ext = name.strip_prefix(stem).unwrap_or_default();
    format!("{}{ext}", numbered(stem, n))
}

fn numbered(name: &str, n: u32) -> String {
    if n <= 1 {
        name.to_string()
    } else {
        format!("{name}_{n}")
    }
}

/// The first leaf under `dir` that `claim` accepts, walking `leaf`'s attempts in order. The claim
/// IS the
/// collision check: two concurrent callers must not pick the same leaf, so a check-then-act probe
/// would let them overwrite each other.
async fn claim_unique<T>(
    dir: &Path,
    leaf: impl Fn(u32) -> String,
    mut claim: impl AsyncFnMut(PathBuf) -> std::io::Result<T>,
) -> std::io::Result<(PathBuf, T)> {
    let mut counter = 1u32;
    loop {
        let candidate = dir.join(leaf(counter));
        match claim(candidate.clone()).await {
            Ok(value) => return Ok((candidate, value)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => counter += 1,
            Err(e) => return Err(e),
        }
    }
}

/// Create a NEW empty file `dir`/`name`, disambiguating with a `_<n>` counting suffix when the name
/// is taken, and return the open handle with its path — the collision check is [`claim_unique`]'s.
///
/// Whatever is written through the handle must be FLUSHED before the path is handed on: a
/// `tokio::fs::File` reports a write complete once its last chunk reached the blocking pool, so a
/// caller that returns the path can pass on a file that is still empty.
pub(crate) async fn create_unique_file(
    dir: &Path,
    name: &str,
) -> std::io::Result<(tokio::fs::File, PathBuf)> {
    let (path, file) = claim_unique(
        dir,
        |n| suffixed_name(name, n),
        async |candidate| {
            tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(candidate)
                .await
        },
    )
    .await?;
    Ok((file, path))
}

pub(crate) async fn create_unique_dir(dir: &Path, name: &str) -> std::io::Result<PathBuf> {
    let (path, ()) = claim_unique(
        dir,
        |n| numbered(name, n),
        async |candidate| tokio::fs::create_dir(candidate).await,
    )
    .await?;
    Ok(path)
}

/// Write `bytes` into a newly created unique file `dir`/`name` and answer its path — a complete
/// file or
/// nothing: a failed write removes the reservation rather than leave a short file behind a path
/// someone will be handed (see [`create_unique_file`] for the flush).
pub(crate) async fn write_unique_file(
    dir: &Path,
    name: &str,
    bytes: &[u8],
) -> std::io::Result<PathBuf> {
    let (mut file, path) = create_unique_file(dir, name).await?;
    let written = async {
        tokio::io::AsyncWriteExt::write_all(&mut file, bytes).await?;
        tokio::io::AsyncWriteExt::flush(&mut file).await
    }
    .await;
    if let Err(e) = written {
        drop(file);
        let _ = tokio::fs::remove_file(&path).await;
        return Err(e);
    }
    Ok(path)
}

/// Strip common ANSI escape codes (colors, bold, underline, cursor movement) from a string, keeping
/// the visible content — for shell output and any text that may carry control sequences.
#[must_use]
pub(crate) fn strip_ansi_escapes(input: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"\x1B\[[0-9;]*[a-zA-Z]|\x1B\][0-9;]*[^\x1B]*\x1B\\|\x1B[\(\)\[\]KM]|\x1B\][0-9;]*\x07",
        )
        .unwrap()
    });
    RE.replace_all(input, "").to_string()
}

/// `key: value` / `key=value` pairs whose key names a credential (`token`, `api_key`, …), capturing
/// the value.
static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(token|api[_-]?key|password|secret|user[_-]?key|bearer|credential)["']?\s*[:=]\s*(?:"([^"]{8,})"|'([^']{8,})'|([a-zA-Z0-9_\-\./+=]{8,}))"#).expect("hardcoded regex is valid")
});

/// Scrub credentials from tool output to prevent accidental exfiltration. A matched value keeps its
/// first four characters for context, then gets a `*[REDACTED]` suffix; the prefix is taken on a
/// char boundary, so a multi-byte value cannot panic.
#[must_use]
pub(crate) fn scrub_credentials(input: &str) -> String {
    SENSITIVE_KV_REGEX
        .replace_all(input, |caps: &regex::Captures| {
            let full_match = &caps[0];
            let key = &caps[1];
            let val = caps
                .get(2)
                .or(caps.get(3))
                .or(caps.get(4))
                .map_or("", |m| m.as_str());

            debug_assert!(val.len() >= 8, "regex guarantees values >= 8 chars");
            let prefix = val
                .char_indices()
                .nth(4)
                .map_or(val, |(byte_idx, _)| &val[..byte_idx]);

            // Quote style comes from which capture group matched: Group 2 double-quoted, Group 3
            // single-quoted, Group 4 (else) unquoted. Group identity avoids false positives from
            // quotes or apostrophes appearing elsewhere in the match.
            let quote = if caps.get(2).is_some() {
                Some('"')
            } else if caps.get(3).is_some() {
                Some('\'')
            } else {
                None
            };

            let redacted = format!("{prefix}*[REDACTED]");

            if full_match.contains(':') {
                match quote {
                    Some('"') => format!("\"{key}\": \"{redacted}\""),
                    Some('\'') => format!("{key}: '{redacted}'"),
                    _ => format!("{key}: {redacted}"),
                }
            } else {
                match quote {
                    Some('"') => format!("{key}=\"{redacted}\""),
                    Some('\'') => format!("{key}='{redacted}'"),
                    _ => format!("{key}={redacted}"),
                }
            }
        })
        .to_string()
}

const REDACTED_MARKER: &str = "[REDACTED]";

/// Replace every `secrets` entry found in `input` with [`REDACTED_MARKER`].
/// Exists in addition to [`scrub_credentials`]: that one recognises credential-shaped text by *key
/// name* and keeps a four-character prefix for context, while a value the caller itself handed over
/// is known exactly and must not survive even in part — nor does it need the shared regex to
/// recognise it, which would leave a bare value under an unknown key untouched. Empty values are
/// skipped: replacing `""` would intersperse the marker at every character boundary.
///
/// Every spelling goes: the value, its JSON-escaped body when it differs, and its percent-encoded
/// form in
/// either hex case. The replace runs over the whole input, so a very short value can mangle the
/// product's own wording around it — accepted: a declared secret must not survive in any form.
#[must_use]
pub(crate) fn redact_secrets(input: &str, secrets: &[String]) -> String {
    let mut out = input.to_string();
    for secret in secrets {
        if secret.is_empty() {
            continue;
        }
        out = out.replace(secret.as_str(), REDACTED_MARKER);
        let escaped = serde_json::Value::String(secret.clone()).to_string();
        let escaped = escaped
            .strip_prefix('"')
            .and_then(|body| body.strip_suffix('"'))
            .unwrap_or(&escaped);
        if escaped != secret {
            out = out.replace(escaped, REDACTED_MARKER);
        }
        for encoded in percent_encoded(secret) {
            out = out.replace(&encoded, REDACTED_MARKER);
        }
    }
    out
}

fn percent_encoded(secret: &str) -> Vec<String> {
    fn encode(secret: &str, upper: bool) -> String {
        use std::fmt::Write as _;

        let mut out = String::with_capacity(secret.len());
        for byte in secret.bytes() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                out.push(char::from(byte));
            } else if upper {
                let _ = write!(out, "%{byte:02X}");
            } else {
                let _ = write!(out, "%{byte:02x}");
            }
        }
        out
    }

    let upper = encode(secret, true);
    if upper == secret {
        return Vec::new();
    }
    let lower = encode(secret, false);
    if lower == upper {
        vec![upper]
    } else {
        vec![upper, lower]
    }
}

/// Extract a provider error detail string from an error-body JSON value with a conservative cascade
/// over the envelope fields providers actually use: the wrapped shape (`{"error": {...}}` /
/// `{"error": "msg"}`), the bare error object and a bare JSON string.
///
/// Cascade:
/// 1. `message` — the standard field; callers that surfaced `error.message` keep that behaviour.
/// 2. `metadata.raw` — OpenRouter forwards the upstream provider's raw error body here when the
///    top-level message is generic; the raw text is often itself JSON, whose detail is preferred.
/// 3. `metadata.provider_error_code` — the upstream provider's error code.
/// 4. `code`, then `type` — bare identifiers when nothing richer exists.
///
/// Used by the image-generation and web-search tool error paths and the input-image-rejection
/// builder.
#[must_use]
pub(crate) fn extract_provider_error_detail(error: &serde_json::Value) -> Option<String> {
    if let serde_json::Value::String(s) = error
        && !s.trim().is_empty()
    {
        return Some(s.clone());
    }

    match error.get("error") {
        Some(serde_json::Value::String(s)) if !s.trim().is_empty() => return Some(s.clone()),
        Some(inner) if !inner.is_null() => {
            if let Some(detail) = extract_provider_error_detail(inner) {
                return Some(detail);
            }
        }
        _ => {}
    }

    let text = |v: Option<&serde_json::Value>| -> Option<String> {
        v.and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };

    if let Some(msg) = text(error.get("message")) {
        return Some(msg);
    }
    if let Some(raw) = text(error.get("metadata").and_then(|m| m.get("raw"))) {
        if let Ok(raw_json) = serde_json::from_str::<serde_json::Value>(&raw)
            && let Some(detail) = extract_provider_error_detail(&raw_json)
        {
            return Some(detail);
        }
        return Some(raw);
    }
    if let Some(code) = text(
        error
            .get("metadata")
            .and_then(|m| m.get("provider_error_code")),
    ) {
        return Some(code);
    }
    if let Some(code) = text(error.get("code")) {
        return Some(code);
    }
    text(error.get("type"))
}

pub(crate) fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.is_file() && std::fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"))
    }
}

#[must_use]
pub(crate) fn cargo_bin_dir() -> Option<PathBuf> {
    if let Ok(cargo_home) = std::env::var("CARGO_HOME")
        && !cargo_home.is_empty()
    {
        return Some(PathBuf::from(cargo_home).join("bin"));
    }

    let dirs = UserDirs::new()?;
    Some(dirs.home_dir().join(".cargo").join("bin"))
}

/// Strip surrounding double-quotes and unescape C-style escapes: if the input starts and ends with
/// `"`, the quotes go and [`unescape_c_style`] runs on the inner content; otherwise it is returned
/// as-is (git only C-quotes paths that contain trigger characters). The same approach as git's
/// `unquote_c_style`.
#[must_use]
pub(crate) fn unquote_c_style(raw: &str) -> Option<String> {
    if let Some(inner) = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        unescape_c_style(inner)
    } else {
        Some(raw.to_string())
    }
}

/// Unescape C-style escape sequences from a git path name, as git's `unquote_c_style` does:
/// `\"`/`\\`, `\t` `\n` `\a` `\b` `\f` `\r` `\v`, and `\0`–`\3` plus 1–3 octal digits.
///
/// Malformed escapes return `None`: a dangling `\`, an unrecognized escape letter, and `\4`–`\7`
/// followed by
/// a digit. Octal bytes go through `String::from_utf8_lossy` on the assembled sequence, so
/// multi-byte characters stay intact: `caf\303\251.txt` → `café.txt`.
fn unescape_c_style(input: &str) -> Option<String> {
    let mut result = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i: usize = 0;

    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 1; // consume backslash
            if i >= bytes.len() {
                tracing::warn!(
                    input = %input,
                    "unescape_c_style: dangling backslash at end of string"
                );
                return None;
            }
            match bytes[i] {
                b'"' => result.push(b'"'),
                b'\\' => result.push(b'\\'),
                b't' => result.push(b'\t'),
                b'n' => result.push(b'\n'),
                b'a' => result.push(0x07),
                b'b' => result.push(0x08),
                b'f' => result.push(0x0c),
                b'r' => result.push(b'\r'),
                b'v' => result.push(0x0b),
                b'0'..=b'3' => {
                    // Octal escape: 1–3 digits. The 0–7 check leaves '8'/'9' unconsumed, so they
                    // emit as literal digits on the next outer iteration.
                    let digits_start = i;
                    i += 1;
                    let mut digit_count = 1;
                    while digit_count < 3 && i < bytes.len() && (b'0'..=b'7').contains(&bytes[i]) {
                        i += 1;
                        digit_count += 1;
                    }
                    let octal_str = std::str::from_utf8(&bytes[digits_start..i]).ok()?;
                    let Ok(byte_val) = u8::from_str_radix(octal_str, 8) else {
                        tracing::warn!(
                            input = %input, octal = %octal_str,
                            "unescape_c_style: invalid octal escape"
                        );
                        return None;
                    };
                    result.push(byte_val);
                    continue; // skip the i += 1 at end of loop
                }
                b'4'..=b'7' => {
                    if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                        tracing::warn!(
                            input = %input,
                            ch = %(bytes[i] as char),
                            "unescape_c_style: invalid octal prefix \\4–\\7 followed by digit"
                        );
                        return None;
                    }
                    // Otherwise: literal digit (backslash consumed, no special meaning).
                    result.push(bytes[i]);
                }
                _ => {
                    tracing::warn!(
                        input = %input,
                        ch = %(bytes[i] as char),
                        "unescape_c_style: unrecognized escape sequence"
                    );
                    return None;
                }
            }
        } else {
            result.push(bytes[i]);
        }
        i += 1;
    }

    Some(String::from_utf8_lossy(&result).into_owned())
}

#[cfg(test)]
mod media_mime_tests {
    use super::*;

    // A mislabelled file (photo.png holding JPEG bytes) must produce a data-URI whose MIME matches
    // the bytes, so the decoder's declared-subtype check accepts it instead of dropping the image.
    #[tokio::test]
    async fn data_uri_mime_follows_bytes_not_extension() {
        let img = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        let mut jpeg = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .expect("test JPEG must encode");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.png");
        tokio::fs::write(&path, &jpeg).await.unwrap();

        let uri = local_image_to_data_uri(&path).await.unwrap();
        assert!(
            uri.starts_with("data:image/jpeg;base64,"),
            "MIME must come from the actual bytes; ext says png: {uri}"
        );
        assert!(
            uri.ends_with(&STANDARD.encode(&jpeg)),
            "payload bytes must pass through unchanged"
        );
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::*;

    #[test]
    fn truncate_keeps_the_cap_including_the_ellipsis() {
        // Cap 0 has no room for anything, not even the marker.
        assert_eq!(truncate("hello", 0), "");
        // Cap 1 is the marker alone.
        assert_eq!(truncate("hello", 1), "…");
        // A cut inside a multi-byte char never splits it.
        assert_eq!(truncate("😀😀😀", 2), "😀…");
        assert_eq!(truncate("ааааа", 4), "ааа…");
        assert_eq!(truncate("hello", 5), "hello");
        assert_eq!(truncate("", 5), "");
        assert_eq!(truncate("ab   cd", 4), "ab…");
        assert_eq!(truncate("hello world", 5), "hell…");
    }

    #[test]
    fn an_input_that_fits_passes_through_unchanged() {
        // Anything at or under the cap comes back byte for byte: the marker is only added to a cut.
        for input in ["hello world".to_string(), "a".repeat(5_000)] {
            assert_eq!(
                truncate_sandwich(&input, 5_000, "test"),
                input,
                "a fitting input ({}) must not be touched",
                input.len()
            );
        }
    }

    #[test]
    fn sandwich_large_input() {
        // Input well over the limit — classic head/tail sandwich with label
        let line = "hello world\n".repeat(200_000);
        assert!(line.len() > 1_048_576, "input should exceed 1MB");
        let result = truncate_sandwich(&line, 1_048_576, "output");
        assert!(result.len() < line.len(), "should truncate");
        assert!(
            result.contains("bytes omitted at output truncation"),
            "should contain label in omission marker"
        );
        assert!(
            result.starts_with("hello world"),
            "head should be preserved"
        );
        let last_line = result.lines().last().unwrap_or("");
        assert_eq!(last_line, "hello world", "tail should be preserved");
    }

    #[test]
    fn sandwich_preserves_utf8_boundaries() {
        // Place a 4-byte char ('🐱') right at the head/tail boundary so it straddles the cut:
        // floor_char_boundary must back up to the character boundary. Build: 3329 'x's, then 🐱,
        // then 'y's, so head_bytes ≈ 3333 makes it the last complete char in head.
        let mut input = String::new();
        input.push_str(&"x".repeat(3_329));
        input.push('🐱'); // bytes 3329..=3332
        input.push_str(&"y".repeat(20_000));
        let result = truncate_sandwich(&input, 5_000, "test");
        assert!(
            result.contains('🐱'),
            "multibyte char at boundary should survive intact"
        );
    }

    #[test]
    fn sandwich_tail_start_rounds_up_to_a_char_boundary() {
        // 16 'a's, a 4-byte char the tail cut of a 24-byte cap lands inside, then a visible tail.
        // Flooring the tail start used to put it at 16 — on the head cut — folding it into the
        // omitted span; ceiling moves it to 20, so the char joins the omitted span and the tail
        // survives.
        let input = format!("{}🐱bbbbcd", "a".repeat(16));
        assert_eq!(input.len(), 26);
        let result = truncate_sandwich(&input, 24, "test");
        assert!(
            result.ends_with("bbbbcd"),
            "the tail must survive the boundary case, got: {result:?}"
        );
        assert!(
            result.contains("bytes omitted at test truncation"),
            "the boundary case must take the omission form, got: {result:?}"
        );
        // The cuts can never meet, whatever the cap or the string shape.
        for cap in [12, 18, 24, 25] {
            assert!(
                truncate_sandwich(&input, cap, "test").contains("bytes omitted at test truncation"),
                "cap {cap} must take the omission form"
            );
        }
    }
}

#[cfg(test)]
mod scrub_tests {
    use super::{REDACTED_MARKER, percent_encoded, redact_secrets, scrub_credentials};

    #[test]
    fn scrub_redacts_credentials() {
        /// Cases verifying `[REDACTED]` appears, with optional negative and prefix checks. Fields:
        /// (name, input, must_not_contain, must_start_with). Empty string = skip the check.
        const CASES: &[(&str, &str, &str, &str)] = &[
            (
                "alphanumeric unquoted value",
                "API_KEY=sk-1234567890abcdef",
                "1234567890abcdef",
                "API_KEY=sk-1",
            ),
            (
                "Base64 unquoted value with plus and slash",
                "api_key=u2FsdGVkX1+h/wZ/L3Y+Q==",
                "u2FsdGVkX1+h/wZ/L3Y+Q==",
                "api_key=u2Fs",
            ),
            (
                "double-quoted value with colon separator",
                r#"token: "abcdefgh1234567890""#,
                "1234567890",
                "",
            ),
            (
                "bearer colon-separated value",
                "bearer: eyJhbGciOiJIUzI1NiJ9",
                "eyJhbG",
                "",
            ),
            (
                "hyphen-key variant",
                "user-key=abcdefgh12345678",
                "12345678",
                "user-key=abcd",
            ),
        ];

        for &(name, input, not_contains, prefix) in CASES {
            let out = scrub_credentials(input);
            assert!(out.contains("[REDACTED]"), "{name}: should redact: {out}");
            if !not_contains.is_empty() {
                assert!(
                    !out.contains(not_contains),
                    "{name}: should not leak value: {out}"
                );
            }
            if !prefix.is_empty() {
                assert!(out.starts_with(prefix), "{name}: should keep prefix: {out}");
            }
        }
    }

    #[test]
    fn scrub_exact_output() {
        const CASES: &[(&str, &str, &str)] = &[
            // Single quotes must be preserved (the bug this test guards against).
            (
                "single-quoted value with colon separator",
                "password: 's3cr3t_p@ssw0rd!!'",
                "password: 's3cr*[REDACTED]'",
            ),
            (
                "single-quoted value with equals separator",
                "password='mysecretvalue123'",
                "password='myse*[REDACTED]'",
            ),
            // The key-level optional quote can put a double-quote inside full_match, e.g.
            // "password": 'secretvalue1234'; the capture-group approach still reads it as
            // single-quoted. The key-suffix `"` is consumed by the regex and not reconstructed —
            // out of scope here.
            (
                "double-quoted key with single-quoted value",
                r#""password": 'secretvalue123'"#,
                "\"password: 'secr*[REDACTED]'",
            ),
        ];

        for &(name, input, expected) in CASES {
            assert_eq!(scrub_credentials(input), expected, "{name}");
        }
    }

    #[test]
    fn scrub_passthrough() {
        /// Cases where the input is not a credential pattern and must pass through. Fields: (name,
        /// input).
        const CASES: &[(&str, &str)] = &[
            ("short unquoted values (under 8 chars)", "key=short"),
            (
                "non-secret lines with = and /",
                "normal line with = equals and / slash",
            ),
        ];

        for &(name, input) in CASES {
            assert_eq!(scrub_credentials(input), input, "{name}");
        }
    }

    #[test]
    fn redact_secrets_removes_values_whole() {
        assert_eq!(
            redact_secrets(
                r#"{"token":"123456:AAHsecret"}"#,
                &["123456:AAHsecret".to_string()]
            ),
            format!(r#"{{"token":"{REDACTED_MARKER}"}}"#)
        );
        assert_eq!(
            redact_secrets("key=abc", &["abc".to_string()]),
            format!("key={REDACTED_MARKER}")
        );
        assert_eq!(redact_secrets("unchanged", &[String::new()]), "unchanged");
        // A value carrying a quote, a backslash and a newline is written ESCAPED inside a JSON blob
        // — how the recorded tool arguments carry it — so the escaped spelling must go too.
        let secret = "quo\"te\\back\nline";
        let serialized = serde_json::json!({ "token": secret }).to_string();
        assert!(
            serialized.contains(r#"\""#)
                && serialized.contains(r"\\")
                && serialized.contains(r"\n"),
            "the case only bites when the value is really escaped: {serialized}"
        );
        assert_eq!(
            redact_secrets(&serialized, &[secret.to_string()]),
            format!(r#"{{"token":"{REDACTED_MARKER}"}}"#)
        );
        // A value that reaches a URL is written percent-encoded there, in either hex case: a
        // request URL or query parameter must not survive its literal form being gone.
        let key = "a/+b=c d";
        let encoded = "a%2F%2Bb%3Dc%20d";
        assert_eq!(
            redact_secrets(
                &format!("https://api.example/?key={encoded}"),
                &[key.to_string()]
            ),
            format!("https://api.example/?key={REDACTED_MARKER}")
        );
        assert_eq!(
            redact_secrets(
                &format!("https://api.example/?key={}", encoded.to_lowercase()),
                &[key.to_string()]
            ),
            format!("https://api.example/?key={REDACTED_MARKER}"),
            "hex may be written lowercase"
        );
        assert!(
            percent_encoded("abc-._~123").is_empty(),
            "a URL-safe value has no second spelling"
        );
    }
}

#[cfg(test)]
mod extract_provider_error_detail_tests {
    use super::extract_provider_error_detail;
    use serde_json::{Value, json};

    /// The field ladder in the order the extractor consults it: the `error` wrapper's `message`,
    /// the same
    /// key on a bare object, a plain string `error`, `metadata.raw` (JSON or prose), the provider
    /// code, then `code` and `type`.
    #[test]
    fn the_detail_comes_from_the_first_field_that_carries_one() {
        let cases: [(Value, &str); 9] = [
            (
                json!({"error": {"message": "upstream busy"}}),
                "upstream busy",
            ),
            (json!({"message": "bare detail"}), "bare detail"),
            (json!({"error": "plain message"}), "plain message"),
            (
                json!({
                    "error": {
                        "code": "data_inspection_failed",
                        "metadata": {"raw": "Input image data may contain inappropriate content."}
                    }
                }),
                "Input image data may contain inappropriate content.",
            ),
            (
                json!({
                    "error": {
                        "message": "generic",
                        "metadata": {"raw": r#"{"error":{"message":"deep upstream detail"}}"#}
                    }
                }),
                "generic",
            ),
            (
                json!({"error": {"metadata": {"raw": r#"{"error":{"message":"deep upstream detail"}}"#}}}),
                "deep upstream detail",
            ),
            (
                json!({"error": {"metadata": {"provider_error_code": "upstream_shared_pool"}}}),
                "upstream_shared_pool",
            ),
            (
                json!({"error": {"code": "invalid_request_error"}}),
                "invalid_request_error",
            ),
            (
                json!({"error": {"type": "rate_limit_exceeded"}}),
                "rate_limit_exceeded",
            ),
        ];
        for (body, expected) in &cases {
            assert_eq!(
                extract_provider_error_detail(body).as_deref(),
                Some(*expected),
                "body: {body}"
            );
        }
    }

    #[test]
    fn empty_and_absent_fields_yield_none() {
        for body in [
            json!({}),
            json!({"error": null}),
            json!({"error": {}}),
            json!({"error": {"message": "  "}}),
        ] {
            assert_eq!(extract_provider_error_detail(&body), None, "body: {body}");
        }
    }
}

#[cfg(test)]
mod unescape_c_style_tests {
    use super::unescape_c_style;

    #[test]
    fn test_unescape_c_style() {
        let cases: &[(&str, Option<&str>)] = &[
            (
                r#"hello\"world\\test\nline\there"#,
                Some("hello\"world\\test\nline\there"),
            ),
            (r"\a\b\f\r\v", Some("\x07\x08\x0c\r\x0b")),
            (r"\0\1", Some("\0\x01")),
            (r"\12\37", Some("\n\x1f")),
            (r"\101\377", Some("A\u{FFFD}")),
            (r"caf\303\251.txt", Some("café.txt")),
            (r"\12x", Some("\nx")),
            (r"\18", Some("\x018")),
            ("plain/path.rs", Some("plain/path.rs")),
            ("", Some("")),
            (r"path\", None),
            // \x looks like a hex escape prefix but git's unquote_c_style rejects it.
            (r"\x", None),
            // \q is not a recognized escape sequence.
            (r"\q", None),
            (r"\40", None),
            (r"\77", None),
            (r"\70", None),
            (r"\4", Some("4")),
            (r"\7x", Some("7x")),
        ];
        for (i, (input, expected)) in cases.iter().enumerate() {
            let result = unescape_c_style(input);
            assert_eq!(
                result.as_deref(),
                *expected,
                "case {i}: unescape_c_style({input:?})"
            );
        }
    }
}

#[cfg(test)]
mod strip_ansi_escapes_tests {
    use super::strip_ansi_escapes;

    #[test]
    fn test_ansi_escape_cases() {
        let cases: &[(&str, &str)] = &[
            ("\x1B[31mred\x1B[0m \x1B[1mbold\x1B[22m", "red bold"),
            ("hello world", "hello world"),
            ("\x1B[32mgreen\x1B[0m", "green"),
            ("no escapes here", "no escapes here"),
            ("", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(strip_ansi_escapes(input), *expected, "input: {input:?}");
        }
    }
}

#[cfg(test)]
mod reference_image_tests {
    use super::*;
    use crate::util::test::noisy_png;

    #[tokio::test]
    async fn released_reference_reports_no_compression_left() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("small.png");
        std::fs::write(&path, noisy_png(64, 64)).unwrap();

        let mut img = load_reference_image(&path, 1_500_000).await.unwrap();
        assert!(img.has_compression_left());
        img.release_source_bytes();
        assert!(!img.has_compression_left());
        assert!(img.compress_more().is_err());
    }

    #[tokio::test]
    async fn over_cap_is_compressed_under_cap() {
        const CAP: u64 = 500_000;
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("big.png");
        let bytes = noisy_png(512, 512);
        assert!(
            bytes.len() as u64 > CAP,
            "noise PNG should exceed the cap (got {} bytes)",
            bytes.len()
        );
        std::fs::write(&path, &bytes).unwrap();

        let img = load_reference_image(&path, CAP).await.unwrap();
        assert!(img.data_uri().starts_with("data:image/jpeg;base64,"));
        let decoded = STANDARD
            .decode(img.data_uri().split(',').nth(1).unwrap())
            .unwrap();
        assert!(decoded.len() as u64 <= CAP);
    }

    /// A reference the loader cannot use is refused, and the refusal names which of the two reasons
    /// it was: a container the pipeline never reads, or bytes that are not an image at all.
    #[tokio::test]
    async fn a_reference_that_cannot_be_used_is_refused_with_its_reason() {
        let cases: [(&str, &[u8], &str); 2] = [
            (
                "photo.heic",
                b"\x00\x00\x00\x18ftypheic",
                "unsupported format",
            ),
            ("fake.png", b"this is not an image", "not a decodable image"),
        ];
        for (name, bytes, reason) in cases {
            let tmp = tempfile::TempDir::new().unwrap();
            let path = tmp.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            let Err(err) = load_reference_image(&path, 1_500_000).await else {
                panic!("{name} should be rejected");
            };
            assert!(err.to_string().contains(reason), "{name}: {err}");
        }
    }

    /// Inject an EXIF orientation tag (0x0112 = 6 → Rotate90) into a JPEG right after the SOI
    /// marker. The stored pixels stay unrotated; a viewer (or the compression path) must rotate
    /// them per the tag.
    fn jpeg_with_exif_orientation(jpeg: &[u8]) -> Vec<u8> {
        assert!(jpeg.starts_with(&[0xFF, 0xD8]));
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1, 0x00, 0x22]);
        out.extend_from_slice(b"Exif\0\0");
        // Little-endian TIFF: header (8) + IFD (2 count + 12 entry + 4 next).
        out.extend_from_slice(&[
            0x49, 0x49, 0x2A, 0x00, // "II", 42
            0x08, 0x00, 0x00, 0x00, // IFD offset = 8
            0x01, 0x00, // 1 entry
            0x12, 0x01, 0x03, 0x00, // tag 0x0112, type SHORT
            0x01, 0x00, 0x00, 0x00, // count = 1
            0x06, 0x00, 0x00, 0x00, // value = 6 (Rotate90)
            0x00, 0x00, 0x00, 0x00, // next IFD = none
        ]);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn over_cap_jpeg_exif_orientation_applied() {
        use image::GenericImageView;
        use image::ImageDecoder;
        // 2x1 JPEG stored with EXIF orientation=6 (Rotate90): the compressed
        // output must come out 1x2 — a silently-rotated reference is exactly
        // the class of invisible corruption this pipeline exists to prevent.
        let src = image::RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        });
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode_image(&image::DynamicImage::ImageRgb8(src))
            .unwrap();
        let oriented = jpeg_with_exif_orientation(&jpeg);

        let reader = image::ImageReader::new(std::io::Cursor::new(&oriented))
            .with_guessed_format()
            .unwrap();
        assert_eq!(
            reader.into_decoder().unwrap().orientation().unwrap(),
            image::metadata::Orientation::Rotate90
        );

        let out = compress_reference_step(&oriented, 0).unwrap();
        let decoded = image::load_from_memory(&out).unwrap();
        assert_eq!(decoded.dimensions(), (1, 2));
    }
}

#[cfg(test)]
mod executable_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn test_is_executable_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_exe");

        assert!(!is_executable(&file_path));

        std::fs::write(&file_path, "content").unwrap();
        std::fs::set_permissions(&file_path, PermissionsExt::from_mode(0o644)).unwrap();
        assert!(
            !is_executable(&file_path),
            "File with mode 644 should not be executable"
        );

        crate::util::test::make_executable(&file_path);
        assert!(
            is_executable(&file_path),
            "File with mode 755 should be executable"
        );

        // Also test with only owner execute bit.
        std::fs::set_permissions(&file_path, PermissionsExt::from_mode(0o100)).unwrap();
        assert!(
            is_executable(&file_path),
            "File with mode 100 should be executable"
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_is_executable_on_windows() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_exe.exe");

        assert!(!is_executable(&file_path));

        std::fs::write(&file_path, "content").unwrap();
        assert!(
            is_executable(&file_path),
            "File with .exe extension should be executable"
        );

        let txt_path = dir.path().join("test.txt");
        std::fs::write(&txt_path, "content").unwrap();
        assert!(
            !is_executable(&txt_path),
            "File with .txt extension should not be executable"
        );
    }
}

#[cfg(test)]
mod verify_sha256_tests {
    use super::verify_sha256;

    fn digest(data: &[u8]) -> String {
        use sha2::{Digest as _, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        super::hex_string(&hasher.finalize())
    }

    #[test]
    fn a_file_is_checked_against_its_pin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"test data").unwrap();
        assert!(verify_sha256(&path, &digest(b"test data")).is_ok());
        assert!(verify_sha256(&path, &digest(b"other data")).is_err());
    }

    #[test]
    fn an_empty_pin_skips_the_check() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.bin");
        assert!(verify_sha256(&missing, "").is_ok());
        assert!(verify_sha256(&missing, &digest(b"anything")).is_err());
    }
}
