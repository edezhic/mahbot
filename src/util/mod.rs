//! Utility modules for shared helper functions.

pub(crate) mod error;
pub(crate) mod html;
pub(crate) mod http;
pub(crate) mod json;
pub(crate) mod macros;
pub(crate) mod model_state;
#[cfg(test)]
pub(crate) mod test;
pub(crate) mod tree_sitter;

use directories::UserDirs;
use regex::Regex;
use regex::RegexBuilder;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context as _, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use rand::{RngExt, SeedableRng};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use tracing::error;

/// Extension trait to unwrap poisoned lock results, replacing
/// `.unwrap_or_else(std::sync::PoisonError::into_inner)` with `.unwrap_poison()`.
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

/// The regex pattern for `[KIND:path]` media markers.
///
/// This is the single source of truth for the marker pattern. Both the case-sensitive
/// [`MEDIA_MARKER_RE`] and the case-insensitive [`TELEGRAM_MEDIA_MARKER_RE`] are built
/// from this constant, so adding a new marker kind here automatically keeps both in sync.
const MEDIA_MARKER_PATTERN: &str = r"\[(?P<kind>IMAGE|AUDIO|VIDEO):(?P<path>[^\]]+)\]";

/// Matches `[IMAGE:path]`, `[AUDIO:path]`, or `[VIDEO:path]` markers in message content.
///
/// **Invariant — multimodal stripping:** When enriching messages in multimodal
/// mode, IMAGE markers are preserved (they're needed for vision API integration
/// via `to_message_content()`), while all non-IMAGE markers (AUDIO, VIDEO, and
/// any future marker kinds) are stripped from the content. This is enforced by
/// the marker-stripping logic at the end of `enrich_message` which mirrors the
/// `parse_image_markers()` pattern. Adding a new marker kind to this regex will
/// cause it to be automatically stripped in multimodal mode unless the closure
/// is explicitly updated to preserve it.
pub(crate) static MEDIA_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(MEDIA_MARKER_PATTERN).expect("MEDIA_MARKER_RE must compile"));

/// Case-insensitive variant of [`MEDIA_MARKER_RE`] used by `telegram.rs` to
/// accept `[image:...]`, `[Image:...]`, etc. Built from the same
/// [`MEDIA_MARKER_PATTERN`] constant to stay in sync.
pub(crate) static TELEGRAM_MEDIA_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(MEDIA_MARKER_PATTERN)
        .case_insensitive(true)
        .build()
        .expect("TELEGRAM_MEDIA_MARKER_RE must compile")
});

/// Extract the `kind` and `path` named groups from a [`MEDIA_MARKER_RE`] / [`TELEGRAM_MEDIA_MARKER_RE`] capture.
///
/// Returns `(kind, path)` as string slices borrowed from the original haystack.
///
/// # Panics
///
/// Panics if either named group is missing — this should never happen with the
/// well-formed regex since the pattern requires both groups to match.
#[must_use]
pub(crate) fn parse_media_marker<'h>(caps: &regex::Captures<'h>) -> (&'h str, &'h str) {
    let kind = caps
        .name("kind")
        .expect("parse_media_marker: expected 'kind' group")
        .as_str();
    let path = caps
        .name("path")
        .expect("parse_media_marker: expected 'path' group")
        .as_str();
    (kind, path)
}

/// Truncate a string to `max_chars` Unicode characters, appending "…" if truncated.
#[must_use]
pub fn truncate(input: &str, max_chars: usize) -> String {
    match input.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", input[..idx].trim_end()),
        None => input.to_string(),
    }
}

/// Convert a string reference to `None` if empty, otherwise `Some(s.to_string())`.
///
/// Useful when building query structs where empty filters mean "no filter".
#[must_use]
pub(crate) fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Current Unix timestamp in milliseconds since the epoch.
///
/// Returns `0` if the system clock is set before the Unix epoch (January 1, 1970).
///
/// Returns `u64` — sufficient for timestamps up to ~500 million years from now.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Format a byte slice as a lowercase hex string.
///
/// Each byte is written as two hex digits, yielding a string of length
/// `bytes.len() * 2`.
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

/// Verify a file's SHA256 hash matches the expected hex string.
///
/// Shared model-integrity verifier extracted from the three near-identical
/// private streaming copies in `audio::tts`, `audio::local_transcriber`, and
/// `audio::voice` (whose empty-`expected` skip semantics and error wording had
/// drifted).  Model-download integrity is security-adjacent, so the copies are
/// consolidated here.
///
/// * If `expected` is empty, verification is skipped (returns `Ok`) — the
///   canonical "no hash configured" semantics that won the reconciliation.
/// * Uses streaming SHA256 via [`Sha256::update`] to avoid loading the entire
///   file into memory (model files can be multiple GB).
///
/// Note: unifying the error wording changes user-facing log text at the former
/// voice.rs call sites (e.g. `"Mel spectrogram model corrupt, re-downloading:
/// {e}"` now shows the path-qualified mismatch message).
pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    if expected.is_empty() {
        return Ok(()); // no hash configured — skip verification
    }

    let mut hasher = Sha256::new();
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open {} for SHA256 verification", path.display()))?;
    let mut buf = vec![0u8; 65536]; // 64 KB heap buffer
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

/// Expand a leading tilde (`~`) to the user's home directory.
///
/// Checks `$HOME` first (Unix, Git Bash on Windows), then `$USERPROFILE`
/// (cmd.exe / PowerShell). If neither is set, returns the path unchanged
/// (which means `~`-prefixed entries will be skipped by callers that
/// check for expansion success).
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

/// Resolve the shared `~/.mahbot/models/` directory via the CONFIG storage root.
///
/// Returns `None` if the storage root hasn't been initialized yet.  Per-model
/// subdirectories are joined by each consumer (e.g. `audio::voice::model_dir`).
#[must_use]
pub(crate) fn models_dir() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .map(|root| root.join("models"))
}

/// Run a blocking I/O operation with awareness of the current Tokio runtime.
///
/// - **Multi-threaded runtime:** wraps the call in
///   [`tokio::task::block_in_place`] so the runtime can re-schedule the
///   blocking thread to other tasks.
/// - **Current-thread runtime** or **no runtime:** calls `f()` directly —
///   blocking is safe in those contexts, and `block_in_place` would panic
///   on a current-thread runtime.
///
/// Use this instead of a bare `std::fs::canonicalize` (or other fast blocking
/// syscall) inside async functions that may run on a multi-threaded worker
/// pool. Prefer this over [`tokio::task::spawn_blocking`] for operations that
/// complete in < ~1 ms (where thread-spawn overhead dominates).
#[must_use]
pub(crate) fn with_block_in_place<T>(f: impl FnOnce() -> T) -> T {
    if let Ok(handle) = tokio::runtime::Handle::try_current()
        && handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread
    {
        return tokio::task::block_in_place(f);
    }
    f()
}

/// Produce a short human-readable summary of tool arguments.
#[must_use]
pub fn summarize_args(args: &serde_json::Value) -> String {
    match args {
        serde_json::Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    let val = match v {
                        serde_json::Value::String(s) => truncate(s, 80),
                        other => truncate(&other.to_string(), 80),
                    };
                    format!("{k}: {val}")
                })
                .collect();
            parts.join(", ")
        }
        other => truncate(&other.to_string(), 120),
    }
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

/// Log panics/cancellations from a `join_all`-aggregated batch of spawned
/// task results, keeping the enclosing loop alive.
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

/// Truncate a string to at most `max_bytes` bytes using a head/tail
/// "sandwich" strategy: keeps the first ~2/3 and last ~1/3, inserting an
/// omission marker between them. Returns the input unchanged if it fits
/// within the limit.
///
/// The marker format is `"... (N bytes omitted at {label} truncation)\n"`,
/// where `label` provides context for the truncation (e.g., `"shell output"`,
/// `"tool output"`, `"stderr"`).
///
/// Slicing respects UTF-8 character boundaries via `floor_char_boundary`.
/// An overlap guard is included as defense-in-depth; it only triggers if
/// the head and tail ranges would intersect (impossible under the 2/3 + 1/3
/// split, but guards against future ratio changes).
#[must_use]
pub fn truncate_sandwich(s: &str, max_bytes: usize, label: &str) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head_bytes = max_bytes * 2 / 3;
    let tail_bytes = max_bytes / 3;
    let head_end = s.floor_char_boundary(head_bytes);
    let tail_start = s.floor_char_boundary(s.len().saturating_sub(tail_bytes));
    if head_end < tail_start {
        let omitted = s[head_end..tail_start].len();
        format!(
            "{}... ({} bytes omitted at {label} truncation)\n{}",
            &s[..head_end],
            omitted,
            &s[tail_start..]
        )
    } else {
        // Head and tail would overlap — simple truncation fallback
        let boundary = s.floor_char_boundary(max_bytes);
        let mut out = s[..boundary].to_string();
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("\n... [{label} truncated at {max_bytes} bytes]"),
        );
        out
    }
}

/// Shared byte budget for tool output passed to the LLM: used by the shell
/// spill threshold, the read tool preview cap, and [`truncate_tool_output`].
pub(crate) const TOOL_OUTPUT_BUDGET_BYTES: usize = 5_000;

/// Truncate tool output for LLM consumption (delegates to [`truncate_sandwich`]
/// with the shared [`TOOL_OUTPUT_BUDGET_BYTES`] limit). Returns input unchanged if within limit.
#[must_use]
pub fn truncate_tool_output(output: &str) -> String {
    truncate_sandwich(output, TOOL_OUTPUT_BUDGET_BYTES, "tool output")
}

/// Read a local image file and return a base64 data URI suitable for native
/// multimodal model input (e.g., `data:image/png;base64,...`).
pub(crate) async fn local_image_to_data_uri(path: &std::path::Path) -> anyhow::Result<String> {
    let bytes = tokio::fs::read(path).await?;
    let mime = mime_for_extension(path);
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(&bytes)))
}

/// Load a reference image from disk, validate it does not exceed `max_bytes`,
/// and return a base64 data URI suitable for multimodal model input.
#[allow(clippy::cast_precision_loss)]
pub(crate) async fn load_reference_image(
    path: &std::path::Path,
    max_bytes: u64,
) -> anyhow::Result<String> {
    if !path.exists() {
        anyhow::bail!("Reference image not found: {}", path.display());
    }
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read reference image {}: {e}", path.display()))?;
    if metadata.len() > max_bytes {
        let mb = max_bytes as f64 / (1024.0 * 1024.0);
        anyhow::bail!(
            "Reference image {} is {} bytes, exceeds {:.1} MB limit. \
             Use a smaller or compressed image.",
            path.display(),
            metadata.len(),
            mb,
        );
    }
    local_image_to_data_uri(path).await
}

/// Map a file path's extension to a MIME type string.
fn mime_for_extension(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    }
}

/// Strip ANSI escape sequences from a string.
///
/// Removes common ANSI escape codes used for terminal text formatting (colors,
/// bold, underline, cursor movement, etc.) while preserving the visible content.
/// This is useful when processing shell command output or any text that may
/// contain terminal control sequences.
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

/// Redact sensitive values for safe logging. Shows first 4 characters + "*[REDACTED]" suffix.
/// Uses char-boundary-safe indexing to avoid panics on multi-byte UTF-8 strings.
static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(token|api[_-]?key|password|secret|user[_-]?key|bearer|credential)["']?\s*[:=]\s*(?:"([^"]{8,})"|'([^']{8,})'|([a-zA-Z0-9_\-\./+=]{8,}))"#).expect("hardcoded regex is valid")
});

/// Scrub credentials from tool output to prevent accidental exfiltration.
/// Replaces known credential patterns with a redacted placeholder while preserving
/// a small prefix for context.
#[must_use]
pub fn scrub_credentials(input: &str) -> String {
    SENSITIVE_KV_REGEX
        .replace_all(input, |caps: &regex::Captures| {
            let full_match = &caps[0];
            let key = &caps[1];
            let val = caps
                .get(2)
                .or(caps.get(3))
                .or(caps.get(4))
                .map_or("", |m| m.as_str());

            // Preserve first 4 chars for context, then redact.
            debug_assert!(val.len() >= 8, "regex guarantees values >= 8 chars");
            let prefix = val
                .char_indices()
                .nth(4)
                .map_or(val, |(byte_idx, _)| &val[..byte_idx]);

            // Determine quote style from which capture group matched the value.
            // Group 2 = double-quoted, Group 3 = single-quoted, Group 4 (else) = unquoted.
            // Using capture-group identity avoids false positives from quotes/apostrophes
            // appearing elsewhere in the match (e.g., a double-quoted key name with a
            // single-quoted value, or an apostrophe in a key like `don't_share`).
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

/// Resolve the cargo bin directory.
///
/// Resolution order:
/// 1. `$CARGO_HOME/bin` if `CARGO_HOME` environment variable is set and non-empty.
/// 2. `~/.cargo/bin` using `directories::UserDirs`.
/// 3. `None` — no home directory available.
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

/// Strip surrounding double-quotes and unescape C-style escapes.
///
/// If the input starts with `"` and ends with `"`, strips the quotes and
/// calls `unescape_c_style` on the inner content. Otherwise returns the
/// input as-is (no unescaping needed — git only C-quotes paths that contain
/// trigger characters).
///
/// This is the standard pattern for handling git's quoted path output
/// (the same approach as git's own `unquote_c_style`).
#[must_use]
pub fn unquote_c_style(raw: &str) -> Option<String> {
    if let Some(inner) = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        unescape_c_style(inner)
    } else {
        Some(raw.to_string())
    }
}

/// Unescape C-style escape sequences from a git path name.
///
/// Supports the same escapes as git's `unquote_c_style`:
/// - `\"` → literal `"`, `\\` → literal `\`
/// - `\t` → tab, `\n` → newline, `\a` → bell, `\b` → backspace
/// - `\f` → form feed, `\r` → carriage return, `\v` → vertical tab
/// - `\0`–`\3` followed by 1–3 octal digits → byte value
///
/// Malformed escapes cause this function to return `None`:
/// - `\` at end of string (dangling backslash)
/// - `\x` or any other unrecognized escape letter
/// - `\4`–`\7` followed by a digit (git rejects these as invalid octal prefixes)
///
/// Non-UTF-8 bytes produced by octal escapes are handled via
/// `String::from_utf8_lossy` — pragmatic for macOS where non-UTF-8 paths
/// are filesystem-impossible.
fn unescape_c_style(input: &str) -> Option<String> {
    let mut result = String::with_capacity(input.len());
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
                b'"' => result.push('"'),
                b'\\' => result.push('\\'),
                b't' => result.push('\t'),
                b'n' => result.push('\n'),
                b'a' => result.push('\x07'),
                b'b' => result.push('\x08'),
                b'f' => result.push('\x0c'),
                b'r' => result.push('\r'),
                b'v' => result.push('\x0b'),
                b'0'..=b'3' => {
                    // Octal escape: 1–3 octal digits.
                    let digits_start = i;
                    i += 1;
                    let mut digit_count = 1;
                    while digit_count < 3 && i < bytes.len() && bytes[i].is_ascii_digit() {
                        if !(b'0'..=b'7').contains(&bytes[i]) {
                            break;
                        }
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
                    result.push_str(&String::from_utf8_lossy(&[byte_val]));
                    continue; // skip the i += 1 at end of loop
                }
                b'4'..=b'7' => {
                    // \4–\7 are not valid octal prefixes in git's unquote_c_style.
                    // If followed by a digit, it's a malformed octal attempt.
                    if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                        tracing::warn!(
                            input = %input,
                            ch = %(bytes[i] as char),
                            "unescape_c_style: invalid octal prefix \\4–\\7 followed by digit"
                        );
                        return None;
                    }
                    // Otherwise: literal digit (backslash consumed, no special meaning).
                    result.push(bytes[i] as char);
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
            result.push(bytes[i] as char);
        }
        i += 1;
    }

    Some(result)
}

/// Resample PCM audio from one sample rate to another using linear
/// interpolation with a 3-tap binomial anti-aliasing filter for downsampling.
///
/// This is the canonical implementation. All other resample functions in the
/// codebase delegate to this one.
///
/// When `from_rate > to_rate` (downsampling), a simple binomial low-pass filter
/// is applied to attenuate frequencies above the new Nyquist before decimation.
/// Without this filter, linear interpolation introduces aliasing — high-frequency
/// content above `to_rate / 2` folds back into the audible range as noise.
///
/// The 3-tap binomial `[0.25, 0.5, 0.25]` gives reasonable stopband attenuation
/// (~6 dB at 0.25 normalised) for speech audio. For 48 kHz → 16 kHz this
/// attenuates content above ~8 kHz.
///
/// # Aliasing trade-off
///
/// Linear interpolation introduces aliasing when downsampling even with the
/// pre-filter — the filter only provides ~6 dB stopband attenuation. This is
/// acceptable for speech processing and wake word training data augmentation.
/// If aliasing artifacts prove problematic, a sinc-based resampler can be
/// substituted here — all call sites benefit automatically.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;

    // Anti-aliasing filter for downsampling
    let filtered: Vec<f32> = if from_rate > to_rate && samples.len() >= 3 {
        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0] * 0.75 + samples[1] * 0.25);
        for i in 1..samples.len() - 1 {
            out.push(samples[i - 1] * 0.25 + samples[i] * 0.5 + samples[i + 1] * 0.25);
        }
        out.push(samples[samples.len() - 2] * 0.25 + samples[samples.len() - 1] * 0.75);
        out
    } else {
        samples.to_vec()
    };

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < filtered.len() {
            output.push(
                (f64::from(filtered[src_idx]) * (1.0 - frac)
                    + f64::from(filtered[src_idx + 1]) * frac) as f32,
            );
        } else if src_idx < filtered.len() {
            output.push(filtered[src_idx]);
        } else {
            output.push(0.0);
        }
    }
    output
}

#[cfg(test)]
mod truncate_tests {
    use super::*;

    // ── truncate_sandwich: passthrough ────────────────────────────────────

    #[test]
    fn passthrough_under_limit() {
        let input = "hello world";
        let result = truncate_sandwich(input, 5_000, "test");
        assert_eq!(
            result, input,
            "should pass through unchanged when under limit"
        );
    }

    #[test]
    fn passthrough_at_exact_limit() {
        let input = "a".repeat(5_000);
        assert_eq!(input.len(), 5_000);
        let result = truncate_sandwich(&input, 5_000, "test");
        assert_eq!(result, input, "exact limit should pass through unchanged");
    }

    // ── truncate_sandwich: head/tail sandwich ─────────────────────────────

    #[test]
    fn sandwich_just_over_limit() {
        // Input is exactly limit+1 bytes — sandwich marker may add overhead
        // making output longer than input, which is expected for tiny overshoot.
        let input = "x".repeat(5_001);
        let result = truncate_sandwich(&input, 5_000, "test");
        assert!(
            result.starts_with("xxx"),
            "head portion should be preserved"
        );
        assert!(
            result.contains("bytes omitted at test truncation"),
            "should contain the omission marker"
        );
        assert!(result.ends_with('x'), "tail should contain input suffix");
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
        // Head portion appears
        assert!(
            result.starts_with("hello world"),
            "head should be preserved"
        );
        // Tail portion appears
        let last_line = result.lines().last().unwrap_or("");
        assert_eq!(last_line, "hello world", "tail should be preserved");
    }

    #[test]
    fn sandwich_preserves_utf8_boundaries() {
        // Place a multibyte char ('🐱', 4 bytes) right at the head/tail
        // boundary so it straddles the cut point. floor_char_boundary must
        // back up to the character boundary. Build: 3329 'x's, then 🐱
        // (bytes 3329-3332), then 'y's. head_bytes ≈ 3333, so 🐱 is the
        // last complete char in head. Verify it appears intact.
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
    fn sandwich_line_boundaries_intact() {
        // Lines should not be concatenated across truncation boundaries
        let line = "hello world!\n".repeat(100_000);
        let result = truncate_sandwich(&line, 500_000, "test");
        assert!(result.len() < line.len(), "should truncate");
        for l in result.lines().filter(|l| !l.starts_with("...")) {
            assert!(
                !l.contains("hello world!hello"),
                "lines should not be concatenated"
            );
        }
    }

    // ── truncate_sandwich: custom label ───────────────────────────────────

    #[test]
    fn custom_label_appears_in_marker() {
        let input = "x".repeat(10_000);
        let result = truncate_sandwich(&input, 5_000, "my custom label");
        assert!(
            result.contains("bytes omitted at my custom label truncation"),
            "custom label should appear verbatim in marker"
        );
    }

    #[test]
    fn empty_label() {
        let input = "x".repeat(10_000);
        let result = truncate_sandwich(&input, 5_000, "");
        assert!(
            result.contains("bytes omitted at  truncation"),
            "empty label should still produce coherent marker"
        );
    }

    // ── truncate_tool_output compatibility ──────────────────────────────────

    #[test]
    fn truncate_tool_output_appends_correct_label() {
        let input = "abc".repeat(2_000); // 6_000 bytes > 5_000 limit
        let result = truncate_tool_output(&input);
        assert!(result.len() < input.len(), "should truncate");
        assert!(
            result.contains("bytes omitted at tool output truncation"),
            "should use 'tool output' label"
        );
        assert!(result.starts_with("abcabc"), "head should be preserved");
    }
}

// ── scrub_credentials tests ────────────────────────────────────────────

#[cfg(test)]
mod scrub_tests {
    use super::scrub_credentials;

    #[test]
    fn scrub_redacts_credentials() {
        /// Cases verifying `[REDACTED]` appears, with optional negative and
        /// prefix checks. Fields: (name, input, must_not_contain, must_start_with).
        /// Empty string for must_not_contain/must_start_with = skip check.
        const CASES: &[(&str, &str, &str, &str)] = &[
            (
                "alphanumeric unquoted value",
                "API_KEY=sk-1234567890abcdef",
                "1234567890abcdef",
                "API_KEY=sk-1",
            ),
            // Standard Base64-encoded secret containing +, /, =
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
            // Hyphen-key variant: regex `user[_-]?key` also matches `user-key`.
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
        /// Cases verifying exact output strings. Exact match is the strictest
        /// assertion — it subsumes containment and non-leakage checks.
        /// Fields: (name, input, expected_output).
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
            // Edge case: the key-level optional quote in the regex can produce
            // full_match containing a double-quote from the key suffix, e.g.
            // "password": 'secretvalue1234'. The capture-group approach correctly
            // identifies this as a single-quoted value despite the double-quote
            // appearing in the full match string.
            // Note: the key-suffix " is consumed by the regex match and not
            // reconstructed — this is a pre-existing cosmetic issue also present
            // in the double-quote path, and out of scope for this fix.
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
        /// Cases where the input is not a credential pattern and must pass
        /// through unchanged. Fields: (name, input).
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
}

#[cfg(test)]
mod unescape_c_style_tests {
    use super::unescape_c_style;

    #[test]
    fn test_unescape_c_style() {
        // Cases: (input, expected_output).
        // Uses Option<&str> — compared via .as_deref() against the
        // function's Option<String> return type.
        let cases: &[(&str, Option<&str>)] = &[
            // ── basic escapes ──
            (
                r#"hello\"world\\test\nline\there"#,
                Some("hello\"world\\test\nline\there"),
            ),
            // Bell, backspace, formfeed, CR, vertical tab.
            (r"\a\b\f\r\v", Some("\x07\x08\x0c\r\x0b")),
            // ── octal escapes, 1–3 digits ──
            // \0 → NUL (0x00), \1 → SOH (0x01)
            (r"\0\1", Some("\0\x01")),
            // \12 → newline (0x0a), \37 → unit separator (0x1f)
            (r"\12\37", Some("\n\x1f")),
            // \101 → 'A' (0x41), \377 → 0xff → U+FFFD (from_utf8_lossy replacement)
            (r"\101\377", Some("A\u{FFFD}")),
            // Octal stops at non-octal-digit: \12x → newline + 'x'
            (r"\12x", Some("\nx")),
            // \18 → \1 (SOH, 0x01) then '8' (8 is not an octal digit)
            (r"\18", Some("\x018")),
            // ── no-op cases (no escape sequences) ──
            ("plain/path.rs", Some("plain/path.rs")),
            ("", Some("")),
            // ── error cases (return None) ──
            // Dangling backslash at end of string.
            (r"path\", None),
            // \x looks like a hex escape prefix but git's unquote_c_style rejects it.
            (r"\x", None),
            // \q is not a recognized escape sequence.
            (r"\q", None),
            // \40 — \4 followed by digit (git rejects as invalid octal prefix).
            (r"\40", None),
            // \77 — \7 followed by digit (git rejects as invalid octal prefix).
            (r"\77", None),
            // \70 — \7 followed by digit (git rejects as invalid octal prefix).
            (r"\70", None),
            // ── literal-digit cases (\4/\7 not followed by digit) ──
            // \4 at end of string → literal '4'.
            (r"\4", Some("4")),
            // \7 followed by non-digit → literal '7' then 'x'.
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

// Audio utility functions (canonical implementations)

/// Generate pink noise (1/f spectrum) using the Voss-McCartney algorithm.
///
/// Uses a seeded RNG for reproducibility.  The high-pass delta variant
/// naturally removes DC bias.  Output is normalized to unit RMS.
///
/// # Voss-McCartney variants
///
/// There are several common variants of Voss-McCartney pink noise:
///
/// * **High-pass delta** (this implementation): stores previous value per
///   octave, emits the difference (new - prev).  Removes DC bias naturally.
/// * **Direct-sum**: sums all octave values directly.  May accumulate DC bias.
///
/// The canonical implementation uses 16 octaves (~3 dB/octave rolloff down
/// to 0.03 Hz at 16 kHz) and the high-pass delta variant for DC-free output.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn generate_pink_noise(len: usize, mut rng: impl rand::Rng) -> Vec<f32> {
    const NUM_OCTAVES: usize = 16;
    let mut values = [0.0f32; NUM_OCTAVES];
    let mut outputs = [0.0f32; NUM_OCTAVES];
    let mut sample_count = 0u64;
    let mut noise = Vec::with_capacity(len);

    for _ in 0..len {
        sample_count += 1;
        let mut sum = 0.0;
        for octave in 0..NUM_OCTAVES {
            // Update this octave's generator at intervals of 2^octave samples.
            if sample_count.is_multiple_of(1u64 << octave) {
                values[octave] = rng.random::<f32>() * 2.0 - 1.0;
            }
            // High-pass delta: emit difference instead of direct value.
            let new_val = values[octave];
            let delta = new_val - outputs[octave];
            outputs[octave] = new_val;
            sum += delta;
        }
        noise.push(sum);
    }

    // Normalize to unit RMS
    let rms = compute_rms(&noise).max(1e-10);
    for s in &mut noise {
        *s /= rms;
    }

    noise
}

/// Pink-noise alias of [`add_noise_color`] — kept for the recipe's variant-4
/// call sites; the SNR-scaling/clamp arithmetic lives in one place.
pub(crate) fn add_noise(pcm: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    add_noise_color(pcm, snr_db, NoiseColor::Pink, seed)
}

/// Apply a fixed gain to PCM audio.
///
/// DETERMINISTIC — no RNG involved, unlike `randomize_volume`.
/// The gain is `10^(gain_db / 20)`.  Negative values attenuate,
/// positive values amplify.
pub(crate) fn apply_gain(pcm: &[f32], gain_db: f32) -> Vec<f32> {
    let amp = 10.0_f32.powf(gain_db / 20.0);
    pcm.iter().map(|&s| s * amp).collect()
}

/// Available noise types for augmentation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(feature = "voice-tests")]
pub(crate) enum NoiseType {
    /// White noise (uniform spectral density).
    White,
}

/// Noise colors supported by the shared augmentation recipe (ungated —
/// production enrollment trains on these cells).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoiseColor {
    White,
    Pink,
    Brown,
}

/// Add color noise to PCM audio at the given SNR (deterministic, seeded).
///
/// Mirrors [`add_noise`]'s arithmetic (unit-RMS noise scaled to the SNR
/// target, clamped mix) with the color selector.  `Brown` is a leaky
/// integration of white noise (DC-free-ish, low-frequency dominant).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn add_noise_color(pcm: &[f32], snr_db: f32, color: NoiseColor, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let signal_rms = compute_rms(pcm).max(1e-10);
    let noise: Vec<f32> = match color {
        NoiseColor::White => (0..pcm.len())
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect(),
        NoiseColor::Pink => generate_pink_noise(pcm.len(), &mut rng),
        NoiseColor::Brown => {
            let mut acc = 0.0f32;
            (0..pcm.len())
                .map(|_| {
                    acc = 0.999 * acc + (rng.random::<f32>() * 2.0 - 1.0);
                    acc
                })
                .collect()
        }
    };
    let noise_rms_target = signal_rms * 10.0_f32.powf(-snr_db / 20.0);
    let noise_rms_current = compute_rms(&noise).max(1e-10);
    let scale = noise_rms_target / noise_rms_current;
    pcm.iter()
        .zip(noise.iter())
        .map(|(&s, &n)| (s + n * scale).clamp(-1.0, 1.0))
        .collect()
}

/// Compute the RMS (root mean square) of audio samples.
///
/// Returns `0.0` for empty input.  Ungated so production hot
/// paths (AGC, noise generation, utterance quality) share one implementation
/// instead of hand-rolling `sum(x²)/n → sqrt`.  Callers that need a
/// divide-by-zero floor for degenerate all-zero input apply `.max(1e-10)` at
/// the call site — it is deliberately NOT part of this function.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute both Box-Muller branches from two uniform draws in `(0, 1]`.
///
/// `z1 = sqrt(-2 ln u1) cos(2π u2)`, `z2 = sqrt(-2 ln u1) sin(2π u2)`.
/// Shared math for the bench's EPSILON-clamp pair sampler below.
#[must_use]
#[inline]
pub(crate) fn gaussian_pair_from_uniforms(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Draw a standard-normal pair via Box-Muller with EPSILON-clamped draws
/// (2 draws per 2 samples).  Preserves the bench's draw sequence and
/// degenerate-input semantics (`u1`/`u2` floored at [`f32::EPSILON`] rather
/// than re-rolled).  Used by the `voice-tests` bench and the seeded-sequence
/// equivalence test.
#[cfg_attr(not(any(feature = "voice-tests", test)), allow(dead_code))]
pub(crate) fn sample_gaussian_pair_clamped(rng: &mut impl rand::Rng) -> (f32, f32) {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
    gaussian_pair_from_uniforms(u1, u2)
}

/// Mix white or pink noise into audio at a given SNR (dB).
///
/// When `rng_seed` is `Some(seed)`, uses a deterministic seeded RNG for
/// reproducible output. When `None`, uses an entropy-seeded RNG (current
/// non-deterministic behavior). The deterministic path is used by the
/// benchmark to ensure stable training data across re-runs (mahbot-906).
///
/// # Arguments
///
/// * `samples` — Clean audio PCM f32 in [-1.0, 1.0].
/// * `snr_db` — Desired signal-to-noise ratio in dB (typical: 10-25).
///   Lower values = more noise. Must be finite.
/// * `noise_type` — [`NoiseType::White`].
/// * `rng_seed` — Optional seed for deterministic RNG. `Some(seed)` produces
///   reproducible noise; `None` uses entropy-based seeding.
///
/// # Returns
///
/// Noisy audio PCM f32 in [-1.0, 1.0] (clamped).
///
/// # Note (mahbot-1029)
///
/// This is the former `tts_data_gen::add_noise` (4-arg) relocated unchanged.
/// It is a DIFFERENT implementation from the 3-arg [`add_noise`] above
/// (different RNG consumption and degenerate-signal behavior); both feed
/// the bench's seeded embeddings, so they must never be merged.
#[must_use]
#[cfg(feature = "voice-tests")]
pub(crate) fn add_noise_white_pink(
    samples: &[f32],
    snr_db: f32,
    noise_type: NoiseType,
    rng_seed: Option<u64>,
) -> Vec<f32> {
    if samples.is_empty() || !snr_db.is_finite() {
        return samples.to_vec();
    }

    // Create a seeded or entropy-based RNG.
    // When None, we seed from rand::random() rather than using the
    // thread-local generator directly, isolating RNG state.
    let mut rng: rand::rngs::StdRng = match rng_seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::seed_from_u64(rand::random()),
    };

    // Generate noise using the single RNG.
    let noise: Vec<f32> = match noise_type {
        NoiseType::White => {
            // Uniform white noise in [-1.0, 1.0]
            (0..samples.len())
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect()
        }
    };

    // Compute RMS of signal and noise
    let signal_rms = compute_rms(samples);
    let noise_rms = compute_rms(&noise);

    if signal_rms <= 1e-10 || noise_rms <= 1e-10 {
        return samples.to_vec(); // Degenerate case — no scaling
    }

    // SNR = 20 * log10(signal_rms / noise_rms * scale)
    // scale = signal_rms / noise_rms * 10^(-SNR/20)
    let scale = (signal_rms / noise_rms) * 10.0_f32.powf(-snr_db / 20.0);

    // Mix
    let mut result = Vec::with_capacity(samples.len());
    for (&s, &n) in samples.iter().zip(noise.iter()) {
        result.push((s + n * scale).clamp(-1.0, 1.0));
    }
    result
}

/// Apply speed perturbation by resampling.
///
/// Changes both speed and pitch (time-domain resampling). For wake word
/// training data diversity, this is acceptable and does not require
/// pitch-preserving time-stretching.
///
/// # Arguments
///
/// * `samples` — Audio PCM f32 at `sample_rate`.
/// * `sample_rate` — Original sample rate in Hz.
/// * `factor` — Speed factor: >1.0 = faster (fewer samples), <1.0 = slower
///   (more samples). Typical range: 0.8-1.2 (±20%).
///
/// # Returns
///
/// Speed-adjusted audio at the original `sample_rate`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn speed_perturbation(samples: &[f32], sample_rate: u32, factor: f32) -> Vec<f32> {
    if samples.is_empty() || (factor - 1.0).abs() < 1e-6 {
        return samples.to_vec();
    }
    // Speed perturbation via resampling: change the effective rate
    // new_rate = sample_rate * factor
    let effective_rate = (sample_rate as f32 * factor) as u32;
    // Resample from effective_rate back to sample_rate
    // This produces the same duration as original but with shifted pitch
    resample_audio(samples, effective_rate, sample_rate)
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

// ── Shared model-integrity verifier ─────────────────────────
// Tests consolidated from the three former private copies (tts,
// local_transcriber, voice).  The file-not-found error path is covered with a
// NON-empty expected hash because the shared empty-hash skip returns Ok before
// opening the file (sanctioned reconciliation of the transcriber's original
// `verify_sha256(path, "").is_err()` assertion).

#[cfg(test)]
mod verify_sha256_tests {
    use super::verify_sha256;

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        super::hex_string(&hasher.finalize())
    }

    #[test]
    fn matching_hash_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"test data").unwrap();
        let hash = sha256_hex(b"test data");
        assert!(verify_sha256(&path, &hash).is_ok());
    }

    #[test]
    fn mismatching_hash_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"different data").unwrap();
        let hash = sha256_hex(b"test data");
        assert!(verify_sha256(&path, &hash).is_err());
    }

    #[test]
    fn empty_hash_skips_verification() {
        // Empty expected hash → Ok without opening the file (skip semantics).
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.bin");
        assert!(
            verify_sha256(&missing, "").is_ok(),
            "empty SHA256 expected hash should skip verification"
        );
    }

    #[test]
    fn file_not_found_with_non_empty_hash_fails() {
        // The file-not-found error path: requires a non-empty expected hash
        // (the empty-hash skip above returns Ok before any file I/O).
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.bin");
        let hash = sha256_hex(b"anything");
        assert!(verify_sha256(&missing, &hash).is_err());
    }
}

// ── Audio utility regression tests ──────────────────────────
// Moved verbatim from voice.rs (post-1029 stranded util tests) so the util
// module has an in-place regression net for speed perturbation, gain, noise,
// and pink noise.

#[cfg(test)]
mod audio_util_tests {
    use super::{add_noise, apply_gain, generate_pink_noise, speed_perturbation};
    use rand::SeedableRng;

    #[test]
    fn test_speed_perturbation_identity() {
        // rate=1.0 should return approximately the original
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let result = speed_perturbation(&pcm, 16000, 1.0);
        assert_eq!(result.len(), pcm.len(), "identity should preserve length");
        for (a, b) in pcm.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5, "identity should preserve values");
        }
    }

    #[test]
    fn test_speed_perturbation_rates() {
        // Slow down: rate=0.5 should produce more samples
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let slowed = speed_perturbation(&pcm, 16000, 0.5);
        assert!(
            slowed.len() > pcm.len(),
            "rate < 1 should increase sample count"
        );
        // Speed up: rate=2.0 should produce fewer samples
        let sped_up = speed_perturbation(&pcm, 16000, 2.0);
        assert!(
            sped_up.len() < pcm.len(),
            "rate > 1 should decrease sample count"
        );
    }

    #[test]
    fn test_speed_perturbation_determinism() {
        // Same input + same rate → same output
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let a = speed_perturbation(&pcm, 16000, 0.95);
        let b = speed_perturbation(&pcm, 16000, 0.95);
        assert_eq!(a, b, "deterministic speed perturbation");
    }

    #[test]
    fn test_apply_gain_determinism() {
        // apply_gain must be deterministic: same PCM + same gain_db → same output
        let pcm: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) / 25.0).collect();
        let a = apply_gain(&pcm, -3.0);
        let b = apply_gain(&pcm, -3.0);
        assert_eq!(a, b, "apply_gain must be deterministic");
    }

    #[test]
    fn test_apply_gain_db_conversion() {
        // 0 dB → unity gain (output == input)
        let pcm: Vec<f32> = vec![0.5, -0.3, 0.1, -0.7, 0.9];
        let unity = apply_gain(&pcm, 0.0);
        for (a, b) in pcm.iter().zip(unity.iter()) {
            assert!((a - b).abs() < 1e-6, "0 dB gain should be unity");
        }
        // -6 dB → amplitude halved (10^(-6/20) ≈ 0.5)
        let attenuated = apply_gain(&pcm, -6.0);
        let expected_amp = 10.0_f32.powf(-6.0 / 20.0);
        for (orig, atten) in pcm.iter().zip(attenuated.iter()) {
            assert!(
                (atten - orig * expected_amp).abs() < 1e-6,
                "-6 dB gain should multiply by {expected_amp}"
            );
        }
        // +6 dB → amplitude doubled (10^(6/20) ≈ 2.0)
        let amplified = apply_gain(&pcm, 6.0);
        let expected_amp2 = 10.0_f32.powf(6.0 / 20.0);
        for (orig, amp) in pcm.iter().zip(amplified.iter()) {
            assert!(
                (amp - orig * expected_amp2).abs() < 1e-6,
                "+6 dB gain should multiply by {expected_amp2}"
            );
        }
    }

    #[test]
    fn test_add_noise_determinism() {
        // Same PCM + same SNR + same seed → same output
        let pcm: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 100.0).collect();
        let a = add_noise(&pcm, 25.0, 42);
        let b = add_noise(&pcm, 25.0, 42);
        assert_eq!(a.len(), b.len(), "add_noise output length should match");
        assert_eq!(a, b, "add_noise with same seed must be deterministic");
    }

    #[test]
    fn test_add_noise_seed_variation() {
        // Different seed → different output
        let pcm: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 100.0).collect();
        let a = add_noise(&pcm, 25.0, 42);
        let b = add_noise(&pcm, 25.0, 99);
        assert_ne!(a, b, "different seeds should produce different output");
    }

    #[test]
    fn test_add_noise_snr_approximation() {
        // At very high SNR (100 dB), the output should be very close to input
        let pcm: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();
        let noisy = add_noise(&pcm, 100.0, 42);
        // Compute actual SNR of output vs input
        let signal_power: f32 = pcm.iter().map(|&s| s * s).sum();
        let noise_power: f32 = pcm
            .iter()
            .zip(noisy.iter())
            .map(|(&s, &n)| (n - s) * (n - s))
            .sum();
        let actual_snr_db = 10.0 * (signal_power / noise_power.max(1e-10)).log10();
        assert!(
            actual_snr_db > 80.0,
            "at 100 dB target, actual SNR should be high (was {actual_snr_db:.1} dB)"
        );
    }

    #[test]
    fn test_generate_pink_noise_properties() {
        // Pink noise should have positive and negative values
        let noise = generate_pink_noise(1000, rand::rngs::StdRng::seed_from_u64(42));
        assert_eq!(noise.len(), 1000);
        let has_positive = noise.iter().any(|&s| s > 0.0);
        let has_negative = noise.iter().any(|&s| s < 0.0);
        assert!(has_positive, "pink noise should have positive values");
        assert!(has_negative, "pink noise should have negative values");
        // RMS should be near 1.0 (normalized)
        let rms = super::compute_rms(&noise);
        assert!(
            (rms - 1.0).abs() < 0.1,
            "pink noise RMS should be near 1.0 (was {rms})"
        );
    }

    #[test]
    fn test_augmentation_variants_are_different() {
        // Verify that the 4 augmentation strategies produce different results
        let pcm: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) / 250.0).collect();
        let speed_down = speed_perturbation(&pcm, 16000, 0.95);
        let speed_up = speed_perturbation(&pcm, 16000, 1.05);
        let volume_down = apply_gain(&pcm, -3.0);
        let noise = add_noise(&pcm, 25.0, 42);

        // Each variant should differ from original AND each other
        assert_ne!(speed_down, pcm, "speed-down should differ from original");
        assert_ne!(speed_up, pcm, "speed-up should differ from original");
        assert_ne!(volume_down, pcm, "volume-down should differ from original");
        assert_ne!(noise, pcm, "noise should differ from original");

        // Verify they also differ from each other (different transforms)
        assert_ne!(
            speed_down, speed_up,
            "speed-down and speed-up should differ"
        );
        assert_ne!(
            speed_down, volume_down,
            "speed-down and volume-down should differ"
        );
        assert_ne!(speed_down, noise, "speed-down and noise should differ");
    }
}

// ── Shared Gaussian sampler seeded-sequence equivalence ─────
// MANDATORY verification from the manager pin: the extraction must consume
// exactly the same RNG draws at every site so seeded outputs stay byte-identical.
// These tests prove the shared helper reproduces the pre-extraction inline
// EPSILON-clamp bench-family formula over ~8000 seeded draw pairs.

#[cfg(test)]
mod gaussian_sampler_tests {
    use super::{gaussian_pair_from_uniforms, sample_gaussian_pair_clamped};
    use rand::RngExt;
    use rand::SeedableRng;

    /// Reference: the bench's pre-extraction EPSILON-clamp pair sampler
    /// (2 draws per 2 samples, cos+sin).
    fn reference_bench_pair(rng: &mut impl rand::Rng) -> (f32, f32) {
        let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
        let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).sin();
        (z1, z2)
    }

    #[test]
    fn seeded_bench_sequence_byte_identical() {
        // Bench seed 43; 8000 samples = 4000 draw pairs (2 draws per 2 samples).
        const DRAW_PAIRS: usize = 4000;
        let mut shared_rng = rand::rngs::StdRng::seed_from_u64(43);
        let mut ref_rng = rand::rngs::StdRng::seed_from_u64(43);
        for i in 0..DRAW_PAIRS {
            let shared = sample_gaussian_pair_clamped(&mut shared_rng);
            let reference = reference_bench_pair(&mut ref_rng);
            assert_eq!(shared.0, reference.0, "bench z1 divergence at pair {i}");
            assert_eq!(shared.1, reference.1, "bench z2 divergence at pair {i}");
        }
    }

    #[test]
    fn pair_math_matches_inline_formulas() {
        // gaussian_pair_from_uniforms must equal the inline cos/sin formulas.
        for &(u1, u2) in &[(0.1, 0.2), (0.5, 0.9), (0.001, 0.999), (0.25, 0.75)] {
            let (z1, z2) = gaussian_pair_from_uniforms(u1, u2);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            assert_eq!(z1, r * theta.cos(), "z1 for ({u1}, {u2})");
            assert_eq!(z2, r * theta.sin(), "z2 for ({u1}, {u2})");
        }
    }
}
