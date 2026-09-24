//! Generic managed-binary release mechanics shared by the rust-installed tool
//! managers (chrome-use via `chrome_daemon`, the managed bun runtime).
//!
//! Each tool is installed at the location its own installer uses — the bun
//! runtime in its own standard user dir (`~/.bun/bin`), the browser helper
//! where its own installer puts it: on unix `/usr/local/bin` when it already
//! holds a copy or can be written, otherwise `~/.local/bin`, and on Windows the
//! `%LOCALAPPDATA%\Programs\chrome-use` its own installer uses — the vendor's
//! defaults for both are quoted at [`chrome_use_user_bin_dir`]. Nothing here puts
//! a location on the owner's own search path: that is `crate::util::owner_path`.
//!
//! Each install is updated in place via an atomic swap and verified against a
//! SHA-256 checksum published with the release.
//!
//! The helpers are release-format agnostic: the chrome-use installer reads a
//! `<asset>.sha256` sidecar and a tar.gz archive, the bun installer reads a
//! combined `SHASUMS256.txt` file and a zip archive — both share the same
//! tag/version/swap/extract primitives.
//!
//! [`install_on_start`] is the shared start-time policy: every tool is brought
//! to its newest release on every product start, straight away when no copy is
//! installed and after the boot has settled otherwise.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ── Release metadata ────────────────────────────────────────────────

/// Follow the GitHub `releases/latest` redirect and return the last path
/// segment (the release tag) — avoids the api.github.com rate limit. reqwest
/// follows the redirect by default; the final response URL is the tag page.
///
/// Every failure message carries the failing step and a classified reason and no
/// URL, so it is safe to record verbatim.
pub(crate) async fn fetch_latest_tag(repo: &str, timeout: Duration) -> Result<String, String> {
    use crate::util::http::build_download_client;

    let url = format!("https://github.com/{repo}/releases/latest");
    let client = build_download_client(timeout)
        .map_err(|_| "the release check's own client could not be built".to_string())?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("release check request failed: {}", request_reason(&e)))?;
    if !response.status().is_success() {
        return Err(format!("release check got HTTP {}", response.status()));
    }
    let tag = response
        .url()
        .path_segments()
        .and_then(|mut segments| segments.next_back().map(str::to_string))
        .unwrap_or_default();
    if tag.is_empty() {
        return Err("release redirect resolved to an empty tag".to_string());
    }
    Ok(tag)
}

// ── Path-free failure reasons ───────────────────────────────────────

/// A short, path-free reason for a failed request: the HTTP status when there
/// was one, else the kind of transport failure. The error's own display text
/// names the URL and is deliberately not used.
pub(crate) fn request_reason(error: &reqwest::Error) -> String {
    if let Some(status) = error.status() {
        return format!("HTTP {status}");
    }
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_request() {
        "request"
    } else if error.is_body() || error.is_decode() {
        "body"
    } else {
        "transport"
    };
    format!("{kind} failure")
}

/// A short, path-free reason for a failed release download: a rejected file, an
/// HTTP status or a transport/io kind from the error's own chain, else a sentence
/// that names no path. The chain's display text carries the URL and the local file
/// names.
pub(crate) fn download_reason(error: &anyhow::Error) -> String {
    for cause in error.chain() {
        if let Some(rejected) = cause.downcast_ref::<crate::util::http::ChecksumMismatch>() {
            return rejected.to_string();
        }
        if let Some(reqwest) = cause.downcast_ref::<reqwest::Error>() {
            return request_reason(reqwest);
        }
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            return format!("download failed ({})", io.kind());
        }
    }
    "the release download could not be completed".to_string()
}

// ── Release-format parsers ──────────────────────────────────────────

/// `(hash, filename)` from a `.sha256` sidecar body (`"<hash>  <filename>"`).
/// The hash must be 64 hex chars (normalized to lowercase to match the
/// computed digest) and a filename must be present — a bare-hash sidecar is
/// rejected so a cross-paired sidecar can never verify another asset.
#[must_use]
pub(crate) fn parse_sha256_sidecar(body: &str) -> Option<(String, String)> {
    let mut tokens = body.split_whitespace();
    let hash = tokens.next()?;
    let filename = tokens.next()?.to_string();
    let valid_hash = hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit());
    valid_hash.then(|| (hash.to_ascii_lowercase(), filename))
}

/// Parse a version token into a [`semver::Version`], stripping a leading `v` so
/// that both spellings the helper's `--version` banner can print parse.
#[must_use]
pub(crate) fn parse_version_token(s: &str) -> Option<semver::Version> {
    semver::Version::parse(s.strip_prefix('v').unwrap_or(s)).ok()
}

// ── In-place binary swap ────────────────────────────────────────────

/// Replace the binary at `dest` with `fresh`, never leaving a broken install.
/// Unix: copy to a `<dest>.mahbot_tmp` sibling, preserve the old file's
/// permissions (default 0o755 when `dest` is new), then rename (atomic, safe
/// over a running binary). Windows: a running exe cannot be overwritten or
/// deleted but CAN be renamed — rename-aside `dest` → `dest.old`, rename the
/// temp copy in, restore the aside on failure, and best-effort remove the
/// aside afterwards (its removal fails while the old binary is still running;
/// the next successful swap clears it).
///
/// Every failure message names the step and the io kind and never a path: they
/// are recorded in the product's own issues view.
fn swap_binary_in_place(fresh: &Path, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("mahbot_tmp");
    let _ = fs::remove_file(&tmp);
    fs::copy(fresh, &tmp).map_err(|e| {
        format!(
            "the staged copy for the swap could not be prepared ({})",
            e.kind()
        )
    })?;
    // Unix: keep the old binary's mode (0o755 default on first install) so the
    // swap never drops the executable bit.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(dest).map_or(0o755, |m| m.permissions().mode() & 0o777);
        fs::set_permissions(&tmp, fs::Permissions::from_mode(mode)).map_err(|e| {
            format!(
                "the staged copy's permissions could not be set ({})",
                e.kind()
            )
        })?;
    }

    if cfg!(target_os = "windows") && dest.exists() {
        rename_aside_swap(&tmp, dest)
    } else {
        fs::rename(&tmp, dest)
            .map_err(|e| format!("the new copy could not be moved into place ({})", e.kind()))
    }
}

/// The rename-aside swap used on Windows, where a running exe cannot be
/// overwritten or deleted but CAN be renamed: move `dest` aside, rename the
/// prepared `tmp` copy into place, and restore the aside if the final rename
/// fails. Aside removal is best-effort — while the old binary still runs it
/// cannot be removed; the next successful swap clears it. Platform-neutral
/// plain `fs::rename`/`remove_file`, so it is unit-testable on every OS.
///
/// Failure messages carry no path (the location may be one the owner's own
/// package manager keeps), only the failing step and its io kind.
fn rename_aside_swap(tmp: &Path, dest: &Path) -> Result<(), String> {
    let aside = dest.with_extension("old");
    let _ = fs::remove_file(&aside);
    fs::rename(dest, &aside)
        .map_err(|e| format!("the installed copy could not be moved aside ({})", e.kind()))?;
    if let Err(e) = fs::rename(tmp, dest) {
        // Last-resort restore: if this also fails the install is genuinely
        // broken, so surface that instead of discarding the error.
        if let Err(restore) = fs::rename(&aside, dest) {
            return Err(format!(
                "the new copy could not be moved into place ({}); the restore also failed ({}) — \
                 the managed binary is missing and must be reinstalled",
                e.kind(),
                restore.kind()
            ));
        }
        return Err(format!(
            "the new copy could not be moved into place ({})",
            e.kind()
        ));
    }
    let _ = fs::remove_file(&aside);
    Ok(())
}

/// Put the freshly extracted `fresh` binary at `dest`, creating the directory it
/// lives in first. The swap is atomic, so whatever sat there is replaced in one
/// step or not at all, and the executable bit is forced back on afterwards because
/// the swap carries the old file's mode over — an install that lost its bit would
/// otherwise survive a reinstall unchanged.
///
/// The caller names the destination once and keeps it, so a swap that does not land
/// still leaves it with the path it must report about.
///
/// The one failure of its own carries no path (the location is not the owner's
/// business and the message is recorded verbatim), only the operation and its io
/// kind.
pub(crate) fn place_extracted(fresh: &Path, dest: &Path) -> Result<(), String> {
    let Some(dir) = dest.parent() else {
        return Err(
            "the directory the product's own tool lives in could not be resolved".to_string(),
        );
    };
    fs::create_dir_all(dir).map_err(|e| {
        format!(
            "the directory the product's own tool lives in could not be created ({})",
            e.kind()
        )
    })?;
    swap_binary_in_place(fresh, dest)?;
    set_executable(dest)
}

// ── Start-time install policy ───────────────────────────────────────

/// How long an existing copy waits before it is refreshed, so the download never
/// competes with the boot.
const REFRESH_DELAY: Duration = Duration::from_mins(5);

/// Bring one of the product's own tools to the newest release: straight away when
/// no copy is installed, else once the boot has settled. `present` reports whether
/// the tool's own directory holds a copy — it decides only WHEN this runs, never
/// what is installed, because whatever sits there is always replaced — and after a
/// failure it says whether the host was left with a working copy at all.
///
/// Every failure is recorded at WARN — the level that survives the 8-hour INFO
/// retention and shows in the product's own issues view — because a tool that is
/// not at its newest release is exactly the state the product must not leave
/// unrecorded. Success stays INFO for a first install and DEBUG for a refresh, and
/// a failure never removes or invalidates the copy already installed — it is
/// retried on the next start.
///
/// The recorded reason is the failing step's own message: it names no local path
/// and no value read from the owner or his environment.
pub(crate) async fn install_on_start(
    tool: &str,
    present: impl Fn() -> bool,
    install: impl std::future::Future<Output = Result<PathBuf, String>>,
) {
    let missing = !present();
    if !missing {
        tokio::time::sleep(REFRESH_DELAY).await;
    }
    match install.await {
        Ok(path) if missing => tracing::info!("{tool} installed at {}", path.display()),
        Ok(path) => tracing::debug!("{tool} brought up to date at {}", path.display()),
        Err(reason) => {
            let reason = crate::util::truncate(&reason, 1024);
            // The step that failed says which part did not finish; the state says
            // what the host is left with, and never more than that: a failed step
            // can leave the copy placed but not registered, and it can fail before
            // anything is placed.
            let state = match (present(), missing) {
                (true, true) => "was installed but not completely",
                (true, false) => "could not be brought fully up to date",
                (false, true) => "could not be installed",
                (false, false) => "left the host without a working copy",
            };
            tracing::warn!("{tool} {state} (retried on the next start): {reason}");
        }
    }
}

// ── Archive extraction ──────────────────────────────────────────────

/// Extract the single `file_name` binary from a tar.gz release archive into
/// `dir` and return its path. The entry is matched by file name at any depth
/// but always written to `<dir>/<file_name>`, so a nested vendor layout still
/// lands correctly; `Err` when the archive has no such regular-file entry.
///
/// Failure messages name the operation and the io kind, never a path.
pub(crate) fn extract_single_file_tar_gz(
    archive: &Path,
    dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("the release archive could not be opened ({})", e.kind()))?;
    let mut tar_archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let out_path = dir.join(file_name);
    let mut unpacked = false;
    let entries = tar_archive
        .entries()
        .map_err(|e| format!("the release archive could not be read ({})", e.kind()))?;
    for entry in entries {
        let mut entry = entry
            .map_err(|e| format!("a release archive entry could not be read ({})", e.kind()))?;
        let path = entry
            .path()
            .map_err(|e| {
                format!(
                    "a release archive entry path could not be read ({})",
                    e.kind()
                )
            })?
            .into_owned();
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if path.file_name().is_some_and(|n| n == OsStr::new(file_name)) {
            // `unpack` writes exactly to `out_path` (the entry's own internal
            // path is ignored), keeping the returned path correct for any
            // archive layout.
            entry.unpack(&out_path).map_err(|e| {
                format!(
                    "{file_name} could not be extracted from the release archive ({})",
                    e.kind()
                )
            })?;
            unpacked = true;
            break;
        }
    }
    if !unpacked {
        return Err(format!("archive contains no {file_name} binary"));
    }

    // The freshly extracted binary must be executable.
    set_executable(&out_path)?;
    Ok(out_path)
}

/// Extract the single `file_name` binary from a zip release archive into
/// `dir` and return its path. The entry is matched by file name at any depth
/// (zip archives use `/` separators internally) but always written to the
/// fixed `<dir>/<file_name>` — never the entry's internal path — so a nested
/// `bun-<target>/` vendor layout still lands correctly; `Err` when the
/// archive has no such regular-file entry.
///
/// Failure messages name the operation and a reason the archive library's own
/// error yields — the io kind it wraps, or its own short sentence for a malformed
/// archive or a missing entry — never a path.
pub(crate) fn extract_single_file_zip(
    archive: &Path,
    dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("the release archive could not be opened ({})", e.kind()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("the release archive could not be read ({})", zip_reason(&e)))?;
    let out_path = dir.join(file_name);
    let mut unpacked = false;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| {
            format!(
                "a release archive entry could not be read ({})",
                zip_reason(&e)
            )
        })?;
        if !entry.is_file() {
            continue;
        }
        if Path::new(entry.name())
            .file_name()
            .is_some_and(|n| n == OsStr::new(file_name))
        {
            let mut out_file = fs::File::create(&out_path)
                .map_err(|e| format!("{file_name} could not be written ({})", e.kind()))?;
            std::io::copy(&mut entry, &mut out_file).map_err(|e| {
                format!(
                    "{file_name} could not be extracted from the release archive ({})",
                    e.kind()
                )
            })?;
            unpacked = true;
            break;
        }
    }
    if !unpacked {
        return Err(format!("archive contains no {file_name} binary"));
    }

    // The freshly extracted binary must be executable.
    set_executable(&out_path)?;
    Ok(out_path)
}

/// A path-free reason for a failed zip operation: the io kind when the archive
/// error wraps one, else the archive library's own short sentence (a malformed or
/// unsupported archive, a missing entry, a compression method it cannot read). The
/// archive's own path and the entries' names are never on it.
#[must_use]
fn zip_reason(error: &zip::result::ZipError) -> String {
    match error {
        zip::result::ZipError::Io(error) => error.kind().to_string(),
        other => other.to_string(),
    }
}

/// Make a freshly extracted/installed binary executable by resetting its full
/// mode to 0o755. The failure message names the io kind only — never the path.
#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).map_err(|e| {
        format!(
            "the executable bit could not be set on the installed copy ({})",
            e.kind()
        )
    })
}

// Callers use `?` uniformly, so the signature mirrors the Unix implementation.
#[cfg(not(unix))]
#[expect(clippy::unnecessary_wraps)]
fn set_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

// ── Host detection ──────────────────────────────────────────────────

/// The compile-time `(os, arch)` pair used to select a release asset, or
/// `Err` naming the unsupported platform/arch. Uses the compile-time target
/// triple (`cfg!`) so it is correct regardless of the runtime host.
pub(crate) fn host_os_arch() -> Result<(&'static str, &'static str), String> {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        return Err(format!("unsupported platform: {}", std::env::consts::OS));
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        return Err(format!("unsupported arch: {}", std::env::consts::ARCH));
    };
    Ok((os, arch))
}

/// Detect a musl libc host by the presence of a musl loader in `/lib`
/// (mirrors the vendor installer — no `ldd`).
#[must_use]
pub(crate) fn linux_host_is_musl() -> bool {
    Path::new("/lib/ld-musl-x86_64.so.1").exists()
        || Path::new("/lib/ld-musl-aarch64.so.1").exists()
}

// ── Install directories ─────────────────────────────────────────────

/// bun-standard user install dir (`~/.bun/bin` on unix, `%USERPROFILE%\.bun\bin`
/// on Windows). Resolved via [`directories::UserDirs`] like the rest of the
/// codebase, not a literal `$HOME` splice.
#[must_use]
pub(crate) fn bun_bin_dir() -> Option<PathBuf> {
    directories::UserDirs::new().map(|d| d.home_dir().join(".bun").join("bin"))
}

/// The standard per-user programs directory the browser helper's own installer
/// puts it in when the system-wide directory cannot be written: `~/.local/bin`
/// on unix, and `%LOCALAPPDATA%\Programs\chrome-use` on Windows, where there is
/// no system-wide step and no elevation.
///
/// Both are the vendor's own defaults, quoted from its installers at the `v1.5.140`
/// release (both files sit at its repository root on every tag): `install.sh:100`
/// takes `/usr/local/bin` when `[ -w /usr/local/bin ]` and `$HOME/.local/bin`
/// otherwise, and `install.ps1:233` is `if (-not $binDir) { $binDir = Join-Path
/// $env:LOCALAPPDATA 'Programs\chrome-use' }` — the `AGENT_BROWSER_BIN_DIR`
/// argument it takes overriding that.
#[must_use]
pub(crate) fn chrome_use_user_bin_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        directories::UserDirs::new().map(|d| d.home_dir().join(".local").join("bin"))
    }
    #[cfg(not(unix))]
    {
        directories::BaseDirs::new().map(|d| d.data_local_dir().join("Programs").join("chrome-use"))
    }
}

/// The full path of the product's own copy of the browser helper (`file_name`): the
/// directory its own installer uses on this host, and that file inside it.
///
/// On unix that directory is the system-wide `/usr/local/bin` when it already holds a
/// copy — the product's or the owner's — or when it can be written, otherwise the
/// standard per-user programs directory [`chrome_use_user_bin_dir`]; on Windows it is
/// that per-user directory, which is where the helper's own installer puts it.
///
/// Installs and resolution both ask this one function, and it resolves from the state
/// of the moment with nothing persisted: a copy in an unwritable system directory is
/// still the copy the agents run, and a system directory that becomes writable later
/// becomes the location, leaving an unmaintained per-user copy beside it that the
/// product neither uses nor removes. Which of the two the owner's own terminal
/// resolves first is his search path's own order, which the product neither knows nor
/// changes.
#[cfg(unix)]
#[must_use]
pub(crate) fn chrome_use_bin_path(file_name: &str) -> Option<PathBuf> {
    let system = Path::new(SYSTEM_BIN_DIR);
    let dir = chrome_use_dir(
        crate::util::is_executable(&system.join(file_name)),
        dir_is_writable(system),
        chrome_use_user_bin_dir(),
    )?;
    Some(dir.join(file_name))
}

/// The same on Windows, where the helper's own installer puts it in a per-user
/// directory that does not depend on the file's name — see the unix definition
/// above for the rule and its reasoning.
#[cfg(not(unix))]
#[must_use]
pub(crate) fn chrome_use_bin_path(file_name: &str) -> Option<PathBuf> {
    Some(chrome_use_user_bin_dir()?.join(file_name))
}

/// The helper's directory on unix, from the copy probe, the writability of the
/// system-wide directory and the per-user one — pure, so the rule is pinned by a
/// test rather than by a host's permissions ([`chrome_use_bin_path`] holds the
/// reasoning). The system-wide directory is decided before the per-user one is
/// ever needed, so a host with no resolvable home — a container, say — still
/// resolves and installs the helper there.
#[cfg(unix)]
#[must_use]
fn chrome_use_dir(
    system_copy: bool,
    system_writable: bool,
    user: Option<PathBuf>,
) -> Option<PathBuf> {
    (system_copy || system_writable)
        .then(|| PathBuf::from(SYSTEM_BIN_DIR))
        .or(user)
}

/// Whether the helper installer's own test (`[ -w dir ]`) passes for `dir`, so a
/// root-owned `/usr/local/bin` is not written. `libc::access` rather than a look
/// at the mode bits: it answers through the group/other-write bit and through a
/// read-only mount, exactly as the installer's own test does.
#[cfg(unix)]
#[must_use]
fn dir_is_writable(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt as _;

    let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `path` is a valid NUL-terminated C string that outlives the call
    // and `access` only reads it.
    unsafe { libc::access(path.as_ptr(), libc::W_OK) == 0 }
}

/// The system-wide directory the helper's own installer uses when it can be
/// written.
#[cfg(unix)]
const SYSTEM_BIN_DIR: &str = "/usr/local/bin";

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[cfg(unix)]
    #[test]
    fn the_helpers_directory_is_the_system_one_whenever_it_holds_a_copy() {
        // A copy in the system directory keeps resolution on that directory, so the
        // agents run that copy — writable or not. Which of the two directories the
        // owner's own terminal resolves first is his search path's own order.
        assert_eq!(
            chrome_use_dir(true, false, None),
            Some(PathBuf::from(SYSTEM_BIN_DIR))
        );
        // No copy there yet: writability alone decides. It is decided before the
        // per-user directory is ever needed, so a container with no resolvable home
        // still resolves and installs there.
        assert_eq!(
            chrome_use_dir(false, true, None),
            Some(PathBuf::from(SYSTEM_BIN_DIR))
        );
        // Neither usable nor a home to fall back to: no directory at all.
        assert_eq!(chrome_use_dir(false, false, None), None);
        let user = PathBuf::from("/home/o/.local/bin");
        assert_eq!(
            chrome_use_dir(false, false, Some(user.clone())),
            Some(user),
            "the per-user directory is the fallback, never the preference"
        );
    }

    #[test]
    fn version_banner_tokens_parse_with_or_without_a_v_prefix() {
        // The helper's `--version` banner prints the version with or without a
        // leading `v`.
        assert_eq!(
            parse_version_token("v1.5.100"),
            Some(semver::Version::new(1, 5, 100))
        );
        assert_eq!(
            parse_version_token("1.5.100"),
            Some(semver::Version::new(1, 5, 100))
        );
        assert_eq!(parse_version_token("latest"), None);
        assert_eq!(parse_version_token(""), None);
    }

    #[test]
    fn sha256_sidecar_parses_hash_and_filename() {
        let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        // Real two-space sidecar format.
        assert_eq!(
            parse_sha256_sidecar(&format!("{hash}  bun-linux-x64.zip")),
            Some((hash.to_string(), "bun-linux-x64.zip".to_string()))
        );
        // Leading whitespace and trailing newline are tolerated; an uppercase
        // hash is normalized to lowercase (the computed digest is lowercase).
        assert_eq!(
            parse_sha256_sidecar(&format!("  {hash}  bun-darwin-arm64.zip\n")),
            Some((hash.to_string(), "bun-darwin-arm64.zip".to_string()))
        );
        assert_eq!(
            parse_sha256_sidecar(&format!("{}  x.tar.gz", hash.to_uppercase())),
            Some((hash.to_string(), "x.tar.gz".to_string()))
        );
        // A bare-hash sidecar (no filename) is rejected — the filename is what
        // guards against cross-paired sidecars.
        assert_eq!(parse_sha256_sidecar(hash), None);
        // Rejects short, non-hex, and empty hashes.
        assert_eq!(parse_sha256_sidecar("abcd  x.tar.gz"), None);
        assert_eq!(
            parse_sha256_sidecar(&format!("{}  x.tar.gz", "g".repeat(64))),
            None
        );
        assert_eq!(parse_sha256_sidecar(""), None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn swap_binary_in_place_preserves_permissions_and_leaves_no_temp() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let dest = dir.path().join("bun");
        let fresh = dir.path().join("fresh");

        fs::write(&dest, "old").expect("write old dest");
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o755)).expect("chmod dest");
        fs::write(&fresh, "new").expect("write fresh");
        fs::set_permissions(&fresh, fs::Permissions::from_mode(0o755)).expect("chmod fresh");

        swap_binary_in_place(&fresh, &dest).expect("swap");
        assert_eq!(fs::read(&dest).expect("read dest"), b"new".as_slice());
        assert_eq!(
            fs::metadata(&dest).expect("stat dest").permissions().mode() & 0o777,
            0o755
        );
        assert!(
            !dest.with_extension("mahbot_tmp").exists(),
            "no temp sibling left"
        );

        // Second swap over an existing install keeps the running binary's mode.
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o700)).expect("chmod dest");
        fs::write(&fresh, "newer").expect("write fresh 2");
        swap_binary_in_place(&fresh, &dest).expect("swap 2");
        assert_eq!(fs::read(&dest).expect("read dest"), b"newer".as_slice());
        assert_eq!(
            fs::metadata(&dest).expect("stat dest").permissions().mode() & 0o777,
            0o700
        );
        assert!(
            !dest.with_extension("mahbot_tmp").exists(),
            "no temp sibling left"
        );
    }

    #[test]
    fn rename_aside_swap_replaces_and_cleans_up() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("bun");
        let tmp = dir.path().join("prepared");
        fs::write(&dest, "old").expect("write dest");
        fs::write(&tmp, "new").expect("write tmp");

        rename_aside_swap(&tmp, &dest).expect("swap");
        assert_eq!(fs::read(&dest).expect("read dest"), b"new");
        // The aside and the prepared copy are both gone after a clean swap.
        assert!(!dest.with_extension("old").exists());
        assert!(!tmp.exists());
    }

    #[test]
    fn rename_aside_swap_restores_dest_when_the_final_rename_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("bun");
        // The prepared copy does not exist, so the final rename fails.
        let tmp = dir.path().join("missing");
        fs::write(&dest, "old").expect("write dest");

        assert!(rename_aside_swap(&tmp, &dest).is_err());
        // The previous binary was restored at its original location.
        assert_eq!(fs::read(&dest).expect("read dest"), b"old");
        assert!(!dest.with_extension("old").exists(), "aside was moved back");
    }

    /// Build a tar.gz archive in `dir` with the given (path, contents) entries.
    fn write_test_archive(dir: &std::path::Path, entries: &[(&str, &[u8])]) -> fs::File {
        let enc = flate2::write::GzEncoder::new(
            fs::File::create(dir.join("pkg.tar.gz")).expect("create archive"),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(enc);
        for (path, contents) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            builder
                .append_data(&mut header, path, *contents)
                .expect("append entry");
        }
        builder
            .into_inner()
            .expect("finish archive")
            .finish()
            .expect("finish gzip")
    }

    #[test]
    fn extract_single_file_tar_gz_lands_at_the_dir_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_archive(dir.path(), &[("bun", b"BIN" as &[u8])]);

        let out = extract_single_file_tar_gz(&dir.path().join("pkg.tar.gz"), dir.path(), "bun")
            .expect("extract");
        assert_eq!(out, dir.path().join("bun"));
        assert_eq!(fs::read(&out).expect("read extracted"), b"BIN");
        // The extracted binary must be executable (the exec-bit chmod is
        // unix-only; on Windows executability is the .exe extension).
        #[cfg(unix)]
        {
            assert_ne!(
                fs::metadata(&out)
                    .expect("stat extracted")
                    .permissions()
                    .mode()
                    & 0o111,
                0
            );
        }
    }

    #[test]
    fn extract_single_file_tar_gz_handles_a_nested_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entry = "pkg/bin/bun";
        write_test_archive(dir.path(), &[(entry, b"NESTED" as &[u8])]);

        let out = extract_single_file_tar_gz(&dir.path().join("pkg.tar.gz"), dir.path(), "bun")
            .expect("extract");
        // Matched by file name at any depth, but always written to the dir root.
        assert_eq!(out, dir.path().join("bun"));
        assert_eq!(fs::read(&out).expect("read extracted"), b"NESTED");
    }

    #[test]
    fn extract_single_file_tar_gz_errors_without_the_binary() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_archive(dir.path(), &[("readme.txt", b"no bin" as &[u8])]);

        assert!(
            extract_single_file_tar_gz(&dir.path().join("pkg.tar.gz"), dir.path(), "bun").is_err()
        );
    }

    /// Build a zip archive in `dir` with the given (path, contents) entries.
    fn write_test_zip(dir: &std::path::Path, entries: &[(&str, &[u8])]) -> fs::File {
        let file = fs::File::create(dir.join("pkg.zip")).expect("create archive");
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (path, contents) in entries {
            writer.start_file(path, options).expect("start entry");
            writer.write_all(contents).expect("write entry");
        }
        writer.finish().expect("finish zip")
    }

    #[test]
    fn extract_single_file_zip_lands_at_the_dir_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_zip(dir.path(), &[("bun", b"BIN" as &[u8])]);

        let out = extract_single_file_zip(&dir.path().join("pkg.zip"), dir.path(), "bun")
            .expect("extract");
        assert_eq!(out, dir.path().join("bun"));
        assert_eq!(fs::read(&out).expect("read extracted"), b"BIN");
        #[cfg(unix)]
        {
            assert_ne!(
                fs::metadata(&out)
                    .expect("stat extracted")
                    .permissions()
                    .mode()
                    & 0o111,
                0
            );
        }
    }

    #[test]
    fn extract_single_file_zip_handles_a_nested_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entry = "bun-darwin-x64/bun";
        write_test_zip(dir.path(), &[(entry, b"NESTED" as &[u8])]);

        let out = extract_single_file_zip(&dir.path().join("pkg.zip"), dir.path(), "bun")
            .expect("extract");
        // Matched by file name at any depth, but always written to the dir root.
        assert_eq!(out, dir.path().join("bun"));
        assert_eq!(fs::read(&out).expect("read extracted"), b"NESTED");
    }

    #[test]
    fn extract_single_file_zip_errors_without_the_binary() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_zip(dir.path(), &[("readme.txt", b"no bin" as &[u8])]);

        assert!(extract_single_file_zip(&dir.path().join("pkg.zip"), dir.path(), "bun").is_err());
    }
}
