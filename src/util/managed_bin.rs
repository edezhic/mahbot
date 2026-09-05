//! Generic managed-binary release mechanics shared by the rust-installed tool
//! managers (chrome-use via `browser_daemon`, the managed bun runtime).
//!
//! These binaries are all installed into a *managed* directory that mahbot
//! owns (never a user PATH-looked-up install), updated in place via an atomic
//! swap, and verified against a SHA-256 checksum published with the release.
//! The helpers are release-format agnostic: the chrome-use installer reads a
//! `<asset>.sha256` sidecar and a tar.gz archive, the bun installer reads a
//! combined `SHASUMS256.txt` file and a zip archive — both share the same
//! tag/version/swap/extract primitives.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::{debug, info, warn};

// ── Release metadata ────────────────────────────────────────────────

/// Follow the GitHub `releases/latest` redirect and return the last path
/// segment (the release tag) — avoids the api.github.com rate limit. reqwest
/// follows the redirect by default; the final response URL is the tag page.
pub(crate) async fn fetch_latest_tag(repo: &str, timeout: Duration) -> Result<String, String> {
    use crate::util::http::build_download_client;

    let url = format!("https://github.com/{repo}/releases/latest");
    let client =
        build_download_client(timeout).map_err(|e| format!("release check client failed: {e}"))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("release check request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "release check got HTTP {} from {url}",
            response.status()
        ));
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

/// Parse a release tag or CLI `--version` output into a [`semver::Version`],
/// stripping `prefix` when present. Release tags carry a prefix (`v` for
/// chrome-use, `bun-v` for bun), while CLI `--version` output is bare — both
/// flow through here, so the strip is a no-op for bare output.
#[must_use]
pub(crate) fn parse_tag_version(s: &str, prefix: &str) -> Option<semver::Version> {
    semver::Version::parse(s.strip_prefix(prefix).unwrap_or(s)).ok()
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
pub(crate) fn swap_binary_in_place(fresh: &Path, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("mahbot_tmp");
    let _ = fs::remove_file(&tmp);
    fs::copy(fresh, &tmp).map_err(|e| {
        format!(
            "failed to copy {} to {}: {e}",
            fresh.display(),
            tmp.display()
        )
    })?;
    // Unix: keep the old binary's mode (0o755 default on first install) so the
    // swap never drops the executable bit.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(dest).map_or(0o755, |m| m.permissions().mode() & 0o777);
        fs::set_permissions(&tmp, fs::Permissions::from_mode(mode))
            .map_err(|e| format!("failed to set permissions on {}: {e}", tmp.display()))?;
    }

    if cfg!(target_os = "windows") && dest.exists() {
        rename_aside_swap(&tmp, dest)
    } else {
        fs::rename(&tmp, dest).map_err(|e| {
            format!(
                "failed to rename {} to {}: {e}",
                tmp.display(),
                dest.display()
            )
        })
    }
}

/// The rename-aside swap used on Windows, where a running exe cannot be
/// overwritten or deleted but CAN be renamed: move `dest` aside, rename the
/// prepared `tmp` copy into place, and restore the aside if the final rename
/// fails. Aside removal is best-effort — while the old binary still runs it
/// cannot be removed; the next successful swap clears it. Platform-neutral
/// plain `fs::rename`/`remove_file`, so it is unit-testable on every OS.
fn rename_aside_swap(tmp: &Path, dest: &Path) -> Result<(), String> {
    let aside = dest.with_extension("old");
    let _ = fs::remove_file(&aside);
    fs::rename(dest, &aside)
        .map_err(|e| format!("failed to move {} aside: {e}", dest.display()))?;
    if let Err(e) = fs::rename(tmp, dest) {
        // Last-resort restore: if this also fails the install is genuinely
        // broken, so surface that instead of discarding the error.
        if let Err(restore) = fs::rename(&aside, dest) {
            return Err(format!(
                "failed to rename {} to {}: {e}; the restore also failed ({restore}) — \
                 {} is missing and the managed binary must be reinstalled",
                tmp.display(),
                dest.display(),
                dest.display()
            ));
        }
        return Err(format!(
            "failed to rename {} to {}: {e}",
            tmp.display(),
            dest.display()
        ));
    }
    let _ = fs::remove_file(&aside);
    Ok(())
}

// ── Archive extraction ──────────────────────────────────────────────

/// Extract the single `file_name` binary from a tar.gz release archive into
/// `dir` and return its path. The entry is matched by file name at any depth
/// but always written to `<dir>/<file_name>`, so a nested vendor layout still
/// lands correctly; `Err` when the archive has no such regular-file entry.
pub(crate) fn extract_single_file_tar_gz(
    archive: &Path,
    dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("failed to open archive {}: {e}", archive.display()))?;
    let mut tar_archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let out_path = dir.join(file_name);
    let mut unpacked = false;
    let entries = tar_archive
        .entries()
        .map_err(|e| format!("failed to read archive {}: {e}", archive.display()))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("failed to read archive entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("failed to read archive entry path: {e}"))?
            .into_owned();
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if path.file_name().is_some_and(|n| n == OsStr::new(file_name)) {
            // `unpack` writes exactly to `out_path` (the entry's own internal
            // path is ignored), keeping the returned path correct for any
            // archive layout.
            entry
                .unpack(&out_path)
                .map_err(|e| format!("failed to extract {}: {e}", out_path.display()))?;
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
pub(crate) fn extract_single_file_zip(
    archive: &Path,
    dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("failed to open archive {}: {e}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("failed to read archive {}: {e}", archive.display()))?;
    let out_path = dir.join(file_name);
    let mut unpacked = false;
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| format!("failed to read archive entry: {e}"))?;
        if !entry.is_file() {
            continue;
        }
        if Path::new(entry.name())
            .file_name()
            .is_some_and(|n| n == OsStr::new(file_name))
        {
            let mut out_file = fs::File::create(&out_path)
                .map_err(|e| format!("failed to create {}: {e}", out_path.display()))?;
            std::io::copy(&mut entry, &mut out_file)
                .map_err(|e| format!("failed to extract {}: {e}", out_path.display()))?;
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

/// Force the executable bit on a freshly extracted/installed binary.
#[cfg(unix)]
pub(crate) fn set_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("failed to set executable bit on {}: {e}", path.display()))
}

#[cfg(not(unix))]
pub(crate) fn set_executable(_path: &Path) -> Result<(), String> {
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

// ── User-shell visibility ───────────────────────────────────────────

/// bun-standard user install dir (`~/.bun/bin` on unix, `%USERPROFILE%\.bun\bin`
/// on Windows). Resolved via [`directories::UserDirs`] like the rest of the
/// codebase, not a literal `$HOME` splice.
#[must_use]
pub(crate) fn bun_bin_dir() -> Option<PathBuf> {
    directories::UserDirs::new().map(|d| d.home_dir().join(".bun").join("bin"))
}

/// Stable mahbot-owned managed-bin install dir (`<storage root>/bin`). The
/// chrome-use binary is placed here and probed FIRST by its resolver on every
/// OS — on Windows it is the only probe that reliably finds mahbot's own
/// install — so the first-install location and the auto-update swap location
/// are always the same path.
#[must_use]
pub(crate) fn storage_bin_dir() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .map(|r| r.join("bin"))
}

/// All managed-bin directories that user shells should resolve by bare name:
/// the mahbot-owned storage dir first, then the bun-standard user dir.
#[cfg(unix)]
fn managed_shell_dirs() -> Vec<PathBuf> {
    [storage_bin_dir(), bun_bin_dir()]
        .into_iter()
        .flatten()
        .collect()
}

/// Ensure the user's interactive shells resolve the managed binaries by bare
/// name. Appends a guarded PATH block (idempotent) to `~/.zshrc`/`~/.bashrc`;
/// a new rc file is only created when it matches the user's login shell — or
/// when the login shell cannot be determined (services run without `$SHELL`),
/// in which case both common rc files are created. Never fails the install.
pub(crate) fn ensure_rc_path_block() {
    #[cfg(unix)]
    ensure_unix_rc_path_block();
}

/// The unix body of [`ensure_rc_path_block`]. No-op on Windows (no rc files).
#[cfg(unix)]
fn ensure_unix_rc_path_block() {
    let dirs = managed_shell_dirs();
    if dirs.is_empty() {
        debug!("managed-bin PATH block skipped: no managed bin dirs resolved");
        return;
    }
    let path_entry = dirs
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    // The managed dirs are APPENDED after `$PATH` — a user's own install
    // (Homebrew, npm, official installer) keeps precedence, matching the
    // agent-shell PATH ordering. The dirs are always listed even when a binary
    // currently resolves elsewhere (e.g. a PATH chrome-use), so a later
    // managed install is immediately visible.
    let block = format!(
        "\n# >>> mahbot managed binaries >>>\nexport PATH=\"$PATH:{path_entry}\"\n# <<< mahbot managed binaries <<<\n"
    );
    let Some(home) = directories::UserDirs::new().map(|d| d.home_dir().to_path_buf()) else {
        debug!("managed-bin PATH block skipped: user home not resolvable");
        return;
    };
    let shell = login_shell_name();
    // When the login shell cannot be determined (launchd/systemd services run
    // without `$SHELL`), create both common rc files — a user-local file
    // holding just the guarded block is harmless, and skipping creation would
    // leave fresh installs invisible to every interactive shell.
    for (name, create_allowed) in [
        (".zshrc", shell.as_deref().is_none_or(|s| s == "zsh")),
        (".bashrc", shell.as_deref().is_none_or(|s| s == "bash")),
    ] {
        let path = home.join(name);
        match append_rc_block(&path, &block, create_allowed) {
            Ok(true) => {
                info!(
                    "added mahbot managed-binaries PATH block to {}",
                    path.display()
                );
            }
            Ok(false) => {
                debug!(
                    "mahbot managed-binaries PATH block already present in {}",
                    path.display()
                );
            }
            Err(e) => {
                warn!(
                    "failed to ensure managed-binaries PATH block in {}: {e}",
                    path.display()
                );
            }
        }
    }
}

/// Append `block` to the rc file at `path` unless it already carries the
/// managed-binaries START marker. Returns `Ok(true)` when written, `Ok(false)`
/// when already present. When the file is missing, creates it only if
/// `create_allowed`. Never writes when the read failed for a non-missing
/// reason.
#[cfg(unix)]
fn append_rc_block(path: &Path, block: &str, create_allowed: bool) -> Result<bool, String> {
    use std::fs::OpenOptions;
    use std::io::Write;

    match fs::read_to_string(path) {
        Ok(content) => {
            if content.contains("# >>> mahbot managed binaries >>>") {
                Ok(false)
            } else {
                let mut f = OpenOptions::new()
                    .append(true)
                    .open(path)
                    .map_err(|e| format!("failed to append to {}: {e}", path.display()))?;
                f.write_all(block.as_bytes())
                    .map_err(|e| format!("failed to append to {}: {e}", path.display()))?;
                Ok(true)
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if create_allowed {
                let mut f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|e| format!("failed to create {}: {e}", path.display()))?;
                f.write_all(block.as_bytes())
                    .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Err(e) => Err(format!("failed to read {}: {e}", path.display())),
    }
}

/// Basename of `$SHELL` (the user's login shell), or `None` when unset/empty —
/// decides which rc file may be created; `None` allows both (see
/// [`ensure_unix_rc_path_block`]).
#[cfg(unix)]
#[must_use]
fn login_shell_name() -> Option<String> {
    std::env::var("SHELL")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| {
            Path::new(&s)
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn release_version_parsing_strips_prefix() {
        // GitHub release tags are v-prefixed; `--version` output is bare.
        assert_eq!(
            parse_tag_version("v1.5.100", "v"),
            Some(semver::Version::new(1, 5, 100))
        );
        assert_eq!(
            parse_tag_version("1.5.100", "v"),
            Some(semver::Version::new(1, 5, 100))
        );
        // Bun release tags are `bun-v`-prefixed (NOT just `v`).
        assert_eq!(
            parse_tag_version("bun-v1.2.3", "bun-v"),
            Some(semver::Version::new(1, 2, 3))
        );
        assert_eq!(parse_tag_version("latest", "v"), None);
        assert_eq!(parse_tag_version("", "v"), None);
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

    #[cfg(unix)]
    #[test]
    fn append_rc_block_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".zshrc");
        let block = "\n# >>> mahbot managed binaries >>>\nexport PATH=\"/x:$PATH\"\n# <<< mahbot managed binaries <<<\n";

        // First append creates/writes.
        assert!(append_rc_block(&path, block, true).expect("first append"));
        let content = fs::read_to_string(&path).expect("read back");
        assert_eq!(
            content.matches("# >>> mahbot managed binaries >>>").count(),
            1
        );
        assert!(content.contains("export PATH=\"/x:$PATH\""));

        // Second append is a no-op (idempotent) and does not duplicate the block.
        assert!(!append_rc_block(&path, block, true).expect("second append"));
        let content = fs::read_to_string(&path).expect("read back");
        assert_eq!(
            content.matches("# >>> mahbot managed binaries >>>").count(),
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn append_rc_block_creates_when_allowed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".bashrc");
        let block = "\n# >>> mahbot managed binaries >>>\nexport PATH=\"/y:$PATH\"\n# <<< mahbot managed binaries <<<\n";

        assert!(append_rc_block(&path, block, true).expect("create"));
        assert_eq!(
            fs::read_to_string(&path).expect("read back"),
            block.to_string()
        );
    }

    #[cfg(unix)]
    #[test]
    fn append_rc_block_skips_creation_when_not_allowed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".bashrc");
        let block = "\n# >>> mahbot managed binaries >>>\nexport PATH=\"/y:$PATH\"\n# <<< mahbot managed binaries <<<\n";

        assert!(!append_rc_block(&path, block, false).expect("no create"));
        assert!(!path.exists(), "no file written");
    }
}
