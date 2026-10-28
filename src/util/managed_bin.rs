//! Generic managed-binary release mechanics shared by every installer that fetches
//! a prebuilt artifact: the rust-installed tool managers (chrome-use via
//! `chrome_daemon`, the managed bun runtime), the self-hosted Telegram Bot API
//! server (`channels::telegram_local`) and the product's own ffmpeg pair
//! (`crate::ffmpeg`).
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
//! tag/version/swap/extract primitives. The two streams that publish prereleases
//! (`crate::ffmpeg` and the Bot API server) are read through the stream's own
//! listing instead of `releases/latest`, on the one client [`download_client`] owns
//! ([`release_page`] is one page of it).
//!
//! [`install_on_start`] is the shared start-time policy: a missing (or
//! non-executable) copy is installed at once, while a copy in place is left
//! untouched until [`REFRESH_DELAY`] has passed and is compared with the newest
//! release only then — the release is downloaded only when the two differ.
//!
//! The product's own install location lives here too — [`mahbot_install_dir`], built
//! on [`user_programs_dir`] — as the one place inside the product that names it: the
//! search-path edit, the shell path list and the update all read it. The install
//! scripts place the ready-made file at the same location, each spelling it with its
//! own platform's variables.

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

/// The `owner/repo` a managed tool's release stream lives in: the product's own
/// repository, named once by [`crate::self_update::RELEASE_REPO`], so no installer
/// can drift onto another host.
pub(crate) fn release_repo() -> &'static str {
    crate::self_update::RELEASE_REPO
        .trim_start_matches("https://github.com/")
        .trim_end_matches('/')
}

/// The `User-Agent` every GitHub API request needs: it answers `403` without one.
const GITHUB_USER_AGENT: &str = concat!("mahbot/", env!("CARGO_PKG_VERSION"));

/// How many releases one page of a listing holds. The streams the product manages
/// share the repository's listing with the product's own releases and each other's,
/// so this is what a caller counts a page's entries against to know it is the last.
pub(crate) const LISTING_PAGE_SIZE: usize = 100;

/// How many pages are read before a stream is reported missing: the streams share one
/// listing, so a stream's newest build is not on the first page once a page worth of
/// newer releases exists.
pub(crate) const LISTING_PAGES: u32 = 10;

/// How long a release's own small files may take — a page of the listing, a checksum
/// list — as opposed to the artifact itself, which runs on the download client's bound.
pub(crate) const METADATA_TIMEOUT: Duration = Duration::from_secs(20);

/// The total bound of one artifact download: a managed binary is tens of megabytes.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(20);

/// The client every managed-binary request runs on: the shared download client, whose
/// total bound and connect timeout are the policy an artifact of this size needs. One
/// client per process — it owns the connection pool.
pub(crate) fn download_client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: std::sync::OnceLock<Result<reqwest::Client, String>> =
        std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            crate::util::http::build_download_client(DOWNLOAD_TIMEOUT)
                .map_err(|e| format!("the release client could not be built: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// One page of a GitHub release listing — the only place a prerelease is named, which
/// is why a stream made of prereleases is read here instead of through
/// `releases/latest`. `url` is the repository's `releases` endpoint or a stand-in
/// answering in the same shape; the page query is appended here. `subject` names what is
/// being looked up, because the failure is recorded and shown verbatim.
///
/// `None` is a body that parses as JSON but is not a release array — the API's own error
/// message, a proxy's answer — which each stream reads the way its own contract needs: a
/// host offering no release, or a listing that could not be read.
pub(crate) async fn release_page(
    url: &str,
    page: u32,
    subject: &str,
) -> Result<Option<Vec<serde_json::Value>>, String> {
    let separator = if url.contains('?') { '&' } else { '?' };
    let url = format!("{url}{separator}per_page={LISTING_PAGE_SIZE}&page={page}");
    let response = download_client()?
        .get(&url)
        .header(reqwest::header::USER_AGENT, GITHUB_USER_AGENT)
        .timeout(METADATA_TIMEOUT)
        .send()
        .await
        .map_err(|e| format!("{subject} could not be fetched: {}", request_reason(&e)))?;
    if !response.status().is_success() {
        return Err(format!(
            "{subject} could not be fetched: HTTP {}",
            response.status()
        ));
    }
    let listing: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("{subject} could not be read: {}", request_reason(&e)))?;
    Ok(match listing {
        serde_json::Value::Array(releases) => Some(releases),
        _ => None,
    })
}

/// A release's `SHA256SUMS` body — the file every digest is read out of. `subject` names it
/// in the failure, which is recorded and shown verbatim.
pub(crate) async fn fetch_sha256_sums(url: &str, subject: &str) -> Result<String, String> {
    let response = download_client()?
        .get(url)
        .timeout(METADATA_TIMEOUT)
        .send()
        .await
        .map_err(|e| format!("{subject} could not be fetched: {}", request_reason(&e)))?;
    if !response.status().is_success() {
        return Err(format!(
            "{subject} could not be fetched: HTTP {}",
            response.status()
        ));
    }
    response
        .text()
        .await
        .map_err(|e| format!("{subject} could not be read: {}", request_reason(&e)))
}

/// The digest `asset` must have, read out of a `SHA256SUMS` body: the line naming that
/// file. `None` when the body names no such file, which each installer reports in its own
/// words — no other artifact is ever substituted.
pub(crate) fn sha256_for(body: &str, asset: &str) -> Option<String> {
    body.lines()
        .filter_map(parse_sha256_sidecar)
        .find(|(_, name)| name == asset)
        .map(|(digest, _)| digest)
}

/// The newest release of a managed tool: the tag its download URLs are built
/// from, and the version the installed copy is compared with. One lookup answers
/// both, so an update never resolves the same release twice (see
/// [`install_on_start`]). The error is a path-free, URL-free sentence because it
/// is recorded verbatim.
pub(crate) struct Release {
    pub(crate) tag: String,
    pub(crate) version: semver::Version,
}

/// The newest release, from the repository's `releases/latest` redirect
/// ([`fetch_latest_tag`] plus [`version_from_tag`]).
pub(crate) async fn latest_release(repo: &str, timeout: Duration) -> Result<Release, String> {
    let tag = fetch_latest_tag(repo, timeout).await?;
    let version = version_from_tag(&tag)
        .ok_or_else(|| "the newest release's tag carries no version".to_string())?;
    Ok(Release { tag, version })
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

/// A version whose publisher numbers it with fewer than three parts (`10.3`,
/// `10.3.1`), padded to semver: the Bot API reports itself as `Bot API 10.3`,
/// and its release tags carry the same two-part number.
#[must_use]
pub(crate) fn parse_short_version(text: &str) -> Option<semver::Version> {
    let text = text.trim();
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(semver::Version::new(major, minor, patch))
}

/// The release version a tag carries. The helper's tags look like `v1.5.140`
/// and bun's like `bun-v1.2.21`; a tag that is a bare version keeps its
/// prerelease suffix (`v1.5.140-rc1`), so the whole tag is tried before its
/// dash-separated segments; `None` when no token is a version.
#[must_use]
pub(crate) fn version_from_tag(tag: &str) -> Option<semver::Version> {
    parse_version_token(tag).or_else(|| tag.split(['-', '_', '/']).find_map(parse_version_token))
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

/// The rename-aside swap: move `dest` aside, rename the prepared `tmp` copy into
/// place, and restore the aside if the final rename fails. On Windows it is what
/// makes replacing a running exe possible at all — one cannot be overwritten or
/// deleted, but it CAN be renamed — and the ffmpeg installer swaps a whole directory
/// through it for the same reason: a half-placed copy must never stand where a whole
/// one did. `dest` must exist; the swap starts by moving it aside.
///
/// Aside removal is best-effort — while the old copy is still in use it cannot be
/// removed; the next successful swap clears it. Platform-neutral plain
/// `fs::rename`/`remove_file`/`remove_dir_all`, so it is unit-testable on every OS.
///
/// Failure messages carry no path (the location may be one the owner's own
/// package manager keeps), only the failing step and its io kind.
pub(crate) fn rename_aside_swap(tmp: &Path, dest: &Path) -> Result<(), String> {
    let aside = dest.with_extension("old");
    remove_any(&aside);
    fs::rename(dest, &aside)
        .map_err(|e| format!("the installed copy could not be moved aside ({})", e.kind()))?;
    if let Err(e) = fs::rename(tmp, dest) {
        // Last-resort restore: if this also fails the install is genuinely
        // broken, so surface that instead of discarding the error.
        if let Err(restore) = fs::rename(&aside, dest) {
            return Err(format!(
                "the new copy could not be moved into place ({}); the restore also failed ({}) — \
                 the file that was there is gone and must be put back by hand",
                e.kind(),
                restore.kind()
            ));
        }
        return Err(format!(
            "the new copy could not be moved into place ({})",
            e.kind()
        ));
    }
    remove_any(&aside);
    Ok(())
}

/// Remove whatever sits at `path`, best effort: a file a failed swap left beside its
/// destination, or the directory an ffmpeg install swaps out.
fn remove_any(path: &Path) {
    if path.is_dir() {
        let _ = fs::remove_dir_all(path);
    } else {
        let _ = fs::remove_file(path);
    }
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
        return Err("the directory it lives in could not be resolved".to_string());
    };
    fs::create_dir_all(dir).map_err(|e| {
        format!(
            "the directory it lives in could not be created ({})",
            e.kind()
        )
    })?;
    swap_binary_in_place(fresh, dest)?;
    set_executable(dest)
}

// ── Start-time install policy ───────────────────────────────────────

/// How long an existing copy waits before its version is checked, so the check
/// (and any download it leads to) never competes with the boot.
const REFRESH_DELAY: Duration = Duration::from_mins(5);

/// Bring one of the product's own tools to the newest release. `present` reports
/// whether the tool's own directory holds an executable copy — it decides
/// whether anything is compared at all and, after a failure, whether the host
/// was left with a working copy. A copy that is absent (or not executable) has
/// nothing to probe or compare, so it is installed straight away; a copy in
/// place is left untouched until [`REFRESH_DELAY`] has passed and only then
/// checked. Each step is a closure this function calls, so none of them can run
/// before that delay: `installed_version` answers with the copy's version (`None`
/// when it cannot run), `latest` looks the newest release up, and `install` runs
/// only when the two differ — a matching version is a [`tracing::debug`] line and
/// no download at all. `install` receives the release the comparison already
/// resolved (`None` when there was nothing to compare, so it resolves one
/// itself), which is what keeps the release tag from being fetched twice on the
/// update path. A working copy whose newest release cannot be looked up is left
/// as it is, with the failure recorded and retried on the next start; an
/// unusable one is repaired, because the copy is what must exist.
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
    installed_version: impl AsyncFnOnce() -> Option<semver::Version>,
    latest: impl AsyncFnOnce() -> Result<Release, String>,
    install: impl AsyncFnOnce(Option<Release>) -> Result<PathBuf, String>,
) {
    let missing = !present();
    let mut release = None;
    if !missing {
        // Both facts a comparison needs are kept off the boot path: the probe
        // spawns the tool and the update it may lead to downloads a release,
        // neither of which may compete with the interface coming up. The closures
        // are called only here, after the delay.
        tokio::time::sleep(REFRESH_DELAY).await;
        let installed = installed_version().await;
        match (installed, latest().await) {
            // The copy is the release on offer: nothing to download. SemVer
            // precedence, so a banner's build metadata never forces a re-download
            // of the same version.
            (Some(installed), Ok(latest)) if installed.cmp_precedence(&latest.version).is_eq() => {
                tracing::debug!("{tool} is already at {} — not downloading", latest.version);
                return;
            }
            // A copy that works but whose newest release cannot be looked up: the
            // failed check says nothing about the copy, so it is left alone.
            (Some(_), Err(reason)) => {
                report_failure(tool, &present, missing, &reason);
                return;
            }
            (_, Ok(latest)) => release = Some(latest),
            // Nothing to compare and no release: install (or repair) the copy and
            // let the step report why the release it needs cannot be reached.
            (None, Err(_)) => {}
        }
    }
    match install(release).await {
        Ok(path) if missing => tracing::info!("{tool} installed at {}", path.display()),
        Ok(path) => tracing::debug!("{tool} brought up to date at {}", path.display()),
        Err(reason) => report_failure(tool, &present, missing, &reason),
    }
}

/// Record a failed attempt at WARN. `present` is re-probed after the attempt so
/// the message can say what the host is left with — a failed step may have placed
/// the copy without registering it, or failed before anything was placed.
fn report_failure(tool: &str, present: &impl Fn() -> bool, missing: bool, reason: &str) {
    let reason = crate::util::truncate(reason, 1024);
    let state = match (present(), missing) {
        (true, true) => "was installed but not completely",
        (true, false) => "could not be brought fully up to date",
        (false, true) => "could not be installed",
        (false, false) => "left the host without a working copy",
    };
    tracing::warn!("{tool} {state} (retried on the next start): {reason}");
}

// ── Archive extraction ──────────────────────────────────────────────

/// Extract the single `file_name` binary from a tar.gz release archive into
/// `dir` and return its path — [`extract_named_files_tar_gz`] with one name.
pub(crate) fn extract_single_file_tar_gz(
    archive: &Path,
    dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let path = sole(extract_named_files_tar_gz(archive, dir, &[file_name], &[])?);
    set_executable(&path)?;
    Ok(path)
}

/// Extract the single `file_name` binary from a zip release archive into `dir`
/// and return its path — [`extract_named_files_zip`] with one name.
pub(crate) fn extract_single_file_zip(
    archive: &Path,
    dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let path = sole(extract_named_files_zip(archive, dir, &[file_name], &[])?);
    set_executable(&path)?;
    Ok(path)
}

/// The one path a one-name extraction yields: the collector returns a path per name
/// or the reason the archive does not carry it.
fn sole(mut paths: Vec<PathBuf>) -> PathBuf {
    paths.pop().expect("one name, one path")
}

/// Extract every name in `required`, and every name in `optional` that the archive
/// carries, from a tar.gz release archive into `dir`, and return the required names'
/// paths in that order. An entry is matched by file name at any depth but always
/// written to `<dir>/<name>`, so a nested vendor layout still lands correctly; `Err`
/// names the first required name that is not there as a regular file, while an
/// optional name that is absent is simply not written. The files land as the archive
/// carried them — a caller extracting programs beside text makes the programs
/// executable itself.
///
/// Failure messages name the operation and the io kind, never a path.
pub(crate) fn extract_named_files_tar_gz(
    archive: &Path,
    dir: &Path,
    required: &[&str],
    optional: &[&str],
) -> Result<Vec<PathBuf>, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("the release archive could not be opened ({})", e.kind()))?;
    let mut tar_archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let names = names_with_optional(required, optional);
    let mut extracted: Vec<Option<PathBuf>> = vec![None; names.len()];
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
        let Some(index) = index_of(&path, &names) else {
            continue;
        };
        let name = names[index];
        let out_path = dir.join(name);
        // `unpack` writes exactly to `out_path` (the entry's own internal path is
        // ignored), keeping the returned path correct for any archive layout.
        entry.unpack(&out_path).map_err(|e| {
            format!(
                "{name} could not be extracted from the release archive ({})",
                e.kind()
            )
        })?;
        extracted[index] = Some(out_path);
        if extracted.iter().all(Option::is_some) {
            break;
        }
    }
    collected(required, extracted)
}

/// Extract every name in `required`, and every name in `optional` that the archive
/// carries, from a zip release archive into `dir`, and return the required names' paths
/// in that order. An entry is matched by file name at any depth (zip archives use `/`
/// separators internally) but always written to the fixed `<dir>/<name>` — never the
/// entry's internal path — so a nested `bun-<target>/` vendor layout still lands
/// correctly; `Err` names the first required name that is not there as a regular file,
/// while an optional name that is absent is simply not written.
///
/// Failure messages name the operation and a reason the archive library's own
/// error yields — the io kind it wraps, or its own short sentence for a malformed
/// archive or a missing entry — never a path.
pub(crate) fn extract_named_files_zip(
    archive: &Path,
    dir: &Path,
    required: &[&str],
    optional: &[&str],
) -> Result<Vec<PathBuf>, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("the release archive could not be opened ({})", e.kind()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("the release archive could not be read ({})", zip_reason(&e)))?;
    let names = names_with_optional(required, optional);
    let mut extracted: Vec<Option<PathBuf>> = vec![None; names.len()];
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
        let Some(index) = index_of(Path::new(entry.name()), &names) else {
            continue;
        };
        let name = names[index];
        let out_path = dir.join(name);
        let mut out_file = fs::File::create(&out_path)
            .map_err(|e| format!("{name} could not be written ({})", e.kind()))?;
        std::io::copy(&mut entry, &mut out_file).map_err(|e| {
            format!(
                "{name} could not be extracted from the release archive ({})",
                e.kind()
            )
        })?;
        extracted[index] = Some(out_path);
        if extracted.iter().all(Option::is_some) {
            break;
        }
    }
    collected(required, extracted)
}

/// The names one extraction looks for: the required ones first, then the optional ones,
/// so `collected` reads the required paths off the front. The scan can stop only once every
/// name is found, since an optional name the archive lacks is known to be absent only at its
/// end.
fn names_with_optional<'a>(required: &[&'a str], optional: &[&'a str]) -> Vec<&'a str> {
    required.iter().chain(optional).copied().collect()
}

/// The position in `names` an archive entry's file name matches, ignoring the
/// entry's own internal path.
fn index_of(path: &Path, names: &[&str]) -> Option<usize> {
    let file_name = path.file_name()?;
    names.iter().position(|name| file_name == OsStr::new(name))
}

/// The required names' paths, read off the front of `extracted` — the optional names
/// follow them in that vector — or the one sentence naming the first required name the
/// archive did not carry.
fn collected(required: &[&str], extracted: Vec<Option<PathBuf>>) -> Result<Vec<PathBuf>, String> {
    extracted
        .into_iter()
        .zip(required)
        .map(|(path, name)| path.ok_or_else(|| format!("the release archive carries no {name}")))
        .collect()
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

/// Make a binary executable by resetting its full mode to 0o755: an archive may
/// carry no mode at all, and both the place-then-install step and the probe that
/// runs an extracted copy before placing it need one. The failure message names
/// the io kind only — never the path.
#[cfg(unix)]
pub(crate) fn set_executable(path: &Path) -> Result<(), String> {
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

// ── Install directories ─────────────────────────────────────────────

/// bun-standard user install dir (`~/.bun/bin` on unix, `%USERPROFILE%\.bun\bin`
/// on Windows). Resolved via [`directories::UserDirs`] like the rest of the
/// codebase, not a literal `$HOME` splice.
#[must_use]
pub(crate) fn bun_bin_dir() -> Option<PathBuf> {
    directories::UserDirs::new().map(|d| d.home_dir().join(".bun").join("bin"))
}

/// The standard per-user programs directory: `~/.local/bin` on unix,
/// `%LOCALAPPDATA%\Programs` on Windows — the two places per-user programs
/// belong, resolved through `directories` like the rest of the codebase.
///
/// It is the directory the product's own command goes into on unix, and the base
/// the product's own folder — and the managed tools' own per-user folders —
/// sit under on Windows.
#[must_use]
fn user_programs_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        directories::UserDirs::new().map(|d| d.home_dir().join(".local").join("bin"))
    }
    #[cfg(not(unix))]
    {
        directories::BaseDirs::new().map(|d| d.data_local_dir().join("Programs"))
    }
}

/// Where the product itself lives when it was installed from a published file:
/// the standard per-user programs directory itself on unix, and the product's
/// own folder inside it on Windows, which has no shared per-user programs
/// directory a single executable can be dropped into. `install.sh`/
/// `install.ps1` put the ready-made file at exactly this path.
#[must_use]
pub(crate) fn mahbot_install_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        user_programs_dir()
    }
    #[cfg(not(unix))]
    {
        user_programs_dir().map(|dir| dir.join("MahBot"))
    }
}

/// The name the product's own file carries: `mahbot.exe` on Windows, `mahbot`
/// elsewhere — in a release archive, in a cargo install's `bin` directory, and at
/// the file's path inside [`mahbot_install_dir`], so one definition serves all
/// three.
#[must_use]
pub(crate) fn product_file_name() -> &'static str {
    if cfg!(windows) {
        "mahbot.exe"
    } else {
        "mahbot"
    }
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
        user_programs_dir()
    }
    #[cfg(not(unix))]
    {
        user_programs_dir().map(|dir| dir.join("chrome-use"))
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
        crate::util::dir_is_writable(system),
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
    fn a_two_part_version_pads_to_semver() {
        assert_eq!(
            parse_short_version("10.3"),
            Some(semver::Version::new(10, 3, 0))
        );
        assert_eq!(
            parse_short_version("10.3.1"),
            Some(semver::Version::new(10, 3, 1))
        );
        assert_eq!(parse_short_version("Telegram Bot API"), None);
        assert_eq!(parse_short_version("10.3.1.2"), None);
    }

    #[test]
    fn release_tags_carry_a_version_after_their_prefix() {
        // chrome-use tags are bare versioned tags, bun's carry a `bun-` prefix.
        assert_eq!(
            version_from_tag("v1.5.140"),
            Some(semver::Version::new(1, 5, 140))
        );
        assert_eq!(
            version_from_tag("bun-v1.2.21"),
            Some(semver::Version::new(1, 2, 21))
        );
        // A bare tagged version keeps its prerelease suffix rather than being
        // read as its release.
        assert_eq!(
            version_from_tag("v1.5.140-rc1"),
            Some(semver::Version::parse("1.5.140-rc1").unwrap())
        );
        assert_eq!(version_from_tag("latest"), None);
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

    #[test]
    fn a_sha256_body_names_the_digest_of_one_file() {
        let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let body = format!(
            "{hash}  bun-linux-x64.zip\n{}  LICENCE.txt\n",
            "f".repeat(64)
        );

        assert_eq!(
            sha256_for(&body, "bun-linux-x64.zip"),
            Some(hash.to_string())
        );
        // A file the body does not name has no digest, so no other one stands in.
        assert_eq!(sha256_for(&body, "bun-darwin-arm64.zip"), None);
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
