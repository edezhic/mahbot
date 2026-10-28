//! Install and update the product's own build of ffmpeg and ffprobe.
//!
//! The pair is built and published by the product itself, in a release stream of
//! its own: prereleases tagged `ffmpeg-v<version>-mahbot.<revision>`. That separate
//! stream is what keeps the product's updater untouched — it reads
//! `releases/latest` and `version.txt`, which never point at one — while the
//! archives stay downloadable by a plain HTTP client with no credentials.
//!
//! Nothing about the build is compiled into the product. At start the stream's
//! listing is read and the newest build it offers for this host is installed,
//! unless that already is the copy in use, so a build published after this binary
//! was needs nothing from a release of the product. The newest is the highest
//! (upstream version, build revision) among the releases that carry this platform's
//! archive: every one of the six is a real build, and an archive missing from a
//! release is a failed release, never a substitution.
//!
//! Integrity comes from the release itself, the way it does for the product's other
//! managed binaries: the release's `SHA256SUMS` is fetched, and the archive is
//! checked against the line for that archive. No digest of a release artefact is
//! carried by this build, so a published release installs as it stands — and a
//! checksum file rewritten together with its archive is not detected, which is the
//! price of that.
//!
//! The build is GPLv3 — upstream ffmpeg with libx264, libx265, libvpx, libopus,
//! zimg, zlib and OpenSSL (SChannel on Windows, which is the system's own) — so
//! what travels with the binaries is that licence, the notices of those components
//! and a `SOURCE.txt` naming the source of every one of them. A build published
//! before those notices existed installs with the licence and source note it carries
//! rather than not at all; the notices arrive with the next build taken.
//!
//! The copy in use is the newest one that runs: a failed install leaves the host
//! with whatever it had, and never with nothing. A failure names the step that
//! failed and carries no path and no URL, because it is recorded and shown verbatim.
//!
//! Nothing is written outside the product's own configuration directory — no
//! directory the owner's package manager owns, and no entry on `PATH`.
//!
//! The three video operations that run this copy belong to [`super`].

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use semver::Version;

use crate::util::UnwrapPoison;
use crate::util::managed_bin;

// ── The stream ──────────────────────────────────────────────────────

/// The tag prefix of this stream, and the marker separating the upstream version
/// from the build revision inside a tag.
const TAG_PREFIX: &str = "ffmpeg-v";
const REVISION_MARKER: &str = "-mahbot.";

/// Test hook: a host answering the way the real pair of hosts does — the listing
/// at `?per_page=&page=`, and each release's files at `{base}/{tag}/{asset}`.
pub(super) const RELEASES_URL_ENV: &str = "MAHBOT_FFMPEG_RELEASES_URL";

/// How long a staged binary may take to print its banner: a local file needing
/// longer than this is a copy that cannot run.
const VERSION_TIMEOUT: Duration = Duration::from_secs(10);

/// The stream's listing, named the way every failure about it names it.
const LISTING_SUBJECT: &str = "the ffmpeg release listing";

/// The text of the licence the build is distributed under, and where that build's source
/// is. Both have travelled since the stream's first build, so a release without either is
/// a failed install — a licence that does not travel with the binary is not honoured by
/// shipping the binary alone.
const LICENCE_AND_SOURCE: [&str; 2] = ["LICENCE.txt", "SOURCE.txt"];

/// The notices of the libraries linked into the build. Every archive the release job
/// publishes carries it — that job refuses one that does not — but a build published
/// before the notices existed carries none, and installing it with the material it has
/// serves a host better than leaving it with no pair at all until a newer build lands.
const COMPONENT_NOTICES: &str = "LICENCES.txt";

// ── Naming ──────────────────────────────────────────────────────────

/// The two binaries' file names on this platform. Windows decides executability
/// by extension ([`crate::util::is_executable`]), so the suffix is part of the
/// name there and nowhere else.
const fn ffmpeg_file() -> &'static str {
    if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    }
}

const fn ffprobe_file() -> &'static str {
    if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    }
}

/// What a tag — or a directory named after one — spells: the upstream version and
/// the build revision (`ffmpeg-v7.1-mahbot.1` → 7.1 and 1). Another stream's tag,
/// a staging directory and a half-written name all read as `None`.
#[must_use]
fn parse_tag(name: &str) -> Option<(semver::Version, u32)> {
    let rest = name.strip_prefix(TAG_PREFIX)?;
    let (upstream, revision) = rest.split_once(REVISION_MARKER)?;
    Some((
        managed_bin::parse_short_version(upstream)?,
        revision.parse().ok()?,
    ))
}

/// A build the stream offers: the tag that names its release, and the upstream version
/// its binaries must report — the tag's own, so an archive carrying another FFmpeg
/// cannot pass for it.
struct Build {
    tag: String,
    upstream: Version,
}

/// This host's archive for release `tag`: `ffmpeg-<tag stem>-<os>-<arch>.<ext>`,
/// the extension being `zip` on Windows and `tar.gz` everywhere else. The stem
/// comes from the tag, so the archive name and the release it lives under cannot
/// disagree; every one of the six combinations is a real build.
#[must_use]
fn asset_name(tag: &str, os: &str, arch: &str) -> String {
    let stem = tag.strip_prefix(TAG_PREFIX).unwrap_or(tag);
    let extension = if os == "windows" { "zip" } else { "tar.gz" };
    format!("ffmpeg-{stem}-{os}-{arch}.{extension}")
}

// ── The copy in use ─────────────────────────────────────────────────

/// The pair in use.
#[derive(Debug)]
pub(super) struct Binaries {
    pub(super) ffmpeg: PathBuf,
    pub(super) ffprobe: PathBuf,
}

/// The directory the product's own copies live in: one subdirectory per tag,
/// inside the product's configuration directory — never the daemon's temp root,
/// which is swept, and never a directory the owner's package manager owns.
fn root() -> Result<PathBuf, String> {
    crate::config::default_config_dir()
        .map(|dir| dir.join("ffmpeg"))
        .map_err(|_| "the product's configuration directory is unavailable".to_string())
}

/// The copy in use under `root`, with the build its directory is named after: the newest
/// usable one by (upstream version, build revision). A directory whose name does not
/// parse, and a directory holding anything but two executable files, are not copies at
/// all.
fn chosen(root: &Path) -> Option<(Build, Binaries)> {
    fs::read_dir(root)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let tag = entry.file_name().into_string().ok()?;
            let rank = parse_tag(&tag)?;
            let binaries = binaries_at(&entry.path())?;
            Some((rank, tag, binaries))
        })
        .max_by(|(left, _, _), (right, _, _)| left.cmp(right))
        .map(|((upstream, _), tag, binaries)| (Build { tag, upstream }, binaries))
}

/// The pair at the fixed layout inside `dir`.
fn binaries_in(dir: &Path) -> Binaries {
    let bin = dir.join("bin");
    Binaries {
        ffmpeg: bin.join(ffmpeg_file()),
        ffprobe: bin.join(ffprobe_file()),
    }
}

/// The pair inside one tag directory, or `None` when either file is missing or
/// cannot be executed.
fn binaries_at(dir: &Path) -> Option<Binaries> {
    let binaries = binaries_in(dir);
    (crate::util::is_executable(&binaries.ffmpeg) && crate::util::is_executable(&binaries.ffprobe))
        .then_some(binaries)
}

// ── Installation ────────────────────────────────────────────────────

/// Holds the one install in flight, so the boot install and a first use never run
/// beside each other.
static INSTALLING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Why this process's attempt at the stream failed, when one did.
static ATTEMPT_FAILURE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Run `attempt` as the one install in flight, so the caller that waited re-reads what is
/// in use instead of downloading what the winner already placed.
async fn installing<T, E>(attempt: impl AsyncFnOnce() -> Result<T, E>) -> Result<T, E> {
    let _queue = INSTALLING.lock().await;
    attempt().await
}

/// The copy a video operation runs: whatever pair is in use, and — when there is
/// none — the newest build the stream offers, installed first.
///
/// A copy already in place is used as it stands, with no proof and no network round-trip:
/// it was proved before it was placed, and the boot task is what re-proves it — a pair
/// damaged after that proof keeps serving video until the next start. Installing is behind
/// [`installing`], so a first use never runs beside the boot task's own install, and the
/// loser of that race re-checks before downloading anything.
///
/// A failed attempt stands for the rest of the process, so no further video file
/// pays for it: the next start is what tries again.
pub(super) async fn ensure_installed() -> Result<Binaries, String> {
    let root = root()?;
    if let Some((_, binaries)) = chosen(&root) {
        return Ok(binaries);
    }
    let tag = installing(async || {
        // The caller that held the lock may have installed a copy while this one
        // waited.
        if let Some((build, _)) = chosen(&root) {
            return Ok(build.tag);
        }
        if let Some(reason) = ATTEMPT_FAILURE.lock().unwrap_poison().clone() {
            return Err(reason);
        }
        let build = newest().await.map_err(recorded)?;
        install(&root, &build).await.map_err(recorded)?;
        Ok(build.tag)
    })
    .await?;
    Ok(binaries_in(&root.join(tag)))
}

/// What the host is running, proved once per start and carried into the failure line.
#[derive(Debug)]
enum Standing {
    /// A copy that runs, exits clean and reports the build its directory is named after.
    Working(String),
    /// A copy that does not: damaged, half-written, or another build under that name.
    Damaged(String),
    /// No copy at all.
    Missing,
}

/// A start that failed: the step that failed, and what the host is left running.
#[derive(Debug)]
struct Failure {
    reason: String,
    standing: Standing,
}

/// The boot task: bring this host to the newest build the stream offers, called once
/// by the binary and never failing the boot.
pub async fn install_on_start() {
    match update().await {
        Ok(None) => tracing::debug!("ffmpeg needs nothing at this start"),
        Ok(Some(tag)) => tracing::info!("ffmpeg {tag} installed"),
        Err(failure) => report_failure(&failure),
    }
}

/// The one WARN line a failed start leaves, stating what the host is left running and
/// naming the step that failed.
fn report_failure(failure: &Failure) {
    let reason = &failure.reason;
    match &failure.standing {
        Standing::Damaged(tag) => tracing::warn!(
            "ffmpeg {tag} is damaged and could not be replaced (retried on the next \
             start): {reason}"
        ),
        Standing::Working(tag) => tracing::warn!(
            "ffmpeg is running {tag}; the newest build is not installed (retried on the \
             next start): {reason}"
        ),
        Standing::Missing => {
            tracing::warn!("ffmpeg is not installed (retried on the next start): {reason}");
        }
    }
}

/// What the host runs now, proved with two local `-version` runs and no network.
async fn standing(root: &Path) -> Standing {
    match chosen(root) {
        Some((in_use, binaries)) if proves_itself(&binaries, &in_use.upstream).await => {
            Standing::Working(in_use.tag)
        }
        Some((in_use, _)) => Standing::Damaged(in_use.tag),
        None => Standing::Missing,
    }
}

/// Install the newest build the stream offers unless the copy in use already is it and
/// still proves itself. `Ok(None)` is nothing to do; `Ok(Some(tag))` names what was
/// installed.
///
/// A stream that cannot be read is not a failure for a host whose copy proves itself: it
/// runs the build it is named after, and the next start that can read the listing takes
/// whatever newer build it offers.
async fn update() -> Result<Option<String>, Failure> {
    let root = root().map_err(|reason| Failure {
        reason,
        standing: Standing::Missing,
    })?;
    installing(async || {
        let current = standing(&root).await;
        let build = match newest().await {
            Ok(build) => build,
            Err(reason) => {
                return match current {
                    Standing::Working(tag) => {
                        tracing::debug!("keeping ffmpeg {tag}: {reason}");
                        Ok(None)
                    }
                    other => Err(Failure {
                        reason: recorded(reason),
                        standing: other,
                    }),
                };
            }
        };
        if let Standing::Working(tag) = &current
            && tag == &build.tag
        {
            return Ok(None);
        }
        install(&root, &build).await.map_err(|reason| Failure {
            reason: recorded(reason),
            standing: current,
        })?;
        Ok(Some(build.tag))
    })
    .await
}

/// Whether this pair is still the build it is named after: both binaries run, exit clean
/// and report `upstream`. Two local `-version` runs and no network — the check a directory
/// name cannot give, and what replaces a pair damaged after it was placed instead of
/// failing every job that runs it.
async fn proves_itself(binaries: &Binaries, upstream: &Version) -> bool {
    reports_upstream(&binaries.ffmpeg, upstream).await
        && reports_upstream(&binaries.ffprobe, upstream).await
}

/// Record a failed attempt, so no later operation in this process pays for it again.
fn recorded(reason: String) -> String {
    *ATTEMPT_FAILURE.lock().unwrap_poison() = Some(reason.clone());
    reason
}

/// The newest build the stream offers for this host: the highest (upstream version,
/// build revision) among the releases that carry this platform's archive.
///
/// Every page of the listing is read before the answer is given, because its order
/// is publication, not version: a repair of an older line is published after the
/// newer line exists. The one thing this cannot pick is a release that exists
/// without this platform's archive — a failed release, never a substitution.
async fn newest() -> Result<Build, String> {
    let (os, arch) = managed_bin::host_os_arch()?;
    let base = stream_url("https://api.github.com/repos", "releases");
    let mut newest: Option<((Version, u32), String)> = None;
    for page in 1..=managed_bin::LISTING_PAGES {
        let releases = managed_bin::release_page(&base, page, LISTING_SUBJECT)
            .await?
            .ok_or_else(|| {
                format!("{LISTING_SUBJECT} could not be read: it is not a JSON array")
            })?;
        for release in &releases {
            let Some(tag) = release["tag_name"].as_str() else {
                continue;
            };
            let Some(rank) = parse_tag(tag) else {
                continue;
            };
            let asset = asset_name(tag, os, arch);
            let carries_ours = release["assets"]
                .as_array()
                .is_some_and(|assets| assets.iter().any(|a| a["name"].as_str() == Some(&asset)));
            if !carries_ours || newest.as_ref().is_some_and(|(best, _)| best >= &rank) {
                continue;
            }
            newest = Some((rank, tag.to_string()));
        }
        if releases.len() < managed_bin::LISTING_PAGE_SIZE {
            break;
        }
    }
    let ((upstream, _), tag) =
        newest.ok_or_else(|| format!("no ffmpeg build is published for {os}-{arch}"))?;
    Ok(Build { tag, upstream })
}

/// Download, verify, prove and place the build's pair, then leave that copy as the only
/// one under the root.
///
/// Runs only as the process's one install ([`installing`]): with one attempt at a time
/// the download directory and the staging beside it are this attempt's own.
///
/// The order is what makes a failure harmless: this platform's line in the release's
/// checksum file is read before anything is downloaded, the archive is verified as it
/// streams, and the pair is proven to run before the directory is renamed into place.
async fn install(root: &Path, build: &Build) -> Result<(), String> {
    use crate::util::http::{DownloadSizeCheck, download_verified};

    let (os, arch) = managed_bin::host_os_arch()?;
    let tag = &build.tag;
    let asset = asset_name(tag, os, arch);
    let base = stream_url("https://github.com", "releases/download");

    let sums = managed_bin::fetch_sha256_sums(
        &format!("{base}/{tag}/SHA256SUMS"),
        "the release's checksum file",
    )
    .await?;
    let digest = managed_bin::sha256_for(&sums, &asset)
        .ok_or_else(|| "the release carries no checksum for this platform's archive".to_string())?;

    let downloads = root.join("downloads");
    let _ = fs::remove_dir_all(&downloads);
    fs::create_dir_all(&downloads)
        .map_err(|e| format!("the download directory could not be created ({})", e.kind()))?;
    let archive = downloads.join(&asset);
    download_verified(
        managed_bin::download_client()?,
        &format!("{base}/{tag}/{asset}"),
        &archive,
        &digest,
        None,
        DownloadSizeCheck::Exact,
        |_, _| {},
    )
    .await
    .map_err(|e| {
        format!(
            "the ffmpeg release could not be downloaded: {}",
            managed_bin::download_reason(&e)
        )
    })?;

    // One staging directory per attempt, on the same filesystem as its destination
    // so the final rename is a rename — the download directory above has already
    // made the root. The guard removes it on every failure the `?` below carries
    // out, and has nothing to remove after a successful `place`.
    let staging = tempfile::Builder::new()
        .prefix(".staging-")
        .tempdir_in(root)
        .map_err(|e| format!("the staging directory could not be created ({})", e.kind()))?;
    extract_into(&archive, staging.path())?;
    let _ = fs::remove_dir_all(&downloads);

    for (name, tool) in [("ffmpeg", ffmpeg_file()), ("ffprobe", ffprobe_file())] {
        if !reports_upstream(&staging.path().join("bin").join(tool), &build.upstream).await {
            return Err(format!(
                "the downloaded {name} could not run, or does not report ffmpeg {}",
                build.upstream
            ));
        }
    }
    place(staging.path(), &root.join(tag))?;
    prune(root, tag);
    Ok(())
}

/// One of this stream's two base URLs on the product's own release host — the repository's
/// `releases/download` for the assets, its API `releases` for the listing — or the test
/// hook's stand-in, which answers both with one base.
fn stream_url(host: &str, path: &str) -> String {
    match std::env::var(RELEASES_URL_ENV) {
        Ok(base) => base.trim_end_matches('/').to_string(),
        Err(_) => format!("{host}/{}/{path}", managed_bin::release_repo()),
    }
}

/// Unpack one release archive into `staging` in a single pass: the two binaries under
/// `bin/`, the licence material at the root — exactly the layout [`binaries_at`] reads.
/// The notices are taken when the archive carries them and are not required: the pair is
/// already staged, so a build from before the notices existed still installs.
fn extract_into(archive: &Path, staging: &Path) -> Result<(), String> {
    let bin_dir = staging.join("bin");
    fs::create_dir_all(&bin_dir).map_err(|e| {
        format!(
            "the staged copy's directory could not be created ({})",
            e.kind()
        )
    })?;
    let tools = [ffmpeg_file(), ffprobe_file()];
    // The tools come first in `required`, in this order: `extract` returns their paths in
    // the same order.
    let required = tools
        .iter()
        .copied()
        .chain(LICENCE_AND_SOURCE)
        .collect::<Vec<_>>();
    let paths = extract(archive, staging, &required, &[COMPONENT_NOTICES])?;
    for (name, path) in tools.into_iter().zip(&paths) {
        managed_bin::set_executable(path)?;
        fs::rename(path, bin_dir.join(name))
            .map_err(|e| format!("the staged copy could not be arranged ({})", e.kind()))?;
    }
    Ok(())
}

/// Extract `required` and `optional` out of `archive` into `dir`, whichever container
/// this platform's release uses.
fn extract(
    archive: &Path,
    dir: &Path,
    required: &[&str],
    optional: &[&str],
) -> Result<Vec<PathBuf>, String> {
    if archive.extension().and_then(OsStr::to_str) == Some("zip") {
        managed_bin::extract_named_files_zip(archive, dir, required, optional)
    } else {
        managed_bin::extract_named_files_tar_gz(archive, dir, required, optional)
    }
}

/// The upstream version the first line of a `-version` banner claims, at the whole
/// version a tag names: the product's own build reports `7.1.3`, a vendor's
/// `7.1.3-static` reads as `7.1.3` too, and a build that spells only `7.1` is 7.1.0 —
/// the same version an `ffmpeg-v7.1-…` tag names.
fn banner_version(stdout: &[u8]) -> Option<Version> {
    let text = String::from_utf8_lossy(stdout);
    let line = text.lines().next()?;
    let token = line
        .split_whitespace()
        .skip_while(|word| *word != "version")
        .nth(1)?;
    managed_bin::parse_short_version(token.split_once('-').map_or(token, |(head, _)| head))
}

/// Whether `bin` runs, exits clean and reports `upstream` — the proof no checksum can
/// give, that the file is the expected build and that it can execute on this host.
async fn reports_upstream(bin: &Path, upstream: &Version) -> bool {
    let Ok(output) = super::run_tool(bin, &["-version".into()], VERSION_TIMEOUT).await else {
        return false;
    };
    output.success && banner_version(&output.stdout).as_ref() == Some(upstream)
}

/// Move `staging` to `final_dir` in one step, so a half-installed directory can
/// never stand where a whole one did. An existing directory — a copy that failed its
/// proof, or what a failed install left behind — is moved aside by the crate's own swap,
/// which also puts it back when the swap itself fails.
fn place(staging: &Path, final_dir: &Path) -> Result<(), String> {
    if !final_dir.exists() {
        return fs::rename(staging, final_dir)
            .map_err(|e| format!("the new copy could not be moved into place ({})", e.kind()));
    }
    managed_bin::rename_aside_swap(staging, final_dir)
}

/// Leave nothing this install made and no longer needs: every copy but the one just
/// installed — an older one, and a newer one the stream no longer offers or that failed
/// to prove itself, which [`chosen`] would otherwise keep picking over the one just proved
/// — plus every leftover that is not a copy at all (an attempt's staging, its aside, the
/// download, a name from another layout).
///
/// Infallible in effect: each removal is best effort, so a file another process is
/// using can never fail an install that already succeeded. On Windows that is also
/// what a copy a running operation still holds looks like: it stays until a later
/// start.
fn prune(root: &Path, tag: &str) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name == OsStr::new(tag) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            let _ = fs::remove_dir_all(&path);
        } else {
            let _ = fs::remove_file(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    #[cfg(unix)]
    use std::{
        collections::HashMap,
        io::Read as _,
        net::{TcpListener, TcpStream},
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use super::*;
    #[cfg(unix)]
    use crate::ffmpeg::ReleasesHook;

    /// The environment lock [`ReleasesHook`] needs, held for the caller's scope. A
    /// `MutexGuard` binding cannot be held across an await point, and this one must outlive
    /// the awaits around the hook it guards, so it travels inside a struct.
    #[cfg(unix)]
    #[must_use]
    struct EnvHeld {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    /// Holds the environment lock for as long as the returned value lives.
    #[cfg(unix)]
    fn env_held() -> EnvHeld {
        EnvHeld {
            _lock: crate::util::test::env_lock().lock().unwrap_poison(),
        }
    }

    /// A tag directory holding two executable files, the way an install leaves it.
    fn install_fake(root: &Path, tag: &str) -> PathBuf {
        let bin = root.join(tag).join("bin");
        fs::create_dir_all(&bin).expect("create tag directory");
        for tool in [ffmpeg_file(), ffprobe_file()] {
            let path = bin.join(tool);
            fs::write(&path, b"BIN").expect("write binary");
            managed_bin::set_executable(&path).expect("make executable");
        }
        root.join(tag)
    }

    /// The build a tag names, for the checks that drive [`install`] and the banner.
    fn build(tag: &str) -> Build {
        let (upstream, _) = parse_tag(tag).expect("a parsable tag");
        Build {
            tag: tag.to_string(),
            upstream,
        }
    }

    #[test]
    fn only_a_tag_name_parses() {
        assert_eq!(
            parse_tag("ffmpeg-v7.1-mahbot.1"),
            Some((semver::Version::new(7, 1, 0), 1))
        );
        assert_eq!(
            parse_tag("ffmpeg-v7.1.2-mahbot.12").map(|(version, _)| version),
            Some(semver::Version::new(7, 1, 2))
        );
        // Another stream's tag, a staging directory, and a half-written name.
        assert_eq!(parse_tag("tgapi-v10.3"), None);
        assert_eq!(parse_tag(".staging-Kd7x2p"), None);
        assert_eq!(parse_tag("ffmpeg-v7.1"), None);
        assert_eq!(parse_tag("ffmpeg-v7.1-mahbot.x"), None);
    }

    #[test]
    fn every_platform_gets_its_own_archive() {
        let tag = "ffmpeg-v7.1-mahbot.2";
        assert_eq!(
            asset_name(tag, "macos", "aarch64"),
            "ffmpeg-7.1-mahbot.2-macos-aarch64.tar.gz"
        );
        assert_eq!(
            asset_name(tag, "macos", "x86_64"),
            "ffmpeg-7.1-mahbot.2-macos-x86_64.tar.gz"
        );
        assert_eq!(
            asset_name(tag, "linux", "aarch64"),
            "ffmpeg-7.1-mahbot.2-linux-aarch64.tar.gz"
        );
        assert_eq!(
            asset_name(tag, "linux", "x86_64"),
            "ffmpeg-7.1-mahbot.2-linux-x86_64.tar.gz"
        );
        assert_eq!(
            asset_name(tag, "windows", "x86_64"),
            "ffmpeg-7.1-mahbot.2-windows-x86_64.zip"
        );
        // Windows on ARM is its own build, never the x64 one.
        assert_eq!(
            asset_name(tag, "windows", "aarch64"),
            "ffmpeg-7.1-mahbot.2-windows-aarch64.zip"
        );
    }

    #[test]
    fn the_newest_copy_on_disk_is_the_one_in_use() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        install_fake(root, "ffmpeg-v6.0-mahbot.1");
        install_fake(root, "ffmpeg-v7.1-mahbot.0");
        install_fake(root, "ffmpeg-v7.2-mahbot.1");
        fs::create_dir_all(root.join(".staging-Kd7x2p")).expect("create staging");
        fs::create_dir(root.join("not-a-tag")).expect("create junk directory");

        // The newest by (upstream version, build revision): neither a staging
        // directory nor a name from another layout is a copy.
        let (build, binaries) = chosen(root).expect("a copy is in use");
        assert_eq!(build.tag, "ffmpeg-v7.2-mahbot.1");
        assert_eq!(
            binaries.ffmpeg,
            root.join("ffmpeg-v7.2-mahbot.1")
                .join("bin")
                .join(ffmpeg_file())
        );

        // A newer revision of the same upstream wins over the older one.
        install_fake(root, "ffmpeg-v7.2-mahbot.2");
        assert_eq!(
            chosen(root).expect("a copy is in use").0.tag,
            "ffmpeg-v7.2-mahbot.2"
        );

        // A tagged directory holding one binary is not a copy.
        let broken = root.join("ffmpeg-v7.2-mahbot.2");
        fs::remove_file(broken.join("bin").join(ffprobe_file())).expect("remove ffprobe");
        assert_eq!(
            chosen(root).expect("a copy is in use").0.tag,
            "ffmpeg-v7.2-mahbot.1"
        );

        // A newer upstream wins over a newer revision of an older one.
        fs::remove_dir_all(root.join("ffmpeg-v7.2-mahbot.1")).expect("remove the newer build");
        assert_eq!(
            chosen(root).expect("a copy is in use").0.tag,
            "ffmpeg-v7.1-mahbot.0"
        );
    }

    #[test]
    fn pruning_leaves_only_the_copy_just_installed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        for tag in [
            "ffmpeg-v6.0-mahbot.1",
            "ffmpeg-v7.1-mahbot.0",
            "ffmpeg-v7.1-mahbot.2",
        ] {
            install_fake(root, tag);
        }
        fs::create_dir_all(root.join(".staging-Kd7x2p")).expect("create staging");
        fs::create_dir_all(root.join("downloads")).expect("create downloads");

        // A copy newer than the one installed goes too: it is one the stream no longer
        // offers or one that failed to prove itself, and `chosen` would keep picking it
        // over the copy just proved — a download on every start that never converges.
        prune(root, "ffmpeg-v7.1-mahbot.0");
        assert!(
            root.join("ffmpeg-v7.1-mahbot.0").exists(),
            "the installed copy stays"
        );
        for gone in [
            "ffmpeg-v6.0-mahbot.1",
            "ffmpeg-v7.1-mahbot.2",
            ".staging-Kd7x2p",
            "downloads",
        ] {
            assert!(!root.join(gone).exists(), "{gone} is gone");
        }
    }

    #[test]
    fn the_banner_names_the_upstream_version() {
        let tagged = build("ffmpeg-v7.1.3-mahbot.2");
        assert_eq!(
            tagged.upstream,
            Version::new(7, 1, 3),
            "the tag names the version every banner is compared against"
        );
        // What the release's own build prints: upstream's banner carries the
        // upstream version and nothing of the product's revision.
        assert_eq!(
            banner_version(b"ffmpeg version 7.1.3 Copyright (c) 2000-2024 the FFmpeg developers\n"),
            Some(tagged.upstream.clone())
        );
        // A vendor's suffix is not part of the version.
        assert_eq!(
            banner_version(b"ffprobe version 7.1.3-static Copyright (c) 2000-2024\n"),
            Some(tagged.upstream)
        );
        // A banner spelling only `major.minor` carries no patch to read.
        assert_eq!(
            banner_version(b"ffmpeg version 7.1 Copyright\n"),
            Some(Version::new(7, 1, 0))
        );
        assert_eq!(banner_version(b"nothing to see\n"), None);
        assert_eq!(banner_version(b""), None);
    }

    /// The entries a release archive carries: the pair under `bin/` inside a vendor
    /// directory, the licence material at the root. The stand-in binaries print the
    /// upstream version the tag names, so the pair is provable the way a real build is.
    fn release_entries(tag: &str) -> Vec<(String, Vec<u8>)> {
        let upstream = parse_tag(tag).expect("a parsable tag").0;
        let script =
            format!("#!/bin/sh\necho 'ffmpeg version {upstream} stand-in build'\n").into_bytes();
        vec![
            (format!("pkg/bin/{}", ffmpeg_file()), script.clone()),
            (format!("pkg/bin/{}", ffprobe_file()), script),
            ("LICENCE.txt".to_string(), b"GPLv3".to_vec()),
            (
                "LICENCES.txt".to_string(),
                b"x264: GPL-2.0-or-later\n".to_vec(),
            ),
            ("SOURCE.txt".to_string(), b"upstream source".to_vec()),
        ]
    }

    fn write_tar_gz(path: &Path, entries: &[(String, Vec<u8>)]) {
        let encoder = flate2::write::GzEncoder::new(
            fs::File::create(path).expect("create archive"),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(encoder);
        for (name, contents) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            builder
                .append_data(&mut header, name.as_str(), contents.as_slice())
                .expect("append entry");
        }
        builder
            .into_inner()
            .expect("finish archive")
            .finish()
            .expect("finish gzip");
    }

    fn write_zip(path: &Path, entries: &[(String, Vec<u8>)]) {
        let mut writer = zip::ZipWriter::new(fs::File::create(path).expect("create archive"));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, contents) in entries {
            writer
                .start_file(name.as_str(), options)
                .expect("start entry");
            writer.write_all(contents).expect("write entry");
        }
        writer.finish().expect("finish zip");
    }

    /// The staged layout is what the installer reads back: two executable binaries
    /// under `bin/`, the licence material at the root — the notices included, which the
    /// release's archives carry.
    fn assert_staged(archive: &Path, staging: &Path) {
        extract_into(archive, staging).expect("extract the release archive");
        for tool in [ffmpeg_file(), ffprobe_file()] {
            let path = staging.join("bin").join(tool);
            assert!(crate::util::is_executable(&path), "{tool} is executable");
        }
        for licence in LICENCE_AND_SOURCE.into_iter().chain([COMPONENT_NOTICES]) {
            assert!(
                staging.join(licence).is_file(),
                "{licence} is at the archive's root"
            );
        }
        assert_eq!(
            fs::read(staging.join("LICENCE.txt")).expect("read the licence"),
            b"GPLv3" as &[u8]
        );
        assert_eq!(
            fs::read(staging.join("SOURCE.txt")).expect("read the source note"),
            b"upstream source" as &[u8]
        );
    }

    #[test]
    fn a_tar_gz_release_lands_in_the_staging_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("pkg.tar.gz");
        write_tar_gz(&archive, &release_entries("ffmpeg-v7.1-mahbot.2"));
        assert_staged(&archive, &dir.path().join("staging"));
    }

    #[test]
    fn a_zip_release_lands_in_the_staging_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("pkg.zip");
        write_zip(&archive, &release_entries("ffmpeg-v7.1-mahbot.2"));
        assert_staged(&archive, &dir.path().join("staging"));
    }

    #[test]
    fn a_release_without_the_licence_is_a_failed_install() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("pkg.tar.gz");
        let entries: Vec<(String, Vec<u8>)> = release_entries("ffmpeg-v7.1-mahbot.2")
            .into_iter()
            .filter(|(name, _)| !name.starts_with("LICENCE"))
            .collect();
        write_tar_gz(&archive, &entries);

        let reason = extract_into(&archive, &dir.path().join("staging"))
            .expect_err("a release without its licence text is refused");
        assert!(reason.contains("LICENCE.txt"), "{reason}");
    }

    /// A build from before the component notices existed installs with the material it
    /// carries: a host with no pair at all is worse served than one running that build
    /// until the next start takes a newer one.
    #[test]
    fn a_build_that_predates_the_notices_installs_with_what_it_carries() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("pkg.tar.gz");
        let entries: Vec<(String, Vec<u8>)> = release_entries("ffmpeg-v7.1-mahbot.2")
            .into_iter()
            .filter(|(name, _)| name != COMPONENT_NOTICES)
            .collect();
        write_tar_gz(&archive, &entries);

        let staging = dir.path().join("staging");
        extract_into(&archive, &staging).expect("a build from before the notices installs");
        for licence in LICENCE_AND_SOURCE {
            assert!(staging.join(licence).is_file(), "{licence} is there");
        }
        assert!(
            !staging.join(COMPONENT_NOTICES).exists(),
            "no notices to copy"
        );
        assert!(
            crate::util::is_executable(&staging.join("bin").join(ffmpeg_file())),
            "the pair is staged regardless"
        );
    }

    /// The floors the archives are built against are the product's own, so an archive
    /// that checksums can still start on every machine the product runs on. One fact,
    /// told by both the workflow that builds them and the product's own constants; a
    /// drift offers a file to a system that cannot load it.
    #[test]
    fn the_floors_are_the_product_own() {
        let workflow = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ffmpeg-release.yml"),
        )
        .expect("the workflow that builds the archives");
        let (macos, glibc) = (
            crate::self_update::MACOS_FLOOR,
            crate::self_update::GLIBC_FLOOR,
        );
        assert!(
            workflow.contains(&format!("-mmacosx-version-min={}.{}", macos.0, macos.1)),
            "the macOS archives must be built against {}.{}",
            macos.0,
            macos.1
        );
        assert!(
            workflow.contains(&format!("GLIBC_FLOOR: \"{}.{}\"", glibc.0, glibc.1)),
            "the Linux archives must be built against glibc {}.{}",
            glibc.0,
            glibc.1
        );
        assert!(
            workflow.contains("getconf GNU_LIBC_VERSION"),
            "the Linux legs must refuse a base newer than the floor"
        );
    }

    // ── The stand-in release stream ─────────────────────────────────

    /// This host's archive name for `tag`: the one the installer asks the release
    /// for.
    #[cfg(unix)]
    fn host_asset(tag: &str) -> String {
        let (os, arch) =
            managed_bin::host_os_arch().expect("this host is one the product builds for");
        asset_name(tag, os, arch)
    }

    /// A real archive of [`release_entries`], made once and served as bytes, since the
    /// host answers with what it was handed.
    #[cfg(unix)]
    fn archive_bytes(dir: &Path, tag: &str) -> Vec<u8> {
        let archive = dir.join(host_asset(tag));
        write_tar_gz(&archive, &release_entries(tag));
        fs::read(&archive).expect("read the stand-in archive")
    }

    /// One line of a `SHA256SUMS` body.
    #[cfg(unix)]
    fn checksum_line(digest: &str, name: &str) -> String {
        format!("{digest}  {name}\n")
    }

    /// The lowercase hex SHA-256 of `bytes`: what a `SHA256SUMS` line for an archive
    /// carries.
    #[cfg(unix)]
    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest as _;
        crate::util::hex_string(&sha2::Sha256::digest(bytes))
    }

    /// The staging directories a finished or failed attempt may have left under
    /// `root`: the prefix the attempt's own guard gives them.
    #[cfg(unix)]
    fn staging_dirs(root: &Path) -> Vec<String> {
        entries(root)
            .into_iter()
            .filter(|name| name.starts_with(".staging-"))
            .collect()
    }

    /// Every name under `root`, sorted: what an install left behind, a copy or
    /// otherwise.
    #[cfg(unix)]
    fn entries(root: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(root)
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|entry| entry.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    /// The releases a stand-in stream offers: each tag and the asset names it carries.
    #[cfg(unix)]
    type Offered = Arc<Mutex<Vec<(String, Vec<String>)>>>;

    /// A stand-in for the release stream: the listing at `?per_page=&page=`, and one
    /// release's files at `{base}/{tag}/SHA256SUMS` and `{base}/{tag}/{asset}`, with a
    /// 404 for anything else. Shaped like `FakeTelegram` in
    /// `channels/telegram_tests.rs` — a non-blocking accept loop polling a stop flag,
    /// one thread and one answer per connection — because the installer is a real HTTP
    /// client and no check may ask the real host.
    #[cfg(unix)]
    struct FakeRelease {
        /// The base [`RELEASES_URL_ENV`] is pointed at.
        base: String,
        /// Every release the stream offers: its tag and the asset names it carries.
        offered: Offered,
        /// The bytes each path answers with, replaced between cases.
        served: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        /// Every path asked for, in arrival order.
        asked: Arc<Mutex<Vec<String>>>,
        /// A listing body to answer with instead of the stream's own, for the check that
        /// asks what a host answering with something that is not a release array does.
        listing: Arc<Mutex<Option<Vec<u8>>>>,
        /// Set by `Drop`, so the accept loop — which polls — lets go of the socket.
        stop: Arc<AtomicBool>,
        accept: Option<std::thread::JoinHandle<()>>,
    }

    #[cfg(unix)]
    impl FakeRelease {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind the stand-in host");
            listener
                .set_nonblocking(true)
                .expect("the stand-in host accepts without blocking");
            let port = listener.local_addr().expect("stand-in host address").port();
            let offered = Arc::new(Mutex::new(Vec::new()));
            let served = Arc::new(Mutex::new(HashMap::new()));
            let asked = Arc::new(Mutex::new(Vec::new()));
            let listing = Arc::new(Mutex::new(None));
            let stop = Arc::new(AtomicBool::new(false));
            let accept = std::thread::spawn({
                let offered = Arc::clone(&offered);
                let served = Arc::clone(&served);
                let asked = Arc::clone(&asked);
                let listing = Arc::clone(&listing);
                let stop = Arc::clone(&stop);
                move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                // The accepted socket can inherit the listener's
                                // non-blocking mode (it does on macOS), and a reader
                                // that polls would see a half-arrived request.
                                let _ = stream.set_nonblocking(false);
                                let offered = Arc::clone(&offered);
                                let served = Arc::clone(&served);
                                let asked = Arc::clone(&asked);
                                let listing = Arc::clone(&listing);
                                std::thread::spawn(move || {
                                    serve(stream, &offered, &served, &asked, &listing);
                                });
                            }
                            // Polling rather than blocking, so the loop can see `stop`:
                            // a blocking accept would outlive the check that drops the
                            // host.
                            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                                std::thread::sleep(Duration::from_millis(2));
                            }
                            // A client that went away before it was accepted is
                            // ordinary, and the next connection is unaffected.
                            Err(_) => {}
                        }
                    }
                }
            });
            Self {
                base: format!("http://127.0.0.1:{port}"),
                offered,
                served,
                asked,
                listing,
                stop,
                accept: Some(accept),
            }
        }

        /// Offer `tag` carrying `asset`, with `sums` as its checksum file and `archive`
        /// as the file that asset names. A second call for the same tag replaces its
        /// files and leaves the listing as it was.
        fn offer(&self, tag: &str, asset: &str, sums: &str, archive: &[u8]) {
            let mut offered = self.offered.lock().expect("fake listing");
            if !offered.iter().any(|(name, _)| name == tag) {
                offered.push((tag.to_string(), vec![asset.to_string()]));
            }
            drop(offered);
            self.served.lock().expect("fake release files").extend([
                (format!("/{tag}/SHA256SUMS"), sums.as_bytes().to_vec()),
                (format!("/{tag}/{asset}"), archive.to_vec()),
            ]);
        }

        /// The base the release hook is pointed at, so every attempt in the check asks
        /// this host and never the real one.
        fn base(&self) -> &str {
            &self.base
        }

        /// Every path asked for, in arrival order.
        fn asked(&self) -> Vec<String> {
            self.asked.lock().expect("fake release log").clone()
        }

        /// Answer the listing with `body`, whatever pages are asked for.
        fn listing_is(&self, body: &[u8]) {
            *self.listing.lock().expect("fake listing override") = Some(body.to_vec());
        }
    }

    #[cfg(unix)]
    impl Drop for FakeRelease {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(accept) = self.accept.take() {
                let _ = accept.join();
            }
        }
    }

    /// Answer one request: the listing, the bytes registered for its path, or a 404.
    #[cfg(unix)]
    fn serve(
        mut stream: TcpStream,
        offered: &Mutex<Vec<(String, Vec<String>)>>,
        served: &Mutex<HashMap<String, Vec<u8>>>,
        asked: &Mutex<Vec<String>>,
        listing: &Mutex<Option<Vec<u8>>>,
    ) {
        let Some(path) = request_path(&mut stream) else {
            return;
        };
        asked.lock().expect("fake release log").push(path.clone());
        let override_body = listing.lock().expect("fake listing override").clone();
        let body = match path.strip_prefix("/?") {
            // A case that wants the host to answer with a body of its own.
            Some(_) if override_body.is_some() => override_body,
            // Page one is the whole stream here; a later page is its end.
            Some(query) if query.contains("page=1") => Some(listing_body(offered)),
            Some(_) => Some(b"[]".to_vec()),
            None => served
                .lock()
                .expect("fake release files")
                .get(&path)
                .cloned(),
        };
        let head = match &body {
            Some(bytes) => format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            ),
            None => "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_string(),
        };
        let _ = stream.write_all(head.as_bytes());
        if let Some(bytes) = body {
            let _ = stream.write_all(&bytes);
        }
    }

    /// The listing, in the shape the GitHub API answers with: one entry per offered
    /// release, each carrying the asset names it was offered under.
    #[cfg(unix)]
    fn listing_body(offered: &Mutex<Vec<(String, Vec<String>)>>) -> Vec<u8> {
        let releases: Vec<serde_json::Value> = offered
            .lock()
            .expect("fake listing")
            .iter()
            .map(|(tag, assets)| {
                serde_json::json!({
                    "tag_name": tag,
                    "assets": assets
                        .iter()
                        .map(|name| serde_json::json!({ "name": name }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        serde_json::Value::Array(releases).to_string().into_bytes()
    }

    /// The target of the one request on this connection: the request line, read up to
    /// the blank line that ends the head. Every request here is a bodyless GET.
    #[cfg(unix)]
    fn request_path(stream: &mut TcpStream) -> Option<String> {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            let read = stream.read(&mut chunk).ok()?;
            if read == 0 {
                return None;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        let head = String::from_utf8_lossy(&buffer);
        Some(head.lines().next()?.split_whitespace().nth(1)?.to_string())
    }

    /// A throwaway install root — `$HOME/.mahbot/ffmpeg`, the directory the product
    /// resolves — with `HOME` relocated for as long as the guard lives.
    #[cfg(unix)]
    fn temp_root() -> (tempfile::TempDir, crate::util::test::EnvVarGuard, PathBuf) {
        let home = tempfile::tempdir().expect("temp home");
        let guard = crate::util::test::set_env_var("HOME", home.path().to_str());
        let root = root().expect("the install root resolves under the temp home");
        (home, guard, root)
    }

    /// The boot task against a stream that offers four builds: the newest one this
    /// host has an archive for is installed, a following start has nothing left to
    /// download, and a build published after that start is picked up by the next one.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn the_boot_installs_the_newest_build_the_stream_offers() {
        let files = tempfile::tempdir().expect("stand-in release files");
        let (_home, _lock, root) = temp_root();
        let host = FakeRelease::start();
        for tag in [
            "ffmpeg-v7.1-mahbot.2",
            "ffmpeg-v7.0-mahbot.9",
            "ffmpeg-v7.1-mahbot.1",
        ] {
            let asset = host_asset(tag);
            let archive = archive_bytes(files.path(), tag);
            let sums = checksum_line(&sha256_hex(&archive), &asset);
            host.offer(tag, &asset, &sums, &archive);
        }
        // Newer than every build above, and carrying this host no archive at all.
        host.offer(
            "ffmpeg-v9.0-mahbot.1",
            &asset_name("ffmpeg-v9.0-mahbot.1", "windows", "aarch64"),
            "",
            b"",
        );
        let _hook = ReleasesHook::at(host.base());

        assert_eq!(
            update()
                .await
                .expect("the newest build installs")
                .as_deref(),
            Some("ffmpeg-v7.1-mahbot.2")
        );
        assert_eq!(
            chosen(&root).expect("a copy is in use").0.tag,
            "ffmpeg-v7.1-mahbot.2",
            "the newest build the stream carries for this host is the one in use"
        );
        assert_eq!(
            entries(&root),
            ["ffmpeg-v7.1-mahbot.2"],
            "one copy: nothing older, no download and no staging left"
        );
        for licence in LICENCE_AND_SOURCE.into_iter().chain([COMPONENT_NOTICES]) {
            assert!(
                root.join("ffmpeg-v7.1-mahbot.2").join(licence).is_file(),
                "{licence} travelled with the pair"
            );
        }
        assert_eq!(
            host.asked().first().map(String::as_str),
            Some("/?per_page=100&page=1"),
            "the listing is what the boot starts from"
        );

        // A following start finds the copy already the newest: the listing is read and
        // nothing is downloaded.
        let asked = host.asked().len();
        assert_eq!(update().await.expect("nothing to do"), None);
        assert_eq!(
            host.asked().len(),
            asked + 1,
            "only the listing is read again"
        );

        // A build published after this product binary was is picked up by a later
        // start, and the build it replaces is pruned.
        let newer = "ffmpeg-v7.1-mahbot.3";
        let asset = host_asset(newer);
        let archive = archive_bytes(files.path(), newer);
        let sums = checksum_line(&sha256_hex(&archive), &asset);
        host.offer(newer, &asset, &sums, &archive);
        assert_eq!(
            update().await.expect("the newer build installs").as_deref(),
            Some(newer)
        );
        assert_eq!(entries(&root), [newer], "the build it replaced is gone");
    }

    /// A pair damaged after it was placed — executable files under the right tag, but no
    /// longer the build they are named after — is replaced at the next start, without a
    /// newer build having to appear for it.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn a_damaged_copy_is_reinstalled_at_the_next_start() {
        let files = tempfile::tempdir().expect("stand-in release files");
        let (_home, _lock, root) = temp_root();
        let host = FakeRelease::start();
        let tag = "ffmpeg-v7.1-mahbot.2";
        let asset = host_asset(tag);
        let archive = archive_bytes(files.path(), tag);
        let sums = checksum_line(&sha256_hex(&archive), &asset);
        host.offer(tag, &asset, &sums, &archive);
        let _hook = ReleasesHook::at(host.base());
        update().await.expect("the build installs");

        // A truncation or a quarantine leaves exactly this: the directory parses and the
        // files are executable, but one of them is not the build anymore.
        let damaged = root.join(tag).join("bin").join(ffmpeg_file());
        fs::write(
            &damaged,
            b"#!/bin/sh\necho 'ffmpeg version 0.0.0 damaged'\n",
        )
        .expect("damage the copy");
        managed_bin::set_executable(&damaged).expect("keep it executable");

        assert_eq!(
            update()
                .await
                .expect("the damaged copy is replaced")
                .as_deref(),
            Some(tag),
            "the same build is installed again"
        );
        assert_eq!(
            update().await.expect("nothing to do").as_deref(),
            None,
            "the replacement proves itself"
        );
    }

    /// A copy whose tag outranks the stream's newest — a withdrawn build, or one that no
    /// longer proves itself — is replaced by the build the stream offers and removed, so a
    /// start converges on one copy instead of downloading the same archive forever.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn a_copy_newer_than_the_stream_is_replaced_once() {
        let files = tempfile::tempdir().expect("stand-in release files");
        let (_home, _lock, root) = temp_root();
        let host = FakeRelease::start();
        let tag = "ffmpeg-v7.1-mahbot.2";
        let asset = host_asset(tag);
        let archive = archive_bytes(files.path(), tag);
        let sums = checksum_line(&sha256_hex(&archive), &asset);
        host.offer(tag, &asset, &sums, &archive);
        let _hook = ReleasesHook::at(host.base());

        install_fake(&root, "ffmpeg-v7.1-mahbot.9");

        assert_eq!(
            update()
                .await
                .expect("the build the stream offers installs")
                .as_deref(),
            Some(tag)
        );
        assert_eq!(
            entries(&root),
            [tag],
            "the copy the stream no longer offers is gone"
        );

        let asked = host.asked().len();
        assert_eq!(
            update().await.expect("nothing to do"),
            None,
            "the following start converges on the one copy"
        );
        assert_eq!(
            host.asked().len(),
            asked + 1,
            "only the listing is read again"
        );
    }

    /// A stream that cannot be read leaves a copy that proves itself alone: nothing is
    /// installed, nothing is reported, and the copy stays in use.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn an_unreadable_stream_leaves_a_proven_copy_alone() {
        let files = tempfile::tempdir().expect("stand-in release files");
        let (_home, _lock, root) = temp_root();
        let host = FakeRelease::start();
        let tag = "ffmpeg-v7.1-mahbot.2";
        let asset = host_asset(tag);
        let archive = archive_bytes(files.path(), tag);
        let sums = checksum_line(&sha256_hex(&archive), &asset);
        host.offer(tag, &asset, &sums, &archive);
        {
            let _hook = ReleasesHook::at(host.base());
            update().await.expect("the build installs");
        }

        let _hook = ReleasesHook::at("http://127.0.0.1:1");
        assert_eq!(
            update()
                .await
                .expect("a copy that proves itself needs no listing"),
            None
        );
        assert_eq!(entries(&root), [tag], "the copy stays in use");
    }

    /// A copy in place is what the operations run, and a stream that cannot be reached
    /// is never asked about it.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn a_copy_in_place_is_used_without_asking_the_stream() {
        let (_home, _lock, root) = temp_root();
        install_fake(&root, "ffmpeg-v7.1-mahbot.1");
        let _hook = ReleasesHook::at("http://127.0.0.1:1");

        let binaries = ensure_installed().await.expect("the copy in place is used");
        assert_eq!(
            binaries.ffmpeg,
            root.join("ffmpeg-v7.1-mahbot.1")
                .join("bin")
                .join(ffmpeg_file())
        );
    }

    /// A stream with no build for this host is a reason, never another platform's
    /// archive.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn no_build_for_this_host_is_a_reason() {
        let host = FakeRelease::start();
        host.offer(
            "ffmpeg-v7.1-mahbot.2",
            &asset_name("ffmpeg-v7.1-mahbot.2", "windows", "aarch64"),
            "",
            b"",
        );
        let _lock = env_held();
        let _hook = ReleasesHook::at(host.base());
        let Err(reason) = newest().await else {
            panic!("this host has no build in the stream");
        };
        let (os, arch) = managed_bin::host_os_arch().expect("a supported host");
        assert!(
            reason.contains(&format!("no ffmpeg build is published for {os}-{arch}")),
            "{reason}"
        );
    }

    /// A host answering the listing with a JSON body that is not a release array — the API's
    /// own rate-limit message, a proxy's answer — is a listing that could not be read, never
    /// the claim that this platform has no published build.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn a_listing_that_is_not_one_is_a_read_failure() {
        let host = FakeRelease::start();
        host.listing_is(br#"{"message":"API rate limit exceeded"}"#);
        let _lock = env_held();
        let _hook = ReleasesHook::at(host.base());
        let Err(reason) = newest().await else {
            panic!("a body that is not a release array names no build");
        };
        assert!(
            reason.contains("could not be read: it is not a JSON array"),
            "{reason}"
        );
    }

    /// The whole install path against a host standing in for the release stream: a
    /// release whose checksum file carries this platform's line lands as the copy in
    /// use, and every way a release can lie — no line for this platform's archive, bytes
    /// that are not the declared ones — installs nothing and leaves the working copy in
    /// place.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn the_install_checks_the_release_against_its_own_checksums() {
        let dir = tempfile::tempdir().expect("install root");
        let root = dir.path();
        let files = tempfile::tempdir().expect("stand-in release files");
        let tag = "ffmpeg-v7.1-mahbot.2";
        let asset = host_asset(tag);
        let archive = archive_bytes(files.path(), tag);
        let sums = checksum_line(&sha256_hex(&archive), &asset);

        // 1. The release's checksum file names this platform's archive: the whole path
        //    runs, and the copy it leaves is the tag's own with nothing of the attempt
        //    beside it.
        let host = FakeRelease::start();
        host.offer(tag, &asset, &sums, &archive);
        let _lock = env_held();
        let _hook = ReleasesHook::at(host.base());
        install(root, &build(tag))
            .await
            .expect("the release installs");
        assert_eq!(
            host.asked(),
            [format!("/{tag}/SHA256SUMS"), format!("/{tag}/{asset}")],
            "the checksum file and this platform's archive, both under the tag"
        );
        assert_eq!(
            chosen(root).expect("a copy is in use").0.tag,
            tag,
            "the copy in use is the one installed"
        );
        for tool in [ffmpeg_file(), ffprobe_file()] {
            assert!(
                crate::util::is_executable(&root.join(tag).join("bin").join(tool)),
                "{tool} is installed executable"
            );
        }
        assert_eq!(
            entries(root),
            [tag.to_string()],
            "the whole install is one directory: no download and no staging left"
        );

        // 2. The checksum file without a line for this platform's archive: no other
        //    platform's archive is ever substituted.
        let elsewhere = checksum_line(
            &sha256_hex(&archive),
            &asset_name(tag, "windows", "aarch64"),
        );
        host.offer(tag, &asset, &elsewhere, &archive);
        let reason = install(root, &build(tag))
            .await
            .expect_err("a release without this platform's archive is refused");
        assert!(reason.contains("no checksum for this platform"), "{reason}");
        assert_eq!(
            chosen(root).expect("the installed copy survives").0.tag,
            tag,
            "the copy in use is untouched"
        );
        assert!(
            !root.join("downloads").exists(),
            "nothing was downloaded for a release without this archive"
        );
        assert!(
            staging_dirs(root).is_empty(),
            "nothing was staged for a release without this archive"
        );

        // 3. The declared digest, with an archive whose bytes are not the declared ones:
        //    the download is refused and nothing of it is kept.
        let mut altered = archive.clone();
        *altered.last_mut().expect("the archive has bytes") ^= 0xff;
        host.offer(tag, &asset, &sums, &altered);
        let reason = install(root, &build(tag))
            .await
            .expect_err("an archive that does not match its digest is refused");
        assert!(reason.contains("does not match the checksum"), "{reason}");
        assert_eq!(
            chosen(root).expect("the installed copy survives").0.tag,
            tag,
            "the copy in use is untouched"
        );
        assert!(
            !root.join("downloads").join(&asset).exists(),
            "the refused archive is not kept"
        );
        assert!(
            staging_dirs(root).is_empty(),
            "nothing was staged for a rejected download"
        );

        // 4. An unusable copy standing where the tag belongs — one of its binaries is
        //    gone — is replaced in a single swap, so the host is never left without a
        //    copy and no half-installed directory survives.
        host.offer(tag, &asset, &sums, &archive);
        fs::remove_file(root.join(tag).join("bin").join(ffprobe_file()))
            .expect("break the installed copy");
        assert!(
            chosen(root).is_none(),
            "a copy that lost a binary is not a copy"
        );
        install(root, &build(tag))
            .await
            .expect("the release replaces the broken copy");
        for tool in [ffmpeg_file(), ffprobe_file()] {
            assert!(
                crate::util::is_executable(&root.join(tag).join("bin").join(tool)),
                "{tool} is back after the swap"
            );
        }
        for licence in LICENCE_AND_SOURCE {
            assert!(
                root.join(tag).join(licence).is_file(),
                "{licence} survives the swap"
            );
        }
    }
}
