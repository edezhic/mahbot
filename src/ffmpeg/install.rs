//! Install and update the product's own build of ffmpeg and ffprobe.
//!
//! The pair is built and published by the product itself, in a release stream of
//! its own: tags prefixed `ffmpeg-v`. That separate stream is what keeps the
//! product's updater untouched — it reads `releases/latest` and `version.txt`,
//! which never point at one — while the archives stay downloadable by a plain HTTP
//! client with no credentials.
//!
//! Every install targets the one version this build of the product pins: nothing
//! is discovered from a third party, so what runs is what was tested with the call
//! sites in [`super`]. Integrity comes from the release itself, the way it does for
//! the product's other managed binaries: the release's `SHA256SUMS` is fetched, and
//! the archive is checked against the line for that archive. No digest of a release
//! artefact is carried by this build, so a published release installs as it stands —
//! and a checksum file rewritten together with its archive is not detected, which is
//! the price of that.
//!
//! All six archives are real builds — one per shipped target, Windows on ARM
//! included and never the x64 one — so an archive missing from a release is a
//! failed release and never a substitution. The build is LGPL-only, so its licence
//! text travels with the binaries.
//!
//! The copy in use is the newest one that runs, so a failed install never leaves the
//! product with nothing to run. A failure names the step that failed and carries no
//! path and no URL, because it is recorded and shown verbatim.
//!
//! Nothing is written outside the product's own configuration directory — no
//! directory the owner's package manager owns, and no entry on `PATH`.
//!
//! The three video operations that run this copy belong to [`super`].

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::util::UnwrapPoison;
use crate::util::managed_bin;

// ── The pin ─────────────────────────────────────────────────────────

/// The release this build of the product pins: upstream ffmpeg 7.1, mahbot build
/// revision 1, published in the `ffmpeg-v` stream beside the product's own releases.
///
/// One tag for every platform. A repair to the build — a patch, a rebuilt
/// dependency — bumps the revision and publishes a new tag; a published tag is never
/// re-published, because every copy in the field was installed against it.
pub(super) const TAG: &str = "ffmpeg-v7.1-mahbot.1";

/// Test hook: a releases base serving the layout the real one does
/// (`{base}/{tag}/SHA256SUMS`, `{base}/{tag}/{asset}`).
pub(super) const RELEASES_URL_ENV: &str = "MAHBOT_FFMPEG_RELEASES_URL";

/// The tag prefix of this stream, and the marker separating the upstream version
/// from the build revision inside a tag.
const TAG_PREFIX: &str = "ffmpeg-v";
const REVISION_MARKER: &str = "-mahbot.";

/// How long one archive download may take: a pair of static binaries is tens of
/// megabytes, which is the size the shared large-download policy is written for.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(20);

/// How long the small requests may take — the checksum file alone.
const SUMS_TIMEOUT: Duration = Duration::from_secs(20);

/// How long a staged binary may take to print its banner: a local file needing
/// longer than this is a copy that cannot run.
const VERSION_TIMEOUT: Duration = Duration::from_secs(10);

/// The two files that travel with the binaries: the text of the licence the build
/// is distributed under, and where that build's source is. A release without them
/// is a failed install — a licence that does not travel with the binary is not
/// honoured by shipping the binary alone.
const LICENCE_FILES: [&str; 2] = ["LICENCE.txt", "SOURCE.txt"];

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

/// The copy in use under `root`, with the tag it was found under: the pinned tag
/// when both its binaries are executable, else the newest usable one by (upstream
/// version, revision). A directory whose name does not parse, and a directory
/// holding anything but two executable files, are not copies at all.
fn chosen(root: &Path) -> Option<(String, Binaries)> {
    if let Some(binaries) = binaries_at(&root.join(TAG)) {
        return Some((TAG.to_string(), binaries));
    }
    fs::read_dir(root)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let tag = parse_tag(&name)?;
            let binaries = binaries_at(&entry.path())?;
            Some((tag, name, binaries))
        })
        .max_by(|(left, _, _), (right, _, _)| left.cmp(right))
        .map(|(_, name, binaries)| (name, binaries))
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
/// beside each other and a queued caller re-checks instead of downloading the same
/// release again.
static INSTALLING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The copy in use: the pinned pair when this host has one that reports the pinned
/// version, and otherwise whatever installing that leaves.
///
/// The one path both entry points take — the boot task and every video operation —
/// so every attempt is behind [`INSTALLING`] and no step can run beside another.
/// Safe from anywhere: every step before the final rename leaves the host exactly as
/// it was, and a copy already in place is never touched.
///
/// A failed attempt stands for the rest of the process, so no video file pays a
/// network round-trip for it: the next start is what tries again.
pub(super) async fn ensure_installed() -> Result<Binaries, String> {
    let root = root()?;
    if let Some(binaries) = verified_pin(&root).await {
        return Ok(binaries);
    }
    let _installing = INSTALLING.lock().await;
    // The caller that held the lock may have done the work while this one waited.
    if let Some(binaries) = verified_pin(&root).await {
        return Ok(binaries);
    }
    let reason = match attempt_failure() {
        Some(reason) => reason,
        None => match install(&root).await {
            Ok(binaries) => {
                tracing::info!("ffmpeg {TAG} installed");
                return Ok(binaries);
            }
            Err(reason) => {
                *ATTEMPT_FAILURE.lock().unwrap_poison() = Some(reason.clone());
                reason
            }
        },
    };
    fallback(&root, &reason)
}

/// The pair at the pinned path, when it reports the pinned version — the only copy
/// this product calls the pinned build. A pair that is there and reports something
/// else is not one: [`ensure_installed`] installs over it, and [`fallback`] says what
/// is left when that cannot be done.
async fn verified_pin(root: &Path) -> Option<Binaries> {
    let binaries = binaries_at(&root.join(TAG))?;
    verified(&binaries).await.then_some(binaries)
}

/// What the pair at the pinned path reported about itself, asked once per process:
/// only an install by this product changes the answer.
static PINNED_VERIFIED: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

/// Whether both binaries of the pair run and report the pinned upstream version.
async fn verified(binaries: &Binaries) -> bool {
    if let Some(verified) = *PINNED_VERIFIED.lock().unwrap_poison() {
        return verified;
    }
    let Some(pinned) = upstream_parts(TAG) else {
        return false;
    };
    let verified = reports_upstream(&binaries.ffmpeg, pinned).await
        && reports_upstream(&binaries.ffprobe, pinned).await;
    *PINNED_VERIFIED.lock().unwrap_poison() = Some(verified);
    verified
}

/// Why this process's install attempt failed, when one did.
static ATTEMPT_FAILURE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// The recorded reason the pinned release cannot be installed in this process, if it
/// is known.
fn attempt_failure() -> Option<String> {
    ATTEMPT_FAILURE.lock().unwrap_poison().clone()
}

/// What is left once the pin cannot be installed: the newest copy that runs — said
/// once, and never called the pinned one unless it is — or the reason nothing can be
/// run.
fn fallback(root: &Path, reason: &str) -> Result<Binaries, String> {
    match chosen(root) {
        // A copy that runs is what the product works with: one revision behind, or even
        // one that does not report the pinned build, beats having nothing to run.
        Some((tag, binaries)) => {
            if first_time(&tag) {
                if tag == TAG {
                    tracing::warn!(
                        "the copy installed as {TAG} is not the pinned build and could not be \
                         replaced (retried on the next start): {reason}"
                    );
                } else {
                    tracing::warn!(
                        "ffmpeg is running {tag}, not the pinned {TAG}, which could not be \
                         installed (retried on the next start): {reason}"
                    );
                }
            }
            Ok(binaries)
        }
        None => Err(reason.to_string()),
    }
}

/// Whether this state is news: one line per copy in use, not one per video, because a
/// daemon that keeps running it asks about every file it processes.
fn first_time(state: &str) -> bool {
    static REPORTED: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    let mut reported = REPORTED.lock().unwrap_poison();
    if reported.as_deref() == Some(state) {
        return false;
    }
    *reported = Some(state.to_string());
    true
}

/// The boot task: install the pin when this host does not have it, called once by the
/// binary and never failing the boot.
///
/// Nothing is looked up first — the version is compile-time, so the only question is
/// whether this host has it — and there is no retry of its own: a failure is left to
/// the next start, and every use in this process answers from the record.
pub async fn install_on_start() {
    if let Err(reason) = ensure_installed().await {
        tracing::warn!("ffmpeg {TAG} could not be installed (retried on the next start): {reason}");
    }
}

/// Download, verify, prove and place this build's pair, then prune what is older.
///
/// Runs only under [`INSTALLING`] — with one attempt at a time the download
/// directory and the staging beside it are this attempt's own — and returns the
/// pair the placed directory now holds.
///
/// The order is what makes a failure harmless: this platform's line in the release's
/// checksum file is read before anything is downloaded, the archive is verified as it
/// streams, and the pair is proven to run before the directory is renamed into place.
async fn install(root: &Path) -> Result<Binaries, String> {
    use crate::util::http::{DownloadSizeCheck, download_verified};

    let (os, arch) = managed_bin::host_os_arch()?;
    let asset = asset_name(TAG, os, arch);
    let base = releases_url();
    let Some(pinned) = upstream_parts(TAG) else {
        return Err(format!("{TAG} does not name an upstream version"));
    };

    let sums = fetch_sums(&format!("{base}/{TAG}/SHA256SUMS")).await?;
    let digest = archive_digest(&sums, &asset)?;

    let downloads = root.join("downloads");
    let _ = fs::remove_dir_all(&downloads);
    fs::create_dir_all(&downloads)
        .map_err(|e| format!("the download directory could not be created ({})", e.kind()))?;
    let archive = downloads.join(&asset);
    download_verified(
        client()?,
        &format!("{base}/{TAG}/{asset}"),
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
        if !reports_upstream(&staging.path().join("bin").join(tool), pinned).await {
            return Err(format!(
                "the downloaded {name} could not run, or does not report ffmpeg {}.{}",
                pinned.0, pinned.1
            ));
        }
    }
    let installed = root.join(TAG);
    place(staging.path(), &installed)?;
    prune(root);
    // Both banners were just read: the pair in place is the pinned build.
    *PINNED_VERIFIED.lock().unwrap_poison() = Some(true);
    Ok(binaries_in(&installed))
}

/// The releases base this stream's assets live under: the product's own release
/// host, or the test hook's stand-in.
fn releases_url() -> String {
    if let Ok(base) = std::env::var(RELEASES_URL_ENV) {
        return base.trim_end_matches('/').to_string();
    }
    let repo = crate::self_update::RELEASE_REPO
        .trim_start_matches("https://github.com/")
        .trim_end_matches('/');
    format!("https://github.com/{repo}/releases/download")
}

/// The client every request here runs on: the shared large-download client, whose
/// total bound and connect timeout are the policy for an artifact of this size.
/// One client per process — it owns the connection pool.
fn client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: std::sync::OnceLock<Result<reqwest::Client, String>> =
        std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            crate::util::http::build_download_client(DOWNLOAD_TIMEOUT)
                .map_err(|_| "the release client could not be built".to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The release's `SHA256SUMS` body, the file every digest is read out of.
async fn fetch_sums(url: &str) -> Result<String, String> {
    let response = client()?
        .get(url)
        .timeout(SUMS_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            format!(
                "the release's checksum file could not be fetched: {}",
                managed_bin::request_reason(&e)
            )
        })?;
    if !response.status().is_success() {
        return Err(format!(
            "the release's checksum file could not be fetched: HTTP {}",
            response.status()
        ));
    }
    response.text().await.map_err(|e| {
        format!(
            "the release's checksum file could not be read: {}",
            managed_bin::request_reason(&e)
        )
    })
}

/// The digest `asset` must have, read out of the release's `SHA256SUMS` body: the
/// line for this platform's archive. A body without one installs nothing, so no
/// other platform's archive is ever substituted.
fn archive_digest(body: &str, asset: &str) -> Result<String, String> {
    body.lines()
        .filter_map(managed_bin::parse_sha256_sidecar)
        .find(|(_, name)| name == asset)
        .map(|(digest, _)| digest)
        .ok_or_else(|| "the release carries no checksum for this platform's archive".to_string())
}

/// Unpack one release archive into `staging` in one pass: the two binaries under
/// `bin/`, the licence pair at the root — exactly the layout [`binaries_at`] reads.
fn extract_into(archive: &Path, staging: &Path) -> Result<(), String> {
    let bin_dir = staging.join("bin");
    fs::create_dir_all(&bin_dir).map_err(|e| {
        format!(
            "the staged copy's directory could not be created ({})",
            e.kind()
        )
    })?;
    let names = [
        ffmpeg_file(),
        ffprobe_file(),
        LICENCE_FILES[0],
        LICENCE_FILES[1],
    ];
    let paths = extract(archive, staging, &names)?;
    for (name, path) in [ffmpeg_file(), ffprobe_file()].into_iter().zip(&paths) {
        managed_bin::set_executable(path)?;
        fs::rename(path, bin_dir.join(name))
            .map_err(|e| format!("the staged copy could not be arranged ({})", e.kind()))?;
    }
    Ok(())
}

/// Extract `names` out of `archive` into `dir`, whichever container this
/// platform's release uses.
fn extract(archive: &Path, dir: &Path, names: &[&str]) -> Result<Vec<PathBuf>, String> {
    if archive.extension().and_then(OsStr::to_str) == Some("zip") {
        managed_bin::extract_named_files_zip(archive, dir, names)
    } else {
        managed_bin::extract_named_files_tar_gz(archive, dir, names)
    }
}

/// The `major.minor` `tag` pins, which is all a banner spells out — and `None` for a
/// tag that does not parse, so an unreadable expectation verifies nothing.
fn upstream_parts(tag: &str) -> Option<(u64, u64)> {
    parse_tag(tag).map(|(version, _)| (version.major, version.minor))
}

/// The `major.minor` the first line of a `-version` banner claims: the product's
/// own build reports a plain `7.1`, a vendor's `7.1-static` reads as 7.1 too, and
/// both are the upstream version the pin is about.
fn banner_version(stdout: &[u8]) -> Option<(u64, u64)> {
    let text = String::from_utf8_lossy(stdout);
    let line = text.lines().next()?;
    let token = line
        .split_whitespace()
        .skip_while(|word| *word != "version")
        .nth(1)?;
    let version = token.split_once('-').map_or(token, |(head, _)| head);
    let version = managed_bin::parse_short_version(version)?;
    Some((version.major, version.minor))
}

/// Whether `bin` runs, exits clean and reports `pinned` — the proof no checksum can
/// give, that the file is the expected build and that it can execute on this host.
async fn reports_upstream(bin: &Path, pinned: (u64, u64)) -> bool {
    let Ok(output) = super::run_tool(bin, &["-version".into()], VERSION_TIMEOUT).await else {
        return false;
    };
    output.success && banner_version(&output.stdout) == Some(pinned)
}

/// Move `staging` to `final_dir` in one step, so a half-installed directory can
/// never stand where a whole one did: an existing directory — unusable, since
/// [`chosen`] did not pick it — is moved aside by the crate's own swap, which also
/// puts it back when the swap itself fails.
fn place(staging: &Path, final_dir: &Path) -> Result<(), String> {
    if !final_dir.exists() {
        return fs::rename(staging, final_dir)
            .map_err(|e| format!("the new copy could not be moved into place ({})", e.kind()));
    }
    managed_bin::rename_aside_swap(staging, final_dir)
}

/// Leave nothing this install made and no longer needs: a tag directory strictly
/// older than the pin — never the pinned one, never a newer one — plus every
/// leftover that is not a copy at all (an attempt's staging, its aside, the
/// download, a name from another layout).
///
/// Infallible in effect: each removal is best effort, so a file another process is
/// using can never fail an install that already succeeded.
fn prune(root: &Path) {
    let Some(pinned) = parse_tag(TAG) else {
        return;
    };
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        // A name that parses is a copy, and the pin or anything newer stays.
        if parse_tag(name).is_some_and(|tag| tag >= pinned) {
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
            Arc, LazyLock, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use super::*;

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
        assert_eq!(
            asset_name(TAG, "macos", "aarch64"),
            "ffmpeg-7.1-mahbot.1-macos-aarch64.tar.gz"
        );
        assert_eq!(
            asset_name(TAG, "macos", "x86_64"),
            "ffmpeg-7.1-mahbot.1-macos-x86_64.tar.gz"
        );
        assert_eq!(
            asset_name(TAG, "linux", "aarch64"),
            "ffmpeg-7.1-mahbot.1-linux-aarch64.tar.gz"
        );
        assert_eq!(
            asset_name(TAG, "linux", "x86_64"),
            "ffmpeg-7.1-mahbot.1-linux-x86_64.tar.gz"
        );
        assert_eq!(
            asset_name(TAG, "windows", "x86_64"),
            "ffmpeg-7.1-mahbot.1-windows-x86_64.zip"
        );
        // Windows on ARM is its own build, never the x64 one.
        assert_eq!(
            asset_name(TAG, "windows", "aarch64"),
            "ffmpeg-7.1-mahbot.1-windows-aarch64.zip"
        );
    }

    #[test]
    fn the_pinned_copy_wins_and_an_absent_pin_falls_back_to_the_newest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        install_fake(root, "ffmpeg-v6.0-mahbot.1");
        install_fake(root, "ffmpeg-v7.1-mahbot.0");
        install_fake(root, "ffmpeg-v7.2-mahbot.1");
        fs::create_dir_all(root.join(".staging-Kd7x2p")).expect("create staging");
        fs::create_dir(root.join("not-a-tag")).expect("create junk directory");

        // No pinned copy yet: the newest usable one, by (upstream, revision).
        let (tag, binaries) = chosen(root).expect("a copy is in use");
        assert_eq!(tag, "ffmpeg-v7.2-mahbot.1");
        assert_eq!(
            binaries.ffmpeg,
            root.join("ffmpeg-v7.2-mahbot.1")
                .join("bin")
                .join(ffmpeg_file())
        );

        // The pin appears: it is used although a newer build is installed.
        let pinned = install_fake(root, TAG);
        let (tag, binaries) = chosen(root).expect("a copy is in use");
        assert_eq!(tag, TAG);
        assert_eq!(binaries.ffprobe, pinned.join("bin").join(ffprobe_file()));

        // A tagged directory holding one binary is not a copy.
        fs::remove_file(pinned.join("bin").join(ffprobe_file())).expect("remove ffprobe");
        assert_eq!(
            chosen(root).expect("a copy is in use").0,
            "ffmpeg-v7.2-mahbot.1"
        );

        // A newer upstream wins over a newer revision of an older one.
        fs::remove_dir_all(root.join("ffmpeg-v7.2-mahbot.1")).expect("remove newer");
        fs::remove_dir_all(&pinned).expect("remove pinned");
        assert_eq!(
            chosen(root).expect("a copy is in use").0,
            "ffmpeg-v7.1-mahbot.0"
        );
    }

    #[test]
    fn pruning_keeps_the_pin_and_anything_newer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        for tag in [
            "ffmpeg-v6.0-mahbot.1",
            "ffmpeg-v7.1-mahbot.0",
            TAG,
            "ffmpeg-v7.2-mahbot.1",
        ] {
            install_fake(root, tag);
        }
        fs::create_dir_all(root.join(".staging-Kd7x2p")).expect("create staging");
        fs::create_dir_all(root.join("downloads")).expect("create downloads");

        prune(root);

        assert!(root.join(TAG).exists(), "the pin stays");
        assert!(
            root.join("ffmpeg-v7.2-mahbot.1").exists(),
            "a newer copy stays"
        );
        for gone in [
            "ffmpeg-v6.0-mahbot.1",
            "ffmpeg-v7.1-mahbot.0",
            ".staging-Kd7x2p",
            "downloads",
        ] {
            assert!(!root.join(gone).exists(), "{gone} is gone");
        }
    }

    #[test]
    fn the_sums_body_must_carry_this_archive() {
        let archive = "ffmpeg-7.1-mahbot.1-macos-aarch64.tar.gz";
        let digest = "0123456789abcdef".repeat(4);
        let body = format!("{digest}  {archive}\n{}  LICENCE.txt\n", "f".repeat(64));

        assert_eq!(
            archive_digest(&body, archive).expect("this archive's line"),
            digest
        );

        // The body without this platform's line is refused: no other archive stands in.
        let reason = archive_digest(&body, "ffmpeg-7.1-mahbot.1-windows-aarch64.zip")
            .expect_err("a body without this archive is refused");
        assert!(reason.contains("no checksum for this platform"), "{reason}");
    }

    #[test]
    fn the_banner_names_the_upstream_version() {
        assert_eq!(
            upstream_parts(TAG),
            Some((7, 1)),
            "the pinned tag names the version every banner is compared against"
        );
        // What the release's own build prints: upstream's banner carries the
        // upstream version and nothing of the product's revision.
        assert_eq!(
            banner_version(b"ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers\n"),
            Some((7, 1))
        );
        // A vendor's suffix is not part of the version.
        assert_eq!(
            banner_version(b"ffprobe version 7.1-static Copyright (c) 2000-2024\n"),
            Some((7, 1))
        );
        assert_eq!(
            banner_version(b"ffmpeg version 6.0 Copyright\n"),
            Some((6, 0))
        );
        assert_eq!(banner_version(b"nothing to see\n"), None);
        assert_eq!(banner_version(b""), None);
    }

    /// The four entries a release archive carries: the pair under `bin/` inside a
    /// vendor directory, the licence pair at the root.
    fn release_entries() -> Vec<(String, &'static [u8])> {
        vec![
            (
                format!("pkg/bin/{}", ffmpeg_file()),
                b"FFMPEG" as &'static [u8],
            ),
            (
                format!("pkg/bin/{}", ffprobe_file()),
                b"FFPROBE" as &'static [u8],
            ),
            ("LICENCE.txt".to_string(), b"LGPL"),
            ("SOURCE.txt".to_string(), b"upstream source"),
        ]
    }

    fn write_tar_gz(path: &Path, entries: &[(String, &'static [u8])]) {
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
                .append_data(&mut header, name.as_str(), *contents)
                .expect("append entry");
        }
        builder
            .into_inner()
            .expect("finish archive")
            .finish()
            .expect("finish gzip");
    }

    fn write_zip(path: &Path, entries: &[(String, &'static [u8])]) {
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

    /// The staged layout is what the installer reads back: two executable
    /// binaries under `bin/`, the licence pair at the root.
    fn assert_staged(archive: &Path, staging: &Path) {
        extract_into(archive, staging).expect("extract the release archive");
        for tool in [ffmpeg_file(), ffprobe_file()] {
            let path = staging.join("bin").join(tool);
            assert!(crate::util::is_executable(&path), "{tool} is executable");
        }
        assert_eq!(
            fs::read(staging.join("LICENCE.txt")).expect("read the licence"),
            b"LGPL" as &[u8]
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
        write_tar_gz(&archive, &release_entries());
        assert_staged(&archive, &dir.path().join("staging"));
    }

    #[test]
    fn a_zip_release_lands_in_the_staging_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("pkg.zip");
        write_zip(&archive, &release_entries());
        assert_staged(&archive, &dir.path().join("staging"));
    }

    #[test]
    fn a_release_without_the_licence_is_a_failed_install() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("pkg.tar.gz");
        let entries: Vec<(String, &'static [u8])> = release_entries()
            .into_iter()
            .filter(|(name, _)| !name.starts_with("LICENCE"))
            .collect();
        write_tar_gz(&archive, &entries);

        let reason = extract_into(&archive, &dir.path().join("staging"))
            .expect_err("a release without its licence text is refused");
        assert!(reason.contains("LICENCE.txt"), "{reason}");
    }

    /// The floors the archives are built against are the product's own, so an
    /// archive that checksums can still start on every machine the product runs on.
    /// One fact, told by both the workflow that builds them and the product's own
    /// constants; a drift offers a file to a system that cannot load it.
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

    // ── The stand-in release host ───────────────────────────────────

    /// The banner a stand-in "binary" prints: a shell script, because the product's
    /// own copies are real executables built per platform and a check has none of
    /// them. It reports the pinned version in the shape the release's own build
    /// does, which is what [`reports_upstream`] reads — and the only thing that
    /// proves the file ran.
    #[cfg(unix)]
    static SCRIPT: LazyLock<Vec<u8>> = LazyLock::new(|| {
        let (major, minor) = upstream_parts(TAG).expect("the pinned tag names a version");
        format!("#!/bin/sh\necho 'ffmpeg version {major}.{minor} stand-in build'\n").into_bytes()
    });

    /// The entries a stand-in release carries: the same layout [`extract_into`]
    /// reads, with scripts where the real binaries go.
    #[cfg(unix)]
    fn scripted_entries() -> Vec<(String, &'static [u8])> {
        let script = &SCRIPT[..];
        vec![
            (format!("pkg/bin/{}", ffmpeg_file()), script),
            (format!("pkg/bin/{}", ffprobe_file()), script),
            ("LICENCE.txt".to_string(), b"LGPL" as &'static [u8]),
            (
                "SOURCE.txt".to_string(),
                b"upstream source" as &'static [u8],
            ),
        ]
    }

    /// This host's archive name: the one the installer asks the release for.
    #[cfg(unix)]
    fn host_asset() -> String {
        let (os, arch) =
            managed_bin::host_os_arch().expect("this host is one the product builds for");
        asset_name(TAG, os, arch)
    }

    /// A real archive of [`scripted_entries`], made once and served as bytes, since
    /// the host answers with what it was handed.
    #[cfg(unix)]
    fn archive_bytes(dir: &Path) -> Vec<u8> {
        let archive = dir.join(host_asset());
        write_tar_gz(&archive, &scripted_entries());
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

    /// Every name under `root`, sorted: what an install left behind, a copy or a
    /// leftover.
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

    /// A stand-in for the release host, serving one release: the checksum file and
    /// this platform's archive at the two paths the installer asks for, and a 404
    /// for anything else. Shaped like `FakeTelegram` in `channels/telegram_tests.rs`
    /// — a non-blocking accept loop polling a stop flag, one thread and one answer
    /// per connection — because the installer is a real HTTP client and no check
    /// may ask the real host.
    #[cfg(unix)]
    struct FakeRelease {
        /// The base [`RELEASES_URL_ENV`] is pointed at.
        base: String,
        tag: String,
        asset: String,
        /// The bytes each path answers with, replaced between cases.
        served: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        /// Every path asked for, in arrival order.
        asked: Arc<Mutex<Vec<String>>>,
        /// Set by `Drop`, so the accept loop — which polls — lets go of the socket.
        stop: Arc<AtomicBool>,
        accept: Option<std::thread::JoinHandle<()>>,
    }

    #[cfg(unix)]
    impl FakeRelease {
        /// Start a host offering the release `tag` names: `sums` as the body of its
        /// `SHA256SUMS` and `archive` as this platform's asset.
        fn start(tag: &str, asset: &str, sums: &str, archive: &[u8]) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind the stand-in host");
            listener
                .set_nonblocking(true)
                .expect("the stand-in host accepts without blocking");
            let port = listener.local_addr().expect("stand-in host address").port();
            let served = Arc::new(Mutex::new(HashMap::new()));
            let asked = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let accept = std::thread::spawn({
                let served = Arc::clone(&served);
                let asked = Arc::clone(&asked);
                let stop = Arc::clone(&stop);
                move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                // The accepted socket can inherit the listener's
                                // non-blocking mode (it does on macOS), and a
                                // reader that polls would see a half-arrived
                                // request.
                                let _ = stream.set_nonblocking(false);
                                let served = Arc::clone(&served);
                                let asked = Arc::clone(&asked);
                                std::thread::spawn(move || serve(stream, &served, &asked));
                            }
                            // Polling rather than blocking, so the loop can see
                            // `stop`: a blocking accept would outlive the check
                            // that drops the host.
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
            let host = Self {
                base: format!("http://127.0.0.1:{port}"),
                tag: tag.to_string(),
                asset: asset.to_string(),
                served,
                asked,
                stop,
                accept: Some(accept),
            };
            host.offer(sums, archive);
            host
        }

        /// Offer this release from now on: one case's bytes, then the next case's.
        /// The archive is served under the path its name spells either way, so a
        /// case that changes only the checksum file changes only what a digest
        /// says.
        fn offer(&self, sums: &str, archive: &[u8]) {
            *self.served.lock().expect("fake release files") = HashMap::from([
                (
                    format!("/{}/SHA256SUMS", self.tag),
                    sums.as_bytes().to_vec(),
                ),
                (format!("/{}/{}", self.tag, self.asset), archive.to_vec()),
            ]);
        }

        /// The base the release hook is pointed at, so every attempt in the check
        /// asks this host and never the real one.
        fn base(&self) -> &str {
            &self.base
        }

        /// Every path asked for, in arrival order.
        fn asked(&self) -> Vec<String> {
            self.asked.lock().expect("fake release log").clone()
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

    /// Answer one request: the bytes registered for its path, or a 404.
    #[cfg(unix)]
    fn serve(
        mut stream: TcpStream,
        served: &Mutex<HashMap<String, Vec<u8>>>,
        asked: &Mutex<Vec<String>>,
    ) {
        let Some(path) = request_path(&mut stream) else {
            return;
        };
        asked.lock().expect("fake release log").push(path.clone());
        let body = served
            .lock()
            .expect("fake release files")
            .get(&path)
            .cloned();
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

    /// The target of the one request on this connection: the request line, read up
    /// to the blank line that ends the head. Every request here is a bodyless GET.
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

    /// The whole install path against a host standing in for the release stream: a
    /// release whose checksum file carries this platform's line installs and becomes
    /// the copy in use, and every way a release can lie — no line for this platform's
    /// archive, bytes that are not the declared ones — installs nothing and leaves the
    /// working copy in place.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)] // points the release hook at a local host
    async fn the_install_checks_the_release_against_its_own_checksums() {
        let dir = tempfile::tempdir().expect("install root");
        let root = dir.path();
        let files = tempfile::tempdir().expect("stand-in release files");
        let asset = host_asset();
        let archive = archive_bytes(files.path());
        let sums = checksum_line(&sha256_hex(&archive), &asset);

        // 1. The release's checksum file names this platform's archive: the whole
        //    path runs, and the copy it leaves is under the pinned tag with nothing
        //    of the attempt beside it.
        let host = FakeRelease::start(TAG, &asset, &sums, &archive);
        // The hook is read on every attempt, so no real host is asked for the rest
        // of the check; the guard puts the variable back when it drops.
        let _hook = crate::util::test::set_env_var(RELEASES_URL_ENV, Some(host.base()));
        install(root).await.expect("the release installs");
        assert_eq!(
            host.asked(),
            [format!("/{TAG}/SHA256SUMS"), format!("/{TAG}/{asset}")],
            "the checksum file and this platform's archive, both under the pin"
        );
        assert_eq!(
            chosen(root).expect("a copy is in use").0,
            TAG,
            "the copy in use is the pinned one"
        );
        for tool in [ffmpeg_file(), ffprobe_file()] {
            assert!(
                crate::util::is_executable(&root.join(TAG).join("bin").join(tool)),
                "{tool} is installed executable"
            );
        }
        for licence in LICENCE_FILES {
            assert!(
                root.join(TAG).join(licence).is_file(),
                "{licence} travelled with the pair"
            );
        }
        assert_eq!(
            entries(root),
            [TAG.to_string()],
            "the whole install is one directory: no download and no staging left"
        );

        // 2. The checksum file without a line for this platform's archive: no other
        //    platform's archive is ever substituted.
        let elsewhere = checksum_line(
            &sha256_hex(&archive),
            &asset_name(TAG, "windows", "aarch64"),
        );
        host.offer(&elsewhere, &archive);
        let reason = install(root)
            .await
            .expect_err("a release without this platform's archive is refused");
        assert!(reason.contains("no checksum for this platform"), "{reason}");
        assert_eq!(
            chosen(root).expect("the installed copy survives").0,
            TAG,
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

        // 3. The declared digest, with an archive whose bytes are not the declared
        //    ones: the download is refused and nothing of it is kept.
        let mut altered = archive.clone();
        *altered.last_mut().expect("the archive has bytes") ^= 0xff;
        host.offer(&sums, &altered);
        let reason = install(root)
            .await
            .expect_err("an archive that does not match its digest is refused");
        assert!(reason.contains("does not match the checksum"), "{reason}");
        assert_eq!(
            chosen(root).expect("the installed copy survives").0,
            TAG,
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

        // 4. An unusable copy standing where the pin belongs — one of its binaries is
        //    gone — is replaced by the pinned one in a single swap, so the host is
        //    never left without a copy and no half-installed directory survives.
        host.offer(&sums, &archive);
        fs::remove_file(root.join(TAG).join("bin").join(ffprobe_file()))
            .expect("break the installed copy");
        assert!(
            chosen(root).is_none(),
            "a copy that lost a binary is not a copy"
        );
        install(root)
            .await
            .expect("the release replaces the broken copy");
        for tool in [ffmpeg_file(), ffprobe_file()] {
            assert!(
                crate::util::is_executable(&root.join(TAG).join("bin").join(tool)),
                "{tool} is back after the swap"
            );
        }
        for licence in LICENCE_FILES {
            assert!(
                root.join(TAG).join(licence).is_file(),
                "{licence} survives the swap"
            );
        }
        assert_eq!(
            entries(root),
            [TAG.to_string()],
            "the swap leaves nothing beside the copy"
        );
    }
}
