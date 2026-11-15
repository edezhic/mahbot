//! Obtaining and updating the product's managed copy of the vendor's `photocraft-cli`.
//!
//! The archive is this platform's own and its digest comes from the release's `SHA256SUMS.txt` —
//! the ffmpeg installer's trust model, in which a checksum file rewritten with its archive is not
//! detected.
//!
//! The order makes a failure harmless: the checksum line is read before anything is downloaded, the
//! archive is verified as it streams into a staging directory inside the install root, and only a
//! copy that runs and reports the release version is swapped into place, exclusively against the
//! runs reading it. One install runs at a time, and no caller's run is ever held behind it.
//!
//! The install root is `<configuration directory>/photocraft`. A failure is recorded at WARN with a
//! reason naming the failed step and carrying no path and no URL, because it is recorded verbatim.
//! What runs the copy belongs to [`super`].

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use semver::Version;

use crate::util::managed_bin::{self, Release, WhenMissing};

/// The vendor's repository, published under the brand ArtCraft.
const REPO: &str = "storytold/photocraft";

/// Where the release's assets live, and the checksum file the vendor publishes beside them.
const DOWNLOADS: &str = "https://github.com/storytold/photocraft/releases/download";
const CHECKSUMS: &str = "SHA256SUMS.txt";

/// How long a staged binary may take to print its banner: a local file needing longer is a copy
/// that cannot run.
const VERSION_TIMEOUT: Duration = Duration::from_secs(10);

/// The one file the product takes out of the vendor's archive. Windows decides executability by
/// extension ([`crate::util::is_executable`]), so the suffix is part of the name there and
/// nowhere else.
const fn program_file() -> &'static str {
    if cfg!(windows) {
        "photocraft-cli.exe"
    } else {
        "photocraft-cli"
    }
}

/// Holds the one install in flight, so the start-time install and a first use never overlap — and
/// two first uses never download twice.
static INSTALLING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Held shared by every run and exclusively while an install replaces the copy, so a refresh never
/// swaps the file out from under a run still reading it.
static IN_USE: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

/// One subdirectory of the product's configuration directory — never the daemon's swept temp
/// root, never a package-manager directory. A test run's root is the test root, so `cargo test`
/// never installs into a live product.
#[cfg_attr(
    test,
    expect(clippy::unnecessary_wraps, reason = "the test root always resolves")
)]
fn root() -> Result<PathBuf, String> {
    #[cfg(test)]
    let dir = crate::util::test::test_root().join("photocraft");
    #[cfg(not(test))]
    let dir = crate::config::default_config_dir()
        .map_err(|_| "the product's configuration directory is unavailable".to_string())?
        .join("photocraft");
    Ok(dir)
}

/// Whether a copy is in place and executable.
fn installed() -> bool {
    root().is_ok_and(|root| crate::util::is_executable(&root.join(program_file())))
}

/// Hold the managed copy for one run and answer the install root and the program's path. The read
/// guard is taken before the path is read and held for as long as the run lasts, so a refresh
/// cannot replace the copy under it.
///
/// A copy in place is used as it stands, with no proof and no network round-trip: it was proved
/// before it was placed and the start-time task re-proves it, so a copy damaged since keeps
/// serving until the next start. A failed install is not remembered — the machine may have had no
/// network yet — so the next use tries again.
pub(super) async fn hold_for_run()
-> Result<(tokio::sync::RwLockReadGuard<'static, ()>, PathBuf, PathBuf), String> {
    ensure_installed().await.map_err(recorded)?;
    let held = IN_USE.read().await;
    let root = root()?;
    let program = root.join(program_file());
    Ok((held, root, program))
}

/// Install the vendor's newest release when no copy is in place. A copy that is there is left as it
/// stands: the start-time task is what refreshes it.
async fn ensure_installed() -> Result<(), String> {
    if installed() {
        return Ok(());
    }
    let _installing = INSTALLING.lock().await;
    // A caller that waited for the lock may find the copy another one just placed.
    if installed() {
        return Ok(());
    }
    let release = managed_bin::latest_release(REPO, managed_bin::METADATA_TIMEOUT).await?;
    let program = install_locked(&release).await?;
    tracing::info!(
        "photocraft {} installed at {}",
        release.version,
        program.display()
    );
    Ok(())
}

/// Bring an installed copy to the newest release at start-up. A machine that never used the tool is
/// left alone — it is installed on first use instead — and the check happens only after the shared
/// refresh delay. A failure is recorded and retried on the next start, never removing a working
/// copy.
pub async fn install_on_start() {
    managed_bin::install_on_start(
        "photocraft",
        installed,
        installed_version,
        async || managed_bin::latest_release(REPO, managed_bin::METADATA_TIMEOUT).await,
        async |release| {
            let release = managed_bin::release_or_reason(release)?;
            install(&release).await
        },
        WhenMissing::Skip,
    )
    .await;
}

/// The installed copy's version, read from its own banner. `None` when there is no copy or it
/// does not run — the answer the start-time policy takes as "needs a (re)install".
async fn installed_version() -> Option<Version> {
    installed_version_at(&root().ok()?).await
}

/// [`installed_version`] for a root that is already resolved.
async fn installed_version_at(root: &Path) -> Option<Version> {
    let program = root.join(program_file());
    if !crate::util::is_executable(&program) {
        return None;
    }
    reports(root, &program).await
}

/// The version `program` reports when run from `cwd`, `None` when it does not run.
async fn reports(cwd: &Path, program: &Path) -> Option<Version> {
    let outcome = crate::tools::shell::run_internal_program_outcome(
        cwd,
        program,
        &["--version".to_string()],
        &[],
        Some(VERSION_TIMEOUT),
    )
    .await;
    if !outcome.success {
        return None;
    }
    banner_version(&outcome.output)
}

/// The version a banner reports: `photocraft-cli 0.20.1 (b8c3f3aa8, 2026-10-28)` is 0.20.1.
fn banner_version(text: &str) -> Option<Version> {
    text.lines()
        .next()?
        .split_whitespace()
        .find_map(managed_bin::parse_version_token)
}

/// This platform's archive, named the way the vendor names it: macOS ships one universal build
/// for both architectures, and Windows spells its architectures `x64`/`arm64`. `None` for a pair the
/// product does not build for or the vendor publishes nothing for — never another pair's archive.
/// (The vendor also ships FreeBSD and Linux riscv64 builds, which the product itself does not target.)
fn asset_name(version: &str, os: &str, arch: &str) -> Option<String> {
    let name = match (os, arch) {
        ("macos", "x86_64" | "aarch64") => format!("photocraft-cli-{version}-macos-universal.zip"),
        ("linux", "x86_64") => format!("photocraft-{version}-linux-x86_64.tar.gz"),
        ("linux", "aarch64") => format!("photocraft-{version}-linux-aarch64.tar.gz"),
        ("windows", "x86_64") => format!("photocraft-{version}-windows-x64-portable.zip"),
        ("windows", "aarch64") => format!("photocraft-{version}-windows-arm64-portable.zip"),
        _ => return None,
    };
    Some(name)
}

/// Download, verify, prove and place the one file, answering the program's path. The copy that is
/// there is left untouched until the new one has proved itself.
async fn install(release: &Release) -> Result<PathBuf, String> {
    let _installing = INSTALLING.lock().await;
    install_locked(release).await
}

/// [`install`] under the lock its caller already holds: one install runs at a time whoever asks, so
/// a third entry point cannot download into the same install root beside these two.
async fn install_locked(release: &Release) -> Result<PathBuf, String> {
    use crate::util::http::{DownloadSizeCheck, download_verified};

    let root = root()?;
    let program = root.join(program_file());
    // A caller that waited for the lock may be looking at the copy the holder just placed — before
    // anything is prepared for a download.
    if installed_version_at(&root).await.as_ref() == Some(&release.version) {
        return Ok(program);
    }
    let asset = prepare(&root, &release.version)?;
    let tag = &release.tag;

    let sums = managed_bin::fetch_sha256_sums(
        &format!("{DOWNLOADS}/{tag}/{CHECKSUMS}"),
        "the release's checksum file",
    )
    .await?;
    // This platform's line is read before anything is downloaded, so a release without one
    // refuses with no traffic at all.
    let digest = managed_bin::sha256_for(&sums, &asset)
        .ok_or_else(|| "the release carries no checksum for this platform's archive".to_string())?;

    // On the same filesystem as its destination, so the final placement is a rename — and the
    // archive it also holds is removed with it on every path out, a failed extraction included.
    let staging = tempfile::Builder::new()
        .prefix(".staging-")
        .tempdir_in(&root)
        .map_err(|e| format!("the staging directory could not be created ({})", e.kind()))?;
    let archive = staging.path().join(&asset);
    download_verified(
        managed_bin::download_client()?,
        &format!("{DOWNLOADS}/{tag}/{asset}"),
        &archive,
        &digest,
        None,
        DownloadSizeCheck::Exact,
        |_, _| {},
    )
    .await
    .map_err(|e| {
        format!(
            "the photocraft release could not be downloaded: {}",
            managed_bin::download_reason(&e)
        )
    })?;
    let staged = extract(&archive, staging.path())?;

    // The proof no checksum can give: the file is the release's build and it runs here.
    if reports(&root, &staged).await.as_ref() != Some(&release.version) {
        return Err(format!(
            "the downloaded photocraft-cli could not run, or does not report version {}",
            release.version
        ));
    }
    let _swap = managed_bin::swap_when_idle(&IN_USE).await;
    managed_bin::place_extracted(&staged, &program)?;
    Ok(program)
}

/// What an install needs before it can fetch anything: the install root — created first, because a
/// fresh storage root holds the configuration directory alone and both the archive and the staging
/// directory live under this one — and this platform's archive name.
fn prepare(root: &Path, version: &Version) -> Result<String, String> {
    fs::create_dir_all(root).map_err(|e| {
        format!(
            "the product's photocraft directory could not be created ({})",
            e.kind()
        )
    })?;
    let (os, arch) = managed_bin::host_os_arch()?;
    asset_name(&version.to_string(), os, arch)
        .ok_or_else(|| format!("the vendor publishes no photocraft build for {os}-{arch}"))
}

/// The one file out of whichever container this platform's release uses: the macOS and Windows zips
/// carry the program under its bare name, while the Linux tar wraps an FHS tree in a top-level
/// directory and keeps the CLI at `bin/photocraft-cli` — beside a completion script carrying the
/// program's own file name under `share/bash-completion/completions/` — so the tar is matched by the
/// member's path, never by its name alone. The rest of the tree is ignored, which is what the
/// vendor's own integration kit does.
fn extract(archive: &Path, dir: &Path) -> Result<PathBuf, String> {
    if archive.extension().and_then(OsStr::to_str) == Some("zip") {
        managed_bin::extract_single_file_zip(archive, dir, program_file())
    } else {
        let member = format!("bin/{}", program_file());
        managed_bin::extract_member_tar_gz(archive, dir, &member)
    }
}

/// Record a failed install at WARN — the level that survives the INFO retention and shows in the
/// issues view — and pass the reason on. The reason names the failed step and carries no path
/// and no URL, because it is recorded verbatim.
fn recorded(reason: String) -> String {
    tracing::warn!(
        "photocraft could not be installed (retried on the next use): {}",
        crate::util::truncate(&reason, 1024)
    );
    reason
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_platform_pair_gets_the_archive_the_vendor_publishes() {
        let version = "0.20.1";
        // One universal macOS build serves both architectures.
        for arch in ["x86_64", "aarch64"] {
            assert_eq!(
                asset_name(version, "macos", arch).as_deref(),
                Some("photocraft-cli-0.20.1-macos-universal.zip")
            );
        }
        assert_eq!(
            asset_name(version, "linux", "x86_64").as_deref(),
            Some("photocraft-0.20.1-linux-x86_64.tar.gz")
        );
        assert_eq!(
            asset_name(version, "linux", "aarch64").as_deref(),
            Some("photocraft-0.20.1-linux-aarch64.tar.gz")
        );
        // Windows on ARM is its own build, never the x64 one, and the vendor's spelling of both
        // differs from the triple's.
        assert_eq!(
            asset_name(version, "windows", "x86_64").as_deref(),
            Some("photocraft-0.20.1-windows-x64-portable.zip")
        );
        assert_eq!(
            asset_name(version, "windows", "aarch64").as_deref(),
            Some("photocraft-0.20.1-windows-arm64-portable.zip")
        );
    }

    #[test]
    fn a_pair_the_product_does_not_build_for_is_refused() {
        // The vendor's own FreeBSD and riscv64 archives are real, but the product targets neither —
        // and an architecture no release carries is refused too, never given a neighbour's archive.
        assert_eq!(asset_name("0.20.1", "freebsd", "x86_64"), None);
        assert_eq!(asset_name("0.20.1", "linux", "riscv64"), None);
        assert_eq!(asset_name("0.20.1", "linux", "armv7"), None);
        assert_eq!(asset_name("0.20.1", "macos", "powerpc"), None);
        // The triple's spelling is not the vendor's: `x86_64` is supported on Linux and Windows
        // alone, and no other spelling of it is a name to fall back to.
        assert_eq!(asset_name("0.20.1", "windows", "amd64"), None);
    }

    #[test]
    fn an_install_creates_its_root_before_it_fetches_anything() {
        // A fresh storage root holds the configuration directory alone: without this the first
        // install fails on its staging directory, and the start-time task cannot repair it.
        let parent = tempfile::tempdir().expect("a temporary parent");
        let root = parent.path().join("photocraft");
        assert!(!root.exists());
        // Whatever the platform check answers, the root is created first.
        let _ = prepare(&root, &Version::new(0, 20, 1));
        assert!(root.is_dir(), "{} was not created", root.display());
    }

    /// Build a tar.gz in `dir` holding the given (path, contents) entries, in that order.
    fn write_tar_gz(dir: &Path, entries: &[(&str, &[u8])]) -> PathBuf {
        let path = dir.join("photocraft-0.20.1-linux-x86_64.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            fs::File::create(&path).expect("the archive is created"),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(encoder);
        for (name, contents) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            builder
                .append_data(&mut header, name, *contents)
                .expect("the entry is written");
        }
        builder
            .into_inner()
            .expect("the archive is finished")
            .finish()
            .expect("the gzip stream is finished");
        path
    }

    #[test]
    fn the_tar_extraction_takes_the_program_and_not_the_completion_script_named_like_it() {
        // The vendor's Linux tree is an FHS layout wrapped in one top-level directory: the CLI at
        // `bin/photocraft-cli`, and a bash-completion script with the program's own file name under
        // `share/bash-completion/completions/`. Matching by name alone would install the script;
        // matching the bare `bin/<name>` would find nothing in the wrapped tree.
        let dir = tempfile::tempdir().expect("a temporary directory");
        let top = "photocraft-0.20.1-linux-x86_64";
        let decoy = format!("{top}/share/bash-completion/completions/{}", program_file());
        let program = format!("{top}/bin/{}", program_file());
        let archive = write_tar_gz(
            dir.path(),
            &[
                (&decoy, b"complete -W 'convert run info' photocraft-cli"),
                (&program, b"the program"),
            ],
        );
        let out = extract(&archive, dir.path()).expect("the program is extracted");
        assert_eq!(out, dir.path().join(program_file()));
        assert_eq!(
            fs::read(&out).expect("the extracted file is readable"),
            b"the program"
        );
    }

    #[test]
    fn the_banner_names_the_release_version() {
        assert_eq!(
            banner_version("photocraft-cli 0.20.1 (b8c3f3aa8, 2026-10-28)"),
            Some(Version::new(0, 20, 1))
        );
        assert_eq!(banner_version("photocraft-cli\n"), None);
    }
}
