//! The Chrome for Testing headless shell HyperFrames renders in.
//!
//! The vendor renders by seeking a timeline in a real browser, and the product
//! gives it one of its own rather than whatever the host happens to have: the
//! `chrome-headless-shell` build Google publishes for this platform, downloaded
//! once into the product's configuration directory and reused from then on. A
//! usable copy means no network at all.
//!
//! Google publishes no checksum for these archives, so nothing about the download
//! is verified by digest. The proof is the binary itself: it is extracted into a
//! scratch directory inside the cache and run with `--version` before it is
//! placed, and a build that answers is moved to its final name with one
//! `rename`, so a half-extracted tree can never appear under a version another run
//! would pick up. A cached copy is proved the same way whenever it is picked up, so
//! a damaged one is passed over for the next that answers instead of failing every
//! render — and a build that proves is reused as it stands: the newest build Google
//! publishes is fetched only when a download is needed, and there is no version of its
//! own to satisfy, because the conditional trust is the proof, not a number.
//! A build that does not run here — it answered wrongly, or its own exit status was a
//! failure — is remembered by its version, so it costs one download rather than one per render
//! and is not kept in the cache either. A build the product could not finish checking — it never
//! started, or its own ceiling or memory watchdog ended the probe — is not the build's verdict:
//! the tree just unpacked is placed the same way without a record against it, and a cached copy
//! is left in place, the older copies are still tried, the reason is reported if none of them
//! proves, and the same copy is tried again next time rather than being written down as one that
//! cannot run here. Either way no second download is spent on the same archive.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::tools::shell::ProgramOutcome;
use crate::util::http::{DownloadSizeCheck, build_download_client, download_verified};

/// The channel Google calls "last known good", and the file that lists each
/// channel's version with the browse downloads beside it.
const LAST_KNOWN_GOOD: &str = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json";

/// How long the extracted binary may take to answer `--version`.
const VERSION_TIMEOUT: Duration = Duration::from_secs(60);

/// Holds the one download in flight (see [`ensure_shell`]).
static INSTALLING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Held shared for the whole of every render and exclusively while the cache is
/// changed, so the build a render launches from is never removed under it — on any
/// platform, not only where the filesystem refuses the removal.
static IN_USE: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

/// The file that records a build this machine could not run: the same archive would
/// be downloaded again for the same answer, so the next call reads it instead. It is
/// keyed by the build's version — a newer Stable is a different binary and is tried
/// normally — and a build that proves overwrites it.
///
/// Bookkeeping, not a build tree: it is written and read under [`IN_USE`]'s *shared*
/// guard, which is what guards the trees themselves against a swap or a removal.
const UNPROVEN: &str = ".unproven";

/// What a build that does not run on this machine is answered with, wherever it is
/// met: a fresh download that failed its proof, or a cached copy that did.
const DOES_NOT_RUN: &str = "the browser build does not run on this machine";

/// The shell one render uses, and the read guard that keeps the cache from removing it
/// while the render launches from it.
pub(super) struct Shell {
    pub(super) binary: PathBuf,
    _held: tokio::sync::RwLockReadGuard<'static, ()>,
}

/// Why the renderer has no browser to launch: there is none to have, or the product
/// could not finish proving the one it has.
#[derive(Debug)]
pub(super) enum Unavailable {
    /// Nothing can be launched: no cached build runs here and no download produced one.
    /// The sentence names the failing step and carries no URL.
    Missing(String),
    /// The build is there, but the product's own check of it was stopped — never the
    /// build's verdict, whether the copy it checked was cached or just downloaded. The
    /// reason is reported when no older copy proves, nothing is recorded against the build,
    /// and the same copy is tried again next time.
    Unchecked(String),
}

/// Why a build did not prove.
enum Unproven {
    /// The product itself ended the check: a verdict about the machine, so it is worth
    /// another attempt and nothing is remembered.
    Stopped(String),
    /// The build does not run on this machine: the same archive would answer the same
    /// way, so it is remembered by its version and reported instead of downloaded.
    DoesNotRun,
}

/// A Chrome for Testing version as the vendor spells it: four numeric components.
/// `semver` cannot read one — Chrome carries a fourth — so ordering is its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ChromeVersion([u64; 4]);

impl ChromeVersion {
    /// The version a cache directory's name or the listing names, `None` for
    /// anything else.
    fn parse(text: &str) -> Option<Self> {
        let mut parts = text.trim().split('.');
        let mut numbers = [0_u64; 4];
        for number in &mut numbers {
            *number = parts.next()?.parse().ok()?;
        }
        if parts.next().is_some() {
            return None;
        }
        Some(Self(numbers))
    }

    /// The spelling the cache directory is named by.
    fn spelling(self) -> String {
        format!("{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// The archive platform tag for a host, `None` on a platform Google publishes
/// nothing for. Windows on ARM takes the ordinary x64 build: there is no
/// `win-arm64`, and the x64 binary runs under Windows 11's own emulation — the
/// same substitution chrome-use's installer makes.
#[must_use]
fn platform_for(os: &str, arch: &str) -> Option<&'static str> {
    Some(match (os, arch) {
        ("macos", "x86_64") => "mac-x64",
        ("macos", "aarch64") => "mac-arm64",
        ("linux", "x86_64") => "linux64",
        ("linux", "aarch64") => "linux-arm64",
        ("windows", "x86_64" | "aarch64") => "win64",
        _ => return None,
    })
}

/// The archive platform tag for THIS build, or `None` when Google publishes no
/// build for it.
#[must_use]
fn platform() -> Option<&'static str> {
    let (os, arch) = crate::util::managed_bin::host_os_arch().ok()?;
    platform_for(os, arch)
}

/// The binary's file name on this platform.
const fn binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "chrome-headless-shell.exe"
    } else {
        "chrome-headless-shell"
    }
}

/// The cache root: one directory per version inside the product's configuration
/// directory.
fn root() -> Result<PathBuf, String> {
    let root = crate::config::default_config_dir()
        .map(|dir| dir.join("chrome-headless-shell"))
        .map_err(|_| "the product's configuration directory is unavailable".to_string())?;
    fs::create_dir_all(&root).map_err(|e| {
        format!(
            "the product's browser directory could not be created ({})",
            e.kind()
        )
    })?;
    Ok(root)
}

/// Every cached copy of this platform's shell, newest first. A directory whose
/// name is not a version, and a version directory with no executable inside, are
/// not copies at all.
fn cached(root: &Path, platform: &str) -> Vec<(ChromeVersion, PathBuf)> {
    let folder = format!("chrome-headless-shell-{platform}");
    let mut copies = Vec::new();
    for entry in fs::read_dir(root).into_iter().flatten().flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let Some(version) = ChromeVersion::parse(&name) else {
            continue;
        };
        let binary = entry.path().join(&folder).join(binary_name());
        if crate::util::is_executable(&binary) {
            copies.push((version, binary));
        }
    }
    copies.sort_by_key(|(version, _)| std::cmp::Reverse(*version));
    copies
}

/// The newest cached copy that proves by running, `None` when none does. A failure that
/// says the build does not run is remembered by its version ([`UNPROVEN`]), so the next
/// call does not spawn it again and [`drop_unproven`] removes the tree. A failure that is
/// the product's own stop is neither passed over as a verdict nor remembered: the copy is
/// left in place, the older copies are still tried — no download is involved either way —
/// and the first such reason is the answer if none of them proves: the machine, not the
/// build, failed to answer.
async fn usable(root: &Path, platform: &str) -> Result<Option<PathBuf>, Unavailable> {
    let mut stopped = None;
    for (version, binary) in cached(root, platform) {
        if recorded_unproven(root, version).is_some() {
            continue;
        }
        match prove_it_runs(&binary, root).await {
            Ok(()) => return Ok(Some(binary)),
            Err(Unproven::DoesNotRun) => record_unproven(root, version, DOES_NOT_RUN),
            Err(Unproven::Stopped(reason)) => {
                stopped.get_or_insert(reason);
            }
        }
    }
    match stopped {
        Some(reason) => Err(Unavailable::Unchecked(reason)),
        None => Ok(None),
    }
}

/// [`usable`] with the guard that keeps the build in place: the guard is taken before
/// the proof, so the cache cannot remove the build between the proof and the render
/// that launches it.
async fn shell_in_use(root: &Path, platform: &str) -> Result<Option<Shell>, Unavailable> {
    let held = IN_USE.read().await;
    Ok(usable(root, platform).await?.map(|binary| Shell {
        binary,
        _held: held,
    }))
}

/// Remove every cached build but the one just placed, and every tree of ours a swap left
/// behind under any name: one build belongs in the cache — the one this product will use —
/// plus the record beside it. A rename-aside tree (`<version>.old`) and a build no longer
/// offered are reclaimed like any other download of ours; a scratch tree is left to
/// [`super::reclaim_scratch`], which knows a download in flight from an abandoned one. The
/// caller holds [`IN_USE`] exclusively, so no render is launching from any of them.
fn prune(root: &Path, keep: ChromeVersion) {
    let kept = keep.spelling();
    for entry in fs::read_dir(root).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != kept && name != UNPROVEN && !name.starts_with(".install-") {
            crate::util::managed_bin::remove_any(&entry.path());
        }
    }
}

/// The build recorded as one that does not run here, and the sentence recorded with
/// it: the file's one reader, so the writer and the removal cannot disagree about its
/// shape.
fn unproven_record(root: &Path) -> Option<(ChromeVersion, String)> {
    let body = fs::read_to_string(root.join(UNPROVEN)).ok()?;
    let (failed, reason) = body.split_once('\n')?;
    Some((
        ChromeVersion::parse(failed.trim())?,
        reason.trim().to_string(),
    ))
}

/// The sentence recorded for a build that did not run here, when the build on offer is
/// that very one.
fn recorded_unproven(root: &Path, version: ChromeVersion) -> Option<String> {
    unproven_record(root)
        .filter(|(recorded, _)| *recorded == version)
        .map(|(_, reason)| reason)
}

/// Remember that a build did not run here, so the next call does not download it again.
fn record_unproven(root: &Path, version: ChromeVersion, reason: &str) {
    let _ = fs::write(
        root.join(UNPROVEN),
        format!("{}\n{reason}", version.spelling()),
    );
}

/// Remove the cached tree of the build recorded as one this machine does not run: a
/// copy that cannot be used is neither kept nor spawned again. The caller holds
/// [`IN_USE`] exclusively, so no render is launching from it.
fn drop_unproven(root: &Path) {
    if let Some((version, _)) = unproven_record(root) {
        let _ = fs::remove_dir_all(root.join(version.spelling()));
    }
}

/// The shell a render is handed — its binary and the guard that keeps the cache from
/// removing it — downloading a build first when the cache holds none that runs. A build
/// this machine already could not run is reported rather than fetched again
/// ([`UNPROVEN`]), and a cached build whose check the product itself stopped is never
/// answered with a fresh download: the reason is reported when no other cached copy proves.
///
/// The download is behind [`INSTALLING`], so two first renders never fetch the
/// same archive beside each other — and neither replaces the directory the other
/// is about to launch from: the loser of that race re-checks the cache before it
/// downloads anything.
pub(super) async fn ensure_shell() -> Result<Shell, Unavailable> {
    let platform = platform()
        .ok_or_else(|| Unavailable::Missing("this platform is not supported".to_string()))?;
    let root = root().map_err(Unavailable::Missing)?;
    // A download killed before it placed anything is ours to clear, whether or not this
    // call downloads.
    super::reclaim_scratch(&root);
    // A tree a swap set aside and a build recorded as one that does not run here are not
    // kept either, and both are removed under the guard that keeps a render from launching
    // out of them. A render holding the cache right now leaves the removal to the next
    // call.
    if let Ok(_swap) = IN_USE.try_write() {
        super::reclaim_aside(&root);
        drop_unproven(&root);
    }
    if let Some(shell) = shell_in_use(&root, platform).await? {
        return Ok(shell);
    }
    let _queue = INSTALLING.lock().await;
    if let Some(shell) = shell_in_use(&root, platform).await? {
        return Ok(shell);
    }
    let client = build_download_client(super::DOWNLOAD_TIMEOUT)
        .map_err(|_| Unavailable::Missing("the download client could not be built".to_string()))?;
    let (version, url) = stable_build(&client, platform)
        .await
        .map_err(Unavailable::Missing)?;
    // A build this machine already could not run is not downloaded again: the same
    // archive would give the same answer.
    if let Some(reason) = recorded_unproven(&root, version) {
        return Err(Unavailable::Missing(reason));
    }
    let binary = install_build(&client, &root, platform, version, &url).await?;
    // The build was proved before it was placed, so it is not proved again: it is
    // handed straight to the render, with the guard that keeps the cache from
    // removing it while that render runs.
    Ok(Shell {
        binary,
        _held: IN_USE.read().await,
    })
}

/// The Stable channel's version and this platform's archive URL, from the
/// listing Google publishes. Every failure names the failing step and carries no
/// URL.
async fn stable_build(
    client: &reqwest::Client,
    platform: &str,
) -> Result<(ChromeVersion, String), String> {
    use crate::util::managed_bin::request_reason;

    let response = client.get(LAST_KNOWN_GOOD).send().await.map_err(|e| {
        format!(
            "the browser build list request failed: {}",
            request_reason(&e)
        )
    })?;
    if !response.status().is_success() {
        return Err(format!(
            "the browser build list got HTTP {}",
            response.status()
        ));
    }
    let body = response.text().await.map_err(|e| {
        format!(
            "the browser build list could not be read: {}",
            request_reason(&e)
        )
    })?;
    let listing = serde_json::from_str::<serde_json::Value>(&body)
        .map_err(|_| "the browser build list was not the expected document".to_string())?;
    let stable = listing
        .pointer("/channels/Stable")
        .ok_or_else(|| "the browser build list carries no stable channel".to_string())?;
    let version = stable
        .get("version")
        .and_then(|v| v.as_str())
        .and_then(ChromeVersion::parse)
        .ok_or_else(|| "the stable channel's answer carries no version".to_string())?;
    let url = stable
        .get("downloads")
        .and_then(|downloads| downloads.get("chrome-headless-shell"))
        .and_then(|builds| builds.as_array())
        .and_then(|builds| {
            builds
                .iter()
                .find(|build| build.get("platform").and_then(|p| p.as_str()) == Some(platform))
        })
        .and_then(|build| build.get("url"))
        .and_then(|url| url.as_str())
        .ok_or_else(|| "the browser build list carries no build for this platform".to_string())?;
    Ok((version, url.to_string()))
}

/// Download and place one build: the archive into scratch space inside the cache,
/// the extracted tree proved by running it, then one rename into
/// `root/<version>/` — the swap the shared managed-binary helper performs when a
/// tree is already there. The build this one replaces is removed once the new one
/// is in place, and the placed binary's path is answered so it is not proved twice.
/// A proof the product itself stopped is handled by [`refused`].
async fn install_build(
    client: &reqwest::Client,
    root: &Path,
    platform: &str,
    version: ChromeVersion,
    url: &str,
) -> Result<PathBuf, Unavailable> {
    // The scratch space lives inside the cache so the final rename stays on one
    // filesystem; it is removed when this function returns either way.
    let work = super::scratch(root).map_err(Unavailable::Missing)?;
    let archive = work.path().join("chrome-headless-shell.zip");
    // Google publishes no digest for these archives, so nothing is checked here:
    // the proof is the binary answering below.
    download_verified(
        client,
        url,
        &archive,
        "",
        None,
        DownloadSizeCheck::None,
        |_, _| {},
    )
    .await
    .map_err(|e| {
        Unavailable::Missing(format!(
            "the browser build could not be downloaded: {}",
            crate::util::managed_bin::download_reason(&e)
        ))
    })?;
    let extracted = work.path().join("extracted");
    fs::create_dir_all(&extracted).map_err(|e| {
        Unavailable::Missing(format!(
            "the browser build could not be unpacked ({})",
            e.kind()
        ))
    })?;
    crate::util::managed_bin::extract_tree_zip(&archive, &extracted).map_err(|reason| {
        Unavailable::Missing(format!("the browser build could not be unpacked: {reason}"))
    })?;
    let folder = format!("chrome-headless-shell-{platform}");
    let binary = extracted.join(&folder).join(binary_name());
    if !binary.is_file() {
        return Err(Unavailable::Missing(
            "the browser archive carries no binary".to_string(),
        ));
    }
    // An archive does not always carry the mode bits; the product's own copy is
    // always meant to be runnable.
    crate::util::managed_bin::set_executable(&binary).map_err(Unavailable::Missing)?;
    if let Err(unproven) = prove_it_runs(&binary, work.path()).await {
        return Err(refused(root, version, &folder, &extracted, unproven).await);
    }

    // The whole extracted tree is the copy: `<version>/<folder>/<binary>`. Nothing a
    // render reads is touched before this point, and the cache is changed under the
    // exclusive guard, so no render is launching from what is replaced.
    let placed = place(root, version, &folder, &extracted).await?;
    // A build that proved replaces whatever an earlier one was remembered for.
    let _ = fs::remove_file(root.join(UNPROVEN));
    Ok(placed)
}

/// What a refused proof does with the tree it ran, and with the record beside the cache: a build
/// that does not run here is remembered and its tree dropped, while a build the product itself
/// stopped keeps its tree with the reason for this attempt only — the next call proves that copy
/// instead of downloading the archive again, and nothing is written down against it.
async fn refused(
    root: &Path,
    version: ChromeVersion,
    folder: &str,
    tree: &Path,
    unproven: Unproven,
) -> Unavailable {
    match unproven {
        Unproven::DoesNotRun => {
            record_unproven(root, version, DOES_NOT_RUN);
            Unavailable::Missing(DOES_NOT_RUN.to_string())
        }
        Unproven::Stopped(reason) => {
            // A tree that cannot be placed costs this attempt its cache entry, not its reason.
            let _ = place(root, version, folder, tree).await;
            Unavailable::Unchecked(reason)
        }
    }
}

/// Move the unpacked tree to `root/<version>` and drop every other build — one build stays in the
/// cache — under the guard that keeps a render from launching out of what is replaced. The placed
/// binary's path is answered, so a build that proved is not proved again.
async fn place(
    root: &Path,
    version: ChromeVersion,
    folder: &str,
    tree: &Path,
) -> Result<PathBuf, Unavailable> {
    let placed = root.join(version.spelling());
    let _swap = IN_USE.write().await;
    crate::util::managed_bin::place_staged_tree(tree, &placed).map_err(Unavailable::Missing)?;
    prune(root, version);
    Ok(placed.join(folder).join(binary_name()))
}

/// The extracted build's own answer: it must exit clean and name itself. This is
/// the only proof a download without a digest can give, and the same check is
/// repeated whenever a cached copy is picked up (see [`usable`]).
async fn prove_it_runs(binary: &Path, cwd: &Path) -> Result<(), Unproven> {
    let args = vec!["--version".to_string()];
    let outcome = crate::tools::shell::run_internal_program_outcome(
        cwd,
        binary,
        &args,
        &[],
        Some(VERSION_TIMEOUT),
    )
    .await;
    proof_verdict(&outcome)
}

/// What one probe's outcome says about the build it ran. A build that answered with its own
/// name runs here; one that exited badly or answered something else does not run here
/// ([`Unproven::DoesNotRun`]), and the same archive would give the same answer next time. A stop
/// is the product's own verdict, never the build's — it never started the build, its ceiling or
/// its memory watchdog ended the probe — so the reason is kept for this attempt only and nothing
/// is written down against the build.
fn proof_verdict(outcome: &ProgramOutcome) -> Result<(), Unproven> {
    if outcome.stopped() {
        return Err(Unproven::Stopped(outcome.detail.clone()));
    }
    if !outcome.success
        || !outcome
            .output
            .trim_start()
            .starts_with("Google Chrome for Testing")
    {
        return Err(Unproven::DoesNotRun);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_for_maps_each_published_platform() {
        assert_eq!(platform_for("macos", "x86_64"), Some("mac-x64"));
        assert_eq!(platform_for("macos", "aarch64"), Some("mac-arm64"));
        assert_eq!(platform_for("linux", "x86_64"), Some("linux64"));
        assert_eq!(platform_for("linux", "aarch64"), Some("linux-arm64"));
        assert_eq!(platform_for("windows", "x86_64"), Some("win64"));
        // Windows on ARM takes the x64 build: Google publishes no `win-arm64`.
        assert_eq!(platform_for("windows", "aarch64"), Some("win64"));
        // Nothing is published for these, and another processor's build is never
        // substituted.
        assert_eq!(platform_for("linux", "armv7l"), None);
        assert_eq!(platform_for("windows", "arm64ec"), None);
        assert_eq!(platform_for("freebsd", "x86_64"), None);
    }

    #[test]
    fn chrome_version_orders_four_numeric_components() {
        let older = ChromeVersion::parse("141.0.7390.54").expect("a version");
        let newer = ChromeVersion::parse("142.0.7444.175").expect("a version");
        assert!(older < newer);
        assert_eq!(newer.spelling(), "142.0.7444.175");
        assert_eq!(ChromeVersion::parse("142.0.7444"), None);
        assert_eq!(ChromeVersion::parse("142.0.7444.175.1"), None);
        assert_eq!(ChromeVersion::parse("stable"), None);
    }

    /// A swap keeps the build it just placed and drops every other tree in the cache,
    /// including the one the swap set aside: nothing of ours survives because of how it
    /// was named. The record beside the builds stays.
    #[test]
    fn a_swap_keeps_the_new_build_and_drops_the_rest() {
        let root = tempfile::tempdir().unwrap();
        for name in ["150.0.1.0", "151.0.1.0", "150.0.1.0.old"] {
            fs::create_dir_all(root.path().join(name)).unwrap();
        }
        fs::write(root.path().join(UNPROVEN), "150.0.1.0\n").unwrap();

        prune(root.path(), ChromeVersion::parse("151.0.1.0").unwrap());

        assert!(
            root.path().join("151.0.1.0").exists(),
            "the new build stays"
        );
        for gone in ["150.0.1.0", "150.0.1.0.old"] {
            assert!(!root.path().join(gone).exists(), "{gone} goes");
        }
        assert!(
            root.path().join(UNPROVEN).exists(),
            "the record is not a build and is not swept"
        );
    }

    /// A build that did not run is remembered for that build only: the same version is
    /// not downloaded again, and a newer one is tried normally.
    #[test]
    fn a_build_that_did_not_run_is_remembered_for_that_build_only() {
        let root = tempfile::tempdir().unwrap();
        let failed = ChromeVersion::parse("156.0.8078.73").expect("a version");
        assert_eq!(recorded_unproven(root.path(), failed), None);

        record_unproven(root.path(), failed, DOES_NOT_RUN);

        assert_eq!(
            recorded_unproven(root.path(), failed).as_deref(),
            Some(DOES_NOT_RUN)
        );
        let newer = ChromeVersion::parse("157.0.1.0").expect("a version");
        assert_eq!(recorded_unproven(root.path(), newer), None);
    }

    /// The cache is proved at use, not only at install: a copy that no longer
    /// answers is not used (the download replaces it, proving the new build first),
    /// and a copy that answers is used as it stands.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_cached_copy_is_proved_at_use() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempfile::tempdir().unwrap();
        let platform = "linux64";
        let write = |version: &str, body: &str| {
            let dir = root
                .path()
                .join(version)
                .join(format!("chrome-headless-shell-{platform}"));
            fs::create_dir_all(&dir).unwrap();
            let binary = dir.join(binary_name());
            fs::write(&binary, body).unwrap();
            fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        };

        write("150.0.1.0", "#!/bin/sh\nexit 1\n");
        assert!(
            usable(root.path(), platform).await.ok().flatten().is_none(),
            "a copy that does not answer is not used"
        );
        let dead = ChromeVersion::parse("150.0.1.0").expect("a version");
        assert!(
            recorded_unproven(root.path(), dead).is_some(),
            "a copy that does not run is remembered, so it is never spawned again"
        );
        assert!(
            root.path().join("150.0.1.0").exists(),
            "the proof itself deletes nothing: no relocation happens where a render may be launching"
        );
        drop_unproven(root.path());
        assert!(
            !root.path().join("150.0.1.0").exists(),
            "a copy that does not run is not kept either"
        );

        write(
            "151.0.1.0",
            "#!/bin/sh\necho 'Google Chrome for Testing 151.0.1.0'\n",
        );
        let binary = usable(root.path(), platform)
            .await
            .expect("the proof completes")
            .expect("the copy that answers is used");
        assert!(
            binary.ends_with("chrome-headless-shell"),
            "got: {}",
            binary.display()
        );

        // A newer copy that does not answer is passed over for the one that does,
        // rather than leaving the machine with no browser at all.
        write("152.0.1.0", "#!/bin/sh\nexit 1\n");
        let binary = usable(root.path(), platform)
            .await
            .expect("the proof completes")
            .expect("the older copy that answers is still used");
        assert!(
            binary.starts_with(root.path().join("151.0.1.0")),
            "got: {}",
            binary.display()
        );
    }

    /// What a probe's own outcome says about the build. A run the machine ended is never the
    /// build's verdict and keeps its reason for this attempt — the build is not written down as
    /// one that does not run, whether it never started or a ceiling ended the probe; only a build
    /// that answered, and answered wrongly, is.
    #[test]
    fn a_stop_is_never_the_builds_verdict() {
        let run =
            |success: bool, output: &str, stop: Option<crate::tools::shell::Stop>| ProgramOutcome {
                success,
                detail: "the machine's own reason".to_string(),
                output: output.to_string(),
                has_output: !output.is_empty(),
                stop,
            };
        assert!(proof_verdict(&run(true, "Google Chrome for Testing 151.0.1.0\n", None)).is_ok());
        assert!(matches!(
            proof_verdict(&run(true, "Chrome 1.0\n", None)),
            Err(Unproven::DoesNotRun)
        ));
        assert!(matches!(
            proof_verdict(&run(false, "", None)),
            Err(Unproven::DoesNotRun)
        ));
        for stop in [
            crate::tools::shell::Stop::Spawn,
            crate::tools::shell::Stop::Memory,
        ] {
            let Err(Unproven::Stopped(reason)) = proof_verdict(&run(false, "", Some(stop))) else {
                panic!("{stop:?} is a stop, not the build's verdict");
            };
            assert_eq!(reason, "the machine's own reason");
        }
    }

    /// A cached copy the product could not start is passed over for one that proves, and nothing
    /// is recorded against it: a build we could not start is tried again, never written down as
    /// one that does not run here. The search costs no download either way.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_cached_copy_the_product_could_not_start_is_passed_over() {
        let root = tempfile::tempdir().unwrap();
        let platform = "linux64";
        let write = |version: &str, body: &str| {
            let dir = root
                .path()
                .join(version)
                .join(format!("chrome-headless-shell-{platform}"));
            fs::create_dir_all(&dir).unwrap();
            let binary = dir.join(binary_name());
            fs::write(&binary, body).unwrap();
            crate::util::managed_bin::set_executable(&binary).unwrap();
        };
        // Listed as a copy, and impossible to start: its interpreter is not on this machine.
        write("153.0.1.0", "#!/nonexistent/hyperframes-test\n");
        write(
            "152.0.1.0",
            "#!/bin/sh\necho 'Google Chrome for Testing 152.0.1.0'\n",
        );

        let binary = usable(root.path(), platform)
            .await
            .expect("the proof completes")
            .expect("the older copy that answers is used");
        assert!(
            binary.starts_with(root.path().join("152.0.1.0")),
            "got: {}",
            binary.display()
        );
        assert!(
            recorded_unproven(root.path(), ChromeVersion::parse("153.0.1.0").unwrap()).is_none(),
            "a build we could not start is not a build that does not run"
        );
        assert!(
            root.path().join("153.0.1.0").exists(),
            "the copy is kept for the next attempt"
        );
    }

    /// A build just unpacked whose proof ends in a product stop is not thrown away: its tree is
    /// placed in the cache with no verdict against it, so the next call proves that copy rather
    /// than downloading the archive again. A build that answered wrongly is remembered and its
    /// tree is not kept.
    #[tokio::test]
    async fn a_stopped_proof_keeps_the_build_and_a_wrong_one_is_remembered() {
        let root = tempfile::tempdir().unwrap();
        let platform = "linux64";
        let folder = format!("chrome-headless-shell-{platform}");
        let version = ChromeVersion::parse("151.0.1.0").expect("a version");
        let unpack = |name: &str| {
            let tree = root.path().join(name);
            let dir = tree.join(&folder);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(binary_name()), "a build").unwrap();
            tree
        };

        let answer = refused(
            root.path(),
            version,
            &folder,
            &unpack("first"),
            Unproven::Stopped("terminated: exceeded memory limit (used ~2 GiB)".to_string()),
        )
        .await;
        assert!(matches!(answer, Unavailable::Unchecked(_)));
        assert!(
            root.path()
                .join(version.spelling())
                .join(&folder)
                .join(binary_name())
                .is_file(),
            "the build stays in the cache"
        );
        assert!(
            recorded_unproven(root.path(), version).is_none(),
            "a stop is the product's verdict, never the build's"
        );

        let answer = refused(
            root.path(),
            version,
            &folder,
            &unpack("second"),
            Unproven::DoesNotRun,
        )
        .await;
        assert!(matches!(answer, Unavailable::Missing(_)));
        assert_eq!(
            recorded_unproven(root.path(), version).as_deref(),
            Some(DOES_NOT_RUN)
        );
    }
}
