//! The HyperFrames package: installed from the npm registry by the product's own managed bun
//! on first use, and kept at the newest release by the start-time task.
//!
//! A copy is built in scratch space inside the product's directory and placed with the
//! shared managed-binary swap — one rename, or a rename-aside when a tree is already there —
//! so a download that fails halfway leaves the copy that was there untouched. The tree a swap
//! sets aside is reclaimed at once, and the sweep at start takes what a killed download left.
//! A failed first use is retried on the next call; a failed refresh leaves the working copy as
//! it was, to be retried on the next start. A failure's reason names the failing step and is
//! safe to write to the log store verbatim.

use std::path::PathBuf;
use std::time::Duration;

use semver::Version;

use crate::util::managed_bin::{self, Release};

const REGISTRY_LATEST: &str = "https://registry.npmjs.org/hyperframes/latest";

/// The package's own name on the registry, in the log store and in its tree.
const PACKAGE: &str = "hyperframes";

/// The vendor's entry point inside the installed `node_modules`, and what a tree without one
/// is answered with.
const ENTRY: &str = "hyperframes/bin/hyperframes.mjs";
const NO_ENTRY: &str = "the installed hyperframes package carries no entry point";

const NODE_MODULES: &str = "node_modules";

const RELEASE_TIMEOUT: Duration = Duration::from_secs(60);

static INSTALLING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Held shared by every run and exclusively while an install replaces the tree, so a release
/// is never swapped under a run still reading it — and never held across a download, which
/// touches nothing a run reads.
static IN_USE: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

/// Where the package lives: the vendor's tree and the installs' scratch space sit inside the
/// product's configuration directory, so no part of the tool reaches the owner's workspace or
/// his own npm state. Nothing is created here — a lookup must not bring the directory into
/// being before the tool is ever used.
fn root() -> Result<PathBuf, String> {
    crate::config::default_config_dir()
        .map(|dir| dir.join(PACKAGE))
        .map_err(|_| "the product's configuration directory is unavailable".to_string())
}

pub(super) fn dir() -> Result<PathBuf, String> {
    let dir = root()?;
    std::fs::create_dir_all(&dir).map_err(|e| {
        format!(
            "the product's hyperframes directory could not be created ({})",
            e.kind()
        )
    })?;
    Ok(dir)
}

pub(super) fn entry() -> Option<PathBuf> {
    let entry = root().ok()?.join(NODE_MODULES).join(ENTRY);
    entry.is_file().then_some(entry)
}

/// Hold the package for one run and answer the entry point it is given: the read guard keeps
/// a start-time refresh from swapping the package under it, and the path is read *after* the
/// guard is taken, so it is the one on disk now.
pub(super) async fn hold_for_run()
-> Result<(tokio::sync::RwLockReadGuard<'static, ()>, PathBuf), String> {
    let held = IN_USE.read().await;
    let entry = entry().ok_or_else(|| "the renderer is not installed".to_string())?;
    Ok((held, entry))
}

/// The installed copy's version, read from the package's own manifest — no process is
/// spawned. `None` means there is no copy or its manifest does not parse, the "needs
/// (re)install" answer the start-time policy takes. The lookup creates nothing.
async fn installed_version() -> Option<Version> {
    let path = root()
        .ok()?
        .join(NODE_MODULES)
        .join("hyperframes/package.json");
    let body = tokio::fs::read_to_string(path).await.ok()?;
    manifest_version(&body)
}

/// The start-time policy's first deferred step: the machine still has a copy, so the scratch a
/// killed install left is reclaimed — after the delay, like everything else the boot defers —
/// and the copy's version read.
async fn start_probe() -> Option<Version> {
    if let Ok(dir) = root() {
        super::reclaim_scratch(&dir);
    }
    installed_version().await
}

fn manifest_version(body: &str) -> Option<Version> {
    let value = serde_json::from_str::<serde_json::Value>(body).ok()?;
    Version::parse(value.get("version")?.as_str()?).ok()
}

/// The newest release the registry offers. Every failure names the failing step and carries
/// no URL, so it is safe to record verbatim.
async fn newest_version() -> Result<Version, String> {
    let client = crate::util::http::build_download_client(RELEASE_TIMEOUT)
        .map_err(|_| "the release check's own client could not be built".to_string())?;
    let response = client.get(REGISTRY_LATEST).send().await.map_err(|e| {
        format!(
            "the release check request failed: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })?;
    if !response.status().is_success() {
        return Err(format!("the release check got HTTP {}", response.status()));
    }
    let body = response.text().await.map_err(|e| {
        format!(
            "the release check's answer could not be read: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })?;
    manifest_version(&body)
        .ok_or_else(|| "the newest release's answer carries no version".to_string())
}

async fn newest_release() -> Result<Release, String> {
    let version = newest_version().await?;
    Ok(Release {
        tag: version.to_string(),
        version,
    })
}

/// The entry point of the package, installed from the newest release when there is no copy. A
/// copy in place is used as it stands: the start-time task is what compares it with the
/// registry. Nothing is remembered about a failed attempt — the machine may have had no
/// network yet, so the next call tries again.
pub(super) async fn ensure_installed() -> Result<(), String> {
    if entry().is_some() {
        return Ok(());
    }
    let version = newest_version().await?;
    install(&version).await.map(|_| ())
}

/// Install `version` with the product's own bun and answer the entry point it left.
/// Serialized by [`INSTALLING`], and exclusive against running renders only for the rename
/// that replaces the live tree — a download takes its own time without holding anything a
/// render waits on ([`managed_bin::swap_when_idle`]).
async fn install(version: &Version) -> Result<PathBuf, String> {
    let _installing = INSTALLING.lock().await;
    // Another call may have installed this very version while this one queued.
    if installed_version().await.as_ref() == Some(version)
        && let Some(entry) = entry()
    {
        return Ok(entry);
    }
    let dir = dir()?;
    // Whatever a killed install or a replaced tree left is ours to clear: nothing of ours may
    // accumulate in the product's directory. The aside goes now because every swap creating one
    // is taken inside an install, and this holds [`INSTALLING`] for its whole body.
    super::reclaim_scratch(&dir);
    super::reclaim_aside(&dir);
    let bun = crate::tools::bun::bun_binary_path()
        .ok_or_else(|| "the product's managed bun runtime is not installed".to_string())?;
    let work = super::scratch(&dir)?;
    let args = vec![
        "add".to_string(),
        "--exact".to_string(),
        format!("{PACKAGE}@{version}"),
    ];
    let outcome = crate::tools::shell::run_internal_program_outcome(
        work.path(),
        &bun,
        &args,
        &[],
        Some(super::DOWNLOAD_TIMEOUT),
    )
    .await;
    if !outcome.success {
        return Err(super::run_failure(
            &outcome,
            "installing the hyperframes package",
            Some(super::DOWNLOAD_TIMEOUT),
        ));
    }
    let staged = work.path().join(NODE_MODULES);
    if !staged.join(ENTRY).is_file() {
        return Err(NO_ENTRY.to_string());
    }
    // Only now is anything a run reads replaced, by the shared managed-binary swap — one
    // rename, or a rename-aside when a tree is already there.
    let _swap = managed_bin::swap_when_idle(&IN_USE).await;
    managed_bin::place_staged_tree(&staged, &dir.join(NODE_MODULES))?;
    entry().ok_or_else(|| NO_ENTRY.to_string())
}

/// The start-time task: bring an installed copy to the newest release. The shared policy
/// ([`managed_bin::install_on_start`]) is told to leave a machine that never used the tool
/// alone ([`managed_bin::WhenMissing::Skip`]) — being present costs nothing until it is used.
/// A failed refresh never invalidates the copy: it keeps serving renders, and the next start
/// tries again.
pub(super) async fn refresh_on_start() {
    managed_bin::install_on_start(
        PACKAGE,
        || entry().is_some(),
        start_probe,
        newest_release,
        async |release| {
            let release = managed_bin::release_or_reason(release)?;
            install(&release.version).await
        },
        managed_bin::WhenMissing::Skip,
    )
    .await;
}
