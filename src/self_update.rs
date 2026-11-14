//! Self-update logic — single-instance guarding, obtaining the new version,
//! putting it in place, and restart.
//!
//! Two modes, chosen at runtime by [`update_mode`]. A **downloaded copy**
//! ([`UpdateMode::Downloaded`]) — a published release file, the package
//! registry's source cache, a moved copy, or a copy whose build source is gone —
//! downloads this system's ready-made release file, extracts it, puts it at the
//! standard per-user location (swapping the working file in place when that is
//! already where the copy runs from, else placing it there and taking the working
//! file away) and restarts. A **source tree** ([`UpdateMode::SourceTree`]) runs
//! `cargo install --path <tree> --root <temp_install> --locked --target-dir
//! <temp_build>`, swaps, and restarts.
//!
//! Both converge on one tail: the new file becomes the one the replacement runs
//! from, the admin is notified, in-flight work is drained, the databases are
//! checkpointed, and a replacement is started from that file before `exit(0)`. A
//! running executable cannot be replaced in place on Windows, so the source-tree
//! build lands in a temp root and is swapped in via `self_replace` (rename-aside
//! of the running exe, copy in, deferred deletion) — the product no longer copies
//! itself into any toolchain's directory, in either mode.
//!
//! A downloaded copy also updates *itself*: the periodic release check
//! ([`run_update_availability_refresh`]) starts the very same update with no
//! prompt while the owner's Autoupdate switch is on, and a round the restart cuts
//! is re-driven after the boot. With the switch off the check still runs and the
//! button offers the update. Which copies update themselves, and for which
//! release, is the whole of [`auto_update_target`].
//!
//! The product is its own release host: [`RELEASE_REPO`] (`CARGO_PKG_REPOSITORY`)
//! holds one release per version, and the release workflow, the install scripts
//! and this file mirror one contract — the newest release's version file is
//! `{base}/releases/latest/download/version.txt` and holds `<version>\n`; a
//! version's archive is `{base}/releases/download/v{version}/{name}`, `{name}`
//! being `mahbot-<version>-<os>-<arch>.<ext>` with the (`os`, `arch`) pair
//! [`crate::util::managed_bin::host_os_arch`] returns and `tar.gz` on macOS/Linux,
//! `zip` on Windows, holding exactly one file at its root; every release whose
//! body lists changes carries that same text as `changelog.txt` under
//! `releases/download/v{version}/`, and its release page
//! (`releases/tag/v{version}`) carries it too. No hosting-service API is called —
//! only those download URLs.
//!
//! A downloaded update takes that change list down with itself
//! ([`capture_update_announcement`]), and the first start of the new version hands
//! it to the admin's assistant as one tagged message in its ordinary queue
//! ([`announce_update_to_assistant`]). An update that brought no usable list — a
//! failed fetch, one made by the install scripts, a release published before the
//! list travelled with it — is answered from the running version's release while
//! the notice is composed; failing that too, the notice says the list is
//! unavailable and names the release page. The captured list is kept until a
//! notice has actually been handed over, so a start that does not announce — a
//! build from source, an unfinished setup, a process that stops early — still
//! announces it later. A first install and a restart of the same version with
//! nothing owed say nothing.
//!
//! ## Test-only hooks
//!
//! [`RELEASE_BASE_URL_ENV`] and [`UPDATE_TO_VERSION_ENV`] drive a whole update
//! with no window, from a release host other than the product's own
//! ([`run_env_named_update`]): the version must be named outright, because a test
//! release is published with a pre-release tag and is deliberately
//! undiscoverable. Both are unset in every ordinary run. Moving only the base
//! leaves the hook inert (it drives an update only for a named version) while the
//! check — and with it the unattended trigger — reads that other host.
//!
//! Single-instance enforcement is an exclusive whole-file lock on `mahbot.lock`,
//! released explicitly before the hand-off spawn — see [`acquire_lock`] and
//! [`crate::util::lock`]. The WAL checkpoint before `exit(0)` is a clean store
//! handoff: `std::process::exit(0)` bypasses all Rust destructors, so Turso
//! connections are never properly closed; the TRUNCATE leaves an empty WAL, and
//! committed data is already fsync-durable at COMMIT.
//!
//! ## macOS Gatekeeper safety
//!
//! Spawning a binary triggers async code-signing validation; deleting the spawn
//! target during it produces empty stderr (SIGKILL by `syspolicyd`). The spawn
//! target is always the working file the update left in place — never a temp
//! root — and the temp roots are removed before the lock release and spawn (see
//! [`finalize_update_and_restart`]).

use crate::ChannelMessage;
use crate::util::UnwrapPoison;
use anyhow::{Context, Result, anyhow};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// The embedded package version (`CARGO_PKG_VERSION`): authoritative for the
/// running binary, and what a downloaded copy's self-update compares the release
/// host's version against.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Environment marker the updating parent sets (to `"1"`) on the replacement
/// instance it spawns; [`acquire_lock`] owns what it does. It reaches an agent's
/// unix shell (that shell's environment is the process's) and the GUI Shell tab's
/// PTY, but not git children (cleared environment) nor Windows (environment
/// assembled from the account), so a `mahbot` launched from the replacement's own
/// terminal takes the hand-off wait once before the ordinary refusal.
const HANDOFF_ENV: &str = "MAHBOT_UPDATE_HANDOFF";

/// Environment marker the updating parent sets on the replacement instance it
/// spawns — `"1"` when the window it replaces was hidden, `"0"` otherwise — so the
/// window opens as it was left (see `gui::app_window`). Written on every spawn,
/// never left to inheritance: the parent may itself have come back hidden.
const START_HIDDEN_ENV: &str = "MAHBOT_START_HIDDEN";

/// The launch this marker makes open hidden: the updater's replacement of a hidden
/// window (see [`START_HIDDEN_ENV`]).
#[must_use]
pub fn starts_hidden() -> bool {
    std::env::var(START_HIDDEN_ENV).is_ok_and(|value| value == "1")
}

/// How long a hand-off-marked instance waits for the previous instance to release
/// the location before refusing ([`acquire_lock`]).
const HANDOFF_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Poll interval of the hand-off wait.
const HANDOFF_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Acquire an exclusive whole-file lock on `mahbot.lock` in `storage_root` (typically
/// [`crate::config::default_config_dir`]), refusing if another instance holds it. The
/// guard lives in `INSTANCE_LOCK` for the process lifetime and is released on termination
/// (`exit(0)` included) unless [`FlockGuard::release`] did so first — `util::lock` owns
/// that story.
///
/// A hand-off-marked instance (see [`HANDOFF_ENV`]) *waits* instead: the parent releases
/// the lock before spawning, but the child can reach this point before that release is
/// visible, and refusing would abandon a completed update's restart. The wait is bounded
/// by [`HANDOFF_LOCK_WAIT`], then falls through to the ordinary refusal; an unmarked
/// process is refused at once, with no sleeps. Panics if called more than once (only
/// `main` at startup calls it).
pub fn acquire_lock(storage_root: &Path) -> Result<()> {
    let lock_path = crate::util::lock::lock_file_path(storage_root);

    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let file = match try_acquire_lock(&lock_path)? {
        Some(file) => file,
        None if is_handoff() => {
            crate::boot::boot_diagnostic(format!(
                "update hand-off: the instance lock is still held — waiting up to {}s for the \
                 previous instance to release it",
                HANDOFF_LOCK_WAIT.as_secs()
            ));
            wait_for_handoff(&lock_path)?
        }
        None => return Err(second_instance_refusal(&lock_path)),
    };

    info!(path = %lock_path.display(), "Acquired instance lock");

    let guard = FlockGuard {
        file: Some(file),
        lock_path,
    };
    INSTANCE_LOCK
        .set(Mutex::new(guard))
        .expect("acquire_lock called more than once");
    Ok(())
}

/// Poll for the previous instance's release until [`HANDOFF_LOCK_WAIT`] expires,
/// then fall back to the ordinary refusal. Reached only by a hand-off-marked
/// instance whose immediate attempt already failed.
fn wait_for_handoff(lock_path: &Path) -> Result<File> {
    let deadline = std::time::Instant::now() + HANDOFF_LOCK_WAIT;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(HANDOFF_POLL_INTERVAL);
        if let Some(file) = try_acquire_lock(lock_path)? {
            return Ok(file);
        }
    }
    Err(second_instance_refusal(lock_path))
}

/// Whether this launch is the updater's replacement of a previous instance
/// ([`HANDOFF_ENV`]), which takes the location over rather than being a second
/// launch of the product.
#[must_use]
pub fn is_handoff() -> bool {
    std::env::var(HANDOFF_ENV).is_ok_and(|value| value == "1")
}

/// The wording every instance refusal carries. It is a matched surface — the shell
/// tool recognises a stale self-update binary by it
/// (`shell::grep_engine::STALE_BINARY_LOCK_MSG`) — so it must not change.
pub(crate) const INSTANCE_REFUSAL_PREFIX: &str = "Another instance of mahbot is already running";

/// The error a launch gets when another instance holds the location.
fn second_instance_refusal(lock_path: &Path) -> anyhow::Error {
    anyhow!(
        "{INSTANCE_REFUSAL_PREFIX} (lock file: {}). \
         The lock is a kernel flock released automatically when that instance exits.",
        lock_path.display()
    )
}

/// Whether `error` is [`acquire_lock`]'s refusal rather than a failure of the lock
/// itself (an unreadable directory, an OS error).
#[must_use]
pub fn is_instance_refusal(error: &anyhow::Error) -> bool {
    error.to_string().starts_with(INSTANCE_REFUSAL_PREFIX)
}

/// Open the file (creating it if needed) and take an exclusive whole-file lock
/// without blocking. `Ok(Some(file))` holds the lock — the caller must keep the
/// `File` alive for its lifetime; `Ok(None)` means another handle holds it (another
/// process, or a second descriptor here, which the kernel treats as independent);
/// `Err` is a non-retryable OS error. A second acquisition on a path is refused
/// here rather than waited for.
///
/// Callers: [`acquire_lock`], which creates the directory first, the hand-off poll,
/// and [`reacquire_instance_lock`], which returns early while it still holds its own guard.
fn try_acquire_lock(path: &Path) -> Result<Option<File>> {
    let file = open_lock_file(path)
        .with_context(|| format!("failed to open lock file {}", path.display()))?;

    if crate::util::lock::try_flock(&file)
        .with_context(|| format!("failed to lock {}", path.display()))?
    {
        Ok(Some(file))
    } else {
        Ok(None)
    }
}

/// Open (or create) the lock file with the standard options, for production and
/// tests alike.
fn open_lock_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// A guard holding the instance lock file. [`release`](FlockGuard::release) unlocks
/// explicitly and then closes the handle; the drop does the same at the latest.
/// [`reacquire_instance_lock`] covers a failed spawn that leaves this process alive.
struct FlockGuard {
    file: Option<File>,
    lock_path: PathBuf,
}

impl std::fmt::Debug for FlockGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlockGuard")
            .field("held", &self.file.is_some())
            .field("lock_path", &self.lock_path)
            .finish()
    }
}

impl FlockGuard {
    /// Release the lock: unlock the file explicitly, then close the handle.
    /// Idempotent. The explicit unlock is what the update hand-off relies on — see
    /// `util::lock`'s module doc for why the close-time release is not enough. A
    /// failed unlock only warns: the close still releases the lock.
    fn release(&mut self) {
        if let Some(file) = self.file.take() {
            if let Err(e) = file.unlock() {
                warn!(
                    error = %e,
                    path = %self.lock_path.display(),
                    "Failed to unlock instance lock file — the close still releases it"
                );
            }
            info!(path = %self.lock_path.display(), "Released instance lock");
        }
    }
}

/// Global instance lock, held for the process lifetime so [`execute_update`] can
/// release and re-acquire it.
static INSTANCE_LOCK: OnceLock<Mutex<FlockGuard>> = OnceLock::new();

/// Release the instance lock so a child process can acquire it on startup: called
/// just before spawning the replacement. No-op if unset or already released.
async fn release_instance_lock() {
    if let Some(mutex) = INSTANCE_LOCK.get() {
        let mut guard = mutex.lock().await;
        guard.release();
    }
}

/// Re-acquire the instance lock after a failed spawn: [`spawn_new_instance_from`]
/// failed, the process stays alive, and it must re-claim the location. A recoverable
/// path — it runs after every agent was cancelled and shutdown signalled, and no
/// browser session is closed on the way out (a session ends by its own lifetime or
/// an explicit `session stop`; see [`crate::chrome::cli_sessions`]).
async fn reacquire_instance_lock() -> Result<()> {
    let mutex = INSTANCE_LOCK
        .get()
        .context("Instance lock not initialized")?;

    let lock_path = {
        let guard = mutex.lock().await;
        if guard.file.is_some() {
            return Ok(()); // Already held.
        }
        guard.lock_path.clone()
    };

    let file = try_acquire_lock(&lock_path)?;

    let mut guard = mutex.lock().await;
    match file {
        Some(file) => {
            info!(path = %guard.lock_path.display(), "Re-acquired instance lock");
            guard.file = Some(file);
            Ok(())
        }
        None => Err(anyhow!(
            "Failed to re-acquire instance lock — another instance may have started"
        )),
    }
}

/// Where the product's own releases live: this crate's repository, one release per
/// version — published by the release workflow, downloaded by the install scripts
/// and by this file, so it is the single source of truth for all three.
pub(crate) const RELEASE_REPO: &str = env!("CARGO_PKG_REPOSITORY");

/// Test-only: the release base a driven update downloads from, in place of
/// [`RELEASE_REPO`].
const RELEASE_BASE_URL_ENV: &str = "MAHBOT_RELEASE_BASE_URL";

/// Test-only: the version a driven update moves to, named outright so no
/// newest-release lookup is needed.
const UPDATE_TO_VERSION_ENV: &str = "MAHBOT_UPDATE_TO_VERSION";

/// How long downloading this system's archive may take. Generous on purpose — tens
/// of megabytes over a slow link — but still a hard deadline, so a stalled transfer
/// cannot hang the update.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(30);
/// How long the host's own `sw_vers`/`getconf` answer may take ([`host_probe`]).
/// A probe still running after this is read as one that answered nothing, which
/// refuses nothing.
const HOST_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// The release base every download is built from — [`RELEASE_REPO`] unless
/// [`RELEASE_BASE_URL_ENV`] moved it.
fn release_base_url() -> String {
    std::env::var(RELEASE_BASE_URL_ENV).unwrap_or_else(|_| RELEASE_REPO.to_string())
}

fn latest_version_url(base: &str) -> String {
    format!("{base}/releases/latest/download/version.txt")
}

/// The name of this system's archive for `version`:
/// `mahbot-<version>-<os>-<arch>.<ext>`, `zip` on Windows and `tar.gz` elsewhere.
/// The release workflow names the assets exactly like this and the install scripts
/// read the same names back, so the three must stay in step.
fn asset_name(version: &semver::Version, os: &str, arch: &str) -> String {
    let extension = if os == "windows" { "zip" } else { "tar.gz" };
    format!("mahbot-{version}-{os}-{arch}.{extension}")
}

fn asset_url(base: &str, version: &semver::Version, os: &str, arch: &str) -> String {
    format!(
        "{base}/releases/download/v{version}/{}",
        asset_name(version, os, arch)
    )
}

/// The floors the published files are built against: the deployment target both
/// macOS legs pass to the compiler, the glibc of the Linux base, and the Windows
/// build each Windows file is for (1809 on x86_64, Windows 11 on ARM — every Windows
/// 10 on ARM is below the ARM64 file). A system below its floor has no file that can
/// load, so it is refused like an os/arch with none published. The macOS and Linux
/// floors are shared with every other prebuilt artifact (`ffmpeg`), because a
/// checksum-correct archive that needs a newer system still cannot start.
pub(crate) const MACOS_FLOOR: (u32, u32) = (14, 0);
pub(crate) const GLIBC_FLOOR: (u32, u32) = (2, 35);
const WINDOWS_X86_64_FLOOR: u32 = 17_763;
const WINDOWS_ARM64_FLOOR: u32 = 22_000;

/// The opening of every refusal for a system with no published file for it — the
/// same words both install scripts begin theirs with.
const NO_RELEASE_FILE: &str = "MahBot has no file for this system";

/// The refusal for a Linux host with no glibc at all: a file built for glibc has
/// nothing to load against there. `install.sh` prints these words too.
const NO_GLIBC_FOUND: &str =
    "the released Linux files are built for glibc, and no glibc was found on this system";

/// The plain reason this system has no published release file for it, or `None`
/// when it has one: the floors above and, on Linux, the musl loader
/// ([`crate::util::managed_bin::linux_host_is_musl`]). Only what the host actually
/// answers is consulted — one that does not answer is not refused — and the install
/// scripts refuse the same systems, so no file is ever put in place that cannot load.
async fn absent_release_file() -> Option<String> {
    if cfg!(target_os = "linux") {
        let report = host_probe("getconf", &["GNU_LIBC_VERSION"]).await;
        let musl_loader = crate::util::managed_bin::linux_host_is_musl();
        if let Some(refusal) = linux_refusal(report.as_deref(), musl_loader) {
            return Some(match refusal {
                LinuxRefusal::GlibcBelowFloor => format!(
                    "{NO_RELEASE_FILE}: Linux with glibc {}.{} or newer is required",
                    GLIBC_FLOOR.0, GLIBC_FLOOR.1
                ),
                LinuxRefusal::NoGlibc => format!("{NO_RELEASE_FILE}: {NO_GLIBC_FOUND}"),
            });
        }
    }
    if cfg!(target_os = "macos") {
        let version = host_probe("sw_vers", &["-productVersion"]).await?;
        if version_below_floor(&version, MACOS_FLOOR) {
            return Some(format!(
                "{NO_RELEASE_FILE}: macOS {}.{} or newer is required",
                MACOS_FLOOR.0, MACOS_FLOOR.1
            ));
        }
    }
    if cfg!(target_os = "windows") {
        let (floor, requirement) = if cfg!(target_arch = "aarch64") {
            // Every Windows 10 on ARM — below Windows 11's own first build — is below
            // the ARM64 file.
            (WINDOWS_ARM64_FLOOR, "Windows 11 on ARM")
        } else {
            (WINDOWS_X86_64_FLOOR, "Windows 10 version 1809")
        };
        if windows_build()? < floor {
            return Some(format!(
                "{NO_RELEASE_FILE}: {requirement} (build {floor}) or newer is required"
            ));
        }
    }
    None
}

/// The Windows build this host runs, when it can be read: `RtlGetVersion`, not
/// `GetVersionExW`, which reports Windows 8's version for an image carrying no
/// compatibility manifest (this one does not), and every published Windows file is
/// newer than that. `None` when it cannot be read — nothing is refused on a guess.
#[cfg(windows)]
fn windows_build() -> Option<u32> {
    use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
    use windows_sys::Win32::System::SystemInformation::OSVERSIONINFOW;

    // SAFETY: every field is an integer or a fixed array of them, so all-zero is a
    // valid value; the size field below is what the call reads.
    let mut info = unsafe { std::mem::zeroed::<OSVERSIONINFOW>() };
    info.dwOSVersionInfoSize =
        u32::try_from(std::mem::size_of_val(&info)).expect("a struct of a few words");
    // SAFETY: `info` is a live, zeroed, correctly sized `OSVERSIONINFOW`, and the
    // call fills exactly the words its size field describes.
    let status = unsafe { RtlGetVersion(&raw mut info) };
    (status == 0 && info.dwBuildNumber != 0).then_some(info.dwBuildNumber)
}

/// Not Windows: there is no Windows build here to read.
#[cfg(not(windows))]
fn windows_build() -> Option<u32> {
    None
}

/// Ask the host what it is the way the install scripts ask it (`sw_vers`,
/// `getconf`): the trimmed stdout when it exits successfully, and nothing at all
/// otherwise — including when it does not answer in time.
async fn host_probe(program: &str, args: &[&str]) -> Option<String> {
    let mut cmd = tokio::process::Command::new(program);
    #[cfg(windows)]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    let output = tokio::time::timeout(HOST_PROBE_TIMEOUT, cmd.args(args).output())
        .await
        .ok()?
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (output.status.success() && !text.is_empty()).then_some(text)
}

/// Whether `version` (`major[.minor[…]]`) is below `floor`. Parts are whole numbers —
/// a minor of 4 is not four tenths of 2.35 — a missing minor counts as zero, and a
/// part that is not a whole number is a version this cannot compare: below the floor,
/// as the install script's own check refuses it.
fn version_below_floor(version: &str, floor: (u32, u32)) -> bool {
    let mut parts = version.trim().split('.');
    let major = parts.next().and_then(|part| part.parse::<u32>().ok());
    let minor = match parts.next() {
        Some(part) => part.parse::<u32>().ok(),
        None => Some(0),
    };
    match (major, minor) {
        (Some(major), Some(minor)) => (major, minor) < floor,
        _ => true,
    }
}

/// Whether a `getconf GNU_LIBC_VERSION` report names glibc at all: another libc's
/// report names none, and neither does no report.
fn names_glibc(report: &str) -> bool {
    report.starts_with("glibc ")
}

/// Which of the two ways a Linux host has no published file for it its own answers
/// amount to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinuxRefusal {
    GlibcBelowFloor,
    NoGlibc,
}

/// The refusal a Linux host's own answers amount to, or `None` when the published
/// file is one it can load: a glibc at or above [`GLIBC_FLOOR`] loads, one below it
/// does not. A host naming no glibc — another libc's report, or no answer at all — is
/// judged by musl's loader being there, the check the vendor installers make: musl
/// beside glibc is not a musl system. A host with neither is not refused.
fn linux_refusal(report: Option<&str>, musl_loader: bool) -> Option<LinuxRefusal> {
    if report.is_some_and(glibc_below_floor) {
        return Some(LinuxRefusal::GlibcBelowFloor);
    }
    (!report.is_some_and(names_glibc) && musl_loader).then_some(LinuxRefusal::NoGlibc)
}

/// Whether a `getconf GNU_LIBC_VERSION` report names a glibc below the floor. A
/// report this does not recognise is not a refusal.
fn glibc_below_floor(report: &str) -> bool {
    report
        .strip_prefix("glibc ")
        .is_some_and(|version| version_below_floor(version, GLIBC_FLOOR))
}

fn current_version() -> Result<semver::Version> {
    semver::Version::parse(VERSION)
        .with_context(|| format!("embedded version {VERSION} is not valid semver"))
}

/// The version a driven update was told to move to, when it named one other than the
/// running version — `None` in every ordinary run. Read by both the availability
/// check and the update itself, so a driven update cannot be shown one version and
/// given another. Taken in either spelling: the release's mark carries the `v`
/// (`v0.7.0`), the version itself and every URL built from it do not.
#[must_use]
fn update_target_override() -> Option<semver::Version> {
    let named = std::env::var(UPDATE_TO_VERSION_ENV).ok()?;
    let named = named.trim();
    let named = semver::Version::parse(named.strip_prefix('v').unwrap_or(named)).ok()?;
    let current = semver::Version::parse(VERSION).ok()?;
    (named != current).then_some(named)
}

/// HTTP client for the release check: a descriptive User-Agent and a short timeout so
/// a hung network never stalls the refresh loop for long. A build failure surfaces as
/// `Err`; both callers tolerate it and report it, and the periodic one retries on its
/// next tick.
fn release_http_client() -> Result<&'static reqwest::Client> {
    /// Short: this is a check, and the tick retries anyway.
    const CHECK_TIMEOUT: Duration = Duration::from_secs(15);
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            crate::util::http::install_ring_provider();
            reqwest::Client::builder()
                .user_agent(format!("mahbot/{VERSION} (self-update check)"))
                .timeout(CHECK_TIMEOUT)
                .build()
                .map_err(|e| format!("failed to build the release-check HTTP client: {e}"))
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

/// The newest published version held by the release host's version file. A version
/// carrying a pre-release part is ignored (`Ok(None)`): a test release is published
/// that way and must never be discovered by an ordinary copy. A malformed body and a
/// network failure both surface as `Err`, which callers tolerate silently.
async fn fetch_latest_release_version() -> Result<Option<semver::Version>> {
    let url = latest_version_url(&release_base_url());
    let response = release_http_client()?
        .get(&url)
        .send()
        .await
        .context("failed to read the newest release version")?;
    if !response.status().is_success() {
        anyhow::bail!(
            "the newest release version request returned HTTP {}",
            response.status()
        );
    }
    let body = response
        .text()
        .await
        .context("failed to read the newest release version")?;
    discoverable_version(&body)
}

/// The version a version file's body names, or `None` when it names a test release: a
/// pre-release is never offered to an ordinary copy, which keeps test releases from
/// reaching anybody who did not ask for one (a driven update names it outright
/// through [`UPDATE_TO_VERSION_ENV`]).
fn discoverable_version(body: &str) -> Result<Option<semver::Version>> {
    let version = semver::Version::parse(body.trim())
        .context("the newest release version file does not hold a version")?;
    Ok(version.pre.is_empty().then_some(version))
}

/// Check the product's own release host for a newer version of itself: `Ok(Some)` when
/// one is published, `Ok(None)` when up to date, `Err` on a failed check (the caller
/// retries later — a failed check is never a user-visible error). A version named
/// through [`UPDATE_TO_VERSION_ENV`] is reported as-is, with no lookup at all.
async fn check_download_update() -> Result<Option<semver::Version>> {
    if let Some(named) = update_target_override() {
        return Ok(Some(named));
    }
    let Some(latest) = fetch_latest_release_version().await? else {
        return Ok(None);
    };
    Ok((latest > current_version()?).then_some(latest))
}

/// How the running binary was obtained — selects the self-update strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpdateMode {
    SourceTree,
    Downloaded,
}

/// Classify the update mode from a manifest-directory probe (`CARGO_MANIFEST_DIR`).
///
/// The probe cannot simply look for `Cargo.toml`: the package registry extracts its
/// crates into `$CARGO_HOME/registry/src` and `$CARGO_HOME/git/checkouts`, both of
/// which contain one. So: the registry's source-cache layout (`registry/src` or
/// `git/checkouts` as path segments; CARGO_HOME may be custom, so no `.cargo` prefix
/// is required) means a downloaded copy, and is checked *first* because cargo's
/// `git/checkouts` entries are full non-bare clones with a real `.git` directory;
/// then `.git` at the manifest dir, or a reachable `Cargo.toml` (a source-tarball
/// extract or a moved checkout still rebuilds fine), means a working tree; with no
/// source at all the build source is gone, so the ready-made file is the only way forward.
fn classify_update_mode(manifest_dir: &Path) -> UpdateMode {
    if is_registry_source_cache(manifest_dir) {
        return UpdateMode::Downloaded;
    }
    if manifest_dir.join(".git").exists() {
        return UpdateMode::SourceTree;
    }
    if manifest_dir.join("Cargo.toml").is_file() {
        UpdateMode::SourceTree
    } else {
        UpdateMode::Downloaded
    }
}

/// Whether `manifest_dir` is the package registry's own source cache — a build from
/// source whose update strategy is nonetheless "download". The layout is matched as
/// adjacent path segments, separator-agnostic (Windows uses backslashes).
fn is_registry_source_cache(manifest_dir: &Path) -> bool {
    let mut prev: Option<&std::ffi::OsStr> = None;
    for component in manifest_dir.components() {
        if let (Some(a), std::path::Component::Normal(b)) = (prev, component)
            && ((a == "registry" && b == "src") || (a == "git" && b == "checkouts"))
        {
            return true;
        }
        prev = match component {
            std::path::Component::Normal(os) => Some(os),
            _ => None,
        };
    }
    false
}

/// Whether this binary is one of the product's ready-made published files. A
/// checkout, a source-tarball extract and the package registry's source cache all
/// keep a build source and none was delivered by the release pipeline — even though
/// the registry cache updates by downloading. Bound once: the Settings page asks on
/// every frame and the answer cannot change while the process runs.
pub(crate) fn is_published_copy() -> bool {
    static PUBLISHED: OnceLock<bool> = OnceLock::new();
    *PUBLISHED.get_or_init(|| is_published_copy_at(Path::new(env!("CARGO_MANIFEST_DIR"))))
}

/// [`is_published_copy`] for a manifest-directory probe: the mode says the file was
/// downloaded, minus the one source build that also downloads — the registry cache.
fn is_published_copy_at(manifest_dir: &Path) -> bool {
    classify_update_mode(manifest_dir) == UpdateMode::Downloaded
        && !is_registry_source_cache(manifest_dir)
}

/// Determine how the running binary was installed — see [`UpdateMode`]. Bound once, for the same
/// reason as [`is_published_copy`].
#[must_use]
pub(crate) fn update_mode() -> UpdateMode {
    static MODE: OnceLock<UpdateMode> = OnceLock::new();
    *MODE.get_or_init(|| classify_update_mode(Path::new(env!("CARGO_MANIFEST_DIR"))))
}

/// A point-in-time snapshot of the shared availability cache, passed by value to pure
/// predicates like [`should_show_update`] so they read a consistent view instead of
/// observing intermediate writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UpdateAvailability {
    /// Mode-aware: always true for a source copy, true for a downloaded one only once
    /// the release host has a newer version.
    pub(crate) available: bool,
    pub(crate) in_progress: bool,
}

/// Process-local, single-source-of-truth cache of self-update availability and
/// progress, read by the GUI (button, tooltip, finalize guards) and the Telegram menu
/// so the surfaces cannot diverge. Reset implicitly on boot, so an already-installed
/// update is never advertised across a restart and a release whose unattended attempt
/// failed is tried once more by the fresh process.
struct UpdateCache {
    /// Seeded by [`update_cache`], then written by [`refresh_update_cache`], which
    /// never touches `in_progress`.
    available: AtomicBool,
    /// The newest published version a downloaded copy found — the GUI's "Update MahBot
    /// to vX" tooltip. `None` for a source copy, and for a downloaded one that is up to
    /// date or not yet checked.
    latest: std::sync::Mutex<Option<semver::Version>>,
    /// Claimed atomically by the triggers (see [`claim_update_start`]), kept set by
    /// [`execute_update`] and cleared by it on failure; a successful update exits.
    in_progress: AtomicBool,
    /// The release whose own unattended attempt failed, written by that trigger alone:
    /// no later check attempts it again nor anything not strictly newer, which keeps a
    /// release that downloads but does not install from re-draining the product on
    /// every tick (see [`auto_update_target`]).
    auto_failed: std::sync::Mutex<Option<semver::Version>>,
}

static UPDATE_CACHE: OnceLock<UpdateCache> = OnceLock::new();

fn update_cache() -> &'static UpdateCache {
    UPDATE_CACHE.get_or_init(|| UpdateCache {
        available: AtomicBool::new(update_mode() == UpdateMode::SourceTree),
        latest: std::sync::Mutex::new(None),
        in_progress: AtomicBool::new(false),
        auto_failed: std::sync::Mutex::new(None),
    })
}

#[must_use]
pub(crate) fn update_availability() -> UpdateAvailability {
    let cache = update_cache();
    UpdateAvailability {
        available: cache.available.load(Ordering::SeqCst),
        in_progress: cache.in_progress.load(Ordering::SeqCst),
    }
}

/// The newest published version a downloaded copy found (see
/// [`refresh_update_cache`]).
#[must_use]
pub(crate) fn update_latest() -> Option<semver::Version> {
    update_cache().latest.lock().unwrap_poison().clone()
}

/// Whether an update is currently in flight (download/build running, the finalizing
/// window inclusive).
#[must_use]
pub(crate) fn update_in_progress() -> bool {
    update_availability().in_progress
}

/// The release whose own unattended attempt failed, if any (see
/// [`auto_update_target`]): `None` for every release a *manual* attempt failed for —
/// only the unattended trigger's own failures are remembered.
#[must_use]
fn auto_failed_version() -> Option<semver::Version> {
    update_cache().auto_failed.lock().unwrap_poison().clone()
}

/// Remember the release the unattended trigger's own attempt failed for, so no later
/// check in this process retries it (see [`auto_update_target`]).
fn record_auto_update_failure(version: &semver::Version) {
    *update_cache().auto_failed.lock().unwrap_poison() = Some(version.clone());
}

/// Claim the shared in-progress flag for an update about to run, `false` when one
/// already runs. Every trigger claims here before [`execute_update`], so an update can
/// never start twice — neither from two of the same kind nor from two different ones —
/// and the loser stands down instead of reporting the collision [`execute_update`]
/// would return. [`execute_update`] keeps the flag set and clears it on failure.
pub(crate) fn claim_update_start() -> bool {
    update_cache()
        .in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

/// Pure visibility predicate for the `/update` menu entry: offered when the cached
/// availability says so. The menu reflects the cache and issues no request per
/// refresh; the GUI button reads [`update_availability`] directly.
#[must_use]
pub(crate) fn should_show_update(availability: UpdateAvailability) -> bool {
    availability.available
}

/// RAII guard from [`set_update_cache_for_test`]: restores the prior state on drop, so
/// a panicking test cannot leak a mutation into later tests.
#[cfg(test)]
pub(crate) struct UpdateCacheTestGuard {
    previous: UpdateAvailability,
}

#[cfg(test)]
impl Drop for UpdateCacheTestGuard {
    fn drop(&mut self) {
        let cache = update_cache();
        cache
            .available
            .store(self.previous.available, Ordering::SeqCst);
        cache
            .in_progress
            .store(self.previous.in_progress, Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(crate) fn set_update_cache_for_test(
    available: bool,
    in_progress: bool,
) -> UpdateCacheTestGuard {
    let previous = update_availability();
    let cache = update_cache();
    cache.available.store(available, Ordering::SeqCst);
    cache.in_progress.store(in_progress, Ordering::SeqCst);
    UpdateCacheTestGuard { previous }
}

/// Write the release check's finding into the shared cache and tell the GUI when it
/// changed — the update button must appear when the check finds a release, not at the
/// next unrelated repaint.
fn set_availability(latest: Option<semver::Version>) {
    let cache = update_cache();
    let available = latest.is_some();
    let changed = {
        let mut current = cache.latest.lock().unwrap_poison();
        let changed = *current != latest;
        if changed {
            *current = latest;
        }
        changed
    };
    if changed {
        cache.available.store(available, Ordering::SeqCst);
        crate::runtime_events::publish(crate::runtime_events::RuntimeEvent::UpdateAvailability);
    }
}

/// Refresh the shared update-availability cache once. A source copy seeds
/// `available = true` (idempotent, no network). A downloaded copy runs the release
/// check, keeping the last-known availability on a transient failure so a network blip
/// never hides a discovered update. While an update is in flight the check is skipped
/// outright — a long download would otherwise waste a request per tick — and
/// `in_progress` is re-checked after the await so the result cannot clobber it.
async fn refresh_update_cache() {
    let cache = update_cache();
    match update_mode() {
        UpdateMode::SourceTree => {
            cache.available.store(true, Ordering::SeqCst);
        }
        UpdateMode::Downloaded => {
            if cache.in_progress.load(Ordering::SeqCst) {
                return;
            }
            let result = check_download_update().await;
            if cache.in_progress.load(Ordering::SeqCst) {
                return;
            }
            match result {
                Ok(Some(latest)) => set_availability(Some(latest)),
                Ok(None) => set_availability(None),
                Err(_) => {
                    // Transient failure — preserve last-known availability.
                }
            }
        }
    }
}

/// Wakes [`run_update_availability_refresh`] for a tick now. A nudge arriving before
/// the loop waits is kept until it does.
static TICK_NOW: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// Run an availability tick now instead of at the next interval: what re-enabling the
/// Autoupdate switch does, so its promise holds without a restart.
pub(crate) fn nudge_update_tick() {
    TICK_NOW.notify_one();
}

/// Periodic refresh of the shared update-availability cache, and the unattended update
/// of a downloaded copy. Spawned from the background task set (cancellable via the
/// global shutdown token), it ticks at once — so a fresh downloaded copy is not hidden
/// until the first interval — then every 10 minutes, or at once when
/// [`nudge_update_tick`] asks. Every tick is also what gives
/// [`start_automatic_update_if_found`] the finding it decides on.
pub async fn run_update_availability_refresh() {
    loop {
        refresh_update_cache().await;
        start_automatic_update_if_found().await;
        tokio::select! {
            () = TICK_NOW.notified() => {}
            alive = crate::shutdown::sleep_or_shutdown_or_drain(Duration::from_mins(10)) => {
                if !alive {
                    return;
                }
            }
        }
    }
}

/// How long an unattended update waits for the product to come fully up — the stores,
/// the channels and the notification path — so an early failure is reportable.
const UNATTENDED_SETTLE: Duration = Duration::from_secs(15);

/// The release this copy must move to by itself, if any — the whole rule of the
/// unattended trigger, pure so it can be read and tested as one thing.
///
/// Never a trigger: a copy built from a working tree ([`UpdateMode::SourceTree`]),
/// since installing there rebuilds that tree — possibly from half-finished edits, and
/// again on every check while the tree is there; a version named through
/// [`UPDATE_TO_VERSION_ENV`] (`named`), which is the hook's own update and must not be
/// raced; a release the availability flag offers that is not newer than the running
/// version, the flag being permanently true on a source copy; and a release no newer
/// than one whose own attempt already failed (`failed`), which keeps a release that
/// downloads but does not install from re-draining the product on every tick.
fn auto_update_target(
    mode: UpdateMode,
    named: bool,
    latest: Option<&semver::Version>,
    current: &semver::Version,
    failed: Option<&semver::Version>,
) -> Option<semver::Version> {
    if mode != UpdateMode::Downloaded || named {
        return None;
    }
    let latest = latest.filter(|&latest| latest > current)?;
    (failed.is_none_or(|failed| latest > failed)).then(|| latest.clone())
}

/// The owner's Autoupdate switch (see `crate::config`): absence of the row means on.
fn autoupdate_enabled() -> bool {
    crate::config::flag_on(crate::config::CONFIG.autoupdate().as_deref())
}

/// The unattended trigger: run the release check's finding by itself, after every
/// availability refresh. The update it starts is the ordinary one in full — same
/// download, drain, checkpoint, restart and recovery — nobody pressed anything.
async fn start_automatic_update_if_found() {
    // Off means nothing installs itself; the check keeps running, so the button still
    // reports the release, and an update in flight is never cancelled here.
    if !autoupdate_enabled() {
        return;
    }
    if update_in_progress() {
        return;
    }
    let Ok(current) = current_version() else {
        return;
    };
    let Some(target) = auto_update_target(
        update_mode(),
        update_target_override().is_some(),
        update_latest().as_ref(),
        &current,
        auto_failed_version().as_ref(),
    ) else {
        return;
    };

    // The product must be fully up before an unattended attempt starts, so an early
    // failure is reportable rather than lost.
    if !crate::shutdown::sleep_or_shutdown_or_drain(UNATTENDED_SETTLE).await {
        return;
    }
    // Losing the claim to an update that started meanwhile is not a failure: nothing
    // is reported and the release is not remembered as failed.
    if !claim_update_start() {
        return;
    }
    // Detached, like the environment-driven hook: the update owns the drain hand-off
    // and must outlive the shutdown token it triggers itself, while this loop is
    // cancelled by exactly that token. The attempt is pinned to `target`, and
    // [`execute_update`] remembers that release as failed before clearing the
    // in-progress flag, so the next check can neither retry it nor start another
    // attempt in between.
    tokio::spawn(async move {
        if let Err(e) = execute_update(Some(&target)).await {
            error!(error = %e, "The automatic update did not complete");
            report_update_failure(&e).await;
        }
    });
}

/// Global mutex ensuring only one update runs at a time: a second trigger gets an
/// immediate error via [`try_lock`](Mutex::try_lock).
static UPDATE_MUTEX: Mutex<()> = Mutex::const_new(());

/// The build from this working tree is done; the restart is reported by
/// [`UPDATE_RESTART_MSG`].
pub(crate) const UPDATE_BUILD_COMPLETE_MSG: &str = "✅ Build complete.";

/// The downloaded file is in place; the restart is reported by
/// [`UPDATE_RESTART_MSG`].
pub(crate) const UPDATE_DOWNLOAD_COMPLETE_MSG: &str = "✅ Update downloaded and put in place.";

/// Sent as the restart phase begins — nothing has restarted yet when this goes out.
pub(crate) const UPDATE_RESTART_MSG: &str =
    "🔄 Waiting for work in progress to finish before the restart…";

/// `config_kv` keys of the update notice's durable state. Owned by this subsystem —
/// preserved by the config store's orphan purge.
pub(crate) const UPDATE_LAST_RUN_VERSION_KV_KEY: &str = "update_last_run_version";
pub(crate) const UPDATE_PENDING_CHANGELOG_KV_KEY: &str = "update_pending_changelog";

/// The change list captured during an update, waiting for the first start of the
/// version it belongs to.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PendingChangelog {
    version: String,
    /// `None` when the update could not take the list down: the start asks the release
    /// for it again while composing the notice.
    changes: Option<String>,
}

/// What a start must do about the update notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateNotice {
    Unchanged,
    RecordSilently,
    Announce,
}

/// The whole rule of which starts produce an update notice.
///
/// A restart of the same version says nothing — unless the update left a list for it,
/// so a start that stopped between composing the notice and handing it over is
/// repeated rather than silenced. A build from source never announces, whatever its
/// version string says, and keeps what the update left for a copy that can show it. An
/// installation that has not finished setup records a first install, because there is
/// no assistant to tell, but an update it was *given* waits — neither version nor list
/// is touched, so the first start after setup announces it. A published copy with
/// finished setup announces, including over a version record an older build wrote.
fn update_notice_action(
    last_run: Option<&str>,
    current: &str,
    published: bool,
    onboarding_finished: bool,
    owed_list: bool,
) -> UpdateNotice {
    if !published {
        return if last_run == Some(current) {
            UpdateNotice::Unchanged
        } else {
            UpdateNotice::RecordSilently
        };
    }
    if owed_list {
        return if onboarding_finished {
            UpdateNotice::Announce
        } else {
            UpdateNotice::Unchanged
        };
    }
    if last_run == Some(current) {
        UpdateNotice::Unchanged
    } else if onboarding_finished {
        UpdateNotice::Announce
    } else {
        UpdateNotice::RecordSilently
    }
}

/// Write the version this start runs from, reporting a failed write rather than hiding
/// it: the caller decides whether its own contract can go on.
async fn record_notice_state(store: &crate::config_db::ConfigStore, version: &str) -> bool {
    match store.set_kv(UPDATE_LAST_RUN_VERSION_KV_KEY, version).await {
        Ok(()) => true,
        Err(e) => {
            warn!(error = %e, "Could not record the update notice state");
            false
        }
    }
}

/// The change list captured for `version`, when the record belongs to it: an
/// unparseable one is the only case treated as absent.
fn pending_changelog_from_str(raw: &str, version: &str) -> Option<PendingChangelog> {
    let pending: PendingChangelog = match serde_json::from_str(raw) {
        Ok(pending) => pending,
        Err(e) => {
            warn!(error = %e, "The captured update change list could not be read");
            return None;
        }
    };
    (pending.version == version).then_some(pending)
}

/// The change list captured for `version`, when the record belongs to it; one of
/// another version is left in place for a release that may still start. A failed read
/// is an error, so the caller settles nothing on this start.
async fn pending_changelog(
    store: &crate::config_db::ConfigStore,
    version: &str,
) -> Result<Option<PendingChangelog>> {
    let Some(raw) = store.get_kv(UPDATE_PENDING_CHANGELOG_KV_KEY).await? else {
        return Ok(None);
    };
    Ok(pending_changelog_from_str(&raw, version))
}

/// The notice itself: one self-contained, tagged block carrying the version and the
/// release's change list, or — with no list to carry — the plain statement that it is
/// unavailable and the release page it can be read on.
fn update_notice_text(version: &str, changes: Option<&str>, release_page: &str) -> String {
    match changes {
        Some(changes) => crate::prompt::substitute(
            &crate::prompt::load_prompt("update_notice.md"),
            &[("{{version}}", version), ("{{changes}}", changes)],
        ),
        None => crate::prompt::substitute(
            &crate::prompt::load_prompt("update_notice_unavailable.md"),
            &[("{{version}}", version), ("{{release_page}}", release_page)],
        ),
    }
}

/// Tell the admin's assistant once that the product was updated to this version,
/// carrying that release's change list. Called once at daemon start, after the message
/// pipeline is up: the notice enters the assistant's ordinary queue exactly as a
/// message from the owner does. See [`settle_update_notice`] for what this start may
/// and may not settle.
pub async fn announce_update_to_assistant() {
    settle_update_notice(
        crate::config_db::store(),
        VERSION,
        is_published_copy(),
        crate::config::CONFIG.onboarding_stage() == crate::config::OnboardingState::Finished,
        changelog_at_notice_time,
        |content| {
            crate::agent::message_router::post_assistant_notice(
                &crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME),
                crate::users::ADMIN_USER_NAME,
                content,
            );
        },
    )
    .await;
}

/// One start's whole notice flow: read the durable state, decide what this start owes,
/// obtain the change list the update did not bring, hand the notice over, and only then
/// record the version and drop the list. Store, fetch and delivery come from the caller
/// so the flow is exercisable without the global store, the network and the queue.
async fn settle_update_notice(
    store: &crate::config_db::ConfigStore,
    version: &str,
    published: bool,
    onboarding_finished: bool,
    fetch_changes: impl AsyncFnOnce(&str) -> Option<String>,
    deliver: impl FnOnce(String),
) {
    // Both reads settle what this start owes, so a failed one settles nothing: nothing
    // recorded, nothing dropped, and the next start decides again.
    let last_run = match store.get_kv(UPDATE_LAST_RUN_VERSION_KV_KEY).await {
        Ok(last_run) => last_run,
        Err(e) => {
            warn!(error = %e, "Could not read the update notice state — no notice is produced");
            return;
        }
    };
    let pending = match pending_changelog(store, version).await {
        Ok(pending) => pending,
        Err(e) => {
            warn!(
                error = %e,
                "Could not read the captured update change list — no notice is produced"
            );
            return;
        }
    };
    match update_notice_action(
        last_run.as_deref(),
        version,
        published,
        onboarding_finished,
        pending.is_some(),
    ) {
        UpdateNotice::Unchanged => {}
        UpdateNotice::RecordSilently => {
            // Its outcome is only load-bearing where a notice follows; a failed write
            // is retried at the next start.
            record_notice_state(store, version).await;
        }
        UpdateNotice::Announce => {
            let release_page = release_page_url(&release_base_url(), version);
            // Only a record belonging to this version is spent at the end.
            let captured = pending.is_some();
            let changes = match pending.and_then(|pending| pending.changes) {
                Some(changes) => Some(changes),
                None => fetch_changes(version).await,
            };
            if changes.is_none() {
                warn!(
                    version,
                    release_page,
                    "The release's change list could not be obtained — the notice points at the release page"
                );
            }
            deliver(update_notice_text(
                version,
                changes.as_deref(),
                &release_page,
            ));
            info!(version, "Told the admin assistant about the update");
            // The once-only record goes down only after the hand-over, and the list
            // only after the record: a start stopping in between repeats the notice,
            // the lesser fault than silence.
            if !record_notice_state(store, version).await {
                return;
            }
            if captured && let Err(e) = store.delete_kv(UPDATE_PENDING_CHANGELOG_KV_KEY).await {
                warn!(error = %e, "Could not clear the captured update change list");
            }
        }
    }
}

/// The release's change list, read while the notice is composed: the second and last
/// time it is asked for, so a fetch that failed before the restart does not decide what
/// the notice says.
async fn changelog_at_notice_time(version: &str) -> Option<String> {
    // The client's own deadline is the update transport's generous one; what bounds
    // this read is [`CHANGELOG_TIMEOUT`] inside the fetch.
    let client = match crate::util::http::build_download_client(DOWNLOAD_TIMEOUT) {
        Ok(client) => client,
        Err(e) => {
            warn!(error = %e, "Could not build the client that reads the release's change list");
            return None;
        }
    };
    fetch_release_changelog(&client, version).await
}

/// The URL of a release's change list — the release body's own text, published as an
/// asset so an update can take it down before the restart and a start that has none can
/// still read it.
fn changelog_url(base: &str, version: &str) -> String {
    format!("{base}/releases/download/v{version}/changelog.txt")
}

/// The release page of `version` — where the same change list is published for a person
/// to read, and what the notice names when it cannot carry the list.
fn release_page_url(base: &str, version: &str) -> String {
    format!("{base}/releases/tag/v{version}")
}

/// How long the release's change list may take before the update or the notice goes on
/// without it: the real cap on both attempts, since the client outlives it.
const CHANGELOG_TIMEOUT: Duration = Duration::from_secs(5);

/// Read a release's change list, if it carries one. A missing asset (a release
/// published before the list travelled with the build) and an empty one are both
/// `None`, and nothing but `None` is remembered, so a failure here never marks the list
/// unavailable for good.
async fn fetch_release_changelog(client: &reqwest::Client, version: &str) -> Option<String> {
    let url = changelog_url(&release_base_url(), version);
    let fetch = async {
        let response = match client.get(&url).send().await {
            Ok(response) => response,
            Err(e) => {
                info!(url = %url, error = %e, "The release's change list could not be fetched");
                return None;
            }
        };
        if !response.status().is_success() {
            info!(url = %url, status = %response.status(), "The release carries no change list");
            return None;
        }
        let body = match response.text().await {
            Ok(body) => body,
            Err(e) => {
                info!(url = %url, error = %e, "The release's change list could not be read");
                return None;
            }
        };
        let body = body.trim();
        (!body.is_empty()).then(|| body.to_string())
    };
    if let Ok(changes) = tokio::time::timeout(CHANGELOG_TIMEOUT, fetch).await {
        changes
    } else {
        info!(url = %url, "The release's change list took too long to fetch");
        None
    }
}

/// Record the release's change list for the first start of `version`, while the update
/// still runs: that start may be offline, and would otherwise announce "unavailable"
/// while the list exists. Only the download path captures, and the file it installs is
/// a published copy, so the notice this record waits for will actually be sent.
async fn capture_update_announcement(
    store: &crate::config_db::ConfigStore,
    version: &str,
    changes: Option<String>,
) {
    // A list already recorded for this version is never replaced, and an unreadable
    // record is never written over: a list that came down the first time must not be
    // lost to a later attempt whose fetch came back empty.
    match pending_changelog(store, version).await {
        Ok(Some(pending)) if pending.changes.is_some() => {
            debug!(version, "The release's change list is already captured");
            return;
        }
        Ok(_) => {}
        Err(e) => {
            warn!(error = %e, "Could not read the captured update change list — it is left alone");
            return;
        }
    }
    let pending = PendingChangelog {
        version: version.to_string(),
        changes,
    };
    let value = match serde_json::to_string(&pending) {
        Ok(value) => value,
        Err(e) => {
            warn!(error = %e, "Could not encode the update change list");
            return;
        }
    };
    if let Err(e) = store.set_kv(UPDATE_PENDING_CHANGELOG_KV_KEY, &value).await {
        warn!(error = %e, "Could not record the update change list");
    }
}

/// The target every partly-finished update is recorded under, and the message it is
/// recorded with. Durable rather than a tracing line because the process restarts
/// moments later, and one row per distinct reason so it never repeats.
const UPDATE_UNFINISHED_TARGET: &str = "mahbot::self_update";
const UPDATE_UNFINISHED_MESSAGE: &str = "the update could not be completed in full";

/// A step of an update that could not be done, as it is recorded durably and told to
/// the admin.
struct UnfinishedStep {
    /// The whole of what is recorded — never a path or anything taken from the owner's environment.
    reason: &'static str,
    /// A step that stops the update is already reported by the ordinary failure line, so
    /// its record raises no second sentence about the same fact; a step the update goes
    /// on past would otherwise be told nowhere.
    stops_update: bool,
}

/// The standard per-user programs directory could not be written to, so the new version
/// could not be put where the install scripts put it. The update stops here and the copy
/// already there keeps working.
const STANDARD_DIR_UNUSABLE: UnfinishedStep = UnfinishedStep {
    reason: "the standard per-user programs directory could not be written to",
    stops_update: true,
};

/// A second copy of the product — the one this instance came from when it moved itself,
/// or the one the old way of installing left in the toolchain's directory — could not be
/// removed. One reason for both, because they are the same failing step on the same kind
/// of file; the update goes through, so this is the only word the admin gets about it.
const SECOND_COPY_REMAINS: UnfinishedStep = UnfinishedStep {
    reason: "a second copy of the product could not be taken away",
    stops_update: false,
};

/// Record a partly-finished update durably, and tell the admin about it once — when this
/// call is what recorded the row, and when the ordinary failure line does not already
/// name the same failing step (see [`UnfinishedStep::stops_update`]).
///
/// The record is the shared Issues-view one ([`crate::logs::record_issue_once`]), which
/// writes straight into the logs store — the update path calls `exit(0)`, so a
/// `tracing::warn!` alone would be lost — and never writes a reason the store already
/// holds. Which shape warns: a fact no row carries
/// ([`crate::logs::IssueWrite::NotRecorded`]), because this path exits right after.
/// Which notifies: only a row this call wrote; one the store already holds was told by
/// the run that recorded it.
async fn record_update_unfinished(admin_target: Option<&str>, step: UnfinishedStep) {
    let recorded = crate::logs::record_issue_once(
        UPDATE_UNFINISHED_MESSAGE,
        step.reason,
        UPDATE_UNFINISHED_TARGET,
        serde_json::json!({}),
    )
    .await;
    match recorded {
        // Fall through: the ordinary failure line does not mention the non-fatal steps.
        crate::logs::IssueWrite::Written => {}
        crate::logs::IssueWrite::AlreadyRecorded => return,
        // This path exits right after, and the shared writer's warning is about the
        // store, not about what was lost — so this line names the fact itself.
        crate::logs::IssueWrite::NotRecorded => {
            warn!(
                reason = step.reason,
                "an unfinished update could not be recorded in the Issues view"
            );
            return;
        }
    }
    // A fatal step adds nothing: the ordinary failure line already names it.
    if step.stops_update {
        return;
    }
    let message = format!("⚠️ {UPDATE_UNFINISHED_MESSAGE}: {}.", step.reason);
    notify_admin(&message, admin_target).await;
}

/// Set while [`execute_update`] runs its finalizing window (drain through `exit(0)`).
/// The update path owns the process then: the GUI exit path waits
/// ([`update_is_finalizing`]) instead of exiting, so a platform quit, a close or SIGINT
/// cannot abort the checkpoint and leave the instance down without a replacement.
static UPDATE_FINALIZING: AtomicBool = AtomicBool::new(false);

/// True while [`execute_update`] is finalizing: the instance is shut down, with the
/// checkpoint, temp-root cleanup, lock release, spawn and `exit(0)` pending.
#[must_use]
pub fn update_is_finalizing() -> bool {
    UPDATE_FINALIZING.load(Ordering::SeqCst)
}

/// Verify `cargo` is on PATH, with an error naming what it was needed for. The
/// build-from-sources path only: updating a downloaded copy never runs cargo.
async fn verify_cargo_on_path(action: &str) -> Result<()> {
    let mut cmd = tokio::process::Command::new("cargo");
    #[cfg(windows)]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    let status = cmd
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    match status {
        Ok(status) if status.success() => Ok(()),
        _ => anyhow::bail!("cargo not found on PATH — cannot {action}"),
    }
}

/// Resolve the admin Telegram reply target, logging why when notifications cannot be
/// sent. Per call — a DB round trip to find the admin and its channel bindings — and
/// shared by both update paths.
async fn resolve_update_admin_target() -> Option<String> {
    let admin_target = resolve_admin_telegram_target().await;
    if admin_target.is_none() {
        if crate::config::CONFIG.telegram_bot_token().is_none() {
            info!("No Telegram bot token configured — skipping update notifications");
        } else {
            warn!(
                "Admin account 'admin' has no Telegram channel binding with a reply_target. \
                 Update notifications will be skipped. \
                 Bind a Telegram channel to the admin to receive update notifications."
            );
        }
    }
    admin_target
}

/// Execute a self-update, dispatching on the update mode: a source tree builds into a
/// temp root, swaps and restarts; a downloaded copy fetches this system's release file,
/// puts it at the standard per-user location and restarts.
///
/// The GUI button, the Telegram `/update` command and the unattended trigger claim the
/// shared in-progress flag ([`claim_update_start`]) before they get here, so only one
/// update runs at a time; the environment-driven hook does not claim — it can never run
/// beside the unattended trigger, so `UPDATE_MUTEX` alone serializes it.
///
/// `auto_target` is the release the unattended trigger's own attempt is pinned to, and
/// on failure that same release is remembered as failed *before* the in-progress flag
/// is cleared. Every other caller passes `None`, so no check can start another attempt
/// for the release in between.
///
/// On success this never returns (`std::process::exit(0)`); on failure it returns an error.
pub(crate) async fn execute_update(auto_target: Option<&semver::Version>) -> Result<()> {
    let Some(_guard) = UPDATE_MUTEX.try_lock().ok() else {
        anyhow::bail!("{UPDATE_IN_PROGRESS_MSG}");
    };
    // Already claimed by the caller; set here too so the GUI and the `/update` gate
    // report the update and the exit-request/finalize guards key off it.
    update_cache().in_progress.store(true, Ordering::SeqCst);
    let result = match update_mode() {
        UpdateMode::SourceTree => execute_source_tree_update().await,
        UpdateMode::Downloaded => execute_downloaded_update(auto_target).await,
    };
    if result.is_err() {
        // Remembered while the flag is still held; a failed manual attempt never is.
        if let Some(target) = auto_target {
            record_auto_update_failure(target);
        }
        update_cache().in_progress.store(false, Ordering::SeqCst);
    }
    result
}

/// Drive a whole update with no window at all, when a version was named through
/// [`UPDATE_TO_VERSION_ENV`]: returns immediately unless that version differs from the
/// running one *and* this copy updates by downloading, then waits for the stores and
/// channels before running [`execute_update`], reporting a failure instead of
/// restarting anything. The hook terminates by itself — the replacement sees the named
/// version as its own and does nothing.
pub async fn run_env_named_update() {
    if update_target_override().is_none() || update_mode() != UpdateMode::Downloaded {
        return;
    }
    if !crate::shutdown::sleep_or_shutdown_or_drain(UNATTENDED_SETTLE).await {
        return;
    }
    if let Err(e) = execute_update(None).await {
        error!(error = %e, "The update to the version named in the environment did not complete");
    }
}

/// The build-from-sources self-update: `cargo install --path` into a temp root, swap the
/// running binary, restart. See [`execute_update`] for the concurrent-guard and exit contracts.
async fn execute_source_tree_update() -> Result<()> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = manifest_dir.join("Cargo.toml");
    if !cargo_toml.is_file() {
        anyhow::bail!(
            "Self-update is not available on this installation. \
             Cargo.toml not found at {}. \
             Self-update only works when running from the original build checkout directory.",
            cargo_toml.display()
        );
    }

    verify_cargo_on_path("build from source").await?;

    let admin_target = resolve_update_admin_target().await;

    notify_admin(
        "🔄 Update started — building from source…",
        admin_target.as_deref(),
    )
    .await;

    // Held in scope so an install/spawn failure RAII-cleans them; `exit(0)` on success
    // bypasses RAII, so the finalize tail removes them before the lock release and
    // spawn. The cold release rebuild is several-GB, and the temp roots land on the
    // boot volume with the repo, so no cross-volume fallback is warranted.
    let temp_install_root =
        tempfile::tempdir().context("Failed to create temp install root for self-update")?;
    let temp_build_dir =
        tempfile::tempdir().context("Failed to create temp build dir for self-update")?;
    let install_root = temp_install_root.path().to_path_buf();
    let build_dir = temp_build_dir.path().to_path_buf();

    // The build never touches the repo's `target/` (which would overwrite a running
    // `target/release` artifact), and cwd is the storage root — guaranteed writable,
    // unlike the launch dir.
    run_cargo_with_timeout(
        &[
            OsStr::new("install"),
            OsStr::new("--path"),
            manifest_dir.as_os_str(),
            OsStr::new("--root"),
            install_root.as_os_str(),
            OsStr::new("--locked"),
            OsStr::new("--target-dir"),
            build_dir.as_os_str(),
        ],
        &crate::config::CONFIG.global_storage_root(),
        Duration::from_hours(1),
        "cargo install --path --locked",
        &CargoStep {
            toast_head: "Build failed",
            admin_head: "Failed to build from source",
        },
    )
    .await?;

    let fresh_binary = temp_bin_path(&install_root);
    finalize_source_tree_update(
        &fresh_binary,
        admin_target.as_deref(),
        vec![install_root, build_dir],
    )
    .await
}

/// The downloaded-copy self-update: fetch this system's ready-made release file, extract
/// it, put it at the standard per-user location, and restart. See [`execute_update`] for
/// the concurrent-guard and exit contracts.
///
/// `pinned` is the release the unattended trigger's own attempt goes after, so the
/// release attempted and the release remembered on failure are the same one. Every other
/// caller passes `None`; the environment hook's named version comes from
/// [`update_target_override`] instead, and the two are never set together.
async fn execute_downloaded_update(pinned: Option<&semver::Version>) -> Result<()> {
    // A system with no published file for it is refused before anything happens: a file
    // that cannot load would leave no working copy, the one state an update must never
    // produce. The install scripts refuse the same systems.
    if let Some(reason) = absent_release_file().await {
        anyhow::bail!("{reason} — no update was made.");
    }

    // The pinned release is taken outright — the check that offered it proved it newer.
    // A version named through [`UPDATE_TO_VERSION_ENV`] is taken outright too, any
    // version being a deliberate step. The ordinary path reads the newest published
    // release and only ever moves forward, and nothing is installed when the version
    // cannot be determined.
    let target = if let Some(pinned) = pinned {
        pinned.clone()
    } else if let Some(named) = update_target_override() {
        named
    } else {
        let latest = fetch_latest_release_version().await?.ok_or_else(|| {
            anyhow!("The newest published MahBot version could not be determined.")
        })?;
        let current = current_version()?;
        anyhow::ensure!(
            latest > current,
            "The newest published MahBot version {latest} is not newer than the running \
             {current} — no update was made."
        );
        latest
    };

    // The admin every notification goes to, and the file the new version goes in place
    // of — or is placed at: the standard per-user programs location the install scripts
    // use. A location that cannot be resolved cannot be written to either, so it is the
    // same failing step as a placement that fails, and nothing is downloaded.
    let admin_target = resolve_update_admin_target().await;
    let Some(dest) = crate::util::managed_bin::mahbot_install_dir()
        .map(|dir| dir.join(crate::util::managed_bin::product_file_name()))
    else {
        record_update_unfinished(admin_target.as_deref(), STANDARD_DIR_UNUSABLE).await;
        anyhow::bail!("The standard install location could not be resolved — no update was made.");
    };

    notify_admin(
        "🔄 Update started — downloading the new version…",
        admin_target.as_deref(),
    )
    .await;

    // The temp dir is held in scope for the whole update (RAII on error; removed in the
    // finalize tail, before the lock release and spawn).
    let temp_dir = tempfile::tempdir().context("Failed to create temp dir for self-update")?;
    let (os, arch) = crate::util::managed_bin::host_os_arch().map_err(|e| anyhow!("{e}"))?;
    let name = asset_name(&target, os, arch);
    let url = asset_url(&release_base_url(), &target, os, arch);
    let version = target.to_string();
    let client = crate::util::http::build_download_client(DOWNLOAD_TIMEOUT)
        .context("Failed to build the download client for self-update")?;
    let archive = temp_dir.path().join(&name);
    // The change list comes down alongside the archive so the start after the restart —
    // which may be offline — still has it; its request is capped at [`CHANGELOG_TIMEOUT`],
    // so a slow or missing asset never costs the update itself.
    let (downloaded, changes) = tokio::join!(
        crate::util::http::download_verified(
            &client,
            &url,
            &archive,
            // The product's own file: published beside the version that names it, so
            // there is deliberately no checksum to verify it against.
            "",
            Some(DOWNLOAD_TIMEOUT),
            crate::util::http::DownloadSizeCheck::Exact,
            |_, _| {},
        ),
        fetch_release_changelog(&client, &version),
    );
    downloaded.with_context(|| format!("Failed to download {name}"))?;
    let extract = if os == "windows" {
        crate::util::managed_bin::extract_single_file_zip
    } else {
        crate::util::managed_bin::extract_single_file_tar_gz
    };
    let fresh = extract(
        &archive,
        temp_dir.path(),
        crate::util::managed_bin::product_file_name(),
    )
    .map_err(|e| anyhow!("Failed to extract {name}: {e}"))?;

    // When the running file IS the standard location the swap rewrites it in place and
    // the restart target is that same path; otherwise this copy "moves itself" there —
    // the new file is placed at the standard location, which becomes the restart target.
    let current_exe = std::env::current_exe().context("Failed to resolve current_exe()")?;
    let (spawn_path, relocated) = if canonicalize_safe(&current_exe) == canonicalize_safe(&dest) {
        // The running file IS the standard location, so the swap rewrites it in place. A swap that
        // fails leaves the copy that is there working.
        if let Err(e) = self_replace::self_replace(&fresh) {
            record_update_unfinished(admin_target.as_deref(), STANDARD_DIR_UNUSABLE).await;
            return Err(e).with_context(|| format!("Failed to swap binary at {}", fresh.display()));
        }
        (current_exe, false)
    } else {
        if let Err(reason) = crate::util::managed_bin::place_extracted(&fresh, &dest) {
            record_update_unfinished(admin_target.as_deref(), STANDARD_DIR_UNUSABLE).await;
            anyhow::bail!("Could not put the new version at the standard location: {reason}");
        }
        (dest.clone(), true)
    };

    // Take away the copies a working installation must not leave behind: the one this
    // instance came from when this copy moved itself, and the one `cargo install mahbot`
    // left in the toolchain's directory — retried at every update, in case an earlier one
    // failed. The running image needs the platform's own removal: `self_delete` unlinks
    // it on unix and, on Windows, renames it aside and schedules the deletion. A leftover
    // is recorded durably, never fatal, and logged below the owner's view since the
    // failure carries the platform's own path and all.
    if relocated && let Err(e) = self_replace::self_delete() {
        debug!(error = %e, "Could not remove the copy this instance came from");
        record_update_unfinished(admin_target.as_deref(), SECOND_COPY_REMAINS).await;
    }
    if let Some(legacy) = legacy_copy(&dest)
        && let Err(e) = fs::remove_file(&legacy)
    {
        debug!(error = %e, "Could not remove the copy the old way of installing left behind");
        record_update_unfinished(admin_target.as_deref(), SECOND_COPY_REMAINS).await;
    }

    // Recorded for the first start of the new version; a release carrying no list is
    // recorded without one, and that start asks again. Neither case can fail the update.
    capture_update_announcement(crate::config_db::store(), &version, changes).await;

    // Before the shutdown in [`finalize_update_and_restart`]: the Telegram channel must
    // still be live.
    notify_admin(UPDATE_DOWNLOAD_COMPLETE_MSG, admin_target.as_deref()).await;
    notify_admin(UPDATE_RESTART_MSG, admin_target.as_deref()).await;

    // The temp dir is never the spawn target: that is always the file the update left in place.
    finalize_update_and_restart(&spawn_path, vec![temp_dir.path().to_path_buf()]).await
}

/// The copy the old way of installing leaves behind — the product's own file in the
/// toolchain's binary directory (`$CARGO_HOME/bin`, else `~/.cargo/bin`) — when one is
/// there that is neither the running file nor the file this update just put in place.
///
/// Computed fresh from the toolchain's convention rather than remembered, so the leftover
/// of a failed move is still found by later updates; the path is never part of what gets
/// recorded. `dest` is excluded because a `CARGO_HOME` can name the install directory
/// itself (`~/.local` on unix), whose binary directory *is* the standard location:
/// removing that would take away the file just placed.
fn legacy_copy(dest: &Path) -> Option<PathBuf> {
    let path = crate::util::cargo_bin_dir()?.join(crate::util::managed_bin::product_file_name());
    if !path.is_file() {
        return None;
    }
    let path = canonicalize_safe(&path);
    if path == canonicalize_safe(&std::env::current_exe().ok()?) || path == canonicalize_safe(dest)
    {
        return None;
    }
    Some(path)
}

/// Path to the freshly built binary inside a cargo install temp root:
/// `<install_root>/bin/mahbot` (`.exe` on Windows).
fn temp_bin_path(install_root: &Path) -> PathBuf {
    install_root
        .join("bin")
        .join(crate::util::managed_bin::product_file_name())
}

/// The build-from-sources tail, after `cargo install --path` produced the fresh binary
/// at `<temp_install>/bin/mahbot`: validate it exists and is non-empty (a silent empty
/// swap would strand the instance), capture `current_exe()` BEFORE the swap — it is the
/// restart target — swap via `self_replace` (the fresh file differs from the running exe,
/// which self-replace requires on Windows), notify, then hand off to
/// [`finalize_update_and_restart`].
///
/// `cleanup_paths` are removed at the end of that tail: on any error the caller's
/// `TempDir` values are still in scope and clean themselves up, while `exit(0)` bypasses
/// RAII on the success path.
async fn finalize_source_tree_update(
    fresh_binary: &Path,
    admin_target: Option<&str>,
    cleanup_paths: Vec<PathBuf>,
) -> Result<()> {
    let len = fs::metadata(fresh_binary)
        .with_context(|| format!("Freshly built binary missing at {}", fresh_binary.display()))?
        .len();
    if len == 0 {
        anyhow::bail!(
            "Freshly built binary at {} is empty",
            fresh_binary.display()
        );
    }

    let current_exe = std::env::current_exe().context("Failed to resolve current_exe()")?;

    self_replace::self_replace(fresh_binary)
        .with_context(|| format!("Failed to swap binary at {}", fresh_binary.display()))?;

    // Both before the shutdown in [`finalize_update_and_restart`]: the Telegram channel
    // must still be live.
    notify_admin(UPDATE_BUILD_COMPLETE_MSG, admin_target).await;
    notify_admin(UPDATE_RESTART_MSG, admin_target).await;

    finalize_update_and_restart(&current_exe, cleanup_paths).await
}

/// Shared finalize tail for both update paths: graceful drain, the final checkpoint,
/// temp-dir removal, instance-lock release, independent spawn of the replacement, and
/// `exit(0)`. `spawn_path` is the working file the update left in place. The ordering is
/// load-bearing and MUST NOT be rearranged:
///
/// 1. The FULL drain (like the tray's quit or ⌘Q, not fast-cancel): in-flight agents
///    finish their round, the drain-watch fires the global token when none remain (or
///    force-cancels at the 10-min cap), and the GUI stays up with input disabled. Its
///    exit path waits ([`update_is_finalizing`]) instead of exiting, so no close, quit
///    or SIGINT can abort this at the checkpoint. No failure transitions fire — agents
///    that cannot finish stay `status='launched'` and boot-resume — and
///    `chrome_release::flush_ended_run_chrome_releases` releases what ended runs left queued.
/// 2. Checkpoint all databases BEFORE releasing the lock or spawning: each store's
///    reader connections are released first (see `checkpoint_all_databases`), but
///    `exit(0)` bypasses destructors, so the writing connections are never formally
///    closed. This is the last checkpoint the retiring process issues — none runs after
///    the replacement is live, and the GUI exit path waits while finalizing
///    (`save_and_exit`). No writer can be in flight, the drain stopped them all, but a
///    periodic round's statement is not a writer and runs detached, so one can outlive a
///    cancelled round and answer this TRUNCATE blocked: a warn, and the replacement
///    replays the journal (`src/db/checkpoint.rs` states the same for the round).
/// 3. Remove the temp update roots BEFORE releasing the lock or spawning: `exit(0)`
///    bypasses destructors, so the (potentially multi-GB) deletion completes here while
///    the old process still holds Turso's exclusive open-time fcntl locks — a child
///    booting while they were held would fail its store open, boot having only a bounded
///    retry. Holding the instance flock during cleanup also keeps a second instance from
///    sneaking in. Deleting before the spawn is safe because the spawn target is ALWAYS
///    the working file the update put in place — `current_exe()` when self-replaced in
///    place, the standard location when placed there — never a temp root, so cleanup
///    cannot delete the binary the child is validating (macOS syspolicyd SIGKILLs a
///    child whose binary is deleted during validation).
/// 4. Release the instance lock so the child can acquire it.
/// 5. Spawn the new instance marked as the update hand-off ([`HANDOFF_ENV`] — see
///    [`acquire_lock`]).
/// 6. `exit(0)`.
///
/// On spawn failure the process stays alive and returns `Err`, its temp roots already
/// gone — acceptable, since they are transient update artifacts.
async fn finalize_update_and_restart(spawn_path: &Path, cleanup_paths: Vec<PathBuf>) -> Result<()> {
    UPDATE_FINALIZING.store(true, Ordering::SeqCst);
    crate::shutdown::drain_begin();
    let token = crate::shutdown::shutdown_token();
    token.cancelled().await;
    crate::tools::chrome_release::flush_ended_run_chrome_releases().await;

    crate::db::checkpoint::checkpoint_all_databases().await;

    // Best-effort: a stale root left behind must not abort the update.
    for path in &cleanup_paths {
        if let Err(e) = fs::remove_dir_all(path) {
            warn!(
                error = %e,
                path = %path.display(),
                "Could not remove temp update root"
            );
        }
    }

    // The local Bot API server is a child of this process and this exit runs no
    // destructors: stopped BEFORE the lock is released, so the replacement cannot adopt
    // a server this process is about to kill.
    crate::channels::stop_telegram_server().await;

    release_instance_lock().await;

    if let Err(e) = spawn_new_instance_from(spawn_path) {
        // The process stays alive unless an exit request arrived during the finalizing
        // window (the GUI then honors it with its own checkpoint and exit via
        // UpdateResult), so the flag is cleared and the lock re-acquired.
        UPDATE_FINALIZING.store(false, Ordering::SeqCst);
        if let Err(lock_err) = reacquire_instance_lock().await {
            error!(%lock_err, "Failed to re-acquire instance lock after spawn failure");
        }
        return Err(e);
    }

    // `exit(0)` runs no destructors, and a Windows notification-area icon its owner never
    // removed stays behind as a stale one. The removal is bounded, so a platform that
    // cannot answer delays this exit by seconds rather than holding it.
    crate::channels::chat_draft::flush_global();
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    crate::tray::remove();
    std::process::exit(0);
}

/// The two heads a cargo step's failure is reported under; see [`CargoStepFailure`].
struct CargoStep {
    toast_head: &'static str,
    admin_head: &'static str,
}

impl CargoStep {
    /// The failure for a step that reported `body` — the text after the head, separator included.
    fn failure(&self, body: String) -> anyhow::Error {
        anyhow::Error::new(CargoStepFailure {
            toast_head: self.toast_head,
            admin_head: self.admin_head,
            body,
        })
    }
}

/// A failed cargo step of an update: one `body`, rendered under two heads.
/// [`Display`](std::fmt::Display) is the desktop toast's text;
/// [`update_failure_notification`] composes the Telegram line, whose head names the step
/// that failed.
#[derive(Debug)]
struct CargoStepFailure {
    toast_head: &'static str,
    admin_head: &'static str,
    /// The captured output of a failed run (`:\n```…````), or why there is none.
    body: String,
}

impl std::fmt::Display for CargoStepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.toast_head, self.body)
    }
}

impl std::error::Error for CargoStepFailure {}

/// Run a long-running cargo subcommand with a timeout — the build-from-sources path's
/// only cargo runner. `args` exclude the `cargo` binary itself and are `&OsStr` so paths
/// and flags pass through without a Unicode assumption; `label` names the step in logs
/// and error messages, `step` the heads a failure is reported under.
///
/// The caller passes `--target-dir` explicitly, and inherited `CARGO_TARGET_DIR` is
/// stripped anyway, so no stray value can redirect the build out of the tree the update
/// cleans up after itself. `CARGO_HOME` is NOT stripped — the child needs it for the
/// toolchain and registry cache.
///
/// On failure the error is returned with no admin notification: the caller owns failure
/// reporting via the single [`handle_update_command`]/GUI path.
async fn run_cargo_with_timeout(
    args: &[&OsStr],
    cwd: &Path,
    timeout: Duration,
    label: &str,
    step: &CargoStep,
) -> Result<()> {
    info!("Starting {label} in {}", cwd.display());
    let mut cmd = tokio::process::Command::new("cargo");
    #[cfg(windows)]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);

    let cargo_result = tokio::time::timeout(
        timeout,
        cmd.args(args)
            .current_dir(cwd)
            // Strip any inherited CARGO_TARGET_DIR so a stray value can never redirect
            // the build out of the tree this update cleans up.
            .env_remove("CARGO_TARGET_DIR")
            // kill_on_drop: a timed-out cargo child must die rather than keep compiling,
            // and an orphaned `cargo install` could even complete and swap the binary
            // after the instance reported "timed out", racing a user retry.
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await;

    match cargo_result {
        Err(_elapsed) => Err(step.failure(format!(
            ": {label} timed out after {} minutes",
            timeout.as_secs() / 60
        ))),
        Ok(Err(e)) => Err(step.failure(format!(": could not start cargo: {e}"))),
        Ok(Ok(output)) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = format!("stdout:\n{stdout}\nstderr:\n{stderr}");
            let truncated = truncate_to_last_64k(&combined);
            Err(step.failure(format!(":\n```\n{truncated}\n```")))
        }
        Ok(Ok(_)) => {
            info!("{label} completed successfully");
            Ok(())
        }
    }
}

/// The admin's Telegram reply target: `Some` only when a bot token is configured and the
/// admin account has a Telegram binding with a non-null `reply_target`.
pub async fn resolve_admin_telegram_target() -> Option<String> {
    let _ = crate::config::CONFIG.telegram_bot_token()?;

    let store = crate::users::store();
    let bindings = store
        .get_user_channels(crate::users::ADMIN_USER_NAME)
        .await
        .ok()?;
    bindings
        .into_iter()
        .find(|b| b.channel == "telegram" && b.reply_target.is_some())
        .and_then(|b| b.reply_target)
}

/// Send a notification to the admin via Telegram. Delivery does not depend on a
/// registered listener: the transport falls back to a one-off channel on the base the
/// current mode serves when none is registered, which is the case of a switch having
/// taken the listener away. A failed send is logged and left alone, and this never waits
/// — the one notice that must arrive waits for the transport itself (see
/// [`announce_startup`]).
pub async fn notify_admin(message: &str, target: Option<&str>) {
    let Some(recipient) = target else {
        return;
    };

    if let Err(e) =
        crate::channels::telegram::send_direct(recipient, message.to_string(), None).await
    {
        error!(error = %e, "Failed to send update notification to admin");
    }
}

/// Compose the Telegram failure text: the marker plus one plain statement of what failed.
/// A failed cargo step names itself (see [`CargoStepFailure`]); every other failure
/// reports its own error text. Shared by the three user-facing entry points, so the
/// `/update` command, the window's button and the unattended trigger report the same
/// line; the desktop toast composes its own.
pub(crate) fn update_failure_notification(err: &anyhow::Error) -> String {
    let statement = match err.downcast_ref::<CargoStepFailure>() {
        Some(failure) => format!("{}{}", failure.admin_head, failure.body),
        None => format!("{err:#}"),
    };
    format!("❌ {statement}")
}

/// Report a failed update to the admin exactly as the manual callers do: the shared
/// failure line, to the admin's bound Telegram target. For the callers with no surface
/// of their own — the unattended trigger and the window's error toast — so a failure is
/// never silent, and a caller that reports here composes no second sentence about it.
pub(crate) async fn report_update_failure(err: &anyhow::Error) {
    let admin_target = resolve_update_admin_target().await;
    notify_admin(&update_failure_notification(err), admin_target.as_deref()).await;
}

/// The ordinary "back online" start-up line, naming the running version's build — what
/// says the version that came up is the new one, with no additional surface.
/// `started_paused` (`mahbot paused`) adds that mode's state, so the owner is told the
/// workspaces standing still are the mode's doing and not a fault.
#[must_use]
pub fn back_online_message(started_paused: bool) -> String {
    let message = format!("✅ MahBot is back online — running v{VERSION}");
    if started_paused {
        format!("{message}\nStarted paused — every managed workspace is on pause.")
    } else {
        message
    }
}

/// Tell the owner this process is up, with the start-up line above. The notice waits for
/// the transport first: the self-hosted address is published only once the server this
/// same process starts has answered, so such a start always opens a window in which
/// nothing can be sent. That wait is this notice's own — [`notify_admin`] serves the
/// whole update flow and must not wait — and only shutdown ends it, so a slow server
/// delays the notice rather than losing it.
pub async fn announce_startup(started_paused: bool) {
    let Some(target) = resolve_admin_telegram_target().await else {
        return;
    };
    // A wait that ends with the process is a notice nobody is left to read.
    if !crate::channels::telegram::wait_for_transport().await {
        return;
    }
    notify_admin(&back_online_message(started_paused), Some(&target)).await;
}

/// Reply used when a command requires the admin, shared by the bin's dispatch and the
/// `/update` handler so the denial wording stays consistent.
pub const ADMIN_ONLY_CMD_MSG: &str = "This command is only available to the admin.";

/// Reply for concurrent `/update` attempts — the fast pre-check, the atomic claim, and
/// the `execute_update` contention error.
const UPDATE_IN_PROGRESS_MSG: &str =
    "An update is already in progress. Please wait for it to complete.";

/// Handle a Telegram `/update` command.
///
/// Gated on the shared availability cache (admin + update available), with a synchronous
/// reply for the early failures — not an admin, already in progress, no update, cargo not
/// on PATH (the last only for a copy that has to build). The update itself runs as a
/// spawned task so it does not block the dispatch loop; progress goes to the admin's
/// Telegram binding and a failure is also reported straight to the invoker. There is NO
/// confirmation modal — invoking `/update` from Telegram is itself sufficient.
pub async fn handle_update_command(msg: &ChannelMessage) {
    if !crate::users::is_admin(&msg.user_name).await {
        crate::channels::telegram::send_reply(&msg.reply_target, ADMIN_ONLY_CMD_MSG).await;
        return;
    }

    // Fast path for an already-running update, then the atomic claim below.
    let availability = update_availability();
    if availability.in_progress {
        crate::channels::telegram::send_reply(&msg.reply_target, UPDATE_IN_PROGRESS_MSG).await;
        return;
    }

    if !should_show_update(availability) {
        crate::channels::telegram::send_reply(
            &msg.reply_target,
            "No update is available at the moment.",
        )
        .await;
        return;
    }

    // Synchronous pre-check so the early-failure reply is guaranteed before the
    // update is spawned. `cargo --version` is cheap, but this briefly awaits a
    // subprocess inline in the dispatch loop — accepted so the invoker gets an
    // immediate answer instead of a silent no-op. Only a copy built from sources
    // needs cargo at all; a downloaded copy is never built here.
    if update_mode() == UpdateMode::SourceTree
        && let Err(e) = verify_cargo_on_path("perform the update").await
    {
        crate::channels::telegram::send_reply(
            &msg.reply_target,
            &format!("Cannot start the update: {e}"),
        )
        .await;
        return;
    }

    // Claimed before spawning, closing the TOCTOU where a concurrent trigger could pass
    // the pre-check above and then lose `UPDATE_MUTEX.try_lock` inside `execute_update`,
    // reported as "An update is already in progress."
    if !claim_update_start() {
        crate::channels::telegram::send_reply(&msg.reply_target, UPDATE_IN_PROGRESS_MSG).await;
        return;
    }

    crate::channels::telegram::send_reply(
        &msg.reply_target,
        "✅ Update triggered — it will run in the background and restart the daemon when complete.",
    )
    .await;

    // Fire-and-forget: the update must not block the dispatch loop. Progress
    // notifications route via the normal update notification path (the admin's
    // bound Telegram target); on failure the invoking admin is
    // also told directly so a non-primary invoker isn't left guessing.
    let invoker_target = msg.reply_target.clone();
    tokio::spawn(async move {
        if let Err(e) = execute_update(None).await {
            // Single failure report: the admin's bound Telegram target (the normal update
            // notification path), plus the invoking admin when they differ.
            let failure = update_failure_notification(&e);
            let admin_target = resolve_update_admin_target().await;
            notify_admin(&failure, admin_target.as_deref()).await;
            if admin_target.as_deref() != Some(invoker_target.as_str()) {
                crate::channels::telegram::send_reply(&invoker_target, &failure).await;
            }
        }
    });
}

/// Canonicalize a path, falling back to the lexical path when the file does not exist yet
/// or canonicalization fails (a broken symlink, permission denied).
fn canonicalize_safe(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Spawn the new mahbot instance as an independent child from `binary_path` — always the
/// working file the update left in place, never a temp root (see
/// [`finalize_update_and_restart`]).
///
/// The child is marked as the update hand-off ([`HANDOFF_ENV`] — see [`acquire_lock`]) and
/// as the replacement of a hidden window when the one it replaces was hidden
/// ([`START_HIDDEN_ENV`]). It inherits the launcher's mode — `paused` when the old process
/// was started paused — and never the old process's argv: a word this binary has no entry
/// point for would be refused and the service would not come back.
///
/// Both platforms get null stdin/stdout and stderr into update.log; Unix adds `setsid`, so
/// a hangup aimed at the replaced instance's session cannot reach the replacement, and
/// Windows adds `CREATE_NO_WINDOW`.
///
/// `CREATE_NO_WINDOW` is inert here: the binary is built windowed (its crate root declares
/// `windows_subsystem = "windows"`), so the instance owns no console and the platform
/// documents the flag as ignored for a non-console application. It stays because it is
/// right for a console-subsystem build and because the shell module's spawn tripwire
/// requires it at every production spawn site.
///
/// `DETACHED_PROCESS` is deliberately *not* set: there is no console for this windowed
/// image to inherit and none would be created, so the flag would change nothing. (A
/// console-subsystem child from a console-less parent does get one; the shell module's
/// spawn sites carry `CREATE_NO_WINDOW` for those.)
///
/// A spawn failure is returned without an admin notification — the caller owns failure
/// reporting — and the process keeps running: the null stdin/stdout and the stderr
/// redirection to update.log matter, since update.log is a real file handle, so the
/// replacement's printing works with no console in sight.
///
/// The caller guarantees `binary_path` is never deleted before or during the child's
/// startup window (see [`finalize_update_and_restart`]): deleting the spawn target while
/// Gatekeeper validates its signature makes `syspolicyd` SIGKILL the child.
fn spawn_new_instance_from(binary_path: &Path) -> Result<()> {
    let launch = std::env::args_os().nth(1);
    let args: Vec<std::ffi::OsString> =
        match crate::cli::entry_point(launch.as_deref().and_then(std::ffi::OsStr::to_str)) {
            Some(crate::cli::EntryPoint::Paused) => vec!["paused".into()],
            _ => Vec::new(),
        };

    info!(
        program = %binary_path.display(),
        args = ?args,
        "Spawning new mahbot instance"
    );

    let mut cmd = std::process::Command::new(binary_path);
    cmd.args(&args);
    cmd.env(HANDOFF_ENV, "1");
    // Written on every spawn, never left to inheritance: the parent may itself have come
    // back hidden, and the marker must describe THIS window.
    cmd.env(
        START_HIDDEN_ENV,
        if crate::gui::app_window::hidden() {
            "1"
        } else {
            "0"
        },
    );

    let update_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(
            crate::config::CONFIG
                .global_storage_root()
                .join("update.log"),
        )
        .context("Failed to open update.log for child stderr")?;

    cmd.stdin(Stdio::null()).stdout(Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // A session of its own, so a hangup aimed at the replaced instance's session
        // cannot reach the replacement — the same reason the Telegram helper has one.
        //
        // SAFETY: `setsid` is async-signal-safe and is the only thing run between the fork
        // and the exec; the child is freshly forked, so it is not a group leader.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    cmd.stderr(Stdio::from(update_log));

    match cmd.spawn() {
        Ok(child) => {
            info!(pid = child.id(), "Spawned new mahbot instance");
            Ok(())
        }
        Err(e) => {
            warn!(
                error = %e,
                "New instance spawn failed — keeping current instance alive"
            );
            Err(anyhow::Error::from(e).context("Failed to spawn new instance after update"))
        }
    }
}

fn truncate_to_last_64k(s: &str) -> String {
    const MAX: usize = 64 * 1024;
    if s.len() <= MAX {
        return s.to_string();
    }
    let start = s.ceil_char_boundary(s.len() - MAX);
    format!(
        "[…output truncated; showing last {} bytes…]\n{}",
        MAX,
        &s[start..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::lock::{lock_file_path, try_flock};

    #[test]
    fn test_try_acquire_lock_held_free() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = lock_file_path(dir.path());

        let holder = open_lock_file(&lock_path).unwrap();
        assert!(try_flock(&holder).unwrap(), "First flock should succeed");
        let contender = open_lock_file(&lock_path).unwrap();
        assert!(
            !try_flock(&contender).unwrap(),
            "Second flock should fail (already locked)"
        );

        assert!(
            try_acquire_lock(&lock_path).unwrap().is_none(),
            "Should return None when lock is held"
        );

        // The explicit release the update hand-off relies on frees the lock while the
        // holder's handle is still open.
        holder.unlock().expect("unlock the held lock file");
        assert!(
            try_flock(&contender).unwrap(),
            "After the explicit unlock, the contender must acquire the lock"
        );

        drop(holder);
        drop(contender);
        assert!(
            lock_becomes_free(&lock_path).expect("lock the released lock file"),
            "After release, the lock must be acquirable again"
        );
    }

    /// Tripwire: a launch of the executable raises nothing, so a window can only come up on a
    /// user's gesture. The tray icon's click is the one producer of a show request
    /// ([`crate::window_requests::show`]), and the dashboard's own boot rescue
    /// ([`crate::gui::app_window::show`]) is the one other place the window is shown — no launch,
    /// no service channel and no reopen of the application may reach either, nor the iced calls
    /// they are built on.
    ///
    /// It is a source scan, not a proof: line comments are skipped, block comments are not, and
    /// the patterns are spelled split so this table does not match itself.
    #[test]
    fn nothing_but_a_gesture_raises_the_window() {
        const WINDOW: &[&str] = &["src/gui/app_window.rs"];
        let patterns = [
            (concat!("window_requests::", "show()"), &["src/tray.rs"][..]),
            (concat!("app_window::", "show()"), &["src/gui/mod.rs"][..]),
            (concat!("Mode::", "Windowed"), WINDOW),
            (concat!("gain_", "focus"), WINDOW),
        ];
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut violations: Vec<String> = Vec::new();
        for file in crate::util::test::rs_files_under(&root.join("src")) {
            let rel = crate::util::test::rel_source_path(root, &file);
            let content = std::fs::read_to_string(&file).expect("read source file");
            let code_only: String = content
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            for (pattern, allowed) in patterns {
                if !allowed.contains(&rel.as_str()) && code_only.contains(pattern) {
                    violations.push(format!("{rel}: {pattern}"));
                }
            }
        }
        assert!(
            violations.is_empty(),
            "the window comes up on a user's gesture alone — the tray icon's click, and the \
             dashboard's boot rescue when a hidden window has no tray to bring it back. A launch \
             of the executable must not raise it (a keepalive automation's start branch, an \
             agent's restart and a script's \"open the app\" all do nothing), and neither must \
             the service channel or the platform's reopen.\n{}",
            violations.join("\n")
        );
    }

    /// The hand-off wait takes the location on the explicit `unlock` with the handle still
    /// open — what [`FlockGuard::release`] does, and why it does not rely on the close.
    #[test]
    fn wait_for_handoff_takes_the_released_location() {
        let dir = tempfile::tempdir().unwrap();
        let path = lock_file_path(dir.path());
        let holder = open_lock_file(&path).unwrap();
        assert!(
            try_flock(&holder).unwrap(),
            "the outgoing instance must hold the location"
        );

        let releaser = std::thread::spawn(move || {
            std::thread::sleep(HANDOFF_POLL_INTERVAL * 2);
            holder.unlock().expect("release the location");
            holder
        });

        let taken =
            wait_for_handoff(&path).expect("a marked instance must take the released location");
        assert!(
            location_is_held(&path),
            "the wait must leave the location held"
        );

        // The outgoing handle is closed only now: the wait's own handle holds the lock
        // and must survive that close.
        drop(releaser.join().unwrap());
        assert!(
            location_is_held(&path),
            "the lock taken by the wait must survive the outgoing handle's close"
        );
        drop(taken);
    }

    fn location_is_held(path: &Path) -> bool {
        !try_flock(&open_lock_file(path).unwrap()).unwrap()
    }

    /// Poll for `path` to become lockable again, for up to ~2s: every `Command::spawn` in
    /// the suite forks and carries this process's open file description until the child
    /// execs, so a release can look held for a moment. A lock never released (the
    /// regression this test exists for) comes back `Ok(false)`; an OS error as it is.
    fn lock_becomes_free(path: &Path) -> Result<bool> {
        for _ in 0..100 {
            match try_acquire_lock(path) {
                Ok(Some(_)) => return Ok(true),
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => return Err(e),
            }
        }
        Ok(false)
    }

    #[test]
    fn test_update_mode_detection() {
        let dir = tempfile::tempdir().unwrap();

        // A working tree: `.git` present → SourceTree even with Cargo.toml.
        let checkout = dir.path().join("checkout");
        std::fs::create_dir_all(checkout.join(".git")).unwrap();
        std::fs::write(checkout.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&checkout), UpdateMode::SourceTree);

        // Registry src cache: `registry/src` → Downloaded even with Cargo.toml (the
        // cargo-installed trap the probe used to misclassify).
        let registry = dir
            .path()
            .join(".cargo")
            .join("registry")
            .join("src")
            .join("index.crates.io-6f17d22bba3b01f9")
            .join("mahbot-0.3.0");
        std::fs::create_dir_all(&registry).unwrap();
        std::fs::write(registry.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&registry), UpdateMode::Downloaded);

        // Git checkouts are full non-bare clones with a real `.git`, so `git/checkouts`
        // must win over the `.git` heuristic (regression guard for that ordering).
        let git_checkout = dir
            .path()
            .join(".cargo")
            .join("git")
            .join("checkouts")
            .join("mahbot-1a2b3c")
            .join("main");
        std::fs::create_dir_all(git_checkout.join(".git")).unwrap();
        std::fs::write(git_checkout.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&git_checkout), UpdateMode::Downloaded);

        // Custom CARGO_HOME (no `.cargo` prefix) is still detected.
        let custom = dir
            .path()
            .join("custom-cargo")
            .join("registry")
            .join("src")
            .join("index.crates.io-hash")
            .join("mahbot-0.3.0");
        std::fs::create_dir_all(&custom).unwrap();
        std::fs::write(custom.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&custom), UpdateMode::Downloaded);

        // Plain source tree: Cargo.toml → SourceTree; no source at all → Downloaded.
        let plain = dir.path().join("plain-src");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&plain), UpdateMode::SourceTree);

        let bare = dir.path().join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(classify_update_mode(&bare), UpdateMode::Downloaded);

        // The registry cache downloads but is still a build from source, so only the copy
        // with no build source came from the release pipeline.
        for source in [&checkout, &registry, &git_checkout, &custom, &plain] {
            assert!(!is_published_copy_at(source), "{}", source.display());
        }
        assert!(is_published_copy_at(&bare));
    }

    /// The unattended trigger's whole rule, one case per clause: only a downloaded copy,
    /// only with no version named for it by the environment hook, only for a strictly
    /// newer release, and never for a release no newer than one whose attempt failed.
    #[test]
    fn test_auto_update_target() {
        let v = |s: &str| semver::Version::parse(s).unwrap();
        let current = v("1.2.3");
        let newer = v("1.2.4");
        let newest = v("1.3.0");
        let is = |mode: UpdateMode,
                  named: bool,
                  latest: Option<&semver::Version>,
                  failed: Option<&semver::Version>| {
            auto_update_target(mode, named, latest, &current, failed)
        };

        // A strictly newer release, and only that one, fires.
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&newer), None),
            Some(newer.clone())
        );
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&newest), None),
            Some(newest.clone())
        );
        assert_eq!(is(UpdateMode::Downloaded, false, None, None), None);
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&current), None),
            None
        );
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&v("1.2.2")), None),
            None
        );

        // A copy built from a working tree never fires by itself: its availability is true
        // by construction, which is why the flag alone is not the trigger.
        assert_eq!(is(UpdateMode::SourceTree, false, Some(&newer), None), None);
        assert_eq!(is(UpdateMode::SourceTree, false, None, None), None);

        // A version named through the environment hook is that hook's update.
        assert_eq!(is(UpdateMode::Downloaded, true, Some(&newer), None), None);

        // A release whose own attempt failed is not retried; a strictly newer one still is.
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&newer), Some(&newer)),
            None
        );
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&newest), Some(&newer)),
            Some(newest.clone())
        );

        // A stale advertisement — older than the release whose attempt failed — re-offers
        // nothing either.
        assert_eq!(
            is(UpdateMode::Downloaded, false, Some(&newer), Some(&newest)),
            None
        );
    }

    /// A restart of the same version, a build from source and a first install during setup
    /// say nothing; a published copy on a different version — or the first start of one
    /// installed by a build that predates the record — announces; a list the update left is
    /// announced even over a version already recorded, and waits while setup is unfinished.
    #[test]
    fn test_the_update_notice_is_only_for_a_downloaded_version_change() {
        let current = "1.2.3";
        let case = |last, published, onboarding, owed| {
            update_notice_action(last, current, published, onboarding, owed)
        };

        // The version already ran with nothing owed: nothing to settle.
        assert_eq!(
            case(Some(current), true, true, false),
            UpdateNotice::Unchanged
        );
        // A build from source never announces, whatever its version string says.
        assert_eq!(
            case(Some(current), false, true, false),
            UpdateNotice::Unchanged
        );
        assert_eq!(
            case(Some("1.2.2"), false, true, false),
            UpdateNotice::RecordSilently
        );
        // A published copy with finished setup on a new version — including the first start
        // of one installed by a build that predates the record — announces; during setup
        // either case only records.
        assert_eq!(case(None, true, true, false), UpdateNotice::Announce);
        assert_eq!(
            case(Some("1.2.2"), true, true, false),
            UpdateNotice::Announce
        );
        assert_eq!(case(None, true, false, false), UpdateNotice::RecordSilently);
        assert_eq!(
            case(Some("1.2.2"), true, false, false),
            UpdateNotice::RecordSilently
        );
        // A brought list is announced even over a recorded version — a start that stopped
        // before the hand-over is repeated — and waits unspent while setup is unfinished.
        assert_eq!(
            case(Some(current), true, true, true),
            UpdateNotice::Announce
        );
        assert_eq!(case(None, true, true, true), UpdateNotice::Announce);
        // An owed list waits unspent while setup is unfinished.
        assert_eq!(
            case(Some("1.2.2"), true, false, true),
            UpdateNotice::Unchanged
        );
        assert_eq!(
            case(Some(current), true, false, true),
            UpdateNotice::Unchanged
        );
        assert_eq!(case(None, true, false, true), UpdateNotice::Unchanged);
        assert_eq!(
            case(Some("1.2.2"), false, true, true),
            UpdateNotice::RecordSilently
        );
        assert_eq!(
            case(Some(current), false, true, true),
            UpdateNotice::Unchanged
        );
    }

    /// A list already captured for a version is never replaced by an attempt whose fetch
    /// came back empty, and a record captured without one is filled in by a later attempt.
    #[tokio::test]
    async fn test_a_captured_list_is_never_replaced_by_an_empty_one() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");
        capture_update_announcement(&store, "1.2.3", Some("* a change".to_string())).await;
        capture_update_announcement(&store, "1.2.3", None).await;
        let raw = store
            .get_kv(UPDATE_PENDING_CHANGELOG_KV_KEY)
            .await
            .unwrap()
            .unwrap();
        let pending = pending_changelog_from_str(&raw, "1.2.3").expect("the record holds");
        assert_eq!(pending.changes.as_deref(), Some("* a change"));

        capture_update_announcement(&store, "1.2.4", None).await;
        capture_update_announcement(&store, "1.2.4", Some("* filled in".to_string())).await;
        let raw = store
            .get_kv(UPDATE_PENDING_CHANGELOG_KV_KEY)
            .await
            .unwrap()
            .unwrap();
        let pending = pending_changelog_from_str(&raw, "1.2.4").expect("the record holds");
        assert_eq!(pending.changes.as_deref(), Some("* filled in"));
    }

    /// An update that brought no list — made by a build predating the capture, by the
    /// install scripts, or with its own fetch failed — still announces the release's list,
    /// read while the notice is composed.
    #[tokio::test]
    async fn test_a_notice_without_a_carried_list_reads_the_releases_own() {
        let (bare, _bare_dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");
        bare.set_kv(UPDATE_LAST_RUN_VERSION_KV_KEY, "1.2.2")
            .await
            .unwrap();

        let mut delivered = Vec::new();
        settle_update_notice(
            &bare,
            "1.2.3",
            true,
            true,
            async |_| Some("* a new capability".to_string()),
            |content| delivered.push(content),
        )
        .await;

        assert_eq!(delivered.len(), 1);
        assert!(delivered[0].contains("* a new capability"));
        assert_eq!(
            bare.get_kv(UPDATE_LAST_RUN_VERSION_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("1.2.3")
        );

        // A capture whose own fetch failed carries no list; the read at notice time fills
        // it in.
        let (failed, _failed_dir) =
            crate::open_test_store!(crate::config_db::ConfigStore, "config");
        failed
            .set_kv(UPDATE_LAST_RUN_VERSION_KV_KEY, "1.2.2")
            .await
            .unwrap();
        failed
            .set_kv(
                UPDATE_PENDING_CHANGELOG_KV_KEY,
                r#"{"version":"1.2.3","changes":null}"#,
            )
            .await
            .unwrap();

        let mut delivered = Vec::new();
        settle_update_notice(
            &failed,
            "1.2.3",
            true,
            true,
            async |_| Some("* a change the fetch missed".to_string()),
            |content| delivered.push(content),
        )
        .await;

        assert_eq!(delivered.len(), 1);
        assert!(delivered[0].contains("* a change the fetch missed"));
    }

    /// A captured list outlives a start that does not announce: an unfinished setup
    /// records nothing and drops nothing, and the next start of the same version announces it.
    #[tokio::test]
    async fn test_a_start_that_does_not_announce_keeps_the_captured_list() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");
        store
            .set_kv(UPDATE_LAST_RUN_VERSION_KV_KEY, "1.2.2")
            .await
            .unwrap();
        let captured = r#"{"version":"1.2.3","changes":"* a change"}"#;
        store
            .set_kv(UPDATE_PENDING_CHANGELOG_KV_KEY, captured)
            .await
            .unwrap();

        let mut delivered = Vec::new();
        settle_update_notice(
            &store,
            "1.2.3",
            true,
            false,
            async |_| None,
            |content| delivered.push(content),
        )
        .await;
        assert!(delivered.is_empty(), "an unfinished setup tells no one");
        assert_eq!(
            store
                .get_kv(UPDATE_LAST_RUN_VERSION_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("1.2.2")
        );
        assert_eq!(
            store
                .get_kv(UPDATE_PENDING_CHANGELOG_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some(captured)
        );

        settle_update_notice(
            &store,
            "1.2.3",
            true,
            true,
            async |_| None,
            |content| delivered.push(content),
        )
        .await;
        assert_eq!(delivered.len(), 1);
        assert!(delivered[0].contains("* a change"));
        assert_eq!(
            store
                .get_kv(UPDATE_LAST_RUN_VERSION_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("1.2.3")
        );
        assert!(
            store
                .get_kv(UPDATE_PENDING_CHANGELOG_KV_KEY)
                .await
                .unwrap()
                .is_none()
        );
    }

    /// A list that cannot be obtained at all is named unavailable with the release page it
    /// can be read on, and a record captured for another version is left for that version's
    /// own start.
    #[tokio::test]
    async fn test_an_unobtainable_list_names_the_release_page() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");
        store
            .set_kv(UPDATE_LAST_RUN_VERSION_KV_KEY, "1.2.2")
            .await
            .unwrap();
        let other = r#"{"version":"1.1.0","changes":"* an older change"}"#;
        store
            .set_kv(UPDATE_PENDING_CHANGELOG_KV_KEY, other)
            .await
            .unwrap();

        let mut delivered = Vec::new();
        settle_update_notice(
            &store,
            "1.2.3",
            true,
            true,
            async |_| None,
            |content| delivered.push(content),
        )
        .await;

        assert_eq!(delivered.len(), 1);
        assert!(delivered[0].contains("/releases/tag/v1.2.3"));
        assert_eq!(
            store
                .get_kv(UPDATE_PENDING_CHANGELOG_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some(other),
            "a list captured for another version is not spent"
        );
    }

    /// The captured list is used only for the version it was taken for: one belonging to
    /// another release is left alone, an unreadable one treated as absent.
    #[test]
    fn test_the_captured_changelog_belongs_to_its_version() {
        let recorded = r#"{"version":"1.2.3","changes":"* a change"}"#;
        let matched =
            pending_changelog_from_str(recorded, "1.2.3").expect("the record belongs to 1.2.3");
        assert_eq!(matched.changes.as_deref(), Some("* a change"));
        assert!(pending_changelog_from_str(recorded, "1.2.4").is_none());
        assert!(pending_changelog_from_str("not a record", "1.2.3").is_none());
    }

    /// An ordinary copy is never offered a test release, and a version file holding none is
    /// an error rather than an absence.
    #[test]
    fn test_a_test_release_is_never_discovered() {
        let stable = semver::Version::new(1, 2, 3);
        assert_eq!(discoverable_version("1.2.3\n").unwrap(), Some(stable));
        assert_eq!(discoverable_version("1.2.3-rc.1\n").unwrap(), None);
        assert_eq!(discoverable_version("  1.2.3-rc.1  ").unwrap(), None);
        assert!(discoverable_version("").is_err());
        assert!(discoverable_version("version 1.2.3").is_err());
    }

    /// The install scripts download what this file publishes and the release workflow
    /// builds it, so one contract is spelled in four files that nothing else in the tree
    /// reads — drift would otherwise be invisible. Pinned: the repository, the two URL
    /// shapes, the asset name [`asset_name`] builds against the workflow's own list of six
    /// files, the install location all three share, the version file's location, and the
    /// compiler both release jobs install against the version `rust-toolchain.toml` names.
    #[test]
    fn test_the_release_contract_is_spelled_the_same_everywhere() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let read = |name: &str| std::fs::read_to_string(root.join(name)).expect("part of the tree");
        let (sh, ps1) = (read("install.sh"), read("install.ps1"));

        // Each URL as this file builds it, with the base and version left as the variables
        // the script sets them from — and the base the constant, not the test hook that can
        // override it.
        for (name, script, base_var, version_var) in [
            ("install.sh", &sh, "$RELEASE_BASE", "$VERSION"),
            ("install.ps1", &ps1, "$ReleaseBase", "$Version"),
        ] {
            assert!(script.contains(RELEASE_REPO), "{name} names {RELEASE_REPO}");
            let pointer = latest_version_url(RELEASE_REPO).replace(RELEASE_REPO, base_var);
            assert!(
                script.contains(&pointer),
                "{name} must look the newest release up at {pointer}"
            );
            let download = asset_url(
                RELEASE_REPO,
                &semver::Version::new(1, 0, 0),
                "linux",
                "x86_64",
            )
            .replace(RELEASE_REPO, base_var)
            .replace("1.0.0", version_var);
            let version_dir = download.rsplit_once('/').expect("an asset name").0;
            assert!(
                script.contains(version_dir),
                "{name} must download a version from {version_dir}"
            );
        }
        // An asset's name is built from the scripts' own variables, so what is pinned is the shape.
        assert!(sh.contains(r#"ASSET="mahbot-$VERSION-$OS-$ARCH.tar.gz""#));
        assert!(ps1.contains(r#"$Asset = "mahbot-$Version-windows-$Arch.zip""#));

        // One install location for all three — the scripts put the ready-made file there,
        // the product moves itself there when its working file is elsewhere, and its
        // search-path edit points at it — since a drift would leave exactly the second copy
        // the arrangement prevents. Each script's literal is checked wherever this runs; the
        // comparison against the location this product computes only where it can be asked.
        assert!(sh.contains(r#"INSTALL_DIR="$HOME/.local/bin""#));
        assert!(ps1.contains(r"'Programs\MahBot'"));
        let install = crate::util::managed_bin::mahbot_install_dir().expect("a home directory");
        #[cfg(unix)]
        assert!(
            install.ends_with(".local/bin"),
            "the product's own install directory is {}",
            install.display()
        );
        #[cfg(windows)]
        assert!(
            install.ends_with(r"Programs\MahBot"),
            "the product's own install directory is {}",
            install.display()
        );

        // The workflow builds one file per system and publishes only once all six are
        // attached, so its own list of them is the release's definition.
        let workflow = read(".github/workflows/release.yml");
        floors_are_spelled_the_same(&sh, &ps1, &workflow);
        assert!(
            workflow.contains("assets+=(\"$RUNNER_TEMP/release/changelog.txt\")"),
            "release.yml must attach the change list the updater takes down"
        );

        // The refusal words are one fact too: both scripts begin with this file's own
        // opening, and name the same requirement for a Linux host with no glibc.
        assert!(sh.contains(&format!("NO_FILE_PREFIX='{NO_RELEASE_FILE}'")));
        assert!(ps1.contains(&format!("$NoFilePrefix = '{NO_RELEASE_FILE}'")));
        assert!(sh.contains(NO_GLIBC_FOUND));

        for (os, arch) in [
            ("macos", "x86_64"),
            ("macos", "aarch64"),
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("windows", "x86_64"),
            ("windows", "aarch64"),
        ] {
            let spelled =
                asset_name(&semver::Version::new(1, 0, 0), os, arch).replace("1.0.0", "${VERSION}");
            assert!(
                workflow.contains(&format!("\"{spelled}\"")),
                "release.yml must publish {spelled}"
            );
        }

        // The compiler is one fact in three places: the workflow's installation points
        // and the toolchain file every local build and diagnostic run resolves. The file
        // is not part of the published package, so only a run from a packaged crate —
        // what `Cargo.toml.orig` alone marks — may lack it.
        match std::fs::read_to_string(root.join("rust-toolchain.toml")) {
            Ok(toolchain) => {
                let channel = toolchain
                    .lines()
                    .filter_map(|line| line.split_once('='))
                    .find(|(key, _)| key.trim() == "channel")
                    .and_then(|(_, value)| value.trim().strip_prefix('"')?.strip_suffix('"'))
                    .expect("rust-toolchain.toml names a channel");
                let installs: Vec<&str> = workflow
                    .lines()
                    .filter_map(|line| line.trim().strip_prefix("uses: dtolnay/rust-toolchain@"))
                    .collect();
                assert!(
                    installs.len() >= 2,
                    "both release jobs install the toolchain"
                );
                for installed in installs {
                    assert_eq!(
                        installed, channel,
                        "the release legs must build with the pinned compiler"
                    );
                }
            }
            Err(error) => assert!(
                root.join("Cargo.toml.orig").exists(),
                "rust-toolchain.toml is part of the tree: {error}"
            ),
        }
    }

    /// The changelog's filter as it ships — the awk program read out of the workflow — keeps
    /// every subject except version bumps, blank titles and `[maint] ` commits; the samples
    /// are built from the same label tokens the commit path writes.
    #[cfg(unix)]
    #[test]
    fn test_the_changelog_skips_maint_commits() {
        use crate::pipeline::board::TicketLabel;
        use std::io::Write as _;

        let workflow = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/release.yml"),
        )
        .expect("release.yml is part of the tree");
        let program = workflow
            .split_once("git log --reverse --pretty=format:%s")
            .and_then(|(_, rest)| rest.split_once("| awk '"))
            .and_then(|(_, rest)| rest.split_once("' > \"$RUNNER_TEMP/changelog.txt\""))
            .map(|(program, _)| program)
            .expect("the changelog step holds its awk program");

        let label = |label: TicketLabel, title: &str| format!("{}{title}", label.commit_prefix());
        let subjects = [
            "Bump version to 1.2.3".to_string(),
            label(TicketLabel::Maint, "Remove the dead helper"),
            label(TicketLabel::Feat, "Add the real fix"),
            String::new(),
            "Document the [maint] prefix".to_string(),
            label(TicketLabel::Maint, "Rename the helper"),
            label(TicketLabel::Fix, "Fix the second bug"),
        ];
        let mut child = std::process::Command::new("awk")
            .arg(program)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("awk runs on unix hosts");
        child
            .stdin
            .take()
            .expect("stdin is piped")
            .write_all(subjects.join("\n").as_bytes())
            .expect("write the subjects");
        let output = child.wait_with_output().expect("the filter runs");
        assert!(output.status.success(), "awk rejected the workflow program");
        assert_eq!(
            String::from_utf8(output.stdout).expect("utf-8 output"),
            "### Changelog\n\n* [feat] Add the real fix\n* Document the [maint] prefix\n\
             * [fix] Fix the second bug\n",
        );
    }

    /// The floors are one fact told by machinery that must agree: the install command, the
    /// update path and the workflow all refuse or build below them, and a drift offers a
    /// file to a system it cannot load on. The README's prose is not pinned — a rewording
    /// is not a drift in any of these.
    fn floors_are_spelled_the_same(sh: &str, ps1: &str, workflow: &str) {
        assert!(sh.contains(&format!("MACOS_FLOOR_MAJOR={}", MACOS_FLOOR.0)));
        assert!(sh.contains(&format!("MACOS_FLOOR_MINOR={}", MACOS_FLOOR.1)));
        assert!(sh.contains(&format!("GLIBC_FLOOR_MAJOR={}", GLIBC_FLOOR.0)));
        assert!(sh.contains(&format!("GLIBC_FLOOR_MINOR={}", GLIBC_FLOOR.1)));
        assert!(
            workflow.contains(&format!(
                "MACOSX_DEPLOYMENT_TARGET: ${{{{ matrix.target_os == 'macos' && '{}.{}' || '' }}}}",
                MACOS_FLOOR.0, MACOS_FLOOR.1
            )),
            "the macOS files must be built against {}.{}",
            MACOS_FLOOR.0,
            MACOS_FLOOR.1
        );
        assert!(
            workflow.contains(&format!("glibc {}.{}", GLIBC_FLOOR.0, GLIBC_FLOOR.1)),
            "the Linux files must be built on a base whose glibc is the floor"
        );
        // The Windows floors are the systems the files exist for, by their own build
        // numbers: Windows 10 1809 on x86_64, and Windows 11 on ARM because every
        // Windows 10 on ARM (below the ARM64 floor) is below the file that exists.
        // Both what is compared and the sentence it is refused with are the constants'.
        assert!(ps1.contains(&format!("$Build -lt {WINDOWS_X86_64_FLOOR}")));
        assert!(ps1.contains(&format!("$Build -lt {WINDOWS_ARM64_FLOOR}")));
        assert!(ps1.contains(&format!("Windows 11 (build {WINDOWS_ARM64_FLOOR})")));
        assert!(ps1.contains(&format!(
            "Windows 10 version 1809 (build {WINDOWS_X86_64_FLOOR})"
        )));
    }

    #[test]
    fn floor_checks_compare_whole_version_numbers() {
        // A minor part is not a decimal: 2.4 is below 2.35, while 2.39 is not.
        assert!(version_below_floor("2.4", GLIBC_FLOOR));
        assert!(!version_below_floor("2.39", GLIBC_FLOOR));
        // A missing minor counts as zero, and a trailing part is not compared.
        assert!(!version_below_floor("14", MACOS_FLOOR));
        assert!(!version_below_floor("14.0", MACOS_FLOOR));
        assert!(!version_below_floor("14.0.1", MACOS_FLOOR));
        // A missing or non-numeric part is not a version to accept, in either position —
        // as the install script's check refuses it.
        assert!(version_below_floor("", MACOS_FLOOR));
        assert!(version_below_floor("14.", MACOS_FLOOR));
        assert!(version_below_floor("13.x", MACOS_FLOOR));
        // The glibc report is read the same way; an unrecognised one is not a refusal.
        assert!(glibc_below_floor("glibc 2.31"));
        assert!(!glibc_below_floor("glibc 2.35"));
        assert!(!glibc_below_floor("musl libc (x86_64)"));
    }

    #[test]
    fn a_linux_host_is_refused_by_its_own_evidence() {
        use LinuxRefusal::{GlibcBelowFloor, NoGlibc};
        // A glibc below the floor is refused whether or not musl's loader is there.
        assert_eq!(
            linux_refusal(Some("glibc 2.31"), false),
            Some(GlibcBelowFloor)
        );
        assert_eq!(
            linux_refusal(Some("glibc 2.31"), true),
            Some(GlibcBelowFloor)
        );
        // At or above the floor there is a file to load: musl's loader being there as well
        // — under a symlinked `/lib` on Debian, musl installs beside glibc — is no refusal.
        assert_eq!(linux_refusal(Some("glibc 2.35"), true), None);
        assert_eq!(linux_refusal(Some("glibc 2.39"), true), None);
        // A host that names no glibc is judged by the loader it has.
        assert_eq!(linux_refusal(None, true), Some(NoGlibc));
        assert_eq!(
            linux_refusal(Some("musl libc (x86_64)"), true),
            Some(NoGlibc)
        );
        // A host with neither is not refused on evidence it does not have.
        assert_eq!(linux_refusal(None, false), None);
        assert_eq!(linux_refusal(Some("musl libc (x86_64)"), false), None);
    }

    #[test]
    fn the_leftover_is_never_the_file_just_put_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let copy = bin.join(crate::util::managed_bin::product_file_name());
        std::fs::write(&copy, b"a copy").unwrap();
        // A `CARGO_HOME` whose binary directory is the install directory itself, so the
        // toolchain's convention names the file the update just placed.
        let _cargo_home = crate::util::test::set_env_var(
            "CARGO_HOME",
            Some(dir.path().to_str().expect("a UTF-8 temp path")),
        );
        let placed = legacy_copy(&copy);
        let elsewhere = legacy_copy(&bin.join("somewhere_else"));
        assert_eq!(placed, None, "the file just placed is not a leftover");
        assert_eq!(
            elsewhere,
            Some(canonicalize_safe(&copy)),
            "a copy that is neither is a leftover"
        );
    }
}
