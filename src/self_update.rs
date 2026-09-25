//! Self-update logic — single-instance guarding, obtaining the new version,
//! putting it in place, and restart.
//!
//! Two update modes are supported, selected at runtime by [`update_mode`]:
//!
//! * **A downloaded copy** ([`UpdateMode::Downloaded`]): this binary is one of the
//!   product's own published files — the package registry's source cache, a moved
//!   copy, or a copy whose build source is gone. Update = download this system's
//!   ready-made release file → extract → put it at the standard per-user location
//!   (in place of the working file when that location is already where this copy
//!   runs from; when it is not, the working file is taken away afterwards) →
//!   restart.
//! * **A copy built from sources** ([`UpdateMode::SourceTree`]): this binary has
//!   its own working tree at `CARGO_MANIFEST_DIR`. Update = `cargo install --path
//!   <tree> --root <temp_install> --locked --target-dir <temp_build>` → swap →
//!   restart.
//!
//! Both modes converge on the same tail: the new file becomes the one the
//! replacement instance runs from (the running file itself when it is already the
//! standard location, a placement at the standard location otherwise), the admin is
//! notified, in-flight work is drained, the databases are checkpointed, and a
//! replacement instance is started from that file before `exit(0)`. The running
//! executable cannot be replaced in place on Windows, so the source-tree build lands
//! in a temp root first and is swapped in via `self_replace` (which rename-asides the
//! running exe, copies the new one in, and schedules deferred deletion) — the product
//! no longer copies itself into any toolchain's directory, in either mode.
//!
//! ## Where the new file comes from
//!
//! The product is its own release host: [`RELEASE_REPO`] (`CARGO_PKG_REPOSITORY`)
//! holds one release per version, and the release workflow, the install scripts
//! and this file mirror one contract:
//!
//! * the newest release's tiny version file is
//!   `{base}/releases/latest/download/version.txt` and holds `<version>\n`;
//! * an exact version's archive is `{base}/releases/download/v{version}/{name}`;
//! * `{name}` is `mahbot-<version>-<os>-<arch>.<ext>`, with the `(os, arch)` pair
//!   [`crate::util::managed_bin::host_os_arch`] returns and `tar.gz` on
//!   macOS/Linux, `zip` on Windows;
//! * the archive holds exactly one file, `mahbot` (`mahbot.exe` on Windows), at
//!   its root.
//!
//! No hosting-service API call is made anywhere — only those download URLs.
//!
//! ## Test-only hooks
//!
//! [`RELEASE_BASE_URL_ENV`] and [`UPDATE_TO_VERSION_ENV`] drive a whole update of
//! this machinery with no window at all, from a release host other than the
//! product's own (see [`run_env_named_update`]): the first moves the
//! release base, the second names the version to move to — a test release is
//! deliberately undiscoverable, so the one being driven must be named outright.
//! Both are unset in every ordinary run, where the release base is [`RELEASE_REPO`]
//! and the version to move to is the newest published one.
//!
//! Single-instance enforcement is an exclusive whole-file lock on `mahbot.lock`,
//! released explicitly before the hand-off spawn — see [`acquire_lock`] and
//! [`crate::util::lock`]. The WAL checkpoint before `exit(0)` is a clean store
//! handoff: `std::process::exit(0)` bypasses all Rust destructors, so Turso
//! connections are never properly closed. The TRUNCATE leaves an empty WAL;
//! committed data is already fsync-durable at COMMIT.
//!
//! ## macOS Gatekeeper safety
//!
//! `posix_spawn` triggers async Gatekeeper code-signing validation; deleting the
//! spawn target during validation produces empty stderr (SIGKILL by
//! `syspolicyd`). The spawn target is always the working file the update left in
//! place — never a temp root — and the temp roots are removed before the lock
//! release and spawn (see [`finalize_update_and_restart`]), so the spawn target is
//! never deleted in its startup window.

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

/// The embedded package version, stamped at build time from `CARGO_PKG_VERSION`.
///
/// Cargo sets this from the `version` field in `Cargo.toml` for every build, so it
/// is the authoritative version of the running binary. The GUI surfaces it on the
/// Settings page, and a downloaded copy's self-update compares the release host's
/// version against it.
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

// ── File-lock based single-instance guard ─────────────────────────────────

/// Environment marker the updating parent sets (to `"1"`) on the replacement
/// instance it spawns; [`acquire_lock`] owns what it does. Nothing else sets it,
/// and it is inherited only where the environment is:
///
/// - git children run with a cleared environment (`tools::shell::apply_internal_env`);
/// - an agent's shell command is handed the owner's own environment, which
///   carries this marker on an instance the updater started — on unix, where that
///   shell's environment is the process's; on Windows the environment is
///   assembled from the account instead, so the marker does not reach it;
/// - the GUI Shell tab's PTY does not clear it either.
///
/// So a `mahbot` launched from the replacement instance's own terminal takes the
/// hand-off wait once before the ordinary refusal.
const HANDOFF_ENV: &str = "MAHBOT_UPDATE_HANDOFF";

/// How long a hand-off-marked instance waits for the previous instance to
/// release the location before refusing ([`acquire_lock`]).
const HANDOFF_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Poll interval of the hand-off wait ([`acquire_lock`]).
const HANDOFF_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Acquire an exclusive whole-file lock on `mahbot.lock`, returning an error if
/// another instance holds it.
///
/// `storage_root` is the directory where `mahbot.lock` is created — typically
/// [`crate::config::default_config_dir`].
///
/// The returned guard is stored in `INSTANCE_LOCK` for the process lifetime,
/// and released on process termination (`exit(0)` included) unless
/// [`FlockGuard::release`] did so first — `util::lock` owns that story.
///
/// # Update hand-off
///
/// The updating parent marks the replacement instance it spawns with
/// [`HANDOFF_ENV`], and this function then *waits* for a held location instead
/// of refusing: the parent releases the lock before spawning, but the child can
/// reach this point before that release is visible, and refusing there would
/// abandon a completed update's restart. The wait is bounded by
/// [`HANDOFF_LOCK_WAIT`] and falls through to the ordinary refusal. An unmarked
/// process (a genuine second launch) is refused immediately — no sleeps.
///
/// # Panics
///
/// Panics if called more than once (only called from `main()` at startup).
pub fn acquire_lock(storage_root: &Path) -> Result<()> {
    let lock_path = crate::util::lock::lock_file_path(storage_root);

    // Ensure parent directory exists.
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let file = match try_acquire_lock(&lock_path)? {
        Some(file) => file,
        None if std::env::var(HANDOFF_ENV).is_ok_and(|v| v == "1") => {
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
/// then fall back to the ordinary second-instance refusal.
///
/// Only reached by a hand-off-marked instance (see [`HANDOFF_ENV`]) whose
/// immediate attempt already failed.
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

/// The error a launch gets when another instance holds the location.
///
/// The wording is a matched surface — the grep engine recognises a stale
/// self-update binary by it (`shell::grep_engine::STALE_BINARY_LOCK_MSG`) — so
/// it must not change.
fn second_instance_refusal(lock_path: &Path) -> anyhow::Error {
    anyhow!(
        "Another instance of mahbot is already running (lock file: {}). \
         The lock is a kernel flock released automatically when that instance exits.",
        lock_path.display()
    )
}

/// Attempt to acquire an exclusive lock on the given file path.
///
/// Opens the file (creating it if necessary) and calls [`try_flock`], which takes
/// an exclusive whole-file lock without blocking. Returns immediately if another
/// process holds the lock.
///
/// # Returns
///
/// - `Ok(Some(file))` — lock acquired successfully. **The caller must keep the
///   returned `File` alive for the lifetime of the lock.**
/// - `Ok(None)` — another process holds the lock.
/// - `Err(...)` — a non-retryable OS error occurred (propagated from [`try_flock`]
///   or file open).
///
/// # Caller responsibilities
///
/// Shared helper used by [`acquire_lock`] (sync, at startup) and
/// [`reacquire_instance_lock`] (async, after failed spawn). Each caller handles
/// its own concerns:
///
/// - **Directory creation**: [`acquire_lock`] ensures the parent directory exists
///   before calling this helper.
/// - **Idempotency guard**: both callers check whether the lock is already held
///   before calling this helper. Calling this helper while already holding the
///   lock via a different `File` would fail (the two file descriptors are
///   independent from the kernel's perspective).
/// - **Error messages**: each caller formats its own success/failure messages.
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

/// Open (or create) the lock file with the standard set of options.
///
/// Extracted from [`try_acquire_lock`] so the same builder pattern is available
/// to both production code and tests.
fn open_lock_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

// ── FlockGuard: releasable instance lock ─────────────────────────────────

/// A guard holding the instance lock file.
///
/// The lock is released via [`release`](FlockGuard::release) — an explicit
/// unlock of the file, then closing the handle — or, at the latest, when the
/// guard is dropped. Re-acquisition after release is handled by
/// [`reacquire_instance_lock`] — needed when a self-update spawn fails and the
/// current process stays alive.
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
    /// Release the lock: explicitly unlock the file, then close the handle.
    /// Idempotent — no-op if already released.
    ///
    /// The explicit unlock is what the update hand-off relies on — see
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

/// Global instance lock, held for the process lifetime.
/// Stored in a static so [`execute_update`] can release and re-acquire it.
static INSTANCE_LOCK: OnceLock<Mutex<FlockGuard>> = OnceLock::new();

/// Release the instance lock so a child process can acquire it on startup.
///
/// Called just before spawning the new instance during self-update. No-op if the
/// lock is not initialized or already released.
async fn release_instance_lock() {
    if let Some(mutex) = INSTANCE_LOCK.get() {
        let mut guard = mutex.lock().await;
        guard.release();
    }
}

/// Re-acquire the instance lock after a failed spawn.
///
/// Called when [`spawn_new_instance_from`] fails — the current process stays
/// alive and must re-claim the lock.
///
/// This is a recoverable path: it runs during self-update after all agents
/// have been cancelled, chrome sessions closed, and shutdown signaled.
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

    // Re-acquire mutex and update guard with the re-acquired file.
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

// ── The product's own release files ───────────────────────────────────────

/// Where the product's own releases live: this crate's repository, one release
/// per version. The release workflow publishes to it and the install scripts
/// download from it, so this is the single source of truth for all three.
const RELEASE_REPO: &str = env!("CARGO_PKG_REPOSITORY");

/// Test-only: the release base a driven update downloads from, in place of
/// [`RELEASE_REPO`].
const RELEASE_BASE_URL_ENV: &str = "MAHBOT_RELEASE_BASE_URL";

/// Test-only: the version a driven update moves to, named outright so no
/// newest-release lookup (and no published newest release) is needed.
const UPDATE_TO_VERSION_ENV: &str = "MAHBOT_UPDATE_TO_VERSION";

/// How long downloading this system's archive may take. Generous on purpose: the
/// file is tens of megabytes and the connection may be slow, but it is still a
/// hard deadline so a stalled transfer cannot leave the update hanging forever.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(30);
/// How long the host's own `sw_vers`/`getconf` answer may take, when the floor
/// check asks it one ([`host_probe`]). Both answer at once, and the wait is bounded
/// like every other one on this path: a probe still running after this much is read
/// as one that answered nothing, which refuses nothing.
const HOST_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// The release base every download is built from — [`RELEASE_REPO`] unless
/// [`RELEASE_BASE_URL_ENV`] moved it.
fn release_base_url() -> String {
    std::env::var(RELEASE_BASE_URL_ENV).unwrap_or_else(|_| RELEASE_REPO.to_string())
}

/// The URL of the newest release's version file.
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

/// The URL of `version`'s archive for this system.
fn asset_url(base: &str, version: &semver::Version, os: &str, arch: &str) -> String {
    format!(
        "{base}/releases/download/v{version}/{}",
        asset_name(version, os, arch)
    )
}

// ── The floors a published file is built against ──────────────────────────

/// The floors the published files are built against: the deployment target both
/// macOS legs pass to the compiler, the glibc of the Linux base they are built on,
/// and the Windows build each Windows file is for (Windows 10 version 1809 on
/// x86_64, and Windows 11 on ARM — every Windows 10 on ARM is below the ARM64
/// file). A system below its floor has no file that can load, so it has no file for
/// it at all — the same refusal as an os/arch with no file published.
const MACOS_FLOOR: (u32, u32) = (12, 3);
const GLIBC_FLOOR: (u32, u32) = (2, 35);
const WINDOWS_X86_64_FLOOR: u32 = 17_763;
const WINDOWS_ARM64_FLOOR: u32 = 22_000;

/// The opening of every refusal for a system with no published file for it — the
/// same words both install scripts begin theirs with.
const NO_RELEASE_FILE: &str = "MahBot has no file for this system";

/// The refusal for a Linux host with no glibc at all: a file built for glibc has
/// nothing to load against there. `install.sh` prints these words too; the Windows
/// script has no glibc refusal to print.
const NO_GLIBC_FOUND: &str =
    "the released Linux files are built for glibc, and no glibc was found on this system";

/// The plain reason this system has no published release file for it, or `None`
/// when it has one.
///
/// The floors above and, on Linux, the musl loader
/// ([`crate::util::managed_bin::linux_host_is_musl`]) are the whole of it, and only
/// what can actually be read is consulted: a host that does not answer is not
/// refused for that, while a host that answers with a glibc below its floor is. The
/// install scripts refuse the same systems, so no file is ever put in place that
/// cannot load — which would leave the machine with no working copy at all.
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

/// The Windows build this host runs, when it can be read.
///
/// `RtlGetVersion` rather than `GetVersionExW`: the latter reports Windows 8's
/// version for an image that carries no compatibility manifest, which this one does
/// not, and every published Windows file is newer than that. `None` when it cannot
/// be read at all — nothing is refused on a guess.
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

/// Ask the host what it is, the way the install scripts ask it (`sw_vers`,
/// `getconf`): a program's trimmed stdout when it exits successfully, and nothing
/// at all otherwise — including when it does not answer in time, so a host that
/// never answers cannot hold the update open.
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

/// Whether `version` (`major[.minor[…]]`) is below `floor`. Each part is compared
/// as a whole number — a minor of 4 is not four tenths of 2.35 — a version with no
/// minor part counts as zero, and a part that is not a whole number is a version
/// this cannot compare: below the floor rather than accepted, exactly as the
/// install script's own check refuses it.
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
/// own report names none, and neither does no report.
fn names_glibc(report: &str) -> bool {
    report.starts_with("glibc ")
}

/// Which of the two ways a Linux host has no published file for it its own answers
/// amount to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinuxRefusal {
    /// A glibc below the floor the Linux files are built against.
    GlibcBelowFloor,
    /// No glibc was found, with musl's loader present: nothing to load a file built
    /// for glibc with.
    NoGlibc,
}

/// The refusal a Linux host's own answers amount to, or `None` when the published
/// file is one it can load.
///
/// What the host answers with decides when it answers with a glibc: at or above the
/// floor ([`GLIBC_FLOOR`]) the file loads and below it there is none for this system.
/// A host that names no glibc — another libc's own report, or no answer at all — is
/// then judged by musl's loader being there, the check the vendor installers make:
/// musl installed beside glibc is not a musl system, while a host that names no glibc
/// has none of its own to load the file with. A host with neither answers nothing
/// this can refuse on, and is not refused.
fn linux_refusal(report: Option<&str>, musl_loader: bool) -> Option<LinuxRefusal> {
    if report.is_some_and(glibc_below_floor) {
        return Some(LinuxRefusal::GlibcBelowFloor);
    }
    (!report.is_some_and(names_glibc) && musl_loader).then_some(LinuxRefusal::NoGlibc)
}

/// Whether a `getconf GNU_LIBC_VERSION` report names a glibc below the floor.
/// A report this does not recognise — a `getconf` that answers something else,
/// or does not answer at all — is not a refusal.
fn glibc_below_floor(report: &str) -> bool {
    report
        .strip_prefix("glibc ")
        .is_some_and(|version| version_below_floor(version, GLIBC_FLOOR))
}

/// The running binary's version, parsed from [`VERSION`].
fn current_version() -> Result<semver::Version> {
    semver::Version::parse(VERSION)
        .with_context(|| format!("embedded version {VERSION} is not valid semver"))
}

/// The version a driven update was told to move to, when it named one other than
/// the running version — `None` in every ordinary run.
///
/// Read by both the availability check and the update itself, so a driven update
/// cannot be shown one version and given another. The version is named by hand, so
/// it is taken in either spelling: the release's own mark carries the `v`
/// (`v0.7.0`), while the version itself — and every URL built from it — does not.
#[must_use]
fn update_target_override() -> Option<semver::Version> {
    let named = std::env::var(UPDATE_TO_VERSION_ENV).ok()?;
    let named = named.trim();
    let named = semver::Version::parse(named.strip_prefix('v').unwrap_or(named)).ok()?;
    let current = semver::Version::parse(VERSION).ok()?;
    (named != current).then_some(named)
}

/// HTTP client for the release check, with a descriptive User-Agent and a short
/// timeout so a hung network never stalls the refresh loop for long.
///
/// Build failure surfaces as `Err` (the builder has no reason to fail in
/// practice — fixed config, no env interaction). Both the background
/// availability-refresh task and the update's own newest-release lookup tolerate
/// the failure and report it; the periodic one retries on its next tick.
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

/// The newest published version held by the release host's version file.
///
/// A version carrying a pre-release part is ignored (reported as `Ok(None)`): a
/// test release is published that way and must never be discovered by an ordinary
/// copy — only named outright through [`UPDATE_TO_VERSION_ENV`]. A malformed body
/// and a network failure both surface as `Err`; callers tolerate them silently and
/// retry on the next tick.
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

/// The version a version file's body names, or `None` when it names a test
/// release: a version carrying a pre-release part is never offered to an ordinary
/// copy, which is what keeps test releases from reaching anybody who did not ask
/// for one (a driven update names the version outright through
/// [`UPDATE_TO_VERSION_ENV`] instead).
fn discoverable_version(body: &str) -> Result<Option<semver::Version>> {
    let version = semver::Version::parse(body.trim())
        .context("the newest release version file does not hold a version")?;
    Ok(version.pre.is_empty().then_some(version))
}

/// Check the product's own release host for a newer version of itself.
///
/// Returns `Ok(Some(latest))` when a newer version is published, `Ok(None)` when
/// up to date, and `Err` on a failed check (the caller retries later — never a
/// user-visible error for a failed check). A version named through
/// [`UPDATE_TO_VERSION_ENV`] is reported as-is, with no lookup at all.
async fn check_download_update() -> Result<Option<semver::Version>> {
    if let Some(named) = update_target_override() {
        return Ok(Some(named));
    }
    let Some(latest) = fetch_latest_release_version().await? else {
        return Ok(None);
    };
    Ok((latest > current_version()?).then_some(latest))
}

// ── Update availability ───────────────────────────────────────────────────

/// How the running binary was obtained — selects the self-update strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpdateMode {
    /// Built from a working tree (a checkout): the update rebuilds from that tree.
    SourceTree,
    /// A copy obtained as a ready-made published file: the update downloads the
    /// ready-made file for this system and puts it at the standard per-user
    /// location — the working file itself when that is already where this copy runs
    /// from, and a placement at that location (the working file taken away
    /// afterwards) when it is not.
    Downloaded,
}

/// Classify the update mode from a manifest-directory probe.
///
/// The manifest dir is the compile-time `CARGO_MANIFEST_DIR`. A copy that arrived
/// as a ready-made file keeps no working tree to rebuild from, but the probe
/// cannot simply look for `Cargo.toml`: the package registry extracts its crates
/// into `$CARGO_HOME/registry/src` and `$CARGO_HOME/git/checkouts`, both of which
/// contain one. So the discriminator:
///
/// 1. The path contains the package registry's source-cache layout (`registry/src`
///    or `git/checkouts` as path segments — CARGO_HOME may be custom, so the
///    `.cargo` prefix is not required) → a downloaded copy. This is checked
///    FIRST, before the `.git` heuristic, because cargo `git/checkouts` entries
///    are full non-bare clones and contain a real `.git` directory.
/// 2. `.git` at the manifest dir → a real working tree.
/// 3. Otherwise a reachable `Cargo.toml` → treat as a working tree (a source
///    tarball extract or a moved checkout still rebuilds fine).
/// 4. No source at all → a downloaded copy: its build source is gone, so the
///    ready-made file is the only way forward.
fn classify_update_mode(manifest_dir: &Path) -> UpdateMode {
    // 1. The package registry's source cache (registry src or git checkouts), as
    //    adjacent path segments — separator-agnostic (Windows uses backslashes).
    let mut prev: Option<&std::ffi::OsStr> = None;
    for component in manifest_dir.components() {
        if let (Some(a), std::path::Component::Normal(b)) = (prev, component)
            && ((a == "registry" && b == "src") || (a == "git" && b == "checkouts"))
        {
            return UpdateMode::Downloaded;
        }
        prev = match component {
            std::path::Component::Normal(os) => Some(os),
            _ => None,
        };
    }
    // 2. Real working tree: git metadata present.
    if manifest_dir.join(".git").exists() {
        return UpdateMode::SourceTree;
    }
    // 3/4. Reachable Cargo.toml → working tree; otherwise a downloaded copy.
    if manifest_dir.join("Cargo.toml").is_file() {
        UpdateMode::SourceTree
    } else {
        UpdateMode::Downloaded
    }
}

/// Determine how the running binary was installed — see [`UpdateMode`].
#[must_use]
pub(crate) fn update_mode() -> UpdateMode {
    classify_update_mode(Path::new(env!("CARGO_MANIFEST_DIR")))
}

// ── Shared update availability cache ──────────────────────────────────────

/// A point-in-time snapshot of the shared self-update availability cache.
///
/// Passed by value to pure predicates like [`should_show_update`] so they read
/// a consistent two-field view instead of observing intermediate cache writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UpdateAvailability {
    /// Whether an update is available (mode-aware: a copy built from sources
    /// always; a downloaded copy only when the release host has a newer version).
    pub(crate) available: bool,
    /// Whether an update is currently in flight (download/build running, and
    /// the finalizing window inclusive).
    pub(crate) in_progress: bool,
}

/// Process-local, single-source-of-truth cache of self-update availability
/// and progress. Both the GUI (button visibility, tooltip, finalize guards)
/// and the Telegram command menu (`/update` visibility + dispatch gate) read
/// this, so the two surfaces cannot diverge. Writers: `available` is
/// boot-seeded by [`update_cache`] and updated by [`refresh_update_cache`]
/// (the periodic background task reads `in_progress` but only ever writes
/// `available`); `in_progress` is written by [`execute_update`] and the
/// `/update` dispatch gate (compare-exchange). Reset implicitly on boot — a
/// fresh process has no stale `available` state from a previous run, so an
/// already-installed update is never advertised across a restart.
struct UpdateCache {
    /// Whether an update is available. A copy built from sources: statically
    /// true (its working tree is reachable). A downloaded copy: derived from the
    /// periodic release check, boot-seeded false until the first refresh tick.
    available: AtomicBool,
    /// The newest published version a downloaded copy found — drives the GUI
    /// "Update MahBot to vX" tooltip. `None` for a copy built from sources, and
    /// for a downloaded copy that is up to date or not yet checked.
    latest: std::sync::Mutex<Option<semver::Version>>,
    /// Whether an update is currently in flight. Set at [`execute_update`]
    /// entry, cleared on failure; a successful update exits the process.
    in_progress: AtomicBool,
}

static UPDATE_CACHE: OnceLock<UpdateCache> = OnceLock::new();

fn update_cache() -> &'static UpdateCache {
    UPDATE_CACHE.get_or_init(|| UpdateCache {
        // Boot-seed mode-aware: a copy built from sources is always available
        // without a network call; a downloaded copy starts unknown until the
        // first refresh tick.
        available: AtomicBool::new(update_mode() == UpdateMode::SourceTree),
        latest: std::sync::Mutex::new(None),
        in_progress: AtomicBool::new(false),
    })
}

/// Snapshot of the shared update-availability cache.
#[must_use]
pub(crate) fn update_availability() -> UpdateAvailability {
    let cache = update_cache();
    UpdateAvailability {
        available: cache.available.load(Ordering::SeqCst),
        in_progress: cache.in_progress.load(Ordering::SeqCst),
    }
}

/// The newest published version a downloaded copy found (see
/// [`refresh_update_cache`]). `None` for a copy built from sources, and for a
/// downloaded copy that is up to date or has not been checked yet.
#[must_use]
pub(crate) fn update_latest() -> Option<semver::Version> {
    update_cache().latest.lock().unwrap_poison().clone()
}

/// Whether an update is currently in flight (download/build running, and the
/// finalizing window inclusive).
#[must_use]
pub(crate) fn update_in_progress() -> bool {
    update_availability().in_progress
}

/// Pure visibility predicate for the `/update` menu entry in Telegram: the
/// entry is offered when the shared availability cache reports an update.
/// The menu reflects the cached state — it does NOT issue a network request
/// per refresh. (The GUI update button reads [`update_availability`] directly
/// and does not go through this predicate.)
#[must_use]
pub(crate) fn should_show_update(availability: UpdateAvailability) -> bool {
    availability.available
}

/// RAII guard returned by [`set_update_cache_for_test`] that restores the
/// cache to its prior state on drop, so a panicking test cannot leak a
/// mutated `available`/`in_progress` into later tests.
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

/// Set the shared update-availability cache for a test's duration and return a
/// guard that restores the prior state on drop.
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

// ── Update availability refresh ──────────────────────────────────────────

/// Refresh the shared update-availability cache once.
///
/// A copy built from sources seeds `available = true` (idempotent, no network).
/// A downloaded copy performs the release check, preserving the last-known
/// availability on a transient check failure so a network blip never hides a
/// previously discovered update. While an update is in flight the check is
/// skipped entirely (not just the write) — a long download would otherwise waste
/// a request per tick, and the InProgress status must never be clobbered. The
/// `in_progress` flag is re-checked after the network await so an update that
/// starts while the check is in flight cannot be clobbered by its result.
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
            // Re-check after the await: an update may have started while this
            // network request was in flight, and the write must not clobber
            // the in-progress state (which would hide the "Updating…" UI and
            // the `/update` command mid-update).
            if cache.in_progress.load(Ordering::SeqCst) {
                return;
            }
            match result {
                Ok(Some(latest)) => {
                    *cache.latest.lock().unwrap_poison() = Some(latest);
                    cache.available.store(true, Ordering::SeqCst);
                }
                Ok(None) => {
                    *cache.latest.lock().unwrap_poison() = None;
                    cache.available.store(false, Ordering::SeqCst);
                }
                Err(_) => {
                    // Transient failure — preserve last-known availability.
                }
            }
        }
    }
}

/// Periodic refresh of the shared update-availability cache.
///
/// Spawned from the binary's background task set (cancellable via the global
/// shutdown token). Ticks immediately so a fresh downloaded copy isn't hidden
/// until the first 10-minute interval elapses, then every 10 minutes. Runs on
/// all platforms/modes to keep the contract uniform — a copy built from sources
/// is a no-op network-wise but keeps `available` seeded.
pub async fn run_update_availability_refresh() {
    loop {
        refresh_update_cache().await;
        if !crate::shutdown::sleep_or_shutdown_or_drain(Duration::from_mins(10)).await {
            return;
        }
    }
}

// ── Update mutex ──────────────────────────────────────────────────────────

/// Global mutex ensuring only one update runs at a time.
/// A second trigger while an update is in progress gets an immediate error
/// via [`try_lock`](Mutex::try_lock).
static UPDATE_MUTEX: Mutex<()> = Mutex::const_new(());

// ── Telegram update notifications ─────────────────────────────────────────

/// The build from this working tree is done; the restart is reported by
/// [`UPDATE_RESTART_MSG`].
pub(crate) const UPDATE_BUILD_COMPLETE_MSG: &str = "✅ Build complete.";

/// The downloaded file is in place; the restart is reported by
/// [`UPDATE_RESTART_MSG`].
pub(crate) const UPDATE_DOWNLOAD_COMPLETE_MSG: &str = "✅ Update downloaded and put in place.";

/// Sent as the restart phase begins — nothing has restarted yet when this goes
/// out.
pub(crate) const UPDATE_RESTART_MSG: &str =
    "🔄 Waiting for work in progress to finish before the restart…";

// ── The durable record of a partly-finished update ────────────────────────

/// The target every partly-finished update is recorded under, and the message it
/// is recorded with. Durable rather than a tracing line because the process
/// restarts moments later, and one row per distinct reason because it must not
/// repeat — see [`record_update_unfinished`].
const UPDATE_UNFINISHED_TARGET: &str = "mahbot::self_update";
const UPDATE_UNFINISHED_MESSAGE: &str = "the update could not be completed in full";

/// A step of an update that could not be done, as it is recorded durably and told
/// to the admin.
struct UnfinishedStep {
    /// The step that failed — the whole of what is recorded, and never a path or
    /// anything else taken from the owner's environment.
    reason: &'static str,
    /// Whether the update stops here. A step that stops the update is already
    /// reported to the admin by the ordinary failure line, so its record raises no
    /// second sentence on the phone about the same fact; a step the update goes on
    /// past would otherwise be told to him nowhere.
    stops_update: bool,
}

/// The standard per-user programs directory could not be written to, so the new
/// version could not be put where the install scripts put it. The update stops here
/// and the copy that is already there keeps working.
const STANDARD_DIR_UNUSABLE: UnfinishedStep = UnfinishedStep {
    reason: "the standard per-user programs directory could not be written to",
    stops_update: true,
};

/// A second copy of the product — the one this instance came from when it moved
/// itself, or the one the old way of installing left in the toolchain's own
/// directory — could not be removed, so it stays on the disk beside the running
/// one. One reason for both, because they are the same failing step on the same kind
/// of file: a second sentence for it would be one fact told twice. The update goes
/// through, so this is the only word the admin gets about it.
const SECOND_COPY_REMAINS: UnfinishedStep = UnfinishedStep {
    reason: "a second copy of the product could not be taken away",
    stops_update: false,
};

/// Record a partly-finished update durably, and tell the admin about it once — when
/// it is not already being told about the same failing step by the ordinary failure
/// line (see [`UnfinishedStep::stops_update`]).
///
/// The record is written straight into the logs store — which is what the
/// product's own issues view reads — because the tracing writer is asynchronous
/// and the update path calls `exit(0)`, so a `tracing::warn!` here would be lost
/// with the process. It names only the step that failed, never a path and never a
/// value read from the owner or his environment, and a reason the store already
/// holds is not written again — across restarts, which is where the repeat would
/// otherwise come from. A read of what the store already holds that fails is no
/// answer at all: the step is recorded regardless, because a possible second row for
/// one reason is better than silence about it.
async fn record_update_unfinished(admin_target: Option<&str>, step: UnfinishedStep) {
    let Some(store) = crate::logs::LOG_STORE.get() else {
        warn!(
            reason = step.reason,
            "Logs store is not up — an unfinished update cannot be recorded"
        );
        return;
    };
    match store
        .has_reason(UPDATE_UNFINISHED_MESSAGE, step.reason)
        .await
    {
        // This reason has been recorded before: one fact, told once.
        Ok(true) => return,
        Ok(false) => {}
        Err(e) => warn!(
            error = %e,
            "Could not read whether this unfinished update was already recorded — recording it again"
        ),
    }
    let entry = crate::logs::LogEntry {
        timestamp: crate::db::now(),
        level: "WARN".to_string(),
        target: UPDATE_UNFINISHED_TARGET.to_string(),
        message: UPDATE_UNFINISHED_MESSAGE.to_string(),
        fields: serde_json::json!({ "reason": step.reason }),
        ..Default::default()
    };
    // The row is the durable record of what happened here.
    if let Err(e) = store.insert_batch(&[entry]).await {
        warn!(error = %e, "Could not record an unfinished update");
        return;
    }
    // The row above is the whole of what a fatal step adds: the ordinary failure
    // line already tells him which step failed.
    if step.stops_update {
        return;
    }
    let message = format!("⚠️ {UPDATE_UNFINISHED_MESSAGE}: {}.", step.reason);
    notify_admin(&message, admin_target).await;
}

// ── Execute update ────────────────────────────────────────────────────────

/// Set while [`execute_update`] runs its finalizing window (drain through
/// `exit(0)`). During this window the update path owns the process:
/// the GUI exit path waits ([`update_is_finalizing`]) instead of exiting, so a
/// window close, a platform quit or SIGINT cannot abort the update's checkpoint
/// on the iced runtime and leave the instance down without a replacement.
static UPDATE_FINALIZING: AtomicBool = AtomicBool::new(false);

/// True while [`execute_update`] is in its finalizing window (the instance is
/// shut down; checkpoint, temp-root cleanup, lock release, spawn, and `exit(0)`
/// pending).
#[must_use]
pub fn update_is_finalizing() -> bool {
    UPDATE_FINALIZING.load(Ordering::SeqCst)
}

/// Verify `cargo` is on PATH, returning an error with a mode-appropriate
/// message otherwise. The build-from-sources path only: updating a downloaded
/// copy never runs cargo.
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

/// Resolve the admin Telegram reply target for update notifications, logging
/// the rationale (info/warn) when notifications cannot be sent. Resolution is
/// per-call — a DB round trip to find the admin and its channel bindings.
/// Shared by both update paths.
async fn resolve_update_admin_target() -> Option<String> {
    let admin_target = resolve_admin_telegram_target().await;
    // Log info-level rationale when notifications cannot be sent.
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

/// Execute a self-update, dispatching on the update mode:
///
/// - [`UpdateMode::SourceTree`]: build from this working tree into a temp root,
///   swap the binary, restart.
/// - [`UpdateMode::Downloaded`]: download this system's ready-made release file,
///   put it at the standard per-user location, restart.
///
/// Called from the GUI update button and the Telegram `/update` command.
/// Only one update runs at a time — concurrent calls return an error immediately.
///
/// On success, this function never returns (`std::process::exit(0)`).
/// On failure, returns an error.
pub(crate) async fn execute_update() -> Result<()> {
    // Concurrent guard — only one update at a time. A second trigger while an
    // update is in progress gets an immediate error.
    let Some(_guard) = UPDATE_MUTEX.try_lock().ok() else {
        anyhow::bail!("{UPDATE_IN_PROGRESS_MSG}");
    };
    // Mark the shared in-progress state so both the GUI and the Telegram
    // `/update` gate report the update, and the GUI's exit-request/finalize
    // guards key off it (a Telegram-initiated update must also protect the
    // finalize/checkpoint window). Cleared on failure; a successful update
    // exits the process.
    update_cache().in_progress.store(true, Ordering::SeqCst);
    let result = match update_mode() {
        UpdateMode::SourceTree => execute_source_tree_update().await,
        UpdateMode::Downloaded => execute_downloaded_update().await,
    };
    if result.is_err() {
        update_cache().in_progress.store(false, Ordering::SeqCst);
    }
    result
}

/// Drive a whole update with no window at all, when a version was named through
/// [`UPDATE_TO_VERSION_ENV`]: wait for the stores and channels, then run
/// [`execute_update`].
///
/// Returns immediately unless the version named through [`UPDATE_TO_VERSION_ENV`]
/// differs from the running one *and* this copy updates by downloading — an
/// ordinary copy (no named version) never reaches the update here. It then waits
/// for the stores and the channels to come up before running [`execute_update`],
/// and reports a failure instead of restarting anything. The hook terminates by
/// itself: after the restart, the replacement instance sees the named version as
/// its own and does nothing. See the module doc — this and
/// [`RELEASE_BASE_URL_ENV`] are the only test-only paths in this file.
pub async fn run_env_named_update() {
    /// Long enough for the stores and the channels to be up when the update runs.
    const SETTLE_DELAY: Duration = Duration::from_secs(15);

    if update_target_override().is_none() || update_mode() != UpdateMode::Downloaded {
        return;
    }
    if !crate::shutdown::sleep_or_shutdown_or_drain(SETTLE_DELAY).await {
        return;
    }
    if let Err(e) = execute_update().await {
        error!(error = %e, "The update to the version named in the environment did not complete");
    }
}

/// The build-from-sources self-update: build from this working tree into a temp
/// root via `cargo install --path`, swap the running binary, and restart. See
/// [`execute_update`] for the concurrent-guard and exit contracts.
async fn execute_source_tree_update() -> Result<()> {
    // 1. Validate prerequisites.
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

    // Verify cargo is on PATH.
    verify_cargo_on_path("build from source").await?;

    // 2. Look up admin Telegram reply_target.
    let admin_target = resolve_update_admin_target().await;

    // 3. Notify: build started.
    notify_admin(
        "🔄 Update started — building from source…",
        admin_target.as_deref(),
    )
    .await;

    // 4. Create temp roots for the install and the cargo build. Held in scope
    //    for the whole update so an install/spawn failure RAII-cleans them; on
    //    success `std::process::exit(0)` bypasses RAII, so the finalize tail
    //    removes them explicitly before the lock release and spawn. The cold
    //    release rebuild of the heavy dep tree can be several-GB; the temp
    //    roots land under the (pinned) system temp dir, which on macOS is the
    //    boot volume with the repo, so no cross-volume fallback is warranted.
    let temp_install_root =
        tempfile::tempdir().context("Failed to create temp install root for self-update")?;
    let temp_build_dir =
        tempfile::tempdir().context("Failed to create temp build dir for self-update")?;
    let install_root = temp_install_root.path().to_path_buf();
    let build_dir = temp_build_dir.path().to_path_buf();

    // 5. Run `cargo install --path <repo> --root <temp_install> --locked
    //    --target-dir <temp_build>` so the build never touches the repo's
    //    `target/` (which would overwrite a running `target/release` artifact)
    //    and the install lands in the temp root. cwd is the storage root (it is
    //    guaranteed writable and exists, unlike the launch dir).
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

    // 6. Tail: validate, swap the running binary, notify, drain, restart.
    let fresh_binary = temp_bin_path(&install_root);
    finalize_source_tree_update(
        &fresh_binary,
        admin_target.as_deref(),
        vec![install_root, build_dir],
    )
    .await
}

/// The downloaded-copy self-update: fetch this system's ready-made release file,
/// extract it, put it at the standard per-user location, and restart. See
/// [`execute_update`] for the concurrent-guard and exit contracts.
async fn execute_downloaded_update() -> Result<()> {
    // 1. A system with no published file for it is refused before anything happens
    //    at all: downloading one and putting it in place would leave this copy
    //    replaced by a file that cannot load — no working copy, the one state an
    //    update must never produce. The install scripts refuse the same systems.
    if let Some(reason) = absent_release_file().await {
        anyhow::bail!("{reason} — no update was made.");
    }

    // 2. The version to move to. A version named through [`UPDATE_TO_VERSION_ENV`]
    //    is taken outright — any version other than the running one is a step it
    //    deliberately asked for; the ordinary path reads the newest published
    //    release and only ever moves forward, so a published release is never a
    //    step back. Nothing at all is installed when the version cannot be
    //    determined.
    let target = if let Some(named) = update_target_override() {
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

    // 3. The admin every notification goes to, and the file the new version goes
    //    in place of — or is placed at, for a copy whose own file sits elsewhere:
    //    the standard per-user programs location the install scripts use. A location
    //    that cannot be resolved cannot be written to either, so it is the same
    //    failing step as a placement that fails — recorded and reported the same way
    //    — and nothing is downloaded.
    let admin_target = resolve_update_admin_target().await;
    let Some(dest) = crate::util::managed_bin::mahbot_install_dir()
        .map(|dir| dir.join(crate::util::managed_bin::product_file_name()))
    else {
        record_update_unfinished(admin_target.as_deref(), STANDARD_DIR_UNUSABLE).await;
        anyhow::bail!("The standard install location could not be resolved — no update was made.");
    };

    // 4. Notify: download started (the Telegram channel must still be live for
    //    every notification in this function).
    notify_admin(
        "🔄 Update started — downloading the new version…",
        admin_target.as_deref(),
    )
    .await;

    // 5. Download this system's archive and extract the single file from it. The
    //    temp dir is held in scope for the whole update (RAII on error; explicit
    //    removal in the finalize tail, before the lock release and spawn).
    let temp_dir = tempfile::tempdir().context("Failed to create temp dir for self-update")?;
    let (os, arch) = crate::util::managed_bin::host_os_arch().map_err(|e| anyhow!("{e}"))?;
    let name = asset_name(&target, os, arch);
    let url = asset_url(&release_base_url(), &target, os, arch);
    let client = crate::util::http::build_download_client(DOWNLOAD_TIMEOUT)
        .context("Failed to build the download client for self-update")?;
    let archive = temp_dir.path().join(&name);
    crate::util::http::download_verified(
        &client,
        &url,
        &archive,
        // The product's own file: it is published beside the version that names
        // it, so there is deliberately no checksum to verify it against.
        "",
        Some(DOWNLOAD_TIMEOUT),
        crate::util::http::DownloadSizeCheck::Exact,
        |_, _| {},
    )
    .await
    .with_context(|| format!("Failed to download {name}"))?;
    // The archive runs a zip for Windows and a tar.gz everywhere else — the
    // platform split [`asset_name`] owns.
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

    // 6. Put it in place. When the running file IS the standard location, the
    //    swap rewrites that file in place and the restart target is that same
    //    path. Otherwise this is a ready-made copy whose file sits outside the
    //    standard location (it "moves itself" there): the new file is placed at
    //    the standard location and the restart target is that new copy.
    let current_exe = std::env::current_exe().context("Failed to resolve current_exe()")?;
    let (spawn_path, relocated) = if canonicalize_safe(&current_exe) == canonicalize_safe(&dest) {
        // The running file IS the standard location, so the swap rewrites it in
        // place. A swap that fails leaves the copy that is there working, and the
        // step that failed is recorded durably like every other.
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

    // 7. Take away the copies a working installation must not leave behind: the one
    //    this instance came from, when this copy moved itself, and the one the old
    //    way of installing (`cargo install mahbot`) left in the toolchain's own
    //    directory — a removal an earlier update may have failed to make, tried
    //    again here because this runs at every update. The running image itself needs
    //    the platform's own way of getting rid of it — `self_delete` unlinks the
    //    file on unix and, on Windows, where a running image cannot be unlinked,
    //    renames it aside and schedules the deletion. The update goes through either
    //    way: a leftover is recorded durably, never fatal.
    //
    //    The failure itself is a developer's line (it carries the platform's own
    //    error, path and all) and so is logged below the owner's view: what he sees
    //    of this step is the durable record, one row per distinct reason.
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

    // 8. Notify: in place, then wrapping up before the restart (MUST be before
    //    the shutdown in finalize_update_and_restart — the Telegram channel must
    //    still be live).
    notify_admin(UPDATE_DOWNLOAD_COMPLETE_MSG, admin_target.as_deref()).await;
    notify_admin(UPDATE_RESTART_MSG, admin_target.as_deref()).await;

    // 9. Shared finalize tail. The temp dir is never the spawn target (that is
    //    always the file the update left in place).
    finalize_update_and_restart(&spawn_path, vec![temp_dir.path().to_path_buf()]).await
}

/// The copy the old way of installing leaves behind — the product's own file in the
/// toolchain's binary directory (`$CARGO_HOME/bin`, else `~/.cargo/bin`) — when one
/// is there that is neither the file this instance is running from nor the file this
/// update has just put in place. `None` otherwise.
///
/// A downloaded copy is updated at the standard per-user programs directory from the
/// second update on, so without this the leftover of a failed move would never be
/// looked at again. It is computed fresh — from the toolchain's own convention, which
/// is where `cargo install` puts a command and where the install scripts look for it
/// too — rather than remembered, and the path is never part of what gets recorded.
///
/// `dest` is excluded because a `CARGO_HOME` can name the install directory itself
/// (`~/.local` on unix), where the toolchain's binary directory *is* the standard
/// per-user programs directory: removing that would take away the file the update
/// just put there, leaving nothing to start.
fn legacy_copy(dest: &Path) -> Option<PathBuf> {
    let path = crate::util::cargo_bin_dir()?.join(crate::util::managed_bin::product_file_name());
    if !path.is_file() {
        return None;
    }
    let path = canonicalize_safe(&path);
    // The file this instance is running from is the move's business, not this one's,
    // and the destination is what this update placed: only a copy that is neither is
    // removed here.
    if path == canonicalize_safe(&std::env::current_exe().ok()?) || path == canonicalize_safe(dest)
    {
        return None;
    }
    Some(path)
}

/// Path to the freshly built `mahbot` binary inside a cargo install temp root.
///
/// `cargo install --root <install_root>` places the produced binary at
/// `<install_root>/bin/mahbot` (`.exe` on Windows).
fn temp_bin_path(install_root: &Path) -> PathBuf {
    install_root
        .join("bin")
        .join(crate::util::managed_bin::product_file_name())
}

/// The build-from-sources tail, after `cargo install --path` produced the fresh
/// binary at `<temp_install>/bin/mahbot`.
///
/// 1. Validate that the fresh binary exists and is non-empty (bail otherwise —
///    a silent empty swap would strand the instance).
/// 2. Capture `current_exe()` BEFORE the swap — it is the restart target
///    (self_replace rewrites it in place, wherever it lives).
/// 3. Swap the running binary with the fresh one via `self_replace`. The source
///    differs from the running exe (it lives in the temp root), which
///    self-replace requires on Windows.
/// 4. Notify [`UPDATE_BUILD_COMPLETE_MSG`], then [`UPDATE_RESTART_MSG`] (the
///    Telegram channel must still be live for both).
/// 5. Hand off to [`finalize_update_and_restart`] for the drain → checkpoint →
///    temp-root removal → unlock → spawn-from-`current_exe` → exit.
///
/// `cleanup_paths` are the temp roots to remove before the instance-lock release
/// and spawn. On any error return the caller keeps the `TempDir` values in
/// scope, so their RAII drops clean them up; `std::process::exit(0)` bypasses
/// RAII, so the success path removes them in [`finalize_update_and_restart`]
/// instead.
async fn finalize_source_tree_update(
    fresh_binary: &Path,
    admin_target: Option<&str>,
    cleanup_paths: Vec<PathBuf>,
) -> Result<()> {
    // 1. Validate the freshly built binary.
    let len = fs::metadata(fresh_binary)
        .with_context(|| format!("Freshly built binary missing at {}", fresh_binary.display()))?
        .len();
    if len == 0 {
        anyhow::bail!(
            "Freshly built binary at {} is empty",
            fresh_binary.display()
        );
    }

    // 2. Capture the running exe before the swap — the restart target.
    let current_exe = std::env::current_exe().context("Failed to resolve current_exe()")?;

    // 3. Swap the running binary with the fresh one.
    self_replace::self_replace(fresh_binary)
        .with_context(|| format!("Failed to swap binary at {}", fresh_binary.display()))?;

    // 4. Notify: build complete (swap succeeded), then wrapping up before the
    //    restart (MUST be before the shutdown in finalize_update_and_restart —
    //    the Telegram channel must still be live).
    notify_admin(UPDATE_BUILD_COMPLETE_MSG, admin_target).await;
    notify_admin(UPDATE_RESTART_MSG, admin_target).await;

    // 5. Shared finalize tail. The temp roots are removed in the tail (before
    //    the instance-lock release and spawn) rather than by RAII: `exit(0)`
    //    bypasses destructors, and they are never the spawn target (which is
    //    always `current_exe`), so removal cannot race macOS Gatekeeper
    //    validation of the child.
    finalize_update_and_restart(&current_exe, cleanup_paths).await
}

/// Shared finalize tail for both update paths: graceful drain, final
/// single-writer checkpoint, temp-dir removal, instance-lock release, independent
/// spawn of the replacement, and `exit(0)`.
///
/// `spawn_path` is the file the update left in place — the working file the
/// replacement instance runs from. The ordering is load-bearing and MUST NOT be
/// rearranged:
///
/// 1. Graceful drain (the FULL drain, same as window close — NOT fast-cancel).
///    In-flight agents complete their current round; the drain-watch task fires
///    the global token when no in-flight agents or orchestrator calls remain
///    (or force-cancels at the 10-min cap). The GUI stays open with input
///    disabled; the GUI exit path waits ([`update_is_finalizing`]) instead of
///    exiting, so this sequence cannot be aborted by a window close, a platform
///    quit or SIGINT racing the checkpoint. No failure transitions with 'service
///    shutting down' comments fire — agents that cannot finish stay
///    status='launched' and boot-resume. The drain semantics are unchanged; then
///    `crate::tools::chrome_release::flush_and_close_all_chrome_sessions` releases
///    the sessions ended runs left queued and closes the rest.
/// 2. Checkpoint all databases BEFORE releasing the instance lock and spawning
///    the replacement. `exit(0)` below bypasses Rust destructors, so Turso
///    connections are never properly closed. With the lock still held this is
///    the last single-writer checkpoint: no checkpoint runs after the
///    replacement is live (the GUI exit path waits while the update is
///    finalizing — see `save_and_exit` / `update_is_finalizing`).
/// 3. Remove the temp update roots BEFORE releasing the instance lock or
///    spawning: `exit(0)` below bypasses Rust destructors, so the (potentially
///    multi-GB) deletion must complete here while the old process still holds
///    Turso's exclusive open-time fcntl locks on the store files (connections are
///    never formally closed). If the child booted while they were held, its
///    store open would fail on the lock — boot has only a bounded retry, not
///    immunity. Keeping the instance flock held during cleanup also prevents a
///    manually started second instance from sneaking in. Safe to delete before
///    spawn because the spawn target is ALWAYS the working file the update put in
///    place — never a temp root — so cleanup can never delete the binary the
///    child is validating; macOS syspolicyd SIGKILLs children whose binary is
///    deleted during async code-signature validation, and that invariant must
///    hold. In the build-from-sources path the working file is `current_exe()`
///    (in-place `self_replace`); in the downloaded-copy path it is the standard
///    location the new file was placed at.
/// 4. Release the instance lock so the child can acquire it.
/// 5. Spawn the new instance, marked as the update hand-off ([`HANDOFF_ENV`] —
///    see [`acquire_lock`]). On macOS, posix_spawn triggers asynchronous
///    Gatekeeper code signature validation. The spawn target is never a temp
///    root (see step 3), so the cleanup cannot race the validation.
/// 6. `exit(0)`.
///
/// On spawn failure (step 5) the process stays alive and the update returns
/// `Err`, having already removed its temp roots — acceptable, since they are
/// transient update artifacts (the caller keeps the `TempDir` values in scope
/// and RAII cleans any leftover).
async fn finalize_update_and_restart(spawn_path: &Path, cleanup_paths: Vec<PathBuf>) -> Result<()> {
    // 1. Begin the graceful drain.
    UPDATE_FINALIZING.store(true, Ordering::SeqCst);
    crate::shutdown::drain_begin();
    let token = crate::shutdown::shutdown_token();
    token.cancelled().await;
    crate::tools::chrome_release::flush_and_close_all_chrome_sessions().await;

    // 2. Checkpoint all databases BEFORE releasing the instance lock and
    //    spawning the replacement (see doc comment above).
    crate::db::checkpoint::checkpoint_all_databases().await;

    // 3. Remove the temp update roots BEFORE releasing the instance lock and
    //    spawning (see doc comment above): `exit(0)` bypasses destructors, so
    //    the multi-GB deletion must complete here while the old process still
    //    holds Turso's exclusive open-time locks. Best-effort — a failure to
    //    remove a stale root must not abort the update.
    for path in &cleanup_paths {
        if let Err(e) = fs::remove_dir_all(path) {
            warn!(
                error = %e,
                path = %path.display(),
                "Could not remove temp update root"
            );
        }
    }

    // 4. Release the instance lock so the child can acquire it on startup.
    release_instance_lock().await;

    // 5. Spawn the new instance from the determined spawn path, marked as the
    //    update hand-off.
    if let Err(e) = spawn_new_instance_from(spawn_path) {
        // Spawn failed — the process stays alive (unless a window close or a
        // platform quit arrived during the finalizing window, in which case the
        // GUI honors it with its own checkpoint + exit via UpdateResult).
        // Clear the finalizing flag and re-acquire the lock.
        UPDATE_FINALIZING.store(false, Ordering::SeqCst);
        // Re-acquire the lock since the process stays alive.
        if let Err(lock_err) = reacquire_instance_lock().await {
            error!(%lock_err, "Failed to re-acquire instance lock after spawn failure");
        }
        return Err(e);
    }

    // 6. Exit — spawn succeeded.
    crate::channels::chat_draft::flush_global();
    std::process::exit(0);
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// The two heads a cargo step's failure is reported under; see
/// [`CargoStepFailure`].
struct CargoStep {
    toast_head: &'static str,
    admin_head: &'static str,
}

impl CargoStep {
    /// The failure for a step that reported `body` — the text after the head,
    /// separator included.
    fn failure(&self, body: String) -> anyhow::Error {
        anyhow::Error::new(CargoStepFailure {
            toast_head: self.toast_head,
            admin_head: self.admin_head,
            body,
        })
    }
}

/// A failed cargo step of an update: one `body`, rendered under two heads.
///
/// [`Display`](std::fmt::Display) is the text the desktop toast has always
/// shown; [`update_failure_notification`] composes the Telegram line, whose
/// head names the step that failed.
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

/// Run a long-running cargo subcommand with a timeout — the build-from-sources
/// update path's only cargo runner.
///
/// `args` are the cargo arguments (excluding the `cargo` binary itself, each
/// as `&OsStr` so paths and flags pass through without Unicode assumption),
/// `cwd` the working directory, `timeout` the hard deadline, `label` a
/// short human-readable description used in logs and error messages (e.g.
/// "cargo install --path --locked"), and `step` the two heads a failure of this
/// step is reported under.
///
/// The caller passes `--target-dir` explicitly, which takes precedence over any
/// inherited `CARGO_TARGET_DIR` — that variable is stripped anyway, so no stray
/// value can redirect the build out of the tree the update cleans up after
/// itself. `CARGO_HOME` is NOT stripped — the child needs it for the
/// toolchain/registry cache.
///
/// On failure the error is returned (no admin notification — the caller owns
/// failure reporting via the single [`handle_update_command`]/GUI path).
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
            // Strip any inherited CARGO_TARGET_DIR so a stray value can never
            // redirect the build out of the tree this update cleans up.
            .env_remove("CARGO_TARGET_DIR")
            // kill_on_drop: if the timeout fires, the cargo child must die
            // too rather than keep compiling in the background. For
            // `cargo install` an orphaned child could even complete and swap
            // the binary after the instance reported "timed out", racing a
            // user retry.
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

/// Look up the admin's Telegram reply target.
///
/// Returns `Some(reply_target)` if the admin account has a Telegram channel
/// binding with a non-null `reply_target` and a bot token is configured.
/// Returns `None` otherwise.
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

/// Send a notification to the admin via Telegram.
pub async fn notify_admin(message: &str, target: Option<&str>) {
    let Some(recipient) = target else {
        return;
    };

    if crate::channel_registry().get("telegram").is_none() {
        warn!("Telegram channel not found in registry — cannot send update notification");
        return;
    }

    if let Err(e) =
        crate::channels::telegram::send_direct(recipient, message.to_string(), None).await
    {
        error!(error = %e, "Failed to send update notification to admin");
    }
}

/// Compose the Telegram failure text: the marker plus one plain statement of
/// what failed. A failed cargo step names itself (see [`CargoStepFailure`]);
/// every other failure reports its own error text.
///
/// Shared by both entry points, so the `/update` command and the window's Update
/// button report the same line. The desktop toast composes its own (see the
/// error's [`Display`](std::fmt::Display)).
pub(crate) fn update_failure_notification(err: &anyhow::Error) -> String {
    let statement = match err.downcast_ref::<CargoStepFailure>() {
        Some(failure) => format!("{}{}", failure.admin_head, failure.body),
        None => format!("{err:#}"),
    };
    format!("❌ {statement}")
}

/// Reply used when a command requires the admin. Used by both
/// the Telegram command dispatch (binary) and the `/update` handler (library)
/// so the denial wording stays consistent.
pub const ADMIN_ONLY_CMD_MSG: &str = "This command is only available to the admin.";

/// Reply for concurrent `/update` attempts — used both in the fast pre-check
/// and the atomic claim below, and as the `execute_update` contention error.
const UPDATE_IN_PROGRESS_MSG: &str =
    "An update is already in progress. Please wait for it to complete.";

/// Handle a Telegram `/update` command.
///
/// Gated on the shared availability cache (admin + update available), with a
/// synchronous reply for the early-failure cases (not an admin / already in
/// progress / no update / cargo not on PATH — the last only when this copy's
/// update has to build, so a downloaded copy is never held up by a toolchain it
/// does not use). The actual update runs as a
/// spawned async task so it does not block the Telegram message dispatch loop.
/// Progress notifications route via the normal update notification path (the
/// admin's Telegram binding); a failure is also reported directly to the
/// invoking admin.
/// There is NO confirmation modal — an admin invoking `/update` from Telegram
/// is itself sufficient confirmation.
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

    // Atomically claim the in-progress flag before spawning. This closes the
    // TOCTOU where a concurrent `/update` could pass the pre-check above, then
    // lose `UPDATE_MUTEX.try_lock` inside `execute_update` and be reported
    // through the failure message as "An update is already in progress."
    // `execute_update` keeps the flag set and clears it on failure.
    if update_cache()
        .in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
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
        if let Err(e) = execute_update().await {
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

/// Canonicalize a path, falling back to the lexical path on failure.
///
/// Used for canonicalized-path comparisons where the file may not exist yet or
/// where canonicalization may fail for other reasons (e.g., broken symlinks,
/// permission denied).
fn canonicalize_safe(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Spawn the new mahbot instance as an independent child process from the given
/// path.
///
/// The `binary_path` must point to an existing, executable binary — always the
/// working file the update left in place, and never a temp root (see
/// [`finalize_update_and_restart`]).
///
/// The child is marked as the update hand-off ([`HANDOFF_ENV`] — see
/// [`acquire_lock`]).
///
/// On Unix: null stdin/stdout, stderr → update.log. On Windows: the same, plus
/// `CREATE_NO_WINDOW`.
///
/// `CREATE_NO_WINDOW` is inert on this product's replacement instance: the binary
/// is built windowed (its crate root declares `windows_subsystem = "windows"`), so
/// the instance owns no console and the platform documents the flag as ignored for
/// a non-console application. It stays because it remains correct for a
/// console-subsystem build, and because the shell module's own spawn tripwire
/// requires the flag at every production spawn site. Every child the service
/// starts is flagged at that module's spawn sites, not here.
///
/// `DETACHED_PROCESS` is deliberately *not* set: the instance is this same windowed
/// image, so there is no console for it to inherit and none would be created for it —
/// the flag would change nothing. (A console-subsystem child started without creation
/// flags from a console-less parent does get a console of its own; the shell module's
/// spawn sites carry `CREATE_NO_WINDOW` to keep that one windowless, and
/// `tools::shell::tree` names them.)
///
/// The null stdin/stdout and the stderr → update.log redirection still matter: on
/// either platform update.log is a real file handle, so the replacement's printing
/// works with no console in sight.
///
/// On spawn failure the error is returned (no admin notification — the caller
/// [`finalize_update_and_restart`] owns failure reporting); the process keeps
/// running (does NOT exit).
///
/// ## macOS Gatekeeper safety
///
/// The caller guarantees that `binary_path` is never deleted before or during
/// the child's startup window (see the temp-root cleanup rationale in
/// [`finalize_update_and_restart`]). Deleting the spawn target while Gatekeeper
/// is validating its code signature causes `syspolicyd` to SIGKILL the child.
fn spawn_new_instance_from(binary_path: &Path) -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();

    info!(
        program = %binary_path.display(),
        args = ?args,
        "Spawning new mahbot instance"
    );

    let mut cmd = std::process::Command::new(binary_path);
    cmd.args(&args);
    // The update hand-off mark — see `acquire_lock`.
    cmd.env(HANDOFF_ENV, "1");

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

    cmd.stderr(Stdio::from(update_log));

    match cmd.spawn() {
        Ok(child) => {
            info!(pid = child.id(), "Spawned new mahbot instance");
            // Nothing waits on the child — it runs independently.
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

/// Truncate a string to its last 64KB, prepending a note if truncated.
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

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::lock::{lock_file_path, try_flock};

    #[test]
    fn test_truncate_to_last_64k_no_truncation() {
        let s = "hello world";
        assert_eq!(truncate_to_last_64k(s), "hello world");
    }

    #[test]
    fn test_truncate_to_last_64k_large_input() {
        let big = "X".repeat(70_000);
        let result = truncate_to_last_64k(&big);
        assert!(result.starts_with("[…output truncated;"));
        let x_count = result.chars().filter(|c| *c == 'X').count();
        assert_eq!(x_count, 64 * 1024);
    }

    #[test]
    fn test_try_acquire_lock_held_free() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = lock_file_path(dir.path());

        // Direct primitives (open_lock_file + try_flock): hold the lock, then
        // verify a second fd on the same file cannot flock while it is held.
        let holder = open_lock_file(&lock_path).unwrap();
        assert!(try_flock(&holder).unwrap(), "First flock should succeed");
        let contender = open_lock_file(&lock_path).unwrap();
        assert!(
            !try_flock(&contender).unwrap(),
            "Second flock should fail (already locked)"
        );

        // While the lock is held, try_acquire_lock should return None.
        assert!(
            try_acquire_lock(&lock_path).unwrap().is_none(),
            "Should return None when lock is held"
        );

        // The explicit release (what the update hand-off relies on) frees the
        // lock while the holder's handle is still open.
        holder.unlock().expect("unlock the held lock file");
        assert!(
            try_flock(&contender).unwrap(),
            "After the explicit unlock, the contender must acquire the lock"
        );

        // Dropping the handles must release the lock — through the production
        // helper, which opens the file the same way.
        drop(holder);
        drop(contender);
        assert!(
            lock_becomes_free(&lock_path).expect("lock the released lock file"),
            "After release, the lock must be acquirable again"
        );
    }

    /// The update hand-off wait: a marked instance must take the location once
    /// the outgoing instance releases it, and the release it waits for is the
    /// explicit `unlock` with the handle still open — exactly what
    /// [`FlockGuard::release`] does, and the reason it does not rely on the
    /// close.
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

        // The outgoing handle is closed only now: the lock the wait took is on
        // its own handle and must survive that close.
        drop(releaser.join().unwrap());
        assert!(
            location_is_held(&path),
            "the lock taken by the wait must survive the outgoing handle's close"
        );
        drop(taken);
    }

    /// Whether a fresh handle on the location finds it locked by someone else.
    fn location_is_held(path: &Path) -> bool {
        !try_flock(&open_lock_file(path).unwrap()).unwrap()
    }

    /// Poll for `path` to become lockable again, for up to ~2s.
    ///
    /// The release is polled rather than read once because every `Command::spawn`
    /// in the suite forks: until the child execs it still carries this process's
    /// open file description, and the kernel holds a flock for as long as any
    /// description of the file exists — so a release can look held for a moment.
    /// A lock that is never released (the regression this test exists for) comes
    /// back `Ok(false)`; an OS error comes back as it is, never polled away.
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
    fn test_lock_file_path_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let path = lock_file_path(dir.path());
        assert!(
            path.ends_with("mahbot.lock"),
            "Lock file path must end with mahbot.lock, got: {}",
            path.display(),
        );
    }

    #[test]
    fn test_update_mode_detection() {
        let dir = tempfile::tempdir().unwrap();

        // A working tree: `.git` present → SourceTree even with Cargo.toml.
        let checkout = dir.path().join("checkout");
        std::fs::create_dir_all(checkout.join(".git")).unwrap();
        std::fs::write(checkout.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&checkout), UpdateMode::SourceTree);

        // Registry src cache: `registry/src` segment → Downloaded even with
        // Cargo.toml (the cargo-installed trap the probe used to misclassify).
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

        // Git checkouts cache: `git/checkouts` segment → Downloaded. Cargo git
        // checkouts are full non-bare clones and contain a real `.git`
        // directory — the source-cache scan must win over the `.git`
        // heuristic (regression guard for that ordering).
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

        // Custom CARGO_HOME layout (no `.cargo` prefix): `registry/src` segment
        // still detected.
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

        // Plain source tree (no .git, no cargo cache): Cargo.toml → SourceTree.
        let plain = dir.path().join("plain-src");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&plain), UpdateMode::SourceTree);

        // No source at all → Downloaded (a ready-made copy is the only way
        // forward).
        let bare = dir.path().join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(classify_update_mode(&bare), UpdateMode::Downloaded);
    }

    /// The asset-name and URL derivation is the contract the release workflow and
    /// the install scripts mirror, so it is pinned here: the platform tag, the
    /// extension, and both URL shapes.
    #[test]
    fn test_release_asset_naming_and_urls() {
        let base = "https://example.test/owner/repo";
        let version = semver::Version::new(1, 2, 3);

        // The platform tag is the `(os, arch)` pair itself; only the extension
        // splits by platform.
        for (os, arch, name) in [
            ("macos", "x86_64", "mahbot-1.2.3-macos-x86_64.tar.gz"),
            ("macos", "aarch64", "mahbot-1.2.3-macos-aarch64.tar.gz"),
            ("linux", "x86_64", "mahbot-1.2.3-linux-x86_64.tar.gz"),
            ("linux", "aarch64", "mahbot-1.2.3-linux-aarch64.tar.gz"),
            ("windows", "x86_64", "mahbot-1.2.3-windows-x86_64.zip"),
            ("windows", "aarch64", "mahbot-1.2.3-windows-aarch64.zip"),
        ] {
            assert_eq!(asset_name(&version, os, arch), name, "{os}-{arch}");
        }

        // The newest release is found through its own tiny file, an exact version
        // through its tag.
        assert_eq!(
            latest_version_url(base),
            "https://example.test/owner/repo/releases/latest/download/version.txt"
        );
        assert_eq!(
            asset_url(base, &version, "windows", "x86_64"),
            "https://example.test/owner/repo/releases/download/v1.2.3/mahbot-1.2.3-windows-x86_64.zip"
        );

        // A pre-release version — what a test release carries — uses the same two
        // shapes, so it needs no separate path.
        let test_release = semver::Version::parse("1.2.3-rc.1").unwrap();
        assert_eq!(
            asset_url(base, &test_release, "linux", "aarch64"),
            "https://example.test/owner/repo/releases/download/v1.2.3-rc.1/\
             mahbot-1.2.3-rc.1-linux-aarch64.tar.gz"
        );
    }

    /// The version file's own rule: an ordinary copy is never offered a test
    /// release, and a file that holds no version is an error rather than an
    /// absence.
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
    /// builds it, so one contract is spelled in four files and only comments tie them
    /// together. Drift in any of them would otherwise be invisible: nothing else in
    /// the tree reads them, and the cross-checks only compile. Pinned here are the
    /// repository the releases live in, the two URL shapes, the asset name
    /// [`asset_name`] builds — against the workflow's own list of the six files, and
    /// the scripts' own spellings — the install location all three share, and the
    /// version file's location.
    #[test]
    fn test_the_release_contract_is_spelled_the_same_everywhere() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let read = |name: &str| std::fs::read_to_string(root.join(name)).expect("part of the tree");
        let (sh, ps1) = (read("install.sh"), read("install.ps1"));

        // The two URLs, each as this file builds it, with the base and the version
        // left as the variables the script sets them from: a path that changes on one
        // side fails here instead of leaving the scripts downloading nothing. The base
        // is the constant, not the test hook that can override it.
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
        // The scripts build an asset's name from their own variables, so what is
        // pinned there is the shape: `mahbot-<version>-<os>-<arch>.<ext>`.
        assert!(sh.contains(r#"ASSET="mahbot-$VERSION-$OS-$ARCH.tar.gz""#));
        assert!(ps1.contains(r#"$Asset = "mahbot-$Version-windows-$Arch.zip""#));

        // One install location for all three: the scripts put the ready-made file
        // there, the product moves itself there when its working file is elsewhere
        // (a downloaded copy that is not in the standard location moves itself in at
        // its next update), and the product's own search-path edit points at it. A
        // drift would leave exactly the second copy the whole arrangement prevents.
        // Each script's own literal is checked wherever this runs; the comparison
        // against the location this product computes only where it can be asked.
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

        // The workflow builds one file per system and publishes the release only once
        // all six are attached, so its own list of them is the release's definition.
        let workflow = read(".github/workflows/release.yml");
        floors_are_spelled_the_same(&sh, &ps1, &workflow);

        // The words the owner is refused with are one fact too: the scripts begin
        // their refusals with this file's own opening, and name the same requirement
        // for a Linux host with no glibc at all.
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
    }

    /// The floors a published file is built against are one fact told by the machinery
    /// that has to agree on them: the install command refuses below them, the update
    /// path refuses below them, and the workflow builds the files against them. A
    /// drift offers a file to a system it cannot load on, which is the one outcome the
    /// whole arrangement exists to prevent. What the README claims in prose is not
    /// pinned here: a rewording of it is not a drift in any of these.
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
        // A version with no minor part counts as zero, and a trailing part is not
        // part of the comparison.
        assert!(version_below_floor("12", MACOS_FLOOR));
        assert!(!version_below_floor("12.3", MACOS_FLOOR));
        assert!(!version_below_floor("12.3.1", MACOS_FLOOR));
        assert!(!version_below_floor("13", MACOS_FLOOR));
        // A part that is not there or not a whole number is not a version to accept,
        // in either position: the install script's check refuses both.
        assert!(version_below_floor("", MACOS_FLOOR));
        assert!(version_below_floor("12.", MACOS_FLOOR));
        assert!(version_below_floor("12.x", MACOS_FLOOR));
        assert!(version_below_floor("13.x", MACOS_FLOOR));
        // The glibc report is read the same way, and a report this does not
        // recognise is not a refusal.
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
        // At or above it there is a file to load: musl's loader being there as well —
        // it is under a symlinked `/lib` on Debian, where musl can be installed
        // beside glibc — is not a refusal.
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
        // A `CARGO_HOME` whose binary directory is the install directory itself, so
        // the toolchain's own convention names the file the update has just placed.
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

    #[test]
    fn test_canonicalize_safe_nonexistent_path() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("does_not_exist");
        // For a nonexistent path, canonicalize_safe should return the lexical path.
        let result = canonicalize_safe(&nonexistent);
        assert_eq!(result, nonexistent);
    }

    #[test]
    fn test_canonicalize_safe_existing_path() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_file.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let result = canonicalize_safe(&file_path);
        assert!(
            result.ends_with("test_file.txt"),
            "Canonicalized path should end with test_file.txt, got {}",
            result.display(),
        );
    }
}
