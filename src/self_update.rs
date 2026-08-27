//! Self-update logic — single-instance guarding, install-to-temp, binary swap,
//! cargo-bin refresh, and restart.
//!
//! Two update modes are supported, selected at runtime by [`update_mode`]. Both
//! converge on a single install-to-temp then `self_replace` flow:
//!
//! * **Local checkout**: the binary was built from a local source checkout
//!   (`.git` present at `CARGO_MANIFEST_DIR`, or a plain source tree with
//!   `Cargo.toml`). Update = `cargo install --path <repo> --root <temp_install>
//!   --locked --target-dir <temp_build>` → swap → copy to cargo bin →
//!   shutdown → checkpoint → restart.
//! * **Registry**: `CARGO_MANIFEST_DIR` points into the cargo registry/git
//!   source cache (e.g. `~/.cargo/registry/src/`), so a local rebuild would be a
//!   same-version no-op. Update = periodic crates.io sparse-index check for a
//!   strictly newer stable version → `cargo install <crate> --root <temp_install>
//!   --force` → swap → copy to cargo bin → shutdown → checkpoint → restart.
//!
//! The running executable cannot be replaced in place on Windows, so each mode
//! installs the freshly built binary into a temp root first, then swaps it in via
//! `self_replace` (which rename-asides the running exe, copies the new one in,
//! and schedules deferred deletion). This removes the previous Windows guards
//! and the bespoke local build-swap machinery.
//!
//! After the swap the fresh binary is also copied to the cargo install bin path
//! (`$CARGO_HOME/bin` / `~/.cargo/bin`) so PATH invocations of `mahbot` stay
//! fresh even when the daemon runs from a different path (e.g. the repo's
//! `target/release`). Uses `flock()` for single-instance enforcement. The WAL
//! checkpoint before `exit(0)` is a clean store handoff: `std::process::exit(0)`
//! bypasses all Rust destructors, so Turso connections are never properly
//! closed. The TRUNCATE leaves an empty WAL; committed data is already
//! fsync-durable at COMMIT.
//!
//! ## macOS Gatekeeper safety
//!
//! `posix_spawn` triggers async Gatekeeper code-signing validation; deleting the
//! spawn target during validation produces empty stderr (SIGKILL by
//! `syspolicyd`). In the temp-root flow the spawn target is always the captured
//! `current_exe()` (never a temp root), and the temp roots are only removed
//! after a successful spawn, so the spawn target is never deleted in its
//! startup window.

use crate::ChannelMessage;
use crate::util::UnwrapPoison;
use anyhow::{Context, Result, anyhow};
#[cfg(test)]
use directories::UserDirs;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

/// The embedded package version, stamped at build time from `CARGO_PKG_VERSION`.
///
/// Cargo sets this from the `version` field in `Cargo.toml` for every build —
/// including `cargo install` builds from crates.io — so it is the authoritative
/// version of the running binary. The GUI surfaces it (Settings → About) and
/// registry-mode self-update compares against it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ── File-lock based single-instance guard ─────────────────────────────────

/// Acquire an exclusive lock on the global lock file, returning an error
/// if another instance holds it.
///
/// `storage_root` is the directory where `mahbot.lock` is created — typically
/// [`crate::config::default_config_dir`].
///
/// The returned guard is stored in `INSTANCE_LOCK` for the process lifetime.
/// The kernel automatically releases the lock on process termination
/// (including `exit(0)`), and [`execute_update`] releases it explicitly during
/// self-update so the child can re-acquire on restart.
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

    match try_acquire_lock(&lock_path)? {
        Some(file) => {
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
        None => Err(anyhow!(
            "Another instance of mahbot is already running (lock file: {}). \
             If no other instance is running, delete this file manually.",
            lock_path.display()
        )),
    }
}

/// Attempt to acquire an exclusive lock on the given file path.
///
/// Opens the file (creating it if necessary) and tries `flock(LOCK_EX|LOCK_NB)`.
/// Returns immediately if another process holds the lock.
///
/// # Returns
///
/// - `Ok(Some(file))` — lock acquired successfully. **The caller must keep the
///   returned `File` alive for the lifetime of the lock** — dropping it releases
///   the kernel-level lock.
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
///   lock via a different `File` would fail with `EAGAIN` (the two file
///   descriptors are independent from the kernel's perspective).
/// - **Error messages**: each caller formats its own success/failure messages.
fn try_acquire_lock(path: &Path) -> Result<Option<File>> {
    let file = open_lock_file(path)
        .with_context(|| format!("failed to open lock file {}", path.display()))?;

    if crate::util::lock::try_flock(&file)
        .with_context(|| format!("flock failed on lock file {}", path.display()))?
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
/// The lock is released via [`release`](FlockGuard::release) (or on drop).
/// Re-acquisition after release is handled by [`reacquire_instance_lock`] —
/// needed when a self-update spawn fails and the current process stays alive.
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
    /// Release the lock by closing the underlying file descriptor.
    /// Idempotent — no-op if already released.
    fn release(&mut self) {
        if self.file.take().is_some() {
            info!(path = %self.lock_path.display(), "Released instance lock");
        }
    }
}

/// Global instance lock, held for the process lifetime.
/// Stored in a static so [`execute_update`] can release and re-acquire it.
static INSTANCE_LOCK: OnceLock<Mutex<FlockGuard>> = OnceLock::new();

/// Release the instance lock so a child process can acquire it on startup.
///
/// Called just before spawning the new instance during self-update.
/// No-op if the lock is not initialized or already released.
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
/// have been cancelled, browser sessions closed, and shutdown signaled.
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

// ── Update availability ───────────────────────────────────────────────────

/// How the running binary was installed — selects the self-update strategy.
///
/// [`LocalCheckout`](UpdateMode::LocalCheckout): built from a local source
/// checkout; update rebuilds from that source.
///
/// [`Registry`](UpdateMode::Registry): installed via `cargo install` (or a
/// binary whose build source is unreachable); update checks crates.io and runs
/// `cargo install <crate> --force`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpdateMode {
    /// Built from a local source checkout (`.git` present, or a plain source
    /// tree with `Cargo.toml` at the compile-time manifest dir).
    LocalCheckout,
    /// Installed via cargo — the manifest dir points into the cargo source
    /// cache (`~/.cargo/registry/src/…` or `~/.cargo/git/checkouts/…`), or no
    /// source is reachable at all. A local rebuild would be a same-version
    /// no-op, so updates come from crates.io.
    Registry,
}

/// Classify the update mode from a manifest-directory probe.
///
/// The manifest dir is the compile-time `CARGO_MANIFEST_DIR`. The historical
/// probe (Cargo.toml present) is NOT a reliable local-checkout discriminator:
/// `cargo install` keeps the extracted crate in `$CARGO_HOME/registry/src` and
/// `cargo install --git` in `$CARGO_HOME/git/checkouts`, both of which contain
/// a `Cargo.toml` — rebuilding there would be a same-version no-op. The
/// intent-aligned discriminator:
///
/// 1. The path contains the cargo source-cache layout (`registry/src` or
///    `git/checkouts` as path segments — CARGO_HOME may be custom, so the
///    `.cargo` prefix is not required) → registry-installed. This is checked
///    FIRST, before the `.git` heuristic, because cargo `git/checkouts`
///    entries are full non-bare clones and contain a real `.git` directory.
/// 2. `.git` at the manifest dir → a real local checkout.
/// 3. Otherwise a reachable `Cargo.toml` → treat as a local build (a source
///    tarball extract or a moved checkout still rebuilds fine).
/// 4. No source at all → registry mode (the only viable update path).
fn classify_update_mode(manifest_dir: &Path) -> UpdateMode {
    // 1. Cargo source cache (registry src or git checkouts), as adjacent path
    //    segments — separator-agnostic (Windows uses backslashes).
    let mut prev: Option<&std::ffi::OsStr> = None;
    for component in manifest_dir.components() {
        if let (Some(a), std::path::Component::Normal(b)) = (prev, component)
            && ((a == "registry" && b == "src") || (a == "git" && b == "checkouts"))
        {
            return UpdateMode::Registry;
        }
        prev = match component {
            std::path::Component::Normal(os) => Some(os),
            _ => None,
        };
    }
    // 2. Real local checkout: git metadata present.
    if manifest_dir.join(".git").exists() {
        return UpdateMode::LocalCheckout;
    }
    // 3/4. Reachable Cargo.toml → local build; otherwise registry.
    if manifest_dir.join("Cargo.toml").is_file() {
        UpdateMode::LocalCheckout
    } else {
        UpdateMode::Registry
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
/// Passed by value to pure predicates like [`should_show_update`] so the
/// registry-hidden and restricted-user branches are deterministically
/// testable without a network call or a real newer version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UpdateAvailability {
    /// Whether an update is available (mode-aware: local checkout always;
    /// registry only when a strictly newer stable version is published).
    pub(crate) available: bool,
    /// Whether an update is currently in flight (build/install running, and
    /// the finalizing window inclusive).
    pub(crate) in_progress: bool,
}

/// Process-local, single-source-of-truth cache of self-update availability
/// and progress. Both the GUI (button visibility, tooltip, finalize guards)
/// and the Telegram command menu (`/update` visibility + dispatch gate) read
/// this; a single periodic background task is the sole writer of the
/// availability fields, so the two surfaces cannot diverge. Reset implicitly
/// on boot — a fresh process has no stale `available` state from a previous
/// run, so an already-installed update is never advertised across a restart.
struct UpdateCache {
    /// Whether an update is available. Local checkout: statically true (the
    /// build checkout is reachable). Registry: derived from the periodic
    /// crates.io check, boot-seeded false until the first refresh tick.
    available: AtomicBool,
    /// The discovered latest published stable version (registry mode) — drives
    /// the GUI "Update MahBot to vX" tooltip. `None` in local-checkout mode,
    /// and in registry mode when up to date / not yet checked.
    latest: std::sync::Mutex<Option<semver::Version>>,
    /// Whether an update is currently in flight. Set at [`execute_update`]
    /// entry, cleared on failure; a successful update exits the process.
    in_progress: AtomicBool,
}

static UPDATE_CACHE: OnceLock<UpdateCache> = OnceLock::new();

fn update_cache() -> &'static UpdateCache {
    UPDATE_CACHE.get_or_init(|| UpdateCache {
        // Boot-seed mode-aware: local checkout is always available without a
        // network call; registry starts unknown until the first refresh tick.
        available: AtomicBool::new(update_mode() == UpdateMode::LocalCheckout),
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

/// The latest published stable version discovered by the periodic registry
/// check (registry mode). `None` in local-checkout mode, and in registry mode
/// when up to date or no newer version exists yet.
#[must_use]
pub(crate) fn update_latest() -> Option<semver::Version> {
    update_cache().latest.lock().unwrap_poison().clone()
}

/// Whether an update is currently in flight (build/install running, and the
/// finalizing window inclusive).
#[must_use]
pub(crate) fn update_in_progress() -> bool {
    update_availability().in_progress
}

/// Pure visibility predicate for the `/update` command (and the GUI button):
/// shown only to full-permission (admin) users when an update is available.
/// The menu reflects the shared cached availability — it does NOT issue a
/// network request per refresh.
#[must_use]
pub(crate) fn should_show_update(availability: UpdateAvailability, is_admin: bool) -> bool {
    is_admin && availability.available
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

// ── crates.io registry check (registry mode) ─────────────────────────────

/// HTTP client for the crates.io sparse index, with a descriptive User-Agent
/// (crates.io requires one; an empty UA is rejected) and a short timeout so a
/// hung network never blocks the GUI tick.
///
/// Build failure surfaces as `Err` (the builder has no reason to fail in
/// practice — fixed config, no env interaction — but an `expect()` here would
/// panic the GUI tick on the off chance it does).
fn registry_http_client() -> Result<&'static reqwest::Client> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            crate::util::http::install_ring_provider();
            reqwest::Client::builder()
                .user_agent(format!("mahbot/{VERSION} (self-update check)"))
                .timeout(Duration::from_secs(15))
                .build()
                .map_err(|e| format!("failed to build crates.io registry HTTP client: {e}"))
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

/// Sparse-index path for a crate name (cargo's own layout):
/// `1/{name}`, `2/{name}`, `3/{first}/{name}`, else `{first2}/{next2}/{name}`.
///
/// Byte-slicing is safe: the only caller passes the compile-time
/// `CARGO_PKG_NAME`, which cargo enforces as ASCII `[a-zA-Z0-9_-]`.
fn sparse_index_path(name: &str) -> String {
    let len = name.len();
    match len {
        1 => format!("1/{name}"),
        2 => format!("2/{name}"),
        3 => format!("3/{}/{name}", &name[..1]),
        _ => format!("{}/{}/{}", &name[..2], &name[2..4], name),
    }
}

/// Parse the crates.io sparse-index NDJSON body and return the newest
/// non-yanked stable version.
///
/// Each line is a JSON object `{"name", "vers", "yanked", …}`. The index
/// appends a line per publish/yank/unyank event, so a version's LAST line is
/// its current state ("last line wins") — a version that went through
/// yank → unyank → re-yank must count as yanked even though an earlier line
/// said unyanked. Pre-release versions are excluded (matching
/// `max_stable_version` — what plain `cargo install <crate>` installs) and
/// yanked versions never count.
fn latest_stable_version(index_body: &str) -> Option<semver::Version> {
    // Resolve each version's final yanked flag first (last line wins).
    let mut yanked: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    for line in index_body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Malformed lines are skipped (tolerance for index format drift).
        let Ok(record) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(vers) = record.get("vers").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let is_yanked = record
            .get("yanked")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        yanked.insert(vers.to_string(), is_yanked);
    }
    yanked
        .into_iter()
        .filter(|(_, is_yanked)| !is_yanked)
        .filter_map(|(vers, _)| {
            let version = semver::Version::parse(&vers).ok()?;
            if !version.pre.is_empty() {
                return None;
            }
            Some(version)
        })
        .max()
}

/// Query the crates.io sparse index for the newest stable published version of
/// this crate. Network failures surface as `Err` — callers tolerate them
/// silently and retry on the next tick.
async fn fetch_latest_stable_version() -> Result<Option<semver::Version>> {
    let name = env!("CARGO_PKG_NAME");
    let url = format!("https://index.crates.io/{}", sparse_index_path(name));
    let response = registry_http_client()?
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to query crates.io index for {name}"))?;
    if !response.status().is_success() {
        anyhow::bail!(
            "crates.io index returned HTTP {} for {name}",
            response.status()
        );
    }
    let body = response
        .text()
        .await
        .with_context(|| format!("failed to read crates.io index response for {name}"))?;
    Ok(latest_stable_version(&body))
}

/// Check whether a strictly newer stable version of this crate is published on
/// crates.io.
///
/// Returns `Ok(Some(latest))` when `latest > embedded VERSION`, `Ok(None)`
/// when up to date, and `Err` on network failure (the caller retries later —
/// never a user-visible error for a failed check).
///
/// Discovery runs on all platforms: an update found here is installed via the
/// unified install-to-temp + `self_replace` flow, which works on Windows too.
pub(crate) async fn check_registry_update() -> Result<Option<semver::Version>> {
    let Some(latest) = fetch_latest_stable_version().await? else {
        return Ok(None);
    };
    let current = semver::Version::parse(VERSION)
        .with_context(|| format!("embedded version {VERSION} is not valid semver"))?;
    Ok((latest > current).then_some(latest))
}

// ── Update availability refresh ──────────────────────────────────────────

/// Refresh the shared update-availability cache once.
///
/// Local-checkout mode seeds `available = true` (idempotent, no network).
/// Registry mode performs the crates.io check, preserving the last-known
/// availability on a transient check failure so a network blip never hides a
/// previously discovered update. While an update is in flight the check is
/// skipped entirely (not just the write) — a long cargo install would
/// otherwise waste a request per tick, and the InProgress status must never be
/// clobbered. The `in_progress` flag is re-checked after the network await so
/// an update that starts while the check is in flight cannot be clobbered by
/// its result.
async fn refresh_update_cache() {
    let cache = update_cache();
    match update_mode() {
        UpdateMode::LocalCheckout => {
            cache.available.store(true, Ordering::SeqCst);
        }
        UpdateMode::Registry => {
            if cache.in_progress.load(Ordering::SeqCst) {
                return;
            }
            let result = check_registry_update().await;
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
/// shutdown token). Ticks immediately so a fresh registry install isn't hidden
/// until the first 10-minute interval elapses, then every 10 minutes. Runs on
/// all platforms/modes to keep the contract uniform — local-checkout mode is a
/// no-op network-wise but keeps `available` seeded.
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

// ── Execute update ────────────────────────────────────────────────────────

/// Set while [`execute_update`] runs its finalizing window (step 10 shutdown
/// through `exit(0)`). During this window the update path owns the process:
/// the GUI exit path waits ([`update_is_finalizing`]) instead of exiting, so a
/// window close or SIGINT cannot abort the update's checkpoint on the iced
/// runtime and leave the daemon down without a replacement.
static UPDATE_FINALIZING: AtomicBool = AtomicBool::new(false);

/// True while [`execute_update`] is in its finalizing window (daemon shut
/// down; checkpoint, lock release, spawn, and `exit(0)` pending).
#[must_use]
pub fn update_is_finalizing() -> bool {
    UPDATE_FINALIZING.load(Ordering::SeqCst)
}

/// Verify `cargo` is on PATH, returning an error with a mode-appropriate
/// message otherwise. Shared by both update modes.
async fn verify_cargo_on_path(action: &str) -> Result<()> {
    match tokio::process::Command::new("cargo")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
    {
        Ok(status) if status.success() => Ok(()),
        _ => anyhow::bail!("cargo not found on PATH — cannot {action}"),
    }
}

/// Resolve the admin Telegram reply target for update notifications,
/// memoizing the rationale when notifications cannot be sent. Shared by both
/// update modes.
async fn resolve_update_admin_target() -> Option<String> {
    let admin_target = resolve_admin_telegram_target().await;
    // Memoize info-level rationale for missing notifications.
    if admin_target.is_none() {
        if crate::config::CONFIG.telegram_bot_token().is_none() {
            info!("No Telegram bot token configured — skipping update notifications");
        } else {
            warn!(
                "Admin user 'admin' has no Telegram channel binding with a reply_target. \
                 Update notifications will be skipped. \
                 Bind a Telegram channel to the admin user to receive update notifications."
            );
        }
    }
    admin_target
}

/// Execute a self-update, dispatching on the install mode:
///
/// - [`UpdateMode::LocalCheckout`]: build from the local checkout into a temp
///   root, swap the binary, refresh the cargo bin copy, restart.
/// - [`UpdateMode::Registry`]: `cargo install <crate> --force` into a temp root
///   from crates.io, swap the binary, refresh the cargo bin copy, restart.
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
        anyhow::bail!("An update is already in progress. Please wait for it to complete.");
    };
    // Mark the shared in-progress state so both the GUI and the Telegram
    // `/update` gate report the update, and the GUI's window-close/finalize
    // guards key off it (a Telegram-initiated update must also protect the
    // finalize/checkpoint window). Cleared on failure; a successful update
    // exits the process.
    update_cache().in_progress.store(true, Ordering::SeqCst);
    let result = match update_mode() {
        UpdateMode::LocalCheckout => execute_local_update().await,
        UpdateMode::Registry => execute_registry_update().await,
    };
    if result.is_err() {
        update_cache().in_progress.store(false, Ordering::SeqCst);
    }
    result
}

/// Local-checkout self-update: build from the source checkout into a temp
/// root via `cargo install --path`, swap the running binary, refresh the
/// cargo bin copy, and restart. See [`execute_update`] for the
/// concurrent-guard and exit contracts.
async fn execute_local_update() -> Result<()> {
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
    //    success `std::process::exit(0)` bypasses RAII, so the `after_spawn`
    //    closure in `finalize_install` removes them explicitly. The cold
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
        "Build",
    )
    .await?;

    // 6. Shared tail: swap, PATH freshness, drain, checkpoint, restart.
    let fresh_binary = temp_bin_path(&install_root);
    finalize_install(
        &fresh_binary,
        admin_target.as_deref(),
        "✅ Build complete. Restarting…",
        vec![install_root, build_dir],
    )
    .await
}

/// Registry self-update: `cargo install <crate> --force` from crates.io into a
/// temp root, swap the running binary, refresh the cargo bin copy, and restart.
/// See [`execute_update`] for the concurrent-guard and exit contracts.
async fn execute_registry_update() -> Result<()> {
    // Verify cargo is on PATH.
    verify_cargo_on_path("install from crates.io").await?;

    // 1. Look up admin Telegram reply_target.
    let admin_target = resolve_update_admin_target().await;

    // 2. Notify: install started.
    notify_admin(
        "🔄 Update started — installing from crates.io…",
        admin_target.as_deref(),
    )
    .await;

    // 3. Create the temp install root, held for the whole update (RAII on error;
    //    explicit cleanup on success — see `finalize_install`).
    let temp_install_root =
        tempfile::tempdir().context("Failed to create temp install root for self-update")?;
    let install_root = temp_install_root.path().to_path_buf();

    // 4. Run `cargo install <crate> --root <temp_install> --force`. No
    //    `--locked`: the published .crate ships a Cargo.lock, but a stale
    //    lockfile would hard-fail the install; plain `cargo install`
    //    re-resolves when the lock is stale. No `--target-dir`: cargo builds in
    //    its own temp target dir (`CARGO_TARGET_DIR` is stripped, so a hostile
    //    env cannot redirect it; a user `build.target-dir` config is a residual
    //    disk edge). cwd is the storage root (it is guaranteed writable and
    //    exists, unlike the launch dir).
    let crate_name = env!("CARGO_PKG_NAME");
    run_cargo_with_timeout(
        &[
            OsStr::new("install"),
            OsStr::new(crate_name),
            OsStr::new("--root"),
            install_root.as_os_str(),
            OsStr::new("--force"),
        ],
        &crate::config::CONFIG.global_storage_root(),
        Duration::from_hours(1),
        &format!("cargo install {crate_name} --force"),
        "Update",
    )
    .await?;

    // 5. Shared tail: swap, PATH freshness, drain, checkpoint, restart.
    let fresh_binary = temp_bin_path(&install_root);
    finalize_install(
        &fresh_binary,
        admin_target.as_deref(),
        "✅ Update installed from crates.io. Restarting…",
        vec![install_root],
    )
    .await
}

/// Path to the freshly built `mahbot` binary inside a cargo install temp root.
///
/// `cargo install --root <install_root>` places the produced binary at
/// `<install_root>/bin/mahbot` (`.exe` on Windows).
fn temp_bin_path(install_root: &Path) -> PathBuf {
    install_root
        .join("bin")
        .join(format!("mahbot{}", std::env::consts::EXE_SUFFIX))
}

/// Shared finalize tail for both update modes, after `cargo install` produced
/// the fresh binary at `<temp_install>/bin/mahbot`.
///
/// 1. Validate that the fresh binary exists and is non-empty (bail otherwise —
///    a silent empty swap would strand the daemon).
/// 2. Capture `current_exe()` BEFORE the swap — it is always the restart
///    target (self_replace rewrites it in place, wherever it lives).
/// 3. Swap the running binary with the fresh one via `self_replace`. The source
///    differs from the running exe (it lives in the temp root), which
///    self-replace requires on Windows.
/// 4. Notify `completion_msg` (mode-specific "build complete" / "installed").
/// 5. Refresh the PATH-visible cargo bin copy (sourced from the freshly-swapped
///    `current_exe`, so the manual-remediation source survives temp-root
///    cleanup), skipping the copy when already running from the cargo bin path
///    (`copy_to_cargo_bin`'s rename would otherwise fail on a Windows binary
///    locked in place).
/// 6. Notify "starting new instance" (Telegram channel must still be live).
/// 7. Hand off to [`finalize_update_and_restart`] for the drain → checkpoint →
///    unlock → spawn-from-`current_exe` → temp-root cleanup → exit.
///
/// `cleanup_paths` are the temp roots to remove after a successful spawn. On
/// any error return the caller keeps the `TempDir` values in scope, so their
/// RAII drops clean them up; `std::process::exit(0)` bypasses RAII, so the
/// success path removes them in the `after_spawn` closure instead.
async fn finalize_install(
    fresh_binary: &Path,
    admin_target: Option<&str>,
    completion_msg: &str,
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

    // 4. Notify: install/build complete (swap succeeded).
    notify_admin(completion_msg, admin_target).await;

    // 5. Refresh the PATH-visible cargo bin copy (non-fatal). The source is the
    //    freshly-swapped `current_exe` (not the temp root), so the manual
    //    remediation in `stale_binary_notification` points at a surviving path.
    refresh_cargo_bin(&current_exe, admin_target).await;

    // 6. Notify: starting new instance (MUST be before the shutdown in
    //    finalize_update_and_restart — the Telegram channel must still be live).
    notify_admin("🔄 Starting new instance…", admin_target).await;

    // 7. Shared finalize tail. The temp roots are removed here (after a
    //    successful spawn) rather than by RAII: `exit(0)` bypasses destructors,
    //    and they are never the spawn target (which is always `current_exe`),
    //    so removal cannot race macOS Gatekeeper validation of the child.
    finalize_update_and_restart(&current_exe, move || {
        for path in &cleanup_paths {
            if let Err(e) = fs::remove_dir_all(path) {
                warn!(
                    error = %e,
                    path = %path.display(),
                    "Could not remove temp update root after successful spawn"
                );
            }
        }
    })
    .await
}

/// Shared finalize tail for both update modes: graceful drain, final
/// single-writer checkpoint, instance-lock release, detached spawn of the
/// replacement, optional post-spawn cleanup, and `exit(0)`.
///
/// The ordering is load-bearing and MUST NOT be rearranged:
///
/// 1. Graceful drain (the FULL drain, same as window close — NOT fast-cancel).
///    In-flight agents complete their current round; the drain-watch task fires
///    the global token when no in-flight agents or orchestrator calls remain
///    (or force-cancels at the 10-min cap). The GUI stays open with input
///    disabled; the GUI exit path waits
///    ([`update_is_finalizing`]) instead of exiting, so this sequence cannot
///    be aborted by a window close or SIGINT racing the checkpoint. No failure
///    transitions with 'service shutting down' comments fire — agents that
///    cannot finish stay status='launched' and boot-resume.
/// 2. Checkpoint all databases BEFORE releasing the instance lock and spawning
///    the replacement. `exit(0)` below bypasses Rust destructors, so Turso
///    connections are never properly closed. With the lock still held this is
///    the last single-writer checkpoint: no checkpoint runs after the
///    replacement is live (the GUI exit path waits while the update is
///    finalizing — see `save_and_exit` / `update_is_finalizing`).
/// 3. Release the instance lock so the child process can acquire it on startup.
/// 4. Spawn the new instance. On macOS, posix_spawn triggers asynchronous
///    Gatekeeper code signature validation. Spawning before `after_spawn`
///    guarantees the spawn target is never deleted during or before the
///    child's startup window.
/// 5. Run `after_spawn` (cleanup of the temp update root(s)), then `exit(0)`.
///
/// `after_spawn` must not fail the update (best-effort cleanup) — a panic here
/// would still exit the process, so it must be infallible in practice.
async fn finalize_update_and_restart(spawn_path: &Path, after_spawn: impl FnOnce()) -> Result<()> {
    // 1. Begin the graceful drain.
    UPDATE_FINALIZING.store(true, Ordering::SeqCst);
    crate::shutdown::drain_begin();
    while !crate::shutdown::shutdown_token().is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    crate::tools::browser::close_all_browser_sessions().await;

    // 2. Checkpoint all databases BEFORE releasing the instance lock and
    //    spawning the replacement (see doc comment above).
    crate::db::checkpoint::checkpoint_all_databases().await;

    // 3. Release instance lock so the child process can acquire it on startup.
    release_instance_lock().await;

    // 4. Spawn the new instance from the determined spawn path.
    if let Err(e) = spawn_new_instance_from(spawn_path) {
        // Spawn failed — the process stays alive (unless a genuine window
        // close was requested during the finalizing window, in which case the
        // GUI honors it with its own checkpoint + exit via UpdateResult).
        // Clear the finalizing flag and re-acquire the lock.
        UPDATE_FINALIZING.store(false, Ordering::SeqCst);
        // Re-acquire the lock since the process stays alive.
        if let Err(lock_err) = reacquire_instance_lock().await {
            error!(%lock_err, "Failed to re-acquire instance lock after spawn failure");
        }
        return Err(e);
    }

    // 5. Post-spawn cleanup (remove the temp update roots).
    after_spawn();

    // Exit — spawn succeeded.
    std::process::exit(0);
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Run a long-running cargo subcommand with a timeout. Shared by the
/// local-build and registry-install update paths (the only differences are the
/// arguments, working directory, timeout, and user-facing labels).
///
/// `args` are the cargo arguments (excluding the `cargo` binary itself, each
/// as `&OsStr` so paths and flags pass through without Unicode assumption),
/// `cwd` the working directory, `timeout` the hard deadline, `label` a
/// short human-readable description used in logs and error messages (e.g.
/// "cargo install --path --locked"), and `failure_kind` the noun used in
/// failure messages ("Build" / "Update").
///
/// `CARGO_TARGET_DIR` is stripped so `cargo install` (registry mode, which
/// passes no `--target-dir`) never redirects its build into an uncleaned tree;
/// local mode passes `--target-dir` explicitly, which takes precedence, so this
/// is purely defensive. `CARGO_HOME` is NOT stripped — the child needs it for
/// the toolchain/registry cache.
///
/// On failure the error is returned (no admin notification — the caller owns
/// failure reporting via the single [`handle_update_command`]/GUI path).
async fn run_cargo_with_timeout(
    args: &[&OsStr],
    cwd: &Path,
    timeout: Duration,
    label: &str,
    failure_kind: &str,
) -> Result<()> {
    info!("Starting {label} in {}", cwd.display());
    let cargo_result = tokio::time::timeout(
        timeout,
        tokio::process::Command::new("cargo")
            .args(args)
            .current_dir(cwd)
            // Strip any inherited CARGO_TARGET_DIR so registry-mode cargo
            // install (no --target-dir) never redirects its build into an
            // uncleaned tree.
            .env_remove("CARGO_TARGET_DIR")
            // kill_on_drop: if the timeout fires, the cargo child must die
            // too rather than keep compiling in the background. For
            // `cargo install` an orphaned child could even complete and swap
            // the binary after the daemon reported "timed out", racing a
            // user retry.
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await;

    match cargo_result {
        Err(_elapsed) => {
            anyhow::bail!(
                "{failure_kind} failed: {label} timed out after {} minutes",
                timeout.as_secs() / 60
            );
        }
        Ok(Err(e)) => {
            anyhow::bail!("{failure_kind} failed: could not start cargo: {e}");
        }
        Ok(Ok(output)) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = format!("stdout:\n{stdout}\nstderr:\n{stderr}");
            let truncated = truncate_to_last_64k(&combined);
            anyhow::bail!("{failure_kind} failed:\n```\n{truncated}\n```");
        }
        Ok(Ok(_)) => {
            info!("{label} completed successfully");
            Ok(())
        }
    }
}

/// Look up an admin user's Telegram reply target.
///
/// Returns `Some(reply_target)` if an admin user (permissions == "full") has
/// a Telegram channel binding with a non-null `reply_target` and a bot token
/// is configured. Returns `None` otherwise. With multiple admins, the first
/// bound one wins.
pub async fn resolve_admin_telegram_target() -> Option<String> {
    let _ = crate::config::CONFIG.telegram_bot_token()?;

    let store = crate::users::store();
    let admin = store.find_admin().await.ok()??;
    let bindings = store.get_user_channels(&admin.name).await.ok()?;
    bindings
        .into_iter()
        .find(|b| b.channel == "telegram" && b.reply_target.is_some())
        .and_then(|b| b.reply_target)
}

/// Send a notification to the admin user via Telegram.
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

/// Reply used when a command requires full (admin) permissions. Used by both
/// the Telegram command dispatch (binary) and the `/update` handler (library)
/// so the denial wording stays consistent.
pub const ADMIN_ONLY_CMD_MSG: &str = "This command is only available to admin users.";

/// Handle a Telegram `/update` command.
///
/// Gated on the shared availability cache (admin + update available), with a
/// synchronous reply for the early-failure cases (not an admin / already in
/// progress / no update / cargo not on PATH). The actual update runs as a
/// spawned async task so it does not block the Telegram message dispatch loop.
/// Progress notifications route via the normal update notification path (first
/// bound admin); a failure is also reported directly to the invoking admin.
/// There is NO confirmation modal — an admin invoking `/update` from Telegram
/// is itself sufficient confirmation.
pub async fn handle_update_command(msg: &ChannelMessage) {
    if !crate::users::is_admin(&msg.user_name).await {
        crate::channels::telegram::send_reply(&msg.reply_target, ADMIN_ONLY_CMD_MSG).await;
        return;
    }

    // Fast path for an already-running update, then the atomic claim below.
    if update_availability().in_progress {
        crate::channels::telegram::send_reply(
            &msg.reply_target,
            "An update is already in progress. Please wait for it to complete.",
        )
        .await;
        return;
    }

    if !should_show_update(update_availability(), true) {
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
    // immediate answer instead of a silent no-op.
    if let Err(e) = verify_cargo_on_path("perform the update").await {
        crate::channels::telegram::send_reply(
            &msg.reply_target,
            &format!("Cannot start the update: {e}"),
        )
        .await;
        return;
    }

    // Atomically claim the in-progress flag before spawning. This closes the
    // TOCTOU where a concurrent `/update` could pass the pre-check above, then
    // lose `UPDATE_MUTEX.try_lock` inside `execute_update` and be misreported
    // as "Update failed: An update is already in progress." `execute_update`
    // keeps the flag set and clears it on failure.
    if update_cache()
        .in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        crate::channels::telegram::send_reply(
            &msg.reply_target,
            "An update is already in progress. Please wait for it to complete.",
        )
        .await;
        return;
    }

    crate::channels::telegram::send_reply(
        &msg.reply_target,
        "✅ Update triggered — it will build/install in the background and restart the daemon when complete.",
    )
    .await;

    // Fire-and-forget: the build/install (10–60 min) must not block the
    // dispatch loop. Progress notifications route via the normal update
    // notification path (first bound admin); on failure the invoking admin is
    // also told directly so a non-primary invoker isn't left guessing.
    let invoker_target = msg.reply_target.clone();
    tokio::spawn(async move {
        if let Err(e) = execute_update().await {
            // Single failure report: the first bound admin (the normal update
            // notification path), plus the invoking admin when they differ.
            let failure = format!("❌ Update failed:\n{e:#}");
            let admin_target = resolve_update_admin_target().await;
            notify_admin(&failure, admin_target.as_deref()).await;
            if admin_target.as_deref() != Some(invoker_target.as_str()) {
                crate::channels::telegram::send_reply(&invoker_target, &failure).await;
            }
        }
    });
}

// ── Cargo bin path resolution and installation ───────────────────────────

/// Resolve the path to the `mahbot` binary in the cargo bin directory.
///
/// Delegates to [`crate::util::cargo_bin_dir`] for directory resolution,
/// then appends the platform-specific executable name.
fn resolve_cargo_bin_path() -> Option<PathBuf> {
    let exe_name = format!("mahbot{}", std::env::consts::EXE_SUFFIX);
    Some(crate::util::cargo_bin_dir()?.join(exe_name))
}

/// Copy the freshly built binary to the PATH-visible cargo bin path, so a
/// `mahbot` invoked from PATH runs the new version even when the daemon started
/// from a different path (e.g. the repo's `target/release`).
///
/// The source is the freshly-swapped `current_exe` (the running binary after
/// `self_replace`), which persists across the temp-root cleanup — so the manual
/// remediation in [`copy_to_cargo_bin`] points at a surviving path.
///
/// Skipped when already running from the cargo bin path (self_replace already
/// updated it in place) — copying onto it would fail the rename on a Windows
/// binary locked in place. Non-fatal: the running binary is already updated via
/// self_replace; failures are logged and reported to the admin inside
/// [`copy_to_cargo_bin`].
async fn refresh_cargo_bin(current_exe: &Path, admin_target: Option<&str>) {
    let Some(cargo_bin) = resolve_cargo_bin_path() else {
        warn!("No cargo bin path resolved — PATH-visible binary not refreshed");
        return;
    };
    if canonicalize_safe(current_exe) == canonicalize_safe(&cargo_bin) {
        info!(
            "Already running from cargo bin path `{}` — skipping install copy",
            cargo_bin.display()
        );
        return;
    }
    copy_to_cargo_bin(current_exe, &cargo_bin, admin_target).await;
}

/// Format an admin-facing notification for a copy-to-cargo-bin failure.
///
/// The message tells the admin that the PATH-visible binary is stale and
/// provides manual remediation steps.
fn stale_binary_notification(reason: &str, source: &Path, dest: &Path) -> String {
    format!(
        "⚠️ {reason}. \
         The running binary is updated, but the PATH-visible binary \
         remains stale. Manually copy `{}` to `{}`.",
        source.display(),
        dest.display(),
    )
}

/// Copy the newly built binary to the cargo install bin path.
///
/// Uses a temp-file + rename pattern for crash safety: writes to a
/// `.mahbot_update_tmp` sibling first, then atomically renames. If the process
/// crashes mid-copy, the install path retains its old (stale but valid) binary.
///
/// This function is intentionally non-fatal — the running process is already
/// updated via `self_replace`, so the caller doesn't need a result. On failure
/// it logs a warning and attempts admin notification.
async fn copy_to_cargo_bin(source: &Path, dest: &Path, admin_target: Option<&str>) {
    // Create parent directory if it doesn't exist.
    if let Some(parent) = dest.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        warn!(
            error = %e,
            path = %parent.display(),
            "Failed to create cargo bin directory"
        );
        notify_admin(
            &stale_binary_notification(
                &format!(
                    "Could not create cargo bin directory `{}`",
                    parent.display()
                ),
                source,
                dest,
            ),
            admin_target,
        )
        .await;
        return;
    }

    // Write to a temp file first, then atomically rename to the target.
    // This prevents a partial/corrupt binary at the install path if the
    // process crashes during the copy.
    let tmp_path = dest.with_extension("mahbot_update_tmp");
    let _ = fs::remove_file(&tmp_path); // Clean up any leftover from a previous crash.

    if let Err(e) = fs::copy(source, &tmp_path) {
        warn!(
            error = %e,
            path = %dest.display(),
            "Failed to copy binary to cargo bin temp path"
        );
        let _ = fs::remove_file(&tmp_path);
        notify_admin(
            &stale_binary_notification(
                &format!("Could not install updated binary to `{}`", dest.display()),
                source,
                dest,
            ),
            admin_target,
        )
        .await;
        return;
    }

    // Atomically replace the target with the temp file.
    if let Err(e) = fs::rename(&tmp_path, dest) {
        warn!(
            error = %e,
            path = %dest.display(),
            source = %tmp_path.display(),
            "Failed to rename temp binary to final path"
        );
        let _ = fs::remove_file(&tmp_path);
        notify_admin(
            &format!(
                "⚠️ Could not install updated binary to `{}`: rename failed: {e}. \
                 The temp file is at `{}`. Manually rename it to complete installation.",
                dest.display(),
                tmp_path.display(),
            ),
            admin_target,
        )
        .await;
        return;
    }

    info!(path = %dest.display(), "Installed new binary to cargo bin path");
}

/// Canonicalize a path, falling back to the lexical path on failure.
///
/// Used for canonicalized-path comparisons where the file may not exist yet
/// (e.g., the cargo bin install path before installation) or where
/// canonicalization may fail for other reasons (e.g., broken symlinks,
/// permission denied).
fn canonicalize_safe(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Spawn the new mahbot instance as a detached child process from the given path.
///
/// The `binary_path` must point to an existing, executable binary — always the
/// captured `current_exe()` after the swap in the unified update flow.
///
/// On Unix: null stdin/stdout, stderr → update.log. On Windows: same + `DETACHED_PROCESS | CREATE_NO_WINDOW`.
/// On spawn failure the error is returned (no admin notification — the caller
/// [`finalize_update_and_restart`] owns failure reporting); the process keeps
/// running (does NOT exit).
///
/// ## macOS Gatekeeper safety
///
/// The caller guarantees that `binary_path` is never deleted before or during
/// the child's startup window (see the temp-root cleanup rationale in
/// [`finalize_install`]). Deleting the spawn target while Gatekeeper is
/// validating its code signature causes `syspolicyd` to SIGKILL the child.
fn spawn_new_instance_from(binary_path: &Path) -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();

    info!(
        program = %binary_path.display(),
        args = ?args,
        "Spawning new mahbot instance"
    );

    let mut cmd = std::process::Command::new(binary_path);
    cmd.args(&args);

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
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW);
    }

    cmd.stderr(Stdio::from(update_log));

    match cmd.spawn() {
        Ok(child) => {
            info!(pid = child.id(), "Spawned new mahbot instance");
            // Detach — the child runs independently.
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
    use crate::util::test::make_executable;

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
    fn test_lock_acquire_and_release_with_temp_dir() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("mahbot.lock");

        let file1 = open_lock_file(&lock_path).unwrap();

        // First lock should succeed.
        assert!(try_flock(&file1).unwrap(), "First flock should succeed");

        // Second lock on a different fd on the same file should fail.
        let file2 = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert!(
            !try_flock(&file2).unwrap(),
            "Second flock should fail (already locked)"
        );

        drop(file1);
        // After dropping file1, the lock should be released.
        assert!(
            try_flock(&file2).unwrap(),
            "After release, flock should succeed"
        );
    }

    #[test]
    fn test_try_acquire_lock_held_free() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("mahbot.lock");

        // Hold the lock on this file so try_acquire_lock will fail.
        let holder = open_lock_file(&lock_path).unwrap();
        assert!(try_flock(&holder).unwrap(), "First flock should succeed");

        // While the lock is held, try_acquire_lock should return None.
        assert!(
            try_acquire_lock(&lock_path).unwrap().is_none(),
            "Should return None when lock is held"
        );

        // After releasing the holder, try_acquire_lock should succeed.
        drop(holder);
        let result = try_acquire_lock(&lock_path).unwrap();
        assert!(result.is_some(), "After release, lock should be acquirable");
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

        // Local checkout: `.git` present → LocalCheckout even with Cargo.toml.
        let checkout = dir.path().join("checkout");
        std::fs::create_dir_all(checkout.join(".git")).unwrap();
        std::fs::write(checkout.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&checkout), UpdateMode::LocalCheckout);

        // Registry src cache: `registry/src` segment → Registry even with
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
        assert_eq!(classify_update_mode(&registry), UpdateMode::Registry);

        // Git checkouts cache: `git/checkouts` segment → Registry. Cargo
        // git checkouts are full non-bare clones and contain a real `.git`
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
        assert_eq!(classify_update_mode(&git_checkout), UpdateMode::Registry);

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
        assert_eq!(classify_update_mode(&custom), UpdateMode::Registry);

        // Plain source tree (no .git, no cargo cache): Cargo.toml → LocalCheckout.
        let plain = dir.path().join("plain-src");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("Cargo.toml"), "").unwrap();
        assert_eq!(classify_update_mode(&plain), UpdateMode::LocalCheckout);

        // No source at all → Registry (the only viable update path).
        let bare = dir.path().join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(classify_update_mode(&bare), UpdateMode::Registry);
    }

    #[test]
    fn test_sparse_index_path() {
        assert_eq!(sparse_index_path("a"), "1/a");
        assert_eq!(sparse_index_path("ab"), "2/ab");
        assert_eq!(sparse_index_path("abc"), "3/a/abc");
        assert_eq!(sparse_index_path("mahbot"), "ma/hb/mahbot");
        assert_eq!(sparse_index_path("serde"), "se/rd/serde");
    }

    #[test]
    fn test_latest_stable_version_filters_yanked_and_prerelease() {
        // NDJSON sparse-index fixture lines (subset of real fields).
        let body = "\
{\"name\":\"mahbot\",\"vers\":\"0.2.0\",\"yanked\":false}
{\"name\":\"mahbot\",\"vers\":\"0.3.0\",\"yanked\":false}
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":true}
{\"name\":\"mahbot\",\"vers\":\"0.3.1-beta.1\",\"yanked\":false}
{\"name\":\"mahbot\",\"vers\":\"0.4.0-rc.1\",\"yanked\":false}
";
        let latest = latest_stable_version(body).expect("a stable non-yanked version exists");
        // 0.3.0 wins: 0.4.0 is yanked, 0.3.1-beta.1 / 0.4.0-rc.1 are prereleases.
        assert_eq!(latest.to_string(), "0.3.0");
    }

    #[test]
    fn test_latest_stable_version_empty_and_malformed() {
        assert_eq!(latest_stable_version(""), None);
        assert_eq!(latest_stable_version("not json\n"), None);
        assert_eq!(
            latest_stable_version("{\"vers\":\"1.0.0\"}\n"),
            Some(semver::Version::new(1, 0, 0))
        );
        // All yanked → None.
        assert_eq!(
            latest_stable_version("{\"vers\":\"1.0.0\",\"yanked\":true}\n"),
            None
        );
    }

    #[test]
    fn test_latest_stable_version_semver_ordering() {
        // Proper semantic ordering, not lexicographic: 0.10.0 > 0.9.0.
        let body = "\
{\"name\":\"mahbot\",\"vers\":\"0.9.0\",\"yanked\":false}
{\"name\":\"mahbot\",\"vers\":\"0.10.0\",\"yanked\":false}
";
        assert_eq!(
            latest_stable_version(body).map(|v| v.to_string()),
            Some("0.10.0".to_string())
        );
    }

    #[test]
    fn test_latest_stable_version_last_line_wins_for_yank_state() {
        // The sparse index appends a line per publish/yank/unyank event and a
        // version's LAST line is its current state. A yank → unyank → re-yank
        // cycle must count as yanked even though an earlier line said
        // unyanked (cargo install would refuse the re-yanked version).
        let re_yanked = "\
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":false}
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":true}
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":false}
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":true}
{\"name\":\"mahbot\",\"vers\":\"0.3.0\",\"yanked\":false}
";
        assert_eq!(
            latest_stable_version(re_yanked).map(|v| v.to_string()),
            Some("0.3.0".to_string())
        );

        // Unyank wins when the last line says unyanked.
        let unyanked = "\
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":true}
{\"name\":\"mahbot\",\"vers\":\"0.4.0\",\"yanked\":false}
";
        assert_eq!(
            latest_stable_version(unyanked).map(|v| v.to_string()),
            Some("0.4.0".to_string())
        );
    }

    // ── New function tests ─────────────────────────────────────────────────

    use crate::util::test::set_env_var;

    #[test]
    fn test_resolve_cargo_bin_path_cargo_home() {
        // Scenario 1: CARGO_HOME is set to a custom path.
        let path_with = {
            let _guard = set_env_var("CARGO_HOME", Some("/custom/cargo"));
            resolve_cargo_bin_path()
        };

        // Scenario 2: CARGO_HOME is set to empty string (falls through to
        // UserDirs — see cargo_bin_dir() in src/util/mod.rs).
        let path_empty = {
            let _guard = set_env_var("CARGO_HOME", Some(""));
            resolve_cargo_bin_path()
        };
        // Both guards have dropped, restoring CARGO_HOME to its original
        // state (typically absent).

        // With custom CARGO_HOME, should use that path.
        assert!(
            path_with.is_some(),
            "resolve_cargo_bin_path should return Some with CARGO_HOME set"
        );
        let path = path_with.unwrap();
        assert!(
            path.starts_with("/custom/cargo/bin/mahbot"),
            "Expected path to start with /custom/cargo/bin/mahbot, got {}",
            path.display(),
        );
        let file_name = path.file_name().unwrap().to_string_lossy();
        assert!(
            file_name.starts_with("mahbot"),
            "Expected file name to start with 'mahbot', got '{file_name}'"
        );

        // With empty CARGO_HOME, should fall through to UserDirs.
        let dirs = UserDirs::new();
        if let Some(dirs) = dirs {
            assert!(
                path_empty.is_some(),
                "Expected a path when CARGO_HOME is empty"
            );
            let path = path_empty.unwrap();
            let expected_prefix = dirs.home_dir().join(".cargo").join("bin");
            assert!(
                path.starts_with(&expected_prefix),
                "Expected path to start with {}, got {}",
                expected_prefix.display(),
                path.display(),
            );
        }
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

    #[tokio::test]
    async fn test_copy_to_cargo_bin_success() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source_bin");
        let dest = dir.path().join("subdir").join("installed_bin");

        // Create a source binary.
        std::fs::write(&source, "binary content").unwrap();
        make_executable(&source);

        // Copy should succeed (non-fatal, returns nothing).
        copy_to_cargo_bin(&source, &dest, None).await;

        // Verify destination exists and has correct content.
        assert!(dest.is_file(), "Destination should exist");
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "binary content");

        // Verify temp file was cleaned up.
        let tmp_path = dest.with_extension("mahbot_update_tmp");
        assert!(!tmp_path.exists(), "Temp file should be cleaned up");
    }

    #[tokio::test]
    async fn test_copy_to_cargo_bin_source_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("nonexistent_source");
        let dest = dir.path().join("dest_bin");

        // Copy should fail gracefully (non-fatal, returns nothing).
        copy_to_cargo_bin(&source, &dest, None).await;
        assert!(!dest.exists(), "Destination should not be created");
    }

    #[tokio::test]
    async fn test_copy_to_cargo_bin_creates_parent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source_bin");
        let dest = dir.path().join("deep").join("nested").join("installed_bin");

        std::fs::write(&source, "content").unwrap();

        // Copy should create parent directories (non-fatal, returns nothing).
        copy_to_cargo_bin(&source, &dest, None).await;
        assert!(dest.is_file(), "Destination should exist");
        assert!(
            dest.parent().unwrap().is_dir(),
            "Parent directory should exist"
        );
    }

    #[test]
    fn test_stale_binary_notification_format() {
        let msg = stale_binary_notification(
            "Test error",
            Path::new("/src/mahbot"),
            Path::new("/dest/mahbot"),
        );
        assert!(msg.contains("⚠️ Test error"));
        assert!(msg.contains("Manually copy"));
        assert!(msg.contains("/src/mahbot"));
        assert!(msg.contains("/dest/mahbot"));
        assert!(msg.contains("PATH-visible binary remains stale"));
    }
}
